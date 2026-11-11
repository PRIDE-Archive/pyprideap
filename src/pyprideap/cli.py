"""Command-line interface for pyprideap.

Usage:
    pyprideap report <file>                        Generate QC report from a local NPX/ADAT file
    pyprideap report -a PAD000001                  Download data from PRIDE and generate report
    pyprideap report <file> -o out.html            Specify output path
    pyprideap report <file> -p olink               Force platform type
    pyprideap report <file> --split                Output individual plots as separate HTML files
    pyprideap proteins-above-lod <file>            List UniProt accessions above LOD
    pyprideap proteins-above-lod -a PAD000001      Download data from PRIDE and list proteins
    pyprideap proteins-above-lod <file> -t 80      Custom threshold (default 50%)
    pyprideap unique-samples <file>                List unique sample identifiers
    pyprideap unique-samples -a PAD000001          Download data from PRIDE and list samples
    pyprideap unique-samples <file> -o samples.txt Save to file

Examples:
    pyprideap report data.npx.csv
    pyprideap report data.parquet -o my_report.html
    pyprideap report data.npx.csv --split -o plots_dir/
    pyprideap report -a PAD000001
    pyprideap report ambiguous.csv -p somascan
    pyprideap report data.npx.csv -v
    pyprideap proteins-above-lod data.npx.csv
    pyprideap proteins-above-lod -a PAD000001
    pyprideap proteins-above-lod data.npx.csv -t 80 -o proteins.txt
    pyprideap unique-samples data.npx.csv
    pyprideap unique-samples -a PAD000001
    pyprideap unique-samples data.npx.csv -o samples.txt
    pyprideap unique-samples data.npx.csv --exclude-controls
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from pyprideap.io.readers.registry import read as _read
from pyprideap.processing.filtering import get_unique_samples as _get_unique_samples
from pyprideap.processing.lod import get_proteins_above_lod as _get_proteins_above_lod
from pyprideap.viz.qc.report import qc_report as _qc_report

logger = logging.getLogger("pyprideap")


def _setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logger.setLevel(level)


def _is_sdrf_filename(name: str) -> bool:
    lower = name.lower()
    return "sdrf" in lower and lower.endswith((".tsv", ".txt"))


_PAD_ACCESSION_RE = re.compile(r"PAD\d{6}", re.IGNORECASE)


def _discover_sdrf(data_path: Path) -> Path | None:
    """Find the SDRF of the same PAD accession next to the data file.

    Only files whose name carries the data file's accession are considered, so
    a folder holding several datasets never pairs data with another dataset's
    SDRF. Community-annotated files are preferred.
    """
    accession = _PAD_ACCESSION_RE.search(data_path.name)
    if accession is None or not data_path.parent.exists():
        return None
    acc = accession.group(0).upper()
    matches: list[Path] = []
    for pattern in ("*sdrf*.tsv", "*sdrf*.txt"):
        matches.extend(p for p in data_path.parent.glob(pattern) if acc in p.name.upper())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        return None
    unique.sort(key=lambda p: (0 if "community" in p.name.lower() else 1, p.name.lower()))
    return unique[0]


_PRIDE_FTP = "ftp://ftp.pride.ebi.ac.uk/"
_PRIDE_HTTPS = "https://ftp.pride.ebi.ac.uk/"
_ARCHIVE_PREFIX = "pride/data/archive/"
# EBI FIRE object store (S3-compatible, public bucket, unsigned reads), as in pridepy
_FIRE_URL = "https://hh.fire.sdo.ebi.ac.uk"
_FIRE_BUCKET = "pride-public"
_DOWNLOAD_TIMEOUT = 300
_CHUNK = 1 << 20


def _download_workers(n_files: int) -> int:
    """Parallel downloads; ``PYPRIDEAP_DOWNLOAD_WORKERS`` overrides the default of 8."""
    try:
        workers = int(os.environ.get("PYPRIDEAP_DOWNLOAD_WORKERS", "8"))
    except ValueError:
        workers = 8
    return max(1, min(workers, n_files or 1))


def _candidate_urls(url: str) -> list[str]:
    """URLs to try, in order, for a PRIDE archive file.

    ``PYPRIDEAP_DOWNLOAD_PROTOCOL`` picks the first one (default ``fire``):

    - ``fire``: EBI FIRE object store (``PYPRIDEAP_FIRE_URL``, default
      https://hh.fire.sdo.ebi.ac.uk). Fastest, especially inside EBI.
    - ``ftp``: ftp://ftp.pride.ebi.ac.uk (often blocked in containers).
    - ``https``: the https://ftp.pride.ebi.ac.uk gateway (~10 s to first byte per file).

    The HTTPS gateway is always the last fallback.
    """
    path = None
    for base in (_PRIDE_FTP, _PRIDE_HTTPS):
        if url.startswith(base):
            path = url[len(base) :]
    if path is None:
        return [url]
    https = _PRIDE_HTTPS + path
    protocol = os.environ.get("PYPRIDEAP_DOWNLOAD_PROTOCOL", "fire").lower()
    first = None
    if protocol == "fire" and path.startswith(_ARCHIVE_PREFIX):
        fire = os.environ.get("PYPRIDEAP_FIRE_URL", _FIRE_URL).rstrip("/")
        first = f"{fire}/{_FIRE_BUCKET}/{path[len(_ARCHIVE_PREFIX) :]}"
    elif protocol == "ftp":
        first = _PRIDE_FTP + path
    return [first, https] if first else [https]


def _fetch(url: str, dest: Path) -> Path:
    """Download one file to *dest* in 1 MiB chunks, falling back across mirrors."""
    import shutil
    import urllib.request

    candidates = _candidate_urls(url)
    for i, candidate in enumerate(candidates):
        logger.debug("URL: %s -> %s", candidate, dest)
        try:
            with urllib.request.urlopen(candidate, timeout=_DOWNLOAD_TIMEOUT) as resp, dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh, _CHUNK)
            return dest
        except OSError as exc:
            if i == len(candidates) - 1:
                raise
            logger.debug("Download from %s failed (%s); trying %s", candidate, exc, candidates[i + 1])
    return dest


# Files the submitter typed as RESULT or RAW hold the measurements; OTHER and
# SEARCH files next to them are usually the same data in another layout, sample
# sheets or analysis outputs, and would add duplicate or failing reports.
_QC_FILE_CATEGORIES = ("RESULT", "RAW")


def _select_data_files(files: dict[str, tuple[str, str | None]]) -> dict[str, str]:
    """Data files to QC: those typed RESULT or RAW, or all of them when none is."""
    typed = {name: url for name, (url, category) in files.items() if category in _QC_FILE_CATEGORIES}
    if typed:
        skipped = sorted(set(files) - set(typed))
        if skipped:
            logger.debug("Skipping data files not typed RESULT/RAW: %s", skipped)
        return typed
    return {name: url for name, (url, _category) in files.items()}


def _download_pad_files(accession: str, dest_dir: Path) -> tuple[list[Path], Path | None]:
    """Download NPX/ADAT files (and SDRF if present) from PRIDE for a PAD accession."""
    from pyprideap.api.pride import PrideClient

    client = PrideClient()
    logger.debug("Fetching file list for %s from PRIDE API...", accession)
    files = client.get_download_files(accession)
    urls = {name: url for name, (url, _category) in files.items()}

    # Filter for supported data files
    data_extensions = {".csv", ".parquet", ".xlsx", ".adat"}
    sdrf_urls = {name: url for name, url in urls.items() if _is_sdrf_filename(name) and "checksum" not in name.lower()}
    data_urls = _select_data_files(
        {
            name: entry
            for name, entry in files.items()
            if any(name.lower().endswith(ext) for ext in data_extensions)
            and "checksum" not in name.lower()
            and name not in sdrf_urls
        }
    )
    # Instrument XMLs are not QC'd themselves but annotate data files next to them
    # (NULISAseq: UniProt IDs and plate QC for the NPQ exports)
    aux_urls = {
        name: url for name, url in urls.items() if name.lower().endswith(".xml") and "checksum" not in name.lower()
    }

    if not data_urls:
        click.echo(f"Error: No supported data files found for {accession}", err=True)
        click.echo(f"Available files: {list(urls.keys())}", err=True)
        sys.exit(1)

    logger.debug("Found %d data file(s): %s", len(data_urls), list(data_urls.keys()))
    if sdrf_urls:
        logger.debug("Found %d SDRF file(s): %s", len(sdrf_urls), list(sdrf_urls.keys()))

    jobs = [(name, url, dest_dir / Path(name).name) for name, url in {**data_urls, **sdrf_urls, **aux_urls}.items()]
    for name, _url, _dest in jobs:
        click.echo(f"Downloading {name}...", err=True)

    # The PRIDE HTTPS gateway takes ~10 s before the first byte of every file, so
    # downloading one file after another dominates the run time. Fetch in parallel.
    failed: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=_download_workers(len(jobs))) as pool:
        futures = {pool.submit(_fetch, url, dest): name for name, url, dest in jobs}
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                failed[futures[future]] = exc

    downloaded: list[Path] = []
    sdrf_files: list[Path] = []
    for name, _url, dest in jobs:
        if name in failed:
            if name in aux_urls:
                # Annotation files only: a failed download must not stop the report
                logger.warning("Could not download auxiliary file %s: %s", name, failed[name])
                continue
            raise failed[name]
        if name in aux_urls:
            continue
        if _is_sdrf_filename(name):
            sdrf_files.append(dest)
        else:
            downloaded.append(dest)

    sdrf_path = None
    if sdrf_files:
        sdrf_files.sort(key=lambda p: (0 if "community" in p.name.lower() else 1, p.name.lower()))
        sdrf_path = sdrf_files[0]
    return downloaded, sdrf_path


def _generate_report(
    input_path: Path,
    output_path: Path | None,
    platform: str | None = None,
    split: bool = False,
    sdrf_path: Path | None = None,
    no_border: bool = True,
) -> Path:
    """Read a data file and generate a QC report."""
    click.echo(f"Reading {input_path.name}...")
    logger.debug("Full path: %s", input_path)
    ds = _read(input_path, platform=platform)
    click.echo(f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})")
    logger.debug("Samples columns: %s", list(ds.samples.columns))
    logger.debug("Features columns: %s", list(ds.features.columns))

    if sdrf_path is None:
        nearby = _discover_sdrf(input_path)
        if nearby is not None:
            sdrf_path = nearby
            click.echo(f"  SDRF (auto): {sdrf_path.name}")
    else:
        click.echo(f"  SDRF: {sdrf_path.name}")
        logger.debug("SDRF full path: %s", sdrf_path)

    if split:
        from pyprideap.viz.qc.report import qc_report_split

        if output_path is None:
            stem = input_path.stem
            if stem.endswith(".npx") or stem.endswith(".ct"):
                stem = Path(stem).stem
            output_path = Path(f"{stem}_qc_plots")

        click.echo("Generating individual plot files...")
        logger.debug("Output directory: %s", output_path)
        result = qc_report_split(ds, output_path, no_border=no_border, sdrf_path=sdrf_path)
        n_files = len(list(result.glob("*.html")))
        click.echo(f"  {n_files} HTML files saved to {result}/")
        return result

    if output_path is None:
        stem = input_path.stem
        # Handle double extensions like .npx.csv
        if stem.endswith(".npx") or stem.endswith(".ct"):
            stem = Path(stem).stem
        output_path = Path(f"{stem}_qc_report.html")

    click.echo("Generating report...")
    logger.debug("Output path: %s", output_path)
    result = _qc_report(ds, output_path, sdrf_path=sdrf_path)
    click.echo(f"  Report saved to {result}")
    return result


@click.group()
@click.version_option(package_name="pyprideap")
def main() -> None:
    """pyprideap - PRIDE Affinity Proteomics tools."""


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option("-a", "--accession", default=None, help="PAD accession to download from PRIDE (e.g. PAD000001).")
@click.option("-o", "--output", default=None, help="Output HTML file or directory path.")
@click.option(
    "-p",
    "--platform",
    type=click.Choice(["olink", "somascan", "nulisa"], case_sensitive=False),
    default=None,
    help="Force platform type (default: auto-detect).",
)
@click.option(
    "--split", is_flag=True, default=False, help="Output individual plot HTML files instead of a single report."
)
@click.option(
    "--sdrf",
    default=None,
    type=click.Path(exists=True),
    help="Path to SDRF TSV (case/control groups for MPI and volcano plots). "
    "If omitted, pyprideap looks for *sdrf*.tsv next to the data file.",
)
@click.option(
    "--no-border/--border", default=True, help="Remove card borders from split plot files (default: no border)."
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging output.")
def report(
    input_file: str | None,
    accession: str | None,
    output: str | None,
    platform: str | None,
    split: bool,
    sdrf: str | None,
    no_border: bool,
    verbose: bool,
) -> None:
    """Generate a QC report from a data file or PAD accession."""
    _setup_logging(verbose)

    if input_file is None and accession is None:
        click.echo("Error: Provide either an input file or --accession / -a.", err=True)
        sys.exit(1)

    if input_file is not None and accession is not None:
        click.echo("Error: Provide either an input file or --accession, not both.", err=True)
        sys.exit(1)

    output_path = Path(output) if output else Path(".")
    sdrf_path = Path(sdrf) if sdrf else None

    if accession is not None:
        accession = accession.upper()
        click.echo(f"Fetching data from PRIDE for {accession}...")

        with tempfile.TemporaryDirectory(prefix="pyprideap_") as tmpdir:
            tmppath = Path(tmpdir)
            files, downloaded_sdrf = _download_pad_files(accession, tmppath)
            if sdrf_path is None and downloaded_sdrf is not None:
                sdrf_path = downloaded_sdrf
                click.echo(f"  SDRF: {sdrf_path.name}")

            for f in files:
                try:
                    stem = f.stem
                    if stem.endswith(".npx") or stem.endswith(".ct"):
                        stem = Path(stem).stem
                    if split:
                        out = Path(f"{output_path}/{stem}")
                    else:
                        out = Path(f"{output_path}/{stem}.html")
                    _generate_report(
                        f,
                        out,
                        platform=platform,
                        split=split,
                        sdrf_path=sdrf_path,
                        no_border=no_border,
                    )
                except Exception as e:
                    logger.debug("Error processing %s: %s", f.name, e, exc_info=True)
                    click.echo(f"  Skipping {f.name}: {e}", err=True)
    else:
        input_path = Path(input_file)  # type: ignore[arg-type]
        if not input_path.exists():
            click.echo(f"Error: File not found: {input_path}", err=True)
            sys.exit(1)
        _generate_report(
            input_path,
            output_path,
            platform=platform,
            split=split,
            sdrf_path=sdrf_path,
            no_border=no_border,
        )


@main.command("proteins-above-lod")
@click.argument("input_file", required=False, default=None)
@click.option("-a", "--accession", default=None, help="PAD accession to download from PRIDE (e.g. PAD000001).")
@click.option("-o", "--output", default=None, help="Output file path (default: print to stdout).")
@click.option(
    "-p",
    "--platform",
    type=click.Choice(["olink", "somascan", "nulisa"], case_sensitive=False),
    default=None,
    help="Force platform type (default: auto-detect).",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=50.0,
    help="Min %% of samples above LOD to include a protein (default: 50).",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging output.")
def proteins_above_lod(
    input_file: str | None,
    accession: str | None,
    output: str | None,
    platform: str | None,
    threshold: float,
    verbose: bool,
) -> None:
    """List UniProt accessions for proteins above LOD."""
    _setup_logging(verbose)

    if input_file is None and accession is None:
        click.echo("Error: Provide either an input file or --accession / -a.", err=True)
        sys.exit(1)

    if input_file is not None and accession is not None:
        click.echo("Error: Provide either an input file or --accession, not both.", err=True)
        sys.exit(1)

    all_proteins: set[str] = set()

    if accession is not None:
        accession = accession.upper()
        click.echo(f"Fetching data from PRIDE for {accession}...", err=True)

        with tempfile.TemporaryDirectory(prefix="pyprideap_") as tmpdir:
            tmppath = Path(tmpdir)
            files, _sdrf = _download_pad_files(accession, tmppath)

            for f in files:
                try:
                    click.echo(f"Reading {f.name}...", err=True)
                    ds = _read(f, platform=platform)
                    click.echo(
                        f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
                        err=True,
                    )
                    proteins = _get_proteins_above_lod(ds, threshold=threshold)
                    click.echo(f"  {len(proteins)} proteins above LOD (threshold={threshold}%)", err=True)
                    all_proteins.update(proteins)
                except Exception as e:
                    logger.debug("Error processing %s: %s", f.name, e, exc_info=True)
                    click.echo(f"  Skipping {f.name}: {e}", err=True)
    else:
        input_path = Path(input_file)  # type: ignore[arg-type]
        if not input_path.exists():
            click.echo(f"Error: File not found: {input_path}", err=True)
            sys.exit(1)

        click.echo(f"Reading {input_path.name}...", err=True)
        ds = _read(input_path, platform=platform)
        click.echo(
            f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
            err=True,
        )
        proteins = _get_proteins_above_lod(ds, threshold=threshold)
        click.echo(f"  {len(proteins)} proteins above LOD (threshold={threshold}%)", err=True)
        all_proteins.update(proteins)

    output_text = "\n".join(sorted(all_proteins))
    if output:
        Path(output).write_text(output_text + "\n")
        click.echo(f"  Saved {len(all_proteins)} unique proteins to {output}", err=True)
    else:
        click.echo(output_text)


@main.command("unique-samples")
@click.argument("input_file", required=False, default=None)
@click.option("-a", "--accession", default=None, help="PAD accession to download from PRIDE (e.g. PAD000001).")
@click.option("-o", "--output", default=None, help="Output file path (default: print to stdout).")
@click.option(
    "-p",
    "--platform",
    type=click.Choice(["olink", "somascan", "nulisa"], case_sensitive=False),
    default=None,
    help="Force platform type (default: auto-detect).",
)
@click.option(
    "--exclude-controls",
    is_flag=True,
    default=False,
    help="Exclude control/QC samples (default: include all).",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging output.")
def unique_samples(
    input_file: str | None,
    accession: str | None,
    output: str | None,
    platform: str | None,
    exclude_controls: bool,
    verbose: bool,
) -> None:
    """List unique sample identifiers in a dataset."""
    _setup_logging(verbose)

    if input_file is None and accession is None:
        click.echo("Error: Provide either an input file or --accession / -a.", err=True)
        sys.exit(1)

    if input_file is not None and accession is not None:
        click.echo("Error: Provide either an input file or --accession, not both.", err=True)
        sys.exit(1)

    all_samples: set[str] = set()

    if accession is not None:
        accession = accession.upper()
        click.echo(f"Fetching data from PRIDE for {accession}...", err=True)

        with tempfile.TemporaryDirectory(prefix="pyprideap_") as tmpdir:
            tmppath = Path(tmpdir)
            files, _sdrf = _download_pad_files(accession, tmppath)

            for f in files:
                try:
                    click.echo(f"Reading {f.name}...", err=True)
                    ds = _read(f, platform=platform)
                    click.echo(
                        f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
                        err=True,
                    )
                    samples = _get_unique_samples(ds, exclude_controls=exclude_controls)
                    click.echo(f"  {len(samples)} unique samples", err=True)
                    all_samples.update(samples)
                except Exception as e:
                    logger.debug("Error processing %s: %s", f.name, e, exc_info=True)
                    click.echo(f"  Skipping {f.name}: {e}", err=True)
    else:
        input_path = Path(input_file)  # type: ignore[arg-type]
        if not input_path.exists():
            click.echo(f"Error: File not found: {input_path}", err=True)
            sys.exit(1)

        click.echo(f"Reading {input_path.name}...", err=True)
        ds = _read(input_path, platform=platform)
        click.echo(
            f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
            err=True,
        )
        samples = _get_unique_samples(ds, exclude_controls=exclude_controls)
        click.echo(f"  {len(samples)} unique samples", err=True)
        all_samples.update(samples)

    output_text = "\n".join(sorted(all_samples))
    if output:
        Path(output).write_text(output_text + "\n")
        click.echo(f"  Saved {len(all_samples)} unique samples to {output}", err=True)
    else:
        click.echo(output_text)


if __name__ == "__main__":
    main()
