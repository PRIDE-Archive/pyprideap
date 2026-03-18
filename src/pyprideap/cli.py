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
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import click

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


def _download_pad_files(accession: str, dest_dir: Path) -> list[Path]:
    """Download NPX/ADAT files from PRIDE for a given PAD accession."""
    from pyprideap.api.pride import PrideClient

    client = PrideClient()
    logger.debug("Fetching file list for %s from PRIDE API...", accession)
    urls = client.get_download_urls(accession)

    # Filter for supported data files
    data_extensions = {".csv", ".parquet", ".xlsx", ".adat"}
    data_urls = {
        name: url
        for name, url in urls.items()
        if any(name.lower().endswith(ext) for ext in data_extensions) and "checksum" not in name.lower()
    }

    if not data_urls:
        click.echo(f"Error: No supported data files found for {accession}", err=True)
        click.echo(f"Available files: {list(urls.keys())}", err=True)
        sys.exit(1)

    logger.debug("Found %d data file(s): %s", len(data_urls), list(data_urls.keys()))

    downloaded: list[Path] = []
    for name, url in data_urls.items():
        dest = dest_dir / Path(name).name  # sanitize: strip directory components
        # Convert FTP URLs to HTTPS for broader compatibility
        if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            url = url.replace("ftp://ftp.pride.ebi.ac.uk/", "https://ftp.pride.ebi.ac.uk/", 1)
        click.echo(f"Downloading {name}...")
        logger.debug("URL: %s -> %s", url, dest)
        import urllib.request

        urllib.request.urlretrieve(url, dest)
        downloaded.append(dest)

    return downloaded


def _generate_report(
    input_path: Path,
    output_path: Path | None,
    platform: str | None = None,
    split: bool = False,
    sdrf_path: Path | None = None,
    no_border: bool = True,
) -> Path:
    """Read a data file and generate a QC report."""
    import pyprideap as pp

    click.echo(f"Reading {input_path.name}...")
    logger.debug("Full path: %s", input_path)
    ds = pp.read(input_path, platform=platform)
    click.echo(f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})")
    logger.debug("Samples columns: %s", list(ds.samples.columns))
    logger.debug("Features columns: %s", list(ds.features.columns))

    if split:
        from pyprideap.viz.qc.report import qc_report_split

        if output_path is None:
            stem = input_path.stem
            if stem.endswith(".npx") or stem.endswith(".ct"):
                stem = Path(stem).stem
            output_path = Path(f"{stem}_qc_plots")

        click.echo("Generating individual plot files...")
        logger.debug("Output directory: %s", output_path)
        result = qc_report_split(ds, output_path, no_border=no_border)
        n_files = len(list(result.glob("*.html")))
        click.echo(f"  {n_files} HTML files saved to {result}/")
        return result

    if output_path is None:
        stem = input_path.stem
        # Handle double extensions like .npx.csv
        if stem.endswith(".npx") or stem.endswith(".ct"):
            stem = Path(stem).stem
        output_path = Path(f"{stem}_qc_report.html")

    if sdrf_path is not None:
        click.echo(f"  SDRF: {sdrf_path.name}")
        logger.debug("SDRF full path: %s", sdrf_path)

    click.echo("Generating report...")
    logger.debug("Output path: %s", output_path)
    result = pp.qc_report(ds, output_path, sdrf_path=sdrf_path)
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
    type=click.Choice(["olink", "somascan"], case_sensitive=False),
    default=None,
    help="Force platform type (default: auto-detect).",
)
@click.option(
    "--split", is_flag=True, default=False, help="Output individual plot HTML files instead of a single report."
)
@click.option("--sdrf", default=None, type=click.Path(exists=True), help="Path to SDRF TSV file for volcano plots.")
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
            files = _download_pad_files(accession, tmppath)

            for f in files:
                try:
                    stem = f.stem
                    if stem.endswith(".npx") or stem.endswith(".ct"):
                        stem = Path(stem).stem
                    if split:
                        out = Path(f"{output_path}/{stem}")
                    else:
                        out = Path(f"{output_path}/{stem}.html")
                    _generate_report(f, out, platform=platform, split=split, sdrf_path=sdrf_path, no_border=no_border)
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
    type=click.Choice(["olink", "somascan"], case_sensitive=False),
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

    import pyprideap as pp

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
            files = _download_pad_files(accession, tmppath)

            for f in files:
                try:
                    click.echo(f"Reading {f.name}...", err=True)
                    ds = pp.read(f, platform=platform)
                    click.echo(
                        f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
                        err=True,
                    )
                    proteins = pp.get_proteins_above_lod(ds, threshold=threshold)
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
        ds = pp.read(input_path, platform=platform)
        click.echo(
            f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
            err=True,
        )
        proteins = pp.get_proteins_above_lod(ds, threshold=threshold)
        click.echo(f"  {len(proteins)} proteins above LOD (threshold={threshold}%)", err=True)
        all_proteins.update(proteins)

    output_text = "\n".join(sorted(all_proteins))
    if output:
        Path(output).write_text(output_text + "\n")
        click.echo(f"  Saved {len(all_proteins)} unique proteins to {output}", err=True)
    else:
        click.echo(output_text)


if __name__ == "__main__":
    main()
