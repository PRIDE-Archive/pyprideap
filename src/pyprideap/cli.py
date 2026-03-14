"""Command-line interface for pyprideap.

Usage:
    pyprideap report <file>               Generate QC report from a local NPX/ADAT file
    pyprideap report <PAD accession>      Download data from PRIDE and generate report
    pyprideap report <file> -o out.html   Specify output path
    pyprideap report <file> -p olink      Force platform type
    pyprideap report <file> --split       Output individual plots as separate HTML files
    pyprideap proteins-above-lod <file>   List UniProt accessions above LOD
    pyprideap proteins-above-lod <file> -t 80   Custom threshold (default 50%)

Examples:
    pyprideap report data.npx.csv
    pyprideap report data.parquet -o my_report.html
    pyprideap report data.npx.csv --split -o plots_dir/
    pyprideap report PAD000001
    pyprideap report ambiguous.csv -p somascan
    pyprideap proteins-above-lod data.npx.csv
    pyprideap proteins-above-lod data.npx.csv -t 80 -o proteins.txt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _download_pad_files(accession: str, dest_dir: Path) -> list[Path]:
    """Download NPX/ADAT files from PRIDE for a given PAD accession."""
    from pyprideap.api.pride import PrideClient

    client = PrideClient()
    urls = client.get_download_urls(accession)

    # Filter for supported data files
    data_extensions = {".csv", ".parquet", ".xlsx", ".adat"}
    data_urls = {
        name: url
        for name, url in urls.items()
        if any(name.lower().endswith(ext) for ext in data_extensions) and "checksum" not in name.lower()
    }

    if not data_urls:
        print(f"Error: No supported data files found for {accession}", file=sys.stderr)
        print(f"Available files: {list(urls.keys())}", file=sys.stderr)
        sys.exit(1)

    downloaded: list[Path] = []
    for name, url in data_urls.items():
        dest = dest_dir / name
        # Convert FTP URLs to HTTPS for broader compatibility
        if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            url = url.replace("ftp://ftp.pride.ebi.ac.uk/", "https://ftp.pride.ebi.ac.uk/", 1)
        print(f"Downloading {name}...")
        import urllib.request

        urllib.request.urlretrieve(url, dest)
        downloaded.append(dest)

    return downloaded


def _generate_report(
    input_path: Path,
    output_path: Path | None,
    platform: str | None = None,
    split: bool = False,
) -> Path:
    """Read a data file and generate a QC report."""
    import pyprideap as pp

    print(f"Reading {input_path.name}...")
    ds = pp.read(input_path, platform=platform)
    print(f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})")

    if split:
        from pyprideap.viz.qc.report import qc_report_split

        if output_path is None:
            stem = input_path.stem
            if stem.endswith(".npx") or stem.endswith(".ct"):
                stem = Path(stem).stem
            output_path = Path(f"{stem}_qc_plots")

        print("Generating individual plot files...")
        result = qc_report_split(ds, output_path)
        n_files = len(list(result.glob("*.html")))
        print(f"  {n_files} HTML files saved to {result}/")
        return result

    if output_path is None:
        stem = input_path.stem
        # Handle double extensions like .npx.csv
        if stem.endswith(".npx") or stem.endswith(".ct"):
            stem = Path(stem).stem
        output_path = Path(f"{stem}_qc_report.html")

    print("Generating report...")
    result = pp.qc_report(ds, output_path)
    print(f"  Report saved to {result}")
    return result


def cmd_report(args: argparse.Namespace) -> None:
    """Handle the 'report' subcommand."""
    input_val = args.input
    output = Path(args.output) if args.output else None
    platform = args.platform
    split = args.split

    # Check if input looks like a PAD accession
    if input_val.upper().startswith("PAD") and not Path(input_val).exists():
        accession = input_val.upper()
        print(f"Fetching data from PRIDE for {accession}...")

        with tempfile.TemporaryDirectory(prefix="pyprideap_") as tmpdir:
            tmppath = Path(tmpdir)
            files = _download_pad_files(accession, tmppath)

            for f in files:
                try:
                    out = output if output and len(files) == 1 else None
                    if out is None:
                        stem = f.stem
                        if stem.endswith(".npx") or stem.endswith(".ct"):
                            stem = Path(stem).stem
                        if split:
                            out = Path(f"{accession}_{stem}_qc_plots")
                        else:
                            out = Path(f"{accession}_{stem}_qc_report.html")
                    _generate_report(f, out, platform=platform, split=split)
                except Exception as e:
                    print(f"  Skipping {f.name}: {e}", file=sys.stderr)
    else:
        input_path = Path(input_val)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        _generate_report(input_path, output, platform=platform, split=split)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyprideap",
        description="pyprideap — PRIDE Affinity Proteomics tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # report subcommand
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a QC report from a data file or PAD accession",
    )
    report_parser.add_argument(
        "input",
        help="Path to NPX/ADAT file, or a PAD accession (e.g. PAD000001)",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        help="Output HTML file path (default: <input>_qc_report.html)",
    )
    report_parser.add_argument(
        "-p",
        "--platform",
        choices=["olink", "somascan"],
        default=None,
        help="Force platform type (default: auto-detect from file)",
    )
    report_parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="Output individual plot HTML files in a folder instead of a single report",
    )

    # proteins-above-lod subcommand
    lod_parser = subparsers.add_parser(
        "proteins-above-lod",
        help="List UniProt accessions for proteins above LOD",
    )
    lod_parser.add_argument(
        "input",
        help="Path to NPX/ADAT file",
    )
    lod_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: print to stdout)",
    )
    lod_parser.add_argument(
        "-p",
        "--platform",
        choices=["olink", "somascan"],
        default=None,
        help="Force platform type (default: auto-detect from file)",
    )
    lod_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=50.0,
        help="Min %% of samples above LOD to include a protein (default: 50)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "report":
        cmd_report(args)
    elif args.command == "proteins-above-lod":
        cmd_proteins_above_lod(args)


def cmd_proteins_above_lod(args: argparse.Namespace) -> None:
    """Handle the 'proteins-above-lod' subcommand."""
    import pyprideap as pp

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path.name}...", file=sys.stderr)
    ds = pp.read(input_path, platform=args.platform)
    print(
        f"  {len(ds.samples)} samples, {len(ds.features)} features ({ds.platform.value})",
        file=sys.stderr,
    )

    proteins = pp.get_proteins_above_lod(ds, threshold=args.threshold)
    print(f"  {len(proteins)} proteins above LOD (threshold={args.threshold}%)", file=sys.stderr)

    output_text = "\n".join(proteins)
    if args.output:
        Path(args.output).write_text(output_text + "\n")
        print(f"  Saved to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
