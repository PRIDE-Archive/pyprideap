"""Readers for Alamar Biosciences NULISAseq exports.

NULISAseq reports protein levels as NPQ (NULISA Protein Quantification), a log2
scale like Olink NPX. Two exports from the Alamar Protein Quantification (APQ)
software are supported:

* the NPQ long format CSV (one row per sample x target), which carries the
  per-target LOD, raw read counts and the per-sample QC metrics, and
* the NPQ workbook (``NPQ`` sheet, samples as columns, with
  ``Annotation-Targets`` and ``Annotation-Samples`` sheets).

Neither export contains UniProt accessions. The instrument output XML
(``<NULISAseq>``, named in the ``PlateID`` column) does; when it sits next to
the data file it is used to add UniProt IDs, protein names and the plate QC
metrics with their thresholds.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.io.readers.olink_csv import _long_to_wide, _plain, _sniff_delimiter, _warn_data_quality

logger = logging.getLogger(__name__)

# Columns identifying the NPQ long format
NULISA_LONG_MARKER_COLS = {"SampleName", "Target", "NPQ"}

# Per-sample QC metrics of the long format, renamed to readable names
_SAMPLE_QC_COLUMNS = {
    "Sample_QC_Status": "SampleQC",
    "Sample_QC_NumReads": "QC_Reads",
    "Sample_QC_NumReads_Status": "QC_Reads_Status",
    "Sample_QC_ICReads": "QC_ICReads",
    "Sample_QC_ICReads_Status": "QC_ICReads_Status",
    "Sample_QC_IC_Median": "QC_ICMedian",
    "Sample_QC_IC_Median_Status": "QC_ICMedian_Status",
    "Sample_QC_Detectability": "QC_Detectability",
    "Sample_QC_Detectability_Status": "QC_Detectability_Status",
}
# Other per-sample columns kept as sample metadata
_SAMPLE_META_COLUMNS = ("PlateID", "SampleType", "SAMPLE_MATRIX", "Disease", "Lab", "sampleBarcode", "InstrumentID")


def is_nulisa_xml(path: Path) -> bool:
    """True for a NULISAseq instrument output XML."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return False
    return "<NULISAseq" in head


def read_nulisa_xml_annotation(path: str | Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Target annotation and plate QC from a NULISAseq instrument XML.

    Returns ``(targets, plate_qc)``: one row per target (``AlamarTargetID``,
    ``UniProt``, ``ProteinName``) and one entry per plate QC metric with its
    value, threshold and whether it passed.
    """
    root = ET.parse(path).getroot()
    targets = {}
    for barcode in root.iter("Barcode"):
        target_id = barcode.get("AlamarTargetID")
        if target_id and target_id not in targets:
            targets[target_id] = {
                "AlamarTargetID": target_id,
                "Target": (barcode.text or "").strip() or pd.NA,  # gene symbol
                "UniProt": barcode.get("UniProtID") or pd.NA,
                "ProteinName": barcode.get("ProteinName") or pd.NA,
            }

    thresholds: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, str]] = {}
    for plate_qc in root.iter("PlateQC"):
        for child in plate_qc:
            name = child.get("name")
            if not name:
                continue
            if child.tag == "Threshold":
                thresholds[name] = dict(child.attrib)
            elif child.tag == "QCFlag":
                values[name] = {**child.attrib, "value": (child.text or "").strip()}

    plate_qc_rows: list[dict[str, object]] = []
    for name, flag in values.items():
        spec = thresholds.get(name, {})
        try:
            value = float(flag["value"])
        except ValueError:
            continue
        plate_qc_rows.append(
            {
                "name": name,
                "label": spec.get("properNames", name),
                "value": value,
                "format": spec.get("format", flag.get("format", "")),
                "threshold": spec.get("thresholds", ""),
                "operator": spec.get("operators", ""),
                "explanation": spec.get("explanations", ""),
                # set="TRUE" means the QC flag was raised
                "passed": str(flag.get("set", "")).upper() != "TRUE",
            }
        )
    return pd.DataFrame(list(targets.values())), plate_qc_rows


def read_nulisa_sample_qc_thresholds(path: str | Path) -> dict[str, dict[str, str]]:
    """Sample QC thresholds declared in a NULISAseq XML (``<SampleQC><Threshold .../>``)."""
    root = ET.parse(path).getroot()
    out: dict[str, dict[str, str]] = {}
    for sample_qc in root.iter("SampleQC"):
        for child in sample_qc:
            if child.tag == "Threshold" and child.get("name"):
                out[str(child.get("name"))] = dict(child.attrib)
    return out


def _find_xml(path: Path, plate_ids: list[str]) -> list[Path]:
    """Instrument XMLs next to *path*: those named in PlateID, else any NULISAseq XML."""
    named = [path.parent / p for p in plate_ids if p and (path.parent / p).is_file()]
    if named:
        return named
    return [p for p in sorted(path.parent.glob("*.xml")) if is_nulisa_xml(p)]


def _annotate(features: pd.DataFrame, metadata: dict[str, object], xmls: list[Path]) -> pd.DataFrame:
    """Add UniProt / protein names and plate QC from instrument XMLs."""
    annotations, plate_qc = [], []
    for xml in xmls:
        try:
            targets, qc = read_nulisa_xml_annotation(xml)
        except (ET.ParseError, OSError) as exc:
            logger.warning("Could not read NULISAseq XML %s: %s", xml.name, exc)
            continue
        annotations.append(targets)
        plate_qc.append({"plate": xml.name, "metrics": qc})
        try:
            metadata.setdefault("nulisa_sample_qc_thresholds", read_nulisa_sample_qc_thresholds(xml))
        except (ET.ParseError, OSError):
            pass
    if plate_qc:
        metadata["nulisa_plate_qc"] = plate_qc
    if not annotations:
        if "UniProt" not in features.columns:
            features["UniProt"] = pd.Series(pd.NA, index=features.index, dtype="object")
        return features
    ann = pd.concat(annotations).drop_duplicates("AlamarTargetID")
    by_id = ann.set_index("AlamarTargetID")
    by_symbol = ann.dropna(subset=["Target"]).drop_duplicates("Target").set_index("Target")
    # The workbook export has gene symbols only: take the target ID from the XML
    no_id = features["AlamarTargetID"] == features["Assay"]
    features.loc[no_id, "AlamarTargetID"] = (
        features.loc[no_id, "Assay"].map(by_symbol["AlamarTargetID"]).fillna(features.loc[no_id, "Assay"])
    )
    for col in ("UniProt", "ProteinName"):
        mapped = features["AlamarTargetID"].map(by_id[col]).fillna(features["Assay"].map(by_symbol[col]))
        features[col] = features[col].fillna(mapped) if col in features.columns else mapped
    metadata["nulisa_annotation_source"] = [x.name for x in xmls]
    return features


def read_nulisa_csv(path: str | Path) -> AffinityDataset:
    """Read the NULISAseq NPQ long format CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sep = _sniff_delimiter(path) or ","
    df = pd.read_csv(path, sep=sep, low_memory=False, dtype={"SampleName": str, "SampleType": str})
    missing = NULISA_LONG_MARKER_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required NULISAseq columns in {path.name}: {sorted(missing)}")
    if "AlamarTargetID" not in df.columns:
        df["AlamarTargetID"] = df["Target"]
    for col in ("NPQ", "LOD"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sample_key = "SampleName"
    sample_cols = [sample_key, *[c for c in (*_SAMPLE_META_COLUMNS, *_SAMPLE_QC_COLUMNS) if c in df.columns]]
    if "AUTO_WELLPOSITION" in df.columns:
        sample_cols.append("AUTO_WELLPOSITION")
    samples = _plain(df[sample_cols].drop_duplicates(subset=[sample_key]).reset_index(drop=True))
    samples = samples.rename(
        columns={**_SAMPLE_QC_COLUMNS, "AUTO_WELLPOSITION": "WellID", "SAMPLE_MATRIX": "SampleMatrix"}
    )
    samples.insert(0, "SampleID", samples[sample_key].astype(str))

    feature_cols = ["AlamarTargetID", "Target", *[c for c in ("Panel", "PanelLotNumber") if c in df.columns]]
    features = df[feature_cols].drop_duplicates(subset=["AlamarTargetID"]).rename(columns={"Target": "Assay"})

    sample_order = pd.Index(samples[sample_key].astype(str))
    df[sample_key] = df[sample_key].astype(str)
    expression = _long_to_wide(df, sample_key, sample_order, "NPQ", assay_col="AlamarTargetID")
    metadata: dict[str, object] = {"source_file": str(path), "value_unit": "NPQ"}
    if "LOD" in df.columns:
        lod = _long_to_wide(df, sample_key, sample_order, "LOD", assay_col="AlamarTargetID")
        metadata["lod_matrix"] = lod.reindex(columns=expression.columns)
    if "UnnormalizedCount" in df.columns:
        counts = _long_to_wide(df, sample_key, sample_order, "UnnormalizedCount", assay_col="AlamarTargetID")
        metadata["read_count_matrix"] = counts.reindex(columns=expression.columns)
    expression.columns.name = None

    features = features.set_index("AlamarTargetID").reindex(expression.columns).reset_index()
    features = features.rename(columns={"index": "AlamarTargetID"})
    plate_ids = sorted(df["PlateID"].dropna().astype(str).unique()) if "PlateID" in df.columns else []
    features = _annotate(features, metadata, _find_xml(path, plate_ids))

    dataset = AffinityDataset(
        platform=Platform.NULISA, samples=samples, features=features, expression=expression, metadata=metadata
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset


def is_nulisa_workbook(path: Path) -> bool:
    """True for the NULISAseq NPQ workbook (sheets ``NPQ`` and ``Annotation-Targets``)."""
    try:
        sheets = set(pd.ExcelFile(path).sheet_names)
    except Exception:  # not a readable workbook
        return False
    return {"NPQ", "Annotation-Targets"} <= sheets


def read_nulisa_xlsx(path: str | Path) -> AffinityDataset:
    """Read the NULISAseq NPQ workbook (targets as rows, samples as columns)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    book = pd.ExcelFile(path)
    npq = book.parse("NPQ").set_index("targetName")
    targets = book.parse("Annotation-Targets")
    sample_ann = book.parse("Annotation-Samples") if "Annotation-Samples" in book.sheet_names else pd.DataFrame()

    sample_names = [str(c) for c in npq.columns]
    expression = npq.T.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    expression.columns = [str(c) for c in npq.index]

    samples = pd.DataFrame({"SampleID": sample_names, "SampleName": sample_names})
    if not sample_ann.empty and "sampleName" in sample_ann.columns:
        ann = sample_ann.drop_duplicates("sampleName").set_index("sampleName")
        rename = {"sampleType": "SampleType", "plateID": "PlateID", "AUTO_WELLPOSITION": "WellID"}
        for src, dst in rename.items():
            if src in ann.columns:
                samples[dst] = samples["SampleName"].map(ann[src])
        for col in ("SAMPLE_MATRIX", "Disease", "Lab", "sampleBarcode"):
            if col in ann.columns:
                samples["SampleMatrix" if col == "SAMPLE_MATRIX" else col] = samples["SampleName"].map(ann[col])

    t = targets.set_index("targetName")
    features = pd.DataFrame({"AlamarTargetID": expression.columns, "Assay": expression.columns})
    if "AlamarTargetID" in t.columns:
        features["AlamarTargetID"] = features["Assay"].map(t["AlamarTargetID"]).fillna(features["Assay"])
    metadata: dict[str, object] = {"source_file": str(path), "value_unit": "NPQ"}
    if "targetLOD_NPQ" in t.columns:
        lod = pd.to_numeric(features["Assay"].map(t["targetLOD_NPQ"]), errors="coerce").to_numpy()
        metadata["lod_matrix"] = pd.DataFrame(
            np.tile(lod, (len(samples), 1)), columns=features["AlamarTargetID"].tolist()
        )
    plate_ids = sorted(samples["PlateID"].dropna().astype(str).unique()) if "PlateID" in samples.columns else []
    features = _annotate(features, metadata, _find_xml(path, plate_ids))
    # Expression and LOD columns keyed by AlamarTargetID, as in the long format
    expression.columns = features["AlamarTargetID"].tolist()
    if "lod_matrix" in metadata:
        lod_matrix = metadata["lod_matrix"]
        assert isinstance(lod_matrix, pd.DataFrame)
        lod_matrix.columns = expression.columns

    dataset = AffinityDataset(
        platform=Platform.NULISA, samples=samples, features=features, expression=expression, metadata=metadata
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset
