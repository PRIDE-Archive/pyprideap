"""NULISAseq (Alamar) support: readers, XML annotation and QC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import Platform

_PLATE = "20250515_Bay1_Inflammation Panel 250 _V2_.xml"
_TARGETS = [("t1", "AGER", "Q15109"), ("t2", "AGRP", "O00253"), ("t3", "ANGPT1", "Q15389"), ("t4", "IL6", "P05231")]

_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<NULISAseq>
  <Barcodes><BarcodeA>
    {
    "".join(
        f'<Barcode name="b{i}" AlamarTargetID="{tid}" ProteinName="Protein {sym}" UniProtID="{uni}">{sym}</Barcode>'
        for i, (tid, sym, uni) in enumerate(_TARGETS)
    )
}
    <Barcode name="ic" AlamarTargetID="t0" ProteinName="mCherry">mCherry</Barcode>
  </BarcodeA></Barcodes>
  <PlateQC>
    <Threshold name="ICRead_CV" thresholds="0.25" operators="&gt;" format="percentage" properNames="IC CV"/>
    <Threshold name="MinReads" thresholds="1e+08" operators="&lt;" format="integer" properNames="Reads"/>
  </PlateQC>
  <PlateQC>
    <QCFlag name="ICRead_CV" method="raw" set="FALSE" format="percentage">0.12</QCFlag>
    <QCFlag name="MinReads" method="raw" set="TRUE" format="integer">5e7</QCFlag>
  </PlateQC>
  <SampleQC>
    <Threshold name="NumReads" thresholds="4e+05" operators="&lt;" format="integer" properNames="Reads"/>
  </SampleQC>
</NULISAseq>
"""


def _samples() -> list[tuple[str, str, str]]:
    """(name, type, disease) for 8 study samples and NC / SC / IPC controls."""
    study = [(f"S{i}", "Sample", "case" if i < 4 else "normal") for i in range(8)]
    controls = [("NC_1", "NC", ""), ("NC_2", "NC", ""), *[(f"SC_{i}", "SC", "") for i in range(3)]]
    return study + controls + [(f"IPC_{i}", "IPC", "") for i in range(3)]


def _write_long(directory, with_xml: bool = True):
    rng = np.random.default_rng(1)
    rows = []
    for name, stype, disease in _samples():
        warn = stype == "NC"
        for tid, sym, _ in _TARGETS:
            npq = rng.normal(2.0 if stype == "NC" else 12.0, 0.3)
            rows.append(
                {
                    "Panel": "Inflammation Panel 250 (V2)",
                    "PlateID": _PLATE,
                    "SampleName": name,
                    "SampleType": stype,
                    "Target": sym,
                    "AlamarTargetID": tid,
                    "LOD": 5.0,
                    "UnnormalizedCount": int(rng.integers(100, 10000)),
                    "NPQ": npq,
                    "Sample_QC_Status": "WARN" if warn else "PASS",
                    "Sample_QC_IC_Median": np.nan if warn else rng.normal(0, 0.1),
                    "Sample_QC_Detectability": 1.0,
                    "Sample_QC_ICReads": 50000,
                    "Sample_QC_NumReads": np.nan if warn else 2e6,
                    "Sample_QC_NumReads_Status": "WARN" if warn else "PASS",
                    "AUTO_WELLPOSITION": "A1",
                    "SAMPLE_MATRIX": "CONTROL" if stype != "Sample" else "PLASMA",
                    "Disease": disease or np.nan,
                }
            )
        # internal control: counts only, no NPQ
        rows.append({**rows[-1], "Target": "mCherry", "AlamarTargetID": "t0", "NPQ": np.nan, "LOD": np.nan})
    path = directory / "NPQ_Long_Format.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    if with_xml:
        (directory / _PLATE).write_text(_XML)
    return path


def _write_workbook(directory):
    names = [n for n, _, _ in _samples()]
    npq = pd.DataFrame({"targetName": [s for _, s, _ in _TARGETS], **{n: [12.0 + k for k in range(4)] for n in names}})
    targets = pd.DataFrame({"targetName": [s for _, s, _ in _TARGETS], "targetLOD_NPQ": [5.0] * 4, "plateID": _PLATE})
    ann = pd.DataFrame(
        {
            "plateID": _PLATE,
            "sampleType": [t for _, t, _ in _samples()],
            "sampleName": names,
            "Disease": [d or np.nan for _, _, d in _samples()],
        }
    )
    path = directory / "NPQ_Values.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"Field": ["Assay Type"], "Value": ["Inflammation Panel 250 (V2)"]}).to_excel(
            xw, sheet_name="Summary", index=False
        )
        npq.to_excel(xw, sheet_name="NPQ", index=False)
        targets.to_excel(xw, sheet_name="Annotation-Targets", index=False)
        ann.to_excel(xw, sheet_name="Annotation-Samples", index=False)
    (directory / _PLATE).write_text(_XML)
    return path


class TestNulisaReaders:
    def test_long_format_with_xml_annotation(self, tmp_path):
        from pyprideap.io.readers.registry import detect_format, read

        path = _write_long(tmp_path)
        assert detect_format(path) == "nulisa_csv"
        ds = read(path)
        assert ds.platform == Platform.NULISA
        assert ds.expression.shape == (len(_samples()), 4)  # mCherry (no NPQ) dropped
        assert ds.features["UniProt"].tolist() == [u for _, _, u in _TARGETS]
        assert ds.samples["SampleType"].value_counts().to_dict() == {"Sample": 8, "SC": 3, "IPC": 3, "NC": 2}
        assert ds.metadata["lod_matrix"].shape == ds.expression.shape
        assert {"SampleQC", "QC_Reads", "QC_ICMedian", "Disease"} <= set(ds.samples.columns)
        qc = ds.metadata["nulisa_plate_qc"][0]["metrics"]
        assert {m["label"]: m["passed"] for m in qc} == {"IC CV": True, "Reads": False}

    def test_long_format_without_xml_has_no_uniprot(self, tmp_path):
        from pyprideap.io.readers.nulisa import read_nulisa_csv

        ds = read_nulisa_csv(_write_long(tmp_path, with_xml=False))
        assert ds.features["UniProt"].isna().all()
        assert "nulisa_plate_qc" not in ds.metadata

    def test_workbook_maps_gene_symbols_through_xml(self, tmp_path):
        from pyprideap.io.readers.registry import detect_format, read

        path = _write_workbook(tmp_path)
        assert detect_format(path) == "nulisa_xlsx"
        ds = read(path)
        assert ds.expression.columns.tolist() == [t for t, _, _ in _TARGETS]
        assert ds.features["UniProt"].tolist() == [u for _, _, u in _TARGETS]
        assert ds.metadata["lod_matrix"].columns.tolist() == ds.expression.columns.tolist()

    def test_instrument_xml_alone_gives_clear_error(self, tmp_path):
        from pyprideap.io.readers.registry import read

        xml = tmp_path / _PLATE
        xml.write_text(_XML)
        with pytest.raises(ValueError, match="NPQ export"):
            read(xml)


class TestNulisaQc:
    def test_qc_uses_nulisa_lod_controls_and_thresholds(self, tmp_path):
        from pyprideap.processing.lod import get_proteins_above_lod, resolve_lod_with_source
        from pyprideap.viz.qc.compute import compute_all

        ds = __import__("pyprideap").read(_write_long(tmp_path))
        assert resolve_lod_with_source(ds)[1] == "Reported LOD"
        assert get_proteins_above_lod(ds) == sorted(u for _, _, u in _TARGETS)
        r = compute_all(ds)
        assert "iqr_median_qc" not in r and "lod_comparison" not in r
        assert r["replicate_cv"].control_label == "Sample controls" and r["replicate_cv"].n_replicates == 3
        assert r["lod_analysis"].unit == "NPQ" and r["lod_analysis"].n_negative == 2
        nq = r["nulisa_qc"]
        reads = next(m for m in nq.metrics if m.key == "QC_Reads")
        assert reads.thresholds == [4e5]  # from the XML, not the 500,000 default
        assert nq.sample_status == {"PASS": 8}  # negative controls excluded
        assert r["mpi"].group_column == "Disease"

    def test_reports_render(self, tmp_path):
        import json

        from pyprideap.viz.qc.report import qc_report, qc_report_split

        ds = __import__("pyprideap").read(_write_long(tmp_path))
        out = qc_report_split(ds, tmp_path / "split", no_border=True)
        manifest = json.loads((out / "manifest.json").read_text())
        technical = next(t for t in manifest["tabs"] if t["id"] == "technical")
        assert "nulisa_qc" in [i["key"] for i in technical["items"]] and not manifest["unplaced"]
        summary = (out / "summary.html").read_text()
        assert "NULISA Run QC" in summary and "Nulisa" in summary
        qc_report(ds, tmp_path / "full.html")
        full = (tmp_path / "full.html").read_text()
        assert 'id="nulisa_qc"' in full and "NULISA Sample" in full
