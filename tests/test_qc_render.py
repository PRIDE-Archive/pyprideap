import pytest

plotly = pytest.importorskip("plotly")

from pyprideap.qc.compute import (  # noqa: E402
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    DistributionData,
    LodAnalysisData,
    MissingValuesData,
    PcaData,
    QcSummaryData,
)
from pyprideap.qc.render import (  # noqa: E402
    render_correlation,
    render_cv_distribution,
    render_detection_rate,
    render_distribution,
    render_lod_analysis,
    render_missing_values,
    render_pca,
    render_qc_summary,
)


class TestRenderFunctions:
    def test_render_distribution(self):
        data = DistributionData(values=[1.0, 2.0, 3.0], xlabel="NPX (log2)")
        fig = render_distribution(data)
        assert fig is not None

    def test_render_qc_summary(self):
        data = QcSummaryData(categories=["PASS", "FAIL"], counts=[10, 2])
        fig = render_qc_summary(data)
        assert fig is not None

    def test_render_lod_analysis(self):
        data = LodAnalysisData(assay_ids=["A1", "A2"], above_lod_pct=[95.0, 80.0], panel=["Inf", "Inf"])
        fig = render_lod_analysis(data)
        assert fig is not None

    def test_render_pca(self):
        data = PcaData(
            pc1=[1.0, 2.0], pc2=[3.0, 4.0], variance_explained=[0.6, 0.3], labels=["S1", "S2"], groups=["A", "B"]
        )
        fig = render_pca(data)
        assert fig is not None

    def test_render_correlation(self):
        data = CorrelationData(matrix=[[1.0, 0.5], [0.5, 1.0]], labels=["S1", "S2"])
        fig = render_correlation(data)
        assert fig is not None

    def test_render_missing_values(self):
        data = MissingValuesData(
            missing_rate_per_sample=[0.1, 0.2],
            missing_rate_per_feature=[0.05, 0.15],
            sample_ids=["S1", "S2"],
            feature_ids=["F1", "F2"],
        )
        fig = render_missing_values(data)
        assert fig is not None

    def test_render_cv_distribution(self):
        data = CvDistributionData(feature_ids=["F1", "F2"], cv_values=[0.1, 0.2])
        fig = render_cv_distribution(data)
        assert fig is not None

    def test_render_detection_rate(self):
        data = DetectionRateData(sample_ids=["S1", "S2"], rates=[0.9, 0.95])
        fig = render_detection_rate(data)
        assert fig is not None
