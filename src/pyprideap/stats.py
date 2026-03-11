from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset


@dataclass
class DatasetStats:
    n_samples: int
    n_features: int
    features_per_sample: pd.Series
    samples_per_feature: pd.Series
    detection_rate: float
    sample_types: dict[str, int]
    panels: dict[str, int]
    qc_summary: dict[str, int]
    value_distribution: dict[str, float]

    def summary(self) -> str:
        lines = [
            f"{self.n_samples} samples",
            f"{self.n_features} features",
            f"Detection rate:   {self.detection_rate:.1%}",
            f"Sample types:     {self.sample_types}",
            f"Panels:           {self.panels}",
            f"QC summary:       {self.qc_summary}",
            f"Value stats:      mean={self.value_distribution['mean']:.2f}, "
            f"median={self.value_distribution['median']:.2f}, "
            f"std={self.value_distribution['std']:.2f}",
        ]
        return "\n".join(lines)


def compute_stats(dataset: AffinityDataset) -> DatasetStats:
    expr = dataset.expression
    total_values = expr.size
    non_nan = expr.count().sum()

    features_per_sample = expr.count(axis=1)
    samples_per_feature = expr.count(axis=0)
    detection_rate = non_nan / total_values if total_values > 0 else 0.0

    sample_types: dict[str, int] = {}
    if "SampleType" in dataset.samples.columns:
        sample_types = {str(k): int(v) for k, v in dataset.samples["SampleType"].value_counts().items()}

    panels: dict[str, int] = {}
    if "Panel" in dataset.features.columns:
        panels = {str(k): int(v) for k, v in dataset.features["Panel"].value_counts().items()}

    qc_summary: dict[str, int] = {}
    if "SampleQC" in dataset.samples.columns:
        qc_summary = {str(k): int(v) for k, v in dataset.samples["SampleQC"].value_counts().items()}

    flat = np.asarray(pd.to_numeric(pd.Series(expr.values.flatten()), errors="coerce").dropna())
    value_distribution = {
        "mean": float(np.mean(flat)) if len(flat) > 0 else 0.0,
        "median": float(np.median(flat)) if len(flat) > 0 else 0.0,
        "std": float(np.std(flat)) if len(flat) > 0 else 0.0,
        "min": float(np.min(flat)) if len(flat) > 0 else 0.0,
        "max": float(np.max(flat)) if len(flat) > 0 else 0.0,
    }

    return DatasetStats(
        n_samples=len(dataset.samples),
        n_features=len(dataset.features),
        features_per_sample=features_per_sample,
        samples_per_feature=samples_per_feature,
        detection_rate=detection_rate,
        sample_types=sample_types,
        panels=panels,
        qc_summary=qc_summary,
        value_distribution=value_distribution,
    )
