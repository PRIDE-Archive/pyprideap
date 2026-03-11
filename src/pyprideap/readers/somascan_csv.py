from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform


def read_somascan_csv(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    seq_cols = [c for c in df.columns if c.startswith("SeqId.")]
    if not seq_cols:
        raise ValueError(f"No SeqId.* columns found in {path.name}")

    meta_cols = [c for c in df.columns if c not in seq_cols]

    samples = df[meta_cols].reset_index(drop=True)
    expression = df[seq_cols].astype(float).reset_index(drop=True)

    seq_ids = [c.replace("SeqId.", "") for c in seq_cols]
    features = pd.DataFrame(
        {
            "SeqId": seq_ids,
            "UniProt": pd.Series([pd.NA] * len(seq_ids), dtype="string"),
            "Target": pd.Series([pd.NA] * len(seq_ids), dtype="string"),
            "Dilution": pd.Series([pd.NA] * len(seq_ids), dtype="string"),
        }
    )

    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expression,
        metadata={"source_file": str(path)},
    )
