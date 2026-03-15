# pyprideap — Getting Started

Python library for reading, validating, and analyzing affinity proteomics datasets from the [PRIDE Affinity Archive (PAD)](https://www.ebi.ac.uk/pride/). Supports **Olink** (Explore, Explore HT, Target, Reveal) and **SomaScan** platforms.

## Installation

pyprideap is not yet published on PyPI. Install directly from source:

```bash
pip install "pyprideap[all] @ git+https://github.com/PRIDE-Archive/pyprideap.git"
```

Or install individual extras:

```bash
# Plotting and QC reports only
pip install "pyprideap[plots] @ git+https://github.com/PRIDE-Archive/pyprideap.git"

# Statistical testing only
pip install "pyprideap[stats] @ git+https://github.com/PRIDE-Archive/pyprideap.git"
```

With conda (using the provided environment file):

```bash
conda env create -f environment.yml
conda activate pyprideap
```

For development:

```bash
git clone https://github.com/PRIDE-Archive/pyprideap.git
cd pyprideap
pip install -e ".[dev]"
```

## Core Data Model

All readers produce an `AffinityDataset`, the central data structure:

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class AffinityDataset:
    platform: Platform          # OLINK_EXPLORE, OLINK_EXPLORE_HT, SOMASCAN, etc.
    samples: pd.DataFrame       # Sample metadata (SampleID, SampleType, QC flags, ...)
    features: pd.DataFrame      # Protein/aptamer annotations (OlinkID, UniProt, Panel, ...)
    expression: pd.DataFrame    # Quantification matrix (NPX or RFU), shape: samples × features
    metadata: dict              # Platform-specific extras (lod_matrix, count_matrix, ...)
```

| Field | Description |
|-------|-------------|
| `platform` | Enum: `OLINK_EXPLORE`, `OLINK_EXPLORE_HT`, `OLINK_REVEAL`, `OLINK_TARGET`, `SOMASCAN` |
| `samples` | One row per sample. Common columns: `SampleID`, `SampleType`, `SampleQC`, `PlateID` |
| `features` | One row per protein/aptamer. Common columns: `OlinkID`/`SeqId`, `UniProt`, `Assay`/`Target`, `Panel` |
| `expression` | Numeric matrix. Olink stores NPX (log2 scale); SomaScan stores RFU (linear scale). Column names match the feature identifier (`OlinkID` or `SeqId`) |
| `metadata` | Dict with platform-specific data: `lod_matrix`, `count_matrix`, `ext_count`, `pc_median`, etc. |

## Reading Data

```python
import pyprideap as pp

# Auto-detect format from file extension and content
dataset = pp.read("olink_npx.csv")
dataset = pp.read("raw_data.adat")
dataset = pp.read("data.parquet")

# Force platform when auto-detection is ambiguous
dataset = pp.read("ambiguous.csv", platform="olink")
dataset = pp.read("ambiguous.csv", platform="somascan")
```

### Supported Formats

| Format | Platform | Description |
|--------|----------|-------------|
| `.npx.csv` | Olink Explore / Target | Long-format CSV with NPX values |
| `.parquet` | Olink Explore HT | Parquet with NPX values |
| `.xlsx` | Olink | Excel NPX file |
| `.adat` | SomaScan | SomaLogic ADAT format (legacy and TABLE_BEGIN) |
| `.csv` (SeqId columns) | SomaScan | Wide CSV with `SeqId.*` column names |
| `.sdrf.tsv` | Any | SDRF sample metadata (see [SDRF Integration](#sdrf-integration)) |

## SDRF Integration

pyprideap can read [SDRF](https://github.com/bigbio/proteomics-metadata-standard) (Sample and Data Relationship Format) files and merge sample metadata into datasets:

```python
from pyprideap.io.readers.sdrf import read_sdrf, merge_sdrf, get_grouping_columns

# Read and parse an SDRF file
sdrf = read_sdrf("samples.sdrf.tsv")

# Merge SDRF metadata into an existing dataset
dataset = pp.read("olink_npx.csv")
dataset = merge_sdrf(dataset, sdrf)

# Identify columns suitable for differential expression grouping
group_cols = get_grouping_columns(sdrf)
# e.g. ["disease", "sex", "treatment"]
```

Column names are automatically shortened from the full SDRF syntax (e.g. `characteristics[disease]` becomes `disease`). Duplicate column names are disambiguated with numeric suffixes.

## Validation

Check a dataset against PRIDE-AP guidelines:

```python
results = pp.validate(dataset)

for r in results:
    print(f"[{r.level.value}] {r.rule}: {r.message}")
```

Validators check required columns, value ranges, QC consistency, and data completeness. Each result has a `level` (`ERROR`, `WARNING`, `INFO`), a `rule` name, and a `message`.

## Filtering

### Remove control samples

Control samples (NEGATIVE_CONTROL, PLATE_CONTROL, BUFFER, etc.) are not biological and should be removed before analysis:

```python
# Remove all control sample types
ds = pp.filter_controls(dataset)

# Filter by QC status (default keeps PASS and WARN)
ds = pp.filter_qc(dataset)

# Keep only PASS samples
ds = pp.filter_qc(dataset, keep=("PASS",))
```

## LOD (Limit of Detection)

pyprideap supports multiple LOD sources with automatic fallback: Reported > NCLOD > FixedLOD/eLOD.

### LOD Methods

| Method | Platform | Formula | When to Use |
|--------|----------|---------|-------------|
| **Reported** | Olink | From the data file's LOD column | Always preferred when available |
| **NCLOD** | Olink | `median(NC) + max(0.2, 3×SD(NC))` | When ≥10 negative controls are present |
| **FixedLOD** | Olink | Pre-computed by Olink (reagent lot-specific) | When NCLOD is not available |
| **eLOD** | SomaScan | `median(buffer) + 3.3 × 1.4826 × MAD(buffer)` | SomaScan datasets with buffer samples |

### Computing LOD

```python
# Get reported LOD from the data file
lod = pp.get_reported_lod(dataset)

# Compute NCLOD from negative controls (with plate adjustment)
lod = pp.compute_nclod(dataset, plate_adjusted=True)

# Load bundled FixedLOD for a platform
lod = pp.load_fixed_lod(dataset)

# Comprehensive LOD statistics
stats = pp.compute_lod_stats(dataset)
print(f"Source: {stats.lod_source}")
print(f"Above-LOD rate: {stats.above_lod_rate:.1%}")
print(f"Assays with LOD: {stats.n_assays_with_lod}/{stats.n_assays_total}")
```

### Proteins Above LOD

```python
# Get UniProt accessions where ≥50% of samples are above LOD
proteins = pp.get_proteins_above_lod(dataset, threshold=50.0)
print(f"{len(proteins)} proteins above LOD")

# Stricter threshold
proteins_80 = pp.get_proteins_above_lod(dataset, threshold=80.0)
```

## Descriptive Statistics

```python
stats = pp.compute_stats(dataset)
print(stats.summary())

# Access individual fields
print(f"Samples: {stats.n_samples}")
print(f"Features: {stats.n_features}")
print(f"Detection rate: {stats.detection_rate:.1%}")
print(f"Sample types: {stats.sample_types}")
```

## QC Reports

### Single HTML report

```python
# Generate a self-contained interactive HTML report
pp.qc_report(dataset, "my_report.html")
```

The report includes:
- Expression distributions per sample
- QC × LOD stacked bar chart
- PCA and t-SNE projections (with label toggle)
- Sample correlation heatmap
- Clustered expression heatmap
- Data completeness (above/below LOD)
- CV distributions
- Platform-specific QC (normalization scales, RowCheck/ColCheck, etc.)

### Individual plot files

```python
# Split report into separate HTML files per plot
output_dir = pp.qc_report_split(dataset, "qc_plots/")
```

### Programmatic access to QC data

```python
# Get the raw computed data for all QC metrics
qc_data = pp.compute_qc(dataset)

# Access individual metrics
pca = qc_data.get("pca")          # PcaData
dist = qc_data.get("distribution") # DistributionData
```

## Normalization

### Bridge normalization

Combine two datasets that share bridge samples. The per-protein adjustment is: `median(bridge in ds1) − median(bridge in ds2)`.

```python
# Select optimal bridge samples from a dataset
bridges = pp.select_bridge_samples(dataset, n=8)

# Normalize dataset2 to match dataset1's scale
normalized = pp.bridge_normalize(dataset1, dataset2, bridge_samples=bridges)
```

### Subset normalization

Adjust using a reference set of stable proteins:

```python
normalized = pp.subset_normalize(dataset1, dataset2, reference_proteins=["P1", "P2", "P3"])
```

### Reference median normalization

Shift each protein to match pre-recorded reference medians:

```python
normalized = pp.reference_median_normalize(dataset, reference_medians=median_dict)
```

### Bridgeability assessment

Check whether two datasets can be reliably bridged:

```python
report = pp.assess_bridgeability(dataset1, dataset2)
print(report[["protein_id", "correlation", "median_diff", "bridgeable"]])
```

## Statistical Testing

Requires `pip install "pyprideap[stats]"`. All functions run per-protein tests with Benjamini-Hochberg FDR correction.

```python
# Two-sample t-test between groups
results = pp.ttest(dataset, group_var="Treatment")

# Paired t-test
results = pp.ttest(dataset, group_var="Treatment", pair_id="SubjectID")

# Mann-Whitney U test (non-parametric)
results = pp.wilcoxon(dataset, group_var="Treatment")

# One-way ANOVA with covariates
results = pp.anova(dataset, variables=["Treatment"], covariates=["Age", "Sex"])

# Tukey HSD post-hoc comparisons
posthoc = pp.anova_posthoc(dataset, variable="Treatment")
```

All results are returned as a pandas DataFrame with columns: `protein_id`, `assay`, `estimate`, `statistic`, `p_value`, `adj_p_value`, `significant`.

## Experimental Design

```python
# Randomize samples to plates
plate_assignment = pp.randomize_plates(
    samples=sample_df,
    n_plates=4,
    plate_size=88,
    keep_paired="SubjectID",  # keep longitudinal samples on same plate
    seed=42,
)
# Returns DataFrame with PlateNumber and WellPosition columns
```

## Visualization Theme

```python
import plotly.express as px

# Apply PRIDE theme to any Plotly figure
fig = px.scatter(df, x="PC1", y="PC2")
pp.set_plot_theme(fig)

# Use PRIDE color palettes
colors = pp.pride_color_discrete(5)   # 5 distinct colors
gradient = pp.pride_color_gradient(10) # 10-step blue-to-red gradient
```

## Command-Line Interface

```bash
# Generate QC report from a local file
pyprideap report data.npx.csv
pyprideap report data.parquet -o my_report.html

# Force platform type
pyprideap report data.csv -p olink

# Generate individual plot files instead of single report
pyprideap report data.npx.csv --split -o plots_dir/

# Download from PRIDE and generate report
pyprideap report -a PAD000001

# Include SDRF metadata for volcano plots
pyprideap report data.npx.csv --sdrf samples.sdrf.tsv

# Enable verbose logging for debugging
pyprideap report data.npx.csv -v

# List proteins above LOD
pyprideap proteins-above-lod data.npx.csv
pyprideap proteins-above-lod data.npx.csv -t 80 -o proteins.txt
```

### Verbose mode

Use `-v` / `--verbose` to see detailed progress through each processing stage (format detection, LOD computation, PCA variance, plot rendering, etc.).

## Package Structure

```
pyprideap/
├── api/                 # PRIDE Archive REST API client
├── io/
│   ├── readers/         # Format-specific readers (CSV, Parquet, XLSX, ADAT, SDRF)
│   └── validators/      # Data validation against PRIDE-AP guidelines
├── processing/
│   ├── filtering.py     # Sample filtering (controls, QC)
│   ├── lod.py           # LOD computation (NCLOD, FixedLOD, eLOD)
│   ├── normalization.py # Bridge, subset, reference, quantile, lift normalization
│   ├── olink/           # Olink preprocessing pipeline
│   └── somascan/        # SomaScan preprocessing pipeline
├── stats/
│   ├── descriptive.py   # Dataset summary statistics
│   ├── design.py        # Plate randomization
│   └── differential.py  # t-test, Wilcoxon, ANOVA
└── viz/
    ├── theme.py         # Color palettes and plot styling
    ├── plots.py         # Standalone plots (boxplot)
    └── qc/
        ├── compute.py   # QC metric computation
        ├── render.py    # Plotly figure rendering
        └── report.py    # HTML report assembly
```
