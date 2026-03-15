# pyprideap

[![PyPI version](https://img.shields.io/pypi/v/pyprideap.svg)](https://pypi.org/project/pyprideap/)
[![Python](https://img.shields.io/pypi/pyversions/pyprideap.svg)](https://pypi.org/project/pyprideap/)
[![CI](https://github.com/PRIDE-Archive/pyprideap/actions/workflows/ci.yml/badge.svg)](https://github.com/PRIDE-Archive/pyprideap/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/PRIDE-Archive/pyprideap.svg)](https://github.com/PRIDE-Archive/pyprideap/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pyprideap.svg)](https://pypi.org/project/pyprideap/)

Python PRIDE Affinity Proteomics (pyprideap), a library for reading, validating, and analyzing affinity proteomics datasets from the [PRIDE Affinity Archive (PAD)](https://www.ebi.ac.uk/pride/).

Supports **Olink** (Explore, Explore HT, Target, Reveal) and **SomaScan** platforms.

## Installation

Install pyprideap directly from PyPI:

```bash
pip install pyprideap
```

Or from source:
```bash
pip install "pyprideap[all] @ git+https://github.com/PRIDE-Archive/pyprideap.git"
```

With plotting and QC report support:

```bash
pip install "pyprideap[plots]"
```

With statistical testing:

```bash
pip install "pyprideap[all]"
```

## Quick Start

### Read a dataset

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

### Generate a QC report

```python
dataset = pp.read("olink_npx.csv")
pp.qc_report(dataset, "my_report.html")
```

The report includes interactive plots: expression distributions, PCA/t-SNE, LOD analysis, sample correlation, data completeness, CV distributions, and more. All plots are rendered with Plotly and include help tooltips explaining how to interpret each visualization.

### Validate against PRIDE-AP guidelines

```python
results = pp.validate(dataset)

for r in results:
    print(f"[{r.level.value}] {r.rule}: {r.message}")
```

### Compute statistics

```python
stats = pp.compute_stats(dataset)
print(stats.summary())
```

### Fetch data from PRIDE Archive

```python
client = pp.PrideClient()
project = client.get_project("PAD000001")
files = client.list_files("PAD000001")
urls = client.get_download_urls("PAD000001")
```

## Command-Line Interface

pyprideap includes a CLI (powered by [Click](https://click.palletsprojects.com/)) for generating QC reports:

```bash
# From a local file (format auto-detected)
pyprideap report data.npx.csv
pyprideap report data.parquet -o my_report.html

# Force platform type
pyprideap report data.csv -p olink
pyprideap report data.adat -p somascan

# From a PRIDE accession (downloads data automatically)
pyprideap report -a PAD000001

# Generate individual plot files instead of a single report
pyprideap report data.npx.csv --split -o plots_dir/

# Include SDRF metadata for volcano plots
pyprideap report data.npx.csv --sdrf samples.sdrf.tsv

# Enable verbose logging (shows format detection, LOD method, PCA variance, etc.)
pyprideap report data.npx.csv -v

# List proteins above LOD
pyprideap proteins-above-lod data.npx.csv
pyprideap proteins-above-lod data.npx.csv -t 80 -o proteins.txt
```

Or via `python -m`:

```bash
python -m pyprideap report data.npx.csv
```

### Verbose mode

Use `-v` / `--verbose` to enable detailed debug logging. This shows progress through each processing stage:

```
Reading olink_npx.csv...
08:12:01 [DEBUG] pyprideap.io.readers.registry: Format detected: olink_csv
08:12:01 [DEBUG] pyprideap.io.readers.olink_csv: Sample key selected: SampleID
08:12:01 [DEBUG] pyprideap.io.readers.olink_csv: Pivot shape: 20 samples x 1470 features
  20 samples, 1470 features (olink_explore)
08:12:01 [DEBUG] pyprideap.processing.lod: LOD method selected: REPORTED
08:12:02 [DEBUG] pyprideap.viz.qc.compute: Computing PCA...
08:12:02 [DEBUG] pyprideap.viz.qc.compute: PCA: variance explained=[0.42, 0.18]
...
```

## QC Report

The HTML report is a self-contained, interactive document with a sidebar table of contents. It includes:

| Section | Plots |
|---------|-------|
| **Quality Overview** | LOD source comparison, QC x LOD stacked bar |
| **Signal & Distribution** | Per-sample expression histograms, protein detectability |
| **Data Completeness** | Per-sample above/below LOD, missing frequency distribution |
| **Sample Relationships** | PCA / t-SNE (dropdown toggle), sample correlation heatmap, clustered expression heatmap |
| **Normalization QC** | Hybridization control scale (SomaScan) |
| **Variability** | CV distribution, intra/inter-plate CV |

Each plot has a **?** help button with guidance on interpretation.

### Embedding reports in web pages

Reports automatically detect when loaded inside an `<iframe>` and switch to an embedded mode that hides the header, sidebar, and footer:

```html
<iframe
  src="my_report.html"
  style="width: 100%; border: none; min-height: 600px;"
  id="qc-report">
</iframe>

<script>
// Auto-resize iframe to fit content
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'pride-qc-resize') {
    document.getElementById('qc-report').style.height = e.data.height + 'px';
  }
});
</script>
```

The embedded report posts `pride-qc-resize` messages with the document height, allowing the parent page to resize the iframe automatically. The CSS class `pride-embedded` is added to the body, which:

- Removes the sidebar navigation, header, and footer
- Makes the background transparent
- Removes card shadows for a seamless look

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

## Supported File Formats

| Format | Platform | Function |
|--------|----------|----------|
| `.npx.csv` | Olink Explore / Target | `pp.read()` |
| `.parquet` | Olink Explore HT | `pp.read()` |
| `.xlsx` | Olink | `pp.read()` |
| `.adat` | SomaScan | `pp.read()` |
| `.csv` (SomaScan) | SomaScan | `pp.read()` |
| `.sdrf.tsv` | Any | `read_sdrf()` |

All readers produce an `AffinityDataset` with a unified structure regardless of input format.

## Data Model

```python
@dataclass
class AffinityDataset:
    platform: Platform          # OLINK_EXPLORE, OLINK_EXPLORE_HT, SOMASCAN, etc.
    samples: pd.DataFrame       # Sample metadata (SampleID, SampleType, QC flags, ...)
    features: pd.DataFrame      # Protein/aptamer annotations (OlinkID, UniProt, Panel, ...)
    expression: pd.DataFrame    # Quantification matrix (NPX or RFU)
    metadata: dict              # Platform-specific extras
```

## LOD (Limit of Detection)

pyprideap supports multiple LOD sources with automatic fallback:

1. **Reported LOD** — from the LOD column in the data file
2. **NCLOD** — computed from negative control samples (requires >= 10 controls)
3. **FixedLOD** — pre-computed Olink reference values (bundled for Explore, Explore HT, Reveal)
4. **eLOD** — estimated from buffer samples using MAD formula (SomaScan)

## Statistical Analysis

With `pip install "pyprideap[stats]"`:

```python
# Per-protein t-test between groups
results = pp.ttest(dataset, group_var="Treatment")

# Wilcoxon rank-sum test
results = pp.wilcoxon(dataset, group_var="Treatment")

# ANOVA with covariates
results = pp.anova(dataset, group_var="Treatment", covariates=["Age", "Sex"])

# Post-hoc pairwise comparisons
posthoc = pp.anova_posthoc(dataset, group_var="Treatment")
```

## Normalization

```python
# Bridge normalization (combining two runs with shared samples)
normalized = pp.bridge_normalize(dataset1, dataset2, bridge_samples=["S1", "S2"])

# Subset normalization using reference proteins
normalized = pp.subset_normalize(dataset1, dataset2, reference_proteins=["P1", "P2"])

# Reference median normalization
normalized = pp.reference_median_normalize(dataset, reference_medians=medians)

# Select optimal bridge samples
bridges = pp.select_bridge_samples(dataset, n=8)

# Assess bridgeability between product versions
report = pp.assess_bridgeability(dataset1, dataset2)
```

Additional normalization methods are available via direct import:

```python
from pyprideap.processing.normalization import (
    lift_somascan,                # Cross-version SomaScan calibration (5k ↔ 7k ↔ 11k)
    quantile_smooth_normalize,    # Quantile normalization with smoothing
    scale_analytes,               # Per-analyte multiplicative scaling
    normalize_n,                  # Multi-step normalization pipeline
)
```

## Preprocessing Pipelines

Platform-specific preprocessing pipelines bundle common QC and filtering steps:

```python
from pyprideap.processing.olink import preprocess_olink
from pyprideap.processing.somascan import preprocess_somascan

# Olink: filter controls, detect outliers, LOD filtering, UniProt dedup
dataset, report = preprocess_olink(
    dataset,
    filter_controls=True,
    filter_qc_outliers=True,
    filter_lod=False,
)

# SomaScan: filter features/controls, RowCheck QC, outlier detection
dataset, report = preprocess_somascan(
    dataset,
    filter_features=True,
    filter_controls=True,
    filter_rowcheck=True,
)

print(report.summary())
```

## Experimental Design

```python
# Randomize samples to plates
plate_assignment = pp.randomize_plates(
    samples=sample_df,
    n_plates=4,
    keep_paired="SubjectID",  # keep longitudinal samples on same plate
    seed=42,
)
```

## License

Apache License 2.0
