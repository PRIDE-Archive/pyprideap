# pyprideap

Python PRIDE Affinity Proteomics (pyprideap), a library for reading, validating, and analyzing affinity proteomics datasets from the [PRIDE Affinity Archive (PAD)](https://www.ebi.ac.uk/pride/).

Supports **Olink** (Explore, Explore HT, Target, Reveal) and **SomaScan** platforms.

## Installation

> **Note:** pyprideap is not yet published on PyPI. Install directly from source:
>
> ```bash
> pip install git+https://github.com/PRIDE-Archive/pyprideap.git
> ```

Once available on PyPI:

```bash
pip install pyprideap
```

With plotting and QC report support:

```bash
pip install "pyprideap[plots]"
```

With statistical testing:

```bash
pip install "pyprideap[all]"
```

For development:

```bash
git clone https://github.com/PRIDE-Archive/pyprideap.git
cd pyprideap
pip install -e ".[dev]"
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

pyprideap includes a CLI for generating QC reports:

```bash
# From a local file (format auto-detected)
pyprideap report data.npx.csv
pyprideap report data.parquet -o my_report.html

# Force platform type
pyprideap report ambiguous.csv -p olink
pyprideap report data.adat -p somascan

# From a PRIDE accession (downloads data automatically)
pyprideap report PAD000001
```

Or via `python -m`:

```bash
python -m pyprideap report data.npx.csv
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

## Supported File Formats

| Format | Platform | Function |
|--------|----------|----------|
| `.npx.csv` | Olink Explore / Target | `pp.read()` |
| `.parquet` | Olink Explore HT | `pp.read()` |
| `.xlsx` | Olink | `pp.read()` |
| `.adat` | SomaScan | `pp.read()` |
| `.csv` (SomaScan) | SomaScan | `pp.read()` |

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
results = pp.anova(dataset, variables=["Treatment"], covariates=["Age", "Sex"])

# Post-hoc pairwise comparisons
posthoc = pp.anova_posthoc(dataset, variable="Treatment")
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

## Package Structure

```
pyprideap/
├── api/             # PRIDE Archive REST API client
├── io/
│   ├── readers/     # Format-specific readers (CSV, Parquet, XLSX, ADAT)
│   └── validators/  # Data validation against PRIDE-AP guidelines
├── processing/
│   ├── filtering.py     # Sample filtering (controls, QC)
│   ├── lod.py           # LOD computation (NCLOD, FixedLOD, eLOD)
│   └── normalization.py # Bridge, subset, reference normalization
├── stats/
│   ├── descriptive.py       # Dataset summary statistics
│   ├── design.py            # Plate randomization
│   └── differential.py      # t-test, Wilcoxon, ANOVA
└── viz/
    ├── theme.py     # Color palettes and plot styling
    ├── plots.py     # Standalone plots (boxplot)
    └── qc/
        ├── compute.py   # QC metric computation
        ├── render.py    # Plotly figure rendering
        └── report.py    # HTML report assembly
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/pyprideap/
```

## License

Apache License 2.0
