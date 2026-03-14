"""Per-protein differential expression testing.

Implements statistical tests analogous to ``olink_ttest``,
``olink_wilcox``, ``olink_anova`` and ``olink_anova_posthoc`` from the
OlinkAnalyze R package.

All functions operate on an :class:`~pyprideap.core.AffinityDataset` and
return a tidy :class:`pandas.DataFrame` of results with Benjamini-Hochberg
adjusted *p*-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset

# ---------------------------------------------------------------------------
# Lazy imports with user-friendly error messages
# ---------------------------------------------------------------------------


def _import_scipy_stats():
    try:
        from scipy import stats as _stats

        return _stats
    except ImportError:
        raise ImportError("scipy is required for differential expression testing. Install it with:  pip install scipy")


def _import_multipletests():
    try:
        from statsmodels.stats.multitest import multipletests as _mt

        return _mt
    except ImportError:
        raise ImportError("statsmodels is required for p-value adjustment. Install it with:  pip install statsmodels")


def _import_tukeyhsd():
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd as _tk

        return _tk
    except ImportError:
        raise ImportError(
            "statsmodels is required for Tukey HSD post-hoc tests. Install it with:  pip install statsmodels"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_assay_map(dataset: AffinityDataset) -> dict[str, str]:
    """Return a mapping from protein/column id to Assay name.

    If the features DataFrame contains an ``Assay`` column the mapping is
    ``{OlinkID_or_SeqId: Assay}``.  Otherwise an empty dict is returned.
    """
    if "Assay" not in dataset.features.columns:
        return {}
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else "SeqId"
    if id_col not in dataset.features.columns:
        return {}
    return dict(zip(dataset.features[id_col], dataset.features["Assay"]))


def _validate_group_var(
    dataset: AffinityDataset,
    group_var: str,
    *,
    min_levels: int = 2,
    exact_levels: int | None = None,
) -> pd.Series:
    """Return the grouping column after validation.

    Parameters
    ----------
    dataset : AffinityDataset
    group_var : str
        Column name expected in ``dataset.samples``.
    min_levels : int
        Minimum number of unique non-NaN levels required.
    exact_levels : int | None
        If set, the column must have exactly this many levels.

    Returns
    -------
    pd.Series
        The grouping column (NaN rows kept; callers decide how to handle).

    Raises
    ------
    ValueError
        If the column is missing or has wrong number of levels.
    """
    if group_var not in dataset.samples.columns:
        raise ValueError(
            f"group_var '{group_var}' not found in dataset.samples. Available columns: {list(dataset.samples.columns)}"
        )

    groups = dataset.samples[group_var]
    n_levels = groups.dropna().nunique()

    if exact_levels is not None and n_levels != exact_levels:
        raise ValueError(
            f"group_var '{group_var}' must have exactly {exact_levels} "
            f"unique non-NaN levels, but found {n_levels}: "
            f"{sorted(groups.dropna().unique())}"
        )
    if n_levels < min_levels:
        raise ValueError(
            f"group_var '{group_var}' must have at least {min_levels} unique non-NaN levels, but found {n_levels}."
        )

    return groups


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction, handling NaNs."""
    multipletests = _import_multipletests()
    mask = ~np.isnan(p_values)
    adjusted = np.full_like(p_values, np.nan)
    if mask.sum() > 0:
        _, adj, _, _ = multipletests(p_values[mask], method="fdr_bh")
        adjusted[mask] = adj
    return adjusted


# ---------------------------------------------------------------------------
# ttest
# ---------------------------------------------------------------------------


def ttest(
    dataset: AffinityDataset,
    group_var: str,
    pair_id: str | None = None,
) -> pd.DataFrame:
    """Welch two-sample *t*-test (or paired *t*-test) per protein.

    Parameters
    ----------
    dataset : AffinityDataset
        Dataset containing expression data, sample metadata, and features.
    group_var : str
        Column in ``dataset.samples`` with exactly 2 unique non-NaN values.
    pair_id : str | None
        If provided, column in ``dataset.samples`` identifying paired
        observations.  A paired *t*-test is performed instead.

    Returns
    -------
    pd.DataFrame
        One row per protein with columns: ``protein_id``, ``assay``,
        ``estimate`` (fold change = mean_g1 - mean_g2 in log/NPX space),
        ``statistic``, ``p_value``, ``adj_p_value``, ``significant``.
    """
    scipy_stats = _import_scipy_stats()
    groups = _validate_group_var(dataset, group_var, exact_levels=2)
    assay_map = _resolve_assay_map(dataset)

    levels = sorted(groups.dropna().unique())
    g1_label, g2_label = levels

    mask_g1 = groups == g1_label
    mask_g2 = groups == g2_label

    records: list[dict] = []

    for protein_id in dataset.expression.columns:
        vals = dataset.expression[protein_id]

        if pair_id is not None:
            # Paired t-test: align by pair_id
            if pair_id not in dataset.samples.columns:
                raise ValueError(f"pair_id '{pair_id}' not found in dataset.samples.")
            df_tmp = pd.DataFrame(
                {
                    "val": vals,
                    "group": groups,
                    "pair": dataset.samples[pair_id],
                }
            ).dropna(subset=["val", "group", "pair"])

            pivot = df_tmp.pivot(index="pair", columns="group", values="val")
            pivot = pivot.dropna()

            if len(pivot) < 2:
                records.append(_empty_ttest_row(protein_id, assay_map))
                continue

            a = pivot[g1_label].values
            b = pivot[g2_label].values
            stat, pval = scipy_stats.ttest_rel(a, b)
            estimate = float(np.nanmean(a) - np.nanmean(b))
        else:
            # Unpaired Welch t-test
            a = vals[mask_g1].dropna().values
            b = vals[mask_g2].dropna().values

            if len(a) < 2 or len(b) < 2:
                records.append(_empty_ttest_row(protein_id, assay_map))
                continue

            stat, pval = scipy_stats.ttest_ind(a, b, equal_var=False)
            estimate = float(np.nanmean(a) - np.nanmean(b))

        records.append(
            {
                "protein_id": protein_id,
                "assay": assay_map.get(protein_id, None),
                "estimate": estimate,
                "statistic": float(stat),
                "p_value": float(pval),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return _empty_ttest_frame()

    result["adj_p_value"] = _bh_adjust(np.asarray(result["p_value"]))
    result["significant"] = result["adj_p_value"] < 0.05
    return result


def _empty_ttest_row(protein_id: str, assay_map: dict) -> dict:
    return {
        "protein_id": protein_id,
        "assay": assay_map.get(protein_id, None),
        "estimate": np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
    }


def _empty_ttest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protein_id",
            "assay",
            "estimate",
            "statistic",
            "p_value",
            "adj_p_value",
            "significant",
        ]
    )


# ---------------------------------------------------------------------------
# wilcoxon
# ---------------------------------------------------------------------------


def wilcoxon(
    dataset: AffinityDataset,
    group_var: str,
    pair_id: str | None = None,
) -> pd.DataFrame:
    """Mann-Whitney U test (or Wilcoxon signed-rank) per protein.

    Parameters
    ----------
    dataset : AffinityDataset
        Dataset containing expression data, sample metadata, and features.
    group_var : str
        Column in ``dataset.samples`` with exactly 2 unique non-NaN values.
    pair_id : str | None
        If provided, a paired Wilcoxon signed-rank test is used.

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`ttest`.
    """
    scipy_stats = _import_scipy_stats()
    groups = _validate_group_var(dataset, group_var, exact_levels=2)
    assay_map = _resolve_assay_map(dataset)

    levels = sorted(groups.dropna().unique())
    g1_label, g2_label = levels

    mask_g1 = groups == g1_label
    mask_g2 = groups == g2_label

    records: list[dict] = []

    for protein_id in dataset.expression.columns:
        vals = dataset.expression[protein_id]

        if pair_id is not None:
            if pair_id not in dataset.samples.columns:
                raise ValueError(f"pair_id '{pair_id}' not found in dataset.samples.")
            df_tmp = pd.DataFrame(
                {
                    "val": vals,
                    "group": groups,
                    "pair": dataset.samples[pair_id],
                }
            ).dropna(subset=["val", "group", "pair"])

            pivot = df_tmp.pivot(index="pair", columns="group", values="val")
            pivot = pivot.dropna()

            if len(pivot) < 2:
                records.append(_empty_ttest_row(protein_id, assay_map))
                continue

            a = pivot[g1_label].values
            b = pivot[g2_label].values
            try:
                stat, pval = scipy_stats.wilcoxon(a, b)
            except ValueError:
                # wilcoxon raises if all differences are zero
                records.append(_empty_ttest_row(protein_id, assay_map))
                continue
            estimate = float(np.nanmean(a) - np.nanmean(b))
        else:
            a = vals[mask_g1].dropna().values
            b = vals[mask_g2].dropna().values

            if len(a) < 1 or len(b) < 1:
                records.append(_empty_ttest_row(protein_id, assay_map))
                continue

            stat, pval = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            estimate = float(np.nanmean(a) - np.nanmean(b))

        records.append(
            {
                "protein_id": protein_id,
                "assay": assay_map.get(protein_id, None),
                "estimate": estimate,
                "statistic": float(stat),
                "p_value": float(pval),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return _empty_ttest_frame()

    result["adj_p_value"] = _bh_adjust(np.asarray(result["p_value"]))
    result["significant"] = result["adj_p_value"] < 0.05
    return result


# ---------------------------------------------------------------------------
# anova
# ---------------------------------------------------------------------------


def anova(
    dataset: AffinityDataset,
    group_var: str,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Per-protein one-way ANOVA (or ANCOVA with covariates).

    Parameters
    ----------
    dataset : AffinityDataset
        Dataset containing expression data, sample metadata, and features.
    group_var : str
        Column in ``dataset.samples`` with 2 or more unique non-NaN values.
    covariates : list[str] | None
        Additional columns in ``dataset.samples`` to include as covariates.
        When provided, an OLS-based Type-II ANOVA is used instead of
        ``scipy.stats.f_oneway``.

    Returns
    -------
    pd.DataFrame
        One row per protein with columns: ``protein_id``, ``assay``,
        ``statistic``, ``df_between``, ``df_within``, ``p_value``,
        ``adj_p_value``, ``significant``.
    """
    scipy_stats = _import_scipy_stats()
    groups = _validate_group_var(dataset, group_var, min_levels=2)
    assay_map = _resolve_assay_map(dataset)

    levels = sorted(groups.dropna().unique())
    n_levels = len(levels)

    use_ols = covariates is not None and len(covariates) > 0

    if use_ols:
        assert covariates is not None  # narrowing for mypy
        # Validate covariates exist
        for cov in covariates:
            if cov not in dataset.samples.columns:
                raise ValueError(f"Covariate '{cov}' not found in dataset.samples.")

    records: list[dict] = []

    for protein_id in dataset.expression.columns:
        vals = dataset.expression[protein_id]

        if use_ols:
            assert covariates is not None  # narrowing for mypy
            stat, pval, df_b, df_w = _anova_ols(vals, groups, dataset.samples, group_var, covariates)
        else:
            # Simple one-way ANOVA via scipy
            group_arrays = []
            for lev in levels:
                arr = vals[groups == lev].dropna().values
                group_arrays.append(arr)

            # Need at least 2 observations in each group
            if any(len(arr) < 1 for arr in group_arrays):
                records.append(_empty_anova_row(protein_id, assay_map))
                continue

            total_n = sum(len(arr) for arr in group_arrays)
            df_b = n_levels - 1
            df_w = total_n - n_levels

            if df_w < 1:
                records.append(_empty_anova_row(protein_id, assay_map))
                continue

            stat, pval = scipy_stats.f_oneway(*group_arrays)

        if np.isnan(pval):
            records.append(_empty_anova_row(protein_id, assay_map))
            continue

        records.append(
            {
                "protein_id": protein_id,
                "assay": assay_map.get(protein_id, None),
                "statistic": float(stat),
                "df_between": int(df_b),
                "df_within": int(df_w),
                "p_value": float(pval),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return _empty_anova_frame()

    result["adj_p_value"] = _bh_adjust(np.asarray(result["p_value"]))
    result["significant"] = result["adj_p_value"] < 0.05
    return result


def _anova_ols(
    vals: pd.Series,
    groups: pd.Series,
    samples: pd.DataFrame,
    group_var: str,
    covariates: list[str],
) -> tuple[float, float, int, int]:
    """Run OLS-based Type-II ANOVA for a single protein (ANCOVA)."""
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        raise ImportError("statsmodels is required for ANCOVA. Install it with:  pip install statsmodels")

    df = pd.DataFrame(
        {
            "y": vals,
            group_var: groups,
        }
    )
    for cov in covariates:
        df[cov] = samples[cov].values

    df = df.dropna()
    if len(df) < 3:
        return np.nan, np.nan, 0, 0

    cov_terms = " + ".join(covariates)
    formula = f"y ~ C({group_var}) + {cov_terms}"

    try:
        model = ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)
    except Exception:
        return np.nan, np.nan, 0, 0

    row_key = f"C({group_var})"
    if row_key not in table.index:
        return np.nan, np.nan, 0, 0

    stat = table.loc[row_key, "F"]
    pval = table.loc[row_key, "PR(>F)"]
    df_b = int(table.loc[row_key, "df"])
    df_w = int(table.loc["Residual", "df"])

    return stat, pval, df_b, df_w


def _empty_anova_row(protein_id: str, assay_map: dict) -> dict:
    return {
        "protein_id": protein_id,
        "assay": assay_map.get(protein_id, None),
        "statistic": np.nan,
        "df_between": np.nan,
        "df_within": np.nan,
        "p_value": np.nan,
    }


def _empty_anova_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protein_id",
            "assay",
            "statistic",
            "df_between",
            "df_within",
            "p_value",
            "adj_p_value",
            "significant",
        ]
    )


# ---------------------------------------------------------------------------
# anova_posthoc (Tukey HSD)
# ---------------------------------------------------------------------------


def anova_posthoc(
    dataset: AffinityDataset,
    group_var: str,
    proteins: list[str] | None = None,
) -> pd.DataFrame:
    """Tukey HSD post-hoc pairwise comparisons per protein.

    Parameters
    ----------
    dataset : AffinityDataset
        Dataset containing expression data, sample metadata, and features.
    group_var : str
        Column in ``dataset.samples`` with 2 or more unique non-NaN values.
    proteins : list[str] | None
        Subset of protein IDs (expression column names) to test.  If *None*,
        all proteins are tested.

    Returns
    -------
    pd.DataFrame
        One row per protein-contrast pair with columns: ``protein_id``,
        ``contrast``, ``estimate``, ``p_value``, ``adj_p_value``,
        ``ci_lower``, ``ci_upper``.
    """
    pairwise_tukeyhsd = _import_tukeyhsd()
    groups = _validate_group_var(dataset, group_var, min_levels=2)
    assay_map = _resolve_assay_map(dataset)

    protein_ids = list(proteins) if proteins is not None else list(dataset.expression.columns)

    records: list[dict] = []

    for protein_id in protein_ids:
        if protein_id not in dataset.expression.columns:
            continue

        vals = dataset.expression[protein_id]

        df_tmp = pd.DataFrame(
            {
                "val": vals,
                "group": groups,
            }
        ).dropna()

        if df_tmp["group"].nunique() < 2 or len(df_tmp) < 3:
            continue

        try:
            result = pairwise_tukeyhsd(
                df_tmp["val"].values,
                df_tmp["group"].values,
                alpha=0.05,
            )
        except Exception:
            continue

        # Extract pairwise results from the TukeyHSDResults object
        for i in range(len(result.groupsunique)):
            for j in range(i + 1, len(result.groupsunique)):
                g_a = result.groupsunique[i]
                g_b = result.groupsunique[j]

                # Find the row index in the summary data for this pair
                idx = _tukey_pair_index(result, i, j)
                if idx is None:
                    continue

                meandiff = float(result.meandiffs[idx])
                pval = float(result.pvalues[idx])
                ci_low = float(result.confint[idx, 0])
                ci_high = float(result.confint[idx, 1])

                records.append(
                    {
                        "protein_id": protein_id,
                        "assay": assay_map.get(protein_id, None),
                        "contrast": f"{g_a} - {g_b}",
                        "estimate": meandiff,
                        "p_value": pval,
                        "ci_lower": ci_low,
                        "ci_upper": ci_high,
                    }
                )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return pd.DataFrame(
            columns=[
                "protein_id",
                "assay",
                "contrast",
                "estimate",
                "p_value",
                "adj_p_value",
                "ci_lower",
                "ci_upper",
            ]
        )

    result_df["adj_p_value"] = _bh_adjust(np.asarray(result_df["p_value"]))
    return result_df


def _tukey_pair_index(result, i: int, j: int) -> int | None:
    """Map group indices (i, j) to the row index in TukeyHSDResults.

    The ``pairwise_tukeyhsd`` result stores comparisons in a flat array
    corresponding to the upper triangle of the group matrix.
    """
    n = len(result.groupsunique)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            if a == i and b == j:
                return idx
            idx += 1
    return None
