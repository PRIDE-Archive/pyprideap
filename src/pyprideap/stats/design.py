"""Plate randomization utilities for Olink experimental design.

Provides functionality equivalent to ``olink_plate_randomizer`` in the
OlinkAnalyze R package, allowing users to randomly assign samples to plates
while optionally keeping paired/longitudinal samples together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Olink 96-well plate layout: 8 rows (A-H) x 12 columns (1-12).
_ROWS = "ABCDEFGH"
_COLS = range(1, 13)
_TOTAL_WELLS = len(_ROWS) * len(_COLS)  # 96

# Default usable wells per plate (96 - 8 control positions).
_DEFAULT_PLATE_SIZE = 88


def _generate_well_positions(n: int) -> list[str]:
    """Return the first *n* well positions in column-major order.

    Positions follow the standard naming convention A1, B1, ... H1, A2, ...
    which mirrors the physical pipetting order on a 96-well plate.
    """
    wells: list[str] = []
    for col in _COLS:
        for row in _ROWS:
            wells.append(f"{row}{col}")
            if len(wells) == n:
                return wells
    return wells


def randomize_plates(
    samples: pd.DataFrame,
    n_plates: int,
    plate_size: int = _DEFAULT_PLATE_SIZE,
    keep_paired: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Randomly assign samples to plates and well positions.

    Parameters
    ----------
    samples:
        DataFrame that must contain a ``SampleID`` (or ``SampleId``) column.
    n_plates:
        Number of plates to distribute samples across.
    plate_size:
        Maximum number of samples per plate.  Defaults to 88 (a 96-well
        Olink plate with 8 positions reserved for controls).
    keep_paired:
        Optional column name whose values define groups that should be kept
        on the same plate (e.g. longitudinal time-points for the same
        subject).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A copy of *samples* with two additional columns: ``PlateNumber``
        (1-based) and ``WellPosition`` (e.g. ``"A1"``).

    Raises
    ------
    ValueError
        If the total number of samples exceeds the available capacity
        (``n_plates * plate_size``), or if a paired group is larger than
        ``plate_size``.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    _resolve_sample_id_column(samples)  # validates that a sample ID column exists

    total_capacity = n_plates * plate_size
    if len(samples) > total_capacity:
        raise ValueError(
            f"Too many samples ({len(samples)}) for the given plate "
            f"configuration ({n_plates} plates x {plate_size} wells = "
            f"{total_capacity} available positions)."
        )

    if plate_size > _TOTAL_WELLS:
        raise ValueError(f"plate_size ({plate_size}) exceeds the number of wells on a 96-well plate ({_TOTAL_WELLS}).")

    rng = np.random.default_rng(seed)
    result = samples.copy()

    # ------------------------------------------------------------------
    # Build an ordered list of indices, optionally grouped
    # ------------------------------------------------------------------
    if keep_paired is not None:
        if keep_paired not in samples.columns:
            raise ValueError(f"Column '{keep_paired}' not found in the samples DataFrame.")
        ordered_indices = _shuffle_paired(samples, keep_paired, plate_size, rng)
    else:
        ordered_indices = _shuffle_unpaired(samples, rng)

    # ------------------------------------------------------------------
    # Distribute indices across plates
    # ------------------------------------------------------------------
    plate_assignments, well_assignments = _assign_plates_and_wells(
        ordered_indices,
        n_plates,
        plate_size,
        keep_paired is not None,
        samples,
        keep_paired,
        rng,
    )

    result.loc[ordered_indices, "PlateNumber"] = plate_assignments
    result.loc[ordered_indices, "WellPosition"] = well_assignments

    result["PlateNumber"] = result["PlateNumber"].astype(int)

    return result  # type: ignore[no-any-return]


# ======================================================================
# Internal helpers
# ======================================================================


def _resolve_sample_id_column(df: pd.DataFrame) -> str:
    """Return the canonical sample-ID column name present in *df*."""
    for candidate in ("SampleID", "SampleId", "sampleid", "sample_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError("The samples DataFrame must contain a 'SampleID' or 'SampleId' column.")


def _shuffle_unpaired(samples: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Return a shuffled array of row indices."""
    indices: np.ndarray = np.asarray(samples.index).copy()
    rng.shuffle(indices)
    return indices


def _shuffle_paired(
    samples: pd.DataFrame,
    paired_col: str,
    plate_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices ordered by shuffled groups.

    Samples that share the same value in *paired_col* are kept contiguous
    so that they end up on the same plate.  The order of groups is random
    and the order within each group is also randomised.
    """
    groups = samples.groupby(paired_col, sort=False).indices  # {val: idx_array}
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    ordered: list[np.ndarray] = []
    for key in group_keys:
        member_indices = np.asarray(groups[key]).copy()
        if len(member_indices) > plate_size:
            raise ValueError(
                f"Paired group '{key}' has {len(member_indices)} samples, "
                f"which exceeds plate_size ({plate_size}). Cannot keep them "
                f"on a single plate."
            )
        rng.shuffle(member_indices)
        ordered.append(member_indices)

    return np.concatenate(ordered)


def _assign_plates_and_wells(
    ordered_indices: np.ndarray,
    n_plates: int,
    plate_size: int,
    is_paired: bool,
    samples: pd.DataFrame | None,
    paired_col: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign plate numbers and well positions to the ordered indices.

    When ``is_paired`` is True, groups (consecutive runs sharing the same
    ``paired_col`` value) are kept on the same plate even if that makes
    the distribution slightly uneven.  When ``is_paired`` is False samples
    are distributed as evenly as possible (round-robin).
    """
    plate_numbers = np.empty(len(ordered_indices), dtype=int)
    well_positions = np.empty(len(ordered_indices), dtype=object)

    if is_paired and paired_col is not None and samples is not None:
        # Assign groups to plates, packing greedily
        plate_counts = [0] * n_plates  # current count per plate
        pos = 0
        while pos < len(ordered_indices):
            # Identify the current group (consecutive indices with same paired value)
            group_val = samples.loc[ordered_indices[pos], paired_col]
            group_start = pos
            while pos < len(ordered_indices) and samples.loc[ordered_indices[pos], paired_col] == group_val:
                pos += 1
            group_size = pos - group_start

            # Find the plate with the fewest samples that still has room
            best_plate = _pick_plate(plate_counts, group_size, plate_size)
            for j in range(group_start, pos):
                plate_numbers[j] = best_plate + 1  # 1-based
                plate_counts[best_plate] += 1
    else:
        # Even round-robin distribution
        # Compute how many samples per plate
        base, extra = divmod(len(ordered_indices), n_plates)
        plate_idx = 0
        count_on_plate = 0
        limit = base + (1 if plate_idx < extra else 0)
        for i in range(len(ordered_indices)):
            plate_numbers[i] = plate_idx + 1  # 1-based
            count_on_plate += 1
            if count_on_plate >= limit:
                plate_idx += 1
                count_on_plate = 0
                if plate_idx < n_plates:
                    limit = base + (1 if plate_idx < extra else 0)

    # Assign well positions within each plate
    for plate_num in range(1, n_plates + 1):
        mask = plate_numbers == plate_num
        n_on_plate = int(mask.sum())
        wells = _generate_well_positions(n_on_plate)
        well_positions[mask] = wells

    return plate_numbers, well_positions


def _pick_plate(plate_counts: list[int], group_size: int, plate_size: int) -> int:
    """Return the index of the plate best suited for a group of *group_size*.

    Prefers the plate with the fewest samples that still has enough room.
    """
    best: int | None = None
    best_count = plate_size + 1
    for idx, count in enumerate(plate_counts):
        if count + group_size <= plate_size and count < best_count:
            best = idx
            best_count = count
    if best is None:
        raise ValueError(
            "Cannot fit all paired groups onto the available plates. Consider increasing n_plates or plate_size."
        )
    return best
