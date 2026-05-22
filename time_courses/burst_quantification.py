#!/usr/bin/env python3
"""
Burst quantification from an intensity trajectory matrix.

Quantifies burst-like intensity dynamics from a matrix of single-trajectory
fluorescence time courses.  Rows = trajectories, columns = time points.

Usage (notebook)::

    from burst_quantification import run_burst_quantification
    result = run_burst_quantification(
        input_matrix=my_matrix,
        output_dir=Path("results/burst_quantification/4sf"),
        time_interval_seconds=5.0,
    )

Usage (CLI)::

    python burst_quantification.py \\
        --input raw_matrix.npy --output results/burst_quantification/run01 \\
        --dt 5 --threshold 0.05
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter1d

try:
    from plotting import (
        CHANNEL_GREEN,
        CHANNEL_MAGENTA,
        TRACE_BLUE,
        TRACE_GRAY,
        TRACE_MAGENTA,
        save_figure,
        set_publication_style,
        style_axes,
        style_legend,
    )
except ImportError:  # pragma: no cover - supports package-style imports
    from time_courses.plotting import (
        CHANNEL_GREEN,
        CHANNEL_MAGENTA,
        TRACE_BLUE,
        TRACE_GRAY,
        TRACE_MAGENTA,
        save_figure,
        set_publication_style,
        style_axes,
        style_legend,
    )

# ---------------------------------------------------------------------------
# MicroLive integration (optional – graceful fallback if unavailable)
# ---------------------------------------------------------------------------
try:
    import os as _os
    import sys as _sys
    _MICROLIVE_ROOT = Path(
        _os.environ.get("MICROLIVE_ROOT", "/Users/nzlab-la/Desktop/microlive")
    )
    if str(_MICROLIVE_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_MICROLIVE_ROOT))
    from microlive import microscopy as mi
    _HAS_MICROLIVE = True
except (ImportError, ModuleNotFoundError):
    _HAS_MICROLIVE = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_intensity_matrix(path, npz_key="intensity"):
    """Load a 2-D intensity matrix from CSV, TSV, NPY or NPZ.

    Returns
    -------
    matrix : ndarray, shape (n_trajectories, n_timepoints)
    trajectory_ids : list[str] or None
    metadata : dict
    """
    path = Path(path)
    metadata: dict = {}

    # Try loading sidecar metadata
    sidecar = path.with_name(f"{path.stem}_metadata.json")
    if sidecar.exists():
        with open(sidecar) as f:
            metadata = json.load(f)

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, header=None)
        matrix = df.values.astype(float)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", header=None)
        matrix = df.values.astype(float)
    elif ext == ".npy":
        matrix = np.load(path).astype(float)
    elif ext == ".npz":
        data = np.load(path)
        if npz_key in data:
            matrix = data[npz_key].astype(float)
        else:
            # Fall back to first array
            matrix = data[list(data.keys())[0]].astype(float)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    trajectory_ids = metadata.get("trajectory_ids", None)
    return matrix, trajectory_ids, metadata


# ═══════════════════════════════════════════════════════════════════════════
# 2. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_intensity_matrix(matrix, trajectory_ids=None):
    """Coerce to float 2-D array, validate shape, remove fully empty rows.

    Returns
    -------
    matrix : ndarray
    trajectory_ids : list[str]
    valid_rows : ndarray of bool
        Row mask applied (True = kept).  Returned so companion matrices
        (e.g. an SNR matrix) can be filtered identically.
    col_slice : slice
        Column slice applied to trim empty leading/trailing columns.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2-D, got {matrix.ndim}-D")

    # Warn if possibly transposed
    n_rows, n_cols = matrix.shape
    if n_rows > 10 * n_cols:
        print(f"  WARNING: matrix has {n_rows} rows but only {n_cols} cols — "
              "check if it should be transposed.")

    # Drop fully empty rows
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows]

    # Drop fully empty leading/trailing columns
    col_has_data = np.any(np.isfinite(matrix), axis=0)
    if np.any(col_has_data):
        first = np.argmax(col_has_data)
        last = len(col_has_data) - np.argmax(col_has_data[::-1])
        col_slice = slice(first, last)
        matrix = matrix[:, col_slice]
    else:
        col_slice = slice(None)

    # Auto-generate trajectory IDs if needed
    if trajectory_ids is None:
        trajectory_ids = [f"traj_{i}" for i in range(matrix.shape[0])]
    else:
        trajectory_ids = [trajectory_ids[i] for i in np.where(valid_rows)[0]]

    return matrix, trajectory_ids, valid_rows, col_slice


# ═══════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING & QC
# ═══════════════════════════════════════════════════════════════════════════

def _count_max_internal_nan_gap(row):
    """Longest consecutive NaN run between first and last finite value."""
    valid = np.where(np.isfinite(row))[0]
    if valid.size < 2:
        return 0
    interior = row[valid[0]:valid[-1] + 1]
    if not np.any(np.isnan(interior)):
        return 0
    nan_mask = np.isnan(interior)
    changes = np.diff(nan_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if nan_mask[0]:
        starts = np.concatenate([[0], starts])
    if nan_mask[-1]:
        ends = np.concatenate([ends, [len(interior)]])
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def preprocess_intensity_matrix(
    matrix,
    trajectory_ids=None,
    min_valid_fraction=0.30,
    max_internal_nan_gap=2,
    align_first_valid=False,
    detrend_method=None,
    smooth_method="median",
    smooth_window=3,
    baseline_percentile=None,
    flat_trace_threshold=1e-6,
):
    """Return processed matrix plus row-level QC table.

    Returns
    -------
    processed : ndarray
    qc_table : DataFrame
    kept_ids : list[str]
    """
    n_rows, n_cols = matrix.shape
    if trajectory_ids is None:
        trajectory_ids = [f"traj_{i}" for i in range(n_rows)]

    qc_records = []
    keep_mask = np.ones(n_rows, dtype=bool)

    for i in range(n_rows):
        row = matrix[i]
        n_valid = int(np.sum(np.isfinite(row)))
        vf = n_valid / n_cols
        max_gap = _count_max_internal_nan_gap(row)
        finite_vals = row[np.isfinite(row)]
        dyn_range = float(np.ptp(finite_vals)) if finite_vals.size > 0 else 0.0

        status, reason = "kept", "ok"
        if n_valid == 0:
            status, reason = "removed", "empty"
            keep_mask[i] = False
        elif vf < min_valid_fraction:
            status, reason = "removed", "too_sparse"
            keep_mask[i] = False
        elif max_gap > max_internal_nan_gap:
            status, reason = "removed", "too_many_internal_nans"
            keep_mask[i] = False
        elif dyn_range < flat_trace_threshold:
            # Constant-high traces are kept as "constitutive"; only truly
            # zero/empty flat traces are removed.
            mean_val = float(np.nanmean(finite_vals)) if finite_vals.size > 0 else 0.0
            if mean_val > flat_trace_threshold:
                status, reason = "kept", "constitutive"
            else:
                status, reason = "removed", "flat_zero"
                keep_mask[i] = False

        qc_records.append({
            "trajectory_id": trajectory_ids[i],
            "trajectory_index": i,
            "n_valid": n_valid,
            "valid_fraction": round(vf, 4),
            "max_internal_nan_gap": max_gap,
            "dynamic_range": round(dyn_range, 4),
            "qc_status": status,
            "qc_reason": reason,
        })

    qc_table = pd.DataFrame(qc_records)
    processed = matrix[keep_mask].copy()
    kept_ids = [trajectory_ids[i] for i in range(n_rows) if keep_mask[i]]

    if processed.shape[0] == 0:
        return processed, qc_table, kept_ids

    # Optional: align first valid frame.
    # NOTE: MicroLive's shift_trajectories applies its own row filter using
    # *total* internal NaNs (not max consecutive gap as our per-row QC does),
    # so it can drop additional rows. When that happens we re-sync qc_table
    # and kept_ids so downstream code (notably `raw_kept` in
    # `run_burst_quantification`) doesn't end up larger than `processed`.
    if align_first_valid and _HAS_MICROLIVE:
        try:
            kept_indices_pre_shift = np.where(keep_mask)[0]
            n_before = processed.shape[0]
            processed = mi.Utilities().shift_trajectories(
                processed,
                min_percentage_data_in_trajectory=min_valid_fraction,
                # Stage 1 already applied the total-gap filter via
                # max_total_internal_nan_frames; skip it here so the
                # consecutive-gap parameter (max_internal_nan_gap) is not
                # misinterpreted as a total count.
                max_missing_frames=None,
            )
            n_after = processed.shape[0]
            if n_after < n_before:
                # Identify which pre-shift rows survived by replicating
                # shift_trajectories' own filter criteria on the pre-shift
                # matrix. Two-row drops can happen if a trace has many
                # scattered single-frame gaps that sum past the threshold.
                pre_shift = matrix[keep_mask]
                n_time = pre_shift.shape[1]
                max_nans_allowed = int(round(n_time * (1 - min_valid_fraction)))
                row_nan_counts = np.isnan(pre_shift).sum(axis=1)
                # Only fraction filter applies in stage 2 (max_missing_frames=None)
                shift_mask = row_nan_counts <= max_nans_allowed
                # Sanity: surviving count should match the shifted matrix
                if int(shift_mask.sum()) != n_after:
                    print(f"  WARNING: shift survival mask ({int(shift_mask.sum())}) "
                          f"disagrees with shifted matrix rows ({n_after}); "
                          f"qc_table may be slightly inaccurate.")
                # Update qc_table for the rows that shift dropped
                for local_j, survived in enumerate(shift_mask):
                    if not survived:
                        orig_i = int(kept_indices_pre_shift[local_j])
                        keep_mask[orig_i] = False
                        qc_table.loc[
                            qc_table["trajectory_index"] == orig_i,
                            ["qc_status", "qc_reason"]
                        ] = ["removed", "shift_alignment_filter"]
                # Rebuild kept_ids to match the post-shift matrix
                kept_ids = [trajectory_ids[i] for i in range(n_rows) if keep_mask[i]]
        except Exception as e:
            print(f"  WARNING: shift_trajectories failed ({e}), skipping alignment")

    # Optional: detrend
    if detrend_method is not None and _HAS_MICROLIVE:
        try:
            processed = mi.Utilities().detrend_trajectories(
                processed, method=detrend_method
            )
        except Exception as e:
            print(f"  WARNING: detrending failed ({e}), skipping")

    # Build valid_mask from the PROCESSED matrix after shift/detrend so the
    # shape matches. Records which frames had real measurements vs NaN
    # before any forward-fill, so smoothed-over-fills are re-stamped below.
    valid_mask = np.isfinite(processed)

    # Smoothing
    if smooth_method != "none" and smooth_window > 1:
        # Forward-fill NaNs for smoothing only
        if _HAS_MICROLIVE:
            filled = mi.Utilities().forward_fill_nan_2d(processed)
        else:
            filled = processed.copy()
            for i in range(filled.shape[0]):
                row = filled[i]
                mask = np.isnan(row)
                if np.any(~mask):
                    idx = np.where(~mask, np.arange(len(row)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    filled[i] = row[idx]

        if smooth_method == "median":
            processed = median_filter(filled, size=(1, smooth_window))
        elif smooth_method == "mean":
            processed = uniform_filter1d(filled, size=smooth_window, axis=1,
                                         mode="nearest")
        elif smooth_method == "gaussian":
            processed = gaussian_filter1d(filled, sigma=smooth_window / 2.0,
                                          axis=1, mode="nearest")

    # Re-stamp NaN positions from pre-fill data so filled frames
    # don't become burst/dwell evidence.
    processed[~valid_mask] = np.nan

    return processed, qc_table, kept_ids


# ═══════════════════════════════════════════════════════════════════════════
# 4. NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_matrix(
    matrix,
    method="per_trace_percentile",
    percentile_low=5,
    percentile_high=95,
    clip=True,
):
    """Normalize each row for thresholding.

    Returns
    -------
    normalized : ndarray  (same shape, values in [0, 1] if clipped)
    """
    X = np.asarray(matrix, dtype=float)
    X_norm = np.full_like(X, np.nan)

    for i in range(X.shape[0]):
        row = X[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        vals = row[finite]

        if method == "per_trace_percentile":
            lo = np.percentile(vals, percentile_low)
            hi = np.percentile(vals, percentile_high)
            scale = max(hi - lo, 1e-9)
            row_norm = (row - lo) / scale
        elif method == "per_trace_max":
            mx = np.nanmax(vals)
            row_norm = row / max(mx, 1e-9)
        elif method == "global_percentile":
            all_finite = X[np.isfinite(X)]
            lo = np.percentile(all_finite, percentile_low)
            hi = np.percentile(all_finite, percentile_high)
            scale = max(hi - lo, 1e-9)
            row_norm = (row - lo) / scale
        elif method == "none":
            row_norm = row.copy()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if clip and method != "none":
            row_norm = np.clip(row_norm, 0.0, 1.0)
        X_norm[i] = row_norm

    return X_norm


# ═══════════════════════════════════════════════════════════════════════════
# 5. BURST / DWELL CALLING
# ═══════════════════════════════════════════════════════════════════════════

def runs_from_binary(binary_row):
    """Return runs as list of (value, start, end_exclusive).

    NaN frames yield value=np.nan runs.
    """
    n = len(binary_row)
    if n == 0:
        return []
    runs = []
    cur_val = binary_row[0]
    cur_start = 0
    for j in range(1, n):
        v = binary_row[j]
        same = (np.isnan(cur_val) and np.isnan(v)) or (cur_val == v)
        if not same:
            runs.append((cur_val, cur_start, j))
            cur_val = v
            cur_start = j
    runs.append((cur_val, cur_start, n))
    return runs


def _off_baseline_bar(row, k_mad, off_quantile):
    """Per-trace bar = median(bottom off_quantile) + k_mad * MAD_off.

    MAD is scaled by 1.4826 so k_mad approximates the sigma-multiplier under
    Gaussian-noise assumptions. Returns NaN if the trace has too few finite
    or too few OFF-pool samples to estimate the baseline robustly.
    """
    finite = row[np.isfinite(row)]
    if finite.size < 4:
        return np.nan
    cutoff = np.percentile(finite, off_quantile * 100.0)
    off_pool = finite[finite <= cutoff]
    if off_pool.size < 2:
        return np.nan
    baseline = np.median(off_pool)
    mad_off = np.median(np.abs(off_pool - baseline)) * 1.4826
    return baseline + k_mad * mad_off


def call_bursts(
    matrix_for_thresholding,
    raw_matrix=None,
    processed_matrix=None,
    trajectory_ids=None,
    time_interval_seconds=5.0,
    threshold=0.05,
    threshold_mode="fraction_of_trace_max",
    off_baseline_quantile=0.25,
    min_burst_duration_seconds=60.0,
    min_event_duration_frames=6,
    max_nan_bridge=2,
    count_initial_dwell=True,
    exclude_terminal_dwell=True,
):
    """Call ON/OFF episodes from a thresholding matrix.

    ON episodes ("bursts") are threshold-positive runs that also pass
    the duration filter.  OFF episodes ("dwells") are the intervals
    between accepted ON episodes.

    Processing order after thresholding:
      Step 2  – merge runs shorter than *min_event_duration_frames*
      Step 2b – re-label ON runs failing *min_burst_duration_seconds* as OFF
      Step 2c -- bridge NaN gaps <= *max_nan_bridge* between same-state neighbours

    Parameters
    ----------
    max_nan_bridge : int
        Maximum NaN gap (frames) to bridge between same-state neighbours.
        Set to 0 to disable bridging.

    Returns
    -------
    binary_matrix : ndarray (0/1/NaN)
        Accepted ON episodes = 1, OFF = 0, unresolved = NaN.
    event_table : DataFrame
    trajectory_summary : DataFrame
    """
    n_traces, n_time = matrix_for_thresholding.shape
    if raw_matrix is None:
        raw_matrix = matrix_for_thresholding
    if processed_matrix is None:
        processed_matrix = matrix_for_thresholding
    if trajectory_ids is None:
        trajectory_ids = [f"traj_{i}" for i in range(n_traces)]

    binary_matrix = np.full_like(matrix_for_thresholding, np.nan)
    threshold_values = np.full(n_traces, np.nan, dtype=float)
    threshold_sources = np.full(n_traces, "", dtype=object)

    # Step 1: Apply threshold
    for i in range(n_traces):
        row = matrix_for_thresholding[i]
        finite = np.isfinite(row)
        if threshold_mode == "normalized_absolute":
            threshold_values[i] = float(threshold)
            threshold_sources[i] = "normalized"
            binary_matrix[i, finite] = (row[finite] >= threshold_values[i]).astype(float)
        elif threshold_mode == "fraction_of_trace_max":
            proc_row = processed_matrix[i]
            compare_mask = finite & np.isfinite(proc_row)
            if not np.any(compare_mask):
                continue
            thresh_val = threshold * np.nanmax(proc_row[compare_mask])
            threshold_values[i] = float(thresh_val)
            threshold_sources[i] = "processed"
            binary_matrix[i, compare_mask] = (proc_row[compare_mask] >= thresh_val).astype(float)
        elif threshold_mode == "absolute_raw":
            raw_row = raw_matrix[i]
            compare_mask = finite & np.isfinite(raw_row)
            threshold_values[i] = float(threshold)
            threshold_sources[i] = "raw"
            binary_matrix[i, compare_mask] = (raw_row[compare_mask] >= threshold_values[i]).astype(float)
        elif threshold_mode == "off_baseline_mad":
            # Goldman-style: bar = OFF-baseline + k * MAD_off, computed
            # from the processed (smoothed) trace. `threshold` is k.
            proc_row = processed_matrix[i]
            compare_mask = finite & np.isfinite(proc_row)
            bar = _off_baseline_bar(proc_row[compare_mask], threshold, off_baseline_quantile)
            if not np.isfinite(bar):
                continue  # leave row as all-NaN; downstream stats skip it
            threshold_values[i] = float(bar)
            threshold_sources[i] = "processed"
            binary_matrix[i, compare_mask] = (proc_row[compare_mask] >= bar).astype(float)
        elif threshold_mode == "snr":
            # SNR is already a self-normalized quality metric.
            # matrix_for_thresholding contains per-frame SNR values;
            # threshold is the SNR cutoff (e.g. 3.0).
            snr_row = matrix_for_thresholding[i]
            compare_mask = finite & np.isfinite(snr_row)
            threshold_values[i] = float(threshold)
            threshold_sources[i] = "snr"
            binary_matrix[i, compare_mask] = (snr_row[compare_mask] >= threshold).astype(float)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    # Step 2: Event cleanup — merge short non-NaN runs into the longer
    # adjacent non-NaN neighbour. Iterate until no short runs remain.
    # Safety cap prevents infinite oscillation.
    MAX_CLEANUP_ITERS = 100
    for i in range(n_traces):
        row = binary_matrix[i]
        for _iter in range(MAX_CLEANUP_ITERS):
            runs = runs_from_binary(row)
            merged_any = False
            for r_idx, (val, start, end) in enumerate(runs):
                dur = end - start
                if np.isnan(val) or dur >= min_event_duration_frames:
                    continue
                # Find the longest adjacent non-NaN neighbour to merge into
                merge_val = None
                # Check left neighbour
                if r_idx > 0:
                    lv, ls, le = runs[r_idx - 1]
                    if not np.isnan(lv):
                        merge_val = lv
                # Check right neighbour — prefer the longer one
                if r_idx < len(runs) - 1:
                    rv, rs, re = runs[r_idx + 1]
                    if not np.isnan(rv):
                        if merge_val is None:
                            merge_val = rv
                        else:
                            # Pick the longer neighbour
                            left_len = runs[r_idx - 1][2] - runs[r_idx - 1][1]
                            right_len = re - rs
                            if right_len > left_len:
                                merge_val = rv
                if merge_val is not None and merge_val != val:
                    row[start:end] = merge_val
                    merged_any = True
                    break  # restart scan after modification
            if not merged_any:
                break
        binary_matrix[i] = row

    # Step 2b: Re-label ON runs failing min_burst_duration_seconds as OFF.
    # After this, binary_matrix represents accepted ON episodes only.
    # fraction_time_on will count only qualifying bursts.
    for i in range(n_traces):
        row = binary_matrix[i]
        runs = runs_from_binary(row)
        for val, start, end in runs:
            if val == 1.0 and (end - start) * time_interval_seconds < min_burst_duration_seconds:
                row[start:end] = 0.0
        binary_matrix[i] = row

    # Step 2c: Bridge short NaN gaps between same-state neighbours.
    # Runs AFTER failed-ON relabeling so OFF-NaN-shortON-NaN-OFF
    # --> OFF-NaN-OFF-NaN-OFF --> OFF (single continuous dwell).
    n_bridged_total = 0
    if max_nan_bridge > 0:
        for i in range(n_traces):
            row = binary_matrix[i]
            runs = runs_from_binary(row)
            for r_idx, (val, start, end) in enumerate(runs):
                if not np.isnan(val):
                    continue
                if (end - start) > max_nan_bridge:
                    continue
                left_val = runs[r_idx - 1][0] if r_idx > 0 else np.nan
                right_val = runs[r_idx + 1][0] if r_idx < len(runs) - 1 else np.nan
                if np.isfinite(left_val) and left_val == right_val:
                    row[start:end] = left_val
                    n_bridged_total += (end - start)
            binary_matrix[i] = row
    if n_bridged_total > 0:
        print(f"    NaN bridging: {n_bridged_total} frames bridged "
              f"(max gap = {max_nan_bridge} frames)")

    # Step 3: Build event table and trajectory summary
    event_records = []
    summary_records = []
    dt = time_interval_seconds

    for i in range(n_traces):
        row_bin = binary_matrix[i]
        row_raw = raw_matrix[i]
        row_proc = processed_matrix[i]
        row_norm = matrix_for_thresholding[i]
        traj_id = trajectory_ids[i]
        threshold_value = threshold_values[i]
        threshold_value_out = float(threshold_value) if np.isfinite(threshold_value) else np.nan
        threshold_source = threshold_sources[i]

        runs = runs_from_binary(row_bin)
        # Filter out NaN-only runs
        runs = [(v, s, e) for v, s, e in runs if not np.isnan(v)]

        events_for_traj = []
        for ev_idx, (val, start, end) in enumerate(runs):
            dur_frames = end - start
            dur_sec = dur_frames * dt
            dur_min = dur_sec / 60.0

            raw_seg = row_raw[start:end]
            proc_seg = row_proc[start:end]
            norm_seg = row_norm[start:end]

            is_on = val == 1.0
            is_first = ev_idx == 0
            is_last = ev_idx == len(runs) - 1
            is_initial_dwell = (not is_on) and is_first
            is_terminal = is_last and not is_on

            # Duration filter
            if is_on:
                passes = dur_sec >= min_burst_duration_seconds
            else:
                passes = True  # dwells always pass duration filter

            event_type = "burst" if is_on else "dwell"
            events_for_traj.append({
                "trajectory_id": traj_id,
                "trajectory_index": i,
                "event_index": ev_idx,
                "event_type": event_type,
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_time_seconds": start * dt,
                "end_time_seconds": end * dt,
                "duration_frames": dur_frames,
                "duration_seconds": dur_sec,
                "duration_minutes": round(dur_min, 4),
                "mean_raw_intensity": float(np.nanmean(raw_seg)),
                "max_raw_intensity": float(np.nanmax(raw_seg)) if np.any(np.isfinite(raw_seg)) else np.nan,
                "mean_processed_intensity": float(np.nanmean(proc_seg)),
                "mean_normalized_intensity": float(np.nanmean(norm_seg)),
                "area_raw": float(np.nansum(raw_seg)) * dt,
                "area_normalized": float(np.nansum(norm_seg)) * dt,
                "is_initial_dwell": is_initial_dwell,
                "is_terminal_event": is_terminal,
                "passes_duration_filter": passes,
                "threshold_mode": threshold_mode,
                "threshold_used": threshold,
                "threshold_value_used": threshold_value_out,
                "threshold_source": threshold_source,
            })
        event_records.extend(events_for_traj)

        # Trajectory summary
        n_valid = int(np.sum(np.isfinite(row_raw)))
        bursts = [e for e in events_for_traj
                  if e["event_type"] == "burst" and e["passes_duration_filter"]]
        dwells = [e for e in events_for_traj
                  if e["event_type"] == "dwell"
                  and not e["is_terminal_event"]
                  and (not e["is_initial_dwell"] or count_initial_dwell)]
        if exclude_terminal_dwell:
            dwells = [d for d in dwells if not d["is_terminal_event"]]

        n_bursts = len(bursts)
        n_dwells = len(dwells)
        total_burst_sec = sum(e["duration_seconds"] for e in bursts)
        total_dwell_sec = sum(e["duration_seconds"] for e in dwells)
        n_on = int(np.nansum(row_bin == 1.0))
        n_valid_bin = int(np.sum(np.isfinite(row_bin)))
        frac_on = n_on / max(n_valid_bin, 1)

        burst_durs_min = [e["duration_minutes"] for e in bursts]
        dwell_durs_min = [e["duration_minutes"] for e in dwells]

        raw_finite = row_raw[np.isfinite(row_raw)]
        proc_finite = row_proc[np.isfinite(row_proc)]
        norm_finite = row_norm[np.isfinite(row_norm)]
        summary_records.append({
            "trajectory_id": traj_id,
            "trajectory_index": i,
            "n_timepoints": n_time,
            "n_valid_timepoints": n_valid,
            "valid_fraction": round(n_valid / n_time, 4),
            "n_bursts": n_bursts,
            "n_dwells": n_dwells,
            "total_burst_time_seconds": round(total_burst_sec, 2),
            "total_dwell_time_seconds": round(total_dwell_sec, 2),
            "fraction_time_on": round(frac_on, 4),
            "mean_burst_duration_minutes": round(float(np.mean(burst_durs_min)), 4) if burst_durs_min else np.nan,
            "median_burst_duration_minutes": round(float(np.median(burst_durs_min)), 4) if burst_durs_min else np.nan,
            "mean_dwell_duration_minutes": round(float(np.mean(dwell_durs_min)), 4) if dwell_durs_min else np.nan,
            "median_dwell_duration_minutes": round(float(np.median(dwell_durs_min)), 4) if dwell_durs_min else np.nan,
            "mean_raw_intensity": float(np.nanmean(raw_finite)) if raw_finite.size > 0 else np.nan,
            "max_raw_intensity": float(np.nanmax(raw_finite)) if raw_finite.size > 0 else np.nan,
            "max_processed_intensity": float(np.nanmax(proc_finite)) if proc_finite.size > 0 else np.nan,
            "max_normalized_intensity": float(np.nanmax(norm_finite)) if norm_finite.size > 0 else np.nan,
            "dynamic_range_raw": float(np.ptp(raw_finite)) if raw_finite.size > 0 else 0.0,
            "threshold_mode": threshold_mode,
            "threshold_used": threshold,
            "threshold_value_used": threshold_value_out,
            "threshold_source": threshold_source,
        })

    event_table = pd.DataFrame(event_records)
    trajectory_summary = pd.DataFrame(summary_records)
    return binary_matrix, event_table, trajectory_summary


# ---------------------------------------------------------------------------
# Dual-channel kymograph from pre-loaded matrices
# ---------------------------------------------------------------------------

def plot_dual_channel_kymograph_from_matrix(
    ch0_matrix,
    ch1_matrix,
    *,
    output_dir=None,
    time_interval_seconds=5.0,
    condition="",
    normalize="per_trace_percentile",
    p_lo=1,
    p_hi=99,
    sort_by="fraction_on",
    trajectory_summary=None,
    max_traces_to_plot=None,
    ch0_color=None,
    ch1_color=None,
    nan_color=(0.0, 0.0, 0.0),
    figsize=(14, 6),
    dpi=300,
    show=False,
    filename_stem="kymograph_dual_channel",
):
    """Render a dual-channel additive-blend kymograph from two matrices.

    Parameters
    ----------
    ch0_matrix : ndarray  (N, T)
        Channel 0 (folding) intensity matrix.
    ch1_matrix : ndarray  (N, T)
        Channel 1 (nascent) intensity matrix.  Must have the same shape.
    output_dir : Path, optional
        Directory to save the figure.  If *None*, figure is returned but
        not saved.
    time_interval_seconds : float
        Time per frame (seconds).
    condition : str
        Label for the title.
    normalize : str
        ``"per_trace_percentile"`` | ``"per_trace_max"`` | ``None``.
    p_lo, p_hi : float
        Percentiles for per-trace normalization.
    sort_by : str
        ``"fraction_on"`` uses *trajectory_summary* to sort; ``"density"``
        sorts by number of finite values; ``None`` keeps original order.
    trajectory_summary : DataFrame, optional
        Needed when *sort_by="fraction_on"*.
    max_traces_to_plot : int
        Cap on displayed trajectories.
    ch0_color, ch1_color : tuple (r, g, b), optional
        Override channel colors.  Defaults: ch0=Green, ch1=Magenta.
    nan_color : tuple
        RGB for missing data (default black).
    figsize, dpi : tuple, int
        Figure size and resolution.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    ch0 = np.asarray(ch0_matrix, dtype=float)
    ch1 = np.asarray(ch1_matrix, dtype=float)
    if ch0.shape != ch1.shape:
        raise ValueError(
            f"Channel shape mismatch: ch0={ch0.shape} vs ch1={ch1.shape}"
        )

    N, T = ch0.shape
    if N == 0:
        raise ValueError("No trajectories to plot")

    # ── colours ──
    c0 = ch0_color if ch0_color is not None else CHANNEL_GREEN    # Folding
    c1 = ch1_color if ch1_color is not None else CHANNEL_MAGENTA  # Nascent
    c_nan = tuple(float(v) for v in nan_color)

    # ── sorting ──
    if sort_by == "fraction_on" and trajectory_summary is not None:
        sort_idx = np.argsort(
            trajectory_summary["fraction_time_on"].values
        )[::-1]
    elif sort_by == "density":
        density = np.sum(np.isfinite(ch0), axis=1) + np.sum(
            np.isfinite(ch1), axis=1
        )
        sort_idx = np.argsort(-density)
    else:
        sort_idx = np.arange(N)

    # ── cap trajectories ──
    if max_traces_to_plot is not None and len(sort_idx) > max_traces_to_plot:
        sort_idx = sort_idx[:max_traces_to_plot]

    ch0 = ch0[sort_idx]
    ch1 = ch1[sort_idx]
    H = ch0.shape[0]

    # ── normalisation (per-trace, NaN safe) ──
    def _norm(X, mode, plo, phi):
        out = np.full_like(X, np.nan)
        for i in range(X.shape[0]):
            row = X[i]
            mask = np.isfinite(row)
            if not mask.any():
                continue
            vals = row[mask]
            if mode == "per_trace_percentile":
                lo = np.percentile(vals, plo)
                hi = np.percentile(vals, phi)
                scale = max(hi - lo, 1e-9)
                normed = (row - lo) / scale
            elif mode == "per_trace_max":
                mx = max(np.max(vals), 1e-9)
                normed = row / mx
            elif mode == "fixed_range":
                # plo/phi are absolute data bounds, not percentiles.
                # Use case: SNR kymograph where plo=0, phi=SNR_CAP.
                lo = float(plo)
                hi = float(phi)
                scale = max(hi - lo, 1e-9)
                normed = (row - lo) / scale
            else:  # raw --> global percentile fallback
                all_f = X[np.isfinite(X)]
                lo = np.percentile(all_f, plo) if all_f.size else 0
                hi = np.percentile(all_f, phi) if all_f.size else 1
                scale = max(hi - lo, 1e-9)
                normed = (row - lo) / scale
            out[i] = np.clip(normed, 0, 1)
        return out

    if normalize is not None:
        ch0_n = _norm(ch0, normalize, p_lo, p_hi)
        ch1_n = _norm(ch1, normalize, p_lo, p_hi)
    else:
        ch0_n = _norm(ch0, "per_trace_percentile", p_lo, p_hi)
        ch1_n = _norm(ch1, "per_trace_percentile", p_lo, p_hi)

    # ── build RGB image (additive blend) ──
    img = np.zeros((H, T, 3), dtype=float)
    nan_both = np.isnan(ch0_n) & np.isnan(ch1_n)
    img[nan_both, 0] = c_nan[0]
    img[nan_both, 1] = c_nan[1]
    img[nan_both, 2] = c_nan[2]

    v0 = np.where(np.isfinite(ch0_n), ch0_n, 0.0)
    v1 = np.where(np.isfinite(ch1_n), ch1_n, 0.0)
    for ci in range(3):
        img[:, :, ci] += v0 * c0[ci] + v1 * c1[ci]
    img = np.clip(img, 0.0, 1.0)

    # ── time axis ──
    dt = float(time_interval_seconds)
    t_max_min = (T - 1) * dt / 60.0

    # ── plot ──
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.imshow(
        img, aspect="auto", interpolation="nearest", origin="upper",
        extent=[0, t_max_min, H, 0],
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Trajectory (sorted)")
    ax.set_title(
        f"{condition} - Dual-Channel Kymograph"
        if condition else "Dual-Channel Kymograph"
    )
    style_axes(ax, grid=False)
    fig.tight_layout()

    if output_dir is not None:
        plots_dir = Path(output_dir) / "plots" / "quality_control"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, plots_dir / filename_stem, dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_burst_results(
    raw_matrix,
    processed_matrix,
    normalized_matrix,
    binary_matrix,
    event_table,
    trajectory_summary,
    qc_table,
    output_dir,
    time_interval_seconds=5.0,
    condition="",
    kymograph_figsize=(14, 6),
    kymograph_dpi=300,
    max_traces_to_plot=None,
    trace_figsize=(12, 8),
    distribution_figsize=(8, 5),
    summary_figsize=(6, 4),
    plot_dpi=300,
    n_example_traces=10,
    example_trace_figsize=(14, 18),
    threshold=0.05,
    threshold_mode="fraction_of_trace_max",
    off_baseline_quantile=0.25,
    kymograph_sort_by="density",
):
    """Generate 8 diagnostic plots and save as PNG + SVG."""
    set_publication_style()
    plots_dir = Path(output_dir) / "plots" / "quality_control"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dt = time_interval_seconds
    n_traces, n_time = raw_matrix.shape
    t_min = np.arange(n_time) * dt / 60.0

    # ---- 1. Mean trace: raw vs processed ----
    fig, axes = plt.subplots(2, 1, figsize=trace_figsize, sharex=True, facecolor="white")
    for ax, mat, label, color in zip(
        axes,
        [raw_matrix, processed_matrix],
        ["Raw", "Processed"],
        [TRACE_GRAY, TRACE_MAGENTA],
    ):
        mean_tr = np.nanmean(mat, axis=0)
        n_valid = np.sum(np.isfinite(mat), axis=0)
        sem = np.divide(
            np.nanstd(mat, axis=0),
            np.sqrt(n_valid),
            out=np.zeros_like(mean_tr, dtype=float),
            where=n_valid > 0,
        )
        ax.plot(t_min, mean_tr, linewidth=2.0, color=color, label=f"{label} mean (n={n_traces})")
        ax.fill_between(t_min, mean_tr - sem, mean_tr + sem, alpha=0.18, color=color, linewidth=0)
        ax.set_ylabel(f"{label} Intensity")
        style_axes(ax, grid=False)
        style_legend(ax.legend(loc="upper right", fontsize=10))
    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"{condition} - Mean Trace", fontsize=14, fontname="Arial")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, plots_dir / "mean_trace_raw_vs_processed", plot_dpi)

    # ---- 2 & 3. Single-channel kymographs: SKIPPED ──────────────────────
    # Superseded by the dual-channel kymograph (plot_dual_channel_kymograph_from_matrix)
    # generated in run_analysis.py after QC filtering.


    # ---- 4. Example traces: SKIPPED ──────────────────────────────────────
    # Sampled example traces are no longer generated here; the per-construct
    # burst/dwell plots (sections 5/6) provide sufficient diagnostic detail.

    # ---- 5/6. Duration distributions ----
    for etype, fname in [("burst", "burst_duration_distribution"),
                          ("dwell", "dwell_duration_distribution")]:
        if len(event_table) == 0:
            continue
        # Apply the same terminal-dwell exclusion used in the trajectory
        # summary so this plot matches the reported statistics.
        mask = (event_table["event_type"] == etype) & (event_table["passes_duration_filter"])
        if etype == "dwell":
            mask = mask & (~event_table["is_terminal_event"])
            mask = mask & (~event_table["is_initial_dwell"])
        sub = event_table[mask]
        if len(sub) == 0:
            continue
        durs = sub["duration_minutes"].values
        color = TRACE_MAGENTA if etype == "burst" else TRACE_BLUE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=distribution_figsize, facecolor="white")
        ax1.hist(durs, bins="auto", color=color, edgecolor="black",
                 linewidth=0.4, alpha=0.78)
        ax1.set_xlabel("Duration (min)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"{etype.capitalize()} Duration Distribution")
        style_axes(ax1, grid=False)
        sorted_d = np.sort(durs)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.step(sorted_d, cdf, color=color, linewidth=2.0)
        ax2.set_xlabel("Duration (min)")
        ax2.set_ylabel("CDF")
        ax2.set_title("Cumulative Distribution")
        style_axes(ax2, grid=False)
        fig.suptitle(f"{condition}", fontsize=14, fontname="Arial")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, plots_dir / fname, plot_dpi)

    # ---- 7. Fraction time ON distribution ----
    fig, ax = plt.subplots(1, 1, figsize=summary_figsize, facecolor="white")
    frac_on = trajectory_summary["fraction_time_on"].values
    ax.hist(frac_on, bins=20, color=TRACE_MAGENTA, edgecolor="black", linewidth=0.4, alpha=0.78)
    ax.set_xlabel("Fraction Time ON")
    ax.set_ylabel("Count")
    ax.set_title(f"{condition} - Fraction Time ON per Trajectory")
    style_axes(ax, grid=False)
    fig.tight_layout()
    save_figure(fig, plots_dir / "fraction_time_on_distribution", plot_dpi)

    # ---- 8. QC summary ----
    fig, ax = plt.subplots(1, 1, figsize=summary_figsize, facecolor="white")
    counts = qc_table["qc_reason"].value_counts()
    colors = {"ok": "#34a853", "constitutive": "#8e44ad",
              "too_sparse": "#f39c12",
              "too_many_internal_nans": "#c0392b",
              "shift_alignment_filter": "#757575",
              "flat_zero": "#9e9e9e", "empty": "#424242"}
    bar_colors = [colors.get(r, "#757575") for r in counts.index]
    ax.barh(counts.index, counts.values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of Trajectories")
    ax.set_title(f"{condition} - QC Summary")
    # Place count labels with enough room so they aren't clipped
    x_max = counts.values.max()
    label_offset = max(x_max * 0.03, 0.5)
    for i, (v, r) in enumerate(zip(counts.values, counts.index)):
        ax.text(v + label_offset, i, str(v), va="center", fontsize=10, color="black")
    ax.set_xlim(right=x_max + x_max * 0.15)  # 15% padding for labels
    style_axes(ax, grid=False)
    fig.tight_layout()
    save_figure(fig, plots_dir / "qc_summary", plot_dpi)


# ═══════════════════════════════════════════════════════════════════════════
# 7. END-TO-END RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_burst_quantification(
    input_path=None,
    input_matrix=None,
    snr_matrix=None,
    output_dir="burst_results",
    time_interval_seconds=5.0,
    condition="",
    # Preprocessing
    min_valid_fraction=0.30,
    max_internal_nan_gap=2,
    align_first_valid=False,
    detrend_method=None,
    smooth_method="median",
    smooth_window=3,
    # Normalization
    normalization_method="per_trace_percentile",
    percentile_low=5,
    percentile_high=95,
    # Burst calling — default to fraction_of_trace_max matching Wu Lab
    # (Mol Cell 2023). See README "Threshold Modes" for alternatives.
    threshold=0.05,
    threshold_mode="fraction_of_trace_max",
    # For threshold_mode="off_baseline_mad": fraction of frames defining the
    # per-trace OFF pool. Bar = median(OFF) + threshold * MAD_off * 1.4826.
    off_baseline_quantile=0.25,
    min_event_duration_frames=6,
    min_burst_duration_seconds=60.0,
    count_initial_dwell=True,
    exclude_terminal_dwell=True,
    # Plot params
    kymograph_figsize=(14, 6),
    kymograph_dpi=300,
    kymograph_sort_by="density",
    max_traces_to_plot=160,
    trace_figsize=(12, 8),
    distribution_figsize=(8, 5),
    summary_figsize=(6, 4),
    comparison_figsize=(10, 6),
    plot_dpi=300,
    n_example_traces=10,
    example_trace_figsize=(14, 18),
    # Misc
    save_intermediates=True,
    generate_plots=True,
):
    """Run the complete burst quantification pipeline.

    Accepts either a file path (input_path) or a NumPy array (input_matrix).

    Returns
    -------
    dict with keys: raw_matrix, processed_matrix, normalized_matrix,
         binary_matrix, qc_table, event_table, trajectory_summary, params
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    if input_matrix is not None:
        raw_matrix = np.asarray(input_matrix, dtype=float)
        trajectory_ids = None
    elif input_path is not None:
        raw_matrix, trajectory_ids, _ = load_intensity_matrix(input_path)
    else:
        raise ValueError("Provide either input_path or input_matrix")

    # 2. Validate
    raw_matrix, trajectory_ids, valid_rows, col_slice = validate_intensity_matrix(
        raw_matrix, trajectory_ids
    )
    print(f"  Validated: {raw_matrix.shape[0]} trajectories x {raw_matrix.shape[1]} timepoints")

    # Apply same row/col filtering to SNR matrix (Stage 5 sync)
    if snr_matrix is not None:
        snr_matrix = np.asarray(snr_matrix, dtype=float)
        snr_matrix = snr_matrix[valid_rows][:, col_slice]

    # 3. Preprocess
    processed_matrix, qc_table, kept_ids = preprocess_intensity_matrix(
        raw_matrix, trajectory_ids,
        min_valid_fraction=min_valid_fraction,
        max_internal_nan_gap=max_internal_nan_gap,
        align_first_valid=align_first_valid,
        detrend_method=detrend_method,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
    )
    n_kept = processed_matrix.shape[0]
    n_removed = raw_matrix.shape[0] - n_kept
    print(f"  QC: {n_kept} kept, {n_removed} removed")

    if n_kept == 0:
        print("  WARNING: no trajectories passed QC")
        # Still save params and QC table for debugging/provenance
        if save_intermediates:
            output_dir = Path(output_dir)
            quant_dir = output_dir / "quantification"
            quant_dir.mkdir(parents=True, exist_ok=True)
            zero_params = {
                "condition": condition,
                "time_interval_seconds": time_interval_seconds,
                "n_trajectories_input": raw_matrix.shape[0],
                "n_trajectories_kept": 0,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
                "note": "No trajectories passed QC.",
            }
            with open(quant_dir / "params.json", "w") as f:
                json.dump(zero_params, f, indent=2)
            qc_table.to_csv(quant_dir / "qc_table.csv", index=False)
            print(f"  Saved params.json and qc_table.csv to {quant_dir}")
        return {
            "raw_matrix": raw_matrix, "processed_matrix": processed_matrix,
            "normalized_matrix": np.empty((0, 0)),
            "binary_matrix": np.empty((0, 0)),
            "qc_table": qc_table, "event_table": pd.DataFrame(),
            "trajectory_summary": pd.DataFrame(), "params": {},
        }

    # Filter raw (and SNR) to match kept rows
    keep_idx = qc_table[qc_table["qc_status"] == "kept"]["trajectory_index"].values
    raw_kept = raw_matrix[keep_idx]
    snr_kept = snr_matrix[keep_idx] if snr_matrix is not None else None
    if snr_kept is not None:
        assert snr_kept.shape == raw_kept.shape, (
            f"SNR/raw shape mismatch after QC: {snr_kept.shape} vs {raw_kept.shape}"
        )

    # 4. Normalize
    normalized_matrix = normalize_matrix(
        processed_matrix, method=normalization_method,
        percentile_low=percentile_low, percentile_high=percentile_high,
    )

    # 5. Call bursts
    # When threshold_mode="snr", threshold the SNR matrix directly;
    # otherwise threshold the normalized intensity matrix.
    if threshold_mode == "snr" and snr_kept is not None:
        thresholding_matrix = snr_kept
    elif threshold_mode == "snr" and snr_kept is None:
        raise ValueError(
            "threshold_mode='snr' requires an snr_matrix, but none was "
            "provided. Pass snr_matrix= to run_burst_quantification() or "
            "switch to a different threshold_mode."
        )
    else:
        thresholding_matrix = normalized_matrix

    binary_matrix, event_table, trajectory_summary = call_bursts(
        matrix_for_thresholding=thresholding_matrix,
        raw_matrix=raw_kept,
        processed_matrix=processed_matrix,
        trajectory_ids=kept_ids,
        time_interval_seconds=time_interval_seconds,
        threshold=threshold,
        threshold_mode=threshold_mode,
        off_baseline_quantile=off_baseline_quantile,
        min_burst_duration_seconds=min_burst_duration_seconds,
        min_event_duration_frames=min_event_duration_frames,
        max_nan_bridge=max_internal_nan_gap,
        count_initial_dwell=count_initial_dwell,
        exclude_terminal_dwell=exclude_terminal_dwell,
    )

    # Merge QC reason into trajectory summary (Fix B1)
    if not trajectory_summary.empty and not qc_table.empty:
        qc_kept = qc_table[qc_table["qc_status"] == "kept"][
            ["trajectory_id", "qc_status", "qc_reason"]
        ]
        trajectory_summary = trajectory_summary.merge(
            qc_kept, on="trajectory_id", how="left"
        )
        # Fill any missing (shouldn't happen)
        trajectory_summary["qc_status"] = trajectory_summary["qc_status"].fillna("kept")
        trajectory_summary["qc_reason"] = trajectory_summary["qc_reason"].fillna("ok")

    n_bursts = int(trajectory_summary["n_bursts"].sum()) if not trajectory_summary.empty else 0
    n_dwells = int(trajectory_summary["n_dwells"].sum()) if not trajectory_summary.empty else 0
    print(f"  Events: {n_bursts} bursts, {n_dwells} dwells")

    # Save params
    params = {
        "condition": condition,
        "time_interval_seconds": time_interval_seconds,
        "n_trajectories_input": int(raw_matrix.shape[0]),
        "n_trajectories_kept": n_kept,
        "n_timepoints": int(processed_matrix.shape[1]),
        "min_valid_fraction": min_valid_fraction,
        "max_internal_nan_gap": max_internal_nan_gap,
        "align_first_valid": align_first_valid,
        "detrend_method": detrend_method,
        "smooth_method": smooth_method,
        "smooth_window": smooth_window,
        "normalization_method": normalization_method,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "off_baseline_quantile": off_baseline_quantile,
        "min_event_duration_frames": min_event_duration_frames,
        "min_burst_duration_seconds": min_burst_duration_seconds,
        "count_initial_dwell": count_initial_dwell,
        "exclude_terminal_dwell": exclude_terminal_dwell,
        "n_bursts_total": n_bursts,
        "n_dwells_total": n_dwells,
        "timestamp": datetime.now().isoformat(),
        "terminology_note": "Events labeled 'burst'/'dwell' are observed "
                           "folding-channel ON/OFF episodes, not direct "
                           "measurements of translational initiation.",
    }

    # Save
    if save_intermediates:
        quant_dir = output_dir / "quantification"
        quant_dir.mkdir(parents=True, exist_ok=True)
        with open(quant_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        qc_table.to_csv(quant_dir / "qc_table.csv", index=False)
        event_table.to_csv(quant_dir / "event_table.csv", index=False)
        trajectory_summary.to_csv(quant_dir / "trajectory_summary.csv", index=False)
        np.save(quant_dir / "processed_matrix.npy", processed_matrix)
        np.save(quant_dir / "normalized_matrix.npy", normalized_matrix)
        np.save(quant_dir / "binary_matrix.npy", binary_matrix)
        print(f"  Saved outputs to {quant_dir}")

    # Plot
    if generate_plots and n_kept > 0:
        plot_burst_results(
            raw_matrix=raw_kept,
            processed_matrix=processed_matrix,
            normalized_matrix=normalized_matrix,
            binary_matrix=binary_matrix,
            event_table=event_table,
            trajectory_summary=trajectory_summary,
            qc_table=qc_table,
            output_dir=output_dir,
            time_interval_seconds=time_interval_seconds,
            condition=condition,
            kymograph_figsize=kymograph_figsize,
            kymograph_dpi=kymograph_dpi,
            max_traces_to_plot=max_traces_to_plot,
            trace_figsize=trace_figsize,
            distribution_figsize=distribution_figsize,
            summary_figsize=summary_figsize,
            plot_dpi=plot_dpi,
            n_example_traces=n_example_traces,
            example_trace_figsize=example_trace_figsize,
            threshold=threshold,
            threshold_mode=threshold_mode,
            off_baseline_quantile=off_baseline_quantile,
            kymograph_sort_by=kymograph_sort_by,
        )
        print(f"  Plots saved to {output_dir / 'plots' / 'quality_control'}")

    return {
        "raw_matrix": raw_kept,
        "processed_matrix": processed_matrix,
        "normalized_matrix": normalized_matrix,
        "binary_matrix": binary_matrix,
        "qc_table": qc_table,
        "event_table": event_table,
        "trajectory_summary": trajectory_summary,
        "params": params,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. SANITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _run_sanity_checks():
    """Run 7 built-in sanity tests. Returns True if all pass."""
    import tempfile
    results = []
    dt = 5.0
    n_time = 200

    def _test(name, matrix, expected_check, **extra_kwargs):
        """Run pipeline on matrix, apply check, report pass/fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                kwargs = dict(
                    input_matrix=matrix,
                    output_dir=tmpdir,
                    time_interval_seconds=dt,
                    generate_plots=False,
                    save_intermediates=False,
                    min_valid_fraction=0.10,
                    max_internal_nan_gap=2,
                )
                kwargs.update(extra_kwargs)
                result = run_burst_quantification(**kwargs)
                ok = expected_check(result)
            except Exception as e:
                ok = False
                print(f"    EXCEPTION: {e}")
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        results.append(ok)

    print("\n=== Sanity Checks ===")

    # 1. All-zero matrix
    _test("all_zero_matrix",
          np.zeros((10, n_time)),
          lambda r: r["trajectory_summary"].empty or
                    r["trajectory_summary"]["n_bursts"].sum() == 0)

    # 2. All-NaN matrix
    _test("all_nan_matrix",
          np.full((10, n_time), np.nan),
          lambda r: r["processed_matrix"].shape[0] == 0)

    # 3. Constant-high trace --> kept as constitutive, 1 long burst
    _test("constant_trace_constitutive",
          np.full((5, n_time), 100.0),
          lambda r: len(r["trajectory_summary"]) == 5 and
                    r["trajectory_summary"]["n_bursts"].sum() >= 5 and
                    r["trajectory_summary"]["fraction_time_on"].mean() > 0.9)

    # 4. Perfect square pulse (one burst)
    pulse = np.zeros((1, n_time))
    pulse[0, 50:150] = 1.0
    _test("single_square_pulse",
          pulse,
          lambda r: len(r["trajectory_summary"]) == 1 and
                    r["trajectory_summary"]["n_bursts"].iloc[0] >= 1)

    # 5. Short run removal
    short = np.zeros((1, n_time))
    short[0, 50:53] = 1.0  # 3 frames < min_event_duration_frames=6
    _test("short_run_removal",
          short,
          lambda r: r["trajectory_summary"].empty or
                    r["trajectory_summary"]["n_bursts"].sum() == 0)

    # 6. Mixed valid/invalid trajectories
    mixed = np.random.rand(20, n_time)
    mixed[0, :] = np.nan  # fully empty
    mixed[1, :] = 0.0     # flat zero --> removed
    mixed[2, :5] = np.nan  # mostly valid
    _test("mixed_validity",
          mixed,
          lambda r: r["processed_matrix"].shape[0] < 20)

    # 7. Large matrix (stress test)
    large = np.random.rand(500, n_time) * 0.5
    _test("large_matrix_500x200",
          large,
          lambda r: r["processed_matrix"].shape[0] > 0)

    # 8. Pure Gaussian noise with normalized_absolute@0.5 should have
    # moderate fraction ON (~0.5, not ~1.0). This validates that the
    # alternative normalized_absolute mode correctly handles noise.
    # (fraction_of_trace_max is designed for bimodal data and will
    # correctly call noise as mostly ON — that's expected, not a bug.)
    np.random.seed(12345)
    noise = np.random.randn(50, n_time) * 10 + 50
    _test("noise_normalized_absolute",
          noise,
          lambda r: r["trajectory_summary"].empty or
                    r["trajectory_summary"]["fraction_time_on"].mean() < 0.65,
          threshold_mode="normalized_absolute", threshold=0.5)

    # 9. off_baseline_mad on noise: bar = baseline + 4*sigma_off --> very few frames
    # cross it, so fraction_on should be small (<< 0.5).
    _test("noise_off_baseline_mad",
          noise,
          lambda r: r["trajectory_summary"].empty or
                    r["trajectory_summary"]["fraction_time_on"].mean() < 0.30,
          threshold_mode="off_baseline_mad", threshold=4.0)

    # 10. Bimodal noise+pulse: pure noise with periodic positive pulses should
    # have fraction_on around the pulse duty cycle (not all-ON), under
    # off_baseline_mad. Pulses every 40 frames, 8 frames wide --> 20% duty.
    rng = np.random.default_rng(7)
    bimodal = rng.standard_normal((30, n_time)) * 5 + 20  # background
    for t0 in range(20, n_time, 40):
        bimodal[:, t0:t0 + 8] += 80                       # clear pulses
    _test("bimodal_off_baseline_mad",
          bimodal,
          lambda r: (not r["trajectory_summary"].empty and
                     0.10 < r["trajectory_summary"]["fraction_time_on"].mean() < 0.45),
          threshold_mode="off_baseline_mad", threshold=4.0)

    # 11. SNR mode: square pulse with matching SNR matrix --> detects the pulse
    snr_pulse_int = np.random.randn(5, n_time) * 2 + 10  # noisy background
    snr_pulse_snr = np.full((5, n_time), 1.0)              # low SNR everywhere
    snr_pulse_snr[:, 50:150] = 5.0                         # high SNR in pulse
    _test("snr_square_pulse",
          snr_pulse_int,
          lambda r: (not r["trajectory_summary"].empty and
                     r["trajectory_summary"]["n_bursts"].sum() >= 5 and
                     r["trajectory_summary"]["fraction_time_on"].mean() < 0.8),
          snr_matrix=snr_pulse_snr,
          threshold_mode="snr", threshold=3.0)

    # 12. SNR mode without snr_matrix --> must raise ValueError
    print("  ", end="")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_burst_quantification(
                input_matrix=np.random.rand(3, n_time),
                output_dir=tmpdir,
                time_interval_seconds=dt,
                generate_plots=False,
                save_intermediates=False,
                threshold_mode="snr",
                threshold=3.5,
                # snr_matrix deliberately omitted
            )
        print("[FAIL] snr_missing_raises_error")
        results.append(False)
    except ValueError:
        print("[PASS] snr_missing_raises_error")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] snr_missing_raises_error — wrong exception: {e}")
        results.append(False)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Results: {n_pass}/{n_total} passed\n")
    return all(results)


# ═══════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Burst quantification from intensity trajectory matrix."
    )
    parser.add_argument("--input", required=False, help="Path to matrix file (csv/npy/npz)")
    parser.add_argument("--output", default="burst_results", help="Output directory")
    parser.add_argument("--dt", type=float, default=5.0, help="Time interval (seconds)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Burst threshold")
    parser.add_argument("--threshold-mode", default="fraction_of_trace_max",
                        choices=["normalized_absolute", "fraction_of_trace_max",
                                 "absolute_raw", "off_baseline_mad"],
                        help="Threshold mode (snr mode is only available "
                             "programmatically via run_analysis.py)")
    parser.add_argument("--off-baseline-quantile", type=float, default=0.25,
                        help="OFF-pool quantile for off_baseline_mad mode (default 0.25)")
    parser.add_argument("--smooth-method", default="median",
                        choices=["median", "mean", "gaussian", "none"])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--min-valid-fraction", type=float, default=0.30)
    parser.add_argument("--max-internal-nan-gap", type=int, default=2)
    parser.add_argument("--min-event-frames", type=int, default=6)
    parser.add_argument("--min-burst-seconds", type=float, default=60.0)
    parser.add_argument("--norm-method", default="per_trace_percentile",
                        choices=["per_trace_percentile", "per_trace_max",
                                 "global_percentile", "none"])
    parser.add_argument("--condition", default="", help="Condition label for plots")
    parser.add_argument("--test", action="store_true", help="Run sanity checks")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    if args.test:
        ok = _run_sanity_checks()
        raise SystemExit(0 if ok else 1)

    if not args.input:
        parser.error("--input is required (or use --test for sanity checks)")

    run_burst_quantification(
        input_path=args.input,
        output_dir=args.output,
        time_interval_seconds=args.dt,
        condition=args.condition,
        min_valid_fraction=args.min_valid_fraction,
        max_internal_nan_gap=args.max_internal_nan_gap,
        smooth_method=args.smooth_method,
        smooth_window=args.smooth_window,
        normalization_method=args.norm_method,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        off_baseline_quantile=args.off_baseline_quantile,
        min_event_duration_frames=args.min_event_frames,
        min_burst_duration_seconds=args.min_burst_seconds,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
