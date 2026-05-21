#!/usr/bin/env python3
"""
Burst Quantification Analysis — CoF Long Movies
=================================================

Processes 4 constructs from CoF_long_movies, quantifying burst/dwell dynamics
on the **folding channel (ch0)** and generating per-construct diagnostics
plus a cross-construct comparison panel.

Constructs:
    sfGFP      → pRS027 (4sfGFP-2mCh)
    GFPuv      → pRS032 (4GFPuv-2mCh)
    sfGFP_Xbp1 → pRS038 (4sfGFP-2mCh-Xbp1)
    Xbp1_sfGFP → pRS048 (Xbp1-4sfGFP-2mCh)

Channel semantics:
    Channel 1 = nascent protein (always present, tracking channel)
    Channel 0 = folding channel  (bursting signal — this is what we quantify)

Usage:
    conda activate microlive
    python run_analysis.py

Note:
    This script requires the ``microlive`` conda environment.
    Activate it before running:  ``conda activate microlive``
"""
from __future__ import annotations

import json
import sys

import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

local_microlive = Path(__file__).resolve().parent.parent.parent / "microlive"
if local_microlive.exists():
    sys.path.insert(0, str(local_microlive))

from microlive import microscopy as mi

# ── Path setup (matches notebook convention) ────────────────────────────────
# This script lives in cof_paper/time_courses/
repo_root = Path(__file__).resolve().parent.parent  # → cof_paper/
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # time_courses/

from utilities.config import REPORTER_PLASMID_NAME_MAPPING, PLASMID_SHORT_NAME_MAPPING
from burst_quantification import (
    plot_dual_channel_kymograph_from_matrix,
    run_burst_quantification,
)
from plotting_montage_crops import (
    ParticleOrigin,
    generate_representative_montage_pdf,
    parse_metadata_txt,
    resolve_lif_path,
)

from plotting import (
    TRACE_BLUE,
    TRACE_GRAY,
    TRACE_MAGENTA,
    box_with_points,
    save_figure,
    set_publication_style,
    style_axes,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — loaded from config.yaml (override via --config in main)
# ═══════════════════════════════════════════════════════════════════════════

_REQUIRED_SECTIONS = {"data_root", "constructs", "analysis", "plots"}
_REQUIRED_CONSTRUCT_KEYS = {"plasmid", "burst_ch", "track_ch"}
_REQUIRED_ANALYSIS_KEYS = {
    "time_interval_seconds", "max_frames", "min_snr", "snr_channel_index",
    "min_valid_fraction", "min_valid_frames",
    "max_total_internal_nan_frames", "max_internal_nan_gap", "align_first_valid",
    "detrend_method", "smooth_method", "smooth_window",
    "normalization_method", "percentile_low", "percentile_high",
    "threshold_mode", "threshold", "off_baseline_quantile",
    "min_event_duration_frames", "min_burst_duration_seconds",
    "exclude_terminal_dwell", "count_initial_dwell",
}


def load_config(config_path: Path) -> tuple[Path, dict, dict, dict]:
    """Load and validate pipeline configuration from a YAML file.

    Returns
    -------
    data_root : Path
        Root directory containing construct data folders.
    construct_registry : dict
        Mapping of construct name → {plasmid, burst_ch, track_ch}.
    params : dict
        Analysis parameters (flat dict, same keys as the old PARAMS).
    plot_params : dict
        Plot parameters (flat dict, same keys as the old PLOT_PARAMS).

    Raises
    ------
    ValueError
        If required sections, construct fields, or analysis keys are missing.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate top-level sections
    missing = _REQUIRED_SECTIONS - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    # Validate constructs
    constructs = cfg["constructs"]
    for name, fields in constructs.items():
        missing_fields = _REQUIRED_CONSTRUCT_KEYS - set(fields.keys())
        if missing_fields:
            raise ValueError(
                f"Construct '{name}' missing required fields: {missing_fields}"
            )

    # Validate analysis params
    analysis = cfg["analysis"]
    missing_params = _REQUIRED_ANALYSIS_KEYS - set(analysis.keys())
    if missing_params:
        raise ValueError(f"Config 'analysis' missing required keys: {missing_params}")

    return Path(cfg["data_root"]), constructs, analysis, cfg["plots"]


_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

DATA_ROOT, CONSTRUCT_REGISTRY, PARAMS, PLOT_PARAMS = load_config(_DEFAULT_CONFIG)

# Default output root — overridden in main() based on SNR threshold
OUTPUT_ROOT = repo_root / "time_courses" / "results" / "burst_quantification"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DETREND DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def run_detrend_diagnostic():
    """Plot population-mean raw trace for each construct (ch0) to check
    for slow decay that would require detrending."""
    set_publication_style()
    print("\n" + "=" * 70)
    print("STEP 1: Detrend Diagnostic (ch0 = folding channel)")
    print("=" * 70)

    diag_dir = OUTPUT_ROOT / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharex=True, facecolor="white")
    axes = axes.flat

    for idx, (construct_name, meta) in enumerate(CONSTRUCT_REGISTRY.items()):
        plasmid = meta["plasmid"]
        full = REPORTER_PLASMID_NAME_MAPPING[plasmid]
        burst_ch = meta["burst_ch"]

        data_folder = DATA_ROOT / construct_name / "results"
        print(f"\n  Loading {full} ({plasmid}) ch{burst_ch} from {data_folder.name}...")

        try:
            matrix_ch0, _, _ = _load_construct_matrix(
                data_folder, burst_ch, verbose=False
            )
        except Exception as e:
            print(f"    ERROR loading {construct_name}: {e}")
            continue

        n_traces, n_time = matrix_ch0.shape
        t_min = np.arange(n_time) * PARAMS["time_interval_seconds"] / 60.0

        mean_trace = np.nanmean(matrix_ch0, axis=0)
        n_valid = np.sum(np.isfinite(matrix_ch0), axis=0)
        sem_trace = np.nanstd(matrix_ch0, axis=0) / np.sqrt(np.maximum(n_valid, 1))

        ax = axes[idx]
        ax.plot(t_min, mean_trace, linewidth=2.0, color=TRACE_MAGENTA)
        ax.fill_between(t_min, mean_trace - sem_trace, mean_trace + sem_trace,
                        alpha=0.18, color=TRACE_MAGENTA, linewidth=0)
        ax.set_title(f"{full} (n={n_traces})", fontsize=10)
        ax.set_ylabel("Raw Intensity (ch0)")
        style_axes(ax, grid=False, spine_width=1.2, tick_size=10, label_size=11, title_size=11)
        if idx >= 2:
            ax.set_xlabel("Time (min)")
        print(f"    ✓ {n_traces} trajectories × {n_time} timepoints")

    fig.suptitle("Detrend Diagnostic - Population Mean (Folding Channel)",
                 fontsize=14, fontname="Arial")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, diag_dir / "detrend_diagnostic", dpi=PLOT_PARAMS["plot_dpi"])
    print(f"\n  Diagnostic saved to {diag_dir}")


def run_signal_contrast_diagnostic():
    """Plot per-trace max/median ratio to validate that the 0.05 × max(FI)
    threshold is appropriate (requires bimodal signal: high bursts vs.
    near-zero background, giving max/median >> 1)."""
    set_publication_style()
    print("\n" + "=" * 70)
    print("STEP 1.5: Signal Contrast Diagnostic (max/median ratio)")
    print("=" * 70)

    diag_dir = OUTPUT_ROOT / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), facecolor="white")
    axes = axes.flat

    for idx, (construct_name, meta) in enumerate(CONSTRUCT_REGISTRY.items()):
        plasmid = meta["plasmid"]
        full = REPORTER_PLASMID_NAME_MAPPING[plasmid]
        burst_ch = meta["burst_ch"]

        data_folder = DATA_ROOT / construct_name / "results"
        try:
            matrix_ch0, _, _ = _load_construct_matrix(data_folder, burst_ch, verbose=False)
        except Exception as e:
            print(f"    ERROR loading {construct_name}: {e}")
            continue

        # Compute per-trace max/median ratio
        ratios = []
        for i in range(matrix_ch0.shape[0]):
            row = matrix_ch0[i]
            finite = row[np.isfinite(row)]
            if finite.size > 0 and np.median(finite) > 0:
                ratios.append(np.max(finite) / np.median(finite))
        ratios = np.array(ratios)

        ax = axes[idx]
        ax.hist(ratios, bins=30, color=TRACE_MAGENTA, edgecolor="black", linewidth=0.4, alpha=0.78)
        ax.axvline(x=2.0, color=TRACE_GRAY, linestyle="--", linewidth=1.4,
                   label="Ratio = 2 (guideline)")
        median_ratio = np.median(ratios) if len(ratios) > 0 else 0
        ax.axvline(x=median_ratio, color=TRACE_BLUE, linestyle="-", linewidth=1.7,
                   label=f"Median = {median_ratio:.1f}")
        ax.set_title(f"{full} (n={len(ratios)})", fontsize=10)
        ax.set_xlabel("max(FI) / median(FI)")
        ax.set_ylabel("Count")
        style_axes(ax, grid=False, spine_width=1.2, tick_size=10, label_size=11, title_size=11)
        ax.legend(fontsize=8, frameon=False)
        print(f"  {full}: median ratio = {median_ratio:.2f}, "
              f"fraction > 2: {np.mean(ratios > 2):.1%}")

    fig.suptitle("Signal Contrast: max/median Ratio per Trace (ch0)\n"
                 "Ratio >> 1 validates 0.05 × max threshold",
                 fontsize=13, fontname="Arial")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, diag_dir / "signal_contrast_diagnostic", dpi=PLOT_PARAMS["plot_dpi"])
    print(f"  Diagnostic saved to {diag_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — PER-CONSTRUCT BURST QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _shift_survival_mask(matrix, min_valid_fraction, max_missing_frames):
    """Replicate MicroLive shift_trajectories row filtering."""
    matrix = np.asarray(matrix, dtype=float)
    n_time = matrix.shape[1]
    max_nans_allowed = int(round(n_time * (1 - min_valid_fraction)))
    row_nan_counts = np.isnan(matrix).sum(axis=1)
    mask_relative = row_nan_counts <= max_nans_allowed

    if max_missing_frames is None:
        return mask_relative

    def count_internal_nans(row):
        valid = np.where(~np.isnan(row))[0]
        if valid.size == 0:
            return np.inf
        first, last = valid[0], valid[-1]
        return int(np.isnan(row[first:last + 1]).sum())

    mask_absolute = np.array(
        [count_internal_nans(row) <= max_missing_frames for row in matrix],
        dtype=bool,
    )
    return mask_relative & mask_absolute


def _resolve_min_valid_frames(params):
    """Derive the absolute minimum number of valid frames per trajectory.

    If ``min_valid_frames`` is explicitly set in *params*, return that.
    Otherwise derive it as ``ceil(max_frames * min_valid_fraction)``.
    Returns *None* if ``max_frames`` is also ``None``.
    """
    explicit = params.get("min_valid_frames")
    if explicit is not None:
        return int(explicit)
    max_frames = params.get("max_frames")
    frac = params["min_valid_fraction"]
    if max_frames is None:
        return None
    return int(np.ceil(int(max_frames) * float(frac)))


def _shift_pair_by_reference(
    reference, companion, min_valid_fraction, max_missing_frames,
):
    """Left-align two matrices using shift offsets from *reference* only.

    Both matrices are filtered using the same row mask (computed from
    *reference*) and shifted by the same per-row offset (first valid
    column in *reference*).  This guarantees that for every row *i*
    and column *j*, both returned matrices refer to the same original
    time-point.

    Returns
    -------
    ref_shifted, comp_shifted : ndarray
        Same shape, same row order, same per-row time alignment.
    survival_mask : ndarray of bool
        Row mask applied to the original matrices.
    """
    ref = np.asarray(reference, dtype=float)
    comp = np.asarray(companion, dtype=float)
    if ref.shape != comp.shape:
        raise ValueError(
            f"Shape mismatch: reference {ref.shape} vs companion {comp.shape}"
        )

    # Row filtering: use reference NaN pattern only
    survival = _shift_survival_mask(ref, min_valid_fraction, max_missing_frames)
    ref = ref[survival]
    comp = comp[survival]

    # Compute per-row shift offsets from reference and apply to both
    n_rows, n_cols = ref.shape
    ref_shifted = np.full_like(ref, np.nan)
    comp_shifted = np.full_like(comp, np.nan)

    for i in range(n_rows):
        valid = np.where(~np.isnan(ref[i]))[0]
        if valid.size == 0:
            continue
        offset = valid[0]
        length = n_cols - offset
        ref_shifted[i, :length] = ref[i, offset:]
        comp_shifted[i, :length] = comp[i, offset:]

    # Trim to last valid column in reference
    col_has_data = np.any(np.isfinite(ref_shifted), axis=0)
    if np.any(col_has_data):
        last = len(col_has_data) - np.argmax(col_has_data[::-1])
        ref_shifted = ref_shifted[:, :last]
        comp_shifted = comp_shifted[:, :last]

    return ref_shifted, comp_shifted, survival


def _load_construct_matrix(data_folder, channel_index, verbose=True):
    """Load all tracking CSVs from a construct's results folder and extract
    the intensity matrix *and* per-frame SNR matrix for the given channel.

    Both matrices are built in the same per-particle loop and survive
    identical row filtering (SNR-ch1 wipe, padding, shift_trajectories)
    so row indices are always synchronised.

    Returns
    -------
    combined : ndarray
        Shifted/filtered intensity matrix.
    snr_combined : ndarray
        Shifted/filtered SNR matrix (same shape as ``combined``).
    particle_origins : list[ParticleOrigin]
        Provenance records parallel to the rows of both matrices.
    """
    data_folder = Path(data_folder)
    construct_name = data_folder.parent.name
    results_dirs = sorted([
        d for d in data_folder.iterdir()
        if d.is_dir() and d.name.startswith("results_")
    ])

    if not results_dirs:
        raise FileNotFoundError(f"No results_* folders in {data_folder}")

    all_matrices = []
    all_snr_matrices = []
    all_origins = []
    field = f"spot_int_ch_{channel_index}"
    snr_field_burst = f"snr_ch_{channel_index}"   # SNR for the burst channel

    for rdir in results_dirs:
        csv_files = list(rdir.glob("tracking_*.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[0]

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if verbose:
                print(f"    WARNING: could not read {csv_path.name}: {e}")
            continue

        if field not in df.columns:
            if verbose:
                print(f"    WARNING: {field} not in {csv_path.name}, skipping")
            continue

        try:
            metadata = parse_metadata_txt(rdir)
        except Exception as e:
            if verbose:
                print(f"    WARNING: could not parse metadata in {rdir.name}: {e}")
            metadata = {}

        raw_lif_path = metadata.get("lif_path_raw")
        try:
            lif_path = resolve_lif_path(raw_lif_path, DATA_ROOT, construct_name) if raw_lif_path else Path()
        except Exception as e:
            if verbose:
                print(f"    WARNING: could not resolve LIF for {rdir.name}: {e}")
            lif_path = Path(raw_lif_path) if raw_lif_path else Path()

        series_index = int(metadata.get("series_index", 0))
        dt = float(metadata.get("time_interval_seconds", PARAMS["time_interval_seconds"]) or PARAMS["time_interval_seconds"])
        if dt <= 0:
            dt = PARAMS["time_interval_seconds"]
        total_frames_movie = int(metadata.get("total_frames_movie", 0) or 0)

        # Build per-particle intensity + SNR matrices in lockstep
        particles = df["particle"].unique()
        if "frame" not in df.columns:
            continue

        # ── Cap to max_frames BEFORE any particle/SNR logic ──
        max_frames_cap = PARAMS.get("max_frames")
        original_total = int(df["frame"].max()) + 1
        if max_frames_cap is not None and max_frames_cap > 0:
            df = df[df["frame"] < max_frames_cap].copy()
            if df.empty:
                if verbose:
                    print(f"    FOV {rdir.name}: all frames beyond "
                          f"max_frames={max_frames_cap} — skipped")
                continue
            # Recompute particles after filtering
            particles = df["particle"].unique()

        total_frames = int(df["frame"].max()) + 1
        if verbose and max_frames_cap is not None and original_total > total_frames:
            print(f"    FOV {rdir.name}: capped {original_total} → {total_frames} frames")

        if total_frames_movie <= 0:
            total_frames_movie = total_frames
        # Also cap total_frames_movie for ParticleOrigin
        if max_frames_cap is not None and max_frames_cap > 0:
            total_frames_movie = min(total_frames_movie, max_frames_cap)

        matrix = np.full((len(particles), total_frames), np.nan)
        snr_matrix = np.full((len(particles), total_frames), np.nan)
        fov_origins = []

        snr_ch = PARAMS.get("snr_channel_index")
        if snr_ch is None:
            snr_ch = channel_index
        snr_field = f"snr_ch_{snr_ch}"

        for p_idx, p in enumerate(particles):
            sub = df[df["particle"] == p]
            frames = sub["frame"].values.astype(int)
            values = sub[field].values.astype(float)
            valid = frames < total_frames
            matrix[p_idx, frames[valid]] = values[valid]

            # Populate burst-channel SNR in lockstep
            if snr_field_burst in df.columns:
                snr_vals = sub[snr_field_burst].values.astype(float)
                snr_matrix[p_idx, frames[valid]] = snr_vals[valid]

            snr_pass = True
            if snr_field in df.columns:
                mean_snr = sub[snr_field].mean()
                snr_pass = not (np.isfinite(mean_snr) and mean_snr < PARAMS["min_snr"])

            finite_valid = valid & np.isfinite(values)
            if snr_pass and np.any(finite_valid):
                first_valid = int(np.min(frames[finite_valid]))
            else:
                first_valid = 0

            fov_origins.append(
                ParticleOrigin(
                    results_dir=rdir,
                    particle_id=int(p),
                    lif_path=lif_path,
                    series_index=series_index,
                    total_frames_movie=total_frames_movie,
                    first_valid_frame=first_valid,
                    time_interval_seconds=dt,
                )
            )

        # SNR filter — keyed on snr_channel_index (default ch1, the nascent
        # tracking channel) so QC reflects localization quality, not how
        # bright the folding signal happens to be.
        # Applied to BOTH matrices (Stage 2 sync).
        if snr_field in df.columns:
            for p_idx, p in enumerate(particles):
                sub = df[df["particle"] == p]
                mean_snr = sub[snr_field].mean()
                if mean_snr < PARAMS["min_snr"]:
                    matrix[p_idx, :] = np.nan
                    snr_matrix[p_idx, :] = np.nan
        if verbose:
            print(f"    FOV {rdir.name}: {matrix.shape[0]} particles × "
                  f"{matrix.shape[1]} frames")
        all_matrices.append(matrix)
        all_snr_matrices.append(snr_matrix)
        all_origins.extend(fov_origins)

    if not all_matrices:
        raise ValueError(f"No valid tracking data found in {data_folder}")

    # Concatenate all FOVs — pad to max frame count with NaN (Stage 3 sync)
    max_cols = max(m.shape[1] for m in all_matrices)
    padded = []
    padded_snr = []
    for m, s in zip(all_matrices, all_snr_matrices):
        if m.shape[1] < max_cols:
            pad_arr = np.full((m.shape[0], max_cols - m.shape[1]), np.nan)
            m = np.hstack([m, pad_arr])
            s = np.hstack([s, np.full((s.shape[0], max_cols - s.shape[1]), np.nan)])
        padded.append(m)
        padded_snr.append(s)
    combined = np.vstack(padded)
    snr_combined = np.vstack(padded_snr)

    # Safety net: enforce max_frames cap after concatenation
    max_frames_cap = PARAMS.get("max_frames")
    if max_frames_cap is not None and max_frames_cap > 0 and combined.shape[1] > max_frames_cap:
        combined = combined[:, :max_frames_cap]
        snr_combined = snr_combined[:, :max_frames_cap]
    assert combined.shape == snr_combined.shape, (
        f"Intensity/SNR shape mismatch after vstack: {combined.shape} vs {snr_combined.shape}"
    )
    if combined.shape[0] != len(all_origins):
        raise RuntimeError(
            f"Origin/matrix row mismatch before shifting: "
            f"{len(all_origins)} origins for {combined.shape[0]} rows"
        )
    if verbose:
        print(f"    Combined: {combined.shape[0]} trajectories × "
              f"{combined.shape[1]} frames (max across FOVs)")

    # Left-align and filter: compute shift offsets from intensity only,
    # apply the SAME per-row offset to SNR so every (row, col) pair in
    # both matrices refers to the same original time-point.
    combined, snr_combined, survival_mask = _shift_pair_by_reference(
        combined, snr_combined,
        PARAMS["min_valid_fraction"],
        PARAMS["max_total_internal_nan_frames"],
    )
    particle_origins = [origin for origin, keep in zip(all_origins, survival_mask) if keep]
    if combined.shape[0] != len(particle_origins):
        raise RuntimeError(
            f"Origin/matrix row mismatch after shifting: "
            f"{len(particle_origins)} origins for {combined.shape[0]} rows"
        )

    # ── Absolute minimum valid-frame floor ──
    # Ensures every surviving trajectory has at least
    # ceil(max_frames × min_valid_fraction) finite values (e.g. 108 of 360).
    min_valid_frames = _resolve_min_valid_frames(PARAMS)
    if min_valid_frames is not None:
        n_valid_per_row = np.sum(np.isfinite(combined), axis=1)
        keep = n_valid_per_row >= min_valid_frames
        n_dropped = int((~keep).sum())
        if n_dropped > 0 and verbose:
            print(f"    Absolute floor: dropped {n_dropped} trajectories "
                  f"with < {min_valid_frames} valid frames")
        combined = combined[keep]
        snr_combined = snr_combined[keep]
        particle_origins = [o for o, k in zip(particle_origins, keep) if k]

    if verbose:
        print(f"    After shift/filter: {combined.shape[0]} trajectories × "
              f"{combined.shape[1]} frames")

    # NOTE: Do NOT forward-fill here. The burst module handles forward-fill
    # internally with proper NaN re-stamping so filled frames don't become
    # false burst/dwell evidence.
    return combined, snr_combined, particle_origins


def run_per_construct_analysis():
    """Run burst quantification on each construct independently."""
    print("\n" + "=" * 70)
    print("STEP 2: Per-Construct Burst Quantification (ch0 = folding)")
    print("=" * 70)

    all_results = {}

    for construct_name, meta in CONSTRUCT_REGISTRY.items():
        plasmid = meta["plasmid"]
        burst_ch = meta["burst_ch"]
        short = PLASMID_SHORT_NAME_MAPPING[plasmid]
        full = REPORTER_PLASMID_NAME_MAPPING[plasmid]
        output_dir = OUTPUT_ROOT / short

        print(f"\n{'─' * 60}")
        print(f"CONSTRUCT: {full}  ({plasmid} / {short})")
        print(f"  Data:   {DATA_ROOT / construct_name / 'results'}")
        print(f"  Output: {output_dir}")
        print(f"{'─' * 60}")

        # Load folding channel (ch0) — returns intensity + SNR matrices
        try:
            matrix_ch0, snr_ch0, origins = _load_construct_matrix(
                DATA_ROOT / construct_name / "results",
                burst_ch,
                verbose=True,
            )
        except Exception as e:
            print(f"  ERROR loading {construct_name}: {e}")
            continue

        print(f"  Loaded: {matrix_ch0.shape[0]} trajectories × {matrix_ch0.shape[1]} timepoints")

        # Save raw matrices for reproducibility
        output_dir.mkdir(parents=True, exist_ok=True)
        quant_dir = output_dir / "quantification"
        quant_dir.mkdir(parents=True, exist_ok=True)
        np.save(quant_dir / "raw_matrix.npy", matrix_ch0)
        np.save(quant_dir / "snr_matrix.npy", snr_ch0)

        # Save provenance for reproducibility
        provenance = {
            "max_frames": PARAMS.get("max_frames"),
            "matrix_shape": list(matrix_ch0.shape),
            "time_interval_seconds": PARAMS["time_interval_seconds"],
        }
        (quant_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2)
        )

        # Run burst quantification
        result = run_burst_quantification(
            input_matrix=matrix_ch0,
            snr_matrix=snr_ch0,
            output_dir=output_dir,
            condition=full,
            **{k: v for k, v in PARAMS.items()
               if k not in ("min_snr", "snr_channel_index",
                            "max_total_internal_nan_frames",
                            "min_valid_frames",
                            "max_frames",
                            "align_first_valid")
               and not k.startswith("montage_")},
            align_first_valid=False,  # already aligned by _shift_pair_by_reference
            **{k: v for k, v in PLOT_PARAMS.items()
               if not k.startswith("montage_") and k not in ("use_bh_fdr",)},
        )

        all_results[short] = {
            "burst": result,
            "origins": origins,
            "output_dir": output_dir,
            "data_folder": DATA_ROOT / construct_name,
            "construct_name": construct_name,
        }

        # Print key stats
        ts = result["trajectory_summary"]
        if not ts.empty:
            n_bursts = int(ts["n_bursts"].sum())
            n_dwells = int(ts["n_dwells"].sum())
            mean_frac = ts["fraction_time_on"].mean()
            print(f"  ✓ {n_bursts} bursts, {n_dwells} dwells, "
                  f"mean fraction ON: {mean_frac:.3f}")
        else:
            print("  ⚠ No trajectories passed QC")



        # ── Load ch1 (nascent) for kymograph plots ──
        # Captures both intensity and SNR matrices, plus origins for
        # provenance verification against ch0.
        nascent_ch = 1  # channel index for nascent
        try:
            matrix_ch1_raw, snr_ch1_raw, origins_ch1 = _load_construct_matrix(
                DATA_ROOT / construct_name / "results",
                nascent_ch,
                verbose=False,
            )
        except Exception as e:
            print(f"  ⚠ ch1 loading failed — skipping kymographs: {e}")
            continue

        # ── Provenance assertion: ch0 and ch1 must have identical rows ──
        if len(origins) != len(origins_ch1):
            print(
                f"  ⚠ ch0/ch1 origin count mismatch "
                f"({len(origins)} vs {len(origins_ch1)}) — skipping kymographs"
            )
            continue
        provenance_ok = True
        for i_prov, (o0, o1) in enumerate(zip(origins, origins_ch1)):
            if (o0.results_dir, o0.particle_id) != (o1.results_dir, o1.particle_id):
                print(
                    f"  ⚠ ch0/ch1 provenance mismatch at row {i_prov}: "
                    f"{o0.results_dir.name}/p{o0.particle_id} vs "
                    f"{o1.results_dir.name}/p{o1.particle_id} — skipping kymographs"
                )
                provenance_ok = False
                break
        if not provenance_ok:
            continue

        # Get QC-passed row indices (same rows kept for ch0 in burst quant)
        qc = result["qc_table"]
        keep_idx = qc[qc["qc_status"] == "kept"]["trajectory_index"].values

        # ── Dual-channel INTENSITY kymograph (Green=Folding ch0, Magenta=Nascent ch1) ──
        try:
            # Filter ch1 to the same rows that ch0 kept
            keep_idx_valid = keep_idx[keep_idx < matrix_ch1_raw.shape[0]]
            ch1_kept = matrix_ch1_raw[keep_idx_valid]

            # ch0 post-QC from burst result
            ch0_kept = result["processed_matrix"]

            # Trim/pad columns to match
            n_cols = ch0_kept.shape[1]
            if ch1_kept.shape[1] > n_cols:
                ch1_kept = ch1_kept[:, :n_cols]
            elif ch1_kept.shape[1] < n_cols:
                pad = np.full(
                    (ch1_kept.shape[0], n_cols - ch1_kept.shape[1]),
                    np.nan,
                )
                ch1_kept = np.hstack([ch1_kept, pad])

            # Row-match (should already be equal, but safety)
            n_rows = min(ch0_kept.shape[0], ch1_kept.shape[0])
            plot_dual_channel_kymograph_from_matrix(
                ch0_matrix=ch0_kept[:n_rows],
                ch1_matrix=ch1_kept[:n_rows],
                output_dir=output_dir,
                time_interval_seconds=PARAMS["time_interval_seconds"],
                condition=full,
                sort_by=PLOT_PARAMS.get("kymograph_sort_by", "density"),
                trajectory_summary=result["trajectory_summary"].iloc[:n_rows],
                max_traces_to_plot=PLOT_PARAMS.get("max_traces_to_plot", None),
                figsize=PLOT_PARAMS.get("kymograph_figsize", (8.5, 4.2)),
                dpi=PLOT_PARAMS.get("kymograph_dpi", 300),
            )
            print(f"  ✓ Dual-channel kymograph saved ({n_rows} QC-passed trajectories)")
        except Exception as e:
            print(f"  ⚠ Dual-channel kymograph skipped: {e}")

        # ── Dual-channel SNR kymograph (Green=Folding ch0, Magenta=Nascent ch1) ──
        # Uses fixed-range normalization so absolute SNR values are preserved:
        # SNR=0 → black, SNR=SNR_CAP → full brightness.
        try:
            if keep_idx.size == 0:
                print("  ⚠ No QC-passing rows for SNR kymograph — skipped")
            else:
                # Validate keep_idx against both pre-QC SNR matrices
                if keep_idx.max() >= snr_ch0.shape[0]:
                    raise RuntimeError(
                        f"keep_idx max ({keep_idx.max()}) exceeds "
                        f"snr_ch0 row count ({snr_ch0.shape[0]})"
                    )
                if keep_idx.max() >= snr_ch1_raw.shape[0]:
                    raise RuntimeError(
                        f"keep_idx max ({keep_idx.max()}) exceeds "
                        f"snr_ch1_raw row count ({snr_ch1_raw.shape[0]})"
                    )

                snr_ch0_kept = snr_ch0[keep_idx]
                snr_ch1_kept = snr_ch1_raw[keep_idx]

                # Trim/pad columns to match
                n_cols_snr = snr_ch0_kept.shape[1]
                if snr_ch1_kept.shape[1] > n_cols_snr:
                    snr_ch1_kept = snr_ch1_kept[:, :n_cols_snr]
                elif snr_ch1_kept.shape[1] < n_cols_snr:
                    pad_snr = np.full(
                        (snr_ch1_kept.shape[0], n_cols_snr - snr_ch1_kept.shape[1]),
                        np.nan,
                    )
                    snr_ch1_kept = np.hstack([snr_ch1_kept, pad_snr])

                # Assert shape equality after padding/trimming
                if snr_ch0_kept.shape != snr_ch1_kept.shape:
                    raise RuntimeError(
                        f"SNR shape mismatch after padding: "
                        f"ch0={snr_ch0_kept.shape} vs ch1={snr_ch1_kept.shape}"
                    )

                n_rows_snr = snr_ch0_kept.shape[0]
                snr_cap = float(PARAMS.get("threshold", 3.0)) * 2  # e.g. 6.0
                plot_dual_channel_kymograph_from_matrix(
                    ch0_matrix=snr_ch0_kept,
                    ch1_matrix=snr_ch1_kept,
                    output_dir=output_dir,
                    time_interval_seconds=PARAMS["time_interval_seconds"],
                    condition=f"{full} (SNR)",
                    normalize="fixed_range",
                    p_lo=0,
                    p_hi=snr_cap,
                    sort_by=PLOT_PARAMS.get("kymograph_sort_by", "density"),
                    trajectory_summary=result["trajectory_summary"].iloc[:n_rows_snr],
                    max_traces_to_plot=PLOT_PARAMS.get("max_traces_to_plot", None),
                    figsize=PLOT_PARAMS.get("kymograph_figsize", (8.5, 4.2)),
                    dpi=PLOT_PARAMS.get("kymograph_dpi", 300),
                    filename_stem="kymograph_dual_channel_snr",
                )
                print(f"  ✓ SNR dual-channel kymograph saved ({n_rows_snr} QC-passed trajectories)")
        except Exception as e:
            print(f"  ⚠ SNR dual-channel kymograph skipped: {e}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2.5 — REPRESENTATIVE CELL-CROP MONTAGES
# ═══════════════════════════════════════════════════════════════════════════
#
# Data flow
# --------
# 1. For each construct, _select_representative_particles() picks N
#    particles from the burst trajectory_summary (half with the most
#    bursts, half closest to the median fraction-ON).
#
# 2. Each selection carries two indices:
#      • origin_row_index  — row in the pre-QC input matrix → indexes
#        into the ParticleOrigin list built during _load_construct_matrix.
#      • binary_row_index  — row in the post-QC binary_matrix → provides
#        the ON/OFF state vector for the figure's state bar.
#
# 3. For every selected particle, the raw LIF scene is loaded via
#    plotting_montage_crops.load_montage_data_cached (with optional
#    photobleaching correction), and a 3-panel figure is drawn:
#      Top     — dual-channel intensity trace (Nascent + Folding)
#      Middle  — compact ON/OFF step bar (from binary_matrix)
#      Bottom  — max-Z crop montage at evenly-spaced snapshot frames
#
# 4. Results are assembled into a multi-page PDF (2 montages per page).
#
# All parameters are read from PARAMS / PLOT_PARAMS defined in the
# CONFIGURATION block at the top of this file.
# ═══════════════════════════════════════════════════════════════════════════


def _build_selection(trajectory_summary_row, origins):
    """Map a single trajectory-summary row to its source data.

    Returns a dict with:
        origin            — ParticleOrigin (LIF path, series, results_dir, …)
        origin_row_index  — row in pre-QC input matrix (from trajectory_id)
        binary_row_index  — row in post-QC binary_matrix (from trajectory_index)
        trajectory_id     — e.g. "traj_3"
    """
    row = trajectory_summary_row
    trajectory_id = str(row["trajectory_id"])
    if not trajectory_id.startswith("traj_"):
        raise ValueError(f"Unexpected trajectory_id format: {trajectory_id}")

    # trajectory_id encodes the pre-QC matrix row (before burst filtering)
    origin_row_index = int(trajectory_id.removeprefix("traj_"))
    # trajectory_index is the post-QC row in binary_matrix
    binary_row_index = int(row["trajectory_index"])

    if origin_row_index >= len(origins):
        raise IndexError(
            f"trajectory_id {trajectory_id} maps to origins[{origin_row_index}], "
            f"but only {len(origins)} origins are available"
        )
    return {
        "origin": origins[origin_row_index],
        "origin_row_index": origin_row_index,
        "binary_row_index": binary_row_index,
        "trajectory_id": trajectory_id,
    }


def _select_representative_particles(trajectory_summary, origins, n=4):
    """Pick N representative trajectories for montage display.

    Selection strategy:
        • If ``n`` is None, select ALL trajectories sorted by trajectory
          length (longest first, by ``n_valid_timepoints``).
        • If ``n`` is an integer, select the top-N **longest** trajectories
          (most valid timepoints), consistent with the density-sorted
          kymograph convention.
    """
    if trajectory_summary.empty:
        return []

    sort_col = "n_valid_timepoints"
    if sort_col not in trajectory_summary.columns:
        # Fallback if column is missing (shouldn't happen in normal use)
        sort_col = "n_bursts"

    by_length = trajectory_summary.sort_values(
        [sort_col, "fraction_time_on"], ascending=[False, False],
    )

    # n=None → all trajectories, longest first
    if n is None:
        return [_build_selection(row, origins) for _, row in by_length.iterrows()]

    n = min(int(n), len(trajectory_summary))
    selected = by_length.head(n)

    return [_build_selection(row, origins) for _, row in selected.iterrows()]


def validate_montage_selection(selection, binary_matrix):
    """Print provenance for one selection and assert all paths/indices are valid.

    Call this before PDF generation as a dry-run check.  If any assertion
    fails, the error message identifies exactly which field is wrong.
    """
    origin = selection["origin"]
    bin_idx = int(selection["binary_row_index"])
    trajectory_id = str(selection["trajectory_id"])
    origin_idx = int(
        selection.get("origin_row_index", trajectory_id.removeprefix("traj_"))
    )
    binary_matrix = np.asarray(binary_matrix)

    if not (0 <= bin_idx < binary_matrix.shape[0]):
        raise IndexError(
            f"binary_row_index {bin_idx} out of range for "
            f"binary_matrix with {binary_matrix.shape[0]} rows"
        )

    row = binary_matrix[bin_idx]
    print(f"    trajectory_id:      {trajectory_id}")
    print(f"    origin_row_index:   {origin_idx}")
    print(f"    binary_row_index:   {bin_idx}")
    print(f"    lif_path:           {origin.lif_path}")
    print(f"    lif_path exists:    {origin.lif_path.exists()}")
    print(f"    series_index:       {origin.series_index}")
    print(f"    results_dir:        {origin.results_dir.name}")
    print(f"    particle_id:        {origin.particle_id}")
    print(f"    first_valid_frame:  {origin.first_valid_frame}")
    print(f"    total_frames_movie: {origin.total_frames_movie}")
    print(f"    time_interval:      {origin.time_interval_seconds}s")
    print(
        f"    binary_matrix row:  shape={row.shape}, "
        f"n_ON={int(np.nansum(row == 1))}, "
        f"n_OFF={int(np.nansum(row == 0))}"
    )

    # Assertions — any failure pinpoints the broken link
    if not origin.lif_path.exists():
        raise FileNotFoundError(f"LIF not found: {origin.lif_path}")
    if not origin.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {origin.results_dir}")
    if (
        origin.total_frames_movie > 0
        and origin.first_valid_frame >= origin.total_frames_movie
    ):
        raise ValueError(
            f"first_valid_frame {origin.first_valid_frame} is outside "
            f"movie length {origin.total_frames_movie}"
        )
    print("    ✓ All checks passed")


def generate_representative_montages(all_results):
    """Generate a multi-page PDF of representative crop montages per construct.

    Reads all montage-related parameters from PARAMS and PLOT_PARAMS,
    so adjustments are made in one place — the CONFIGURATION block.
    """
    print("\n" + "=" * 70)
    print("STEP 2.5: Representative Cell-Crop Montages")
    print("=" * 70)

    # ── Read parameters from the central configuration ──
    n_particles         = PARAMS["montage_n_particles"]
    apply_pb            = PARAMS["montage_apply_photobleaching"]
    pb_mode             = PARAMS["montage_photobleaching_mode"]
    n_snapshots         = PARAMS["montage_n_snapshots"]
    crop_size_px        = PARAMS["montage_crop_size_px"]
    gaussian_filter_val = PARAMS["montage_gaussian_filter_value"]
    coordinate_mode     = PARAMS["montage_coordinate_mode"]
    coordinate_max_gap  = PARAMS["montage_coordinate_max_gap"]
    crop_norm_mode      = PARAMS["montage_crop_norm_mode"]
    crop_colormap       = PARAMS.get("montage_crop_colormap", None)
    trace_norm_mode     = PARAMS["montage_trace_norm_mode"]
    smooth_window       = PARAMS["smooth_window"]
    height_ratios       = PARAMS["montage_section_height_ratios"]
    show_crop_time_labels = PARAMS.get("montage_show_crop_time_labels", True)
    trim_to_valid       = PARAMS.get("montage_trim_to_valid", True)
    display_3_averaged  = PARAMS.get("montage_display_3_crops_averaged", False)
    max_frames          = PARAMS.get("max_frames")
    montages_per_page   = PLOT_PARAMS["montage_panels_per_page"]
    panel_figsize       = PLOT_PARAMS.get("montage_panel_figsize", None)
    pdf_dpi             = PLOT_PARAMS.get("montage_pdf_dpi", 200)

    for construct_name, meta in CONSTRUCT_REGISTRY.items():
        plasmid = meta["plasmid"]
        short = PLASMID_SHORT_NAME_MAPPING[plasmid]
        entry = all_results.get(short)
        if entry is None:
            continue

        result = entry["burst"]
        origins = entry["origins"]
        trajectory_summary = result["trajectory_summary"]

        if trajectory_summary.empty:
            print(f"  {short}: no valid trajectories — skipping montage PDF")
            continue

        # Select representative particles
        selections = _select_representative_particles(
            trajectory_summary, origins, n=n_particles
        )
        if not selections:
            print(f"  {short}: no selections — skipping montage PDF")
            continue

        output_path = entry["output_dir"] / "plots" / f"montages_{short}.pdf"
        print(
            f"\n  {short}: {len(selections)} representative particles "
            f"→ {output_path.name}"
        )

        # Validate provenance before committing to LIF loading
        try:
            for selection in selections:
                validate_montage_selection(selection, result["binary_matrix"])

            # Generate the PDF — all visual/crop params forwarded here
            generate_representative_montage_pdf(
                selections=selections,
                binary_matrix=result["binary_matrix"],
                output_path=output_path,
                apply_photobleaching=apply_pb,
                photobleaching_mode=pb_mode,
                montages_per_page=montages_per_page,
                n_snapshots=n_snapshots,
                panel_figsize=panel_figsize,
                pdf_dpi=pdf_dpi,
                save_individual_montages=PLOT_PARAMS.get("montage_save_individual_montages", True),
                save_combined_pdf=PLOT_PARAMS.get("montage_save_combined_pdf", True),
                verbose=True,
                max_frames=max_frames,
                # Visual kwargs forwarded to plot_cell_crop_timecourse_montage
                crop_size_px=crop_size_px,
                gaussian_filter_value=gaussian_filter_val,
                coordinate_mode=coordinate_mode,
                coordinate_max_frame_distance=coordinate_max_gap,
                crop_norm_mode=crop_norm_mode,
                crop_colormap=crop_colormap,
                trace_norm_mode=trace_norm_mode,
                smooth_window=smooth_window,
                section_height_ratios=height_ratios,
                show_crop_time_labels=show_crop_time_labels,
                trim_to_valid=trim_to_valid,
                display_3_crops_averaged=display_3_averaged,
            )
        except Exception as e:
            print(
                f"  WARNING: representative montage generation failed "
                f"for {short}: {e}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — CROSS-CONSTRUCT COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_construct_comparison(all_results):
    """Generate comparison plots across all constructs."""
    set_publication_style()
    print("\n" + "=" * 70)
    print("STEP 3: Cross-Construct Comparison")
    print("=" * 70)

    comp_dir = OUTPUT_ROOT / "comparison" / "run_analysis"
    plots_dir = comp_dir / "plots"
    quant_dir = comp_dir / "quantification"
    plots_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    constructs = []
    burst_durs = {}
    dwell_durs = {}
    frac_on_vals = {}
    n_cells = {}          # unique FOVs (= cells)
    n_trajectories = {}   # QC-passing trajectories
    n_on_events = {}      # burst events (passing duration filter)
    n_off_events = {}     # dwell events (non-terminal)

    for short, entry in all_results.items():
        result = entry["burst"] if isinstance(entry, dict) and "burst" in entry else entry
        origins = entry.get("origins", []) if isinstance(entry, dict) else []
        ts = result["trajectory_summary"]
        et = result["event_table"]
        if ts.empty:
            continue
        constructs.append(short)
        frac_on_vals[short] = ts["fraction_time_on"].values
        n_trajectories[short] = len(ts)

        # Count cells: each unique results_dir in origins = 1 FOV = 1 cell.
        # origins is parallel to the pre-QC matrix; trajectory_summary rows
        # survived QC.  Map QC-passing trajectory IDs back to origins.
        if origins:
            qc_origin_indices = set()
            for _, row in ts.iterrows():
                tid = row.get("trajectory_id", "")
                if isinstance(tid, str) and tid.startswith("traj_"):
                    qc_origin_indices.add(int(tid.removeprefix("traj_")))
            unique_dirs = set()
            for idx in qc_origin_indices:
                if idx < len(origins):
                    unique_dirs.add(origins[idx].results_dir)
            n_cells[short] = len(unique_dirs)
        else:
            n_cells[short] = 0

        bursts = et[(et["event_type"] == "burst") & et["passes_duration_filter"]]
        dwells = et[
            (et["event_type"] == "dwell")
            & ~et["is_terminal_event"]
            & ~et["is_initial_dwell"]
        ]

        # Trajectory-level median durations (one value per trajectory)
        burst_durs[short] = (
            bursts.groupby("trajectory_id")["duration_minutes"].median().values
            if not bursts.empty else np.array([])
        )
        dwell_durs[short] = (
            dwells.groupby("trajectory_id")["duration_minutes"].median().values
            if not dwells.empty else np.array([])
        )
        n_on_events[short] = len(bursts)
        n_off_events[short] = len(dwells)

    if not constructs:
        print("  No constructs with valid data — skipping comparison")
        return

    # Map short names to full names for titles
    short_to_full = {}
    for _, meta in CONSTRUCT_REGISTRY.items():
        p = meta["plasmid"]
        short_to_full[PLASMID_SHORT_NAME_MAPPING[p]] = REPORTER_PLASMID_NAME_MAPPING[p]

    figsize = PLOT_PARAMS["comparison_figsize"]
    stats_rows = []

    def _build_xlabels(constructs, event_counts=None):
        """Build multi-line x-axis labels: name / cells / traj / (events)."""
        labels = []
        for c in constructs:
            parts = [
                c,
                f"{n_cells.get(c, 0)} cells",
                f"{n_trajectories.get(c, 0)} traj",
            ]
            if event_counts is not None:
                parts.append(f"({event_counts.get(c, 0)} events)")
            labels.append("\n".join(parts))
        return labels

    # ── 1. ON episode duration comparison ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [burst_durs.get(c, []) for c in constructs]
    xlabels_on = _build_xlabels(constructs, n_on_events)
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Observed ON Episode Duration (min)",
        xlabels=xlabels_on,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
        use_bh_fdr=PLOT_PARAMS.get("use_bh_fdr", False),
    )
    for row in stats:
        stats_rows.append({"metric": "on_duration_minutes", **row})
    fig.tight_layout()
    save_figure(fig, plots_dir / "burst_duration_comparison", PLOT_PARAMS["plot_dpi"])

    # ── 2. OFF episode duration comparison ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [dwell_durs.get(c, []) for c in constructs]
    xlabels_off = _build_xlabels(constructs, n_off_events)
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Observed OFF Episode Duration (min)",
        xlabels=xlabels_off,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
        use_bh_fdr=PLOT_PARAMS.get("use_bh_fdr", False),
    )
    for row in stats:
        stats_rows.append({"metric": "off_duration_minutes", **row})
    fig.tight_layout()
    save_figure(fig, plots_dir / "dwell_duration_comparison", PLOT_PARAMS["plot_dpi"])

    # ── 3. Fraction of observed time ON ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [frac_on_vals.get(c, []) for c in constructs]
    xlabels_frac = _build_xlabels(constructs)  # no event count for fraction
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Fraction of Observed Time ON",
        ylim=(-0.05, 1.05),
        xlabels=xlabels_frac,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
        use_bh_fdr=PLOT_PARAMS.get("use_bh_fdr", False),
    )
    for row in stats:
        stats_rows.append({"metric": "fraction_time_on", **row})
    fig.tight_layout()
    save_figure(fig, plots_dir / "fraction_on_comparison", PLOT_PARAMS["plot_dpi"])

    stats_df = pd.DataFrame(stats_rows)
    stats_path = quant_dir / "pairwise_mannwhitney_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # ── 4. Summary table ──
    summary_rows = []
    for short in constructs:
        entry = all_results[short]
        result = entry["burst"] if isinstance(entry, dict) and "burst" in entry else entry
        ts = result["trajectory_summary"]
        full = short_to_full.get(short, short)
        bd = burst_durs.get(short, np.array([]))
        dd = dwell_durs.get(short, np.array([]))
        fo = frac_on_vals.get(short, np.array([]))

        summary_rows.append({
            "construct": full,
            "short_name": short,
            "n_cells": n_cells.get(short, 0),
            "n_trajectories": len(ts),
            "n_bursts": int(ts["n_bursts"].sum()),
            "n_dwells": int(ts["n_dwells"].sum()),
            "mean_burst_dur_min": round(float(np.mean(bd)), 3) if len(bd) > 0 else np.nan,
            "median_burst_dur_min": round(float(np.median(bd)), 3) if len(bd) > 0 else np.nan,
            "std_burst_dur_min": round(float(np.std(bd, ddof=1)), 3) if len(bd) > 1 else np.nan,
            "sem_burst_dur_min": round(float(np.std(bd, ddof=1) / np.sqrt(len(bd))), 3) if len(bd) > 1 else np.nan,
            "mean_dwell_dur_min": round(float(np.mean(dd)), 3) if len(dd) > 0 else np.nan,
            "median_dwell_dur_min": round(float(np.median(dd)), 3) if len(dd) > 0 else np.nan,
            "std_dwell_dur_min": round(float(np.std(dd, ddof=1)), 3) if len(dd) > 1 else np.nan,
            "sem_dwell_dur_min": round(float(np.std(dd, ddof=1) / np.sqrt(len(dd))), 3) if len(dd) > 1 else np.nan,
            "mean_fraction_on": round(float(np.mean(fo)), 4) if len(fo) > 0 else np.nan,
            "median_fraction_on": round(float(np.median(fo)), 4) if len(fo) > 0 else np.nan,
            "std_fraction_on": round(float(np.std(fo, ddof=1)), 4) if len(fo) > 1 else np.nan,
            "sem_fraction_on": round(float(np.std(fo, ddof=1) / np.sqrt(len(fo))), 4) if len(fo) > 1 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(quant_dir / "summary_table.csv", index=False)
    print(f"\n  Summary table saved to {quant_dir / 'summary_table.csv'}")
    print(f"  Pairwise Mann-Whitney statistics saved to {stats_path}")
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\n  Comparison plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Burst Quantification Analysis — CoF Long Movies",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Path to YAML config file (default: ./config.yaml)",
    )
    args = parser.parse_args()

    # Reload config if a non-default path was given
    global DATA_ROOT, CONSTRUCT_REGISTRY, PARAMS, PLOT_PARAMS, OUTPUT_ROOT
    if args.config != _DEFAULT_CONFIG:
        DATA_ROOT, CONSTRUCT_REGISTRY, PARAMS, PLOT_PARAMS = load_config(args.config)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Burst Quantification Analysis — CoF Long Movies          ║")
    print("║   Channel 0 = folding (bursting)                           ║")
    print("║   Channel 1 = nascent protein (tracking)                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Config: {args.config}")

    # Verify data drive
    if not DATA_ROOT.exists():
        print(f"\n  ERROR: Data root not found: {DATA_ROOT}")
        print("  Please mount the external drive and try again.")
        sys.exit(1)

    # ── Build output directory from the SNR threshold in PARAMS ──
    snr_val = PARAMS["threshold"]
    # Format: results_snr_3 for 3.0, results_snr_2,_5 for 2.5, etc.
    snr_str = str(snr_val).replace(".", ",_") if snr_val != int(snr_val) else str(int(snr_val))
    output_dir_name = f"results_snr_{snr_str}"

    OUTPUT_ROOT = repo_root / "time_courses" / output_dir_name
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n  SNR threshold = {snr_val}  →  {output_dir_name}/")
    print("=" * 70)

    # Step 1: Detrend diagnostic
    run_detrend_diagnostic()

    # Step 1.5: Signal contrast diagnostic
    run_signal_contrast_diagnostic()

    # Step 2: Per-construct burst quantification
    all_results = run_per_construct_analysis()

    # Step 2.5: Representative crop montages
    if all_results:
        generate_representative_montages(all_results)

    # Step 3: Cross-construct comparison
    if all_results:
        run_cross_construct_comparison(all_results)

    plt.close("all")

    print("\n" + "=" * 70)
    print(f"DONE. Results → {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
