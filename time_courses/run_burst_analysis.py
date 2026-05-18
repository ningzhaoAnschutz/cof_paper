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
    python run_burst_analysis.py

Note:
    This script requires the ``microlive`` conda environment.
    Activate it before running:  ``conda activate microlive``
"""
from __future__ import annotations

import sys
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
from generate_all_traces_pdf import generate_pdf
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DATA_ROOT = Path("/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF_long_movies")
OUTPUT_ROOT = repo_root / "time_courses" / "results" / "burst_quantification"

# Construct registry: burst_ch=0 (folding), track_ch=1 (nascent protein)
CONSTRUCT_REGISTRY = {
    "sfGFP":      {"plasmid": "pRS027", "burst_ch": 0, "track_ch": 1},
    "GFPuv":      {"plasmid": "pRS032", "burst_ch": 0, "track_ch": 1},
    "sfGFP_Xbp1": {"plasmid": "pRS038", "burst_ch": 0, "track_ch": 1},
    "Xbp1_sfGFP": {"plasmid": "pRS048", "burst_ch": 0, "track_ch": 1},
}

# Analysis parameters — single source of truth for the entire pipeline.
# Loader keys (SNR filter, shift_trajectories) and burst-module keys
# (threshold, smoothing, normalization) are all here.
PARAMS = dict(
    # ── Time ──
    time_interval_seconds=5.0,
    # ── Loader: SNR filter ──
    # SNR is checked on the tracking channel (ch1) so QC reflects
    # localization quality, not folding-signal brightness.
    min_snr=3,
    snr_channel_index=1,           # None → filter on the loaded channel
    # ── Loader: shift_trajectories ──
    shift_min_data_fraction=0.4,   # min fraction of finite frames to keep
    shift_max_missing_frames=3,    # max total internal NaN frames
    # ── Burst module: QC ──
    min_valid_fraction=0.50,
    max_internal_nan_gap=2,
    align_first_valid=True,
    # ── Burst module: smoothing ──
    detrend_method=None,
    smooth_method="median",
    smooth_window=3,
    # ── Burst module: normalization (for kymograph visualization) ──
    normalization_method="per_trace_percentile",
    percentile_low=5,
    percentile_high=95,
    # ── Burst module: thresholding ──
    # SNR-based ON/OFF: frames with per-frame SNR (folding ch) >= threshold
    # are ON.  SNR is self-normalised by local noise, so the same cutoff
    # works across constructs with different absolute brightness.
    threshold_mode="snr",
    threshold=3.5,                     # SNR cutoff (standard microscopy detection)
    off_baseline_quantile=0.5,        # unused in snr mode, kept for reference
    # ── Burst module: event cleanup ──
    min_event_duration_frames=3,
    min_burst_duration_seconds=30.0,
    exclude_terminal_dwell=True,
    count_initial_dwell=True,
    # ── Step 2.5: Representative montage ──
    # These control which particles are selected and how the crop montage
    # is generated.  Photobleaching correction is applied to the raw LIF
    # scene before crop extraction (not to the burst matrix).
    montage_n_particles=None,            # particles per construct
    montage_apply_photobleaching=True, # correct whole-scene before cropping
    montage_photobleaching_mode="entire_image",
    montage_crop_size_px=15,           # NxN max-Z projected crop
    montage_gaussian_filter_value=1.0, # 0 = raw pixels; >0 smooths full max-Z frame
    montage_n_snapshots=60,            # evenly spaced across movie
    montage_coordinate_mode="nearest_valid",
    montage_coordinate_max_gap=2,      # frames; beyond → blank crop
    montage_crop_norm_mode="per_crop_percentile",  # high-contrast per-crop stretch
    montage_crop_colormap="gray",      # "gray" for grayscale; None for legacy RGB
    montage_trace_norm_mode="raw",     # "raw" | "min_max" | "total_intensity"
    montage_smooth_window=1,           # 1 = no smoothing on traces
    # Vertical layout: [trace, on_off_bar, crop_montage] as proportions.
    # e.g. [0.60, 0.15, 0.25] = 60 % trace, 15 % state bar, 25 % crops.
    montage_section_height_ratios=[0.72, 0.1, 0.18],
    montage_show_crop_time_labels=False,  # False to hide time text above crops
)

# Plot parameters (kept separate — these don't affect scientific results)
PLOT_PARAMS = dict(
    kymograph_figsize=(8.5, 4.2),
    kymograph_dpi=300,
    kymograph_sort_by="density",        # "density" = longest first; "fraction_on" = by ON time
    max_traces_to_plot=160,
    trace_figsize=(7, 5),
    distribution_figsize=(7, 3.2),
    summary_figsize=(4.5, 3.2),
    comparison_figsize=(5.2, 4.5),
    plot_dpi=300,
    # ── Montage PDF layout ──
    montage_montages_per_page=2,       # panels per PDF page
    montage_panel_figsize=(16, 6),     # (width, height_per_panel) in inches
    montage_pdf_dpi=200,
)


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
        total_frames = int(df["frame"].max()) + 1
        if total_frames_movie <= 0:
            total_frames_movie = total_frames

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

    # Left-align and filter using MicroLive (Stage 4 sync)
    # survival_mask from intensity matrix applied to BOTH matrices.
    survival_mask = _shift_survival_mask(
        combined,
        PARAMS["shift_min_data_fraction"],
        PARAMS["shift_max_missing_frames"],
    )
    combined = mi.Utilities().shift_trajectories(
        combined,
        min_percentage_data_in_trajectory=PARAMS["shift_min_data_fraction"],
        max_missing_frames=PARAMS["shift_max_missing_frames"],
    )
    # Apply the SAME left-alignment shift to the SNR matrix so columns
    # stay synchronised with intensity.
    snr_combined = mi.Utilities().shift_trajectories(
        snr_combined,
        min_percentage_data_in_trajectory=PARAMS["shift_min_data_fraction"],
        max_missing_frames=PARAMS["shift_max_missing_frames"],
    )
    assert combined.shape == snr_combined.shape, (
        f"Intensity/SNR shape mismatch after shift: {combined.shape} vs {snr_combined.shape}"
    )
    particle_origins = [origin for origin, keep in zip(all_origins, survival_mask) if keep]
    if combined.shape[0] != len(particle_origins):
        raise RuntimeError(
            f"Origin/matrix row mismatch after shifting: "
            f"{len(particle_origins)} origins for {combined.shape[0]} rows"
        )

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
        np.save(output_dir / "raw_matrix.npy", matrix_ch0)
        np.save(output_dir / "snr_matrix.npy", snr_ch0)

        # Run burst quantification
        result = run_burst_quantification(
            input_matrix=matrix_ch0,
            snr_matrix=snr_ch0,
            output_dir=output_dir,
            condition=full,
            **{k: v for k, v in PARAMS.items()
               if k not in ("min_snr", "snr_channel_index",
                            "shift_min_data_fraction",
                            "shift_max_missing_frames")
               and not k.startswith("montage_")},
            **{k: v for k, v in PLOT_PARAMS.items()
               if not k.startswith("montage_")},
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

        # Generate all-traces PDF (10 traces per page, sequential order)
        generate_pdf(output_dir)

        # ── Dual-channel kymograph (Green=Folding ch0, Magenta=Nascent ch1) ──
        # Use only QC-passed trajectories (same as montage/burst results)
        nascent_ch = 1  # channel index for nascent
        try:
            matrix_ch1_raw, _, _ = _load_construct_matrix(
                DATA_ROOT / construct_name / "results",
                nascent_ch,
                verbose=False,
            )
            # Get QC-passed row indices (same rows kept for ch0 in burst quant)
            qc = result["qc_table"]
            keep_idx = qc[qc["qc_status"] == "kept"]["trajectory_index"].values

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
                max_traces_to_plot=PLOT_PARAMS.get("max_traces_to_plot", 160),
                figsize=PLOT_PARAMS.get("kymograph_figsize", (8.5, 4.2)),
                dpi=PLOT_PARAMS.get("kymograph_dpi", 300),
            )
            print(f"  ✓ Dual-channel kymograph saved ({n_rows} QC-passed trajectories)")
        except Exception as e:
            print(f"  ⚠ Dual-channel kymograph skipped: {e}")

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
        • If ``n`` is None, select ALL trajectories (sorted by burst count).
        • Top  n//2  — highest burst count  (most active particles).
        • Next n//2  — closest to the population-median fraction_time_on
          (typical particles).  Ties broken by burst count.
    """
    if trajectory_summary.empty:
        return []

    # n=None → all trajectories
    if n is None:
        by_bursts = trajectory_summary.sort_values(
            ["n_bursts", "fraction_time_on"], ascending=[False, False],
        )
        return [_build_selection(row, origins) for _, row in by_bursts.iterrows()]

    n = min(int(n), len(trajectory_summary))

    # ── Top half: most bursts ──
    by_bursts = trajectory_summary.sort_values(
        ["n_bursts", "fraction_time_on"],
        ascending=[False, False],
    )
    n_top = max(1, n // 2)
    top = by_bursts.head(n_top)

    # ── Bottom half: closest to median fraction-ON ──
    remaining = trajectory_summary.drop(index=top.index)
    n_remaining = n - len(top)
    if n_remaining > 0 and not remaining.empty:
        median_fraction_on = trajectory_summary["fraction_time_on"].median()
        mid = (
            remaining.assign(
                dist=(remaining["fraction_time_on"] - median_fraction_on).abs()
            )
            .sort_values(["dist", "n_bursts"], ascending=[True, False])
            .head(n_remaining)
        )
        selected = pd.concat([top, mid])
    else:
        selected = top

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
    smooth_window       = PARAMS["montage_smooth_window"]
    height_ratios       = PARAMS["montage_section_height_ratios"]
    show_crop_time_labels = PARAMS.get("montage_show_crop_time_labels", True)
    montages_per_page   = PLOT_PARAMS["montage_montages_per_page"]
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

        output_path = entry["output_dir"] / "plots" / "representative_montages.pdf"
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
                verbose=True,
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

    comp_dir = OUTPUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    constructs = []
    burst_durs = {}
    dwell_durs = {}
    frac_on_vals = {}

    for short, entry in all_results.items():
        result = entry["burst"] if isinstance(entry, dict) and "burst" in entry else entry
        ts = result["trajectory_summary"]
        et = result["event_table"]
        if ts.empty:
            continue
        constructs.append(short)
        frac_on_vals[short] = ts["fraction_time_on"].values

        bursts = et[(et["event_type"] == "burst") & et["passes_duration_filter"]]
        dwells = et[(et["event_type"] == "dwell") & ~et["is_terminal_event"]]
        burst_durs[short] = bursts["duration_minutes"].values
        dwell_durs[short] = dwells["duration_minutes"].values

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

    # ── 1. Burst duration comparison (Fig 3-style whisker plot) ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [burst_durs.get(c, []) for c in constructs]
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Burst Duration (min)",
        title="Burst Duration",
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
    )
    for row in stats:
        stats_rows.append({"metric": "burst_duration_minutes", **row})
    fig.tight_layout()
    save_figure(fig, comp_dir / "burst_duration_comparison", PLOT_PARAMS["plot_dpi"])

    # ── 2. Dwell duration comparison (Fig 3-style whisker plot) ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [dwell_durs.get(c, []) for c in constructs]
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Dwell Duration (min)",
        title="Dwell Duration",
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
    )
    for row in stats:
        stats_rows.append({"metric": "dwell_duration_minutes", **row})
    fig.tight_layout()
    save_figure(fig, comp_dir / "dwell_duration_comparison", PLOT_PARAMS["plot_dpi"])

    # ── 3. Fraction ON comparison (Fig 3-style whisker plot) ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [frac_on_vals.get(c, []) for c in constructs]
    stats = box_with_points(
        ax,
        data,
        constructs,
        ylabel="Fraction Time ON",
        title="Fraction Time ON",
        ylim=(-0.05, 1.05),
        show_stats=True,
        only_significant=True,
        max_percentile_significance=99.5,
    )
    for row in stats:
        stats_rows.append({"metric": "fraction_time_on", **row})
    fig.tight_layout()
    save_figure(fig, comp_dir / "fraction_on_comparison", PLOT_PARAMS["plot_dpi"])

    stats_df = pd.DataFrame(stats_rows)
    stats_path = comp_dir / "pairwise_mannwhitney_stats.csv"
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
    summary_df.to_csv(comp_dir / "summary_table.csv", index=False)
    print(f"\n  Summary table saved to {comp_dir / 'summary_table.csv'}")
    print(f"  Pairwise Mann-Whitney statistics saved to {stats_path}")
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\n  Comparison plots saved to {comp_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Burst Quantification Analysis — CoF Long Movies          ║")
    print("║   Channel 0 = folding (bursting)                           ║")
    print("║   Channel 1 = nascent protein (tracking)                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Verify data drive
    if not DATA_ROOT.exists():
        print(f"\n  ERROR: Data root not found: {DATA_ROOT}")
        print("  Please mount the external drive and try again.")
        sys.exit(1)

    # Step 1: Detrend diagnostic
    run_detrend_diagnostic()

    # Step 1.5: Signal contrast diagnostic (validates threshold choice)
    run_signal_contrast_diagnostic()

    # Step 2: Per-construct burst quantification
    all_results = run_per_construct_analysis()

    # Step 2.5: Representative crop montages
    if all_results:
        generate_representative_montages(all_results)

    # Step 3: Cross-construct comparison
    if all_results:
        run_cross_construct_comparison(all_results)

    print("\n" + "=" * 70)
    print("DONE. All results saved to:")
    print(f"  {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
