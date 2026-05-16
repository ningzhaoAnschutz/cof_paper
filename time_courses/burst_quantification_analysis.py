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
    python burst_quantification_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from microlive import microscopy as mi

# ── Path setup (matches notebook convention) ────────────────────────────────
# This script lives in cof_paper/time_courses/
repo_root = Path(__file__).resolve().parent.parent  # → cof_paper/
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # time_courses/

from utilities.config import REPORTER_PLASMID_NAME_MAPPING, PLASMID_SHORT_NAME_MAPPING
from burst_quantification_from_matrix import run_burst_quantification
from generate_all_traces_pdf import generate_pdf
from plot_style import (
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
    # OFF-baseline + k × MAD_off (noise-floor anchored)
    # k=4 ≈ 4σ above the per-trace noise floor
    threshold_mode="off_baseline_mad",
    threshold=4.0,
    off_baseline_quantile=0.25,
    # ── Burst module: event cleanup ──
    min_event_duration_frames=6,
    min_burst_duration_seconds=60.0,
    exclude_terminal_dwell=True,
    count_initial_dwell=True,
)

# Plot parameters (kept separate — these don't affect scientific results)
PLOT_PARAMS = dict(
    kymograph_figsize=(8.5, 4.2),
    kymograph_dpi=300,
    max_traces_to_plot=160,
    trace_figsize=(7, 5),
    distribution_figsize=(7, 3.2),
    summary_figsize=(4.5, 3.2),
    comparison_figsize=(5.2, 4.5),
    plot_dpi=300,
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
            matrix_ch0 = _load_construct_matrix(
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
            matrix_ch0 = _load_construct_matrix(data_folder, burst_ch, verbose=False)
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

def _load_construct_matrix(data_folder, channel_index, verbose=True):
    """Load all tracking CSVs from a construct's results folder and extract
    the intensity matrix for the given channel.

    Builds a rectangular matrix (n_particles × max_frames) with NaN padding
    for shorter FOVs, then uses MicroLive shift_trajectories for left-alignment.
    """
    data_folder = Path(data_folder)
    results_dirs = sorted([
        d for d in data_folder.iterdir()
        if d.is_dir() and d.name.startswith("results_")
    ])

    if not results_dirs:
        raise FileNotFoundError(f"No results_* folders in {data_folder}")

    all_matrices = []
    field = f"spot_int_ch_{channel_index}"

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

        # Build per-particle intensity matrix
        particles = df["particle"].unique()
        if "frame" not in df.columns:
            continue
        total_frames = int(df["frame"].max()) + 1

        matrix = np.full((len(particles), total_frames), np.nan)
        for p_idx, p in enumerate(particles):
            sub = df[df["particle"] == p]
            frames = sub["frame"].values.astype(int)
            values = sub[field].values.astype(float)
            valid = frames < total_frames
            matrix[p_idx, frames[valid]] = values[valid]

        # SNR filter — keyed on snr_channel_index (default ch1, the nascent
        # tracking channel) so QC reflects localization quality, not how
        # bright the folding signal happens to be.
        snr_ch = PARAMS.get("snr_channel_index")
        if snr_ch is None:
            snr_ch = channel_index
        snr_field = f"snr_ch_{snr_ch}"
        if snr_field in df.columns:
            for p_idx, p in enumerate(particles):
                sub = df[df["particle"] == p]
                mean_snr = sub[snr_field].mean()
                if mean_snr < PARAMS["min_snr"]:
                    matrix[p_idx, :] = np.nan
        if verbose:
            print(f"    FOV {rdir.name}: {matrix.shape[0]} particles × "
                  f"{matrix.shape[1]} frames")
        all_matrices.append(matrix)

    if not all_matrices:
        raise ValueError(f"No valid tracking data found in {data_folder}")

    # Concatenate all FOVs — pad to max frame count with NaN
    max_cols = max(m.shape[1] for m in all_matrices)
    padded = []
    for m in all_matrices:
        if m.shape[1] < max_cols:
            pad_arr = np.full((m.shape[0], max_cols - m.shape[1]), np.nan)
            m = np.hstack([m, pad_arr])
        padded.append(m)
    combined = np.vstack(padded)
    if verbose:
        print(f"    Combined: {combined.shape[0]} trajectories × "
              f"{combined.shape[1]} frames (max across FOVs)")

    # Left-align and filter using MicroLive (same as existing notebooks)
    combined = mi.Utilities().shift_trajectories(
        combined,
        min_percentage_data_in_trajectory=PARAMS["shift_min_data_fraction"],
        max_missing_frames=PARAMS["shift_max_missing_frames"],
    )

    if verbose:
        print(f"    After shift/filter: {combined.shape[0]} trajectories × "
              f"{combined.shape[1]} frames")

    # NOTE: Do NOT forward-fill here. The burst module handles forward-fill
    # internally with proper NaN re-stamping so filled frames don't become
    # false burst/dwell evidence.
    return combined


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

        # Load folding channel (ch0)
        try:
            matrix_ch0 = _load_construct_matrix(
                DATA_ROOT / construct_name / "results",
                burst_ch,
                verbose=True,
            )
        except Exception as e:
            print(f"  ERROR loading {construct_name}: {e}")
            continue

        print(f"  Loaded: {matrix_ch0.shape[0]} trajectories × {matrix_ch0.shape[1]} timepoints")

        # Save raw matrix for reproducibility
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "raw_matrix.npy", matrix_ch0)

        # Run burst quantification
        result = run_burst_quantification(
            input_matrix=matrix_ch0,
            output_dir=output_dir,
            condition=full,
            **{k: v for k, v in PARAMS.items()
               if k not in ("min_snr", "snr_channel_index",
                            "shift_min_data_fraction",
                            "shift_max_missing_frames")},
            **PLOT_PARAMS,
        )

        all_results[short] = result

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

    return all_results


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

    for short, result in all_results.items():
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
        result = all_results[short]
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
            "mean_dwell_dur_min": round(float(np.mean(dd)), 3) if len(dd) > 0 else np.nan,
            "median_dwell_dur_min": round(float(np.median(dd)), 3) if len(dd) > 0 else np.nan,
            "mean_fraction_on": round(float(np.mean(fo)), 4) if len(fo) > 0 else np.nan,
            "median_fraction_on": round(float(np.median(fo)), 4) if len(fo) > 0 else np.nan,
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

    # Step 3: Cross-construct comparison
    if all_results:
        run_cross_construct_comparison(all_results)

    print("\n" + "=" * 70)
    print("DONE. All results saved to:")
    print(f"  {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
