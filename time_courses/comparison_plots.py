# =======================================================================
# # Cross-Construct Comparison Plots
#
# Regenerates the three comparison box-with-swarm plots and summary statistics
# from the **already-saved CSV data** produced by `run_burst_analysis.py`.
#
# No need to re-run the pipeline or mount the data drive.
#
# **Plots generated** (PNG + SVG via `save_figure()`):
# 1. Observed ON Episode Duration (trajectory-level median)
# 2. Observed OFF Episode Duration (trajectory-level median)
# 3. Fraction of Observed Time ON
#
# **Statistical notes:**
# - ON/OFF duration plots use **trajectory-level medians** (one value per trajectory), not pooled events
# - Initial OFF dwells are **excluded** (left-censored)
# - Terminal OFF dwells are **excluded**
# - All pairwise Mann-Whitney tests include **optional Benjamini-Hochberg FDR correction**
#
# **Data source:** `results_snr_3/{construct}/trajectory_summary.csv` and `event_table.csv`
# =======================================================================

# ── Imports ──
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Robust path setup ──
# Walk up from cwd to find the repo root (.git marker), then add both
# repo root and time_courses/ to sys.path so local imports resolve.
cwd = Path.cwd()
repo_root = cwd
while repo_root != repo_root.parent and not (repo_root / ".git").exists():
    repo_root = repo_root.parent
if not (repo_root / ".git").exists():
    raise RuntimeError(
        f"Could not find repo root (.git) walking up from {cwd}. "
        "Please run from within the cof_paper repository."
    )
tc_dir = repo_root / "time_courses"
for p in [str(repo_root), str(tc_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from plotting import box_with_points, save_figure, set_publication_style

set_publication_style()

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit this section for different pipeline runs
# ═══════════════════════════════════════════════════════════════════

# Which results directory to read.  Change this for a different SNR run.
RESULTS_DIR = repo_root / "time_courses" / "results_snr_3"

# All available constructs — must match the pipeline's processing order.
CONSTRUCT_ORDER = ["4sf", "4uv", "4sf-Xbp1", "Xbp1-4sf"]

# ── Plot groups ──
# Each entry defines a subset of constructs to compare in one figure.
# Format: (group_name, [short_names], [display_labels])
#   - group_name:    used for the output subdirectory and file suffix
#   - short_names:   keys into the loaded data (must be in CONSTRUCT_ORDER)
#   - display_labels: x-axis labels (same length as short_names)
#
# Add / remove / reorder groups freely.
PLOT_GROUPS = [
    (
        "all",
        ["4sf", "4uv", "4sf-Xbp1", "Xbp1-4sf"],
        ["4sfGFP-2mCh", "4GFPuv-2mCh", "4sfGFP-2mCh-Xbp1", "Xbp1-4sfGFP-2mCh"],
    ),
    (
        "sf_vs_uv",
        ["4sf", "4uv"],
        ["4sfGFP-2mCh", "4GFPuv-2mCh"],
    ),
    (
        "xbp1_context",
        ["4sf", "4sf-Xbp1", "Xbp1-4sf"],
        ["4sfGFP-2mCh", "4sfGFP-2mCh-Xbp1", "Xbp1-4sfGFP-2mCh"],
    ),
]

# Cell counts: read from the summary_table.csv produced by run_burst_analysis.py
_summary_path = RESULTS_DIR / "comparison" / "summary_table.csv"
if _summary_path.exists():
    _summary_df = pd.read_csv(_summary_path)
    if "n_cells" in _summary_df.columns and "short_name" in _summary_df.columns:
        N_CELLS = dict(zip(_summary_df["short_name"], _summary_df["n_cells"].astype(int)))
    else:
        print("WARNING: summary_table.csv missing n_cells or short_name column; cell counts will show 0")
        N_CELLS = {}
else:
    print(f"WARNING: {_summary_path} not found — run run_burst_analysis.py first; cell counts will show 0")
    N_CELLS = {}

# Plot settings — mirror PLOT_PARAMS from config.yaml / run_burst_analysis.py
FIGSIZE = (5.5, 5.5)
PLOT_DPI = 300
USE_BH_FDR = False  # True → apply Benjamini-Hochberg FDR correction
MAX_PERCENTILE = 99.0  # Visual outlier capping percentile (99th percentile)

# Output directory (top-level; per-group subdirs created automatically)
COMP_DIR = RESULTS_DIR / "comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

print(f"Repo root:   {repo_root}")
print(f"Results dir: {RESULTS_DIR}")
print(f"Output dir:  {COMP_DIR}")
print(f"Constructs:  {CONSTRUCT_ORDER}")
print(f"Plot groups: {[g[0] for g in PLOT_GROUPS]}")

# ═══════════════════════════════════════════════════════════════════
#  Load per-construct data
#  ON/OFF durations use trajectory-level medians (one value per
#  trajectory) to avoid pseudoreplication from pooled events.
#  Initial and terminal OFF dwells are excluded (censored).
# ═══════════════════════════════════════════════════════════════════

frac_on_vals = {}   # short_name -> array of fraction_time_on (per trajectory)
burst_durs = {}     # short_name -> array of traj-level median ON durations
dwell_durs = {}     # short_name -> array of traj-level median OFF durations
n_trajectories = {} # short_name -> int
n_on_events = {}    # short_name -> int (burst events passing duration filter)
n_off_events = {}   # short_name -> int (dwell events, non-initial, non-terminal)

for short in CONSTRUCT_ORDER:
    construct_dir = RESULTS_DIR / short
    traj_summary_path = construct_dir / "trajectory_summary.csv"
    event_table_path = construct_dir / "event_table.csv"
    if not traj_summary_path.exists() or not event_table_path.exists():
        print(f"  WARNING: Missing data for {short}, skipping")
        continue
    traj_summary_df = pd.read_csv(traj_summary_path)
    event_table_df = pd.read_csv(event_table_path)
    frac_on_vals[short] = traj_summary_df["fraction_time_on"].values
    n_trajectories[short] = len(traj_summary_df)
    # Burst (ON) durations: events passing duration filter
    bursts_df = event_table_df[
        (event_table_df["event_type"] == "burst") & event_table_df["passes_duration_filter"]
    ]
    n_on_events[short] = len(bursts_df)
    # Dwell (OFF) durations: exclude initial (left-censored) and terminal
    dwells_df = event_table_df[
        (event_table_df["event_type"] == "dwell")
        & ~event_table_df["is_terminal_event"]
        & ~event_table_df["is_initial_dwell"]
    ]
    n_off_events[short] = len(dwells_df)
    # Trajectory-level median durations (one value per trajectory)
    burst_durs[short] = (
        bursts_df.groupby("trajectory_id")["duration_minutes"].median().values
        if not bursts_df.empty else np.array([])
    )
    dwell_durs[short] = (
        dwells_df.groupby("trajectory_id")["duration_minutes"].median().values
        if not dwells_df.empty else np.array([])
    )
    print(
        f"  {short:12s} | {n_trajectories[short]:4d} traj | "
        f"{n_on_events[short]:4d} ON events | {n_off_events[short]:4d} OFF events | "
        f"{len(burst_durs[short]):4d} traj w/ ON | {len(dwell_durs[short]):4d} traj w/ OFF"
    )
print(f"\nLoaded {len(frac_on_vals)} constructs")


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def build_xlabels(short_names, display_labels, event_counts=None):
    """Build multi-line x-axis labels: label / cells / traj / (events)."""
    labels = []
    for short, label in zip(short_names, display_labels):
        parts = [
            label,
            f"{N_CELLS.get(short, 0)} cells",
            f"{n_trajectories.get(short, 0)} traj",
        ]
        if event_counts is not None:
            parts.append(f"({event_counts.get(short, 0)} events)")
        labels.append("\n".join(parts))
    return labels


def plot_group(group_name, short_names, display_labels):
    """Generate the 3 comparison plots + stats + summary for one group."""
    out_dir = COMP_DIR / group_name if group_name != "all" else COMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    group_stats_rows = []
    # Adjust figure width for number of conditions
    n_conditions = len(short_names)
    fig_width = max(3.5, 1.5 * n_conditions + 1.5)
    figsize = (fig_width, FIGSIZE[1])

    # ── Plot 1: ON Episode Duration ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [burst_durs.get(c, []) for c in short_names]
    xlabels = build_xlabels(short_names, display_labels, n_on_events)
    stats = box_with_points(
        ax, data, display_labels,
        ylabel="Observed ON Episode Duration (min)",
        title="Observed ON Episode Duration (traj. median)",
        xlabels=xlabels,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=MAX_PERCENTILE,
        use_bh_fdr=USE_BH_FDR,
    )
    for row in stats:
        group_stats_rows.append({"metric": "on_duration_minutes", **row})
    fig.tight_layout()
    plt.show()
    save_figure(fig, out_dir / "burst_duration_comparison", PLOT_DPI)

    # ── Plot 2: OFF Episode Duration ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [dwell_durs.get(c, []) for c in short_names]
    xlabels = build_xlabels(short_names, display_labels, n_off_events)
    stats = box_with_points(
        ax, data, display_labels,
        ylabel="Observed OFF Episode Duration (min)",
        title="Observed OFF Episode Duration (traj. median)",
        xlabels=xlabels,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=MAX_PERCENTILE,
        use_bh_fdr=USE_BH_FDR,
    )
    for row in stats:
        group_stats_rows.append({"metric": "off_duration_minutes", **row})
    fig.tight_layout()
    plt.show()
    save_figure(fig, out_dir / "dwell_duration_comparison", PLOT_DPI)

    # ── Plot 3: Fraction of Observed Time ON ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    data = [frac_on_vals.get(c, []) for c in short_names]
    xlabels = build_xlabels(short_names, display_labels)
    stats = box_with_points(
        ax, data, display_labels,
        ylabel="Fraction of Observed Time ON",
        title="Fraction of Observed Time ON",
        ylim=(-0.05, 1.05),
        xlabels=xlabels,
        show_stats=True,
        only_significant=True,
        max_percentile_significance=MAX_PERCENTILE,
        use_bh_fdr=USE_BH_FDR,
    )
    for row in stats:
        group_stats_rows.append({"metric": "fraction_time_on", **row})
    fig.tight_layout()
    plt.show()
    save_figure(fig, out_dir / "fraction_on_comparison", PLOT_DPI)

    # ── Pairwise stats ──
    stats_df = pd.DataFrame(group_stats_rows)
    stats_path = out_dir / "pairwise_mannwhitney_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Pairwise statistics → {stats_path}")
    print(stats_df)

    # ── Summary table ──
    summary_rows = []
    for short in short_names:
        on_durs = burst_durs.get(short, np.array([]))
        off_durs = dwell_durs.get(short, np.array([]))
        frac_on = frac_on_vals.get(short, np.array([]))
        summary_rows.append({
            "short_name": short,
            "n_cells": N_CELLS.get(short, 0),
            "n_trajectories": n_trajectories.get(short, 0),
            "n_on_events": n_on_events.get(short, 0),
            "n_off_events": n_off_events.get(short, 0),
            "n_traj_with_on": len(on_durs),
            "n_traj_with_off": len(off_durs),
            "mean_on_dur_min": round(float(np.mean(on_durs)), 3) if len(on_durs) > 0 else np.nan,
            "median_on_dur_min": round(float(np.median(on_durs)), 3) if len(on_durs) > 0 else np.nan,
            "std_on_dur_min": round(float(np.std(on_durs, ddof=1)), 3) if len(on_durs) > 1 else np.nan,
            "sem_on_dur_min": round(float(np.std(on_durs, ddof=1) / np.sqrt(len(on_durs))), 3) if len(on_durs) > 1 else np.nan,
            "mean_off_dur_min": round(float(np.mean(off_durs)), 3) if len(off_durs) > 0 else np.nan,
            "median_off_dur_min": round(float(np.median(off_durs)), 3) if len(off_durs) > 0 else np.nan,
            "std_off_dur_min": round(float(np.std(off_durs, ddof=1)), 3) if len(off_durs) > 1 else np.nan,
            "sem_off_dur_min": round(float(np.std(off_durs, ddof=1) / np.sqrt(len(off_durs))), 3) if len(off_durs) > 1 else np.nan,
            "mean_fraction_on": round(float(np.mean(frac_on)), 4) if len(frac_on) > 0 else np.nan,
            "median_fraction_on": round(float(np.median(frac_on)), 4) if len(frac_on) > 0 else np.nan,
            "std_fraction_on": round(float(np.std(frac_on, ddof=1)), 4) if len(frac_on) > 1 else np.nan,
            "sem_fraction_on": round(float(np.std(frac_on, ddof=1) / np.sqrt(len(frac_on))), 4) if len(frac_on) > 1 else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary table → {summary_path}")
    print(summary_df)
    return group_stats_rows


# ═══════════════════════════════════════════════════════════════════
#  Generate plots for each group
# ═══════════════════════════════════════════════════════════════════

for group_name, short_names, display_labels in PLOT_GROUPS:
    print(f"\n{'─' * 60}")
    print(f"  Group: {group_name}  ({', '.join(display_labels)})")
    print(f"{'─' * 60}")
    plot_group(group_name, short_names, display_labels)

print("\nDone.")
