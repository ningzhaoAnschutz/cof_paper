#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot intensity trajectories directly from the original MicroLive tracking CSVs.

This script reads exclusively from the result folders (tracking_*.csv) without
any reprocessing, photobleaching correction, or intensity re-extraction.

It produces trajectory plots for each condition that can be compared with
runner.py output to verify whether the photobleaching correction is working.

Usage:
    python plot_raw_trajectories.py
"""

import sys
import os
from pathlib import Path

MICROLIVE_ENV = '/opt/anaconda3/envs/microlive'
if sys.prefix != MICROLIVE_ENV:
    python_exe = os.path.join(MICROLIVE_ENV, 'bin', 'python')
    if os.path.exists(python_exe):
        print(f"Relaunching in microlive environment...\n")
        os.execl(python_exe, python_exe, *sys.argv)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
    'font.family':        'sans-serif',
    'font.sans-serif':    'Arial',
    'text.color':         'black',
    'axes.labelcolor':    'black',
    'xtick.color':        'black',
    'ytick.color':        'black',
    'axes.edgecolor':     'black',
    'axes.linewidth':     1.5,
    'axes.spines.top':    True,
    'axes.spines.right':  True,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.grid':          False,
    'axes.labelsize':     18,
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
    'legend.fontsize':    12,
})

# ============================================================
# ★★★  CONFIGURATION  ★★★
# ============================================================

CONDITIONS = [
    {
        'name':        'sfGFP',
        'color':       '#A8B4A9',
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/sfGFP/results'),
    },
    {
        'name':        'GFPuv',
        'color':       '#4CAF50',
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv/results'),
    },
    {
        'name':        'sfGFP_ex',
        'color':       '#FF5722',
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS038/results'),
    },
    {
        'name':        'sfGFP_sx',
        'color':       '#5B7FAD',
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS048/results'),
    },
]

CHANNEL = 1
TIME_INTERVAL_S = 5.0
MAX_TRAJECTORIES = 50
MIN_TRAJECTORY_LENGTH = 108   # minimum frames a particle must have
OUTPUT_DIR = Path(__file__).resolve().parent / 'results' / 'raw_tracking_trajectories'

# ============================================================
# LOAD ALL TRACKING CSVs FOR ONE CONDITION
# ============================================================
def load_all_tracking_csvs(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all tracking_*.csv from result subfolders."""
    all_frames = []
    result_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith('results_')
    )
    for rd in result_dirs:
        tracking_csvs = list(rd.glob('tracking_*.csv'))
        if not tracking_csvs:
            continue
        csv_path = tracking_csvs[0]
        df = pd.read_csv(csv_path)
        # Tag with source folder
        df['result_dir'] = rd.name
        df['global_particle_id'] = rd.name + '::' + df['unique_particle'].astype(str)
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# ============================================================
# PLOT TRAJECTORIES
# ============================================================
def plot_trajectories(
    df: pd.DataFrame,
    condition_name: str,
    color: str,
    channel: int,
    output_dir: Path,
    max_trajectories: int = 50,
    min_length: int = 20,
    time_interval_s: float = 5.0,
    figsize: tuple = (7, 4.5),
):
    """Plot individual intensity trajectories + mean ± SEM."""
    int_col = f'spot_int_ch_{channel}'
    if int_col not in df.columns:
        print(f"  [{condition_name}] Column '{int_col}' not found, skipping.")
        return

    # Filter: finite positive values
    df = df[df[int_col].notna() & np.isfinite(df[int_col]) & (df[int_col] > 0)].copy()

    # Filter by trajectory length
    traj_lengths = df.groupby('global_particle_id')['frame'].nunique()
    long_enough = traj_lengths[traj_lengths >= min_length].index
    df = df[df['global_particle_id'].isin(long_enough)]

    if df.empty:
        print(f"  [{condition_name}] No valid trajectories.")
        return

    particles = df['global_particle_id'].unique()
    total_particles = len(particles)
    print(f"  [{condition_name}] {total_particles} trajectories (≥{min_length} frames)")

    fig, ax = plt.subplots(figsize=figsize)

    # Build traces from ALL particles for mean ± SEM.
    all_traces = []
    for pid in particles:
        sub = df[df['global_particle_id'] == pid].sort_values('frame')
        frames = sub['frame'].to_numpy()
        values = sub[int_col].to_numpy()
        all_traces.append(pd.Series(values, index=frames, name=pid))

    # Draw only a subsample of individual lines to keep the plot readable.
    plot_particles = particles
    if max_trajectories is not None and len(particles) > max_trajectories:
        rng = np.random.RandomState(42)
        plot_particles = rng.choice(particles, size=max_trajectories, replace=False)

    for pid in plot_particles:
        sub = df[df['global_particle_id'] == pid].sort_values('frame')
        times = sub['frame'].to_numpy() * time_interval_s
        values = sub[int_col].to_numpy()
        ax.plot(times, values, color=color, linewidth=0.6, alpha=0.25, zorder=1)

    # Mean ± SEM from ALL trajectories.
    if all_traces:
        traces_df = pd.concat(all_traces, axis=1).sort_index()
        mean_trace = traces_df.mean(axis=1)
        sem_trace = traces_df.sem(axis=1)
        t_mean = mean_trace.index.to_numpy() * time_interval_s

        ax.fill_between(
            t_mean,
            (mean_trace - sem_trace).to_numpy(),
            (mean_trace + sem_trace).to_numpy(),
            color=color, alpha=0.3, zorder=2,
        )
        ax.plot(t_mean, mean_trace.to_numpy(), color=color, linewidth=2.5,
                zorder=3, label=f'Mean (n = {total_particles})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Intensity ch{channel} (a.u.)')
    ax.set_title(f'{condition_name} — ch{channel}  (raw tracking CSV)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'raw_trajectories_{condition_name}_ch{channel}'
    plt.tight_layout()
    plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {stem}.png / .svg → {output_dir}")

    # Print summary stats
    all_vals = df[int_col]
    early = df[df['frame'] < 50][int_col].mean()
    late = df[df['frame'] > 300][int_col].mean()
    ratio = late / early if early > 0 else float('nan')
    print(f"    Stats: mean={all_vals.mean():.1f}  early(<50)={early:.1f}  "
          f"late(>300)={late:.1f}  late/early={ratio:.3f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}\n")

    for cond in CONDITIONS:
        name = cond['name']
        color = cond['color']
        results_dir = cond['results_dir']

        print(f"\n{'='*60}")
        print(f"  {name}: {results_dir}")
        print(f"{'='*60}")

        if not results_dir.exists():
            print(f"  ⚠ Directory not found, skipping.")
            continue

        df = load_all_tracking_csvs(results_dir)
        if df.empty:
            print(f"  ⚠ No tracking data found.")
            continue

        print(f"  Loaded {len(df)} rows, "
              f"{df['global_particle_id'].nunique()} particles, "
              f"{df['result_dir'].nunique()} result folders")

        plot_trajectories(
            df,
            condition_name=name,
            color=color,
            channel=CHANNEL,
            output_dir=OUTPUT_DIR,
            max_trajectories=MAX_TRAJECTORIES,
            min_length=MIN_TRAJECTORY_LENGTH,
            time_interval_s=TIME_INTERVAL_S,
        )

    print(f"\n{'='*60}")
    print(f"  Done. All plots → {OUTPUT_DIR}")
    print(f"{'='*60}\n")
