"""Standalone PSF Amplitude vs Sigma plot from a saved master_intensity_dataset.csv.

Usage
-----
    python plot_psf.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Environment self-relaunch: re-exec with the microlive conda Python if needed.
try:
    import microlive  # noqa: F401
except ModuleNotFoundError:
    conda_result = subprocess.run(
        ['conda', 'run', '-n', 'microlive', 'which', 'python'],
        capture_output=True, text=True,
    )
    env_python = conda_result.stdout.strip()
    if not env_python:
        raise RuntimeError(
            "Could not find the 'microlive' conda environment. "
            "Make sure conda is on PATH and the environment exists."
        )
    print(f"Re-launching with microlive env Python: {env_python}")
    os.execv(env_python, [env_python] + sys.argv)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde


# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'font.family':       'sans-serif',
    'font.sans-serif':   'Arial',
    'text.color':        'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'axes.edgecolor':    'black',
    'axes.linewidth':    1.2,
    'axes.spines.top':   True,
    'axes.spines.right': True,
    'axes.spines.left':  True,
    'axes.spines.bottom': True,
    'axes.grid':          False,
    'axes.labelsize':     18,
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
})

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH      = Path('/Users/nzlab-la/Desktop/cof_paper/notebooks/reprocessing_data/results/sz5_fast_peak/master_intensity_dataset_sz5_fast_peak.csv')
OUTPUT_DIR    = CSV_PATH.parent
CHANNEL_INDEX = 1          # 0-based channel to plot
MIN_SNR       = 0.5
SIGMA_LIM     = (1.4, 2.0)
AMPLITUDE_LIM = (100, 7000)
KDE_BW        = 0.25
FIG_SIZE_DIS  = (5, 4.5)
FIG_SIZE_ACF  = (5, 4.5)

# ─────────────────────────────────────────────────────────────────────────────

CONDITION_COLORS = {
    'sfGFP':  '#A8B4A9',   # gray
    'GFPuv':  '#4CAF50',   # green
    'sfGFP_ex': '#FF5722',   # deep orange  (pRS038, end-XBP1)
    'sfGFP_sx': '#5B7FAD',   # blue
}

# Maps old condition names in the CSV → current display names.
# Applied automatically when loading master_intensity_dataset.csv.
CONDITION_RENAME = {
    'pRS038': 'sfGFP_ex',
    'pRS048': 'sfGFP_sx',
    'GFP_ex': 'sfGFP_ex',
    'GFP_sx': 'sfGFP_sx',
}


def plot_psf_amplitude_vs_sigma(
    master_df: pd.DataFrame,
    channel_index: int = CHANNEL_INDEX,
    min_snr: float = MIN_SNR,
    sigma_lim: tuple = SIGMA_LIM,
    amplitude_lim: tuple = AMPLITUDE_LIM,
    kde_bw: float = KDE_BW,
    output_dir: Path = OUTPUT_DIR,
    save_name: str = None,
) -> None:
    amp_col = f'psf_amplitude_ch_{channel_index}'
    sig_col = f'psf_sigma_ch_{channel_index}'
    snr_col = f'snr_ch_{channel_index}'

    fig, ax = plt.subplots(figsize=(7, 6))

    conditions = master_df['condition'].unique()

    # ── First pass: collect all per-particle values to set percentile limits ──
    all_sig_vals, all_amp_vals = [], []
    per_cond_data = {}  # cache so we don't recompute in second pass
    for cond in conditions:
        df = master_df[master_df['condition'] == cond].copy()
        if snr_col in df.columns:
            df = df[df[snr_col] >= min_snr]
        df = df[df[amp_col].notna() & (df[amp_col] > 0)]
        df = df[df[sig_col].notna() & (df[sig_col] > 0)]
        if df.empty:
            continue
        per_particle = df.groupby('global_particle_id')[[amp_col, sig_col]].mean()
        all_sig_vals.append(per_particle[sig_col].to_numpy())
        all_amp_vals.append(per_particle[amp_col].to_numpy())
        per_cond_data[cond] = per_particle

    if not all_sig_vals:
        print("No data to plot.")
        plt.close()
        return

    pooled_sig = np.concatenate(all_sig_vals)
    pooled_amp = np.concatenate(all_amp_vals)
    sig_lo, sig_hi = np.percentile(pooled_sig, 1), np.percentile(pooled_sig, 95)
    amp_lo, amp_hi = np.percentile(pooled_amp, 1), np.percentile(pooled_amp, 95)

    # ── Second pass: draw KDE contours using percentile-derived limits ────────
    for cond, per_particle in per_cond_data.items():
        color = CONDITION_COLORS.get(cond, '#888888')
        amp = per_particle[amp_col].to_numpy()
        sig = per_particle[sig_col].to_numpy()
        n   = len(per_particle)

        mask = (
            (sig >= sig_lo) & (sig <= sig_hi) &
            (amp >= amp_lo) & (amp <= amp_hi)
        )
        sig_c, amp_c = sig[mask], amp[mask]
        if sig_c.size < 4:
            continue

        try:
            kde = gaussian_kde(np.vstack([sig_c, amp_c]), bw_method=kde_bw)
            xg  = np.linspace(sig_lo, sig_hi, 120)
            yg  = np.linspace(amp_lo, amp_hi, 120)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ZZ /= ZZ.max()

            ax.contourf(XX, YY, ZZ,
                        levels=[0.20, 0.50, 0.80, 1.01],
                        colors=[color], alpha=0.18)
            ax.contour(XX, YY, ZZ,
                       levels=[0.20, 0.50, 0.80],
                       colors=[color], linewidths=[0.7, 1.0, 1.4], alpha=0.85)

            med_x, med_y = np.median(sig_c), np.median(amp_c)
            ax.plot(med_x, med_y, marker='+', color=color,
                    markersize=10, markeredgewidth=1.8,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                    zorder=5, label=f'{cond}  (n={n})')

        except Exception as exc:
            print(f"  [{cond}] KDE failed ({exc}), using scatter.")
            ax.scatter(sig_c, amp_c, color=color, alpha=0.4, s=8,
                       linewidths=0, label=f'{cond}  (n={n})')

    ax.set_xlabel('PSF σ (px)')
    ax.set_ylabel('PSF Amplitude (a.u.)')
    #ax.set_title(f'PSF Amplitude vs Sigma — Channel {channel_index}',
    #             fontsize=12, fontweight='bold')
    ax.set_xlim(sig_lo, sig_hi)
    ax.set_ylim(amp_lo, amp_hi)
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()

    stem = save_name or f'psf_amplitude_vs_sigma_ch{channel_index}'
    plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    print(f"Saved {stem}.svg / .png -> {output_dir}")
    plt.close()



# ── In-memory intensity distribution extractor ────────────────────────────────

def extract_field_distributions(
    master_df: pd.DataFrame,
    field: str,
    channel_index: int = 0,
    min_snr: float = 0.5,
    condition: str = None,
) -> dict:
    """Extract per-particle and all-timepoint distributions for any numeric field.

    In-memory replacement for pipeline_time_courses.extract_intensity_distributions.
    Returns mean-per-particle (mode 1), all observations (mode 2), and a snapshot
    at the median occupied frame (mode 3).

    Args:
        master_df: Output of runner.run_pipeline.
        field: Column prefix, e.g. 'snr_ch_' or 'spot_int_ch_'.
        channel_index: Channel suffix appended to field.
        min_snr: Minimum SNR filter (applied to snr_ch_<c>).
        condition: Condition name to filter on. None = all conditions.

    Returns:
        dict with keys mean_per_particle, all_timepoints, at_timepoint (all ndarrays),
        n_particles, n_result_dirs.
    """
    col     = f'{field}{channel_index}'
    snr_col = f'snr_ch_{channel_index}'

    df = master_df.copy()
    if condition is not None:
        df = df[df['condition'] == condition]
    if snr_col in df.columns:
        df = df[df[snr_col] >= min_snr]
    df = df[df[col].notna() & np.isfinite(df[col]) & (df[col] > 0)].copy()

    if df.empty:
        raise ValueError(f"No valid data for field='{col}' condition='{condition}'.")

    # Mode 1: mean per particle.
    per_particle = df.groupby('global_particle_id')[col].mean().to_numpy()

    # Mode 2: all observations.
    all_timepoints = df[col].to_numpy()

    # Mode 3: snapshot at median occupied frame.
    median_frame = int(np.median(df['frame'].to_numpy()))
    at_timepoint = df[df['frame'] == median_frame][col].to_numpy()

    return {
        'mean_per_particle':    per_particle,
        'all_timepoints':       all_timepoints,
        'at_timepoint':         at_timepoint,
        'n_particles':          len(per_particle),
        'n_result_dirs':        df['result_dir_id'].nunique(),
    }


# ── KDE distribution plot ─────────────────────────────────────────────────────

def plot_intensity_distributions(
    master_df: pd.DataFrame,
    field: str,
    channel_index: int = 0,
    min_snr: float = 0.5,
    x_label: str = 'Value',
    xlim: tuple = None,
    y_label: str = 'Probability Density',
    bins: int = 60,
    kde_bw: str = 'scott',
    output_dir: Path = None,
    save_name: str = None,
    figsize=(FIG_SIZE_DIS),
) -> None:
    """Plot KDE + histogram of a field distribution for each condition.

    Generates a single-panel plot (mean per particle, mode 1) with one
    KDE curve per condition, plus a dashed vertical median marker.

    Args:
        master_df: Output of runner.run_pipeline.
        field: Column prefix, e.g. 'snr_ch_' or 'spot_int_ch_'.
        channel_index: Channel suffix.
        min_snr: Minimum SNR filter.
        x_label: X-axis label.
        xlim: (min, max) x-axis limits, or None for auto.
        bins: Histogram bin count (fallback if KDE fails).
        kde_bw: Gaussian KDE bandwidth method.
        output_dir: Directory to save SVG + PNG.
        save_name: File stem override. Defaults to '<field>ch<channel_index>'.
    """
    from scipy.stats import gaussian_kde as _kde

    fig, ax = plt.subplots(figsize=figsize)

    conditions = master_df['condition'].unique()

    # First pass: collect all per-particle values to compute percentile xlim.
    all_vals = []
    per_cond_vals = {}
    for cond in conditions:
        try:
            dist = extract_field_distributions(
                master_df, field=field, channel_index=channel_index,
                min_snr=min_snr, condition=cond,
            )
        except ValueError:
            continue
        v = dist['mean_per_particle']
        v = v[np.isfinite(v) & (v > 0)]
        per_cond_vals[cond] = (v, dist['n_particles'])
        all_vals.append(v)

    # Auto xlim from pooled 1st–99.55th percentiles if not specified.
    if xlim is None and all_vals:
        pooled = np.concatenate(all_vals)
        xlim = (np.percentile(pooled, 0), np.percentile(pooled, 99.5))

    # Second pass: draw KDE curves.
    for cond, (vals, n) in per_cond_vals.items():
        color = CONDITION_COLORS.get(cond, '#888888')
        if vals.size > 3:
            try:
                xs = np.linspace(vals.min(), vals.max(), 500)
                ys = _kde(vals, bw_method=kde_bw)(xs)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=f'{cond}  (n={n})')
                ax.fill_between(xs, ys, alpha=0.12, color=color)
            except Exception:
                ax.hist(vals, bins=bins, density=True, color=color,
                        alpha=0.35, label=f'{cond}  (n={n})')
        else:
            ax.hist(vals, bins=bins, density=True, color=color,
                    alpha=0.35, label=f'{cond}  (n={n})')

        ax.axvline(np.median(vals), color=color, linewidth=1.0,
                   linestyle='--', alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{field}{channel_index}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = save_name or f'{field}ch{channel_index}'
        plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        print(f"Saved {stem}.svg / .png → {output_dir}")
    plt.close()


# ── ACF overlay plot ───────────────────────────────────────────────────────────

def plot_acf_comparison(
    acf_results: list,
    param_tag: str,
    output_dir,
    x_max: float = 800.0,
    y_lim: tuple = (-0.02, 0.06),
    figsize: tuple = FIG_SIZE_ACF,
) -> None:
    """Generate one overlay ACF plot per channel from acf_results.

    Args:
        acf_results: List of dicts with keys: condition, condition_color,
            channel_index, lags, mean_correlation, std_correlation,
            dwell_time, number_of_cells_final, number_of_trajectories_final.
        param_tag: Short parameter identifier embedded in the filename.
        output_dir: Destination directory for saved SVG/PNG files.
        x_max: Upper x-axis limit in seconds.
        y_lim: Tuple (y_min, y_max) for the ACF y-axis.
    """
    if not acf_results:
        print("  [ACF plot] No ACF results to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = sorted({r['channel_index'] for r in acf_results})

    for ch in channels:
        ch_results = [r for r in acf_results if r['channel_index'] == ch]
        if not ch_results:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        for r in ch_results:
            cond        = r['condition']
            color       = r.get('condition_color', '#888888')
            lags        = np.asarray(r['lags'])
            mean_corr   = np.asarray(r['mean_correlation'])
            std_corr    = np.asarray(r['std_correlation'])
            dwell_time  = r.get('dwell_time', float('nan'))
            n_cells     = r.get('number_of_cells_final', '?')
            n_traces    = r.get('number_of_trajectories_final', '?')

            mask = (lags > 0) & (lags <= x_max)
            try:
                dwell_float = float(dwell_time)
            except (TypeError, ValueError):
                dwell_float = float('nan')
            label = f'{cond} |  {n_cells} cells  {n_traces} traces'

            ax.plot(lags[mask], mean_corr[mask], color=color, linewidth=1.8, label=label)
            ax.fill_between(
                lags[mask],
                mean_corr[mask] - std_corr[mask],
                mean_corr[mask] + std_corr[mask],
                color=color, alpha=0.15,
            )

        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_xlabel('τ (s)')
        ax.set_ylabel('G(τ)')
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(*y_lim)

        plt.tight_layout()

        stem = f'acf_ch{ch}_{param_tag}'
        plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved {stem}.svg / .png → {output_dir}")
        plt.close()


def plot_acf_individual(
    acf_results: list,
    param_tag: str,
    output_dir,
    x_max: float = 800.0,
    y_lim: tuple = (-0.02, 0.06),
    figsize: tuple = FIG_SIZE_ACF,
) -> None:
    """Generate one separate ACF figure per condition per channel.

    Same visual style as plot_acf_comparison but each condition gets
    its own standalone plot, making it easier to inspect individual curves.

    Args:
        acf_results: List of dicts (same format as plot_acf_comparison).
        param_tag: Short parameter identifier embedded in the filename.
        output_dir: Destination directory for saved SVG/PNG files.
        x_max: Upper x-axis limit in seconds.
        y_lim: Tuple (y_min, y_max) for the ACF y-axis.
        figsize: Figure size for each individual plot.
    """
    if not acf_results:
        print("  [ACF individual] No ACF results to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in acf_results:
        cond   = r['condition']
        ch     = r['channel_index']
        color  = r.get('condition_color', '#888888')
        lags   = np.asarray(r['lags'])
        mean_corr = np.asarray(r['mean_correlation'])
        std_corr  = np.asarray(r['std_correlation'])
        dwell_time = r.get('dwell_time', float('nan'))
        n_cells    = r.get('number_of_cells_final', '?')
        n_traces   = r.get('number_of_trajectories_final', '?')

        mask = (lags > 0) & (lags <= x_max)
        try:
            dwell_float = float(dwell_time)
        except (TypeError, ValueError):
            dwell_float = float('nan')
        label = f'{cond} |  {n_cells} cells  {n_traces} traces'

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lags[mask], mean_corr[mask], color=color, linewidth=1.8, label=label)
        ax.fill_between(
            lags[mask],
            mean_corr[mask] - std_corr[mask],
            mean_corr[mask] + std_corr[mask],
            color=color, alpha=0.15,
        )
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_xlabel('τ (s)')
        ax.set_ylabel('G(τ)')
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(*y_lim)

        plt.tight_layout()

        stem = f'acf_{cond}_ch{ch}_{param_tag}'
        plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved {stem}.svg / .png → {output_dir}")
        plt.close()


def load_acf_results_from_disk(acf_dir: Path) -> list:
    """Reconstruct the acf_results list-of-dicts from saved CSVs on disk.

    Reads acf_summary_all_conditions.csv for metadata, then loads each
    per-condition curve CSV for lags/mean/std arrays.

    Args:
        acf_dir: Path to the acf/ folder (e.g. results/sz5_fast_peak/acf/).

    Returns:
        List of dicts compatible with plot_acf_comparison().
    """
    summary_path = acf_dir / 'acf_summary_all_conditions.csv'
    if not summary_path.exists():
        print(f"  [ACF loader] Summary not found: {summary_path}")
        return []

    summary_df = pd.read_csv(summary_path)
    acf_results = []
    for _, row in summary_df.iterrows():
        curve_path = Path(row['curve_csv_path'])
        if not curve_path.exists():
            # Try relative to acf_dir (in case paths were saved on another machine)
            curve_path = acf_dir / curve_path.name
        if not curve_path.exists():
            print(f"  [ACF loader] Curve CSV not found: {row['curve_csv_path']}, skipping.")
            continue
        curve_df = pd.read_csv(curve_path)
        # Apply rename and resolve color from current CONDITION_COLORS.
        cond_name = CONDITION_RENAME.get(row['condition'], row['condition'])
        cond_color = CONDITION_COLORS.get(cond_name, row.get('condition_color', '#888888'))
        acf_results.append({
            'condition':                    cond_name,
            'condition_color':              cond_color,
            'channel_index':                int(row['channel_index']),
            'lags':                         curve_df['lags'].to_numpy(),
            'mean_correlation':             curve_df['mean_correlation'].to_numpy(),
            'std_correlation':              curve_df['std_correlation'].to_numpy(),
            'dwell_time':                   row.get('dwell_time', float('nan')),
            'number_of_cells_final':        row.get('number_of_cells_final', '?'),
            'number_of_trajectories_final': row.get('number_of_trajectories_final', '?'),
        })
    print(f"  [ACF loader] Loaded {len(acf_results)} condition(s) from {acf_dir}")
    return acf_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CSV_PATH} ...")
    master_df = pd.read_csv(CSV_PATH)
    if CONDITION_RENAME:
        master_df['condition'] = master_df['condition'].replace(CONDITION_RENAME)
    print(f"Loaded: {len(master_df)} rows, {master_df['condition'].nunique()} conditions")
    print(f"Conditions: {master_df['condition'].unique().tolist()}")


    # ── PSF amplitude vs sigma ────────────────────────────────────────────────
    for ch in [0, 1]:
        print(f"\n[PSF plot] channel {ch}")
        plot_psf_amplitude_vs_sigma(master_df, channel_index=ch, output_dir=PLOTS_DIR)

    # ── SNR distribution ──────────────────────────────────────────────────────
    for ch in [0, 1]:
        print(f"\n[SNR distribution] channel {ch}")
        plot_intensity_distributions(
            master_df,
            field         = 'snr_ch_',
            channel_index = ch,
            min_snr       = MIN_SNR,
            x_label       = 'SNR',
            xlim          = (2.5, 18),
            output_dir    = PLOTS_DIR,
        )

    # ── Spot intensity distribution ──────────────────────────────────────────
    for ch in [0, 1]:
        print(f"\n[Spot intensity distribution] channel {ch}")
        plot_intensity_distributions(
            master_df,
            field         = 'spot_int_ch_',
            channel_index = ch,
            min_snr       = MIN_SNR,
            x_label       = 'Spot Intensity (a.u.)',
            xlim          = None,
            output_dir    = PLOTS_DIR,
        )

    # ── ACF overlay (from saved CSVs on disk) ─────────────────────────────────
    acf_dir = CSV_PATH.parent / 'acf'
    if acf_dir.exists():
        param_tag = CSV_PATH.parent.name  # e.g. 'sz5_fast_peak'
        acf_results = load_acf_results_from_disk(acf_dir)
        plot_acf_comparison(acf_results, param_tag=param_tag, output_dir=PLOTS_DIR)
        plot_acf_individual(acf_results, param_tag=param_tag, output_dir=PLOTS_DIR)
    else:
        print(f"\n[ACF] No acf/ folder found at {acf_dir}, skipping.")
