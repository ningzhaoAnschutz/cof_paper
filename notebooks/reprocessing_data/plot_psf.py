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
# Stored as a dict so any function that lazy-imports microlive (which overrides
# rcParams with a gray background + grid) can call apply_plot_style() to restore.
PLOT_STYLE = {
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
}

def apply_plot_style():
    """Re-apply the project plot style (call after any microlive import)."""
    plt.rcParams.update(PLOT_STYLE)

apply_plot_style()

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


# ── ACF with model fit + individual traces ──────────────────────────────────

def _setup_acf_fit_axes(ax, lags, mean_corr, std_corr, color, n_cells, n_traces,
                        name, x_max,
                        correlations_array=None, show_individual=False):
    """Draw ACF data + SEM band and style axes. Returns taus_fit for model curves."""
    mask = lags > 0

    # Individual per-trajectory ACFs (plotted first, behind everything)
    if show_individual and correlations_array is not None:
        for i in range(correlations_array.shape[0]):
            ax.plot(lags[mask], correlations_array[i, mask],
                    color=color, linewidth=0.4, alpha=0.15, zorder=1)

    # Mean ACF + SEM band
    ax.plot(lags[mask], mean_corr[mask],
            color=color, linewidth=1.8, zorder=3,
            label=f'{name} (n = {n_traces})')
    ax.fill_between(lags[mask],
                    mean_corr[mask] - std_corr[mask],
                    mean_corr[mask] + std_corr[mask],
                    color=color, alpha=0.15, zorder=2)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_xlabel('τ (s)')
    ax.set_ylabel('G(τ)')
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 200))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    taus_fit = np.linspace(lags[mask].min(), x_max, 500)
    return taus_fit


def _save_acf_and_close(fig, ax, output_path):
    """Add legend, tight-layout, save PNG + SVG, and close."""
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=2, framealpha=0.9, edgecolor='black', fontsize=10)
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path.with_suffix(".png").name}')


def plot_acf_exponential_fit(
    result: dict,
    output_path,
    x_max: float = 800.0,
    show_individual: bool = True,
    figsize: tuple = (7, 4.5),
) -> None:
    """Plot ACF data for one condition with exponential fit overlay.

    Optionally shows individual per-trajectory ACF curves behind the mean.
    The result dict must have keys: lags, mean_correlation, std_correlation,
    correlations_array, fit_params_, condition, condition_color,
    number_of_cells_final, number_of_trajectories_final.
    """
    lags      = np.asarray(result['lags'])
    mean_corr = np.asarray(result['mean_correlation'])
    std_corr  = np.asarray(result['std_correlation'])
    name      = result['condition']
    color     = result.get('condition_color') or CONDITION_COLORS.get(name, '#888888')
    n_cells   = result['number_of_cells_final']
    n_traces  = result['number_of_trajectories_final']
    fit_exp   = result.get('fit_params_')
    corr_arr  = np.asarray(result.get('correlations_array', [])) if show_individual else None

    fig, ax = plt.subplots(figsize=figsize)
    taus_fit = _setup_acf_fit_axes(ax, lags, mean_corr, std_corr, color,
                                   n_cells, n_traces, name, x_max,
                                   correlations_array=corr_arr,
                                   show_individual=show_individual)

    if fit_exp is not None and isinstance(fit_exp, dict):
        A  = fit_exp.get('A', 0)
        tc = fit_exp.get('tau_c', 1)
        C  = fit_exp.get('C', 0)
        dwell_time = 2.0 * tc
        G_exp = A * np.exp(-taus_fit / tc) + C
        ax.plot(taus_fit, G_exp, '--', color='dimgray', linewidth=1.5,
                label=f'τ_c = {tc:.0f} s,  T_dwell = {dwell_time:.0f} s',
                zorder=4)
        ax.plot(taus_fit, np.full_like(taus_fit, C), ':', color='gray',
                linewidth=1.0, alpha=0.6, label=f'Baseline C = {C:.4f}')
        # Dwell-time marker
        y_dwell = A * np.exp(-dwell_time / tc) + C
        ax.plot(dwell_time, y_dwell, 'o',
                color='red', markersize=9, zorder=7)

    _save_acf_and_close(fig, ax, output_path)


def plot_acf_heaviside_fit(
    result: dict,
    output_path,
    x_max: float = 800.0,
    show_individual: bool = True,
    figsize: tuple = (7, 4.5),
) -> None:
    """Plot ACF data for one condition with Heaviside (Larson 2011) fit overlay.

    The result dict must have keys: lags, mean_correlation, std_correlation,
    correlations_array, hfit, kin_hev, condition, condition_color,
    number_of_cells_final, number_of_trajectories_final.
    """
    # Import the model function from runner to avoid code duplication
    try:
        from runner import heaviside_acf_model
    except ImportError:
        import importlib, sys
        spec = importlib.util.spec_from_file_location(
            'runner', Path(__file__).parent / 'runner.py')
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        heaviside_acf_model = _mod.heaviside_acf_model
    apply_plot_style()  # runner imports microlive which overrides rcParams

    lags      = np.asarray(result['lags'])
    mean_corr = np.asarray(result['mean_correlation'])
    std_corr  = np.asarray(result['std_correlation'])
    name      = result['condition']
    color     = result.get('condition_color') or CONDITION_COLORS.get(name, '#888888')
    n_cells   = result['number_of_cells_final']
    n_traces  = result['number_of_trajectories_final']
    hfit      = result.get('hfit')
    kin_hev   = result.get('kin_hev')
    corr_arr  = np.asarray(result.get('correlations_array', [])) if show_individual else None

    fig, ax = plt.subplots(figsize=figsize)
    taus_fit = _setup_acf_fit_axes(ax, lags, mean_corr, std_corr, color,
                                   n_cells, n_traces, name, x_max,
                                   correlations_array=corr_arr,
                                   show_individual=show_individual)

    if hfit is not None:
        G_hev = heaviside_acf_model(taus_fit, hfit['A'], hfit['T'], hfit['C'])
        T_err_str = f" ± {hfit['T_err']:.0f}" if 'T_err' in hfit else ""
        ax.plot(taus_fit, G_hev, '--', color='tab:blue', linewidth=1.5,
                label=f'T = {hfit["T"]:.0f}{T_err_str} s',
                zorder=4)
        ax.plot(taus_fit, np.full_like(taus_fit, hfit['C']), ':', color='gray',
                linewidth=1.0, alpha=0.6, label=f'Baseline C = {hfit["C"]:.4f}')
        # Dwell-time marker at T (where function drops to C)
        ke_str = f'{kin_hev["ke"]:.2f}' if kin_hev else '?'
        ki_str = f'{kin_hev["ki"]:.4f}' if kin_hev else '?'
        ax.plot(hfit['T'], hfit['C'], 'D',
                color='tab:blue', markersize=9, zorder=7,
                label=f'kₑ = {ke_str} aa/s,  kᵢ = {ki_str} 1/s')

    _save_acf_and_close(fig, ax, output_path)


# ── Per-trajectory intensity time courses ──────────────────────────────────

def plot_individual_trajectories_detrended(
    master_df: pd.DataFrame,
    condition: str,
    channel_index: int = 1,
    min_snr: float = 0.5,
    max_trajectories: int = 50,
    time_interval_s: float = 5.0,
    output_dir: Path = None,
    save_name: str = None,
    figsize: tuple = (7, 4.5),
    detrend_method: str = 'exponential',
    min_percentage_data: float = 0.3,
    max_missing_frames: int = 1,
    maximum_columns: int = 360,
) -> None:
    """Plot detrended intensity traces for one condition.

    Uses the same trajectory array construction and detrending method as
    the ACF pipeline (mi.Utilities.detrend_trajectories), so the plot
    shows exactly what the Correlation engine sees after photobleaching
    correction.

    Args:
        master_df: Output of runner.run_pipeline (long-format).
        condition: Condition name to filter on.
        channel_index: Which channel to plot (0 or 1).
        min_snr: Minimum mean-SNR filter per trajectory.
        max_trajectories: Max individual traces to draw.
        time_interval_s: Seconds per frame (for x-axis).
        output_dir: Folder to save SVG + PNG.
        save_name: File stem override.
        figsize: Figure dimensions.
        detrend_method: 'exponential' or 'linear' (passed to detrend_trajectories).
        min_percentage_data: Min fraction of non-NaN data required per trajectory.
        max_missing_frames: Max consecutive NaN frames allowed in a trajectory.
        maximum_columns: Column cap for the trajectory array.
    """
    from reprocess_intensities import build_correlation_input_from_reprocessed_df
    from microlive import microscopy as mi
    apply_plot_style()  # microlive overrides rcParams

    int_col = f'spot_int_ch_{channel_index}'
    df_cond = master_df[master_df['condition'] == condition].copy()
    if df_cond.empty:
        print(f"  [Detrended] No data for '{condition}' ch{channel_index}")
        return

    color = CONDITION_COLORS.get(condition, '#888888')

    # Build the same array the ACF pipeline uses
    try:
        prepared = build_correlation_input_from_reprocessed_df(
            df_cond,
            channel_index=channel_index,
            condition_name=condition,
            min_percentage_data_in_trajectory=min_percentage_data,
            max_missing_frames=max_missing_frames,
            maximum_columns=maximum_columns,
            min_snr=min_snr,
            smooth_window=1,
            verbose=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"  [Detrended] Skipping '{condition}' ch{channel_index}: {e}")
        return

    raw_array = prepared['primary_data']  # (n_traj, n_cols), left-aligned

    # Apply the same detrending as mi.Correlation
    utilities = mi.Utilities()
    detrended_array = utilities.detrend_trajectories(raw_array, method=detrend_method)

    n_traj = detrended_array.shape[0]
    n_cols = detrended_array.shape[1]
    n_plot = min(n_traj, max_trajectories)

    # Pick a random subset for plotting
    rng = np.random.RandomState(42)
    plot_indices = rng.choice(n_traj, size=n_plot, replace=False) if n_traj > n_plot else np.arange(n_traj)
    times = np.arange(n_cols) * time_interval_s

    fig, ax = plt.subplots(figsize=figsize)

    # Individual traces
    for idx in plot_indices:
        row = detrended_array[idx]
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        ax.plot(times[valid], row[valid],
                color=color, linewidth=0.6, alpha=0.25, zorder=1)

    # Mean ± SEM overlay
    mean_trace = np.nanmean(detrended_array, axis=0)
    n_per_col = np.sum(np.isfinite(detrended_array), axis=0)
    sem_trace = np.nanstd(detrended_array, axis=0) / np.sqrt(np.maximum(n_per_col, 1))
    valid_cols = n_per_col >= 3  # only plot mean where ≥3 traces contribute

    ax.fill_between(times[valid_cols],
                    mean_trace[valid_cols] - sem_trace[valid_cols],
                    mean_trace[valid_cols] + sem_trace[valid_cols],
                    color=color, alpha=0.2, zorder=2)
    ax.plot(times[valid_cols], mean_trace[valid_cols],
            color=color, linewidth=2.0, zorder=3,
            label=f'Mean detrended (n = {n_traj})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Intensity ch{channel_index} (a.u.)')
    ax.set_title(f'{condition}  —  ch{channel_index}  (detrended)',
                 fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = save_name or f'trajectories_detrended_{condition}_ch{channel_index}'
        plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved {stem}.svg / .png → {output_dir}")
    plt.close()


def plot_individual_trajectories(
    master_df: pd.DataFrame,
    condition: str,
    channel_index: int = 1,
    min_snr: float = 0.5,
    max_trajectories: int = 50,
    time_interval_s: float = 5.0,
    output_dir: Path = None,
    save_name: str = None,
    figsize: tuple = (7, 4.5),
    show_mean: bool = True,
) -> None:
    """Plot individual particle intensity time courses for one condition.

    Draws up to `max_trajectories` individual traces in light color behind
    a thick mean ± SEM overlay.  Useful for visual inspection of trajectory
    quality, photobleaching, and signal heterogeneity.

    Args:
        master_df: Output of runner.run_pipeline (long-format).
        condition: Condition name to filter on.
        channel_index: Which channel to plot (0 or 1).
        min_snr: Minimum SNR filter.
        max_trajectories: Maximum number of individual traces to draw.
        time_interval_s: Time between frames (for x-axis in seconds).
        output_dir: Folder to save SVG + PNG.
        save_name: File stem override.
        figsize: Figure dimensions.
        show_mean: Whether to overlay the mean ± SEM.
    """
    int_col = f'spot_int_ch_{channel_index}'
    snr_col = f'snr_ch_{channel_index}'

    df = master_df[master_df['condition'] == condition].copy()
    if snr_col in df.columns:
        df = df[df[snr_col] >= min_snr]
    df = df[df[int_col].notna() & np.isfinite(df[int_col]) & (df[int_col] > 0)]

    if df.empty:
        print(f"  [Trajectories] No valid data for '{condition}' ch{channel_index}")
        return

    color = CONDITION_COLORS.get(condition, '#888888')

    # Pivot to (particle, frame) matrix
    particles = df['global_particle_id'].unique()
    total_particles = len(particles)

    fig, ax = plt.subplots(figsize=figsize)

    all_traces = []
    for pid in particles:
        sub = df[df['global_particle_id'] == pid].sort_values('frame')
        frames = sub['frame'].to_numpy()
        values = sub[int_col].to_numpy()
        times = frames * time_interval_s
        ax.plot(times, values, color=color, linewidth=0.6, alpha=0.25, zorder=1)
        all_traces.append(pd.Series(values, index=frames, name=pid))

    # Mean ± SEM overlay
    if show_mean and all_traces:
        traces_df = pd.concat(all_traces, axis=1).sort_index()
        mean_trace = traces_df.mean(axis=1)
        sem_trace = traces_df.sem(axis=1)
        t_mean = mean_trace.index.to_numpy() * time_interval_s
        ax.fill_between(t_mean,
                        mean_trace.values - sem_trace.values,
                        mean_trace.values + sem_trace.values,
                        color=color, alpha=0.2, zorder=2)
        ax.plot(t_mean, mean_trace.values, color=color, linewidth=2.0, zorder=3,
                label=f'Mean (n = {total_particles})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Intensity ch{channel_index} (a.u.)')
    ax.set_title(f'{condition}  —  ch{channel_index}', fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = save_name or f'trajectories_{condition}_ch{channel_index}'
        plt.savefig(output_dir / f'{stem}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved {stem}.svg / .png → {output_dir}")
    plt.close()


def load_acf_results_from_disk(acf_dir: Path) -> list:
    """Reconstruct the acf_results list-of-dicts from saved CSVs on disk.

    Reads acf_summary_all_conditions.csv for metadata + fit parameters,
    then loads each per-condition curve CSV for lags/mean/std arrays.

    Args:
        acf_dir: Path to the acf/ folder (e.g. results/sz5_fast_peak/acf/).

    Returns:
        List of dicts compatible with plot_acf_comparison() and
        plot_acf_individual_fits().
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

        # Build fit_params_ dict if exponential fit was stored
        fit_params_ = None
        if not pd.isna(row.get('fit_A')) and not pd.isna(row.get('fit_tau_c')):
            fit_params_ = {
                'A': float(row['fit_A']),
                'tau_c': float(row['fit_tau_c']),
                'C': float(row.get('fit_C', 0)),
            }

        # Build hfit dict if Heaviside fit was stored
        hfit = None
        if not pd.isna(row.get('hev_A')) and not pd.isna(row.get('hev_T')):
            hfit = {
                'A': float(row['hev_A']),
                'T': float(row['hev_T']),
                'C': float(row.get('hev_C', 0)),
                'A_err': float(row.get('hev_A_err', 0)),
                'T_err': float(row.get('hev_T_err', 0)),
                'C_err': float(row.get('hev_C_err', 0)),
            }

        # Build kin_hev dict
        kin_hev = None
        if not pd.isna(row.get('hev_ke')) and not pd.isna(row.get('hev_ki')):
            kin_hev = {
                'ke': float(row['hev_ke']),
                'ki': float(row['hev_ki']),
                'T_dwell': float(row.get('hev_dwell_time', row.get('hev_T', 0))),
            }

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
            'fit_params_':                  fit_params_,
            'hfit':                         hfit,
            'kin_hev':                      kin_hev,
            'gene_length_half_HA':          row.get('gene_length_half_HA', 1659),
        })
    print(f"  [ACF loader] Loaded {len(acf_results)} condition(s) from {acf_dir}")
    return acf_results


def plot_acf_individual_fits(
    acf_results: list,
    param_tag: str,
    output_dir,
    x_max: float = 1000.0,
) -> None:
    """Generate per-condition ACF plots with exponential AND Heaviside fit overlays.

    Matches the style of acf_individual.py: one plot per condition per model,
    with fit curves, parameter labels, dwell-time markers, and kinetics in
    the legend at the top.

    Works entirely from saved data (no in-memory correlations_array needed).

    Args:
        acf_results: List of dicts from load_acf_results_from_disk or run_pipeline.
        param_tag: Short parameter identifier embedded in filenames.
        output_dir: Destination directory for saved SVG/PNG files.
        x_max: Upper x-axis limit in seconds.
    """
    from runner import heaviside_acf_model
    apply_plot_style()  # runner imports microlive which overrides rcParams

    if not acf_results:
        print("  [ACF fits] No ACF results to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in acf_results:
        cond     = r['condition']
        ch       = r['channel_index']
        color    = r.get('condition_color', '#888888')
        lags     = np.asarray(r['lags'])
        mean_corr = np.asarray(r['mean_correlation'])
        std_corr  = np.asarray(r['std_correlation'])
        n_cells   = r.get('number_of_cells_final', '?')
        n_traces  = r.get('number_of_trajectories_final', '?')
        fit_exp   = r.get('fit_params_')
        hfit      = r.get('hfit')
        kin_hev   = r.get('kin_hev')
        dwell_time = r.get('dwell_time', float('nan'))
        gene_len  = r.get('gene_length_half_HA', 1659)

        mask = lags > 0
        taus_fit = np.linspace(lags[mask].min(), x_max, 500)
        safe = cond.replace(' ', '_')

        # ── Individual per-trajectory ACFs (if available) ─────────────
        corr_arr = r.get('correlations_array')

        # ── EXPONENTIAL FIT PLOT ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))

        if corr_arr is not None:
            for i in range(corr_arr.shape[0]):
                ax.plot(lags[mask], corr_arr[i, mask],
                        color=color, linewidth=0.4, alpha=0.15, zorder=1)

        ax.plot(lags[mask], mean_corr[mask], color=color, linewidth=1.8, zorder=3,
                label=f'{cond} (n = {n_traces})')
        ax.fill_between(lags[mask],
                        mean_corr[mask] - std_corr[mask],
                        mean_corr[mask] + std_corr[mask],
                        color=color, alpha=0.15, zorder=2)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)

        if fit_exp is not None:
            A  = fit_exp['A']
            tc = fit_exp['tau_c']
            C  = fit_exp['C']
            G_exp = A * np.exp(-taus_fit / tc) + C
            ax.plot(taus_fit, G_exp, '--', color='dimgray', linewidth=1.5,
                    label=f'τ_c = {tc:.0f} s,  T_dwell = {dwell_time:.0f} s',
                    zorder=4)
            ax.plot(taus_fit, np.full_like(taus_fit, C), ':', color='gray',
                    linewidth=1.0, alpha=0.6, label=f'Baseline C = {C:.4f}')
            try:
                dw = float(dwell_time)
                exp_ke = gene_len / dw if dw > 0 else float('nan')
            except (TypeError, ValueError):
                exp_ke = float('nan')
            y_dwell = A * np.exp(-dwell_time / tc) + C
            ax.plot(dwell_time, y_dwell, 'o', color='red', markersize=9, zorder=7,
                    label=f'kₑ = {exp_ke:.2f} aa/s')

        ax.set_xlabel('τ (s)'); ax.set_ylabel('G(τ)')
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.arange(0, x_max + 1, 200))
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=2, framealpha=0.9, edgecolor='black', fontsize=10)
        plt.tight_layout()
        out = output_dir / f'acf_{safe}_ch{ch}_exponential_{param_tag}'
        plt.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.savefig(out.with_suffix('.svg'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out.with_suffix(".png").name}')

        # ── HEAVISIDE FIT PLOT ────────────────────────────────────────
        if hfit is None:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        if corr_arr is not None:
            for i in range(corr_arr.shape[0]):
                ax.plot(lags[mask], corr_arr[i, mask],
                        color=color, linewidth=0.4, alpha=0.15, zorder=1)

        ax.plot(lags[mask], mean_corr[mask], color=color, linewidth=1.8, zorder=3,
                label=f'{cond} (n = {n_traces})')
        ax.fill_between(lags[mask],
                        mean_corr[mask] - std_corr[mask],
                        mean_corr[mask] + std_corr[mask],
                        color=color, alpha=0.15, zorder=2)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)

        G_hev = heaviside_acf_model(taus_fit, hfit['A'], hfit['T'], hfit['C'])
        ax.plot(taus_fit, G_hev, '--', color='tab:blue', linewidth=1.5,
                label=f'T = {hfit["T"]:.0f} ± {hfit["T_err"]:.0f} s',
                zorder=4)
        ax.plot(taus_fit, np.full_like(taus_fit, hfit['C']), ':', color='gray',
                linewidth=1.0, alpha=0.6, label=f'Baseline C = {hfit["C"]:.4f}')
        if kin_hev is not None:
            ax.plot(hfit['T'], hfit['C'], 'D', color='tab:blue', markersize=9, zorder=7,
                    label=f'kₑ = {kin_hev["ke"]:.2f} aa/s,  kᵢ = {kin_hev["ki"]:.4f} 1/s')
        else:
            ax.plot(hfit['T'], hfit['C'], 'D', color='tab:blue', markersize=9, zorder=7)

        ax.set_xlabel('τ (s)'); ax.set_ylabel('G(τ)')
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.arange(0, x_max + 1, 200))
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=2, framealpha=0.9, edgecolor='black', fontsize=10)
        plt.tight_layout()
        out = output_dir / f'acf_{safe}_ch{ch}_heaviside_{param_tag}'
        plt.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.savefig(out.with_suffix('.svg'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out.with_suffix(".png").name}')


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
        plot_acf_individual_fits(acf_results, param_tag=param_tag, output_dir=PLOTS_DIR)
    else:
        print(f"\n[ACF] No acf/ folder found at {acf_dir}, skipping.")
