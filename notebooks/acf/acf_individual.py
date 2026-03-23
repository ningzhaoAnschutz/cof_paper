#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Individual ACF Analysis

Computes the ACF for each dataset separately, plots each with its exponential
fit curve and dwell-time marker, and saves a kinetics summary CSV.

Outputs are saved to: notebooks/acf/results_individual/
"""

# ============================================================
# IMPORTS
# ============================================================
import sys
import os
from pathlib import Path

# Auto-relaunch in microlive conda environment if not already active
MICROLIVE_ENV = '/opt/anaconda3/envs/microlive'
if sys.prefix != MICROLIVE_ENV:
    python_exe = os.path.join(MICROLIVE_ENV, 'bin', 'python')
    if os.path.exists(python_exe):
        print(f"Relaunching script using the microlive environment Python...\n")
        os.execl(python_exe, python_exe, *sys.argv)
    else:
        print(f"Warning: microlive environment not found at {MICROLIVE_ENV}. Proceeding with current python.\n")

# --- Third-party imports (safe after env check) ---
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib

current_dir = Path(__file__).resolve().parent

# pipeline_time_courses lives in the sibling autocorrelations/ folder
sys.path.insert(0, str(current_dir.parent / 'autocorrelations'))

from microlive.imports import *
from microlive import microscopy as mi
from pipeline_time_courses import compute_autocorrelation_for_dataset, compute_kinetics_from_acf

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'Arial',
    'text.color':         'black',
    'axes.labelcolor':    'black',
    'xtick.color':        'black',
    'ytick.color':        'black',
    'axes.edgecolor':     'black',
    'axes.linewidth':     1.5,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.grid':          False,
    'axes.labelsize':     16,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'legend.fontsize':    12,
})

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
output_dir = current_dir / 'results_individual'
output_dir.mkdir(exist_ok=True)

# ============================================================
# DATASET PATHS
# ============================================================
data_folder_sf         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/sfGFP/results')
data_folder_uv         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv/results')
data_folder_end_xbp1   = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS038/results')
data_folder_start_xbp1 = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS048/results')

list_datasets = [data_folder_sf, data_folder_uv, data_folder_end_xbp1, data_folder_start_xbp1]
list_names    = ['sfGFP', 'GFPuv', 'sfGFP_ex', 'sfGFP_sx']
list_colors   = ['gray', 'tab:orange', 'tab:blue', 'tab:purple']

# ============================================================
# SHARED ACF PARAMETERS
# ============================================================
step_size_in_sec                  = 5
start_lag                         = 1
channel_index                     = 1
selected_field                    = 'spot_int_ch_'
min_percentage_data_in_trajectory = 0.3
max_missing_frames                = 1
downsample                        = False
downsampling_factor               = 3
control_spots_mode                = False
use_global_mean                   = False
MAD_THRESHOLD_FACTOR              = 4
multi_tau_raw_points              = 60
multi_tau_bins_per_stage          = 16
min_snr                           = 1
smooth_window                     = 1
remove_outliers                   = False
correct_baseline                  = True
multi_tau                         = True
max_lag                           = 240        # 240 × 5s = 1200s
x_axes_min_max_list_values        = [-10, 1000]
y_axes_min_max_list_values        = [-0.02, 0.04]
fit_type                          = 'exponential'
de_correlation_threshold          = 0.001
use_linear_projection_for_lag_0   = True
gene_length                       = 1826       # codons
gen_length_half_HA                = 1659        # codons

# Plot x-axis limit (derived from user setting above)
X_MAX_PLOT = x_axes_min_max_list_values[1]

# Show individual per-trajectory ACF curves behind the mean?
PLOT_INDIVIDUAL_TRACES = True

# Per-trajectory exponential photobleaching detrend before ACF?
DETREND_PHOTOBLEACHING = False


# ============================================================
# HEAVISIDE (LARSON 2011) ACF MODEL
# ============================================================
def heaviside_acf_model(tau, A, T, C):
    """Larson 2011 Eq. 1 with baseline offset.

    G(τ) = A · (1 − τ/T) · H(T − τ)  +  C

    Parameters
    ----------
    tau : array-like   – lag times (seconds)
    A   : float        – amplitude  (≈ 1/c, inverse initiation rate)
    T   : float        – dwell time (seconds)
    C   : float        – baseline offset
    """
    tau = np.asarray(tau, dtype=float)
    out = np.where(tau <= T, A * (1.0 - tau / T) + C, C)
    return out


def fit_heaviside_acf(lags, mean_corr, start_lag_idx=1):
    """Fit the Heaviside ACF model to data.

    Returns
    -------
    params : dict  with keys 'A', 'T', 'C'   (or None on failure)
    """
    lags = np.asarray(lags, dtype=float)
    mc   = np.asarray(mean_corr, dtype=float)

    # Use only positive lags starting from start_lag_idx
    sl = max(start_lag_idx, 1)
    T_vals = lags[sl:]
    G_vals = mc[sl:]

    # Remove NaNs
    good = np.isfinite(G_vals) & np.isfinite(T_vals)
    T_vals = T_vals[good]
    G_vals = G_vals[good]
    if len(T_vals) < 4:
        return None

    # --- Initial guesses ---
    # Baseline C: mean of the last 20% of data points
    tail = max(1, len(G_vals) // 5)
    C0 = float(np.mean(G_vals[-tail:]))
    # Amplitude A: value at first lag minus baseline
    A0 = max(float(G_vals[0]) - C0, 1e-8)
    # Dwell time T: find where G falls to baseline level
    crossings = np.where(G_vals <= C0)[0]
    if len(crossings) > 0:
        T0 = float(T_vals[crossings[0]])
    else:
        T0 = float(T_vals[-1] / 2)
    T0 = max(T0, 10.0)  # at least 10 seconds

    try:
        popt, pcov = curve_fit(
            heaviside_acf_model,
            T_vals, G_vals,
            p0=[A0, T0, C0],
            bounds=([0, 1, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            'A': float(popt[0]),
            'T': float(popt[1]),
            'C': float(popt[2]),
            'A_err': float(perr[0]),
            'T_err': float(perr[1]),
            'C_err': float(perr[2]),
        }
    except Exception as e:
        print(f'    Heaviside fit failed: {e}')
        return None


def compute_kinetics_heaviside(hfit, mean_corr, gene_length,
                               ribosomal_footprint=10):
    """Derive kinetics from the Heaviside fit (Larson 2011).

    Unlike the exponential model, T is the dwell time directly
    (no factor-of-2 conversion).

    Initiation rate:  c = 1 / (G(0) · T)
    where G(0) = A (the fitted amplitude), per Larson 2011 Eq. 1.
    """
    T_dwell = hfit['T']            # dwell time = T directly
    ke = gene_length / T_dwell     # elongation rate (codons/s)
    A  = hfit['A']                 # fitted amplitude = G(0) from Heaviside
    ki = 1.0 / (A * T_dwell)      # initiation rate (1/s)  [Larson Eq.1]

    rho = (ki * ribosomal_footprint / ke) * 100
    n_rib = (ki * gene_length) / ke
    rib_dist = gene_length / n_rib if n_rib > 0 else np.nan

    return {
        'T_dwell': round(T_dwell, 2),
        'ke': round(ke, 4),
        'ki': round(ki, 4),
        'ribosomal_density': round(rho, 3),
        'n_ribosomes': round(n_rib, 3),
        'ribosomal_distance': round(rib_dist, 3),
    }


# ============================================================
# HELPER: Shared ACF data plotting (used by both plot functions)
# ============================================================
def _setup_acf_axes(ax, lags, mean_corr, std_corr, color, n_cells, n_traces,
                    name, model_label, x_max,
                    correlations_array=None, show_individual=False):
    """Draw ACF data + SEM band and style the axes. Returns taus_fit array."""
    mask = lags > 0  # plot all positive lags; set_xlim clips visually

    # ── Individual per-trajectory ACFs (plotted first, behind everything) ──
    if show_individual and correlations_array is not None:
        for i in range(correlations_array.shape[0]):
            ax.plot(lags[mask], correlations_array[i, mask],
                    color=color, linewidth=0.4, alpha=0.15, zorder=1)

    # ── Mean ACF + SEM band ───────────────────────────────────────────────
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


def _save_and_close(fig, ax, output_path):
    """Add legend, tight-layout, save PNG + SVG, and close."""
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=2, framealpha=0.9, edgecolor='black', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path.with_suffix(".png").name}')


# ============================================================
# Plot 1: Exponential model fit
# ============================================================
def plot_exponential_acf(result, kin_exp, name, color,
                         output_path, x_max=X_MAX_PLOT,
                         show_individual=False):
    """Plot ACF data with the exponential fit overlay only."""
    lags      = np.array(result['lags'])
    mean_corr = np.array(result['mean_correlation'])
    std_corr  = np.array(result['std_correlation'])
    n_cells   = result['number_of_cells_final']
    n_traces  = result['number_of_trajectories_final']
    fit_exp   = result.get('fit_params_')
    corr_arr  = np.array(result.get('correlations_array', [])) if show_individual else None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    taus_fit = _setup_acf_axes(ax, lags, mean_corr, std_corr, color,
                               n_cells, n_traces, name, 'Exponential', x_max,
                               correlations_array=corr_arr,
                               show_individual=show_individual)

    if fit_exp is not None:
        A  = fit_exp['A']
        tc = fit_exp['tau_c']
        C  = fit_exp['C']
        G_exp = A * np.exp(-taus_fit / tc) + C
        ax.plot(taus_fit, G_exp, '--', color='dimgray', linewidth=1.5,
                label=f'τ_c = {tc:.0f} s,  T_dwell = {kin_exp["dwell_time"]:.0f} s',
                zorder=4)
        ax.plot(taus_fit, np.full_like(taus_fit, C), ':', color='gray',
                linewidth=1.0, alpha=0.6, label=f'Baseline C = {C:.4f}')
        # Dwell-time marker
        y_dwell = A * np.exp(-kin_exp['dwell_time'] / tc) + C
        ax.plot(kin_exp['dwell_time'], y_dwell, 'o',
                color='red', markersize=9, zorder=7,
                label=f'kₑ = {kin_exp["ke"]:.2f} aa/s,  kᵢ = {kin_exp["ki"]:.4f} 1/s')

    _save_and_close(fig, ax, output_path)


# ============================================================
# Plot 2: Heaviside model fit
# ============================================================
def plot_heaviside_acf(result, hfit, kin_hev, name, color,
                       output_path, x_max=X_MAX_PLOT,
                       show_individual=False):
    """Plot ACF data with the Heaviside fit overlay only."""
    lags      = np.array(result['lags'])
    mean_corr = np.array(result['mean_correlation'])
    std_corr  = np.array(result['std_correlation'])
    n_cells   = result['number_of_cells_final']
    n_traces  = result['number_of_trajectories_final']
    corr_arr  = np.array(result.get('correlations_array', [])) if show_individual else None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    taus_fit = _setup_acf_axes(ax, lags, mean_corr, std_corr, color,
                               n_cells, n_traces, name, 'Heaviside', x_max,
                               correlations_array=corr_arr,
                               show_individual=show_individual)

    if hfit is not None:
        G_hev = heaviside_acf_model(taus_fit, hfit['A'], hfit['T'], hfit['C'])
        ax.plot(taus_fit, G_hev, '--', color='tab:blue', linewidth=1.5,
                label=f'T = {hfit["T"]:.0f} ± {hfit["T_err"]:.0f} s',
                zorder=4)
        ax.plot(taus_fit, np.full_like(taus_fit, hfit['C']), ':', color='gray',
                linewidth=1.0, alpha=0.6, label=f'Baseline C = {hfit["C"]:.4f}')
        # Dwell-time marker (at T the function drops to C)
        ax.plot(hfit['T'], hfit['C'], 'D',
                color='tab:blue', markersize=9, zorder=7,
                label=f'kₑ = {kin_hev["ke"]:.2f} aa/s,  kᵢ = {kin_hev["ki"]:.4f} 1/s')

    _save_and_close(fig, ax, output_path)


# ============================================================
# MAIN: Run ACF + plot + collect kinetics for each dataset
# ============================================================
all_results = []
kinetics_rows = []

for data_folder, name, color in zip(list_datasets, list_names, list_colors):

    print(f'\n{"="*60}')
    print(f'  {name}  →  {data_folder}')
    print(f'{"="*60}')

    r = compute_autocorrelation_for_dataset(
        dataset                          = 'cof',
        data_folder                      = data_folder,
        results_folder                   = None,
        save_results                     = False,
        selected_field                   = selected_field,
        channel_index                    = channel_index,
        step_size_in_sec                 = step_size_in_sec,
        start_lag                        = start_lag,
        min_percentage_data_in_trajectory= min_percentage_data_in_trajectory,
        max_missing_frames               = max_missing_frames,
        downsample                       = downsample,
        downsampling_factor              = downsampling_factor,
        use_global_mean                  = use_global_mean,
        control_spots_mode               = control_spots_mode,
        correct_baseline                 = correct_baseline,
        min_snr                          = min_snr,
        smooth_window                    = smooth_window,
        remove_outliers                  = remove_outliers,
        MAD_THRESHOLD_FACTOR             = MAD_THRESHOLD_FACTOR,
        multi_tau                        = multi_tau,
        multi_tau_raw_points             = multi_tau_raw_points,
        multi_tau_bins_per_stage         = multi_tau_bins_per_stage,
        x_axes_min_max_list_values       = x_axes_min_max_list_values,
        max_lag                          = max_lag,
        index_max_lag_for_fit            = None,
        fit_type                         = fit_type,
        de_correlation_threshold         = de_correlation_threshold,
        use_linear_projection_for_lag_0  = use_linear_projection_for_lag_0,
        verbose                          = False,
        simulation_mode                  = False,
        SSA_data                         = None,
        line_color                       = color,
        line_color_fit                   = 'dimgray',
        plot_name                        = None,
        save_plots                       = False,
        show_plot                        = False,
        figsize                          = (3.2, 2.2),
        detrend_photobleaching           = DETREND_PHOTOBLEACHING,
    )
    all_results.append(r)

    # ── Exponential kinetics (existing) ────────────────────────────────
    kin_exp = compute_kinetics_from_acf(r, gene_length=gen_length_half_HA)

    # ── Heaviside / Larson 2011 fit ───────────────────────────────────
    lags_arr = np.array(r['lags'])
    mc_arr   = np.array(r['mean_correlation'])
    hfit = fit_heaviside_acf(lags_arr, mc_arr, start_lag_idx=start_lag)

    kin_hev = None
    if hfit is not None:
        kin_hev = compute_kinetics_heaviside(
            hfit, mc_arr, gene_length=gen_length_half_HA)
        print(f'  Heaviside fit:  T = {hfit["T"]:.1f} ± {hfit["T_err"]:.1f} s,  '
              f'ke = {kin_hev["ke"]:.2f} aa/s,  ki = {kin_hev["ki"]:.4f} 1/s')
        print(f'  Exponential fit: T_dwell = {kin_exp["dwell_time"]:.1f} s  '
              f'(2·τ_c = 2×{kin_exp["tau_c"]:.1f}),  '
              f'ke = {kin_exp["ke"]:.2f} aa/s,  ki = {kin_exp["ki"]:.4f} 1/s')
    else:
        print(f'  Heaviside fit: FAILED')
        print(f'  Exponential fit: T_dwell = {kin_exp["dwell_time"]:.1f} s,  '
              f'ke = {kin_exp["ke"]:.2f} aa/s')

    # ── Plot 1: Exponential fit ─────────────────────────────────────────
    plot_exp_path = output_dir / f'ACF_{name}_ch{channel_index}_exponential'
    plot_exponential_acf(r, kin_exp, name, color, plot_exp_path,
                         show_individual=PLOT_INDIVIDUAL_TRACES)

    # ── Plot 2: Heaviside fit ─────────────────────────────────────────
    if hfit is not None and kin_hev is not None:
        plot_hev_path = output_dir / f'ACF_{name}_ch{channel_index}_heaviside'
        plot_heaviside_acf(r, hfit, kin_hev, name, color, plot_hev_path,
                           show_individual=PLOT_INDIVIDUAL_TRACES)

    # ── Collect row for CSV ────────────────────────────────────────────
    row = {
        'dataset':                 name,
        'n_cells':                 r['number_of_cells_final'],
        'n_traces':                r['number_of_trajectories_final'],
        'gene_length_codons':      gene_length,
        'half_gene_length_codons': gen_length_half_HA,
        # --- Exponential model ---
        'exp_A':                   r['fit_params_']['A'] if r.get('fit_params_') else None,
        'exp_tau_c_s':             kin_exp['tau_c'],
        'exp_C':                   r['fit_params_']['C'] if r.get('fit_params_') else None,
        'exp_dwell_time_s':        kin_exp['dwell_time'],
        'exp_ke':                  kin_exp['ke'],
        'exp_ki':                  kin_exp['ki'],
    }
    # --- Heaviside model ---
    if kin_hev is not None:
        row.update({
            'hev_A':              hfit['A'],
            'hev_T_s':            hfit['T'],
            'hev_T_err_s':        hfit['T_err'],
            'hev_C':              hfit['C'],
            'hev_ke':             kin_hev['ke'],
            'hev_ki':             kin_hev['ki'],
        })
    else:
        row.update({
            'hev_A': None, 'hev_T_s': None, 'hev_T_err_s': None,
            'hev_C': None, 'hev_ke': None, 'hev_ki': None,
        })
    kinetics_rows.append(row)


# ============================================================
# SAVE KINETICS SUMMARY CSV
# ============================================================
csv_path = output_dir / 'kinetics_summary.csv'
df_kinetics = pd.DataFrame(kinetics_rows)
df_kinetics.to_csv(csv_path, index=False)
print(f'\n{"="*60}')
print(f'  Kinetics summary saved → {csv_path}')
print(f'{"="*60}')
print(df_kinetics.to_string(index=False))
print()
