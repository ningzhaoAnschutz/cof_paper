#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACF Sensitivity Analysis: Elongation Rate vs (min_percentage_data, max_lag)

Sweeps over combinations of min_percentage_data_in_trajectory and max_lag,
computes the elongation rate (ke) for each gene construct, and generates
2D heatmaps showing the sensitivity of ke to these two parameters.

Supports two ACF fitting models:
  - 'exponential':  G(τ) = A·exp(−τ/τ_c) + C   →  T_dwell = 2·τ_c
  - 'heaviside':    G(τ) = A·(1 − τ/T)·H(T−τ) + C  →  T_dwell = T  (Larson 2011)

Optimization: data is loaded ONCE per (dataset, min_pct), then correlation
is run for each max_lag in parallel using joblib.

Outputs are saved to: notebooks/acf/results_sensitivity_<MODEL>_mad_<MAD>/
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal execution
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

current_dir = Path(__file__).resolve().parent

# pipeline_time_courses lives in the sibling autocorrelations/ folder
sys.path.insert(0, str(current_dir.parent / 'autocorrelations'))

from microlive.imports import *
from microlive import microscopy as mi
from pipeline_time_courses import (
    dataset_selection,
    load_tracking_data,
    compute_kinetics_from_acf,
)

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
# ★★★  USER-CONFIGURABLE FLAGS  ★★★
# ============================================================
# Model selection: 'exponential' or 'heaviside'
FIT_MODEL = 'heaviside'

# ============================================================
# DATASET PATHS
# ============================================================
# data_folder_sf         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/sfGFP/results')
# data_folder_uv         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv/results')
# data_folder_end_xbp1   = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS038/results')
# data_folder_start_xbp1 = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS048/results')

# list_datasets = [data_folder_sf, data_folder_uv, data_folder_end_xbp1, data_folder_start_xbp1]
# list_names    = ['sfGFP', 'GFPuv', 'sfGFP_ex', 'sfGFP_sx']
# list_colors   = ['gray', 'tab:orange', 'tab:blue', 'tab:purple']

data_folder_sf         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF_long_movies/sfGFP/results')
data_folder_uv         = Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF_long_movies/GFPuv/results')

list_datasets = [data_folder_sf, data_folder_uv]
list_names    = ['sfGFP', 'GFPuv']
list_colors   = ['gray', 'tab:orange']

# ============================================================
# FIXED ACF PARAMETERS (same as acf_individual.py)
# ============================================================
step_size_in_sec                  = 5
start_lag                         = 1
channel_index                     = 1
#selected_field                    = 'spot_int_ch_' # 'psf_amplitude_ch_', 'snr_ch_', 'total_spot_int_ch_'
selected_field                    = 'spot_int_ch_'
max_missing_frames                = 2
downsample                        = False
downsampling_factor               = 3
control_spots_mode                = False
use_global_mean                   = False
MAD_THRESHOLD_FACTOR              = 4
multi_tau_raw_points              = 60
multi_tau_bins_per_stage          = 16
min_snr                           = 0.5
smooth_window                     = 1
remove_outliers                   = True
correct_baseline                  = True
multi_tau                         = True
x_axes_min_max_list_values        = [-10, 1000]
y_axes_min_max_list_values        = [-0.02, 0.04]
fit_type                          = 'exponential'
de_correlation_threshold          = 0.001
use_linear_projection_for_lag_0   = True
DETREND_PHOTOBLEACHING            = False   # per-trajectory exponential detrend
max_trajectory_length_percentile  = None  # None = no filter, 99 = remove top 1%
gene_length                       = 1826       # codons
gen_length_half_HA                = 1659        # codons

# ============================================================
# OUTPUT DIRECTORY (dynamic: includes model + MAD)
# ============================================================
_pct_tag = f'_pct_{max_trajectory_length_percentile}' if max_trajectory_length_percentile is not None else ''
output_dir = current_dir / f'results_sensitivity_{FIT_MODEL}_mad_{MAD_THRESHOLD_FACTOR}_detrend_{DETREND_PHOTOBLEACHING}_out_{remove_outliers}{_pct_tag}'
output_dir.mkdir(exist_ok=True)
print(f"\n★  Output directory: {output_dir.name}")
print(f"★  Fit model:        {FIT_MODEL}")
print(f"★  MAD threshold:    {MAD_THRESHOLD_FACTOR}")
print(f"★  Detrend photobleaching:    {DETREND_PHOTOBLEACHING}\n")

# ============================================================
# SWEEP RANGES
# ============================================================
number_of_swap_points = 20
min_pct_range  = np.linspace(0.1, 0.5, num=number_of_swap_points, endpoint=True)
max_lag_range  = np.linspace(180, 360, num=number_of_swap_points, endpoint=True).astype(int)

print(f"min_percentage_data_in_trajectory values: {min_pct_range}")
print(f"max_lag values: {max_lag_range}")
print(f"Total combinations per dataset: {len(min_pct_range) * len(max_lag_range)}")


# ============================================================
# HEAVISIDE (LARSON 2011) ACF MODEL
# ============================================================
def heaviside_acf_model(tau, A, T, C):
    """G(τ) = A · (1 − τ/T) · H(T − τ)  +  C"""
    tau = np.asarray(tau, dtype=float)
    return np.where(tau <= T, A * (1.0 - tau / T) + C, C)


def fit_heaviside(lags, mean_corr, start_lag_idx=1):
    """Fit the Heaviside ACF model. Returns dict with A, T, C (or None)."""
    lags = np.asarray(lags, dtype=float)
    mc   = np.asarray(mean_corr, dtype=float)

    sl = max(start_lag_idx, 1)
    T_vals = lags[sl:]
    G_vals = mc[sl:]

    good = np.isfinite(G_vals) & np.isfinite(T_vals)
    T_vals = T_vals[good]
    G_vals = G_vals[good]
    if len(T_vals) < 4:
        return None

    tail = max(1, len(G_vals) // 5)
    C0 = float(np.mean(G_vals[-tail:]))
    A0 = max(float(G_vals[0]) - C0, 1e-8)
    crossings = np.where(G_vals <= C0)[0]
    T0 = float(T_vals[crossings[0]]) if len(crossings) > 0 else float(T_vals[-1] / 2)
    T0 = max(T0, 10.0)

    try:
        popt, _ = curve_fit(
            heaviside_acf_model, T_vals, G_vals,
            p0=[A0, T0, C0],
            bounds=([0, 1, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        return {'A': float(popt[0]), 'T': float(popt[1]), 'C': float(popt[2])}
    except Exception:
        return None


def compute_kinetics_heaviside(hfit, mean_corr, gene_len,
                               ribosomal_footprint=10):
    """Derive kinetics from Heaviside fit.
    Initiation rate: c = 1/(G(0)·T) where G(0) = A (fitted amplitude).
    """
    T_dwell = hfit['T']
    ke = gene_len / T_dwell
    A  = hfit['A']                 # fitted amplitude = G(0) [Larson Eq.1]
    ki = 1.0 / (A * T_dwell)

    rho = (ki * ribosomal_footprint / ke) * 100
    n_rib = (ki * gene_len) / ke
    rib_dist = gene_len / n_rib if n_rib > 0 else np.nan

    return {
        'tau_c': round(T_dwell / 2, 4),   # equivalent τ_c for comparison
        'dwell_time': round(T_dwell, 2),
        'ke': round(ke, 4),
        'ki': round(ki, 4),
        'ribosomal_density': round(rho, 3),
        'n_ribosomes': round(n_rib, 3),
        'ribosomal_distance': round(rib_dist, 3),
    }


# ============================================================
# HELPER: Run correlation + fit for a single max_lag value
# ============================================================
def run_correlation_for_lag(primary_data, ml, cell_ids, n_cells):
    """Run mi.Correlation and fit (exponential or heaviside) for one max_lag.

    Returns dict with ke, ki, tau_c, dwell_time, n_traces, n_cells or None.
    """
    try:
        with joblib.parallel_backend('threading', n_jobs=1):
            corr_obj = mi.Correlation(
                primary_data=primary_data,
                max_lag=int(ml),
                nan_handling='ignore',
                shift_data=False,
                return_full=False,
                time_interval_between_frames_in_seconds=step_size_in_sec,
                use_bootstrap=True,
                show_plot=False,
                start_lag=start_lag,
                fit_type=fit_type,
                de_correlation_threshold=de_correlation_threshold,
                correct_baseline=correct_baseline,
                use_linear_projection_for_lag_0=use_linear_projection_for_lag_0,
                save_plots=False,
                use_global_mean=use_global_mean,
                remove_outliers=remove_outliers,
                MAD_THRESHOLD_FACTOR=MAD_THRESHOLD_FACTOR,
                plot_individual_trajectories=False,
                y_axes_min_max_list_values=y_axes_min_max_list_values,
                x_axes_min_max_list_values=x_axes_min_max_list_values,
                multi_tau=multi_tau,
                multi_tau_raw_points=multi_tau_raw_points,
                multi_tau_bins_per_stage=multi_tau_bins_per_stage,
                plot_title='',
                index_max_lag_for_fit=None,
                line_color='blue',
                line_color_fit='dimgray',
                plot_name=None,
                figsize=(3.2, 2.2),
                detrend_photobleaching=DETREND_PHOTOBLEACHING,
            )
            mean_correlation, std_correlation, lags, correlations_array, dwell_time = corr_obj.run()

        # Post-filter cell count
        n_traces = correlations_array.shape[0]
        if cell_ids is not None and hasattr(corr_obj, 'keep_mask_'):
            surviving = cell_ids[corr_obj.keep_mask_]
            n_cells_final = int(np.unique(surviving).size)
        else:
            n_cells_final = n_cells

        _mc = np.asarray(mean_correlation, dtype=float)
        _lg = np.asarray(lags, dtype=float)

        # ── MODEL SELECTION ───────────────────────────────────────────────
        if FIT_MODEL == 'heaviside':
            # Heaviside fit
            hfit = fit_heaviside(_lg, _mc, start_lag_idx=start_lag)
            if hfit is None:
                return None
            kin = compute_kinetics_heaviside(
                hfit, _mc, gene_len=gen_length_half_HA)
            return {
                'ke': kin['ke'],
                'ki': kin['ki'],
                'tau_c': kin['tau_c'],
                'dwell_time': kin['dwell_time'],
                'ribosomal_density': kin['ribosomal_density'],
                'n_ribosomes': kin['n_ribosomes'],
                'ribosomal_distance': kin['ribosomal_distance'],
                'n_traces': n_traces,
                'n_cells': n_cells_final,
            }

        else:  # 'exponential'
            fit_params_ = getattr(corr_obj, 'fit_params_', None)
            if fit_params_ is None and fit_type == 'exponential':
                _si = int(max(start_lag, 0))
                _G = _mc[_si:]
                _T = _lg[_si:]
                _G = np.nan_to_num(_G)
                if len(_G) >= 3:
                    _tail = max(1, len(_G) // 10)
                    _C0 = float(np.mean(_G[-_tail:]))
                    _A0 = max(float(_G[0]) - _C0, 1e-6)
                    _tv = _C0 + _A0 / np.e
                    _it = int(np.argmin(np.abs(_G - _tv)))
                    _tc0 = float(_T[_it]) if _it > 0 else float(
                        _T[-1] / 2 if len(_T) > 1 else 1.0)
                    _tc0 = max(_tc0, 1e-6)
                    try:
                        _p, _ = curve_fit(
                            lambda t, A, tc, C: A * np.exp(-t / tc) + C,
                            _T, _G, p0=[_A0, _tc0, _C0], maxfev=100000,
                            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                        )
                        fit_params_ = {
                            'A': float(_p[0]),
                            'tau_c': float(_p[1]),
                            'C': float(_p[2]),
                        }
                        dwell_time = 2.0 * float(_p[1])
                    except Exception:
                        pass

            if fit_params_ is None:
                return None

            result = {
                'fit_params_': fit_params_,
                'mean_correlation': mean_correlation,
            }
            kin = compute_kinetics_from_acf(result, gene_length=gen_length_half_HA)
            return {
                'ke': kin['ke'],
                'ki': kin['ki'],
                'tau_c': kin['tau_c'],
                'dwell_time': kin['dwell_time'],
                'ribosomal_density': kin['ribosomal_density'],
                'n_ribosomes': kin['n_ribosomes'],
                'ribosomal_distance': kin['ribosomal_distance'],
                'n_traces': n_traces,
                'n_cells': n_cells_final,
            }

    except Exception as e:
        print(f'      Correlation failed for max_lag={ml}: {e}')
        return None


# ============================================================
# MAIN SWEEP
# ============================================================
all_sweep_rows = []

for data_folder, name, color in zip(list_datasets, list_names, list_colors):
    print(f'\n{"="*70}')
    print(f'  DATASET: {name}  |  model={FIT_MODEL}  MAD={MAD_THRESHOLD_FACTOR}')
    print(f'{"="*70}')

    # Resolve folder info once per dataset
    folder_with_files, plot_name_data, dataframe_prefix = dataset_selection(
        'cof', data_folder, control_spots_mode, downsample
    )

    ke_grid = np.full((len(min_pct_range), len(max_lag_range)), np.nan)
    ki_grid = np.full((len(min_pct_range), len(max_lag_range)), np.nan)
    n_traces_grid = np.full((len(min_pct_range), len(max_lag_range)), np.nan)

    for i, min_pct in enumerate(min_pct_range):
        # ── Load data ONCE per min_pct ────────────────────────────────────
        print(f'\n  Loading data with min_pct={min_pct:.3f} ...')
        try:
            primary_data, n_cells, n_spots, cell_ids = load_tracking_data(
                folder_with_files,
                selected_field=selected_field + str(channel_index),
                min_percentage_data_in_trajectory=min_pct,
                dataframe_prefix=dataframe_prefix,
                smooth_window=smooth_window,
                min_snr=min_snr,
                max_missing_frames=max_missing_frames,
                verbose=False,
                max_trajectory_length_percentile=max_trajectory_length_percentile,
            )
        except Exception as e:
            print(f'    Data loading FAILED: {e}')
            for j, ml in enumerate(max_lag_range):
                all_sweep_rows.append({
                    'dataset': name, 'min_pct': min_pct, 'max_lag': int(ml),
                    'ke': np.nan, 'ki': np.nan, 'tau_c': np.nan,
                    'dwell_time': np.nan, 'ribosomal_density': np.nan,
                    'n_ribosomes': np.nan, 'ribosomal_distance': np.nan,
                    'n_traces': 0, 'n_cells': 0,
                })
            continue

        print(f'    Loaded {primary_data.shape[0]} trajectories from {n_cells} cells')

        # ── Sweep max_lag in PARALLEL ─────────────────────────────────────
        results_for_row = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_correlation_for_lag)(primary_data, ml, cell_ids, n_cells)
            for ml in max_lag_range
        )

        # Collect results
        for j, (ml, res) in enumerate(zip(max_lag_range, results_for_row)):
            if res is not None:
                ke_grid[i, j] = res['ke']
                ki_grid[i, j] = res['ki']
                n_traces_grid[i, j] = res['n_traces']
                all_sweep_rows.append({
                    'dataset': name, 'min_pct': min_pct, 'max_lag': int(ml),
                    **res,
                })
            else:
                all_sweep_rows.append({
                    'dataset': name, 'min_pct': min_pct, 'max_lag': int(ml),
                    'ke': np.nan, 'ki': np.nan, 'tau_c': np.nan,
                    'dwell_time': np.nan, 'ribosomal_density': np.nan,
                    'n_ribosomes': np.nan, 'ribosomal_distance': np.nan,
                    'n_traces': 0, 'n_cells': 0,
                })

        ke_vals = [r['ke'] for r in results_for_row if r is not None]
        if ke_vals:
            print(f'    ke range: {min(ke_vals):.2f} – {max(ke_vals):.2f} aa/s')

    # ── Save grids for later re-plotting ──────────────────────────────────
    np.savez(output_dir / f'grids_{name}.npz',
             ke_grid=ke_grid, ki_grid=ki_grid, n_traces_grid=n_traces_grid,
             min_pct_range=min_pct_range, max_lag_range=max_lag_range)
    print(f'  Grids saved: grids_{name}.npz')

    # ── Helper for shared axis/spine styling ───────────────────────────────
    def style_heatmap_ax(ax):
        n_pts = len(max_lag_range)
        step = max(1, n_pts // 5)
        ax.set_xticks(range(0, n_pts, step))
        ax.set_xticklabels([str(int(max_lag_range[i])) for i in range(0, n_pts, step)])
        n_pct = len(min_pct_range)
        step_y = max(1, n_pct // 5)
        ax.set_yticks(range(0, n_pct, step_y))
        ax.set_yticklabels([f'{min_pct_range[i]:.2f}' for i in range(0, n_pct, step_y)])
        ax.set_xlabel('max lag (frames)')
        ax.set_ylabel('min % data in trajectory')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)


    # ── Combined heatmap: ke, ki, n_traces (1 row × 3 cols) ─────────────
    def pct_bounds(grid, lo=5, hi=95):
        valid = grid[np.isfinite(grid)]
        if len(valid) == 0:
            return None, None
        return np.percentile(valid, lo), np.percentile(valid, hi)

    model_label = 'Heaviside' if FIT_MODEL == 'heaviside' else 'Exponential'

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{name}  —  {model_label} model, MAD={MAD_THRESHOLD_FACTOR}',
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: ke
    vmin_ke, vmax_ke = pct_bounds(ke_grid)
    im0 = axes[0].imshow(ke_grid, origin='lower', aspect='auto',
                         cmap='RdYlBu_r', interpolation='nearest',
                         vmin=vmin_ke, vmax=vmax_ke)
    style_heatmap_ax(axes[0])
    axes[0].set_title(f'{name} — $k_e$ (aa/s)', fontsize=14, fontweight='bold')
    cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.85, extend='both')
    cbar0.set_label('$k_e$ (aa/s)', fontsize=14)

    # Panel 2: ki
    vmin_ki, vmax_ki = pct_bounds(ki_grid)
    im1 = axes[1].imshow(ki_grid, origin='lower', aspect='auto',
                         cmap='RdYlBu_r', interpolation='nearest',
                         vmin=vmin_ki, vmax=vmax_ki)
    style_heatmap_ax(axes[1])
    axes[1].set_title(f'{name} — $k_i$ (1/s)', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.85, extend='both')
    cbar1.set_label('$k_i$ (1/s)', fontsize=14)

    # Panel 3: number of trajectories
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import ScalarFormatter
    flat = n_traces_grid[np.isfinite(n_traces_grid)]
    if len(flat) > 0:
        vlo = max(np.percentile(flat, 2), 1)
        vhi = np.percentile(flat, 98)
        tmin_real = int(np.min(flat))
        tmax_real = int(np.max(flat))
        if vlo >= vhi:
            vlo, vhi = tmin_real, tmax_real
    else:
        vlo, vhi = 1, 100
        tmin_real, tmax_real = 1, 100

    im2 = axes[2].imshow(n_traces_grid, origin='lower', aspect='auto',
                         cmap='RdYlBu_r',
                         norm=LogNorm(vmin=vlo, vmax=vhi),
                         interpolation='nearest')
    style_heatmap_ax(axes[2])
    axes[2].set_title(f'{name} — # trajectories', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.85, extend='both')
    cbar2.set_label('trajectories', fontsize=14)
    # Force plain integer labels (no scientific notation)
    nice_ticks = np.array([t for t in [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500]
                           if vlo <= t <= vhi])
    if len(nice_ticks) < 2:
        nice_ticks = np.linspace(vlo, vhi, 5).astype(int)
    cbar2.set_ticks(nice_ticks)
    cbar2.set_ticklabels([str(int(t)) for t in nice_ticks])
    cbar2.ax.minorticks_off()

    plt.tight_layout()
    save_base = output_dir / f'heatmap_combined_{name}'
    plt.savefig(save_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_base.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {save_base.with_suffix(".png").name}')

# ============================================================
# SAVE FULL SWEEP CSV
# ============================================================
csv_path = output_dir / 'sensitivity_sweep.csv'
df_sweep = pd.DataFrame(all_sweep_rows)
df_sweep.to_csv(csv_path, index=False)
print(f'\n{"="*70}')
print(f'  Sensitivity sweep saved → {csv_path}')
print(f'  Model: {FIT_MODEL}  |  MAD: {MAD_THRESHOLD_FACTOR}')
print(f'{"="*70}')
print(df_sweep.to_string(index=False))
print()
