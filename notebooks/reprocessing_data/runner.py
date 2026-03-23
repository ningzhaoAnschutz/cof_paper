"""Batch intensity extraction pipeline + PSF Amplitude vs Sigma comparison.

Usage
-----
Run directly:
    cd notebooks/reprocessing_data
    python runner.py

Or import from a notebook:
    import sys; sys.path.insert(0, 'notebooks/reprocessing_data')
    from runner import run_pipeline, plot_psf_amplitude_vs_sigma
"""

import gc
import json
import os
import re
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

import pandas as pd
import numpy as np
import joblib
from scipy.optimize import curve_fit
from scipy.stats import linregress

from microlive import microscopy as mi

from reprocess_intensities import (
    associate_lif_to_results,
    build_intensity_dataset,
    build_correlation_input_from_reprocessed_df,
)



# ── Configuration ─────────────────────────────────────────────────────────────

# --- Conditions ---
# Each entry defines one experimental condition to process and compare.
# The 'name' becomes the legend label; 'color' is used in the PSF plot.
CONDITIONS = [
    {
        'name':        'sfGFP',
        'color':       '#4CAF50',
        'lif_dir':     Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/sfGFP'),
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/sfGFP/results'),
    },
     {
         'name':        'GFPuv',
         'color':       '#2196F3',
         'lif_dir':     Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv'),
         'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv/results'),
     },
    {
        'name':        'sfGFP_ex',
        'color':       '#FF9800',
        'lif_dir':     Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS038'),
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS038/results'),
    },
    {
        'name':        'sfGFP_sx',
        'color':       '#E91E63',
        'lif_dir':     Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS048'),
        'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/pRS048/results'),
    },
]

CONDA_ENV_NAME = 'microlive'

# --- Photobleaching ---
# Whether to apply whole-image photobleaching correction before intensity extraction.
# Set to False to extract intensities from the RAW images (no global correction).
APPLY_PHOTOBLEACHING  = False
# TIME_INTERVAL_SECONDS is extracted automatically from the LIF file metadata.

# Per-trajectory exponential detrend applied INSIDE mi.Correlation.run().
# This corrects each trajectory individually (I(t)/fit → rescale) rather than
# correcting all pixels globally.  Only takes effect when RUN_ACF_ANALYSIS = True.
DETREND_PHOTOBLEACHING = True

# --- Intensity extraction ---
# Disk diameter in pixels for the disk-doughnut photometry window.
# The background annulus extends 3 px beyond this disk.  Typical values: 3, 5, 7, 9, 11.
SPOT_SIZE_PX      = 5
# If True, PSF amplitude/sigma are estimated via fast moment-based (centre-of-mass) method.
# If False, a full scipy.curve_fit 2-D Gaussian is used (slower, more accurate for dim spots).
# Note: this parameter does NOT affect the disk-doughnut intensity value itself.
FAST_GAUSSIAN_FIT = True
# SNR method: 'peak'  → (max_disk − mean_bg) / std_bg  [standard, default]
#             'disk_doughnut' → (mean_disk − mean_bg) / std_bg  [robust for noisy/dim spots]
SNR_METHOD        = 'peak'
# If True, collapse Z via max-projection before measuring intensity (mirrors 2-D tracking mode).
USE_MAX_PROJECTION = True

# Output directory for CSV and plot files.
# All results are saved here when running python runner.py directly.
OUTPUT_DIR = Path('/Users/nzlab-la/Desktop/cof_paper/notebooks/reprocessing_data/results')

# --- Autocorrelation analysis (reprocessed intensities) ---
RUN_ACF_ANALYSIS = True
ACF_CHANNELS = [1]  # set to [0], [1], or [0, 1]

# --- Performance / parallelism ---
# Parallelize independent conditions using joblib. Threading is the safest default
# here because batch_runner mutates module-level config globals before calling
# run_pipeline(), and threads see those values without pickling/copying.
PARALLELIZE_CONDITIONS = False      # disabled: per-LIF streaming relies on sequential memory
CONDITION_PARALLEL_N_JOBS = -2      # use all CPUs except one
CONDITION_PARALLEL_PREFER = 'threads'  # 'threads' or 'processes'

# Preprocessing (DataFrame -> trajectory array)
ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY = 0.25
ACF_MAX_MISSING_FRAMES = 2
ACF_MAX_COLUMNS = 360
ACF_MIN_SNR = 0.5
ACF_SMOOTH_WINDOW = 1

# Correlation parameters (aligned with the existing notebook defaults where practical)
ACF_START_LAG = 1
ACF_MAX_LAG = 200
ACF_DOWNSAMPLE = False
ACF_DOWNSAMPLING_FACTOR = 3
ACF_USE_GLOBAL_MEAN = False
ACF_CORRECT_BASELINE = True
ACF_USE_LINEAR_PROJECTION_FOR_LAG_0 = True
ACF_USE_BOOTSTRAP = True
ACF_BOOTSTRAP_ITERATIONS = 1000  # lower (e.g. 200) for faster exploratory runs
ACF_REMOVE_OUTLIERS = False
ACF_MAD_THRESHOLD_FACTOR = 6
ACF_MULTI_TAU = True
ACF_MULTI_TAU_RAW_POINTS = 60
ACF_MULTI_TAU_BINS_PER_STAGE = 16
ACF_FIT_TYPE = 'exponential'
ACF_DE_CORRELATION_THRESHOLD = 0.001
ACF_INDEX_MAX_LAG_FOR_FIT = None
ACF_BASELINE_METHOD = 'auto_plateau'
ACF_SHOW_PLOT = False         # keep batch runs non-interactive
ACF_SAVE_PLOTS = False
ACF_X_LIMS = None
ACF_Y_LIMS = None

# --- Gene lengths (codons) for kinetics derivation ---
GENE_LENGTH = 1826          # full gene in codons
GENE_LENGTH_HALF_HA = 1659  # half-HA tag construct (codons)

# ─────────────────────────────────────────────────────────────────────────────


def load_config(yaml_path: str = None) -> str:
    """Load configuration from a YAML file and overwrite module-level globals.

    Returns the auto-generated param_tag string (used for folder naming).
    If yaml_path is None, looks for config.yaml next to this script.
    """
    import yaml  # pyyaml

    global CONDITIONS, APPLY_PHOTOBLEACHING, DETREND_PHOTOBLEACHING
    global SPOT_SIZE_PX, FAST_GAUSSIAN_FIT, SNR_METHOD, USE_MAX_PROJECTION
    global OUTPUT_DIR, RUN_ACF_ANALYSIS, ACF_CHANNELS
    global PARALLELIZE_CONDITIONS, CONDITION_PARALLEL_N_JOBS, CONDITION_PARALLEL_PREFER
    global ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY, ACF_MAX_MISSING_FRAMES
    global ACF_MAX_COLUMNS, ACF_MIN_SNR, ACF_SMOOTH_WINDOW
    global ACF_START_LAG, ACF_MAX_LAG, ACF_DOWNSAMPLE, ACF_DOWNSAMPLING_FACTOR
    global ACF_USE_GLOBAL_MEAN, ACF_CORRECT_BASELINE
    global ACF_USE_LINEAR_PROJECTION_FOR_LAG_0, ACF_USE_BOOTSTRAP
    global ACF_BOOTSTRAP_ITERATIONS, ACF_REMOVE_OUTLIERS, ACF_MAD_THRESHOLD_FACTOR
    global ACF_MULTI_TAU, ACF_MULTI_TAU_RAW_POINTS, ACF_MULTI_TAU_BINS_PER_STAGE
    global ACF_FIT_TYPE, ACF_DE_CORRELATION_THRESHOLD, ACF_INDEX_MAX_LAG_FOR_FIT
    global ACF_BASELINE_METHOD, ACF_SHOW_PLOT, ACF_SAVE_PLOTS
    global ACF_X_LIMS, ACF_Y_LIMS, GENE_LENGTH, GENE_LENGTH_HALF_HA

    if yaml_path is None:
        yaml_path = Path(__file__).parent / 'config.yaml'
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"[load_config] No config file at {yaml_path}, using hard-coded defaults.")
        return _build_param_tag()

    print(f"[load_config] Reading {yaml_path.name}")
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # ── Conditions ────────────────────────────────────────────────────
    if 'conditions' in cfg:
        CONDITIONS = []
        for c in cfg['conditions']:
            CONDITIONS.append({
                'name':        c['name'],
                'color':       c.get('color', '#888888'),
                'lif_dir':     Path(c['lif_dir']),
                'results_dir': Path(c['results_dir']),
            })

    # ── Photobleaching ────────────────────────────────────────────────
    if 'apply_photobleaching' in cfg:
        APPLY_PHOTOBLEACHING = bool(cfg['apply_photobleaching'])
    if 'detrend_photobleaching' in cfg:
        DETREND_PHOTOBLEACHING = bool(cfg['detrend_photobleaching'])

    # ── Intensity extraction ──────────────────────────────────────────
    if 'spot_size_px' in cfg:
        SPOT_SIZE_PX = int(cfg['spot_size_px'])
    if 'fast_gaussian_fit' in cfg:
        FAST_GAUSSIAN_FIT = bool(cfg['fast_gaussian_fit'])
    if 'snr_method' in cfg:
        SNR_METHOD = str(cfg['snr_method'])
    if 'use_max_projection' in cfg:
        USE_MAX_PROJECTION = bool(cfg['use_max_projection'])

    # ── Output ────────────────────────────────────────────────────────
    if 'output_dir' in cfg:
        OUTPUT_DIR = Path(cfg['output_dir'])

    # ── ACF analysis ──────────────────────────────────────────────────
    if 'run_acf_analysis' in cfg:
        RUN_ACF_ANALYSIS = bool(cfg['run_acf_analysis'])
    if 'acf_channels' in cfg:
        ACF_CHANNELS = list(cfg['acf_channels'])

    # ── Performance ───────────────────────────────────────────────────
    if 'parallelize_conditions' in cfg:
        PARALLELIZE_CONDITIONS = bool(cfg['parallelize_conditions'])
    if 'condition_parallel_n_jobs' in cfg:
        CONDITION_PARALLEL_N_JOBS = int(cfg['condition_parallel_n_jobs'])
    if 'condition_parallel_prefer' in cfg:
        CONDITION_PARALLEL_PREFER = str(cfg['condition_parallel_prefer'])

    # ── ACF preprocessing ─────────────────────────────────────────────
    _map = {
        'acf_min_percentage_data_in_trajectory': ('ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY', float),
        'acf_max_missing_frames':  ('ACF_MAX_MISSING_FRAMES', int),
        'acf_max_columns':         ('ACF_MAX_COLUMNS', int),
        'acf_min_snr':             ('ACF_MIN_SNR', float),
        'acf_smooth_window':       ('ACF_SMOOTH_WINDOW', int),
        'acf_start_lag':           ('ACF_START_LAG', int),
        'acf_max_lag':             ('ACF_MAX_LAG', int),
        'acf_downsample':          ('ACF_DOWNSAMPLE', bool),
        'acf_downsampling_factor': ('ACF_DOWNSAMPLING_FACTOR', int),
        'acf_use_global_mean':     ('ACF_USE_GLOBAL_MEAN', bool),
        'acf_correct_baseline':    ('ACF_CORRECT_BASELINE', bool),
        'acf_use_linear_projection_for_lag_0': ('ACF_USE_LINEAR_PROJECTION_FOR_LAG_0', bool),
        'acf_use_bootstrap':       ('ACF_USE_BOOTSTRAP', bool),
        'acf_bootstrap_iterations': ('ACF_BOOTSTRAP_ITERATIONS', int),
        'acf_remove_outliers':     ('ACF_REMOVE_OUTLIERS', bool),
        'acf_mad_threshold_factor': ('ACF_MAD_THRESHOLD_FACTOR', int),
        'acf_multi_tau':           ('ACF_MULTI_TAU', bool),
        'acf_multi_tau_raw_points': ('ACF_MULTI_TAU_RAW_POINTS', int),
        'acf_multi_tau_bins_per_stage': ('ACF_MULTI_TAU_BINS_PER_STAGE', int),
        'acf_fit_type':            ('ACF_FIT_TYPE', str),
        'acf_de_correlation_threshold': ('ACF_DE_CORRELATION_THRESHOLD', float),
        'acf_baseline_method':     ('ACF_BASELINE_METHOD', str),
        'acf_show_plot':           ('ACF_SHOW_PLOT', bool),
        'acf_save_plots':          ('ACF_SAVE_PLOTS', bool),
    }
    this_module = sys.modules[__name__]
    for yaml_key, (global_name, type_fn) in _map.items():
        if yaml_key in cfg:
            setattr(this_module, global_name, type_fn(cfg[yaml_key]))

    # Nullable keys (can be null / None)
    if 'acf_index_max_lag_for_fit' in cfg:
        v = cfg['acf_index_max_lag_for_fit']
        ACF_INDEX_MAX_LAG_FOR_FIT = int(v) if v is not None else None
    if 'acf_x_lims' in cfg:
        ACF_X_LIMS = cfg['acf_x_lims']  # list or None
    if 'acf_y_lims' in cfg:
        ACF_Y_LIMS = cfg['acf_y_lims']  # list or None

    # ── Gene lengths ──────────────────────────────────────────────────
    if 'gene_length' in cfg:
        GENE_LENGTH = int(cfg['gene_length'])
    if 'gene_length_half_ha' in cfg:
        GENE_LENGTH_HALF_HA = int(cfg['gene_length_half_ha'])

    tag = _build_param_tag()
    _print_settings_banner(tag)
    return tag


def _build_param_tag() -> str:
    """Build a descriptive folder-name tag from the current module globals."""
    _fit_tag = 'fast' if FAST_GAUSSIAN_FIT else 'full'
    _snr_tag = SNR_METHOD.replace('_', '')
    _pb_tag  = 'noPB' if not APPLY_PHOTOBLEACHING else 'PB'
    _det_tag = 'detrend' if DETREND_PHOTOBLEACHING else 'noDetrend'
    _mad_tag = f'mad{ACF_MAD_THRESHOLD_FACTOR}'
    _out_tag = f'out{ACF_REMOVE_OUTLIERS}'
    return f'sz{SPOT_SIZE_PX}_{_fit_tag}_{_snr_tag}_{_pb_tag}_{_det_tag}_{_mad_tag}_{_out_tag}'


def _print_settings_banner(param_tag: str) -> None:
    """Print a human-readable summary of the active settings."""
    cond_names = [c['name'] for c in CONDITIONS]
    print(f"\n{'='*65}")
    print(f"  ★  param_tag:               {param_tag}")
    print(f"  ★  Conditions:              {cond_names}")
    print(f"  ★  Photobleaching corr:     {APPLY_PHOTOBLEACHING}")
    print(f"  ★  Detrend photobleaching:  {DETREND_PHOTOBLEACHING}")
    print(f"  ★  Spot size (px):          {SPOT_SIZE_PX}")
    print(f"  ★  SNR method:              {SNR_METHOD}")
    print(f"  ★  ACF min_snr:             {ACF_MIN_SNR}")
    print(f"  ★  ACF max_lag:             {ACF_MAX_LAG}")
    print(f"  ★  ACF MAD threshold:       {ACF_MAD_THRESHOLD_FACTOR}")
    print(f"  ★  ACF remove_outliers:     {ACF_REMOVE_OUTLIERS}")
    print(f"  ★  ACF bootstrap iters:     {ACF_BOOTSTRAP_ITERATIONS}")
    print(f"  ★  Gene length (half-HA):   {GENE_LENGTH_HALF_HA}")
    print(f"  ★  Output dir:              {OUTPUT_DIR}")
    print(f"{'='*65}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Filesystem-safe slug used for per-condition output files."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(text)).strip('_') or 'condition'


def _infer_param_tag(output_csv: Path = None) -> str:
    if output_csv is None:
        return 'manual_run'
    stem = Path(output_csv).stem
    prefix = 'master_intensity_dataset_'
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def get_acf_config_dict() -> dict:
    """Return the current runner-level ACF configuration (for params.json/logging)."""
    return {
        'run_acf_analysis': RUN_ACF_ANALYSIS,
        'apply_photobleaching': APPLY_PHOTOBLEACHING,
        'detrend_photobleaching': DETREND_PHOTOBLEACHING,
        'channels': list(ACF_CHANNELS),
        'parallelize_conditions': PARALLELIZE_CONDITIONS,
        'condition_parallel_n_jobs': CONDITION_PARALLEL_N_JOBS,
        'condition_parallel_prefer': CONDITION_PARALLEL_PREFER,
        'min_percentage_data_in_trajectory': ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
        'max_missing_frames': ACF_MAX_MISSING_FRAMES,
        'max_columns': ACF_MAX_COLUMNS,
        'min_snr': ACF_MIN_SNR,
        'smooth_window': ACF_SMOOTH_WINDOW,
        'start_lag': ACF_START_LAG,
        'max_lag': ACF_MAX_LAG,
        'downsample': ACF_DOWNSAMPLE,
        'downsampling_factor': ACF_DOWNSAMPLING_FACTOR,
        'use_global_mean': ACF_USE_GLOBAL_MEAN,
        'correct_baseline': ACF_CORRECT_BASELINE,
        'use_linear_projection_for_lag_0': ACF_USE_LINEAR_PROJECTION_FOR_LAG_0,
        'use_bootstrap': ACF_USE_BOOTSTRAP,
        'bootstrap_iterations': ACF_BOOTSTRAP_ITERATIONS,
        'remove_outliers': ACF_REMOVE_OUTLIERS,
        'mad_threshold_factor': ACF_MAD_THRESHOLD_FACTOR,
        'multi_tau': ACF_MULTI_TAU,
        'multi_tau_raw_points': ACF_MULTI_TAU_RAW_POINTS,
        'multi_tau_bins_per_stage': ACF_MULTI_TAU_BINS_PER_STAGE,
        'fit_type': ACF_FIT_TYPE,
        'de_correlation_threshold': ACF_DE_CORRELATION_THRESHOLD,
        'index_max_lag_for_fit': ACF_INDEX_MAX_LAG_FOR_FIT,
        'baseline_method': ACF_BASELINE_METHOD,
        'show_plot': ACF_SHOW_PLOT,
        'save_plots': ACF_SAVE_PLOTS,
        'x_lims': ACF_X_LIMS,
        'y_lims': ACF_Y_LIMS,
    }


def _estimate_dwell_time_without_plot(
    mean_correlation: np.ndarray,
    lags: np.ndarray,
    *,
    start_lag: int,
    fit_type: str,
    de_correlation_threshold: float,
    index_max_lag_for_fit: int = None,
):
    """Non-plot fallback dwell-time fit, mirroring the plotting code logic."""
    fit_params = None
    dwell_time = None

    if mean_correlation is None or lags is None:
        return dwell_time, fit_params
    mean_correlation = np.asarray(mean_correlation, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if mean_correlation.size == 0 or lags.size == 0:
        return dwell_time, fit_params

    start_idx = int(max(start_lag, 0))
    if start_idx >= len(mean_correlation):
        return dwell_time, fit_params

    if fit_type == 'linear':
        normalized_correlation = mean_correlation
        if index_max_lag_for_fit is None:
            max_idx = len(normalized_correlation)
        else:
            max_idx = int(index_max_lag_for_fit)
        try:
            autocorrelations = normalized_correlation[start_idx:]
            selected_lags = lags[start_idx + 1:start_idx + max_idx]
            selected_corr = autocorrelations[1:max_idx]
            if len(selected_lags) >= 2 and len(selected_lags) == len(selected_corr):
                slope, intercept, _, _, _ = linregress(selected_lags, selected_corr)
                if slope != 0:
                    dwell_time = float(-intercept / slope)
        except Exception:
            pass
        return dwell_time, fit_params

    if fit_type != 'exponential':
        return dwell_time, fit_params

    if index_max_lag_for_fit is not None:
        G_tau = mean_correlation[start_idx:int(index_max_lag_for_fit)]
        taus = lags[start_idx:int(index_max_lag_for_fit)]
    else:
        G_tau = mean_correlation[start_idx:]
        taus = lags[start_idx:]

    if len(G_tau) < 3:
        return dwell_time, fit_params

    G_tau = np.nan_to_num(np.asarray(G_tau, dtype=float))
    taus = np.asarray(taus, dtype=float)

    def single_exponential_decay(tau, A, tau_c, C):
        return A * np.exp(-tau / tau_c) + C

    tail_length = max(1, len(G_tau) // 10)
    C_guess = float(np.mean(G_tau[-tail_length:]))
    G0 = float(G_tau[0])
    A_guess = G0 - C_guess
    if A_guess == 0:
        A_guess = float((np.max(G_tau) - np.min(G_tau)) / 2.0)
    A_guess = max(float(A_guess), 1e-6)

    target_value = C_guess + A_guess / np.e
    idx_tau_c = int(np.argmin(np.abs(G_tau - target_value)))
    if idx_tau_c == 0:
        tau_c_guess = float((taus[-1] / 2) if len(taus) > 1 else 1.0)
    else:
        tau_c_guess = float(taus[idx_tau_c])
    tau_c_guess = max(tau_c_guess, 1e-6)

    try:
        params, _ = curve_fit(
            single_exponential_decay,
            taus,
            G_tau,
            p0=[A_guess, tau_c_guess, C_guess],
            maxfev=100000,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        )
    except Exception:
        return dwell_time, fit_params

    A_fitted, tau_c_fitted, C_fitted = [float(v) for v in params]
    dwell_time = 2.0 * tau_c_fitted          # AUC equivalence: T_dwell = 2 * τ_c
    fit_params = {
        'A': A_fitted,
        'tau_c': tau_c_fitted,
        'C': C_fitted,
        'taus': taus,
    }
    return dwell_time, fit_params


# ── Heaviside (Larson 2011) ACF Model ─────────────────────────────────────────

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
    return np.where(tau <= T, A * (1.0 - tau / T) + C, C)


def fit_heaviside_acf(lags, mean_corr, start_lag_idx=1):
    """Fit the Heaviside ACF model to data.

    Returns
    -------
    params : dict  with keys 'A', 'T', 'C', 'A_err', 'T_err', 'C_err'
             (or None on failure)
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


def _run_acf_for_condition(
    cond_df: pd.DataFrame,
    *,
    condition_name: str,
    condition_color: str = None,
    run_dir: Path = None,
    intensity_csv_path: Path = None,
    param_tag: str = None,
) -> tuple[list[dict], list[dict]]:
    """Run ACF for one condition dataframe on the configured channel(s)."""
    if cond_df.empty:
        return [], []

    acf_results = []
    summary_rows = []
    acf_dir = (Path(run_dir) / 'acf') if run_dir is not None else None
    if acf_dir is not None:
        acf_dir.mkdir(parents=True, exist_ok=True)

    time_interval_values = sorted(
        pd.to_numeric(cond_df.get('time_interval_s', pd.Series(dtype=float)), errors='coerce')
        .dropna()
        .unique()
        .tolist()
    )
    if len(time_interval_values) > 1:
        raise ValueError(
            f"Condition '{condition_name}' has multiple time intervals: {time_interval_values}. "
            "Split by time interval before ACF."
        )
    default_step_s = float(time_interval_values[0]) if len(time_interval_values) == 1 else 1.0

    for ch in ACF_CHANNELS:
        curve_csv_path = None
        plot_path = None
        try:
            prepared = build_correlation_input_from_reprocessed_df(
                cond_df,
                channel_index=int(ch),
                condition_name=condition_name,
                min_percentage_data_in_trajectory=ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                max_missing_frames=ACF_MAX_MISSING_FRAMES,
                maximum_columns=ACF_MAX_COLUMNS,
                min_snr=ACF_MIN_SNR,
                smooth_window=ACF_SMOOTH_WINDOW,
                verbose=False,
            )

            primary_data = prepared['primary_data']
            step_s = prepared['time_interval_s'] if prepared['time_interval_s'] is not None else default_step_s
            if step_s is None:
                step_s = 1.0

            if ACF_DOWNSAMPLE:
                primary_data = mi.Utilities().downsample_array(
                    primary_data,
                    factor=int(ACF_DOWNSAMPLING_FACTOR),
                    method='average',
                )
                step_s = float(step_s) * int(ACF_DOWNSAMPLING_FACTOR)

            safe_cond = _slugify(condition_name)
            if acf_dir is not None:
                curve_csv_path = acf_dir / f'acf_{safe_cond}_ch{ch}.csv'
                if ACF_SAVE_PLOTS:
                    plot_path = acf_dir / f'acf_{safe_cond}_ch{ch}.png'

            with joblib.parallel_backend('threading', n_jobs=1):
                corr_obj = mi.Correlation(
                    primary_data=primary_data,
                    max_lag=ACF_MAX_LAG,
                    nan_handling='ignore',
                    shift_data=False,
                    return_full=False,
                    time_interval_between_frames_in_seconds=float(step_s),
                    use_bootstrap=ACF_USE_BOOTSTRAP,
                    show_plot=ACF_SHOW_PLOT,
                    start_lag=ACF_START_LAG,
                    fit_type=ACF_FIT_TYPE,
                    de_correlation_threshold=ACF_DE_CORRELATION_THRESHOLD,
                    correct_baseline=ACF_CORRECT_BASELINE,
                    use_linear_projection_for_lag_0=ACF_USE_LINEAR_PROJECTION_FOR_LAG_0,
                    save_plots=ACF_SAVE_PLOTS,
                    use_global_mean=ACF_USE_GLOBAL_MEAN,
                    remove_outliers=ACF_REMOVE_OUTLIERS,
                    MAD_THRESHOLD_FACTOR=ACF_MAD_THRESHOLD_FACTOR,
                    plot_individual_trajectories=False,
                    y_axes_min_max_list_values=ACF_Y_LIMS,
                    x_axes_min_max_list_values=ACF_X_LIMS,
                    multi_tau=ACF_MULTI_TAU,
                    multi_tau_raw_points=ACF_MULTI_TAU_RAW_POINTS,
                    multi_tau_bins_per_stage=ACF_MULTI_TAU_BINS_PER_STAGE,
                    plot_title=f'{condition_name} ch{ch}',
                    index_max_lag_for_fit=ACF_INDEX_MAX_LAG_FOR_FIT,
                    baseline_method=ACF_BASELINE_METHOD,
                    line_color=(condition_color if condition_color is not None else 'blue'),
                    line_color_fit='dimgray',
                    plot_name=(str(plot_path) if plot_path is not None else None),
                    figsize=(3.2, 2.2),
                    detrend_photobleaching=DETREND_PHOTOBLEACHING,
                )
                if hasattr(corr_obj, 'BOOTSTRAP_ITERATIONS'):
                    corr_obj.BOOTSTRAP_ITERATIONS = int(ACF_BOOTSTRAP_ITERATIONS)
                mean_corr, std_corr, lags, correlations_array, dwell_time = corr_obj.run()
            fit_params = getattr(corr_obj, 'fit_params_', None)

            if dwell_time is None:
                dwell_time_est, fit_params_est = _estimate_dwell_time_without_plot(
                    mean_correlation=np.asarray(mean_corr),
                    lags=np.asarray(lags),
                    start_lag=ACF_START_LAG,
                    fit_type=ACF_FIT_TYPE,
                    de_correlation_threshold=ACF_DE_CORRELATION_THRESHOLD,
                    index_max_lag_for_fit=ACF_INDEX_MAX_LAG_FOR_FIT,
                )
                dwell_time = dwell_time_est
                if fit_params is None:
                    fit_params = fit_params_est

            curve_df = pd.DataFrame({
                'lags': np.asarray(lags),
                'mean_correlation': np.asarray(mean_corr),
                'std_correlation': np.asarray(std_corr),
            })
            if curve_csv_path is not None:
                curve_df.to_csv(curve_csv_path, index=False)

            number_of_trajectories_final = int(correlations_array.shape[0])
            if hasattr(corr_obj, 'keep_mask_') and len(prepared['image_ids']) == len(corr_obj.keep_mask_):
                surviving_image_ids = np.asarray(prepared['image_ids'])[corr_obj.keep_mask_]
                number_of_cells_final = int(np.unique(surviving_image_ids).size)
            else:
                number_of_cells_final = int(prepared['total_number_of_cells'])

            fit_A = fit_tau_c = fit_C = np.nan
            if isinstance(fit_params, dict):
                fit_A = float(fit_params.get('A', np.nan))
                fit_tau_c = float(fit_params.get('tau_c', np.nan))
                fit_C = float(fit_params.get('C', np.nan))

            # ── Heaviside / Larson 2011 fit ────────────────────────────────
            lags_arr = np.asarray(lags)
            mc_arr = np.asarray(mean_corr)
            hfit = fit_heaviside_acf(lags_arr, mc_arr, start_lag_idx=ACF_START_LAG)

            hev_A = hev_T = hev_C = hev_A_err = hev_T_err = hev_C_err = np.nan
            hev_ke = hev_ki = hev_dwell_time = np.nan
            kin_hev = None
            if hfit is not None:
                hev_A = float(hfit['A'])
                hev_T = float(hfit['T'])
                hev_C = float(hfit['C'])
                hev_A_err = float(hfit['A_err'])
                hev_T_err = float(hfit['T_err'])
                hev_C_err = float(hfit['C_err'])
                kin_hev = compute_kinetics_heaviside(
                    hfit, mc_arr, gene_length=GENE_LENGTH_HALF_HA)
                hev_ke = float(kin_hev['ke'])
                hev_ki = float(kin_hev['ki'])
                hev_dwell_time = float(kin_hev['T_dwell'])

            result_dict = {
                'condition': condition_name,
                'condition_color': condition_color,
                'channel_index': int(ch),
                'mean_correlation': np.asarray(mean_corr),
                'std_correlation': np.asarray(std_corr),
                'lags': np.asarray(lags),
                'correlations_array': correlations_array,
                'dwell_time': dwell_time,
                'total_number_of_cells': int(prepared['total_number_of_cells']),
                'total_number_of_spots': int(prepared['total_number_of_spots']),
                'number_of_trajectories_final': number_of_trajectories_final,
                'number_of_cells_final': number_of_cells_final,
                'fit_params_': fit_params,
                'hfit': hfit,
                'kin_hev': kin_hev,
                'plot_name': f'{condition_name}_ch{ch}',
                'df': curve_df,
                'dataset': condition_name,
                'data_source': 'reprocessed',
                'curve_csv_path': (str(curve_csv_path) if curve_csv_path is not None else None),
                'intensity_csv_path': (str(intensity_csv_path) if intensity_csv_path is not None else None),
                'time_interval_s': float(step_s),
                'adapter_stats': prepared.get('adapter_stats', {}),
            }
            acf_results.append(result_dict)

            summary_rows.append({
                'param_tag': param_tag,
                'condition': condition_name,
                'channel_index': int(ch),
                'channel_label': f'ch{ch}',
                'condition_color': condition_color,
                'time_interval_s': float(step_s),
                'selected_field': prepared['selected_field'],
                'snr_field': prepared['snr_field'],
                'min_snr': ACF_MIN_SNR,
                'min_percentage_data_in_trajectory': ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                'max_missing_frames': ACF_MAX_MISSING_FRAMES,
                'smooth_window': ACF_SMOOTH_WINDOW,
                'maximum_columns': ACF_MAX_COLUMNS,
                'multi_tau': ACF_MULTI_TAU,
                'multi_tau_raw_points': ACF_MULTI_TAU_RAW_POINTS,
                'multi_tau_bins_per_stage': ACF_MULTI_TAU_BINS_PER_STAGE,
                'max_lag': ACF_MAX_LAG,
                'fit_type': ACF_FIT_TYPE,
                'de_correlation_threshold': ACF_DE_CORRELATION_THRESHOLD,
                'correct_baseline': ACF_CORRECT_BASELINE,
                'use_global_mean': ACF_USE_GLOBAL_MEAN,
                'use_bootstrap': ACF_USE_BOOTSTRAP,
                'bootstrap_iterations': (int(ACF_BOOTSTRAP_ITERATIONS) if ACF_USE_BOOTSTRAP else 0),
                'remove_outliers': ACF_REMOVE_OUTLIERS,
                'MAD_THRESHOLD_FACTOR': ACF_MAD_THRESHOLD_FACTOR,
                'total_number_of_cells': int(prepared['total_number_of_cells']),
                'total_number_of_spots': int(prepared['total_number_of_spots']),
                'number_of_cells_final': number_of_cells_final,
                'number_of_trajectories_final': number_of_trajectories_final,
                'dwell_time': (float(dwell_time) if dwell_time is not None else np.nan),
                'fit_A': fit_A,
                'fit_tau_c': fit_tau_c,
                'fit_C': fit_C,
                'hev_A': hev_A,
                'hev_T': hev_T,
                'hev_C': hev_C,
                'hev_A_err': hev_A_err,
                'hev_T_err': hev_T_err,
                'hev_C_err': hev_C_err,
                'hev_ke': hev_ke,
                'hev_ki': hev_ki,
                'hev_dwell_time': hev_dwell_time,
                'gene_length_half_HA': GENE_LENGTH_HALF_HA,
                'curve_csv_path': (str(curve_csv_path) if curve_csv_path is not None else ''),
                'intensity_csv_path': (str(intensity_csv_path) if intensity_csv_path is not None else ''),
                'apply_photobleaching': APPLY_PHOTOBLEACHING,
                'detrend_photobleaching': DETREND_PHOTOBLEACHING,
                'status': 'ok',
                'error_message': '',
            })
            # ── Console summary with both models ──────────────────────────
            exp_info = f", exp_dwell={dwell_time:.2f}s" if dwell_time is not None else ""
            hev_info = (
                f", hev_T={hev_T:.1f}s, hev_ke={hev_ke:.2f}aa/s"
                if hfit is not None else ", hev=FAILED"
            )
            print(
                f"  [ACF] {condition_name} ch{ch}: "
                f"{number_of_cells_final} cells/images, {number_of_trajectories_final} traces"
                + exp_info + hev_info
            )

        except Exception as exc:
            msg = str(exc)
            print(f"  [ACF] ERROR for {condition_name} ch{ch}: {msg}")
            summary_rows.append({
                'param_tag': param_tag,
                'condition': condition_name,
                'channel_index': int(ch),
                'channel_label': f'ch{ch}',
                'condition_color': condition_color,
                'time_interval_s': np.nan,
                'selected_field': f'spot_int_ch_{ch}',
                'snr_field': f'snr_ch_{ch}',
                'min_snr': ACF_MIN_SNR,
                'min_percentage_data_in_trajectory': ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                'max_missing_frames': ACF_MAX_MISSING_FRAMES,
                'smooth_window': ACF_SMOOTH_WINDOW,
                'maximum_columns': ACF_MAX_COLUMNS,
                'multi_tau': ACF_MULTI_TAU,
                'multi_tau_raw_points': ACF_MULTI_TAU_RAW_POINTS,
                'multi_tau_bins_per_stage': ACF_MULTI_TAU_BINS_PER_STAGE,
                'max_lag': ACF_MAX_LAG,
                'fit_type': ACF_FIT_TYPE,
                'de_correlation_threshold': ACF_DE_CORRELATION_THRESHOLD,
                'correct_baseline': ACF_CORRECT_BASELINE,
                'use_global_mean': ACF_USE_GLOBAL_MEAN,
                'use_bootstrap': ACF_USE_BOOTSTRAP,
                'bootstrap_iterations': (int(ACF_BOOTSTRAP_ITERATIONS) if ACF_USE_BOOTSTRAP else 0),
                'remove_outliers': ACF_REMOVE_OUTLIERS,
                'MAD_THRESHOLD_FACTOR': ACF_MAD_THRESHOLD_FACTOR,
                'total_number_of_cells': np.nan,
                'total_number_of_spots': np.nan,
                'number_of_cells_final': np.nan,
                'number_of_trajectories_final': np.nan,
                'dwell_time': np.nan,
                'fit_A': np.nan,
                'fit_tau_c': np.nan,
                'fit_C': np.nan,
                'hev_A': np.nan,
                'hev_T': np.nan,
                'hev_C': np.nan,
                'hev_A_err': np.nan,
                'hev_T_err': np.nan,
                'hev_C_err': np.nan,
                'hev_ke': np.nan,
                'hev_ki': np.nan,
                'hev_dwell_time': np.nan,
                'gene_length_half_HA': GENE_LENGTH_HALF_HA,
                'curve_csv_path': (str(curve_csv_path) if curve_csv_path is not None else ''),
                'intensity_csv_path': (str(intensity_csv_path) if intensity_csv_path is not None else ''),
                'apply_photobleaching': APPLY_PHOTOBLEACHING,
                'detrend_photobleaching': DETREND_PHOTOBLEACHING,
                'status': 'error',
                'error_message': msg,
            })

    return acf_results, summary_rows


def _process_single_condition(
    cond: dict,
    *,
    run_dir: Path = None,
    per_condition_dir: Path = None,
    param_tag: str = None,
) -> dict:
    """Process one condition end-to-end (intensity reprocessing + optional ACF)."""
    name = cond['name']
    color = cond.get('color')
    lif_dir = cond['lif_dir']
    results_dir = cond['results_dir']

    print(f"\n{'='*60}")
    print(f" Processing condition: {name}")
    print(f"{'='*60}")

    try:
        lif_dir     = Path(lif_dir)
        results_dir = Path(results_dir)

        # Discover all LIF files WITHOUT loading them yet.
        all_lif_files = sorted(
            f for f in lif_dir.iterdir()
            if f.is_file() and f.suffix.lower() == '.lif'
        )
        if not all_lif_files:
            print(f"  [{name}] no LIF files found in {lif_dir}")
            cond_df = pd.DataFrame()
        else:
            # Process ONE LIF file at a time to cap peak memory usage.
            partial_dfs     = []
            next_dir_id     = 0
            for lif_file in all_lif_files:
                print(f"\n  [{name}] loading LIF: {lif_file.name}")
                association = associate_lif_to_results(
                    lif_dir=lif_dir,
                    results_dir=results_dir,
                    lif_files_filter=[lif_file],
                )
                if not any(association.values()):
                    del association
                    gc.collect()
                    continue
                partial_df = build_intensity_dataset(
                    association_dict=association,
                    apply_photobleaching=APPLY_PHOTOBLEACHING,
                    spot_size=SPOT_SIZE_PX,
                    fast_gaussian_fit=FAST_GAUSSIAN_FIT,
                    snr_method=SNR_METHOD,
                    use_max_projection=USE_MAX_PROJECTION,
                    start_result_dir_id=next_dir_id,
                )
                del association
                gc.collect()
                if not partial_df.empty:
                    next_dir_id = int(partial_df['result_dir_id'].max()) + 1
                    partial_dfs.append(partial_df)
                del partial_df
                gc.collect()

            cond_df = pd.concat(partial_dfs, ignore_index=True) if partial_dfs else pd.DataFrame()
            del partial_dfs
            gc.collect()

    except Exception as exc:
        print(f"  [{name}] condition processing failed: {exc}")
        acf_summary_rows = []
        if RUN_ACF_ANALYSIS:
            for ch in ACF_CHANNELS:
                acf_summary_rows.append({
                    'param_tag': param_tag,
                    'condition': name,
                    'channel_index': int(ch),
                    'channel_label': f'ch{ch}',
                    'condition_color': color,
                    'time_interval_s': np.nan,
                    'selected_field': f'spot_int_ch_{ch}',
                    'snr_field': f'snr_ch_{ch}',
                    'min_snr': ACF_MIN_SNR,
                    'min_percentage_data_in_trajectory': ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                    'max_missing_frames': ACF_MAX_MISSING_FRAMES,
                    'smooth_window': ACF_SMOOTH_WINDOW,
                    'maximum_columns': ACF_MAX_COLUMNS,
                    'multi_tau': ACF_MULTI_TAU,
                    'multi_tau_raw_points': ACF_MULTI_TAU_RAW_POINTS,
                    'multi_tau_bins_per_stage': ACF_MULTI_TAU_BINS_PER_STAGE,
                    'max_lag': ACF_MAX_LAG,
                    'fit_type': ACF_FIT_TYPE,
                    'de_correlation_threshold': ACF_DE_CORRELATION_THRESHOLD,
                    'correct_baseline': ACF_CORRECT_BASELINE,
                    'use_global_mean': ACF_USE_GLOBAL_MEAN,
                    'use_bootstrap': ACF_USE_BOOTSTRAP,
                    'bootstrap_iterations': (int(ACF_BOOTSTRAP_ITERATIONS) if ACF_USE_BOOTSTRAP else 0),
                    'remove_outliers': ACF_REMOVE_OUTLIERS,
                    'MAD_THRESHOLD_FACTOR': ACF_MAD_THRESHOLD_FACTOR,
                    'total_number_of_cells': np.nan,
                    'total_number_of_spots': np.nan,
                    'number_of_cells_final': np.nan,
                    'number_of_trajectories_final': np.nan,
                    'dwell_time': np.nan,
                    'fit_A': np.nan,
                    'fit_tau_c': np.nan,
                    'fit_C': np.nan,
                    'hev_A': np.nan,
                    'hev_T': np.nan,
                    'hev_C': np.nan,
                    'hev_A_err': np.nan,
                    'hev_T_err': np.nan,
                    'hev_C_err': np.nan,
                    'hev_ke': np.nan,
                    'hev_ki': np.nan,
                    'hev_dwell_time': np.nan,
                    'gene_length_half_HA': GENE_LENGTH_HALF_HA,
                    'curve_csv_path': '',
                    'intensity_csv_path': '',
                    'apply_photobleaching': APPLY_PHOTOBLEACHING,
                    'detrend_photobleaching': DETREND_PHOTOBLEACHING,
                    'status': 'error',
                    'error_message': str(exc),
                })
        return {
            'condition': name,
            'color': color,
            'cond_df': None,
            'acf_results': [],
            'acf_summary_rows': acf_summary_rows,
            'condition_csv_path': None,
        }

    if cond_df.empty:
        print(f"  [{name}] no data — skipping.")
        del cond_df
        gc.collect()
        return {
            'condition': name,
            'color': color,
            'cond_df': None,
            'acf_results': [],
            'acf_summary_rows': [],
            'condition_csv_path': None,
        }

    cond_df.insert(0, 'condition', name)

    condition_csv_path = None
    if per_condition_dir is not None:
        condition_csv_path = per_condition_dir / f'intensity_{_slugify(name)}.csv'
        cond_df.to_csv(condition_csv_path, index=False)
        print(f"  Saved condition intensity CSV → {condition_csv_path.name}  ({len(cond_df)} rows)")

    acf_results = []
    acf_summary_rows = []
    if RUN_ACF_ANALYSIS:
        try:
            acf_results, acf_summary_rows = _run_acf_for_condition(
                cond_df,
                condition_name=name,
                condition_color=color,
                run_dir=run_dir,
                intensity_csv_path=condition_csv_path,
                param_tag=param_tag,
            )
        except Exception as exc:
            print(f"  [ACF] condition-level failure for {name}: {exc}")
            for ch in ACF_CHANNELS:
                acf_summary_rows.append({
                    'param_tag': param_tag,
                    'condition': name,
                    'channel_index': int(ch),
                    'channel_label': f'ch{ch}',
                    'condition_color': color,
                    'time_interval_s': np.nan,
                    'selected_field': f'spot_int_ch_{ch}',
                    'snr_field': f'snr_ch_{ch}',
                    'min_snr': ACF_MIN_SNR,
                    'min_percentage_data_in_trajectory': ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                    'max_missing_frames': ACF_MAX_MISSING_FRAMES,
                    'smooth_window': ACF_SMOOTH_WINDOW,
                    'maximum_columns': ACF_MAX_COLUMNS,
                    'multi_tau': ACF_MULTI_TAU,
                    'multi_tau_raw_points': ACF_MULTI_TAU_RAW_POINTS,
                    'multi_tau_bins_per_stage': ACF_MULTI_TAU_BINS_PER_STAGE,
                    'max_lag': ACF_MAX_LAG,
                    'fit_type': ACF_FIT_TYPE,
                    'de_correlation_threshold': ACF_DE_CORRELATION_THRESHOLD,
                    'correct_baseline': ACF_CORRECT_BASELINE,
                    'use_global_mean': ACF_USE_GLOBAL_MEAN,
                    'use_bootstrap': ACF_USE_BOOTSTRAP,
                    'bootstrap_iterations': (int(ACF_BOOTSTRAP_ITERATIONS) if ACF_USE_BOOTSTRAP else 0),
                    'remove_outliers': ACF_REMOVE_OUTLIERS,
                    'MAD_THRESHOLD_FACTOR': ACF_MAD_THRESHOLD_FACTOR,
                    'total_number_of_cells': np.nan,
                    'total_number_of_spots': np.nan,
                    'number_of_cells_final': np.nan,
                    'number_of_trajectories_final': np.nan,
                    'dwell_time': np.nan,
                    'fit_A': np.nan,
                    'fit_tau_c': np.nan,
                    'fit_C': np.nan,
                    'hev_A': np.nan,
                    'hev_T': np.nan,
                    'hev_C': np.nan,
                    'hev_A_err': np.nan,
                    'hev_T_err': np.nan,
                    'hev_C_err': np.nan,
                    'hev_ke': np.nan,
                    'hev_ki': np.nan,
                    'hev_dwell_time': np.nan,
                    'gene_length_half_HA': GENE_LENGTH_HALF_HA,
                    'curve_csv_path': '',
                    'intensity_csv_path': (str(condition_csv_path) if condition_csv_path is not None else ''),
                    'apply_photobleaching': APPLY_PHOTOBLEACHING,
                    'detrend_photobleaching': DETREND_PHOTOBLEACHING,
                    'status': 'error',
                    'error_message': str(exc),
                })

    # (image arrays were already freed inside the per-LIF loop)
    gc.collect()

    return {
        'condition': name,
        'color': color,
        'cond_df': cond_df,
        'acf_results': acf_results,
        'acf_summary_rows': acf_summary_rows,
        'condition_csv_path': (str(condition_csv_path) if condition_csv_path is not None else None),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    conditions: list = None,
    output_csv: Path = None,
) -> pd.DataFrame:
    """Process all conditions and return a single combined master DataFrame.

    Iterates over *conditions* (defaults to the module-level CONDITIONS list).
    For each condition, calls associate_lif_to_results + build_intensity_dataset
    and stamps a 'condition' column so rows can be grouped downstream.
    When ``RUN_ACF_ANALYSIS`` is True and ``output_csv`` is provided, it also:
      - writes one intensity CSV per condition
      - runs ACF per condition/channel
      - writes one ACF curve CSV per (condition, channel)
      - writes one ACF summary CSV across all conditions/channels for the run

    Args:
        conditions: List of condition dicts (name, color, lif_dir, results_dir).
            Defaults to the module-level CONDITIONS list.
        output_csv: If provided, save the combined master_df here as CSV.

    Returns:
        master_df — one row per (global_particle_id, frame), with a 'condition'
        column identifying which experimental group each row belongs to.
    """
    if conditions is None:
        conditions = CONDITIONS

    output_csv = Path(output_csv) if output_csv is not None else None
    run_dir = output_csv.parent if output_csv is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    per_condition_dir = (run_dir / 'conditions') if run_dir is not None else None
    if per_condition_dir is not None:
        per_condition_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    all_acf_results: list[dict] = []
    all_acf_summary_rows: list[dict] = []
    param_tag = _infer_param_tag(output_csv)

    if PARALLELIZE_CONDITIONS and len(conditions) > 1:
        print(
            f"\n[runner] Parallelizing conditions with joblib "
            f"(n_jobs={CONDITION_PARALLEL_N_JOBS}, prefer='{CONDITION_PARALLEL_PREFER}')"
        )
        processed_conditions = joblib.Parallel(
            n_jobs=CONDITION_PARALLEL_N_JOBS,
            prefer=CONDITION_PARALLEL_PREFER,
        )(
            joblib.delayed(_process_single_condition)(
                cond,
                run_dir=run_dir,
                per_condition_dir=per_condition_dir,
                param_tag=param_tag,
            )
            for cond in conditions
        )
    else:
        processed_conditions = [
            _process_single_condition(
                cond,
                run_dir=run_dir,
                per_condition_dir=per_condition_dir,
                param_tag=param_tag,
            )
            for cond in conditions
        ]

    for result in processed_conditions:
        all_acf_results.extend(result.get('acf_results', []))
        all_acf_summary_rows.extend(result.get('acf_summary_rows', []))
        cond_df = result.get('cond_df')
        if cond_df is not None and not cond_df.empty:
            all_frames.append(cond_df)

    if not all_frames:
        print("No data collected across all conditions.")
        return pd.DataFrame()

    master_df = pd.concat(all_frames, ignore_index=True)

    acf_summary_df = pd.DataFrame(all_acf_summary_rows) if all_acf_summary_rows else pd.DataFrame()
    acf_summary_path = None
    if RUN_ACF_ANALYSIS and run_dir is not None and not acf_summary_df.empty:
        acf_summary_path = run_dir / 'acf' / 'acf_summary_all_conditions.csv'
        acf_summary_path.parent.mkdir(parents=True, exist_ok=True)
        acf_summary_df.to_csv(acf_summary_path, index=False)
        print(f"\nSaved ACF summary → {acf_summary_path}  ({len(acf_summary_df)} rows)")

    if output_csv is not None:
        master_df.to_csv(output_csv, index=False)
        print(f"\nSaved → {output_csv}  ({len(master_df)} rows)")

    # global_particle_id can collide across conditions because result_dir_id resets per condition.
    composite_particle_ids = (
        master_df['condition'].astype(str) + '::' + master_df['global_particle_id'].astype(str)
    )
    print(
        f"\n{'='*60}\n"
        f" Combined master_df: {len(master_df)} rows, "
        f"{composite_particle_ids.nunique()} condition-scoped unique particles, "
        f"{master_df['condition'].nunique()} conditions\n"
        f"{'='*60}\n"
    )

    # Attach run metadata so batch_runner/notebooks can inspect outputs without changing the return type.
    master_df.attrs['param_tag'] = param_tag
    master_df.attrs['run_dir'] = (str(run_dir) if run_dir is not None else None)
    master_df.attrs['per_condition_dir'] = (str(per_condition_dir) if per_condition_dir is not None else None)
    master_df.attrs['acf_summary_path'] = (str(acf_summary_path) if acf_summary_path is not None else None)
    master_df.attrs['acf_summary_df'] = acf_summary_df
    master_df.attrs['acf_results'] = all_acf_results
    master_df.attrs['acf_config'] = get_acf_config_dict()
    return master_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reprocessing intensity pipeline')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML config file (default: config.yaml next to this script)')
    args = parser.parse_args()

    # Load YAML config → overwrites module globals, returns param_tag.
    param_tag = load_config(args.config)

    # Each run gets its own subfolder under OUTPUT_DIR.
    run_dir    = OUTPUT_DIR / param_tag
    output_csv = run_dir / f'master_intensity_dataset_{param_tag}.csv'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the effective config as JSON for reproducibility.
    params_record = {
        'param_tag': param_tag,
        'conditions': [c['name'] for c in CONDITIONS],
        'spot_size_px': SPOT_SIZE_PX,
        'fast_gaussian_fit': FAST_GAUSSIAN_FIT,
        'snr_method': SNR_METHOD,
        'apply_photobleaching': APPLY_PHOTOBLEACHING,
        'detrend_photobleaching': DETREND_PHOTOBLEACHING,
        'acf_config': get_acf_config_dict(),
    }
    with open(run_dir / 'params.json', 'w') as f:
        json.dump(params_record, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"CSV           : {output_csv.name}")
    print(f"params.json   : saved")

    master_df = run_pipeline(output_csv=output_csv)

    # ── Generate all plots ────────────────────────────────────────────────
    from plot_psf import (
        plot_acf_comparison,
        plot_acf_exponential_fit,
        plot_acf_heaviside_fit,
        plot_acf_individual_fits,
        plot_individual_trajectories,
        plot_individual_trajectories_detrended,
        plot_psf_amplitude_vs_sigma,
        plot_intensity_distributions,
        load_acf_results_from_disk,
        CONDITION_RENAME,
        MIN_SNR,
    )

    # Apply any condition renames so colors resolve correctly.
    if CONDITION_RENAME:
        master_df['condition'] = master_df['condition'].replace(CONDITION_RENAME)

    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating plots → {plots_dir}")
    print(f"{'='*60}\n")

    # PSF amplitude vs sigma + distributions per channel.
    for ch in ACF_CHANNELS:
        plot_psf_amplitude_vs_sigma(
            master_df, channel_index=ch, output_dir=plots_dir,
            save_name=f'psf_amplitude_vs_sigma_ch{ch}_{param_tag}',
        )
        plot_intensity_distributions(
            master_df, field='snr_ch_', channel_index=ch,
            min_snr=MIN_SNR, x_label='SNR', output_dir=plots_dir,
            save_name=f'snr_ch{ch}_{param_tag}',
        )
        plot_intensity_distributions(
            master_df, field='spot_int_ch_', channel_index=ch,
            min_snr=MIN_SNR, x_label='Spot Intensity (a.u.)', output_dir=plots_dir,
            save_name=f'spot_int_ch{ch}_{param_tag}',
        )

    # ACF overlay plot (one per channel, all conditions on one figure).
    acf_results = master_df.attrs.get('acf_results', [])

    # Fallback: if in-memory results are empty, load from saved CSVs.
    if not acf_results:
        acf_dir = run_dir / 'acf'
        if acf_dir.exists():
            acf_results = load_acf_results_from_disk(acf_dir)

    plot_acf_comparison(acf_results, param_tag=param_tag, output_dir=plots_dir)

    # Per-condition ACF plots with fit overlays (acf_individual.py style).
    plot_acf_individual_fits(acf_results, param_tag=param_tag, output_dir=plots_dir)

    # Per-condition ACF plots with individual trace overlays (if correlations_array is in memory).
    for r in acf_results:
        cond  = r['condition']
        ch    = r['channel_index']
        safe  = cond.replace(' ', '_')

        if r.get('correlations_array') is not None:
            plot_acf_exponential_fit(
                r,
                output_path=plots_dir / f'acf_{safe}_ch{ch}_exponential_traces_{param_tag}',
                show_individual=True,
            )
            if r.get('hfit') is not None:
                plot_acf_heaviside_fit(
                    r,
                    output_path=plots_dir / f'acf_{safe}_ch{ch}_heaviside_traces_{param_tag}',
                    show_individual=True,
                )

    # Per-condition individual intensity trajectories.
    conditions = master_df['condition'].unique()
    for cond in conditions:
        for ch in ACF_CHANNELS:
            plot_individual_trajectories(
                master_df,
                condition=cond,
                channel_index=ch,
                min_snr=MIN_SNR,
                output_dir=plots_dir,
                save_name=f'trajectories_{cond}_ch{ch}_{param_tag}',
            )

    # Per-condition detrended intensity trajectories
    # (uses same array construction + detrending as the ACF engine).
    if DETREND_PHOTOBLEACHING:
        for cond in conditions:
            for ch in ACF_CHANNELS:
                plot_individual_trajectories_detrended(
                    master_df,
                    condition=cond,
                    channel_index=ch,
                    min_snr=ACF_MIN_SNR,
                    output_dir=plots_dir,
                    save_name=f'trajectories_detrended_{cond}_ch{ch}_{param_tag}',
                    min_percentage_data=ACF_MIN_PERCENTAGE_DATA_IN_TRAJECTORY,
                    max_missing_frames=ACF_MAX_MISSING_FRAMES,
                    maximum_columns=ACF_MAX_COLUMNS,
                )

    print(f"\n{'='*60}")
    print(f"  All plots saved → {plots_dir}")
    print(f"{'='*60}\n")

    del master_df
    gc.collect()
