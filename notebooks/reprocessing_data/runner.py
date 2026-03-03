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
    # {
    #     'name':        'GFPuv',
    #     'color':       '#2196F3',
    #     'lif_dir':     Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv'),
    #     'results_dir': Path('/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF AC/GFPuv/results'),
    # },
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
# Whether to apply photobleaching correction before intensity extraction.
APPLY_PHOTOBLEACHING  = True
# TIME_INTERVAL_SECONDS is extracted automatically from the LIF file metadata.

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
ACF_REMOVE_OUTLIERS = True
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

# ─────────────────────────────────────────────────────────────────────────────


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
    G_fitted = single_exponential_decay(taus, *params)
    below_threshold = np.where(G_fitted < float(de_correlation_threshold))[0]
    if len(below_threshold) > 0:
        dwell_time = float(taus[int(below_threshold[0])])
    fit_params = {
        'A': A_fitted,
        'tau_c': tau_c_fitted,
        'C': C_fitted,
        'taus': taus,
    }
    return dwell_time, fit_params


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
                'curve_csv_path': (str(curve_csv_path) if curve_csv_path is not None else ''),
                'intensity_csv_path': (str(intensity_csv_path) if intensity_csv_path is not None else ''),
                'status': 'ok',
                'error_message': '',
            })
            print(
                f"  [ACF] {condition_name} ch{ch}: "
                f"{number_of_cells_final} cells/images, {number_of_trajectories_final} traces"
                + (f", dwell={dwell_time:.2f}s" if dwell_time is not None else "")
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
                'curve_csv_path': (str(curve_csv_path) if curve_csv_path is not None else ''),
                'intensity_csv_path': (str(intensity_csv_path) if intensity_csv_path is not None else ''),
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
                    'curve_csv_path': '',
                    'intensity_csv_path': '',
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
                    'curve_csv_path': '',
                    'intensity_csv_path': (str(condition_csv_path) if condition_csv_path is not None else ''),
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
    # Short identifier encoding the three key extraction parameters.
    # Example: sz5_fast_peak  |  sz7_full_disk_doughnut
    _fit_tag = 'fast' if FAST_GAUSSIAN_FIT else 'full'
    _snr_tag = SNR_METHOD.replace('_', '')   # 'peak' or 'diskdoughnut'
    param_tag = f'sz{SPOT_SIZE_PX}_{_fit_tag}_{_snr_tag}'

    # Each run gets its own subfolder under OUTPUT_DIR.
    run_dir    = OUTPUT_DIR / param_tag
    output_csv = run_dir / f'master_intensity_dataset_{param_tag}.csv'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parameter tag : {param_tag}")
    print(f"Run directory : {run_dir}")
    print(f"CSV           : {output_csv.name}")

    master_df = run_pipeline(output_csv=output_csv)
    gc.collect()
