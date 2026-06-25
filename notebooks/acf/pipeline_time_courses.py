# !/usr/bin/env python3
"""
Pipeline for time course analysis of cotranslational folding data.

This module provides functions for:
- Dataset selection and configuration
- TASEP simulation for generating synthetic dual-channel data
- Intensity extraction and processing from tracking dataframes
- Dual-channel data alignment and filtering
- Cross-correlation and autocorrelation analysis
- Kymograph generation for visualizing time-resolved data

Usage Example
-------------
>>> from pipeline_time_courses import (
...     compute_autocorrelation_for_dataset,
...     compute_cross_correlation_for_dataset,
...     analyze_crosscorr,
...     analyze_dual_channel_time_courses,
...     run_simulation_and_generate_data,
...     plot_dual_channel_kymograph,
... )
>>> 
>>> # Analyze experimental data
>>> results = analyze_dual_channel_time_courses(
...     data_folder=Path('/path/to/data'),
...     selected_field='spot_int_ch_',
...     min_snr=1.0,
... )

Inputs
------
- Tracking CSV files from microlive processing pipelines
- Gene sequence files (.dna format) for TASEP simulations

Outputs
-------
- Processed intensity arrays
- Correlation analysis results (autocorrelation, cross-correlation)
- Kymograph visualizations
"""
from pathlib import Path
from unicodedata import normalize
current_dir = Path().resolve()
from microlive.imports import *
from microlive import microscopy as mi
import tasep_models as tm
from tasep_models import *
import joblib
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.signal import peak_widths
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde



def dataset_selection(plot_name, data_folder, control_spots_mode=False, downsample=False):
    """
    Select dataset configuration based on parameters.
    
    Parameters
    ----------
    plot_name : str
        Base name for plots
    data_folder : Path
        Folder containing data
    control_spots_mode : bool
        Whether analyzing control spots
    downsample : bool
        Whether data is downsampled
    
    Returns
    -------
    folder_with_files : Path
        Folder containing tracking files
    plot_name : str
        Modified plot name with suffixes
    dataframe_prefix : str
        Prefix for dataframe files
    """
    if control_spots_mode:
        dataframe_prefix = 'random_location_spots_'
    else:
        dataframe_prefix = 'tracking_'
    
    folder_with_files = data_folder
    
    if control_spots_mode:
        plot_name = plot_name + '_control_spots'
    if downsample:
        plot_name = plot_name + '_downsampled'
    
    return folder_with_files, plot_name, dataframe_prefix


def run_simulation_and_generate_data(gene_sequence_path, ki, ke_global=5, step_size_in_sec=5, 
                                     folding_delay=0, t_max=360*5):
    """
    Run TASEP simulation and generate synthetic data for two tags.
    
    Parameters
    ----------
    gene_sequence_path : Path
        Path to gene sequence file
    ki : float
        Initiation rate
    ke_global : float
        Global elongation rate
    step_size_in_sec : float
        Time step in seconds
    folding_delay : float
        Folding delay in seconds
    t_max : float
        Maximum simulation time in seconds
    
    Returns
    -------
    SSA_HA : ndarray
        Simulated HA tag signal (n_trajectories, n_timepoints)
    SSA_GFP : ndarray
        Simulated GFP tag signal (n_trajectories, n_timepoints)
    """
    HA_TAG = 'YPYDVPDYA'
    GFP_TAG = 'LEFVTAA'  
    TAG_list = [HA_TAG, GFP_TAG]
    
    # Load sequence and parameters for simulation
    data_sequence = read_gene_sequence(gene_sequence_path, TAG_list=TAG_list)    
    # Run the simulation to generate synthetic data
    _, _, SSA_HA, SSA_GFP = simulate_TASEP_SSA(
        ki, 
        ke=calculate_codon_elongation_rates(data_sequence['rna'], global_elongation_rate=ke_global),
        gene_length=data_sequence['gene_length'],
        t_max=t_max,
        time_interval_in_seconds=step_size_in_sec,
        number_repetitions=200, 
        first_probe_position_vector=data_sequence['probe_data']['tag_0']['position_cumulative_vector'],
        second_probe_position_vector=data_sequence['probe_data']['tag_1']['position_cumulative_vector'],
        burnin_time=2000,
        folding_delay=folding_delay,
        constant_elongation_rate=None,
        efficiency_list=[1, 1],
        fast_output=False
    )
    
    return SSA_HA, SSA_GFP




def extract_intensity_and_shift_data(dataframe, selected_field='spot_int_ch_0', 
                                     min_percentage_data_in_trajectory=0.4, 
                                     padd_with_nans=True, maximum_columns=360, 
                                     min_snr=1, 
                                     smooth_window=1,
                                     max_missing_frames=5,
                                     verbose=True):
    """
    Extract intensity data with proper SNR filtering applied before trajectory shifting.
    
    Parameters
    ----------
    dataframe : DataFrame
        Input dataframe containing tracking data
    selected_field : str
        Field name for intensity data
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data required
    padd_with_nans : bool
        Whether to pad arrays to maximum_columns
    maximum_columns : int
        Target number of columns after padding/truncation
    min_snr : float
        Minimum SNR threshold
    smooth_window : int
        Smoothing window size
    max_missing_frames : int
        Maximum consecutive NaN frames allowed
    verbose : bool
        Print progress messages
    
    Returns
    -------
    intensity_array_shifted : ndarray
        Processed intensity array (n_trajectories, maximum_columns)
    """
    # Extract raw arrays
    intensity_array = mi.Utilities().df_trajectories_to_array(
        dataframe=dataframe, selected_field=selected_field, fill_value='nans')
    
    # Validate non-empty array
    if intensity_array.shape[0] == 0:
        if verbose:
            print(f"ERROR: No trajectories found for field '{selected_field}'")
        return np.array([]).reshape(0, maximum_columns)
    
    # SNR filtering
    try:
        snr_array = mi.Utilities().df_trajectories_to_array(
            dataframe=dataframe, 
            selected_field='snr_ch_' + selected_field[-1], 
            fill_value='nans'
        )
        mean_snr_array = np.nanmean(snr_array, axis=1)
        snr_mask = mean_snr_array >= min_snr
        intensity_array_filtered = intensity_array[snr_mask, :]
        
        if verbose:
            print(f"Initial array shape: {intensity_array.shape}")
            print(f"SNR filtering: {intensity_array.shape[0]} -> {intensity_array_filtered.shape[0]} trajectories "
                  f"(removed {np.sum(~snr_mask)} with SNR < {min_snr})")
        
        # Handle case where ALL trajectories fail SNR filter
        if intensity_array_filtered.shape[0] == 0:
            if verbose:
                print(f"WARNING: All trajectories failed SNR filter (threshold: {min_snr})")
            return np.array([]).reshape(0, maximum_columns)
            
    except KeyError as e:
        if verbose:
            print(f"WARNING: SNR field 'snr_ch_{selected_field[-1]}' not found, skipping SNR filter")
        intensity_array_filtered = intensity_array
    except Exception as e:
        if verbose:
            print(f"ERROR in SNR filtering: {e}")
        return np.array([]).reshape(0, maximum_columns)
    
    # Shift trajectories
    try:
        intensity_array_shifted = mi.Utilities().shift_trajectories(
            intensity_array_filtered,
            min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
            max_missing_frames=max_missing_frames
        )
    except Exception as e:
        if verbose:
            print(f"ERROR in shift_trajectories: {e}")
        return np.array([]).reshape(0, maximum_columns)
    
    # Check for empty array BEFORE smoothing
    if intensity_array_shifted.shape[0] == 0:
        if verbose:
            print(f"WARNING: No trajectories survived quality filtering")
        return np.array([]).reshape(0, maximum_columns)


    # Smoothing and forward fill
    if smooth_window > 1:
        try:
            intensity_array_shifted = mi.Utilities().forward_fill_nan_2d(intensity_array_shifted)
            if verbose:
                print("Applying smoothing to the intensity data")
            intensity_array_shifted = uniform_filter1d(
                intensity_array_shifted, size=smooth_window, axis=1, mode='nearest'
            )
        except Exception as e:
            if verbose:
                print(f"WARNING: Smoothing failed ({e}), continuing without smoothing")
    
    if verbose:
        print(f"Quality filtering: {intensity_array_filtered.shape[0]} -> {intensity_array_shifted.shape[0]} trajectories")
    
    # Padding/truncation
    number_of_columns = intensity_array_shifted.shape[1]
    if padd_with_nans:        
        if number_of_columns < maximum_columns:
            intensity_array_shifted = np.pad(
                intensity_array_shifted, 
                ((0, 0), (0, maximum_columns - number_of_columns)), 
                mode='constant', constant_values=np.nan
            )
    if number_of_columns > maximum_columns:
        intensity_array_shifted = intensity_array_shifted[:, :maximum_columns]
    
    if verbose:
        print(f"Final array shape: {intensity_array_shifted.shape}")
    
    return intensity_array_shifted


def filter_long_trajectories(intensity_array, max_trajectory_length_percentile=None,
                             verbose=True):
    """Remove trajectories whose length exceeds a percentile threshold.

    "Length" = number of finite (non-NaN) frames per trajectory row,
    measured after SNR filtering, shift, and concatenation.

    Parameters
    ----------
    intensity_array : ndarray
        2-D array (n_trajectories, n_timepoints). NaN = missing frame.
    max_trajectory_length_percentile : float or None
        Percentile threshold (0-100). Trajectories with more finite frames
        than ``np.percentile(lengths, pct)`` are removed.
        None or >= 100 disables the filter.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    filtered_array : ndarray
        Array with long trajectories removed.
    keep_mask : ndarray of bool
        Boolean mask (True = kept) of length n_trajectories.
    """
    if (max_trajectory_length_percentile is None
            or max_trajectory_length_percentile >= 100
            or intensity_array.shape[0] == 0):
        return intensity_array, np.ones(intensity_array.shape[0], dtype=bool)

    lengths = np.sum(np.isfinite(intensity_array), axis=1)
    cutoff = np.percentile(lengths, max_trajectory_length_percentile)
    keep_mask = lengths <= cutoff

    if verbose:
        n_total = intensity_array.shape[0]
        n_removed = int(np.sum(~keep_mask))
        print(f"  Trajectory-length filter (≤ P{max_trajectory_length_percentile:.0f} = "
              f"{cutoff:.0f} frames): {n_total} → "
              f"{n_total - n_removed} trajectories (removed {n_removed})")
        if n_removed > 0:
            removed_lengths = lengths[~keep_mask]
            print(f"    Removed trajectory lengths: min={int(removed_lengths.min())}, "
                  f"max={int(removed_lengths.max())}, "
                  f"median={int(np.median(removed_lengths))}")

    return intensity_array[keep_mask], keep_mask


def plot_trajectory_coverage(primary_data, step_size_in_sec=5,
                             dataset_name='', color='steelblue',
                             save_path=None, figsize=(4.5, 2.5),
                             show_plot=True, ax=None):
    """Plot the number of trajectories with valid (finite) data at each frame.

    This diagnostic reveals how statistical power decreases at long lag times:
    the ACF at lag τ can only use trajectories that have data at both frame t
    and frame t+τ, so the effective sample size drops as τ grows.

    Parameters
    ----------
    primary_data : ndarray, shape (n_traj, n_frames)
        Intensity array (NaN = missing frame).
    step_size_in_sec : float
        Time interval between frames, in seconds.
    dataset_name : str
        Label for the plot title.
    color : str
        Fill color for the coverage area.
    save_path : str or Path or None
        If provided, save the figure (without extension — .png is appended).
    figsize : tuple
        Figure size in inches.
    show_plot : bool
        Whether to display the figure.
    ax : matplotlib Axes or None
        If provided, plot into this axes instead of creating a new figure.

    Returns
    -------
    fig : matplotlib Figure or None
        The figure object (None if an external ax was provided).
    coverage : ndarray, shape (n_frames,)
        Number of valid trajectories at each frame.
    """
    import matplotlib.pyplot as plt

    n_traj, n_frames = primary_data.shape
    # Count finite values per column (frame)
    coverage = np.sum(np.isfinite(primary_data), axis=0)
    time_axis = np.arange(n_frames) * step_size_in_sec

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    else:
        fig = None

    ax.fill_between(time_axis, coverage, alpha=0.35, color=color, linewidth=0)
    ax.plot(time_axis, coverage, color=color, linewidth=1.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('N trajectories with data')
    ax.set_title(f'{dataset_name} — trajectory coverage' if dataset_name else 'Trajectory coverage',
                 fontsize=9)
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(0, None)

    # Annotate key stats
    ax.axhline(n_traj, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.text(time_axis[-1] * 0.02, n_traj * 0.95, f'total = {n_traj}',
            fontsize=7, color='gray', va='top')

    # Mark where coverage drops below 50% and 25%
    for frac, ls in [(0.5, ':'), (0.25, '-.')]:
        threshold = int(n_traj * frac)
        below = np.where(coverage < threshold)[0]
        if len(below) > 0:
            t_drop = time_axis[below[0]]
            ax.axvline(t_drop, color='crimson', linestyle=ls, linewidth=0.7, alpha=0.6)
            ax.text(t_drop + time_axis[-1] * 0.01, n_traj * (frac + 0.05),
                    f'<{int(frac*100)}% @ {t_drop:.0f}s', fontsize=6,
                    color='crimson', va='bottom')

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
            print(f'  Saved: {Path(save_path).name}.png')
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return fig, coverage


def extract_dual_channel_data(dataframe, primary_field, secondary_field, 
                              min_percentage_data_in_trajectory=0.3, shift_data=False,
                              max_missing_frames=5, min_snr=1, maximum_columns=360, 
                              use_forward_fill=False, smooth_window=1, verbose=True):
    """
    Extract synchronized dual-channel intensity data with proper filtering.
    NOW REQUIRES BOTH CHANNELS TO PASS SNR THRESHOLD.
    
    Parameters
    ----------
    dataframe : DataFrame
        Input dataframe containing tracking data
    primary_field : str
        Field name for primary channel
    secondary_field : str
        Field name for secondary channel
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data required
    shift_data : bool
        Whether to apply trajectory shifting
    max_missing_frames : int
        Maximum consecutive NaN frames allowed
    min_snr : float
        Minimum SNR threshold (applied to BOTH channels)
    maximum_columns : int
        Target number of columns after padding/truncation
    use_forward_fill : bool
        Whether to forward-fill NaN values
    smooth_window : int
        Smoothing window size
    verbose : bool
        Print progress messages
    
    Returns
    -------
    primary_shifted : ndarray or None
        Processed primary channel data
    secondary_shifted : ndarray or None
        Processed secondary channel data
    """
    # Input validation
    if dataframe is None or dataframe.empty:
        if verbose:
            print("ERROR: Empty or None dataframe provided")
        return None, None
    
    if primary_field not in dataframe.columns:
        if verbose:
            print(f"ERROR: Primary field '{primary_field}' not in dataframe")
            print(f"  Available fields: {[c for c in dataframe.columns if 'spot_int' in c]}")
        return None, None
    
    if secondary_field not in dataframe.columns:
        if verbose:
            print(f"ERROR: Secondary field '{secondary_field}' not in dataframe")
        return None, None
    
    # Extract raw arrays for both channels
    primary_raw = mi.Utilities().df_trajectories_to_array(
        dataframe=dataframe, selected_field=primary_field, fill_value='nans')
    secondary_raw = mi.Utilities().df_trajectories_to_array(
        dataframe=dataframe, selected_field=secondary_field, fill_value='nans')
    
    # Validate same number of trajectories
    if primary_raw.shape[0] != secondary_raw.shape[0]:
        print(f"    WARNING: Channel shape mismatch - Primary: {primary_raw.shape}, Secondary: {secondary_raw.shape}")
        return None, None
    
    # ===== SNR FILTERING (BOTH CHANNELS) - CRITICAL FIX =====
    # Extract channel indices from field names
    primary_ch_idx = int(primary_field[-1])  # e.g., 'spot_int_ch_1' -> 1
    secondary_ch_idx = int(secondary_field[-1])  # e.g., 'spot_int_ch_0' -> 0
    
    # Load SNR for BOTH channels
    try:
        snr_primary = mi.Utilities().df_trajectories_to_array(
            dataframe=dataframe, 
            selected_field=f'snr_ch_{primary_ch_idx}',
            fill_value='nans'
        )
        snr_secondary = mi.Utilities().df_trajectories_to_array(
            dataframe=dataframe, 
            selected_field=f'snr_ch_{secondary_ch_idx}',
            fill_value='nans'
        )
        
        # Calculate mean SNR for each trajectory in BOTH channels
        mean_snr_primary = np.nanmean(snr_primary, axis=1)
        mean_snr_secondary = np.nanmean(snr_secondary, axis=1)
        
        # REQUIRE BOTH CHANNELS TO PASS SNR THRESHOLD
        snr_mask = (mean_snr_primary >= min_snr) & (mean_snr_secondary >= min_snr)
        
        primary_filtered = primary_raw[snr_mask, :]
        secondary_filtered = secondary_raw[snr_mask, :]
        
        if verbose:
            n_removed_snr = np.sum(~snr_mask)
            print(f"    SNR filtering (BOTH channels >={min_snr}): {primary_raw.shape[0]} -> {primary_filtered.shape[0]} trajectories")
            if n_removed_snr > 0:
                print(f"      Removed {n_removed_snr} trajectories failing SNR in one or both channels")
    
    except KeyError as e:
        # SNR field doesn't exist - this is a real error, don't silently ignore
        if verbose:
            print(f"    ERROR: SNR field not found ({e})")
            print(f"      Available fields: {dataframe.columns.tolist()}")
        return None, None
    
    except Exception as e:
        # Other unexpected error - report it
        if verbose:
            print(f"    ERROR: SNR filtering failed ({e})")
        return None, None
    
    # ===== QUALITY FILTERING =====
    n_time = primary_filtered.shape[1]
    max_nans_allowed = int(round(n_time * (1 - min_percentage_data_in_trajectory)))
    primary_nan_counts = np.isnan(primary_filtered).sum(axis=1)
    secondary_nan_counts = np.isnan(secondary_filtered).sum(axis=1)
    quality_mask = (primary_nan_counts <= max_nans_allowed) & (secondary_nan_counts <= max_nans_allowed)
    primary_filtered = primary_filtered[quality_mask, :]
    secondary_filtered = secondary_filtered[quality_mask, :]
    
    if verbose:
        print(f"    Pre-quality filtering (BOTH channels): {primary_nan_counts.shape[0]} -> {primary_filtered.shape[0]} trajectories")
    
    if primary_filtered.shape[0] == 0:
        print(f"    ERROR: No trajectories passed dual-channel quality filter")
        return None, None
    
    # ===== SHIFTING =====
    try:
        if shift_data:
            primary_shifted, secondary_shifted = mi.Utilities().shift_trajectories(
                array_ch0=primary_filtered,
                array_ch1=secondary_filtered,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                max_missing_frames=max_missing_frames
            )
        else:
            primary_shifted = primary_filtered
            secondary_shifted = secondary_filtered
        
        if verbose:
            print(f"    Shift/alignment: {primary_filtered.shape[0]} -> {primary_shifted.shape[0]} trajectories")
    
    except Exception as e:
        print(f"    ERROR: Shifting failed ({e})")
        return None, None
    
    # ===== FORWARD FILL =====
    if use_forward_fill:
        primary_shifted = mi.Utilities().forward_fill_nan_2d(primary_shifted)
        secondary_shifted = mi.Utilities().forward_fill_nan_2d(secondary_shifted)
        if verbose:
            print(f"    Applied forward-fill to handle NaNs")
    
    # ===== VALIDATION =====
    if primary_shifted.shape != secondary_shifted.shape:
        print(f"    ERROR: Final shape mismatch - Primary: {primary_shifted.shape}, Secondary: {secondary_shifted.shape}")
        return None, None
    
    # ===== PADDING/TRUNCATION =====
    number_of_columns = primary_shifted.shape[1]
    if number_of_columns < maximum_columns:
        pad_width = maximum_columns - number_of_columns
        primary_shifted = np.pad(primary_shifted, ((0, 0), (0, pad_width)), 
                                mode='constant', constant_values=np.nan)
        secondary_shifted = np.pad(secondary_shifted, ((0, 0), (0, pad_width)), 
                                  mode='constant', constant_values=np.nan)
    elif number_of_columns > maximum_columns:
        primary_shifted = primary_shifted[:, :maximum_columns]
        secondary_shifted = secondary_shifted[:, :maximum_columns]
    
    # ===== POST-PADDING QUALITY CHECK =====
    primary_valid_counts = np.sum(np.isfinite(primary_shifted), axis=1)
    secondary_valid_counts = np.sum(np.isfinite(secondary_shifted), axis=1)
    min_valid_points = int(round(maximum_columns * min_percentage_data_in_trajectory))
    
    final_quality_mask = (primary_valid_counts >= min_valid_points) & (secondary_valid_counts >= min_valid_points)
    primary_shifted = primary_shifted[final_quality_mask, :]
    secondary_shifted = secondary_shifted[final_quality_mask, :]
    
    n_removed_post_padding = np.sum(~final_quality_mask)
    if n_removed_post_padding > 0:
        if verbose:
            print(f"    Post-padding quality check: removed {n_removed_post_padding} trajectories with <{min_valid_points} valid points")
    
    if primary_shifted.shape[0] == 0:
        if verbose:
            print(f"    ERROR: No trajectories passed final quality check")
        return None, None
    
    # ===== SMOOTHING (LAST STEP) =====
    if smooth_window > 1:
        primary_shifted = uniform_filter1d(primary_shifted, size=smooth_window, axis=1, mode='nearest')
        secondary_shifted = uniform_filter1d(secondary_shifted, size=smooth_window, axis=1, mode='nearest')
        if verbose:
            print(f"    Applied smoothing with window size: {smooth_window}")
    
    if verbose:
        print(f"    Final synchronized shapes: {primary_shifted.shape}")
    
    return primary_shifted, secondary_shifted


def _find_tracking_files(root_folder: Path, dataframe_prefix: str = 'tracking_') -> list:
    """Return a sorted list of tracking CSV Paths from all results_* subfolders.

    Shared helper used by load_tracking_data, load_dual_channel_tracking_data,
    and extract_intensity_distributions to avoid triplicating the same scan logic.
    """
    if not root_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_folder}")
    subfolders = [f for f in root_folder.iterdir()
                  if f.is_dir() and 'results_' in f.name]
    if not subfolders:
        raise ValueError(f"No results_* subfolders found in {root_folder}")
    return sorted([
        fp for sf in subfolders
        for fp in sf.iterdir()
        if dataframe_prefix in fp.name
        and fp.suffix.lower() == '.csv'
        and not fp.name.startswith('._')   # skip macOS resource-fork files
    ])


def load_tracking_data(root_folder: Path, selected_field: str, 
                      min_percentage_data_in_trajectory=0.3, dataframe_prefix='tracking_',
                      min_snr=1, max_missing_frames=5, smooth_window=1, verbose=True,
                      max_trajectory_length_percentile=None):
    """
    Load and process all tracking CSV files found in subfolders whose names include 'results_'.
    
    Parameters
    ----------
    root_folder : Path
        Root folder containing results subfolders
    selected_field : str
        Field name for intensity data
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data required
    dataframe_prefix : str
        Prefix for tracking files
    min_snr : float
        Minimum SNR threshold
    max_missing_frames : int
        Maximum consecutive NaN frames allowed
    smooth_window : int
        Smoothing window size
    verbose : bool
        Print progress messages
    max_trajectory_length_percentile : float or None
        Percentile threshold (0-100) for removing extra-long trajectories
        (likely aggregates). Trajectories with more finite frames than
        ``np.percentile(lengths, pct)`` are removed after concatenation.
        None or >= 100 disables the filter.
    
    Returns
    -------
    concatenated_intensity_arrays : ndarray
        Concatenated intensity data from all files
    number_of_cells : int
        Number of cells (files) processed
    total_number_of_spots : int
        Total number of trajectories
    cell_id_per_trajectory : ndarray
        1-D integer array (length = total_number_of_spots) mapping each
        trajectory row to its source cell index (0-based).  Use together
        with ``mi.Correlation.keep_mask_`` to count post-filter cells.
    """
    # Validation
    if not root_folder.exists():
        raise FileNotFoundError(f"root_folder does not exist: {root_folder}")

    tracking_files = _find_tracking_files(root_folder, dataframe_prefix)

    
    intensity_arrays = []
    cell_id_arrays  = []   # parallel list: cell index for each trajectory
    number_of_cells = 0
    
    for tracking_file in tracking_files:
        try:
            # Check file size (catch corrupted/empty files)
            if tracking_file.stat().st_size == 0:
                if verbose:
                    print(f"SKIP: {tracking_file.name} (empty file)")
                continue
            
            df = pd.read_csv(tracking_file)
            
            # Validate dataframe not empty
            if df.empty:
                if verbose:
                    print(f"SKIP: {tracking_file.name} (empty dataframe)")
                continue
            
            intensity_array = extract_intensity_and_shift_data(
                df, selected_field=selected_field,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                smooth_window=smooth_window,
                min_snr=min_snr,
                max_missing_frames=max_missing_frames,
                verbose=verbose
            )
            
            if intensity_array.shape[0] > 0:
                intensity_arrays.append(intensity_array)
                # tag every trajectory in this file with the current cell index
                cell_id_arrays.append(np.full(intensity_array.shape[0], number_of_cells, dtype=int))
                number_of_cells += 1
            elif verbose:
                print(f"SKIP: {tracking_file.name} (no valid trajectories)")
                
        except pd.errors.EmptyDataError:
            if verbose:
                print(f"SKIP: {tracking_file.name} (empty CSV)")
            continue
        except pd.errors.ParserError as e:
            if verbose:
                print(f"ERROR parsing {tracking_file.name}: {e}")
            continue
        except Exception as e:
            print(f'Error processing {tracking_file}: {e}')
    
    if not intensity_arrays:
        raise ValueError(
            f"No valid data found in {len(tracking_files)} files.\n"
            f"  Possible issues:\n"
            f"    - Field '{selected_field}' missing\n"
            f"    - All trajectories failed quality filtering (min_snr={min_snr})\n"
            f"    - SNR fields not available"
        )
    
    concatenated_intensity_arrays = np.concatenate(intensity_arrays, axis=0)
    cell_id_per_trajectory        = np.concatenate(cell_id_arrays,   axis=0)
    total_number_of_spots = np.shape(concatenated_intensity_arrays)[0]
    
    # Apply upper-bound trajectory-length filter (aggregate removal)
    concatenated_intensity_arrays, length_mask = filter_long_trajectories(
        concatenated_intensity_arrays,
        max_trajectory_length_percentile=max_trajectory_length_percentile,
        verbose=verbose,
    )
    cell_id_per_trajectory = cell_id_per_trajectory[length_mask]
    n_after_length_filter = concatenated_intensity_arrays.shape[0]
    
    # Guard: raise if filter removed everything
    if n_after_length_filter == 0:
        raise ValueError(
            f"All {total_number_of_spots} trajectories were removed by the "
            f"trajectory-length filter (max_trajectory_length_percentile="
            f"{max_trajectory_length_percentile}). Consider raising the percentile."
        )
    
    return concatenated_intensity_arrays, number_of_cells, total_number_of_spots, cell_id_per_trajectory


def compute_kinetics_from_acf(results: dict, gene_length: int,
                              ribosomal_footprint: int = 10) -> dict:
    """Derive translation kinetics from an ACF result dictionary.

    Uses the exponential decay constant τ_c from the fit and the Area-Under-
    Curve equivalence T_dwell = 2·τ_c to compute the geometric transit dwell
    time, elongation rate, initiation rate, and ribosomal loading.

    Parameters
    ----------
    results : dict
        Output of compute_autocorrelation_for_dataset(). Must contain keys
        'fit_params_' (with sub-key 'tau_c') and 'mean_correlation'.
    gene_length : int
        Gene length in codons.
    ribosomal_footprint : int
        Ribosomal footprint in codons (default 10).

    Returns
    -------
    dict with keys: tau_c, dwell_time, ke, ki, ribosomal_density,
                    n_ribosomes, ribosomal_distance
    """
    fit = results['fit_params_']
    if fit is None:
        raise ValueError(
            "No exponential fit parameters available (fit_params_ is None). "
            "Ensure the ACF was computed with fit_type='exponential'."
        )
    tau_c = fit['tau_c']
    dwell_time = 2.0 * tau_c   # AUC equivalence: T_dwell = 2 * τ_c

    G1 = results['mean_correlation'][1]
    ke = gene_length / dwell_time
    ki = 1.0 / (G1 * dwell_time)

    rho = (ki * ribosomal_footprint / ke) * 100
    n_rib = (ki * gene_length) / ke
    rib_dist = gene_length / n_rib

    return {
        'tau_c': round(tau_c, 4),
        'dwell_time': round(dwell_time, 2),
        'ke': round(ke, 4),
        'ki': round(ki, 4),
        'ribosomal_density': round(rho, 3),
        'n_ribosomes': round(n_rib, 3),
        'ribosomal_distance': round(rib_dist, 3),
    }



def compute_autocorrelation_for_dataset(
    dataset,
    data_folder: Path = None,  # Made optional for simulation mode
    results_folder=None,  # Made optional for simulation mode
    selected_field='spot_int_ch_',
    channel_index=1,
    step_size_in_sec=5,
    start_lag=1,
    min_percentage_data_in_trajectory=0.2,
    max_missing_frames=2,
    max_lag=None,
    downsample=False,
    downsampling_factor=3,
    use_global_mean=False,
    control_spots_mode=False,
    correct_baseline=True,
    use_linear_projection_for_lag_0=False,
    min_snr=1,
    smooth_window=1,
    remove_outliers=True,
    MAD_THRESHOLD_FACTOR=4,
    multi_tau=True,
    multi_tau_raw_points=60,
    multi_tau_bins_per_stage=30,
    y_axes_min_max_list_values=None,
    x_axes_min_max_list_values=None,
    save_results=True,
    plot_name=None,
    show_plot=True,
    verbose=True,
    save_plots=False,
    fit_type='linear',
    plot_individual_trajectories =False,
    detrend_photobleaching=False,
    de_correlation_threshold=0.001,
    # Aggregate removal
    max_trajectory_length_percentile=None,  # None = no filter, 99 = remove top 1%
    # Simulation mode parameters
    simulation_mode=False,  # Enable simulation mode
    SSA_data=None,  # Simulated data array (n_trajectories, n_timepoints)
    index_max_lag_for_fit=None,
    baseline_method = 'auto_plateau',     # 'auto_plateau' | 'exp_tail' | 'percentile' | 'none'
    baseline_plateau_fraction = 0.25,   # fallback: last fraction of positive lags
    baseline_percentile = 10.0,         # fallback percentile (for 'percentile' and last-tail guards)
    baseline_smooth_window = 7,           # odd number
    baseline_min_points = 5,
    baseline_weight_by_pairs = True,
    line_color='blue',
    line_color_fit='red',
    figsize=(8, 6),
):
    """
    Compute autocorrelation function (ACF) for experimental or simulated data.
    
    Parameters:
    -----------
    dataset : str or list of str
        Dataset identifier (e.g., 'cof') or list of identifiers
    data_folder : Path or None
        Folder containing data (required if simulation_mode=False)
    results_folder : Path or None
        Folder to save results (optional if simulation_mode=True)
    selected_field : str
        Base field name for intensity (default: 'spot_int_ch_')
    channel_index : int
        Channel number to analyze (0 or 1)
    step_size_in_sec : float
        Time interval between frames in seconds
    start_lag : int
        Starting lag index for analysis
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data points per trajectory
    max_missing_frames : int
        Maximum consecutive missing frames allowed
    max_lag : int or None
        Maximum lag index for correlation calculation
    downsample : bool
        Whether to downsample the data
    downsampling_factor : int
        Downsampling factor if downsample=True
    control_spots_mode : bool
        Whether analyzing control spots
    min_snr : float
        Minimum signal-to-noise ratio threshold
    smooth_window : int
        Smoothing window size
    remove_outliers : bool
        Whether to remove outlier trajectories
    MAD_THRESHOLD_FACTOR : float
        Median absolute deviation threshold for outlier removal
    multi_tau : bool
        Whether to use multi-tau correlation
    multi_tau_raw_points : int
        Number of raw points for multi-tau
    multi_tau_bins_per_stage : int
        Bins per stage for multi-tau
    y_axes_min_max_list_values : list or None
        Y-axis limits for plotting [min, max]
    x_axes_min_max_list_values : list or None
        X-axis limits for plotting [min, max]
    save_results : bool
        Whether to save results to CSV and text files
    show_plot : bool
        Whether to display plots
    verbose : bool
        Whether to print progress messages
    
    Simulation Mode:
    ---------------
    simulation_mode : bool
        If True, uses SSA_data instead of loading from files
    SSA_data : array-like or None
        Simulated data with shape (n_trajectories, n_timepoints)
        Required if simulation_mode=True
    
    Returns:
    --------
    dict or list of dict:
        Each dict contains:
            - 'mean_correlation': Mean autocorrelation values
            - 'std_correlation': Standard deviation of autocorrelation
            - 'lags': Lag values in seconds
            - 'correlations_array': Full correlation array
            - 'dwell_time': Calculated dwell time
            - 'total_number_of_cells': Number of cells processed
            - 'total_number_of_spots': Total number of spots
            - 'number_of_trajectories_final': Final trajectory count
            - 'plot_name': Name used for saving files
            - 'df': DataFrame with correlation results
            - 'dataset': Dataset identifier
            - 'data_source': 'simulation' or 'experimental'
    """
    
    # Validate simulation mode inputs
    if simulation_mode:
        if SSA_data is None:
            raise ValueError("simulation_mode=True requires SSA_data to be provided")
        SSA_data = np.asarray(SSA_data, dtype=float)
        if SSA_data.ndim != 2:
            raise ValueError(f"SSA_data must be 2D (n_trajectories, n_timepoints), got shape {SSA_data.shape}")
        if verbose:
            print(f"SIMULATION MODE: Using provided SSA data")
            print(f"  SSA data shape: {SSA_data.shape}")
    else:
        if data_folder is None:
            raise ValueError("data_folder is required when simulation_mode=False")
    
    # Helper function to process a single dataset
    def _process_single_dataset(ds):
        if verbose:
            print(f'Processing dataset: {ds}')
            print(f'Data source: {"SIMULATION" if simulation_mode else "EXPERIMENTAL"}\n')
        
        # ===== DATA LOADING BRANCH =====
        if simulation_mode:
            # Use simulated data directly
            primary_data = SSA_data.copy()
            total_number_of_cells = 1  # Simulations don't have "cells"
            total_number_of_spots = SSA_data.shape[0]
            plot_name_data = f'{ds}_simulation'
            
            if verbose:
                print(f"Using simulated data:")
                print(f"  Trajectories: {total_number_of_spots}")
                print(f"  Timepoints: {SSA_data.shape[1]}")
            primary_data_cell_ids = None  # no cell concept in simulation
        else:
            # Load experimental data
            folder_with_files, plot_name_data, dataframe_prefix = dataset_selection(
                ds, data_folder, control_spots_mode, downsample
            )
            
            # Load tracking data for specified channel
            (array_int_all_days, total_number_of_cells,
             total_number_of_spots, primary_data_cell_ids) = load_tracking_data(
                folder_with_files,
                selected_field=selected_field + str(channel_index),
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                dataframe_prefix=dataframe_prefix,
                smooth_window=smooth_window,
                min_snr=min_snr,
                max_missing_frames=max_missing_frames,
                verbose=verbose,
                max_trajectory_length_percentile=max_trajectory_length_percentile,
            )
            primary_data = array_int_all_days
        
        # ===== COMMON PROCESSING (both experimental and simulation) =====
        if downsample:
            primary_data = mi.Utilities().downsample_array(
                primary_data, factor=downsampling_factor, method='average'
            )
            step_size_in_sec_downsampled = step_size_in_sec * downsampling_factor
            if verbose:
                print(f"Data downsampled by factor {downsampling_factor}")
        else:
            step_size_in_sec_downsampled = step_size_in_sec
        
        # Calculate autocorrelation
        if verbose:
            print("Computing autocorrelation...")
        
        with joblib.parallel_backend('threading', n_jobs=1):
            corr_obj = mi.Correlation(
                primary_data=primary_data,
                max_lag=max_lag,
                nan_handling='ignore',   # 'forward_fill' biases the mean: last value repeats into NaN-padded tail
                shift_data=False,        # data already shifted upstream by extract_intensity_and_shift_data
                return_full=False,
                time_interval_between_frames_in_seconds=step_size_in_sec_downsampled,
                use_bootstrap=True,
                show_plot=show_plot,
                start_lag=start_lag,
                fit_type=fit_type,
                de_correlation_threshold=de_correlation_threshold,
                correct_baseline=correct_baseline,
                use_linear_projection_for_lag_0=use_linear_projection_for_lag_0,
                save_plots=save_plots,
                use_global_mean=use_global_mean,
                remove_outliers=remove_outliers,
                MAD_THRESHOLD_FACTOR=MAD_THRESHOLD_FACTOR,
                plot_individual_trajectories=plot_individual_trajectories,
                y_axes_min_max_list_values=y_axes_min_max_list_values,
                x_axes_min_max_list_values=x_axes_min_max_list_values,
                multi_tau=multi_tau,
                multi_tau_raw_points=multi_tau_raw_points,
                multi_tau_bins_per_stage=multi_tau_bins_per_stage,
                plot_title=f"{'[SIM] ' if simulation_mode else ''}{plot_name_data}",
                index_max_lag_for_fit=index_max_lag_for_fit,
                baseline_method = baseline_method,     # 'auto_plateau' | 'exp_tail' | 'percentile' | 'none'
                baseline_plateau_fraction = baseline_plateau_fraction,   # fallback: last fraction of positive lags
                baseline_percentile = baseline_percentile,         # fallback percentile (for 'percentile' and last-tail guards)
                baseline_smooth_window = baseline_smooth_window,           # odd number
                baseline_min_points = baseline_min_points,
                baseline_weight_by_pairs = baseline_weight_by_pairs,
                line_color=line_color,
                line_color_fit=line_color_fit,
                plot_name=plot_name,
                figsize=figsize,
                detrend_photobleaching=detrend_photobleaching,
            )
            mean_correlation, std_correlation, lags, correlations_array, dwell_time = corr_obj.run()

        # ── Fallback exponential fit when engine didn't expose fit_params_ ──
        # (happens when show_plot=False — the fit is tied to the plotting code)
        fit_params_ = getattr(corr_obj, 'fit_params_', None)
        if fit_params_ is None and fit_type == 'exponential':
            from scipy.optimize import curve_fit as _curve_fit
            _mc = np.asarray(mean_correlation, dtype=float)
            _lg = np.asarray(lags, dtype=float)
            _si = int(max(start_lag, 0))
            if index_max_lag_for_fit is not None:
                _G = _mc[_si:int(index_max_lag_for_fit)]
                _T = _lg[_si:int(index_max_lag_for_fit)]
            else:
                _G = _mc[_si:]
                _T = _lg[_si:]
            _G = np.nan_to_num(_G)
            if len(_G) >= 3:
                _tail = max(1, len(_G) // 10)
                _C0 = float(np.mean(_G[-_tail:]))
                _A0 = max(float(_G[0]) - _C0, 1e-6)
                _tv = _C0 + _A0 / np.e
                _it = int(np.argmin(np.abs(_G - _tv)))
                _tc0 = float(_T[_it]) if _it > 0 else float(_T[-1] / 2 if len(_T) > 1 else 1.0)
                _tc0 = max(_tc0, 1e-6)
                try:
                    _p, _ = _curve_fit(
                        lambda t, A, tc, C: A * np.exp(-t / tc) + C,
                        _T, _G, p0=[_A0, _tc0, _C0], maxfev=100000,
                        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    )
                    fit_params_ = {'A': float(_p[0]), 'tau_c': float(_p[1]), 'C': float(_p[2]), 'taus': _T}
                    dwell_time = 2.0 * float(_p[1])
                except Exception:
                    pass  # fit_params_ stays None

        # Create results DataFrame
        df = pd.DataFrame(data={
            'lags': lags,
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation
        })

        # Compute post-filter cell count using the outlier mask exposed by mi.Correlation
        number_of_trajectories_final = np.shape(correlations_array)[0]
        if primary_data_cell_ids is not None and hasattr(corr_obj, 'keep_mask_'):
            surviving_cell_ids = primary_data_cell_ids[corr_obj.keep_mask_]
            number_of_cells_final = int(np.unique(surviving_cell_ids).size)
        else:
            number_of_cells_final = total_number_of_cells  # simulation: no subset possible

        # ===== SAVE RESULTS =====
        if save_results and results_folder is not None:
            # Construct filename
            df_name = f'df_ACF_{plot_name_data}'
            if control_spots_mode:
                df_name = df_name + '_control_spots'
            if simulation_mode:
                df_name = df_name + '_simulation'
            
            # Save CSV
            df.to_csv(results_folder.joinpath(df_name + '.csv'), index=False)
            if verbose:
                print(f"Saved correlation data to: {df_name}.csv")
            
            # Save text report
            report_file = results_folder.joinpath(f'report_ACF_{plot_name_data}.txt')
            with open(report_file, 'w') as f:
                f.write(f'Data source: {"Simulation" if simulation_mode else "Experimental"}\n')
                f.write(f'Dataset: {ds}\n')
                f.write(f'Total number of cells: {total_number_of_cells}\n')
                f.write(f'Total number of spots/trajectories: {total_number_of_spots}\n')
                f.write(f'Number of trajectories after filtering: {number_of_trajectories_final}\n')
                if simulation_mode:
                    f.write(f'Simulation array shape: {SSA_data.shape}\n')
            
            if verbose:
                print(f"Saved report to: {report_file.name}")
        elif save_results and results_folder is None:
            if verbose:
                print("Warning: save_results=True but results_folder=None, skipping save")
        
        # Print summary
        #if verbose:
        print(f'Number of trajectories: {number_of_trajectories_final}')
        print(f'Number of cells (post-filter): {number_of_cells_final}')
        #    print(f'Data source: {"Simulation" if simulation_mode else "Experimental"}')
        #    print('-----------------------------------------------------\n')
        
        # Return results dictionary
        return {
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'lags': lags,
            'correlations_array': correlations_array,
            'dwell_time': dwell_time,
            'total_number_of_cells': total_number_of_cells,
            'total_number_of_spots': total_number_of_spots,
            'number_of_trajectories_final': number_of_trajectories_final,
            'number_of_cells_final': number_of_cells_final,   # post-filter cell count
            'fit_params_': fit_params_,  # exact A, tau_c, C from curve_fit (engine or fallback)
            'plot_name': plot_name_data,
            'df': df,
            'dataset': ds,
            'data_source': 'simulation' if simulation_mode else 'experimental',
            'max_trajectory_length_percentile': max_trajectory_length_percentile,
            'primary_data': primary_data,  # (n_traj, n_frames) for quality diagnostics
        }
    
    # ===== HANDLE SINGLE OR MULTIPLE DATASETS =====
    if isinstance(dataset, str):
        return _process_single_dataset(dataset)
    elif isinstance(dataset, (list, tuple)):
        if verbose:
            print(f"Processing {len(dataset)} datasets...\n")
        
        results_list = []
        for ds in dataset:
            result = _process_single_dataset(ds)
            results_list.append(result)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETED: Processed {len(results_list)} datasets")
            print(f"{'='*60}\n")
        
        return results_list
    else:
        raise TypeError(f"dataset must be str or list of str, got {type(dataset)}")




def load_dual_channel_tracking_data(
    root_folder: Path, 
    base_field: str, 
    primary_channel=1,  
    secondary_channel=0,  
    min_percentage_data_in_trajectory=0.3, 
    dataframe_prefix='tracking_',
    min_snr=1,
    max_missing_frames=5, 
    verbose=True, 
    smooth_window=1,
    max_trajectory_length_percentile=None,
):
    """
    Load and process dual-channel tracking data ensuring synchronized trajectories.
    
    Parameters
    ----------
    root_folder : Path
        Root folder containing results subfolders
    base_field : str
        Base field name (e.g., 'spot_int_ch_')
    primary_channel : int
        Channel index for primary data (0 or 1)
    secondary_channel : int
        Channel index for secondary data (0 or 1)
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data required
    dataframe_prefix : str
        Prefix for tracking files
    min_snr : float
        Minimum SNR threshold (applied to BOTH channels)
    max_missing_frames : int
        Maximum consecutive NaN frames allowed
    verbose : bool
        Print progress messages
    smooth_window : int
        Smoothing window size
    
    Returns
    -------
    primary_concatenated : ndarray
        Concatenated primary channel data
    secondary_concatenated : ndarray
        Concatenated secondary channel data
    total_files_processed : int
        Number of files successfully processed
    """
    # Validation
    if not root_folder.exists():
        raise FileNotFoundError(f"root_folder does not exist: {root_folder}")

    tracking_files = _find_tracking_files(root_folder, dataframe_prefix)

    primary_arrays = []
    secondary_arrays = []
    total_files_processed = 0
    
    if verbose:
        print(f"Processing {len(tracking_files)} files for dual-channel analysis...")
        print(f"Primary channel: {primary_channel}, Secondary channel: {secondary_channel}")
    
    for tracking_file in tracking_files:
        try:
            # Check file size
            if tracking_file.stat().st_size == 0:
                if verbose:
                    print(f"\nSKIP: {tracking_file.name} (empty file)")
                continue
            
            df = pd.read_csv(tracking_file)
            
            # Validate dataframe not empty
            if df.empty:
                if verbose:
                    print(f"\nSKIP: {tracking_file.name} (empty dataframe)")
                continue
            
            if verbose:
                print(f"\nProcessing: {tracking_file.name}")
            
            # Extract both channels using specified indices
            primary_field = base_field + str(primary_channel)
            secondary_field = base_field + str(secondary_channel)
            
            # Check if both fields exist
            if verbose:
                print(f"  Checking for fields: {primary_field}, {secondary_field}")
            
            if primary_field not in df.columns or secondary_field not in df.columns:
                if verbose:
                    print(f"  SKIP: Missing channels. Available: {[col for col in df.columns if 'spot_int_ch' in col]}")
                continue
            
            # Extract synchronized data
            primary_array, secondary_array = extract_dual_channel_data(
                df, 
                primary_field=primary_field,
                secondary_field=secondary_field,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                min_snr=min_snr,
                max_missing_frames=max_missing_frames,
                smooth_window=smooth_window,
                verbose=verbose
            )
            
            if primary_array is not None and secondary_array is not None:
                primary_arrays.append(primary_array)
                secondary_arrays.append(secondary_array)
                total_files_processed += 1
                if verbose:
                    print(f"  SUCCESS: {primary_array.shape[0]} synchronized trajectories")
            else:
                if verbose:
                    print(f"  SKIP: No valid synchronized trajectories")
                    
        except pd.errors.EmptyDataError:
            if verbose:
                print(f"\nSKIP: {tracking_file.name} (empty CSV)")
            continue
        except pd.errors.ParserError as e:
            if verbose:
                print(f"\nERROR parsing {tracking_file.name}: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if not primary_arrays:
        raise ValueError(
            f"No valid dual-channel data found in {len(tracking_files)} files.\n"
            f"  Possible issues:\n"
            f"    - Channel fields '{base_field}{primary_channel}' or '{base_field}{secondary_channel}' missing\n"
            f"    - All trajectories failed quality filtering (min_snr={min_snr}, min_pct={min_percentage_data_in_trajectory})\n"
            f"    - SNR fields not available in data"
        )
    
    # Concatenate all arrays
    primary_concatenated = np.concatenate(primary_arrays, axis=0)
    secondary_concatenated = np.concatenate(secondary_arrays, axis=0)
    
    # Apply upper-bound trajectory-length filter (aggregate removal)
    # Use primary channel lengths to compute the cutoff, apply same mask to both
    _, length_mask = filter_long_trajectories(
        primary_concatenated,
        max_trajectory_length_percentile=max_trajectory_length_percentile,
        verbose=verbose,
    )
    primary_concatenated = primary_concatenated[length_mask]
    secondary_concatenated = secondary_concatenated[length_mask]
    
    if primary_concatenated.shape[0] == 0:
        raise ValueError(
            f"All trajectories removed by trajectory-length filter "
            f"(max_trajectory_length_percentile={max_trajectory_length_percentile})."
        )
    
    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"Files processed: {total_files_processed}/{len(tracking_files)}")
        print(f"Final primary (ch_{primary_channel}) shape: {primary_concatenated.shape}")
        print(f"Final secondary (ch_{secondary_channel}) shape: {secondary_concatenated.shape}")
        print(f"Total synchronized trajectories: {primary_concatenated.shape[0]}")
    
    return primary_concatenated, secondary_concatenated, total_files_processed



def compute_cross_correlation_for_dataset(
    dataset,
    data_folder: Path = None,  # Made optional for simulation mode
    results_folder=None,  # Made optional for simulation mode
    selected_field='spot_int_ch_',
    primary_channel=1,  # Primary channel index (default: 1 for HA_TAG)
    secondary_channel=0,  # Secondary channel index (default: 0 for GFP_TAG)
    step_size_in_sec=5,
    start_lag=1,
    min_percentage_data_in_trajectory=0.2,
    max_missing_frames=2,
    downsample=False,
    downsampling_factor=3,
    control_spots_mode=False,
    min_snr=1,
    smooth_window=1,
    MAD_THRESHOLD_FACTOR=4,
    x_axes_min_max_list_values=None,
    y_axes_min_max_list_values=None,
    save_results=True,
    show_plot=True,
    verbose=True,
    # Cross-correlation specific parameters
    use_bootstrap=True,
    fit_type='exponential',
    de_correlation_threshold=0.005,
    correct_baseline=False,
    use_linear_projection_for_lag_0=False,
    use_global_mean=True,
    remove_outliers=False,
    return_full=True,
    max_lag=None,
    shift_data=True,
    nan_handling='forward_fill',
    # Simulation mode parameters
    simulation_mode=False,
    SSA_data_1=None,
    SSA_data_2=None,
    detrend=True,
    # Aggregate removal
    max_trajectory_length_percentile=None,
):
    """
    Compute cross-correlation function (CCF) for experimental or simulated dual-channel data.
    
    Parameters:
    -----------
    dataset : str or list of str
        Dataset identifier (e.g., 'cof') or list of identifiers
    data_folder : Path or None
        Folder containing data (required if simulation_mode=False)
    results_folder : Path or None
        Folder to save results (optional if simulation_mode=True)
    selected_field : str
        Base field name for intensity (default: 'spot_int_ch_')
    primary_channel : int
        Channel index for primary data (default: 1 for HA_TAG)
    secondary_channel : int
        Channel index for secondary data (default: 0 for GFP_TAG)
    step_size_in_sec : float
        Time interval between frames in seconds
    start_lag : int
        Starting lag index for analysis
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data points per trajectory
    max_missing_frames : int
        Maximum consecutive missing frames allowed
    downsample : bool
        Whether to downsample the data
    downsampling_factor : int
        Downsampling factor if downsample=True
    control_spots_mode : bool
        Whether analyzing control spots
    min_snr : float
        Minimum signal-to-noise ratio threshold
    smooth_window : int
        Smoothing window size
    MAD_THRESHOLD_FACTOR : float
        Median absolute deviation threshold for outlier removal
    x_axes_min_max_list_values : list or None
        X-axis limits for plotting [min, max]
    y_axes_min_max_list_values : list or None
        Y-axis limits for plotting [min, max]
    save_results : bool
        Whether to save results to CSV and text files
    show_plot : bool
        Whether to display plots
    verbose : bool
        Whether to print progress messages
    use_bootstrap : bool
        Whether to use bootstrap for error estimation
    fit_type : str
        Type of fit to use ('exponential', etc.)
    de_correlation_threshold : float
        Threshold for decorrelation
    correct_baseline : bool
        Whether to correct baseline (typically False for CCF)
    use_linear_projection_for_lag_0 : bool
        Whether to use linear projection for lag 0 (typically False for CCF)
    use_global_mean : bool
        Whether to use global mean (typically True for CCF)
    remove_outliers : bool
        Whether to remove outlier trajectories
    return_full : bool
        Whether to return full correlation array (typically True for CCF)
    max_lag : int or None
        Maximum lag to compute
    shift_data : bool
        Whether to shift data (typically False for simulations)
    nan_handling : str
        How to handle NaN values ('forward_fill', 'ignore', etc.)
    
    Simulation Mode:
    ---------------
    simulation_mode : bool
        If True, uses SSA_data_1 and SSA_data_2 instead of loading from files
    SSA_data_1 : array-like or None
        Primary channel simulated data with shape (n_trajectories, n_timepoints)
        Required if simulation_mode=True
    SSA_data_2 : array-like or None
        Secondary channel simulated data with shape (n_trajectories, n_timepoints)
        Required if simulation_mode=True
    
    Returns:
    --------
    dict or list of dict:
        Each dict contains:
            - 'mean_correlation': Mean cross-correlation values
            - 'std_correlation': Standard deviation of cross-correlation
            - 'lags': Lag values in seconds
            - 'correlations_array': Full correlation array
            - 'dwell_time': Calculated dwell time
            - 'total_number_of_cells': Number of cells processed
            - 'total_number_of_spots': Total number of synchronized spots
            - 'number_of_trajectories_final': Final trajectory count after filtering
            - 'plot_name': Name used for saving files
            - 'df': DataFrame with correlation results
            - 'dataset': Dataset identifier
            - 'primary_data': Primary channel data array
            - 'secondary_data': Secondary channel data array
            - 'primary_channel': Channel index used for primary
            - 'secondary_channel': Channel index used for secondary
            - 'analysis_type': 'cross-correlation'
            - 'data_source': 'simulation' or 'experimental'
    """
    
    # Validate inputs
    if primary_channel not in [0, 1]:
        raise ValueError(f"primary_channel must be 0 or 1, got {primary_channel}")
    if secondary_channel not in [0, 1]:
        raise ValueError(f"secondary_channel must be 0 or 1, got {secondary_channel}")
    
    # Validate simulation mode inputs
    if simulation_mode:
        if SSA_data_1 is None or SSA_data_2 is None:
            raise ValueError("simulation_mode=True requires both SSA_data_1 and SSA_data_2 to be provided")
        
        SSA_data_1 = np.asarray(SSA_data_1, dtype=float)
        SSA_data_2 = np.asarray(SSA_data_2, dtype=float)
        
        if SSA_data_1.ndim != 2:
            raise ValueError(f"SSA_data_1 must be 2D (n_trajectories, n_timepoints), got shape {SSA_data_1.shape}")
        if SSA_data_2.ndim != 2:
            raise ValueError(f"SSA_data_2 must be 2D (n_trajectories, n_timepoints), got shape {SSA_data_2.shape}")
        if SSA_data_1.shape != SSA_data_2.shape:
            raise ValueError(f"SSA_data_1 and SSA_data_2 must have same shape, got {SSA_data_1.shape} vs {SSA_data_2.shape}")
        
        if verbose:
            print(f"SIMULATION MODE: Using provided SSA data")
            print(f"  Primary data shape: {SSA_data_1.shape}")
            print(f"  Secondary data shape: {SSA_data_2.shape}")
    else:
        if data_folder is None:
            raise ValueError("data_folder is required when simulation_mode=False")
    
    # Helper function to process a single dataset
    def _process_single_dataset(ds):
        if verbose:
            print(f'Processing dataset: {ds}')
            print(f'Data source: {"SIMULATION" if simulation_mode else "EXPERIMENTAL"}')
            print('Analysis type: Cross-correlation (CCF)')
            print(f'Primary channel: {primary_channel} (reference)')
            print(f'Secondary channel: {secondary_channel} (delayed)\n')
        
        # ===== DATA LOADING BRANCH =====
        if simulation_mode:
            # Use simulated data directly
            primary_data = SSA_data_1.copy()
            secondary_data = SSA_data_2.copy()
            total_number_of_cells = 1  # Simulations don't have "cells"
            total_number_of_spots = SSA_data_1.shape[0]
            plot_name = f'{ds}_simulation'
            
            if verbose:
                print(f"Using simulated data:")
                print(f"  Primary trajectories: {total_number_of_spots}")
                print(f"  Secondary trajectories: {total_number_of_spots}")
                print(f"  Timepoints: {SSA_data_1.shape[1]}")
        else:
            # Load experimental data
            folder_with_files, plot_name, dataframe_prefix = dataset_selection(
                ds, data_folder, control_spots_mode, downsample
            )
            
            # Load synchronized dual-channel data
            if verbose:
                print("Loading synchronized dual-channel data...")
            
            primary_data, secondary_data, total_number_of_cells = load_dual_channel_tracking_data(
                folder_with_files,
                base_field=selected_field,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                dataframe_prefix=dataframe_prefix,
                min_snr=min_snr,
                max_missing_frames=max_missing_frames,
                smooth_window=smooth_window,
                verbose=verbose,
                primary_channel=primary_channel,
                secondary_channel=secondary_channel,
                max_trajectory_length_percentile=max_trajectory_length_percentile,
            )
            total_number_of_spots = primary_data.shape[0]
            
            if verbose:
                print(f"Synchronized data shapes:")
                print(f"  Primary (ch_{primary_channel}): {primary_data.shape}")
                print(f"  Secondary (ch_{secondary_channel}): {secondary_data.shape}")
        
        # ===== COMMON PROCESSING (both experimental and simulation) =====
        
        # Handle downsampling
        if downsample:
            primary_data = mi.Utilities().downsample_array(
                primary_data, factor=downsampling_factor, method='average'
            )
            secondary_data = mi.Utilities().downsample_array(
                secondary_data, factor=downsampling_factor, method='average'
            )
            step_size_in_sec_downsampled = step_size_in_sec * downsampling_factor
            if verbose:
                print(f"Data downsampled by factor {downsampling_factor}")
        else:
            step_size_in_sec_downsampled = step_size_in_sec
        
        # Compute cross-correlation
        if verbose:
            print("Computing cross-correlation...")
        
        with joblib.parallel_backend('threading', n_jobs=1):
            mean_correlation, std_correlation, lags, correlations_array, dwell_time = mi.Correlation(
                primary_data=primary_data,
                secondary_data=secondary_data,
                max_lag=max_lag,
                nan_handling=nan_handling,
                shift_data=shift_data if not simulation_mode else False,  # Don't shift simulated data
                return_full=return_full,
                time_interval_between_frames_in_seconds=step_size_in_sec_downsampled,
                use_bootstrap=use_bootstrap,
                show_plot=show_plot,
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
                multi_tau=False,  # Typically False for CCF
                plot_title=f"{'[SIM] ' if simulation_mode else ''}{plot_name}",
                detrend=detrend,
            ).run()
        
        # Create results DataFrame
        df = pd.DataFrame(data={
            'lags': lags,
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation
        })
        
        # Calculate final trajectory count
        number_of_trajectories_final = np.shape(correlations_array)[0]
        
        # ===== SAVE RESULTS =====
        if save_results and results_folder is not None:
            # Save CSV with channel info in filename
            df_name = f'df_CrossCorr_{plot_name}_ch{primary_channel}_vs_ch{secondary_channel}'
            if control_spots_mode:
                df_name = df_name + '_control_spots'
            if simulation_mode:
                df_name = df_name + '_simulation'
            
            df.to_csv(results_folder.joinpath(df_name + '.csv'), index=False)
            if verbose:
                print(f"Saved cross-correlation data to: {df_name}.csv")
            
            # Save text report
            report_file = results_folder.joinpath(f'report_crosscorr_{plot_name}_ch{primary_channel}_vs_ch{secondary_channel}.txt')
            with open(report_file, 'w') as f:
                f.write(f'Data source: {"Simulation" if simulation_mode else "Experimental"}\n')
                f.write(f'Cross-correlation: ch{primary_channel} vs ch{secondary_channel}\n')
                f.write(f'Dataset: {ds}\n')
                f.write(f'Total number of cells: {total_number_of_cells}\n')
                f.write(f'Total number of synchronized spots: {total_number_of_spots}\n')
                f.write(f'Number of trajectories after all filtering: {number_of_trajectories_final}\n')
                if simulation_mode:
                    f.write(f'Simulation array shapes: {SSA_data_1.shape}\n')
            
            if verbose:
                print(f"Saved report to: {report_file.name}")
        elif save_results and results_folder is None:
            if verbose:
                print("Warning: save_results=True but results_folder=None, skipping save")
        
        # Print summary
        #if verbose:
        print(f'Final synchronized trajectories: {number_of_trajectories_final}')
        print(f'Number of cells: {total_number_of_cells}')
        #    print(f'Data source: {"Simulation" if simulation_mode else "Experimental"}')
        #    print('-----------------------------------------------------\n')
        
        # Return results dictionary
        return {
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'lags': lags,
            'correlations_array': correlations_array,
            'dwell_time': dwell_time,
            'total_number_of_cells': total_number_of_cells,
            'total_number_of_spots': total_number_of_spots,
            'number_of_trajectories_final': number_of_trajectories_final,
            'plot_name': plot_name,
            'df': df,
            'dataset': ds,
            'primary_data': primary_data,
            'secondary_data': secondary_data,
            'primary_channel': primary_channel,
            'secondary_channel': secondary_channel,
            'analysis_type': 'cross-correlation',
            'data_source': 'simulation' if simulation_mode else 'experimental'
        }
    
    # ===== HANDLE SINGLE OR MULTIPLE DATASETS =====
    if isinstance(dataset, str):
        return _process_single_dataset(dataset)
    elif isinstance(dataset, (list, tuple)):
        if verbose:
            print(f"Processing {len(dataset)} datasets for cross-correlation...\n")
        
        results_list = []
        for ds in dataset:
            result = _process_single_dataset(ds)
            results_list.append(result)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETED: Processed {len(results_list)} datasets (cross-correlation)")
            print(f"{'='*60}\n")
        
        return results_list
    else:
        raise TypeError(f"dataset must be str or list of str, got {type(dataset)}")


def analyze_crosscorr(
    corr,
    dt=5.0,
    lags_s=None,
    zero_index=None,
    window_s=600.0,
    slope_window_pts=3,
    halfmax_fraction=0.5,
    plot=False,
    ax=None,
    plateau_tolerance=0.02,  # New parameter for plateau detection
    sigma_smooth=1,  # Added sigma_smooth parameter
    min_max_normalize=True,  # New parameter for min-max normalization
    axis_lims =None,  # New parameter for axis limits
    figsize=(8, 4.2),
    results_folder=None,
    save_plot=False,
    dataset_name='crosscorr_analysis',
):
    """
    Analyze a cross-correlation curve to extract delay/shape metrics.
    Now handles flat plateaus by finding the center.
    
    Parameters
    ----------
    corr : (N,) array
        Cross-correlation values vs. lag. Can contain NaNs.
    dt : float, default 5.0
        Sampling interval (seconds per frame).
    lags_s : (N,) array or None
        Lag axis in seconds. If None, built from dt.
    zero_index : int or None
        Index corresponding to lag = 0 s. Defaults to N//2.
    window_s : float, default 600.0
        Time window (±window_s) for area-asymmetry calculation.
    slope_window_pts : int, default 3
        Number of points on each side of 0-lag for local slope fits.
    halfmax_fraction : float, default 0.5
        Fraction of peak for "dominant range" threshold.
    plot : bool, default False
        If True, plots the correlation and annotations.
    ax : matplotlib.axes.Axes or None
        Existing axis to plot into.
    plateau_tolerance : float, default 0.02
        Tolerance for plateau detection (2% below peak).
    sigma_smooth : float, default 1
        Standard deviation for Gaussian smoothing of correlation.
    min_max_normalize : bool, default True
        If True, normalize corr to [0, 1] range based on window_s range.
    results_folder : Path, optional
        Folder to save plots (if save_plot=True)
    save_plot : bool
        Whether to save the plot to results_folder
    dataset_name : str
        Name prefix for saved files
    """
    corr = np.asarray(corr).astype(float).squeeze()
    if corr.ndim != 1:
        raise ValueError(f"corr must be 1D, got shape {corr.shape}")

    N = len(corr)

    # Build lag axis if not provided
    if lags_s is None:
        if zero_index is None:
            zero_index = N // 2
        lags_frames = np.arange(N) - int(zero_index)
        lags_s = lags_frames * float(dt)
    else:
        lags_s = np.asarray(lags_s).astype(float).squeeze()
        if lags_s.shape != corr.shape:
            raise ValueError("lags_s must have same shape as corr")
    # Interpolate NaNs linearly for stable metrics/plotting
    finite = np.isfinite(corr)
    corr_interp = corr.copy()
    if not np.all(finite):
        good = np.where(finite)[0]
        bad = np.where(~finite)[0]
        if good.size == 0:
            raise ValueError("corr contains no finite values")
        corr_interp[bad] = np.interp(bad, good, corr[good])
    # Apply min-max normalization using ONLY the windowed region
    # Define the window mask
    mask = (lags_s >= -window_s) & (lags_s <= window_s)
    # Find min/max ONLY within the window
    windowed_data = corr_interp[mask]
    corr_min = np.min(windowed_data)
    corr_max = np.max(windowed_data)
    if min_max_normalize:    
        #print(f"Normalization window: ±{window_s} s ({np.sum(mask)} points)")
        print(f"Window range: [{corr_min:.4f}, {corr_max:.4f}]")
        if corr_max > corr_min:
            # Apply normalization to entire array using window min/max
            corr_interp = (corr_interp - corr_min) / (corr_max - corr_min)
            print(f"Min-max normalization applied to entire array using window range")
        else:
            # All values in window are the same
            corr_interp = corr_interp - corr_min
            print(f"Warning: Constant correlation values in window, centering at zero")
    # Apply Gaussian smoothing
    corr_smooth = gaussian_filter1d(corr_interp, sigma=sigma_smooth)
    # Peak finding: handle flat plateaus by finding center
    global_max_val = float(np.max(corr_smooth))
    threshold = global_max_val * (1.0 - plateau_tolerance)
    # Find all points in plateau
    plateau_mask = corr_smooth >= threshold
    plateau_indices = np.where(plateau_mask)[0]
    if len(plateau_indices) > 1:
        # Find center of plateau
        peak_idx = plateau_indices[len(plateau_indices) // 2]
        print(f"Flat plateau detected: {len(plateau_indices)} points, using center at lag {lags_s[peak_idx]:.1f} s")
        print(f"Plateau tolerance: {plateau_tolerance*100:.1f}%, corresponds to {global_max_val*plateau_tolerance:.3f} units below peak value {global_max_val:.3f}")
        print(f"Plateau lag range: {lags_s[plateau_indices[0]]:.1f} s to {lags_s[plateau_indices[-1]]:.1f} s")
    else:
        # Single peak
        peak_idx = plateau_indices[0]
    peak_val = float(corr_smooth[peak_idx])
    peak_lag = float(lags_s[peak_idx])
    # Weighted centroid (shift corr to be nonnegative)
    y_shift = corr_smooth - np.min(corr_smooth)
    denom = y_shift.sum()
    centroid = float(np.sum(lags_s * y_shift) / denom) if denom > 0 else np.nan
    # Half-maximum crossings & FWHM (linear interpolation) - use smoothed data
    def _halfmax_crossings(x, y, frac=0.5):
        p = int(np.argmax(y))
        pk = float(y[p])
        hm = frac * pk
        left_x = np.nan
        for i in range(p, 0, -1):
            y1, y2 = y[i-1], y[i]
            if (y1 - hm) * (y2 - hm) <= 0:
                x1, x2 = x[i-1], x[i]
                left_x = x1 if y2 == y1 else x1 + (hm - y1) * (x2 - x1) / (y2 - y1)
                break
        right_x = np.nan
        for i in range(p, len(y)-1):
            y1, y2 = y[i], y[i+1]
            if (y1 - hm) * (y2 - hm) <= 0:
                x1, x2 = x[i], x[i+1]
                right_x = x2 if y2 == y1 else x1 + (hm - y1) * (x2 - x1) / (y2 - y1)
                break
        width = float(right_x - left_x) if np.isfinite(left_x) and np.isfinite(right_x) else np.nan
        return width, left_x, right_x
    FWHM, left_hm, right_hm = _halfmax_crossings(lags_s, corr_smooth, frac=0.5)
    # Dominant range at specified fraction (contiguous region around peak above threshold)
    def _dominant_range(x, y, frac=0.5):
        p = int(np.argmax(y))
        thresh = frac * float(y[p])
        L = p
        while L-1 >= 0 and y[L-1] >= thresh:
            L -= 1
        R = p
        while R+1 < len(y) and y[R+1] >= thresh:
            R += 1
        return float(x[L]), float(x[R])
    dom_left, dom_right = _dominant_range(lags_s, corr_smooth, frac=halfmax_fraction)
    # Local slopes at 0 (least-squares line fits on small neighborhoods) - use smoothed data
    idx0 = int(np.argmin(np.abs(lags_s - 0.0)))
    L_slice = slice(max(0, idx0 - slope_window_pts), idx0 + 1)
    R_slice = slice(idx0, min(N, idx0 + slope_window_pts + 1))
    def _fit_slope(x, y):
        if len(x) < 2:
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)
    slope_left = _fit_slope(lags_s[L_slice], corr_smooth[L_slice])
    slope_right = _fit_slope(lags_s[R_slice], corr_smooth[R_slice])
    # Area asymmetry in ±window_s (only positive parts) - use smoothed data
    step = float(np.median(np.diff(lags_s)))
    wmask = np.abs(lags_s) <= float(window_s)
    xw = lags_s[wmask]
    yw = corr_smooth[wmask]
    pos_area = float(np.sum(np.clip(yw[xw > 0], 0, None)) * step) if np.any(xw > 0) else 0.0
    neg_area = float(np.sum(np.clip(yw[xw < 0], 0, None)) * step) if np.any(xw < 0) else 0.0
    asym = (pos_area - neg_area) / (pos_area + neg_area + 1e-12)
    metrics = {
        "n_points": N,
        "dt_seconds": float(dt),
        "peak_lag_seconds": peak_lag,
        "peak_value": peak_val,
        "centroid_lag_seconds": centroid,
        "FWHM_seconds": FWHM,
        "FWHM_left_crossing_seconds": left_hm,
        "FWHM_right_crossing_seconds": right_hm,
        "slope_left_at_0_per_s": slope_left,
        "slope_right_at_0_per_s": slope_right,
        "area_asymmetry_pm_window": asym,
        "dominant_left_seconds": dom_left,
        "dominant_right_seconds": dom_right,
    }
    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)        
        # Plot both raw and smoothed correlations
        ax.plot(lags_s, corr_interp, lw=1, color="blue", alpha=0.7, label="Raw (normalized)")
        # plot the corrected line, removing the time point where the lag values are 0.
        ax.plot(lags_s, corr_smooth, lw=2, color="k", label=f"Smoothed (σ={sigma_smooth})")
        ax.axvline(0, ls="--", lw=2, color="gray", alpha=0.5)
        ax.axvline(peak_lag, ls=":", lw=2, color="red", label=f"Peak: {peak_lag:.1f}s")
        # Highlight plateau region
        if len(plateau_indices) > 1:
            ax.axvspan(lags_s[plateau_indices[0]], lags_s[plateau_indices[-1]], 
                      alpha=0.2, color="yellow", label=f"Plateau ({len(plateau_indices)} pts)")
        # Plot FWHM markers
        if np.isfinite(FWHM):
            ax.axvline(left_hm, ls="-", lw=1, color="red", alpha=0.5)
            ax.axvline(right_hm, ls="-", lw=1, color="red", alpha=0.5)
            ax.hlines(halfmax_fraction * peak_val, left_hm, right_hm, color="red", lw=1, alpha=0.5)
        # Set plot limits
        if np.isfinite(window_s):
            ax.set_xlim(-1.1 * window_s, 1.1 * window_s)
        if axis_lims is not None:
            ax.set_ylim(axis_lims)
        else:
            y_min = np.min(corr_smooth[mask]) if np.any(mask) else np.min(corr_smooth)
            y_max = np.max(corr_smooth[mask]) if np.any(mask) else np.max(corr_smooth)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        #ax.set_title("Cross-correlation (Plateau-aware)")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Cross-correlation (normalized)" if min_max_normalize else "Cross-correlation (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        if ax is None:
            plt.tight_layout()
            # Save plot if requested
            if save_plot and results_folder is not None:
                results_folder = Path(results_folder)
                results_folder.mkdir(parents=True, exist_ok=True)
                base_name = f'{dataset_name}_crosscorr'
                plt.savefig(results_folder / f'{base_name}.png', dpi=150, bbox_inches='tight')
                plt.savefig(results_folder / f'{base_name}.svg', bbox_inches='tight')
                print(f"Saved cross-correlation plot to: {results_folder / base_name}.[png|svg]")
            plt.show()
    return metrics, corr_interp, lags_s




def _normalize_traces(X, method="min_max", percentile=95, percentile_min=5, percentile_max=95, eps=1e-9):
    """
    Normalize each row (trace) using specified method.
    
    Parameters
    ----------
    X : array-like, shape (n_traces, n_timepoints)
        Input data to normalize.
    method : str
        Normalization method:
        - "min_max": True min-max [0, 1]
        - "min_5_max_95": Percentile-based [P5, P95] → [0, 1]
        - "per_trace_percentile": Scale by percentile (legacy)
        - "per_trace_max": Scale by max
    percentile : int
        Percentile for "per_trace_percentile" method (default: 95).
    percentile_min : int
        Lower percentile for "min_5_max_95" (default: 5).
    percentile_max : int
        Upper percentile for "min_5_max_95" (default: 95).
    eps : float
        Small epsilon to prevent division by zero.
    
    Returns
    -------
    X_normalized : array
        Normalized traces.
    """
    X = np.asarray(X, dtype=float)
    
    if method == "min_max":
        # True min-max normalization
        X_min = np.min(X, axis=1, keepdims=True)
        X_max = np.max(X, axis=1, keepdims=True)
        range_val = np.maximum(X_max - X_min, eps)
        return (X - X_min) / range_val
    
    elif method == "min_5_max_95":
        # Percentile-based normalization (robust to outliers)
        X_min = np.percentile(X, percentile_min, axis=1, keepdims=True)
        X_max = np.percentile(X, percentile_max, axis=1, keepdims=True)
        range_val = np.maximum(X_max - X_min, eps)
        # Normalize to [0, 1] based on percentile range
        X_norm = (X - X_min) / range_val
        # Optional: clip values outside [0, 1] (if data exceeds percentiles)
        return np.clip(X_norm, 0.0, 1.0)
        
    elif method == "per_trace_percentile":
        # Your original percentile-based method (scales by single percentile)
        scale = np.percentile(X, percentile, axis=1, keepdims=True)
        scale = np.maximum(scale, eps)
        return np.clip(X / scale, 0.0, 1.0)
        
    elif method == "per_trace_max":
        # Max scaling
        scale = np.max(X, axis=1, keepdims=True)
        scale = np.maximum(scale, eps)
        return X / scale
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'min_max', 'min_5_max_95', 'per_trace_percentile', 'per_trace_max'")

def _gauss(t, baseline, amp, mu, sigma):
    """Gaussian with arbitrary sign amplitude: baseline + amp * exp(-0.5*((t-mu)/sigma)^2)."""
    return baseline + amp * np.exp(-0.5 * ((t - mu) / np.maximum(sigma, 1e-9)) ** 2)

def _fit_gaussian(y, t=None):
    """
    Fit Gaussian to a 1D signal y (amp can be negative for valleys).
    Returns dict with params and derived metrics; robust fallbacks on failure.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if t is None:
        t = np.arange(n) - (n // 2)

    # Heuristic initial guesses
    if n >= 6:
        edge_mean = np.mean([np.mean(y[:3]), np.mean(y[-3:])])
    else:
        edge_mean = np.mean([y[0], y[-1]]) if n >= 2 else float(y.mean())

    mu0 = 0.0
    amp0 = float(np.min(y) - edge_mean)  # negative for valleys
    sigma0 = max(n / 6.0, 1.0)

    p0 = [edge_mean, amp0, mu0, sigma0]
    bounds = ([-np.inf, -np.inf, -n, 1e-3], [np.inf, 0.0, n, n])  # enforce amp <= 0

    try:
        popt, _ = curve_fit(_gauss, t, y, p0=p0, bounds=bounds, maxfev=20000)
        baseline, amp, mu, sigma = popt
    except Exception:
        baseline, amp, mu, sigma = edge_mean, amp0, 0.0, sigma0

    return {
        "baseline": float(baseline),
        "amp": float(amp),
        "mu": float(mu),
        "sigma": float(sigma),
        "ymin": float(baseline + amp),
        "magnitude": float(-amp),  # positive dip magnitude
    }

def _mean_sem(X):
    """
    Calculate mean and standard error of the mean (SEM) for each time point.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_timepoints)
        Data array where each row is a sample/trajectory.
    
    Returns
    -------
    mean : array, shape (n_timepoints,)
        Mean across samples at each time point.
    sem : array, shape (n_timepoints,)
        Standard error of the mean at each time point.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    sem = X.std(axis=0, ddof=1) / np.sqrt(max(1, X.shape[0]))
    return mean, sem


def _mw_p(a, b):
    try:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def analyze_minima_vs_secondary(
    primary_data,
    secondary_data,
    threshold=0.20,
    window=7,
    normalization="per_trace_percentile",
    percentile=95,
    min_prominence=None,
    n_random=500,
    n_boot=200,
    random_state=0,
    smooth_window=1,
    trajectory_colors=None,
):
    """
    Reproduce the “Analysis of minima signals” workflow.

    Parameters
    ----------
    primary_data : (n_traces, n_time)
    secondary_data : (n_traces, n_time)
    threshold : float
        Normalized intensity threshold to accept minima (default 0.20).
    window : int
        Frames before/after the minimum to extract (±window, inclusive center).
    normalization : {"per_trace_percentile","per_trace_max"}
    percentile : int
        Used if normalization="per_trace_percentile".
    min_prominence : float or None
        If None, adapt per trace based on dynamic range (10% of [P95-P5], floored at 0.01).
    n_random : int
        Number of random centers for control analysis.
    n_boot : int
        Number of bootstrap resamples of observed windows.
    random_state : int
    smooth_window : int
        Moving-average window (frames). If >1, applies uniform_filter1d along time.

    Returns
    -------
    results : dict  (always contains an "aligned" block)
    """
    rng = np.random.default_rng(random_state)

    P = np.asarray(primary_data, dtype=float)
    S = np.asarray(secondary_data, dtype=float)
    if P.shape != S.shape:
        raise ValueError(f"primary_data and secondary_data must have identical shape, got {P.shape} vs {S.shape}")
    n_traces, n_time = P.shape

    # optional smoothing
    if smooth_window > 1:
        P = uniform_filter1d(P, size=smooth_window, axis=1, mode='nearest')
        S = uniform_filter1d(S, size=smooth_window, axis=1, mode='nearest')

    # normalization
    Pn = _normalize_traces(P, method=normalization, percentile=percentile)
    Sn = _normalize_traces(S, method=normalization, percentile=percentile)

    seg_len = 2 * window + 1
    t = np.arange(seg_len) - window
    accepted = []
    # Tunable knobs (safe defaults)
    k_sigma      = 4    # prominence must be >= k_sigma * noise (noise from first differences, robust MAD)
    min_distance = 10      # minimum separation (frames) between minima within a trace (also passed to find_peaks)
    min_width    = 4.0    # minimum half-prominence width (in frames) of the valley
    max_width    = 3*window   # set e.g. 2*window to reject very broad trends
    z_min        = 1    # minimum z-score for local drop (baseline - valley) / SEM(flanks)

    for i in range(n_traces):
        y = Pn[i].astype(float)

        # Skip traces that are all-NaN or constant after normalization
        if not np.isfinite(y).any():
            continue
        if np.nanmax(y) - np.nanmin(y) < 1e-9:
            continue

        # Optional smoothing for detection robustness (doesn't touch Pn)
        #yd = uniform_filter1d(y, size=max(1, smooth_window), mode="nearest") if smooth_window > 1 else y
        yd = y  # use unsmoothed for detection to preserve features
        # NaN-safety: replace isolated NaNs by nearest finite values
        if np.isnan(yd).any():
            finite = np.where(np.isfinite(yd))[0]
            if finite.size == 0:
                continue
            yd = np.interp(np.arange(len(yd)), finite, yd[finite])

        # Adaptive prominence: user-provided OR 10% dynamic range, floored; then max with noise floor
        prom_i = min_prominence
        if prom_i is None:
            dyn = np.percentile(yd, 95) - np.percentile(yd, 5)
            prom_i = max(0.01, 0.10 * float(dyn))

        # Robust noise from first differences (MAD)
        diffs = np.diff(yd)
        if diffs.size:
            mad = np.median(np.abs(diffs - np.median(diffs)))
            noise_sigma = 1.4826 * mad / np.sqrt(2.0)
        else:
            noise_sigma = 0.0

        prom_thresh = max(prom_i, k_sigma * noise_sigma)

        # Candidate valleys via peaks on -yd
        peaks, props = find_peaks(-yd, prominence=prom_thresh, distance=min_distance)
        if peaks.size == 0:
            continue

        # Valley width at half prominence
        widths, _, _, _ = peak_widths(-yd, peaks, rel_height=0.5)

        for k, idx in enumerate(peaks):
            # Enforce usable window and absolute threshold on the (possibly smoothed) signal
            if not (idx - window >= 0 and idx + window < n_time):
                continue
            if yd[idx] > threshold:
                continue

            # Width constraints (reject needle-thin or ultra-broad dips)
            w = widths[k] if k < len(widths) else 1.0
            if w < min_width:
                continue
            if (max_width is not None) and (w > max_width):
                continue

            # Positive curvature (2nd derivative) at the minimum
            if 0 < idx < len(yd) - 1:
                curv = yd[idx - 1] - 2 * yd[idx] + yd[idx + 1]
            else:
                curv = 0.0
            if curv <= 0:
                continue

            # Local baseline (mean of flanks) and z-score of the drop
            L = max(1, window)
            left  = yd[max(0, idx - L): idx]
            right = yd[idx + 1: min(len(yd), idx + 1 + L)]
            flanks = np.concatenate([left, right]) if (left.size + right.size) > 0 else np.array([yd[idx]])

            # Guard against pathological flanks
            if not np.isfinite(flanks).any():
                continue
            base = float(np.mean(flanks))
            sem  = float(np.std(flanks, ddof=1 if flanks.size > 1 else 0) / np.sqrt(max(1, flanks.size)))
            delta = base - yd[idx]
            z = delta / (sem + 1e-12)

            if z < z_min:
                continue

            accepted.append((i, int(idx)))

    # print the number of accepted minima
    print(f'Number of accepted minima: {len(accepted)}')


    # if nothing accepted: return safe, structured result
    if len(accepted) == 0:
        empty = np.full(seg_len, np.nan)
        nan_ci = np.column_stack([empty, empty])
        return {
            "normalization": {"method": normalization, "percentile": percentile},
            "selection": {"n_traces": n_traces, "n_time": n_time, "n_minima_found": 0, "n_minima_kept": 0},
            "aligned": {
                "primary_mean": empty, "primary_sem": empty, "primary_ci95": nan_ci,
                "secondary_mean": empty, "secondary_sem": empty, "secondary_ci95": nan_ci,
                "t": t,
            },
            "fits": {
                "primary": {"baseline": np.nan, "amp": np.nan, "mu": np.nan, "sigma": np.nan, "ymin": np.nan, "magnitude": np.nan},
                "secondary": {"baseline": np.nan, "amp": np.nan, "mu": np.nan, "sigma": np.nan, "ymin": np.nan, "magnitude": np.nan},
                "delay_secondary_minus_primary": np.nan,
            },
            "bootstrap": {
                "primary": {"mu": np.array([]), "magnitude": np.array([])},
                "secondary": {"mu": np.array([]), "magnitude": np.array([])},
                "delay": np.array([]),
            },
            "control": {
                "primary": {"mu": np.array([]), "magnitude": np.array([])},
                "secondary": {"mu": np.array([]), "magnitude": np.array([])},
                "delay": np.array([]),
            },
            "stats": {
                "p_magnitude_vs_control_primary": np.nan,
                "p_magnitude_vs_control_secondary": np.nan,
                "p_mu_vs_zero_primary": np.nan,
                "p_mu_vs_zero_secondary": np.nan,
                "p_delay_vs_zero": np.nan,
            },
            "indices": {"accepted_minima": []},
            "message": "No minima passed the threshold/window criteria. Try increasing smooth_window, raising threshold, or lowering prominence.",
        }

    # build aligned windows around each accepted center
    P_windows = np.vstack([Pn[i, j - window: j + window + 1] for (i, j) in accepted])
    S_windows = np.vstack([Sn[i, j - window: j + window + 1] for (i, j) in accepted])

    

    P_mean, P_sem = _mean_sem(P_windows)
    S_mean, S_sem = _mean_sem(S_windows)

    # gaussian fits to averaged traces
    fit_P = _fit_gaussian(P_mean, t)
    fit_S = _fit_gaussian(S_mean, t)
    delay_hat = float(fit_S["mu"] - fit_P["mu"])

    # bootstrap on observed windows
    def _bootstrap_fit(X_windows, nboot):
        mu_list, mag_list = [], []
        m = X_windows.shape[0]
        for _ in range(nboot):
            idx = rng.integers(0, m, size=m)  # resample windows
            mean_boot = X_windows[idx].mean(axis=0)
            fit = _fit_gaussian(mean_boot, t)
            mu_list.append(fit["mu"])
            mag_list.append(fit["magnitude"])
        return np.array(mu_list), np.array(mag_list)

    muP_b, magP_b = _bootstrap_fit(P_windows, n_boot)
    muS_b, magS_b = _bootstrap_fit(S_windows, n_boot)
    delay_b = muS_b - muP_b

    # control: random centers
    def _control_distrib(Xn, nctrl):
        mu_list, mag_list = [], []
        for _ in range(nctrl):
            i = rng.integers(0, n_traces)
            j = rng.integers(window, n_time - window)
            w = Xn[i, j - window: j + window + 1][None, :]
            mean_ctrl = w.mean(axis=0)
            fit = _fit_gaussian(mean_ctrl, t)
            mu_list.append(fit["mu"])
            mag_list.append(fit["magnitude"])
        return np.array(mu_list), np.array(mag_list)

    muP_c, magP_c = _control_distrib(Pn, n_random)
    muS_c, magS_c = _control_distrib(Sn, n_random)
    delay_c = muS_c - muP_c

    # stats
    p_mag_P = _mw_p(magP_b, magP_c)
    p_mag_S = _mw_p(magS_b, magS_c)
    p_mu_P = _mw_p(muP_b, np.zeros_like(muP_b))
    p_mu_S = _mw_p(muS_b, np.zeros_like(muS_b))
    p_delay = _mw_p(delay_b, np.zeros_like(delay_b))

    results = {
        "normalization": {"method": normalization, "percentile": percentile},
        "selection": {
            "n_traces": n_traces,
            "n_time": n_time,
            "n_minima_found": int(len(accepted)),
            "n_minima_kept": int(len(accepted)),
        },
        "aligned": {
            "primary_mean": P_mean,
            "primary_sem": P_sem,
            #"primary_ci95": P_ci,
            "secondary_mean": S_mean,
            "secondary_sem": S_sem,
            #"secondary_ci95": S_ci,
            "t": t,
        },
        "fits": {
            "primary": fit_P,
            "secondary": fit_S,
            "delay_secondary_minus_primary": delay_hat,
        },
        "bootstrap": {
            "primary": {"mu": muP_b, "magnitude": magP_b},
            "secondary": {"mu": muS_b, "magnitude": magS_b},
            "delay": delay_b,
        },
        "control": {
            "primary": {"mu": muP_c, "magnitude": magP_c},
            "secondary": {"mu": muS_c, "magnitude": magS_c},
            "delay": delay_c,
        },
        "stats": {
            "p_magnitude_vs_control_primary": p_mag_P,
            "p_magnitude_vs_control_secondary": p_mag_S,
            "p_mu_vs_zero_primary": p_mu_P,
            "p_mu_vs_zero_secondary": p_mu_S,
            "p_delay_vs_zero": p_delay,
        },
        "indices": {"accepted_minima": accepted},
    }
    return results



def plot_aligned_with_shared_normalization(
    results, 
    primary_label="Primary", 
    secondary_label="Secondary", 
    title="Aligned Traces Around Minima", 
    step_size_in_sec=5.0, 
    figsize=(10, 6), 
    show_fits=True,
    normalization_mode='shared',  # NEW: 'shared', 'independent', or 'none'
    show_delay=True,
    trajectory_colors=None,
    results_folder=None,
    save_plot=False,
    dataset_name='minima_alignment',
):
    """
    Plot aligned minima with flexible normalization modes.
    
    Parameters
    ----------
    results : dict
        Results from analyze_minima_vs_secondary
    primary_label : str
        Label for primary signal
    secondary_label : str
        Label for secondary signal
    title : str
        Plot title
    step_size_in_sec : float
        Time step in seconds
    figsize : tuple
        Figure size
    show_fits : bool
        Whether to show Gaussian fits
    normalization_mode : str
        Normalization strategy:
        - 'shared': Use combined min/max from both signals (preserves delays)
        - 'independent': Normalize each signal separately [0, 1] (exaggerates differences)
        - 'none': No normalization (raw units)
    results_folder : Path, optional
        Folder to save plots (if save_plot=True)
    save_plot : bool
        Whether to save the plot to results_folder
    dataset_name : str
        Name prefix for saved files
    
    Returns
    -------
    None (displays plot)
    """
    
    t = results["aligned"]["t"]
    P_mean = results["aligned"]["primary_mean"]
    P_sem = results["aligned"]["primary_sem"]
    S_mean = results["aligned"]["secondary_mean"]
    S_sem = results["aligned"]["secondary_sem"]

    # Fits
    fit_P = results["fits"]["primary"]
    fit_S = results["fits"]["secondary"]
    delay = results["fits"]["delay_secondary_minus_primary"]

    if np.isnan(P_mean).all():
        print(results.get("message", "No aligned minima available."))
        return

    # ============================================================
    # FLEXIBLE NORMALIZATION
    # ============================================================
    if normalization_mode == 'shared':
        # ===== SHARED NORMALIZATION (preserves delays) =====
        combined = np.concatenate([P_mean, S_mean])
        c_min = np.nanmin(combined)
        c_max = np.nanmax(combined)
        
        if c_max - c_min < 1e-9:
            P_mean_norm = S_mean_norm = np.full_like(P_mean, 0.5)
            P_sem_norm = S_sem_norm = np.zeros_like(P_sem)
            fit_P_vals_norm = fit_S_vals_norm = np.full_like(t, 0.5, dtype=float)
        else:
            # Both signals normalized with SAME scale
            P_mean_norm = (P_mean - c_min) / (c_max - c_min)
            S_mean_norm = (S_mean - c_min) / (c_max - c_min)
            
            # Scale SEM proportionally
            P_sem_norm = P_sem / (c_max - c_min)
            S_sem_norm = S_sem / (c_max - c_min)
            
            # Normalize fits
            fit_P_vals = _gauss(t, fit_P["baseline"], fit_P["amp"], fit_P["mu"], fit_P["sigma"])
            fit_S_vals = _gauss(t, fit_S["baseline"], fit_S["amp"], fit_S["mu"], fit_S["sigma"])
            fit_P_vals_norm = (fit_P_vals - c_min) / (c_max - c_min)
            fit_S_vals_norm = (fit_S_vals - c_min) / (c_max - c_min)
        
        ylabel = 'Normalized Intensity\n(Shared Scale)'
        
    elif normalization_mode == 'independent':
        # ===== INDEPENDENT NORMALIZATION (each signal [0, 1]) =====
        P_min, P_max = np.nanmin(P_mean), np.nanmax(P_mean)
        S_min, S_max = np.nanmin(S_mean), np.nanmax(S_mean)
        
        # Primary normalization
        if P_max - P_min < 1e-9:
            P_mean_norm = np.full_like(P_mean, 0.5)
            P_sem_norm = np.zeros_like(P_sem)
        else:
            P_mean_norm = (P_mean - P_min) / (P_max - P_min)
            P_sem_norm = P_sem / (P_max - P_min)
        
        # Secondary normalization
        if S_max - S_min < 1e-9:
            S_mean_norm = np.full_like(S_mean, 0.5)
            S_sem_norm = np.zeros_like(S_sem)
        else:
            S_mean_norm = (S_mean - S_min) / (S_max - S_min)
            S_sem_norm = S_sem / (S_max - S_min)
        
        # Normalize fits independently
        fit_P_vals = _gauss(t, fit_P["baseline"], fit_P["amp"], fit_P["mu"], fit_P["sigma"])
        fit_S_vals = _gauss(t, fit_S["baseline"], fit_S["amp"], fit_S["mu"], fit_S["sigma"])
        
        if P_max - P_min > 1e-9:
            fit_P_vals_norm = (fit_P_vals - P_min) / (P_max - P_min)
        else:
            fit_P_vals_norm = np.full_like(t, 0.5, dtype=float)
            
        if S_max - S_min > 1e-9:
            fit_S_vals_norm = (fit_S_vals - S_min) / (S_max - S_min)
        else:
            fit_S_vals_norm = np.full_like(t, 0.5, dtype=float)
        
        ylabel = 'Normalized Intensity\n(Independent Scale)'
        print("⚠ WARNING: Independent normalization may obscure true amplitude differences!")
        
    elif normalization_mode == 'none':
        # ===== NO NORMALIZATION (raw units) =====
        P_mean_norm = P_mean
        S_mean_norm = S_mean
        P_sem_norm = P_sem
        S_sem_norm = S_sem
        
        # Raw fits
        fit_P_vals_norm = _gauss(t, fit_P["baseline"], fit_P["amp"], fit_P["mu"], fit_P["sigma"])
        fit_S_vals_norm = _gauss(t, fit_S["baseline"], fit_S["amp"], fit_S["mu"], fit_S["sigma"])
        
        ylabel = 'Intensity (a.u.)'
        
    else:
        raise ValueError(f"normalization_mode must be 'shared', 'independent', or 'none', got '{normalization_mode}'")

    # ============================================================
    # PLOTTING (unchanged from here)
    # ============================================================
    # Time axis in seconds
    t_sec = t * step_size_in_sec
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot Primary signal with SEM bands
    ax.plot(t_sec, P_mean_norm, 'o-', label=f'{primary_label}', 
            color=trajectory_colors[0], linewidth=2.5, markersize=5, zorder=3)
    ax.fill_between(t_sec, 
                    P_mean_norm - P_sem_norm, 
                    P_mean_norm + P_sem_norm, 
                    color=trajectory_colors[0], alpha=0.2, zorder=1)
    
    # Plot Secondary signal with SEM bands
    ax.plot(t_sec, S_mean_norm, 'o-', label=f'{secondary_label}', 
            color=trajectory_colors[1], linewidth=2.5, markersize=5, zorder=3)
    
    ax.fill_between(t_sec, 
                    S_mean_norm - S_sem_norm, 
                    S_mean_norm + S_sem_norm, 
                    color=trajectory_colors[1], alpha=0.2, zorder=1)
    
    # Plot Gaussian fits if requested
    if show_fits:
        ax.plot(t_sec, fit_P_vals_norm, '--', 
                label=f'{primary_label} Fit', color=trajectory_colors[0], linewidth=2, alpha=0.8, zorder=2)
        ax.plot(t_sec, fit_S_vals_norm, '--', 
                label=f'{secondary_label} Fit', color=trajectory_colors[1], linewidth=2, alpha=0.8, zorder=2)
    
    # Plot vertical lines at fitted minimum positions
    mu_P_sec = fit_P["mu"] * step_size_in_sec
    mu_S_sec = fit_S["mu"] * step_size_in_sec
    if show_delay:
        ax.axvline(mu_P_sec, color='forestgreen', linestyle=':', linewidth=2.5, alpha=0.9,
                label=f'{primary_label} min: {mu_P_sec:.1f}s', zorder=4)
        ax.axvline(mu_S_sec, color='indigo', linestyle=':', linewidth=2.5, alpha=0.9,
                label=f'{secondary_label} min: {mu_S_sec:.1f}s', zorder=4)
    
        # Annotate the delay with arrow (only if meaningful and shared/none mode)
        if abs(delay) > 0.5 and normalization_mode in ['shared', 'none']:
            # Find a good y-position for the arrow
            y_range = ax.get_ylim()
            y_arrow = y_range[0] + 0.8 * (y_range[1] - y_range[0])
            
            # Draw double-headed arrow between the two minima
            ax.annotate('', xy=(mu_S_sec, y_arrow), xytext=(mu_P_sec, y_arrow),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=3, shrinkA=0, shrinkB=0))
            
            # Label the delay
            ax.text((mu_P_sec + mu_S_sec) / 2, y_arrow + 0.02 * (y_range[1] - y_range[0]), 
                    f'Delay = {delay * step_size_in_sec:.1f}s',
                    ha='center', va='bottom', fontsize=12, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
        
        # Reference line at detection center
        ax.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.6, 
                label='Detection Center', zorder=2)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
    #ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    #ax.set_title(f"{title}\n(Normalization: {normalization_mode.capitalize()})", 
    #             fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='top left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot and results_folder is not None:
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        base_name = f'{dataset_name}_{normalization_mode}_normalization'
        plt.savefig(results_folder / f'{base_name}.png', dpi=150, bbox_inches='tight')
        plt.savefig(results_folder / f'{base_name}.svg', bbox_inches='tight')
        print(f"Saved minima alignment plot to: {results_folder / base_name}.[png|svg]")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"DELAY ANALYSIS SUMMARY (Normalization: {normalization_mode.upper()})")
    print(f"{'='*70}")
    print(f"{primary_label} minimum position: {fit_P['mu']:6.2f} frames = {mu_P_sec:7.2f} seconds")
    print(f"{secondary_label} minimum position: {fit_S['mu']:6.2f} frames = {mu_S_sec:7.2f} seconds")
    print(f"{'-'*70}")
    print(f"Delay ({secondary_label} - {primary_label}): {delay:6.2f} frames = {delay * step_size_in_sec:7.2f} seconds")
    
    # Bootstrap confidence interval
    delay_boot = results["bootstrap"]["delay"]
    ci_lower, ci_upper = np.percentile(delay_boot, [2.5, 97.5])
    print(f"\nBootstrap 95% CI: [{ci_lower:6.2f}, {ci_upper:6.2f}] frames")
    print(f"                  [{ci_lower * step_size_in_sec:7.2f}, {ci_upper * step_size_in_sec:7.2f}] seconds")
    
    # Statistical significance
    p_delay = results["stats"]["p_delay_vs_zero"]
    print(f"\nMann-Whitney p-value (delay vs zero): p = {p_delay:.4e}")
    
    # Interpretation
    print(f"\n{'='*70}")
    if p_delay < 0.05 and ci_lower > 0:
        print(f"CONCLUSION: Significant POSITIVE delay detected")
        print(f"  {secondary_label} lags behind {primary_label} by ~{delay * step_size_in_sec:.1f} seconds")
    elif p_delay < 0.05 and ci_upper < 0:
        print(f"CONCLUSION: Significant NEGATIVE delay detected")
        print(f"  {secondary_label} leads {primary_label} by ~{-delay * step_size_in_sec:.1f} seconds")
    elif ci_lower <= 0 <= ci_upper:
        print(f"CONCLUSION: No conclusive delay")
        print(f"  Confidence interval includes zero - high uncertainty")
    else:
        print(f"CONCLUSION: Ambiguous result")
    print(f"{'='*70}\n")




def analyze_dual_channel_time_courses(
    dataset,
    data_folder: Path = None,  # Made optional for simulation mode
    results_folder: Path = None,  # Made optional for simulation mode
    selected_field='spot_int_ch_',
    primary_channel=1,
    secondary_channel=0,
    step_size_in_sec=5,
    min_percentage_data_in_trajectory=0.2,
    max_missing_frames=5,
    min_snr=1,
    smooth_window=1,
    max_trajectory_length_percentile=None,
    control_spots_mode=False,
    downsample=False,
    # Time course plotting parameters
    show_time_courses=5,
    trajectory_smooth_window=9,
    trajectory_figsize=(10, 2),
    trajectory_colors=None,
    trajectory_labels=None,
    normalize_trajectories=True,
    # Minima analysis parameters
    show_minima_analysis=True,
    minima_threshold=0.3,
    minima_window=12,
    normalization='min_max',
    percentile=95,
    min_prominence=None,
    n_random=500,
    n_boot=200,
    random_state=0,
    minima_smooth_window=5,
    # Minima plot parameters
    primary_label="HA",
    secondary_label="GFP",
    minima_plot_title="Aligned Traces Around Minima",
    minima_figsize=(10, 5),
    show_fits=True,
    minima_normalization_mode='shared',
    # General parameters
    save_results=True,
    verbose=True,
    # Simulation mode parameters
    simulation_mode=False,
    SSA_data_1=None,
    SSA_data_2=None,
    show_delay=True,
    detect_maxima=False,
):
    """
    Analyze dual-channel time courses with optional trajectory plotting and minima analysis.
    Supports both experimental and simulated data.
    
    Parameters:
    -----------
    dataset : str
        Dataset identifier (e.g., 'cof')
    data_folder : Path or None
        Folder containing data (required if simulation_mode=False)
    results_folder : Path or None
        Folder to save results (optional if simulation_mode=True)
    selected_field : str
        Base field name for intensity (default: 'spot_int_ch_')
    primary_channel : int
        Channel index for primary data (default: 1 for HA_TAG)
    secondary_channel : int
        Channel index for secondary data (default: 0 for GFP_TAG)
    step_size_in_sec : float
        Time interval between frames in seconds
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data points per trajectory
    max_missing_frames : int
        Maximum consecutive missing frames allowed
    min_snr : float
        Minimum signal-to-noise ratio threshold
    smooth_window : int
        Smoothing window size for data loading
    control_spots_mode : bool
        Whether analyzing control spots
    downsample : bool
        Whether data is downsampled
    
    Time Course Plotting:
    --------------------
    show_time_courses : int
        Number of individual trajectories to plot (0 = skip plotting)
    trajectory_smooth_window : int
        Smoothing window for trajectory plots
    trajectory_figsize : tuple
        Figure size for each trajectory plot
    trajectory_colors : list or None
        Colors for [primary, secondary] channels
    trajectory_labels : list or None
        Labels for [primary, secondary] channels
    normalize_trajectories : bool
        Whether to normalize trajectories for plotting
    
    Minima Analysis:
    ---------------
    show_minima_analysis : bool
        Whether to perform and plot minima analysis
    minima_threshold : float
        Threshold for minima detection
    minima_window : int
        Window size for minima alignment
    normalization : str
        Normalization method ('min_max', 'min_5_max_95', etc.)
    percentile : int
        Percentile for normalization
    min_prominence : float or None
        Minimum prominence for peak detection
    n_random : int
        Number of random samples for bootstrapping
    n_boot : int
        Number of bootstrap iterations
    random_state : int
        Random seed for reproducibility
    minima_smooth_window : int
        Smoothing window for minima analysis
    
    Minima Plotting:
    ---------------
    primary_label : str
        Label for primary channel in minima plot
    secondary_label : str
        Label for secondary channel in minima plot
    minima_plot_title : str
        Title for minima plot
    minima_figsize : tuple
        Figure size for minima plot
    show_fits : bool
        Whether to show fitted curves in minima plot
    minima_normalization_mode : str
        Normalization mode for minima plot ('shared', 'independent', 'none')
    
    Simulation Mode:
    ---------------
    simulation_mode : bool
        If True, uses SSA_data_1 and SSA_data_2 instead of loading from files
    SSA_data_1 : array-like or None
        Primary channel simulated data (n_trajectories, n_timepoints)
        Required if simulation_mode=True
    SSA_data_2 : array-like or None
        Secondary channel simulated data (n_trajectories, n_timepoints)
        Required if simulation_mode=True
    
    General:
    -------
    save_results : bool
        Whether to save analysis results
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    dict:
        Contains:
            - 'primary_data': Primary channel data array
            - 'secondary_data': Secondary channel data array
            - 'total_number_of_cells': Number of cells processed
            - 'time_array': Time array for trajectories
            - 'minima_results': Results from minima analysis
            - 'dataset': Dataset identifier
            - 'plot_name': Name used for saving files
            - 'data_source': 'simulation' or 'experimental'
    """
    
    # ===== VALIDATION =====
    if simulation_mode:
        if SSA_data_1 is None or SSA_data_2 is None:
            raise ValueError("simulation_mode=True requires both SSA_data_1 and SSA_data_2 to be provided")
        
        SSA_data_1 = np.asarray(SSA_data_1, dtype=float)
        SSA_data_2 = np.asarray(SSA_data_2, dtype=float)
        
        if SSA_data_1.ndim != 2:
            raise ValueError(f"SSA_data_1 must be 2D, got shape {SSA_data_1.shape}")
        if SSA_data_2.ndim != 2:
            raise ValueError(f"SSA_data_2 must be 2D, got shape {SSA_data_2.shape}")
        if SSA_data_1.shape != SSA_data_2.shape:
            raise ValueError(f"SSA_data_1 and SSA_data_2 must have same shape")
        
        if verbose:
            print(f"SIMULATION MODE: Using provided SSA data")
            print(f"  SSA data shape: {SSA_data_1.shape}")
    else:
        if data_folder is None:
            raise ValueError("data_folder is required when simulation_mode=False")
    
    # Set default colors and labels
    if trajectory_colors is None:
        trajectory_colors = ['forestgreen', 'indigo']
    
    if trajectory_labels is None:
        ch_primary_name = 'HA_TAG' if primary_channel == 1 else 'GFP_TAG'
        ch_secondary_name = 'GFP_TAG' if secondary_channel == 0 else 'HA_TAG'
        trajectory_labels = [
            f'{ch_primary_name} (ch_{primary_channel})',
            f'{ch_secondary_name} (ch_{secondary_channel})'
        ]
    
    if verbose:
        print(f"{'='*60}")
        print(f"Analyzing Dual-Channel Time Courses")
        print(f"Data source: {'SIMULATION' if simulation_mode else 'EXPERIMENTAL'}")
        print(f"Dataset: {dataset}")
        print(f"Primary channel: {primary_channel} ({trajectory_labels[0]})")
        print(f"Secondary channel: {secondary_channel} ({trajectory_labels[1]})")
        print(f"{'='*60}\n")
    
    # ===== DATA LOADING BRANCH =====
    if simulation_mode:
        # Use simulated data directly
        primary_data = SSA_data_1.copy()
        secondary_data = SSA_data_2.copy()
        total_number_of_cells = 1  # Simulations don't have "cells"
        plot_name = f'{dataset}_simulation'
        
        if verbose:
            print("Using simulated data:")
            print(f"  Primary shape: {primary_data.shape}")
            print(f"  Secondary shape: {secondary_data.shape}")
            print(f"  Total trajectories: {primary_data.shape[0]}\n")
    else:
        # Get dataset configuration
        folder_with_files, plot_name, dataframe_prefix = dataset_selection(
            dataset, data_folder, control_spots_mode, downsample
        )
        
        # Load synchronized dual-channel data
        if verbose:
            print("Loading synchronized dual-channel tracking data...")
        
        primary_data, secondary_data, total_number_of_cells = load_dual_channel_tracking_data(
            folder_with_files,
            base_field=selected_field,
            primary_channel=primary_channel,
            secondary_channel=secondary_channel,
            min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
            dataframe_prefix=dataframe_prefix,
            min_snr=min_snr,
            max_missing_frames=max_missing_frames,
            verbose=verbose,
            smooth_window=1,  # No extra smoothing during loading
            max_trajectory_length_percentile=max_trajectory_length_percentile,
        )
        
        if verbose:
            print(f"\nData loaded:")
            print(f"  Primary (ch_{primary_channel}): {primary_data.shape}")
            print(f"  Secondary (ch_{secondary_channel}): {secondary_data.shape}")
            print(f"  Total cells: {total_number_of_cells}")
            print(f"  Total trajectories: {primary_data.shape[0]}\n")
    
    # ===== COMMON PROCESSING (both experimental and simulation) =====
    
    # Create time array
    time_array = np.arange(0, primary_data.shape[1] * step_size_in_sec, step_size_in_sec)
    
    # Plot individual time courses if requested
    if show_time_courses > 0:
        if verbose:
            print(f"Plotting {show_time_courses} individual trajectories...")
        
        n_trajectories_to_plot = min(show_time_courses, primary_data.shape[0])
        
        for i in range(n_trajectories_to_plot):
            try:
                plot_dual_signal_trajectories(
                    primary_data, 
                    secondary_data,
                    time_array, 
                    trajectory_index=i,
                    colors=trajectory_colors,
                    labels=trajectory_labels,
                    smooth_window=trajectory_smooth_window,
                    figsize=trajectory_figsize,
                    verbose=False,
                    normalize=normalize_trajectories
                )
            except Exception as e:
                if verbose:
                    print(f"  Error plotting trajectory {i}: {e}")
        
        if verbose:
            print(f"Plotted {n_trajectories_to_plot} trajectories\n")
    else:
        if verbose:
            print("Skipping individual trajectory plots (show_time_courses=0)\n")
    
    if detect_maxima:
        if verbose:
            print("Detecting maxima instead of minima...")
        # ✅ Create inverted COPIES to avoid modifying originals
        primary_data = -primary_data.copy()
        secondary_data = -secondary_data.copy()

    # Perform minima analysis if requested
    minima_results = None
    if show_minima_analysis:
        if verbose:
            print("Performing minima analysis...")
        try:
            minima_results = analyze_minima_vs_secondary(
                primary_data=primary_data,
                secondary_data=secondary_data,
                threshold=minima_threshold,
                window=minima_window,
                normalization=normalization,
                percentile=percentile,
                min_prominence=min_prominence,
                n_random=n_random,
                n_boot=n_boot,
                random_state=random_state,
                smooth_window=minima_smooth_window,
                
            )
            
            if verbose:
                print("Minima analysis complete\n")
            
            # Plot aligned traces
            if verbose:
                print("Plotting aligned traces around minima...")
            
            plot_aligned_with_shared_normalization(
                minima_results,
                primary_label=primary_label,
                secondary_label=secondary_label,
                title=f"{minima_plot_title}\n({'Simulation' if simulation_mode else 'Experimental'})",
                step_size_in_sec=step_size_in_sec,
                figsize=minima_figsize,
                show_fits=show_fits,
                normalization_mode=minima_normalization_mode,
                show_delay=show_delay,
                trajectory_colors = trajectory_colors,
            )
            
            if verbose:
                print("Minima plot generated\n")
                
        except Exception as e:
            if verbose:
                print(f"Error in minima analysis: {e}\n")
            minima_results = None
    else:
        if verbose:
            print("Skipping minima analysis (show_minima_analysis=False)\n")
    
    # ===== SAVE RESULTS =====
    if save_results and minima_results is not None and results_folder is not None:
        suffix = '_simulation' if simulation_mode else ''
        results_file = results_folder / f'minima_analysis_{plot_name}_ch{primary_channel}_vs_ch{secondary_channel}{suffix}.npz'
        
        # Save key results
        np.savez(
            results_file,
            primary_aligned_mean=minima_results['aligned']['primary_mean'],
            secondary_aligned_mean=minima_results['aligned']['secondary_mean'],
            primary_aligned_sem=minima_results['aligned']['primary_sem'],
            secondary_aligned_sem=minima_results['aligned']['secondary_sem'],
            time_lags=minima_results['aligned']['t'],
            delay=minima_results['fits']['delay_secondary_minus_primary'],
            dataset=dataset,
            data_source='simulation' if simulation_mode else 'experimental'
        )
        
        if verbose:
            print(f"Results saved to: {results_file.name}\n")
    elif save_results and results_folder is None:
        if verbose:
            print("Warning: save_results=True but results_folder=None, skipping save\n")
    
    # ===== SUMMARY =====
    if verbose:
        print(f"{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Data source: {'Simulation' if simulation_mode else 'Experimental'}")
        print(f"Trajectories analyzed: {primary_data.shape[0]}")
        print(f"Time courses plotted: {show_time_courses if show_time_courses > 0 else 0}")
        print(f"Minima analysis: {'Yes' if show_minima_analysis and minima_results else 'No'}")
        print(f"{'='*60}\n")
    
    # ===== RETURN RESULTS =====
    return {
        'primary_data': primary_data,
        'secondary_data': secondary_data,
        'total_number_of_cells': total_number_of_cells,
        'time_array': time_array,
        'minima_results': minima_results,
        'dataset': dataset,
        'plot_name': plot_name,
        'primary_channel': primary_channel,
        'secondary_channel': secondary_channel,
        'data_source': 'simulation' if simulation_mode else 'experimental'
    }


def plot_dual_signal_trajectories(
    primary_data,
    secondary_data,
    time_array,
    trajectory_index=0,
    colors=None,
    labels=None,
    smooth_window=1,
    figsize=(10, 2),
    normalize=False,
    verbose=True,
    results_folder=None,
    save_plot=False,
    dataset_name='trajectories',
):
    """
    Plot individual dual-channel trajectories.
    
    Parameters
    ----------
    primary_data : ndarray
        Primary channel data (n_trajectories, n_timepoints)
    secondary_data : ndarray
        Secondary channel data (n_trajectories, n_timepoints)
    time_array : ndarray
        Time axis in seconds
    trajectory_index : int
        Index of trajectory to plot
    colors : list of 2 colors, optional
        Colors for [primary, secondary] channels
    labels : list of 2 strings, optional
        Labels for [primary, secondary] channels
    smooth_window : int
        Smoothing window size (1 = no smoothing)
    figsize : tuple
        Figure size (width, height)
    normalize : bool
        Whether to normalize each channel to [0, 1]
    verbose : bool
        Print progress messages
    results_folder : Path, optional
        Folder to save plots (if save_plot=True)
    save_plot : bool
        Whether to save the plot to results_folder
    dataset_name : str
        Name prefix for saved files
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Default colors and labels
    if colors is None:
        colors = ['forestgreen', 'indigo']
    if labels is None:
        labels = ['Primary (ch_1)', 'Secondary (ch_0)']
    
    # Validate trajectory index
    n_trajectories = primary_data.shape[0]
    if trajectory_index >= n_trajectories:
        raise ValueError(f"trajectory_index {trajectory_index} >= n_trajectories {n_trajectories}")
    
    # Extract single trajectory
    primary_trace = primary_data[trajectory_index, :].copy()
    secondary_trace = secondary_data[trajectory_index, :].copy()
    
    # Apply smoothing if requested
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d

        def _smooth_preserve_nans(trace, window):
            """Interpolate *only within* the valid range, smooth, then restore
            NaNs outside [first_valid, last_valid] so trailing frames stay blank."""
            finite_idx = np.where(np.isfinite(trace))[0]
            if len(finite_idx) < 2:
                return trace  # nothing to smooth
            first, last = finite_idx[0], finite_idx[-1]
            # Interpolate internal NaNs only (no extrapolation beyond last valid frame)
            all_idx = np.arange(first, last + 1)
            interp_vals = np.interp(all_idx, finite_idx, trace[finite_idx])
            smoothed = uniform_filter1d(interp_vals, size=window, mode='nearest')
            # Write back into a NaN array — positions outside [first, last] stay NaN
            out = np.full_like(trace, np.nan)
            out[first:last + 1] = smoothed
            return out

        primary_trace = _smooth_preserve_nans(primary_trace, smooth_window)
        secondary_trace = _smooth_preserve_nans(secondary_trace, smooth_window)

    # Normalize if requested
    if normalize:
        p_min, p_max = np.nanmin(primary_trace), np.nanmax(primary_trace)
        s_min, s_max = np.nanmin(secondary_trace), np.nanmax(secondary_trace)
        
        if p_max - p_min > 1e-9:
            primary_trace = (primary_trace - p_min) / (p_max - p_min)
        if s_max - s_min > 1e-9:
            secondary_trace = (secondary_trace - s_min) / (s_max - s_min)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time_array[:len(primary_trace)], primary_trace, 
            color=colors[0], linewidth=1.5, label=labels[0], alpha=0.9)
    ax.plot(time_array[:len(secondary_trace)], secondary_trace, 
            color=colors[1], linewidth=1.5, label=labels[1], alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)', fontsize=10)
    ax.set_title(f'Trajectory {trajectory_index}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_plot and results_folder is not None:
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        base_name = f'{dataset_name}_trajectory_{trajectory_index}'
        fig.savefig(results_folder / f'{base_name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(results_folder / f'{base_name}.svg', bbox_inches='tight')
        if verbose:
            print(f"Saved trajectory plot to: {results_folder / base_name}.[png|svg]")
    
    plt.show()
    
    return fig, ax



def plot_dual_channel_kymograph(
    dataset='cof',
    data_folder=None,
    results_folder=None,
    selected_field='snr_ch_',
    primary_channel=1,
    secondary_channel=0,
    min_percentage_data_in_trajectory=0.2,
    max_missing_frames=3,
    min_snr=1,
    smooth_window=1,
    max_trajectory_length_percentile=None,
    selected_indices=None,
    orientation="rows=trajectories",
    normalize="per_trace_percentile",
    p_lo=1,
    p_hi=99,
    gamma=None,
    sort_by_data_density=True,
    nan_color=(0.0, 0.0, 0.0),
    channel_colors=None,
    title=None,
    figsize=(12, 6),
    dpi=150,
    save_path=None,
    show=True,
    verbose=True,
    simulation_mode=False,
    SSA_data_1=None,
    SSA_data_2=None,
    shift_data=False,
    use_binarization=False,
    binarization_threshold=0.5,
    max_traces_to_plot=160,
):
    """
    Create a dual-channel kymograph for cotranslational folding analysis.
    
    Compatible with both simulated and experimental data, handles missing timepoints,
    and supports custom color schemes and binarization.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    data_folder : Path
        Folder containing experimental data
    results_folder : Path
        Folder to save results
    selected_field : str
        Base field name (e.g., 'spot_int_ch_')
    primary_channel : int
        Primary channel index (0 or 1)
    secondary_channel : int
        Secondary channel index (0 or 1)
    min_percentage_data_in_trajectory : float
        Minimum percentage of valid data required
    max_missing_frames : int
        Maximum consecutive NaN frames allowed
    min_snr : float
        Minimum SNR threshold
    smooth_window : int
        Smoothing window size
    selected_indices : array-like, optional
        Indices or boolean mask to subset trajectories
    orientation : str
        'rows=trajectories' or 'rows=time'
    normalize : str or list
        Normalization mode for each channel
    p_lo, p_hi : float
        Percentiles for normalization
    gamma : float, optional
        Gamma correction factor (ignored if use_binarization=True)
    sort_by_data_density : bool
        If True, sort rows by number of valid timepoints (most data at top)
    nan_color : tuple or str
        RGB color for missing data (default black)
    channel_colors : list of 2 colors, optional
        Colors for [primary, secondary] channels.
        Each color can be:
        - Single letter: 'r', 'g', 'b', 'm', 'c', 'y', 'k', 'w'
        - Color name: 'red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'gray', 'white', 'black'
        - RGB tuple: (r, g, b) with values in [0, 1]
        - Hex string: '#ff0000'
        Default: None (uses green and magenta from imports.py)
    title : str, optional
        Plot title
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    save_path : Path, optional
        Path to save figure
    show : bool
        Whether to display the plot
    verbose : bool
        Print progress messages
    simulation_mode : bool
        Use simulated data instead of loading from files
    SSA_data_1 : ndarray, optional
        Simulated primary channel data (n_trajectories, n_timepoints)
    SSA_data_2 : ndarray, optional
        Simulated secondary channel data (n_trajectories, n_timepoints)
    shift_data : bool, optional
        If True, shift trajectories to align first valid data point to the left.
        Default: False (preserve original temporal alignment)
    use_binarization : bool, optional
        If True, binarize signals using threshold (reduces noise).
        - Values below threshold → 0 (black/OFF)
        - Values above threshold → 1 (full color/ON)
        - When both channels are ON → overlap color (additive blend)
        Default: False (use continuous normalized values)
    binarization_threshold : float, optional
        Threshold for binarization (applied after normalization).
        Range: [0.0, 1.0], Default: 0.5
    max_traces_to_plot : int, optional
        Maximum number of trajectories to plot in the kymograph.
        If the dataset contains more trajectories, only the top N (after sorting/filtering) will be used.
        Default: 160
    
    Returns
    -------
    img : ndarray
        RGB kymograph image (H, W, 3)
    primary_data : ndarray
        Primary channel data used
    secondary_data : ndarray
        Secondary channel data used
    """
    
    # ===== COLOR PARSING HELPERS =====
    def _parse_color(color_spec):
        """Parse color specification to RGB tuple in [0, 1]."""
        LETTER_TO_COLOR = {
            'r': (1.0, 0.0, 0.0), 'g': (0.0, 1.0, 0.0), 'b': (0.0, 0.0, 1.0),
            'm': (1.0, 0.0, 1.0), 'c': (0.0, 1.0, 1.0), 'y': (1.0, 1.0, 0.0),
            'k': (0.0, 0.0, 0.0), 'w': (1.0, 1.0, 1.0),
        }
        IMAGEJ_COLORS = {
            "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
            "magenta": (1.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0), "yellow": (1.0, 1.0, 0.0),
            "gray": (1/3, 1/3, 1/3), "white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
        }
        
        if isinstance(color_spec, (list, tuple, np.ndarray)) and len(color_spec) == 3:
            r, g, b = map(float, color_spec)
            return (float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)))
        
        color_str = str(color_spec).lower().strip()
        if len(color_str) == 1 and color_str in LETTER_TO_COLOR:
            return LETTER_TO_COLOR[color_str]
        if color_str in IMAGEJ_COLORS:
            return IMAGEJ_COLORS[color_str]
        
        try:
            return to_rgb(color_spec)
        except:
            raise ValueError(f"Could not parse color '{color_spec}'")
    
    def _get_color_name(rgb_tuple):
        """Get readable name for RGB color."""
        COLOR_NAMES = {
            (1.0, 0.0, 0.0): "Red", (0.0, 1.0, 0.0): "Green", (0.0, 0.0, 1.0): "Blue",
            (1.0, 0.0, 1.0): "Magenta", (0.0, 1.0, 1.0): "Cyan", (1.0, 1.0, 0.0): "Yellow",
            (1.0, 1.0, 1.0): "White", (0.0, 0.0, 0.0): "Black",
        }
        rounded = tuple(round(c, 1) for c in rgb_tuple)
        return COLOR_NAMES.get(rounded, f"RGB{rgb_tuple}")
    
    def _compute_additive_blend(color1, color2):
        """Compute additive color blend."""
        r = min(color1[0] + color2[0], 1.0)
        g = min(color1[1] + color2[1], 1.0)
        b = min(color1[2] + color2[2], 1.0)
        return (r, g, b)
    
    # ===== SETUP COLORS =====
    if channel_colors is None:
        try:
            from microlive.imports import color_green, color_magenta
            primary_color = color_green
            secondary_color = color_magenta
        except ImportError:
            primary_color = (0.0, 1.0, 0.0)
            secondary_color = (1.0, 0.0, 1.0)
    else:
        if not isinstance(channel_colors, (list, tuple)) or len(channel_colors) != 2:
            raise ValueError("channel_colors must be a list of 2 colors")
        primary_color = _parse_color(channel_colors[0])
        secondary_color = _parse_color(channel_colors[1])
    
    overlap_color = _compute_additive_blend(primary_color, secondary_color)
    nan_color_rgb = _parse_color(nan_color)
    
    if verbose:
        print("=" * 70)
        print("DUAL-CHANNEL KYMOGRAPH GENERATION")
        print("=" * 70)
        print(f"Dataset: {dataset}")
        print(f"Mode: {'SIMULATION' if simulation_mode else 'EXPERIMENTAL'}")
        print(f"Primary channel: {primary_channel} (color: {_get_color_name(primary_color)})")
        print(f"Secondary channel: {secondary_channel} (color: {_get_color_name(secondary_color)})")
        print(f"Overlap color: {_get_color_name(overlap_color)}")
        print(f"Shift data: {'YES' if shift_data else 'NO'}")
        print(f"Binarization: {'YES (threshold={})'.format(binarization_threshold) if use_binarization else 'NO'}")
    
    # ===== DATA LOADING =====
    if simulation_mode:
        if SSA_data_1 is None or SSA_data_2 is None:
            raise ValueError("simulation_mode=True requires SSA_data_1 and SSA_data_2")
        primary_data = SSA_data_1.copy()
        secondary_data = SSA_data_2.copy()
        if verbose:
            print(f"\nLoaded simulated data:")
            print(f"  Primary shape: {primary_data.shape}")
            print(f"  Secondary shape: {secondary_data.shape}")
    else:
        if data_folder is None:
            raise ValueError("data_folder required when simulation_mode=False")
        
        folder_with_files, _, dataframe_prefix = dataset_selection(
            dataset, data_folder, control_spots_mode=False, downsample=False
        )
        
        if verbose:
            print(f"\nLoading experimental data from: {folder_with_files}")
        
        try:
            primary_data, secondary_data, n_cells = load_dual_channel_tracking_data(
                folder_with_files, base_field=selected_field,
                primary_channel=primary_channel, secondary_channel=secondary_channel,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                dataframe_prefix=dataframe_prefix, min_snr=min_snr,
                max_missing_frames=max_missing_frames, smooth_window=smooth_window,
                verbose=verbose,
                max_trajectory_length_percentile=max_trajectory_length_percentile,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dual-channel data: {e}")
        
        if verbose:
            print(f"\nLoaded experimental data:")
            print(f"  Number of cells: {n_cells}")
            print(f"  Primary shape: {primary_data.shape}")
            print(f"  Secondary shape: {secondary_data.shape}")
    
    # ===== VALIDATION =====
    if primary_data.shape != secondary_data.shape:
        raise ValueError(f"Shape mismatch: {primary_data.shape} vs {secondary_data.shape}")
    
    n_traces, n_time = primary_data.shape
    if n_traces == 0:
        raise ValueError("No trajectories loaded")
    
    if verbose:
        print(f"\nData summary:")
        print(f"  Total trajectories: {n_traces}")
        print(f"  Timepoints per trajectory: {n_time}")
    
    # ===== SHIFT DATA =====
    if shift_data:
        if verbose:
            print("\nShifting trajectories to align first valid datapoint...")
        try:
            primary_shifted, secondary_shifted = mi.Utilities().shift_trajectories(
                array_ch0=primary_data, array_ch1=secondary_data,
                min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
                max_missing_frames=max_missing_frames
            )
            primary_data = primary_shifted
            secondary_data = secondary_shifted
            if verbose:
                print(f"  Trajectories after shifting: {primary_data.shape[0]}")
                if primary_data.shape[0] < n_traces:
                    print(f"  Removed {n_traces - primary_data.shape[0]} trajectories")
                n_traces = primary_data.shape[0]
        except Exception as e:
            if verbose:
                print(f"  WARNING: Shifting failed ({e})")
    
    # ===== SORT BY DATA DENSITY =====
    if sort_by_data_density:
        valid_counts_primary = np.sum(np.isfinite(primary_data), axis=1)
        valid_counts_secondary = np.sum(np.isfinite(secondary_data), axis=1)
        valid_counts_total = valid_counts_primary + valid_counts_secondary
        sort_indices = np.argsort(-valid_counts_total)
        primary_data = primary_data[sort_indices, :]
        secondary_data = secondary_data[sort_indices, :]
        if verbose:
            print(f"\nSorted trajectories by data density:")
            print(f"  Top: {int(valid_counts_total[sort_indices[0]])} valid points")
            print(f"  Bottom: {int(valid_counts_total[sort_indices[-1]])} valid points")
    

    # ===== SUBSET SELECTION =====
    if selected_indices is not None:
        selected_indices = np.asarray(selected_indices)
        if selected_indices.dtype == bool:
            primary_data = primary_data[selected_indices, :]
            secondary_data = secondary_data[selected_indices, :]
        else:
            primary_data = primary_data[selected_indices]
            secondary_data = secondary_data[selected_indices]
        if verbose:
            print(f"\nApplied trajectory selection: {primary_data.shape[0]} trajectories")
    
    # ===== LIMIT NUMBER OF TRACES (NEW SECTION) =====
    if max_traces_to_plot is not None and primary_data.shape[0] > max_traces_to_plot:
        if verbose:
            print(f"\nLimiting trajectories for plotting:")
            print(f"  Total available: {primary_data.shape[0]}")
            print(f"  Maximum to plot: {max_traces_to_plot}")
            print(f"  Taking top {max_traces_to_plot} trajectories (already sorted by data density)")
        
        # Take only the first max_traces_to_plot rows (top trajectories after sorting)
        primary_data = primary_data[:max_traces_to_plot, :]
        secondary_data = secondary_data[:max_traces_to_plot, :]
        
        if verbose:
            print(f"  Final shape for plotting: {primary_data.shape}")
    elif verbose and max_traces_to_plot is not None:
        print(f"\nNo trajectory limiting needed ({primary_data.shape[0]} ≤ {max_traces_to_plot})")
    

    # ===== NORMALIZATION WITH NAN HANDLING =====
    def _normalize_with_nans(X, mode="per_trace_percentile", p_lo=1, p_hi=99, eps=1e-9):
        """Normalize array handling NaNs properly."""
        X = np.asarray(X, dtype=float)
        X_norm = np.full_like(X, np.nan)
        
        for i in range(X.shape[0]):
            row = X[i, :]
            finite_mask = np.isfinite(row)
            if not np.any(finite_mask):
                continue
            finite_vals = row[finite_mask]
            
            if mode == "per_trace_percentile":
                lo = np.percentile(finite_vals, p_lo)
                hi = np.percentile(finite_vals, p_hi)
                scale = max(hi - lo, eps)
                row_norm = (row - lo) / scale
            elif mode == "per_trace_max":
                m = np.max(finite_vals)
                row_norm = row / max(m, eps)
            elif mode == "global_percentile":
                all_finite = X[np.isfinite(X)]
                if all_finite.size > 0:
                    lo = np.percentile(all_finite, p_lo)
                    hi = np.percentile(all_finite, p_hi)
                    scale = max(hi - lo, eps)
                    row_norm = (row - lo) / scale
                else:
                    row_norm = row
            elif mode == "zscore":
                mu = np.mean(finite_vals)
                sd = np.std(finite_vals) + eps
                Z = (row - mu) / sd
                row_norm = (Z + 2) / 4.0
            else:
                raise ValueError(f"Unknown normalization mode: {mode}")
            
            X_norm[i, :] = np.clip(row_norm, 0.0, 1.0)
        return X_norm
    



    # ===== HANDLE NORMALIZATION =====
    if normalize is None:
        # Skip normalization - use raw data scaled to [0, 1] range using percentiles
        if verbose:
            print(f"\nSkipping normalization (normalize=None)")
            print(f"  Scaling raw data to [0, 1] using global percentiles (robust to outliers)")
        
        # Find global min/max across both channels (ignoring NaNs)
        all_finite_primary = primary_data[np.isfinite(primary_data)]
        all_finite_secondary = secondary_data[np.isfinite(secondary_data)]
        
        if all_finite_primary.size > 0 and all_finite_secondary.size > 0:
            # ✅ USE PERCENTILES INSTEAD OF MIN/MAX (robust to outliers)
            # Combine both channels to get consistent scaling
            all_finite_combined = np.concatenate([all_finite_primary, all_finite_secondary])
            
            # Use 1st and 99th percentiles instead of absolute min/max
            global_min = np.percentile(all_finite_combined, p_lo)  # Default: 1st percentile
            global_max = np.percentile(all_finite_combined, p_hi)  # Default: 99th percentile
            
            # Scale to [0, 1] using percentile range
            if global_max - global_min > 1e-9:
                primary_norm = (primary_data - global_min) / (global_max - global_min)
                secondary_norm = (secondary_data - global_min) / (global_max - global_min)
            else:
                # All values are the same - set to 0.5
                primary_norm = np.full_like(primary_data, 0.5)
                secondary_norm = np.full_like(secondary_data, 0.5)
            
            # ✅ CLIP to [0, 1] - this handles outliers beyond percentiles
            # Values below p_lo → 0, values above p_hi → 1
            primary_norm = np.clip(primary_norm, 0.0, 1.0)
            secondary_norm = np.clip(secondary_norm, 0.0, 1.0)
            
            # Preserve NaNs
            primary_norm[~np.isfinite(primary_data)] = np.nan
            secondary_norm[~np.isfinite(secondary_data)] = np.nan
            
            if verbose:
                print(f"  Global percentile range (P{p_lo}-P{p_hi}): [{global_min:.2f}, {global_max:.2f}]")
                
                # Report how many values were clipped
                n_clipped_low_p = np.sum(primary_data[np.isfinite(primary_data)] < global_min)
                n_clipped_high_p = np.sum(primary_data[np.isfinite(primary_data)] > global_max)
                n_clipped_low_s = np.sum(secondary_data[np.isfinite(secondary_data)] < global_min)
                n_clipped_high_s = np.sum(secondary_data[np.isfinite(secondary_data)] > global_max)
                
                total_clipped = n_clipped_low_p + n_clipped_high_p + n_clipped_low_s + n_clipped_high_s
                total_finite = all_finite_primary.size + all_finite_secondary.size
                
                if total_clipped > 0:
                    print(f"  Clipped outliers: {total_clipped}/{total_finite} ({100*total_clipped/total_finite:.1f}%)")
                    print(f"    Primary below P{p_lo}: {n_clipped_low_p}, above P{p_hi}: {n_clipped_high_p}")
                    print(f"    Secondary below P{p_lo}: {n_clipped_low_s}, above P{p_hi}: {n_clipped_high_s}")
        else:
            if verbose:
                print(f"  WARNING: No finite data found, using zeros")
            primary_norm = np.zeros_like(primary_data)
            secondary_norm = np.zeros_like(secondary_data)

    else:
        # Apply normalization
        if isinstance(normalize, str):
            normalize_modes = [normalize, normalize]
        else:
            normalize_modes = list(normalize)
        
        if verbose:
            print(f"\nNormalizing data:")
            print(f"  Primary mode: {normalize_modes[0]}")
            print(f"  Secondary mode: {normalize_modes[1]}")
        
        primary_norm = _normalize_with_nans(primary_data, mode=normalize_modes[0], p_lo=p_lo, p_hi=p_hi)
        secondary_norm = _normalize_with_nans(secondary_data, mode=normalize_modes[1], p_lo=p_lo, p_hi=p_hi)


    # ===== HANDLE NORMALIZATION =====
    # if normalize is None:
    #     # Skip normalization - use raw data scaled to [0, 1] range
    #     if verbose:
    #         print(f"\nSkipping normalization (normalize=None)")
    #         print(f"  Scaling raw data to [0, 1] using global min/max")
        
    #     # Find global min/max across both channels (ignoring NaNs)
    #     all_finite_primary = primary_data[np.isfinite(primary_data)]
    #     all_finite_secondary = secondary_data[np.isfinite(secondary_data)]
        
    #     if all_finite_primary.size > 0 and all_finite_secondary.size > 0:
    #         global_min = min(np.min(all_finite_primary), np.min(all_finite_secondary))
    #         global_max = max(np.max(all_finite_primary), np.max(all_finite_secondary))
            
    #         # Scale to [0, 1] using global range
    #         if global_max - global_min > 1e-9:
    #             primary_norm = (primary_data - global_min) / (global_max - global_min)
    #             secondary_norm = (secondary_data - global_min) / (global_max - global_min)
    #         else:
    #             # All values are the same - set to 0.5
    #             primary_norm = np.full_like(primary_data, 0.5)
    #             secondary_norm = np.full_like(secondary_data, 0.5)
            
    #         # Clip to [0, 1] and preserve NaNs
    #         primary_norm = np.clip(primary_norm, 0.0, 1.0)
    #         secondary_norm = np.clip(secondary_norm, 0.0, 1.0)
    #         primary_norm[~np.isfinite(primary_data)] = np.nan
    #         secondary_norm[~np.isfinite(secondary_data)] = np.nan
            
    #         if verbose:
    #             print(f"  Global range: [{global_min:.2f}, {global_max:.2f}]")
    #     else:
    #         if verbose:
    #             print(f"  WARNING: No finite data found, using zeros")
    #         primary_norm = np.zeros_like(primary_data)
    #         secondary_norm = np.zeros_like(secondary_data)

    # else:
    #     # Apply normalization
    #     if isinstance(normalize, str):
    #         normalize_modes = [normalize, normalize]
    #     else:
    #         normalize_modes = list(normalize)
        
    #     if verbose:
    #         print(f"\nNormalizing data:")
    #         print(f"  Primary mode: {normalize_modes[0]}")
    #         print(f"  Secondary mode: {normalize_modes[1]}")
        
    #     primary_norm = _normalize_with_nans(primary_data, mode=normalize_modes[0], p_lo=p_lo, p_hi=p_hi)
    #     secondary_norm = _normalize_with_nans(secondary_data, mode=normalize_modes[1], p_lo=p_lo, p_hi=p_hi)

    # ===== GAMMA CORRECTION =====

    # ===== GAMMA CORRECTION =====
    if gamma is not None and not use_binarization:
        if verbose:
            print(f"  Applying gamma correction: {gamma}")
        finite_p = np.isfinite(primary_norm)
        finite_s = np.isfinite(secondary_norm)
        primary_norm[finite_p] = np.power(primary_norm[finite_p], 1.0 / float(gamma))
        secondary_norm[finite_s] = np.power(secondary_norm[finite_s], 1.0 / float(gamma))
    
    # ===== BINARIZATION =====
    if use_binarization:
        if verbose:
            print(f"\nApplying binarization:")
            print(f"  Threshold: {binarization_threshold}")
        
        # Create binary masks (1 where above threshold, 0 where below, NaN stays NaN)
        primary_binary = np.full_like(primary_norm, np.nan)
        secondary_binary = np.full_like(secondary_norm, np.nan)
        
        # Apply threshold to finite values only
        finite_p = np.isfinite(primary_norm)
        finite_s = np.isfinite(secondary_norm)
        
        primary_binary[finite_p] = (primary_norm[finite_p] >= binarization_threshold).astype(float)
        secondary_binary[finite_s] = (secondary_norm[finite_s] >= binarization_threshold).astype(float)
        
        # Count ON pixels
        n_primary_on = np.sum(primary_binary == 1.0)
        n_secondary_on = np.sum(secondary_binary == 1.0)
        n_both_on = np.sum((primary_binary == 1.0) & (secondary_binary == 1.0))
        
        if verbose:
            total_valid = np.sum(finite_p | finite_s)
            print(f"  Primary ON: {n_primary_on}/{total_valid} ({100*n_primary_on/max(1,total_valid):.1f}%)")
            print(f"  Secondary ON: {n_secondary_on}/{total_valid} ({100*n_secondary_on/max(1,total_valid):.1f}%)")
            print(f"  Both ON (overlap): {n_both_on}/{total_valid} ({100*n_both_on/max(1,total_valid):.1f}%)")
        
        # Use binary values for rendering
        primary_norm = primary_binary
        secondary_norm = secondary_binary
    
    # ===== BUILD RGB IMAGE (CORRECTED) =====
    H, W = primary_norm.shape
    img = np.zeros((H, W, 3), dtype=float)
    
    r_p, g_p, b_p = primary_color
    r_s, g_s, b_s = secondary_color
    r_nan, g_nan, b_nan = nan_color_rgb
    
    # ✅ FIX: Only mark pixels that are NaN in BOTH channels
    nan_both = np.isnan(primary_norm) & np.isnan(secondary_norm)
    
    # Initialize ONLY NaN pixels with nan_color
    img[nan_both, 0] = r_nan
    img[nan_both, 1] = g_nan
    img[nan_both, 2] = b_nan
    
    if use_binarization:
        # ===== BINARIZED RENDERING =====
        primary_on = (primary_norm == 1.0)
        secondary_on = (secondary_norm == 1.0)
        both_on = primary_on & secondary_on
        
        # Primary ONLY (not in overlap)
        primary_only = primary_on & ~secondary_on
        img[primary_only, 0] = r_p
        img[primary_only, 1] = g_p
        img[primary_only, 2] = b_p
        
        # Secondary ONLY (not in overlap)
        secondary_only = secondary_on & ~primary_on
        img[secondary_only, 0] = r_s
        img[secondary_only, 1] = g_s
        img[secondary_only, 2] = b_s
        
        # Both ON (overlap color = additive blend)
        overlap_r, overlap_g, overlap_b = overlap_color
        img[both_on, 0] = overlap_r
        img[both_on, 1] = overlap_g
        img[both_on, 2] = overlap_b
        
    else:
        # ===== CONTINUOUS RENDERING =====
        valid_primary = np.isfinite(primary_norm)
        valid_secondary = np.isfinite(secondary_norm)
        
        img[valid_primary, 0] += primary_norm[valid_primary] * r_p
        img[valid_primary, 1] += primary_norm[valid_primary] * g_p
        img[valid_primary, 2] += primary_norm[valid_primary] * b_p
        
        img[valid_secondary, 0] += secondary_norm[valid_secondary] * r_s
        img[valid_secondary, 1] += secondary_norm[valid_secondary] * g_s
        img[valid_secondary, 2] += secondary_norm[valid_secondary] * b_s
    
    # Clip to valid range
    img = np.clip(img, 0.0, 1.0)
    
    # ===== PLOTTING =====
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img, origin="upper", aspect="auto", interpolation='nearest')
    
    ax.set_xlabel("Time (frames)", fontsize=12)
    ax.set_ylabel("Trajectory index" if orientation == "rows=trajectories" else "Time index", fontsize=12)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.grid(False)
    
    # Title
    if title is None:
        mode_str = "Simulation" if simulation_mode else "Experimental"
        shift_str = " (Shifted)" if shift_data else ""
        binary_str = f" (Binary, t={binarization_threshold})" if use_binarization else ""
        title = f"Dual-Channel Kymograph - {dataset} ({mode_str}){shift_str}{binary_str}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # ===== SAVE =====
    # Determine save path: explicit save_path takes priority, otherwise use results_folder
    if save_path is None and results_folder is not None:
        # Auto-generate filename based on parameters
        results_folder = Path(results_folder)
        mode_suffix = '_simulation' if simulation_mode else '_experimental'
        shift_suffix = '_shifted' if shift_data else ''
        binary_suffix = f'_binary_t{binarization_threshold}' if use_binarization else ''
        save_path = results_folder / f'kymograph_{dataset}_ch{primary_channel}_vs_ch{secondary_channel}{mode_suffix}{shift_suffix}{binary_suffix}.png'
        if verbose:
            print(f"\nAuto-generated save path from results_folder: {save_path}")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and SVG with same base name
        base_path = save_path.with_suffix('')  # Remove extension
        fig.savefig(f'{base_path}.png', bbox_inches="tight", dpi=dpi)
        fig.savefig(f'{base_path}.svg', bbox_inches="tight")
        if verbose:
            print(f"Saved kymograph to: {base_path}.[png|svg]")
    
    # ===== DISPLAY =====
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    if verbose:
        print("\nKymograph generation complete")
        print("=" * 70)
    
    return img, primary_data, secondary_data





# ══════════════════════════════════════════════════════════════════════════════
# INTENSITY DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def extract_intensity_distributions(
    data_folder: Path,
    dataset: str = 'cof',
    selected_field: str = 'spot_int_ch_',
    channel_index: int = 1,
    snr_field: str = 'snr_ch_',
    min_snr: float = 0.5,
    timepoint_frame: int = None,   # for mode 3; None → uses median occupied frame
    control_spots_mode: bool = False,
    dataframe_prefix: str = 'tracking_',
    verbose: bool = True,
):
    """
    Extract intensity distributions from tracking CSVs in three complementary modes.

    Parameters
    ----------
    data_folder : Path
        Root folder that contains results_* subfolders with tracking CSVs.
    dataset : str
        Dataset selection key (passed to dataset_selection).
    selected_field : str
        Column prefix for intensity, e.g. 'spot_int_ch_'.
    channel_index : int
        Channel number appended to selected_field (1-based).
    snr_field : str
        Column prefix for SNR, e.g. 'snr_ch_'.
    min_snr : float
        Minimum SNR to keep a data point.
    timepoint_frame : int or None
        Frame index used for mode 3.  None → median of all occupied frames.
    control_spots_mode : bool
        Passed to dataset_selection.
    dataframe_prefix : str
        Prefix for tracking CSV filenames.
    verbose : bool

    Returns
    -------
    dict with keys
        'mean_per_particle'   : 1-D array  — one value per particle (mode 1)
        'all_timepoints'      : 1-D array  — every (particle × frame) value (mode 2)
        'at_timepoint'        : 1-D array  — values at chosen frame (mode 3)
        'timepoint_frame_used': int        — frame index actually used for mode 3
        'n_cells'             : int
        'n_particles'         : int        — unique particles that passed SNR filter
        'cell_ids'            : list       — which cell each particle belongs to (mode 1)
    """
    # ── resolve folder ────────────────────────────────────────────────────────
    folder_with_files, _, _ = dataset_selection(
        dataset, data_folder, control_spots_mode, downsample=False
    )
    int_col = selected_field + str(channel_index)
    snr_col = snr_field + str(channel_index)

    tracking_files = _find_tracking_files(folder_with_files, dataframe_prefix)

    # accumulators
    mean_per_particle   = []   # mode 1
    all_particle_series = []   # mode 1 — Series per cell, indexed by global particle key
    all_timepoints      = []   # mode 2
    at_timepoint_vals   = []   # mode 3
    cell_ids_out        = []   # cell tag per particle (mode 1)
    all_frames          = []   # track occupied frames for auto timepoint
    cached_dfs          = []   # filtered DataFrames cached here; reused for mode 3
    n_cells = 0
    n_particles_total = 0

    for cell_idx, fp in enumerate(tracking_files):
        try:
            df = pd.read_csv(fp)
            if df.empty or int_col not in df.columns:
                continue

            # optional SNR filter
            if snr_col in df.columns:
                df = df[df[snr_col] >= min_snr].copy()
            df = df[df[int_col].notna() & (df[int_col] > 0)].copy()
            if df.empty:
                continue

            cached_dfs.append(df)   # cache for mode 3 — avoids second disk read
            n_cells += 1
            all_frames.extend(df['frame'].tolist())

            # ── Mode 1: mean intensity per particle ───────────────────────────
            particle_series = (
                df.groupby('particle')[int_col]
                  .mean()                    # Series: index=particle_id, values=mean intensity
            )
            particle_series_tagged = particle_series.copy()
            particle_series_tagged.index = [
                f"{cell_idx}_{pid}" for pid in particle_series.index
            ]  # unique global particle key = cell_idx + particle_id
            mean_per_particle.extend(particle_series.values.tolist())
            cell_ids_out.extend([cell_idx] * len(particle_series))
            n_particles_total += len(particle_series)
            all_particle_series.append(particle_series_tagged)

            # ── Mode 2: all (particle × frame) as independent samples ─────────
            all_timepoints.extend(df[int_col].tolist())

        except Exception as e:
            if verbose:
                print(f"  SKIP {fp.name}: {e}")
            continue

    if not mean_per_particle:
        raise ValueError("No valid intensity data found.")

    # ── Mode 3: particles at a specific time point ────────────────────────────
    # Reuse cached_dfs — no second disk read needed.
    if timepoint_frame is None:
        timepoint_frame = int(np.median(all_frames))
    if verbose:
        print(f"  Mode 3: using frame {timepoint_frame}  "
              f"(pass timepoint_frame= to override)")

    for df in cached_dfs:
        df_t = df[df['frame'] == timepoint_frame].copy()
        df_t = df_t[df_t[int_col].notna() & (df_t[int_col] > 0)]
        at_timepoint_vals.extend(df_t[int_col].tolist())

    result = {
        'mean_per_particle'       : np.array(mean_per_particle),
        'mean_per_particle_series': pd.concat(all_particle_series) if all_particle_series else pd.Series(dtype=float),
        'all_timepoints'          : np.array(all_timepoints),
        'at_timepoint'            : np.array(at_timepoint_vals),
        'timepoint_frame_used'    : timepoint_frame,
        'n_cells'                 : n_cells,
        'n_particles'             : n_particles_total,
        'cell_ids'                : cell_ids_out,
    }
    if verbose:
        print(f"  Cells: {n_cells}  |  Particles: {n_particles_total}  "
              f"|  Mode-2 points: {len(all_timepoints)}  "
              f"|  Mode-3 points (frame {timepoint_frame}): {len(at_timepoint_vals)}")
    return result


# def plot_intensity_distributions(
#     dist_results: list,
#     list_names: list,
#     list_colors: list,
#     channel_index: int = 1,
#     mode: int = None,          # 1, 2, or 3 → single panel. None → original 3-panel
#     x_label: str = 'Intensity (a.u.)',
#     figsize_single: tuple = (5, 4),
#     figsize_triple: tuple = (14, 4.5),
#     figsize: tuple = None,     # backward-compatible alias for figsize_triple
#     bins: int = 60,
#     kde: bool = True,
#     xlim: tuple = None,
#     save_name: str = 'intensity_distributions_ch',
#     show: bool = True,
# ):
#     """
#     Plot intensity distributions.

#     Parameters
#     ----------
#     dist_results : list of dicts from extract_intensity_distributions
#     list_names   : dataset labels
#     list_colors  : one colour per dataset
#     mode         : 1=mean per particle, 2=all timepoints, 3=snapshot at frame.
#                    None → three-panel figure (original behaviour).
#     x_label      : x-axis label (only used in single-mode plot)
#     """

#     _mode_map = {
#         1: 'mean_per_particle',
#         2: 'all_timepoints',
#         3: 'at_timepoint',
#     }

#     def _plot_one(ax, key):
#         for r, name, color in zip(dist_results, list_names, list_colors):
#             vals = r[key]
#             vals = vals[np.isfinite(vals) & (vals > 0)]
#             if vals.size == 0:
#                 continue
#             if kde and vals.size > 3:
#                 xs = np.linspace(vals.min(), vals.max(), 500)
#                 try:
#                     ys = gaussian_kde(vals, bw_method='scott')(xs)
#                     ax.plot(xs, ys, color=color, linewidth=1.8, label=name)
#                     ax.fill_between(xs, ys, alpha=0.12, color=color)
#                 except Exception:
#                     ax.hist(vals, bins=bins, density=True, color=color,
#                             alpha=0.35, label=name)
#             else:
#                 ax.hist(vals, bins=bins, density=True, color=color,
#                         alpha=0.35, label=name)
#             ax.axvline(np.median(vals), color=color, linewidth=1.0,
#                        linestyle='--', alpha=0.7)
#         #ax.legend(fontsize=7, framealpha=0.85)
#         ax.legend(fontsize=12, framealpha=0.85,
#           loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(list_names))
#         ax.grid(True, alpha=0.2, linewidth=0.4)
#         if xlim is not None:
#             ax.set_xlim(xlim)

#     # ── Single-mode (one panel) ───────────────────────────────────────────────
#     if mode is not None:
#         if mode not in _mode_map:
#             raise ValueError(f"mode must be 1, 2 or 3, got {mode}")
#         key = _mode_map[mode]
#         fig, ax = plt.subplots(figsize=figsize_single)
#         _plot_one(ax, key)
#         ax.set_xlabel(x_label, fontsize=16)
#         ax.set_ylabel('Probability Density', fontsize=16)
#         plt.tight_layout()
#         suffix = f'_mode{mode}'

#     # ── Three-panel (original) ────────────────────────────────────────────────
#     else:
#         _titles = [
#             ('mean_per_particle', 'Mean intensity per particle\n(one value per trajectory)'),
#             ('all_timepoints',    'All observations\n(every particle × frame)'),
#             ('at_timepoint',      'Snapshot\n(particles at median frame)'),
#         ]
#         fig, axes = plt.subplots(1, 3, figsize=figsize or figsize_triple, sharey=False)
#         #fig.suptitle(f'Intensity Distributions — Channel {channel_index}',
#         #             fontsize=13, fontweight='bold', y=1.01)
#         for ax, (key, title) in zip(axes, _titles):
#             _plot_one(ax, key)
#             #ax.set_title(title, fontsize=14)
#             ax.set_xlabel('Intensity (a.u.)', fontsize=16)
#             ax.set_ylabel('Probability Density', fontsize=16)
#         suffix = ''

#     svg_path = f'{save_name}{channel_index}{suffix}.svg'
#     png_path = f'{save_name}{channel_index}{suffix}.png'
#     plt.savefig(svg_path, dpi=300, bbox_inches='tight')
#     plt.savefig(png_path, dpi=300, bbox_inches='tight')
#     if show:
#         plt.show()
#         plt.close(fig)
#         print(f'Saved: {svg_path} / {png_path}')
#         return None   # returning fig causes Jupyter to re-render it; None prevents double plot
#     else:
#         plt.close(fig)
#         print(f'Saved: {svg_path} / {png_path}')
#         return fig

def plot_intensity_distributions(
    dist_results: list,
    list_names: list,
    list_colors: list,
    channel_index: int = 1,
    mode: int = 1,
    x_label: str = 'PSF Amplitude (a.u.)',
    figsize: tuple = (5, 5),
    bins: int = 60,
    kde: bool = True,
    xlim: tuple = None,
    save_name: str = 'intensity_distributions_ch',
    show: bool = True,
):
    """
    Plot intensity distributions as a single panel.

    Parameters
    ----------
    dist_results : list of dicts from extract_intensity_distributions
    list_names   : dataset labels
    list_colors  : one colour per dataset
    mode         : 1=mean per particle, 2=all timepoints, 3=snapshot at frame
    x_label      : x-axis label
    """
    _mode_map = {
        1: 'mean_per_particle',
        2: 'all_timepoints',
        3: 'at_timepoint',
    }
    if mode not in _mode_map:
        raise ValueError(f"mode must be 1, 2 or 3, got {mode}")

    fig, ax = plt.subplots(figsize=figsize)

    for r, name, color in zip(dist_results, list_names, list_colors):
        vals = r[_mode_map[mode]]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        if kde and vals.size > 3:
            xs = np.linspace(vals.min(), vals.max(), 500)
            try:
                ys = gaussian_kde(vals, bw_method='scott')(xs)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=name)
                ax.fill_between(xs, ys, alpha=0.12, color=color)
            except Exception:
                ax.hist(vals, bins=bins, density=True, color=color,
                        alpha=0.35, label=name)
        else:
            ax.hist(vals, bins=bins, density=True, color=color,
                    alpha=0.35, label=name)
        ax.axvline(np.median(vals), color=color, linewidth=1.0,
                   linestyle='--', alpha=0.7)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel('Probability Density', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, framealpha=0.85,
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=len(list_names))
    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    suffix = f'_mode{mode}'
    svg_path = f'{save_name}{channel_index}{suffix}.svg'
    png_path = f'{save_name}{channel_index}{suffix}.png'
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {svg_path} / {png_path}')
