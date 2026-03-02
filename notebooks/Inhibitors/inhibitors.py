# !/usr/bin/env python3
from pathlib import Path
current_dir = Path().resolve()
from microlive.imports import *
from microlive import microscopy as mi
import tasep_models as tm
from tasep_models import *


plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
    })



def calculate_number_of_particles_per_frame(particle_counts_per_frame, inhibitor_frame_index):
    """
    Normalize the number of particles per frame to the average before treatment.

    All frames are divided by the average particle count before treatment.
    If that average is zero, returns an array of zeros.

    Parameters:
        particle_counts_per_frame (np.ndarray): 1D array of particle counts per frame.
        inhibitor_frame_index (int): The frame index at which treatment starts.

    Returns:
        tuple: (normalized_particles, average_particles_before_treatment)
            - normalized_particles: Particle counts normalized to pre-treatment average
            - average_particles_before_treatment: The average used for normalization
    """
    # Compute the average particle count before treatment
    pre_counts = particle_counts_per_frame[:inhibitor_frame_index]
    average_particles_before_treatment = pre_counts.mean()

    # Normalize all frames by the pre-treatment average
    if average_particles_before_treatment == 0:
        normalized_particles = np.zeros_like(particle_counts_per_frame, dtype=float)
    else:
        normalized_particles = particle_counts_per_frame / average_particles_before_treatment

    return normalized_particles, average_particles_before_treatment




def calculate_intensity(particle_counts_per_frame, sum_intensities_per_frame, inhibitor_frame_index, normalization_method='mean', percentile_range=(5, 95)):
    """
    Normalize the intensity per frame.

    For frames before the treatment, each frame's intensity is given by
    sum_intensities / particle_counts. If the particle count is zero in a frame,
    the normalized intensity is set to zero.

    For frames after the treatment, the sum intensities are divided by the average
    particle count before treatment. If that average is zero, zeros are returned for
    all frames after treatment.

    Parameters:
        particle_counts_per_frame (np.ndarray): 1D array of particle counts per frame.
        sum_intensities_per_frame (np.ndarray): 1D array of sum intensities per frame.
        inhibitor_frame_index (int): The frame index at which treatment starts.
        normalization_method (str): 'mean' (default) divides by pre-treatment mean,
            'minmax' scales the trajectory to [0, 1] range,
            'percentile' scales using percentile bounds (robust to outliers).
            None: no normalization.
        percentile_range (tuple): Percentile bounds for 'percentile' method.
            Default (5, 95). Use (1, 99) for wider range.

    Returns:
        tuple: (normalized_intensities, raw_avg_intensities, avg_particles_before_treatment)
    """
    # Compute the average particle count before treatment.
    pre_counts = particle_counts_per_frame[:inhibitor_frame_index]
    average_particles_before_treatment = pre_counts.mean()

    # For frames before treatment, avoid division by zero:
    pre_intensities = sum_intensities_per_frame[:inhibitor_frame_index]
    intensity_before_treatment = np.where(pre_counts == 0, 0, pre_intensities / pre_counts)
    # For frames after treatment, if the average is zero then return zeros.
    post_intensities = sum_intensities_per_frame[inhibitor_frame_index:]
    if average_particles_before_treatment == 0:
        intensity_after_treatment = np.zeros_like(post_intensities)
    else:
        intensity_after_treatment = post_intensities / average_particles_before_treatment
    # Combine the two segments and return the result.
    average_intensity_with_respect_number_particles = np.concatenate([intensity_before_treatment, intensity_after_treatment])

    # Apply normalization
    if normalization_method == 'minmax':
        val_min = average_intensity_with_respect_number_particles.min()
        val_max = average_intensity_with_respect_number_particles.max()
        if val_max - val_min == 0:
            intensities_normalized_before_treatment_intensity = np.zeros_like(average_intensity_with_respect_number_particles)
        else:
            intensities_normalized_before_treatment_intensity = (average_intensity_with_respect_number_particles - val_min) / (val_max - val_min)
    elif normalization_method == 'percentile':
        val_low = np.percentile(average_intensity_with_respect_number_particles, percentile_range[0])
        val_high = np.percentile(average_intensity_with_respect_number_particles, percentile_range[1])
        if val_high - val_low == 0:
            intensities_normalized_before_treatment_intensity = np.zeros_like(average_intensity_with_respect_number_particles)
        else:
            intensities_normalized_before_treatment_intensity = (average_intensity_with_respect_number_particles - val_low) / (val_high - val_low)
    elif normalization_method ==     'mean':
        mean_before_treatment = average_intensity_with_respect_number_particles[:inhibitor_frame_index].mean()
        intensities_normalized_before_treatment_intensity = average_intensity_with_respect_number_particles / mean_before_treatment
    elif normalization_method is None:
        intensities_normalized_before_treatment_intensity = average_intensity_with_respect_number_particles

    return intensities_normalized_before_treatment_intensity, average_intensity_with_respect_number_particles, average_particles_before_treatment




def plot_inhibitor(full_frames,intensities_normalized,inhibitor_frame_index, results_folder=None,plot_name='HT',list_param=None,responding_indices=None,figsize=(6, 3),time_array_min=None,
        mean_intensity_ssa_inh=None, err_intensity_ssa_inh=None, use_sem=True,show_individual_trajectories=True,threshold_percentage_for_runoff = 0.2,show_runoff_time=True, ylims=(0, 1.5), y_label='Norm. Intensity', treatment_label='Inhibitor', show_treatment_line=True):
    if results_folder is None: # create a new folder called results_HT in the current directory
        results_folder = Path(current_dir).joinpath('results_HT')
        results_folder.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    # test if intensities_normalized is not empty or None or full of zeros
    if intensities_normalized is None or len(intensities_normalized) == 0:
        print('No data to plot.')
        return

    if responding_indices is None:
        responding_indices = list(range(len(intensities_normalized)))

    if show_individual_trajectories:
        if responding_indices:
            for i in responding_indices:
                ax.plot(full_frames, intensities_normalized[i],
                        linestyle='-', color='dimgray', linewidth=0.2)
    if responding_indices:  # check if the list is not empty
        mean_trajectory = np.mean(intensities_normalized[responding_indices, :], axis=0)
        std_trajectory = np.std(intensities_normalized[responding_indices, :], axis=0)
        if use_sem:
            err_trajectory = std_trajectory / np.sqrt(len(responding_indices))
        else:
            err_trajectory = std_trajectory
    line_mean, = ax.plot(full_frames, mean_trajectory, 'o-',
                        color='blue', linewidth=1, label='Experimental (mean)', markersize=7)
    ax.fill_between(full_frames,
                    mean_trajectory - err_trajectory,
                    mean_trajectory + err_trajectory,
                    color='blue', alpha=0.07)

    if mean_intensity_ssa_inh is not None and err_intensity_ssa_inh is not None:
        legend_label_sim = (fr'Model Fit ($k_e$={np.round(list_param[1],1)}, $k_i$={np.round(list_param[0],3)})'
                        if list_param[1] is not None and list_param[0] is not None else 'Simulation')
        plt.plot(time_array_min-5, mean_intensity_ssa_inh,'-',  color='red', linewidth=3,label=legend_label_sim)
        plt.fill_between(time_array_min-5, mean_intensity_ssa_inh - err_intensity_ssa_inh,
                        mean_intensity_ssa_inh + err_intensity_ssa_inh, color='red', alpha=0.1)

    use_sigmoidal_fit = True
    if use_sigmoidal_fit:
        # in this section fit the sigmoidal function to the data and determine the runoff time as the time when the curve reaches threshold
        def decreasing_sigmoid(t, ymin, ymax, t_half, slope):
            return ymin + (ymax - ymin) / (1.0 + np.exp((t - t_half) / slope))
        # Initial guess: t_half is slightly after treatment (time=0)
        t_half_guess = (full_frames.max() - full_frames.min()) / 4  # Guess t_half as 1/4 of the time range
        popt, _ = curve_fit(decreasing_sigmoid, full_frames, mean_trajectory, p0=[0.1, 1.0, t_half_guess, (full_frames.max()-full_frames.min())/4], maxfev=10000)
        ymin, ymax, t_half_fit, slope_fit = popt
        fitted = decreasing_sigmoid(full_frames, *popt)
        threshold = threshold_percentage_for_runoff

        # Find the time when fitted curve crosses threshold
        idxs = np.where(fitted <= threshold)[0]
        if len(idxs) == 0:
            time_fit = None
        else:
            # Get the actual time value at the threshold crossing
            time_fit = full_frames[idxs[0]]
        
        if time_fit is not None:
            ax.axvline(x=time_fit, color='g', linestyle='--', linewidth=1,
                    label=r' $\tau_{INH}$'+ f' ~ {time_fit:.1f} min')

        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1,
                    label=f'runoff threshold ~ {threshold:.1f}')
        plt.plot(full_frames, fitted, '-', color='r', linewidth=1.5, label='Sigmoidal Fit')

    # plot inhibitor line at zero
    if show_treatment_line:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1,
                    label=f'{treatment_label} Treatment')
    ax.set_xlabel("Time (min)", fontdict={'size': 16, 'color': 'black'})
    ax.set_ylabel(y_label, fontdict={'size': 16, 'color': 'black'})
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black', colors='black')

    # # Set spines color to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.ylim(ylims)
    plt.tight_layout()
    ax.legend(fontsize=10)
    plt.savefig(results_folder.joinpath('HT_'+plot_name+'.png'), dpi=600)
    #plt.savefig(results_folder.joinpath('HT_'+plot_name+'.svg'), dpi=600)

    plt.show()

    return None


def plot_multiple_inhibitors(full_frames_list,
                                intensities_normalized_list,
                                inhibitor_frame_index,
                                results_folder=None,
                                plot_name='HT_multi',
                                responding_indices_list=None,
                                figsize=(6, 3),
                                colors=None,
                                legend_labels=None,
                                use_sem=True,
                                show_individual_trajectories=True,
                                threshold_percentage_for_runoff=0.2,
                                use_sigmoidal_fit=False,
                                ylims=(0, 1.5),
                                xlims=None,
                                show_runoff_time=True,
                                y_label='Norm. Intensity',
                                treatment_label='Inhibitor',
                                show_treatment_line=True):
    """
    Plot multiple inhibitor datasets on the same axes.

    Parameters
    ----------
    full_frames_list : list or array
        Either a single 1D array of frame times (applies to all datasets)
        or a list of 1D arrays, one per dataset.
    intensities_normalized_list : list of 2D arrays
        Each element is an (n_cells × n_frames) array of normalized intensities.
    inhibitor_frame_index : int
        Frame index at which inhibitor treatment starts.
    results_folder : Path or str, optional
        Where to save the figure (will be created if needed).
    plot_name : str, optional
        Filename suffix for the saved figure.
    responding_indices_list : list of lists, optional
        Per‐dataset lists of cell indices to include. Defaults to all.
    figsize : tuple, optional
    colors : list of str, optional
        Matplotlib color codes for each dataset.
    legend_labels : list of str, optional
        Text labels for each dataset's mean trace.
    use_sem : bool, optional
        If True, error bands show SEM; else SD.
    show_individual_trajectories : bool, optional
    threshold_percentage_for_runoff : float, optional
        Fraction of plateau to mark as "runoff" threshold (for τ lines).
    use_sigmoidal_fit : bool, optional
        If True, fit and overlay a decreasing sigmoid per dataset.
    ylims : tuple, optional
        (ymin, ymax) for the plot.
    xlims : tuple, optional
        (xmin, xmax) for the plot. If None, auto-scaled.
    show_runoff_time : bool, optional
        If True, draw horizontal threshold & vertical τ lines when fitting.
    """
    # Prepare output folder
    results_folder = Path(results_folder or Path('.')).joinpath(f"results_{plot_name}")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loop over each dataset
    for idx, intensities in enumerate(intensities_normalized_list):
        # Select frames array
        frames = (full_frames_list[idx]
                  if isinstance(full_frames_list, (list, tuple))
                  else full_frames_list)
        color = colors[idx % len(colors)]

        # Select responding cell indices
        resp_idx = (responding_indices_list[idx]
                    if (responding_indices_list and idx < len(responding_indices_list))
                    else list(range(intensities.shape[0])))

        # Plot individual trajectories
        if show_individual_trajectories:
            for i in resp_idx:
                ax.plot(frames, intensities[i],
                        linestyle='-', color=color,
                        linewidth=0.3, alpha=0.4)

        # Compute mean & error
        data = intensities[resp_idx, :]
        mean_traj = np.mean(data, axis=0)
        std_traj  = np.std(data, axis=0)
        err_traj  = (std_traj / np.sqrt(data.shape[0])) if use_sem else std_traj

        # Determine legend text
        label_text = (legend_labels[idx]
                      if (legend_labels and idx < len(legend_labels))
                      else f'Dataset {idx+1}')

        # Plot mean ± error band
        ax.plot(frames, mean_traj, 'o-', color=color,
                linewidth=1.5, markersize=6,
                label=label_text)
        ax.fill_between(frames,
                        mean_traj - err_traj,
                        mean_traj + err_traj,
                        color=color, alpha=0.2)

        # Optional sigmoidal fit
        if use_sigmoidal_fit:
            def decreasing_sigmoid(t, ymin, ymax, t_half, slope):
                return ymin + (ymax - ymin) / (1.0 + np.exp((t - t_half) / slope))
            t_half_guess = inhibitor_frame_index + 3
            try:
                popt, _ = curve_fit(
                    decreasing_sigmoid, frames, mean_traj,
                    p0=[mean_traj.min(), mean_traj.max(), t_half_guess, (frames.max()-frames.min())/4],
                    maxfev=5000
                )
                fitted = decreasing_sigmoid(frames, *popt)
                ax.plot(frames, fitted, '--', color=color,
                        linewidth=1.5, label=f'{label_text} Fit')

                if show_runoff_time:
                    threshold = threshold_percentage_for_runoff
                    idxs = np.where(fitted <= threshold)[0]
                    if len(idxs) > 0:
                        tau = frames[idxs[0]]
                        ax.axvline(x=tau, color=color, linestyle=':', linewidth=1,
                                   label=f'{label_text} τ~{tau:.1f}')
                        ax.axhline(y=threshold, color=color, linestyle=':', linewidth=1)
            except Exception:
                # if fitting fails, just skip
                pass
        # if use_sigmoidal_fit is False and show_runoff_time is True:
        # plot the threshold line
        if show_runoff_time and not use_sigmoidal_fit:
            threshold = threshold_percentage_for_runoff
            ax.axhline(y=threshold, color='k', linestyle=':', linewidth=1,)

    # Plot treatment line at zero
    if show_treatment_line:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1,
                    label=treatment_label)

    # Styling
    ax.set_xlabel("Time (min)", fontdict={'size': 16, 'color': 'black'})
    ax.set_ylabel(y_label, fontdict={'size': 16, 'color': 'black'})
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.set_ylim(*ylims)
    if xlims is not None:
        ax.set_xlim(*xlims)
    ax.legend(fontsize=10)
    plt.tight_layout()

    # Save & show
    plt.savefig(results_folder.joinpath(f'HT_{plot_name}.png'), dpi=600)
    plt.show()



def process_inhibitor_data(data_dir, inhibitor_frame_index, substring_in_data_dir='', selected_field='spot_int_ch_0', use_sem=True, show_summary=True, max_percentage_threshold_after_treatment=None, frame_rate_min=1,
                     frame_interval_sec=60, simulation_dna_sequence=None, inhibitor_delay_time_seconds=60, list_tag_sequences=[HA_TAG], ki_simulation=0.04, ke_simulation=4.5,
                     normalization_method='mean', percentile_range=(5, 95), verbose=False):
    """
    Process inhibitor runoff experiment data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing results subfolders with tracking CSV files.
    inhibitor_frame_index : int
        Frame index at which treatment starts.
    substring_in_data_dir : str
        Filter string to select specific subfolders.
    selected_field : str
        Column name for intensity values (e.g., 'spot_int_ch_0').
    use_sem : bool
        If True, use SEM for error bars; else use SD.
    show_summary : bool
        If True, print summary statistics.
    max_percentage_threshold_after_treatment : float or None
        Threshold (0-1) to classify responding vs non-responding cells.
    frame_rate_min : int
        Frame rate for downsampling (1 = every frame).
    frame_interval_sec : float
        Time interval between frames in seconds. Default 60 (1 min).
        Use 20 for 20-second intervals, etc.
    simulation_dna_sequence : str or None
        DNA sequence for TASEP simulation (optional).
    inhibitor_delay_time_seconds : float
        Delay time for drug to enter cell in simulation.
    list_tag_sequences : list
        Tag sequences for probe detection.
    ki_simulation : float
        Initiation rate for simulation.
    ke_simulation : float
        Elongation rate for simulation.
    normalization_method : str, optional
        'mean' (default) divides by pre-treatment mean intensity (values start ~1.0).
        'minmax' scales each cell's trajectory to [0, 1] range.
        'percentile' scales using percentile bounds (robust to outliers).
        None: no normalization.
    percentile_range : tuple, optional
        Percentile bounds for 'percentile' method. Default (5, 95).
        Use (1, 99) for wider range.
    verbose : bool, optional
        If True (default), print processing details and summary statistics.
        Set to False to suppress all print output.
        
    Returns
    -------
    tuple
        (responding_indices, time_min_recentered, intensities_normalized, array_particles,
         list_simulation_parameters, time_array_sim_min, mean_intensity_ssa_inh, err_intensity_ssa_inh)
    """

    list_dataframes = []
    subfolders = [folder for folder in data_dir.iterdir() if folder.is_dir()]
    subfolders = [folder for folder in subfolders if 'results_' in folder.name and substring_in_data_dir in folder.name]
    if verbose:
        print('List of processed dataframes:')
    for subfolder in subfolders:
        files = [f for f in subfolder.iterdir() if f.is_file() and 'tracking_' in f.name and f.suffix == '.csv']
        if not files:
            if verbose:
                print(f'Warning: No tracking CSV found in {subfolder.name}, skipping.')
            continue
        dfs = pd.read_csv(files[0])
        list_dataframes.append(dfs)
        if verbose:
            print('subfolder:', subfolder)

    # detect the maximum frame number across all dataframes
    max_frame = 0
    for df in list_dataframes:
        max_frame = max(max_frame, df['frame'].max())

    # terminate the program if the maximum frame is less than 1
    if max_frame < 10:
        if verbose:
            print('No dataframes found with frame number greater than 10 frames.')
        return None, None, None, None, None, None, None, None

    full_frames = np.arange(0, max_frame)
    frame_indices = np.arange(0, max_frame, frame_rate_min)  # Frame indices for processing
    array_particles = np.zeros((len(list_dataframes), len(frame_indices)))
    average_intensity = np.zeros((len(list_dataframes), len(frame_indices)))
    intensities_normalized = np.zeros((len(list_dataframes), len(frame_indices)))
    average_molecules_before_treatment = np.zeros(len(list_dataframes))
    for i, df in enumerate(list_dataframes):
        # Group by 'frame' and count unique particles
        particle_counts_per_frame = df.groupby('frame')['particle'].nunique()
        sum_intensities_per_frame = df.groupby('frame')[selected_field].sum()
        # Re-index to include frames with no particles (fill missing with 0)
        particle_counts_per_frame = particle_counts_per_frame.reindex(frame_indices, fill_value=0).values
        sum_intensities_per_frame = sum_intensities_per_frame.reindex(frame_indices, fill_value=0).values
        # Ensure the lengths match the max frame
        if len(particle_counts_per_frame) > max_frame:
            particle_counts_per_frame = particle_counts_per_frame[:max_frame]
            sum_intensities_per_frame = sum_intensities_per_frame[:max_frame]
        # Compute raw intensities (always use 'mean' per-cell first)
        intensities_normalized_before_treatment_intensity, average_intensity_with_respect_number_particles, average_particles_before_treatment = calculate_intensity(
            particle_counts_per_frame, sum_intensities_per_frame, inhibitor_frame_index,
            normalization_method=normalization_method
        )
        # Store in array
        normalized_particles, _ = calculate_number_of_particles_per_frame(particle_counts_per_frame, inhibitor_frame_index)
        array_particles[i] = normalized_particles
        average_intensity[i] = average_intensity_with_respect_number_particles
        intensities_normalized[i] = intensities_normalized_before_treatment_intensity
        average_molecules_before_treatment[i] = average_particles_before_treatment

    # Apply global normalization across all cells (after the loop)
    if normalization_method == 'minmax':
        global_min = intensities_normalized.min()
        global_max = intensities_normalized.max()
        if global_max - global_min > 0:
            intensities_normalized = (intensities_normalized - global_min) / (global_max - global_min)
        if verbose:
            print(f'Global minmax normalization applied (min={global_min:.4f}, max={global_max:.4f})')
    elif normalization_method == 'percentile':
        global_low = np.percentile(intensities_normalized, percentile_range[0])
        global_high = np.percentile(intensities_normalized, percentile_range[1])
        if global_high - global_low > 0:
            intensities_normalized = (intensities_normalized - global_low) / (global_high - global_low)
        if verbose:
            print(f'Global percentile normalization applied (P{percentile_range[0]}={global_low:.4f}, P{percentile_range[1]}={global_high:.4f})')

    treatment_start_index = inhibitor_frame_index #np.argmin(np.abs(full_frames - inhibitor_frame_index))
    non_responding_indices = []  # average post-treatment is not decreasing below the threshold
    responding_indices = []
    # Classify each trajectory
    if max_percentage_threshold_after_treatment is not None:
        for i, trajectory in enumerate(intensities_normalized):
            baseline = np.mean(trajectory[:treatment_start_index])
            threshold = max_percentage_threshold_after_treatment * baseline
            avg_post_treatment = np.mean(trajectory[treatment_start_index:])
            if avg_post_treatment >= threshold:
                non_responding_indices.append(i)
            else:
                responding_indices.append(i)
    else:
        # print a warning if the threshold is not provided
        if verbose:
            print('Warning: No threshold provided. All trajectories are considered responding.')
        # if no threshold is provided, all trajectories are considered responding
        responding_indices = list(range(len(intensities_normalized)))
        non_responding_indices = []

    if show_summary and verbose:
        # report percentage of non-responding trajectories
        print('---------Summary: ----------')
        print('\n----------Cells------------')
        # report total cells
        total_cells = len(intensities_normalized)
        print('total_cells: ', total_cells)
        number_of_responding_cells = len(responding_indices)
        print('number_of_responding_cells: ', number_of_responding_cells)
        number_of_non_responding_cells = len(non_responding_indices)
        print('number_of_non_responding_cells: ', number_of_non_responding_cells)
        percentage_non_responding_cells = number_of_non_responding_cells / (number_of_responding_cells + number_of_non_responding_cells) * 100
        print('percentage_non_responding_cells: ', np.round(percentage_non_responding_cells,1))
        print('\n----------RNA------------')
        total_molecules_before_treatment = average_molecules_before_treatment.sum()
        print('average_total_molecules_before_treatment: ', total_molecules_before_treatment)
        average_molecules_before_treatment_responding = average_molecules_before_treatment[responding_indices].sum()
        print('average_molecules_before_treatment_responding: ', np.round(average_molecules_before_treatment_responding,1))
        average_molecules_before_treatment_non_responding = average_molecules_before_treatment[non_responding_indices].sum()
        print('average_molecules_before_treatment_non_responding: ', average_molecules_before_treatment_non_responding)

    # Convert frame indices to time in minutes, centered on treatment
    time_min_recentered = (frame_indices - inhibitor_frame_index) * frame_interval_sec / 60.0


    if simulation_dna_sequence is not None:
        list_simulation_parameters = [ki_simulation, ke_simulation]
        ########################## Modeling  #########################################
        #inhibitor_delay_time_seconds = 60 # seconds. According to Tanenbaum paper 2016.
        number_repetitions = 100
        burnin_time = 2000
        #t_max = 360*5 #timePerturbationApplication + 25*60  # Maximum time
        t_max = max_frame * frame_interval_sec  # Total experiment time in seconds
        step_size_in_sec = 1
        time_array = np.arange(0, t_max, step_size_in_sec)
        #plot_name, rna, first_probe_position_vector, gene_length,list_param = model_and_data_selection(processed_data)
        _, rna, gene_length, first_probe_position_vector, second_probe_position_vector = read_gene_sequence_return_probes(simulation_dna_sequence, min_protein_length=50, list_tag_sequences=list_tag_sequences)
        ke = calculate_codon_elongation_rates (rna, global_elongation_rate=list_simulation_parameters[1],)
        timePerturbationApplication = inhibitor_frame_index * frame_interval_sec + inhibitor_delay_time_seconds
        evaluatingInhibitor = 1
        ssa_array = simulate_TASEP_SSA(list_simulation_parameters[0],
                                    ke,
                                    gene_length,
                                    t_max,
                                    time_interval_in_seconds=step_size_in_sec,
                                    number_repetitions=number_repetitions,
                                    first_probe_position_vector=first_probe_position_vector,
                                    timePerturbationApplication = timePerturbationApplication,
                                    inhibitor_effectiveness=95,
                                    evaluatingInhibitor=evaluatingInhibitor,
                                    burnin_time=burnin_time,
                                    constant_elongation_rate=list_simulation_parameters[1],
                                    fast_output=True)[2]
        # downsample time array and simulation output to match experimental frame interval
        downsample_factor = int(frame_interval_sec)  # e.g., 60 for 1 min, 20 for 20 sec
        time_array_downsampled = time_array[::downsample_factor]
        ssa_array_downsampled = ssa_array[:, ::downsample_factor]
        # plotting
        time_array_sim_min = time_array_downsampled/60
        normalized_data = np.zeros_like(ssa_array_downsampled)
        for i in range(ssa_array_downsampled.shape[0]):
            mean_before_treatment = np.mean(ssa_array_downsampled[i,:inhibitor_frame_index+1])
            normalized_data[i] = ssa_array_downsampled[i]/mean_before_treatment
        mean_intensity_ssa_inh = np.mean(normalized_data, axis=0)
        if use_sem:
            err_intensity_ssa_inh = np.std(normalized_data, axis=0) / np.sqrt(normalized_data.shape[0])
        else:
            err_intensity_ssa_inh = np.std(normalized_data, axis=0) #/np.sqrt(normalized_data.shape[0])
    else:
        mean_intensity_ssa_inh = None
        err_intensity_ssa_inh = None
        time_array_sim_min = None
        list_simulation_parameters = [None, None]

    return responding_indices, time_min_recentered, intensities_normalized, array_particles, list_simulation_parameters, time_array_sim_min, mean_intensity_ssa_inh, err_intensity_ssa_inh



def simulate_inhibitor(gene_sequence, ki=0.04, ke_global=5, use_sem=False, max_frame=20,
                       list_tag_sequences=None, inhibitor_frame=5, inhibitor_delay_seconds=60):
    """
    Run a TASEP simulation for a gene sequence with inhibitor treatment.

    This is a simulation-only function (no experimental data). It uses
    codon-usage-aware elongation rates and supports multi-tag probe detection.

    Parameters
    ----------
    gene_sequence : str
        DNA sequence to simulate.
    ki : float
        Initiation rate.
    ke_global : float
        Global elongation rate.
    use_sem : bool
        If True, error bars use SEM; else SD.
    max_frame : int
        Maximum number of frames (minutes) to simulate.
    list_tag_sequences : list or None
        Tag sequences for probe detection. Defaults to [HA_TAG, GFP_TAG].
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts.
    inhibitor_delay_seconds : float
        Delay time for drug to enter cell in simulation (seconds).

    Returns
    -------
    tuple
        (time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh)
    """
    if list_tag_sequences is None:
        list_tag_sequences = [HA_TAG, GFP_TAG]

    # reading the gene sequence
    protein, rna, _, indexes_tags, _, _ ,_ = read_sequence(seq=gene_sequence, min_protein_length=50, TAG=list_tag_sequences)
    gene_length = len(protein)
    tag_positions_first_probe_vector = indexes_tags[0]
    first_probe_position_vector = create_probe_vector(tag_positions_first_probe_vector, gene_length)

    # second probe vector.
    tag_positions_second_probe_vector = indexes_tags[1]
    second_probe_position_vector = create_probe_vector(tag_positions_second_probe_vector, gene_length)

    # print the first and second probe positions
    print(f"First probe positions: {tag_positions_first_probe_vector}")
    print(f"Second probe positions: {tag_positions_second_probe_vector}")

    ke_codon_dependent = calculate_codon_elongation_rates(rna, global_elongation_rate=ke_global)

    full_frames = np.arange(0, max_frame)
    full_frames = full_frames - inhibitor_frame

    ########################## Modeling  #########################################
    number_repetitions = 100
    burnin_time = 2000
    t_max = max_frame * 60
    step_size_in_sec = 1
    time_array = np.arange(0, t_max, step_size_in_sec)
    timePerturbationApplication = inhibitor_frame * 60 + inhibitor_delay_seconds
    evaluatingInhibitor = 1
    ssa_array = simulate_TASEP_SSA(ki,
                                ke_codon_dependent,
                                gene_length,
                                t_max,
                                time_interval_in_seconds=step_size_in_sec,
                                number_repetitions=number_repetitions,
                                first_probe_position_vector=first_probe_position_vector,
                                timePerturbationApplication=timePerturbationApplication,
                                inhibitor_effectiveness=95,
                                evaluatingInhibitor=evaluatingInhibitor,
                                burnin_time=burnin_time,
                                constant_elongation_rate=None,  # codon-usage aware
                                fast_output=False)[2]
    # downsample time array and simulation output to 1 minute resolution.
    time_array_downsampled = time_array[::60]
    ssa_array_downsampled = ssa_array[:, ::60]
    # plotting
    time_array_min = time_array_downsampled / 60
    normalized_data = np.zeros_like(ssa_array_downsampled)
    for i in range(ssa_array_downsampled.shape[0]):
        mean_before_treatment = np.mean(ssa_array_downsampled[i, :inhibitor_frame + 1])
        normalized_data[i] = ssa_array_downsampled[i] / mean_before_treatment
    mean_intensity_ssa_inh = np.mean(normalized_data, axis=0)

    if use_sem:
        err_intensity_ssa_inh = np.std(normalized_data, axis=0) / np.sqrt(normalized_data.shape[0])
    else:
        err_intensity_ssa_inh = np.std(normalized_data, axis=0)

    return time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh


def plot_inhibitor_simulation(legend_label, time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh,
                              figsize=(6, 3), ylims=(0, 1.4), inhibitor_frame=5):
    """
    Plot a single inhibitor simulation result.

    Parameters
    ----------
    legend_label : str
        Label for the simulation trace.
    time_array_min : np.ndarray
        Time array in minutes.
    mean_intensity_ssa_inh : np.ndarray
        Mean normalized intensity from simulation.
    err_intensity_ssa_inh : np.ndarray
        Error (SD or SEM) of normalized intensity.
    figsize : tuple
        Figure size.
    ylims : tuple
        Y-axis limits.
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts (for time offset).
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Plotting the model
    plt.plot(time_array_min - inhibitor_frame, mean_intensity_ssa_inh, '-', color='red', linewidth=4, label=legend_label)
    plt.fill_between(time_array_min - inhibitor_frame, mean_intensity_ssa_inh - err_intensity_ssa_inh,
                    mean_intensity_ssa_inh + err_intensity_ssa_inh, color='red', alpha=0.1)

    # Plot the inhibitor frame as a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel("Time (min)", fontdict={'size': 16})
    ax.set_ylabel("Norm. Intensity", fontdict={'size': 16})
    ax.tick_params(axis='both', which='major', labelsize=12)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.ylim(ylims)
    plt.tight_layout()
    ax.legend(fontsize=10)
    plt.show()


def plot_multiple_inhibitor_simulations(
    legend_labels,
    time_arrays_min,
    mean_intensities,
    err_intensities,
    figsize=(8, 5),
    ylims=(0, 1.5),
    inhibitor_frame=5
):
    """
    Overlay multiple inhibitor simulation results on a single plot.

    Parameters
    ----------
    legend_labels : str or list of str
        Labels for each simulation trace.
    time_arrays_min : np.ndarray or list of np.ndarray
        Time arrays in minutes.
    mean_intensities : np.ndarray or list of np.ndarray
        Mean normalized intensities from each simulation.
    err_intensities : np.ndarray or list of np.ndarray
        Errors (SD or SEM) for each simulation.
    figsize : tuple
        Figure size.
    ylims : tuple
        Y-axis limits.
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts (for time offset).
    """
    # --- normalize legend_labels to list ---
    if not isinstance(legend_labels, (list, tuple)):
        legend_labels = [legend_labels]
    n = len(legend_labels)

    # --- broadcast time_arrays_min if needed ---
    if not isinstance(time_arrays_min, (list, tuple)):
        time_arrays_min = [time_arrays_min] * n

    # --- similarly ensure mean_intensities and err_intensities are lists ---
    if not isinstance(mean_intensities, (list, tuple)):
        mean_intensities = [mean_intensities] * n
    if not isinstance(err_intensities, (list, tuple)):
        err_intensities = [err_intensities] * n

    # sanity check
    assert len(time_arrays_min) == n == len(mean_intensities) == len(err_intensities), \
        "All four inputs must be lists of the same length."

    # 1) create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # 2) choose a colormap and generate n distinct colors
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))

    # 3) plot each simulation with its assigned color
    for idx, (label, t_min, mean_inh, err_inh) in enumerate(zip(
        legend_labels, time_arrays_min, mean_intensities, err_intensities
    )):
        t_plot = np.array(t_min) - inhibitor_frame
        color = colors[idx]
        ax.plot(t_plot, mean_inh,
                '-', color=color, linewidth=3, label=label)
        ax.fill_between(
            t_plot,
            mean_inh - err_inh,
            mean_inh + err_inh,
            color=color,
            alpha=0.05
        )

    # vertical line at t=0
    ax.axvline(x=0, linestyle='--', color='k', linewidth=1)

    # axes labels and formatting
    ax.set_xlabel("Time (min)", size=16)
    ax.set_ylabel("Norm. Intensity", size=16)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_ylim(ylims)

    # legend and layout
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.show()
