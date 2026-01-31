from pathlib import Path
from microlive.imports import *
from microlive import microscopy as mi
current_dir = Path().resolve()
parent_dir = current_dir.parents[1]



def extract_data_from_tracking_df(list_of_tracking_dataframes, column_name):
    list_data = []
    for i, tracking_df in enumerate(list_of_tracking_dataframes):
        temp_data = tracking_df[column_name].values
        list_data.append(temp_data)
    mean_data = [np.round(np.nanmean(data),2) for data in list_data]
    median_data = [np.round(np.nanmedian(data),2) for data in list_data]
    std_data = [np.round(np.nanstd(data),2) for data in list_data]
    number_of_spots = [len(data) for data in list_data]
    print('Data for column: ', column_name)
    print('Mean :   ', mean_data)
    #print('Median : ',median_data)
    #print('std :    ',std_data)
    print('Number of spots:  ',number_of_spots)
    return mean_data, median_data, std_data, list_data, number_of_spots





def extract_data_from_folders (dataframes_dir, folder_substring, folder_substring_to_avoid = '', folder_substring_second_condition ='',  show_file_names = False):
    condition_folders = [folder for folder in dataframes_dir.iterdir() if folder_substring in folder.name]
    # check if the folder_substring_to_avoid is in the folder name and remove it from the list
    if folder_substring_to_avoid != '':
        condition_folders = [folder for folder in condition_folders if folder_substring_to_avoid not in folder.name]
    # check if the folder_substring_second_condition is in the folder name and remove it from the list
    if folder_substring_second_condition != '':
        condition_folders = [folder for folder in condition_folders if folder_substring_second_condition in folder.name]
    condition_folders = sorted(condition_folders, key = lambda x: x.name.split(' ')[-1])
    list_of_tracking_dataframes = []
    efficiency_cells = []
    efficiency_manual = []
    list_max_frame = []
    list_channel_counts = []  # Track channel counts per folder for validation
    for folder in condition_folders:
        temp_tracking_df =[]
        temp_efficiency_df = []
        temp_manual_efficiency_df = []
        tracking_files = [file for file in folder.iterdir() if file.suffix == '.csv' and 'tracking_' in file.stem]
        if not tracking_files:
            raise FileNotFoundError(f"No 'tracking_*.csv' file found in folder: {folder}")
        temp_tracking_df = pd.read_csv(tracking_files[0])
        list_of_tracking_dataframes.append(temp_tracking_df)
        # determine the number of color channels by checking the columns in the dataframe. If the column name contains 'ch', it is a color channel and the maximum value after 'ch_' is determined for all the columns.
        list_number_color_channels = [int(col.split('_')[-1]) for col in temp_tracking_df.columns if 'ch' in col]
        folder_channel_count = max(list_number_color_channels)+1
        list_channel_counts.append(folder_channel_count)
        # check if the efficiency file exists
        if any('colocalization_data' in file.stem for file in folder.iterdir()):
            temp_efficiency_df = [pd.read_csv(file) for file in folder.iterdir() if file.suffix == '.csv' and 'colocalization_data' in file.stem][0]
            efficiency_data = temp_efficiency_df['colocalization percentage'].values[0]
            efficiency_cells.append(efficiency_data)
        else:
            efficiency_data =  np.nan
            efficiency_cells.append(efficiency_data)

        if any('colocalization_manual' in file.stem for file in folder.iterdir()):
            temp_manual_efficiency_df = [pd.read_csv(file) for file in folder.iterdir() if file.suffix == '.csv' and 'colocalization_manual' in file.stem][0]
            efficiency_manual_data = temp_manual_efficiency_df['colocalization percentage'].values[0]
            efficiency_manual.append(efficiency_manual_data)
        else:
            efficiency_manual_data =  np.nan
            efficiency_manual.append(efficiency_manual_data)


        # determine the maximum value in 'frame' column in the dataframe.
        max_frame = temp_tracking_df['frame'].max()
        list_max_frame.append(max_frame)
        if show_file_names:
            print(f'Loaded {folder.name} ({folder_channel_count} channels)')

    # Validate that all folders have the same number of channels
    if len(set(list_channel_counts)) > 1:
        raise ValueError(f"Inconsistent channel counts across folders: {list_channel_counts}. "
                        f"Folders: {[f.name for f in condition_folders]}")
    number_color_channels = list_channel_counts[0] if list_channel_counts else 1

    list_datasets = ['spot_int', 'spot_size', 'total_spot_int','snr', 'psf_amplitude', 'psf_sigma']

    # create a dictionary to store the data for each dataset and for each channel.
    for dataset in list_datasets:
        # clean the temp_extracted_data_dict
        temp_extracted_data_dict = {}
        for channel in range(0, number_color_channels):
            selected_field = f'{dataset}_ch_{channel}'
            mean_data, median_data, std_data, list_data, number_of_spots = extract_data_from_tracking_df(list_of_tracking_dataframes, selected_field)
            # create an initial dictionary to store the data that is constant for each dataset
            if dataset == 'spot_int' and channel == 0:
                extracted_data_dict = {
                    'number_of_color_channels':number_color_channels,
                    'number_of_spots': number_of_spots,
                    'max_frame': list_max_frame,
                    'average_number_spots': np.round(np.array(number_of_spots) / np.array(list_max_frame), 2),
                    'efficiency_cells': efficiency_cells,
                    'efficiency_manual': efficiency_manual,
                }
            # create a dictionary to store the data for each dataset and for each channel
            temp_extracted_data_dict = {
                dataset+'_ch_'+str(channel)+'_mean': mean_data,
                dataset+'_ch_'+str(channel)+'_median': median_data,
                dataset+'_ch_'+str(channel)+'_std': std_data,
                dataset+'_ch_'+str(channel)+'_data': list_data,
            }
            # add the constant data to the dictionary
            extracted_data_dict.update(temp_extracted_data_dict)

    # Create a DataFrame from the dictionary
    return extracted_data_dict



def aggregate_folder_data(
    dataframes_dir,
    list_folder_substrings,
    list_folder_substring_to_avoid=None,
    list_folder_substring_second_condition =None,
    show_file_names=False
):
    """
    For each substring in list_folder_substrings, calls extract_data_from_folders and
    builds up lists of metrics for channel 0 and channel 1.
    If only one channel is present, channel 1 lists get None.
    Returns a dict of all the lists.
    """
    # Prepare empty result lists
    list_directories              = []
    list_int_ch_0                 = []
    list_int_ch_1                 = []
    list_snr_ch_0                 = []
    list_snr_ch_1                 = []
    list_spot_size_ch_0           = []
    list_spot_size_ch_1           = []
    list_total_spot_int_ch_0      = []
    list_total_spot_int_ch_1      = []
    list_spot_amplitude_ch_0      = []
    list_spot_amplitude_ch_1      = []
    list_spot_sigma_ch_0          = []
    list_spot_sigma_ch_1          = []
    list_number_spots             = []
    list_frames                   = []
    list_number_of_color_channels = []
    list_average_number_spots     = []
    list_efficiency_ml            = []
    list_efficiency_manual        = []

    if list_folder_substring_to_avoid is None:
        list_folder_substring_to_avoid = [''] * len(list_folder_substrings)
    if list_folder_substring_second_condition is None:
        list_folder_substring_second_condition = [''] * len(list_folder_substrings)

    for i, subfolder in enumerate(list_folder_substrings):
        print('-----------------------------------')
        print(f'Processing : {subfolder}')

        avoid  = list_folder_substring_to_avoid[i]
        second = list_folder_substring_second_condition[i]

        # call your existing helper
        extracted = extract_data_from_folders(
            dataframes_dir,
            subfolder,
            avoid,
            second,
            show_file_names=show_file_names
        )

        # read how many channels this folder actually has
        num_ch = extracted.get('number_of_color_channels', 1)
        list_number_of_color_channels.append(num_ch)

        # constants
        list_number_spots.append(extracted['number_of_spots'])
        list_frames.append(     extracted['max_frame'])
        avg_spots = np.round(
            np.array(extracted['number_of_spots']) /
            np.array(extracted['max_frame']),
            2
        )
        list_average_number_spots.append(avg_spots)
        list_directories.append(extracted)

        # channel 0 always exists
        list_int_ch_0.append(           extracted.get('spot_int_ch_0_data'))
        list_snr_ch_0.append(           extracted.get('snr_ch_0_data'))
        list_spot_size_ch_0.append(     extracted.get('spot_size_ch_0_data'))
        list_total_spot_int_ch_0.append(extracted.get('total_spot_int_ch_0_data'))
        list_spot_amplitude_ch_0.append(extracted.get('psf_amplitude_ch_0_data'))
        list_spot_sigma_ch_0.append(    extracted.get('psf_sigma_ch_0_data'))

        # channel 1 only if num_ch > 1
        if num_ch > 1:
            list_int_ch_1.append(           extracted.get('spot_int_ch_1_data'))
            list_snr_ch_1.append(           extracted.get('snr_ch_1_data'))
            list_spot_size_ch_1.append(     extracted.get('spot_size_ch_1_data'))
            list_total_spot_int_ch_1.append(extracted.get('total_spot_int_ch_1_data'))
            list_spot_amplitude_ch_1.append(extracted.get('psf_amplitude_ch_1_data'))
            list_spot_sigma_ch_1.append(    extracted.get('psf_sigma_ch_1_data'))
        else:
            # pad channel 1 lists with None
            list_int_ch_1.append(           None)
            list_snr_ch_1.append(           None)
            list_spot_size_ch_1.append(     None)
            list_total_spot_int_ch_1.append(None)
            list_spot_amplitude_ch_1.append(None)
            list_spot_sigma_ch_1.append(    None)

        # efficiency
        list_efficiency_ml.append(extracted['efficiency_cells'])
        list_efficiency_manual.append(extracted['efficiency_manual'])

        print('-----------------------------------')

    return {
        'directories'              : list_directories,
        'int_ch_0'                 : list_int_ch_0,
        'int_ch_1'                 : list_int_ch_1,
        'snr_ch_0'                 : list_snr_ch_0,
        'snr_ch_1'                 : list_snr_ch_1,
        'spot_size_ch_0'           : list_spot_size_ch_0,
        'spot_size_ch_1'           : list_spot_size_ch_1,
        'total_spot_int_ch_0'      : list_total_spot_int_ch_0,
        'total_spot_int_ch_1'      : list_total_spot_int_ch_1,
        'spot_amplitude_ch_0'      : list_spot_amplitude_ch_0,
        'spot_amplitude_ch_1'      : list_spot_amplitude_ch_1,
        'spot_sigma_ch_0'          : list_spot_sigma_ch_0,
        'spot_sigma_ch_1'          : list_spot_sigma_ch_1,
        'number_spots'             : list_number_spots,
        'frames'                   : list_frames,
        'number_of_color_channels' : list_number_of_color_channels,
        'average_number_spots'     : list_average_number_spots,
        'efficiency_ml'            : list_efficiency_ml,
        'efficiency_manual'        : list_efficiency_manual,

    }




def plot_swarm_plot(
    conditions_data, # list of np.ndarray: each element is a numpy array of repetition data for a condition
    condition_labels,  # list of str: label for each condition (x-axis category)
    x_label="Condition",
    y_label="Mean Value",
    title="",
    figsize=(6, 3),
    tick_size=16,
    swarm_color="black",
    y_lim=None,
    show_stats=False,
    only_significant=True,
    save_dir=None,  # Path or str; if provided, the plot will be saved in this directory
    plot_name = 'temp',
    max_percentile_significance=99.5,
    x_tick_rotation=0,  # Rotation angle for x-axis labels (0=horizontal, 45=diagonal, 90=vertical)
):
    """
    Calculate the mean of each repetition within each condition and create a boxplot
    with whiskers overlaid by a swarm plot. Optionally, perform pairwise statistical
    comparisons between conditions and add significance bars.

    Each element in conditions_data should be an iterable (e.g., list) of NumPy arrays,
    where each array represents one repetition of a given condition.

    Parameters
    ----------
    conditions_data : list
        A list where each element corresponds to a condition. Each condition is an iterable of
        NumPy arrays representing repetition data.
    condition_labels : list of str
        Labels for each condition that will appear on the x-axis.
    x_label : str, optional
        Label for the x-axis (default "Condition").
    y_label : str, optional
        Label for the y-axis (default "Mean Value").
    title : str, optional
        Plot title (default is an empty string).
    figsize : tuple, optional
        Figure size in inches (default (6, 4)).
    tick_size : int, optional
        Font size for tick labels (default 16).
    swarm_color : str, optional
        Color for the swarmplot points (default "black").
    y_lim : tuple, optional
        y-axis limits as a tuple (min, max); if provided, applied via plt.ylim().
    show_stats : bool, optional
        If True, pairwise significance bars (using the Mann–Whitney U test) are added.
    only_significant : bool, optional
        If True, only significance bars for comparisons with p < 0.05 are plotted.
    save_dir : Path or str, optional
        Directory in which to save the plot (as PNG and SVG). If None, the plot is not saved.
    plot_name : str, optional
        Base filename for saved plots (without extension).
    max_percentile_significance : float, optional
        Percentile of the data to use as baseline for significance bars.

    Returns
    -------
    ax : matplotlib Axes object
        The Axes object containing the plot.
    """

    # Set Seaborn style.
    sns.set_style("ticks")

    # Calculate repetition means for each condition.
    mean_values = []
    condition_list = []
    for cond_idx, repetitions in enumerate(conditions_data):
        for rep in repetitions:
            rep_mean = np.nanmean(rep)
            mean_values.append(rep_mean)
            condition_list.append(condition_labels[cond_idx])

    # Create a DataFrame with the calculated means.
    df = pd.DataFrame({
        "Mean": mean_values,
        "Condition": condition_list
    })

    # Create the figure with a white background.
    plt.figure(figsize=figsize, facecolor='white')
    ax = sns.boxplot(
        x="Condition",
        y="Mean",
        data=df,
        order=condition_labels,
        showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black'},
        medianprops={'color': 'red'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        linewidth=1.5,
        whis=[5, 95],
        width=0.5,
    )
    ax.set_facecolor('white')

    # Overlay the swarmplot.
    sns.swarmplot(
        x="Condition",
        y="Mean",
        data=df,
        order=condition_labels,
        color=swarm_color,
        size=5,
    )

    # Set labels and title using Arial font with the specified tick size.
    plt.xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    plt.ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    plt.title(title, fontsize=tick_size + 4, fontname="Arial", color='black')

    if y_lim is not None and not show_stats:
        plt.ylim(y_lim)

    ax.tick_params(axis='x', labelsize=tick_size + 4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.xticks(fontname="Arial", rotation=x_tick_rotation, ha='right' if x_tick_rotation != 0 else 'center')
    plt.yticks(fontname="Arial")

    plt.tight_layout()

    # Add pairwise statistical significance bars if requested.
    if show_stats and len(condition_labels) > 1:
        # Compute global max to set the baseline for significance bars.
        global_max = np.nanpercentile(df["Mean"].to_numpy(), max_percentile_significance)
        global_min = np.nanmin(df["Mean"].to_numpy())
        global_range = global_max - global_min if (global_max - global_min) != 0 else 1
        offset = 0.1 * global_range   # vertical offset for each new bar
        bar_height = 0.02 * global_range
        k = 0  # counter to determine vertical positioning
        num_conditions = len(condition_labels)
        for i in range(num_conditions - 1):
            for j in range(i + 1, num_conditions):
                group1 = df[df["Condition"] == condition_labels[i]]["Mean"].dropna()
                group2 = df[df["Condition"] == condition_labels[j]]["Mean"].dropna()
                if len(group1) > 0 and len(group2) > 0:
                    stat, p = mannwhitneyu(group1, group2)
                else:
                    p = np.nan

                print(f"Comparing {condition_labels[i]} vs {condition_labels[j]}: p-value = {p:.4f}")
                # Determine significance marker.
                if p < 0.0001:
                    sig = '****'
                elif p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'

                # Skip non-significant comparisons if only_significant is True
                if only_significant and sig == 'ns':
                    continue

                x1 = i
                x2 = j
                y_line = global_max + offset * (k + 1)
                # Draw the significance bar.
                ax.plot([x1, x1, x2, x2],
                        [y_line, y_line + bar_height, y_line + bar_height, y_line],
                        lw=1.5, c='k')
                # Add the significance text.
                ax.text((x1 + x2) * 0.5, y_line + bar_height,
                        sig, ha='center', va='bottom',
                        color='k', fontsize=tick_size, fontname="Arial")
                k += 1

    # Save the plot as PNG and SVG if save_dir is provided.
    if save_dir is not None:
        plt.savefig(save_dir.joinpath(plot_name + ".png"), dpi=600, bbox_inches='tight')
        plt.savefig(save_dir.joinpath(plot_name + ".svg"), dpi=600, bbox_inches='tight')

    plt.show()
    return ax


def plot_swarm_plo_efficiency(
    conditions_data_ch0,  # list of condition data; each element is an iterable of NumPy arrays (repetitions) for channel 0
    conditions_data_ch1,  # list of condition data; each element is an iterable of NumPy arrays (repetitions) for channel 1
    condition_labels,     # list of str: label for each condition (x-axis category)
    x_label="Condition",
    y_label="Efficiency (%)",
    title="",
    figsize=(6, 3),
    tick_size=16,
    swarm_color="black",
    y_lim=None,
    show_stats=False,
    only_significant=True,
    save_dir=None,        # Path or str; if provided, the plot will be saved in this directory
    plot_name='temp',
    max_percentile_significance=99.5,
    threshold_ch0=0.5,
    threshold_ch1=0.5,
):


    """
    Calculate the efficiency for each repetition within each condition and create a boxplot
    with whiskers overlaid by a swarm plot. The efficiency is defined as the percentage of
    spots where the value in channel 0 exceeds threshold_ch0 and the value in channel 1 exceeds
    threshold_ch1, relative to the total number of spots (from channel 0). Optionally, perform
    pairwise statistical comparisons between conditions and add significance bars.

    Parameters
    ----------
    conditions_data_ch0 : list
        A list where each element corresponds to a condition. Each condition is an iterable
        of NumPy arrays representing repetition data for channel 0.
    conditions_data_ch1 : list
        A list where each element corresponds to a condition. Each condition is an iterable
        of NumPy arrays representing repetition data for channel 1.
    condition_labels : list of str
        Labels for each condition that will appear on the x-axis.
    x_label : str, optional
        Label for the x-axis (default "Condition").
    y_label : str, optional
        Label for the y-axis (default "Efficiency (%)").
    title : str, optional
        Plot title (default is an empty string).
    figsize : tuple, optional
        Figure size in inches (default (6, 3)).
    tick_size : int, optional
        Font size for tick labels (default 16).
    swarm_color : str, optional
        Color for the swarmplot points (default "black").
    y_lim : tuple, optional
        y-axis limits as a tuple (min, max); if provided, applied via plt.ylim().
    show_stats : bool, optional
        If True, pairwise significance bars (using the Mann–Whitney U test) are added.
    only_significant : bool, optional
        If True, only bars for comparisons with p < 0.05 are plotted.
    save_dir : Path or str, optional
        Directory in which to save the plot (as PNG and SVG). If None, the plot is not saved.
    plot_name : str, optional
        The filename (without extension) to save the plot under (default 'temp').
    max_percentile_significance : float, optional
        The percentile used to compute the baseline for significance bars (default 99.5).
    threshold_ch0 : float, optional
        Threshold for channel 0 data (default 0.5).
    threshold_ch1 : float, optional
        Threshold for channel 1 data (default 0.5).
    """
    sns.set_style("ticks")

    # Calculate efficiency for each repetition in each condition.
    efficiency_values = []
    condition_list = []
    for cond_idx, reps_ch0 in enumerate(conditions_data_ch0):
        reps_ch1 = conditions_data_ch1[cond_idx]
        for rep_idx in range(len(reps_ch0)):
            data_ch0 = np.asarray(reps_ch0[rep_idx]).flatten()
            data_ch0 = data_ch0[~np.isnan(data_ch0)]
            data_ch1 = np.asarray(reps_ch1[rep_idx]).flatten()
            data_ch1 = data_ch1[~np.isnan(data_ch1)]
            total_points = len(data_ch0)
            if total_points > 0:
                eff = np.sum((data_ch0 > threshold_ch0) & (data_ch1 > threshold_ch1)) / total_points * 100
            else:
                eff = np.nan
            efficiency_values.append(eff)
            condition_list.append(condition_labels[cond_idx])

    df = pd.DataFrame({"Efficiency": efficiency_values, "Condition": condition_list})

    plt.figure(figsize=figsize, facecolor='white')
    ax = sns.boxplot(
        x="Condition", y="Efficiency", data=df, order=condition_labels,
        showfliers=False, boxprops={'facecolor': 'white','edgecolor': 'black'},
        medianprops={'color': 'red'}, whiskerprops={'color': 'black'},
        capprops={'color': 'black'}, linewidth=1.5, whis=[5, 95], width=0.5,
    )
    ax.set_facecolor('white')
    sns.swarmplot(
        x="Condition", y="Efficiency", data=df, order=condition_labels,
        color=swarm_color, size=5,
    )
    plt.xlabel(x_label, fontsize=tick_size+4, fontname="Arial", color='black')
    plt.ylabel(y_label, fontsize=tick_size+4, fontname="Arial", color='black')
    plt.title(title, fontsize=tick_size+4, fontname="Arial", color='black')
    if y_lim is not None and not show_stats:
        plt.ylim(y_lim)
    ax.tick_params(axis='x', labelsize=tick_size+4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.xticks(fontname="Arial")
    plt.yticks(fontname="Arial")
    plt.tight_layout()

    if show_stats and len(condition_labels) > 1:
        global_max = np.nanpercentile(df["Efficiency"], max_percentile_significance)
        global_min = np.nanmin(df["Efficiency"])
        global_range = global_max - global_min if (global_max - global_min) != 0 else 1
        offset = 0.1 * global_range
        bar_height = 0.02 * global_range
        k = 0
        num_conditions = len(condition_labels)
        for i in range(num_conditions - 1):
            for j in range(i + 1, num_conditions):
                group1 = df[df["Condition"] == condition_labels[i]]["Efficiency"].dropna()
                group2 = df[df["Condition"] == condition_labels[j]]["Efficiency"].dropna()
                if len(group1) and len(group2):
                    stat, p = mannwhitneyu(group1, group2)
                else:
                    p = np.nan
                if p < 0.0001:
                    sig = '****'
                elif p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                # Skip non-significant if requested
                if only_significant and sig == 'ns':
                    continue
                x1, x2 = i, j
                y_line = global_max + offset * (k + 1)
                ax.plot([x1, x1, x2, x2], [y_line, y_line+bar_height, y_line+bar_height, y_line], lw=1.5, c='k')
                ax.text((x1+x2)*0.5, y_line+bar_height, sig,
                        ha='center', va='bottom', fontsize=tick_size, fontname="Arial")
                k += 1

    if save_dir is not None:
        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir.joinpath(plot_name + ".png"), dpi=600, bbox_inches='tight')
        plt.savefig(save_dir.joinpath(plot_name + ".svg"), dpi=600, bbox_inches='tight')

    plt.show()
    return ax


def plot_efficiency_vs_intensity_scatter(
    data_dict,          # dictionary returned by aggregate_folder_data
    condition_labels,   # list of str: labels for each condition
    x_label="Mean Spot Intensity (Ch0)",
    y_label="CoF Efficiency (%)",
    title="",
    figsize=(8, 6),
    tick_size=12,
    save_dir=None,
    plot_name='efficiency_vs_intensity',
):
    """
    Create a scatter plot comparing per-cell CoF efficiency vs per-cell mean spot intensity.
    
    Uses pre-calculated efficiency from MicroLive (efficiency_ml) and mean spot intensity (int_ch_0).
    Each condition is plotted in a different color.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary returned by aggregate_folder_data containing:
        - 'int_ch_0': list of conditions, each is list of cells, each is array of spot intensities
        - 'efficiency_ml': list of conditions, each is list of efficiency values per cell
    condition_labels : list of str
        Labels for each condition
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    figsize : tuple
        Figure size
    tick_size : int
        Font size for ticks
    save_dir : Path or str, optional
        Directory to save the plot
    plot_name : str
        Filename for saved plot
    
    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    sns.set_style("ticks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Get colormap for conditions
    colors = cm.tab10(np.linspace(0, 1, len(condition_labels)))
    
    # Collect all data for R² calculation
    all_intensities = []
    all_efficiencies = []
    
    # Data for each condition
    for cond_idx, label in enumerate(condition_labels):
        # Get data for this condition
        int_ch0_data = data_dict['int_ch_0'][cond_idx]
        efficiency_data = data_dict['efficiency_ml'][cond_idx]
        
        if int_ch0_data is None or efficiency_data is None:
            continue
        
        # Calculate per-cell mean intensities
        cell_intensities = []
        cell_efficiencies = []
        
        for cell_idx in range(len(int_ch0_data)):
            # Mean intensity for this cell
            intensities = np.asarray(int_ch0_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            
            if len(intensities) == 0:
                continue
            
            mean_int = np.nanmean(intensities)
            
            # Get pre-calculated efficiency for this cell
            if cell_idx < len(efficiency_data):
                eff = efficiency_data[cell_idx]
                if np.isnan(eff):
                    continue
            else:
                continue
            
            cell_intensities.append(mean_int)
            cell_efficiencies.append(eff)
            all_intensities.append(mean_int)
            all_efficiencies.append(eff)
        
        # Plot this condition
        ax.scatter(
            cell_intensities, 
            cell_efficiencies, 
            c=[colors[cond_idx]], 
            label=label,
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
        )
    
    # Calculate R² and add trend line
    if len(all_intensities) > 1:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_intensities, all_efficiencies)
        r_squared = r_value ** 2
        
        # Trend line
        x_line = np.linspace(min(all_intensities), max(all_intensities), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', lw=1.5, alpha=0.7, label=f'Trend (R² = {r_squared:.3f})')
        
        # R² annotation
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}\np = {p_value:.2e}', 
                transform=ax.transAxes, fontsize=tick_size, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and styling
    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial")
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.legend(loc='upper right', fontsize=tick_size - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir.joinpath(plot_name + ".png"), dpi=600, bbox_inches='tight')
        plt.savefig(save_dir.joinpath(plot_name + ".svg"), dpi=600, bbox_inches='tight')
    
    plt.show()
    return ax

def plot_efficiency_vs_intensity_scatter_means(
    data_dict,          # dictionary returned by aggregate_folder_data
    condition_labels,   # list of str: labels for each condition
    x_label="Mean Spot Intensity (Ch0)",
    y_label="CoF Efficiency (%)",
    title="",
    figsize=(8, 6),
    tick_size=12,
    marker_size=150,
    show_individual_cells=False,  # Show individual cell measurements as small dots in background
    individual_cell_alpha=0.3,    # Transparency of individual cell dots
    individual_cell_size=20,      # Size of individual cell dots
    save_dir=None,
    plot_name='efficiency_vs_intensity_means',
):
    """
    Create a scatter plot comparing per-condition mean CoF efficiency vs mean spot intensity.
    
    Each condition is shown as a single point (mean of all cells) with 2D error bars 
    representing standard deviation in both X (intensity) and Y (efficiency) directions.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary returned by aggregate_folder_data containing:
        - 'int_ch_0': list of conditions, each is list of cells, each is array of spot intensities
        - 'efficiency_ml': list of conditions, each is list of efficiency values per cell
    condition_labels : list of str
        Labels for each condition
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    figsize : tuple
        Figure size
    tick_size : int
        Font size for ticks
    marker_size : int
        Size of scatter points
    show_individual_cells : bool
        If True, show individual cell measurements as small dots in background
    individual_cell_alpha : float
        Transparency of individual cell dots (0-1)
    individual_cell_size : int
        Size of individual cell dots
    save_dir : Path or str, optional
        Directory to save the plot
    plot_name : str
        Filename for saved plot
    
    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    sns.set_style("ticks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Publication-quality color palette (vibrant and distinct)
    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#17becf',  # Cyan
        '#bcbd22',  # Olive
        '#7f7f7f',  # Gray
    ]
    colors = [color_palette[i % len(color_palette)] for i in range(len(condition_labels))]
    
    # Store mean values for R² calculation
    all_mean_intensities = []
    all_mean_efficiencies = []
    
    # First pass: collect all individual cell data and plot as small dots (if enabled)
    all_cell_data = []  # Store for potential background plotting
    
    # Data for each condition
    for cond_idx, label in enumerate(condition_labels):
        # Get data for this condition
        int_ch0_data = data_dict['int_ch_0'][cond_idx]
        efficiency_data = data_dict['efficiency_ml'][cond_idx]
        
        if int_ch0_data is None or efficiency_data is None:
            continue
        
        # Calculate per-cell mean intensities
        cell_intensities = []
        cell_efficiencies = []
        
        for cell_idx in range(len(int_ch0_data)):
            # Mean intensity for this cell
            intensities = np.asarray(int_ch0_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            
            if len(intensities) == 0:
                continue
            
            mean_int = np.nanmean(intensities)
            
            # Get pre-calculated efficiency for this cell
            if cell_idx < len(efficiency_data):
                eff = efficiency_data[cell_idx]
                if np.isnan(eff):
                    continue
            else:
                continue
            
            cell_intensities.append(mean_int)
            cell_efficiencies.append(eff)
        
        if len(cell_intensities) == 0:
            continue
        
        # Plot individual cells as small dots in background (if enabled)
        if show_individual_cells:
            ax.scatter(
                cell_intensities,
                cell_efficiencies,
                c=colors[cond_idx],
                s=individual_cell_size,
                alpha=individual_cell_alpha,
                edgecolors='none',
                zorder=1,  # Behind the mean points
            )
        
        # Calculate condition mean and SEM
        n_cells = len(cell_intensities)
        mean_intensity = np.mean(cell_intensities)
        sem_intensity = np.std(cell_intensities) / np.sqrt(n_cells)
        mean_efficiency = np.mean(cell_efficiencies)
        sem_efficiency = np.std(cell_efficiencies) / np.sqrt(n_cells)
        
        all_mean_intensities.append(mean_intensity)
        all_mean_efficiencies.append(mean_efficiency)
        
        # Plot point with 2D error bars (on top)
        ax.errorbar(
            mean_intensity, 
            mean_efficiency,
            xerr=sem_intensity,
            yerr=sem_efficiency,
            fmt='o',
            color=colors[cond_idx],
            markersize=np.sqrt(marker_size),
            capsize=5,
            capthick=2.5,
            elinewidth=2.5,
            label=label,
            markeredgecolor='white',
            markeredgewidth=1,
            zorder=2,  # On top of individual dots
        )
    
    # Calculate R² if enough points
    if len(all_mean_intensities) >= 2:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_mean_intensities, all_mean_efficiencies)
        r_squared = r_value ** 2
        
        # Trend line
        x_line = np.linspace(min(all_mean_intensities), max(all_mean_intensities), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', lw=1.5, alpha=0.7)
        
        # R² annotation
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}\np = {p_value:.2e}', 
                transform=ax.transAxes, fontsize=tick_size + 2, va='top', ha='left',
                fontname="Arial", color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Labels and styling (matching plot_swarm_plot)
    ax.set_xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.set_ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.tick_params(axis='x', labelsize=tick_size + 4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size + 4, colors='black')
    plt.xticks(fontname="Arial")
    plt.yticks(fontname="Arial")
    ax.legend(loc='lower right', fontsize=tick_size, frameon=True, facecolor='white', edgecolor='black')
    # Box around plot (all 4 spines visible)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir.joinpath(plot_name + ".png"), dpi=600, bbox_inches='tight')
        plt.savefig(save_dir.joinpath(plot_name + ".svg"), dpi=600, bbox_inches='tight')
    
    plt.show()
    return ax
