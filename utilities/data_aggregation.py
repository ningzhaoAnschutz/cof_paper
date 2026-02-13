"""
Data aggregation function for MicroLive tracking data.
"""

import numpy as np
from .data_loading import extract_data_from_folders


def aggregate_folder_data(
    dataframes_dir,
    list_folder_substrings,
    list_folder_substring_to_avoid=None,
    list_folder_substring_second_condition=None,
    show_file_names=False,
    min_avg_spots_threshold=None,
    verbose=True,
):
    """
    Aggregate data from multiple result folders.
    
    For each substring in list_folder_substrings, calls extract_data_from_folders and
    builds up lists of metrics for channel 0 and channel 1.
    """
    # Prepare empty result lists
    list_directories = []
    list_int_ch_0, list_int_ch_1 = [], []
    list_snr_ch_0, list_snr_ch_1 = [], []
    list_spot_size_ch_0, list_spot_size_ch_1 = [], []
    list_total_spot_int_ch_0, list_total_spot_int_ch_1 = [], []
    list_spot_amplitude_ch_0, list_spot_amplitude_ch_1 = [], []
    list_spot_sigma_ch_0, list_spot_sigma_ch_1 = [], []
    list_number_spots, list_frames = [], []
    list_number_of_color_channels, list_average_number_spots = [], []
    list_efficiency_ml, list_efficiency_manual = [], []

    if list_folder_substring_to_avoid is None:
        list_folder_substring_to_avoid = [''] * len(list_folder_substrings)
    if list_folder_substring_second_condition is None:
        list_folder_substring_second_condition = [''] * len(list_folder_substrings)

    for i, subfolder in enumerate(list_folder_substrings):
        if verbose:
            print('-----------------------------------')
            print(f'Processing : {subfolder}')

        extracted = extract_data_from_folders(
            dataframes_dir, subfolder,
            list_folder_substring_to_avoid[i],
            list_folder_substring_second_condition[i],
            show_file_names=show_file_names, verbose=verbose,
        )

        num_ch = extracted.get('number_of_color_channels', 1)
        list_number_of_color_channels.append(num_ch)
        list_number_spots.append(extracted['number_of_spots'])
        list_frames.append(extracted['max_frame'])
        avg_spots = np.round(np.array(extracted['number_of_spots']) / (np.array(extracted['max_frame']) + 1), 2)
        list_average_number_spots.append(avg_spots)
        list_directories.append(extracted)

        # Channel 0
        list_int_ch_0.append(extracted.get('spot_int_ch_0_data'))
        list_snr_ch_0.append(extracted.get('snr_ch_0_data'))
        list_spot_size_ch_0.append(extracted.get('spot_size_ch_0_data'))
        list_total_spot_int_ch_0.append(extracted.get('total_spot_int_ch_0_data'))
        list_spot_amplitude_ch_0.append(extracted.get('psf_amplitude_ch_0_data'))
        list_spot_sigma_ch_0.append(extracted.get('psf_sigma_ch_0_data'))

        # Channel 1
        if num_ch > 1:
            list_int_ch_1.append(extracted.get('spot_int_ch_1_data'))
            list_snr_ch_1.append(extracted.get('snr_ch_1_data'))
            list_spot_size_ch_1.append(extracted.get('spot_size_ch_1_data'))
            list_total_spot_int_ch_1.append(extracted.get('total_spot_int_ch_1_data'))
            list_spot_amplitude_ch_1.append(extracted.get('psf_amplitude_ch_1_data'))
            list_spot_sigma_ch_1.append(extracted.get('psf_sigma_ch_1_data'))
        else:
            list_int_ch_1.append(None)
            list_snr_ch_1.append(None)
            list_spot_size_ch_1.append(None)
            list_total_spot_int_ch_1.append(None)
            list_spot_amplitude_ch_1.append(None)
            list_spot_sigma_ch_1.append(None)

        list_efficiency_ml.append(extracted['efficiency_cells'])
        list_efficiency_manual.append(extracted['efficiency_manual'])
        if verbose:
            print('-----------------------------------')
    
    # Apply filtering if specified
    if min_avg_spots_threshold is not None:
        if verbose:
            print(f"\n=== Cell Filtering (min_avg_spots_threshold={min_avg_spots_threshold}) ===")
        total_kept, total_excluded = 0, 0
        
        for cond_idx in range(len(list_folder_substrings)):
            avg_spots_data = list_average_number_spots[cond_idx]
            if avg_spots_data is None:
                continue
            
            keep_mask = [avg >= min_avg_spots_threshold if avg is not None and not np.isnan(avg) else False 
                        for avg in avg_spots_data]
            n_kept, n_excluded = sum(keep_mask), len(keep_mask) - sum(keep_mask)
            total_kept += n_kept
            total_excluded += n_excluded
            
            if verbose:
                print(f"  {list_folder_substrings[cond_idx]}: {n_kept}/{len(keep_mask)} cells kept")
            
            def filter_list(data_list):
                return [d for d, keep in zip(data_list, keep_mask) if keep] if data_list else None
            
            list_int_ch_0[cond_idx] = filter_list(list_int_ch_0[cond_idx])
            list_snr_ch_0[cond_idx] = filter_list(list_snr_ch_0[cond_idx])
            list_spot_size_ch_0[cond_idx] = filter_list(list_spot_size_ch_0[cond_idx])
            list_total_spot_int_ch_0[cond_idx] = filter_list(list_total_spot_int_ch_0[cond_idx])
            list_spot_amplitude_ch_0[cond_idx] = filter_list(list_spot_amplitude_ch_0[cond_idx])
            list_spot_sigma_ch_0[cond_idx] = filter_list(list_spot_sigma_ch_0[cond_idx])
            list_int_ch_1[cond_idx] = filter_list(list_int_ch_1[cond_idx])
            list_snr_ch_1[cond_idx] = filter_list(list_snr_ch_1[cond_idx])
            list_spot_size_ch_1[cond_idx] = filter_list(list_spot_size_ch_1[cond_idx])
            list_total_spot_int_ch_1[cond_idx] = filter_list(list_total_spot_int_ch_1[cond_idx])
            list_spot_amplitude_ch_1[cond_idx] = filter_list(list_spot_amplitude_ch_1[cond_idx])
            list_spot_sigma_ch_1[cond_idx] = filter_list(list_spot_sigma_ch_1[cond_idx])
            list_number_spots[cond_idx] = filter_list(list_number_spots[cond_idx])
            list_frames[cond_idx] = filter_list(list_frames[cond_idx])
            list_average_number_spots[cond_idx] = filter_list(list_average_number_spots[cond_idx].tolist()) if list_average_number_spots[cond_idx] is not None else None
            list_efficiency_ml[cond_idx] = filter_list(list_efficiency_ml[cond_idx])
            list_efficiency_manual[cond_idx] = filter_list(list_efficiency_manual[cond_idx])
        
        if verbose:
            print(f"  TOTAL: {total_kept}/{total_kept + total_excluded} cells kept\n")

    return {
        'directories': list_directories, 'int_ch_0': list_int_ch_0, 'int_ch_1': list_int_ch_1,
        'snr_ch_0': list_snr_ch_0, 'snr_ch_1': list_snr_ch_1,
        'spot_size_ch_0': list_spot_size_ch_0, 'spot_size_ch_1': list_spot_size_ch_1,
        'total_spot_int_ch_0': list_total_spot_int_ch_0, 'total_spot_int_ch_1': list_total_spot_int_ch_1,
        'spot_amplitude_ch_0': list_spot_amplitude_ch_0, 'spot_amplitude_ch_1': list_spot_amplitude_ch_1,
        'spot_sigma_ch_0': list_spot_sigma_ch_0, 'spot_sigma_ch_1': list_spot_sigma_ch_1,
        'number_spots': list_number_spots, 'frames': list_frames,
        'number_of_color_channels': list_number_of_color_channels,
        'average_number_spots': list_average_number_spots,
        'efficiency_ml': list_efficiency_ml, 'efficiency_manual': list_efficiency_manual,
    }
