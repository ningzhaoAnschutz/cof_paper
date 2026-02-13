"""
Data loading and aggregation utilities for MicroLive tracking data.

This module provides functions for:
- Extracting folder substrings and names from experiment directories
- Loading tracking DataFrames from result folders
- Aggregating data across multiple conditions/experiments
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from .config import REPORTER_PLASMID_NAME_MAPPING, PLASMID_SHORT_NAME_MAPPING


def get_folder_substrings_and_names(
    dataframes_dir,
    process_individual_days=False,
    plasmid_name_mapping=None,
    plasmid_order=None,
    verbose=True
):
    """
    Automatically extract list_folder_substrings and list_names from a data directory.
    
    Scans the results folders in dataframes_dir, identifies plasmid names and dates
    from folder names, and returns appropriate lists for data aggregation.
    """
    if isinstance(dataframes_dir, str):
        dataframes_dir = Path(dataframes_dir)
    
    if plasmid_name_mapping is None:
        plasmid_name_mapping = REPORTER_PLASMID_NAME_MAPPING
    
    if plasmid_order is None:
        plasmid_order = list(plasmid_name_mapping.keys())
    
    # Get all result folders
    result_folders = [folder for folder in dataframes_dir.iterdir() 
                      if folder.is_dir() and folder.name.startswith('results_')]
    
    if not result_folders:
        raise ValueError(f"No results folders found in {dataframes_dir}")
    
    # Collect all unique date+plasmid combinations
    date_plasmid_combinations = {}
    unrecognized_folders = []
    
    for folder in result_folders:
        folder_name = folder.name
        date_match = re.match(r'results_(\d{7,8})', folder_name)
        date = date_match.group(1) if date_match else None
        
        plasmid = None
        for p in plasmid_order:
            if p in folder_name:
                plasmid = p
                break
        
        if plasmid is None:
            unrecognized_folders.append(folder_name)
            continue
        
        name_without_prefix = folder_name[8:]
        plasmid_idx = name_without_prefix.find(plasmid)
        if plasmid_idx >= 0:
            full_prefix = name_without_prefix[:plasmid_idx + len(plasmid)]
        elif date:
            full_prefix = f"{date} {plasmid}"
        else:
            full_prefix = plasmid
        
        if plasmid not in date_plasmid_combinations:
            date_plasmid_combinations[plasmid] = []
        
        if (date, full_prefix) not in [(d, p) for d, p in date_plasmid_combinations[plasmid]]:
            date_plasmid_combinations[plasmid].append((date, full_prefix))
    
    for plasmid in date_plasmid_combinations:
        date_plasmid_combinations[plasmid] = sorted(
            date_plasmid_combinations[plasmid], 
            key=lambda x: x[0] if x[0] else ''
        )
    
    if verbose:
        print(f"Found {len(date_plasmid_combinations)} plasmids in {len(result_folders)} folders")
        for plasmid, combos in date_plasmid_combinations.items():
            print(f"  {plasmid}: {len(combos)} experiment(s)")
        if unrecognized_folders:
            print(f"\n⚠️  {len(unrecognized_folders)} folder(s) NOT RECOGNIZED")
    
    if not process_individual_days:
        list_folder_substrings = []
        list_names = []
        for plasmid in plasmid_order:
            if plasmid in date_plasmid_combinations:
                list_folder_substrings.append(plasmid)
                list_names.append(plasmid_name_mapping.get(plasmid, plasmid))
        return list_folder_substrings, list_names
    else:
        list_folder_substrings = []
        list_names = []
        plasmid_counts = {}
        for plasmid in plasmid_order:
            if plasmid not in date_plasmid_combinations:
                continue
            short_name = PLASMID_SHORT_NAME_MAPPING.get(plasmid, plasmid[:4])
            if short_name not in plasmid_counts:
                plasmid_counts[short_name] = 0
            for date, full_prefix in date_plasmid_combinations[plasmid]:
                plasmid_counts[short_name] += 1
                list_folder_substrings.append(full_prefix)
                list_names.append(f"{short_name}-{plasmid_counts[short_name]}")
        return list_folder_substrings, list_names


def extract_data_from_tracking_df(list_of_tracking_dataframes, column_name, verbose=True):
    """Extract data from a list of tracking DataFrames for a specific column."""
    list_data = []
    for tracking_df in list_of_tracking_dataframes:
        temp_data = tracking_df[column_name].values
        list_data.append(temp_data)
    mean_data = [np.round(np.nanmean(data), 2) for data in list_data]
    median_data = [np.round(np.nanmedian(data), 2) for data in list_data]
    std_data = [np.round(np.nanstd(data), 2) for data in list_data]
    number_of_spots = [len(data) for data in list_data]
    if verbose:
        print('Data for column: ', column_name)
        print('Mean :   ', mean_data)
        print('Number of spots:  ', number_of_spots)
    return mean_data, median_data, std_data, list_data, number_of_spots


def extract_data_from_folders(
    dataframes_dir, folder_substring, folder_substring_to_avoid='',
    folder_substring_second_condition='', show_file_names=False, verbose=True
):
    """Extract tracking data and efficiency metrics from result folders."""
    condition_folders = [f for f in dataframes_dir.iterdir() 
                        if f.is_dir() and folder_substring in f.name]
    if folder_substring_to_avoid:
        condition_folders = [f for f in condition_folders if folder_substring_to_avoid not in f.name]
    if folder_substring_second_condition:
        condition_folders = [f for f in condition_folders if folder_substring_second_condition in f.name]
    condition_folders = sorted(condition_folders, key=lambda x: x.name.split(' ')[-1])
    
    list_of_tracking_dataframes = []
    efficiency_cells, efficiency_manual = [], []
    list_max_frame, list_channel_counts = [], []
    
    for folder in condition_folders:
        csv_files = [f for f in folder.iterdir() if f.suffix == '.csv']
        tracking_files = [f for f in csv_files if 'tracking_' in f.stem]
        if not tracking_files:
            raise FileNotFoundError(f"No 'tracking_*.csv' file found in folder: {folder}")
        
        temp_tracking_df = pd.read_csv(tracking_files[0], encoding='latin-1')
        list_of_tracking_dataframes.append(temp_tracking_df)
        
        ch_cols = [int(col.split('_')[-1]) for col in temp_tracking_df.columns if 'ch' in col]
        folder_channel_count = max(ch_cols) + 1
        list_channel_counts.append(folder_channel_count)
        
        coloc_files = [f for f in csv_files if 'colocalization_data' in f.stem]
        if coloc_files:
            eff_df = pd.read_csv(coloc_files[0], encoding='latin-1')
            efficiency_cells.append(eff_df['colocalization percentage'].values[0])
        else:
            efficiency_cells.append(np.nan)
        
        manual_files = [f for f in csv_files if 'colocalization_manual' in f.stem]
        if manual_files:
            manual_df = pd.read_csv(manual_files[0], encoding='latin-1')
            efficiency_manual.append(manual_df['colocalization percentage'].values[0])
        else:
            efficiency_manual.append(np.nan)
        
        list_max_frame.append(temp_tracking_df['frame'].max())
        if show_file_names:
            print(f'Loaded {folder.name} ({folder_channel_count} channels)')
    
    if len(set(list_channel_counts)) > 1:
        raise ValueError(f"Inconsistent channel counts: {list_channel_counts}")
    
    if not list_of_tracking_dataframes:
        return {}
    
    number_color_channels = list_channel_counts[0]
    list_datasets = ['spot_int', 'spot_size', 'total_spot_int', 'snr', 'psf_amplitude', 'psf_sigma']
    extracted_data_dict = {}
    
    for dataset in list_datasets:
        for channel in range(number_color_channels):
            selected_field = f'{dataset}_ch_{channel}'
            mean_data, median_data, std_data, list_data, number_of_spots = extract_data_from_tracking_df(
                list_of_tracking_dataframes, selected_field, verbose=verbose)
            if dataset == 'spot_int' and channel == 0:
                extracted_data_dict = {
                    'number_of_color_channels': number_color_channels,
                    'number_of_spots': number_of_spots,
                    'max_frame': list_max_frame,
                    'average_number_spots': np.round(np.array(number_of_spots) / (np.array(list_max_frame) + 1), 2),
                    'efficiency_cells': efficiency_cells,
                    'efficiency_manual': efficiency_manual,
                }
            extracted_data_dict.update({
                f'{dataset}_ch_{channel}_mean': mean_data,
                f'{dataset}_ch_{channel}_median': median_data,
                f'{dataset}_ch_{channel}_std': std_data,
                f'{dataset}_ch_{channel}_data': list_data,
            })
    return extracted_data_dict


# Import aggregate_folder_data from the separate file
from .data_aggregation import aggregate_folder_data

__all__ = [
    'get_folder_substrings_and_names',
    'extract_data_from_tracking_df', 
    'extract_data_from_folders',
    'aggregate_folder_data',
]
