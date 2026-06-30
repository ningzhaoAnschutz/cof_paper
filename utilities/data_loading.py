"""
Data loading and aggregation utilities for MicroLive tracking data.

This module provides functions for:
- Extracting folder substrings and names from experiment directories
- Loading tracking DataFrames from result folders
- Aggregating data across multiple conditions/experiments
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from .config import PLASMID_SHORT_NAME_MAPPING, REPORTER_PLASMID_NAME_MAPPING


def get_folder_substrings_and_names(
    dataframes_dir,
    process_individual_days=False,
    plasmid_name_mapping=None,
    plasmid_order=None,
    verbose=True,
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
    result_folders = [
        folder
        for folder in dataframes_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("results_")
    ]

    if not result_folders:
        raise ValueError(f"No results folders found in {dataframes_dir}")

    # Collect all unique date+plasmid combinations
    date_plasmid_combinations = {}
    unrecognized_folders = []

    for folder in result_folders:
        folder_name = folder.name
        date_match = re.match(r"results_(\d{7,8})", folder_name)
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
            full_prefix = name_without_prefix[: plasmid_idx + len(plasmid)]
        elif date:
            full_prefix = f"{date} {plasmid}"
        else:
            full_prefix = plasmid

        if plasmid not in date_plasmid_combinations:
            date_plasmid_combinations[plasmid] = []

        if (date, full_prefix) not in [
            (d, p) for d, p in date_plasmid_combinations[plasmid]
        ]:
            date_plasmid_combinations[plasmid].append((date, full_prefix))

    for plasmid in date_plasmid_combinations:
        date_plasmid_combinations[plasmid] = sorted(
            date_plasmid_combinations[plasmid], key=lambda x: x[0] if x[0] else ""
        )

    if verbose:
        print(
            f"Found {len(date_plasmid_combinations)} plasmids in "
            f"{len(result_folders)} folders"
        )
        for plasmid, combos in date_plasmid_combinations.items():
            print(f"  {plasmid}: {len(combos)} experiment(s)")
        if unrecognized_folders:
            print(f"\nWARNING: {len(unrecognized_folders)} folder(s) NOT RECOGNIZED")

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


def extract_data_from_tracking_df(
    list_of_tracking_dataframes, column_name, verbose=True
):
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
        print("Data for column: ", column_name)
        print("Mean :   ", mean_data)
        print("Number of spots:  ", number_of_spots)
    return mean_data, median_data, std_data, list_data, number_of_spots


def _get_particle_column(tracking_df):
    """Return the best particle identifier column, or None if not found."""
    if "unique_particle" in tracking_df.columns:
        return "unique_particle"
    if "particle" in tracking_df.columns:
        return "particle"
    return None


def _extract_trajectory_means(tracking_df, metric_column):
    """Compute time-averaged metric values per trajectory.

    If a particle column exists, groups by particle and returns the
    mean of ``metric_column`` per trajectory.  Otherwise, returns
    the raw column values (one per row).

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for one cell.
    metric_column : str
        Column name of the metric (e.g., 'spot_int_ch_0').

    Returns
    -------
    np.ndarray
        One value per trajectory (or per row if no particle column).
    """
    particle_col = _get_particle_column(tracking_df)
    if particle_col is not None and metric_column in tracking_df.columns:
        return tracking_df.groupby(particle_col)[metric_column].mean().values
    return tracking_df[metric_column].values


def _extract_trajectory_data_by_colocalization(tracking_df, metric_column):
    """
    Compute time-averaged metric values split by colocalization status.

    For each trajectory (particle), computes the time-average of
    ``metric_column`` across all frames.  Trajectories are classified as
    colocalized if ANY of their detections have ``is_colocalized == True``.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for one cell.
    metric_column : str
        Column name of the metric (e.g., 'spot_int_ch_0').

    Returns
    -------
    coloc_means : np.ndarray or None
        Trajectory-level mean values for colocalized trajectories.
        None if no colocalized trajectories exist, or if required
        columns are missing.
    not_coloc_means : np.ndarray or None
        Trajectory-level mean values for non-colocalized trajectories.
        None if no non-colocalized trajectories exist, or if required
        columns are missing.
    """
    particle_col = _get_particle_column(tracking_df)
    if particle_col is None:
        return None, None
    if "is_colocalized" not in tracking_df.columns:
        return None, None
    if metric_column not in tracking_df.columns:
        return None, None

    # Determine colocalization per trajectory
    traj_coloc = tracking_df.groupby(particle_col)["is_colocalized"].any()

    # Compute time-averaged metric per trajectory
    traj_means = tracking_df.groupby(particle_col)[metric_column].mean()

    # Split by colocalization status
    coloc_particles = traj_coloc[traj_coloc].index
    not_coloc_particles = traj_coloc[~traj_coloc].index

    coloc_means = (
        traj_means.loc[coloc_particles].values if len(coloc_particles) > 0 else None
    )
    not_coloc_means = (
        traj_means.loc[not_coloc_particles].values
        if len(not_coloc_particles) > 0
        else None
    )

    return coloc_means, not_coloc_means


def extract_data_from_folders(
    dataframes_dir,
    folder_substring,
    folder_substring_to_avoid="",
    folder_substring_second_condition="",
    show_file_names=False,
    verbose=True,
):
    """Extract tracking data and efficiency metrics from result folders."""
    dataframes_dir = Path(dataframes_dir)
    condition_folders = [
        f for f in dataframes_dir.iterdir() if f.is_dir() and folder_substring in f.name
    ]
    if folder_substring_to_avoid:
        condition_folders = [
            f for f in condition_folders if folder_substring_to_avoid not in f.name
        ]
    if folder_substring_second_condition:
        condition_folders = [
            f for f in condition_folders if folder_substring_second_condition in f.name
        ]
    condition_folders = sorted(condition_folders, key=lambda x: x.name.split(" ")[-1])

    list_of_tracking_dataframes = []
    efficiency_cells, efficiency_manual = [], []
    list_max_frame, list_channel_counts = [], []

    for folder in condition_folders:
        csv_files = [f for f in folder.iterdir() if f.suffix == ".csv"]
        tracking_files = [f for f in csv_files if "tracking_" in f.stem]
        if not tracking_files:
            raise FileNotFoundError(
                f"No 'tracking_*.csv' file found in folder: {folder}"
            )

        temp_tracking_df = pd.read_csv(tracking_files[0], encoding="latin-1")
        list_of_tracking_dataframes.append(temp_tracking_df)

        ch_cols = [
            int(col.split("_")[-1]) for col in temp_tracking_df.columns if "ch" in col
        ]
        folder_channel_count = max(ch_cols) + 1
        list_channel_counts.append(folder_channel_count)

        coloc_files = [f for f in csv_files if "colocalization_data" in f.stem]
        if coloc_files:
            eff_df = pd.read_csv(coloc_files[0], encoding="latin-1")
            efficiency_cells.append(eff_df["colocalization percentage"].values[0])
        else:
            efficiency_cells.append(np.nan)

        manual_files = [f for f in csv_files if "colocalization_manual" in f.stem]
        if manual_files:
            manual_df = pd.read_csv(manual_files[0], encoding="latin-1")
            efficiency_manual.append(manual_df["colocalization percentage"].values[0])
        else:
            efficiency_manual.append(np.nan)

        list_max_frame.append(temp_tracking_df["frame"].max())
        if show_file_names:
            print(f"Loaded {folder.name} ({folder_channel_count} channels)")

    if len(set(list_channel_counts)) > 1:
        raise ValueError(f"Inconsistent channel counts: {list_channel_counts}")

    if not list_of_tracking_dataframes:
        return {}

    number_color_channels = list_channel_counts[0]
    list_datasets = [
        "spot_int",
        "spot_size",
        "total_spot_int",
        "snr",
        "psf_amplitude",
        "psf_sigma",
    ]
    extracted_data_dict = {}

    for dataset in list_datasets:
        for channel in range(number_color_channels):
            selected_field = f"{dataset}_ch_{channel}"
            mean_data, median_data, std_data, list_data, number_of_spots = (
                extract_data_from_tracking_df(
                    list_of_tracking_dataframes, selected_field, verbose=verbose
                )
            )
            if dataset == "spot_int" and channel == 0:
                extracted_data_dict = {
                    "number_of_color_channels": number_color_channels,
                    "number_of_spots": number_of_spots,
                    "max_frame": list_max_frame,
                    "average_number_spots": np.round(
                        np.array(number_of_spots) / (np.array(list_max_frame) + 1), 2
                    ),
                    "efficiency_cells": efficiency_cells,
                    "efficiency_manual": efficiency_manual,
                }
            # Compute colocalized / non-colocalized trajectory-level data
            list_coloc_data = []
            list_not_coloc_data = []
            for tracking_df in list_of_tracking_dataframes:
                coloc_means, not_coloc_means = (
                    _extract_trajectory_data_by_colocalization(
                        tracking_df, selected_field
                    )
                )
                list_coloc_data.append(coloc_means)
                list_not_coloc_data.append(not_coloc_means)

            extracted_data_dict.update(
                {
                    f"{dataset}_ch_{channel}_mean": mean_data,
                    f"{dataset}_ch_{channel}_median": median_data,
                    f"{dataset}_ch_{channel}_std": std_data,
                    f"{dataset}_ch_{channel}_data": [
                        _extract_trajectory_means(df, selected_field)
                        for df in list_of_tracking_dataframes
                    ],
                    f"{dataset}_ch_{channel}_coloc_data": list_coloc_data,
                    f"{dataset}_ch_{channel}_not_coloc_data": list_not_coloc_data,
                }
            )
    # Count unique trajectories per cell
    list_n_trajectories = []
    for tracking_df in list_of_tracking_dataframes:
        particle_col = _get_particle_column(tracking_df)
        if particle_col is not None:
            n_traj = tracking_df[particle_col].nunique()
        else:
            n_traj = len(tracking_df)
        list_n_trajectories.append(n_traj)
    extracted_data_dict["n_trajectories"] = list_n_trajectories

    return extracted_data_dict


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
    # Colocalized trajectory-level data
    list_int_ch_0_coloc, list_int_ch_1_coloc = [], []
    list_snr_ch_0_coloc, list_snr_ch_1_coloc = [], []
    list_spot_size_ch_0_coloc, list_spot_size_ch_1_coloc = [], []
    list_total_spot_int_ch_0_coloc, list_total_spot_int_ch_1_coloc = [], []
    list_spot_amplitude_ch_0_coloc, list_spot_amplitude_ch_1_coloc = [], []
    list_spot_sigma_ch_0_coloc, list_spot_sigma_ch_1_coloc = [], []
    # Non-colocalized trajectory-level data
    list_int_ch_0_not_coloc, list_int_ch_1_not_coloc = [], []
    list_snr_ch_0_not_coloc, list_snr_ch_1_not_coloc = [], []
    list_spot_size_ch_0_not_coloc, list_spot_size_ch_1_not_coloc = [], []
    list_total_spot_int_ch_0_not_coloc, list_total_spot_int_ch_1_not_coloc = [], []
    list_spot_amplitude_ch_0_not_coloc, list_spot_amplitude_ch_1_not_coloc = [], []
    list_spot_sigma_ch_0_not_coloc, list_spot_sigma_ch_1_not_coloc = [], []
    list_number_spots, list_frames = [], []
    list_number_of_color_channels, list_average_number_spots = [], []
    list_efficiency_ml, list_efficiency_manual = [], []
    list_n_trajectories = []

    if list_folder_substring_to_avoid is None:
        list_folder_substring_to_avoid = [""] * len(list_folder_substrings)
    if list_folder_substring_second_condition is None:
        list_folder_substring_second_condition = [""] * len(list_folder_substrings)

    for i, subfolder in enumerate(list_folder_substrings):
        if verbose:
            print("-----------------------------------")
            print(f"Processing : {subfolder}")

        extracted = extract_data_from_folders(
            dataframes_dir,
            subfolder,
            list_folder_substring_to_avoid[i],
            list_folder_substring_second_condition[i],
            show_file_names=show_file_names,
            verbose=verbose,
        )

        num_ch = extracted.get("number_of_color_channels", 1)
        list_number_of_color_channels.append(num_ch)
        list_number_spots.append(extracted["number_of_spots"])
        list_frames.append(extracted["max_frame"])
        avg_spots = np.round(
            np.array(extracted["number_of_spots"])
            / (np.array(extracted["max_frame"]) + 1),
            2,
        )
        list_average_number_spots.append(avg_spots)
        list_directories.append(extracted)
        list_n_trajectories.append(extracted.get("n_trajectories"))

        # Channel 0
        list_int_ch_0.append(extracted.get("spot_int_ch_0_data"))
        list_snr_ch_0.append(extracted.get("snr_ch_0_data"))
        list_spot_size_ch_0.append(extracted.get("spot_size_ch_0_data"))
        list_total_spot_int_ch_0.append(extracted.get("total_spot_int_ch_0_data"))
        list_spot_amplitude_ch_0.append(extracted.get("psf_amplitude_ch_0_data"))
        list_spot_sigma_ch_0.append(extracted.get("psf_sigma_ch_0_data"))
        # Channel 0 — colocalized / not colocalized
        list_int_ch_0_coloc.append(extracted.get("spot_int_ch_0_coloc_data"))
        list_snr_ch_0_coloc.append(extracted.get("snr_ch_0_coloc_data"))
        list_spot_size_ch_0_coloc.append(extracted.get("spot_size_ch_0_coloc_data"))
        list_total_spot_int_ch_0_coloc.append(
            extracted.get("total_spot_int_ch_0_coloc_data")
        )
        list_spot_amplitude_ch_0_coloc.append(
            extracted.get("psf_amplitude_ch_0_coloc_data")
        )
        list_spot_sigma_ch_0_coloc.append(extracted.get("psf_sigma_ch_0_coloc_data"))
        list_int_ch_0_not_coloc.append(extracted.get("spot_int_ch_0_not_coloc_data"))
        list_snr_ch_0_not_coloc.append(extracted.get("snr_ch_0_not_coloc_data"))
        list_spot_size_ch_0_not_coloc.append(
            extracted.get("spot_size_ch_0_not_coloc_data")
        )
        list_total_spot_int_ch_0_not_coloc.append(
            extracted.get("total_spot_int_ch_0_not_coloc_data")
        )
        list_spot_amplitude_ch_0_not_coloc.append(
            extracted.get("psf_amplitude_ch_0_not_coloc_data")
        )
        list_spot_sigma_ch_0_not_coloc.append(
            extracted.get("psf_sigma_ch_0_not_coloc_data")
        )

        # Channel 1
        if num_ch > 1:
            list_int_ch_1.append(extracted.get("spot_int_ch_1_data"))
            list_snr_ch_1.append(extracted.get("snr_ch_1_data"))
            list_spot_size_ch_1.append(extracted.get("spot_size_ch_1_data"))
            list_total_spot_int_ch_1.append(extracted.get("total_spot_int_ch_1_data"))
            list_spot_amplitude_ch_1.append(extracted.get("psf_amplitude_ch_1_data"))
            list_spot_sigma_ch_1.append(extracted.get("psf_sigma_ch_1_data"))
            # Channel 1 — colocalized / not colocalized
            list_int_ch_1_coloc.append(extracted.get("spot_int_ch_1_coloc_data"))
            list_snr_ch_1_coloc.append(extracted.get("snr_ch_1_coloc_data"))
            list_spot_size_ch_1_coloc.append(extracted.get("spot_size_ch_1_coloc_data"))
            list_total_spot_int_ch_1_coloc.append(
                extracted.get("total_spot_int_ch_1_coloc_data")
            )
            list_spot_amplitude_ch_1_coloc.append(
                extracted.get("psf_amplitude_ch_1_coloc_data")
            )
            list_spot_sigma_ch_1_coloc.append(
                extracted.get("psf_sigma_ch_1_coloc_data")
            )
            list_int_ch_1_not_coloc.append(
                extracted.get("spot_int_ch_1_not_coloc_data")
            )
            list_snr_ch_1_not_coloc.append(extracted.get("snr_ch_1_not_coloc_data"))
            list_spot_size_ch_1_not_coloc.append(
                extracted.get("spot_size_ch_1_not_coloc_data")
            )
            list_total_spot_int_ch_1_not_coloc.append(
                extracted.get("total_spot_int_ch_1_not_coloc_data")
            )
            list_spot_amplitude_ch_1_not_coloc.append(
                extracted.get("psf_amplitude_ch_1_not_coloc_data")
            )
            list_spot_sigma_ch_1_not_coloc.append(
                extracted.get("psf_sigma_ch_1_not_coloc_data")
            )
        else:
            list_int_ch_1.append(None)
            list_snr_ch_1.append(None)
            list_spot_size_ch_1.append(None)
            list_total_spot_int_ch_1.append(None)
            list_spot_amplitude_ch_1.append(None)
            list_spot_sigma_ch_1.append(None)
            list_int_ch_1_coloc.append(None)
            list_snr_ch_1_coloc.append(None)
            list_spot_size_ch_1_coloc.append(None)
            list_total_spot_int_ch_1_coloc.append(None)
            list_spot_amplitude_ch_1_coloc.append(None)
            list_spot_sigma_ch_1_coloc.append(None)
            list_int_ch_1_not_coloc.append(None)
            list_snr_ch_1_not_coloc.append(None)
            list_spot_size_ch_1_not_coloc.append(None)
            list_total_spot_int_ch_1_not_coloc.append(None)
            list_spot_amplitude_ch_1_not_coloc.append(None)
            list_spot_sigma_ch_1_not_coloc.append(None)

        list_efficiency_ml.append(extracted["efficiency_cells"])
        list_efficiency_manual.append(extracted["efficiency_manual"])
        if verbose:
            print("-----------------------------------")

    # Apply filtering if specified
    if min_avg_spots_threshold is not None:
        if verbose:
            print(
                "\n=== Cell Filtering "
                f"(min_avg_spots_threshold={min_avg_spots_threshold}) ==="
            )
        total_kept, total_excluded = 0, 0

        for cond_idx in range(len(list_folder_substrings)):
            avg_spots_data = list_average_number_spots[cond_idx]
            if avg_spots_data is None:
                continue

            keep_mask = [
                (
                    avg >= min_avg_spots_threshold
                    if avg is not None and not np.isnan(avg)
                    else False
                )
                for avg in avg_spots_data
            ]
            n_kept, n_excluded = sum(keep_mask), len(keep_mask) - sum(keep_mask)
            total_kept += n_kept
            total_excluded += n_excluded

            if verbose:
                condition_name = list_folder_substrings[cond_idx]
                print(f"  {condition_name}: {n_kept}/{len(keep_mask)} cells kept")

            def filter_list(data_list):
                return (
                    [d for d, keep in zip(data_list, keep_mask) if keep]
                    if data_list
                    else None
                )

            list_int_ch_0[cond_idx] = filter_list(list_int_ch_0[cond_idx])
            list_snr_ch_0[cond_idx] = filter_list(list_snr_ch_0[cond_idx])
            list_spot_size_ch_0[cond_idx] = filter_list(list_spot_size_ch_0[cond_idx])
            list_total_spot_int_ch_0[cond_idx] = filter_list(
                list_total_spot_int_ch_0[cond_idx]
            )
            list_spot_amplitude_ch_0[cond_idx] = filter_list(
                list_spot_amplitude_ch_0[cond_idx]
            )
            list_spot_sigma_ch_0[cond_idx] = filter_list(list_spot_sigma_ch_0[cond_idx])
            list_int_ch_1[cond_idx] = filter_list(list_int_ch_1[cond_idx])
            list_snr_ch_1[cond_idx] = filter_list(list_snr_ch_1[cond_idx])
            list_spot_size_ch_1[cond_idx] = filter_list(list_spot_size_ch_1[cond_idx])
            list_total_spot_int_ch_1[cond_idx] = filter_list(
                list_total_spot_int_ch_1[cond_idx]
            )
            list_spot_amplitude_ch_1[cond_idx] = filter_list(
                list_spot_amplitude_ch_1[cond_idx]
            )
            list_spot_sigma_ch_1[cond_idx] = filter_list(list_spot_sigma_ch_1[cond_idx])
            # Filter colocalized / not colocalized lists
            list_int_ch_0_coloc[cond_idx] = filter_list(list_int_ch_0_coloc[cond_idx])
            list_snr_ch_0_coloc[cond_idx] = filter_list(list_snr_ch_0_coloc[cond_idx])
            list_spot_size_ch_0_coloc[cond_idx] = filter_list(
                list_spot_size_ch_0_coloc[cond_idx]
            )
            list_total_spot_int_ch_0_coloc[cond_idx] = filter_list(
                list_total_spot_int_ch_0_coloc[cond_idx]
            )
            list_spot_amplitude_ch_0_coloc[cond_idx] = filter_list(
                list_spot_amplitude_ch_0_coloc[cond_idx]
            )
            list_spot_sigma_ch_0_coloc[cond_idx] = filter_list(
                list_spot_sigma_ch_0_coloc[cond_idx]
            )
            list_int_ch_1_coloc[cond_idx] = filter_list(list_int_ch_1_coloc[cond_idx])
            list_snr_ch_1_coloc[cond_idx] = filter_list(list_snr_ch_1_coloc[cond_idx])
            list_spot_size_ch_1_coloc[cond_idx] = filter_list(
                list_spot_size_ch_1_coloc[cond_idx]
            )
            list_total_spot_int_ch_1_coloc[cond_idx] = filter_list(
                list_total_spot_int_ch_1_coloc[cond_idx]
            )
            list_spot_amplitude_ch_1_coloc[cond_idx] = filter_list(
                list_spot_amplitude_ch_1_coloc[cond_idx]
            )
            list_spot_sigma_ch_1_coloc[cond_idx] = filter_list(
                list_spot_sigma_ch_1_coloc[cond_idx]
            )
            list_int_ch_0_not_coloc[cond_idx] = filter_list(
                list_int_ch_0_not_coloc[cond_idx]
            )
            list_snr_ch_0_not_coloc[cond_idx] = filter_list(
                list_snr_ch_0_not_coloc[cond_idx]
            )
            list_spot_size_ch_0_not_coloc[cond_idx] = filter_list(
                list_spot_size_ch_0_not_coloc[cond_idx]
            )
            list_total_spot_int_ch_0_not_coloc[cond_idx] = filter_list(
                list_total_spot_int_ch_0_not_coloc[cond_idx]
            )
            list_spot_amplitude_ch_0_not_coloc[cond_idx] = filter_list(
                list_spot_amplitude_ch_0_not_coloc[cond_idx]
            )
            list_spot_sigma_ch_0_not_coloc[cond_idx] = filter_list(
                list_spot_sigma_ch_0_not_coloc[cond_idx]
            )
            list_int_ch_1_not_coloc[cond_idx] = filter_list(
                list_int_ch_1_not_coloc[cond_idx]
            )
            list_snr_ch_1_not_coloc[cond_idx] = filter_list(
                list_snr_ch_1_not_coloc[cond_idx]
            )
            list_spot_size_ch_1_not_coloc[cond_idx] = filter_list(
                list_spot_size_ch_1_not_coloc[cond_idx]
            )
            list_total_spot_int_ch_1_not_coloc[cond_idx] = filter_list(
                list_total_spot_int_ch_1_not_coloc[cond_idx]
            )
            list_spot_amplitude_ch_1_not_coloc[cond_idx] = filter_list(
                list_spot_amplitude_ch_1_not_coloc[cond_idx]
            )
            list_spot_sigma_ch_1_not_coloc[cond_idx] = filter_list(
                list_spot_sigma_ch_1_not_coloc[cond_idx]
            )
            list_number_spots[cond_idx] = filter_list(list_number_spots[cond_idx])
            list_frames[cond_idx] = filter_list(list_frames[cond_idx])
            list_n_trajectories[cond_idx] = filter_list(
                list_n_trajectories[cond_idx]
            )
            list_average_number_spots[cond_idx] = (
                filter_list(list_average_number_spots[cond_idx].tolist())
                if list_average_number_spots[cond_idx] is not None
                else None
            )
            list_efficiency_ml[cond_idx] = filter_list(list_efficiency_ml[cond_idx])
            list_efficiency_manual[cond_idx] = filter_list(
                list_efficiency_manual[cond_idx]
            )

        if verbose:
            print(f"  TOTAL: {total_kept}/{total_kept + total_excluded} cells kept\n")

    return {
        "directories": list_directories,
        "int_ch_0": list_int_ch_0,
        "int_ch_1": list_int_ch_1,
        "snr_ch_0": list_snr_ch_0,
        "snr_ch_1": list_snr_ch_1,
        "spot_size_ch_0": list_spot_size_ch_0,
        "spot_size_ch_1": list_spot_size_ch_1,
        "total_spot_int_ch_0": list_total_spot_int_ch_0,
        "total_spot_int_ch_1": list_total_spot_int_ch_1,
        "spot_amplitude_ch_0": list_spot_amplitude_ch_0,
        "spot_amplitude_ch_1": list_spot_amplitude_ch_1,
        "spot_sigma_ch_0": list_spot_sigma_ch_0,
        "spot_sigma_ch_1": list_spot_sigma_ch_1,
        # Colocalized trajectory-level data
        "int_ch_0_coloc": list_int_ch_0_coloc,
        "int_ch_1_coloc": list_int_ch_1_coloc,
        "snr_ch_0_coloc": list_snr_ch_0_coloc,
        "snr_ch_1_coloc": list_snr_ch_1_coloc,
        "spot_size_ch_0_coloc": list_spot_size_ch_0_coloc,
        "spot_size_ch_1_coloc": list_spot_size_ch_1_coloc,
        "total_spot_int_ch_0_coloc": list_total_spot_int_ch_0_coloc,
        "total_spot_int_ch_1_coloc": list_total_spot_int_ch_1_coloc,
        "spot_amplitude_ch_0_coloc": list_spot_amplitude_ch_0_coloc,
        "spot_amplitude_ch_1_coloc": list_spot_amplitude_ch_1_coloc,
        "spot_sigma_ch_0_coloc": list_spot_sigma_ch_0_coloc,
        "spot_sigma_ch_1_coloc": list_spot_sigma_ch_1_coloc,
        # Non-colocalized trajectory-level data
        "int_ch_0_not_coloc": list_int_ch_0_not_coloc,
        "int_ch_1_not_coloc": list_int_ch_1_not_coloc,
        "snr_ch_0_not_coloc": list_snr_ch_0_not_coloc,
        "snr_ch_1_not_coloc": list_snr_ch_1_not_coloc,
        "spot_size_ch_0_not_coloc": list_spot_size_ch_0_not_coloc,
        "spot_size_ch_1_not_coloc": list_spot_size_ch_1_not_coloc,
        "total_spot_int_ch_0_not_coloc": list_total_spot_int_ch_0_not_coloc,
        "total_spot_int_ch_1_not_coloc": list_total_spot_int_ch_1_not_coloc,
        "spot_amplitude_ch_0_not_coloc": list_spot_amplitude_ch_0_not_coloc,
        "spot_amplitude_ch_1_not_coloc": list_spot_amplitude_ch_1_not_coloc,
        "spot_sigma_ch_0_not_coloc": list_spot_sigma_ch_0_not_coloc,
        "spot_sigma_ch_1_not_coloc": list_spot_sigma_ch_1_not_coloc,
        "number_spots": list_number_spots,
        "frames": list_frames,
        "number_of_color_channels": list_number_of_color_channels,
        "average_number_spots": list_average_number_spots,
        "efficiency_ml": list_efficiency_ml,
        "efficiency_manual": list_efficiency_manual,
        "n_trajectories": list_n_trajectories,
    }


__all__ = [
    "get_folder_substrings_and_names",
    "extract_data_from_tracking_df",
    "extract_data_from_folders",
    "aggregate_folder_data",
]
