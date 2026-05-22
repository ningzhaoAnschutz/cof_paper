"""
Export data_dict efficiency data to an Excel file matching the
'Dark mCh Cells.xlsx' format.

Usage in a notebook:
    import utilities as ld

    list_folder_substrings, list_names = ld.get_folder_substrings_and_names(
        dataframes_dir, process_individual_days=True,
    )
    data_dict = ld.aggregate_folder_data(dataframes_dir, list_folder_substrings, ...)

    ld.export_efficiency_to_excel(
        data_dict, list_folder_substrings, list_names,
        save_path='output.xlsx',
    )
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from .config import REPORTER_PLASMID_NAME_MAPPING

def _parse_folder_substring(folder_substring):
    """
    Extract the date and plasmid ID from a folder substring.

    Folder substrings from get_folder_substrings_and_names
    (process_individual_days=True) look like:
        '20260210 pRS026'       → date='20260210', plasmid='pRS026'
        '20260210-1 pRS031'     → date='20260210', plasmid='pRS031'

    When process_individual_days=False, substrings are just plasmid IDs:
        'pRS026'                → date=None, plasmid='pRS026'

    Returns:
        tuple: (date_str or None, plasmid_id or None)
    """
    # Try to match "DATE PLASMID" or "DATE-N PLASMID" format
    match = re.match(r"^(\d{7,8}(?:-\d+)?)\s+(\w+)", folder_substring)
    if match:
        return match.group(1), match.group(2)

    # Try to match a bare plasmid ID (e.g., 'pRS026', 'pNZ212')
    for plasmid_id in REPORTER_PLASMID_NAME_MAPPING:
        if folder_substring.strip() == plasmid_id:
            return None, plasmid_id

    # Check if the substring contains a known plasmid ID
    for plasmid_id in REPORTER_PLASMID_NAME_MAPPING:
        if plasmid_id in folder_substring:
            # Try to extract date from the beginning
            date_match = re.match(r"^(\d{7,8})", folder_substring)
            date = date_match.group(1) if date_match else None
            return date, plasmid_id

    return None, None


def _build_reporter_variant(plasmid_id, reporter_prefix, reporter_suffix):
    """
    Build the full reporter variant string from a plasmid ID using config.py.

    Example:
        'pRS026' → 'pUB-RBsmHA-5xsfGFP-1xmCh-24xMS2'

    Uses REPORTER_PLASMID_NAME_MAPPING to get the short name (e.g., '5sfGFP-1mCh'),
    then expands it with 'x' notation.
    """
    short_name = REPORTER_PLASMID_NAME_MAPPING.get(plasmid_id)
    if short_name is None:
        return plasmid_id  # Fallback: return raw ID

    # Expand '5sfGFP-1mCh' → '5xsfGFP-1xmCh'
    parts = short_name.split("-")
    expanded = []
    for part in parts:
        match = re.match(r"^(\d+)(.+)$", part)
        if match:
            expanded.append(f"{match.group(1)}x{match.group(2)}")
        else:
            expanded.append(part)

    return f"{reporter_prefix}{'-'.join(expanded)}{reporter_suffix}"


def export_efficiency_to_excel(
    data_dict,
    list_folder_substrings,
    list_names,
    save_path,
    title="GFP Fast-mCherry Reporters",
    efficiency_key="efficiency_manual",
    reporter_prefix="pUB-RBsmHA-",
    reporter_suffix="-24xMS2",
    col_type_label="ML 0.5",
):
    """
    Convert data_dict (from aggregate_folder_data) into a structured Excel file.

    The output format matches Dark mCh Cells.xlsx:
        Row 0: Title
        Row 1: Date per column
        Row 2: Reporter Variant per column
        Row 3: (empty)
        Row 4: CoF Efficiency / column type
        Rows 5+: Per-cell efficiency values

    Dates and reporter variant names are extracted from list_folder_substrings
    (e.g., '20260210 pRS026') and resolved using the canonical mappings in
    utilities/config.py.

    Args:
        data_dict: Dictionary returned by aggregate_folder_data().
        list_folder_substrings: Folder substrings from
            get_folder_substrings_and_names().
            These contain the date and plasmid ID (e.g., '20260210 pRS026').
        list_names: List of condition names from get_folder_substrings_and_names().
        save_path: Output Excel file path.
        title: Title string for the first row.
        efficiency_key: Which efficiency to export ('efficiency_manual' or
            'efficiency_ml').
        reporter_prefix: Prefix for building reporter variant strings.
        reporter_suffix: Suffix for building reporter variant strings.
        col_type_label: Label for the column type row (e.g., 'ML 0.5').

    Returns:
        pd.DataFrame: The constructed DataFrame (also saved to save_path).
    """
    save_path = Path(save_path)

    all_efficiencies = data_dict[efficiency_key]
    n_conditions = len(all_efficiencies)

    # --- Parse dates and plasmid IDs from folder substrings ---
    dates = []
    reporters = []
    for substring in list_folder_substrings:
        date, plasmid_id = _parse_folder_substring(substring)
        dates.append(date or "")
        if plasmid_id:
            reporters.append(
                _build_reporter_variant(plasmid_id, reporter_prefix, reporter_suffix)
            )
        else:
            reporters.append(substring)  # Fallback

    # Find max number of cells across all conditions
    max_cells = max(
        (len(eff) for eff in all_efficiencies if eff is not None),
        default=0,
    )

    # --- Build header rows ---
    n_cols = 1 + n_conditions

    # Row 0: Title
    row_title = [title] + [""] * n_conditions

    # Row 1: Date — extracted from folder substrings
    row_date = ["Date"] + dates

    # Row 2: Reporter Variant — resolved from plasmid ID via config.py
    row_reporter = ["Reporter Variant"] + reporters

    # Row 3: Empty
    row_empty = [""] * n_cols

    # Row 4: Column type
    row_col_type = ["CoF Efficiency"] + [col_type_label] * n_conditions

    # --- Build data rows ---
    data_rows = []
    for cell_idx in range(max_cells):
        row = [f"Cell {cell_idx + 1}"]
        for cond_idx in range(n_conditions):
            eff_list = all_efficiencies[cond_idx]
            if eff_list is not None and cell_idx < len(eff_list):
                val = eff_list[cell_idx]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    row.append(round(val, 2))
                else:
                    row.append("")
            else:
                row.append("")
        data_rows.append(row)

    # --- Assemble into DataFrame ---
    all_rows = [row_title, row_date, row_reporter, row_empty, row_col_type] + data_rows
    df = pd.DataFrame(all_rows)

    # --- Save ---
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(save_path), index=False, header=False)
    print(f"Saved efficiency data to: {save_path}")
    print(f"  {n_conditions} conditions, {max_cells} max cells")
    for i, (d, r) in enumerate(zip(dates, reporters)):
        print(f"    Col {i+1}: {d} | {r}")

    return df
