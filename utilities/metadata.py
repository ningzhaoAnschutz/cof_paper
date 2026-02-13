"""
Metadata extraction utilities for microscopy files.

This module provides functions for extracting hardware and acquisition
metadata from microscopy file formats (LIF, etc.).
"""

from pathlib import Path
import pandas as pd
from microlive import microscopy as mi


def extract_laser_intensities(folder_with_lif_files, verbose=False):
    """
    Extract laser intensities and wave ranges from all LIF files.
    Only reports channels where the laser is actually used (intensity > 0).
    Columns are named by laser wavelength (e.g., "488 Intensity (%)").
    
    Parameters
    ----------
    folder_with_lif_files : Path or str
        Directory containing LIF files
    verbose : bool, optional
        If True, print detailed output for each image. Default False.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - File Name
        - Image Name
        - For each USED laser: {wavelength} Wave Range, {wavelength} Intensity (%)
    
    Examples
    --------
    >>> data_dir = Path('/path/to/lif/files')
    >>> df = extract_laser_intensities(data_dir)
    >>> df.to_csv('laser_intensities.csv', index=False)
    """
    if isinstance(folder_with_lif_files, str):
        folder_with_lif_files = Path(folder_with_lif_files)
    
    # Find all LIF files
    files = [f for f in folder_with_lif_files.iterdir() if f.is_file() and f.suffix == '.lif']
    print(f'Found {len(files)} LIF files in {folder_with_lif_files.name}')
    
    # Collect all data
    all_data = []
    
    for lif_file in files:
        print(f'Processing: {lif_file.name}', end='')
        
        try:
            # Read LIF file metadata
            (list_images, list_names, pixel_xy_um, voxel_z_um, channel_names, 
             number_color_channels, list_time_intervals, bit_depth, 
             list_laser_lines, list_intensities, list_wave_ranges) = mi.ReadLif(
                lif_file, show_metadata=False, save_tif=False, save_png=False, format='TZYXC'
            ).read()
            
            print(f' ({len(list_names)} images)')
            
            # Extract data for each image in the LIF file
            for i, image_name in enumerate(list_names):
                laser_lines = list_laser_lines[i] if list_laser_lines else []
                intensities = list_intensities[i] if list_intensities else []
                wave_ranges = list_wave_ranges[i] if list_wave_ranges else []
                
                row = {
                    'File Name': lif_file.name,
                    'Image Name': image_name,
                }
                
                # Only add columns for USED lasers (intensity > 0)
                used_lasers = []
                for ch_idx in range(len(laser_lines)):
                    intensity = intensities[ch_idx] if ch_idx < len(intensities) else 0
                    
                    if intensity == 0:
                        continue
                    
                    laser_nm = laser_lines[ch_idx]
                    used_lasers.append(laser_nm)
                    
                    if ch_idx < len(wave_ranges):
                        wr = wave_ranges[ch_idx]
                        if isinstance(wr, (list, tuple)) and len(wr) == 2:
                            row[f'{laser_nm} Wave Range'] = f'{wr[0]}-{wr[1]} nm'
                        else:
                            row[f'{laser_nm} Wave Range'] = str(wr)
                    
                    row[f'{laser_nm} Intensity (%)'] = intensity
                
                all_data.append(row)
                
                # Only print per-image details if verbose
                if verbose:
                    if used_lasers:
                        print(f'  - {image_name}: Lasers used: {used_lasers}')
                    else:
                        print(f'  - {image_name}: No lasers used!')
                
        except Exception as e:
            print(f' ERROR: {e}')
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns
    base_cols = ['File Name', 'Image Name']
    other_cols = [c for c in df.columns if c not in base_cols]
    
    def get_wavelength(col_name):
        try:
            return int(col_name.split()[0])
        except:
            return 9999
    
    other_cols_sorted = sorted(other_cols, key=get_wavelength)
    df = df[base_cols + other_cols_sorted]
    
    print(f'Total: {len(df)} images processed')
    
    return df


__all__ = ['extract_laser_intensities']
