"""
Shared utilities for COF paper analysis notebooks.

This package provides reusable components for data loading, plotting, and configuration
that can be shared across different analysis notebooks (CoF_project, Inhibitors, etc.).

Modules:
    - config: Plasmid name mappings and other constants
    - data_loading: Functions for loading and aggregating MicroLive tracking data
    - plotting: Visualization functions for swarm plots, efficiency plots, etc.
    - metadata: Functions for extracting microscopy file metadata (LIF, etc.)

Example usage:
    from utilities import config, data_loading, plotting
    
    # Or import specific items
    from utilities.config import REPORTER_PLASMID_NAME_MAPPING
    from utilities.data_loading import aggregate_folder_data
    from utilities.plotting import plot_swarm_plot
    from utilities.metadata import extract_laser_intensities
"""

from . import config
from . import data_loading
from . import plotting
from . import metadata

# Convenience re-exports for common items
from .config import REPORTER_PLASMID_NAME_MAPPING, PLASMID_SHORT_NAME_MAPPING
from .data_loading import (
    get_folder_substrings_and_names,
    extract_data_from_tracking_df,
    extract_data_from_folders,
    aggregate_folder_data,
)
from .plotting import (
    plot_swarm_plot,
    plot_swarm_plot_efficiency,
    plot_efficiency_vs_intensity_scatter,
    plot_efficiency_vs_intensity_scatter_means,
)
from .metadata import extract_laser_intensities

__all__ = [
    # Modules
    'config',
    'data_loading', 
    'plotting',
    'metadata',
    # Config items
    'REPORTER_PLASMID_NAME_MAPPING',
    'PLASMID_SHORT_NAME_MAPPING',
    # Data loading functions
    'get_folder_substrings_and_names',
    'extract_data_from_tracking_df',
    'extract_data_from_folders',
    'aggregate_folder_data',
    # Plotting functions
    'plot_swarm_plot',
    'plot_swarm_plot_efficiency',
    'plot_efficiency_vs_intensity_scatter',
    'plot_efficiency_vs_intensity_scatter_means',
    # Metadata functions
    'extract_laser_intensities',
]
