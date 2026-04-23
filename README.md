# Cotranslational Folding Project

**Authors:** Rhiannon Sears, Luis Aguilera, Ning Zhao

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Codes for: **Cotranslational Folding Paper**

## Installation

```bash
# Create environment (recommended)
conda create -n microlive python=3.10 -y
conda activate microlive

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

This project uses:

- **[microlive](https://pypi.org/project/microlive/)** (≥1.0.17) - Microscopy image analysis library
  - Includes tasep-models, cellpose, trackpy, and all imaging/analysis dependencies
- **[biopython](https://pypi.org/project/biopython/)** - Sequence analysis

## Project Structure

```text
cof_paper/
├── utilities/                    # Shared Python package (importable as `import utilities`)
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Plasmid name mappings and constants
│   ├── data_loading.py           # Folder discovery and tracking data extraction
│   ├── data_aggregation.py       # Multi-condition data aggregation
│   ├── plotting.py               # Swarm plots, efficiency plots, KDE plots
│   ├── metadata.py               # Microscopy file metadata (LIF laser intensities)
│   └── export_to_excel.py        # Export efficiency data to Excel
├── notebooks/                    # Analysis notebooks
│   ├── CoF_project/              # Cotranslational folding analysis & figure generation
│   │   ├── Fig 1_CoF Analysis Notebook.ipynb
│   │   ├── Fig 2_GFPFast_CoF Analysis Notebook.ipynb
│   │   ├── Fig 2_GFPSlow_CoF Analysis Notebook.ipynb
│   │   ├── Fig 3_CoF Analysis Notebook.ipynb
│   │   ├── Fig 4_GFPFast_CoF Analysis Notebook.ipynb
│   │   ├── Fig 4_GFPSlow_CoF Analysis Notebook.ipynb
│   │   ├── Fig 5_GFPFast_CoF Analysis Notebook.ipynb
│   │   ├── Fig 5_GFPSlow_CoF Analysis Notebook.ipynb
│   │   ├── Fig X CoF Analysis Notebook.ipynb
│   │   └── All Plots_20260310.ipynb
│   ├── Inhibitors/               # Inhibitor runoff experiments
│   │   ├── inhibitors.py         # Inhibitor analysis module (fitting, plotting, simulation)
│   │   ├── Fig 1_Harringtonine_dmCh dishes.ipynb
│   │   ├── Fig 1_Puromycin_all reporters.ipynb
│   │   ├── Fig 3_Harringtonine_Fast dishes.ipynb
│   │   ├── Fig 3_Harringtonine_Fast vs Slow.ipynb
│   │   ├── Fig 3_Harringtonine_Slow dishes.ipynb
│   │   ├── Fig 3_Puromycin_Fast vs Slow.ipynb
│   │   ├── Fig 4_Harringtonine_*.ipynb  (3 notebooks)
│   │   ├── Fig 4_Puromycin_*.ipynb      (2 notebooks)
│   │   ├── Fig 5_Harringtonine_Deopt Fast dishes.ipynb
│   │   ├── Fig 5_Puromycin_Fast deopt vs Slow deopt.ipynb
│   │   ├── LA_Fig 3_Harringtonine_Fast vs Slow.ipynb
│   │   ├── Harringtonine_example.ipynb
│   │   └── Puromycin_example.ipynb
│   ├── acf/                      # Autocorrelation function analysis
│   │   ├── acf_individual.py     # Individual ACF analysis
│   │   ├── acf_sensitivity.py    # Sensitivity analysis
│   │   ├── pipeline_time_courses.py  # ACF processing pipeline
│   │   └── processing_time_courses.ipynb
│   ├── FRAP_analyses/            # FRAP processing and visualization
│   │   ├── FRAP_processing_RS.ipynb
│   │   ├── FRAP_representative_images_RS.ipynb
│   │   └── frap_plotting_module.py
│   ├── kk_autocorrelations/      # Legacy autocorrelation notebooks
│   ├── codon_optimization/       # CAI analysis and codon studies
│   ├── image_conditions/         # Laser intensity audits
│   └── reprocessing_data/        # Data reprocessing workflows
├── modeling/                     # Simulation and modeling
│   ├── mechanistic_model/        # Kinetic models (One-Pool, Two-Pool)
│   ├── TASEP/                    # TASEP ribosome translation simulations
│   └── cellpose_models/          # Custom Cellpose segmentation models
├── data/                         # Data files (not tracked in git)
│   └── gene_sequences/           # Gene sequence files (.dna)
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

All notebooks can be run with:

```bash
conda activate microlive
jupyter lab
```

Navigate to any notebook in `notebooks/` or `modeling/` to reproduce figures.

---
