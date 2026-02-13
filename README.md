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

```
cof_paper/
├── notebooks/                    # Analysis notebooks
│   ├── CoF_project/              # Cotranslational folding analysis & figure generation
│   │   ├── *_CoF_notebook.ipynb  # Per-construct analysis notebooks
│   │   ├── notebook_comparing_datasets.ipynb
│   │   ├── notebook_folding_efficiency.ipynb
│   │   └── load_data.py          # Shared data loading utilities
│   ├── FRAP_analyses/            # FRAP processing and visualization
│   │   ├── FRAP_processing_RS.ipynb
│   │   ├── FRAP_representative_images_RS.ipynb
│   │   └── frap_plotting_module.py
│   ├── Inhibitors/               # Inhibitor runoff experiments
│   ├── autocorrelations/         # Time-course correlation analysis
│   │   ├── processing_time_courses.ipynb
│   │   └── cross_correlation_cof.ipynb
│   └── codon_optimization/       # CAI analysis and codon studies
├── modeling/                     # Simulation and modeling
│   ├── mechanistic_model/        # Kinetic models (One-Pool, Two-Pool)
│   │   ├── modeling/             # Model fitting scripts
│   │   └── writing/              # LaTeX documentation
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
