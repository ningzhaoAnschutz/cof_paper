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
cotranslational_folding/
├── notebooks/                    # Analysis notebooks
│   ├── FRAP_analyses/            # FRAP analysis notebooks
│   ├── Harringtonine/            # Harringtonine chase experiments
│   ├── CoF_project/              # Cotranslational folding analysis
│   ├── codon_optimization/       # Codon optimization studies
│   └── processing_long_movies/   # Long timecourse processing
├── modeling/                     # Simulation and modeling
│   ├── TASEP/                    # TASEP simulation notebooks
│   └── cellpose_models/          # Custom Cellpose segmentation models
├── data/                         # Data files (not tracked in git)
│   └── gene_sequences/           # Gene sequence files (.dna)
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
