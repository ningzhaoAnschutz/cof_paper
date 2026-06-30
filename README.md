# Co-Translational Folding Tracking (coTFT)

Code repository for: **"Live-cell co-translational folding tracking reveals bidirectional translation-folding coupling"**

**Authors:** Rhiannon M. Sears, Luis U. Aguilera, Tristan M. Bunting, Ning Zhao\*

Department of Biochemistry and Molecular Genetics, University of Colorado Anschutz Medical Campus; Aurora, CO, 80045, USA.
\*Corresponding author. Email: ning.zhao@cuanschutz.edu

---

## About

This repository contains the Jupyter notebooks, analysis modules, and mathematical model used to reproduce the computational figures in the manuscript. The coTFT platform enables direct, simultaneous tracking of translation and protein folding from individual mRNAs in live cells. Image analysis is performed with [microlive](https://github.com/NingZhao-Lab/microlive), which bundles all microscopy, tracking, and simulation dependencies.

---

## Repository Structure

```
cof_paper/
├── notebooks/
│   ├── CoF_project/                  # Co-translational folding efficiency analysis
│   │   ├── Fig 1K_6x reporters CoF analysis.ipynb
│   │   ├── Fig 2D-G_4xFast vs 4xSlow CoF analysis.ipynb
│   │   ├── Fig 3F-I_4xFast XBP1u variants CoF analysis.ipynb
│   │   ├── Fig 4C_1x-6xFast Model CoF analysis.ipynb
│   │   ├── Fig 4D_1x-6xSlow Model CoF analysis.ipynb
│   │   └── Fig S4G-J_4xSlow XBP1u variants CoF analysis.ipynb
│   │
│   ├── Inhibitors/                   # Puromycin and Harringtonine run-off assays
│   │   ├── inhibitors.py            # Shared analysis module (fitting, plotting, simulation)
│   │   ├── Fig 1J_Puro_6x reporters.ipynb
│   │   ├── Fig 2C_Puro_4xFast vs 4xSlow.ipynb
│   │   ├── Fig 2H-J_Har_Fits_4xFast vs 4xSlow.ipynb
│   │   ├── Fig 3B_Puro_4xFast XBP1u variants.ipynb
│   │   ├── Fig 3C-E_Har_Fits_4xFast XBP1u variants.ipynb
│   │   ├── Fig S3C_Har_Fits_4xFast without anti-GFP-IB.ipynb
│   │   ├── Fig S3D_Har_Fits_4xSlow without anti-GFP-IB.ipynb
│   │   └── Fig S4F_Puro_4xSlow XBP1u variants.ipynb
│   │
│   ├── FRAP_analyses/                # FRAP processing and visualization
│   │   ├── FRAP_processing_RS.ipynb
│   │   ├── FRAP_processing_line_command.py
│   │   ├── Fig 1C_FRAP_representative_images.ipynb
│   │   ├── Fig 1D-E_FRAP_interpreatation.ipynb
│   │   ├── frap_plotting_module.py
│   │   └── README.md
│   │
│   ├── codon_optimization/           # Codon Adaptation Index and codon usage analysis
│   │   ├── calculating_CAI_RS.ipynb
│   │   ├── codon_optimization_Harringtonine_Simulation.ipynb
│   │   └── Clark_lab_min_max/Single_GFP.ipynb
│   │
│   └── image_conditions/             # Laser intensity quality control
│       └── laser_intensities.ipynb
│
├── model/                            # Geometric folding time model
│   ├── code/
│   │   └── geometric_model.py        # Spatial delay model (Figs. 4E-H, S6)
│   └── databases/
│       ├── Dark_mCh_Cells_new.xlsx   # Per-cell CoF efficiency data (dGFPFast series)
│       ├── Dark_mCh_Cells_slow.xlsx  # Per-cell CoF efficiency data (dGFPSlow series)
│       └── GFP_positions.ipynb       # GFP domain position calculations from plasmid sequences
│
├── utilities/                        # Shared Python package
│   ├── __init__.py
│   ├── config.py                     # Plasmid name mappings and constants
│   ├── data_loading.py               # Folder discovery and tracking data extraction
│   ├── plotting.py                   # Swarm plots, efficiency plots, KDE plots
│   ├── intensity_distributions.py    # Per-particle intensity distribution analysis
│   ├── metadata.py                   # LIF laser intensity metadata extraction
│   └── export_to_excel.py            # Export efficiency data to Excel
│
├── requirements.txt
├── LICENSE                           # GNU GPL v3
└── README.md
```

---

## Notebooks

### Figure 1 -- Tracking Single-mRNA Co-Translational Folding

| Notebook | Panels | Content |
|----------|--------|---------|
| `Fig 1C_FRAP_representative_images.ipynb` | 1C | Representative FRAP images and kymographs |
| `Fig 1D-E_FRAP_interpreatation.ipynb` | 1D, 1E | FRAP recovery curves and half-recovery times |
| `FRAP_processing_RS.ipynb` | (processing) | Raw FRAP data processing pipeline |
| `Fig 1J_Puro_6x reporters.ipynb` | 1J | Puromycin assay for 6x reporters |
| `Fig 1K_6x reporters CoF analysis.ipynb` | 1K | CoF efficiencies for 6xdGFPFast, 6xdGFPSlow, 6xdmCh |

### Figure 2 -- Fast- and Slow-Folding Reporters and Translation Elongation

| Notebook | Panels | Content |
|----------|--------|---------|
| `Fig 2C_Puro_4xFast vs 4xSlow.ipynb` | 2C | Puromycin assay for 4xdGFPFast and 4xdGFPSlow |
| `Fig 2D-G_4xFast vs 4xSlow CoF analysis.ipynb` | 2D-G | CoF efficiency, NC counts, NC and folding spot intensities |
| `Fig 2H-J_Har_Fits_4xFast vs 4xSlow.ipynb` | 2H-J | Harringtonine run-off curves and elongation rate fits |

### Figure 3 -- XBP1u Perturbation of Translation Elongation

| Notebook | Panels | Content |
|----------|--------|---------|
| `Fig 3B_Puro_4xFast XBP1u variants.ipynb` | 3B | Puromycin assay for XBP1u-containing reporters |
| `Fig 3C-E_Har_Fits_4xFast XBP1u variants.ipynb` | 3C-E | Harringtonine run-off curves and elongation rates for XBP1u variants |
| `Fig 3F-I_4xFast XBP1u variants CoF analysis.ipynb` | 3F-I | CoF efficiencies and spot intensities for XBP1u reporters |

### Figure 4 -- Folding Kinetics and Geometric Model

| Notebook / Script | Panels | Content |
|-------------------|--------|---------|
| `Fig 4C_1x-6xFast Model CoF analysis.ipynb` | 4C | CoF efficiencies for 1x-6x dGFPFast reporter series |
| `Fig 4D_1x-6xSlow Model CoF analysis.ipynb` | 4D | CoF efficiencies for 1x-6x dGFPSlow reporter series |
| `model/code/geometric_model.py` | 4E-H | Geometric spatial delay model fits and folding time predictions |

### Supplementary Figures

| Notebook / Script | Panels | Content |
|-------------------|--------|---------|
| `Fig S3C_Har_Fits_4xFast without anti-GFP-IB.ipynb` | S3C | Harringtonine run-off without anti-GFP intrabody (4xdGFPFast) |
| `Fig S3D_Har_Fits_4xSlow without anti-GFP-IB.ipynb` | S3D | Harringtonine run-off without anti-GFP intrabody (4xdGFPSlow) |
| `Fig S4F_Puro_4xSlow XBP1u variants.ipynb` | S4F | Puromycin assay for 4xdGFPSlow XBP1u variants |
| `Fig S4G-J_4xSlow XBP1u variants CoF analysis.ipynb` | S4G-J | CoF efficiencies for 4xdGFPSlow XBP1u reporters |
| `model/code/geometric_model.py` | S6A-B | Parameter sensitivity analysis (chi-squared landscape) |

---

## Geometric Folding Time Model

The `model/code/geometric_model.py` script implements a geometric spatial delay model for co-translational GFP folding. Each GFP domain requires a minimum folding distance along the mRNA after it emerges from the ribosome exit tunnel. Assuming uniformly distributed ribosomes, the model estimates two parameters per GFP variant: an amplitude factor (A) and an inherent folding time (tau_fold). The model reads per-cell CoF efficiency data from the Excel files in `model/databases/` and outputs fit results, model plots, and parameter sensitivity heatmaps.

To run the model:

```bash
cd model/code
python geometric_model.py
```

Output files are saved to `model/figures/`.

---

## Environment Setup

```bash
# 1. Create the conda environment
conda create -n microlive python=3.10 -y
conda activate microlive

# 2. Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.10 or later. **Conda environment:** `microlive`.

The `requirements.txt` installs [microlive](https://pypi.org/project/microlive/), which bundles all imaging and analysis dependencies (NumPy, SciPy, scikit-image, trackpy, Cellpose, matplotlib, seaborn, tasep-models, and others).

---

## Microscopy Data

Live-cell confocal microscopy data used in this study are publicly available on Zenodo:

> **Download:** [https://doi.org/10.5281/zenodo.20936426](https://doi.org/10.5281/zenodo.20936426)

The dataset is large and must be downloaded independently. Data files are not included in this repository. Each notebook specifies the expected data directory path in its first code cells (see the `dataframes_dir` or `data_dir` variable). The CoF analysis notebooks expect MicroLive tracking output directories containing per-cell DataFrames, while the Inhibitors notebooks expect time-course tracking data organized by reporter plasmid ID.

---

## Reproducing Figures

1. Activate the `microlive` conda environment.
2. Obtain the microscopy tracking data and place it in the expected directory (see the `dataframes_dir` or `data_dir` variable in each notebook).
3. Open the appropriate notebook from `notebooks/` for the figure of interest.
4. Run all cells. The `utilities/` package is auto-discovered from the repository root.
5. For Figures 4E-H and S6, run `python model/code/geometric_model.py` directly.

---

## Key Dependencies

| Category | Libraries |
|----------|-----------|
| **Microscopy and Tracking** | [microlive](https://github.com/NingZhao-Lab/microlive) (includes scikit-image, trackpy, Cellpose, tifffile, readlif) |
| **Scientific Computing** | NumPy, SciPy, pandas, Numba |
| **Visualization** | matplotlib, seaborn |
| **Sequence Analysis** | Biopython |

See [`requirements.txt`](requirements.txt) for the complete specification.

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
