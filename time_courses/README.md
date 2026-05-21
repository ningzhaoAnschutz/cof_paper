# Burst Quantification Pipeline

Quantifies translational bursting dynamics from single-trajectory fluorescence intensity time courses. Processes the **folding channel (ch0)** from dual-channel live-cell imaging data across four CoF constructs.

![Pipeline Overview](../diagrams_methods/folding_states_methodology.png)

> The methodology diagram above is also available as [PDF](../diagrams_methods/folding_states_methodology.pdf) and editable [LaTeX source](../diagrams_methods/folding_states_methodology.tex).

---

## Directory Layout

```
time_courses/
├── README.md                              ← you are here
│
│   # Burst quantification pipeline (active, current paper figures)
├── run_analysis.py                        ← entry point: runs 4 constructs end-to-end
├── burst_quantification.py                ← core module + CLI + sanity tests
├── comparison_plots.py                    ← cross-construct comparison plots (fast rerun from CSVs)
├── plotting_montage_crops.py              ← representative cell-crop montage PDFs
├── plotting.py                            ← shared publication styling & stats helpers
└── config.yaml                            ← parameter configuration for the pipeline
```

## File Overview

| File | Purpose |
|------|---------|
| `run_analysis.py` | **Entry point** — processes 4 CoF constructs end-to-end with cross-construct comparison |
| `burst_quantification.py` | **Core module** — loading, QC, preprocessing, normalization, burst calling, dual-channel kymograph, and CLI |
| `comparison_plots.py` | **Comparison re-runner** — regenerates the cross-construct box-with-swarm comparison plots and MW U statistics directly from saved CSVs |
| `plotting_montage_crops.py` | Generates representative cell-crop montage PDFs with intensity traces, ON/OFF state bars, and thumbnail crops |
| `plotting.py` | Shared publication plotting style, figure saving, and comparison statistics helpers |
| `config.yaml` | YAML configuration file defining the thresholds, QC limits, and plotting layout parameters |

---

## Channel Semantics

| Channel | Signal | Role |
|---------|--------|------|
| **Channel 1** | Nascent protein (SunTag/scFv-sfGFP) | Tracking — always present, defines trajectory identity |
| **Channel 0** | Folding reporter | **Bursting signal** — quantified by this pipeline |

---

## Pipeline Stages

### Stage 1: Load Intensity Matrix

Loads a 2D matrix (**rows** = trajectories, **columns** = time frames) from:
- MicroLive tracking CSVs (via the analysis script)
- Direct NumPy arrays (notebook/programmatic use)
- CSV, TSV, NPY, or NPZ files (CLI use)

```python
from burst_quantification import load_intensity_matrix
matrix, ids, metadata = load_intensity_matrix("data.csv")
```

### Stage 2: Quality Control & Preprocessing

Each trajectory is evaluated and either **kept**, **removed**, or marked **constitutive**:

| Condition | Action | QC Reason |
|-----------|--------|-----------|
| All NaN | Remove | `empty` |
| Valid fraction < 30% | Remove | `too_sparse` |
| Max internal NaN gap > 2 frames | Remove | `too_many_internal_nans` |
| Total scattered internal NaNs > 3 | Remove | `too_many_scattered_nans` |
| Dynamic range ≈ 0 **and** mean ≈ 0 | Remove | `flat_zero` |
| Dynamic range ≈ 0 **but** mean > 0 | **Keep** (`qc_status="kept"`, `qc_reason="constitutive"`) | Constitutive (always-ON) |
| Dropped by `shift_trajectories` | Remove | `shift_alignment_filter` |

After QC:
1. **Left-align** first valid frame (optional, via MicroLive `shift_trajectories`)
2. **Forward-fill** NaN gaps for smoothing only
3. **Median smooth** (window = 3 frames)
4. **Re-stamp NaN mask** from original data — filled frames do **not** become event evidence

### Stage 3: Per-Trace Percentile Normalization

Used for **kymograph visualization only** and for `normalized_absolute`
thresholds. The active `snr` burst call uses the per-frame SNR matrix,
not this normalized matrix:

$$\hat{I}_i(t) = \frac{I_i(t) - P_5}{P_{95} - P_5} \quad \text{clipped to } [0, 1]$$

### Stage 4: Thresholding → Binary ON/OFF

Each per-trace intensity is converted to a binary ON/OFF call using one of five threshold modes (see [Threshold Modes](#threshold-modes) below). The **active method** for this paper is `snr`:

**`snr`** (active default) — per-frame SNR thresholding:

$$B_i(t) = \begin{cases} 1 \text{ (ON)} & \text{if } \text{SNR}_i(t) \geq \theta \\ 0 \text{ (OFF)} & \text{otherwise} \end{cases}$$

with $\theta = 3.5$. SNR is a self-normalized quality metric computed during spot detection (signal / local noise), so the same cutoff applies uniformly across constructs with different absolute brightness. Requires an SNR matrix to be provided alongside the intensity matrix.

**`fraction_of_trace_max`** — matches [Goldman et al., Mol Cell 2023](https://doi.org/10.1016/j.molcel.2023.06.003):

$$B_i(t) = \begin{cases} 1 \text{ (ON)} & \text{if } I_{\text{proc},i}(t) \geq \theta \cdot \max(I_{\text{proc},i}) \\ 0 \text{ (OFF)} & \text{otherwise} \end{cases}$$

with $\theta = 0.05$. Bar is anchored on the processed/smoothed trace peak.
For transparency, `threshold_used` stores $\theta$ and `threshold_value_used`
stores the actual per-trace intensity cutoff.

**`off_baseline_mad`** — OFF-baseline + k · MAD<sub>off</sub>, robust against noise-dominated traces:

$$B_i(t) = \begin{cases} 1 \text{ (ON)} & \text{if } I_{\text{proc},i}(t) \geq \mu_{\text{off},i} + k \cdot \tilde{\sigma}_{\text{off},i} \\ 0 \text{ (OFF)} & \text{otherwise} \end{cases}$$

where $\mu_{\text{off},i}$ is the **median** of the lowest-Q quantile of finite frames (`off_baseline_quantile`, default Q = 25%) and $\tilde{\sigma}_{\text{off},i} = 1.4826 \cdot \text{MAD}(I_{\text{off},i})$ is the robust σ-equivalent. The `threshold` parameter is interpreted as $k$ (default 4.0). The bar is anchored on the per-trace noise floor rather than the peak, so traces whose peak is set by a single outlier frame no longer drag the bar down into the noise band. This is closer in spirit to the original Wu-lab `5σ_dark + ⟨I_dark⟩` rule and produces biologically interpretable fraction-ON values (~0.2–0.4) where `fraction_of_trace_max` saturates near 1.

Post-thresholding event processing and cleanup:

After initial thresholding, the binary matrix goes through a robust sequence of filtering and refinement steps:
1. **Step 2 (Short-event merging)**: Events shorter than `min_event_duration_frames` (default 3 frames / 15 s) are merged into the neighbouring state.
2. **Step 2b (Short-ON relabeling)**: ON runs (bursts) shorter than `min_burst_duration_seconds` (default 30.0 s) are re-labeled as OFF. This ensures transient spikes do not inflate `fraction_time_on` or fragment surrounding OFF dwells. Following this step, `binary_matrix` represents accepted ON episodes.
3. **Step 2c (NaN gap bridging)**: Internal NaN gaps $\leq$ `max_internal_nan_gap` (default 2 frames / 10 s) are bridged between same-state neighbours. Because bridging is executed *after* failed-ON relabeling, patterns like `OFF-NaN-shortON-NaN-OFF` become `OFF-NaN-OFF-NaN-OFF` and are successfully bridged into a single continuous `OFF` dwell, preventing dwell fragmentation.
4. **Step 3 (Censoring of boundary dwells)**: 
   - **Initial dwells** (observation starts in the OFF state, left-censored) are excluded from dwell statistics when `count_initial_dwell` is `False` (default).
   - **Terminal dwells** (observation ends in the OFF state, right-censored) are excluded from dwell statistics when `exclude_terminal_dwell` is `True` (default).

### Stage 5: Event Table & Outputs

Two tables are generated:

**Per-event table** (`event_table.csv`):

| Column | Description |
|--------|-------------|
| `trajectory_id` | Trajectory identifier |
| `event_type` | `burst` or `dwell` |
| `start_frame`, `end_frame_exclusive` | Frame range |
| `duration_seconds`, `duration_minutes` | Duration in time units |
| `mean_raw_intensity` | Mean raw intensity during event |
| `area_raw`, `area_normalized` | Integrated intensity × Δt |
| `is_initial_dwell`, `is_terminal_event` | Boundary flags |
| `passes_duration_filter` | Whether event meets minimum duration |
| `threshold_used`, `threshold_value_used`, `threshold_source` | Threshold parameter, actual per-trace cutoff, and signal scale used |

**Per-trajectory summary** (`trajectory_summary.csv`):

| Column | Description |
|--------|-------------|
| `n_bursts`, `n_dwells` | Event counts (dwells exclude terminal) |
| `fraction_time_on` | Fraction of valid frames in ON state |
| `mean_burst_duration_minutes` | Mean burst duration |
| `mean_dwell_duration_minutes` | Mean dwell duration |
| `max_processed_intensity` | Maximum processed/smoothed intensity for that trajectory |
| `threshold_used`, `threshold_value_used`, `threshold_source` | Threshold parameter, actual per-trace cutoff, and signal scale used |
| `qc_status` | `kept` or `constitutive` |

---

## Diagnostic Plots

All plots are saved in **dual format** (PNG for preview + SVG for publication).

| Plot | Filename | Description |
|------|----------|-------------|
| Mean trace | `mean_trace_raw_vs_processed` | Population-average intensity (raw vs. smoothed) with SEM |
| Dual-channel kymograph | `kymograph_dual_channel` | Additive RGB kymograph (Green = folding ch0, Magenta = nascent ch1), sorted by trajectory length (longest first), QC-passed trajectories only. Uses per-trace percentile normalization. |
| SNR dual-channel kymograph | `kymograph_dual_channel_snr` | Same additive RGB layout as the intensity kymograph, but input data is per-frame SNR instead of intensity. Uses **fixed-range normalization**: SNR values are linearly mapped from `[0, SNR_CAP]` to `[0, 1]`, where `SNR_CAP = threshold × 2` (e.g. 6.0 when threshold = 3.0). This means SNR = 0 → black, SNR = threshold → 50% brightness, SNR ≥ SNR_CAP → full color. No per-trace rescaling is applied, so absolute SNR is directly interpretable and dim trajectories genuinely appear dim. |
| Representative montages | `montages_{construct}.pdf` | Cell-crop thumbnails + intensity traces + ON/OFF state bars for selected particles. Individual PNG/SVG in `plots_time_courses/`. Generated by `plotting_montage_crops.py`. |
| Burst durations | `burst_duration_distribution` | Histogram + CDF of burst durations (min) |
| Dwell durations | `dwell_duration_distribution` | Histogram + CDF of dwell durations (min, terminal excluded) |
| Fraction ON | `fraction_time_on_distribution` | Histogram of per-trajectory fraction ON |
| QC summary | `qc_summary` | Bar chart of QC outcomes (ok, constitutive, removed reasons) |

Plot styling is centralized in `plotting.py` so burst diagnostics, comparison
whisker plots, kymographs, and time-course panels share the same white
background, Arial text, black axes, and original green/magenta channel colors.
To avoid pseudoreplication and scientific bias, the cross-construct comparison box plots and statistics aggregate data at the **trajectory level** (i.e. computing one value per trajectory—the trajectory's median burst or dwell duration—rather than pooling all individual events across all trajectories).

The comparison plots perform pairwise two-sided Mann-Whitney U tests, draw significance brackets, and output a detailed stats table as `pairwise_mannwhitney_stats.csv` containing both raw p-values and Benjamini-Hochberg adjusted p-values (`p_adjusted_bh`). By default (`use_bh_fdr: false` in plot configuration), the significance stars are drawn using raw p-values to preserve the visual styling of prior figures. If `use_bh_fdr: true` is configured, significance stars in both the plots and the output table will reflect the BH-FDR adjusted p-values.

---

## Constructs

The analysis script processes these four constructs from the CoF long-movie dataset:

| Construct | Plasmid | Short Name | Full Name |
|-----------|---------|------------|-----------|
| sfGFP | pRS027 | 4sf | 4sfGFP-2mCh |
| GFPuv | pRS032 | 4uv | 4GFPuv-2mCh |
| sfGFP_Xbp1 | pRS038 | 4sf-Xbp1 | 4sfGFP-2mCh-Xbp1 |
| Xbp1_sfGFP | pRS048 | Xbp1-4sf | Xbp1-4sfGFP-2mCh |

Names are resolved from `utilities/config.py` (`REPORTER_PLASMID_NAME_MAPPING`, `PLASMID_SHORT_NAME_MAPPING`).

---

## Usage

### Option A: Full Analysis Script (recommended)

Processes all 4 constructs, generates per-construct diagnostics and cross-construct comparison:

```bash
python run_analysis.py
```

**Requirements:** External drive mounted at `/Volumes/Luis_DRIVE/CoF Manuscript LIFs/CoF_long_movies/`

**Output structure:**
```
results/burst_quantification/
├── diagnostics/
│   ├── detrend_diagnostic.png/.svg
│   └── signal_contrast_diagnostic.png/.svg
├── 4sf/                          # per-construct
│   ├── quantification/
│   │   ├── params.json
│   │   ├── raw_matrix.npy
│   │   ├── processed_matrix.npy
│   │   ├── normalized_matrix.npy
│   │   ├── binary_matrix.npy
│   │   ├── qc_table.csv
│   │   ├── event_table.csv
│   │   └── trajectory_summary.csv
│   └── plots/
│       ├── montages_{construct}.pdf
│       ├── plots_time_courses/               ← individual PNG+SVG per particle
│       └── quality_control/
│           ├── kymograph_dual_channel.png/.svg
│           ├── kymograph_dual_channel_snr.png/.svg
│           ├── mean_trace_raw_vs_processed.png/.svg
│           ├── burst_duration_distribution.png/.svg
│           ├── dwell_duration_distribution.png/.svg
│           ├── fraction_time_on_distribution.png/.svg
│           └── qc_summary.png/.svg
├── 4uv/
├── 4sf-Xbp1/
├── Xbp1-4sf/
└── comparison/
    ├── run_analysis/                 # written by run_analysis.py
    │   ├── plots/
    │   │   ├── burst_duration_comparison.png/.svg
    │   │   ├── dwell_duration_comparison.png/.svg
    │   │   └── fraction_on_comparison.png/.svg
    │   └── quantification/
    │       ├── pairwise_mannwhitney_stats.csv
    │       └── summary_table.csv
    ├── all_constructs/               # written by comparison_plots.py
    │   ├── plots/
    │   └── quantification/
    ├── sf_vs_uv/
    │   ├── plots/
    │   └── quantification/
    └── xbp1_context/
        ├── plots/
        └── quantification/
```

### Option B: Programmatic / Notebook Use

```python
from burst_quantification import run_burst_quantification

# Default: per-trace-max threshold (Goldman et al. 2023)
result = run_burst_quantification(
    input_matrix=my_matrix,           # numpy array (n_traces × n_timepoints)
    output_dir="results/burst_quantification/my_construct",
    condition="My Construct",
    time_interval_seconds=5.0,
    threshold=0.05,
    threshold_mode="fraction_of_trace_max",
)

# Alternative: OFF-baseline + 4·MAD threshold (noise-floor anchored)
result_mad = run_burst_quantification(
    input_matrix=my_matrix,
    output_dir="results/burst_quantification/my_construct_mad",
    condition="My Construct",
    time_interval_seconds=5.0,
    threshold=4.0,                    # k_MAD multiplier
    threshold_mode="off_baseline_mad",
    off_baseline_quantile=0.25,       # bottom 25% defines the OFF pool
)

# Access results
binary = result["binary_matrix"]        # 0/1/NaN array
events = result["event_table"]          # pandas DataFrame
summary = result["trajectory_summary"]  # pandas DataFrame
```

### Option C: CLI

```bash
# Default mode: fraction_of_trace_max with θ=0.05
python burst_quantification.py \
    --input raw_matrix.npy \
    --output results/burst_quantification/run01 \
    --dt 5 --threshold 0.05 \
    --threshold-mode fraction_of_trace_max \
    --condition "sfGFP"

# Alternative mode: off_baseline_mad with k=4 (anchored on noise floor)
python burst_quantification.py \
    --input raw_matrix.npy \
    --output results/burst_quantification/run01_mad \
    --dt 5 --threshold 4.0 \
    --threshold-mode off_baseline_mad \
    --off-baseline-quantile 0.25 \
    --condition "sfGFP"

# Run sanity tests
python burst_quantification.py --test
```

---

## Key Parameters

All parameters live in `config.yaml`. Loader keys (SNR filter, `shift_trajectories`) and burst-module keys (threshold, smoothing, normalization) are kept together as one source of truth.

### Burst Module Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_interval_seconds` | 5.0 | Seconds per frame |
| `threshold` | 3.0 | Threshold value (meaning depends on mode — see [Threshold Modes](#threshold-modes)). For `snr`: SNR cutoff (standard default is `3.0`); for `fraction_of_trace_max`: θ; for `off_baseline_mad`: k multiplier. |
| `threshold_mode` | `snr` | One of `snr`, `fraction_of_trace_max`, `off_baseline_mad`, `normalized_absolute`, `absolute_raw` |
| `off_baseline_quantile` | 0.50 | For `off_baseline_mad` only. Fraction of lowest-intensity frames defining the per-trace OFF pool used to estimate baseline + MAD<sub>off</sub>. |
| `min_valid_fraction` | 0.30 | Minimum fraction of finite frames to keep a trajectory (e.g. 30% of 360 frames = 108 valid frames) |
| `max_total_internal_nan_frames` | 3 | Maximum total scattered internal NaN frames allowed by MicroLive alignment |
| `max_internal_nan_gap` | 2 | Maximum consecutive internal NaN frames allowed and bridged during event segmentation |
| `smooth_method` | `median` | Smoothing filter type (`median`, `mean`, `gaussian`, `none`) |
| `smooth_window` | 3 | Smoothing window size (frames) |
| `min_event_duration_frames` | 3 | Events shorter than this are merged (15 s at 5 s/frame) |
| `min_burst_duration_seconds` | 30.0 | Bursts/ON episodes shorter than this are flagged (`passes_duration_filter = False`) and re-labeled as OFF before NaN bridging |
| `exclude_terminal_dwell` | `True` | Exclude terminal dwells (right-censored) from dwell statistics |
| `count_initial_dwell` | `False` | Exclude initial dwells (left-censored) from dwell statistics |

### Tracking-Loader Parameters

These control how raw `tracking_*.csv` files are read by `_load_construct_matrix` before burst-quantification preprocessing. They also live in `PARAMS`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_snr` | 1 | Minimum per-particle mean SNR. Particles below this threshold are dropped at load. |
| `snr_channel_index` | 1 | **Which channel's SNR to filter on.** Defaults to channel 1 (nascent / tracking), so QC reflects spot-localization quality rather than how bright the folding signal happens to be. Set to `None` to filter on whichever channel is being loaded (legacy behavior). |

#### Why `snr_channel_index` defaults to 1

The nascent channel (ch1) is the stable tracking channel — every translating mRNA produces nascent-protein signal whenever ribosomes are present, so its SNR uniformly reflects tracking quality across constructs. The folding channel (ch0) has dramatically different SNR by construct (median ranges 3–8 across the 4 CoF constructs), because constructs that fold dimly will have low ch0 SNR for *biological* reasons, not tracking failure.

Filtering on ch0 SNR confounds tracking-quality QC with the signal-of-interest brightness: at `min_snr=1`, ch0-based filtering would cull a significant fraction of dimmer particles. ch1-based filtering keeps 100% of particles across all four constructs. See "Threshold Modes" and the OFF-baseline discussion for the analogous distinction in burst calling.

### Plot & Comparison Parameters

These control figure generation, sorting, and statistical adjustments. They live in the `plots` block of the configuration file.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_bh_fdr` | `false` | Whether to apply Benjamini-Hochberg FDR correction to pairwise Mann-Whitney U test p-values for star annotations in comparison plots. If `false` (default), raw p-values are used for plotting annotations to match prior figures. Both raw and adjusted p-values are written to `pairwise_mannwhitney_stats.csv`. |
| `kymograph_sort_by` | `"density"` | Sorting method for the dual-channel kymograph traces: `"density"` sorts by trajectory length (longest first); `"fraction_on"` sorts by ON time fraction. |
| `max_traces_to_plot` | `null` | Maximum number of traces displayed in the kymograph. Set to `null` to show all QC-passing trajectories. |

### Threshold Modes

| Mode | Formula | Typical `threshold` | Use Case |
|------|---------|---------------------|----------|
| `snr` | ON if `SNR(t) ≥ θ` | 3.5 | **Active default.** Per-frame SNR from spot detection. Self-normalized by local noise, so the same cutoff works across constructs with different absolute brightness. Requires `snr_matrix`. |
| `fraction_of_trace_max` | ON if `I_proc ≥ θ · max(I_proc)` | 0.05 | Anchored on the processed/smoothed per-trace peak. Best for clean bimodal traces with high contrast (Goldman et al. 2023). On noisy traces the bar can fall into the noise band and saturate `fraction_on → 1`. |
| `off_baseline_mad` | ON if `I ≥ median(bottom Q%) + θ · 1.4826 · MAD_off` | 4.0 | Anchored on per-trace **noise floor**. Robust to outlier peaks and to background-subtracted traces with negative values. Produces biologically interpretable fraction-ON values (~0.2–0.4). Q is controlled by `off_baseline_quantile`. |
| `normalized_absolute` | ON if `Î ≥ θ` (on normalized [0,1]) | 0.3–0.5 | Sensitivity analysis against the percentile-normalized matrix. |
| `absolute_raw` | ON if `I_raw ≥ θ` | construct-specific | When you have an externally calibrated intensity cutoff. |

#### Choosing between `fraction_of_trace_max` and `off_baseline_mad`

These two modes can give very different answers on the same data:

- **Use `fraction_of_trace_max`** when traces have **clean bimodal signal** (`max/median ≫ 1`, see the [Signal Contrast Validation](#signal-contrast-validation) diagnostic). The bar lands just above zero and you capture long sustained bursts at their full extent.
- **Use `off_baseline_mad`** when traces have **near-zero or negative background** (typical of background-subtracted spot intensities) or when the **per-trace peak is dominated by a single outlier frame**. The bar sits above the noise floor regardless of where the peak is, so noise-dominated traces are correctly called as mostly-OFF rather than nearly all-ON.

To pick numerically, run both modes and compare `mean_fraction_on` in the trajectory summary. If `fraction_of_trace_max` gives `frac_on ≳ 0.8` for most trajectories, the bar has saturated and `off_baseline_mad` is the better choice.

---

## Signal Contrast Validation

The `fraction_of_trace_max` threshold assumes bimodal signal (high bursts vs. near-zero background). The analysis script includes a **signal contrast diagnostic** (Step 1.5) that plots the `max(FI) / median(FI)` ratio per trace:

- **Ratio >> 1** (e.g., > 5): Strong contrast, threshold is appropriate
- **Ratio ≈ 1**: Signal is unimodal (noise-like), consider using `normalized_absolute` with a higher threshold instead

---

## Sanity Tests

Ten built-in tests validate the pipeline:

```bash
python burst_quantification.py --test
```

| Test | Validates |
|------|-----------|
| `all_zero_matrix` | Zero-valued traces removed (flat_zero) |
| `all_nan_matrix` | Fully empty traces removed |
| `constant_trace_constitutive` | Constant-high traces **kept** as constitutive with fraction ON ≈ 1.0 |
| `single_square_pulse` | Clean ON/OFF transition detected as 1 burst |
| `short_run_removal` | Events < 6 frames merged away |
| `mixed_validity` | Mixed QC outcomes handled correctly |
| `large_matrix_500x200` | Stress test (500 traces) |
| `noise_normalized_absolute` | `normalized_absolute` @ 0.5 gives moderate ON fraction on noise |
| `noise_off_baseline_mad` | `off_baseline_mad` @ k=4 calls pure noise as mostly-OFF (frac_on < 0.30) |
| `bimodal_off_baseline_mad` | `off_baseline_mad` @ k=4 on baseline+periodic pulses gives frac_on in (0.10, 0.45) |

---

## Terminology Note

Events labeled "burst" and "dwell" are **observed folding-channel ON/OFF episodes**. They reflect the combined effect of translation initiation, ribosome transit, co-translational folding, and fluorophore maturation — not direct measurements of a single kinetic step. See the `terminology_note` field in `params.json` for the disclaimer saved with each run.

---

## Dependencies

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib
- MicroLive (optional — enables `shift_trajectories`, `detrend_trajectories`, `forward_fill_nan_2d`)
- `utilities/config.py` (construct name mappings)

---

## References

- Goldman DH, et al. "Live-cell imaging reveals kinetic determinants of quality control triggered by ribosome stalling." *Molecular Cell* 83(11):1830–1840 (2023). [DOI](https://doi.org/10.1016/j.molcel.2023.06.003)
