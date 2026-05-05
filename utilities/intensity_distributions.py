"""
Intensity distribution analysis functions.

Provides tools for extracting and visualizing per-particle intensity
distributions from MicroLive tracking data.

Display Modes
-------------
Mode 1 ("mean per particle"):
    For each tracked particle, compute the time-averaged intensity across
    all frames it appears in.  The distribution contains ONE value per
    particle.  This removes frame-to-frame noise and reflects the
    *characteristic* brightness of each molecule.

Mode 2 ("all timepoints"):
    Every individual (particle × frame) intensity measurement is treated
    as an independent sample.  The distribution contains MANY values per
    particle (one per frame).  This preserves temporal fluctuations and
    yields a much larger sample size.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu

# ── Shared style constants (match plot_swarm_plot in utilities/plotting.py) ───
_FONT = 'Arial'
_TICK_SIZE = 12
_LABEL_SIZE = 14
_TITLE_SIZE = 13

# ── Data Loading ──────────────────────────────────────────────────────────────

def _find_tracking_file(folder_path: Path, dataframe_prefix: str) -> Path:
    """Return the single tracking CSV in a results folder."""
    tracking_files = sorted([
        file_path for file_path in folder_path.iterdir()
        if file_path.suffix.lower() == '.csv'
        and file_path.name.startswith(dataframe_prefix)
        and not file_path.name.startswith('._')
    ])
    if not tracking_files:
        raise FileNotFoundError(
            f"No {dataframe_prefix}*.csv found in {folder_path.name}"
        )
    if len(tracking_files) > 1:
        names = ', '.join(file_path.name for file_path in tracking_files)
        raise ValueError(
            f"Expected one {dataframe_prefix}*.csv in {folder_path.name}; found {names}"
        )
    return tracking_files[0]


def _get_particle_column(tracking_df: pd.DataFrame) -> str:
    """Return the best particle identifier column."""
    if 'unique_particle' in tracking_df.columns:
        return 'unique_particle'
    if 'particle' in tracking_df.columns:
        return 'particle'
    raise ValueError("Tracking CSV must contain 'unique_particle' or 'particle'")


def _prepare_tracking_dataframe(
    tracking_df: pd.DataFrame,
    int_col: str,
    snr_col: str,
    min_snr: float,
    drop_nonpositive: bool,
) -> pd.DataFrame:
    """Filter tracking rows for the requested intensity channel."""
    if tracking_df.empty:
        raise ValueError('Tracking CSV is empty')
    if int_col not in tracking_df.columns:
        raise ValueError(f"Missing required intensity column '{int_col}'")
    if min_snr is not None:
        if snr_col not in tracking_df.columns:
            raise ValueError(f"Missing required SNR column '{snr_col}'")
        tracking_df = tracking_df[tracking_df[snr_col] >= min_snr].copy()
    valid_mask = tracking_df[int_col].notna()
    if drop_nonpositive:
        valid_mask = valid_mask & (tracking_df[int_col] > 0)
    tracking_df = tracking_df[valid_mask].copy()
    if tracking_df.empty:
        raise ValueError('No rows remain after SNR/intensity filtering')
    return tracking_df


def load_condition_intensities(
    dataframes_dir: Path,
    folder_substring: str,
    selected_field: str = 'spot_int_ch_',
    channel_index: int = 1,
    snr_field: str = 'snr_ch_',
    min_snr: float = 0,
    dataframe_prefix: str = 'tracking_',
    drop_nonpositive: bool = False,
    strict: bool = True,
    verbose: bool = False,
) -> dict:
    """Load intensity data for a single experimental condition.

    Both conditions share the same dataframes_dir; folder_substring
    selects which results_* sub-folders belong to this condition.

    Args:
        dataframes_dir: Top-level folder containing all results_* sub-folders.
        folder_substring: Substring to match in folder names (e.g. '20260422 pNZ212').
        selected_field: Column-name prefix for intensity (e.g. 'spot_int_ch_').
        channel_index: Channel number appended to selected_field.
        snr_field: Column-name prefix for SNR filtering.
        min_snr: Minimum SNR to keep a data point. Set None to skip SNR filtering.
        dataframe_prefix: Filename prefix for tracking CSVs inside each folder.
        drop_nonpositive: If True, discard rows with intensity <= 0 before
            averaging.  Set False to keep negative/zero background-corrected
            values (important for the folding channel).
        strict: If True, raise on malformed/missing data. If False, skip bad
            folders and report them in the returned skipped_folders list.
        verbose: If True, print progress information.

    Returns:
        dict with keys per_cell, pooled_mean_per_particle,
        pooled_all_timepoints, n_cells, n_particles_total, skipped_folders.

    Raises:
        ValueError: If no matching folders or no valid data found.
    """
    dataframes_dir = Path(dataframes_dir)
    if not dataframes_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {dataframes_dir}")
    int_col = f'{selected_field}{channel_index}'
    snr_col = f'{snr_field}{channel_index}'
    matching_folders = sorted([
        f for f in dataframes_dir.iterdir()
        if f.is_dir() and f.name.startswith('results_') and folder_substring in f.name
    ], key=lambda x: x.name)

    if not matching_folders:
        raise ValueError(
            f"No results_* folders matching '{folder_substring}' in {dataframes_dir}"
        )

    per_cell = []
    all_mean_per_particle = []
    all_timepoints_pool = []
    skipped_folders = []

    for folder_path in matching_folders:
        try:
            tracking_file_path = _find_tracking_file(folder_path, dataframe_prefix)
            df = pd.read_csv(tracking_file_path, encoding='latin-1')
            particle_col = _get_particle_column(df)
            df = _prepare_tracking_dataframe(
                df, int_col, snr_col, min_snr, drop_nonpositive
            )
            if df[particle_col].isna().any():
                raise ValueError(f"Missing particle IDs in '{particle_col}'")
            cell_all_timepoints = df[int_col].values.copy()
            particle_means = df.groupby(particle_col)[int_col].mean().values
            cell_name = folder_path.name.replace('results_', '')
            per_cell.append({
                'cell_name': cell_name,
                'mean_per_particle': particle_means,
                'all_timepoints': cell_all_timepoints,
                'n_particles': len(particle_means),
                'n_observations': len(cell_all_timepoints),
                'particle_column': particle_col,
            })
            all_mean_per_particle.append(particle_means)
            all_timepoints_pool.append(cell_all_timepoints)
            if verbose:
                print(f"  Loaded {cell_name}: {len(particle_means)} particles, "
                      f"{len(cell_all_timepoints)} observations")
        except Exception as error:
            if strict:
                raise
            skipped_folders.append({'folder': folder_path.name, 'reason': str(error)})
            if verbose:
                print(f"  SKIP {folder_path.name}: {error}")

    if not per_cell:
        raise ValueError(f"No valid data found for '{folder_substring}'")

    pooled_mpp = np.concatenate(all_mean_per_particle)
    pooled_at = np.concatenate(all_timepoints_pool)

    result = {
        'per_cell': per_cell,
        'pooled_mean_per_particle': pooled_mpp,
        'pooled_all_timepoints': pooled_at,
        'n_cells': len(per_cell),
        'n_particles_total': len(pooled_mpp),
        'n_observations_total': len(pooled_at),
        'skipped_folders': skipped_folders,
    }
    if verbose:
        print(f"  → {result['n_cells']} cells, {result['n_particles_total']} particles total")
    return result

# ── X-limit Helpers ───────────────────────────────────────────────────────────

def _get_data_for_mode(result, mode):
    """Return the appropriate pooled array for the given mode."""
    if mode == 1:
        return result['pooled_mean_per_particle']
    elif mode == 2:
        return result['pooled_all_timepoints']
    raise ValueError(f"mode must be 1 or 2, got {mode}")


def _get_cell_data_for_mode(cell, mode):
    """Return the appropriate per-cell array for the given mode."""
    if mode == 1:
        return cell['mean_per_particle']
    elif mode == 2:
        return cell['all_timepoints']
    raise ValueError(f"mode must be 1 or 2, got {mode}")


def _finite_values(data, drop_nonpositive=False):
    """Return finite values, optionally excluding values <= 0."""
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    if drop_nonpositive:
        values = values[values > 0]
    return values


def _auto_xlim(data, percentiles=(1, 99), drop_nonpositive=False):
    """Compute auto x-limits from data using percentiles + 5% padding."""
    values = _finite_values(data, drop_nonpositive=drop_nonpositive)
    if len(values) == 0:
        return (0, 1)
    lo = np.percentile(values, percentiles[0])
    hi = np.percentile(values, percentiles[1])
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return (lo - pad, hi + pad)


def _shared_bin_edges(data, bins, drop_nonpositive=False):
    """Return common histogram bin edges over the full data range."""
    if not isinstance(bins, (int, np.integer)):
        return bins
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    values = _finite_values(data, drop_nonpositive=drop_nonpositive)
    if len(values) == 0:
        return np.linspace(0, 1, int(bins) + 1)
    lo = np.min(values)
    hi = np.max(values)
    if hi <= lo:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, int(bins) + 1)

# ── Per-Cell Plots ────────────────────────────────────────────────────────────

def plot_per_cell_distributions(
    result: dict,
    color: str,
    save_dir: Path,
    channel_index: int,
    mode: int = 1,
    xlim=None,
    xlim_percentiles: tuple = (1, 99),
    kde_bw_method='scott',
    kde_alpha: float = 0.25,
    bins: int = 40,
    drop_nonpositive: bool = False,
):
    """Plot a multi-panel figure (one subplot per cell/image) for one condition.

    Args:
        result: Output of load_condition_intensities with a 'name' key.
        color: Colour for histograms / KDE.
        save_dir: Directory to save figures.
        channel_index: Channel number (for filename / title).
        mode: 1 = mean per particle, 2 = all timepoints.
        xlim: Fixed x-axis limits, or None for auto (percentile-based).
        xlim_percentiles: (lo, hi) percentiles for auto x-limits.
        kde_bw_method: Bandwidth for gaussian_kde (default 'scott').
        kde_alpha: Fill alpha for KDE / histogram.
        bins: Number of histogram bins, or explicit shared bin edges.
        drop_nonpositive: If True, exclude intensity <= 0 from plots.
    """
    cells = result['per_cell']
    n = len(cells)
    if n == 0:
        return

    sns.set_style('ticks')

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    # Auto x-limits from pooled data
    if xlim is None:
        xlim = _auto_xlim(_get_data_for_mode(result, mode), xlim_percentiles,
                          drop_nonpositive=drop_nonpositive)
    hist_bins = _shared_bin_edges(
        _get_data_for_mode(result, mode),
        bins,
        drop_nonpositive=drop_nonpositive,
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False, facecolor='white')
    cond_name = result.get('name', 'condition')

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor('white')
        vals = _finite_values(_get_cell_data_for_mode(cell, mode), drop_nonpositive)
        if len(vals) == 0:
            ax.set_visible(False)
            continue

        ax.hist(vals, bins=hist_bins, density=True, color=color,
                alpha=kde_alpha, edgecolor='none')

        if len(vals) > 3:
            try:
                xs = np.linspace(xlim[0], xlim[1], 300)
                ys = gaussian_kde(vals, bw_method=kde_bw_method)(xs)
                ax.plot(xs, ys, color=color, linewidth=1.5)
            except Exception:
                pass

        ax.axvline(np.median(vals), color=color, linestyle='--',
                   linewidth=1.0, alpha=0.7)

        count_label = cell['n_particles'] if mode == 1 else cell['n_observations']
        ax.set_title(f"{cell['cell_name']}\nn = {count_label}",
                     fontsize=9, fontname=_FONT, color='black')
        ax.set_xlim(xlim)
        ax.tick_params(labelsize=8, colors='black')
        ax.set_xlabel('', fontname=_FONT, color='black')
        ax.set_ylabel('', fontname=_FONT, color='black')

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    mode_label = 'mean_per_particle' if mode == 1 else 'all_timepoints'
    fig.suptitle(f'{cond_name} — ch{channel_index} ({mode_label})',
                 fontsize=_TITLE_SIZE, fontweight='bold', fontname=_FONT, color='black')
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(save_dir / f'{cond_name}_per_cell_mode{mode}_ch{channel_index}.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ── Pooled Overlay Plot ───────────────────────────────────────────────────────

def plot_pooled_distributions(
    all_results: list,
    list_colors: list,
    channel_index: int,
    x_label: str,
    save_dir: Path,
    mode: int = 1,
    xlim=None,
    xlim_percentiles: tuple = (1, 99),
    figsize: tuple = (5, 5),
    bins: int = 60,
    kde: bool = True,
    show_hist: bool = False,
    print_stats: bool = True,
    kde_bw_method='scott',
    kde_alpha: float = 0.25,
    drop_nonpositive: bool = False,
):
    """Overlay KDE / histogram for all conditions on a single panel.

    Args:
        all_results: List of dicts from load_condition_intensities with 'name'.
        list_colors: One colour per condition.
        channel_index: Channel number (for filename).
        x_label: X-axis label.
        save_dir: Directory to save figures.
        mode: 1 = mean per particle, 2 = all timepoints.
        xlim: Fixed x-limits or None for auto.
        xlim_percentiles: Percentiles for auto x-limits.
        show_hist: If True, draw density-normalised histogram behind KDE.
        print_stats: If True, print mean/median/std per condition to stdout.
        kde_bw_method: Bandwidth for gaussian_kde.
        kde_alpha: Fill alpha for overlapping distributions.
        drop_nonpositive: If True, exclude intensity <= 0 from plots.
        bins: Number of histogram bins, or explicit shared bin edges.
    """
    sns.set_style('ticks')
    if len(all_results) != len(list_colors):
        raise ValueError('all_results and list_colors must have the same length')
    if not all_results:
        raise ValueError('all_results must contain at least one condition')
    combined = np.concatenate([_get_data_for_mode(r, mode) for r in all_results])
    if xlim is None:
        xlim = _auto_xlim(combined, xlim_percentiles,
                          drop_nonpositive=drop_nonpositive)
    hist_bins = _shared_bin_edges(combined, bins, drop_nonpositive=drop_nonpositive)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    has_plotted_data = False

    for r, color in zip(all_results, list_colors):
        vals = _finite_values(_get_data_for_mode(r, mode), drop_nonpositive)
        if vals.size == 0:
            continue
        has_plotted_data = True

        name = r.get('name', '?')
        label = f"{name}  ({r['n_cells']} cells, {r['n_particles_total']} particles)"

        # Optional histogram behind the KDE
        if show_hist:
            hist_alpha = kde_alpha * 0.6 if kde else kde_alpha
            ax.hist(vals, bins=hist_bins, density=True, color=color,
                    alpha=hist_alpha, edgecolor='none',
                    label=label if not kde else None)

        if kde and vals.size > 3:
            xs = np.linspace(xlim[0], xlim[1], 500)
            try:
                ys = gaussian_kde(vals, bw_method=kde_bw_method)(xs)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
                ax.fill_between(xs, ys, alpha=kde_alpha, color=color)
            except Exception:
                if not show_hist:
                    ax.hist(vals, bins=hist_bins, density=True, color=color,
                            alpha=kde_alpha, label=label)
        elif not show_hist:
            ax.hist(vals, bins=hist_bins, density=True, color=color,
                    alpha=kde_alpha, label=label)

        med = np.median(vals)
        ax.axvline(med, color=color, linewidth=1.0,
                   linestyle='--', alpha=0.7)

        # Print summary statistics
        if print_stats:
            print(f"  {name:<30s}  "
                  f"mean={np.mean(vals):.2f}   "
                  f"median={med:.2f}   "
                  f"std={np.std(vals):.2f}   "
                  f"n={vals.size}   "
                  f"n_cells={r['n_cells']}")
    if not has_plotted_data:
        raise ValueError('No finite values available for pooled plot')

    ax.set_xlabel(x_label, fontsize=_LABEL_SIZE, fontname=_FONT, color='black')
    ax.set_ylabel('Probability Density', fontsize=_LABEL_SIZE, fontname=_FONT, color='black')
    ax.tick_params(axis='both', which='major', labelsize=_TICK_SIZE, colors='black')
    ax.legend(fontsize=10, framealpha=0.85,
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=1, prop={'family': _FONT})
    ax.set_xlim(xlim)
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mode_label = 'mode1' if mode == 1 else 'mode2'
    for ext in ('svg', 'png'):
        fig.savefig(save_dir / f'intensity_dist_ch{channel_index}_{mode_label}.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ── Cell-Level Summary Plot (SuperPlot) ───────────────────────────────────────

def plot_cell_summary(
    all_results: list,
    list_colors: list,
    channel_index: int,
    save_dir: Path,
    mode: int = 1,
    cell_stat: str = 'median',
    drop_nonpositive: bool = False,
    show_test: bool = True,
    figsize: tuple = (4, 5),
):
    """Swarm + box plot of per-cell summary statistics (one point per cell).

    This avoids pseudoreplication by treating each cell as the independent
    biological unit rather than pooling all spots.

    Args:
        all_results: List of dicts from load_condition_intensities with 'name'.
        list_colors: One colour per condition.
        channel_index: Channel number (for filename / axis label).
        save_dir: Directory to save figures.
        mode: 1 = mean per particle, 2 = all timepoints.
        cell_stat: 'mean' or 'median' — summary computed per cell.
        drop_nonpositive: If True, exclude intensity <= 0 before summarising.
        show_test: If True and exactly 2 conditions, run Mann-Whitney U and
            print the p-value.
        figsize: Figure size (width, height).
    """
    if cell_stat not in ('mean', 'median'):
        raise ValueError(f"cell_stat must be 'mean' or 'median', got '{cell_stat}'")
    stat_fn = np.mean if cell_stat == 'mean' else np.median

    sns.set_style('ticks')

    # Build a tidy DataFrame: one row per cell
    rows = []
    for r in all_results:
        name = r.get('name', '?')
        for cell in r['per_cell']:
            vals = _finite_values(_get_cell_data_for_mode(cell, mode),
                                 drop_nonpositive)
            if vals.size == 0:
                continue
            rows.append({
                'Condition': name,
                'Cell': cell['cell_name'],
                f'Cell {cell_stat}': stat_fn(vals),
            })
    if not rows:
        raise ValueError('No cells with valid data for cell summary plot')
    summary_df = pd.DataFrame(rows)
    y_col = f'Cell {cell_stat}'

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Box plot (no outlier markers — swarm shows all points)
    palette = {r.get('name', '?'): c for r, c in zip(all_results, list_colors)}
    sns.boxplot(
        data=summary_df, x='Condition', y=y_col, ax=ax,
        palette=palette, width=0.45, showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='red', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )
    # Swarm overlay — one point per cell
    sns.swarmplot(
        data=summary_df, x='Condition', y=y_col, ax=ax,
        palette=palette, size=7, edgecolor='black', linewidth=0.5,
    )

    ax.set_ylabel(f'{y_col} intensity (ch {channel_index})',
                  fontsize=_LABEL_SIZE, fontname=_FONT, color='black')
    ax.set_xlabel('', fontsize=_LABEL_SIZE, fontname=_FONT, color='black')
    ax.tick_params(axis='both', which='major', labelsize=_TICK_SIZE, colors='black')

    # Print per-condition stats
    conditions = summary_df['Condition'].unique()
    for cond in conditions:
        vals = summary_df.loc[summary_df['Condition'] == cond, y_col].values
        print(f"  {cond:<30s}  "
              f"mean={np.mean(vals):.2f}   "
              f"median={np.median(vals):.2f}   "
              f"std={np.std(vals, ddof=1):.2f}   "
              f"n_cells={len(vals)}")

    # Mann-Whitney U test (only if exactly 2 conditions)
    if show_test and len(conditions) == 2:
        g1 = summary_df.loc[summary_df['Condition'] == conditions[0], y_col].values
        g2 = summary_df.loc[summary_df['Condition'] == conditions[1], y_col].values
        if len(g1) >= 2 and len(g2) >= 2:
            stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            print(f"  Mann-Whitney U: U={stat:.1f}, p={p_val:.4g}")
            # Annotate p-value on the figure
            y_max = summary_df[y_col].max()
            y_bar = y_max * 1.08
            ax.plot([0, 0, 1, 1], [y_bar, y_bar * 1.02, y_bar * 1.02, y_bar],
                    lw=1.0, color='black')
            p_text = f'p = {p_val:.4g}' if p_val >= 0.001 else f'p = {p_val:.2e}'
            ax.text(0.5, y_bar * 1.03, p_text, ha='center', va='bottom',
                    fontsize=10, fontname=_FONT, color='black')

    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mode_label = 'mode1' if mode == 1 else 'mode2'
    for ext in ('svg', 'png'):
        fig.savefig(
            save_dir / f'cell_summary_ch{channel_index}_{cell_stat}_{mode_label}.{ext}',
            dpi=300, bbox_inches='tight',
        )
    plt.show()
    plt.close(fig)
