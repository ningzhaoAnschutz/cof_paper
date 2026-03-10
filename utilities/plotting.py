"""
Plotting utilities for COF analysis.

This module provides visualization functions for:
- Swarm plots with statistical comparisons
- Efficiency plots
- Scatter plots comparing efficiency vs intensity
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from matplotlib.patches import Ellipse


def plot_swarm_plot(
    conditions_data, condition_labels, x_label="", y_label="Mean Value", title="",
    figsize=(6, 4), tick_size=14, swarm_color="black", y_lim=None, show_stats=False,
    only_significant=True, save_dir=None, plot_name='temp', max_percentile_significance=99.5,
    x_tick_rotation=0, show_n=True, swarm_size=6, min_spots_threshold=None,
):
    """
    Create a boxplot with swarm overlay and optional statistical comparisons.
    
    Each element in conditions_data should be an iterable of NumPy arrays,
    where each array represents one cell/repetition of a given condition.
    """
    sns.set_style("ticks")
    
    mean_values, condition_list = [], []
    cell_counts = {label: 0 for label in condition_labels}
    cells_excluded = {label: 0 for label in condition_labels}
    
    for cond_idx, repetitions in enumerate(conditions_data):
        if repetitions is None:
            continue
        for rep in repetitions:
            if rep is None:
                continue
            valid_spots = np.sum(~np.isnan(np.asarray(rep).flatten()))
            if min_spots_threshold is not None and valid_spots < min_spots_threshold:
                cells_excluded[condition_labels[cond_idx]] += 1
                continue
            rep_mean = np.nanmean(rep)
            mean_values.append(rep_mean)
            condition_list.append(condition_labels[cond_idx])
            cell_counts[condition_labels[cond_idx]] += 1
    
    if min_spots_threshold is not None:
        print(f"\n=== Cell Filtering (min_spots_threshold={min_spots_threshold}) ===")
        for label in condition_labels:
            print(f"  {label}: {cell_counts[label]}/{cell_counts[label] + cells_excluded[label]} cells kept")
    
    df = pd.DataFrame({"Mean": mean_values, "Condition": condition_list})
    valid_labels = [l for l in condition_labels if cell_counts[l] > 0]
    if not valid_labels:
        raise ValueError("No valid data found in any condition")
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    sns.boxplot(x="Condition", y="Mean", data=df, order=valid_labels, showfliers=False,
                boxprops={'facecolor': 'white', 'edgecolor': 'black'},
                medianprops={'color': 'red'}, whiskerprops={'color': 'black'},
                capprops={'color': 'black'}, linewidth=1.5, whis=[5, 95], width=0.5, ax=ax)
    sns.swarmplot(x="Condition", y="Mean", data=df, order=valid_labels, color=swarm_color, size=swarm_size, ax=ax)
    
    if show_n:
        labels_with_n = [f"{l}\n(n={cell_counts[l]})" for l in valid_labels]
        ax.set_xticklabels(labels_with_n, fontname="Arial", fontsize=tick_size, rotation=x_tick_rotation, ha='center')
    else:
        ax.set_xticklabels(valid_labels, fontname="Arial", fontsize=tick_size, rotation=x_tick_rotation,
                          ha='right' if x_tick_rotation else 'center')
    
    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial", color='black')
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial", color='black')
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color='black')
    if y_lim is not None and not show_stats:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.tight_layout()
    
    if show_stats and len(valid_labels) > 1:
        global_max = np.nanpercentile(df["Mean"], max_percentile_significance)
        global_min = np.nanmin(df["Mean"])
        global_range = global_max - global_min if (global_max - global_min) != 0 else 1
        offset = 0.10 * global_range
        bar_height = 0.02 * global_range
        k = 0
        for i in range(len(valid_labels) - 1):
            for j in range(i + 1, len(valid_labels)):
                g1 = df[df["Condition"] == valid_labels[i]]["Mean"].dropna()
                g2 = df[df["Condition"] == valid_labels[j]]["Mean"].dropna()
                p = mannwhitneyu(g1, g2, alternative='two-sided')[1] if len(g1) > 0 and len(g2) > 0 else np.nan
                sig = '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                if only_significant and sig == 'ns':
                    continue
                y_line = global_max + offset * (k + 1)
                ax.plot([i, i, j, j], [y_line, y_line + bar_height, y_line + bar_height, y_line], lw=1.2, c='k')
                ax.text((i + j) * 0.5, y_line + bar_height - 0.01 * global_range, sig, ha='center', va='bottom',
                        color='k', fontsize=tick_size - 2, fontname="Arial")
                k += 1
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches='tight')
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches='tight')
    
    plt.show()
    return ax


def plot_swarm_plot_efficiency(
    conditions_data_ch0, conditions_data_ch1, condition_labels, x_label="Condition",
    y_label="Efficiency (%)", title="", figsize=(6, 3), tick_size=16, swarm_color="black",
    y_lim=None, show_stats=False, only_significant=True, save_dir=None, plot_name='temp',
    max_percentile_significance=99.5, threshold_ch0=0.5, threshold_ch1=0.5, x_tick_rotation=0,
):
    """Calculate efficiency for each cell and create boxplot with swarm overlay."""
    sns.set_style("ticks")
    
    efficiency_values, condition_list = [], []
    for cond_idx, reps_ch0 in enumerate(conditions_data_ch0):
        reps_ch1 = conditions_data_ch1[cond_idx]
        for rep_idx in range(len(reps_ch0)):
            data_ch0 = np.asarray(reps_ch0[rep_idx]).flatten()
            data_ch1 = np.asarray(reps_ch1[rep_idx]).flatten()
            valid_mask = ~np.isnan(data_ch0) & ~np.isnan(data_ch1)
            data_ch0, data_ch1 = data_ch0[valid_mask], data_ch1[valid_mask]
            total_points = len(data_ch0)
            eff = np.sum((data_ch0 > threshold_ch0) & (data_ch1 > threshold_ch1)) / total_points * 100 if total_points > 0 else np.nan
            efficiency_values.append(eff)
            condition_list.append(condition_labels[cond_idx])
    
    df = pd.DataFrame({"Efficiency": efficiency_values, "Condition": condition_list})
    
    plt.figure(figsize=figsize, facecolor='white')
    ax = sns.boxplot(x="Condition", y="Efficiency", data=df, order=condition_labels, showfliers=False,
                     boxprops={'facecolor': 'white', 'edgecolor': 'black'},
                     medianprops={'color': 'red'}, whiskerprops={'color': 'black'},
                     capprops={'color': 'black'}, linewidth=1.5, whis=[5, 95], width=0.5)
    ax.set_facecolor('white')
    sns.swarmplot(x="Condition", y="Efficiency", data=df, order=condition_labels, color=swarm_color, size=5)
    
    plt.xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    plt.ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    plt.title(title, fontsize=tick_size + 4, fontname="Arial", color='black')
    if y_lim is not None and not show_stats:
        plt.ylim(y_lim)
    ax.tick_params(axis='x', labelsize=tick_size + 4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.xticks(fontname="Arial", rotation=x_tick_rotation, ha='right' if x_tick_rotation else 'center')
    plt.tight_layout()
    
    if show_stats and len(condition_labels) > 1:
        global_max = np.nanpercentile(df["Efficiency"], max_percentile_significance)
        global_range = (global_max - np.nanmin(df["Efficiency"])) or 1
        offset, bar_height, k = 0.1 * global_range, 0.02 * global_range, 0
        for i in range(len(condition_labels) - 1):
            for j in range(i + 1, len(condition_labels)):
                g1 = df[df["Condition"] == condition_labels[i]]["Efficiency"].dropna()
                g2 = df[df["Condition"] == condition_labels[j]]["Efficiency"].dropna()
                p = mannwhitneyu(g1, g2, alternative='two-sided')[1] if len(g1) and len(g2) else np.nan
                sig = '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                if only_significant and sig == 'ns':
                    continue
                y_line = global_max + offset * (k + 1)
                ax.plot([i, i, j, j], [y_line, y_line + bar_height, y_line + bar_height, y_line], lw=1.5, c='k')
                ax.text((i + j) * 0.5, y_line + bar_height, sig, ha='center', va='bottom', fontsize=tick_size)
                k += 1
    
    if save_dir is not None:
        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches='tight')
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches='tight')
    
    plt.show()
    return ax


def plot_efficiency_vs_intensity_scatter(
    data_dict, condition_labels, x_label="Mean Spot Intensity (Ch0)", y_label="CoF Efficiency (%)",
    title="", figsize=(8, 6), tick_size=12, save_dir=None, plot_name='efficiency_vs_intensity',
):
    """Create scatter plot comparing per-cell CoF efficiency vs mean spot intensity."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    colors = cm.tab10(np.linspace(0, 1, len(condition_labels)))
    all_intensities, all_efficiencies = [], []
    
    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict['int_ch_0'][cond_idx]
        eff_data = data_dict['efficiency_ml'][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if len(intensities) == 0 or cell_idx >= len(eff_data) or np.isnan(eff_data[cell_idx]):
                continue
            mean_int = np.nanmean(intensities)
            cell_int.append(mean_int)
            cell_eff.append(eff_data[cell_idx])
            all_intensities.append(mean_int)
            all_efficiencies.append(eff_data[cell_idx])
        ax.scatter(cell_int, cell_eff, c=[colors[cond_idx]], label=label, s=60, alpha=0.7, edgecolors='white')
    
    if len(all_intensities) > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(all_intensities, all_efficiencies)
        x_line = np.linspace(min(all_intensities), max(all_intensities), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, alpha=0.7)
        ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np = {p_value:.2e}', transform=ax.transAxes,
                fontsize=tick_size, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial")
    ax.legend(loc='upper right', fontsize=tick_size - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches='tight')
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches='tight')
    plt.show()
    return ax


def plot_efficiency_vs_intensity_scatter_means(
    data_dict, condition_labels, intensity_key='int_ch_0', x_label="Mean Spot Intensity (Ch0)", y_label="CoF Efficiency (%)",
    title="", figsize=(8, 6), tick_size=12, marker_size=150, show_individual_cells=False,
    individual_cell_alpha=0.3, individual_cell_size=20, colors=None, save_dir=None, plot_name='efficiency_vs_intensity_means',show_legend=False,
):
    """Create scatter plot of per-condition means with 2D error bars."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
    all_mean_int, all_mean_eff = [], []
    
    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict[intensity_key][cond_idx]
        eff_data = data_dict['efficiency_ml'][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if len(intensities) == 0 or cell_idx >= len(eff_data) or np.isnan(eff_data[cell_idx]):
                continue
            cell_int.append(np.nanmean(intensities))
            cell_eff.append(eff_data[cell_idx])
        if not cell_int:
            continue
        
        if show_individual_cells:
            ax.scatter(cell_int, cell_eff, c=colors[cond_idx % len(colors)], s=individual_cell_size, alpha=individual_cell_alpha, zorder=1)
        
        n = len(cell_int)
        mean_int, sem_int = np.mean(cell_int), np.std(cell_int) / np.sqrt(n)
        mean_eff, sem_eff = np.mean(cell_eff), np.std(cell_eff) / np.sqrt(n)
        all_mean_int.append(mean_int)
        all_mean_eff.append(mean_eff)
        
        ax.errorbar(mean_int, mean_eff, xerr=sem_int, yerr=sem_eff, fmt='o', color=colors[cond_idx % len(colors)],
                   markersize=np.sqrt(marker_size), capsize=5, capthick=2.5, elinewidth=2.5, label=label,
                   markeredgecolor='white', markeredgewidth=1, zorder=10)
    
    if len(all_mean_int) >= 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(all_mean_int, all_mean_eff)
        x_line = np.linspace(min(all_mean_int), max(all_mean_int), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, alpha=0.7)
        ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np = {p_value:.2e}', transform=ax.transAxes,
                fontsize=tick_size + 2, va='top', fontname="Arial",
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    ax.set_xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.set_ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.tick_params(labelsize=tick_size + 4)
    if show_legend:
        ax.legend(loc='lower right', fontsize=tick_size, frameon=True, facecolor='white', edgecolor='black')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout()
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches='tight')
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches='tight')
    plt.show()
    return ax




def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw an n_std confidence ellipse around the mean of x and y."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def plot_efficiency_vs_intensity_kde(
    data_dict, condition_labels, intensity_key='int_ch_0', x_label="Spot Intensity", y_label="CoF Efficiency (%)",
    title="", figsize=(8, 6), tick_size=12, marker_size=20, marker_alpha=0.5, kde_alpha =0.12,
    show_condition_means=False, mean_marker_size=100, show_marginals=False,
    show_kde=True, show_ellipse=False, ellipse_std=2.0,
    show_regression=False, x_lim =None, y_lim = None, colors=None,
    marginal_kws=None, save_dir=None, plot_name='efficiency_vs_intensity_hexbin',
):
    """Create joint scatter plot with marginal distributions, colored per condition.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys 'int_ch_0' and 'efficiency_ml'.
    condition_labels : list of str
        Labels for each condition.
    x_label, y_label : str
        Axis labels.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    tick_size : int
        Base tick label font size.
    marker_size : int
        Size of individual scatter dots.
    marker_alpha : float
        Transparency of scatter dots.
    show_condition_means : bool
        If True, overlay per-condition mean markers with error bars.
    mean_marker_size : int
        Marker size for condition means.
    show_marginals : bool
        If True, show marginal histograms on both axes.
    show_kde : bool
        If True, overlay filled KDE contours per condition.
    show_ellipse : bool
        If True, draw confidence ellipses around each condition.
    ellipse_std : float
        Number of standard deviations for the confidence ellipse (2.0 ≈ 95%).
    show_regression : bool
        If True, show linear regression line with R² and p-value.
    marginal_kws : dict or None
        Extra keyword arguments for marginal histograms.
    save_dir : str or None
        Directory to save the figure (PNG + SVG at 600 dpi).
    plot_name : str
        Base filename for saved figures.

    Returns
    -------
    g : seaborn.JointGrid
        The JointGrid object for further customization.
    """
    sns.set_style("ticks")
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['text.color'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']

    # --- Collect per-cell data, separated by condition ---
    condition_data = []
    condition_means = []

    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict[intensity_key][cond_idx]
        eff_data = data_dict['efficiency_ml'][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if len(intensities) == 0 or cell_idx >= len(eff_data) or np.isnan(eff_data[cell_idx]):
                continue
            cell_int.append(np.nanmean(intensities))
            cell_eff.append(eff_data[cell_idx])
        if not cell_int:
            continue

        color = colors[cond_idx % len(colors)]
        marker = markers[cond_idx % len(markers)]
        condition_data.append((np.array(cell_int), np.array(cell_eff), label, color, marker))

        n = len(cell_int)
        condition_means.append((
            np.mean(cell_int), np.std(cell_int) / np.sqrt(n),
            np.mean(cell_eff), np.std(cell_eff) / np.sqrt(n),
            label, color,
        ))

    if not condition_data:
        print("No valid data to plot.")
        return None

    # --- Build JointGrid manually ---
    g = sns.JointGrid(height=figsize[1], ratio=5)
    ax = g.ax_joint

    # --- KDE contours per condition ---
    if show_kde:
        for x_arr, y_arr, label, color, marker in condition_data:
            if len(x_arr) >= 3:
                sns.kdeplot(x=x_arr, y=y_arr, ax=ax, color=color, fill=True,
                            levels=3, alpha=0.2, linewidths=0.5)

    # --- Confidence ellipses ---
    if show_ellipse:
        for x_arr, y_arr, label, color, marker in condition_data:
            confidence_ellipse(x_arr, y_arr, ax, n_std=ellipse_std,
                               facecolor=color, alpha=kde_alpha, edgecolor=color, linewidth=2)

    # --- Scatter per condition ---
    for x_arr, y_arr, label, color, marker in condition_data:
        ax.scatter(x_arr, y_arr, c=color, s=marker_size, alpha=marker_alpha,
                   marker=marker, label=label, edgecolors='white', linewidths=0.3, zorder=2)

    # --- Marginal histograms ---
    if show_marginals:
        _marginal_kws = dict(bins=30, alpha=0.5, edgecolor='white', linewidth=0.5)
        if marginal_kws is not None:
            _marginal_kws.update(marginal_kws)
        for x_arr, y_arr, label, color, marker in condition_data:
            g.ax_marg_x.hist(x_arr, color=color, **_marginal_kws)
            g.ax_marg_y.hist(y_arr, color=color, orientation='horizontal', **_marginal_kws)
    else:
        g.ax_marg_x.set_visible(False)
        g.ax_marg_y.set_visible(False)

    # --- Condition means with error bars ---
    if show_condition_means and condition_means:
        for mean_int, sem_int, mean_eff, sem_eff, label, color in condition_means:
            ax.errorbar(
                mean_int, mean_eff, xerr=sem_int, yerr=sem_eff, fmt='o',
                color=color, markersize=np.sqrt(mean_marker_size), capsize=5,
                capthick=2.5, elinewidth=2.5,
                markeredgecolor='white', markeredgewidth=1, zorder=10,
            )

    # --- Linear regression across condition means ---
    if show_regression:
        if len(condition_means) >= 2:
            all_mean_int = [m[0] for m in condition_means]
            all_mean_eff = [m[2] for m in condition_means]
            slope, intercept, r_value, p_value, _ = stats.linregress(all_mean_int, all_mean_eff)
            x_line = np.linspace(min(all_mean_int), max(all_mean_int), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, alpha=0.7)
            ax.text(
                0.05, 0.95, f'R² = {r_value**2:.3f}\np = {p_value:.2e}',
                transform=ax.transAxes, fontsize=tick_size + 2, va='top',
                fontname="Arial",
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'),
            )

    # --- Styling ---
    ax.set_xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.set_ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color='black')
    ax.tick_params(labelsize=tick_size + 4)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(condition_data),
        fontsize=tick_size,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        prop={'family': 'Arial'}, 
        labelcolor='black', 
    )
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title:
        g.figure.suptitle(title, fontsize=tick_size + 4, fontname="Arial", color='black', y=1.02)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    g.figure.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        g.figure.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches='tight')
        g.figure.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches='tight')
    plt.show()
    return g




__all__ = [
    'plot_swarm_plot',
    'plot_swarm_plot_efficiency', 
    'plot_efficiency_vs_intensity_scatter',
    'plot_efficiency_vs_intensity_scatter_means',
    'plot_efficiency_vs_intensity_kde',
]
