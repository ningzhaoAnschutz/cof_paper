"""
Plotting utilities for COF analysis.

This module provides visualization functions for:
- Swarm plots with statistical comparisons
- Efficiency plots
- Scatter plots comparing efficiency vs intensity
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import mannwhitneyu


def plot_swarm_plot(
    conditions_data,
    condition_labels,
    x_label="",
    y_label=None,
    title="",
    figsize=(6, 4),
    tick_size=14,
    swarm_color="black",
    y_lim=None,
    show_stats=False,
    only_significant=True,
    save_dir=None,
    plot_name="temp",
    max_percentile_significance=99.5,
    x_tick_rotation=0,
    show_n=True,
    swarm_size=6,
    min_spots_threshold=None,
    cell_summary="median",
    show_n_trajectories=True,
    trajectory_counts=None,
):
    """
    Create a boxplot with swarm overlay and optional statistical comparisons.

    Each element in conditions_data should be an iterable of NumPy arrays,
    where each array represents one cell/repetition of a given condition.
    For each cell, a single summary statistic (mean or median) is computed
    from its array of spot measurements and plotted as one dot in the swarm.

    Parameters
    ----------
    conditions_data : list of list of array-like
        Outer list: one element per condition. Inner list: one NumPy array
        per cell, containing all spot-level measurements for that cell.
    condition_labels : list of str
        Display labels for each condition (same length as conditions_data).
    x_label : str, optional
        Label for the x-axis. Default is "".
    y_label : str or None, optional
        Label for the y-axis. If None (default), automatically set to
        "Median Value" or "Mean Value" based on ``cell_summary``.
    title : str, optional
        Plot title. Default is "".
    figsize : tuple, optional
        Figure size as (width, height). Default is (6, 4).
    tick_size : int, optional
        Base font size for tick labels. Default is 14.
    swarm_color : str, optional
        Color of the swarm dots. Default is "black".
    y_lim : tuple or None, optional
        Y-axis limits as (ymin, ymax). Default is None (auto).
    show_stats : bool, optional
        If True, show pairwise Mann-Whitney U significance brackets.
        Default is False.
    only_significant : bool, optional
        If True and show_stats is True, only display significant pairs
        (p < 0.05). Default is True.
    save_dir : str or Path or None, optional
        Directory to save PNG and SVG files. Default is None (no saving).
    plot_name : str, optional
        Base filename for saved figures. Default is 'temp'.
    max_percentile_significance : float, optional
        Percentile used to set the top of the significance bracket range.
        Default is 99.5.
    x_tick_rotation : int, optional
        Rotation angle for x-axis tick labels. Default is 0.
    show_n : bool, optional
        If True, append "(n=...)" to each x-axis label. Default is True.
    swarm_size : int, optional
        Size of swarm dots. Default is 6.
    min_spots_threshold : int or None, optional
        Minimum number of valid spots required for a cell to be included.
        Cells below this threshold are excluded. Default is None (no filter).
    cell_summary : str, optional
        Summary statistic to compute per cell.  Must be ``'mean'`` or
        ``'median'``.  Each dot in the swarm plot represents this statistic
        computed over all spots detected in that cell.  Default is
        ``'median'``.

        Example – switch to mean::

            ld.plot_swarm_plot(
                data_dict['int_ch_0'], list_names,
                cell_summary='mean',       # each dot = np.nanmean per cell
                y_label='Mean Intensity',   # update axis label to match
                ...
            )

    show_n_trajectories : bool, optional
        If True, display the total number of trajectories alongside the
        cell count on the x-axis labels (e.g. ``n=12 cells, 43 traj``).
        Default is True.
    trajectory_counts : list of list of int, or None, optional
        Per-cell trajectory counts, structured identically to
        ``conditions_data``.  When provided, these counts are used
        instead of ``len(rep)`` to compute trajectory totals.  Pass
        ``data_dict['n_trajectories']`` for plots that use all-trajectory
        data (e.g. ``'int_ch_0'``).  For colocalization-filtered data
        (e.g. ``'int_ch_0_coloc'``), omit this parameter — the array
        lengths already equal the trajectory count.  Default is None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object for further customization.
    """
    if cell_summary not in ("mean", "median"):
        raise ValueError(
            f"cell_summary must be 'mean' or 'median', got '{cell_summary}'"
        )
    if y_label is None:
        y_label = "Median Value" if cell_summary == "median" else "Mean Value"
    _agg_func = np.nanmean if cell_summary == "mean" else np.nanmedian

    sns.set_style("ticks")

    summary_values, condition_list = [], []
    cell_counts = {label: 0 for label in condition_labels}
    cells_excluded = {label: 0 for label in condition_labels}
    traj_counts = {label: 0 for label in condition_labels}

    for cond_idx, repetitions in enumerate(conditions_data):
        if repetitions is None:
            continue
        for rep_idx, rep in enumerate(repetitions):
            if rep is None:
                continue
            valid_rep = np.asarray(rep).flatten()
            valid_spots = np.sum(~np.isnan(valid_rep))
            if valid_spots == 0:
                continue
            if min_spots_threshold is not None and valid_spots < min_spots_threshold:
                cells_excluded[condition_labels[cond_idx]] += 1
                continue
            rep_value = _agg_func(rep)
            summary_values.append(rep_value)
            condition_list.append(condition_labels[cond_idx])
            cell_counts[condition_labels[cond_idx]] += 1
            if show_n_trajectories:
                label = condition_labels[cond_idx]
                if (
                    trajectory_counts is not None
                    and trajectory_counts[cond_idx] is not None
                ):
                    traj_counts[label] += int(
                        trajectory_counts[cond_idx][rep_idx]
                    )
                else:
                    traj_counts[label] += int(valid_spots)

    if min_spots_threshold is not None:
        print(f"\n=== Cell Filtering (min_spots_threshold={min_spots_threshold}) ===")
        for label in condition_labels:
            total_cells = cell_counts[label] + cells_excluded[label]
            print(f"  {label}: {cell_counts[label]}/{total_cells} cells kept")

    df = pd.DataFrame({"Value": summary_values, "Condition": condition_list})
    valid_labels = [label for label in condition_labels if cell_counts[label] > 0]
    if not valid_labels:
        print("⚠️  No valid data found in any condition — skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    sns.boxplot(
        x="Condition",
        y="Value",
        data=df,
        order=valid_labels,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "red"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        linewidth=1.5,
        whis=[5, 95],
        width=0.5,
        ax=ax,
    )
    sns.swarmplot(
        x="Condition",
        y="Value",
        data=df,
        order=valid_labels,
        color=swarm_color,
        size=swarm_size,
        ax=ax,
    )

    if show_n:
        # Auto-detect cell-level data: if traj count == cell count for all
        # conditions, the data is already one scalar per cell (e.g. efficiency,
        # average_number_spots) and showing "traj" is meaningless.
        _traj_is_meaningful = show_n_trajectories and any(
            traj_counts[label] != cell_counts[label] for label in valid_labels
        )
        # Build multi-line labels: name on line 1, counts on lines below
        ax.set_xticklabels([""] * len(valid_labels))  # clear default labels
        for idx, label in enumerate(valid_labels):
            # Line 1: condition name
            ax.text(
                idx, -0.02, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=tick_size, fontname="Arial",
                color="black",
            )
            # Line 2: cell count
            ax.text(
                idx, -0.07, f"cells={cell_counts[label]}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=tick_size, fontname="Arial",
                color="black",
            )
            # Line 3: trajectory count (only if meaningful)
            if _traj_is_meaningful:
                ax.text(
                    idx, -0.12, f"traj={traj_counts[label]}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=tick_size, fontname="Arial",
                    color="black",
                )
    else:
        ax.set_xticklabels(
            valid_labels,
            fontname="Arial",
            fontsize=tick_size,
            rotation=x_tick_rotation,
            ha="right" if x_tick_rotation else "center",
        )

    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial", color="black")
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial", color="black")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color="black")
    if y_lim is not None and not show_stats:
        ax.set_ylim(y_lim)
    ax.tick_params(axis="y", labelsize=tick_size, colors="black")
    plt.tight_layout()

    if show_stats and len(valid_labels) > 1:
        global_max = np.nanpercentile(df["Value"], max_percentile_significance)
        global_min = np.nanmin(df["Value"])
        global_range = global_max - global_min if (global_max - global_min) != 0 else 1
        offset = 0.10 * global_range
        bar_height = 0.02 * global_range
        k = 0
        for i in range(len(valid_labels) - 1):
            for j in range(i + 1, len(valid_labels)):
                g1 = df[df["Condition"] == valid_labels[i]]["Value"].dropna()
                g2 = df[df["Condition"] == valid_labels[j]]["Value"].dropna()
                p = (
                    mannwhitneyu(g1, g2, alternative="two-sided")[1]
                    if len(g1) > 0 and len(g2) > 0
                    else np.nan
                )
                sig = (
                    "****"
                    if p < 0.0001
                    else (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                )
                if only_significant and sig == "ns":
                    continue
                y_line = global_max + offset * (k + 1)
                ax.plot(
                    [i, i, j, j],
                    [y_line, y_line + bar_height, y_line + bar_height, y_line],
                    lw=1.2,
                    c="k",
                )
                ax.text(
                    (i + j) * 0.5,
                    y_line + bar_height - 0.01 * global_range,
                    sig,
                    ha="center",
                    va="bottom",
                    color="k",
                    fontsize=tick_size - 2,
                    fontname="Arial",
                )
                k += 1

    if save_dir is not None:
        save_dir = Path(save_dir)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")

    plt.show()
    return ax


def plot_swarm_plot_grouped(
    data_sources,
    condition_labels,
    x_label="",
    y_label=None,
    title="",
    figsize=(8, 4),
    tick_size=14,
    y_lim=None,
    show_stats=False,
    only_significant=True,
    save_dir=None,
    plot_name="temp_grouped",
    max_percentile_significance=99.5,
    x_tick_rotation=0,
    show_n=True,
    swarm_size=5,
    min_spots_threshold=None,
    cell_summary="median",
    show_n_trajectories=True,
    group_colors=None,
    **kwargs,
):
    """
    Create a grouped boxplot with swarm overlay comparing sub-populations.

    Each condition shows side-by-side sub-groups (e.g. colocalized vs
    non-colocalized), distinguished by color.

    Parameters
    ----------
    data_sources : list of (data, label) tuples
        Each tuple contains ``(conditions_data, group_label)``.
        ``conditions_data`` has the same structure as in
        :func:`plot_swarm_plot` (list of list of array-like, one per
        condition, one array per cell).

        Example::

            data_sources=[
                (data_dict['int_ch_0_coloc'],     'Coloc'),
                (data_dict['int_ch_0_not_coloc'], 'Not Coloc'),
            ]

    condition_labels : list of str
        Display labels for each condition.
    x_label : str, optional
        Label for the x-axis. Default is ``""``.
    y_label : str or None, optional
        Label for the y-axis. If None, auto-set from ``cell_summary``.
    title : str, optional
        Plot title. Default is ``""``.
    figsize : tuple, optional
        Figure size. Default is ``(8, 4)``.
    tick_size : int, optional
        Base font size. Default is 14.
    y_lim : tuple or None, optional
        Y-axis limits. Default is None.
    show_stats : bool, optional
        If True, show within-condition pairwise significance between
        groups.  Default is False.
    only_significant : bool, optional
        If True and show_stats is True, only show significant pairs.
        Default is True.
    save_dir : str or Path or None, optional
        Directory to save PNG and SVG. Default is None.
    plot_name : str, optional
        Base filename for saved figures. Default is ``'temp_grouped'``.
    max_percentile_significance : float, optional
        Percentile for significance bracket placement. Default is 99.5.
    x_tick_rotation : int, optional
        Rotation of x-axis tick labels. Default is 0.
    show_n : bool, optional
        If True, show sample sizes below condition labels. Default is True.
    swarm_size : int, optional
        Size of swarm dots. Default is 5.
    min_spots_threshold : int or None, optional
        Minimum valid spots for a cell to be included. Default is None.
    cell_summary : str, optional
        ``'mean'`` or ``'median'``. Default is ``'median'``.
    show_n_trajectories : bool, optional
        If True and trajectory counts are meaningful, show them below
        condition labels. Default is True.
    group_colors : list of str or None, optional
        Colors for each group. Default uses a curated palette.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object for further customization.
    """
    if cell_summary not in ("mean", "median"):
        raise ValueError(
            f"cell_summary must be 'mean' or 'median', got '{cell_summary}'"
        )
    if y_label is None:
        y_label = "Median Value" if cell_summary == "median" else "Mean Value"
    _agg_func = np.nanmean if cell_summary == "mean" else np.nanmedian

    if group_colors is None:
        group_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    sns.set_style("ticks")
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"

    group_labels = [gs[1] for gs in data_sources]
    palette = {
        gl: group_colors[i % len(group_colors)]
        for i, gl in enumerate(group_labels)
    }

    # Build DataFrame with Condition + Group columns
    rows = []
    # Track per-condition, per-group counts
    cell_counts = {
        label: {gl: 0 for gl in group_labels} for label in condition_labels
    }
    traj_counts = {
        label: {gl: 0 for gl in group_labels} for label in condition_labels
    }

    for group_idx, (group_data, group_label) in enumerate(data_sources):
        for cond_idx, repetitions in enumerate(group_data):
            if repetitions is None:
                continue
            cond_label = condition_labels[cond_idx]
            for rep in repetitions:
                if rep is None:
                    continue
                valid_rep = np.asarray(rep).flatten()
                valid_spots = np.sum(~np.isnan(valid_rep))
                if (
                    min_spots_threshold is not None
                    and valid_spots < min_spots_threshold
                ):
                    continue
                rows.append(
                    {
                        "Value": _agg_func(rep),
                        "Condition": cond_label,
                        "Group": group_label,
                    }
                )
                cell_counts[cond_label][group_label] += 1
                if show_n_trajectories:
                    traj_counts[cond_label][group_label] += int(valid_spots)

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️  No valid data found — skipping grouped plot.")
        return None

    valid_labels = [
        label
        for label in condition_labels
        if any(cell_counts[label][gl] > 0 for gl in group_labels)
    ]
    if not valid_labels:
        print("⚠️  No valid conditions — skipping grouped plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    sns.boxplot(
        x="Condition",
        y="Value",
        hue="Group",
        data=df,
        order=valid_labels,
        hue_order=group_labels,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "red"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        linewidth=1.5,
        whis=[5, 95],
        width=0.6,
        ax=ax,
    )
    sns.swarmplot(
        x="Condition",
        y="Value",
        hue="Group",
        data=df,
        order=valid_labels,
        hue_order=group_labels,
        palette=palette,
        dodge=True,
        size=swarm_size,
        ax=ax,
    )

    # Fix legend: use swarm handles (colored dots) not box handles (white)
    handles, labels_legend = ax.get_legend_handles_labels()
    n_groups = len(group_labels)
    legend = ax.legend(
        handles[n_groups: 2 * n_groups],
        labels_legend[n_groups: 2 * n_groups],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=n_groups,
        fontsize=tick_size - 2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"family": "Arial", "size": tick_size - 2},
    )
    for text in legend.get_texts():
        text.set_color("black")

    # X-axis labels with counts
    if show_n:
        ax.set_xticklabels([""] * len(valid_labels))
        _traj_is_meaningful = show_n_trajectories and any(
            traj_counts[label][gl] != cell_counts[label][gl]
            for label in valid_labels
            for gl in group_labels
        )
        for idx, label in enumerate(valid_labels):
            # Line 1: condition name
            ax.text(
                idx, -0.02, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=tick_size, fontname="Arial",
                color="black",
            )
            # Line 2: cell counts per group (e.g. "cells=12 | 10")
            cell_str = " | ".join(
                str(cell_counts[label][gl]) for gl in group_labels
            )
            ax.text(
                idx, -0.07, f"cells={cell_str}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=tick_size, fontname="Arial",
                color="black",
            )
            # Line 3: trajectory counts per group
            if _traj_is_meaningful:
                traj_str = " | ".join(
                    str(traj_counts[label][gl]) for gl in group_labels
                )
                ax.text(
                    idx, -0.12, f"traj={traj_str}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=tick_size, fontname="Arial",
                    color="black",
                )
    else:
        ax.set_xticklabels(
            valid_labels,
            fontname="Arial",
            fontsize=tick_size,
            rotation=x_tick_rotation,
            ha="right" if x_tick_rotation else "center",
        )

    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial", color="black")
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial", color="black")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color="black")
    if y_lim is not None and not show_stats:
        ax.set_ylim(y_lim)
    ax.tick_params(axis="y", labelsize=tick_size, colors="black")
    plt.tight_layout()

    # Within-condition pairwise significance between groups
    if show_stats and len(group_labels) > 1:
        global_max = np.nanpercentile(df["Value"], max_percentile_significance)
        global_min = np.nanmin(df["Value"])
        global_range = (global_max - global_min) if (global_max - global_min) != 0 else 1
        offset = 0.10 * global_range
        bar_height = 0.02 * global_range

        for cond_idx, cond_label in enumerate(valid_labels):
            k = 0
            for gi in range(len(group_labels) - 1):
                for gj in range(gi + 1, len(group_labels)):
                    g1 = df[
                        (df["Condition"] == cond_label)
                        & (df["Group"] == group_labels[gi])
                    ]["Value"].dropna()
                    g2 = df[
                        (df["Condition"] == cond_label)
                        & (df["Group"] == group_labels[gj])
                    ]["Value"].dropna()
                    p = (
                        mannwhitneyu(g1, g2, alternative="two-sided")[1]
                        if len(g1) > 0 and len(g2) > 0
                        else np.nan
                    )
                    sig = (
                        "****"
                        if p < 0.0001
                        else (
                            "***"
                            if p < 0.001
                            else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        )
                    )
                    if only_significant and sig == "ns":
                        continue
                    # Position brackets over the sub-groups within this condition
                    n_groups_here = len(group_labels)
                    width = 0.6  # boxplot width
                    group_width = width / n_groups_here
                    x_left = cond_idx - width / 2 + group_width * (gi + 0.5)
                    x_right = cond_idx - width / 2 + group_width * (gj + 0.5)
                    y_line = global_max + offset * (k + 1)
                    ax.plot(
                        [x_left, x_left, x_right, x_right],
                        [y_line, y_line + bar_height, y_line + bar_height, y_line],
                        lw=1.2,
                        c="k",
                    )
                    ax.text(
                        (x_left + x_right) * 0.5,
                        y_line + bar_height - 0.01 * global_range,
                        sig,
                        ha="center",
                        va="bottom",
                        color="k",
                        fontsize=tick_size - 2,
                        fontname="Arial",
                    )
                    k += 1

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")

    plt.show()
    return ax


def plot_swarm_plot_efficiency(
    conditions_data_ch0,
    conditions_data_ch1,
    condition_labels,
    x_label="Condition",
    y_label="Efficiency (%)",
    title="",
    figsize=(6, 3),
    tick_size=16,
    swarm_color="black",
    y_lim=None,
    show_stats=False,
    only_significant=True,
    save_dir=None,
    plot_name="temp",
    max_percentile_significance=99.5,
    threshold_ch0=0.5,
    threshold_ch1=0.5,
    x_tick_rotation=0,
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
            eff = (
                np.sum((data_ch0 > threshold_ch0) & (data_ch1 > threshold_ch1))
                / total_points
                * 100
                if total_points > 0
                else np.nan
            )
            efficiency_values.append(eff)
            condition_list.append(condition_labels[cond_idx])

    df = pd.DataFrame({"Efficiency": efficiency_values, "Condition": condition_list})

    plt.figure(figsize=figsize, facecolor="white")
    ax = sns.boxplot(
        x="Condition",
        y="Efficiency",
        data=df,
        order=condition_labels,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "red"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        linewidth=1.5,
        whis=[5, 95],
        width=0.5,
    )
    ax.set_facecolor("white")
    sns.swarmplot(
        x="Condition",
        y="Efficiency",
        data=df,
        order=condition_labels,
        color=swarm_color,
        size=5,
    )

    plt.xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    plt.ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    plt.title(title, fontsize=tick_size + 4, fontname="Arial", color="black")
    if y_lim is not None and not show_stats:
        plt.ylim(y_lim)
    ax.tick_params(axis="x", labelsize=tick_size + 4, colors="black")
    ax.tick_params(axis="y", labelsize=tick_size, colors="black")
    plt.xticks(
        fontname="Arial",
        rotation=x_tick_rotation,
        ha="right" if x_tick_rotation else "center",
    )
    plt.tight_layout()

    if show_stats and len(condition_labels) > 1:
        global_max = np.nanpercentile(df["Efficiency"], max_percentile_significance)
        global_range = (global_max - np.nanmin(df["Efficiency"])) or 1
        offset, bar_height, k = 0.1 * global_range, 0.02 * global_range, 0
        for i in range(len(condition_labels) - 1):
            for j in range(i + 1, len(condition_labels)):
                g1 = df[df["Condition"] == condition_labels[i]]["Efficiency"].dropna()
                g2 = df[df["Condition"] == condition_labels[j]]["Efficiency"].dropna()
                p = (
                    mannwhitneyu(g1, g2, alternative="two-sided")[1]
                    if len(g1) and len(g2)
                    else np.nan
                )
                sig = (
                    "****"
                    if p < 0.0001
                    else (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                )
                if only_significant and sig == "ns":
                    continue
                y_line = global_max + offset * (k + 1)
                ax.plot(
                    [i, i, j, j],
                    [y_line, y_line + bar_height, y_line + bar_height, y_line],
                    lw=1.5,
                    c="k",
                )
                ax.text(
                    (i + j) * 0.5,
                    y_line + bar_height,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=tick_size,
                )
                k += 1

    if save_dir is not None:
        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")

    plt.show()
    return ax


def plot_efficiency_vs_intensity_scatter(
    data_dict,
    condition_labels,
    x_label="Mean Spot Intensity (Ch0)",
    y_label="CoF Efficiency (%)",
    title="",
    figsize=(8, 6),
    tick_size=12,
    save_dir=None,
    plot_name="efficiency_vs_intensity",
):
    """Create scatter plot comparing per-cell CoF efficiency vs mean spot intensity."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    colors = cm.tab10(np.linspace(0, 1, len(condition_labels)))
    all_intensities, all_efficiencies = [], []

    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict["int_ch_0"][cond_idx]
        eff_data = data_dict["efficiency_ml"][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if (
                len(intensities) == 0
                or cell_idx >= len(eff_data)
                or np.isnan(eff_data[cell_idx])
            ):
                continue
            mean_int = np.nanmean(intensities)
            cell_int.append(mean_int)
            cell_eff.append(eff_data[cell_idx])
            all_intensities.append(mean_int)
            all_efficiencies.append(eff_data[cell_idx])
        ax.scatter(
            cell_int,
            cell_eff,
            c=[colors[cond_idx]],
            label=label,
            s=60,
            alpha=0.7,
            edgecolors="white",
        )

    if len(all_intensities) > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            all_intensities, all_efficiencies
        )
        x_line = np.linspace(min(all_intensities), max(all_intensities), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, alpha=0.7)
        ax.text(
            0.05,
            0.95,
            f"R² = {r_value**2:.3f}\np = {p_value:.2e}",
            transform=ax.transAxes,
            fontsize=tick_size,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(x_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_ylabel(y_label, fontsize=tick_size + 2, fontname="Arial")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial")
    ax.legend(loc="upper right", fontsize=tick_size - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")
    plt.show()
    return ax


def plot_efficiency_vs_intensity_scatter_means(
    data_dict,
    condition_labels,
    intensity_key="int_ch_0",
    x_label="Mean Spot Intensity (Ch0)",
    y_label="CoF Efficiency (%)",
    title="",
    figsize=(8, 6),
    tick_size=12,
    marker_size=150,
    show_individual_cells=False,
    show_regression_line=True,
    x_lim=None,
    y_lim=None,
    individual_cell_alpha=0.3,
    individual_cell_size=20,
    colors=None,
    save_dir=None,
    plot_name="efficiency_vs_intensity_means",
    show_legend=False,
):
    """Create scatter plot of per-condition means with 2D error bars."""
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
            "#bcbd22",
            "#7f7f7f",
        ]
    all_mean_int, all_mean_eff = [], []

    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict[intensity_key][cond_idx]
        eff_data = data_dict["efficiency_ml"][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if (
                len(intensities) == 0
                or cell_idx >= len(eff_data)
                or np.isnan(eff_data[cell_idx])
            ):
                continue
            cell_int.append(np.nanmean(intensities))
            cell_eff.append(eff_data[cell_idx])
        if not cell_int:
            continue

        if show_individual_cells:
            ax.scatter(
                cell_int,
                cell_eff,
                c=colors[cond_idx % len(colors)],
                s=individual_cell_size,
                alpha=individual_cell_alpha,
                zorder=1,
            )

        n = len(cell_int)
        mean_int, sem_int = np.mean(cell_int), np.std(cell_int) / np.sqrt(n)
        mean_eff, sem_eff = np.mean(cell_eff), np.std(cell_eff) / np.sqrt(n)
        all_mean_int.append(mean_int)
        all_mean_eff.append(mean_eff)

        ax.errorbar(
            mean_int,
            mean_eff,
            xerr=sem_int,
            yerr=sem_eff,
            fmt="o",
            color=colors[cond_idx % len(colors)],
            markersize=np.sqrt(marker_size),
            capsize=5,
            capthick=2.5,
            elinewidth=2.5,
            label=label,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=10,
        )

    if show_regression_line:
        if len(all_mean_int) >= 2:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                all_mean_int, all_mean_eff
            )
            x_line = np.linspace(min(all_mean_int), max(all_mean_int), 100)
            ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, alpha=0.7)
            ax.text(
                0.05,
                0.95,
                f"R² = {r_value**2:.3f}\np = {p_value:.2e}",
                transform=ax.transAxes,
                fontsize=tick_size + 2,
                va="top",
                fontname="Arial",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"
                ),
            )

    ax.set_xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    ax.set_ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    ax.set_title(title, fontsize=tick_size + 4, fontname="Arial", color="black")
    ax.tick_params(labelsize=tick_size + 4)
    # define x and y limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if show_legend:
        ax.legend(
            loc="lower right",
            fontsize=tick_size,
            frameon=True,
            facecolor="white",
            edgecolor="black",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        plt.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")
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
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)


def plot_efficiency_vs_intensity_kde(
    data_dict,
    condition_labels,
    intensity_key="int_ch_0",
    x_label="Spot Intensity",
    y_label="CoF Efficiency (%)",
    title="",
    figsize=(8, 6),
    tick_size=12,
    marker_size=20,
    marker_alpha=0.5,
    kde_alpha=0.12,
    show_condition_means=False,
    mean_marker_size=100,
    show_marginals=False,
    show_kde=True,
    show_ellipse=False,
    ellipse_std=2.0,
    show_regression=False,
    x_lim=None,
    y_lim=None,
    colors=None,
    marginal_kws=None,
    save_dir=None,
    plot_name="efficiency_vs_intensity_hexbin",
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
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"

    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
            "#bcbd22",
            "#7f7f7f",
        ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

    # --- Collect per-cell data, separated by condition ---
    condition_data = []
    condition_means = []

    for cond_idx, label in enumerate(condition_labels):
        int_data = data_dict[intensity_key][cond_idx]
        eff_data = data_dict["efficiency_ml"][cond_idx]
        if int_data is None or eff_data is None:
            continue
        cell_int, cell_eff = [], []
        for cell_idx in range(len(int_data)):
            intensities = np.asarray(int_data[cell_idx]).flatten()
            intensities = intensities[~np.isnan(intensities)]
            if (
                len(intensities) == 0
                or cell_idx >= len(eff_data)
                or np.isnan(eff_data[cell_idx])
            ):
                continue
            cell_int.append(np.nanmean(intensities))
            cell_eff.append(eff_data[cell_idx])
        if not cell_int:
            continue

        color = colors[cond_idx % len(colors)]
        marker = markers[cond_idx % len(markers)]
        condition_data.append(
            (np.array(cell_int), np.array(cell_eff), label, color, marker)
        )

        n = len(cell_int)
        condition_means.append(
            (
                np.mean(cell_int),
                np.std(cell_int) / np.sqrt(n),
                np.mean(cell_eff),
                np.std(cell_eff) / np.sqrt(n),
                label,
                color,
            )
        )

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
                sns.kdeplot(
                    x=x_arr,
                    y=y_arr,
                    ax=ax,
                    color=color,
                    fill=True,
                    levels=3,
                    alpha=0.2,
                    linewidths=0.5,
                )

    # --- Confidence ellipses ---
    if show_ellipse:
        for x_arr, y_arr, label, color, marker in condition_data:
            confidence_ellipse(
                x_arr,
                y_arr,
                ax,
                n_std=ellipse_std,
                facecolor=color,
                alpha=kde_alpha,
                edgecolor=color,
                linewidth=2,
            )

    # --- Scatter per condition ---
    for x_arr, y_arr, label, color, marker in condition_data:
        ax.scatter(
            x_arr,
            y_arr,
            c=color,
            s=marker_size,
            alpha=marker_alpha,
            marker=marker,
            label=label,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

    # --- Marginal histograms ---
    if show_marginals:
        _marginal_kws = dict(bins=30, alpha=0.5, edgecolor="white", linewidth=0.5)
        if marginal_kws is not None:
            _marginal_kws.update(marginal_kws)
        for x_arr, y_arr, label, color, marker in condition_data:
            g.ax_marg_x.hist(x_arr, color=color, **_marginal_kws)
            g.ax_marg_y.hist(
                y_arr, color=color, orientation="horizontal", **_marginal_kws
            )
    else:
        g.ax_marg_x.set_visible(False)
        g.ax_marg_y.set_visible(False)

    # --- Condition means with error bars ---
    if show_condition_means and condition_means:
        for mean_int, sem_int, mean_eff, sem_eff, label, color in condition_means:
            ax.errorbar(
                mean_int,
                mean_eff,
                xerr=sem_int,
                yerr=sem_eff,
                fmt="o",
                color=color,
                markersize=np.sqrt(mean_marker_size),
                capsize=5,
                capthick=2.5,
                elinewidth=2.5,
                markeredgecolor="white",
                markeredgewidth=1,
                zorder=10,
            )

    # --- Linear regression across condition means ---
    if show_regression:
        if len(condition_means) >= 2:
            all_mean_int = [m[0] for m in condition_means]
            all_mean_eff = [m[2] for m in condition_means]
            slope, intercept, r_value, p_value, _ = stats.linregress(
                all_mean_int, all_mean_eff
            )
            x_line = np.linspace(min(all_mean_int), max(all_mean_int), 100)
            ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, alpha=0.7)
            ax.text(
                0.05,
                0.95,
                f"R² = {r_value**2:.3f}\np = {p_value:.2e}",
                transform=ax.transAxes,
                fontsize=tick_size + 2,
                va="top",
                fontname="Arial",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"
                ),
            )

    # --- Styling ---
    ax.set_xlabel(x_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    ax.set_ylabel(y_label, fontsize=tick_size + 4, fontname="Arial", color="black")
    ax.tick_params(labelsize=tick_size + 4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(condition_data),
        fontsize=tick_size,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"family": "Arial"},
        labelcolor="black",
    )
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title:
        g.figure.suptitle(
            title, fontsize=tick_size + 4, fontname="Arial", color="black", y=1.02
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    g.figure.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        g.figure.savefig(save_dir / f"{plot_name}.png", dpi=600, bbox_inches="tight")
        g.figure.savefig(save_dir / f"{plot_name}.svg", dpi=600, bbox_inches="tight")
    plt.show()
    return g


__all__ = [
    "plot_swarm_plot",
    "plot_swarm_plot_grouped",
    "plot_swarm_plot_efficiency",
    "plot_efficiency_vs_intensity_scatter",
    "plot_efficiency_vs_intensity_scatter_means",
    "plot_efficiency_vs_intensity_kde",
]
