"""Shared plotting style for CoF time-course figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import mannwhitneyu


FONT_FAMILY = "Arial"
DEFAULT_DPI = 300

# Match the original time-course notebook / ImageJ channel convention.
CHANNEL_GREEN = (0.0, 1.0, 0.0)
CHANNEL_MAGENTA = (1.0, 0.0, 1.0)
CHANNEL_BLACK = (0.0, 0.0, 0.0)


TRACE_MAGENTA = "#d600d6"
TRACE_GREEN = "#00a651"
TRACE_BLUE = "#2b7bba"
TRACE_GRAY = "#4d4d4d"

def set_publication_style() -> None:
    """Use the white/Arial/black-axis style from the reference notebooks."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.5,
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(
    ax,
    *,
    grid: bool = False,
    spine_width: float = 1.5,
    tick_size: int = 12,
    label_size: int = 14,
    title_size: int = 14,
    boxed: bool = True,
) -> None:
    """Apply clean white axes with black labels and optional boxed spines."""
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
    else:
        ax.grid(False, which="both", axis="both")
    ax.tick_params(axis="both", which="major", labelsize=tick_size, colors="black")
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_visible(boxed)
        spine.set_color("black")
        spine.set_linewidth(spine_width)


def style_legend(legend) -> None:
    """Give legends the black-framed look used in the TASEP reference."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_linewidth(1.2)
    frame.set_alpha(1.0)

def save_figure(fig, path, dpi: int = DEFAULT_DPI) -> None:
    """Save a figure in PNG and SVG formats with publication-friendly defaults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _p_to_stars(p_value: float) -> str:
    """Convert a p-value to the significance-star convention in Fig 3 plots."""
    if not np.isfinite(p_value):
        return "ns"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def pairwise_mannwhitney_stats(data, labels) -> list[dict]:
    """Compute all pairwise two-sided Mann-Whitney U tests."""
    clean_data = []
    clean_labels = []
    for vals, label in zip(data, labels):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        clean_data.append(arr)
        clean_labels.append(label)

    stats = []
    for i in range(len(clean_data) - 1):
        for j in range(i + 1, len(clean_data)):
            group_a = clean_data[i]
            group_b = clean_data[j]
            try:
                test = mannwhitneyu(group_a, group_b, alternative="two-sided")
                u_stat = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                u_stat = np.nan
                p_value = np.nan
            stats.append(
                {
                    "group_1": clean_labels[i],
                    "group_2": clean_labels[j],
                    "n_1": int(group_a.size),
                    "n_2": int(group_b.size),
                    "median_1": float(np.median(group_a)),
                    "median_2": float(np.median(group_b)),
                    "mean_1": float(np.mean(group_a)),
                    "mean_2": float(np.mean(group_b)),
                    "mannwhitney_u": u_stat,
                    "p_value": p_value,
                    "significance": _p_to_stars(p_value),
                }
            )
    return stats


def box_with_points(
    ax,
    data,
    labels,
    *,
    ylabel: str,
    title: str = "",
    ylim=None,
    max_points_per_group: int = 800,
    seed: int = 7,
    show_stats: bool = False,
    only_significant: bool = True,
    max_percentile_significance: float = 99.5,
) -> list[dict]:
    """Draw Fig-3-like white box/whisker plots with black jittered points."""
    clean_data = []
    clean_labels = []
    for vals, label in zip(data, labels):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        clean_data.append(arr)
        clean_labels.append(label)

    if not clean_data:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return []

    ax.boxplot(
        clean_data,
        positions=np.arange(1, len(clean_data) + 1),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.5},
        medianprops={"color": "red", "linewidth": 1.6},
        whiskerprops={"color": "black", "linewidth": 1.3},
        capprops={"color": "black", "linewidth": 1.3},
    )

    rng = np.random.default_rng(seed)
    for xpos, vals in enumerate(clean_data, start=1):
        plot_vals = vals
        if vals.size > max_points_per_group:
            plot_vals = rng.choice(vals, size=max_points_per_group, replace=False)
        jitter = rng.uniform(-0.16, 0.16, size=plot_vals.size)
        ax.scatter(
            np.full(plot_vals.size, xpos) + jitter,
            plot_vals,
            s=10,
            c="black",
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks(np.arange(1, len(clean_labels) + 1))
    ax.set_xticklabels(
        [f"{label}\nn={len(vals)}" for label, vals in zip(clean_labels, clean_data)],
        fontname=FONT_FAMILY,
    )
    ax.set_ylabel(ylabel, fontname=FONT_FAMILY)
    ax.set_title(title, fontname=FONT_FAMILY)
    if ylim is not None and not show_stats:
        ax.set_ylim(ylim)
    style_axes(ax, grid=False)

    stats = pairwise_mannwhitney_stats(clean_data, clean_labels)
    if show_stats and len(clean_data) > 1:
        visible_stats = [
            row for row in stats
            if not (only_significant and row["significance"] == "ns")
        ]
        if visible_stats:
            pooled = np.concatenate(clean_data)
            global_max = float(np.nanpercentile(pooled, max_percentile_significance))
            global_min = float(np.nanmin(pooled))
            if ylim is not None:
                global_min = min(global_min, float(ylim[0]))
            data_range = global_max - global_min
            if data_range <= 0:
                data_range = 1.0
            offset = 0.10 * data_range
            bar_height = 0.025 * data_range

            ymax = global_max + offset * (len(visible_stats) + 1) + bar_height
            ymin = float(ylim[0]) if ylim is not None else global_min - 0.04 * data_range
            ax.set_ylim(ymin, ymax)

            label_to_position = {label: idx + 1 for idx, label in enumerate(clean_labels)}
            for bracket_idx, row in enumerate(visible_stats):
                x1 = label_to_position[row["group_1"]]
                x2 = label_to_position[row["group_2"]]
                y = global_max + offset * (bracket_idx + 1)
                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + bar_height, y + bar_height, y],
                    lw=1.2,
                    c="black",
                    clip_on=False,
                )
                ax.text(
                    (x1 + x2) * 0.5,
                    y + bar_height,
                    row["significance"],
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=10,
                    fontname=FONT_FAMILY,
                    clip_on=False,
                )

    return stats
