#!/usr/bin/env python3
"""Generate multi-page PDF with ALL burst-quantification traces.

Each page shows 10 traces (intensity + ON/OFF status), ordered
sequentially by trajectory index (0, 1, 2, …). One PDF per construct.

Usage:
    python generate_all_traces_pdf.py          # all constructs
    python generate_all_traces_pdf.py 4sf      # single construct
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ── Path setup ──────────────────────────────────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import (
    TRACE_GRAY,
    TRACE_GREEN,
    set_publication_style,
    style_axes,
)
from burst_quantification import runs_from_binary, _off_baseline_bar

# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_ROOT = (
    Path(__file__).resolve().parent / "results" / "burst_quantification"
)
TRACES_PER_PAGE = 10
ON_COLOR = TRACE_GREEN
OFF_COLOR = "#c0392b"


def generate_pdf(construct_dir: Path, traces_per_page: int = TRACES_PER_PAGE):
    """Generate a multi-page PDF for one construct directory."""
    construct_dir = Path(construct_dir)

    # Load saved arrays
    processed = np.load(construct_dir / "processed_matrix.npy")
    binary = np.load(construct_dir / "binary_matrix.npy")
    with open(construct_dir / "params.json") as f:
        params = json.load(f)

    condition = params.get("condition", construct_dir.name)
    dt = params.get("time_interval_seconds", 5.0)
    threshold = params.get("threshold", 0.05)
    threshold_mode = params.get("threshold_mode", "fraction_of_trace_max")
    off_q = params.get("off_baseline_quantile", 0.25)

    n_traces, n_time = processed.shape
    t_min = np.arange(n_time) * dt / 60.0
    n_pages = int(np.ceil(n_traces / traces_per_page))

    pdf_path = construct_dir / "plots" / "all_traces.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    print(f"  {condition}: {n_traces} traces → {n_pages} pages → {pdf_path.name}")

    with PdfPages(str(pdf_path)) as pdf:
        for page in range(n_pages):
            start = page * traces_per_page
            end = min(start + traces_per_page, n_traces)
            n_on_page = end - start

            fig = plt.figure(figsize=(14, 2.4 * n_on_page), facecolor="white")
            outer = gridspec.GridSpec(
                n_on_page, 1, hspace=0.55, figure=fig
            )

            for row_idx, i in enumerate(range(start, end)):
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer[row_idx],
                    height_ratios=[3.0, 1.0], hspace=0.08,
                )
                ax_int = fig.add_subplot(inner[0])
                ax_stat = fig.add_subplot(inner[1], sharex=ax_int)

                # ── Top: smoothed intensity trace + threshold line ──
                ax_int.plot(t_min, processed[i], linewidth=0.9, color=TRACE_GRAY)

                if threshold_mode == "fraction_of_trace_max":
                    thr_val = threshold * np.nanmax(processed[i])
                    ax_int.axhline(
                        thr_val, color=OFF_COLOR, linestyle="--",
                        linewidth=0.7, alpha=0.7,
                    )
                elif threshold_mode == "absolute_raw":
                    ax_int.axhline(
                        threshold, color=OFF_COLOR, linestyle="--",
                        linewidth=0.7, alpha=0.7,
                    )
                elif threshold_mode == "off_baseline_mad":
                    thr_val = _off_baseline_bar(processed[i], threshold, off_q)
                    if np.isfinite(thr_val):
                        ax_int.axhline(
                            thr_val, color=OFF_COLOR, linestyle="--",
                            linewidth=0.7, alpha=0.7,
                        )

                ax_int.set_ylabel("Intensity", fontsize=9)
                ax_int.set_title(
                    f"Trajectory #{i}", fontsize=10, loc="left"
                )
                style_axes(
                    ax_int, grid=False, spine_width=1.0,
                    tick_size=9, label_size=9, title_size=10,
                )
                plt.setp(ax_int.get_xticklabels(), visible=False)
                ax_int.tick_params(axis="x", which="both", length=0)

                # ── Bottom: ON/OFF state ──
                bin_row = binary[i]
                runs = runs_from_binary(bin_row)
                prev_was_state = False
                for val, s, e in runs:
                    if np.isnan(val):
                        prev_was_state = False
                        continue
                    y = 1.0 if val == 1.0 else 0.0
                    c = ON_COLOR if val == 1.0 else OFF_COLOR
                    ax_stat.hlines(
                        y, s * dt / 60.0, e * dt / 60.0,
                        color=c, linewidth=2.6,
                    )
                    if prev_was_state:
                        ax_stat.axvline(
                            s * dt / 60.0, color="gray", linestyle="--",
                            linewidth=0.6, alpha=0.55,
                            ymin=0.15, ymax=0.85,
                        )
                    prev_was_state = True

                ax_stat.set_ylim(-0.4, 1.4)
                ax_stat.set_yticks([0.0, 1.0])
                ax_stat.set_yticklabels(["OFF", "ON"], fontsize=9)
                style_axes(
                    ax_stat, grid=False, spine_width=1.0,
                    tick_size=9, label_size=9, title_size=10,
                )
                if row_idx == n_on_page - 1:
                    ax_stat.set_xlabel("Time (min)", fontsize=10)
                else:
                    plt.setp(ax_stat.get_xticklabels(), visible=False)
                    ax_stat.tick_params(axis="x", which="both", length=0)

            fig.suptitle(
                f"{condition} — All Traces  (page {page + 1}/{n_pages})",
                fontsize=14, fontname="Arial",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    print(f"    ✓ Saved {pdf_path}")
    return pdf_path


def main():
    constructs = sorted([
        d for d in RESULTS_ROOT.iterdir()
        if d.is_dir()
        and (d / "processed_matrix.npy").exists()
        and d.name not in ("comparison", "diagnostics")
    ])

    # Optional filter by CLI argument
    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        constructs = [c for c in constructs if c.name in requested]

    if not constructs:
        print("No construct directories found.")
        return

    print(f"\nGenerating all-traces PDFs for {len(constructs)} construct(s)…\n")
    for cdir in constructs:
        generate_pdf(cdir)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
