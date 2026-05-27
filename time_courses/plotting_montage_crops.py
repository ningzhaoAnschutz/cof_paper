#!/usr/bin/env python3
"""Cell crop time-course montage utilities for CoF burst examples."""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import matplotlib as _mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter, uniform_filter1d

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))
local_microlive = repo_root.parent / "microlive"
if local_microlive.exists():
    sys.path.insert(0, str(local_microlive))

from microlive import microscopy as mi
from plotting import (
    CHANNEL_GREEN,
    CHANNEL_MAGENTA,
    TRACE_BLUE,
    TRACE_GREEN,
    TRACE_MAGENTA,
    set_publication_style,
    style_legend,
)


@dataclass(frozen=True)
class ParticleOrigin:
    """Source-data provenance for one input matrix row."""

    results_dir: Path
    particle_id: int
    lif_path: Path
    series_index: int
    total_frames_movie: int
    first_valid_frame: int
    time_interval_seconds: float


DEFAULT_CHANNELS = [
    {
        "index": 1,
        "label": "Nascent",
        "trace_color": TRACE_MAGENTA,
        "crop_color": CHANNEL_MAGENTA,
    },
    {
        "index": 0,
        "label": "Folding",
        "trace_color": TRACE_GREEN,
        "crop_color": CHANNEL_GREEN,
    },
]

_LIF_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_MAX_CACHE_ITEMS = 2


def parse_metadata_txt(results_dir) -> dict:
    """Parse a MicroLive Metadata_*.txt file from a results directory."""
    results_dir = Path(results_dir)
    meta_files = sorted(results_dir.glob("Metadata_*.txt"))
    if not meta_files:
        raise FileNotFoundError(f"No Metadata_*.txt in {results_dir}")

    info: dict[str, object] = {}
    with open(meta_files[0], encoding="utf-8") as handle:
        for line in handle:
            parts = re.split(r"\.{2,}\s*", line.strip(), maxsplit=1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            if key == "Data Folder Path":
                info["lif_path_raw"] = val
            elif key == "Image Name":
                info["series_name"] = val
            elif key == "Selected Image Index":
                info["series_index"] = int(val)
            elif key == "Time Interval (s)":
                info["time_interval_seconds"] = float(val)
            elif "Image Dimensions" in key:
                dims = [int(x.strip()) for x in val.strip("()").split(",")]
                if dims:
                    info["total_frames_movie"] = dims[0]
    return info


def resolve_lif_path(raw_path, data_root, construct_name) -> Path:
    """Resolve stale metadata LIF paths to the current mounted data location."""
    path = Path(raw_path)
    if path.exists():
        return path

    data_root = Path(data_root)
    basename = path.name
    search_root = data_root / construct_name
    candidates = sorted(search_root.glob(basename))
    if candidates:
        return candidates[0]

    recursive_candidates = sorted(search_root.rglob(basename))
    if recursive_candidates:
        return recursive_candidates[0]

    raise FileNotFoundError(f"Cannot resolve LIF path from metadata: {raw_path}")


def auto_snapshot_frames(
    n_frames: int,
    n_snapshots: int = 10,
    time_interval_seconds: float = 5.0,
) -> list[int]:
    """Return evenly spaced frame indices across the movie duration.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the movie.
    n_snapshots : int
        Desired number of evenly spaced snapshots.  Capped at *n_frames*.
    time_interval_seconds : float
        Retained for API compatibility; not used internally.
    """
    n_frames = int(n_frames)
    n_snapshots = int(n_snapshots)
    if n_frames <= 0:
        return []
    if n_snapshots <= 1:
        return [0]

    # Cap so we never request more snapshots than available frames
    n_req = min(n_snapshots, n_frames)
    frames = np.linspace(0, n_frames - 1, n_req)
    unique_frames = sorted({int(round(f)) for f in frames})
    return unique_frames


def _clear_lif_cache() -> None:
    """Free cached image stacks and tracking tables."""
    _LIF_CACHE.clear()


def load_montage_data_cached(
    origin: ParticleOrigin,
    *,
    apply_photobleaching: bool = False,
    photobleaching_mode: str = "entire_image",
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict:
    """Load one origin's scene/data with a tiny LRU cache."""
    key = (
        str(origin.lif_path),
        int(origin.series_index),
        str(origin.results_dir),
        bool(apply_photobleaching),
        photobleaching_mode,
        float(origin.time_interval_seconds),
        max_frames,
    )
    if key in _LIF_CACHE:
        _LIF_CACHE.move_to_end(key)
        return _LIF_CACHE[key]

    data = load_montage_data(
        origin.lif_path,
        origin.series_index,
        origin.results_dir,
        apply_photobleaching=apply_photobleaching,
        photobleaching_mode=photobleaching_mode,
        time_interval_seconds=origin.time_interval_seconds,
        max_frames=max_frames,
        verbose=verbose,
    )
    _LIF_CACHE[key] = data
    _LIF_CACHE.move_to_end(key)
    while len(_LIF_CACHE) > _MAX_CACHE_ITEMS:
        _LIF_CACHE.popitem(last=False)
    return data


def load_montage_data(
    lif_path,
    series_index,
    results_folder,
    *,
    apply_photobleaching: bool = False,
    photobleaching_mode: str = "entire_image",
    time_interval_seconds: float | None = None,
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict:
    """Load the original image scene, tracking CSV, and mask for one FOV."""
    lif_path = Path(lif_path)
    results_folder = Path(results_folder)
    if not lif_path.exists():
        raise FileNotFoundError(f"LIF file not found: {lif_path}")
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")

    reader = mi.ReadLif(
        lif_path,
        show_metadata=False,
        save_tif=False,
        save_png=False,
        format="TZYXC",
        lazy=True,
    )
    scenes = list(reader._aics.scenes)
    if not (0 <= int(series_index) < len(scenes)):
        raise IndexError(
            f"series_index {series_index} out of range for {lif_path.name} "
            f"({len(scenes)} scenes)"
        )
    image_TZYXC = reader.read_scene(int(series_index))

    # Truncate to max_frames BEFORE photobleaching correction
    if max_frames is not None and max_frames > 0:
        image_TZYXC = image_TZYXC[:max_frames]

    pixel_size_xy_um = abs(reader._aics.physical_pixel_sizes.Y or 0) or np.nan
    series_name = scenes[int(series_index)]

    metadata = {}
    if time_interval_seconds is None:
        try:
            metadata = parse_metadata_txt(results_folder)
            time_interval_seconds = metadata.get("time_interval_seconds")
        except Exception:
            time_interval_seconds = None
    dt = float(time_interval_seconds or 5.0)
    if dt <= 0:
        dt = 5.0

    csv_files = sorted(results_folder.glob("tracking_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tracking_*.csv in {results_folder}")
    tracking_df = pd.read_csv(csv_files[0])

    # Truncate tracking data to max_frames
    if max_frames is not None and max_frames > 0 and "frame" in tracking_df.columns:
        tracking_df = tracking_df[tracking_df["frame"] < max_frames].copy()

    mask_files = sorted(results_folder.glob("mask_*.tif"))
    mask_YX = None
    if mask_files:
        mask_YX = tifffile.imread(mask_files[0])
        while getattr(mask_YX, "ndim", 0) > 2:
            mask_YX = np.max(mask_YX, axis=0)
        mask_YX = mask_YX > 0

    decay_rates = None
    pb_data = None
    if apply_photobleaching:
        if verbose:
            print(f"    Applying photobleaching correction: {lif_path.name} scene {series_index}")
        corrected, pb_data = mi.Photobleaching(
            image_TZYXC=image_TZYXC,
            mask_YX=mask_YX,
            show_plot=False,
            mode=photobleaching_mode,
            time_interval_seconds=dt,
            verbose=False,
        ).apply_photobleaching_correction()
        image_TZYXC = corrected
        decay_rates = pb_data.get("decay_rates") if isinstance(pb_data, dict) else None

    return {
        "image_TZYXC": image_TZYXC,
        "tracking_df": tracking_df,
        "mask_YX": mask_YX,
        "time_interval_seconds": dt,
        "pixel_size_xy_um": float(pixel_size_xy_um),
        "series_name": metadata.get("series_name", series_name),
        "photobleaching_applied": bool(apply_photobleaching),
        "decay_rates": decay_rates,
        "photobleaching_data": pb_data,
        "projection_cache": {},
    }


def _validate_inputs(image_TZYXC, tracking_df, particle_id, channels) -> None:
    if np.asarray(image_TZYXC).ndim != 5:
        raise ValueError("image_TZYXC must be a 5D array with shape [T,Z,Y,X,C]")
    required = {"particle", "frame", "x", "y"}
    missing = required.difference(tracking_df.columns)
    if missing:
        raise ValueError(f"tracking_df missing required columns: {sorted(missing)}")
    if not (tracking_df["particle"] == particle_id).any():
        raise ValueError(f"particle_id {particle_id!r} not found in tracking_df")
    n_channels = image_TZYXC.shape[-1]
    for ch in channels:
        idx = int(ch["index"])
        if not (0 <= idx < n_channels):
            raise ValueError(f"Channel index {idx} out of range for {n_channels} channels")
        field = f"spot_int_ch_{idx}"
        if field not in tracking_df.columns:
            raise ValueError(f"tracking_df missing required column: {field}")


def _smooth_preserve_nans(trace, window: int):
    trace = np.asarray(trace, dtype=float)
    finite_idx = np.where(np.isfinite(trace))[0]
    if int(window) <= 1 or finite_idx.size < 2:
        return trace
    first, last = finite_idx[0], finite_idx[-1]
    all_idx = np.arange(first, last + 1)
    interp_vals = np.interp(all_idx, finite_idx, trace[finite_idx])
    smoothed = uniform_filter1d(interp_vals, size=int(window), mode="nearest")
    out = np.full_like(trace, np.nan, dtype=float)
    out[first:last + 1] = smoothed
    return out


def _extract_trace(particle_df, channel_index: int, n_frames: int, smooth_window: int):
    field = f"spot_int_ch_{channel_index}"
    trace = np.full(n_frames, np.nan, dtype=float)
    sub = particle_df.dropna(subset=["frame"])
    frames = sub["frame"].to_numpy(dtype=int)
    values = sub[field].to_numpy(dtype=float)
    valid = (frames >= 0) & (frames < n_frames)
    trace[frames[valid]] = values[valid]
    if smooth_window > 1:
        trace = _smooth_preserve_nans(trace, smooth_window)
    return trace


def _normalize_trace(trace, mode: str):
    trace = np.asarray(trace, dtype=float)
    if mode == "raw":
        return trace
    out = trace.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    if mode == "min_max":
        lo = np.nanmin(out)
        hi = np.nanmax(out)
        if hi > lo:
            out = (out - lo) / (hi - lo)
        return out
    if mode == "total_intensity":
        total = np.nansum(out)
        if total > 0:
            out = out / total
        return out
    raise ValueError(f"Unknown trace_norm_mode: {mode}")


def _get_coordinate(
    particle_df,
    frame: int,
    coordinate_mode: str,
    *,
    max_frame_distance: int | None = 2,
):
    coords = particle_df.dropna(subset=["frame", "x", "y"])
    if coords.empty:
        if coordinate_mode == "raise":
            raise ValueError("No finite coordinates for particle")
        return None
    frames = coords["frame"].to_numpy(dtype=int)
    exact = np.where(frames == int(frame))[0]
    if exact.size:
        row = coords.iloc[int(exact[0])]
        return float(row["x"]), float(row["y"])

    if coordinate_mode == "blank":
        return None
    if coordinate_mode == "raise":
        raise ValueError(f"No coordinate at frame {frame}")
    if coordinate_mode != "nearest_valid":
        raise ValueError(f"Unknown coordinate_mode: {coordinate_mode}")

    nearest = int(np.argmin(np.abs(frames - int(frame))))
    distance = abs(int(frames[nearest]) - int(frame))
    if max_frame_distance is not None and distance > int(max_frame_distance):
        return None
    row = coords.iloc[nearest]
    return float(row["x"]), float(row["y"])


def _coerce_gaussian_filter_value(value) -> float:
    sigma = float(value or 0)
    if sigma < 0:
        raise ValueError("gaussian_filter_value must be >= 0")
    return sigma


def _projection_cache_image_key(image_TZYXC) -> tuple:
    """Namespace projection-cache entries to one in-memory image stack."""
    arr = np.asarray(image_TZYXC)
    return id(arr), tuple(arr.shape)


def _resolve_averaged_frames(selected_frame: int, n_total_frames: int) -> list[int]:
    """Return 3 frame indices for temporal averaging with edge clamping.

    Normal case:  [selected_frame-1, selected_frame, selected_frame+1]
    First frame (0):  [0, 1, 2]
    Last frame (N-1): [N-3, N-2, N-1]
    Second-to-last (N-2): [N-3, N-2, N-1]
    """
    n = int(n_total_frames)
    f = int(selected_frame)
    if n <= 3:
        return list(range(n))
    if f <= 0:
        return [0, 1, 2]
    if f >= n - 1:
        return [n - 3, n - 2, n - 1]
    if f >= n - 2:
        return [n - 3, n - 2, n - 1]
    return [f - 1, f, f + 1]


def _full_frame_max_projection(
    image_TZYXC,
    frame: int,
    channel: int,
    gaussian_filter_value: float = 0,
    projection_cache: dict | None = None,
):
    """Return the full max-Z frame, optionally Gaussian filtered before crops."""
    image_TZYXC = np.asarray(image_TZYXC)
    n_frames, _, height, width, _ = image_TZYXC.shape
    if not (0 <= int(frame) < n_frames):
        raise IndexError(f"Frame {frame} out of range for {n_frames} frames")
    if not (0 <= int(channel) < image_TZYXC.shape[-1]):
        raise IndexError(f"Channel {channel} out of range for {image_TZYXC.shape[-1]} channels")

    sigma = _coerce_gaussian_filter_value(gaussian_filter_value)
    key = ("raw", _projection_cache_image_key(image_TZYXC), int(frame), int(channel), sigma)
    if projection_cache is not None and key in projection_cache:
        return projection_cache[key]

    projection = np.max(image_TZYXC[int(frame), :, :, :, int(channel)], axis=0)
    projection = np.asarray(projection, dtype=float)
    if sigma > 0:
        projection = gaussian_filter(projection, sigma=sigma, mode="nearest")
    assert projection.shape == (height, width)

    if projection_cache is not None:
        projection_cache[key] = projection
    return projection


def _full_frame_max_projection_averaged(
    image_TZYXC,
    selected_frame: int,
    channel: int,
    gaussian_filter_value: float = 0,
    projection_cache: dict | None = None,
):
    """Return a temporally averaged max-Z projection over 3 frames.

    Averages ``selected_frame-1``, ``selected_frame``, and ``selected_frame+1``
    with edge clamping via :func:`_resolve_averaged_frames`.
    """
    image_TZYXC = np.asarray(image_TZYXC)
    n_frames = image_TZYXC.shape[0]
    avg_frames = _resolve_averaged_frames(selected_frame, n_frames)

    sigma = _coerce_gaussian_filter_value(gaussian_filter_value)
    key = ("avg3", _projection_cache_image_key(image_TZYXC), int(selected_frame), int(channel), sigma)
    if projection_cache is not None and key in projection_cache:
        return projection_cache[key]

    projections = []
    for f in avg_frames:
        proj = _full_frame_max_projection(
            image_TZYXC,
            int(f),
            channel,
            gaussian_filter_value=0,
            projection_cache=projection_cache,
        )
        projections.append(np.asarray(proj, dtype=float))
    averaged = np.mean(projections, axis=0)
    if sigma > 0:
        averaged = gaussian_filter(averaged, sigma=sigma, mode="nearest")

    if projection_cache is not None:
        projection_cache[key] = averaged
    return averaged


def _crop_from_projection(projection_YX, y: float, x: float, crop_size_px: int):
    """Extract a fixed-size crop from a precomputed full-frame projection."""
    projection_YX = np.asarray(projection_YX, dtype=float)
    height, width = projection_YX.shape
    crop_size_px = int(crop_size_px)
    x_c, y_c = int(np.round(x)), int(np.round(y))
    half = crop_size_px // 2
    y0, x0 = y_c - half, x_c - half
    y1, x1 = y0 + crop_size_px, x0 + crop_size_px

    src_y0, src_y1 = max(0, y0), min(height, y1)
    src_x0, src_x1 = max(0, x0), min(width, x1)
    crop = np.zeros((crop_size_px, crop_size_px), dtype=float)
    if src_y1 > src_y0 and src_x1 > src_x0:
        source = projection_YX[src_y0:src_y1, src_x0:src_x1]
        dst_y0 = src_y0 - y0
        dst_x0 = src_x0 - x0
        crop[
            dst_y0:dst_y0 + source.shape[0],
            dst_x0:dst_x0 + source.shape[1],
        ] = source
    assert crop.shape == (crop_size_px, crop_size_px)
    return crop


def _extract_crop(
    image_TZYXC, frame: int, y: float, x: float, channel: int,
    crop_size_px: int, gaussian_filter_value: float = 0,
    projection_cache: dict | None = None,
    average_3_frames: bool = False,
):
    """Extract a crop from the full max-Z frame after optional Gaussian filtering.

    When ``average_3_frames`` is True, the crop is extracted from a temporally
    averaged max-Z projection of 3 neighbouring frames (see
    :func:`_full_frame_max_projection_averaged`).
    """
    if average_3_frames:
        projection = _full_frame_max_projection_averaged(
            image_TZYXC,
            frame,
            channel,
            gaussian_filter_value=gaussian_filter_value,
            projection_cache=projection_cache,
        )
    else:
        projection = _full_frame_max_projection(
            image_TZYXC,
            frame,
            channel,
            gaussian_filter_value=gaussian_filter_value,
            projection_cache=projection_cache,
        )
    return _crop_from_projection(projection, y, x, crop_size_px)


def _blank_crop(crop_size_px: int):
    return np.zeros((int(crop_size_px), int(crop_size_px)), dtype=float)


def _normalize_channel_crops(snapshot_crops, trajectory_crops, mode: str):
    if mode == "raw":
        return [np.asarray(c, dtype=float) for c in snapshot_crops]
    if not snapshot_crops:
        return []
    if mode == "trajectory_max":
        values = trajectory_crops if trajectory_crops else snapshot_crops
        scale = max((float(np.nanmax(c)) for c in values if np.size(c)), default=0.0)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return [np.asarray(c, dtype=float) / scale for c in snapshot_crops]
    if mode == "trajectory_min_max":
        # Global percentile stretch across the full trajectory — robust to
        # outlier hot pixels.  Background → black, bright spots → white,
        # relative intensity changes preserved across time.
        values = trajectory_crops if trajectory_crops else snapshot_crops
        all_vals = np.concatenate([np.asarray(c, dtype=float).ravel() for c in values])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            return [np.zeros_like(c, dtype=float) for c in snapshot_crops]
        lo, hi = np.percentile(finite, [5, 95])
        if hi <= lo:
            hi = lo + 1.0
        return [np.clip((np.asarray(c, dtype=float) - lo) / (hi - lo), 0, 1) for c in snapshot_crops]
    if mode == "per_channel_percentile":
        stack = np.concatenate([np.asarray(c, dtype=float).ravel() for c in snapshot_crops])
        finite = stack[np.isfinite(stack)]
        if finite.size == 0:
            return [np.zeros_like(c, dtype=float) for c in snapshot_crops]
        lo, hi = np.percentile(finite, [1, 99.8])
        if hi <= lo:
            hi = lo + 1.0
        return [np.clip((np.asarray(c, dtype=float) - lo) / (hi - lo), 0, 1) for c in snapshot_crops]
    if mode == "per_crop_percentile":
        # Per-crop contrast stretch — each crop is independently stretched
        # so that spots pop bright against a black background.
        out = []
        for c in snapshot_crops:
            c = np.asarray(c, dtype=float)
            finite = c[np.isfinite(c)]
            if finite.size == 0 or np.ptp(finite) == 0:
                out.append(np.zeros_like(c, dtype=float))
                continue
            lo, hi = np.percentile(finite, [5, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            out.append(np.clip((c - lo) / (hi - lo), 0, 1))
        return out
    raise ValueError(f"Unknown crop_norm_mode: {mode}")


def _to_rgb(norm_crop, color_tuple):
    norm_crop = np.asarray(norm_crop, dtype=float)
    rgb = np.zeros((*norm_crop.shape, 3), dtype=float)
    color = np.asarray(color_tuple, dtype=float)
    for idx in range(3):
        rgb[..., idx] = norm_crop * color[idx]
    return np.clip(rgb, 0, 1)


def plot_cell_crop_timecourse_montage(
    image_TZYXC,
    tracking_df,
    particle_id,
    snapshot_frames,
    *,
    time_interval_seconds: float = 5.0,
    channels=None,
    crop_size_px: int = 13,
    coordinate_mode: str = "nearest_valid",
    coordinate_max_frame_distance: int | None = 2,
    crop_norm_mode: str = "trajectory_max",
    trace_norm_mode: str = "raw",
    smooth_window: int = 1,
    gaussian_filter_value: float = 1.0,
    binary_state=None,
    first_valid_frame: int = 0,
    show_merge: bool = True,
    trace_scale: float | None = None,
    crop_colormap: str | None = None,
    section_height_ratios: list | tuple | None = None,
    show_crop_time_labels: bool = True,
    display_3_crops_averaged: bool = False,
    projection_cache: dict | None = None,
    fig=None,
    subplot_spec=None,
    panel_label: str | None = None,
    title: str | None = None,
    trim_to_valid: bool = True,
):
    """Plot a trace, ON/OFF bar, and max-Z crop montage for one particle.

    Parameters
    ----------
    section_height_ratios : list of 3 floats, optional
        Relative heights of [trace, on_off_bar, crop_montage].  Values are
        proportional (e.g. [0.60, 0.15, 0.25] means 60 % trace, 15 % state
        bar, 25 % crops).  Default ``[0.60, 0.15, 0.25]``.
    gaussian_filter_value : float, optional
        Sigma for a 2-D Gaussian filter applied to each full max-Z frame
        before crop extraction.  Reduces pixelation in small crops.
        Default ``1.0``.  Set to ``0`` to show raw pixels.
    crop_colormap : str or None, optional
        Matplotlib colormap name (e.g. ``"gray"``) to render crops.  When
        set, each channel's crops are drawn in grayscale (or the chosen
        colormap) instead of the legacy per-channel RGB tinting.  This
        greatly improves spot-to-background contrast.  Default ``None``
        (legacy RGB).
    display_3_crops_averaged : bool, optional
        When True, each crop is extracted from a temporally averaged
        max-Z projection spanning 3 frames: ``selected_frame-1``,
        ``selected_frame``, and ``selected_frame+1``.  Edge handling:
        frame 0 --> averages [0, 1, 2]; last frame N-1 --> averages
        [N-3, N-2, N-1].  This also applies to the merged channel
        row.  Default ``False``.
    projection_cache : dict, optional
        Shared cache for expensive full-frame max projections and Gaussian
        filters.  Pass the same dict for multiple particles from the same FOV
        to avoid recomputing identical frame/channel projections.
    trim_to_valid : bool, optional
        When True (default), trim the plot to only the frame range that
        contains actual tracking data.  Leading/trailing NaN regions are
        removed so the time axis starts at 0 and ends at the last valid
        data point.  Snapshot frames are recomputed within the valid
        range.  Set to False to plot the full movie duration.
    """
    # ── Publication-quality defaults: Arial, black text, thick ticks ──

    _mpl.rcParams.update({
        "font.family": "Arial",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 1.8,
    })

    _SPINE_W = 1.8    # axis box thickness (pt)
    _TICK_W = 1.4     # tick mark thickness
    _TICK_LEN = 6     # tick length (pt)
    _TRACE_LW = 2.4   # intensity trace linewidth
    _STEP_LW = 2.5    # ON/OFF step linewidth

    image_TZYXC = np.asarray(image_TZYXC)
    channels = [dict(ch) for ch in (channels or DEFAULT_CHANNELS)]
    gaussian_filter_value = _coerce_gaussian_filter_value(gaussian_filter_value)
    n_frames = image_TZYXC.shape[0]
    snapshot_frames = [
        int(f) for f in snapshot_frames
        if 0 <= int(f) < n_frames
    ]
    if not snapshot_frames:
        raise ValueError("No snapshot frames fall within the movie frame range")

    _validate_inputs(image_TZYXC, tracking_df, particle_id, channels)
    particle_df = tracking_df[tracking_df["particle"] == particle_id]
    dt = float(time_interval_seconds)
    if projection_cache is None:
        projection_cache = {}

    # ── Optional trim to valid data range ──
    # Determine the first and last movie frame containing actual data for
    # this particle.  When trim_to_valid is True, all plot vectors (traces,
    # binary state, crop snapshots, time labels) are shifted so the plot
    # starts at t = 0 and ends at the last frame with data.
    trim_start = 0
    trim_end = n_frames - 1
    if trim_to_valid and not particle_df.empty and "frame" in particle_df.columns:
        valid_frames = particle_df["frame"].dropna().astype(int)
        if len(valid_frames) > 0:
            trim_start = int(valid_frames.min())
            trim_end = int(valid_frames.max())
    trimmed_n = trim_end - trim_start + 1

    # Recompute snapshot frames within the valid range
    n_snapshots_requested = len(snapshot_frames)
    snapshot_frames_movie = [
        int(f) + trim_start
        for f in auto_snapshot_frames(trimmed_n, n_snapshots_requested, dt)
    ]
    # Clamp to valid movie bounds
    snapshot_frames_movie = [
        f for f in snapshot_frames_movie if 0 <= f < n_frames
    ]
    if not snapshot_frames_movie:
        snapshot_frames_movie = [trim_start]

    t_min = np.arange(trimmed_n) * dt / 60.0
    x_max = max((trimmed_n - 1) * dt / 60.0, 0.1)
    n_cols = len(snapshot_frames_movie)
    n_channel_rows = len(channels)
    use_cmap = crop_colormap is not None
    add_merge = bool(show_merge and len(channels) == 2)
    n_crop_rows = n_channel_rows + (1 if add_merge else 0)

    # Section layout — 3 vertical sections [trace+state, (unused), crops]
    # We merge trace and state into one nested gridspec so they appear
    # tightly integrated with a shared x-axis.
    if section_height_ratios is None:
        section_height_ratios = [0.60, 0.15, 0.25]
    if len(section_height_ratios) != 3:
        raise ValueError(
            f"section_height_ratios must have 3 elements, got {len(section_height_ratios)}"
        )
    # Merge first two ratios for the trace+state block
    upper_ratio = section_height_ratios[0] + section_height_ratios[1]
    crop_ratio = section_height_ratios[2]
    # Internal split: trace vs state within the upper block
    trace_frac = section_height_ratios[0] / upper_ratio
    state_frac = section_height_ratios[1] / upper_ratio

    if fig is None:
        fig = plt.figure(figsize=(max(18, n_cols * 0.6), 7), facecolor="white")
        outer_gs = fig.add_gridspec(
            2, 1, height_ratios=[upper_ratio, crop_ratio], hspace=0.30,
        )
        standalone = True
    else:
        if subplot_spec is None:
            raise ValueError("subplot_spec must be provided when drawing into an existing fig")
        outer_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplot_spec,
            height_ratios=[upper_ratio, crop_ratio], hspace=0.45,
        )
        standalone = False

    # Nested gridspec for trace + state bar — very tight vertical coupling
    upper_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0],
        height_ratios=[trace_frac, state_frac], hspace=0.05,
    )
    ax_trace = fig.add_subplot(upper_gs[0])
    ax_state = fig.add_subplot(upper_gs[1], sharex=ax_trace)

    # Crop gridspec — tight horizontal packing so square crops sit close.
    # wspace=0 eliminates padding between columns; the small crop pixels
    # plus the imshow aspect="equal" will keep squares centred.
    crop_gs = gridspec.GridSpecFromSubplotSpec(
        n_crop_rows, n_cols, subplot_spec=outer_gs[1], wspace=0.0, hspace=0.0
    )

    # ── Intensity traces ──
    for ch in channels:
        trace = _extract_trace(particle_df, int(ch["index"]), n_frames, int(smooth_window))
        trace = _normalize_trace(trace, trace_norm_mode)
        if trace_scale is not None and trace_scale != 0:
            trace = trace / float(trace_scale)
        # Trim to valid range
        trace = trace[trim_start:trim_end + 1]
        # Fill internal NaN gaps with linear interpolation for a continuous
        # plot line.  Edge NaNs (before first / after last valid frame) are
        # left as NaN so the trace starts and stops at true data boundaries.
        finite_mask = np.isfinite(trace)
        if finite_mask.any() and not finite_mask.all():
            valid_idx = np.where(finite_mask)[0]
            first, last = valid_idx[0], valid_idx[-1]
            interior = slice(first, last + 1)
            inner_mask = finite_mask[interior]
            if not inner_mask.all():
                trace[interior] = np.interp(
                    np.arange(interior.start, interior.stop),
                    np.where(finite_mask)[0],
                    trace[finite_mask],
                )
        ax_trace.plot(t_min, trace, color=ch["trace_color"], lw=_TRACE_LW, label=ch["label"])


    ax_trace.set_xlim(0, x_max)
    ax_trace.set_ylabel("Intensity (a.u.)", fontsize=14, color="black")
    if title:
        ax_trace.set_title(title, fontsize=12, loc="left", color="black", fontweight="bold")
    if panel_label:
        ax_trace.text(
            -0.04, 1.06, panel_label, transform=ax_trace.transAxes,
            ha="left", va="bottom", fontsize=16, fontweight="bold", color="black",
        )
    legend = ax_trace.legend(frameon=False, fontsize=12, loc="upper right")
    style_legend(legend)
    for spine in ax_trace.spines.values():
        spine.set_linewidth(_SPINE_W)
    ax_trace.tick_params(
        axis="both", which="major",
        width=_TICK_W, length=_TICK_LEN, labelsize=12, colors="black",
    )
    # Hide x-axis labels on trace — the state bar below shows them
    plt.setp(ax_trace.get_xticklabels(), visible=False)
    ax_trace.tick_params(axis="x", length=0)

    # ── ON/OFF state bar ──
    if binary_state is None:
        state = np.ones(trimmed_n, dtype=float)
        t_state = t_min
    else:
        binary_state = np.asarray(binary_state, dtype=float)
        # binary_state is aligned to the *preprocessed* matrix, which may
        # have been left-shifted by align_first_valid.  In that case,
        # binary frame 0 corresponds to movie frame ``first_valid_frame``.
        # Convert the movie-coordinate trim range to binary-vector indices.
        align_offset = first_valid_frame  # 0 when no alignment was applied
        bs_start = max(0, trim_start - align_offset)
        bs_end = min(binary_state.size, trim_end - align_offset + 1)
        state = binary_state[bs_start:bs_end]
        t_state = np.arange(len(state)) * dt / 60.0
    # Fill internal NaN gaps with nearest neighbour
    nan_mask = np.isnan(state)
    if nan_mask.any() and not nan_mask.all():
        s = pd.Series(state)
        state = s.ffill().bfill().to_numpy()
    if t_state.size:
        ax_state.step(t_state, state, where="post", color=TRACE_BLUE, linewidth=_STEP_LW)
    ax_state.set_ylim(-0.25, 1.35)
    ax_state.set_yticks([0, 1])
    ax_state.set_yticklabels(["OFF", "ON"], fontsize=11, color="black")
    ax_state.set_xlim(0, x_max)
    # Explicit 1-minute resolution ticks: 0, 1, 2, …, floor(x_max)
    ax_state.set_xticks(np.arange(0, int(x_max) + 1, 1))
    ax_state.set_xlabel("Time (min)", fontsize=14, color="black")
    for spine in ax_state.spines.values():
        spine.set_linewidth(_SPINE_W)
    ax_state.tick_params(
        axis="both", which="major",
        width=_TICK_W, length=_TICK_LEN, labelsize=12, colors="black",
    )

    snapshot_coords = [
        _get_coordinate(
            particle_df,
            frame,
            coordinate_mode,
            max_frame_distance=coordinate_max_frame_distance,
        )
        for frame in snapshot_frames_movie
    ]

    normalized_by_channel: list[list[np.ndarray]] = []
    for ch in channels:
        ch_idx = int(ch["index"])
        snapshot_crops = []
        for frame, coord in zip(snapshot_frames_movie, snapshot_coords):
            if coord is None:
                snapshot_crops.append(_blank_crop(crop_size_px))
            else:
                x, y = coord
                snapshot_crops.append(_extract_crop(
                    image_TZYXC, frame, y, x, ch_idx, crop_size_px,
                    gaussian_filter_value=gaussian_filter_value,
                    projection_cache=projection_cache,
                    average_3_frames=display_3_crops_averaged,
                ))

        trajectory_crops = []
        if crop_norm_mode in ("trajectory_max", "trajectory_min_max"):
            valid_coords = particle_df.dropna(subset=["frame", "x", "y"])
            # For trajectory_min_max, use the same gaussian filter as the
            # displayed crops so the normalization range matches what's shown.
            # For trajectory_max, keep raw (gaussian=0) for backward compat.
            traj_gfv = gaussian_filter_value if crop_norm_mode == "trajectory_min_max" else 0
            for _, row in valid_coords.iterrows():
                frame = int(row["frame"])
                if 0 <= frame < n_frames:
                    trajectory_crops.append(
                        _extract_crop(
                            image_TZYXC, frame, float(row["y"]), float(row["x"]),
                            ch_idx, crop_size_px,
                            gaussian_filter_value=traj_gfv,
                            projection_cache=projection_cache,
                        )
                    )
        normalized_by_channel.append(
            _normalize_channel_crops(snapshot_crops, trajectory_crops, crop_norm_mode)
        )

    crop_axes: list[list] = []
    rgb_rows: list[list[np.ndarray]] = []
    # Label every crop, but keep stamps compact so dense 60-crop montages stay readable.
    crop_time_label_fontsize = 3.5
    crop_time_label_pad = 0.5
    for row_idx, ch in enumerate(channels):
        row_axes = []
        rgb_row = []
        for col_idx, frame in enumerate(snapshot_frames_movie):
            ax = fig.add_subplot(crop_gs[row_idx, col_idx])
            norm_crop = normalized_by_channel[row_idx][col_idx]
            if use_cmap:
                ax.imshow(norm_crop, interpolation="nearest",
                          cmap=crop_colormap, vmin=0, vmax=1, aspect="equal")
            else:
                rgb = _to_rgb(norm_crop, ch["crop_color"])
                ax.imshow(rgb, interpolation="nearest", aspect="equal")
            # Always build RGB-tinted crop for merge row
            rgb_for_merge = _to_rgb(norm_crop, ch["crop_color"])
            rgb_row.append(rgb_for_merge)
            ax.set_axis_off()
            if row_idx == 0 and show_crop_time_labels:
                # Frame number relative to trimmed range
                frame_label = frame - trim_start
                ax.set_title(
                    f"{frame_label}",
                    fontsize=crop_time_label_fontsize,
                    pad=crop_time_label_pad,
                    color="black",
                )
            if col_idx == 0:
                ax.text(
                    -0.12, 0.5, ch["label"], transform=ax.transAxes,
                    ha="right", va="center", fontsize=12, color="black",
                )
            row_axes.append(ax)
        crop_axes.append(row_axes)
        rgb_rows.append(rgb_row)

    if add_merge:
        row_axes = []
        merge_idx = len(channels)
        for col_idx, frame in enumerate(snapshot_frames_movie):
            ax = fig.add_subplot(crop_gs[merge_idx, col_idx])
            rgb = np.clip(rgb_rows[0][col_idx] + rgb_rows[1][col_idx], 0, 1)
            ax.imshow(rgb, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
            if col_idx == 0:
                ax.text(
                    -0.12, 0.5, "Merge", transform=ax.transAxes,
                    ha="right", va="center", fontsize=12, color="black",
                )
            row_axes.append(ax)
        crop_axes.append(row_axes)

    return fig, {"trace": ax_trace, "state_bar": ax_state, "crops": crop_axes}


def _selection_data_key(
    selection,
    apply_photobleaching: bool,
    photobleaching_mode: str,
    max_frames: int | None,
):
    """Cache key for grouping montage selections from the same loaded FOV."""
    origin = selection["origin"]
    return (
        str(origin.lif_path),
        int(origin.series_index),
        str(origin.results_dir),
        bool(apply_photobleaching),
        photobleaching_mode,
        float(origin.time_interval_seconds),
        max_frames,
    )


def _group_selections_by_loaded_data(
    selections,
    apply_photobleaching: bool,
    photobleaching_mode: str,
    max_frames: int | None,
):
    """Return selections ordered to maximize LIF/projection cache reuse."""
    indexed = list(enumerate(selections))
    indexed.sort(
        key=lambda item: (
            _selection_data_key(
                item[1],
                apply_photobleaching,
                photobleaching_mode,
                max_frames,
            ),
            item[0],
        )
    )
    return [selection for _, selection in indexed]


def export_crop_tif(
    image_TZYXC,
    tracking_df,
    particle_id: int,
    channels: list[dict],
    crop_size_px: int,
    output_path,
    *,
    gaussian_filter_value: float = 0,
    max_frames: int | None = None,
):
    """Export a uint16 max-Z crop time-series for one particle as a TIF.

    The output is **trimmed** to the particle's valid frame range
    (first tracked frame → last tracked frame).  Frames inside that
    range where the particle has no coordinate are zero-filled (black).

    Output shape is ``(T_valid, C, Y, X)`` with ``imagej=True`` and
    ``axes="TCYX"`` — opens natively in ImageJ with Time and Channel
    sliders.

    Parameters
    ----------
    image_TZYXC : ndarray
        Full 5-D image stack ``(T, Z, Y, X, C)``.
    tracking_df : DataFrame
        Must contain ``particle``, ``frame``, ``x``, ``y`` columns.
    particle_id : int
        Particle to extract.
    channels : list of dict
        Each dict needs at least ``{"index": int}``.
    crop_size_px : int
        Side length of the square crop.
    output_path : Path or str
        Destination ``.tif`` file.
    gaussian_filter_value : float
        Sigma for optional Gaussian smoothing of each max-Z frame before
        cropping.  ``0`` keeps raw pixels.
    max_frames : int or None
        Truncate the movie to the first N frames.

    Returns
    -------
    bool
        True if the TIF was written, False if skipped (no valid data).
    """
    image_TZYXC = np.asarray(image_TZYXC)
    n_movie_frames = image_TZYXC.shape[0]
    if max_frames is not None and max_frames > 0:
        n_movie_frames = min(n_movie_frames, int(max_frames))

    # Build coordinate lookup from tracking data
    particle_df = tracking_df[tracking_df["particle"] == particle_id]
    valid_rows = particle_df.dropna(subset=["frame", "x", "y"])
    coord_lookup: dict[int, tuple[float, float]] = {}
    for _, row in valid_rows.iterrows():
        f = int(row["frame"])
        if 0 <= f < n_movie_frames:
            coord_lookup[f] = (float(row["x"]), float(row["y"]))

    if not coord_lookup:
        return False  # no valid data — skip export

    # Trim to valid frame range
    first_frame = min(coord_lookup.keys())
    last_frame = max(coord_lookup.keys())
    n_valid_frames = last_frame - first_frame + 1

    n_ch = len(channels)
    stack = np.zeros(
        (n_valid_frames, n_ch, crop_size_px, crop_size_px), dtype=np.float64,
    )
    projection_cache: dict = {}

    # For frames inside the valid range without coordinates, use the
    # nearest available coordinate (up to 2 frames away) for continuity.
    valid_frame_arr = np.array(sorted(coord_lookup.keys()))

    for t_idx in range(n_valid_frames):
        t_movie = first_frame + t_idx  # absolute movie frame

        if t_movie in coord_lookup:
            x, y = coord_lookup[t_movie]
        else:
            # Fill internal gap with nearest tracked coordinate
            distances = np.abs(valid_frame_arr - t_movie)
            nearest_idx = int(np.argmin(distances))
            if distances[nearest_idx] <= 2:
                nearest_frame = int(valid_frame_arr[nearest_idx])
                x, y = coord_lookup[nearest_frame]
            else:
                continue  # leave as zeros (black)

        for ci, ch in enumerate(channels):
            proj = _full_frame_max_projection(
                image_TZYXC, t_movie, int(ch["index"]),
                gaussian_filter_value=gaussian_filter_value,
                projection_cache=projection_cache,
            )
            crop = _crop_from_projection(proj, y, x, crop_size_px)
            stack[t_idx, ci] = crop

    # Convert to uint16
    stack = np.clip(stack, 0, 65535).astype(np.uint16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ch_labels = [ch.get("label", f"ch{ch['index']}") for ch in channels]

    tifffile.imwrite(
        str(output_path),
        stack,
        imagej=True,
        metadata={
            "axes": "TCYX",
            "Labels": ch_labels,
            "Info": (
                f"particle_id={particle_id}  "
                f"first_frame={first_frame}  "
                f"last_frame={last_frame}  "
                f"n_valid_coords={len(coord_lookup)}/{n_valid_frames}"
            ),
        },
    )
    return True


def export_scene_tif(
    image_TZYXC,
    output_path,
    *,
    max_frames: int | None = None,
):
    """Export the full-FOV max-Z projection as a uint16 TIF.

    Output shape is ``(T, C, Y, X)`` — opens in ImageJ with T/C sliders.

    Parameters
    ----------
    image_TZYXC : ndarray
        Full 5-D image stack ``(T, Z, Y, X, C)``.
    output_path : Path or str
        Destination ``.tif`` file.
    max_frames : int or None
        Truncate the movie to the first N frames.
    """
    image_TZYXC = np.asarray(image_TZYXC)
    if max_frames is not None and max_frames > 0:
        image_TZYXC = image_TZYXC[:int(max_frames)]

    # Max-Z projection: (T, Z, Y, X, C) → (T, Y, X, C)
    maxz = np.max(image_TZYXC, axis=1)
    # Transpose to (T, C, Y, X) for ImageJ
    maxz = np.moveaxis(maxz, -1, 1)

    maxz = np.clip(maxz, 0, 65535).astype(np.uint16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(output_path),
        maxz,
        imagej=True,
        metadata={"axes": "TCYX"},
    )


def generate_representative_montage_pdf(
    selections,
    binary_matrix,
    output_path,
    *,
    apply_photobleaching: bool = True,
    photobleaching_mode: str = "entire_image",
    montages_per_page: int = 2,
    n_snapshots: int = 10,
    panel_figsize: tuple | None = None,
    pdf_dpi: int = 200,
    save_individual_montages: bool = True,
    save_combined_pdf: bool = True,
    max_frames: int | None = None,
    verbose: bool = True,
    **plot_kwargs,
):
    """Generate a multi-page PDF of representative crop montages.

    When ``save_individual_montages`` is True (default), each montage is
    also saved as a separate PNG and SVG in a ``plots_time_courses/``
    subfolder next to the PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_matrix = np.asarray(binary_matrix, dtype=float)
    set_publication_style()

    # Subfolders for individual montage exports (only when enabled)
    singles_dir = None
    png_dir = None
    svg_dir = None
    if save_individual_montages:
        singles_dir = output_path.parent / "plots_time_courses"
        png_dir = singles_dir / "png"
        svg_dir = singles_dir / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

    if save_combined_pdf:
        with PdfPages(str(output_path)) as pdf:
            for page_start in range(0, len(selections), montages_per_page):
                page = selections[page_start:page_start + montages_per_page]
                # Panel size: default (18, 8) per panel, user-overridable
                pw, ph = panel_figsize if panel_figsize is not None else (18, 8)
                fig = plt.figure(figsize=(pw, ph * len(page)), facecolor="white")
                page_gs = fig.add_gridspec(len(page), 1, hspace=0.40)

                for slot, selection in enumerate(page):
                    origin = selection["origin"]
                    data = load_montage_data_cached(
                        origin,
                        apply_photobleaching=apply_photobleaching,
                        photobleaching_mode=photobleaching_mode,
                        max_frames=max_frames,
                        verbose=verbose,
                    )
                    n_movie_frames = int(data["image_TZYXC"].shape[0])
                    snapshot_frames = auto_snapshot_frames(
                        n_movie_frames, n_snapshots, origin.time_interval_seconds,
                    )
                    binary_row_index = int(selection["binary_row_index"])
                    traj_id = selection.get("trajectory_id", "trajectory")
                    title = None  # suppress provenance header on the plot
                    plot_cell_crop_timecourse_montage(
                        image_TZYXC=data["image_TZYXC"],
                        tracking_df=data["tracking_df"],
                        particle_id=origin.particle_id,
                        snapshot_frames=snapshot_frames,
                        time_interval_seconds=origin.time_interval_seconds,
                        binary_state=binary_matrix[binary_row_index, :],
                        first_valid_frame=origin.first_valid_frame,
                        projection_cache=data.setdefault("projection_cache", {}),
                        fig=fig,
                        subplot_spec=page_gs[slot],
                        title=title,
                        **plot_kwargs,
                    )

                    if verbose:
                        print(
                            "    Montage:",
                            traj_id,
                            f"origin={selection.get('origin_row_index', '?')}",
                            f"binary={binary_row_index}",
                            f"lif={origin.lif_path.name}",
                            f"scene={origin.series_index}",
                            f"first_valid={origin.first_valid_frame}",
                            f"movie_shape={data['image_TZYXC'].shape}",
                            f"snapshots={snapshot_frames}",
                        )

                pdf.savefig(fig, dpi=pdf_dpi)
                plt.close(fig)

                # Save each montage as individual PNG + SVG
                if save_individual_montages:
                    for slot, selection in enumerate(page):
                        origin = selection["origin"]
                        data = load_montage_data_cached(
                            origin,
                            apply_photobleaching=apply_photobleaching,
                            photobleaching_mode=photobleaching_mode,
                            max_frames=max_frames,
                            verbose=False,
                        )
                        n_movie_frames = int(data["image_TZYXC"].shape[0])
                        snapshot_frames = auto_snapshot_frames(
                            n_movie_frames, n_snapshots, origin.time_interval_seconds,
                        )
                        binary_row_index = int(selection["binary_row_index"])
                        traj_id = selection.get("trajectory_id", "trajectory")
                        title = None  # suppress provenance header
                        single_fig = plt.figure(
                            figsize=panel_figsize if panel_figsize is not None else (18, 8),
                            facecolor="white",
                        )
                        single_gs = single_fig.add_gridspec(1, 1)
                        plot_cell_crop_timecourse_montage(
                            image_TZYXC=data["image_TZYXC"],
                            tracking_df=data["tracking_df"],
                            particle_id=origin.particle_id,
                            snapshot_frames=snapshot_frames,
                            time_interval_seconds=origin.time_interval_seconds,
                            binary_state=binary_matrix[binary_row_index, :],
                            first_valid_frame=origin.first_valid_frame,
                            projection_cache=data.setdefault("projection_cache", {}),
                            fig=single_fig,
                            subplot_spec=single_gs[0],
                            title=title,
                            **plot_kwargs,
                        )
                        safe_name = str(traj_id).replace("/", "_")
                        single_fig.savefig(
                            png_dir / f"{safe_name}.png",
                            dpi=pdf_dpi, bbox_inches="tight", facecolor="white",
                        )
                        single_fig.savefig(
                            svg_dir / f"{safe_name}.svg",
                            bbox_inches="tight", facecolor="white",
                        )
                        plt.close(single_fig)

                _clear_lif_cache()
    else:
        # Only save individual montages
        if save_individual_montages:
            # Individual output filenames are independent, so process by FOV to
            # reuse loaded image stacks and projection/filter caches.
            grouped_selections = _group_selections_by_loaded_data(
                selections,
                apply_photobleaching,
                photobleaching_mode,
                max_frames,
            )
            last_data_key = None
            for selection in grouped_selections:
                data_key = _selection_data_key(
                    selection,
                    apply_photobleaching,
                    photobleaching_mode,
                    max_frames,
                )
                if last_data_key is not None and data_key != last_data_key:
                    _clear_lif_cache()
                last_data_key = data_key

                origin = selection["origin"]
                data = load_montage_data_cached(
                    origin,
                    apply_photobleaching=apply_photobleaching,
                    photobleaching_mode=photobleaching_mode,
                    max_frames=max_frames,
                    verbose=verbose,
                )
                n_movie_frames = int(data["image_TZYXC"].shape[0])
                snapshot_frames = auto_snapshot_frames(
                    n_movie_frames, n_snapshots, origin.time_interval_seconds,
                )
                binary_row_index = int(selection["binary_row_index"])
                traj_id = selection.get("trajectory_id", "trajectory")
                title = None  # suppress provenance header
                single_fig = plt.figure(
                    figsize=panel_figsize if panel_figsize is not None else (18, 8),
                    facecolor="white",
                )
                single_gs = single_fig.add_gridspec(1, 1)
                plot_cell_crop_timecourse_montage(
                    image_TZYXC=data["image_TZYXC"],
                    tracking_df=data["tracking_df"],
                    particle_id=origin.particle_id,
                    snapshot_frames=snapshot_frames,
                    time_interval_seconds=origin.time_interval_seconds,
                    binary_state=binary_matrix[binary_row_index, :],
                    first_valid_frame=origin.first_valid_frame,
                    projection_cache=data.setdefault("projection_cache", {}),
                    fig=single_fig,
                    subplot_spec=single_gs[0],
                    title=title,
                    **plot_kwargs,
                )
                safe_name = str(traj_id).replace("/", "_")
                single_fig.savefig(
                    png_dir / f"{safe_name}.png",
                    dpi=pdf_dpi, bbox_inches="tight", facecolor="white",
                )
                single_fig.savefig(
                    svg_dir / f"{safe_name}.svg",
                    bbox_inches="tight", facecolor="white",
                )
                plt.close(single_fig)

                if verbose:
                    print(
                        "    Montage (Single only):",
                        traj_id,
                        f"origin={selection.get('origin_row_index', '?')}",
                        f"binary={binary_row_index}",
                        f"lif={origin.lif_path.name}",
                        f"scene={origin.series_index}",
                        f"first_valid={origin.first_valid_frame}",
                        f"movie_shape={data['image_TZYXC'].shape}",
                        f"snapshots={snapshot_frames}",
                    )
            _clear_lif_cache()

    _clear_lif_cache()
    if verbose:
        if save_combined_pdf:
            print(f"    Saved representative montage PDF: {output_path}")
        if save_individual_montages:
            print(f"    Individual montages saved to: {singles_dir}")
    return output_path
