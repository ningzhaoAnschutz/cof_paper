"""Associate LIF file images with their MicroLive analysis result folders.

Result folders follow the naming convention:
    results_<lif_stem>_<image_name>[_region]

Run directly:
    python associate_lif_to_results.py

Or import from a notebook:
    from associate_lif_to_results import associate_lif_to_results
"""

# Standard library
import gc
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# Environment self-relaunch: re-exec with the microlive conda Python if needed.
try:
    import microlive  # noqa: F401
except ModuleNotFoundError:
    conda_result = subprocess.run(
        ['conda', 'run', '-n', 'microlive', 'which', 'python'],
        capture_output=True, text=True,
    )
    env_python = conda_result.stdout.strip()
    if not env_python:
        raise RuntimeError(
            "Could not find the 'microlive' conda environment. "
            "Make sure conda is on PATH and the environment exists."
        )
    print(f"Re-launching with microlive env Python: {env_python}")
    os.execv(env_python, [env_python] + sys.argv)

# Third-party (microlive)
from microlive import microscopy as mi

# Repo-root on sys.path so that `utilities` is importable from notebooks.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Configuration lives in runner.py — do not add constants here.


def strip_non_alphanumeric(text: str) -> str:
    """Return lowercase text with all non-alphanumeric characters removed.

    Used to normalise LIF filenames and result folder names for comparison,
    absorbing GUI sanitisation differences (e.g. '#3' stored as '_3').
    """
    return re.sub(r'[^a-z0-9]', '', text.lower())


def collect_result_dirs(results_dir: Path) -> list[Path]:
    """Return all immediate subdirectories inside results_dir."""
    if not results_dir.exists():
        print(f"⚠  Results directory not found: {results_dir}")
        return []
    return [entry for entry in results_dir.iterdir() if entry.is_dir()]


def find_matching_result_dirs(lif_stem: str, image_name: str, candidate_dirs: list[Path]) -> list[Path]:
    """Return every result directory matching a given LIF stem and image name.

    Matching strategy:
      1. Strip the 'results_' prefix from the candidate folder name.
      2. Locate image_name literally (case-insensitive) to find the split point
         between the LIF-stem part and the image-name part. This works for any
         image naming convention (Series001, TileScan, custom names, etc.).
      3. Compare the LIF-stem part to lif_stem via strip_non_alphanumeric(),
         which absorbs GUI character sanitisation differences.
      4. Fallback: compare the fully-cleaned folder name (minus 'results_')
         to clean_lif_stem + clean_image_name, for cases where the GUI also
         sanitised the image name itself.

    Returns a list because the same image may have been processed more than
    once (e.g. two cell regions in one field of view).

    Args:
        lif_stem: LIF filename without extension.
        image_name: Scene/image name as reported by ReadLif.
        candidate_dirs: All subdirectories to search through.

    Returns:
        List of matching result directories (empty if none found).
    """
    clean_lif_stem  = strip_non_alphanumeric(lif_stem)
    clean_image_name = strip_non_alphanumeric(image_name)
    image_suffix_pattern = re.compile(r'_' + re.escape(image_name) + r'(.*?)$', re.IGNORECASE)

    matching_dirs = []
    for candidate_dir in candidate_dirs:
        folder_name = candidate_dir.name
        if not folder_name.lower().startswith('results_'):
            continue
        folder_body = folder_name[len('results_'):]  # everything after 'results_'
        match = image_suffix_pattern.search(folder_body)
        if match:
            lif_stem_part = folder_body[:match.start()]
            if strip_non_alphanumeric(lif_stem_part) == clean_lif_stem:
                matching_dirs.append(candidate_dir)
        elif strip_non_alphanumeric(folder_body) == clean_lif_stem + clean_image_name:
            matching_dirs.append(candidate_dir)
    return matching_dirs


def inspect_result_dir(result_dir: Path) -> dict:
    """Scan a result directory for its tracking CSV and mask TIF files.

    Args:
        result_dir: Path to the MicroLive result directory.

    Returns:
        Dict with keys 'tracking_files' (list[Path]), 'mask_files' (list[Path]),
        and 'is_complete' (bool, True only when both are present).
    """
    tracking_files = sorted(result_dir.glob('tracking_*.csv'))
    mask_files     = sorted(result_dir.glob('mask_*.tif'))
    return {
        'tracking_files': tracking_files,
        'mask_files':     mask_files,
        'is_complete':    bool(tracking_files) and bool(mask_files),
    }


def print_result_dir_contents(result_dir: Path, contents: dict) -> None:
    """Print the tracking/mask file status for one result directory."""
    status_label = '✓' if contents['is_complete'] else 'SKIP'
    print(f'│          [{status_label}] Result folder : {result_dir.name}')
    tracking_label = contents['tracking_files'][0].name if contents['tracking_files'] else '⚠ NOT FOUND (image skipped)'
    mask_label     = contents['mask_files'][0].name     if contents['mask_files']     else '⚠ NOT FOUND (image skipped)'
    print(f'│            ├ tracking  : {tracking_label}')
    print(f'│            └ mask      : {mask_label}')


def associate_lif_to_results(
    lif_dir: Path = None,
    results_dir: Path = None,
    show_metadata: bool = False,
    lif_files_filter: list = None,
) -> dict:
    """Associate each LIF image with its result directory, image array, and metadata.

    An image is considered processed only when its result directory contains
    both a tracking_*.csv and a mask_*.tif file.

    Args:
        lif_dir: Directory containing .lif files.
        results_dir: Directory containing MicroLive result subdirectories.
        show_metadata: If True, pass show_metadata=True to mi.ReadLif.
        lif_files_filter: Optional list of Path(s). When set, only those
            specific LIF files are processed (enables one-at-a-time loading).

    Returns:
        Nested dict: {lif_filename: {image_name: entry_dict}}.
        Each entry_dict contains:
            result_dirs    – list[Path] of complete result directories.
            image          – ndarray [T, Z, Y, X, C] image array.
            time_interval_s – float, seconds between frames (from LIF metadata).
            pixel_xy_um    – float, XY pixel size in micrometres.
            pixel_z_um     – float, Z step size in micrometres.
            ch_names       – list[str] of channel names.
    """
    if lif_dir is None or results_dir is None:
        raise ValueError(
            "lif_dir and results_dir must be provided. "
            "Set them in runner.py and pass them explicitly."
        )
    lif_dir     = Path(lif_dir)
    results_dir = Path(results_dir)

    all_lif_files = sorted(
        f for f in lif_dir.iterdir()
        if f.is_file() and f.suffix.lower() == '.lif'
    )
    if lif_files_filter is not None:
        filter_set = {Path(p).resolve() for p in lif_files_filter}
        lif_files  = [f for f in all_lif_files if f.resolve() in filter_set]
    else:
        lif_files = all_lif_files

    print(f"\n{'='*60}")
    print(f" LIF directory  : {lif_dir}")
    print(f" Results dir    : {results_dir}")
    print(f" LIF files found: {len(all_lif_files)}  (processing {len(lif_files)})")
    print(f"{'='*60}\n")
    if not lif_files:
        print("No .lif files to process.")
        return {}
    all_result_dirs = collect_result_dirs(results_dir)
    print(f"Result folders found: {len(all_result_dirs)}\n")
    association: dict[str, dict[str, dict]] = {}
    for lif_file in lif_files:
        print(f"┌─ LIF file: {lif_file.name}")
        try:
            (
                list_images,
                list_names,
                pixel_xy_um,
                pixel_z_um,
                ch_names,
                _n_channels,
                list_time_intervals,
                _bit_depth,
                _list_laser_lines,
                _list_intensities,
                _list_wave_ranges,
            ) = mi.ReadLif(
                lif_file,
                show_metadata=show_metadata,
                save_tif=False,
                save_png=False,
                format='TZYXC',
            ).read()
        except Exception as exc:
            print(f"│  ⚠  Could not read file: {exc}\n└─\n")
            association[lif_file.name] = {}
            continue
        print(f"│  Images in file: {len(list_names)}")
        processed_images: dict[str, dict] = {}
        for idx, image_name in enumerate(list_names):
            matching_dirs = find_matching_result_dirs(lif_file.stem, image_name, all_result_dirs)
            if not matching_dirs:
                print(f"│    [SKIP]  {image_name!r}  →  no result folder")
                continue
            match_label = '[✓]' if len(matching_dirs) == 1 else f'[✓×{len(matching_dirs)}]'
            print(f"│    {match_label}  {image_name!r}")
            complete_dirs = []
            for result_dir in matching_dirs:
                contents = inspect_result_dir(result_dir)
                print_result_dir_contents(result_dir, contents)
                if contents['is_complete']:
                    complete_dirs.append(result_dir)
            if complete_dirs:
                time_interval_s = list_time_intervals[idx] if list_time_intervals else 1.0
                processed_images[image_name] = {
                    'result_dirs':     complete_dirs,
                    'image':           list_images[idx],
                    'time_interval_s': time_interval_s,
                    'pixel_xy_um':     pixel_xy_um,
                    'pixel_z_um':      pixel_z_um,
                    'ch_names':        ch_names,
                }
        association[lif_file.name] = processed_images
        n_processed = len(processed_images)
        n_total     = len(list_names)
        n_skipped   = n_total - n_processed
        skip_note   = f"  ({n_skipped} skipped – not processed)" if n_skipped else ""
        print(f"│  Processed: {n_processed}/{n_total}{skip_note}")
        print("└─\n")
    total_processed_images = sum(len(v) for v in association.values())
    total_result_dirs      = sum(len(e['result_dirs']) for v in association.values() for e in v.values())
    print(f"{'='*60}")
    print(f" Fully processed images : {total_processed_images}")
    print(f" Result folders kept    : {total_result_dirs}  (>1 = multi-region)")
    print(f" (Images with no/incomplete results were skipped)")
    print(f"{'='*60}\n")
    return association


# ── Trajectory Extraction ─────────────────────────────────────────────────────

def extract_trajectories(result_dir: Path) -> tuple:
    """Extract per-particle trajectories from a result directory's tracking CSV.

    Uses the GUI's ``unique_particle`` column (format ``image_id_cell_id_particle``)
    when present, which guarantees uniqueness within the result folder.  Falls back
    to the plain ``particle`` integer column for older CSV formats.

    Args:
        result_dir: Path to the MicroLive result directory containing tracking_*.csv.

    Returns:
        Tuple of (trajectories_df, summary_df).  trajectories_df has one row per
        (unique_particle, frame) with columns x, y, z (pixels), spot_type, time.
        summary_df has one row per trajectory with n_frames, frame_start/end, mean xyz.

    Raises:
        FileNotFoundError: If no tracking_*.csv is found in result_dir.
    """
    csv_files = sorted(result_dir.glob('tracking_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No tracking_*.csv found in '{result_dir}'")
    tracking_df = pd.read_csv(csv_files[0])

    # Prefer the GUI's unique_particle (e.g. "0_1_2" = image_cell_particle).
    # Fall back to the plain integer particle column for older CSV formats.
    if 'unique_particle' in tracking_df.columns:
        id_col = 'unique_particle'
    else:
        tracking_df = tracking_df.rename(columns={'particle': 'unique_particle'})
        id_col = 'unique_particle'

    coordinate_cols = [id_col, 'frame', 'x', 'y', 'z']
    optional_cols   = ['spot_type', 'time', 'cluster_size', 'is_nuc']
    keep_cols       = coordinate_cols + [c for c in optional_cols if c in tracking_df.columns]
    trajectories_df = (
        tracking_df[keep_cols]
        .sort_values(['unique_particle', 'frame'])
        .reset_index(drop=True)
    )
    summary_df = (
        trajectories_df
        .groupby('unique_particle')
        .agg(
            n_frames    = ('frame', 'count'),
            frame_start = ('frame', 'min'),
            frame_end   = ('frame', 'max'),
            x_mean      = ('x', 'mean'),
            y_mean      = ('y', 'mean'),
            z_mean      = ('z', 'mean'),
        )
        .reset_index()
    )
    print(f"'{result_dir.name}': {len(summary_df)} trajectories, {len(trajectories_df)} total observations")
    return trajectories_df, summary_df


# ── Intensity Extraction from Trajectories ────────────────────────────────────


def extract_intensities_from_trajectories(
    trajectories_df: 'pandas.DataFrame',
    image_tzyxc: np.ndarray,
    spot_size: int = 5,
    fast_gaussian_fit: bool = True,
    snr_method: str = 'peak',
    use_max_projection: bool = False,
    image_label: str = 'raw',
) -> 'pandas.DataFrame':
    """Re-measure spot intensities at pre-computed trajectory coordinates.

    For each (frame, particle) in *trajectories_df*, slices the image volume to
    that time-point and calls ``mi.Intensity`` at the known (z, y, x) positions.
    No spot detection or linking is performed — only photometry.

    The function is designed to be called multiple times with different images
    (raw, photobleaching-corrected, drift-registered) or different parameter
    sets to build a comparison table.

    Args:
        trajectories_df: Output of ``extract_trajectories``.  Must contain
            columns ``unique_particle, frame, x, y, z``.
        image_tzyxc: 5-D image array with shape ``[T, Z, Y, X, C]``.
        spot_size: Disk diameter in pixels for the disk-doughnut photometry.
            The doughnut extends 3 px beyond the disk edge.  Try 3, 5, 7, 9, 11.
        fast_gaussian_fit: If True, PSF amplitude/sigma are estimated with fast
            moment-based (centre-of-mass) method.  If False, a full
            ``scipy.curve_fit`` 2-D Gaussian is used (slower, more accurate for
            dim spots).  Does not affect the disk-doughnut intensity values.
        snr_method: ``'peak'`` (max pixel − background mean) / background std, or
            ``'disk_doughnut'`` (disk mean − background mean) / background std.
        use_max_projection: If True, collapse the Z axis with max-projection
            before measuring intensity (mirrors the GUI's 2-D tracking mode).
        image_label: Short label stored in the ``image_source`` column so you
            can concatenate results from multiple images/runs.

    Returns:
        pandas.DataFrame with one row per (unique_particle, frame).  Columns:

        ``unique_particle, frame, x, y, z``           — original coordinates
        ``spot_int_ch_<c>``                            — background-subtracted intensity
        ``snr_ch_<c>``                                 — signal-to-noise ratio
        ``psf_amplitude_ch_<c>, psf_sigma_ch_<c>``    — PSF Gaussian fit results
        ``total_int_ch_<c>``                           — raw sum of disk pixels
        ``spot_size, fast_gaussian_fit, snr_method``   — parameters used
        ``image_source``                               — value of *image_label*
    """
    n_channels = image_tzyxc.shape[-1]
    rows = []
    for frame_idx, frame_group in trajectories_df.groupby('frame'):
        if frame_idx >= image_tzyxc.shape[0]:
            print(f"  [WARNING] frame {frame_idx} out of image range (T={image_tzyxc.shape[0]}); skipping.")
            continue
        image_frame = image_tzyxc[int(frame_idx)]
        coords_zyx  = frame_group[['z', 'y', 'x']].to_numpy().astype(float)
        (
            intensities, _intensities_std, intensities_snr,
            _bg_mean, _bg_std, psf_amplitudes, psf_sigmas, intensities_total,
        ) = mi.Intensity(
            original_image=image_frame,
            spot_size=spot_size,
            array_spot_location_z_y_x=coords_zyx,
            use_max_projection=use_max_projection,
            fast_gaussian_fit=fast_gaussian_fit,
            snr_method=snr_method,
        ).calculate_intensity()
        for i, (_, spot_row) in enumerate(frame_group.iterrows()):
            row = {
                'unique_particle': spot_row['unique_particle'],
                'frame': int(frame_idx),
                'x': spot_row['x'],
                'y': spot_row['y'],
                'z': spot_row['z'],
            }
            for c in range(n_channels):
                row[f'spot_int_ch_{c}']      = float(intensities[i, c])
                row[f'snr_ch_{c}']           = float(intensities_snr[i, c])
                row[f'psf_amplitude_ch_{c}'] = float(psf_amplitudes[i, c])
                row[f'psf_sigma_ch_{c}']     = float(psf_sigmas[i, c])
                row[f'total_int_ch_{c}']     = float(intensities_total[i, c])
            row['spot_size']         = spot_size
            row['fast_gaussian_fit'] = fast_gaussian_fit
            row['snr_method']        = snr_method
            row['image_source']      = image_label
            rows.append(row)
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(['unique_particle', 'frame']).reset_index(drop=True)
    print(
        f"[{image_label}] spot_size={spot_size}, fast_gaussian={fast_gaussian_fit}, "
        f"snr='{snr_method}' → {len(result_df)} rows, {n_channels} channels"
    )
    return result_df


# ── Photobleaching Correction ─────────────────────────────────────────────────

def load_and_apply_photobleaching_correction(
    image_tzyxc: np.ndarray,
    time_interval_s: float,
    image_for_correction: np.ndarray = None,
    show_plot: bool = False,
) -> tuple:
    """Apply photobleaching correction using the two-step approach from the MicroLive GUI.

    Mirrors app.py compute_photobleaching exactly:

    Step 1 – Fit I(t) = I0·exp(−k·t) per channel on the *raw* image to obtain
             per-channel decay parameters (k_ch, I0_ch).  This avoids absorbing
             registration artifacts into the fit.

    Step 2 – Apply the pre-computed params to *image_for_correction* (typically
             a drift-registered image if available, otherwise the raw image) via
             ``precalulated_list_decay_rates``.  Each channel is corrected by its
             own independent exponential: multiplying every frame by exp(k_ch * t).

    The decay_rates list is ordered as [k_ch0, I0_ch0, k_ch1, I0_ch1, ...].

    Args:
        image_tzyxc: Raw image array [T, Z, Y, X, C].  Decay parameters are
            always estimated from this array.
        time_interval_s: Seconds between frames (from LIF metadata).
        image_for_correction: Array to apply the correction to.  Defaults to
            image_tzyxc (no registration).  Pass a registered image if available.
        show_plot: If True, plot the decay fit and corrected intensity curves.

    Returns:
        Tuple (image_corrected, decay_rates, photobleaching_data):
            image_corrected     – ndarray [T, Z, Y, X, C], corrected.
            decay_rates         – list [k_ch0, I0_ch0, k_ch1, I0_ch1, ...].
            photobleaching_data – dict with full correction metadata.
    """
    if image_for_correction is None:
        image_for_correction = image_tzyxc
    # Step 1: fit decay parameters from raw image (avoids registration artefacts).
    raw_pb = mi.Photobleaching(
        image_TZYXC=image_tzyxc,
        mask_YX=None,
        show_plot=False,
        mode='entire_image',
        time_interval_seconds=time_interval_s,
    )
    decay_rates = raw_pb.calculate_photobleaching()
    n_channels = image_tzyxc.shape[-1]
    for ch in range(n_channels):
        k   = decay_rates[2 * ch]
        I0  = decay_rates[2 * ch + 1]
        print(f"  ch {ch}: k={k:.6f} s⁻¹, I0={I0:.1f}")
    # Step 2: apply pre-computed params to the correction target.
    correction_pb = mi.Photobleaching(
        image_TZYXC=image_for_correction,
        mask_YX=None,
        show_plot=show_plot,
        mode='entire_image',
        time_interval_seconds=time_interval_s,
        precalulated_list_decay_rates=decay_rates,
    )
    image_corrected, photobleaching_data = correction_pb.apply_photobleaching_correction()
    print(f"Corrected image shape: {image_corrected.shape}")
    return image_corrected, decay_rates, photobleaching_data


# ── Master Dataset Builder ────────────────────────────────────────────────────

def build_intensity_dataset(
    association_dict: dict,
    apply_photobleaching: bool = True,
    spot_size: int = 5,
    fast_gaussian_fit: bool = True,
    snr_method: str = 'peak',
    use_max_projection: bool = False,
    start_result_dir_id: int = 0,
) -> pd.DataFrame:
    """Build a master intensity DataFrame from all processed images.

    Iterates every LIF file → every image → every result folder in
    *association_dict* (output of ``associate_lif_to_results``).  For each
    result folder a ``result_dir_id`` is assigned (sequential integer, starting
    at 0), and each particle receives a ``global_particle_id`` string of the
    form ``"{result_dir_id}_{unique_particle}"`` that is unique across the full
    dataset.

    Args:
        association_dict: Output of ``associate_lif_to_results``.
        apply_photobleaching: If True, apply two-step photobleaching correction
            before intensity extraction.
        spot_size: Disk diameter in pixels for the disk-doughnut photometry.
        fast_gaussian_fit: If True, use fast moment-based PSF estimation.
        snr_method: ``'peak'`` or ``'disk_doughnut'``.
        use_max_projection: If True, collapse Z before measuring intensity.

    Returns:
        pandas.DataFrame with one row per (global_particle_id, frame).
        Key columns:
            global_particle_id, result_dir_id, unique_particle,
            lif_file, image_name, result_dir,
            time_interval_s, pixel_xy_um, pixel_z_um,
            frame, x, y, z,
            spot_int_ch_<c>, snr_ch_<c>, psf_amplitude_ch_<c>,
            psf_sigma_ch_<c>, total_int_ch_<c>,
            spot_size, fast_gaussian_fit, snr_method, image_source.
    """
    # Flush accumulated DataFrames to a temp CSV every N images processed
    # so all_frames never holds the whole dataset in RAM at once.
    FLUSH_EVERY = 3

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv', prefix='_build_intensity_')
    os.close(tmp_fd)
    tmp_csv = Path(tmp_path)
    header_written = False
    flush_counter  = 0

    def _flush(frames, path, written_header):
        if not frames:
            return written_header
        chunk = pd.concat(frames, ignore_index=True)
        chunk.to_csv(path, mode='a', header=not written_header, index=False)
        return True   # header now written

    all_frames   = []
    result_dir_id = start_result_dir_id

    for lif_file, images in association_dict.items():
        for image_name, entry in images.items():
            image_raw       = entry['image']
            time_interval_s = entry['time_interval_s']

            if apply_photobleaching:
                print(f"\n[Photobleaching] {lif_file} / {image_name}")
                image_to_use, _, _ = load_and_apply_photobleaching_correction(
                    image_raw, time_interval_s
                )
                image_source = 'corrected'
            else:
                image_to_use = image_raw
                image_source = 'raw'

            for result_dir in entry['result_dirs']:
                print(f"\n[result_dir_id={result_dir_id}] {result_dir.name}")
                trajectories_df, _ = extract_trajectories(result_dir)

                intensities_df = extract_intensities_from_trajectories(
                    trajectories_df,
                    image_to_use,
                    spot_size=spot_size,
                    fast_gaussian_fit=fast_gaussian_fit,
                    snr_method=snr_method,
                    use_max_projection=use_max_projection,
                    image_label=image_source,
                )

                # Assign global identity columns.
                intensities_df['result_dir_id']      = result_dir_id
                intensities_df['global_particle_id'] = (
                    str(result_dir_id) + '_'
                    + intensities_df['unique_particle'].astype(str)
                )

                # Attach provenance and image metadata.
                intensities_df['lif_file']        = lif_file
                intensities_df['image_name']      = image_name
                intensities_df['result_dir']      = result_dir.name
                intensities_df['time_interval_s'] = time_interval_s
                intensities_df['pixel_xy_um']     = entry['pixel_xy_um']
                intensities_df['pixel_z_um']      = entry['pixel_z_um']

                all_frames.append(intensities_df)
                result_dir_id += 1

            # ── Fix 1: free the raw image AND the (possibly corrected) working copy ──
            del image_to_use
            del entry['image']
            image_raw = None   # drop local reference too

            flush_counter += 1

            # ── Fix 2: periodically flush accumulated frames to disk ──────────
            if flush_counter >= FLUSH_EVERY:
                header_written = _flush(all_frames, tmp_csv, header_written)
                all_frames = []
                flush_counter = 0
                gc.collect()
                print(f"  [memory] flushed batch to {tmp_csv.name}")

    # Final flush for any remaining frames.
    header_written = _flush(all_frames, tmp_csv, header_written)
    all_frames = []
    gc.collect()

    if not header_written:
        tmp_csv.unlink(missing_ok=True)
        print("No data collected — check association_dict.")
        return pd.DataFrame()

    # Read back the full dataset from the temp file (one pass, avoids holding
    # every batch simultaneously in RAM).
    print(f"\n[memory] reading master_df from temp CSV ({tmp_csv.stat().st_size // 1024} KB)…")
    master_df = pd.read_csv(tmp_csv)
    tmp_csv.unlink(missing_ok=True)

    # Reorder so identity columns come first.
    first_cols = [
        'global_particle_id', 'result_dir_id', 'unique_particle',
        'lif_file', 'image_name', 'result_dir',
        'time_interval_s', 'pixel_xy_um', 'pixel_z_um',
        'frame', 'x', 'y', 'z',
    ]
    remaining_cols = [c for c in master_df.columns if c not in first_cols]
    master_df = master_df[first_cols + remaining_cols]

    print(
        f"\n{'='*60}\n"
        f" master_df: {len(master_df)} rows, "
        f"{master_df['global_particle_id'].nunique()} unique particles, "
        f"{result_dir_id} result folders\n"
        f"{'='*60}\n"
    )
    return master_df



def _quality_mask_for_trajectory_array(
    intensity_array: np.ndarray,
    min_percentage_data_in_trajectory: float = 0.25,
    max_missing_frames: int = 2,
) -> np.ndarray:
    """Return the row mask used before trajectory shifting for single-channel ACF prep.

    Mirrors the row-filtering logic in ``mi.Utilities().shift_trajectories(...)`` so
    we can keep trajectory/image provenance aligned with the final array rows.
    """
    if intensity_array.ndim != 2:
        raise ValueError("intensity_array must be 2D (trajectories x timepoints)")
    if intensity_array.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if not (0.0 <= min_percentage_data_in_trajectory <= 1.0):
        raise ValueError("min_percentage_data_in_trajectory must be between 0 and 1")

    n_time = intensity_array.shape[1]
    max_nans_allowed = int(round(n_time * (1 - min_percentage_data_in_trajectory)))
    row_nan_counts = np.isnan(intensity_array).sum(axis=1)
    mask_relative = row_nan_counts <= max_nans_allowed

    if max_missing_frames is None:
        return mask_relative

    def count_internal_nans(row: np.ndarray) -> int:
        valid = np.where(~np.isnan(row))[0]
        if valid.size == 0:
            return int(1e9)
        first, last = valid[0], valid[-1]
        return int(np.isnan(row[first:last + 1]).sum())

    mask_absolute = np.array(
        [count_internal_nans(row) <= max_missing_frames for row in intensity_array],
        dtype=bool,
    )
    return mask_relative & mask_absolute


def build_correlation_input_from_reprocessed_df(
    master_df: pd.DataFrame,
    *,
    channel_index: int,
    condition_name: str = None,
    selected_field_base: str = 'spot_int_ch_',
    snr_field_base: str = 'snr_ch_',
    min_percentage_data_in_trajectory: float = 0.25,
    max_missing_frames: int = 2,
    maximum_columns: int = 360,
    min_snr: float = 0.5,
    smooth_window: int = 1,
    verbose: bool = True,
) -> dict:
    """Convert a reprocessed master dataframe (one condition) into a Correlation-ready array.

    The returned row order is stable and accompanied by per-trajectory provenance,
    including an ``image_uid`` key where one ``result_dir`` folder is treated as one
    image/cell unit for ACF counting.
    """
    if master_df is None or master_df.empty:
        raise ValueError("master_df is empty")

    required_cols = {'result_dir_id', 'unique_particle', 'frame'}
    missing = required_cols - set(master_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in master_df: {sorted(missing)}")

    if condition_name is None and 'condition' in master_df.columns:
        condition_values = master_df['condition'].dropna().unique()
        if len(condition_values) == 1:
            condition_name = str(condition_values[0])
        elif len(condition_values) > 1:
            raise ValueError(
                "master_df contains multiple conditions. Pass a single-condition dataframe "
                "or specify condition_name after subsetting."
            )
    elif condition_name is not None and 'condition' in master_df.columns:
        condition_values = master_df['condition'].dropna().astype(str).unique()
        if len(condition_values) > 0 and any(v != str(condition_name) for v in condition_values):
            raise ValueError(
                "master_df contains rows from a different condition than condition_name. "
                "Pass a single-condition dataframe to build_correlation_input_from_reprocessed_df."
            )

    selected_field = f"{selected_field_base}{channel_index}"
    snr_field = f"{snr_field_base}{channel_index}"
    if selected_field not in master_df.columns:
        raise ValueError(f"Selected intensity field not found: {selected_field}")
    snr_available = snr_field in master_df.columns

    # Build stable trajectory/image identities compatible with per-condition ACF runs.
    df = master_df.copy()
    result_dir_ids = df['result_dir_id']
    if result_dir_ids.isna().any():
        raise ValueError("result_dir_id contains NaN values")

    result_dir_ids_int = pd.to_numeric(result_dir_ids, errors='raise').astype(int)
    if condition_name is None:
        condition_prefix = "NA"
    else:
        condition_prefix = str(condition_name)

    df['trajectory_uid'] = (
        condition_prefix + "__r" + result_dir_ids_int.astype(str) + "__p" + df['unique_particle'].astype(str)
    )
    df['image_uid'] = condition_prefix + "__r" + result_dir_ids_int.astype(str)
    df['result_dir_id'] = result_dir_ids_int

    dup_mask = df.duplicated(subset=['trajectory_uid', 'frame'], keep=False)
    if dup_mask.any():
        example = (
            df.loc[dup_mask, ['trajectory_uid', 'frame']]
            .drop_duplicates()
            .head(5)
            .to_dict(orient='records')
        )
        raise ValueError(
            "Duplicate trajectory-frame rows found in reprocessed dataframe. "
            "This would corrupt ACF array construction. "
            f"Examples: {example}"
        )

    # Preserve trajectory order explicitly (sorted for deterministic CSV/array output).
    sort_cols = ['trajectory_uid', 'frame']
    df = df.sort_values(sort_cols).reset_index(drop=True)
    trajectory_meta = (
        df.drop_duplicates(subset=['trajectory_uid'])[
            ['trajectory_uid', 'image_uid', 'result_dir_id']
            + [c for c in ['result_dir', 'image_name', 'lif_file', 'condition', 'time_interval_s'] if c in df.columns]
        ]
        .sort_values('trajectory_uid')
        .reset_index(drop=True)
    )
    trajectory_ids_order = trajectory_meta['trajectory_uid'].astype(str).to_numpy()
    image_ids_order = trajectory_meta['image_uid'].astype(str).to_numpy()

    # Build a minimal dataframe so Utilities.df_trajectories_to_array uses our stable IDs.
    cols_for_array = ['trajectory_uid', 'frame', selected_field]
    if snr_available:
        cols_for_array.append(snr_field)
    array_df = df[cols_for_array].rename(columns={'trajectory_uid': 'particle'})

    utilities = mi.Utilities()
    intensity_array = utilities.df_trajectories_to_array(
        dataframe=array_df,
        selected_field=selected_field,
        fill_value='nans',
    )
    if intensity_array.shape[0] != trajectory_ids_order.shape[0]:
        raise RuntimeError(
            "Trajectory metadata length does not match intensity array row count "
            f"({trajectory_ids_order.shape[0]} vs {intensity_array.shape[0]})."
        )
    if intensity_array.shape[0] == 0:
        raise ValueError(f"No trajectories found for field '{selected_field}'")

    stats = {
        'channel_index': int(channel_index),
        'selected_field': selected_field,
        'snr_field': snr_field,
        'n_rows_input': int(len(df)),
        'n_trajectories_raw': int(intensity_array.shape[0]),
        'n_images_raw': int(np.unique(image_ids_order).size),
    }

    # Per-trajectory mean SNR filter (same semantics as extract_intensity_and_shift_data).
    if snr_available:
        snr_array = utilities.df_trajectories_to_array(
            dataframe=array_df,
            selected_field=snr_field,
            fill_value='nans',
        )
        mean_snr = np.nanmean(snr_array, axis=1)
        snr_mask = np.isfinite(mean_snr) & (mean_snr >= float(min_snr))
        stats['n_trajectories_after_snr'] = int(np.sum(snr_mask))
        stats['n_removed_snr'] = int(np.sum(~snr_mask))
    else:
        if verbose:
            print(f"[ACF adapter] WARNING: SNR field '{snr_field}' not found, skipping SNR filter")
        snr_mask = np.ones(intensity_array.shape[0], dtype=bool)
        mean_snr = np.full(intensity_array.shape[0], np.nan)
        stats['n_trajectories_after_snr'] = int(intensity_array.shape[0])
        stats['n_removed_snr'] = 0

    intensity_snr = intensity_array[snr_mask, :]
    trajectory_ids_snr = trajectory_ids_order[snr_mask]
    image_ids_snr = image_ids_order[snr_mask]
    mean_snr_snr = mean_snr[snr_mask]
    trajectory_meta = trajectory_meta.loc[snr_mask].reset_index(drop=True)
    trajectory_meta['mean_snr'] = mean_snr_snr

    if intensity_snr.shape[0] == 0:
        raise ValueError(
            f"All trajectories failed SNR filtering for {selected_field} (min_snr={min_snr})."
        )

    quality_mask = _quality_mask_for_trajectory_array(
        intensity_snr,
        min_percentage_data_in_trajectory=min_percentage_data_in_trajectory,
        max_missing_frames=max_missing_frames,
    )
    intensity_qc = intensity_snr[quality_mask, :]
    trajectory_ids_qc = trajectory_ids_snr[quality_mask]
    image_ids_qc = image_ids_snr[quality_mask]
    trajectory_meta = trajectory_meta.loc[quality_mask].reset_index(drop=True)
    stats['n_trajectories_after_qc'] = int(intensity_qc.shape[0])
    stats['n_removed_qc'] = int(np.sum(~quality_mask))

    if intensity_qc.shape[0] == 0:
        raise ValueError(
            "All trajectories were filtered out by trajectory quality thresholds "
            f"(min_percentage_data_in_trajectory={min_percentage_data_in_trajectory}, "
            f"max_missing_frames={max_missing_frames})."
        )

    # Left-align and trim exactly like the single-channel shift_trajectories path.
    intensity_shifted = utilities.shift_initial_nans(intensity_qc)
    last_valid_col = utilities.find_last_valid_column(intensity_shifted)
    if last_valid_col < 0:
        raise ValueError("No valid data remain after trajectory shifting")
    intensity_shifted = intensity_shifted[:, :last_valid_col + 1]

    if smooth_window > 1:
        intensity_ffill = utilities.forward_fill_nan_2d(intensity_shifted)
        intensity_shifted = uniform_filter1d(
            intensity_ffill, size=int(smooth_window), axis=1, mode='nearest'
        )

    n_cols_before_pad = int(intensity_shifted.shape[1])
    if maximum_columns is not None:
        maximum_columns = int(maximum_columns)
        if n_cols_before_pad < maximum_columns:
            intensity_shifted = np.pad(
                intensity_shifted,
                ((0, 0), (0, maximum_columns - n_cols_before_pad)),
                mode='constant',
                constant_values=np.nan,
            )
        elif n_cols_before_pad > maximum_columns:
            intensity_shifted = intensity_shifted[:, :maximum_columns]

    if 'time_interval_s' in trajectory_meta.columns:
        time_intervals = trajectory_meta['time_interval_s'].dropna().astype(float).unique()
        if len(time_intervals) > 1:
            raise ValueError(
                "Mixed time_interval_s detected within one condition dataframe. "
                "Split by time interval before ACF calculation."
            )
        time_interval_s = float(time_intervals[0]) if len(time_intervals) == 1 else None
    else:
        time_interval_s = None

    stats.update({
        'n_trajectories_final_input': int(intensity_shifted.shape[0]),
        'n_images_final_input': int(np.unique(image_ids_qc).size),
        'n_timepoints_before_pad': n_cols_before_pad,
        'n_timepoints_final': int(intensity_shifted.shape[1]),
    })

    if verbose:
        cond_label = condition_name if condition_name is not None else "<unknown>"
        print(
            f"[ACF adapter] condition={cond_label} ch{channel_index}: "
            f"{stats['n_trajectories_raw']} raw traj -> "
            f"{stats['n_trajectories_after_snr']} after SNR -> "
            f"{stats['n_trajectories_after_qc']} after QC; "
            f"final array shape={intensity_shifted.shape}"
        )

    return {
        'primary_data': intensity_shifted,
        'trajectory_ids': trajectory_ids_qc,
        'image_ids': image_ids_qc,
        'trajectory_metadata': trajectory_meta.reset_index(drop=True),
        'total_number_of_spots': int(intensity_shifted.shape[0]),
        'total_number_of_cells': int(np.unique(image_ids_qc).size),
        'time_interval_s': time_interval_s,
        'selected_field': selected_field,
        'snr_field': snr_field,
        'channel_index': int(channel_index),
        'condition': condition_name,
        'adapter_stats': stats,
    }
