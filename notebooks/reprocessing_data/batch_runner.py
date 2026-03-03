"""Batch parameter sweep — runs the full pipeline for each combination.

Runs all 4 parameter sets sequentially and saves each to its own subfolder:

    results/sz5_full_peak/   master_intensity_dataset_sz5_full_peak.csv
    results/sz5_fast_peak/   master_intensity_dataset_sz5_fast_peak.csv
    results/sz7_full_peak/   master_intensity_dataset_sz7_full_peak.csv
    results/sz7_fast_peak/   master_intensity_dataset_sz7_fast_peak.csv

Usage
-----
    python batch_runner.py
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

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

import pandas as pd

# Import shared config and functions from runner.py.
from runner import (
    OUTPUT_DIR,
    get_acf_config_dict,
    run_pipeline,
)
import runner  # needed to override module-level globals

# Import plotting functions from plot_psf.py.
from plot_psf import (
    plot_acf_comparison,
    plot_psf_amplitude_vs_sigma,
    plot_intensity_distributions,
    CONDITION_RENAME,
    MIN_SNR,
)


# ── Parameter grid ─────────────────────────────────────────────────────────────

PARAM_GRID = [
    # Run 1
    {'spot_size': 5, 'fast_gaussian_fit': True, 'snr_method': 'peak'},
    # # Run 2
    # {'spot_size': 5, 'fast_gaussian_fit': True,  'snr_method': 'disk_doughnut'},
    # # Run 3
    # {'spot_size': 7, 'fast_gaussian_fit': False, 'snr_method': 'peak'},
    # # Run 4
    # {'spot_size': 7, 'fast_gaussian_fit': True,  'snr_method': 'peak'},
    # # Run 5
    # {'spot_size': 5, 'fast_gaussian_fit': False, 'snr_method': 'disk_doughnut'},
    # # Run 6
    # {'spot_size': 7, 'fast_gaussian_fit': False, 'snr_method': 'disk_doughnut'},
]



# ── Sweep ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n = len(PARAM_GRID)
    for i, params in enumerate(PARAM_GRID, 1):
        spot_size        = params['spot_size']
        fast_gaussian    = params['fast_gaussian_fit']
        snr_method       = params['snr_method']

        _fit_tag  = 'fast' if fast_gaussian else 'full'
        _snr_tag  = snr_method.replace('_', '')
        param_tag = f'sz{spot_size}_{_fit_tag}_{_snr_tag}'
        run_dir   = OUTPUT_DIR / param_tag

        print(f"\n{'#'*65}")
        print(f"  Run {i}/{n}  —  {param_tag}")
        print(f"  spot_size={spot_size}, fast_gaussian={fast_gaussian}, snr='{snr_method}'")
        print(f"  Output: {run_dir}")
        print(f"{'#'*65}\n")

        # Override the relevant globals in the runner module so run_pipeline
        # picks them up without any signature changes.
        runner.SPOT_SIZE_PX      = spot_size
        runner.FAST_GAUSSIAN_FIT = fast_gaussian
        runner.SNR_METHOD        = snr_method

        run_dir.mkdir(parents=True, exist_ok=True)
        output_csv = run_dir / f'master_intensity_dataset_{param_tag}.csv'

        # Write a params.json so the folder is self-documenting.
        params_record = {
            'param_tag':        param_tag,
            'spot_size_px':     spot_size,
            'fast_gaussian_fit': fast_gaussian,
            'snr_method':       snr_method,
            'output_csv':       str(output_csv),
            'acf_config':       get_acf_config_dict(),
        }
        with open(run_dir / 'params.json', 'w') as f:
            json.dump(params_record, f, indent=2)

        master_df = run_pipeline(output_csv=output_csv)
        acf_summary_path = master_df.attrs.get('acf_summary_path')
        if acf_summary_path:
            print(f"ACF summary CSV: {acf_summary_path}")

        # Apply any condition renames so colors resolve correctly.
        if CONDITION_RENAME:
            master_df['condition'] = master_df['condition'].replace(CONDITION_RENAME)

        # Generate all plots — filenames carry param_tag for easy identification.
        plots_dir = run_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        for ch in [0, 1]:
            plot_psf_amplitude_vs_sigma(
                master_df, channel_index=ch, output_dir=plots_dir,
                save_name=f'psf_amplitude_vs_sigma_ch{ch}_{param_tag}',
            )
            plot_intensity_distributions(
                master_df, field='snr_ch_', channel_index=ch,
                min_snr=MIN_SNR, x_label='SNR', output_dir=plots_dir,
                save_name=f'snr_ch{ch}_{param_tag}',
            )
            plot_intensity_distributions(
                master_df, field='spot_int_ch_', channel_index=ch,
                min_snr=MIN_SNR, x_label='Spot Intensity (a.u.)', output_dir=plots_dir,
                save_name=f'spot_int_ch{ch}_{param_tag}',
            )

        # ACF overlay plot (one per channel, all conditions on one figure).
        acf_results = master_df.attrs.get('acf_results', [])
        plot_acf_comparison(acf_results, param_tag=param_tag, output_dir=plots_dir)

        del master_df
        gc.collect()


    print(f"\n{'='*65}")
    print(f"  All {n} runs complete.")
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
