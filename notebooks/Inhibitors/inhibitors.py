# !/usr/bin/env python3
from pathlib import Path
current_dir = Path().resolve()
from microlive.imports import *
from microlive import microscopy as mi
import tasep_models as tm
from tasep_models import *


plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
    })

def calculate_number_of_particles_per_frame(particle_counts_per_frame, inhibitor_frame_index):
    """
    Normalize the number of particles per frame to the average before treatment.

    All frames are divided by the average particle count before treatment.
    If that average is zero, returns an array of zeros.

    Parameters:
        particle_counts_per_frame (np.ndarray): 1D array of particle counts per frame.
        inhibitor_frame_index (int): The frame index at which treatment starts.

    Returns:
        tuple: (normalized_particles, average_particles_before_treatment)
            - normalized_particles: Particle counts normalized to pre-treatment average
            - average_particles_before_treatment: The average used for normalization
    """
    # Compute the average particle count before treatment
    pre_counts = particle_counts_per_frame[:inhibitor_frame_index]
    average_particles_before_treatment = pre_counts.mean()

    # Normalize all frames by the pre-treatment average
    if average_particles_before_treatment == 0:
        normalized_particles = np.zeros_like(particle_counts_per_frame, dtype=float)
    else:
        normalized_particles = particle_counts_per_frame / average_particles_before_treatment

    return normalized_particles, average_particles_before_treatment

def calculate_intensity(particle_counts_per_frame, sum_intensities_per_frame, inhibitor_frame_index, normalization_method='mean', percentile_range=(5, 95)):
    """
    Normalize the intensity per frame.

    For frames before the treatment, each frame's intensity is given by
    sum_intensities / particle_counts. If the particle count is zero in a frame,
    the normalized intensity is set to zero.

    For frames after the treatment, the sum intensities are divided by the average
    particle count before treatment. If that average is zero, zeros are returned for
    all frames after treatment.

    Parameters:
        particle_counts_per_frame (np.ndarray): 1D array of particle counts per frame.
        sum_intensities_per_frame (np.ndarray): 1D array of sum intensities per frame.
        inhibitor_frame_index (int): The frame index at which treatment starts.
        normalization_method (str): 'mean' (default) divides by pre-treatment mean,
            'minmax' scales the trajectory to [0, 1] range,
            'percentile' scales using percentile bounds (robust to outliers).
            None: no normalization.
        percentile_range (tuple): Percentile bounds for 'percentile' method.
            Default (5, 95). Use (1, 99) for wider range.

    Returns:
        tuple: (normalized_intensities, raw_avg_intensities, avg_particles_before_treatment)
    """
    # Compute the average particle count before treatment.
    pre_counts = particle_counts_per_frame[:inhibitor_frame_index]
    average_particles_before_treatment = pre_counts.mean()

    # For frames before treatment, avoid division by zero:
    pre_intensities = sum_intensities_per_frame[:inhibitor_frame_index]
    intensity_before_treatment = np.where(pre_counts == 0, 0, pre_intensities / pre_counts)
    # For frames after treatment, if the average is zero then return zeros.
    post_intensities = sum_intensities_per_frame[inhibitor_frame_index:]
    if average_particles_before_treatment == 0:
        intensity_after_treatment = np.zeros_like(post_intensities)
    else:
        intensity_after_treatment = post_intensities / average_particles_before_treatment
    # Combine the two segments and return the result.
    average_intensity_with_respect_number_particles = np.concatenate([intensity_before_treatment, intensity_after_treatment])

    # Apply normalization
    if normalization_method == 'minmax':
        val_min = average_intensity_with_respect_number_particles.min()
        val_max = average_intensity_with_respect_number_particles.max()
        if val_max - val_min == 0:
            intensities_normalized_before_treatment_intensity = np.zeros_like(average_intensity_with_respect_number_particles)
        else:
            intensities_normalized_before_treatment_intensity = (average_intensity_with_respect_number_particles - val_min) / (val_max - val_min)
    elif normalization_method == 'percentile':
        val_low = np.percentile(average_intensity_with_respect_number_particles, percentile_range[0])
        val_high = np.percentile(average_intensity_with_respect_number_particles, percentile_range[1])
        if val_high - val_low == 0:
            intensities_normalized_before_treatment_intensity = np.zeros_like(average_intensity_with_respect_number_particles)
        else:
            intensities_normalized_before_treatment_intensity = (average_intensity_with_respect_number_particles - val_low) / (val_high - val_low)
    elif normalization_method == 'mean':
        mean_before_treatment = average_intensity_with_respect_number_particles[:inhibitor_frame_index].mean()
        intensities_normalized_before_treatment_intensity = average_intensity_with_respect_number_particles / mean_before_treatment
    elif normalization_method is None:
        intensities_normalized_before_treatment_intensity = average_intensity_with_respect_number_particles

    return intensities_normalized_before_treatment_intensity, average_intensity_with_respect_number_particles, average_particles_before_treatment


# ── Model definitions ────────────────────────────────────────────────────────

def _linear_model(x, a, b):
    """Linear decay: y = a*x + b"""
    return a * x + b


def _exponential_model(x, A, tau, C):
    """Exponential decay: y = A * exp(-x/τ) + C"""
    return A * np.exp(-x / tau) + C


def _heaviside_model(x, A, T, C):
    """Larson 2011 Heaviside-ramp: y = A*(1 - x/T)*H(T - x) + C

    Linear decay from A+C to C, then flat at C for x > T.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= T, A * (1.0 - x / T) + C, C)


# ── Fitting function ─────────────────────────────────────────────────────────

def fit_inhibitor_model(x_data, y_data, err_data=None, model='exponential',
                        fit_start_idx=None, fit_end_idx=None,
                        runoff_fraction=0.95):
    """Fit inhibitor run-off data to a decay model.

    Parameters
    ----------
    x_data : np.ndarray
        Time array (e.g., time in minutes, recentered so 0 = inhibitor).
    y_data : np.ndarray
        Mean intensity trajectory (1D).
    err_data : np.ndarray or None
        Per-point measurement uncertainty (e.g., SEM from individual cells).
        Same length as y_data. When provided, curve_fit performs weighted
        least-squares and χ² is computed as Σ[(y-f)²/σ²].
        When None, unweighted fitting is used.
    model : str
        One of 'linear', 'exponential', 'heaviside'.
    fit_start_idx : int or None
        Index into x_data/y_data for the start of the fitting range.
        Defaults to 0 (start of the array).
    fit_end_idx : int or None
        Index into x_data/y_data for the end of the fitting range (inclusive).
        Defaults to len(x_data) - 1 (end of the array).
    runoff_fraction : float
        Fraction of total decay used to define run-off time (default 0.95).

    Returns
    -------
    dict or None
        On success, a dictionary with:
            'model'       : str   – model name
            'params'      : dict  – fitted parameter values
            'fitted_curve': np.ndarray – fitted y-values over the FULL x_data range
            't_half'      : float – half-time (time for 50 % decay)
            't_runoff'    : float – run-off time (time for `runoff_fraction` decay)
            'R2'          : float – coefficient of determination
            'RSS'         : float – residual sum of squares
            'chi2'        : float – chi-squared (weighted if err_data provided)
            'chi2_reduced': float – χ²/dof
            'dof'         : int   – degrees of freedom (n_data − n_params)
        Returns None if fitting fails.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if err_data is not None:
        err_data = np.asarray(err_data, dtype=float)

    # Default range: full array
    i0 = fit_start_idx if fit_start_idx is not None else 0
    i1 = (fit_end_idx + 1) if fit_end_idx is not None else len(x_data)

    x_fit = x_data[i0:i1]
    y_fit = y_data[i0:i1]
    err_fit = err_data[i0:i1] if err_data is not None else None

    # Remove NaN values (e.g., from artifact removal at inhibitor frame)
    valid = np.isfinite(x_fit) & np.isfinite(y_fit)
    if err_fit is not None:
        valid = valid & np.isfinite(err_fit)
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    if err_fit is not None:
        err_fit = err_fit[valid]
        # Replace zero uncertainties with a small value to avoid division by zero
        err_fit = np.where(err_fit == 0, np.min(err_fit[err_fit > 0]) * 0.1 if np.any(err_fit > 0) else 1e-10, err_fit)
        sigma_kwarg = {'sigma': err_fit, 'absolute_sigma': True}
    else:
        sigma_kwarg = {}

    if len(x_fit) < 3:
        print('fit_inhibitor_model: not enough data points to fit.')
        return None

    model = model.lower().strip()

    try:
        if model == 'linear':
            # y = a*x + b
            popt, pcov = curve_fit(_linear_model, x_fit, y_fit, **sigma_kwarg)
            a, b = popt
            perr = np.sqrt(np.diag(pcov))
            fitted_full = _linear_model(x_data, *popt)

            # Derived quantities
            # Estimate actual baseline from last 20% of data
            tail = max(1, len(y_fit) // 5)
            Iss = float(np.mean(y_fit[-tail:]))
            I0 = b  # intensity at x = 0 (fitted intercept)
            if a != 0 and I0 != Iss:
                t_half = (I0 - (I0 + Iss) / 2.0) / (-a)   # when y = midpoint
                t_runoff = (I0 - Iss) / (-a)                # when y = baseline
            else:
                t_half = np.inf
                t_runoff = np.inf

            params = {'a (slope)': a, 'b (intercept)': b,
                      'Iss (baseline)': Iss,
                      'a_err': perr[0], 'b_err': perr[1]}

        elif model == 'exponential':
            # y = A * exp(-x/τ) + C
            # Initial guesses (robust to negative/positive baselines)
            tail = max(1, len(y_fit) // 5)
            C0 = float(np.mean(y_fit[-tail:]))   # baseline from last 20%
            A0 = max(float(y_fit[0]) - C0, 1e-8)
            tau0 = (x_fit[-1] - x_fit[0]) / 3.0
            popt, pcov = curve_fit(
                _exponential_model, x_fit, y_fit,
                p0=[A0, tau0, C0],
                bounds=([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=50000,
                **sigma_kwarg,
            )
            A, tau, C = popt
            perr = np.sqrt(np.diag(pcov))
            fitted_full = _exponential_model(x_data, *popt)

            # Derived quantities
            t_half = tau * np.log(2)
            t_runoff = -tau * np.log(1.0 - runoff_fraction)

            params = {'A (amplitude)': A, 'tau (time constant)': tau,
                      'C (baseline)': C,
                      'A_err': perr[0], 'tau_err': perr[1], 'C_err': perr[2]}

        elif model == 'heaviside':
            # y = A * (1 - x/T) * H(T - x) + C   (Larson 2011)
            # Initial guesses (robust to negative/positive baselines)
            tail = max(1, len(y_fit) // 5)
            C0 = float(np.mean(y_fit[-tail:]))           # baseline from last 20%
            A0 = max(float(y_fit[0]) - C0, 1e-8)         # amplitude above baseline
            # Smart T guess: find where data first drops to baseline level
            crossings = np.where(y_fit <= C0)[0]
            if len(crossings) > 0:
                T0 = float(x_fit[crossings[0]] - x_fit[0])
            else:
                T0 = float((x_fit[-1] - x_fit[0]) / 2.0)  # fallback: half the range
            T0 = max(T0, 1.0)  # at least 1 minute
            popt, pcov = curve_fit(
                _heaviside_model, x_fit, y_fit,
                p0=[A0, T0, C0],
                bounds=([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=50000,
                **sigma_kwarg,
            )
            A, T, C = popt
            perr = np.sqrt(np.diag(pcov))
            fitted_full = _heaviside_model(x_data, *popt)

            # Derived quantities
            t_half = T / 2.0
            t_runoff = T  # T IS the run-off time for this model

            params = {'A (amplitude)': A, 'T (dwell/run-off time)': T,
                      'C (baseline)': C,
                      'A_err': perr[0], 'T_err': perr[1], 'C_err': perr[2]}

        else:
            print(f'fit_inhibitor_model: unknown model "{model}". '
                  f'Choose from: linear, exponential, heaviside.')
            return None

        # Goodness-of-fit metrics (computed on the fitting window only)
        n_params = len(popt)
        n_data = len(y_fit)
        dof = n_data - n_params
        y_pred_fit = fitted_full[i0:i1][valid]
        residuals = y_fit - y_pred_fit

        # Unweighted sums of squares (always computed)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # Chi-squared: weighted if err_data provided, unweighted otherwise
        if err_fit is not None:
            chi2 = float(np.sum((residuals / err_fit) ** 2))
        else:
            chi2 = ss_res  # equivalent to unweighted χ²
        chi2_red = chi2 / dof if dof > 0 else np.nan

        result = {
            'model': model,
            'params': params,
            'fitted_curve': fitted_full,
            't_half': t_half,
            't_runoff': t_runoff,
            'runoff_fraction': runoff_fraction,
            'chi2_reduced': chi2_red,
            'dof': dof,
        }
        return result

    except Exception as e:
        print(f'fit_inhibitor_model ({model}): fitting failed – {e}')
        return None


def plot_inhibitor(full_frames, intensities_normalized, inhibitor_frame_index,
                   results_folder=None, plot_name='HT', list_param=None,
                   responding_indices=None, figsize=(6, 3), time_array_min=None,
                   mean_intensity_ssa_inh=None, err_intensity_ssa_inh=None,
                   use_sem=True, show_individual_trajectories=True,
                   ylims=(0, 1.5), xlims=None,
                   y_label='Norm. Intensity',
                   treatment_label='Inhibitor', show_treatment_line=True,
                   # ── New fitting parameters ──
                   fit_model=None, fit_start_idx=None, fit_end_idx=None,
                   show_fit=True, show_runoff_time=True,
                   colors=None,
                   runoff_fraction=0.95):
    """Plot inhibitor run-off data with optional model fit.

    Parameters
    ----------
    full_frames : np.ndarray
        Time array (e.g., minutes, recentered so 0 = inhibitor application).
    intensities_normalized : np.ndarray
        2D array (n_cells × n_frames) of normalized intensities.
    inhibitor_frame_index : int
        Frame index at which treatment starts.
    fit_model : str or None
        Model to fit: 'linear', 'exponential', 'heaviside', or None (no fit).
    fit_start_idx : int or None
        Start index for fitting range. Defaults to inhibitor_frame_index (t=0).
    fit_end_idx   : int or None
        End index for fitting range (inclusive). Defaults to last frame.
    show_fit : bool
        If True (default), overlay the fitted curve on the plot.
    show_runoff_time : bool
        If True (default), draw vertical lines for t½ and τ_runoff.
    runoff_fraction : float
        Fraction of total decay for run-off time definition (default 0.95).
    colors : list of str or None
        List of colors for the trajectories. If None, default colors are used.

    Returns
    -------
    dict or None
        The fit result dictionary from fit_inhibitor_model, or None.
    """
    if colors is None:
        colors = [ 'blue']

    # if colors is not a list, make it a list
    if not isinstance(colors, list):
        colors = [colors]

    if results_folder is None:
        results_folder = Path(current_dir).joinpath('results_HT')
        results_folder.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    if intensities_normalized is None or len(intensities_normalized) == 0:
        print('No data to plot.')
        return None

    if responding_indices is None:
        responding_indices = list(range(len(intensities_normalized)))

    # Individual trajectories
    if show_individual_trajectories:
        if responding_indices:
            for i in responding_indices:
                ax.plot(full_frames, intensities_normalized[i],
                        linestyle='-', color='dimgray', linewidth=0.2)

    # Mean ± error (NaN-safe for artifact-removed frames)
    if responding_indices:
        mean_trajectory = np.nanmedian(intensities_normalized[responding_indices, :], axis=0)
        std_trajectory = np.nanstd(intensities_normalized[responding_indices, :], axis=0)
        if use_sem:
            # Count non-NaN cells per frame for correct SEM
            n_valid = np.sum(np.isfinite(intensities_normalized[responding_indices, :]), axis=0)
            n_valid = np.maximum(n_valid, 1)  # avoid division by zero
            err_trajectory = std_trajectory / np.sqrt(n_valid)
        else:
            err_trajectory = std_trajectory

    ax.plot(full_frames, mean_trajectory, 'o-',
            color=colors[0], linewidth=1, label='Experimental (mean)', markersize=6)
    ax.fill_between(full_frames,
                    mean_trajectory - err_trajectory,
                    mean_trajectory + err_trajectory,
                    color=colors[0], alpha=0.2)

    # TASEP simulation overlay (if provided)
    if mean_intensity_ssa_inh is not None and err_intensity_ssa_inh is not None:
        legend_label_sim = (fr'Model Fit ($k_e$={np.round(list_param[1],1)}, $k_i$={np.round(list_param[0],3)})'
                        if list_param[1] is not None and list_param[0] is not None else 'Simulation')
        plt.plot(time_array_min-5, mean_intensity_ssa_inh, '-', color='red', linewidth=3, label=legend_label_sim)
        plt.fill_between(time_array_min-5, mean_intensity_ssa_inh - err_intensity_ssa_inh,
                        mean_intensity_ssa_inh + err_intensity_ssa_inh, color='red', alpha=0.1)

    # ── Model fit ────────────────────────────────────────────────────────
    fit_result = None
    if fit_model is not None:
        # Default fit range: from inhibitor application to end
        start = fit_start_idx if fit_start_idx is not None else inhibitor_frame_index
        end = fit_end_idx  # None → full array end (handled inside fit_inhibitor_model)

        fit_result = fit_inhibitor_model(
            full_frames, mean_trajectory,
            err_data=err_trajectory,
            model=fit_model,
            fit_start_idx=start,
            fit_end_idx=end,
            runoff_fraction=runoff_fraction,
        )

        if fit_result is not None:
            model_labels = {'linear': 'Linear Fit', 'exponential': 'Exponential Fit',
                            'heaviside': 'Heaviside Fit'}
            label = model_labels.get(fit_result['model'], 'Fit')



            t_half = fit_result['t_half']
            t_runoff = fit_result['t_runoff']
            frac_pct = int(fit_result['runoff_fraction'] * 100)

            if show_fit:
                ax.plot(full_frames[start:], fit_result['fitted_curve'][start:], '-',
                        color='red', linewidth=1.5, label=label)
            if show_runoff_time:
                ax.axvline(x=t_half, color='green', linestyle='--', linewidth=1,
                           label=fr'$t_{{1/2}}$ ~ {t_half:.1f} min')
                ax.axvline(x=t_runoff, color='orange', linestyle='--', linewidth=1,
                           label=fr'$\tau_{{runoff}}$ ({frac_pct}%) ~ {t_runoff:.1f} min')

            # Print fitted parameters
            print(f'── {label} ──')
            for k, v in fit_result['params'].items():
                print(f'  {k}: {v:.4f}')
            print(f'  t½:      {fit_result["t_half"]:.2f} min')
            print(f'  τ_runoff ({frac_pct}%): {fit_result["t_runoff"]:.2f} min')
            chi2r = fit_result['chi2_reduced']
            print(f'  χ²_red:  {chi2r:.4f}  (dof={fit_result["dof"]})')

    # Treatment line at t = 0
    if show_treatment_line:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1,
                    label=f'{treatment_label} Treatment')

    ax.set_xlabel("Time (min)", fontdict={'size': 16, 'color': 'black'})
    ax.set_ylabel(y_label, fontdict={'size': 16, 'color': 'black'})
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black', colors='black')

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    plt.tight_layout()
    legend = ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5),
                       framealpha=0.9, edgecolor='black')
    plt.savefig(results_folder.joinpath('HT_'+plot_name+'.png'), dpi=600,
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.savefig(results_folder.joinpath('HT_'+plot_name+'.svg'), dpi=600,
                bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    return fit_result


def plot_multiple_inhibitors(full_frames_list,
                                intensities_normalized_list,
                                inhibitor_frame_index,
                                results_folder=None,
                                plot_name='HT_multi',
                                responding_indices_list=None,
                                figsize=(6, 3),
                                colors=None,
                                legend_labels=None,
                                use_sem=True,
                                show_individual_trajectories=True,
                                ylims=(0, 1.5),
                                xlims=None,
                                y_label='Norm. Intensity',
                                treatment_label='Inhibitor',
                                show_treatment_line=True,
                                # ── Fitting parameters ──
                                fit_model=None,
                                fit_start_idx=None,
                                fit_end_idx=None,
                                show_fit=True,
                                show_runoff_time=True,
                                runoff_fraction=0.95):
    """Plot multiple inhibitor datasets on the same axes with optional model fits.

    Parameters
    ----------
    full_frames_list : list or array
        Either a single 1D array of frame times (applies to all datasets)
        or a list of 1D arrays, one per dataset.
    intensities_normalized_list : list of 2D arrays
        Each element is an (n_cells × n_frames) array of normalized intensities.
    inhibitor_frame_index : int
        Frame index at which inhibitor treatment starts.
    results_folder : Path or str, optional
        Where to save the figure (will be created if needed).
    plot_name : str, optional
        Filename suffix for the saved figure.
    responding_indices_list : list of lists, optional
        Per‐dataset lists of cell indices to include. Defaults to all.
    figsize : tuple, optional
    colors : list of str, optional
        Matplotlib color codes for each dataset.
    legend_labels : list of str, optional
        Text labels for each dataset's mean trace.
    use_sem : bool, optional
        If True, error bands show SEM; else SD.
    show_individual_trajectories : bool, optional
    ylims : tuple, optional
        (ymin, ymax) for the plot.
    xlims : tuple, optional
        (xmin, xmax) for the plot. If None, auto-scaled.
    fit_model : str or None
        Model to fit per dataset: 'linear', 'exponential', 'heaviside', or None.
    fit_start_idx : int or None
        Start index for fitting range. Defaults to inhibitor_frame_index.
    fit_end_idx : int or None
        End index for fitting range (inclusive). Defaults to last frame.
    show_fit : bool
        If True (default), overlay the fitted curve on the plot.
    show_runoff_time : bool
        If True (default), draw vertical lines for t½ and τ_runoff.
    runoff_fraction : float
        Fraction of total decay for run-off time definition (default 0.95).

    Returns
    -------
    list of dict or None
        One fit result dictionary per dataset (from fit_inhibitor_model),
        or None for datasets where fitting was not performed or failed.
    """
    # Prepare output folder
    if results_folder is None:
        results_folder = Path(current_dir).joinpath('results_HT')
    results_folder.mkdir(parents=True, exist_ok=True)

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fit_results = []

    # Loop over each dataset
    for idx, intensities in enumerate(intensities_normalized_list):
        # Select frames array
        frames = (full_frames_list[idx]
                  if isinstance(full_frames_list, (list, tuple))
                  else full_frames_list)
        color = colors[idx % len(colors)]

        # Select responding cell indices
        resp_idx = (responding_indices_list[idx]
                    if (responding_indices_list and idx < len(responding_indices_list))
                    else list(range(intensities.shape[0])))

        # Plot individual trajectories
        if show_individual_trajectories:
            for i in resp_idx:
                ax.plot(frames, intensities[i],
                        linestyle='-', color=color,
                        linewidth=0.3, alpha=0.4)

        # Compute mean & error (NaN-safe for artifact-removed frames)
        data = intensities[resp_idx, :]
        mean_traj = np.nanmedian(data, axis=0)
        std_traj  = np.nanstd(data, axis=0)
        if use_sem:
            n_valid = np.sum(np.isfinite(data), axis=0)
            n_valid = np.maximum(n_valid, 1)
            err_traj = std_traj / np.sqrt(n_valid)
        else:
            err_traj = std_traj

        # Determine legend text
        label_text = (legend_labels[idx]
                      if (legend_labels and idx < len(legend_labels))
                      else f'Dataset {idx+1}')

        # Plot mean ± error band
        ax.plot(frames, mean_traj, 'o-', color=color,
                linewidth=1, markersize=6,
                label=label_text)
        ax.fill_between(frames,
                        mean_traj - err_traj,
                        mean_traj + err_traj,
                        color=color, alpha=0.2)

        # ── Model fit ────────────────────────────────────────────────
        if fit_model is not None:
            start = fit_start_idx if fit_start_idx is not None else inhibitor_frame_index
            end = fit_end_idx

            fit_result = fit_inhibitor_model(
                frames, mean_traj,
                err_data=err_traj,
                model=fit_model,
                fit_start_idx=start,
                fit_end_idx=end,
                runoff_fraction=runoff_fraction,
            )

            if fit_result is not None:
                model_labels = {'linear': 'Linear Fit', 'exponential': 'Exponential Fit',
                                'heaviside': 'Heaviside Fit'}
                fit_label = model_labels.get(fit_result['model'], 'Fit')
                t_half = fit_result['t_half']
                t_runoff = fit_result['t_runoff']
                frac_pct = int(runoff_fraction * 100)
                
                if show_fit:
                    ax.plot(frames[start:], fit_result['fitted_curve'][start:], '-',
                            color='red', linewidth=1.5,
                            label=f'{label_text} {fit_label}')

                if show_runoff_time:
                    ax.axvline(x=t_half, color=color, linestyle=':', linewidth=1,
                               label=f'{label_text} t½ ~ {t_half:.1f}')
                    ax.axvline(x=t_runoff, color=color, linestyle='--', linewidth=1,
                               label=f'{label_text} τ ({frac_pct}%) ~ {t_runoff:.1f}')

                # Print fitted parameters
                print(f'── {label_text}: {fit_label} ──')
                for k, v in fit_result['params'].items():
                    print(f'  {k}: {v:.4f}')
                print(f'  t½:      {fit_result["t_half"]:.2f} min')
                print(f'  τ_runoff ({frac_pct}%): {fit_result["t_runoff"]:.2f} min')
                chi2r = fit_result['chi2_reduced']
                print(f'  χ²_red:  {chi2r:.4f}  (dof={fit_result["dof"]})')

            fit_results.append(fit_result)
        else:
            fit_results.append(None)

    # Plot treatment line at zero
    if show_treatment_line:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1,
                    label=treatment_label)

    # Styling
    ax.set_xlabel("Time (min)", fontdict={'size': 16, 'color': 'black'})
    ax.set_ylabel(y_label, fontdict={'size': 16, 'color': 'black'})
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.set_ylim(*ylims)
    if xlims is not None:
        ax.set_xlim(*xlims)

    legend = ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5),
                       framealpha=0.9, edgecolor='black')
    plt.tight_layout()

    # Save & show
    plt.savefig(results_folder.joinpath(f'HT_{plot_name}.png'), dpi=600,
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.savefig(results_folder.joinpath(f'HT_{plot_name}.svg'), dpi=600,
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()

    return fit_results



def process_inhibitor_data(data_dir, inhibitor_frame_index, substring_in_data_dir='', selected_field='spot_int_ch_0', use_sem=True, show_summary=True, max_percentage_threshold_after_treatment=None, frame_rate_min=1,
                     frame_interval_sec=60, simulation_dna_sequence=None, inhibitor_delay_time_seconds=60, list_tag_sequences=[HA_TAG], ki_simulation=0.04, ke_simulation=4.5,
                     normalization_method='mean', percentile_range=(5, 95), verbose=False,
                     remove_frame_at_inhibitor_application=False):
    """
    Process inhibitor runoff experiment data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing results subfolders with tracking CSV files.
    inhibitor_frame_index : int
        Frame index at which treatment starts.
    substring_in_data_dir : str
        Filter string to select specific subfolders.
    selected_field : str
        Column name for intensity values (e.g., 'spot_int_ch_0').
    use_sem : bool
        If True, use SEM for error bars; else use SD.
    show_summary : bool
        If True, print summary statistics.
    max_percentage_threshold_after_treatment : float or None
        Threshold (0-1) to classify responding vs non-responding cells.
    frame_rate_min : int
        Frame rate for downsampling (1 = every frame).
    frame_interval_sec : float
        Time interval between frames in seconds. Default 60 (1 min).
        Use 20 for 20-second intervals, etc.
    simulation_dna_sequence : str or None
        DNA sequence for TASEP simulation (optional).
    inhibitor_delay_time_seconds : float
        Delay time for drug to enter cell in simulation.
    list_tag_sequences : list
        Tag sequences for probe detection.
    ki_simulation : float
        Initiation rate for simulation.
    ke_simulation : float
        Elongation rate for simulation.
    normalization_method : str, optional
        'mean' (default) divides by pre-treatment mean intensity (values start ~1.0).
        'minmax' scales each cell's trajectory to [0, 1] range.
        'percentile' scales using percentile bounds (robust to outliers).
        None: no normalization.
    percentile_range : tuple, optional
        Percentile bounds for 'percentile' method. Default (5, 95).
        Use (1, 99) for wider range.
    verbose : bool, optional
        If True (default), print processing details and summary statistics.
        Set to False to suppress all print output.
    remove_frame_at_inhibitor_application : bool, optional
        If True, replaces the frame at inhibitor application with NaN
        to remove the focus artifact (default False). During live-cell
        inhibitor experiments, the physical act of adding the drug
        (e.g., pipetting media into the dish) often causes cells to
        briefly go out of focus. This produces a transient intensity
        dip/spike at the treatment frame that is not biological but
        rather a mechanical artifact. Setting this flag to True replaces
        that single frame with NaN, which is then gracefully skipped
        during mean calculation, error estimation, and model fitting.
        
    Returns
    -------
    tuple
        (responding_indices, time_min_recentered, intensities_normalized, array_particles,
         list_simulation_parameters, time_array_sim_min, mean_intensity_ssa_inh, err_intensity_ssa_inh)
    """

    list_dataframes = []
    subfolders = [folder for folder in data_dir.iterdir() if folder.is_dir()]
    subfolders = [folder for folder in subfolders if 'results_' in folder.name and substring_in_data_dir in folder.name]
    if verbose:
        print('List of processed dataframes:')
    for subfolder in subfolders:
        files = [f for f in subfolder.iterdir() if f.is_file() and 'tracking_' in f.name and f.suffix == '.csv']
        if not files:
            if verbose:
                print(f'Warning: No tracking CSV found in {subfolder.name}, skipping.')
            continue
        dfs = pd.read_csv(files[0])
        list_dataframes.append(dfs)
        if verbose:
            print('subfolder:', subfolder)

    # detect the maximum frame number across all dataframes
    max_frame = 0
    for df in list_dataframes:
        max_frame = max(max_frame, df['frame'].max())

    # terminate the program if the maximum frame is less than 1
    if max_frame < 10:
        if verbose:
            print('No dataframes found with frame number greater than 10 frames.')
        return None, None, None, None, None, None, None, None

    full_frames = np.arange(0, max_frame)
    frame_indices = np.arange(0, max_frame, frame_rate_min)  # Frame indices for processing
    array_particles = np.zeros((len(list_dataframes), len(frame_indices)))
    average_intensity = np.zeros((len(list_dataframes), len(frame_indices)))
    intensities_normalized = np.zeros((len(list_dataframes), len(frame_indices)))
    average_molecules_before_treatment = np.zeros(len(list_dataframes))
    for i, df in enumerate(list_dataframes):
        # Group by 'frame' and count unique particles
        particle_counts_per_frame = df.groupby('frame')['particle'].nunique()
        sum_intensities_per_frame = df.groupby('frame')[selected_field].sum()
        # Re-index to include frames with no particles (fill missing with 0)
        particle_counts_per_frame = particle_counts_per_frame.reindex(frame_indices, fill_value=0).values
        sum_intensities_per_frame = sum_intensities_per_frame.reindex(frame_indices, fill_value=0).values
        # Ensure the lengths match the max frame
        if len(particle_counts_per_frame) > max_frame:
            particle_counts_per_frame = particle_counts_per_frame[:max_frame]
            sum_intensities_per_frame = sum_intensities_per_frame[:max_frame]
        # Compute raw intensities (always use 'mean' per-cell first)
        intensities_normalized_before_treatment_intensity, average_intensity_with_respect_number_particles, average_particles_before_treatment = calculate_intensity(
            particle_counts_per_frame, sum_intensities_per_frame, inhibitor_frame_index,
            normalization_method=normalization_method
        )
        # Store in array
        normalized_particles, _ = calculate_number_of_particles_per_frame(particle_counts_per_frame, inhibitor_frame_index)
        array_particles[i] = normalized_particles
        average_intensity[i] = average_intensity_with_respect_number_particles
        intensities_normalized[i] = intensities_normalized_before_treatment_intensity
        average_molecules_before_treatment[i] = average_particles_before_treatment

    # Apply global normalization across all cells (after the loop)
    if normalization_method == 'minmax':
        global_min = intensities_normalized.min()
        global_max = intensities_normalized.max()
        if global_max - global_min > 0:
            intensities_normalized = (intensities_normalized - global_min) / (global_max - global_min)
        if verbose:
            print(f'Global minmax normalization applied (min={global_min:.4f}, max={global_max:.4f})')
    elif normalization_method == 'percentile':
        global_low = np.percentile(intensities_normalized, percentile_range[0])
        global_high = np.percentile(intensities_normalized, percentile_range[1])
        if global_high - global_low > 0:
            intensities_normalized = (intensities_normalized - global_low) / (global_high - global_low)
        if verbose:
            print(f'Global percentile normalization applied (P{percentile_range[0]}={global_low:.4f}, P{percentile_range[1]}={global_high:.4f})')

    # ── Remove artifact frame at inhibitor application ────────────────
    # During live-cell experiments, adding the inhibitor (e.g., pipetting
    # harringtonine into the dish) mechanically perturbs the sample,
    # causing cells to transiently go out of focus. This creates an
    # artificial intensity dip at the treatment frame that does not
    # reflect actual translational run-off. Replacing this frame with
    # NaN removes the artifact while preserving the time axis, so the
    # mean, SEM, and model fits are not biased by the focus disturbance.
    if remove_frame_at_inhibitor_application:
        intensities_normalized[:, inhibitor_frame_index] = np.nan
        array_particles[:, inhibitor_frame_index] = np.nan
        if verbose:
            print(f'Removed frame at inhibitor application '
                  f'(index {inhibitor_frame_index}) → replaced with NaN')

    treatment_start_index = inhibitor_frame_index #np.argmin(np.abs(full_frames - inhibitor_frame_index))
    non_responding_indices = []  # average post-treatment is not decreasing below the threshold
    responding_indices = []
    # Classify each trajectory
    if max_percentage_threshold_after_treatment is not None:
        for i, trajectory in enumerate(intensities_normalized):
            baseline = np.mean(trajectory[:treatment_start_index])
            threshold = max_percentage_threshold_after_treatment * baseline
            avg_post_treatment = np.mean(trajectory[treatment_start_index:])
            if avg_post_treatment >= threshold:
                non_responding_indices.append(i)
            else:
                responding_indices.append(i)
    else:
        # print a warning if the threshold is not provided
        if verbose:
            print('Warning: No threshold provided. All trajectories are considered responding.')
        # if no threshold is provided, all trajectories are considered responding
        responding_indices = list(range(len(intensities_normalized)))
        non_responding_indices = []

    if show_summary and verbose:
        # report percentage of non-responding trajectories
        print('---------Summary: ----------')
        print('\n----------Cells------------')
        # report total cells
        total_cells = len(intensities_normalized)
        print('total_cells: ', total_cells)
        number_of_responding_cells = len(responding_indices)
        print('number_of_responding_cells: ', number_of_responding_cells)
        number_of_non_responding_cells = len(non_responding_indices)
        print('number_of_non_responding_cells: ', number_of_non_responding_cells)
        percentage_non_responding_cells = number_of_non_responding_cells / (number_of_responding_cells + number_of_non_responding_cells) * 100
        print('percentage_non_responding_cells: ', np.round(percentage_non_responding_cells,1))
        print('\n----------RNA------------')
        total_molecules_before_treatment = average_molecules_before_treatment.sum()
        print('average_total_molecules_before_treatment: ', total_molecules_before_treatment)
        average_molecules_before_treatment_responding = average_molecules_before_treatment[responding_indices].sum()
        print('average_molecules_before_treatment_responding: ', np.round(average_molecules_before_treatment_responding,1))
        average_molecules_before_treatment_non_responding = average_molecules_before_treatment[non_responding_indices].sum()
        print('average_molecules_before_treatment_non_responding: ', average_molecules_before_treatment_non_responding)

    # Convert frame indices to time in minutes, centered on treatment
    time_min_recentered = (frame_indices - inhibitor_frame_index) * frame_interval_sec / 60.0


    if simulation_dna_sequence is not None:
        list_simulation_parameters = [ki_simulation, ke_simulation]
        ########################## Modeling  #########################################
        #inhibitor_delay_time_seconds = 60 # seconds. According to Tanenbaum paper 2016.
        number_repetitions = 100
        burnin_time = 2000
        #t_max = 360*5 #timePerturbationApplication + 25*60  # Maximum time
        t_max = max_frame * frame_interval_sec  # Total experiment time in seconds
        step_size_in_sec = 1
        time_array = np.arange(0, t_max, step_size_in_sec)
        #plot_name, rna, first_probe_position_vector, gene_length,list_param = model_and_data_selection(processed_data)
        _, rna, gene_length, first_probe_position_vector, second_probe_position_vector = read_gene_sequence_return_probes(simulation_dna_sequence, min_protein_length=50, list_tag_sequences=list_tag_sequences)
        ke = calculate_codon_elongation_rates (rna, global_elongation_rate=list_simulation_parameters[1],)
        timePerturbationApplication = inhibitor_frame_index * frame_interval_sec + inhibitor_delay_time_seconds
        evaluatingInhibitor = 1
        ssa_array = simulate_TASEP_SSA(list_simulation_parameters[0],
                                    ke,
                                    gene_length,
                                    t_max,
                                    time_interval_in_seconds=step_size_in_sec,
                                    number_repetitions=number_repetitions,
                                    first_probe_position_vector=first_probe_position_vector,
                                    timePerturbationApplication = timePerturbationApplication,
                                    inhibitor_effectiveness=95,
                                    evaluatingInhibitor=evaluatingInhibitor,
                                    burnin_time=burnin_time,
                                    constant_elongation_rate=list_simulation_parameters[1],
                                    fast_output=True)[2]
        # downsample time array and simulation output to match experimental frame interval
        downsample_factor = int(frame_interval_sec)  # e.g., 60 for 1 min, 20 for 20 sec
        time_array_downsampled = time_array[::downsample_factor]
        ssa_array_downsampled = ssa_array[:, ::downsample_factor]
        # plotting
        time_array_sim_min = time_array_downsampled/60
        normalized_data = np.zeros_like(ssa_array_downsampled)
        for i in range(ssa_array_downsampled.shape[0]):
            mean_before_treatment = np.mean(ssa_array_downsampled[i,:inhibitor_frame_index+1])
            normalized_data[i] = ssa_array_downsampled[i]/mean_before_treatment
        mean_intensity_ssa_inh = np.mean(normalized_data, axis=0)
        if use_sem:
            err_intensity_ssa_inh = np.std(normalized_data, axis=0) / np.sqrt(normalized_data.shape[0])
        else:
            err_intensity_ssa_inh = np.std(normalized_data, axis=0) #/np.sqrt(normalized_data.shape[0])
    else:
        mean_intensity_ssa_inh = None
        err_intensity_ssa_inh = None
        time_array_sim_min = None
        list_simulation_parameters = [None, None]

    return responding_indices, time_min_recentered, intensities_normalized, array_particles, list_simulation_parameters, time_array_sim_min, mean_intensity_ssa_inh, err_intensity_ssa_inh



def simulate_inhibitor(gene_sequence, ki=0.04, ke_global=5, use_sem=False, max_frame=20,
                       list_tag_sequences=None, inhibitor_frame=5, inhibitor_delay_seconds=60):
    """
    Run a TASEP simulation for a gene sequence with inhibitor treatment.

    This is a simulation-only function (no experimental data). It uses
    codon-usage-aware elongation rates and supports multi-tag probe detection.

    Parameters
    ----------
    gene_sequence : str
        DNA sequence to simulate.
    ki : float
        Initiation rate.
    ke_global : float
        Global elongation rate.
    use_sem : bool
        If True, error bars use SEM; else SD.
    max_frame : int
        Maximum number of frames (minutes) to simulate.
    list_tag_sequences : list or None
        Tag sequences for probe detection. Defaults to [HA_TAG, GFP_TAG].
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts.
    inhibitor_delay_seconds : float
        Delay time for drug to enter cell in simulation (seconds).

    Returns
    -------
    tuple
        (time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh)
    """
    if list_tag_sequences is None:
        list_tag_sequences = [HA_TAG, GFP_TAG]

    # reading the gene sequence
    protein, rna, _, indexes_tags, _, _ ,_ = read_sequence(seq=gene_sequence, min_protein_length=50, TAG=list_tag_sequences)
    gene_length = len(protein)
    tag_positions_first_probe_vector = indexes_tags[0]
    first_probe_position_vector = create_probe_vector(tag_positions_first_probe_vector, gene_length)

    # second probe vector.
    tag_positions_second_probe_vector = indexes_tags[1]
    second_probe_position_vector = create_probe_vector(tag_positions_second_probe_vector, gene_length)

    # print the first and second probe positions
    print(f"First probe positions: {tag_positions_first_probe_vector}")
    print(f"Second probe positions: {tag_positions_second_probe_vector}")

    ke_codon_dependent = calculate_codon_elongation_rates(rna, global_elongation_rate=ke_global)

    full_frames = np.arange(0, max_frame)
    full_frames = full_frames - inhibitor_frame

    ########################## Modeling  #########################################
    number_repetitions = 100
    burnin_time = 2000
    t_max = max_frame * 60
    step_size_in_sec = 1
    time_array = np.arange(0, t_max, step_size_in_sec)
    timePerturbationApplication = inhibitor_frame * 60 + inhibitor_delay_seconds
    evaluatingInhibitor = 1
    ssa_array = simulate_TASEP_SSA(ki,
                                ke_codon_dependent,
                                gene_length,
                                t_max,
                                time_interval_in_seconds=step_size_in_sec,
                                number_repetitions=number_repetitions,
                                first_probe_position_vector=first_probe_position_vector,
                                timePerturbationApplication=timePerturbationApplication,
                                inhibitor_effectiveness=95,
                                evaluatingInhibitor=evaluatingInhibitor,
                                burnin_time=burnin_time,
                                constant_elongation_rate=None,  # codon-usage aware
                                fast_output=False)[2]
    # downsample time array and simulation output to 1 minute resolution.
    time_array_downsampled = time_array[::60]
    ssa_array_downsampled = ssa_array[:, ::60]
    # plotting
    time_array_min = time_array_downsampled / 60
    normalized_data = np.zeros_like(ssa_array_downsampled)
    for i in range(ssa_array_downsampled.shape[0]):
        mean_before_treatment = np.mean(ssa_array_downsampled[i, :inhibitor_frame + 1])
        normalized_data[i] = ssa_array_downsampled[i] / mean_before_treatment
    mean_intensity_ssa_inh = np.mean(normalized_data, axis=0)

    if use_sem:
        err_intensity_ssa_inh = np.std(normalized_data, axis=0) / np.sqrt(normalized_data.shape[0])
    else:
        err_intensity_ssa_inh = np.std(normalized_data, axis=0)

    return time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh


def plot_inhibitor_simulation(legend_label, time_array_min, mean_intensity_ssa_inh, err_intensity_ssa_inh,
                              figsize=(6, 3), ylims=(0, 1.4), inhibitor_frame=5):
    """
    Plot a single inhibitor simulation result.

    Parameters
    ----------
    legend_label : str
        Label for the simulation trace.
    time_array_min : np.ndarray
        Time array in minutes.
    mean_intensity_ssa_inh : np.ndarray
        Mean normalized intensity from simulation.
    err_intensity_ssa_inh : np.ndarray
        Error (SD or SEM) of normalized intensity.
    figsize : tuple
        Figure size.
    ylims : tuple
        Y-axis limits.
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts (for time offset).
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Plotting the model
    plt.plot(time_array_min - inhibitor_frame, mean_intensity_ssa_inh, '-', color='red', linewidth=4, label=legend_label)
    plt.fill_between(time_array_min - inhibitor_frame, mean_intensity_ssa_inh - err_intensity_ssa_inh,
                    mean_intensity_ssa_inh + err_intensity_ssa_inh, color='red', alpha=0.1)

    # Plot the inhibitor frame as a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel("Time (min)", fontdict={'size': 16})
    ax.set_ylabel("Norm. Intensity", fontdict={'size': 16})
    ax.tick_params(axis='both', which='major', labelsize=12)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.ylim(ylims)
    plt.tight_layout()
    ax.legend(fontsize=10)
    plt.show()


def plot_multiple_inhibitor_simulations(
    legend_labels,
    time_arrays_min,
    mean_intensities,
    err_intensities,
    figsize=(8, 5),
    ylims=(0, 1.5),
    inhibitor_frame=5
):
    """
    Overlay multiple inhibitor simulation results on a single plot.

    Parameters
    ----------
    legend_labels : str or list of str
        Labels for each simulation trace.
    time_arrays_min : np.ndarray or list of np.ndarray
        Time arrays in minutes.
    mean_intensities : np.ndarray or list of np.ndarray
        Mean normalized intensities from each simulation.
    err_intensities : np.ndarray or list of np.ndarray
        Errors (SD or SEM) for each simulation.
    figsize : tuple
        Figure size.
    ylims : tuple
        Y-axis limits.
    inhibitor_frame : int
        Frame (minute) at which inhibitor treatment starts (for time offset).
    """
    # --- normalize legend_labels to list ---
    if not isinstance(legend_labels, (list, tuple)):
        legend_labels = [legend_labels]
    n = len(legend_labels)

    # --- broadcast time_arrays_min if needed ---
    if not isinstance(time_arrays_min, (list, tuple)):
        time_arrays_min = [time_arrays_min] * n

    # --- similarly ensure mean_intensities and err_intensities are lists ---
    if not isinstance(mean_intensities, (list, tuple)):
        mean_intensities = [mean_intensities] * n
    if not isinstance(err_intensities, (list, tuple)):
        err_intensities = [err_intensities] * n

    # sanity check
    assert len(time_arrays_min) == n == len(mean_intensities) == len(err_intensities), \
        "All four inputs must be lists of the same length."

    # 1) create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # 2) choose a colormap and generate n distinct colors
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))

    # 3) plot each simulation with its assigned color
    for idx, (label, t_min, mean_inh, err_inh) in enumerate(zip(
        legend_labels, time_arrays_min, mean_intensities, err_intensities
    )):
        t_plot = np.array(t_min) - inhibitor_frame
        color = colors[idx]
        ax.plot(t_plot, mean_inh,
                '-', color=color, linewidth=3, label=label)
        ax.fill_between(
            t_plot,
            mean_inh - err_inh,
            mean_inh + err_inh,
            color=color,
            alpha=0.05
        )

    # vertical line at t=0
    ax.axvline(x=0, linestyle='--', color='k', linewidth=1)

    # axes labels and formatting
    ax.set_xlabel("Time (min)", size=16)
    ax.set_ylabel("Norm. Intensity", size=16)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_ylim(ylims)

    # legend and layout
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    plt.show()
