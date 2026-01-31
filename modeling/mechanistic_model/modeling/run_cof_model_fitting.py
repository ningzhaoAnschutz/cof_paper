"""Co-Translational Folding (CoF) Mechanistic Model Fitting.

This module implements kinetic models for co-translational protein folding as
measured by the coTFT (Co-Translational Folding Tracking) technology. The system
uses tandem GFP domains fused to an smHA tag, where folding is detected by
intrabody binding to properly folded GFP.

Key Insight:
    The CoF efficiency is measured as the RATIO of green/magenta signals,
    reflecting the NUMBER of bound intrabodies divided by NUMBER of nascent chains.
    The data shows INCREASING efficiency with more GFP copies because more domains
    provide more opportunities for intrabody detection.

Models:
    - OnePoolModel: Pure kinetics model (1 parameter: k_fold)
        Assumes all nascent chains are competent for folding. Predicts decreasing
        efficiency with more domains (fails to match experimental data).
    - TwoPoolModel: Population heterogeneity model (3 parameters: k_fold, f_base, f_gain)
        Proposes that only a fraction of nascent chains can productively fold,
        with an avidity effect increasing the effective fraction with more domains.

Usage:
    Run as script to fit models to experimental data and generate figures:
        $ python run_cof_model_fitting.py

Author: CoF Project Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# PARAMETERS
# =============================================================================

# Global configuration for model prediction visualization
# Dashed lines indicate discrete predictions connected for visual guidance only
MODEL_PLOT_STYLE = {
    'linestyle': '--',        # Dashed line (use '-' for solid)
    'linewidth': 1.5,         # Line thickness (reduced from 2.5)
    'line_alpha': 0.8,        # Line transparency
    'marker_size': 80,        # Scatter marker size (s parameter)
    'marker_edge_width': 1.5, # Marker edge thickness
}


def load_gfp_positions(csv_path=None):
    """Load GFP domain positions from CSV file.
    
    Reads sequence position data for each GFP domain in the tandem construct.
    
    Args:
        csv_path: Path to CSV file containing GFP positions. If None, uses
            the default 'GFP_positions.csv' in the same directory as this script.
            
    Returns:
        tuple: A tuple containing:
            - gfp_domain_ends (list[int]): C-terminus positions (aa) for each GFP.
            - gfp_emergence_positions (list[int]): Positions after clearing the
              ribosome exit tunnel (aa) for each GFP.
              
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        
    Example:
        >>> ends, emergence = load_gfp_positions()
        >>> print(f"GFP1 emerges at position {emergence[0]} aa")
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / 'GFP_positions.csv'
    
    df = pd.read_csv(csv_path)
    gfp_domain_ends = df['GFP_tag_position_end'].tolist()
    gfp_emergence_positions = df['GFP_tag_position_end_plus_exit_tunnel'].tolist()
    
    return gfp_domain_ends, gfp_emergence_positions


class TranslationParameters:
    """Translation parameters derived from experimental measurements.
    
    Contains fixed kinetic parameters from autocorrelation analysis and sequence
    information for the tandem GFP reporter construct.
    
    Attributes:
        k_elongation: Ribosome elongation rate in aa/s (4.85 aa/s from autocorrelation).
        k_init: Translation initiation rate in s⁻¹ (0.063 s⁻¹, ~1 ribosome/16s).
        L_tunnel: Amino acids required to clear ribosome exit tunnel (35 aa).
        L_total: Total gene length in amino acids (1826 aa).
        ribosome_footprint: Amino acids covered by one ribosome (10 aa = 30 nt).
        HA_end: C-terminus position of the HA tag (326 aa).
        GFP_domain_ends: List of C-terminus positions for each GFP domain.
        GFP_emergence_positions: List of positions after clearing exit tunnel.
    
    Example:
        >>> params = TranslationParameters()
        >>> print(f"Total translation time: {params.T_total:.1f}s")
        >>> print(f"GFP1 has {params.time_window(1):.0f}s to fold")
    """
    
    k_elongation = 4.85  # aa/s (elongation rate, from autocorrelation analysis)
    k_init = 0.063       # s^-1 (initiation rate, from autocorrelation analysis)
    L_tunnel = 35        # aa required to clear ribosome exit tunnel
    L_total = 1826       # Total gene length in aa
    ribosome_footprint = 10  # aa covered by one ribosome (10 codons = 30 nt)
    
    # Sequence positions (aa) - loaded from CSV
    HA_end = 326
    
    def __init__(self, csv_path=None):
        """Initialize with GFP positions from CSV file.
        
        Args:
            csv_path: Path to CSV file with GFP positions. Uses default if None.
        """
        gfp_ends, gfp_emergence = load_gfp_positions(csv_path)
        self.GFP_domain_ends = gfp_ends  # C-terminus of each GFP domain
        self.GFP_emergence_positions = gfp_emergence  # Position after clearing tunnel
    
    @property
    def T_total(self):
        """Total translation time in seconds."""
        return self.L_total / self.k_elongation
    
    @property
    def mean_ribosome_spacing(self):
        """Average spacing between ribosomes in amino acids."""
        return self.k_elongation / self.k_init if self.k_init > 0 else float('inf')
    
    @property
    def avg_ribosomes_per_mRNA(self):
        """Average number of ribosomes on one mRNA."""
        return self.L_total / self.mean_ribosome_spacing if self.mean_ribosome_spacing > 0 else 0
    
    def domain_emergence_time(self, position):
        """Calculate when a GFP domain fully emerges from the ribosome tunnel.
        
        Args:
            position: GFP domain position (1-indexed, 1-6).
            
        Returns:
            float: Time in seconds when the domain emerges.
            
        Raises:
            ValueError: If position is not in range 1-6.
        """
        if position < 1 or position > 6:
            raise ValueError(f"Position must be 1-6, got {position}")
        # Use the pre-computed emergence position from CSV
        aa_position = self.GFP_emergence_positions[position - 1]
        return aa_position / self.k_elongation
    
    def time_window(self, position):
        """Calculate time available for a domain to fold before translation ends.
        
        Args:
            position: GFP domain position (1-indexed, 1-6).
            
        Returns:
            float: Time window in seconds for folding.
        """
        return self.T_total - self.domain_emergence_time(position)


# =============================================================================
# MODELS
# =============================================================================

class CoFModel:
    """Base kinetic model for co-translational folding.
    
    Implements a single-step folding kinetics model: U → F (unfolded → folded).
    The binding step (intrabody binding to folded GFP) is assumed to be fast
    relative to folding and is not rate-limiting.
    
    Attributes:
        params: TranslationParameters instance with kinetic constants.
    """
    
    def __init__(self, params=None):
        """Initialize the model with translation parameters.
        
        Args:
            params: TranslationParameters instance. Uses defaults if None.
        """
        self.params = params or TranslationParameters()
    
    def folded_fraction(self, t, k_fold):
        """Calculate the probability of being in the folded state after time t.
        
        Uses simple first-order kinetics: U → F with rate k_fold.
        Closed-form solution: F(t) = 1 - exp(-k_fold * t)
        
        Args:
            t: Time in seconds since domain emergence.
            k_fold: Folding rate constant in s⁻¹.
            
        Returns:
            float: Probability of being folded (0 to 1).
        """
        if t <= 0:
            return 0.0
        return 1.0 - np.exp(-k_fold * t)


class TwoPoolModel(CoFModel):
    """Two-Pool model with population heterogeneity.
    
    This model proposes that nascent chains fall into two populations:
        - Pool 1 (fraction f): Can achieve productive co-translational folding
        - Pool 2 (fraction 1-f): Cannot fold efficiently (misfolded, aggregated)
    
    The effective foldable fraction increases with more GFP copies due to an
    avidity effect - more domains provide more opportunities to "catch" at
    least one properly folded domain.
    
    Parameters:
        - k_fold: Folding rate constant (s⁻¹)
        - f_base: Baseline foldable fraction (typically ~0.4)
        - f_gain: Increase in foldable fraction per additional GFP domain
    
    Example:
        >>> model = TwoPoolModel()
        >>> efficiency = model.efficiency(n_gfp=4, k_fold=0.013, f_base=0.42, f_gain=0.12)
        >>> print(f"4xGFP efficiency: {efficiency:.1f}%")
    """
    
    def efficiency(self, n_gfp, k_fold, f_base=0.4, f_gain=0.1):
        """Calculate CoF efficiency for a given number of GFP domains.
        
        Args:
            n_gfp: Number of GFP domains in the construct (0-6).
            k_fold: Folding rate constant in s⁻¹.
            f_base: Baseline foldable fraction (default 0.4).
            f_gain: Foldable fraction increase per additional GFP (default 0.1).
            
        Returns:
            float: CoF efficiency as percentage (0-100%).
        """
        if n_gfp == 0:
            return 0.0
        
        # Fraction of chains that can fold increases with n_gfp (avidity effect)
        f_eff = min(1.0, f_base + f_gain * (n_gfp - 1))
        
        # Sum folded fractions for all domains, weighted by time available
        folded = sum(self.folded_fraction(self.params.time_window(p), k_fold)
                   for p in range(1, n_gfp + 1))
        
        return f_eff * (folded / n_gfp) * 100


class OnePoolModel(CoFModel):
    """One-Pool model with pure kinetics (null hypothesis).
    
    This model assumes ALL nascent chains are competent for co-translational
    folding (f_eff = 1.0). It serves as the null hypothesis to test whether
    pure kinetics alone can explain the experimental data.
    
    The model predicts that CoF efficiency should DECREASE with more GFP domains
    because later-synthesized domains have progressively shorter time windows
    to fold before translation terminates.
    
    Parameters:
        - k_fold: Folding rate constant (s⁻¹) - single free parameter
    
    Note:
        This model fails to reproduce experimental data, indicating
        that pure kinetics alone cannot explain the observed trend.
    """
    
    def efficiency(self, n_gfp, k_fold):
        """Calculate CoF efficiency for a given number of GFP domains.
        
        Args:
            n_gfp: Number of GFP domains in the construct (0-6).
            k_fold: Folding rate constant in s⁻¹.
            
        Returns:
            float: CoF efficiency as percentage (0-100%).
        """
        if n_gfp == 0:
            return 0.0
        
        # All chains are foldable (pure kinetic model)
        f_eff = 1.0
        
        # Sum folded fractions for all domains, weighted by time available
        folded = sum(self.folded_fraction(self.params.time_window(p), k_fold)
                   for p in range(1, n_gfp + 1))
        
        return f_eff * (folded / n_gfp) * 100


# =============================================================================
# FITTING
# =============================================================================

def load_data(filepath):
    """Load experimental CoF efficiency data from Excel file.
    
    Parses an Excel file containing co-translational folding efficiency
    measurements organized by reporter variant. Extracts individual cell
    measurements for each GFP copy number (0-6).
    
    Args:
        filepath: Path to the Excel file containing experimental data.
            Expected file structure:
            - Row 3: Reporter variant names (e.g., pUB-RBsmHA-6xsfGFP-24xMS2)
            - Row 5: Column type headers ("ML 0.5" = individual cell data)
            - Rows 6+: Individual cell CoF Efficiency values
    
    Returns:
        dict: Dictionary mapping n_gfp (0-6) to numpy arrays of efficiency values.
            Example: {0: array([...]), 1: array([...]), ..., 6: array([...])}
    
    Example:
        >>> data = load_data('Dark mCh Cells.xlsx')
        >>> print(f"Found {len(data[6])} cells with 6xGFP")
        >>> print(f"Mean efficiency at 6xGFP: {np.mean(data[6]):.1f}%")
    """
    import re
    
    df = pd.read_excel(filepath, header=None)
    
    gfp_data = {n: [] for n in range(7)}
    
    for col_idx in range(1, df.shape[1]):
        # Get column header from row 5
        header = str(df.iloc[5, col_idx]) if pd.notna(df.iloc[5, col_idx]) else ''
        
        # Only process ML 0.5 columns
        if 'ML 0.5' not in header:
            continue
        
        # Get reporter variant from row 3
        reporter = str(df.iloc[3, col_idx]) if pd.notna(df.iloc[3, col_idx]) else ''
        
        # Parse GFP count from reporter variant
        if '6xsfGFP' in reporter:
            n_gfp = 6
        elif '5xsfGFP' in reporter:
            n_gfp = 5
        elif '4xsfGFP' in reporter:
            n_gfp = 4
        elif '3xsfGFP' in reporter:
            n_gfp = 3
        elif '2xsfGFP' in reporter:
            n_gfp = 2
        elif '1xsfGFP' in reporter:
            n_gfp = 1
        elif '6xmCh' in reporter and 'sfGFP' not in reporter:
            n_gfp = 0
        else:
            continue
        
        # Extract data from rows 6+ (data starts at row 6 after header cleanup)
        col_data = df.iloc[6:, col_idx].dropna()
        for v in col_data:
            if isinstance(v, (int, float)) and v > 0:
                gfp_data[n_gfp].append(float(v))
    
    # Convert to numpy arrays
    results = {n: np.array(vals) for n, vals in gfp_data.items()}
    
    return results


def weighted_mse(pred_list, obs_list, sem_list):
    """Calculate weighted mean squared error using SEM as weights.
    
    Computes sum of squared residuals weighted by inverse variance (1/SEM²).
    This gives more weight to data points with smaller measurement uncertainty.
    
    Args:
        pred_list: List of predicted values.
        obs_list: List of observed values.
        sem_list: List of standard errors of the mean for each observation.
        
    Returns:
        float: Weighted sum of squared residuals.
    """
    return sum(((p - o) / s)**2 for p, o, s in zip(pred_list, obs_list, sem_list))



def fit_two_pool(exp_data):
    """
    Fit Two-Pool model with biologically-constrained parameter bounds.
    
    Parameters (3 free parameters):
    --------------------------------
    k_fold: Folding rate constant
        τ_fold = 30-250s (GFP folding time range)
        - sfGFP folding: ~30-60s in vitro (Pédelacq et al., 2006)
        - In-cell folding can be slower due to crowding: 60-180s
        - Extended upper bound to 250s for comprehensive exploration
    
    f_base: Baseline foldable fraction (0.2-0.8)
        Fraction of nascent chains competent for folding at n=1
    
    f_gain: Gain per additional GFP domain (0.01-0.20)
        Increase in foldable fraction per additional GFP copy
    """
    model = TwoPoolModel()
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def objective(params):
        k_fold = 10**params[0]
        f_base, f_gain = params[1], params[2]
        pred = [model.efficiency(n, k_fold, f_base, f_gain) for n in range(1, 7)]
        return weighted_mse(pred, obs, sem)
    
    # Parameter bounds:
    # k_fold: log10(0.004) = -2.40 to log10(0.033) = -1.48  (τ = 30-250s)
    # f_base: 0.2-0.8 (baseline foldable fraction)
    # f_gain: 0.01-0.20 (gain per additional GFP)
    bounds = [(-2.40, -1.48), (0.2, 0.8), (0.01, 0.20)]
    result = differential_evolution(objective, bounds, maxiter=1000, seed=42, polish=True, disp=False)
    
    k_fold = 10**result.x[0]
    f_base, f_gain = result.x[1], result.x[2]
    tau_fold = 1/k_fold
    preds = {n: (model.efficiency(n, k_fold, f_base, f_gain) if n > 0 else 0) for n in range(7)}
    
    return {
        'k_fold': k_fold, 
        'tau_fold': tau_fold,
        'f_base': f_base, 'f_gain': f_gain,
        'params': result.x, 'cost': result.fun, 'predictions': preds,
        'model_name': 'TwoPool',
        'n_params': 3,
        'bounds_info': {
            'tau_fold_range': '30-250s',
            'f_base_range': '0.2-0.8',
            'f_gain_range': '0.01-0.20'
        }
    }



def fit_one_pool(exp_data):
    """
    Fit One-Pool (pure kinetic) model with EXTENDED parameter bounds.
    
    This is the null hypothesis: all nascent chains are competent for folding.
    The model has only 1 free parameter: k_fold.
    
    Kinetic constraints:
    - τ_fold: 30-250s (extended range)
    """
    model = OnePoolModel()
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def objective(params):
        k_fold = 10**params[0]
        pred = [model.efficiency(n, k_fold) for n in range(1, 7)]
        return weighted_mse(pred, obs, sem)
    
    # τ_fold: 30-250s → k_fold: 0.004-0.033 s⁻¹ → log10: -2.40 to -1.48
    bounds = [(-2.40, -1.48)]
    result = differential_evolution(objective, bounds, maxiter=500, seed=42, polish=True, disp=False)
    
    k_fold = 10**result.x[0]
    tau_fold = 1/k_fold
    preds = {n: (model.efficiency(n, k_fold) if n > 0 else 0) for n in range(7)}
    
    return {
        'k_fold': k_fold,
        'tau_fold': tau_fold,
        'params': result.x, 'cost': result.fun, 'predictions': preds,
        'model_name': 'OnePool',
        'n_params': 1,
        'bounds_info': {
            'tau_fold_range': '30-250s (extended range)'
        }
    }



def twopool_sensitivity_analysis(exp_data, best_params, chi2_threshold=None):
    """
    Perform parameter sensitivity analysis on the Two-Pool model.
    
    For each parameter, sweep through a range of values while holding other
    parameters fixed, to determine how sensitive the fit is to each parameter.
    
    Uses χ² (weighted MSE) for consistency with the fitting objective function.
    
    Parameters:
    -----------
    exp_data : dict
        Experimental data
    best_params : dict
        Best-fit parameters from Two-Pool model fitting
    chi2_threshold : float
        Maximum χ² to consider as "acceptable" fit. If None, uses 2x best χ².
    
    Returns:
    --------
    dict with sensitivity analysis results
    """
    model = TwoPoolModel()
    
    # Best fit values
    k_fold_best = best_params['k_fold']
    f_base_best = best_params['f_base']
    f_gain_best = best_params['f_gain']
    
    # Get experimental data for χ² calculation
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def compute_chi2(k_fold, f_base, f_gain):
        preds = [model.efficiency(n, k_fold, f_base, f_gain) for n in range(1, 7)]
        return weighted_mse(preds, obs, sem)
    
    best_chi2 = compute_chi2(k_fold_best, f_base_best, f_gain_best)
    
    # Default threshold: 2x best χ²
    if chi2_threshold is None:
        chi2_threshold = best_chi2 * 2
    
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nBest-fit χ² = {best_chi2:.2f}")
    print(f"Threshold for acceptable fit: χ² ≤ {chi2_threshold:.2f}")
    
    results = {}
    
    # 1. SENSITIVITY TO τ_fold (k_fold)
    print("\n" + "-"*60)
    print("τ_fold (GFP folding time) sensitivity:")
    print("-"*60)
    
    tau_fold_values = np.logspace(np.log10(20), np.log10(300), 50)  # 20-300s range
    tau_fold_chi2 = []
    for tau in tau_fold_values:
        k_f = 1/tau
        chi2 = compute_chi2(k_f, f_base_best, f_gain_best)
        tau_fold_chi2.append(chi2)
    
    # Find acceptable range (χ² ≤ threshold)
    acceptable_tau_fold = tau_fold_values[np.array(tau_fold_chi2) <= chi2_threshold]
    if len(acceptable_tau_fold) > 0:
        tau_fold_min, tau_fold_max = acceptable_tau_fold.min(), acceptable_tau_fold.max()
    else:
        tau_fold_min, tau_fold_max = np.nan, np.nan
    
    print(f"  Best value: τ_fold = {1/k_fold_best:.1f} s")
    print(f"  Acceptable range (χ² ≤ {chi2_threshold:.1f}): {tau_fold_min:.1f} - {tau_fold_max:.1f} s")
    
    results['tau_fold'] = {
        'best': 1/k_fold_best,
        'min': tau_fold_min,
        'max': tau_fold_max,
        'sweep_values': tau_fold_values.tolist(),
        'sweep_chi2': tau_fold_chi2
    }
    
    # 2. SENSITIVITY TO f_base
    print("\n" + "-"*60)
    print("f_base (baseline foldable fraction) sensitivity:")
    print("-"*60)
    
    f_base_values = np.linspace(0.1, 0.9, 50)
    f_base_chi2 = []
    for fb in f_base_values:
        chi2 = compute_chi2(k_fold_best, fb, f_gain_best)
        f_base_chi2.append(chi2)
    
    acceptable_f_base = f_base_values[np.array(f_base_chi2) <= chi2_threshold]
    if len(acceptable_f_base) > 0:
        f_base_min, f_base_max = acceptable_f_base.min(), acceptable_f_base.max()
    else:
        f_base_min, f_base_max = np.nan, np.nan
    
    print(f"  Best value: f_base = {f_base_best:.2f}")
    print(f"  Acceptable range (χ² ≤ {chi2_threshold:.1f}): {f_base_min:.2f} - {f_base_max:.2f}")
    
    results['f_base'] = {
        'best': f_base_best,
        'min': f_base_min,
        'max': f_base_max,
        'sweep_values': f_base_values.tolist(),
        'sweep_chi2': f_base_chi2
    }
    
    # 3. SENSITIVITY TO f_gain
    print("\n" + "-"*60)
    print("f_gain (foldable fraction gain per GFP) sensitivity:")
    print("-"*60)
    
    f_gain_values = np.linspace(0.01, 0.3, 50)
    f_gain_chi2 = []
    for fg in f_gain_values:
        chi2 = compute_chi2(k_fold_best, f_base_best, fg)
        f_gain_chi2.append(chi2)
    
    acceptable_f_gain = f_gain_values[np.array(f_gain_chi2) <= chi2_threshold]
    if len(acceptable_f_gain) > 0:
        f_gain_min, f_gain_max = acceptable_f_gain.min(), acceptable_f_gain.max()
    else:
        f_gain_min, f_gain_max = np.nan, np.nan
    
    print(f"  Best value: f_gain = {f_gain_best:.2f}")
    print(f"  Acceptable range (χ² ≤ {chi2_threshold:.1f}): {f_gain_min:.2f} - {f_gain_max:.2f}")
    
    results['f_gain'] = {
        'best': f_gain_best,
        'min': f_gain_min,
        'max': f_gain_max,
        'sweep_values': f_gain_values.tolist(),
        'sweep_chi2': f_gain_chi2
    }
    
    # Summary table
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("="*80)
    print(f"\n{'Parameter':<15} {'Best Value':<15} {'Acceptable Range':<25} {'Sensitivity':<15}")
    print("-"*70)
    
    for param_name, param_data in results.items():
        best_val = param_data['best']
        range_min = param_data['min']
        range_max = param_data['max']
        
        # Calculate relative range as sensitivity metric
        if not np.isnan(range_min) and not np.isnan(range_max):
            rel_range = (range_max - range_min) / best_val * 100
            if param_name.startswith('f_'):
                range_str = f"{range_min:.2f} - {range_max:.2f}"
                best_str = f"{best_val:.2f}"
            elif param_name == 'scale':
                range_str = f"{range_min:.2f} - {range_max:.2f}"
                best_str = f"{best_val:.2f}"
            else:
                range_str = f"{range_min:.1f} - {range_max:.1f} s"
                best_str = f"{best_val:.1f} s"
            
            if rel_range < 50:
                sensitivity = "HIGH"
            elif rel_range < 150:
                sensitivity = "MODERATE"
            else:
                sensitivity = "LOW"
        else:
            range_str = "N/A"
            sensitivity = "N/A"
            best_str = f"{best_val:.2f}"
        
        print(f"{param_name:<15} {best_str:<15} {range_str:<25} {sensitivity:<15}")
    
    return results


def create_twopool_sensitivity_figure(sensitivity_results, save_path):
    """Create a figure showing parameter sensitivity analysis - matches old Fig 6 style.
    
    Uses 1x3 horizontal layout with:
    - Blue sensitivity curves showing χ² (weighted MSE)
    - Green dashed vertical lines for best-fit values
    - Green shading for acceptable χ² regions
    - Panel labels A, B, C
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    
    param_order = ['tau_fold', 'f_base', 'f_gain']
    param_labels = {
        'tau_fold': (r'$\tau_{fold}$ (s)', 'A'),
        'f_base': (r'$f_{base}$', 'B'),
        'f_gain': (r'$f_{gain}$ (domain$^{-1}$)', 'C'),
    }
    
    # Get the best χ² value to determine threshold for shading
    best_chi2 = min(sensitivity_results['tau_fold']['sweep_chi2'])
    chi2_threshold = best_chi2 * 2  # Same threshold as in sensitivity analysis
    
    for idx, param_name in enumerate(param_order):
        if param_name not in sensitivity_results:
            continue
            
        ax = axes[idx]
        data = sensitivity_results[param_name]
        
        x_vals = data['sweep_values']
        y_vals = data['sweep_chi2']
        best_val = data['best']
        
        # Red line for sensitivity curve
        ax.plot(x_vals, y_vals, '-', color='red', lw=2.5)
        
        # Horizontal threshold line (gray dashed)
        ax.axhline(y=chi2_threshold, color='gray', linestyle='--', lw=1.5)
        
        # Vertical best-fit line (black dashed)
        ax.axvline(x=best_val, color='black', linestyle='--', lw=2)
        
        # Light red shading for acceptable region (χ² <= threshold)
        acceptable_mask = np.array(y_vals) <= chi2_threshold
        if any(acceptable_mask):
            ax.fill_between(x_vals, 0, y_vals, where=acceptable_mask, 
                          alpha=0.2, color='red')
        
        xlabel, panel_label = param_labels[param_name]
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('χ²', fontsize=11)
        ax.set_ylim(0, max(y_vals) * 1.1)
        ax.set_xlim(min(x_vals), max(x_vals))
        
        # Panel label (A, B, C) outside subplot - above and to the left
        ax.text(-0.1, 1.1, panel_label, transform=ax.transAxes, fontsize=14, 
                fontweight='bold', va='bottom', ha='left')
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSensitivity figure saved: {save_path}")
    return fig



# =============================================================================
# VISUALIZATION
# =============================================================================

def create_onepool_sensitivity_figure(exp_data, result, save_path):
    """Create One-Pool model parameter analysis figure (Fig 3).
    
    Since the model now has only 1 parameter (τ_fold), this shows
    a single sensitivity curve using χ² (weighted MSE) for consistency
    with the fitting objective function.
    """
    model = OnePoolModel()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    
    # Get experimental data for χ² calculation
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    # Helper function to get predictions using OnePoolModel
    def get_onepool_predictions(k_fold):
        return [model.efficiency(n, k_fold) for n in range(1, 7)]
    
    # Sweep tau_fold and calculate χ² (weighted MSE)
    tau_fold_range = np.linspace(30, 250, 50)
    chi2_fold = []
    for tau in tau_fold_range:
        pred = get_onepool_predictions(1/tau)
        chi2_fold.append(weighted_mse(pred, obs, sem))
    
    # Find minimum χ² and corresponding tau
    min_chi2 = min(chi2_fold)
    best_tau_idx = chi2_fold.index(min_chi2)
    best_tau = tau_fold_range[best_tau_idx]
    
    ax.plot(tau_fold_range, chi2_fold, 'r-', linewidth=2.5)
    ax.axvline(x=result['tau_fold'], color='black', linestyle=':', linewidth=2)
    ax.set_xlabel('τ_fold (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('χ² (weighted MSE)', fontsize=12, fontweight='bold')
    ax.set_title('One-Pool Model: τ_fold Sensitivity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add best-fit marker at the minimum
    chi2_at_best = weighted_mse(get_onepool_predictions(result['k_fold']), obs, sem)
    # Mark the minimum point with a red dot
    ax.scatter([result['tau_fold']], [chi2_at_best], color='red', s=100, zorder=5, marker='o', edgecolors='black', linewidths=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"One-Pool parameter analysis saved: {save_path}")



def create_twopool_parameter_space(exp_data, result, save_path):
    """Create Two-Pool model parameter space analysis figure (Fig 5).
    
    Shows pairwise parameter combinations for the 3 model parameters:
    τ_fold, f_base, f_gain
    
    Uses χ² (weighted MSE) for consistency with the fitting objective function.
    """
    model = TwoPoolModel()
    
    # Get experimental data for χ² calculation
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    # Helper function to get predictions using TwoPoolModel
    def get_twopool_predictions(k_fold, f_base, f_gain):
        return [model.efficiency(n, k_fold, f_base, f_gain) for n in range(1, 7)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Define parameter ranges
    tau_fold_range = np.linspace(30, 100, 30)
    f_base_range = np.linspace(0.2, 0.6, 30)
    f_gain_range = np.linspace(0.05, 0.20, 30)
    
    # Panel A: tau_fold vs f_base
    ax = axes[0]
    chi2_grid = np.zeros((len(f_base_range), len(tau_fold_range)))
    for i, fb in enumerate(f_base_range):
        for j, tf in enumerate(tau_fold_range):
            pred = get_twopool_predictions(1/tf, fb, result['f_gain'])
            chi2_grid[i, j] = weighted_mse(pred, obs, sem)
    im = ax.contourf(tau_fold_range, f_base_range, chi2_grid, levels=20, cmap='viridis_r')
    ax.plot(result['tau_fold'], result['f_base'], 'r*', markersize=15, markeredgecolor='white')
    ax.set_xlabel('τ_fold (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('f_base', fontsize=11, fontweight='bold')
    ax.set_title('(A) τ_fold vs f_base', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='χ²')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel B: tau_fold vs f_gain
    ax = axes[1]
    chi2_grid = np.zeros((len(f_gain_range), len(tau_fold_range)))
    for i, fg in enumerate(f_gain_range):
        for j, tf in enumerate(tau_fold_range):
            pred = get_twopool_predictions(1/tf, result['f_base'], fg)
            chi2_grid[i, j] = weighted_mse(pred, obs, sem)
    im = ax.contourf(tau_fold_range, f_gain_range, chi2_grid, levels=20, cmap='viridis_r')
    ax.plot(result['tau_fold'], result['f_gain'], 'r*', markersize=15, markeredgecolor='white')
    ax.set_xlabel('τ_fold (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('f_gain', fontsize=11, fontweight='bold')
    ax.set_title('(B) τ_fold vs f_gain', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='χ²')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel C: f_base vs f_gain
    ax = axes[2]
    chi2_grid = np.zeros((len(f_gain_range), len(f_base_range)))
    for i, fg in enumerate(f_gain_range):
        for j, fb in enumerate(f_base_range):
            pred = get_twopool_predictions(result['k_fold'], fb, fg)
            chi2_grid[i, j] = weighted_mse(pred, obs, sem)
    im = ax.contourf(f_base_range, f_gain_range, chi2_grid, levels=20, cmap='viridis_r')
    ax.plot(result['f_base'], result['f_gain'], 'r*', markersize=15, markeredgecolor='white')
    ax.set_xlabel('f_base', fontsize=11, fontweight='bold')
    ax.set_ylabel('f_gain', fontsize=11, fontweight='bold')
    ax.set_title('(C) f_base vs f_gain', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='χ²')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle('Two-Pool Model: Parameter Space Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Two-Pool parameter space saved: {save_path}")


def create_onepool_comprehensive_figure(exp_data, result, save_path):
    """
    Create a comprehensive One-Pool figure matching the old Fig 2 style:
    - Top row: One-Pool model schematic + Model fit + Parameters
    - Bottom row: Time window diagram
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7], 
                  width_ratios=[0.9, 1.3, 0.8], hspace=0.30, wspace=0.20)
    
    params = TranslationParameters()
    k_fold = result['k_fold']
    tau_fold = result['tau_fold']
    
    # Color palette for domains (green -> yellow -> orange -> red)
    domain_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    
    # ==================== TOP ROW ====================
    
    # --- Panel A: One-Pool Mechanistic Diagram ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.7, 'A. One-Pool Model', fontsize=12, fontweight='bold', ha='center')
    
    # Nascent chain population box (all foldable) - larger box, lightgrey color
    pop_box = FancyBboxPatch((0.1, 4.5), 4.0, 5.0, boxstyle="round,pad=0.1",
                              facecolor='lightgrey', edgecolor='gray', linewidth=2, alpha=0.9)
    ax.add_patch(pop_box)
    ax.text(2.1, 9.0, 'Nascent Chains', fontsize=10, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(2.1, 7.0, '100% foldable', fontsize=10, ha='center', color='#2c3e50')
    
    # Arrow from population to kinetic pathway
    ax.annotate('', xy=(4.8, 7.5), xytext=(4.1, 7.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2.5))
    
    # Kinetic pathway: U -> F (simplified model)
    state_y = 7.5
    ax.text(5.5, state_y, 'U', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgray', edgecolor='gray', lw=2),
            color='white')
    ax.annotate('', xy=(7.0, state_y), xytext=(6.1, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.5, state_y + 0.5, r'$k_{fold}$', fontsize=10, ha='center', fontweight='bold')
    
    ax.text(7.6, state_y, 'F', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', edgecolor='darkred', lw=2),
            color='white')
    
    ax.text(5.5, state_y - 1.1, 'Unfolded', fontsize=8, ha='center', color='#555')
    ax.text(7.6, state_y - 1.1, 'Folded', fontsize=8, ha='center', color='#555')
    
    # Simple label (no equation - see Eq. 9 in text)
    ax.text(5, 2.8, 'Pure kinetics only', fontsize=10, ha='center', fontstyle='italic', color='#555')
    
    # --- Panel B: Model Fit ---
    ax = fig.add_subplot(gs[0, 1])
    
    n_vals = list(range(7))
    exp_means = [np.mean(exp_data[n]) for n in n_vals]
    exp_sems = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in n_vals]
    preds = [result['predictions'][n] for n in n_vals]
    
    # Gray jittered individual data points
    for n in range(7):
        jitter = np.random.normal(0, 0.05, len(exp_data[n]))
        ax.scatter([n + j for j in jitter], exp_data[n], alpha=0.35, color='lightgray', s=18, zorder=1)
    
    ax.errorbar(n_vals, exp_means, yerr=exp_sems, fmt='o', capsize=5, capthick=2, 
                color='dimgray', markersize=10, label='Data (mean ± SEM)', zorder=3, linewidth=2)
    ax.scatter(n_vals, preds, color='red', s=200, zorder=4, marker='+', linewidths=3.5,
               label='One-Pool Model')
    
    ax.set_xlabel('Number of GFP Domains', fontsize=12, fontweight='bold')
    ax.set_ylabel('CoF Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Model Fit', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- Panel C: Parameters Table ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title('C. Fitted Parameters', fontsize=13, fontweight='bold')
    
    k_fold_val = 1/tau_fold
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        [r'$k_{fold}$', f'{k_fold_val:.4f}', r's$^{-1}$'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.30, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 2.5)
    
    for j in range(3):
        table[(0, j)].set_facecolor('dimgray')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, 2):
        for j in range(3):
            table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    # ==================== BOTTOM ROW: Time Window Diagram ====================
    ax = fig.add_subplot(gs[1, :])
    
    ax.set_xlim(-30, 420)
    ax.set_ylim(-6.5, 2)
    ax.axis('off')
    ax.set_title('D. Translation Timeline: Time Windows for Folding', fontsize=13, fontweight='bold', pad=10)
    
    T_total = params.T_total
    scale = 360 / T_total
    
    # Main timeline arrow
    ax.arrow(0, 0, 385, 0, head_width=0.25, head_length=5, fc='black', ec='black', linewidth=2)
    ax.text(392, 0, 'Time (s)', fontsize=10, va='center')
    
    ax.plot([0, 0], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(0, 0.5, '0s', fontsize=9, ha='center', va='bottom')
    
    tau_total = tau_fold
    
    for i in range(1, 6):  # Only GFP1-5, GFP6 combined with Stop
        t_emerge = params.domain_emergence_time(i)
        x = t_emerge * scale
        ax.plot([x, x], [-0.25, 0.25], 'k-', linewidth=1.5)
        ax.text(x, 0.5, f'{t_emerge:.0f}s', fontsize=8, ha='center', va='bottom')
        ax.text(x, 1.1, f'GFP{i}', fontsize=9, ha='center', va='bottom', 
                color=domain_colors[i-1], fontweight='bold')
    
    x_term = T_total * scale
    ax.plot([x_term, x_term], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(x_term, 0.5, f'{T_total:.0f}s', fontsize=9, ha='center', va='bottom')
    ax.text(x_term, 1.1, 'GFP6/Stop', fontsize=9, ha='center', va='bottom', color='red', fontweight='bold')
    
    # Time window bars
    for i in range(1, 7):
        t_emerge = params.domain_emergence_time(i)
        tw = params.time_window(i)
        x_start = t_emerge * scale
        x_end = T_total * scale
        y = -0.8 - (i-1) * 0.85
        
        ax.plot([x_start, x_end], [y, y], color=domain_colors[i-1], linewidth=10, solid_capstyle='butt')
        ax.text(-25, y, f'GFP{i}:', fontsize=10, ha='right', va='center', 
                color=domain_colors[i-1], fontweight='bold')
        
        # Time window value and status (matching Fig 4 style)
        if tw > tau_total:
            status = 'ok'
            status_color = 'green'
        else:
            status = 'x'
            status_color = 'red'
        ax.text(x_end + 8, y, f'{tw:.0f}s {status}', fontsize=10, ha='left', va='center',
                color=status_color, fontweight='bold')
    

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"One-Pool comprehensive figure saved: {save_path}")


def create_twopool_comprehensive_figure(exp_data, best, save_path):
    """
    Create a comprehensive figure combining:
    - Top row: Two-Pool model schematic + Model fit + Parameters (like cof_summary_figure)
    - Bottom row: Time window diagram showing dependence on k_elong, k_init
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(15, 9))
    # Panel A wider, Panel B same, Panel C narrower
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7], 
                  width_ratios=[1, 1.3, 0.7], hspace=0.30, wspace=0.20)
    
    params = TranslationParameters()
    k_fold = best['k_fold']
    
    # Color palette for domains (green -> yellow -> orange -> red)
    domain_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    
    # ==================== TOP ROW ====================
    
    # --- Panel A: Two-Pool Mechanistic Diagram ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.7, 'A. Two-Pool Model', fontsize=12, fontweight='bold', ha='center')
    
    # Nascent chain population box - wider, lightgray
    pop_box = FancyBboxPatch((0.0, 4.5), 4.5, 5.0, boxstyle="round,pad=0.1",
                              facecolor='lightgray', edgecolor='gray', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(2.25, 9.0, 'Nascent Chains', fontsize=10, fontweight='bold', ha='center', color='#2c3e50')
    
    # Foldable pool
    fold_box = FancyBboxPatch((0.3, 7.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                               facecolor='#27ae60', edgecolor='#1e8449', linewidth=2, alpha=0.9)
    ax.add_patch(fold_box)
    ax.text(2.25, 7.8, 'Foldable', fontsize=11, fontweight='bold', ha='center', color='white')
    
    # Non-foldable pool
    non_box = FancyBboxPatch((0.3, 5.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                              facecolor='#7f8c8d', edgecolor='#5d6d7e', linewidth=2, alpha=0.9)
    ax.add_patch(non_box)
    ax.text(2.25, 5.8, 'Non-Foldable', fontsize=11, fontweight='bold', ha='center', color='white')
    
    # Arrow from foldable to kinetic pathway
    ax.annotate('', xy=(5.0, 7.5), xytext=(4.3, 7.8),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    
    # Kinetic pathway: U -> F (simplified model)
    state_y = 7.5
    ax.text(5.5, state_y, 'U', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgray', edgecolor='gray', lw=2),
            color='white')
    ax.annotate('', xy=(7.0, state_y), xytext=(6.1, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.5, state_y + 0.5, r'$k_{fold}$', fontsize=10, ha='center', fontweight='bold')
    
    ax.text(7.6, state_y, 'F', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', edgecolor='darkred', lw=2),
            color='white')
    
    ax.text(5.5, state_y - 1.1, 'Unfolded', fontsize=8, ha='center', color='#555')
    ax.text(7.6, state_y - 1.1, 'Folded', fontsize=8, ha='center', color='#555')
    
    # Simple label (no equations - see Eq. 10 in text)
    ax.text(5, 2.8, 'Population heterogeneity + kinetics', fontsize=10, ha='center', fontstyle='italic', color='#555')
    
    # --- Panel B: Model Fit (WIDER) ---
    ax = fig.add_subplot(gs[0, 1])
    
    n_vals = list(range(7))
    exp_means = [np.mean(exp_data[n]) for n in n_vals]
    exp_sems = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in n_vals]
    preds = [best['predictions'][n] for n in n_vals]
    
    for n in range(7):
        jitter = np.random.normal(0, 0.05, len(exp_data[n]))
        ax.scatter([n + j for j in jitter], exp_data[n], alpha=0.35, color='lightgray', s=18, zorder=1)
    
    ax.errorbar(n_vals, exp_means, yerr=exp_sems, fmt='o', capsize=5, capthick=2, 
                color='dimgray', markersize=10, label='Data (mean ± SEM)', zorder=3, linewidth=2)
    ax.scatter(n_vals, preds, color='red', s=200, zorder=4, marker='+', linewidths=3.5,
               label='Two-Pool Model')
    
    ax.set_xlabel('Number of GFP Domains', fontsize=12, fontweight='bold')
    ax.set_ylabel('CoF Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Model Fit', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- Panel C: Parameters Table ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title('C. Fitted Parameters', fontsize=13, fontweight='bold')
    
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        [r'$k_{fold}$', f'{k_fold:.4f}', r's$^{-1}$'],
        [r'$f_{base}$', f'{best["f_base"]:.2f}', '-'],
        [r'$f_{gain}$', f'{best["f_gain"]:.2f}', '/dom'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.30, 0.25, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.2)
    
    for j in range(3):
        table[(0, j)].set_facecolor('dimgray')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, 4):
        for j in range(3):
            table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    # ==================== BOTTOM ROW: Time Window Diagram ====================
    ax = fig.add_subplot(gs[1, :])  # Span all columns
    
    ax.set_xlim(-30, 420)
    ax.set_ylim(-6.5, 2)
    ax.axis('off')
    ax.set_title('D. Translation Timeline: Time Windows for Folding', fontsize=13, fontweight='bold', pad=10)
    
    # Parameters
    T_total = params.T_total
    scale = 360 / T_total  # pixels per second
    
    # Main timeline arrow
    ax.arrow(0, 0, 385, 0, head_width=0.25, head_length=5, fc='black', ec='black', linewidth=2)
    ax.text(392, 0, 'Time (s)', fontsize=10, va='center')
    
    # Time tick marks
    ax.plot([0, 0], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(0, 0.5, '0s', fontsize=9, ha='center', va='bottom')
    
    positions = list(range(1, 7))
    tau_total = 1/k_fold
    
    for i, pos in enumerate(positions[:5]):  # Only GFP1-5, GFP6 combined with Stop
        t_emerge = params.domain_emergence_time(pos)
        x = t_emerge * scale
        ax.plot([x, x], [-0.25, 0.25], 'k-', linewidth=1.5)
        ax.text(x, 0.5, f'{t_emerge:.0f}s', fontsize=8, ha='center', va='bottom')
        ax.text(x, 1.1, f'GFP{pos}', fontsize=9, ha='center', va='bottom', 
                color=domain_colors[i], fontweight='bold')
    
    # Termination line - combined GFP6/Stop label
    x_term = T_total * scale
    ax.plot([x_term, x_term], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(x_term, 0.5, f'{T_total:.0f}s', fontsize=9, ha='center', va='bottom')
    ax.text(x_term, 1.1, 'GFP6/Stop', fontsize=9, ha='center', va='bottom', color='red', fontweight='bold')
    
    # Time window bars
    for i, pos in enumerate(positions):
        t_emerge = params.domain_emergence_time(pos)
        tw = params.time_window(pos)
        x_start = t_emerge * scale
        x_end = T_total * scale
        y = -0.8 - i * 0.85
        
        # Bar
        ax.plot([x_start, x_end], [y, y], color=domain_colors[i], linewidth=10, solid_capstyle='butt')
        
        # Label
        ax.text(-25, y, f'GFP{pos}:', fontsize=10, ha='right', va='center', 
                color=domain_colors[i], fontweight='bold')
        
        # Time window value and status
        if tw > tau_total:
            status = 'ok'
            status_color = 'green'
        else:
            status = 'x'
            status_color = 'red'
        ax.text(x_end + 8, y, f'{tw:.0f}s {status}', fontsize=10, ha='left', va='center',
                color=status_color, fontweight='bold')
    

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive figure saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("CO-TRANSLATIONAL FOLDING - MECHANISTIC MODEL FITTING")
    print("="*70)
    
    params = TranslationParameters()
    
    # Report translation parameters and polysome context
    print("\n" + "-"*70)
    print("TRANSLATION PARAMETERS (FIXED FROM LITERATURE)")
    print("-"*70)
    print(f"  k_elong (elongation rate):     {params.k_elongation} aa/s")
    print(f"  k_init (initiation rate):      {params.k_init} s⁻¹  (1 ribosome every {1/params.k_init:.0f}s)")
    print(f"  L_total (construct length):    {params.L_total} aa")
    print(f"  Ribosome footprint:            {params.ribosome_footprint} aa")
    
    print("\n" + "-"*70)
    print("POLYSOME CONTEXT (DERIVED)")
    print("-"*70)
    print(f"  Total translation time:        {params.T_total:.1f}s ({params.T_total/60:.1f} min)")
    print(f"  Mean ribosome spacing:         {params.mean_ribosome_spacing:.0f} aa")
    print(f"  Avg ribosomes per mRNA:        {params.avg_ribosomes_per_mRNA:.1f} ribosomes")
    
    print("\nDomain Time Windows (time available for folding):")
    for p in range(1, 7):
        print(f"  Position {p}: {params.time_window(p):.0f}s")
    
    data_path = Path(__file__).parent / "Dark mCh Cells.xlsx"
    print(f"\nLoading: {data_path}")
    exp_data = load_data(str(data_path))
    
    print("\nExperimental Data:")
    for n in range(7):
        v = exp_data[n]
        print(f"  {n}xGFP: {np.mean(v):.1f}% ± {np.std(v)/np.sqrt(len(v)):.1f}% (n={len(v)})")
    
    # Fit models
    print("\n" + "="*70)
    print("FITTING MODELS")
    print("="*70)
    
    print("\n[1/2] One-Pool Model (pure kinetic)...")
    print("    Assumption: All nascent chains are competent for folding (f_eff = 1.0)")
    res_one_pool = fit_one_pool(exp_data)
    print(f"    χ² = {res_one_pool['cost']:.1f}")
    print(f"    τ_fold = {res_one_pool['tau_fold']:.1f}s")
    
    print("\n[2/2] Two-Pool Model...")
    print("    Bounds: τ_fold = 30-250s (GFP folding)")
    res_pool = fit_two_pool(exp_data)
    print(f"    χ² = {res_pool['cost']:.1f}")
    print(f"    τ_fold = {res_pool['tau_fold']:.1f}s")
    print(f"    f_base = {res_pool['f_base']:.2f}")
    print(f"    f_gain = {res_pool['f_gain']:.2f}")
    
    # Model comparison (χ² only)
    print("\n" + "="*70)
    print("MODEL COMPARISON: ONE-POOL vs TWO-POOL")
    print("="*70)
    
    print(f"\n{'Model':<20} {'k':>4} {'χ²':>10}")
    print("-"*40)
    print(f"{'One-Pool (kinetic)':<20} {1:>4} {res_one_pool['cost']:>10.1f}")
    print(f"{'Two-Pool (preferred)':<20} {3:>4} {res_pool['cost']:>10.1f}")
    print(f"\n→ Two-Pool model is preferred (lower χ², captures increasing trend)")
    
    # Compare predictions
    print("\n" + "-"*55)
    print(f"{'n':>4} {'Data':>8} {'One-Pool':>10} {'Two-Pool':>10}")
    print("-"*35)
    for n in range(1, 7):
        obs = np.mean(exp_data[n])
        pred_one = res_one_pool['predictions'][n]
        pred_two = res_pool['predictions'][n]
        print(f"{n:>4} {obs:>8.1f} {pred_one:>10.1f} {pred_two:>10.1f}")
    
    # Use Two-Pool model as best
    best = res_pool
    
    print("\n" + "="*70)
    print(f"BEST MODEL: Two-Pool - χ² = {res_pool['cost']:.1f}")
    print("="*70)
    
    print(f"\nFitted Parameters:")
    print(f"  τ_fold = {res_pool['tau_fold']:.1f}s  (GFP folding time)")
    print(f"  f_base = {res_pool['f_base']:.2f}   (baseline foldable fraction)")
    print(f"  f_gain = {res_pool['f_gain']:.2f}   (gain per additional GFP)")
    
    print("\nPredictions vs Data:")
    for n in range(1, 7):
        obs = np.mean(exp_data[n])
        pred = res_pool['predictions'][n]
        print(f"  {n}xGFP: Data={obs:.1f}%, Model={pred:.1f}%, Δ={obs-pred:+.1f}")
    
    # Generate figures
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Fig 2: One-Pool model comprehensive (schematic + fit + params + timeline)
    create_onepool_comprehensive_figure(exp_data, res_one_pool, str(output_dir / "fig2_onepool_solution.png"))
    
    # Fig 3: One-Pool parameter analysis  
    create_onepool_sensitivity_figure(exp_data, res_one_pool, str(output_dir / "fig3_onepool_parameters.png"))
    
    # Fig 4: Two-Pool model comprehensive (schematic + fit + params + timeline)
    create_twopool_comprehensive_figure(exp_data, res_pool, str(output_dir / "fig4_twopool_solution.png"))
    
    # Fig 5: Two-Pool parameter space analysis (2x3 grid)
    create_twopool_parameter_space(exp_data, res_pool, str(output_dir / "fig5_twopool_parameter_space.png"))
    
    # Fig 6: Two-Pool parameter sensitivity (1x4 horizontal)
    sensitivity_results = twopool_sensitivity_analysis(exp_data, res_pool)
    create_twopool_sensitivity_figure(sensitivity_results, str(output_dir / "fig6_twopool_sensitivity.png"))
    
    # Save results
    with open(output_dir / "fit_results.json", 'w') as f:
        json.dump({
            'best_model': 'TwoPool',
            'two_pool': {
                'k_fold': res_pool['k_fold'],
                'tau_fold': res_pool['tau_fold'],
                'f_base': res_pool['f_base'],
                'f_gain': res_pool['f_gain'],
                'chi2': res_pool['cost'],
                'predictions': res_pool['predictions'],
                'n_params': 3
            },
            'one_pool': {
                'k_fold': res_one_pool['k_fold'],
                'tau_fold': res_one_pool['tau_fold'],
                'chi2': res_one_pool['cost'],
                'predictions': res_one_pool['predictions'],
                'n_params': 1
            },
            'translation_params': {
                'k_elong': params.k_elongation,
                'k_init': params.k_init,
                'L_total': params.L_total,
                'T_total': params.T_total,
                'avg_ribosomes': params.avg_ribosomes_per_mRNA,
                'mean_spacing': params.mean_ribosome_spacing
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nResults saved: {output_dir / 'fit_results.json'}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return best

if __name__ == '__main__':
    main()
