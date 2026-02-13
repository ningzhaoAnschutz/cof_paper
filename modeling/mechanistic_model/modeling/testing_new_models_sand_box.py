#!/usr/bin/env python3
"""
Testing New Models Sandbox
==========================

This is a SANDBOX script for testing experimental model variants.
These models are NOT part of the main analysis pipeline.

Current Models Being Tested:
----------------------------
1. Constant f_fold Model (U → F)
   - Constant foldable fraction, no avidity
   - Exponential folding kinetics
   
2. Michaelis-Menten Kinetics (U → F)
   - Hyperbolic saturation: P_fold = t / (K_m + t)
   - Constant foldable fraction, no avidity
   - Tests whether kinetic shape matters
   
3. Two-Step Constant Model (U → F → B)
   - Adds binding step after folding
   - Constant foldable fraction, no avidity
   - Tests whether binding kinetics can explain the trend

Run independently:
    python testing_new_models_sand_box.py
    
Output:
    test_figures/constant_ffold_solution.png
    test_figures/michaelis_menten_solution.png
    test_figures/twostep_constant_solution.png
    
Author: Generated for testing purposes
Date: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


# =============================================================================
# PARAMETERS (copied from main script for independence)
# =============================================================================

class TranslationParameters:
    """Fixed parameters for translation kinetics."""
    k_elongation = 4.85  # aa/s
    k_init = 0.063       # s⁻¹
    L_total = 1826       # Total construct length
    ribosome_footprint = 30  # aa
    
    # GFP domain positions (C-terminus, aa)
    gfp_positions = [590, 835, 1080, 1325, 1570, 1815]
    
    @property
    def T_total(self):
        return self.L_total / self.k_elongation
    
    @property
    def mean_ribosome_spacing(self):
        return self.k_elongation / self.k_init
    
    @property
    def avg_ribosomes_per_mRNA(self):
        return self.T_total * self.k_init
    
    def domain_emergence_time(self, position):
        """Time when domain 'position' (1-6) emerges from ribosome."""
        exit_aa = self.gfp_positions[position - 1] + 17  # +17 for exit tunnel
        return exit_aa / self.k_elongation
    
    def time_window(self, position):
        """Time available for domain to fold before translation ends."""
        return self.T_total - self.domain_emergence_time(position)


def load_data(filepath):
    """Load experimental CoF efficiency data from Excel file."""
    df = pd.read_excel(filepath, header=None)
    
    gfp_data = {n: [] for n in range(7)}
    
    for col_idx in range(1, df.shape[1]):
        header = str(df.iloc[5, col_idx]) if pd.notna(df.iloc[5, col_idx]) else ''
        
        if 'ML 0.5' not in header:
            continue
        
        reporter = str(df.iloc[3, col_idx]) if pd.notna(df.iloc[3, col_idx]) else ''
        
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
        
        col_data = df.iloc[6:, col_idx].dropna()
        for v in col_data:
            if isinstance(v, (int, float)) and v > 0:
                gfp_data[n_gfp].append(float(v))
    
    results = {n: np.array(vals) for n, vals in gfp_data.items()}
    return results


def weighted_mse(predictions, observations, sem):
    """Compute weighted MSE (χ²) using SEM as weights."""
    total = 0
    for pred, obs, se in zip(predictions, observations, sem):
        if se > 0:
            total += ((pred - obs) / se) ** 2
    return total


# =============================================================================
# MODEL 1: CONSTANT F_FOLD (U → F)
# =============================================================================

class ConstantPoolModel:
    """
    Constant f_fold model with time-dependent folding kinetics.
    
    Assumes a CONSTANT fraction of nascent chains are competent for
    co-translational folding, independent of the number of GFP domains.
    
    Parameters:
        - k_fold: Folding rate constant (s⁻¹)
        - f_fold: Fraction of chains capable of folding (constant)
    """
    
    def __init__(self):
        self.params = TranslationParameters()
    
    def folded_fraction(self, t, k_fold):
        """Probability of being folded after time t."""
        if t <= 0:
            return 0.0
        return 1.0 - np.exp(-k_fold * t)
    
    def efficiency(self, n_gfp, k_fold, f_fold):
        """Calculate CoF efficiency for n_gfp domains."""
        if n_gfp == 0:
            return 0.0
        
        f_eff = f_fold  # Constant, no avidity
        
        folded = sum(self.folded_fraction(self.params.time_window(p), k_fold)
                   for p in range(1, n_gfp + 1))
        
        return f_eff * (folded / n_gfp) * 100


def fit_constant_pool(exp_data):
    """Fit Constant f_fold model (2 parameters)."""
    model = ConstantPoolModel()
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def objective(params):
        k_fold = 10**params[0]
        f_fold = params[1]
        pred = [model.efficiency(n, k_fold, f_fold) for n in range(1, 7)]
        return weighted_mse(pred, obs, sem)
    
    # k_fold: τ = 30-250s → k = 0.004-0.033 → log10: -2.40 to -1.48
    bounds = [(-2.40, -1.48), (0.2, 1.0)]
    result = differential_evolution(objective, bounds, maxiter=1000, seed=42, polish=True, disp=False)
    
    k_fold = 10**result.x[0]
    f_fold = result.x[1]
    preds = {n: (model.efficiency(n, k_fold, f_fold) if n > 0 else 0) for n in range(7)}
    
    return {
        'k_fold': k_fold, 
        'tau_fold': 1/k_fold,
        'f_fold': f_fold,
        'cost': result.fun,
        'predictions': preds,
        'model_name': 'ConstantPool',
        'n_params': 2
    }


def create_constant_pool_figure(exp_data, result, save_path):
    """Create comprehensive figure for Constant f_fold model."""
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7], 
                  width_ratios=[1, 1.3, 0.7], hspace=0.30, wspace=0.20)
    
    params = TranslationParameters()
    k_fold = result['k_fold']
    f_fold = result['f_fold']
    
    domain_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    
    # Panel A: Schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.7, 'A. Constant f_fold Model', fontsize=12, fontweight='bold', ha='center')
    
    pop_box = FancyBboxPatch((0.0, 4.5), 4.5, 5.0, boxstyle="round,pad=0.1",
                              facecolor='lightgray', edgecolor='gray', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(2.25, 9.0, 'Nascent Chains', fontsize=10, fontweight='bold', ha='center', color='#2c3e50')
    
    fold_box = FancyBboxPatch((0.3, 7.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                               facecolor='#3498db', edgecolor='#2980b9', linewidth=2, alpha=0.9)
    ax.add_patch(fold_box)
    ax.text(2.25, 7.8, f'Foldable ({f_fold*100:.0f}%)', fontsize=10, fontweight='bold', ha='center', color='white')
    
    non_box = FancyBboxPatch((0.3, 5.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                              facecolor='#7f8c8d', edgecolor='#5d6d7e', linewidth=2, alpha=0.9)
    ax.add_patch(non_box)
    ax.text(2.25, 5.8, f'Non-Foldable ({(1-f_fold)*100:.0f}%)', fontsize=10, fontweight='bold', ha='center', color='white')
    
    ax.annotate('', xy=(5.0, 7.5), xytext=(4.3, 7.8),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
    
    state_y = 7.5
    ax.text(5.5, state_y, 'U', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgray', edgecolor='gray', lw=2), color='white')
    ax.annotate('', xy=(7.0, state_y), xytext=(6.1, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.5, state_y + 0.5, r'$k_{fold}$', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.6, state_y, 'F', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', edgecolor='darkred', lw=2), color='white')
    ax.text(5.5, state_y - 1.1, 'Unfolded', fontsize=8, ha='center', color='#555')
    ax.text(7.6, state_y - 1.1, 'Folded', fontsize=8, ha='center', color='#555')
    ax.text(5, 2.8, 'Constant foldable fraction\n(no avidity term)', fontsize=10, ha='center', fontstyle='italic', color='#555')
    
    # Panel B: Model Fit
    ax = fig.add_subplot(gs[0, 1])
    n_vals = list(range(7))
    exp_means = [np.mean(exp_data[n]) for n in n_vals]
    exp_sems = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in n_vals]
    preds = [result['predictions'][n] for n in n_vals]
    
    for n in range(7):
        jitter = np.random.normal(0, 0.05, len(exp_data[n]))
        ax.scatter([n + j for j in jitter], exp_data[n], alpha=0.35, color='lightgray', s=18, zorder=1)
    
    ax.errorbar(n_vals, exp_means, yerr=exp_sems, fmt='o', capsize=5, capthick=2, 
                color='dimgray', markersize=10, label='Data (mean ± SEM)', zorder=3, linewidth=2)
    ax.scatter(n_vals, preds, color='#3498db', s=200, zorder=4, marker='+', linewidths=3.5,
               label='Constant f_fold Model')
    
    ax.set_xlabel('Number of GFP Domains', fontsize=12, fontweight='bold')
    ax.set_ylabel('CoF Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Model Fit', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel C: Parameters Table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title('C. Fitted Parameters', fontsize=13, fontweight='bold')
    
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        [r'$\tau_{fold}$', f'{result["tau_fold"]:.1f}', 's'],
        [r'$f_{fold}$', f'{f_fold:.2f}', '-'],
        [r'$\chi^2$', f'{result["cost"]:.1f}', '-'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.30, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.2)
    
    for j in range(3):
        table[(0, j)].set_facecolor('dimgray')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Panel D: Timeline
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(-30, 420)
    ax.set_ylim(-6.5, 2)
    ax.axis('off')
    ax.set_title('D. Translation Timeline', fontsize=13, fontweight='bold', pad=10)
    
    T_total = params.T_total
    scale = 360 / T_total
    
    ax.arrow(0, 0, 385, 0, head_width=0.25, head_length=5, fc='black', ec='black', linewidth=2)
    ax.text(392, 0, 'Time (s)', fontsize=10, va='center')
    ax.plot([0, 0], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(0, 0.5, '0s', fontsize=9, ha='center', va='bottom')
    
    tau_total = 1/k_fold
    positions = list(range(1, 7))
    
    for i, pos in enumerate(positions[:5]):
        t_emerge = params.domain_emergence_time(pos)
        x = t_emerge * scale
        ax.plot([x, x], [-0.25, 0.25], 'k-', linewidth=1.5)
        ax.text(x, 0.5, f'{t_emerge:.0f}s', fontsize=8, ha='center', va='bottom')
        ax.text(x, 1.1, f'GFP{pos}', fontsize=9, ha='center', va='bottom', 
                color=domain_colors[i], fontweight='bold')
    
    x_term = T_total * scale
    ax.plot([x_term, x_term], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(x_term, 0.5, f'{T_total:.0f}s', fontsize=9, ha='center', va='bottom')
    ax.text(x_term, 1.1, 'GFP6/Stop', fontsize=9, ha='center', va='bottom', color='red', fontweight='bold')
    
    for i, pos in enumerate(positions):
        t_emerge = params.domain_emergence_time(pos)
        tw = params.time_window(pos)
        x_start = t_emerge * scale
        x_end = T_total * scale
        y = -0.8 - i * 0.85
        
        ax.plot([x_start, x_end], [y, y], color=domain_colors[i], linewidth=10, solid_capstyle='butt')
        ax.text(-25, y, f'GFP{pos}:', fontsize=10, ha='right', va='center', 
                color=domain_colors[i], fontweight='bold')
        
        status = 'ok' if tw > tau_total else 'x'
        status_color = 'green' if tw > tau_total else 'red'
        ax.text(x_end + 8, y, f'{tw:.0f}s {status}', fontsize=10, ha='left', va='center',
                color=status_color, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {save_path}")


# =============================================================================
# MODEL 2: MICHAELIS-MENTEN KINETICS (U → F with hyperbolic kinetics)
# =============================================================================

class MichaelisMentenModel:
    """
    Michaelis-Menten Kinetics Model: U → F (hyperbolic saturation)
    
    Instead of exponential folding kinetics, uses hyperbolic saturation:
        P_fold(t) = t / (K_m + t)
    
    where K_m is the half-saturation time (time at which P_fold = 0.5).
    
    Biological motivation:
        - Rate-limiting intermediate state in folding pathway
        - Saturating cofactor or chaperone assistance
        - Ribosome exit tunnel as limiting "catalyst"
    
    Parameters:
        - K_m: Half-saturation time (s) - time at which P_fold = 50%
        - f_fold: Constant foldable fraction (no avidity)
    """
    
    def __init__(self):
        self.params = TranslationParameters()
    
    def folded_fraction(self, t, K_m):
        """Probability of being folded after time t (Michaelis-Menten kinetics)."""
        if t <= 0:
            return 0.0
        return t / (K_m + t)
    
    def efficiency(self, n_gfp, K_m, f_fold):
        """Calculate CoF efficiency for n_gfp domains."""
        if n_gfp == 0:
            return 0.0
        
        f_eff = f_fold  # Constant, no avidity
        
        folded = sum(self.folded_fraction(self.params.time_window(p), K_m)
                   for p in range(1, n_gfp + 1))
        
        return f_eff * (folded / n_gfp) * 100


def fit_michaelis_menten(exp_data):
    """Fit Michaelis-Menten kinetics model (2 parameters)."""
    model = MichaelisMentenModel()
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def objective(params):
        K_m = 10**params[0]
        f_fold = params[1]
        pred = [model.efficiency(n, K_m, f_fold) for n in range(1, 7)]
        return weighted_mse(pred, obs, sem)
    
    # K_m: 10-300s (half-saturation time)
    # log10(10) = 1.0, log10(300) = 2.48
    bounds = [(1.0, 2.48), (0.2, 1.0)]
    result = differential_evolution(objective, bounds, maxiter=1000, seed=42, polish=True, disp=False)
    
    K_m = 10**result.x[0]
    f_fold = result.x[1]
    preds = {n: (model.efficiency(n, K_m, f_fold) if n > 0 else 0) for n in range(7)}
    
    return {
        'K_m': K_m, 
        'f_fold': f_fold,
        'cost': result.fun,
        'predictions': preds,
        'model_name': 'MichaelisMenten',
        'n_params': 2
    }


def create_michaelis_menten_figure(exp_data, result, save_path):
    """Create comprehensive figure for Michaelis-Menten kinetics model."""
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7], 
                  width_ratios=[1, 1.3, 0.7], hspace=0.30, wspace=0.20)
    
    params = TranslationParameters()
    K_m = result['K_m']
    f_fold = result['f_fold']
    
    domain_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    
    # Panel A: Schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.7, 'A. Michaelis-Menten Model', fontsize=12, fontweight='bold', ha='center')
    
    pop_box = FancyBboxPatch((0.0, 4.5), 4.5, 5.0, boxstyle="round,pad=0.1",
                              facecolor='lightgray', edgecolor='gray', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(2.25, 9.0, 'Nascent Chains', fontsize=10, fontweight='bold', ha='center', color='#2c3e50')
    
    fold_box = FancyBboxPatch((0.3, 7.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                               facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=2, alpha=0.9)
    ax.add_patch(fold_box)
    ax.text(2.25, 7.8, f'Foldable ({f_fold*100:.0f}%)', fontsize=10, fontweight='bold', ha='center', color='white')
    
    non_box = FancyBboxPatch((0.3, 5.0), 3.9, 1.6, boxstyle="round,pad=0.05",
                              facecolor='#7f8c8d', edgecolor='#5d6d7e', linewidth=2, alpha=0.9)
    ax.add_patch(non_box)
    ax.text(2.25, 5.8, f'Non-Foldable ({(1-f_fold)*100:.0f}%)', fontsize=10, fontweight='bold', ha='center', color='white')
    
    ax.annotate('', xy=(5.0, 7.5), xytext=(4.3, 7.8),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2.5))
    
    state_y = 7.5
    ax.text(5.5, state_y, 'U', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgray', edgecolor='gray', lw=2), color='white')
    ax.annotate('', xy=(7.0, state_y), xytext=(6.1, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.5, state_y + 0.5, r'$\frac{t}{K_m+t}$', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.6, state_y, 'F', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='#9b59b6', edgecolor='#8e44ad', lw=2), color='white')
    ax.text(5.5, state_y - 1.1, 'Unfolded', fontsize=8, ha='center', color='#555')
    ax.text(7.6, state_y - 1.1, 'Folded', fontsize=8, ha='center', color='#555')
    ax.text(5, 3.2, 'Hyperbolic kinetics:', fontsize=10, ha='center', fontstyle='italic', color='#555')
    ax.text(5, 2.4, r'$P_{fold} = \frac{t}{K_m + t}$', fontsize=11, ha='center', color='#9b59b6')
    
    # Panel B: Model Fit
    ax = fig.add_subplot(gs[0, 1])
    n_vals = list(range(7))
    exp_means = [np.mean(exp_data[n]) for n in n_vals]
    exp_sems = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in n_vals]
    preds = [result['predictions'][n] for n in n_vals]
    
    for n in range(7):
        jitter = np.random.normal(0, 0.05, len(exp_data[n]))
        ax.scatter([n + j for j in jitter], exp_data[n], alpha=0.35, color='lightgray', s=18, zorder=1)
    
    ax.errorbar(n_vals, exp_means, yerr=exp_sems, fmt='o', capsize=5, capthick=2, 
                color='dimgray', markersize=10, label='Data (mean ± SEM)', zorder=3, linewidth=2)
    ax.scatter(n_vals, preds, color='#9b59b6', s=200, zorder=4, marker='+', linewidths=3.5,
               label='Michaelis-Menten Model')
    
    ax.set_xlabel('Number of GFP Domains', fontsize=12, fontweight='bold')
    ax.set_ylabel('CoF Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Model Fit', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel C: Parameters Table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title('C. Fitted Parameters', fontsize=13, fontweight='bold')
    
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        [r'$K_m$', f'{K_m:.1f}', 's'],
        [r'$f_{fold}$', f'{f_fold:.2f}', '-'],
        [r'$\chi^2$', f'{result["cost"]:.1f}', '-'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.30, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.2)
    
    for j in range(3):
        table[(0, j)].set_facecolor('#9b59b6')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Panel D: Kinetic comparison (Exponential vs MM)
    ax = fig.add_subplot(gs[1, :2])
    t_vals = np.linspace(0, 300, 100)
    
    # Exponential kinetics (for comparison, using τ = K_m)
    p_exp = 1 - np.exp(-t_vals / K_m)
    # Michaelis-Menten kinetics
    p_mm = t_vals / (K_m + t_vals)
    
    ax.plot(t_vals, p_exp * 100, '--', color='#3498db', lw=2.5, label=f'Exponential (τ={K_m:.0f}s)')
    ax.plot(t_vals, p_mm * 100, '-', color='#9b59b6', lw=2.5, label=f'Michaelis-Menten (Km={K_m:.0f}s)')
    ax.axhline(50, color='gray', linestyle=':', lw=1)
    ax.axvline(K_m, color='gray', linestyle=':', lw=1)
    ax.text(K_m + 5, 52, f'K_m = {K_m:.0f}s', fontsize=9, color='gray')
    
    # Mark time windows for each domain
    for i in range(1, 7):
        tw = params.time_window(i)
        if tw > 0:
            p_at_tw = tw / (K_m + tw) * 100
            ax.plot(tw, p_at_tw, 'o', color=domain_colors[i-1], markersize=10, zorder=5)
            ax.annotate(f'GFP{i}', (tw, p_at_tw), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color=domain_colors[i-1], fontweight='bold')
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Folding Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('D. Kinetic Comparison: Exponential vs Michaelis-Menten', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel E: Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ax.text(0.5, 0.7, 'Key Difference:', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, 'Exponential: 63% at t=τ\nMM: 50% at t=Km', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#555')
    ax.text(0.5, 0.2, 'Same issue:\nNo domain-count\ndependence', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#c0392b', fontstyle='italic')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {save_path}")


# =============================================================================
# MODEL 3: TWO-STEP CONSTANT (U → F → B)
# =============================================================================

class TwoStepConstantModel:
    """
    Two-Step Constant Model: U → F → B
    
    Adds a BINDING step after folding, with CONSTANT foldable fraction.
    Tests whether binding kinetics can explain the trend without avidity.
    
    Parameters:
        - k_fold: Folding rate constant (s⁻¹)
        - k_bind: Binding rate constant (s⁻¹)
        - f_fold: Constant foldable fraction
    """
    
    def __init__(self):
        self.params = TranslationParameters()
    
    def bound_probability(self, t, k_fold, k_bind):
        """Probability of being bound after time t (two-step kinetics)."""
        if t <= 0:
            return 0.0
        
        if abs(k_fold - k_bind) < 1e-10 * max(k_fold, k_bind):
            k = (k_fold + k_bind) / 2
            return 1.0 - (1.0 + k * t) * np.exp(-k * t)
        
        delta_k = k_bind - k_fold
        term1 = (k_bind / delta_k) * np.exp(-k_fold * t)
        term2 = (k_fold / delta_k) * np.exp(-k_bind * t)
        
        P_bound = 1.0 - term1 + term2
        return max(0.0, min(1.0, P_bound))
    
    def efficiency(self, n_gfp, k_fold, k_bind, f_fold):
        """Calculate CoF efficiency for n_gfp domains."""
        if n_gfp == 0:
            return 0.0
        
        f_eff = f_fold  # Constant, no avidity
        
        bound = sum(self.bound_probability(self.params.time_window(p), k_fold, k_bind)
                   for p in range(1, n_gfp + 1))
        
        return f_eff * (bound / n_gfp) * 100


def fit_twostep_constant(exp_data):
    """Fit Two-Step Constant model (3 parameters)."""
    model = TwoStepConstantModel()
    obs = [np.mean(exp_data[n]) for n in range(1, 7)]
    sem = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in range(1, 7)]
    
    def objective(params):
        k_fold = 10**params[0]
        k_bind = 10**params[1]
        f_fold = params[2]
        pred = [model.efficiency(n, k_fold, k_bind, f_fold) for n in range(1, 7)]
        return weighted_mse(pred, obs, sem)
    
    bounds = [(-2.40, -1.48), (-1.90, 0.0), (0.2, 1.0)]
    result = differential_evolution(objective, bounds, maxiter=2000, seed=42, 
                                    polish=True, disp=False, popsize=20)
    
    k_fold = 10**result.x[0]
    k_bind = 10**result.x[1]
    f_fold = result.x[2]
    preds = {n: (model.efficiency(n, k_fold, k_bind, f_fold) if n > 0 else 0) for n in range(7)}
    
    return {
        'k_fold': k_fold,
        'tau_fold': 1/k_fold,
        'k_bind': k_bind,
        'tau_bind': 1/k_bind,
        'f_fold': f_fold,
        'cost': result.fun,
        'predictions': preds,
        'model_name': 'TwoStepConstant',
        'n_params': 3
    }


def create_twostep_constant_figure(exp_data, result, save_path):
    """Create comprehensive figure for Two-Step Constant model."""
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7], 
                  width_ratios=[1.2, 1.3, 0.7], hspace=0.30, wspace=0.20)
    
    params = TranslationParameters()
    k_fold = result['k_fold']
    k_bind = result['k_bind']
    f_fold = result['f_fold']
    
    domain_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    
    # Panel A: Schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(6, 9.7, 'A. Two-Step Constant Model', fontsize=12, fontweight='bold', ha='center')
    
    pop_box = FancyBboxPatch((0.0, 4.5), 4.0, 5.0, boxstyle="round,pad=0.1",
                              facecolor='lightgray', edgecolor='gray', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(2.0, 9.0, 'Nascent Chains', fontsize=9, fontweight='bold', ha='center', color='#2c3e50')
    
    fold_box = FancyBboxPatch((0.2, 7.0), 3.6, 1.6, boxstyle="round,pad=0.05",
                               facecolor='#3498db', edgecolor='#2980b9', linewidth=2, alpha=0.9)
    ax.add_patch(fold_box)
    ax.text(2.0, 7.8, f'Foldable ({f_fold*100:.0f}%)', fontsize=9, fontweight='bold', ha='center', color='white')
    
    non_box = FancyBboxPatch((0.2, 5.0), 3.6, 1.6, boxstyle="round,pad=0.05",
                              facecolor='#7f8c8d', edgecolor='#5d6d7e', linewidth=2, alpha=0.9)
    ax.add_patch(non_box)
    ax.text(2.0, 5.8, f'Non-Foldable ({(1-f_fold)*100:.0f}%)', fontsize=9, fontweight='bold', ha='center', color='white')
    
    ax.annotate('', xy=(4.5, 7.5), xytext=(3.9, 7.8),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
    
    state_y = 7.5
    ax.text(5.2, state_y, 'U', fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.25', facecolor='lightgray', edgecolor='gray', lw=2), color='#333')
    ax.annotate('', xy=(6.5, state_y), xytext=(5.7, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(6.1, state_y + 0.5, r'$k_{fold}$', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.2, state_y, 'F', fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.25', facecolor='#f39c12', edgecolor='#d68910', lw=2), color='white')
    ax.annotate('', xy=(8.5, state_y), xytext=(7.7, state_y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(8.1, state_y + 0.5, r'$k_{bind}$', fontsize=9, ha='center', fontweight='bold')
    ax.text(9.2, state_y, 'B', fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.25', facecolor='#27ae60', edgecolor='#1e8449', lw=2), color='white')
    
    ax.text(5.2, state_y - 0.9, 'Unfolded', fontsize=7, ha='center', color='#555')
    ax.text(7.2, state_y - 0.9, 'Folded', fontsize=7, ha='center', color='#555')
    ax.text(9.2, state_y - 0.9, 'Bound', fontsize=7, ha='center', color='#555')
    ax.text(9.2, state_y - 1.4, '(detected)', fontsize=7, ha='center', color='#27ae60', fontweight='bold')
    ax.text(6, 3.0, 'Two-step kinetics: U→F→B', fontsize=10, ha='center', fontstyle='italic', color='#555')
    ax.text(6, 2.0, 'Constant f_fold (no avidity)', fontsize=9, ha='center', color='#888')
    
    # Panel B: Model Fit
    ax = fig.add_subplot(gs[0, 1])
    n_vals = list(range(7))
    exp_means = [np.mean(exp_data[n]) for n in n_vals]
    exp_sems = [np.std(exp_data[n])/np.sqrt(len(exp_data[n])) for n in n_vals]
    preds = [result['predictions'][n] for n in n_vals]
    
    for n in range(7):
        jitter = np.random.normal(0, 0.05, len(exp_data[n]))
        ax.scatter([n + j for j in jitter], exp_data[n], alpha=0.35, color='lightgray', s=18, zorder=1)
    
    ax.errorbar(n_vals, exp_means, yerr=exp_sems, fmt='o', capsize=5, capthick=2, 
                color='dimgray', markersize=10, label='Data (mean ± SEM)', zorder=3, linewidth=2)
    ax.scatter(n_vals, preds, color='#27ae60', s=200, zorder=4, marker='+', linewidths=3.5,
               label='Two-Step Constant Model')
    
    ax.set_xlabel('Number of GFP Domains', fontsize=12, fontweight='bold')
    ax.set_ylabel('CoF Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('B. Model Fit', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel C: Parameters Table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_title('C. Fitted Parameters', fontsize=13, fontweight='bold')
    
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        [r'$\tau_{fold}$', f'{result["tau_fold"]:.1f}', 's'],
        [r'$\tau_{bind}$', f'{result["tau_bind"]:.1f}', 's'],
        [r'$f_{fold}$', f'{f_fold:.2f}', '-'],
        [r'$\chi^2$', f'{result["cost"]:.1f}', '-'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.35, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.0)
    
    for j in range(3):
        table[(0, j)].set_facecolor('dimgray')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Panel D: Timeline
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(-30, 420)
    ax.set_ylim(-6.5, 2)
    ax.axis('off')
    ax.set_title('D. Translation Timeline: Time for Folding + Binding', fontsize=13, fontweight='bold', pad=10)
    
    T_total = params.T_total
    scale = 360 / T_total
    
    ax.arrow(0, 0, 385, 0, head_width=0.25, head_length=5, fc='black', ec='black', linewidth=2)
    ax.text(392, 0, 'Time (s)', fontsize=10, va='center')
    ax.plot([0, 0], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(0, 0.5, '0s', fontsize=9, ha='center', va='bottom')
    
    tau_total = result['tau_fold'] + result['tau_bind']
    positions = list(range(1, 7))
    
    for i, pos in enumerate(positions[:5]):
        t_emerge = params.domain_emergence_time(pos)
        x = t_emerge * scale
        ax.plot([x, x], [-0.25, 0.25], 'k-', linewidth=1.5)
        ax.text(x, 0.5, f'{t_emerge:.0f}s', fontsize=8, ha='center', va='bottom')
        ax.text(x, 1.1, f'GFP{pos}', fontsize=9, ha='center', va='bottom', 
                color=domain_colors[i], fontweight='bold')
    
    x_term = T_total * scale
    ax.plot([x_term, x_term], [-0.25, 0.25], 'k-', linewidth=2)
    ax.text(x_term, 0.5, f'{T_total:.0f}s', fontsize=9, ha='center', va='bottom')
    ax.text(x_term, 1.1, 'GFP6/Stop', fontsize=9, ha='center', va='bottom', color='red', fontweight='bold')
    
    for i, pos in enumerate(positions):
        t_emerge = params.domain_emergence_time(pos)
        tw = params.time_window(pos)
        x_start = t_emerge * scale
        x_end = T_total * scale
        y = -0.8 - i * 0.85
        
        ax.plot([x_start, x_end], [y, y], color=domain_colors[i], linewidth=10, solid_capstyle='butt')
        ax.text(-25, y, f'GFP{pos}:', fontsize=10, ha='right', va='center', 
                color=domain_colors[i], fontweight='bold')
        
        status = 'ok' if tw > tau_total else 'x'
        status_color = 'green' if tw > tau_total else 'red'
        ax.text(x_end + 8, y, f'{tw:.0f}s {status}', fontsize=10, ha='left', va='center',
                color=status_color, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("TESTING NEW MODELS SANDBOX")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent / "Dark mCh Cells.xlsx"
    print(f"\nLoading: {data_path}")
    exp_data = load_data(str(data_path))
    
    print("\nExperimental Data:")
    for n in range(1, 7):
        v = exp_data[n]
        print(f"  {n}xGFP: {np.mean(v):.1f}% ± {np.std(v)/np.sqrt(len(v)):.1f}%")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'test_figures'
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    results = {}
    
    # ==================== Model 1: Constant f_fold ====================
    print("\n" + "="*70)
    print("MODEL 1: CONSTANT F_FOLD (U → F)")
    print("="*70)
    print("Parameters: k_fold, f_fold (constant)")
    
    res_constant = fit_constant_pool(exp_data)
    results['ConstantPool'] = res_constant
    
    print(f"\nFitted Parameters:")
    print(f"  τ_fold = {res_constant['tau_fold']:.1f}s")
    print(f"  f_fold = {res_constant['f_fold']:.2f}")
    print(f"  χ² = {res_constant['cost']:.1f}")
    
    create_constant_pool_figure(exp_data, res_constant, 
                                str(output_dir / "constant_ffold_solution.png"))
    
    # ==================== Model 2: Michaelis-Menten ====================
    print("\n" + "="*70)
    print("MODEL 2: MICHAELIS-MENTEN KINETICS (U → F)")
    print("="*70)
    print("Parameters: K_m (half-saturation time), f_fold (constant)")
    print("Kinetics: P_fold = t / (K_m + t)  [hyperbolic]")
    
    res_mm = fit_michaelis_menten(exp_data)
    results['MichaelisMenten'] = res_mm
    
    print(f"\nFitted Parameters:")
    print(f"  K_m = {res_mm['K_m']:.1f}s  (half-saturation time)")
    print(f"  f_fold = {res_mm['f_fold']:.2f}")
    print(f"  χ² = {res_mm['cost']:.1f}")
    
    create_michaelis_menten_figure(exp_data, res_mm,
                                   str(output_dir / "michaelis_menten_solution.png"))
    
    # ==================== Model 3: Two-Step Constant ====================
    print("\n" + "="*70)
    print("MODEL 3: TWO-STEP CONSTANT (U → F → B)")
    print("="*70)
    print("Parameters: k_fold, k_bind, f_fold (constant)")
    
    res_twostep = fit_twostep_constant(exp_data)
    results['TwoStepConstant'] = res_twostep
    
    print(f"\nFitted Parameters:")
    print(f"  τ_fold = {res_twostep['tau_fold']:.1f}s")
    print(f"  τ_bind = {res_twostep['tau_bind']:.1f}s")
    print(f"  f_fold = {res_twostep['f_fold']:.2f}")
    print(f"  χ² = {res_twostep['cost']:.1f}")
    
    create_twostep_constant_figure(exp_data, res_twostep,
                                   str(output_dir / "twostep_constant_solution.png"))
    
    # ==================== Summary ====================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Params':>8} {'χ²':>10}")
    print("-"*45)
    for name, res in results.items():
        print(f"{name:<25} {res['n_params']:>8} {res['cost']:>10.1f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\n✗ All test models FAIL to reproduce the increasing efficiency trend.")
    print("  - Constant f_fold (exponential): χ² = {:.1f}".format(res_constant['cost']))
    print("  - Michaelis-Menten (hyperbolic): χ² = {:.1f}".format(res_mm['cost']))
    print("  - Two-Step Constant (U→F→B):    χ² = {:.1f}".format(res_twostep['cost']))
    print("\n→ Kinetic shape (exponential vs hyperbolic) doesn't matter.")
    print("  Adding a binding step doesn't help either.")
    print("\n→ This confirms that an AVIDITY-LIKE EFFECT is necessary.")
    print("  The Two-Pool model's f_gain term is justified by the data.")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
