#!/usr/bin/env python
"""
Compare predictions from regular and log flag 1 models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
print("Loading predictions...")
pred_regular = pd.read_parquet('data/processed/predictions_rf_gaia_colors_flag1_20251217_104707.parquet')
pred_log = pd.read_parquet('data/processed/predictions_rf_gaia_colors_flag1_log_20251217_110953.parquet')

# Basic statistics
print("\n" + "="*70)
print("PREDICTION COMPARISON: Regular vs Log Models")
print("="*70)

print(f"\nDataset size:")
print(f"  Regular model: {len(pred_regular):,} predictions")
print(f"  Log model:     {len(pred_log):,} predictions")

# Merge on source_id
merged = pred_regular.merge(pred_log, on='source_id', suffixes=('_reg', '_log'))
print(f"  Matched:       {len(merged):,} objects")

# Compare predictions
print(f"\n{'='*70}")
print("PREDICTION COMPARISON")
print(f"{'='*70}")

pred_diff = merged['teff_predicted_reg'] - merged['teff_predicted_log']
pred_diff_pct = 100 * pred_diff / merged['teff_predicted_reg']

print(f"\nTemperature Predictions:")
print(f"  Mean Teff (Regular): {merged['teff_predicted_reg'].mean():.1f} K")
print(f"  Mean Teff (Log):     {merged['teff_predicted_log'].mean():.1f} K")
print(f"\nDifferences (Regular - Log):")
print(f"  Mean difference:   {pred_diff.mean():.1f} K")
print(f"  Median difference: {pred_diff.median():.1f} K")
print(f"  Std difference:    {pred_diff.std():.1f} K")
print(f"  Max |difference|:  {abs(pred_diff).max():.1f} K")
print(f"\nRelative Differences:")
print(f"  Mean:   {pred_diff_pct.mean():.3f}%")
print(f"  Median: {pred_diff_pct.median():.3f}%")
print(f"  Std:    {pred_diff_pct.std():.3f}%")

# Compare uncertainties
print(f"\n{'='*70}")
print("UNCERTAINTY COMPARISON")
print(f"{'='*70}")

unc_diff = merged['teff_uncertainty_reg'] - merged['teff_uncertainty_log']
unc_diff_pct = 100 * unc_diff / merged['teff_uncertainty_reg']

print(f"\nRF Uncertainties:")
print(f"  Mean (Regular): {merged['teff_uncertainty_reg'].mean():.1f} K")
print(f"  Mean (Log):     {merged['teff_uncertainty_log'].mean():.1f} K")
print(f"  Median (Regular): {merged['teff_uncertainty_reg'].median():.1f} K")
print(f"  Median (Log):     {merged['teff_uncertainty_log'].median():.1f} K")

print(f"\nUncertainty Differences (Regular - Log):")
print(f"  Mean difference:   {unc_diff.mean():.1f} K")
print(f"  Median difference: {unc_diff.median():.1f} K")
print(f"  Std difference:    {unc_diff.std():.1f} K")

print(f"\nRelative Uncertainty Differences:")
print(f"  Mean:   {unc_diff_pct.mean():.3f}%")
print(f"  Median: {unc_diff_pct.median():.3f}%")

# Which model has lower uncertainty more often?
log_lower = (merged['teff_uncertainty_log'] < merged['teff_uncertainty_reg']).sum()
reg_lower = (merged['teff_uncertainty_reg'] < merged['teff_uncertainty_log']).sum()
equal = (merged['teff_uncertainty_reg'] == merged['teff_uncertainty_log']).sum()

print(f"\nLower Uncertainty:")
print(f"  Log model:     {log_lower:,} objects ({100*log_lower/len(merged):.1f}%)")
print(f"  Regular model: {reg_lower:,} objects ({100*reg_lower/len(merged):.1f}%)")
print(f"  Equal:         {equal:,} objects ({100*equal/len(merged):.1f}%)")

# Correlation
corr_pred = np.corrcoef(merged['teff_predicted_reg'], merged['teff_predicted_log'])[0, 1]
corr_unc = np.corrcoef(merged['teff_uncertainty_reg'], merged['teff_uncertainty_log'])[0, 1]

print(f"\n{'='*70}")
print("CORRELATIONS")
print(f"{'='*70}")
print(f"  Predictions:   r = {corr_pred:.6f}")
print(f"  Uncertainties: r = {corr_unc:.6f}")

# Summary statistics
print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")

if abs(pred_diff.mean()) < 1 and unc_diff.mean() < 0:
    print("\n✅ Log model has:")
    print("   - Nearly identical predictions (< 1K difference)")
    print("   - Lower average uncertainty")
    print("\n   RECOMMENDATION: Use LOG model for ensemble")
elif abs(pred_diff.mean()) < 1 and unc_diff.mean() > 0:
    print("\n✅ Regular model has:")
    print("   - Nearly identical predictions (< 1K difference)")
    print("   - Lower average uncertainty")
    print("\n   RECOMMENDATION: Use REGULAR model for ensemble")
else:
    print("\n⚠️  Models show significant differences:")
    print(f"   - Mean prediction difference: {pred_diff.mean():.1f} K")
    print(f"   - Mean uncertainty difference: {unc_diff.mean():.1f} K")
    print("\n   RECOMMENDATION: Further investigation needed")

print(f"\n{'='*70}\n")

# Create comparison plot
print("Creating comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Prediction scatter
ax = axes[0, 0]
ax.hexbin(merged['teff_predicted_reg'], merged['teff_predicted_log'],
          gridsize=50, cmap='viridis', mincnt=1)
ax.plot([2000, 25000], [2000, 25000], 'r--', lw=2, label='1:1 line')
ax.set_xlabel('Teff Regular Model (K)', fontsize=12)
ax.set_ylabel('Teff Log Model (K)', fontsize=12)
ax.set_title(f'Prediction Comparison (r={corr_pred:.4f})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Uncertainty scatter
ax = axes[0, 1]
ax.hexbin(merged['teff_uncertainty_reg'], merged['teff_uncertainty_log'],
          gridsize=50, cmap='plasma', mincnt=1)
ax.plot([0, 2000], [0, 2000], 'r--', lw=2, label='1:1 line')
ax.set_xlabel('Uncertainty Regular Model (K)', fontsize=12)
ax.set_ylabel('Uncertainty Log Model (K)', fontsize=12)
ax.set_title(f'Uncertainty Comparison (r={corr_unc:.4f})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Prediction difference histogram
ax = axes[1, 0]
ax.hist(pred_diff, bins=100, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero difference')
ax.axvline(pred_diff.mean(), color='green', linestyle='--', lw=2,
           label=f'Mean = {pred_diff.mean():.1f} K')
ax.set_xlabel('Teff Difference (Regular - Log) [K]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Prediction Differences', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Uncertainty difference histogram
ax = axes[1, 1]
ax.hist(unc_diff, bins=100, color='orange', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero difference')
ax.axvline(unc_diff.mean(), color='green', linestyle='--', lw=2,
           label=f'Mean = {unc_diff.mean():.1f} K')
ax.set_xlabel('Uncertainty Difference (Regular - Log) [K]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Uncertainty Differences', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/flag1_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved comparison plot: reports/figures/flag1_model_comparison.png")

print("\n✅ Comparison complete!")
