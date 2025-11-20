"""
Visualize propagated uncertainty analysis.

Creates comprehensive plots showing:
- Distribution of RF, logg, and total uncertainties
- Uncertainty vs temperature relationships
- Gradient distribution
- Relative contributions
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def main():
    print("=" * 80)
    print("VISUALIZING PROPAGATED UNCERTAINTIES")
    print("=" * 80)

    # Load final predictions with uncertainties
    print("\n[1/4] Loading data...")
    data_path = Path('data/processed/teff_predictions_with_logg_propagated_final.parquet')
    df = pl.read_parquet(data_path)

    print(f"  Loaded {len(df):,} objects")

    # Convert to numpy for plotting
    teff = df['teff_predicted'].to_numpy()
    unc_rf = df['teff_unc_rf'].to_numpy()
    unc_logg = df['teff_unc_logg'].to_numpy()
    unc_total = df['teff_unc_total'].to_numpy()
    gradient = df['gradient_teff_logg'].to_numpy()
    logg_unc = df['logg_uncertainty'].to_numpy()

    # Create output directory
    output_dir = Path('reports/figures/teff_uncertainty_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/4] Creating uncertainty distribution plots...")

    # ========================================================================
    # PLOT 1: Uncertainty Distributions
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Teff Prediction Uncertainty Analysis', fontsize=16, fontweight='bold')

    # Top left: Histogram comparison
    ax = axes[0, 0]
    bins = np.linspace(0, 3000, 60)
    ax.hist(unc_rf, bins=bins, alpha=0.5, label='Teff Only', color='C0', density=True)
    ax.hist(unc_logg, bins=bins, alpha=0.5, label='Teff + logg', color='C1', density=True)
    ax.hist(unc_total, bins=bins, alpha=0.5, label='Teff + Clustering', color='C2', density=True)
    ax.axvline(unc_rf.mean(), color='C0', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(unc_logg.mean(), color='C1', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(unc_total.mean(), color='C2', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Teff Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of Teff Uncertainties', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Add statistics text box
    stats_text = f"Teff Only: μ={unc_rf.mean():.0f}K, σ={unc_rf.std():.0f}K\n"
    stats_text += f"Teff + logg: μ={unc_logg.mean():.0f}K, σ={unc_logg.std():.0f}K\n"
    stats_text += f"Teff + Clustering: μ={unc_total.mean():.0f}K, σ={unc_total.std():.0f}K"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Top right: Kernel density
    ax = axes[0, 1]
    from scipy.stats import gaussian_kde
    x_range = np.linspace(0, 3000, 300)
    kde_rf = gaussian_kde(unc_rf[unc_rf < 3000])
    kde_logg = gaussian_kde(unc_logg[unc_logg < 3000])
    kde_total = gaussian_kde(unc_total[unc_total < 3000])
    ax.plot(x_range, kde_rf(x_range), label='Teff Only', linewidth=2, color='C0')
    ax.plot(x_range, kde_logg(x_range), label='Teff + logg', linewidth=2, color='C1')
    ax.plot(x_range, kde_total(x_range), label='Teff + Clustering', linewidth=2, color='C2')
    ax.set_xlabel('Teff Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Density [KDE]', fontsize=11)
    ax.set_title('Kernel Density Estimate of Teff Uncertainties', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom left: Cumulative distribution
    ax = axes[1, 0]
    sorted_rf = np.sort(unc_rf)
    sorted_logg = np.sort(unc_logg)
    sorted_total = np.sort(unc_total)
    cumulative = np.linspace(0, 1, len(sorted_rf))
    ax.plot(sorted_rf, cumulative, label='Teff Only', linewidth=2, color='C0')
    ax.plot(sorted_logg, cumulative, label='Teff + logg', linewidth=2, color='C1')
    ax.plot(sorted_total, cumulative, label='Teff + Clustering', linewidth=2, color='C2')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    ax.text(2700, 0.5, '50%', fontsize=9, color='gray')
    ax.text(2700, 0.9, '90%', fontsize=9, color='gray')
    ax.text(2700, 0.95, '95%', fontsize=9, color='gray')
    ax.set_xlabel('Teff Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title('Cumulative Distribution of Teff Uncertainties', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 3000)
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom right: Distribution of logg uncertainties
    ax = axes[1, 1]
    ax.hist(logg_unc, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(logg_unc.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {logg_unc.mean():.3f} dex')
    ax.axvline(np.median(logg_unc), color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {np.median(logg_unc):.3f} dex')
    ax.set_xlabel('logg Uncertainty (dex)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of logg Uncertainties (Teff + logg model)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add statistics
    stats_text = f"Mean: {logg_unc.mean():.3f} dex\n"
    stats_text += f"Median: {np.median(logg_unc):.3f} dex\n"
    stats_text += f"Std: {logg_unc.std():.3f} dex\n"
    stats_text += f"Range: [{logg_unc.min():.3f}, {logg_unc.max():.3f}] dex"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = output_dir / 'teff_uncertainty_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 2: Uncertainty vs Temperature
    # ========================================================================
    print(f"\n[3/4] Creating uncertainty vs temperature plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty vs Temperature Analysis', fontsize=16, fontweight='bold')

    # Filter for visualization (remove extreme outliers)
    mask = (unc_total < 2000) & (teff > 3000) & (teff < 15000)

    # Top left: RF uncertainty vs Teff
    ax = axes[0, 0]
    h = ax.hexbin(teff[mask], unc_rf[mask], gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('RF Uncertainty (K)', fontsize=11)
    ax.set_title('RF Tree Uncertainty vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Top right: logg contribution vs Teff
    ax = axes[0, 1]
    h = ax.hexbin(teff[mask], unc_logg[mask], gridsize=50, cmap='Oranges', mincnt=1)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('logg-Propagated Uncertainty (K)', fontsize=11)
    ax.set_title('logg Contribution vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom left: Total uncertainty vs Teff
    ax = axes[1, 0]
    h = ax.hexbin(teff[mask], unc_total[mask], gridsize=50, cmap='Greens', mincnt=1)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('Total Uncertainty (K)', fontsize=11)
    ax.set_title('Total Combined Uncertainty vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom right: Relative uncertainty vs Teff
    ax = axes[1, 1]
    rel_unc = (unc_total / teff) * 100
    mask2 = mask & (rel_unc < 30)  # Filter extreme relative uncertainties
    h = ax.hexbin(teff[mask2], rel_unc[mask2], gridsize=50, cmap='Reds', mincnt=1)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('Relative Uncertainty (%)', fontsize=11)
    ax.set_title('Relative Uncertainty vs Temperature', fontsize=12, fontweight='bold')
    ax.axhline(10, color='white', linestyle='--', linewidth=2, alpha=0.8, label='10% threshold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'uncertainty_vs_temperature.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 3: Gradient and Contribution Analysis
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient and Uncertainty Contribution Analysis', fontsize=16, fontweight='bold')

    # Top left: Gradient distribution
    ax = axes[0, 0]
    mask_grad = np.abs(gradient) < 3000  # Filter extreme gradients
    ax.hist(gradient[mask_grad], bins=60, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(gradient.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {gradient.mean():.0f} K/dex')
    ax.axvline(np.median(gradient), color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {np.median(gradient):.0f} K/dex')
    ax.set_xlabel('∂Teff/∂logg (K/dex)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Numerical Gradient ∂Teff/∂logg', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Top right: Gradient vs Teff
    ax = axes[0, 1]
    mask_hex = mask_grad & (teff > 3000) & (teff < 15000)
    h = ax.hexbin(teff[mask_hex], gradient[mask_hex], gridsize=50, cmap='viridis', mincnt=1)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('∂Teff/∂logg (K/dex)', fontsize=11)
    ax.set_title('Gradient vs Temperature', fontsize=12, fontweight='bold')
    ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.8)
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom left: Uncertainty ratio
    ax = axes[1, 0]
    ratio = unc_logg / unc_rf
    mask_ratio = (ratio < 5) & (ratio > 0)
    ax.hist(ratio[mask_ratio], bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(ratio.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {ratio.mean():.3f}')
    ax.axvline(np.median(ratio), color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {np.median(ratio):.3f}')
    ax.set_xlabel('σ_logg / σ_RF', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Ratio of logg to RF Uncertainties', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom right: Relative increase
    ax = axes[1, 1]
    increase_pct = ((unc_total - unc_rf) / unc_rf) * 100
    mask_inc = (increase_pct < 200) & (increase_pct > -50)
    ax.hist(increase_pct[mask_inc], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(increase_pct.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {increase_pct.mean():.1f}%')
    ax.axvline(np.median(increase_pct), color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {np.median(increase_pct):.1f}%')
    ax.set_xlabel('Relative Increase (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Relative Increase in Total vs RF Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'gradient_contribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 4: Correlation Analysis
    # ========================================================================
    print(f"\n[4/4] Creating correlation analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty Correlation Analysis', fontsize=16, fontweight='bold')

    # Top left: logg_unc vs logg-propagated uncertainty
    ax = axes[0, 0]
    mask_corr = (unc_logg < 1500) & (logg_unc < 1.5)
    h = ax.hexbin(logg_unc[mask_corr], unc_logg[mask_corr], gridsize=50, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('logg Uncertainty (dex)', fontsize=11)
    ax.set_ylabel('Teff Uncertainty from logg (K)', fontsize=11)
    ax.set_title('logg Uncertainty vs Propagated Teff Uncertainty', fontsize=12, fontweight='bold')

    # Calculate and display correlation
    corr, pval = stats.pearsonr(logg_unc[mask_corr], unc_logg[mask_corr])
    ax.text(0.05, 0.95, f'Pearson r: {corr:.3f}\np < {pval:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Top right: Absolute gradient vs logg-propagated uncertainty
    ax = axes[0, 1]
    abs_gradient = np.abs(gradient)
    mask_grad2 = (abs_gradient < 2000) & (unc_logg < 1500)
    h = ax.hexbin(abs_gradient[mask_grad2], unc_logg[mask_grad2], gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('|∂Teff/∂logg| (K/dex)', fontsize=11)
    ax.set_ylabel('Teff Uncertainty from logg (K)', fontsize=11)
    ax.set_title('Gradient Magnitude vs Propagated Uncertainty', fontsize=12, fontweight='bold')

    # Calculate correlation
    corr, pval = stats.pearsonr(abs_gradient[mask_grad2], unc_logg[mask_grad2])
    ax.text(0.05, 0.95, f'Pearson r: {corr:.3f}\np < {pval:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom left: RF uncertainty vs Total uncertainty
    ax = axes[1, 0]
    mask_total = (unc_rf < 1500) & (unc_total < 1500)
    h = ax.hexbin(unc_rf[mask_total], unc_total[mask_total], gridsize=50, cmap='Greens', mincnt=1)
    ax.plot([0, 1500], [0, 1500], 'r--', linewidth=2, alpha=0.7, label='1:1 line')
    ax.set_xlabel('RF Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Total Uncertainty (K)', fontsize=11)
    ax.set_title('RF vs Total Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom right: Scatter matrix summary
    ax = axes[1, 1]
    ax.axis('off')

    # Summary statistics table
    summary_data = [
        ['Metric', 'RF Only', 'logg Contrib', 'Total'],
        ['Mean (K)', f'{unc_rf.mean():.1f}', f'{unc_logg.mean():.1f}', f'{unc_total.mean():.1f}'],
        ['Median (K)', f'{np.median(unc_rf):.1f}', f'{np.median(unc_logg):.1f}', f'{np.median(unc_total):.1f}'],
        ['Std (K)', f'{unc_rf.std():.1f}', f'{unc_logg.std():.1f}', f'{unc_total.std():.1f}'],
        ['P90 (K)', f'{np.percentile(unc_rf, 90):.1f}', f'{np.percentile(unc_logg, 90):.1f}', f'{np.percentile(unc_total, 90):.1f}'],
        ['P95 (K)', f'{np.percentile(unc_rf, 95):.1f}', f'{np.percentile(unc_logg, 95):.1f}', f'{np.percentile(unc_total, 95):.1f}'],
    ]

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Uncertainty Statistics Summary', fontsize=12, fontweight='bold', pad=20)

    # Add additional statistics
    info_text = f"Total objects: {len(df):,}\n"
    info_text += f"Objects with σ_total < 500 K: {(unc_total < 500).sum():,} ({(unc_total < 500).sum() / len(df) * 100:.1f}%)\n"
    info_text += f"Mean gradient: {gradient.mean():.1f} K/dex\n"
    info_text += f"Mean σ_logg/σ_RF ratio: {ratio.mean():.3f}"

    ax.text(0.5, 0.1, info_text, transform=ax.transAxes,
            fontsize=10, horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_file = output_dir / 'uncertainty_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nPlots created:")
    print(f"  1. teff_uncertainty_distributions.png - Distribution comparison")
    print(f"  2. uncertainty_vs_temperature.png - Uncertainty as function of Teff")
    print(f"  3. gradient_contribution_analysis.png - Gradient and contributions")
    print(f"  4. uncertainty_correlations.png - Correlation analysis")
    print("\n✓ All visualization plots successfully created!")

if __name__ == '__main__':
    main()
