"""
Compare three Teff prediction models:
1. Teff Only - Basic Gaia colors model
2. Teff + logg - Model with predicted logg + uncertainty propagation
3. Teff + Clustering - Model with clustering features

Generates comprehensive comparison plots and statistics.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_predictions():
    """Load predictions from all three models."""
    print("=" * 80)
    print("LOADING PREDICTIONS FROM THREE MODELS")
    print("=" * 80)

    # Model 1: Teff Only
    print("\n[1/3] Loading Teff Only predictions...")
    df1 = pl.read_parquet('data/processed/teff_predictions_for_predicted_teff.parquet')
    print(f"  Loaded {len(df1):,} predictions")
    print(f"  Columns: {df1.columns}")

    # Rename columns for clarity
    df1 = df1.select([
        'source_id',
        pl.col('teff_predicted').alias('teff_only'),
        pl.col('teff_uncertainty').alias('unc_only')
    ])

    # Model 2: Teff + logg (with propagated uncertainty)
    print("\n[2/3] Loading Teff + logg predictions (with propagated uncertainty)...")
    df2 = pl.read_parquet('data/processed/teff_predictions_with_logg_propagated_final.parquet')
    print(f"  Loaded {len(df2):,} predictions")

    # Select relevant columns
    df2 = df2.select([
        'source_id', 'ra', 'dec',
        'g', 'bp', 'rp', 'bp_rp',
        'teff_source', 'logg_source',
        pl.col('teff_predicted').alias('teff_logg'),
        pl.col('teff_unc_total').alias('unc_logg'),  # Use total propagated uncertainty
        pl.col('gradient_teff_logg')
    ])

    # Model 3: Teff + Clustering
    print("\n[3/3] Loading Teff + Clustering predictions...")
    df3 = pl.read_parquet('data/processed/teff_predictions_with_clustering.parquet')
    print(f"  Loaded {len(df3):,} predictions")
    print(f"  Columns: {df3.columns}")

    # Rename columns
    df3 = df3.select([
        'source_id',
        pl.col('teff_predicted').alias('teff_cluster'),
        pl.col('teff_uncertainty').alias('unc_cluster')
    ])

    # Merge all three
    print("\n[4/4] Merging predictions...")
    df_merged = (df2
                 .join(df1, on='source_id', how='left')
                 .join(df3, on='source_id', how='left'))

    print(f"  Merged dataset: {len(df_merged):,} objects")
    print(f"  Columns: {df_merged.columns}")

    # Check for missing matches
    n_only = df_merged['teff_only'].is_null().sum()
    n_cluster = df_merged['teff_cluster'].is_null().sum()
    print(f"\n  Missing Teff Only: {n_only:,}")
    print(f"  Missing Clustering: {n_cluster:,}")

    return df_merged

def compute_statistics(df):
    """Compute comparison statistics."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON STATISTICS")
    print("=" * 80)

    # Filter to objects with all three predictions
    df_complete = df.filter(
        pl.col('teff_only').is_not_null() &
        pl.col('teff_logg').is_not_null() &
        pl.col('teff_cluster').is_not_null()
    )

    print(f"\nObjects with all three predictions: {len(df_complete):,}")

    # Convert to numpy
    teff_only = df_complete['teff_only'].to_numpy()
    teff_logg = df_complete['teff_logg'].to_numpy()
    teff_cluster = df_complete['teff_cluster'].to_numpy()

    unc_only = df_complete['unc_only'].to_numpy()
    unc_logg = df_complete['unc_logg'].to_numpy()
    unc_cluster = df_complete['unc_cluster'].to_numpy()

    # Temperature statistics
    print("\n" + "-" * 80)
    print("TEMPERATURE PREDICTIONS")
    print("-" * 80)

    stats_data = []
    for name, teff in [('Teff Only', teff_only),
                       ('Teff + logg', teff_logg),
                       ('Teff + Clustering', teff_cluster)]:
        print(f"\n{name}:")
        print(f"  Mean:   {teff.mean():.1f} K")
        print(f"  Median: {np.median(teff):.1f} K")
        print(f"  Std:    {teff.std():.1f} K")
        print(f"  Range:  [{teff.min():.1f}, {teff.max():.1f}] K")

        stats_data.append({
            'Model': name,
            'Mean': teff.mean(),
            'Median': np.median(teff),
            'Std': teff.std()
        })

    # Uncertainty statistics
    print("\n" + "-" * 80)
    print("UNCERTAINTY ESTIMATES")
    print("-" * 80)

    for name, unc in [('Teff Only', unc_only),
                      ('Teff + logg', unc_logg),
                      ('Teff + Clustering', unc_cluster)]:
        print(f"\n{name}:")
        print(f"  Mean:   {unc.mean():.1f} K")
        print(f"  Median: {np.median(unc):.1f} K")
        print(f"  Std:    {unc.std():.1f} K")
        print(f"  Range:  [{unc.min():.1f}, {unc.max():.1f}] K")
        print(f"  P90:    {np.percentile(unc, 90):.1f} K")
        print(f"  P95:    {np.percentile(unc, 95):.1f} K")

    # Pairwise differences
    print("\n" + "-" * 80)
    print("PAIRWISE TEMPERATURE DIFFERENCES")
    print("-" * 80)

    diff_logg_only = teff_logg - teff_only
    diff_cluster_only = teff_cluster - teff_only
    diff_logg_cluster = teff_logg - teff_cluster

    print("\nTeff+logg - Teff Only:")
    print(f"  Mean diff:   {diff_logg_only.mean():.1f} K")
    print(f"  Median diff: {np.median(diff_logg_only):.1f} K")
    print(f"  Std diff:    {diff_logg_only.std():.1f} K")
    print(f"  RMS diff:    {np.sqrt(np.mean(diff_logg_only**2)):.1f} K")

    print("\nTeff+Clustering - Teff Only:")
    print(f"  Mean diff:   {diff_cluster_only.mean():.1f} K")
    print(f"  Median diff: {np.median(diff_cluster_only):.1f} K")
    print(f"  Std diff:    {diff_cluster_only.std():.1f} K")
    print(f"  RMS diff:    {np.sqrt(np.mean(diff_cluster_only**2)):.1f} K")

    print("\nTeff+logg - Teff+Clustering:")
    print(f"  Mean diff:   {diff_logg_cluster.mean():.1f} K")
    print(f"  Median diff: {np.median(diff_logg_cluster):.1f} K")
    print(f"  Std diff:    {diff_logg_cluster.std():.1f} K")
    print(f"  RMS diff:    {np.sqrt(np.mean(diff_logg_cluster**2)):.1f} K")

    # Correlations
    print("\n" + "-" * 80)
    print("TEMPERATURE CORRELATIONS")
    print("-" * 80)

    corr_logg_only, _ = stats.pearsonr(teff_logg, teff_only)
    corr_cluster_only, _ = stats.pearsonr(teff_cluster, teff_only)
    corr_logg_cluster, _ = stats.pearsonr(teff_logg, teff_cluster)

    print(f"\nTeff+logg vs Teff Only:       r = {corr_logg_only:.4f}")
    print(f"Teff+Clustering vs Teff Only: r = {corr_cluster_only:.4f}")
    print(f"Teff+logg vs Teff+Clustering: r = {corr_logg_cluster:.4f}")

    return df_complete

def create_comparison_plots(df):
    """Create comprehensive comparison plots."""
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80)

    # Filter complete cases
    df_complete = df.filter(
        pl.col('teff_only').is_not_null() &
        pl.col('teff_logg').is_not_null() &
        pl.col('teff_cluster').is_not_null()
    )

    # Convert to numpy
    teff_only = df_complete['teff_only'].to_numpy()
    teff_logg = df_complete['teff_logg'].to_numpy()
    teff_cluster = df_complete['teff_cluster'].to_numpy()

    unc_only = df_complete['unc_only'].to_numpy()
    unc_logg = df_complete['unc_logg'].to_numpy()
    unc_cluster = df_complete['unc_cluster'].to_numpy()

    output_dir = Path('reports/figures/three_model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PLOT 1: Temperature Comparison Scatter
    # ========================================================================
    print("\n[1/4] Creating temperature comparison scatter plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Temperature Prediction Comparison: Three Models', fontsize=16, fontweight='bold')

    # Filter for plotting (remove extreme outliers)
    mask = (teff_only > 3000) & (teff_only < 15000) & \
           (teff_logg > 3000) & (teff_logg < 15000) & \
           (teff_cluster > 3000) & (teff_cluster < 15000)

    # Top left: Teff+logg vs Teff Only
    ax = axes[0, 0]
    h = ax.hexbin(teff_only[mask], teff_logg[mask], gridsize=60, cmap='Blues', mincnt=1)
    ax.plot([3000, 15000], [3000, 15000], 'r--', linewidth=2, alpha=0.7, label='1:1 line')
    ax.set_xlabel('Teff Only (K)', fontsize=11)
    ax.set_ylabel('Teff + logg (K)', fontsize=11)
    ax.set_title('Teff+logg vs Teff Only', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(h, ax=ax, label='Count')

    # Add correlation
    corr, _ = stats.pearsonr(teff_only[mask], teff_logg[mask])
    ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Top right: Teff+Clustering vs Teff Only
    ax = axes[0, 1]
    h = ax.hexbin(teff_only[mask], teff_cluster[mask], gridsize=60, cmap='Greens', mincnt=1)
    ax.plot([3000, 15000], [3000, 15000], 'r--', linewidth=2, alpha=0.7, label='1:1 line')
    ax.set_xlabel('Teff Only (K)', fontsize=11)
    ax.set_ylabel('Teff + Clustering (K)', fontsize=11)
    ax.set_title('Teff+Clustering vs Teff Only', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(h, ax=ax, label='Count')

    corr, _ = stats.pearsonr(teff_only[mask], teff_cluster[mask])
    ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom left: Teff+logg vs Teff+Clustering
    ax = axes[1, 0]
    h = ax.hexbin(teff_cluster[mask], teff_logg[mask], gridsize=60, cmap='Oranges', mincnt=1)
    ax.plot([3000, 15000], [3000, 15000], 'r--', linewidth=2, alpha=0.7, label='1:1 line')
    ax.set_xlabel('Teff + Clustering (K)', fontsize=11)
    ax.set_ylabel('Teff + logg (K)', fontsize=11)
    ax.set_title('Teff+logg vs Teff+Clustering', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(h, ax=ax, label='Count')

    corr, _ = stats.pearsonr(teff_cluster[mask], teff_logg[mask])
    ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom right: Temperature distributions
    ax = axes[1, 1]
    bins = np.linspace(3000, 15000, 60)
    ax.hist(teff_only[mask], bins=bins, alpha=0.5, label='Teff Only', color='C0', density=True)
    ax.hist(teff_logg[mask], bins=bins, alpha=0.5, label='Teff + logg', color='C1', density=True)
    ax.hist(teff_cluster[mask], bins=bins, alpha=0.5, label='Teff + Clustering', color='C2', density=True)
    ax.axvline(teff_only[mask].mean(), color='C0', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(teff_logg[mask].mean(), color='C1', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(teff_cluster[mask].mean(), color='C2', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Temperature Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'temperature_comparison_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 2: Uncertainty Comparison
    # ========================================================================
    print("\n[2/4] Creating uncertainty comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty Comparison: Three Models', fontsize=16, fontweight='bold')

    # Top left: Uncertainty distributions
    ax = axes[0, 0]
    bins_unc = np.linspace(0, 2000, 60)
    ax.hist(unc_only, bins=bins_unc, alpha=0.5, label='Teff Only', color='C0', density=True)
    ax.hist(unc_logg, bins=bins_unc, alpha=0.5, label='Teff + logg', color='C1', density=True)
    ax.hist(unc_cluster, bins=bins_unc, alpha=0.5, label='Teff + Clustering', color='C2', density=True)
    ax.axvline(unc_only.mean(), color='C0', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(unc_logg.mean(), color='C1', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(unc_cluster.mean(), color='C2', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Uncertainty Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add statistics
    stats_text = f"Teff Only: μ={unc_only.mean():.0f}K\n"
    stats_text += f"Teff + logg: μ={unc_logg.mean():.0f}K\n"
    stats_text += f"Teff + Clustering: μ={unc_cluster.mean():.0f}K"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Top right: Cumulative uncertainty
    ax = axes[0, 1]
    sorted_only = np.sort(unc_only)
    sorted_logg = np.sort(unc_logg)
    sorted_cluster = np.sort(unc_cluster)
    cumulative = np.linspace(0, 1, len(sorted_only))
    ax.plot(sorted_only, cumulative, label='Teff Only', linewidth=2, color='C0')
    ax.plot(sorted_logg, cumulative, label='Teff + logg', linewidth=2, color='C1')
    ax.plot(sorted_cluster, cumulative, label='Teff + Clustering', linewidth=2, color='C2')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title('Cumulative Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2000)
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom left: Uncertainty vs Temperature (Teff+logg)
    ax = axes[1, 0]
    mask_unc = (teff_logg > 3000) & (teff_logg < 15000) & (unc_logg < 2000)
    h = ax.hexbin(teff_logg[mask_unc], unc_logg[mask_unc], gridsize=50, cmap='viridis', mincnt=1)
    ax.set_xlabel('Teff + logg (K)', fontsize=11)
    ax.set_ylabel('Uncertainty (K)', fontsize=11)
    ax.set_title('Teff+logg: Uncertainty vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    # Bottom right: Box plot comparison
    ax = axes[1, 1]
    data_box = [unc_only[unc_only < 2000],
                unc_logg[unc_logg < 2000],
                unc_cluster[unc_cluster < 2000]]
    bp = ax.boxplot(data_box, labels=['Teff Only', 'Teff + logg', 'Teff + Clustering'],
                    showfliers=False, patch_artist=True)
    colors = ['C0', 'C1', 'C2']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Uncertainty (K)', fontsize=11)
    ax.set_title('Uncertainty Box Plot Comparison', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'uncertainty_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 3: Difference Analysis
    # ========================================================================
    print("\n[3/4] Creating difference analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Difference Analysis', fontsize=16, fontweight='bold')

    diff_logg_only = teff_logg - teff_only
    diff_cluster_only = teff_cluster - teff_only
    diff_logg_cluster = teff_logg - teff_cluster

    # Top left: Teff+logg - Teff Only
    ax = axes[0, 0]
    mask_diff = np.abs(diff_logg_only) < 2000
    ax.hist(diff_logg_only[mask_diff], bins=60, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(diff_logg_only.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {diff_logg_only.mean():.1f} K')
    ax.set_xlabel('Teff Difference (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Teff+logg - Teff Only', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Top right: Teff+Clustering - Teff Only
    ax = axes[0, 1]
    mask_diff2 = np.abs(diff_cluster_only) < 2000
    ax.hist(diff_cluster_only[mask_diff2], bins=60, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(diff_cluster_only.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {diff_cluster_only.mean():.1f} K')
    ax.set_xlabel('Teff Difference (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Teff+Clustering - Teff Only', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom left: Teff+logg - Teff+Clustering
    ax = axes[1, 0]
    mask_diff3 = np.abs(diff_logg_cluster) < 2000
    ax.hist(diff_logg_cluster[mask_diff3], bins=60, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(diff_logg_cluster.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {diff_logg_cluster.mean():.1f} K')
    ax.set_xlabel('Teff Difference (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Teff+logg - Teff+Clustering', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom right: Difference vs Temperature
    ax = axes[1, 1]
    mask_hex = mask & mask_diff
    h = ax.hexbin(teff_only[mask_hex], diff_logg_only[mask_hex], gridsize=50, cmap='RdBu_r', mincnt=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Teff Only (K)', fontsize=11)
    ax.set_ylabel('Teff+logg - Teff Only (K)', fontsize=11)
    ax.set_title('Difference vs Temperature', fontsize=12, fontweight='bold')
    plt.colorbar(h, ax=ax, label='Count')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'temperature_differences.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 4: Summary Statistics
    # ========================================================================
    print("\n[4/4] Creating summary statistics plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')

    # Top left: Mean temperature comparison
    ax = axes[0, 0]
    models = ['Teff Only', 'Teff + logg', 'Teff + Clustering']
    means = [teff_only.mean(), teff_logg.mean(), teff_cluster.mean()]
    stds = [teff_only.std(), teff_logg.std(), teff_cluster.std()]
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['C0', 'C1', 'C2'])
    ax.set_ylabel('Mean Temperature (K)', fontsize=11)
    ax.set_title('Mean Temperature Predictions', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(i, mean + std + 50, f'{mean:.0f}±{std:.0f}K',
                ha='center', fontsize=9)

    # Top right: Mean uncertainty comparison
    ax = axes[0, 1]
    unc_means = [unc_only.mean(), unc_logg.mean(), unc_cluster.mean()]
    unc_stds = [unc_only.std(), unc_logg.std(), unc_cluster.std()]
    bars = ax.bar(x_pos, unc_means, yerr=unc_stds, capsize=5, alpha=0.7, color=['C0', 'C1', 'C2'])
    ax.set_ylabel('Mean Uncertainty (K)', fontsize=11)
    ax.set_title('Mean Uncertainty Estimates', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, unc_means, unc_stds)):
        ax.text(i, mean + std + 20, f'{mean:.0f}±{std:.0f}K',
                ha='center', fontsize=9)

    # Bottom left: RMS differences
    ax = axes[1, 0]
    rms_data = [
        ('logg - Only', np.sqrt(np.mean(diff_logg_only**2))),
        ('Cluster - Only', np.sqrt(np.mean(diff_cluster_only**2))),
        ('logg - Cluster', np.sqrt(np.mean(diff_logg_cluster**2)))
    ]
    labels = [x[0] for x in rms_data]
    values = [x[1] for x in rms_data]
    bars = ax.bar(range(len(labels)), values, alpha=0.7, color=['purple', 'green', 'orange'])
    ax.set_ylabel('RMS Difference (K)', fontsize=11)
    ax.set_title('RMS Temperature Differences', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(i, val + 5, f'{val:.0f} K', ha='center', fontsize=10)

    # Bottom right: Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'Teff Only', 'Teff + logg', 'Teff + Clustering'],
        ['Mean Teff (K)', f'{teff_only.mean():.0f}', f'{teff_logg.mean():.0f}', f'{teff_cluster.mean():.0f}'],
        ['Median Teff (K)', f'{np.median(teff_only):.0f}', f'{np.median(teff_logg):.0f}', f'{np.median(teff_cluster):.0f}'],
        ['Mean Unc (K)', f'{unc_only.mean():.0f}', f'{unc_logg.mean():.0f}', f'{unc_cluster.mean():.0f}'],
        ['Median Unc (K)', f'{np.median(unc_only):.0f}', f'{np.median(unc_logg):.0f}', f'{np.median(unc_cluster):.0f}'],
        ['P90 Unc (K)', f'{np.percentile(unc_only, 90):.0f}', f'{np.percentile(unc_logg, 90):.0f}', f'{np.percentile(unc_cluster, 90):.0f}'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.2, 0.9, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = output_dir / 'summary_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    print(f"\n✓ All plots saved to: {output_dir}")

def main():
    # Load predictions
    df = load_predictions()

    # Compute statistics
    df_complete = compute_statistics(df)

    # Create plots
    create_comparison_plots(df_complete)

    print("\n" + "=" * 80)
    print("THREE MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: reports/figures/three_model_comparison")
    print("\nPlots created:")
    print("  1. temperature_comparison_scatter.png - Pairwise scatter plots")
    print("  2. uncertainty_comparison.png - Uncertainty distributions and analysis")
    print("  3. temperature_differences.png - Difference histograms and trends")
    print("  4. summary_statistics.png - Summary bar plots and statistics table")
    print("\n✓ Done!")

if __name__ == '__main__':
    main()
