"""
Create "best-of-three" ensemble by selecting prediction with lowest uncertainty.

For each object, choose the model (Teff Only, Teff+logg, or Teff+Clustering)
that produces the lowest uncertainty estimate.

This maximizes the number of high-confidence predictions.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_predictions():
    """Load predictions from all three models."""
    print("=" * 80)
    print("LOADING PREDICTIONS FROM THREE MODELS")
    print("=" * 80)

    # Model 1: Teff Only
    print("\n[1/3] Loading Teff Only...")
    df1 = pl.read_parquet('data/processed/teff_predictions_for_predicted_teff.parquet')
    df1 = df1.select([
        'source_id', 'ra', 'dec',
        'g', 'bp', 'rp', 'bp_rp',
        'teff_source', 'logg_source',
        pl.col('teff_predicted').alias('teff_only'),
        pl.col('teff_uncertainty').alias('unc_only')
    ])
    print(f"  Loaded {len(df1):,} predictions")

    # Model 2: Teff + logg
    print("\n[2/3] Loading Teff + logg...")
    df2 = pl.read_parquet('data/processed/teff_predictions_with_logg_propagated_final.parquet')
    df2 = df2.select([
        'source_id',
        pl.col('teff_predicted').alias('teff_logg'),
        pl.col('teff_unc_total').alias('unc_logg'),
        'gradient_teff_logg'
    ])
    print(f"  Loaded {len(df2):,} predictions")

    # Model 3: Teff + Clustering
    print("\n[3/3] Loading Teff + Clustering...")
    df3 = pl.read_parquet('data/processed/teff_predictions_with_clustering.parquet')
    df3 = df3.select([
        'source_id',
        pl.col('teff_predicted').alias('teff_cluster'),
        pl.col('teff_uncertainty').alias('unc_cluster')
    ])
    print(f"  Loaded {len(df3):,} predictions")

    # Merge all three
    print("\n[4/4] Merging...")
    df_merged = (df1
                 .join(df2, on='source_id', how='inner')
                 .join(df3, on='source_id', how='inner'))

    print(f"  Merged: {len(df_merged):,} objects with all three predictions")

    return df_merged

def select_best_by_uncertainty(df):
    """For each object, select the model with lowest uncertainty."""
    print("\n" + "=" * 80)
    print("SELECTING BEST PREDICTIONS BY UNCERTAINTY")
    print("=" * 80)

    # Create model selection based on lowest uncertainty
    df_best = df.with_columns([
        # Find which model has minimum uncertainty
        pl.when(
            (pl.col('unc_only') <= pl.col('unc_logg')) &
            (pl.col('unc_only') <= pl.col('unc_cluster'))
        ).then(pl.lit('teff_only'))
        .when(
            (pl.col('unc_logg') <= pl.col('unc_only')) &
            (pl.col('unc_logg') <= pl.col('unc_cluster'))
        ).then(pl.lit('teff_logg'))
        .otherwise(pl.lit('teff_cluster'))
        .alias('best_model')
    ])

    # Select the corresponding prediction and uncertainty
    df_best = df_best.with_columns([
        pl.when(pl.col('best_model') == 'teff_only')
          .then(pl.col('teff_only'))
        .when(pl.col('best_model') == 'teff_logg')
          .then(pl.col('teff_logg'))
        .otherwise(pl.col('teff_cluster'))
        .alias('teff_best'),

        pl.when(pl.col('best_model') == 'teff_only')
          .then(pl.col('unc_only'))
        .when(pl.col('best_model') == 'teff_logg')
          .then(pl.col('unc_logg'))
        .otherwise(pl.col('unc_cluster'))
        .alias('unc_best')
    ])

    # Statistics
    print("\nModel selection statistics:")
    model_counts = df_best.group_by('best_model').agg(pl.count()).sort('best_model')

    total = len(df_best)
    for row in model_counts.iter_rows():
        model, count = row
        pct = count / total * 100
        print(f"  {model}: {count:,} ({pct:.1f}%)")

    return df_best

def analyze_improvements(df_best):
    """Analyze improvements from best-of-three selection."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    # Convert to numpy for analysis
    unc_only = df_best['unc_only'].to_numpy()
    unc_logg = df_best['unc_logg'].to_numpy()
    unc_cluster = df_best['unc_cluster'].to_numpy()
    unc_best = df_best['unc_best'].to_numpy()

    print("\n" + "-" * 80)
    print("UNCERTAINTY COMPARISON")
    print("-" * 80)

    for name, unc in [('Teff Only', unc_only),
                      ('Teff + logg', unc_logg),
                      ('Teff + Clustering', unc_cluster),
                      ('Best-of-Three', unc_best)]:
        print(f"\n{name}:")
        print(f"  Mean:   {unc.mean():.1f} K")
        print(f"  Median: {np.median(unc):.1f} K")
        print(f"  Std:    {unc.std():.1f} K")
        print(f"  P90:    {np.percentile(unc, 90):.1f} K")
        print(f"  P95:    {np.percentile(unc, 95):.1f} K")

    print("\n" + "-" * 80)
    print("HIGH-CONFIDENCE OBJECT COUNTS")
    print("-" * 80)

    thresholds = [300, 400, 500, 600, 700, 800, 900, 1000]

    print("\nUncertainty < Threshold:")
    print(f"{'Threshold':<12} {'Teff Only':<12} {'Teff+logg':<12} {'Teff+Cluster':<14} {'Best-of-3':<12} {'Improvement'}")
    print("-" * 95)

    for thresh in thresholds:
        n_only = (unc_only < thresh).sum()
        n_logg = (unc_logg < thresh).sum()
        n_cluster = (unc_cluster < thresh).sum()
        n_best = (unc_best < thresh).sum()

        # Best individual model
        n_best_individual = max(n_only, n_logg, n_cluster)
        improvement = n_best - n_best_individual
        improvement_pct = (improvement / n_best_individual) * 100 if n_best_individual > 0 else 0

        print(f"< {thresh:3d} K     {n_only:>8,}     {n_logg:>8,}     {n_cluster:>9,}     {n_best:>8,}     +{improvement:,} ({improvement_pct:+.1f}%)")

    # Calculate mean improvement
    print("\n" + "-" * 80)
    print("MEAN IMPROVEMENT")
    print("-" * 80)

    best_individual_mean = min(unc_only.mean(), unc_logg.mean(), unc_cluster.mean())
    improvement_abs = best_individual_mean - unc_best.mean()
    improvement_pct = (improvement_abs / best_individual_mean) * 100

    print(f"\nBest individual model mean: {best_individual_mean:.1f} K")
    print(f"Best-of-three mean:         {unc_best.mean():.1f} K")
    print(f"Improvement:                {improvement_abs:.1f} K ({improvement_pct:.1f}%)")

    return df_best

def create_comparison_plots(df_best):
    """Create visualization comparing methods."""
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80)

    output_dir = Path('reports/figures/best_uncertainty_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    unc_only = df_best['unc_only'].to_numpy()
    unc_logg = df_best['unc_logg'].to_numpy()
    unc_cluster = df_best['unc_cluster'].to_numpy()
    unc_best = df_best['unc_best'].to_numpy()

    # ========================================================================
    # PLOT 1: Uncertainty Distributions
    # ========================================================================
    print("\n[1/3] Creating distribution comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best-of-Three Uncertainty Selection', fontsize=16, fontweight='bold')

    # Top left: Histogram comparison
    ax = axes[0, 0]
    bins = np.linspace(0, 2000, 60)
    ax.hist(unc_only, bins=bins, alpha=0.4, label='Teff Only', color='C0', density=True)
    ax.hist(unc_logg, bins=bins, alpha=0.4, label='Teff + logg', color='C1', density=True)
    ax.hist(unc_cluster, bins=bins, alpha=0.4, label='Teff + Clustering', color='C2', density=True)
    ax.hist(unc_best, bins=bins, alpha=0.7, label='Best-of-Three', color='red',
            density=True, histtype='step', linewidth=2.5)

    ax.axvline(unc_only.mean(), color='C0', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(unc_logg.mean(), color='C1', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(unc_cluster.mean(), color='C2', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(unc_best.mean(), color='red', linestyle='--', linewidth=2.5)

    ax.set_xlabel('Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Uncertainty Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Top right: Cumulative distributions
    ax = axes[0, 1]
    sorted_only = np.sort(unc_only)
    sorted_logg = np.sort(unc_logg)
    sorted_cluster = np.sort(unc_cluster)
    sorted_best = np.sort(unc_best)
    cumulative = np.linspace(0, 1, len(sorted_only))

    ax.plot(sorted_only, cumulative, label='Teff Only', linewidth=2, color='C0', alpha=0.7)
    ax.plot(sorted_logg, cumulative, label='Teff + logg', linewidth=2, color='C1', alpha=0.7)
    ax.plot(sorted_cluster, cumulative, label='Teff + Clustering', linewidth=2, color='C2', alpha=0.7)
    ax.plot(sorted_best, cumulative, label='Best-of-Three', linewidth=2.5, color='red')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Uncertainty (K)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2000)
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom left: High-confidence object counts
    ax = axes[1, 0]
    thresholds = [300, 400, 500, 600, 700, 800, 900, 1000]
    counts_only = [(unc_only < t).sum() for t in thresholds]
    counts_logg = [(unc_logg < t).sum() for t in thresholds]
    counts_cluster = [(unc_cluster < t).sum() for t in thresholds]
    counts_best = [(unc_best < t).sum() for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.2

    ax.bar(x - 1.5*width, counts_only, width, label='Teff Only', color='C0', alpha=0.7)
    ax.bar(x - 0.5*width, counts_logg, width, label='Teff + logg', color='C1', alpha=0.7)
    ax.bar(x + 0.5*width, counts_cluster, width, label='Teff + Clustering', color='C2', alpha=0.7)
    ax.bar(x + 1.5*width, counts_best, width, label='Best-of-Three', color='red', alpha=0.8)

    ax.set_xlabel('Uncertainty Threshold (K)', fontsize=11)
    ax.set_ylabel('Number of Objects', fontsize=11)
    ax.set_title('High-Confidence Object Counts', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'<{t}' for t in thresholds], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Bottom right: Model selection pie chart
    ax = axes[1, 1]
    model_counts = df_best.group_by('best_model').agg(pl.count()).sort('best_model')
    labels = []
    sizes = []
    colors_pie = {'teff_only': 'C0', 'teff_logg': 'C1', 'teff_cluster': 'C2'}
    colors = []

    for row in model_counts.iter_rows():
        model, count = row
        labels.append(model.replace('_', ' ').title())
        sizes.append(count)
        colors.append(colors_pie[model])

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax.set_title('Model Selection Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'best_uncertainty_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # ========================================================================
    # PLOT 2: Improvement Analysis
    # ========================================================================
    print("\n[2/3] Creating improvement analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best-of-Three Improvement Analysis', fontsize=16, fontweight='bold')

    # Top left: Mean uncertainty comparison
    ax = axes[0, 0]
    models = ['Teff Only', 'Teff + logg', 'Teff + Clustering', 'Best-of-Three']
    means = [unc_only.mean(), unc_logg.mean(), unc_cluster.mean(), unc_best.mean()]
    stds = [unc_only.std(), unc_logg.std(), unc_cluster.std(), unc_best.std()]
    colors_bar = ['C0', 'C1', 'C2', 'red']

    bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5,
                   color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars[-1].set_alpha(1.0)  # Highlight best-of-three

    ax.set_ylabel('Mean Uncertainty (K)', fontsize=11)
    ax.set_title('Mean Uncertainty Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(i, mean + std + 20, f'{mean:.0f}±{std:.0f}K',
                ha='center', fontsize=9, fontweight='bold' if i == 3 else 'normal')

    # Top right: Percentage improvement
    ax = axes[0, 1]
    best_individual = min(unc_only.mean(), unc_logg.mean(), unc_cluster.mean())
    improvements_pct = []
    thresholds_plot = [300, 400, 500, 600, 700, 800, 900, 1000]

    for thresh in thresholds_plot:
        n_best_individual = max((unc_only < thresh).sum(),
                                 (unc_logg < thresh).sum(),
                                 (unc_cluster < thresh).sum())
        n_best = (unc_best < thresh).sum()
        improvement = (n_best - n_best_individual) / n_best_individual * 100
        improvements_pct.append(improvement)

    ax.bar(range(len(thresholds_plot)), improvements_pct, color='green', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Uncertainty Threshold (K)', fontsize=11)
    ax.set_ylabel('Improvement over Best Individual (%)', fontsize=11)
    ax.set_title('Relative Improvement in Object Count', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(thresholds_plot)))
    ax.set_xticklabels([f'<{t}' for t in thresholds_plot], rotation=45)
    ax.grid(alpha=0.3, axis='y')

    # Bottom left: Uncertainty reduction per object
    ax = axes[1, 0]
    unc_reduction = np.minimum(unc_only, np.minimum(unc_logg, unc_cluster)) - unc_best
    mask = unc_reduction < 500  # Filter extreme outliers

    ax.hist(unc_reduction[mask], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(unc_reduction.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {unc_reduction.mean():.1f} K')
    ax.axvline(np.median(unc_reduction), color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {np.median(unc_reduction):.1f} K')

    ax.set_xlabel('Uncertainty Reduction (K)', fontsize=11)
    ax.set_ylabel('Number of Objects', fontsize=11)
    ax.set_title('Per-Object Uncertainty Reduction', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom right: Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'Best Individual', 'Best-of-Three', 'Improvement'],
        ['Mean Unc (K)', f'{best_individual:.0f}', f'{unc_best.mean():.0f}',
         f'{best_individual - unc_best.mean():.0f} ({(best_individual - unc_best.mean())/best_individual*100:.1f}%)'],
        ['Median Unc (K)', f'{min(np.median(unc_only), np.median(unc_logg), np.median(unc_cluster)):.0f}',
         f'{np.median(unc_best):.0f}', ''],
        ['Unc < 500 K', f'{max((unc_only<500).sum(), (unc_logg<500).sum(), (unc_cluster<500).sum()):,}',
         f'{(unc_best<500).sum():,}',
         f'+{(unc_best<500).sum() - max((unc_only<500).sum(), (unc_logg<500).sum(), (unc_cluster<500).sum()):,}'],
        ['Unc < 400 K', f'{max((unc_only<400).sum(), (unc_logg<400).sum(), (unc_cluster<400).sum()):,}',
         f'{(unc_best<400).sum():,}',
         f'+{(unc_best<400).sum() - max((unc_only<400).sum(), (unc_logg<400).sum(), (unc_cluster<400).sum()):,}'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.2, 0.9, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight improvement column
    for i in range(1, 5):
        table[(i, 3)].set_facecolor('#e8f5e9')
        table[(i, 3)].set_text_props(weight='bold')

    ax.set_title('Improvement Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = output_dir / 'improvement_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    print(f"\n✓ All plots saved to: {output_dir}")

def save_best_predictions(df_best):
    """Save the best-of-three predictions."""
    print("\n" + "=" * 80)
    print("SAVING BEST-OF-THREE PREDICTIONS")
    print("=" * 80)

    # Select output columns
    df_output = df_best.select([
        'source_id', 'ra', 'dec',
        'g', 'bp', 'rp', 'bp_rp',
        'teff_source', 'logg_source',
        'best_model',
        'teff_best',
        'unc_best',
        # Keep all three predictions for reference
        'teff_only', 'unc_only',
        'teff_logg', 'unc_logg',
        'teff_cluster', 'unc_cluster',
        'gradient_teff_logg'  # Keep for Teff+logg cases
    ])

    output_path = Path('data/processed/teff_predictions_best_of_three.parquet')
    df_output.write_parquet(output_path)

    print(f"\n✓ Saved: {output_path}")
    print(f"  Objects: {len(df_output):,}")
    print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")

    print("\n  Columns:")
    for col in df_output.columns:
        print(f"    - {col}")

    print("\n✓ Best-of-three dataset ready!")
    print("\nUsage:")
    print("  - teff_best: Temperature from model with lowest uncertainty")
    print("  - unc_best: Uncertainty (lowest among three models)")
    print("  - best_model: Which model was selected (teff_only, teff_logg, or teff_cluster)")

def main():
    # Load all predictions
    df = load_all_predictions()

    # Select best by uncertainty
    df_best = select_best_by_uncertainty(df)

    # Analyze improvements
    df_best = analyze_improvements(df_best)

    # Create comparison plots
    create_comparison_plots(df_best)

    # Save results
    save_best_predictions(df_best)

    print("\n" + "=" * 80)
    print("BEST-OF-THREE ENSEMBLE COMPLETE")
    print("=" * 80)
    print("\n✓ Successfully created intelligent ensemble based on uncertainty selection!")

if __name__ == '__main__':
    main()
