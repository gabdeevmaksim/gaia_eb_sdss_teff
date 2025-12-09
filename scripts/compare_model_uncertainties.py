#!/usr/bin/env python3
"""
Compare RF Uncertainties: New Corrected+Log Model vs Previous Models

Analyzes how many objects fall under different uncertainty levels for:
- New model: Gaia → log(corrected Teff)
- Previous: Best-of-three ensemble OR other specified model
"""

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Uncertainty bins (in Kelvin)
UNCERTAINTY_BINS = [
    (0, 200, "Excellent (<200K)"),
    (200, 300, "Very Good (200-300K)"),
    (300, 500, "Good (300-500K)"),
    (500, 800, "Fair (500-800K)"),
    (800, 1200, "Poor (800-1200K)"),
    (1200, np.inf, "Very Poor (>1200K)")
]


def load_predictions(file_path, model_name):
    """Load prediction file and extract key columns"""
    print(f"\n{'='*80}")
    print(f"Loading: {model_name}")
    print(f"{'='*80}")

    df = pl.read_parquet(file_path)
    print(f"Total objects: {len(df):,}")

    # Check what columns are available
    print(f"Available columns: {df.columns[:10]}...")

    # Try to identify uncertainty column
    unc_cols = [col for col in df.columns if 'unc' in col.lower() or 'uncertainty' in col.lower()]
    print(f"Uncertainty columns found: {unc_cols}")

    return df


def analyze_uncertainty_distribution(df, unc_column, model_name):
    """Analyze uncertainty distribution and bin objects"""

    uncertainties = df[unc_column].to_numpy()

    print(f"\n{model_name} - Uncertainty Statistics:")
    print(f"  Mean:   {np.mean(uncertainties):.1f} K")
    print(f"  Median: {np.median(uncertainties):.1f} K")
    print(f"  Std:    {np.std(uncertainties):.1f} K")
    print(f"  Min:    {np.min(uncertainties):.1f} K")
    print(f"  Max:    {np.max(uncertainties):.1f} K")

    # Count objects per uncertainty bin
    print(f"\nObjects by Uncertainty Level:")
    print(f"{'Bin':<25} {'Count':>12} {'Percent':>10}")
    print("-" * 50)

    bin_counts = []
    for tmin, tmax, label in UNCERTAINTY_BINS:
        if tmax == np.inf:
            mask = uncertainties >= tmin
        else:
            mask = (uncertainties >= tmin) & (uncertainties < tmax)

        count = np.sum(mask)
        percent = 100 * count / len(uncertainties)

        print(f"{label:<25} {count:>12,} {percent:>9.1f}%")
        bin_counts.append((label, count, percent))

    return bin_counts, uncertainties


def compare_models(new_df, new_unc_col, prev_df, prev_unc_col):
    """Compare two models side by side"""

    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")

    # Analyze both models
    new_bins, new_unc = analyze_uncertainty_distribution(new_df, new_unc_col, "NEW: Corrected+Log Model")
    prev_bins, prev_unc = analyze_uncertainty_distribution(prev_df, prev_unc_col, "PREVIOUS: Uncorrected Log Model")

    # Side-by-side comparison
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Uncertainty Bin':<25} {'New Model':>15} {'Previous Model':>15} {'Improvement':>15}")
    print("-" * 72)

    for i, (label, tmin, tmax) in enumerate([(b[0], *UNCERTAINTY_BINS[i][:2]) for i, b in enumerate(new_bins)]):
        new_count = new_bins[i][1]
        prev_count = prev_bins[i][1]

        diff = new_count - prev_count
        diff_pct = 100 * diff / len(new_df) if len(new_df) > 0 else 0

        print(f"{label:<25} {new_count:>15,} {prev_count:>15,} {diff:>+14,} ({diff_pct:+.1f}%)")

    # Overall statistics comparison
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'New Model':>15} {'Previous Model':>15} {'Improvement':>15}")
    print("-" * 68)

    new_mean = np.mean(new_unc)
    prev_mean = np.mean(prev_unc)
    mean_improvement = prev_mean - new_mean
    mean_improvement_pct = 100 * mean_improvement / prev_mean

    print(f"{'Mean Uncertainty':<20} {new_mean:>14.1f}K {prev_mean:>14.1f}K {mean_improvement:>+14.1f}K ({mean_improvement_pct:+.1f}%)")

    new_median = np.median(new_unc)
    prev_median = np.median(prev_unc)
    median_improvement = prev_median - new_median
    median_improvement_pct = 100 * median_improvement / prev_median

    print(f"{'Median Uncertainty':<20} {new_median:>14.1f}K {prev_median:>14.1f}K {median_improvement:>+14.1f}K ({median_improvement_pct:+.1f}%)")

    return new_bins, prev_bins, new_unc, prev_unc


def plot_comparison(new_bins, prev_bins, new_unc, prev_unc, output_dir):
    """Create visualization comparing uncertainties"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [b[0] for b in new_bins]
    new_counts = [b[1] for b in new_bins]
    prev_counts = [b[1] for b in prev_bins]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, new_counts, width, label='New: Corrected+Log', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, prev_counts, width, label='Previous: Uncorrected Log', color='#3498db', alpha=0.8)

    ax.set_xlabel('Uncertainty Level')
    ax.set_ylabel('Number of Objects')
    ax.set_title('Object Count by Uncertainty Level: Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_comparison_bars.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'uncertainty_comparison_bars.png'}")
    plt.close()

    # 2. Histogram comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(new_unc, bins=50, alpha=0.7, edgecolor='black', color='#2ecc71')
    axes[0].axvline(np.mean(new_unc), color='red', linestyle='--',
                   label=f'Mean: {np.mean(new_unc):.1f}K')
    axes[0].axvline(np.median(new_unc), color='blue', linestyle='--',
                   label=f'Median: {np.median(new_unc):.1f}K')
    axes[0].set_xlabel('Uncertainty (K)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('New Model: Corrected+Log Uncertainty Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(prev_unc, bins=50, alpha=0.7, edgecolor='black', color='#3498db')
    axes[1].axvline(np.mean(prev_unc), color='red', linestyle='--',
                   label=f'Mean: {np.mean(prev_unc):.1f}K')
    axes[1].axvline(np.median(prev_unc), color='blue', linestyle='--',
                   label=f'Median: {np.median(prev_unc):.1f}K')
    axes[1].set_xlabel('Uncertainty (K)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Previous Model: Uncorrected Log Uncertainty Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_distributions.png'}")
    plt.close()

    # 3. Cumulative distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    new_sorted = np.sort(new_unc)
    prev_sorted = np.sort(prev_unc)

    new_cumulative = np.arange(1, len(new_sorted) + 1) / len(new_sorted) * 100
    prev_cumulative = np.arange(1, len(prev_sorted) + 1) / len(prev_sorted) * 100

    ax.plot(new_sorted, new_cumulative, label='New: Corrected+Log', color='#2ecc71', linewidth=2)
    ax.plot(prev_sorted, prev_cumulative, label='Previous: Uncorrected Log', color='#3498db', linewidth=2)

    # Add reference lines
    for unc_val, label in [(200, '200K'), (300, '300K'), (500, '500K'), (800, '800K')]:
        ax.axvline(unc_val, color='gray', linestyle=':', alpha=0.5)
        ax.text(unc_val, 95, label, rotation=90, va='top', fontsize=8)

    ax.set_xlabel('Uncertainty (K)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Uncertainty Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1500)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_cumulative.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_cumulative.png'}")
    plt.close()


def main():
    print("="*80)
    print("MODEL UNCERTAINTY COMPARISON")
    print("="*80)

    # Load new model predictions
    new_file = Path('data/processed/predictions_gaia_teff_corrected_log.parquet')
    new_df = load_predictions(new_file, "New: Corrected+Log Model")

    # Load previous model predictions (teff_only = uncorrected log model)
    prev_file = Path('data/processed/teff_predictions_best_of_three.parquet')
    prev_df = load_predictions(prev_file, "Previous: Gaia Log (Uncorrected)")

    # Identify uncertainty columns
    new_unc_col = 'teff_predicted_uncertainty'  # Adjust if needed
    prev_unc_col = 'unc_only'  # Teff-only model (uncorrected log)

    # Check if columns exist
    if new_unc_col not in new_df.columns:
        print(f"\nWarning: '{new_unc_col}' not found in new predictions")
        print(f"Available columns: {new_df.columns}")
        # Try to find the right column
        unc_cols = [c for c in new_df.columns if 'unc' in c.lower()]
        if unc_cols:
            new_unc_col = unc_cols[0]
            print(f"Using: {new_unc_col}")

    if prev_unc_col not in prev_df.columns:
        print(f"\nWarning: '{prev_unc_col}' not found in previous predictions")
        print(f"Available columns: {prev_df.columns}")
        # Try to find the right column
        unc_cols = [c for c in prev_df.columns if 'unc' in c.lower()]
        if unc_cols:
            prev_unc_col = unc_cols[0]
            print(f"Using: {prev_unc_col}")

    # Match objects (use source_id)
    print(f"\nMatching objects between datasets...")
    matched = new_df.join(prev_df, on='source_id', how='inner', suffix='_prev')
    print(f"Matched {len(matched):,} objects")

    # Get the right column names after join
    prev_unc_col_matched = prev_unc_col if prev_unc_col in matched.columns else f"{prev_unc_col}_right"

    # Compare matched objects
    new_bins, prev_bins, new_unc, prev_unc = compare_models(
        matched, new_unc_col, matched, prev_unc_col_matched
    )

    # Create visualizations
    output_dir = Path('reports/figures/uncertainty_comparison')
    plot_comparison(new_bins, prev_bins, new_unc, prev_unc, output_dir)

    # Save detailed results
    results_path = Path('reports/uncertainty_comparison_results.txt')
    print(f"\nSaving detailed results to: {results_path}")

    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL UNCERTAINTY COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write(f"New Model: Corrected+Log (log10(corrected Teff))\n")
        f.write(f"Previous Model: Uncorrected Log (log10(original Teff))\n")
        f.write(f"Matched Objects: {len(matched):,}\n\n")

        f.write("UNCERTAINTY DISTRIBUTION:\n")
        f.write(f"{'Bin':<25} {'New Model':>15} {'Previous Model':>15} {'Difference':>15}\n")
        f.write("-" * 72 + "\n")

        for i, (label, _, _) in enumerate(UNCERTAINTY_BINS):
            new_count = new_bins[i][1]
            prev_count = prev_bins[i][1]
            diff = new_count - prev_count

            f.write(f"{label:<25} {new_count:>15,} {prev_count:>15,} {diff:>+14,}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICS:\n")
        f.write(f"  New Mean:      {np.mean(new_unc):.1f} K\n")
        f.write(f"  Previous Mean: {np.mean(prev_unc):.1f} K\n")
        f.write(f"  Improvement:   {np.mean(prev_unc) - np.mean(new_unc):+.1f} K ({100*(np.mean(prev_unc) - np.mean(new_unc))/np.mean(prev_unc):+.1f}%)\n")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
