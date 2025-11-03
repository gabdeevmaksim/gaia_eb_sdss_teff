"""
Plot weighted training distributions vs prediction distributions.

This script visualizes how well the reweighted training set matches
the prediction set distribution.

Usage:
    python scripts/plot_weighted_distributions.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from src.config import get_config

# Configuration
config = get_config()
MISSING_VALUE = config.get('processing', 'missing_value')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Color features
COLOR_FEATURES = ['g_r_color', 'r_i_color', 'i_z_color', 'bp_rp']


def plot_weighted_vs_prediction(train_weighted_df, predict_df, features=COLOR_FEATURES):
    """
    Create comparison plots of weighted training vs prediction distributions.
    """
    logger.info("Creating weighted distribution comparison plots...")

    output_dir = Path(config.get_path('figures')) / 'distribution_matching'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Individual plots for each feature
    for feature in features:
        logger.info(f"Plotting {feature}...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Get data
        train_vals = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        train_weights = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)['sample_weight'].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

        # Normalize weights for histogram
        train_weights_norm = train_weights / train_weights.sum() * len(train_vals)

        # Define bins
        bins = np.linspace(
            min(train_vals.min(), predict_vals.min()),
            max(train_vals.max(), predict_vals.max()),
            50
        )

        # Plot 1: Weighted histogram
        ax1.hist(train_vals, bins=bins, weights=train_weights_norm,
                alpha=0.6, label='Weighted Training', density=True,
                color='blue', edgecolor='black', linewidth=0.5)
        ax1.hist(predict_vals, bins=bins,
                alpha=0.6, label='Prediction', density=True,
                color='red', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel(f'{feature}', fontsize=13)
        ax1.set_ylabel('Density', fontsize=13)
        ax1.set_title(f'{feature} - Weighted Histogram Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        train_mean = np.average(train_vals, weights=train_weights)
        train_std = np.sqrt(np.average((train_vals - train_mean)**2, weights=train_weights))
        predict_mean = np.mean(predict_vals)
        predict_std = np.std(predict_vals)

        stats_text = f'Weighted Train: μ={train_mean:.3f}, σ={train_std:.3f}\n'
        stats_text += f'Prediction: μ={predict_mean:.3f}, σ={predict_std:.3f}\n'
        stats_text += f'Difference: Δμ={predict_mean - train_mean:+.3f}'

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Weighted KDE
        # For weighted KDE, we'll use resampling
        np.random.seed(42)
        train_weights_prob = train_weights / train_weights.sum()
        train_resampled = np.random.choice(train_vals, size=50000, replace=True, p=train_weights_prob)

        kde_train = stats.gaussian_kde(train_resampled)
        kde_predict = stats.gaussian_kde(predict_vals)

        x_range = np.linspace(bins[0], bins[-1], 200)

        ax2.plot(x_range, kde_train(x_range), label='Weighted Training',
                linewidth=2.5, color='blue')
        ax2.plot(x_range, kde_predict(x_range), label='Prediction',
                linewidth=2.5, color='red')
        ax2.fill_between(x_range, kde_train(x_range), alpha=0.3, color='blue')
        ax2.fill_between(x_range, kde_predict(x_range), alpha=0.3, color='red')

        ax2.set_xlabel(f'{feature}', fontsize=13)
        ax2.set_ylabel('Density', fontsize=13)
        ax2.set_title(f'{feature} - Weighted KDE Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        output_file = output_dir / f'{feature}_weighted_comparison.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  ✓ Saved: {output_file.name}")

    # 2. Combined 2x2 grid
    logger.info("Creating combined weighted comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Get data
        train_vals = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        train_weights = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)['sample_weight'].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

        # Normalize weights
        train_weights_norm = train_weights / train_weights.sum() * len(train_vals)

        bins = np.linspace(
            min(train_vals.min(), predict_vals.min()),
            max(train_vals.max(), predict_vals.max()),
            40
        )

        ax.hist(train_vals, bins=bins, weights=train_weights_norm,
               alpha=0.6, label='Weighted Training', density=True,
               color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(predict_vals, bins=bins,
               alpha=0.6, label='Prediction', density=True,
               color='red', edgecolor='black', linewidth=0.5)

        ax.set_xlabel(f'{feature}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{feature}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add mean difference
        train_mean = np.average(train_vals, weights=train_weights)
        predict_mean = np.mean(predict_vals)
        diff = predict_mean - train_mean

        ax.text(0.98, 0.98, f'Δμ = {diff:+.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    fig.suptitle('Weighted Training vs Prediction Distribution Comparison',
                fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout()

    output_file = output_dir / 'all_colors_weighted_comparison.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved: {output_file.name}")

    # 3. Before/After comparison for one feature (most significant: bp_rp)
    logger.info("Creating before/after comparison for bp_rp...")

    feature = 'bp_rp'

    # Load original unweighted training data
    train_unweighted_df = pl.read_parquet(
        config.get_dataset_path('eb_unified_features_engineered_train', 'processed')
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get data
    train_orig_vals = train_unweighted_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
    train_weighted_vals = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
    train_weights = train_weighted_df.filter(pl.col(feature) != MISSING_VALUE)['sample_weight'].to_numpy()
    predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

    train_weights_norm = train_weights / train_weights.sum() * len(train_weighted_vals)

    bins = np.linspace(
        min(train_orig_vals.min(), predict_vals.min()),
        max(train_orig_vals.max(), predict_vals.max()),
        50
    )

    # Before
    ax1.hist(train_orig_vals, bins=bins, alpha=0.6, label='Original Training',
            density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax1.hist(predict_vals, bins=bins, alpha=0.6, label='Prediction',
            density=True, color='red', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel(f'{feature}', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title(f'BEFORE Reweighting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    orig_mean = np.mean(train_orig_vals)
    pred_mean = np.mean(predict_vals)
    ax1.text(0.02, 0.98, f'Δμ = {pred_mean - orig_mean:+.3f}',
            transform=ax1.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # After
    ax2.hist(train_weighted_vals, bins=bins, weights=train_weights_norm,
            alpha=0.6, label='Weighted Training', density=True,
            color='blue', edgecolor='black', linewidth=0.5)
    ax2.hist(predict_vals, bins=bins, alpha=0.6, label='Prediction',
            density=True, color='red', edgecolor='black', linewidth=0.5)

    ax2.set_xlabel(f'{feature}', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title(f'AFTER Reweighting', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    weighted_mean = np.average(train_weighted_vals, weights=train_weights)
    ax2.text(0.02, 0.98, f'Δμ = {pred_mean - weighted_mean:+.3f}',
            transform=ax2.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    fig.suptitle(f'{feature}: Impact of Distribution Matching',
                fontsize=15, fontweight='bold')
    fig.tight_layout()

    output_file = output_dir / f'{feature}_before_after_reweighting.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved: {output_file.name}")

    logger.info(f"\nAll plots saved to: {output_dir}")


def main():
    logger.info("="*70)
    logger.info("PLOTTING WEIGHTED DISTRIBUTIONS")
    logger.info("="*70)

    # Load data
    logger.info("\nLoading datasets...")

    train_weighted_path = Path(config.get_path('processed')) / 'eb_unified_features_engineered_train_weighted.parquet'
    predict_path = config.get_dataset_path('eb_unified_features_engineered_predict', 'processed')

    train_weighted_df = pl.read_parquet(train_weighted_path)
    predict_df = pl.read_parquet(predict_path)

    logger.info(f"Weighted training: {len(train_weighted_df):,} objects")
    logger.info(f"Prediction: {len(predict_df):,} objects")

    # Create plots
    plot_weighted_vs_prediction(train_weighted_df, predict_df)

    logger.info("\n" + "="*70)
    logger.info("PLOTTING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
