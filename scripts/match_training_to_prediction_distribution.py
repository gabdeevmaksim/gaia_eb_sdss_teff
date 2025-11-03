"""
Match training sample distribution to prediction sample distribution.

This script compares color distributions between training and prediction sets,
then corrects the training set to match the prediction set distribution using
sample reweighting or resampling.

Focus on 4 main color features:
- g_r_color (Pan-STARRS)
- r_i_color (Pan-STARRS)
- i_z_color (Pan-STARRS)
- bp_rp (Gaia)

Usage:
    python scripts/match_training_to_prediction_distribution.py --method reweight
    python scripts/match_training_to_prediction_distribution.py --method resample --visualize
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity
import logging
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from src.config import get_config

# Configuration
config = get_config()
MISSING_VALUE = config.get('processing', 'missing_value')

# Setup logging - will be configured in main() to include file handler
logger = logging.getLogger(__name__)

# Color features to match
COLOR_FEATURES = ['g_r_color', 'r_i_color', 'i_z_color', 'bp_rp']


def _score_samples_chunk(kde, X_chunk):
    """Helper function for parallel KDE scoring."""
    return kde.score_samples(X_chunk)


def score_samples_parallel(kde, X, n_jobs=-1, chunk_size=10000):
    """
    Score samples in parallel using multiprocessing.

    Args:
        kde: Fitted KernelDensity object
        X: Data to score
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        chunk_size: Size of chunks for parallel processing
    """
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Split data into chunks
    n_samples = len(X)
    chunks = [X[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]

    logger.info(f"  Processing {n_samples:,} samples in {len(chunks)} chunks using {n_jobs} CPUs...")

    # Score in parallel
    with Pool(processes=n_jobs) as pool:
        score_func = partial(_score_samples_chunk, kde)
        results = pool.map(score_func, chunks)

    # Concatenate results
    return np.concatenate(results)


def load_datasets():
    """Load training and prediction datasets."""
    logger.info("Loading datasets...")

    train_path = config.get_dataset_path('eb_unified_features_engineered_train', 'processed')
    predict_path = config.get_dataset_path('eb_unified_features_engineered_predict', 'processed')

    logger.info(f"Training path: {train_path}")
    logger.info(f"Prediction path: {predict_path}")

    start = time.time()
    train_df = pl.read_parquet(train_path)
    logger.info(f"Training set loaded in {time.time()-start:.2f}s: {len(train_df):,} objects")

    start = time.time()
    predict_df = pl.read_parquet(predict_path)
    logger.info(f"Prediction set loaded in {time.time()-start:.2f}s: {len(predict_df):,} objects")

    return train_df, predict_df


def compare_distributions(train_df, predict_df, features=COLOR_FEATURES):
    """
    Compare color distributions between training and prediction sets.

    Returns statistical tests and distribution parameters.
    """
    logger.info("="*70)
    logger.info("DISTRIBUTION COMPARISON")
    logger.info("="*70)

    results = []

    for feature in features:
        logger.info(f"\n{feature}:")
        logger.info("-" * 50)

        # Get valid values (not missing)
        train_vals = train_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

        # Basic statistics
        train_mean = np.mean(train_vals)
        train_std = np.std(train_vals)
        train_median = np.median(train_vals)

        predict_mean = np.mean(predict_vals)
        predict_std = np.std(predict_vals)
        predict_median = np.median(predict_vals)

        logger.info(f"  Training:   n={len(train_vals):,}, mean={train_mean:.3f}, std={train_std:.3f}, median={train_median:.3f}")
        logger.info(f"  Prediction: n={len(predict_vals):,}, mean={predict_mean:.3f}, std={predict_std:.3f}, median={predict_median:.3f}")
        logger.info(f"  Difference: mean={predict_mean - train_mean:+.3f}, std={predict_std - train_std:+.3f}")

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(train_vals, predict_vals)
        logger.info(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pval:.4e}")

        if ks_pval < 0.001:
            logger.info(f"  ⚠️  Distributions are SIGNIFICANTLY DIFFERENT (p < 0.001)")
        elif ks_pval < 0.05:
            logger.info(f"  ⚠️  Distributions are different (p < 0.05)")
        else:
            logger.info(f"  ✓ Distributions are similar (p >= 0.05)")

        results.append({
            'feature': feature,
            'train_n': len(train_vals),
            'predict_n': len(predict_vals),
            'train_mean': train_mean,
            'predict_mean': predict_mean,
            'train_std': train_std,
            'predict_std': predict_std,
            'mean_diff': predict_mean - train_mean,
            'std_diff': predict_std - train_std,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval
        })

    return pl.DataFrame(results)


def visualize_distributions(train_df, predict_df, features=COLOR_FEATURES, output_dir=None):
    """
    Create visualization comparing distributions.
    """
    if output_dir is None:
        output_dir = Path(config.get_path('figures')) / 'distribution_matching'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n" + "="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")

    # 1. Individual histograms for each feature
    for feature in features:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        train_vals = train_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

        # Histogram comparison
        bins = np.linspace(min(train_vals.min(), predict_vals.min()),
                          max(train_vals.max(), predict_vals.max()), 50)

        ax1.hist(train_vals, bins=bins, alpha=0.6, label='Training', density=True, color='blue', edgecolor='black')
        ax1.hist(predict_vals, bins=bins, alpha=0.6, label='Prediction', density=True, color='red', edgecolor='black')
        ax1.set_xlabel(f'{feature}', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'{feature} - Histogram Comparison', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # KDE comparison
        kde_train = stats.gaussian_kde(train_vals)
        kde_predict = stats.gaussian_kde(predict_vals)

        x_range = np.linspace(bins[0], bins[-1], 200)
        ax2.plot(x_range, kde_train(x_range), label='Training', linewidth=2, color='blue')
        ax2.plot(x_range, kde_predict(x_range), label='Prediction', linewidth=2, color='red')
        ax2.fill_between(x_range, kde_train(x_range), alpha=0.3, color='blue')
        ax2.fill_between(x_range, kde_predict(x_range), alpha=0.3, color='red')
        ax2.set_xlabel(f'{feature}', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'{feature} - KDE Comparison', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        output_file = output_dir / f'{feature}_distribution_comparison.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  ✓ Saved: {output_file.name}")

    # 2. Combined 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        train_vals = train_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()

        bins = np.linspace(min(train_vals.min(), predict_vals.min()),
                          max(train_vals.max(), predict_vals.max()), 40)

        ax.hist(train_vals, bins=bins, alpha=0.6, label='Training', density=True, color='blue', edgecolor='black')
        ax.hist(predict_vals, bins=bins, alpha=0.6, label='Prediction', density=True, color='red', edgecolor='black')
        ax.set_xlabel(f'{feature}', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Color Distribution Comparison: Training vs Prediction', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    output_file = output_dir / 'all_colors_comparison.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved: {output_file.name}")


def calculate_density_ratio_weights(train_df, predict_df, features=COLOR_FEATURES, use_sampling=True, sample_size=50000):
    """
    Calculate importance weights for training samples using density ratio estimation.

    Weight = P(x in prediction set) / P(x in training set)

    This reweights training samples to match the prediction distribution.

    Args:
        use_sampling: If True, use subset for KDE fitting (faster)
        sample_size: Number of samples to use for KDE fitting
    """
    logger.info("\n" + "="*70)
    logger.info("CALCULATING DENSITY RATIO WEIGHTS")
    logger.info("="*70)

    # Extract feature arrays
    train_features = []
    predict_features = []

    for feature in features:
        train_vals = train_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        train_features.append(train_vals)
        predict_features.append(predict_vals)

    train_X = np.column_stack(train_features)
    predict_X = np.column_stack(predict_features)

    logger.info(f"Training features shape: {train_X.shape}")
    logger.info(f"Prediction features shape: {predict_X.shape}")

    # Optionally subsample for KDE fitting (much faster)
    if use_sampling and len(train_X) > sample_size:
        logger.info(f"\nSubsampling {sample_size:,} samples for KDE fitting (speed optimization)...")
        train_indices = np.random.choice(len(train_X), size=sample_size, replace=False)
        predict_indices = np.random.choice(len(predict_X), size=min(sample_size, len(predict_X)), replace=False)
        train_X_sample = train_X[train_indices]
        predict_X_sample = predict_X[predict_indices]
    else:
        train_X_sample = train_X
        predict_X_sample = predict_X

    # Fit KDE on both distributions
    logger.info(f"\nFitting KDE on training set ({len(train_X_sample):,} samples)...")
    start = time.time()
    kde_train = KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde_train.fit(train_X_sample)
    logger.info(f"  Completed in {time.time()-start:.2f}s")

    logger.info(f"Fitting KDE on prediction set ({len(predict_X_sample):,} samples)...")
    start = time.time()
    kde_predict = KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde_predict.fit(predict_X_sample)
    logger.info(f"  Completed in {time.time()-start:.2f}s")

    # Calculate density ratio for training samples using parallel processing
    logger.info(f"\nCalculating density ratios for {len(train_X):,} training samples...")
    logger.info("Using parallel processing for KDE scoring...")

    start = time.time()
    logger.info("Scoring training density...")
    log_dens_train = score_samples_parallel(kde_train, train_X, n_jobs=-1, chunk_size=10000)
    logger.info(f"  Training density calculated in {time.time()-start:.2f}s")

    start = time.time()
    logger.info("Scoring prediction density...")
    log_dens_predict = score_samples_parallel(kde_predict, train_X, n_jobs=-1, chunk_size=10000)
    logger.info(f"  Prediction density calculated in {time.time()-start:.2f}s")

    # Weight = P_predict / P_train
    weights = np.exp(log_dens_predict - log_dens_train)

    # Clip extreme weights to prevent instability
    weights = np.clip(weights, 0.01, 100)

    # Normalize weights to sum to number of samples
    weights = weights * len(weights) / weights.sum()

    logger.info(f"\nWeight statistics:")
    logger.info(f"  Min weight: {weights.min():.3f}")
    logger.info(f"  Max weight: {weights.max():.3f}")
    logger.info(f"  Mean weight: {weights.mean():.3f}")
    logger.info(f"  Median weight: {np.median(weights):.3f}")
    logger.info(f"  Std weight: {weights.std():.3f}")

    # Visualize weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(weights, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal weight')
    ax1.set_xlabel('Weight', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Sample Weights', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(np.log10(weights), bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal weight (log scale)')
    ax2.set_xlabel('Log10(Weight)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Log Sample Weights', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    output_dir = Path(config.get_path('figures')) / 'distribution_matching'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'sample_weights_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"\n✓ Saved weight distribution plot")

    return weights


def create_weighted_training_set(train_df, weights):
    """
    Create a new training set with weights column.
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING WEIGHTED TRAINING SET")
    logger.info("="*70)

    # Add weights column
    train_weighted = train_df.with_columns([
        pl.Series('sample_weight', weights)
    ])

    # Save weighted training set
    output_path = Path(config.get_path('processed')) / 'eb_unified_features_engineered_train_weighted.parquet'
    train_weighted.write_parquet(output_path)

    logger.info(f"✓ Saved weighted training set: {output_path}")
    logger.info(f"  Total samples: {len(train_weighted):,}")
    logger.info(f"  Effective sample size: {(weights.sum()**2 / (weights**2).sum()):.0f}")

    return train_weighted


def create_resampled_training_set(train_df, predict_df, features=COLOR_FEATURES, n_samples=None):
    """
    Create a resampled training set that matches prediction distribution.

    Uses stratified sampling based on color bins.
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING RESAMPLED TRAINING SET")
    logger.info("="*70)

    if n_samples is None:
        n_samples = len(train_df)

    # For simplicity, we'll use the weights to sample with replacement
    weights = calculate_density_ratio_weights(train_df, predict_df, features)

    # Normalize to probabilities
    probs = weights / weights.sum()

    # Sample indices
    logger.info(f"\nResampling {n_samples:,} samples from training set...")
    indices = np.random.choice(len(train_df), size=n_samples, replace=True, p=probs)

    # Create resampled dataframe
    train_resampled = train_df[indices]

    # Save
    output_path = Path(config.get_path('processed')) / 'eb_unified_features_engineered_train_resampled.parquet'
    train_resampled.write_parquet(output_path)

    logger.info(f"✓ Saved resampled training set: {output_path}")
    logger.info(f"  Total samples: {len(train_resampled):,}")
    logger.info(f"  Unique samples: {len(set(indices)):,}")

    return train_resampled


def validate_matching(train_corrected, predict_df, features=COLOR_FEATURES, method='reweighted'):
    """
    Validate that the corrected training set matches prediction distribution.
    """
    logger.info("\n" + "="*70)
    logger.info(f"VALIDATING {method.upper()} DISTRIBUTION MATCH")
    logger.info("="*70)

    for feature in features:
        logger.info(f"\n{feature}:")
        logger.info("-" * 50)

        if 'sample_weight' in train_corrected.columns:
            # Weighted statistics
            train_vals = train_corrected.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
            weights = train_corrected.filter(pl.col(feature) != MISSING_VALUE)['sample_weight'].to_numpy()

            train_mean = np.average(train_vals, weights=weights)
            train_std = np.sqrt(np.average((train_vals - train_mean)**2, weights=weights))
        else:
            # Unweighted (resampled)
            train_vals = train_corrected.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
            train_mean = np.mean(train_vals)
            train_std = np.std(train_vals)

        predict_vals = predict_df.filter(pl.col(feature) != MISSING_VALUE)[feature].to_numpy()
        predict_mean = np.mean(predict_vals)
        predict_std = np.std(predict_vals)

        logger.info(f"  Corrected train: mean={train_mean:.3f}, std={train_std:.3f}")
        logger.info(f"  Prediction:      mean={predict_mean:.3f}, std={predict_std:.3f}")
        logger.info(f"  Difference:      mean={predict_mean - train_mean:+.3f}, std={predict_std - train_std:+.3f}")

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(train_vals, predict_vals)
        logger.info(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pval:.4e}")

        if ks_pval >= 0.05:
            logger.info(f"  ✓ Distributions match well (p >= 0.05)")
        else:
            logger.info(f"  ⚠️  Some difference remains (p < 0.05)")


def main():
    parser = argparse.ArgumentParser(description='Match training distribution to prediction distribution')
    parser.add_argument('--method', choices=['reweight', 'resample', 'both'], default='reweight',
                       help='Method to use: reweight (add weights) or resample (bootstrap)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    parser.add_argument('--features', nargs='+', default=COLOR_FEATURES,
                       help='Features to match (default: g_r_color r_i_color i_z_color bp_rp)')

    args = parser.parse_args()

    # Setup logging with file handler
    log_dir = Path(config.get_path('reports')) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'distribution_matching.log'

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Write mode - overwrite each run
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger.info(f"Logging to: {log_file}")
    logger.info("="*70)

    # Set features
    features = args.features
    logger.info(f"Matching features: {features}")

    # Load datasets
    train_df, predict_df = load_datasets()

    # Compare distributions
    comparison = compare_distributions(train_df, predict_df, features)

    # Visualize
    if args.visualize:
        visualize_distributions(train_df, predict_df, features)

    # Apply correction method
    if args.method in ['reweight', 'both']:
        weights = calculate_density_ratio_weights(train_df, predict_df, features)
        train_weighted = create_weighted_training_set(train_df, weights)
        validate_matching(train_weighted, predict_df, features, method='reweighted')

    if args.method in ['resample', 'both']:
        train_resampled = create_resampled_training_set(train_df, predict_df, features)
        validate_matching(train_resampled, predict_df, features, method='resampled')

    logger.info("\n" + "="*70)
    logger.info("DISTRIBUTION MATCHING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
