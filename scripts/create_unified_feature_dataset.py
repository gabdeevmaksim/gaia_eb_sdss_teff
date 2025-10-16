#!/usr/bin/env python3
"""
Create unified feature dataset for training and prediction.

This script creates a single dataset with engineered features for ALL objects,
ensuring perfect consistency between training and prediction data.

Key principles:
1. Load ALL objects (with and without ground truth) into single dataset
2. Apply feature engineering to ENTIRE dataset simultaneously
3. Split AFTER feature engineering (train/predict)
4. Compare feature distributions between splits
5. Generate comprehensive validation report

Usage:
    # Create unified feature dataset
    python scripts/create_unified_feature_dataset.py

    # Create with specific model type
    python scripts/create_unified_feature_dataset.py --model-type engineered
    python scripts/create_unified_feature_dataset.py --model-type basic

Author: Claude Code
Date: 2025-10-15
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config
from src.features.engineering import engineer_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def load_all_data(config) -> pd.DataFrame:
    """
    Load all objects with Pan-STARRS photometry and Gaia data.

    Returns
    -------
    pd.DataFrame
        Complete dataset with all required input features
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading complete dataset...")

    # Load the prepared dataset (has both objects with and without teff_gspphot)
    input_file = config.get_path('processed') / 'eb_panstarrs_for_prediction.parquet'

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run scripts/prepare_data_for_basic_prediction.py first")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)}")

    # Check for required input features
    required_features = ['g_r_color', 'r_i_color', 'i_z_color', 'gPSFMag']
    missing = [f for f in required_features if f not in df.columns]

    if missing:
        logger.error(f"Missing required features: {missing}")
        raise ValueError(f"Dataset missing required features: {missing}")

    # Add B-V if not present (should be there from prepare script)
    if 'B_V_color' not in df.columns:
        logger.info("  Adding B-V_color (synthetic from g-r)")
        df['B_V_color'] = 0.98 * df['g_r_color'] + 0.22

    # Check Gaia colors (bp_rp)
    if 'bp_rp' not in df.columns:
        logger.warning("  bp_rp not found - loading from Gaia catalog")
        # Load Gaia catalog to get bp_rp
        gaia_file = config.get_path('processed') / 'eb_catalog.parquet'
        df_gaia = pd.read_parquet(gaia_file, columns=['original_ext_source_id', 'bp_rp'])

        df = df.merge(df_gaia, on='original_ext_source_id', how='left')
        logger.info(f"  Added bp_rp: {df['bp_rp'].notna().sum():,} objects have values")

    # Fill missing bp_rp with 0 (will be handled in cleaning step)
    df['bp_rp'] = df['bp_rp'].fillna(0)
    logger.info(f"  Filled {(df['bp_rp'] == 0).sum():,} missing bp_rp values with 0")

    # Analyze data coverage
    logger.info("\nData coverage:")
    logger.info(f"  Total objects: {len(df):,}")
    logger.info(f"  With teff_gspphot: {df['teff_gspphot'].notna().sum():,} ({100*df['teff_gspphot'].notna().sum()/len(df):.1f}%)")
    logger.info(f"  Without teff_gspphot: {df['teff_gspphot'].isna().sum():,} ({100*df['teff_gspphot'].isna().sum()/len(df):.1f}%)")

    # Feature availability
    logger.info("\nInput feature availability (before cleaning):")
    for feat in ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp', 'gPSFMag']:
        if feat in df.columns:
            valid = df[feat].notna().sum()
            logger.info(f"  {feat}: {valid:,} ({100*valid/len(df):.1f}%)")

    # Clean outliers in base features BEFORE feature engineering
    logger.info("\nCleaning outliers in base features...")

    # Define realistic ranges for colors (based on stellar physics)
    color_limits = {
        'g_r_color': (-0.5, 3.0),    # Typical stellar colors
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'B_V_color': (-0.5, 3.0),
        'bp_rp': (-0.5, 4.0),
        'gPSFMag': (10.0, 25.0)      # Reasonable magnitude range
    }

    total_filtered = 0
    for feat, (min_val, max_val) in color_limits.items():
        if feat in df.columns:
            before = len(df)
            # Keep only values within reasonable range
            mask = (df[feat] >= min_val) & (df[feat] <= max_val)
            n_outliers = (~mask & df[feat].notna()).sum()
            df.loc[~mask, feat] = np.nan
            total_filtered += n_outliers
            if n_outliers > 0:
                logger.info(f"  {feat}: removed {n_outliers:,} outliers outside [{min_val}, {max_val}]")

    logger.info(f"  Total outliers removed: {total_filtered:,}")

    # After cleaning, remove rows with too many missing values
    logger.info("\nRemoving objects with insufficient data...")

    # Require all essential colors including bp_rp (since it's 60% of model importance)
    required_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'bp_rp']
    before = len(df)

    # Remove objects with missing required colors
    for col in required_cols:
        n_missing = df[col].isna().sum()
        n_zero = (df[col] == 0).sum()
        if n_missing > 0 or n_zero > 0:
            logger.info(f"  {col}: {n_missing:,} NaN, {n_zero:,} zeros")

    df = df.dropna(subset=required_cols, how='any')
    # Also remove zeros in bp_rp (these are actually missing values)
    df = df[df['bp_rp'] != 0]

    removed = before - len(df)
    logger.info(f"  Removed {removed:,} objects with missing required colors")
    logger.info(f"  Remaining: {len(df):,} objects")

    # Recalculate B_V_color after cleaning (since it depends on g_r_color)
    df['B_V_color'] = 0.98 * df['g_r_color'] + 0.22

    # Fill any remaining NaN values in optional features (should be none now)
    for col in ['B_V_color', 'gPSFMag']:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                logger.info(f"  Filled {n_missing:,} missing values in {col} with 0")

    return df


def engineer_features_unified(
    df: pd.DataFrame,
    model_type: str = 'engineered',
    missing_value: float = -999.0
) -> pd.DataFrame:
    """
    Apply feature engineering to entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with base features
    model_type : str
        'basic' or 'engineered'
    missing_value : float
        Value representing missing data

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\nEngineering features (model_type: {model_type})...")

    # Define feature engineering parameters based on model type
    if model_type == 'basic':
        # Basic model: 5 simple features, no complex engineering
        logger.info("  Using BASIC feature engineering:")
        logger.info("    - 5 base features only")
        logger.info("    - g_r_color, r_i_color, i_z_color, B_V_color, gPSFMag")
        logger.info("    - No polynomial/interaction/log features")

        # For basic model, just return the base features
        feature_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag']
        df_features = df.copy()

    elif model_type == 'engineered':
        # Engineered model: complex feature engineering
        logger.info("  Using ENGINEERED feature engineering:")
        logger.info("    - Polynomial features (degree 3)")
        logger.info("    - Interaction features")
        logger.info("    - Log features")
        logger.info("    - Temperature-dependent features")
        logger.info("    - NO magnitude features (avoiding bias)")

        # Define color columns (exclude IDs and target)
        exclude_cols = ['original_ext_source_id', 'gaia_source_id', 'teff_gspphot']
        magnitude_cols = ['gPSFMag', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']

        # Get color columns (EXCLUDING magnitude features)
        color_cols = [col for col in df.columns
                     if col not in exclude_cols
                     and col not in magnitude_cols
                     and ('color' in col.lower() or col in ['bp_rp', 'g_r', 'r_i', 'i_z'])]

        logger.info(f"    - Color features: {color_cols}")
        logger.info(f"    - EXCLUDING all magnitude features (gPSFMag, etc.)")

        # Apply feature engineering
        df_features = engineer_all_features(
            df,
            color_cols=color_cols,
            mag_cols=None,  # No magnitude features
            include_polynomials=True,
            include_interactions=True,
            include_log=True,
            include_temp_dependent=True,
            include_mag_features=False
        )

        # Explicitly drop magnitude columns if they exist
        mag_drop = [col for col in df_features.columns if 'Mag' in col or 'mag' in col.lower()]
        if mag_drop:
            logger.info(f"    - Dropping magnitude columns: {mag_drop}")
            df_features = df_features.drop(columns=mag_drop)

        logger.info(f"  Total features created: {df_features.shape[1]}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'basic' or 'engineered'")

    # Clean features: handle missing values, inf, nan
    logger.info("\nCleaning features...")

    # Get feature columns (exclude metadata)
    exclude_cols = ['original_ext_source_id', 'gaia_source_id', 'teff_gspphot',
                   'Te_gr', 'Te_ri', 'Te_iz']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    # Replace missing value marker with NaN for cleaning
    for col in feature_cols:
        if col in df_features.columns:
            df_features.loc[df_features[col] == missing_value, col] = np.nan

    # Replace inf with NaN
    df_features[feature_cols] = df_features[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with 0 (standard practice for engineered features)
    df_features[feature_cols] = df_features[feature_cols].fillna(0)

    # Report cleaning results
    logger.info(f"  Cleaned {len(feature_cols)} feature columns")
    logger.info(f"  Feature matrix shape: {df_features[feature_cols].shape}")

    return df_features


def compare_feature_distributions(
    df_train: pd.DataFrame,
    df_predict: pd.DataFrame,
    feature_cols: list,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compare feature distributions between training and prediction sets.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset
    df_predict : pd.DataFrame
        Prediction dataset
    feature_cols : list
        Feature columns to compare
    output_dir : Path
        Output directory for plots

    Returns
    -------
    pd.DataFrame
        Comparison statistics for each feature
    """
    logger = logging.getLogger(__name__)

    logger.info("\nComparing feature distributions...")
    logger.info(f"  Training set: {len(df_train):,} objects")
    logger.info(f"  Prediction set: {len(df_predict):,} objects")
    logger.info(f"  Features to compare: {len(feature_cols)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect statistics for each feature
    comparison_stats = []

    for feat in feature_cols:
        # Get data and filter out zeros (represent missing values after cleaning)
        train_data = df_train[feat].dropna()
        pred_data = df_predict[feat].dropna()

        # Filter out zeros
        train_data = train_data[train_data != 0]
        pred_data = pred_data[pred_data != 0]

        if len(train_data) == 0 or len(pred_data) == 0:
            logger.warning(f"  Skipping {feat}: insufficient data")
            continue

        # Calculate statistics
        stats_dict = {
            'feature': feat,
            'train_count': len(train_data),
            'train_mean': train_data.mean(),
            'train_std': train_data.std(),
            'train_median': train_data.median(),
            'train_min': train_data.min(),
            'train_max': train_data.max(),
            'pred_count': len(pred_data),
            'pred_mean': pred_data.mean(),
            'pred_std': pred_data.std(),
            'pred_median': pred_data.median(),
            'pred_min': pred_data.min(),
            'pred_max': pred_data.max(),
        }

        # Statistical tests
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(train_data, pred_data)
        stats_dict['ks_statistic'] = ks_stat
        stats_dict['ks_pvalue'] = ks_pval

        # T-test for means
        t_stat, t_pval = stats.ttest_ind(train_data, pred_data, equal_var=False)
        stats_dict['t_statistic'] = t_stat
        stats_dict['t_pvalue'] = t_pval

        # Difference in means (as percentage of training mean)
        mean_diff_pct = 100 * (pred_data.mean() - train_data.mean()) / train_data.mean()
        stats_dict['mean_diff_percent'] = mean_diff_pct

        comparison_stats.append(stats_dict)

    # Create comparison dataframe
    df_comparison = pd.DataFrame(comparison_stats)

    # Report summary
    logger.info("\nDistribution comparison summary:")
    logger.info(f"  Features compared: {len(df_comparison)}")

    # Count features with significant differences
    sig_ks = (df_comparison['ks_pvalue'] < 0.05).sum()
    sig_t = (df_comparison['t_pvalue'] < 0.05).sum()

    logger.info(f"  Significant KS test (p<0.05): {sig_ks} ({100*sig_ks/len(df_comparison):.1f}%)")
    logger.info(f"  Significant t-test (p<0.05): {sig_t} ({100*sig_t/len(df_comparison):.1f}%)")

    # Features with large mean differences
    large_diff = (df_comparison['mean_diff_percent'].abs() > 10).sum()
    logger.info(f"  Mean difference >10%: {large_diff} ({100*large_diff/len(df_comparison):.1f}%)")

    # Save comparison table
    comparison_file = output_dir / 'feature_comparison_statistics.csv'
    df_comparison.to_csv(comparison_file, index=False)
    logger.info(f"\n  Saved comparison statistics: {comparison_file}")

    return df_comparison


def create_distribution_plots(
    df_train: pd.DataFrame,
    df_predict: pd.DataFrame,
    feature_cols: list,
    output_dir: Path,
    n_features_plot: int = 20
):
    """
    Create distribution comparison plots.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset
    df_predict : pd.DataFrame
        Prediction dataset
    feature_cols : list
        Features to plot
    output_dir : Path
        Output directory
    n_features_plot : int
        Number of features to plot in detail
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\nCreating distribution plots (top {n_features_plot} features)...")

    # Select features to plot (first n_features_plot)
    features_to_plot = feature_cols[:n_features_plot]

    # Create grid of plots
    n_cols = 4
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]

        # Get data and filter out zeros (represent missing values after cleaning)
        train_data = df_train[feat].dropna()
        pred_data = df_predict[feat].dropna()

        # Filter out zeros and extreme outliers (likely missing value artifacts)
        train_data = train_data[train_data != 0]
        pred_data = pred_data[pred_data != 0]

        # Remove extreme outliers (beyond 5 std devs)
        if len(train_data) > 0 and len(pred_data) > 0:
            combined = pd.concat([train_data, pred_data])
            mean_val = combined.mean()
            std_val = combined.std()
            if std_val > 0:
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                train_data = train_data[(train_data >= lower_bound) & (train_data <= upper_bound)]
                pred_data = pred_data[(pred_data >= lower_bound) & (pred_data <= upper_bound)]

        # Plot histograms
        if len(train_data) > 0 and len(pred_data) > 0:
            ax.hist(train_data, bins=50, alpha=0.5, label='Train', density=True, color='blue')
            ax.hist(pred_data, bins=50, alpha=0.5, label='Predict', density=True, color='orange')

            # Set reasonable x-limits based on percentiles
            combined = pd.concat([train_data, pred_data])
            x_min = combined.quantile(0.001)
            x_max = combined.quantile(0.999)
            ax.set_xlim(x_min, x_max)

        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_title(f'{feat}', fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plot_file = output_dir / 'feature_distributions_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved distribution plots: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create unified feature dataset for training and prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create with engineered features (default)
  python scripts/create_unified_feature_dataset.py

  # Create with basic features only
  python scripts/create_unified_feature_dataset.py --model-type basic
        """
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['basic', 'engineered'],
        default='engineered',
        help='Type of features to create: basic (5 features) or engineered (38+ features)'
    )

    parser.add_argument(
        '--missing-value',
        type=float,
        default=-999.0,
        help='Value representing missing data (default: -999.0)'
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    config = get_config()

    logger.info("=" * 80)
    logger.info("CREATE UNIFIED FEATURE DATASET")
    logger.info("=" * 80)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Missing value marker: {args.missing_value}")
    logger.info("")

    # Step 1: Load all data
    df_all = load_all_data(config)

    # Step 2: Engineer features on entire dataset
    df_features = engineer_features_unified(
        df_all,
        model_type=args.model_type,
        missing_value=args.missing_value
    )

    # Step 3: Split into train and predict sets
    logger.info("\nSplitting into train/predict sets...")
    df_train = df_features[df_features['teff_gspphot'].notna()].copy()
    df_predict = df_features[df_features['teff_gspphot'].isna()].copy()

    logger.info(f"  Training set: {len(df_train):,} objects")
    logger.info(f"  Prediction set: {len(df_predict):,} objects")

    # Get feature columns for comparison
    exclude_cols = ['original_ext_source_id', 'gaia_source_id', 'teff_gspphot',
                   'Te_gr', 'Te_ri', 'Te_iz']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    logger.info(f"  Feature columns: {len(feature_cols)}")

    # Step 4: Compare distributions
    output_dir = config.get_path('reports') / 'figures' / 'feature_validation'
    df_comparison = compare_feature_distributions(
        df_train, df_predict, feature_cols, output_dir
    )

    # Step 5: Create visualization
    create_distribution_plots(
        df_train, df_predict, feature_cols, output_dir
    )

    # Step 6: Save datasets
    logger.info("\nSaving unified datasets...")
    processed_dir = config.get_path('processed')

    # Save complete dataset
    output_all = processed_dir / f'eb_unified_features_{args.model_type}.parquet'
    df_features.to_parquet(output_all, index=False)
    file_size_mb = output_all.stat().st_size / 1024 / 1024
    logger.info(f"  Complete dataset: {output_all}")
    logger.info(f"    Objects: {len(df_features):,}")
    logger.info(f"    Features: {len(feature_cols)}")
    logger.info(f"    Size: {file_size_mb:.2f} MB")

    # Save training set
    output_train = processed_dir / f'eb_unified_features_{args.model_type}_train.parquet'
    df_train.to_parquet(output_train, index=False)
    logger.info(f"  Training set: {output_train}")
    logger.info(f"    Objects: {len(df_train):,}")

    # Save prediction set
    output_predict = processed_dir / f'eb_unified_features_{args.model_type}_predict.parquet'
    df_predict.to_parquet(output_predict, index=False)
    logger.info(f"  Prediction set: {output_predict}")
    logger.info(f"    Objects: {len(df_predict):,}")

    # Step 7: Create summary report
    summary_file = processed_dir / f'eb_unified_features_{args.model_type}_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("UNIFIED FEATURE DATASET CREATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"DATASET SIZES:\n")
        f.write(f"  Total objects: {len(df_features):,}\n")
        f.write(f"  Training objects (with teff_gspphot): {len(df_train):,} ({100*len(df_train)/len(df_features):.1f}%)\n")
        f.write(f"  Prediction objects (without teff_gspphot): {len(df_predict):,} ({100*len(df_predict)/len(df_features):.1f}%)\n\n")

        f.write(f"FEATURES:\n")
        f.write(f"  Total features: {len(feature_cols)}\n")
        f.write(f"  Feature list: {feature_cols[:20]}...\n\n")

        f.write(f"DISTRIBUTION COMPARISON:\n")
        f.write(f"  Features compared: {len(df_comparison)}\n")
        sig_ks = (df_comparison['ks_pvalue'] < 0.05).sum()
        sig_t = (df_comparison['t_pvalue'] < 0.05).sum()
        f.write(f"  Significant KS test (p<0.05): {sig_ks} ({100*sig_ks/len(df_comparison):.1f}%)\n")
        f.write(f"  Significant t-test (p<0.05): {sig_t} ({100*sig_t/len(df_comparison):.1f}%)\n")
        large_diff = (df_comparison['mean_diff_percent'].abs() > 10).sum()
        f.write(f"  Mean difference >10%: {large_diff} ({100*large_diff/len(df_comparison):.1f}%)\n\n")

        f.write(f"OUTPUT FILES:\n")
        f.write(f"  Complete dataset: {output_all.name}\n")
        f.write(f"  Training set: {output_train.name}\n")
        f.write(f"  Prediction set: {output_predict.name}\n")
        f.write(f"  Comparison stats: {output_dir / 'feature_comparison_statistics.csv'}\n")
        f.write(f"  Distribution plots: {output_dir / 'feature_distributions_comparison.png'}\n")

    logger.info(f"  Summary: {summary_file}")

    logger.info("\n" + "=" * 80)
    logger.info("UNIFIED FEATURE DATASET CREATION COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review feature distribution comparison: {output_dir / 'feature_comparison_statistics.csv'}")
    logger.info(f"  2. Check distribution plots: {output_dir / 'feature_distributions_comparison.png'}")
    logger.info(f"  3. Train model: python scripts/train_model_unified_features.py --model-type {args.model_type}")


if __name__ == '__main__':
    main()
