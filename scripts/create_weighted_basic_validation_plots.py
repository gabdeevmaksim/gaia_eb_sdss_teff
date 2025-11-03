#!/usr/bin/env python3
"""
Create standardized validation plots for Weighted Basic Colors model.

Uses shared visualization module for consistent style across all models.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. temp_distributions - Training vs Test comparison (histogram + CDF)
5. color_distributions - Compare color features between train/test sets
6. color_temp_relations - Color-Temperature diagrams (bp-rp, most important)
7. feature_importance - Feature importance bar chart
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.visualization.validation_plots import (
    plot_test_scatter,
    plot_residuals,
    plot_performance_by_temp,
    plot_temp_distributions,
    plot_color_distributions,
    plot_color_temp_relations,
    plot_feature_importance,
    calculate_bin_statistics,
    print_distribution_statistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Generate all validation plots."""

    config = get_config()
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("CREATE WEIGHTED BASIC COLORS VALIDATION PLOTS")
    logger.info("="*80)

    # Find most recent weighted basic colors model
    model_files = sorted(models_dir.glob('rf_weighted_basic_colors_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No weighted basic colors model found!")

    model_path = model_files[-1]
    MODEL_ID = model_path.stem
    MODEL_NAME = 'Weighted Basic Colors Model (Distribution-Matched)'
    SUBDIR = 'weighted_basic_validation'

    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load metadata
    logger.info("\nLoading data...")
    with open(models_dir / f'{MODEL_ID}_metadata.json', 'r') as f:
        metadata = json.load(f)
    logger.info("   Metadata loaded")

    # Load model to get test predictions
    with open(models_dir / f'{MODEL_ID}.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load weighted training data
    train_path = Path(config.get_path('processed')) / 'eb_unified_features_engineered_train_weighted.parquet'
    import polars as pl
    df_full = pl.read_parquet(train_path)

    # Filter out missing values (same as training)
    MISSING_VALUE = config.get('processing', 'missing_value')
    COLOR_FEATURES = metadata['features']

    mask = pl.lit(True)
    for feature in COLOR_FEATURES:
        mask = mask & (pl.col(feature) != MISSING_VALUE)
    mask = mask & (pl.col('teff_gspphot') != MISSING_VALUE)

    df_clean = df_full.filter(mask)

    # Prepare features
    X = df_clean.select(COLOR_FEATURES).to_numpy()
    y = df_clean['teff_gspphot'].to_numpy()
    weights = df_clean['sample_weight'].to_numpy()

    # Train/test split (same as training)
    from sklearn.model_selection import train_test_split
    TEST_SIZE = config.get('ml', 'test_size')
    RANDOM_STATE = config.get('ml', 'random_state')

    X_train, X_test, y_train, y_test, train_weights, test_weights, train_idx, test_idx = train_test_split(
        X, y, weights, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"   Test samples: {len(X_test):,}")

    # Generate predictions
    y_test_pred = model.predict(X_test)

    # Create test predictions DataFrame
    test_pred = pd.DataFrame({
        'true_temperature': y_test,
        'predicted_temperature': y_test_pred,
        'residual': y_test - y_test_pred,
        'sample_weight': test_weights,
        'teff_true': y_test,  # Alias for compatibility
        'teff_pred': y_test_pred  # Alias for compatibility
    })

    # Calculate weighted metrics
    mae = np.average(np.abs(test_pred['residual']), weights=test_weights)
    rmse = np.sqrt(np.average(test_pred['residual']**2, weights=test_weights))
    ss_res = np.sum(test_weights * test_pred['residual']**2)
    ss_tot = np.sum(test_weights * (y_test - np.average(y_test, weights=test_weights))**2)
    r2 = 1 - (ss_res / ss_tot)

    logger.info(f"   MAE: {mae:.1f} K, RMSE: {rmse:.1f} K, R²: {r2:.3f}")

    # Prepare training and test dataframes
    df_train_subset = df_clean[train_idx].to_pandas()
    df_test_subset = df_clean[test_idx].to_pandas()

    # Rename columns for compatibility
    df_train_subset['teff_gspphot'] = y_train
    df_test_subset['teff_predicted'] = y_test_pred

    logger.info(f"   Training data: {len(df_train_subset):,} objects")

    # Calculate performance by temperature range
    logger.info("\nCalculating performance by temperature range...")
    bin_stats_df = calculate_bin_statistics(test_pred)

    # Generate plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING VALIDATION PLOTS")
    logger.info("="*80)

    # 1. Test scatter
    plot_test_scatter(test_pred, mae, rmse, r2, MODEL_ID, SUBDIR, MODEL_NAME)

    # 2. Residuals
    plot_residuals(test_pred, MODEL_ID, SUBDIR)

    # 3. Performance by temperature
    plot_performance_by_temp(test_pred, bin_stats_df, MODEL_ID, SUBDIR)

    # 4. Temperature distributions (training vs test)
    plot_temp_distributions(df_train_subset, df_test_subset, MODEL_ID, SUBDIR,
                           train_col='teff_gspphot', pred_col='teff_predicted',
                           model_name=MODEL_NAME)

    # 5. Color distributions
    colors_to_compare = COLOR_FEATURES
    color_labels = ['g-r', 'r-i', 'i-z', 'BP-RP']

    plot_color_distributions(df_train_subset, df_test_subset, colors_to_compare, color_labels,
                            MODEL_ID, SUBDIR)

    # 6. Color-temperature relations (use bp_rp, most important feature)
    plot_color_temp_relations(df_train_subset, df_test_subset, 'bp_rp', 'BP-RP',
                             MODEL_ID, SUBDIR,
                             train_temp_col='teff_gspphot',
                             pred_temp_col='teff_predicted')

    # 7. Feature importance
    plot_feature_importance(metadata['feature_importance'], MODEL_ID, SUBDIR,
                           MODEL_NAME, top_n=4)

    # Statistical comparison
    print_distribution_statistics(df_train_subset, df_test_subset,
                                 train_col='teff_gspphot',
                                 pred_col='teff_predicted')

    logger.info("\n" + "="*80)
    logger.info("VALIDATION PLOTS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: reports/figures/{SUBDIR}/")
    logger.info(f"Total plots generated: 7 figures")


if __name__ == '__main__':
    main()
