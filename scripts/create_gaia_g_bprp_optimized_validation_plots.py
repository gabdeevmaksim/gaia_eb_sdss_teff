#!/usr/bin/env python3
"""
Create standardized validation plots for Gaia G + BP-RP Optimized model.

Uses shared visualization module for consistent style across all models.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. temp_distributions - Training vs Test comparison (histogram + CDF)
5. color_distributions - Compare color features between train/test sets
6. color_temp_relations - Color-Temperature diagrams (bp-rp)
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

    models_dir = Path('models')
    data_dir = Path('data/processed')

    logger.info("="*80)
    logger.info("CREATE GAIA G + BP-RP OPTIMIZED VALIDATION PLOTS")
    logger.info("="*80)

    # Use the optimized model
    MODEL_ID = 'rf_gaia_g_bprp_optimized_20251022_091954'
    MODEL_NAME = 'Gaia G + BP-RP Optimized Model'
    SUBDIR = 'gaia_g_bprp_optimized_validation'

    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load metadata
    logger.info("\nLoading data...")
    metadata_path = models_dir / f'{MODEL_ID}_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    logger.info("   Metadata loaded")

    # Load model
    model_path = models_dir / f'{MODEL_ID}.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("   Model loaded")

    # Load training data
    import polars as pl
    train_data_path = data_dir / 'gaia_g_bprp_train.parquet'
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    df_full = pl.read_parquet(train_data_path)
    logger.info(f"   Loaded {len(df_full):,} training samples")

    # Prepare features
    COLOR_FEATURES = metadata['features']
    X = df_full.select(COLOR_FEATURES).to_numpy()
    y = df_full['teff_gspphot'].to_numpy()

    # Train/test split (same as training)
    from sklearn.model_selection import train_test_split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"   Test samples: {len(X_test):,}")

    # Generate predictions
    logger.info("\nGenerating predictions...")
    y_test_pred = model.predict(X_test)

    # Create test predictions DataFrame
    test_pred = pd.DataFrame({
        'true_temperature': y_test,
        'predicted_temperature': y_test_pred,
        'residual': y_test - y_test_pred
    })

    # Calculate metrics
    mae = np.abs(test_pred['residual']).mean()
    rmse = np.sqrt((test_pred['residual']**2).mean())
    ss_res = np.sum(test_pred['residual']**2)
    ss_tot = np.sum((y_test - y_test.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    within_10pct = 100 * np.sum(np.abs(test_pred['residual'] / test_pred['true_temperature']) <= 0.1) / len(test_pred)

    logger.info(f"   MAE: {mae:.1f} K, RMSE: {rmse:.1f} K, R²: {r2:.3f}, Within 10%: {within_10pct:.1f}%")

    # Prepare training and test dataframes
    df_train_subset = df_full[train_idx].to_pandas()
    df_test_subset = df_full[test_idx].to_pandas()

    # Add predictions to test subset
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
    colors_to_compare = ['bp_rp']
    color_labels = ['BP-RP']

    plot_color_distributions(df_train_subset, df_test_subset, colors_to_compare, color_labels,
                            MODEL_ID, SUBDIR)

    # 6. Color-temperature relations (bp_rp)
    plot_color_temp_relations(df_train_subset, df_test_subset, 'bp_rp', 'BP-RP',
                             MODEL_ID, SUBDIR,
                             train_temp_col='teff_gspphot',
                             pred_temp_col='teff_predicted')

    # 7. Feature importance
    plot_feature_importance(metadata['feature_importance'], MODEL_ID, SUBDIR,
                           MODEL_NAME, top_n=2)

    # Statistical comparison
    print_distribution_statistics(df_train_subset, df_test_subset,
                                 train_col='teff_gspphot',
                                 pred_col='teff_predicted')

    logger.info("\n" + "="*80)
    logger.info("VALIDATION PLOTS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: reports/figures/{SUBDIR}/")
    logger.info(f"Total plots generated: 7 figures")
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"  MAE:  {mae:.1f} K")
    logger.info(f"  RMSE: {rmse:.1f} K")
    logger.info(f"  R²:   {r2:.3f}")
    logger.info(f"  Within 10%: {within_10pct:.1f}%")


if __name__ == '__main__':
    main()
