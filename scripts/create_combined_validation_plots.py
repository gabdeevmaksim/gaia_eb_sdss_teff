#!/usr/bin/env python3
"""
Create standardized validation plots for Combined Pan-STARRS + 2MASS + Gaia model.

Uses shared visualization module for consistent style across all models.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. temp_distributions - Training vs Predictions comparison (histogram + CDF)
5. color_distributions - Compare color features between train/predict sets
6. color_temp_relations - Color-Temperature diagrams (J-H color)
7. feature_importance - Feature importance bar chart
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json

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
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("CREATE COMBINED PAN-STARRS + 2MASS VALIDATION PLOTS")
    logger.info("="*80)

    MODEL_ID = 'rf_combined_colors_20251018_115824'
    MODEL_NAME = 'Combined Pan-STARRS + 2MASS + Gaia Colors Model'
    SUBDIR = 'combined_validation'

    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load data
    logger.info("\nLoading data...")

    # Metadata
    with open(models_dir / f'{MODEL_ID}_metadata.json', 'r') as f:
        metadata = json.load(f)
    logger.info("   Metadata loaded")

    # Test predictions
    test_pred = pd.read_parquet(models_dir / f'{MODEL_ID}_test_predictions.parquet')
    logger.info(f"   Test predictions: {len(test_pred):,} objects")

    # Calculate metrics
    test_pred['abs_error'] = np.abs(test_pred['residual'])
    mae = test_pred['abs_error'].mean()
    rmse = np.sqrt((test_pred['residual']**2).mean())
    r2 = 1 - (test_pred['residual']**2).sum() / ((test_pred['true_temperature'] - test_pred['true_temperature'].mean())**2).sum()

    logger.info(f"   MAE: {mae:.1f} K, RMSE: {rmse:.1f} K, R²: {r2:.3f}")

    # Training data
    df_train = pd.read_parquet(processed_dir / 'combined_panstarrs_2mass_colors_training.parquet')
    logger.info(f"   Training data: {len(df_train):,} objects")

    # Predictions
    pred_file = processed_dir / f'combined_predictions_{MODEL_ID}.parquet'
    predictions = pd.read_parquet(pred_file)
    logger.info(f"   Predictions: {len(predictions):,} objects")

    # Calculate performance by temperature range
    logger.info("\nCalculating performance by temperature range...")
    bin_stats_df = calculate_bin_statistics(test_pred)

    # Generate plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING VALIDATION PLOTS")
    logger.info("="*80)

    plot_test_scatter(test_pred, mae, rmse, r2, MODEL_ID, SUBDIR, MODEL_NAME)

    plot_residuals(test_pred, MODEL_ID, SUBDIR)

    plot_performance_by_temp(test_pred, bin_stats_df, MODEL_ID, SUBDIR)

    plot_temp_distributions(df_train, predictions, MODEL_ID, SUBDIR,
                           train_col='teff_gspphot', pred_col='teff_predicted',
                           model_name=MODEL_NAME)

    # Color distributions (8 colors)
    colors_to_compare = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color',
                         'j_h_color', 'h_k_color']
    color_labels = ['BP-RP', 'g-r', 'r-i', 'i-z', 'J-H', 'H-K']

    plot_color_distributions(df_train, predictions, colors_to_compare, color_labels,
                            MODEL_ID, SUBDIR)

    # Color-temperature relations (use J-H, most important feature)
    plot_color_temp_relations(df_train, predictions, 'j_h_color', 'J-H',
                             MODEL_ID, SUBDIR,
                             train_temp_col='teff_gspphot',
                             pred_temp_col='teff_predicted')

    # Feature importance
    plot_feature_importance(metadata['feature_importance'], MODEL_ID, SUBDIR,
                           MODEL_NAME, top_n=20)

    # Statistical comparison
    print_distribution_statistics(df_train, predictions,
                                 train_col='teff_gspphot',
                                 pred_col='teff_predicted')

    logger.info("\n" + "="*80)
    logger.info("VALIDATION PLOTS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: reports/figures/{SUBDIR}/")
    logger.info(f"Total plots generated: 7 figures")


if __name__ == '__main__':
    main()
