#!/usr/bin/env python3
"""
Create standardized validation plots for Pan-STARRS + Gaia + 2MASS (with g mag) model.

Uses shared visualization module for consistent style across all models.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. temp_distributions - Training vs Predictions comparison (histogram + CDF)
5. color_distributions - Compare color features between train/predict sets
6. color_temp_relations - Color-Temperature diagrams (J-H color)
7. feature_importance - Top 8 features bar chart
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
    logger.info("CREATE PAN-STARRS + GAIA + 2MASS (WITH G MAG) VALIDATION PLOTS")
    logger.info("="*80)

    MODEL_ID = 'rf_ps_gaia_2mass_gmag_20251020_142522'
    MODEL_NAME = 'Pan-STARRS + Gaia + 2MASS (g mag) Model'
    SUBDIR = 'ps_gaia_2mass_gmag_validation'

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
    df_train = pd.read_parquet(processed_dir / 'ps_gaia_2mass_gmag_training.parquet')
    logger.info(f"   Training data: {len(df_train):,} objects")

    # Predictions
    pred_file = processed_dir / f'predictions_{MODEL_ID}.parquet'
    predictions = pd.read_parquet(pred_file)
    logger.info(f"   Predictions: {len(predictions):,} objects")

    # Calculate performance by temperature range
    logger.info("\nCalculating performance by temperature range...")
    bin_stats_df = calculate_bin_statistics(test_pred)

    # Generate plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING VALIDATION PLOTS")
    logger.info("="*80)

    # 1. Test scatter plot
    logger.info("\n1. Creating test scatter plot...")
    plot_test_scatter(test_pred, mae, rmse, r2, MODEL_ID, SUBDIR, MODEL_NAME)
    logger.info("   ✓ Test scatter plot saved")

    # 2. Residuals plot
    logger.info("\n2. Creating residuals plot...")
    plot_residuals(test_pred, MODEL_ID, SUBDIR)
    logger.info("   ✓ Residuals plot saved")

    # 3. Performance by temperature
    logger.info("\n3. Creating performance by temperature plot...")
    plot_performance_by_temp(test_pred, bin_stats_df, MODEL_ID, SUBDIR)
    logger.info("   ✓ Performance by temperature plot saved")

    # 4. Temperature distributions
    logger.info("\n4. Creating temperature distributions plot...")

    # Pass DataFrames with correct column names
    plot_temp_distributions(
        df_train, predictions, MODEL_ID, SUBDIR,
        train_col='teff_gspphot',
        pred_col='teff_predicted',
        model_name=MODEL_NAME
    )
    logger.info("   ✓ Temperature distributions plot saved")

    # Print distribution statistics
    print_distribution_statistics(
        df_train, predictions,
        train_col='teff_gspphot',
        pred_col='teff_predicted'
    )

    # 5. Color distributions
    logger.info("\n5. Creating color distributions plot...")

    # Define top 6 colors (grid supports max 6, exclude magnitude and least important color)
    # Most important colors based on feature importance: j_h, bp_rp, i_z, g_r, r_i, j_k
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color', 'j_h_color', 'j_k_color']
    color_labels = ['BP-RP', 'g-r', 'r-i', 'i-z', 'J-H', 'J-K']

    # Prepare dataframes
    df_train_colors = df_train[color_cols].copy()
    df_pred_colors = predictions[color_cols].copy()

    plot_color_distributions(df_train_colors, df_pred_colors, color_cols, color_labels, MODEL_ID, SUBDIR)
    logger.info("   ✓ Color distributions plot saved")

    # 6. Color-temperature relations
    logger.info("\n6. Creating color-temperature relations plot...")

    # Use J-H as primary color (most important after g_mag)
    color_col = 'j_h_color'
    color_label = 'J-H'

    plot_color_temp_relations(
        df_train, predictions,
        color_col, color_label,
        MODEL_ID, SUBDIR
    )
    logger.info("   ✓ Color-temperature relations plot saved")

    # 7. Feature importance
    logger.info("\n7. Creating feature importance plot...")

    # Get feature importance from metadata
    feature_importance = metadata['feature_importance']

    # Pass the dictionary directly
    plot_feature_importance(feature_importance, MODEL_ID, SUBDIR, model_name=MODEL_NAME, top_n=8)
    logger.info("   ✓ Feature importance plot saved")

    # Create DataFrame for summary display
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION PLOTS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nModel: {MODEL_NAME}")
    logger.info(f"Test MAE: {mae:.1f} K")
    logger.info(f"Test RMSE: {rmse:.1f} K")
    logger.info(f"Test R²: {r2:.3f}")

    logger.info(f"\nFeature importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:<20} {row['importance']:.4f}")

    logger.info(f"\nKey observations:")
    logger.info(f"  - Gaia g magnitude: {importance_df.iloc[0]['importance']:.1%} importance")
    logger.info(f"  - J-H color (2MASS): {importance_df.iloc[1]['importance']:.1%} importance")
    logger.info(f"  - IR colors provide valuable temperature information")
    logger.info(f"  - Magnitude bias present but reduced compared to optical-only model")

    figures_dir = Path('reports/figures') / SUBDIR
    logger.info(f"\nAll plots saved to: {figures_dir}")


if __name__ == '__main__':
    main()
