#!/usr/bin/env python3
"""
Create standardized validation plots for Pan-STARRS + Gaia (with g magnitude) model.

Uses shared visualization module for consistent style across all models.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. temp_distributions - Training vs Predictions comparison (histogram + CDF)
5. color_distributions - Compare color features between train/predict sets
6. color_temp_relations - Color-Temperature diagrams (BP-RP color)
7. feature_importance - Top 5 features bar chart
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
    logger.info("CREATE PAN-STARRS + GAIA (WITH G MAG) VALIDATION PLOTS")
    logger.info("="*80)

    MODEL_ID = 'rf_ps_gaia_gmag_20251020_141321'
    MODEL_NAME = 'Pan-STARRS + Gaia (g mag) Model'
    SUBDIR = 'ps_gaia_gmag_validation'

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
    df_train = pd.read_parquet(processed_dir / 'panstarrs_gaia_with_gmag_training.parquet')
    logger.info(f"   Training data: {len(df_train):,} objects")

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

    # Note: We don't have predictions for objects without Gaia Teff yet
    # For now, just show training distribution
    logger.info("   Note: Prediction data not available yet")
    logger.info("   Skipping temperature distribution comparison")

    # 5. Color distributions
    logger.info("\n5. Creating color distributions plot...")

    # Define color columns (4 colors only, no magnitude)
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color']

    logger.info("   Note: Prediction data not available yet")
    logger.info("   Skipping color distribution comparison")

    # 6. Color-temperature relations
    logger.info("\n6. Creating color-temperature relations plot...")

    # Use BP-RP as primary color for plotting
    train_color_col = 'bp_rp'
    train_color_label = 'BP-RP'

    # For now, only show training data
    logger.info("   Note: Prediction data not available yet")
    logger.info("   Skipping color-temperature relations")

    # 7. Feature importance
    logger.info("\n7. Creating feature importance plot...")

    # Get feature importance from metadata
    feature_importance = metadata['feature_importance']

    # Pass the dictionary directly (not DataFrame)
    plot_feature_importance(feature_importance, MODEL_ID, SUBDIR, model_name=MODEL_NAME, top_n=5)
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
    logger.info(f"  - Gaia g magnitude dominates: {importance_df.iloc[0]['importance']:.1%} importance")
    logger.info(f"  - This indicates strong magnitude/distance bias")
    logger.info(f"  - Compare with color-only unified model (no g mag)")

    figures_dir = Path('reports/figures') / SUBDIR
    logger.info(f"\nAll plots saved to: {figures_dir}")
    logger.info(f"\nNote: Temperature and color distribution plots skipped")
    logger.info(f"      Run prediction script first to generate full validation")


if __name__ == '__main__':
    main()
