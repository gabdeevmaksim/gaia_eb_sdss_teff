#!/usr/bin/env python3
"""
Create standardized validation plots for Pan-STARRS only model.

This script generates 7 standardized validation plots for the basic
Random Forest model trained on Pan-STARRS colors only (g-r, r-i, i-z, B-V).

Model: rf_temperature_regressor_20251001_125556
Features: Pan-STARRS g-r, r-i, i-z colors + synthetic B-V + g magnitude
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl
import json
import logging

# Add parent directory to path for imports
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = 'rf_temperature_regressor_20251001_125556'
MODEL_NAME = 'Pan-STARRS Only Model'
SUBDIR = 'panstarrs_validation'

def main():
    """Generate all standardized validation plots for Pan-STARRS only model."""

    logger.info("=" * 70)
    logger.info(f"CREATING VALIDATION PLOTS FOR {MODEL_NAME}")
    logger.info("=" * 70)

    config = get_config()
    models_dir = config.get_path('models')
    processed_dir = config.get_path('processed')

    # Load metadata
    logger.info(f"Loading model metadata: {MODEL_ID}_metadata.json")
    metadata_file = models_dir / f'{MODEL_ID}_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Extract performance metrics
    test_metrics = metadata['performance']['test_set']
    mae = test_metrics['MAE_K']
    rmse = test_metrics['RMSE_K']
    r2 = test_metrics['R2_score']

    logger.info(f"Performance - MAE: {mae:.1f} K, RMSE: {rmse:.1f} K, R²: {r2:.3f}")

    # Load test predictions
    logger.info(f"Loading test predictions: {MODEL_ID}_test_predictions.parquet")
    test_pred_file = models_dir / f'{MODEL_ID}_test_predictions.parquet'
    test_pred = pd.read_parquet(test_pred_file)

    # Rename columns to match validation plot expectations
    test_pred = test_pred.rename(columns={
        'actual_teff_K': 'true_temperature',
        'predicted_teff_K': 'predicted_temperature',
        'residual_K': 'residual'
    })

    logger.info(f"Test predictions: {len(test_pred):,} objects")

    # Load full training data
    logger.info("Loading training data: ml_training_data_clean.parquet")
    train_data_file = processed_dir / 'ml_training_data_clean.parquet'
    train_data_pl = pl.read_parquet(train_data_file)
    train_data = train_data_pl.to_pandas()

    logger.info(f"Training data: {len(train_data):,} objects")

    # Load predictions dataset
    logger.info("Loading predictions: eb_panstarrs_basic_model_predictions.parquet")
    predictions_file = processed_dir / 'eb_panstarrs_basic_model_predictions.parquet'
    predictions = pd.read_parquet(predictions_file)

    logger.info(f"Predictions: {len(predictions):,} objects")

    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 70)

    # Plot 1: Test Scatter
    logger.info("1/7 Creating test scatter plot...")
    plot_test_scatter(test_pred, mae, rmse, r2, MODEL_ID, SUBDIR, MODEL_NAME)
    logger.info("    ✓ Test scatter complete")

    # Plot 2: Residuals
    logger.info("2/7 Creating residual analysis plots...")
    plot_residuals(test_pred, MODEL_ID, SUBDIR)
    logger.info("    ✓ Residuals complete")

    # Plot 3: Performance by Temperature
    logger.info("3/7 Creating performance by temperature plot...")
    bin_stats_df = calculate_bin_statistics(test_pred)
    plot_performance_by_temp(test_pred, bin_stats_df, MODEL_ID, SUBDIR)
    logger.info("    ✓ Performance by temperature complete")

    # Plot 4: Temperature Distributions
    logger.info("4/7 Creating temperature distribution plots...")
    plot_temp_distributions(train_data, predictions, MODEL_ID, SUBDIR,
                           train_col='teff_gspphot', pred_col='teff_predicted',
                           model_name=MODEL_NAME)
    logger.info("    ✓ Temperature distributions complete")

    # Plot 5: Color Distributions
    logger.info("5/7 Creating color distribution plots...")
    color_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color']
    color_labels = ['g-r', 'r-i', 'i-z', 'B-V']
    plot_color_distributions(train_data, predictions, color_cols, color_labels,
                            MODEL_ID, SUBDIR)
    logger.info("    ✓ Color distributions complete")

    # Plot 6: Color-Temperature Relations
    logger.info("6/7 Creating color-temperature relation plots...")
    # Use g-r as primary color
    plot_color_temp_relations(train_data, predictions, 'g_r_color', 'g-r',
                             MODEL_ID, SUBDIR,
                             train_temp_col='teff_gspphot',
                             pred_temp_col='teff_predicted')
    logger.info("    ✓ Color-temperature relations complete")

    # Plot 7: Feature Importance
    logger.info("7/7 Creating feature importance plot...")
    # Check for correct key name
    if 'feature_importance' in metadata:
        feature_imp = metadata['feature_importance']
    elif 'features' in metadata and 'feature_importances' in metadata['features']:
        feature_imp = metadata['features']['feature_importances']
    else:
        raise KeyError("Could not find feature importance in metadata")

    plot_feature_importance(feature_imp, MODEL_ID, SUBDIR, MODEL_NAME, top_n=5)
    logger.info("    ✓ Feature importance complete")

    # Print distribution statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("DISTRIBUTION STATISTICS")
    logger.info("=" * 70)
    print_distribution_statistics(train_data, predictions,
                                 train_col='teff_gspphot',
                                 pred_col='teff_predicted')

    # Print binned performance statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("PERFORMANCE BY TEMPERATURE RANGE")
    logger.info("=" * 70)
    bin_stats = calculate_bin_statistics(test_pred)

    figures_dir = config.get_path('figures')
    output_dir = figures_dir / SUBDIR

    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION PLOTS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"All figures saved to: {output_dir}/")
    logger.info("Total plots generated: 7 figures")
    logger.info("")
    logger.info("Generated plots:")
    logger.info(f"  1. {MODEL_ID}_test_scatter.png")
    logger.info(f"  2. {MODEL_ID}_residuals.png")
    logger.info(f"  3. {MODEL_ID}_performance_by_temp.png")
    logger.info(f"  4. {MODEL_ID}_temp_distributions.png")
    logger.info(f"  5. {MODEL_ID}_color_distributions.png")
    logger.info(f"  6. {MODEL_ID}_color_temp_relations.png")
    logger.info(f"  7. {MODEL_ID}_feature_importance.png")
    logger.info("")

if __name__ == '__main__':
    main()
