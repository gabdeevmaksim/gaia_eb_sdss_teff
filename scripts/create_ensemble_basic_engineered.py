#!/usr/bin/env python3
"""
Create ensemble by averaging basic colors model and engineered features model.

This script combines predictions from:
1. Basic Pan-STARRS colors model (rf_temperature_regressor_20251001_125556)
   - 5 features: g-r, r-i, i-z, B-V, gPSFMag
   - MAE: 576 K
2. Engineered features model (rf_temperature_regressor_feature_engineering_20251002_210423)
   - 20 engineered features from same base colors
   - MAE: 318 K

The goal is to reduce systematic bias where one model may overestimate hot stars
while the other underestimates them.

Usage:
    python scripts/create_ensemble_basic_engineered.py

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Create ensemble predictions and evaluate performance."""

    models_dir = Path('models')

    logger.info("=" * 80)
    logger.info("CREATE ENSEMBLE: BASIC COLORS vs ENGINEERED FEATURES")
    logger.info("=" * 80)

    # Model identifiers
    MODEL1_ID = 'rf_temperature_regressor_20251001_125556'
    MODEL2_ID = 'rf_temperature_regressor_feature_engineering_20251002_210423'

    MODEL1_NAME = 'Basic Pan-STARRS Colors'
    MODEL2_NAME = 'Engineered Features'

    logger.info(f"\nModel 1: {MODEL1_NAME}")
    logger.info(f"  ID: {MODEL1_ID}")
    logger.info(f"  Features: 5 (g-r, r-i, i-z, B-V, gPSFMag)")
    logger.info(f"  Expected: MAE ~576 K")

    logger.info(f"\nModel 2: {MODEL2_NAME}")
    logger.info(f"  ID: {MODEL2_ID}")
    logger.info(f"  Features: 20 engineered features")
    logger.info(f"  Expected: MAE ~318 K (45% improvement)")

    # Load test predictions from both models
    logger.info("\n" + "=" * 80)
    logger.info("LOADING TEST PREDICTIONS")
    logger.info("=" * 80)

    logger.info(f"\nLoading Model 1 predictions...")
    pred1_file = models_dir / f'{MODEL1_ID}_test_predictions.parquet'
    df_pred1 = pd.read_parquet(pred1_file)
    logger.info(f"  Loaded {len(df_pred1):,} predictions")
    logger.info(f"  Columns: {list(df_pred1.columns)}")

    logger.info(f"\nLoading Model 2 predictions...")
    pred2_file = models_dir / f'{MODEL2_ID}_test_predictions.parquet'
    df_pred2 = pd.read_parquet(pred2_file)
    logger.info(f"  Loaded {len(df_pred2):,} predictions")
    logger.info(f"  Columns: {list(df_pred2.columns)}")

    # Standardize column names
    logger.info("\n" + "=" * 80)
    logger.info("STANDARDIZING COLUMN NAMES")
    logger.info("=" * 80)

    # Model 1: actual_teff_K -> true_temperature, predicted_teff_K -> predicted_temperature
    df_pred1 = df_pred1.rename(columns={
        'actual_teff_K': 'true_temperature',
        'predicted_teff_K': 'predicted_temperature',
        'residual_K': 'residual'
    })
    logger.info("✓ Model 1 columns standardized")

    # Model 2: actual_temperature -> true_temperature, residuals -> residual
    df_pred2 = df_pred2.rename(columns={
        'actual_temperature': 'true_temperature',
        'residuals': 'residual'
    })
    logger.info("✓ Model 2 columns standardized")

    # Check test set alignment
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING TEST SETS")
    logger.info("=" * 80)

    if len(df_pred1) != len(df_pred2):
        logger.error(f"Different test set sizes: {len(df_pred1)} vs {len(df_pred2)}")
        logger.error("Cannot ensemble - models have different test splits!")
        return

    logger.info(f"✓ Same test set size: {len(df_pred1):,} objects")

    # Check if they have the same test set (same true temperatures)
    if not np.allclose(df_pred1['true_temperature'].values,
                       df_pred2['true_temperature'].values,
                       rtol=0, atol=0.1):
        logger.error("Test sets do not match! Cannot ensemble.")
        logger.error("Models have different random splits or different data filtering")
        return

    logger.info("✓ Test sets are aligned (same objects, same order)")

    # Create ensemble dataframe
    logger.info("\n" + "=" * 80)
    logger.info("CREATING ENSEMBLE")
    logger.info("=" * 80)

    df_ensemble = pd.DataFrame({
        'true_temperature': df_pred1['true_temperature'],
        'pred_basic': df_pred1['predicted_temperature'],
        'pred_engineered': df_pred2['predicted_temperature']
    })

    # Calculate ensemble prediction (simple average)
    logger.info("\nCalculating ensemble prediction (simple average)...")
    df_ensemble['predicted_temperature'] = (df_ensemble['pred_basic'] +
                                             df_ensemble['pred_engineered']) / 2
    df_ensemble['residual'] = df_ensemble['predicted_temperature'] - df_ensemble['true_temperature']
    df_ensemble['abs_error'] = np.abs(df_ensemble['residual'])

    logger.info(f"  Ensemble predictions created: {len(df_ensemble):,}")

    # Calculate metrics for all three approaches
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    # Model 1 metrics (basic)
    mae1 = mean_absolute_error(df_ensemble['true_temperature'], df_ensemble['pred_basic'])
    rmse1 = np.sqrt(mean_squared_error(df_ensemble['true_temperature'], df_ensemble['pred_basic']))
    r2_1 = r2_score(df_ensemble['true_temperature'], df_ensemble['pred_basic'])

    # Model 2 metrics (engineered)
    mae2 = mean_absolute_error(df_ensemble['true_temperature'], df_ensemble['pred_engineered'])
    rmse2 = np.sqrt(mean_squared_error(df_ensemble['true_temperature'], df_ensemble['pred_engineered']))
    r2_2 = r2_score(df_ensemble['true_temperature'], df_ensemble['pred_engineered'])

    # Ensemble metrics
    mae_ens = df_ensemble['abs_error'].mean()
    rmse_ens = np.sqrt((df_ensemble['residual']**2).mean())
    r2_ens = r2_score(df_ensemble['true_temperature'], df_ensemble['predicted_temperature'])

    # Calculate accuracy within thresholds
    def calc_accuracy(y_true, y_pred):
        error_pct = np.abs(y_pred - y_true) / y_true * 100
        within_5 = (error_pct <= 5).sum()
        within_10 = (error_pct <= 10).sum()
        within_20 = (error_pct <= 20).sum()
        return within_5, within_10, within_20

    w5_1, w10_1, w20_1 = calc_accuracy(df_ensemble['true_temperature'], df_ensemble['pred_basic'])
    w5_2, w10_2, w20_2 = calc_accuracy(df_ensemble['true_temperature'], df_ensemble['pred_engineered'])
    w5_ens, w10_ens, w20_ens = calc_accuracy(df_ensemble['true_temperature'], df_ensemble['predicted_temperature'])

    # Calculate improvement
    best_individual_mae = min(mae1, mae2)
    mae_improvement = (best_individual_mae - mae_ens) / best_individual_mae * 100

    logger.info(f"\n{MODEL1_NAME}:")
    logger.info(f"  MAE:  {mae1:.1f} K")
    logger.info(f"  RMSE: {rmse1:.1f} K")
    logger.info(f"  R²:   {r2_1:.3f}")
    logger.info(f"  Within 10%: {w10_1:,} ({100*w10_1/len(df_ensemble):.1f}%)")

    logger.info(f"\n{MODEL2_NAME}:")
    logger.info(f"  MAE:  {mae2:.1f} K")
    logger.info(f"  RMSE: {rmse2:.1f} K")
    logger.info(f"  R²:   {r2_2:.3f}")
    logger.info(f"  Within 10%: {w10_2:,} ({100*w10_2/len(df_ensemble):.1f}%)")

    logger.info(f"\nENSEMBLE (Average):")
    logger.info(f"  MAE:  {mae_ens:.1f} K")
    logger.info(f"  RMSE: {rmse_ens:.1f} K")
    logger.info(f"  R²:   {r2_ens:.3f}")
    logger.info(f"  Within 10%: {w10_ens:,} ({100*w10_ens/len(df_ensemble):.1f}%)")

    logger.info(f"\nIMPROVEMENT:")
    if mae_ens < best_individual_mae:
        logger.info(f"  ✓ Ensemble is BETTER than both individual models")
        logger.info(f"  Improvement: {mae_improvement:+.1f}% MAE reduction")
    else:
        logger.info(f"  ✗ Ensemble is WORSE than best individual model")
        logger.info(f"  Change: {mae_improvement:+.1f}% MAE")

    # Analyze by temperature range
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BY TEMPERATURE RANGE")
    logger.info("=" * 80)

    temp_bins = [0, 4000, 5000, 6000, 7000, 8000, 100000]
    temp_labels = ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-8000K', '>8000K']

    df_ensemble['temp_bin'] = pd.cut(df_ensemble['true_temperature'],
                                      bins=temp_bins, labels=temp_labels)

    logger.info(f"\n{'Range':<15} {'Count':>8} {'Basic':>12} {'Engineered':>12} {'Ensemble':>12} {'Best':>12}")
    logger.info("-" * 82)

    for label in temp_labels:
        mask = df_ensemble['temp_bin'] == label
        if mask.sum() == 0:
            continue

        count = mask.sum()
        mae1_bin = np.abs(df_ensemble.loc[mask, 'pred_basic'] - df_ensemble.loc[mask, 'true_temperature']).mean()
        mae2_bin = np.abs(df_ensemble.loc[mask, 'pred_engineered'] - df_ensemble.loc[mask, 'true_temperature']).mean()
        mae_ens_bin = df_ensemble.loc[mask, 'abs_error'].mean()

        best_model = "Basic" if mae1_bin < mae2_bin else "Engineered"
        if mae_ens_bin < min(mae1_bin, mae2_bin):
            best_model = "Ensemble"

        logger.info(f"{label:<15} {count:>8,} {mae1_bin:>11.1f} K {mae2_bin:>11.1f} K {mae_ens_bin:>11.1f} K {best_model:>12}")

    # Save ensemble predictions
    logger.info("\n" + "=" * 80)
    logger.info("SAVING ENSEMBLE PREDICTIONS")
    logger.info("=" * 80)

    ensemble_id = 'ensemble_basic_engineered'
    output_file = models_dir / f'{ensemble_id}_test_predictions.parquet'

    # Select columns for output
    output_cols = ['true_temperature', 'predicted_temperature', 'residual',
                   'pred_basic', 'pred_engineered']
    df_output = df_ensemble[output_cols].copy()

    logger.info(f"\nSaving ensemble predictions: {output_file.name}")
    df_output.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Save summary
    summary_file = models_dir / f'{ensemble_id}_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("ENSEMBLE MODEL - BASIC COLORS vs ENGINEERED FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Ensemble ID: {ensemble_id}\n")
        f.write(f"Method: Simple average of two models\n\n")

        f.write(f"COMPONENT MODELS:\n")
        f.write(f"  Model 1: {MODEL1_NAME}\n")
        f.write(f"    ID: {MODEL1_ID}\n")
        f.write(f"    Features: 5 (g-r, r-i, i-z, B-V, gPSFMag)\n")
        f.write(f"    MAE: {mae1:.1f} K, RMSE: {rmse1:.1f} K, R²: {r2_1:.3f}\n")
        f.write(f"    Within 10%: {100*w10_1/len(df_ensemble):.1f}%\n\n")

        f.write(f"  Model 2: {MODEL2_NAME}\n")
        f.write(f"    ID: {MODEL2_ID}\n")
        f.write(f"    Features: 20 engineered features\n")
        f.write(f"    MAE: {mae2:.1f} K, RMSE: {rmse2:.1f} K, R²: {r2_2:.3f}\n")
        f.write(f"    Within 10%: {100*w10_2/len(df_ensemble):.1f}%\n\n")

        f.write(f"ENSEMBLE PERFORMANCE:\n")
        f.write(f"  Test objects: {len(df_ensemble):,}\n")
        f.write(f"  MAE:  {mae_ens:.1f} K\n")
        f.write(f"  RMSE: {rmse_ens:.1f} K\n")
        f.write(f"  R²:   {r2_ens:.3f}\n")
        f.write(f"  Within 10%: {100*w10_ens/len(df_ensemble):.1f}%\n\n")

        if mae_ens < best_individual_mae:
            f.write(f"IMPROVEMENT:\n")
            f.write(f"  Ensemble is BETTER than both individual models\n")
            f.write(f"  MAE reduction: {mae_improvement:.1f}%\n\n")
        else:
            f.write(f"RESULT:\n")
            f.write(f"  Ensemble performance: {mae_improvement:+.1f}% vs best\n\n")

        f.write(f"PERFORMANCE BY TEMPERATURE RANGE:\n")
        f.write(f"  Range          Count      Basic  Engineered   Ensemble       Best\n")
        f.write(f"  {'-'*70}\n")
        for label in temp_labels:
            mask = df_ensemble['temp_bin'] == label
            if mask.sum() == 0:
                continue
            count = mask.sum()
            mae1_bin = np.abs(df_ensemble.loc[mask, 'pred_basic'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae2_bin = np.abs(df_ensemble.loc[mask, 'pred_engineered'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae_ens_bin = df_ensemble.loc[mask, 'abs_error'].mean()
            best_model = "Basic" if mae1_bin < mae2_bin else "Engineered"
            if mae_ens_bin < min(mae1_bin, mae2_bin):
                best_model = "Ensemble"
            f.write(f"  {label:<12} {count:>8,} {mae1_bin:>9.1f} K {mae2_bin:>10.1f} K {mae_ens_bin:>10.1f} K {best_model:>12}\n")

        f.write(f"\nNOTES:\n")
        f.write(f"  - Basic model uses 5 simple color features\n")
        f.write(f"  - Engineered model uses 20 polynomial/interaction features\n")
        f.write(f"  - Ensemble combines predictions via simple averaging\n")
        f.write(f"  - Goal: Reduce systematic bias in temperature extremes\n")

    logger.info(f"  Summary saved: {summary_file.name}")

    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nEnsemble MAE: {mae_ens:.1f} K")
    logger.info(f"Best individual MAE: {best_individual_mae:.1f} K")
    if mae_ens < best_individual_mae:
        logger.info(f"Result: ✓ IMPROVED by {mae_improvement:.1f}%")
    else:
        logger.info(f"Result: ✗ Changed by {mae_improvement:+.1f}%")

    logger.info(f"\nOutput files:")
    logger.info(f"  {output_file}")
    logger.info(f"  {summary_file}")


if __name__ == '__main__':
    main()
