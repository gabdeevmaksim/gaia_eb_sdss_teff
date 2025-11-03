#!/usr/bin/env python3
"""
Create ensemble by predicting on the same test set.

Loads both trained models and makes predictions on the same test data,
then averages the predictions to create an ensemble.

Models:
1. Unified WITH g magnitude (rf_unified_engineered_20251015_180614)
2. Unified WITHOUT g magnitude (rf_unified_engineered_20251016_112332)

Usage:
    python scripts/create_ensemble_same_test_set.py

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
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Create ensemble predictions on same test set."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("=" * 80)
    logger.info("ENSEMBLE: PREDICT ON SAME TEST SET")
    logger.info("=" * 80)

    MODEL1_ID = 'rf_unified_engineered_20251015_180614'
    MODEL2_ID = 'rf_unified_engineered_20251016_112332'

    # Load color-only training data (compatible with both models)
    logger.info("\nLoading color-only training data...")
    train_file = processed_dir / 'eb_unified_features_engineered_train.parquet'
    df_train = pd.read_parquet(train_file)
    logger.info(f"  Loaded: {len(df_train):,} objects")
    logger.info(f"  Columns: {len(df_train.columns)}")

    # Split with same random_state as original training
    logger.info("\nSplitting data (random_state=42, test_size=0.2)...")

    # Prepare features and target
    feature_cols = [col for col in df_train.columns
                    if col not in ['original_ext_source_id', 'gaia_source_id', 'teff_gspphot']]

    X = df_train[feature_cols]
    y = df_train['teff_gspphot']
    source_ids = df_train['gaia_source_id']

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, source_ids,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"  Test set: {len(X_test):,} objects")

    # Load Model 1 (with g magnitude)
    logger.info("\nLoading Model 1 (WITH g magnitude)...")
    model1_file = models_dir / f'{MODEL1_ID}.pkl'
    selector1_file = models_dir / f'{MODEL1_ID}_selector.pkl'

    model1 = joblib.load(model1_file)
    selector1 = joblib.load(selector1_file)
    logger.info(f"  ✓ Model 1 loaded")

    # Load Model 2 (without g magnitude - color only)
    logger.info("\nLoading Model 2 (WITHOUT g magnitude)...")
    model2_file = models_dir / f'{MODEL2_ID}.pkl'
    selector2_file = models_dir / f'{MODEL2_ID}_selector.pkl'

    model2 = joblib.load(model2_file)
    selector2 = joblib.load(selector2_file)
    logger.info(f"  ✓ Model 2 loaded")

    # Make predictions with Model 1
    logger.info("\nMaking predictions with Model 1...")
    X_test_selected1 = selector1.transform(X_test)
    pred1 = model1.predict(X_test_selected1)
    logger.info(f"  Predictions: {len(pred1):,}")

    # Make predictions with Model 2
    logger.info("\nMaking predictions with Model 2...")
    X_test_selected2 = selector2.transform(X_test)
    pred2 = model2.predict(X_test_selected2)
    logger.info(f"  Predictions: {len(pred2):,}")

    # Create ensemble
    logger.info("\nCreating ensemble (simple average)...")
    pred_ensemble = (pred1 + pred2) / 2

    # Calculate metrics
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    mae1 = mean_absolute_error(y_test, pred1)
    rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
    r2_1 = r2_score(y_test, pred1)

    mae2 = mean_absolute_error(y_test, pred2)
    rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
    r2_2 = r2_score(y_test, pred2)

    mae_ens = mean_absolute_error(y_test, pred_ensemble)
    rmse_ens = np.sqrt(mean_squared_error(y_test, pred_ensemble))
    r2_ens = r2_score(y_test, pred_ensemble)

    def calc_accuracy(y_true, y_pred):
        error_pct = np.abs(y_pred - y_true) / y_true * 100
        return (error_pct <= 10).sum()

    w10_1 = calc_accuracy(y_test, pred1)
    w10_2 = calc_accuracy(y_test, pred2)
    w10_ens = calc_accuracy(y_test, pred_ensemble)

    logger.info(f"\nModel 1 (WITH g magnitude):")
    logger.info(f"  MAE:  {mae1:.1f} K")
    logger.info(f"  RMSE: {rmse1:.1f} K")
    logger.info(f"  R²:   {r2_1:.3f}")
    logger.info(f"  Within 10%: {100*w10_1/len(y_test):.1f}%")

    logger.info(f"\nModel 2 (WITHOUT g magnitude):")
    logger.info(f"  MAE:  {mae2:.1f} K")
    logger.info(f"  RMSE: {rmse2:.1f} K")
    logger.info(f"  R²:   {r2_2:.3f}")
    logger.info(f"  Within 10%: {100*w10_2/len(y_test):.1f}%")

    logger.info(f"\nENSEMBLE (Average):")
    logger.info(f"  MAE:  {mae_ens:.1f} K")
    logger.info(f"  RMSE: {rmse_ens:.1f} K")
    logger.info(f"  R²:   {r2_ens:.3f}")
    logger.info(f"  Within 10%: {100*w10_ens/len(y_test):.1f}%")

    best_mae = min(mae1, mae2)
    improvement = (best_mae - mae_ens) / best_mae * 100

    logger.info(f"\nRESULT:")
    if mae_ens < best_mae:
        logger.info(f"  ✓ Ensemble BETTER: {improvement:.1f}% MAE reduction")
    else:
        logger.info(f"  ✗ Ensemble worse: {improvement:+.1f}% MAE change")

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    df_results = pd.DataFrame({
        'gaia_source_id': ids_test.values,
        'true_temperature': y_test.values,
        'pred_with_gmag': pred1,
        'pred_without_gmag': pred2,
        'predicted_temperature': pred_ensemble,
        'residual': pred_ensemble - y_test.values
    })

    output_file = models_dir / 'ensemble_unified_with_without_gmag_test_predictions.parquet'
    df_results.to_parquet(output_file, index=False)
    logger.info(f"  ✓ Saved: {output_file.name}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
