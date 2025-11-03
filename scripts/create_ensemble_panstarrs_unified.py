#!/usr/bin/env python3
"""
Create ensemble by averaging Pan-STARRS basic model and unified color-only model.

This script combines predictions from:
1. Basic Pan-STARRS model (rf_temperature_regressor_20251001_125556)
   - 5 features: g-r, r-i, i-z, B-V, gPSFMag
   - MAE: ~576 K
2. Unified color-only model (rf_unified_engineered_20251016_112332)
   - 20 engineered color features (NO magnitude)
   - MAE: ~765 K

Since the models were trained on different datasets, this script:
1. Loads the unified features training data (common data source)
2. Loads both trained models
3. Makes predictions on the same test split
4. Creates ensemble by averaging predictions

Usage:
    python scripts/create_ensemble_panstarrs_unified.py

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
from src.features import engineer_all_features

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
    logger.info("ENSEMBLE: PAN-STARRS BASIC + UNIFIED COLOR-ONLY (SAME TEST SET)")
    logger.info("=" * 80)

    MODEL1_ID = 'rf_temperature_regressor_20251001_125556'
    MODEL2_ID = 'rf_unified_engineered_20251016_112332'

    MODEL1_NAME = 'Pan-STARRS Basic'
    MODEL2_NAME = 'Unified Color-Only'

    logger.info(f"\nModel 1: {MODEL1_NAME}")
    logger.info(f"  ID: {MODEL1_ID}")
    logger.info(f"  Features: 5 (g-r, r-i, i-z, B-V, gPSFMag)")
    logger.info(f"  Expected: MAE ~576 K, includes magnitude")

    logger.info(f"\nModel 2: {MODEL2_NAME}")
    logger.info(f"  ID: {MODEL2_ID}")
    logger.info(f"  Features: 20 engineered color features (NO magnitude)")
    logger.info(f"  Expected: MAE ~765 K, physically correct")

    # Load unified features training data (common data source)
    logger.info("\n" + "=" * 80)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 80)

    logger.info("\nLoading unified features training data...")
    train_file = processed_dir / 'eb_unified_features_engineered_train.parquet'
    df_train = pd.read_parquet(train_file)
    logger.info(f"  Loaded: {len(df_train):,} objects")
    logger.info(f"  Columns: {len(df_train.columns)}")

    # Split with same random_state as original training (42)
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

    logger.info(f"  Training set: {len(X_train):,} objects")
    logger.info(f"  Test set: {len(X_test):,} objects")

    # Load Model 1 (Pan-STARRS Basic)
    logger.info("\n" + "=" * 80)
    logger.info("LOADING MODELS")
    logger.info("=" * 80)

    logger.info(f"\nLoading Model 1 ({MODEL1_NAME})...")
    model1_file = models_dir / f'{MODEL1_ID}.pkl'

    if not model1_file.exists():
        logger.error(f"Model 1 file not found: {model1_file}")
        logger.error("The Pan-STARRS basic model may need to be retrained or located")
        return

    model1 = joblib.load(model1_file)
    logger.info(f"  ✓ Model 1 loaded")

    # Load Model 2 (Unified Color-Only)
    logger.info(f"\nLoading Model 2 ({MODEL2_NAME})...")
    model2_file = models_dir / f'{MODEL2_ID}.pkl'
    selector2_file = models_dir / f'{MODEL2_ID}_selector.pkl'

    model2 = joblib.load(model2_file)
    selector2 = joblib.load(selector2_file)
    logger.info(f"  ✓ Model 2 loaded")

    # Prepare features for Model 1 (Pan-STARRS Basic: needs 5 specific features)
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 80)

    logger.info("\nPreparing features for Model 1 (Pan-STARRS Basic)...")
    # Model 1 needs: g-r, r-i, i-z, B-V, gPSFMag
    model1_features = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag']

    # Check if all features are available
    missing_features = [f for f in model1_features if f not in X_test.columns]
    if missing_features:
        logger.error(f"Missing features for Model 1: {missing_features}")
        logger.error("The unified dataset may not have gPSFMag")
        logger.info("Will attempt to load original Pan-STARRS data...")

        # Try loading original ML data with gPSFMag
        ml_data_file = processed_dir / 'ml_training_data_with_gaia.parquet'
        if not ml_data_file.exists():
            logger.error(f"Cannot find ML data with gPSFMag: {ml_data_file}")
            return

        logger.info(f"Loading {ml_data_file.name}...")
        df_ml = pd.read_parquet(ml_data_file)
        logger.info(f"  Loaded: {len(df_ml):,} objects")

        # Merge with test set to get gPSFMag
        logger.info("Merging to get gPSFMag for test objects...")
        df_test_with_mag = X_test.copy()
        df_test_with_mag['gaia_source_id'] = ids_test.values
        df_test_with_mag = df_test_with_mag.merge(
            df_ml[['original_ext_source_id', 'gPSFMag']],
            left_on='gaia_source_id',
            right_on='original_ext_source_id',
            how='left'
        )

        X_test_model1 = df_test_with_mag[model1_features].values
        logger.info(f"  Test features prepared: {X_test_model1.shape}")
    else:
        X_test_model1 = X_test[model1_features].values
        logger.info(f"  Test features prepared: {X_test_model1.shape}")

    logger.info("\nPreparing features for Model 2 (Unified Color-Only)...")
    # Remove temperature columns that may not have been in training data
    temp_cols_to_drop = ['Te_gr', 'Te_ri', 'Te_iz']
    X_test_clean = X_test.drop(columns=temp_cols_to_drop, errors='ignore')
    logger.info(f"  Features before selection: {X_test_clean.shape}")

    X_test_model2 = selector2.transform(X_test_clean)
    logger.info(f"  Test features selected: {X_test_model2.shape}")

    # Make predictions
    logger.info("\n" + "=" * 80)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 80)

    logger.info("\nMaking predictions with Model 1 (Pan-STARRS Basic)...")
    pred1 = model1.predict(X_test_model1)
    logger.info(f"  Predictions: {len(pred1):,}")

    logger.info("\nMaking predictions with Model 2 (Unified Color-Only)...")
    pred2 = model2.predict(X_test_model2)
    logger.info(f"  Predictions: {len(pred2):,}")

    # Create ensemble
    logger.info("\n" + "=" * 80)
    logger.info("CREATING ENSEMBLE")
    logger.info("=" * 80)

    logger.info("\nCalculating ensemble prediction (simple average)...")
    pred_ensemble = (pred1 + pred2) / 2

    df_ensemble = pd.DataFrame({
        'gaia_source_id': ids_test.values,
        'true_temperature': y_test.values,
        'pred_panstarrs': pred1,
        'pred_unified': pred2,
        'predicted_temperature': pred_ensemble,
        'residual': pred_ensemble - y_test.values
    })
    df_ensemble['abs_error'] = np.abs(df_ensemble['residual'])

    logger.info(f"  Ensemble predictions created: {len(df_ensemble):,}")

    # Calculate metrics for all three approaches
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    # Model 1 metrics (Pan-STARRS)
    mae1 = mean_absolute_error(y_test, pred1)
    rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
    r2_1 = r2_score(y_test, pred1)

    # Model 2 metrics (Unified)
    mae2 = mean_absolute_error(y_test, pred2)
    rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
    r2_2 = r2_score(y_test, pred2)

    # Ensemble metrics
    mae_ens = mean_absolute_error(y_test, pred_ensemble)
    rmse_ens = np.sqrt(mean_squared_error(y_test, pred_ensemble))
    r2_ens = r2_score(y_test, pred_ensemble)

    # Calculate accuracy within thresholds
    def calc_accuracy(y_true, y_pred):
        error_pct = np.abs(y_pred - y_true) / y_true * 100
        within_5 = (error_pct <= 5).sum()
        within_10 = (error_pct <= 10).sum()
        within_20 = (error_pct <= 20).sum()
        return within_5, within_10, within_20

    w5_1, w10_1, w20_1 = calc_accuracy(y_test, pred1)
    w5_2, w10_2, w20_2 = calc_accuracy(y_test, pred2)
    w5_ens, w10_ens, w20_ens = calc_accuracy(y_test, pred_ensemble)

    # Calculate improvement
    best_individual_mae = min(mae1, mae2)
    mae_improvement = (best_individual_mae - mae_ens) / best_individual_mae * 100

    n_test = len(y_test)

    logger.info(f"\n{MODEL1_NAME}:")
    logger.info(f"  MAE:  {mae1:.1f} K")
    logger.info(f"  RMSE: {rmse1:.1f} K")
    logger.info(f"  R²:   {r2_1:.3f}")
    logger.info(f"  Within 10%: {w10_1:,} ({100*w10_1/n_test:.1f}%)")

    logger.info(f"\n{MODEL2_NAME}:")
    logger.info(f"  MAE:  {mae2:.1f} K")
    logger.info(f"  RMSE: {rmse2:.1f} K")
    logger.info(f"  R²:   {r2_2:.3f}")
    logger.info(f"  Within 10%: {w10_2:,} ({100*w10_2/n_test:.1f}%)")

    logger.info(f"\nENSEMBLE (Average):")
    logger.info(f"  MAE:  {mae_ens:.1f} K")
    logger.info(f"  RMSE: {rmse_ens:.1f} K")
    logger.info(f"  R²:   {r2_ens:.3f}")
    logger.info(f"  Within 10%: {w10_ens:,} ({100*w10_ens/n_test:.1f}%)")

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

    logger.info(f"\n{'Range':<15} {'Count':>8} {'Pan-STARRS':>12} {'Unified':>12} {'Ensemble':>12} {'Best':>12}")
    logger.info("-" * 82)

    for label in temp_labels:
        mask = df_ensemble['temp_bin'] == label
        if mask.sum() == 0:
            continue

        count = mask.sum()
        mae1_bin = np.abs(df_ensemble.loc[mask, 'pred_panstarrs'] - df_ensemble.loc[mask, 'true_temperature']).mean()
        mae2_bin = np.abs(df_ensemble.loc[mask, 'pred_unified'] - df_ensemble.loc[mask, 'true_temperature']).mean()
        mae_ens_bin = df_ensemble.loc[mask, 'abs_error'].mean()

        best_model = "PanSTARRS" if mae1_bin < mae2_bin else "Unified"
        if mae_ens_bin < min(mae1_bin, mae2_bin):
            best_model = "Ensemble"

        logger.info(f"{label:<15} {count:>8,} {mae1_bin:>11.1f} K {mae2_bin:>11.1f} K {mae_ens_bin:>11.1f} K {best_model:>12}")

    # Save ensemble predictions
    logger.info("\n" + "=" * 80)
    logger.info("SAVING ENSEMBLE PREDICTIONS")
    logger.info("=" * 80)

    ensemble_id = 'ensemble_panstarrs_unified'
    output_file = models_dir / f'{ensemble_id}_test_predictions.parquet'

    # Save ensemble predictions
    logger.info(f"\nSaving ensemble predictions: {output_file.name}")
    df_ensemble.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Save summary
    summary_file = models_dir / f'{ensemble_id}_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("ENSEMBLE MODEL - PAN-STARRS BASIC vs UNIFIED COLOR-ONLY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Ensemble ID: {ensemble_id}\n")
        f.write(f"Method: Simple average of two models (predicted on same test set)\n\n")

        f.write(f"COMPONENT MODELS:\n")
        f.write(f"  Model 1: {MODEL1_NAME}\n")
        f.write(f"    ID: {MODEL1_ID}\n")
        f.write(f"    Features: 5 (g-r, r-i, i-z, B-V, gPSFMag)\n")
        f.write(f"    MAE: {mae1:.1f} K, RMSE: {rmse1:.1f} K, R²: {r2_1:.3f}\n")
        f.write(f"    Within 10%: {100*w10_1/n_test:.1f}%\n\n")

        f.write(f"  Model 2: {MODEL2_NAME}\n")
        f.write(f"    ID: {MODEL2_ID}\n")
        f.write(f"    Features: 20 engineered color features (NO magnitude)\n")
        f.write(f"    MAE: {mae2:.1f} K, RMSE: {rmse2:.1f} K, R²: {r2_2:.3f}\n")
        f.write(f"    Within 10%: {100*w10_2/n_test:.1f}%\n\n")

        f.write(f"ENSEMBLE PERFORMANCE:\n")
        f.write(f"  Test objects: {n_test:,}\n")
        f.write(f"  MAE:  {mae_ens:.1f} K\n")
        f.write(f"  RMSE: {rmse_ens:.1f} K\n")
        f.write(f"  R²:   {r2_ens:.3f}\n")
        f.write(f"  Within 10%: {100*w10_ens/n_test:.1f}%\n\n")

        if mae_ens < best_individual_mae:
            f.write(f"IMPROVEMENT:\n")
            f.write(f"  Ensemble is BETTER than both individual models\n")
            f.write(f"  MAE reduction: {mae_improvement:.1f}%\n\n")
        else:
            f.write(f"RESULT:\n")
            f.write(f"  Ensemble performance: {mae_improvement:+.1f}% vs best\n\n")

        f.write(f"PERFORMANCE BY TEMPERATURE RANGE:\n")
        f.write(f"  Range          Count  Pan-STARRS     Unified   Ensemble       Best\n")
        f.write(f"  {'-'*70}\n")
        for label in temp_labels:
            mask = df_ensemble['temp_bin'] == label
            if mask.sum() == 0:
                continue
            count = mask.sum()
            mae1_bin = np.abs(df_ensemble.loc[mask, 'pred_panstarrs'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae2_bin = np.abs(df_ensemble.loc[mask, 'pred_unified'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae_ens_bin = df_ensemble.loc[mask, 'abs_error'].mean()
            best_model = "PanSTARRS" if mae1_bin < mae2_bin else "Unified"
            if mae_ens_bin < min(mae1_bin, mae2_bin):
                best_model = "Ensemble"
            f.write(f"  {label:<12} {count:>8,} {mae1_bin:>9.1f} K {mae2_bin:>10.1f} K {mae_ens_bin:>10.1f} K {best_model:>12}\n")

        f.write(f"\nNOTES:\n")
        f.write(f"  - Models trained on different datasets (Pan-STARRS basic vs unified)\n")
        f.write(f"  - Both models predicted on same test set from unified data\n")
        f.write(f"  - Pan-STARRS model includes magnitude feature (may have bias)\n")
        f.write(f"  - Unified model uses only colors (physically correct)\n")
        f.write(f"  - Ensemble combines predictions via simple averaging\n")
        f.write(f"  - Goal: Reduce systematic bias while maintaining accuracy\n")

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
