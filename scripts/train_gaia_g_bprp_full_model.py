#!/usr/bin/env python3
"""
Train Gaia G + BP-RP model on full dataset with optimized hyperparameters.

This script trains a Random Forest model using:
- Full training dataset: 1,275,924 objects (73% more than previous)
- Optimized hyperparameters from randomized search
- Features: phot_g_mean_mag, bp_rp
- Target: teff_gaia

Best hyperparameters (from optimization):
- n_estimators: 300
- max_depth: 30
- min_samples_split: 10
- min_samples_leaf: 2
- max_features: None
- bootstrap: False

Author: Claude Code
Date: 2025-10-22
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Within 10% accuracy
    pct_error = np.abs((y_true - y_pred) / y_true)
    within_10pct = 100 * np.sum(pct_error <= 0.1) / len(y_true)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'within_10pct': within_10pct
    }


def main():
    """Main training workflow."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Timestamp for model versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_ID = f'rf_gaia_g_bprp_full_{timestamp}'

    logger.info("="*80)
    logger.info("TRAIN GAIA G + BP-RP MODEL ON FULL DATASET")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"\nModel ID: {MODEL_ID}")

    # Step 1: Load training data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD TRAINING DATA")
    logger.info("="*80)

    train_file = processed_dir / 'gaia_g_bprp_full_train.parquet'
    logger.info(f"\nLoading: {train_file}")

    df = pd.read_parquet(train_file)
    logger.info(f"Dataset loaded: {len(df):,} objects")
    logger.info(f"Columns: {list(df.columns)}")

    # Step 2: Prepare features and target
    logger.info("\n" + "="*80)
    logger.info("STEP 2: PREPARE FEATURES AND TARGET")
    logger.info("="*80)

    FEATURE_COLS = ['phot_g_mean_mag', 'bp_rp']
    TARGET_COL = 'teff_gaia'

    logger.info(f"\nFeatures: {FEATURE_COLS}")
    logger.info(f"Target: {TARGET_COL}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    logger.info(f"\nFeature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")

    # Step 3: Train/test split
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAIN/TEST SPLIT")
    logger.info("="*80)

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    logger.info(f"\nTest size: {TEST_SIZE}")
    logger.info(f"Random state: {RANDOM_STATE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"\nTraining set: {len(X_train):,} samples")
    logger.info(f"Test set:     {len(X_test):,} samples")

    # Step 4: Configure model with optimized hyperparameters
    logger.info("\n" + "="*80)
    logger.info("STEP 4: CONFIGURE MODEL (OPTIMIZED HYPERPARAMETERS)")
    logger.info("="*80)

    # Best hyperparameters from randomized search
    HYPERPARAMETERS = {
        'n_estimators': 300,
        'max_depth': 30,
        'min_samples_split': 10,
        'min_samples_leaf': 2,
        'max_features': None,
        'bootstrap': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,  # Use all CPU cores
        'verbose': 1
    }

    logger.info("\nOptimized hyperparameters:")
    for key, value in HYPERPARAMETERS.items():
        if key not in ['random_state', 'n_jobs', 'verbose']:
            logger.info(f"  {key:20s}: {value}")

    model = RandomForestRegressor(**HYPERPARAMETERS)

    # Step 5: Train model
    logger.info("\n" + "="*80)
    logger.info("STEP 5: TRAIN MODEL")
    logger.info("="*80)

    logger.info(f"\nTraining Random Forest...")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Features: {X_train.shape[1]}")

    train_start = datetime.now()
    model.fit(X_train, y_train)
    train_end = datetime.now()
    train_duration = (train_end - train_start).total_seconds()

    logger.info(f"\nTraining completed in {train_duration:.1f} seconds ({train_duration/60:.1f} minutes)")

    # Step 6: Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("STEP 6: EVALUATE ON TEST SET")
    logger.info("="*80)

    logger.info("\nGenerating test predictions...")
    y_test_pred = model.predict(X_test)

    test_metrics = calculate_metrics(y_test, y_test_pred)

    logger.info("\nTest set performance:")
    logger.info(f"  MAE:  {test_metrics['mae']:.1f} K")
    logger.info(f"  RMSE: {test_metrics['rmse']:.1f} K")
    logger.info(f"  R²:   {test_metrics['r2']:.3f}")
    logger.info(f"  Within 10%: {test_metrics['within_10pct']:.1f}%")

    # Also evaluate on training set
    logger.info("\nEvaluating on training set...")
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)

    logger.info("\nTraining set performance:")
    logger.info(f"  MAE:  {train_metrics['mae']:.1f} K")
    logger.info(f"  RMSE: {train_metrics['rmse']:.1f} K")
    logger.info(f"  R²:   {train_metrics['r2']:.3f}")
    logger.info(f"  Within 10%: {train_metrics['within_10pct']:.1f}%")

    # Check for overfitting
    mae_diff = train_metrics['mae'] - test_metrics['mae']
    logger.info(f"\nOverfitting check:")
    logger.info(f"  Train MAE - Test MAE: {mae_diff:.1f} K")
    if abs(mae_diff) < 50:
        logger.info(f"  → Minimal overfitting (good generalization)")
    elif abs(mae_diff) < 100:
        logger.info(f"  → Moderate overfitting")
    else:
        logger.info(f"  → Significant overfitting (consider regularization)")

    # Step 7: Feature importance
    logger.info("\n" + "="*80)
    logger.info("STEP 7: FEATURE IMPORTANCE")
    logger.info("="*80)

    feature_importance = dict(zip(FEATURE_COLS, model.feature_importances_))

    logger.info("\nFeature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature:20s}: {importance:.3f} ({100*importance:.1f}%)")

    # Step 8: Save model
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SAVE MODEL")
    logger.info("="*80)

    model_path = models_dir / f'{MODEL_ID}.pkl'
    logger.info(f"\nSaving model to: {model_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    model_size = model_path.stat().st_size / 1e6
    logger.info(f"Model saved: {model_size:.1f} MB")

    # Step 9: Save metadata
    logger.info("\n" + "="*80)
    logger.info("STEP 9: SAVE METADATA")
    logger.info("="*80)

    metadata = {
        'model_id': MODEL_ID,
        'timestamp': timestamp,
        'training_data': 'gaia_g_bprp_full_train.parquet',
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'n_features': len(FEATURE_COLS),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'hyperparameters': {k: v for k, v in HYPERPARAMETERS.items() if k not in ['random_state', 'n_jobs', 'verbose']},
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'feature_importance': feature_importance,
        'training_duration_seconds': train_duration
    }

    metadata_path = models_dir / f'{MODEL_ID}_metadata.json'
    logger.info(f"\nSaving metadata to: {metadata_path}")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved")

    # Step 10: Save summary
    logger.info("\n" + "="*80)
    logger.info("STEP 10: SAVE SUMMARY")
    logger.info("="*80)

    summary_path = models_dir / f'{MODEL_ID}_SUMMARY.txt'
    logger.info(f"\nWriting summary to: {summary_path}")

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write("="*80 + "\n\n")

        f.write("TRAINING DATASET:\n")
        f.write(f"  Source: gaia_g_bprp_full_train.parquet\n")
        f.write(f"  Total samples: {len(df):,}\n")
        f.write(f"  Training samples: {len(X_train):,}\n")
        f.write(f"  Test samples: {len(X_test):,}\n")
        f.write(f"  Training duration: {train_duration:.1f}s ({train_duration/60:.1f}min)\n\n")

        f.write("FEATURES:\n")
        for feature in FEATURE_COLS:
            f.write(f"  - {feature}\n")
        f.write(f"\nTARGET:\n  - {TARGET_COL}\n\n")

        f.write("OPTIMIZED HYPERPARAMETERS:\n")
        for key, value in HYPERPARAMETERS.items():
            if key not in ['random_state', 'n_jobs', 'verbose']:
                f.write(f"  {key:20s}: {value}\n")
        f.write("\n")

        f.write("TEST SET PERFORMANCE:\n")
        f.write(f"  MAE:  {test_metrics['mae']:.1f} K\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.1f} K\n")
        f.write(f"  R²:   {test_metrics['r2']:.3f}\n")
        f.write(f"  Within 10%: {test_metrics['within_10pct']:.1f}%\n\n")

        f.write("TRAINING SET PERFORMANCE:\n")
        f.write(f"  MAE:  {train_metrics['mae']:.1f} K\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.1f} K\n")
        f.write(f"  R²:   {train_metrics['r2']:.3f}\n")
        f.write(f"  Within 10%: {train_metrics['within_10pct']:.1f}%\n\n")

        f.write("FEATURE IMPORTANCE:\n")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {feature:20s}: {importance:.3f} ({100*importance:.1f}%)\n")
        f.write("\n")

        f.write("COMPARISON WITH PREVIOUS MODEL (rf_gaia_g_bprp_optimized_20251022_091954):\n")
        f.write(f"  Previous training samples: 589,613\n")
        f.write(f"  New training samples:      {len(X_train):,}\n")
        f.write(f"  Increase: +{len(X_train) - 589613:,} samples (+{100*(len(X_train) - 589613)/589613:.1f}%)\n\n")
        f.write(f"  Previous test MAE:  659.5 K\n")
        f.write(f"  New test MAE:       {test_metrics['mae']:.1f} K\n")
        f.write(f"  Improvement: {659.5 - test_metrics['mae']:.1f} K ({100*(659.5 - test_metrics['mae'])/659.5:.1f}%)\n")

    logger.info("Summary saved")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info(f"\nModel files saved:")
    logger.info(f"  Model:    {model_path}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Summary:  {summary_path}")

    logger.info(f"\nTest Performance:")
    logger.info(f"  MAE:  {test_metrics['mae']:.1f} K")
    logger.info(f"  RMSE: {test_metrics['rmse']:.1f} K")
    logger.info(f"  R²:   {test_metrics['r2']:.3f}")
    logger.info(f"  Within 10%: {test_metrics['within_10pct']:.1f}%")

    logger.info(f"\nReady for validation and prediction!")


if __name__ == '__main__':
    main()
