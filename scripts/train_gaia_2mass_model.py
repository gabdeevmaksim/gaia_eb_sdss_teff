#!/usr/bin/env python3
"""
Train Random Forest model using only Gaia + 2MASS colors.

Features: bp_rp, j_h_color, h_k_color, j_k_color (4 colors, no magnitudes)
Target: Gaia GSP-Phot effective temperature
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Train Random Forest model on Gaia + 2MASS colors."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    logger.info("="*80)
    logger.info("TRAIN RANDOM FOREST MODEL: GAIA + 2MASS COLORS")
    logger.info("="*80)

    # Load training data
    logger.info("\n1. Loading training data...")
    input_file = processed_dir / 'gaia_2mass_colors_training.parquet'
    df = pd.read_parquet(input_file)
    logger.info(f"   Loaded: {len(df):,} objects")
    logger.info(f"   Columns: {df.columns.tolist()}")

    # Define features and target
    feature_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
    target_col = 'teff_gspphot'

    logger.info(f"\n2. Preparing features and target...")
    logger.info(f"   Features ({len(feature_cols)}): {feature_cols}")
    logger.info(f"   Target: {target_col}")

    X = df[feature_cols].values
    y = df[target_col].values

    logger.info(f"   Feature matrix: {X.shape}")
    logger.info(f"   Target vector: {y.shape}")
    logger.info(f"   Target range: {y.min():.1f} - {y.max():.1f} K")
    logger.info(f"   Target mean: {y.mean():.1f} K")

    # Split data
    test_size = 0.2
    random_state = 42

    logger.info(f"\n3. Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"   Training set: {len(X_train):,} samples")
    logger.info(f"   Test set: {len(X_test):,} samples")

    # Train Random Forest
    logger.info("\n" + "="*80)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("="*80)

    # Model parameters
    n_estimators = 300
    max_depth = 20
    min_samples_split = 5
    min_samples_leaf = 4
    n_jobs = -1

    logger.info(f"\nModel parameters:")
    logger.info(f"   n_estimators: {n_estimators}")
    logger.info(f"   max_depth: {max_depth}")
    logger.info(f"   min_samples_split: {min_samples_split}")
    logger.info(f"   min_samples_leaf: {min_samples_leaf}")
    logger.info(f"   n_jobs: {n_jobs} (use all cores)")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    logger.info("\nTraining model...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"   Training completed in {training_time:.1f} seconds")

    # Make predictions
    logger.info("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate model
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*80)

    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    logger.info("\nTraining metrics:")
    logger.info(f"   MAE:  {train_mae:.1f} K")
    logger.info(f"   RMSE: {train_rmse:.1f} K")
    logger.info(f"   R²:   {train_r2:.3f}")

    # Test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info("\nTest metrics:")
    logger.info(f"   MAE:  {test_mae:.1f} K")
    logger.info(f"   RMSE: {test_rmse:.1f} K")
    logger.info(f"   R²:   {test_r2:.3f}")

    # Prediction accuracy (percentage error)
    test_errors = np.abs(y_test - y_test_pred) / y_test * 100
    mean_pct_error = test_errors.mean()
    median_pct_error = np.median(test_errors)

    logger.info(f"   Mean error: {mean_pct_error:.2f}%")
    logger.info(f"   Median error: {median_pct_error:.2f}%")

    # Accuracy within thresholds
    within_5pct = (test_errors <= 5).sum()
    within_10pct = (test_errors <= 10).sum()
    within_20pct = (test_errors <= 20).sum()

    logger.info("\nPrediction accuracy:")
    logger.info(f"   Within 5%:  {within_5pct:,} ({100*within_5pct/len(y_test):.1f}%)")
    logger.info(f"   Within 10%: {within_10pct:,} ({100*within_10pct/len(y_test):.1f}%)")
    logger.info(f"   Within 20%: {within_20pct:,} ({100*within_20pct/len(y_test):.1f}%)")

    # Feature importance
    logger.info("\nFeature importance:")
    importances = model.feature_importances_
    for i, (feat, imp) in enumerate(zip(feature_cols, importances)):
        logger.info(f"   {i+1}. {feat}: {imp:.4f}")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f'rf_gaia_2mass_colors_{timestamp}'

    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL")
    logger.info("="*80)
    logger.info(f"Model ID: {model_id}")

    # Save model
    model_file = models_dir / f'{model_id}.pkl'
    joblib.dump(model, model_file)
    logger.info(f"   Model saved: {model_file.name}")

    # Save test predictions
    test_pred_df = pd.DataFrame({
        'true_temperature': y_test,
        'predicted_temperature': y_test_pred,
        'residual': y_test - y_test_pred,
        'percent_error': test_errors
    })
    test_pred_file = models_dir / f'{model_id}_test_predictions.parquet'
    test_pred_df.to_parquet(test_pred_file, index=False)
    logger.info(f"   Test predictions saved: {test_pred_file.name}")

    # Save metadata
    metadata = {
        'model_id': model_id,
        'created': timestamp,
        'dataset': 'gaia_2mass_colors_training.parquet',
        'n_objects': len(df),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': feature_cols,
        'n_features': len(feature_cols),
        'target': target_col,
        'model_type': 'RandomForestRegressor',
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        },
        'training_time_seconds': training_time,
        'metrics': {
            'train': {
                'mae': float(train_mae),
                'rmse': float(train_rmse),
                'r2': float(train_r2)
            },
            'test': {
                'mae': float(test_mae),
                'rmse': float(test_rmse),
                'r2': float(test_r2),
                'mean_pct_error': float(mean_pct_error),
                'median_pct_error': float(median_pct_error),
                'within_5pct': int(within_5pct),
                'within_10pct': int(within_10pct),
                'within_20pct': int(within_20pct)
            }
        },
        'feature_importance': {
            feat: float(imp) for feat, imp in zip(feature_cols, importances)
        }
    }

    metadata_file = models_dir / f'{model_id}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   Metadata saved: {metadata_file.name}")

    # Save summary
    summary = f"""RANDOM FOREST MODEL: GAIA + 2MASS COLORS
{'='*80}

Model ID: {model_id}
Created: {timestamp}

DATASET
-------
File: gaia_2mass_colors_training.parquet
Total objects: {len(df):,}
Training set: {len(X_train):,}
Test set: {len(X_test):,}

FEATURES (4 colors, no magnitudes)
----------------------------------
{chr(10).join(f'{i+1}. {feat}' for i, feat in enumerate(feature_cols))}

TARGET
------
{target_col} (Gaia GSP-Phot effective temperature)

MODEL PARAMETERS
----------------
Random Forest Regressor
  n_estimators: {n_estimators}
  max_depth: {max_depth}
  min_samples_split: {min_samples_split}
  min_samples_leaf: {min_samples_leaf}

PERFORMANCE
-----------
Training:
  MAE:  {train_mae:.1f} K
  RMSE: {train_rmse:.1f} K
  R²:   {train_r2:.3f}

Test:
  MAE:  {test_mae:.1f} K
  RMSE: {test_rmse:.1f} K
  R²:   {test_r2:.3f}
  Mean error: {mean_pct_error:.2f}%
  Median error: {median_pct_error:.2f}%

Prediction accuracy:
  Within 5%:  {within_5pct:,} ({100*within_5pct/len(y_test):.1f}%)
  Within 10%: {within_10pct:,} ({100*within_10pct/len(y_test):.1f}%)
  Within 20%: {within_20pct:,} ({100*within_20pct/len(y_test):.1f}%)

FEATURE IMPORTANCE
------------------
{chr(10).join(f'{i+1}. {feat}: {imp:.4f}' for i, (feat, imp) in enumerate(zip(feature_cols, importances)))}

TRAINING TIME
-------------
{training_time:.1f} seconds

FILES
-----
Model: {model_file.name}
Metadata: {metadata_file.name}
Test predictions: {test_pred_file.name}
"""

    summary_file = models_dir / f'{model_id}_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"   Summary saved: {summary_file.name}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nModel ID: {model_id}")
    logger.info(f"Test MAE: {test_mae:.1f} K")
    logger.info(f"Test R²: {test_r2:.3f}")

if __name__ == '__main__':
    main()
