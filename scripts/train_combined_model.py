#!/usr/bin/env python3
"""
Train Random Forest model on combined Pan-STARRS + 2MASS + Gaia colors.

This script trains a temperature prediction model using 8 color features spanning
optical to near-infrared wavelengths:
- Gaia: BP-RP
- Pan-STARRS: g-r, r-i, i-z
- Synthetic: B-V
- 2MASS: J-H, H-K, J-K

No magnitude features (color-only model to avoid distance bias).

Usage:
    # Train with default parameters
    nohup python scripts/train_combined_model.py > train_combined.log 2>&1 &

    # Custom hyperparameters
    python scripts/train_combined_model.py --n-estimators 500 --max-depth 25

Author: Claude Code
Date: 2025-10-16
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
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


def load_training_data(config):
    """Load combined Pan-STARRS + 2MASS training dataset."""
    input_file = config.get_path('processed') / 'combined_panstarrs_2mass_colors_training.parquet'

    if not input_file.exists():
        logger.error(f"Training data not found: {input_file}")
        logger.error("Please run: python scripts/create_combined_panstarrs_2mass_dataset.py")
        raise FileNotFoundError(f"Training data not found: {input_file}")

    logger.info(f"Loading combined training data: {input_file.name}")
    df = pd.read_parquet(input_file)

    logger.info(f"  Loaded {len(df):,} training objects")
    logger.info(f"  Columns: {len(df.columns)}")

    return df


def prepare_features_and_target(df):
    """
    Prepare feature matrix and target variable.

    Features: 8 color indices only (no magnitudes)
    Target: teff_gspphot

    Returns
    -------
    tuple
        (X, y, feature_names)
    """
    # Define color features (8 total)
    color_features = [
        'bp_rp',       # Gaia
        'g_r_color',   # Pan-STARRS
        'r_i_color',
        'i_z_color',
        'B_V_color',   # Synthetic
        'j_h_color',   # 2MASS
        'h_k_color',
        'j_k_color'
    ]

    logger.info(f"\nPreparing features:")
    logger.info(f"  Color features: {len(color_features)}")
    for i, feat in enumerate(color_features, 1):
        logger.info(f"    {i}. {feat}")

    # Create feature matrix
    X = df[color_features].copy()
    y = df['teff_gspphot'].copy()

    logger.info(f"\n  Feature matrix shape: {X.shape}")
    logger.info(f"  Target shape: {y.shape}")
    logger.info(f"  Target range: {y.min():.1f} - {y.max():.1f} K")
    logger.info(f"  Target mean: {y.mean():.1f} K")

    return X, y, color_features


def train_model(
    X_train, X_test, y_train, y_test,
    feature_names,
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=4,
    min_samples_split=5,
    random_state=42
):
    """
    Train Random Forest model.

    Returns
    -------
    dict
        Dictionary with model and performance metrics
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 70)

    # Train Random Forest
    logger.info(f"\nTraining Random Forest...")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  min_samples_leaf: {min_samples_leaf}")
    logger.info(f"  min_samples_split: {min_samples_split}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"  Training completed in {training_time:.1f} seconds")

    # Make predictions
    logger.info("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate error percentages
    test_error_pct = (np.abs(y_test - y_test_pred) / y_test * 100)

    within_5pct = (test_error_pct <= 5).sum()
    within_10pct = (test_error_pct <= 10).sum()
    within_20pct = (test_error_pct <= 20).sum()

    logger.info("\n" + "=" * 70)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 70)
    logger.info(f"\nTraining metrics:")
    logger.info(f"  MAE:  {train_mae:.1f} K")
    logger.info(f"  RMSE: {train_rmse:.1f} K")
    logger.info(f"  R²:   {train_r2:.3f}")

    logger.info(f"\nTest metrics:")
    logger.info(f"  MAE:  {test_mae:.1f} K")
    logger.info(f"  RMSE: {test_rmse:.1f} K")
    logger.info(f"  R²:   {test_r2:.3f}")
    logger.info(f"  Mean error: {test_error_pct.mean():.2f}%")
    logger.info(f"  Median error: {test_error_pct.median():.2f}%")

    logger.info(f"\nPrediction accuracy:")
    logger.info(f"  Within 5%:  {within_5pct:,} ({100*within_5pct/len(y_test):.1f}%)")
    logger.info(f"  Within 10%: {within_10pct:,} ({100*within_10pct/len(y_test):.1f}%)")
    logger.info(f"  Within 20%: {within_20pct:,} ({100*within_20pct/len(y_test):.1f}%)")

    # Feature importance
    logger.info(f"\nFeature importance:")
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importances.iterrows():
        logger.info(f"  {row['feature']:<15} {row['importance']:.4f}")

    return {
        'model': model,
        'feature_names': feature_names,
        'training_time': training_time,
        'train_metrics': {
            'mae': train_mae,
            'rmse': train_rmse,
            'r2': train_r2
        },
        'test_metrics': {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2,
            'mean_error_pct': test_error_pct.mean(),
            'median_error_pct': test_error_pct.median(),
            'within_5pct': int(within_5pct),
            'within_10pct': int(within_10pct),
            'within_20pct': int(within_20pct)
        },
        'feature_importances': feature_importances,
        'test_predictions': {
            'y_true': y_test,
            'y_pred': y_test_pred
        }
    }


def save_model_and_metadata(results, config, hyperparameters, dataset_info):
    """Save model, metadata, and predictions."""
    # Create timestamp and model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"rf_combined_colors_{timestamp}"

    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODEL AND METADATA")
    logger.info("=" * 70)
    logger.info(f"Model ID: {model_id}")

    models_dir = config.get_path('models')

    # Save model
    model_file = models_dir / f"{model_id}.pkl"
    joblib.dump(results['model'], model_file)
    logger.info(f"  Model saved: {model_file.name}")

    # Save test predictions
    pred_df = pd.DataFrame({
        'true_temperature': results['test_predictions']['y_true'],
        'predicted_temperature': results['test_predictions']['y_pred'],
        'residual': results['test_predictions']['y_pred'] - results['test_predictions']['y_true']
    })
    pred_file = models_dir / f"{model_id}_test_predictions.parquet"
    pred_df.to_parquet(pred_file, index=False)
    logger.info(f"  Test predictions saved: {pred_file.name}")

    # Create metadata
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'model_type': 'RandomForestRegressor',
        'dataset_type': 'combined_panstarrs_2mass_colors',
        'training_samples': dataset_info['train_samples'],
        'test_samples': dataset_info['test_samples'],
        'n_features': len(results['feature_names']),
        'features': results['feature_names'],
        'hyperparameters': hyperparameters,
        'performance_metrics': {
            'train': results['train_metrics'],
            'test': results['test_metrics']
        },
        'training_time_seconds': results['training_time'],
        'feature_importance': {
            row['feature']: row['importance']
            for _, row in results['feature_importances'].iterrows()
        }
    }

    # Save metadata
    metadata_file = models_dir / f"{model_id}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=float)
    logger.info(f"  Metadata saved: {metadata_file.name}")

    # Save summary
    summary_file = models_dir / f"{model_id}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"COMBINED PAN-STARRS + 2MASS + GAIA COLORS MODEL\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write(f"DATASET:\n")
        f.write(f"  Training samples: {dataset_info['train_samples']:,}\n")
        f.write(f"  Test samples: {dataset_info['test_samples']:,}\n")
        f.write(f"  Features: {len(results['feature_names'])} color indices\n\n")

        f.write(f"COLOR FEATURES (8 total):\n")
        f.write(f"  Optical:\n")
        f.write(f"    - bp_rp (Gaia BP-RP)\n")
        f.write(f"    - g_r_color, r_i_color, i_z_color (Pan-STARRS)\n")
        f.write(f"    - B_V_color (Synthetic)\n")
        f.write(f"  Near-Infrared:\n")
        f.write(f"    - j_h_color, h_k_color, j_k_color (2MASS)\n\n")

        f.write(f"PERFORMANCE METRICS:\n")
        f.write(f"  Test MAE:  {results['test_metrics']['mae']:.1f} K\n")
        f.write(f"  Test RMSE: {results['test_metrics']['rmse']:.1f} K\n")
        f.write(f"  Test R²:   {results['test_metrics']['r2']:.3f}\n")
        f.write(f"  Mean error: {results['test_metrics']['mean_error_pct']:.2f}%\n")
        f.write(f"  Median error: {results['test_metrics']['median_error_pct']:.2f}%\n\n")

        f.write(f"ACCURACY:\n")
        f.write(f"  Within 5%:  {results['test_metrics']['within_5pct']:,} ({100*results['test_metrics']['within_5pct']/dataset_info['test_samples']:.1f}%)\n")
        f.write(f"  Within 10%: {results['test_metrics']['within_10pct']:,} ({100*results['test_metrics']['within_10pct']/dataset_info['test_samples']:.1f}%)\n")
        f.write(f"  Within 20%: {results['test_metrics']['within_20pct']:,} ({100*results['test_metrics']['within_20pct']/dataset_info['test_samples']:.1f}%)\n\n")

        f.write(f"HYPERPARAMETERS:\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\nFEATURE IMPORTANCE:\n")
        for idx, row in results['feature_importances'].iterrows():
            f.write(f"  {row['feature']:<15} {row['importance']:.4f}\n")

        f.write(f"\nTraining time: {results['training_time']:.1f} seconds\n")

    logger.info(f"  Summary saved: {summary_file.name}")

    return model_id


def main():
    parser = argparse.ArgumentParser(
        description='Train RF model on combined Pan-STARRS + 2MASS + Gaia colors',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n-estimators', type=int, default=300, help='Number of trees (default: 300)')
    parser.add_argument('--max-depth', type=int, default=20, help='Maximum tree depth (default: 20)')
    parser.add_argument('--min-samples-leaf', type=int, default=4, help='Minimum samples per leaf (default: 4)')
    parser.add_argument('--min-samples-split', type=int, default=5, help='Minimum samples to split (default: 5)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    config = get_config()

    logger.info("=" * 80)
    logger.info("TRAIN COMBINED PAN-STARRS + 2MASS + GAIA MODEL")
    logger.info("=" * 80)

    # Load data
    df_train = load_training_data(config)

    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df_train)

    # Train-test split
    logger.info(f"\nSplitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    logger.info(f"  Training set: {len(X_train):,} samples")
    logger.info(f"  Test set: {len(X_test):,} samples")

    # Train model
    results = train_model(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )

    # Hyperparameters for metadata
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
        'min_samples_split': args.min_samples_split,
        'random_state': args.random_state,
        'n_jobs': -1
    }

    # Save model and metadata
    dataset_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    model_id = save_model_and_metadata(results, config, hyperparameters, dataset_info)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nModel ID: {model_id}")
    logger.info(f"Test MAE: {results['test_metrics']['mae']:.1f} K")
    logger.info(f"Test R²: {results['test_metrics']['r2']:.3f}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Create validation plots")
    logger.info(f"  2. Predict temperatures for objects without Gaia Teff")


if __name__ == '__main__':
    main()
