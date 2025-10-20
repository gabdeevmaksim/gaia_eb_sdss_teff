#!/usr/bin/env python3
"""
Train Random Forest model on Gaia + 2MASS engineered features.

This script trains a temperature prediction model using engineered features
from Gaia BP-RP color and 2MASS near-infrared colors (J-H, H-K, J-K).

Features include:
- Polynomial features (degree 2, 3)
- Interaction features (color1 * color2)
- Log features
- Temperature-dependent features (hot/cool/mid regimes)

Usage:
    # Train with default parameters
    nohup python scripts/train_gaia_2mass_engineered_model.py > train_gaia_2mass_eng.log 2>&1 &

    # Custom hyperparameters
    python scripts/train_gaia_2mass_engineered_model.py --n-estimators 500 --max-depth 25

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
from sklearn.feature_selection import SelectKBest, f_regression

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_training_data(config):
    """Load engineered training dataset."""
    input_file = config.get_path('processed') / 'gaia_2mass_colors_engineered_train.parquet'

    if not input_file.exists():
        logger.error(f"Training data not found: {input_file}")
        logger.error("Please run: python scripts/create_gaia_2mass_engineered_features.py")
        raise FileNotFoundError(f"Training data not found: {input_file}")

    logger.info(f"Loading engineered training data: {input_file.name}")
    df = pd.read_parquet(input_file)

    logger.info(f"  Loaded {len(df):,} training objects")
    logger.info(f"  Total columns: {len(df.columns)}")

    return df


def prepare_features_and_target(df):
    """
    Prepare feature matrix and target variable.

    Excludes: source_id, teff_gspphot

    Returns
    -------
    tuple
        (X, y, feature_names)
    """
    # Define columns to exclude from features
    exclude_cols = ['source_id', 'teff_gspphot']

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"\nPreparing features:")
    logger.info(f"  Total columns: {len(df.columns)}")
    logger.info(f"  Feature columns: {len(feature_cols)}")
    logger.info(f"  Excluded columns: {exclude_cols}")

    # Create feature matrix
    X = df[feature_cols].copy()
    y = df['teff_gspphot'].copy()

    logger.info(f"  Feature matrix shape: {X.shape}")
    logger.info(f"  Target shape: {y.shape}")
    logger.info(f"  Target range: {y.min():.1f} - {y.max():.1f} K")
    logger.info(f"  Target mean: {y.mean():.1f} K")

    return X, y, feature_cols


def train_model(
    X_train, X_test, y_train, y_test,
    feature_names,
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=4,
    min_samples_split=5,
    n_features_select=30,
    random_state=42
):
    """
    Train Random Forest model with feature selection.

    Returns
    -------
    dict
        Dictionary with model, selector, and performance metrics
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 70)

    # Feature selection
    logger.info(f"\nSelecting {n_features_select} best features...")
    selector = SelectKBest(score_func=f_regression, k=n_features_select)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feat for feat, sel in zip(feature_names, selected_mask) if sel]

    logger.info(f"  Selected features:")
    for i, feat in enumerate(selected_features, 1):
        logger.info(f"    {i:2d}. {feat}")

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
    model.fit(X_train_selected, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"  Training completed in {training_time:.1f} seconds")

    # Make predictions
    logger.info("\nMaking predictions...")
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

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
    logger.info(f"\nTop 10 most important features:")
    feature_importances = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importances.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return {
        'model': model,
        'selector': selector,
        'selected_features': selected_features,
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
    """Save model, selector, metadata, and predictions."""
    # Create timestamp and model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"rf_gaia_2mass_engineered_{timestamp}"

    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODEL AND METADATA")
    logger.info("=" * 70)
    logger.info(f"Model ID: {model_id}")

    models_dir = config.get_path('models')

    # Save model
    model_file = models_dir / f"{model_id}.pkl"
    joblib.dump(results['model'], model_file)
    logger.info(f"  Model saved: {model_file.name}")

    # Save selector
    selector_file = models_dir / f"{model_id}_selector.pkl"
    joblib.dump(results['selector'], selector_file)
    logger.info(f"  Selector saved: {selector_file.name}")

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
        'dataset_type': 'gaia_2mass_engineered',
        'training_samples': dataset_info['train_samples'],
        'test_samples': dataset_info['test_samples'],
        'total_features': dataset_info['total_features'],
        'selected_features': results['selected_features'],
        'n_features_selected': len(results['selected_features']),
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
        f.write(f"GAIA + 2MASS ENGINEERED FEATURES MODEL\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write(f"DATASET:\n")
        f.write(f"  Training samples: {dataset_info['train_samples']:,}\n")
        f.write(f"  Test samples: {dataset_info['test_samples']:,}\n")
        f.write(f"  Total features: {dataset_info['total_features']}\n")
        f.write(f"  Selected features: {len(results['selected_features'])}\n\n")

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

        f.write(f"\nSELECTED FEATURES:\n")
        for i, feat in enumerate(results['selected_features'], 1):
            imp = results['feature_importances'][results['feature_importances']['feature'] == feat]['importance'].values[0]
            f.write(f"  {i:2d}. {feat:<40} {imp:.4f}\n")

        f.write(f"\nTraining time: {results['training_time']:.1f} seconds\n")

    logger.info(f"  Summary saved: {summary_file.name}")

    return model_id


def main():
    parser = argparse.ArgumentParser(
        description='Train RF model on Gaia + 2MASS engineered features',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n-estimators', type=int, default=300, help='Number of trees (default: 300)')
    parser.add_argument('--max-depth', type=int, default=20, help='Maximum tree depth (default: 20)')
    parser.add_argument('--min-samples-leaf', type=int, default=4, help='Minimum samples per leaf (default: 4)')
    parser.add_argument('--min-samples-split', type=int, default=5, help='Minimum samples to split (default: 5)')
    parser.add_argument('--n-features', type=int, default=30, help='Number of features to select (default: 30)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    config = get_config()

    logger.info("=" * 80)
    logger.info("TRAIN GAIA + 2MASS ENGINEERED FEATURES MODEL")
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
        random_state=args.random_state,
        n_features_select=args.n_features
    )

    # Hyperparameters for metadata
    hyperparameters = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
        'min_samples_split': args.min_samples_split,
        'random_state': args.random_state,
        'n_jobs': -1,
        'n_features_select': args.n_features
    }

    # Save model and metadata
    dataset_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(feature_names)
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
