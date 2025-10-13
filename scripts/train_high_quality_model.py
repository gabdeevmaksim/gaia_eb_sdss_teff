#!/usr/bin/env python3
"""
Train Random Forest model on high-quality dataset (flag 1 sources only).

This script trains a Random Forest regressor using only high-quality temperature
estimates (flag 1) and excludes magnitude features to avoid bias.

Usage:
    # Train with default parameters
    python scripts/train_high_quality_model.py

    # Train with custom parameters
    python scripts/train_high_quality_model.py --n-estimators 500 --max-depth 25

Author: Auto-generated
Date: 2025-10-13
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
from src.features.engineering import (
    engineer_all_features,
    select_best_features,
    get_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def train_high_quality_model(
    input_file: Path,
    n_estimators: int = 300,
    max_depth: int = 20,
    min_samples_leaf: int = 4,
    min_samples_split: int = 5,
    n_features_select: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Train Random Forest model on high-quality dataset.
    
    Parameters
    ----------
    input_file : Path
        Path to high-quality dataset
    n_estimators : int, default 300
        Number of trees in the forest
    max_depth : int, default 20
        Maximum depth of trees
    min_samples_leaf : int, default 4
        Minimum samples per leaf
    min_samples_split : int, default 5
        Minimum samples to split a node
    n_features_select : int, default 20
        Number of features to select
    test_size : float, default 0.2
        Fraction of data for testing
    random_state : int, default 42
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Training results and metadata
    """
    logger = logging.getLogger(__name__)
    config = get_config()
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"rf_temperature_regressor_high_quality_{timestamp}"
    
    logger.info("=" * 70)
    logger.info("TRAIN HIGH-QUALITY RANDOM FOREST MODEL")
    logger.info("=" * 70)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Input file: {input_file}")
    logger.info("")
    
    # Load high-quality dataset
    logger.info("Loading high-quality dataset...")
    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Define color columns (exclude magnitude and ID columns)
    exclude_cols = ['original_ext_source_id', 'source_id', 'teff_gspphot']
    color_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"  Color features: {color_cols}")
    
    # Feature engineering (excluding magnitude features)
    logger.info("\nEngineering features...")
    logger.info("  Creating polynomial features (degree 3)...")
    logger.info("  Creating interaction features...")
    logger.info("  Creating log features...")
    logger.info("  Creating temperature-dependent features...")
    logger.info("  Skipping magnitude features (avoiding bias)...")
    
    df_features = engineer_all_features(
        df,
        color_cols=color_cols,
        mag_cols=None,  # No magnitude features
        include_polynomials=True,
        include_interactions=True,
        include_log=True,
        include_temp_dependent=True,
        include_mag_features=False  # Skip magnitude features
    )
    
    logger.info(f"  Features created: {df_features.shape[1]}")
    
    # Prepare features and target
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    X = df_features[feature_cols]
    y = df_features['teff_gspphot']
    
    # Clean features: replace inf and NaN with 0
    logger.info("  Cleaning features (replacing inf/NaN with 0)...")
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    logger.info(f"  Feature matrix shape: {X.shape}")
    logger.info(f"  Target range: {y.min():.1f} - {y.max():.1f} K")
    
    # Feature selection
    logger.info(f"\nSelecting {n_features_select} best features...")
    X_selected, selector = select_best_features(X, y, k=n_features_select)
    selected_features = X_selected.columns.tolist()
    
    logger.info(f"  Selected features: {selected_features}")
    
    # Train-test split
    logger.info(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"  Training set: {len(X_train):,} samples")
    logger.info(f"  Test set: {len(X_test):,} samples")
    
    # Train model
    logger.info("\nTraining Random Forest model...")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  min_samples_leaf: {min_samples_leaf}")
    logger.info(f"  min_samples_split: {min_samples_split}")
    logger.info("  Training in progress... (this may take several minutes)")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1   # Show training progress
    )
    
    # Train with progress tracking
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
    
    # Calculate relative error
    relative_error = (test_mae / y_test.mean()) * 100
    
    logger.info("\n" + "=" * 70)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 70)
    logger.info(f"Training metrics:")
    logger.info(f"  MAE:  {train_mae:.1f} K")
    logger.info(f"  RMSE: {train_rmse:.1f} K")
    logger.info(f"  R²:   {train_r2:.3f}")
    logger.info(f"\nTest metrics:")
    logger.info(f"  MAE:  {test_mae:.1f} K")
    logger.info(f"  RMSE: {test_rmse:.1f} K")
    logger.info(f"  R²:   {test_r2:.3f}")
    logger.info(f"  Relative error: {relative_error:.2f}%")
    
    # Feature importance
    logger.info(f"\nTop 10 most important features:")
    feature_importance = get_feature_importance(model, selected_features, top_n=10)
    for _, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and metadata
    logger.info(f"\nSaving model and metadata...")
    models_dir = config.get_path('models', ensure_exists=True)
    
    # Save model
    model_file = models_dir / f"{model_id}.pkl"
    joblib.dump(model, model_file)
    logger.info(f"  Model saved: {model_file}")
    
    # Save feature selector
    selector_file = models_dir / f"{model_id}_selector.pkl"
    joblib.dump(selector, selector_file)
    logger.info(f"  Selector saved: {selector_file}")
    
    # Save test predictions
    test_predictions = pd.DataFrame({
        'true_temperature': y_test,
        'predicted_temperature': y_test_pred,
        'residual': y_test - y_test_pred
    })
    predictions_file = models_dir / f"{model_id}_test_predictions.parquet"
    test_predictions.to_parquet(predictions_file, index=False)
    logger.info(f"  Test predictions saved: {predictions_file}")
    
    # Create metadata
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'model_type': 'RandomForestRegressor',
        'dataset': 'high_quality_flag_1_only',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            'n_jobs': -1
        },
        'performance_metrics': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'relative_error_percent': float(relative_error)
        },
        'feature_engineering': {
            'color_features': color_cols,
            'polynomial_features': True,
            'interaction_features': True,
            'log_features': True,
            'temperature_dependent_features': True,
            'magnitude_features': False  # Excluded to avoid bias
        },
        'data_info': {
            'input_file': str(input_file),
            'total_samples': len(df),
            'temperature_range': [float(y.min()), float(y.max())],
            'feature_selection_method': 'SelectKBest_f_regression',
            'n_features_selected': n_features_select
        },
        'training_info': {
            'training_time_seconds': training_time,
            'test_size': test_size
        }
    }
    
    # Save metadata
    metadata_file = models_dir / f"{model_id}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved: {metadata_file}")
    
    # Create summary file
    summary_file = models_dir / f"{model_id}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"RANDOM FOREST TEMPERATURE REGRESSOR - HIGH QUALITY MODEL\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: High-quality sources only (flag 1)\n")
        f.write(f"Training samples: {len(X_train):,}\n")
        f.write(f"Test samples: {len(X_test):,}\n")
        f.write(f"Features: {len(selected_features)}\n\n")
        f.write(f"PERFORMANCE METRICS:\n")
        f.write(f"  Test MAE:  {test_mae:.1f} K\n")
        f.write(f"  Test RMSE: {test_rmse:.1f} K\n")
        f.write(f"  Test R²:   {test_r2:.3f}\n")
        f.write(f"  Relative error: {relative_error:.2f}%\n\n")
        f.write(f"HYPERPARAMETERS:\n")
        f.write(f"  n_estimators: {n_estimators}\n")
        f.write(f"  max_depth: {max_depth}\n")
        f.write(f"  min_samples_leaf: {min_samples_leaf}\n")
        f.write(f"  min_samples_split: {min_samples_split}\n\n")
        f.write(f"SELECTED FEATURES:\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"  {i:2d}. {feat}\n")
        f.write(f"\nTraining time: {training_time:.1f} seconds\n")
    
    logger.info(f"  Summary saved: {summary_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    return {
        'model_id': model_id,
        'model': model,
        'selector': selector,
        'metadata': metadata,
        'test_metrics': {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2,
            'relative_error': relative_error
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest model on high-quality dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input high-quality dataset (default: ml_training_data_high_quality.parquet)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=300,
        help='Number of trees (default: 300)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum tree depth (default: 20)'
    )
    
    parser.add_argument(
        '--min-samples-leaf',
        type=int,
        default=4,
        help='Minimum samples per leaf (default: 4)'
    )
    
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=5,
        help='Minimum samples to split (default: 5)'
    )
    
    parser.add_argument(
        '--n-features',
        type=int,
        default=20,
        help='Number of features to select (default: 20)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = config.project_root / 'data' / 'processed' / 'ml_training_data_high_quality.parquet'
    
    # Train model
    results = train_high_quality_model(
        input_file=input_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        n_features_select=args.n_features,
        test_size=args.test_size
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"\nModel training completed!")
    logger.info(f"Model ID: {results['model_id']}")
    logger.info(f"Test MAE: {results['test_metrics']['mae']:.1f} K")
    logger.info(f"Test R²: {results['test_metrics']['r2']:.3f}")


if __name__ == '__main__':
    main()
