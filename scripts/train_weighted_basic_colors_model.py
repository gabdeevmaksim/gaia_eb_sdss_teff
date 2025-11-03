"""
Train Random Forest model using basic colors with weighted distribution.

This script trains a temperature prediction model using:
- 4 basic color features: g-r, r-i, i-z, bp-rp
- Weighted training samples to match prediction distribution
- Standard Random Forest regressor

Usage:
    python scripts/train_weighted_basic_colors_model.py
    python scripts/train_weighted_basic_colors_model.py --n-estimators 500 --max-depth 25
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import polars as pl
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging

from src.config import get_config

# Configuration
config = get_config()
MISSING_VALUE = config.get('processing', 'missing_value')
RANDOM_STATE = config.get('ml', 'random_state')
TEST_SIZE = config.get('ml', 'test_size')

# Color features to use
COLOR_FEATURES = ['g_r_color', 'r_i_color', 'i_z_color', 'bp_rp']

# Setup logging
log_dir = Path(config.get_path('reports')) / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_dir / 'train_weighted_model.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load weighted training data."""
    logger.info("Loading weighted training data...")

    train_path = Path(config.get_path('processed')) / 'eb_unified_features_engineered_train_weighted.parquet'

    df = pl.read_parquet(train_path)
    logger.info(f"Loaded {len(df):,} training samples")

    # Filter out missing values
    mask = pl.lit(True)
    for feature in COLOR_FEATURES:
        mask = mask & (pl.col(feature) != MISSING_VALUE)
    mask = mask & (pl.col('teff_gspphot') != MISSING_VALUE)

    df_clean = df.filter(mask)
    logger.info(f"After removing missing values: {len(df_clean):,} samples")

    return df_clean


def prepare_features(df):
    """Prepare feature matrix and target."""
    logger.info("Preparing features and target...")

    X = df.select(COLOR_FEATURES).to_numpy()
    y = df['teff_gspphot'].to_numpy()
    weights = df['sample_weight'].to_numpy()

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Weights shape: {weights.shape}")

    # Log feature statistics
    for i, feature in enumerate(COLOR_FEATURES):
        logger.info(f"  {feature}: mean={np.average(X[:, i], weights=weights):.3f}, "
                   f"std={np.sqrt(np.average((X[:, i] - np.average(X[:, i], weights=weights))**2, weights=weights)):.3f}")

    logger.info(f"Target: mean={np.average(y, weights=weights):.1f} K, "
               f"std={np.sqrt(np.average((y - np.average(y, weights=weights))**2, weights=weights)):.1f} K")

    return X, y, weights


def train_model(X_train, y_train, train_weights, X_test, y_test, test_weights,
                n_estimators=300, max_depth=20, min_samples_leaf=4, min_samples_split=5):
    """Train Random Forest model with sample weights."""
    logger.info("="*70)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("="*70)

    logger.info(f"Hyperparameters:")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  min_samples_leaf: {min_samples_leaf}")
    logger.info(f"  min_samples_split: {min_samples_split}")
    logger.info(f"  random_state: {RANDOM_STATE}")

    logger.info(f"\nTraining with sample weights...")
    logger.info(f"  Effective training samples: {(train_weights.sum()**2 / (train_weights**2).sum()):.0f}")

    import time
    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train, sample_weight=train_weights)

    training_time = time.time() - start_time
    logger.info(f"\nTraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

    return model, training_time


def evaluate_model(model, X_train, y_train, train_weights, X_test, y_test, test_weights):
    """Evaluate model performance."""
    logger.info("="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)

    # Training set predictions
    y_train_pred = model.predict(X_train)

    # Weighted metrics for training set
    train_mae = np.average(np.abs(y_train - y_train_pred), weights=train_weights)
    train_rmse = np.sqrt(np.average((y_train - y_train_pred)**2, weights=train_weights))

    # R² with weights (manual calculation)
    ss_res = np.sum(train_weights * (y_train - y_train_pred)**2)
    ss_tot = np.sum(train_weights * (y_train - np.average(y_train, weights=train_weights))**2)
    train_r2 = 1 - (ss_res / ss_tot)

    logger.info(f"\nTraining Set (weighted):")
    logger.info(f"  MAE:  {train_mae:.1f} K")
    logger.info(f"  RMSE: {train_rmse:.1f} K")
    logger.info(f"  R²:   {train_r2:.3f}")

    # Test set predictions
    y_test_pred = model.predict(X_test)

    # Weighted metrics for test set
    test_mae = np.average(np.abs(y_test - y_test_pred), weights=test_weights)
    test_rmse = np.sqrt(np.average((y_test - y_test_pred)**2, weights=test_weights))

    ss_res_test = np.sum(test_weights * (y_test - y_test_pred)**2)
    ss_tot_test = np.sum(test_weights * (y_test - np.average(y_test, weights=test_weights))**2)
    test_r2 = 1 - (ss_res_test / ss_tot_test)

    logger.info(f"\nTest Set (weighted):")
    logger.info(f"  MAE:  {test_mae:.1f} K")
    logger.info(f"  RMSE: {test_rmse:.1f} K")
    logger.info(f"  R²:   {test_r2:.3f}")

    # Accuracy within thresholds
    errors_pct = np.abs(y_test - y_test_pred) / y_test * 100

    within_5 = np.sum((errors_pct < 5) * test_weights)
    within_10 = np.sum((errors_pct < 10) * test_weights)
    within_20 = np.sum((errors_pct < 20) * test_weights)
    total_weight = np.sum(test_weights)

    logger.info(f"\nAccuracy (weighted):")
    logger.info(f"  Within 5%:  {within_5:.0f} / {total_weight:.0f} ({100*within_5/total_weight:.1f}%)")
    logger.info(f"  Within 10%: {within_10:.0f} / {total_weight:.0f} ({100*within_10/total_weight:.1f}%)")
    logger.info(f"  Within 20%: {within_20:.0f} / {total_weight:.0f} ({100*within_20/total_weight:.1f}%)")

    # Feature importance
    logger.info(f"\nFeature Importance:")
    for feature, importance in zip(COLOR_FEATURES, model.feature_importances_):
        logger.info(f"  {feature:15s}: {importance:.4f}")

    metrics = {
        'train': {
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'r2': float(train_r2)
        },
        'test': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2),
            'within_5pct': float(within_5),
            'within_10pct': float(within_10),
            'within_20pct': float(within_20),
            'within_5pct_frac': float(within_5/total_weight),
            'within_10pct_frac': float(within_10/total_weight),
            'within_20pct_frac': float(within_20/total_weight)
        }
    }

    return y_test_pred, metrics


def save_model(model, metrics, training_time, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    """Save trained model and metadata."""
    logger.info("="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"rf_weighted_basic_colors_{timestamp}"

    models_dir = Path(config.get_path('models'))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / f"{model_id}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ Saved model: {model_path}")

    # Save metadata
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'model_type': 'RandomForestRegressor',
        'features': COLOR_FEATURES,
        'n_features': len(COLOR_FEATURES),
        'weighted_training': True,
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'performance_metrics': metrics,
        'training_time_seconds': training_time,
        'feature_importance': {
            feature: float(importance)
            for feature, importance in zip(COLOR_FEATURES, model.feature_importances_)
        }
    }

    metadata_path = models_dir / f"{model_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_path}")

    # Save summary
    summary_path = models_dir / f"{model_id}_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("RANDOM FOREST MODEL - WEIGHTED BASIC COLORS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Weighted Training: Yes (distribution-matched)\n\n")

        f.write("FEATURES:\n")
        for feature in COLOR_FEATURES:
            f.write(f"  - {feature}\n")

        f.write(f"\nHYPERPARAMETERS:\n")
        f.write(f"  n_estimators: {n_estimators}\n")
        f.write(f"  max_depth: {max_depth}\n")
        f.write(f"  min_samples_leaf: {min_samples_leaf}\n")
        f.write(f"  min_samples_split: {min_samples_split}\n")
        f.write(f"  random_state: {RANDOM_STATE}\n")

        f.write(f"\nPERFORMANCE METRICS:\n")
        f.write(f"  Training MAE:  {metrics['train']['mae']:.1f} K\n")
        f.write(f"  Training RMSE: {metrics['train']['rmse']:.1f} K\n")
        f.write(f"  Training R²:   {metrics['train']['r2']:.3f}\n\n")

        f.write(f"  Test MAE:  {metrics['test']['mae']:.1f} K\n")
        f.write(f"  Test RMSE: {metrics['test']['rmse']:.1f} K\n")
        f.write(f"  Test R²:   {metrics['test']['r2']:.3f}\n\n")

        f.write(f"ACCURACY:\n")
        f.write(f"  Within 5%:  {metrics['test']['within_5pct_frac']*100:.1f}%\n")
        f.write(f"  Within 10%: {metrics['test']['within_10pct_frac']*100:.1f}%\n")
        f.write(f"  Within 20%: {metrics['test']['within_20pct_frac']*100:.1f}%\n\n")

        f.write(f"FEATURE IMPORTANCE:\n")
        for feature, importance in zip(COLOR_FEATURES, model.feature_importances_):
            f.write(f"  {feature:15s}: {importance:.4f}\n")

        f.write(f"\nTraining time: {training_time:.1f} seconds\n")

    logger.info(f"✓ Saved summary: {summary_path}")

    return model_id


def main():
    parser = argparse.ArgumentParser(description='Train weighted Random Forest model with basic colors')
    parser.add_argument('--n-estimators', type=int, default=300, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=20, help='Maximum tree depth')
    parser.add_argument('--min-samples-leaf', type=int, default=4, help='Minimum samples per leaf')
    parser.add_argument('--min-samples-split', type=int, default=5, help='Minimum samples to split')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("WEIGHTED BASIC COLORS MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Features: {COLOR_FEATURES}")
    logger.info(f"Using distribution-matched sample weights\n")

    # Load data
    df = load_data()

    # Prepare features
    X, y, weights = prepare_features(df)

    # Train/test split (preserving weights)
    logger.info(f"\nSplitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(
        X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Test samples: {len(X_test):,}")
    logger.info(f"Effective training samples (weighted): {(train_weights.sum()**2 / (train_weights**2).sum()):.0f}")
    logger.info(f"Effective test samples (weighted): {(test_weights.sum()**2 / (test_weights**2).sum()):.0f}")

    # Train model
    model, training_time = train_model(
        X_train, y_train, train_weights,
        X_test, y_test, test_weights,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split
    )

    # Evaluate model
    y_test_pred, metrics = evaluate_model(
        model, X_train, y_train, train_weights,
        X_test, y_test, test_weights
    )

    # Save model
    model_id = save_model(
        model, metrics, training_time,
        args.n_estimators, args.max_depth,
        args.min_samples_leaf, args.min_samples_split
    )

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Test MAE: {metrics['test']['mae']:.1f} K")
    logger.info(f"Test RMSE: {metrics['test']['rmse']:.1f} K")
    logger.info(f"Test R²: {metrics['test']['r2']:.3f}")
    logger.info(f"Within 10%: {metrics['test']['within_10pct_frac']*100:.1f}%")


if __name__ == '__main__':
    main()
