#!/usr/bin/env python3
"""
Train Random Forest temperature regression model using Gaia + 2MASS IR features.

Features:
- phot_g_mean_mag (Gaia G-band magnitude)
- bp_rp (Gaia BP-RP color)
- j_h_color (2MASS J-H color)
- h_k_color (2MASS H-K color)
- j_k_color (2MASS J-K color)

Target: teff_gspphot (Gaia GSP-Phot effective temperature)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = ['phot_g_mean_mag', 'bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
TARGET = 'teff_gspphot'

# Model hyperparameters
N_ESTIMATORS = 300
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 4
MIN_SAMPLES_SPLIT = 5
N_JOBS = -1

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Calculate percentage within 10%
    within_10pct = np.abs((y_pred - y_true) / y_true) <= 0.1
    pct_within_10 = within_10pct.mean() * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'within_10pct': pct_within_10
    }

def calculate_performance_by_temp(y_true, y_pred, n_bins=5):
    """Calculate performance metrics by temperature range."""
    # Define temperature bins
    bins = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    bins[0] = y_true.min() - 1  # Ensure all values are included
    bins[-1] = y_true.max() + 1

    bin_stats = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() == 0:
            continue

        y_true_bin = y_true[mask]
        y_pred_bin = y_pred[mask]

        metrics = calculate_metrics(y_true_bin, y_pred_bin)

        bin_stats.append({
            'temp_min': bins[i],
            'temp_max': bins[i+1],
            'temp_center': (bins[i] + bins[i+1]) / 2,
            'n_samples': mask.sum(),
            **metrics
        })

    return bin_stats

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f'rf_gaia_2mass_ir_{timestamp}'

    print("=" * 80)
    print("Training Random Forest: Gaia G + BP-RP + 2MASS IR")
    print("=" * 80)
    print(f"Model ID: {model_id}")
    print(f"Start time: {datetime.now()}")

    # Load training data
    print("\n1. Loading training data...")
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    df = pl.read_parquet(data_dir / 'gaia_2mass_ir_train.parquet')
    print(f"   Loaded {len(df):,} training samples")

    # Prepare features and target
    print("\n2. Preparing features and target...")
    X = df.select(FEATURES).to_numpy()
    y = df[TARGET].to_numpy()

    print(f"   Features ({len(FEATURES)}): {FEATURES}")
    print(f"   Target: {TARGET}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    # Feature statistics
    print("\n   Feature statistics:")
    for i, feat in enumerate(FEATURES):
        print(f"     {feat:20s}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}, "
              f"min={X[:, i].min():.3f}, max={X[:, i].max():.3f}")

    print(f"\n   Target statistics:")
    print(f"     {TARGET:20s}: mean={y.mean():.1f}, std={y.std():.1f}, "
          f"min={y.min():.1f}, max={y.max():.1f}")

    # Train/test split
    print(f"\n3. Splitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")

    # Train model
    print(f"\n4. Training Random Forest...")
    print(f"   Hyperparameters:")
    print(f"     n_estimators: {N_ESTIMATORS}")
    print(f"     max_depth: {MAX_DEPTH}")
    print(f"     min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"     min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"     random_state: {RANDOM_STATE}")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1
    )

    model.fit(X_train, y_train)
    print("   Training complete!")

    # Evaluate on training set
    print("\n5. Evaluating on training set...")
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)

    print(f"   MAE:  {train_metrics['mae']:.1f} K")
    print(f"   RMSE: {train_metrics['rmse']:.1f} K")
    print(f"   R²:   {train_metrics['r2']:.3f}")
    print(f"   Within 10%: {train_metrics['within_10pct']:.1f}%")

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print(f"   MAE:  {test_metrics['mae']:.1f} K")
    print(f"   RMSE: {test_metrics['rmse']:.1f} K")
    print(f"   R²:   {test_metrics['r2']:.3f}")
    print(f"   Within 10%: {test_metrics['within_10pct']:.1f}%")

    # Feature importance
    print("\n7. Feature importance...")
    feature_importance = dict(zip(FEATURES, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    for feat, importance in sorted_features:
        print(f"   {feat:20s}: {importance:.4f} ({importance*100:.1f}%)")

    # Performance by temperature range
    print("\n8. Performance by temperature range...")
    bin_stats = calculate_performance_by_temp(y_test, y_test_pred, n_bins=5)

    for stat in bin_stats:
        print(f"   {stat['temp_min']:.0f}-{stat['temp_max']:.0f} K (n={stat['n_samples']:,}): "
              f"MAE={stat['mae']:.0f} K, RMSE={stat['rmse']:.0f} K, "
              f"R²={stat['r2']:.3f}, Within 10%={stat['within_10pct']:.1f}%")

    # Save model
    print("\n9. Saving model...")
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f'{model_id}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Saved model: {model_path}")

    # Save metadata
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'features': FEATURES,
        'target': TARGET,
        'n_features': len(FEATURES),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'hyperparameters': {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF,
            'min_samples_split': MIN_SAMPLES_SPLIT,
            'random_state': RANDOM_STATE
        },
        'train_metrics': {k: float(v) for k, v in train_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'performance_by_temp': [
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
             for k, v in stat.items()}
            for stat in bin_stats
        ]
    }

    metadata_path = models_dir / f'{model_id}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata: {metadata_path}")

    # Save test predictions
    test_pred_df = pl.DataFrame({
        'true_temperature': y_test,
        'predicted_temperature': y_test_pred,
        'residual': y_test_pred - y_test,
        'abs_error': np.abs(y_test_pred - y_test),
        'pct_error': np.abs((y_test_pred - y_test) / y_test) * 100
    })

    pred_path = models_dir / f'{model_id}_test_predictions.parquet'
    test_pred_df.write_parquet(pred_path)
    print(f"   Saved test predictions: {pred_path}")

    # Save summary
    summary_path = models_dir / f'{model_id}_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_id}\n")
        f.write("=" * 80 + "\n\n")
        f.write("FEATURES:\n")
        f.write(f"  - phot_g_mean_mag (Gaia G-band magnitude)\n")
        f.write(f"  - bp_rp (Gaia BP-RP color)\n")
        f.write(f"  - j_h_color (2MASS J-H color)\n")
        f.write(f"  - h_k_color (2MASS H-K color)\n")
        f.write(f"  - j_k_color (2MASS J-K color)\n\n")
        f.write("DATASET:\n")
        f.write(f"  Training samples: {len(X_train):,}\n")
        f.write(f"  Test samples: {len(X_test):,}\n\n")
        f.write("TEST SET PERFORMANCE:\n")
        f.write(f"  MAE:  {test_metrics['mae']:.1f} K\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.1f} K\n")
        f.write(f"  R²:   {test_metrics['r2']:.3f}\n")
        f.write(f"  Within 10%: {test_metrics['within_10pct']:.1f}%\n\n")
        f.write("FEATURE IMPORTANCE:\n")
        for feat, importance in sorted_features:
            f.write(f"  {feat:20s}: {importance:.4f} ({importance*100:.1f}%)\n")
        f.write("\n")
        f.write("PERFORMANCE BY TEMPERATURE RANGE:\n")
        for stat in bin_stats:
            f.write(f"  {stat['temp_min']:.0f}-{stat['temp_max']:.0f} K (n={stat['n_samples']:,}):\n")
            f.write(f"    MAE={stat['mae']:.0f} K, RMSE={stat['rmse']:.0f} K, ")
            f.write(f"R²={stat['r2']:.3f}, Within 10%={stat['within_10pct']:.1f}%\n")

    print(f"   Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"\nModel saved as: {model_id}")
    print(f"Test MAE: {test_metrics['mae']:.1f} K")
    print(f"Test R²: {test_metrics['r2']:.3f}")

if __name__ == '__main__':
    main()
