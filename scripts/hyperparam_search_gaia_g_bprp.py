#!/usr/bin/env python3
"""
Hyperparameter search for Gaia G + BP-RP model using RandomizedSearchCV.

Searches over Random Forest hyperparameters to optimize temperature prediction.
Uses 5-fold cross-validation with MAE as the scoring metric.
Includes checkpointing to resume from interruptions.

Usage:
    python scripts/hyperparam_search_gaia_g_bprp.py [--n-jobs N]

Arguments:
    --n-jobs N    Number of CPU cores to use (default: 4)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
import pickle
import json
import os
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = ['phot_g_mean_mag', 'bp_rp']
TARGET = 'teff_gspphot'

# Hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Search parameters
N_ITER = 50  # Number of parameter settings sampled
CV_FOLDS = 5

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_10pct = (np.abs((y_pred - y_true) / y_true) <= 0.1).mean() * 100

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'within_10pct': within_10pct}

def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_checkpoint(checkpoint_path, results, tested_params):
    """Save checkpoint."""
    checkpoint_data = {
        'results': results,
        'tested_params': tested_params,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def generate_random_params(param_distributions, random_state):
    """Generate random parameter combination."""
    rng = np.random.RandomState(random_state)
    params = {}
    for param, values in param_distributions.items():
        params[param] = values[rng.randint(len(values))]
    return params

def params_to_hashable(params):
    """Convert parameters dict to hashable tuple for deduplication."""
    return tuple(sorted((k, str(v)) for k, v in params.items()))

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for Gaia G + BP-RP temperature model'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of CPU cores to use (default: 4)'
    )
    args = parser.parse_args()

    n_jobs = args.n_jobs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Force unbuffered output for logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80, flush=True)
    print("Hyperparameter Search: Gaia G + BP-RP Model (with Checkpointing)", flush=True)
    print("=" * 80, flush=True)
    print(f"Start time: {datetime.now()}", flush=True)
    print(f"Using {n_jobs} CPU cores", flush=True)

    # Setup checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent / 'models' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'hyperparam_search_gaia_g_bprp.pkl'

    # Load training data
    print("\n1. Loading training data...", flush=True)
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    df = pl.read_parquet(data_dir / 'gaia_g_bprp_train.parquet')
    print(f"   Loaded {len(df):,} training samples", flush=True)

    # Prepare features and target
    print("\n2. Preparing features and target...", flush=True)
    X = df.select(FEATURES).to_numpy()
    y = df[TARGET].to_numpy()

    print(f"   Features: {FEATURES}", flush=True)
    print(f"   Target: {TARGET}", flush=True)
    print(f"   Dataset shape: {X.shape}", flush=True)

    # Train/test split (same random state ensures consistent splits)
    print(f"\n3. Splitting data (test_size={TEST_SIZE})...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"   Training set: {len(X_train):,} samples", flush=True)
    print(f"   Test set: {len(X_test):,} samples", flush=True)

    # Check for checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        results = checkpoint['results']
        tested_params = checkpoint['tested_params']
        print(f"\n4. Resuming from checkpoint...", flush=True)
        print(f"   Found {len(results)} completed iterations", flush=True)
        print(f"   Checkpoint saved at: {checkpoint['timestamp']}", flush=True)
    else:
        results = []
        tested_params = set()
        print(f"\n4. Starting new hyperparameter search...", flush=True)

    print(f"   Total iterations: {N_ITER}", flush=True)
    print(f"   Remaining iterations: {N_ITER - len(results)}", flush=True)
    print(f"   Cross-validation folds: {CV_FOLDS}", flush=True)
    print(f"   Scoring metric: Negative MAE", flush=True)
    print(f"\n   Hyperparameter search space:", flush=True)
    for param, values in param_distributions.items():
        print(f"     {param}: {values}", flush=True)

    # Create scorer (negative MAE since we want to minimize)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Run randomized search with checkpointing
    print(f"\n5. Running randomized search...", flush=True)
    print(f"   Checkpointing enabled - progress will be saved after each iteration", flush=True)

    iteration_offset = len(results)
    for i in range(iteration_offset, N_ITER):
        # Generate random parameters (with unique random state per iteration)
        max_attempts = 100
        for attempt in range(max_attempts):
            params = generate_random_params(param_distributions, RANDOM_STATE + i * 1000 + attempt)
            params_hash = params_to_hashable(params)
            if params_hash not in tested_params:
                tested_params.add(params_hash)
                break
        else:
            print(f"   WARNING: Could not find unique parameters after {max_attempts} attempts", flush=True)
            continue

        print(f"\n   Iteration {i+1}/{N_ITER}", flush=True)
        print(f"   Parameters: {params}", flush=True)

        try:
            # Create model with current parameters
            rf = RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                **params
            )

            # Perform cross-validation
            cv_scores = cross_val_score(
                rf, X_train, y_train,
                cv=CV_FOLDS,
                scoring=mae_scorer,
                n_jobs=1,  # Avoid nested parallelism
                verbose=0
            )

            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            print(f"   CV MAE: {-mean_cv_score:.1f} ± {std_cv_score:.1f} K", flush=True)

            # Store results
            results.append({
                'params': params,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores.tolist()
            })

            # Save checkpoint after each iteration
            save_checkpoint(checkpoint_path, results, tested_params)
            print(f"   Checkpoint saved", flush=True)

        except Exception as e:
            print(f"   ERROR in iteration {i+1}: {e}", flush=True)
            continue

    print("\n   Search complete!", flush=True)

    # Find best parameters
    print(f"\n6. Finding best hyperparameters...", flush=True)
    best_result = min(results, key=lambda x: x['mean_cv_score'])
    best_params = best_result['params']
    best_cv_mae = -best_result['mean_cv_score']

    print(f"   Best hyperparameters found:", flush=True)
    for param, value in sorted(best_params.items()):
        print(f"   {param}: {value}", flush=True)

    print(f"\n   Best CV MAE: {best_cv_mae:.1f} K", flush=True)

    # Train final model on full training set with best parameters
    print(f"\n7. Training final model with best parameters...", flush=True)
    best_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        **best_params
    )
    best_model.fit(X_train, y_train)

    # Evaluate best model on test set
    print(f"\n8. Evaluating best model on test set...", flush=True)
    y_test_pred = best_model.predict(X_test)

    test_metrics = calculate_metrics(y_test, y_test_pred)

    print(f"   MAE:  {test_metrics['mae']:.1f} K", flush=True)
    print(f"   RMSE: {test_metrics['rmse']:.1f} K", flush=True)
    print(f"   R²:   {test_metrics['r2']:.3f}", flush=True)
    print(f"   Within 10%: {test_metrics['within_10pct']:.1f}%", flush=True)

    # Compare to default model
    print(f"\n9. Comparison to default hyperparameters:", flush=True)
    print(f"   Default model (n_estimators=300, max_depth=20): MAE=573.5 K", flush=True)
    print(f"   Optimized model: MAE={test_metrics['mae']:.1f} K", flush=True)
    improvement = ((573.5 - test_metrics['mae']) / 573.5) * 100
    print(f"   Improvement: {improvement:.2f}%", flush=True)

    # Save best model
    print(f"\n10. Saving best model...", flush=True)
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    model_id = f'rf_gaia_g_bprp_optimized_{timestamp}'
    model_path = models_dir / f'{model_id}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"   Saved model: {model_path}", flush=True)

    # Save search results
    search_results = {
        'model_id': model_id,
        'timestamp': timestamp,
        'features': FEATURES,
        'target': TARGET,
        'best_params': best_params,
        'best_cv_mae': float(best_cv_mae),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'n_iter': N_ITER,
        'cv_folds': CV_FOLDS,
        'search_space': {k: [str(v) for v in vals] for k, vals in param_distributions.items()},
        'all_results': [
            {
                'params': r['params'],
                'mean_cv_mae': float(-r['mean_cv_score']),
                'std_cv_score': float(r['std_cv_score'])
            }
            for r in results
        ]
    }

    results_path = models_dir / f'{model_id}_search_results.json'
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=2)
    print(f"   Saved search results: {results_path}", flush=True)

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
    print(f"   Saved test predictions: {pred_path}", flush=True)

    # Save metadata (for validation plots)
    feature_importance = dict(zip(FEATURES, best_model.feature_importances_))
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'features': FEATURES,
        'target': TARGET,
        'n_features': len(FEATURES),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'hyperparameters': best_params,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'feature_importance': {k: float(v) for k, v in feature_importance.items()}
    }

    metadata_path = models_dir / f'{model_id}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata: {metadata_path}", flush=True)

    # Save summary
    summary_path = models_dir / f'{model_id}_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_id}\n")
        f.write("=" * 80 + "\n\n")
        f.write("HYPERPARAMETER OPTIMIZATION (with Checkpointing)\n\n")
        f.write(f"Search iterations: {N_ITER}\n")
        f.write(f"Cross-validation folds: {CV_FOLDS}\n")
        f.write(f"CPU cores used: {n_jobs} (half of available)\n")
        f.write(f"Best CV MAE: {best_cv_mae:.1f} K\n\n")
        f.write("BEST HYPERPARAMETERS:\n")
        for param, value in sorted(best_params.items()):
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        f.write("TEST SET PERFORMANCE:\n")
        f.write(f"  MAE:  {test_metrics['mae']:.1f} K\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.1f} K\n")
        f.write(f"  R²:   {test_metrics['r2']:.3f}\n")
        f.write(f"  Within 10%: {test_metrics['within_10pct']:.1f}%\n\n")
        f.write("COMPARISON:\n")
        f.write(f"  Default model: MAE=573.5 K\n")
        f.write(f"  Optimized model: MAE={test_metrics['mae']:.1f} K\n")
        f.write(f"  Improvement: {improvement:.2f}%\n")

    print(f"   Saved summary: {summary_path}", flush=True)

    # Clean up checkpoint
    print(f"\n11. Cleaning up checkpoint...", flush=True)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"   Removed checkpoint file", flush=True)

    print("\n" + "=" * 80)
    print("Hyperparameter search complete!", flush=True)
    print("=" * 80)
    print(f"End time: {datetime.now()}", flush=True)
    print(f"\nBest model saved as: {model_id}", flush=True)
    print(f"Test MAE: {test_metrics['mae']:.1f} K", flush=True)
    print(f"\nCheckpoint was saved to: {checkpoint_path.parent}", flush=True)
    print(f"Resume capability: Script can be interrupted and will resume from last checkpoint", flush=True)

if __name__ == '__main__':
    main()
