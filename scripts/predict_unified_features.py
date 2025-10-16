#!/usr/bin/env python3
"""
Predict temperatures using unified feature dataset.

This script predicts temperatures using pre-engineered features,
ensuring perfect consistency with training data.

Key advantages:
- Features already engineered (super fast!)
- No feature engineering errors possible
- Guaranteed consistency with training

Usage:
    # Predict using latest unified model
    python scripts/predict_unified_features.py --model-type engineered

    # Specify exact model
    python scripts/predict_unified_features.py --model models/rf_unified_engineered_20251015_173000.pkl

Author: Claude Code
Date: 2025-10-15
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def load_prediction_data(config, model_type: str) -> pd.DataFrame:
    """Load pre-engineered prediction dataset."""
    logger = logging.getLogger(__name__)

    input_file = config.get_path('processed') / f'eb_unified_features_{model_type}_predict.parquet'

    if not input_file.exists():
        logger.error(f"Prediction data not found: {input_file}")
        logger.error(f"Please run: python scripts/create_unified_feature_dataset.py --model-type {model_type}")
        raise FileNotFoundError(f"Prediction data not found: {input_file}")

    logger.info(f"Loading prediction data: {input_file.name}")
    df = pd.read_parquet(input_file)

    logger.info(f"  Loaded {len(df):,} objects for prediction")
    logger.info(f"  Columns: {len(df.columns)}")

    return df


def load_model_and_selector(model_file: Path) -> tuple:
    """Load trained model and feature selector."""
    logger = logging.getLogger(__name__)

    logger.info(f"\nLoading model: {model_file.name}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Load model
    model = joblib.load(model_file)
    logger.info(f"  Model loaded: {model.__class__.__name__}")

    # Load selector
    selector_file = model_file.parent / (model_file.stem + '_selector.pkl')

    if not selector_file.exists():
        raise FileNotFoundError(f"Selector file not found: {selector_file}")

    selector = joblib.load(selector_file)
    logger.info(f"  Selector loaded: {selector.__class__.__name__}")

    # Get selected feature names
    return model, selector


def predict_temperatures(
    df: pd.DataFrame,
    model,
    selector,
    model_file: Path
) -> pd.DataFrame:
    """
    Make temperature predictions using pre-engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction dataset with engineered features
    model : sklearn model
        Trained model
    selector : sklearn selector
        Feature selector
    model_file : Path
        Path to model file (for metadata)

    Returns
    -------
    pd.DataFrame
        Dataset with predictions
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("MAKING TEMPERATURE PREDICTIONS")
    logger.info("=" * 70)

    # Define columns to exclude from features
    exclude_cols = [
        'original_ext_source_id',
        'gaia_source_id',
        'teff_gspphot',
        'Te_gr',
        'Te_ri',
        'Te_iz'
    ]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"  Total features available: {len(feature_cols)}")

    # Create feature matrix
    X = df[feature_cols].copy()

    logger.info(f"  Feature matrix shape: {X.shape}")

    # Apply feature selection
    logger.info(f"\nApplying feature selector...")
    X_selected = selector.transform(X)

    selected_mask = selector.get_support()
    selected_features = [feat for feat, sel in zip(feature_cols, selected_mask) if sel]

    logger.info(f"  Selected {len(selected_features)} features")
    logger.info(f"  Selected features: {selected_features[:10]}...")

    # Make predictions
    logger.info(f"\nPredicting temperatures...")
    start_time = datetime.now()
    predictions = model.predict(X_selected)
    prediction_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"  Predictions completed in {prediction_time:.1f} seconds")
    logger.info(f"  Predicted temperature range: {predictions.min():.1f} - {predictions.max():.1f} K")
    logger.info(f"  Predicted temperature mean: {predictions.mean():.1f} K")
    logger.info(f"  Predicted temperature median: {np.median(predictions):.1f} K")

    # Add predictions to dataframe
    results = df.copy()
    results['teff_predicted'] = predictions
    results['model_id'] = model_file.stem
    results['prediction_method'] = 'rf_unified_features'

    logger.info(f"\n  Results dataframe shape: {results.shape}")

    return results


def save_predictions(
    results: pd.DataFrame,
    model_file: Path,
    config
):
    """Save predictions and summary."""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("SAVING PREDICTIONS")
    logger.info("=" * 70)

    processed_dir = config.get_path('processed')

    # Create output filename
    model_id = model_file.stem
    output_file = processed_dir / f'predictions_{model_id}.parquet'

    # Save predictions
    results.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file.name}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Objects: {len(results):,}")

    # Create summary
    summary_file = processed_dir / f'predictions_{model_id}_SUMMARY.txt'

    with open(summary_file, 'w') as f:
        f.write(f"TEMPERATURE PREDICTIONS - UNIFIED FEATURES\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"PREDICTIONS:\n")
        f.write(f"  Objects predicted: {len(results):,}\n")
        f.write(f"  Predicted temperature range: {results['teff_predicted'].min():.1f} - {results['teff_predicted'].max():.1f} K\n")
        f.write(f"  Predicted temperature mean: {results['teff_predicted'].mean():.1f} K\n")
        f.write(f"  Predicted temperature median: {results['teff_predicted'].median():.1f} K\n")
        f.write(f"  Predicted temperature std: {results['teff_predicted'].std():.1f} K\n\n")

        f.write(f"OUTPUT:\n")
        f.write(f"  File: {output_file.name}\n")
        f.write(f"  Size: {file_size_mb:.2f} MB\n")
        f.write(f"  Columns: {list(results.columns)}\n")

    logger.info(f"  Summary: {summary_file.name}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Predict temperatures using unified features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest unified model (auto-detect)
  python scripts/predict_unified_features.py --model-type engineered

  # Use specific model file
  python scripts/predict_unified_features.py --model models/rf_unified_engineered_20251015_173000.pkl
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (if not specified, uses latest)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['basic', 'engineered'],
        default='engineered',
        help='Type of features (default: engineered)'
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    config = get_config()

    logger.info("=" * 80)
    logger.info("PREDICT TEMPERATURES - UNIFIED FEATURES")
    logger.info("=" * 80)

    # Determine model file
    if args.model:
        model_file = Path(args.model)
    else:
        # Find latest unified model for this model type
        logger.info(f"Looking for latest {args.model_type} model...")
        models_dir = config.get_path('models')

        pattern = f'rf_unified_{args.model_type}_*.pkl'
        models = [f for f in models_dir.glob(pattern)
                 if not f.name.endswith('_selector.pkl')]

        if not models:
            raise FileNotFoundError(f"No unified {args.model_type} models found. Train one first!")

        model_file = max(models, key=lambda x: x.stat().st_mtime)
        logger.info(f"  Found: {model_file.name}")

    # Load prediction data
    df_predict = load_prediction_data(config, args.model_type)

    # Load model and selector
    model, selector = load_model_and_selector(model_file)

    # Make predictions
    results = predict_temperatures(df_predict, model, selector, model_file)

    # Save results
    output_file = save_predictions(results, model_file, config)

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nPredictions: {len(results):,} objects")
    logger.info(f"Output: {output_file}")
    logger.info(f"Temperature range: {results['teff_predicted'].min():.0f} - {results['teff_predicted'].max():.0f} K")


if __name__ == '__main__':
    main()
