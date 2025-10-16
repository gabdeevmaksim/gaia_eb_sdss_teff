#!/usr/bin/env python3
"""
Predict temperatures using the basic Random Forest model (no feature engineering).

This script loads the basic Random Forest model (trained on 5 simple features)
and uses it to predict temperatures for sources with Pan-STARRS photometry.

Features required:
- g_r_color (Pan-STARRS g-r color)
- r_i_color (Pan-STARRS r-i color)
- i_z_color (Pan-STARRS i-z color)
- B_V_color (Synthetic B-V from g-r transformation)
- gPSFMag (Pan-STARRS g-band magnitude)

Usage:
    # Predict temperatures using basic model
    python scripts/predict_temperatures_basic_model.py

    # Specify model and input files
    python scripts/predict_temperatures_basic_model.py --model models/rf_temperature_regressor_20251001_125556.pkl --input data/processed/eb_full_catalog_temperatures.parquet

    # Predict for specific subset (e.g., stars_types catalog)
    python scripts/predict_temperatures_basic_model.py --input data/processed/eb_2mass_photometry.parquet --output data/processed/basic_model_predictions.parquet

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


def prepare_basic_features(df: pd.DataFrame, missing_value: float = -999.0) -> pd.DataFrame:
    """
    Prepare the 5 basic features required by the basic Random Forest model.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with Pan-STARRS photometry
    missing_value : float
        Value used to represent missing data

    Returns
    -------
    pd.DataFrame
        Dataframe with only the 5 required features
    """
    logger = logging.getLogger(__name__)

    # Check required columns - accept both naming conventions
    color_cols_v1 = ['g_r', 'r_i', 'i_z']  # Original naming
    color_cols_v2 = ['g_r_color', 'r_i_color', 'i_z_color']  # Alternative naming

    # Determine which naming convention is used
    has_v1 = all(col in df.columns for col in color_cols_v1)
    has_v2 = all(col in df.columns for col in color_cols_v2)

    if not has_v1 and not has_v2:
        logger.error(f"Missing color columns. Need either {color_cols_v1} or {color_cols_v2}")
        raise ValueError(f"Input data must contain color columns")

    if 'gPSFMag' not in df.columns:
        logger.error(f"Missing gPSFMag column")
        raise ValueError(f"Input data must contain gPSFMag")

    # Create feature dataframe
    features = pd.DataFrame()

    # Color features - handle both naming conventions
    if has_v1:
        features['g_r_color'] = df['g_r']
        features['r_i_color'] = df['r_i']
        features['i_z_color'] = df['i_z']
        g_r_source = df['g_r']
    else:
        features['g_r_color'] = df['g_r_color']
        features['r_i_color'] = df['r_i_color']
        features['i_z_color'] = df['i_z_color']
        g_r_source = df['g_r_color']

    # Calculate synthetic B-V from g-r if not already present
    # Using Jester et al. (2005) transformation: B-V ≈ 0.98(g-r) + 0.22
    if 'B_V_color' in df.columns:
        features['B_V_color'] = df['B_V_color']
    else:
        features['B_V_color'] = 0.98 * g_r_source + 0.22

    # Magnitude feature
    features['gPSFMag'] = df['gPSFMag']

    logger.info(f"  Created 5 basic features: {list(features.columns)}")

    return features


def clean_features(X: pd.DataFrame, missing_value: float = -999.0) -> tuple:
    """
    Clean features by removing objects with missing values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    missing_value : float
        Value representing missing data

    Returns
    -------
    tuple
        (cleaned features, valid indices mask)
    """
    logger = logging.getLogger(__name__)

    # Create mask for valid objects (no missing values)
    valid_mask = (X != missing_value).all(axis=1)

    # Also check for NaN and Inf
    valid_mask &= ~X.isna().any(axis=1)
    valid_mask &= ~np.isinf(X).any(axis=1)

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"  Removing {n_invalid:,} objects with missing/invalid values")

    X_clean = X[valid_mask].copy()

    logger.info(f"  Valid objects: {len(X_clean):,} ({100*len(X_clean)/len(X):.1f}%)")

    return X_clean, valid_mask


def predict_temperatures_basic(
    model_file: Path,
    input_file: Path,
    output_file: Path,
    missing_value: float = -999.0
) -> pd.DataFrame:
    """
    Predict temperatures using the basic Random Forest model.

    Parameters
    ----------
    model_file : Path
        Path to trained basic model
    input_file : Path
        Path to dataset with Pan-STARRS photometry
    output_file : Path
        Path to save predictions
    missing_value : float
        Value representing missing data

    Returns
    -------
    pd.DataFrame
        Sources with predicted temperatures
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PREDICT TEMPERATURES - BASIC RANDOM FOREST MODEL")
    logger.info("=" * 70)
    logger.info(f"Model: {model_file.name}")
    logger.info(f"Input: {input_file.name}")
    logger.info("")

    # Load model
    logger.info("Loading trained model...")
    model = joblib.load(model_file)
    logger.info(f"  Model loaded: {model.__class__.__name__}")
    logger.info(f"  Trained features: 5 basic features (no feature engineering)")

    # Load data
    logger.info("\nLoading dataset...")
    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)[:10]}...")

    # Prepare basic features
    logger.info("\nPreparing basic features...")
    X = prepare_basic_features(df, missing_value=missing_value)

    # Clean features
    logger.info("\nCleaning features...")
    X_clean, valid_mask = clean_features(X, missing_value=missing_value)

    if len(X_clean) == 0:
        logger.error("No valid objects after cleaning!")
        return pd.DataFrame()

    logger.info(f"  Feature matrix shape: {X_clean.shape}")
    logger.info(f"  Feature summary:")
    for col in X_clean.columns:
        logger.info(f"    {col}: min={X_clean[col].min():.3f}, max={X_clean[col].max():.3f}, mean={X_clean[col].mean():.3f}")

    # Make predictions
    logger.info("\nMaking temperature predictions...")
    start_time = datetime.now()
    predictions = model.predict(X_clean)
    prediction_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"  Predictions completed in {prediction_time:.1f} seconds")
    logger.info(f"  Predicted temperature range: {predictions.min():.1f} - {predictions.max():.1f} K")
    logger.info(f"  Predicted temperature median: {np.median(predictions):.1f} K")
    logger.info(f"  Predicted temperature mean: {np.mean(predictions):.1f} K")

    # Create results dataframe - only for valid objects
    results = df[valid_mask].copy()
    results['teff_predicted'] = predictions

    # Calculate residuals if ground truth exists
    if 'teff_gspphot' in results.columns:
        # Filter for objects with valid ground truth
        has_truth = (results['teff_gspphot'] != missing_value) & results['teff_gspphot'].notna()

        if has_truth.sum() > 0:
            results.loc[has_truth, 'prediction_residual'] = (
                results.loc[has_truth, 'teff_gspphot'] - results.loc[has_truth, 'teff_predicted']
            )
            results.loc[has_truth, 'prediction_error_percent'] = (
                np.abs(results.loc[has_truth, 'prediction_residual']) /
                results.loc[has_truth, 'teff_gspphot']
            ) * 100

            # Calculate statistics
            mae = np.mean(np.abs(results.loc[has_truth, 'prediction_residual']))
            rmse = np.sqrt(np.mean(results.loc[has_truth, 'prediction_residual']**2))
            median_ae = np.median(np.abs(results.loc[has_truth, 'prediction_residual']))
            mean_error_percent = results.loc[has_truth, 'prediction_error_percent'].mean()
            median_error_percent = results.loc[has_truth, 'prediction_error_percent'].median()

            # Calculate R²
            ss_res = np.sum(results.loc[has_truth, 'prediction_residual']**2)
            ss_tot = np.sum((results.loc[has_truth, 'teff_gspphot'] - results.loc[has_truth, 'teff_gspphot'].mean())**2)
            r2 = 1 - (ss_res / ss_tot)

            logger.info("\n" + "=" * 70)
            logger.info("PREDICTION VALIDATION (vs. Gaia GSP-Phot)")
            logger.info("=" * 70)
            logger.info(f"Objects with ground truth: {has_truth.sum():,}")
            logger.info(f"MAE: {mae:.1f} K")
            logger.info(f"Median AE: {median_ae:.1f} K")
            logger.info(f"RMSE: {rmse:.1f} K")
            logger.info(f"R²: {r2:.3f}")
            logger.info(f"Mean error: {mean_error_percent:.2f}%")
            logger.info(f"Median error: {median_error_percent:.2f}%")
        else:
            logger.info("\nNo ground truth temperatures available for validation")
            mae = rmse = r2 = mean_error_percent = None
    else:
        logger.info("\nNo ground truth column (teff_gspphot) in input data - skipping validation")
        mae = rmse = r2 = mean_error_percent = None

    # Save results
    logger.info(f"\nSaving predictions to {output_file}...")
    results.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Objects: {len(results):,}")
    logger.info(f"  Columns: {list(results.columns)}")

    # Create summary
    summary_file = output_file.parent / f"{output_file.stem}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"BASIC MODEL TEMPERATURE PREDICTIONS\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Model: {model_file.name}\n")
        f.write(f"Model type: Basic Random Forest (5 features, no feature engineering)\n")
        f.write(f"Input file: {input_file.name}\n")
        f.write(f"Output file: {output_file.name}\n\n")

        f.write(f"FEATURES USED:\n")
        f.write(f"  1. g_r_color (Pan-STARRS g-r color)\n")
        f.write(f"  2. r_i_color (Pan-STARRS r-i color)\n")
        f.write(f"  3. i_z_color (Pan-STARRS i-z color)\n")
        f.write(f"  4. B_V_color (Synthetic B-V from g-r)\n")
        f.write(f"  5. gPSFMag (Pan-STARRS g-band magnitude)\n\n")

        f.write(f"PREDICTION RESULTS:\n")
        f.write(f"  Total objects in input: {len(df):,}\n")
        f.write(f"  Valid objects (with complete photometry): {len(results):,}\n")
        f.write(f"  Success rate: {100*len(results)/len(df):.1f}%\n")
        f.write(f"  Prediction time: {prediction_time:.1f} seconds\n\n")

        f.write(f"PREDICTED TEMPERATURE DISTRIBUTION:\n")
        f.write(f"  Minimum: {predictions.min():.1f} K\n")
        f.write(f"  Maximum: {predictions.max():.1f} K\n")
        f.write(f"  Mean: {np.mean(predictions):.1f} K\n")
        f.write(f"  Median: {np.median(predictions):.1f} K\n")
        f.write(f"  Std dev: {np.std(predictions):.1f} K\n\n")

        if mae is not None:
            f.write(f"VALIDATION METRICS (vs. Gaia GSP-Phot):\n")
            f.write(f"  MAE: {mae:.1f} K\n")
            f.write(f"  Median AE: {median_ae:.1f} K\n")
            f.write(f"  RMSE: {rmse:.1f} K\n")
            f.write(f"  R²: {r2:.3f}\n")
            f.write(f"  Mean error: {mean_error_percent:.2f}%\n")
            f.write(f"  Median error: {median_error_percent:.2f}%\n")
        else:
            f.write(f"VALIDATION METRICS: Not available (no ground truth)\n")

    logger.info(f"  Summary saved: {summary_file}")

    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict temperatures using basic Random Forest model (no feature engineering)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model and data
  python scripts/predict_temperatures_basic_model.py

  # Specify custom model
  python scripts/predict_temperatures_basic_model.py --model models/rf_temperature_regressor_20251001_125556.pkl

  # Predict for custom dataset
  python scripts/predict_temperatures_basic_model.py --input data/processed/eb_2mass_photometry.parquet --output data/processed/my_predictions.parquet
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained basic model (default: rf_temperature_regressor_20251001_125556.pkl)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input dataset with Pan-STARRS photometry (default: ml_training_data_with_gaia.parquet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for predictions (default: basic_model_predictions.parquet)'
    )

    parser.add_argument(
        '--missing-value',
        type=float,
        default=-999.0,
        help='Value representing missing data (default: -999.0)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config()

    # Determine model file
    if args.model:
        model_file = Path(args.model)
    else:
        # Use the basic model (without feature engineering)
        model_file = config.get_path('models') / 'rf_temperature_regressor_20251001_125556.pkl'

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = config.get_dataset_path('ml_training_data_with_gaia', 'processed')

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.get_path('processed') / 'basic_model_predictions.parquet'

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Predict temperatures
    results = predict_temperatures_basic(
        model_file=model_file,
        input_file=input_file,
        output_file=output_file,
        missing_value=args.missing_value
    )

    logger = logging.getLogger(__name__)
    logger.info(f"\n✓ Prediction completed!")
    logger.info(f"✓ Objects predicted: {len(results):,}")
    logger.info(f"✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
