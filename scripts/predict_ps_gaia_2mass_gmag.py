#!/usr/bin/env python3
"""
Generate temperature predictions using Pan-STARRS + Gaia + 2MASS + g mag model.

This script loads the trained model and predicts temperatures for objects
that have the required features but no Gaia GSP-Phot temperature.

Usage:
    python scripts/predict_ps_gaia_2mass_gmag.py --model models/rf_ps_gaia_2mass_gmag_20251020_142522.pkl

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_prediction_data(config):
    """Load data for objects without Gaia Teff."""
    processed_dir = config.get_path('processed')

    logger.info("\n1. Loading main catalog...")
    catalog_file = processed_dir / 'eb_catalog.parquet'
    df_catalog = pd.read_parquet(catalog_file)
    logger.info(f"   Total objects: {len(df_catalog):,}")

    # Filter objects WITHOUT Gaia Teff
    df_no_teff = df_catalog[df_catalog['teff_gspphot'].isna()].copy()
    logger.info(f"   Objects without Gaia Teff: {len(df_no_teff):,}")

    # Load Pan-STARRS photometry
    logger.info("\n2. Loading Pan-STARRS photometry...")
    ps_file = processed_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pd.read_parquet(ps_file)

    # Keep only necessary Pan-STARRS columns
    ps_cols = ['original_ext_source_id', 'g_r_color', 'r_i_color', 'i_z_color']
    df_ps_clean = df_ps[ps_cols].copy()

    # Merge with Pan-STARRS
    logger.info("\n3. Merging with Pan-STARRS...")
    df_merged = df_no_teff.merge(df_ps_clean, on='original_ext_source_id', how='inner')
    logger.info(f"   Objects with Pan-STARRS: {len(df_merged):,}")

    # Load 2MASS photometry
    logger.info("\n4. Loading 2MASS photometry...")
    tmass_file = processed_dir / 'eb_2mass_photometry.parquet'
    df_2mass = pd.read_parquet(tmass_file)

    # Calculate 2MASS colors
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']

    # Keep only necessary 2MASS columns
    tmass_cols = ['source_id', 'j_h_color', 'h_k_color', 'j_k_color']
    df_2mass_clean = df_2mass[tmass_cols].copy()

    # Merge with 2MASS
    logger.info("\n5. Merging with 2MASS...")
    df_combined = df_merged.merge(df_2mass_clean, on='source_id', how='inner')
    logger.info(f"   Objects with Pan-STARRS + 2MASS: {len(df_combined):,}")

    # Filter for valid colors and magnitude
    logger.info("\n6. Filtering for valid features...")

    feature_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color',
                    'j_h_color', 'h_k_color', 'j_k_color', 'phot_g_mean_mag']

    # Remove NaN values
    before_nan = len(df_combined)
    df_valid = df_combined.dropna(subset=feature_cols).copy()
    after_nan = len(df_valid)
    logger.info(f"   Removed NaN values: {before_nan - after_nan:,}")

    # Apply same filters as training data
    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g_r_color': (-0.5, 3.0),
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5),
        'phot_g_mean_mag': (10.0, 21.0)
    }

    valid_mask = pd.Series(True, index=df_valid.index)
    for col, (min_val, max_val) in color_ranges.items():
        valid_mask &= (df_valid[col] >= min_val) & (df_valid[col] <= max_val)

    df_final = df_valid[valid_mask].copy()
    logger.info(f"   Final prediction dataset: {len(df_final):,} objects")

    return df_final, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description='Generate temperature predictions using trained model'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pkl)')

    args = parser.parse_args()

    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("=" * 80)
    logger.info("PREDICT TEMPERATURES: PAN-STARRS + GAIA + 2MASS + G MAG")
    logger.info("=" * 80)

    # Load model
    logger.info(f"\nLoading model: {args.model}")
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    model = joblib.load(model_path)
    logger.info(f"   Model loaded successfully")

    # Extract model ID from filename
    model_id = model_path.stem
    logger.info(f"   Model ID: {model_id}")

    # Load prediction data
    df_predict, feature_cols = load_prediction_data(config)

    # Prepare features
    logger.info("\n7. Preparing features for prediction...")
    X_predict = df_predict[feature_cols].values
    logger.info(f"   Feature matrix shape: {X_predict.shape}")

    # Generate predictions
    logger.info("\n8. Generating temperature predictions...")
    predictions = model.predict(X_predict)

    logger.info(f"   Predictions generated: {len(predictions):,}")
    logger.info(f"   Prediction range: {predictions.min():.1f} - {predictions.max():.1f} K")
    logger.info(f"   Prediction mean: {predictions.mean():.1f} K")
    logger.info(f"   Prediction median: {np.median(predictions):.1f} K")

    # Add predictions to dataframe
    df_predict['teff_predicted'] = predictions

    # Select output columns
    output_cols = ['source_id', 'original_ext_source_id'] + feature_cols + ['teff_predicted']
    df_output = df_predict[output_cols].copy()

    # Save predictions
    output_file = processed_dir / f'predictions_{model_id}.parquet'
    logger.info(f"\n9. Saving predictions: {output_file.name}")
    df_output.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"   File size: {file_size_mb:.1f} MB")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nObjects predicted: {len(predictions):,}")
    logger.info(f"Temperature statistics:")
    logger.info(f"  Range:  {predictions.min():.1f} - {predictions.max():.1f} K")
    logger.info(f"  Mean:   {predictions.mean():.1f} K")
    logger.info(f"  Median: {np.median(predictions):.1f} K")
    logger.info(f"  Std:    {predictions.std():.1f} K")
    logger.info(f"\nOutput file: {output_file}")


if __name__ == '__main__':
    main()
