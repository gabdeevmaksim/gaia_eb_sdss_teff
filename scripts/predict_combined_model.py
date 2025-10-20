#!/usr/bin/env python3
"""
Predict temperatures using combined Pan-STARRS + 2MASS + Gaia model.

This script:
1. Loads objects without Gaia GSP-Phot temperatures
2. Merges with Pan-STARRS and 2MASS photometry
3. Makes temperature predictions using all 8 color features
4. Saves predictions to parquet file

Usage:
    python scripts/predict_combined_model.py

Author: Claude Code
Date: 2025-10-18
"""

import sys
from pathlib import Path
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


def load_model(model_id, models_dir):
    """Load trained model."""
    model_file = models_dir / f'{model_id}.pkl'

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    logger.info(f"Loading model: {model_file.name}")
    model = joblib.load(model_file)

    return model


def load_and_prepare_data(config):
    """
    Load catalog, Pan-STARRS, and 2MASS data for objects without Gaia Teff.

    Returns
    -------
    pd.DataFrame
        Objects without Gaia Teff with all 8 color features
    """
    logger.info("\nLoading data...")

    # Load main catalog
    catalog_file = config.get_path('processed') / 'eb_catalog.parquet'
    logger.info(f"  1. Loading catalog: {catalog_file.name}")
    df_catalog = pd.read_parquet(catalog_file)
    logger.info(f"     Total objects: {len(df_catalog):,}")

    # Filter objects WITHOUT Gaia Teff
    df_no_teff = df_catalog[df_catalog['teff_gspphot'].isna()].copy()
    logger.info(f"     Objects without Gaia Teff: {len(df_no_teff):,}")

    # Load Pan-STARRS photometry
    logger.info(f"  2. Loading Pan-STARRS photometry...")
    ps_file = config.get_path('processed') / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pd.read_parquet(ps_file)

    # Keep only necessary Pan-STARRS columns
    ps_cols = ['original_ext_source_id', 'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color']
    df_ps_clean = df_ps[ps_cols].copy()
    logger.info(f"     Total Pan-STARRS objects: {len(df_ps_clean):,}")

    # Merge with Pan-STARRS
    logger.info(f"  3. Merging with Pan-STARRS...")
    df_merged = df_no_teff.merge(
        df_ps_clean,
        on='original_ext_source_id',
        how='inner'
    )
    logger.info(f"     Objects with Pan-STARRS: {len(df_merged):,}")

    # Load 2MASS photometry
    logger.info(f"  4. Loading 2MASS photometry...")
    tmass_file = config.get_path('processed') / 'eb_2mass_photometry.parquet'
    df_2mass = pd.read_parquet(tmass_file)

    # Calculate 2MASS colors
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']

    logger.info(f"     Total 2MASS objects: {len(df_2mass):,}")

    # Keep only necessary 2MASS columns
    tmass_cols = ['source_id', 'j_h_color', 'h_k_color', 'j_k_color',
                  'j_snr', 'h_snr', 'k_snr', 'ph_qual']
    df_2mass_clean = df_2mass[tmass_cols].copy()

    # Merge with 2MASS
    logger.info(f"  5. Merging with 2MASS...")
    df_combined = df_merged.merge(
        df_2mass_clean,
        on='source_id',
        how='inner'
    )
    logger.info(f"     Objects with both Pan-STARRS and 2MASS: {len(df_combined):,}")

    # Filter for valid colors (same criteria as training)
    logger.info(f"\n  6. Filtering for valid colors...")

    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g_r_color': (-0.5, 3.0),
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'B_V_color': (-0.5, 3.0),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5)
    }

    valid_mask = pd.Series(True, index=df_combined.index)
    for col, (min_val, max_val) in color_ranges.items():
        valid_mask &= (df_combined[col] >= min_val) & (df_combined[col] <= max_val)

    df_valid = df_combined[valid_mask].copy()
    logger.info(f"     Objects with valid colors: {len(df_valid):,}")

    # Remove any remaining NaN values in color columns
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color',
                  'j_h_color', 'h_k_color', 'j_k_color']

    df_valid = df_valid.dropna(subset=color_cols)
    logger.info(f"     Final objects for prediction: {len(df_valid):,}")

    return df_valid


def make_predictions(df_data, model, model_id):
    """
    Make temperature predictions using trained model.

    Parameters
    ----------
    df_data : pd.DataFrame
        Data with all 8 color features
    model : sklearn model
        Trained Random Forest model
    model_id : str
        Model identifier

    Returns
    -------
    pd.DataFrame
        Predictions with metadata
    """
    logger.info("\nMaking predictions...")

    # Define color features in correct order
    color_features = [
        'bp_rp',
        'g_r_color',
        'r_i_color',
        'i_z_color',
        'B_V_color',
        'j_h_color',
        'h_k_color',
        'j_k_color'
    ]

    # Prepare feature matrix
    X = df_data[color_features].copy()

    logger.info(f"  Feature matrix shape: {X.shape}")
    logger.info(f"  Features: {color_features}")

    # Make predictions
    predictions = model.predict(X)

    logger.info(f"  Predictions made: {len(predictions):,}")
    logger.info(f"  Temperature range: {predictions.min():.0f} - {predictions.max():.0f} K")
    logger.info(f"  Mean: {predictions.mean():.0f} K")
    logger.info(f"  Median: {np.median(predictions):.0f} K")

    # Create output dataframe
    df_pred = pd.DataFrame({
        'source_id': df_data['source_id'],
        'original_ext_source_id': df_data['original_ext_source_id'],
        'teff_predicted': predictions,
        'bp_rp': df_data['bp_rp'],
        'g_r_color': df_data['g_r_color'],
        'r_i_color': df_data['r_i_color'],
        'i_z_color': df_data['i_z_color'],
        'B_V_color': df_data['B_V_color'],
        'j_h_color': df_data['j_h_color'],
        'h_k_color': df_data['h_k_color'],
        'j_k_color': df_data['j_k_color'],
        'j_snr': df_data['j_snr'],
        'h_snr': df_data['h_snr'],
        'k_snr': df_data['k_snr'],
        'ph_qual': df_data['ph_qual'],
        'model_id': model_id
    })

    return df_pred


def main():
    """Main prediction workflow."""
    config = get_config()
    models_dir = config.get_path('models')
    processed_dir = config.get_path('processed')

    # Model ID
    MODEL_ID = 'rf_combined_colors_20251018_115824'

    logger.info("=" * 80)
    logger.info("PREDICT TEMPERATURES - COMBINED PAN-STARRS + 2MASS MODEL")
    logger.info("=" * 80)
    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load model
    model = load_model(MODEL_ID, models_dir)

    # Load and prepare data
    df_data = load_and_prepare_data(config)

    # Make predictions
    df_predictions = make_predictions(df_data, model, MODEL_ID)

    # Save predictions
    output_file = processed_dir / f'combined_predictions_{MODEL_ID}.parquet'
    logger.info(f"\nSaving predictions: {output_file.name}")
    df_predictions.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nTotal predictions: {len(df_predictions):,}")
    logger.info(f"Temperature range: {df_predictions['teff_predicted'].min():.0f} - {df_predictions['teff_predicted'].max():.0f} K")
    logger.info(f"Temperature mean: {df_predictions['teff_predicted'].mean():.0f} K")
    logger.info(f"Temperature median: {df_predictions['teff_predicted'].median():.0f} K")
    logger.info(f"Temperature std: {df_predictions['teff_predicted'].std():.0f} K")

    # Temperature distribution
    logger.info(f"\nTemperature distribution:")
    temp_bins = [0, 4000, 5000, 6000, 8000, 50000]
    temp_labels = ['<4000 K', '4000-5000 K', '5000-6000 K', '6000-8000 K', '>8000 K']
    df_predictions['temp_bin'] = pd.cut(df_predictions['teff_predicted'], bins=temp_bins, labels=temp_labels)

    for label in temp_labels:
        count = (df_predictions['temp_bin'] == label).sum()
        pct = 100 * count / len(df_predictions)
        logger.info(f"  {label:<15} {count:>6,} ({pct:>5.1f}%)")

    # 2MASS quality statistics
    high_quality = df_predictions['ph_qual'].isin(['AAA', 'AAB']).sum()
    logger.info(f"\n2MASS photometry quality:")
    logger.info(f"  High quality (AAA/AAB): {high_quality:,} ({100*high_quality/len(df_predictions):.1f}%)")

    # Color statistics
    logger.info(f"\nColor statistics:")
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color',
                  'j_h_color', 'h_k_color', 'j_k_color']
    for col in color_cols:
        logger.info(f"  {col:<15} mean: {df_predictions[col].mean():>6.3f}  "
                   f"range: ({df_predictions[col].min():>6.3f}, {df_predictions[col].max():>6.3f})")

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")

    # Create schema documentation
    docs_dir = Path('docs')
    schema_file = docs_dir / f'COMBINED_PREDICTIONS_SCHEMA.md'

    with open(schema_file, 'w') as f:
        f.write("# Combined Pan-STARRS + 2MASS Predictions Schema\n\n")
        f.write(f"**Dataset**: `{output_file.name}`\n\n")
        f.write(f"**Model**: {MODEL_ID}\n\n")
        f.write(f"**Total predictions**: {len(df_predictions):,}\n\n")

        f.write("## Columns\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| source_id | int64 | Gaia DR3 source identifier (common key) |\n")
        f.write("| original_ext_source_id | int64 | Pan-STARRS source identifier |\n")
        f.write("| teff_predicted | float64 | Predicted effective temperature (K) |\n")
        f.write("| bp_rp | float64 | Gaia BP-RP color |\n")
        f.write("| g_r_color | float64 | Pan-STARRS g-r color |\n")
        f.write("| r_i_color | float64 | Pan-STARRS r-i color |\n")
        f.write("| i_z_color | float64 | Pan-STARRS i-z color |\n")
        f.write("| B_V_color | float64 | Synthetic B-V color |\n")
        f.write("| j_h_color | float64 | 2MASS J-H color |\n")
        f.write("| h_k_color | float64 | 2MASS H-K color |\n")
        f.write("| j_k_color | float64 | 2MASS J-K color |\n")
        f.write("| j_snr | float64 | 2MASS J-band signal-to-noise ratio |\n")
        f.write("| h_snr | float64 | 2MASS H-band signal-to-noise ratio |\n")
        f.write("| k_snr | float64 | 2MASS K-band signal-to-noise ratio |\n")
        f.write("| ph_qual | str | 2MASS photometric quality flag (AAA=best) |\n")
        f.write("| model_id | str | Model identifier |\n\n")

        f.write("## Statistics\n\n")
        f.write(f"```\n")
        f.write(f"Temperature range: {df_predictions['teff_predicted'].min():.0f} - {df_predictions['teff_predicted'].max():.0f} K\n")
        f.write(f"Temperature mean:  {df_predictions['teff_predicted'].mean():.0f} K\n")
        f.write(f"Temperature median: {df_predictions['teff_predicted'].median():.0f} K\n")
        f.write(f"Temperature std:   {df_predictions['teff_predicted'].std():.0f} K\n")
        f.write(f"\nHigh quality 2MASS (AAA/AAB): {high_quality:,} ({100*high_quality/len(df_predictions):.1f}%)\n")
        f.write(f"```\n\n")

        f.write("## Temperature Distribution\n\n")
        f.write("```\n")
        for label in temp_labels:
            count = (df_predictions['temp_bin'] == label).sum()
            pct = 100 * count / len(df_predictions)
            f.write(f"{label:<15} {count:>6,} ({pct:>5.1f}%)\n")
        f.write("```\n\n")

        f.write("## Model Performance (Test Set)\n\n")
        f.write(f"```\n")
        f.write(f"MAE:  694.6 K\n")
        f.write(f"RMSE: 1080.3 K\n")
        f.write(f"R²:   0.477\n")
        f.write(f"Within 10%: 56.5%\n")
        f.write(f"```\n\n")

        f.write("## Feature Importance\n\n")
        f.write("```\n")
        f.write("j_h_color  (37.95%) - Most important!\n")
        f.write("bp_rp      (23.54%)\n")
        f.write("h_k_color  (12.05%)\n")
        f.write("i_z_color  ( 7.59%)\n")
        f.write("r_i_color  ( 5.89%)\n")
        f.write("j_k_color  ( 5.68%)\n")
        f.write("g_r_color  ( 3.66%)\n")
        f.write("B_V_color  ( 3.64%)\n")
        f.write("```\n\n")

        f.write("## Common Keys\n\n")
        f.write("- Use `source_id` to join with Gaia catalog\n")
        f.write("- Use `original_ext_source_id` to join with Pan-STARRS catalog\n\n")

        f.write("## Wavelength Coverage\n\n")
        f.write("Predictions use photometry spanning:\n")
        f.write("- **Optical**: 330 nm (Gaia BP) to 1000 nm (Pan-STARRS z)\n")
        f.write("- **Near-IR**: 1.25 μm (2MASS J) to 2.17 μm (2MASS K)\n\n")
        f.write("**Total range**: 330 nm to 2.17 μm (~4 orders of magnitude)\n")

    logger.info(f"Schema documentation: {schema_file}")


if __name__ == '__main__':
    main()
