#!/usr/bin/env python3
"""
Predict temperatures using Gaia + 2MASS engineered features model.

This script:
1. Loads objects without Gaia GSP-Phot temperatures
2. Merges with 2MASS photometry
3. Engineers features identically to training data
4. Makes temperature predictions
5. Saves predictions to parquet file

Usage:
    python scripts/predict_gaia_2mass_engineered.py

Author: Claude Code
Date: 2025-10-16
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
from src.features.engineering import engineer_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model_and_selector(model_id, models_dir):
    """Load trained model and feature selector."""
    model_file = models_dir / f'{model_id}.pkl'
    selector_file = models_dir / f'{model_id}_selector.pkl'

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    if not selector_file.exists():
        raise FileNotFoundError(f"Selector not found: {selector_file}")

    logger.info(f"Loading model: {model_file.name}")
    model = joblib.load(model_file)

    logger.info(f"Loading selector: {selector_file.name}")
    selector = joblib.load(selector_file)

    return model, selector


def load_and_prepare_data(config):
    """
    Load catalog and 2MASS data, filter objects without Gaia Teff.

    Returns
    -------
    pd.DataFrame
        Objects without Gaia Teff with all required color features
    """
    logger.info("\nLoading data...")

    # Load main catalog
    catalog_file = config.get_path('processed') / 'eb_catalog.parquet'
    logger.info(f"  Loading catalog: {catalog_file.name}")
    df_catalog = pd.read_parquet(catalog_file)
    logger.info(f"    Total objects: {len(df_catalog):,}")

    # Filter objects WITHOUT Gaia Teff
    df_no_teff = df_catalog[df_catalog['teff_gspphot'].isna()].copy()
    logger.info(f"    Objects without Gaia Teff: {len(df_no_teff):,}")

    # Load 2MASS photometry
    tmass_file = config.get_path('processed') / 'eb_2mass_photometry.parquet'
    logger.info(f"  Loading 2MASS data: {tmass_file.name}")
    df_2mass = pd.read_parquet(tmass_file)

    # Calculate 2MASS colors
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']

    logger.info(f"    Total 2MASS objects: {len(df_2mass):,}")

    # Merge on Gaia source_id
    logger.info("\n  Merging catalogs on source_id...")
    df_merged = df_no_teff.merge(
        df_2mass[['source_id', 'j_h_color', 'h_k_color', 'j_k_color',
                  'j_snr', 'h_snr', 'k_snr', 'ph_qual']],
        on='source_id',
        how='inner'
    )
    logger.info(f"    Objects with 2MASS matches: {len(df_merged):,}")

    # Filter for valid colors (same criteria as training data)
    logger.info("\n  Filtering for valid colors...")
    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5)
    }

    valid_mask = pd.Series(True, index=df_merged.index)
    for col, (min_val, max_val) in color_ranges.items():
        valid_mask &= (df_merged[col] >= min_val) & (df_merged[col] <= max_val)

    df_valid = df_merged[valid_mask].copy()
    logger.info(f"    Objects with valid colors: {len(df_valid):,}")

    # Remove any remaining NaN values in color columns
    color_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
    df_valid = df_valid.dropna(subset=color_cols)
    logger.info(f"    Final objects for prediction: {len(df_valid):,}")

    return df_valid


def engineer_prediction_features(df):
    """
    Engineer features identically to training data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with base color features

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    logger.info("\nEngineering features...")

    color_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']

    # Keep only necessary columns for feature engineering
    df_colors = df[['source_id'] + color_cols].copy()

    # Apply same feature engineering as training
    df_engineered = engineer_all_features(
        df_colors,
        color_cols=color_cols,
        mag_cols=None,
        include_polynomials=True,
        include_interactions=True,
        include_log=True,
        include_temp_dependent=True,
        include_mag_features=False
    )

    # Replace NaN and Inf with 0 (same as training)
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
    df_engineered = df_engineered.fillna(0)

    logger.info(f"  Engineered features: {len(df_engineered.columns)} total columns")

    return df_engineered


def make_predictions(df_engineered, df_original, model, selector, model_id):
    """
    Make temperature predictions using trained model.

    Parameters
    ----------
    df_engineered : pd.DataFrame
        Engineered features
    df_original : pd.DataFrame
        Original data with metadata
    model : sklearn model
        Trained Random Forest model
    selector : SelectKBest
        Fitted feature selector
    model_id : str
        Model identifier

    Returns
    -------
    pd.DataFrame
        Predictions with metadata
    """
    logger.info("\nMaking predictions...")

    # Prepare feature matrix (exclude source_id)
    feature_cols = [col for col in df_engineered.columns if col != 'source_id']
    X = df_engineered[feature_cols]

    logger.info(f"  Feature matrix shape: {X.shape}")

    # Apply feature selection
    X_selected = selector.transform(X)
    logger.info(f"  Selected features: {X_selected.shape[1]}")

    # Make predictions
    predictions = model.predict(X_selected)

    logger.info(f"  Predictions made: {len(predictions):,}")
    logger.info(f"  Temperature range: {predictions.min():.0f} - {predictions.max():.0f} K")
    logger.info(f"  Mean: {predictions.mean():.0f} K")
    logger.info(f"  Median: {np.median(predictions):.0f} K")

    # Create output dataframe
    df_pred = pd.DataFrame({
        'source_id': df_engineered['source_id'],
        'teff_predicted': predictions,
        'bp_rp': df_original['bp_rp'],
        'j_h_color': df_original['j_h_color'],
        'h_k_color': df_original['h_k_color'],
        'j_k_color': df_original['j_k_color'],
        'j_snr': df_original['j_snr'],
        'h_snr': df_original['h_snr'],
        'k_snr': df_original['k_snr'],
        'ph_qual': df_original['ph_qual'],
        'model_id': model_id
    })

    return df_pred


def main():
    """Main prediction workflow."""
    config = get_config()
    models_dir = config.get_path('models')
    processed_dir = config.get_path('processed')

    # Model ID
    MODEL_ID = 'rf_gaia_2mass_engineered_20251016_205640'

    logger.info("=" * 80)
    logger.info("PREDICT TEMPERATURES - GAIA + 2MASS ENGINEERED MODEL")
    logger.info("=" * 80)
    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load model and selector
    model, selector = load_model_and_selector(MODEL_ID, models_dir)

    # Load and prepare data
    df_data = load_and_prepare_data(config)

    # Engineer features
    df_engineered = engineer_prediction_features(df_data)

    # Make predictions
    df_predictions = make_predictions(df_engineered, df_data, model, selector, MODEL_ID)

    # Save predictions
    output_file = processed_dir / f'gaia_2mass_engineered_predictions_{MODEL_ID}.parquet'
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

    # 2MASS quality statistics
    high_quality = df_predictions['ph_qual'].isin(['AAA', 'AAB']).sum()
    logger.info(f"\n2MASS photometry quality:")
    logger.info(f"  High quality (AAA/AAB): {high_quality:,} ({100*high_quality/len(df_predictions):.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")

    # Create schema documentation
    docs_dir = Path('docs')
    schema_file = docs_dir / f'GAIA_2MASS_ENGINEERED_PREDICTIONS_SCHEMA.md'

    with open(schema_file, 'w') as f:
        f.write("# Gaia + 2MASS Engineered Model Predictions Schema\n\n")
        f.write(f"**Dataset**: `{output_file.name}`\n\n")
        f.write(f"**Model**: {MODEL_ID}\n\n")
        f.write(f"**Total predictions**: {len(df_predictions):,}\n\n")

        f.write("## Columns\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| source_id | int64 | Gaia DR3 source identifier (common key) |\n")
        f.write("| teff_predicted | float64 | Predicted effective temperature (K) |\n")
        f.write("| bp_rp | float64 | Gaia BP-RP color |\n")
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

        f.write("## Model Performance (Test Set)\n\n")
        f.write(f"```\n")
        f.write(f"MAE:  722.3 K\n")
        f.write(f"RMSE: 1138.0 K\n")
        f.write(f"R²:   0.442\n")
        f.write(f"Within 10%: 55.3%\n")
        f.write(f"```\n\n")

        f.write("## Common Key\n\n")
        f.write("Use `source_id` to join with other catalogs (Gaia, Pan-STARRS, etc.)\n")

    logger.info(f"Schema documentation: {schema_file}")


if __name__ == '__main__':
    main()
