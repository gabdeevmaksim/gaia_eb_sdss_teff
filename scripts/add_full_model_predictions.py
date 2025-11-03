#!/usr/bin/env python3
"""
Add full model predictions to existing colours_of_eb_stars_with_predictions files.

This script:
1. Loads existing predictions (from optimized model)
2. Generates predictions using the new full model
3. Adds new columns with "_full" suffix
4. Updates both Parquet and FITS files

New columns added:
- teff_predicted_full: Predicted temperature from full model (K)
- teff_expected_mae_full: Expected MAE from full model (K)
- teff_expected_rmse_full: Expected RMSE from full model (K)
- teff_prediction_std_full: Uncertainty from RF tree variance (K)
- within_10pct_probability_full: Probability within 10% (%)
- temp_bin_full: Temperature classification
- model_difference: Difference between full and optimized predictions (K)

Author: Claude Code
Date: 2025-10-22
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from astropy.table import Table
from astropy.io import fits

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Temperature-dependent error statistics from full model validation
TEMP_BIN_STATS_FULL = {
    '<4000K': {'mae': 413.8, 'rmse': 780.4, 'within_10': 68.4},
    '4000-5000K': {'mae': 390.6, 'rmse': 681.7, 'within_10': 74.1},
    '5000-6000K': {'mae': 524.0, 'rmse': 804.1, 'within_10': 63.9},
    '6000-7000K': {'mae': 636.7, 'rmse': 966.5, 'within_10': 64.6},
    '7000-10000K': {'mae': 1779.9, 'rmse': 2252.9, 'within_10': 27.3},
    '>10000K': {'mae': 3767.3, 'rmse': 4816.9, 'within_10': 23.6}
}


def assign_temp_bin(temp):
    """Assign temperature to bin category."""
    if temp < 4000:
        return '<4000K'
    elif temp < 5000:
        return '4000-5000K'
    elif temp < 6000:
        return '5000-6000K'
    elif temp < 7000:
        return '6000-7000K'
    elif temp < 10000:
        return '7000-10000K'
    else:
        return '>10000K'


def get_bin_statistics(temp_bin):
    """Get MAE, RMSE, and within_10% for a temperature bin."""
    stats = TEMP_BIN_STATS_FULL.get(temp_bin, TEMP_BIN_STATS_FULL['5000-6000K'])
    return stats['mae'], stats['rmse'], stats['within_10']


def main():
    """Main workflow."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("ADD FULL MODEL PREDICTIONS TO EXISTING FILES")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")

    # File names
    FULL_MODEL_ID = 'rf_gaia_g_bprp_full_20251022_135438'
    INPUT_PARQUET = 'colours_of_eb_stars_with_predictions.parquet'
    OUTPUT_PARQUET = 'colours_of_eb_stars_with_predictions.parquet'
    OUTPUT_FITS = 'colours_of_eb_stars_with_predictions.fits'
    SUMMARY_FILE = 'colours_of_eb_stars_with_predictions_SUMMARY.txt'

    logger.info(f"\nFull model: {FULL_MODEL_ID}")
    logger.info(f"Input: {INPUT_PARQUET}")

    # Step 1: Load existing predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD EXISTING PREDICTIONS")
    logger.info("="*80)

    input_path = processed_dir / INPUT_PARQUET
    logger.info(f"\nLoading: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} objects")
    logger.info(f"Existing columns: {len(df.columns)}")

    # Check for existing predictions
    if 'teff_predicted' in df.columns:
        logger.info(f"\n✓ Found optimized model predictions")
        logger.info(f"  Mean optimized prediction: {df['teff_predicted'].mean():.1f} K")

    # Step 2: Load full model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOAD FULL MODEL")
    logger.info("="*80)

    model_path = models_dir / f'{FULL_MODEL_ID}.pkl'
    logger.info(f"\nLoading model: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Model loaded successfully")
    logger.info(f"Number of trees: {model.n_estimators}")

    # Step 3: Prepare features
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREPARE FEATURES FOR FULL MODEL")
    logger.info("="*80)

    FEATURE_COLS = ['phot_g_mean_mag', 'bp_rp']

    # Check that required features exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' not found in data!")

    logger.info(f"\nFeatures: {FEATURE_COLS}")
    logger.info(f"All required features present: ✓")

    # Step 4: Generate predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 4: GENERATE FULL MODEL PREDICTIONS")
    logger.info("="*80)

    logger.info(f"\nPreparing feature matrix...")
    X = df[FEATURE_COLS].values
    logger.info(f"Feature matrix shape: {X.shape}")

    logger.info(f"\nMaking predictions...")
    y_pred_full = model.predict(X)
    logger.info(f"Predictions generated: {len(y_pred_full):,}")

    # Get prediction uncertainty from individual tree predictions
    logger.info(f"\nCalculating prediction uncertainty from RF trees...")
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    y_pred_std_full = np.std(tree_predictions, axis=0)
    logger.info(f"Tree variance calculated for {len(y_pred_std_full):,} predictions")

    # Step 5: Calculate temperature-dependent metrics
    logger.info("\n" + "="*80)
    logger.info("STEP 5: CALCULATE TEMPERATURE-DEPENDENT METRICS")
    logger.info("="*80)

    logger.info("\nAssigning temperature bins...")
    temp_bins_full = [assign_temp_bin(temp) for temp in y_pred_full]

    logger.info("Calculating temperature-dependent error estimates...")
    mae_values_full = []
    rmse_values_full = []
    within_10_values_full = []

    for temp_bin in temp_bins_full:
        mae, rmse, within_10 = get_bin_statistics(temp_bin)
        mae_values_full.append(mae)
        rmse_values_full.append(rmse)
        within_10_values_full.append(within_10)

    # Step 6: Add new columns
    logger.info("\n" + "="*80)
    logger.info("STEP 6: ADD NEW COLUMNS")
    logger.info("="*80)

    logger.info(f"\nAdding full model prediction columns...")

    df['teff_predicted_full'] = y_pred_full
    df['teff_expected_mae_full'] = mae_values_full
    df['teff_expected_rmse_full'] = rmse_values_full
    df['teff_prediction_std_full'] = y_pred_std_full
    df['within_10pct_probability_full'] = within_10_values_full
    df['temp_bin_full'] = temp_bins_full

    # Calculate difference between models
    if 'teff_predicted' in df.columns:
        df['model_difference'] = df['teff_predicted_full'] - df['teff_predicted']
        logger.info(f"  - model_difference (full - optimized)")

    logger.info(f"\nNew columns added: 7")
    logger.info(f"  - teff_predicted_full")
    logger.info(f"  - teff_expected_mae_full")
    logger.info(f"  - teff_expected_rmse_full")
    logger.info(f"  - teff_prediction_std_full")
    logger.info(f"  - within_10pct_probability_full")
    logger.info(f"  - temp_bin_full")

    logger.info(f"\nTotal columns now: {len(df.columns)}")

    # Step 7: Statistics comparison
    logger.info("\n" + "="*80)
    logger.info("STEP 7: COMPARE PREDICTIONS")
    logger.info("="*80)

    logger.info(f"\nFull model predictions:")
    logger.info(f"  Mean:   {y_pred_full.mean():.1f} K")
    logger.info(f"  Median: {np.median(y_pred_full):.1f} K")
    logger.info(f"  Std:    {y_pred_full.std():.1f} K")
    logger.info(f"  Min:    {y_pred_full.min():.1f} K")
    logger.info(f"  Max:    {y_pred_full.max():.1f} K")

    if 'teff_predicted' in df.columns:
        logger.info(f"\nComparison with optimized model:")
        logger.info(f"  Mean difference: {df['model_difference'].mean():.1f} K")
        logger.info(f"  Mean absolute difference: {df['model_difference'].abs().mean():.1f} K")
        logger.info(f"  Std of difference: {df['model_difference'].std():.1f} K")

        # Correlation
        corr = df['teff_predicted'].corr(df['teff_predicted_full'])
        logger.info(f"  Correlation: {corr:.4f}")

        # Percentage of objects with similar predictions (within 100K)
        similar = (df['model_difference'].abs() <= 100).sum()
        pct_similar = 100 * similar / len(df)
        logger.info(f"  Within 100K: {similar:,} ({pct_similar:.1f}%)")

    # Temperature distribution
    logger.info(f"\nFull model temperature distribution:")
    for bin_name in ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']:
        count = temp_bins_full.count(bin_name)
        pct = 100 * count / len(temp_bins_full)
        logger.info(f"  {bin_name:15s}: {count:7,} objects ({pct:5.1f}%)")

    # Step 8: Save updated Parquet file
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SAVE UPDATED FILES")
    logger.info("="*80)

    parquet_path = processed_dir / OUTPUT_PARQUET
    logger.info(f"\nSaving Parquet file: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    parquet_size = parquet_path.stat().st_size / 1e6
    logger.info(f"Saved: {parquet_size:.1f} MB")

    # Step 9: Save updated FITS file
    fits_path = processed_dir / OUTPUT_FITS
    logger.info(f"\nSaving FITS file: {fits_path}")

    # Convert to Astropy Table
    output_table = Table.from_pandas(df)

    # Update metadata
    output_table.meta['COMMENT'] = f'Updated with full model predictions on {datetime.now().isoformat()}'
    output_table.meta['MODEL1'] = 'rf_gaia_g_bprp_optimized_20251022_091954'
    output_table.meta['MODEL2'] = FULL_MODEL_ID
    output_table.meta['M1_MAE'] = 659.5
    output_table.meta['M2_MAE'] = 665.2

    # Write FITS
    output_table.write(fits_path, format='fits', overwrite=True)
    fits_size = fits_path.stat().st_size / 1e6
    logger.info(f"Saved: {fits_size:.1f} MB")

    # Step 10: Update summary
    logger.info("\n" + "="*80)
    logger.info("STEP 10: UPDATE SUMMARY")
    logger.info("="*80)

    summary_path = processed_dir / SUMMARY_FILE
    logger.info(f"\nUpdating summary: {summary_path}")

    with open(summary_path, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("FULL MODEL PREDICTIONS ADDED\n")
        f.write("="*80 + "\n")
        f.write(f"Updated: {datetime.now().isoformat()}\n")
        f.write(f"Full Model: {FULL_MODEL_ID}\n\n")

        f.write("FULL MODEL PERFORMANCE (Test Set):\n")
        f.write(f"  MAE:  665.2 K\n")
        f.write(f"  RMSE: 1208.2 K\n")
        f.write(f"  R²:   0.504\n")
        f.write(f"  Within 10%: 63.6%\n\n")

        f.write("FULL MODEL PREDICTIONS:\n")
        f.write(f"  Mean:   {y_pred_full.mean():.1f} K\n")
        f.write(f"  Median: {np.median(y_pred_full):.1f} K\n")
        f.write(f"  Std:    {y_pred_full.std():.1f} K\n\n")

        if 'teff_predicted' in df.columns:
            f.write("COMPARISON WITH OPTIMIZED MODEL:\n")
            f.write(f"  Mean difference: {df['model_difference'].mean():.1f} K\n")
            f.write(f"  Mean absolute difference: {df['model_difference'].abs().mean():.1f} K\n")
            f.write(f"  Correlation: {corr:.4f}\n")
            f.write(f"  Within 100K: {pct_similar:.1f}%\n\n")

        f.write("ADDITIONAL COLUMNS ADDED:\n")
        f.write("  - teff_predicted_full: Predicted temperature from full model (K)\n")
        f.write("  - teff_expected_mae_full: Expected MAE from full model (K)\n")
        f.write("  - teff_expected_rmse_full: Expected RMSE from full model (K)\n")
        f.write("  - teff_prediction_std_full: Uncertainty from RF tree variance (K)\n")
        f.write("  - within_10pct_probability_full: Probability within 10% (%)\n")
        f.write("  - temp_bin_full: Temperature classification\n")
        f.write("  - model_difference: Difference between full and optimized (K)\n\n")

        f.write("TEMPERATURE DISTRIBUTION (FULL MODEL):\n")
        for bin_name in ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']:
            count = temp_bins_full.count(bin_name)
            pct = 100 * count / len(temp_bins_full)
            f.write(f"  {bin_name:15s}: {count:7,} objects ({pct:5.1f}%)\n")

    logger.info("Summary updated")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("UPDATE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info(f"\nUpdated files:")
    logger.info(f"  Parquet: {parquet_path} ({parquet_size:.1f} MB)")
    logger.info(f"  FITS:    {fits_path} ({fits_size:.1f} MB)")
    logger.info(f"  Summary: {summary_path}")

    logger.info(f"\nTotal predictions: {len(df):,} objects")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"\nNew columns available:")
    logger.info(f"  - Optimized model: teff_predicted")
    logger.info(f"  - Full model:      teff_predicted_full")
    logger.info(f"  - Difference:      model_difference")


if __name__ == '__main__':
    main()
