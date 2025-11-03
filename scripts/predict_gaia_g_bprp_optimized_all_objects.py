#!/usr/bin/env python3
"""
Predict temperatures for all objects in colours_of_eb_stars-result.fits

Uses optimized Gaia G + BP-RP Random Forest model with temperature-dependent
error estimates and prediction confidence metrics.

Model: rf_gaia_g_bprp_optimized_20251022_091954
Features: phot_g_mean_mag, bp_rp
Test Performance: MAE=659.5K, RMSE=1158.2K, R²=0.346, Within10%=62.8%

Output columns added:
- teff_predicted: Predicted effective temperature (K)
- teff_expected_mae: Expected MAE based on temperature bin (K)
- teff_expected_rmse: Expected RMSE based on temperature bin (K)
- teff_prediction_std: Uncertainty from RF tree variance (K)
- within_10pct_probability: Probability prediction within 10% (%)
- temp_bin: Temperature classification

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


# Temperature-dependent error statistics from validation
TEMP_BIN_STATS = {
    '<4000K': {'mae': 412.4, 'rmse': 764.3, 'within_10': 68.2},
    '4000-5000K': {'mae': 403.8, 'rmse': 696.3, 'within_10': 72.6},
    '5000-6000K': {'mae': 545.6, 'rmse': 847.2, 'within_10': 63.5},
    '6000-7000K': {'mae': 680.8, 'rmse': 953.0, 'within_10': 58.7},
    '7000-10000K': {'mae': 1818.8, 'rmse': 2246.5, 'within_10': 26.5},
    '>10000K': {'mae': 4726.4, 'rmse': 5408.0, 'within_10': 6.4}
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
    stats = TEMP_BIN_STATS.get(temp_bin, TEMP_BIN_STATS['5000-6000K'])  # Default to middle range
    return stats['mae'], stats['rmse'], stats['within_10']


def main():
    """Main prediction workflow."""

    config = get_config()
    raw_dir = config.get_path('raw')
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("PREDICT TEMPERATURES: ALL OBJECTS FROM FITS FILE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")

    # Model information
    MODEL_ID = 'rf_gaia_g_bprp_optimized_20251022_091954'
    INPUT_FILE = 'colours_of_eb_stars-result.fits'
    OUTPUT_PARQUET = 'colours_of_eb_stars_with_predictions.parquet'
    OUTPUT_FITS = 'colours_of_eb_stars_with_predictions.fits'
    OUTPUT_SUMMARY = 'colours_of_eb_stars_with_predictions_SUMMARY.txt'

    logger.info(f"\nModel: {MODEL_ID}")
    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output (Parquet): {OUTPUT_PARQUET}")
    logger.info(f"Output (FITS): {OUTPUT_FITS}")

    # Step 1: Load FITS file
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD INPUT DATA")
    logger.info("="*80)

    input_path = raw_dir / INPUT_FILE
    logger.info(f"\nLoading FITS file: {input_path}")

    table = Table.read(input_path)
    logger.info(f"Total objects in FITS: {len(table):,}")
    logger.info(f"Columns: {', '.join(table.colnames)}")

    # Convert to pandas for easier manipulation
    df = table.to_pandas()
    logger.info(f"Converted to pandas DataFrame")

    # Step 2: Load model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOAD TRAINED MODEL")
    logger.info("="*80)

    model_path = models_dir / f'{MODEL_ID}.pkl'
    logger.info(f"\nLoading model: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Model loaded successfully")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Number of trees: {model.n_estimators}")

    # Step 3: Prepare features
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREPARE FEATURES")
    logger.info("="*80)

    # Required features
    FEATURE_COLS = ['phot_g_mean_mag', 'bp_rp']

    # Map FITS columns to feature names
    logger.info("\nMapping FITS columns to model features:")
    logger.info("  g -> phot_g_mean_mag")
    logger.info("  bp_rp -> bp_rp")

    df['phot_g_mean_mag'] = df['g']
    # bp_rp already exists

    # Check data availability
    logger.info("\nChecking data availability:")
    total_objects = len(df)

    for col in FEATURE_COLS:
        n_valid = df[col].notna().sum()
        n_missing = df[col].isna().sum()
        pct_valid = 100 * n_valid / total_objects
        logger.info(f"  {col:20s}: {n_valid:,} valid ({pct_valid:.1f}%), {n_missing:,} missing")

    # Filter for valid features
    logger.info("\nFiltering for valid features...")
    n_before = len(df)
    df_valid = df.dropna(subset=FEATURE_COLS).copy()
    n_after = len(df_valid)
    n_removed = n_before - n_after

    logger.info(f"  Before: {n_before:,} objects")
    logger.info(f"  After:  {n_after:,} objects")
    logger.info(f"  Removed: {n_removed:,} objects ({100*n_removed/n_before:.2f}%)")

    # Apply quality filters (same as training)
    logger.info("\nApplying quality filters (same as training)...")
    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'phot_g_mean_mag': (5.0, 22.0)  # Reasonable magnitude range
    }

    n_before_filters = len(df_valid)
    for col, (min_val, max_val) in color_ranges.items():
        mask = (df_valid[col] >= min_val) & (df_valid[col] <= max_val)
        n_outliers = (~mask).sum()
        if n_outliers > 0:
            logger.info(f"  {col:20s}: removing {n_outliers:,} outliers outside [{min_val}, {max_val}]")
        df_valid = df_valid[mask].copy()

    n_after_filters = len(df_valid)
    n_filtered = n_before_filters - n_after_filters
    logger.info(f"\n  Objects after quality filters: {n_after_filters:,}")
    logger.info(f"  Removed by filters: {n_filtered:,}")

    # Step 4: Make predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 4: GENERATE PREDICTIONS")
    logger.info("="*80)

    logger.info(f"\nPreparing feature matrix...")
    X = df_valid[FEATURE_COLS].values
    logger.info(f"Feature matrix shape: {X.shape}")

    logger.info(f"\nMaking predictions...")
    y_pred = model.predict(X)
    logger.info(f"Predictions generated: {len(y_pred):,}")

    # Get prediction uncertainty from individual tree predictions
    logger.info(f"\nCalculating prediction uncertainty from RF trees...")
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    y_pred_std = np.std(tree_predictions, axis=0)
    logger.info(f"Tree variance calculated for {len(y_pred_std):,} predictions")

    # Step 5: Add error and confidence metrics
    logger.info("\n" + "="*80)
    logger.info("STEP 5: ADD ERROR AND CONFIDENCE METRICS")
    logger.info("="*80)

    # Assign temperature bins
    logger.info("\nAssigning temperature bins...")
    temp_bins = [assign_temp_bin(temp) for temp in y_pred]

    # Get temperature-dependent statistics
    logger.info("Calculating temperature-dependent error estimates...")
    mae_values = []
    rmse_values = []
    within_10_values = []

    for temp_bin in temp_bins:
        mae, rmse, within_10 = get_bin_statistics(temp_bin)
        mae_values.append(mae)
        rmse_values.append(rmse)
        within_10_values.append(within_10)

    # Add all prediction columns
    df_valid['teff_predicted'] = y_pred
    df_valid['teff_expected_mae'] = mae_values
    df_valid['teff_expected_rmse'] = rmse_values
    df_valid['teff_prediction_std'] = y_pred_std
    df_valid['within_10pct_probability'] = within_10_values
    df_valid['temp_bin'] = temp_bins

    logger.info(f"Added {6} new columns:")
    logger.info(f"  - teff_predicted (K)")
    logger.info(f"  - teff_expected_mae (K)")
    logger.info(f"  - teff_expected_rmse (K)")
    logger.info(f"  - teff_prediction_std (K)")
    logger.info(f"  - within_10pct_probability (%)")
    logger.info(f"  - temp_bin")

    # Step 6: Save results
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVE RESULTS")
    logger.info("="*80)

    # Save Parquet (fast for analysis)
    parquet_path = processed_dir / OUTPUT_PARQUET
    logger.info(f"\nSaving Parquet file: {parquet_path}")
    df_valid.to_parquet(parquet_path, index=False)
    parquet_size = parquet_path.stat().st_size / 1e6
    logger.info(f"Saved: {parquet_size:.1f} MB")

    # Save FITS (astronomical standard)
    fits_path = processed_dir / OUTPUT_FITS
    logger.info(f"\nSaving FITS file: {fits_path}")

    # Convert back to Astropy Table
    # Handle string columns (temp_bin)
    df_for_fits = df_valid.copy()

    # Convert to Astropy Table
    output_table = Table.from_pandas(df_for_fits)

    # Add metadata to FITS header
    output_table.meta['COMMENT'] = f'Temperature predictions generated on {datetime.now().isoformat()}'
    output_table.meta['MODEL'] = MODEL_ID
    output_table.meta['FEATURES'] = 'phot_g_mean_mag, bp_rp'
    output_table.meta['TEST_MAE'] = 659.5
    output_table.meta['TEST_RMSE'] = 1158.2
    output_table.meta['TEST_R2'] = 0.346
    output_table.meta['WITHIN10'] = 62.8

    # Write FITS file
    output_table.write(fits_path, format='fits', overwrite=True)
    fits_size = fits_path.stat().st_size / 1e6
    logger.info(f"Saved: {fits_size:.1f} MB")

    # Step 7: Generate statistics and summary
    logger.info("\n" + "="*80)
    logger.info("STEP 7: GENERATE SUMMARY STATISTICS")
    logger.info("="*80)

    logger.info(f"\nPredicted temperature statistics:")
    logger.info(f"  Minimum:  {y_pred.min():.1f} K")
    logger.info(f"  Maximum:  {y_pred.max():.1f} K")
    logger.info(f"  Mean:     {y_pred.mean():.1f} K")
    logger.info(f"  Median:   {np.median(y_pred):.1f} K")
    logger.info(f"  Std Dev:  {y_pred.std():.1f} K")

    logger.info(f"\nPrediction uncertainty statistics:")
    logger.info(f"  Mean tree std dev: {y_pred_std.mean():.1f} K")
    logger.info(f"  Median tree std dev: {np.median(y_pred_std):.1f} K")

    logger.info(f"\nTemperature distribution by bin:")
    for bin_name in ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']:
        count = temp_bins.count(bin_name)
        pct = 100 * count / len(temp_bins)
        logger.info(f"  {bin_name:15s}: {count:7,} objects ({pct:5.1f}%)")

    # Expected accuracy distribution
    logger.info(f"\nExpected accuracy distribution:")
    logger.info(f"  Mean expected MAE:  {np.mean(mae_values):.1f} K")
    logger.info(f"  Mean expected RMSE: {np.mean(rmse_values):.1f} K")
    logger.info(f"  Mean within 10% probability: {np.mean(within_10_values):.1f}%")

    # Write summary file
    summary_path = processed_dir / OUTPUT_SUMMARY
    logger.info(f"\nWriting summary file: {summary_path}")

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Temperature Predictions Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL_ID}\n\n")

        f.write("MODEL PERFORMANCE (Test Set):\n")
        f.write(f"  MAE:  659.5 K\n")
        f.write(f"  RMSE: 1158.2 K\n")
        f.write(f"  R²:   0.346\n")
        f.write(f"  Within 10%: 62.8%\n\n")

        f.write("INPUT DATA:\n")
        f.write(f"  Input file: {INPUT_FILE}\n")
        f.write(f"  Total objects: {total_objects:,}\n")
        f.write(f"  Objects with valid features: {n_after:,}\n")
        f.write(f"  Objects after quality filters: {n_after_filters:,}\n")
        f.write(f"  Predictions generated: {len(y_pred):,}\n\n")

        f.write("PREDICTED TEMPERATURE STATISTICS:\n")
        f.write(f"  Minimum:  {y_pred.min():.1f} K\n")
        f.write(f"  Maximum:  {y_pred.max():.1f} K\n")
        f.write(f"  Mean:     {y_pred.mean():.1f} K\n")
        f.write(f"  Median:   {np.median(y_pred):.1f} K\n")
        f.write(f"  Std Dev:  {y_pred.std():.1f} K\n\n")

        f.write("PREDICTION UNCERTAINTY:\n")
        f.write(f"  Mean tree std dev: {y_pred_std.mean():.1f} K\n")
        f.write(f"  Median tree std dev: {np.median(y_pred_std):.1f} K\n\n")

        f.write("TEMPERATURE DISTRIBUTION:\n")
        for bin_name in ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']:
            count = temp_bins.count(bin_name)
            pct = 100 * count / len(temp_bins)
            f.write(f"  {bin_name:15s}: {count:7,} objects ({pct:5.1f}%)\n")

        f.write("\nEXPECTED ACCURACY:\n")
        f.write(f"  Mean expected MAE:  {np.mean(mae_values):.1f} K\n")
        f.write(f"  Mean expected RMSE: {np.mean(rmse_values):.1f} K\n")
        f.write(f"  Mean within 10% probability: {np.mean(within_10_values):.1f}%\n\n")

        f.write("TEMPERATURE-DEPENDENT ERROR ESTIMATES:\n")
        f.write("(Based on validation set performance)\n\n")
        for bin_name, stats in TEMP_BIN_STATS.items():
            f.write(f"  {bin_name:15s}: MAE={stats['mae']:6.1f}K, RMSE={stats['rmse']:6.1f}K, Within10%={stats['within_10']:5.1f}%\n")

        f.write("\nOUTPUT FILES:\n")
        f.write(f"  Parquet: {OUTPUT_PARQUET} ({parquet_size:.1f} MB)\n")
        f.write(f"  FITS:    {OUTPUT_FITS} ({fits_size:.1f} MB)\n\n")

        f.write("OUTPUT COLUMNS ADDED:\n")
        f.write("  - teff_predicted: Predicted effective temperature (K)\n")
        f.write("  - teff_expected_mae: Expected MAE based on temperature bin (K)\n")
        f.write("  - teff_expected_rmse: Expected RMSE based on temperature bin (K)\n")
        f.write("  - teff_prediction_std: Uncertainty from RF tree variance (K)\n")
        f.write("  - within_10pct_probability: Probability prediction within 10% (%)\n")
        f.write("  - temp_bin: Temperature classification\n")

    logger.info("Summary file written")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info(f"\nResults saved to:")
    logger.info(f"  Parquet: {parquet_path}")
    logger.info(f"  FITS:    {fits_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"\nTotal predictions: {len(y_pred):,} objects")
    logger.info(f"Success rate: {100*len(y_pred)/total_objects:.1f}%")


if __name__ == '__main__':
    main()
