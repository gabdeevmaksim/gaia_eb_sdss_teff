#!/usr/bin/env python3
"""
Merge temperature predictions with stars_types.dat catalog.

This script:
1. Loads stars_types.dat (2.1M eclipsing binaries)
2. Loads temperature predictions (optimized + full models)
3. Merges on Gaia source_id
4. Adds prediction columns to stars_types data
5. Saves as FITS file

Output columns:
- All original stars_types.dat columns
- teff_predicted (optimized model)
- teff_predicted_full (full model)
- teff_expected_mae (optimized)
- teff_expected_mae_full (full)
- model_difference
- Plus other prediction metadata

Author: Claude Code
Date: 2025-10-22
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from astropy.table import Table

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main merge workflow."""

    config = get_config()
    raw_dir = config.get_path('raw')
    processed_dir = config.get_path('processed')

    logger.info("="*80)
    logger.info("MERGE PREDICTIONS WITH STARS_TYPES.DAT")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")

    # File names
    STARS_TYPES_FILE = 'stars_types.dat'
    PREDICTIONS_FILE = 'colours_of_eb_stars_with_predictions.parquet'
    OUTPUT_FITS = 'stars_types_with_predictions.fits'
    OUTPUT_PARQUET = 'stars_types_with_predictions.parquet'
    SUMMARY_FILE = 'stars_types_with_predictions_SUMMARY.txt'

    logger.info(f"\nInput files:")
    logger.info(f"  Stars catalog: {STARS_TYPES_FILE}")
    logger.info(f"  Predictions:   {PREDICTIONS_FILE}")
    logger.info(f"\nOutput files:")
    logger.info(f"  FITS:    {OUTPUT_FITS}")
    logger.info(f"  Parquet: {OUTPUT_PARQUET}")

    # Step 1: Load stars_types.dat
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD STARS_TYPES.DAT")
    logger.info("="*80)

    stars_path = raw_dir / STARS_TYPES_FILE
    logger.info(f"\nLoading: {stars_path}")

    # Read CSV with proper handling
    df_stars = pd.read_csv(stars_path, dtype={'Name': str})

    # Rename 'Name' column to 'source_id' for merging
    df_stars.rename(columns={'Name': 'source_id'}, inplace=True)

    # Clean source_id: remove any suffixes like " [conflicted]"
    df_stars['source_id'] = df_stars['source_id'].astype(str).str.split(' ').str[0]

    # Convert source_id to int64
    df_stars['source_id'] = df_stars['source_id'].astype('int64')

    logger.info(f"Loaded {len(df_stars):,} objects")
    logger.info(f"Columns: {list(df_stars.columns)}")

    # Check Teff coverage in original file
    # Replace '--' with NaN
    df_stars['Teff'] = df_stars['Teff'].replace('--', np.nan)
    df_stars['Teff'] = pd.to_numeric(df_stars['Teff'], errors='coerce')

    n_with_teff = df_stars['Teff'].notna().sum()
    n_without_teff = df_stars['Teff'].isna().sum()
    pct_with_teff = 100 * n_with_teff / len(df_stars)

    logger.info(f"\nOriginal Teff coverage:")
    logger.info(f"  With Teff:    {n_with_teff:,} ({pct_with_teff:.1f}%)")
    logger.info(f"  Without Teff: {n_without_teff:,} ({100-pct_with_teff:.1f}%)")

    # Step 2: Load predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOAD PREDICTIONS")
    logger.info("="*80)

    pred_path = processed_dir / PREDICTIONS_FILE
    logger.info(f"\nLoading: {pred_path}")

    df_pred = pd.read_parquet(pred_path)
    logger.info(f"Loaded {len(df_pred):,} predictions")

    # Keep only the columns we want to add
    pred_cols = [
        'source_id',
        'teff_predicted',
        'teff_predicted_full',
        'teff_expected_mae',
        'teff_expected_mae_full',
        'teff_expected_rmse',
        'teff_expected_rmse_full',
        'teff_prediction_std',
        'teff_prediction_std_full',
        'within_10pct_probability',
        'within_10pct_probability_full',
        'temp_bin',
        'temp_bin_full',
        'model_difference'
    ]

    # Check which columns exist
    available_cols = [col for col in pred_cols if col in df_pred.columns]
    logger.info(f"\nPrediction columns to merge: {len(available_cols)}")
    for col in available_cols:
        logger.info(f"  - {col}")

    df_pred_subset = df_pred[available_cols].copy()

    # Step 3: Merge catalogs
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MERGE CATALOGS ON SOURCE_ID")
    logger.info("="*80)

    logger.info(f"\nMerging...")
    logger.info(f"  Stars catalog: {len(df_stars):,} objects")
    logger.info(f"  Predictions:   {len(df_pred_subset):,} objects")

    # Left join to keep all stars_types objects
    df_merged = df_stars.merge(df_pred_subset, on='source_id', how='left')

    logger.info(f"\nMerge result: {len(df_merged):,} objects")

    # Check merge statistics
    n_matched = df_merged['teff_predicted'].notna().sum()
    n_unmatched = df_merged['teff_predicted'].isna().sum()
    pct_matched = 100 * n_matched / len(df_merged)

    logger.info(f"\nMerge statistics:")
    logger.info(f"  Matched (have predictions):     {n_matched:,} ({pct_matched:.1f}%)")
    logger.info(f"  Unmatched (no predictions):     {n_unmatched:,} ({100-pct_matched:.1f}%)")

    # Step 4: Temperature coverage analysis
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TEMPERATURE COVERAGE ANALYSIS")
    logger.info("="*80)

    # Count objects with any temperature
    has_original = df_merged['Teff'].notna()
    has_optimized = df_merged['teff_predicted'].notna()
    has_full = df_merged['teff_predicted_full'].notna()
    has_any = has_original | has_optimized | has_full

    logger.info(f"\nTemperature coverage:")
    logger.info(f"  Original Teff only:     {(has_original & ~has_optimized).sum():,}")
    logger.info(f"  ML predictions only:    {(~has_original & has_optimized).sum():,}")
    logger.info(f"  Both original and ML:   {(has_original & has_optimized).sum():,}")
    logger.info(f"  Total with any Teff:    {has_any.sum():,} ({100*has_any.sum()/len(df_merged):.1f}%)")
    logger.info(f"  No temperature at all:  {(~has_any).sum():,}")

    # Create combined best estimate column
    logger.info(f"\nCreating combined temperature columns...")

    # teff_best: Use Gaia if available, otherwise optimized model
    df_merged['teff_best'] = df_merged['Teff'].fillna(df_merged['teff_predicted'])

    # teff_best_full: Use Gaia if available, otherwise full model
    df_merged['teff_best_full'] = df_merged['Teff'].fillna(df_merged['teff_predicted_full'])

    n_best = df_merged['teff_best'].notna().sum()
    n_best_full = df_merged['teff_best_full'].notna().sum()

    logger.info(f"  teff_best (Gaia or optimized): {n_best:,} ({100*n_best/len(df_merged):.1f}%)")
    logger.info(f"  teff_best_full (Gaia or full): {n_best_full:,} ({100*n_best_full/len(df_merged):.1f}%)")

    # Step 5: Statistics
    logger.info("\n" + "="*80)
    logger.info("STEP 5: CALCULATE STATISTICS")
    logger.info("="*80)

    logger.info(f"\nBest temperature statistics (teff_best):")
    teff_best_values = df_merged['teff_best'].dropna()
    logger.info(f"  Count:  {len(teff_best_values):,}")
    logger.info(f"  Mean:   {teff_best_values.mean():.1f} K")
    logger.info(f"  Median: {teff_best_values.median():.1f} K")
    logger.info(f"  Std:    {teff_best_values.std():.1f} K")
    logger.info(f"  Min:    {teff_best_values.min():.1f} K")
    logger.info(f"  Max:    {teff_best_values.max():.1f} K")

    # Binary type statistics
    logger.info(f"\nTemperature coverage by binary type:")
    for btype in df_merged['Type'].unique():
        if pd.notna(btype):
            mask = df_merged['Type'] == btype
            n_type = mask.sum()
            n_teff = (mask & has_any).sum()
            pct = 100 * n_teff / n_type if n_type > 0 else 0
            logger.info(f"  {btype:12s}: {n_teff:7,} / {n_type:7,} ({pct:5.1f}%)")

    # Step 6: Save Parquet
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVE PARQUET FILE")
    logger.info("="*80)

    parquet_path = processed_dir / OUTPUT_PARQUET
    logger.info(f"\nSaving: {parquet_path}")

    df_merged.to_parquet(parquet_path, index=False)
    parquet_size = parquet_path.stat().st_size / 1e6

    logger.info(f"Saved: {parquet_size:.1f} MB")
    logger.info(f"Rows: {len(df_merged):,}")
    logger.info(f"Columns: {len(df_merged.columns)}")

    # Step 7: Save FITS
    logger.info("\n" + "="*80)
    logger.info("STEP 7: SAVE FITS FILE")
    logger.info("="*80)

    fits_path = processed_dir / OUTPUT_FITS
    logger.info(f"\nSaving: {fits_path}")

    # Convert to Astropy Table
    # Handle string columns and NaN values properly
    df_for_fits = df_merged.copy()

    # Replace NaN with appropriate fill values for FITS
    for col in df_for_fits.columns:
        if df_for_fits[col].dtype == 'object':
            df_for_fits[col] = df_for_fits[col].fillna('--')
        elif df_for_fits[col].dtype in ['float64', 'float32']:
            # Keep NaN as is, FITS will handle it
            pass

    output_table = Table.from_pandas(df_for_fits)

    # Add metadata
    output_table.meta['COMMENT'] = f'Stars_types.dat merged with ML temperature predictions on {datetime.now().isoformat()}'
    output_table.meta['NSTARS'] = len(df_merged)
    output_table.meta['NTEFF'] = int(has_any.sum())
    output_table.meta['NORIGINAL'] = int(has_original.sum())
    output_table.meta['NMLPRED'] = int(has_optimized.sum())
    output_table.meta['MODEL1'] = 'rf_gaia_g_bprp_optimized_20251022_091954'
    output_table.meta['MODEL2'] = 'rf_gaia_g_bprp_full_20251022_135438'

    # Write FITS
    output_table.write(fits_path, format='fits', overwrite=True)
    fits_size = fits_path.stat().st_size / 1e6

    logger.info(f"Saved: {fits_size:.1f} MB")

    # Step 8: Save summary
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SAVE SUMMARY")
    logger.info("="*80)

    summary_path = processed_dir / SUMMARY_FILE
    logger.info(f"\nWriting: {summary_path}")

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Stars Types with ML Temperature Predictions\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("INPUT FILES:\n")
        f.write(f"  Catalog: {STARS_TYPES_FILE}\n")
        f.write(f"  Total objects: {len(df_stars):,}\n")
        f.write(f"  Original Teff coverage: {pct_with_teff:.1f}%\n\n")

        f.write(f"  Predictions: {PREDICTIONS_FILE}\n")
        f.write(f"  Total predictions: {len(df_pred):,}\n\n")

        f.write("MERGE RESULTS:\n")
        f.write(f"  Total objects: {len(df_merged):,}\n")
        f.write(f"  Matched (have predictions): {n_matched:,} ({pct_matched:.1f}%)\n")
        f.write(f"  Unmatched (no predictions): {n_unmatched:,} ({100-pct_matched:.1f}%)\n\n")

        f.write("TEMPERATURE COVERAGE:\n")
        f.write(f"  Original Teff only:     {(has_original & ~has_optimized).sum():,}\n")
        f.write(f"  ML predictions only:    {(~has_original & has_optimized).sum():,}\n")
        f.write(f"  Both original and ML:   {(has_original & has_optimized).sum():,}\n")
        f.write(f"  Total with any Teff:    {has_any.sum():,} ({100*has_any.sum()/len(df_merged):.1f}%)\n")
        f.write(f"  No temperature:         {(~has_any).sum():,}\n\n")

        f.write("COMBINED TEMPERATURE COLUMNS:\n")
        f.write(f"  teff_best: Gaia Teff if available, else optimized model\n")
        f.write(f"  teff_best_full: Gaia Teff if available, else full model\n")
        f.write(f"  Coverage: {100*n_best/len(df_merged):.1f}%\n\n")

        f.write("TEMPERATURE STATISTICS (teff_best):\n")
        f.write(f"  Count:  {len(teff_best_values):,}\n")
        f.write(f"  Mean:   {teff_best_values.mean():.1f} K\n")
        f.write(f"  Median: {teff_best_values.median():.1f} K\n")
        f.write(f"  Std:    {teff_best_values.std():.1f} K\n")
        f.write(f"  Min:    {teff_best_values.min():.1f} K\n")
        f.write(f"  Max:    {teff_best_values.max():.1f} K\n\n")

        f.write("COVERAGE BY BINARY TYPE:\n")
        for btype in sorted(df_merged['Type'].unique()):
            if pd.notna(btype):
                mask = df_merged['Type'] == btype
                n_type = mask.sum()
                n_teff = (mask & has_any).sum()
                pct = 100 * n_teff / n_type if n_type > 0 else 0
                f.write(f"  {btype:12s}: {n_teff:7,} / {n_type:7,} ({pct:5.1f}%)\n")

        f.write("\nMODELS USED:\n")
        f.write(f"  Optimized: rf_gaia_g_bprp_optimized_20251022_091954\n")
        f.write(f"    Test MAE: 659.5 K, RMSE: 1158.2 K, R²: 0.346\n")
        f.write(f"  Full:      rf_gaia_g_bprp_full_20251022_135438\n")
        f.write(f"    Test MAE: 665.2 K, RMSE: 1208.2 K, R²: 0.504\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  FITS:    {OUTPUT_FITS} ({fits_size:.1f} MB)\n")
        f.write(f"  Parquet: {OUTPUT_PARQUET} ({parquet_size:.1f} MB)\n\n")

        f.write("COLUMNS ADDED:\n")
        for col in available_cols:
            if col != 'source_id':
                f.write(f"  - {col}\n")
        f.write(f"  - teff_best (combined: Gaia or optimized)\n")
        f.write(f"  - teff_best_full (combined: Gaia or full)\n")

    logger.info("Summary written")

    # Final message
    logger.info("\n" + "="*80)
    logger.info("MERGE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info(f"\nOutput files:")
    logger.info(f"  FITS:    {fits_path} ({fits_size:.1f} MB)")
    logger.info(f"  Parquet: {parquet_path} ({parquet_size:.1f} MB)")
    logger.info(f"  Summary: {summary_path}")

    logger.info(f"\nFinal statistics:")
    logger.info(f"  Total objects: {len(df_merged):,}")
    logger.info(f"  With temperature: {has_any.sum():,} ({100*has_any.sum()/len(df_merged):.1f}%)")
    logger.info(f"  Coverage improvement: +{100*(has_any.sum() - n_with_teff)/len(df_merged):.1f}%")


if __name__ == '__main__':
    main()
