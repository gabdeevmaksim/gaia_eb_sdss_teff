#!/usr/bin/env python3
"""
Create full Gaia G + BP-RP training dataset from colours_of_eb_stars-result.fits

This script creates a training dataset using ALL objects with Gaia temperatures,
not limited to those with Pan-STARRS photometry.

Improvement over previous dataset:
- Previous: 737,017 objects (Pan-STARRS crossmatch only)
- New: ~1,275,924 objects (all with Gaia temperatures)
- Increase: +73.1%

Features:
- phot_g_mean_mag (from g column)
- bp_rp
Target:
- teff_gaia

Quality filters applied:
- BP-RP: [-0.5, 4.0]
- G mag: [5.0, 22.0]
- Teff: [2500, 50000] K

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
    """Create full training dataset."""

    config = get_config()
    raw_dir = config.get_path('raw')
    processed_dir = config.get_path('processed')

    logger.info("="*80)
    logger.info("CREATE FULL GAIA G + BP-RP TRAINING DATASET")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")

    # Input/output files
    INPUT_FILE = 'colours_of_eb_stars-result.fits'
    OUTPUT_FILE = 'gaia_g_bprp_full_train.parquet'
    SUMMARY_FILE = 'gaia_g_bprp_full_train_SUMMARY.txt'

    logger.info(f"\nInput: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")

    # Step 1: Load FITS file
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD FITS DATA")
    logger.info("="*80)

    input_path = raw_dir / INPUT_FILE
    logger.info(f"\nLoading: {input_path}")

    table = Table.read(input_path)
    df = table.to_pandas()

    logger.info(f"Total objects in FITS: {len(df):,}")
    logger.info(f"Columns: {', '.join(df.columns)}")

    # Step 2: Filter for objects with Gaia temperature
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FILTER FOR OBJECTS WITH GAIA TEMPERATURE")
    logger.info("="*80)

    n_total = len(df)
    df_with_teff = df[df['teff_gaia'].notna()].copy()
    n_with_teff = len(df_with_teff)

    logger.info(f"\nObjects with teff_gaia: {n_with_teff:,} ({100*n_with_teff/n_total:.1f}%)")
    logger.info(f"Objects without teff_gaia: {n_total - n_with_teff:,} ({100*(n_total - n_with_teff)/n_total:.1f}%)")

    # Step 3: Check required features
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CHECK REQUIRED FEATURES")
    logger.info("="*80)

    required_cols = ['g', 'bp_rp', 'teff_gaia']
    logger.info(f"\nRequired features: {', '.join(required_cols)}")

    for col in required_cols:
        n_valid = df_with_teff[col].notna().sum()
        n_missing = df_with_teff[col].isna().sum()
        pct_valid = 100 * n_valid / len(df_with_teff)
        logger.info(f"  {col:15s}: {n_valid:,} valid ({pct_valid:.2f}%), {n_missing:,} missing")

    # Remove any objects with missing features
    n_before_dropna = len(df_with_teff)
    df_complete = df_with_teff.dropna(subset=required_cols).copy()
    n_after_dropna = len(df_complete)
    n_dropped = n_before_dropna - n_after_dropna

    if n_dropped > 0:
        logger.info(f"\nRemoved {n_dropped:,} objects with missing features")
    logger.info(f"Objects with all features: {n_after_dropna:,}")

    # Step 4: Apply quality filters
    logger.info("\n" + "="*80)
    logger.info("STEP 4: APPLY QUALITY FILTERS")
    logger.info("="*80)

    # Define quality ranges (same as optimized model)
    quality_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g': (5.0, 22.0),
        'teff_gaia': (2500, 50000)
    }

    logger.info("\nQuality filter ranges:")
    for col, (min_val, max_val) in quality_ranges.items():
        logger.info(f"  {col:15s}: [{min_val}, {max_val}]")

    n_before_filters = len(df_complete)
    df_clean = df_complete.copy()

    logger.info("\nApplying filters:")
    for col, (min_val, max_val) in quality_ranges.items():
        mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
        n_pass = mask.sum()
        n_fail = (~mask).sum()
        logger.info(f"  {col:15s}: {n_pass:,} pass, {n_fail:,} fail")
        df_clean = df_clean[mask].copy()

    n_after_filters = len(df_clean)
    n_removed = n_before_filters - n_after_filters

    logger.info(f"\nFinal dataset after filters: {n_after_filters:,}")
    logger.info(f"Total removed by filters: {n_removed:,} ({100*n_removed/n_before_filters:.2f}%)")

    # Step 5: Prepare training dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 5: PREPARE TRAINING DATASET")
    logger.info("="*80)

    # Rename columns to match model expectations
    logger.info("\nMapping columns:")
    logger.info("  g -> phot_g_mean_mag")
    logger.info("  bp_rp -> bp_rp (keep)")
    logger.info("  teff_gaia -> teff_gaia (target)")

    df_train = pd.DataFrame({
        'source_id': df_clean['source_id'],
        'phot_g_mean_mag': df_clean['g'],
        'bp_rp': df_clean['bp_rp'],
        'teff_gaia': df_clean['teff_gaia']
    })

    logger.info(f"\nTraining dataset shape: {df_train.shape}")
    logger.info(f"Columns: {list(df_train.columns)}")

    # Step 6: Dataset statistics
    logger.info("\n" + "="*80)
    logger.info("STEP 6: DATASET STATISTICS")
    logger.info("="*80)

    logger.info(f"\nTarget (teff_gaia) statistics:")
    logger.info(f"  Min:    {df_train['teff_gaia'].min():.1f} K")
    logger.info(f"  Max:    {df_train['teff_gaia'].max():.1f} K")
    logger.info(f"  Mean:   {df_train['teff_gaia'].mean():.1f} K")
    logger.info(f"  Median: {df_train['teff_gaia'].median():.1f} K")
    logger.info(f"  Std:    {df_train['teff_gaia'].std():.1f} K")

    logger.info(f"\nFeature (phot_g_mean_mag) statistics:")
    logger.info(f"  Min:    {df_train['phot_g_mean_mag'].min():.3f}")
    logger.info(f"  Max:    {df_train['phot_g_mean_mag'].max():.3f}")
    logger.info(f"  Mean:   {df_train['phot_g_mean_mag'].mean():.3f}")
    logger.info(f"  Median: {df_train['phot_g_mean_mag'].median():.3f}")

    logger.info(f"\nFeature (bp_rp) statistics:")
    logger.info(f"  Min:    {df_train['bp_rp'].min():.3f}")
    logger.info(f"  Max:    {df_train['bp_rp'].max():.3f}")
    logger.info(f"  Mean:   {df_train['bp_rp'].mean():.3f}")
    logger.info(f"  Median: {df_train['bp_rp'].median():.3f}")

    # Temperature distribution
    temp_bins = [0, 4000, 5000, 6000, 7000, 10000, 50000]
    temp_labels = ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']
    df_train['temp_bin'] = pd.cut(df_train['teff_gaia'], bins=temp_bins, labels=temp_labels)

    logger.info(f"\nTemperature distribution:")
    for label in temp_labels:
        count = (df_train['temp_bin'] == label).sum()
        pct = 100 * count / len(df_train)
        logger.info(f"  {label:15s}: {count:7,} objects ({pct:5.1f}%)")

    # Drop temp_bin column before saving
    df_train = df_train.drop('temp_bin', axis=1)

    # Step 7: Compare with previous dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 7: COMPARISON WITH PREVIOUS DATASET")
    logger.info("="*80)

    try:
        prev_train_path = processed_dir / 'gaia_g_bprp_train.parquet'
        if prev_train_path.exists():
            df_prev = pd.read_parquet(prev_train_path)
            n_prev = len(df_prev)
            n_new = len(df_train)
            increase = n_new - n_prev
            pct_increase = 100 * increase / n_prev

            logger.info(f"\nPrevious dataset: {n_prev:,} objects")
            logger.info(f"New dataset:      {n_new:,} objects")
            logger.info(f"Increase:         +{increase:,} objects (+{pct_increase:.1f}%)")

            # Compare temperature distributions
            logger.info(f"\nMean temperature:")
            logger.info(f"  Previous: {df_prev['teff_gspphot'].mean():.1f} K")
            logger.info(f"  New:      {df_train['teff_gaia'].mean():.1f} K")
            logger.info(f"  Difference: {df_train['teff_gaia'].mean() - df_prev['teff_gspphot'].mean():.1f} K")
        else:
            logger.info("\nNo previous dataset found for comparison")
    except Exception as e:
        logger.warning(f"\nCould not compare with previous dataset: {e}")

    # Step 8: Save dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SAVE DATASET")
    logger.info("="*80)

    output_path = processed_dir / OUTPUT_FILE
    logger.info(f"\nSaving to: {output_path}")

    df_train.to_parquet(output_path, index=False)
    file_size = output_path.stat().st_size / 1e6

    logger.info(f"Saved: {file_size:.1f} MB")
    logger.info(f"Rows: {len(df_train):,}")
    logger.info(f"Columns: {len(df_train.columns)}")

    # Step 9: Save summary
    logger.info("\n" + "="*80)
    logger.info("STEP 9: SAVE SUMMARY")
    logger.info("="*80)

    summary_path = processed_dir / SUMMARY_FILE
    logger.info(f"\nWriting summary to: {summary_path}")

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Gaia G + BP-RP Full Training Dataset\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Source: {INPUT_FILE}\n\n")

        f.write("DATASET SIZE:\n")
        f.write(f"  Total objects in FITS: {n_total:,}\n")
        f.write(f"  Objects with Gaia temperature: {n_with_teff:,}\n")
        f.write(f"  Final training dataset: {len(df_train):,}\n")
        f.write(f"  File size: {file_size:.1f} MB\n\n")

        f.write("FEATURES:\n")
        f.write("  - phot_g_mean_mag (Gaia G magnitude)\n")
        f.write("  - bp_rp (Gaia BP-RP color)\n\n")

        f.write("TARGET:\n")
        f.write("  - teff_gaia (Gaia GSP-Phot effective temperature)\n\n")

        f.write("QUALITY FILTERS APPLIED:\n")
        for col, (min_val, max_val) in quality_ranges.items():
            f.write(f"  {col:15s}: [{min_val}, {max_val}]\n")
        f.write(f"  Removed by filters: {n_removed:,}\n\n")

        f.write("TEMPERATURE STATISTICS:\n")
        f.write(f"  Min:    {df_train['teff_gaia'].min():.1f} K\n")
        f.write(f"  Max:    {df_train['teff_gaia'].max():.1f} K\n")
        f.write(f"  Mean:   {df_train['teff_gaia'].mean():.1f} K\n")
        f.write(f"  Median: {df_train['teff_gaia'].median():.1f} K\n")
        f.write(f"  Std:    {df_train['teff_gaia'].std():.1f} K\n\n")

        f.write("TEMPERATURE DISTRIBUTION:\n")
        df_train['temp_bin'] = pd.cut(df_train['teff_gaia'], bins=temp_bins, labels=temp_labels)
        for label in temp_labels:
            count = (df_train['temp_bin'] == label).sum()
            pct = 100 * count / len(df_train)
            f.write(f"  {label:15s}: {count:7,} objects ({pct:5.1f}%)\n")

        if prev_train_path.exists():
            f.write(f"\nCOMPARISON WITH PREVIOUS DATASET:\n")
            f.write(f"  Previous: {n_prev:,} objects\n")
            f.write(f"  New:      {n_new:,} objects\n")
            f.write(f"  Increase: +{increase:,} objects (+{pct_increase:.1f}%)\n")

    logger.info("Summary written")

    # Final message
    logger.info("\n" + "="*80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Dataset: {output_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"\nReady for model training!")


if __name__ == '__main__':
    main()
