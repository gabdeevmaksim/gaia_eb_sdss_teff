#!/usr/bin/env python3
"""
Merge stars_types.dat catalog with predicted temperatures.

This script:
1. Loads stars_types.dat and adds teff_flag (0=absent, 1=original)
2. Loads eb_full_catalog_temperatures.parquet and adds teff_flag=2 (predicted)
3. Merges on Gaia source_id
4. Combines teff columns: prefers original (flag 1), then predicted (flag 2), then absent (flag 0)
5. Saves enhanced catalog with teff_flag column

Usage:
    python scripts/merge_stars_types_with_predictions.py

    # Specify custom files
    python scripts/merge_stars_types_with_predictions.py \
        --stars-types data/raw/stars_types.dat \
        --predictions data/processed/eb_full_catalog_temperatures.parquet \
        --output data/processed/stars_types_with_predictions.parquet

Author: Auto-generated
Date: 2025-10-14
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def load_stars_types_with_flag(stars_types_file: Path) -> pd.DataFrame:
    """
    Load stars_types.dat and add teff_flag column.

    teff_flag:
    - 0: Teff is absent (value is '--')
    - 1: Teff is original (value is not '--')

    Parameters
    ----------
    stars_types_file : Path
        Path to stars_types.dat

    Returns
    -------
    pd.DataFrame
        Stars types catalog with teff_flag column
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading stars_types.dat from {stars_types_file.name}...")

    # Load CSV with Name as string to handle large integers
    df = pd.read_csv(stars_types_file, dtype={'Name': str}, low_memory=False)

    logger.info(f"  Loaded {len(df):,} stars")
    logger.info(f"  Columns: {list(df.columns)}")

    # Add teff_flag column
    logger.info("  Adding teff_flag column...")

    # Check for '--' in Teff column
    df['teff_flag'] = np.where(df['Teff'] == '--', 0, 1)

    n_absent = (df['teff_flag'] == 0).sum()
    n_original = (df['teff_flag'] == 1).sum()

    logger.info(f"  Teff absent (flag 0): {n_absent:,} ({n_absent/len(df)*100:.1f}%)")
    logger.info(f"  Teff original (flag 1): {n_original:,} ({n_original/len(df)*100:.1f}%)")

    # Convert Teff to numeric (replace '--' with NaN)
    df['Teff_original'] = pd.to_numeric(df['Teff'], errors='coerce')

    return df


def load_predictions_with_flag(predictions_file: Path) -> pd.DataFrame:
    """
    Load predicted temperatures and add teff_flag=2 for all rows.

    Parameters
    ----------
    predictions_file : Path
        Path to eb_full_catalog_temperatures.parquet

    Returns
    -------
    pd.DataFrame
        Predictions with teff_flag=2
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\nLoading predictions from {predictions_file.name}...")

    df = pd.read_parquet(predictions_file)

    logger.info(f"  Loaded {len(df):,} predictions")
    logger.info(f"  Columns: {list(df.columns)}")

    # Add teff_flag=2 for all predicted temperatures
    df['teff_flag'] = 2

    # Check what temperature column exists
    teff_cols = [col for col in df.columns if 'teff' in col.lower() and 'flag' not in col.lower()]
    logger.info(f"  Temperature columns found: {teff_cols}")

    # Rename to Teff_predicted for clarity
    if 'teff_predicted' in df.columns:
        df = df.rename(columns={'teff_predicted': 'Teff_predicted'})
    elif 'predicted_teff' in df.columns:
        df = df.rename(columns={'predicted_teff': 'Teff_predicted'})
    elif 'teff' in df.columns:
        df = df.rename(columns={'teff': 'Teff_predicted'})
    elif 'teff_gspphot' in df.columns:
        df = df.rename(columns={'teff_gspphot': 'Teff_predicted'})

    if 'Teff_predicted' not in df.columns:
        logger.error(f"  Could not find temperature column! Available: {list(df.columns)}")
        raise ValueError("No temperature column found in predictions")

    logger.info(f"  Renamed temperature column to Teff_predicted")
    logger.info(f"  All rows flagged as predicted (flag 2)")

    return df


def merge_and_combine(
    stars_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_file: Path
) -> pd.DataFrame:
    """
    Merge stars_types with predictions and combine teff columns.

    Priority:
    1. Original Teff (flag 1)
    2. Predicted Teff (flag 2)
    3. Absent (flag 0)

    Parameters
    ----------
    stars_df : pd.DataFrame
        Stars types with teff_flag (0 or 1)
    predictions_df : pd.DataFrame
        Predictions with teff_flag=2
    output_file : Path
        Output parquet file

    Returns
    -------
    pd.DataFrame
        Merged catalog with combined teff and teff_flag
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("MERGING CATALOGS")
    logger.info("=" * 70)

    # Merge on Name (Gaia source_id)
    logger.info("\nMerging on Gaia source_id...")
    logger.info(f"  Stars catalog: {len(stars_df):,} sources")
    logger.info(f"  Predictions: {len(predictions_df):,} sources")

    # Check if Name column exists in predictions
    if 'Name' not in predictions_df.columns:
        # Try to find source_id column
        id_cols = [col for col in predictions_df.columns if 'source' in col.lower() and 'id' in col.lower()]
        if id_cols:
            logger.info(f"  Renaming {id_cols[0]} to Name in predictions...")
            predictions_df = predictions_df.rename(columns={id_cols[0]: 'Name'})
        else:
            logger.error("  Cannot find source_id column in predictions!")
            raise ValueError("No source_id column found in predictions")

    # Convert Name to string to match stars_types
    predictions_df['Name'] = predictions_df['Name'].astype(str)

    # Merge
    merged = stars_df.merge(
        predictions_df[['Name', 'Teff_predicted', 'teff_flag']],
        on='Name',
        how='left',
        suffixes=('_stars', '_pred')
    )

    logger.info(f"  Merged catalog: {len(merged):,} sources")

    # Count matches
    n_with_predictions = merged['Teff_predicted'].notna().sum()
    logger.info(f"  Sources with predictions: {n_with_predictions:,} ({n_with_predictions/len(merged)*100:.1f}%)")

    # Combine teff columns with priority: original > predicted > absent
    logger.info("\nCombining temperature columns...")

    # Initialize final columns
    merged['Teff_final'] = np.nan
    merged['teff_flag_final'] = 0

    # Priority 1: Original Teff (flag 1)
    mask_original = merged['teff_flag_stars'] == 1
    merged.loc[mask_original, 'Teff_final'] = merged.loc[mask_original, 'Teff_original']
    merged.loc[mask_original, 'teff_flag_final'] = 1

    # Priority 2: Predicted Teff (flag 2) - only where original is absent
    mask_predicted = (merged['teff_flag_stars'] == 0) & (merged['Teff_predicted'].notna())
    merged.loc[mask_predicted, 'Teff_final'] = merged.loc[mask_predicted, 'Teff_predicted']
    merged.loc[mask_predicted, 'teff_flag_final'] = 2

    # Priority 3: Absent (flag 0) - remains as NaN
    mask_absent = (merged['teff_flag_stars'] == 0) & (merged['Teff_predicted'].isna())
    merged.loc[mask_absent, 'teff_flag_final'] = 0

    # Statistics
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEMPERATURE STATISTICS")
    logger.info("=" * 70)

    n_flag_0 = (merged['teff_flag_final'] == 0).sum()
    n_flag_1 = (merged['teff_flag_final'] == 1).sum()
    n_flag_2 = (merged['teff_flag_final'] == 2).sum()

    logger.info(f"\nTemperature sources:")
    logger.info(f"  Flag 0 (absent): {n_flag_0:,} ({n_flag_0/len(merged)*100:.1f}%)")
    logger.info(f"  Flag 1 (original): {n_flag_1:,} ({n_flag_1/len(merged)*100:.1f}%)")
    logger.info(f"  Flag 2 (predicted): {n_flag_2:,} ({n_flag_2/len(merged)*100:.1f}%)")
    logger.info(f"  Total with Teff: {n_flag_1 + n_flag_2:,} ({(n_flag_1 + n_flag_2)/len(merged)*100:.1f}%)")

    # Round Teff_final to 1 decimal place
    logger.info("\nRounding temperatures to 1 decimal place...")
    merged['Teff_final'] = merged['Teff_final'].round(1)

    # Drop intermediate columns BEFORE renaming (including original Teff)
    cols_to_drop = ['Teff', 'Teff_original', 'Teff_predicted', 'teff_flag_stars', 'teff_flag_pred']
    merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])

    # Now rename final columns to Teff
    merged = merged.rename(columns={
        'Teff_final': 'Teff',
        'teff_flag_final': 'teff_flag'
    })

    # Reorder columns to place Teff and teff_flag right after Period
    logger.info("Reordering columns (Teff and teff_flag after Period)...")

    # Get all columns
    cols = list(merged.columns)

    # Remove Teff and teff_flag from their current positions
    cols.remove('Teff')
    cols.remove('teff_flag')

    # Find Period column position
    if 'Period' in cols:
        period_idx = cols.index('Period')
        # Insert Teff and teff_flag right after Period
        cols.insert(period_idx + 1, 'Teff')
        cols.insert(period_idx + 2, 'teff_flag')
    else:
        # If no Period column, put at the beginning after Name
        cols.insert(1, 'Teff')
        cols.insert(2, 'teff_flag')

    # Reorder dataframe
    merged = merged[cols]

    # Save
    logger.info(f"\nSaving enhanced catalog to {output_file}...")
    merged.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Rows: {len(merged):,}")
    logger.info(f"  Columns: {len(merged.columns)}")

    # Temperature range
    if merged['Teff'].notna().any():
        logger.info(f"\nTemperature range:")
        logger.info(f"  Min: {merged['Teff'].min():.1f} K")
        logger.info(f"  Max: {merged['Teff'].max():.1f} K")
        logger.info(f"  Mean: {merged['Teff'].mean():.1f} K")
        logger.info(f"  Median: {merged['Teff'].median():.1f} K")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge stars_types.dat with predicted temperatures',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--stars-types',
        type=str,
        default=None,
        help='Path to stars_types.dat (default: data/raw/stars_types.dat)'
    )

    parser.add_argument(
        '--predictions',
        type=str,
        default=None,
        help='Path to predictions parquet (default: data/processed/eb_full_catalog_temperatures.parquet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet file (default: data/processed/stars_types_with_predictions.parquet)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config()

    # Determine input files
    if args.stars_types:
        stars_types_file = Path(args.stars_types)
    else:
        stars_types_file = config.project_root / 'data' / 'raw' / 'stars_types.dat'

    if args.predictions:
        predictions_file = Path(args.predictions)
    else:
        predictions_file = config.get_dataset_path('eb_full_catalog_temps', 'processed')

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.project_root / 'data' / 'processed' / 'stars_types_with_predictions.parquet'

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("MERGE STARS_TYPES.DAT WITH PREDICTED TEMPERATURES")
    logger.info("=" * 70)
    logger.info(f"\nStars types catalog: {stars_types_file}")
    logger.info(f"Predictions: {predictions_file}")
    logger.info(f"Output: {output_file}")
    logger.info("")

    # Load and process
    stars_df = load_stars_types_with_flag(stars_types_file)
    predictions_df = load_predictions_with_flag(predictions_file)
    merged_df = merge_and_combine(stars_df, predictions_df, output_file)

    logger.info("\n" + "=" * 70)
    logger.info("MERGE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("\nteff_flag values:")
    logger.info("  0 = Temperature absent (no original, no prediction)")
    logger.info("  1 = Original temperature from stars_types.dat")
    logger.info("  2 = Predicted temperature from ML model")


if __name__ == '__main__':
    main()
