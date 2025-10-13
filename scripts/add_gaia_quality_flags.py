#!/usr/bin/env python3
"""
Add Gaia GSP-Phot temperature quality flags to eclipsing binary catalog.

This script cross-matches eclipsing binary sources with Gaia GSP-Phot quality
flags from FITS files. The quality flags (0 or 1) indicate temperature quality
and can be used to filter training data for machine learning models.

Quality flag meanings:
- 0: Lower quality temperature estimate
- 1: Higher quality temperature estimate

Algorithm:
1. Load eclipsing binary catalog (~700k sources)
2. Create a set of source_ids to search for
3. For each FITS file, load data and find matches
4. Update dataframe with quality flags

This approach is much faster than loading all 470M FITS rows into memory.

Usage:
    # Add quality flags to ML training data
    python scripts/add_gaia_quality_flags.py

    # Specify input dataset
    python scripts/add_gaia_quality_flags.py --input data/processed/ml_training_data_with_gaia.parquet

    # Specify output file
    python scripts/add_gaia_quality_flags.py --output data/processed/ml_training_data_with_quality_flags.parquet

Author: Auto-generated
Date: 2025-10-09
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Set
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from astropy.io import fits

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def search_fits_for_matches(
    fits_dir: Path,
    source_ids_to_find: Set[int]
) -> Dict[int, int]:
    """
    Search FITS files for matching source_ids and extract quality flags.
    
    This is much more efficient than loading all FITS data into memory.
    We only keep the source_ids that match our eclipsing binary catalog.
    
    Parameters
    ----------
    fits_dir : Path
        Directory containing Gaia quality flag FITS files
    source_ids_to_find : set of int
        Set of source_ids to search for
        
    Returns
    -------
    dict
        Dictionary mapping source_id to quality flag (0 or 1)
    """
    logger = logging.getLogger(__name__)
    
    fits_files = sorted(fits_dir.glob("*.fits"))
    
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {fits_dir}")
    
    logger.info(f"Searching {len(fits_files)} FITS files for {len(source_ids_to_find):,} source_ids...")
    
    quality_flags = {}
    total_processed = 0
    
    for i, fits_file in enumerate(fits_files, 1):
        logger.info(f"[{i}/{len(fits_files)}] Reading {fits_file.name}...")
        
        with fits.open(fits_file) as hdul:
            data = hdul[1].data
            n_rows = len(data)
            
            logger.info(f"  Total rows in file: {n_rows:,}")
            logger.info(f"  Searching for matches...")
            
            # Get source_ids and flags
            source_ids = data['source_id']
            flags = data['flag']
            
            # Find matches efficiently using set intersection
            matches_found = 0
            for source_id, flag in zip(source_ids, flags):
                if source_id in source_ids_to_find:
                    quality_flags[int(source_id)] = int(flag)
                    matches_found += 1
            
            total_processed += n_rows
            logger.info(f"  Found {matches_found:,} matches")
            logger.info(f"  Total matches so far: {len(quality_flags):,}/{len(source_ids_to_find):,}")
    
    logger.info(f"\nProcessed {total_processed:,} FITS rows total")
    logger.info(f"Found matches for {len(quality_flags):,}/{len(source_ids_to_find):,} sources ({len(quality_flags)/len(source_ids_to_find)*100:.1f}%)")
    
    # Calculate statistics
    if quality_flags:
        n_flag_0 = sum(1 for f in quality_flags.values() if f == 0)
        n_flag_1 = sum(1 for f in quality_flags.values() if f == 1)
        
        logger.info(f"\nQuality flag distribution:")
        logger.info(f"  Flag 0 (lower quality): {n_flag_0:,} ({n_flag_0/len(quality_flags)*100:.1f}%)")
        logger.info(f"  Flag 1 (higher quality): {n_flag_1:,} ({n_flag_1/len(quality_flags)*100:.1f}%)")
    
    return quality_flags


def load_catalog_and_add_flags(
    input_file: Path,
    output_file: Path,
    fits_dir: Path
) -> pd.DataFrame:
    """
    Load eclipsing binary catalog, search for quality flags, and save results.
    
    Parameters
    ----------
    input_file : Path
        Input Parquet file with eclipsing binary data
    output_file : Path
        Output Parquet file with added quality flags
    fits_dir : Path
        Directory containing Gaia quality flag FITS files
        
    Returns
    -------
    pd.DataFrame
        Updated dataframe with quality flag column
    """
    logger = logging.getLogger(__name__)
    config = get_config()
    
    logger.info(f"Loading eclipsing binary data from {input_file.name}...")
    df = pd.read_parquet(input_file)
    
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Check if we have Gaia source_id column
    if 'source_id' not in df.columns:
        logger.warning("  'source_id' (Gaia) column not found in input data!")
        logger.info("  Loading original catalog to get Gaia source_id...")
        
        # Load original catalog with source_id
        catalog_file = config.project_root / 'data' / 'processed' / 'eb_catalog.parquet'
        catalog = pd.read_parquet(catalog_file, columns=['original_ext_source_id', 'source_id'])
        
        logger.info(f"  Loaded {len(catalog):,} objects from original catalog")
        logger.info(f"  Merging on 'original_ext_source_id' to add Gaia source_id...")
        
        # Merge to add source_id
        df = df.merge(catalog, on='original_ext_source_id', how='left')
        
        n_with_source_id = df['source_id'].notna().sum()
        logger.info(f"  Successfully added source_id to {n_with_source_id:,}/{len(df):,} objects")
    
    # Create set of Gaia source_ids to search for
    logger.info("\nPreparing Gaia source_ids for cross-match...")
    source_ids = df['source_id'].dropna().astype(np.int64)
    source_ids_set = set(source_ids.values)
    logger.info(f"  Unique Gaia source_ids to find: {len(source_ids_set):,}")
    
    # Search FITS files for matches
    quality_flags = search_fits_for_matches(fits_dir, source_ids_set)
    
    # Add quality flags to dataframe
    logger.info("\nAdding quality flags to dataframe...")
    # Map using Gaia source_id
    df['teff_gspphot_flag'] = df['source_id'].map(quality_flags).fillna(-1).astype(int)
    
    # Statistics
    n_matched = (df['teff_gspphot_flag'] >= 0).sum()
    n_no_match = (df['teff_gspphot_flag'] == -1).sum()
    n_flag_0 = (df['teff_gspphot_flag'] == 0).sum()
    n_flag_1 = (df['teff_gspphot_flag'] == 1).sum()
    
    logger.info(f"\nFinal cross-match results:")
    logger.info(f"  Matched: {n_matched:,} ({n_matched/len(df)*100:.1f}%)")
    logger.info(f"  No match: {n_no_match:,} ({n_no_match/len(df)*100:.1f}%)")
    
    if n_matched > 0:
        logger.info(f"\nQuality flag distribution (matched sources only):")
        logger.info(f"  Flag 0 (lower quality): {n_flag_0:,} ({n_flag_0/n_matched*100:.1f}%)")
        logger.info(f"  Flag 1 (higher quality): {n_flag_1:,} ({n_flag_1/n_matched*100:.1f}%)")
    
    # Save updated dataset
    logger.info(f"\nSaving updated dataset to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Add Gaia GSP-Phot quality flags to eclipsing binary catalog',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input Parquet file (default: ml_training_data_with_gaia.parquet from config)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output Parquet file (default: input file with _with_flags suffix)'
    )
    
    parser.add_argument(
        '--fits-dir',
        type=str,
        default=None,
        help='Directory containing Gaia quality flag FITS files (default: data/external/gsp_teff_quality_flags/)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        # Use ML training data with Gaia colors
        input_file = config.get_dataset_path('ml_training_with_gaia', 'processed')
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        # Add suffix to input filename
        output_file = input_file.parent / input_file.name.replace('.parquet', '_with_flags.parquet')
    
    # Determine FITS directory
    if args.fits_dir:
        fits_dir = Path(args.fits_dir)
    else:
        fits_dir = config.project_root / 'data' / 'external' / 'gsp_teff_quality_flags'
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Add Gaia GSP-Phot Quality Flags")
    logger.info("=" * 70)
    logger.info(f"\nInput file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"FITS directory: {fits_dir}")
    logger.info("")
    
    # Load catalog and add quality flags
    df = load_catalog_and_add_flags(input_file, output_file, fits_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Quality flags added successfully!")
    logger.info("=" * 70)
    logger.info("\nTo use high-quality sources only:")
    logger.info("  df_high_quality = df[df['teff_gspphot_flag'] == 1]")
    logger.info("\nTo exclude unmatched sources:")
    logger.info("  df_matched = df[df['teff_gspphot_flag'] >= 0]")


if __name__ == '__main__':
    main()

