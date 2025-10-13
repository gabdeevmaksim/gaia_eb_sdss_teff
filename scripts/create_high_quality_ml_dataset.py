#!/usr/bin/env python3
"""
Create high-quality ML training dataset using only flag 1 sources.

This script filters the eclipsing binary catalog to include only sources with
Gaia GSP-Phot quality flag = 1 (higher quality temperature estimates) and
removes magnitude-dependent features to avoid bias.

Usage:
    # Create high-quality dataset
    python scripts/create_high_quality_ml_dataset.py

    # Specify output filename
    python scripts/create_high_quality_ml_dataset.py --output data/processed/ml_training_data_high_quality.parquet

Author: Auto-generated
Date: 2025-10-13
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def create_high_quality_dataset(
    input_file: Path,
    output_file: Path
) -> pd.DataFrame:
    """
    Create high-quality ML training dataset with only flag 1 sources.
    
    Parameters
    ----------
    input_file : Path
        Input Parquet file with quality flags
    output_file : Path
        Output Parquet file for high-quality dataset
        
    Returns
    -------
    pd.DataFrame
        Filtered dataset with only flag 1 sources
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {input_file.name}...")
    df = pd.read_parquet(input_file)
    
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Filter for flag 1 sources only
    logger.info("\nFiltering for high-quality sources (flag 1)...")
    df_high_quality = df[df['teff_gspphot_flag'] == 1].copy()
    
    logger.info(f"  Flag 1 sources: {len(df_high_quality):,}")
    logger.info(f"  Removed {len(df) - len(df_high_quality):,} flag 0 sources")
    logger.info(f"  Retention rate: {len(df_high_quality)/len(df)*100:.1f}%")
    
    # Remove magnitude-dependent features to avoid bias
    logger.info("\nRemoving magnitude-dependent features...")
    magnitude_features = [
        'phot_g_mean_mag',      # Gaia G magnitude
        'phot_bp_mean_mag',     # Gaia BP magnitude  
        'phot_rp_mean_mag',     # Gaia RP magnitude
        'gPSFMag'              # Pan-STARRS g magnitude
    ]
    
    # Check which magnitude features exist
    existing_mag_features = [col for col in magnitude_features if col in df_high_quality.columns]
    logger.info(f"  Removing magnitude features: {existing_mag_features}")
    
    # Remove magnitude features
    df_high_quality = df_high_quality.drop(columns=existing_mag_features)
    
    # Also remove the quality flag column (no longer needed)
    if 'teff_gspphot_flag' in df_high_quality.columns:
        df_high_quality = df_high_quality.drop(columns=['teff_gspphot_flag'])
        logger.info("  Removed 'teff_gspphot_flag' column")
    
    logger.info(f"\nFinal dataset columns: {list(df_high_quality.columns)}")
    logger.info(f"Final dataset shape: {df_high_quality.shape}")
    
    # Save the filtered dataset
    logger.info(f"\nSaving high-quality dataset to {output_file}...")
    df_high_quality.to_parquet(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("HIGH-QUALITY DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total sources: {len(df_high_quality):,}")
    logger.info(f"Features: {len(df_high_quality.columns)}")
    logger.info(f"Feature columns: {list(df_high_quality.columns)}")
    
    # Temperature statistics
    if 'teff_gspphot' in df_high_quality.columns:
        logger.info(f"\nTemperature statistics:")
        logger.info(f"  Mean: {df_high_quality['teff_gspphot'].mean():.1f} K")
        logger.info(f"  Median: {df_high_quality['teff_gspphot'].median():.1f} K")
        logger.info(f"  Range: {df_high_quality['teff_gspphot'].min():.1f} - {df_high_quality['teff_gspphot'].max():.1f} K")
    
    return df_high_quality


def main():
    parser = argparse.ArgumentParser(
        description='Create high-quality ML training dataset with only flag 1 sources',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input Parquet file with quality flags (default: ml_training_data_with_gaia_with_flags.parquet)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output Parquet file (default: ml_training_data_high_quality.parquet)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = config.project_root / 'data' / 'processed' / 'ml_training_data_with_gaia_with_flags.parquet'
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.project_root / 'data' / 'processed' / 'ml_training_data_high_quality.parquet'
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Create High-Quality ML Training Dataset")
    logger.info("=" * 70)
    logger.info(f"\nInput file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info("")
    
    # Create high-quality dataset
    df = create_high_quality_dataset(input_file, output_file)
    
    logger.info("\n" + "=" * 70)
    logger.info("High-quality dataset created successfully!")
    logger.info("=" * 70)
    logger.info("\nThis dataset is ready for ML training with:")
    logger.info("• Only high-quality temperature estimates (flag 1)")
    logger.info("• No magnitude-dependent features (removes bias)")
    logger.info("• Clean feature set for temperature prediction")


if __name__ == '__main__':
    main()
