#!/usr/bin/env python3
"""
Prepare dataset for basic model predictions by merging Pan-STARRS photometry
with Gaia temperatures to identify objects missing teff_gspphot.

This script:
1. Loads Pan-STARRS photometry dataset (with g,r,i,z photometry)
2. Merges with Gaia catalog to get teff_gspphot values
3. Identifies objects with/without teff_gspphot
4. Creates dataset ready for prediction with basic RF model

Usage:
    python scripts/prepare_data_for_basic_prediction.py

Author: Claude Code
Date: 2025-10-15
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    logger = logging.getLogger(__name__)
    config = get_config()

    logger.info("=" * 70)
    logger.info("PREPARE DATA FOR BASIC MODEL PREDICTION")
    logger.info("=" * 70)

    # Load Pan-STARRS photometry
    logger.info("\n1. Loading Pan-STARRS photometry...")
    ps_file = config.get_path('processed') / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pl.read_parquet(ps_file)
    logger.info(f"  Loaded: {len(df_ps):,} objects")
    logger.info(f"  Columns: {df_ps.columns}")

    # Load Gaia catalog
    logger.info("\n2. Loading Gaia catalog with teff_gspphot...")
    gaia_file = config.get_path('processed') / 'eb_catalog.parquet'
    df_gaia = pl.read_parquet(gaia_file)
    logger.info(f"  Loaded: {len(df_gaia):,} objects")

    # Merge datasets
    logger.info("\n3. Merging Pan-STARRS with Gaia on original_ext_source_id...")
    df_merged = df_ps.join(
        df_gaia.select(['original_ext_source_id', 'source_id', 'teff_gspphot']),
        on='original_ext_source_id',
        how='left'
    )
    logger.info(f"  Merged: {len(df_merged):,} objects")

    # Analyze coverage
    logger.info("\n4. Analyzing teff_gspphot coverage...")
    has_teff = df_merged.filter(pl.col('teff_gspphot').is_not_null()).height
    missing_teff = len(df_merged) - has_teff

    logger.info(f"  With teff_gspphot: {has_teff:,} ({100*has_teff/len(df_merged):.1f}%)")
    logger.info(f"  Missing teff_gspphot: {missing_teff:,} ({100*missing_teff/len(df_merged):.1f}%)")

    # Prepare columns for basic model
    logger.info("\n5. Preparing columns for basic RF model...")

    # Rename columns to match model expectations
    df_final = df_merged.select([
        'original_ext_source_id',
        pl.col('source_id').alias('gaia_source_id'),
        'g_r_color',
        'r_i_color',
        'i_z_color',
        'gPSFMag',
        'teff_gspphot',
        'Te_gr',
        'Te_ri',
        'Te_iz'
    ])

    # Add B-V color (synthetic from g-r)
    df_final = df_final.with_columns([
        (0.98 * pl.col('g_r_color') + 0.22).alias('B_V_color')
    ])

    logger.info(f"  Final columns: {df_final.columns}")

    # Save merged dataset
    output_file = config.get_path('processed') / 'eb_panstarrs_for_prediction.parquet'
    logger.info(f"\n6. Saving prepared dataset...")
    df_final.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")

    # Create summary
    summary_file = output_file.parent / 'eb_panstarrs_for_prediction_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("DATASET FOR BASIC MODEL PREDICTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total objects: {len(df_final):,}\n")
        f.write(f"With teff_gspphot: {has_teff:,} ({100*has_teff/len(df_final):.1f}%)\n")
        f.write(f"Missing teff_gspphot: {missing_teff:,} ({100*missing_teff/len(df_final):.1f}%)\n\n")
        f.write(f"Columns:\n")
        for col in df_final.columns:
            f.write(f"  - {col}\n")
        f.write(f"\nFeatures for basic RF model:\n")
        f.write(f"  - g_r_color\n")
        f.write(f"  - r_i_color\n")
        f.write(f"  - i_z_color\n")
        f.write(f"  - B_V_color (synthetic from g-r)\n")
        f.write(f"  - gPSFMag\n")
        f.write(f"\nGround truth for validation:\n")
        f.write(f"  - teff_gspphot (available for {100*has_teff/len(df_final):.1f}%)\n")
        f.write(f"  - Te_gr, Te_ri, Te_iz (color-based temperatures)\n")

    logger.info(f"  Summary saved: {summary_file}")

    logger.info("\n" + "=" * 70)
    logger.info("DATA PREPARATION COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"\nNext step:")
    logger.info(f"  python scripts/predict_temperatures_basic_model.py --input {output_file}")


if __name__ == '__main__':
    main()
