#!/usr/bin/env python3
"""
Create Pan-STARRS + Gaia + 2MASS training dataset with Gaia g magnitude.

This script creates a training dataset with:
- Pan-STARRS colors: g-r, r-i, i-z
- Gaia color: BP-RP
- 2MASS NIR colors: J-H, H-K, J-K
- Gaia magnitude: phot_g_mean_mag

Total: 7 colors + 1 magnitude = 8 features

This version EXCLUDES:
- Synthetic B-V color (use measured colors only)

Usage:
    python scripts/create_ps_gaia_2mass_gmag_dataset.py

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path
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
logger = logging.getLogger(__name__)


def main():
    """Create Pan-STARRS + Gaia + 2MASS dataset with g magnitude."""

    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("=" * 80)
    logger.info("CREATE PAN-STARRS + GAIA + 2MASS WITH G MAGNITUDE DATASET")
    logger.info("=" * 80)

    # Load main catalog
    logger.info("\n1. Loading main catalog...")
    catalog_file = processed_dir / 'eb_catalog.parquet'
    df_catalog = pd.read_parquet(catalog_file)
    logger.info(f"   Total objects in catalog: {len(df_catalog):,}")

    # Filter objects WITH Gaia Teff
    df_with_teff = df_catalog[df_catalog['teff_gspphot'].notna()].copy()
    logger.info(f"   Objects with Gaia Teff: {len(df_with_teff):,}")

    # Check for Gaia g magnitude
    logger.info("\n2. Checking Gaia g magnitude availability...")
    if 'phot_g_mean_mag' in df_with_teff.columns:
        has_gmag = df_with_teff['phot_g_mean_mag'].notna().sum()
        logger.info(f"   Objects with Gaia g magnitude: {has_gmag:,}")
    else:
        raise ValueError("Gaia g magnitude not found in catalog")

    # Load Pan-STARRS photometry
    logger.info("\n3. Loading Pan-STARRS photometry...")
    ps_file = processed_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pd.read_parquet(ps_file)
    logger.info(f"   Total Pan-STARRS objects: {len(df_ps):,}")

    # Keep only necessary Pan-STARRS columns (NO B-V)
    ps_cols = ['original_ext_source_id', 'g_r_color', 'r_i_color', 'i_z_color']
    df_ps_clean = df_ps[ps_cols].copy()

    # Merge catalog with Pan-STARRS
    logger.info("\n4. Merging catalog with Pan-STARRS...")
    df_merged = df_with_teff.merge(
        df_ps_clean,
        on='original_ext_source_id',
        how='inner'
    )
    logger.info(f"   Objects with Pan-STARRS: {len(df_merged):,}")

    # Load 2MASS photometry
    logger.info("\n5. Loading 2MASS photometry...")
    tmass_file = processed_dir / 'eb_2mass_photometry.parquet'
    df_2mass = pd.read_parquet(tmass_file)

    # Calculate 2MASS colors
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']

    logger.info(f"   Total 2MASS objects: {len(df_2mass):,}")

    # Keep only necessary 2MASS columns
    tmass_cols = ['source_id', 'j_h_color', 'h_k_color', 'j_k_color',
                  'j_snr', 'h_snr', 'k_snr', 'ph_qual']
    df_2mass_clean = df_2mass[tmass_cols].copy()

    # Merge with 2MASS
    logger.info("\n6. Merging with 2MASS...")
    df_combined = df_merged.merge(
        df_2mass_clean,
        on='source_id',
        how='inner'
    )
    logger.info(f"   Objects with Pan-STARRS + 2MASS: {len(df_combined):,}")

    # Filter for valid colors and magnitude
    logger.info("\n7. Filtering for valid colors and magnitude...")

    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g_r_color': (-0.5, 3.0),
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5),
        'phot_g_mean_mag': (10.0, 21.0)
    }

    valid_mask = pd.Series(True, index=df_combined.index)

    logger.info("   Applying filters:")
    for col, (min_val, max_val) in color_ranges.items():
        before = valid_mask.sum()
        valid_mask &= (df_combined[col] >= min_val) & (df_combined[col] <= max_val)
        after = valid_mask.sum()
        logger.info(f"     {col}: {before:,} → {after:,} (removed {before-after:,})")

    df_valid = df_combined[valid_mask].copy()
    logger.info(f"   Objects after filtering: {len(df_valid):,}")

    # Remove any remaining NaN values
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color',
                  'j_h_color', 'h_k_color', 'j_k_color']
    feature_cols = color_cols + ['phot_g_mean_mag']

    before_nan = len(df_valid)
    df_valid = df_valid.dropna(subset=feature_cols + ['teff_gspphot'])
    after_nan = len(df_valid)
    logger.info(f"   Removed NaN values: {before_nan - after_nan:,}")
    logger.info(f"   Final dataset size: {len(df_valid):,}")

    # Select final columns
    logger.info("\n8. Selecting final columns...")
    final_cols = ['source_id', 'original_ext_source_id', 'teff_gspphot'] + feature_cols + \
                 ['j_snr', 'h_snr', 'k_snr', 'ph_qual']

    df_final = df_valid[final_cols].copy()

    logger.info(f"   Final columns ({len(final_cols)}):")
    for col in final_cols:
        logger.info(f"     - {col}")

    # Save dataset
    output_file = processed_dir / 'ps_gaia_2mass_gmag_training.parquet'
    logger.info(f"\n9. Saving dataset: {output_file.name}")
    df_final.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"   File size: {file_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nTotal objects: {len(df_final):,}")
    logger.info(f"\nFeatures (8 total):")
    logger.info(f"  Optical colors (4):")
    logger.info(f"    - bp_rp (Gaia)")
    logger.info(f"    - g_r_color, r_i_color, i_z_color (Pan-STARRS)")
    logger.info(f"  Near-IR colors (3):")
    logger.info(f"    - j_h_color, h_k_color, j_k_color (2MASS)")
    logger.info(f"  Magnitude (1):")
    logger.info(f"    - phot_g_mean_mag (Gaia)")

    logger.info(f"\nTemperature statistics:")
    logger.info(f"  Range: {df_final['teff_gspphot'].min():.0f} - {df_final['teff_gspphot'].max():.0f} K")
    logger.info(f"  Mean:  {df_final['teff_gspphot'].mean():.0f} K")
    logger.info(f"  Median: {df_final['teff_gspphot'].median():.0f} K")
    logger.info(f"  Std:   {df_final['teff_gspphot'].std():.0f} K")

    # Feature statistics
    logger.info(f"\nFeature statistics:")
    for col in feature_cols:
        logger.info(f"  {col}:")
        logger.info(f"    Range: {df_final[col].min():.3f} to {df_final[col].max():.3f}")
        logger.info(f"    Mean:  {df_final[col].mean():.3f}")

    # 2MASS quality
    high_qual = df_final['ph_qual'].isin(['AAA', 'AAB']).sum()
    logger.info(f"\n2MASS quality:")
    logger.info(f"  High quality (AAA/AAB): {high_qual:,} ({100*high_qual/len(df_final):.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Train model: python scripts/train_ps_gaia_2mass_gmag_model.py")


if __name__ == '__main__':
    main()
