#!/usr/bin/env python3
"""
Create Pan-STARRS + Gaia training dataset with Gaia g magnitude.

This script creates a training dataset with:
- Pan-STARRS colors: g-r, r-i, i-z
- Gaia color: BP-RP
- Gaia magnitude: phot_g_mean_mag

Total: 4 colors + 1 magnitude = 5 features

This version EXCLUDES:
- Synthetic B-V color (use measured colors only)
- 2MASS NIR colors (keep optical only)

Usage:
    python scripts/create_panstarrs_gaia_with_gmag_dataset.py

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
    """Create Pan-STARRS + Gaia dataset with g magnitude."""

    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("=" * 80)
    logger.info("CREATE PAN-STARRS + GAIA DATASET WITH G MAGNITUDE")
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
        logger.error("   ERROR: phot_g_mean_mag column not found in catalog!")
        logger.info("   Available magnitude columns:")
        mag_cols = [col for col in df_with_teff.columns if 'mag' in col.lower() or 'phot' in col.lower()]
        for col in mag_cols:
            logger.info(f"     - {col}")
        raise ValueError("Gaia g magnitude not found in catalog")

    # Load Pan-STARRS photometry
    logger.info("\n3. Loading Pan-STARRS photometry...")
    ps_file = processed_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pd.read_parquet(ps_file)
    logger.info(f"   Total Pan-STARRS objects: {len(df_ps):,}")

    # Keep only necessary Pan-STARRS columns (NO B-V synthetic color)
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

    # Filter for valid colors and magnitude
    logger.info("\n5. Filtering for valid colors and magnitude...")

    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g_r_color': (-0.5, 3.0),
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'phot_g_mean_mag': (10.0, 21.0)  # Reasonable Gaia magnitude range
    }

    valid_mask = pd.Series(True, index=df_merged.index)

    logger.info("   Applying filters:")
    for col, (min_val, max_val) in color_ranges.items():
        before = valid_mask.sum()
        valid_mask &= (df_merged[col] >= min_val) & (df_merged[col] <= max_val)
        after = valid_mask.sum()
        logger.info(f"     {col}: {before:,} → {after:,} (removed {before-after:,})")

    df_valid = df_merged[valid_mask].copy()
    logger.info(f"   Objects after filtering: {len(df_valid):,}")

    # Remove any remaining NaN values
    feature_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color', 'phot_g_mean_mag']

    before_nan = len(df_valid)
    df_valid = df_valid.dropna(subset=feature_cols + ['teff_gspphot'])
    after_nan = len(df_valid)
    logger.info(f"   Removed NaN values: {before_nan - after_nan:,}")
    logger.info(f"   Final dataset size: {len(df_valid):,}")

    # Select final columns
    logger.info("\n6. Selecting final columns...")
    final_cols = ['source_id', 'original_ext_source_id', 'teff_gspphot'] + feature_cols

    df_final = df_valid[final_cols].copy()

    logger.info(f"   Final columns ({len(final_cols)}):")
    for col in final_cols:
        logger.info(f"     - {col}")

    # Save dataset
    output_file = processed_dir / 'panstarrs_gaia_with_gmag_training.parquet'
    logger.info(f"\n7. Saving dataset: {output_file.name}")
    df_final.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"   File size: {file_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nTotal objects: {len(df_final):,}")
    logger.info(f"\nFeatures (5 total):")
    logger.info(f"  Pan-STARRS colors (3):")
    logger.info(f"    - g_r_color, r_i_color, i_z_color")
    logger.info(f"  Gaia color (1):")
    logger.info(f"    - bp_rp")
    logger.info(f"  Gaia magnitude (1):")
    logger.info(f"    - phot_g_mean_mag")

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
        logger.info(f"    Median: {df_final[col].median():.3f}")

    # Create schema documentation
    docs_dir = Path('docs')
    schema_file = docs_dir / 'PANSTARRS_GAIA_GMAG_SCHEMA.md'

    logger.info(f"\n8. Creating schema documentation...")
    with open(schema_file, 'w') as f:
        f.write("# Pan-STARRS + Gaia (with g magnitude) Training Dataset Schema\n\n")
        f.write(f"**Dataset**: `panstarrs_gaia_with_gmag_training.parquet`\n\n")
        f.write(f"**Total objects**: {len(df_final):,}\n\n")
        f.write(f"**Purpose**: Training temperature prediction with optical colors + brightness\n\n")

        f.write("## Features\n\n")
        f.write("### Color Features (4 total)\n\n")
        f.write("| Feature | Description | Wavelength Range |\n")
        f.write("|---------|-------------|------------------|\n")
        f.write("| bp_rp | Gaia BP-RP color | Optical (330-680 nm - 630-1050 nm) |\n")
        f.write("| g_r_color | Pan-STARRS g-r color | Optical (481-587 nm) |\n")
        f.write("| r_i_color | Pan-STARRS r-i color | Optical (587-755 nm) |\n")
        f.write("| i_z_color | Pan-STARRS i-z color | Optical-NIR (755-922 nm) |\n\n")

        f.write("### Magnitude Feature (1 total)\n\n")
        f.write("| Feature | Description | Unit |\n")
        f.write("|---------|-------------|------|\n")
        f.write("| phot_g_mean_mag | Gaia G-band mean magnitude | mag |\n\n")

        f.write("### Target Variable\n\n")
        f.write("| Column | Description | Unit |\n")
        f.write("|--------|-------------|------|\n")
        f.write("| teff_gspphot | Gaia GSP-Phot effective temperature | Kelvin (K) |\n\n")

        f.write("### Identifiers\n\n")
        f.write("| Column | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| source_id | Gaia DR3 source identifier |\n")
        f.write("| original_ext_source_id | Pan-STARRS source identifier |\n\n")

        f.write("## Data Quality\n\n")
        f.write(f"- All objects have valid measurements in all 4 colors + 1 magnitude\n")
        f.write(f"- Color ranges enforced based on stellar physics\n")
        f.write(f"- Gaia g magnitude range: 10-21 mag\n")
        f.write(f"- No NaN or infinite values\n\n")

        f.write("## Statistics\n\n")
        f.write("```\n")
        f.write(f"Objects: {len(df_final):,}\n\n")
        f.write(f"Temperature:\n")
        f.write(f"  Range:  {df_final['teff_gspphot'].min():.0f} - {df_final['teff_gspphot'].max():.0f} K\n")
        f.write(f"  Mean:   {df_final['teff_gspphot'].mean():.0f} K\n")
        f.write(f"  Median: {df_final['teff_gspphot'].median():.0f} K\n")
        f.write(f"  Std:    {df_final['teff_gspphot'].std():.0f} K\n\n")

        f.write("Feature Ranges:\n")
        for col in feature_cols:
            f.write(f"  {col:<20} {df_final[col].min():>7.3f} to {df_final[col].max():>7.3f}  (mean: {df_final[col].mean():>7.3f})\n")
        f.write("```\n\n")

        f.write("## Comparison with Other Models\n\n")
        f.write("| Model | Colors | Magnitude | Objects | Note |\n")
        f.write("|-------|--------|-----------|---------|------|\n")
        f.write("| Unified (no gPSF) | PS + Gaia BP-RP | None | 701k | Color-only, no distance bias |\n")
        f.write("| Combined | PS + 2MASS + Gaia | None | 509k | Maximum color coverage, NIR |\n")
        f.write("| **This model** | **PS + Gaia BP-RP** | **Gaia g** | **~700k** | **Optical + brightness info** |\n\n")

        f.write("## Key Differences from Other Models\n\n")
        f.write("- **No synthetic B-V color**: Uses only measured colors\n")
        f.write("- **No 2MASS colors**: Optical-only (more objects available)\n")
        f.write("- **Includes Gaia g magnitude**: Can leverage brightness information\n")
        f.write("- **Expected to have magnitude bias**: Predictions may depend on distance/brightness\n")
        f.write("- **Higher sample size than Combined model**: More training data available\n\n")

        f.write("## Usage Notes\n\n")
        f.write("- Use `source_id` to join with Gaia catalog\n")
        f.write("- Use `original_ext_source_id` to join with Pan-STARRS catalog\n")
        f.write("- Be aware of potential magnitude bias in predictions\n")
        f.write("- Compare with color-only unified model to assess magnitude impact\n")

    logger.info(f"   Schema saved: {schema_file}")

    logger.info("\n" + "=" * 80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Documentation: {schema_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Train model: python scripts/train_panstarrs_gaia_gmag_model.py")


if __name__ == '__main__':
    main()
