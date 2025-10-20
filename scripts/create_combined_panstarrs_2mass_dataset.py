#!/usr/bin/env python3
"""
Create combined Pan-STARRS + 2MASS + Gaia training dataset.

This script merges Pan-STARRS optical photometry, 2MASS near-infrared photometry,
and Gaia colors for objects that have all measurements and Gaia GSP-Phot temperatures.

Features:
- Gaia: BP-RP color
- Pan-STARRS: g-r, r-i, i-z colors
- 2MASS: J-H, H-K, J-K colors
- Synthetic: B-V color

Total: 8 color features (no magnitudes to avoid distance bias)

Usage:
    python scripts/create_combined_panstarrs_2mass_dataset.py

Author: Claude Code
Date: 2025-10-16
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
    """Create combined Pan-STARRS + 2MASS training dataset."""

    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("=" * 80)
    logger.info("CREATE COMBINED PAN-STARRS + 2MASS + GAIA DATASET")
    logger.info("=" * 80)

    # Load main catalog
    logger.info("\n1. Loading main catalog...")
    catalog_file = processed_dir / 'eb_catalog.parquet'
    df_catalog = pd.read_parquet(catalog_file)
    logger.info(f"   Total objects in catalog: {len(df_catalog):,}")

    # Filter objects WITH Gaia Teff
    df_with_teff = df_catalog[df_catalog['teff_gspphot'].notna()].copy()
    logger.info(f"   Objects with Gaia Teff: {len(df_with_teff):,}")

    # Load Pan-STARRS photometry
    logger.info("\n2. Loading Pan-STARRS photometry...")
    ps_file = processed_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet'
    df_ps = pd.read_parquet(ps_file)
    logger.info(f"   Total Pan-STARRS objects: {len(df_ps):,}")

    # Keep only necessary Pan-STARRS columns
    ps_cols = ['original_ext_source_id', 'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color']
    df_ps_clean = df_ps[ps_cols].copy()

    # Merge catalog with Pan-STARRS
    logger.info("\n3. Merging catalog with Pan-STARRS...")
    df_merged = df_with_teff.merge(
        df_ps_clean,
        on='original_ext_source_id',
        how='inner'
    )
    logger.info(f"   Objects with Pan-STARRS: {len(df_merged):,}")

    # Load 2MASS photometry
    logger.info("\n4. Loading 2MASS photometry...")
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
    logger.info("\n5. Merging with 2MASS...")
    df_combined = df_merged.merge(
        df_2mass_clean,
        on='source_id',
        how='inner'
    )
    logger.info(f"   Objects with both Pan-STARRS and 2MASS: {len(df_combined):,}")

    # Filter for valid colors
    logger.info("\n6. Filtering for valid colors...")

    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'g_r_color': (-0.5, 3.0),
        'r_i_color': (-0.5, 2.0),
        'i_z_color': (-0.5, 1.5),
        'B_V_color': (-0.5, 3.0),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5)
    }

    valid_mask = pd.Series(True, index=df_combined.index)

    logger.info("   Applying color range filters:")
    for col, (min_val, max_val) in color_ranges.items():
        before = valid_mask.sum()
        valid_mask &= (df_combined[col] >= min_val) & (df_combined[col] <= max_val)
        after = valid_mask.sum()
        logger.info(f"     {col}: {before:,} → {after:,} (removed {before-after:,})")

    df_valid = df_combined[valid_mask].copy()
    logger.info(f"   Objects after color filtering: {len(df_valid):,}")

    # Remove any remaining NaN values in color columns
    color_cols = ['bp_rp', 'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color',
                  'j_h_color', 'h_k_color', 'j_k_color']

    before_nan = len(df_valid)
    df_valid = df_valid.dropna(subset=color_cols + ['teff_gspphot'])
    after_nan = len(df_valid)
    logger.info(f"   Removed NaN values: {before_nan - after_nan:,}")
    logger.info(f"   Final dataset size: {len(df_valid):,}")

    # Select final columns
    logger.info("\n7. Selecting final columns...")
    final_cols = ['source_id', 'original_ext_source_id', 'teff_gspphot'] + color_cols + \
                 ['j_snr', 'h_snr', 'k_snr', 'ph_qual']

    df_final = df_valid[final_cols].copy()

    logger.info(f"   Final columns ({len(final_cols)}):")
    for col in final_cols:
        logger.info(f"     - {col}")

    # Save dataset
    output_file = processed_dir / 'combined_panstarrs_2mass_colors_training.parquet'
    logger.info(f"\n8. Saving combined dataset: {output_file.name}")
    df_final.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"   File size: {file_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nTotal objects: {len(df_final):,}")
    logger.info(f"\nColor features (8 total):")
    logger.info(f"  Gaia:")
    logger.info(f"    - bp_rp")
    logger.info(f"  Pan-STARRS:")
    logger.info(f"    - g_r_color, r_i_color, i_z_color")
    logger.info(f"  Synthetic:")
    logger.info(f"    - B_V_color")
    logger.info(f"  2MASS:")
    logger.info(f"    - j_h_color, h_k_color, j_k_color")

    logger.info(f"\nTemperature statistics:")
    logger.info(f"  Range: {df_final['teff_gspphot'].min():.0f} - {df_final['teff_gspphot'].max():.0f} K")
    logger.info(f"  Mean:  {df_final['teff_gspphot'].mean():.0f} K")
    logger.info(f"  Median: {df_final['teff_gspphot'].median():.0f} K")
    logger.info(f"  Std:   {df_final['teff_gspphot'].std():.0f} K")

    logger.info(f"\n2MASS quality:")
    high_qual = df_final['ph_qual'].isin(['AAA', 'AAB']).sum()
    logger.info(f"  High quality (AAA/AAB): {high_qual:,} ({100*high_qual/len(df_final):.1f}%)")

    # Color statistics
    logger.info(f"\nColor statistics:")
    for col in color_cols:
        logger.info(f"  {col}:")
        logger.info(f"    Range: {df_final[col].min():.3f} to {df_final[col].max():.3f}")
        logger.info(f"    Mean:  {df_final[col].mean():.3f}")

    # Create schema documentation
    docs_dir = Path('docs')
    schema_file = docs_dir / 'COMBINED_PANSTARRS_2MASS_SCHEMA.md'

    logger.info(f"\n9. Creating schema documentation...")
    with open(schema_file, 'w') as f:
        f.write("# Combined Pan-STARRS + 2MASS + Gaia Training Dataset Schema\n\n")
        f.write(f"**Dataset**: `combined_panstarrs_2mass_colors_training.parquet`\n\n")
        f.write(f"**Total objects**: {len(df_final):,}\n\n")
        f.write(f"**Purpose**: Training temperature prediction models with maximum color coverage\n\n")

        f.write("## Features\n\n")
        f.write("### Color Features (8 total)\n\n")
        f.write("| Feature | Description | Wavelength Range |\n")
        f.write("|---------|-------------|------------------|\n")
        f.write("| bp_rp | Gaia BP-RP color | Optical (330-680 nm - 630-1050 nm) |\n")
        f.write("| g_r_color | Pan-STARRS g-r color | Optical (481-587 nm) |\n")
        f.write("| r_i_color | Pan-STARRS r-i color | Optical (587-755 nm) |\n")
        f.write("| i_z_color | Pan-STARRS i-z color | Optical-NIR (755-922 nm) |\n")
        f.write("| B_V_color | Synthetic B-V color | Optical (445-551 nm) |\n")
        f.write("| j_h_color | 2MASS J-H color | NIR (1.25-1.65 μm) |\n")
        f.write("| h_k_color | 2MASS H-K color | NIR (1.65-2.17 μm) |\n")
        f.write("| j_k_color | 2MASS J-K color | NIR (1.25-2.17 μm) |\n\n")

        f.write("### Target Variable\n\n")
        f.write("| Column | Description | Unit |\n")
        f.write("|--------|-------------|------|\n")
        f.write("| teff_gspphot | Gaia GSP-Phot effective temperature | Kelvin (K) |\n\n")

        f.write("### Quality Indicators\n\n")
        f.write("| Column | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| j_snr | 2MASS J-band signal-to-noise ratio |\n")
        f.write("| h_snr | 2MASS H-band signal-to-noise ratio |\n")
        f.write("| k_snr | 2MASS K-band signal-to-noise ratio |\n")
        f.write("| ph_qual | 2MASS photometric quality flag (AAA=best) |\n\n")

        f.write("### Common Keys\n\n")
        f.write("| Column | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| source_id | Gaia DR3 source identifier |\n")
        f.write("| original_ext_source_id | Pan-STARRS source identifier |\n\n")

        f.write("## Data Quality\n\n")
        f.write(f"- All objects have valid measurements in all 8 colors\n")
        f.write(f"- Color ranges enforced based on stellar physics\n")
        f.write(f"- No NaN or infinite values\n")
        f.write(f"- {100*high_qual/len(df_final):.1f}% have high-quality 2MASS photometry (AAA/AAB)\n\n")

        f.write("## Statistics\n\n")
        f.write("```\n")
        f.write(f"Objects: {len(df_final):,}\n\n")
        f.write(f"Temperature:\n")
        f.write(f"  Range:  {df_final['teff_gspphot'].min():.0f} - {df_final['teff_gspphot'].max():.0f} K\n")
        f.write(f"  Mean:   {df_final['teff_gspphot'].mean():.0f} K\n")
        f.write(f"  Median: {df_final['teff_gspphot'].median():.0f} K\n")
        f.write(f"  Std:    {df_final['teff_gspphot'].std():.0f} K\n\n")

        f.write("Color Ranges:\n")
        for col in color_cols:
            f.write(f"  {col:<15} {df_final[col].min():>6.3f} to {df_final[col].max():>6.3f}  (mean: {df_final[col].mean():>6.3f})\n")
        f.write("```\n\n")

        f.write("## Wavelength Coverage\n\n")
        f.write("This dataset spans from optical to near-infrared:\n")
        f.write("- **Optical**: Pan-STARRS grizy (400-1000 nm) + Gaia BP/RP (330-1050 nm)\n")
        f.write("- **Near-IR**: 2MASS JHK (1.25-2.17 μm)\n\n")
        f.write("Combined coverage: **~330 nm to 2.17 μm**\n\n")

        f.write("## Comparison with Other Datasets\n\n")
        f.write("| Dataset | Objects | Colors | Coverage |\n")
        f.write("|---------|---------|--------|----------|\n")
        f.write(f"| Gaia only | 737,028 | 1 (BP-RP) | Optical |\n")
        f.write(f"| Gaia + Pan-STARRS | 701,644 | 5 | Optical |\n")
        f.write(f"| Gaia + 2MASS | 525,738 | 4 | Optical + NIR |\n")
        f.write(f"| **Combined (this)** | **{len(df_final):,}** | **8** | **Optical + NIR** |\n\n")

        f.write("## Usage Notes\n\n")
        f.write("- Use `source_id` to join with Gaia catalog\n")
        f.write("- Use `original_ext_source_id` to join with Pan-STARRS catalog\n")
        f.write("- This is the most complete color dataset but has fewer objects than individual datasets\n")
        f.write("- Best for exploring maximum predictive power with all available colors\n")

    logger.info(f"   Schema saved: {schema_file}")

    logger.info("\n" + "=" * 80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Documentation: {schema_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Train basic model: python scripts/train_combined_model.py")
    logger.info("  2. Engineer features: python scripts/create_combined_engineered_features.py")


if __name__ == '__main__':
    main()
