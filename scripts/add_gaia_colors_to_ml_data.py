#!/usr/bin/env python3
"""
Add Gaia colors (bp_rp) to the ML training dataset.

This script merges the cleaned ML training data with Gaia catalog
to include Gaia photometric colors (BP-RP, G, BP, RP magnitudes).

Input:
    - data/processed/ml_training_data_clean.parquet
    - data/processed/eb_catalog_with_pm.parquet

Output:
    - data/processed/ml_training_data_with_gaia.parquet
"""

import sys
import pandas as pd
import polars as pl
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def main():
    # Load configuration
    config = get_config()

    # Get paths from config
    ml_file = config.get_dataset_path('ml_training_clean', 'processed')
    gaia_file = config.get_dataset_path('eb_catalog_with_pm', 'processed')
    output_file = config.get_dataset_path('ml_training_with_gaia', 'processed')

    print("=" * 70)
    print("Adding Gaia colors to ML training dataset")
    print("=" * 70)
    print(f"Project root: {config.project_root}")
    print()

    # Load the cleaned ML dataset
    print(f"1. Loading ML training data")
    print(f"   {ml_file.relative_to(config.project_root)}")
    ml_data = pl.read_parquet(ml_file)
    print(f"   ✓ Loaded {len(ml_data):,} objects")

    # Load Gaia catalog
    print(f"\n2. Loading Gaia catalog")
    print(f"   {gaia_file.relative_to(config.project_root)}")
    gaia_cat = pl.read_parquet(gaia_file)
    print(f"   ✓ Loaded {len(gaia_cat):,} objects")

    # Select Gaia photometric columns
    gaia_phot_cols = [
        'original_ext_source_id',
        'bp_rp',
        'phot_g_mean_mag',
        'phot_bp_mean_mag',
        'phot_rp_mean_mag'
    ]

    # Merge to add Gaia colors
    print(f"\n3. Merging datasets on 'original_ext_source_id'...")
    merged = ml_data.join(
        gaia_cat.select(gaia_phot_cols),
        on='original_ext_source_id',
        how='left'
    )
    print(f"   ✓ Merged data: {len(merged):,} objects")

    # Check for missing Gaia colors
    missing_bp_rp = merged.filter(pl.col('bp_rp').is_null())
    print(f"\n4. Data quality check:")
    print(f"   Objects with bp_rp color: {len(merged) - len(missing_bp_rp):,}")
    print(f"   Objects missing bp_rp:    {len(missing_bp_rp):,}")

    # Save enhanced dataset
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n5. Saving enhanced dataset")
    print(f"   {output_file.relative_to(config.project_root)}")
    merged.write_parquet(output_file)
    print(f"   ✓ Saved {len(merged):,} objects")

    # Show summary statistics
    print("\n" + "=" * 70)
    print("Summary of available colors:")
    print("=" * 70)

    color_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp']

    for col in color_cols:
        if col in merged.columns:
            min_val = merged.select(pl.col(col).min()).item()
            max_val = merged.select(pl.col(col).max()).item()
            median_val = merged.select(pl.col(col).median()).item()
            n_valid = len(merged.filter(pl.col(col).is_not_null()))
            print(f"  {col:15s}: [{min_val:8.3f}, {max_val:8.3f}], "
                  f"median={median_val:8.3f}, n={n_valid:,}")

    # Show magnitude summary
    print("\nGaia magnitudes:")
    mag_cols = ['gPSFMag', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
    for col in mag_cols:
        if col in merged.columns:
            min_val = merged.select(pl.col(col).min()).item()
            max_val = merged.select(pl.col(col).max()).item()
            median_val = merged.select(pl.col(col).median()).item()
            n_valid = len(merged.filter(pl.col(col).is_not_null()))
            print(f"  {col:20s}: [{min_val:8.3f}, {max_val:8.3f}], "
                  f"median={median_val:8.3f}, n={n_valid:,}")

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
