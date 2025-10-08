#!/usr/bin/env python3
"""
Fast merge of duplicate original_ext_source_id entries in Pan-STARRS photometry data.

This script uses vectorized operations for efficiency:
1. For each source_id group, prioritizes non-missing values (-999.0)
2. For coordinates, takes mean of valid values
3. For photometry, uses weighted average based on inverse error squared
4. Missing values are indicated by -999.0
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def merge_duplicates_fast(input_file: Path, output_file: Path, missing_value: float = -999.0):
    """Efficiently merge duplicate original_ext_source_id entries."""

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return None

    print(f"Loading data from {input_file.name}...")
    df = pl.read_csv(input_file)

    print(f"Total rows before merging: {len(df):,}")
    print(f"Unique source_ids: {df['original_ext_source_id'].n_unique():,}")

    MISSING = missing_value

    # Get photometric columns from config
    config = get_config()
    phot_columns = config.get('processing', 'magnitude_columns')
    err_columns = config.get('processing', 'error_columns')

    print("\nCreating aggregation expressions...")

    # Build aggregation expressions
    agg_exprs = []

    # For coordinates: mean of non-missing values
    for coord_col in ['rra', 'rdec']:
        agg_exprs.append(
            pl.when(pl.col(coord_col).filter(pl.col(coord_col) != MISSING).len() > 0)
            .then(pl.col(coord_col).filter(pl.col(coord_col) != MISSING).mean())
            .otherwise(MISSING)
            .alias(coord_col)
        )

    # For photometry: weighted average or best single measurement
    for phot_col, err_col in zip(phot_columns, err_columns):
        # Create mask for valid data (not missing)
        valid_mask = (pl.col(phot_col) != MISSING) & (pl.col(err_col) != MISSING)

        # Count valid measurements
        valid_count = valid_mask.sum()

        # For single valid measurement, just take it
        # For multiple valid measurements, compute weighted average
        mag_expr = (
            pl.when(valid_count == 0)
            .then(MISSING)
            .when(valid_count == 1)
            .then(pl.col(phot_col).filter(valid_mask).first())
            .otherwise(
                # Weighted average: sum(mag/err^2) / sum(1/err^2)
                (pl.col(phot_col).filter(valid_mask) / pl.col(err_col).filter(valid_mask).pow(2)).sum() /
                (1.0 / pl.col(err_col).filter(valid_mask).pow(2)).sum()
            )
            .alias(phot_col)
        )

        err_expr = (
            pl.when(valid_count == 0)
            .then(MISSING)
            .when(valid_count == 1)
            .then(pl.col(err_col).filter(valid_mask).first())
            .otherwise(
                # Error of weighted mean: sqrt(1/sum(1/err^2))
                (1.0 / (1.0 / pl.col(err_col).filter(valid_mask).pow(2)).sum()).sqrt()
            )
            .alias(err_col)
        )

        agg_exprs.extend([mag_expr, err_expr])

    print("Performing merge operation...")

    # Group by source_id and aggregate
    merged_df = df.group_by('original_ext_source_id').agg(agg_exprs)

    print(f"Total rows after merging: {len(merged_df):,}")
    print(f"Rows removed: {len(df) - len(merged_df):,}")

    # Sort by source_id
    merged_df = merged_df.sort('original_ext_source_id')

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save merged data
    print(f"\nSaving merged data to {output_file.name}...")
    merged_df.write_csv(output_file)

    # Print statistics
    print("\nMerge Statistics:")
    for phot_col in phot_columns:
        orig_missing = df.filter(pl.col(phot_col) == MISSING).height
        merged_missing = merged_df.filter(pl.col(phot_col) == MISSING).height
        print(f"  {phot_col}: {orig_missing:,} → {merged_missing:,} missing ({orig_missing - merged_missing:,} recovered)")

    return output_file


def main():
    """Main function."""
    config = get_config()

    print("=" * 70)
    print("MERGE PAN-STARRS DUPLICATES (FAST)")
    print("=" * 70)
    print(f"Project root: {config.project_root}")
    print()

    # Get paths and parameters from config
    input_file = config.get_dataset_path('panstarrs_duplicates', 'external')
    output_file = config.get_dataset_path('panstarrs_duplicates_merged', 'external')
    missing_value = config.get('processing', 'missing_value')

    # Check if input exists
    if not input_file.exists():
        print(f"Error: Input file does not exist:")
        print(f"  {input_file}")
        print("\nTip: Run extract_panstarrs_duplicates.py first")
        sys.exit(1)

    result_file = merge_duplicates_fast(input_file, output_file, missing_value)

    if result_file:
        print("\n" + "=" * 70)
        print("✅ Duplicates merged successfully!")
        print(f"Output: {result_file.relative_to(config.project_root)}")
        print("=" * 70)
    else:
        print("\n❌ Merge failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
