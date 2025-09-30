#!/usr/bin/env python3
"""
Fast merge of duplicate original_ext_source_id entries in Pan-STARRS photometry data.

This script uses vectorized operations for efficiency:
1. For each source_id group, prioritizes non-missing values (-999.0)
2. For coordinates, takes mean of valid values
3. For photometry, uses weighted average based on inverse error squared
4. Missing values are indicated by -999.0
"""

import polars as pl
import numpy as np
import sys
from pathlib import Path

def merge_duplicates_fast(input_file: str, output_file: str = None):
    """Efficiently merge duplicate original_ext_source_id entries."""

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_merged{input_path.suffix}"

    print(f"Loading data from {input_file}...")
    df = pl.read_csv(input_file)

    print(f"Total rows before merging: {len(df):,}")
    print(f"Unique source_ids: {df['original_ext_source_id'].n_unique():,}")

    MISSING = -999.0

    # Photometric columns and their errors
    phot_columns = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
    err_columns = ['gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr']

    print("Creating aggregation expressions...")

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

    # Save merged data
    print(f"Saving merged data to {output_file}...")
    merged_df.write_csv(output_file)

    # Print statistics
    print("\nMerge Statistics:")
    for phot_col in phot_columns:
        orig_missing = df.filter(pl.col(phot_col) == MISSING).height
        merged_missing = merged_df.filter(pl.col(phot_col) == MISSING).height
        print(f"{phot_col}: {orig_missing:,} → {merged_missing:,} missing ({orig_missing - merged_missing:,} recovered)")

    return str(output_file)

def create_deduplicated_full_dataset(original_file: str, output_file: str = None):
    """Create a deduplicated version of the full dataset."""

    original_path = Path(original_file)
    if output_file is None:
        output_file = original_path.parent / f"{original_path.stem}_deduplicated{original_path.suffix}"

    print(f"Processing full dataset: {original_file}")
    return merge_duplicates_fast(original_file, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_panstarrs_duplicates_fast.py <input_file> [output_file]")
        print("       python merge_panstarrs_duplicates_fast.py --full <full_dataset_file> [output_file]")
        sys.exit(1)

    if sys.argv[1] == "--full":
        if len(sys.argv) < 3:
            print("Error: --full option requires input file")
            sys.exit(1)
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        result_file = create_deduplicated_full_dataset(input_file, output_file)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        result_file = merge_duplicates_fast(input_file, output_file)

    print(f"\nDone! Merged data saved to: {result_file}")