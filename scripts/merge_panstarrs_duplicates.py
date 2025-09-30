#!/usr/bin/env python3
"""
Merge duplicate original_ext_source_id entries in Pan-STARRS photometry data.

This script combines multiple rows for the same source_id by:
1. Preferring non-missing values (-999.0) over missing ones
2. For coordinates (rra, rdec), taking the average of valid values
3. For photometry, using weighted average based on inverse error squared
4. If all measurements are valid, keeps the one with smallest error
5. If errors are equal, keeps the first occurrence

Missing values are indicated by -999.0
"""

import polars as pl
import numpy as np
import sys
from pathlib import Path

def merge_duplicates(input_file: str, output_file: str = None):
    """Merge duplicate original_ext_source_id entries intelligently."""

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

    # Define photometric columns and their error columns
    phot_columns = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
    err_columns = ['gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr']
    coord_columns = ['rra', 'rdec']

    # Missing value indicator
    MISSING = -999.0

    def merge_group(group_df):
        """Merge a group of rows with the same source_id."""
        if len(group_df) == 1:
            return group_df

        # Start with the first row as base
        result = group_df[0].clone()

        # Merge coordinates (average of valid values)
        for coord_col in coord_columns:
            valid_coords = group_df.filter(pl.col(coord_col) != MISSING)[coord_col]
            if len(valid_coords) > 0:
                result = result.with_columns(pl.lit(valid_coords.mean()).alias(coord_col))

        # Merge photometric data
        for phot_col, err_col in zip(phot_columns, err_columns):
            # Get rows with valid measurements (not -999.0)
            valid_data = group_df.filter(
                (pl.col(phot_col) != MISSING) & (pl.col(err_col) != MISSING)
            )

            if len(valid_data) > 0:
                if len(valid_data) == 1:
                    # Only one valid measurement, use it
                    result = result.with_columns([
                        pl.lit(valid_data[phot_col][0]).alias(phot_col),
                        pl.lit(valid_data[err_col][0]).alias(err_col)
                    ])
                else:
                    # Multiple valid measurements - use weighted average
                    errors = valid_data[err_col].to_numpy()
                    mags = valid_data[phot_col].to_numpy()

                    # Weight by inverse variance (1/error^2)
                    weights = 1.0 / (errors ** 2)
                    weights_sum = np.sum(weights)

                    if weights_sum > 0:
                        weighted_mag = np.sum(mags * weights) / weights_sum
                        # Error of weighted mean
                        weighted_err = np.sqrt(1.0 / weights_sum)

                        result = result.with_columns([
                            pl.lit(weighted_mag).alias(phot_col),
                            pl.lit(weighted_err).alias(err_col)
                        ])
                    else:
                        # Fallback: use measurement with smallest error
                        best_idx = valid_data[err_col].arg_min()
                        result = result.with_columns([
                            pl.lit(valid_data[phot_col][best_idx]).alias(phot_col),
                            pl.lit(valid_data[err_col][best_idx]).alias(err_col)
                        ])

        return result

    print("Merging duplicate rows...")

    # Group by source_id and apply merge function
    merged_df = df.group_by('original_ext_source_id').map_groups(merge_group)

    print(f"Total rows after merging: {len(merged_df):,}")
    print(f"Rows removed: {len(df) - len(merged_df):,}")

    # Sort by source_id for consistency
    merged_df = merged_df.sort('original_ext_source_id')

    # Save merged data
    print(f"Saving merged data to {output_file}...")
    merged_df.write_csv(output_file)

    # Print some statistics about the merge
    print("\nMerge Statistics:")

    # Count how many sources had data filled in
    original_missing = df.filter(
        (pl.col('gPSFMag') == MISSING) | (pl.col('rPSFMag') == MISSING) |
        (pl.col('iPSFMag') == MISSING) | (pl.col('zPSFMag') == MISSING) |
        (pl.col('yPSFMag') == MISSING)
    )
    merged_missing = merged_df.filter(
        (pl.col('gPSFMag') == MISSING) | (pl.col('rPSFMag') == MISSING) |
        (pl.col('iPSFMag') == MISSING) | (pl.col('zPSFMag') == MISSING) |
        (pl.col('yPSFMag') == MISSING)
    )

    print(f"Sources with missing photometry before merge: {len(original_missing):,}")
    print(f"Sources with missing photometry after merge: {len(merged_missing):,}")
    print(f"Missing data recovered: {len(original_missing) - len(merged_missing):,}")

    return str(output_file)

def create_deduplicated_full_dataset(original_file: str, output_file: str = None):
    """Create a deduplicated version of the full dataset."""

    original_path = Path(original_file)
    if output_file is None:
        output_file = original_path.parent / f"{original_path.stem}_deduplicated{original_path.suffix}"

    print(f"Processing full dataset: {original_file}")
    return merge_duplicates(original_file, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_panstarrs_duplicates.py <input_file> [output_file]")
        print("       python merge_panstarrs_duplicates.py --full <full_dataset_file> [output_file]")
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
        result_file = merge_duplicates(input_file, output_file)

    print(f"\nDone! Merged data saved to: {result_file}")