#!/usr/bin/env python3
"""
Extract duplicate original_ext_source_id entries from Pan-STARRS photometry data.

This script identifies all rows with duplicate original_ext_source_id values
and saves them to a separate file for analysis, without modifying the original data.
"""

import polars as pl
import sys
from pathlib import Path

def extract_duplicates(input_file: str, output_file: str = None):
    """Extract duplicate original_ext_source_id entries to a separate file."""

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_duplicates{input_path.suffix}"

    print(f"Loading data from {input_file}...")

    # Load the data using Polars for performance
    df = pl.read_csv(input_file)

    print(f"Total rows: {len(df):,}")
    print(f"Unique original_ext_source_id: {df['original_ext_source_id'].n_unique():,}")

    # Find rows with duplicate original_ext_source_id
    print("Identifying duplicates...")

    # Count occurrences of each original_ext_source_id
    source_id_counts = df.group_by('original_ext_source_id').agg(pl.len().alias('count'))

    # Get source_ids that appear more than once
    duplicate_source_ids = source_id_counts.filter(pl.col('count') > 1)['original_ext_source_id']

    print(f"Number of source_ids with duplicates: {len(duplicate_source_ids):,}")

    # Extract all rows (including first occurrence) for duplicate source_ids
    duplicates_df = df.filter(pl.col('original_ext_source_id').is_in(duplicate_source_ids))

    print(f"Total duplicate rows: {len(duplicates_df):,}")

    # Sort by original_ext_source_id for easier inspection
    duplicates_df = duplicates_df.sort('original_ext_source_id')

    # Save duplicates to file
    print(f"Saving duplicates to {output_file}...")
    duplicates_df.write_csv(output_file)

    # Print some statistics
    print("\nDuplicate Statistics:")
    duplicate_counts = duplicates_df.group_by('original_ext_source_id').agg(pl.len().alias('count'))
    print(f"Max duplicates for a single source_id: {duplicate_counts['count'].max()}")
    print(f"Mean duplicates per source_id: {duplicate_counts['count'].mean():.2f}")

    # Show distribution of duplicate counts
    count_distribution = duplicate_counts.group_by('count').agg(pl.len().alias('num_sources'))
    print("\nDistribution of duplicate counts:")
    for row in count_distribution.sort('count').iter_rows():
        count, num_sources = row
        print(f"  {count} duplicates: {num_sources:,} source_ids")

    return str(output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_panstarrs_duplicates.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result_file = extract_duplicates(input_file, output_file)
    print(f"\nDone! Duplicates saved to: {result_file}")