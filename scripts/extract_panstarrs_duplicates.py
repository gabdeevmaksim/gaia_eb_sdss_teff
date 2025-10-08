#!/usr/bin/env python3
"""
Extract duplicate original_ext_source_id entries from Pan-STARRS photometry data.

This script identifies all rows with duplicate original_ext_source_id values
and saves them to a separate file for analysis, without modifying the original data.
"""

import sys
import polars as pl
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def extract_duplicates(input_file: Path, output_file: Path = None):
    """Extract duplicate original_ext_source_id entries to a separate file."""

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return None

    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_duplicates{input_file.suffix}"

    print(f"Loading data from {input_file.name}...")

    # Load the data using Polars for performance
    df = pl.read_csv(input_file)

    print(f"Total rows: {len(df):,}")
    print(f"Unique original_ext_source_id: {df['original_ext_source_id'].n_unique():,}")

    # Find rows with duplicate original_ext_source_id
    print("\nIdentifying duplicates...")

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

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save duplicates to file
    print(f"\nSaving duplicates to {output_file.name}...")
    duplicates_df.write_csv(output_file)

    # Print some statistics
    print("\nDuplicate Statistics:")
    duplicate_counts = duplicates_df.group_by('original_ext_source_id').agg(pl.len().alias('count'))
    print(f"  Max duplicates for a single source_id: {duplicate_counts['count'].max()}")
    print(f"  Mean duplicates per source_id: {duplicate_counts['count'].mean():.2f}")

    # Show distribution of duplicate counts
    count_distribution = duplicate_counts.group_by('count').agg(pl.len().alias('num_sources'))
    print("\n  Distribution of duplicate counts:")
    for row in count_distribution.sort('count').iter_rows():
        count, num_sources = row
        print(f"    {count} duplicates: {num_sources:,} source_ids")

    return output_file


def main():
    """Main function."""
    config = get_config()

    print("=" * 70)
    print("EXTRACT PAN-STARRS DUPLICATES")
    print("=" * 70)
    print(f"Project root: {config.project_root}")
    print()

    # Get paths from config
    input_file = config.get_dataset_path('panstarrs_phot', 'external')
    output_file = config.get_dataset_path('panstarrs_duplicates', 'external')

    # Check if input exists
    if not input_file.exists():
        print(f"Error: Input file does not exist:")
        print(f"  {input_file}")
        sys.exit(1)

    result_file = extract_duplicates(input_file, output_file)

    if result_file:
        print("\n" + "=" * 70)
        print("✅ Duplicates extracted successfully!")
        print(f"Output: {result_file.relative_to(config.project_root)}")
        print("=" * 70)
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
