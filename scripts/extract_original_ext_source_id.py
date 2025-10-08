#!/usr/bin/env python3
"""
Script to extract the original_ext_source_id column from the eb table
and save it to a CSV file.

This script reads the eb_catalog.parquet file and extracts only the
original_ext_source_id column, saving it to a CSV file with the same column name.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def extract_original_ext_source_id(input_file, output_file):
    """
    Extract original_ext_source_id column from parquet file and save to CSV.

    Parameters:
    -----------
    input_file : Path
        Path to the input parquet file
    output_file : Path
        Path to the output CSV file
    """
    try:
        # Read the parquet file
        print(f"Reading data from: {input_file}")
        df = pd.read_parquet(input_file)

        # Check if the column exists
        if 'original_ext_source_id' not in df.columns:
            print("Error: 'original_ext_source_id' column not found in the data.")
            print(f"Available columns: {df.columns.tolist()}")
            return False

        # Extract only the original_ext_source_id column
        print(f"Extracting 'original_ext_source_id' column...")
        print(f"Total rows: {len(df):,}")

        # Create a new DataFrame with just the column we want
        result_df = df[['original_ext_source_id']].copy()

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        print(f"Saving to: {output_file}")
        result_df.to_csv(output_file, index=False)

        print(f"✓ Successfully saved {len(result_df):,} rows")
        print(f"  Column name: {result_df.columns[0]}")

        return True

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the extraction."""
    config = get_config()

    print("=" * 70)
    print("EXTRACT ORIGINAL_EXT_SOURCE_ID")
    print("=" * 70)
    print(f"Project root: {config.project_root}")
    print()

    # Get paths from config
    input_file = config.get_dataset_path('eb_catalog_parquet', 'processed')
    output_file = config.get_dataset_path('source_ids', 'processed')

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file does not exist:")
        print(f"  {input_file}")
        print("Please make sure the eb_catalog.parquet file is in the correct location.")
        sys.exit(1)

    # Extract the column
    success = extract_original_ext_source_id(input_file, output_file)

    if success:
        print("\n" + "=" * 70)
        print("✅ Extraction completed successfully!")
        print(f"Output: {output_file.relative_to(config.project_root)}")
        print("=" * 70)
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
