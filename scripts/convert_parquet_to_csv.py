#!/usr/bin/env python3
"""
Convert Parquet file to CSV format.

This is a standalone script that checks for required packages and provides
installation instructions if they are missing.

Usage:
    python convert_parquet_to_csv.py input.parquet output.csv
    python convert_parquet_to_csv.py input.parquet output.dat
    python convert_parquet_to_csv.py input.parquet  # Auto-generates output name

Author: Auto-generated
Date: 2025-10-14
"""

import sys
from pathlib import Path


def check_and_import_packages():
    """
    Check if required packages are installed and import them.

    Returns
    -------
    tuple
        (success: bool, pandas_module or None, error_message or None)
    """
    missing_packages = []

    # Check for pandas
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append('pandas')
        pd = None

    # Check for pyarrow (required for parquet)
    try:
        import pyarrow
    except ImportError:
        missing_packages.append('pyarrow')

    if missing_packages:
        error_msg = f"""
{'='*70}
ERROR: Required packages are missing!
{'='*70}

Missing packages: {', '.join(missing_packages)}

To install the required packages, run ONE of the following commands:

Option 1 - Using pip:
    pip install pandas pyarrow

Option 2 - Using conda:
    conda install pandas pyarrow

Option 3 - Install from requirements.txt (if available):
    pip install -r requirements.txt

After installation, run this script again.
{'='*70}
"""
        return False, None, error_msg

    return True, pd, None


def convert_parquet_to_csv(input_file: Path, output_file: Path, pd):
    """
    Convert a Parquet file to CSV format.

    Parameters
    ----------
    input_file : Path
        Path to input Parquet file
    output_file : Path
        Path to output CSV/DAT file
    pd : module
        Pandas module

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"Reading Parquet file: {input_file}")
        print(f"  File size: {input_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Read Parquet file
        df = pd.read_parquet(input_file)

        print(f"  Loaded {len(df):,} rows")
        print(f"  Columns ({len(df.columns)}): {', '.join(df.columns)}")

        # Process Teff column if it exists
        if 'Teff' in df.columns:
            print(f"\nProcessing Teff column...")

            # Round Teff to 1 decimal place where present
            df['Teff'] = df['Teff'].round(1)

            # Replace NaN/None with '--'
            df['Teff'] = df['Teff'].fillna('--')

            # Convert to string and keep '--' for missing values
            df['Teff'] = df['Teff'].astype(str)
            df['Teff'] = df['Teff'].replace('nan', '--')

            print(f"  ✓ Rounded temperatures to 1 decimal place")
            print(f"  ✓ Replaced missing values with '--'")

        # Convert to CSV
        print(f"\nWriting CSV file: {output_file}")
        df.to_csv(output_file, index=False)

        output_size = output_file.stat().st_size / 1024 / 1024
        print(f"  Output size: {output_size:.2f} MB")
        print(f"\n{'='*70}")
        print("SUCCESS: Conversion completed!")
        print(f"{'='*70}")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print(f"Rows:   {len(df):,}")

        return True

    except FileNotFoundError:
        print(f"\n{'='*70}")
        print(f"ERROR: Input file not found!")
        print(f"{'='*70}")
        print(f"File: {input_file}")
        print(f"Please check that the file path is correct.")
        return False

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Conversion failed!")
        print(f"{'='*70}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return False


def main():
    """Main function."""

    # Print header
    print(f"{'='*70}")
    print("PARQUET TO CSV CONVERTER")
    print(f"{'='*70}\n")

    # Check for required packages
    print("Checking required packages...")
    success, pd, error_msg = check_and_import_packages()

    if not success:
        print(error_msg)
        sys.exit(1)

    print("  ✓ All required packages found\n")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_parquet_to_csv.py input.parquet output.csv")
        print("  python convert_parquet_to_csv.py input.parquet output.dat")
        print("  python convert_parquet_to_csv.py input.parquet  # Auto-generates output name")
        print("\nExamples:")
        print("  python convert_parquet_to_csv.py data.parquet data.csv")
        print("  python convert_parquet_to_csv.py data.parquet data.dat")
        print("  python convert_parquet_to_csv.py data.parquet")
        sys.exit(1)

    # Input file
    input_file = Path(sys.argv[1])

    # Output file (auto-generate if not provided)
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Auto-generate output filename (.dat extension)
        output_file = input_file.with_suffix('.dat')
        print(f"Output file not specified, using: {output_file}\n")

    # Check if input file exists
    if not input_file.exists():
        print(f"{'='*70}")
        print("ERROR: Input file does not exist!")
        print(f"{'='*70}")
        print(f"File: {input_file}")
        print(f"Absolute path: {input_file.absolute()}")
        print("\nPlease check that the file path is correct.")
        sys.exit(1)

    # Check if input file is Parquet
    if input_file.suffix.lower() not in ['.parquet', '.pq']:
        print(f"{'='*70}")
        print("WARNING: Input file does not have .parquet extension")
        print(f"{'='*70}")
        print(f"File: {input_file}")
        print(f"Extension: {input_file.suffix}")
        print("\nContinuing anyway...\n")

    # Convert
    success = convert_parquet_to_csv(input_file, output_file, pd)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
