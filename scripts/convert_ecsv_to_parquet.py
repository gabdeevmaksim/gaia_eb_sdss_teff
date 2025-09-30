#!/usr/bin/env python3
"""
Convert ECSV catalog files to Parquet format for faster loading.

This script converts the eclipsing binary catalog from ECSV format
to Parquet format, preserving data types and metadata where possible.
"""

import sys
from pathlib import Path
import argparse
from astropy.table import Table
import polars as pl
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def convert_ecsv_to_parquet(input_path: Path, output_path: Path = None, verbose: bool = True):
    """
    Convert ECSV file to Parquet format.
    
    Parameters
    ----------
    input_path : Path
        Path to input ECSV file
    output_path : Path, optional
        Path for output Parquet file. If None, uses same name with .parquet extension
    verbose : bool, default True
        Whether to print progress information
    """
    if output_path is None:
        output_path = input_path.with_suffix('.parquet')
    
    if verbose:
        print(f"Converting {input_path.name} to Parquet format...")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
    
    # Load ECSV with astropy to preserve astronomical metadata
    start_time = time.time()
    if verbose:
        print("\n1. Loading ECSV file with astropy...")
    
    table = Table.read(input_path, format='ascii.ecsv')
    load_time = time.time() - start_time
    
    if verbose:
        print(f"   ✓ Loaded {len(table):,} rows with {len(table.colnames)} columns in {load_time:.1f}s")
    
    # Convert to Polars DataFrame for efficient handling
    start_time = time.time()
    if verbose:
        print("\n2. Converting to Polars DataFrame...")
    
    # Convert via pandas (astropy -> pandas -> polars)
    pandas_df = table.to_pandas()
    polars_df = pl.from_pandas(pandas_df)
    convert_time = time.time() - start_time
    
    if verbose:
        print(f"   ✓ Converted to Polars in {convert_time:.1f}s")
        print(f"   Data shape: {polars_df.height:,} rows × {polars_df.width} columns")
    
    # Save as Parquet
    start_time = time.time()
    if verbose:
        print(f"\n3. Saving as Parquet to {output_path.name}...")
    
    polars_df.write_parquet(output_path)
    save_time = time.time() - start_time
    
    if verbose:
        print(f"   ✓ Saved Parquet file in {save_time:.1f}s")
    
    # Compare file sizes
    input_size = input_path.stat().st_size / (1024**2)  # MB
    output_size = output_path.stat().st_size / (1024**2)  # MB
    compression_ratio = input_size / output_size
    
    if verbose:
        print(f"\n4. Conversion Summary:")
        print(f"   Input size:  {input_size:.1f} MB")
        print(f"   Output size: {output_size:.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}x smaller")
        print(f"   Total time:  {load_time + convert_time + save_time:.1f}s")
    
    return output_path


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Convert ECSV catalog files to Parquet format"
    )
    parser.add_argument(
        "input_file", 
        type=Path,
        help="Path to input ECSV file"
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path,
        help="Output Parquet file path (default: same name with .parquet extension)"
    )
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    try:
        output_path = convert_ecsv_to_parquet(
            args.input_file, 
            args.output, 
            verbose=not args.quiet
        )
        print(f"\n✅ Successfully converted to {output_path}")
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
