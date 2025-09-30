#!/usr/bin/env python3
"""
Script to extract the original_ext_source_id column from the eb table
and save it to a CSV file.

This script reads the eb_catalog.parquet file and extracts only the
original_ext_source_id column, saving it to a CSV file with the same column name.
"""

import pandas as pd
import os
import sys
from pathlib import Path

def extract_original_ext_source_id(input_file, output_file):
    """
    Extract original_ext_source_id column from parquet file and save to CSV.
    
    Parameters:
    -----------
    input_file : str
        Path to the input parquet file
    output_file : str
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
        print(f"Total rows: {len(df)}")
        
        # Create a new DataFrame with just the column we want
        result_df = df[['original_ext_source_id']].copy()
        
        # Save to CSV
        print(f"Saving to: {output_file}")
        result_df.to_csv(output_file, index=False)
        
        print(f"Successfully saved {len(result_df)} rows to {output_file}")
        print(f"Column name: {result_df.columns[0]}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Main function to run the extraction."""
    # Define file paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "processed" / "eb_catalog.parquet"
    output_file = project_root / "data" / "processed" / "original_ext_source_id.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        print("Please make sure the eb_catalog.parquet file is in the correct location.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract the column
    success = extract_original_ext_source_id(str(input_file), str(output_file))
    
    if success:
        print("\n✅ Extraction completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

