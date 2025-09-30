#!/usr/bin/env python3
"""
Clean Pan-STARRS photometry data by removing duplicates, adding merged duplicates,
and filtering for rows with at least one magnitude pair (g,r), (r,i), or (i,z).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/external")

    # Load the original data with duplicates
    print("Loading original data...")
    original_file = data_dir / "gaia_eb_panstarrs_phot_Dedulek.csv"
    df_original = pd.read_csv(original_file)
    print(f"Original data shape: {df_original.shape}")

    # Load the merged duplicates
    print("Loading merged duplicates...")
    merged_file = data_dir / "gaia_eb_panstarrs_phot_Dedulek_duplicates_merged.csv"
    df_merged = pd.read_csv(merged_file)
    print(f"Merged duplicates shape: {df_merged.shape}")

    # Get the source IDs that are in the merged duplicates file
    merged_source_ids = set(df_merged['original_ext_source_id'].values)
    print(f"Number of unique source IDs in merged duplicates: {len(merged_source_ids)}")

    # Remove duplicate rows from original data
    # Keep rows that are NOT in the merged duplicates
    df_no_duplicates = df_original[~df_original['original_ext_source_id'].isin(merged_source_ids)]
    print(f"Data after removing duplicates: {df_no_duplicates.shape}")

    # Combine the cleaned original data with merged duplicates
    df_combined = pd.concat([df_no_duplicates, df_merged], ignore_index=True)
    print(f"Combined data shape: {df_combined.shape}")

    # Filter for rows with at least one magnitude pair
    print("Filtering for magnitude pairs...")

    # -999.0 indicates missing measurements
    missing_val = -999.0

    # Check for magnitude pairs: (g,r), (r,i), (i,z)
    has_g_r = (df_combined['gPSFMag'] != missing_val) & (df_combined['rPSFMag'] != missing_val)
    has_r_i = (df_combined['rPSFMag'] != missing_val) & (df_combined['iPSFMag'] != missing_val)
    has_i_z = (df_combined['iPSFMag'] != missing_val) & (df_combined['zPSFMag'] != missing_val)

    # Keep rows with at least one of these pairs
    mask = has_g_r | has_r_i | has_i_z
    df_filtered = df_combined[mask]

    print(f"Rows with g,r pair: {has_g_r.sum()}")
    print(f"Rows with r,i pair: {has_r_i.sum()}")
    print(f"Rows with i,z pair: {has_i_z.sum()}")
    print(f"Final filtered data shape: {df_filtered.shape}")

    # Create filter encoding column
    print("Creating filter encoding...")
    def get_filter_encoding(row):
        filters = []
        if row['gPSFMag'] != missing_val:
            filters.append('g')
        if row['rPSFMag'] != missing_val:
            filters.append('r')
        if row['iPSFMag'] != missing_val:
            filters.append('i')
        if row['zPSFMag'] != missing_val:
            filters.append('z')
        return ''.join(filters)

    df_filtered['filter_encoding'] = df_filtered.apply(get_filter_encoding, axis=1)

    # Calculate colors
    print("Calculating colors...")

    # g-r color
    df_filtered['g_r_color'] = np.where(
        (df_filtered['gPSFMag'] != missing_val) & (df_filtered['rPSFMag'] != missing_val),
        df_filtered['gPSFMag'] - df_filtered['rPSFMag'],
        missing_val
    )

    # r-i color
    df_filtered['r_i_color'] = np.where(
        (df_filtered['rPSFMag'] != missing_val) & (df_filtered['iPSFMag'] != missing_val),
        df_filtered['rPSFMag'] - df_filtered['iPSFMag'],
        missing_val
    )

    # i-z color
    df_filtered['i_z_color'] = np.where(
        (df_filtered['iPSFMag'] != missing_val) & (df_filtered['zPSFMag'] != missing_val),
        df_filtered['iPSFMag'] - df_filtered['zPSFMag'],
        missing_val
    )

    # Print filter encoding statistics
    print("\nFilter encoding distribution:")
    encoding_counts = df_filtered['filter_encoding'].value_counts().sort_index()
    for encoding, count in encoding_counts.items():
        print(f"  {encoding}: {count:,} objects")

    # Print color availability
    colors_available = {
        'g-r': (df_filtered['g_r_color'] != missing_val).sum(),
        'r-i': (df_filtered['r_i_color'] != missing_val).sum(),
        'i-z': (df_filtered['i_z_color'] != missing_val).sum()
    }
    print(f"\nColor availability:")
    for color, count in colors_available.items():
        print(f"  {color}: {count:,} objects")

    # Save the final cleaned table
    output_file = data_dir / "gaia_eb_panstarrs_phot_cleaned.csv"
    df_filtered.to_csv(output_file, index=False)
    print(f"\nSaved cleaned data to: {output_file}")
    print(f"Final table shape: {df_filtered.shape}")

    return df_filtered

if __name__ == "__main__":
    df_final = main()