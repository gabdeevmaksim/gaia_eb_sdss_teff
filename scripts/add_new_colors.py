#!/usr/bin/env python3
"""
Script to add new color calculations and temperature estimates to the gaia_eb_panstarrs_phot_with_temperatures table.

This script calculates the following new columns:

New Colors:
- B magnitude: gPSFMag + 0.313*g_r_color + 0.227
- V magnitude: gPSFMag - 0.5784*g_r_color - 0.004
- B-V color: B - V
- R-I color: r_i_color + 0.212
- V-K color: 1.896*B_V + 1.131*R_I

New Temperature Estimates:
- Te_bv: Effective temperature from B-V color using piecewise polynomial relations
- Te_vk: Effective temperature from V-K color using piecewise polynomial relations

The script reads the existing parquet file, adds the new columns, and saves the updated data.
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def calculate_bv_temperature(df: pl.DataFrame) -> pl.Expr:
    """
    Calculate effective temperature from B-V color using piecewise polynomial relations.
    
    Returns a Polars expression that calculates Te_bv based on B_V_color ranges.
    """
    bv = pl.col("B_V_color")
    
    # Initialize with null values
    te_bv = pl.lit(None, dtype=pl.Float64)
    
    # Apply piecewise polynomial relations based on B-V ranges
    te_bv = pl.when((bv >= -0.4) & (bv < -0.24)).then(
        53792.0 + 396497.0 * bv + 1063170.0 * (bv ** 2)
    ).when((bv >= -0.24) & (bv < 0.01)).then(
        9646.0 - 9088.0 * bv + 135032.0 * (bv ** 2)
    ).when((bv >= 0.01) & (bv < 0.31)).then(
        9451.0 - 15715.0 * bv + 50067.0 * (bv ** 2) - 74680.0 * (bv ** 3)
    ).when((bv >= 0.31) & (bv < 1.01)).then(
        8856.0 - 6276.0 * bv + 2777.0 * (bv ** 2) - 485.0 * (bv ** 3)
    ).when((bv >= 1.01) & (bv < 1.61)).then(
        5522.0 + 607.0 * bv - 1253.0 * (bv ** 2) - 13.0 * (bv ** 3)
    ).when((bv >= 1.61) & (bv <= 2.10)).then(
        107429.0 - 173013.0 * bv + 96069.0 * (bv ** 2) - 17901.0 * (bv ** 3)
    ).otherwise(te_bv)
    
    return te_bv


def calculate_vk_temperature(df: pl.DataFrame) -> pl.Expr:
    """
    Calculate effective temperature from V-K color using piecewise polynomial relations.
    
    Returns a Polars expression that calculates Te_vk based on V_K_color ranges.
    """
    vk = pl.col("V_K_color")
    
    # Initialize with null values
    te_vk = pl.lit(None, dtype=pl.Float64)
    
    # Apply piecewise polynomial relations based on V-K ranges
    te_vk = pl.when((vk >= -1.1) & (vk < -0.61)).then(
        38394.0 + 88550.0 * vk + 95683.0 * (vk ** 2)
    ).when((vk >= -0.61) & (vk < 0.01)).then(
        9827.0 - 8566.0 * vk + 11843.0 * (vk ** 2)
    ).when((vk >= 0.1) & (vk < 1.01)).then(
        9710.0 - 6754.0 * vk + 7556.0 * (vk ** 2) - 3829.0 * (vk ** 3)
    ).when((vk >= 1.01) & (vk < 3.01)).then(
        9582.0 - 3709.0 * vk + 969.0 * (vk ** 2) - 108.0 * (vk ** 3)
    ).when((vk >= 3.01) & (vk < 5.01)).then(
        12519.0 - 5335.0 * vk + 1110.0 * (vk ** 2) - 84.0 * (vk ** 3)
    ).when((vk >= 5.01) & (vk < 10.5)).then(
        5373.0 - 724.0 * vk + 74.0 * (vk ** 2) - 3.5 * (vk ** 3)
    ).otherwise(te_vk)
    
    return te_vk


def calculate_new_colors_and_temperatures(input_file: str, output_file: str = None) -> None:
    """
    Calculate new color indices and temperature estimates, then add them to the photometry table.
    
    Parameters:
    -----------
    input_file : str
        Path to the input parquet file
    output_file : str, optional
        Path to save the updated file. If None, overwrites the input file.
    """
    
    print(f"Loading data from {input_file}...")
    df = pl.read_parquet(input_file)
    
    print(f"Loaded {len(df):,} objects")
    print(f"Original columns: {len(df.columns)}")
    
    # Calculate new colors using the provided formulas
    print("Calculating new color indices...")
    
    df_with_colors = df.with_columns([
        # B magnitude: gPSFMag + 0.313*g_r_color + 0.227
        (pl.col("gPSFMag") + 0.313 * pl.col("g_r_color") + 0.227).alias("B_mag"),
        
        # V magnitude: gPSFMag - 0.5784*g_r_color - 0.004
        (pl.col("gPSFMag") - 0.5784 * pl.col("g_r_color") - 0.004).alias("V_mag"),
    ]).with_columns([
        # B-V color: B - V
        (pl.col("B_mag") - pl.col("V_mag")).alias("B_V_color"),
        
        # R-I color: r_i_color + 0.212
        (pl.col("r_i_color") + 0.212).alias("R_I_color"),
    ]).with_columns([
        # V-K color: 1.896*B_V + 1.131*R_I
        (1.896 * pl.col("B_V_color") + 1.131 * pl.col("R_I_color")).alias("V_K_color")
    ])
    
    # Calculate temperature estimates
    print("Calculating temperature estimates from colors...")
    
    df_with_temps = df_with_colors.with_columns([
        # Te_bv: Temperature from B-V color using piecewise polynomials
        calculate_bv_temperature(df_with_colors).alias("Te_bv"),
        
        # Te_vk: Temperature from V-K color using piecewise polynomials
        calculate_vk_temperature(df_with_colors).alias("Te_vk"),
    ])
    
    # Show statistics for the new columns
    print("\nNew color and temperature statistics:")
    new_columns = ["B_mag", "V_mag", "B_V_color", "R_I_color", "V_K_color", "Te_bv", "Te_vk"]
    
    for col in new_columns:
        valid_data = df_with_temps.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
        if len(valid_data) > 0:
            stats = valid_data.select([
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).median().alias("median"),
                pl.col(col).std().alias("std")
            ]).row(0)
            
            print(f"{col}:")
            print(f"  Valid measurements: {len(valid_data):,} / {len(df_with_temps):,} ({100*len(valid_data)/len(df_with_temps):.1f}%)")
            if "Te_" in col:
                print(f"  Range: {stats[0]:.0f} to {stats[1]:.0f} K")
                print(f"  Mean ± std: {stats[2]:.0f} ± {stats[4]:.0f} K")
                print(f"  Median: {stats[3]:.0f} K")
            else:
                print(f"  Range: {stats[0]:.3f} to {stats[1]:.3f}")
                print(f"  Mean ± std: {stats[2]:.3f} ± {stats[4]:.3f}")
                print(f"  Median: {stats[3]:.3f}")
        else:
            print(f"{col}: No valid data")
        print()
    
    # Show coverage statistics for temperature ranges
    print("Temperature calculation coverage by color ranges:")
    
    # B-V coverage
    bv_ranges = [
        ("B-V: -0.4 to -0.24", (-0.4, -0.24)),
        ("B-V: -0.24 to 0.01", (-0.24, 0.01)),
        ("B-V: 0.01 to 0.31", (0.01, 0.31)),
        ("B-V: 0.31 to 1.01", (0.31, 1.01)),
        ("B-V: 1.01 to 1.61", (1.01, 1.61)),
        ("B-V: 1.61 to 2.10", (1.61, 2.10))
    ]
    
    for range_name, (min_val, max_val) in bv_ranges:
        count = len(df_with_temps.filter(
            (pl.col("B_V_color") >= min_val) & 
            (pl.col("B_V_color") < max_val) if max_val != 2.10 else 
            (pl.col("B_V_color") >= min_val) & (pl.col("B_V_color") <= max_val)
        ))
        print(f"  {range_name}: {count:,} objects")
    
    # V-K coverage
    vk_ranges = [
        ("V-K: -1.1 to -0.61", (-1.1, -0.61)),
        ("V-K: -0.61 to 0.01", (-0.61, 0.01)),
        ("V-K: 0.1 to 1.01", (0.1, 1.01)),
        ("V-K: 1.01 to 3.01", (1.01, 3.01)),
        ("V-K: 3.01 to 5.01", (3.01, 5.01)),
        ("V-K: 5.01 to 10.5", (5.01, 10.5))
    ]
    
    for range_name, (min_val, max_val) in vk_ranges:
        count = len(df_with_temps.filter(
            (pl.col("V_K_color") >= min_val) & 
            (pl.col("V_K_color") < max_val) if max_val != 10.5 else 
            (pl.col("V_K_color") >= min_val) & (pl.col("V_K_color") <= max_val)
        ))
        print(f"  {range_name}: {count:,} objects")
    
    # Determine output file
    if output_file is None:
        output_file = input_file
    
    print(f"\nSaving updated data to {output_file}...")
    df_with_temps.write_parquet(output_file)
    
    print(f"Updated file saved with {len(df_with_temps.columns)} columns")
    print(f"New columns added: {', '.join(new_columns)}")
    
    return df_with_temps


def main():
    """Main function to run the color calculation script."""
    config = get_config()

    print("=" * 70)
    print("ADD NEW COLORS AND TEMPERATURES")
    print("=" * 70)
    print(f"Project root: {config.project_root}")
    print()

    # Get file paths from config
    input_file = config.get_dataset_path('panstarrs_with_temps', 'processed')

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found:")
        print(f"  {input_file}")
        sys.exit(1)

    # Create backup before modifying
    backup_file = input_file.parent / f"{input_file.stem}_backup.parquet"
    print(f"Creating backup...")
    print(f"  {backup_file.relative_to(config.project_root)}")

    # Read and save backup
    df_backup = pl.read_parquet(input_file)
    df_backup.write_parquet(backup_file)
    print(f"  ✓ Backup created")

    # Calculate new colors and temperatures
    try:
        print("\nCalculating new colors and temperatures...")
        df_updated = calculate_new_colors_and_temperatures(str(input_file))

        print("\n" + "=" * 70)
        print("✅ Color and temperature calculation completed successfully!")
        print(f"  Rows: {len(df_updated):,}")
        print(f"  Columns: {len(df_updated.columns)}")
        print(f"  Output: {input_file.relative_to(config.project_root)}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during calculation: {e}")
        print(f"Backup preserved at: {backup_file.relative_to(config.project_root)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
