#!/usr/bin/env python3
"""
Calculate effective temperatures from Pan-STARRS colors using empirical relations.
Applies color constraints (>= -0.5) and computes average temperatures.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def main():
    # Load configuration
    config = get_config()

    # Get paths from config
    input_file = config.get_dataset_path('panstarrs_cleaned', 'external')
    output_file = config.get_dataset_path('panstarrs_with_temps', 'external')

    # Get parameters from config
    missing_val = config.get('processing', 'missing_value')
    color_threshold = config.get('processing', 'color_threshold')

    # Get temperature calculation coefficients
    gr_coef = config.get('temperature', 'gr_coefficients')
    ri_coef = config.get('temperature', 'ri_coefficients')
    iz_coef = config.get('temperature', 'iz_coefficients')

    # Load the cleaned photometry data
    print("=" * 70)
    print("TEMPERATURE CALCULATION")
    print("=" * 70)
    print(f"Input:  {input_file.relative_to(config.project_root)}")
    print(f"Output: {output_file.relative_to(config.project_root)}")
    print()

    print("Loading cleaned photometry data...")
    df = pd.read_csv(input_file)
    print(f"Input data shape: {df.shape}")

    # Calculate temperatures with color constraints
    print("\nCalculating temperatures...")

    # Te_gr: Only for g_r_color >= color_threshold
    df['Te_gr'] = np.where(
        (df['g_r_color'] != missing_val) & (df['g_r_color'] >= color_threshold),
        gr_coef['A'] / (df['g_r_color'] + gr_coef['B']) * 1e4,
        missing_val
    )

    # Te_ri: Only for r_i_color >= color_threshold
    df['Te_ri'] = np.where(
        (df['r_i_color'] != missing_val) & (df['r_i_color'] >= color_threshold),
        ri_coef['A'] / (df['r_i_color'] + ri_coef['B']) * 1e4,
        missing_val
    )

    # Te_iz: Only for i_z_color >= color_threshold
    df['Te_iz'] = np.where(
        (df['i_z_color'] != missing_val) & (df['i_z_color'] >= color_threshold),
        iz_coef['A'] / (df['i_z_color'] + iz_coef['B']) * 1e4,
        missing_val
    )

    # Calculate average temperature from available measurements
    def calculate_avg_temperature(row):
        temps = [row['Te_gr'], row['Te_ri'], row['Te_iz']]
        valid_temps = [t for t in temps if t != missing_val]
        return np.mean(valid_temps) if valid_temps else missing_val

    df['Te_avg'] = df.apply(calculate_avg_temperature, axis=1)

    # Count temperature measurements per object
    def count_temperatures(row):
        temps = [row['Te_gr'], row['Te_ri'], row['Te_iz']]
        return sum(1 for t in temps if t != missing_val)

    df['temp_measurement_count'] = df.apply(count_temperatures, axis=1)

    # Print statistics
    print(f"\nTemperature availability:")
    temp_stats = {
        'Te_gr': (df['Te_gr'] != missing_val).sum(),
        'Te_ri': (df['Te_ri'] != missing_val).sum(),
        'Te_iz': (df['Te_iz'] != missing_val).sum(),
        'Te_avg': (df['Te_avg'] != missing_val).sum()
    }
    for temp, count in temp_stats.items():
        print(f"  {temp}: {count:,} objects")

    print(f"\nObjects by number of temperature measurements:")
    for i in range(4):
        count = (df['temp_measurement_count'] == i).sum()
        print(f"  {i} temperatures: {count:,} objects")

    # Print temperature ranges for valid measurements
    print(f"\nTemperature ranges:")
    for temp_col in ['Te_gr', 'Te_ri', 'Te_iz', 'Te_avg']:
        valid_temps = df[df[temp_col] != missing_val][temp_col]
        if len(valid_temps) > 0:
            print(f"  {temp_col}: {valid_temps.min():.0f} - {valid_temps.max():.0f} K (median: {valid_temps.median():.0f} K)")

    # Save the data with temperatures
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved data with temperatures")
    print(f"  Final table shape: {df.shape}")
    print("=" * 70)

    return df

if __name__ == "__main__":
    df_final = main()
