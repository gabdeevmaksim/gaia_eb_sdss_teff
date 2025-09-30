#!/usr/bin/env python3
"""
Calculate effective temperatures from Pan-STARRS colors using empirical relations.
Applies color constraints (>= -0.5) and computes average temperatures.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data/external")

    # Load the cleaned photometry data
    print("Loading cleaned photometry data...")
    input_file = data_dir / "gaia_eb_panstarrs_phot_cleaned.csv"
    df = pd.read_csv(input_file)
    print(f"Input data shape: {df.shape}")

    # Missing value indicator
    missing_val = -999.0

    # Calculate temperatures with color constraints
    print("Calculating temperatures...")

    # Te_gr: Only for g_r_color >= -0.5
    df['Te_gr'] = np.where(
        (df['g_r_color'] != missing_val) & (df['g_r_color'] >= -0.5),
        1.09 / (df['g_r_color'] + 1.47) * 1e4,
        missing_val
    )

    # Te_ri: Only for r_i_color >= -0.5
    df['Te_ri'] = np.where(
        (df['r_i_color'] != missing_val) & (df['r_i_color'] >= -0.5),
        0.5 / (df['r_i_color'] + 0.78) * 1e4,
        missing_val
    )

    # Te_iz: Only for i_z_color >= -0.5
    df['Te_iz'] = np.where(
        (df['i_z_color'] != missing_val) & (df['i_z_color'] >= -0.5),
        0.38 / (df['i_z_color'] + 0.64) * 1e4,
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
    output_file = data_dir / "gaia_eb_panstarrs_phot_with_temperatures.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved data with temperatures to: {output_file}")
    print(f"Final table shape: {df.shape}")

    return df

if __name__ == "__main__":
    df_final = main()