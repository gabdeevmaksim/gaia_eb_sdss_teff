#!/usr/bin/env python3
"""
Apply Teff correction to training dataset

Creates a new dataset with corrected Teff values for comparison experiment.
Corrects Gaia Teff > 10000K using polynomial coefficients.
"""

import pickle
import numpy as np
import polars as pl
from pathlib import Path

def load_correction_coefficients():
    """Load Teff correction polynomial"""
    coeff_path = Path("data/teff_correction_coeffs_deg2.pkl")
    with open(coeff_path, 'rb') as f:
        coeffs = pickle.load(f)

    print(f"Loaded correction coefficients:")
    print(f"  Threshold: {coeffs['teff_threshold']}K")
    print(f"  Polynomial degree: {coeffs['degree']}")
    print(f"  RMS: {coeffs['rms']:.1f}K")

    return coeffs['polynomial'], coeffs['teff_threshold']


def apply_correction(df, teff_col, poly, threshold):
    """
    Apply correction to Teff column

    Args:
        df: Polars DataFrame
        teff_col: Name of temperature column
        poly: numpy.polynomial.Polynomial
        threshold: Temperature threshold for correction

    Returns:
        DataFrame with new corrected column
    """
    # Get Teff values as numpy array
    teff = df[teff_col].to_numpy()

    # Apply correction
    teff_corrected = teff.copy()
    mask = teff > threshold

    if np.any(mask):
        teff_corrected[mask] = poly(teff[mask])

    n_corrected = np.sum(mask)
    pct_corrected = 100 * n_corrected / len(teff)

    print(f"  Applied correction to {n_corrected:,} values ({pct_corrected:.1f}%)")
    print(f"  Mean change for corrected values: {np.mean(teff_corrected[mask] - teff[mask]):.1f}K")

    # Add corrected column and log-corrected column
    df = df.with_columns([
        pl.Series(name=f'{teff_col}_corrected', values=teff_corrected),
        pl.Series(name=f'{teff_col}_corrected_log', values=np.log10(teff_corrected))
    ])

    return df


def main():
    print("=" * 80)
    print("Apply Teff Correction to Training Data")
    print("=" * 80)

    # Load data (use gaia_all_colors_train which has all 6 Gaia color features)
    input_path = Path('data/processed/gaia_all_colors_train.parquet')
    print(f"\nLoading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns}")

    # Load correction
    poly, threshold = load_correction_coefficients()

    # Apply correction
    print(f"\nApplying correction (threshold: {threshold}K)...")
    df = apply_correction(df, 'teff_gaia', poly, threshold)

    # Save corrected dataset
    output_path = Path('data/processed/gaia_all_colors_teff_corrected.parquet')
    df.write_parquet(output_path)
    print(f"\nSaved: {output_path}")
    print(f"New columns: teff_gaia_corrected, teff_gaia_corrected_log")
    print(f"All Gaia features: g, bp, rp, bp_rp, g_bp, g_rp")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    stats = df.select([
        pl.col('teff_gaia').mean().alias('Original_Mean'),
        pl.col('teff_gaia').median().alias('Original_Median'),
        pl.col('teff_gaia_corrected').mean().alias('Corrected_Mean'),
        pl.col('teff_gaia_corrected').median().alias('Corrected_Median'),
        pl.col('teff_gaia_corrected_log').mean().alias('Log_Corrected_Mean'),
        (pl.col('teff_gaia_corrected') - pl.col('teff_gaia')).mean().alias('Mean_Difference'),
    ])

    print(stats)

    # Statistics by temperature range
    print("\nStatistics by temperature range:")
    for tmin, tmax, label in [(0, 6000, 'Cool'), (6000, 10000, 'Mid'),
                                (10000, 15000, 'Hot'), (15000, 50000, 'Very Hot')]:
        mask = (df['teff_gaia'] >= tmin) & (df['teff_gaia'] < tmax)
        n = mask.sum()
        if n > 0:
            df_range = df.filter(mask)
            mean_diff = (df_range['teff_gaia_corrected'] - df_range['teff_gaia']).mean()
            print(f"  {label:10s} ({tmin:5d}-{tmax:5d}K): n={n:7,}, mean_diff={mean_diff:7.1f}K")


if __name__ == '__main__':
    main()
