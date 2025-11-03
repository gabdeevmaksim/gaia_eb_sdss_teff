#!/usr/bin/env python3
"""
Create Gaia G + BP-RP dataset for temperature prediction.

Features:
- phot_g_mean_mag (Gaia G-band magnitude)
- bp_rp (Gaia BP-RP color)

This is a minimal 2-feature model to test the contribution of just Gaia photometry.

Outputs:
- gaia_g_bprp_train.parquet (objects with teff_gspphot)
- gaia_g_bprp_predict.parquet (objects without teff_gspphot)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from src.config import get_config

def main():
    config = get_config()

    print("=" * 80)
    print("Creating Gaia G + BP-RP Dataset")
    print("=" * 80)

    # Load Gaia catalog
    print("\n1. Loading Gaia catalog...")
    gaia_path = config.get_path('processed') / 'eb_catalog.parquet'
    df = pl.read_parquet(gaia_path)
    print(f"   Loaded {len(df):,} objects")

    # Select required columns
    required_cols = [
        'source_id',
        'phot_g_mean_mag',
        'bp_rp',
        'teff_gspphot'
    ]

    print("\n2. Selecting features...")
    df = df.select(required_cols)

    # Remove rows with missing or invalid values
    initial_count = len(df)
    print(f"\n3. Filtering for complete data...")

    df = df.filter(
        pl.col('phot_g_mean_mag').is_not_null() &
        pl.col('phot_g_mean_mag').is_finite() &
        pl.col('bp_rp').is_not_null() &
        pl.col('bp_rp').is_finite() &
        (pl.col('bp_rp') != 0)  # Remove bp_rp = 0
    )

    print(f"   After filtering: {len(df):,} objects ({len(df)/initial_count*100:.1f}%)")

    # Remove outliers (BP-RP beyond reasonable range)
    print("\n4. Removing color outliers...")
    before_outlier_filter = len(df)

    df = df.filter(
        (pl.col('bp_rp') >= -0.5) & (pl.col('bp_rp') <= 5.0)
    )

    print(f"   After outlier removal: {len(df):,} objects ({len(df)/before_outlier_filter*100:.1f}%)")

    # Split into train (with teff) and predict (without teff)
    print("\n5. Splitting into train and predict sets...")

    df_train = df.filter(
        pl.col('teff_gspphot').is_not_null() &
        pl.col('teff_gspphot').is_finite() &
        (pl.col('teff_gspphot') > 0)
    )

    df_predict = df.filter(
        pl.col('teff_gspphot').is_null() |
        ~pl.col('teff_gspphot').is_finite() |
        (pl.col('teff_gspphot') <= 0)
    ).drop('teff_gspphot')

    print(f"   Training set: {len(df_train):,} objects")
    print(f"   Prediction set: {len(df_predict):,} objects")

    # Display statistics
    print("\n6. Dataset statistics:")
    print("\n   Training set:")
    for col in ['phot_g_mean_mag', 'bp_rp', 'teff_gspphot']:
        if col in df_train.columns:
            mean_val = df_train[col].mean()
            std_val = df_train[col].std()
            min_val = df_train[col].min()
            max_val = df_train[col].max()
            print(f"     {col:20s}: mean={mean_val:.3f}, std={std_val:.3f}, "
                  f"min={min_val:.3f}, max={max_val:.3f}")

    print("\n   Prediction set:")
    for col in ['phot_g_mean_mag', 'bp_rp']:
        mean_val = df_predict[col].mean()
        std_val = df_predict[col].std()
        min_val = df_predict[col].min()
        max_val = df_predict[col].max()
        print(f"     {col:20s}: mean={mean_val:.3f}, std={std_val:.3f}, "
              f"min={min_val:.3f}, max={max_val:.3f}")

    # Save datasets
    print("\n7. Saving datasets...")

    train_path = config.get_path('processed') / 'gaia_g_bprp_train.parquet'
    df_train.write_parquet(train_path)
    print(f"   Saved training set: {train_path}")
    print(f"   Size: {train_path.stat().st_size / 1024 / 1024:.1f} MB")

    predict_path = config.get_path('processed') / 'gaia_g_bprp_predict.parquet'
    df_predict.write_parquet(predict_path)
    print(f"   Saved prediction set: {predict_path}")
    print(f"   Size: {predict_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print(f"\nTotal objects: {len(df):,}")
    print(f"  Training: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Prediction: {len(df_predict):,} ({len(df_predict)/len(df)*100:.1f}%)")
    print(f"\nFeatures: phot_g_mean_mag, bp_rp")
    print(f"Target: teff_gspphot")

if __name__ == '__main__':
    main()
