#!/usr/bin/env python3
"""
Create engineered feature dataset from Gaia G + BP-RP.

Base features:
- phot_g_mean_mag (Gaia G-band magnitude)
- bp_rp (Gaia BP-RP color)

Engineered features:
- Polynomial terms (squared, cubed)
- Interaction terms
- Logarithmic transforms
- Temperature-dependent features (hot/cool/mid star indicators)

Outputs:
- gaia_g_bprp_engineered_train.parquet
- gaia_g_bprp_engineered_predict.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np

def engineer_features(df):
    """
    Engineer features from phot_g_mean_mag and bp_rp.

    Creates polynomial, interaction, logarithmic, and temperature-dependent features.
    """

    # Start with base features
    result = df.clone()

    # === POLYNOMIAL FEATURES ===
    # BP-RP powers
    result = result.with_columns([
        (pl.col('bp_rp') ** 2).alias('bp_rp^2'),
        (pl.col('bp_rp') ** 3).alias('bp_rp^3'),
    ])

    # G magnitude powers
    result = result.with_columns([
        (pl.col('phot_g_mean_mag') ** 2).alias('g_mag^2'),
        (pl.col('phot_g_mean_mag') ** 3).alias('g_mag^3'),
    ])

    # === INTERACTION FEATURES ===
    # G mag × BP-RP interactions
    result = result.with_columns([
        (pl.col('phot_g_mean_mag') * pl.col('bp_rp')).alias('g_mag_x_bp_rp'),
        (pl.col('phot_g_mean_mag') * pl.col('bp_rp^2')).alias('g_mag_bp_rp^2'),
        (pl.col('phot_g_mean_mag') ** 2 * pl.col('bp_rp')).alias('g_mag^2_bp_rp'),
    ])

    # === LOGARITHMIC FEATURES ===
    # Handle negative/zero values by adding offset
    bp_rp_shifted = pl.col('bp_rp') + 1.0  # Shift to ensure positive values
    result = result.with_columns([
        bp_rp_shifted.log().alias('log_bp_rp'),
    ])

    # === TEMPERATURE-DEPENDENT FEATURES ===
    # These encode color behavior that varies with stellar temperature
    # Hot stars: bp_rp < 0.5 (roughly > 8000 K)
    # Cool stars: bp_rp > 2.0 (roughly < 5000 K)
    # Mid stars: 0.5 <= bp_rp <= 2.0

    # Hot star indicators
    result = result.with_columns([
        pl.when(pl.col('bp_rp') < 0.5)
          .then(pl.col('bp_rp'))
          .otherwise(0.0)
          .alias('hot_bp_rp'),
        pl.when(pl.col('bp_rp') < 0.5)
          .then(pl.col('phot_g_mean_mag'))
          .otherwise(0.0)
          .alias('hot_g_mag'),
    ])

    # Cool star indicators
    result = result.with_columns([
        pl.when(pl.col('bp_rp') > 2.0)
          .then(pl.col('bp_rp'))
          .otherwise(0.0)
          .alias('cool_bp_rp'),
        pl.when(pl.col('bp_rp') > 2.0)
          .then(pl.col('phot_g_mean_mag'))
          .otherwise(0.0)
          .alias('cool_g_mag'),
    ])

    # Mid-range star indicators
    result = result.with_columns([
        pl.when((pl.col('bp_rp') >= 0.5) & (pl.col('bp_rp') <= 2.0))
          .then(pl.col('bp_rp'))
          .otherwise(0.0)
          .alias('mid_bp_rp'),
        pl.when((pl.col('bp_rp') >= 0.5) & (pl.col('bp_rp') <= 2.0))
          .then(pl.col('phot_g_mean_mag'))
          .otherwise(0.0)
          .alias('mid_g_mag'),
    ])

    # === RATIO FEATURES ===
    # Color-to-magnitude ratio (can capture distance/extinction effects)
    result = result.with_columns([
        (pl.col('bp_rp') / (pl.col('phot_g_mean_mag') + 1.0)).alias('bp_rp_per_g_mag'),
    ])

    return result

def main():
    print("=" * 80)
    print("Creating Gaia G + BP-RP Engineered Features Dataset")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / 'data' / 'processed'

    # Load base datasets
    print("\n1. Loading base datasets...")
    df_train = pl.read_parquet(data_dir / 'gaia_g_bprp_train.parquet')
    df_predict = pl.read_parquet(data_dir / 'gaia_g_bprp_predict.parquet')

    print(f"   Training: {len(df_train):,} objects")
    print(f"   Prediction: {len(df_predict):,} objects")

    # Engineer features for training set
    print("\n2. Engineering features for training set...")
    df_train_eng = engineer_features(df_train)
    n_features = len(df_train_eng.columns) - 2  # Exclude source_id and teff_gspphot
    print(f"   Created {n_features} features (from 2 base features)")

    # Engineer features for prediction set
    print("\n3. Engineering features for prediction set...")
    df_predict_eng = engineer_features(df_predict)

    # Display feature list
    print("\n4. Engineered features:")
    feature_cols = [c for c in df_train_eng.columns if c not in ['source_id', 'teff_gspphot']]
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")

    # Check for NaN/Inf values
    print("\n5. Checking for invalid values...")
    for col in feature_cols:
        n_null = df_train_eng[col].is_null().sum()
        n_inf = (~df_train_eng[col].is_finite()).sum()
        if n_null > 0 or n_inf > 0:
            print(f"   WARNING: {col} has {n_null} nulls, {n_inf} inf values")

    # Statistics
    print("\n6. Feature statistics (training set):")
    sample_features = ['phot_g_mean_mag', 'bp_rp', 'bp_rp^2', 'g_mag_x_bp_rp', 'log_bp_rp']
    for feat in sample_features:
        if feat in df_train_eng.columns:
            mean_val = df_train_eng[feat].mean()
            std_val = df_train_eng[feat].std()
            min_val = df_train_eng[feat].min()
            max_val = df_train_eng[feat].max()
            print(f"   {feat:20s}: mean={mean_val:8.3f}, std={std_val:8.3f}, "
                  f"min={min_val:8.3f}, max={max_val:8.3f}")

    # Save datasets
    print("\n7. Saving engineered datasets...")

    train_path = data_dir / 'gaia_g_bprp_engineered_train.parquet'
    df_train_eng.write_parquet(train_path)
    print(f"   Saved training set: {train_path}")
    print(f"   Size: {train_path.stat().st_size / 1024 / 1024:.1f} MB")

    predict_path = data_dir / 'gaia_g_bprp_engineered_predict.parquet'
    df_predict_eng.write_parquet(predict_path)
    print(f"   Saved prediction set: {predict_path}")
    print(f"   Size: {predict_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 80)
    print("Feature engineering complete!")
    print("=" * 80)
    print(f"\nBase features: 2 (phot_g_mean_mag, bp_rp)")
    print(f"Engineered features: {n_features}")
    print(f"Total features: {n_features}")
    print(f"\nFeature categories:")
    print(f"  - Polynomial (6): bp_rp^2, bp_rp^3, g_mag^2, g_mag^3, ...")
    print(f"  - Interactions (3): g_mag × bp_rp, g_mag × bp_rp^2, g_mag^2 × bp_rp")
    print(f"  - Logarithmic (1): log(bp_rp)")
    print(f"  - Temperature-dependent (6): hot/cool/mid indicators for color and magnitude")
    print(f"  - Ratio (1): bp_rp per g_mag")

if __name__ == '__main__':
    main()
