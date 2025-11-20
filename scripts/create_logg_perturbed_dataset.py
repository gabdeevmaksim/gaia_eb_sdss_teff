"""
Create expanded dataset with logg perturbations for uncertainty propagation.

For each object, create 3 variants:
  - Baseline: logg = logg_predicted
  - Plus: logg = logg_predicted + logg_uncertainty
  - Minus: logg = logg_predicted - logg_uncertainty

This allows us to compute numerical gradient ∂Teff/∂logg using the prediction pipeline.
"""

import polars as pl
from pathlib import Path

def main():
    print("=" * 80)
    print("CREATING LOGG-PERTURBED DATASET")
    print("=" * 80)

    # Load data with predicted logg
    print("\n[1/3] Loading data with predicted logg...")
    data_path = Path('data/processed/data_for_teff_prediction_with_logg.parquet')
    df = pl.read_parquet(data_path)
    print(f"  Loaded {len(df):,} objects")

    # Get logg values
    logg_predicted = df['logg_gaia'].to_numpy()
    logg_uncertainty = df['logg_uncertainty'].to_numpy()

    print(f"  Mean logg: {logg_predicted.mean():.3f} dex")
    print(f"  Mean logg_unc: {logg_uncertainty.mean():.3f} dex")

    # Create 3 variants per object
    print("\n[2/3] Creating 3 variants per object...")

    # Baseline variant (keep as is)
    df_baseline = df.with_columns([
        pl.lit(0).alias('variant'),  # 0 = baseline
        pl.Series('original_index', range(len(df)))
    ])

    # Plus variant (logg + uncertainty)
    df_plus = df.with_columns([
        (pl.col('logg_gaia') + pl.col('logg_uncertainty')).alias('logg_gaia'),
        pl.lit(1).alias('variant'),  # 1 = plus
        pl.Series('original_index', range(len(df)))
    ])

    # Minus variant (logg - uncertainty)
    df_minus = df.with_columns([
        (pl.col('logg_gaia') - pl.col('logg_uncertainty')).alias('logg_gaia'),
        pl.lit(2).alias('variant'),  # 2 = minus
        pl.Series('original_index', range(len(df)))
    ])

    # Concatenate all variants
    df_expanded = pl.concat([df_baseline, df_plus, df_minus])

    print(f"  Original objects: {len(df):,}")
    print(f"  Expanded rows: {len(df_expanded):,} (3x)")
    print(f"  Variants:")
    print(f"    0 (baseline): {len(df_baseline):,}")
    print(f"    1 (+σ):       {len(df_plus):,}")
    print(f"    2 (-σ):       {len(df_minus):,}")

    # Save expanded dataset
    print("\n[3/3] Saving expanded dataset...")
    output_path = Path('data/processed/data_for_teff_logg_perturbed.parquet')
    df_expanded.write_parquet(output_path)

    print(f"✓ Saved: {output_path}")
    print(f"  Rows: {len(df_expanded):,}")
    print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")

    # Print column info
    print(f"\n  Columns ({len(df_expanded.columns)}):")
    for col in df_expanded.columns:
        print(f"    - {col}")

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    # Check logg ranges per variant
    for variant_id, variant_name in [(0, 'Baseline'), (1, '+σ'), (2, '-σ')]:
        df_var = df_expanded.filter(pl.col('variant') == variant_id)
        logg_vals = df_var['logg_gaia'].to_numpy()
        print(f"\n{variant_name} (variant={variant_id}):")
        print(f"  logg range: [{logg_vals.min():.3f}, {logg_vals.max():.3f}] dex")
        print(f"  logg mean:  {logg_vals.mean():.3f} dex")
        print(f"  logg std:   {logg_vals.std():.3f} dex")

    print("\n✓ Done! Ready for prediction pipeline.")
    print("\nNext step:")
    print("  python pipeline.py --predict --pred-config config/prediction/predict_teff_logg_perturbed.yaml")

if __name__ == '__main__':
    main()
