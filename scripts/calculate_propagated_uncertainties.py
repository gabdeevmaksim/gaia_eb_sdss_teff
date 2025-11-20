"""
Calculate propagated uncertainties from logg-perturbed predictions.

Uses numerical gradient ∂Teff/∂logg to propagate logg uncertainty into Teff uncertainty,
then combines with RF tree-based uncertainty.
"""

import polars as pl
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("CALCULATING PROPAGATED UNCERTAINTIES")
    print("=" * 80)

    # Load perturbed predictions
    print("\n[1/4] Loading perturbed predictions...")
    pred_path = Path('data/processed/teff_predictions_logg_perturbed.parquet')
    df = pl.read_parquet(pred_path)

    print(f"  Loaded {len(df):,} predictions (3x per object)")
    print(f"  Unique objects: {df['original_index'].n_unique():,}")

    # Check that we have all 3 variants
    variant_counts = df.group_by('variant').agg(pl.count()).sort('variant')
    print(f"\n  Variant counts:")
    for row in variant_counts.iter_rows():
        variant_id, count = row
        variant_name = {0: 'Baseline', 1: '+σ', 2: '-σ'}[variant_id]
        print(f"    {variant_name} (variant={variant_id}): {count:,}")

    # Convert predictions from log10 to Kelvin
    print("\n[2/4] Converting log(Teff) predictions to Kelvin...")
    df = df.with_columns([
        (10 ** pl.col('teff_predicted')).alias('teff_kelvin')
    ])

    # Also convert uncertainties (these are in log space from RF trees)
    # For comparison: σ_kelvin ≈ Teff * σ_log * ln(10)
    df = df.with_columns([
        (pl.col('teff_kelvin') * pl.col('teff_uncertainty') * np.log(10)).alias('teff_uncertainty_rf_kelvin')
    ])

    print(f"  Mean Teff (baseline): {df.filter(pl.col('variant') == 0)['teff_kelvin'].mean():.1f} K")

    # Pivot to get baseline, plus, minus predictions per object
    print("\n[3/4] Computing numerical gradients...")

    # Group by original_index and calculate gradient
    df_baseline = df.filter(pl.col('variant') == 0).select([
        'original_index', 'source_id', 'ra', 'dec',
        'g', 'bp', 'rp', 'bp_rp',
        'logg_gaia', 'logg_uncertainty', 'logg_predicted_original',
        'teff_source', 'logg_source',
        pl.col('teff_kelvin').alias('teff_baseline'),
        pl.col('teff_uncertainty_rf_kelvin').alias('teff_unc_rf')
    ])

    df_plus = df.filter(pl.col('variant') == 1).select([
        'original_index',
        pl.col('teff_kelvin').alias('teff_plus'),
        pl.col('teff_uncertainty_rf_kelvin').alias('teff_unc_rf_plus')
    ])

    df_minus = df.filter(pl.col('variant') == 2).select([
        'original_index',
        pl.col('teff_kelvin').alias('teff_minus'),
        pl.col('teff_uncertainty_rf_kelvin').alias('teff_unc_rf_minus')
    ])

    # Join all variants
    df_combined = (df_baseline
                   .join(df_plus, on='original_index', how='left')
                   .join(df_minus, on='original_index', how='left'))

    print(f"  Combined dataset: {len(df_combined):,} objects")

    # Calculate numerical gradient: ∂Teff/∂logg
    df_combined = df_combined.with_columns([
        ((pl.col('teff_plus') - pl.col('teff_minus')) /
         (2 * pl.col('logg_uncertainty'))).alias('gradient_teff_logg')
    ])

    # Calculate logg contribution to uncertainty
    df_combined = df_combined.with_columns([
        (pl.col('gradient_teff_logg').abs() * pl.col('logg_uncertainty')).alias('teff_unc_logg')
    ])

    # Combine RF and logg uncertainties in quadrature
    df_combined = df_combined.with_columns([
        (pl.col('teff_unc_rf')**2 + pl.col('teff_unc_logg')**2).sqrt().alias('teff_unc_total')
    ])

    # Rename columns for output
    df_output = df_combined.select([
        'source_id', 'ra', 'dec',
        'g', 'bp', 'rp', 'bp_rp',
        pl.col('logg_predicted_original').alias('logg_predicted'),
        'logg_uncertainty',
        'teff_source', 'logg_source',
        pl.col('teff_baseline').alias('teff_predicted'),
        'teff_unc_rf',
        'teff_unc_logg',
        'teff_unc_total',
        'gradient_teff_logg'
    ])

    # Statistics
    print("\n" + "=" * 80)
    print("UNCERTAINTY STATISTICS")
    print("=" * 80)

    print("\nRF Tree Uncertainty (σ_RF):")
    unc_rf = df_output['teff_unc_rf'].to_numpy()
    print(f"  Mean:   {unc_rf.mean():.1f} K")
    print(f"  Median: {np.median(unc_rf):.1f} K")
    print(f"  Std:    {unc_rf.std():.1f} K")
    print(f"  Range:  [{unc_rf.min():.1f}, {unc_rf.max():.1f}] K")

    print("\nlogg Propagated Uncertainty (σ_logg):")
    unc_logg = df_output['teff_unc_logg'].to_numpy()
    print(f"  Mean:   {unc_logg.mean():.1f} K")
    print(f"  Median: {np.median(unc_logg):.1f} K")
    print(f"  Std:    {unc_logg.std():.1f} K")
    print(f"  Range:  [{unc_logg.min():.1f}, {unc_logg.max():.1f}] K")

    print("\nTotal Combined Uncertainty (σ_total):")
    unc_total = df_output['teff_unc_total'].to_numpy()
    print(f"  Mean:   {unc_total.mean():.1f} K")
    print(f"  Median: {np.median(unc_total):.1f} K")
    print(f"  Std:    {unc_total.std():.1f} K")
    print(f"  Range:  [{unc_total.min():.1f}, {unc_total.max():.1f}] K")

    print("\nGradient ∂Teff/∂logg:")
    gradient = df_output['gradient_teff_logg'].to_numpy()
    print(f"  Mean:   {gradient.mean():.1f} K/dex")
    print(f"  Median: {np.median(gradient):.1f} K/dex")
    print(f"  Std:    {gradient.std():.1f} K/dex")
    print(f"  Range:  [{gradient.min():.1f}, {gradient.max():.1f}] K/dex")

    print("\nRelative Contributions:")
    increase = unc_total - unc_rf
    increase_pct = (increase / unc_rf) * 100
    print(f"  Mean increase:   {increase.mean():.1f} K ({increase_pct.mean():.1f}%)")
    print(f"  Median increase: {np.median(increase):.1f} K ({np.median(increase_pct):.1f}%)")

    # Ratio of uncertainties
    ratio = unc_logg / unc_rf
    print(f"\n  σ_logg / σ_RF ratio:")
    print(f"    Mean:   {ratio.mean():.3f}")
    print(f"    Median: {np.median(ratio):.3f}")

    # Save results
    print("\n[4/4] Saving results...")
    output_path = Path('data/processed/teff_predictions_with_logg_propagated_final.parquet')
    df_output.write_parquet(output_path)

    print(f"✓ Saved: {output_path}")
    print(f"  Objects: {len(df_output):,}")
    print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")

    print("\n  Columns:")
    for col in df_output.columns:
        print(f"    - {col}")

    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # Check for unreasonable uncertainties
    unreasonable_total = (unc_total > 5000).sum()
    unreasonable_logg = (unc_logg > 3000).sum()

    print(f"\nObjects with σ_total > 5000 K: {unreasonable_total:,} ({unreasonable_total/len(df_output)*100:.2f}%)")
    print(f"Objects with σ_logg > 3000 K: {unreasonable_logg:,} ({unreasonable_logg/len(df_output)*100:.2f}%)")

    # Check correlation between logg_unc and teff_unc_logg
    from scipy.stats import pearsonr
    logg_unc = df_output['logg_uncertainty'].to_numpy()
    corr, pval = pearsonr(logg_unc, unc_logg)
    print(f"\nCorrelation between logg_unc and σ_logg:")
    print(f"  Pearson r: {corr:.3f} (p < {pval:.2e})")

    print("\n✓ Done!")
    print("\nOutput file contains:")
    print("  - teff_predicted: Predicted temperature (K)")
    print("  - teff_unc_rf: RF tree-based uncertainty (K)")
    print("  - teff_unc_logg: logg-propagated uncertainty (K)")
    print("  - teff_unc_total: Combined uncertainty (K)")
    print("  - gradient_teff_logg: Numerical gradient ∂Teff/∂logg (K/dex)")

if __name__ == '__main__':
    main()
