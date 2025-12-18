"""
Create best-of-four ensemble from three corrected models + flag 1 model.

Selects the prediction with the lowest RF uncertainty for each object.
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config

def main():
    config = get_config()

    print("="*70)
    print("BEST-OF-FOUR ENSEMBLE CREATION")
    print("="*70)
    print()

    # Load existing best-of-three predictions (3 models: teff_only, teff_logg, teff_cluster)
    print("Loading existing best-of-three predictions...")
    best3_file = Path(config.get_path('processed')) / 'teff_predictions_best_of_three_corrected.parquet'
    best3 = pl.read_parquet(best3_file)
    print(f"  Loaded {len(best3):,} predictions from best-of-three")
    print(f"  Columns: {best3.columns}")
    print()

    # Load new flag 1 model predictions
    print("Loading flag 1 model predictions...")
    flag1_file = Path(config.get_path('processed')) / 'predictions_rf_gaia_colors_flag1_20251218_094937.parquet'
    flag1 = pl.read_parquet(flag1_file)
    print(f"  Loaded {len(flag1):,} predictions from flag 1 model")
    print(f"  Columns: {flag1.columns}")
    print()

    # Rename flag1 columns to match naming convention
    flag1 = flag1.rename({
        'teff_gaia_corrected_predicted': 'teff_flag1',
        'teff_gaia_corrected_uncertainty': 'unc_flag1'
    })

    # Merge flag1 predictions with best-of-three
    print("Merging flag 1 predictions with best-of-three...")
    merged = best3.join(flag1, on='source_id', how='left')
    print(f"  Merged dataset: {len(merged):,} objects")
    print()

    # Check how many objects have flag1 predictions
    n_with_flag1 = merged['teff_flag1'].is_not_null().sum()
    print(f"Objects with flag 1 predictions: {n_with_flag1:,} ({n_with_flag1/len(merged)*100:.1f}%)")
    print()

    # For each object, find the model with lowest uncertainty
    print("Selecting best prediction for each object (lowest uncertainty)...")

    # Create arrays of all uncertainties
    unc_only = merged['unc_only'].to_numpy()
    unc_logg = merged['unc_logg'].to_numpy()
    unc_cluster = merged['unc_cluster'].to_numpy()
    unc_flag1 = merged['unc_flag1'].to_numpy()

    # Stack uncertainties (shape: n_objects x 4 models)
    unc_stack = np.stack([unc_only, unc_logg, unc_cluster, unc_flag1], axis=1)

    # Find index of minimum uncertainty for each object (ignoring NaNs)
    best_idx = np.nanargmin(unc_stack, axis=1)

    # Create arrays of all predictions
    teff_only = merged['teff_only'].to_numpy()
    teff_logg = merged['teff_logg'].to_numpy()
    teff_cluster = merged['teff_cluster'].to_numpy()
    teff_flag1 = merged['teff_flag1'].to_numpy()

    teff_stack = np.stack([teff_only, teff_logg, teff_cluster, teff_flag1], axis=1)

    # Select best prediction and uncertainty for each object
    n_objects = len(merged)
    teff_best = np.array([teff_stack[i, best_idx[i]] for i in range(n_objects)])
    unc_best = np.array([unc_stack[i, best_idx[i]] for i in range(n_objects)])

    # Map index to model name
    model_names = ['teff_only', 'teff_logg', 'teff_cluster', 'teff_flag1']
    best_model = np.array([model_names[idx] for idx in best_idx])

    # Add best predictions to dataframe
    result = merged.with_columns([
        pl.Series('teff_best_4', teff_best),
        pl.Series('unc_best_4', unc_best),
        pl.Series('best_model_4', best_model)
    ])

    # Report results
    print()
    print("="*70)
    print("BEST-OF-FOUR RESULTS")
    print("="*70)
    print()

    # Model selection distribution
    print("Model selection distribution:")
    model_counts = result['best_model_4'].value_counts().sort('best_model_4')
    for row in model_counts.iter_rows():
        model, count = row
        pct = count / len(result) * 100
        print(f"  {model:15s}: {count:7,} ({pct:5.1f}%)")
    print()

    # Uncertainty statistics
    print("Uncertainty statistics (best-of-four):")
    print(f"  Mean:   {result['unc_best_4'].mean():.1f} K")
    print(f"  Median: {result['unc_best_4'].median():.1f} K")
    print(f"  Min:    {result['unc_best_4'].min():.1f} K")
    print(f"  Max:    {result['unc_best_4'].max():.1f} K")
    print()

    # Compare with best-of-three
    print("Comparison with best-of-three:")
    print(f"  Best-of-three mean uncertainty: {result['unc_best'].mean():.1f} K")
    print(f"  Best-of-four mean uncertainty:  {result['unc_best_4'].mean():.1f} K")
    improvement = (result['unc_best'].mean() - result['unc_best_4'].mean()) / result['unc_best'].mean() * 100
    print(f"  Improvement: {improvement:.1f}%")
    print()

    # How many objects changed their best model?
    n_changed = (result['best_model'] != result['best_model_4']).sum()
    print(f"Objects with different best model: {n_changed:,} ({n_changed/len(result)*100:.1f}%)")
    print()

    # Save best-of-four ensemble
    output_file = Path(config.get_path('processed')) / 'teff_predictions_best_of_four.parquet'
    result.write_parquet(output_file)
    print(f"✓ Saved best-of-four ensemble: {output_file}")
    print(f"  Total objects: {len(result):,}")
    print(f"  Mean uncertainty: {result['unc_best_4'].mean():.1f} K")
    print()

    # Also create a simplified version with just the best predictions
    simple = result.select([
        'source_id',
        'teff_best_4',
        'unc_best_4',
        'best_model_4'
    ])
    simple_file = Path(config.get_path('processed')) / 'teff_predictions_best_of_four_simple.parquet'
    simple.write_parquet(simple_file)
    print(f"✓ Saved simplified best-of-four: {simple_file}")
    print()

if __name__ == '__main__':
    main()
