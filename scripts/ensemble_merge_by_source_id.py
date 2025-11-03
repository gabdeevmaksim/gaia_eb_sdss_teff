#!/usr/bin/env python3
"""
Create ensemble by merging predictions on source_id.

This script merges predictions from two unified models using gaia_source_id
and calculates ensemble predictions as the average.

Models:
1. Unified WITH g magnitude (rf_unified_engineered_20251015_180614)
2. Unified WITHOUT g magnitude (rf_unified_engineered_20251016_112332)

Usage:
    python scripts/ensemble_merge_by_source_id.py

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Create ensemble by merging predictions on source_id."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("=" * 80)
    logger.info("ENSEMBLE BY MERGING ON SOURCE_ID")
    logger.info("=" * 80)

    MODEL1_ID = 'rf_unified_engineered_20251015_180614'
    MODEL2_ID = 'rf_unified_engineered_20251016_112332'

    MODEL1_NAME = 'Unified WITH g magnitude'
    MODEL2_NAME = 'Unified WITHOUT g magnitude'

    logger.info(f"\nModel 1: {MODEL1_NAME} ({MODEL1_ID})")
    logger.info(f"Model 2: {MODEL2_NAME} ({MODEL2_ID})")

    # Load prediction files (these contain training+test data with true temperatures)
    logger.info("\n" + "=" * 80)
    logger.info("LOADING PREDICTIONS")
    logger.info("=" * 80)

    logger.info(f"\nLoading Model 1 predictions...")
    pred1_file = processed_dir / f'predictions_{MODEL1_ID}.parquet'
    df1 = pd.read_parquet(pred1_file)
    logger.info(f"  Shape: {df1.shape}")
    logger.info(f"  Columns: {df1.columns.tolist()[:15]}...")

    logger.info(f"\nLoading Model 2 predictions...")
    pred2_file = processed_dir / f'predictions_{MODEL2_ID}.parquet'
    df2 = pd.read_parquet(pred2_file)
    logger.info(f"  Shape: {df2.shape}")
    logger.info(f"  Columns: {df2.columns.tolist()[:15]}...")

    # Check if these files have the predicted temperatures column
    # The column name might be different - let's find it
    pred_col1 = [c for c in df1.columns if 'predict' in c.lower() or 'teff' in c.lower()]
    pred_col2 = [c for c in df2.columns if 'predict' in c.lower() or 'teff' in c.lower()]

    logger.info(f"\nPrediction columns in Model 1: {pred_col1}")
    logger.info(f"Prediction columns in Model 2: {pred_col2}")

    # These files contain the features, not predictions!
    # We need to look for files with actual predictions
    # Let's check what columns we have
    if 'teff_predicted' not in df1.columns:
        logger.error("Model 1 file doesn't have predicted temperatures!")
        logger.info("This file contains features, not predictions.")
        logger.info("We need the actual prediction output files.")

        # Let's check if there's a different naming pattern
        logger.info("\nSearching for prediction files...")
        pred_files = list(processed_dir.glob('*unified*20251015*.parquet')) + \
                    list(processed_dir.glob('*unified*20251016*.parquet'))

        for f in pred_files:
            if 'SUMMARY' not in str(f):
                logger.info(f"  Found: {f.name}")

        return

    # Merge on gaia_source_id
    logger.info("\n" + "=" * 80)
    logger.info("MERGING ON GAIA_SOURCE_ID")
    logger.info("=" * 80)

    # Merge the two datasets
    df_merged = df1[['gaia_source_id', 'teff_gspphot', 'teff_predicted']].merge(
        df2[['gaia_source_id', 'teff_predicted']],
        on='gaia_source_id',
        how='inner',
        suffixes=('_with_gmag', '_without_gmag')
    )

    logger.info(f"  Merged on {len(df_merged):,} common objects")

    # Calculate ensemble
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING ENSEMBLE")
    logger.info("=" * 80)

    df_merged['teff_ensemble'] = (df_merged['teff_predicted_with_gmag'] +
                                   df_merged['teff_predicted_without_gmag']) / 2

    # Remove rows with NaN values
    logger.info("\nFiltering for valid predictions...")
    before_filter = len(df_merged)
    df_merged = df_merged.dropna(subset=['teff_gspphot', 'teff_predicted_with_gmag',
                                          'teff_predicted_without_gmag'])
    after_filter = len(df_merged)
    logger.info(f"  Removed {before_filter - after_filter:,} rows with NaN")
    logger.info(f"  Final dataset: {after_filter:,} objects")

    # Calculate metrics
    logger.info("\nCalculating metrics...")

    true_temp = df_merged['teff_gspphot']

    # Model 1
    mae1 = mean_absolute_error(true_temp, df_merged['teff_predicted_with_gmag'])
    rmse1 = np.sqrt(mean_squared_error(true_temp, df_merged['teff_predicted_with_gmag']))
    r2_1 = r2_score(true_temp, df_merged['teff_predicted_with_gmag'])

    # Model 2
    mae2 = mean_absolute_error(true_temp, df_merged['teff_predicted_without_gmag'])
    rmse2 = np.sqrt(mean_squared_error(true_temp, df_merged['teff_predicted_without_gmag']))
    r2_2 = r2_score(true_temp, df_merged['teff_predicted_without_gmag'])

    # Ensemble
    mae_ens = mean_absolute_error(true_temp, df_merged['teff_ensemble'])
    rmse_ens = np.sqrt(mean_squared_error(true_temp, df_merged['teff_ensemble']))
    r2_ens = r2_score(true_temp, df_merged['teff_ensemble'])

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    logger.info(f"\n{MODEL1_NAME}:")
    logger.info(f"  MAE:  {mae1:.1f} K")
    logger.info(f"  RMSE: {rmse1:.1f} K")
    logger.info(f"  R²:   {r2_1:.3f}")

    logger.info(f"\n{MODEL2_NAME}:")
    logger.info(f"  MAE:  {mae2:.1f} K")
    logger.info(f"  RMSE: {rmse2:.1f} K")
    logger.info(f"  R²:   {r2_2:.3f}")

    logger.info(f"\nENSEMBLE (Average):")
    logger.info(f"  MAE:  {mae_ens:.1f} K")
    logger.info(f"  RMSE: {rmse_ens:.1f} K")
    logger.info(f"  R²:   {r2_ens:.3f}")

    best_mae = min(mae1, mae2)
    improvement = (best_mae - mae_ens) / best_mae * 100

    logger.info(f"\nIMPROVEMENT:")
    if mae_ens < best_mae:
        logger.info(f"  ✓ Ensemble BETTER: {improvement:.1f}% MAE reduction")
    else:
        logger.info(f"  ✗ Ensemble WORSE: {improvement:+.1f}% MAE change")

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_file = models_dir / 'ensemble_unified_with_without_gmag_test_predictions.parquet'
    df_merged.to_parquet(output_file, index=False)
    logger.info(f"  Saved: {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
