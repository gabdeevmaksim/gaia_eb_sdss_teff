#!/usr/bin/env python3
"""
Create engineered feature dataset for Gaia + 2MASS colors.

This script applies feature engineering transformations to Gaia + 2MASS color data:
- Polynomial features (degree 2, 3)
- Interaction features (color1 * color2)
- Log features
- Temperature-dependent features (hot/cool/mid regimes)

Output: data/processed/gaia_2mass_colors_engineered_train.parquet

Usage:
    python scripts/create_gaia_2mass_engineered_features.py

Author: Claude Code
Date: 2025-10-16
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import get_config
from src.features.engineering import engineer_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Create engineered feature dataset."""
    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("=" * 80)
    logger.info("CREATE GAIA + 2MASS ENGINEERED FEATURES")
    logger.info("=" * 80)

    # Load base training data
    input_file = processed_dir / 'gaia_2mass_colors_training.parquet'
    logger.info(f"\nLoading base training data: {input_file.name}")
    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} objects")
    logger.info(f"  Columns: {list(df.columns)}")

    # Define color features
    color_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']

    logger.info(f"\nBase color features: {color_cols}")
    logger.info(f"  Target: teff_gspphot")

    # Apply feature engineering
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING FEATURE ENGINEERING")
    logger.info("=" * 80)

    logger.info("\nEngineering transformations:")
    logger.info("  1. Polynomial features (degree 3)")
    logger.info("  2. Interaction features (pairwise products)")
    logger.info("  3. Log features (with offset for negative values)")
    logger.info("  4. Temperature-dependent features (hot/cool/mid regimes)")

    df_engineered = engineer_all_features(
        df,
        color_cols=color_cols,
        mag_cols=None,  # No magnitude features (color-only model)
        include_polynomials=True,
        include_interactions=True,
        include_log=True,
        include_temp_dependent=True,
        include_mag_features=False
    )

    # Get list of new features
    original_cols = set(df.columns)
    engineered_cols = set(df_engineered.columns)
    new_features = sorted(engineered_cols - original_cols)

    logger.info(f"\nFeature engineering complete!")
    logger.info(f"  Original features: {len(original_cols)}")
    logger.info(f"  Total features: {len(engineered_cols)}")
    logger.info(f"  New features created: {len(new_features)}")

    # Display sample of new features
    logger.info(f"\nSample of new features:")
    for feat in new_features[:20]:
        logger.info(f"  - {feat}")
    if len(new_features) > 20:
        logger.info(f"  ... and {len(new_features) - 20} more")

    # Check for any NaN or infinite values
    logger.info("\nData quality check:")
    nan_counts = df_engineered.isna().sum()
    inf_counts = np.isinf(df_engineered.select_dtypes(include=[np.number])).sum()

    total_nans = nan_counts.sum()
    total_infs = inf_counts.sum()

    logger.info(f"  NaN values: {total_nans}")
    logger.info(f"  Inf values: {total_infs}")

    if total_nans > 0:
        logger.warning(f"\nColumns with NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  {col}: {count:,} NaN values")

    if total_infs > 0:
        logger.warning(f"\nColumns with Inf values:")
        for col, count in inf_counts[inf_counts > 0].items():
            logger.warning(f"  {col}: {count:,} Inf values")

    # Replace NaN and Inf with 0 for robust training
    if total_nans > 0 or total_infs > 0:
        logger.info("\nReplacing NaN and Inf values with 0...")
        df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
        df_engineered = df_engineered.fillna(0)
        logger.info("  Replacement complete")

    # Save engineered dataset
    output_file = processed_dir / 'gaia_2mass_colors_engineered_train.parquet'
    logger.info(f"\nSaving engineered features: {output_file.name}")
    df_engineered.to_parquet(output_file, index=False)
    logger.info(f"  Saved {len(df_engineered):,} objects")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Create feature list documentation
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE DOCUMENTATION")
    logger.info("=" * 80)

    # Group features by type
    base_features = color_cols + ['teff_gspphot', 'source_id']
    poly_features = [f for f in new_features if any(c in f for c in ['^2', '^3', '**2', '**3'])]
    interaction_features = [f for f in new_features if '_x_' in f]
    log_features = [f for f in new_features if f.startswith('log_')]
    temp_features = [f for f in new_features if any(t in f for t in ['hot_', 'cool_', 'mid_'])]
    other_features = [f for f in new_features if f not in poly_features + interaction_features + log_features + temp_features]

    logger.info(f"\nFeature breakdown:")
    logger.info(f"  Base features: {len(base_features)}")
    logger.info(f"  Polynomial features: {len(poly_features)}")
    logger.info(f"  Interaction features: {len(interaction_features)}")
    logger.info(f"  Log features: {len(log_features)}")
    logger.info(f"  Temperature-dependent features: {len(temp_features)}")
    logger.info(f"  Other features: {len(other_features)}")

    # Save feature list to docs
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    feature_doc_file = docs_dir / 'GAIA_2MASS_ENGINEERED_FEATURES.md'

    with open(feature_doc_file, 'w') as f:
        f.write("# Gaia + 2MASS Engineered Features\n\n")
        f.write(f"**Dataset**: `gaia_2mass_colors_engineered_train.parquet`\n\n")
        f.write(f"**Total objects**: {len(df_engineered):,}\n\n")
        f.write(f"**Total features**: {len(df_engineered.columns)}\n\n")

        f.write("## Feature Types\n\n")
        f.write(f"- Base color features: {len(color_cols)}\n")
        f.write(f"- Polynomial features: {len(poly_features)}\n")
        f.write(f"- Interaction features: {len(interaction_features)}\n")
        f.write(f"- Log features: {len(log_features)}\n")
        f.write(f"- Temperature-dependent features: {len(temp_features)}\n\n")

        f.write("## Base Features\n\n")
        f.write("```\n")
        for feat in color_cols:
            f.write(f"- {feat}\n")
        f.write("```\n\n")

        f.write("## Polynomial Features\n\n")
        f.write(f"Degree 2 and 3 polynomials of base colors ({len(poly_features)} features)\n\n")
        f.write("```\n")
        for feat in poly_features:
            f.write(f"- {feat}\n")
        f.write("```\n\n")

        f.write("## Interaction Features\n\n")
        f.write(f"Pairwise products of colors ({len(interaction_features)} features)\n\n")
        f.write("```\n")
        for feat in interaction_features:
            f.write(f"- {feat}\n")
        f.write("```\n\n")

        f.write("## Log Features\n\n")
        f.write(f"Log transformations with offset=0.5 ({len(log_features)} features)\n\n")
        f.write("```\n")
        for feat in log_features:
            f.write(f"- {feat}\n")
        f.write("```\n\n")

        f.write("## Temperature-Dependent Features\n\n")
        f.write(f"Hot/cool/mid regime features ({len(temp_features)} features)\n\n")
        f.write("```\n")
        for feat in temp_features:
            f.write(f"- {feat}\n")
        f.write("```\n\n")

        f.write("## Target Variable\n\n")
        f.write("- `teff_gspphot`: Gaia GSP-Phot effective temperature (K)\n\n")

        f.write("## Common Key\n\n")
        f.write("- `source_id`: Gaia DR3 source identifier\n\n")

        f.write("## Statistics\n\n")
        f.write(f"```\n")
        f.write(f"Temperature range: {df_engineered['teff_gspphot'].min():.0f} - {df_engineered['teff_gspphot'].max():.0f} K\n")
        f.write(f"Temperature mean:  {df_engineered['teff_gspphot'].mean():.0f} K\n")
        f.write(f"Temperature std:   {df_engineered['teff_gspphot'].std():.0f} K\n")
        f.write(f"```\n")

    logger.info(f"\nFeature documentation saved: {feature_doc_file.name}")

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Documentation: {feature_doc_file}")
    logger.info("\nNext step:")
    logger.info("  python scripts/train_gaia_2mass_engineered_model.py")


if __name__ == '__main__':
    main()
