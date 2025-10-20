#!/usr/bin/env python3
"""
Create training dataset with Gaia colors and 2MASS infrared colors.

This script merges:
- Gaia catalog (for source_id and bp_rp color)
- 2MASS photometry (for J-H, H-K, J-K colors)
- Gaia GSP-Phot temperatures (target variable)

Output: Clean dataset with only color features (no magnitudes)
Common key: Gaia source_id
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Create clean training dataset with Gaia + 2MASS colors."""

    config = get_config()
    processed_dir = config.get_path('processed')

    logger.info("="*80)
    logger.info("CREATE GAIA + 2MASS COLOR TRAINING DATASET")
    logger.info("="*80)

    # Load base Gaia catalog
    logger.info("\n1. Loading Gaia catalog...")
    df_catalog = pd.read_parquet(processed_dir / 'eb_catalog.parquet')
    logger.info(f"   Loaded: {len(df_catalog):,} objects")
    logger.info(f"   Columns: source_id, original_ext_source_id, bp_rp, teff_gspphot, etc.")

    # Filter objects with Gaia Teff
    logger.info("\n2. Filtering objects with Gaia GSP-Phot temperatures...")
    df_with_teff = df_catalog[df_catalog['teff_gspphot'].notna()].copy()
    logger.info(f"   Objects with Teff: {len(df_with_teff):,}")

    # Load 2MASS photometry
    logger.info("\n3. Loading 2MASS photometry...")
    df_2mass = pd.read_parquet(processed_dir / 'eb_2mass_photometry.parquet')
    logger.info(f"   Loaded: {len(df_2mass):,} objects")

    # Calculate 2MASS colors
    logger.info("\n4. Calculating 2MASS colors...")
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']
    logger.info("   Calculated: j_h_color, h_k_color, j_k_color")

    # Merge on Gaia source_id
    logger.info("\n5. Merging catalogs on Gaia source_id...")
    df_merged = df_with_teff.merge(
        df_2mass[['source_id', 'j_h_color', 'h_k_color', 'j_k_color',
                  'j_snr', 'h_snr', 'k_snr', 'ph_qual']],
        on='source_id',
        how='inner'
    )
    logger.info(f"   Merged dataset: {len(df_merged):,} objects")
    logger.info(f"   (Objects with both Gaia Teff and 2MASS photometry)")

    # Select color features only
    logger.info("\n6. Selecting color features (no magnitudes)...")
    feature_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
    target_col = 'teff_gspphot'
    id_col = 'source_id'

    df_features = df_merged[[id_col] + feature_cols + [target_col]].copy()
    logger.info(f"   Features: {feature_cols}")
    logger.info(f"   Target: {target_col}")

    # Clean data - remove missing values
    logger.info("\n7. Cleaning data...")
    logger.info(f"   Initial dataset: {len(df_features):,} objects")

    # Check for missing values
    for col in feature_cols + [target_col]:
        n_missing = df_features[col].isna().sum()
        if n_missing > 0:
            logger.info(f"   Missing in {col}: {n_missing:,} ({100*n_missing/len(df_features):.2f}%)")

    # Remove rows with any missing values
    df_clean = df_features.dropna(subset=feature_cols + [target_col])
    logger.info(f"   After removing missing: {len(df_clean):,} objects")
    logger.info(f"   Removed: {len(df_features) - len(df_clean):,} objects")

    # Filter unrealistic color values (physical bounds)
    logger.info("\n8. Filtering unrealistic color values...")
    color_ranges = {
        'bp_rp': (-0.5, 4.0),      # Gaia BP-RP
        'j_h_color': (-0.5, 2.0),   # 2MASS J-H
        'h_k_color': (-0.3, 1.0),   # 2MASS H-K
        'j_k_color': (-0.5, 2.5)    # 2MASS J-K
    }

    n_before = len(df_clean)
    for col, (min_val, max_val) in color_ranges.items():
        mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
        n_outliers = (~mask).sum()
        if n_outliers > 0:
            logger.info(f"   {col}: removing {n_outliers:,} outliers outside [{min_val}, {max_val}]")
        df_clean = df_clean[mask]

    logger.info(f"   After filtering: {len(df_clean):,} objects")
    logger.info(f"   Removed: {n_before - len(df_clean):,} outliers")

    # Filter unrealistic temperatures
    logger.info("\n9. Filtering unrealistic temperatures...")
    temp_min, temp_max = 2500, 50000
    mask_temp = (df_clean[target_col] >= temp_min) & (df_clean[target_col] <= temp_max)
    n_outliers = (~mask_temp).sum()
    if n_outliers > 0:
        logger.info(f"   Removing {n_outliers:,} objects with Teff outside [{temp_min}, {temp_max}] K")
    df_clean = df_clean[mask_temp]
    logger.info(f"   Final dataset: {len(df_clean):,} objects")

    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("DATASET SUMMARY")
    logger.info("="*80)
    logger.info(f"Total objects: {len(df_clean):,}")
    logger.info(f"\nFeature statistics:")
    for col in feature_cols:
        logger.info(f"  {col:12s}: {df_clean[col].min():.3f} to {df_clean[col].max():.3f} "
                   f"(mean={df_clean[col].mean():.3f}, std={df_clean[col].std():.3f})")

    logger.info(f"\nTarget statistics:")
    logger.info(f"  {target_col}: {df_clean[target_col].min():.1f} to {df_clean[target_col].max():.1f} K")
    logger.info(f"  Mean: {df_clean[target_col].mean():.1f} K")
    logger.info(f"  Median: {df_clean[target_col].median():.1f} K")
    logger.info(f"  Std: {df_clean[target_col].std():.1f} K")

    # Save dataset
    output_file = processed_dir / 'gaia_2mass_colors_training.parquet'
    logger.info(f"\n" + "="*80)
    logger.info(f"SAVING DATASET")
    logger.info("="*80)
    logger.info(f"Output file: {output_file}")
    df_clean.to_parquet(output_file, index=False)
    logger.info(f"Saved: {len(df_clean):,} objects, {len(df_clean.columns)} columns")

    # Create schema documentation
    schema_doc = f"""# Gaia + 2MASS Colors Training Dataset Schema

**File:** `data/processed/gaia_2mass_colors_training.parquet`
**Created:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Objects:** {len(df_clean):,}

## Description
Clean training dataset containing only Gaia and 2MASS color indices (no magnitude features).
Includes objects with Gaia GSP-Phot effective temperatures for supervised learning.

## Common Key
- **source_id** (int64): Gaia DR3 source identifier - unique key for joining with other tables

## Columns

### Identifier
- **source_id** (int64): Gaia DR3 source identifier

### Feature Columns (4 colors)
- **bp_rp** (float64): Gaia BP-RP color [mag]
  - Range: {df_clean['bp_rp'].min():.3f} to {df_clean['bp_rp'].max():.3f}
  - Mean: {df_clean['bp_rp'].mean():.3f} ± {df_clean['bp_rp'].std():.3f}

- **j_h_color** (float64): 2MASS J-H color [mag]
  - Range: {df_clean['j_h_color'].min():.3f} to {df_clean['j_h_color'].max():.3f}
  - Mean: {df_clean['j_h_color'].mean():.3f} ± {df_clean['j_h_color'].std():.3f}

- **h_k_color** (float64): 2MASS H-K color [mag]
  - Range: {df_clean['h_k_color'].min():.3f} to {df_clean['h_k_color'].max():.3f}
  - Mean: {df_clean['h_k_color'].mean():.3f} ± {df_clean['h_k_color'].std():.3f}

- **j_k_color** (float64): 2MASS J-K color [mag]
  - Range: {df_clean['j_k_color'].min():.3f} to {df_clean['j_k_color'].max():.3f}
  - Mean: {df_clean['j_k_color'].mean():.3f} ± {df_clean['j_k_color'].std():.3f}

### Target Column
- **teff_gspphot** (float64): Gaia GSP-Phot effective temperature [K]
  - Range: {df_clean['teff_gspphot'].min():.1f} to {df_clean['teff_gspphot'].max():.1f} K
  - Mean: {df_clean['teff_gspphot'].mean():.1f} ± {df_clean['teff_gspphot'].std():.1f} K
  - Median: {df_clean['teff_gspphot'].median():.1f} K

## Data Quality
- No missing values in any column
- Color values filtered to physically realistic ranges:
  - bp_rp: -0.5 to 4.0 mag
  - j_h_color: -0.5 to 2.0 mag
  - h_k_color: -0.3 to 1.0 mag
  - j_k_color: -0.5 to 2.5 mag
- Temperatures filtered to 2,500 - 50,000 K

## Source Tables
1. **eb_catalog.parquet**: Gaia source_id, bp_rp color, teff_gspphot
2. **eb_2mass_photometry.parquet**: 2MASS infrared magnitudes and colors

## Usage
```python
import pandas as pd
df = pd.read_parquet('data/processed/gaia_2mass_colors_training.parquet')

# Features for ML
features = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
X = df[features]
y = df['teff_gspphot']

# Gaia source_id for joins
source_ids = df['source_id']
```

## Notes
- All objects have both Gaia photometry and 2MASS infrared photometry
- Dataset is clean and ready for machine learning
- No magnitude features included (following best practice to avoid distance bias)
- Common key (source_id) enables joining with other Gaia-based tables
"""

    schema_file = Path('docs') / 'GAIA_2MASS_COLORS_SCHEMA.md'
    schema_file.parent.mkdir(exist_ok=True)
    with open(schema_file, 'w') as f:
        f.write(schema_doc)
    logger.info(f"Schema documentation: {schema_file}")

    logger.info("\n" + "="*80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nNext step: Train model with 4 color features")
    logger.info(f"  python scripts/train_gaia_2mass_model.py")

if __name__ == '__main__':
    main()
