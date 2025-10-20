#!/usr/bin/env python3
"""
Predict temperatures for objects without Gaia GSP-Phot Teff using Gaia + 2MASS colors model.

Uses trained Random Forest model with:
- Gaia BP-RP color
- 2MASS J-H, H-K, J-K colors

Predictions are made only for objects with all 4 color features available.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

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
    """Predict temperatures for objects without Gaia Teff."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("PREDICT TEMPERATURES: GAIA + 2MASS COLORS MODEL")
    logger.info("="*80)

    # Model ID
    MODEL_ID = 'rf_gaia_2mass_colors_20251016_155128'
    logger.info(f"\nModel ID: {MODEL_ID}")

    # Load model
    logger.info("\n1. Loading trained model...")
    model_file = models_dir / f'{MODEL_ID}.pkl'
    model = joblib.load(model_file)
    logger.info(f"   Model loaded: {model_file.name}")

    # Load base catalog
    logger.info("\n2. Loading Gaia catalog...")
    df_catalog = pd.read_parquet(processed_dir / 'eb_catalog.parquet')
    logger.info(f"   Total objects: {len(df_catalog):,}")

    # Filter objects WITHOUT Gaia Teff
    logger.info("\n3. Filtering objects without Gaia GSP-Phot temperatures...")
    df_no_teff = df_catalog[df_catalog['teff_gspphot'].isna()].copy()
    logger.info(f"   Objects without Teff: {len(df_no_teff):,}")
    logger.info(f"   (These are candidates for temperature prediction)")

    # Load 2MASS photometry
    logger.info("\n4. Loading 2MASS photometry...")
    df_2mass = pd.read_parquet(processed_dir / 'eb_2mass_photometry.parquet')
    logger.info(f"   2MASS objects: {len(df_2mass):,}")

    # Calculate 2MASS colors
    logger.info("\n5. Calculating 2MASS colors...")
    df_2mass['j_h_color'] = df_2mass['j_m'] - df_2mass['h_m']
    df_2mass['h_k_color'] = df_2mass['h_m'] - df_2mass['k_m']
    df_2mass['j_k_color'] = df_2mass['j_m'] - df_2mass['k_m']

    # Merge on Gaia source_id
    logger.info("\n6. Merging catalogs on Gaia source_id...")
    df_predict = df_no_teff.merge(
        df_2mass[['source_id', 'j_h_color', 'h_k_color', 'j_k_color',
                  'j_snr', 'h_snr', 'k_snr', 'ph_qual']],
        on='source_id',
        how='inner'
    )
    logger.info(f"   Objects with both Gaia and 2MASS: {len(df_predict):,}")

    # Define features
    feature_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']

    # Check for missing values
    logger.info("\n7. Checking data quality...")
    for col in feature_cols:
        n_missing = df_predict[col].isna().sum()
        n_valid = df_predict[col].notna().sum()
        logger.info(f"   {col:12s}: {n_valid:,} valid, {n_missing:,} missing")

    # Remove objects with missing features
    df_clean = df_predict.dropna(subset=feature_cols).copy()
    logger.info(f"\n   Objects with all 4 colors: {len(df_clean):,}")
    logger.info(f"   Removed (missing colors): {len(df_predict) - len(df_clean):,}")

    # Filter unrealistic color values (same ranges as training)
    logger.info("\n8. Filtering unrealistic color values...")
    color_ranges = {
        'bp_rp': (-0.5, 4.0),
        'j_h_color': (-0.5, 2.0),
        'h_k_color': (-0.3, 1.0),
        'j_k_color': (-0.5, 2.5)
    }

    n_before = len(df_clean)
    for col, (min_val, max_val) in color_ranges.items():
        mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
        n_outliers = (~mask).sum()
        if n_outliers > 0:
            logger.info(f"   {col:12s}: removing {n_outliers:,} outliers outside [{min_val}, {max_val}]")
        df_clean = df_clean[mask]

    logger.info(f"\n   Final dataset for prediction: {len(df_clean):,} objects")
    logger.info(f"   Removed outliers: {n_before - len(df_clean):,}")

    # Make predictions
    logger.info("\n9. Making temperature predictions...")
    X = df_clean[feature_cols].values
    logger.info(f"   Feature matrix shape: {X.shape}")

    predictions = model.predict(X)
    logger.info(f"   Predictions generated: {len(predictions):,}")

    # Add predictions to dataframe
    df_clean['teff_predicted'] = predictions

    # Statistics
    logger.info("\n" + "="*80)
    logger.info("PREDICTION STATISTICS")
    logger.info("="*80)
    logger.info(f"\nPredicted temperatures:")
    logger.info(f"  Minimum: {predictions.min():.1f} K")
    logger.info(f"  Maximum: {predictions.max():.1f} K")
    logger.info(f"  Mean: {predictions.mean():.1f} K")
    logger.info(f"  Median: {np.median(predictions):.1f} K")
    logger.info(f"  Std: {predictions.std():.1f} K")

    # Temperature distribution
    temp_bins = [0, 4000, 5000, 6000, 7000, 10000, 50000]
    temp_labels = ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']
    df_clean['temp_bin'] = pd.cut(df_clean['teff_predicted'], bins=temp_bins, labels=temp_labels)

    logger.info("\nTemperature distribution:")
    for label in temp_labels:
        count = (df_clean['temp_bin'] == label).sum()
        pct = 100 * count / len(df_clean)
        logger.info(f"  {label:15s}: {count:6,} ({pct:5.1f}%)")

    # Color statistics
    logger.info("\n" + "="*80)
    logger.info("COLOR STATISTICS (PREDICTED SAMPLE)")
    logger.info("="*80)
    for col in feature_cols:
        logger.info(f"\n{col}:")
        logger.info(f"  Range: {df_clean[col].min():.3f} to {df_clean[col].max():.3f}")
        logger.info(f"  Mean: {df_clean[col].mean():.3f} ± {df_clean[col].std():.3f}")
        logger.info(f"  Median: {df_clean[col].median():.3f}")

    # 2MASS quality
    logger.info("\n" + "="*80)
    logger.info("2MASS DATA QUALITY")
    logger.info("="*80)
    quality_counts = df_clean['ph_qual'].value_counts().head(10)
    logger.info("\nTop 10 photometric quality flags:")
    for qual, count in quality_counts.items():
        pct = 100 * count / len(df_clean)
        logger.info(f"  {qual:3s}: {count:6,} ({pct:5.1f}%)")

    # High quality subset
    high_quality = df_clean[df_clean['ph_qual'].isin(['AAA', 'AAB', 'ABA', 'ABB'])]
    logger.info(f"\nHigh quality (AAA/AAB/ABA/ABB): {len(high_quality):,} ({100*len(high_quality)/len(df_clean):.1f}%)")

    # Save predictions
    logger.info("\n" + "="*80)
    logger.info("SAVING PREDICTIONS")
    logger.info("="*80)

    # Select columns to save
    output_cols = ['source_id', 'bp_rp', 'j_h_color', 'h_k_color', 'j_k_color',
                   'teff_predicted', 'ph_qual', 'j_snr', 'h_snr', 'k_snr']
    df_output = df_clean[output_cols].copy()

    output_file = processed_dir / f'gaia_2mass_temperature_predictions_{MODEL_ID}.parquet'
    df_output.to_parquet(output_file, index=False)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Saved: {len(df_output):,} objects with predicted temperatures")
    logger.info(f"Columns: {output_cols}")

    # Create summary document
    summary = f"""TEMPERATURE PREDICTIONS: GAIA + 2MASS COLORS MODEL
{'='*80}

Model: {MODEL_ID}
Prediction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT DATA
----------
Total Gaia catalog objects: {len(df_catalog):,}
Objects without Gaia Teff: {len(df_no_teff):,}
Objects with 2MASS photometry: {len(df_predict):,}
Objects with all 4 colors: {len(df_clean):,}

FEATURES USED
-------------
1. bp_rp (Gaia BP-RP color)
2. j_h_color (2MASS J-H color)
3. h_k_color (2MASS H-K color)
4. j_k_color (2MASS J-K color)

PREDICTIONS
-----------
Total predictions: {len(df_clean):,}
Temperature range: {predictions.min():.1f} - {predictions.max():.1f} K
Mean: {predictions.mean():.1f} K
Median: {np.median(predictions):.1f} K
Std: {predictions.std():.1f} K

TEMPERATURE DISTRIBUTION
------------------------
{chr(10).join(f'{label:15s}: {(df_clean["temp_bin"] == label).sum():6,} ({100*(df_clean["temp_bin"] == label).sum()/len(df_clean):5.1f}%)' for label in temp_labels)}

DATA QUALITY
------------
High quality 2MASS (AAA/AAB/ABA/ABB): {len(high_quality):,} ({100*len(high_quality)/len(df_clean):.1f}%)

Top 5 quality flags:
{chr(10).join(f'  {qual:3s}: {count:6,} ({100*count/len(df_clean):5.1f}%)' for qual, count in quality_counts.head(5).items())}

OUTPUT FILE
-----------
{output_file}

COLUMNS
-------
- source_id: Gaia DR3 source identifier (common key)
- bp_rp: Gaia BP-RP color [mag]
- j_h_color: 2MASS J-H color [mag]
- h_k_color: 2MASS H-K color [mag]
- j_k_color: 2MASS J-K color [mag]
- teff_predicted: Predicted effective temperature [K]
- ph_qual: 2MASS photometric quality flag
- j_snr, h_snr, k_snr: 2MASS signal-to-noise ratios

NOTES
-----
- All predictions use color indices only (no magnitude bias)
- Common key (source_id) enables joining with other Gaia-based tables
- Model trained on 525,738 objects with Gaia Teff
- Model test performance: MAE = 722 K, R² = 0.442
- Color ranges filtered to physically realistic values
"""

    summary_file = processed_dir / f'gaia_2mass_temperature_predictions_{MODEL_ID}_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Summary: {summary_file}")

    # Create schema documentation
    schema_doc = f"""# Gaia + 2MASS Temperature Predictions Schema

**File:** `data/processed/gaia_2mass_temperature_predictions_{MODEL_ID}.parquet`
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Objects:** {len(df_output):,}

## Description
Predicted effective temperatures for eclipsing binaries without Gaia GSP-Phot temperatures.
Uses Random Forest model trained on Gaia BP-RP and 2MASS J-H, H-K, J-K colors.

## Common Key
- **source_id** (int64): Gaia DR3 source identifier - unique key for joining with other tables

## Columns

### Identifier
- **source_id** (int64): Gaia DR3 source identifier

### Feature Columns (4 colors)
- **bp_rp** (float64): Gaia BP-RP color [mag]
  - Range: {df_output['bp_rp'].min():.3f} to {df_output['bp_rp'].max():.3f}
  - Mean: {df_output['bp_rp'].mean():.3f} ± {df_output['bp_rp'].std():.3f}

- **j_h_color** (float64): 2MASS J-H color [mag]
  - Range: {df_output['j_h_color'].min():.3f} to {df_output['j_h_color'].max():.3f}
  - Mean: {df_output['j_h_color'].mean():.3f} ± {df_output['j_h_color'].std():.3f}

- **h_k_color** (float64): 2MASS H-K color [mag]
  - Range: {df_output['h_k_color'].min():.3f} to {df_output['h_k_color'].max():.3f}
  - Mean: {df_output['h_k_color'].mean():.3f} ± {df_output['h_k_color'].std():.3f}

- **j_k_color** (float64): 2MASS J-K color [mag]
  - Range: {df_output['j_k_color'].min():.3f} to {df_output['j_k_color'].max():.3f}
  - Mean: {df_output['j_k_color'].mean():.3f} ± {df_output['j_k_color'].std():.3f}

### Prediction Column
- **teff_predicted** (float64): Predicted effective temperature [K]
  - Range: {df_output['teff_predicted'].min():.1f} to {df_output['teff_predicted'].max():.1f} K
  - Mean: {df_output['teff_predicted'].mean():.1f} ± {df_output['teff_predicted'].std():.1f} K
  - Median: {df_output['teff_predicted'].median():.1f} K

### Quality Columns
- **ph_qual** (str): 2MASS photometric quality flag (e.g., 'AAA', 'AAB')
- **j_snr** (float64): J-band signal-to-noise ratio
- **h_snr** (float64): H-band signal-to-noise ratio
- **k_snr** (float64): K-band signal-to-noise ratio

## Model Information
- **Model ID:** {MODEL_ID}
- **Training objects:** 525,738
- **Test MAE:** 722 K
- **Test R²:** 0.442
- **Features:** 4 color indices (no magnitudes)

## Data Quality
- All objects have valid colors in all 4 bands
- Color values filtered to physically realistic ranges:
  - bp_rp: -0.5 to 4.0 mag
  - j_h_color: -0.5 to 2.0 mag
  - h_k_color: -0.3 to 1.0 mag
  - j_k_color: -0.5 to 2.5 mag
- High quality 2MASS (AAA/AAB/ABA/ABB): {100*len(high_quality)/len(df_clean):.1f}%

## Source Tables
1. **eb_catalog.parquet**: Gaia source_id, bp_rp color
2. **eb_2mass_photometry.parquet**: 2MASS infrared colors

## Usage
```python
import pandas as pd

# Load predictions
df = pd.read_parquet('data/processed/gaia_2mass_temperature_predictions_{MODEL_ID}.parquet')

# Access predicted temperatures
temps = df['teff_predicted']

# Join with other tables using source_id
# df_merged = df.merge(other_table, on='source_id')
```

## Notes
- These predictions are for objects WITHOUT Gaia GSP-Phot temperatures
- Model is color-only (no magnitude bias, distance-independent)
- Common key (source_id) enables joining with other Gaia-based catalogs
- For training data with true Gaia Teff, see gaia_2mass_colors_training.parquet
"""

    schema_file = Path('docs') / f'GAIA_2MASS_PREDICTIONS_SCHEMA.md'
    with open(schema_file, 'w') as f:
        f.write(schema_doc)
    logger.info(f"Schema documentation: {schema_file}")

    logger.info("\n" + "="*80)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nPredicted temperatures for {len(df_output):,} objects")
    logger.info(f"Output: {output_file.name}")

if __name__ == '__main__':
    main()
