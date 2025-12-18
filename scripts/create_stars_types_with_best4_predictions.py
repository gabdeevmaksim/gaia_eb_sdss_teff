#!/usr/bin/env python3
"""
Create stars_types catalog with best-of-four temperature predictions.

Merges stars_types.dat (2.1M eclipsing binaries) with best-of-four
predictions (847k objects). Uses Gaia Teff when available, otherwise
ML predictions with lowest uncertainty from four models.

Output: FITS binary table + comprehensive description file
"""

import polars as pl
import numpy as np
from astropy.table import Table
from astropy.io import fits
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_stars_types(filepath):
    """
    Load stars_types.dat catalog.

    Format (CSV with header):
    - Name: Gaia DR3 source_id
    - Alpha, Delta: RA, Dec (J2000, degrees)
    - Period: Orbital period (days)
    - Teff: Effective temperature (K) from Gaia, '--' if missing
    - Type_model020-100: Binary type from different models
    - Type: Final binary type (detached/overcontact)
    - Amplitude: Light curve amplitude (mag), '--' if missing

    Note: Some lines have "[conflicted]" suffix on source_id - these are cleaned
    """
    logger.info(f"Loading stars_types.dat from {filepath}")

    # Read with Polars, reading Name as string first to handle "[conflicted]" suffix
    df = pl.read_csv(
        filepath,
        separator=',',
        null_values=['--'],
        schema_overrides={
            'Name': pl.Utf8,  # Read as string first
            'Alpha': pl.Float64,
            'Delta': pl.Float64,
            'Period': pl.Float64,
            'Teff': pl.Float64,  # '--' will become null
            'Type': pl.Utf8,
            'Amplitude': pl.Float64  # '--' will become null
        }
    )

    # Clean source_id: remove " [conflicted]" suffix and convert to Int64
    df = df.with_columns([
        pl.col('Name').str.replace(' \\[conflicted\\]', '').cast(pl.Int64).alias('source_id')
    ])

    # Select and rename columns we need
    df = df.select([
        'source_id',
        pl.col('Alpha').alias('ra'),
        pl.col('Delta').alias('dec'),
        pl.col('Period').alias('period'),
        pl.col('Teff').alias('teff_gaia'),
        pl.col('Type').alias('binary_type'),
        pl.col('Amplitude').alias('amplitude')
    ])

    logger.info(f"Loaded {len(df):,} objects from stars_types.dat")
    logger.info(f"  Objects with Gaia Teff: {df['teff_gaia'].is_not_null().sum():,}")
    logger.info(f"  Objects without Gaia Teff: {df['teff_gaia'].is_null().sum():,}")

    return df

def load_best_predictions(filepath):
    """Load best-of-four predictions."""
    logger.info(f"Loading best-of-four predictions from {filepath}")

    df = pl.read_parquet(filepath)

    logger.info(f"Loaded {len(df):,} predictions")
    logger.info(f"  Mean Teff: {df['teff_best_4'].mean():.0f} K")
    logger.info(f"  Mean uncertainty: {df['unc_best_4'].mean():.0f} K")

    return df

def merge_and_create_catalog(df_stars, df_predictions):
    """
    Merge stars_types with predictions and create final catalog.

    Logic:
    - Use Gaia Teff if available
    - Otherwise use ML prediction (best-of-four)
    - Create quality flags based on uncertainty thresholds
    """
    logger.info("Merging catalogs...")

    # Left join: keep all stars_types objects, add predictions where available
    df_merged = df_stars.join(
        df_predictions.select([
            'source_id', 'teff_best_4', 'unc_best_4', 'best_model_4'
        ]),
        on='source_id',
        how='left'
    )

    logger.info(f"Merged catalog: {len(df_merged):,} objects")
    logger.info(f"  Matched predictions: {df_merged['teff_best_4'].is_not_null().sum():,}")

    # Create final temperature column (Gaia if available, else ML)
    df_merged = df_merged.with_columns([
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.col('teff_gaia'))
          .otherwise(pl.col('teff_best_4'))
          .alias('teff_final'),

        # Temperature source indicator
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.lit('Gaia'))
          .when(pl.col('teff_best_4').is_not_null())
          .then(pl.col('best_model_4'))
          .otherwise(pl.lit('none'))
          .alias('teff_source'),

        # Uncertainty (only for ML predictions)
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.lit(None))  # No uncertainty for Gaia
          .otherwise(pl.col('unc_best_4'))
          .alias('teff_uncertainty')
    ])

    # Create quality flags based on uncertainty
    df_merged = df_merged.with_columns([
        pl.when(pl.col('teff_source') == 'Gaia')
          .then(pl.lit('A'))  # Gaia = highest quality
        .when(pl.col('teff_uncertainty') < 300)
          .then(pl.lit('B'))  # Low uncertainty ML
        .when(pl.col('teff_uncertainty') < 500)
          .then(pl.lit('C'))  # Medium uncertainty ML
        .when(pl.col('teff_uncertainty').is_not_null())
          .then(pl.lit('D'))  # High uncertainty ML
        .otherwise(pl.lit('X'))  # No prediction
        .alias('quality_flag')
    ])

    # Calculate statistics
    total = len(df_merged)
    with_teff = df_merged['teff_final'].is_not_null().sum()
    from_gaia = (df_merged['teff_source'] == 'Gaia').sum()
    from_ml = df_merged['teff_source'].is_in(['teff_only', 'teff_logg', 'teff_cluster', 'teff_flag1']).sum()

    logger.info(f"\nFinal catalog statistics:")
    logger.info(f"  Total objects: {total:,}")
    logger.info(f"  Objects with Teff: {with_teff:,} ({100*with_teff/total:.1f}%)")
    logger.info(f"    From Gaia: {from_gaia:,} ({100*from_gaia/total:.1f}%)")
    logger.info(f"    From ML: {from_ml:,} ({100*from_ml/total:.1f}%)")
    logger.info(f"  No Teff: {total - with_teff:,} ({100*(total-with_teff)/total:.1f}%)")

    # Quality distribution
    quality_counts = df_merged.group_by('quality_flag').len().sort('quality_flag')
    logger.info(f"\nQuality flag distribution:")
    for row in quality_counts.iter_rows(named=True):
        flag = row['quality_flag']
        count = row['len']  # len() function creates 'len' column
        pct = 100 * count / total
        logger.info(f"  {flag}: {count:,} ({pct:.1f}%)")

    return df_merged

def save_fits_catalog(df, output_path):
    """Save catalog as FITS binary table."""
    logger.info(f"Saving FITS catalog to {output_path}")

    # Convert to Astropy Table
    table = Table()
    table['source_id'] = df['source_id'].to_numpy().astype(np.int64)
    table['ra'] = df['ra'].to_numpy().astype(np.float64)
    table['dec'] = df['dec'].to_numpy().astype(np.float64)
    table['period'] = df['period'].to_numpy().astype(np.float64)
    table['teff_gaia'] = df['teff_gaia'].to_numpy().astype(np.float64)
    table['binary_type'] = df['binary_type'].to_numpy().astype('U11')  # Fixed length string
    table['amplitude'] = df['amplitude'].to_numpy().astype(np.float64)
    table['teff_predicted'] = df['teff_best_4'].to_numpy().astype(np.float64)
    table['teff_uncertainty'] = df['teff_uncertainty'].to_numpy().astype(np.float64)
    table['teff_final'] = df['teff_final'].to_numpy().astype(np.float64)
    table['teff_source'] = df['teff_source'].to_numpy().astype('U12')  # Fixed length string
    table['quality_flag'] = df['quality_flag'].to_numpy().astype('U1')  # Fixed length string

    # Add column descriptions
    table['source_id'].description = 'Gaia DR3 source identifier'
    table['ra'].description = 'Right Ascension (J2000, degrees)'
    table['dec'].description = 'Declination (J2000, degrees)'
    table['period'].description = 'Orbital period (days)'
    table['teff_gaia'].description = 'Effective temperature from Gaia GSP-Phot (K)'
    table['binary_type'].description = 'Binary type (D=detached, C=overcontact)'
    table['amplitude'].description = 'Light curve amplitude (mag)'
    table['teff_predicted'].description = 'ML predicted temperature from best-of-four (K)'
    table['teff_uncertainty'].description = 'ML prediction uncertainty (K)'
    table['teff_final'].description = 'Final temperature (Gaia if available, else ML)'
    table['teff_source'].description = 'Temperature source (Gaia, teff_only, teff_logg, teff_cluster, teff_flag1, none)'
    table['quality_flag'].description = 'Quality flag (A=Gaia, B=ML<300K, C=ML<500K, D=ML>=500K, X=none)'

    # Add units
    table['ra'].unit = 'deg'
    table['dec'].unit = 'deg'
    table['period'].unit = 'd'
    table['teff_gaia'].unit = 'K'
    table['amplitude'].unit = 'mag'
    table['teff_predicted'].unit = 'K'
    table['teff_uncertainty'].unit = 'K'
    table['teff_final'].unit = 'K'

    # Write FITS file
    table.write(output_path, format='fits', overwrite=True)
    logger.info(f"FITS catalog saved: {output_path.stat().st_size / 1e6:.1f} MB")

def create_description_file(df, output_path):
    """Create comprehensive description file."""
    logger.info(f"Creating description file: {output_path}")

    # Calculate statistics
    total = len(df)
    with_teff = df['teff_final'].is_not_null().sum()
    from_gaia = (df['teff_source'] == 'Gaia').sum()
    from_ml = df['teff_source'].is_in(['teff_only', 'teff_logg', 'teff_cluster', 'teff_flag1']).sum()

    # Quality distribution
    quality_counts = df.group_by('quality_flag').len().sort('quality_flag')

    # Model selection distribution (for ML predictions)
    df_ml = df.filter(pl.col('teff_source').is_in(['teff_only', 'teff_logg', 'teff_cluster', 'teff_flag1']))
    model_counts = df_ml.group_by('teff_source').len().sort('teff_source')

    # Uncertainty statistics (for ML predictions)
    unc_mean = df_ml['teff_uncertainty'].mean()
    unc_median = df_ml['teff_uncertainty'].median()
    unc_std = df_ml['teff_uncertainty'].std()
    unc_min = df_ml['teff_uncertainty'].min()
    unc_max = df_ml['teff_uncertainty'].max()
    unc_25 = df_ml['teff_uncertainty'].quantile(0.25)
    unc_75 = df_ml['teff_uncertainty'].quantile(0.75)

    today = datetime.now().strftime('%Y-%m-%d')

    description = f"""# stars_types_with_best4_predictions.fits - Description

## Overview

This catalog contains effective temperature predictions for 2.1 million eclipsing binary stars.
It merges the base stars_types.dat catalog with ML predictions using a "best-of-four" ensemble
approach that selects the prediction with the lowest uncertainty for each object.

**Creation Date**: {today}
**Total Objects**: {total:,}
**Objects with Teff**: {with_teff:,} ({100*with_teff/total:.1f}%)
**Temperature Range**: {df['teff_final'].min():.0f} - {df['teff_final'].max():.0f} K

## Temperature Sources

1. **Gaia GSP-Phot**: {from_gaia:,} objects ({100*from_gaia/total:.1f}%)
   - High-quality spectrophotometric temperatures from Gaia DR3
   - No uncertainty estimates provided

2. **ML Predictions**: {from_ml:,} objects ({100*from_ml/total:.1f}%)
   - Best-of-four ensemble (selects lowest uncertainty)
   - Four models compared:
     * teff_only: Gaia photometry only (g, BP, RP, BP-RP)
     * teff_logg: Gaia photometry + log(g) with uncertainty propagation
     * teff_cluster: Gaia photometry + cluster probabilities
     * teff_flag1: Gaia photometry (trained on flag 1 high-quality sources)
   - Mean uncertainty: {df_ml['teff_uncertainty'].mean():.0f} K

3. **No Prediction**: {total - with_teff:,} objects ({100*(total-with_teff)/total:.1f}%)
   - Objects without Gaia Teff and outside ML training domain

## Quality Flags

Quality assessment based on temperature source and uncertainty:

"""

    # Add quality flag descriptions
    for row in quality_counts.iter_rows(named=True):
        flag = row['quality_flag']
        count = row['len']
        pct = 100 * count / total

        descriptions = {
            'A': 'Gaia GSP-Phot temperature (highest quality)',
            'B': 'ML prediction with uncertainty < 300 K (high confidence)',
            'C': 'ML prediction with uncertainty < 500 K (medium confidence)',
            'D': 'ML prediction with uncertainty >= 500 K (low confidence)',
            'X': 'No temperature available'
        }

        description += f"- **{flag}**: {descriptions.get(flag, 'Unknown')} - {count:,} objects ({pct:.1f}%)\n"

    description += f"""
## Model Selection Distribution (ML predictions only)

Best-of-four ensemble selects the model with lowest uncertainty for each object:

"""

    for row in model_counts.iter_rows(named=True):
        model = row['teff_source']
        count = row['len']
        pct = 100 * count / from_ml if from_ml > 0 else 0
        description += f"- **{model}**: {count:,} objects ({pct:.1f}%)\n"

    description += f"""
## Uncertainty Statistics (ML predictions only)

- **Mean**: {unc_mean:.0f} K
- **Median**: {unc_median:.0f} K
- **Std Dev**: {unc_std:.0f} K
- **Min**: {unc_min:.0f} K
- **Max**: {unc_max:.0f} K
- **25th percentile**: {unc_25:.0f} K
- **75th percentile**: {unc_75:.0f} K

## Column Descriptions

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| source_id | int64 | - | Gaia DR3 source identifier |
| ra | float64 | deg | Right Ascension (J2000) |
| dec | float64 | deg | Declination (J2000) |
| period | float64 | d | Orbital period |
| teff_gaia | float64 | K | Effective temperature from Gaia GSP-Phot (null if unavailable) |
| binary_type | str | - | Binary type (D=detached, C=overcontact) |
| amplitude | float64 | mag | Light curve amplitude |
| teff_predicted | float64 | K | ML predicted temperature from best-of-four (null if no prediction) |
| teff_uncertainty | float64 | K | ML prediction uncertainty (null for Gaia temperatures) |
| teff_final | float64 | K | Final temperature (Gaia if available, else ML, null if neither) |
| teff_source | str | - | Temperature source (Gaia, teff_only, teff_logg, teff_cluster, teff_flag1, none) |
| quality_flag | str | - | Quality flag (A/B/C/D/X, see above) |

## Usage Examples

### Python (Astropy)

```python
from astropy.table import Table

# Load catalog
catalog = Table.read('stars_types_with_best4_predictions.fits')

# Filter by quality
high_quality = catalog[catalog['quality_flag'] <= 'B']  # Gaia or low-uncertainty ML
print(f"High-quality objects: {{len(high_quality):,}}")

# Access temperatures
teff = catalog['teff_final']
uncertainty = catalog['teff_uncertainty']

# Filter by temperature range
cool_stars = catalog[(catalog['teff_final'] > 3000) & (catalog['teff_final'] < 5000)]
```

### Python (Polars)

```python
import polars as pl
from astropy.table import Table

# Load FITS as Astropy Table, convert to Polars
table = Table.read('stars_types_with_best4_predictions.fits')
df = pl.from_pandas(table.to_pandas())

# Filter high-quality predictions
high_quality = df.filter(
    pl.col('quality_flag').is_in(['A', 'B'])
)

# Analyze by source
source_summary = df.group_by('teff_source').agg([
    pl.count().alias('count'),
    pl.col('teff_final').mean().alias('mean_teff')
])
```

## Quality Recommendations

**For scientific analysis**:
- **Best quality**: Use quality_flag == 'A' (Gaia only) for highest reliability
- **Good quality**: Use quality_flag <= 'B' (Gaia + low-uncertainty ML) for larger sample
- **Acceptable**: Use quality_flag <= 'C' for exploratory analysis

**For specific temperature ranges**:
- Cool stars (<5000 K): Quality B-C recommended (ML performs well)
- Hot stars (>6000 K): Quality A-B recommended (Gaia more reliable)

**For uncertainty-aware analysis**:
- Always propagate `teff_uncertainty` when available
- Gaia temperatures (quality_flag='A') have no formal uncertainties but are generally reliable

## Methodology

### Best-of-Four Ensemble

For each object, four ML models were evaluated:

1. **Teff Only** (Gaia photometry)
   - Features: g, BP, RP, BP-RP
   - Predicts: Corrected Teff (polynomial correction for T>10000K)
   - Uncertainty: Random Forest tree variance (full 300 trees)

2. **Teff with log(g)** (Gaia photometry + surface gravity)
   - Features: g, BP, RP, BP-RP, log(g)
   - Uncertainty propagation: Numerical gradient method
   - Combined uncertainty: RF + log(g) contribution in quadrature

3. **Teff with Clustering** (Gaia photometry + cluster probabilities)
   - Features: g, BP, RP, BP-RP + cluster membership probabilities
   - K-means clustering in color-magnitude space
   - Uncertainty: Random Forest tree variance

4. **Teff Flag 1** (Gaia photometry, high-quality training)
   - Features: g, BP, RP, BP-RP
   - Training: Only Gaia GSP-Phot flag 1 sources (highest quality)
   - Corrected Teff target with polynomial correction
   - Uncertainty: Random Forest tree variance (full 300 trees)

**Selection criteria**: For each object, the model with the lowest uncertainty was selected.
This approach maximizes the number of high-confidence predictions while maintaining accuracy.

**Improvement**: Mean uncertainty reduced by 22.8% compared to best-of-three (263K → 203K).

### Training Data

- **Source**: Gaia DR3 GSP-Phot temperatures (high-quality subsample)
- **Size**: ~700,000 eclipsing binaries with reliable Teff
- **Filters**: Quality flag filtering, outlier removal, photometric quality cuts
- **Model**: Random Forest Regressor (300 trees)
- **Validation**: Cross-validation on held-out test set

## References

- **Gaia DR3**: https://www.cosmos.esa.int/web/gaia/dr3
- **GSP-Phot**: Gaia Spectro-Photometric analysis pipeline
- **Best-of-Four Methodology**: See `reports/figures/best_of_four_ensemble/`

## Contact

For questions or issues with this catalog, please contact the repository maintainer.

## Version History

- **v2.0** ({today}): Best-of-four ensemble with flag 1 model (22.8% improvement)
- **v1.0** (2025-11-20): Initial release with best-of-three ensemble predictions
"""

    # Write description file
    with open(output_path, 'w') as f:
        f.write(description)

    logger.info(f"Description file created: {output_path}")

def main():
    """Main execution."""
    logger.info("=== Creating stars_types catalog with best-of-four predictions ===\n")

    # Define paths
    base_dir = Path(__file__).parent.parent
    stars_types_file = base_dir / 'data' / 'raw' / 'stars_types.dat'
    predictions_file = base_dir / 'data' / 'processed' / 'teff_predictions_best_of_four.parquet'
    output_fits = base_dir / 'data' / 'processed' / 'stars_types_with_best4_predictions.fits'
    output_desc = base_dir / 'data' / 'processed' / 'stars_types_with_best4_predictions_DESCRIPTION.txt'

    # Load data
    df_stars = load_stars_types(stars_types_file)
    df_predictions = load_best_predictions(predictions_file)

    # Merge and create catalog
    df_final = merge_and_create_catalog(df_stars, df_predictions)

    # Save FITS catalog
    save_fits_catalog(df_final, output_fits)

    # Create description file
    create_description_file(df_final, output_desc)

    logger.info("\n=== Catalog creation complete ===")
    logger.info(f"Output files:")
    logger.info(f"  FITS catalog: {output_fits}")
    logger.info(f"  Description: {output_desc}")

if __name__ == '__main__':
    main()
