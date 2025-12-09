#!/usr/bin/env python3
"""
Create final stars_types catalog with best-of-three corrected Teff predictions.

Uses three models trained on CORRECTED Teff (polynomial correction for >10000K):
1. Gaia colors only → log(Teff_corrected)
2. Gaia + logg → log(Teff_corrected)
3. Gaia + clustering → log(Teff_corrected)

Selects prediction with lowest uncertainty for each object.
Merges with stars_types.dat (2.1M eclipsing binaries).

Output: FITS binary table + comprehensive description file
"""

import polars as pl
import numpy as np
from astropy.table import Table
from astropy.io import fits
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_predictions():
    """Load predictions from all three corrected Teff models."""
    logger.info("=" * 80)
    logger.info("LOADING PREDICTIONS FROM THREE MODELS (CORRECTED TEFF)")
    logger.info("=" * 80)

    # Model 1: Gaia colors only → log(Teff_corrected)
    logger.info("\n[1/3] Loading Gaia Colors Only...")
    df1 = pl.read_parquet('data/processed/predictions_gaia_teff_corrected_log.parquet')
    df1 = df1.select([
        'source_id',
        pl.col('teff_gaia_corrected_log_predicted').alias('teff_only'),
        pl.col('teff_gaia_corrected_log_uncertainty').alias('unc_only')
    ])
    logger.info(f"  Loaded {len(df1):,} predictions")
    logger.info(f"  Mean: {df1['teff_only'].mean():.0f} K, Unc: {df1['unc_only'].mean():.1f} K")

    # Model 2: Gaia + logg → log(Teff_corrected)
    logger.info("\n[2/3] Loading Gaia + logg...")
    df2 = pl.read_parquet('data/processed/predictions_gaia_logg_teff_log_corrected.parquet')
    df2 = df2.select([
        'source_id',
        pl.col('teff_gaia_corrected_log_predicted').alias('teff_logg'),
        pl.col('teff_gaia_corrected_log_uncertainty').alias('unc_logg')
    ])
    logger.info(f"  Loaded {len(df2):,} predictions")
    logger.info(f"  Mean: {df2['teff_logg'].mean():.0f} K, Unc: {df2['unc_logg'].mean():.1f} K")

    # Model 3: Gaia + clustering → log(Teff_corrected)
    logger.info("\n[3/3] Loading Gaia + Clustering...")
    df3 = pl.read_parquet('data/processed/teff_predictions_cluster_corrected_log.parquet')
    df3 = df3.select([
        'source_id',
        pl.col('teff_gaia_corrected_log_predicted').alias('teff_cluster'),
        pl.col('teff_gaia_corrected_log_uncertainty').alias('unc_cluster')
    ])
    logger.info(f"  Loaded {len(df3):,} predictions")
    logger.info(f"  Mean: {df3['teff_cluster'].mean():.0f} K, Unc: {df3['unc_cluster'].mean():.1f} K")

    # Merge all three using inner join (only objects with all 3 predictions)
    logger.info("\n[4/4] Merging...")
    df_merged = (df1
                 .join(df2, on='source_id', how='inner')
                 .join(df3, on='source_id', how='inner'))

    logger.info(f"  Merged: {len(df_merged):,} objects with all three predictions")

    return df_merged


def select_best_by_uncertainty(df):
    """For each object, select the model with lowest uncertainty."""
    logger.info("\n" + "=" * 80)
    logger.info("SELECTING BEST PREDICTIONS BY UNCERTAINTY")
    logger.info("=" * 80)

    # Find which model has minimum uncertainty for each object
    df_best = df.with_columns([
        pl.when(
            (pl.col('unc_only') <= pl.col('unc_logg')) &
            (pl.col('unc_only') <= pl.col('unc_cluster'))
        ).then(pl.lit('teff_only'))
        .when(
            (pl.col('unc_logg') <= pl.col('unc_only')) &
            (pl.col('unc_logg') <= pl.col('unc_cluster'))
        ).then(pl.lit('teff_logg'))
        .otherwise(pl.lit('teff_cluster'))
        .alias('best_model')
    ])

    # Select the corresponding prediction and uncertainty
    df_best = df_best.with_columns([
        pl.when(pl.col('best_model') == 'teff_only')
          .then(pl.col('teff_only'))
        .when(pl.col('best_model') == 'teff_logg')
          .then(pl.col('teff_logg'))
        .otherwise(pl.col('teff_cluster'))
        .alias('teff_best'),

        pl.when(pl.col('best_model') == 'teff_only')
          .then(pl.col('unc_only'))
        .when(pl.col('best_model') == 'teff_logg')
          .then(pl.col('unc_logg'))
        .otherwise(pl.col('unc_cluster'))
        .alias('unc_best')
    ])

    # Report statistics
    logger.info("\nModel selection distribution:")
    model_counts = df_best.group_by('best_model').agg(pl.count().alias('count'))
    total = len(df_best)
    for row in model_counts.iter_rows(named=True):
        model = row['best_model']
        count = row['count']
        pct = 100 * count / total
        logger.info(f"  {model}: {count:,} ({pct:.1f}%)")

    logger.info(f"\nUncertainty statistics (best-of-three):")
    unc_best = df_best['unc_best'].to_numpy()
    logger.info(f"  Mean:   {np.mean(unc_best):.1f} K")
    logger.info(f"  Median: {np.median(unc_best):.1f} K")
    logger.info(f"  Std:    {np.std(unc_best):.1f} K")
    logger.info(f"  Min:    {np.min(unc_best):.1f} K")
    logger.info(f"  Max:    {np.max(unc_best):.1f} K")

    return df_best


def load_stars_types(filepath):
    """
    Load stars_types.dat catalog (2.1M eclipsing binaries).

    Format: CSV with header
    - Name: Gaia DR3 source_id
    - Alpha, Delta: RA, Dec (J2000, degrees)
    - Period: Orbital period (days)
    - Teff: Effective temperature (K) from Gaia GSP-Phot, '--' if missing
    - Type: Binary type (detached/overcontact)
    - Amplitude: Light curve amplitude (mag), '--' if missing
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING STARS_TYPES.DAT")
    logger.info("=" * 80)

    df = pl.read_csv(
        filepath,
        separator=',',
        null_values=['--'],
        schema_overrides={
            'Name': pl.Utf8,  # Read as string to handle "[conflicted]" suffix
            'Alpha': pl.Float64,
            'Delta': pl.Float64,
            'Period': pl.Float64,
            'Teff': pl.Float64,
            'Type': pl.Utf8,
            'Amplitude': pl.Float64
        }
    )

    # Clean source_id: remove " [conflicted]" suffix
    df = df.with_columns([
        pl.col('Name').str.replace(' \\[conflicted\\]', '').cast(pl.Int64).alias('source_id')
    ])

    # Select and rename columns
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


def merge_and_create_catalog(df_stars, df_predictions):
    """
    Merge stars_types with predictions and create final catalog.

    Logic:
    - Use Gaia Teff if available (source='Gaia')
    - Otherwise use ML prediction with lowest uncertainty
    - Create quality flags:
      * A: Gaia GSP-Phot (highest quality)
      * B: ML with unc < 300 K
      * C: ML with unc < 500 K
      * D: ML with unc >= 500 K
      * X: No temperature available
    """
    logger.info("\n" + "=" * 80)
    logger.info("MERGING CATALOGS")
    logger.info("=" * 80)

    # Left join: keep all stars_types objects, add predictions where available
    df_merged = df_stars.join(
        df_predictions.select([
            'source_id', 'teff_best', 'unc_best', 'best_model'
        ]),
        on='source_id',
        how='left'
    )

    logger.info(f"Merged catalog: {len(df_merged):,} objects")
    logger.info(f"  Matched predictions: {df_merged['teff_best'].is_not_null().sum():,}")

    # Rename ML columns for clarity
    df_merged = df_merged.rename({
        'teff_best': 'teff_predicted',
        'unc_best': 'teff_uncertainty'
    })

    # Create final temperature column (Gaia if available, else ML)
    df_merged = df_merged.with_columns([
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.col('teff_gaia'))
          .otherwise(pl.col('teff_predicted'))
          .alias('teff_final'),

        # Temperature source
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.lit('Gaia'))
          .when(pl.col('teff_predicted').is_not_null())
          .then(pl.col('best_model'))
          .otherwise(pl.lit('none'))
          .alias('teff_source'),

        # Keep uncertainty only for ML predictions
        pl.when(pl.col('teff_gaia').is_not_null())
          .then(pl.lit(None))  # No uncertainty for Gaia
          .otherwise(pl.col('teff_uncertainty'))
          .alias('teff_uncertainty')
    ])

    # Drop best_model column (info now in teff_source)
    df_merged = df_merged.drop('best_model')

    # Create quality flags
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

    # Report statistics
    logger.info("\nFinal catalog statistics:")
    logger.info(f"  Total objects: {len(df_merged):,}")
    logger.info(f"  Objects with Teff: {df_merged['teff_final'].is_not_null().sum():,}")
    logger.info(f"  Objects without Teff: {df_merged['teff_final'].is_null().sum():,}")

    logger.info("\nTemperature sources:")
    for source in ['Gaia', 'teff_only', 'teff_logg', 'teff_cluster', 'none']:
        count = (df_merged['teff_source'] == source).sum()
        pct = 100 * count / len(df_merged)
        logger.info(f"  {source}: {count:,} ({pct:.1f}%)")

    logger.info("\nQuality distribution:")
    for flag in ['A', 'B', 'C', 'D', 'X']:
        count = (df_merged['quality_flag'] == flag).sum()
        pct = 100 * count / len(df_merged)
        logger.info(f"  {flag}: {count:,} ({pct:.1f}%)")

    return df_merged


def save_as_fits(df, output_file):
    """Save catalog as FITS binary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SAVING AS FITS")
    logger.info("=" * 80)

    # Convert to pandas for astropy (polars FITS support is limited)
    df_pd = df.to_pandas()

    # Create astropy Table
    table = Table.from_pandas(df_pd)

    # Add column descriptions
    table['source_id'].description = 'Gaia DR3 source identifier'
    table['ra'].description = 'Right Ascension (J2000)'
    table['dec'].description = 'Declination (J2000)'
    table['period'].description = 'Orbital period'
    table['teff_gaia'].description = 'Effective temperature from Gaia GSP-Phot'
    table['binary_type'].description = 'Binary type (D=detached, C=overcontact)'
    table['amplitude'].description = 'Light curve amplitude'
    table['teff_predicted'].description = 'ML predicted temperature (best-of-three)'
    table['teff_uncertainty'].description = 'ML prediction uncertainty'
    table['teff_final'].description = 'Final temperature (Gaia if available, else ML)'
    table['teff_source'].description = 'Temperature source (Gaia, teff_only, teff_logg, teff_cluster, none)'
    table['quality_flag'].description = 'Quality flag (A/B/C/D/X)'

    # Set units
    table['ra'].unit = 'deg'
    table['dec'].unit = 'deg'
    table['period'].unit = 'd'
    table['teff_gaia'].unit = 'K'
    table['amplitude'].unit = 'mag'
    table['teff_predicted'].unit = 'K'
    table['teff_uncertainty'].unit = 'K'
    table['teff_final'].unit = 'K'

    # Write FITS file
    table.write(output_file, format='fits', overwrite=True)

    logger.info(f"✓ Saved: {output_file}")
    logger.info(f"  File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")


def main():
    """Main execution."""
    logger.info("\n" + "=" * 80)
    logger.info("CREATE FINAL CATALOG WITH CORRECTED TEFF PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Load and merge all 3 model predictions
    df_predictions = load_all_predictions()

    # Step 2: Select best prediction by uncertainty
    df_best = select_best_by_uncertainty(df_predictions)

    # Step 3: Save intermediate best-of-three ensemble
    output_best = 'data/processed/teff_predictions_best_of_three_corrected.parquet'
    df_best.write_parquet(output_best)
    logger.info(f"\n✓ Saved best-of-three ensemble: {output_best}")

    # Step 4: Load stars_types.dat
    df_stars = load_stars_types('data/raw/stars_types.dat')

    # Step 5: Merge with predictions
    df_final = merge_and_create_catalog(df_stars, df_best)

    # Step 6: Save as FITS
    output_fits = 'data/processed/stars_types_with_best_predictions_corrected.fits'
    save_as_fits(df_final, output_fits)

    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  1. {output_best}")
    logger.info(f"  2. {output_fits}")


if __name__ == '__main__':
    main()
