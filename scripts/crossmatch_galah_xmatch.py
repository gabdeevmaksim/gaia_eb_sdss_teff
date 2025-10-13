#!/usr/bin/env python3
"""
Cross-match with GALAH DR3 using VizieR Xmatch service.

This script uses VizieR's Xmatch service to cross-match flag 0 predictions
with GALAH DR3 spectroscopic data.

Usage:
    python scripts/crossmatch_galah_xmatch.py

Author: Auto-generated
Date: 2025-10-13
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astroquery.xmatch import XMatch
from astroquery.gaia import Gaia

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def get_coordinates_from_gaia(source_ids: np.ndarray, batch_size: int = 10000) -> pd.DataFrame:
    """Get RA/Dec coordinates from Gaia for source_ids."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Querying Gaia for coordinates ({len(source_ids):,} sources)...")
    
    n_batches = (len(source_ids) + batch_size - 1) // batch_size
    all_coords = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(source_ids))
        batch_ids = source_ids[start_idx:end_idx]
        
        logger.info(f"  Batch {i+1}/{n_batches} ({len(batch_ids):,} sources)...")
        
        query = f"""
        SELECT source_id, ra, dec
        FROM gaiadr3.gaia_source
        WHERE source_id IN ({','.join(map(str, batch_ids))})
        """
        
        job = Gaia.launch_job(query)
        coords_batch = job.get_results().to_pandas()
        all_coords.append(coords_batch)
    
    coords_df = pd.concat(all_coords, ignore_index=True)
    logger.info(f"  Retrieved {len(coords_df):,} coordinates")
    
    return coords_df


def crossmatch_galah_xmatch(
    input_file: Path,
    output_file: Path,
    max_distance: float = 2.0
) -> pd.DataFrame:
    """
    Cross-match with GALAH DR3 using VizieR Xmatch.
    
    Parameters
    ----------
    input_file : Path
        Input predictions file
    output_file : Path
        Output cross-matched file
    max_distance : float
        Maximum matching distance in arcseconds
        
    Returns
    -------
    pd.DataFrame
        Cross-matched data
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("CROSS-MATCH WITH GALAH DR3 VIA VIZIER XMATCH")
    logger.info("=" * 70)
    logger.info(f"Input: {input_file.name}")
    logger.info(f"Output: {output_file.name}")
    logger.info(f"Max distance: {max_distance} arcsec")
    logger.info("")
    
    # Load input
    logger.info("Loading flag 0 predictions...")
    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} sources")
    
    # Get coordinates from Gaia
    coords_df = get_coordinates_from_gaia(df['source_id'].values)
    
    # Merge coordinates
    df = df.merge(coords_df, on='source_id', how='left')
    
    n_with_coords = df['ra'].notna().sum()
    logger.info(f"  Sources with coordinates: {n_with_coords:,}/{len(df):,}")
    
    # Prepare upload table
    logger.info("\nPreparing table for Xmatch...")
    upload_table = Table({
        'source_id': df['source_id'].values,
        'ra': df['ra'].values,
        'dec': df['dec'].values,
        'gaia_teff': df['teff_gspphot'].values,
        'predicted_teff': df['predicted_teff'].values
    })
    
    logger.info(f"  Upload table: {len(upload_table):,} rows")
    
    # Perform Xmatch with GALAH DR3
    logger.info("\nSubmitting Xmatch query to VizieR...")
    logger.info("  Catalog: J/A+A/673/A155/galahdr3")
    logger.info("  This may take a few minutes...")
    
    try:
        result = XMatch.query(
            cat1=upload_table,
            cat2='vizier:J/A+A/673/A155/galahdr3',
            max_distance=max_distance * u.arcsec,
            colRA1='ra',
            colDec1='dec'
        )
        
        logger.info(f"✓ Xmatch completed: {len(result):,} matches found")
        
        # Convert to pandas
        matched = result.to_pandas()
        
        logger.info(f"  Result columns: {list(matched.columns)[:20]}...")
        
    except Exception as e:
        logger.error(f"✗ Xmatch failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Apply quality cuts
    logger.info("\nApplying quality criteria...")
    n_initial = len(matched)
    
    mask = np.ones(len(matched), dtype=bool)
    
    # Check what temperature columns are available
    temp_cols = [c for c in matched.columns if 'teff' in c.lower()]
    logger.info(f"  Available temperature columns: {temp_cols}")
    
    # Use Teff50 if available (median temperature)
    if 'Teff50' in matched.columns:
        n_before = mask.sum()
        mask &= matched['Teff50'].notna()
        mask &= (matched['Teff50'] > 0)
        mask &= (matched['Teff50'] >= 2500) & (matched['Teff50'] <= 50000)
        n_after = mask.sum()
        logger.info(f"  Valid Teff50: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
    
    matched_clean = matched[mask].copy()
    logger.info(f"\n  Final sample: {len(matched_clean):,}/{n_initial:,} ({len(matched_clean)/n_initial*100:.1f}%)")
    
    # Rename columns
    rename_map = {
        'Teff50': 'galah_teff',
        'logg50': 'galah_logg',
        'Mass50': 'galah_mass',
        'Dist50': 'galah_distance',
        'AV50': 'galah_av',
        'Age50': 'galah_age'
    }
    
    for old_col, new_col in rename_map.items():
        if old_col in matched_clean.columns:
            matched_clean = matched_clean.rename(columns={old_col: new_col})
    
    # Calculate statistics
    if 'gaia_teff' in matched_clean.columns and 'predicted_teff' in matched_clean.columns and 'galah_teff' in matched_clean.columns:
        logger.info("\n" + "=" * 70)
        logger.info("TEMPERATURE COMPARISON VS GALAH SPECTROSCOPY")
        logger.info("=" * 70)
        
        # Original vs GALAH
        gaia_residual = matched_clean['gaia_teff'] - matched_clean['galah_teff']
        gaia_mae = np.mean(np.abs(gaia_residual))
        gaia_rmse = np.sqrt(np.mean(gaia_residual**2))
        gaia_median = np.median(gaia_residual)
        
        logger.info(f"\nOriginal Gaia GSP-Phot (flag 0) vs GALAH:")
        logger.info(f"  MAE:  {gaia_mae:.1f} K")
        logger.info(f"  RMSE: {gaia_rmse:.1f} K")
        logger.info(f"  Median residual: {gaia_median:.1f} K")
        
        # Predicted vs GALAH
        pred_residual = matched_clean['predicted_teff'] - matched_clean['galah_teff']
        pred_mae = np.mean(np.abs(pred_residual))
        pred_rmse = np.sqrt(np.mean(pred_residual**2))
        pred_median = np.median(pred_residual)
        
        logger.info(f"\nPredicted (high-quality model) vs GALAH:")
        logger.info(f"  MAE:  {pred_mae:.1f} K")
        logger.info(f"  RMSE: {pred_rmse:.1f} K")
        logger.info(f"  Median residual: {pred_median:.1f} K")
        
        # Improvement
        mae_improvement = ((gaia_mae - pred_mae) / gaia_mae) * 100
        rmse_improvement = ((gaia_rmse - pred_rmse) / gaia_rmse) * 100
        median_improvement = ((abs(gaia_median) - abs(pred_median)) / abs(gaia_median)) * 100
        
        logger.info(f"\nImprovement:")
        logger.info(f"  MAE:  {mae_improvement:+.1f}%")
        logger.info(f"  RMSE: {rmse_improvement:+.1f}%")
        logger.info(f"  Median residual: {median_improvement:+.1f}%")
        
        if mae_improvement > 0:
            logger.info(f"\n✓ Predictions IMPROVED temperature estimates!")
    
    # Save
    logger.info(f"\nSaving to {output_file}...")
    matched_clean.to_parquet(output_file, index=False)
    
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Sources: {len(matched_clean):,}")
    
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-MATCH COMPLETED!")
    logger.info("=" * 70)
    
    return matched_clean


def main():
    logger = logging.getLogger(__name__)
    config = get_config()

    input_file = config.project_root / 'data' / 'processed' / 'flag0_temperature_predictions.parquet'
    output_file = config.project_root / 'data' / 'processed' / 'galah_validated_predictions.parquet'

    results = crossmatch_galah_xmatch(input_file, output_file)

    if len(results) > 0:
        logger.info(f"\n✓ Cross-match completed: {len(results):,} sources validated with GALAH")


if __name__ == '__main__':
    main()

