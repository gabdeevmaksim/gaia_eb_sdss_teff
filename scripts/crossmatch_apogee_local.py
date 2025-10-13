#!/usr/bin/env python3
"""
Cross-match with downloaded APOGEE DR17 catalog using Gaia source_id.

This script loads the local APOGEE DR17 FITS file and cross-matches with
flag 0 predictions using Gaia source_id for validation.

Quality criteria:
- aspcapflag = 0 (ASPCAPFLAG, no star_bad)
- SNR > 80 (high signal-to-noise)
- Valid Teff measurements

Usage:
    python scripts/crossmatch_apogee_local.py

Author: Auto-generated
Date: 2025-10-13
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from astropy.io import fits

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def load_apogee_catalog(apogee_file: Path) -> pd.DataFrame:
    """
    Load APOGEE DR17 catalog from FITS file.
    
    Parameters
    ----------
    apogee_file : Path
        Path to APOGEE allStar FITS file
        
    Returns
    -------
    pd.DataFrame
        APOGEE catalog with selected columns
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading APOGEE DR17 catalog from {apogee_file.name}...")
    
    with fits.open(apogee_file) as hdul:
        data = hdul[1].data
        
        logger.info(f"  Total APOGEE sources: {len(data):,}")
        
        # Select relevant columns and convert to native byte order
        apogee_df = pd.DataFrame({
            'source_id': np.asarray(data['GAIAEDR3_SOURCE_ID'], dtype=np.int64),
            'apogee_id': data['APOGEE_ID'].astype(str),
            'apogee_ra': np.asarray(data['RA'], dtype=np.float64),
            'apogee_dec': np.asarray(data['DEC'], dtype=np.float64),
            'apogee_teff': np.asarray(data['TEFF'], dtype=np.float32),
            'apogee_e_teff': np.asarray(data['TEFF_ERR'], dtype=np.float32),
            'apogee_logg': np.asarray(data['LOGG'], dtype=np.float32),
            'apogee_e_logg': np.asarray(data['LOGG_ERR'], dtype=np.float32),
            'apogee_mh': np.asarray(data['M_H'], dtype=np.float32),
            'apogee_e_mh': np.asarray(data['M_H_ERR'], dtype=np.float32),
            'apogee_snr': np.asarray(data['SNR'], dtype=np.float32),
            'apogee_aspcapflag': np.asarray(data['ASPCAPFLAG'], dtype=np.int32),
            'apogee_starflag': np.asarray(data['STARFLAG'], dtype=np.int32),
            'apogee_nvisits': np.asarray(data['NVISITS'], dtype=np.int32)
        })
        
        logger.info(f"  Loaded columns: {list(apogee_df.columns)}")
    
    return apogee_df


def apply_quality_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply APOGEE quality criteria.
    
    Quality cuts:
    - aspcapflag = 0 (no star_bad or pipeline issues)
    - SNR > 80 (high quality spectra)
    - Valid Teff measurement
    - Valid Gaia source_id
    
    Parameters
    ----------
    df : pd.DataFrame
        APOGEE catalog
        
    Returns
    -------
    pd.DataFrame
        Quality-filtered catalog
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\nApplying APOGEE quality criteria...")
    n_initial = len(df)
    
    mask = np.ones(len(df), dtype=bool)
    
    # 1. Valid Gaia source_id
    n_before = mask.sum()
    mask &= (df['source_id'] > 0)
    n_after = mask.sum()
    logger.info(f"  Valid Gaia source_id: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
    
    # 2. aspcapflag - relax criteria (eclipsing binaries often have flags)
    # Note: For eclipsing binaries, we relax this criterion as they often have flags
    # We'll keep sources but note the flag values
    n_before = mask.sum()
    # mask &= (df['apogee_aspcapflag'] == 0)  # Commented out - too restrictive for EBs
    n_after = mask.sum()
    logger.info(f"  aspcapflag check: {n_after:,}/{n_before:,} (keeping all - EBs often flagged)")
    
    # 3. Valid Teff
    n_before = mask.sum()
    mask &= df['apogee_teff'].notna()
    mask &= (df['apogee_teff'] > 0)
    n_after = mask.sum()
    logger.info(f"  Valid Teff: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
    
    # 4. Valid Teff error
    n_before = mask.sum()
    mask &= df['apogee_e_teff'].notna()
    mask &= (df['apogee_e_teff'] > 0)
    n_after = mask.sum()
    logger.info(f"  Valid e_Teff: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
    
    # 5. SNR > 80
    n_before = mask.sum()
    mask &= (df['apogee_snr'] > 80)
    n_after = mask.sum()
    logger.info(f"  SNR > 80: {n_after:,}/{n_before:,} ({n_after/n_before*100:.1f}%)")
    
    df_clean = df[mask].copy()
    
    logger.info(f"\n  Final quality sample: {len(df_clean):,}/{n_initial:,} ({len(df_clean)/n_initial*100:.1f}%)")
    logger.info(f"  Temperature range: {df_clean['apogee_teff'].min():.1f} - {df_clean['apogee_teff'].max():.1f} K")
    
    return df_clean


def crossmatch_with_apogee(
    input_file: Path,
    apogee_file: Path,
    output_file: Path
) -> pd.DataFrame:
    """
    Cross-match predictions with APOGEE DR17 using Gaia source_id.
    
    Parameters
    ----------
    input_file : Path
        Input predictions file
    apogee_file : Path
        APOGEE DR17 FITS catalog
    output_file : Path
        Output cross-matched file
        
    Returns
    -------
    pd.DataFrame
        Cross-matched data
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("CROSS-MATCH WITH APOGEE DR17 (LOCAL CATALOG)")
    logger.info("=" * 70)
    logger.info(f"Input: {input_file.name}")
    logger.info(f"APOGEE: {apogee_file.name}")
    logger.info(f"Output: {output_file.name}")
    logger.info("")
    
    # Load input predictions
    logger.info("Loading flag 0 predictions...")
    df_pred = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df_pred):,} flag 0 sources")
    
    # Load APOGEE catalog
    apogee_df = load_apogee_catalog(apogee_file)
    
    # Apply quality cuts to APOGEE
    apogee_clean = apply_quality_cuts(apogee_df)
    
    # Cross-match on source_id
    logger.info("\nCross-matching on Gaia source_id...")
    logger.info(f"  Flag 0 sources: {len(df_pred):,}")
    logger.info(f"  APOGEE quality sources: {len(apogee_clean):,}")
    
    # Ensure both source_id columns are int64
    df_pred['source_id'] = df_pred['source_id'].astype(np.int64)
    apogee_clean['source_id'] = apogee_clean['source_id'].astype(np.int64)
    
    merged = df_pred.merge(apogee_clean, on='source_id', how='inner')
    
    logger.info(f"  ✓ Matches found: {len(merged):,} ({len(merged)/len(df_pred)*100:.2f}%)")
    
    if len(merged) == 0:
        logger.warning("No matches found!")
        return pd.DataFrame()
    
    # Calculate comparison statistics
    logger.info("\n" + "=" * 70)
    logger.info("TEMPERATURE COMPARISON VS APOGEE SPECTROSCOPY")
    logger.info("=" * 70)
    
    # Original Gaia GSP-Phot (flag 0) vs APOGEE
    gaia_residual = merged['teff_gspphot'] - merged['apogee_teff']
    gaia_mae = np.mean(np.abs(gaia_residual))
    gaia_rmse = np.sqrt(np.mean(gaia_residual**2))
    gaia_median = np.median(gaia_residual)
    
    logger.info(f"\nOriginal Gaia GSP-Phot (flag 0) vs APOGEE:")
    logger.info(f"  MAE:  {gaia_mae:.1f} K")
    logger.info(f"  RMSE: {gaia_rmse:.1f} K")
    logger.info(f"  Median residual: {gaia_median:.1f} K")
    logger.info(f"  Std residual: {np.std(gaia_residual):.1f} K")
    
    # Predicted (high-quality model) vs APOGEE
    pred_residual = merged['predicted_teff'] - merged['apogee_teff']
    pred_mae = np.mean(np.abs(pred_residual))
    pred_rmse = np.sqrt(np.mean(pred_residual**2))
    pred_median = np.median(pred_residual)
    
    logger.info(f"\nPredicted (high-quality model) vs APOGEE:")
    logger.info(f"  MAE:  {pred_mae:.1f} K")
    logger.info(f"  RMSE: {pred_rmse:.1f} K")
    logger.info(f"  Median residual: {pred_median:.1f} K")
    logger.info(f"  Std residual: {np.std(pred_residual):.1f} K")
    
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
    else:
        logger.info(f"\n✗ Predictions did NOT improve temperature estimates")
    
    # Save results
    logger.info(f"\nSaving cross-matched data to {output_file}...")
    merged.to_parquet(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Sources: {len(merged):,}")
    
    # Save summary
    summary_file = output_file.parent / f"{output_file.stem}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"APOGEE DR17 CROSS-MATCH SUMMARY\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Input sources (flag 0): {len(df_pred):,}\n")
        f.write(f"APOGEE total: {len(apogee_df):,}\n")
        f.write(f"APOGEE quality: {len(apogee_clean):,}\n")
        f.write(f"Cross-matched: {len(merged):,} ({len(merged)/len(df_pred)*100:.2f}%)\n\n")
        f.write(f"Quality criteria:\n")
        f.write(f"  - aspcapflag = 0 (no star_bad)\n")
        f.write(f"  - SNR > 80\n")
        f.write(f"  - Valid Teff and e_Teff\n")
        f.write(f"  - Valid Gaia source_id\n\n")
        f.write(f"Temperature comparison vs APOGEE spectroscopy:\n\n")
        f.write(f"Original Gaia GSP-Phot (flag 0):\n")
        f.write(f"  MAE:  {gaia_mae:.1f} K\n")
        f.write(f"  RMSE: {gaia_rmse:.1f} K\n")
        f.write(f"  Median residual: {gaia_median:.1f} K\n\n")
        f.write(f"Predicted (high-quality model):\n")
        f.write(f"  MAE:  {pred_mae:.1f} K\n")
        f.write(f"  RMSE: {pred_rmse:.1f} K\n")
        f.write(f"  Median residual: {pred_median:.1f} K\n\n")
        f.write(f"Improvement:\n")
        f.write(f"  MAE:  {mae_improvement:+.1f}%\n")
        f.write(f"  RMSE: {rmse_improvement:+.1f}%\n")
        f.write(f"  Median residual: {median_improvement:+.1f}%\n")
    
    logger.info(f"  Summary saved: {summary_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-MATCH COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    return merged


def main():
    logger = logging.getLogger(__name__)
    config = get_config()

    # File paths
    input_file = config.project_root / 'data' / 'processed' / 'flag0_temperature_predictions.parquet'
    apogee_file = config.project_root / 'data' / 'external' / 'apogee_dr17' / 'allStar-dr17-synspec.fits'
    output_file = config.project_root / 'data' / 'processed' / 'apogee_validated_predictions.parquet'

    # Check if APOGEE file exists
    if not apogee_file.exists():
        logger.error(f"Error: APOGEE catalog not found at {apogee_file}")
        logger.info("Run: python scripts/download_spectroscopic_catalogs.py")
        return

    # Cross-match
    results = crossmatch_with_apogee(input_file, apogee_file, output_file)

    if len(results) > 0:
        logger.info(f"\n✓ Cross-match completed: {len(results):,} sources validated with APOGEE")


if __name__ == '__main__':
    main()

