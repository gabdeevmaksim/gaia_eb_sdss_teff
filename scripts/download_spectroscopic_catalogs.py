#!/usr/bin/env python3
"""
Download APOGEE DR17 and GALAH DR4 catalogs for cross-matching.

This script downloads the full spectroscopic catalogs with Gaia source_id
for efficient local cross-matching.

Usage:
    python scripts/download_spectroscopic_catalogs.py

Author: Auto-generated
Date: 2025-10-13
"""

import sys
from pathlib import Path
import logging
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Downloading {description}...")
    logger.info(f"  URL: {url}")
    logger.info(f"  Output: {output_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  Downloaded: {file_size_mb:.1f} MB")
    logger.info(f"  Saved to: {output_path}")
    
    return output_path


def download_apogee_dr17():
    """Download APOGEE DR17 allStar catalog."""
    logger = logging.getLogger(__name__)
    config = get_config()
    
    logger.info("=" * 70)
    logger.info("DOWNLOADING APOGEE DR17")
    logger.info("=" * 70)
    
    # APOGEE DR17 allStar catalog
    url = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/allStar-dr17-synspec.fits"
    output_dir = config.project_root / 'data' / 'external' / 'apogee_dr17'
    output_file = output_dir / 'allStar-dr17-synspec.fits'
    
    if output_file.exists():
        logger.info(f"File already exists: {output_file}")
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        return output_file
    
    try:
        download_file(url, output_file, "APOGEE DR17 allStar")
        return output_file
    except Exception as e:
        logger.error(f"Failed to download APOGEE DR17: {e}")
        return None


def download_galah_dr3():
    """Download GALAH DR3 main catalog with spectroscopic parameters."""
    logger = logging.getLogger(__name__)
    config = get_config()
    
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOADING GALAH DR3 MAIN CATALOG")
    logger.info("=" * 70)
    
    # GALAH DR3 main catalog with spectroscopic parameters
    # Contains teff, logg, fe_h, and dr3_source_id
    url = "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_main_star_v2.fits"
    output_dir = config.project_root / 'data' / 'external' / 'galah_dr3'
    output_file = output_dir / 'GALAH_DR3_main_star_v2.fits'
    
    if output_file.exists():
        logger.info(f"File already exists: {output_file}")
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"  Size: {file_size_mb:.1f} MB")
        return output_file
    
    try:
        download_file(url, output_file, "GALAH DR3 main catalog")
        return output_file
    except Exception as e:
        logger.error(f"Failed to download GALAH DR3: {e}")
        logger.info("  You may need to download manually from: https://www.galah-survey.org/dr3/the_catalogues")
        return None


def inspect_catalogs():
    """Inspect downloaded catalogs to check structure."""
    logger = logging.getLogger(__name__)
    config = get_config()
    
    logger.info("\n" + "=" * 70)
    logger.info("INSPECTING CATALOGS")
    logger.info("=" * 70)
    
    from astropy.io import fits
    
    # Check APOGEE
    apogee_file = config.project_root / 'data' / 'external' / 'apogee_dr17' / 'allStar-dr17-synspec.fits'
    if apogee_file.exists():
        logger.info("\nAPOGEE DR17 structure:")
        with fits.open(apogee_file) as hdul:
            hdul.info()
            logger.info(f"\n  Columns in main table:")
            for col in hdul[1].columns[:20]:
                logger.info(f"    {col.name}: {col.format}")
            
            # Check for Gaia column
            gaia_cols = [col.name for col in hdul[1].columns if 'gaia' in col.name.lower()]
            if gaia_cols:
                logger.info(f"\n  ✓ Gaia columns found: {gaia_cols}")
            
            # Check temperature column
            temp_cols = [col.name for col in hdul[1].columns if 'teff' in col.name.lower()]
            if temp_cols:
                logger.info(f"  ✓ Temperature columns found: {temp_cols}")
            
            logger.info(f"\n  Total sources: {len(hdul[1].data):,}")
    
    # Check GALAH
    galah_file = config.project_root / 'data' / 'external' / 'galah_dr3' / 'GALAH_DR3_main_star_v2.fits'
    if galah_file.exists():
        logger.info("\n\nGALAH DR3 structure:")
        with fits.open(galah_file) as hdul:
            hdul.info()
            logger.info(f"\n  Columns in main table:")
            for col in hdul[1].columns[:20]:
                logger.info(f"    {col.name}: {col.format}")
            
            # Check for Gaia column
            gaia_cols = [col.name for col in hdul[1].columns if 'gaia' in col.name.lower() or 'dr3' in col.name.lower()]
            if gaia_cols:
                logger.info(f"\n  ✓ Gaia columns found: {gaia_cols}")
            
            # Check temperature column
            temp_cols = [col.name for col in hdul[1].columns if 'teff' in col.name.lower()]
            if temp_cols:
                logger.info(f"  ✓ Temperature columns found: {temp_cols}")
            
            logger.info(f"\n  Total sources: {len(hdul[1].data):,}")


def main():
    logger = logging.getLogger(__name__)

    # Download catalogs
    apogee_file = download_apogee_dr17()
    galah_file = download_galah_dr3()

    # Inspect if successful
    if apogee_file or galah_file:
        inspect_catalogs()

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)

    if apogee_file:
        logger.info(f"\n✓ APOGEE DR17: {apogee_file}")
    else:
        logger.warning(f"\n✗ APOGEE DR17: Download failed")

    if galah_file:
        logger.info(f"✓ GALAH DR3: {galah_file}")
    else:
        logger.warning(f"✗ GALAH DR3: Download failed or needs manual download")


if __name__ == '__main__':
    main()

