#!/usr/bin/env python3
"""
Query IRSA 2MASS catalog for eclipsing binaries by exact designation.

This script uses IRSA's TAP service with pyvo (recommended by IRSA) to query
the 2MASS Point Source Catalog by exact designation match.

This script:
1. Loads 2MASS source IDs from eb_with_tmass_ids-result.ecsv
2. Queries IRSA TAP service for exact matches by designation
3. Retrieves J, H, K photometry and quality flags
4. Saves results to processed data directory

Usage:
    python scripts/query_2mass_irsa.py [--max-sources N] [--batch-size N]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from typing import List, Optional

import pandas as pd
import pyvo
from astropy.table import Table

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_2mass_designation(tmass_id: str) -> str:
    """
    Format 2MASS identifier to standard designation with J prefix.
    
    Parameters
    ----------
    tmass_id : str
        2MASS identifier (e.g., '02582808+0047586' or 'J02582808+0047586')
    
    Returns
    -------
    str
        Formatted designation (e.g., 'J02582808+0047586')
    """
    if not tmass_id.startswith('J'):
        return 'J' + tmass_id
    return tmass_id


def query_2mass_tap(
    source_ids: List[str],
    batch_size: int = 10000
) -> pd.DataFrame:
    """
    Query IRSA 2MASS catalog using TAP service by exact designation.
    
    Uses IRSA's TAP (Table Access Protocol) service to query the 2MASS Point
    Source Catalog. This is the recommended method by IRSA for programmatic access.
    
    Parameters
    ----------
    source_ids : list of str
        2MASS source identifiers
    batch_size : int
        Number of sources to query per batch (default: 10000)
        TAP async queries can handle large batches efficiently.
        Use 50000+ for faster execution on large datasets.
    
    Returns
    -------
    pd.DataFrame
        Combined results with 2MASS photometry
    """
    # Initialize TAP service
    tap_url = 'https://irsa.ipac.caltech.edu/TAP'
    service = pyvo.dal.TAPService(tap_url)
    
    all_results = []
    total = len(source_ids)
    
    logger.info(f"Querying {total} sources from IRSA 2MASS catalog using TAP service...")
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = source_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} sources)...")
        
        try:
            # Use IDs without 'J' prefix (catalog stores them without prefix)
            # Remove 'J' prefix if present
            designations = [sid.replace('J', '', 1) if sid.startswith('J') else sid for sid in batch]
            
            # Create SQL query with IN clause for batch
            designation_list = "','".join(designations)
            
            query = f"""
                SELECT 
                    designation,
                    ra, dec,
                    j_m, j_msigcom, j_snr,
                    h_m, h_msigcom, h_snr,
                    k_m, k_msigcom, k_snr,
                    ph_qual, cc_flg, gal_contam
                FROM fp_psc
                WHERE designation IN ('{designation_list}')
            """
            
            # Execute query
            result = service.run_async(query)
            
            if len(result) > 0:
                # Convert to pandas
                result_df = result.to_table().to_pandas()
                
                # The designation is already in the format we want (without J prefix)
                result_df['original_ext_source_id'] = result_df['designation']
                
                all_results.append(result_df)
                logger.info(f"  Found {len(result_df)} matches in this batch")
            else:
                logger.warning(f"  No matches found in batch {batch_num}")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"Successfully queried {len(combined)} total sources")
        return combined
    else:
        logger.warning("No results found")
        return pd.DataFrame()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Query IRSA 2MASS catalog using TAP service by exact designation'
    )
    parser.add_argument(
        '--max-sources',
        type=int,
        help='Maximum number of sources to query (for testing)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Number of sources per batch query (default: 10000, can use 50000+ for faster queries)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Load 2MASS cross-match file
    input_file = config.project_root / 'data' / 'raw' / 'eb_with_tmass_ids-result.ecsv'
    logger.info(f"Loading 2MASS IDs from: {input_file}")
    
    # Read ECSV file
    data = Table.read(input_file, format='ascii.ecsv')
    logger.info(f"Loaded {len(data)} sources with 2MASS IDs")
    
    # Extract 2MASS IDs
    tmass_ids = data['original_ext_source_id'].tolist()
    
    # Limit sources if specified
    if args.max_sources:
        tmass_ids = tmass_ids[:args.max_sources]
        logger.info(f"Limited to first {args.max_sources} sources for testing")
    
    # Query IRSA using TAP
    results = query_2mass_tap(
        tmass_ids,
        batch_size=args.batch_size
    )
    
    if len(results) > 0:
        # Add Gaia source_id by merging back
        data_df = data.to_pandas()
        results = results.merge(
            data_df[['source_id', 'original_ext_source_id']],
            on='original_ext_source_id',
            how='left'
        )
        
        # Save results
        output_file = config.get_path('processed', ensure_exists=True) / 'eb_2mass_photometry.parquet'
        results.to_parquet(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")
        
        # Print summary
        logger.info("\n=== Query Summary ===")
        logger.info(f"Total sources queried: {len(tmass_ids) if not args.max_sources else args.max_sources}")
        logger.info(f"Successful matches: {len(results)}")
        logger.info(f"Match rate: {100*len(results)/len(tmass_ids):.1f}%")
        
        # Print column info
        logger.info(f"\nColumns in output: {len(results.columns)}")
        logger.info(f"Key photometry columns:")
        for col in ['j_m', 'h_m', 'k_m', 'j_msigcom', 'h_msigcom', 'k_msigcom']:
            if col in results.columns:
                logger.info(f"  - {col}")
    else:
        logger.error("No results to save")


if __name__ == '__main__':
    main()

