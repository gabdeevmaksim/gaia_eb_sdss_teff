#!/usr/bin/env python3
"""
Master pipeline orchestrator for eclipsing binary temperature analysis.

This script provides a command-line interface to run different pipelines:
- Data processing pipeline
- ML training pipeline
- Complete end-to-end pipeline

Usage:
    # Run complete pipeline
    python pipeline.py --all

    # Run data processing only
    python pipeline.py --data

    # Run ML training only
    python pipeline.py --ml

    # Dry run (show what would be executed)
    python pipeline.py --all --dry-run

    # Custom model parameters
    python pipeline.py --ml --n-estimators 500 --max-depth 25
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Import pipelines
from src.pipeline import DataProcessingPipeline, MLTrainingPipeline
from src.config import get_config


def run_data_pipeline(dry_run=False):
    """Run the data processing pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute Data Processing Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Convert ECSV to Parquet")
        logger.info("    2. Extract Pan-STARRS Duplicates")
        logger.info("    3. Merge Duplicates")
        logger.info("    4. Clean Photometry")
        logger.info("    5. Calculate Temperatures")
        return {}

    pipeline = DataProcessingPipeline()
    context = pipeline.run()
    return context


def run_ml_pipeline(n_estimators=None, max_depth=None, dry_run=False):
    """Run the ML training pipeline."""
    if dry_run:
        logger.info("DRY RUN: Would execute ML Training Pipeline")
        logger.info("  Steps:")
        logger.info("    1. Load ML Data")
        logger.info("    2. Engineer Features")
        logger.info("    3. Prepare Train/Test Split")
        logger.info("    4. Train Model")
        logger.info("    5. Evaluate Performance")
        logger.info("    6. Save Model")
        if n_estimators:
            logger.info(f"  Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        return {}

    pipeline = MLTrainingPipeline(n_estimators=n_estimators, max_depth=max_depth)
    context = pipeline.run()
    return context


def run_complete_pipeline(n_estimators=None, max_depth=None, dry_run=False):
    """Run complete end-to-end pipeline."""
    logger.info("=" * 70)
    logger.info("COMPLETE PIPELINE: Data Processing + ML Training")
    logger.info("=" * 70)
    logger.info("")

    # Run data processing
    logger.info("STAGE 1: Data Processing")
    logger.info("-" * 70)
    data_context = run_data_pipeline(dry_run=dry_run)

    if not dry_run:
        logger.info("")
        logger.info("Data processing complete!")
        logger.info("")

    # Run ML training
    logger.info("STAGE 2: ML Training")
    logger.info("-" * 70)
    ml_context = run_ml_pipeline(n_estimators=n_estimators, max_depth=max_depth, dry_run=dry_run)

    if not dry_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("COMPLETE PIPELINE FINISHED")
        logger.info("=" * 70)

        if 'model_id' in ml_context:
            logger.info(f"✓ Model saved: {ml_context['model_id']}")
            logger.info(f"  MAE: {ml_context['metrics']['mae']:.0f} K")
            logger.info(f"  R²: {ml_context['metrics']['r2']:.4f}")

    return {**data_context, **ml_context}


def main():
    parser = argparse.ArgumentParser(
        description='Eclipsing Binary Temperature Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py --all

  # Run data processing only
  python pipeline.py --data

  # Run ML training only
  python pipeline.py --ml

  # Custom model parameters
  python pipeline.py --ml --n-estimators 500 --max-depth 25

  # Dry run (see what would be executed)
  python pipeline.py --all --dry-run
        """
    )

    # Pipeline selection
    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (data processing + ML training)'
    )
    pipeline_group.add_argument(
        '--data',
        action='store_true',
        help='Run data processing pipeline only'
    )
    pipeline_group.add_argument(
        '--ml',
        action='store_true',
        help='Run ML training pipeline only'
    )

    # Model parameters
    parser.add_argument(
        '--n-estimators',
        type=int,
        help='Number of trees in Random Forest (default: from config)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        help='Maximum depth of trees (default: from config)'
    )

    # Options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show configuration
    config = get_config()
    logger.info("Configuration loaded:")
    logger.info(f"  Project root: {config.project_root}")
    logger.info("")

    # Run selected pipeline
    try:
        if args.all:
            run_complete_pipeline(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                dry_run=args.dry_run
            )

        elif args.data:
            run_data_pipeline(dry_run=args.dry_run)

        elif args.ml:
            run_ml_pipeline(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                dry_run=args.dry_run
            )

        if not args.dry_run:
            logger.info("")
            logger.info("✓ Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
