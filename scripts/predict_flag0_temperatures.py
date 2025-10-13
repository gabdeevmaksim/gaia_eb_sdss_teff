#!/usr/bin/env python3
"""
Predict temperatures for flag 0 objects using the high-quality model.

This script loads the high-quality Random Forest model (trained on flag 1 sources)
and uses it to predict temperatures for flag 0 sources (lower quality).

Usage:
    # Predict temperatures for flag 0 sources
    python scripts/predict_flag0_temperatures.py

    # Specify model and input files
    python scripts/predict_flag0_temperatures.py --model models/rf_temperature_regressor_high_quality_20251013_105249.pkl

Author: Auto-generated
Date: 2025-10-13
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import get_config
from src.features.engineering import engineer_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def predict_flag0_temperatures(
    model_file: Path,
    input_file: Path,
    output_file: Path
) -> pd.DataFrame:
    """
    Predict temperatures for flag 0 sources using high-quality model.
    
    Parameters
    ----------
    model_file : Path
        Path to trained high-quality model
    input_file : Path
        Path to dataset with quality flags
    output_file : Path
        Path to save predictions
        
    Returns
    -------
    pd.DataFrame
        Flag 0 sources with predicted temperatures
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("PREDICT TEMPERATURES FOR FLAG 0 SOURCES")
    logger.info("=" * 70)
    logger.info(f"Model: {model_file.name}")
    logger.info(f"Input: {input_file.name}")
    logger.info("")
    
    # Load model and feature selector
    logger.info("Loading trained model...")
    model = joblib.load(model_file)
    
    # Load selector (should be in same directory with _selector suffix)
    selector_file = model_file.parent / (model_file.stem + '_selector.pkl')
    if selector_file.exists():
        selector = joblib.load(selector_file)
        logger.info(f"  Loaded feature selector: {selector_file.name}")
    else:
        logger.error(f"  Feature selector not found: {selector_file}")
        raise FileNotFoundError(f"Selector file not found: {selector_file}")
    
    # Load data with quality flags
    logger.info("Loading dataset with quality flags...")
    df = pd.read_parquet(input_file)
    logger.info(f"  Loaded {len(df):,} objects")
    
    # Filter for flag 0 sources only
    logger.info("Filtering for flag 0 sources...")
    df_flag0 = df[df['teff_gspphot_flag'] == 0].copy()
    logger.info(f"  Flag 0 sources: {len(df_flag0):,}")
    
    if len(df_flag0) == 0:
        logger.error("No flag 0 sources found!")
        return pd.DataFrame()
    
    # Define color columns (same as training - only color features, no magnitude)
    exclude_cols = ['original_ext_source_id', 'source_id', 'teff_gspphot', 'teff_gspphot_flag']
    magnitude_cols = ['gPSFMag', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
    color_cols = [col for col in df_flag0.columns if col not in exclude_cols and col not in magnitude_cols]
    logger.info(f"  Color features: {color_cols}")
    logger.info(f"  Excluded magnitude features: {magnitude_cols}")
    
    # Feature engineering (same as training)
    logger.info("Engineering features...")
    df_features = engineer_all_features(
        df_flag0,
        color_cols=color_cols,
        mag_cols=None,  # No magnitude features
        include_polynomials=True,
        include_interactions=True,
        include_log=True,
        include_temp_dependent=True,
        include_mag_features=False
    )
    
    logger.info(f"  Features created: {df_features.shape[1]}")
    
    # Prepare features - only use features that the model was trained on
    logger.info("Filtering features to match training features...")
    trained_features = selector.feature_names_in_
    available_features = [col for col in df_features.columns if col in trained_features]
    missing_features = [col for col in trained_features if col not in df_features.columns]
    
    logger.info(f"  Available features: {len(available_features)}")
    logger.info(f"  Missing features: {len(missing_features)}")
    if missing_features:
        logger.warning(f"  Missing features: {missing_features[:5]}...")
    
    # Create feature matrix with only trained features
    X = df_features[available_features]
    
    # Clean features (same as training)
    logger.info("Cleaning features...")
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    logger.info(f"  Feature matrix shape: {X.shape}")
    
    # Select features using trained selector
    logger.info("Selecting features using trained selector...")
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    logger.info(f"  Selected {len(selected_features)} features")
    logger.info(f"  Selected features: {selected_features}")
    
    # Make predictions
    logger.info("Making temperature predictions...")
    start_time = datetime.now()
    predictions = model.predict(X_selected)
    prediction_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"  Predictions completed in {prediction_time:.1f} seconds")
    logger.info(f"  Predicted temperature range: {predictions.min():.1f} - {predictions.max():.1f} K")
    
    # Create results dataframe
    results = df_flag0.copy()
    results['predicted_teff'] = predictions
    results['prediction_residual'] = results['teff_gspphot'] - results['predicted_teff']
    results['prediction_error_percent'] = (np.abs(results['prediction_residual']) / results['teff_gspphot']) * 100
    
    # Calculate statistics
    mae = np.mean(np.abs(results['prediction_residual']))
    rmse = np.sqrt(np.mean(results['prediction_residual']**2))
    r2 = 1 - (np.sum(results['prediction_residual']**2) / np.sum((results['teff_gspphot'] - results['teff_gspphot'].mean())**2))
    mean_error_percent = results['prediction_error_percent'].mean()
    
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Sources predicted: {len(results):,}")
    logger.info(f"MAE: {mae:.1f} K")
    logger.info(f"RMSE: {rmse:.1f} K")
    logger.info(f"R²: {r2:.3f}")
    logger.info(f"Mean error: {mean_error_percent:.2f}%")
    logger.info(f"Prediction time: {prediction_time:.1f} seconds")
    
    # Save results
    logger.info(f"\nSaving predictions to {output_file}...")
    results.to_parquet(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Columns: {list(results.columns)}")
    
    # Create summary
    summary_file = output_file.parent / f"{output_file.stem}_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"FLAG 0 TEMPERATURE PREDICTIONS\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Model used: {model_file.name}\n")
        f.write(f"Sources predicted: {len(results):,}\n")
        f.write(f"MAE: {mae:.1f} K\n")
        f.write(f"RMSE: {rmse:.1f} K\n")
        f.write(f"R²: {r2:.3f}\n")
        f.write(f"Mean error: {mean_error_percent:.2f}%\n")
        f.write(f"Prediction time: {prediction_time:.1f} seconds\n\n")
        f.write(f"Predicted temperature range: {predictions.min():.1f} - {predictions.max():.1f} K\n")
        f.write(f"Original temperature range: {results['teff_gspphot'].min():.1f} - {results['teff_gspphot'].max():.1f} K\n")
    
    logger.info(f"  Summary saved: {summary_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict temperatures for flag 0 sources using high-quality model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: latest high-quality model)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input dataset with quality flags (default: ml_training_data_with_gaia_with_flags.parquet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for predictions (default: flag0_temperature_predictions.parquet)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Determine model file
    if args.model:
        model_file = Path(args.model)
    else:
        # Find latest high-quality model (exclude selector files)
        models_dir = config.get_path('models')
        high_quality_models = [f for f in models_dir.glob('rf_temperature_regressor_high_quality_*.pkl') 
                              if not f.name.endswith('_selector.pkl')]
        if not high_quality_models:
            raise FileNotFoundError("No high-quality models found in models/ directory")
        model_file = max(high_quality_models, key=lambda x: x.stat().st_mtime)
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = config.project_root / 'data' / 'processed' / 'ml_training_data_with_gaia_with_flags.parquet'
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.project_root / 'data' / 'processed' / 'flag0_temperature_predictions.parquet'
    
    # Predict temperatures
    results = predict_flag0_temperatures(model_file, input_file, output_file)
    
    logger = logging.getLogger(__name__)
    logger.info(f"\nPrediction completed!")
    logger.info(f"Flag 0 sources predicted: {len(results):,}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
