#!/usr/bin/env python3
"""
Create validation plots for Ensemble PanSTARRS-Unified model.

This script creates residual plots and performance visualizations for the
ensemble model that combines PanSTARRS Basic and Unified Color-Only predictions.

Generated plots:
1. test_scatter - Predicted vs True with 1:1 and ±10% lines
2. residuals - Residual analysis (2 subplots)
3. performance_by_temp - MAE, RMSE, % error, Within 10% by temperature bins
4. component_comparison - Compare individual models vs ensemble
5. residuals_by_component - Residual comparison for each model

Author: Claude Code
Date: 2025-10-21
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.visualization.validation_plots import (
    plot_test_scatter,
    plot_residuals,
    plot_performance_by_temp,
    calculate_bin_statistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def plot_component_comparison(df_ensemble, ensemble_id, subdir, model_name):
    """Plot comparison of individual models vs ensemble."""

    logger.info("   Creating component comparison plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Plot 1: MAE by temperature range
    temp_bins = [0, 4000, 5000, 6000, 7000, 8000, 100000]
    temp_labels = ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-8000K', '>8000K']

    df_ensemble['temp_bin'] = pd.cut(df_ensemble['true_temperature'],
                                      bins=temp_bins, labels=temp_labels)

    mae_data = []
    for label in temp_labels:
        mask = df_ensemble['temp_bin'] == label
        if mask.sum() > 0:
            mae_ps = np.abs(df_ensemble.loc[mask, 'pred_panstarrs'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae_uni = np.abs(df_ensemble.loc[mask, 'pred_unified'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae_ens = np.abs(df_ensemble.loc[mask, 'predicted_temperature'] - df_ensemble.loc[mask, 'true_temperature']).mean()
            mae_data.append({'Temp Range': label, 'PanSTARRS': mae_ps, 'Unified': mae_uni, 'Ensemble': mae_ens})

    mae_df = pd.DataFrame(mae_data)
    x = np.arange(len(mae_df))
    width = 0.25

    axes[0].bar(x - width, mae_df['PanSTARRS'], width, label='PanSTARRS Basic', alpha=0.8)
    axes[0].bar(x, mae_df['Unified'], width, label='Unified Color-Only', alpha=0.8)
    axes[0].bar(x + width, mae_df['Ensemble'], width, label='Ensemble', alpha=0.8, color='green')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mae_df['Temp Range'], rotation=45, ha='right')
    axes[0].set_ylabel('MAE (K)', fontsize=12)
    axes[0].set_title('MAE by Temperature Range', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Scatter - PanSTARRS vs Unified predictions
    axes[1].hexbin(df_ensemble['pred_panstarrs'], df_ensemble['pred_unified'],
                   gridsize=50, cmap='viridis', mincnt=1, bins='log')
    axes[1].plot([3000, 10000], [3000, 10000], 'r--', alpha=0.5, label='1:1 line')
    axes[1].set_xlabel('PanSTARRS Prediction (K)', fontsize=12)
    axes[1].set_ylabel('Unified Prediction (K)', fontsize=12)
    axes[1].set_title('Model Predictions Comparison', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot 3: Within 10% accuracy by temperature range
    acc_data = []
    for label in temp_labels:
        mask = df_ensemble['temp_bin'] == label
        if mask.sum() > 0:
            n = mask.sum()
            err_ps = np.abs(df_ensemble.loc[mask, 'pred_panstarrs'] - df_ensemble.loc[mask, 'true_temperature']) / df_ensemble.loc[mask, 'true_temperature'] * 100
            err_uni = np.abs(df_ensemble.loc[mask, 'pred_unified'] - df_ensemble.loc[mask, 'true_temperature']) / df_ensemble.loc[mask, 'true_temperature'] * 100
            err_ens = np.abs(df_ensemble.loc[mask, 'predicted_temperature'] - df_ensemble.loc[mask, 'true_temperature']) / df_ensemble.loc[mask, 'true_temperature'] * 100

            acc_ps = (err_ps <= 10).sum() / n * 100
            acc_uni = (err_uni <= 10).sum() / n * 100
            acc_ens = (err_ens <= 10).sum() / n * 100
            acc_data.append({'Temp Range': label, 'PanSTARRS': acc_ps, 'Unified': acc_uni, 'Ensemble': acc_ens})

    acc_df = pd.DataFrame(acc_data)

    axes[2].bar(x - width, acc_df['PanSTARRS'], width, label='PanSTARRS Basic', alpha=0.8)
    axes[2].bar(x, acc_df['Unified'], width, label='Unified Color-Only', alpha=0.8)
    axes[2].bar(x + width, acc_df['Ensemble'], width, label='Ensemble', alpha=0.8, color='green')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(acc_df['Temp Range'], rotation=45, ha='right')
    axes[2].set_ylabel('Within 10% (%)', fontsize=12)
    axes[2].set_title('Accuracy by Temperature Range', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0, 100])

    plt.tight_layout()

    # Save
    config = get_config()
    output_dir = config.get_path('figures') / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{ensemble_id}_component_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"      Saved: {output_file}")


def plot_residuals_by_component(df_ensemble, ensemble_id, subdir, model_name):
    """Plot residuals for each component model."""

    logger.info("   Creating residuals by component plot...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)

    # Calculate residuals
    res_ps = df_ensemble['pred_panstarrs'] - df_ensemble['true_temperature']
    res_uni = df_ensemble['pred_unified'] - df_ensemble['true_temperature']
    res_ens = df_ensemble['predicted_temperature'] - df_ensemble['true_temperature']

    # Row 1: Residuals vs Predicted Temperature
    # PanSTARRS
    axes[0, 0].hexbin(df_ensemble['pred_panstarrs'], res_ps,
                      gridsize=50, cmap='RdBu_r', mincnt=1, bins='log',
                      reduce_C_function=np.median)
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Predicted Temperature (K)', fontsize=11)
    axes[0, 0].set_ylabel('Residual (K)', fontsize=11)
    axes[0, 0].set_title('PanSTARRS Basic', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Unified
    axes[0, 1].hexbin(df_ensemble['pred_unified'], res_uni,
                      gridsize=50, cmap='RdBu_r', mincnt=1, bins='log',
                      reduce_C_function=np.median)
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Predicted Temperature (K)', fontsize=11)
    axes[0, 1].set_ylabel('Residual (K)', fontsize=11)
    axes[0, 1].set_title('Unified Color-Only', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Ensemble
    axes[0, 2].hexbin(df_ensemble['predicted_temperature'], res_ens,
                      gridsize=50, cmap='RdBu_r', mincnt=1, bins='log',
                      reduce_C_function=np.median)
    axes[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Predicted Temperature (K)', fontsize=11)
    axes[0, 2].set_ylabel('Residual (K)', fontsize=11)
    axes[0, 2].set_title('Ensemble', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # Row 2: Residual histograms
    bins = np.linspace(-3000, 3000, 50)

    axes[1, 0].hist(res_ps, bins=bins, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Residual (K)', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title(f'PanSTARRS: MAE={np.abs(res_ps).mean():.1f} K', fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(res_uni, bins=bins, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Residual (K)', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title(f'Unified: MAE={np.abs(res_uni).mean():.1f} K', fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].hist(res_ens, bins=bins, alpha=0.7, edgecolor='black', color='green')
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Residual (K)', fontsize=11)
    axes[1, 2].set_ylabel('Count', fontsize=11)
    axes[1, 2].set_title(f'Ensemble: MAE={np.abs(res_ens).mean():.1f} K', fontsize=11)
    axes[1, 2].grid(alpha=0.3)

    plt.suptitle(f'{model_name} - Residual Analysis by Component',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    config = get_config()
    output_dir = config.get_path('figures') / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{ensemble_id}_residuals_by_component.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"      Saved: {output_file}")


def main():
    """Generate validation plots for ensemble model."""

    config = get_config()
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("CREATE ENSEMBLE PANSTARRS-UNIFIED VALIDATION PLOTS")
    logger.info("="*80)

    ENSEMBLE_ID = 'ensemble_panstarrs_unified'
    MODEL_NAME = 'Ensemble: PanSTARRS Basic + Unified Color-Only'
    SUBDIR = 'ensemble_panstarrs_unified_validation'

    logger.info(f"\nEnsemble ID: {ENSEMBLE_ID}")

    # Load ensemble test predictions
    logger.info("\nLoading ensemble test predictions...")
    test_pred_file = models_dir / f'{ENSEMBLE_ID}_test_predictions.parquet'
    df_ensemble = pd.read_parquet(test_pred_file)
    logger.info(f"   Loaded: {len(df_ensemble):,} test predictions")
    logger.info(f"   Columns: {list(df_ensemble.columns)}")

    # Calculate metrics
    df_ensemble['abs_error'] = np.abs(df_ensemble['residual'])
    mae = df_ensemble['abs_error'].mean()
    rmse = np.sqrt((df_ensemble['residual']**2).mean())
    r2 = 1 - (df_ensemble['residual']**2).sum() / ((df_ensemble['true_temperature'] - df_ensemble['true_temperature'].mean())**2).sum()

    logger.info(f"\nEnsemble Performance:")
    logger.info(f"   MAE: {mae:.1f} K")
    logger.info(f"   RMSE: {rmse:.1f} K")
    logger.info(f"   R²: {r2:.3f}")

    # Calculate performance by temperature range
    logger.info("\nCalculating performance by temperature range...")
    bin_stats_df = calculate_bin_statistics(df_ensemble)

    # Generate plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING VALIDATION PLOTS")
    logger.info("="*80)

    # Plot 1: Test scatter
    logger.info("\n1. Test scatter plot")
    plot_test_scatter(df_ensemble, mae, rmse, r2, ENSEMBLE_ID, SUBDIR, MODEL_NAME)

    # Plot 2: Residuals (standard 2-panel)
    logger.info("\n2. Residuals plot")
    plot_residuals(df_ensemble, ENSEMBLE_ID, SUBDIR)

    # Plot 3: Performance by temperature
    logger.info("\n3. Performance by temperature plot")
    plot_performance_by_temp(df_ensemble, bin_stats_df, ENSEMBLE_ID, SUBDIR)

    # Plot 4: Component comparison
    logger.info("\n4. Component comparison plot")
    plot_component_comparison(df_ensemble, ENSEMBLE_ID, SUBDIR, MODEL_NAME)

    # Plot 5: Residuals by component
    logger.info("\n5. Residuals by component plot")
    plot_residuals_by_component(df_ensemble, ENSEMBLE_ID, SUBDIR, MODEL_NAME)

    logger.info("\n" + "="*80)
    logger.info("VALIDATION PLOTS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput directory: reports/figures/{SUBDIR}/")
    logger.info(f"Total plots created: 5")


if __name__ == '__main__':
    main()
