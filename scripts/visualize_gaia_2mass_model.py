#!/usr/bin/env python3
"""
Create standardized visualizations for Gaia + 2MASS colors model.

Produces unified format figures for:
- Model performance
- Feature analysis
- Temperature predictions
- Comparison plots

All figures follow standardized layout for easy comparison between models.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

# Plotting setup
plt.style.use('default')
sns.set_palette("husl")

def save_figure(fig, filename, subdir='gaia_2mass'):
    """Save figure to reports/figures directory."""
    fig_dir = Path('reports/figures') / subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    filepath = fig_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"   Saved: {filepath}")

def plot_model_performance(test_pred, metadata, model_id):
    """
    Standard model performance plot (4 subplots).
    - Predicted vs True (hexbin)
    - Residuals vs True
    - Residual distribution
    - Percent error distribution
    """
    logger.info("\n1. Creating model performance plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Predicted vs True
    ax = axes[0, 0]
    hb = ax.hexbin(test_pred['true_temperature'], test_pred['predicted_temperature'],
                   gridsize=80, cmap='YlOrRd', mincnt=1, bins='log')
    min_temp = test_pred['true_temperature'].min()
    max_temp = test_pred['true_temperature'].max()
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', lw=2, alpha=0.5, label='1:1 line')
    ax.set_xlabel('True Temperature (K)', fontsize=11)
    ax.set_ylabel('Predicted Temperature (K)', fontsize=11)
    ax.set_title('Predicted vs True Temperature', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='log10(count)')

    mae = metadata['metrics']['test']['mae']
    r2 = metadata['metrics']['test']['r2']
    ax.text(0.05, 0.95, f"MAE = {mae:.0f} K\\nR² = {r2:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Residuals
    ax = axes[0, 1]
    hb = ax.hexbin(test_pred['true_temperature'], test_pred['residual'],
                   gridsize=80, cmap='RdBu_r', mincnt=1)
    ax.axhline(0, color='black', linestyle='--', lw=2, alpha=0.5)
    ax.set_xlabel('True Temperature (K)', fontsize=11)
    ax.set_ylabel('Residual (True - Predicted) [K]', fontsize=11)
    ax.set_title('Residuals vs True Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='count')

    # 3. Residual distribution
    ax = axes[1, 0]
    ax.hist(test_pred['residual'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax.axvline(test_pred['residual'].median(), color='orange', linestyle='--', lw=2,
               label=f"Median = {test_pred['residual'].median():.1f} K")
    ax.set_xlabel('Residual (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Percent error
    ax = axes[1, 1]
    pct_err_filtered = test_pred['percent_error'][test_pred['percent_error'] < 50]
    ax.hist(pct_err_filtered, bins=100, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(test_pred['percent_error'].median(), color='red', linestyle='--', lw=2,
               label=f"Median = {test_pred['percent_error'].median():.2f}%")
    ax.set_xlabel('Percent Error (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Percent Error Distribution (<50%)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Model Performance: {model_id}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_figure(fig, f'{model_id}_performance.png')
    plt.close()

def plot_performance_by_temperature(test_pred, model_id):
    """
    Standard performance by temperature range plot (2 subplots).
    - MAE by temperature bin
    - Median percent error by temperature bin
    """
    logger.info("\n2. Creating performance by temperature plots...")

    temp_bins = [0, 4000, 5000, 6000, 7000, 10000, 50000]
    temp_labels = ['<4000K', '4000-5000K', '5000-6000K', '6000-7000K', '7000-10000K', '>10000K']
    test_pred['temp_bin'] = pd.cut(test_pred['true_temperature'], bins=temp_bins, labels=temp_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE by bin
    ax = axes[0]
    mae_by_bin = test_pred.groupby('temp_bin', observed=True).apply(
        lambda x: np.abs(x['residual']).mean()
    )
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(mae_by_bin)))
    bars = ax.bar(range(len(mae_by_bin)), mae_by_bin.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(mae_by_bin)))
    ax.set_xticklabels(mae_by_bin.index, rotation=45, ha='right')
    ax.set_ylabel('MAE (K)', fontsize=11)
    ax.set_title('MAE by Temperature Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    counts = test_pred.groupby('temp_bin', observed=True).size()
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)

    # Median % error by bin
    ax = axes[1]
    pct_err_by_bin = test_pred.groupby('temp_bin', observed=True)['percent_error'].median()
    bars = ax.bar(range(len(pct_err_by_bin)), pct_err_by_bin.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(pct_err_by_bin)))
    ax.set_xticklabels(pct_err_by_bin.index, rotation=45, ha='right')
    ax.set_ylabel('Median Percent Error (%)', fontsize=11)
    ax.set_title('Median Error by Temperature Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Performance by Temperature: {model_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'{model_id}_performance_by_temp.png')
    plt.close()

def plot_feature_importance(metadata, df_train, model_id):
    """
    Standard feature importance and distributions (2x3 grid).
    - Feature importance bar chart
    - Individual feature distributions
    - Target distribution
    """
    logger.info("\n3. Creating feature importance plots...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Feature importance
    ax = axes[0, 0]
    features = list(metadata['feature_importance'].keys())
    importances = list(metadata['feature_importance'].values())
    colors_feat = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(features, importances, color=colors_feat, edgecolor='black')
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, imp in zip(bars, importances):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {imp:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Feature distributions
    feature_cols = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
    titles = ['Gaia BP-RP', '2MASS J-H', '2MASS H-K', '2MASS J-K']
    colors_dist = ['blue', 'red', 'green', 'purple']

    for idx, (feat, title, color) in enumerate(zip(feature_cols, titles, colors_dist)):
        if idx < 2:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, idx - 2]

        ax.hist(df_train[feat], bins=100, edgecolor='black', alpha=0.7, color=color)
        ax.axvline(df_train[feat].median(), color='red', linestyle='--', lw=2,
                   label=f"Median = {df_train[feat].median():.3f}")
        ax.set_xlabel(f'{title} (mag)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Target distribution
    ax = axes[1, 2]
    ax.hist(df_train['teff_gspphot'], bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(df_train['teff_gspphot'].median(), color='red', linestyle='--', lw=2,
               label=f"Median = {df_train['teff_gspphot'].median():.0f} K")
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Target Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Feature Analysis: {model_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'{model_id}_features.png')
    plt.close()

def plot_color_temperature_relations(df_train, model_id):
    """
    Standard color-temperature relation plots (2x2 grid).
    - BP-RP vs J-H (colored by temp)
    - J-H vs H-K (colored by temp)
    - BP-RP vs Temperature
    - J-K vs Temperature
    """
    logger.info("\n4. Creating color-temperature relation plots...")

    sample_size = min(50000, len(df_train))
    df_sample = df_train.sample(n=sample_size, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # BP-RP vs J-H
    ax = axes[0, 0]
    scatter = ax.scatter(df_sample['bp_rp'], df_sample['j_h_color'],
                         c=df_sample['teff_gspphot'], s=1, alpha=0.5,
                         cmap='coolwarm', vmin=3000, vmax=10000)
    ax.set_xlabel('Gaia BP-RP (mag)', fontsize=11)
    ax.set_ylabel('2MASS J-H (mag)', fontsize=11)
    ax.set_title('Gaia vs 2MASS Colors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)', fontsize=10)

    # J-H vs H-K
    ax = axes[0, 1]
    scatter = ax.scatter(df_sample['j_h_color'], df_sample['h_k_color'],
                         c=df_sample['teff_gspphot'], s=1, alpha=0.5,
                         cmap='coolwarm', vmin=3000, vmax=10000)
    ax.set_xlabel('J-H (mag)', fontsize=11)
    ax.set_ylabel('H-K (mag)', fontsize=11)
    ax.set_title('2MASS Color-Color Diagram', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)', fontsize=10)

    # BP-RP vs Temperature
    ax = axes[1, 0]
    hb = ax.hexbin(df_sample['bp_rp'], df_sample['teff_gspphot'],
                   gridsize=80, cmap='YlOrRd', mincnt=1, bins='log')
    ax.set_xlabel('BP-RP (mag)', fontsize=11)
    ax.set_ylabel('Temperature (K)', fontsize=11)
    ax.set_title('BP-RP vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='log10(count)')

    # J-K vs Temperature
    ax = axes[1, 1]
    hb = ax.hexbin(df_sample['j_k_color'], df_sample['teff_gspphot'],
                   gridsize=80, cmap='YlOrRd', mincnt=1, bins='log')
    ax.set_xlabel('J-K (mag)', fontsize=11)
    ax.set_ylabel('Temperature (K)', fontsize=11)
    ax.set_title('J-K vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='log10(count)')

    plt.suptitle(f'Color-Temperature Relations: {model_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'{model_id}_color_temp_relations.png')
    plt.close()

def plot_predictions_overview(df_pred, model_id):
    """
    Standard predictions overview (2x2 grid).
    - Temperature distribution
    - Temperature by quality flag
    - Color distributions for predicted sample
    - Quality flag distribution
    """
    logger.info("\n5. Creating predictions overview plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Temperature distribution
    ax = axes[0, 0]
    ax.hist(df_pred['teff_predicted'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(df_pred['teff_predicted'].median(), color='red', linestyle='--', lw=2,
               label=f"Median = {df_pred['teff_predicted'].median():.0f} K")
    ax.set_xlabel('Predicted Temperature (K)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Predicted Temperature Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature by quality
    ax = axes[0, 1]
    quality_groups = ['AAA', 'AAB', 'ABA', 'ABB', 'Other']
    df_pred['quality_group'] = df_pred['ph_qual'].apply(
        lambda x: x if x in ['AAA', 'AAB', 'ABA', 'ABB'] else 'Other'
    )
    df_pred.boxplot(column='teff_predicted', by='quality_group', ax=ax)
    ax.set_xlabel('2MASS Quality Flag', fontsize=11)
    ax.set_ylabel('Predicted Temperature (K)', fontsize=11)
    ax.set_title('Temperature by 2MASS Quality', fontsize=12, fontweight='bold')
    ax.get_figure().suptitle('')  # Remove default title
    ax.grid(True, alpha=0.3, axis='y')

    # Color distributions
    ax = axes[1, 0]
    ax.hist(df_pred['bp_rp'], bins=50, alpha=0.5, label='BP-RP', color='blue', edgecolor='black')
    ax.hist(df_pred['j_k_color'], bins=50, alpha=0.5, label='J-K', color='red', edgecolor='black')
    ax.set_xlabel('Color (mag)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Color Distributions (Predicted Sample)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Quality flag distribution
    ax = axes[1, 1]
    quality_counts = df_pred['ph_qual'].value_counts().head(10)
    bars = ax.barh(range(len(quality_counts)), quality_counts.values, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(quality_counts)))
    ax.set_yticklabels(quality_counts.index)
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Top 10 Quality Flags', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (flag, count) in enumerate(quality_counts.items()):
        pct = 100 * count / len(df_pred)
        ax.text(count, i, f' {pct:.1f}%', va='center', fontsize=9)

    plt.suptitle(f'Predictions Overview: {model_id}\\n{len(df_pred):,} objects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'{model_id}_predictions_overview.png')
    plt.close()

def main():
    """Generate all visualizations."""

    config = get_config()
    processed_dir = config.get_path('processed')
    models_dir = Path('models')

    logger.info("="*80)
    logger.info("VISUALIZE GAIA + 2MASS COLORS MODEL")
    logger.info("="*80)

    MODEL_ID = 'rf_gaia_2mass_colors_20251016_155128'
    logger.info(f"\\nModel ID: {MODEL_ID}")

    # Load data
    logger.info("\\nLoading data...")

    # Metadata
    with open(models_dir / f'{MODEL_ID}_metadata.json', 'r') as f:
        metadata = json.load(f)
    logger.info("   Metadata loaded")

    # Test predictions
    test_pred = pd.read_parquet(models_dir / f'{MODEL_ID}_test_predictions.parquet')
    logger.info(f"   Test predictions: {len(test_pred):,} objects")

    # Training data
    df_train = pd.read_parquet(processed_dir / 'gaia_2mass_colors_training.parquet')
    logger.info(f"   Training data: {len(df_train):,} objects")

    # Predictions
    pred_file = processed_dir / f'gaia_2mass_temperature_predictions_{MODEL_ID}.parquet'
    df_pred = pd.read_parquet(pred_file)
    logger.info(f"   Predictions: {len(df_pred):,} objects")

    # Generate plots
    logger.info("\\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    plot_model_performance(test_pred, metadata, MODEL_ID)
    plot_performance_by_temperature(test_pred, MODEL_ID)
    plot_feature_importance(metadata, df_train, MODEL_ID)
    plot_color_temperature_relations(df_train, MODEL_ID)
    plot_predictions_overview(df_pred, MODEL_ID)

    logger.info("\\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\\nAll figures saved to: reports/figures/gaia_2mass/")
    logger.info(f"Total plots generated: 5 figures")

if __name__ == '__main__':
    main()
