"""
Standardized validation plots for temperature prediction models.

This module provides reusable plotting functions to ensure consistent
visualization style across all models.

Style conventions:
- Hexbin plots with log-scale colormaps
- Blue for training data, orange for predictions
- DPI 300 for publication quality
- Inverted Y-axis for color-temperature relations (astronomical convention)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Dict
from scipy import stats

# Plotting setup
plt.style.use('default')
sns.set_palette("husl")


def save_figure(fig, filename: str, subdir: str, dpi: int = 300):
    """
    Save figure to reports/figures directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    subdir : str
        Subdirectory under reports/figures/
    dpi : int, default 300
        Resolution for saved figure
    """
    fig_dir = Path('reports/figures') / subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    filepath = fig_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"   Saved: {filepath}")


def plot_test_scatter(
    test_pred: pd.DataFrame,
    mae: float,
    rmse: float,
    r2: float,
    model_id: str,
    subdir: str,
    model_name: str = "Model"
):
    """
    Test scatter: Predicted vs. Ground Truth with 1:1 and ±10% lines.

    Parameters
    ----------
    test_pred : pd.DataFrame
        Test predictions with columns 'true_temperature', 'predicted_temperature'
    mae, rmse, r2 : float
        Performance metrics
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    model_name : str
        Name for plot title
    """
    print("\n1. Creating test scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Hexbin plot for density
    hb = ax.hexbin(test_pred['true_temperature'], test_pred['predicted_temperature'],
                   gridsize=50, cmap='YlOrRd', mincnt=1, bins='log')

    # 1:1 line
    ax.plot([3000, 35000], [3000, 35000], 'k--', lw=2, label='1:1 line')

    # ±10% lines
    x = np.array([3000, 35000])
    ax.plot(x, x * 1.1, 'k:', lw=1, alpha=0.5, label='±10%')
    ax.plot(x, x * 0.9, 'k:', lw=1, alpha=0.5)

    ax.set_xlabel('True Teff (K)', fontsize=12)
    ax.set_ylabel('Predicted Teff (K)', fontsize=12)
    ax.set_title(f'Test Set: Predicted vs. Ground Truth ({model_name})\n'
                 f'MAE = {mae:.0f} K, RMSE = {rmse:.0f} K, R² = {r2:.3f}',
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3000, 35000)
    ax.set_ylim(3000, 35000)

    plt.colorbar(hb, ax=ax, label='log10(counts)')
    plt.tight_layout()
    save_figure(fig, f'{model_id}_test_scatter.png', subdir)
    plt.close()


def plot_residuals(
    test_pred: pd.DataFrame,
    model_id: str,
    subdir: str
):
    """
    Residual analysis with 2 subplots.

    Parameters
    ----------
    test_pred : pd.DataFrame
        Test predictions with columns 'true_temperature', 'residual'
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    """
    print("\n2. Creating residual plots...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Residuals vs. ground truth
    ax = axes[0]
    hb = ax.hexbin(test_pred['true_temperature'], test_pred['residual'],
                   gridsize=50, cmap='RdBu_r', mincnt=1, vmin=-2000, vmax=2000)
    ax.axhline(0, color='k', linestyle='--', lw=2)
    ax.set_xlabel('True Teff (K)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - True) [K]', fontsize=12)
    ax.set_title('Residuals vs. Ground Truth Temperature', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='counts')

    # Residual distribution
    ax = axes[1]
    ax.hist(test_pred['residual'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero residual')
    ax.axvline(test_pred['residual'].median(), color='blue', linestyle='-', lw=2,
               label=f"Median = {test_pred['residual'].median():.1f} K")
    ax.set_xlabel('Residual (K)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Residuals', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f'{model_id}_residuals.png', subdir)
    plt.close()


def plot_performance_by_temp(
    test_pred: pd.DataFrame,
    bin_stats_df: pd.DataFrame,
    model_id: str,
    subdir: str
):
    """
    Performance by temperature range (4 subplots).

    Parameters
    ----------
    test_pred : pd.DataFrame
        Test predictions
    bin_stats_df : pd.DataFrame
        Statistics by temperature bin with columns:
        'bin', 'mae', 'rmse', 'mean_pct', 'within_10'
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    """
    print("\n3. Creating performance by temperature plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE by temperature bin
    ax = axes[0, 0]
    ax.bar(bin_stats_df['bin'], bin_stats_df['mae'], color='steelblue', edgecolor='black')
    ax.set_ylabel('MAE (K)', fontsize=11)
    ax.set_title('Mean Absolute Error by Temperature Range', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # RMSE by temperature bin
    ax = axes[0, 1]
    ax.bar(bin_stats_df['bin'], bin_stats_df['rmse'], color='coral', edgecolor='black')
    ax.set_ylabel('RMSE (K)', fontsize=11)
    ax.set_title('Root Mean Square Error by Temperature Range', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Mean percent error by bin
    ax = axes[1, 0]
    ax.bar(bin_stats_df['bin'], bin_stats_df['mean_pct'], color='seagreen', edgecolor='black')
    ax.set_ylabel('Mean Percent Error (%)', fontsize=11)
    ax.set_title('Mean Percent Error by Temperature Range', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Fraction within 10% by bin
    ax = axes[1, 1]
    ax.bar(bin_stats_df['bin'], bin_stats_df['within_10'], color='mediumpurple', edgecolor='black')
    ax.set_ylabel('Objects Within 10% (%)', fontsize=11)
    ax.set_title('Accuracy (Within 10%) by Temperature Range', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, f'{model_id}_performance_by_temp.png', subdir)
    plt.close()


def plot_temp_distributions(
    train_data: pd.DataFrame,
    predictions: pd.DataFrame,
    model_id: str,
    subdir: str,
    train_col: str = 'teff_gspphot',
    pred_col: str = 'teff_predicted',
    model_name: str = "Model"
):
    """
    Temperature distribution comparison (2 subplots).

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data with temperature column
    predictions : pd.DataFrame
        Predictions with temperature column
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    train_col : str
        Column name for training temperatures
    pred_col : str
        Column name for predicted temperatures
    model_name : str
        Name for plot title
    """
    print("\n4. Creating temperature distribution plots...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram
    ax = axes[0]
    ax.hist(train_data[train_col], bins=100, alpha=0.6, label='Training (with Gaia Teff)',
            color='blue', edgecolor='black', density=True)
    ax.hist(predictions[pred_col], bins=100, alpha=0.6, label='Predictions (no Gaia Teff)',
            color='orange', edgecolor='black', density=True)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Normalized Frequency', fontsize=12)
    ax.set_title(f'Temperature Distribution: Training vs. Predictions ({model_name})',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3000, 20000)

    # Add vertical lines for means
    ax.axvline(train_data[train_col].mean(), color='blue', linestyle='--', lw=2)
    ax.axvline(predictions[pred_col].mean(), color='orange', linestyle='--', lw=2)

    # Cumulative distribution
    ax = axes[1]
    sorted_train = np.sort(train_data[train_col])
    sorted_pred = np.sort(predictions[pred_col])
    ax.plot(sorted_train, np.linspace(0, 1, len(sorted_train)),
            label='Training (with Gaia Teff)', color='blue', lw=2)
    ax.plot(sorted_pred, np.linspace(0, 1, len(sorted_pred)),
            label='Predictions (no Gaia Teff)', color='orange', lw=2)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3000, 20000)

    plt.tight_layout()
    save_figure(fig, f'{model_id}_temp_distributions.png', subdir)
    plt.close()


def plot_color_distributions(
    train_data: pd.DataFrame,
    predictions: pd.DataFrame,
    colors_to_compare: list,
    color_labels: list,
    model_id: str,
    subdir: str
):
    """
    Color distribution comparison (2x3 grid).

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    predictions : pd.DataFrame
        Predictions
    colors_to_compare : list
        Color column names to plot
    color_labels : list
        Display labels for colors
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    """
    print("\n5. Creating color distribution plots...")

    # Ensure we have 6 plots for consistent layout (pad if needed)
    while len(colors_to_compare) < 6:
        colors_to_compare.append(colors_to_compare[0])
        color_labels.append(f'({color_labels[0]} repeat)')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (col, label) in enumerate(zip(colors_to_compare, color_labels)):
        ax = axes[idx // 3, idx % 3]

        # Filter valid values
        train_valid = train_data[col][(train_data[col] > -5) & (train_data[col] < 5) & (~train_data[col].isna())]
        pred_valid = predictions[col][(predictions[col] > -5) & (predictions[col] < 5) & (~predictions[col].isna())]

        if len(train_valid) > 0 and len(pred_valid) > 0:
            ax.hist(train_valid, bins=50, alpha=0.6, label='Training', color='blue',
                    edgecolor='black', density=True)
            ax.hist(pred_valid, bins=50, alpha=0.6, label='Predictions', color='orange',
                    edgecolor='black', density=True)

            # Add means
            ax.axvline(train_valid.mean(), color='blue', linestyle='--', lw=1.5, alpha=0.7)
            ax.axvline(pred_valid.mean(), color='orange', linestyle='--', lw=1.5, alpha=0.7)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Normalized Frequency', fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f'{model_id}_color_distributions.png', subdir)
    plt.close()


def plot_color_temp_relations(
    train_data: pd.DataFrame,
    predictions: pd.DataFrame,
    color_col: str,
    color_label: str,
    model_id: str,
    subdir: str,
    train_temp_col: str = 'teff_gspphot',
    pred_temp_col: str = 'teff_predicted',
    sample_size: int = 10000
):
    """
    Color-Temperature relation diagrams (1x3 grid): Training, Predictions, Overlay.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    predictions : pd.DataFrame
        Predictions
    color_col : str
        Color column to plot
    color_label : str
        Display label for color
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    train_temp_col : str
        Temperature column in training data
    pred_temp_col : str
        Temperature column in predictions
    sample_size : int
        Number of points to sample for visualization
    """
    print("\n6. Creating color-temperature relation plots...")

    # Sample for visualization
    train_sample_size = min(sample_size, len(train_data))
    train_sample = train_data.sample(n=train_sample_size, random_state=42)

    pred_sample_size = min(sample_size, len(predictions))
    pred_sample = predictions.sample(n=pred_sample_size, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Training Set
    ax = axes[0]
    valid_mask_train = (train_sample[color_col] > -0.5) & (train_sample[color_col] < 3) & \
                       (train_sample[train_temp_col] > 2000) & (train_sample[train_temp_col] < 12000)
    if valid_mask_train.sum() > 0:
        hb = ax.hexbin(train_sample.loc[valid_mask_train, color_col],
                       train_sample.loc[valid_mask_train, train_temp_col],
                       gridsize=50, cmap='Blues', mincnt=1, bins='log')
        plt.colorbar(hb, ax=ax, label='log10(counts)')
    ax.set_xlabel(f'{color_label} color', fontsize=12)
    ax.set_ylabel('Teff_Gaia [K]', fontsize=12)
    ax.set_title(f'Training Set (n={valid_mask_train.sum():,})', fontsize=13)
    ax.set_ylim(12000, 2000)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Panel 2: Prediction Set
    ax = axes[1]
    valid_mask_pred = (pred_sample[color_col] > -0.5) & (pred_sample[color_col] < 3) & \
                      (pred_sample[pred_temp_col] > 2000) & (pred_sample[pred_temp_col] < 12000)
    if valid_mask_pred.sum() > 0:
        hb = ax.hexbin(pred_sample.loc[valid_mask_pred, color_col],
                       pred_sample.loc[valid_mask_pred, pred_temp_col],
                       gridsize=50, cmap='Oranges', mincnt=1, bins='log')
        plt.colorbar(hb, ax=ax, label='log10(counts)')
    ax.set_xlabel(f'{color_label} color', fontsize=12)
    ax.set_ylabel('Teff_predicted [K]', fontsize=12)
    ax.set_title(f'Predictions (n={valid_mask_pred.sum():,})', fontsize=13)
    ax.set_ylim(12000, 2000)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Panel 3: Overlay comparison
    ax = axes[2]
    if valid_mask_train.sum() > 0 and valid_mask_pred.sum() > 0:
        ax.scatter(train_sample.loc[valid_mask_train, color_col],
                   train_sample.loc[valid_mask_train, train_temp_col],
                   s=1, alpha=0.3, label='Training', color='blue')
        ax.scatter(pred_sample.loc[valid_mask_pred, color_col],
                   pred_sample.loc[valid_mask_pred, pred_temp_col],
                   s=1, alpha=0.3, label='Predictions', color='orange')
        ax.legend(fontsize=10, markerscale=5)
    ax.set_xlabel(f'{color_label} color', fontsize=12)
    ax.set_ylabel('Teff [K]', fontsize=12)
    ax.set_title(f'Overlay ({sample_size//1000}k sample each)', fontsize=13)
    ax.set_ylim(12000, 2000)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f'{model_id}_color_temp_relations.png', subdir)
    plt.close()


def plot_feature_importance(
    feature_importance: Dict[str, float],
    model_id: str,
    subdir: str,
    model_name: str = "Model",
    top_n: int = 20
):
    """
    Feature importance bar chart.

    Parameters
    ----------
    feature_importance : dict
        Dictionary of feature names to importance values
    model_id : str
        Model identifier for filename
    subdir : str
        Subdirectory for saving
    model_name : str
        Name for plot title
    top_n : int
        Number of top features to display
    """
    print("\n7. Creating feature importance plot...")

    # Extract and sort
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    # Plot top N features
    fig, ax = plt.subplots(figsize=(10, 8))
    n_features = min(top_n, len(sorted_features))
    y_pos = np.arange(n_features)
    ax.barh(y_pos, sorted_importances[:n_features], color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features[:n_features])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {n_features} Most Important Features ({model_name})', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_figure(fig, f'{model_id}_feature_importance.png', subdir)
    plt.close()


def calculate_bin_statistics(test_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance statistics by temperature bin.

    Parameters
    ----------
    test_pred : pd.DataFrame
        Test predictions with columns 'true_temperature', 'residual'

    Returns
    -------
    bin_stats_df : pd.DataFrame
        Statistics by bin
    """
    temp_bins = [0, 4000, 5000, 6000, 8000, 50000]
    temp_labels = ['<4000 K (Cool)', '4000-5000 K (Solar)', '5000-6000 K',
                   '6000-8000 K', '>8000 K (Hot)']
    test_pred['temp_bin'] = pd.cut(test_pred['true_temperature'], bins=temp_bins, labels=temp_labels)

    test_pred['abs_error'] = np.abs(test_pred['residual'])

    bin_stats = []
    for bin_label in temp_labels:
        mask = test_pred['temp_bin'] == bin_label
        subset = test_pred[mask]

        if len(subset) > 0:
            mae_bin = subset['abs_error'].mean()
            rmse_bin = np.sqrt((subset['residual']**2).mean())
            mean_pct = (100 * subset['abs_error'] / subset['true_temperature']).mean()
            within_10 = ((100 * subset['abs_error'] / subset['true_temperature']) <= 10).sum()
            within_10_pct = 100 * within_10 / len(subset)

            bin_stats.append({
                'bin': bin_label,
                'count': len(subset),
                'mae': mae_bin,
                'rmse': rmse_bin,
                'mean_pct': mean_pct,
                'within_10': within_10_pct
            })

    return pd.DataFrame(bin_stats)


def print_distribution_statistics(
    train_data: pd.DataFrame,
    predictions: pd.DataFrame,
    train_col: str = 'teff_gspphot',
    pred_col: str = 'teff_predicted'
):
    """
    Print statistical comparison of distributions.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    predictions : pd.DataFrame
        Predictions
    train_col : str
        Temperature column in training data
    pred_col : str
        Temperature column in predictions
    """
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON STATISTICS")
    print("="*80)

    ks_stat, ks_pval = stats.ks_2samp(train_data[train_col], predictions[pred_col])
    wasserstein_dist = stats.wasserstein_distance(train_data[train_col], predictions[pred_col])

    print(f"\nTraining set: {len(train_data):,} objects")
    print(f"  Mean: {train_data[train_col].mean():.0f} K")
    print(f"  Median: {train_data[train_col].median():.0f} K")
    print(f"\nPrediction set: {len(predictions):,} objects")
    print(f"  Mean: {predictions[pred_col].mean():.0f} K")
    print(f"  Median: {predictions[pred_col].median():.0f} K")
    print(f"\nDifference in means: {train_data[train_col].mean() - predictions[pred_col].mean():.0f} K")
    print(f"Kolmogorov-Smirnov test: p-value = {ks_pval:.2e}")
    print(f"Wasserstein Distance: {wasserstein_dist:.2f} K")
