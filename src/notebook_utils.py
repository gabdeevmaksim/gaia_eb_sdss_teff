"""
Utility functions for Jupyter notebooks.

This module provides convenient functions for common notebook operations:
- Data loading with configuration
- Model loading
- Plotting utilities
- Feature engineering

Usage in notebooks:
    import sys
    sys.path.insert(0, '..')
    from src.notebook_utils import load_eb_data, load_ml_data, load_model
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd
import polars as pl
import joblib

from .config import get_config


def load_eb_catalog(
    with_pm: bool = True,
    format: str = 'pandas'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load the eclipsing binary catalog.

    Parameters
    ----------
    with_pm : bool, default True
        If True, load catalog with proper motion
        If False, load basic catalog
    format : str, default 'pandas'
        Output format: 'pandas' or 'polars'

    Returns
    -------
    DataFrame
        Eclipsing binary catalog

    Examples
    --------
    >>> df = load_eb_catalog()
    >>> df = load_eb_catalog(with_pm=False, format='polars')
    """
    config = get_config()

    if with_pm:
        file_path = config.get_dataset_path('eb_catalog_with_pm', 'processed')
    else:
        file_path = config.get_dataset_path('eb_catalog_parquet', 'processed')

    if format == 'pandas':
        return pd.read_parquet(file_path)
    elif format == 'polars':
        return pl.read_parquet(file_path)
    else:
        raise ValueError(f"format must be 'pandas' or 'polars', got {format}")


def load_panstarrs_data(
    with_temps: bool = True,
    format: str = 'pandas'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load Pan-STARRS photometry data.

    Parameters
    ----------
    with_temps : bool, default True
        If True, load data with temperature estimates
        If False, load cleaned photometry only
    format : str, default 'pandas'
        Output format: 'pandas' or 'polars'

    Returns
    -------
    DataFrame
        Pan-STARRS photometry data

    Examples
    --------
    >>> df = load_panstarrs_data()
    >>> df = load_panstarrs_data(with_temps=False)
    """
    config = get_config()

    if with_temps:
        file_path = config.get_dataset_path('panstarrs_with_temps', 'processed')
    else:
        file_path = config.get_dataset_path('panstarrs_cleaned', 'external')

    if format == 'pandas':
        return pd.read_parquet(file_path) if file_path.suffix == '.parquet' else pd.read_csv(file_path)
    elif format == 'polars':
        return pl.read_parquet(file_path) if file_path.suffix == '.parquet' else pl.read_csv(file_path)
    else:
        raise ValueError(f"format must be 'pandas' or 'polars', got {format}")


def load_ml_data(
    with_gaia: bool = True,
    format: str = 'pandas'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load ML training data.

    Parameters
    ----------
    with_gaia : bool, default True
        If True, load data with Gaia colors (BP-RP)
        If False, load cleaned data without Gaia
    format : str, default 'pandas'
        Output format: 'pandas' or 'polars'

    Returns
    -------
    DataFrame
        ML training data

    Examples
    --------
    >>> df = load_ml_data()
    >>> df = load_ml_data(with_gaia=False, format='polars')
    """
    config = get_config()

    if with_gaia:
        file_path = config.get_dataset_path('ml_training_with_gaia', 'processed')
    else:
        file_path = config.get_dataset_path('ml_training_clean', 'processed')

    if format == 'pandas':
        return pd.read_parquet(file_path)
    elif format == 'polars':
        return pl.read_parquet(file_path)
    else:
        raise ValueError(f"format must be 'pandas' or 'polars', got {format}")


def load_colors_temperatures(
    format: str = 'pandas'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load multi-band colors and temperatures dataset.

    Parameters
    ----------
    format : str, default 'pandas'
        Output format: 'pandas' or 'polars'

    Returns
    -------
    DataFrame
        Colors and temperatures data

    Examples
    --------
    >>> df = load_colors_temperatures()
    """
    config = get_config()
    file_path = config.get_dataset_path('colors_temps', 'processed')

    if format == 'pandas':
        return pd.read_parquet(file_path)
    elif format == 'polars':
        return pl.read_parquet(file_path)
    else:
        raise ValueError(f"format must be 'pandas' or 'polars', got {format}")


def load_model(
    model_id: Optional[str] = None,
    return_metadata: bool = False
) -> Union[object, Tuple[object, dict]]:
    """
    Load a trained model.

    Parameters
    ----------
    model_id : str, optional
        Model ID (timestamp). If None, loads the most recent model.
    return_metadata : bool, default False
        If True, also return metadata dictionary

    Returns
    -------
    model : object
        Trained scikit-learn model
    metadata : dict, optional
        Model metadata (only if return_metadata=True)

    Examples
    --------
    >>> model = load_model()
    >>> model, metadata = load_model(return_metadata=True)
    >>> model = load_model('20251002_210423')
    """
    config = get_config()
    models_dir = config.get_path('models')

    if model_id is None:
        # Find most recent model
        model_files = sorted(models_dir.glob('rf_temperature_regressor_*.pkl'))
        if not model_files:
            raise FileNotFoundError("No trained models found in models/")
        model_file = model_files[-1]
    else:
        model_file = models_dir / f'rf_temperature_regressor_{model_id}.pkl'
        if not model_file.exists():
            # Try with feature engineering variant
            model_file = models_dir / f'rf_temperature_regressor_feature_engineering_{model_id}.pkl'

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Load model
    model = joblib.load(model_file)

    if return_metadata:
        # Load metadata
        import json
        metadata_file = model_file.with_name(model_file.stem + '_metadata.json')
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        return model, metadata
    else:
        return model


def get_config_value(*keys, default=None):
    """
    Get a configuration value (convenience function for notebooks).

    Parameters
    ----------
    *keys : str
        Keys to navigate nested config
    default : any, optional
        Default value if key not found

    Returns
    -------
    value
        Configuration value

    Examples
    --------
    >>> missing_val = get_config_value('processing', 'missing_value')
    >>> test_size = get_config_value('ml', 'test_size')
    """
    config = get_config()
    return config.get(*keys, default=default)


def get_path(key: str, ensure_exists: bool = False) -> Path:
    """
    Get a path from configuration (convenience function for notebooks).

    Parameters
    ----------
    key : str
        Path key in config
    ensure_exists : bool, default False
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Absolute path

    Examples
    --------
    >>> data_dir = get_path('processed')
    >>> figures_dir = get_path('figures', ensure_exists=True)
    """
    config = get_config()
    return config.get_path(key, ensure_exists=ensure_exists)


def save_figure(
    fig,
    filename: str,
    subdir: Optional[str] = None,
    dpi: int = 300,
    **kwargs
):
    """
    Save a matplotlib/seaborn figure to the reports/figures directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (without path)
    subdir : str, optional
        Subdirectory within figures/ (e.g., 'temperature_analysis')
    dpi : int, default 300
        Resolution
    **kwargs
        Additional arguments passed to savefig()

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> # ... plotting ...
    >>> save_figure(fig, 'temperature_distribution.png')
    >>> save_figure(fig, 'hr_diagram.png', subdir='hr_analysis')
    """
    config = get_config()
    figures_dir = config.get_path('figures', ensure_exists=True)

    if subdir:
        save_dir = figures_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = figures_dir

    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"✓ Saved figure: {save_path.relative_to(config.project_root)}")


# Common configuration values as module-level constants for convenience
CONFIG = get_config()
MISSING_VALUE = CONFIG.get('processing', 'missing_value')
COLOR_THRESHOLD = CONFIG.get('processing', 'color_threshold')
TEST_SIZE = CONFIG.get('ml', 'test_size')
RANDOM_STATE = CONFIG.get('ml', 'random_state')
