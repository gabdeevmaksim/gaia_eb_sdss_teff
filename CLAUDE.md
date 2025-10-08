# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an astronomical data science project for analyzing eclipsing binary stars using Gaia, Pan-STARRS, and SDSS data. The project focuses on effective temperature (Teff) analysis of eclipsing binaries with a ~1.2 million row catalog dataset.

## Environment Setup

**Virtual Environment**: Use `.venv/` directory (not `venv/`)
```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- `astropy` - Astronomical data handling, ECSV format support
- `polars` - High-performance data analysis (preferred over pandas for large datasets)
- `pandas` - Alternative data analysis (available but prefer polars)
- `pyarrow` - Parquet format support for efficient data storage
- `healpy` - HEALPix sky mapping
- `aplpy` - Astronomical plotting
- `reproject` - Astronomical image reprojection
- `numpy` - Numerical computing foundation
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning algorithms
- `jupyter`/`notebook` - Interactive analysis environment

## Data Architecture

**Data Flow**: ECSV → Parquet → Analysis
- Raw data in `data/raw/` (ECSV format, ~226MB, 1.2M rows)
- Processed data in `data/processed/` (Parquet format for ~5-10x faster loading)
- External data in `data/external/` for third-party datasets
- Interim data in `data/interim/` for intermediate processing steps
- Cache system in `data/cache/` for expensive computations

**Primary Dataset**: `data/raw/eb_panstarrs_with_param-result.ecsv`
- Eclipsing binary catalog with Pan-STARRS photometry
- Use `scripts/convert_ecsv_to_parquet.py` to convert for performance

**Current Processed Files**:
- `data/processed/eb_catalog.parquet` - Main catalog (82MB)
- `data/processed/eb_catalog_with_pm.parquet` - Catalog with proper motion (92MB)
- `data/processed/original_ext_source_id.csv` - Extracted source IDs (23MB)
- `data/processed/gaia_eb_panstarrs_phot_with_temperatures.parquet` - Pan-STARRS photometry with effective temperatures (1.17M objects)
- `data/processed/gaia_eb_sdss_teff.parquet` - SDSS photometry with effective temperatures
- `data/processed/gaia_eb_colors_temperatures.parquet` - Multi-band colors (B-V, V-K) and temperatures
- `data/processed/ml_training_data_clean.parquet` - Cleaned ML training dataset
- `data/processed/ml_training_data_with_gaia.parquet` - ML dataset enhanced with Gaia colors

## Pipelines

**Automated Workflows** - See `docs/PIPELINES.md` for complete guide

```bash
# Run complete pipeline (data processing + ML training)
python pipeline.py --all

# Run data processing only
python pipeline.py --data

# Run ML training only
python pipeline.py --ml

# Custom model parameters
python pipeline.py --ml --n-estimators 500 --max-depth 25

# Dry run (see what would be executed)
python pipeline.py --all --dry-run
```

**Pipeline Architecture**:
- `src/pipeline/base.py` - Base pipeline classes
- `src/pipeline/data_pipeline.py` - Data processing workflow
- `src/pipeline/ml_pipeline.py` - ML training workflow
- `pipeline.py` - Master orchestrator (CLI)

## Core Scripts and Usage

**Data Conversion**:
```bash
# Convert ECSV to Parquet for faster loading
python scripts/convert_ecsv_to_parquet.py data/raw/eb_panstarrs_with_param-result.ecsv
```

**Data Extraction**:
```bash
# Extract original_ext_source_id column to CSV
python scripts/extract_original_ext_source_id.py
```

**Pan-STARRS Photometry Processing**:
```bash
# Extract duplicate Pan-STARRS entries
python scripts/extract_panstarrs_duplicates.py

# Merge duplicate measurements (fast version)
python scripts/merge_panstarrs_duplicates_fast.py

# Clean photometry data and filter for magnitude pairs
python scripts/clean_panstarrs_photometry.py

# Calculate effective temperatures from colors
python scripts/calculate_temperatures.py

# Add new colors (B-V, V-K) and temperatures to existing dataset
python scripts/add_new_colors.py

# Add Gaia colors (BP-RP) to ML training dataset
python scripts/add_gaia_colors_to_ml_data.py
```

**System Monitoring**:
```bash
# Monitor memory usage during intensive training
python scripts/memory_monitor.py --threshold 85 --check-interval 30 --process-name python
```

**Jupyter Analysis**:
```bash
# Start Jupyter for interactive analysis
jupyter lab notebooks/

# Template for new notebooks:
# - examples/notebook_template.ipynb - Shows best practices with utilities

# Analysis notebooks:
# - eclipsing_binary_analysis.ipynb - Main analysis notebook
# - sdss_temperature_analysis.ipynb - SDSS temperature analysis
# - bv_vk_temperature_analysis.ipynb - B-V and V-K color-temperature analysis
# - color_quality_analysis.ipynb - Color data quality assessment
# - rf_regression_training.ipynb - Random Forest regression model training
# - rf_temperature_prediction.ipynb - Temperature prediction using trained RF model
# - rf_regression_feature_engineering.ipynb - Advanced RF model with feature engineering
# - rf_classification_balanced_temp.ipynb - Temperature classification with balanced classes
# - rf_classification_fixed_temp_bins.ipynb - Temperature classification with fixed bins
# - hierarchical_clustering_hr.ipynb - HR diagram hierarchical clustering analysis

# Note: All notebooks should be updated to use src/notebook_utils
# See docs/NOTEBOOK_CONVERSION.md for migration guide
```

## Code Architecture

**Module Structure**:
- `src/config/` - Configuration management
  - `settings.py` - Configuration API with auto-detection
- `src/data/` - Data loading and caching utilities
  - `load_data.py` - Multi-format data loader (ECSV, Parquet, CSV)
  - `cache_manager.py` - Caching system for expensive computations
- `src/visualization/` - Plotting and visualization functions
  - `plots.py` - Astronomical plots, sky maps, histograms
- `src/features/` - Feature engineering utilities
  - `engineering.py` - Reusable feature engineering functions
- `src/notebook_utils.py` - Convenience functions for notebooks
- `src/models/` - Machine learning models (placeholder)

**Data Loading Pattern**:
```python
from src.data.load_data import load_eb_catalog

# Load as Polars DataFrame (recommended for performance)
data = load_eb_catalog('data/processed/eb_catalog.parquet', convert_to='polars')

# Load as Astropy Table (preserves astronomical metadata)
table = load_eb_catalog('data/raw/eb_catalog.ecsv', convert_to='astropy')
```

**Cache Usage**:
```python
from src.data.cache_manager import CacheManager
cache = CacheManager()  # Uses data/cache/ by default
```

## Performance Considerations

**Large Dataset Handling**:
- Always prefer Parquet over ECSV for repeated analysis
- Use Polars over Pandas for operations on the full dataset
- Implement caching for expensive astronomical computations
- Use astropy.Table only when astronomical metadata is required

**Data Processing Pipeline**:
1. Load raw ECSV with astropy (preserves metadata)
2. Convert to Polars for analysis (performance)
3. Cache intermediate results
4. Save processed data as Parquet

## Astronomical Specifics

**Coordinate Systems**: Use `astropy.coordinates.SkyCoord` for astronomical coordinates
**Sky Visualization**: HEALPix maps with `healpy` for all-sky distributions
**Photometric Data**: Pan-STARRS photometry accessed via MAST API
**File Formats**: ECSV (Enhanced CSV) is the astronomical standard for tabular data

## Configuration System

**All scripts use centralized configuration** - no hardcoded paths!

```python
from src.config import get_config

config = get_config()

# Get paths
data_dir = config.get_path('processed')
input_file = config.get_dataset_path('eb_catalog', 'raw')

# Get parameters
missing_val = config.get('processing', 'missing_value')
test_size = config.get('ml', 'test_size')
```

**Configuration file**: `config/config.yaml`
- All paths (relative to project root)
- Dataset filenames
- Processing parameters
- ML hyperparameters
- Temperature coefficients

**Documentation**: See `docs/CONFIGURATION.md` for complete API guide

**Example**: Run `python examples/configuration_example.py` to see it in action

## Development Workflow

**Working with Large Data**:
- Test on subsets before processing full catalog
- Use `data.sample(n=1000)` (Polars) or `data[:1000]` (astropy) for development
- Monitor memory usage when working with full dataset

**Notebook Development**:
- **Use reusable modules** - See `NOTEBOOK_MODULES_README.md`
- **No hardcoded paths** - Use `src/notebook_utils` functions
- Start with template: `examples/notebook_template.ipynb`
- See migration guide: `docs/NOTEBOOK_CONVERSION.md`

**Notebook Setup (Standard)**:
```python
import sys
sys.path.insert(0, '..')

from src.notebook_utils import (
    load_eb_catalog,
    load_panstarrs_data,
    load_ml_data,
    save_figure,
    MISSING_VALUE,
    RANDOM_STATE
)

from src.features import engineer_all_features
```

**Adding New Scripts**:
- Always use configuration system (see examples in `scripts/`)
- Import: `from src.config import get_config`
- Get paths: `config.get_dataset_path('key', 'location')`
- Get params: `config.get('section', 'param')`

## Pan-STARRS Data Processing

**Photometry Pipeline**:
- Raw Pan-STARRS data contains duplicate measurements for many sources
- Duplicates are merged using weighted averages based on photometric errors
- Final dataset includes colors (g-r, r-i, i-z) and effective temperatures
- Filter encoding system tracks which photometric bands are available per object
- Temperature calculations use empirical color-temperature relations with quality cuts

**Data Quality**:
- Missing measurements encoded as -999.0
- Color constraints (>= -0.5) applied before temperature calculations
- Filter encoding: "gr", "ri", "griz" etc. indicates available photometric bands
- ~95% of objects have all three temperature estimates (Te_gr, Te_ri, Te_iz)

## Machine Learning Models

**Trained Models** (stored in `models/`):
- Random Forest Temperature Regression models with versioned filenames
- Each model includes: `.pkl` (model), `_metadata.json` (config), `_SUMMARY.txt` (performance), `_test_predictions.parquet` (predictions)

**Model Performance**:
- **Basic RF Model** (20251001_125556):
  - Features: g-r, r-i, i-z colors, B-V synthetic color, g-band magnitude
  - Test MAE: 576 K, RMSE: 983 K, R²: 0.52
  - Objects within 10%: 65.52%

- **Feature-Engineered RF Model** (20251002_210423):
  - Features: 20 selected from 38 engineered features (polynomial, interactions, log transforms, temperature-dependent)
  - Test MAE: 318 K, RMSE: 524 K, R²: 0.86
  - **45% improvement** over basic model
  - Objects within 10%: significantly improved

**Model Usage**:
```python
import joblib
import polars as pl

# Load model and metadata
model = joblib.load('models/rf_temperature_regressor_feature_engineering_20251002_210423.pkl')
selector = joblib.load('models/rf_temperature_regressor_feature_engineering_20251002_210423_selector.pkl')

# Load new data and predict
data = pl.read_parquet('data/processed/ml_training_data_with_gaia.parquet')
# ... feature engineering ...
predictions = model.predict(selector.transform(features))
```

## Directory Structure

```
├── scripts/                    # Data processing scripts
│   ├── convert_ecsv_to_parquet.py         # ECSV → Parquet conversion
│   ├── extract_original_ext_source_id.py  # Source ID extraction
│   ├── extract_panstarrs_duplicates.py    # Find duplicate Pan-STARRS measurements
│   ├── merge_panstarrs_duplicates_fast.py # Merge duplicates with weighted averages
│   ├── clean_panstarrs_photometry.py      # Clean and filter photometry data
│   ├── calculate_temperatures.py          # Calculate effective temperatures
│   ├── add_new_colors.py                  # Add B-V, V-K colors and temperatures
│   ├── add_colors_and_temperatures.py     # Alternative color/temperature script
│   ├── add_gaia_colors_to_ml_data.py      # Add Gaia BP-RP colors to ML dataset
│   └── memory_monitor.py                  # System memory monitoring for training
├── src/                        # Source code modules
│   ├── data/                   # Data handling utilities
│   │   ├── load_data.py       # Multi-format data loader
│   │   └── cache_manager.py   # Caching system
│   ├── visualization/          # Plotting and visualization
│   │   └── plots.py           # Astronomical plots, sky maps
│   ├── features/              # Feature engineering (placeholder)
│   └── models/                # Machine learning models (placeholder)
├── notebooks/                  # Jupyter notebooks
│   ├── eclipsing_binary_analysis.ipynb          # Main analysis notebook
│   ├── sdss_temperature_analysis.ipynb          # SDSS temperature analysis
│   ├── bv_vk_temperature_analysis.ipynb         # B-V, V-K color analysis
│   ├── color_quality_analysis.ipynb             # Color data quality assessment
│   ├── rf_regression_training.ipynb             # Basic Random Forest model training
│   ├── rf_temperature_prediction.ipynb          # RF temperature predictions
│   ├── rf_regression_feature_engineering.ipynb  # Advanced RF with feature engineering
│   ├── rf_classification_balanced_temp.ipynb    # Temperature classification (balanced)
│   ├── rf_classification_fixed_temp_bins.ipynb  # Temperature classification (fixed bins)
│   └── hierarchical_clustering_hr.ipynb         # HR diagram clustering analysis
├── models/                     # Trained ML models
│   ├── rf_temperature_regressor_*.pkl           # Random Forest models
│   ├── rf_temperature_regressor_*_metadata.json # Model configurations
│   ├── rf_temperature_regressor_*_SUMMARY.txt   # Performance summaries
│   └── rf_temperature_regressor_*_test_predictions.parquet  # Test predictions
├── data/                       # Data storage
│   ├── raw/                   # Original ECSV files
│   ├── processed/             # Converted Parquet files
│   ├── external/              # Third-party datasets
│   ├── interim/               # Intermediate processing
│   └── cache/                 # Computation cache
├── config/                     # Configuration files
├── docs/                       # Documentation
├── reports/                    # Generated reports
│   ├── figures/               # Generated plots
│   └── presentations/         # Presentation materials
└── tests/                      # Unit tests
```

---

## ML Project Best Practices (from .cursorrules)

This project follows production-ready ML patterns. For detailed guidelines on building similar projects, see `.cursorrules` in the project root.

### Key Principles

**1. Configuration-Driven Development**
- All paths, parameters, and settings in `config/config.yaml`
- No hardcoded values anywhere in code
- Portable across environments and machines

**2. DRY (Don't Repeat Yourself)**
- Reusable modules in `src/` for common operations
- Shared utilities in `src/notebook_utils.py` for notebooks
- Feature engineering in `src/features/engineering.py`

**3. Pipeline Orchestration**
- Automated workflows with `pipeline.py`
- Sequential step execution with timing and logging
- Single command to reproduce entire analysis

**4. Production Readiness**
- Proper logging (not print statements)
- Error handling with informative messages
- Versioned models with metadata
- Reproducible results (fixed random seeds)

### Quick Reference Patterns

**Configuration Usage**:
```python
from src.config import get_config
config = get_config()
path = config.get_dataset_path('dataset_key', 'location')
param = config.get('section', 'parameter')
```

**Notebook Setup**:
```python
import sys
sys.path.insert(0, '..')
from src.notebook_utils import load_ml_data, save_figure, MISSING_VALUE
from src.features import engineer_all_features

data = load_ml_data(with_gaia=True)
features = engineer_all_features(data, color_cols=['g_r', 'r_i'])
save_figure(fig, 'analysis.png', subdir='exploratory')
```

**Pipeline Execution**:
```bash
# Complete workflow
python pipeline.py --all

# Individual pipelines
python pipeline.py --data  # Data processing only
python pipeline.py --ml    # ML training only

# Custom parameters
python pipeline.py --ml --n-estimators 500 --max-depth 25
```

For complete ML project patterns and templates, refer to `.cursorrules`.