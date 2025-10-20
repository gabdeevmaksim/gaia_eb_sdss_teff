# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an astronomical data science project for analyzing eclipsing binary stars using Gaia, Pan-STARRS, and SDSS data. The project focuses on effective temperature (Teff) analysis of eclipsing binaries with a ~1.2 million row catalog dataset.

## Naming Conventions

This project follows consistent naming conventions across all files to ensure clarity and maintainability.

### Python Files (Scripts and Modules)

**Scripts** (`scripts/`):
- Use `snake_case` for all script filenames
- Name describes the action performed (verb-based)
- Format: `{action}_{object}_{modifier}.py`
- Examples:
  - `convert_ecsv_to_parquet.py` - Converts ECSV files to Parquet format
  - `extract_panstarrs_duplicates.py` - Extracts duplicate Pan-STARRS entries
  - `add_gaia_colors_to_ml_data.py` - Adds Gaia colors to ML data
  - `train_high_quality_model.py` - Trains model on high-quality data
  - `crossmatch_apogee_local.py` - Cross-matches with APOGEE catalog
  - `query_2mass_irsa.py` - Queries 2MASS via IRSA service

**Source Modules** (`src/`):
- Use `snake_case` for all module filenames
- Name describes the content/purpose (noun-based)
- Avoid abbreviations unless standard in astronomy (e.g., `eb` for eclipsing binary)
- Examples:
  - `load_data.py` - Data loading utilities
  - `cache_manager.py` - Caching system manager
  - `settings.py` - Configuration settings
  - `engineering.py` - Feature engineering functions
  - `notebook_utils.py` - Notebook convenience functions

**Module Organization**:
- Each module should have a single, clear responsibility
- Use `__init__.py` to expose public API
- Private functions start with underscore (`_helper_function`)

### Jupyter Notebooks

**Naming Pattern**:
- Use `snake_case` for all notebook filenames
- Name describes the analysis type and subject
- Format: `{subject}_{analysis_type}.ipynb` OR `{model}_{task}_{variant}.ipynb`
- Examples:
  - `eclipsing_binary_analysis.ipynb` - General EB analysis
  - `sdss_temperature_analysis.ipynb` - Temperature analysis using SDSS
  - `bv_vk_temperature_analysis.ipynb` - Color-temperature analysis
  - `gaia_quality_flag_analysis.ipynb` - Quality flag analysis
  - `rf_regression_training.ipynb` - Random Forest regression training
  - `rf_classification_balanced_temp.ipynb` - RF classification with balanced classes
  - `hierarchical_clustering_hr.ipynb` - Hierarchical clustering on HR diagram

**Notebook Organization**:
- Analysis notebooks: `{data_source}_{analysis_focus}.ipynb`
- ML model notebooks: `{model_type}_{task}_{variant}.ipynb`
- Exploratory notebooks: `exploratory_{topic}.ipynb` (delete when done or move to archive)

### Data Files

**Raw Data** (`data/raw/`):
- Preserve original filenames from data sources (e.g., `eb_panstarrs_with_param-result.ecsv`)
- Use astronomy standard formats (ECSV, FITS)
- Include provenance in filename when possible

**Processed Data** (`data/processed/`):
- Use `snake_case` with descriptive names
- Include data content and processing stage
- Format: `{source}_{content}_{stage}.{ext}`
- Examples:
  - `eb_catalog.parquet` - Main eclipsing binary catalog
  - `eb_catalog_with_pm.parquet` - Catalog with proper motion
  - `gaia_eb_panstarrs_phot_with_temperatures.parquet` - Pan-STARRS photometry with Teff
  - `ml_training_data_clean.parquet` - Cleaned ML training data
  - `ml_training_data_with_gaia.parquet` - ML data with Gaia colors
  - `ml_training_data_high_quality.parquet` - High-quality subset
  - `flag0_temperature_predictions.parquet` - Predictions for flag 0 sources
  - `apogee_validated_predictions.parquet` - Predictions validated with APOGEE

**Companion Files**:
- Summary files: `{dataset_name}_SUMMARY.txt` (uppercase for visibility)
- Metadata files: `{dataset_name}_metadata.json`
- Backup files: `{dataset_name}_backup.parquet` (temporary, should be deleted)

**File Extensions**:
- `.parquet` - Processed tabular data (preferred for performance)
- `.csv` - Simple tabular exports
- `.ecsv` - Astronomy standard Enhanced CSV with metadata
- `.dat` - Legacy format compatibility (e.g., for external software)
- `.fits` - FITS images and tables (astronomy standard)

### Machine Learning Models

**Model Files** (`models/`):
- Format: `{model_type}_{task}_{variant}_{timestamp}.{ext}`
- Timestamp: `YYYYMMDD_HHMMSS` format
- Examples:
  - `rf_temperature_regressor_20251001_125556.pkl` - Basic RF model
  - `rf_temperature_regressor_feature_engineering_20251002_210423.pkl` - Enhanced model
  - `rf_temperature_regressor_high_quality_20251013_105249.pkl` - High-quality data model

**Model Artifacts** (same base name as model):
- `.pkl` - Serialized model (primary)
- `_metadata.json` - Model configuration and parameters
- `_SUMMARY.txt` - Performance metrics and training summary (uppercase for visibility)
- `_test_predictions.parquet` - Test set predictions for validation
- `_selector.pkl` - Feature selector (if applicable)

**Versioning Strategy**:
- Never overwrite existing models - always create new timestamped versions
- Keep at least 3 most recent versions of each model type
- Document major model changes in `_metadata.json`

### Documentation Files

**Documentation** (`docs/`):
- Use `UPPER_SNAKE_CASE.md` for major documentation (stands out in listings)
- Use sentence case for topic-specific guides
- Format: `{TOPIC}_{SUBTOPIC}.md` or `{topic}_guide.md`
- Examples:
  - `CONFIGURATION.md` - Configuration system documentation
  - `PIPELINES.md` - Pipeline documentation
  - `NOTEBOOK_CONVERSION.md` - Notebook best practices
  - `AI_TEMPERATURE_PREDICTION_RESEARCH.md` - Research notes
  - `FEATURE_PREPARATION_FOR_PREDICTION.md` - Feature engineering guide

**Special Documentation**:
- `README.md` - Project overview (repository root)
- `CLAUDE.md` - AI assistant guidelines (repository root)
- `CHANGELOG.md` - Version history (if applicable)
- `LICENSE` - License file (no extension)

### Configuration Files

**Configuration** (`config/`):
- Use `snake_case.yaml` or `snake_case.json`
- Main config: `config.yaml`
- Environment-specific: `config_{env}.yaml` (e.g., `config_dev.yaml`, `config_prod.yaml`)

**Project Metadata**:
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Modern Python project configuration
- `.gitignore` - Git ignore patterns
- `.cursorrules` - AI assistant rules (project-specific)

### Report and Figure Files

**Figures** (`reports/figures/`):
- Use `snake_case` for figure filenames
- Include analysis type and subject
- Organize in subdirectories by analysis category
- Format: `{analysis}_{subject}_{variant}.{ext}`
- Examples:
  - `hr_diagram_all_sources.png`
  - `temperature_distribution_by_filter.png`
  - `color_color_diagram_griz.png`
  - `validation/apogee_comparison_scatter.png`
  - `validation/galah_residuals_histogram.png`

**Figure Organization**:
- Group by analysis type in subdirectories: `exploratory/`, `validation/`, `modeling/`, `publication/`
- Use high-quality formats: `.png` (default), `.pdf` (publication), `.svg` (vector)

### Variable and Column Naming

**Code Variables**:
- `snake_case` for all variables and functions
- Descriptive names, avoid single letters except iterators (`i`, `j`) or astronomy conventions (`g`, `r`, `i` for bands)
- Examples: `temperature_kelvin`, `source_id`, `mag_error`, `color_index`

**DataFrame Columns**:
- `snake_case` for all column names
- Use astronomy standard abbreviations when appropriate
- Examples:
  - Magnitudes: `g_mag`, `r_mag`, `i_mag`, `z_mag`
  - Colors: `g_r`, `r_i`, `bp_rp` (band1_band2 format)
  - Temperatures: `teff_gspphot`, `teff_predicted`, `teff_apogee`
  - Errors: `g_mag_error`, `pmra_error`
  - IDs: `source_id`, `designation`, `tmass_id`
  - Flags: `flag_quality`, `flag_duplicated`

**Constants**:
- `UPPER_SNAKE_CASE` for constants
- Examples: `MISSING_VALUE`, `RANDOM_STATE`, `TEST_SIZE`

### Astronomy-Specific Conventions

**Photometric Bands**:
- Use lowercase for band names: `g`, `r`, `i`, `z`, `y` (Pan-STARRS/SDSS)
- Colors as band pairs: `g_r`, `r_i`, `i_z` (no spaces or dashes)
- 2MASS bands: `j`, `h`, `k` or `J`, `H`, `K` (preserve source convention)
- Gaia bands: `bp`, `rp`, `g_gaia` (lowercase, underscore for clarity)

**Astronomical Objects**:
- `eb` - Eclipsing binary
- `wd` - White dwarf
- `ms` - Main sequence
- `rg` - Red giant
- Full words preferred in documentation and user-facing names

**Catalogs**:
- Use standard abbreviations: `gaia`, `panstarrs`, `sdss`, `2mass`, `apogee`, `galah`
- Version when relevant: `gaia_dr3`, `apogee_dr17`, `galah_dr3`

### General Naming Principles

1. **Be Descriptive**: Names should clearly indicate purpose without needing comments
2. **Be Consistent**: Use the same naming pattern for similar items
3. **Be Concise**: Avoid unnecessary words, but don't sacrifice clarity
4. **Avoid Abbreviations**: Except for standard astronomy terms or very common abbreviations
5. **Use Astronomy Standards**: Follow IAU and VO conventions when applicable
6. **No Special Characters**: Avoid spaces, hyphens in code (use underscores), except in data filenames from external sources
7. **Lowercase Preferred**: Except for constants, classes (PascalCase), and special documentation files
8. **Timestamps**: Use ISO format `YYYYMMDD_HHMMSS` for versioned files

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
- `pyvo` - Virtual Observatory access
- `astroquery` - Astronomical catalog queries (2MASS, etc.)
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

**Primary Datasets**:
- `data/raw/eb_panstarrs_with_param-result.ecsv` - Eclipsing binary catalog with Pan-STARRS photometry
- `data/raw/stars_types.dat` - Stellar classification catalog (2.1M eclipsing binaries)
  - Contains: Gaia source_id, coordinates, period, Teff (58% have values), binary type (detached/overcontact), amplitude
  - Missing Teff values marked as `--`
  - Used as base catalog for temperature prediction
- `data/raw/eb_with_tmass_ids-result.ecsv` - Eclipsing binaries with 2MASS cross-match IDs
- Use `scripts/convert_ecsv_to_parquet.py` to convert ECSV files for performance

**Current Processed Files**:
- `data/processed/eb_catalog.parquet` - Main eclipsing binary catalog (82MB)
- `data/processed/eb_catalog_with_pm.parquet` - Catalog with proper motion (92MB)
- `data/processed/original_ext_source_id.csv` - Extracted source IDs (23MB)
- `data/processed/gaia_eb_panstarrs_phot_with_temperatures.parquet` - Pan-STARRS photometry with effective temperatures (1.17M objects)
- `data/processed/gaia_eb_panstarrs_phot_with_temperatures_and_flags.parquet` - Pan-STARRS photometry with temperatures and quality flags
- `data/processed/gaia_eb_sdss_teff.parquet` - SDSS photometry with effective temperatures
- `data/processed/gaia_eb_colors_temperatures.parquet` - Multi-band colors (B-V, V-K) and temperatures
- `data/processed/ml_training_data_clean.parquet` - Cleaned ML training dataset
- `data/processed/ml_training_data_with_gaia.parquet` - ML dataset enhanced with Gaia colors
- `data/processed/ml_training_data_with_gaia_with_flags.parquet` - ML dataset with Gaia GSP-Phot quality flags
- `data/processed/ml_training_data_high_quality.parquet` - High-quality dataset (flag 1 only, no magnitude features)
- `data/processed/flag0_temperature_predictions.parquet` - Temperature predictions for flag 0 sources
- `data/processed/apogee_validated_predictions.parquet` - Predictions validated against APOGEE DR17 spectroscopy
- `data/processed/apogee_xmatch.parquet` - APOGEE cross-match results
- `data/processed/galah_validated_predictions.parquet` - Predictions validated against GALAH DR3 spectroscopy
- `data/processed/eb_2mass_photometry.parquet` - 2MASS photometry for eclipsing binaries
- `data/processed/eb_full_catalog_temperatures.parquet` - Full catalog with predicted temperatures (429k sources)
- `data/processed/stars_types_with_predictions.parquet` - Combined stars_types.dat with ML predictions (2.1M sources, 78% with Teff)
- `data/processed/stars_types_with_predictions.dat` - CSV/DAT export of above (271 MB)
- `data/processed/eb_unified_features_engineered.parquet` - Unified feature dataset with engineered features (1.1M objects, 85 features)
- `data/processed/eb_unified_features_engineered_train.parquet` - Training subset with Gaia Teff (701k objects)
- `data/processed/eb_unified_features_engineered_predict.parquet` - Prediction subset without Gaia Teff (401k objects)
- `data/processed/predictions_rf_unified_engineered_*.parquet` - Temperature predictions using unified features model

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

**Quality Flag Processing and High-Quality Model Training**:
```bash
# Add Gaia GSP-Phot quality flags to ML training data
python scripts/add_gaia_quality_flags.py

# Create high-quality dataset (flag 1 sources only, no magnitude features)
python scripts/create_high_quality_ml_dataset.py

# Train Random Forest model on high-quality data
python scripts/train_high_quality_model.py

# Predict temperatures for flag 0 sources using high-quality model
python scripts/predict_flag0_temperatures.py
```

**Unified Features Workflow (Recommended)**:
```bash
# Create unified feature dataset with consistent feature engineering
# This applies the same feature engineering to ALL objects (train + predict)
python scripts/create_unified_feature_dataset.py --model-type engineered

# Train model on unified features
python scripts/train_model_unified_features.py --model-type engineered --n-estimators 300 --max-depth 20

# Generate predictions using trained model
python scripts/predict_unified_features.py --model models/rf_unified_engineered_YYYYMMDD_HHMMSS.pkl
```

**Spectroscopic Validation**:
```bash
# Download APOGEE DR17 and GALAH DR3 spectroscopic catalogs
python scripts/download_spectroscopic_catalogs.py

# Cross-match predictions with APOGEE DR17 for validation
python scripts/crossmatch_apogee_local.py

# Cross-match predictions with GALAH DR3 using VizieR Xmatch
python scripts/crossmatch_galah_xmatch.py
```

**Full Catalog Temperature Prediction**:
```bash
# Query 2MASS photometry via IRSA
python scripts/query_2mass_irsa.py

# Predict temperatures for full catalog (all eclipsing binaries)
python scripts/predict_temperatures_full_catalog.py

# Merge stars_types.dat with predicted temperatures
python scripts/merge_stars_types_with_predictions.py

# Convert Parquet to CSV/DAT format (standalone, works without venv)
python scripts/convert_parquet_to_csv.py input.parquet output.dat
python scripts/convert_parquet_to_csv.py input.parquet  # Auto-generates output name
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
# - gaia_quality_flag_analysis.ipynb - Gaia GSP-Phot quality flag analysis and comparison
# - rf_regression_training.ipynb - Random Forest regression model training
# - rf_temperature_prediction.ipynb - Temperature prediction using trained RF model
# - rf_regression_feature_engineering.ipynb - Advanced RF model with feature engineering
# - rf_classification_balanced_temp.ipynb - Temperature classification with balanced classes
# - rf_classification_fixed_temp_bins.ipynb - Temperature classification with fixed bins
# - hierarchical_clustering_hr.ipynb - HR diagram hierarchical clustering analysis
# - unified_features_model_validation.ipynb - Validation of unified features models (with magnitude)
# - unified_features_no_gpsf_model_validation.ipynb - Validation of color-only model (RECOMMENDED)

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
- **Use reusable modules** - See `docs/NOTEBOOK_UTILITIES.md`
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

- **Color-Only RF Model** (rf_unified_engineered_20251016_112332) - **RECOMMENDED**:
  - Features: 20 selected from 85 color-only features (NO magnitude features to avoid bias)
  - Colors: g-r, r-i, i-z, B-V, BP-RP with polynomial, interaction, log, and temperature-dependent transforms
  - Test MAE: 765.1 K, RMSE: 1168.4 K, R²: 0.315
  - Objects within 10%: 43.4%
  - **Key advantage**: Physically correct predictions based on colors (SED), not brightness
  - **BP-RP features dominate**: ~60% of total feature importance
  - **Data quality**: All objects have valid BP-RP colors (filtered out 11,248 with bp_rp=0)
  - Training: 701,644 objects | Predictions: 401,111 objects
  - Prediction mean: 4,862 K (much closer to training mean of 5,308 K, no magnitude bias)

**Model Usage**:
```python
import joblib
import pandas as pd

# Load color-only model (RECOMMENDED)
model = joblib.load('models/rf_unified_engineered_20251016_112332.pkl')
selector = joblib.load('models/rf_unified_engineered_20251016_112332_selector.pkl')

# Load prediction data with unified features
data = pd.read_parquet('data/processed/eb_unified_features_engineered_predict.parquet')

# Extract features (exclude ID and target columns)
feature_cols = [col for col in data.columns
                if col not in ['original_ext_source_id', 'gaia_source_id', 'teff_gspphot']]
X = data[feature_cols].values

# Apply feature selector and predict
X_selected = selector.transform(X)
predictions = model.predict(X_selected)
```

**Why Use the Color-Only Model?**:
1. **Physically Correct**: Based on spectral energy distribution (colors), not brightness
2. **No Magnitude Bias**: Predictions are independent of object brightness/distance
3. **High Data Quality**: All objects have valid BP-RP colors (no missing critical features)
4. **Consistent Features**: Same feature engineering applied to both training and prediction sets
5. **Better Generalization**: Prediction mean (4,862 K) closer to training mean (5,308 K)

## Model Validation Plots

**All temperature prediction models have standardized 7-plot validation** using the centralized `src/visualization/validation_plots.py` module. This ensures consistent visual style and metrics across all models.

### Validation Plot Set (7 plots per model)

Each model has these standardized validation plots:

1. **Test Scatter** - Predicted vs. Ground Truth with ±10% bounds
2. **Residuals** - 2-panel residual analysis (vs predicted, vs true)
3. **Performance by Temperature** - MAE/RMSE/Accuracy across temperature ranges
4. **Temperature Distributions** - Training vs. Predictions distributions (histogram + CDF)
5. **Color Distributions** - Multi-panel color distributions comparison
6. **Color-Temperature Relations** - 3-panel (training, predictions, overlay)
7. **Feature Importance** - Top 20 most important features

### Validation Scripts

All validation scripts follow the same pattern and use shared plotting functions:

```bash
# Generate validation plots for a specific model
python scripts/create_panstarrs_validation_plots.py      # Pan-STARRS only model
python scripts/create_combined_validation_plots.py       # Combined Pan-STARRS+2MASS+Gaia
python scripts/create_gaia_2mass_validation_plots.py     # Gaia+2MASS Basic
python scripts/create_gaia_2mass_engineered_validation_plots.py  # Gaia+2MASS Engineered
python scripts/create_unified_validation_plots.py        # Unified Pan-STARRS+Gaia
```

### Current Validation Coverage

**All 5 main models have complete validation (100% coverage)**:

| Model | Features | Validation Directory | Status |
|-------|----------|---------------------|--------|
| **Pan-STARRS Only** | g-r, r-i, i-z, B-V, g_mag | `panstarrs_validation/` | ✅ 7 plots |
| **Combined** | PS colors + 2MASS NIR + BP-RP | `combined_validation/` | ✅ 7 plots |
| **Gaia+2MASS Basic** | BP-RP + J-H, H-K, J-K | `gaia_2mass_validation/` | ✅ 7 plots |
| **Gaia+2MASS Engineered** | 30 engineered features | `gaia_2mass_engineered_validation/` | ✅ 7 plots |
| **Unified** | PS colors + BP-RP (color-only) | `unified_validation/` | ✅ 7 plots |

**Total**: 35 standardized validation plots across 5 models

### Validation Plot Style Standards

All plots follow these conventions (enforced by `src/visualization/validation_plots.py`):

- **DPI 300** for publication quality
- **Hexbin plots** with log-scale colormaps for density visualization
- **Color scheme**: Blue for training data, orange for predictions
- **Inverted Y-axis** for color-temperature relations (astronomical convention: hotter stars at bottom)
- **Consistent layouts** and figure sizes across all models
- **Astronomical conventions**: Temperature in Kelvin, colors as magnitude differences

### Creating Validation Plots for New Models

When training a new temperature prediction model, create standardized validation plots:

1. **During training**: Save test predictions as `{MODEL_ID}_test_predictions.parquet`
2. **Create validation script**: Follow the pattern in existing scripts
3. **Use shared functions**: Import from `src.visualization.validation_plots`
4. **Generate all 7 plots**: Ensure consistency with other models
5. **Organize by model**: Save to `reports/figures/{model_name}_validation/`

**Example script structure**:
```python
from src.visualization.validation_plots import (
    plot_test_scatter,
    plot_residuals,
    plot_performance_by_temp,
    plot_temp_distributions,
    plot_color_distributions,
    plot_color_temp_relations,
    plot_feature_importance,
    calculate_bin_statistics
)

# Load data
test_pred = pd.read_parquet(f'models/{MODEL_ID}_test_predictions.parquet')
metadata = json.load(open(f'models/{MODEL_ID}_metadata.json'))

# Generate all 7 plots
plot_test_scatter(test_pred, mae, rmse, r2, MODEL_ID, SUBDIR, MODEL_NAME)
plot_residuals(test_pred, MODEL_ID, SUBDIR)
bin_stats = calculate_bin_statistics(test_pred)
plot_performance_by_temp(test_pred, bin_stats, MODEL_ID, SUBDIR)
# ... remaining plots
```

### Model Comparison

See `docs/ALL_MODELS_COMPARISON.md` for detailed performance comparison across all models, including:
- Performance metrics (MAE, RMSE, R², accuracy within thresholds)
- Feature importance rankings
- Training details and hyperparameters
- Physical insights from color-temperature relationships
- Recommendations for production use

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
│   ├── add_gaia_quality_flags.py          # Add Gaia GSP-Phot quality flags
│   ├── create_high_quality_ml_dataset.py  # Create flag 1 only dataset
│   ├── train_high_quality_model.py        # Train RF model on high-quality data
│   ├── predict_flag0_temperatures.py      # Predict temperatures for flag 0 sources
│   ├── create_unified_feature_dataset.py  # Create unified features for train+predict (RECOMMENDED)
│   ├── train_model_unified_features.py    # Train model on unified features
│   ├── predict_unified_features.py        # Generate predictions with unified model
│   ├── download_spectroscopic_catalogs.py # Download APOGEE/GALAH catalogs
│   ├── crossmatch_apogee_local.py         # Validate with APOGEE DR17
│   ├── crossmatch_galah_xmatch.py         # Validate with GALAH DR3
│   ├── query_2mass_irsa.py                # Query 2MASS photometry via IRSA
│   ├── predict_temperatures_full_catalog.py # Predict Teff for full catalog
│   ├── merge_stars_types_with_predictions.py # Merge stars_types.dat with predictions
│   ├── convert_parquet_to_csv.py          # Convert Parquet to CSV/DAT (standalone)
│   ├── memory_monitor.py                  # System memory monitoring for training
│   ├── create_panstarrs_validation_plots.py      # Pan-STARRS model validation plots
│   ├── create_combined_validation_plots.py       # Combined model validation plots
│   ├── create_gaia_2mass_validation_plots.py     # Gaia+2MASS Basic validation plots
│   ├── create_gaia_2mass_engineered_validation_plots.py  # Gaia+2MASS Engineered validation plots
│   └── create_unified_validation_plots.py        # Unified model validation plots
├── src/                        # Source code modules
│   ├── data/                   # Data handling utilities
│   │   ├── load_data.py       # Multi-format data loader
│   │   └── cache_manager.py   # Caching system
│   ├── visualization/          # Plotting and visualization
│   │   ├── plots.py           # Astronomical plots, sky maps
│   │   └── validation_plots.py # Standardized model validation plots
│   ├── features/              # Feature engineering
│   │   └── engineering.py     # Reusable feature engineering functions
│   └── models/                # Machine learning models (placeholder)
├── notebooks/                  # Jupyter notebooks
│   ├── eclipsing_binary_analysis.ipynb          # Main analysis notebook
│   ├── sdss_temperature_analysis.ipynb          # SDSS temperature analysis
│   ├── bv_vk_temperature_analysis.ipynb         # B-V, V-K color analysis
│   ├── color_quality_analysis.ipynb             # Color data quality assessment
│   ├── gaia_quality_flag_analysis.ipynb         # Gaia GSP-Phot quality flag analysis
│   ├── rf_regression_training.ipynb             # Basic Random Forest model training
│   ├── rf_temperature_prediction.ipynb          # RF temperature predictions
│   ├── rf_regression_feature_engineering.ipynb  # Advanced RF with feature engineering
│   ├── rf_classification_balanced_temp.ipynb    # Temperature classification (balanced)
│   ├── rf_classification_fixed_temp_bins.ipynb  # Temperature classification (fixed bins)
│   ├── hierarchical_clustering_hr.ipynb         # HR diagram clustering analysis
│   ├── unified_features_model_validation.ipynb  # Unified features validation (with magnitude)
│   └── unified_features_no_gpsf_model_validation.ipynb  # Color-only model validation (RECOMMENDED)
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
│   ├── AI_TEMPERATURE_PREDICTION_RESEARCH.md  # ML temperature prediction research notes
│   ├── CONFIGURATION.md               # Configuration system guide
│   ├── FEATURE_PREPARATION_FOR_PREDICTION.md  # Feature engineering guide for predictions
│   ├── MIGRATION_GUIDE.md             # Migration guide for configuration system
│   ├── NOTEBOOK_CONVERSION.md         # Notebook best practices
│   ├── NOTEBOOK_GUIDE.md              # Notebook development guide
│   ├── NOTEBOOK_UTILITIES.md          # Notebook utilities and modules guide
│   ├── PIPELINES.md                   # Pipeline documentation
│   └── SCRIPTS_MIGRATION.md           # Scripts migration status and notes
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

---

## Unified Features Workflow

### Overview

The unified features workflow ensures **consistent feature engineering** across training and prediction sets. This approach solves the critical problem of feature distribution mismatch between objects with Gaia Teff (training) and objects without Gaia Teff (prediction).

### Key Advantages

1. **Consistent Feature Engineering**: Same transformations applied to ALL objects
2. **No Distribution Mismatch**: Train and predict sets derived from same processing pipeline
3. **Data Quality Control**: Outlier filtering and missing value handling applied uniformly
4. **Physical Correctness**: Color-only model avoids magnitude bias
5. **Reproducibility**: Single source of truth for feature engineering

### Workflow Steps

#### Step 1: Create Unified Feature Dataset

```bash
python scripts/create_unified_feature_dataset.py --model-type engineered
```

**What it does**:
- Loads complete catalog (1.17M objects from `gaia_eb_panstarrs_phot_with_temperatures.parquet`)
- Applies consistent outlier filtering to base features BEFORE feature engineering
- Creates engineered features: polynomial (degree 3), interactions, log transforms, temperature-dependent
- **Removes magnitude features** (gPSFMag) to avoid magnitude bias
- Filters objects with missing critical colors (requires g-r, r-i, i-z, BP-RP)
- Removes objects with bp_rp=0 (missing Gaia BP-RP colors)
- Splits into training (with Gaia Teff) and prediction (without Gaia Teff) sets
- Validates distributions between train/predict sets with KS tests

**Output files**:
- `eb_unified_features_engineered.parquet` - Complete dataset (1.1M objects, 85 features)
- `eb_unified_features_engineered_train.parquet` - Training set (701k objects)
- `eb_unified_features_engineered_predict.parquet` - Prediction set (401k objects)
- `eb_unified_features_engineered_SUMMARY.txt` - Dataset creation summary
- `reports/figures/feature_validation/feature_distributions_comparison.png` - Distribution comparison plots
- `reports/figures/feature_validation/feature_comparison_statistics.csv` - KS/t-test statistics

#### Step 2: Train Model on Unified Features

```bash
python scripts/train_model_unified_features.py --model-type engineered \
    --n-estimators 300 --max-depth 20 --n-features 20
```

**What it does**:
- Loads training set with unified features
- Applies SelectKBest feature selection (f_regression scoring)
- Trains Random Forest with specified hyperparameters
- Evaluates on hold-out test set (20% split)
- Saves model, selector, metadata, and test predictions

**Output files**:
- `rf_unified_engineered_YYYYMMDD_HHMMSS.pkl` - Trained model
- `rf_unified_engineered_YYYYMMDD_HHMMSS_selector.pkl` - Feature selector
- `rf_unified_engineered_YYYYMMDD_HHMMSS_metadata.json` - Model configuration
- `rf_unified_engineered_YYYYMMDD_HHMMSS_SUMMARY.txt` - Performance metrics
- `rf_unified_engineered_YYYYMMDD_HHMMSS_test_predictions.parquet` - Test set predictions

#### Step 3: Generate Predictions

```bash
python scripts/predict_unified_features.py \
    --model models/rf_unified_engineered_20251016_112332.pkl
```

**What it does**:
- Loads prediction set with unified features
- Applies the same feature selector used during training
- Generates temperature predictions for all objects without Gaia Teff
- Saves predictions with all input features

**Output file**:
- `predictions_rf_unified_engineered_YYYYMMDD_HHMMSS.parquet` - 401k predictions with features

#### Step 4: Validate Model (Jupyter Notebook)

```bash
jupyter lab notebooks/unified_features_no_gpsf_model_validation.ipynb
```

**What the notebook does**:
- Loads test predictions and compares with ground truth
- Analyzes performance by temperature range
- Compares temperature distributions (train vs predict)
- Compares color distributions to understand sample differences
- Creates HR diagrams (using gPSFMag for visualization only)
- Displays feature importance
- Generates validation figures in `reports/figures/validation/`

### Data Quality Checks

**Outlier Filtering (before feature engineering)**:
```python
color_limits = {
    'g_r_color': (-0.5, 3.0),    # Typical stellar colors
    'r_i_color': (-0.5, 2.0),
    'i_z_color': (-0.5, 1.5),
    'B_V_color': (-0.5, 3.0),
    'bp_rp': (-0.5, 4.0)
}
```

**Required Features**:
- All objects must have: g-r, r-i, i-z, B-V, BP-RP colors
- Objects with bp_rp=0 (missing Gaia BP-RP) are removed
- This ensures all predictions use complete feature sets

**Distribution Validation**:
- KS tests compare train/predict distributions for each feature
- t-tests compare means
- Plots show overlapping histograms for visual validation
- Statistics saved to CSV for review

### Color-Only Model vs Magnitude Model

| Aspect | Color-Only Model (RECOMMENDED) | Magnitude Model |
|--------|-------------------------------|-----------------|
| **Features** | 85 color-only features | 86 features (including gPSFMag) |
| **Magnitude bias** | None - predictions independent of brightness | Strong - 57% feature importance on gPSFMag |
| **Test R²** | 0.315 | 0.543 |
| **Test MAE** | 765 K | 550 K |
| **Physical correctness** | ✓ Based on SED (colors) | ✗ Brightness-dependent |
| **Prediction mean** | 4,862 K (close to training: 5,308 K) | 4,437 K (878 K systematic error) |
| **Data quality** | All objects have valid BP-RP | Includes objects with missing BP-RP |
| **Best for** | Science applications, fainter objects | Internal validation only |

**Why the color-only model has lower R² but is better**:
- R² is calculated on test set, which has similar magnitude distribution to training
- The magnitude model overfits to brightness, giving high R² on test but biased predictions
- The color-only model generalizes better to objects with different magnitude distributions
- Physically, temperature should be determined by SED (colors), not by brightness

### Model Files

**Current recommended model**: `rf_unified_engineered_20251016_112332`
- Training: 701,644 objects
- Predictions: 401,111 objects
- Test MAE: 765.1 K, R²: 0.315
- Within 10%: 43.4% of test objects
- BP-RP features: ~60% combined importance
- All predictions have valid BP-RP colors (no zeros)

### Troubleshooting

**Issue**: Distribution comparison shows wide x-axis ranges

**Cause**: Outliers in original features amplified by polynomial transforms

**Solution**: Filter outliers in BASE features before feature engineering (already implemented)

---

**Issue**: Zero bp_rp values in predictions

**Cause**: Some objects missing Gaia BP-RP photometry

**Solution**: Filter out objects with bp_rp=0 in Step 1 (already implemented)

---

**Issue**: Prediction mean temperature differs from training mean

**Cause**: Magnitude bias if using magnitude features, OR real physical difference in samples

**Solution**: Use color-only model to eliminate magnitude bias. Remaining difference reflects true sample differences (prediction set is redder/cooler)

### Validation Notebooks

**For magnitude model validation**:
- `notebooks/unified_features_model_validation.ipynb`
- Shows impact of magnitude features (57% importance on gPSFMag)
- Demonstrates systematic bias toward fainter objects

**For color-only model validation** (RECOMMENDED):
- `notebooks/unified_features_no_gpsf_model_validation.ipynb`
- Validates physically correct color-based predictions
- Shows BP-RP feature dominance (~60% importance)
- Demonstrates no magnitude bias
- Loads gPSFMag separately for HRD visualization only