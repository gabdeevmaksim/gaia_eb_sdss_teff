# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Astronomical data science project for analyzing eclipsing binary stars using Gaia, Pan-STARRS, and SDSS data. Focus: effective temperature (Teff) prediction for ~1.2M eclipsing binaries using ML models.

## Naming Conventions

### Python Files
- **Scripts**: `snake_case`, verb-based (e.g., `train_high_quality_model.py`, `crossmatch_apogee_local.py`)
- **Modules**: `snake_case`, noun-based (e.g., `load_data.py`, `engineering.py`, `cache_manager.py`)
- Private functions: prefix with `_`

### Notebooks
- Format: `{subject}_{analysis_type}.ipynb` or `{model}_{task}_{variant}.ipynb`
- Examples: `eclipsing_binary_analysis.ipynb`, `rf_regression_training.ipynb`

### Data Files
- **Raw**: Preserve original names (ECSV, FITS formats)
- **Processed**: `{source}_{content}_{stage}.{ext}` (prefer `.parquet` for performance)
- **Companion files**: `{dataset}_SUMMARY.txt`, `{dataset}_metadata.json`

### Models
- Format: `{model_type}_{task}_{variant}_{timestamp}.pkl` (timestamp: `YYYYMMDD_HHMMSS`)
- Artifacts: `.pkl`, `_metadata.json`, `_SUMMARY.txt`, `_test_predictions.parquet`
- **Never overwrite models** - always create new timestamped versions

### Variables & Columns
- Code: `snake_case` (e.g., `temperature_kelvin`, `mag_error`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MISSING_VALUE`, `RANDOM_STATE`)
- Magnitudes: `g_mag`, `r_mag`, `i_mag`
- Colors: `g_r`, `r_i`, `bp_rp` (band1_band2 format)
- Temperatures: `teff_gspphot`, `teff_predicted`

### Astronomy Conventions
- Bands: lowercase (`g`, `r`, `i`, `z` for Pan-STARRS/SDSS; `bp`, `rp` for Gaia; `j`, `h`, `k` for 2MASS)
- Catalogs: `gaia`, `panstarrs`, `sdss`, `2mass`, `apogee`, `galah` (add version when relevant: `gaia_dr3`)

## Environment Setup

```bash
source .venv/bin/activate  # Use .venv/ not venv/
pip install -r requirements.txt
```

**Key Dependencies**: `astropy`, `polars` (preferred), `pandas`, `pyarrow`, `healpy`, `scikit-learn`, `jupyter`

## Data Architecture

**Flow**: ECSV → Parquet → Analysis
- `data/raw/` - Original ECSV files (~226MB, 1.2M rows)
- `data/processed/` - Parquet files (5-10x faster loading)
- `data/cache/` - Expensive computation cache

**Primary Datasets**:
- `eb_panstarrs_with_param-result.ecsv` - EB catalog with Pan-STARRS photometry
- `stars_types.dat` - 2.1M EBs with Teff (58% have values, `--` = missing)
- Key processed files: `ml_training_data_with_gaia.parquet`, `eb_unified_features_engineered.parquet`

**Uncertainty Propagation Files**:
- `data_for_teff_logg_perturbed.parquet` - Expanded dataset (3 variants per object: baseline, +σ, -σ)
- `teff_predictions_logg_perturbed.parquet` - Predictions for all variants (2.5M rows)
- `teff_predictions_with_logg_propagated_final.parquet` - Final with propagated uncertainties (847k objects)
  - Columns: `teff_predicted`, `teff_unc_rf`, `teff_unc_logg`, `teff_unc_total`, `gradient_teff_logg`

**Best-of-Three Ensemble**:
- `teff_predictions_best_of_three.parquet` - Best prediction per object (lowest uncertainty, 847k objects)
  - Columns: `source_id`, `teff_best`, `unc_best`, `best_model` (teff_only/teff_logg/teff_cluster)
  - Mean uncertainty: 288K (18.8% improvement vs single model)
- `stars_types_with_best_predictions.fits` - Full catalog (2.1M EBs, 97.2% with Teff, 196MB)
  - Quality flags: A=Gaia, B=ML<300K, C=ML<500K, D=ML≥500K, X=none
  - Description file: `stars_types_with_best_predictions_DESCRIPTION.txt`

**Use `scripts/convert_ecsv_to_parquet.py` to convert for performance**

## Pipelines

**Automated Workflows** - See `docs/PIPELINES.md` and `docs/CONFIGURABLE_PIPELINE.md`

```bash
# Complete pipeline
python pipeline.py --all

# Individual components
python pipeline.py --data                                           # Data processing
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml  # ML training (RECOMMENDED)
python pipeline.py --predict --pred-config config/prediction/predict_gaia_2mass_ir.yaml
python pipeline.py --validate --val-config config/validation/validate_gaia_2mass_ir.yaml

# Dry run
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml --dry-run
```

**Pipeline Modules**:
- `src/pipeline/configurable_ml_pipeline.py` - Configurable ML training (RECOMMENDED)
- `src/pipeline/prediction_pipeline.py` - Prediction workflow
- `src/pipeline/validation_pipeline.py` - Validation workflow
- `pipeline.py` - Master orchestrator

**Target Transformation** (NEW):
- Add `target_transform: "log"` to model config (options: `log`, `log2`, `ln`, `none`)
- Auto inverse-transforms predictions to original scale
- Effective for targets spanning orders of magnitude (e.g., Teff: 3000-30000K)

## Core Scripts

**Common Workflows**:
```bash
# Data conversion & processing
python scripts/convert_ecsv_to_parquet.py data/raw/*.ecsv
python scripts/clean_panstarrs_photometry.py
python scripts/add_gaia_colors_to_ml_data.py

# Unified features (RECOMMENDED)
python scripts/create_unified_feature_dataset.py --model-type engineered
python scripts/train_model_unified_features.py --model-type engineered
python scripts/predict_unified_features.py --model models/rf_unified_*.pkl

# Ensemble models
python scripts/create_ensemble_panstarrs_unified.py
python scripts/create_ensemble_panstarrs_unified_validation_plots.py

# Spectroscopic validation
python scripts/crossmatch_apogee_local.py
python scripts/crossmatch_galah_xmatch.py

# Full catalog prediction
python scripts/query_2mass_irsa.py
python scripts/predict_temperatures_full_catalog.py
python scripts/merge_stars_types_with_predictions.py

# Jupyter analysis
jupyter lab notebooks/
```

**Uncertainty Propagation Workflow** (NEW):
```bash
# Step 1: Create perturbed dataset (3 variants per object: baseline, +σ, -σ)
python scripts/create_logg_perturbed_dataset.py

# Step 2: Predict Teff for all variants
python pipeline.py --predict --pred-config config/prediction/predict_teff_logg_perturbed.yaml

# Step 3: Calculate propagated uncertainties (numerical gradient + quadrature)
python scripts/calculate_propagated_uncertainties.py

# Step 4: Visualize uncertainty analysis
python scripts/visualize_propagated_uncertainties.py

# Output: data/processed/teff_predictions_with_logg_propagated_final.parquet
# Columns: teff_predicted, teff_unc_rf, teff_unc_logg, teff_unc_total, gradient_teff_logg
```

**Three-Model Comparison**:
```bash
# Compare Teff Only, Teff+logg, Teff+Clustering with full tree uncertainties
python scripts/compare_three_teff_models.py
# Output: reports/figures/three_model_comparison/ (4 comparison plots)
```

**Best-of-Three Ensemble**:
```bash
# Create ensemble by selecting lowest uncertainty prediction per object
python scripts/create_best_uncertainty_ensemble.py
# Output: teff_predictions_best_of_three.parquet (mean unc: 288K, 18.8% improvement)

# Create final catalog merging stars_types.dat with best predictions
python scripts/create_stars_types_with_best_predictions.py
# Output: stars_types_with_best_predictions.fits (2.1M EBs, 97.2% with Teff)
#         + _DESCRIPTION.txt (comprehensive documentation)
```

## Code Architecture

**Module Structure**:
- `src/config/` - Configuration (`settings.py`)
- `src/data/` - Data loading (`load_data.py`, `cache_manager.py`)
- `src/visualization/` - Plots (`plots.py`, `validation_plots.py`)
- `src/features/` - Feature engineering (`engineering.py`)
- `src/notebook_utils.py` - Notebook convenience functions

**Usage Patterns**:
```python
from src.data.load_data import load_eb_catalog
data = load_eb_catalog('data/processed/eb_catalog.parquet', convert_to='polars')

from src.data.cache_manager import CacheManager
cache = CacheManager()
```

## Performance

- **Prefer Parquet** over ECSV for repeated analysis
- **Use Polars** over Pandas for large datasets
- **Cache** expensive computations
- **Test on subsets** before processing full catalog: `data.sample(n=1000)`

**RF Uncertainty Estimation**:
- **Fast sampling** (`fast: true, n_sample_trees: 20`): Quick (~30s) but overestimates by ~73%
- **Full tree method** (`fast: false`): Accurate (~3 min), use for production predictions
- **Important**: Always use `fast: false` in prediction configs for accurate uncertainties

**Log-Space Models**:
- Models with `_log` suffix predict log10(Teff), not Kelvin
- **Conversion required**:
  - Teff_K = 10^(prediction)
  - Uncertainty_K = Teff_K × uncertainty_log × ln(10)
- Pipeline auto-converts when saving, but manual conversions need this formula

## Configuration System

**All scripts use centralized config** - no hardcoded paths!

```python
from src.config import get_config
config = get_config()

# Paths & parameters
data_dir = config.get_path('processed')
input_file = config.get_dataset_path('eb_catalog', 'raw')
missing_val = config.get('processing', 'missing_value')
```

**Config file**: `config/config.yaml` - See `docs/CONFIGURATION.md` for details

## Development Workflow

**Notebooks**:
```python
import sys
sys.path.insert(0, '..')

from src.notebook_utils import (
    load_eb_catalog, load_ml_data, save_figure,
    MISSING_VALUE, RANDOM_STATE
)
from src.features import engineer_all_features
```

- Use reusable modules - See `docs/NOTEBOOK_UTILITIES.md`
- No hardcoded paths - Use `src/notebook_utils`
- Template: `examples/notebook_template.ipynb`

**Adding Scripts**:
- Always use config: `from src.config import get_config`
- Get paths: `config.get_dataset_path('key', 'location')`

## Machine Learning Models

**Model Selection Guide**:
- **Best overall**: Ensemble (MAE 720K) - balances accuracy & physical correctness
- **Best for Gaia-only**: Log-Transformed Gaia All Colors (MAE 557K) - better for cool/mid stars
- **Best single model**: Color-Only (MAE 765K) - physically correct, no magnitude bias
- **Lowest MAE**: Feature-Engineered (MAE 318K) - includes magnitude, may have bias
- **Best for cool stars (<6000K)**: Log-Transformed - 8-11% improvement

**Key Models** (stored in `models/`):

1. **Ensemble** (ensemble_panstarrs_unified) - **RECOMMENDED**:
   - Combines PanSTARRS Basic + Unified Color-Only
   - MAE: 720.4K, RMSE: 1183.7K, R²: 0.297, Within 10%: 53.0%
   - Balances magnitude accuracy with color-based physics

2. **Color-Only** (rf_unified_engineered_20251016_112332):
   - 85 color-only features (NO magnitude to avoid bias)
   - MAE: 765.1K, RMSE: 1168.4K, R²: 0.315, Within 10%: 43.4%
   - BP-RP features dominate (~60% importance)

3. **Gaia Log-Transformed** (rf_gaia_all_colors_teff_log_20251112_162857):
   - 6 Gaia colors + bands, log10(teff) target
   - MAE: 556.9K, RMSE: 1021.3K, R²: 0.640, Within 10%: 68.5%
   - 8-11% better for cool/mid stars (<6000K)
   - Config: `config/models/gaia_all_colors_teff_log.yaml`

**Model Artifacts**: Each model has `.pkl`, `_metadata.json`, `_SUMMARY.txt`, `_test_predictions.parquet`

## Model Validation

**Standardized plots** via `src/visualization/validation_plots.py`:
- Test scatter, residuals, performance by temp, distributions, color-temp relations, feature importance

**Validation Scripts**:
```bash
python scripts/create_panstarrs_validation_plots.py
python scripts/create_gaia_2mass_validation_plots.py
python scripts/create_unified_validation_plots.py
python scripts/create_ensemble_panstarrs_unified_validation_plots.py
```

**Coverage**: 6 models × 5-7 plots = 40 validation plots (DPI 300, hexbin density)

**Generalized validation**: Supports any target (Teff, logg, [Fe/H]) with dynamic limits, units, labels

## Feature Correlation Analysis

**Purpose**: Guide feature selection for Teff and log(g) models

**Key Findings** (`reports/FEATURE_CORRELATION_ANALYSIS.md`):
- **Teff**: Colors (g_rp, bp_rp) are best (|r| ~ 0.50-0.54)
- **log(g)**: Magnitudes (rp, g, bp) are best (|r| ~ 0.71-0.77)
- Colors 3-5× more important for Teff | Magnitudes 1.3-1.7× more important for log(g)
- **Physical**: Colors encode temp (Wien's law), magnitudes encode luminosity → size

**Recommendations**:
- Teff models: Emphasize color features
- log(g) models: Emphasize magnitude features
- Multi-output: Weight features appropriately per target

## Directory Structure

```
├── scripts/                 # Processing & ML scripts
│   ├── convert_ecsv_to_parquet.py, extract_*, clean_*, calculate_*
│   ├── train_*, predict_*, validate_*, crossmatch_*
│   ├── create_unified_feature_dataset.py  # RECOMMENDED workflow
│   └── create_ensemble_panstarrs_unified.py
├── src/                     # Source modules
│   ├── config/, data/, visualization/, features/
│   ├── pipeline/            # ML pipeline orchestration
│   └── notebook_utils.py
├── notebooks/               # Jupyter analysis notebooks
├── models/                  # Trained models + artifacts
├── data/                    # raw/, processed/, cache/
├── config/                  # models/, prediction/, validation/
├── docs/                    # Documentation (see list below)
├── reports/figures/         # Validation plots, correlation analysis
└── tests/
```

**Key Documentation**:
- `CONFIGURATION.md` - Config system API
- `PIPELINES.md`, `CONFIGURABLE_PIPELINE.md` - Pipeline guides
- `UNIFIED_FEATURES_WORKFLOW.md` - Unified features workflow
- `ALL_MODELS_COMPARISON.md`, `GAIA_TEFF_MODELS_COMPARISON.md` - Model comparisons
- `LOGG_UNCERTAINTY_PROPAGATION_ANALYSIS.md` - Uncertainty propagation workflow (NEW)
- `NOTEBOOK_UTILITIES.md`, `NOTEBOOK_CONVERSION.md` - Notebook guides
- `FEATURE_CORRELATION_ANALYSIS.md` - Feature correlation study

**Key Figure Directories**:
- `reports/figures/teff_uncertainty_analysis/` - Uncertainty propagation plots (4 plots)
- `reports/figures/three_model_comparison/` - Model comparison plots (4 plots)
- `reports/figures/correlation_analysis/` - Feature correlation plots
- `reports/figures/{model}_validation/` - Per-model validation plots (5-7 each)

## ML Best Practices

**Key Principles** (see `.cursorrules` for details):

1. **Configuration-Driven**: All paths/params in `config/config.yaml`, no hardcoded values
2. **DRY**: Reusable modules in `src/`, shared utilities in `src/notebook_utils.py`
3. **Pipeline Orchestration**: Single command to reproduce analysis (`pipeline.py`)
4. **Production Readiness**: Proper logging, error handling, versioned models, reproducible results

**Quick Patterns**:
```python
# Config usage
from src.config import get_config
config = get_config()
path = config.get_dataset_path('dataset_key', 'location')

# Notebook setup
from src.notebook_utils import load_ml_data, save_figure
data = load_ml_data(with_gaia=True)
save_figure(fig, 'analysis.png', subdir='exploratory')

# Pipeline execution
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml
```

## Unified Features Workflow

**Purpose**: Consistent feature engineering across training & prediction (eliminates distribution mismatch)

**Steps**:
1. `python scripts/create_unified_feature_dataset.py --model-type engineered`
2. `python scripts/train_model_unified_features.py --model-type engineered`
3. `python scripts/predict_unified_features.py --model models/rf_unified_*.pkl`
4. Validate: `jupyter lab notebooks/unified_features_no_gpsf_model_validation.ipynb`

**Benefits**: Same transforms for ALL objects, color-only avoids magnitude bias, validates distributions with KS tests

**See `docs/UNIFIED_FEATURES_WORKFLOW.md` for details**

## Recent Updates

### Corrected Teff Pipeline & Best-of-Three Ensemble (2025-12-09)
- **Problem**: Gaia GSP-Phot systematically underestimates Teff for stars >10000K
- **Solution**: Polynomial correction (degree 2) applied to training data before ML training
- **Correction coefficients**: `data/teff_correction_coeffs_deg2.pkl` (threshold 10000K, RMS 3209K)

**Pipeline Updates**:
- Added `ConvertLogPredictionsStep` to prediction pipeline
- Automatically converts BOTH predictions AND uncertainties from log space to Kelvin
- Formula: `Teff = 10^prediction`, `unc_kelvin = Teff × unc_log × ln(10)`

**Three Corrected Models Trained**:
1. **Gaia colors only** → log(Teff_corrected): 290K mean uncertainty
2. **Gaia + logg** → log(Teff_corrected): 277K mean uncertainty
3. **Gaia + clustering** → log(Teff_corrected): 346K mean uncertainty

**Best-of-Three Ensemble**:
- Selects prediction with lowest uncertainty for each object
- **Result**: 263K mean uncertainty (18% improvement vs single best model)
- Model distribution: 44.9% colors-only, 31.0% +logg, 24.1% +clustering
- Output: `teff_predictions_best_of_three_corrected.parquet` (847k predictions)

**Final Catalog**:
- **File**: `stars_types_with_best_predictions_corrected.fits` (196 MB)
- **Total objects**: 2.1M eclipsing binaries
- **Coverage**: 97.2% with Teff (58.3% Gaia original, 38.9% ML predictions)
- **Quality flags**: A (Gaia), B (ML<300K), C (ML<500K), D (ML≥500K), X (none)

**Key Findings**:
- Teff correction improves uncertainties by **18%** for color-only models
- Correction **hurts** performance when logg is included (-6.4%)
- **Recommendation**: Use correction only for photometry-only models
- Log transformation helps across all feature combinations (3.8-16% improvement)

**Scripts**:
- `apply_teff_correction_to_training_data.py` - Apply polynomial correction
- `create_final_catalog_with_corrected_teff.py` - Create best-of-three + merge with stars_types.dat
- `compare_model_uncertainties.py` - Compare RF uncertainties between models

### Target Transformation (2025-11-12)
- Added to `src/pipeline/configurable_ml_pipeline.py`
- Usage: `target_transform: "log"` in model config (options: `log`, `log2`, `ln`, `none`)
- Auto inverse-transforms predictions, reports metrics on original scale
- Stabilizes variance for targets spanning orders of magnitude

### Gaia Log-Transformed Model
- **Model**: `rf_gaia_all_colors_teff_log_20251112_162857`
- **Improvement**: 3.1% better overall MAE (575K → 557K)
- **Cool stars (<6000K)**: 8-11% better (75% of data)
- **Hot stars (>6000K)**: 3-5% worse (acceptable trade-off)
- More uniform error distribution (4.4% lower std)

### Feature Correlation Analysis
- **Dataset**: 1.27M stars from Gaia All Colors training
- **Findings**: Colors best for Teff (g_rp, bp_rp), magnitudes best for log(g) (rp, g, bp)
- **Visualizations**: `reports/figures/correlation_analysis/`
- **Impact**: Guides feature selection & explains why log transform helps

### Future Work Recommendations
1. Test log transform on log(g), [Fe/H] targets
2. Weight features per target in multi-output models
3. Consider separate models for cool (<6000K) vs hot (>6000K) stars
4. Create target-specific polynomial features

---

**For ML project patterns & templates, see `.cursorrules`**
