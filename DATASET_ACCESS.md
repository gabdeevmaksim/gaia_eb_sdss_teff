# Dataset Access Guide

Complete guide for downloading and using the Gaia EB Teff datasets and pre-trained models.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Datasets Available](#datasets-available)
- [Download Methods](#download-methods)
- [Pre-trained Models](#pre-trained-models)
- [Authentication](#authentication)
- [Dataset Structure](#dataset-structure)
- [Usage Examples](#usage-examples)

## Overview

All datasets and models are hosted on HuggingFace Hub for easy access, version control, and distribution.

**Repositories**:
- **Datasets**: [YOUR_ORG/gaia-eb-teff-datasets](https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets)
- **Models**: [YOUR_ORG/gaia-eb-teff-models](https://huggingface.co/models/YOUR_ORG/gaia-eb-teff-models)

## Quick Start

```bash
# Install HuggingFace Hub
pip install huggingface_hub

# Set token (optional for public datasets)
export HF_TOKEN=your_token_here

# Download using provided script
python scripts/download_datasets.py --datasets training
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

## Datasets Available

### 1. Final Catalogs

#### stars_types_with_best_predictions.fits (196 MB)

**Description**: Complete catalog of 2.1M eclipsing binary stars with Teff predictions

**Content**:
- 2,117,932 eclipsing binaries
- 97.2% coverage with Teff predictions
- Best-of-three ensemble (lowest uncertainty per object)

**Columns**:
- `source_id`: Gaia DR3 source identifier
- `teff_best`: Best Teff prediction (Kelvin)
- `unc_best`: Uncertainty in Teff (Kelvin)
- `best_model`: Model source (teff_only/teff_logg/teff_cluster)
- `quality_flag`: Quality indicator
  - `A`: Gaia GSP-Phot (original)
  - `B`: ML prediction with uncertainty < 300K
  - `C`: ML prediction with uncertainty < 500K
  - `D`: ML prediction with uncertainty ≥ 500K
  - `X`: No Teff available

**Quality Distribution**:
- A (Gaia): 58.3% (1.23M stars)
- B (ML<300K): 18.1% (383k stars)
- C (ML<500K): 15.4% (326k stars)
- D (ML≥500K): 5.4% (114k stars)
- X (None): 2.8% (59k stars)

**Mean Uncertainty**: 263K (18% improvement over single models)

### 2. Training Datasets

#### gaia_all_colors_teff_corrected.parquet (44 MB)

**Description**: Main training dataset with corrected Gaia Teff

**Content**:
- 1,273,456 training stars
- Polynomial correction applied for Teff > 10,000K
- 6 Gaia colors + 3 bands

**Features**:
- Colors: `bp_rp`, `g_rp`, `g_bp`, `bp_g`, `rp_g`, `g_r`
- Bands: `bp`, `rp`, `g`
- Target: `teff_corrected`

**Use Case**: Training Gaia-only models (no 2MASS or Pan-STARRS required)

#### gaia_all_colors_train.parquet (36 MB)

**Description**: Original Gaia training data (without correction)

**Content**: Same as above but with original `teff_gspphot` values

**Use Case**: Comparison studies, testing correction impact

#### gaia_all_colors_train_with_logg.parquet (48 MB)

**Description**: Extended training data including log(g) parameter

**Additional Columns**:
- `logg`: Surface gravity [dex]
- `logg_uncertainty`: Uncertainty in log(g)

**Use Case**: Training multi-parameter models (Teff + log(g))

### 3. Prediction Datasets

#### gaia_all_colors_predict.parquet (21 MB)

**Description**: Preprocessed Gaia photometry for eclipsing binaries without Teff

**Content**: 847,000 stars ready for prediction

**Use Case**: Making predictions on new eclipsing binaries

#### eb_2mass_photometry.parquet (69 MB)

**Description**: Eclipsing binaries with 2MASS infrared photometry

**Content**: Cross-matched Gaia + 2MASS data

**Features**: Optical (Gaia) + Infrared (2MASS J, H, K bands)

**Use Case**: Training/predicting with infrared colors

## Download Methods

### Method 1: Using Provided Script (Recommended)

```bash
# Download training datasets
python scripts/download_datasets.py --datasets training

# Download catalog
python scripts/download_datasets.py --datasets catalog

# Download all datasets
python scripts/download_datasets.py --datasets all

# Download specific model
python scripts/download_datasets.py --model gaia_teff_corrected_log

# Download all models
python scripts/download_datasets.py --model all
```

**Advantages**: Automatic verification, organized output, error handling

### Method 2: HuggingFace Hub Python API

```python
from huggingface_hub import hf_hub_download

# Download catalog
catalog_path = hf_hub_download(
    repo_id="YOUR_ORG/gaia-eb-teff-datasets",
    filename="catalogs/stars_types_with_best_predictions.fits",
    repo_type="dataset"
)

# Download training data
train_path = hf_hub_download(
    repo_id="YOUR_ORG/gaia-eb-teff-datasets",
    filename="training/gaia_all_colors_teff_corrected.parquet",
    repo_type="dataset"
)

# Load in Python
from astropy.table import Table
catalog = Table.read(catalog_path)

import polars as pl
train_data = pl.read_parquet(train_path)
```

### Method 3: HuggingFace CLI

```bash
# Install CLI
pip install huggingface_hub[cli]

# Login (optional for public datasets)
huggingface-cli login

# Download specific file
huggingface-cli download YOUR_ORG/gaia-eb-teff-datasets \
    catalogs/stars_types_with_best_predictions.fits \
    --repo-type dataset \
    --local-dir data/processed

# Download all training data
huggingface-cli download YOUR_ORG/gaia-eb-teff-datasets \
    --include "training/*" \
    --repo-type dataset \
    --local-dir data/processed
```

### Method 4: Git LFS (For Complete Repository)

```bash
# Clone entire dataset repository
git clone https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets

# Or sparse checkout (specific files only)
git clone --no-checkout https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets
cd gaia-eb-teff-datasets
git sparse-checkout set catalogs/
git checkout
```

**Note**: Requires Git LFS for large files. Install with: `git lfs install`

### Method 5: Direct URL Download

```bash
# Catalog
wget https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/catalogs/stars_types_with_best_predictions.fits

# Training data
wget https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/training/gaia_all_colors_teff_corrected.parquet

# Using curl
curl -L -o catalog.fits https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets/resolve/main/catalogs/stars_types_with_best_predictions.fits
```

## Pre-trained Models

All models are available in the [model repository](https://huggingface.co/models/YOUR_ORG/gaia-eb-teff-models).

### Available Models

#### 1. gaia_teff_corrected_log (RECOMMENDED)

**File**: `rf_gaia_teff_corrected_log_20251126_130144.pkl` (1.2 GB)

**Description**: Best overall model - log-transformed Gaia colors with corrected Teff

**Performance**:
- MAE: 556.9K
- RMSE: 1021.3K
- R²: 0.640
- Within 10%: 68.5%

**Features**: 6 Gaia colors + 3 bands (BP, RP, G)

**Training**: 1.27M stars, corrected for Teff > 10,000K

**Download**:
```bash
python scripts/download_datasets.py --model gaia_teff_corrected_log
```

#### 2. gaia_2mass_ir

**File**: `rf_gaia_2mass_ir_20251103_141119.pkl` (1.2 GB)

**Description**: Gaia optical + 2MASS infrared photometry

**Performance**:
- MAE: 765.1K
- RMSE: 1168.4K
- R²: 0.315
- Within 10%: 43.4%

**Features**: Gaia colors + 2MASS J, H, K bands

**Use Case**: When infrared data is available

#### 3. gaia_all_colors_teff_log

**File**: `rf_gaia_all_colors_teff_log_20251112_162857.pkl` (2.0 GB)

**Description**: All Gaia colors, log-transformed target

**Performance**: Similar to gaia_teff_corrected_log

**Difference**: Uses uncorrected Gaia Teff (for comparison)

### Model Registry

All models are cataloged in `config/models/model_registry.yaml`:

```yaml
models:
  gaia_teff_corrected_log:
    file: "rf_gaia_teff_corrected_log_20251126_130144.pkl"
    url: "https://huggingface.co/.../rf_gaia_teff_corrected_log_20251126_130144.pkl"
    checksum: "sha256:..."
    mae_kelvin: 556.9
    features: [bp_rp, g_rp, g_bp, bp_g, rp_g, bp, rp, g]
```

### Model Files

Each model includes:
- **`.pkl`**: Trained model (scikit-learn RandomForest)
- **`_metadata.json`**: Features, hyperparameters, training info
- **`_SUMMARY.txt`**: Human-readable performance summary

## Authentication

### Public Datasets

Most datasets are **public** and don't require authentication. You can download without a token.

### Private Datasets / Models

If datasets are private, you'll need a HuggingFace token:

#### Option 1: CLI Login

```bash
huggingface-cli login
# Enter your token when prompted
```

#### Option 2: Environment Variable

```bash
export HF_TOKEN=your_huggingface_token_here
```

#### Option 3: Python Code

```python
from huggingface_hub import login
login(token="your_token_here")
```

#### Getting a Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access sufficient)
3. Copy and save securely

## Dataset Structure

Complete structure on HuggingFace Hub:

```
gaia-eb-teff-datasets/
├── README.md                  # Dataset card
├── catalogs/
│   ├── stars_types_with_best_predictions.fits         (196 MB)
│   └── stars_types_with_best_predictions_DESCRIPTION.txt
├── training/
│   ├── gaia_all_colors_teff_corrected.parquet         (44 MB)
│   ├── gaia_all_colors_train.parquet                  (36 MB)
│   └── gaia_all_colors_train_with_logg.parquet        (48 MB)
└── prediction/
    ├── gaia_all_colors_predict.parquet                (21 MB)
    └── eb_2mass_photometry.parquet                    (69 MB)

gaia-eb-teff-models/
├── README.md                  # Model card
├── model_registry.yaml        # Model manifest
├── rf_gaia_teff_corrected_log_20251126_130144.pkl    (1.2 GB)
├── rf_gaia_teff_corrected_log_20251126_130144_metadata.json
├── rf_gaia_teff_corrected_log_20251126_130144_SUMMARY.txt
├── rf_gaia_2mass_ir_20251103_141119.pkl              (1.2 GB)
├── rf_gaia_2mass_ir_20251103_141119_metadata.json
├── rf_gaia_2mass_ir_20251103_141119_SUMMARY.txt
└── ... (other models)
```

## Usage Examples

### Example 1: Load Catalog in Python

```python
from astropy.table import Table
import numpy as np

# Load catalog
catalog = Table.read('catalogs/stars_types_with_best_predictions.fits')

# Filter by quality
high_quality = catalog[catalog['quality_flag'].isin(['A', 'B'])]

# Filter by uncertainty
low_unc = catalog[catalog['unc_best'] < 300]

# Statistics
print(f"Total stars: {len(catalog)}")
print(f"Mean Teff: {np.nanmean(catalog['teff_best']):.0f} K")
print(f"Mean uncertainty: {np.nanmean(catalog['unc_best']):.0f} K")
```

### Example 2: Train Custom Model

```python
import polars as pl
from sklearn.ensemble import RandomForestRegressor

# Load training data
train = pl.read_parquet('training/gaia_all_colors_teff_corrected.parquet')

# Prepare features
features = ['bp_rp', 'g_rp', 'g_bp', 'bp_g', 'rp_g', 'g_r']
X = train[features].to_numpy()
y = train['teff_corrected'].to_numpy()

# Train model
model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
model.fit(X, y)

# Save model
import joblib
joblib.dump(model, 'my_custom_model.pkl')
```

### Example 3: Make Predictions

```python
import joblib
import polars as pl

# Load pre-trained model
model = joblib.load('models/rf_gaia_teff_corrected_log_20251126_130144.pkl')

# Load new data
new_data = pl.read_parquet('prediction/gaia_all_colors_predict.parquet')

# Prepare features (same as training)
features = ['bp_rp', 'g_rp', 'g_bp', 'bp_g', 'rp_g', 'bp', 'rp', 'g']
X_new = new_data[features].to_numpy()

# Make predictions
predictions = model.predict(X_new)

# Note: If model uses log transform, convert back
# predictions_kelvin = 10 ** predictions

# Add to dataframe
new_data = new_data.with_columns(
    pl.Series('teff_predicted', predictions)
)
```

### Example 4: Docker Auto-Download

```bash
# Datasets download automatically in Docker containers
export HF_TOKEN=your_token
docker compose run --rm train \
  --ml --ml-config config/models/gaia_teff_corrected_log.yaml
# Container downloads training data if not present
```

## File Formats

### Parquet Files

**Advantages**: Fast I/O, columnar storage, efficient compression

**Reading**:
```python
# Polars (recommended)
import polars as pl
data = pl.read_parquet('file.parquet')

# Pandas
import pandas as pd
data = pd.read_parquet('file.parquet')

# PyArrow
import pyarrow.parquet as pq
table = pq.read_table('file.parquet')
```

### FITS Files

**Advantages**: Standard astronomy format, header metadata

**Reading**:
```python
# Astropy (recommended)
from astropy.table import Table
catalog = Table.read('file.fits')

# Astropy.io.fits
from astropy.io import fits
hdul = fits.open('file.fits')
data = hdul[1].data  # Binary table
```

## License

All datasets and models are released under **CC BY 4.0** license.

You are free to:
- Share and redistribute
- Adapt and build upon
- Use for commercial purposes

With attribution required.

## Citation

If you use these datasets or models, please cite:

```bibtex
@article{your_paper,
  title={Effective Temperature Predictions for Eclipsing Binary Stars},
  author={Your Name},
  journal={Journal},
  year={2025},
  note={Dataset: https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets}
}
```

## Support

For issues with datasets or models:
- **GitHub Issues**: https://github.com/YOUR_ORG/gaia-eb-teff/issues
- **HuggingFace Discussions**: Use the discussion tab on the dataset page
- **Email**: your.email@example.com

## Updates

Dataset versions are tracked on HuggingFace Hub. Check the repository for updates:
- https://huggingface.co/datasets/YOUR_ORG/gaia-eb-teff-datasets
- https://huggingface.co/models/YOUR_ORG/gaia-eb-teff-models
