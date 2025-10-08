# Notebook Guide - Using Configuration and Reusable Modules

## Overview

All notebooks now use:
1. **Configuration system** - No hardcoded paths
2. **Reusable modules** - Common operations in `src/`
3. **Feature engineering utilities** - Consistent transformations

This makes notebooks:
- ✅ Portable (work on any machine)
- ✅ Maintainable (update in one place)
- ✅ Reusable (same code for pipelines)
- ✅ Clean (less boilerplate)

## Quick Start

### Standard Notebook Setup

Every notebook should start with:

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project to path
import sys
sys.path.insert(0, '..')

# Import utilities
from src.notebook_utils import (
    load_eb_catalog,
    load_panstarrs_data,
    load_ml_data,
    load_model,
    save_figure,
    MISSING_VALUE,
    RANDOM_STATE
)

from src.features import engineer_all_features, select_best_features
```

## Data Loading

### No More Hardcoded Paths!

**Old way (DON'T DO THIS):**
```python
# ❌ Hardcoded path
data_dir = Path('../data/processed')
gaia_pm = pd.read_parquet(data_dir / 'eb_catalog_with_pm.parquet')
```

**New way (DO THIS):**
```python
# ✅ Use utility function
gaia_pm = load_eb_catalog(with_pm=True)
```

### Available Data Loaders

#### 1. Eclipsing Binary Catalog

```python
# With proper motion
gaia_pm = load_eb_catalog(with_pm=True)

# Basic catalog
gaia = load_eb_catalog(with_pm=False)

# As Polars DataFrame
gaia_pl = load_eb_catalog(format='polars')
```

#### 2. Pan-STARRS Data

```python
# With temperatures
panstarrs_temp = load_panstarrs_data(with_temps=True)

# Cleaned photometry only
panstarrs_clean = load_panstarrs_data(with_temps=False)
```

#### 3. ML Training Data

```python
# With Gaia colors (BP-RP)
ml_data = load_ml_data(with_gaia=True)

# Without Gaia colors
ml_data_basic = load_ml_data(with_gaia=False)
```

#### 4. Colors and Temperatures

```python
# Multi-band colors and temperatures
colors_temps = load_colors_temperatures()
```

#### 5. Trained Models

```python
# Load most recent model
model = load_model()

# With metadata
model, metadata = load_model(return_metadata=True)

# Specific model by ID
model = load_model('20251002_210423')
```

## Feature Engineering

### Use Reusable Functions

All feature engineering functions are in `src.features`:

```python
from src.features import engineer_all_features

# Define your columns
color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
mag_cols = ['gPSFMag']

# Create all features at once
df_features = engineer_all_features(
    df,
    color_cols=color_cols,
    mag_cols=mag_cols
)
```

### Individual Feature Types

```python
from src.features import (
    create_polynomial_features,
    create_interaction_features,
    create_log_features,
    create_temperature_dependent_features,
    create_magnitude_features
)

# Polynomial features (x^2, x^3)
df_poly = create_polynomial_features(df, color_cols, degree=3)

# Interaction features (color1 * color2)
df_int = create_interaction_features(df, color_cols)

# Log features
df_log = create_log_features(df, color_cols)

# Temperature-dependent features
df_temp = create_temperature_dependent_features(df, color_cols)

# Magnitude polynomial features
df_mag = create_magnitude_features(df, mag_cols)
```

### Feature Selection

```python
from src.features import select_best_features, get_feature_importance

# Select top 20 features
X_selected, selector = select_best_features(X, y, k=20)

# Get feature importances from model
importances = get_feature_importance(model, feature_names, top_n=10)
print(importances)
```

## Configuration Access

### Get Configuration Values

```python
from src.notebook_utils import get_config_value

# Get any config value
missing_val = get_config_value('processing', 'missing_value')
test_size = get_config_value('ml', 'test_size')
n_estimators = get_config_value('ml', 'rf_n_estimators')
```

### Use Constants

Pre-defined constants for common values:

```python
from src.notebook_utils import (
    MISSING_VALUE,      # -999.0
    COLOR_THRESHOLD,    # -0.5
    TEST_SIZE,          # 0.2
    RANDOM_STATE        # 42
)

# Use in your code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
```

### Get Paths

```python
from src.notebook_utils import get_path

# Get any path from config
data_dir = get_path('processed')
models_dir = get_path('models')

# Create if doesn't exist
figures_dir = get_path('figures', ensure_exists=True)
```

## Saving Figures

### Automatic Save Location

```python
from src.notebook_utils import save_figure

# Create your plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Save automatically to reports/figures/
save_figure(fig, 'my_plot.png')

# Save to subdirectory
save_figure(fig, 'hr_diagram.png', subdir='hr_analysis')

# Custom DPI
save_figure(fig, 'high_res.png', dpi=600)
```

## Migration Guide for Existing Notebooks

### Step 1: Add Setup Cell

Add this at the top:

```python
import sys
sys.path.insert(0, '..')

from src.notebook_utils import (
    load_eb_catalog,
    load_panstarrs_data,
    load_ml_data,
    MISSING_VALUE
)
```

### Step 2: Replace Data Loading

Find:
```python
data_dir = Path('../data/processed')
df = pd.read_parquet(data_dir / 'some_file.parquet')
```

Replace with:
```python
df = load_panstarrs_data()  # or appropriate loader
```

### Step 3: Replace Hardcoded Values

Find:
```python
missing_val = -999.0
test_size = 0.2
```

Replace with:
```python
from src.notebook_utils import MISSING_VALUE, TEST_SIZE
```

### Step 4: Use Feature Engineering Modules

Find:
```python
# Repeated feature engineering code in notebook
df['g_r_squared'] = df['g_r_color'] ** 2
df['g_r_x_r_i'] = df['g_r_color'] * df['r_i_color']
# ... more manual feature creation
```

Replace with:
```python
from src.features import engineer_all_features

df_features = engineer_all_features(
    df,
    color_cols=['g_r_color', 'r_i_color'],
    mag_cols=['gPSFMag']
)
```

## Example: Complete Notebook Cell

```python
# Setup
import sys
sys.path.insert(0, '..')

from src.notebook_utils import (
    load_ml_data,
    save_figure,
    MISSING_VALUE,
    TEST_SIZE,
    RANDOM_STATE
)
from src.features import engineer_all_features

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
print("Loading data...")
data = load_ml_data(with_gaia=True)
print(f"Loaded {len(data):,} objects")

# Prepare features
color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
mag_cols = ['gPSFMag']

# Engineer features
data_features = engineer_all_features(data, color_cols, mag_cols)

# Prepare X and y
feature_cols = [c for c in data_features.columns if c not in ['Te_avg', 'original_ext_source_id']]
X = data_features[feature_cols]
y = data_features['Te_avg']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Train model
model = RandomForestRegressor(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Plot results
fig, ax = plt.subplots()
y_pred = model.predict(X_test)
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('True Temperature (K)')
ax.set_ylabel('Predicted Temperature (K)')

# Save figure
save_figure(fig, 'predictions.png', subdir='temperature_models')

plt.show()
```

## Template Notebook

See `examples/notebook_template.ipynb` for a complete template showing all features.

## Benefits

### Before (Old Way)
- ❌ Hardcoded paths everywhere
- ❌ Repeated feature engineering code
- ❌ Manual path management
- ❌ Difficult to run on different machines
- ❌ Can't reuse in pipelines

### After (New Way)
- ✅ No hardcoded paths
- ✅ Reusable feature engineering
- ✅ Automatic path management
- ✅ Portable across machines
- ✅ Same code works in pipelines

## API Reference

See:
- `src/notebook_utils.py` - Data loading and utilities
- `src/features/engineering.py` - Feature engineering functions
- `docs/CONFIGURATION.md` - Configuration system details
- `examples/notebook_template.ipynb` - Working template
