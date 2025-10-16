# Notebook Modules and Utilities - Complete Guide

## Overview

The project now includes **reusable modules** that eliminate hardcoded paths and duplicated code in notebooks.

### What Was Created

1. **`src/notebook_utils.py`** - Data loading and notebook utilities
2. **`src/features/engineering.py`** - Reusable feature engineering functions
3. **`examples/notebook_template.ipynb`** - Template showing best practices
4. **Documentation**:
   - `docs/NOTEBOOK_GUIDE.md` - Complete usage guide
   - `docs/NOTEBOOK_CONVERSION.md` - Migration guide for existing notebooks

---

## Quick Start for Notebooks

### Standard Setup (Top of Every Notebook)

```python
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
    TEST_SIZE,
    RANDOM_STATE
)

from src.features import engineer_all_features
```

### Load Data (No Paths Needed!)

```python
# Load eclipsing binary catalog
gaia_pm = load_eb_catalog(with_pm=True)

# Load Pan-STARRS with temperatures
panstarrs = load_panstarrs_data(with_temps=True)

# Load ML training data
ml_data = load_ml_data(with_gaia=True)

# Load a trained model
model = load_model()  # loads most recent
```

### Feature Engineering (Reusable!)

```python
# Define columns
color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
mag_cols = ['gPSFMag']

# Engineer all features at once
df_features = engineer_all_features(
    df,
    color_cols=color_cols,
    mag_cols=mag_cols
)

# This creates:
# - Polynomial features (x^2, x^3)
# - Interaction features (color1 * color2)
# - Log features
# - Temperature-dependent features
# - Magnitude features
```

### Save Figures (Auto-Location!)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)

# Automatically saves to reports/figures/
save_figure(fig, 'my_plot.png')

# Or to subdirectory
save_figure(fig, 'hr_diagram.png', subdir='hr_analysis')
```

---

## Available Functions

### Data Loading (`src/notebook_utils`)

| Function | Purpose | Example |
|----------|---------|---------|
| `load_eb_catalog()` | Load Gaia eclipsing binary catalog | `gaia = load_eb_catalog(with_pm=True)` |
| `load_panstarrs_data()` | Load Pan-STARRS photometry | `ps = load_panstarrs_data(with_temps=True)` |
| `load_ml_data()` | Load ML training data | `ml = load_ml_data(with_gaia=True)` |
| `load_colors_temperatures()` | Load colors and temps dataset | `ct = load_colors_temperatures()` |
| `load_model()` | Load trained model | `model = load_model()` |
| `get_config_value()` | Get config value | `val = get_config_value('ml', 'test_size')` |
| `get_path()` | Get path from config | `dir = get_path('processed')` |
| `save_figure()` | Save figure to reports/ | `save_figure(fig, 'plot.png')` |

### Feature Engineering (`src/features`)

| Function | Purpose |
|----------|---------|
| `engineer_all_features()` | Apply all transformations at once |
| `create_polynomial_features()` | Create x^2, x^3 features |
| `create_interaction_features()` | Create color1 * color2 features |
| `create_log_features()` | Create log(color) features |
| `create_temperature_dependent_features()` | Hot/cool star features |
| `create_magnitude_features()` | Magnitude polynomials |
| `select_best_features()` | Feature selection |
| `get_feature_importance()` | Get model feature importances |

### Constants (Pre-defined)

```python
MISSING_VALUE = -999.0    # Missing value indicator
COLOR_THRESHOLD = -0.5    # Color threshold for temps
TEST_SIZE = 0.2           # Train/test split
RANDOM_STATE = 42         # Random seed
```

---

## Migration Guide for Existing Notebooks

### Before (Old Way - DON'T DO THIS)

```python
# ❌ Hardcoded paths
from pathlib import Path
data_dir = Path('../data/processed')
gaia_pm = pd.read_parquet(data_dir / 'eb_catalog_with_pm.parquet')
panstarrs = pd.read_parquet(data_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet')

# ❌ Hardcoded values
missing_val = -999.0
test_size = 0.2

# ❌ Manual feature engineering (repeated everywhere)
df['g_r_squared'] = df['g_r_color'] ** 2
df['g_r_cubed'] = df['g_r_color'] ** 3
df['r_i_squared'] = df['r_i_color'] ** 2
df['g_r_x_r_i'] = df['g_r_color'] * df['r_i_color']
# ... 50 more lines of this ...

# ❌ Manual figure saving
fig_dir = Path('../reports/figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'plot.png', dpi=300, bbox_inches='tight')
```

### After (New Way - DO THIS)

```python
# ✅ Use utilities
import sys
sys.path.insert(0, '..')
from src.notebook_utils import (
    load_eb_catalog,
    load_panstarrs_data,
    save_figure,
    MISSING_VALUE,
    TEST_SIZE
)
from src.features import engineer_all_features

# ✅ Load data (no paths!)
gaia_pm = load_eb_catalog(with_pm=True)
panstarrs = load_panstarrs_data(with_temps=True)

# ✅ Use constants
# MISSING_VALUE and TEST_SIZE already available

# ✅ Reusable feature engineering
color_cols = ['g_r_color', 'r_i_color']
df_features = engineer_all_features(df, color_cols)

# ✅ Easy figure saving
save_figure(fig, 'plot.png')
```

---

## Benefits

### Code Reduction

**Before**: ~30 lines per notebook for setup and paths
**After**: ~10 lines

**Before**: ~100+ lines for feature engineering (duplicated in each notebook)
**After**: 1 function call (reusable across notebooks and pipelines)

### Portability

- ✅ Works on any machine (no path changes needed)
- ✅ Same code works in notebooks AND scripts
- ✅ Can be deployed in production pipelines

### Maintainability

- ✅ Update feature engineering in ONE place
- ✅ All notebooks use same logic
- ✅ Easy to add new features

### Consistency

- ✅ Same missing value everywhere
- ✅ Same test/train split
- ✅ Same feature engineering across all models

---

## File Structure

```
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration API
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py       # ✨ Feature engineering functions
│   ├── notebook_utils.py        # ✨ Data loading & utilities
│   └── ...
├── examples/
│   └── notebook_template.ipynb  # ✨ Template notebook
├── docs/
│   ├── NOTEBOOK_GUIDE.md        # ✨ Complete usage guide
│   └── NOTEBOOK_CONVERSION.md   # ✨ Migration guide
├── notebooks/
│   ├── *.ipynb                  # Your notebooks (to be updated)
│   └── ...
└── config/
    └── config.yaml              # Configuration file
```

---

## Examples

### Complete Notebook Example

See `examples/notebook_template.ipynb` for a working example showing all features.

### Quick Example: ML Model Training

```python
# Setup
import sys
sys.path.insert(0, '..')

from src.notebook_utils import load_ml_data, RANDOM_STATE, TEST_SIZE
from src.features import engineer_all_features, select_best_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
data = load_ml_data(with_gaia=True)

# Engineer features
color_cols = ['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp']
data_features = engineer_all_features(data, color_cols, mag_cols=['gPSFMag'])

# Prepare X, y
X = data_features.drop(columns=['Te_avg', 'original_ext_source_id'])
y = data_features['Te_avg']

# Select best features
X_selected, selector = select_best_features(X, y, k=20)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Train
model = RandomForestRegressor(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.0f} K")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

That's it! About 30 lines vs 200+ lines with manual feature engineering.

---

## Next Steps

### For New Notebooks

1. Use `examples/notebook_template.ipynb` as starting point
2. Import utilities at top
3. Use data loading functions
4. Use feature engineering functions
5. Use `save_figure()` for plots

### For Existing Notebooks

1. Read `docs/NOTEBOOK_CONVERSION.md`
2. Add setup cell with imports
3. Replace hardcoded paths with utility functions
4. Replace manual feature engineering with `engineer_all_features()`
5. Test notebook works

### For Pipelines

Same functions work in scripts! Example:

```python
# In a script (not notebook)
from src.notebook_utils import load_ml_data
from src.features import engineer_all_features

# Same code as notebook
data = load_ml_data(with_gaia=True)
data_features = engineer_all_features(data, color_cols, mag_cols)
# ... continue with pipeline
```

---

## Documentation

- **Usage Guide**: `docs/NOTEBOOK_GUIDE.md`
- **Conversion Guide**: `docs/NOTEBOOK_CONVERSION.md`
- **Configuration Guide**: `docs/CONFIGURATION.md`
- **Template**: `examples/notebook_template.ipynb`

---

## API Quick Reference

### Load Data
```python
gaia = load_eb_catalog(with_pm=True, format='pandas')
ps = load_panstarrs_data(with_temps=True)
ml = load_ml_data(with_gaia=True)
ct = load_colors_temperatures()
model = load_model()  # or load_model('model_id')
```

### Feature Engineering
```python
df_features = engineer_all_features(df, color_cols, mag_cols)
X_selected, selector = select_best_features(X, y, k=20)
importances = get_feature_importance(model, feature_names, top_n=10)
```

### Utilities
```python
val = get_config_value('section', 'key')
path = get_path('processed', ensure_exists=True)
save_figure(fig, 'plot.png', subdir='analysis', dpi=300)
```

### Constants
```python
MISSING_VALUE    # -999.0
COLOR_THRESHOLD  # -0.5
TEST_SIZE        # 0.2
RANDOM_STATE     # 42
```

---

## Summary

✅ **Created**: Reusable modules for notebooks
✅ **Eliminated**: Hardcoded paths and duplicated code
✅ **Provided**: Complete documentation and examples
✅ **Enabled**: Same code works in notebooks AND pipelines

**Result**: Cleaner, more maintainable, portable notebooks!
