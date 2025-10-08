# Notebook Conversion Guide

## Quick Conversion Checklist

For each notebook, follow these steps:

### 1. Add Setup Cell (at top)

```python
import sys
sys.path.insert(0, '..')

from src.notebook_utils import (
    load_eb_catalog,
    load_panstarrs_data,
    load_ml_data,
    load_colors_temperatures,
    load_model,
    save_figure,
    get_config_value,
    get_path,
    MISSING_VALUE,
    COLOR_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE
)

from src.features import (
    engineer_all_features,
    select_best_features,
    get_feature_importance
)
```

### 2. Find and Replace Patterns

#### Pattern 1: Data Directory
**Find:**
```python
data_dir = Path('../data/processed')
```

**Replace:**
```python
# Not needed anymore - use utility functions
```

#### Pattern 2: Gaia Catalog Loading
**Find:**
```python
data_dir = Path('../data/processed')
gaia_pm = pd.read_parquet(data_dir / 'eb_catalog_with_pm.parquet')
```

**Replace:**
```python
gaia_pm = load_eb_catalog(with_pm=True)
```

#### Pattern 3: Pan-STARRS Loading
**Find:**
```python
data_dir = Path('../data/processed')
panstarrs_temp = pd.read_parquet(data_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet')
```

**Replace:**
```python
panstarrs_temp = load_panstarrs_data(with_temps=True)
```

#### Pattern 4: ML Data Loading
**Find:**
```python
data_path = Path("../data/processed/ml_training_data_with_gaia.parquet")
data = pd.read_parquet(data_path)
```

**Replace:**
```python
data = load_ml_data(with_gaia=True)
```

#### Pattern 5: Models Directory
**Find:**
```python
models_dir = Path("../models")
model_file = models_dir / "rf_temperature_regressor_20251002_210423.pkl"
model = joblib.load(model_file)
```

**Replace:**
```python
model = load_model('20251002_210423')
# or for most recent:
model = load_model()
```

#### Pattern 6: Hardcoded Constants
**Find:**
```python
missing_val = -999.0
test_size = 0.2
random_state = 42
```

**Replace:**
```python
# Use imported constants
# MISSING_VALUE, TEST_SIZE, RANDOM_STATE already imported
```

#### Pattern 7: Figure Saving
**Find:**
```python
fig_dir = Path('../reports/figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'my_plot.png', dpi=300, bbox_inches='tight')
```

**Replace:**
```python
save_figure(fig, 'my_plot.png')
```

### 3. Feature Engineering Consolidation

If you have repeated feature engineering code:

**Find:**
```python
# Manual feature engineering
df['g_r_squared'] = df['g_r_color'] ** 2
df['g_r_cubed'] = df['g_r_color'] ** 3
df['r_i_squared'] = df['r_i_color'] ** 2
df['g_r_x_r_i'] = df['g_r_color'] * df['r_i_color']
df['g_r_x_bp_rp'] = df['g_r_color'] * df['bp_rp']
# ... many more lines
```

**Replace:**
```python
# Use reusable function
color_cols = ['g_r_color', 'r_i_color', 'bp_rp']
mag_cols = ['gPSFMag']

df_features = engineer_all_features(
    df,
    color_cols=color_cols,
    mag_cols=mag_cols
)
```

## Specific Notebook Conversions

### rf_regression_training.ipynb

**Key changes:**

1. Replace data loading:
```python
# OLD
data_path = Path("../data/processed/ml_training_data_with_gaia.parquet")
data = pd.read_parquet(data_path)

# NEW
data = load_ml_data(with_gaia=True)
```

2. Use constants:
```python
# OLD
test_size = 0.2
random_state = 42

# NEW
# Already imported: TEST_SIZE, RANDOM_STATE
```

3. Model saving location:
```python
# OLD
models_dir = Path("../models")
model_file = models_dir / f"model_{timestamp}.pkl"

# NEW
models_dir = get_path('models', ensure_exists=True)
model_file = models_dir / f"model_{timestamp}.pkl"
```

### rf_regression_feature_engineering.ipynb

**Key changes:**

1. Replace feature engineering:
```python
# OLD: Many lines of manual feature creation
# ...

# NEW: One function call
from src.features import engineer_all_features

df_features = engineer_all_features(
    data,
    color_cols=['g_r_color', 'r_i_color', 'B_V_color', 'bp_rp'],
    mag_cols=['gPSFMag']
)
```

2. Feature selection:
```python
# OLD: Manual SelectKBest code
# ...

# NEW:
from src.features import select_best_features

X_selected, selector = select_best_features(X, y, k=20)
```

### hierarchical_clustering_hr.ipynb

**Key changes:**

1. Data loading:
```python
# OLD
data_dir = Path('../data/processed')
gaia_pm = pd.read_parquet(data_dir / 'eb_catalog_with_pm.parquet')
panstarrs_temp = pd.read_parquet(data_dir / 'gaia_eb_panstarrs_phot_with_temperatures.parquet')

# NEW
gaia_pm = load_eb_catalog(with_pm=True)
panstarrs_temp = load_panstarrs_data(with_temps=True)
```

2. Figure saving:
```python
# OLD
fig.savefig('../reports/figures/dendrogram.png')

# NEW
save_figure(fig, 'dendrogram.png', subdir='clustering')
```

### rf_classification_*.ipynb

**Key changes:**

1. Data and constants:
```python
# OLD
data = pd.read_parquet("../data/processed/ml_training_data_with_gaia.parquet")
test_size = 0.2
random_state = 42

# NEW
data = load_ml_data(with_gaia=True)
# Use TEST_SIZE, RANDOM_STATE constants
```

## Testing After Conversion

After converting a notebook, test it:

### 1. Restart Kernel
In Jupyter: Kernel → Restart & Run All

### 2. Check for Errors
- No path-related errors
- All data loads correctly
- Figures save properly

### 3. Verify Output
- Same results as before
- All plots display correctly
- Models save/load correctly

## Common Issues and Solutions

### Issue 1: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Add at top of notebook:
```python
import sys
sys.path.insert(0, '..')
```

### Issue 2: Config File Not Found

**Error:**
```
FileNotFoundError: Configuration file not found
```

**Solution:**
Make sure `config/config.yaml` exists in project root. Check that notebook is in `notebooks/` directory.

### Issue 3: Data File Not Found

**Error:**
```
FileNotFoundError: ... not found
```

**Solution:**
1. Check file exists in the expected location
2. Check filename in `config/config.yaml` matches actual file
3. Make sure you're using correct dataset key

## Automated Conversion (Future)

Future improvement: Create a script to automatically convert notebooks:

```bash
python scripts/convert_notebook.py notebooks/old_notebook.ipynb
```

This would:
1. Add setup cell
2. Replace common patterns
3. Create backup of original
4. Validate converted notebook

## Summary

**What to do:**
1. ✅ Add setup cell with imports
2. ✅ Replace `Path('../data/...')` with `load_*()` functions
3. ✅ Replace hardcoded values with constants
4. ✅ Consolidate feature engineering
5. ✅ Use `save_figure()` for plots
6. ✅ Test notebook works

**What NOT to do:**
1. ❌ Keep hardcoded paths
2. ❌ Duplicate feature engineering code
3. ❌ Manual path construction
4. ❌ Skip testing after conversion

## Getting Help

- See `docs/NOTEBOOK_GUIDE.md` for full API reference
- See `examples/notebook_template.ipynb` for working example
- Check `src/notebook_utils.py` for available functions
- Check `src/features/engineering.py` for feature functions
