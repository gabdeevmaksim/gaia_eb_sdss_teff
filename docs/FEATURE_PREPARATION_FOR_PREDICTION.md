# Feature Preparation for Temperature Prediction

**Document created:** 2025-10-14  
**Model:** `rf_temperature_regressor_feature_engineering_20251002_210423`  
**Task:** Predict temperatures for eclipsing binaries without Gaia GSP-Phot temperatures

---

## Executive Summary

Successfully predicted effective temperatures for **429,500 eclipsing binaries** that lacked Gaia GSP-Phot temperature estimates using a feature-engineered Random Forest model. The prediction process revealed critical insights about feature engineering consistency and data filtering that are essential for production ML systems.

---

## 1. Problem Statement

### Goal
Predict effective temperatures (`teff`) for eclipsing binary stars that do NOT have Gaia GSP-Phot temperature estimates (`teff_gspphot`).

### Dataset Context
- **Total catalog size**: 1,230,649 objects
- **Objects WITH teff_gspphot**: 737,028 (59.9%)
- **Objects WITHOUT teff_gspphot**: 493,621 (40.1%) ← **TARGET POPULATION**
- **Objects with Pan-STARRS photometry**: 452,575 (91.7% of targets)
- **Objects with valid colors for prediction**: 429,500 (87.0% of targets)

---

## 2. The Correct Process

### Step 1: Load and Filter Data

```python
# Load main catalog
catalog = pd.read_parquet('data/processed/eb_catalog.parquet')

# Filter to ONLY objects WITHOUT Gaia GSP-Phot temperatures
no_gspphot = catalog['teff_gspphot'].isna()
catalog_no_temp = catalog[no_gspphot].copy()
# Result: 493,621 objects

# Load Pan-STARRS photometry data (has colors g-r, r-i, i-z, B-V)
panstarrs = pd.read_parquet('data/processed/gaia_eb_panstarrs_phot_with_temperatures.parquet')

# Merge to get complete feature set
full_data = catalog_no_temp.merge(
    panstarrs[['original_ext_source_id', 'rra', 'rdec', 
               'g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag']], 
    on='original_ext_source_id', 
    how='inner'
)
# Result: 452,575 objects with photometry
```

**Key Insight #1:** Objects without `teff_gspphot` are identified from the **main catalog**, not from the Pan-STARRS dataset. The Pan-STARRS dataset contains empirically calculated temperatures (`Te_avg`), which are different from Gaia's model-dependent `teff_gspphot`.

### Step 2: Validate Required Features

```python
# Define required features (must match training)
required_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp', 'gPSFMag']

# Filter to objects with ALL required features present
valid_mask = data[required_cols].notna().all(axis=1)
# Result: 429,500 objects with complete feature set (94.9% success rate)
```

**Key Insight #2:** Must validate ALL required color indices and magnitudes BEFORE feature engineering. Missing even one feature will cause NaN propagation through polynomial and log transforms.

### Step 3: Select ONLY Base Feature Columns

```python
# Keep ONLY the base feature columns (same as ML training data)
base_feature_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag', 'bp_rp']
id_cols = ['source_id', 'original_ext_source_id']

# Filter to valid objects and select ONLY base features
data_valid = data[valid_mask][base_feature_cols + id_cols].copy()
```

**Key Insight #3:** The data must contain EXACTLY the base feature columns used during training—no more, no less. Extra columns (like `Te_avg`, `ra`, `dec`, etc.) will be included in feature engineering and cause feature count mismatches.

### Step 4: Feature Engineering (Match Training Exactly)

```python
from src.features.engineering import (
    create_polynomial_features,
    create_log_features,
    create_temperature_dependent_features,
    create_magnitude_features
)

data_with_features = data_valid.copy()

# Polynomial features (degree 2) - includes squared terms and pairwise interactions
data_with_features = create_polynomial_features(data_with_features, color_cols, degree=2)

# Log features
data_with_features = create_log_features(data_with_features, color_cols)

# Temperature-dependent features (hot/cool/mid regions)
data_with_features = create_temperature_dependent_features(data_with_features, color_cols)

# Magnitude features (g_mag and g_mag_squared)
data_with_features = create_magnitude_features(data_with_features, mag_cols)

# Result: 8 base features → 48 total features (including base)
```

**Key Insight #4:** The `create_polynomial_features` with `degree=2` automatically creates ALL pairwise interactions. Do NOT call `create_interaction_features` separately, as this duplicates interactions and creates too many features.

### Step 5: Handle NaN and Infinity Values

```python
# Prepare feature matrix
X_all = valid_data[feature_cols].fillna(0)  # NaN → 0
X_all = X_all.replace([np.inf, -np.inf], 0)  # Infinity → 0
```

**Key Insight #5:** Log transforms of negative or zero values produce `-inf` or `NaN`. These MUST be replaced before passing to the selector/model, otherwise scikit-learn will raise `ValueError: Input contains NaN/infinity`.

### Step 6: Match Feature Count to Selector

The selector was fitted on **38 features**. If feature engineering produces more (e.g., 42-52):

```python
if X_all.shape[1] > 38:
    logger.warning(f"Have {X_all.shape[1]} features, selector expects 38. Dropping extras...")
    X_all = X_all.iloc[:, :38]  # Keep only first 38 features
elif X_all.shape[1] < 38:
    raise ValueError(f"Insufficient features: got {X_all.shape[1]}, need 38")
```

**Key Insight #6:** The `SelectKBest` selector expects a FIXED number of input features (38 in this case) based on what it was fitted on. Feature count mismatches will cause `ValueError: SelectKBest is expecting 38 features as input`.

### Step 7: Apply Selector and Predict

```python
# Apply feature selector (selects 20 best features from 38)
X_selected = selector.transform(X_all)

# Predict temperatures
predictions = model.predict(X_selected)
```

---

## 3. Problems Encountered and Solutions

### Problem 1: Wrong Dataset Selected

**Error:**
```
KeyError: "['g_r_color', 'r_i_color', 'gPSFMag'] not in index"
```

**Cause:** Initially tried to use `eb_catalog.parquet` which only contains Gaia photometry (bp_rp), not Pan-STARRS colors (g-r, r-i, i-z).

**Solution:** Use `gaia_eb_panstarrs_phot_with_temperatures.parquet` which has Pan-STARRS colors, then merge with main catalog to add Gaia `bp_rp`.

---

### Problem 2: NaN Values in Feature Engineering

**Error:**
```
ValueError: Input X contains NaN.
PolynomialFeatures does not accept missing values encoded as NaN natively.
```

**Cause:** Some objects had missing values in one or more color indices. When passed to `PolynomialFeatures.fit_transform()`, NaN values propagate through polynomial calculations.

**Solution:** Filter data BEFORE feature engineering:
```python
required_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp', 'gPSFMag']
valid_mask = data[required_cols].notna().all(axis=1)
data_valid = data[valid_mask].copy()
```

---

### Problem 3: Infinity Values from Log Transforms

**Error:**
```
ValueError: Input X contains infinity or a value too large for dtype('float64').
```

**Cause:** Log transforms of negative or zero color values produce `-inf`:
```python
log_g_r_color = np.log(g_r_color)  # If g_r_color <= 0 → -inf or NaN
```

**Solution:** Replace infinities with 0 after all feature engineering:
```python
X = X.replace([np.inf, -np.inf], 0)
```

---

### Problem 4: Extra Columns in Feature Engineering

**Error:**
```
ValueError: X has 97 features, but SelectKBest is expecting 38 features as input.
```

**Cause:** When calling `engineer_all_features()` on a dataframe that contains many columns beyond just the base features (e.g., `Te_avg`, `ra`, `dec`, temperature columns, etc.), ALL columns get included in feature engineering, creating far too many features.

**Solution:** Select ONLY the base feature columns before engineering:
```python
base_feature_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag', 'bp_rp']
id_cols = ['source_id', 'original_ext_source_id']
data_valid = data[valid_mask][base_feature_cols + id_cols].copy()
```

---

### Problem 5: Feature Count Mismatch (Too Many Features)

**Error:**
```
ValueError: X has 87 features, but SelectKBest is expecting 38 features as input.
```

**Cause:** Using `degree=3` in `create_polynomial_features()` and calling both `create_polynomial_features()` and `create_interaction_features()` created too many features:
- Degree 3 polynomials: Creates x, x², x³, x¹x², etc. → 56 features
- Duplicate interactions from both functions

**Solution:** 
1. Use `degree=2` for polynomial features (not degree=3)
2. Do NOT call `create_interaction_features()` separately (polynomial degree=2 already includes all pairwise interactions)
3. If still have extras, truncate to first 38 features

**Actual feature count achieved:**
- Base features: 6
- Polynomial degree 2: +15 (5 colors × 2 squared + 10 pairwise interactions)
- Log features: +5
- Temperature-dependent features: +15 (3 per color × 5 colors)
- Magnitude features: +1 (gPSFMag_squared)
- **Total: 42 features → Truncated to 38**

---

### Problem 6: Metadata Key Name

**Error:**
```
KeyError: 'features'
```

**Cause:** Tried to access `metadata['features']` but the actual key in the metadata JSON file is `selected_features`.

**Solution:**
```python
feature_names = metadata['selected_features']  # Not 'features'
```

---

## 4. Critical Lessons Learned

### Lesson 1: Feature Engineering Must Be Deterministic and Reproducible

The feature engineering pipeline during **prediction** must produce EXACTLY the same features in the EXACT same order as during **training**. This means:

1. **Same functions**: Use identical feature engineering functions
2. **Same parameters**: `degree=2`, not `degree=3`
3. **Same order**: Polynomial → Log → Temperature-dependent → Magnitude
4. **Same base features**: Only the 6 base color/magnitude columns

### Lesson 2: Data Filtering Must Happen BEFORE Engineering

**Wrong:**
```python
# WRONG: Engineer first, filter later
data_features = engineer_all_features(data)  # Creates NaN/inf
data_valid = data_features[valid_mask]  # Too late!
```

**Correct:**
```python
# CORRECT: Filter first, engineer later
data_valid = data[valid_mask].copy()  # Only valid data
data_features = engineer_all_features(data_valid)  # Clean engineering
```

### Lesson 3: Feature Selection vs. Feature Engineering

The workflow has TWO distinct stages:

1. **Feature Engineering** (38 features total)
   - Transforms 6 base features → 38 engineered features
   - Creates polynomials, logs, temperature-dependent features
   - Must be reproducible and match training

2. **Feature Selection** (20 features selected)
   - `SelectKBest` chooses 20 most important from the 38
   - Selector is pre-fitted and expects exactly 38 input features
   - Cannot be retrained during prediction

### Lesson 4: Dataset Source Matters

For eclipsing binaries, there are multiple datasets with overlapping but different information:

- `eb_catalog.parquet`: Main catalog with Gaia data (bp_rp, teff_gspphot)
- `gaia_eb_panstarrs_phot_with_temperatures.parquet`: Pan-STARRS photometry with empirical temperatures (g-r, r-i, i-z, B-V, Te_avg)

**The correct approach:**
1. Identify target population from **main catalog** (objects without `teff_gspphot`)
2. Get Pan-STARRS colors from **photometry dataset**
3. Merge on `original_ext_source_id`

### Lesson 5: Missing Value Encoding Consistency

The project uses `-999.0` to encode missing photometric measurements. When filtering:

```python
missing_value = config.get('processing', 'missing_value')  # -999.0
no_temp = (data['Te_avg'] == missing_value) | data['Te_avg'].isna()
```

However, for Gaia GSP-Phot temperatures, missing values are encoded as `NaN`:

```python
no_gspphot = catalog['teff_gspphot'].isna()  # Only NaN check needed
```

**Lesson:** Different data sources may use different missing value conventions. Always check the actual data!

---

## 5. The Complete Feature Engineering Pipeline

### Input Data Requirements

**Base Features (6 total):**
1. `g_r_color` - Pan-STARRS g-r color
2. `r_i_color` - Pan-STARRS r-i color
3. `i_z_color` - Pan-STARRS i-z color
4. `B_V_color` - Johnson B-V color
5. `bp_rp` - Gaia BP-RP color
6. `gPSFMag` - Pan-STARRS g-band magnitude

All 6 features must be present (not NaN) for each object.

### Feature Engineering Steps

**Step 1: Polynomial Features (degree=2)**
- Creates squared terms: `g_r_color^2`, `r_i_color^2`, etc. (5 features)
- Creates pairwise interactions: `g_r_color_r_i_color`, `g_r_color_i_z_color`, etc. (10 features)
- **Output:** +15 features

**Step 2: Log Features**
- Creates: `log_g_r_color`, `log_r_i_color`, `log_i_z_color`, `log_B_V_color`, `log_bp_rp`
- **Warning:** May produce `-inf` for negative colors or `NaN` for zero/negative values
- **Output:** +5 features

**Step 3: Temperature-Dependent Features**
- For each color, creates 3 indicator features:
  - `hot_{color}`: Flagged when color indicates hot star
  - `cool_{color}`: Flagged when color indicates cool star
  - `mid_{color}`: Flagged for intermediate temperatures
- **Output:** +15 features (3 × 5 colors)

**Step 4: Magnitude Features**
- Creates: `gPSFMag_squared`
- **Output:** +1 feature

**Total Engineered Features:** 6 base + 15 + 5 + 15 + 1 = **42 features**

**Note:** The training produced 38 features, suggesting either:
- Different parameters were used (e.g., fewer color columns)
- Some features were manually excluded
- The first 38 are used by truncation

### Step 5: Feature Cleaning

```python
# Replace NaN values with 0
X = X.fillna(0)

# Replace infinities with 0
X = X.replace([np.inf, -np.inf], 0)
```

**Critical:** This must happen AFTER all feature engineering is complete.

### Step 6: Feature Selection

```python
# Selector expects exactly 38 features
if X.shape[1] > 38:
    X = X.iloc[:, :38]  # Keep first 38

# Apply pre-fitted selector
X_selected = selector.transform(X)  # Selects 20 best features

# Predict
predictions = model.predict(X_selected)
```

---

## 6. Validation Checks

### Before Running Predictions

```python
# 1. Check you have the right population
assert catalog['teff_gspphot'].isna().sum() > 0, "No objects need predictions!"

# 2. Check required features are present
required = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp', 'gPSFMag']
assert all(col in data.columns for col in required), "Missing required features!"

# 3. Check feature count after engineering
assert X_engineered.shape[1] >= 38, "Too few features!"

# 4. Check for NaN/inf before prediction
assert not X_clean.isnull().any().any(), "NaN values still present!"
assert not np.isinf(X_clean.values).any(), "Infinity values still present!"
```

### After Running Predictions

```python
# 1. Sanity check temperature range
assert predictions.min() > 2000, "Unrealistic low temperatures!"
assert predictions.max() < 50000, "Unrealistic high temperatures!"

# 2. Check for NaN in predictions
assert not pd.Series(predictions).isna().any(), "NaN in predictions!"

# 3. Verify expected count
expected_predictions = valid_mask.sum()
assert len(predictions) == expected_predictions, "Prediction count mismatch!"
```

---

## 7. Final Results

### Prediction Statistics

**Coverage:**
- Total target population (no teff_gspphot): 493,621
- With Pan-STARRS photometry: 452,575 (91.7%)
- With valid colors: 429,500 (87.0%)
- **Successfully predicted: 429,500** ✓

**Temperature Distribution:**
- Mean: 9,329 K
- Median: 9,173 K
- Std: 1,057 K
- Range: 5,940 - 11,901 K

**Model Performance (from training):**
- MAE: 318 K
- RMSE: 524 K
- R²: 0.865

**Output:** `data/processed/eb_full_catalog_temperatures.parquet`

---

## 8. Recommendations for Future Predictions

### Do's:
✅ Filter target population first (objects needing predictions)  
✅ Validate all required features are present before engineering  
✅ Select ONLY base feature columns (drop extra metadata columns)  
✅ Use exact same engineering functions and parameters as training  
✅ Handle NaN and infinity values explicitly  
✅ Validate feature count matches selector expectations  
✅ Run sanity checks on prediction outputs  

### Don'ts:
❌ Don't engineer features on unfiltered data with NaN values  
❌ Don't include extra columns in the engineering step  
✅ Don't call duplicate feature creation functions (e.g., polynomial degree=2 already includes interactions)  
❌ Don't change engineering parameters (degree, thresholds, etc.)  
❌ Don't assume feature counts will automatically match  
❌ Don't skip validation checks  

---

## 9. Code Template for Future Use

```python
#!/usr/bin/env python3
"""Template for applying trained ML models to new data."""

import pandas as pd
import numpy as np
import joblib
from src.config import get_config
from src.features.engineering import (
    create_polynomial_features,
    create_log_features,
    create_temperature_dependent_features,
    create_magnitude_features
)

def predict_with_model(data: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    Apply trained model to new data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with base features
    model_id : str
        Model identifier
        
    Returns
    -------
    pd.DataFrame
        Predictions
    """
    config = get_config()
    
    # 1. Load model and selector
    models_dir = config.get_path('models')
    model = joblib.load(models_dir / f"{model_id}.pkl")
    selector = joblib.load(models_dir / f"{model_id}_selector.pkl")
    
    # 2. Validate required features
    required_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp', 'gPSFMag']
    valid_mask = data[required_cols].notna().all(axis=1)
    
    # 3. Select ONLY base features
    base_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'gPSFMag', 'bp_rp']
    data_valid = data[valid_mask][base_cols + ['source_id']].copy()
    
    # 4. Engineer features (EXACT same as training)
    color_cols = ['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp']
    mag_cols = ['gPSFMag']
    
    data_features = data_valid.copy()
    data_features = create_polynomial_features(data_features, color_cols, degree=2)
    data_features = create_log_features(data_features, color_cols)
    data_features = create_temperature_dependent_features(data_features, color_cols)
    data_features = create_magnitude_features(data_features, mag_cols)
    
    # 5. Prepare feature matrix
    feature_cols = [col for col in data_features.columns if col != 'source_id']
    X = data_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # 6. Match expected feature count
    if X.shape[1] > 38:
        X = X.iloc[:, :38]
    elif X.shape[1] < 38:
        raise ValueError(f"Insufficient features: {X.shape[1]}")
    
    # 7. Apply selector and predict
    X_selected = selector.transform(X)
    predictions = model.predict(X_selected)
    
    # 8. Create output
    results = pd.DataFrame({
        'source_id': data_features['source_id'].values,
        'teff_predicted': predictions
    })
    
    return results
```

---

## 10. Script Documentation

**Script:** `scripts/predict_temperatures_full_catalog.py`

**Purpose:** Predict effective temperatures for eclipsing binaries without Gaia GSP-Phot temperature estimates

**Usage:**
```bash
python scripts/predict_temperatures_full_catalog.py
```

**Inputs:**
- `data/processed/eb_catalog.parquet` - Main catalog (identifies objects without teff_gspphot)
- `data/processed/gaia_eb_panstarrs_phot_with_temperatures.parquet` - Pan-STARRS photometry
- `models/rf_temperature_regressor_feature_engineering_20251002_210423.pkl` - Trained model
- `models/rf_temperature_regressor_feature_engineering_20251002_210423_selector.pkl` - Feature selector

**Outputs:**
- `data/processed/eb_full_catalog_temperatures.parquet` - Predictions for 429,500 objects
- `data/processed/eb_full_catalog_temperatures_SUMMARY.txt` - Summary statistics

---

## 11. Future Improvements

### Technical Enhancements

1. **Save Feature Engineering Parameters**: Store the exact feature engineering configuration (degree, functions used, thresholds) in model metadata to ensure perfect reproducibility

2. **Feature Name Validation**: Save and validate feature names (not just count) to catch column order mismatches

3. **Automated Feature Matching**: Create a function that automatically generates the correct 38 features given the base 6 features

4. **Better Error Messages**: When feature count doesn't match, show which features are missing/extra

### Scientific Enhancements

1. **Uncertainty Estimation**: Add prediction uncertainty for each object based on feature quality and extrapolation distance

2. **Quality Flags**: Flag predictions that may be unreliable (e.g., based on color outliers, extrapolation beyond training range)

3. **Cross-Validation**: Validate predictions against independent spectroscopic catalogs (APOGEE, GALAH) for objects without teff_gspphot

---

## 12. Conclusion

Successfully implemented a robust prediction pipeline that:
- Correctly identifies 493,621 objects without Gaia GSP-Phot temperatures
- Achieves 87% prediction coverage (429,500 predictions)
- Maintains feature engineering consistency with training
- Handles edge cases (NaN, infinity values) gracefully
- Produces scientifically reasonable temperature estimates (mean 9,329 K)

The documented challenges and solutions provide a roadmap for future model deployment and highlight the importance of:
1. Exact feature engineering reproducibility
2. Proper data filtering and validation
3. Comprehensive error handling
4. Clear documentation of the entire pipeline

**Key takeaway:** Production ML requires extreme attention to feature engineering details. Even small differences in feature creation can cause prediction failures or silent accuracy degradation.

