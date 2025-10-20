# Gaia + 2MASS Temperature Prediction Models - Comparison

This document compares the Basic and Engineered Features models for temperature prediction using Gaia BP-RP and 2MASS near-infrared colors.

**Date**: 2025-10-16

---

## Models Overview

### Model 1: Basic Colors (rf_gaia_2mass_colors_20251016_155128)

**Features**: 4 color indices only
- `bp_rp` - Gaia BP-RP color
- `j_h_color` - 2MASS J-H color
- `h_k_color` - 2MASS H-K color
- `j_k_color` - 2MASS J-K color

**Training**: 420,590 samples, Test: 105,148 samples

### Model 2: Engineered Features (rf_gaia_2mass_engineered_20251016_205640)

**Features**: 56 total features → 30 selected (via SelectKBest)
- Base colors: 4
- Polynomial features (degree 2, 3): 20
- Interaction features: 6
- Log transformations: 4
- Temperature-dependent features: 12
- Other engineered features: 10

**Training**: 420,590 samples, Test: 105,148 samples

---

## Performance Comparison

### Test Set Performance (105,148 objects)

| Metric | Basic Model | Engineered Model | Difference |
|--------|-------------|------------------|------------|
| **MAE** | 722.4 K | 722.3 K | -0.1 K (0.01%) |
| **RMSE** | 1138.0 K | 1138.0 K | 0.0 K (0.00%) |
| **R²** | 0.442 | 0.442 | 0.000 |
| **Mean Error %** | 12.28% | 12.28% | 0.00% |
| **Median Error %** | 8.86% | 8.86% | 0.00% |
| **Within 5%** | 29.2% | 29.2% | 0.0% |
| **Within 10%** | 55.2% | 55.3% | +0.1% |
| **Within 20%** | 81.9% | 81.9% | 0.0% |

**Conclusion**: **Identical performance** - Feature engineering provided no improvement.

### Performance by Temperature Range

Both models show similar patterns:
- **Best**: 4000-6000 K (Solar-type stars) - MAE ~450-500 K, ~65% within 10%
- **Worst**: >8000 K (Hot stars) - MAE ~2700 K, only ~13% within 10%
- **Cool stars** (<4000 K): MAE ~750 K, ~24% within 10%

---

## Feature Importance Analysis

### Basic Model
All 4 features used directly:
1. **bp_rp**: 38.8% importance
2. **j_h_color**: 36.9% importance
3. **h_k_color**: 15.1% importance
4. **j_k_color**: 9.3% importance

**Key insight**: BP-RP and J-H dominate the predictions.

### Engineered Model (Top 10 features)
Selected 30 features from 56:
1. **bp_rp_j_h_color_j_k_color**: 31.8% - Triple interaction term (dominant!)
2. **j_h_color_j_k_color**: 5.1% - 2MASS color interaction
3. **j_h_color_x_j_k_color**: 4.8% - Interaction feature
4. **bp_rp^2_h_k_color**: 4.8% - Polynomial × color
5. **bp_rp^2_j_h_color**: 4.4% - Polynomial × color
6. **log_bp_rp**: 4.2% - Log transformation
7. **bp_rp^2**: 4.2% - Polynomial
8. **bp_rp^3**: 4.2% - Polynomial
9. **bp_rp**: 4.2% - Base color
10. **bp_rp_h_k_color_j_k_color**: 4.0% - Triple interaction

**Key insight**: The most important engineered feature is a complex 3-way interaction between all major colors. However, this complexity didn't improve overall performance.

---

## Predictions for Objects Without Gaia Teff

Both models predicted temperatures for **101,508 objects**:

| Statistic | Basic Model | Engineered Model | Difference |
|-----------|-------------|------------------|------------|
| **Mean Teff** | 5,134 K | 5,139 K | +5 K |
| **Median Teff** | 4,851 K | 4,851 K | 0 K |
| **Min Teff** | 3,346 K | 3,311 K | -35 K |
| **Max Teff** | 14,467 K | 15,625 K | +1,158 K |
| **Std Dev** | 1,043 K | 1,043 K | 0 K |

**Conclusion**: Nearly identical predictions. Predictions are ~400 K cooler than training mean (5,534 K).

### Distribution Comparison

**Wasserstein Distance** (Training vs Predictions):
- Basic model: 437.52 K
- Engineered model: 436.06 K

Both models show the same systematic shift toward cooler temperatures in the prediction set.

---

## Training Efficiency

| Aspect | Basic Model | Engineered Model | Difference |
|--------|-------------|------------------|------------|
| **Training time** | 68.8 seconds | 653.6 seconds | **9.5× slower** |
| **Number of features** | 4 | 30 selected (56 total) | 7.5× more |
| **Model complexity** | Simple | Complex | - |
| **Prediction time** | Fast | Moderate | ~2× slower |

---

## Why Feature Engineering Didn't Help

1. **Random Forests Already Capture Non-linearity**: Tree-based models inherently capture polynomial relationships and interactions through recursive splitting. Explicit polynomial and interaction features are redundant.

2. **Color-Temperature Relationship**: The relationship between stellar colors and temperature is already well-represented by the 4 basic color indices. The RF model can learn the complex mapping without explicit feature engineering.

3. **Overfitting Risk**: Adding 52 engineered features increases model complexity without improving generalization. The feature selection (30 from 56) helps, but doesn't add predictive power.

4. **Data Limitations**: The ~R² = 0.44 ceiling suggests limitations in the data itself (measurement errors, intrinsic scatter, binary nature of eclipsing binaries) rather than model limitations.

---

## Recommendations

### **Use the Basic Model for Production**

**Reasons**:
1. ✅ **Identical performance** to engineered model
2. ✅ **9.5× faster training** (68s vs 654s)
3. ✅ **Simpler and more interpretable** (4 features vs 30)
4. ✅ **Faster predictions**
5. ✅ **Lower computational requirements**
6. ✅ **Easier to maintain and debug**

### When Feature Engineering Might Help

Feature engineering **could** be beneficial when:
- Using linear models (e.g., Linear Regression, Ridge, Lasso)
- Working with simpler algorithms (e.g., KNN)
- Need to explicitly enforce physical constraints
- Domain knowledge suggests specific functional forms

For tree-based models (Random Forest, XGBoost, etc.), the data and hyperparameter tuning matter more than explicit feature engineering.

---

## Files Generated

### Basic Model
**Model files**:
- `models/rf_gaia_2mass_colors_20251016_155128.pkl`
- `models/rf_gaia_2mass_colors_20251016_155128_metadata.json`
- `models/rf_gaia_2mass_colors_20251016_155128_SUMMARY.txt`
- `models/rf_gaia_2mass_colors_20251016_155128_test_predictions.parquet`

**Predictions**:
- `data/processed/gaia_2mass_temperature_predictions_rf_gaia_2mass_colors_20251016_155128.parquet`

**Validation plots**:
- `reports/figures/gaia_2mass_validation/` (7 figures)

### Engineered Model
**Feature engineering**:
- `data/processed/gaia_2mass_colors_engineered_train.parquet` (105.2 MB, 56 features)
- `docs/GAIA_2MASS_ENGINEERED_FEATURES.md`

**Model files**:
- `models/rf_gaia_2mass_engineered_20251016_205640.pkl`
- `models/rf_gaia_2mass_engineered_20251016_205640_selector.pkl`
- `models/rf_gaia_2mass_engineered_20251016_205640_metadata.json`
- `models/rf_gaia_2mass_engineered_20251016_205640_SUMMARY.txt`
- `models/rf_gaia_2mass_engineered_20251016_205640_test_predictions.parquet`

**Predictions**:
- `data/processed/gaia_2mass_engineered_predictions_rf_gaia_2mass_engineered_20251016_205640.parquet`

**Validation plots**:
- `reports/figures/gaia_2mass_engineered_validation/` (7 figures)

**Training log**:
- `train_gaia_2mass_eng.log`

---

## Comparison with Unified Model (Pan-STARRS + Gaia)

For reference, the unified model (using Pan-STARRS colors + Gaia BP-RP):

| Model | MAE | R² | Within 10% |
|-------|-----|-----|------------|
| **Unified (Pan-STARRS+Gaia)** | 765 K | 0.315 | 43.4% |
| **Gaia+2MASS (Basic)** | 722 K | 0.442 | 55.2% |
| **Gaia+2MASS (Engineered)** | 722 K | 0.442 | 55.3% |

**Gaia + 2MASS models outperform the Unified model**, likely because:
- 2MASS near-infrared colors are more sensitive to temperature than optical colors
- J-H and H-K colors have tighter correlations with temperature for cool stars
- Combined optical (BP-RP) + infrared (J-H, H-K, J-K) provides better temperature coverage

---

## Conclusion

The **Basic Gaia + 2MASS model** achieves excellent performance with just 4 color features and should be the preferred choice for production use. The engineered features model demonstrates that complex feature engineering is unnecessary when using tree-based models on well-chosen color indices.

The fundamental limitation is **R² ≈ 0.44**, suggesting that further improvements require:
1. Better photometric data (higher precision, more bands)
2. Additional physical parameters (metallicity, surface gravity, extinction)
3. Spectroscopic validation and calibration
4. Accounting for binary nature (mass ratios, inclination, eclipsing geometry)
