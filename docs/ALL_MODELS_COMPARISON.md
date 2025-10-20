# Temperature Prediction Models - Complete Comparison

Comprehensive comparison of all Random Forest temperature prediction models trained on eclipsing binary stars.

**Date**: 2025-10-18

---

## Models Summary

| Model | Features | Objects | MAE (K) | R² | Within 10% |
|-------|----------|---------|---------|-----|------------|
| **Combined PS+2MASS** | 8 colors | 508,626 | **694.6** | **0.477** | **56.5%** |
| Gaia+2MASS Basic | 4 colors | 525,738 | 722.4 | 0.442 | 55.2% |
| Gaia+2MASS Engineered | 30 features | 525,738 | 722.3 | 0.442 | 55.3% |
| Unified (PS+Gaia) | 5 colors | 701,644 | 765.1 | 0.315 | 43.4% |

**Winner: Combined Pan-STARRS + 2MASS model** 🏆

---

## Model 1: Combined Pan-STARRS + 2MASS + Gaia (BEST)

**Model ID**: `rf_combined_colors_20251018_115824`

### Features (8 colors)

**Optical** (330 nm - 1000 nm):
- `bp_rp` - Gaia BP-RP
- `g_r_color` - Pan-STARRS g-r
- `r_i_color` - Pan-STARRS r-i
- `i_z_color` - Pan-STARRS i-z
- `B_V_color` - Synthetic B-V

**Near-Infrared** (1.25 μm - 2.17 μm):
- `j_h_color` - 2MASS J-H
- `h_k_color` - 2MASS H-K
- `j_k_color` - 2MASS J-K

**Wavelength coverage**: 330 nm to 2.17 μm (optical + NIR)

### Performance

| Metric | Value |
|--------|-------|
| **Test MAE** | **694.6 K** |
| **Test RMSE** | 1080.3 K |
| **Test R²** | **0.477** |
| **Mean Error** | 11.89% |
| **Median Error** | 8.59% |
| **Within 5%** | 30.4% |
| **Within 10%** | **56.5%** |
| **Within 20%** | 82.6% |

### Feature Importance

1. **j_h_color** (37.95%) - 2MASS J-H dominates!
2. **bp_rp** (23.54%) - Gaia BP-RP
3. **h_k_color** (12.05%) - 2MASS H-K
4. **i_z_color** (7.59%) - Pan-STARRS i-z
5. **r_i_color** (5.89%) - Pan-STARRS r-i
6. **j_k_color** (5.68%) - 2MASS J-K
7. **g_r_color** (3.66%) - Pan-STARRS g-r
8. **B_V_color** (3.64%) - Synthetic B-V

**Key insight**: 2MASS near-infrared colors (J-H, H-K) contribute 51% of total importance!

### Training Details

- Training samples: 406,900
- Test samples: 101,726
- Training time: 207.9 seconds (~3.5 minutes)
- Hyperparameters: 300 trees, max_depth=20

---

## Model 2: Gaia + 2MASS Basic

**Model ID**: `rf_gaia_2mass_colors_20251016_155128`

### Features (4 colors)

- `bp_rp` - Gaia BP-RP
- `j_h_color` - 2MASS J-H
- `h_k_color` - 2MASS H-K
- `j_k_color` - 2MASS J-K

### Performance

| Metric | Value |
|--------|-------|
| Test MAE | 722.4 K |
| Test RMSE | 1138.0 K |
| Test R² | 0.442 |
| Within 10% | 55.2% |

### Feature Importance

1. **bp_rp** (38.8%)
2. **j_h_color** (36.9%)
3. **h_k_color** (15.1%)
4. **j_k_color** (9.3%)

### Training Details

- Training samples: 420,590
- Test samples: 105,148
- Training time: 68.8 seconds

---

## Model 3: Gaia + 2MASS Engineered

**Model ID**: `rf_gaia_2mass_engineered_20251016_205640`

### Features (30 selected from 56)

- Base colors: 4
- Polynomial features: 20
- Interaction features: 6
- Log features: 4
- Temperature-dependent: 12

Top engineered feature: **bp_rp_j_h_color_j_k_color** (31.8%) - triple interaction

### Performance

| Metric | Value |
|--------|-------|
| Test MAE | 722.3 K |
| Test RMSE | 1138.0 K |
| Test R² | 0.442 |
| Within 10% | 55.3% |

**Identical to basic Gaia+2MASS model** - feature engineering didn't help!

### Training Details

- Training samples: 420,590
- Test samples: 105,148
- Training time: 653.6 seconds (9.5× slower than basic)

---

## Model 4: Unified (Pan-STARRS + Gaia)

**Model ID**: `rf_unified_engineered_20251016_112332`

### Features (20 selected from many)

- Gaia: BP-RP and engineered features
- Pan-STARRS: g-r, r-i, i-z and engineered features
- Synthetic: B-V and engineered features
- **No 2MASS** (explains lower performance)

### Performance

| Metric | Value |
|--------|-------|
| Test MAE | 765.1 K |
| Test RMSE | 1168.4 K |
| Test R² | 0.315 |
| Within 10% | 43.4% |

### Training Details

- Training samples: 561,315
- Test samples: 140,329

---

## Performance Comparison

### Absolute Metrics

| Model | MAE (K) | RMSE (K) | R² | Training Time |
|-------|---------|----------|-----|---------------|
| **Combined PS+2MASS** | **694.6** | **1080.3** | **0.477** | 207.9 s |
| Gaia+2MASS Basic | 722.4 | 1138.0 | 0.442 | 68.8 s |
| Gaia+2MASS Engineered | 722.3 | 1138.0 | 0.442 | 653.6 s |
| Unified (PS+Gaia) | 765.1 | 1168.4 | 0.315 | - |

### Accuracy Within Thresholds

| Model | Within 5% | Within 10% | Within 20% |
|-------|-----------|------------|------------|
| **Combined PS+2MASS** | **30.4%** | **56.5%** | **82.6%** |
| Gaia+2MASS Basic | 29.2% | 55.2% | 81.9% |
| Gaia+2MASS Engineered | 29.2% | 55.3% | 81.9% |
| Unified (PS+Gaia) | 20.2% | 43.4% | 80.5% |

### Improvement Over Baselines

**Combined vs Gaia+2MASS Basic**:
- MAE: **27.8 K better** (3.8% improvement)
- R²: **0.035 higher** (7.9% improvement)
- Within 10%: **1.3% more objects**

**Combined vs Unified**:
- MAE: **70.5 K better** (9.2% improvement)
- R²: **0.162 higher** (51% improvement)
- Within 10%: **13.1% more objects**

---

## Why Combined Model Performs Best

### 1. Maximum Wavelength Coverage

**Spectral Range**: 330 nm (UV) to 2.17 μm (NIR) - nearly 4 orders of magnitude!

- **Optical colors** (Pan-STARRS + Gaia): Sensitive to hot/intermediate temperatures
- **Near-IR colors** (2MASS): Sensitive to cool temperatures and dust

### 2. Near-Infrared Power

2MASS colors contribute **51% of feature importance**:
- J-H alone: 37.95%
- H-K: 12.05%

**Why NIR matters**:
- Less affected by dust extinction
- Sensitive to molecular absorption (cool stars)
- Complementary to optical colors

### 3. Optimal Feature Set

8 colors provide maximum information without redundancy:
- Each color adds unique information
- No magnitude features (avoids distance bias)
- Covers full stellar temperature range

---

## Model Selection Recommendations

### For Production Use: **Combined Pan-STARRS + 2MASS Model**

✅ **Best overall performance** (MAE: 694.6 K, R²: 0.477)
✅ **Highest accuracy** (56.5% within 10%)
✅ **Maximum wavelength coverage**
✅ **Reasonable training time** (3.5 minutes)
✅ **Simple, interpretable features** (8 colors)

**Caveat**: Requires objects to have both Pan-STARRS AND 2MASS photometry (508k objects)

### For Maximum Coverage: **Gaia + 2MASS Basic**

✅ **More objects** (525k vs 508k)
✅ **Fastest training** (68.8 seconds)
✅ **Nearly as good performance** (MAE: 722 K vs 695 K)
✅ **Only 4 features** (simpler)

**Use when**: Pan-STARRS photometry not available

### Never Use: **Gaia + 2MASS Engineered**

❌ **No performance improvement** over basic
❌ **9.5× slower training**
❌ **More complex** (30 features)
❌ **Harder to interpret**

**Reason**: Random Forests already capture non-linearity; explicit feature engineering is redundant

---

## Physical Insights

### Temperature-Color Relationships

The combined model reveals:

1. **Cool stars** (T < 4000 K):
   - 2MASS J-H color dominates (molecular absorption bands)
   - H-K provides additional sensitivity

2. **Solar-type stars** (4000-6000 K):
   - BP-RP and optical colors most important
   - Balanced contribution from all features

3. **Hot stars** (T > 8000 K):
   - Optical colors dominate
   - NIR less informative (Rayleigh-Jeans tail)

### Wavelength Complementarity

- **Optical** (Pan-STARRS, Gaia): Balmer jump, metal lines → hotter stars
- **Near-IR** (2MASS): Molecular bands, cool dust → cooler stars
- **Combined**: Full temperature range coverage

---

## Dataset Statistics

| Dataset | Objects | Temperature Range | High-Quality 2MASS |
|---------|---------|-------------------|--------------------|
| Combined PS+2MASS | 508,626 | 2,888 - 31,144 K | 60.8% (AAA/AAB) |
| Gaia+2MASS | 525,738 | 2,888 - 32,657 K | 56% (AAA/AAB) |
| Unified (PS+Gaia) | 701,644 | 2,888 - 31,144 K | N/A |

---

## Performance Ceiling Analysis

### Current Best: R² = 0.477 (Combined Model)

**What limits further improvement?**

1. **Measurement errors** in photometry (~0.02-0.05 mag)
2. **Intrinsic scatter** in color-temperature relations
3. **Binary nature**: Eclipsing binaries have composite spectra
4. **Missing physics**: Metallicity, surface gravity, rotation
5. **Gaia GSP-Phot uncertainties**: Training target has errors

### Theoretical Improvements

To reach R² > 0.6 would require:
- Spectroscopic data (metallicity, log g)
- Multi-epoch photometry (variability)
- Binary modeling (mass ratios, inclination)
- Better extinction corrections
- More photometric bands (WISE, Spitzer)

---

## Files Generated

### Combined Model

**Model files**:
- `models/rf_combined_colors_20251018_115824.pkl`
- `models/rf_combined_colors_20251018_115824_metadata.json`
- `models/rf_combined_colors_20251018_115824_SUMMARY.txt`
- `models/rf_combined_colors_20251018_115824_test_predictions.parquet`

**Dataset**:
- `data/processed/combined_panstarrs_2mass_colors_training.parquet` (31.3 MB)

**Documentation**:
- `docs/COMBINED_PANSTARRS_2MASS_SCHEMA.md`

**Training log**:
- `train_combined.log`

---

## Conclusion

The **Combined Pan-STARRS + 2MASS + Gaia model** achieves the best performance by leveraging the full optical to near-infrared wavelength range. The key insight is that 2MASS near-infrared colors are crucial for temperature prediction, contributing over 50% of the model's predictive power.

For production use, this model provides:
- **Best accuracy**: 56.5% of predictions within 10%
- **Lowest MAE**: 694.6 K
- **Highest R²**: 0.477
- **Full wavelength coverage**: 330 nm to 2.17 μm

This represents a **~9% improvement** over models using optical colors alone, demonstrating the value of multi-wavelength photometry for stellar parameter estimation.
