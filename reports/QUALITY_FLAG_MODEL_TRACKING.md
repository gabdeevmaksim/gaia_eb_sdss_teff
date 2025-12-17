# Quality Flag Model Tracking

**Purpose**: Track the high-quality Teff model trained only on Gaia GSP-Phot flag 1 (best quality) sources.

---

## 1. External Data Source

**Location**: `data/external/gsp_teff_quality_flags/`

**Files**: 5 large FITS files with Gaia GSP-Phot quality flags (16 GB total)
- `gaia_teff_lfrom0_to90.fits` (5.1 GB)
- `gaia_teff_lfrom90_to180.fits` (2.1 GB)
- `gaia_teff_lfrom180_to270.fits` (2.3 GB)
- `gaia_teff_lfrom270_to315.fits` (2.7 GB)
- `gaia_teff_lfrom315_to360.fits` (3.8 GB)

**Coverage**: ~470M Gaia sources with temperature quality flags
**Flag meanings**:
- **Flag 0**: Lower quality temperature estimate
- **Flag 1**: Higher quality temperature estimate (best quality)

---

## 2. Data Processing Pipeline

### Step 1: Add Quality Flags
**Script**: `scripts/add_gaia_quality_flags.py`
**Input**: `data/processed/ml_training_data_with_gaia.parquet` (706k EBs)
**Output**: `data/processed/ml_training_data_with_gaia_with_flags.parquet` (43 MB, 706k rows)

**Algorithm**:
1. Load EB catalog with Gaia source_ids
2. Create search set of source_ids
3. For each FITS file, load and find matches
4. Update dataframe with quality flags

**Results**:
- Total sources: 705,998
- Flag 0 (lower quality): 287,146 (40.7%)
- Flag 1 (higher quality): 418,852 (59.3%)

### Step 2: Filter Training Data
**Selection**: Only flag 1 sources used for training
**Training samples**: 418,852 high-quality sources
**Temperature range**: 2,737 - 21,906 K

---

## 3. Model Training

### Model File
**Location**: `models/rf_temperature_regressor_high_quality_20251013_105249.pkl`
**Model ID**: `rf_temperature_regressor_high_quality_20251013_105249`
**Timestamp**: October 13, 2024, 10:52:49
**Size**: ~2.3 GB (300 trees)

### Additional Files
- `rf_temperature_regressor_high_quality_20251013_105249_metadata.json` - Full model metadata
- `rf_temperature_regressor_high_quality_20251013_105249_SUMMARY.txt` - Performance summary
- `rf_temperature_regressor_high_quality_20251013_105249_test_predictions.parquet` - Test set predictions
- `rf_temperature_regressor_high_quality_20251013_105249_selector.pkl` - Feature selector

### Training Configuration

**Dataset**: High-quality flag 1 sources only
- Training samples: 335,081 (80%)
- Test samples: 83,771 (20%)
- Temperature range: 2,737 - 21,906 K

**Features**: 20 color-based features (NO magnitudes to avoid bias)

**Base colors**:
1. `g_r_color` (Pan-STARRS)
2. `r_i_color` (Pan-STARRS)
3. `B_V_color` (derived)
4. `bp_rp` (Gaia)

**Engineered features**:
- Polynomial terms: `bp_rp^2`, `bp_rp^3`
- Color combinations: `g_r_color_bp_rp`, `r_i_color_bp_rp`, `B_V_color_bp_rp`, etc.
- Interactions: `g_r_color_x_bp_rp`, `r_i_color_x_bp_rp`, `B_V_color_x_bp_rp`
- Log features: `log_g_r_color`, `log_r_i_color`, `log_B_V_color`, `log_bp_rp`
- Cool star features: `cool_g_r_color`, `cool_B_V_color`, `cool_bp_rp`

**Feature selection**: SelectKBest with f_regression, top 20 features

**Hyperparameters**:
- `n_estimators`: 300
- `max_depth`: 20
- `min_samples_leaf`: 4
- `min_samples_split`: 5
- `random_state`: 42

### Performance Metrics

**Training set**:
- MAE: 223.2 K
- RMSE: 296.2 K
- R²: 0.812

**Test set** (83,771 sources):
- **MAE: 264.6 K** ⭐ (Best performance!)
- **RMSE: 348.8 K**
- **R²: 0.739**
- Relative error: 5.51%

**Training time**: 458.9 seconds (~7.6 minutes)

---

## 4. Predictions

### Flag 0 Predictions
**Script**: `scripts/predict_flag0_temperatures.py`
**Input**: Flag 0 sources (lower quality Gaia temperatures)
**Output**: `data/processed/flag0_temperature_predictions.parquet` (28 MB, 287k predictions)

**Summary file**: `data/processed/flag0_temperature_predictions_SUMMARY.txt`

**Prediction statistics**:
- Sources predicted: 287,146 (all flag 0 sources)
- Prediction time: 5.1 seconds
- Predicted temperature range: 3,141 - 16,563 K
- Original temperature range: 3,071 - 31,144 K

**Performance on flag 0 sources** (testing model on lower-quality data):
- MAE: 1,329.2 K (much worse than test set)
- RMSE: 1,968.5 K
- R²: -0.152 (negative! worse than mean)
- Mean error: 18.54%

**Interpretation**: Model trained on flag 1 performs poorly on flag 0, as expected. This shows flag 0 data has fundamentally different characteristics.

---

## 5. Validation

### GALAH Cross-match
**Script**: `scripts/crossmatch_galah_xmatch.py`
**Input**: `data/processed/flag0_temperature_predictions.parquet`
**Output**: `data/processed/galah_validated_predictions.parquet` (18 KB, 39 matches)

**Validation results**: See `reports/GALAH_VALIDATION_SUMMARY.md`

### Quality Analysis
**Notebook**: `notebooks/gaia_quality_flag_analysis.ipynb`
**Figures directory**: `reports/figures/quality_analysis/`

**Available plots**:
1. `quality_flag_distribution.png` - Flag 0 vs Flag 1 distribution
2. `temperature_comparison_by_flag.png` - Teff distributions by flag
3. `magnitude_comparison_by_flag.png` - Magnitude distributions by flag
4. `bp_rp_color_comparison_by_flag.png` - BP-RP color by flag
5. `panstarrs_colors_comparison_by_flag.png` - Pan-STARRS colors by flag
6. `hr_diagram_by_quality_flag.png` - HR diagram color-coded by flag
7. `hr_diagram_hexbin_by_flag.png` - HR diagram density plot by flag

---

## 6. Key Findings

### Model Performance
✅ **Excellent performance on flag 1 test data**: 264.6 K MAE, R²=0.739
❌ **Poor performance on flag 0 data**: 1,329 K MAE, R²=-0.152

### Physical Interpretation
- **Flag 1 sources** have consistent, reliable Gaia GSP-Phot temperatures → ML can learn patterns
- **Flag 0 sources** have problematic Gaia temperatures → ML cannot reliably predict from photometry alone
- Color-based features work well for high-quality data but fail for lower quality

### Quality Flag Statistics
- 59.3% of EBs have flag 1 (high quality)
- 40.7% of EBs have flag 0 (lower quality)
- Flag distribution varies with stellar parameters (cooler stars tend to have better quality)

---

## 7. Model Comparison

| Model | Dataset | Test MAE | Test R² | Features | Purpose |
|-------|---------|----------|---------|----------|---------|
| **High Quality** | Flag 1 only | **264.6 K** | **0.739** | 20 colors | Best quality training |
| Gaia All Colors | All flags | 556.9 K | 0.640 | 6 colors | General purpose |
| Gaia 2MASS IR | All flags | 765.1 K | 0.315 | 9 features | IR coverage |

**Conclusion**: Training on flag 1 only gives **52% better MAE** than training on all data.

---

## 8. Workflow Summary

```
External Data (16 GB FITS)
    ↓
[add_gaia_quality_flags.py]
    ↓
ml_training_data_with_gaia_with_flags.parquet (706k sources, flags 0+1)
    ↓
Filter to flag 1 only (419k sources)
    ↓
[train_high_quality_model.py]
    ↓
rf_temperature_regressor_high_quality_20251013_105249.pkl (MAE: 264.6 K)
    ↓
[predict_flag0_temperatures.py]
    ↓
flag0_temperature_predictions.parquet (287k predictions, MAE: 1,329 K)
    ↓
[crossmatch_galah_xmatch.py]
    ↓
galah_validated_predictions.parquet (39 spectroscopic matches)
```

---

## 9. Usage Notes

### When to Use This Model
✅ **Best for**: Predicting temperatures for flag 1 (high-quality) sources
✅ **Training baseline**: Demonstrates best possible performance with clean data
✅ **Quality benchmark**: Reference for evaluating other models

### When NOT to Use This Model
❌ **Flag 0 predictions**: Model fails on lower-quality data (MAE: 1,329 K)
❌ **Missing Gaia Teff**: Model trained on Gaia, can't extrapolate to no-Gaia cases
❌ **Production use**: Use corrected all-data models instead

### Recommended Model for Production
Use: `rf_gaia_teff_corrected_log_20251126_130144.pkl`
- Trained on ALL data (flags 0+1)
- Includes Teff correction for hot stars
- Log transformation for better scaling
- MAE: 556.9 K (still excellent, more robust)

---

## 10. Related Documentation

- `reports/GALAH_VALIDATION_SUMMARY.md` - Spectroscopic validation results
- `notebooks/gaia_quality_flag_analysis.ipynb` - Quality flag analysis
- `notebooks/color_quality_analysis.ipynb` - Color quality by flag
- `docs/GAIA_TEFF_MODELS_COMPARISON.md` - Model comparison study

---

## 11. Future Work

### Potential Improvements
1. **Separate models by flag**: Train dedicated models for flag 0 and flag 1
2. **Flag prediction**: Predict quality flag from photometry first, then route to appropriate model
3. **Uncertainty by flag**: Model uncertainty should vary by input quality flag
4. **Transfer learning**: Fine-tune flag 1 model for flag 0 with careful regularization

### Research Questions
1. What makes flag 0 temperatures unreliable? (crowding, extinction, binarity?)
2. Can we predict which flag 0 sources have actually good temperatures?
3. Should we exclude flag 0 from training entirely (current approach) or include with sample weights?

---

**Last updated**: December 17, 2025
**Branch**: paper-draft
