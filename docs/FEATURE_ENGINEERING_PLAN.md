# Feature Engineering Plan for Temperature Prediction

## Problem Statement

When training a model with engineered features and then predicting on new data, we need to ensure:
1. **Identical feature engineering** applied to both training and prediction datasets
2. **Feature distributions are similar** between datasets
3. **All required input features are available** in prediction data
4. **Reproducible and auditable** feature creation process

## Current Issues

### 1. Feature Mismatch
- Model trained on 20 selected features from ~38 engineered features
- Risk of feature distribution differences between train/predict
- No validation that prediction data has similar feature distributions

### 2. Data Separation
- Training data: objects with `teff_gspphot` (flag 1 only for high-quality model)
- Prediction data: objects without `teff_gspphot` OR with flag 0
- Features currently engineered separately → potential inconsistencies

### 3. No Distribution Comparison
- No way to verify that feature distributions match between datasets
- Could lead to out-of-distribution predictions

## Solution: Unified Feature Engineering Pipeline

### Phase 1: Create Master Feature Dataset

**Goal**: Create a single dataset with ALL objects (with and without ground truth) and engineered features.

**Steps**:

1. **Load and merge all data sources**
   - `eb_panstarrs_for_prediction.parquet` (1.17M objects, Pan-STARRS photometry)
   - Add Gaia colors (`bp_rp`)
   - Mark objects with/without `teff_gspphot`

2. **Apply feature engineering to ALL objects simultaneously**
   ```python
   # Same feature engineering for everyone
   df_all = engineer_all_features(
       df_all,
       color_cols=['g_r_color', 'r_i_color', 'i_z_color', 'B_V_color', 'bp_rp'],
       mag_cols=None,  # Or ['gPSFMag'] for basic model
       include_polynomials=True,
       include_interactions=True,
       include_log=True,
       include_temp_dependent=True,
       include_mag_features=False  # Or True for basic model
   )
   ```

3. **Split dataset after feature engineering**
   ```python
   df_train = df_all[df_all['teff_gspphot'].notna()]  # For training/validation
   df_predict = df_all[df_all['teff_gspphot'].isna()]  # For prediction
   ```

4. **Save unified dataset**
   - `data/processed/eb_unified_features.parquet` - All objects with all features
   - `data/processed/eb_unified_features_train.parquet` - Training subset
   - `data/processed/eb_unified_features_predict.parquet` - Prediction subset

### Phase 2: Feature Distribution Validation

**Goal**: Verify that feature distributions are similar between training and prediction sets.

**Metrics to compute**:

```python
for feature in feature_cols:
    # Summary statistics
    train_mean, train_std = df_train[feature].mean(), df_train[feature].std()
    pred_mean, pred_std = df_predict[feature].mean(), df_predict[feature].std()

    # Distribution tests
    ks_statistic, p_value = ks_2samp(df_train[feature], df_predict[feature])

    # Outliers
    train_outliers = (df_train[feature] < q1 - 1.5*iqr) | (df_train[feature] > q3 + 1.5*iqr)
    pred_outliers = (df_predict[feature] < q1 - 1.5*iqr) | (df_predict[feature] > q3 + 1.5*iqr)
```

**Visualizations**:
- Side-by-side histograms for each feature
- Q-Q plots to check distribution similarity
- Box plots to compare ranges and outliers
- Correlation matrices for both datasets

### Phase 3: Model Training on Unified Features

**Goal**: Train model using the unified feature dataset.

**Process**:
1. Load `eb_unified_features_train.parquet`
2. Apply feature selection (SelectKBest) on training data
3. Save feature selector with model
4. **Important**: Save list of selected feature names in metadata

### Phase 4: Prediction on Unified Features

**Goal**: Predict using the unified feature dataset.

**Process**:
1. Load `eb_unified_features_predict.parquet` (features already engineered!)
2. Load model + selector
3. Select same features using selector
4. Predict
5. **No feature engineering needed** - already done!

## Implementation Scripts

### Script 1: `create_unified_feature_dataset.py`

```python
#!/usr/bin/env python3
"""
Create unified feature dataset for training and prediction.

This script:
1. Loads all data sources (with and without ground truth)
2. Applies feature engineering to ALL objects simultaneously
3. Saves unified dataset and splits (train/predict)
4. Generates feature distribution comparison report

Usage:
    python scripts/create_unified_feature_dataset.py

Output:
    - data/processed/eb_unified_features.parquet
    - data/processed/eb_unified_features_train.parquet
    - data/processed/eb_unified_features_predict.parquet
    - reports/feature_distribution_comparison.html
"""

# Key steps:
# 1. Load eb_panstarrs_for_prediction.parquet (all objects with Pan-STARRS)
# 2. Ensure all required input features present (g_r_color, r_i_color, etc.)
# 3. Apply engineer_all_features() to ENTIRE dataset at once
# 4. Split by teff_gspphot availability
# 5. Compare feature distributions
# 6. Save datasets and report
```

### Script 2: `validate_feature_distributions.py`

```python
#!/usr/bin/env python3
"""
Validate that feature distributions are similar between train and predict sets.

Creates comprehensive comparison report with:
- Statistical tests (KS test, t-test)
- Distribution plots (histograms, KDE, Q-Q plots)
- Summary tables
- Outlier analysis

Usage:
    python scripts/validate_feature_distributions.py

Output:
    - reports/feature_validation_report.html
    - reports/figures/feature_distributions/
"""

# Key analyses:
# - Per-feature KS test
# - Mean/std comparison
# - Percentile comparison (5th, 25th, 50th, 75th, 95th)
# - Correlation matrix comparison
# - PCA to check overall distribution shift
```

### Script 3: `train_model_unified_features.py`

```python
#!/usr/bin/env python3
"""
Train model on unified feature dataset.

Usage:
    python scripts/train_model_unified_features.py --model-type basic
    python scripts/train_model_unified_features.py --model-type engineered

Input:
    - data/processed/eb_unified_features_train.parquet

Output:
    - models/rf_unified_{model_type}_{timestamp}.pkl
    - models/rf_unified_{model_type}_{timestamp}_selector.pkl
    - models/rf_unified_{model_type}_{timestamp}_metadata.json
"""

# Advantages:
# - Features already engineered
# - Just load, select, train
# - Guaranteed consistency with prediction data
```

### Script 4: `predict_unified_features.py`

```python
#!/usr/bin/env python3
"""
Predict temperatures using unified feature dataset.

Usage:
    python scripts/predict_unified_features.py --model MODEL_FILE

Input:
    - data/processed/eb_unified_features_predict.parquet (features already present!)
    - models/MODEL_FILE.pkl
    - models/MODEL_FILE_selector.pkl

Output:
    - data/processed/predictions_{model_id}.parquet
"""

# Super simple:
# 1. Load feature dataset (features already there!)
# 2. Load model + selector
# 3. Select features (using selector.transform)
# 4. Predict
# No feature engineering → no mistakes!
```

## Benefits of This Approach

1. **Consistency Guaranteed**: Features engineered once for all data
2. **Distribution Validation**: Explicit comparison before training
3. **Reproducibility**: Same feature engineering code path for all
4. **Debugging**: Easy to trace feature issues
5. **Efficiency**: Feature engineering only done once (can be slow)
6. **Confidence**: Statistical validation that train/predict are similar

## Notebook for Analysis

Create `notebooks/unified_features_validation.ipynb`:

```python
# Load unified datasets
df_train = load_data('eb_unified_features_train.parquet')
df_predict = load_data('eb_unified_features_predict.parquet')

# Compare distributions for each feature
for feature in selected_features:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram comparison
    axes[0].hist(df_train[feature], alpha=0.5, label='Train', bins=50)
    axes[0].hist(df_predict[feature], alpha=0.5, label='Predict', bins=50)

    # KDE comparison
    df_train[feature].plot.kde(ax=axes[1], label='Train')
    df_predict[feature].plot.kde(ax=axes[1], label='Predict')

    # Q-Q plot
    plot_qq(df_train[feature], df_predict[feature], ax=axes[2])

    plt.tight_layout()
    plt.show()
```

## Timeline

1. **Week 1**: Implement `create_unified_feature_dataset.py` and `validate_feature_distributions.py`
2. **Week 1**: Run analysis, review distribution comparison report
3. **Week 2**: Train models using unified features, compare with existing models
4. **Week 2**: Generate predictions, validate results

## Success Criteria

- [ ] Unified feature dataset created with >1.1M objects
- [ ] Feature distributions validated (KS test p-value > 0.05 for key features)
- [ ] Model trained on unified features achieves similar or better performance
- [ ] Predictions generated without feature engineering errors
- [ ] Comprehensive validation report showing distribution similarity
