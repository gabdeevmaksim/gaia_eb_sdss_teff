# Validation Configuration Files

Configuration files for model validation (plots + metrics generation).

## Usage

```bash
# Validate using configuration file
python pipeline.py validate --config config/validation/validate_gaia_2mass_ir.yaml

# Or use Python API
from src.pipeline import ValidationPipeline
pipeline = ValidationPipeline('config/validation/validate_gaia_2mass_ir.yaml')
context = pipeline.run()
```

## Available Configurations

### `validate_gaia_2mass_ir.yaml`
- Validates Gaia + 2MASS IR model
- Model pattern: `rf_gaia_2mass_ir_*` (latest version)
- All plots enabled

### `validate_latest_model.yaml`
- Validates most recently trained model
- Model pattern: `rf_*` (any model, most recent)
- All plots enabled

## Configuration Schema

```yaml
model:
  # Model pattern (supports wildcards for latest version)
  model_pattern: "rf_model_*"
  # Or specific model ID:
  # model_pattern: "rf_model_20251103_141119"

plots:
  # Enable/disable individual plots
  test_scatter: true              # Predicted vs. true scatter plot
  residuals: true                  # Residuals vs. true temperature
  performance_by_temp: true        # Metrics by temperature bins
  temp_distributions: true         # Distribution comparison
  feature_importance: true         # Feature importance bar plot

output:
  # Subdirectory for validation figures
  figures_subdir: "model_validation"

  # Optional: custom report file location
  # report_file: "reports/my_validation.txt"
```

## Generated Outputs

### Validation Plots

Saved to `reports/figures/{figures_subdir}/`:

1. **Test Scatter** (`{model_id}_test_scatter.png`)
   - Predicted vs. ground truth
   - 1:1 line and ±10% bounds
   - Hexbin density plot
   - MAE, RMSE, R² displayed

2. **Residuals** (`{model_id}_residuals.png`)
   - Residuals vs. true temperature
   - Shows prediction bias
   - ±10% reference lines

3. **Performance by Temperature** (`{model_id}_performance_by_temp.png`)
   - MAE/RMSE by temperature bins
   - Identifies where model struggles
   - Bin statistics

4. **Temperature Distributions** (`{model_id}_temp_distributions.png`)
   - Distribution of true vs. predicted
   - KS test statistic
   - Histogram comparison

5. **Feature Importance** (`{model_id}_feature_importance.png`)
   - Top N features by importance
   - Shows what drives predictions
   - Only if model supports feature_importances_

### Validation Report

**Text file** (`reports/validation_report_{model_id}.txt`):
```
Validation Report: Gaia + 2MASS IR
================================================================================

Model ID: rf_gaia_2mass_ir_20251103_141119
Model File: models/rf_gaia_2mass_ir_20251103_141119.pkl
Validation Date: 2025-11-03 15:45:00

Test Set: 107,932 samples

OVERALL PERFORMANCE:
  MAE:  589.0 K
  RMSE: 965.5 K
  R²:   0.5931
  Mean Relative Error: 12.50%

ACCURACY THRESHOLDS:
  Within  5%: 42.3%
  Within 10%: 64.8%
  Within 20%: 85.1%

PERFORMANCE BY TEMPERATURE RANGE:
  [3500 - 4500] K (n=21,586):
    MAE:  425 K
    RMSE: 678 K
    R²:   0.621
  ...

VALIDATION PLOTS:
  Location: reports/figures/gaia_2mass_ir_validation/
  Number of plots: 5
```

**JSON file** (`reports/validation_report_{model_id}.json`):
- Machine-readable metrics
- For programmatic comparison
- Same data as text report

## Creating New Validation Configs

1. **Copy a template:**
   ```bash
   cp config/validation/validate_gaia_2mass_ir.yaml config/validation/validate_my_model.yaml
   ```

2. **Update model pattern:**
   ```yaml
   model:
     model_pattern: "my_model_*"  # Latest version
     # or
     model_pattern: "my_model_20251103_141119"  # Specific version
   ```

3. **Choose plots:**
   ```yaml
   plots:
     test_scatter: true
     residuals: true
     performance_by_temp: true
     temp_distributions: false  # Disable if not needed
     feature_importance: true
   ```

4. **Set output directory:**
   ```yaml
   output:
     figures_subdir: "my_model_validation"
   ```

5. **Run validation:**
   ```bash
   python pipeline.py validate --config config/validation/validate_my_model.yaml
   ```

## Use Cases

### Validate After Training

```bash
# 1. Train model
python pipeline.py --ml --ml-config config/models/gaia_2mass_ir.yaml

# 2. Validate immediately
python pipeline.py validate --config config/validation/validate_gaia_2mass_ir.yaml
```

### Compare Multiple Models

```bash
# Validate all models
for model in gaia_2mass_ir gaia_g_bprp panstarrs_unified; do
    python pipeline.py validate --config config/validation/validate_${model}.yaml
done

# Compare reports
cat reports/validation_report_rf_gaia_2mass_ir_*.txt
cat reports/validation_report_rf_gaia_g_bprp_*.txt
cat reports/validation_report_rf_unified_*.txt
```

### Automated Validation

```bash
#!/bin/bash
# validate_latest.sh

# Find most recent model
latest_model=$(ls -t models/rf_*.pkl | head -1)
model_id=$(basename $latest_model .pkl)

echo "Validating: $model_id"

# Create temp config
cat > /tmp/validate_temp.yaml <<EOF
model:
  model_pattern: "$model_id"
plots:
  test_scatter: true
  residuals: true
  performance_by_temp: true
  temp_distributions: true
  feature_importance: true
output:
  figures_subdir: "${model_id}_validation"
EOF

# Run validation
python pipeline.py validate --config /tmp/validate_temp.yaml
```

## Model Wildcards

Using wildcards selects the **most recent** model based on filename timestamp.

**Examples:**

```yaml
# Any Gaia+2MASS model, latest
model_pattern: "rf_gaia_2mass_*"

# Any model of specific type, latest
model_pattern: "rf_unified_*"

# Any RF model at all, latest
model_pattern: "rf_*"

# Specific version (no wildcard)
model_pattern: "rf_gaia_2mass_ir_20251103_141119"
```

## Troubleshooting

### Error: "Model not found"

```
FileNotFoundError: No model found matching: rf_model_*
```

**Solution:** Check models directory:
```bash
ls models/rf_model_*.pkl
```

### Error: "Test predictions not found"

```
FileNotFoundError: Test predictions not found: rf_model_*_test_predictions.parquet
```

**Solution:** Model doesn't have test predictions file. Retrain with current pipeline to generate predictions file.

### Error: "Column not found in predictions"

```
KeyError: 'true_temperature'
```

**Solution:** Predictions file has unexpected column names. Pipeline tries to detect various naming conventions, but older models may have incompatible formats.

### Plots look wrong

**Check:**
1. Predictions file format matches expected columns
2. Temperature units (should be Kelvin)
3. No extreme outliers corrupting scale
4. Feature names match metadata

## Best Practices

### 1. Validate After Every Training

Always validate immediately after training:
```bash
python pipeline.py --ml --ml-config config/models/my_model.yaml && \
python pipeline.py validate --config config/validation/validate_my_model.yaml
```

### 2. Compare Similar Models

Keep validation outputs organized:
```
reports/figures/
├── gaia_2mass_basic_validation/
├── gaia_2mass_engineered_validation/
└── gaia_2mass_optimized_validation/
```

### 3. Track Validation History

Save validation reports with timestamps:
```yaml
output:
  report_file: "reports/validations/validation_${DATE}_${MODEL}.txt"
```

### 4. Automate in CI/CD

```yaml
# .github/workflows/validate.yml
- name: Train and Validate
  run: |
    python pipeline.py --ml --ml-config config/models/production.yaml
    python pipeline.py validate --config config/validation/validate_production.yaml

- name: Check Metrics
  run: |
    # Fail if MAE > threshold
    python scripts/check_validation_metrics.py --max-mae 600
```

## See Also

- `docs/PIPELINES.md` - Complete pipeline documentation
- `config/models/` - Training configurations
- `src/pipeline/validation_pipeline.py` - Implementation
- `src/visualization/validation_plots.py` - Plotting functions
