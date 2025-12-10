# Prediction Configuration Files

Configuration files for temperature prediction using trained models.

## Usage

```bash
# Predict using configuration file
python pipeline.py predict --config config/prediction/predict_gaia_2mass_ir.yaml

# Or use Python API
from src.pipeline import PredictionPipeline
pipeline = PredictionPipeline('config/prediction/predict_gaia_2mass_ir.yaml')
context = pipeline.run()
```

## Available Configurations

### `predict_gaia_2mass_ir.yaml`
- **Model:** Gaia + 2MASS infrared
- **Features:** phot_g_mean_mag, bp_rp, j_h_color, h_k_color, j_k_color
- **Data:** `gaia_2mass_ir_predict.parquet`

### `predict_panstarrs_unified.yaml`
- **Model:** PanSTARRS unified color-only (RECOMMENDED)
- **Features:** 85 engineered color features
- **Data:** `eb_unified_features_engineered_predict.parquet`

## Configuration Schema

```yaml
model:
  # Model file (supports wildcards)
  model_file: "rf_model_*.pkl"

data:
  # Input data file
  source_file: "prediction_data.parquet"

  # ID column for tracking
  id_column: "source_id"

preprocessing:
  # Filter missing values
  filter_missing: true
  missing_value: -999.0

feature_engineering:
  # Apply feature engineering (must match training)
  enabled: false

  # If enabled, specify columns (must match training config)
  color_cols: []
  mag_cols: []

output:
  # Output filename
  output_file: "predictions.parquet"

  # Optional: custom output directory
  # output_dir: "data/predictions"

  # Include input features in output
  include_features: false

  # Include specific columns from input
  include_columns:
    - "source_id"
    - "ra"
    - "dec"

  # Save summary text file
  save_summary: true
```

## Creating New Prediction Configs

1. **Copy a template:**
   ```bash
   cp config/prediction/predict_gaia_2mass_ir.yaml config/prediction/my_prediction.yaml
   ```

2. **Update model reference:**
   ```yaml
   model:
     model_file: "my_model_*.pkl"  # Wildcard for latest
     # or
     model_file: "my_model_20251103_141119.pkl"  # Exact version
   ```

3. **Update data source:**
   ```yaml
   data:
     source_file: "my_prediction_data.parquet"
   ```

4. **Match feature engineering to training:**
   - If training used `feature_engineering.enabled: true`, set same here
   - Use same `color_cols` and `mag_cols` as training

5. **Configure output:**
   ```yaml
   output:
     output_file: "my_predictions.parquet"
     include_columns:
       - "source_id"
       # Add any columns you want in output
   ```

## Important Notes

### Feature Engineering Consistency

⚠️ **Critical:** Feature engineering settings MUST match training configuration!

- If model was trained with `feature_engineering.enabled: true`:
  - Set `enabled: true` here
  - Use SAME `color_cols` and `mag_cols`

- If model was trained with pre-engineered features:
  - Set `enabled: false`
  - Ensure prediction data has same engineered features

### Model Wildcards

Using wildcards like `rf_model_*.pkl` will select the **most recent** matching model based on filename timestamp.

**Example:**
```
models/rf_gaia_2mass_ir_20251101_120000.pkl
models/rf_gaia_2mass_ir_20251103_141119.pkl  ← Selected (most recent)
```

To use a specific version, use exact filename.

### Output Formats

Supported output formats:
- `.parquet` (recommended, compressed)
- `.csv` (human-readable)

Format determined by output file extension.

## Examples

### Basic Prediction

```bash
python pipeline.py predict --config config/prediction/predict_gaia_2mass_ir.yaml
```

**Output:**
```
Predictions saved: data/processed/predictions_gaia_2mass_ir.parquet
Summary saved: data/processed/predictions_gaia_2mass_ir.txt
Objects: 401,111
Mean temperature: 4,862 K
```

### Prediction with Custom Output

```yaml
output:
  output_file: "my_predictions.csv"
  output_dir: "results/predictions"
  include_features: true  # Include all features used
  include_columns:
    - "source_id"
    - "ra"
    - "dec"
    - "period"
```

### Batch Predictions

Predict with multiple models:

```bash
for config in config/prediction/*.yaml; do
    echo "Running: $config"
    python pipeline.py predict --config "$config"
done
```

## Output Files

### Predictions File

**Parquet/CSV with columns:**
- `teff_predicted` - Predicted temperature (K)
- `source_id` - Object identifier (if available)
- Any columns specified in `include_columns`
- Input features (if `include_features: true`)

### Summary File

**Text file with:**
- Model information
- Input data source
- Prediction statistics (mean, std, min, max)
- Output file details

**Example summary:**
```
Prediction Summary
================================================================================

Model: rf_gaia_2mass_ir_20251103_141119
Model file: models/rf_gaia_2mass_ir_20251103_141119.pkl
Timestamp: 2025-11-03 14:30:15

Input data: gaia_2mass_ir_predict.parquet
Objects: 401,111

Prediction Statistics:
  Mean:  4862.3 K
  Std:   1245.7 K
  Min:   2500.0 K
  Max:   9500.0 K

Output file: predictions_gaia_2mass_ir.parquet
Output columns: 4
```

## Troubleshooting

### Error: "Model not found"

```
FileNotFoundError: No model found matching: rf_model_*.pkl
```

**Solution:** Check model exists in `models/` directory:
```bash
ls models/rf_model_*.pkl
```

### Error: "Prediction data not found"

```
FileNotFoundError: Prediction data not found: data/processed/my_data.parquet
```

**Solution:** Verify data file exists:
```bash
ls data/processed/my_data.parquet
```

### Error: "Missing required features"

```
ValueError: Missing required features: ['j_h_color', 'h_k_color']
```

**Solution:** Input data doesn't have all features needed by model. Either:
1. Use dataset with those features
2. Use different model
3. If features should be engineered, enable `feature_engineering.enabled: true`

### Wrong Temperature Range

Predictions seem unrealistic (too high/low)?

**Possible causes:**
1. Feature engineering mismatch (training vs. prediction)
2. Wrong model for this data
3. Input data quality issues

**Check:**
- Model metadata: what features does it expect?
- Feature engineering: same as training?
- Input data: valid ranges for all features?

## See Also

- `docs/PIPELINES.md` - Complete pipeline documentation
- `config/models/` - Training configurations
- `src/pipeline/prediction_pipeline.py` - Implementation
