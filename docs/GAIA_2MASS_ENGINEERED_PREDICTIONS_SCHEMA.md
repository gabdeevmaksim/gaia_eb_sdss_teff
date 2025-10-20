# Gaia + 2MASS Engineered Model Predictions Schema

**Dataset**: `gaia_2mass_engineered_predictions_rf_gaia_2mass_engineered_20251016_205640.parquet`

**Model**: rf_gaia_2mass_engineered_20251016_205640

**Total predictions**: 101,508

## Columns

| Column | Type | Description |
|--------|------|-------------|
| source_id | int64 | Gaia DR3 source identifier (common key) |
| teff_predicted | float64 | Predicted effective temperature (K) |
| bp_rp | float64 | Gaia BP-RP color |
| j_h_color | float64 | 2MASS J-H color |
| h_k_color | float64 | 2MASS H-K color |
| j_k_color | float64 | 2MASS J-K color |
| j_snr | float64 | 2MASS J-band signal-to-noise ratio |
| h_snr | float64 | 2MASS H-band signal-to-noise ratio |
| k_snr | float64 | 2MASS K-band signal-to-noise ratio |
| ph_qual | str | 2MASS photometric quality flag (AAA=best) |
| model_id | str | Model identifier |

## Statistics

```
Temperature range: 3311 - 15625 K
Temperature mean:  5139 K
Temperature median: 4851 K
Temperature std:   1043 K

High quality 2MASS (AAA/AAB): 53,890 (53.1%)
```

## Model Performance (Test Set)

```
MAE:  722.3 K
RMSE: 1138.0 K
R²:   0.442
Within 10%: 55.3%
```

## Common Key

Use `source_id` to join with other catalogs (Gaia, Pan-STARRS, etc.)
