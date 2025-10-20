# Combined Pan-STARRS + 2MASS Predictions Schema

**Dataset**: `combined_predictions_rf_combined_colors_20251018_115824.parquet`

**Model**: rf_combined_colors_20251018_115824

**Total predictions**: 96,100

## Columns

| Column | Type | Description |
|--------|------|-------------|
| source_id | int64 | Gaia DR3 source identifier (common key) |
| original_ext_source_id | int64 | Pan-STARRS source identifier |
| teff_predicted | float64 | Predicted effective temperature (K) |
| bp_rp | float64 | Gaia BP-RP color |
| g_r_color | float64 | Pan-STARRS g-r color |
| r_i_color | float64 | Pan-STARRS r-i color |
| i_z_color | float64 | Pan-STARRS i-z color |
| B_V_color | float64 | Synthetic B-V color |
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
Temperature range: 3218 - 14677 K
Temperature mean:  5142 K
Temperature median: 4877 K
Temperature std:   1087 K

High quality 2MASS (AAA/AAB): 50,539 (52.6%)
```

## Temperature Distribution

```
<4000 K          8,061 (  8.4%)
4000-5000 K     44,050 ( 45.8%)
5000-6000 K     24,993 ( 26.0%)
6000-8000 K     17,200 ( 17.9%)
>8000 K          1,796 (  1.9%)
```

## Model Performance (Test Set)

```
MAE:  694.6 K
RMSE: 1080.3 K
R²:   0.477
Within 10%: 56.5%
```

## Feature Importance

```
j_h_color  (37.95%) - Most important!
bp_rp      (23.54%)
h_k_color  (12.05%)
i_z_color  ( 7.59%)
r_i_color  ( 5.89%)
j_k_color  ( 5.68%)
g_r_color  ( 3.66%)
B_V_color  ( 3.64%)
```

## Common Keys

- Use `source_id` to join with Gaia catalog
- Use `original_ext_source_id` to join with Pan-STARRS catalog

## Wavelength Coverage

Predictions use photometry spanning:
- **Optical**: 330 nm (Gaia BP) to 1000 nm (Pan-STARRS z)
- **Near-IR**: 1.25 μm (2MASS J) to 2.17 μm (2MASS K)

**Total range**: 330 nm to 2.17 μm (~4 orders of magnitude)
