# Gaia + 2MASS Temperature Predictions Schema

**File:** `data/processed/gaia_2mass_temperature_predictions_rf_gaia_2mass_colors_20251016_155128.parquet`
**Created:** 2025-10-16 20:04:50
**Total Objects:** 101,508

## Description
Predicted effective temperatures for eclipsing binaries without Gaia GSP-Phot temperatures.
Uses Random Forest model trained on Gaia BP-RP and 2MASS J-H, H-K, J-K colors.

## Common Key
- **source_id** (int64): Gaia DR3 source identifier - unique key for joining with other tables

## Columns

### Identifier
- **source_id** (int64): Gaia DR3 source identifier

### Feature Columns (4 colors)
- **bp_rp** (float64): Gaia BP-RP color [mag]
  - Range: -0.349 to 4.000
  - Mean: 1.795 ± 0.632

- **j_h_color** (float64): 2MASS J-H color [mag]
  - Range: -0.499 to 2.000
  - Mean: 0.588 ± 0.292

- **h_k_color** (float64): 2MASS H-K color [mag]
  - Range: -0.300 to 1.000
  - Mean: 0.251 ± 0.221

- **j_k_color** (float64): 2MASS J-K color [mag]
  - Range: -0.499 to 2.500
  - Mean: 0.838 ± 0.385

### Prediction Column
- **teff_predicted** (float64): Predicted effective temperature [K]
  - Range: 3346.1 to 14466.9 K
  - Mean: 5133.7 ± 1046.8 K
  - Median: 4851.2 K

### Quality Columns
- **ph_qual** (str): 2MASS photometric quality flag (e.g., 'AAA', 'AAB')
- **j_snr** (float64): J-band signal-to-noise ratio
- **h_snr** (float64): H-band signal-to-noise ratio
- **k_snr** (float64): K-band signal-to-noise ratio

## Model Information
- **Model ID:** rf_gaia_2mass_colors_20251016_155128
- **Training objects:** 525,738
- **Test MAE:** 722 K
- **Test R²:** 0.442
- **Features:** 4 color indices (no magnitudes)

## Data Quality
- All objects have valid colors in all 4 bands
- Color values filtered to physically realistic ranges:
  - bp_rp: -0.5 to 4.0 mag
  - j_h_color: -0.5 to 2.0 mag
  - h_k_color: -0.3 to 1.0 mag
  - j_k_color: -0.5 to 2.5 mag
- High quality 2MASS (AAA/AAB/ABA/ABB): 56.0%

## Source Tables
1. **eb_catalog.parquet**: Gaia source_id, bp_rp color
2. **eb_2mass_photometry.parquet**: 2MASS infrared colors

## Usage
```python
import pandas as pd

# Load predictions
df = pd.read_parquet('data/processed/gaia_2mass_temperature_predictions_rf_gaia_2mass_colors_20251016_155128.parquet')

# Access predicted temperatures
temps = df['teff_predicted']

# Join with other tables using source_id
# df_merged = df.merge(other_table, on='source_id')
```

## Notes
- These predictions are for objects WITHOUT Gaia GSP-Phot temperatures
- Model is color-only (no magnitude bias, distance-independent)
- Common key (source_id) enables joining with other Gaia-based catalogs
- For training data with true Gaia Teff, see gaia_2mass_colors_training.parquet
