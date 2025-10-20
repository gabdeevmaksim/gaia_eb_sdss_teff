# Gaia + 2MASS Colors Training Dataset Schema

**File:** `data/processed/gaia_2mass_colors_training.parquet`
**Created:** 2025-10-16 15:46:19
**Total Objects:** 525,738

## Description
Clean training dataset containing only Gaia and 2MASS color indices (no magnitude features).
Includes objects with Gaia GSP-Phot effective temperatures for supervised learning.

## Common Key
- **source_id** (int64): Gaia DR3 source identifier - unique key for joining with other tables

## Columns

### Identifier
- **source_id** (int64): Gaia DR3 source identifier

### Feature Columns (4 colors)
- **bp_rp** (float64): Gaia BP-RP color [mag]
  - Range: -0.405 to 3.995
  - Mean: 1.351 ± 0.473

- **j_h_color** (float64): 2MASS J-H color [mag]
  - Range: -0.500 to 1.998
  - Mean: 0.469 ± 0.228

- **h_k_color** (float64): 2MASS H-K color [mag]
  - Range: -0.300 to 1.000
  - Mean: 0.187 ± 0.200

- **j_k_color** (float64): 2MASS J-K color [mag]
  - Range: -0.500 to 2.497
  - Mean: 0.656 ± 0.303

### Target Column
- **teff_gspphot** (float64): Gaia GSP-Phot effective temperature [K]
  - Range: 2888.2 to 32656.8 K
  - Mean: 5533.8 ± 1520.7 K
  - Median: 5156.1 K

## Data Quality
- No missing values in any column
- Color values filtered to physically realistic ranges:
  - bp_rp: -0.5 to 4.0 mag
  - j_h_color: -0.5 to 2.0 mag
  - h_k_color: -0.3 to 1.0 mag
  - j_k_color: -0.5 to 2.5 mag
- Temperatures filtered to 2,500 - 50,000 K

## Source Tables
1. **eb_catalog.parquet**: Gaia source_id, bp_rp color, teff_gspphot
2. **eb_2mass_photometry.parquet**: 2MASS infrared magnitudes and colors

## Usage
```python
import pandas as pd
df = pd.read_parquet('data/processed/gaia_2mass_colors_training.parquet')

# Features for ML
features = ['bp_rp', 'j_h_color', 'h_k_color', 'j_k_color']
X = df[features]
y = df['teff_gspphot']

# Gaia source_id for joins
source_ids = df['source_id']
```

## Notes
- All objects have both Gaia photometry and 2MASS infrared photometry
- Dataset is clean and ready for machine learning
- No magnitude features included (following best practice to avoid distance bias)
- Common key (source_id) enables joining with other Gaia-based tables
