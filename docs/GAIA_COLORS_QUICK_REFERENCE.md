# Gaia Colors Quick Reference

## Available Gaia Photometry at a Glance

### The 3 Gaia Bands
```
G  (330-1050 nm)  → phot_g_mean_mag  or  g    [100.0% coverage]
BP (330-680 nm)   → phot_bp_mean_mag or  bp   [98.1% coverage]
RP (640-1050 nm)  → phot_rp_mean_mag or  rp   [98.2% coverage]
```

### The 3 Gaia Colors Available
```
1. BP-RP = phot_bp_mean_mag - phot_rp_mean_mag  [DOMINANT - 97.2% coverage]
2. G-BP  = phot_g_mean_mag - phot_bp_mean_mag   [calculable, used in multioutput]
3. G-RP  = phot_g_mean_mag - phot_rp_mean_mag   [calculable, used in multioutput]
```

## Where to Find Them

| Dataset | File | Columns | Format |
|---------|------|---------|--------|
| Raw Catalog | `data/processed/eb_catalog.parquet` | `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag` | Parquet |
| Unified Phot. | `data/processed/eb_unified_photometry.parquet` | `g`, `bp`, `rp`, `bp_rp` | Parquet |

## Feature Engineering from Gaia Colors

### Basic Engineering (Gaia G + BP-RP only)
Script: `scripts/create_gaia_g_bprp_engineered_dataset.py`

Creates 17 features from 2 base inputs:
- Polynomials: `bp_rp^2`, `bp_rp^3`, `g_mag^2`, `g_mag^3`
- Interactions: `g_mag_x_bp_rp`, `g_mag_bp_rp^2`, `g_mag^2_bp_rp`
- Logs: `log_bp_rp`
- Temp-dependent: `hot_bp_rp`, `cool_bp_rp`, `mid_bp_rp` (and magnitude variants)

### Advanced Engineering (Gaia + 2MASS)
Script: `scripts/create_gaia_2mass_engineered_features.py`

Creates 56 features from 4 base colors:
- All squared terms
- All interaction combinations
- Logarithmic transforms
- Temperature regimes

## Quick Usage Examples

### Load Gaia Photometry
```python
import polars as pl

# Option 1: Raw names (from main catalog)
df = pl.read_parquet('data/processed/eb_catalog.parquet')
g_mag = df['phot_g_mean_mag']
bp_mag = df['phot_bp_mean_mag']
rp_mag = df['phot_rp_mean_mag']

# Option 2: Short names (from unified dataset)
df = pl.read_parquet('data/processed/eb_unified_photometry.parquet')
g_mag = df['g']
bp_mag = df['bp']
rp_mag = df['rp']
bp_rp_color = df['bp_rp']
```

### Calculate Gaia Colors
```python
# BP-RP (already pre-calculated)
color_bp_rp = df['bp_rp']

# Calculate G-BP
color_g_bp = df['g'] - df['bp']

# Calculate G-RP
color_g_rp = df['g'] - df['rp']
```

### Classification by BP-RP
```python
# Classify stars by color
hot_stars = df.filter(df['bp_rp'] < 0.5)      # >8000K
cool_stars = df.filter(df['bp_rp'] > 2.0)     # <5000K
mid_stars = df.filter((df['bp_rp'] >= 0.5) & (df['bp_rp'] <= 2.0))
```

## Feature Importance Hierarchy

Based on models trained in this project:

1. **BP-RP Color** - 60% of importance (DOMINANT)
   - Works for all star types
   - Pre-calculated and complete

2. **G Magnitude** - 20% of importance
   - Adds brightness information
   - Helps calibration

3. **G-BP / G-RP Colors** - 10% of importance
   - Useful for multioutput models
   - Better for stellar classification

4. **BP/RP Individual Magnitudes** - 10% of importance
   - Some redundancy with BP-RP
   - Used only in 4-feature baseline

## Model Recommendations

### For Temperature Prediction
- Best single model: Use BP-RP + G magnitude (2 features)
- Best baseline: Use g, bp, rp, bp_rp (4 features)
- Best accuracy: Add 2MASS infrared colors (5+ colors)

### For Stellar Classification
- Multi-parameter model: Use gaia_multioutput.yaml
- Includes engineered G-BP and G-RP colors
- Predicts Teff, logg, [Fe/H] simultaneously

### For Hot/Cool Stars
- Use temperature-dependent features
- bp_rp < 0.5 features for hot stars
- bp_rp > 2.0 features for cool stars

## Data Quality Notes

- G-band: 100% coverage (most reliable)
- BP/RP: ~98% coverage (97.2% valid non-missing)
- BP-RP: 97.2% coverage (pre-calculated, fewer errors)
- Missing values: Encoded as -999.0 in unified dataset
- Good enough for robust ML modeling

## Related Documentation

- Full details: `docs/GAIA_PHOTOMETRY_AVAILABLE.md`
- Multioutput config: `config/models/gaia_multioutput.yaml`
- Feature engineering: `scripts/create_gaia_g_bprp_engineered_dataset.py`

