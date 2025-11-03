# Pan-STARRS + Gaia (with g magnitude) Training Dataset Schema

**Dataset**: `panstarrs_gaia_with_gmag_training.parquet`

**Total objects**: 701,639

**Purpose**: Training temperature prediction with optical colors + brightness

## Features

### Color Features (4 total)

| Feature | Description | Wavelength Range |
|---------|-------------|------------------|
| bp_rp | Gaia BP-RP color | Optical (330-680 nm - 630-1050 nm) |
| g_r_color | Pan-STARRS g-r color | Optical (481-587 nm) |
| r_i_color | Pan-STARRS r-i color | Optical (587-755 nm) |
| i_z_color | Pan-STARRS i-z color | Optical-NIR (755-922 nm) |

### Magnitude Feature (1 total)

| Feature | Description | Unit |
|---------|-------------|------|
| phot_g_mean_mag | Gaia G-band mean magnitude | mag |

### Target Variable

| Column | Description | Unit |
|--------|-------------|------|
| teff_gspphot | Gaia GSP-Phot effective temperature | Kelvin (K) |

### Identifiers

| Column | Description |
|--------|-------------|
| source_id | Gaia DR3 source identifier |
| original_ext_source_id | Pan-STARRS source identifier |

## Data Quality

- All objects have valid measurements in all 4 colors + 1 magnitude
- Color ranges enforced based on stellar physics
- Gaia g magnitude range: 10-21 mag
- No NaN or infinite values

## Statistics

```
Objects: 701,639

Temperature:
  Range:  2888 - 31144 K
  Mean:   5308 K
  Median: 4961 K
  Std:    1411 K

Feature Ranges:
  bp_rp                 -0.395 to   3.986  (mean:   1.359)
  g_r_color             -0.500 to   2.999  (mean:   0.754)
  r_i_color             -0.500 to   1.994  (mean:   0.380)
  i_z_color             -0.500 to   1.496  (mean:   0.213)
  phot_g_mean_mag       10.297 to  19.000  (mean:  17.177)
```

## Comparison with Other Models

| Model | Colors | Magnitude | Objects | Note |
|-------|--------|-----------|---------|------|
| Unified (no gPSF) | PS + Gaia BP-RP | None | 701k | Color-only, no distance bias |
| Combined | PS + 2MASS + Gaia | None | 509k | Maximum color coverage, NIR |
| **This model** | **PS + Gaia BP-RP** | **Gaia g** | **~700k** | **Optical + brightness info** |

## Key Differences from Other Models

- **No synthetic B-V color**: Uses only measured colors
- **No 2MASS colors**: Optical-only (more objects available)
- **Includes Gaia g magnitude**: Can leverage brightness information
- **Expected to have magnitude bias**: Predictions may depend on distance/brightness
- **Higher sample size than Combined model**: More training data available

## Usage Notes

- Use `source_id` to join with Gaia catalog
- Use `original_ext_source_id` to join with Pan-STARRS catalog
- Be aware of potential magnitude bias in predictions
- Compare with color-only unified model to assess magnitude impact
