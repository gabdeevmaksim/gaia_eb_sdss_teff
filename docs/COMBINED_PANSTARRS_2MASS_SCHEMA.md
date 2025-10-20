# Combined Pan-STARRS + 2MASS + Gaia Training Dataset Schema

**Dataset**: `combined_panstarrs_2mass_colors_training.parquet`

**Total objects**: 508,626

**Purpose**: Training temperature prediction models with maximum color coverage

## Features

### Color Features (8 total)

| Feature | Description | Wavelength Range |
|---------|-------------|------------------|
| bp_rp | Gaia BP-RP color | Optical (330-680 nm - 630-1050 nm) |
| g_r_color | Pan-STARRS g-r color | Optical (481-587 nm) |
| r_i_color | Pan-STARRS r-i color | Optical (587-755 nm) |
| i_z_color | Pan-STARRS i-z color | Optical-NIR (755-922 nm) |
| B_V_color | Synthetic B-V color | Optical (445-551 nm) |
| j_h_color | 2MASS J-H color | NIR (1.25-1.65 μm) |
| h_k_color | 2MASS H-K color | NIR (1.65-2.17 μm) |
| j_k_color | 2MASS J-K color | NIR (1.25-2.17 μm) |

### Target Variable

| Column | Description | Unit |
|--------|-------------|------|
| teff_gspphot | Gaia GSP-Phot effective temperature | Kelvin (K) |

### Quality Indicators

| Column | Description |
|--------|-------------|
| j_snr | 2MASS J-band signal-to-noise ratio |
| h_snr | 2MASS H-band signal-to-noise ratio |
| k_snr | 2MASS K-band signal-to-noise ratio |
| ph_qual | 2MASS photometric quality flag (AAA=best) |

### Common Keys

| Column | Description |
|--------|-------------|
| source_id | Gaia DR3 source identifier |
| original_ext_source_id | Pan-STARRS source identifier |

## Data Quality

- All objects have valid measurements in all 8 colors
- Color ranges enforced based on stellar physics
- No NaN or infinite values
- 60.8% have high-quality 2MASS photometry (AAA/AAB)

## Statistics

```
Objects: 508,626

Temperature:
  Range:  2888 - 31144 K
  Mean:   5511 K
  Median: 5145 K
  Std:    1497 K

Color Ranges:
  bp_rp           -0.395 to  3.986  (mean:  1.359)
  g_r_color       -0.500 to  2.999  (mean:  0.752)
  r_i_color       -0.500 to  1.994  (mean:  0.380)
  i_z_color       -0.500 to  1.496  (mean:  0.215)
  B_V_color       -0.215 to  2.904  (mean:  0.901)
  j_h_color       -0.500 to  1.998  (mean:  0.471)
  h_k_color       -0.300 to  1.000  (mean:  0.188)
  j_k_color       -0.500 to  2.497  (mean:  0.659)
```

## Wavelength Coverage

This dataset spans from optical to near-infrared:
- **Optical**: Pan-STARRS grizy (400-1000 nm) + Gaia BP/RP (330-1050 nm)
- **Near-IR**: 2MASS JHK (1.25-2.17 μm)

Combined coverage: **~330 nm to 2.17 μm**

## Comparison with Other Datasets

| Dataset | Objects | Colors | Coverage |
|---------|---------|--------|----------|
| Gaia only | 737,028 | 1 (BP-RP) | Optical |
| Gaia + Pan-STARRS | 701,644 | 5 | Optical |
| Gaia + 2MASS | 525,738 | 4 | Optical + NIR |
| **Combined (this)** | **508,626** | **8** | **Optical + NIR** |

## Usage Notes

- Use `source_id` to join with Gaia catalog
- Use `original_ext_source_id` to join with Pan-STARRS catalog
- This is the most complete color dataset but has fewer objects than individual datasets
- Best for exploring maximum predictive power with all available colors
