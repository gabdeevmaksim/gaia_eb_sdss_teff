# Gaia + 2MASS Engineered Features

**Dataset**: `gaia_2mass_colors_engineered_train.parquet`

**Total objects**: 525,738

**Total features**: 58

## Feature Types

- Base color features: 4
- Polynomial features: 20
- Interaction features: 6
- Log features: 4
- Temperature-dependent features: 12

## Base Features

```
- bp_rp
- j_h_color
- h_k_color
- j_k_color
```

## Polynomial Features

Degree 2 and 3 polynomials of base colors (20 features)

```
- bp_rp^2
- bp_rp^2_h_k_color
- bp_rp^2_j_h_color
- bp_rp^2_j_k_color
- bp_rp^3
- bp_rp_h_k_color^2
- bp_rp_j_h_color^2
- bp_rp_j_k_color^2
- h_k_color^2
- h_k_color^2_j_k_color
- h_k_color^3
- h_k_color_j_k_color^2
- j_h_color^2
- j_h_color^2_h_k_color
- j_h_color^2_j_k_color
- j_h_color^3
- j_h_color_h_k_color^2
- j_h_color_j_k_color^2
- j_k_color^2
- j_k_color^3
```

## Interaction Features

Pairwise products of colors (6 features)

```
- bp_rp_x_h_k_color
- bp_rp_x_j_h_color
- bp_rp_x_j_k_color
- h_k_color_x_j_k_color
- j_h_color_x_h_k_color
- j_h_color_x_j_k_color
```

## Log Features

Log transformations with offset=0.5 (4 features)

```
- log_bp_rp
- log_h_k_color
- log_j_h_color
- log_j_k_color
```

## Temperature-Dependent Features

Hot/cool/mid regime features (12 features)

```
- cool_bp_rp
- cool_h_k_color
- cool_j_h_color
- cool_j_k_color
- hot_bp_rp
- hot_h_k_color
- hot_j_h_color
- hot_j_k_color
- mid_bp_rp
- mid_h_k_color
- mid_j_h_color
- mid_j_k_color
```

## Target Variable

- `teff_gspphot`: Gaia GSP-Phot effective temperature (K)

## Common Key

- `source_id`: Gaia DR3 source identifier

## Statistics

```
Temperature range: 2888 - 32657 K
Temperature mean:  5534 K
Temperature std:   1521 K
```
