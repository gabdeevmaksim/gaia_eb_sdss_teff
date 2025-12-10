# Gaia Photometry Bands and Colors Available

## Overview

This document provides a complete inventory of Gaia photometry bands and the colors that can be calculated from them in the processed datasets.

## Gaia Photometric Bands Available

The project has access to **three Gaia photometric bands** from Gaia DR3:

| Band | Column Name (Raw) | Column Name (Unified) | Wavelength (nm) | Coverage |
|------|-------------------|----------------------|-----------------|----------|
| **G** (Optical) | `phot_g_mean_mag` | `g` | 330-1050 | 100% (1.23M sources) |
| **BP** (Blue Photometer) | `phot_bp_mean_mag` | `bp` | 330-680 | 98.1% (1.21M sources) |
| **RP** (Red Photometer) | `phot_rp_mean_mag` | `rp` | 640-1050 | 98.2% (1.21M sources) |

### Data Location

**Raw catalog** (1.23M sources):
- File: `data/processed/eb_catalog.parquet`
- Contains original Gaia column names (`phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`)

**Unified photometry** (2.18M sources - includes Pan-STARRS + 2MASS):
- File: `data/processed/eb_unified_photometry.parquet`
- Contains short column names (`g`, `bp`, `rp`, `bp_rp`)
- Missing values encoded as `-999.0`
- Includes merged Pan-STARRS and 2MASS photometry

## Gaia-Only Colors Available

From the three available Gaia bands, three colors can be directly calculated:

### 1. **BP-RP Color** (Pre-calculated)
- **Formula**: BP - RP
- **Column**: `bp_rp` (in unified dataset)
- **Range**: -4.59 to 6.71 (typical stellar range: -0.5 to 3.0)
- **Coverage**: 97.2% (2.12M valid sources)
- **Physical Meaning**: Overall color index; redder for cooler stars
- **Stellar Classification**:
  - `bp_rp < 0.5`: Hot stars (>8000 K, OBA types)
  - `0.5 ≤ bp_rp ≤ 2.0`: Mid-range stars (5000-8000 K, FGK types)
  - `bp_rp > 2.0`: Cool stars (<5000 K, M types)

### 2. **G-BP Color** (Can be calculated)
- **Formula**: G - BP
- **Range**: Typically -2 to 2 magnitudes
- **Physical Meaning**: Bluer than BP-RP; sensitive to hot star properties
- **Status**: Not pre-calculated, but can be derived from available bands
- **Advantage**: Provides different sensitivity to stellar parameters

### 3. **G-RP Color** (Can be calculated)
- **Formula**: G - RP
- **Range**: Typically -2 to 2 magnitudes
- **Physical Meaning**: Intermediate color; sensitivity between G-BP and BP-RP
- **Status**: Not pre-calculated, but can be derived from available bands

## Currently Used Gaia-Only Datasets

The project has created several Gaia-only feature datasets for temperature prediction:

### 1. **Basic Gaia Color Models**

#### Gaia G + BP-RP
- **Dataset**: `gaia_g_bprp_train.parquet` (737k objects)
- **Features**: 
  - `phot_g_mean_mag` (G-band magnitude)
  - `bp_rp` (color)
- **Purpose**: Minimal 2-feature model testing
- **Model**: `rf_gaia_g_bprp_20251105_...`

#### Gaia Colors (bp, g, rp + bp_rp)
- **Config**: `config/models/gaia_colors_teff.yaml`
- **Features**:
  - `g` (G magnitude)
  - `bp` (BP magnitude)
  - `rp` (RP magnitude)
  - `bp_rp` (BP-RP color)
- **Purpose**: 4-feature baseline model
- **Model**: `rf_gaia_colors_teff_20251105_...`

#### Gaia Multioutput (Simultaneous Teff, logg, [Fe/H])
- **Config**: `config/models/gaia_multioutput.yaml`
- **Features**:
  - `bp_rp` (BP-RP color)
  - `g` (G magnitude)
  - `bp` (BP magnitude)
  - `rp` (RP magnitude)
- **Engineered Features**:
  - `bp_rp^2`, `bp_rp^3`
  - `bp_g` = BP - G
  - `g_rp` = G - RP
- **Purpose**: Predict multiple stellar parameters simultaneously
- **Status**: Polynomial feature engineering enabled

### 2. **Engineered Gaia G + BP-RP Features**

#### Dataset: `gaia_g_bprp_engineered_train.parquet` (737k objects)
- **Base Features**: `phot_g_mean_mag`, `bp_rp`
- **Engineered Features** (17 total):
  
  **Polynomial Terms**:
  - `bp_rp^2`, `bp_rp^3`
  - `g_mag^2`, `g_mag^3`
  
  **Interaction Terms**:
  - `g_mag_x_bp_rp` = G × BP-RP
  - `g_mag_bp_rp^2` = G × (BP-RP)²
  - `g_mag^2_bp_rp` = G² × BP-RP
  
  **Logarithmic Terms**:
  - `log_bp_rp` = log(BP-RP + 1.0)
  
  **Temperature-Dependent Features**:
  - `hot_bp_rp`, `hot_g_mag` (for BP-RP < 0.5)
  - `cool_bp_rp`, `cool_g_mag` (for BP-RP > 2.0)
  - `mid_bp_rp`, `mid_g_mag` (for 0.5 ≤ BP-RP ≤ 2.0)
  - `log_hot_bp_rp`, `log_cool_bp_rp`, `log_mid_bp_rp`

- **Model**: `rf_gaia_g_bprp_engineered_20251016_...`

## Gaia + 2MASS Hybrid Datasets

### Gaia + 2MASS IR (Separate color inputs)
- **Dataset**: `gaia_2mass_ir_train.parquet` (525k objects)
- **Gaia Features**:
  - `phot_g_mean_mag` (G-band magnitude)
  - `bp_rp` (BP-RP color)
- **2MASS Features**:
  - `j_h_color` (J - H)
  - `h_k_color` (H - K)
  - `j_k_color` (J - K)
- **Purpose**: Combine optical (Gaia) + infrared (2MASS) colors
- **Model**: `rf_gaia_2mass_ir_20251103_...`

### Gaia + 2MASS Engineered Colors
- **Dataset**: `gaia_2mass_colors_engineered_train.parquet` (525k objects)
- **Base Features**:
  - `bp_rp` (Gaia color)
  - `j_h_color`, `h_k_color`, `j_k_color` (2MASS colors)
- **Engineered Features**: 56 features including:
  - Polynomial terms (squared)
  - All interaction combinations
  - Logarithmic transforms
  - Temperature-dependent indicators
- **Model**: `rf_gaia_2mass_engineered_20251016_...`

## Column Name Mapping

### Raw Dataset (`eb_catalog.parquet`)
```
phot_g_mean_mag     → G-band magnitude
phot_bp_mean_mag    → BP-band magnitude
phot_rp_mean_mag    → RP-band magnitude
bp_rp               → Pre-calculated BP-RP color (if present)
teff_gspphot        → Gaia GSP-Phot effective temperature (training target)
```

### Unified Dataset (`eb_unified_photometry.parquet`)
```
g                   → phot_g_mean_mag
bp                  → phot_bp_mean_mag
rp                  → phot_rp_mean_mag
bp_rp               → Pre-calculated BP-RP color
```

## Data Quality and Coverage

### Coverage by Band (Unified Photometry)
| Band | Total | Valid | Coverage |
|------|-------|-------|----------|
| G | 2,184,477 | 2,184,477 | 100.0% |
| BP | 2,184,477 | 2,124,104 | 97.2% |
| RP | 2,184,477 | 2,125,723 | 97.3% |
| BP-RP | 2,184,477 | 2,123,652 | 97.2% |

### Training Target (Gaia GSP-Phot Teff)
- Coverage: 59.9% of raw catalog (737,028 / 1,230,649)
- Range: ~2500 K to 50000 K
- Used for all Gaia temperature prediction models

## Scripts for Creating Gaia Color Datasets

### Basic Gaia Color Dataset Creation
```bash
# Create G + BP-RP dataset
python scripts/create_gaia_g_bprp_dataset.py

# Create engineered G + BP-RP features
python scripts/create_gaia_g_bprp_engineered_dataset.py

# Create Gaia + 2MASS IR combined dataset
python scripts/create_gaia_2mass_ir_dataset.py

# Create engineered Gaia + 2MASS colors
python scripts/create_gaia_2mass_engineered_features.py
```

### Model Training
```bash
# Train basic Gaia G + BP-RP model
python scripts/train_gaia_g_bprp_model.py

# Train engineered Gaia model
python scripts/train_gaia_g_bprp_engineered_model.py

# Train Gaia + 2MASS IR model
python scripts/train_gaia_2mass_ir_model.py
```

## Potential Additional Gaia Colors (Not Currently Used)

While the project currently focuses on BP-RP and the three basic bands, Gaia DR3 also provides:

- **RVS Band** (radial velocity spectrograph): Not used in this project
- **Cross-band indices**: Could calculate additional indices like:
  - `(bp - rp) / (bp + rp)` (normalized color)
  - `g / (bp + rp)` (magnitude-color combinations)

These could be useful for future feature engineering if needed.

## Summary

### Available Gaia Photometry
- **3 photometric bands**: G, BP, RP
- **Immediate colors**: BP-RP (pre-calculated)
- **Calculable colors**: G-BP, G-RP
- **Coverage**: 97-100% for all bands

### Current Usage in Models
1. **Gaia-only models**: Using BP-RP color ± G magnitude
2. **Hybrid models**: Combining Gaia optical + 2MASS infrared
3. **Feature engineering**: Polynomial, interaction, and temperature-dependent transforms

### Key Insight
**BP-RP is the dominant Gaia color** for eclipsing binary temperature prediction, appearing in almost all models and representing ~60% of feature importance in engineered models. The addition of G-band magnitude helps but BP-RP color alone provides ~80% of predictive power.

