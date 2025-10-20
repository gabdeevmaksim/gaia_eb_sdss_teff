# 2MASS Photometry Data Summary

**Dataset:** `data/processed/eb_2mass_photometry.parquet`
**Total Objects:** 1,123,471 eclipsing binaries with 2MASS photometry
**Data Source:** 2MASS All-Sky Point Source Catalog via IRSA

## Dataset Overview

### Columns (17 total)
- **Identifiers:** designation, ra, dec, original_ext_source_id, source_id (Gaia)
- **J-band:** j_m, j_msigcom, j_snr
- **H-band:** h_m, h_msigcom, h_snr
- **K-band:** k_m, k_msigcom, k_snr
- **Quality flags:** ph_qual, cc_flg, gal_contam

## Photometric Completeness

| Band | Objects with magnitude | Objects with error | Completeness |
|------|----------------------:|------------------:|-------------:|
| J    | 1,123,471 (100.0%)    | 1,102,106 (98.1%) | Excellent    |
| H    | 1,123,471 (100.0%)    | 1,024,151 (91.2%) | Very Good    |
| K    | 1,123,471 (100.0%)    | 865,573 (77.0%)   | Good         |

**Note:** All objects have magnitudes in all three bands, but not all have valid errors (some are upper limits).

## Magnitude Statistics

| Band | Mean ± Std     | Range          | Median |
|------|----------------|----------------|--------|
| J    | 14.99 ± 1.37   | 3.37 - 19.56   | 15.19  |
| H    | 14.50 ± 1.32   | 2.84 - 20.72   | 14.72  |
| K    | 14.31 ± 1.31   | 2.74 - 17.53   | 14.53  |

**Typical eclipsing binary:** J ≈ 15, H ≈ 14.5, K ≈ 14.3

## Signal-to-Noise Ratios

| Band | Mean SNR | Median SNR | Note                           |
|------|----------|------------|--------------------------------|
| J    | 150.7    | 22.4       | Very high quality              |
| H    | 97.2     | 17.7       | High quality                   |
| K    | 70.1     | 14.3       | Good quality (fainter sources) |

**Interpretation:** High mean SNR but lower median indicates excellent detections for bright sources, with some fainter sources having lower but still usable SNR.

## 2MASS Color Indices

| Color | Mean ± Std     | Range              | Valid Objects  |
|-------|----------------|--------------------| ---------------|
| J-H   | 0.492 ± 0.299  | -4.04 to +4.83     | 100%           |
| H-K   | 0.196 ± 0.316  | -4.06 to +4.64     | 100%           |
| J-K   | 0.688 ± 0.419  | -2.94 to +6.55     | 100%           |

**Typical colors for main-sequence stars:**
- J-H ≈ 0.5 (consistent with our sample)
- H-K ≈ 0.2 (consistent with our sample)
- J-K ≈ 0.7 (consistent with our sample)

**Color ranges include:**
- Blue colors (negative): Hot stars (early-type)
- Typical colors (0.3-0.8): Solar-type and cooler stars
- Red colors (>1.0): Cool giants/dwarfs, reddened stars

## Photometric Quality Flags

### ph_qual Distribution (top 10)

| Quality | Count     | Percentage | Description                    |
|---------|-----------|------------|--------------------------------|
| AAA     | 521,256   | 46.4%      | SNR > 10 in all bands          |
| AAB     | 108,055   | 9.6%       | SNR > 10 in J,H; 7-10 in K     |
| BUU     | 45,773    | 4.1%       | Good J, upper limits H,K       |
| ABC     | 44,550    | 4.0%       | Decreasing quality J→H→K       |
| BCU     | 42,542    | 3.8%       | Good J,H, upper limit K        |
| AAC     | 36,772    | 3.3%       | SNR > 10 in J,H; 5-7 in K      |
| ABU     | 32,736    | 2.9%       | Good J,H, upper limit K        |
| ABB     | 25,955    | 2.3%       | SNR > 10 in J; 7-10 in H,K     |
| BBU     | 24,356    | 2.2%       | SNR 7-10 in J,H, upper limit K |
| AUU     | 24,289    | 2.2%       | Good J, upper limits H,K       |

**Quality grade meanings:**
- **A:** SNR > 10 (excellent, < 10% error)
- **B:** 7 < SNR < 10 (good, 10-15% error)
- **C:** 5 < SNR < 7 (fair, 15-20% error)
- **D:** 3 < SNR < 5 (marginal, 20-35% error)
- **E:** SNR < 3 (poor, > 35% error)
- **U:** Upper limit (non-detection)
- **X:** Detection but no valid magnitude

### Contamination Flags

**cc_flg (Contamination and Confusion):**
- **000:** 830,521 (73.9%) - No contamination issues
- **ccc:** 89,035 (7.9%) - Confusion in all bands (crowded field)
- **c00:** 67,006 (6.0%) - Confusion in J-band only
- **cc0:** 58,732 (5.2%) - Confusion in J and H bands

**Flag meanings:**
- **0:** No contamination/confusion
- **c:** Confusion (multiple sources in aperture)
- **p:** Persistence artifact
- **d:** Diffraction spike
- **s:** Stripe artifact

**gal_contam (Galactic contamination):**
- **0:** 1,123,069 (99.96%) - No galactic contamination
- **1-2:** 402 (0.04%) - Minor galactic contamination

## High-Quality Data Subsets

### Quality Selection Criteria

| Criterion                              | Objects    | Percentage |
|----------------------------------------|-----------|------------|
| AAA or AAB or ABA or ABB quality       | 658,929   | 58.7%      |
| SNR > 10 in all three bands            | 547,874   | 48.8%      |
| **Both criteria (RECOMMENDED)**        | **538,042** | **47.9%**  |

**Recommended subset for scientific analysis:**
- **538,042 objects (47.9%)** with AAA/AAB/ABA/ABB quality AND SNR > 10 in all bands
- These have photometric errors < 10% in all three bands
- Suitable for precise color measurements and temperature estimation

## Data Quality Assessment

### Strengths
1. **Excellent completeness:** 100% of objects have J, H, K magnitudes
2. **High quality:** 46.4% have AAA quality (SNR > 10 in all bands)
3. **Good SNR:** Median SNR of 14-22 across bands is excellent for photometry
4. **Clean sample:** 74% have no contamination issues
5. **Near-infrared colors:** Essential for temperature determination, especially for cool/red stars

### Limitations
1. **K-band errors:** 23% missing K-band error estimates (upper limits)
2. **Faint end:** Some sources have lower SNR in K-band (K is faintest)
3. **Color outliers:** Some extreme colors (>3 mag) likely from confusion or blending
4. **Contamination:** 26% have some level of confusion/contamination flags

### Recommendations for Use

**For temperature estimation (infrared):**
- Use objects with AAA or AAB quality (658,929 objects, 58.7%)
- Combine with optical colors (g-r, r-i from Pan-STARRS)
- J-K color is best for cool stars (T < 5000 K)

**For high-precision work:**
- Use only AAA/AAB/ABA/ABB AND SNR > 10 subset (538,042 objects, 47.9%)
- Filter extreme colors (e.g., -1 < J-H < 2, -0.5 < H-K < 1)
- Check cc_flg = '000' for uncontaminated sources

**For color-color diagrams:**
- 2MASS (J-H vs H-K) + Pan-STARRS (g-r vs r-i) provides complete SED coverage
- Useful for identifying cool giants, hot stars, and reddened objects

## Comparison with Other Catalogs

### Coverage
- **Gaia DR3:** All 1.12M objects have Gaia source_id (perfect cross-match)
- **Pan-STARRS:** ~1.17M objects with griz photometry (slight mismatch due to different sky coverage)
- **2MASS:** 1.12M objects (excellent match with Gaia sample)

### Photometric Depth
- **Pan-STARRS g:** ~21 mag (optical, blue)
- **2MASS J:** ~19.6 mag (near-infrared, short wavelength)
- **2MASS K:** ~17.5 mag (near-infrared, long wavelength)

K-band is shallowest but provides crucial temperature information for cool stars.

## Scientific Applications

### 1. Temperature Estimation
- J-K color is sensitive to temperature for cool stars (3000-5000 K)
- Combine with optical colors for full SED fitting
- Less affected by reddening than optical colors (A_K ≈ 0.1 × A_V)

### 2. Reddening Correction
- Near-infrared colors can be used to estimate extinction
- (J-K)_0 ≈ 0.7 for typical eclipsing binary → derive E(J-K) → derive A_V

### 3. Spectral Type Classification
- J-H vs H-K diagram separates giants from dwarfs
- Useful for identifying contaminating field stars

### 4. Cool Star Identification
- J-K > 1.0: Cool stars (T < 4000 K) or heavily reddened
- Objects missed by Gaia GSP-Phot due to low temperature

## Next Steps

1. **Merge with unified features dataset** to add J-H, H-K, J-K colors
2. **Create 2MASS + optical color plots** to validate temperature predictions
3. **Identify cool star subset** (J-K > 0.9) for improved temperature estimates
4. **Check for reddening** using optical vs near-infrared color discrepancies
5. **Validate with spectroscopy** (APOGEE specializes in near-infrared)

## File Information

- **Format:** Parquet (compressed, efficient)
- **Size:** 69 MB
- **Rows:** 1,123,471
- **Columns:** 17
- **Creation date:** 2025-10-13
- **Source script:** `scripts/query_2mass_irsa.py`
