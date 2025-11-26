# Optimizing Effective Temperature Estimation for Eclipsing Binaries using Gaia DR3 Photometry and Ensemble Machine Learning

## Abstract

Eclipsing binaries (EBs) are fundamental benchmarks for stellar physics, yet a significant fraction of the ~2.5 million EBs in Gaia DR3 lack effective temperature ($T_{\rm eff}$) estimates or possess low-confidence parameters. This work presents a comprehensive machine learning framework to predict $T_{\rm eff}$ for approximately 1.2 million EBs using exclusively Gaia DR3 photometry ($G, BP, RP$). We initially explore multi-survey enrichment (SDSS, 2MASS) but prioritize a Gaia-only approach to maximize catalog coverage. To address the significant domain shift between the training set (objects with known $T_{\rm eff}$) and the full prediction set, we implement and compare three distinct modeling strategies: (1) a baseline Random Forest regressor on log-transformed temperatures, (2) a physics-informed model incorporating predicted surface gravity ($\log g$), and (3) a structure-augmented model leveraging unsupervised clustering of the color space. We introduce a "Best-of-Three" ensemble method that dynamically selects the prediction with the lowest internal model uncertainty for each source. Our final catalog achieves a Mean Absolute Error (MAE) of ~X K and provides robust uncertainty estimates, significantly improving coverage and reliability for eclipsing binary research.

## Paper Outline

### 1. Introduction
*   **Context:** Importance of EBs as distance indicators and stellar laboratories.
*   **Problem Statement:** The gap in Gaia DR3 parameters for EBs. Incompleteness and quality issues in existing catalogs.
*   **Objective:** Create a homogeneous, high-coverage $T_{\rm eff}$ catalog using ML, overcoming the limitations of spectroscopic availability.

### 2. Data Selection and Exploration
*   **Source Data:** Gaia DR3 Eclipsing Binary catalog (~2.5M sources).
*   **Photometry:** Focus on Gaia $G, BP, RP$ bands.
*   **Reference Labels:** compilation of "Ground Truth" temperatures (e.g., from spectroscopic surveys like APOGEE, GALAH, LAMOST used for training).
*   **Enrichment Experiments (The "Road Not Taken"):**
    *   Attempts to cross-match with SDSS and 2MASS.
    *   Trade-off analysis: Accuracy vs. Completeness. Justification for sticking to Gaia-only data.

### 3. Methodology: Addressing Domain Shift and Feature Engineering
*   **The Domain Shift Challenge:**
    *   Comparison of Training vs. Prediction distributions (Color-Magnitude Diagrams).
    *   Failed mitigation attempts: Density-based reweighting and filtering by high-quality flags (why they didn't work).
*   **Feature Engineering Evolution:**
    *   Initial feature importance analysis: Dominance of $G$ and $BP-RP$.
    *   The "Overfitting Paradox": Why simple models failed to generalize.
    *   Expansion to full color combinations, colors-from-bands, and auxiliary features.
*   **Target Transformation:**
    *   Use of $\log_{10}(T_{\rm eff})$ to handle the dynamic range (3000K - 30000K) and improve performance on cool stars.

### 4. Advanced Modeling Strategies
*   **Model 1: Baseline Log-Teff:** Standard Random Forest on extended Gaia color features.
*   **Model 2: Physics-Augmented (The "Logg" Path):**
    *   Rationale: Breaking degeneracy between dwarf and giant stars.
    *   Step 1: Predicting $\log g$ and [Fe/H] (and why [Fe/H] was dropped).
    *   Step 2: Using predicted $\log g$ as a feature for $T_{\rm eff}$.
    *   Uncertainty Propagation: Monte Carlo/Analytical propagation of $\log g$ errors.
*   **Model 3: Structure-Augmented (The "Clustering" Path):**
    *   Unsupervised learning (e.g., K-Means/GMM) on color space.
    *   Using Cluster IDs to capture local non-linearities and distinct stellar populations.

### 5. The "Best-of-Three" Ensemble
*   **Internal Uncertainty Estimation:** Using Random Forest variance/quantile spread.
*   **Selection Mechanism:** Dynamic selection of the model with the lowest uncertainty per object.
*   **Rationale:** Different models perform better in different regimes (e.g., Logg model for giants, Clustering model for complex color regions).

### 6. Results and Validation
*   **Model Comparison:** Statistical performance (MAE, RMSE, $R^2$) of the three approaches on the test set.
*   **Ensemble Performance:** Demonstration of uncertainty reduction and outlier mitigation.
*   **Catalog Characteristics:** Distribution of the final predicted temperatures.
*   **Validation against "Gold Standard" subsets:** Performance on high-quality literature EBs.

### 7. Conclusion
*   Summary of the "Best-of-Three" methodology.
*   Impact on the field (availability of the catalog).
*   Future work (e.g., inclusion of time-series features).

### Appendices
*   A. Feature Importance Plots.
*   B. Uncertainty Validation.

