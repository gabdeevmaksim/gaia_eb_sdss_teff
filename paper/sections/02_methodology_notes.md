# Methodology Notes

## 1. The Road Not Taken: Failed Experiments

Before arriving at our final ensemble strategy, several standard approaches to handling domain shift and data quality were tested but ultimately discarded. These negative results provides important context for our final methodological choices.

### 1.1 Distribution Matching via Reweighting
*   **Hypothesis:** The training set (objects with known $T_{\rm eff}$) has a different color distribution than the full prediction set. Correcting this mismatch should improve generalization.
*   **Experiment:** We implemented a Density Ratio Estimation method (using Kernel Density Estimation) to calculate importance weights: $w(x) = P_{predict}(x) / P_{train}(x)$.
*   **Implementation:** `scripts/match_training_to_prediction_distribution.py`
    *   Features matched: `g_r`, `r_i`, `i_z` (Pan-STARRS) and `bp_rp` (Gaia).
    *   Strategies: Reweighting the loss function and Resampling the training set.
*   **Outcome:** While the reweighted distributions matched visually (confirmed via KS tests), the resulting models did not show significant improvement in validation accuracy on the held-out test set. The "effective sample size" was reduced drastically, leading to higher variance in predictions without a commensurate drop in bias.

### 1.2 High-Quality Filtering ("The Gold Standard Trap")
*   **Hypothesis:** Training only on the highest quality data (Gaia quality flag = 1) would produce a more robust model, even if the training size is smaller.
*   **Experiment:** We trained a Random Forest regressor exclusively on sources with optimal data quality flags, explicitly excluding magnitude features to avoid distance bias.
*   **Implementation:** `scripts/train_high_quality_model.py`
*   **Outcome:** The model overfitted to the "clean" subset and failed to generalize to the noisier, fainter objects that make up the bulk of the catalog. The strict filtering removed critical examples of "edge case" objects, making the model blind to common photometric anomalies.

## 2. Auxiliary Task Learning: The $\log g$ Strategy

To overcome the degeneracy between dwarf and giant stars—which often have similar colors but vastly different effective temperatures—we adopted a multi-stage prediction pipeline.

### 2.1 Step 1: Surface Gravity Prediction
*   **Method:** We trained a dedicated Random Forest regressor to predict $\log g$ using Gaia colors.
*   **Quality Control:** Unlike standard regression, we computed the internal uncertainty (standard deviation of tree predictions) for every object.
*   **Filtering:** We implemented a dynamic filter `unc < 0.2 dex` (moderate threshold) to retain only high-confidence $\log g$ predictions.
*   **Implementation:** `scripts/predict_logg_then_teff_with_quality_filter.py`

### 2.2 Step 2: Uncertainty Propagation
*   **Integration:** The predicted $\log g$ was added as an input feature to the final $T_{\rm eff}$ model.
*   **Propagation:** Crucially, we did not just treat the predicted $\log g$ as a ground truth. We propagated the prediction uncertainty into the final $T_{\rm eff}$ estimate, ensuring that downstream predictions reflected the confidence of the upstream auxiliary task.

## 3. Target Transformation: The Log-Space Advantage

*   **Problem:** Stellar effective temperatures span an order of magnitude (3,000 K to >30,000 K). Minimizing Mean Squared Error (MSE) in linear space disproportionately penalizes errors on hot stars while neglecting relative errors on cool stars.
*   **Solution:** We trained models to predict $\log_{10}(T_{\rm eff})$.
*   **Benefit:** This effectively optimizes for Mean Absolute Percentage Error (MAPE), providing balanced performance across the full H-R diagram.
*   **Reference:** `models/rf_gaia_all_colors_teff_log_*.json`

## 4. The Final Ensemble Strategy

Instead of choosing one "best" model, we recognized that different models excel in different regimes. Our final "Best-of-Three" ensemble leverages:
1.  **Baseline Model:** Robust for average main-sequence stars.
2.  **Logg-Augmented Model:** Superior for separating giants/dwarfs.
3.  **Cluster-Augmented Model:** Captures local non-linearities in complex color spaces.

Selection is performed per-object based on the lowest predicted uncertainty.

