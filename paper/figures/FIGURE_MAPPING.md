# Figure Mapping for Paper

## Section 2: Data Selection & Exploration
*   **Fig 1: Distribution Mismatch.** Visualizing the domain shift between training (Gaia with Teff) and prediction (Gaia without Teff) sets.
    *   *Source:* `reports/figures/distribution_matching/all_colors_comparison.png`
    *   *Alternative:* `reports/figures/distribution_matching/bp_rp_distribution_comparison.png`

## Section 3: Methodology
*   **Fig 2: Uncertainty Propagation.** Demonstrating how `log g` uncertainty affects `Teff` prediction.
    *   *Source:* `reports/figures/teff_uncertainty_analysis/gradient_contribution_analysis.png`
    *   *Source:* `reports/figures/teff_uncertainty_analysis/teff_unc_vs_logg_unc.png`

## Section 4: Model Comparison
*   **Fig 3: The Three Models.** Pairwise comparison of predictions from Baseline, Logg-Augmented, and Cluster-Augmented models.
    *   *Source:* `reports/figures/three_model_comparison/temperature_comparison_scatter.png`
    *   *Source:* `reports/figures/three_model_comparison/summary_statistics.png`

## Section 5: Ensemble Strategy
*   **Fig 4: Uncertainty Selection.** How the "Best-of-Three" strategy picks the lowest uncertainty.
    *   *Source:* `reports/figures/best_uncertainty_ensemble/best_uncertainty_comparison.png`
*   **Fig 5: Improvement Analysis.** Quantifying the gain in high-confidence predictions.
    *   *Source:* `reports/figures/best_uncertainty_ensemble/improvement_analysis.png`

## Section 6: Validation & Results
*   **Fig 6: Final Catalog Performance.** Accuracy of the final ensemble against the test set.
    *   *Source:* `reports/figures/validation/test_scatter.png` (or specific best model validation plot)
    *   *Source:* `reports/figures/combined_validation/rf_combined_colors_*_residuals.png`
*   **Fig 7: Catalog Distribution.** The final temperature distribution of the 2.1M EBs.
    *   *Source:* `reports/figures/three_model_comparison/temperature_distributions.png`

