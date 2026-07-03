# V1 Claisen — Hammett & EDA regression summary

OLS fit of each channel against σₚ across the 15 para-substituents.
Slope is the effective Hammett constant ρ (kcal · mol⁻¹ · σ⁻¹).

| channel | ρ (kcal/mol per σ) | R² | Pearson r | p-value |
|---|---|---|---|---|
| ΔE‡ (wB97X-3c) | +1.847 ± 3.205 | 0.025 | +0.158 | 5.74e-01 |
| ΔE_strain | -2.585 ± 0.669 | 0.535 | -0.731 | 1.95e-03 |
| ΔV_elst | +10.888 ± 3.408 | 0.440 | +0.663 | 7.04e-03 |
| ΔE_Pauli | -22.472 ± 7.164 | 0.431 | -0.656 | 7.87e-03 |
| ΔE_oi | +7.433 ± 4.402 | 0.180 | +0.424 | 1.15e-01 |
| ΔE_disp | -0.170 ± 0.428 | 0.012 | -0.109 | 6.98e-01 |

## Swain–Lupton dual-parameter fit of ΔE‡
ΔE‡ ≈ -16.177 · F  +9.099 · R  + +37.059
R² = 0.481

## Correlation matrix
See `figures/channel_correlation.png`. Full matrix stored in
`results/correlation_matrix.csv`.