# SPEC_01 — ridge α optimization for m3 (787-rxn cohort)

- descriptor width: **24** (asserted)
- cond(XᵀX) mean: 6.267e+15
- rank(X) per fold: [25, 25, 25, 25, 25]

## α*_A (CV) / α*_B (GCV) per channel

- **strain**: α*_A = 1.000e+02, α*_B (GCV) = 4.642e+00
- **Pauli**: α*_A = 1.468e+00, α*_B (GCV) = 1.468e+00
- **V_elst**: α*_A = 1.468e+00, α*_B (GCV) = 1.468e+00
- **oi**: α*_A = 6.813e-01, α*_B (GCV) = 2.154e+00
- **disp**: α*_A = 4.642e+00, α*_B (GCV) = 4.642e-01
- **barrier**: α*_A = 1.468e+01, α*_B (GCV) = 6.813e+00

## Test NMAE at α ∈ {≈0, 1, α*_A} (mean ± std over 5 folds)

| channel | α ≈ 0 | α = 1 | α = α* |
|---|---|---|---|
| strain | 0.836 ± 0.030 | 0.835 ± 0.032 | 0.826 ± 0.044 |
| Pauli | 0.878 ± 0.048 | 0.879 ± 0.046 | 0.882 ± 0.045 |
| V_elst | 0.908 ± 0.050 | 0.909 ± 0.049 | 0.911 ± 0.048 |
| oi | 0.902 ± 0.097 | 0.904 ± 0.096 | 0.905 ± 0.098 |
| disp | 0.286 ± 0.015 | 0.286 ± 0.015 | 0.286 ± 0.014 |
| barrier | 0.539 ± 0.031 | 0.538 ± 0.031 | 0.535 ± 0.034 |

## Notes
- Intercept NOT penalised.
- σ_c-normalised loss + system-level α tuning is out of scope (SPEC_02).