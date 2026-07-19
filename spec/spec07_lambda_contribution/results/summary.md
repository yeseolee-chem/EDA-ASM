# SPEC_07 — λ-contribution sweep — summary

- Cohort (common across all λ): 783 rxns
- λ grid: ['0.00', '0.25', '0.50', '0.75', '1.00']
- Bootstrap: B=1000, seed=42, reaction-level resampling
- Model: xgb-28d base (b) + ModelM1Delta residual (δ), retrained per λ
- Blend: ŷ = (1−λ)·δ + λ·b   (λ=1 = base-only, δ untrained)

## Per-channel + barrier NMAE vs λ (95% CI in brackets)

| channel | λ=0.00 | λ=0.25 | λ=0.50 | λ=0.75 | λ=1.00 |
|---|---|---|---|---|---|
| strain | 0.395 [0.363, 0.428] | 0.338 [0.311, 0.367] | 0.286 [0.263, 0.312] | 0.288 [0.265, 0.314] | 0.227 [0.207, 0.248] |
| Pauli | 0.443 [0.398, 0.490] | 0.351 [0.314, 0.392] | 0.276 [0.247, 0.310] | 0.212 [0.190, 0.239] | 0.205 [0.185, 0.228] |
| elst | 0.463 [0.418, 0.510] | 0.370 [0.334, 0.408] | 0.305 [0.278, 0.337] | 0.272 [0.250, 0.297] | 0.265 [0.243, 0.292] |
| oi | 0.436 [0.394, 0.479] | 0.351 [0.316, 0.387] | 0.275 [0.249, 0.303] | 0.198 [0.179, 0.219] | 0.149 [0.134, 0.167] |
| disp | 0.195 [0.176, 0.216] | 0.155 [0.140, 0.171] | 0.146 [0.132, 0.161] | 0.232 [0.210, 0.255] | 0.150 [0.135, 0.165] |
| barrier | 0.633 [0.591, 0.677] | 0.512 [0.476, 0.549] | 0.415 [0.386, 0.445] | 0.379 [0.353, 0.408] | 0.299 [0.276, 0.322] |

## λ* (argmin NMAE) per channel

| channel | λ* | NMAE(λ*) | 95% CI |
|---|---|---|---|
| strain | 1.00 | 0.227 | [0.207, 0.248] |
| Pauli | 1.00 | 0.205 | [0.185, 0.228] |
| elst | 1.00 | 0.265 | [0.243, 0.292] |
| oi | 1.00 | 0.149 | [0.134, 0.167] |
| disp | 0.50 | 0.146 | [0.132, 0.161] |
| barrier | 1.00 | 0.299 | [0.276, 0.322] |

## Cell-count check per λ

| λ | pooled rxns | early-stopped folds | did NOT early-stop |
|---|---|---|---|
| 0.00 | 783 | 25 | 0 |
| 0.25 | 783 | 25 | 0 |
| 0.50 | 783 | 25 | 0 |
| 0.75 | 783 | 25 | 0 |
| 1.00 | 783 | 25 | 0 |

## Files
- pooled_oof.parquet, lambda_curve.csv, lambda_star.json
- figures/lambda_nmae.png, figures/lambda_rmse.png,
  figures/parity_at_lamstar.png