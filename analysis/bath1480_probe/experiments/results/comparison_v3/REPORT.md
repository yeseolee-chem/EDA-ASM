# Phase 1/2/3/4/5 v3 (oracle-clean + MAD-normalized)

Dataset: bath_1480, 3504 rxns  

## Methodology notes
- HPs (`hps_phase23_v2.pkl`) are **aliased from paper Arm A tuning** (46 Espley features). No per-arm re-tuning against v2 60-feature space; sklearn arms are therefore slightly HP-underfit for their actual feature set. XGB uses fixed defaults + early-stopping-on-val, so any XGB advantage reported here is conservative under this HP handicap.
## Fixes applied
- Removed oracle features `disp_xtb`, `dispd4_xtb`, `eint_total_xtb` from all datasets.
- Fixed strain frag index swap: `strain_1_xtb ↔ strain_2_xtb`.
- NMAE = MAE / meanAD (v9 house standard; label meanAD across cohort, seed-invariant).
- `disp` channel declared **ORACLE** (analytical D4 dispersion) — reported but excluded from best-model ranking.

## Phase definitions
- Phase 1: paper reproduction (Espley features + labels, unchanged)
- Phase 2: xTB-aug, τ=1e-10, our own labels + 6 EDA channels
- Phase 3: xTB-aug, τ=0.05
- Phase 5: xTB-aug, no filter (paper methodology)
- Phase 4: XGB + MACE residual (5-fold CV, different protocol)

## Test MAE per channel × model × phase

### d1

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.313 ± 0.222 | 0.592 |
| Phase 1 (paper) | krr | 2.986 ± 0.122 | 0.533 |
| Phase 1 (paper) | svr | 2.643 ± 0.150 | 0.472 |
| Phase 2 (τ=1e-10) | ridge | 2.575 ± 0.033 | 0.335 |
| Phase 2 (τ=1e-10) | krr | 2.696 ± 0.076 | 0.350 |
| Phase 2 (τ=1e-10) | svr | 2.222 ± 0.108 | 0.289 |
| Phase 2 (τ=1e-10) | xgb | 1.817 ± 0.041 | 0.236 |
| Phase 3 (τ=0.05) | ridge | 2.606 ± 0.014 | 0.339 |
| Phase 3 (τ=0.05) | krr | 2.638 ± 0.085 | 0.343 |
| Phase 3 (τ=0.05) | svr | 2.185 ± 0.107 | 0.284 |
| Phase 3 (τ=0.05) | xgb | 1.806 ± 0.024 | 0.235 |
| Phase 5 (no filter) | ridge | 2.576 ± 0.033 | 0.335 |
| Phase 5 (no filter) | krr | 2.701 ± 0.083 | 0.351 |
| Phase 5 (no filter) | svr | 2.249 ± 0.115 | 0.292 |
| Phase 5 (no filter) | xgb | 1.814 ± 0.023 | 0.236 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.397 ± 0.103 | 0.311 |

### d2

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 1 (paper) | ridge | 3.156 ± 0.131 | 0.611 |
| Phase 1 (paper) | krr | 2.765 ± 0.155 | 0.535 |
| Phase 1 (paper) | svr | 2.370 ± 0.125 | 0.459 |
| Phase 2 (τ=1e-10) | ridge | 2.133 ± 0.088 | 0.321 |
| Phase 2 (τ=1e-10) | krr | 2.460 ± 0.083 | 0.370 |
| Phase 2 (τ=1e-10) | svr | 1.902 ± 0.046 | 0.286 |
| Phase 2 (τ=1e-10) | xgb | 1.437 ± 0.053 | 0.216 |
| Phase 3 (τ=0.05) | ridge | 2.129 ± 0.098 | 0.320 |
| Phase 3 (τ=0.05) | krr | 2.485 ± 0.097 | 0.373 |
| Phase 3 (τ=0.05) | svr | 1.954 ± 0.071 | 0.294 |
| Phase 3 (τ=0.05) | xgb | 1.553 ± 0.067 | 0.233 |
| Phase 5 (no filter) | ridge | 2.135 ± 0.089 | 0.321 |
| Phase 5 (no filter) | krr | 2.481 ± 0.106 | 0.373 |
| Phase 5 (no filter) | svr | 1.954 ± 0.048 | 0.294 |
| Phase 5 (no filter) | xgb | 1.439 ± 0.068 | 0.216 |
| Phase 4 (XGB+MACE) | XGB+MACE | 1.971 ± 0.035 | 0.296 |

### pauli

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 7.463 ± 0.600 | 0.378 |
| Phase 2 (τ=1e-10) | krr | 10.021 ± 0.656 | 0.507 |
| Phase 2 (τ=1e-10) | svr | 5.909 ± 0.222 | 0.299 |
| Phase 2 (τ=1e-10) | xgb | 5.461 ± 0.176 | 0.276 |
| Phase 3 (τ=0.05) | ridge | 8.410 ± 0.467 | 0.425 |
| Phase 3 (τ=0.05) | krr | 9.922 ± 0.715 | 0.502 |
| Phase 3 (τ=0.05) | svr | 6.567 ± 0.292 | 0.332 |
| Phase 3 (τ=0.05) | xgb | 6.074 ± 0.232 | 0.307 |
| Phase 5 (no filter) | ridge | 7.465 ± 0.606 | 0.378 |
| Phase 5 (no filter) | krr | 10.137 ± 0.734 | 0.513 |
| Phase 5 (no filter) | svr | 6.074 ± 0.257 | 0.307 |
| Phase 5 (no filter) | xgb | 5.566 ± 0.174 | 0.282 |
| Phase 4 (XGB+MACE) | XGB+MACE | 10.320 ± 0.555 | 0.522 |

### oi

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 3.996 ± 0.221 | 0.392 |
| Phase 2 (τ=1e-10) | krr | 4.427 ± 0.287 | 0.434 |
| Phase 2 (τ=1e-10) | svr | 3.102 ± 0.128 | 0.304 |
| Phase 2 (τ=1e-10) | xgb | 3.009 ± 0.059 | 0.295 |
| Phase 3 (τ=0.05) | ridge | 4.244 ± 0.140 | 0.416 |
| Phase 3 (τ=0.05) | krr | 4.425 ± 0.256 | 0.434 |
| Phase 3 (τ=0.05) | svr | 3.249 ± 0.172 | 0.319 |
| Phase 3 (τ=0.05) | xgb | 3.197 ± 0.144 | 0.314 |
| Phase 5 (no filter) | ridge | 3.998 ± 0.223 | 0.392 |
| Phase 5 (no filter) | krr | 4.486 ± 0.339 | 0.440 |
| Phase 5 (no filter) | svr | 3.161 ± 0.155 | 0.310 |
| Phase 5 (no filter) | xgb | 3.012 ± 0.078 | 0.296 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.219 ± 0.244 | 0.512 |

### elst

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 4.269 ± 0.216 | 0.423 |
| Phase 2 (τ=1e-10) | krr | 5.173 ± 0.254 | 0.512 |
| Phase 2 (τ=1e-10) | svr | 3.812 ± 0.199 | 0.378 |
| Phase 2 (τ=1e-10) | xgb | 3.763 ± 0.121 | 0.373 |
| Phase 3 (τ=0.05) | ridge | 4.816 ± 0.159 | 0.477 |
| Phase 3 (τ=0.05) | krr | 5.290 ± 0.323 | 0.524 |
| Phase 3 (τ=0.05) | svr | 4.131 ± 0.229 | 0.409 |
| Phase 3 (τ=0.05) | xgb | 3.986 ± 0.106 | 0.395 |
| Phase 5 (no filter) | ridge | 4.269 ± 0.216 | 0.423 |
| Phase 5 (no filter) | krr | 5.219 ± 0.246 | 0.517 |
| Phase 5 (no filter) | svr | 3.853 ± 0.184 | 0.382 |
| Phase 5 (no filter) | xgb | 3.722 ± 0.124 | 0.369 |
| Phase 4 (XGB+MACE) | XGB+MACE | 5.666 ± 0.284 | 0.561 |

### disp ⚠ ORACLE

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 1.348 ± 0.047 | 0.361 |
| Phase 2 (τ=1e-10) | krr | 1.337 ± 0.051 | 0.358 |
| Phase 2 (τ=1e-10) | svr | 1.022 ± 0.057 | 0.274 |
| Phase 2 (τ=1e-10) | xgb | 1.009 ± 0.049 | 0.270 |
| Phase 3 (τ=0.05) | ridge | 1.446 ± 0.050 | 0.387 |
| Phase 3 (τ=0.05) | krr | 1.479 ± 0.051 | 0.396 |
| Phase 3 (τ=0.05) | svr | 1.197 ± 0.060 | 0.321 |
| Phase 3 (τ=0.05) | xgb | 1.116 ± 0.025 | 0.299 |
| Phase 5 (no filter) | ridge | 1.348 ± 0.048 | 0.361 |
| Phase 5 (no filter) | krr | 1.354 ± 0.059 | 0.363 |
| Phase 5 (no filter) | svr | 1.037 ± 0.060 | 0.278 |
| Phase 5 (no filter) | xgb | 1.010 ± 0.045 | 0.271 |
| Phase 4 (XGB+MACE) | XGB+MACE | 1.526 ± 0.087 | 0.409 |

### cpcm

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 2.677 ± 0.155 | 0.538 |
| Phase 2 (τ=1e-10) | krr | 2.784 ± 0.105 | 0.559 |
| Phase 2 (τ=1e-10) | svr | 2.411 ± 0.093 | 0.484 |
| Phase 2 (τ=1e-10) | xgb | 2.391 ± 0.039 | 0.480 |
| Phase 3 (τ=0.05) | ridge | 2.952 ± 0.159 | 0.593 |
| Phase 3 (τ=0.05) | krr | 2.874 ± 0.106 | 0.577 |
| Phase 3 (τ=0.05) | svr | 2.563 ± 0.056 | 0.515 |
| Phase 3 (τ=0.05) | xgb | 2.441 ± 0.047 | 0.490 |
| Phase 5 (no filter) | ridge | 2.677 ± 0.155 | 0.538 |
| Phase 5 (no filter) | krr | 2.792 ± 0.096 | 0.561 |
| Phase 5 (no filter) | svr | 2.418 ± 0.098 | 0.486 |
| Phase 5 (no filter) | xgb | 2.382 ± 0.060 | 0.479 |
| Phase 4 (XGB+MACE) | XGB+MACE | 2.899 ± 0.089 | 0.582 |

### cds

| Phase | Model | MAE | NMAE (÷meanAD) |
|---|---|---:|---:|
| Phase 2 (τ=1e-10) | ridge | 0.409 ± 0.020 | 0.714 |
| Phase 2 (τ=1e-10) | krr | 0.380 ± 0.017 | 0.664 |
| Phase 2 (τ=1e-10) | svr | 0.338 ± 0.019 | 0.590 |
| Phase 2 (τ=1e-10) | xgb | 0.291 ± 0.017 | 0.509 |
| Phase 3 (τ=0.05) | ridge | 0.423 ± 0.009 | 0.738 |
| Phase 3 (τ=0.05) | krr | 0.410 ± 0.016 | 0.715 |
| Phase 3 (τ=0.05) | svr | 0.369 ± 0.019 | 0.644 |
| Phase 3 (τ=0.05) | xgb | 0.309 ± 0.012 | 0.538 |
| Phase 5 (no filter) | ridge | 0.410 ± 0.020 | 0.715 |
| Phase 5 (no filter) | krr | 0.383 ± 0.016 | 0.669 |
| Phase 5 (no filter) | svr | 0.340 ± 0.018 | 0.593 |
| Phase 5 (no filter) | xgb | 0.294 ± 0.014 | 0.514 |
| Phase 4 (XGB+MACE) | XGB+MACE | 0.446 ± 0.013 | 0.779 |

