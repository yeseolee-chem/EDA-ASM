# SPEC_05 - d25 + soft sum-consistency (XGB) on 776 v7

- Cohort: 776 rxns (v7 m3 bundle)
- d25 SCF ok: 762/776
- pearson(d25, y_strain) = +0.029  (expect > 0)

- Tuned M2 (lambda, eps): (0.3, 0.5)
- Tuned M3 (lambda, eps): (0.3, 0.5)

## 2x2 grid NMAE (5-fold pooled)

| channel | M0 (24-d, per-ch) | M1 (25-d, per-ch) | M2 (24-d, +sum) | M3 (25-d, +sum) |
|---|---|---|---|---|
| strain | 0.598 | 0.533 | 0.599 | 0.534 |
| Pauli | 0.426 | 0.427 | 0.424 | 0.424 |
| elst | 0.408 | 0.413 | 0.408 | 0.412 |
| oi | 0.405 | 0.393 | 0.405 | 0.393 |
| disp | 0.183 | 0.175 | 0.183 | 0.175 |
| barrier | 0.557 | 0.549 | 0.536 | 0.526 |

## Ablation deltas (see ablation_deltas.csv)
- d25 effect: (M1 - M0) and (M3 - M2)
- sum effect: (M2 - M0) and (M3 - M1)
- interaction: (M3 - M1 - M2 + M0)

## Notes
- Sum-consistency for XGB is post-hoc reconciliation (not training-time term).
- Channel-protection gate: any (lambda, eps) that inflates a channel by
  more than +0.02 NMAE is rejected during inner CV tuning.
- All energies in kcal/mol; xTB SP-only (no relaxation) so SCF failures are rare.