# MODEL CARD — GNN3

| | |
|---|---|
| model | GNN encoder (shared) + antisymmetric pair head; rung `lf3`, variant A+G (29-wide atom tokens + 182 global xTB features) |
| predicts | ΔΔ = value(a) − value(H reference b) of strain, inter, solv and the barrier, kcal/mol, B3LYP-D3(BJ)/def2-TZVP SMD(water) electronic energies |
| training data | D1 dipolar cycloadditions (5 dipole classes), 7,388 usable pairs in the 5 cross-validation folds (each fold's networks saw the other 4 folds' training pairs); labels `F_pairs_cr2.csv` |
| inputs | GFN2-xTB only: g2 TS (autodE + xTB) → finalize → 182 global features + atom tokens; no DFT |
| networks | 25 cross-validation networks (5 folds × 5 seeds); cells chosen per fold: 4×64 (fold 0), 6×64 (folds 1–4). Final model rule → 6 layers × 64, 49 epochs (**not trained yet**) |
| source of the numbers | `analysis/ch3m/results/EM_metrics.csv`, `EM_rank.csv`, `EM_identity.csv`, `EM_ad.json` (CH3M), `analysis/abs3/results/EA_cause.csv` (ABS3), all GNN3 rows; recomputed copies in `results/eval/` (`metrics.csv`, `rank.csv`, `cause.csv`, `identity.csv`, `applicability.json`; equal to 4 decimals, E-4); per position / skeleton R²: `results/eval/breakdown.csv` (`scripts/evaluate.py --published`) |

All numbers are cross-validation test numbers: every pair was predicted by the 5 networks of the fold whose training
never saw its core (dipole × skeleton). 7,279 test pairs.

## Performance

### Pair MAE (kcal/mol)

| set | n | strain | inter | solv | barrier |
|---|---|---|---|---|---|
| **P_main para** (44 cores; 95 % core-bootstrap CI) | 1,658 | 0.359 [0.290, 0.430] | 0.368 [0.292, 0.455] | 0.298 [0.230, 0.376] | 0.318 [0.244, 0.392] |
| P_main all positions | 6,403 | 0.808 | 0.621 | 0.501 | 0.724 |
| all test pairs (incl. anchors) | 7,279 | 0.824 | 0.617 | 0.491 | 0.751 |

### Sign accuracy (pairs with |ΔΔ| ≥ 1 kcal/mol; n scored in parentheses)

| set | strain | inter | solv | barrier |
|---|---|---|---|---|
| P_main para | 0.886 (237) | 0.876 (234) | 0.804 (153) | 0.927 (331) |
| P_main all | 0.909 (2,279) | 0.935 (2,439) | 0.845 (1,430) | 0.889 (2,375) |

### Ranking inside a family (barrier; P_sens_ref; H added as 0; families with ≥ 8 points)

| scope | families / points | pairwise order accuracy, \|Δy\| 0.41–1 / 1–2 / > 2 kcal/mol | true lowest ranked first (random) | in top 3 (random) | Spearman median / 25th pct |
|---|---|---|---|---|---|
| para | 95 / 1,759 | 0.885 / 0.953 / 0.914 | 0.663 (0.056) | 0.832 (0.168) | 0.916 / 0.621 |
| all positions | 377 / 6,677 | 0.825 / 0.853 / 0.909 | 0.549 (0.059) | 0.767 (0.178) | 0.818 / 0.586 |

### Cause agreement (which channel drives the barrier change; |Δbarrier| ≥ 1; P_sens_ref; model's own sign)

| level | n | dominant channel (baseline = most common true class) | type S / I / SI (baseline) |
|---|---|---|---|
| single pairs | 2,405 | 0.727 (0.631) | 0.710 (0.448) |
| **cell means** (dipole × configuration × series × position × substituent, ≥ 4 pairs) | 219 | **0.858** (0.708) | **0.863** (0.530) |

Cell-mean fit over 667 cells (r / MAE): strain 0.945 / 0.462, inter 0.954 / 0.258, solv 0.891 / 0.222,
barrier 0.952 / 0.411.

Identity: |ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)| mean 0.006, 95th pct 0.024, max 0.091 kcal/mol (the labels: max
0.096).

### By position (P_main; MAE kcal/mol / R²)

| position | n | strain | inter | solv | barrier |
|---|---|---|---|---|---|
| para (p) | 1,658 | 0.359 / 0.70 | 0.368 / 0.37 | 0.298 / 0.32 | 0.318 / 0.65 |
| meta (m) | 1,695 | 0.312 / 0.18 | 0.413 / 0.30 | 0.304 / 0.05 | 0.283 / 0.26 |
| ortho (o) | 1,557 | 0.912 / 0.46 | 0.800 / 0.37 | 0.614 / 0.30 | 0.888 / 0.48 |
| direct | 1,493 | 1.759 / 0.75 | 0.953 / 0.66 | 0.834 / 0.58 | 1.505 / 0.79 |

R² = 1 − SSE / SST inside the group. Meta strain and barrier effects are small (SD of the true ΔΔ 0.68 / 0.67
kcal/mol against 1.13 / 1.02 for para; solv 0.55 against 0.69), so their R² is low although the MAE is small. For
inter the meta and para spreads are about the same (SD 0.85 / 0.83), and R² is low at both positions (0.30 / 0.37).

### By dipolarophile skeleton (P_main; MAE / R²; D, Dp = cyclooctyne family; other codes as in the label file)

| skeleton | n | strain | inter | solv | barrier |
|---|---|---|---|---|---|
| A | 630 | 0.746 / 0.66 | 0.687 / 0.57 | 0.569 / 0.42 | 0.704 / 0.64 |
| Ap | 648 | 0.655 / 0.83 | 0.474 / 0.76 | 0.372 / 0.58 | 0.575 / 0.83 |
| AppE | 440 | 0.616 / 0.88 | 0.428 / 0.83 | 0.336 / 0.64 | 0.561 / 0.88 |
| AppZ | 455 | 0.564 / 0.86 | 0.417 / 0.81 | 0.324 / 0.59 | 0.483 / 0.86 |
| B | 1,844 | 0.913 / 0.73 | 0.654 / 0.58 | 0.553 / 0.45 | 0.843 / 0.73 |
| Bp | 559 | 1.154 / 0.58 | 0.630 / 0.70 | 0.505 / 0.63 | 0.974 / 0.68 |
| C | 672 | 0.638 / 0.67 | 0.627 / 0.60 | 0.559 / 0.46 | 0.558 / 0.68 |
| Cp | 320 | 0.561 / 0.85 | 0.468 / 0.72 | 0.396 / 0.38 | 0.549 / 0.81 |
| **D** (cyclooctyne) | 458 | 0.918 / **−0.01** | 0.872 / 0.30 | 0.706 / 0.15 | 0.837 / **0.03** |
| Dp | 377 | 1.038 / 0.57 | 0.879 / 0.44 | 0.496 / 0.50 | 0.851 / 0.60 |

On all test pairs (incl. anchors, n 541) skeleton D has barrier R² 0.13 and strain R² 0.12.

## Intended use

- Substituent effects **inside one reaction family**: how replacing H by X at a given position changes the barrier, and
  whether the change comes from strain, interaction or solvation (3-channel ΔΔ).
- Ranking substituents of one family by barrier (para: true best substituent ranked first in 66 % of families, in the
  top 3 in 83 %).
- Screening with xTB-level cost; flag `out_of_range` when the seed spread exceeds the cross-validation 95th percentile
  (strain 0.986, inter 0.589, solv 0.507, barrier 0.902 kcal/mol). Today the existing input code builds inputs for D1
  reactions only; new SMILES need a change of that code first (CODE_MAP.md P-1).

## Limits — do not use it for

- **Comparing reactions of different skeletons or families.** Training pairs never cross families; the model has no
  absolute scale.
- **Cyclooctyne (skeleton D)**: barrier R² 0.03 on P_main (0.13 on all test pairs); predictions are close to a guess.
- **Individual ortho and meta values**: ortho MAE 0.80–0.91 kcal/mol for strain, inter and barrier and 0.61 for
  solv (2–3 × para in every channel), meta R² 0.05–0.30. Use para and
  direct with more confidence; ortho / meta only in aggregate.
- **Experimental rates or selectivities**: accuracy is measured against DFT labels, not experiment.
- **Other conditions**: electronic energies in SMD(water) only — no free energies, other solvents or temperatures.
- **Reactions outside the D1 domain**: elements other than H C N O F Si S Cl Br, charged reactants, dipoles other than
  azide, azomethine ylide, carbonyl ylide, nitrile oxide, nitrone, TSs with more than 63 atoms. The existing input code
  also accepts only D1 reactions after the TS search (CODE_MAP.md P-1).
- **Reactions whose g2 TS search fails**: 7.0 % of D1 reactions (650 / 9,266) got no TS, 8.2 % (764 / 9,266) no usable input; such
  pairs cannot be predicted, and the H reference must come from the same g2 procedure.

## Training data and procedure

- Reactions: 9,266 D1 targets → 8,616 g2 TSs → 8,504 with features → 8,502 usable (token + finite features).
- Pairs: label file 8,389 (7,253 H-reference, 1,136 anchor) → 7,460 with both reactions usable → 7,388 in the T1
  folds → 7,279 test pairs (anchor-only cores are always in training). P_main reference pairs also pass the label
  hygiene (d1, d2 ≥ 0, d2 ≤ 50 kcal/mol).
- 5-fold cross-validation over cores (dipole × skeleton); per fold: 4-cell grid tuned with early stopping on an inner
  core split, then 5 seeds for the chosen cell; 5.0 GPU-h on an RTX 3090 in total.
- Loss: standardised L1 of the 4 outputs + 0.1 × identity penalty |barrier − Σ channels| / s_barrier.

## Ethical / safety notes

None specific: a research model of reaction energetics. Predictions are not a substitute for a DFT check of a
candidate reaction.
