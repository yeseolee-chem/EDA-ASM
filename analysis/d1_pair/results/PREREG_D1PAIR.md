# PREREG — D1PAIR rev 9 (written before any g training)

Spec `docs/specs/D1PAIR_REV9.md`; constants `analysis/d1_pair/d1pair_common.py`; machine-readable record `prereg_d1pair.json` (inputs sha256, pair-list sha256, S_folds sha256, grids). Date 2026-10-19.

## Data

- P_all: 7253
- P_feat: 6953
- P_main: 6885
- P_sens: 6953
- A_feat: 1111
- A_main: 1085
- T_main: 7970

- f0 representative per target: {'barrier': 'KRR', 'c_ghost': 'SVR', 'cds': 'KRR', 'cpcm': 'SVR', 'd1': 'KRR', 'd2': 'KRR', 'disp': 'KRR', 'elst': 'SVR', 'oi': 'SVR', 'pauli': 'SVR'}

## Arms

- **null**: 0
- **A**: f0(x_a) - f0(x_b) (f0 representative)
- **B**: f0 diff + g, g on dd - f0 diff, T_main
- **C**: g on dd, T_main
- **P**: KRR f_pool on D0 4,963 + training-side D1 reactions, f_pool(x_a) - f_pool(x_b)
- **B_ref**: B with g = KRR on training-side P_main only

## Units

- T1/LODO/LOSO: arms B, C x 6 models x 10 targets x outer folds
- T2: arm B x 6 models x 10 targets x 5 folds
- P, B_ref: KRR x 10 targets x T1 5 folds

## Rules

- scaling: z = [dx / s_delta, (xbar - mu) / s] on the (inner) training pairs, no mean removed from dx or y; y / s_y; columns with s < 1e-12 dropped and recorded
- antisymmetry: Ridge on dx only, no intercept; KRR / SVR antisymmetric precomputed kernel [k(dx_p, dx_q) - k(dx_p, -dx_q)] k(xbar_p, xbar_q); SVR / XGB trained both ways, predict 1/2 [s(dx) - s(-dx)]; NN g = h([dx, xbar]) - h([-dx, xbar]) with shared h (r6_nn._layers); gate M-1 |g(dx) + g(-dx)| <= 1e-6 and |g(0)| <= 1e-6 kcal/mol on 200 random test pairs
- tuning: GroupKFold(4), group = core, on the training-side T_main pairs; NN: Hyperband on inner fold 1, CV MAE = mean over the 4 inner folds refitted with the best hps (500 epochs)
- representative model: per split x arm x target: lowest mean (over outer folds) inner-CV MAE among the 6 models; ties by model order; never by test score
- metrics: test pairs of all outer folds pooled; primary = P_main reference pairs; anchor test MAE separately

## Stage criteria (arm B representative, T1 test pairs)

- stage 0: MAE(B) < MAE(null) and MAE(A) - MAE(B) 95 % CI low > 0 and Holm p < 0.05 (core cluster bootstrap 2,000, seed 20261019)
- stage 1: median R2_fam of the test para families >= 0.5
- stage 2: oi and cpcm para-pair MAE <= 0.13, bend-call sensitivity >= 0.7, false positive rate <= 0.1

## Tests

- primary (confirmatory, 9): B representative vs A per target in d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier; metric = test-pair MAE difference A - B; 95 % CI and two-sided p by core cluster bootstrap (2,000, seed 20261019, the cores of the T1 test pairs (44 after the user decision) resampled with replacement); Holm over the 9; folds where B is lower counted
- secondary (descriptive, uncorrected): B vs C, B vs null, B vs P, B(KRR) vs B_ref(KRR), every model pair, sum-of-9 barrier vs direct, LODO / LOSO, P_sens, without D0-overlap reactions, T2 vs T1 (hidden V 6, hidden bend 30; Clopper-Pearson 95 % CI for sensitivities)

## T2 (model that never saw a V)

- excluded families: ['azide/A/R2//N-//', 'azide/AppZ/R1////', 'carbonyl ylide/Bp/R1/t//endo/', 'nitrile oxide/B/R1///endo/', 'nitrile oxide/B/R1///exo/', 'nitrone/A/R2//C//', 'nitrone/A/R2//O-//', 'nitrone/B/R1/c//exo/cis', 'nitrone/B/R2//O-/exo/', 'nitrone/C/R2//O-//', 'nitrone/Cp/R1/t///']
- hidden V: ['carbonyl ylide/Bp/R1/t//endo/', 'nitrone/A/R2//C//', 'nitrone/A/R2//O-//', 'nitrone/B/R1/c//exo/cis', 'nitrone/C/R2//O-//', 'nitrone/Cp/R1/t///']
- hidden bend: 30 families

## User decisions recorded before training

- drop_v2_mismatch (user 2026-10-19)
- Si added to fragmenter.COV (1.11 A) / Z (14), rev5_common.BONDI (2.10 A) / GFN2_VALENCE (4) (user 2026-10-19)
- accept: last-printed-digit xtb energy differences (max non-B1 2.5e-7 in scan_curv_Eint) (user 2026-10-19)
- continue with 44 evaluation cores; azomethine ylide/A and carbonyl ylide/A lost all reference pairs (user 2026-10-19)
- appendix E rerun on feature rows with P_feat as pairs file; reference-less families become anchor families (user 2026-10-19)
- D1_JOBS extracted from the D1 archives (2026-10-19)
- G1 rows whose ORCA died at MPI start-up rerun with the unchanged first attempt (results/G_infra_retry.csv)

Any later change to arms, models, pairs, metrics or criteria is reported as a deviation in SUMMARY.md.
