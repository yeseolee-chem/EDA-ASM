# spec11_electronic_33d

Two-arm validation of a 33-descriptor set (spec06 28-d + five electronic
primitives d29..d33) on the v9 783-reaction cohort. Method identical to
spec06 (xgb_28d vs xgb28 + delta), just upgraded to 33-d.

## Descriptors added

| new | channel | definition | inputs |
|---|---|---|---|
| d29 | elst | anisotropic elst (q-d + d-d, monopole-monopole omitted) | isolated fragment dipoles, formal charges, TS geometry |
| d30 | oi   | inter-fragment charge transfer = d21 - Q_A_formal        | Mulliken on complex (d21) + charges table |
| d31 | Pauli| sum S_uv^2 over inter-fragment AO block                   | complex overlap matrix |
| d32 | oi   | sum |H0_uv| over inter-fragment AO block (Hartree->kcal) | complex core-Hamiltonian matrix |
| d33 | strain| sum ||g_X||_F over frozen-TS isolated fragments (Ha/Bohr)| gradients of isolated fragment SPs |

X33 = X28 (d1..d28) + d29 + d30 + d31 + d32 + d33.

## Layout

```
spec/spec11_electronic_33d/
  README.md
  code/
    xtb_extract.py         # extended tblite driver (does not touch stage3)
    compute_d29_d33.py     # Stage 1 (2 xTB passes per rxn)
    merge_d29_d33.py       # shard -> canonical parquet
    descriptors33.py       # build_X33(rids, X24)
    train_xgb33_delta.py   # arm-1 + arm-2 in one JSON per fold
    aggregate_2arm.py      # NMAE/RMSE/CIs + parity + summary.md
    test_multipole_units.py  # Gate-A
    test_ao_blocks.py        # Gate-B (also tblite version probe)
    submit_gates.sh        # cpu2, 48h
    submit_compute.sh      # cpu2/3/4, 8-shard array %8, 48h
    submit_merge.sh        # cpu2, 48h
    submit_train.sh        # gpu3/4/5 array 0-4 %3, 48h
    submit_aggregate.sh    # cpu2, 48h
  data/descriptors_d29_d33.parquet
  splits/outer_folds.json  # copy of spec06's (783 cohort verified identical)
  oof/xgb33_delta/fold{0..4}/member0.json
  results/{summary.md, leaderboard.csv, head_to_head.csv, metrics.csv,
           pooled_oof.parquet, xgb_33d_oof.parquet}
  figures/{nmae_bars.png, rmse_bars.png, parity_grid.png}
```

## Execution order

```
# 1) Gate-A + Gate-B (also tblite version probe)
sbatch spec/spec11_electronic_33d/code/submit_gates.sh

# 2) Stage 1: compute d29..d33 (dependent on gates passing)
sbatch --dependency=afterok:<gates_jid> spec/spec11_electronic_33d/code/submit_compute.sh

# 3) Merge shards
sbatch --dependency=afterok:<compute_jid> spec/spec11_electronic_33d/code/submit_merge.sh

# 4) Train 2-arm learner (5 folds, spread gpu3/4/5)
sbatch --dependency=afterok:<merge_jid> spec/spec11_electronic_33d/code/submit_train.sh

# 5) Aggregate -> summary.md + figures
sbatch --dependency=afterany:<train_jid> spec/spec11_electronic_33d/code/submit_aggregate.sh
```

Everything is idempotent; if the 48h wall clips a shard/fold, just resubmit.

## Notes on unit tests

- Gate-A: synthetic +/- charge dimer; d29 formula must agree with an
  explicit Coulomb sum (mono-mono subtracted) to <1 %, same sign, at
  R >> l. Failure => the T_qd sign convention or dipole recentring is
  wrong.
- Gate-B: complex SP on first cohort reaction; checks S is 1 on the
  diagonal, symmetric, and that the AO map from `build_ao_atom_map(Z)`
  matches n_orb. Also probes tblite for `overlap-matrix` /
  `hamiltonian-matrix` keys; some older builds lack them.
