# spec19_espley_s2_structures — Stage 2 summary (DIPOLAR-400)

Assembles the 5 DIAS structures per reaction (r_A, r_B, ts, d_A, d_B) on the 400 dipolar [3+2] cycloaddition cohort, plus the three bookkeeping artifacts (manifest, mapping, mol_types).

## Environment

- python: 3.10.14
- pandas: 2.3.3

**G2-0 (pandas round-trip):** Stage 1 pickle loaded cleanly in reactot (pandas 2.3.3). The `espley_repro` env (pandas 2.1.1) is not present on this HPC; a round-trip smoke test in that env is DEFERRED and must run before Stage 5 consumes the artifact.

## Fragment split (USER-MANDATED, INHERITED)

Fragment A = ORCA EDA `(1)` atoms → `strain_A_kcal` → Stage-1 dict key `{rn}_1`.
Fragment B = ORCA EDA `(2)` atoms → `strain_B_kcal` → Stage-1 dict key `{rn}_2`.

Never re-derived. `diassep.py` is not used. Any disagreement with external partition methods is a finding to surface, not a defect to fix. See Deviation #7.

## Cohort composition

| sub_source | n | r_A source |
|---|---:|---|
| locked_778 | 192 | R.xyz atom subset (Deviation #8) |
| spec16 | 208 | opt.xyz (isolated frag opt at BLYP-D3BJ/def2-TZVP) |
| **total** | **400** | — |

## Artifacts

- `results/manifest.pkl` (sha256 `c448f5db8f0c1fcf…`)
  Per-row: `dir`, `natoms`, `charge`, `mult`, `ts_idx_A`, `ts_idx_B`, `r_A_provenance`, `r_B_provenance`.
- `results/common_atoms.pkl` — {rn → {r_A_k, r_B_k, ts_k, d_A_k, d_B_k, reacting_A/B_map_ids}}. Reacting-atom counts per the Espley (3,2,5,3,2) contract; index enumeration deferred to Stage 4 where SMILES↔xyz atom order matching happens.
- `results/mapping.pkl` — {rn → {ts_idx_A, ts_idx_B, n_atoms}}.
- `results/mol_types.pkl` — {rn → {A: 'dipole'|'dipolarophile', B: …}}.
- `structures/rxn_XXXX/{r_A,r_B,ts,d_A,d_B}.xyz` — 2000 files, NOT committed (large + easily regenerated). On-HPC location: `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec19_espley_s2_structures/structures/`.

## Gates

**0 FAIL, 1 WARN** — see `logs/gates.log`.

- G2-0: Stage 1 pickle round-trip in reactot; espley_repro test deferred.
- G2-A: 2000 xyz files present, no zero-byte, no NaN coords.
- G2-B: `natoms(ts) == natoms(d_A) + natoms(d_B)`; element multisets of `d_A`↔`r_A` and `d_B`↔`r_B` identical.
- G2-C: `d_A ∪ d_B` coordinates are exact atom subsets of `ts` (tolerance 1e-6 Å).
- G2-D: Fragment-A ↔ dict-key `_1` contract holds by construction; manifest records `r_A/r_B` provenance for every row.
- G2-E: reacting-atom k-shape check vs Espley (3,2,5,3,2). 0 anomalies logged (report-only, no exclusion).
- G2-F: charge conservation + open-shell list (0 rxns with fragment mult ≠ 1).
- G2-G: diassep cross-check informational; 20 rows in `results/diassep_agreement.csv`.

### Gate digest
```
    [G2-A PASS] 2000 files present, no zero-byte, no NaN coords
    [G2-B PASS] atom conservation OK on all 400 reactions
    [G2-C PASS] d_A ∪ d_B == ts atom-set exactly (worst |Δ|=0.00e+00 Å < 1e-06)
    [G2-D PASS] Fragment A ↔ dict key '_1' contract holds by construction; r_A/r_B provenance recorded in manifest for every row.
    [G2-E-distribution INFO] n_processed=400 conform_to_(3,2,5,3,2)_or_(2,3,5,2,3)=400 anomalies=0
    [G2-E PASS] all 400 reactions match (3,2,5,3,2) contract or its A/B swap
    [G2-F PASS] open_shell=0, charge_conserve_fail=0, non_int=0
    [G2-manifest-sha256 INFO] c448f5db8f0c1fcfb399114f8dcd9f0474761fb6f8815330591c65960d1fd5e5
    === SUMMARY: 0 FAIL, 0 WARN ===
```

## Files

```
Ref Comparison/spec19_espley_s2_structures/
  code/{discover_geometry_sources.py, build_structures.py,
        build_common_atoms.py, verify_structures.py,
        diassep_crosscheck.py, aggregate.py, submit_s2.sh}
  data/                                (empty — cohort_notes inherited)
  logs/{discovery.json, build.log, gates.log}
  results/{manifest.pkl, common_atoms.pkl, mapping.pkl, mol_types.pkl,
           common_atom_anomalies.csv, open_shell.csv,
           diassep_agreement.csv, DEVIATIONS.md, summary.md}
  figures/{natoms_hist.png, fragment_size_scatter.png}
  structures/rxn_XXXX/{r_A,r_B,ts,d_A,d_B}.xyz   (NOT committed)
```

