# spec20_locked778_fragment_relax — Stage 3 G20-0 summary

Blocking protocol discovery for the proposed dipolar-192 relaxation. Audits the exact ORCA input files that produced `strain_A_kcal` / `strain_B_kcal` on all 400 rows.

Env: python 3.10.14, pandas 2.3.3.

## Per-half protocol profile

### E(TS_A) — fragment A at TS geometry

| half | n | route (first seen) | CP | Opt | solvent |
|---|---:|---|---|---|---|
| locked_778 | 192 | `BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3` | True=192 | False=192 | none=192 |
| spec16 | 208 | `BLYP D3BJ def2-TZVP NoSym TightSCF` | True=208 | False=208 | none=208 |

### E(TS_B) — fragment B at TS geometry

| half | n | route (first seen) | CP | Opt | solvent |
|---|---:|---|---|---|---|
| locked_778 | 192 | `BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3` | True=192 | False=192 | none=192 |
| spec16 | 208 | `BLYP D3BJ def2-TZVP NoSym TightSCF` | True=208 | False=208 | none=208 |

### E(R_A) — fragment A 'relaxed' reference

| half | n | route (first seen) | CP | Opt | solvent |
|---|---:|---|---|---|---|
| locked_778 | 192 | `BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3` | True=192 | False=192 | none=192 |
| spec16 | 208 | `BLYP D3BJ def2-TZVP NoSym Opt TightSCF SlowConv KDIIS` | False=208 | True=208 | none=208 |

### E(R_B) — fragment B 'relaxed' reference

| half | n | route (first seen) | CP | Opt | solvent |
|---|---:|---|---|---|---|
| locked_778 | 192 | `BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3` | True=192 | False=192 | none=192 |
| spec16 | 208 | `BLYP D3BJ def2-TZVP NoSym Opt TightSCF SlowConv KDIIS` | False=208 | True=208 | none=208 |

## G20-0 outcome

**G20-0 HALT** — see below.

Per §7 open item 1 the pre-registered default action on any cross-half or within-half divergence beyond relaxed-fragment geometry is **halt and report**, rather than fix one axis while leaving another. The following divergences were found:

### Cross-half
- R_A.cp: locked_778=['True'] spec16=['False']
- R_A.opt: locked_778=['False'] spec16=['True']
- R_B.cp: locked_778=['True'] spec16=['False']
- R_B.opt: locked_778=['False'] spec16=['True']

### Within spec16
- spec16 internal CP inconsistency: TS_A.cp=['True'] vs R_A.cp=['False']
- spec16 internal CP inconsistency: TS_B.cp=['True'] vs R_B.cp=['False']

## Interpretation and next-step decision

The finding is more nuanced than the spec §7 anticipated:

- **Both halves apply CP correction to E(TS_A) / E(TS_B).** The ORCA EDA-NOCV recipe produces per-fragment single points at the TS geometry using the paired fragment's basis as ghost atoms (visible as `:(1)` / `:(2)` tags in `eda_frag{1,2}.inp`).
- **The two halves treat E(R_A) / E(R_B) differently.** locked_778 uses `v9_review/strain_sp_cp/{rid}/frag{A,B}_R.inp` — a CP-corrected single point at the reactant-complex-subset geometry. spec16 uses `spec16_orca_strain/inputs/{rid}__f{A,B}/opt.inp` — an isolated fragment optimization with NO ghost basis.
- **This means spec16 is internally inconsistent** on the CP axis: TS-side energies are CP-corrected, R-side energies are not. locked_778 is internally consistent (CP on both sides).

### What spec20 proposed vs. what the data requires

Spec20 §5 proposes moving locked_778's R-side geometry to match spec16's (fully-optimized isolated fragment), leaving the CP treatment unchanged. Under the actual data that would:

- Fix the *geometry* axis (both halves would now use isolated-fragment R-side geometry).
- **Leave the CP axis divergent**: locked_778 R-side CP-corrected, spec16 R-side not.
- **Leave the spec16 internal CP asymmetry unchanged**: TS-side CP-corrected, R-side not.

Per §7 open item 1 default, **spec20 is halted at G20-0 pending a user decision** among:

1. **Full unification (largest scope)** — recompute strain for all 400 rows under a single protocol (either all CP or none). This subsumes spec20 and goes beyond it.
2. **Geometry-only unification** — proceed with spec20 as written, accepting the residual CP-axis divergence. Requires documenting a new deviation (spec20 fixes #8 partially; a new deviation records the residual CP mismatch).
3. **Fallback to Option B (§7 open item 4)** — restrict the downstream training cohort to the 208 spec16 rows, accept spec16's own internal CP asymmetry as a documented caveat, and skip the relaxation altogether.
4. **Standby** — halt the Espley replication until the label pipeline is redone under a single protocol.

No production compute (the 384-job array) has been submitted.

## Files

```
Ref Comparison/spec20_locked778_fragment_relax/
  code/{discover_protocol.py, aggregate.py, submit_s20.sh}
  logs/{protocol_discovery.json, discover.log, G20_0_HALT.flag}
  results/{DEVIATIONS.md, summary.md}
```

