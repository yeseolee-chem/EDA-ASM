# Geometry-level validation — REPORT (spec14)

_Generated: 2026-08-30T12:47:48.019774+00:00, commit `eb9d6e1d857c`_

## Setup
- Question: how much does the choice of TS geometry level (low-level
  semi-empirical vs B3LYP-D3BJ/def2-TZVP full DFT) shift the EDA channels?
- Cheap experiment: 218 EDA single-points at B3LYP-D3BJ/def2-TZVP CPCM(SMD,water)
  on ds3 (Stuyver/Coley 2023) B3LYP-optimised TS geometries, then compare
  channel-by-channel against current phase5 labels.
- Same B3LYP EDA header — differ only in xyz coordinates.
- strain (d1, d2) excluded: requires fragment re-optimization.

## Gates
| Gate | Status |
|---|---|
| GATE-0 (inputs present) | ✅ PASS n_profiles=5269 n_cohort=3504 missing=0 |
| GATE-1 (218 stratified sample) | ✅ PASS n=218 |
| GATE-2 (eda.inp built 218/218) | ✅ PASS built=218/218 failures=0 |
| GATE-3 (ORCA normal termination ≥95%) | ✅ PASS parsed=218/218 rate=1.000 sign_violations=0 |
| GATE-4 (channel δ-propagation verdict) | ℹ️ worst_delta_prop=48.588 PASS=1 CONDITIONAL=1 FAIL=4 |

## Main — channel shift table (B3LYP TS − low-level TS)
```
channel   n  mean_signed    mae  median_abs  p95_abs  max_abs  rel_to_scale  delta_propagated     verdict
   elst 218       14.807 15.805      15.582   28.741   42.288         0.304            22.351        FAIL
  pauli 218      -33.046 34.357      33.532   61.914   83.577         0.272            48.588        FAIL
     oi 218       17.634 19.383      20.079   34.893   40.104         0.308            27.412        FAIL
   disp 218       -0.633  1.280       0.769    3.839    8.657         0.083             1.810 CONDITIONAL
   cpcm 218        1.941  2.828       1.560   10.778   15.733         0.297             3.999        FAIL
    cds 218       -0.167  0.333       0.214    1.187    2.531         0.144             0.471        PASS
```

- `mae` = single-reaction absolute channel shift.
- `delta_propagated` = mae × √2 (independent-noise assumption for δ = B − A).
- Verdict scale: PASS if δ<1, CONDITIONAL if 1≤δ<3, FAIL if δ≥3.

## Formed-bond length shifts
- Sample: n = 218
  - d1: mean = +0.162 Å,  |diff| median = 0.168,  p95 = 0.368
  - d2: mean = +0.087 Å,  |diff| median = 0.154,  p95 = 0.382
- If |Δ formed-bond| correlates with |Δ channel|, the channel shift
  is attributable to geometry (not to solvent / SCF noise).

## Cohort-bias caveat
- The stratified sample includes only 9 CCNNN and 11 CCNNO reactions
  (multi-N dipoles are rare in the 3504 cohort). Any claim specific
  to these cores has low statistical weight; note in the paper.

## What this SPEC decides
- All PASS   → keep current labels; the 1 kcal/mol δ-MAE target is meaningful.
  Also expand training set with 960 fresh reactions (cost-justified).
- Mixed     → per-channel target revision; document geometry limit in paper.
- Any FAIL  → recompute 3504 labels at B3LYP-D3BJ TS geometry (EDA SPE
  only; no TS re-optimization needed since ds3 provides B3LYP TS).
- Either way → paper Methods must state the low-level TS + B3LYP SPE two-step
  protocol explicitly and cite this validation.

## Figures
- `figures/fig1_geometry_error.png`
- `figures/fig2_parity_by_channel.png`
- `figures/fig3_bondlen_vs_channel.png`

## Notes
- Header of every built eda.inp is byte-identical to the existing
  reactions/rxn_XXXX/eda.inp header. Only xyz coordinates differ.
- Channel parser reuses regexes from `scripts/parse_orca_5channel.py`
  (Pauli combined with Delta E^0(XC) per project convention).
- CPCM / CDS regex patterns verified on real output on execution day;
  see `artifacts/parser_notes.md` if reconciliation was needed.