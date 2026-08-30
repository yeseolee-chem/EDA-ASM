# B3LYP relabel validation — REPORT (spec15)

_Generated: 2026-08-30T22:01:47.706634+00:00, commit `63678f44aa7f`_

## Setup
- Goal: validate B3LYP-TS relabeling pipeline on 200-sample batch
  before scaling to the full 5269 ds3 reactions.
- 5 SPEs per rxn: complex EDA, frag1/2 distorted (BARE, no ghost),
  frag1/2 relaxed (at ds3 r*.xyz).
- All 218 spec14 rxns reused (EDA cached, only fragments new).
- strain = (dist − rel) × 627.5094740631 (matches existing convention).

## Gates
| Gate | Status |
|---|---|
| GATE-0 (inputs present) | ✅ PASS profiles=5269 results_218=218 cohort=3504 |
| GATE-1 (sample includes all 218) | ✅ PASS n=218 eda_reuse=218 eda_new=0 |
| GATE-2 (inputs built ≥98%) | ✅ PASS built=218/218 rate=1.000 parity rxn_0003 header: IDENTICAL |
| GATE-3 (parse ≥95%, no sign viol.) | ❌ FAIL parsed=218/218 rate=1.000 sign_viol_channels=1 |
| GATE-4 (δ_prop within ±20% of spec14) | ✅ PASS violations=[] |
| GATE-5 (strain validation 4/4) | ❌ FAIL internal=True sanity=True nonneg=False corr=True |

## Label shift (AM1 TS → B3LYP TS)
```
channel   n  mean_signed    std    mae  median_abs    p95  delta_prop  pearson_r
   elst 218       14.807  9.736 15.805      15.582 28.741      13.768      0.843
  pauli 218      -33.046 18.872 34.357      33.532 61.914      26.689      0.837
     oi 218       17.634 12.552 19.383      20.079 34.893      17.752      0.754
   disp 218       -0.633  1.799  1.280       0.769  3.839       2.544      0.927
   cpcm 218        1.941  3.890  2.828       1.560 10.778       5.501      0.742
    cds 218       -0.167  0.473  0.333       0.214  1.187       0.670      0.892
     d1 218       -8.320  6.306  8.781       7.819 17.867       8.918      0.703
     d2 218       -6.576  4.961  6.790       6.025 14.672       7.016      0.781
```

- `mean_signed` = systematic bias (B3LYP − AM1)
- `delta_prop` = std × √2 = pair-difference noise floor
- `pearson_r` = agreement between old and new labels

## Full 5269 run authorization
- All gates PASS: **NO — resolve failing gates first**

## Preservation of old (AM1-geom) labels
- DO NOT delete phase5_dataset_v2.pkl or reactions/rxn_XXXX/ tree.
- Rename in v2 rollout: `labels_am1geom_v1/` (existing),
  `labels_b3lypgeom_v2/` (new).
- README should explicitly note that v1 is preserved for reproducibility
  and comparison; only v2 is used by training scripts going forward.

## Files
- `artifacts/GATE0_STATUS.txt`
- `artifacts/GATE1_STATUS.txt`
- `artifacts/GATE2_STATUS.txt`
- `artifacts/GATE3_STATUS.txt`
- `artifacts/GATE4_STATUS.txt`
- `artifacts/GATE5_STATUS.txt`
- `artifacts/input_meta.csv`
- `artifacts/label_shift.csv`
- `artifacts/labels_b3lyp.csv`
- `artifacts/sample.csv`