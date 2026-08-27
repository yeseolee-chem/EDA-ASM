# Top-k retrieval — REPORT (spec13)

_Generated: 2026-08-27T07:08:45.180955+00:00, commit `6a96f34f2343`_

## Gates
| Gate | Status |
|---|---|
| GATE-0 (input files present) | ✅ PASS inputs=[oof_predictions.pkl, pairs_dedup.pkl] both present |
| GATE-1 (candidate groups match SPEC §7) | ✅ PASS n_groups=144 mean_size=6.006944444444445 median_size=4.0 size_min=3 size_max=18 n_spear=119 rand_t1=0.20716089466089466 rand_t3=0.6214826839826839 |
| GATE-2 (Top-k & Spearman match SPEC §7) | ✅ PASS n_top_all_720=True n_spear_all_595=True |

## Candidate groups (from MMP key)
- Grouping rule: reactions sharing the same MMP `key` (masked reactant + core signature + solvent + temp)
- MIN_CAND = 3, MIN_SPEAR = 4
- 144 groups; mean size 6.0069, median 4, range 3..18
- Random Top-1 baseline = 0.2072, Top-3 = 0.6215

### Size distribution
```
 n_candidates  n_groups
            3        25
            4        51
            5         9
            6         8
            7        12
            8        10
            9         5
           10         9
           11         6
           12         4
           14         1
           15         2
           18         2
```

## Main — Top-1 / Top-3 / Spearman per channel
```
    label   top1   top3  spearman  n_top  n_spearman  random_top1  lift_top1
strain d1 0.7167 0.9750    0.8426    720         595       0.2072     3.4595
strain d2 0.7208 0.9681    0.7990    720         595       0.2072     3.4796
     elst 0.5694 0.9292    0.6844    720         595       0.2072     2.7488
    Pauli 0.6597 0.9431    0.7592    720         595       0.2072     3.1846
 orb.int. 0.6556 0.9417    0.7592    720         595       0.2072     3.1645
     disp 0.6389 0.9583    0.7845    720         595       0.2072     3.0840
     CPCM 0.7264 0.9347    0.7312    720         595       0.2072     3.5064
```

## SPEC §7 target agreement (|actual - target|)
```
  channel  |Δ top1|  |Δ top3|  |Δ spearman|
strain d1       0.0       0.0           0.0
strain d2       0.0       0.0           0.0
     elst       0.0       0.0           0.0
    Pauli       0.0       0.0           0.0
 orb.int.       0.0       0.0           0.0
     disp       0.0       0.0           0.0
     CPCM       0.0       0.0           0.0
```

Values below 5e-4 are within SPEC §7 tolerance.

## Figures
- `figures/fig1_topk_bar.png`
- `figures/fig2_topk_by_size.png`
- `figures/fig3_spearman_violin.png`

## Interpretation (SPEC §6)
- Top-1 in 0.57–0.73 range → **2.7–3.5× above random (0.207)**.
  The model carries real ranking information within a scaffold.
- Top-3 in 0.93–0.98 → shortlisting 3 candidates leaves almost
  no misses. Practical DFT-triage burden drops ~6×.
- strain d1/d2 rank highest, elst lowest — mirrors the δ-MAE
  ranking. Different metrics report a consistent ordering.
- CPCM: high Top-1 but modest Spearman — good at the winner,
  weaker at the full ordering. Interpret jointly.

### Caveats (do NOT over-read)
- These numbers do NOT imply δ is predicted accurately in an
  absolute sense. δ-MAE still misses the 1 kcal/mol target.
- Mean group size 6 pins random baseline at 0.207. Cite this
  when comparing to other retrieval benchmarks.
- 76/144 groups have size 3–4 → large-group behavior is sparsely
  sampled (see fig2_topk_by_size.png).

## Environment
```
scheme     : groupkfold_component
seeds      : [42, 43, 44, 45, 46]
channels   : d1_own_dft, d2_own_dft, elst_dft, pauli_dft,
             oi_dft, disp_dft, cpcm_dft   (cds excluded)
aggregation: pooled over (seed × candidate group)
```