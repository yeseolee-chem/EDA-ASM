# spec22 cohort recommendation

Consumes spec21 diagnostics. One decision, stated at the top.

## Decision: **Re-label all 400. Cohort is skewed vs. Stuyver, but the two halves overlap.**

Carry `sub_source` as a reported covariate downstream. The skew is a cohort property, not a between-half divergence.

- G21-B: PASS — halves geometry-homogeneous.
- D1 KS p (ours vs full): 2.211e-03
- D1 KS p (locked vs spec16): 3.700e-01

## Provisional caveats to carry forward

- **DEVIATIONS #4 is wrong today.** Every route line in `spec20/logs/protocol_discovery.json` reads `BLYP D3BJ def2-TZVP` — not ωB97X-3c. Correct this deviation at spec22 write-time.
- **spec20's CP asymmetry** is still open: spec16's R-side has no CP; the TS side does. Unifying labels also unifies CP.
- **G21-C spotcheck unreviewed:** D2 fractions are PROVISIONAL.

