# DEVIATIONS — spec20 delta

This file records what spec20's G20-0 protocol discovery found. It supplements (does not replace) the cross-stage `DEVIATIONS.md` in spec19_espley_s2_structures.

## Update to Deviation #8 (from Stage 2)

Deviation #8 said the 192 `locked_778` rows use R.xyz atom subsets (not independently-optimized isolated fragments) as their relaxed-fragment reference, while the 208 `spec16` rows use fully-optimized isolated fragments (`opt.xyz`).

G20-0 confirms this geometric divergence AND reveals a further protocol asymmetry:

- Both halves compute E(TS_A), E(TS_B) with counterpoise correction (`:(1)`/`:(2)` ghost atoms in `eda_frag{1,2}.inp`).
- locked_778 computes E(R_A), E(R_B) with CP correction (`v9_review/strain_sp_cp/{rid}/frag{A,B}_R.inp` — 7-11 ghost atoms).
- spec16 computes E(R_A), E(R_B) as isolated fragment optimizations with NO CP (`spec16_orca_strain/inputs/{rid}__f{A,B}/opt.inp`).

Consequently: **spec16 has an internal CP asymmetry** on strain (TS-side CP, R-side no CP); **locked_778 is internally consistent** (CP on both sides).

## Provisional Deviation #9 (needs user decision at G20-0)

Whichever unification path the user chooses at G20-0, the CP treatment of the R-side energies must be equalised before Deviation #8 can be marked resolved. Options are enumerated in `summary.md` §Interpretation.
