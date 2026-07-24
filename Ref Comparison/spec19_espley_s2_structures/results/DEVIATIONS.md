# DEVIATIONS — Espley et al. replication line

Started at Stage 1 (spec18r1_espley_s1_labels_fix). Every downstream stage inherits and appends to this list.

| # | deviation | stage | rationale |
|---|---|---|---|
| 1 | ORCA replaces Gaussian 16 | 3 | user-mandated; no Gaussian in project stack for this line |
| 2 | GFN2-xTB replaces AM1 as the SQM level | 3 | ORCA has no AM1; GFN2 is the nearest SQM-tier substitute |
| 3 | q_barrier (ΔG‡) omitted from both targets and features | 1, 4 | electronic-energy consistency (SPEC_14); 44 features vs their 45 |
| 4 | Reference DFT is ORCA ωB97X-3c EDA-NOCV, not B3LYP-D3(BJ)/def2-TZVP + SMD | 1 | our labels; must be noted on every cross-paper kcal/mol comparison |
| 5 | Cohort is 400 dipolar [3+2] cycloadditions, not 3510 | 1 | user-requested restriction to the 400-set (192 locked_778 + 208 spec16) |
| 6 | f_select.py line ~226 substring `am1` → `am1|gfn2` (one-line patch required) | 3 | Rev 1 G-E confirmed `_gfn2` targets are dropped by the unmodified _manual_runner |
| 7 | diassep.py NOT used — TS fragment partition inherited from EDA-NOCV label pipeline | 2 | User-mandated: fragment A/B split follows the (1)/(2) labels in eda.inp verbatim. Re-deriving from imaginary-mode displacements would risk a split inconsistent with the strain_A/strain_B labels fixed at Stage 1 |
| 8 | locked_778 r_A/r_B are R.xyz atom subsets, NOT independently-optimized isolated fragments | 2 | The 192 locked_778 reactions inherited relaxed-fragment geometries from the reactant-complex (R.xyz), which is what strain_A_kcal was computed against. The 208 spec16 reactions use fully-optimized opt.xyz. Both are consistent with their own strain-label semantics; the two halves have different DIAS definitions of 'relaxed reactant'. |

## Fragment A/B convention (INVARIANT — INHERITED, NEVER RE-DERIVED)

Fragment A = ORCA EDA `(1)` atoms in `eda.inp`  =  `strain_A_kcal`  =  Stage-1 dict key `{reaction_number}_1`.

Fragment B = ORCA EDA `(2)` atoms in `eda.inp`  =  `strain_B_kcal`  =  Stage-1 dict key `{reaction_number}_2`.

This is the user-mandated convention. No stage of this pipeline is permitted to derive fragmentation from independent methods (Svatunek displacement analysis, adjacency-graph reasoning, RDKit fragment detection, etc.). Downstream anomalies MUST be reported, never silently corrected.
