# DEVIATIONS — Espley et al. replication line

Started at Stage 1 (spec18r1_espley_s1_labels_fix). Every downstream stage inherits and appends to this list.

| # | deviation | stage | rationale |
|---|---|---|---|
| 1 | ORCA replaces Gaussian 16 | 3 | user-mandated; no Gaussian in project stack for this line |
| 2 | GFN2-xTB replaces AM1 as the SQM level | 3 | ORCA has no AM1; GFN2 is the nearest SQM-tier substitute and the project's existing SQM engine |
| 3 | q_barrier (ΔG‡) omitted from both targets and features | 1, 4 | electronic-energy consistency (SPEC_14); 44 features vs their 45 |
| 4 | Reference DFT is ORCA ωB97X-3c EDA-NOCV, not B3LYP-D3(BJ)/def2-TZVP + SMD | 1 | our labels; must be noted on every cross-paper kcal/mol comparison |
| 5 | Cohort is 400 dipolar [3+2] cycloadditions, not 3510 | 1 | user-requested restriction to the 400-set (192 locked_778 + 208 spec16) |
| 6 | f_select.py line ~226 substring `am1` → `am1|gfn2` (one-line patch required) | 3 | Rev 1 G-E confirmed `_gfn2` targets are dropped by the unmodified _manual_runner; either patch the substring or rename `_gfn2` → `_am1` at Stage 3 |

## Implementation notes

**Artifact of record is `.pkl`, not `.parquet`.** pyarrow unions per-row dict keys into a single struct schema and drops keys not present on every row. With `distortion_contributions_dft` keys of the form `{n_1, n_2}` differing per row, parquet expands into an 800-field struct populated with ~319,200 `None`s (Rev 0's failure mode, F1). Pickle preserves the dicts exactly. An inspection parquet is also written with the dict column DROPPED to prevent confusion.

**Sign convention.** `interaction_energies_dft = −int_eda_kcal` (paper stores positive, our EDA E_int is negative and stabilising). Independence verified in G-C by re-summing the four channels from the source parquet.

**ASM identity residual = source rounding floor.** G-D reports `|strain + resum(channels) − act_kcal|` per row. `int_eda_kcal` in the source is a 2-decimal round of the channel sum, so this residual sits at ~0.02 kcal/mol; larger residuals surface real label-pipeline drift.

**Reaction_number is order-stable.** `reaction_number` is assigned after `sort_values('reaction_id', kind='mergesort')` (fixes F7). G-F confirms sorting the artifact by reaction_id reproduces `reaction_number == 0..399`.

**Contributions dict sum tolerance 5e-3, not 1e-6.** Source-parquet rounding of `strain_A_kcal`, `strain_B_kcal`, `strain_kcal` produces up to 1e-3 kcal/mol drift on the 208 spec16 rows. A real schema bug would show ≥ 0.1 kcal/mol and still be caught.

**No CONTAM filter applied.** None of the 5 SPEC_10 dipolar CONTAM ids appear in the 400-set (verified 2026-07-24).
