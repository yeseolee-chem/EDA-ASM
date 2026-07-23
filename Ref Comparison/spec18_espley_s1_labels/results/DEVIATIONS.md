# DEVIATIONS — Espley et al. replication line

Started at Stage 1 (spec18_espley_s1_labels). Every downstream stage inherits and appends to this list.

| # | deviation | stage | rationale |
|---|---|---|---|
| 1 | ORCA replaces Gaussian 16 | 3 | user-mandated; no Gaussian in project stack for this line |
| 2 | GFN2-xTB replaces AM1 as the SQM level | 3 | ORCA has no AM1; GFN2 is the nearest SQM-tier substitute and the project's existing SQM engine |
| 3 | q_barrier (ΔG‡) omitted from both targets and features | 1, 4 | electronic-energy consistency (SPEC_14); 44 features vs their 45 |
| 4 | Reference DFT is ORCA ωB97X-3c EDA-NOCV, not B3LYP-D3(BJ)/def2-TZVP + SMD | 1 | our labels; must be noted on every cross-paper kcal/mol comparison |
| 5 | Cohort is 400 dipolar [3+2] cycloadditions, not 3510 | 1 | user-requested restriction to the 400-set (192 from LOCKED_778 + 208 from spec16 LC-extension) |

## Implementation notes

**Interaction reconstructed from `int_eda_kcal`, not re-summed.** The source parquet rounds `pauli_kcal + elst_kcal + orb_kcal + disp_kcal` to a slightly-rounded `int_eda_kcal` (max drift 0.02 kcal/mol). Using the recorded `int_eda_kcal` reproduces `act_kcal` exactly (Gate #4 diff = 0.000 kcal/mol); re-summing the channels would leave a 0.02 kcal/mol floor.

**No CONTAM filter applied.** None of the 5 SPEC_10 dipolar CONTAM ids appear in the 400-set (verified 2026-07-24). Cohort size = 400 = 192 (locked_778) + 208 (spec16 LC-extension).

**Contributions dict sum tolerance 5e-3, not 1e-6.** The source parquet stores `strain_A_kcal`, `strain_B_kcal`, and `strain_kcal` as independently rounded floats; strain_A + strain_B differs from strain by up to 1e-3 kcal/mol on the spec16 half of the cohort. A tighter tolerance would fail on source rounding. A real schema bug (e.g. swapped fragments, missing atoms) would show ≥ 0.1 kcal/mol drift and still be caught.
