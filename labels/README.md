# labels/ — 789-reaction EDA-ASM label datasets

Curated EDA-ASM labels across 4 reaction families (dipolar, rgd1,
qmrxn20_e2, qmrxn20_sn2), used as supervision for the Stage-6 delta
learners (m1 / m2 / m3).

## Contents

### `adf/adf_labels_v6_multifamily.parquet` (789 rows × 32 cols)
ADF NOCV-EDA labels — the canonical label set.
Channels: `E_strain_kcal`, `Pauli_kcal`, `V_elst_kcal`, `E_orb_kcal`,
`E_disp_kcal` (five decomposed features feeding Stage-6 GPR-ARD /
delta learners), plus per-fragment energies and derived quantities.

### `orca/orca_eda_labels.parquet` (789 rows × 23 cols)
ORCA EDA recompute on the same 789 reactions. Channels align 1:1
with the ADF set via the ORCA→ADF channel mapping convention.

### `orca/orca_strain_labels.parquet` (789 rows × 16 cols)
Fragment-relaxed energies from ORCA optimizations, used to build
the ORCA strain channel independently of the ADF pipeline.

### `adf_vs_orca_comparison.parquet` (789 rows × 61 cols)
### `adf_vs_orca_full_comparison.parquet` (789 rows × 49 cols)
Side-by-side ADF vs. ORCA labels for cross-engine sanity checks.

## Provenance

- Reaction cohort: multi-family expansion (v6), 789 accepted reactions.
- ADF settings: PBE0-D3(BJ), TZ2P, scalar ZORA when Br present, NOSYM.
- ORCA settings: functional/basis matched to ADF where possible; see
  `runs/orca_recompute/` (not committed) for raw inputs/outputs.
- Units: kcal/mol unless the column name says otherwise.

## Downstream consumers

- `m1/`, `m2/`, `m3/` — three delta-learner variants trained against
  these labels. See each folder's README.
