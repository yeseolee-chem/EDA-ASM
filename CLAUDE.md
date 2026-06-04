# CLAUDE.md — eda-asm-prediction

## Project Overview

EDA (Energy Decomposition Analysis) and ASM (Activation Strain Model) based
prediction of activation energies (E_a). This is **Stage 5–6** of an
explainable E_a pipeline; **Stage 2** (TS structure generation via React-OT
/ OT-FM) lives in `/home1/yeseo1ee/projects/ts-structure-prediction`.

Folder/distribution name: `eda-asm-prediction` (hyphenated).
Python import name: `eda_asm` (hyphens are not valid in Python identifiers).

## Pipeline

```
Stage 1: Energy prediction (AIMNet2)
Stage 2: TS structure (React-OT / OT-FM)            [external repo]
Stage 5: EDA/ASM proxy decomposition                [this repo]
Stage 6: GPR-ARD over decomposed components → E_a   [this repo]
```

Inputs: (R, TS, P) geometries — TS may come from Stage 2 predictions or
DFT references. Outputs: per-reaction decomposed energy components and an
E_a estimate with uncertainty.

## Decomposition Components (target signature)

ASM (two-fragment): ΔE‡ = ΔE_strain + ΔE_int

EDA on ΔE_int (Morokuma / ALMO-style proxies, ADF NOCV-EDA):
1. **Pauli repulsion** — `E_Pauli`
2. **Electrostatic** — `E_elstat`
3. **Orbital interaction** (charge transfer + polarization) — `E_orb`
4. **Dispersion** — `E_disp`
5. **Strain (preparation/distortion)** — `E_strain` = sum of fragment ΔE_prep

These five channels feed Stage 6 GPR-ARD as features.

> The first four are "interaction" channels from EDA on ΔE_int.
> The fifth is the ASM strain term. Together with `E_int = E_Pauli +
> E_elstat + E_orb + E_disp`, they reconstruct ΔE‡ = E_strain + E_int.

## Cross-Repo Consistency Note

The weighted-RMSD weighting used in Stage 5 must share the same continuous
weight function (graph-distance / element / hierarchical kernel) as the FM
loss weighting in Stage 2 (cb-* branches in `ts-structure-prediction`).
Keep the weighting kernel definitions in sync.

---

## Data: Halo8 reaction trajectories

Source (read-only, mounted via symlink):
```
data/Halo8/   →  /home1/yeseo1ee/projects/ts_prediction_project/data/
```
Contents: `Halo_1.db` … `Halo_10.db` — ten ASE SQLite databases,
~22M rows total. Each **row is one frame** of a reaction trajectory.

### Trajectory key (very important)

Frame-level identifier:           `row.data["dand_id"]`
Examples:
- `T1x_C2H2N2O_rxn00001_99`        (T1x family)
- `Halogen_C4FH5N2O_rxn14045_17`   (Halogen family)

The trailing `_<int>` is the **frame index** along the trajectory.
The **trajectory id** is `dand_id.rsplit("_", 1)[0]`, e.g.
`Halogen_C4FH5N2O_rxn14045`. Frame count per trajectory is typically
200–500, indexed from 0.

### Two reaction families

Both prefixes are **case-sensitive** in the database:
- `T1x_*` — Transition1x-derived reactions (organic, no halogens). The
  user often refers to these as "t1x"; match with
  `dand_id.lower().startswith("t1x")` to be robust.
- `Halogen_*` — halogenated reactions (F/Cl/Br substituted). The user
  often refers to these as "halo"; match with
  `dand_id.lower().startswith("halo")`. Note: lowercase `halo` is a
  prefix of lowercase `halogen`, so this match is correct.

Do **not** filter by `old_dand_id` — that field uses different prefixes
(`T1x_N_…`, `8F_N_…`, `567Br_…`) and is for cross-referencing to the
original source datasets only.

### Per-frame fields available (preserve all of these in outputs)

`row.data` keys:
```
dand_id, old_dand_id,
Mulliken_charges (list, len=natoms),
Lowdin_charges    (list, len=natoms),
Dipole_moment     (list[3]),
Nuclear_repulsion_energy, Electronic_energy,
One_electron_energy, Two_electron_energy,
Exchange_energy, Correlation_energy, Dispersion_correction,
HOMO_idx, HOMO_level, LUMO_idx, LUMO_level
```
`row` attributes: `positions (natoms,3)`, `forces (natoms,3)`,
`energy` (total), `formula`, `natoms`, `charge`, `numbers`, `symbols`.

`cell` is zero and `pbc` is False — these are isolated-molecule
calculations. The TS frame for each trajectory is the **max-energy
frame** along that trajectory (this is the convention; verify by
checking that R = frame 0 and P = last frame have lower energy than
the interior maximum before promoting it as TS).

---

## Current task: ADF EDA-ASM on 400 trajectories

Build a labeled dataset of decomposed activation energies for downstream
GPR-ARD training. **ADF is not yet installed on this machine — setup is
part of the task.**

### Selection

Sample **uniformly at random without replacement** from each family:
- 200 trajectories whose `dand_id` matches `^T1x_` (case-insensitive).
- 200 trajectories whose `dand_id` matches `^Halogen` (case-insensitive
  — equivalent to lowercase `startswith("halo")`).

Total: **400 trajectories**. Use a fixed `numpy.random.default_rng(seed)`
with the seed recorded in the run config; persist the chosen trajectory
ids to `data/selection/selected_trajectories.json` so the run is
reproducible and incremental re-runs hit the same set.

Sampling procedure:
1. Stream all rows across `Halo_1.db` … `Halo_10.db`, group by trajectory
   id (`dand_id.rsplit("_", 1)[0]`). Streaming + a dict keyed on traj id
   is fine — the unique-trajectory count fits in memory even if the row
   count does not.
2. Bucket trajectory ids into `T1x` / `Halogen` (case-insensitive prefix).
3. Drop trajectories shorter than some minimum (e.g. < 20 frames) or
   missing a clear interior energy maximum — log how many were dropped.
4. Sample 200 from each bucket with the seeded RNG.

### Per-trajectory ADF input prep

For each selected trajectory:
1. Load all frames, sorted by frame index.
2. Identify R = frame 0, P = last frame, TS = argmax of `row.energy`
   over interior frames. Sanity-check `E(TS) > E(R)` and `E(TS) > E(P)`;
   skip + log otherwise.
3. Decide ASM fragmentation. These are mostly intramolecular reactions,
   so two-fragment ASM is non-trivial. Default approach (override
   per-reaction if needed):
   - Run `ase.neighborlist` on R and on P.
   - Take the symmetric difference of bond sets → reactive bonds.
   - Cut the molecule along the bond(s) that change between R and P;
     the two resulting connected components are fragment A / fragment B.
   - If the cut produces > 2 components or 1 component (concerted ring
     reactions, etc.), record the case and skip ASM for that trajectory
     — still run a single-system EDA on the TS for diagnostics.
4. Write ADF input decks to `runs/<trajectory_id>/`:
   - `R.xyz`, `TS.xyz`, `P.xyz` — geometries.
   - `frag_A.xyz`, `frag_B.xyz` — frozen fragment geometries at TS.
   - `eda.run` — ADF EDA-NOCV input (single-point on TS with frags A/B).
   - `strain_A.run`, `strain_B.run` — single-point ADF on each fragment
     at TS geometry and at its R/P-relaxed geometry, for ΔE_prep.

### ADF settings (default — record in config, document any deviation)

- Functional: PBE0-D3(BJ) (or BLYP-D3 if matching DFT reference set).
- Basis: TZ2P, all-electron, frozen core None for elements ≤ Ar.
- Relativity: scalar ZORA only if Br is present in the formula.
- Symmetry: NOSYM.
- EDA scheme: NOCV / ETS-NOCV via ADF's `EDA` block; output the four
  channels listed above.
- Numerical quality: `Good`.

### Output

Write one row per trajectory to `data/processed/eda_asm_dataset.parquet`
(or .json lines if parquet unavailable). Schema:

```
trajectory_id          : str   # e.g. "Halogen_C4FH5N2O_rxn14045"
family                 : str   # "T1x" | "Halogen"
formula                : str
natoms                 : int
n_frames               : int
ts_frame_idx           : int
R_dand_id, TS_dand_id, P_dand_id : str
R_positions, TS_positions, P_positions : (natoms,3) arrays
R_forces, TS_forces, P_forces          : (natoms,3) arrays
numbers, symbols       : list[int], list[str]
charge                 : float

# Reference (Halo8) per-frame fields, replicated for R/TS/P:
{R,TS,P}_energy                       : float
{R,TS,P}_Nuclear_repulsion_energy     : float
{R,TS,P}_Electronic_energy            : float
{R,TS,P}_One_electron_energy          : float
{R,TS,P}_Two_electron_energy          : float
{R,TS,P}_Exchange_energy              : float
{R,TS,P}_Correlation_energy           : float
{R,TS,P}_Dispersion_correction        : float
{R,TS,P}_HOMO_idx, {R,TS,P}_HOMO_level: int, float
{R,TS,P}_LUMO_idx, {R,TS,P}_LUMO_level: int, float
{R,TS,P}_Mulliken_charges             : list[float]
{R,TS,P}_Lowdin_charges               : list[float]
{R,TS,P}_Dipole_moment                : list[float, 3]

# ADF-derived (this is the new content):
adf_total_E_TS          : float       # ADF SP energy at TS (full system)
adf_total_E_fragA_TS    : float       # at TS geometry
adf_total_E_fragB_TS    : float
adf_total_E_fragA_R     : float       # relaxed-fragment ref energies
adf_total_E_fragB_P     : float

# EDA-ASM channels (the 5 features for Stage 6):
E_Pauli                 : float
E_elstat                : float
E_orb                   : float
E_disp                  : float
E_strain                : float       # = ΔE_prep_A + ΔE_prep_B
E_int                   : float       # = E_Pauli + E_elstat + E_orb + E_disp
Ea_reconstructed        : float       # = E_strain + E_int  (sanity check)
Ea_from_trajectory      : float       # = E_TS - E_R from Halo8 energies

# Provenance:
adf_version, adf_functional, adf_basis, adf_settings_hash : str
fragmentation_method    : str         # "neighborlist_diff" | "manual" | ...
fragA_atom_indices      : list[int]
fragB_atom_indices      : list[int]
seed                    : int
run_timestamp           : str (ISO 8601)
status                  : str         # "ok" | "skipped:<reason>" | "failed:<reason>"
```

Also keep raw ADF outputs under `runs/<trajectory_id>/adf_out/` — do
**not** delete them after parsing; they are needed for audits and
re-parsing if the EDA breakdown definition changes.

### ADF setup (not yet done — first task before any production run)

1. Decide deployment: HPC module (`module load adf/...`) vs local
   install vs containerized. Record the chosen path in
   `configs/adf.yaml`.
2. License: ADF requires a valid license file. Confirm with the user
   before downloading or activating.
3. Provide a smoke-test driver `scripts/smoke_test_adf.py` that runs
   one EDA-NOCV on a tiny system (e.g. ethane → 2 × methyl) and
   verifies output parsing. CI / dev iterations should call this
   first; do not launch the 400-job batch until it passes.
4. Wrap ADF invocation behind `eda_asm.adf.AdfRunner` so that
   alternative engines (PySCF EDA, Q-Chem ALMO-EDA) can be swapped
   in later if license/throughput demands it. Same I/O contract.

### Job orchestration

400 trajectories × (1 EDA + 2 strain SPs + 2 fragment-relaxed SPs)
≈ 2000 ADF jobs. Required:
- One Slurm array job per family, `--array=0-199%<concurrency>`.
- Idempotent: if `runs/<traj>/eda.out` exists and parses cleanly, skip.
- Log failures to `runs/_failures.jsonl` with stderr tail.
- Aggregator script that walks `runs/` and writes the parquet output.

---

## Conventions

- Random seeds: always pass via config; never hard-code.
- All energies in eV unless explicitly noted (ADF natively reports
  hartree / kcal·mol⁻¹ — convert at parse time and store eV in the
  parquet).
- Geometries in Å.
- Never commit `data/` (gitignored) or `runs/` (add to `.gitignore`).
- Symlinks under `data/` are fine and expected; treat them as read-only.
