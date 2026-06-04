# eda-asm-prediction

EDA (Energy Decomposition Analysis) and ASM (Activation Strain Model)
based prediction of activation energies (E_a). Stage 5–6 of an
explainable E_a pipeline; complements Stage 2 (TS structure generation)
in the sibling repo `ts-structure-prediction`.

## What this repo does

1. Selects 400 reaction trajectories (200 `T1x_*` + 200 `Halogen_*`)
   from the Halo8 ASE-DB collection.
2. Identifies R / TS / P frames per trajectory.
3. Runs ADF NOCV-EDA + ASM strain calculations to decompose the
   activation energy into 5 channels:
   `E_Pauli, E_elstat, E_orb, E_disp, E_strain`.
4. Writes a single labeled dataset preserving every Halo8 per-frame
   field plus the new ADF decomposition.
5. (Stage 6, separate scripts) Trains GPR-ARD over those 5 channels →
   E_a with uncertainty.

## Layout

```
configs/   YAML run configs (ADF settings, sampling seed, paths)
data/
  Halo8/   read-only symlink → ts_prediction_project/data (10 ASE DBs)
  selection/      sampled trajectory ids (json, version-controlled run output)
  processed/      eda_asm_dataset.parquet (the deliverable)
runs/      one dir per trajectory: ADF inputs, outputs, logs (gitignored)
scripts/   CLI entrypoints (sampling, ADF launch, aggregation)
src/eda_asm/
  io/      ASE-DB readers, parquet writer
  traj/    R/TS/P picking, fragment splitting
  adf/     AdfRunner, input templating, output parser
  agg/     dataset aggregator
notebooks/ exploratory only
tests/     pytest, unit + smoke
```

## Data source — Halo8

`data/Halo8/Halo_{1..10}.db` are ASE SQLite databases, ~22M frames
total. **One row = one frame** of a reaction trajectory.

Frame id: `row.data["dand_id"]`, e.g.
`Halogen_C4FH5N2O_rxn14045_17` — trailing `_17` is the frame index.
Trajectory id: `dand_id.rsplit("_", 1)[0]`.

Two families, both case-sensitive in the DB but matched
case-insensitively in code:
- `T1x_*`   → user shorthand "t1x"   → `dand_id.lower().startswith("t1x")`
- `Halogen_*` → user shorthand "halo" → `dand_id.lower().startswith("halo")`

Per-frame fields preserved in the output dataset: `Mulliken_charges`,
`Lowdin_charges`, `Dipole_moment`, `Nuclear_repulsion_energy`,
`Electronic_energy`, `One_electron_energy`, `Two_electron_energy`,
`Exchange_energy`, `Correlation_energy`, `Dispersion_correction`,
`HOMO_idx/level`, `LUMO_idx/level`, plus `positions`, `forces`,
`energy`, `formula`, `numbers`, `symbols`, `charge`. See
[`CLAUDE.md`](CLAUDE.md) for the full output schema.

## Setup

```bash
mamba env create -f env.yaml
mamba activate eda_asm
pip install -e .[dev]
```

ADF is **not yet installed**. Before running production EDA, follow
the ADF-setup checklist in [`CLAUDE.md`](CLAUDE.md#adf-setup-not-yet-done).
At minimum:
- decide on HPC module / local install / container
- confirm a working license
- pass `scripts/smoke_test_adf.py` (one EDA-NOCV on ethane → 2×CH₃)

## How to run the 400-trajectory dataset build

```bash
# 1. Sample and freeze the trajectory selection (deterministic).
python scripts/select_trajectories.py \
    --config configs/sampling.yaml \
    --out data/selection/selected_trajectories.json

# 2. Stage R/TS/P + fragment splits + ADF input decks.
python scripts/prepare_adf_inputs.py \
    --selection data/selection/selected_trajectories.json \
    --out runs/

# 3. Submit ADF jobs (Slurm array, one task per trajectory).
sbatch scripts/run_adf_array.sh

# 4. Aggregate parsed results into the parquet deliverable.
python scripts/aggregate_dataset.py \
    --runs runs/ \
    --out data/processed/eda_asm_dataset.parquet
```

Each step is **idempotent**: re-running skips trajectories whose
outputs are already present and parseable. Failures are logged to
`runs/_failures.jsonl` and not retried automatically.

## Output

`data/processed/eda_asm_dataset.parquet` — one row per trajectory.
Contains: trajectory id + family, R/TS/P geometries and forces,
all per-frame Halo8 fields for R/TS/P, ADF-derived total energies
for full system and fragments, the 5 EDA-ASM channels, sanity
checks (`Ea_reconstructed` vs `Ea_from_trajectory`), and
provenance (ADF version, basis, functional, fragmentation method,
seed, run timestamp, status). Full schema: [`CLAUDE.md`](CLAUDE.md#output).

## Stage-2 ↔ Stage-5 consistency

The weighted-RMSD weighting kernel used here when comparing predicted
TS to DFT TS **must match** the FM loss weighting in
`ts-structure-prediction` (cb-* branches). Keep
`src/eda_asm/weights.py` in sync with the upstream definition; a unit
test (`tests/test_weight_consistency.py`) loads both definitions and
asserts numerical equality on a fixed (R, TS, P) example.

## Phase 1 — Halo8 sampling and fragment definition

Implemented in [`src/eda_asm/phase1/`](src/eda_asm/phase1/) and driven by
[`scripts/run_phase1.py`](scripts/run_phase1.py). Resumable, stage-by-stage:

```bash
# Stages 3.1 → 3.8 (stops at the manual-review gate)
python scripts/run_phase1.py --to 3.8

# After populating outputs/phase1/manual_review_log.json (one entry per
# reviewed reaction), run the final integration:
python scripts/run_phase1.py --finalize
```

Outputs land under [`outputs/phase1/`](outputs/phase1/):
- `phase1_output.h5` — the main dataset (one HDF5 group per reaction).
- `fragments_final.json` — human-readable fragment definitions.
- `selected_reactions.csv` — sampling metadata + cell labels.
- `sampling_report.html` — sample vs population marginals + final stats.
- `manual_review_queue/<rxn>.html` — 3D viewers for flagged reactions.
- `bond_changes.json`, `case_classification.json`, `fragments_auto.json`.

Halo8 indexing artefacts cached at `data/halo8_index/` (parquet).
Per-stage logs stream to `logs/phase1.log` plus `logs/stage_3_*.console.log`.

### Phase 1 status (2026-05-08)

- 400 reactions sampled (T1x:190, Halo_F/Cl/Br:70 each), seed 42.
- 5-point IRC bundles extracted for all 400 (R, ζ=0.25/0.5/0.75, TS).
- Fragment definitions for **355 / 400** reactions (233 Case B, 122 Case
  C). 45 reactions are truly concerted multi-bond — no single, pairwise,
  or union-graph cut produces 2 components — so they remain in the
  manual-review queue and are absent from `phase1_output.h5`.
- 100% SMILES + atom-index validity for the 355 included.
- Definition-of-Done (spec §9) checks pass.

## Phase 1.5 — Comprehensive Fragment Review Tool

A Flask + py3Dmol web app for reviewing every fragment definition by hand.
Source: [`tools/phase1_5_review/`](tools/phase1_5_review/).

```bash
cd tools/phase1_5_review
flask run --port 8888 --host 0.0.0.0   # or:  python app.py
# open http://localhost:8888

# When done reviewing all 400 reactions:
python tools/phase1_5_review/finalize_phase1.py
```

What it does:
- Loads all 400 reactions from `phase1_output.h5` + `.tmp/<rxn>.npz` so
  even rejected/Case-C reactions are reviewable.
- Dashboard with progress, status filter, jump-to-next-unreviewed.
- Per-reaction page: 3 viewers (R / midpoint / TS), atom-index labels,
  fragment colors, dashed reactive bonds.
- Modify mode: click any atom in any viewer to flip its fragment.
- Real-time validation (atom partition, SMILES, H-cap geometry).
- Auto-save on every decision; atomic JSON writes; 5-min snapshots
  under `outputs/phase1.5/snapshots/`.
- Audit trail at `outputs/phase1.5/review_audit.json`.
- `finalize_phase1.py` copies the review log into `manual_review_log.json`
  and re-runs Stages 3.9 + 3.10 so the modified definitions land in
  `phase1_output.h5`.

## Status

2026-05-08 — Phase 1 done (355/400 reactions, awaits user sign-off).
ADF not installed; required for Phase 2.
See [`CLAUDE.md`](CLAUDE.md) for the authoritative task specification.
