# build_strategy — 5-layer fragmentation for spec16rev b3lyp_full

Consolidates the primary rule (originally in `../01_build_inputs.py`)
and the four recovery strategies (originally scattered in
`../viz/recover_*.py`) into a single self-contained builder that
covers all 5269 Coley reactions.

## Files

- **`build_inputs.py`** — the 5-layer builder. Iterates through
  Coley rxn_ids, tries strategies in order, writes 5 ORCA `.inp`
  files per rxn to `analysis/b3lyp_full/inputs/rxn_NNNN/`.
- **`expected_recovery.json`** — reference mapping of the 7
  non-primary rxns to the strategy that recovers them.
- **`test_partition.py`** — validates that the actual assignment
  matches the reference.

## Strategy chain (first-winner-per-rxn)

| # | strategy      | reference wins | notes |
|---|---------------|----------------|-------|
| 1 | `primary`     | 5262 | geom factor 1.25, strict reactant match. Forming bonds NOT explicitly excluded (auto-excluded by distance). |
| 2 | `self_cyclo`  | 4 (4327-4330) | strategy 1 + allow both fragments to point to same reactant file |
| 3 | `factor_1_20` | 1 (4252) | strategy 1 with covalent factor lowered to 1.20 |
| 4 | `smiles_map`  | 1 (3090) | Coley SMILES atom-map k → TS atom index (k-1); H by nearest-heavy |
| 5 | `heavy_prefix`| 1 (3766) | TS heavy atoms = [rA heavies verbatim] + [rB heavies (any perm)] + H (nearest) |

Reference total: **5269/5269 (100%)**.

## Usage

```bash
# Sbatch (per CLAUDE.md — no Python on login node)
cat > /tmp/build.sh <<EOF
#!/bin/bash
#SBATCH --job-name=b3build
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/build.%j.log
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/build_inputs.py
python /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/test_partition.py
EOF
sbatch /tmp/build.sh
```

- Idempotent: skips rxns whose 5 `.inp` files already exist.
- Writes per-rxn recovery method to
  `build_strategy/artifacts/input_meta.csv`.
- `--dry-run`: report method-per-rxn without writing any files
  (useful for verification on a fresh checkout).

## Why the split from `01_build_inputs.py`

The original `01_build_inputs.py` only implements the `primary` rule
and skips the 7 remaining rxns as `build_failures.csv`. During spec16rev
development (2026-09-07) we found that:

1. Different failure modes need different recovery methods (see
   `../viz/diagnose_fails.py` diagnostic output).
2. Using a single alternate rule (e.g. SMILES atom-map only) covers
   just 43% of the dataset — worse than primary.
3. Chaining the 5 strategies in order covers 100%.

`01_build_inputs.py` is kept unchanged for historical reference; this
consolidated builder should be preferred for new runs.
