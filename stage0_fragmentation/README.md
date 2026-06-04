# Stage 0 — Electron-Flow Guided Fragmentation

BE-matrix-based automatic fragment partitioning for ASM-EDA. Implements the
spec at [`../stage0_fragmentation_spec.md`](../stage0_fragmentation_spec.md).

## What it does

Given atom-mapped reactant and product molecules (RDKit `Mol`), the module:

1. Builds Ugi–Dugundji bond–electron (BE) matrices for both endpoints.
2. Computes ΔBE = B_P − B_R and extracts:
   - **reactive bonds** — every bond whose order changed,
   - **migrating atoms** — atoms with at least one fully-broken loss *and*
     one fully-formed gain (so an SN2 carbon and a [1,3]-H shift hydrogen
     both qualify, but Diels–Alder atoms do not).
3. Runs connected-component analysis on the R skeleton with reactive bonds
   and migrating atoms removed.
4. Picks fragment seeds via a **bipartite 2-colouring of the component
   graph induced by full-transition reactive bonds** — this avoids the
   "two arbitrary single Hs" failure mode of the naïve "top-2 by size"
   approach.
5. Routes each migrating atom into whichever fragment contains its dominant
   destination, and merges any stray non-migrating components into the
   fragment whose R-graph neighbours dominate.
6. Computes H-cap site lists for each fragment.

Falls back to:
- `user_hint["split_bond"]` — force a cut and partition.
- migration clustering — group migrating atoms into one moving unit.
- `strain_only` — flag the case for Stage 5 to use only the ASM strain
  channel and NaN out the EDA features.

## Usage

```python
from rdkit import Chem
from stage0_fragmentation import run_fragmentation

mol_R = Chem.AddHs(Chem.MolFromSmiles("[Cl-].C[Br]"))   # atom-mapped externally
mol_P = Chem.AddHs(Chem.MolFromSmiles("ClC.[Br-]"))

result = run_fragmentation(mol_R, mol_P)

print(result.fragments)            # [{0}, {1, 2, 3, 4, 5}]
print(result.migrating_atoms)      # [{"atom": 1, "from": [5], "to": [0], ...}]
print(result.reactive_bonds)       # [(0, 1), (1, 5)]
print(result.cap_sites)            # {0: [(0, 1)], 1: [(1, 0)]}
```

## Requirements

- Python ≥ 3.10
- `rdkit ≥ 2023.09`
- `numpy ≥ 1.24`
- `networkx ≥ 3.0`

For tests: `pytest`, `pytest-cov`. For lint: `mypy`, `ruff`.

## Limitations (v1)

- **Radicals not supported.** BE matrices encode lone-pair *electron count*
  per atom; an unpaired electron is rejected via `RadicalNotSupportedError`.
- **Pure concerted rearrangements** (e.g. Cope) often produce many tiny
  components after reactive-bond removal. The bipartite-colouring step
  recovers a 2-fragment split when feasible; truly irreducible cases
  trigger the `strain_only` fallback.
- **Aromatic resonance ambiguity.** RDKit's default Kekulization is used;
  cases where multiple Kekulé forms give different ΔBE patterns are not
  averaged. v2 might use a dual-Kekulé average per the FlowER convention.

## Tests

```bash
pytest stage0_fragmentation/tests/ -v
```

Reference reactions covered:
- **SN2** (`test_sn2.py`) — Cl⁻ + CH₃Br → ClCH₃ + Br⁻.
- **Diels–Alder** (`test_diels_alder.py`) — butadiene + ethylene → cyclohexene.
- **Keto–enol [1,3]-H shift** (`test_ring_contraction.py`) — used as a
  miniature stand-in for the spec's ring-contraction case (radical-free,
  same migrating-atom mechanism).
- **Cope** (`test_cope.py`) — 1,5-hexadiene rearrangement.
- BE-matrix unit tests (`test_be_matrix.py`).

## Layout

```
stage0_fragmentation/
├── __init__.py          # public re-exports
├── be_matrix.py         # BE matrix construction + validation
├── migration.py         # migrating-atom detection (strict definition)
├── partition.py         # connected components + routing + result validation
├── capping.py           # H-cap site discovery + per-fragment SMILES
├── rearrangement.py     # fallback strategies for pure rearrangements
├── api.py               # run_fragmentation orchestrator
├── debug.py             # Appendix-B helpers (matrix pretty-printers, draw)
├── types.py             # FragmentationResult dataclass
└── tests/               # pytest reference reactions + unit tests
```
