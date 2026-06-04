"""Phase 0 — geometry build per V1 Claisen ASR/EDA spec §6.

For each substrate in substrates.csv:
  SMILES -> RDKit MolFromSmiles -> AddHs -> EmbedMolecule(ETKDGv3) -> MMFFOptimize
  Identify core atoms C1=C2-O3-C4-C5=C6 + R group atoms via substructure match
  Write runs/<id>/build/mol.xyz and atom_map.json, then .done_build sentinel

Idempotent: rows whose .done_build exists are skipped.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import deque
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT = Path(__file__).resolve().parent.parent
SUBSTRATES_CSV = PROJECT / "substrates.csv"
RUNS_DIR = PROJECT / "runs"

# SMARTS matches the 6-atom Claisen skeleton C1=C2-O3-C4-C5=C6
# (in SMILES order of `C=C(R)OCC=C`). Independent of R substituent at C2.
CORE_SMARTS = "[#6]=[#6][#8][#6][#6]=[#6]"
CORE = Chem.MolFromSmarts(CORE_SMARTS)
assert CORE is not None


def collect_r_group(mol: Chem.Mol, c2_idx: int, core_set: set[int]) -> list[int]:
    """BFS from C2 along non-core heavy-atom neighbors to enumerate the R group."""
    seen: set[int] = set()
    q: deque[int] = deque()
    for n in mol.GetAtomWithIdx(c2_idx).GetNeighbors():
        if n.GetIdx() not in core_set and n.GetAtomicNum() > 1:
            q.append(n.GetIdx())
            seen.add(n.GetIdx())
    while q:
        idx = q.popleft()
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = n.GetIdx()
            if j in core_set or j in seen or n.GetAtomicNum() == 1:
                continue
            seen.add(j)
            q.append(j)
    return sorted(seen)


def build_one(row: dict) -> tuple[str, str]:
    rxn_id = row["id"]
    R = row["R"]
    smiles = row["smiles"]
    out_dir = RUNS_DIR / rxn_id / "build"
    sentinel = out_dir / ".done_build"

    if sentinel.exists():
        return rxn_id, "skipped"

    out_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return rxn_id, "failed:smiles_parse"
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol_h, params)
    if cid < 0:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol_h, params)
    if cid < 0:
        return rxn_id, "failed:embed"

    rc = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
    if rc not in (0, 1):
        return rxn_id, f"failed:mmff_rc{rc}"

    matches = mol_h.GetSubstructMatches(CORE)
    if not matches:
        return rxn_id, "failed:no_core_match"
    # Multiple matches can occur if the molecule has more than one C=C-O-C-C=C
    # skeleton (none of our 15 substrates should). Take the first; warn.
    core = matches[0]
    C1, C2, O3, C4, C5, C6 = core

    # Sanity asserts (spec §6 Phase 0 verification)
    assert mol_h.GetAtomWithIdx(O3).GetAtomicNum() == 8, "O3 not oxygen"
    assert mol_h.GetAtomWithIdx(C2).GetHybridization() in (
        Chem.HybridizationType.SP2,
    ), "C2 not sp2"
    core_set = set(core)
    r_atoms = collect_r_group(mol_h, C2, core_set)
    if R == "H":
        assert not r_atoms, f"R=H but found R atoms: {r_atoms}"
    else:
        assert r_atoms, f"R={R} but no R atoms attached to C2"

    xyz_path = out_dir / "mol.xyz"
    Chem.MolToXYZFile(mol_h, str(xyz_path))

    atom_map = {
        "rxn_id": rxn_id,
        "R": R,
        "smiles": smiles,
        "natoms": mol_h.GetNumAtoms(),
        "core_smarts": CORE_SMARTS,
        "core_indices": {"C1": C1, "C2": C2, "O3": O3, "C4": C4, "C5": C5, "C6": C6},
        "R_first_atom_indices": [
            n.GetIdx()
            for n in mol_h.GetAtomWithIdx(C2).GetNeighbors()
            if n.GetIdx() not in core_set and n.GetAtomicNum() > 1
        ],
        "R_all_atom_indices": r_atoms,
        "n_core_matches": len(matches),
        "rdkit_etkdg_seed": 42,
    }
    with (out_dir / "atom_map.json").open("w") as fh:
        json.dump(atom_map, fh, indent=2)

    sentinel.touch()
    return rxn_id, "ok"


def main() -> int:
    with SUBSTRATES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    print(f"building {len(rows)} substrates -> {RUNS_DIR}/")
    results: list[tuple[str, str]] = []
    for row in rows:
        rxn_id, status = build_one(row)
        results.append((rxn_id, status))
        print(f"  [{rxn_id:6s}] {status}")

    ok = sum(1 for _, s in results if s == "ok")
    skipped = sum(1 for _, s in results if s == "skipped")
    failed = [r for r, s in results if s.startswith("failed")]
    print()
    print(f"summary: ok={ok}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print("failed ids:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
