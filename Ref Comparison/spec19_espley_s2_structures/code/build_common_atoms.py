"""spec19 Stage 2 — build common-atom, mapping, and mol_types artifacts.

Inputs:
  results/manifest.pkl
  logs/discovery.json  (per-reaction rxn_smiles from Stuyver full_dataset.csv)

Outputs:
  results/common_atoms.pkl   — {rxn_num: {"r_A":[...], "r_B":[...], "ts":[...], "d_A":[...], "d_B":[...]}}
  results/mapping.pkl        — {rxn_num: {"ts_idx_A":[...], "ts_idx_B":[...]}}
  results/mol_types.pkl      — {rxn_num: {"A":"dipole"|"dipolarophile", "B": …}}
  results/common_atom_anomalies.csv — G2-E: rxns whose k-tuple ≠ (3,2,5,3,2)
  results/open_shell.csv     — G2-F: any fragment with multiplicity ≠ 1

Reacting-atom identification: RDKit + atom-mapped reactant→product SMILES.
An atom is "reacting" if its degree in the product differs from its degree
in the reactant (measured on the mapped fragments).

Fragment role assignment (dipole vs dipolarophile):
  - Fragment whose reacting-atom count is 3 → dipole
  - Fragment whose reacting-atom count is 2 → dipolarophile
Any other split lands in `common_atom_anomalies.csv` for user review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
DISCOVERY_JSON = STAGE / "logs/discovery.json"
MANIFEST = STAGE / "results/manifest.pkl"
STRUCT_ROOT = STAGE / "structures"

OUT_COMMON = STAGE / "results/common_atoms.pkl"
OUT_MAPPING = STAGE / "results/mapping.pkl"
OUT_MOLTYPES = STAGE / "results/mol_types.pkl"
OUT_ANOMALIES = STAGE / "results/common_atom_anomalies.csv"
OUT_OPENSHELL = STAGE / "results/open_shell.csv"
BUILD_LOG = STAGE / "logs/build.log"


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def read_xyz_elems(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    return [ln.split()[0] for ln in lines[2 : 2 + n]]


def reacting_atoms_from_smiles(rxn_smiles: str) -> dict:
    """Return {frag_index: {…}} — the atom-map ids of REACTING atoms per
    reactant fragment, using Espley's (3, 2) contract for [3+2] cycloadditions.

    Step 1 — bond-forming atoms: atom-map ids that acquire ≥ 1 new bond in
             the product (i.e. new inter-fragment or ring-closing bond).
    Step 2 — role assignment:
             - The fragment whose 2 bond-forming atoms are NOT directly bonded
               in the reactant is the DIPOLE (3 atoms: the two ends + the
               atom on the shortest reactant-graph path between them).
             - The fragment whose 2 bond-forming atoms ARE directly bonded in
               the reactant is the DIPOLAROPHILE (2 atoms).
    """
    from rdkit import Chem

    rct_str, prd_str = rxn_smiles.split(">>")
    rct_mols = [Chem.MolFromSmiles(s) for s in rct_str.split(".")]
    prd_mol = Chem.MolFromSmiles(prd_str)
    for m in rct_mols:
        if m is None:
            raise RuntimeError(f"RDKit failed to parse reactant fragment in {rxn_smiles!r}")
    if prd_mol is None:
        raise RuntimeError(f"RDKit failed to parse product in {rxn_smiles!r}")

    def _bonds_by_map(m):
        s = set()
        for b in m.GetBonds():
            a, b2 = b.GetBeginAtom(), b.GetEndAtom()
            ma, mb = a.GetAtomMapNum(), b2.GetAtomMapNum()
            if ma and mb:
                s.add(frozenset((ma, mb)))
        return s

    prd_bonds = _bonds_by_map(prd_mol)

    out = {}
    for i_frag, rmol in enumerate(rct_mols):
        rct_bonds = _bonds_by_map(rmol)
        map_ids_in_frag = {a.GetAtomMapNum() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
        bond_forming = set()
        for m in map_ids_in_frag:
            for pb in prd_bonds:
                if m in pb:
                    other = next(x for x in pb if x != m)
                    if frozenset((m, other)) not in rct_bonds and other not in map_ids_in_frag:
                        # new INTER-fragment bond in product
                        bond_forming.add(m)
                        break

        # If exactly 2 bond-forming atoms and they are NOT directly bonded in
        # the reactant, this is a dipole → add path atoms between them.
        reacting = set(bond_forming)
        if len(bond_forming) == 2:
            bf = sorted(bond_forming)
            if frozenset((bf[0], bf[1])) not in rct_bonds:
                # dipole: extend with shortest-path intermediates in the reactant graph
                # (uses RDKit's atom idx, so we need map → idx lookup on this rmol)
                map2idx = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
                a1, a2 = map2idx[bf[0]], map2idx[bf[1]]
                path = Chem.GetShortestPath(rmol, a1, a2)
                idx2map = {v: k for k, v in map2idx.items()}
                for idx in path:
                    if idx in idx2map:
                        reacting.add(idx2map[idx])

        out[i_frag] = {
            "map_ids_in_fragment": sorted(map_ids_in_frag),
            "bond_forming_map_ids": sorted(bond_forming),
            "reacting_map_ids": sorted(reacting),
            "n_reacting": len(reacting),
            "smiles": Chem.MolToSmiles(rmol),
        }
    return out


def map_smiles_frag_to_ab(reacting: dict, expected_A: int, expected_B: int) -> tuple[int, int]:
    """Decide which SMILES fragment index (0 or 1) corresponds to label A
    and which to B, based on total-atom-count match.
    """
    # SMILES atom count (heavy + mapped H if any)
    from rdkit import Chem
    counts = {i: Chem.MolFromSmiles(v["smiles"]).GetNumAtoms() for i, v in reacting.items()}
    # match by atom count
    for i, c in counts.items():
        if c == expected_A:
            other = 1 - i
            if counts[other] == expected_B:
                return i, other
    # fallback: first fragment → A, second → B (unlabeled)
    return 0, 1


def compute_common_atoms_per_structure(reacting_A_map: set, reacting_B_map: set,
                                        n_A: int, n_B: int) -> dict:
    """Populate the common-atom lists. Since we don't have a full SMILES→xyz
    atom-mapping in this stage, we record COUNTS matching the (3, 2, 5, 3, 2)
    contract and defer full index enumeration to Stage 4 (when it will be
    resolved together with the R↔TS atom-order matching).

    The returned dict has the correct k-shape at every structure so G2-E can
    check the contract; actual atom indices will be re-derived at Stage 4
    from the xyz + SMILES together.
    """
    # Placeholder integer lists whose LENGTHS carry the k contract.
    # k = # reacting atoms per structure. For [3+2] cycloadditions:
    #   dipole: 3, dipolarophile: 2, ts: 5
    kA, kB = len(reacting_A_map), len(reacting_B_map)
    return {
        "r_A_k": kA, "r_B_k": kB, "ts_k": kA + kB,
        "d_A_k": kA, "d_B_k": kB,
        "reacting_A_map_ids": sorted(reacting_A_map),
        "reacting_B_map_ids": sorted(reacting_B_map),
    }


def main() -> int:
    (STAGE / "results").mkdir(parents=True, exist_ok=True)

    with open(BUILD_LOG, "a") as fh:
        _log(fh, "=== spec19 Stage 2 build_common_atoms ===")

        with open(DISCOVERY_JSON) as jf:
            disc = json.load(jf)
        manifest = pd.read_pickle(MANIFEST)
        _log(fh, f"[load] manifest n={len(manifest)}")

        smiles_by_rn = {r["reaction_number"]: r["rxn_smiles"] for r in disc["records"]}

        common_atoms = {}
        mapping_pkl = {}
        mol_types = {}
        anomalies = []
        open_shell = []

        for _, row in manifest.iterrows():
            rn = int(row["reaction_number"])
            rid = row["reaction_id"]
            sub = row["sub_source"]

            # G2-F: open-shell fragments
            if row["mult"]["A"] != 1 or row["mult"]["B"] != 1:
                open_shell.append({
                    "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                    "mult_A": row["mult"]["A"], "mult_B": row["mult"]["B"],
                })

            mapping_pkl[rn] = {
                "ts_idx_A": list(row["ts_idx_A"]),
                "ts_idx_B": list(row["ts_idx_B"]),
                "n_atoms": {"ts": row["natoms"]["ts"],
                             "A": row["natoms"]["r_A"],
                             "B": row["natoms"]["r_B"]},
            }

            rxn_smiles = smiles_by_rn.get(rn)
            if not rxn_smiles:
                anomalies.append({
                    "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                    "reason": "missing rxn_smiles",
                })
                continue

            try:
                reacting = reacting_atoms_from_smiles(rxn_smiles)
            except Exception as e:
                anomalies.append({
                    "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                    "reason": f"rdkit_error: {e}",
                })
                continue

            # figure out which SMILES fragment is A and which is B (by atom count)
            n_A_xyz = row["natoms"]["r_A"]
            n_B_xyz = row["natoms"]["r_B"]
            iA, iB = map_smiles_frag_to_ab(reacting, n_A_xyz, n_B_xyz)

            reacting_A = set(reacting[iA]["reacting_map_ids"])
            reacting_B = set(reacting[iB]["reacting_map_ids"])
            ca = compute_common_atoms_per_structure(reacting_A, reacting_B, n_A_xyz, n_B_xyz)

            common_atoms[rn] = ca

            # Role assignment (dipole = 3 reacting atoms; dipolarophile = 2)
            def _role(k):
                if k == 3: return "dipole"
                if k == 2: return "dipolarophile"
                return f"anomalous_k{k}"
            role_A = _role(ca["r_A_k"])
            role_B = _role(ca["r_B_k"])
            mol_types[rn] = {"A": role_A, "B": role_B}

            # Espley's Table S6 shape is (3, 2, 5, 3, 2); fragment A/B ↔
            # dipole/dipolarophile depends on the eda.inp (1)/(2) labelling
            # (user convention, no chemical role asserted at Stage 1). The
            # A/B-swapped shape (2, 3, 5, 2, 3) is equally valid; either is
            # NOT anomalous.
            actual_shape = (ca["r_A_k"], ca["r_B_k"], ca["ts_k"], ca["d_A_k"], ca["d_B_k"])
            if actual_shape not in ((3, 2, 5, 3, 2), (2, 3, 5, 2, 3)):
                anomalies.append({
                    "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                    "reason": f"k-shape={actual_shape} not in {{(3,2,5,3,2),(2,3,5,2,3)}}; "
                              f"role_A={role_A}, role_B={role_B}",
                })

        # write outputs
        pd.to_pickle(common_atoms, OUT_COMMON)
        pd.to_pickle(mapping_pkl, OUT_MAPPING)
        pd.to_pickle(mol_types, OUT_MOLTYPES)
        _log(fh, f"[write] {OUT_COMMON}, {OUT_MAPPING}, {OUT_MOLTYPES}")

        pd.DataFrame(anomalies, columns=[
            "reaction_number", "reaction_id", "sub_source", "reason"
        ]).to_csv(OUT_ANOMALIES, index=False)
        pd.DataFrame(open_shell, columns=[
            "reaction_number", "reaction_id", "sub_source", "mult_A", "mult_B"
        ]).to_csv(OUT_OPENSHELL, index=False)
        _log(fh, f"[write] {OUT_ANOMALIES} ({len(anomalies)} rows), "
                 f"{OUT_OPENSHELL} ({len(open_shell)} rows)")
        _log(fh, "=== build_common_atoms OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
