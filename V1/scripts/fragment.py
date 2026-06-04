"""Phase 2 — Manual fragmentation per V1 Claisen spec §6 (D2AF unavailable).

Splits each substrate's TS geometry into two allyl-type doublet fragments:
  Fragment A = {C1, C2, O3} + R-group atoms attached to C2 + their Hs
  Fragment B = {C4, C5, C6} + their Hs
The two homolytically cleaved bonds are O3-C4 (breaking) and C1-C6 (forming).

Writes for each substrate:
  runs/<id>/eda/fragments.json   indices + spin / charge metadata
  runs/<id>/eda/frag_A.xyz       fragment A geometry at TS
  runs/<id>/eda/frag_B.xyz       fragment B geometry at TS
  runs/<id>/eda/.done_fragment   sentinel
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"

# Distance cutoff for H-X bond detection in TS geometry. Standard C-H ~1.1 Å,
# X-H ~ 1.0-1.5 Å. 1.30 covers everything except elongated/breaking bonds.
H_BOND_CUTOFF_A = 1.30


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    syms = []
    coords = []
    for L in lines[2 : 2 + n]:
        p = L.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, coords


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def fragment_one(rxn_id: str) -> tuple[str, str]:
    am_path = RUNS / rxn_id / "build" / "atom_map.json"
    ts_path = RUNS / rxn_id / "orca" / "ts.xyz"
    if not ts_path.exists():
        return rxn_id, "skip:no_ts.xyz"
    out_dir = RUNS / rxn_id / "eda"
    sentinel = out_dir / ".done_fragment"
    if sentinel.exists():
        return rxn_id, "skipped"

    am = json.loads(am_path.read_text())
    core = am["core_indices"]
    C1, C2, O3, C4, C5, C6 = (core[k] for k in ("C1", "C2", "O3", "C4", "C5", "C6"))
    r_heavy = set(am["R_all_atom_indices"])  # may be empty for h

    syms, coords = read_xyz(ts_path)
    n = len(syms)

    # Identify H atoms and their nearest heavy attachment by distance.
    h_attach: dict[int, int] = {}
    for i, sym in enumerate(syms):
        if sym != "H":
            continue
        best_j, best_d = None, 1e9
        for j, sj in enumerate(syms):
            if sj == "H" or j == i:
                continue
            d = dist(coords[i], coords[j])
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= H_BOND_CUTOFF_A:
            h_attach[i] = best_j

    # Assemble Fragment A: C1, C2, O3, all R heavy atoms, all Hs attached to them
    A_heavy = {C1, C2, O3} | r_heavy
    A_atoms = set(A_heavy) | {h for h, j in h_attach.items() if j in A_heavy}

    # Fragment B: C4, C5, C6 + their Hs (no R there)
    B_heavy = {C4, C5, C6}
    B_atoms = set(B_heavy) | {h for h, j in h_attach.items() if j in B_heavy}

    # Sanity: A and B must be disjoint and cover all atoms
    overlap = A_atoms & B_atoms
    missing = set(range(n)) - A_atoms - B_atoms
    if overlap:
        return rxn_id, f"failed:atoms_overlap={sorted(overlap)}"
    if missing:
        # H atoms whose nearest heavy was too far — try to assign by graph-BFS
        # but most likely they belong to A (R group H attached to a stretched bond)
        for h in list(missing):
            j = h_attach.get(h)
            if j in A_heavy:
                A_atoms.add(h)
                missing.remove(h)
            elif j in B_heavy:
                B_atoms.add(h)
                missing.remove(h)
        if missing:
            return rxn_id, f"failed:unassigned_atoms={sorted(missing)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    # Write fragment xyzs
    def write_frag(path: Path, idx_set: set[int], label: str) -> None:
        idx = sorted(idx_set)
        path.write_text(
            f"{len(idx)}\n"
            f"frag_{label} of {rxn_id} at TS (atoms: {idx})\n"
            + "\n".join(
                f"{syms[i]:>3s} {coords[i][0]:>14.8f} {coords[i][1]:>14.8f} {coords[i][2]:>14.8f}"
                for i in idx
            )
            + "\n"
        )

    write_frag(out_dir / "frag_A.xyz", A_atoms, "A")
    write_frag(out_dir / "frag_B.xyz", B_atoms, "B")

    # Atoms tally
    elem_A = {}
    elem_B = {}
    for i in A_atoms:
        elem_A[syms[i]] = elem_A.get(syms[i], 0) + 1
    for i in B_atoms:
        elem_B[syms[i]] = elem_B.get(syms[i], 0) + 1

    info = {
        "rxn_id": rxn_id,
        "method": "manual_atom_map_bfs",
        "core_indices": core,
        "R_group_heavy_indices": sorted(r_heavy),
        "cleaved_bonds": [
            {"breaking": [O3, C4], "name": "O3-C4"},
            {"forming": [C1, C6], "name": "C1-C6"},
        ],
        "fragment_A": {
            "label": "C1=C2(R)-O3 allyl-O radical",
            "atom_indices": sorted(A_atoms),
            "natoms": len(A_atoms),
            "elements": elem_A,
            "charge": 0,
            "spin_multiplicity": 2,  # doublet (allyl-type radical)
        },
        "fragment_B": {
            "label": "C4=C5=C6 allyl radical",
            "atom_indices": sorted(B_atoms),
            "natoms": len(B_atoms),
            "elements": elem_B,
            "charge": 0,
            "spin_multiplicity": 2,
        },
        "system": {
            "total_charge": 0,
            "spin_coupling": "broken-symmetry singlet (Sz=0)",
        },
    }
    (out_dir / "fragments.json").write_text(json.dumps(info, indent=2))

    sentinel.touch()
    return rxn_id, f"ok (A:{len(A_atoms)}atoms B:{len(B_atoms)}atoms)"


def main() -> int:
    ids = ["h", "nme2", "nh2", "oh", "ome", "me", "ph", "f", "i", "br", "cl",
           "cf3", "ac", "cn", "no2"]
    print(f"fragmenting {len(ids)} substrates")
    results = []
    for rxn in ids:
        rxn_id, status = fragment_one(rxn)
        results.append((rxn_id, status))
        print(f"  [{rxn_id:6s}] {status}")
    failed = [r for r, s in results if s.startswith("failed")]
    skipped_noTS = [r for r, s in results if s.startswith("skip:no_ts")]
    print()
    print(f"summary: ok={sum(1 for _,s in results if s.startswith('ok'))} "
          f"skipped={sum(1 for _,s in results if s=='skipped')} "
          f"no_ts={len(skipped_noTS)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
