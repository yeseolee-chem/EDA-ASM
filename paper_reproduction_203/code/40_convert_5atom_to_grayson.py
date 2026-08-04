#!/usr/bin/env python3
"""
Convert 5-atom labeling + geometry meta to Grayson f_extract.py's expected 3 pkl files.

Grayson expects (verified via reading grayson_code/feature_extraction/f_extract.py):

  common_atoms.pkl
    { '<rn>': {
        'di':       [3 dipole atom indices, 1-based, in TS complex numbering],
        'dp':       [2 dipolarophile atom indices, 1-based],
        'reacting': [di[0], dp[0], di[2], dp[1]]  (paired forming bonds, 1-based),
        'name':     'ts_<rn>' (base filename without .out),
      }
      ... }

  mapping.pkl
    { '<rn>': {
        'reactant_1': {ts_atom_1based: r_atom_1based, ...},
        'reactant_2': {ts_atom_1based: r_atom_1based, ...},
      }
      ... }
    Only the 5 reacting atoms need mapping entries — Grayson's f_extract only
    loops over di+dp when using this dict.

  mol_types.pkl
    { '<rn>': {
        'reactant_1': 'di' or 'dp',
        'reactant_2': 'di' or 'dp',
      }
      ... }

Convention decisions:

  - reactant_1 = ORCA FRAG1 atoms (from eda.inp (1) tags), reactant_2 = FRAG2.
  - reactant_1_type ('di'/'dp') taken from user's 5-atom picks (authoritative,
    disagrees with pkg auto for 86/203 rxns — see 20_prepare_geometries.py).
  - Reacting bond order: for each di atom, pair with a dp atom by nearest-distance
    in TS complex → [di[0], dp[0'], di[2], dp[1']] where 0'/1' are chosen
    to give the two forming bonds.

Indices are 0-based internally, +1 shifted to 1-based on write (Grayson convention).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _to_1based(lst: list[int]) -> list[int]:
    return [int(i) + 1 for i in lst]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.asarray(coords)


def compute_reacting_pairs(di_ts_0: list[int], dp_ts_0: list[int],
                            ts_coords: np.ndarray) -> list[int]:
    """Pair terminal di atoms with dp atoms by nearest-neighbour distance.

    For [3+2] cycloaddition, two forming bonds:
      di_terminal_0 --- dp_?
      di_terminal_2 --- dp_?'

    di_ts is [di_0, di_1 (middle), di_2 (other terminal)]. Middle di_1 does not
    form a new bond. Assign di_0 and di_2 to whichever dp atom is closer.

    Return 4-list (1-based) for Grayson: [di_0, dp_0', di_2, dp_1']
    """
    di_0, di_1, di_2 = di_ts_0
    dp_0, dp_1 = dp_ts_0

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(ts_coords[a] - ts_coords[b]))

    # try both assignments; pick the one with smaller total distance
    ass1 = dist(di_0, dp_0) + dist(di_2, dp_1)
    ass2 = dist(di_0, dp_1) + dist(di_2, dp_0)

    if ass1 <= ass2:
        return _to_1based([di_0, dp_0, di_2, dp_1])
    else:
        return _to_1based([di_0, dp_1, di_2, dp_0])


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return rotation matrix that best aligns P → Q (both centered)."""
    H = P.T @ Q
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def match_atoms_r_to_d(r_elems: list[str], r_coords: np.ndarray,
                        d_elems: list[str], d_coords: np.ndarray) -> dict[int, int]:
    """Return dict {d_idx: r_idx} pairing each distorted-fragment atom to
    the corresponding relaxed-reactant atom.

    Steps:
      1. Sanity check element multiset match.
      2. Initial guess: k-th same-element atom in r maps to k-th in d.
      3. Center both, Kabsch-rotate r → d using initial mapping.
      4. Refine: greedy nearest-same-element assignment on aligned r.
    """
    from collections import defaultdict
    r_by_e: dict[str, list[int]] = defaultdict(list)
    d_by_e: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(r_elems):
        r_by_e[e].append(i)
    for i, e in enumerate(d_elems):
        d_by_e[e].append(i)
    if set(r_by_e.keys()) != set(d_by_e.keys()):
        raise ValueError(f"element sets differ: r={set(r_by_e.keys())} d={set(d_by_e.keys())}")
    for e in d_by_e:
        if len(r_by_e[e]) != len(d_by_e[e]):
            raise ValueError(f"element {e} count differs: r={len(r_by_e[e])} d={len(d_by_e[e])}")

    n = len(d_elems)
    if n != len(r_elems):
        raise ValueError("atom count differs")

    # Initial correspondence: k-th same-elem atom in r to k-th in d
    init = {d_idx: r_by_e[e][k] for e in d_by_e for k, d_idx in enumerate(d_by_e[e])}

    # Center coords for Kabsch (use init mapping to compute centroids)
    r_ordered = np.stack([r_coords[init[d_idx]] for d_idx in range(n)])
    d_centered = d_coords - d_coords.mean(axis=0)
    r_centered = r_ordered - r_ordered.mean(axis=0)
    R = _kabsch(r_centered, d_centered)
    r_aligned = (r_coords - r_ordered.mean(axis=0)) @ R + d_coords.mean(axis=0)

    # Greedy nearest-same-element refinement
    used_r: set[int] = set()
    refined: dict[int, int] = {}
    # Process d atoms in a deterministic order (by index)
    for d_idx in range(n):
        e = d_elems[d_idx]
        candidates = [i for i in r_by_e[e] if i not in used_r]
        best = min(candidates,
                    key=lambda i: float(np.linalg.norm(r_aligned[i] - d_coords[d_idx])))
        refined[d_idx] = best
        used_r.add(best)
    return refined


def build_mapping_for_rxn(meta: dict, ts_coords: np.ndarray, ts_elems: list[str],
                           r1_elems: list[str], r2_elems: list[str],
                           r1_coords: np.ndarray, r2_coords: np.ndarray,
                           d1_elems: list[str], d2_elems: list[str],
                           d1_coords: np.ndarray, d2_coords: np.ndarray) -> dict:
    """Build ts→reactant atom mapping for the 5 reacting atoms.

    Uses match_atoms_r_to_d to determine correspondence between each relaxed
    reactant xyz (Stuyver's original atom order) and its distorted xyz
    (TS-subset order). Then for a reacting TS atom, its distorted-fragment
    position is trivially the TS-subset index, and the mapping table gives
    the relaxed-reactant index.
    """
    frag1_ts_idx = meta["orca_frag1_ts_idx_0based"]
    frag2_ts_idx = meta["orca_frag2_ts_idx_0based"]

    # reactant_1.xyz and reactant_1_distorted.xyz now have matching atom order
    # (via reordering in 20_prepare_geometries.py). So d_pos == r_pos (identity).
    def ts_to_r_atom(ts_atom_0: int, frag_ts_idx: list[int], _unused=None) -> int:
        return frag_ts_idx.index(ts_atom_0)

    di_ts = meta["user_di_ts_0based"]
    dp_ts = meta["user_dp_ts_0based"]

    m1: dict[int, int] = {}   # {ts_1based: r_1based}
    m2: dict[int, int] = {}

    frag1_set = set(frag1_ts_idx)
    for ts_atom in di_ts + dp_ts:
        if ts_atom in frag1_set:
            r_atom = ts_to_r_atom(ts_atom, frag1_ts_idx)
            m1[ts_atom + 1] = r_atom + 1
        else:
            r_atom = ts_to_r_atom(ts_atom, frag2_ts_idx)
            m2[ts_atom + 1] = r_atom + 1

    return {"reactant_1": m1, "reactant_2": m2}


def process_rxn(rid: str, rn: int, geom_dir: Path) -> tuple[dict, dict, dict]:
    """Return (ca_entry, map_entry, moltypes_entry) for one rxn."""
    meta = json.loads((geom_dir / "meta.json").read_text())

    ts_elems, ts_coords = read_xyz(geom_dir / "TS.xyz")
    r1_elems, r1_coords = read_xyz(geom_dir / "reactant_1.xyz")
    r2_elems, r2_coords = read_xyz(geom_dir / "reactant_2.xyz")
    d1_elems, d1_coords = read_xyz(geom_dir / "reactant_1_distorted.xyz")
    d2_elems, d2_coords = read_xyz(geom_dir / "reactant_2_distorted.xyz")

    di_ts_0 = meta["user_di_ts_0based"]
    dp_ts_0 = meta["user_dp_ts_0based"]

    reacting = compute_reacting_pairs(di_ts_0, dp_ts_0, ts_coords)
    ca_entry = {
        "di": _to_1based(di_ts_0),
        "dp": _to_1based(dp_ts_0),
        "reacting": reacting,
        "name": f"ts_{rn}",
    }

    map_entry = build_mapping_for_rxn(meta, ts_coords, ts_elems,
                                        r1_elems, r2_elems, r1_coords, r2_coords,
                                        d1_elems, d2_elems, d1_coords, d2_coords)

    # mol_types entry
    r1_type = meta["reactant_1_type"]
    r2_type = meta["reactant_2_type"]
    to_short = {"dipole": "di", "dipolarophile": "dp"}
    mt_entry = {
        "reactant_1": to_short[r1_type],
        "reactant_2": to_short[r2_type],
    }

    return ca_entry, map_entry, mt_entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent.parent / "MANIFEST.txt")
    ap.add_argument("--geom-root", type=Path,
                    default=Path(__file__).parent.parent / "geometries")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent.parent / "grayson_pkls")
    args = ap.parse_args()

    rxns: list[tuple[int, str]] = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rxns.append((int(parts[0]), parts[1]))
    print(f"building Grayson pkl files for {len(rxns)} rxns")

    common_atoms: dict = {}
    mapping: dict = {}
    mol_types: dict = {}

    fails = []
    for rn, rid in rxns:
        try:
            ca, mp, mt = process_rxn(rid, rn, args.geom_root / rid)
            key = str(rn)
            common_atoms[key] = ca
            mapping[key] = mp
            mol_types[key] = mt
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s), aborting")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "common_atoms.pkl").open("wb").write(pickle.dumps(common_atoms))
    (args.out_dir / "mapping.pkl").open("wb").write(pickle.dumps(mapping))
    (args.out_dir / "mol_types.pkl").open("wb").write(pickle.dumps(mol_types))

    print(f"\nwrote:")
    print(f"  {args.out_dir}/common_atoms.pkl  ({len(common_atoms)} entries)")
    print(f"  {args.out_dir}/mapping.pkl       ({len(mapping)} entries)")
    print(f"  {args.out_dir}/mol_types.pkl     ({len(mol_types)} entries)")

    # Sanity samples
    print(f"\nsample common_atoms['0']:  {common_atoms['0']}")
    print(f"sample mapping['0']:        {mapping['0']}")
    print(f"sample mol_types['0']:      {mol_types['0']}")

    # Sanity: mol_types di/dp count
    from collections import Counter
    r1_types = Counter(v["reactant_1"] for v in mol_types.values())
    r2_types = Counter(v["reactant_2"] for v in mol_types.values())
    print(f"\nreactant_1 types: {dict(r1_types)}")
    print(f"reactant_2 types: {dict(r2_types)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
