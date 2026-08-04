#!/usr/bin/env python3
"""
Extract xyz geometries from ORCA .inp files (spec23_paper_setup) into clean .xyz
files ready for Gaussian AM1 input generation.

Two orthogonal naming inconsistencies exist in the ORCA workspace:

  (1) ORCA filename swap: for ~49 rxns, spe_R_A.inp actually holds FRAG2 atoms
      while spe_R_B.inp holds FRAG1 atoms. We use spe_dA / spe_dB (correctly tagged
      via eda.inp fragment tags) as the ground truth and pair each relaxed R_
      file with its matching distorted d_ file by element sequence.

  (2) Package A/B swap: for the same/different ~49 rxns, the 5-atom package's
      fragment_A (ts_idx_A) may correspond to ORCA's FRAG1 or FRAG2. We resolve
      this by matching ORCA eda.inp (1)-tagged atom indices vs pkg's ts_idx_A.

After both resolutions, this bundle adopts a canonical convention:

  reactant_1 = ORCA eda.inp FRAG1 (tag (1) atoms), relaxed = spe_R_X matching spe_dA
  reactant_2 = ORCA eda.inp FRAG2 (tag (2) atoms), relaxed = spe_R_Y matching spe_dB
  reactant_1_type = 'dipole' or 'dipolarophile' (from pkg cross-reference)
  reactant_2_type = the other

Output:
  geometries/dipolar_XXXXXX/
    reactant_1.xyz              relaxed FRAG1
    reactant_2.xyz              relaxed FRAG2
    reactant_1_distorted.xyz    FRAG1 @ TS geom
    reactant_2_distorted.xyz    FRAG2 @ TS geom
    TS.xyz                      TS complex (frag tags stripped)
    meta.json                   full mapping details
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RE_XYZ_LINE = re.compile(
    r"^\s*([A-Za-z]{1,2})(?:\s*\((\d+)\))?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def parse_inp_xyz(inp_path: Path) -> tuple[list[tuple[str, tuple[float, float, float], int | None]], int, int]:
    lines = inp_path.read_text().splitlines()
    in_block = False
    atoms = []
    charge, mult = 0, 1
    for line in lines:
        s = line.strip()
        if s.startswith("* xyz"):
            parts = s.split()
            charge = int(parts[2])
            mult = int(parts[3])
            in_block = True
            continue
        if in_block and s.startswith("*"):
            break
        if in_block:
            m = RE_XYZ_LINE.match(line)
            if m:
                elem = m.group(1)
                tag = int(m.group(2)) if m.group(2) else None
                x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                atoms.append((elem, (x, y, z), tag))
    if not atoms:
        raise ValueError(f"no atoms parsed from {inp_path}")
    return atoms, charge, mult


def write_xyz(atoms: list[tuple[str, tuple[float, float, float], int | None]],
              out_path: Path, comment: str = "") -> None:
    lines = [str(len(atoms)), comment]
    for elem, (x, y, z), _ in atoms:
        lines.append(f"{elem:2s}  {x:14.7f}  {y:14.7f}  {z:14.7f}")
    out_path.write_text("\n".join(lines) + "\n")


def match_fragment(candidate_atoms: list, target_atoms: list) -> bool:
    """Return True if candidate_atoms and target_atoms are the same fragment.

    Match by (a) same atom count and (b) same element multiset (Counter).
    """
    from collections import Counter
    if len(candidate_atoms) != len(target_atoms):
        return False
    c_elem = Counter(a[0] for a in candidate_atoms)
    t_elem = Counter(a[0] for a in target_atoms)
    return c_elem == t_elem


def _kabsch(P, Q):
    """Return rotation matrix best-aligning P → Q (both centered)."""
    import numpy as np
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def reorder_relaxed_to_distorted(
    relaxed_atoms: list, distorted_atoms: list,
) -> tuple[list, list[int]]:
    """Reorder `relaxed_atoms` (element,coords,tag) to match `distorted_atoms`
    element-sequence order via Kabsch alignment + greedy nearest-same-element.

    Returns:
      reordered_atoms: relaxed_atoms in new order matching distorted_atoms
      perm: perm[i] = original relaxed_atoms index that becomes new position i

    Both inputs must be same fragment (same element multiset, verified upstream).
    Relaxed and distorted may have different atom sequence and coordinates.
    """
    import numpy as np
    from collections import defaultdict
    n = len(distorted_atoms)
    assert n == len(relaxed_atoms)

    r_elems = [a[0] for a in relaxed_atoms]
    d_elems = [a[0] for a in distorted_atoms]
    r_coords = np.array([a[1] for a in relaxed_atoms])
    d_coords = np.array([a[1] for a in distorted_atoms])

    r_by_e: dict[str, list[int]] = defaultdict(list)
    d_by_e: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(r_elems):
        r_by_e[e].append(i)
    for i, e in enumerate(d_elems):
        d_by_e[e].append(i)

    # Initial correspondence: k-th same-element atom in r → k-th in d
    init = {d_idx: r_by_e[e][k] for e in d_by_e for k, d_idx in enumerate(d_by_e[e])}
    r_reordered_init = np.stack([r_coords[init[i]] for i in range(n)])

    # Kabsch align r_init → d
    r_c = r_reordered_init - r_reordered_init.mean(axis=0)
    d_c = d_coords - d_coords.mean(axis=0)
    R = _kabsch(r_c, d_c)
    r_aligned_all = (r_coords - r_reordered_init.mean(axis=0)) @ R + d_coords.mean(axis=0)

    # Greedy nearest-same-element to refine
    used: set[int] = set()
    perm: list[int] = []
    for d_idx in range(n):
        e = d_elems[d_idx]
        cands = [i for i in r_by_e[e] if i not in used]
        best = min(cands, key=lambda i: float(np.linalg.norm(r_aligned_all[i] - d_coords[d_idx])))
        perm.append(best)
        used.add(best)

    reordered = [relaxed_atoms[p] for p in perm]
    return reordered, perm


def process_rxn(rid: str, workdir: Path, out_root: Path, frag: dict, picks: dict) -> dict:
    rxn_dir = workdir / rid
    out_dir = out_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 5 ORCA .inp files
    atoms_R_A, _, _ = parse_inp_xyz(rxn_dir / "spe_R_A" / "spe_R_A.inp")
    atoms_R_B, _, _ = parse_inp_xyz(rxn_dir / "spe_R_B" / "spe_R_B.inp")
    atoms_dA,  _, _ = parse_inp_xyz(rxn_dir / "cpcm_extra" / "spe_dA.inp")
    atoms_dB,  _, _ = parse_inp_xyz(rxn_dir / "cpcm_extra" / "spe_dB.inp")
    atoms_TS, ts_charge, ts_mult = parse_inp_xyz(rxn_dir / "eda" / "eda.inp")

    # Verify eda.inp has fragment tags
    orca_frag1_ts_idx = [i for i, (_e, _c, tag) in enumerate(atoms_TS) if tag == 1]
    orca_frag2_ts_idx = [i for i, (_e, _c, tag) in enumerate(atoms_TS) if tag == 2]
    if not orca_frag1_ts_idx or not orca_frag2_ts_idx:
        raise ValueError(f"{rid}: eda.inp missing (1)/(2) fragment tags")

    # STEP 1: pair spe_R_* with spe_d* by element-sequence match.
    #         spe_dA is FRAG1 (from eda.inp tag), spe_dB is FRAG2 (established convention).
    #         Determine which relaxed file matches spe_dA.
    if match_fragment(atoms_R_A, atoms_dA):
        frag1_relaxed = atoms_R_A
        frag2_relaxed = atoms_R_B
        orca_filename_swap = "no_swap"
        # verify frag2 relaxed matches spe_dB
        if not match_fragment(atoms_R_B, atoms_dB):
            raise ValueError(f"{rid}: spe_R_B elements don't match spe_dB")
    elif match_fragment(atoms_R_B, atoms_dA):
        frag1_relaxed = atoms_R_B
        frag2_relaxed = atoms_R_A
        orca_filename_swap = "swap"
        # verify frag2 relaxed matches spe_dB
        if not match_fragment(atoms_R_A, atoms_dB):
            raise ValueError(f"{rid}: spe_R_A elements don't match spe_dB")
    else:
        raise ValueError(
            f"{rid}: neither spe_R_A nor spe_R_B matches spe_dA by element sequence"
        )
    frag1_distorted = atoms_dA
    frag2_distorted = atoms_dB

    # STEP 2: determine FRAG1 type from USER's 5-atom picks (authoritative)
    #
    # pkg fragment_assignment.json's type_A/type_B is auto-derived (SMILES/RDKit)
    # and disagrees with user's manual review for 212/400 rxns. The user's di_ts
    # atoms (indexed into TS complex) ARE the dipole atoms by definition, so the
    # fragment containing them IS the dipole fragment.
    pkg_A_ts_idx = frag[rid]["ts_idx_A"]
    pkg_B_ts_idx = frag[rid]["ts_idx_B"]

    if sorted(orca_frag1_ts_idx) == sorted(pkg_A_ts_idx):
        pkg_swap = "no_swap"
    elif sorted(orca_frag1_ts_idx) == sorted(pkg_B_ts_idx):
        pkg_swap = "swap"
    else:
        raise ValueError(
            f"{rid}: eda.inp FRAG1 atoms {sorted(orca_frag1_ts_idx)} match neither "
            f"pkg_A {sorted(pkg_A_ts_idx)} nor pkg_B {sorted(pkg_B_ts_idx)}"
        )

    # Derive frag types from user picks: whichever fragment contains di_ts atoms IS dipole
    user_pick = picks[rid]
    di_ts_set = set(user_pick["di_ts"])
    dp_ts_set = set(user_pick["dp_ts"])
    frag1_set = set(orca_frag1_ts_idx)
    frag2_set = set(orca_frag2_ts_idx)

    if di_ts_set.issubset(frag1_set) and dp_ts_set.issubset(frag2_set):
        frag1_type, frag2_type = "dipole", "dipolarophile"
        user_di_in = "frag1"
    elif di_ts_set.issubset(frag2_set) and dp_ts_set.issubset(frag1_set):
        frag1_type, frag2_type = "dipolarophile", "dipole"
        user_di_in = "frag2"
    else:
        raise ValueError(
            f"{rid}: user di_ts {sorted(di_ts_set)} + dp_ts {sorted(dp_ts_set)} do not "
            f"partition cleanly across frag1 {sorted(frag1_set)} / frag2 {sorted(frag2_set)}"
        )

    # Cross-check with pkg auto type (for audit; user is authoritative)
    pkg_type_A = frag[rid]["type_A"]
    pkg_type_B = frag[rid]["type_B"]
    if pkg_swap == "no_swap":
        pkg_frag1_type, pkg_frag2_type = pkg_type_A, pkg_type_B
    else:
        pkg_frag1_type, pkg_frag2_type = pkg_type_B, pkg_type_A
    user_pkg_agree = (pkg_frag1_type == frag1_type)

    # Reorder relaxed reactants to match distorted (=TS-subset) atom order.
    # This makes reactant_i.xyz and reactant_i_distorted.xyz atom-index aligned,
    # so features/mapping downstream can use identity indexing.
    frag1_relaxed_reordered, perm1 = reorder_relaxed_to_distorted(frag1_relaxed, frag1_distorted)
    frag2_relaxed_reordered, perm2 = reorder_relaxed_to_distorted(frag2_relaxed, frag2_distorted)

    # Write xyz files (canonical reactant_1/2 = ORCA FRAG1/2, in d order)
    write_xyz(frag1_relaxed_reordered, out_dir / "reactant_1.xyz",           f"{rid} reactant_1 (FRAG1 relaxed, reordered to d) type={frag1_type}")
    write_xyz(frag2_relaxed_reordered, out_dir / "reactant_2.xyz",           f"{rid} reactant_2 (FRAG2 relaxed, reordered to d) type={frag2_type}")
    write_xyz(frag1_distorted,         out_dir / "reactant_1_distorted.xyz", f"{rid} reactant_1 @ TS geom")
    write_xyz(frag2_distorted,         out_dir / "reactant_2_distorted.xyz", f"{rid} reactant_2 @ TS geom")
    write_xyz(atoms_TS,                out_dir / "TS.xyz",                   f"{rid} TS complex (tags stripped)")

    meta = {
        "reaction_id": rid,
        "reaction_number": frag[rid]["reaction_number"],
        "sub_source": frag[rid]["sub_source"],
        "n_atoms_ts": len(atoms_TS),
        "n_atoms_reactant_1": len(frag1_relaxed),
        "n_atoms_reactant_2": len(frag2_relaxed),
        "orca_frag1_ts_idx_0based": orca_frag1_ts_idx,
        "orca_frag2_ts_idx_0based": orca_frag2_ts_idx,
        # AUTHORITATIVE: derived from user's manual 5-atom picks
        "reactant_1_type": frag1_type,           # from user di_ts / dp_ts location
        "reactant_2_type": frag2_type,
        "user_di_in_reactant": user_di_in,       # 'frag1' or 'frag2'
        # audit: file naming swaps
        "orca_filename_swap": orca_filename_swap,
        "pkg_swap_status": pkg_swap,
        # audit: pkg auto-derived types (may disagree with user)
        "pkg_A_ts_idx_0based": pkg_A_ts_idx,
        "pkg_B_ts_idx_0based": pkg_B_ts_idx,
        "pkg_type_A": pkg_type_A,
        "pkg_type_B": pkg_type_B,
        "pkg_frag1_type_auto": pkg_frag1_type,
        "pkg_frag2_type_auto": pkg_frag2_type,
        "user_agrees_with_pkg_auto": user_pkg_agree,   # False for 42% of rxns
        # user picks (5-atom labels, TS 0-based indices)
        "user_di_ts_0based": user_pick["di_ts"],
        "user_dp_ts_0based": user_pick["dp_ts"],
        "user_di_local_in_r_0based": user_pick["di_local_in_r"],
        "user_dp_local_in_r_0based": user_pick["dp_local_in_r"],
        "user_is_A_dipole": user_pick["is_A_dipole"],
        "charge_total": ts_charge,
        "mult_total": ts_mult,
        # atom reordering: relaxed reactant xyz was reordered to match distorted
        "reactant_1_perm_orig_to_new": perm1,   # perm[i]=j means new pos i = original pos j
        "reactant_2_perm_orig_to_new": perm2,
        "source_R_1": str(rxn_dir / ("spe_R_A" if orca_filename_swap == "no_swap" else "spe_R_B") / (f"spe_R_A.inp" if orca_filename_swap == "no_swap" else "spe_R_B.inp")),
        "source_R_2": str(rxn_dir / ("spe_R_B" if orca_filename_swap == "no_swap" else "spe_R_A") / (f"spe_R_B.inp" if orca_filename_swap == "no_swap" else "spe_R_A.inp")),
        "source_d_1": str(rxn_dir / "cpcm_extra" / "spe_dA.inp"),
        "source_d_2": str(rxn_dir / "cpcm_extra" / "spe_dB.inp"),
        "source_TS":  str(rxn_dir / "eda" / "eda.inp"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "rid": rid,
        "orca_swap": orca_filename_swap,
        "pkg_swap": pkg_swap,
        "user_pkg_agree": user_pkg_agree,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path,
                    default=Path("/gpfs/tmp_cpu2/yeseo1ee/spec23_paper_setup"))
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent.parent / "MANIFEST.txt")
    ap.add_argument("--fragment-json", type=Path,
                    default=Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/400_rxns_5_atoms_labelling_dataset/fragment_assignment.json"))
    ap.add_argument("--picks-json", type=Path,
                    default=Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/400_rxns_5_atoms_labelling_dataset/5_atom_picks.json"))
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent.parent / "geometries")
    args = ap.parse_args()

    frag = json.loads(args.fragment_json.read_text())
    picks = json.loads(args.picks_json.read_text())

    rxns: list[str] = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rxns.append(parts[1])

    print(f"processing {len(rxns)} rxns → {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    processed = []
    fails = []
    for rid in rxns:
        try:
            r = process_rxn(rid, args.workdir, args.out_root, frag, picks)
            processed.append(r)
        except Exception as e:
            fails.append((rid, str(e)))
            print(f"  FAIL {rid}: {e}")

    if fails:
        print(f"\n{len(fails)} FAIL(s), aborting")
        return 1

    n_orca_swap = sum(1 for r in processed if r["orca_swap"] == "swap")
    n_pkg_swap = sum(1 for r in processed if r["pkg_swap"] == "swap")
    n_user_disagree = sum(1 for r in processed if not r["user_pkg_agree"])
    print(f"\nprocessed: {len(processed)} rxns")
    print(f"  ORCA filename swap (spe_R_A holds FRAG2): {n_orca_swap}")
    print(f"  pkg swap (eda FRAG1 == pkg fragB): {n_pkg_swap}")
    print(f"  user disagrees with pkg auto type: {n_user_disagree}")

    xyz_files = list(args.out_root.rglob("*.xyz"))
    meta_files = list(args.out_root.rglob("meta.json"))
    print(f"\n  xyz files: {len(xyz_files)}  (expect {len(processed) * 5})")
    print(f"  meta.json: {len(meta_files)}  (expect {len(processed)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
