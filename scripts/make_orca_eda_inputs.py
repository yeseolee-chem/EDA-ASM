"""Generate ORCA 6.1 EDA-NOCV inputs for 789 reactions from manual_partitions.json.

Correct EDA-NOCV syntax (per ORCA 6.1 manual, verified via web docs):

  ! BLYP D3BJ def2-TZVP NoSym EDA TightSCF
  %pal nprocs 8 end
  %maxcore 3500

  %eda
    FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF"
    FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF"
    FRAG1_C 0
    FRAG1_M 1
    FRAG2_C 0
    FRAG2_M 1
  end

  * xyz 0 1
  C(1)  x y z         ← inline fragment tag (1) or (2)
  H(2)  x y z
  ...
  *

Fragment atom identity comes from user's manual_partitions.json (R-based),
translated to TS-native indices via family-aware R→TS mapping:
  - rgd1: same atom order as TS → indices used directly
  - qmrxn20: nearest-element-neighbour R→TS mapping
  - dipolar: use auto SMILES-based TS-native partition (matches user's identity)

TS geometry (physical, real coordinates) is used for the ORCA input.

Emits per-reaction directory:
  outputs/orca_eda/inputs/<rid>/eda.inp
  outputs/orca_eda/inputs/<rid>/meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"
_V6 = REPO / "labels/adf/adf_labels_v6_multifamily.parquet"
LABELS_PQ = _V7 if _V7.exists() else _V6
MANUAL_PART = REPO / "outputs/frag_review/manual_partitions.json"
AUTO_PART = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
OUT_ROOT = REPO / "outputs/orca_eda/inputs"


def load_ts(rid: str):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    z = np.asarray(d["TS"]["z"], dtype=int)
    pos = np.asarray(d["TS"]["pos"], dtype=float)
    return z, pos


def _ts_two_fragments(z_TS, pos_TS, mult_max=1.20, mult_min=0.60, step=0.05):
    """Sweep cutoff on TS geometry; return (comp1_indices, comp2_indices) at
    the LARGEST mult that yields exactly 2 connected components. This finds
    the natural chemical split at TS (where reactants are close but each
    fragment is internally connected)."""
    import ase, networkx as nx
    from ase.neighborlist import build_neighbor_list, natural_cutoffs
    m = mult_max
    while m >= mult_min - 1e-9:
        atoms = ase.Atoms(numbers=[int(x) for x in z_TS], positions=pos_TS)
        cutoffs = [c * m for c in natural_cutoffs(atoms)]
        nl = build_neighbor_list(atoms, cutoffs, self_interaction=False, bothways=True)
        g = nx.Graph(); g.add_nodes_from(range(len(z_TS)))
        for i in range(len(z_TS)):
            for j in nl.get_neighbors(i)[0]: g.add_edge(i, int(j))
        comps = [sorted(c) for c in nx.connected_components(g)]
        if len(comps) == 2:
            comps.sort(key=len, reverse=True)
            return comps[0], comps[1], float(m)
        m -= step
    return None


def _match_user_identity_to_ts_split(z_TS, ts_split, z_R, user_A_R, user_B_R):
    """Given TS 2-fragment split and user's R-based fragments, decide which
    TS component is 'A' and which is 'B' by matching element compositions."""
    from collections import Counter
    c1, c2 = ts_split
    z_c1 = Counter(int(z_TS[i]) for i in c1)
    z_c2 = Counter(int(z_TS[i]) for i in c2)
    z_uA = Counter(int(z_R[i]) for i in user_A_R)
    z_uB = Counter(int(z_R[i]) for i in user_B_R)
    if z_c1 == z_uA and z_c2 == z_uB:
        return sorted(c1), sorted(c2)
    if z_c1 == z_uB and z_c2 == z_uA:
        return sorted(c2), sorted(c1)
    return None  # composition mismatch


def _element_sequence_map(z_src, z_tgt):
    """Legacy fallback (position-free element-sequence matching)."""
    used = [False] * len(z_tgt)
    out = [-1] * len(z_src)
    for i in range(len(z_src)):
        zi = int(z_src[i])
        for j in range(len(z_tgt)):
            if not used[j] and int(z_tgt[j]) == zi:
                out[i] = j; used[j] = True; break
    return out


def _greedy_growth_from_seed(z_TS, pos_TS, z_R, user_A_R, user_B_R):
    """Given user's fragment element composition, find a spatially coherent
    TS-atom assignment via BFS-style growth from single-atom seeds.

    Seeds: element that appears in only one user fragment (unique element).
    Growth: at each step, add the nearest unassigned atom to whichever side
    still needs its element (per composition quota).
    """
    from collections import Counter
    from scipy.spatial.distance import pdist, squareform
    n = len(z_TS)
    D = squareform(pdist(pos_TS))

    quotaA = Counter(int(z_R[i]) for i in user_A_R)
    quotaB = Counter(int(z_R[i]) for i in user_B_R)
    if sum(quotaA.values()) + sum(quotaB.values()) != n:
        return None
    if Counter(int(z) for z in z_TS) != quotaA + quotaB:
        return None

    only_A = [e for e in quotaA if e not in quotaB]
    only_B = [e for e in quotaB if e not in quotaA]
    if not only_A or not only_B:
        # No unique element seed → use greedy from most-abundant-diff element
        return None

    def by_dist_from_center(elem):
        cands = [i for i in range(n) if int(z_TS[i]) == elem]
        if not cands: return -1
        center = pos_TS.mean(axis=0)
        cands.sort(key=lambda i: -np.linalg.norm(pos_TS[i] - center))  # farthest first
        return cands[0]

    seedA = by_dist_from_center(only_A[0])
    seedB = by_dist_from_center(only_B[0])
    if seedA == seedB or seedA < 0 or seedB < 0:
        return None

    assign = {seedA: "A", seedB: "B"}
    countsA = Counter([int(z_TS[seedA])])
    countsB = Counter([int(z_TS[seedB])])

    while len(assign) < n:
        best = None
        for i in range(n):
            if i in assign: continue
            elem = int(z_TS[i])
            can_A = countsA[elem] < quotaA.get(elem, 0)
            can_B = countsB[elem] < quotaB.get(elem, 0)
            if not can_A and not can_B: continue
            dA = min((D[i, j] for j, s in assign.items() if s == "A"), default=float("inf"))
            dB = min((D[i, j] for j, s in assign.items() if s == "B"), default=float("inf"))
            if can_A and can_B:
                side = "A" if dA <= dB else "B"; dist = min(dA, dB)
            elif can_A:
                side, dist = "A", dA
            else:
                side, dist = "B", dB
            cand = (dist, i, side)
            if best is None or cand < best: best = cand
        if best is None: break
        _, i, side = best
        assign[i] = side
        (countsA if side == "A" else countsB)[int(z_TS[i])] += 1

    if len(assign) != n: return None
    A = sorted([i for i, s in assign.items() if s == "A"])
    B = sorted([i for i, s in assign.items() if s == "B"])
    if Counter(int(z_TS[i]) for i in A) == quotaA:
        return A, B
    return None


def _fragment_max_spread(pos, indices):
    """Max pairwise distance within a fragment (Å)."""
    if len(indices) < 2: return 0.0
    p = pos[np.array(indices, int)]
    from scipy.spatial.distance import pdist
    return float(pdist(p).max())


def _molecular_graph(z, pos, mult=1.2):
    """Return NetworkX graph with atomic-number node attributes and bonds
    determined by ASE natural cutoffs × `mult`."""
    import ase, networkx as nx
    from ase.neighborlist import build_neighbor_list, natural_cutoffs
    atoms = ase.Atoms(numbers=[int(x) for x in z], positions=pos)
    cutoffs = [c * mult for c in natural_cutoffs(atoms)]
    nl = build_neighbor_list(atoms, cutoffs, self_interaction=False, bothways=True)
    g = nx.Graph()
    for i in range(len(z)):
        g.add_node(i, elem=int(z[i]))
    for i in range(len(z)):
        for j in nl.get_neighbors(i)[0]:
            g.add_edge(i, int(j))
    return g


def _find_subgraph_in_ts(z_sub, pos_sub, z_TS, pos_TS, mult=1.2):
    """Find atoms of a raw reactant XYZ (subgraph) in the TS geometry via
    element-preserving graph isomorphism (VF2). Returns list of TS atom
    indices in the ORDER of the subgraph's atoms, or None if no match.
    Sweeps mult values because TS bonds may be looser than reactant bonds."""
    import networkx as nx
    from networkx.algorithms import isomorphism
    for m in [mult, 1.15, 1.10, 1.25, 1.30, 1.05]:
        g_sub = _molecular_graph(z_sub, pos_sub, mult=1.15)  # reactant is well-defined
        g_ts = _molecular_graph(z_TS, pos_TS, mult=m)
        matcher = isomorphism.GraphMatcher(
            g_ts, g_sub,
            node_match=lambda a, b: a["elem"] == b["elem"],
        )
        for iso in matcher.subgraph_isomorphisms_iter():
            # iso is dict: ts_atom_idx → sub_atom_idx
            mapping = [None] * len(z_sub)
            for ts_i, sub_i in iso.items():
                mapping[sub_i] = ts_i
            if all(x is not None for x in mapping):
                return mapping
    return None


def _dipolar_r0_r1_to_ts(rid, z_TS, pos_TS):
    """Load raw r0.xyz and r1.xyz for a dipolar reaction and return two lists
    of TS atom indices via graph isomorphism. Returns (r0_ts_idx, r1_ts_idx)
    or None on failure."""
    import ase.io
    RAW_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
    idx = int(rid.split("_")[-1])
    d = RAW_ROOT / str(idx)
    r0p = next(iter(d.glob("r0_*.xyz")), None)
    r1p = next(iter(d.glob("r1_*.xyz")), None)
    if r0p is None or r1p is None:
        return None
    r0 = ase.io.read(str(r0p))
    r1 = ase.io.read(str(r1p))
    z0 = np.asarray(r0.get_atomic_numbers(), int)
    p0 = r0.get_positions()
    z1 = np.asarray(r1.get_atomic_numbers(), int)
    p1 = r1.get_positions()
    m0 = _find_subgraph_in_ts(z0, p0, z_TS, pos_TS)
    if m0 is None: return None
    used = set(m0)
    remaining = sorted(set(range(len(z_TS))) - used)
    # Check that remaining atoms have the right composition and count for r1
    from collections import Counter
    if Counter(int(z_TS[i]) for i in remaining) != Counter(int(z) for z in z1):
        return None
    if len(remaining) != len(z1):
        return None
    return sorted(m0), remaining  # r0's TS indices, r1's TS indices


def _compact_fragment_search(z_TS, pos_TS, quota_small):
    """Find the compact subset of TS atoms whose element composition matches
    `quota_small` (typically the smaller fragment) and minimises the maximum
    pairwise distance. Returns None if the combinatorial search space is too
    large (fallback to next path)."""
    import itertools
    from math import comb
    from collections import Counter
    from scipy.spatial.distance import pdist

    by_elem = {}
    for i, z in enumerate(z_TS):
        by_elem.setdefault(int(z), []).append(i)
    for e, k in quota_small.items():
        if k > len(by_elem.get(e, [])):
            return None

    # Pre-check total combo count WITHOUT materialising any lists (math.comb).
    total = 1
    for e, k in quota_small.items():
        total *= comb(len(by_elem[e]), k)
        if total > 20000:
            return None

    order = sorted(quota_small.items(), key=lambda x: -x[1])
    element_choices = [list(itertools.combinations(by_elem[e], k)) for e, k in order]

    best_score = float("inf")
    best_subset = None
    for tup in itertools.product(*element_choices):
        subset = []
        for grp in tup: subset.extend(grp)
        p = pos_TS[np.array(subset, int)]
        score = float(pdist(p).max()) if len(p) >= 2 else 0.0
        if score < best_score:
            best_score = score
            best_subset = sorted(subset)
    return best_subset, best_score


def resolve_ts_fragments(rid, family, manual, auto):
    """Preserve user fragment IDENTITY (element composition) and produce
    SPATIALLY COMPACT groups in TS."""
    from collections import Counter
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    z_TS = np.asarray(d["TS"]["z"], dtype=int)
    pos_TS = np.asarray(d["TS"]["pos"], dtype=float)
    A_R = manual[rid]["frag_A_indices"]
    B_R = manual[rid]["frag_B_indices"]

    if family == "rgd1":
        return list(A_R), list(B_R)

    z_R = np.asarray(d["R"]["z"], dtype=int) if "R" in d else None
    if z_R is None:
        raise KeyError(f"no R geom in .pt for {rid}")

    # FAST PATH: if R and TS have identical z-sequence, atom indices are
    # already TS-native → use user's manual indices directly. This applies to
    # rgd1, ~45% of dipolar, and some qmrxn20 cases.
    if len(z_R) == len(np.asarray(d["TS"]["z"], int)) and np.array_equal(z_R, np.asarray(d["TS"]["z"], int)):
        return list(A_R), list(B_R)

    quotaA = Counter(int(z_R[i]) for i in A_R)
    quotaB = Counter(int(z_R[i]) for i in B_R)

    # Path 0 (dipolar): graph isomorphism from raw r0.xyz + r1.xyz to TS.
    # Uses raw reactant connectivity as ground truth for atom identity.
    if family == "dipolar":
        r0_r1 = _dipolar_r0_r1_to_ts(rid, z_TS, pos_TS)
        if r0_r1 is not None:
            r0_ts, r1_ts = r0_r1
            # Match r0/r1 identity to user's fragA/fragB by element composition.
            zr0 = Counter(int(z_TS[i]) for i in r0_ts)
            zr1 = Counter(int(z_TS[i]) for i in r1_ts)
            if zr0 == quotaA and zr1 == quotaB:
                return r0_ts, r1_ts
            if zr0 == quotaB and zr1 == quotaA:
                return r1_ts, r0_ts

    # Path 1: natural CC split on TS + composition match.
    ts_split = _ts_two_fragments(z_TS, pos_TS)
    if ts_split is not None:
        comp1, comp2, mult = ts_split
        matched = _match_user_identity_to_ts_split(z_TS, (comp1, comp2), z_R, A_R, B_R)
        if matched is not None:
            return matched

    # Path 2: compact-fragment search on the SMALLER side (smaller = fewer combos).
    smaller_quota = quotaA if sum(quotaA.values()) <= sum(quotaB.values()) else quotaB
    small_is_A = smaller_quota is quotaA
    result = _compact_fragment_search(z_TS, pos_TS, smaller_quota)
    if result is not None:
        small_subset, _ = result
        all_idx = set(range(len(z_TS)))
        large_subset = sorted(all_idx - set(small_subset))
        # Verify large subset composition matches
        if Counter(int(z_TS[i]) for i in large_subset) == (quotaB if small_is_A else quotaA):
            if small_is_A:
                return small_subset, large_subset
            return large_subset, small_subset

    # Path 3: greedy growth from unique-element seeds.
    matched = _greedy_growth_from_seed(z_TS, pos_TS, z_R, A_R, B_R)
    if matched is not None:
        return matched

    # Last resort: legacy element-sequence map.
    mp = _element_sequence_map(z_R, z_TS)
    A_TS = sorted(mp[i] for i in A_R if mp[i] >= 0)
    B_TS = sorted(mp[i] for i in B_R if mp[i] >= 0)
    return A_TS, B_TS


def charge_and_mult(family: str) -> dict:
    if family in ("dipolar", "rgd1"):
        return dict(total=0, qa=0, ma=1, qb=0, mb=1)
    return dict(total=-1, qa=0, ma=1, qb=-1, mb=1)


def _parity_ok(numbers, charge, mult):
    n_electrons = int(sum(numbers)) - int(charge)
    n_unpaired = mult - 1
    return (n_electrons % 2) == (n_unpaired % 2)


def _fix_mult(numbers, charge, mult):
    if _parity_ok(numbers, charge, mult): return mult
    for cand in (mult - 1, mult + 1, 1, 2, 3):
        if cand >= 1 and _parity_ok(numbers, charge, cand):
            return cand
    return 1


def render_eda_input(rid, family, z, pos, frag_A, frag_B,
                     ncpu=8, maxcore=3500,
                     functional="BLYP", basis="def2-TZVP") -> str:
    cm = charge_and_mult(family)

    # Parity-adjust mults per fragment.
    idxA = np.array(frag_A, int)
    idxB = np.array(frag_B, int)
    ma = _fix_mult(z[idxA], cm["qa"], cm["ma"])
    mb = _fix_mult(z[idxB], cm["qb"], cm["mb"])
    total_mult = _fix_mult(z, cm["total"], 1)

    # Spin-flip on fragment 2 iff both fragments are open-shell (radical-radical).
    frag_sf_line = ""
    if ma > 1 and mb > 1:
        frag_sf_line = "  FRAG2_SF TRUE\n"

    open_shell_kw = " UKS" if (ma > 1 or mb > 1 or total_mult > 1) else ""

    quoted_method = f'"{functional} D3BJ {basis} NoSym TightSCF"'

    # Serial ORCA: this cluster lacks openmpi in PATH; no %pal block.
    header = (
        f"! {functional} D3BJ {basis} NoSym EDA TightSCF{open_shell_kw}\n"
        f"%maxcore {maxcore}\n"
        f"\n"
        f"%eda\n"
        f"  FRAG1 {quoted_method}\n"
        f"  FRAG2 {quoted_method}\n"
        f"  FRAG1_C {cm['qa']}\n"
        f"  FRAG1_M {ma}\n"
        f"  FRAG2_C {cm['qb']}\n"
        f"  FRAG2_M {mb}\n"
        f"{frag_sf_line}"
        f"end\n"
        f"\n"
        f"* xyz {cm['total']} {total_mult}\n"
    )
    A_set = set(frag_A)
    body_lines = []
    for i in range(len(z)):
        fid = 1 if i in A_set else 2
        sym = chemical_symbols[int(z[i])]
        body_lines.append(
            f"{sym}({fid})".ljust(6) + f" {pos[i,0]:>15.8f} {pos[i,1]:>15.8f} {pos[i,2]:>15.8f}"
        )
    return header + "\n".join(body_lines) + "\n*\n", ma, mb, total_mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--maxcore", type=int, default=3500)
    ap.add_argument("--functional", default="BLYP")
    ap.add_argument("--basis", default="def2-TZVP")
    ap.add_argument("--only-reviewed", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    labels = pd.read_parquet(LABELS_PQ)
    with open(MANUAL_PART) as f: manual = json.load(f)
    auto = {}
    if AUTO_PART.exists():
        with open(AUTO_PART) as f: auto = json.load(f)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = 0
    for i, row in enumerate(labels.itertuples(index=False)):
        if args.limit is not None and i >= args.limit: break
        rid, fam = row.reaction_id, row.family
        m = manual.get(rid)
        if not m or "frag_A_indices" not in m:
            n_skip += 1; continue
        if args.only_reviewed and not m.get("reviewed", False):
            n_skip += 1; continue
        try:
            z, pos = load_ts(rid)
            A, B = resolve_ts_fragments(rid, fam, manual, auto)
            assigned = set(A) | set(B)
            if len(assigned) != len(z):
                raise RuntimeError(f"unassigned TS atoms {[i for i in range(len(z)) if i not in assigned]}")
            if set(A) & set(B):
                raise RuntimeError("A/B overlap")

            inp, ma, mb, mC = render_eda_input(
                rid, fam, z, pos, A, B,
                ncpu=args.ncpu, maxcore=args.maxcore,
                functional=args.functional, basis=args.basis,
            )
            out_dir = OUT_ROOT / rid
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "eda.inp").write_text(inp)
            meta = dict(
                reaction_id=rid, family=fam, n_atoms=int(len(z)),
                frag_A_indices=A, frag_B_indices=B,
                total_charge=charge_and_mult(fam)["total"],
                charge_A=charge_and_mult(fam)["qa"],
                charge_B=charge_and_mult(fam)["qb"],
                mult_A=ma, mult_B=mb, mult_complex=mC,
                note=m.get("note", ""),
                method=f"{args.functional} D3BJ {args.basis}",
                ncpu=args.ncpu, maxcore=args.maxcore,
                inp_sha1=hashlib.sha1(inp.encode()).hexdigest(),
            )
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {rid}: {exc}", flush=True)

    print(f"done: ok={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
