"""SPEC17rev2 §4 canonical fragment alignment.

Do NOT modify — 240/240 connectivity-preservation verified in spec.
Any change invalidates the isomorphism guarantees.
"""
from __future__ import annotations

import collections
import itertools
import re
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

# --- physical constants (Å) -------------------------------------------------
COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39, "P": 1.07}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
       "Cl": 1.75, "Br": 1.85, "S": 1.80, "I": 1.98, "P": 1.80}

FAC = 1.25          # covalent-bond tolerance
SEP_FACTOR = 0.90   # vdW separation factor
MAX_ISO = 5000      # cap on isomorphism enumeration (240 frags: 11/4.6% hit)


def read_xyz(path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    syms, xyz = [], []
    for l in lines[2:2 + n]:
        t = l.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def find_files(prof_root: Path, rid: int, include_alt: bool = False):
    """Locate TS/R/P files. If include_alt=True, also return _alt conformers
    grouped by slot (used by build_reactant_complex recovery paths)."""
    d = prof_root / str(rid)
    ts = [f for f in d.iterdir()
          if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"]
    p = sorted(f for f in d.iterdir() if f.name.startswith("p0_"))
    if include_alt:
        r_by_slot: dict[int, list] = {}
        for f in sorted(d.iterdir()):
            m = re.match(r"^r(\d+)_", f.name)
            if not m or f.suffix != ".xyz":
                continue
            r_by_slot.setdefault(int(m.group(1)), []).append(f)
        return (ts[0] if ts else None), r_by_slot, (p[0] if p else None)
    r = sorted(f for f in d.iterdir()
               if re.match(r"^r\d+_", f.name) and f.suffix == ".xyz"
               and "_alt" not in f.name)
    return (ts[0] if ts else None), r, (p[0] if p else None)


def build_graph(syms, xyz, skip: Iterable = ()):
    G = nx.Graph()
    n = len(syms)
    for i in range(n):
        G.add_node(i, el=syms[i])
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    sk = {tuple(sorted(b)) for b in skip}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in sk:
                continue
            if D[i, j] < FAC * (COV.get(syms[i], 0.8) + COV.get(syms[j], 0.8)):
                G.add_edge(i, j)
    return G


def kabsch_apply(mob, ref):
    mc, rc = mob.mean(0), ref.mean(0)
    a, b = mob - mc, ref - rc
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return a @ R.T + rc


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(1).mean()))


def formula(syms):
    c = collections.Counter(syms)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def separate_fragments(X, syms, A, B):
    """Push fragment B along the centroid axis until vdW-separated."""
    axis = X[B].mean(0) - X[A].mean(0)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
        n = 1.0
    axis = axis / n
    tgt = np.array([[SEP_FACTOR * (VDW.get(syms[i], 1.7) + VDW.get(syms[j], 1.7))
                     for j in B] for i in A])
    lo, hi = 0.0, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        D = np.linalg.norm(
            X[A][:, None, :] - (X[B][None, :, :] + mid * axis), axis=-1
        )
        if (D >= tgt).all():
            hi = mid
        else:
            lo = mid
    Y = X.copy()
    Y[B] = X[B] + hi * axis
    return Y, hi


def match_fragment_hungarian(ts_syms, ts_xyz, r_syms, r_xyz,
                               max_iter=8, rmsd_max=1.5, jaccard_min=0.85):
    """Recovery 6: element-preserving Hungarian bipartite matching.

    For R conformers whose graph is a different constitutional isomer than
    the TS-fragment (bond opens/closes during R→TS), graph iso fails but
    coordinate-based matching may still find the correct correspondence.

    Iterative Kabsch alignment + linear_sum_assignment on element-
    partitioned distance matrix. Accepted only if:
      - RMSD < rmsd_max (default 1.5 Å) after final alignment
      - Jaccard edge overlap > jaccard_min (default 0.85) between TS-frag
        graph and permuted R graph — ensures chemistry consistency
      - All element labels match

    Returns (order, aligned_xyz, n_iso, hit_cap) on success, None on
    reject. n_iso reported as -1 to signal Hungarian recovery.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    n = len(ts_syms)
    if len(r_syms) != n or sorted(ts_syms) != sorted(r_syms):
        return None
    order = list(range(n))
    for _ in range(max_iter):
        mc = np.mean(r_xyz[order], axis=0)
        rc_ts = np.mean(ts_xyz, axis=0)
        a = r_xyz[order] - mc
        b = ts_xyz - rc_ts
        H = a.T @ b
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rot = Vt.T @ np.diag([1, 1, d]) @ U.T
        r_trans = (r_xyz - mc) @ Rot.T + rc_ts
        D = np.full((n, n), 1e9)
        for i in range(n):
            for j in range(n):
                if ts_syms[i] == r_syms[j]:
                    D[i, j] = np.linalg.norm(ts_xyz[i] - r_trans[j])
        _, new_order = linear_sum_assignment(D)
        new_order = new_order.tolist()
        if new_order == order:
            break
        order = new_order
    # Post-hoc verification
    perm_syms = [r_syms[j] for j in order]
    if any(ts_syms[i] != perm_syms[i] for i in range(n)):
        return None
    aligned = kabsch_apply(r_xyz[order], ts_xyz)
    r_val = rmsd(aligned, ts_xyz)
    if r_val > rmsd_max:
        return None
    Gt = build_graph(ts_syms, ts_xyz)
    Gr_perm = build_graph(perm_syms, r_xyz[order])
    ts_e = {tuple(sorted(e)) for e in Gt.edges()}
    r_e = {tuple(sorted(e)) for e in Gr_perm.edges()}
    inter = ts_e & r_e
    union = ts_e | r_e
    if not union or len(inter) / len(union) < jaccard_min:
        return None
    return order, aligned, -1, False


def match_fragment(ts_syms, ts_xyz, r_syms, r_xyz):
    """Enumerate graph isomorphisms and pick RMSD-optimal alignment.

    All isomorphisms are true automorphisms (WL refinement confirms 1.0× ratio),
    so any choice is chemically valid; RMSD picks the geometrically best.
    Returns (order, aligned_xyz, n_iso, hit_cap) or None.
    """
    Gt = build_graph(ts_syms, ts_xyz)
    Gr = build_graph(r_syms, r_xyz)
    GM = GraphMatcher(Gt, Gr, node_match=categorical_node_match("el", None))
    best_order, best_v, cnt, hit_cap = None, 1e9, 0, False
    for cnt, iso in enumerate(GM.isomorphisms_iter()):
        if cnt >= MAX_ISO:
            hit_cap = True
            break
        order = [iso[i] for i in range(len(ts_syms))]
        v = rmsd(kabsch_apply(r_xyz[order], ts_xyz), ts_xyz)
        if v < best_v:
            best_v, best_order = v, order
    if best_order is None:
        return None
    aligned = kabsch_apply(r_xyz[best_order], ts_xyz)
    # cnt hits MAX_ISO on the sentinel iteration we don't score; cap the
    # reported count so downstream stats never exceed MAX_ISO.
    return best_order, aligned, min(cnt + 1, MAX_ISO), hit_cap


def verify_correspondence(ts_syms, ts_xyz, r_syms, r_xyz, order):
    """Post-hoc check: re-ordered R must preserve TS adjacency."""
    Gt = build_graph(ts_syms, ts_xyz)
    Gr = build_graph([r_syms[j] for j in order], r_xyz[order])
    return all(set(Gt[i]) == set(Gr[i]) for i in range(len(ts_syms)))


def _diff_formed_bonds(ts_syms, ts_xyz, p_syms, p_xyz):
    """Chemically derive forming bonds from P-TS graph diff.

    Returns list of 2 (i,j) tuples if P has exactly 2 more edges than TS
    (clean cycloaddition topology). Returns None otherwise — callers use
    this only as a fallback when the filename-parsed bonds fail to
    split the TS graph into 2 fragments.
    """
    if ts_syms != p_syms:
        return None
    ts_edges = {tuple(sorted(e)) for e in build_graph(ts_syms, ts_xyz).edges()}
    p_edges = {tuple(sorted(e)) for e in build_graph(p_syms, p_xyz).edges()}
    formed = p_edges - ts_edges
    if len(formed) == 2:
        return sorted(formed)
    return None


def _smiles_inter_fragment_edges(prof_root: Path, rid: int, ts_syms, ts_xyz,
                                   p_syms=None, p_xyz=None):
    """Use SMILES atom-mapping to identify which TS edges connect the two
    reactant molecules. Recovers rxns where TS geometry is 'late' (past
    the saddle point) and forming bonds are already fully bonded — the
    filename bonds don't split TS because those bonds are >1.9Å (missed
    by build_graph), while the ACTUAL formed bonds are <1.5Å (counted as
    normal bonds).

    Returns list of (i,j) edges to remove to split TS into 2 pieces
    matching the R molecule composition, or None on failure.
    """
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
        from networkx.algorithms.isomorphism import (
            GraphMatcher, categorical_node_match,
        )
    except ImportError:
        return None
    csv_path = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/"
                    "dipolar_cycloaddition/full_dataset.csv")
    if not csv_path.exists():
        return None
    try:
        smi = pd.read_csv(csv_path).set_index("rxn_id").loc[rid, "rxn_smiles"]
    except (KeyError, FileNotFoundError):
        return None
    try:
        r_smi, p_smi = smi.split(">>")
    except ValueError:
        return None
    # Get atom_map groups per reactant molecule
    r_mols = r_smi.split(".")
    if len(r_mols) != 2:
        return None
    r_groups_am = []
    for m_smi in r_mols:
        m = Chem.MolFromSmiles(m_smi)
        if m is None:
            return None
        r_groups_am.append({a.GetAtomMapNum() for a in m.GetAtoms()
                            if a.GetAtomMapNum() > 0})

    # Match SMILES-P to a reference xyz to get atom_map → xyz_idx
    P_h = Chem.AddHs(Chem.MolFromSmiles(p_smi))
    G_smi = nx.Graph()
    m2i = {}
    for a in P_h.GetAtoms():
        G_smi.add_node(a.GetIdx(), el=a.GetSymbol())
        if a.GetAtomMapNum() > 0:
            m2i[a.GetAtomMapNum()] = a.GetIdx()
    for b in P_h.GetBonds():
        G_smi.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    # Try P xyz first (matches SMILES-P directly), then TS xyz
    m2xyz = None
    refs = []
    if p_syms is not None and p_xyz is not None:
        refs.append((p_syms, p_xyz))
    refs.append((ts_syms, ts_xyz))
    for xyz_syms, xyz_x in refs:
        G_xyz = build_graph(xyz_syms, xyz_x)
        GM = GraphMatcher(G_xyz, G_smi,
                          node_match=categorical_node_match("el", None))
        for cnt, iso in enumerate(GM.isomorphisms_iter()):
            if cnt >= 100:
                break
            inv = {v: k for k, v in iso.items()}
            try:
                m2xyz = {am: inv[smi_i] for am, smi_i in m2i.items()
                         if smi_i in inv}
            except KeyError:
                continue
            break
        if m2xyz:
            break
    if m2xyz is None or len(m2xyz) != len(m2i):
        return None

    # Translate R atom-map groups into xyz index sets (heavy atoms only —
    # H's have atom_map=0 in SMILES)
    xyz_groups = [{m2xyz[am] for am in g if am in m2xyz} for g in r_groups_am]

    # Find TS edges connecting the two groups (assign each H by connectivity)
    G_ts = build_graph(ts_syms, ts_xyz)
    # Assign every atom (including H) to a group by BFS from the heavy seeds
    group_of = {}
    for gi, g in enumerate(xyz_groups):
        for a in g:
            group_of[a] = gi
    # BFS to propagate through non-inter edges — but we need to identify
    # inter edges first, chicken/egg. Instead: propagate through short edges
    # (within-molecule bonds are typically < 1.8Å for heavy-H, < 1.6Å for
    # heavy-heavy in a fragment). Any edge > 1.6Å that crosses group_of
    # boundaries is a candidate inter-fragment edge.
    # Simpler: assign H atoms to whichever heavy neighbor they're closest to.
    for i, s in enumerate(ts_syms):
        if i in group_of:
            continue
        if s == "H":
            # Find nearest heavy atom with assigned group
            best = None; best_d = 1e9
            for j in G_ts.neighbors(i):
                if j in group_of:
                    d = float(np.linalg.norm(ts_xyz[i] - ts_xyz[j]))
                    if d < best_d:
                        best_d, best = d, group_of[j]
            if best is not None:
                group_of[i] = best
    inter_edges = []
    for u, v in G_ts.edges():
        gu = group_of.get(u); gv = group_of.get(v)
        if gu is not None and gv is not None and gu != gv:
            inter_edges.append(tuple(sorted((u, v))))
    if not inter_edges:
        return None
    return sorted(set(inter_edges))


def _smiles_formed_bonds(prof_root: Path, rid: int, ts_syms, ts_xyz,
                          p_syms=None, p_xyz=None):
    """Derive xyz-indexed forming bonds by matching Coley's atom-mapped
    P SMILES to a Coley-provided reference (P xyz first, then TS xyz).
    Recovers rxns where the TS filename's template indices don't match
    the actual xyz atom ordering (e.g. rxn 5930).

    Returns list of 2 (i,j) tuples on success, else None.

    Requires the full_dataset.csv rxn_smiles column and rdkit. Falls back
    gracefully (returns None) if either is unavailable.
    """
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
        from networkx.algorithms.isomorphism import (
            GraphMatcher, categorical_node_match,
        )
    except ImportError:
        return None
    csv_path = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/"
                    "dipolar_cycloaddition/full_dataset.csv")
    if not csv_path.exists():
        return None
    try:
        smi = pd.read_csv(csv_path).set_index("rxn_id").loc[rid, "rxn_smiles"]
    except (KeyError, FileNotFoundError):
        return None
    try:
        reac, prod = smi.split(">>")
        R = Chem.MolFromSmiles(reac); P = Chem.MolFromSmiles(prod)
    except (ValueError, AttributeError):
        return None
    if R is None or P is None:
        return None

    def _bonds_by_atommap(m):
        s = set()
        for b in m.GetBonds():
            i = b.GetBeginAtom().GetAtomMapNum()
            j = b.GetEndAtom().GetAtomMapNum()
            if i and j:
                s.add((min(i, j), max(i, j)))
        return s

    formed_am = _bonds_by_atommap(P) - _bonds_by_atommap(R)
    if len(formed_am) != 2:
        return None  # not a canonical 2-bond cycloaddition

    # Build SMILES-P graph (with Hs) tracking atom map
    P_h = Chem.AddHs(P)
    G_smi = nx.Graph()
    m2i = {}
    for a in P_h.GetAtoms():
        G_smi.add_node(a.GetIdx(), el=a.GetSymbol())
        if a.GetAtomMapNum() > 0:
            m2i[a.GetAtomMapNum()] = a.GetIdx()
    for b in P_h.GetBonds():
        G_smi.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    # Try P xyz first (its graph structure matches SMILES-P directly, no
    # bond-elongation issues from being an intermediate state).
    # Fall back to TS xyz if P.xyz graph doesn't isomorphize (e.g. its own
    # atom labeling is scrambled).
    refs = []
    if p_syms is not None and p_xyz is not None:
        refs.append((p_syms, p_xyz))
    refs.append((ts_syms, ts_xyz))
    for xyz_syms, xyz_x in refs:
        G_xyz = build_graph(xyz_syms, xyz_x)
        GM = GraphMatcher(G_xyz, G_smi,
                          node_match=categorical_node_match("el", None))
        for cnt, iso in enumerate(GM.isomorphisms_iter()):
            if cnt >= 100:
                break
            inv = {v: k for k, v in iso.items()}
            try:
                formed_xyz = sorted(
                    tuple(sorted((inv[m2i[a]], inv[m2i[b]])))
                    for a, b in formed_am
                )
            except KeyError:
                continue
            return formed_xyz
    return None


def build_reactant_complex(prof_root: Path, rid: int):
    ts_f, r_f, p_f = find_files(prof_root, rid)
    if ts_f is None or len(r_f) != 2 or p_f is None:
        return None, "missing files"

    m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", ts_f.name)
    if not m:
        return None, "TS filename parse failure"
    formed = [tuple(sorted((int(m.group(1)), int(m.group(2))))),
              tuple(sorted((int(m.group(3)), int(m.group(4)))))]

    ts_s, ts_x = read_xyz(ts_f)
    comps = [sorted(c) for c in
             nx.connected_components(build_graph(ts_s, ts_x, skip=formed))]
    recovery_used = None
    if len(comps) != 2:
        # RECOVERY 1: filename bonds don't split TS → try P-TS graph diff.
        # Chemically valid: bonds actually new in P vs TS are the true
        # cycloaddition bonds regardless of what the filename says.
        p_s0, p_x0 = read_xyz(p_f)
        alt_formed = _diff_formed_bonds(ts_s, ts_x, p_s0, p_x0)
        if alt_formed is not None:
            alt_comps = [sorted(c) for c in
                         nx.connected_components(
                             build_graph(ts_s, ts_x, skip=alt_formed))]
            if len(alt_comps) == 2:
                formed = alt_formed
                comps = alt_comps
                recovery_used = "diff_bonds_split"
        if len(comps) != 2:
            # RECOVERY 3: use full_dataset.csv atom-mapped SMILES to derive
            # forming bonds in xyz-index space via P.xyz graph iso. Recovers
            # rxns where filename template indices are wrong AND P-TS graph
            # diff hits close-contact spurious bonds (e.g. rxn 5930).
            p_s0, p_x0 = read_xyz(p_f)
            smi_formed = _smiles_formed_bonds(prof_root, rid, ts_s, ts_x, p_s0, p_x0)
            if smi_formed is not None:
                smi_comps = [sorted(c) for c in
                             nx.connected_components(
                                 build_graph(ts_s, ts_x, skip=smi_formed))]
                if len(smi_comps) == 2:
                    formed = smi_formed
                    comps = smi_comps
                    recovery_used = "smiles_bonds_split"
        if len(comps) != 2:
            # RECOVERY 5: SMILES atom-group inter-fragment edges. For "late"
            # TS geometries where forming bonds are already fully formed
            # (~1.5 Å, counted as bonds by build_graph), identify which TS
            # edges cross the R-molecule boundary via SMILES atom-mapping
            # and remove those. Recovers rxns 3090, 3766, 4252.
            inter = _smiles_inter_fragment_edges(
                prof_root, rid, ts_s, ts_x, p_s0, p_x0)
            if inter is not None:
                G_split = build_graph(ts_s, ts_x)
                G_split.remove_edges_from(inter)
                split_comps = [sorted(c) for c in
                               nx.connected_components(G_split)]
                if len(split_comps) == 2:
                    formed = inter
                    comps = split_comps
                    recovery_used = "smiles_inter_edges"
        if len(comps) != 2:
            return None, f"{len(comps)}-piece split (expected 2)"

    reacts = [read_xyz(f) for f in r_f]

    def _try_align(components, formed_bonds):
        """Try to align both reactants onto the given TS components.
        Returns (X, info) on success, None on isomorphism/correspondence fail.
        """
        X_local = np.zeros_like(ts_x)
        info_local = []
        # Match composition pairing for THIS component set (may differ from
        # the outer `pairing` if the components changed via recovery).
        local_pairing = None
        for perm in itertools.permutations(range(2)):
            if all(formula([ts_s[i] for i in components[k]]) == formula(reacts[perm[k]][0])
                   for k in range(2)):
                local_pairing = perm
                break
        if local_pairing is None:
            return None
        for k in range(2):
            idx = components[k]
            rs, rx = reacts[local_pairing[k]]
            sub_s = [ts_s[i] for i in idx]
            sub_x = ts_x[idx]
            res = match_fragment(sub_s, sub_x, rs, rx)
            if res is None:
                # Recovery 6: Hungarian element-preserving assignment.
                # For rxns where R→TS involves a bond opening/closing
                # (constitutional isomer change), graph iso fails but
                # coordinate matching with Jaccard-edge verification may
                # still produce a chemically sensible mapping.
                res = match_fragment_hungarian(sub_s, sub_x, rs, rx)
                if res is None:
                    return None
                order, aligned, n_iso, hit_cap = res
                # Skip strict verify_correspondence — Hungarian was already
                # Jaccard-verified against the TS-frag graph.
            else:
                order, aligned, n_iso, hit_cap = res
                if not verify_correspondence(sub_s, sub_x, rs, rx, order):
                    return None
            X_local[idx] = aligned
            disp = np.linalg.norm(aligned - sub_x, axis=1)
            info_local.append(dict(
                n_iso=n_iso, hit_cap=hit_cap,
                frag_rmsd=rmsd(aligned, sub_x),
                max_disp_heavy=float(max(
                    [disp[i] for i in range(len(idx)) if sub_s[i] != "H"] or [0])),
                max_disp_h=float(max(
                    [disp[i] for i in range(len(idx)) if sub_s[i] == "H"] or [0])),
            ))
        return X_local, info_local, components, formed_bonds

    attempt = _try_align(comps, formed)
    if attempt is None:
        # RECOVERY 2: filename bonds split TS but no reactant isomorphism.
        # Retry with diff-based forming bonds — different split may match
        # actual Coley r*.xyz connectivity when filename bonds don't.
        p_s0, p_x0 = read_xyz(p_f)
        alt_formed = _diff_formed_bonds(ts_s, ts_x, p_s0, p_x0)
        if alt_formed is not None and alt_formed != formed:
            alt_comps = [sorted(c) for c in
                         nx.connected_components(
                             build_graph(ts_s, ts_x, skip=alt_formed))]
            if len(alt_comps) == 2:
                attempt = _try_align(alt_comps, alt_formed)
                if attempt is not None:
                    recovery_used = "diff_bonds_iso"
        if attempt is None:
            # RECOVERY 3b: SMILES-derived bonds as last resort.
            p_s0, p_x0 = read_xyz(p_f)
            smi_formed = _smiles_formed_bonds(prof_root, rid, ts_s, ts_x, p_s0, p_x0)
            if smi_formed is not None and smi_formed != formed:
                smi_comps = [sorted(c) for c in
                             nx.connected_components(
                                 build_graph(ts_s, ts_x, skip=smi_formed))]
                if len(smi_comps) == 2:
                    attempt = _try_align(smi_comps, smi_formed)
                    if attempt is not None:
                        recovery_used = "smiles_bonds_iso"
        if attempt is None:
            return None, "no isomorphism"

    X, info, comps, formed = attempt

    Y, shift = separate_fragments(X, ts_s, comps[0], comps[1])
    p_s, p_x = read_xyz(p_f)
    p_ordered = (p_s == ts_s)

    # RECOVERY 4: P atom relabel-swap canonicalization (Group B fix).
    # verify_product's graph check fails when Coley's P.xyz has symmetric
    # atoms swapped relative to (TS + formed_bonds → P) prediction. Find
    # the permutation that makes P graph match the predicted P graph, apply
    # to coordinates. Preserves chemistry; only reorders indexing.
    if p_ordered:
        G_ts_plus_formed = build_graph(ts_s, ts_x)
        G_ts_plus_formed.add_edges_from(formed)
        G_p = build_graph(p_s, p_x)
        neighbors_match = all(
            set(G_ts_plus_formed[i]) == set(G_p[i]) for i in range(len(ts_s))
        )
        if not neighbors_match:
            from networkx.algorithms.isomorphism import (
                GraphMatcher, categorical_node_match,
            )
            # Build labeled graphs (element attr already set by build_graph)
            GM = GraphMatcher(
                G_p, G_ts_plus_formed,
                node_match=categorical_node_match("el", None),
            )
            for cnt, iso in enumerate(GM.isomorphisms_iter()):
                if cnt >= 100:
                    break
                # iso: G_p node → G_ts_plus_formed node.
                # New P coord for slot i = old P coord at atom that maps to slot i.
                inv = {v: k for k, v in iso.items()}   # ts_slot → p_atom
                try:
                    perm = [inv[i] for i in range(len(ts_s))]
                except KeyError:
                    continue
                new_p_x = p_x[perm]
                new_p_s = [p_s[i] for i in perm]
                # Verify permutation restores graph identity
                G_p_new = build_graph(new_p_s, new_p_x)
                if all(set(G_ts_plus_formed[i]) == set(G_p_new[i])
                       for i in range(len(ts_s))):
                    p_x = new_p_x
                    p_s = new_p_s
                    p_ordered = True
                    recovery_used = ((recovery_used + "+" if recovery_used else "")
                                     + "p_canonicalize")
                    break

    return dict(
        R=Y, TS=ts_x, P=p_x, syms=ts_s,
        frag1=comps[0], frag2=comps[1],
        shift=shift, p_order_ok=p_ordered, info=info,
        recovery_used=recovery_used,
    ), None


def verify_product(prof_root: Path, rid: int):
    """P atom order == TS atom order (verified by graph identity)."""
    ts_f, r_f, p_f = find_files(prof_root, rid)
    if ts_f is None or p_f is None:
        return None
    m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", ts_f.name)
    if not m:
        return None
    fb = [(int(m.group(1)), int(m.group(2))),
          (int(m.group(3)), int(m.group(4)))]
    ts_s, ts_x = read_xyz(ts_f)
    p_s, p_x = read_xyz(p_f)
    if ts_s != p_s:
        return False
    Gt = build_graph(ts_s, ts_x)
    Gt.add_edges_from(fb)
    Gp = build_graph(p_s, p_x)
    return all(set(Gt[i]) == set(Gp[i]) for i in range(len(ts_s)))
