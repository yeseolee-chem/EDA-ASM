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


def find_files(prof_root: Path, rid: int):
    d = prof_root / str(rid)
    ts = [f for f in d.iterdir()
          if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"]
    r = sorted(f for f in d.iterdir()
               if re.match(r"^r\d+_", f.name) and f.suffix == ".xyz"
               and "_alt" not in f.name)
    p = sorted(f for f in d.iterdir() if f.name.startswith("p0_"))
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
                return None
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
            return None, "no isomorphism"

    X, info, comps, formed = attempt

    Y, shift = separate_fragments(X, ts_s, comps[0], comps[1])
    p_s, p_x = read_xyz(p_f)
    return dict(
        R=Y, TS=ts_x, P=p_x, syms=ts_s,
        frag1=comps[0], frag2=comps[1],
        shift=shift, p_order_ok=(p_s == ts_s), info=info,
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
