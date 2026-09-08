#!/usr/bin/env python3
"""fragmenter.py — graph-based reactant→TS fragmentation for EDA / ASM.

Rule (one sentence)
-------------------
    Fragment k  =  the image of reactant k's bond graph embedded (subgraph
                   monomorphism) into the TS bond graph, accepted only if
                   the two images are disjoint, exhaustive, and each is
                   label-isomorphic to its reactant.

Everything else in this file is either (a) how the graphs are built from
xyz, or (b) audits that use the atom mapping the rule produces.

Definitions
-----------
bond graph      atoms = nodes; edge iff  d_ij < FACTOR * (r_i + r_j)   (Cordero 2008 radii)
heavy graph     bond graph restricted to heavy atoms; node label = (element, #H attached)
                — H atoms never change owner in a cycloaddition, so folding them into a
                  count removes the H-permutation symmetry without losing information
monomorphism    every edge of the small graph exists in the big graph; the big graph may
                have extra edges (the two forming bonds at the TS are exactly such extras)
isomorphism     same nodes, same edges, same labels (used for the remainder ↔ reactant B test)

Outputs of partition():
    A, B        frozensets of TS atom indices (heavy + H)
    map0, map1  dict  reactant-heavy-index -> TS-index   (needed for strain references,
                per-atom descriptors, SMILES-defined forming bonds)
    diag        n_maps, cap_hit, h_unattached

Dependencies: numpy, networkx (>=3), rdkit (only for the SMILES helpers).
"""
from __future__ import annotations

import collections
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

# --------------------------------------------------------------------------- constants
COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}          # Cordero et al. 2008
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "Br": 35, "S": 16, "I": 53}
FACTOR = 1.25          # covalent-distance criterion (same as the primary rule)
MAX_MAPS = 5000        # safety cap on enumerated monomorphisms (dataset max observed: 432)


# --------------------------------------------------------------------------- xyz
def read_xyz(path: str | Path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.asarray(xyz, dtype=float)


def dist_matrix(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


def bond_matrix(syms, xyz, factor: float = FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """Boolean adjacency from covalent radii. Returns (adj, D)."""
    D = dist_matrix(xyz)
    r = np.array([COV[s] for s in syms])
    thr = factor * (r[:, None] + r[None, :])
    adj = D < thr
    np.fill_diagonal(adj, False)
    return adj, D


# --------------------------------------------------------------------------- graphs
def heavy_graph(syms, xyz, factor: float = FACTOR):
    """Heavy-atom graph with (element, nH) node labels.

    Each H is attached to its nearest heavy atom (owner). Returns
    (G, owner, adj, D, n_unattached) where n_unattached counts H atoms whose
    nearest heavy atom is beyond the covalent threshold (should be 0).
    """
    adj, D = bond_matrix(syms, xyz, factor)
    heavy = [i for i, s in enumerate(syms) if s != "H"]
    owner: dict[int, int] = {}
    n_unattached = 0
    for i, s in enumerate(syms):
        if s != "H":
            continue
        j = min(heavy, key=lambda k: D[i, k])
        owner[i] = j
        if not adj[i, j]:
            n_unattached += 1
    n_h = collections.Counter(owner.values())
    G = nx.Graph()
    for i in heavy:
        G.add_node(i, lab=(syms[i], n_h.get(i, 0)))
    for a, b in itertools.combinations(heavy, 2):
        if adj[a, b]:
            G.add_edge(a, b)
    return G, owner, adj, D, n_unattached


def _nm(a, b) -> bool:
    return a["lab"] == b["lab"]


# --------------------------------------------------------------------------- the rule
@dataclass
class Partition:
    status: str                       # "ok" | "heavy_count_mismatch" | "no_partition" | "ambiguous"
    A: frozenset = frozenset()        # TS atom indices belonging to reactant 0
    B: frozenset = frozenset()        # ... reactant 1
    map0: dict = field(default_factory=dict)   # reactant0 heavy idx -> TS idx
    map1: dict = field(default_factory=dict)   # reactant1 heavy idx -> TS idx
    diag: dict = field(default_factory=dict)


def partition(ts, r0, r1, factor: float = FACTOR, max_maps: int = MAX_MAPS) -> Partition:
    """ts, r0, r1 are (syms, xyz) tuples. See module docstring for the rule."""
    ts_syms, ts_xyz = ts
    G_TS, owner, adj, D, n_unatt = heavy_graph(ts_syms, ts_xyz, factor)
    G0, _, _, _, _ = heavy_graph(*r0, factor)
    G1, _, _, _, _ = heavy_graph(*r1, factor)
    heavy_ts = set(G_TS.nodes)
    diag = dict(h_unattached=n_unatt, n_maps=0, cap_hit=False)

    if len(G0) + len(G1) != len(heavy_ts):
        return Partition("heavy_count_mismatch", diag=diag)

    # enumerate embeddings of reactant-0 graph into the TS graph
    images: dict[frozenset, tuple[dict, dict]] = {}
    for m in GraphMatcher(G_TS, G0, node_match=_nm).subgraph_monomorphisms_iter():
        diag["n_maps"] += 1
        img = frozenset(m.keys())                       # m: TS node -> G0 node
        if img not in images:
            rest = heavy_ts - img
            gm1 = GraphMatcher(G_TS.subgraph(rest), G1, node_match=_nm)
            if gm1.is_isomorphic():                     # remainder must BE reactant 1
                inv0 = {v: k for k, v in m.items()}       # G0 node -> TS node
                inv1 = {v: k for k, v in gm1.mapping.items()}
                images[img] = (inv0, inv1)
        if diag["n_maps"] >= max_maps:
            diag["cap_hit"] = True
            break

    if not images:
        return Partition("no_partition", diag=diag)
    if len(images) > 1:
        diag["n_images"] = len(images)
        return Partition("ambiguous", diag=diag)

    A_heavy, (map0, map1) = next(iter(images.items()))
    A = frozenset(A_heavy) | {h for h, o in owner.items() if o in A_heavy}
    B = frozenset(range(len(ts_syms))) - A

    # gates: sizes and composition (cheap redundancy on top of the isomorphism)
    ok = (len(A) == len(r0[0]) and len(B) == len(r1[0])
          and collections.Counter(ts_syms[i] for i in A) == collections.Counter(r0[0])
          and collections.Counter(ts_syms[i] for i in B) == collections.Counter(r1[0]))
    if not ok:
        return Partition("size_or_formula_mismatch", diag=diag)
    return Partition("ok", A, B, map0, map1, diag)


def geometric_components(syms, xyz, factor: float = FACTOR):
    """Connected components of the full bond graph (= the primary rule, for comparison)."""
    adj, _ = bond_matrix(syms, xyz, factor)
    G = nx.Graph()
    G.add_nodes_from(range(len(syms)))
    G.add_edges_from(zip(*np.where(np.triu(adj))))
    return [frozenset(c) for c in nx.connected_components(G)]


# --------------------------------------------------------------------------- SMILES helpers (rdkit)
def smiles_reactants_and_formed(rxn_smiles: str):
    """Atom-mapped reaction SMILES -> (reactant mols, formed bonds, broken bonds) on map numbers."""
    from rdkit import Chem
    r_str, p_str = rxn_smiles.split(">>")
    rmols = [Chem.MolFromSmiles(s) for s in r_str.split(".")]
    pmol = Chem.MolFromSmiles(p_str)

    def bonds(m):
        return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                for b in m.GetBonds()}
    rb = set().union(*[bonds(m) for m in rmols])
    pb = bonds(pmol)
    return rmols, pb - rb, rb - pb


def mol_heavy_graph(mol):
    """RDKit mol -> heavy graph keyed by atom-map number, labels (element, total H)."""
    G = nx.Graph()
    for a in mol.GetAtoms():
        G.add_node(a.GetAtomMapNum(), lab=(a.GetSymbol(), a.GetTotalNumHs()))
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())
    return G


def match_smiles_to_reactants(rmols, r0, r1, factor: float = FACTOR):
    """Which SMILES reactant is r0 / r1?  Returns (idx_for_r0, idx_for_r1) or None."""
    G0, _, _, _, _ = heavy_graph(*r0, factor)
    G1, _, _, _, _ = heavy_graph(*r1, factor)
    Gm = [mol_heavy_graph(m) for m in rmols]
    for i0, i1 in ((0, 1), (1, 0)):
        if nx.is_isomorphic(G0, Gm[i0], node_match=_nm) and nx.is_isomorphic(G1, Gm[i1], node_match=_nm):
            return i0, i1
    return None


@dataclass
class BondAudit:
    formed_ts: set                 # forming bonds as TS index pairs (best-realised mapping)
    d_formed: list                 # their lengths, sorted
    foreign: list                  # inter-fragment heavy pairs within covalent distance that are NOT forming bonds
    assign: tuple                  # (smiles idx for r0, smiles idx for r1)


def audit_forming_bonds(P: Partition, ts, r0, r1, rxn_smiles: str, factor: float = FACTOR,
                        max_iso: int = 200) -> BondAudit | None:
    """Map the SMILES-defined forming bonds onto the TS and look for 'foreign' covalent contacts.

    Symmetry-equivalent atoms give several SMILES->xyz isomorphisms; the one that
    minimises the sum of forming-bond lengths is taken as the physically realised one.
    """
    rmols, formed, _ = smiles_reactants_and_formed(rxn_smiles)
    assign = match_smiles_to_reactants(rmols, r0, r1, factor)
    if assign is None or len(formed) == 0:
        return None
    ts_syms, ts_xyz = ts
    adj, D = bond_matrix(ts_syms, ts_xyz, factor)
    G0, _, _, _, _ = heavy_graph(*r0, factor)
    G1, _, _, _, _ = heavy_graph(*r1, factor)
    Gm0, Gm1 = mol_heavy_graph(rmols[assign[0]]), mol_heavy_graph(rmols[assign[1]])
    iso0 = list(itertools.islice(GraphMatcher(G0, Gm0, node_match=_nm).isomorphisms_iter(), max_iso))
    iso1 = list(itertools.islice(GraphMatcher(G1, Gm1, node_match=_nm).isomorphisms_iter(), max_iso))
    best = None
    for m0 in iso0:                                   # m0: r0 heavy idx -> map number
        inv0 = {v: P.map0[k] for k, v in m0.items()}  # map number -> TS idx
        for m1 in iso1:
            inv = dict(inv0)
            inv.update({v: P.map1[k] for k, v in m1.items()})
            try:
                fp = {tuple(sorted((inv[a], inv[b]))) for a, b in formed}
            except KeyError:
                continue
            ds = sorted(D[i, j] for i, j in fp)
            if best is None or sum(ds) < sum(best[1]):
                best = (fp, ds)
    if best is None:
        return None
    fp, ds = best
    heavy = [i for i, s in enumerate(ts_syms) if s != "H"]
    foreign = [(i, j, float(D[i, j])) for i in heavy for j in heavy
               if i < j and ((i in P.A) != (j in P.A)) and adj[i, j] and (i, j) not in fp]
    return BondAudit(fp, [float(x) for x in ds], foreign, assign)


def dipole_smiles_index(rxn_smiles: str):
    """Which SMILES reactant is the 1,3-dipole?

    The new ring in the product is the smallest ring containing both forming bonds;
    the reactant that contributes 3 of its atoms is the dipole, the one contributing 2
    is the dipolarophile. Returns (dipole_idx, ring_map_numbers) or None if no 3+2 split.
    """
    from rdkit import Chem
    rmols, formed, _ = smiles_reactants_and_formed(rxn_smiles)
    pmol = Chem.MolFromSmiles(rxn_smiles.split(">>")[1])
    owner = {a.GetAtomMapNum(): k for k, m in enumerate(rmols) for a in m.GetAtoms()}
    idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in pmol.GetAtoms()}
    rings = [set(idx2map[i] for i in ring) for ring in pmol.GetRingInfo().AtomRings()]
    rings = [r for r in rings if all(a in r and b in r for a, b in formed)]
    if not rings:
        return None
    ring = min(rings, key=len)
    cnt = collections.Counter(owner[m] for m in ring)
    dip = [k for k, v in cnt.items() if v == 3]
    if len(ring) != 5 or len(dip) != 1:
        return None
    return dip[0], ring


def formal_charges(rmols, assign):
    from rdkit import Chem
    return Chem.GetFormalCharge(rmols[assign[0]]), Chem.GetFormalCharge(rmols[assign[1]])


def n_electrons(syms, charge: int) -> int:
    return sum(Z[s] for s in syms) - charge
