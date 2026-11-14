#!/usr/bin/env python3
"""GNN3R2 Phase A (docs/specs/GNN3R2.md 3): row attributes of the 9,252 targets with c3 re-judged by c3_core (D-12)
-> $GR2/A/rowattr_r.pkl.  Copied from analysis/gnn3r/rowattr_r.py (GNN3R Phase A); c3_core and main are new.

    python analysis/gnn3r2/rowattr_r.py           PYENV=reactot (analysis/d1_pair/jobs/py.slurm), CPU

build_rowattr.py runs on import, so DIP, PAT, any_bond_query, backbone_in_product, termini and c3_orientation below are
copied verbatim from analysis/d1_pair/build_rowattr.py
(sha256 81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4, lines 17-19, 46-79, 121-147).
c3_orientation_bp is the GNN3R B' rule (GNN3R spec 0-5); its methyl choice is reused by c3_core.

c3_core (GNN3R2 0-3): the atoms C3, xs (first atom of X), C4 and C4's reference substituent are chosen exactly as
c3_orientation does (B': the methyl rule). The X branch is cut back to xs and xs becomes CH3 (Kekulized product, other
stereo tags kept); the core is renumbered canonically (CanonicalRankAtoms) and embedded with seeds 7, 11, 23, 42, 97
(EmbedMolecule + MMFF); ring normal (5-atom SVD) and decision (u . v > 0 -> cis) as before. All 5 must agree, else '?'.
Rows without a C3 substituent (X = H) get ''. Identical canonical cores (SMILES + key atoms) are computed once.

Outputs: $GR2/A/rowattr_r.pkl (9,252 rows, the columns of rowattr.pkl.all.pkl, job_id order), results/A_c3_core.csv,
results/A_report.json. An existing all-pass report with unchanged outputs is not rebuilt. Exit 0 pass, 3 = STOP.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

RDLogger.DisableLog("rdApp.*")
BUILD_ROWATTR_SHA = "81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4"
EXPECT_NEED, EXPECT_BP = 2667, 922                       # GNN3R spec 3-2 (local counts)
EXPECT_EST_BP = 356                                       # GNN3R spec 0-5: B' rows of the reference list with c3_est
EXPECT_SUBSTITUTED, EXPECT_H = 2653, 14                   # GNN3R2 A-3: need rows with / without a C3 substituent
EXPECT_CORES = 72                                         # GNN3R2 0-3 (local count, reported)

# ---------------------------------------------------------------- copied verbatim from build_rowattr.py
DIP = {"nitrile oxide": "[#6]#[N+]-[O-]", "azide": "[#7]~[N+]~[#7-,#7]", "nitrone": "[#6]=[N+](-[#6])-[O-]",
       "azomethine ylide": "[#6-]-[N+]=[#6]", "carbonyl ylide": "[#6-]-[O+]=[#6]"}
PAT = {k: Chem.MolFromSmarts(v) for k, v in DIP.items()}


def any_bond_query(m):
    """same heavy atoms (element only), any bond order: finds the reactant skeleton inside the product"""
    m2 = Chem.RWMol(m)
    for at in m2.GetAtoms():
        at.SetFormalCharge(0); at.SetNumExplicitHs(0); at.SetNoImplicit(False)
    m2 = m2.GetMol(); m2.UpdatePropertyCache(False)
    prm = Chem.AdjustQueryParameters.NoAdjustments(); prm.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(m2, prm)


def backbone_in_product(row):
    """-> product mol, (pa, pc, pb) product indices of the dipole backbone, labels, 5-ring"""
    dip = Chem.MolFromSmiles(row.dip_smi); prod = Chem.MolFromSmiles(row.prod_smi)
    (a, c, b), lab = termini(dip, row.dipole)
    for mt in prod.GetSubstructMatches(any_bond_query(dip), useChirality=False, uniquify=False, maxMatches=5000):
        pa, pc, pb = mt[a], mt[c], mt[b]
        # every dipole atom keeps its hydrogens (HN3: NH end vs bare N end; nitrone/ylide: ring CH2 vs N-CH3)
        if any(prod.GetAtomWithIdx(mt[i]).GetTotalNumHs() != dip.GetAtomWithIdx(i).GetTotalNumHs() for i in range(dip.GetNumAtoms())):
            continue
        ring = [set(r) for r in prod.GetRingInfo().AtomRings() if len(r) == 5 and {pa, pc, pb} <= set(r)]
        if ring:
            return prod, (pa, pc, pb), {pa: lab[a], pb: lab[b]}, ring[0]
    return None


def termini(dip, name):
    """backbone atoms of the dipole (a, centre, b) and the terminus labels: anionic end -> 'O-'/'N-'/'C-', other -> symbol"""
    mt = dip.GetSubstructMatches(PAT[name])[0]
    if name == "nitrone":
        a, c, b = mt[0], mt[1], mt[3]
    else:
        a, c, b = mt[0], mt[1], mt[2]
    lab = lambda i: dip.GetAtomWithIdx(i).GetSymbol() + ("-" if dip.GetAtomWithIdx(i).GetFormalCharge() < 0 else "")
    return (a, c, b), {a: lab(a), b: lab(b)}


def c3_orientation(row):
    try:
        bb = backbone_in_product(row)
        if bb is None:
            return "?"
        prod, (pa, pc, pb), lab, ring = bb
        p = Chem.AddHs(prod)
        c3 = None
        for t in (pa, pb):
            at = p.GetAtomWithIdx(t)
            if at.GetSymbol() == "C" and any(n.GetAtomicNum() > 1 and n.GetIdx() not in ring for n in at.GetNeighbors()):
                c3 = at
        if c3 is None:
            return ""
        xs = [n for n in c3.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring][0]
        c4 = [n for n in c3.GetNeighbors() if n.GetIdx() in ring and n.GetIdx() not in (pa, pc, pb)][0]
        ms4 = [n for n in c4.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring]
        if len(ms4) != 1 or c4.GetHybridization() != Chem.HybridizationType.SP3:
            return ""
        if AllChem.EmbedMolecule(p, randomSeed=7) != 0:
            return "?"
        AllChem.MMFFOptimizeMolecule(p); pos = p.GetConformer().GetPositions()
        rp = pos[sorted(ring)]; nrm = np.linalg.svd(rp - rp.mean(0))[2][-1]
        u = np.dot(pos[xs.GetIdx()] - pos[c3.GetIdx()], nrm); v = np.dot(pos[ms4[0].GetIdx()] - pos[c4.GetIdx()], nrm)
        return "cis" if u * v > 0 else "trans"
    except Exception:
        return "?"
# ---------------------------------------------------------------- end of the verbatim copy


def _c4_and_ms4(row):
    """(p, ring, c3, xs, c4, ms4) exactly as c3_orientation finds them, or a c3 string when it stops before."""
    bb = backbone_in_product(row)
    if bb is None:
        return "?"
    prod, (pa, pc, pb), lab, ring = bb
    p = Chem.AddHs(prod)
    c3 = None
    for t in (pa, pb):
        at = p.GetAtomWithIdx(t)
        if at.GetSymbol() == "C" and any(n.GetAtomicNum() > 1 and n.GetIdx() not in ring for n in at.GetNeighbors()):
            c3 = at
    if c3 is None:
        return ""
    xs = [n for n in c3.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring][0]
    c4 = [n for n in c3.GetNeighbors() if n.GetIdx() in ring and n.GetIdx() not in (pa, pc, pb)][0]
    ms4 = [n for n in c4.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring]
    return p, ring, c3, xs, c4, ms4


def _methyls(c4, ms4):
    """the ms4 atoms that are a methyl: carbon whose only heavy neighbour is C4"""
    return [n for n in ms4 if n.GetAtomicNum() == 6
            and [q.GetIdx() for q in n.GetNeighbors() if q.GetAtomicNum() > 1] == [c4.GetIdx()]]


def c3_orientation_bp(row):
    """c3_orientation with the B' rule (spec 0-5): when C4 has two exocyclic heavy atoms and exactly one of them is a
    methyl, that methyl is C4's reference substituent. Embedding (seed 7, MMFF), ring normal (SVD) and the decision
    (u . v > 0 -> cis) are those of c3_orientation."""
    try:
        found = _c4_and_ms4(row)
        if isinstance(found, str):
            return found
        p, ring, c3, xs, c4, ms4 = found
        if len(ms4) == 2 and len(_methyls(c4, ms4)) == 1:
            ms4 = _methyls(c4, ms4)
        if len(ms4) != 1 or c4.GetHybridization() != Chem.HybridizationType.SP3:
            return ""
        if AllChem.EmbedMolecule(p, randomSeed=7) != 0:
            return "?"
        AllChem.MMFFOptimizeMolecule(p); pos = p.GetConformer().GetPositions()
        rp = pos[sorted(ring)]; nrm = np.linalg.svd(rp - rp.mean(0))[2][-1]
        u = np.dot(pos[xs.GetIdx()] - pos[c3.GetIdx()], nrm); v = np.dot(pos[ms4[0].GetIdx()] - pos[c4.GetIdx()], nrm)
        return "cis" if u * v > 0 else "trans"
    except Exception:
        return "?"


def ms4_description(row):
    """C4's exocyclic heavy atoms as 'element:heavy degree' (sorted), with '*' on the methyl the B' rule uses."""
    try:
        found = _c4_and_ms4(row)
        if isinstance(found, str):
            return f"(c3 {found!r})"
        p, ring, c3, xs, c4, ms4 = found
        me = {n.GetIdx() for n in _methyls(c4, ms4)} if len(ms4) == 2 and len(_methyls(c4, ms4)) == 1 else set()
        return ",".join(sorted(f"{n.GetSymbol()}:{sum(q.GetAtomicNum() > 1 for q in n.GetNeighbors())}"
                               + ("*" if n.GetIdx() in me else "") for n in ms4)) + f"|c4 {c4.GetHybridization()}"
    except Exception as e:                                                    # noqa: BLE001
        return f"(error {type(e).__name__})"


def ref_aryl_duplicates(r: pd.DataFrame) -> int:
    """rows with pos ref_aryl whose family key F is shared with another ref_aryl row (keep=False, F fillna(''))"""
    q = r.loc[r.pos == "ref_aryl", C.F].fillna("").astype(str)
    return int(q.duplicated(keep=False).sum())


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(
        a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True)) for c in a.columns)


# ---------------------------------------------------------------- GNN3R2: c3_core (D-12, spec 0-3)
_CORE_CACHE: dict = {}
_FLIP = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
         Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW}


def core_parts(row):
    """The CH3 core of one product and its key atoms, or a c3 string when the atom selection stops before.

    Returns '' (no C3 substituent: X = H), '?' (no backbone, C4 without one sp3 reference substituent, or an X branch
    that reaches the ring) or (core SMILES, canonical core mol, keys) with keys = c3, xs, c4, ms4 (atom index) and ring
    (5 sorted atom indices) in the canonical core.
    """
    bb = backbone_in_product(row)
    if bb is None:
        return "?"
    prod, (pa, pc, pb), lab, ring = bb
    p = Chem.AddHs(prod)                                     # atom choice exactly as c3_orientation (heavy indices kept)
    c3 = None
    for t in (pa, pb):
        at = p.GetAtomWithIdx(t)
        if at.GetSymbol() == "C" and any(n.GetAtomicNum() > 1 and n.GetIdx() not in ring for n in at.GetNeighbors()):
            c3 = at
    if c3 is None:
        return ""
    xs = [n for n in c3.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring][0]
    c4 = [n for n in c3.GetNeighbors() if n.GetIdx() in ring and n.GetIdx() not in (pa, pc, pb)][0]
    ms4 = [n for n in c4.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring]
    if row.skeleton == "Bp" and len(ms4) == 2 and len(_methyls(c4, ms4)) == 1:      # GNN3R B' rule
        ms4 = _methyls(c4, ms4)
    if len(ms4) != 1 or c4.GetHybridization() != Chem.HybridizationType.SP3:
        return "?"
    i3, ix, i4, im = c3.GetIdx(), xs.GetIdx(), c4.GetIdx(), ms4[0].GetIdx()
    branch, todo = set(), [ix]                               # X branch: atoms reached from xs without passing C3
    while todo:
        a = todo.pop()
        if a in branch:
            continue
        branch.add(a)
        todo += [n.GetIdx() for n in prod.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != i3]
    if branch & (set(ring) | {i3, i4, im}):
        return "?"
    m = Chem.RWMol(prod)
    Chem.Kekulize(m, clearAromaticFlags=True)
    for name, idx in (("c3", i3), ("xs", ix), ("c4", i4), ("ms4", im)):
        m.GetAtomWithIdx(idx).SetProp("gr2_key", name)
    for idx in ring:
        m.GetAtomWithIdx(idx).SetBoolProp("gr2_ring", True)
    x = m.GetAtomWithIdx(ix)                                 # xs -> sp3 CH3 (implicit hydrogens)
    x.SetAtomicNum(6); x.SetFormalCharge(0); x.SetIsAromatic(False); x.SetIsotope(0); x.SetNumRadicalElectrons(0)
    x.SetNumExplicitHs(0); x.SetNoImplicit(False); x.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    bond = m.GetBondBetweenAtoms(i3, ix)
    bond.SetBondType(Chem.BondType.SINGLE); bond.SetIsAromatic(False)
    for idx in sorted(branch - {ix}, reverse=True):
        m.RemoveAtom(idx)
    core = m.GetMol()
    Chem.SanitizeMol(core)
    ranks = list(Chem.CanonicalRankAtoms(core, breakTies=True))
    order = sorted(range(core.GetNumAtoms()), key=ranks.__getitem__)
    new_of = {old: new for new, old in enumerate(order)}
    canon = Chem.RenumberAtoms(core, order)
    Chem.AssignStereochemistry(canon, cleanIt=False, force=True)
    keys = {a.GetProp("gr2_key"): new_of[a.GetIdx()] for a in core.GetAtoms() if a.HasProp("gr2_key")}
    keys["ring"] = tuple(sorted(new_of[a.GetIdx()] for a in core.GetAtoms() if a.HasProp("gr2_ring")))
    if set(keys) != {"c3", "xs", "c4", "ms4", "ring"} or len(keys["ring"]) != 5:
        return "?"
    return Chem.MolToSmiles(core), canon, keys


def judge_core(canon, keys, seed: int) -> str:
    """one embedding of the core: ETKDG (EmbedMolecule) + MMFF, 5-atom ring normal (SVD), u . v > 0 -> cis"""
    p = Chem.AddHs(canon)
    if AllChem.EmbedMolecule(p, randomSeed=seed) != 0:
        return "?"
    AllChem.MMFFOptimizeMolecule(p); pos = p.GetConformer().GetPositions()
    rp = pos[list(keys["ring"])]; nrm = np.linalg.svd(rp - rp.mean(0))[2][-1]
    u = np.dot(pos[keys["xs"]] - pos[keys["c3"]], nrm); v = np.dot(pos[keys["ms4"]] - pos[keys["c4"]], nrm)
    return "cis" if u * v > 0 else "trans"


def c3_core(row) -> tuple[str, str, tuple]:
    """(c3, core SMILES, per-seed decisions): c3 = the common decision of the 5 seeds, '?' if they differ; '' for X = H"""
    try:
        parts = core_parts(row)
    except Exception as e:                                                    # noqa: BLE001
        return "?", f"(error {type(e).__name__}: {e})", ()
    if isinstance(parts, str):
        return parts, "", ()
    smi, canon, keys = parts
    key = (smi, keys["c3"], keys["xs"], keys["c4"], keys["ms4"], keys["ring"])
    if key not in _CORE_CACHE:
        try:
            _CORE_CACHE[key] = tuple(judge_core(canon, keys, s) for s in C.CORE_SEEDS)
        except Exception:                                                     # noqa: BLE001
            _CORE_CACHE[key] = ("?",) * len(C.CORE_SEEDS)
    seeds = _CORE_CACHE[key]
    return (seeds[0] if len(set(seeds)) == 1 and seeds[0] in ("cis", "trans") else "?"), smi, seeds


def mirror_smiles(smi: str) -> str:
    """canonical SMILES of the mirror image (every tetrahedral tag inverted; double-bond geometry is mirror-invariant)"""
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetChiralTag() in _FLIP:
            a.SetChiralTag(_FLIP[a.GetChiralTag()])
    # round trip: the parse cached stereo perception / CIP ranks of the original, so re-perceive before canonicalizing
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))


def main() -> int:
    report_path = C.RES / "A_report.json"
    if report_path.is_file() and C.ROWATTR_R.is_file():
        old = C.read_json(report_path)
        if old.get("all_pass") and all(Path(p).is_file() and C.sha256(p) == s for p, s in old.get("outputs", {}).items()):
            print(f"Phase A done ({report_path}); not rebuilt")
            return 0
    import rdkit
    t0 = time.time()
    s_src = C.sha256(C.D1PAIR_CODE / "build_rowattr.py")
    allr = pd.read_pickle(C.ROWATTR_ALL)
    X = pd.read_csv(C.TARGETS)
    # 1. the 9,252 targets in job_id order (= GNN3R)
    r = allr[allr.job_id.isin(set(X.job_id))].sort_values("job_id", kind="mergesort").reset_index(drop=True)
    gates = {"A-1": dict(n_rows=int(len(r)), job_ids_equal_targets=set(r.job_id) == set(X.job_id),
                         job_id_unique=bool(r.job_id.is_unique),
                         ok=len(r) == C.N_TARGETS and set(r.job_id) == set(X.job_id) and r.job_id.is_unique)}
    need = r.dipole.isin(list(C.NEED_DIPOLES)) & (r.series == "R1") & r.skeleton.isin(list(C.NEED_SKELETONS))
    # 2. A-2 (= GNN3R): the copied c3_orientation reproduces the stored c3 on every need row
    again = r[need].apply(c3_orientation, axis=1).astype(str).to_numpy()
    stored = r.loc[need, "c3"].fillna("").astype(str)
    neq = again != stored.to_numpy()
    diff = pd.DataFrame(dict(job_id=r.loc[need, "job_id"].to_numpy()[neq], stored=stored.to_numpy()[neq],
                             recomputed=again[neq]))
    gates["A-2"] = dict(n_need=int(need.sum()), expect_need=EXPECT_NEED, n_differ=int(neq.sum()),
                        differ=diff.head(20).to_dict("records"), stored_counts=stored.value_counts().to_dict(),
                        ok=int(neq.sum()) == 0)
    # 3. c3_core on every need row ('' = X = H); every other column untouched
    rows_need = r[need]
    res = [c3_core(x) for x in rows_need.itertuples(index=False)]
    c3v = np.array([t[0] for t in res], dtype=object)
    smis = np.array([t[1] for t in res], dtype=object)
    out = r.copy()
    out.loc[need, "c3"] = c3v
    g = pd.read_pickle(C.ROWATTR_GNN3R)
    c3_g = g.set_index("job_id").c3.reindex(rows_need.job_id).fillna("").astype(str).to_numpy()
    # A-3: substituted need rows -> cis / trans with 5 equal seeds; X = H rows -> ''; mirror cores agree
    sub = c3v != ""
    h_pos = rows_need.pos.to_numpy()[~sub]
    by_smi = {}
    for s_, v_ in zip(smis[sub], c3v[sub]):
        by_smi.setdefault(s_, set()).add(v_)
    same_core_conflicts = sorted(s_ for s_, v_ in by_smi.items() if len(v_) > 1)
    mirror_pairs, mirror_conflicts = 0, []
    for s_ in sorted(by_smi):
        m_ = mirror_smiles(s_) if not s_.startswith("(") else s_
        if m_ != s_ and m_ in by_smi and s_ < m_:
            mirror_pairs += 1
            if by_smi[s_] != by_smi[m_]:
                mirror_conflicts.append(dict(core=s_, mirror=m_, c3=sorted(by_smi[s_]), c3_mirror=sorted(by_smi[m_])))
    unanimous = [len(set(t[2])) == 1 for t, s_ in zip(res, sub) if s_]
    gates["A-3"] = dict(n_substituted=int(sub.sum()), expect_substituted=EXPECT_SUBSTITUTED,
                        counts=pd.Series(c3v).value_counts().to_dict(), n_seed_unanimous=int(sum(unanimous)),
                        seeds=list(C.CORE_SEEDS), n_h=int((~sub).sum()), expect_h=EXPECT_H,
                        h_rows_all_ref_direct=bool((h_pos == "ref_direct").all()), n_cores=len(by_smi),
                        expect_cores_local=EXPECT_CORES, n_mirror_core_pairs=mirror_pairs,
                        mirror_conflicts=mirror_conflicts, same_core_conflicts=same_core_conflicts[:20],
                        failed_rows=[dict(job_id=j, c3=v_, core=s_) for j, v_, s_ in
                                     zip(rows_need.job_id, c3v, smis) if v_ not in ("cis", "trans", "")][:20],
                        ok=bool(int(sub.sum()) == EXPECT_SUBSTITUTED and np.isin(c3v[sub], ["cis", "trans"]).all()
                                and all(unanimous) and int((~sub).sum()) == EXPECT_H
                                and (h_pos == "ref_direct").all() and not mirror_conflicts
                                and not same_core_conflicts))
    # A-4 (= GNN3R): the 356 reference-list c3_est rows
    est = pd.read_csv(C.REF_REACTIONS, dtype=str, keep_default_na=False)
    est = est[est.c3_est != ""]
    mine = out.set_index("job_id").c3.reindex(est.job_id).fillna("<absent>").astype(str).to_numpy()
    bad4 = est.loc[mine != est.c3_est.to_numpy(), ["job_id", "c3_est"]].assign(c3_server=mine[mine != est.c3_est.to_numpy()])
    gates["A-4"] = dict(n_est=int(len(est)), expect=EXPECT_EST_BP, est_counts=est.c3_est.value_counts().to_dict(),
                        n_differ=int(len(bad4)), differ=bad4.head(20).to_dict("records"),
                        ok=len(est) == EXPECT_EST_BP and len(bad4) == 0)
    # A-5: against the GNN3R rowattr_r.pkl - same rows / columns, only c3 differs and exactly on EXPECT_C3_CHANGED
    others = [c for c in g.columns if c != "c3"]
    go = g.c3.fillna("").astype(str).to_numpy()
    oo = out.c3.fillna("").astype(str).to_numpy()
    changed = {j: (a_, b_) for j, a_, b_ in zip(out.job_id, go, oo) if a_ != b_} \
        if list(g.job_id) == list(out.job_id) else {"<rows differ>": ("", "")}
    gates["A-5"] = dict(columns_equal=list(out.columns) == list(g.columns), job_ids_equal=list(g.job_id) == list(out.job_id),
                        other_columns_equal=frames_equal(out[others], g[others]) if list(out.columns) == list(g.columns)
                        else False, c3_changed={j: list(v) for j, v in changed.items()},
                        expected={j: list(v) for j, v in C.EXPECT_C3_CHANGED.items()})
    gates["A-5"]["ok"] = bool(gates["A-5"]["columns_equal"] and gates["A-5"]["job_ids_equal"]
                              and gates["A-5"]["other_columns_equal"] and changed == C.EXPECT_C3_CHANGED)
    # A-6 (= GNN3R)
    dup_after = ref_aryl_duplicates(out)
    gates["A-6"] = dict(ref_aryl_duplicated_in_F_gnn3r=ref_aryl_duplicates(g), after=dup_after, ok=dup_after == 0)
    # outputs
    table = pd.DataFrame(dict(job_id=rows_need.job_id.to_numpy(), rxn_id=rows_need.rxn_id.to_numpy(),
                              pos=rows_need.pos.to_numpy(), Xb=rows_need.Xb.to_numpy(), c3_gnn3r=c3_g, c3_core=c3v,
                              core_smiles=smis,
                              **{f"seed_{s_}": [t[2][i] if len(t[2]) == len(C.CORE_SEEDS) else "" for t in res]
                                 for i, s_ in enumerate(C.CORE_SEEDS)}))
    C.write_atomic(C.RES / "A_c3_core.csv", table.to_csv(index=False))
    C.ROWATTR_R.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.ROWATTR_R.with_name(f".{C.ROWATTR_R.name}.{os.getpid()}.tmp")
    out.to_pickle(tmp)
    os.replace(tmp, C.ROWATTR_R)
    bad = [k for k, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/GNN3R2.md 3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               versions=dict(rdkit=rdkit.__version__, pandas=pd.__version__, numpy=np.__version__,
                             python=sys.version.split()[0]),
               copied_from=dict(path="analysis/d1_pair/build_rowattr.py", sha256=s_src, want=BUILD_ROWATTR_SHA,
                                ok=s_src == BUILD_ROWATTR_SHA),
               inputs={str(p): C.sha256(p) for p in (C.ROWATTR_ALL, C.TARGETS, C.REF_REACTIONS, C.ROWATTR_GNN3R)},
               counts=dict(rows=int(len(out)), need=int(need.sum()), n_cores_computed=len(_CORE_CACHE),
                           c3_need_after=pd.Series(c3v).value_counts().to_dict()),
               rule="D-12 c3_core: X cut back to CH3 at xs (Kekulized product, other stereo kept), canonical atom order, "
                    "EmbedMolecule + MMFF with seeds 7, 11, 23, 42, 97, 5-atom ring normal, u . v > 0 -> cis; all 5 "
                    "equal else '?'; atom choice = c3_orientation (+ GNN3R B' methyl rule); X = H -> ''",
               gates=gates, outputs={str(C.ROWATTR_R): C.sha256(C.ROWATTR_R),
                                     str(C.RES / "A_c3_core.csv"): C.sha256(C.RES / "A_c3_core.csv")},
               all_pass=not bad and s_src == BUILD_ROWATTR_SHA, failed=bad)
    C.write_json(report_path, rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], failed=bad, counts=rep["counts"], c3_changed=gates["A-5"]["c3_changed"],
                          **{k: v["ok"] for k, v in gates.items()}), indent=1, default=str))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
