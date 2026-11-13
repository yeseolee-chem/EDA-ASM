#!/usr/bin/env python3
"""GNN3R Phase A (docs/specs/GNN3R.md 3): row attributes of the 9,252 targets with the B' c3 fix -> $GR/A/rowattr_r.pkl.

    python analysis/gnn3r/rowattr_r.py            PYENV=reactot (analysis/d1_pair/jobs/py.slurm), CPU

build_rowattr.py runs on import, so DIP, PAT, any_bond_query, backbone_in_product, termini and c3_orientation below are
copied verbatim from analysis/d1_pair/build_rowattr.py
(sha256 81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4, lines 17-19, 46-79, 121-147).
c3_orientation_bp is the B' rule of spec 0-5 (the only new behaviour).

Outputs: $GR/A/rowattr_r.pkl (9,252 rows, the columns of rowattr.pkl.all.pkl, job_id order), results/A_c3_bp.csv,
results/A_report.json. An existing all-pass report with an unchanged pickle is not rebuilt. Exit 0 pass, 3 = STOP.
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
EXPECT_NEED, EXPECT_BP = 2667, 922                       # spec 3-2 (local counts)
EXPECT_EST_BP = 356                                       # spec 0-5: B' rows of the reference list with c3_est

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
    # 1. the 9,252 targets in job_id order
    r = allr[allr.job_id.isin(set(X.job_id))].sort_values("job_id", kind="mergesort").reset_index(drop=True)
    gates = {"A-1": dict(n_rows=int(len(r)), job_ids_equal_targets=set(r.job_id) == set(X.job_id),
                         job_id_unique=bool(r.job_id.is_unique),
                         ok=len(r) == C.N_TARGETS and set(r.job_id) == set(X.job_id) and r.job_id.is_unique)}
    # 2. rows whose c3 is computed (= build_rowattr.py `need`)
    need = r.dipole.isin(list(C.NEED_DIPOLES)) & (r.series == "R1") & r.skeleton.isin(list(C.NEED_SKELETONS))
    bp = need & (r.skeleton == "Bp")
    # 3. A-2: the copied c3_orientation reproduces the stored c3 on every need row
    again = r[need].apply(c3_orientation, axis=1).astype(str).to_numpy()
    stored = r.loc[need, "c3"].fillna("").astype(str)
    neq = again != stored.to_numpy()
    diff = pd.DataFrame(dict(job_id=r.loc[need, "job_id"].to_numpy()[neq], stored=stored.to_numpy()[neq],
                             recomputed=again[neq]))
    gates["A-2"] = dict(n_need=int(need.sum()), expect_need=EXPECT_NEED, n_differ=int(neq.sum()),
                        differ=diff.head(20).to_dict("records"), stored_counts=stored.value_counts().to_dict(),
                        ok=int(neq.sum()) == 0)
    # 4. the B' rule on need & Bp rows only
    new = r[bp].apply(c3_orientation_bp, axis=1)
    out = r.copy()
    out.loc[bp, "c3"] = new
    # A-3 (decision DEC-1, results/DECISIONS.md): the X = H reference rows (pos ref_direct) have no C3 substituent, so
    # c3 is '' for them by construction (c3_orientation and the reference list agree); every other Bp row: cis / trans
    bp_h = bp & (r.pos == "ref_direct")
    bp_x = bp & ~bp_h
    c3_new = out.loc[bp, "c3"].astype(str)
    c3_x, c3_h = out.loc[bp_x, "c3"].astype(str), out.loc[bp_h, "c3"].astype(str)
    gates["A-3"] = dict(n_bp=int(bp.sum()), expect_bp=EXPECT_BP, counts=c3_new.value_counts().to_dict(),
                        n_bp_substituted=int(bp_x.sum()), counts_substituted=c3_x.value_counts().to_dict(),
                        n_bp_h_reference=int(bp_h.sum()), h_reference_jobs=r.loc[bp_h, "job_id"].tolist(),
                        h_reference_c3=c3_h.value_counts().to_dict(), decision="DEC-1",
                        ok=bool(int(bp.sum()) == EXPECT_BP and c3_x.isin(["cis", "trans"]).all() and (c3_h == "").all()))
    est = pd.read_csv(C.REF_REACTIONS, dtype=str, keep_default_na=False)
    est = est[est.c3_est != ""]
    mine = out.set_index("job_id").c3.reindex(est.job_id).fillna("<absent>").astype(str).to_numpy()
    bad4 = est.loc[mine != est.c3_est.to_numpy(), ["job_id", "c3_est"]].assign(c3_server=mine[mine != est.c3_est.to_numpy()])
    gates["A-4"] = dict(n_est=int(len(est)), expect=EXPECT_EST_BP, est_counts=est.c3_est.value_counts().to_dict(),
                        n_differ=int(len(bad4)), differ=bad4.head(20).to_dict("records"),
                        ok=len(est) == EXPECT_EST_BP and len(bad4) == 0)
    others = [c for c in allr.columns if c != "c3"]
    orig = allr.set_index("job_id").loc[out.job_id].reset_index()[list(allr.columns)]   # rowattr.pkl.all.pkl rows
    keep = ~bp.to_numpy()
    gates["A-5"] = dict(columns_equal_all_pkl=list(out.columns) == list(allr.columns),
                        other_columns_equal=frames_equal(out[others], orig[others]),
                        c3_outside_bp_need_equal=frames_equal(out.loc[keep, ["c3"]], orig.loc[keep, ["c3"]]))
    gates["A-5"]["ok"] = all(gates["A-5"].values())
    dup_before, dup_after = ref_aryl_duplicates(r), ref_aryl_duplicates(out)
    gates["A-6"] = dict(ref_aryl_duplicated_in_F_before=dup_before, after=dup_after, ok=dup_after == 0)
    # outputs
    bp_rows = r[bp]
    table = pd.DataFrame(dict(job_id=bp_rows.job_id.to_numpy(), rxn_id=bp_rows.rxn_id.to_numpy(),
                              c3_old=r.loc[bp, "c3"].fillna("").astype(str).to_numpy(),
                              c3_new=c3_new.to_numpy(), ms4=bp_rows.apply(ms4_description, axis=1).to_numpy()))
    C.write_atomic(C.RES / "A_c3_bp.csv", table.to_csv(index=False))
    C.ROWATTR_R.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.ROWATTR_R.with_name(f".{C.ROWATTR_R.name}.{os.getpid()}.tmp")
    out.to_pickle(tmp)
    os.replace(tmp, C.ROWATTR_R)
    bad = [k for k, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/GNN3R.md 3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               versions=dict(rdkit=rdkit.__version__, pandas=pd.__version__, numpy=np.__version__,
                             python=sys.version.split()[0]),
               copied_from=dict(path="analysis/d1_pair/build_rowattr.py", sha256=s_src, want=BUILD_ROWATTR_SHA,
                                ok=s_src == BUILD_ROWATTR_SHA),
               inputs={str(p): C.sha256(p) for p in (C.ROWATTR_ALL, C.TARGETS, C.REF_REACTIONS)},
               counts=dict(rows=int(len(out)), need=int(need.sum()), bp_need=int(bp.sum()),
                           c3_need_after=out.loc[need, "c3"].fillna("").astype(str).value_counts().to_dict()),
               rule="B' (skeleton Bp) need rows: C4 with two exocyclic heavy atoms of which exactly one is a methyl "
                    "(carbon whose only heavy neighbour is C4) -> that methyl is the reference; else c3_orientation",
               gates=gates, outputs={str(C.ROWATTR_R): C.sha256(C.ROWATTR_R),
                                     str(C.RES / "A_c3_bp.csv"): C.sha256(C.RES / "A_c3_bp.csv")},
               all_pass=not bad and s_src == BUILD_ROWATTR_SHA, failed=bad)
    C.write_json(report_path, rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], failed=bad, counts=rep["counts"],
                          **{k: v["ok"] for k, v in gates.items()}), indent=1, default=str))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
