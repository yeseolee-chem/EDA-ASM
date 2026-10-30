#!/usr/bin/env python3
"""GEO1b Phase D: records without G1 -> $GEO1B/data/records.pkl + results/D_dataset_report.json (D1 run env).

Same as analysis/geo/data.py except (docs/specs/GEO1B.md 3-2):
  * no G1 input is read anywhere: every record has xyz_g1 = None and same = None (layout unchanged for train/model code);
  * each record with a G2 input gets map_rmsd_g2 = rmsd.mapped_heavy_rmsd(G2 TS, label TS) (heavy atoms, graph
    isomorphisms and mirror images; inf -> nan). It is used only for descriptive bins in the evaluation, never in training.
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

bond_types, spd_matrix = GC.bond_types, GC.spd_matrix


def canon_rxn(smi, stereo=False):
    """unchanged from analysis/geo/data.py (decision D-1)."""
    from rdkit import Chem
    out = []
    for side in smi.split(">>"):
        ms = []
        for s in side.split("."):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            m = Chem.RemoveHs(m)
            if not stereo:
                Chem.RemoveStereochemistry(m)
            ms.append(Chem.MolToSmiles(m))
        out.append(".".join(sorted(ms)))
    return ">>".join(out)


def d0_overlap_ids():
    """unchanged from analysis/geo/data.py."""
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False)
    s1 = man[man.kind == "D1"].rxn_smiles
    c1 = [canon_rxn(s) for s in s1]
    d1 = {c for c in c1 if c is not None}
    d1s = {c for c in (canon_rxn(s, stereo=True) for s in s1) if c is not None}
    smi = GC.load_d0_smiles()
    c0 = {rid: canon_rxn(s) for rid, s in smi.items()}
    c0s = {rid: canon_rxn(s, stereo=True) for rid, s in smi.items()}
    stats = dict(n_d1_smiles=int(len(s1)), n_d1_parsed=sum(c is not None for c in c1), n_d1_canon=len(d1),
                 n_d1_canon_with_stereo=len(d1s), n_d0_canon_none=sum(v is None for v in c0.values()),
                 n_overlap_with_stereo=sum(c is not None and c in d1s for c in c0s.values()))
    return {rid for rid, c in c0.items() if c is not None and c in d1}, stats


def map_rmsd(g2, lab):
    r = float(rm.mapped_heavy_rmsd(g2, lab))
    return r if math.isfinite(r) else float("nan")


def make_rec(src, rid, core, fold, lab, g2, A_idx, pairs, charge, overlap, log):
    syms, lx = lab
    n = len(syms)
    if any(s not in GC.TYPE for s in syms):
        log.append(f"{src} {rid}: element outside ELEM -> dropped")
        return None
    rec = dict(uid=f"{src}_{rid}", src=src, src_id=0 if src == "D1" else 1, rxn_id=int(rid), core=core, fold=int(fold),
               n=n, types=np.array([GC.TYPE[s] for s in syms], np.int64), lab=lx.astype(np.float64),
               frag=np.array([1 if i in set(A_idx) else 2 for i in range(n)], np.int64),
               form=np.array([1 if any(i in p for p in pairs) else 0 for i in range(n)], np.int64),
               pairs=np.array(pairs, np.int64), charge=int(charge), same=None, overlap=bool(overlap),
               xyz_g2=None, xyz_g1=None, map_rmsd_g2=float("nan"))
    if g2 is None:
        return rec
    if list(g2[0]) != list(syms):
        log.append(f"{src} {rid}: g2 element order differs -> input dropped")
        return rec
    x = g2[1].astype(np.float64)
    al, r, refl = GC.kabsch_align(lx, x, reflect=True)
    B = bond_types(syms, x, pairs)
    rec.update(xyz_g2=x, lab_on_g2=al, rmsd_g2=r, refl_g2=refl, bond_g2=B, spd_g2=spd_matrix(B),
               map_rmsd_g2=map_rmsd(g2, (syms, lx)))
    return rec


def main():
    log, recs = [], []
    T1 = GC.d1_core_table()                          # rxn_id -> (core, fold, None)
    meta1 = GC.load_meta(GC.D1_META_G2)
    for d in GC.load_d1_labels():
        rid = int(d["rxn_id"])
        core, fold, _ = T1[rid]
        if rid not in meta1:
            log.append(f"D1 {rid}: no A_idx in input_meta_d1_g2.csv -> dropped")
            continue
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        r = make_rec("D1", rid, core, fold, lab, GC.read_if_done(GC.D1_G2_FINAL / str(rid)), meta1[rid],
                     GC.parse_pairs(d["formed_pairs_ts"]), int(d["charge1"]) + int(d["charge2"]), False, log)
        if r is not None:
            recs.append(r)
    meta0 = GC.load_meta(GC.D0_META)
    ov, ov_stats = d0_overlap_ids()
    if ov_stats["n_d1_parsed"] < 9000:
        print(f"STOP: only {ov_stats['n_d1_parsed']} D1 reaction SMILES parsed for the overlap check", file=sys.stderr)
        return 3
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
        r = make_rec("D0", rid, "D0", -1, lab, GC.read_if_done(GC.D0G2 / "g2_final" / str(rid)), meta0[rid],
                     GC.parse_pairs(L["formed_pairs_ts"]), int(L["charge1"]) + int(L["charge2"]), rid in ov, log)
        if r is not None:
            recs.append(r)
    out = GC.GEO / "data" / "records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    tmp.replace(out)
    df = pd.DataFrame([dict(src=r["src"], fold=r["fold"], g2=r["xyz_g2"] is not None, overlap=r["overlap"],
                            refl_g2=bool(r.get("refl_g2")), map_rmsd=r["map_rmsd_g2"]) for r in recs])
    cnt = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}   # noqa: E731
    n_bad_order = sum("element order differs" in m for m in log)

    def mq(g):
        v = g.map_rmsd[g.g2].to_numpy(float)
        f = v[np.isfinite(v)]
        return dict(n_g2=int(g.g2.sum()), n_nan=int(np.isnan(v).sum()),
                    q10_50_90=[float(x) for x in np.quantile(f, [0.1, 0.5, 0.9])] if len(f) else None,
                    share_le=dict((str(t), float((f <= t).mean())) for t in (0.25, 0.5, 0.75, 1.0)) if len(f) else None)
    GC.write_json(GC.RES / "D_dataset_report.json", dict(
        n=len(recs), g1_used=False, overlap_check=ov_stats, log=log[:200], n_log=len(log),
        n_element_order_differs=n_bad_order,
        by_src={s: dict(n=int(len(g)), g2=int(g.g2.sum()), map_rmsd_g2=mq(g)) for s, g in df.groupby("src")},
        d0_overlap=int(df.overlap.sum()), d1_fold_counts=cnt(df[df.src == "D1"].fold),
        reflected=dict(g2=int(df.refl_g2.sum())), records=str(out), sha256=GC.sha256(out)))
    if n_bad_order > 0.01 * len(recs):
        print(f"STOP: element order differs for {n_bad_order} inputs (> 1 %)", file=sys.stderr)
        return 3
    d1 = df[(df.src == "D1") & df.g2]
    if d1.map_rmsd.isna().mean() > 0.05:
        print(f"STOP: mapped RMSD undefined (graph mismatch) for {d1.map_rmsd.isna().mean():.3f} of D1 G2 inputs",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
