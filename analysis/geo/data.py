#!/usr/bin/env python3
"""Phase D: GEO-1 records -> $GEO/data/records.pkl + results/D_dataset_report.json (D1 run env)."""
from __future__ import annotations

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


def canon_rxn(smi):
    """map-free canonical reaction SMILES (explicit [H:n] atoms of the mapped D0 SMILES removed)."""
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
            ms.append(Chem.MolToSmiles(m))
        out.append(".".join(sorted(ms)))
    return ">>".join(out)


def d0_overlap_ids():
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False)
    s1 = man[man.kind == "D1"].rxn_smiles
    d1 = {c for c in (canon_rxn(s) for s in s1) if c is not None}
    smi = GC.load_d0_smiles()
    c0 = {rid: canon_rxn(s) for rid, s in smi.items()}
    stats = dict(n_d1_smiles=int(len(s1)), n_d1_canon=len(d1), n_d0_canon_none=sum(v is None for v in c0.values()))
    return {rid for rid, c in c0.items() if c is not None and c in d1}, stats


def make_rec(src, rid, core, fold, lab, g2, g1, A_idx, pairs, charge, same, overlap, log):
    syms, lx = lab
    n = len(syms)
    if any(s not in GC.TYPE for s in syms):
        log.append(f"{src} {rid}: element outside ELEM -> dropped")
        return None
    rec = dict(uid=f"{src}_{rid}", src=src, src_id=0 if src == "D1" else 1, rxn_id=int(rid), core=core, fold=int(fold),
               n=n, types=np.array([GC.TYPE[s] for s in syms], np.int64), lab=lx.astype(np.float64),
               frag=np.array([1 if i in set(A_idx) else 2 for i in range(n)], np.int64),
               form=np.array([1 if any(i in p for p in pairs) else 0 for i in range(n)], np.int64),
               pairs=np.array(pairs, np.int64), charge=int(charge), same=same, overlap=bool(overlap))
    for kind, inp in (("g2", g2), ("g1", g1)):
        rec[f"xyz_{kind}"] = None
        if inp is None:
            continue
        if list(inp[0]) != list(syms):
            log.append(f"{src} {rid}: {kind} element order differs -> input dropped")
            continue
        x = inp[1].astype(np.float64)
        al, r, refl = GC.kabsch_align(lx, x, reflect=True)
        B = bond_types(syms, x, pairs)
        rec.update({f"xyz_{kind}": x, f"lab_on_{kind}": al, f"rmsd_{kind}": r, f"refl_{kind}": refl,
                    f"bond_{kind}": B, f"spd_{kind}": spd_matrix(B)})
    return rec


def same_ts_check_d1(labs, T1, k=300):
    """rmsd.same_ts(G2, G1) as computed here vs C_compare.same_ts_g1 on k random D1 reactions."""
    rng = np.random.default_rng(20261026)
    cand = [d for d in labs if T1.get(int(d["rxn_id"]), (None, None, None))[2] is not None]
    pick = [cand[i] for i in rng.choice(len(cand), min(k, len(cand)), replace=False)]
    agree = []
    for d in pick:
        rid = int(d["rxn_id"])
        g2, g1 = GC.read_if_done(GC.D1_G2_FINAL / str(rid)), GC.read_if_done(GC.D1_G1_FINAL / str(rid))
        if g2 is None or g1 is None:
            continue
        pairs = GC.parse_pairs(d["formed_pairs_ts"])
        s, _, _ = rm.same_ts(g2, g1, GC.form_d(g2[1], pairs), GC.form_d(g1[1], pairs))
        agree.append(bool(s) == T1[rid][2])
    return float(np.mean(agree)), len(agree)


def main():
    log, recs = [], []
    T1 = GC.d1_core_table()                          # rxn_id -> (core, fold, same_ts_g1)
    labs1 = GC.load_d1_labels()
    agree, n_agree = same_ts_check_d1(labs1, T1)
    if agree < 0.99 or n_agree < 250:
        print(f"STOP: rmsd.same_ts reproduces C_compare only {agree:.3f} (n={n_agree})", file=sys.stderr)
        return 3
    meta1 = GC.load_meta(GC.D1_META_G2)
    for d in labs1:
        rid = int(d["rxn_id"])
        core, fold, same = T1[rid]
        if rid not in meta1:
            log.append(f"D1 {rid}: no A_idx in input_meta_d1_g2.csv -> dropped")
            continue
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        r = make_rec("D1", rid, core, fold, lab, GC.read_if_done(GC.D1_G2_FINAL / str(rid)),
                     GC.read_if_done(GC.D1_G1_FINAL / str(rid)), meta1[rid], GC.parse_pairs(d["formed_pairs_ts"]),
                     int(d["charge1"]) + int(d["charge2"]), same, False, log)
        if r is not None:
            recs.append(r)
    meta0 = GC.load_meta(GC.D0_META)
    ov, ov_stats = d0_overlap_ids()
    if ov_stats["n_d1_canon"] < 9000:
        print(f"STOP: only {ov_stats['n_d1_canon']} D1 reaction SMILES parsed for the overlap check", file=sys.stderr)
        return 3
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        pairs = GC.parse_pairs(L["formed_pairs_ts"])
        lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
        g2, g1 = GC.read_if_done(GC.D0G2 / "g2_final" / str(rid)), GC.read_if_done(GC.D0_G1_ROOT / str(rid))
        same = None
        if g2 is not None and g1 is not None and list(g2[0]) == list(g1[0]):
            same = bool(rm.same_ts(g2, g1, GC.form_d(g2[1], pairs), GC.form_d(g1[1], pairs))[0])
        r = make_rec("D0", rid, "D0", -1, lab, g2, g1, meta0[rid], pairs, int(L["charge1"]) + int(L["charge2"]),
                     same, rid in ov, log)
        if r is not None:
            recs.append(r)
    out = GC.GEO / "data" / "records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    tmp.replace(out)
    df = pd.DataFrame([dict(src=r["src"], fold=r["fold"], g2=r["xyz_g2"] is not None, g1=r["xyz_g1"] is not None,
                            same=str(r["same"]), overlap=r["overlap"], refl_g2=bool(r.get("refl_g2")),
                            refl_g1=bool(r.get("refl_g1"))) for r in recs])
    cnt = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}   # noqa: E731  JSON-safe keys
    n_bad_order = sum("element order differs" in m for m in log)
    GC.write_json(GC.RES / "D_dataset_report.json", dict(
        n=len(recs), same_ts_check=dict(agree=agree, n=n_agree), overlap_check=ov_stats, log=log[:200], n_log=len(log),
        n_element_order_differs=n_bad_order,
        by_src={s: dict(n=int(len(g)), g2=int(g.g2.sum()), g1=int(g.g1.sum()), same=cnt(g.same))
                for s, g in df.groupby("src")},
        d0_overlap=int(df.overlap.sum()), d1_fold_counts=cnt(df[df.src == "D1"].fold),
        reflected=dict(g2=int(df.refl_g2.sum()), g1=int(df.refl_g1.sum())),
        records=str(out), sha256=GC.sha256(out)))
    if n_bad_order > 0.01 * len(recs):
        print(f"STOP: element order differs for {n_bad_order} inputs (> 1 %)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
