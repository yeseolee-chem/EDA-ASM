#!/usr/bin/env python3
"""GEO1b Phase D: records without G1 -> $GEO1B/data/records.pkl + results/D_dataset_report.json (D1 run env).
  python data.py                      build the records                    (sbatch ... jobs/cpu.slurm data.py)
  python data.py baseline [--force]   G2-raw baseline of the primary metric -> results/G2RAW_baseline.json; run after
                                      the build and BEFORE training (frozen: a changed value is a STOP unless --force)

Same as analysis/geo/data.py except (docs/specs/GEO1B.md 3-2):
  * no G1 input is read anywhere: every record has xyz_g1 = None and same = None (layout unchanged for train/model code);
  * each record with a G2 input gets map_rmsd_g2 = rmsd.mapped_heavy_rmsd(G2 TS, label TS) (heavy atoms, graph
    isomorphisms and mirror images; inf -> nan). It is used only for descriptive bins in the evaluation, never in training.
GEO1b decisions A-G2c and D0D1 (results/DECISIONS.md; judgment calls in results/DECISIONS_GB.md):
  * the G2 inputs are the G2c build roots GC.D1_G2_FINAL / GC.D0_G2_FINAL ($G2C/{d1,d0}/g2_final). Every used input
    records the run it came from: rec['g2c'] = g2c.json 'chosen' (orig / r1 / r2; None without a G2 input). Nothing is
    written when a .done folder has no valid g2c.json or the folders disagree with the G2c C_select.csv (STOP: wrong,
    partial or unfinished G2c root);
  * a D0 record gets core 'D0/<component>', fold (0-4) and twin_of (D1 rxn_id or None) from $GEO1B/d0/d0_folds.csv
    (d0_pairs.py build); the 5 D-1 overlaps keep overlap True, fold -9 and core 'D0/overlap' (in no set). Nothing is
    written when d0_folds.csv disagrees with the D0 labels, with d0_overlap_ids() or with the T1 fold of a twin;
  * GB-twin-m1: a D0 record also gets twin_kind (d0_twins.csv kind or None) and twin_m1 (a stereo-free D1 twin of T1
    fold -1: that twin trains in every fold, an accepted limitation, DECISIONS_PAIR P-twin-m1), counted in the D report
    (d0_twin_table) and carried into E_rxn_d0.csv. d0_twins.csv must match D0_pairs_report.json and d0_folds.csv;
  * GB-g2-count: per source, the G2c inputs read must equal the G2c TSs of C_select.csv, and used + element-order drops
    + inputs dropped with their record must equal the inputs read (GEO1B.md 3-1 G2 counts), else nothing is written.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

bond_types, spd_matrix = GC.bond_types, GC.spd_matrix
BASELINE_KEYS = ("n", "share", "median", "n_test")      # frozen values of results/G2RAW_baseline.json


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


def stop(problems):
    """print the STOP reasons (nothing is written; an earlier records.pkl / report stays as it was)."""
    head = problems[:50]
    more = f"\n  ... {len(problems)} problems in total" if len(problems) > len(head) else ""
    print("STOP (data.py):\n  " + "\n  ".join(map(str, head)) + more, file=sys.stderr)
    return 3


def load_g2c_select(bad):
    """{(src 'd1'|'d0', rxn_id): chosen run or ''} from the G2c C_select.csv (written last by select_g2c.py build)."""
    if not GC.G2C_SELECT.is_file():
        bad.append(f"{GC.G2C_SELECT} missing: the G2c build (select_g2c.py build) has not finished")
        return {}
    S = pd.read_csv(GC.G2C_SELECT, dtype=str, keep_default_na=False, usecols=["src", "rxn_id", "chosen"])
    sel = {}
    for s, r, c in zip(S.src, S.rxn_id, S.chosen):
        key = (s, int(float(r)))
        if key in sel:
            bad.append(f"C_select.csv: {key} listed twice")
        if c not in ("",) + GC.G2C_RUNS:
            bad.append(f"C_select.csv: {key} has chosen {c!r}")
        sel[key] = c
    return sel


def g2c_input(folder, src, rid, sel, bad):
    """(G2 TS (syms, xyz) or None, chosen run or None) of one G2c folder. Checks (decision A-G2c, GB-prov): the folder
    is .done iff C_select.csv names a chosen run, and its g2c.json names the same reaction, source and run."""
    g2 = GC.read_if_done(folder)
    c_sel = sel.get((src.lower(), int(rid)))
    if c_sel is None:
        bad.append(f"{src} {rid}: not in {GC.G2C_SELECT.name}")
        return g2, None
    if (g2 is not None) != (c_sel != ""):
        bad.append(f"{src} {rid}: C_select chosen {c_sel!r} but {folder} .done = {g2 is not None}")
        return g2, None
    if g2 is None:
        return None, None
    try:
        js = json.loads((Path(folder) / "g2c.json").read_text())
        ch, jsrc, jrid = js.get("chosen"), str(js.get("src")), int(js.get("rxn_id"))
    except (OSError, ValueError, TypeError) as e:
        bad.append(f"{src} {rid}: no valid g2c.json in {folder} ({type(e).__name__}: {e})")
        return g2, None
    if ch not in GC.G2C_RUNS or ch != c_sel or jsrc != src.lower() or jrid != int(rid):
        bad.append(f"{src} {rid}: g2c.json (rxn_id {jrid}, src {jsrc}, chosen {ch!r}) disagrees with C_select {c_sel!r}")
        return g2, None
    return g2, ch


def make_rec(src, rid, core, fold, lab, g2, A_idx, pairs, charge, overlap, log, g2c=None, twin_of=None,
             twin_kind=None, twin_m1=False):
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
               xyz_g2=None, xyz_g1=None, map_rmsd_g2=float("nan"),
               g2c=None, twin_of=None if twin_of is None else int(twin_of),     # A-G2c run of the used input / D0 twin
               twin_kind=twin_kind, twin_m1=bool(twin_m1))      # GB-twin-m1: d0_twins.csv kind / a T1 fold -1 D1 twin
    if g2 is None:
        return rec
    if list(g2[0]) != list(syms):
        log.append(f"{src} {rid}: g2 element order differs -> input dropped")
        return rec
    x = g2[1].astype(np.float64)
    al, r, refl = GC.kabsch_align(lx, x, reflect=True)
    B = bond_types(syms, x, pairs)
    rec.update(xyz_g2=x, lab_on_g2=al, rmsd_g2=r, refl_g2=refl, bond_g2=B, spd_g2=spd_matrix(B),
               map_rmsd_g2=map_rmsd(g2, (syms, lx)), g2c=g2c)
    return rec


def main():
    log, recs, bad = [], [], []
    for p in (GC.D1_G2_FINAL, GC.D0_G2_FINAL):                 # A-G2c: the G2c build roots, never the single-run G2
        if not p.is_dir():
            bad.append(f"G2c root {p} missing")
    sel = load_g2c_select(bad)
    folds0, fbad = GC.load_d0_folds()                           # D0D1: D0 folds / components / twins (d0_pairs.py)
    bad += fbad
    twins0, tbad = GC.load_d0_twins()                           # GB-twin-m1: every stereo-free D1 twin (d0_pairs.py)
    bad += tbad
    try:                                                        # DECISIONS_PAIR P-g2-gate: D0_pairs.csv anchors must
        prep = json.loads(GC.D0_PAIRS_REPORT.read_text())       # come from this very G2c selection, never the orig
    except (OSError, ValueError):                               # fallback of a preliminary d0_pairs.py run
        prep = {}
    p_sha = (prep.get("g2_info") or {}).get("c_select_sha256")
    if prep.get("g2_source") != "g2c" or not GC.G2C_SELECT.is_file() or p_sha != GC.sha256(GC.G2C_SELECT):
        bad.append(f"{GC.D0_PAIRS_REPORT.name}: g2_source {prep.get('g2_source')!r}, c_select_sha256 {p_sha!r} is not "
                   f"the current {GC.G2C_SELECT}: rerun d0_pairs.py build after select_g2c.py build")
    if bad:
        return stop(bad)
    T1 = GC.d1_core_table()                          # rxn_id -> (core, fold, None)
    meta1 = GC.load_meta(GC.D1_META_G2)
    acc = {s: Counter() for s in ("D1", "D0")}       # GB-g2-count: G2c inputs read / dropped with their record
    for d in GC.load_d1_labels():
        rid = int(d["rxn_id"])
        core, fold, _ = T1[rid]
        g2, ch = g2c_input(GC.D1_G2_FINAL / str(rid), "D1", rid, sel, bad)
        acc["D1"]["read"] += int(g2 is not None)
        if rid not in meta1:
            log.append(f"D1 {rid}: no A_idx in input_meta_d1_g2.csv -> dropped")
            acc["D1"]["record_dropped"] += int(g2 is not None)
            continue
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        r = make_rec("D1", rid, core, fold, lab, g2, meta1[rid],
                     GC.parse_pairs(d["formed_pairs_ts"]), int(d["charge1"]) + int(d["charge2"]), False, log, g2c=ch)
        if r is not None:
            recs.append(r)
        else:
            acc["D1"]["record_dropped"] += int(g2 is not None)
    meta0 = GC.load_meta(GC.D0_META)
    ov, ov_stats = d0_overlap_ids()
    if ov_stats["n_d1_parsed"] < 9000:
        print(f"STOP: only {ov_stats['n_d1_parsed']} D1 reaction SMILES parsed for the overlap check", file=sys.stderr)
        return 3
    L0 = GC.load_d0_labels()
    rid0 = {int(L["rxn_id"]) for L in L0}
    if set(folds0) != rid0:
        bad.append(f"d0_folds.csv rxn_ids != the {len(rid0)} D0 labels: missing {sorted(rid0 - set(folds0))[:10]}, "
                   f"extra {sorted(set(folds0) - rid0)[:10]}")
    ov_f = {rid for rid, f in folds0.items() if f["overlap"]}
    if ov_f != (ov & rid0):
        bad.append(f"d0_folds.csv overlaps {sorted(ov_f)} != d0_overlap_ids() {sorted(ov & rid0)}")
    for rid, f in sorted(folds0.items()):
        t = f["twin_of"]
        if t is not None and (t not in T1 or T1[t][1] != f["fold"]):
            bad.append(f"D0 {rid}: fold {f['fold']} but twin D1 {t} has T1 fold {T1[t][1] if t in T1 else 'none'}")
        tw = twins0.get(rid)                                    # GB-twin-m1: d0_twins.csv agrees with d0_folds.csv
        if f["overlap"] and tw is not None:
            bad.append(f"D0 {rid}: a D-1 overlap with a d0_twins.csv row (overlaps are outside the d0_pairs population)")
        elif not f["overlap"] and (None if tw is None else tw["twin_of"]) != t:
            bad.append(f"D0 {rid}: d0_folds.csv twin_of {t} != d0_twins.csv {None if tw is None else tw['twin_of']}")
    if set(twins0) - set(folds0):
        bad.append(f"d0_twins.csv lists {len(set(twins0) - set(folds0))} rxn_ids that are not in d0_folds.csv")
    for L in L0:
        rid = int(L["rxn_id"])
        f = folds0.get(rid)
        if f is None:
            continue                                            # reported above
        g2, ch = g2c_input(GC.D0_G2_FINAL / str(rid), "D0", rid, sel, bad)
        acc["D0"]["read"] += int(g2 is not None)
        core = "D0/overlap" if f["overlap"] else f"D0/{f['component']}"
        tw = None if f["overlap"] else twins0.get(rid)
        lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
        r = make_rec("D0", rid, core, f["fold"], lab, g2, meta0[rid], GC.parse_pairs(L["formed_pairs_ts"]),
                     int(L["charge1"]) + int(L["charge2"]), rid in ov, log, g2c=ch, twin_of=f["twin_of"],
                     twin_kind=None if tw is None else tw["kind"], twin_m1=tw is not None and tw["m1"])
        if r is not None:
            recs.append(r)
        else:
            acc["D0"]["record_dropped"] += int(g2 is not None)
    # GB-g2-count (GEO1B.md 3-1 G2 counts, enforced): per source, the G2c inputs read = the C_select G2c TSs, and each
    # one is used (xyz_g2) or dropped for element order or with its record (no A_idx / element outside ELEM)
    n_sel = {s: sum(1 for (src, _), c in sel.items() if src == s.lower() and c != "") for s in ("D1", "D0")}
    n_ord = {s: sum(m.startswith(f"{s} ") and "element order differs" in m for m in log) for s in ("D1", "D0")}
    n_use = {s: sum(r["src"] == s and r["xyz_g2"] is not None for r in recs) for s in ("D1", "D0")}
    for s in ("D1", "D0"):
        if acc[s]["read"] != n_sel[s] or n_use[s] + n_ord[s] + acc[s]["record_dropped"] != acc[s]["read"]:
            bad.append(f"{s}: G2c inputs read {acc[s]['read']}, C_select G2c TSs {n_sel[s]}, used {n_use[s]} + element "
                       f"order {n_ord[s]} + dropped with the record {acc[s]['record_dropped']} (must add up)")
    if bad:
        return stop(bad)
    out = GC.GEO / "data" / "records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    tmp.replace(out)
    df = pd.DataFrame([dict(src=r["src"], fold=r["fold"], g2=r["xyz_g2"] is not None, overlap=r["overlap"],
                            refl_g2=bool(r.get("refl_g2")), map_rmsd=r["map_rmsd_g2"], g2c=r["g2c"] or "",
                            twin=r["twin_of"] is not None, twin_kind=r["twin_kind"] or "", twin_m1=r["twin_m1"])
                       for r in recs])
    cnt = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}   # noqa: E731
    n_bad_order = sum(n_ord.values())

    def mq(g):
        v = g.map_rmsd[g.g2].to_numpy(float)
        f = v[np.isfinite(v)]
        return dict(n_g2=int(g.g2.sum()), n_nan=int(np.isnan(v).sum()),
                    q10_50_90=[float(x) for x in np.quantile(f, [0.1, 0.5, 0.9])] if len(f) else None,
                    share_le=dict((str(t), float((f <= t).mean())) for t in (0.25, 0.5, 0.75, 1.0)) if len(f) else None)
    g2_source = prep.get("g2_source")                           # = 'g2c' (P-g2-gate above)
    d0 = df[df.src == "D0"]
    m1g = d0.twin_m1 & d0.g2 & ~d0.overlap
    GC.write_json(GC.RES / "D_dataset_report.json", dict(
        n=len(recs), g1_used=False, overlap_check=ov_stats, log=log[:200], n_log=len(log),
        n_element_order_differs=n_bad_order,
        by_src={s: dict(n=int(len(g)), g2=int(g.g2.sum()), map_rmsd_g2=mq(g), g2c_chosen=cnt(g.g2c[g.g2]),
                        g2c_select_chosen=n_sel[s], n_g2c_read=int(acc[s]["read"]),
                        n_element_order_differs=int(n_ord[s]), n_g2_dropped_with_record=int(acc[s]["record_dropped"]))
                for s, g in df.groupby("src")},
        d0_overlap=int(df.overlap.sum()), d1_fold_counts=cnt(df[df.src == "D1"].fold),
        d0_fold_counts=cnt(d0.fold), d0_fold_counts_g2=cnt(d0.fold[d0.g2]), d0_twins=int(d0.twin.sum()),
        d0_twin_table=dict(path=str(GC.D0_TWINS), sha256=GC.sha256(GC.D0_TWINS), n_rows=len(twins0),
                           by_kind=cnt(d0.twin_kind[d0.twin_kind != ""]), n_m1=int(d0.twin_m1.sum()),
                           n_m1_test_g2=int(m1g.sum()), n_m1_test_g2_by_fold=cnt(d0.fold[m1g]),
                           note="m1 = a stereo-free D1 twin of T1 fold -1, which is in every final_f{k} training set "
                                "while this D0 reaction is tested in its own fold (DECISIONS_PAIR P-twin-m1, accepted "
                                "limitation; E_rxn_d0.csv column twin_m1)"),
        g2_input=dict(kind="G2c (decision A-G2c)", d1=str(GC.D1_G2_FINAL), d0=str(GC.D0_G2_FINAL),
                      c_select=str(GC.G2C_SELECT), c_select_sha256=GC.sha256(GC.G2C_SELECT)),
        d0_folds=dict(path=str(GC.D0_FOLDS), sha256=GC.sha256(GC.D0_FOLDS), d0_pairs_g2_source=g2_source),
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


def baseline(force=False):
    """A-G2c: the G2-raw baseline (G2c TS as is) of the primary metric = ALL para P_main pairs of the D1 test folds 0-4,
    computed from the records BEFORE training and frozen in results/G2RAW_baseline.json. tests.py U-8b and evaluate.py
    read it (and check its records_sha256). An existing file is kept when the values are unchanged; changed values are
    a STOP unless --force (decision GB-baseline). This is a freeze (same metrics code, same records as U-8b / E1), not
    an independent check: the independent regression is the single-run part of tests.py U-8b (GB-U8r)."""
    from metrics import evaluate_preds              # lazy: d0_pairs.py imports this module for d0_overlap_ids()
    rp, rec_path = GC.RES / "D_dataset_report.json", GC.GEO / "data" / "records.pkl"
    if not rp.is_file() or not rec_path.is_file():
        return stop([f"{rp} or {rec_path} missing: run data.py (build) first"])
    sha = GC.sha256(rec_path)
    if json.loads(rp.read_text()).get("sha256") != sha:
        return stop([f"{rec_path} sha256 differs from D_dataset_report.json (records changed after the D report)"])
    recs = GC.load_records()
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in (0, 1, 2, 3, 4) and r["xyz_g2"] is not None]
    m = evaluate_preds({r["uid"]: r["xyz_g2"] for r in test}, test, with_rmsd=False)
    per_fold = {}
    for k in range(5):
        tk = [r for r in test if r["fold"] == k]
        mk = evaluate_preds({r["uid"]: r["xyz_g2"] for r in tk}, tk, with_rmsd=False)
        per_fold[str(k)] = dict(n_test=len(tk), n=mk["n_primary"], share=mk["primary_share"],
                                median=mk["pair_pe_median_para"], rxn_form_abs_median=mk["rxn_form_abs_median"])
    new = dict(n=int(m["n_primary"]), share=float(m["primary_share"]), median=float(m["pair_pe_median_para"]),
               n_test=int(len(test)))
    out = dict(new, all_pairs_share=m["all_pairs_share"], n_pairs=m["n_pairs"],
               rxn_form_abs_median=m["rxn_form_abs_median"],
               n_test_by_g2c={str(k): int(v) for k, v in sorted(Counter(r["g2c"] for r in test).items())},
               per_fold=per_fold, records=str(rec_path), records_sha256=sha, g2_input="G2c (decision A-G2c)",
               metric="share of ALL para P_main pairs (D1 test folds 0-4) with pair error <= 0.005 A; median pair error",
               tolerances=dict(GC.G2RAW_TOL), single_run_reference=GC.G2RAW_SINGLE_RUN_REF,
               written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    old = GC.load_g2raw_baseline()
    if old is not None:
        same = all(old.get(k) == new[k] for k in BASELINE_KEYS)
        if same and old.get("records_sha256") == sha:
            print(f"G2RAW baseline unchanged: {new}", flush=True)
            return 0
        if not same and not force:
            return stop([f"{GC.G2RAW_BASELINE} is frozen: old {dict((k, old.get(k)) for k in BASELINE_KEYS)} != new "
                         f"{new}. Find what changed; '--force' only if no U / R output has used the old file"])
        out["previous"] = {k: old.get(k) for k in BASELINE_KEYS + ("records_sha256", "written")}
    GC.write_json(GC.G2RAW_BASELINE, out)
    print(f"G2RAW baseline -> {GC.G2RAW_BASELINE}: {new} (by run {out['n_test_by_g2c']})", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a in ([], ["build"]):
        sys.exit(main())
    if a[:1] == ["baseline"] and set(a[1:]) <= {"--force"}:
        sys.exit(baseline("--force" in a[1:]))
    raise SystemExit(__doc__)
