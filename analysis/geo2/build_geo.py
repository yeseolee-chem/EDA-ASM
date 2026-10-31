#!/usr/bin/env python3
"""GEO2b Phase Z: the geo-rung structure folders $GEO2/geo_final/<rxn_id>/ of D1 and D0 from the GEO1b test predictions
(docs/specs/GEO2.md 4 read with GEO1B.md 4-3; decisions A-G2c and D0D1, design step 5). GPU env on a CPU partition
(analysis/geo1b/jobs/gpu.slurm; it reads the GEO1b pickles).

  python build_geo.py build     -> $GEO2/geo_final/<rxn_id>/{ts.xyz, rel1.xyz, rel2.xyz, result.json, .done}
                                   + results/Z_status.csv + results/Z_report.json
                                   exit 0 = done; 2 = not ready (an upstream file missing: GEO1b pickles, E_rxn*.csv,
                                   D_dataset_report.json, d0 folds / report, G2c C_select.csv; nothing written);
                                   3 = STOP (spec 4-3)
Targets: D1 = the 9,266 rows of labels_d1_g2.json (uid D1_<rid>, rid >= 1000001, label TS d1_jobs/<job_id>/ts.xyz);
D0 = the status-ok rows of labels_all.json without the 5 D-1 overlaps of d0_folds.csv (uid D0_<rid>, rid 0-5973,
job_id D0G2_<rid:04d>, label TS = the Coley profile TS full_dataset_profiles/<rid>/<ts_file>, as GEO1b data.py).
D1 and D0 rxn ids are disjoint, so both datasets share $GEO2/geo_final/<rxn_id>/.
Each reaction gets exactly one TS: the 3-seed mean of the GEO1b 'final' model of the fold whose test set holds it
(folds 0-4: final_f<k>_s<seed>.pkl = D1 T1 fold k + D0 fold k; D1 T1 fold -1: final_fm1_s<seed>.pkl of train_m1.py;
D0 has no fold -1). rel1 / rel2 = byte copies of the G2c folder whose TS was the GEO1b input (decision A-G2c). A
structure that fails the sanity rules (4-2) is replaced by that G2c TS (deployable fallback, flagged).
Gates (STOP), D1 and D0 checked separately: forming-distance errors of the mean coordinates = GEO1b E_rxn.csv (D1) /
E_rxn_d0.csv (D0) to 1e-6 A, and the file rows = exactly the reactions built from folds 0-4 (D1 fold -1 = fm1 is in no
file); fallback share <= 2 %. Also STOP: a reaction predicted by two folds, a D0 prediction in another fold than
d0_folds.csv (checked before any write), a prediction that is no target, G2c folders that disagree with C_select.csv /
g2c.json, a C_select.csv other than the one GEO1b trained on (D_dataset_report.json g2_input.c_select_sha256), a .done
geo_final folder of a target that is not built now (stale).
Idempotent: an existing .done folder whose sources are unchanged is kept. results/Z_report.json is first rewritten with
stop true + running true and gets its final content (running false, complete true) last; a STOP inside the run rewrites
it with stop true, running false and the reason. A killed run leaves running true (readers: exit 2, resubmit Z), so a
killed or stopped run never leaves a passing report.
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo2_common as C  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
MIN_CONTACT = 0.60          # d_ij >= 0.60 (r_i + r_j) for every pair
BOND_IN, BOND_OUT = 1.10, 1.40   # G2 bond (d < 1.10 sum r) must stay < 1.40 sum r; G2 non-bond (> 1.40) stays > 1.10
FORM_RANGE = (1.4, 4.5)     # forming distances (A); 4.5 = GEO1 decision I-1 (11 asynchronous label TSs 4.02-4.36 A)
FALLBACK_MAX = 0.02         # spec 4-3, per dataset (design step 5)
E_RXN_TOL = 1e-6            # A, forming-distance errors vs GEO1b E_rxn.csv / E_rxn_d0.csv
SRCS = ("D1", "D0")
E_RXN_FILE = {"D1": "E_rxn.csv", "D0": "E_rxn_d0.csv"}      # GEO1b evaluate.py (D0: GB-eval-d0, same columns)
Z_FIELDS = ["rxn_id", "job_id", "src", "status", "source", "fallback", "reason", "g2c"]
D_REPORT = C.GEO1_RES / "D_dataset_report.json"            # GEO1b data.py: g2_input.c_select_sha256 (the G2c it read)
Z_REPORT = C.RES / "Z_report.json"


def upstream_missing():
    """GEO1b / G2c files Z needs that do not exist yet (-> exit 2, nothing written; they arrive with the GEO1b runs, the
    geo1b -> geo2b merge M1 and the G2c build)."""
    need = [C.GEO1 / "preds" / f"final_{t}_s{s}.pkl" for t in [f"f{k}" for k in FOLDS] + ["fm1"] for s in C.SEEDS]
    need += [C.GEO1_RES / f for f in E_RXN_FILE.values()] + [D_REPORT, C.D0_FOLDS, C.D0_PAIRS_REPORT, C.G2C_SELECT]
    return [str(p) for p in need if not p.is_file()]


def c_select_vs_geo1b():
    """DECISIONS_ZE 연결 S-6: the G2c selection Z reads (C_select.csv) = the one GEO1b data.py read (D_dataset_report.json
    g2_input.c_select_sha256), else STOP (structures and GEO1b inputs from different G2c builds)."""
    want = (json.loads(D_REPORT.read_text()).get("g2_input") or {}).get("c_select_sha256")
    got = C.sha256(C.G2C_SELECT)
    if want != got:
        C.stop(f"{C.G2C_SELECT} sha256 {got[:12]} != GEO1b {D_REPORT.name} g2_input.c_select_sha256 {str(want)[:12]} "
               f"(the G2c selection changed after GEO1b data.py read it)")
    return dict(d_report=str(D_REPORT), d_report_sha256=C.sha256(D_REPORT), c_select_sha256=got, equal=True)


def load_preds():
    """uid -> (mean xyz, source tag, pkl sha256 list). Folds 0-4 and fold -1; seed sets must agree."""
    out = {}
    files = {f"f{k}": [C.GEO1 / "preds" / f"final_f{k}_s{s}.pkl" for s in C.SEEDS] for k in FOLDS}
    files["fm1"] = [C.GEO1 / "preds" / f"final_fm1_s{s}.pkl" for s in C.SEEDS]
    for tag, fs in files.items():
        miss = [str(f) for f in fs if not f.is_file()]
        if miss:
            C.stop(f"GEO1 prediction files missing: {miss}")
        parts = []
        for f in fs:
            with open(f, "rb") as fh:
                parts.append(pickle.load(fh))
        if any(set(p) != set(parts[0]) for p in parts):
            C.stop(f"{tag}: the seed files hold different reaction sets")
        shas = [C.sha256(f) for f in fs]
        for u in parts[0]:
            if u in out:
                C.stop(f"{u} predicted by two folds ({out[u][1]} and {tag})")
            out[u] = (np.mean([np.asarray(p[u], float) for p in parts], axis=0), tag, shas)
    return out


def sanity(syms, X, Xg2, pairs):
    """-> (ok, reason, info). X = geo TS, Xg2 = G2 TS (same atom order = label order)."""
    if X.shape != Xg2.shape or not np.isfinite(X).all():
        return False, "shape_or_nonfinite", {}
    r = np.array([C.COV[s] for s in syms])
    S = r[:, None] + r[None, :]
    D, Dg = C.dmat(X), C.dmat(Xg2)
    iu = np.triu_indices(len(syms), 1)
    ratio = (D / S)[iu]
    info = dict(min_contact_ratio=float(ratio.min()))
    if ratio.min() < MIN_CONTACT:
        return False, "close_contact", info
    fp = {tuple(p) for p in pairs}
    mask = np.array([(i, j) not in fp for i, j in zip(*iu)])
    rg, rx = (Dg / S)[iu][mask], (D / S)[iu][mask]
    broke = int(((rg < BOND_IN) & (rx > BOND_OUT)).sum())
    made = int(((rg > BOND_OUT) & (rx < BOND_IN)).sum())
    info.update(n_bonds_broken=broke, n_bonds_made=made)
    if broke or made:
        return False, "bond_graph_changed", info
    d = C.form_d(X, pairs)
    info["d_form"] = d.tolist()
    if d.min() < FORM_RANGE[0] or d.max() > FORM_RANGE[1]:
        return False, "forming_distance_range", info
    return True, "", info


def e_sorted(X, lab, pairs):
    dl = C.form_d(lab, pairs)
    o = np.argsort(dl)
    return (C.form_d(X, pairs) - dl)[o]


# ---------------------------------------------------------------- GEO2b inputs (A-G2c, D0D1)
def load_g2c_select():
    """{(src 'd1'|'d0', rxn_id): chosen run or ''} of the G2c C_select.csv (select_g2c.py build writes it after the
    folders, so it marks a finished build)."""
    if not C.G2C_SELECT.is_file():
        C.stop(f"{C.G2C_SELECT} missing: the G2c build (select_g2c.py build) has not finished (decision A-G2c)")
    sel = {}
    with open(C.G2C_SELECT, newline="") as f:
        for r in csv.DictReader(f):
            key, ch = (str(r["src"]), int(float(r["rxn_id"]))), str(r.get("chosen") or "")
            if key in sel or ch not in ("",) + C.G2C_RUNS:
                C.stop(f"{C.G2C_SELECT.name}: {key} listed twice or chosen {ch!r}")
            sel[key] = ch
    return sel


def g2c_state(folder, src, rid, sel):
    """A-G2c provenance of one G2c folder (= GEO1b data.py g2c_input): .done iff C_select names a chosen run, and its
    g2c.json names the same reaction, source and run. -> (marker, chosen run or ''); a disagreement = STOP."""
    mk, ch = C.marker(folder), sel.get((src.lower(), rid))
    if ch is None:
        C.stop(f"{src} {rid}: not in {C.G2C_SELECT.name}")
    if (mk == ".done") != (ch != ""):
        C.stop(f"{src} {rid}: {C.G2C_SELECT.name} chosen {ch!r} but {folder} marker {mk} (partial or changed G2c root)")
    if mk != ".done":
        return mk, ""
    try:
        js = json.loads((folder / "g2c.json").read_text())
        ok = js.get("chosen") == ch and str(js.get("src")) == src.lower() and int(js.get("rxn_id")) == rid
    except (OSError, ValueError, TypeError):
        ok = False
    if not ok or not all((folder / f).is_file() for f in ("ts.xyz", "rel1.xyz", "rel2.xyz")):
        C.stop(f"{src} {rid}: {folder} has no valid g2c.json for chosen {ch!r} or lacks ts / rel1 / rel2")
    return mk, ch


def load_d0_folds():
    """$GEO1B/d0/d0_folds.csv (d0_pairs.py) -> {rxn_id: dict(fold, overlap, component)}; reader rule P-stop
    (D0_pairs_report.json stop false and its outputs sha256 = the file) and the GEO1b layout (overlap <=> fold -9)."""
    why = C.report_output_ok(C.D0_PAIRS_REPORT, C.D0_FOLDS)
    if why:
        C.stop(f"D0 folds: {why}")
    out = {}
    with open(C.D0_FOLDS, newline="") as f:
        for r in csv.DictReader(f):
            rid, fold, ov = int(r["rxn_id"]), int(r["fold"]), str(r["overlap"])
            bad = rid in out or ov not in ("True", "False") or (ov == "True") != (fold == C.D0_FOLD_OVERLAP) \
                or (ov == "False" and fold not in FOLDS)
            if bad:
                C.stop(f"{C.D0_FOLDS.name} rxn {rid}: fold {fold}, overlap {ov!r} (or a duplicate row)")
            out[rid] = dict(fold=fold, overlap=ov == "True", component=r.get("component", ""))
    return out


def load_targets(d0f):
    """[target dict(src, rid, uid, job, lab, pairs, g2, fold)] (D1 in label order, then D0 by rxn_id), n D0 overlaps."""
    T = []
    for d in json.loads(C.D1_LABELS_G2.read_text()):                    # 9,266 D1 targets (job_id, rxn_id, pairs)
        rid = int(d["rxn_id"])
        T.append(dict(src="D1", rid=rid, uid=f"D1_{rid}", job=str(d["job_id"]), lab=C.D1_JOBS / d["job_id"] / "ts.xyz",
                      pairs=d["formed_pairs_ts"], g2=C.G2_FINAL / str(rid), fold=None))
    L0 = sorted((d for d in json.loads(C.D0_LABELS.read_text()) if int(d["rxn_id"]) not in C.D0_EXCLUDE),
                key=lambda d: int(d["rxn_id"]))
    if any(d.get("status") != "ok" for d in L0):
        C.stop(f"{C.D0_LABELS}: a D0 label outside D0_EXCLUDE is not status ok")
    ids0 = {int(d["rxn_id"]) for d in L0}
    if ids0 != set(d0f):
        C.stop(f"{C.D0_FOLDS.name} rxn ids != the {len(ids0)} status-ok D0 labels (missing "
               f"{sorted(ids0 - set(d0f))[:5]}, extra {sorted(set(d0f) - ids0)[:5]})")
    n_ov = 0
    for d in L0:
        rid = int(d["rxn_id"])
        if d0f[rid]["overlap"]:                                         # GEO1 rule D-1: in no set (fold -9)
            n_ov += 1
            continue
        T.append(dict(src="D0", rid=rid, uid=f"D0_{rid}", job=C.d0_job_id(rid), lab=C.D0_PROF / str(rid) / d["ts_file"],
                      pairs=d["formed_pairs_ts"], g2=C.G2_FINAL_D0 / str(rid), fold=int(d0f[rid]["fold"])))
    rids = Counter(t["rid"] for t in T)
    if max(rids.values()) > 1:
        C.stop(f"rxn ids shared by D1 and D0 (geo_final/<rxn_id> would collide): {[r for r, n in rids.items() if n > 1][:5]}")
    return T, n_ov


def read_erxn(src):
    p = C.GEO1_RES / E_RXN_FILE[src]
    if not p.is_file():
        C.stop(f"{p} missing (GEO1b Phase E1 evaluate.py must be finished)")
    out = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["rxn_id"])] = (float(row["e_short_model"]), float(row["e_long_model"]))
    return p, out


# ---------------------------------------------------------------- build
def build():
    t0 = time.time()
    d0f = load_d0_folds()
    T, n_ov = load_targets(d0f)
    sel = load_g2c_select()
    g2c_geo1b = c_select_vs_geo1b()                                      # before any write
    prov = {t["uid"]: g2c_state(t["g2"], t["src"], t["rid"], sel) for t in T}          # all checked before any write
    P = load_preds()
    stray = sorted(set(P) - {t["uid"] for t in T})
    if stray:
        C.stop(f"{len(stray)} GEO1b predictions are not GEO2b targets (stale pickles or other labels): {stray[:5]}")
    for t in T:                                                          # D0D1: final_f{k} predicts D0 fold k only
        if t["src"] == "D0" and t["uid"] in P and P[t["uid"]][1] != f"f{t['fold']}":   # (before any folder write)
            C.stop(f"{t['uid']}: predicted by {P[t['uid']][1]} but its d0_folds.csv fold is {t['fold']}")
    erxn = {s: read_erxn(s) for s in SRCS}
    rows = []
    acc = {s: dict(n_fb=0, e_dev=0.0, n_e=0, n_built_not_in_e=0, ids_f04=set()) for s in SRCS}
    for t in T:
        src, rid, T_, uid = t["src"], t["rid"], t["job"], t["uid"]
        g2 = t["g2"]
        mk, ch = prov[uid]
        row = dict(rxn_id=rid, job_id=T_, src=src, status="", source="", fallback=False, reason="", g2c=ch)
        if mk != ".done":
            row.update(status="no_g2")
            rows.append(row)
            continue
        if uid not in P:
            row.update(status="no_geo1_prediction")
            rows.append(row)
            continue
        X, tag, shas = P[uid]
        syms_l, lab = C.read_xyz(t["lab"])
        syms_g, Xg2 = C.read_xyz(g2 / "ts.xyz")
        if list(syms_g) != list(syms_l):
            C.stop(f"{src} {rid}: G2c ts.xyz element order != label")
        pairs = C.parse_pairs(t["pairs"])
        a = acc[src]
        if tag != "fm1":                                                 # folds 0-4: must be a row of the E_rxn file
            a["ids_f04"].add(rid)
        if rid in erxn[src][1]:                                          # consistency with the GEO1b evaluation
            dev = float(np.abs(e_sorted(X, lab, pairs) - np.array(erxn[src][1][rid])).max())
            a["e_dev"], a["n_e"] = max(a["e_dev"], dev), a["n_e"] + 1
        else:
            a["n_built_not_in_e"] += 1                                   # D1 fold -1 (no GEO1b test fold): expected
        ok, reason, info = sanity(syms_l, X, Xg2, pairs)
        Xuse = X if ok else Xg2
        a["n_fb"] += 0 if ok else 1
        out = C.GEO_FINAL / str(rid)
        src_sha = {"g2_ts": C.sha256(g2 / "ts.xyz"), "g2_rel1": C.sha256(g2 / "rel1.xyz"),
                   "g2_rel2": C.sha256(g2 / "rel2.xyz"), "g2c_chosen": ch, "pkl": shas}
        if C.marker(out) == ".done":
            old = json.loads((out / "result.json").read_text())
            if old.get("sources") == src_sha and old.get("fallback") == (not ok):
                row.update(status="kept", source=tag, fallback=not ok, reason=reason)
                rows.append(row)
                continue
            C.stop(f"{out} exists with other sources: move it aside explicitly (never automatic)")
        tmp = C.GEO_FINAL / f"{rid}.tmp"
        C.assert_path(tmp)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        (tmp / "ts.xyz").write_text(C.xyz_text(syms_l, Xuse, f"geo TS {src} rxn {rid} "
                                               f"({'G2c fallback: ' + reason if not ok else 'GEO1b ' + tag + ' 3-seed mean'})"))
        for k in (1, 2):
            shutil.copyfile(g2 / f"rel{k}.xyz", tmp / f"rel{k}.xyz")
        heavy = [i for i, s in enumerate(syms_l) if s != "H"]
        res = dict(rxn_id=rid, job_id=T_, src=src, uid=uid, geom="geo", source=f"GEO1b {tag} seeds {list(C.SEEDS)} mean",
                   fold_tag=tag, d0_fold=t["fold"], g2c_chosen=ch, g2_folder=str(g2), label_ts=str(t["lab"]),
                   fallback=not ok, fallback_reason=reason, sanity=info, sources=src_sha,
                   formed_pairs_label=[list(p) for p in pairs], d_form_geo=C.form_d(X, pairs).tolist(),
                   d_form_used=C.form_d(Xuse, pairs).tolist(), d_form_g2=C.form_d(Xg2, pairs).tolist(),
                   d_form_label=C.form_d(lab, pairs).tolist(),
                   rmsd_heavy_label_geo=C.kabsch_rmsd(X[heavy], lab[heavy]),
                   rmsd_heavy_label_g2=C.kabsch_rmsd(Xg2[heavy], lab[heavy]),
                   status="ok", written=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (tmp / "result.json").write_text(json.dumps(res, indent=1, default=C._jsonable))
        (tmp / ".done").write_text(res["written"] + "\n")
        if out.exists():
            shutil.rmtree(C.assert_path(out))
        tmp.replace(out)
        row.update(status="ok", source=tag, fallback=not ok, reason=reason)
        rows.append(row)
    rows += [dict(rxn_id=rid, job_id=C.d0_job_id(rid), src="D0", status="d0_overlap_excluded", source="", fallback=False,
                  reason="GEO1 rule D-1 (fold -9, in no set)", g2c="") for rid, f in sorted(d0f.items()) if f["overlap"]]
    stops, by_src = [], {}
    stale = sorted(r["rxn_id"] for r in rows if r["status"] not in ("ok", "kept")
                   and C.marker(C.GEO_FINAL / str(r["rxn_id"])) == ".done")
    if stale:                                                            # feat_geo would take them as geo inputs
        stops.append(f"{len(stale)} geo_final folders with .done for targets that are not built now (no G2c, no GEO1b "
                     f"prediction or D-1 overlap): {stale[:5]}; move them aside explicitly (never automatic)")
    for s in SRCS:
        rs = [r for r in rows if r["src"] == s and r["status"] != "d0_overlap_excluded"]
        n_ok = sum(r["status"] in ("ok", "kept") for r in rs)
        a = acc[s]
        share = a["n_fb"] / n_ok if n_ok else None
        if a["n_e"] == 0 or a["e_dev"] > E_RXN_TOL:
            stops.append(f"{s}: forming-distance errors differ from GEO1b {E_RXN_FILE[s]} (max {a['e_dev']:.2e} A over "
                         f"{a['n_e']} rxns)")
        f_ids = set(erxn[s][1])                                          # the file = the GEO1b evaluation of these folds
        miss, extra = sorted(a["ids_f04"] - f_ids), sorted(f_ids - a["ids_f04"])
        if miss or extra:
            stops.append(f"{s}: {E_RXN_FILE[s]} rows != the reactions built from final_f0..4 ({len(miss)} built but not in "
                         f"the file, e.g. {miss[:5]}; {len(extra)} in the file but not built from folds 0-4, e.g. "
                         f"{extra[:5]}): stale GEO1b evaluation, pickles or G2c root")
        if n_ok and share > FALLBACK_MAX:
            stops.append(f"{s}: fallback rate {share:.3f} > {FALLBACK_MAX}")
        by_src[s] = dict(n_targets=len(rs), n_folders=n_ok, by_status=dict(Counter(r["status"] for r in rs)),
                         by_source=dict(Counter(r["source"] for r in rs if r["source"])), n_fallback=a["n_fb"],
                         fallback_share=share, fallback_max=FALLBACK_MAX,
                         fallback_reasons=dict(Counter(r["reason"] for r in rs if r["fallback"])),
                         g2c_chosen=dict(Counter(r["g2c"] for r in rs if r["status"] in ("ok", "kept"))),
                         e_rxn_check=dict(file=str(C.GEO1_RES / E_RXN_FILE[s]), n=a["n_e"], max_abs_dev_A=a["e_dev"],
                                          tol=E_RXN_TOL, n_built_not_in_file=a["n_built_not_in_e"],
                                          n_file=len(f_ids), n_built_f04_not_in_file=len(miss),
                                          n_file_not_built_f04=len(extra)))
    C.write_atomic(C.RES / "Z_status.csv", C.csv_text(rows, Z_FIELDS))
    rr = [r for r in rows if r["status"] != "d0_overlap_excluded"]
    rep = dict(n_targets=len(T), n_folders=sum(v["n_folders"] for v in by_src.values()),
               by_status=dict(Counter(r["status"] for r in rr)), by_source=dict(Counter(r["source"] for r in rr if r["source"])),
               n_fallback=sum(v["n_fallback"] for v in by_src.values()),
               fallback_reasons=dict(Counter(r["reason"] for r in rr if r["fallback"])),
               e_rxn_check={s: by_src[s]["e_rxn_check"] for s in SRCS}, by_src=by_src, n_d0_overlap_excluded=n_ov,
               n_stale_folders=len(stale),
               layout=dict(root=str(C.GEO_FINAL), folder="<rxn_id> (D1 >= 1000001, D0 0-5973)",
                           files=["ts.xyz", "rel1.xyz", "rel2.xyz", "result.json", ".done"],
                           z_status_fields=Z_FIELDS),
               inputs=dict(d1_labels=str(C.D1_LABELS_G2), d0_labels=str(C.D0_LABELS), d0_folds=str(C.D0_FOLDS),
                           d0_folds_sha256=C.sha256(C.D0_FOLDS), g2c_select=str(C.G2C_SELECT),
                           g2c_select_sha256=C.sha256(C.G2C_SELECT), g2_final_d1=str(C.G2_FINAL),
                           g2_final_d0=str(C.G2_FINAL_D0), preds=str(C.GEO1 / "preds"), g2c_vs_geo1b=g2c_geo1b),
               decisions="results/DECISIONS.md (A-G2c, D0D1), results/DECISIONS_ZE.md",
               stop=bool(stops), stop_reasons=stops, running=False, complete=True, wall_s=time.time() - t0)
    C.write_json(Z_REPORT, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("inputs", "layout")}, indent=1, default=str), flush=True)
    return 3 if stops else 0


def main():
    """DECISIONS_ZE 연결 S-2: upstream missing = exit 2 with nothing written; then the running report, build(), and on a
    STOP / crash inside build() a stopped report (stop true, running false, the reason)."""
    miss = upstream_missing()
    if miss:
        print(f"NOT READY: {len(miss)} upstream file(s) missing: {miss[:6]} (GEO1b R / E1 / Y, the geo1b -> geo2b merge "
              f"M1, the G2c build; RUNBOOK_geo2b.md)", file=sys.stderr, flush=True)
        return 2
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.write_json(Z_REPORT, dict(stop=True, running=True, complete=False, started=started,
                                stop_reasons=["build_geo.py build running (or killed before its final report: resubmit)"]))
    try:
        return build()
    except BaseException as e:                                           # noqa: BLE001 — record, then re-raise
        code = e.code if isinstance(e, SystemExit) else None
        if not (isinstance(e, SystemExit) and code in (0, None)):
            why = [f"STOP: {m}" for m in C.LAST_STOP] or [f"{type(e).__name__}: {e}"]
            C.write_json(Z_REPORT, dict(stop=True, running=False, complete=False, started=started,
                                        finished=time.strftime("%Y-%m-%dT%H:%M:%S"), stop_reasons=why))
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        sys.exit(main())
    raise SystemExit(__doc__)
