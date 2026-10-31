#!/usr/bin/env python3
"""Synthetic tests of the GEO2b Phase Z / E code of this role (no GPU, no real GEO1b / G2c / trainer output needed).

  python tests_ze.py z   build_geo.py on a synthetic D1 + D0 fixture (60 + 62 reactions) in a throwaway directory,
                         run as a subprocess with the test-only root overrides of geo2_common: layout, idempotent rerun,
                         and every STOP path -> results/Z_tests_synth.json         (GPU env or D1 run env: numpy only)
  python tests_ze.py e   the D0 part of eval_d0d1.py (unit records, prediction checks, D0 frames / metrics / bootstrap)
                         on synthetic frames -> results/E_tests_synth.json              (any pandas env, e.g. reactot)
Exit 0 = every check passed, 3 = a check failed. The throwaway directory (tempfile) is removed at the end.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo2_common as C  # noqa: E402     (imported WITHOUT overrides: writes its reports to the real results/)

SYMS = ["C", "N", "O", "C", "C", "H"]
BASE = np.array([[0.0, 0.0, 0.0], [1.1, 0.8, 0.0], [2.2, 0.0, 0.0], [0.4, 0.0, 2.2], [1.74, 0.0, 2.2],
                 [2.3, 0.9, 2.6]])          # frag 1 = C-N-O (dipole), frag 2 = C=C-H; forming 0-3, 2-4 (~2.24 A)
PAIRS = "0-3 2-4"


def rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q if np.linalg.det(Q) > 0 else -Q


def xyz(syms, X, comment="t"):
    return f"{len(syms)}\n{comment}\n" + "".join(f"{s} {float(x)!r} {float(y)!r} {float(z)!r}\n"
                                                 for s, (x, y, z) in zip(syms, X))   # float(): numpy 2 repr


def e_sorted(X, lab, pairs):
    pp = [tuple(int(v) for v in p.split("-")) for p in pairs.split()]
    dl = np.array([np.linalg.norm(lab[i] - lab[j]) for i, j in pp])
    dx = np.array([np.linalg.norm(X[i] - X[j]) for i, j in pp])
    o = np.argsort(dl)
    return (dx - dl)[o]


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fixture(root, mods=()):
    """the synthetic GEO1b / G2c / label world under root -> (env, expectations). mods: names of deliberate faults."""
    rng = np.random.default_rng(20261031)
    for d in ("geo1/preds", "geo1/d0", "geo1res", "g2c/d1/g2_final", "g2c/d0/g2_final", "g2cres", "d1_jobs", "prof",
              "res", "geo2"):
        (root / d).mkdir(parents=True, exist_ok=True)
    preds = {(k, s): {} for k in ("f0", "f1", "f2", "f3", "f4", "fm1") for s in (0, 1, 2)}
    sel, lab1, lab0, d0rows, erxn, erxn0 = [], [], [], [], [], []
    exp = dict(D1={}, D0={})

    def reaction(src, rid, tag, g2c, fallback=False, predict=True):
        G2 = BASE @ rot(rng).T + rng.normal(0, 3, 3) + rng.normal(0, 0.02, BASE.shape)
        lab = (G2 + rng.normal(0, 0.03, BASE.shape)) @ rot(rng).T + rng.normal(0, 3, 3)       # other frame
        Rl = np.linalg.lstsq(np.c_[lab, np.ones(6)], np.c_[G2, np.ones(6)], rcond=None)[0]   # map label -> G2 frame
        lab_g2 = np.c_[lab, np.ones(6)] @ Rl
        seeds = [lab_g2[:, :3] + rng.normal(0, 0.01, BASE.shape) for _ in range(3)]
        if fallback:
            for s in seeds:
                s[1] = s[0] + np.array([0.3, 0.0, 0.0])                                      # close contact
        folder = root / "g2c" / src.lower() / "g2_final" / str(rid)
        folder.mkdir(parents=True)
        sel.append(dict(src=src.lower(), rxn_id=rid, chosen=g2c, n_done=1 if g2c else 0))
        if g2c:
            (folder / "ts.xyz").write_text(xyz(SYMS, G2))
            (folder / "rel1.xyz").write_text(xyz(SYMS[:3], G2[:3] + 0.1, f"rel1 {rid}"))
            (folder / "rel2.xyz").write_text(xyz(SYMS[3:], G2[3:] - 0.1, f"rel2 {rid}"))
            (folder / "result.json").write_text("{}")
            (folder / "g2c.json").write_text(json.dumps(dict(rxn_id=rid, src=src.lower(), chosen=g2c)))
            (folder / ".done").write_text("x\n")
        else:
            (folder / ".fail_autode").write_text("x\n")
        if g2c and predict:
            uid = f"{src}_{rid}"
            for s in range(3):
                preds[(tag, s)][uid] = seeds[s]
            mean = np.mean(seeds, axis=0)
            e = e_sorted(mean, lab, PAIRS)
            if tag != "fm1":
                (erxn if src == "D1" else erxn0).append(dict(rxn_id=rid, core="x", e_short_model=repr(float(e[0])),
                                                             e_long_model=repr(float(e[1])), abs_max=0, bin="low"))
            exp[src][rid] = dict(mean=mean, g2=G2, fallback=fallback, tag=tag, g2c=g2c)
        return lab

    for i in range(60):                                           # D1: T1 folds -1..4, 2 without G2c, 1 unpredicted
        rid, job = 1000001 + i, f"T{i + 1:05d}"
        fold = i % 6 - 1
        g2c = "" if i in (58, 59) else ("orig", "r1", "r2")[i % 3]
        lab = reaction("D1", rid, "fm1" if fold < 0 else f"f{fold}", g2c, fallback=i == 7, predict=i != 57)
        (root / "d1_jobs" / job).mkdir()
        (root / "d1_jobs" / job / "ts.xyz").write_text(xyz(SYMS, lab))
        lab1.append(dict(job_id=job, rxn_id=rid, formed_pairs_ts=PAIRS))
    n_fb0 = 2 if "d0_fallback_excess" in mods else 1
    for rid in range(62):                                         # D0: folds 0-4, 60/61 = D-1 overlaps, 59 without G2c
        ov = rid >= 60
        fold = -9 if ov else rid % 5
        g2c = "" if rid == 59 else ("orig", "r2")[rid % 2]
        lab = reaction("D0", rid, f"f{fold}", g2c, fallback=rid in (5, 6)[:n_fb0], predict=not ov)
        ts = f"TS_{rid}_x.xyz"
        (root / "prof" / str(rid)).mkdir()
        (root / "prof" / str(rid) / ts).write_text(xyz(SYMS, lab))
        lab0.append(dict(rxn_id=rid, ts_file=ts, formed_pairs_ts=PAIRS, status="ok"))
        d0rows.append(dict(rxn_id=rid, component=f"c{rid // 4:04d}" if not ov else "", fold=fold, twin_of="",
                           overlap=str(ov)))
    lab0.append(dict(rxn_id=3090, ts_file="none", formed_pairs_ts=PAIRS, status="no_forming_bond_ts"))   # D0_EXCLUDE
    if "d0_wrong_fold" in mods:                                   # D0_1 (fold 1) predicted by the fold-2 model
        for s in range(3):
            preds[("f2", s)]["D0_1"] = preds[("f1", s)].pop("D0_1")
    if "two_folds" in mods:                                       # D1_1000003 is T1 fold 1: also in the fold-2 files
        for s in range(3):
            preds[("f2", s)]["D1_1000003"] = preds[("f1", s)]["D1_1000003"]
    if "stray" in mods:
        for s in range(3):
            preds[("f0", s)]["D1_999"] = BASE.copy()
    if "e_rxn_d0" in mods:
        erxn0[3]["e_short_model"] = repr(float(erxn0[3]["e_short_model"]) + 1e-4)
    if "c_select" in mods:
        sel[10]["chosen"] = "r2" if sel[10]["chosen"] != "r2" else "r1"
    if "e_rxn_missing" in mods:                                   # D1_1000002 (fold 0, built) missing from E_rxn.csv
        erxn.pop(0)
    if "e_rxn_extra" in mods:                                     # D0 59 (no G2c, not built) listed in E_rxn_d0.csv
        erxn0.append(dict(erxn0[0], rxn_id=59))
    if "stale_folder" in mods:                                    # an old .done folder of the no-G2c target 1000059
        (root / "geo2" / "geo_final" / "1000059").mkdir(parents=True)
        (root / "geo2" / "geo_final" / "1000059" / ".done").write_text("old\n")
    for (k, s), P in preds.items():
        with open(root / "geo1" / "preds" / f"final_{k}_s{s}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
    write_csv(root / "g2cres" / "C_select.csv", sel, ["src", "rxn_id", "chosen", "n_done"])
    (root / "labels_d1.json").write_text(json.dumps(lab1))
    (root / "labels_all.json").write_text(json.dumps(lab0))
    write_csv(root / "geo1" / "d0" / "d0_folds.csv", d0rows, ["rxn_id", "component", "fold", "twin_of", "overlap"])
    fields = ["rxn_id", "core", "e_short_model", "e_long_model", "abs_max", "bin"]
    write_csv(root / "geo1res" / "E_rxn.csv", erxn, fields)
    write_csv(root / "geo1res" / "E_rxn_d0.csv", erxn0, fields)
    fp = root / "geo1" / "d0" / "d0_folds.csv"
    (root / "geo1res" / "D0_pairs_report.json").write_text(json.dumps(dict(
        stop=False, outputs={str(fp): dict(sha256=C.sha256(fp))})))
    if "d0_folds_sha" in mods:                                    # changed after the report recorded its sha256
        fp.write_text(fp.read_text() + "\n")
    env = dict(os.environ, GEO2_SCRATCH=str(root / "geo2"), GEO_SCRATCH=str(root / "geo1"),
               GEO2_GEO1_RES=str(root / "geo1res"), G2C_SCRATCH=str(root / "g2c"), G2C_RESULTS=str(root / "g2cres"),
               GEO2_D1_JOBS=str(root / "d1_jobs"), GEO2_D1_LABELS=str(root / "labels_d1.json"),
               GEO2_D0_LABELS=str(root / "labels_all.json"), GEO2_D0_PROF=str(root / "prof"),
               GEO2_RESULTS=str(root / "res"), PYTHONDONTWRITEBYTECODE="1")
    for k in ("GEO2_G2_FINAL", "GEO2_G2_FINAL_D0"):
        env.pop(k, None)
    return env, exp


def build(env):
    p = subprocess.run([sys.executable, str(HERE / "build_geo.py"), "build"], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def report(root):
    p = root / "res" / "Z_report.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def cmd_z():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="geo2b_ztest_"))
    ck, det = {}, {}
    try:
        root = tmp / "base"
        env, exp = fixture(root)
        rc, out, err = build(env)
        rep = report(root)
        det["base_rc"], det["base_stderr"] = rc, err[-1500:]
        bs = rep.get("by_src", {})
        ck["base_exit0_stop_false"] = rc == 0 and rep.get("stop") is False
        ck["base_counts"] = (bs.get("D1", {}).get("n_folders") == 57 and bs.get("D0", {}).get("n_folders") == 59
                             and bs["D1"]["by_status"] == {"ok": 57, "no_g2": 2, "no_geo1_prediction": 1}
                             and bs["D0"]["by_status"] == {"ok": 59, "no_g2": 1} and rep["n_d0_overlap_excluded"] == 2
                             and rep["n_targets"] == 120 and bs["D1"]["n_fallback"] == 1 and bs["D0"]["n_fallback"] == 1)
        n_f04 = sum(1 for v in exp["D1"].values() if v["tag"] != "fm1")
        ck["base_e_rxn_check"] = (bs["D1"]["e_rxn_check"]["n"] == n_f04 and bs["D0"]["e_rxn_check"]["n"] == 59
                                  and bs["D1"]["e_rxn_check"]["max_abs_dev_A"] <= 1e-9
                                  and bs["D0"]["e_rxn_check"]["max_abs_dev_A"] <= 1e-9
                                  and bs["D1"]["e_rxn_check"]["n_built_not_in_file"] == len(exp["D1"]) - n_f04
                                  and bs["D1"]["e_rxn_check"]["n_file"] == n_f04 and bs["D0"]["e_rxn_check"]["n_file"] == 59
                                  and all(bs[s]["e_rxn_check"][k] == 0 for s in ("D1", "D0")
                                          for k in ("n_built_f04_not_in_file", "n_file_not_built_f04"))
                                  and rep.get("n_stale_folders") == 0)
        bad = []
        for s in ("D1", "D0"):
            for rid, v in exp[s].items():
                d = root / "geo2" / "geo_final" / str(rid)
                g2 = root / "g2c" / s.lower() / "g2_final" / str(rid)
                if not all((d / f).is_file() for f in ("ts.xyz", "rel1.xyz", "rel2.xyz", "result.json", ".done")):
                    bad.append(f"{s} {rid}: files")
                    continue
                if any((d / f).read_bytes() != (g2 / f).read_bytes() for f in ("rel1.xyz", "rel2.xyz")):
                    bad.append(f"{s} {rid}: rel not a byte copy")
                _, X = C.read_xyz(d / "ts.xyz")
                want = v["g2"] if v["fallback"] else v["mean"]
                if np.abs(X - want).max() > 1e-9:
                    bad.append(f"{s} {rid}: ts.xyz != {'G2c TS' if v['fallback'] else '3-seed mean'}")
                r = json.loads((d / "result.json").read_text())
                if (r["src"], r["uid"], r["fallback"], r["fold_tag"], r["g2c_chosen"]) != \
                        (s, f"{s}_{rid}", v["fallback"], v["tag"], v["g2c"]) or \
                        (v["fallback"] and r["fallback_reason"] != "close_contact"):
                    bad.append(f"{s} {rid}: result.json {r['src'], r['uid'], r['fallback'], r['fold_tag']}")
                if s == "D0" and r["job_id"] != f"D0G2_{rid:04d}":
                    bad.append(f"{s} {rid}: job_id {r['job_id']}")
        ck["base_folders"], det["base_folder_problems"] = not bad, bad[:10]
        with open(root / "res" / "Z_status.csv", newline="") as f:
            zs = list(csv.DictReader(f))
        ck["base_z_status"] = (len(zs) == 122 and sum(r["status"] == "d0_overlap_excluded" for r in zs) == 2
                               and set(zs[0]) == {"rxn_id", "job_id", "src", "status", "source", "fallback", "reason",
                                                  "g2c"}
                               and not (root / "geo2" / "geo_final" / "60").exists())
        stamp = json.loads((root / "geo2" / "geo_final" / "1000002" / "result.json").read_text())["written"]
        rc2, _, err2 = build(env)
        rep2 = report(root)
        ck["rerun_all_kept"] = (rc2 == 0 and rep2["by_status"].get("kept") == 116 and "ok" not in rep2["by_status"]
                                and json.loads((root / "geo2" / "geo_final" / "1000002" / "result.json").read_text())
                                ["written"] == stamp)
        r1 = root / "g2c" / "d0" / "g2_final" / "3" / "rel1.xyz"
        keep = r1.read_bytes()
        r1.write_bytes(keep + b"\n")
        rc3, _, err3 = build(env)
        r1.write_bytes(keep)
        ck["changed_source_stops"] = rc3 == 3 and "exists with other sources" in err3 and report(root).get("stop") is True
        cases = {"e_rxn_d0": ("D0: forming-distance errors", "D1: forming"),
                 "d0_fallback_excess": ("D0: fallback rate", "D1: fallback"),
                 "d0_wrong_fold": ("predicted by f2 but its d0_folds.csv fold is 1", None),
                 "two_folds": ("predicted by two folds", None),
                 "stray": ("are not GEO2b targets", None),
                 "c_select": ("no valid g2c.json", None),
                 "d0_folds_sha": ("sha256 differs", None),
                 "e_rxn_missing": ("D1: E_rxn.csv rows != the reactions built", "D0: E_rxn_d0.csv rows"),
                 "e_rxn_extra": ("D0: E_rxn_d0.csv rows != the reactions built", "D1: E_rxn.csv rows"),
                 "stale_folder": ("geo_final folders with .done for targets that are not built now", "rows !=")}
        for name, (want, notwant) in cases.items():
            rootc = tmp / name
            envc, _ = fixture(rootc, mods=(name,))
            rcc, _, errc = build(envc)
            repc = report(rootc)
            text = errc + json.dumps(repc.get("stop_reasons", []))
            okc = rcc == 3 and want in text and (notwant is None or notwant not in text) and repc.get("stop") is True
            if name in ("c_select", "d0_folds_sha", "stray", "two_folds", "d0_wrong_fold"):
                okc &= not (rootc / "geo2" / "geo_final").exists()               # stopped before any folder write
            if name == "d0_fallback_excess":
                okc &= json.loads((rootc / "geo2" / "geo_final" / "6" / "result.json").read_text())["fallback"] is True
            ck[f"stop_{name}"], det[f"stop_{name}"] = okc, dict(rc=rcc, tail=errc[-400:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(ck.values())
    C.write_json(C.RES / "Z_tests_synth.json", dict(test="tests_ze.py z (build_geo.py, synthetic)", all_pass=ok,
                                                     checks=ck, detail=det, python=sys.version.split()[0],
                                                     numpy=np.__version__, seconds=time.time() - t0,
                                                     date=time.strftime("%F %T")))
    for k, v in ck.items():
        print(f"[z] {k}: {'PASS' if v else 'FAIL'}", flush=True)
    if not ok:
        print(json.dumps(det, indent=1, default=str)[:4000], flush=True)
    return 0 if ok else 3


# ---------------------------------------------------------------- e: eval_d0d1 D0 part on synthetic frames
def cmd_e():
    import pandas as pd
    import eval_d0d1 as E
    import eval_geo2 as EG
    t0 = time.time()
    rng = np.random.default_rng(7)
    ck, det = {}, {}
    T, A, M = E.TARGETS, E.ARMS, E.MODELS
    n = 48
    D = pd.DataFrame(dict(pair_id=[f"D{i:05d}" for i in range(n)], ref_kind="d0", a_rxn=np.arange(n) * 2,
                          b_rxn=np.arange(n) * 2 + 1, core=[f"D0/c{i // 4:04d}" for i in range(n)],
                          edge_kind=[E.EDGE_KINDS[i % 3] for i in range(n)], ok=[i not in (5, 17) for i in range(n)],
                          **{f"dd_{t}": rng.normal(0, 3, n) for t in T})).set_index("pair_id")
    folds = {int(r): (i // 4) % 5 for i in range(n) for r in (2 * i, 2 * i + 1)}
    m1 = {2 * 9}                                               # pair D00009 has an m1 end
    rows = []
    for a in A:
        for m in M:
            for t in T:
                y = D[f"dd_{t}"].to_numpy()
                noise = 0.0 if (a, m) == ("SH", "AW") else 0.2 * (1 + M.index(m))
                rows.append(pd.DataFrame(dict(pair_id=D.index, ref_kind="d0", pos="d0", fold=[folds[int(x)] for x in D.a_rxn],
                                              rung="geo", arm=a, model=m, target=t, y=y,
                                              yhat=y + rng.normal(0, noise, n) if noise else y.copy(), base=0.0, beta=0.0,
                                              yhat_seed_sd=0.0)))
    P0 = pd.concat(rows, ignore_index=True)
    rep = {("geo", a, t): "GNN" for a in A for t in T}
    rep8 = {("geo", t): ("SH", "TF") for t in T}
    fr, chk = E.d0_frames(P0, D, folds, rep, rep8)
    ck["frames_counts"] = chk == dict(n_pairs_predicted=48, n_not_ok_left_out=2, n_pairs_evaluated=46)
    R = E.d0_metrics(fr, D, m1)
    ck["n_rows"] = len(R) == len(E.D0_SELS) * len(T) * 4 * 2
    x = R[R.sel == "AW·SH"]
    ck["exact_model_zero"] = bool((x.mae.fillna(0) == 0).all() and (x.mae_ci_hi.fillna(0) == 0).all())
    ck["rep_equals_gnn"] = bool(np.array_equal(R[R.sel == "REP·S"].mae.to_numpy(), R[R.sel == "GNN·S"].mae.to_numpy(),
                                               equal_nan=True))
    ck["rep8_equals_tf_sh"] = bool(np.array_equal(R[R.sel == "REP8"].mae.to_numpy(), R[R.sel == "TF·SH"].mae.to_numpy(),
                                                  equal_nan=True))
    nl = R[(R.sel == "null") & (R.n > 0)]
    ck["null_mean_abs_y"] = bool(np.allclose(nl.mae, nl.mae_null, rtol=0, atol=1e-12))
    al = R[(R.edge_kind == "all") & (R.subset == "all")]
    ck["n_all_is_46"] = bool((al.n == 46).all())
    sub = R[(R.edge_kind == "all") & (R.subset == "no_m1_twin")]
    ck["m1_subset_drops_one"] = bool((sub.n == 45).all())
    w = R[(R.subset == "all") & (R.edge_kind != "all")].groupby(["sel", "target"]).n.sum()
    ck["edge_kinds_add_up"] = bool((w.reindex(al.set_index(["sel", "target"]).index) == al.set_index(["sel", "target"]).n)
                                   .all())
    f = fr["ENS4·S"]
    mean4 = pd.concat([fr[f"{m}·S"] for m in M]).groupby(["pair_id", "target"]).yhat.mean()
    ck["ens4_mean"] = bool(np.allclose(f.set_index(["pair_id", "target"]).yhat.reindex(mean4.index), mean4, atol=1e-12,
                                       rtol=0))
    diffs = []
    for sel, t, ek in (("GNN·S", "oi", "all"), ("XGB·SH", "barrier", "dipole"), ("REP8", "cpcm", "leaf")):
        q = fr[sel][fr[sel].target == t]
        Dq = D.loc[q.pair_id]
        mk = np.ones(len(q), bool) if ek == "all" else (Dq.edge_kind == ek).to_numpy()
        e = (q.yhat - q.y).to_numpy()[mk]
        a_ = E.boot_mae_ci(e, Dq.core.to_numpy()[mk])
        b_ = EG.boot_diff(e, np.zeros(len(e)), Dq.core.to_numpy()[mk])[:3]
        r = R[(R.sel == sel) & (R.target == t) & (R.edge_kind == ek) & (R.subset == "all")].iloc[0]
        diffs.append(max(abs(u - v) for u, v in zip(a_, b_)))
        diffs.append(max(abs(r.mae - b_[0]), abs(r.mae_ci_lo - b_[1]), abs(r.mae_ci_hi - b_[2])))
    ck["boot_equals_eval_geo2"], det["boot_max_diff"] = max(diffs) <= 1e-12, max(diffs)
    t1 = P0.copy()
    t1.loc[t1.pair_id == "D00000", "fold"] = 3
    t2 = P0.copy()
    t2.loc[t2.index[5], "y"] += 1e-6
    t3 = P0.drop(index=P0.index[100])
    t4 = P0.copy()
    t4.loc[t4.index[7], "yhat"] = np.nan
    for name, bad in (("wrong_fold", t1), ("y_mismatch", t2), ("missing_row", t3), ("nonfinite_yhat", t4)):
        ck[f"stop_{name}"] = E._expect_stop(lambda b=bad: E.d0_frames(b, D, folds, rep, rep8))
    # check_predictions: D1 rows + D0 rows, ref_kind / id agreement, completeness
    G1 = P0.assign(pair_id=P0.pair_id.str.replace("D", "P", regex=False), ref_kind="reference", pos="p")
    G = pd.concat([G1, P0], ignore_index=True)
    p1, p0 = E.check_predictions(G.assign(extra_col=1), set(G1.pair_id), set(D.index))
    ck["check_predictions_split"] = len(p1) == len(G1) and len(p0) == len(P0)
    g = G.copy()
    g.loc[g.index[-1], "ref_kind"] = "reference"
    ck["stop_ref_kind_mismatch"] = E._expect_stop(lambda: E.check_predictions(g, set(G1.pair_id), set(D.index)))
    ck["stop_unknown_d1_id"] = E._expect_stop(lambda: E.check_predictions(G, set(G1.pair_id) - {"P00000"},
                                                                            set(D.index)))
    ck["stop_other_rung"] = E._expect_stop(lambda: E.check_predictions(G.assign(rung="g2"), set(G1.pair_id),
                                                                         set(D.index)))
    ck["stop_missing_model"] = E._expect_stop(lambda: E.check_predictions(G[G.model != "XGB"], set(G1.pair_id),
                                                                            set(D.index)))
    ck["stop_missing_column"] = E._expect_stop(lambda: E.check_predictions(G.drop(columns=["base"]), set(G1.pair_id),
                                                                             set(D.index)))
    f1 = dict(zip(G1.pair_id, G1.fold))                        # D1 fold check (S_folds fold_T1 stand-in)
    p1f, _ = E.check_predictions(G, set(G1.pair_id), set(D.index), f1)
    ck["check_predictions_d1_fold_ok"] = len(p1f) == len(G1)
    ck["stop_d1_wrong_fold"] = E._expect_stop(lambda: E.check_predictions(
        G, set(G1.pair_id), set(D.index), dict(f1, P00000=(f1["P00000"] + 1) % 5)))
    # unit records: robust names, two reports, folds 0-4
    tmp = Path(tempfile.mkdtemp(prefix="geo2b_etest_"))
    try:
        u1 = {f"L__geo__T1D0__{a}__{m}__f{k}": dict(val_kcal_mae={t: 0.1 + 0.01 * M.index(m) + 0.001 * k for t in T})
              for a in A for m in M if m != "XGB" for k in range(5)}
        u1["L__g2__S__AW__f0"] = dict(val_kcal_mae={t: 9.0 for t in T})      # another rung: ignored
        u2 = {f"L__geo__{a}__XGB__f{k}": dict(val_kcal_mae={t: 0.05 for t in T}) for a in A for k in range(5)}
        (tmp / "a.json").write_text(json.dumps(dict(units=u1)))
        (tmp / "b.json").write_text(json.dumps(dict(units=dict(u2, **{"L__geo__S__AW__f0": u1["L__geo__T1D0__S__AW__f0"]}))))
        U = E.load_units_geo((tmp / "a.json", tmp / "b.json"))
        rp, rp8, _, _ = E.rep_tables(U)
        ck["units_parse_and_rep"] = (len(U) == 8 * 5 * 10 and rp[("geo", "S", "oi")] == "XGB"
                                     and rp8[("geo", "oi")] == ("SH", "XGB"))
        bad = dict(u2)
        bad.pop("L__geo__SH__XGB__f3")
        (tmp / "c.json").write_text(json.dumps(dict(units=bad)))
        ck["stop_units_missing_fold"] = E._expect_stop(lambda: E.load_units_geo((tmp / "a.json", tmp / "c.json")))
        dup = dict(u2, **{"L__geo__S__AW__f0": dict(val_kcal_mae={t: 1.0 for t in T})})
        (tmp / "d.json").write_text(json.dumps(dict(units=dup)))
        ck["stop_units_conflict"] = E._expect_stop(lambda: E.load_units_geo((tmp / "a.json", tmp / "d.json")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(ck.values())
    C.write_json(C.RES / "E_tests_synth.json", dict(test="tests_ze.py e (eval_d0d1.py D0 part, synthetic)", all_pass=ok,
                                                     checks=ck, detail=det, python=sys.version.split()[0],
                                                     pandas=pd.__version__, seconds=time.time() - t0,
                                                     date=time.strftime("%F %T")))
    for k, v in ck.items():
        print(f"[e] {k}: {'PASS' if v else 'FAIL'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["z"]:
        sys.exit(cmd_z())
    if a == ["e"]:
        sys.exit(cmd_e())
    raise SystemExit(__doc__)
