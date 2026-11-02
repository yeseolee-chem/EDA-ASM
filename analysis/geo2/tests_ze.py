#!/usr/bin/env python3
"""Synthetic tests of the GEO2b Phase Z / E code of this role (no GPU, no real GEO1b / G2c / trainer output needed).

  python tests_ze.py z   build_geo.py on a synthetic D1 + D0 fixture (60 + 62 reactions) in a throwaway directory,
                         run as a subprocess with the test-only root overrides of geo2_common: layout, idempotent rerun,
                         the E2-rule coordinate candidates (mean3 / mean_pass / seed<k> / g2c_fallback, the E_rxn scope,
                         the written-coordinate check, a rule change on rerun), every STOP path (stopped report: running
                         false), upstream files missing (exit 2, nothing written), a crash inside the run
                         -> results/Z_tests_synth.json   (GPU env or D1 run env: numpy only)
  python tests_ze.py e   the D0 part of eval_d0d1.py (unit records, prediction checks, D0 frames / metrics / bootstrap,
                         load_d0 final rules E-9) on synthetic frames -> results/E_tests_synth.json   (any pandas env)
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


def _expect_exit(fn, code):
    try:
        fn()
    except SystemExit as e:
        return e.code == code
    return False


def fixture(root, mods=()):
    """the synthetic GEO1b / G2c / label world under root -> (env, expectations). mods: names of deliberate faults."""
    rng = np.random.default_rng(20261031)
    for d in ("geo1/preds", "geo1/d0", "geo1res", "g2c/d1/g2_final", "g2c/d0/g2_final", "g2cres", "d1_jobs", "prof",
              "res", "geo2"):
        (root / d).mkdir(parents=True, exist_ok=True)
    preds = {(k, s): {} for k in ("f0", "f1", "f2", "f3", "f4", "fm1") for s in (0, 1, 2)}
    sel, lab1, lab0, d0rows, erxn, erxn0 = [], [], [], [], [], []
    exp = dict(D1={}, D0={})

    def reaction(src, rid, tag, g2c, fallback=False, predict=True, rescue=None):
        G2 = BASE @ rot(rng).T + rng.normal(0, 3, 3) + rng.normal(0, 0.02, BASE.shape)
        lab = (G2 + rng.normal(0, 0.03, BASE.shape)) @ rot(rng).T + rng.normal(0, 3, 3)       # other frame
        Rl = np.linalg.lstsq(np.c_[lab, np.ones(6)], np.c_[G2, np.ones(6)], rcond=None)[0]   # map label -> G2 frame
        lab_g2 = np.c_[lab, np.ones(6)] @ Rl
        seeds = [lab_g2[:, :3] + rng.normal(0, 0.01, BASE.shape) for _ in range(3)]
        rule, used = ("g2c_fallback", []) if fallback else ("mean3", [0, 1, 2])            # E2-rule expected
        if fallback:                                              # rule 4: no seed passes
            for s in seeds:
                s[1] = s[0] + np.array([0.3, 0.0, 0.0])                                      # close contact
        if rescue == "mean_pass":                                 # rule 2: seed 0 breaks C0-N1 so far that the 3-seed
            nb = (seeds[1][1] + seeds[2][1]) / 2                  # mean puts N1 on C0; seeds 1, 2 and their mean pass
            seeds[0][1] = seeds[0][0] - 2 * (nb - seeds[0][0])
            rule, used = "mean_pass", [1, 2]
        elif rescue == "one_pass":                                # rule 3: only seed 1 passes (0, 2 close contacts)
            for k in (0, 2):
                seeds[k][1] = seeds[k][0] + np.array([0.3, 0.0, 0.0])
            rule, used = "seed1", [1]
        elif rescue == "pass_mean_fails":                         # rule 3 after rule 2 fails: seeds 0, 2 pass with H5
            for k, sg in ((0, 1.0), (2, -1.0)):                   # on opposite sides of C4 (their mean puts H5 on C4),
                s = seeds[k]                                      # seed 1 close contact
                ay = s[1] - (s[0] + s[2]) / 2
                s[5] = s[4] + sg * 1.13 * ay / np.linalg.norm(ay)
            seeds[1][1] = seeds[1][0] + np.array([0.3, 0.0, 0.0])
            rule, used = "seed0", [0]
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
            want = G2 if fallback else np.mean([seeds[k] for k in used], axis=0)
            exp[src][rid] = dict(mean=mean, g2=G2, fallback=fallback, tag=tag, g2c=g2c, rule=rule, used=used, want=want)
        return lab

    resc1 = {13: "mean_pass", 24: "mean_pass", 20: "one_pass", 27: "pass_mean_fails"}   # D1 i (24 = fold -1)
    for i in range(60):                                           # D1: T1 folds -1..4, 2 without G2c, 1 unpredicted
        rid, job = 1000001 + i, f"T{i + 1:05d}"
        fold = i % 6 - 1
        g2c = "" if i in (58, 59) else ("orig", "r1", "r2")[i % 3]
        lab = reaction("D1", rid, "fm1" if fold < 0 else f"f{fold}", g2c, fallback=i == 7, predict=i != 57,
                       rescue=resc1.get(i))
        (root / "d1_jobs" / job).mkdir()
        (root / "d1_jobs" / job / "ts.xyz").write_text(xyz(SYMS, lab))
        lab1.append(dict(job_id=job, rxn_id=rid, formed_pairs_ts=PAIRS))
    n_fb0 = 2 if "d0_fallback_excess" in mods else 1
    for rid in range(62):                                         # D0: folds 0-4, 60/61 = D-1 overlaps, 59 without G2c
        ov = rid >= 60
        fold = -9 if ov else rid % 5
        g2c = "" if rid == 59 else ("orig", "r2")[rid % 2]
        lab = reaction("D0", rid, f"f{fold}", g2c, fallback=rid in (5, 6)[:n_fb0], predict=not ov,
                       rescue="mean_pass" if rid == 8 else None)
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
    if "e_rxn_rescued" in mods:                                   # E2-rule scope: the row of the mean_pass reaction
        r = next(r for r in erxn if r["rxn_id"] == 1000014)       # 1000014 is not compared (no STOP)
        r["e_short_model"] = repr(float(r["e_short_model"]) + 1e-4)
    if "e_rxn_missing_rescued" in mods:                           # ... but stays in the row-set check (seed1 1000021)
        erxn[:] = [r for r in erxn if r["rxn_id"] != 1000021]
    if "stale_folder" in mods:                                    # an old .done folder of the no-G2c target 1000059
        (root / "geo2" / "geo_final" / "1000059").mkdir(parents=True)
        (root / "geo2" / "geo_final" / "1000059" / ".done").write_text("old\n")
    for (k, s), P in preds.items():
        with open(root / "geo1" / "preds" / f"final_{k}_s{s}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
    if "no_pkl" in mods:                                          # GEO1b R / Y not finished (exit 2)
        (root / "geo1" / "preds" / "final_f3_s1.pkl").unlink()
    if "bad_pkl" in mods:                                         # a crash inside build() (stopped report, not running)
        (root / "geo1" / "preds" / "final_f2_s0.pkl").write_bytes(b"not a pickle")
    sp = root / "g2cres" / "C_select.csv"
    write_csv(sp, sel, ["src", "rxn_id", "chosen", "n_done"])
    dsha = C.sha256(sp)                                           # GEO1b data.py D_dataset_report g2_input (S-6)
    if "no_c_select" in mods:
        sp.unlink()
    if "d_report_c_select" in mods:
        dsha = "0" * 64
    if "no_d_report" not in mods:
        (root / "geo1res" / "D_dataset_report.json").write_text(json.dumps(dict(
            g2_input=dict(kind="G2c (decision A-G2c)", c_select=str(sp), c_select_sha256=dsha))))
    (root / "labels_d1.json").write_text(json.dumps(lab1))
    (root / "labels_all.json").write_text(json.dumps(lab0))
    write_csv(root / "geo1" / "d0" / "d0_folds.csv", d0rows, ["rxn_id", "component", "fold", "twin_of", "overlap"])
    fields = ["rxn_id", "core", "e_short_model", "e_long_model", "abs_max", "bin"]
    write_csv(root / "geo1res" / "E_rxn.csv", erxn, fields)
    if "no_e_rxn" not in mods:
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
        ck["base_exit0_stop_false"] = (rc == 0 and rep.get("stop") is False and rep.get("running") is False
                                       and rep.get("complete") is True
                                       and rep["inputs"]["g2c_vs_geo1b"]["equal"] is True)        # S-2, S-6
        ck["base_counts"] = (bs.get("D1", {}).get("n_folders") == 57 and bs.get("D0", {}).get("n_folders") == 59
                             and bs["D1"]["by_status"] == {"ok": 57, "no_g2": 2, "no_geo1_prediction": 1}
                             and bs["D0"]["by_status"] == {"ok": 59, "no_g2": 1} and rep["n_d0_overlap_excluded"] == 2
                             and rep["n_targets"] == 120 and bs["D1"]["n_fallback"] == 1 and bs["D0"]["n_fallback"] == 1)
        n_f04 = sum(1 for v in exp["D1"].values() if v["tag"] != "fm1")
        n_f04_m3 = sum(1 for v in exp["D1"].values() if v["tag"] != "fm1" and v["rule"] == "mean3")   # E2-rule scope
        n_d0_m3 = sum(1 for v in exp["D0"].values() if v["rule"] == "mean3")
        ck["base_e_rxn_check"] = (bs["D1"]["e_rxn_check"]["n"] == n_f04_m3 and bs["D0"]["e_rxn_check"]["n"] == n_d0_m3
                                  and (n_f04 - n_f04_m3, 59 - n_d0_m3) == (4, 2)
                                  and bs["D1"]["e_rxn_check"]["n_in_file_not_mean3"] == n_f04 - n_f04_m3
                                  and bs["D0"]["e_rxn_check"]["n_in_file_not_mean3"] == 59 - n_d0_m3
                                  and bs["D1"]["e_rxn_check"]["max_abs_dev_A"] <= 1e-9
                                  and bs["D0"]["e_rxn_check"]["max_abs_dev_A"] <= 1e-9
                                  and bs["D1"]["e_rxn_check"]["n_built_not_in_file"] == len(exp["D1"]) - n_f04
                                  and bs["D1"]["e_rxn_check"]["n_file"] == n_f04 and bs["D0"]["e_rxn_check"]["n_file"] == 59
                                  and all(bs[s]["e_rxn_check"][k] == 0 for s in ("D1", "D0")
                                          for k in ("n_built_f04_not_in_file", "n_file_not_built_f04"))
                                  and rep.get("n_stale_folders") == 0)
        want_rules = {s: {k: sum(v["rule"] == k for v in exp[s].values()) for k in
                          ("mean3", "mean_pass", "seed0", "seed1", "seed2", "g2c_fallback")} for s in ("D1", "D0")}
        want_rules = {s: {k: n for k, n in w.items() if n} for s, w in want_rules.items()}
        ck["base_coord_rules"] = (want_rules == {"D1": {"mean3": 52, "mean_pass": 2, "seed0": 1, "seed1": 1,
                                                        "g2c_fallback": 1},
                                                 "D0": {"mean3": 57, "mean_pass": 1, "g2c_fallback": 1}}
                                  and all(bs[s]["by_coord_rule"] == want_rules[s] for s in ("D1", "D0"))
                                  and (bs["D1"]["n_rescued"], bs["D0"]["n_rescued"]) == (4, 1)
                                  and (bs["D1"]["n_fallback"], bs["D0"]["n_fallback"]) == (1, 1)  # rules 2-3: no fallback
                                  and abs(bs["D1"]["fallback_share"] - 1 / 57) < 1e-12
                                  and bs["D1"]["mean3_fail_reasons"] == {"close_contact": 5}
                                  and rep["by_coord_rule"] == {k: want_rules["D1"].get(k, 0) + want_rules["D0"].get(k, 0)
                                                               for k in rep["by_coord_rule"]}
                                  and all(bs[s]["written_check"]["n_by_rule"] == want_rules[s]
                                          and bs[s]["written_check"]["max_abs_dev_A"] <= 1e-9 for s in ("D1", "D0")))
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
                if np.abs(X - v["want"]).max() > 1e-9:                   # E2-rule: mean3 / seed-subset mean / seed / G2c
                    bad.append(f"{s} {rid}: ts.xyz != the {v['rule']} coordinates of seeds {v['used']}")
                r = json.loads((d / "result.json").read_text())
                if (r["src"], r["uid"], r["fallback"], r["fold_tag"], r["g2c_chosen"]) != \
                        (s, f"{s}_{rid}", v["fallback"], v["tag"], v["g2c"]) or \
                        (v["fallback"] and r["fallback_reason"] != "close_contact"):
                    bad.append(f"{s} {rid}: result.json {r['src'], r['uid'], r['fallback'], r['fold_tag']}")
                if (r["coord_rule"], r["coord_seeds"], r["sources"]["coord_rule"], r["sources"]["coord_seeds"]) != \
                        (v["rule"], v["used"], v["rule"], v["used"]) or \
                        (v["rule"] != "mean3" and (r["mean3_sanity_reason"] != "close_contact"
                                                   or r["fallback_reason"] != ("close_contact" if v["fallback"] else "")
                                                   or len(r["seed_sanity_reasons"]) != 3)):
                    bad.append(f"{s} {rid}: result.json coord_rule {r['coord_rule']} {r['coord_seeds']} != {v['rule']}")
                if s == "D0" and r["job_id"] != f"D0G2_{rid:04d}":
                    bad.append(f"{s} {rid}: job_id {r['job_id']}")
        ck["base_folders"], det["base_folder_problems"] = not bad, bad[:10]
        with open(root / "res" / "Z_status.csv", newline="") as f:
            zs = list(csv.DictReader(f))
        zr = {(r["src"], int(r["rxn_id"])): (r["coord_rule"], r["coord_seeds"]) for r in zs}
        ck["base_z_status"] = (len(zs) == 122 and sum(r["status"] == "d0_overlap_excluded" for r in zs) == 2
                               and set(zs[0]) == {"rxn_id", "job_id", "src", "status", "source", "fallback", "reason",
                                                  "g2c", "coord_rule", "coord_seeds"}
                               and all(zr[(s, rid)] == (v["rule"], " ".join(map(str, v["used"])))
                                       for s in ("D1", "D0") for rid, v in exp[s].items())
                               and all(r["coord_rule"] == "" for r in zs if r["status"] not in ("ok", "kept"))
                               and not (root / "geo2" / "geo_final" / "60").exists())
        stamp = json.loads((root / "geo2" / "geo_final" / "1000002" / "result.json").read_text())["written"]
        rc2, _, err2 = build(env)
        rep2 = report(root)
        ck["rerun_all_kept"] = (rc2 == 0 and rep2["by_status"].get("kept") == 116 and "ok" not in rep2["by_status"]
                                and json.loads((root / "geo2" / "geo_final" / "1000002" / "result.json").read_text())
                                ["written"] == stamp
                                and all(rep2["by_src"][s]["written_check"]["n_by_rule"] == want_rules[s]
                                        for s in ("D1", "D0")))                  # kept folders: written check too
        r1 = root / "g2c" / "d0" / "g2_final" / "3" / "rel1.xyz"
        keep = r1.read_bytes()
        r1.write_bytes(keep + b"\n")
        rc3, _, err3 = build(env)
        r1.write_bytes(keep)
        r3 = report(root)
        ck["changed_source_stops"] = (rc3 == 3 and "exists with other sources" in err3 and r3.get("stop") is True
                                      and r3.get("running") is False and "exists with other sources"
                                      in json.dumps(r3.get("stop_reasons")))      # S-2: a stopped report, not 'running'
        ts = root / "geo2" / "geo_final" / "1000014" / "ts.xyz"                   # E2-rule: a kept mean_pass folder
        keep = ts.read_bytes()                                                   # whose ts.xyz changed afterwards
        syms_w, Xw = C.read_xyz(ts)
        ts.write_text(C.xyz_text(syms_w, Xw + np.array([1e-6, 0.0, 0.0]), "tampered"))
        rc4, _, err4 = build(env)
        ts.write_bytes(keep)
        r4 = report(root)
        ck["stop_written_coords"] = (rc4 == 3 and r4.get("stop") is True and r4.get("running") is False
                                     and "D1: a written ts.xyz differs from the coordinates of its coord_rule"
                                     in json.dumps(r4.get("stop_reasons")) and "D0: a written" not in
                                     json.dumps(r4.get("stop_reasons")))
        rj = root / "geo2" / "geo_final" / "1000002" / "result.json"            # a folder of the pre-E2-rule code
        keep = rj.read_bytes()                                                   # (sources without coord_rule)
        old = json.loads(keep)
        for k in ("coord_rule", "coord_seeds"):
            old["sources"].pop(k)
        rj.write_text(json.dumps(old, indent=1))
        rc5, _, err5 = build(env)
        rj.write_bytes(keep)
        r5 = report(root)
        ck["stop_rule_change"] = (rc5 == 3 and "1000002 exists with other sources" in err5 and r5.get("stop") is True
                                  and r5.get("running") is False)
        rc6, _, _ = build(env)                                                   # restored: passes again
        ck["restored_rerun_kept"] = rc6 == 0 and report(root)["by_status"].get("kept") == 116
        cases = {"d_report_c_select": ("g2_input.c_select_sha256", None),        # S-6
                 "e_rxn_d0": ("D0: forming-distance errors", "D1: forming"),
                 "d0_fallback_excess": ("D0: fallback rate", "D1: fallback"),
                 "d0_wrong_fold": ("predicted by f2 but its d0_folds.csv fold is 1", None),
                 "two_folds": ("predicted by two folds", None),
                 "stray": ("are not GEO2b targets", None),
                 "c_select": ("no valid g2c.json", None),
                 "d0_folds_sha": ("sha256 differs", None),
                 "e_rxn_missing": ("D1: E_rxn.csv rows != the reactions built", "D0: E_rxn_d0.csv rows"),
                 "e_rxn_extra": ("D0: E_rxn_d0.csv rows != the reactions built", "D1: E_rxn.csv rows"),
                 "e_rxn_missing_rescued": ("D1: E_rxn.csv rows != the reactions built", "D0: E_rxn_d0.csv rows"),
                 "stale_folder": ("geo_final folders with .done for targets that are not built now", "rows !=")}
        for name, (want, notwant) in cases.items():
            rootc = tmp / name
            envc, _ = fixture(rootc, mods=(name,))
            rcc, _, errc = build(envc)
            repc = report(rootc)
            text = errc + json.dumps(repc.get("stop_reasons", []))
            okc = rcc == 3 and want in text and (notwant is None or notwant not in text) and repc.get("stop") is True \
                and repc.get("running") is False                                 # S-2: STOP is never 'running'
            if name in ("c_select", "d0_folds_sha", "stray", "two_folds", "d0_wrong_fold", "d_report_c_select"):
                okc &= not (rootc / "geo2" / "geo_final").exists()               # stopped before any folder write
            if name == "d0_fallback_excess":
                okc &= json.loads((rootc / "geo2" / "geo_final" / "6" / "result.json").read_text())["fallback"] is True
            ck[f"stop_{name}"], det[f"stop_{name}"] = okc, dict(rc=rcc, tail=errc[-400:])
        for name in ("no_pkl", "no_c_select", "no_e_rxn", "no_d_report"):        # S-2: upstream missing = exit 2
            rootc = tmp / name
            envc, _ = fixture(rootc, mods=(name,))
            rcc, _, errc = build(envc)
            okc = rcc == 2 and "NOT READY" in errc and not (rootc / "res" / "Z_report.json").exists() \
                and not (rootc / "geo2" / "geo_final").exists()                  # nothing written
            ck[f"not_ready_{name}"], det[f"not_ready_{name}"] = okc, dict(rc=rcc, tail=errc[-400:])
        rootc = tmp / "bad_pkl"                                                  # S-2: a crash = stopped report
        envc, _ = fixture(rootc, mods=("bad_pkl",))
        rcc, _, errc = build(envc)
        repc = report(rootc)
        ck["crash_stopped_report"] = (rcc not in (0, 2) and repc.get("stop") is True and repc.get("running") is False
                                      and repc.get("complete") is False and "Unpickling" in json.dumps(repc)
                                      and not (rootc / "geo2" / "geo_final").exists())
        det["crash_stopped_report"] = dict(rc=rcc, report=repc, tail=errc[-400:])
        rootc = tmp / "e_rxn_rescued"                                            # E2-rule: E_rxn compares mean3 rows
        envc, _ = fixture(rootc, mods=("e_rxn_rescued",))                        # only (a wrong mean_pass row passes)
        rcc, _, errc = build(envc)
        ec = report(rootc).get("by_src", {}).get("D1", {}).get("e_rxn_check", {})
        ck["e_rxn_scope_rescued_not_compared"] = (rcc == 0 and report(rootc).get("stop") is False
                                                  and ec.get("n") == n_f04_m3 and ec.get("max_abs_dev_A", 1) <= 1e-9
                                                  and ec.get("n_in_file_not_mean3") == 4)
        det["e_rxn_scope_rescued_not_compared"] = dict(rc=rcc, tail=errc[-400:])
        import build_geo as B                                                    # E2-rule units on BASE (G2 = BASE)
        pp = C.parse_pairs(PAIRS)
        ey, ez = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        S3 = np.array([BASE.copy() for _ in range(3)])                           # all 3 seeds pass, H5 on 3 sides of
        for k, u in enumerate((ey, -ey / 2 + np.sqrt(3) / 2 * ez, -ey / 2 - np.sqrt(3) / 2 * ez)):   # C4: their mean
            S3[k][5] = BASE[4] + 1.133 * u                                       # (= mean3 = mean_pass) puts H5 on C4
        cr = B.choose_coords(SYMS, S3, BASE, pp)
        ck["unit_all_pass_mean_fails_seed0"] = (
            cr["rule"] == "seed0" and cr["seeds"] == [0] and cr["seed_reasons"] == ["", "", ""]
            and [(t["rule"], t["ok"], t["seeds"]) for t in cr["tried"]] == [("mean3", False, [0, 1, 2]),
                                                                           ("mean_pass", False, [0, 1, 2]),
                                                                           ("seed0", True, [0])]
            and np.array_equal(cr["X"], S3[0]) and cr["mean3"][1] == "close_contact")
        cr1 = B.choose_coords(SYMS, np.array([BASE.copy() for _ in range(3)]), BASE, pp)
        S0 = np.array([BASE.copy() for _ in range(3)])
        S0[:, 1] = BASE[0] + np.array([0.3, 0.0, 0.0])
        cr4 = B.choose_coords(SYMS, S0, BASE, pp)
        ck["unit_mean3_and_fallback"] = (cr1["rule"] == "mean3" and cr1["seed_reasons"] is None
                                         and len(cr1["tried"]) == 1 and np.allclose(cr1["X"], BASE, rtol=0, atol=1e-12)
                                         and cr4["rule"] == "g2c_fallback" and cr4["seeds"] == [] and cr4["X"] is BASE
                                         and cr4["seed_reasons"] == ["close_contact"] * 3
                                         and np.allclose(B.rule_coords(S3, [0, 2], BASE), (S3[0] + S3[2]) / 2, rtol=0,
                                                         atol=1e-12)
                                         and B.rule_coords(S3, [], BASE) is BASE)
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
    tmp = Path(tempfile.mkdtemp(prefix="geo2b_etest_"))
    # load_d0 final rules (E-9, 연결 S-4): g2_source g2c, D0_pairs.csv = the S_splits_d0d1.json record
    save = (C.D0_PAIRS_REPORT, C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS, E.S_SPLITS)
    try:
        C.D0_PAIRS_REPORT, C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS, E.S_SPLITS = (
            tmp / "D0_pairs_report.json", tmp / "D0_pairs.csv", tmp / "d0_folds.csv", tmp / "d0_twins.csv",
            tmp / "S_splits_d0d1.json")
        D.reset_index().to_csv(C.D0_PAIRS_CSV, index=False)
        write_csv(C.D0_FOLDS, [dict(rxn_id=r, fold=f) for r, f in folds.items()], ["rxn_id", "fold"])
        write_csv(C.D0_TWINS, [dict(rxn_id=r, m1="True") for r in m1], ["rxn_id", "m1"])

        def d0rep(src, csv_sha=None):
            outs = {str(p): dict(sha256=C.sha256(p)) for p in (C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS)}
            if csv_sha:
                outs[str(C.D0_PAIRS_CSV)] = dict(sha256=csv_sha)
            C.D0_PAIRS_REPORT.write_text(json.dumps(dict(stop=False, g2_source=src, outputs=outs)))
        sha0 = C.sha256(C.D0_PAIRS_CSV)
        E.S_SPLITS.write_text(json.dumps(dict(d0=dict(d0_pairs_sha256=sha0))))
        d0rep("g2c")
        Dl, fl, ml, info = E.load_d0(final=True)
        ck["load_d0_final_ok"] = (len(Dl) == n and fl == folds and ml == m1 and str(E.S_SPLITS) in info)
        d0rep("d0_g2_orig")
        ck["load_d0_preliminary_not_ready"] = _expect_exit(lambda: E.load_d0(final=True), 2)
        ck["load_d0_preliminary_selftest_ok"] = len(E.load_d0(final=False)[0]) == n
        d0rep("d0_g2_orig", csv_sha="f" * 64)                    # preliminary report + D0_pairs.csv rebuilt: still 2
        ck["load_d0_preliminary_rebuilt_not_ready"] = _expect_exit(lambda: E.load_d0(final=True), 2)
        d0rep("g2c")
        E.S_SPLITS.write_text(json.dumps(dict(d0=dict(d0_pairs_sha256="0" * 64))))
        ck["stop_load_d0_rebuilt_after_splits"] = E._expect_stop(lambda: E.load_d0(final=True))
        E.S_SPLITS.unlink()
        ck["load_d0_no_splits_not_ready"] = _expect_exit(lambda: E.load_d0(final=True), 2)
    finally:
        C.D0_PAIRS_REPORT, C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS, E.S_SPLITS = save
    # unit records: robust names, two reports, folds 0-4
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
