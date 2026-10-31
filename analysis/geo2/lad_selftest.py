#!/usr/bin/env python3
r"""lad_selftest.py — CPU self-tests of the GEO2b LADDER-training slice (train_atoms.py / train_xgb.py stage 3, split T1D0,
d0d1_splits.py) that run before any geo input exists (DECISIONS_lad.md LD-T*). Compute node only (srun inside a running
allocation or sbatch); writes only below $GEO2B/tests_lad (the test split build and one json per part). Never trains a
production unit, never writes $LAD/units or any LADDER / analysis/*/results file.

  python lad_selftest.py splits     any pandas env: d0d1_splits build --test-dir on the current GEO1b D0 files
                                    (--allow-preliminary while GEO1b is preliminary), idempotency, reader check, negative
                                    rule tests, production refusal of a preliminary build
  python lad_selftest.py lists      any env: `train_atoms.py list --stage 1|2|3` (env_d1fu2) and `train_xgb.py list
                                    --stage 1|2|3` (env_espley) in fresh processes = the LADDER unit lists recorded in
                                    results/L_report.json / L_xgb_report.json (regression 1) and the 30 / 10 geo units
  python lad_selftest.py xgb        env_espley: train_xgb L-11' label load + load_folds('T1D0') on the test build;
                                    rung g2 forced to split T1D0 (RUNG_SPLIT, in-process only) on the test labels: every
                                    fold's usable lists / inner folds = the LADDER T1 loader's (D0 pairs have no g2 token)
                                    and the T1D0 rules (fold 0 = file, no D0 on a validation side); diag rule on it
  python lad_selftest.py atoms      env_d1fu2 (CPU is enough): the same for train_atoms (LadData)
  exit 0 = every check passed, 3 = a check failed, 2 = an input missing. Results: $GEO2B/tests_lad/selftest_<part>.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LADC = REPO / "analysis" / "d1_ladder"
sys.path.insert(0, str(HERE))
import d0d1_splits as DD  # noqa: E402

TEST = DD.GEO2 / "tests_lad"
TDIR = TEST / "splits_prelim"
PY_ATOMS = "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python"
PY_XGB = "/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python"
RES_LAD = LADC / "results"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def out_json(part, rec):
    p = TEST / f"selftest_{part}.json"
    DD.write_bytes_atomic(p, (json.dumps(rec, indent=1, default=str) + "\n").encode())
    return p


def run(part, checks):
    """checks = [(name, fn)] -> each fn returns (ok, info); exceptions = fail with the traceback."""
    rec = dict(part=part, started=now(), host=os.uname().nodename, python=sys.version.split()[0], checks={})
    ok_all = True
    for name, fn in checks:
        t0 = time.time()
        try:
            ok, info = fn()
        except SystemExit as e:
            ok, info = False, dict(exit=e.code, why="SystemExit inside the production code")
        except Exception as e:                                         # noqa: BLE001
            ok, info = False, dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        rec["checks"][name] = dict(ok=bool(ok), seconds=round(time.time() - t0, 1), info=info)
        ok_all &= bool(ok)
        print(f"[{part}] {name}: {'PASS' if ok else 'FAIL'} ({rec['checks'][name]['seconds']} s)", flush=True)
    rec.update(all_pass=bool(ok_all), finished=now())
    print(f"[{part}] all_pass {ok_all} -> {out_json(part, rec)}", flush=True)
    return 0 if ok_all else 3


def use_test_build():
    DD.set_paths(*DD.test_paths(TDIR))
    DD.ALLOW_PRELIMINARY = True


# ================================================================ splits
def part_splits():
    def build():
        prelim = bool((json.loads(DD.D0_REPORT.read_text())).get("preliminary") is not False)
        r1 = DD.build(TDIR, allow_preliminary=prelim)
        shas = {k: v for k, v in r1["outputs"].items()}
        jpath = DD.test_paths(TDIR)[2]
        j1 = DD.sha256(jpath)
        r2 = DD.build(TDIR, allow_preliminary=prelim)
        same = shas == r2["outputs"] and DD.sha256(jpath) == j1 and bool(r2.get("up_to_date"))
        n = {k: v["n"] for k, v in r1["units"].items()}
        nd0 = {k: v["n_d0"] for k, v in r1["units"].items()}
        ok = same and all(v["inner_val"] == 0 for v in nd0.values()) and r1["d0"]["n_ok"] > 0
        return ok, dict(preliminary=r1["preliminary"], idempotent_outputs=same, n=n, n_d0=nd0, d0=r1["d0"],
                        labels=r1["labels"], twins=r1["twins"])

    def reader():
        use_test_build()
        rec, lists, files = DD.load_s_splits()
        t1, _ = DD.fu2_t1()
        same_d1 = all([q for q in lists[k][key] if not DD.is_d0_id(q)] == list(t1[k][key])
                      for k in range(DD.NF) for key in DD.FOLD_KEYS)
        return same_d1, dict(n_files=len(files), d1_parts_equal_FU2_T1=same_d1)

    def negatives():
        use_test_build()
        rec, lists, files = DD.load_s_splits()
        t1, _ = DD.fu2_t1()
        T = DD.read_d0_pairs()
        pf = DD.pair_folds(T, DD.read_d0_folds())
        okd = sorted(str(q) for q, o in zip(T.pair_id, T.ok) if o == "True")
        f0 = lists[0]
        d0tr = [q for q in f0["train"] if DD.is_d0_id(q)]
        cases = {
            "D0 pair in inner_val": dict(f0, inner_val=list(f0["inner_val"]) + d0tr[:1]),
            "D1 train list changed": dict(f0, train=list(f0["train"])[1:]),
            "D0 before D1": dict(f0, test=[q for q in f0["test"] if DD.is_d0_id(q)]
                                 + [q for q in f0["test"] if not DD.is_d0_id(q)]),
            "D0 test pair in train": dict(f0, train=list(f0["train"]) + [q for q in f0["test"] if DD.is_d0_id(q)][:1]),
            "inner_train D0 != train D0": dict(f0, inner_train=[q for q in f0["inner_train"] if q not in set(d0tr[:3])]),
            "an ok D0 pair missing": dict(f0, train=[q for q in f0["train"] if q != (d0tr[-1] if d0tr else "")],
                                          inner_train=[q for q in f0["inner_train"]
                                                       if q != (d0tr[-1] if d0tr else "")]),
        }
        res = {}
        for name, f in cases.items():
            try:
                DD.check_lists(0, f, t1[0], pf, okd)
                res[name] = "NOT caught"
            except DD.Stop as e:
                res[name] = f"caught: {str(e)[:120]}"
        c = DD.CR2.read_bytes()
        b = DD.LABELS.read_bytes()
        bad_bytes = {"D1 byte changed": b[:100] + (b"X" if b[100:101] != b"X" else b"Y") + b[101:],
                     "D0 row dropped": b[:b.rstrip(b"\n").rfind(b"\n") + 1],
                     "only D1 rows": c}
        for name, bb in bad_bytes.items():
            try:
                DD.l11_d0d1(label_bytes=bb)
                res[name] = "NOT caught"
            except DD.Stop as e:
                res[name] = f"caught: {str(e)[:120]}"
        ok = all(v.startswith("caught") for v in res.values())
        return ok, res

    def production_refuses_preliminary():
        rep = json.loads(DD.D0_REPORT.read_text())
        if rep.get("preliminary") is False:
            return True, dict(note="GEO1b report is final: production build allowed (not run by this test)")
        DD.set_paths(*DD.test_paths(TDIR))
        DD.ALLOW_PRELIMINARY = False
        try:
            DD.load_s_splits()
            return False, dict(why="the reader accepted a preliminary build without ALLOW_PRELIMINARY")
        except DD.Stop as e:
            r = str(e)
        try:
            DD.build(None, allow_preliminary=True)
            return False, dict(why="build --allow-preliminary without --test-dir accepted")
        except DD.Stop as e:
            r2 = str(e)
        return True, dict(reader=r[:200], build=r2[:200])
    return run("splits", [("build_test_dir", build), ("reader", reader), ("negatives", negatives),
                          ("production_refuses_preliminary", production_refuses_preliminary)])


# ================================================================ lists (regression 1)
def _ladder_lists():
    lr = json.loads((RES_LAD / "L_report.json").read_text())
    xr = json.loads((RES_LAD / "L_xgb_report.json").read_text())
    atoms = [n for n in lr["units"] if "__XGB__" not in n]
    xgb = list(xr["units"])
    st = {1: ("dft", "g1"), 2: ("g2",)}
    return ({s: [n for n in atoms if n.split("__")[1] in st[s]] for s in st},
            {s: [n for n in xgb if n.split("__")[1] in st[s]] for s in st})


def part_lists():
    A, X = _ladder_lists()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")

    def one(py, script, want):
        def f():
            info, ok = {}, True
            for s in (1, 2, 3):
                p = subprocess.run([py, str(LADC / script), "list", "--stage", str(s)], capture_output=True, text=True,
                                   cwd=str(LADC), env=env)
                got = p.stdout.splitlines()
                w = want[s]
                e = dict(exit=p.returncode, n=len(got), n_want=len(w), equal=got == w,
                         stderr=p.stderr[-800:] if p.returncode else "")
                info[f"stage{s}"] = e
                ok &= p.returncode == 0 and got == w
            return ok, info
        return f
    geo_atoms = [f"L__geo__{a}__{m}__f{k}" for m in ("GNN", "AW", "TF") for a in ("S", "SH") for k in range(5)]
    geo_xgb = [f"L__geo__{a}__XGB__f{k}" for a in ("S", "SH") for k in range(5)]
    return run("lists", [("train_atoms_list", one(PY_ATOMS, "train_atoms.py", {1: A[1], 2: A[2], 3: geo_atoms})),
                         ("train_xgb_list", one(PY_XGB, "train_xgb.py", {1: X[1], 2: X[2], 3: geo_xgb}))])


# ================================================================ trainers on the test build (rung g2 forced to T1D0)
def _trainer_checks(M, tool):
    import numpy as np
    use_test_build()
    lab = DD.LABELS

    def labels():
        P, info = M.load_labels(lab)
        d0 = P.ref_kind.eq("d0")
        ok = bool(isinstance(info["l11"], dict) and info["l11"].get("d1_prefix_equal_F_pairs_cr2")
                  and info["l11"].get("d0_rows_equal_D0_pairs_text") and int(d0.sum()) == info["l11"]["n_d0_rows"])
        return ok, dict(provenance=info["provenance"], l11={k: v for k, v in info["l11"].items()
                                                            if k != "F_pairs_cr2_l11"}, n=int(len(P)),
                        n_d0=int(d0.sum()))

    def folds():
        r = M.load_folds("T1D0")
        n = len(r[1]) if isinstance(r, tuple) else len(r["folds"])
        return n == 5, dict(n_folds=n)

    def forced_g2():
        M.RUNG_SPLIT["g2"] = "T1D0"
        try:
            Dz = M.get_data("g2", "S", lab)
        finally:
            M.RUNG_SPLIT.pop("g2", None)
        D1 = M.get_data("g2", "S", M.DEFAULT_LABELS)
        info, ok = {}, Dz.split_name == "T1D0" and D1.split_name == "T1"
        pidz, pid1 = (np.asarray(Dz.P.index), np.asarray(D1.P.index)) if tool == "atoms" else (Dz.pid, D1.pid)
        for k in range(5):
            a, b = Dz.split(k), D1.split(k)
            e = {key: [str(q) for q in pidz[a[key]]] == [str(q) for q in pid1[b[key]]] for key in M.FOLD_KEYS}
            e["val_sha"] = a["val_sha"] == b["val_sha"]
            e["test_sha"] = a["test_sha"] == b["test_sha"]
            e["n_d0"] = a.get("n_d0")
            e["d0_dropped_unusable"] = {key: a["dropped"][key]["not_usable"] - b["dropped"][key]["not_usable"]
                                        for key in M.FOLD_KEYS}
            if tool == "xgb":
                e["inner_equal"] = all([str(q) for q in pidz[fa[s]]] == [str(q) for q in pid1[fb[s]]]
                                       for fa, fb in zip(a["inner"], b["inner"]) for s in ("train", "val"))
            ok &= all(v for kk, v in e.items() if kk not in ("n_d0", "d0_dropped_unusable")) \
                and a["n_d0"] == {key: 0 for key in M.FOLD_KEYS} and e["d0_dropped_unusable"]["inner_val"] == 0
            info[f"f{k}"] = e
        g = M.gate_d0_train(None, None, Dz, Dz.split(0)["train"], 0) if tool == "atoms" else "n/a"
        info["gate_d0_train_without_usable_d0"] = g
        ok &= g is None or tool != "atoms"
        return ok, info
    return [("labels_l11_prime", labels), ("load_folds_T1D0", folds), ("g2_forced_T1D0_equals_T1", forced_g2)]


def part_xgb():
    sys.path.insert(0, str(LADC))
    import train_xgb as TX
    checks = _trainer_checks(TX, "xgb")

    def gates_absent():
        # the stage-3 gates refuse without PREREG_GEO2B / C_tests_geo / P (nothing of geo exists yet): exit 3 or 2
        res = {}
        for name, fn in (("gate_prereg_geo2", lambda: TX.gate_prereg_geo2({str(DD.LABELS)}, ("geo",))),
                         ("gate_c_geo", TX.gate_c_geo), ("gate_p_geo", lambda: TX.gate_p_geo(("geo",), DD.LABELS))):
            try:
                fn()
                res[name] = "passed (unexpected before the geo inputs / prereg exist)"
            except SystemExit as e:
                res[name] = f"exit {e.code}"
        ok = all(v.startswith("exit") for v in res.values())
        return ok, res

    def guard():
        try:
            TX.guard_ladder_write(1, "selftest")
            return False, dict(why="stage-1 write command not refused with the modified code")
        except SystemExit as e:
            return e.code == 3, dict(exit=e.code)
    return run("xgb", checks + [("stage3_gates_refuse_now", gates_absent), ("ladder_write_guard", guard),
                                ("synthetic_T1D0_split_and_run_unit", lambda: synthetic_xgb(TX))])


def synthetic_xgb(TX):
    """the production XData.split (T1D0 branch) and run_unit (zero model, as run_tests L-3) on a synthetic rung with D0
    pairs: GroupKFold on the D1 part only, every D0 training pair on every inner training side and no validation side,
    fold 0 = the 'split file', a D0 pair in inner_val = STOP, D0 gate pairs added, n_d0 recorded."""
    import numpy as np
    import pandas as pd
    import types
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(20261031)
    NT = len(TX.TARGETS)
    n1, n0 = 80, 40                                            # D1 reactions 0..79 (cores c0..c7), D0 reactions 80..119
    nr = n1 + n0
    H10 = np.zeros((nr, NT))
    H10[:, TX.H_IDX] = rng.normal(size=(nr, len(TX.H_IDX))) * 3.0
    pairs = []
    for c in range(8):                                         # D1 pairs inside each core
        rx = list(range(10 * c, 10 * c + 10))
        for i in range(1, 10):
            pairs.append((f"P{len(pairs):05d}", "reference", rx[i], rx[0], f"core{c}"))
    nd1 = len(pairs)
    for i in range(1, 30):                                     # D0 pairs: train side 80..109, test side 110..119
        pairs.append((f"D{i:05d}", "d0", 80 + i, 80, "D0/c0000"))
    for i in range(1, 10):
        pairs.append((f"D{100 + i:05d}", "d0", 110 + i, 110, "D0/c0001"))
    pid = np.array([p[0] for p in pairs], dtype=object)
    ra = np.array([p[2] for p in pairs])
    rb = np.array([p[3] for p in pairs])
    core = np.array([p[4] for p in pairs], dtype=object)
    is_d0 = np.array([p[1] == "d0" for p in pairs])
    dH = H10[ra] - H10[rb]
    y = 2.0 * dH + rng.normal(0, 0.3, size=dH.shape)
    d1_test = [p for p, c in zip(pid[:nd1], core[:nd1]) if c in ("core0", "core1")]
    d1_train = [p for p, c in zip(pid[:nd1], core[:nd1]) if c not in ("core0", "core1")]
    row = {p: k for k, p in enumerate(pid)}
    tr1 = np.array([row[p] for p in d1_train])
    ia, ib = next(iter(GroupKFold(4).split(tr1, groups=core[tr1])))
    d0_train = sorted(p for p in pid[nd1:] if p < "D00100")
    d0_test = sorted(p for p in pid[nd1:] if p >= "D00100")
    rec = dict(split="T1D0", fold=0, train=d1_train + d0_train, test=d1_test + d0_test,
               inner_train=[pid[tr1[i]] for i in ia] + d0_train, inner_val=[pid[tr1[i]] for i in ib])
    P = pd.DataFrame(dict(ref_kind=[p[1] for p in pairs], pos=["p" if p[1] != "d0" else "d0" for p in pairs],
                          core=core), index=pid)
    X0 = rng.normal(size=(nr, 6))

    def make(arm, folds):
        D = TX.XData.__new__(TX.XData)
        D.rung, D.arm, D.split_name, D.labels = "geo", arm, "T1D0", "synthetic"
        D.rids = np.arange(nr)
        D.X = X0 if arm == "S" else np.concatenate([X0, H10[:, TX.H_IDX]], 1)
        D.feat_cols = [f"x{i}" for i in range(D.X.shape[1])]
        D.feat_cols_sha = "synthetic"
        D.H10, D.dH, D.y, D.P, D.pid, D.pid_row = H10, dH, y, P, pid, row
        D.ra, D.rb, D.core, D.is_d0 = ra, rb, core, is_d0
        D.has_tok = np.ones(len(pid), bool)
        D.usable_pair = np.ones(len(pid), bool)
        D.usable_rxn = np.ones(nr, bool)
        D.folds, D._splits = folds, {}
        D.sha = {k: "synthetic" for k in ("inputs_table", "labels", "s_splits", "s_folds", "t_report", "t_manifest",
                                          "q_report", "x182")}
        D.pair_list_sha, D.hinfo, D.inputs, D.linfo, D.opened = "synthetic", dict(h_cols=[]), {}, {}, []
        return D
    info, ok = {}, True
    D = make("SH", {0: rec})
    sp = D.split(0)
    d0rows = {row[p] for p in d0_train}
    inner = [dict(train_has_all_d0=d0rows <= set(f["train"].tolist()), val_d0=int(is_d0[f["val"]].sum()),
                  n_train=len(f["train"]), n_val=len(f["val"])) for f in sp["inner"]]
    info["inner"] = inner
    ok &= all(x["train_has_all_d0"] and x["val_d0"] == 0 for x in inner) and len(inner) == 4
    info["n_d0"] = sp["n_d0"]
    ok &= sp["n_d0"] == dict(train=len(d0_train), test=len(d0_test), inner_train=len(d0_train), inner_val=0)
    bad = dict(rec, inner_val=rec["inner_val"] + d0_train[:1], inner_train=[q for q in rec["inner_train"]
                                                                            if q != d0_train[0]])
    try:
        make("SH", {0: bad}).split(0)
        info["d0_in_inner_val"] = "NOT caught"
        ok = False
    except SystemExit as e:
        info["d0_in_inner_val"] = f"caught (exit {e.code})"
        ok &= e.code == 3
    real = TX.run_parallel

    def zero(calls):
        return [({k: np.zeros(len(zd)) for k, (zd, zm) in c[5].items()}, 0.0) for c in calls]
    TX.run_parallel = zero
    try:
        root = TX.L.LAD / "tests_geo" / "lad_selftest_xgb"
        TX.L.assert_path(root / "probe")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        r = TX.run_unit(D, dict(rung="geo", arm="SH", model="XGB", fold=0), root=root, cells=[0], smoke=True)
    finally:
        TX.run_parallel = real
    info["unit"] = dict(n_gate_pairs=r["n_gate_pairs"], n_gate_pairs_d0_train=r.get("n_gate_pairs_d0_train"),
                        n_d0=r.get("n_d0"), n_d0_inner=r.get("n_d0_inner"), split=r.get("split"),
                        fp_split=r["fingerprint"].get("split"), fp_d0d1_code=bool(r["fingerprint"].get("d0d1_code")))
    ok &= (r.get("n_gate_pairs_d0_train") == len(d0_train) and r["n_gate_pairs"] == len(sp["test"]) + len(d0_train)
           and r.get("split") == "T1D0" and all(x["val"] == 0 and x["train"] == len(d0_train) for x in r["n_d0_inner"]))
    return ok, info


def part_atoms():
    sys.path.insert(0, str(LADC))
    import train_atoms as TA
    real, cache = TA.load_tokens, {}

    def load_tokens_once(r):                       # test speed only: the g2 tokens are read once for both label files
        if r not in cache:
            cache[r] = real(r)
        return dict(cache[r])                      # rung_base pops 'inputs' / 'sha' from its copy
    TA.load_tokens = load_tokens_once
    checks = _trainer_checks(TA, "atoms")

    def guard():
        try:
            TA.guard_ladder_write(1, "selftest")
            return False, dict(why="stage-1 write command not refused with the modified code")
        except SystemExit as e:
            return e.code == 3, dict(exit=e.code)

    def estimate_rule():
        # smoke_estimate_geo on a fake geo smoke record = the LADDER g1 smoke itself -> ratio 1 -> = LADDER hours
        sm = json.loads((RES_LAD / "C_smoke_atoms.json").read_text())
        fake = dict(versions=sm.get("versions"), configs={n.replace("L__g1__", "L__geo__"): c
                                                          for n, c in sm["configs"].items() if n.startswith("L__g1__")})
        e = TA.smoke_estimate_geo(fake)
        lr = json.loads((RES_LAD / "L_report.json").read_text())
        g1h = sum(float(v["wall_seconds"]) for k, v in lr["units"].items()
                  if k.startswith("L__g1__") and "__XGB__" not in k) / 3600
        ok = bool(e.get("ok") and abs(e["by_reference"]["g1"] - g1h) < 1e-9 and e["limit_gpu_hours"] == 160.0)
        return ok, dict(estimate=e, g1_ladder_hours=g1h)
    return run("atoms", checks + [("ladder_write_guard", guard), ("smoke_estimate_rule", estimate_rule)])


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        if a == ["splits"]:
            sys.exit(part_splits())
        if a == ["lists"]:
            sys.exit(part_lists())
        if a == ["xgb"]:
            sys.exit(part_xgb())
        if a == ["atoms"]:
            sys.exit(part_atoms())
    except DD.NotReady as e:
        print(f"NOT READY: {e}", file=sys.stderr)
        sys.exit(2)
    raise SystemExit(__doc__)
