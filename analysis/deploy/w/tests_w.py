#!/usr/bin/env python3
r"""tests_w.py — DEPLOY-MODEL-W Phase C code checks (docs/specs/DEPLOY_MODEL_W.md 6) + the predict_tf extension test
(13-1) + I2 (13-2). Any failing check = STOP (exit 3); not ready = exit 2.

  python tests_w.py cpu [--source lf|g2]   C-W1 (labels), C-W2 (leak), C-W3 (r = 0 = fu2 fit, CPU, bitwise), C-W5 (V_nl),
                                           C-W6 (sampling), C-W7 (graph readout); env_d1fu2 on CPU (jobs/cpu_py.slurm
                                           DEPENV=d1fu2). lf -> results/w/C_part_cpu_w.json; g2 (dry run before the lf
                                           inputs exist; within pools from $DEPW/dev/within_g2) -> $DEPW/dev/
  python tests_w.py cw4 [--source lf|g2]   C-W4: the standardisation of fit_w on fold k = the statistics stored in
                                           $DEP/units/D__lf__S__TF__f<k>__s0..4.pt (<= 1e-12); exit 2 until those exist.
                                           lf -> results/w/C_part_cw4_w.json; g2 = against the DEPLOY developer units
                                           $DEP/dev_tf/kunits (limit 150, train_tf._limit) -> $DEPW/dev/
  python tests_w.py gpu                    C-W9 (train_w.py smoke on GPU, lf data) + C-W8 of its fits
                                           -> results/w/C_part_gpu_w.json
  python tests_w.py merge                  every part present, ok, run on the current code -> results/w/C_tests_w.json
  python tests_w.py patch                  13-1 patch file analysis/deploy/w/predict_tf_ext.patch: applies to the current
                                           predict_tf.py (git apply --check); patched copy in $DEPW/dev/patch behaves as the
                                           original with default options (cv / final outputs and the I-1 / I-2 records on
                                           the DEPLOY developer units); --readout graph = w_common.graph_values
                                           -> results/w/patch_test.json
  python tests_w.py i2                     I2-0 .. I2-3 (+ I2-4 when DEPW_I3_DIR is set) with the patched predict_tf.py
                                           -> results/w/I2_selftest.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402

C = W.C
PART_FILES = {"cpu": "C_part_cpu_w.json", "cw4": "C_part_cw4_w.json", "gpu": "C_part_gpu_w.json"}
PART_CHECKS = {"cpu": ("C-W1", "C-W2", "C-W3", "C-W5", "C-W6", "C-W7"), "cw4": ("C-W4",), "gpu": ("C-W8", "C-W9")}
PART_CODE = {"cpu": ("w/tests_w.py", "w/w_common.py", "w/within.py", "w/train_w.py", "data_lf.py", "dep_common.py",
                     "train_tf.py"),
             "cw4": ("w/tests_w.py", "w/w_common.py", "w/within.py", "w/train_w.py", "data_lf.py", "dep_common.py",
                     "train_tf.py"),
             "gpu": ("w/tests_w.py", "w/w_common.py", "w/within.py", "w/train_w.py", "data_lf.py", "dep_common.py",
                     "train_tf.py")}
TRAINER_FILES = ("w/train_w.py", "w/within.py", "w/w_common.py", "train_tf.py", "data_lf.py", "dep_common.py")
TOL_LABEL = 1e-6
TOL_STD = 1e-12
TOL_GRAPH = 1e-12
TOL_VNL = 1e-9
SAMPLE_EPOCHS = 10000
SAMPLE_TOL = 0.02


def code_sha(names):
    return {n: (C.sha256(W.DEPLOY_DIR / n) if (W.DEPLOY_DIR / n).is_file() else None) for n in names}


def _ok(rec):
    return bool(isinstance(rec, dict) and rec.get("ok") is True)


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except SystemExit as e:
        return dict(ok=False, error=f"SystemExit {e.code}", traceback=traceback.format_exc()[-3000:])
    except Exception as e:                                         # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])


def _write_part(part, checks, out=None, extra=None):
    rec = dict(part=part, ok=bool(all(_ok(v) for v in checks.values())), checks=checks,
               code_sha256=code_sha(PART_CODE[part]), python=sys.version.split()[0], host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), time=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    path = out or (W.RESW / PART_FILES[part])
    W.write_json(path, rec)
    print(json.dumps(dict(part=part, ok=rec["ok"], checks={k: _ok(v) for k, v in checks.items()}, file=str(path)),
                     indent=1))
    return 0 if rec["ok"] else 3


def _within_root(source):
    import within as WI
    return WI.default_root(source)


# ================================================================ C-W1 labels
def _label_frame(records, cols):
    import pandas as pd
    Y = pd.DataFrame([{**{"rxn_id": int(d["rxn_id"])}, **{t: float(d[c]) for t, c in cols.items()}} for d in records])
    if Y.rxn_id.duplicated().any():
        W.stop("duplicated rxn_id in a label file")
    return Y.set_index("rxn_id")


def cr2_records():
    """$LAD/labels_d1_cr2.json after the LADDER provenance rule (V_report.json all_pass, outputs sha256 equal)."""
    p = C.LAD / "labels_d1_cr2.json"
    rep_p = C.REPO / "analysis" / "d1_ladder" / "results" / "V_report.json"
    rep = W.read_json(rep_p)
    if not rep.get("all_pass") or (rep.get("outputs") or {}).get(str(p)) != W.sha256(p):
        W.stop(f"{p}: not the V_report.json all_pass output")
    return json.loads(p.read_text()), dict(path=str(p), sha256=W.sha256(p), v_report=str(rep_p))


def cw1(n_sample=2000):
    """dd of F_pairs_cr2.csv = y_a - (common y_b of the group) for every group and target; random within pairs:
    dd_i - dd_j = y_a_i - y_a_j (per-reaction values). Per-reaction values first from d1pair_common.load_d1_labels();
    groups that fail are checked against the cr2 corrections (labels_d1_cr2.json) and recomputed with them."""
    import numpy as np
    import data_lf as DL
    DP = DL.DP
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_pair"))
    import d1_pairs as PB
    cols = dict(PB.TARGETS)
    if cols != dict(DP.D1_TCOL) or list(cols) != list(W.TARGETS):
        W.stop("d1_pairs.TARGETS / d1pair_common.D1_TCOL / TARGETS disagree")
    P, linfo = DL.load_labels(DL.DEFAULT_LABELS)
    S = W.load_s_folds()
    agree = W.check_pairs_vs_sfolds(S, P)
    _, gid = W.group_table(S)
    g = gid.reindex(P.index).to_numpy()
    dd = P[[f"dd_{t}" for t in W.TARGETS]].to_numpy(float)
    a, b = P.a_rxn.to_numpy(int), P.b_rxn.to_numpy(int)
    Y0 = _label_frame(DP.load_d1_labels(), cols)
    rec2, info2 = cr2_records()
    Y2 = _label_frame(rec2, cols)

    def run(Y, name):
        miss = sorted((set(a) | set(b)) - set(Y.index))
        if miss:
            return dict(ok=False, error=f"{len(miss)} reactions missing in {name}")
        ya, yb = Y.loc[a].to_numpy(float), Y.loc[b].to_numpy(float)
        yb_hat = ya - dd
        import pandas as pd
        F = pd.DataFrame(yb_hat, columns=list(W.TARGETS)).assign(gid=g)
        sd = F.groupby("gid").std(ddof=0)
        bad = sd[(sd > TOL_LABEL).any(axis=1)]
        rng = np.random.default_rng(W.SEED)
        groups = [np.where(g == x)[0] for x in np.unique(g)]
        groups = [q for q in groups if len(q) >= 2]
        sizes = np.array([len(q) * (len(q) - 1) // 2 for q in groups], dtype=float)
        pick = rng.choice(len(groups), size=n_sample, p=sizes / sizes.sum())
        ii, jj = [], []
        for q in pick:
            i, j = rng.choice(groups[q], size=2, replace=False)
            ii.append(i)
            jj.append(j)
        ii, jj = np.array(ii), np.array(jj)
        d_label = dd[ii] - dd[jj]
        d_rxn = Y.loc[a[ii]].to_numpy(float) - Y.loc[a[jj]].to_numpy(float)
        pmax = np.abs(d_label - d_rxn).max(0)
        return dict(ok=bool(len(bad) == 0 and (pmax <= TOL_LABEL).all()),
                    max_within_group_sd={t: float(sd[t].max()) for t in W.TARGETS},
                    n_groups=int(len(sd)), n_groups_failing=int(len(bad)), failing_gids=[int(x) for x in bad.index[:50]],
                    failing_targets={t: int((sd[t] > TOL_LABEL).sum()) for t in W.TARGETS},
                    yb_hat_vs_yb_max={t: float(np.abs(yb_hat[:, j] - yb[:, j]).max()) for j, t in enumerate(W.TARGETS)},
                    n_sample=int(n_sample), sample_max_abs_diff={t: float(pmax[j]) for j, t in enumerate(W.TARGETS)},
                    _bad=set(int(x) for x in bad.index))
    r0, r2 = run(Y0, "labels_d1.json"), run(Y2, "labels_d1_cr2.json")
    # cr2 corrections of the a-side reactions of the groups that fail with the original values
    changed = set()
    common = sorted(set(Y0.index) & set(Y2.index))
    dif = (Y0.loc[common] - Y2.loc[common]).abs().max(axis=1)
    changed = set(int(x) for x in dif.index[dif.to_numpy() > 0])
    bad0 = r0.pop("_bad", set()) if isinstance(r0, dict) else set()
    r2.pop("_bad", None)
    a_changed = {int(x): bool(set(a[g == x].tolist()) & changed) for x in bad0}
    explained = bool(bad0) and all(a_changed.values())
    if r0.get("ok"):
        verdict, ok = "original labels_d1.json values pass", True
    elif r2.get("ok") and explained:
        verdict, ok = "fails with labels_d1.json only where cr2 corrected an a-side reaction; passes with labels_d1_cr2.json", True
    else:
        verdict, ok = "STOP: within-group differences cannot be made by subtraction (spec C-W1)", False
    # the B pools (within.py) against the per-reaction values used
    Yuse = Y0 if r0.get("ok") else Y2
    pool_chk = {}
    try:
        import within as WI
        src = os.environ.get("DEPW_TEST_SOURCE", "lf")
        Dsrc = DL.get_data(src)
        arr = WI.load_pool(Dsrc, "all", "train", root=_within_root(src))
        rng = np.random.default_rng(W.SEED + 1)
        q = rng.choice(len(arr["y"]), size=min(n_sample, len(arr["y"])), replace=False)
        d_rxn = Yuse.loc[arr["a_rxn_i"][q]].to_numpy(float) - Yuse.loc[arr["a_rxn_j"][q]].to_numpy(float)
        mx = np.abs(arr["y"][q] - d_rxn).max(0)
        pool_chk = dict(source=src, file=arr["file"], n=int(len(q)), max_abs_diff={t: float(mx[j]) for j, t in
                                                                              enumerate(W.TARGETS)},
                        ok=bool((mx <= TOL_LABEL).all()))
        ok = ok and pool_chk["ok"]
    except SystemExit as e:
        pool_chk = dict(ok=False, error=f"exit {e.code} (within pools missing?)")
        ok = False
    return dict(ok=bool(ok), verdict=verdict, tol=TOL_LABEL, original=r0, cr2=r2, cr2_file=info2,
                n_reactions_changed_by_cr2=len(changed), failing_groups_a_side_changed=a_changed, pool_check=pool_chk,
                labels=linfo, pair_table_vs_s_folds=agree)


# ================================================================ C-W2 leak
def cw2(source):
    import numpy as np
    import data_lf as DL
    import within as WI
    import train_w as TW
    D = DL.get_data(source)
    root = _within_root(source)
    out, ok = {}, True
    rx = lambda rows: set(D.ra[rows].tolist()) | set(D.rb[rows].tolist())                 # noqa: E731
    for k in list(range(W.NF)) + ["all"]:
        sp = D.split(k)
        keys = ("train",) if k == "all" else ("train", "inner_train")
        for key in keys:
            arr = WI.load_pool(D, k, key, root=root)
            wr = set(arr["ra"].tolist()) | set(arr["rb"].tolist())
            base = sp[key]
            base_rx = rx(base)
            base_a = set(D.P.a_rxn.to_numpy()[base].astype(int).tolist())
            rec = dict(n_within=int(len(arr["ra"])), n_rxn=len(wr),
                       outside_fit_reactions=len(wr - base_rx),
                       a_rxn_not_a_of_base=int(len((set(arr["a_rxn_i"].tolist()) | set(arr["a_rxn_j"].tolist())) - base_a)),
                       pair_rows_not_in_base=int((~np.isin(arr["pi"], base)).sum() + (~np.isin(arr["pj"], base)).sum()))
            if k != "all":
                rec["test_reactions"] = len(wr & rx(sp["test"]))
                if key == "inner_train":
                    rec["inner_val_reactions"] = len(wr & rx(sp["inner_val"]))
            viol = sum(v for kk, v in rec.items() if kk not in ("n_within", "n_rxn"))
            rec["violations"] = int(viol)
            ok &= viol == 0
            out[f"{k}/{key}"] = rec
    # negative control: a planted test reaction must STOP the run-time check (train_w.within_leak)
    sp = D.split(0)
    arr = WI.load_pool(D, 0, "inner_train", root=root)
    bad = dict(ra=np.r_[arr["ra"][:10], D.ra[sp["test"][:1]]], rb=np.r_[arr["rb"][:10], arr["rb"][:1]])
    std = TW.FT.Std(D, sp["inner_train"])
    neg = _guard(TW.within_leak, D, std, bad, (sp["inner_val"], sp["test"]))
    neg_ok = neg.get("ok") is False
    good = _guard(TW.within_leak, D, std, arr, (sp["inner_val"], sp["test"]))
    return dict(ok=bool(ok and neg_ok and good.get("ok") is True), source=source, pools=out, planted_leak_stops=neg_ok,
                runtime_check_fold0_inner=good)


# ================================================================ C-W3 r = 0 reduction
def cw3(source, n_pairs=300, n_val=200, n_epochs=5):
    """fit_w(r=0, select='mae') vs fu2 fit on CPU: identical epoch losses / validation values / best epoch / final weights
    (bitwise; fu2 fit computes in float32 by construction, train.py 155 / 226-227, so both run in that dtype)."""
    import numpy as np
    import torch
    import data_lf as DL
    import within as WI
    import train_w as TW
    FT = DL.FT
    if str(FT.DEV) != "cpu":
        return dict(ok=False, error=f"C-W3 must run on CPU (device {FT.DEV})")
    n_thr = torch.get_num_threads()
    torch.set_num_threads(1)              # multi-threaded CPU reductions are not bitwise reproducible (dev run, 4 threads)
    try:
        return _cw3(DL, WI, TW, FT, source, n_pairs, n_val, n_epochs)
    finally:
        torch.set_num_threads(n_thr)


def _cw3(DL, WI, TW, FT, source, n_pairs, n_val, n_epochs):
    import numpy as np
    import torch
    D = DL.get_data(source)
    sp = D.split(0)
    tr, va = sp["inner_train"][:n_pairs], sp["inner_val"][:n_val]
    S = W.load_s_folds()
    _, gid = W.group_table(S)
    pool, _ = WI.pool(D, tr, gid)
    vg = WI.val_groups(D, va, S, min_n=3)
    cell = TW.GRID[0]
    res = {}

    def same(a, b):
        keys = ("epoch", "loss", "l1", "pen", "val")
        ha = [{k: x.get(k) for k in keys} for x in a["history"]]
        hb = [{k: x.get(k) for k in keys} for x in b["history"]]
        sa, sb = a["model"].state_dict(), b["model"].state_dict()
        w_eq = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
        st = all(np.array_equal(getattr(a["std"], k), getattr(b["std"], k)) for k in ("mu", "sd", "gmu", "gsd", "sy"))
        return dict(history_equal=ha == hb, best_epoch=[a["best_epoch"], b["best_epoch"]],
                    best_val_equal=(a["best_val"] == b["best_val"]) or (np.isinf(a["best_val"]) and np.isinf(b["best_val"])),
                    per_target_equal=a["best_val_per_target"] == b["best_val_per_target"], weights_equal=bool(w_eq),
                    std_equal=bool(st), n_epochs=len(ha), max_abs_loss_diff=float(max(abs(x["loss"] - y["loss"])
                                                                                      for x, y in zip(ha, hb))))
    t0 = time.time()
    a = FT.fit(D, "TF", cell, "A+G", tr, va, seed=F_SEED(), n_epochs=n_epochs)
    a0 = FT.fit(D, "TF", cell, "A+G", tr, va, seed=F_SEED(), n_epochs=n_epochs)
    control = same(a, a0)                                        # fu2 fit vs itself: CPU determinism of this process
    b = TW.fit_w(D, "TF", cell, "A+G", tr, va, seed=F_SEED(), n_epochs=n_epochs, r=0, within=pool, select="mae",
                 vgroups=vg)
    res["with_val"] = same(a, b)
    a2 = FT.fit(D, "TF", cell, "A+G", tr, None, seed=F_SEED() + 1, n_epochs=n_epochs)
    b2 = TW.fit_w(D, "TF", cell, "A+G", tr, None, seed=F_SEED() + 1, n_epochs=n_epochs, r=0, within=pool)
    res["final_mode"] = same(a2, b2)
    b3 = TW.fit_w(D, "TF", cell, "A+G", tr, va, seed=F_SEED(), n_epochs=n_epochs, r=1, within=pool, select="mae",
                  vgroups=vg)
    neg = same(a, b3)
    b4 = TW.fit_w(D, "TF", cell, "A+G", tr, va, seed=F_SEED(), n_epochs=n_epochs, r=0, within=pool, select="nl",
                  vgroups=vg)
    nl_traj = [x["loss"] for x in b4["history"]] == [x["loss"] for x in a["history"]][:len(b4["history"])]
    ok = all(v["history_equal"] and v["best_val_equal"] and v["per_target_equal"] and v["weights_equal"]
             and v["std_equal"] and v["best_epoch"][0] == v["best_epoch"][1] for v in res.values())
    ok = ok and not neg["history_equal"] and nl_traj
    ctl_ok = control["history_equal"] and control["weights_equal"]
    return dict(ok=bool(ok), source=source, device=str(FT.DEV), dtype="float32 (fu2 fit fixed dtype)", cell=cell,
                determinism_control_fit_vs_fit=dict(control, ok=bool(ctl_ok)),
                deterministic_settings=dict(threads=torch.get_num_threads(),
                                            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                            MKL_CBWR=os.environ.get("MKL_CBWR"), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")),
                n_pairs=int(len(tr)), n_val=int(len(va)), n_epochs=n_epochs, n_vnl_groups=len(vg), checks=res,
                negative_control_r1_differs=not neg["history_equal"], select_nl_same_training_trajectory=bool(nl_traj),
                seconds=time.time() - t0, threads=__import__("torch").get_num_threads(),
                note="spec says 'CPU float64'; fu2 fit hard-codes float32 tensors, so both run on CPU in float32 and "
                     "must agree bit for bit")


def F_SEED():
    import data_lf as DL
    return DL.F.SEED


# ================================================================ C-W4 standardisation vs the lf unit .pt
def cw4(source):
    import numpy as np
    import torch
    import data_lf as DL
    import within as WI
    import train_w as TW
    import train_tf as TT
    D = DL.get_data(source)
    if source == "lf":
        udir, limit, rung = C.UNITS, None, "lf"
    else:
        udir, limit, rung = C.DEP / "dev_tf" / "kunits", 150, source
    files = [udir / f"D__{rung}__S__TF__f{k}__s{j}.pt" for k in range(W.NF) for j in range(5)]
    miss = [str(p) for p in files if not p.is_file()]
    if miss:
        W.not_ready(f"C-W4: {len(miss)} unit files missing ({miss[:2]}; DEPLOY_MODEL K)")
    out, ok = {}, True
    for k in range(W.NF):
        sp = D.split(k)
        tr = sp["train"] if not limit else TT._limit(sp, limit)["train"]
        pool = WI.load_pool(D, k, "train", root=_within_root(source)) if not limit else \
            WI.pool(D, tr, W.group_table(W.load_s_folds())[1])[0]
        res = TW.fit_w(D, "TF", TW.GRID[0], "A+G", tr, None, seed=F_SEED(), n_epochs=0, r=4, within=pool)
        mine = TT.std_dict(res["std"], res["sy"])
        rec = {}
        for j in range(5):
            obj = torch.load(udir / f"D__{rung}__S__TF__f{k}__s{j}.pt", map_location="cpu")
            d = {x: float(np.max(np.abs(np.asarray(mine[x], float) - np.asarray(obj["std"][x], float))))
                 for x in ("sy", "mu", "sd", "gmu", "gsd")}
            rec[f"s{j}"] = d
        mx = max(v for r in rec.values() for v in r.values())
        out[str(k)] = dict(max_abs_diff=mx, per_seed=rec, n_train=int(len(tr)), n_within_pool=int(len(pool["w"])))
        ok &= mx <= TOL_STD
    return dict(ok=bool(ok), source=source, units=str(udir), limit=limit, tol=TOL_STD, folds=out)


# ================================================================ C-W5 V_nl
def cw5():
    import numpy as np
    import pandas as pd
    import within as WI
    rng = np.random.default_rng(5)
    rec = {}
    # (a) exact straight-line errors through H (e = b sigma) -> r = 0, V_nl = 0
    sig = rng.uniform(-0.8, 0.8, size=17)
    E = np.outer(sig, rng.normal(size=10))
    g = [dict(idx=np.arange(17), sigma=sig)]
    v = W.vnl(E, g, np.ones(10))
    rec["a_line_zero"] = dict(vnl=v, ok=abs(v) <= 1e-12)
    # (b) V kink e = k |sigma| on a symmetric design: r = k SD(|sigma| incl. H) (closed form)
    cases = []
    for s, k, want in (((-1, 1), 3.0, 3.0 * np.sqrt(2.0) / 3.0), ((-2, -1, 1, 2), 2.5, 2.5 * np.sqrt(14.0) / 5.0),
                       ((-0.5, -0.25, 0.25, 0.5, -0.75, 0.75), 1.7, None)):
        s = np.asarray(s, float)
        e = k * np.abs(s)
        x, y = W.with_h(s, e)
        got = W.line_rms(x, y)
        ax = np.abs(x)
        closed = k * float(np.sqrt((ax ** 2).mean() - ax.mean() ** 2))
        cases.append(dict(sigma=s.tolist(), k=k, got=got, closed_form=closed, hand=want,
                          ok=abs(got - closed) <= TOL_VNL and (want is None or abs(got - want) <= TOL_VNL)))
    rec["b_v_kink"] = dict(cases=cases, ok=all(c["ok"] for c in cases))
    # (c) hinge on a general design: residual = component of the hinge orthogonal to span{1, x} (projection by hand)
    s = np.sort(rng.uniform(-0.6, 0.9, size=14))
    h = np.maximum(0.0, s - 0.2) * 4.0
    x, y = W.with_h(s, h + 0.7 * s)
    A = np.c_[np.ones(len(x)), x]
    Q, _ = np.linalg.qr(A)
    res = y - Q @ (Q.T @ y)
    want = float(np.sqrt((res ** 2).mean()))
    got = W.line_rms(x, y)
    rec["c_hinge_projection"] = dict(got=got, want=want, ok=abs(got - want) <= TOL_VNL)
    # (d) V_nl = mean_t [mean_g r_gt / sy_t] over PRIMARY only, c_ghost ignored
    n1, n2 = 8, 11
    s1, s2 = rng.uniform(-0.5, 0.5, n1), rng.uniform(-0.5, 0.5, n2)
    K1, K2 = rng.uniform(0.5, 2, 10), rng.uniform(0.5, 2, 10)
    E = np.zeros((n1 + n2, 10))
    E[:n1] = np.abs(s1)[:, None] * K1[None, :] + np.outer(s1, rng.normal(size=10))
    E[n1:] = np.abs(s2)[:, None] * K2[None, :]
    E[:, W.TARGETS.index("c_ghost")] = 1e6 * rng.normal(size=n1 + n2)
    sy = rng.uniform(0.3, 3.0, 10)
    groups = [dict(idx=np.arange(n1), sigma=s1), dict(idx=np.arange(n1, n1 + n2), sigma=s2)]
    man = []
    for j in W.I_PRIM:
        r1 = W.line_rms(*W.with_h(s1, E[:n1, j]))
        r2 = W.line_rms(*W.with_h(s2, E[n1:, j]))
        man.append((r1 + r2) / 2 / sy[j])
    want, got = float(np.mean(man)), W.vnl(E, groups, sy)
    rec["d_mean_structure"] = dict(got=got, want=want, ok=abs(got - want) <= TOL_VNL)
    # (e) val_groups: pos p / m only, >= MIN_GROUP_VAL pairs with sigma, NaN sigma not counted
    S = pd.DataFrame(dict(pair_id=[f"X{i:03d}" for i in range(40)], ref_kind="reference", a_rxn=np.arange(40) + 100,
                          b_rxn=[1] * 10 + [2] * 10 + [3] * 10 + [4] * 10, core="c", fam=["f1"] * 10 + ["f2"] * 10 +
                          ["f3"] * 10 + ["f4"] * 10, pos=["p"] * 10 + ["m"] * 10 + ["o"] * 10 + ["p"] * 10, Xb="X",
                          sigma=np.r_[np.linspace(-.5, .5, 10), np.linspace(-.4, .6, 10), np.zeros(10),
                                      np.r_[np.linspace(-.5, .5, 5), np.full(5, np.nan)]],
                          in_T_main=True, in_P_main=True, fold_T1=0)).set_index("pair_id")

    class _D:
        pass
    Dm = _D()
    Dm.P = S
    vg = WI.val_groups(Dm, np.arange(40), S)
    keys = sorted(gg["key"] for gg in vg)
    rec["e_val_groups"] = dict(keys=[list(k) for k in keys], ok=keys == [(1, "f1", "p"), (2, "f2", "m")])
    return dict(ok=all(v["ok"] for v in rec.values()), tol=TOL_VNL, checks=rec)


# ================================================================ C-W6 sampling
def cw6(source, epochs=SAMPLE_EPOCHS):
    import numpy as np
    import data_lf as DL
    import within as WI
    D = DL.get_data(source)
    arr = WI.load_pool(D, 0, "inner_train", root=_within_root(source))
    n_base = int(len(D.split(0)["inner_train"]))
    gid, w = arr["gid"], arr["w"]
    ug, ginv = np.unique(gid, return_inverse=True)
    n_g = np.array([int(arr["n_g"][gid == x][0]) for x in ug])
    out, ok = {}, True
    for r in (1, 4):
        n_w = r * n_base
        pi = W.inclusion(w, n_w)
        exp_g = np.bincount(ginv, pi)
        cap = np.bincount(ginv, pi >= 1.0) > 0
        rng = np.random.default_rng(W.SEED + r)
        cnt = np.zeros(len(ug))
        sizes, dup = [], 0
        t0 = time.time()
        for _ in range(epochs):
            s = W.draw(rng, pi)
            sizes.append(len(s))
            dup += int(len(s) != len(np.unique(s)))
            cnt += np.bincount(ginv[s], minlength=len(ug))
        mean = cnt / epochs
        rel = np.abs(mean / exp_g - 1.0)
        c_hat = exp_g[~cap] / n_g[~cap]
        prop = np.abs(mean[~cap] / (c_hat.mean() * n_g[~cap]) - 1.0) if (~cap).any() else np.zeros(0)
        exact_cap = bool(np.all(np.abs(mean[cap] - n_g[cap] * (n_g[cap] - 1) / 2) <= 1e-9)) if cap.any() else True
        rec = dict(n_w=n_w, pool=int(len(w)), fraction=n_w / len(w), n_groups=int(len(ug)), n_capped_groups=int(cap.sum()),
                   capped_group_sizes=sorted(set(n_g[cap].tolist())), epochs=epochs, sum_pi=float(pi.sum()),
                   draw_size=dict(min=int(min(sizes)), max=int(max(sizes))), duplicates=dup,
                   max_rel_dev_vs_expected=float(rel.max()), max_rel_dev_vs_ng_proportional_uncapped=float(prop.max())
                   if len(prop) else 0.0, expected_proportional_spread=float(c_hat.max() / c_hat.min() - 1)
                   if len(c_hat) else 0.0, capped_groups_drawn_whole=exact_cap,
                   capped_share_of_proportional=float(exp_g[cap].sum() / (n_g[cap].sum() / n_g.sum() * n_w))
                   if cap.any() else None, seconds=time.time() - t0)
        rec["ok"] = bool(rel.max() <= SAMPLE_TOL and (prop.max() if len(prop) else 0) <= SAMPLE_TOL and dup == 0
                         and exact_cap and abs(rec["expected_proportional_spread"]) <= 1e-9
                         and max(sizes) - min(sizes) <= 1 and abs(np.mean(sizes) - min(n_w, len(w))) <= 1)
        ok &= rec["ok"]
        # descriptive: the naive numpy weighted draw without replacement (not used) on fewer epochs
        rng2 = np.random.default_rng(W.SEED + 100 + r)
        cnt2 = np.zeros(len(ug))
        ne = 300
        for _ in range(ne):
            s = rng2.choice(len(w), size=min(n_w, len(w)), replace=False, p=w / w.sum())
            cnt2 += np.bincount(ginv[s], minlength=len(ug))
        prop2 = np.abs((cnt2 / ne) / (n_w * n_g / n_g.sum()) - 1.0)
        rec["naive_numpy_choice_max_rel_dev_vs_ng_proportional"] = dict(epochs=ne, all=float(prop2.max()),
                                                                        uncapped=float(prop2[~cap].max()) if (~cap).any() else None)
        out[f"r{r}"] = rec
    return dict(ok=bool(ok), source=source, tol=SAMPLE_TOL, n_base=n_base,
                design="pi_i = min(1, c w_i), w = 2/(n_g-1), systematic sampling on a random order (w_common.inclusion/draw)",
                runs=out)


# ================================================================ C-W7 graph readout
def cw7():
    import numpy as np
    import pandas as pd
    import within_predict as WP
    import eval_w as EW
    rng = np.random.default_rng(7)
    sizes = [2, 3, 7, 20, 21]
    u, v, grp, refs, m, gadd, ganti, dir_add, ref_ans = [], [], [], [], [], [], [], [], []
    node0 = 0
    node_group = []
    for gi, n in enumerate(sizes):
        f = rng.normal(size=n)
        A = rng.normal(size=(n, n))
        M = A - A.T
        ref = n - 1
        for i in range(n):
            for j in range(i + 1, n):
                u.append(node0 + i)
                v.append(node0 + j)
                gadd.append(f[i] - f[j])
                ganti.append(M[i, j])
        node_group += [gi] * n
        refs.append(node0 + ref)
        m.append(n)
        dir_add.append(f - f[ref])
        ref_ans.append(W.graph_direct_sum(M, ref))
        node0 += n
    u, v = np.array(u), np.array(v)
    ga = W.graph_values(u, v, np.array(gadd), np.array(node_group), np.array(m), np.array(refs))
    gb = W.graph_values(u, v, np.array(ganti), np.array(node_group), np.array(m), np.array(refs))
    da = float(np.abs(ga - np.concatenate(dir_add)).max())
    db = float(np.abs(gb - np.concatenate(ref_ans)).max())
    # within_predict pair plan + eval_w assembly on a synthetic group table
    T = pd.DataFrame(dict(pair_id=[f"P{i}" for i in range(9)], gid=[5] * 4 + [9] * 5, a_rxn=[11, 12, 13, 14, 21, 22, 23, 24, 25],
                          b_rxn=[10] * 4 + [20] * 5))
    plan = WP.pair_plan(T)
    n_ok = plan.groupby("gid").size().to_dict() == {5: 10, 9: 15} and bool((plan.groupby("gid").b_rxn.last()
                                                                           == pd.Series({5: 10, 9: 20})).all())
    fr = {10: 0.0, 11: 0.3, 12: -0.1, 13: 1.0, 14: 0.2, 20: 0.5, 21: 0.1, 22: 0.9, 23: -0.4, 24: 0.0, 25: 2.0}
    rows = []
    for s in (0, 1):
        for t in W.TARGETS:
            for _, q in plan.iterrows():
                rows.append(dict(gid=q.gid, a_rxn=q.a_rxn, b_rxn=q.b_rxn, seed=s, target=t,
                                 g=(fr[q.a_rxn] - fr[q.b_rxn]) * (1 + s)))
    G = EW.graph_predictions(pd.DataFrame(rows), T, fold=0)
    want = {(pid, t): (fr[a] - fr[b]) * 1.5 for pid, a, b in zip(T.pair_id, T.a_rxn, T.b_rxn) for t in W.TARGETS}
    dg = max(abs(G.set_index(["pair_id", "target"]).yhat.loc[k] - vv) for k, vv in want.items())
    ok = da <= TOL_GRAPH and db <= TOL_GRAPH and n_ok and dg <= TOL_GRAPH
    return dict(ok=bool(ok), additive_graph_minus_direct_max=da, antisym_formula_minus_direct_sum_max=db,
                pair_plan_counts_ok=bool(n_ok), eval_assembly_max=float(dg), tol=TOL_GRAPH, sizes=sizes)


# ================================================================ parts
def part_cpu(source):
    os.environ["DEPW_TEST_SOURCE"] = source
    checks = {"C-W1": _guard(cw1), "C-W2": _guard(cw2, source), "C-W3": _guard(cw3, source), "C-W5": _guard(cw5),
              "C-W6": _guard(cw6, source), "C-W7": _guard(cw7)}
    out = None if source == "lf" else W.DEV / f"C_part_cpu_w_{source}.json"
    return _write_part("cpu", checks, out, extra=dict(source=source))


def part_cw4(source):
    rec = _guard(cw4, source)
    if rec.get("error") == "SystemExit 2":
        print("NOT READY: C-W4 needs the DEPLOY_MODEL lf units", file=sys.stderr)
        return 2
    out = None if source == "lf" else W.DEV / f"C_part_cw4_w_{source}.json"
    return _write_part("cw4", {"C-W4": rec}, out, extra=dict(source=source))


def part_gpu():
    import train_w as TW
    rc = _guard(TW.cmd_smoke)
    p = W.RESW / "C9_smoke_w.json"
    c9 = W.read_json(p) if p.is_file() else {}
    fresh = (c9.get("code_sha256") or {}).get("w/train_w.py") == TW.CODE_SHA["w/train_w.py"]
    c9rec = dict(c9, ok=bool(rc == 0 and c9.get("ok") and fresh), exit=rc if isinstance(rc, int) else str(rc))
    g8 = dict(ok=bool(c9.get("gates_ok") and fresh), rule="|g(a,b)+g(b,a)| <= 1e-4 and |g(a,a)| <= 1e-4 kcal/mol (float64) "
              "on 200 random test pairs and 200 random within pairs of every tuning / final fit (train_w stops otherwise)",
              tune=c9.get("gate_CW8_tune"), final=c9.get("gate_CW8_final"))
    return _write_part("gpu", {"C-W8": g8, "C-W9": c9rec}, extra=dict(trainer_code_sha256=code_sha(TRAINER_FILES)))


def merge():
    parts, bad = {}, []
    for part, fn in PART_FILES.items():
        p = W.RESW / fn
        if not p.is_file():
            bad.append(f"{fn} missing")
            continue
        r = W.read_json(p)
        now = code_sha(PART_CODE[part])
        stale = [f for f, s in (r.get("code_sha256") or {}).items() if s != now.get(f)]
        parts[part] = dict(file=fn, sha256=W.sha256(p), ok=bool(r.get("ok")), stale_code=stale,
                           checks={k: _ok(v) for k, v in (r.get("checks") or {}).items()})
        if part in ("cpu", "cw4") and r.get("source") != "lf":
            bad.append(f"{fn} did not run on the lf data")
        if stale:
            bad.append(f"{fn} ran on other code: {stale}")
        if not r.get("ok"):
            bad.append(f"{fn} not ok")
    checks = {c: bool(parts.get(p, {}).get("checks", {}).get(c)) for p, cs in PART_CHECKS.items() for c in cs}
    trainer = code_sha(TRAINER_FILES)
    g = W.RESW / PART_FILES["gpu"]
    if g.is_file() and W.read_json(g).get("trainer_code_sha256") != trainer:
        bad.append("C-W9 ran on other trainer code than the current files")
    rec = dict(all_pass=bool(not bad and all(checks.values())), checks=dict(sorted(checks.items())), parts=parts,
               problems=bad, trainer_code_sha256=trainer, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    W.write_json(W.RESW / "C_tests_w.json", rec)
    print(json.dumps({k: rec[k] for k in ("all_pass", "checks", "problems")}, indent=1))
    return 0 if rec["all_pass"] else 3


def main(argv):
    a = argv[1:]
    src = "lf"
    if "--source" in a:
        i = a.index("--source")
        src = a[i + 1]
        a = a[:i] + a[i + 2:]
    if a == ["cpu"]:
        return part_cpu(src)
    if a == ["cw4"]:
        return part_cw4(src)
    if a == ["gpu"]:
        return part_gpu()
    if a == ["merge"]:
        return merge()
    if a == ["patch"]:
        import patch_test as PTT
        return PTT.main()
    if a == ["i2"]:
        import i2_checks as I2
        return I2.main()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
