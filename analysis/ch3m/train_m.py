#!/usr/bin/env python3
r"""train_m.py — CH3M TF · GNN units (docs/specs/CH3M_MODELS.md 3, 5, 7, 9). GPU work in env_d1fu2 (torch 2.2.1+cu118)
through jobs/gpu_m.slurm (RTX 3090, gpu1); collect on CPU (jobs/cpu_m.slurm CH3MENV=d1fu2).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python train_m.py list                          unit names of K (TF 10, GNN 10, GNN 3)
  python train_m.py smoke                         C-8 (GNN): 10 and 3 modes, fold 0, smallest + largest cell, 2 epochs,
                                                  1 seed, C-4 / C-5 on every fit, C-T on the 3-mode .pt
                                                  -> results/C8_smoke_nn.json (scratch $CH3M/smoke)
  python train_m.py cr <TF|GNN>                   C-RT / C-RG: 10 modes on DeployData('g2') (the LADDER g2 inputs), fold 0,
                                                  full rules -> $CH3M/cr/<model>/D__g2__S__<model>__f0.*, compared with
                                                  the LADDER unit L__g2__S__<model>__f0 (L_report.json) and the fold-0
                                                  rows of L_predictions.parquet -> results/CR_<tf|gnn>.json
                                                  (`cr GNN rg2` = the repeat C-RG2 in $CH3M/cr/rg2 -> results/CR_gnn2.json)
  python train_m.py run <TF|GNN> <10|3> <i|@I> <n>  units [i::n] -> $CH3M/units/D__<lf10|lf3>__S__<model>__f<k>.*
                                                  Gates: PREREG (prereg_m.check), results/C_tests.json all_pass on this
                                                  trainer code. Idempotent (fingerprint incl. GPU name; mismatch = STOP)
  python train_m.py collect                       TF10 / GNN10 / GNN3 units -> $CH3M/collect/nn.parquet + nn_report.json
                                                  (xgb_m.py collect merges them into results/KM_predictions.parquet)
  python train_m.py final <TF10|GNN10|GNN3>        spec 9 (DEPLOY 13 rule) -> $CH3M/model/D__<rung>__S__<model>__all__s{0..4}.pt,
                                                  MODEL_CARD.json (+ results/MODEL_CARD_ch3m.json), results/WM_report.json

Rules = DEPLOY 11-2 / ch3 train_ch3.run_unit with the model as an argument: fu2 train fit / Std / gate_m1 / predict
imported unchanged; GRID = fu2 models.GRID[model] (TF 2/3 x 64/128, GNN 4/6 x 64/128); Mo.build(model, cell, 29, n_out,
182, 'A+G'); tuning on inner_train, cell by inner_val (mean standardised MAE: 9 primary targets in 10 mode, 4 targets in
3 mode), final = outer train pairs, tuned best epoch, seeds 20261019 + 0..4, mean; gate C-4 and leak check C-5 on every
fit. Target modes through analysis/ch3/data_ch3.use_targets (10 = deploy constants, 3 = ch3 constants).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402
import data_ch3 as DC3  # noqa: E402            (analysis/ch3, on sys.path through ch3m_common)

DL, FT, Mo, F = DC3.DL, DC3.FT, DC3.Mo, DC3.F
import train_tf as TT  # noqa: E402               (deploy helpers: leak_check, kcal, std_dict, std_sha, norm, choose_final)
import train_ch3 as TC  # noqa: E402              (ch3 helpers: ct_checks, val_metric_check)
import torch  # noqa: E402

CODE_VERSION = "ch3m-nn-1"
VARIANT, ARM = TT.VARIANT, TT.ARM
PRED_COLS = list(TT.PRED_COLS)
NF = DL.NF
T10, T3 = list(C.TARGETS10), list(C.TARGETS3)
L_REPORT = C.LAD_RES / "L_report.json"
L_PRED = C.LAD_RES / "L_predictions.parquet"
CR_TOL = dict(val_kcal=1e-6, yhat=1e-4)
W_CHECK_N = 1000
CODE_FILES = ("train_m.py", "ch3m_common.py")
DEP_FILES = ("ch3/data_ch3.py", "ch3/ch3_common.py", "ch3/train_ch3.py", "deploy/train_tf.py", "deploy/data_lf.py",
             "deploy/dep_common.py")
CODE_SHA = {**{f: C.sha256(HERE / f) for f in CODE_FILES},
            **{f: C.sha256(C.REPO / "analysis" / f) for f in DEP_FILES},
            **{(k if k == "d1pair_common.py" else f"fu2/{k}"): v for k, v in FT.CODE_SHA.items()}}
TRAINER_FILES = CODE_FILES
K_UNITS = [("TF", 10), ("GNN", 10), ("GNN", 3)]
_GATE: dict = {}
norm = TT.norm


def grid(model):
    return list(Mo.GRID[model])


def targets_of(mode):
    return T10 if int(mode) == 10 else T3


def write_json(path, obj):
    return C.write_json(path, TT._jsonable(obj))


def unit_list():
    return [dict(rung=C.RUNGS[m], arm=ARM, model=mo, fold=k, mode=m) for mo, m in K_UNITS for k in range(NF)]


def uname(u):
    return C.uname(u["rung"], u["model"], u["fold"])


def final_name(rung, model, j):
    return f"D__{rung}__{ARM}__{model}__all__s{j}"


def save_torch(path, obj):
    path = C.assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def data_for(mode, source="lf"):
    """10 -> DeployData('lf') (y10), 3 -> Ch3Data (y10 @ M); 'g2' -> DeployData('g2') (10 only, C-RT / C-RG)."""
    if source == "g2":
        if int(mode) != 10:
            C.stop("g2 data only in 10 mode")
        if "g2" not in _DATA:
            _DATA["g2"] = DL.DeployData("g2")
        D = _DATA["g2"]
    else:
        D = DC3.get_data("lf" if int(mode) == 10 else "ch3")
    DC3.use_targets(targets_of(mode))
    return D


_DATA: dict = {}


def mode_check(targets):
    want = DC3.MODE3 if list(targets) == T3 else DC3.MODE10
    if DC3.current_mode() != want:
        C.stop(f"fu2 target constants {DC3.current_mode()} != {want} for {targets}")


def _outdim(r, D, rows):
    rows = np.asarray(rows)[:2]
    return int(FT.predict(r["model"], r["T"], D.ra[rows], D.rb[rows]).shape[1])


def pt_payload(D, r, cell, u, targets, model):
    p = dict(state=r["model"].state_dict(), cell=cell, unit=u, std=TT.std_dict(r["std"], r["sy"]), rung=u["rung"],
             arm=ARM, variant=VARIANT, model=model, d_in=int(D.d_in), d_glob=int(D.d_glob), TARGETS=list(targets),
             layout=D.layout())
    if list(targets) == T3:
        p["M"], p["M_rows"], p["M_cols"] = C.M_ROWS, T10, T3
    return p


def _check_root(root, smoke):
    r = C.assert_write_ok(Path(root) / "probe").parent
    for prod in (C.UNITS, C.MODEL_DIR):
        p = Path(prod).resolve()
        if r == p or p in r.parents:
            if smoke:
                C.stop(f"smoke / test runs never write {p}")
            if not _GATE.get("prereg"):
                C.stop(f"writing {p} needs the PREREG gate (train_m.py run / final)")
    return r


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


def fingerprint(D, u, sp, targets, model, smoke=False, limit=None):
    return norm(dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=u["rung"], arm=ARM, model=model, variant=VARIANT,
        targets=list(targets), conversion=C.M_ROWS if list(targets) == T3 else None, data_rung=D.rung,
        usable_rule=D.usable_rule, inputs_sha256=D.sha, pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"],
        prereg=None if smoke else _GATE.get("prereg"), grid=grid(model), limit=limit, smoke=bool(smoke),
        fu2_mode=DC3.current_mode(),
        rules=dict(lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                   patience=F.PATIENCE, seeds=F.FINAL_SEEDS, tune_seed=F.SEED, antisym_tol=F.ANTISYM_TOL,
                   antisym_n=F.ANTISYM_N, val="first inner fold (FU2 split files)"),
        **FT.versions()))


def _limit(sp, limit):
    if not limit:
        return sp
    sp = dict(sp)
    for k in DL.FOLD_KEYS:
        sp[k] = sp[k][:int(limit)]
    return sp


# ================================================================ one CV unit (= train_ch3.run_unit, model as argument)
def run_unit(D, u, targets, model, root=None, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50,
             smoke=False, limit=None):
    targets = list(targets)
    root = C.UNITS if root is None else root
    root = _check_root(root, smoke or bool(limit))
    root.mkdir(parents=True, exist_ok=True)
    mode_check(targets)
    if D.y.shape[1] != len(targets):
        C.stop(f"data y has {D.y.shape[1]} columns, targets {targets}")
    if u["model"] != model or u["arm"] != ARM:
        C.stop(f"unit {u} does not match model {model}")
    GRID = grid(model)
    name = uname(u)
    sp = _limit(D.split(u["fold"]), limit)
    fp = fingerprint(D, u, sp, targets, model, smoke, limit)
    js = root / f"{name}.json"
    if js.is_file():
        old = C.read_json(js)
        ofp = old.get("fingerprint") or {}
        if ofp != fp:
            C.stop(f"cache {js} has a different fingerprint: keys "
                   f"{sorted(k for k in set(fp) | set(ofp) if ofp.get(k) != fp.get(k))}")
        print(f"  {name}: cached", flush=True)
        return old
    t_unit = time.time()
    idx = list(range(len(GRID))) if cells is None else [int(c) for c in cells]
    tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
    if min(len(tr), len(te), len(itr), len(iva)) == 0:
        C.stop(f"{name}: empty split list (train {len(tr)}, test {len(te)}, inner {len(itr)} / {len(iva)})")
    tune = []
    for ci, gi in enumerate(idx):
        cell = GRID[gi]
        r = FT.fit(D, model, cell, VARIANT, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log)
        lk = TT.leak_check(D, r["std"], itr, (iva, te), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=F.SEED + ci)
        vk = np.abs(TT.kcal(r, D, iva) - D.y[iva]).mean(0)
        od = _outdim(r, D, iva)
        if od != len(targets) or len(r["best_val_per_target"]) != len(targets):
            C.stop(f"C-T: model output {od} / validation width {len(r['best_val_per_target'])} != {len(targets)}")
        tune.append(dict(cell=cell, grid_index=gi, best_epoch=r["best_epoch"], val=r["best_val"],
                         val_per_target=r["best_val_per_target"], val_kcal_mae=dict(zip(targets, vk.tolist())),
                         epochs_run=len(r["history"]), seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
                         out_dim=od,
                         history=[dict(epoch=h["epoch"], loss=h["loss"], val=h.get("val")) for h in r["history"]]))
        print(f"  {name} cell {cell}: val {r['best_val']:.4f} at epoch {r['best_epoch']} ({r['seconds']:.0f} s) "
              f"gate {g}", flush=True)
        if not g["ok"]:
            C.stop(f"gate C-4 failed ({name}, tuning cell {cell}): {g}")
        del r
    best = min(range(len(tune)), key=lambda i: (tune[i]["val"], i))
    cell, ep = tune[best]["cell"], tune[best]["best_epoch"]
    if not (isinstance(ep, int) and ep >= 1 and np.isfinite(tune[best]["val"])):
        C.stop(f"{name}: chosen cell {cell} has best_epoch {ep} / validation {tune[best]['val']}")
    preds, finals = [], []
    for s in seeds:
        r = FT.fit(D, model, cell, VARIANT, tr, None, seed=s, n_epochs=ep, log=log)
        lk = TT.leak_check(D, r["std"], tr, (te,), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=s)
        if not g["ok"]:
            C.stop(f"gate C-4 failed ({name}, seed {s}): {g}")
        if _outdim(r, D, te) != len(targets):
            C.stop(f"C-T: final model output != {len(targets)}")
        preds.append(TT.kcal(r, D, te))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
                           std_sha256=TT.std_sha(TT.std_dict(r["std"], r["sy"])), history_tail=r["history"][-3:]))
        save_torch(root / f"{name}__s{s - F.SEED}.pt", pt_payload(D, r, cell, u, targets, model))
        del r
    Y = np.stack(preds)
    pid = D.P.index.to_numpy()[te].astype(str)
    rows = [pd.DataFrame(dict(pair_id=pid, ref_kind=D.P.ref_kind.to_numpy()[te], pos=D.P.pos.to_numpy()[te],
                              fold=int(u["fold"]), rung=u["rung"], arm=ARM, model=model, target=t,
                              y=D.y[te, j], yhat=Y[:, :, j].mean(0), base=0.0, beta=0.0,
                              yhat_seed_sd=Y[:, :, j].std(0))) for j, t in enumerate(targets)]
    pred = pd.concat(rows, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    C.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, model=model, targets=targets, fingerprint=fp, smoke=bool(smoke), limit=limit,
               variant=VARIANT, grid=grid(model), tune=tune,
               chosen=dict(cell=cell, best_epoch=ep, index=best, grid_index=tune[best]["grid_index"]), final=finals,
               val_kcal_mae=tune[best]["val_kcal_mae"], val_pairs_sha256=sp["val_sha"], test_pairs_sha256=sp["test_sha"],
               n_val=len(iva), n_train=len(tr), n_test=len(te), n_inner_train=len(itr), n_inner_val=len(iva),
               dropped=sp["dropped"], parquet_sha256=C.sha256(pq), inputs=D.inputs, usable_rule=D.usable_rule,
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    write_json(js, rec)
    return rec


# ================================================================ gates
def gate_prereg():
    import prereg_m as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    p = C.RES / "C_tests.json"
    c = C.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K / W)")
    rec = (c.get("trainer_code_sha256") or {}).get("train_m") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun the C tests")
    return dict(c_tests_sha256=C.sha256(p))


# ================================================================ commands
def cmd_smoke():
    """C-8 (GNN): 10 and 3 modes, fold 0, cells smallest + largest, 2 epochs, 1 seed; C-T on the 3-mode .pt."""
    FT.require_gpu()
    root = C.SMOKE / "nn"
    _check_root(root, True)
    if root.is_dir():
        for p in root.glob("D__*"):
            C.assert_write_ok(p)
            p.unlink()
    out = dict(check="C-8 (GNN) + C-T", versions=FT.versions(), runs={}, code_sha256=CODE_SHA)
    ok = True
    for mode in (10, 3):
        D = data_for(mode)
        tg = targets_of(mode)
        u = dict(rung=C.RUNGS[mode], arm=ARM, model="GNN", fold=0, mode=mode)
        t0 = time.time()
        rec = run_unit(D, u, tg, "GNN", root=root, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, len(grid("GNN")) - 1],
                       log=1, smoke=True)
        r = dict(seconds=time.time() - t0, gates_C4=TT.gates_ok(rec),
                 leak_C5=all(t["leak"]["ok"] for t in rec["tune"] + rec["final"]),
                 gate_C4_max=max(max(t["gate_M1"]["anti_max"], t["gate_M1"]["self_max"]) for t in rec["tune"] + rec["final"]),
                 cells=[t["cell"] for t in rec["tune"]], out_dims=[t["out_dim"] for t in rec["tune"]])
        if mode == 3:
            pt = root / f"{uname(u)}__s0.pt"
            ct = TC.ct_checks(D, pt)                         # fu2 constants, output 4, loss Σ3 (ch3 C-T)
            obj = torch.load(pt, map_location="cpu")
            ct["checks"]["pt_model_GNN"] = obj.get("model") == "GNN"
            mm = Mo.build("GNN", obj["cell"], int(obj["d_in"]), 4, int(obj["d_glob"]), VARIANT, F.DROPOUT)
            try:
                mm.load_state_dict(obj["state"], strict=True)
                ct["checks"]["pt_state_GNN_4_outputs"] = True
            except RuntimeError as e:
                ct["checks"]["pt_state_GNN_4_outputs"] = False
                ct["detail"]["pt_gnn_load_error"] = str(e)[:300]
            # train_ch3.ct_checks loads the .pt into a TF model (its MODEL constant): replaced by the GNN load above
            ct["detail"]["pt_state_4_outputs_TF_load_replaced"] = ct["checks"].pop("pt_state_4_outputs", None)
            ct["checks"]["grid_is_Mo_GRID_GNN"] = rec["grid"] == norm(Mo.GRID["GNN"])
            vm = TC.val_metric_check(rec, 4)
            ct["checks"]["val_metric_mean4"] = vm["ok"]
            ct["ok"] = bool(all(ct["checks"].values()))
            r["C_T"] = ct
            ok &= ct["ok"]
        ok &= r["gates_C4"] and r["leak_C5"]
        out["runs"][uname(u)] = r
        print(f"[smoke] {uname(u)}: {json.dumps(r, default=str)[:400]}", flush=True)
    out["ok"] = bool(ok)
    write_json(C.RES / "C8_smoke_nn.json", out)
    return 0 if ok else 3


def lad_unit(model):
    rep = C.read_json(L_REPORT)
    key = f"L__g2__S__{model}__f0"
    u = (rep.get("units") or {}).get(key)
    if u is None:
        C.stop(f"{L_REPORT}: no unit {key}")
    return key, u


def cmd_cr(model, tag=None):
    """C-RT / C-RG (/ C-RG2): 10 mode on DeployData('g2'), fold 0, full rules, vs the LADDER unit and predictions."""
    FT.require_gpu()
    if model not in ("TF", "GNN"):
        C.stop(f"cr {model}?")
    key, ref = lad_unit(model)
    D = data_for(10, "g2")
    u = dict(rung="g2", arm=ARM, model=model, fold=0, mode=10)
    root = C.CR / (tag or model)
    rec = run_unit(D, u, T10, model, root=root)
    mine = pd.read_parquet(root / f"{uname(u)}.parquet")
    L = pd.read_parquet(L_PRED)
    theirs = L[(L.rung == "g2") & (L.arm == "S") & (L.model == model) & (L.fold == 0)]
    m = mine.merge(theirs[["pair_id", "target", "y", "yhat"]], on=["pair_id", "target"], suffixes=("", "_lad"),
                   how="outer", indicator=True)
    same_pairs = bool((m._merge == "both").all())
    dyhat = float(np.abs(m.yhat - m.yhat_lad).max()) if same_pairs else float("inf")
    dy = float(np.abs(m.y - m.y_lad).max()) if same_pairs else float("inf")
    vk_ref = ref.get("val_kcal_mae") or {}
    dvk = {t: abs(float(rec["val_kcal_mae"][t]) - float(vk_ref[t])) for t in T10 if t in vk_ref}
    cell_ok = norm(rec["chosen"]["cell"]) == norm((ref.get("chosen") or {}).get("cell"))
    ep_ok = int(rec["chosen"]["best_epoch"]) == int((ref.get("chosen") or {}).get("best_epoch", -1))
    checks = dict(same_cell=cell_ok, same_best_epoch=ep_ok,
                  val_kcal_mae_within_1e6=bool(dvk) and len(dvk) == len(T10) and max(dvk.values()) <= CR_TOL["val_kcal"],
                  yhat_within_1e4=same_pairs and dyhat <= CR_TOL["yhat"], same_test_pairs=same_pairs, same_y=dy <= 1e-12,
                  gates_C4=TT.gates_ok(rec))
    out = dict(check=f"C-R{'T' if model == 'TF' else 'G'}{'2' if tag == 'rg2' else ''}", model=model,
               ok=bool(all(checks.values())), checks=checks, gpu=FT.versions()["gpu"], gpu_ladder=ref.get("gpu"),
               chosen=rec["chosen"], chosen_ladder=ref.get("chosen"),
               tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in rec["tune"]],
               tune_ladder=ref.get("tune"), val_kcal_mae=rec["val_kcal_mae"], val_kcal_mae_ladder=vk_ref,
               max_abs_dval_kcal=max(dvk.values()) if dvk else None, max_abs_dyhat=dyhat, max_abs_dy=dy,
               max_abs_dyhat_per_target={t: float(np.abs(g.yhat - g.yhat_lad).max()) for t, g in m.groupby("target")}
               if same_pairs else None, n_test=rec["n_test"], wall_seconds=rec["wall_seconds"],
               wall_seconds_ladder=ref.get("wall_seconds"), unit_json_sha256=C.sha256(root / f"{uname(u)}.json"),
               ladder_unit=key, code_sha256=CODE_SHA, tolerances=CR_TOL)
    if tag == "rg2":
        first = root.parent / "GNN" / f"{uname(u)}.parquet"
        if first.is_file():
            a = pd.read_parquet(first).merge(mine, on=["pair_id", "target"], suffixes=("_1", "_2"))
            out["rg2_vs_rg_max_abs_dyhat"] = float(np.abs(a.yhat_1 - a.yhat_2).max())
    name = {"TF": "CR_tf.json", "GNN": "CR_gnn.json"}[model] if tag != "rg2" else "CR_gnn2.json"
    write_json(C.RES / name, out)
    print(json.dumps({k: out[k] for k in ("check", "ok", "checks", "chosen", "max_abs_dyhat", "max_abs_dval_kcal")},
                     indent=1, default=str))
    if model == "GNN" and tag is None and not out["ok"]:
        print("[C-RG] failed: running the repeat C-RG2 (spec 5 exception rule; tests_m.py merge decides)", flush=True)
        cmd_cr("GNN", "rg2")
    return 0 if out["ok"] else 3


def cmd_run(model, mode, i, n):
    FT.require_gpu()
    mode = int(mode)
    if (model, mode) not in K_UNITS:
        C.stop(f"run {model} {mode}: not a K unit of CH3M ({K_UNITS})")
    i, n = C.array_task(i), int(n)
    us = [u for u in unit_list() if u["model"] == model and u["mode"] == mode][i::n]
    if not us:
        return 0
    pre = gate_prereg()
    gc = gate_c_tests()
    D = data_for(mode)
    print(f"[K] PREREG ok (git {pre.get('git_head')}, md {_GATE['prereg'][:12]}), C_tests ok ({gc['c_tests_sha256'][:12]}), "
          f"{FT.versions()}, mode {DC3.current_mode()['TARGETS']}, {model} slice {i}/{n}: {len(us)} units", flush=True)
    for u in us:
        t0 = time.time()
        rec = run_unit(D, u, targets_of(mode), model, root=C.UNITS)
        print(f"[K] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


def load_units(model, mode, root=None):
    root = C.UNITS if root is None else Path(root)
    names = [uname(u) for u in unit_list() if u["model"] == model and u["mode"] == mode]
    missing = [x for x in names if not ((root / f"{x}.json").is_file() and (root / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C.read_json(root / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C.sha256(root / f"{x}.parquet") != r.get("parquet_sha256"):
            C.stop(f"{x}.parquet changed after its json")
        if r.get("smoke") or r.get("limit") or not TT.gates_ok(r):
            C.stop(f"gate C-4 failed (or smoke / limited unit) in {x}")
        if not all(f["leak"]["ok"] for f in r["tune"] + r["final"]):
            C.stop(f"C-5 failed in {x}")
        if len(r.get("final") or []) != len(F.FINAL_SEEDS):
            C.stop(f"{x}: {len(r.get('final') or [])} final seeds")
        if list(r.get("targets") or []) != targets_of(mode):
            C.stop(f"{x}: targets {r.get('targets')}")
    return recs, []


def cmd_collect():
    import prereg_m as PR
    md_now = C.sha256(PR.MD) if PR.MD.is_file() else None
    _GATE["prereg"] = md_now
    parts, report = [], dict(units={}, written=C.now(), code_sha256=CODE_SHA, prereg_md_sha256=md_now)
    missing_all = []
    for model, mode in K_UNITS:
        recs, missing = load_units(model, mode)
        if missing:
            missing_all += missing
            continue
        D = data_for(mode)
        for x, r in recs.items():
            u = r["unit"]
            now = fingerprint(D, u, D.split(u["fold"]), targets_of(mode), model)
            if _no_gpu(r["fingerprint"]) != _no_gpu(now):
                C.stop(f"{x} was computed with other code / inputs: keys "
                       f"{sorted(k for k in set(now) | set(r['fingerprint']) if k != 'gpu' and r['fingerprint'].get(k) != now.get(k))}")
            parts.append(pd.read_parquet(C.UNITS / f"{x}.parquet")[PRED_COLS])
            report["units"][x] = dict(chosen=r["chosen"], n_train=r["n_train"], n_test=r["n_test"], n_val=r["n_val"],
                                      val_kcal_mae=r["val_kcal_mae"],
                                      tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in r["tune"]],
                                      gpu=(r.get("fingerprint") or {}).get("gpu"), wall_seconds=r.get("wall_seconds"),
                                      gate_C4_max=max(max(t["gate_M1"]["anti_max"], t["gate_M1"]["self_max"])
                                                      for t in r["tune"] + r["final"]))
    if missing_all:
        print(f"GATE: {len(missing_all)} units missing: {missing_all}", file=sys.stderr)
        return 2
    pred = pd.concat(parts, ignore_index=True)
    if pred.duplicated(["pair_id", "rung", "model", "target"]).any():
        C.stop("duplicated prediction rows")
    out = C.CH3M / "collect" / "nn.parquet"
    C.write_parquet(out, pred)
    report.update(n_rows=int(len(pred)), parquet=str(out), parquet_sha256=C.sha256(out),
                  gpu_hours=float(sum(u.get("wall_seconds") or 0 for u in report["units"].values()) / 3600))
    write_json(C.CH3M / "collect" / "nn_report.json", report)
    print(f"[collect nn] {len(report['units'])} units, {len(pred)} rows, {report['gpu_hours']:.2f} GPU-h", flush=True)
    return 0


# ---------------------------------------------------------------- Phase W (spec 9)
def model_card(D, sel, tr, finals, model, rung, targets, std0, train_mae):
    rx = D.rxn_rows(tr)
    return dict(model=f"{rung} . S . {model} (variant A+G)", rung=rung, arm=ARM, model_type=model, variant=VARIANT,
                cell=sel["cell"], n_epochs=sel["n_epochs"], seeds=list(F.FINAL_SEEDS), selection=sel,
                files=[f"{final_name(rung, model, j)}.pt" for j in range(len(F.FINAL_SEEDS))],
                n_train_pairs=int(len(tr)), train_pairs_sha256=DL.list_sha(D._ids(tr)), n_train_reactions=int(len(rx)),
                targets=list(targets), conversion=dict(M=C.M_ROWS, rows=T10, cols=T3),
                token=dict(version=D.trep.get("token_version"), atom_version=DC3.DC.ATOM_VERSION,
                           elements=D.layout()["elements"], bin_cols=list(D.bin_cols), num_cols=list(D.num_cols),
                           d_in=int(D.d_in)),
                glob_cols=list(D.trep["glob_cols"]), d_glob=int(D.d_glob),
                s_y=dict(zip(targets, np.asarray(std0["sy"], float).tolist())), std_sha256=TT.std_sha(std0),
                code_sha256={**{f"analysis/ch3m/{p.name}": C.sha256(p) for p in sorted(HERE.glob("*.py"))}, **CODE_SHA},
                versions=FT.versions(), inputs_sha256=D.sha, train_mae_reference_only=train_mae,
                cv_performance=dict(source="results/EM_report.json (K cross-validation)"),
                applicability=dict(thresholds_yhat_sd=None, elements=list(DC3.DC.ELEMENTS), charge="neutral (charge 0)",
                                   ts_atoms=dict(min=int(D.n[rx].min()), max=int(D.n[rx].max()))))


def cmd_final(cand):
    """spec 9: DEPLOY 13 rule for TF10 / GNN10 / GNN3 -> $CH3M/model."""
    FT.require_gpu()
    if cand not in ("TF10", "GNN10", "GNN3"):
        C.stop(f"final {cand}: only TF10 / GNN10 / GNN3 (spec 9)")
    model, mode = cand[:-2] if cand.endswith("10") else cand[:-1], 10 if cand.endswith("10") else 3
    pre = gate_prereg()
    gate_c_tests()
    er = C.RES / "EM_report.json"
    if not er.is_file():
        C.not_ready(f"{er} missing (eval_m.py run)")
    rep = C.read_json(er)
    want = (rep.get("choice") or {}).get("final_model")
    if want != cand:
        C.stop(f"EM_report choice final_model {want} != {cand}")
    if cand == "TF10":
        recs = {}
        for x, r in (load_units("TF", 10)[0] or {}).items():
            recs[x] = r
    else:
        recs, missing = load_units(model, mode)
        if missing:
            C.not_ready(f"units missing {missing}")
    sel = TT.choose_final(recs) if model == "TF" else choose_final_grid(recs, model)
    D = data_for(mode)
    return train_final(D, sel, recs, model, mode, root=C.MODEL_DIR, prereg_head=pre.get("git_head"))


def choose_final_grid(recs, model):
    """train_tf.choose_final with the model's grid (mode cell; ties: lowest mean tuning val, then grid order)."""
    from collections import Counter
    G = grid(model)
    keys = [json.dumps(norm(r["chosen"]["cell"]), sort_keys=True) for r in recs.values()]
    cnt = Counter(keys)
    top = max(cnt.values())
    cand = [k for k in cnt if cnt[k] == top]
    gidx = {json.dumps(norm(c), sort_keys=True): i for i, c in enumerate(G)}

    def mean_val(k):
        v = [t["val"] for r in recs.values() for t in r["tune"] if json.dumps(norm(t["cell"]), sort_keys=True) == k]
        return float(np.mean(v)) if v else np.inf
    ck = min(cand, key=lambda k: (mean_val(k), gidx[k]))
    eps = sorted(int(r["chosen"]["best_epoch"]) for r in recs.values())
    return dict(cell=json.loads(ck), n_epochs=int(np.median(eps)), cell_counts=dict(cnt),
                tie_break=dict(candidates=cand, mean_tuning_val={k: mean_val(k) for k in cand}), best_epochs=eps,
                rule="mode of the 5 chosen cells (ties: lowest mean tuning val, then grid order); n_epochs = median(best_epoch)")


def train_final(D, sel, recs, model, mode, root, prereg_head=None, check_n=W_CHECK_N):
    targets = targets_of(mode)
    rung = C.RUNGS[mode]
    mode_check(targets)
    sp = D.split("all")
    tr = sp["train"]
    ep = int(sel["n_epochs"])
    root = _check_root(root, False)
    root.mkdir(parents=True, exist_ok=True)
    wfp = norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, model=model, rung=rung, inputs_sha256=D.sha,
                    pair_list=D.pair_list_sha, targets=targets, train_pairs=DL.list_sha(D._ids(tr)),
                    selection=dict(cell=sel["cell"], n_epochs=ep), k_units={x: r["parquet_sha256"] for x, r in recs.items()},
                    prereg=_GATE.get("prereg"), rules=dict(seeds=F.FINAL_SEEDS, check_n=check_n), **FT.versions()))
    wj = root / f"W_unit_{rung}_{model}.json"
    if wj.is_file():
        if C.read_json(wj).get("fingerprint") != wfp:
            C.stop(f"{wj} has a different fingerprint")
        print("[W] cached", flush=True)
        return 0
    rng = np.random.default_rng(F.SEED)
    chk = np.sort(rng.choice(tr, size=min(check_n, len(tr)), replace=False))
    u = dict(rung=rung, arm=ARM, model=model, fold="all")
    finals, preds_chk, preds_tr, std0 = [], [], [], None
    t0 = time.time()
    for s in F.FINAL_SEEDS:
        r = FT.fit(D, model, sel["cell"], VARIANT, tr, None, seed=s, n_epochs=ep, log=10)
        lk = TT.leak_check(D, r["std"], tr, (), np.zeros(0, dtype=int))
        g = FT.gate_m1(r["model"], D, r["std"], tr, seed=s)
        if not g["ok"]:
            C.stop(f"gate C-4 failed (final, seed {s}): {g}")
        if _outdim(r, D, tr) != len(targets):
            C.stop("C-T: final model output width")
        sd = TT.std_dict(r["std"], r["sy"])
        if std0 is None:
            std0 = sd
        elif TT.std_sha(sd) != TT.std_sha(std0):
            C.stop("final fits: standardisation differs between seeds")
        preds_chk.append(TT.kcal(r, D, chk))
        preds_tr.append(TT.kcal(r, D, tr))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, leak=lk, std_sha256=TT.std_sha(sd),
                           history_tail=r["history"][-3:]))
        save_torch(root / f"{final_name(rung, model, s - F.SEED)}.pt", pt_payload(D, r, sel["cell"], u, targets, model))
        print(f"[W] seed {s}: {r['seconds']:.0f} s, gate {g}", flush=True)
        del r
    Yc, Yt = np.stack(preds_chk), np.stack(preds_tr)
    pid_c = D.P.index.to_numpy()[chk].astype(str)
    C.write_csv(root / f"W_check_pairs_{rung}_{model}.csv",
                pd.DataFrame(dict(pair_id=pid_c, a_rxn=D.P.a_rxn.to_numpy()[chk], b_rxn=D.P.b_rxn.to_numpy()[chk])))
    rows = [pd.DataFrame(dict(pair_id=pid_c, target=t, yhat=Yc[:, :, j].mean(0), yhat_sd=Yc[:, :, j].std(0),
                              **{f"yhat_s{k}": Yc[k, :, j] for k in range(len(Yc))})) for j, t in enumerate(targets)]
    C.write_parquet(root / f"W_check_pred_{rung}_{model}.parquet", pd.concat(rows, ignore_index=True))
    train_mae = {t: float(np.abs(Yt[:, :, j].mean(0) - D.y[tr, j]).mean()) for j, t in enumerate(targets)}
    card = model_card(D, dict(sel, n_epochs=ep), tr, finals, model, rung, targets, std0, train_mae)
    adp = C.RES / "EM_ad.json"
    if adp.is_file():
        th = (C.read_json(adp).get("thresholds") or {}).get(C.label_of(rung, model))
        card["applicability"]["thresholds_yhat_sd"] = th
    write_json(root / "MODEL_CARD.json", card)
    C.write_bytes(C.RES / "MODEL_CARD_ch3m.json", (root / "MODEL_CARD.json").read_bytes())
    rep = dict(phase="W", candidate=C.label_of(rung, model), selection=sel, n_train_pairs=int(len(tr)),
               n_in_T_main=sp.get("n_in_T_main"), finals=finals, gate_C4_ok=all(f["gate_M1"]["ok"] for f in finals),
               train_mae_reference_only=train_mae, model_card_sha256=C.sha256(root / "MODEL_CARD.json"),
               pt_sha256={f"{final_name(rung, model, j)}.pt": C.sha256(root / f"{final_name(rung, model, j)}.pt")
                          for j in range(len(F.FINAL_SEEDS))},
               prereg_git_head=prereg_head, wall_seconds=time.time() - t0, versions=FT.versions(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    write_json(C.RES / "WM_report.json", rep)
    write_json(wj, dict(fingerprint=wfp, report_sha256=C.sha256(C.RES / "WM_report.json")))
    print(json.dumps(dict(cell=sel["cell"], n_epochs=ep, n_train=len(tr), gate=rep["gate_C4_ok"]), indent=1))
    return 0


def main(argv):
    a = argv[1:]
    if a == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a == ["smoke"]:
        return cmd_smoke()
    if a[:1] == ["cr"] and len(a) in (2, 3):
        return cmd_cr(a[1], a[2] if len(a) == 3 else None)
    if a[:1] == ["run"] and len(a) == 5:
        return cmd_run(a[1], a[2], a[3], a[4])
    if a == ["collect"]:
        return cmd_collect()
    if a[:1] == ["final"] and len(a) == 2:
        return cmd_final(a[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
