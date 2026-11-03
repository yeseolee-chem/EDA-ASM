#!/usr/bin/env python3
r"""train_tf.py — DEPLOY-MODEL phases C-3 / C-8 / K / W: the lf . S . TF model (docs/specs/DEPLOY_MODEL.md 9, 11, 13).
GPU work in env_d1fu2 (torch 2.2.1+cu118) through jobs/gpu_fu2.slurm; collect on CPU (jobs/cpu_py.slurm DEPENV=d1fu2).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python train_tf.py list                    unit names D__lf__S__TF__f0..4
  python train_tf.py smoke                   C-8: fold 0, grid cells smallest + largest, 2 epochs, 1 seed, gate C-4 on
                                             every fit -> results/C8_smoke.json (scratch $DEP/smoke, never read by K)
  python train_tf.py c3                      C-3: fold 0 on the C-2 data (DeployData('g2'): LADDER g2 tokens + LADDER
                                             usable rule) with the full rules (4 cells + 5 seeds) -> $DEP/c3/
                                             D__g2__S__TF__f0.*, compared with $LAD/units/L__g2__S__TF__f0 (same cell,
                                             |best epoch diff| <= 2, fold-0 test para MAE within +-2 % per target, max
                                             |dyhat| recorded) -> results/C3_regress.json
  python train_tf.py run <i|@I> <n>          units [i::n] -> $DEP/units/D__lf__S__TF__f<k>.{json,parquet}, __s0..4.pt.
                                             Gates (exit 3): PREREG (prereg_deploy.check: both files tracked, unchanged
                                             since HEAD, md sha256 = json md_sha256, inputs unchanged) and
                                             results/C_tests.json all_pass on this trainer code. Idempotent: a unit
                                             whose json exists is skipped after its fingerprint is compared (mismatch
                                             = STOP).
  python train_tf.py collect                 the 5 units -> results/K_predictions.parquet (L_predictions columns),
                                             K_val.csv, K_report.json; exit 2 while a unit is missing
  python train_tf.py final                   Phase W (spec 13; only after results/E_report.json G-C pass on the current
                                             K_predictions): mode cell (ties: lowest mean tuning validation of that
                                             cell over the folds, then grid order), n_epochs = median(best_epoch)
                                             (1-based, run_unit 399-405), seeds 20261019 + 0..4, standardisation on
                                             split('all') -> $DEP/model/D__lf__S__TF__all__s{0..4}.pt, MODEL_CARD.json,
                                             W_check_pairs.csv + W_check_pred.parquet (1,000 training pairs, I-2),
                                             W_unit.json; results/W_report.json

Training rules (spec 11-2 = FU2 M-1 / LADDER arm S): fu2/train.py fit (219), Std (140), gate_m1 (270), predict (186),
batch (163), epoch_batches (207) imported and called unchanged; run_unit = train_atoms.run_unit (1275-1373) S-arm path for
TF: Mo.build('TF', cell, d_in=29, 10, d_glob=182, 'A+G'), grid 2/3 layers x 64/128 width, tuning on inner_train with
early stopping / cell choice on inner_val (mean standardised MAE of the 9 primary targets, ties grid order), final =
outer train pairs, the tuned best epoch, seeds 20261019 + 0..4, mean; gate C-4 (= M-1', 200 random test pairs; W: 200
random training pairs) on every fit; leak check C-5 (train_atoms.leak_check 985-1003) on every fit.
Each .pt = dict(state, cell, unit, std=dict(mu, sd, gmu, gsd, sy), rung, arm, variant, model, d_in, d_glob, TARGETS,
layout) (run_unit 1343-1347 format + d_in / d_glob / TARGETS / input layout).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402
import data_lf as DL  # noqa: E402

FT, Mo, F = DL.FT, DL.Mo, DL.F
import torch  # noqa: E402

CODE_VERSION = "deploy-tf-1"
MODEL, VARIANT, ARM = "TF", "A+G", "S"
GRID = list(Mo.GRID[MODEL])
TARGETS = list(F.TARGETS)
NF = DL.NF
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                                     # = train_atoms.PRED_COLS (L_predictions.parquet)
SMOKE_ROOT = C.DEP / "smoke"
C3_ROOT = C.DEP / "c3"
LAD_UNITS = C.LAD / "units"
C3_REF = "L__g2__S__TF__f0"
C3_TOL = dict(epoch=2, rel_mae=0.02)
W_CHECK_N = 1000
CODE_FILES = ("train_tf.py", "data_lf.py", "dep_common.py")
# hashed once at import: the code this process runs (fu2 train.py review F3 rule)
CODE_SHA = {**{f: C.sha256(HERE / f) for f in CODE_FILES},
            **{(k if k == "d1pair_common.py" else f"fu2/{k}"): v for k, v in FT.CODE_SHA.items()}}
TRAINER_FILES = CODE_FILES                                       # the files C_tests.json must have tested (GA-14 rule)
_GATE: dict = {}


# ================================================================ helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def norm(o):
    """the value as it reads back from json (fingerprints are compared this way; = train_atoms.norm)."""
    return json.loads(json.dumps(_jsonable(o)))


def write_json(path, obj):
    return C.write_json(path, _jsonable(obj))


def uname(u):
    return f"D__{u['rung']}__{ARM}__{MODEL}__f{u['fold']}"


def final_name(rung, j):
    return f"D__{rung}__{ARM}__{MODEL}__all__s{j}"


def unit_list(rung=C.RUNG):
    return [dict(rung=rung, arm=ARM, model=MODEL, fold=k) for k in range(NF)]


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


def std_dict(std, sy):
    return dict(mu=std.mu, sd=std.sd, gmu=std.gmu, gsd=std.gsd, sy=sy)


def pt_payload(D, r, cell, u):
    return dict(state=r["model"].state_dict(), cell=cell, unit=u, std=std_dict(r["std"], r["sy"]), rung=D.rung,
                arm=ARM, variant=VARIANT, model=MODEL, d_in=int(D.d_in), d_glob=int(D.d_glob), TARGETS=list(TARGETS),
                layout=D.layout())


def std_sha(std_like):
    """sha256 of the standardisation statistics (mu, sd, gmu, gsd, sy as float64 bytes, this order)."""
    import hashlib
    h = hashlib.sha256()
    for k in ("mu", "sd", "gmu", "gsd", "sy"):
        h.update(np.ascontiguousarray(np.asarray(std_like[k], dtype=np.float64)).tobytes())
    return h.hexdigest()


def kcal(r, D, rows):
    """S-arm final prediction (kcal/mol) = train_atoms.final_kcal S branch (978-981): FT.predict x s_y."""
    return FT.predict(r["model"], r["T"], D.ra[rows], D.rb[rows]) * r["sy"]


def leak_check(D, std, fit_rows, others, test_rows):
    """C-5 = train_atoms.leak_check (985-1003) S branch: scaling pairs = the fit's training pairs, none of them a
    validation / test pair, no test reaction among the scaling reactions (STOP otherwise)."""
    used = set(np.asarray(std.pair_rows).tolist())
    if used != set(np.asarray(fit_rows).tolist()):
        C.stop("C-5: scaling pairs differ from the fit's training pairs")
    for o in others:
        if used & set(np.asarray(o).tolist()):
            C.stop("C-5: a validation / test pair is among the fit's scaling pairs")
    trx = set(D.ra[test_rows].tolist()) | set(D.rb[test_rows].tolist()) if len(test_rows) else set()
    if trx & set(np.asarray(std.rxn_rows).tolist()):
        C.stop("C-5: a test reaction is among the fit's scaling reactions")
    want = set(D.ra[np.asarray(fit_rows)].tolist()) | set(D.rb[np.asarray(fit_rows)].tolist())
    if set(np.asarray(std.rxn_rows).tolist()) != want:
        C.stop("C-5: scaling reactions differ from the reactions of the fit's training pairs")
    return dict(n_fit_pairs=len(used), n_scale_rxns=int(len(std.rxn_rows)), n_test_rxn_overlap=0, ok=True)


def gates_ok(r):
    fits = list(r.get("tune") or []) + list(r.get("final") or [])
    return bool(fits) and all(f["gate_M1"]["ok"] for f in fits)


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


# ================================================================ gates
def gate_prereg():
    """spec 10: prereg_deploy.check() (exit 3 otherwise); records the md sha256 for the unit fingerprint."""
    import prereg_deploy as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    """spec 9 / 16 C: results/C_tests.json all_pass, and its trainer code sha256 = the files now (the C tests ran on the
    code that trains; GEO2 GA-14 precedent)."""
    p = C.RES / "C_tests.json"
    c = C.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K / W)")
    rec = c.get("trainer_code_sha256") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun the C tests (tests.py)")
    return dict(c_tests_sha256=C.sha256(p))


def _check_root(root, smoke):
    r = C.assert_write_ok(Path(root) / "probe").parent
    for prod in (C.UNITS, C.MODEL_DIR):
        p = Path(os.path.abspath(prod))
        if r == p or p in r.parents:
            if smoke:
                C.stop(f"smoke / test runs never write {p}")
            if not _GATE.get("prereg"):
                C.stop(f"writing {p} needs the PREREG gate (train_tf.py run / final)")
    return r


def fingerprint(D, u, sp, smoke=False, limit=None):
    return norm(dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=D.rung, arm=ARM, model=MODEL, variant=VARIANT,
        usable_rule=D.usable_rule, inputs_sha256=D.sha, pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"],
        prereg=None if smoke else _GATE.get("prereg"), grid=GRID, limit=limit, smoke=bool(smoke),
        rules=dict(lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                   patience=F.PATIENCE, seeds=F.FINAL_SEEDS, tune_seed=F.SEED, antisym_tol=F.ANTISYM_TOL,
                   antisym_n=F.ANTISYM_N, val="first inner fold (FU2 split files)"),
        **FT.versions()))


# ================================================================ one CV unit (= train_atoms.run_unit S arm)
def _limit(sp, limit):
    if not limit:
        return sp
    sp = dict(sp)
    for k in DL.FOLD_KEYS:
        sp[k] = sp[k][:int(limit)]
    return sp


def run_unit(D, u, root=C.UNITS, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50, smoke=False,
             limit=None):
    """tune (grid cells on inner_train, early stopping on inner_val) -> final (seeds, best epoch, train) -> test
    predictions + gate C-4 + leak check C-5 for one unit (= train_atoms.run_unit 1275-1373, arm S, model TF)."""
    root = _check_root(root, smoke or bool(limit))
    root.mkdir(parents=True, exist_ok=True)
    if u["rung"] != D.rung or u["model"] != MODEL or u["arm"] != ARM:
        C.stop(f"unit {u} does not match the data ({D.rung}, {ARM}, {MODEL})")
    name = uname(u)
    sp = _limit(D.split(u["fold"]), limit)
    fp = fingerprint(D, u, sp, smoke, limit)
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
        r = FT.fit(D, MODEL, cell, VARIANT, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log)
        lk = leak_check(D, r["std"], itr, (iva, te), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=F.SEED + ci)        # inputs only, no labels (M-1)
        vk = np.abs(kcal(r, D, iva) - D.y[iva]).mean(0)
        tune.append(dict(cell=cell, grid_index=gi, best_epoch=r["best_epoch"], val=r["best_val"],
                         val_per_target=r["best_val_per_target"], val_kcal_mae=dict(zip(TARGETS, vk.tolist())),
                         epochs_run=len(r["history"]), seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
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
        r = FT.fit(D, MODEL, cell, VARIANT, tr, None, seed=s, n_epochs=ep, log=log)
        lk = leak_check(D, r["std"], tr, (te,), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=s)
        if not g["ok"]:
            C.stop(f"gate C-4 failed ({name}, seed {s}): {g}")
        preds.append(kcal(r, D, te))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
                           std_sha256=std_sha(std_dict(r["std"], r["sy"])), history_tail=r["history"][-3:]))
        save_torch(root / f"{name}__s{s - F.SEED}.pt", pt_payload(D, r, cell, u))
        del r
    Y = np.stack(preds)                                                    # [seeds, n_test, 10] kcal/mol
    pid = D.P.index.to_numpy()[te].astype(str)
    rows = [pd.DataFrame(dict(pair_id=pid, ref_kind=D.P.ref_kind.to_numpy()[te], pos=D.P.pos.to_numpy()[te],
                              fold=int(u["fold"]), rung=u["rung"], arm=ARM, model=MODEL, target=t,
                              y=D.y[te, j], yhat=Y[:, :, j].mean(0), base=0.0, beta=0.0,
                              yhat_seed_sd=Y[:, :, j].std(0))) for j, t in enumerate(TARGETS)]
    pred = pd.concat(rows, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    C.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, fingerprint=fp, smoke=bool(smoke), limit=limit, variant=VARIANT, tune=tune,
               chosen=dict(cell=cell, best_epoch=ep, index=best, grid_index=tune[best]["grid_index"]), final=finals,
               val_kcal_mae=tune[best]["val_kcal_mae"], val_pairs_sha256=sp["val_sha"], test_pairs_sha256=sp["test_sha"],
               n_val=len(iva), n_train=len(tr), n_test=len(te), n_inner_train=len(itr), n_inner_val=len(iva),
               dropped=sp["dropped"], parquet_sha256=C.sha256(pq), inputs=D.inputs, usable_rule=D.usable_rule,
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    write_json(js, rec)
    return rec


# ================================================================ commands
def cmd_smoke():
    """C-8 (spec 9; = FU2 / LADDER smoke): fold 0, cells smallest + largest, 2 epochs, 1 final seed, gate C-4 on every
    fit, on the lf data -> results/C8_smoke.json; scratch $DEP/smoke (its D__* files removed first)."""
    FT.require_gpu()
    _check_root(SMOKE_ROOT, True)
    if SMOKE_ROOT.is_dir():
        for p in SMOKE_ROOT.glob("D__*"):
            C.assert_write_ok(p)
            p.unlink()
    D = DL.get_data("lf")
    u = dict(rung=D.rung, arm=ARM, model=MODEL, fold=0)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rec = run_unit(D, u, root=SMOKE_ROOT, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, len(GRID) - 1], log=1,
                   smoke=True)
    eps = [t["seconds"] / max(1, t["epochs_run"]) for t in rec["tune"]]
    n_par = [sum(p.numel() for p in Mo.build(MODEL, GRID[c], D.d_in, len(TARGETS), D.d_glob, VARIANT).parameters())
             for c in (0, len(GRID) - 1)]
    ok = gates_ok(rec)
    out = dict(check="C-8", ok=bool(ok), versions=FT.versions(), device=str(FT.DEV), unit=uname(u),
               seconds=time.time() - t0, tune_epoch_seconds=dict(smallest=eps[0], largest=eps[-1]),
               n_params=dict(smallest=n_par[0], largest=n_par[1]),
               gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
               gate_C4_tune=[t["gate_M1"] for t in rec["tune"]], gate_C4_final=[f["gate_M1"] for f in rec["final"]],
               leak_C5=[t["leak"] for t in rec["tune"]] + [f["leak"] for f in rec["final"]],
               n_train=rec["n_train"], n_test=rec["n_test"], n_inner_train=rec["n_inner_train"],
               n_inner_val=rec["n_inner_val"], dropped=rec["dropped"], d_in=int(D.d_in), d_glob=int(D.d_glob),
               code_sha256=CODE_SHA)
    write_json(C.RES / "C8_smoke.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "seconds", "tune_epoch_seconds", "gpu_mem_peak_gb")}, indent=1))
    return 0 if ok else 3


def para_mae(pred, P):
    """fold test para MAE per target: P_main reference pairs, pos p (the FU2 / LADDER primary para set)."""
    pm = P.reindex(pred.pair_id.unique())
    keep = set(pm.index[(pm.ref_kind == "reference") & pm.in_P_main.astype(bool) & (pm.pos == "p")])
    q = pred[pred.pair_id.isin(keep)]
    return {t: float((g.yhat - g.y).abs().mean()) for t, g in q.groupby("target")}, len(keep)


def cmd_c3():
    """C-3 (spec 9): train_tf fold 0 on the C-2 data on the LADDER unit GPU type, compared with L__g2__S__TF__f0."""
    FT.require_gpu()
    ref_js, ref_pq = LAD_UNITS / f"{C3_REF}.json", LAD_UNITS / f"{C3_REF}.parquet"
    if not (ref_js.is_file() and ref_pq.is_file()):
        C.not_ready(f"{ref_js} / {ref_pq} missing")
    ref = C.read_json(ref_js)
    gpu_ref = (ref.get("fingerprint") or {}).get("gpu")
    D = DL.get_data("g2")
    u = dict(rung="g2", arm=ARM, model=MODEL, fold=0)
    rec = run_unit(D, u, root=C3_ROOT)                     # full rules; not a production root (no PREREG needed)
    mine = pd.read_parquet(C3_ROOT / f"{uname(u)}.parquet")
    theirs = pd.read_parquet(ref_pq)
    m = mine.merge(theirs[["pair_id", "target", "y", "yhat"]], on=["pair_id", "target"], suffixes=("", "_lad"),
                   how="outer", indicator=True)
    same_pairs = bool((m._merge == "both").all())
    dy = float(np.abs(m.y - m.y_lad).max()) if same_pairs else float("inf")
    dyhat = float(np.abs(m.yhat - m.yhat_lad).max()) if same_pairs else float("inf")
    mae_me, n_para = para_mae(mine, D.P)
    mae_lad, _ = para_mae(theirs, D.P)
    rel = {t: (mae_me[t] - mae_lad[t]) / mae_lad[t] for t in TARGETS}
    cell_ok = norm(rec["chosen"]["cell"]) == norm(ref["chosen"]["cell"])
    ep_diff = int(rec["chosen"]["best_epoch"]) - int(ref["chosen"]["best_epoch"])
    gpu_now = FT.versions()["gpu"]
    checks = dict(i_same_cell=cell_ok, ii_best_epoch_within_2=abs(ep_diff) <= C3_TOL["epoch"],
                  iii_para_mae_within_2pct=all(abs(v) <= C3_TOL["rel_mae"] for v in rel.values()),
                  same_test_pairs=same_pairs, same_y=dy <= 1e-12, gates_C4=gates_ok(rec))
    out = dict(check="C-3", ok=bool(all(checks.values())), checks=checks, gpu=gpu_now, gpu_ladder=gpu_ref,
               same_gpu_type=gpu_now == gpu_ref, chosen=rec["chosen"], chosen_ladder=ref["chosen"],
               best_epoch_diff=ep_diff, tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"])
                                             for t in rec["tune"]],
               tune_ladder=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in ref["tune"]],
               n_test_para=n_para, para_mae=mae_me, para_mae_ladder=mae_lad, para_mae_rel_diff=rel,
               max_abs_dyhat=dyhat, max_abs_dy=dy, n_test=rec["n_test"], n_train=rec["n_train"],
               wall_seconds=rec["wall_seconds"], wall_seconds_ladder=ref.get("wall_seconds"),
               unit_json_sha256=C.sha256(C3_ROOT / f"{uname(u)}.json"), code_sha256=CODE_SHA,
               tolerances=C3_TOL)
    if gpu_now != gpu_ref:
        out["note"] = "GPU type differs from the LADDER unit; the spec asks for the same type (RTX 3090, gpu1)"
    write_json(C.RES / "C3_regress.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "checks", "chosen", "best_epoch_diff", "max_abs_dyhat")}, indent=1,
                     default=str))
    return 0 if out["ok"] else 3


def cmd_run(i, n):
    FT.require_gpu()
    i = C.array_task(i)
    n = int(n)
    if not (0 <= i < n):
        print(f"usage: run <i|@I> <n> with 0 <= i < n (got {i}, {n})", file=sys.stderr)
        return 2
    us = unit_list()[i::n]
    if not us:
        print(f"[K] slice {i} of {n}: no units", flush=True)
        return 0
    pre = gate_prereg()
    gc = gate_c_tests()
    print(f"[K] PREREG ok (git {pre.get('git_head')}, md {_GATE['prereg'][:12]}), C_tests ok "
          f"({gc['c_tests_sha256'][:12]}), {FT.versions()}, slice {i}/{n}: {len(us)} units", flush=True)
    D = DL.get_data("lf")
    for u in us:
        t0 = time.time()
        rec = run_unit(D, u, root=C.UNITS)
        print(f"[K] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


def load_units(root=None, rung=C.RUNG, allow_limited=False):
    """the 5 fold units of a rung: json / parquet / 5 .pt present, parquet sha256 = json, every gate ok, no smoke /
    limited unit (allow_limited: developer dry runs only) -> (records, missing names)."""
    root = C.UNITS if root is None else Path(root)
    names = [uname(u) for u in unit_list(rung)]
    missing = [x for x in names if not ((root / f"{x}.json").is_file() and (root / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C.read_json(root / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C.sha256(root / f"{x}.parquet") != r.get("parquet_sha256"):
            C.stop(f"{x}.parquet changed after its json")
        if r.get("smoke") or (r.get("limit") and not allow_limited) or not gates_ok(r):
            C.stop(f"gate C-4 failed (or smoke / limited unit) in {x}")
        for j in range(len(r.get("final") or [])):
            if not (root / f"{x}__s{j}.pt").is_file():
                C.stop(f"{x}__s{j}.pt missing")
        if len(r.get("final") or []) != len(F.FINAL_SEEDS):
            C.stop(f"{x}: {len(r.get('final') or [])} final seeds, not {len(F.FINAL_SEEDS)}")
    return recs, []


def cmd_collect(source="lf", allow_limited=False):
    D = DL.get_data(source)
    recs, missing = load_units(C.UNITS, D.rung, allow_limited)
    if missing:
        print(f"GATE: {len(missing)} units missing: {missing}", file=sys.stderr)
        return 2
    import prereg_deploy as PR
    md_now = C.sha256(PR.MD) if PR.MD.is_file() else None
    _GATE["prereg"] = md_now                          # the units must have been trained under the current PREREG md
    for x, r in recs.items():                         # every unit must carry the CURRENT fingerprint (FU2 collect)
        u = r["unit"]
        now = fingerprint(D, u, _limit(D.split(u["fold"]), r.get("limit")), False, r.get("limit"))
        if _no_gpu(r["fingerprint"]) != _no_gpu(now):
            C.stop(f"{x} was computed with other code / inputs: keys "
                   f"{sorted(k for k in set(now) | set(r['fingerprint']) if k != 'gpu' and r['fingerprint'].get(k) != now.get(k))}")
    pred = pd.concat([pd.read_parquet(C.UNITS / f"{x}.parquet")[PRED_COLS] for x in recs], ignore_index=True)
    if pred.duplicated(["pair_id", "target"]).any():
        C.stop("duplicated prediction rows (a pair in two test folds?)")
    if set(pred.rung) != {D.rung} or set(pred.arm) != {ARM} or set(pred.model) != {MODEL}:
        C.stop(f"K predictions must be rung {D.rung}, arm S, model TF only")
    out = C.RES / "K_predictions.parquet"
    C.write_parquet(out, pred)
    vrows = [dict(rung=D.rung, arm=ARM, model=MODEL, fold=int(r["unit"]["fold"]), target=t,
                  val_kcal_mae=float(r["val_kcal_mae"][t]), n_val=r.get("n_val"),
                  val_pairs_sha256=r.get("val_pairs_sha256")) for x, r in recs.items() for t in TARGETS]
    vout = C.RES / "K_val.csv"
    C.write_csv(vout, pd.DataFrame(vrows))
    rep = dict(n_units=len(recs), rung=D.rung, arm=ARM, model=MODEL, code=CODE_VERSION, code_sha256=CODE_SHA,
               prereg_md_sha256=md_now, inputs_sha256=D.sha, pair_list_sha=D.pair_list_sha, n_pairs_P_lf=len(D.pair_list),
               units={x: dict(chosen=r["chosen"], n_train=r["n_train"], n_test=r["n_test"], n_val=r["n_val"],
                              dropped=r["dropped"], val_kcal_mae=r["val_kcal_mae"],
                              tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in r["tune"]],
                              gpu=(r.get("fingerprint") or {}).get("gpu"), wall_seconds=r.get("wall_seconds"),
                              gate_C4_anti_max=max([t["gate_M1"]["anti_max"] for t in r["tune"]]
                                                   + [f["gate_M1"]["anti_max"] for f in r["final"]]),
                              gate_C4_self_max=max([t["gate_M1"]["self_max"] for t in r["tune"]]
                                                   + [f["gate_M1"]["self_max"] for f in r["final"]]))
                      for x, r in recs.items()},
               gpu_hours=float(sum(r.get("wall_seconds") or 0 for r in recs.values()) / 3600),
               n_test_pairs=int(pred.pair_id.nunique()), predictions=str(out), predictions_sha256=C.sha256(out),
               val_csv=str(vout), val_csv_sha256=C.sha256(vout))
    write_json(C.RES / "K_report.json", rep)
    print(f"[collect] {len(recs)} units -> {out.name} ({len(pred)} rows, {rep['n_test_pairs']} pairs)", flush=True)
    return 0


# ---------------------------------------------------------------- Phase W
def choose_final(recs):
    """spec 13: cell = mode of the 5 chosen cells (ties: lowest mean tuning validation of that cell over the folds,
    then grid order); n_epochs = median of the 5 best epochs (1-based fit epoch numbers; 5 values -> an integer)."""
    keys = [json.dumps(norm(r["chosen"]["cell"]), sort_keys=True) for r in recs.values()]
    cnt = Counter(keys)
    top = max(cnt.values())
    cand = [k for k in cnt if cnt[k] == top]
    gidx = {json.dumps(norm(c), sort_keys=True): i for i, c in enumerate(GRID)}

    def mean_val(k):
        v = [t["val"] for r in recs.values() for t in r["tune"] if json.dumps(norm(t["cell"]), sort_keys=True) == k]
        return float(np.mean(v)) if v else np.inf
    cell_key = min(cand, key=lambda k: (mean_val(k), gidx[k]))
    eps = sorted(int(r["chosen"]["best_epoch"]) for r in recs.values())
    if len(eps) % 2 != 1:
        C.stop(f"median of {len(eps)} best epochs is not an integer by rule (need an odd count)")
    ep = int(np.median(eps))
    return dict(cell=json.loads(cell_key), n_epochs=ep, cell_counts=dict(cnt), tie_break=dict(
        candidates=cand, mean_tuning_val={k: mean_val(k) for k in cand}), best_epochs=eps,
        rule="mode of the 5 chosen cells (ties: lowest mean tuning val of that cell, then grid order); n_epochs = "
             "median(best_epoch), best_epoch 1-based (fu2 fit), the fold final fits ran n_epochs = best_epoch")


def gate_e_report():
    """W only after G-C passed (spec 13 / 16 E) on the current K_predictions."""
    p = C.RES / "E_report.json"
    if not p.is_file():
        C.not_ready(f"{p} missing (eval_tf.py run)")
    e = C.read_json(p)
    kp = C.RES / "K_predictions.parquet"
    if (e.get("inputs_sha256") or {}).get("K_predictions") != C.sha256(kp):
        C.stop("E_report.json was computed on another K_predictions.parquet")
    gcv = e.get("G_C") or {}
    if gcv.get("pass") is not True:
        C.stop(f"G-C not passed (E_report.json G_C.pass = {gcv.get('pass')}): W is not run (spec 16 E)")
    return e


def model_card(D, sel, tr, finals, e_rep, ad, tr_ids_sha, train_mae, std0):
    files = sorted(HERE.glob("*.py"))
    rx = D.rxn_rows(tr)
    dip = sorted(set(D.P.dipole.to_numpy()[tr].astype(str))) if "dipole" in D.P.columns else None
    pos = sorted(set(D.P.pos.to_numpy()[tr].astype(str)))
    return dict(
        model="lf . S . TF (FU2 M-2 TF, variant A+G)", rung=D.rung, arm=ARM, variant=VARIANT, cell=sel["cell"],
        n_epochs=sel["n_epochs"], seeds=list(F.FINAL_SEEDS), selection=sel,
        files=[f"{final_name(D.rung, j)}.pt" for j in range(len(F.FINAL_SEEDS))],
        n_train_pairs=int(len(tr)), train_pairs_sha256=tr_ids_sha, n_train_reactions=int(len(rx)),
        token=dict(version=D.trep.get("token_version"), atom_version=C.ATOM_VERSION, elements=D.layout()["elements"],
                   bin_cols=list(D.bin_cols), num_cols=list(D.num_cols), d_in=int(D.d_in)),
        glob_cols=list(D.trep["glob_cols"]), d_glob=int(D.d_glob), targets=list(TARGETS),
        s_y=dict(zip(TARGETS, np.asarray(std0["sy"], float).tolist())), std_sha256=std_sha(std0),
        std_sha256_per_seed=[f["std_sha256"] for f in finals],
        code_sha256={**{f"analysis/deploy/{p.name}": C.sha256(p) for p in files},
                     "analysis/d1_pair/fu2/train.py": C.sha256(DL.FU2_DIR / "train.py"),
                     "analysis/d1_pair/fu2/models.py": C.sha256(DL.FU2_DIR / "models.py")},
        versions=FT.versions(), inputs_sha256=D.sha, usable_rule=DL.USABLE_RULES[D.usable_rule],
        cv_performance=dict(source="results/E_report.json (Phase K cross-validation; the final model has no test set)",
                            summary=e_rep.get("summary"), G_C=e_rep.get("G_C")),
        train_mae_reference_only=train_mae,
        applicability=dict(thresholds_yhat_sd=ad.get("thresholds"), threshold_rule=ad.get("rule"),
                           elements=list(C.ELEMENTS), charge="neutral (charge 0)",
                           ts_atoms=dict(min=int(D.n[rx].min()), max=int(D.n[rx].max()), spec="8-63"),
                           dipoles=dip, positions_spec=["p", "m"], positions_trained=pos))


def cmd_final():
    """Phase W (spec 13): gates (GPU, PREREG, C_tests, G-C passed on the current K predictions, E_ad present), the
    5 K units, the selection rule, then train_final on DeployData('lf') split('all')."""
    FT.require_gpu()
    pre = gate_prereg()
    gate_c_tests()
    e_rep = gate_e_report()
    adp = C.RES / "E_ad.json"
    if not adp.is_file():
        C.not_ready(f"{adp} missing (eval_tf.py run)")
    ad = C.read_json(adp)
    recs, missing = load_units()
    if missing:
        C.not_ready(f"K units missing: {missing}")
    sel = choose_final(recs)
    D = DL.get_data("lf")
    return train_final(D, sel, recs, e_rep, ad, root=C.MODEL_DIR, prereg_head=pre.get("git_head"))


def train_final(D, sel, recs, e_rep, ad, root, prereg_head=None, limit=None, n_epochs=None, check_n=W_CHECK_N,
                smoke=False):
    """the W fits (spec 13): split('all') training pairs (limit / n_epochs: developer dry runs only, never with the
    production root), seeds 20261019 + 0..4, gate C-4 on 200 random training pairs, leak check C-5, the 5 .pt,
    W_check_pairs.csv + W_check_pred.parquet (I-2), MODEL_CARD.json, results/W_report.json, W_unit.json (cache)."""
    sp = D.split("all")
    tr = sp["train"][:int(limit)] if limit else sp["train"]
    ep = int(n_epochs) if n_epochs else int(sel["n_epochs"])
    root = _check_root(root, smoke or bool(limit) or bool(n_epochs))
    root.mkdir(parents=True, exist_ok=True)
    wfp = norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, inputs_sha256=D.sha, pair_list=D.pair_list_sha,
                    train_pairs=DL.list_sha(D._ids(tr)), selection=dict(cell=sel["cell"], n_epochs=ep),
                    k_units={x: r["parquet_sha256"] for x, r in recs.items()}, prereg=_GATE.get("prereg"),
                    e_report=(C.sha256(C.RES / "E_report.json") if (C.RES / "E_report.json").is_file() else None),
                    rules=dict(seeds=F.FINAL_SEEDS, check_n=check_n), limit=limit, **FT.versions()))
    wj = root / "W_unit.json"
    if wj.is_file():
        old = C.read_json(wj)
        if old.get("fingerprint") != wfp:
            C.stop(f"{wj} has a different fingerprint: keys "
                   f"{sorted(k for k in set(wfp) | set(old.get('fingerprint') or {}) if (old.get('fingerprint') or {}).get(k) != wfp.get(k))}")
        print("[W] cached", flush=True)
        return 0
    rng = np.random.default_rng(F.SEED)
    chk = np.sort(rng.choice(tr, size=min(check_n, len(tr)), replace=False))
    u = dict(rung=D.rung, arm=ARM, model=MODEL, fold="all")
    finals, preds_chk, preds_tr, std0 = [], [], [], None
    t0 = time.time()
    for s in F.FINAL_SEEDS:
        r = FT.fit(D, MODEL, sel["cell"], VARIANT, tr, None, seed=s, n_epochs=ep, log=10)
        lk = leak_check(D, r["std"], tr, (), np.zeros(0, dtype=int))
        g = FT.gate_m1(r["model"], D, r["std"], tr, seed=s)               # C-4 on 200 random training pairs
        if not g["ok"]:
            C.stop(f"gate C-4 failed (final, seed {s}): {g}")
        sd = std_dict(r["std"], r["sy"])
        if std0 is None:
            std0 = sd
        elif std_sha(sd) != std_sha(std0):
            C.stop("final fits: standardisation differs between seeds (same training pairs)")
        preds_chk.append(kcal(r, D, chk))
        preds_tr.append(kcal(r, D, tr))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, leak=lk,
                           std_sha256=std_sha(sd), history_tail=r["history"][-3:]))
        save_torch(root / f"{final_name(D.rung, s - F.SEED)}.pt", pt_payload(D, r, sel["cell"], u))
        print(f"[W] seed {s}: {r['seconds']:.0f} s, gate {g}", flush=True)
        del r
    Yc, Yt = np.stack(preds_chk), np.stack(preds_tr)
    pid_c = D.P.index.to_numpy()[chk].astype(str)
    C.write_csv(root / "W_check_pairs.csv", pd.DataFrame(dict(pair_id=pid_c, a_rxn=D.P.a_rxn.to_numpy()[chk],
                                                              b_rxn=D.P.b_rxn.to_numpy()[chk])))
    rows = [pd.DataFrame(dict(pair_id=pid_c, target=t, yhat=Yc[:, :, j].mean(0), yhat_sd=Yc[:, :, j].std(0),
                              **{f"yhat_s{k}": Yc[k, :, j] for k in range(len(Yc))})) for j, t in enumerate(TARGETS)]
    C.write_parquet(root / "W_check_pred.parquet", pd.concat(rows, ignore_index=True))
    train_mae = {t: float(np.abs(Yt[:, :, j].mean(0) - D.y[tr, j]).mean()) for j, t in enumerate(TARGETS)}
    tid = D._ids(tr)
    card = model_card(D, dict(sel, n_epochs=ep), tr, finals, e_rep, ad, DL.list_sha(tid), train_mae, std0)
    write_json(root / "MODEL_CARD.json", card)
    rep = dict(phase="W", selection=sel, n_train_pairs=int(len(tr)), n_in_T_main=sp.get("n_in_T_main"),
               dropped=sp["dropped"], finals=finals, gate_C4_ok=all(f["gate_M1"]["ok"] for f in finals),
               train_mae_reference_only=train_mae, check_pairs=dict(n=int(len(chk)), file=str(root / "W_check_pairs.csv"),
                                                                    pred=str(root / "W_check_pred.parquet")),
               model_card=str(root / "MODEL_CARD.json"), model_card_sha256=C.sha256(root / "MODEL_CARD.json"),
               pt_sha256={f"{final_name(D.rung, j)}.pt": C.sha256(root / f"{final_name(D.rung, j)}.pt")
                          for j in range(len(F.FINAL_SEEDS))},
               prereg_git_head=prereg_head, limit=limit, wall_seconds=time.time() - t0, versions=FT.versions(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    write_json(C.RES / "W_report.json", rep)
    write_json(wj, dict(fingerprint=wfp, report_sha256=C.sha256(C.RES / "W_report.json")))
    print(json.dumps(dict(cell=sel["cell"], n_epochs=ep, n_train=len(tr), gate=rep["gate_C4_ok"]),
                     indent=1))
    return 0


def main(argv):
    a = argv[1:]
    if a[:1] == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a[:1] == ["smoke"] and len(a) == 1:
        return cmd_smoke()
    if a[:1] == ["c3"] and len(a) == 1:
        return cmd_c3()
    if a[:1] == ["run"] and len(a) == 3:
        return cmd_run(a[1], a[2])
    if a[:1] == ["collect"] and len(a) == 1:
        return cmd_collect()
    if a[:1] == ["final"] and len(a) == 1:
        return cmd_final()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
