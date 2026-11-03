#!/usr/bin/env python3
r"""train_w.py — DEPLOY-MODEL-W phases C-W9 / K / W2: lf . S . TF with within-group pairs (lfW) and its control lfV
(docs/specs/DEPLOY_MODEL_W.md 4, 5, 8, 12). GPU work in env_d1fu2 (torch 2.2.1+cu118) through
analysis/deploy/jobs/gpu_fu2.slurm (submitted from analysis/deploy/w); collect on CPU (jobs/cpu_py.slurm DEPENV=d1fu2).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python train_w.py list                 unit names D__lfW__S__TF__f0..4 (+ lfV controls)
  python train_w.py smoke                C-W9: fold 0 on the lf data, r in {0, 4} x grid cells smallest + largest, 2 epochs,
                                         1 final seed, gate C-W8 on every fit -> results/w/C9_smoke_w.json, the per-epoch
                                         times -> results/w/K_estimate.json (STOP when the estimate > 20 h per fold);
                                         scratch $DEPW/smoke (never read by K)
  python train_w.py run <i|@I> <n>       folds [i::n] -> $DEPW/units/D__lfW__S__TF__f<k>.{json,parquet}, __s0..4.pt and the
                                         control D__lfV__S__TF__f<k>.* (the same job). Gates (exit 3): PREREG_W
                                         (prereg_w.check) and results/w/C_tests_w.json all_pass on this trainer code.
                                         Idempotent: a fold whose two jsons exist is skipped after the fingerprint check
                                         (mismatch = STOP).
  python train_w.py collect              -> results/w/K_predictions_w.parquet (K_predictions columns, rung lfW / lfV),
                                         K_val_w.csv, K_report_w.json; exit 2 while a unit is missing
  python train_w.py final                Phase W2 (spec 12; only when results/w/D_decision.json adopts lfW): r = mode of
                                         the 5 lfW r (ties: smaller r), cell = mode among those folds (ties: lower mean
                                         tuning V_nl, then grid order), n_epochs = lower median of their best epochs,
                                         seeds 20261019 + 0..4, base standardisation on split('all') + all__train.npz
                                         -> $DEPW/model/D__lfW__S__TF__all__s{0..4}.pt, MODEL_CARD_W.json,
                                         W_check_pairs.csv / W_check_pred.parquet (I2-3), W2_unit.json; results/w/W2_report.json

fit_w = fu2/train.py fit (219-267) copied, changed only by spec 4 W-1 .. W-6:
  W-1 std = Std(D, tr_base) (base pairs only; within pairs never touch mu / sd / gmu / gsd / sy)
  W-2 per epoch n_w = r x len(tr_base) within pairs drawn without replacement with inclusion probability
      pi_i = min(1, c w_i) (w_common.inclusion / draw; all when the pool is small), appended to the base pairs and passed
      to fu2 epoch_batches (order shuffle, (a, b) / (b, a) flips unchanged); y / std.sy
  W-3 the draws use rng_w = default_rng(seed + 7919) only; r = 0 never touches rng_w -> the fit's random streams
  W-4 per epoch on the inner-validation pairs: V_mae (= FU2 rec['val']) and V_nl (spec 5); early stopping / best epoch
      on J = select ('nl' | 'mae')
  W-5 loss unchanged (fu2 loss_fn), W-6 everything else unchanged (AdamW 1e-3 / 1e-4, cosine T_max 300, dropout 0.1,
      batch 64, patience 30, max 300 epochs).
Imported unchanged: fu2 train Std, batch (through predict / epoch_batches), orient (in epoch_batches), loss_fn, predict,
gate_m1, seed_all, epoch_batches, models.build; train_tf leak_check, kcal, std_dict, std_sha, save_torch, model_card,
PRED_COLS.
Selection (spec 8-1): 12 tuning fits (cells x r, select 'nl', inner_train + its within pairs, inner_val).
  lfW: eligible = V_mae <= 1.05 x min over the r = 0 cells of V_mae (each at its best-J epoch); min V_nl; ties smaller r,
       then grid order.   lfV: the r = 0 cells, min V_nl, ties grid order.
  final per fold: the selected (cell, r) on the outer training pairs (+ their within pairs), n_epochs = the tuning best
  epoch, seeds 20261019 + 0..4, mean. lfV = lfW selection -> one set of fits, the lfV .pt are hard links.
Gate C-W8 (= DEPLOY C-4) on every fit: 200 random test pairs (W2: training pairs) + 200 random within pairs of the fit
(r > 0); leak checks: train_tf.leak_check (C-5) + within pairs inside the fit's training reactions (C-W2).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402
import within as WI  # noqa: E402

C = W.C
import data_lf as DL  # noqa: E402

FT, Mo, F = DL.FT, DL.Mo, DL.F
import torch  # noqa: E402
import train_tf as TT  # noqa: E402

CODE_VERSION = "deploy-w-train-1"
MODEL, VARIANT, ARM = "TF", "A+G", "S"
GRID = list(Mo.GRID[MODEL])
TARGETS = list(F.TARGETS)
PRED_COLS = list(TT.PRED_COLS)
W_FILES = ("train_w.py", "within.py", "w_common.py")
# hashed once at import (fu2 review F3 rule): the code this process runs
CODE_SHA = {**{f"w/{f}": C.sha256(HERE / f) for f in W_FILES}, **TT.CODE_SHA}
TRAINER_FILES = tuple(f"w/{f}" for f in W_FILES) + tuple(TT.TRAINER_FILES)
W_CHECK_N = 1000
EST_MAX_H = 20.0
EST_EPOCHS = dict(typical_tune=100, typical_final=70, upper=F.MAX_EPOCHS)
_GATE: dict = {}
DEV = FT.DEV


# ================================================================ fit_w (fu2/train.py fit + W-1 .. W-6)
def fit_w(D, model_name, cell, variant, tr, val=None, seed=F.SEED, n_epochs=F.MAX_EPOCHS, patience=F.PATIENCE,
          dropout=F.DROPOUT, batch_size=F.BATCH, log=None, r=0, within=None, select="mae", vgroups=None):
    """one training run -> dict(model, std, sy, T, history, best_epoch, best_val, best_val_per_target, seconds,
    best_val_mae, best_val_nl, select, r, n_within_pool). within = pool of this fit's base pairs (within.load_pool /
    within.pool: ra, rb, y, w); vgroups = V_nl groups of `val` (within.val_groups)."""
    if select not in ("mae", "nl"):
        W.stop(f"select {select}")
    if r and (within is None or not len(within["w"])):
        W.stop(f"r = {r} without a within-pair pool")
    FT.seed_all(seed)
    rng = np.random.default_rng(seed)
    std = FT.Std(D, tr)                                                           # W-1: base pairs only
    T = std.tensors(D)
    sy = torch.tensor(std.sy, dtype=torch.float32, device=DEV)
    ytr = torch.tensor(D.y[tr] / std.sy, dtype=torch.float32, device=DEV)
    model = Mo.build(model_name, cell, D.d_in, len(F.TARGETS), D.d_glob, variant, dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=F.LR, weight_decay=F.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=F.MAX_EPOCHS)
    ra_all, rb_all = D.ra[tr], D.rb[tr]
    if r:                                                                         # W-2 / W-3
        rng_w = np.random.default_rng(seed + W.RNG_W_OFFSET)
        pi = W.inclusion(within["w"], int(r) * len(tr))
        ra_w, rb_w = np.asarray(within["ra"]), np.asarray(within["rb"])
        yw = torch.tensor(np.asarray(within["y"]) / std.sy, dtype=torch.float32, device=DEV)
    hist, best = [], dict(val=np.inf, epoch=0, state=None, per_target=None, val_mae=np.nan, val_nl=np.nan)
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        if r:
            sel = W.draw(rng_w, pi)
            ra_ep, rb_ep = np.r_[ra_all, ra_w[sel]], np.r_[rb_all, rb_w[sel]]
            y_ep = torch.cat([ytr, yw[torch.as_tensor(sel, device=DEV)]])
        else:
            sel, ra_ep, rb_ep, y_ep = (), ra_all, rb_all, ytr
        tl = []
        for ra, rb, y in FT.epoch_batches(rng, ra_ep, rb_ep, y_ep, batch_size):
            R, ia, ib = FT.batch(T, ra, rb)
            loss, l1, pen = FT.loss_fn(model(R, ia, ib), y, sy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append((float(loss), float(l1), float(pen)))
        sched.step()
        rec = dict(epoch=ep, loss=float(np.mean([x[0] for x in tl])), l1=float(np.mean([x[1] for x in tl])),
                   pen=float(np.mean([x[2] for x in tl])), t=time.time() - t0)
        if not all(np.isfinite([x for t3 in tl for x in t3])):
            W.stop(f"training loss not finite at epoch {ep} ({model_name} {cell} {variant} r={r})")
        if val is not None:
            yh = FT.predict(model, T, D.ra[val], D.rb[val])
            mae_t = np.abs(yh - D.y[val] / std.sy).mean(0)
            rec["val"] = float(mae_t[F.I_PRIM].mean())                              # V_mae = FU2 rec['val']
            rec["val_mae"] = rec["val"]
            if not np.isfinite(rec["val"]):
                W.stop(f"validation MAE not finite at epoch {ep} ({model_name} {cell} {variant} r={r})")
            rec["val_nl"] = W.vnl(yh * std.sy - D.y[val], vgroups, std.sy) if vgroups else float("nan")
            J = rec["val"] if select == "mae" else rec["val_nl"]
            if not np.isfinite(J):
                W.stop(f"validation J ({select}) not finite at epoch {ep} ({model_name} {cell} r={r}; "
                       f"{len(vgroups or [])} V_nl groups)")
            if J < best["val"]:
                best = dict(val=J, epoch=ep, state=copy.deepcopy(model.state_dict()), per_target=mae_t.tolist(),
                            val_mae=rec["val"], val_nl=rec["val_nl"])
        rec["n_within"] = int(len(sel))
        hist.append(rec)
        if log and (ep % log == 0 or ep == 1):
            print(f"    ep {ep} {rec}", flush=True)
        if val is not None and ep - best["epoch"] >= patience:
            break
    if val is not None and (best["state"] is None or best["epoch"] < 1):
        W.stop(f"no finite best epoch ({model_name} {cell} {variant} r={r})")
    if val is not None:
        model.load_state_dict(best["state"])
    return dict(model=model, std=std, sy=std.sy, T=T, history=hist,
                best_epoch=best["epoch"] if val is not None else n_epochs, best_val=best["val"],
                best_val_per_target=best["per_target"], seconds=time.time() - t0, best_val_mae=best["val_mae"],
                best_val_nl=best["val_nl"], select=select, r=int(r),
                n_within_pool=int(len(within["w"])) if within is not None else 0)


# ================================================================ gates / checks
class PairView:
    """the data object with other pair arrays (within pairs) for fu2 gate_m1 (reads D.ra / D.rb of the given rows)."""

    def __init__(self, D, ra, rb):
        self._D, self.ra, self.rb = D, np.asarray(ra), np.asarray(rb)

    def __getattr__(self, k):
        return getattr(self._D, k)


def gate_within(model, D, std, pool, seed):
    """C-W8 on 200 random within pairs of the fit's pool (fu2 gate_m1 through PairView)."""
    if pool is None or not len(pool["ra"]):
        return None
    return FT.gate_m1(model, PairView(D, pool["ra"], pool["rb"]), std, np.arange(len(pool["ra"])), seed=seed)


def within_leak(D, std, pool, other_rows):
    """C-W2 at run time: every within reaction is a training (scaling) reaction of the fit; none is a reaction of the
    validation / test lists."""
    if pool is None or not len(pool["ra"]):
        return dict(n=0, ok=True)
    wr = set(np.asarray(pool["ra"]).tolist()) | set(np.asarray(pool["rb"]).tolist())
    if not wr <= set(np.asarray(std.rxn_rows).tolist()):
        W.stop("C-W2: a within-pair reaction is not a training reaction of the fit")
    for o in other_rows:
        o = np.asarray(o, dtype=int)
        if len(o) and wr & (set(D.ra[o].tolist()) | set(D.rb[o].tolist())):
            W.stop("C-W2: a within-pair reaction is a validation / test reaction")
    return dict(n=int(len(pool["ra"])), n_rxn=len(wr), ok=True)


def gates_ok(rec):
    fits = list(rec.get("tune") or []) + list(rec.get("final") or [])
    return bool(fits) and all(f["gate_M1"]["ok"] and (f.get("gate_M1_within") or {"ok": True})["ok"] for f in fits)


def gate_prereg():
    import prereg_w as PRW
    rec = PRW.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    p = W.RESW / "C_tests_w.json"
    c = W.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        W.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K / W2)")
    rec = c.get("trainer_code_sha256") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA.get(f)]
    if diff:
        W.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun tests_w.py")
    return dict(c_tests_sha256=W.sha256(p))


def _check_root(root, smoke):
    r = W.assert_write_ok(Path(root) / "probe").parent
    for prod in (W.UNITS_W, W.MODEL_W):
        p = Path(os.path.abspath(prod))
        if r == p or p in r.parents:
            if smoke:
                W.stop(f"smoke / test runs never write {p}")
            if not _GATE.get("prereg"):
                W.stop(f"writing {p} needs the PREREG_W gate (train_w.py run / final)")
    return r


# ================================================================ names / fingerprint
def uname(rung, k):
    return f"D__{rung}__{ARM}__{MODEL}__f{k}"


def final_name(j):
    return f"D__lfW__{ARM}__{MODEL}__all__s{j}"


def tune_plan(cells=None, r_grid=W.R_GRID):
    gis = list(range(len(GRID))) if cells is None else [int(c) for c in cells]
    return [(gi, GRID[gi], int(r)) for gi in gis for r in r_grid]


def rules():
    return dict(r_grid=list(W.R_GRID), mae_guard=W.MAE_GUARD, min_group_val=W.MIN_GROUP_VAL, vnl_pos=list(W.VNL_POS),
                select="nl", rng_w_offset=W.RNG_W_OFFSET, sampling="pi_i = min(1, c w_i), systematic, w = 2/(n_g-1)",
                lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                patience=F.PATIENCE, seeds=F.FINAL_SEEDS, tune_seed=F.SEED, antisym_tol=F.ANTISYM_TOL,
                antisym_n=F.ANTISYM_N, val="first inner fold (FU2 split files)")


def fingerprint(D, k, sp, pools, smoke=False, limit=None, plan=None):
    return TT.norm(dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, data=WI.data_fingerprint(D), split_pairs=sp["pairs_sha"],
        within={k2: (v.get("sha256") if v is not None else None) for k2, v in pools.items()},
        prereg=None if smoke else _GATE.get("prereg"), grid=GRID, plan=plan, limit=limit, smoke=bool(smoke),
        rules=rules(), fold=k, **FT.versions()))


def _limit(sp, limit):
    """developer runs only: the training lists cut (inner_val and test kept whole for V_nl and evaluation)."""
    if not limit:
        return sp
    sp = dict(sp)
    for key in ("train", "inner_train"):
        sp[key] = np.sort(sp[key])[:int(limit)]
    return sp


def pools_for(D, k, sp, limit, within_root):
    if limit:                                    # developer runs: pools rebuilt in memory on the cut lists
        S = W.load_s_folds()
        _, gid = W.group_table(S)
        out = {}
        for key in ("inner_train", "train"):
            arr, _ = WI.pool(D, sp[key], gid)
            arr["sha256"] = None
            out[key] = arr
        return out
    return {key: WI.load_pool(D, k, key, root=within_root) for key in ("inner_train", "train")}


def kcal(res, D, rows):
    return TT.kcal(res, D, rows)


def payload(D, res, cell, u, rung, r, sel_rec):
    p = TT.pt_payload(D, res, cell, u)
    p.update(rung=rung, r=int(r), select="nl", selection=sel_rec, code=CODE_VERSION)
    return p


# ================================================================ one fold (lfW + lfV)
def run_unit_w(D, k, root=None, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, r_grid=W.R_GRID, log=50,
               smoke=False, limit=None, within_root=None):
    root = _check_root(root or W.UNITS_W, smoke or bool(limit))
    root.mkdir(parents=True, exist_ok=True)
    names = {rg: uname(rg, k) for rg in W.RUNGS}
    sp = _limit(D.split(k), limit)
    tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
    if min(len(tr), len(te), len(itr), len(iva)) == 0:
        W.stop(f"fold {k}: empty split list")
    pools = pools_for(D, k, sp, limit, within_root)
    plan = tune_plan(cells, r_grid)
    fp = fingerprint(D, k, sp, pools, smoke, limit, [[gi, rr] for gi, _, rr in plan])
    js = {rg: root / f"{names[rg]}.json" for rg in W.RUNGS}
    have = [rg for rg in W.RUNGS if js[rg].is_file()]
    if have:
        olds = {rg: W.read_json(js[rg]) for rg in have}
        bad = [rg for rg, o in olds.items() if (o.get("fingerprint") or {}) != fp]
        if bad:
            o = olds[bad[0]].get("fingerprint") or {}
            W.stop(f"cache {js[bad[0]]} has a different fingerprint: keys "
                   f"{sorted(x for x in set(fp) | set(o) if o.get(x) != fp.get(x))}")
        if len(have) == len(W.RUNGS):
            print(f"  fold {k}: cached", flush=True)
            return olds
    t_unit = time.time()
    S = W.load_s_folds()
    vg = WI.val_groups(D, iva, S)
    vg_info = dict(n_groups=len(vg), n_pairs=int(sum(len(g["idx"]) for g in vg)),
                   by_pos=dict(Counter(g["key"][2] for g in vg)))
    print(f"  fold {k}: {len(tr)} train / {len(itr)} inner_train / {len(iva)} inner_val / {len(te)} test pairs; "
          f"within pools inner {len(pools['inner_train']['w'])}, outer {len(pools['train']['w'])}; V_nl groups {vg_info}",
          flush=True)
    tune = []
    for ti, (gi, cell, r) in enumerate(plan):
        res = fit_w(D, MODEL, cell, VARIANT, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log, r=r,
                    within=pools["inner_train"], select="nl", vgroups=vg)
        lk = TT.leak_check(D, res["std"], itr, (iva, te), te)
        wl = within_leak(D, res["std"], pools["inner_train"] if r else None, (iva, te))
        g = FT.gate_m1(res["model"], D, res["std"], te, seed=F.SEED + ti)
        gw = gate_within(res["model"], D, res["std"], pools["inner_train"], seed=F.SEED + ti) if r else None
        vk = np.abs(kcal(res, D, iva) - D.y[iva]).mean(0)
        h = res["history"]
        tune.append(dict(cell=cell, grid_index=gi, r=r, best_epoch=res["best_epoch"], J=res["best_val"],
                         val_mae=res["best_val_mae"], val_nl=res["best_val_nl"], val_per_target=res["best_val_per_target"],
                         val_kcal_mae=dict(zip(TARGETS, vk.tolist())), epochs_run=len(h), seconds=res["seconds"],
                         epoch_seconds=res["seconds"] / max(1, len(h)), n_within_pool=res["n_within_pool"],
                         n_within_epoch=float(np.mean([x["n_within"] for x in h])) if h else 0.0, gate_M1=g,
                         gate_M1_within=gw, leak=lk, within_leak=wl, sy=res["sy"].tolist(),
                         history=[dict(epoch=x["epoch"], loss=x["loss"], val_mae=x.get("val"), val_nl=x.get("val_nl"),
                                       n_within=x["n_within"]) for x in h]))
        print(f"  fold {k} cell {cell} r={r}: J(V_nl) {res['best_val']:.4f} V_mae {res['best_val_mae']:.4f} at epoch "
              f"{res['best_epoch']} ({res['seconds']:.0f} s) gate {g['ok']}/{(gw or {}).get('ok')}", flush=True)
        if not g["ok"] or (gw is not None and not gw["ok"]):
            W.stop(f"gate C-W8 failed (fold {k}, tuning cell {cell} r={r}): {g} / {gw}")
        del res
    sel = select(tune)
    finals_by_key, preds_by_key = {}, {}
    t_rows = te
    for rg in W.RUNGS:
        ti = sel[rg]["index"]
        key = (tune[ti]["grid_index"], tune[ti]["r"], tune[ti]["best_epoch"])
        if key in finals_by_key:
            continue
        cell, r, ep = tune[ti]["cell"], tune[ti]["r"], tune[ti]["best_epoch"]
        preds, finals = [], []
        for s in seeds:
            res = fit_w(D, MODEL, cell, VARIANT, tr, None, seed=s, n_epochs=ep, log=log, r=r, within=pools["train"])
            lk = TT.leak_check(D, res["std"], tr, (te,), te)
            wl = within_leak(D, res["std"], pools["train"] if r else None, (te,))
            g = FT.gate_m1(res["model"], D, res["std"], te, seed=s)
            gw = gate_within(res["model"], D, res["std"], pools["train"], seed=s) if r else None
            if not g["ok"] or (gw is not None and not gw["ok"]):
                W.stop(f"gate C-W8 failed (fold {k}, final {rg} seed {s}): {g} / {gw}")
            preds.append(kcal(res, D, t_rows))
            u = dict(rung=rg, arm=ARM, model=MODEL, fold=k)
            pt = root / f"{names[rg]}__s{s - F.SEED}.pt"
            TT.save_torch(W.assert_write_ok(pt), payload(D, res, cell, u, rg, r, sel[rg]))
            finals.append(dict(seed=s, epochs=ep, r=r, cell=cell, seconds=res["seconds"], gate_M1=g, gate_M1_within=gw,
                               leak=lk, within_leak=wl, sy=res["sy"].tolist(),
                               std_sha256=TT.std_sha(TT.std_dict(res["std"], res["sy"])),
                               n_within_epoch=float(np.mean([x["n_within"] for x in res["history"]])),
                               history_tail=res["history"][-3:], pt=pt.name))
            del res
        finals_by_key[key] = (rg, finals)
        preds_by_key[key] = np.stack(preds)
    recs = {}
    for rg in W.RUNGS:
        ti = sel[rg]["index"]
        key = (tune[ti]["grid_index"], tune[ti]["r"], tune[ti]["best_epoch"])
        owner, finals = finals_by_key[key]
        if owner != rg:                                        # lfV = lfW selection: hard links to the same .pt
            for j in range(len(seeds)):
                src, dst = root / f"{names[owner]}__s{j}.pt", root / f"{names[rg]}__s{j}.pt"
                W.assert_write_ok(dst)
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst)
        Y = preds_by_key[key]
        pid = D.P.index.to_numpy()[t_rows].astype(str)
        rows = [pd.DataFrame(dict(pair_id=pid, ref_kind=D.P.ref_kind.to_numpy()[t_rows], pos=D.P.pos.to_numpy()[t_rows],
                                  fold=int(k), rung=rg, arm=ARM, model=MODEL, target=t, y=D.y[t_rows, j],
                                  yhat=Y[:, :, j].mean(0), base=0.0, beta=0.0, yhat_seed_sd=Y[:, :, j].std(0)))
                for j, t in enumerate(TARGETS)]
        pred = pd.concat(rows, ignore_index=True)[PRED_COLS]
        pq = root / f"{names[rg]}.parquet"
        W.write_parquet(pq, pred)
        recs[rg] = dict(unit=dict(rung=rg, arm=ARM, model=MODEL, fold=k), name=names[rg], fingerprint=fp,
                        smoke=bool(smoke), limit=limit, variant=VARIANT, tune=tune, selection=sel[rg], selections=sel,
                        chosen=dict(cell=tune[ti]["cell"], r=tune[ti]["r"], best_epoch=tune[ti]["best_epoch"],
                                    index=ti, grid_index=tune[ti]["grid_index"]),
                        final=finals, pt_files=[f"{names[owner]}__s{j}.pt" for j in range(len(seeds))],
                        pt_shared_with=None if owner == rg else names[owner],
                        val_kcal_mae=tune[ti]["val_kcal_mae"], val_nl=tune[ti]["val_nl"], val_mae=tune[ti]["val_mae"],
                        vnl_groups=vg_info, val_pairs_sha256=sp["val_sha"], test_pairs_sha256=sp["test_sha"],
                        n_val=len(iva), n_train=len(tr), n_test=len(te), n_inner_train=len(itr), n_inner_val=len(iva),
                        n_within_pool=dict(inner_train=int(len(pools["inner_train"]["w"])),
                                           train=int(len(pools["train"]["w"]))),
                        within_sha256={kk: v.get("sha256") for kk, v in pools.items()}, dropped=sp["dropped"],
                        parquet_sha256=W.sha256(pq), inputs=D.inputs, usable_rule=D.usable_rule,
                        wall_seconds=time.time() - t_unit, host=socket.gethostname(),
                        slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    for rg in ("lfV", "lfW"):                                  # lfW json last (the cache needs both)
        W.write_json(js[rg], recs[rg])
    return recs


def select(tune):
    """spec 8-1 selections -> {lfW: {...}, lfV: {...}} (index into tune + the rule numbers)."""
    r0 = [i for i, t in enumerate(tune) if t["r"] == 0]
    if not r0:
        W.stop("selection needs the r = 0 cells (guard / lfV)")
    for t in tune:
        if not (isinstance(t["best_epoch"], int) and t["best_epoch"] >= 1 and np.isfinite(t["val_nl"])
                and np.isfinite(t["val_mae"])):
            W.stop(f"tuning fit {t['cell']} r={t['r']}: best_epoch {t['best_epoch']} V_nl {t['val_nl']} V_mae {t['val_mae']}")
    mae0 = min(tune[i]["val_mae"] for i in r0)
    guard = W.MAE_GUARD * mae0
    elig = [i for i, t in enumerate(tune) if t["val_mae"] <= guard]
    iw = min(elig, key=lambda i: (tune[i]["val_nl"], tune[i]["r"], tune[i]["grid_index"]))
    iv = min(r0, key=lambda i: (tune[i]["val_nl"], tune[i]["grid_index"]))
    return dict(lfW=dict(index=iw, rule="V_mae <= 1.05 x min_{r=0} V_mae; min V_nl; ties smaller r, grid order",
                         min_r0_val_mae=mae0, guard=guard, eligible=elig),
                lfV=dict(index=iv, rule="r = 0 cells: min V_nl; ties grid order"))


# ================================================================ commands
def cmd_smoke():
    """C-W9 (spec 6) + K_estimate.json (spec 8-3)."""
    FT.require_gpu()
    _check_root(W.SMOKE_W, True)
    if W.SMOKE_W.is_dir():
        for p in W.SMOKE_W.glob("D__*"):
            W.assert_write_ok(p)
            p.unlink()
    D = DL.get_data("lf")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cells = [0, len(GRID) - 1]
    recs = run_unit_w(D, 0, root=W.SMOKE_W, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=cells, r_grid=(0, 4), log=1,
                      smoke=True)
    rec = recs["lfW"]
    ok = gates_ok(recs["lfW"]) and gates_ok(recs["lfV"])
    ept = {f"{t['grid_index']}_r{t['r']}": t["epoch_seconds"] for t in rec["tune"]}
    fin = {f"{f['cell']['n_layers']}x{f['cell']['width']}_r{f['r']}": f["seconds"] / max(1, f["epochs"])
           for rg in W.RUNGS for f in recs[rg]["final"]}
    est = estimate(rec, ept)
    out = dict(check="C-W9", ok=bool(ok and est["ok"]), versions=FT.versions(), device=str(FT.DEV), unit=rec["name"],
               seconds=time.time() - t0, tune_epoch_seconds=ept, final_epoch_seconds=fin,
               gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
               gate_CW8_tune=[(t["gate_M1"], t["gate_M1_within"]) for t in rec["tune"]],
               gate_CW8_final=[(f["gate_M1"], f["gate_M1_within"]) for rg in W.RUNGS for f in recs[rg]["final"]],
               gates_ok=bool(ok), n_train=rec["n_train"], n_inner_train=rec["n_inner_train"], n_test=rec["n_test"],
               n_within_pool=rec["n_within_pool"], vnl_groups=rec["vnl_groups"], selections=rec["selections"],
               estimate=est, code_sha256=CODE_SHA)
    W.write_json(W.RESW / "C9_smoke_w.json", out)
    W.write_json(W.RESW / "K_estimate.json", est)
    print(json.dumps({k: out[k] for k in ("ok", "seconds", "tune_epoch_seconds", "gpu_mem_peak_gb")}, indent=1))
    print(json.dumps({k: est[k] for k in ("per_fold_hours_typical", "per_fold_hours_upper", "total_gpu_hours_typical",
                                          "ok")}, indent=1))
    if not est["ok"]:
        print(f"STOP: K estimate {est['per_fold_hours_typical']:.1f} h per fold > {EST_MAX_H} h (spec 8-3)",
              file=sys.stderr)
        return 3
    return 0 if ok else 3


def estimate(rec, ept):
    """spec 8-3: per-fold GPU hours from the smoke per-epoch times (inner_train fits, cells 0 and 3, r 0 and 4). Cells 1, 2
    = mean of 0 and 3 (FU2 smoke rule); r = 1 interpolated linearly in r; outer fits x n_train / n_inner_train.
    typical: 100 tuning epochs, 70 final epochs; upper: 300 everywhere, finals on the slowest (cell 3, r 4)."""
    def t(gi, r):
        if f"{gi}_r{r}" in ept:
            return ept[f"{gi}_r{r}"]
        if r not in (0, 4):
            return t(gi, 0) + (t(gi, 4) - t(gi, 0)) * r / 4.0
        return 0.5 * (t(0, r) + t(len(GRID) - 1, r))
    scale = rec["n_train"] / max(1, rec["n_inner_train"])
    tune_typ = sum(t(gi, r) for gi in range(len(GRID)) for r in W.R_GRID) * EST_EPOCHS["typical_tune"]
    tune_up = sum(t(gi, r) for gi in range(len(GRID)) for r in W.R_GRID) * EST_EPOCHS["upper"]
    slow = t(len(GRID) - 1, max(W.R_GRID)) * scale
    mid_w = np.mean([t(gi, r) for gi in range(len(GRID)) for r in W.R_GRID]) * scale
    mid_v = np.mean([t(gi, 0) for gi in range(len(GRID))]) * scale
    fin_typ = 5 * (mid_w + mid_v) * EST_EPOCHS["typical_final"]
    fin_up = 5 * 2 * slow * EST_EPOCHS["upper"]
    h_typ, h_up = (tune_typ + fin_typ) / 3600, (tune_up + fin_up) / 3600
    return dict(per_fold_hours_typical=h_typ, per_fold_hours_upper=h_up, total_gpu_hours_typical=5 * h_typ,
                total_gpu_hours_upper=5 * h_up, epoch_seconds=ept, outer_scale=scale, epochs=EST_EPOCHS,
                limit_hours_per_fold=EST_MAX_H, ok=bool(h_typ <= EST_MAX_H),
                rule="typical: 12 tuning fits x 100 epochs + 5 seeds x (lfW + lfV) x 70 epochs on the outer pairs; "
                     "upper: 300 epochs everywhere, finals on cell 3 / r 4")


def cmd_run(i, n):
    FT.require_gpu()
    i = C.array_task(i)
    n = int(n)
    if not (0 <= i < n):
        print(f"usage: run <i|@I> <n> with 0 <= i < n (got {i}, {n})", file=sys.stderr)
        return 2
    folds = list(range(W.NF))[i::n]
    if not folds:
        return 0
    pre = gate_prereg()
    gc = gate_c_tests()
    print(f"[K-W] PREREG_W ok (git {pre.get('git_head')}, md {_GATE['prereg'][:12]}), C_tests_w ok "
          f"({gc['c_tests_sha256'][:12]}), {FT.versions()}, folds {folds}", flush=True)
    D = DL.get_data("lf")
    for k in folds:
        t0 = time.time()
        recs = run_unit_w(D, k, root=W.UNITS_W)
        print(f"[K-W] fold {k} done ({time.time() - t0:.0f} s) lfW {recs['lfW']['chosen']} lfV {recs['lfV']['chosen']}",
              flush=True)
    return 0


def load_units(root=None, allow_limited=False):
    """the 10 unit records (lfW, lfV x 5 folds): json / parquet / 5 .pt present, parquet sha256 = json, gates ok, no smoke
    -> ({name: rec}, missing names)."""
    root = W.UNITS_W if root is None else Path(root)
    names = [uname(rg, k) for rg in W.RUNGS for k in range(W.NF)]
    missing = [x for x in names if not ((root / f"{x}.json").is_file() and (root / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: W.read_json(root / f"{x}.json") for x in names}
    for x, r in recs.items():
        if W.sha256(root / f"{x}.parquet") != r.get("parquet_sha256"):
            W.stop(f"{x}.parquet changed after its json")
        if r.get("smoke") or (r.get("limit") and not allow_limited) or not gates_ok(r):
            W.stop(f"gate C-W8 failed (or smoke / limited unit) in {x}")
        for j in range(len(F.FINAL_SEEDS)):
            if not (root / f"{x}__s{j}.pt").is_file():
                W.stop(f"{x}__s{j}.pt missing")
    return recs, []


def cmd_collect(source="lf", root=None, allow_limited=False):
    D = DL.get_data(source)
    root = W.UNITS_W if root is None else Path(root)
    recs, missing = load_units(root, allow_limited)
    if missing:
        print(f"GATE: {len(missing)} units missing: {missing}", file=sys.stderr)
        return 2
    import prereg_w as PRW
    _GATE["prereg"] = W.sha256(PRW.MD) if PRW.MD.is_file() else None
    for x, r in recs.items():
        if r["unit"]["rung"] != "lfW":
            continue
        k = int(r["unit"]["fold"])
        sp = _limit(D.split(k), r.get("limit"))
        pools = pools_for(D, k, sp, r.get("limit"), None)
        plan = r["fingerprint"].get("plan")
        now = fingerprint(D, k, sp, pools, False, r.get("limit"), plan)
        strip = lambda f: {kk: v for kk, v in f.items() if kk != "gpu"}       # noqa: E731
        if strip(r["fingerprint"]) != strip(now):
            W.stop(f"{x} was computed with other code / inputs: keys "
                   f"{sorted(kk for kk in set(now) | set(r['fingerprint']) if kk != 'gpu' and r['fingerprint'].get(kk) != now.get(kk))}")
    pred = pd.concat([pd.read_parquet(root / f"{x}.parquet")[PRED_COLS] for x in recs], ignore_index=True)
    if pred.duplicated(["rung", "pair_id", "target"]).any():
        W.stop("duplicated prediction rows")
    out = W.RESW / "K_predictions_w.parquet"
    W.write_parquet(out, pred)
    vrows = [dict(rung=r["unit"]["rung"], fold=int(r["unit"]["fold"]), target=t, val_kcal_mae=float(r["val_kcal_mae"][t]),
                  val_nl=r["val_nl"], val_mae=r["val_mae"], n_val=r["n_val"], n_vnl_groups=r["vnl_groups"]["n_groups"],
                  cell=json.dumps(r["chosen"]["cell"], sort_keys=True), r=r["chosen"]["r"],
                  best_epoch=r["chosen"]["best_epoch"]) for x, r in recs.items() for t in TARGETS]
    vout = W.RESW / "K_val_w.csv"
    W.write_csv(vout, pd.DataFrame(vrows))
    gpu_s = sum(r.get("wall_seconds") or 0 for r in recs.values() if r["unit"]["rung"] == "lfW")
    rep = dict(n_units=len(recs), code=CODE_VERSION, code_sha256=CODE_SHA, prereg_w_md_sha256=_GATE["prereg"],
               inputs_sha256=D.sha, pair_list_sha=D.pair_list_sha,
               units={x: dict(chosen=r["chosen"], selection=r["selection"], pt_shared_with=r.get("pt_shared_with"),
                              n_train=r["n_train"], n_test=r["n_test"], n_within_pool=r["n_within_pool"],
                              vnl_groups=r["vnl_groups"], val_nl=r["val_nl"], val_mae=r["val_mae"],
                              tune=[dict(cell=t["cell"], r=t["r"], val_mae=t["val_mae"], val_nl=t["val_nl"],
                                         best_epoch=t["best_epoch"], seconds=t["seconds"],
                                         n_within_epoch=t["n_within_epoch"]) for t in r["tune"]],
                              gpu=(r.get("fingerprint") or {}).get("gpu"), wall_seconds=r.get("wall_seconds"),
                              gate_anti_max=max([t["gate_M1"]["anti_max"] for t in r["tune"]]
                                                + [f["gate_M1"]["anti_max"] for f in r["final"]]),
                              gate_within_anti_max=max([(t.get("gate_M1_within") or {}).get("anti_max", 0.0)
                                                        for t in r["tune"]] +
                                                       [(f.get("gate_M1_within") or {}).get("anti_max", 0.0)
                                                        for f in r["final"]]))
                      for x, r in recs.items()},
               gpu_hours=float(gpu_s / 3600), n_test_pairs=int(pred.pair_id.nunique()), predictions=str(out),
               predictions_sha256=W.sha256(out), val_csv=str(vout), val_csv_sha256=W.sha256(vout))
    W.write_json(W.RESW / "K_report_w.json", rep)
    print(f"[collect-W] {len(recs)} units -> {out.name} ({len(pred)} rows)", flush=True)
    return 0


# ---------------------------------------------------------------- Phase W2
def choose_final(recs):
    """spec 12: r = mode of the 5 lfW selected r (ties smaller r); cell = mode over the folds with that r (ties: lower mean
    tuning V_nl of the cell at that r over those folds, then grid order); n_epochs = lower median of those folds' best
    epochs (1-based fit epochs; the fold finals ran n_epochs = best_epoch)."""
    w = [r for r in recs.values() if r["unit"]["rung"] == "lfW"]
    if len(w) != W.NF:
        W.stop(f"{len(w)} lfW units, not {W.NF}")
    rc = Counter(int(r["chosen"]["r"]) for r in w)
    top = max(rc.values())
    r_mode = min(x for x in rc if rc[x] == top)
    fr = [r for r in w if int(r["chosen"]["r"]) == r_mode]
    ck = lambda c: json.dumps(TT.norm(c), sort_keys=True)                 # noqa: E731
    cc = Counter(ck(r["chosen"]["cell"]) for r in fr)
    topc = max(cc.values())
    cand = [x for x in cc if cc[x] == topc]
    gidx = {ck(c): i for i, c in enumerate(GRID)}

    def mean_nl(key):
        v = [t["val_nl"] for r in fr for t in r["tune"] if ck(t["cell"]) == key and int(t["r"]) == r_mode]
        return float(np.mean(v)) if v else np.inf
    cell_key = min(cand, key=lambda x: (mean_nl(x), gidx[x]))
    eps = sorted(int(r["chosen"]["best_epoch"]) for r in fr)
    ep = eps[(len(eps) - 1) // 2]
    return dict(cell=json.loads(cell_key), r=r_mode, n_epochs=ep, r_counts=dict(rc), cell_counts=dict(cc),
                folds_with_mode_r=[int(r["unit"]["fold"]) for r in fr], best_epochs=eps,
                tie_break=dict(candidates=cand, mean_tuning_val_nl={x: mean_nl(x) for x in cand}),
                rule="r = mode of the 5 lfW r (ties smaller r); cell = mode over those folds (ties lower mean tuning V_nl "
                     "at that r, then grid order); n_epochs = lower median of their best epochs (1-based)")


def gate_decision():
    p = W.RESW / "D_decision.json"
    if not p.is_file():
        W.not_ready(f"{p} missing (eval_w.py decide)")
    d = W.read_json(p)
    er = W.RESW / "E_report_w.json"
    if d.get("e_report_sha256") != (W.sha256(er) if er.is_file() else None):
        W.stop("D_decision.json was made from another E_report_w.json")
    return dict(d, _sha256=W.sha256(p))


def cmd_final():
    FT.require_gpu()
    pre = gate_prereg()
    gate_c_tests()
    dec = gate_decision()
    if dec.get("adopted") != "lfW":                       # spec 11: lf adopted -> no W2 (not an error)
        W.write_json(W.RESW / "W2_report.json", dict(phase="W2", skipped=True, adopted=dec.get("adopted"),
                                                     readout=dec.get("readout"), decision_sha256=dec["_sha256"],
                                                     reason="spec 11: the adopted model is lf; W2 is not run"))
        print(f"[W2] skipped: adopted {dec.get('adopted')} (readout {dec.get('readout')})", flush=True)
        return 0
    recs, missing = load_units()
    if missing:
        W.not_ready(f"K units missing: {missing}")
    sel = choose_final(recs)
    D = DL.get_data("lf")
    return train_final_w(D, sel, recs, dec, root=W.MODEL_W, prereg_head=pre.get("git_head"))


def train_final_w(D, sel, recs, dec, root, prereg_head=None, limit=None, n_epochs=None, check_n=W_CHECK_N, smoke=False,
                  within_root=None, seeds=F.FINAL_SEEDS):
    sp = D.split("all")
    tr = np.sort(sp["train"])[:int(limit)] if limit else sp["train"]
    ep = int(n_epochs) if n_epochs else int(sel["n_epochs"])
    root = _check_root(root, smoke or bool(limit) or bool(n_epochs))
    root.mkdir(parents=True, exist_ok=True)
    if limit:
        _, gid = W.group_table(W.load_s_folds())
        pool, _ = WI.pool(D, tr, gid)
        pool["sha256"] = None
    else:
        pool = WI.load_pool(D, "all", "train", root=within_root)
    r = int(sel["r"])
    wfp = TT.norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, data=WI.data_fingerprint(D),
                       train_pairs=DL.list_sha(D._ids(tr)), within=pool.get("sha256"),
                       selection=dict(cell=sel["cell"], r=r, n_epochs=ep),
                       k_units={x: rr["parquet_sha256"] for x, rr in recs.items()}, prereg=_GATE.get("prereg"),
                       decision=dec.get("_sha256"), rules=dict(seeds=list(seeds), check_n=check_n), limit=limit,
                       **FT.versions()))
    wj = root / "W2_unit.json"
    if wj.is_file():
        old = W.read_json(wj)
        if old.get("fingerprint") != wfp:
            W.stop(f"{wj} has a different fingerprint")
        print("[W2] cached", flush=True)
        return 0
    rng = np.random.default_rng(F.SEED)
    chk = np.sort(rng.choice(tr, size=min(check_n, len(tr)), replace=False))
    u = dict(rung="lfW", arm=ARM, model=MODEL, fold="all")
    finals, preds_chk, preds_tr, std0 = [], [], [], None
    t0 = time.time()
    for s in seeds:
        res = fit_w(D, MODEL, sel["cell"], VARIANT, tr, None, seed=s, n_epochs=ep, log=10, r=r, within=pool)
        lk = TT.leak_check(D, res["std"], tr, (), np.zeros(0, dtype=int))
        wl = within_leak(D, res["std"], pool if r else None, ())
        g = FT.gate_m1(res["model"], D, res["std"], tr, seed=s)                       # 200 random training pairs
        gw = gate_within(res["model"], D, res["std"], pool, seed=s) if r else None  # + 200 within pairs
        if not g["ok"] or (gw is not None and not gw["ok"]):
            W.stop(f"gate C-W8 failed (W2, seed {s}): {g} / {gw}")
        sd = TT.std_dict(res["std"], res["sy"])
        if std0 is None:
            std0 = sd
        elif TT.std_sha(sd) != TT.std_sha(std0):
            W.stop("W2 fits: standardisation differs between seeds")
        preds_chk.append(kcal(res, D, chk))
        preds_tr.append(kcal(res, D, tr))
        finals.append(dict(seed=s, epochs=ep, r=r, seconds=res["seconds"], gate_M1=g, gate_M1_within=gw, leak=lk,
                           within_leak=wl, std_sha256=TT.std_sha(sd), history_tail=res["history"][-3:]))
        TT.save_torch(W.assert_write_ok(root / f"{final_name(s - F.SEED)}.pt"),
                      payload(D, res, sel["cell"], u, "lfW", r, dict(sel, readout=dec.get("readout"))))
        print(f"[W2] seed {s}: {res['seconds']:.0f} s, gate {g} / {gw}", flush=True)
        del res
    Yc, Yt = np.stack(preds_chk), np.stack(preds_tr)
    pid_c = D.P.index.to_numpy()[chk].astype(str)
    W.write_csv(root / "W_check_pairs.csv", pd.DataFrame(dict(pair_id=pid_c, a_rxn=D.P.a_rxn.to_numpy()[chk],
                                                              b_rxn=D.P.b_rxn.to_numpy()[chk])))
    rows = [pd.DataFrame(dict(pair_id=pid_c, target=t, yhat=Yc[:, :, j].mean(0), yhat_sd=Yc[:, :, j].std(0),
                              **{f"yhat_s{q}": Yc[q, :, j] for q in range(len(Yc))})) for j, t in enumerate(TARGETS)]
    W.write_parquet(root / "W_check_pred.parquet", pd.concat(rows, ignore_index=True))
    train_mae = {t: float(np.abs(Yt[:, :, j].mean(0) - D.y[tr, j]).mean()) for j, t in enumerate(TARGETS)}
    erep = W.read_json(W.RESW / "E_report_w.json") if (W.RESW / "E_report_w.json").is_file() else {}
    adp = W.RESW / "E_ad_w.json"
    ad = W.read_json(adp) if adp.is_file() else {}
    card = TT.model_card(D, dict(cell=sel["cell"], n_epochs=ep), tr, finals,
                         dict(summary=erep.get("summary"), G_C=None), ad, DL.list_sha(D._ids(tr)), train_mae, std0)
    card.update(model="lfW . S . TF (FU2 M-2 TF, variant A+G; base pairs + within-group pairs, DEPLOY_MODEL_W)",
                rung="lfW", r=r, readout=dec.get("readout") or "direct", selection=sel,
                files=[f"{final_name(j)}.pt" for j in range(len(seeds))],
                within=dict(n_pairs=int(len(pool["w"])), sha256=pool.get("sha256"), file=pool.get("file"),
                            n_within_per_epoch=float(np.mean([f["history_tail"][-1].get("n_within", 0) for f in finals]))),
                graph=dict(group_id_required=(dec.get("readout") == "graph"), group_key="(b_rxn, fam, pos)",
                           max_group_size=W.GRAPH_MAX),
                cv_performance=dict(source="results/w/E_report_w.json (Phase K cross-validation; the final model has no "
                                           "test set)", summary=erep.get("summary"), A1=erep.get("A1"),
                                    A2=erep.get("A2"), grade=erep.get("grade"), secondary=erep.get("secondary")),
                decision=dict(file=str(W.RESW / "D_decision.json"), adopted=dec.get("adopted"),
                              readout=dec.get("readout"), grade=dec.get("grade")))
    card["code_sha256"].update({k: v for k, v in W.w_code_sha().items() if v})
    W.write_json(root / "MODEL_CARD_W.json", card)
    rep = dict(phase="W2", selection=sel, n_train_pairs=int(len(tr)), n_within=int(len(pool["w"])),
               dropped=sp["dropped"], finals=finals,
               gate_CW8_ok=all(f["gate_M1"]["ok"] and (f["gate_M1_within"] or {"ok": True})["ok"] for f in finals),
               train_mae_reference_only=train_mae,
               check_pairs=dict(n=int(len(chk)), file=str(root / "W_check_pairs.csv"),
                                pred=str(root / "W_check_pred.parquet")),
               model_card=str(root / "MODEL_CARD_W.json"), model_card_sha256=W.sha256(root / "MODEL_CARD_W.json"),
               pt_sha256={f"{final_name(j)}.pt": W.sha256(root / f"{final_name(j)}.pt") for j in range(len(seeds))},
               prereg_git_head=prereg_head, limit=limit, wall_seconds=time.time() - t0, versions=FT.versions(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    W.write_json((W.RESW / "W2_report.json") if not (smoke or limit or n_epochs) else root / "W2_report.json", rep)
    W.write_json(wj, dict(fingerprint=wfp, report=rep["model_card_sha256"]))
    print(json.dumps(dict(cell=sel["cell"], r=r, n_epochs=ep, n_train=len(tr), gate=rep["gate_CW8_ok"]), indent=1))
    return 0


def main(argv):
    a = argv[1:]
    if a[:1] == ["list"]:
        print("\n".join(uname(rg, k) for rg in W.RUNGS for k in range(W.NF)))
        return 0
    if a == ["smoke"]:
        return cmd_smoke()
    if a[:1] == ["run"] and len(a) == 3:
        return cmd_run(a[1], a[2])
    if a == ["collect"]:
        return cmd_collect()
    if a == ["final"]:
        return cmd_final()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
