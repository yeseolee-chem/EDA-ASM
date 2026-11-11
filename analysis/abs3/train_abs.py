#!/usr/bin/env python3
r"""train_abs.py — ABS3 TF · GNN absolute-value units (docs/specs/ABS3_MODELS.md 0-3, 0-4, 3, 5, 7). GPU work in env_d1fu2
(torch 2.2.1+cu118) through jobs/gpu_abs.slurm (RTX 3090, gpu1); collect on CPU (jobs/cpu_abs.slurm ABS3ENV=d1fu2).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python train_abs.py list                        unit names (A__lf3__S__<TF|GNN>__f<0..4>)
  python train_abs.py smoke                       C-8 (TF, GNN): fold 0, smallest + largest cell, 2 epochs, 1 seed, C-5 on
                                                  every fit -> $ABS3/smoke/nn (printed; tests_abs.py gpu records it)
  python train_abs.py overfit <TF|GNN>            C-O: 64 fold-0 training reactions, largest cell, 300 epochs, dropout 0,
                                                  1 seed (printed; tests_abs.py gpu records it)
  python train_abs.py run <TF|GNN> <i|@I> <n>     units [i::n] -> $ABS3/units/A__lf3__S__<model>__f<k>.{json,parquet} +
                                                  __s{0..4}.pt. Gates: PREREG (prereg_abs.check), results/C_tests.json
                                                  all_pass on this trainer code. Idempotent (fingerprint incl. GPU name;
                                                  a different fingerprint = STOP)
  python train_abs.py collect                     the 10 TF · GNN units -> $ABS3/collect/nn.parquet + nn_report.json
                                                  (xgb_abs.py collect merges them into results/KA_*)

Rules (spec 0-4) = the pair models' (DEPLOY 11-2 = FU2 M-1, ch3 3 mode) at the reaction level: fu2 models.GRID[model]
(TF 2/3 x 64/128, GNN 4/6 x 64/128), AdamW lr 1e-3 / wd 1e-4, cosine over 300 epochs (the final fit stops at the tuned
best epoch on the same schedule), dropout 0.1, batch 64 reactions (random order per epoch, no swap augmentation),
patience 30; tuning on the inner_train reactions, cell = lowest inner_val metric (mean of the 4 standardised MAE),
ties grid order; final = the outer training reactions, tuned best epoch, seeds 20261019 + 0..4, mean. Standardisation
from the fit's training reactions only (AbsStd); y_std = (y - mu) / s; loss = mean |yhat - y| (standardised) +
LAMBDA * mean |kc_barrier - (kc_strain + kc_inter + kc_solv)| / s_barrier with kc = mu + s * yhat (kcal).
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import abs_data as AD  # noqa: E402

DC3, DL, FT, Mo, F = AD.DC3, AD.DL, AD.FT, AD.Mo, AD.F
import abs_model as AM  # noqa: E402
import torch  # noqa: E402

CODE_VERSION = "abs3-nn-1"
VARIANT, ARM = "A+G", C.ARM
T3 = list(C.TARGETS3)
NF = C.NF
NN_MODELS = ("TF", "GNN")
OVERFIT_N, OVERFIT_EPOCHS, OVERFIT_FRAC = 64, 300, 0.10
CODE_FILES = C.NN_TRAINER_FILES
DEP_FILES = ("ch3/data_ch3.py", "ch3/ch3_common.py", "deploy/data_lf.py", "deploy/dep_common.py")
# hashed once at import: the code this process runs
CODE_SHA = {**{f: C.sha256(HERE / f) for f in CODE_FILES},
            **{f: C.sha256(C.REPO / "analysis" / f) for f in DEP_FILES},
            **{(k if k == "d1pair_common.py" else f"fu2/{k}"): v for k, v in FT.CODE_SHA.items()}}
TRAINER_FILES = CODE_FILES
SEED_COLS = [f"yhat_s{j}" for j in range(len(F.FINAL_SEEDS))]
UNIT_COLS = ["rxn_id", "fold", "model", "target", "y", "yhat", "yhat_seed_sd"] + SEED_COLS
_GATE: dict = {}
DEV = FT.DEV


def grid(model):
    return list(Mo.GRID[model])


def unit_list():
    return [dict(rung=C.RUNG, arm=ARM, model=m, fold=k) for m in NN_MODELS for k in range(NF)]


def uname(u):
    return C.uname(u["model"], u["fold"])


def mode_check():
    if DC3.current_mode() != DC3.MODE3:
        C.stop(f"fu2 target constants {DC3.current_mode()} != the 4-target values {DC3.MODE3}")


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


# ================================================================ fit (= fu2 train.fit at the reaction level)
def loss_abs(yhat, y, sy, ymu):
    """FU2 loss_fn form (spec 0-4): standardised L1 + LAMBDA * identity penalty on kc = mu + s * yhat (kcal)."""
    l1 = (yhat - y).abs().mean()
    kc = ymu + yhat * sy
    pen = ((kc[:, F.I_BAR] - kc[:, F.I_CH9].sum(1)).abs() / sy[F.I_BAR]).mean()
    return l1 + F.LAMBDA * pen, l1, pen


@torch.no_grad()
def predict_std(model, T, rows, bs=256):
    """standardised predictions [n, 4] of the reactions `rows` (token rows), eval mode."""
    model.eval()
    rows = np.asarray(rows, dtype=int)
    out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        R, ia, _ = FT.batch(T, r, r)
        out.append(model(R, ia).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(T3)))


def kcal(r, rows):
    """kcal/mol predictions = mu + s * standardised prediction."""
    return r["std"].ymu + predict_std(r["model"], r["T"], rows) * r["std"].sy


def fit_abs(A, model_name, cell, tr, val=None, seed=F.SEED, n_epochs=F.MAX_EPOCHS, patience=F.PATIENCE,
            dropout=F.DROPOUT, batch_size=F.BATCH, log=None):
    """one training run on the reactions `tr` (token rows) -> dict(model, std, T, history, best_epoch, best_val,
    best_val_per_target, seconds); `val` = validation reactions (early stopping / tuning), None = run n_epochs."""
    mode_check()
    tr = np.unique(np.asarray(tr, dtype=int))
    FT.seed_all(seed)
    rng = np.random.default_rng(seed)
    std = AD.AbsStd(A, tr)
    T = std.tensors(A.D)
    sy = torch.tensor(std.sy, dtype=torch.float32, device=DEV)
    ymu = torch.tensor(std.ymu, dtype=torch.float32, device=DEV)
    ytr = torch.tensor((A.y4[tr] - std.ymu) / std.sy, dtype=torch.float32, device=DEV)
    model = AM.build(model_name, cell, A.d_in, A.d_glob, VARIANT, dropout).to(DEV)
    opt = torch.optim.AdamW(model.used_parameters(), lr=F.LR, weight_decay=F.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=F.MAX_EPOCHS)
    n = len(tr)
    hist, best = [], dict(val=np.inf, epoch=0, state=None, per_target=None)
    yval = (A.y4[val] - std.ymu) / std.sy if val is not None else None
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        tl = []
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            rows = tr[b]
            R, ia, _ = FT.batch(T, rows, rows)
            loss, l1, pen = loss_abs(model(R, ia), ytr[torch.as_tensor(b, device=DEV)], sy, ymu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append((float(loss), float(l1), float(pen)))
        sched.step()
        rec = dict(epoch=ep, loss=float(np.mean([x[0] for x in tl])), l1=float(np.mean([x[1] for x in tl])),
                   pen=float(np.mean([x[2] for x in tl])), t=time.time() - t0)
        if not all(np.isfinite([x for t3 in tl for x in t3])):
            C.stop(f"training loss not finite at epoch {ep} ({model_name} {cell})")
        if val is not None:
            yh = predict_std(model, T, val)
            mae_t = np.abs(yh - yval).mean(0)
            rec["val"] = float(mae_t[F.I_PRIM].mean())
            if not np.isfinite(rec["val"]):
                C.stop(f"validation MAE not finite at epoch {ep} ({model_name} {cell})")
            if rec["val"] < best["val"]:
                best = dict(val=rec["val"], epoch=ep, state=copy.deepcopy(model.state_dict()), per_target=mae_t.tolist())
        hist.append(rec)
        if log and (ep % log == 0 or ep == 1):
            print(f"    ep {ep} {rec}", flush=True)
        if val is not None and ep - best["epoch"] >= patience:
            break
    if val is not None and (best["state"] is None or best["epoch"] < 1):
        C.stop(f"no finite best epoch ({model_name} {cell})")
    if val is not None:
        model.load_state_dict(best["state"])
    return dict(model=model, std=std, T=T, history=hist, best_epoch=best["epoch"] if val is not None else n_epochs,
                best_val=best["val"], best_val_per_target=best["per_target"], seconds=time.time() - t0, rows=tr)


def leak_check(std, fit_rows, others, test_rows):
    """C-5 (spec 5): the standardisation reactions = the fit's training reactions; no validation / test reaction among
    them (STOP otherwise)."""
    used = set(np.asarray(std.rxn_rows).tolist())
    if used != set(np.asarray(fit_rows).tolist()):
        C.stop("C-5: standardisation reactions differ from the fit's training reactions")
    for o in others:
        if used & set(np.asarray(o).tolist()):
            C.stop("C-5: a validation / test reaction is among the fit's training / standardisation reactions")
    if used & set(np.asarray(test_rows).tolist()):
        C.stop("C-5: a test reaction is among the fit's training reactions")
    return dict(n_fit_rxn=len(used), n_test_rxn_overlap=0, ok=True)


def out_width(r, rows):
    rows = np.asarray(rows, dtype=int)[:2]
    return int(predict_std(r["model"], r["T"], rows).shape[1])


def pt_payload(A, r, cell, u, model):
    return dict(state=r["model"].state_dict(), cell=cell, unit=u, std=r["std"].as_dict(), rung=C.RUNG, arm=ARM,
                variant=VARIANT, model=model, kind="AbsNet", d_in=int(A.d_in), d_glob=int(A.d_glob), TARGETS=T3,
                M=C.M_ROWS, M_rows=list(C.TARGETS10), layout=A.D.layout(), unused=r["model"].unused_parameter_names())


# ================================================================ unit
def _check_root(root, smoke):
    r = C.assert_write_ok(Path(root) / "probe").parent
    p = Path(C.UNITS).resolve()
    if r == p or p in r.parents:
        if smoke:
            C.stop(f"smoke / test runs never write {p}")
        if not _GATE.get("prereg"):
            C.stop(f"writing {p} needs the PREREG gate (train_abs.py run)")
    return r


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


def fingerprint(A, u, sp, model, smoke=False):
    return C.norm(dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=u["rung"], arm=ARM, model=model, variant=VARIANT, targets=T3,
        conversion=C.M_ROWS, inputs_sha256=A.sha(), pair_list=A.D.pair_list_sha, split_pairs=sp["pairs_sha"],
        split_rxn=sp["rxn_sha"], prereg=None if smoke else _GATE.get("prereg"), grid=grid(model), smoke=bool(smoke),
        fu2_mode=DC3.current_mode(),
        rules=dict(lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                   patience=F.PATIENCE, seeds=F.FINAL_SEEDS, tune_seed=F.SEED, val="inner_val reactions (FU2 split "
                   "file inner lists -> reactions)", head="mlp(d_r, 2 width, 4, n_hidden=2)", std="training reactions"),
        **FT.versions()))


def run_unit(A, u, model, root=None, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50, smoke=False):
    root = C.UNITS if root is None else root
    root = _check_root(root, smoke)
    root.mkdir(parents=True, exist_ok=True)
    mode_check()
    if u["model"] != model or u["arm"] != ARM or model not in NN_MODELS:
        C.stop(f"unit {u} does not match model {model}")
    G = grid(model)
    name = uname(u)
    sp = A.split(u["fold"])
    fp = fingerprint(A, u, sp, model, smoke)
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
    idx = list(range(len(G))) if cells is None else [int(c) for c in cells]
    tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
    if min(len(tr), len(te), len(itr), len(iva)) == 0:
        C.stop(f"{name}: empty reaction set (train {len(tr)}, test {len(te)}, inner {len(itr)} / {len(iva)})")
    if set(te.tolist()) & (set(tr.tolist()) | set(itr.tolist()) | set(iva.tolist())):
        C.stop(f"C-5: {name}: a test reaction is in the training or an inner set")
    tune = []
    for gi in idx:
        cell = G[gi]
        r = fit_abs(A, model, cell, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log)
        lk = leak_check(r["std"], itr, (iva, te), te)
        vk = np.abs(kcal(r, iva) - A.y4[iva]).mean(0)
        od = out_width(r, iva)
        if od != len(T3) or len(r["best_val_per_target"]) != len(T3):
            C.stop(f"C-T: model output {od} / validation width {len(r['best_val_per_target'])} != {len(T3)}")
        tune.append(dict(cell=cell, grid_index=gi, best_epoch=r["best_epoch"], val=r["best_val"],
                         val_per_target=r["best_val_per_target"], val_kcal_mae=dict(zip(T3, vk.tolist())),
                         epochs_run=len(r["history"]), seconds=r["seconds"], ymu=r["std"].ymu.tolist(),
                         sy=r["std"].sy.tolist(), leak=lk, out_dim=od,
                         history=[dict(epoch=h["epoch"], loss=h["loss"], val=h.get("val")) for h in r["history"]]))
        print(f"  {name} cell {cell}: val {r['best_val']:.4f} at epoch {r['best_epoch']} ({r['seconds']:.0f} s)", flush=True)
        del r
    best = min(range(len(tune)), key=lambda i: (tune[i]["val"], i))
    cell, ep = tune[best]["cell"], tune[best]["best_epoch"]
    if not (isinstance(ep, int) and ep >= 1 and np.isfinite(tune[best]["val"])):
        C.stop(f"{name}: chosen cell {cell} has best_epoch {ep} / validation {tune[best]['val']}")
    preds, finals = [], []
    for s in seeds:
        r = fit_abs(A, model, cell, tr, None, seed=s, n_epochs=ep, log=log)
        lk = leak_check(r["std"], tr, (te,), te)
        if out_width(r, te) != len(T3):
            C.stop(f"C-T: final model output != {len(T3)}")
        preds.append(kcal(r, te))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], ymu=r["std"].ymu.tolist(), sy=r["std"].sy.tolist(),
                           leak=lk, std_sha256=AD.std_sha(r["std"].as_dict()), history_tail=r["history"][-3:]))
        save_torch(root / f"{name}__s{s - F.SEED}.pt", pt_payload(A, r, cell, u, model))
        del r
    Y = np.stack(preds)                                              # [seeds, n_test, 4] kcal/mol
    rid = A.rxn_ids(te)
    rows = []
    for j, t in enumerate(T3):
        d = dict(rxn_id=rid, fold=int(u["fold"]), model=model, target=t, y=A.y4[te, j], yhat=Y[:, :, j].mean(0),
                 yhat_seed_sd=Y[:, :, j].std(0))
        for k in range(len(Y)):
            d[f"yhat_s{k}"] = Y[k, :, j]
        rows.append(pd.DataFrame(d))
    pred = pd.concat(rows, ignore_index=True)
    cols = [c for c in UNIT_COLS if c in pred.columns]
    pred = pred[cols]
    pq = root / f"{name}.parquet"
    C.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, model=model, targets=T3, fingerprint=fp, smoke=bool(smoke), variant=VARIANT,
               grid=grid(model), tune=tune, chosen=dict(cell=cell, best_epoch=ep, index=best,
                                                        grid_index=tune[best]["grid_index"]),
               final=finals, val_kcal_mae=tune[best]["val_kcal_mae"], val_rxn_sha256=sp["val_sha"],
               test_rxn_sha256=sp["test_sha"], n_train=int(len(tr)), n_test=int(len(te)), n_inner_train=int(len(itr)),
               n_inner_val=int(len(iva)), n_val=int(len(iva)), n_seeds=len(seeds), parquet_sha256=C.sha256(pq),
               parquet_columns=list(pred.columns), inputs=A.D.inputs, wall_seconds=time.time() - t_unit,
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    C.write_json(js, rec)
    return rec


# ================================================================ C helpers (tests_abs.py gpu)
def overfit(A, model):
    """C-O: 64 fold-0 training reactions (rng seed 20261019, sorted), largest cell, 300 epochs, dropout 0, 1 seed; training
    MAE of those 64 <= 10 % of each target's std (ddof 0) over the same 64."""
    tr0 = A.split(0)["train"]
    rng = np.random.default_rng(C.SEED)
    rows = np.sort(rng.choice(tr0, size=OVERFIT_N, replace=False))
    cell = grid(model)[-1]
    r = fit_abs(A, model, cell, rows, None, seed=C.SEED, n_epochs=OVERFIT_EPOCHS, dropout=0.0, log=100)
    lk = leak_check(r["std"], rows, (), np.zeros(0, dtype=int))
    yh = kcal(r, rows)
    y = A.y4[rows]
    mae = np.abs(yh - y).mean(0)
    sd64, sd_tr = y.std(0), A.y4[tr0].std(0)
    ok = bool((mae <= OVERFIT_FRAC * sd64).all())
    return dict(model=model, cell=cell, n=OVERFIT_N, epochs=OVERFIT_EPOCHS, dropout=0.0, seed=C.SEED,
                rxn_ids=A.rxn_ids(rows).tolist(), train_mae=dict(zip(T3, mae.tolist())), std_64=dict(zip(T3, sd64.tolist())),
                ratio=dict(zip(T3, (mae / sd64).tolist())), std_fold0_train=dict(zip(T3, sd_tr.tolist())),
                threshold=f"train MAE <= {OVERFIT_FRAC:g} x std (ddof 0) of the 64 reactions, every target",
                final_loss=r["history"][-1], seconds=r["seconds"], leak=lk, ok=ok)


def smoke(A, model):
    """C-8: fold 0, cells smallest + largest, 2 epochs, 1 seed -> $ABS3/smoke/nn (cleared first)."""
    root = C.SMOKE / "nn"
    _check_root(root, True)
    if root.is_dir():
        for p in root.glob(f"A__*__{model}__*"):
            C.assert_write_ok(p)
            p.unlink()
    u = dict(rung=C.RUNG, arm=ARM, model=model, fold=0)
    t0 = time.time()
    rec = run_unit(A, u, model, root=root, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, len(grid(model)) - 1], log=1,
                   smoke=True)
    return rec, root / f"{uname(u)}.parquet", time.time() - t0


# ================================================================ gates
def gate_prereg():
    import prereg_abs as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    p = C.RES / "C_tests.json"
    c = C.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K)")
    rec = (c.get("trainer_code_sha256") or {}).get("train_abs") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun the C tests")
    return dict(c_tests_sha256=C.sha256(p))


# ================================================================ commands
def cmd_run(model, i, n):
    FT.require_gpu()
    if model not in NN_MODELS:
        C.stop(f"run {model}: TF or GNN")
    i, n = C.array_task(i), int(n)
    us = [u for u in unit_list() if u["model"] == model][i::n]
    if not us:
        return 0
    pre = gate_prereg()
    gc = gate_c_tests()
    A = AD.get_data()
    print(f"[K] PREREG ok (git {pre.get('git_head')}, md {_GATE['prereg'][:12]}), C_tests ok ({gc['c_tests_sha256'][:12]}), "
          f"{FT.versions()}, {model} slice {i}/{n}: {len(us)} units", flush=True)
    for u in us:
        t0 = time.time()
        rec = run_unit(A, u, model, root=C.UNITS)
        print(f"[K] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


def load_units(model, root=None):
    root = C.UNITS if root is None else Path(root)
    names = [uname(u) for u in unit_list() if u["model"] == model]
    missing = [x for x in names if not ((root / f"{x}.json").is_file() and (root / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C.read_json(root / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C.sha256(root / f"{x}.parquet") != r.get("parquet_sha256"):
            C.stop(f"{x}.parquet changed after its json")
        if r.get("smoke"):
            C.stop(f"{x} is a smoke unit")
        if not all(f["leak"]["ok"] for f in r["tune"] + r["final"]):
            C.stop(f"C-5 failed in {x}")
        if len(r.get("final") or []) != len(F.FINAL_SEEDS):
            C.stop(f"{x}: {len(r.get('final') or [])} final seeds")
        if list(r.get("targets") or []) != T3:
            C.stop(f"{x}: targets {r.get('targets')}")
    return recs, []


def cmd_collect():
    import prereg_abs as PR
    _GATE["prereg"] = C.sha256(PR.MD) if PR.MD.is_file() else None
    A = None
    parts, report, missing_all = [], dict(units={}, written=C.now(), code_sha256=CODE_SHA,
                                          prereg_md_sha256=_GATE["prereg"]), []
    for model in NN_MODELS:
        recs, missing = load_units(model)
        if missing:
            missing_all += missing
            continue
        if A is None:
            A = AD.get_data()
        for x, r in recs.items():
            u = r["unit"]
            now = fingerprint(A, u, A.split(u["fold"]), model)
            if _no_gpu(r["fingerprint"]) != _no_gpu(now):
                C.stop(f"{x} was computed with other code / inputs: keys "
                       f"{sorted(k for k in set(now) | set(r['fingerprint']) if k != 'gpu' and r['fingerprint'].get(k) != now.get(k))}")
            parts.append(pd.read_parquet(C.UNITS / f"{x}.parquet")[UNIT_COLS])
            report["units"][x] = dict(chosen=r["chosen"], n_train=r["n_train"], n_test=r["n_test"], n_val=r["n_val"],
                                      val_kcal_mae=r["val_kcal_mae"],
                                      tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in r["tune"]],
                                      gpu=(r.get("fingerprint") or {}).get("gpu"), wall_seconds=r.get("wall_seconds"))
    if missing_all:
        print(f"GATE: {len(missing_all)} units missing: {missing_all}", file=sys.stderr)
        return 2
    pred = pd.concat(parts, ignore_index=True)
    if pred.duplicated(["rxn_id", "fold", "model", "target"]).any():
        C.stop("duplicated reaction prediction rows")
    out = C.COLLECT / "nn.parquet"
    C.write_parquet(out, pred)
    report.update(n_rows=int(len(pred)), parquet=str(out), parquet_sha256=C.sha256(out),
                  gpu_hours=float(sum(u.get("wall_seconds") or 0 for u in report["units"].values()) / 3600))
    C.write_json(C.COLLECT / "nn_report.json", report)
    print(f"[collect nn] {len(report['units'])} units, {len(pred)} rows, {report['gpu_hours']:.2f} GPU-h", flush=True)
    return 0


def main(argv):
    a = argv[1:]
    if a == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a == ["smoke"]:
        FT.require_gpu()
        A = AD.get_data()
        for m in NN_MODELS:
            rec, pq, sec = smoke(A, m)
            print(f"[smoke] {m}: {sec:.0f} s chosen {rec['chosen']}", flush=True)
        return 0
    if a[:1] == ["overfit"] and len(a) == 2:
        FT.require_gpu()
        print(json.dumps(C.clean(overfit(AD.get_data(), a[1])), indent=1))
        return 0
    if a[:1] == ["run"] and len(a) == 4:
        return cmd_run(a[1], a[2], a[3])
    if a == ["collect"]:
        return cmd_collect()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
