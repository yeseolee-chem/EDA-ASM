#!/usr/bin/env python3
r"""train_ch3.py — CH3-MODEL phases C-3′ / C-8 / K / W (docs/specs/CH3_MODEL.md 6, 8, 10).
GPU work in env_d1fu2 (torch 2.2.1+cu118) through jobs/gpu_ch3.slurm; collect on CPU (jobs/cpu_ch3.slurm CH3ENV=d1fu2).
Exit 0 = done, 2 = not ready / usage, 3 = STOP.

  python train_ch3.py list               unit names D__lf3__S__TF__f0..4
  python train_ch3.py smoke              C-8 + C-T: ch3 mode, fold 0, grid cells smallest + largest, 2 epochs, 1 seed,
                                         gate C-4 on every fit -> results/C8_smoke.json (scratch $CH3/smoke)
  python train_ch3.py c3r                C-3′: 10-target mode on DeployData('lf') (the regenerated inputs), fold 0 with
                                         the full rules (4 cells + 5 seeds) -> $CH3/c3r/D__lf__S__TF__f0.*, compared with
                                         the deploy K_report fold-0 unit and K_predictions fold 0 (same cell, |best epoch
                                         diff| <= 2, fold-0 test para MAE within +-2 % per target, max |dyhat| recorded)
                                         -> results/C3r_regress.json
  python train_ch3.py run <i|@I> <n>     units [i::n] -> $CH3/units/D__lf3__S__TF__f<k>.{json,parquet}, __s0..4.pt.
                                         Gates (exit 3): PREREG (prereg_ch3.check) and results/C_tests.json all_pass
                                         on this trainer code. Idempotent: a unit whose json exists is skipped after its
                                         fingerprint (labels, inputs, splits, code sha256, GPU name) is compared
                                         (mismatch = STOP).
  python train_ch3.py collect            the 5 units -> results/K3_predictions.parquet (deploy K_predictions columns,
                                         rung lf3, target in TARGETS3), K3_val.csv, K3_report.json; exit 2 while a unit
                                         is missing
  python train_ch3.py final              Phase W (spec 10; after eval_ch3.py run on the current K3 predictions, whatever
                                         the 9-2 verdict): mode cell (ties: lowest mean tuning validation, then grid
                                         order), n_epochs = median(best_epoch), seeds 20261019 + 0..4, Ch3Data.split('all')
                                         -> $CH3/model/D__lf3__S__TF__all__s{0..4}.pt, MODEL_CARD.json (+ copy
                                         results/MODEL_CARD_ch3.json), W_check_pairs.csv + W_check_pred.parquet (I-2);
                                         results/W3_report.json

Rules = DEPLOY 11-2 (analysis/deploy/train_tf.py run_unit / train_final, copied here with the target list as an
argument): fu2/train.py fit, Std, gate_m1, predict, batch, epoch_batches imported and called unchanged;
Mo.build('TF', cell, d_in=29, n_out=len(targets), d_glob=182, 'A+G'), grid 2/3 layers x 64/128 width, tuning on
inner_train with early stopping / cell choice on inner_val, final = outer train pairs, tuned best epoch, seeds
20261019 + 0..4, mean; gate C-4 (200 random test pairs; W: training pairs) and leak check C-5 on every fit.
The only differences in ch3 mode (spec 0-2): 4 outputs (strain, inter, solv, barrier), the identity penalty
0.1 x mean |yhat_barrier - (yhat_strain + yhat_inter + yhat_solv)| / s_barrier (fu2 loss_fn with I_BAR 3, I_CH9
[0, 1, 2]) and the tuning metric = mean standardised MAE of the 4 targets (I_PRIM [0, 1, 2, 3]) — set by
data_ch3.use_targets before any fit. Each .pt = the train_tf payload (state, cell, unit, std, rung, arm, variant,
model, d_in, d_glob, TARGETS, layout) + M (conversion matrix) in ch3 mode.
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
import ch3_common as C3  # noqa: E402
import data_ch3 as DC3  # noqa: E402

DL, FT, Mo, F = DC3.DL, DC3.FT, DC3.Mo, DC3.F
import train_tf as TT  # noqa: E402    (deploy helpers: leak_check, kcal, std_dict, std_sha, norm, choose_final, para_mae)
import torch  # noqa: E402

CODE_VERSION = "ch3-tf-1"
MODEL, VARIANT, ARM = TT.MODEL, TT.VARIANT, TT.ARM
GRID = list(TT.GRID)
PRED_COLS = list(TT.PRED_COLS)
NF = DL.NF
TARGETS3, TARGETS10 = list(C3.TARGETS3), list(C3.TARGETS10)
DEP_K_REPORT = C3.DEPLOY_RES / "K_report.json"
DEP_K_PRED = C3.DEPLOY_RES / "K_predictions.parquet"
C3R_TOL = dict(epoch=2, rel_mae=0.02)
W_CHECK_N = 1000
CODE_FILES = ("train_ch3.py", "data_ch3.py", "ch3_common.py")
DEPLOY_FILES = ("train_tf.py", "data_lf.py", "dep_common.py")
# hashed once at import: the code this process runs (fu2 train.py review F3 rule)
CODE_SHA = {**{f: C3.sha256(HERE / f) for f in CODE_FILES},
            **{f"deploy/{f}": C3.sha256(C3.DEPLOY_DIR / f) for f in DEPLOY_FILES},
            **{(k if k == "d1pair_common.py" else f"fu2/{k}"): v for k, v in FT.CODE_SHA.items()}}
TRAINER_FILES = CODE_FILES                       # the files C_tests.json must have tested
_GATE: dict = {}
norm = TT.norm


# ================================================================ helpers
def write_json(path, obj):
    return C3.write_json(path, TT._jsonable(obj))


def uname(u):
    return f"D__{u['rung']}__{ARM}__{MODEL}__f{u['fold']}"


def final_name(rung, j):
    return f"D__{rung}__{ARM}__{MODEL}__all__s{j}"


def unit_list(rung=C3.RUNG3):
    return [dict(rung=rung, arm=ARM, model=MODEL, fold=k) for k in range(NF)]


def save_torch(path, obj):
    path = C3.assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def pt_payload(D, r, cell, u, targets):
    p = dict(state=r["model"].state_dict(), cell=cell, unit=u, std=TT.std_dict(r["std"], r["sy"]), rung=D.rung,
             arm=ARM, variant=VARIANT, model=MODEL, d_in=int(D.d_in), d_glob=int(D.d_glob), TARGETS=list(targets),
             layout=D.layout())
    if list(targets) == TARGETS3:
        p["M"] = C3.M_ROWS
        p["M_rows"], p["M_cols"] = TARGETS10, TARGETS3
    return p


def mode_check(targets):
    """the fu2 constants must be those of `targets` (STOP otherwise; C-T during every fit)."""
    want = DC3.MODE3 if list(targets) == TARGETS3 else DC3.MODE10
    now = DC3.current_mode()
    if now != want:
        C3.stop(f"fu2 target constants {now} != {want} for targets {targets}")
    return now


def _check_root(root, smoke):
    r = C3.assert_write_ok(Path(root) / "probe").parent
    for prod in (C3.UNITS, C3.MODEL_DIR):
        p = Path(prod).resolve()
        if r == p or p in r.parents:
            if smoke:
                C3.stop(f"smoke / test runs never write {p}")
            if not _GATE.get("prereg"):
                C3.stop(f"writing {p} needs the PREREG gate (train_ch3.py run / final)")
    return r


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


# ================================================================ gates
def gate_prereg():
    """spec 7: prereg_ch3.check() (exit 3 otherwise); records the md sha256 for the unit fingerprint."""
    import prereg_ch3 as PR
    rec = PR.check()
    _GATE["prereg"] = rec["md_sha256"]
    return rec


def gate_c_tests():
    """spec 6 / 15: results/C_tests.json all_pass, and its trainer code sha256 = the files now."""
    p = C3.RES / "C_tests.json"
    c = C3.read_json(p) if p.is_file() else {}
    if not c.get("all_pass"):
        C3.stop(f"{p}: all_pass {c.get('all_pass')} (Phase C must pass before K / W)")
    rec = c.get("trainer_code_sha256") or {}
    diff = [f for f in TRAINER_FILES if rec.get(f) != CODE_SHA[f]]
    if diff:
        C3.stop(f"{p.name} trainer_code_sha256 differs from the code now for {diff}: rerun the C tests")
    return dict(c_tests_sha256=C3.sha256(p))


def fingerprint(D, u, sp, targets, smoke=False, limit=None):
    """unit fingerprint (spec 8: labels, inputs, splits, code sha256, GPU name) — compared in full (GPU included)."""
    return norm(dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=D.rung, arm=ARM, model=MODEL, variant=VARIANT,
        targets=list(targets), conversion=C3.M_ROWS if list(targets) == TARGETS3 else None,
        usable_rule=D.usable_rule, inputs_sha256=D.sha, pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"],
        prereg=None if smoke else _GATE.get("prereg"), grid=GRID, limit=limit, smoke=bool(smoke),
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


def _outdim(r, D, rows):
    """output width of a fitted model: fu2 predict on two of `rows` (C-T)."""
    rows = np.asarray(rows)[:2]
    return int(FT.predict(r["model"], r["T"], D.ra[rows], D.rb[rows]).shape[1])


# ================================================================ one CV unit (= train_tf.run_unit, targets as argument)
def run_unit(D, u, targets, root=None, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50, smoke=False,
             limit=None):
    """tune (grid cells on inner_train, early stopping on inner_val) -> final (seeds, best epoch, train) -> test
    predictions + gate C-4 + leak check C-5 for one unit; targets = TARGETS10 (C-3′) or TARGETS3 (ch3)."""
    targets = list(targets)
    root = C3.UNITS if root is None else root
    root = _check_root(root, smoke or bool(limit))
    root.mkdir(parents=True, exist_ok=True)
    mode_check(targets)
    if D.y.shape[1] != len(targets):
        C3.stop(f"data y has {D.y.shape[1]} columns, targets {targets}")
    if u["rung"] != D.rung or u["model"] != MODEL or u["arm"] != ARM:
        C3.stop(f"unit {u} does not match the data ({D.rung}, {ARM}, {MODEL})")
    name = uname(u)
    sp = _limit(D.split(u["fold"]), limit)
    fp = fingerprint(D, u, sp, targets, smoke, limit)
    js = root / f"{name}.json"
    if js.is_file():
        old = C3.read_json(js)
        ofp = old.get("fingerprint") or {}
        if ofp != fp:
            C3.stop(f"cache {js} has a different fingerprint: keys "
                    f"{sorted(k for k in set(fp) | set(ofp) if ofp.get(k) != fp.get(k))}")
        print(f"  {name}: cached", flush=True)
        return old
    t_unit = time.time()
    idx = list(range(len(GRID))) if cells is None else [int(c) for c in cells]
    tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
    if min(len(tr), len(te), len(itr), len(iva)) == 0:
        C3.stop(f"{name}: empty split list (train {len(tr)}, test {len(te)}, inner {len(itr)} / {len(iva)})")
    tune = []
    for ci, gi in enumerate(idx):
        cell = GRID[gi]
        r = FT.fit(D, MODEL, cell, VARIANT, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log)
        lk = TT.leak_check(D, r["std"], itr, (iva, te), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=F.SEED + ci)        # inputs only, no labels (M-1)
        vk = np.abs(TT.kcal(r, D, iva) - D.y[iva]).mean(0)
        od = _outdim(r, D, iva)
        if od != len(targets) or len(r["best_val_per_target"]) != len(targets):
            C3.stop(f"C-T: model output {od} / validation width {len(r['best_val_per_target'])} != {len(targets)}")
        tune.append(dict(cell=cell, grid_index=gi, best_epoch=r["best_epoch"], val=r["best_val"],
                         val_per_target=r["best_val_per_target"], val_kcal_mae=dict(zip(targets, vk.tolist())),
                         epochs_run=len(r["history"]), seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
                         out_dim=od,
                         history=[dict(epoch=h["epoch"], loss=h["loss"], val=h.get("val")) for h in r["history"]]))
        print(f"  {name} cell {cell}: val {r['best_val']:.4f} at epoch {r['best_epoch']} ({r['seconds']:.0f} s) "
              f"gate {g}", flush=True)
        if not g["ok"]:
            C3.stop(f"gate C-4 failed ({name}, tuning cell {cell}): {g}")
        del r
    best = min(range(len(tune)), key=lambda i: (tune[i]["val"], i))
    cell, ep = tune[best]["cell"], tune[best]["best_epoch"]
    if not (isinstance(ep, int) and ep >= 1 and np.isfinite(tune[best]["val"])):
        C3.stop(f"{name}: chosen cell {cell} has best_epoch {ep} / validation {tune[best]['val']}")
    preds, finals = [], []
    for s in seeds:
        r = FT.fit(D, MODEL, cell, VARIANT, tr, None, seed=s, n_epochs=ep, log=log)
        lk = TT.leak_check(D, r["std"], tr, (te,), te)
        g = FT.gate_m1(r["model"], D, r["std"], te, seed=s)
        if not g["ok"]:
            C3.stop(f"gate C-4 failed ({name}, seed {s}): {g}")
        if _outdim(r, D, te) != len(targets):
            C3.stop(f"C-T: final model output {_outdim(r, D, te)} != {len(targets)}")
        preds.append(TT.kcal(r, D, te))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(), leak=lk,
                           std_sha256=TT.std_sha(TT.std_dict(r["std"], r["sy"])), history_tail=r["history"][-3:]))
        save_torch(root / f"{name}__s{s - F.SEED}.pt", pt_payload(D, r, cell, u, targets))
        del r
    Y = np.stack(preds)                                                    # [seeds, n_test, n_targets] kcal/mol
    pid = D.P.index.to_numpy()[te].astype(str)
    rows = [pd.DataFrame(dict(pair_id=pid, ref_kind=D.P.ref_kind.to_numpy()[te], pos=D.P.pos.to_numpy()[te],
                              fold=int(u["fold"]), rung=u["rung"], arm=ARM, model=MODEL, target=t,
                              y=D.y[te, j], yhat=Y[:, :, j].mean(0), base=0.0, beta=0.0,
                              yhat_seed_sd=Y[:, :, j].std(0))) for j, t in enumerate(targets)]
    pred = pd.concat(rows, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    C3.write_parquet(pq, pred)
    rec = dict(unit=u, name=name, targets=targets, fingerprint=fp, smoke=bool(smoke), limit=limit, variant=VARIANT,
               tune=tune, chosen=dict(cell=cell, best_epoch=ep, index=best, grid_index=tune[best]["grid_index"]),
               final=finals, val_kcal_mae=tune[best]["val_kcal_mae"], val_pairs_sha256=sp["val_sha"],
               test_pairs_sha256=sp["test_sha"], n_val=len(iva), n_train=len(tr), n_test=len(te),
               n_inner_train=len(itr), n_inner_val=len(iva), dropped=sp["dropped"], parquet_sha256=C3.sha256(pq),
               inputs=D.inputs, usable_rule=D.usable_rule, wall_seconds=time.time() - t_unit, host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    write_json(js, rec)
    return rec


# ================================================================ C-T (spec 6)
def ct_checks(D, pt_path, targets=TARGETS3):
    """C-T: ch3 mode — fu2 constants, model output width 4 on a real batch, loss identity penalty = barrier vs the sum
    of [0, 1, 2], validation metric = mean of the 4 targets (checked on the unit records by the caller), saved .pt
    TARGETS = TARGETS3 (+ its state loads into a 4-output model)."""
    chk, det = {}, {}
    now = DC3.current_mode()
    chk["fu2_constants_ch3"] = now == DC3.MODE3
    det["fu2_mode"] = now
    rows = D.split(0)["train"][:64]
    std = FT.Std(D, rows)
    T = std.tensors(D)
    m = Mo.build(MODEL, GRID[0], D.d_in, len(F.TARGETS), D.d_glob, VARIANT, F.DROPOUT).to(FT.DEV)
    y = FT.predict(m, T, D.ra[rows[:8]], D.rb[rows[:8]])
    chk["model_output_4"] = tuple(y.shape) == (8, len(targets)) == (8, 4)
    det["output_shape"] = list(y.shape)
    g = torch.Generator().manual_seed(1)
    yh, yy = torch.randn(16, 4, generator=g, dtype=torch.float64), torch.randn(16, 4, generator=g, dtype=torch.float64)
    sy = torch.rand(4, generator=g, dtype=torch.float64) + 0.5
    tot, l1, pen = FT.loss_fn(yh, yy, sy)
    kc = yh * sy
    pen_w = ((kc[:, 3] - (kc[:, 0] + kc[:, 1] + kc[:, 2])).abs() / sy[3]).mean()
    want = (yh - yy).abs().mean() + 0.1 * pen_w
    chk["loss_identity_sum3"] = bool(abs(float(pen) - float(pen_w)) <= 1e-12 and abs(float(tot) - float(want)) <= 1e-12)
    det["loss"] = dict(total=float(tot), want=float(want), pen=float(pen), pen_want=float(pen_w))
    obj = torch.load(pt_path, map_location="cpu")
    chk["pt_TARGETS3"] = list(obj.get("TARGETS") or []) == list(TARGETS3)
    mm = Mo.build(MODEL, obj["cell"], int(obj["d_in"]), 4, int(obj["d_glob"]), VARIANT, F.DROPOUT)
    try:
        mm.load_state_dict(obj["state"], strict=True)
        chk["pt_state_4_outputs"] = True
    except RuntimeError as e:                                              # noqa: PERF203
        chk["pt_state_4_outputs"] = False
        det["pt_load_error"] = str(e)[:300]
    det["pt"] = str(pt_path)
    del T
    return dict(ok=bool(all(chk.values())), checks=chk, detail=det)


def val_metric_check(rec, n=4):
    """C-T: tuning validation = mean of the per-target standardised MAEs (n of them) on every tuning fit."""
    dev = [abs(float(t["val"]) - float(np.mean(t["val_per_target"]))) for t in rec["tune"]]
    ok = all(len(t["val_per_target"]) == n for t in rec["tune"]) and max(dev) <= 1e-12
    return dict(ok=bool(ok), max_abs_dev=float(max(dev)), widths=[len(t["val_per_target"]) for t in rec["tune"]])


# ================================================================ commands
def cmd_smoke():
    """C-8 + C-T (spec 6): ch3 mode, fold 0, cells smallest + largest, 2 epochs, 1 final seed, gate C-4 on every fit
    -> results/C8_smoke.json; scratch $CH3/smoke (its D__* files removed first)."""
    FT.require_gpu()
    root = C3.SMOKE
    _check_root(root, True)
    if root.is_dir():
        for p in root.glob("D__*"):
            C3.assert_write_ok(p)
            p.unlink()
    D = DC3.get_data("ch3")
    DC3.use_targets(TARGETS3)
    u = dict(rung=D.rung, arm=ARM, model=MODEL, fold=0)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rec = run_unit(D, u, TARGETS3, root=root, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, len(GRID) - 1], log=1,
                   smoke=True)
    ct = ct_checks(D, root / f"{uname(u)}__s0.pt")
    vm = val_metric_check(rec, 4)
    ct["checks"]["val_metric_mean4"] = vm["ok"]
    ct["detail"]["val_metric"] = vm
    ct["ok"] = bool(all(ct["checks"].values()))
    eps = [t["seconds"] / max(1, t["epochs_run"]) for t in rec["tune"]]
    ok = TT.gates_ok(rec) and ct["ok"]
    out = dict(check="C-8 + C-T", ok=bool(ok), C8_ok=bool(TT.gates_ok(rec)), CT=ct, versions=FT.versions(),
               device=str(FT.DEV), unit=uname(u), seconds=time.time() - t0,
               tune_epoch_seconds=dict(smallest=eps[0], largest=eps[-1]),
               gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
               gate_C4_tune=[t["gate_M1"] for t in rec["tune"]], gate_C4_final=[f["gate_M1"] for f in rec["final"]],
               leak_C5=[t["leak"] for t in rec["tune"]] + [f["leak"] for f in rec["final"]],
               n_train=rec["n_train"], n_test=rec["n_test"], n_inner_train=rec["n_inner_train"],
               n_inner_val=rec["n_inner_val"], dropped=rec["dropped"], d_in=int(D.d_in), d_glob=int(D.d_glob),
               code_sha256=CODE_SHA)
    write_json(C3.RES / "C8_smoke.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "C8_ok", "seconds", "tune_epoch_seconds", "gpu_mem_peak_gb")}, indent=1))
    print(json.dumps(ct["checks"], indent=1))
    return 0 if ok else 3


def cmd_c3r():
    """C-3′ (spec 6): 10-target mode, DeployData('lf'), fold 0, full rules on RTX 3090, vs deploy K fold 0."""
    FT.require_gpu()
    kr = C3.read_json(DEP_K_REPORT)
    ref = kr["units"]["D__lf__S__TF__f0"]
    D = DC3.get_data("lf")
    DC3.use_targets(TARGETS10)
    u = dict(rung="lf", arm=ARM, model=MODEL, fold=0)
    rec = run_unit(D, u, TARGETS10, root=C3.C3R)
    mine = pd.read_parquet(C3.C3R / f"{uname(u)}.parquet")
    K = pd.read_parquet(DEP_K_PRED)
    theirs = K[K.fold == 0]
    m = mine.merge(theirs[["pair_id", "target", "y", "yhat", "yhat_seed_sd"]], on=["pair_id", "target"],
                   suffixes=("", "_dep"), how="outer", indicator=True)
    same_pairs = bool((m._merge == "both").all())
    dy = float(np.abs(m.y - m.y_dep).max()) if same_pairs else float("inf")
    dyhat = float(np.abs(m.yhat - m.yhat_dep).max()) if same_pairs else float("inf")
    dyhat_t = {t: float(np.abs(g.yhat - g.yhat_dep).max()) for t, g in m.groupby("target")} if same_pairs else None
    mae_me, n_para = TT.para_mae(mine, D.P)
    mae_dep, _ = TT.para_mae(theirs, D.P)
    rel = {t: (mae_me[t] - mae_dep[t]) / mae_dep[t] for t in TARGETS10}
    cell_ok = norm(rec["chosen"]["cell"]) == norm(ref["chosen"]["cell"])
    ep_diff = int(rec["chosen"]["best_epoch"]) - int(ref["chosen"]["best_epoch"])
    gpu_now, gpu_ref = FT.versions()["gpu"], ref.get("gpu")
    checks = dict(i_same_cell=cell_ok, ii_best_epoch_within_2=abs(ep_diff) <= C3R_TOL["epoch"],
                  iii_para_mae_within_2pct=all(abs(v) <= C3R_TOL["rel_mae"] for v in rel.values()),
                  same_test_pairs=same_pairs, same_y=dy <= 1e-12, gates_C4=TT.gates_ok(rec))
    out = dict(check="C-3′", ok=bool(all(checks.values())), checks=checks, gpu=gpu_now, gpu_deploy=gpu_ref,
               same_gpu_type=gpu_now == gpu_ref, chosen=rec["chosen"], chosen_deploy=ref["chosen"],
               best_epoch_diff=ep_diff,
               tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in rec["tune"]],
               tune_deploy=ref.get("tune"), n_test_para=n_para, para_mae=mae_me, para_mae_deploy=mae_dep,
               para_mae_rel_diff=rel, max_abs_dyhat=dyhat, max_abs_dyhat_per_target=dyhat_t, max_abs_dy=dy,
               n_test=rec["n_test"], n_test_deploy=ref.get("n_test"), n_train=rec["n_train"],
               n_train_deploy=ref.get("n_train"), wall_seconds=rec["wall_seconds"],
               wall_seconds_deploy=ref.get("wall_seconds"),
               unit_json_sha256=C3.sha256(C3.C3R / f"{uname(u)}.json"), code_sha256=CODE_SHA, tolerances=C3R_TOL,
               deploy_k_report_sha256=C3.sha256(DEP_K_REPORT), deploy_k_predictions_sha256=C3.sha256(DEP_K_PRED))
    if gpu_now != gpu_ref:
        out["note"] = "GPU type differs from the deploy K unit; the spec asks for RTX 3090 (gpu1)"
    write_json(C3.RES / "C3r_regress.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "checks", "chosen", "best_epoch_diff", "max_abs_dyhat")}, indent=1,
                     default=str))
    return 0 if out["ok"] else 3


def cmd_run(i, n):
    FT.require_gpu()
    i = C3.array_task(i)
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
    D = DC3.get_data("ch3")
    DC3.use_targets(TARGETS3)
    print(f"[K] PREREG ok (git {pre.get('git_head')}, md {_GATE['prereg'][:12]}), C_tests ok "
          f"({gc['c_tests_sha256'][:12]}), {FT.versions()}, mode {DC3.current_mode()}, slice {i}/{n}: {len(us)} units",
          flush=True)
    for u in us:
        t0 = time.time()
        rec = run_unit(D, u, TARGETS3, root=C3.UNITS)
        print(f"[K] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


def load_units(root=None, rung=C3.RUNG3):
    """the 5 fold units: json / parquet / 5 .pt present, parquet sha256 = json, every gate ok, no smoke / limited unit."""
    root = C3.UNITS if root is None else Path(root)
    names = [uname(u) for u in unit_list(rung)]
    missing = [x for x in names if not ((root / f"{x}.json").is_file() and (root / f"{x}.parquet").is_file())]
    if missing:
        return None, missing
    recs = {x: C3.read_json(root / f"{x}.json") for x in names}
    for x, r in recs.items():
        if C3.sha256(root / f"{x}.parquet") != r.get("parquet_sha256"):
            C3.stop(f"{x}.parquet changed after its json")
        if r.get("smoke") or r.get("limit") or not TT.gates_ok(r):
            C3.stop(f"gate C-4 failed (or smoke / limited unit) in {x}")
        if len(r.get("final") or []) != len(F.FINAL_SEEDS):
            C3.stop(f"{x}: {len(r.get('final') or [])} final seeds, not {len(F.FINAL_SEEDS)}")
        for j in range(len(F.FINAL_SEEDS)):
            if not (root / f"{x}__s{j}.pt").is_file():
                C3.stop(f"{x}__s{j}.pt missing")
        if list(r.get("targets") or []) != TARGETS3:
            C3.stop(f"{x}: targets {r.get('targets')} != {TARGETS3}")
    return recs, []


def cmd_collect():
    D = DC3.get_data("ch3")
    DC3.use_targets(TARGETS3)
    recs, missing = load_units()
    if missing:
        print(f"GATE: {len(missing)} units missing: {missing}", file=sys.stderr)
        return 2
    import prereg_ch3 as PR
    md_now = C3.sha256(PR.MD) if PR.MD.is_file() else None
    _GATE["prereg"] = md_now                          # the units must have been trained under the current PREREG md
    for x, r in recs.items():                         # every unit must carry the CURRENT fingerprint (GPU aside: CPU)
        u = r["unit"]
        now = fingerprint(D, u, D.split(u["fold"]), TARGETS3)
        if _no_gpu(r["fingerprint"]) != _no_gpu(now):
            C3.stop(f"{x} was computed with other code / inputs: keys "
                    f"{sorted(k for k in set(now) | set(r['fingerprint']) if k != 'gpu' and r['fingerprint'].get(k) != now.get(k))}")
    pred = pd.concat([pd.read_parquet(C3.UNITS / f"{x}.parquet")[PRED_COLS] for x in recs], ignore_index=True)
    if pred.duplicated(["pair_id", "target"]).any():
        C3.stop("duplicated prediction rows (a pair in two test folds?)")
    if set(pred.rung) != {C3.RUNG3} or set(pred.arm) != {ARM} or set(pred.model) != {MODEL} \
            or set(pred.target) != set(TARGETS3):
        C3.stop(f"K3 predictions must be rung {C3.RUNG3}, arm S, model TF, targets {TARGETS3}")
    out = C3.RES / "K3_predictions.parquet"
    C3.write_parquet(out, pred)
    vrows = [dict(rung=C3.RUNG3, arm=ARM, model=MODEL, fold=int(r["unit"]["fold"]), target=t,
                  val_kcal_mae=float(r["val_kcal_mae"][t]), n_val=r.get("n_val"),
                  val_pairs_sha256=r.get("val_pairs_sha256")) for x, r in recs.items() for t in TARGETS3]
    vout = C3.RES / "K3_val.csv"
    C3.write_csv(vout, pd.DataFrame(vrows))
    rep = dict(n_units=len(recs), rung=C3.RUNG3, arm=ARM, model=MODEL, targets=TARGETS3, conversion=C3.M_ROWS,
               code=CODE_VERSION, code_sha256=CODE_SHA, prereg_md_sha256=md_now, inputs_sha256=D.sha,
               pair_list_sha=D.pair_list_sha, n_pairs_P_lf=len(D.pair_list),
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
               n_test_pairs=int(pred.pair_id.nunique()), predictions=str(out), predictions_sha256=C3.sha256(out),
               val_csv=str(vout), val_csv_sha256=C3.sha256(vout))
    write_json(C3.RES / "K3_report.json", rep)
    print(f"[collect] {len(recs)} units -> {out.name} ({len(pred)} rows, {rep['n_test_pairs']} pairs)", flush=True)
    return 0


# ---------------------------------------------------------------- Phase W
def gate_e3_report():
    """W after eval_ch3.py run on the current K3 predictions (spec 10: whatever the 9-2 verdict)."""
    p = C3.RES / "E3_report.json"
    if not p.is_file():
        C3.not_ready(f"{p} missing (eval_ch3.py run)")
    e = C3.read_json(p)
    kp = C3.RES / "K3_predictions.parquet"
    if (e.get("inputs_sha256") or {}).get("K3_predictions") != C3.sha256(kp):
        C3.stop("E3_report.json was computed on another K3_predictions.parquet")
    return e


def model_card(D, sel, tr, finals, e_rep, ad, tr_ids_sha, train_mae, std0):
    files = sorted(HERE.glob("*.py"))
    rx = D.rxn_rows(tr)
    dip = sorted(set(D.P.dipole.to_numpy()[tr].astype(str))) if "dipole" in D.P.columns else None
    pos = sorted(set(D.P.pos.to_numpy()[tr].astype(str)))
    return dict(
        model="lf3 . S . TF (FU2 M-2 TF, variant A+G; 3 channels + barrier)", rung=D.rung, arm=ARM, variant=VARIANT,
        cell=sel["cell"], n_epochs=sel["n_epochs"], seeds=list(F.FINAL_SEEDS), selection=sel,
        files=[f"{final_name(D.rung, j)}.pt" for j in range(len(F.FINAL_SEEDS))],
        n_train_pairs=int(len(tr)), train_pairs_sha256=tr_ids_sha, n_train_reactions=int(len(rx)),
        targets=list(TARGETS3), conversion=dict(M=C3.M_ROWS, rows=TARGETS10, cols=TARGETS3,
                                                groups={k: list(v) for k, v in C3.GROUPS.items()}),
        adoption=dict(adopted=(e_rep.get("adoption") or {}).get("adopted"), rule=(e_rep.get("adoption") or {}).get("rule"),
                      per_target=(e_rep.get("adoption") or {}).get("per_target"), source="results/E3_report.json"),
        token=dict(version=D.trep.get("token_version"), atom_version=DC3.DC.ATOM_VERSION,
                   elements=D.layout()["elements"], bin_cols=list(D.bin_cols), num_cols=list(D.num_cols),
                   d_in=int(D.d_in)),
        glob_cols=list(D.trep["glob_cols"]), d_glob=int(D.d_glob),
        s_y=dict(zip(TARGETS3, np.asarray(std0["sy"], float).tolist())), std_sha256=TT.std_sha(std0),
        std_sha256_per_seed=[f["std_sha256"] for f in finals],
        code_sha256={**{f"analysis/ch3/{p.name}": C3.sha256(p) for p in files},
                     **{f"analysis/deploy/{f}": C3.sha256(C3.DEPLOY_DIR / f) for f in DEPLOY_FILES},
                     "analysis/d1_pair/fu2/train.py": C3.sha256(DL.FU2_DIR / "train.py"),
                     "analysis/d1_pair/fu2/models.py": C3.sha256(DL.FU2_DIR / "models.py")},
        versions=FT.versions(), inputs_sha256=D.sha, usable_rule=DL.USABLE_RULES[D.usable_rule],
        cv_performance=dict(source="results/E3_report.json (Phase K cross-validation; the final model has no test set)",
                            summary=e_rep.get("summary"), compare=e_rep.get("compare_para")),
        train_mae_reference_only=train_mae,
        applicability=dict(thresholds_yhat_sd=ad.get("thresholds"), threshold_rule=ad.get("rule"),
                           elements=list(DC3.DC.ELEMENTS), charge="neutral (charge 0)",
                           ts_atoms=dict(min=int(D.n[rx].min()), max=int(D.n[rx].max()), spec="8-63"),
                           dipoles=dip, positions_spec=["p", "m"], positions_trained=pos))


def cmd_final():
    """Phase W (spec 10): gates (GPU, PREREG, C_tests, E3 run on the current K3 predictions, E3_ad present), the 5 K3
    units, the selection rule (deploy train_tf.choose_final), then train_final on Ch3Data split('all')."""
    FT.require_gpu()
    pre = gate_prereg()
    gate_c_tests()
    e_rep = gate_e3_report()
    adp = C3.RES / "E3_ad.json"
    if not adp.is_file():
        C3.not_ready(f"{adp} missing (eval_ch3.py run)")
    ad = C3.read_json(adp)
    recs, missing = load_units()
    if missing:
        C3.not_ready(f"K3 units missing: {missing}")
    sel = TT.choose_final(recs)
    D = DC3.get_data("ch3")
    DC3.use_targets(TARGETS3)
    return train_final(D, sel, recs, e_rep, ad, root=C3.MODEL_DIR, prereg_head=pre.get("git_head"))


def train_final(D, sel, recs, e_rep, ad, root, prereg_head=None, limit=None, n_epochs=None, check_n=W_CHECK_N,
                smoke=False):
    """the W fits (= train_tf.train_final with TARGETS3): split('all') training pairs, seeds 20261019 + 0..4, gate C-4
    on 200 random training pairs, C-T on the .pt, leak check C-5, the 5 .pt, W_check_pairs.csv + W_check_pred.parquet
    (I-2), MODEL_CARD.json (+ results/MODEL_CARD_ch3.json), results/W3_report.json, W_unit.json (cache)."""
    mode_check(TARGETS3)
    sp = D.split("all")
    tr = sp["train"][:int(limit)] if limit else sp["train"]
    ep = int(n_epochs) if n_epochs else int(sel["n_epochs"])
    root = _check_root(root, smoke or bool(limit) or bool(n_epochs))
    root.mkdir(parents=True, exist_ok=True)
    wfp = norm(dict(code=CODE_VERSION, code_sha256=CODE_SHA, inputs_sha256=D.sha, pair_list=D.pair_list_sha,
                    targets=TARGETS3, train_pairs=DL.list_sha(D._ids(tr)), selection=dict(cell=sel["cell"], n_epochs=ep),
                    k_units={x: r["parquet_sha256"] for x, r in recs.items()}, prereg=_GATE.get("prereg"),
                    e_report=(C3.sha256(C3.RES / "E3_report.json") if (C3.RES / "E3_report.json").is_file() else None),
                    rules=dict(seeds=F.FINAL_SEEDS, check_n=check_n), limit=limit, **FT.versions()))
    wj = root / "W_unit.json"
    if wj.is_file():
        old = C3.read_json(wj)
        if old.get("fingerprint") != wfp:
            C3.stop(f"{wj} has a different fingerprint: keys "
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
        lk = TT.leak_check(D, r["std"], tr, (), np.zeros(0, dtype=int))
        g = FT.gate_m1(r["model"], D, r["std"], tr, seed=s)               # C-4 on 200 random training pairs
        if not g["ok"]:
            C3.stop(f"gate C-4 failed (final, seed {s}): {g}")
        if _outdim(r, D, tr) != len(TARGETS3):
            C3.stop(f"C-T: final model output {_outdim(r, D, tr)} != 4")
        sd = TT.std_dict(r["std"], r["sy"])
        if std0 is None:
            std0 = sd
        elif TT.std_sha(sd) != TT.std_sha(std0):
            C3.stop("final fits: standardisation differs between seeds (same training pairs)")
        preds_chk.append(TT.kcal(r, D, chk))
        preds_tr.append(TT.kcal(r, D, tr))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, leak=lk,
                           std_sha256=TT.std_sha(sd), history_tail=r["history"][-3:]))
        save_torch(root / f"{final_name(D.rung, s - F.SEED)}.pt", pt_payload(D, r, sel["cell"], u, TARGETS3))
        print(f"[W] seed {s}: {r['seconds']:.0f} s, gate {g}", flush=True)
        del r
    ct = ct_checks(D, root / f"{final_name(D.rung, 0)}.pt")
    if not ct["ok"]:
        C3.stop(f"C-T failed on the final model: {ct['checks']}")
    Yc, Yt = np.stack(preds_chk), np.stack(preds_tr)
    pid_c = D.P.index.to_numpy()[chk].astype(str)
    C3.write_csv(root / "W_check_pairs.csv", pd.DataFrame(dict(pair_id=pid_c, a_rxn=D.P.a_rxn.to_numpy()[chk],
                                                               b_rxn=D.P.b_rxn.to_numpy()[chk])))
    rows = [pd.DataFrame(dict(pair_id=pid_c, target=t, yhat=Yc[:, :, j].mean(0), yhat_sd=Yc[:, :, j].std(0),
                              **{f"yhat_s{k}": Yc[k, :, j] for k in range(len(Yc))})) for j, t in enumerate(TARGETS3)]
    C3.write_parquet(root / "W_check_pred.parquet", pd.concat(rows, ignore_index=True))
    train_mae = {t: float(np.abs(Yt[:, :, j].mean(0) - D.y[tr, j]).mean()) for j, t in enumerate(TARGETS3)}
    tid = D._ids(tr)
    card = model_card(D, dict(sel, n_epochs=ep), tr, finals, e_rep, ad, DL.list_sha(tid), train_mae, std0)
    write_json(root / "MODEL_CARD.json", card)
    C3.write_bytes(C3.RES / "MODEL_CARD_ch3.json", (root / "MODEL_CARD.json").read_bytes())
    rep = dict(phase="W", selection=sel, n_train_pairs=int(len(tr)), n_in_T_main=sp.get("n_in_T_main"),
               dropped=sp["dropped"], finals=finals, gate_C4_ok=all(f["gate_M1"]["ok"] for f in finals), C_T=ct,
               train_mae_reference_only=train_mae, check_pairs=dict(n=int(len(chk)), file=str(root / "W_check_pairs.csv"),
                                                                    pred=str(root / "W_check_pred.parquet")),
               model_card=str(root / "MODEL_CARD.json"), model_card_sha256=C3.sha256(root / "MODEL_CARD.json"),
               pt_sha256={f"{final_name(D.rung, j)}.pt": C3.sha256(root / f"{final_name(D.rung, j)}.pt")
                          for j in range(len(F.FINAL_SEEDS))},
               prereg_git_head=prereg_head, limit=limit, wall_seconds=time.time() - t0, versions=FT.versions(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    write_json(C3.RES / "W3_report.json", rep)
    write_json(wj, dict(fingerprint=wfp, report_sha256=C3.sha256(C3.RES / "W3_report.json")))
    print(json.dumps(dict(cell=sel["cell"], n_epochs=ep, n_train=len(tr), gate=rep["gate_C4_ok"], C_T=ct["ok"]),
                     indent=1))
    return 0


def main(argv):
    a = argv[1:]
    if a[:1] == ["list"]:
        print("\n".join(uname(u) for u in unit_list()))
        return 0
    if a == ["smoke"]:
        return cmd_smoke()
    if a == ["c3r"]:
        return cmd_c3r()
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
