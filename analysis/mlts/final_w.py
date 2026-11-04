#!/usr/bin/env python3
r"""final_w.py — DEPLOY-V2 (MLTS) Phase W (docs/specs/DEPLOY_V2_MLTS.md 13 W-1, W-2): the final channel model and the
MLTS committee flag. Compute nodes only (jobs/mlt.slurm or jobs/sbx.sh); every part re-executes itself in its env.

  python final_w.py channel [--smoke]
        W-1: the channel model of the K primary config trained on all D1. C1 (no retraining in K) = the LADDER dft
        configuration (its REP8 per target: arm x model) trained on every usable in_T_main pair of the dft rung (= the
        union of the 5 fold training lists); C2 = the same on rung 'mlts' (MLTS inputs, the K unit jsons
        $MLT/units/C2); C3 = the same on the train_k mixed data (MLTS + label-TS inputs, $MLT/units/C3).
        Parts (children): `_atoms <arm> <model> <rung> <units_dir>` (env_d1fu2, GPU: train_atoms fit / fit_sh, 5 seeds)
        and `_xgb <arm> <rung> <units_dir> <targets>` (env_espley: train_xgb.prep_fit + d1_train._xgb_fit per target).
        -> $MLT/w/channel/{W__<rung>__<arm>__<model>__all__s<j>.pt, W__<rung>__<arm>__XGB__all__<t>.json + scale npz,
        part_*.json, final_on_E_*.parquet (the final models on the E pairs with the deployed MLTS inputs; P replay),
        MODEL_CARD.json (written last)}. --smoke: 2 epochs, 1 seed, 1,500 pairs, 1 XGB target, under $MLT/w/smoke.
  python final_w.py committee <i|@I> 3 [--models a,b,c] [--limit N]
        W-2: member i (FT_all_s<i>_al) re-optimises the TS of every E reaction (the ends of the E common P_main para
        pairs, $MLT/eval/E_pairs.csv) from the main-engine MLTS TS ($MLT/ade/jobs/<job_id>/ts.xyz, its formed_pairs_ts)
        with mlts_bridge.optimize_ts (A-1 scheme, TightOpt) -> $MLT/w/committee/<model>/recs/<rxn_id>.json. Exit 0 all
        records, 2 incomplete (stops claiming 2 h before the 48 h wall), 3 > 20 pipeline errors.
  python final_w.py report
        -> results/W_report.json {complete, channel, committee {delta_com quantiles, threshold}, relation (delta_com vs
        the E channel errors), cost per reaction} + $MLT/w/delta_com.csv. Exit 0 / 2 (channel or committee unfinished).

Decisions (for DECISIONS.md; also in W_report.json)"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)

CODE_VERSION = "mlts-w-1"
REPO = M.REPO
LAD_DIR = REPO / "analysis" / "d1_ladder"
W_DIR = M.MLT / "w"
K_REPORT = M.RES / "K_report.json"
E_REPORT = M.RES / "E_report.json"
W_REPORT = M.RES / "W_report.json"
E_PAIRS = M.MLT / "eval" / "E_pairs.csv"
E_ERR = M.MLT / "eval" / "E_pair_errors.parquet"
TARGETS = list(M.TARGETS)
PRIMARY9 = list(M.PRIMARY)
H9 = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
SEED = 20261019
N_GATE = 200
WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
RESERVE_H = float(os.environ.get("MLT_RESERVE_H", "2"))
MAX_ERR = 20
THR_PCT = 95.0
COM_TIMEOUT_S = float(os.environ.get("MLTS_W_TIMEOUT_S", "7200"))    # per ORCA run of a committee optimisation
T0 = time.time()

DECISIONS = {
    "W-final-pairs": "final training pairs = every usable in_T_main pair of the training rung = the union of the 5 T1 "
                     "fold training lists (DEPLOY W / data_lf split('all') rule); no test / validation lists",
    "W-final-cell": "atom models: grid cell = the most frequent chosen cell of the 5 fold units (ties: lowest grid index), "
                    "epochs = median of the 5 fold best epochs (DEPLOY W precedent: 3x64 of 5, median 46), seeds "
                    "20261019 + 0..4, prediction = mean of the 5 (= LADDER run_unit final); XGB: per target the most "
                    "frequent chosen cell of the 5 fold units (ties lowest index), one fit (deterministic)",
    "W-final-C1": "C1 primary (spec 13 W-1 'C1이면 LADDER dft 설정을 전체 D1으로 학습'): no LADDER final unit exists "
                  "($LAD/units has fold units only), so the dft REP8 (arm x model per target) is trained here on the "
                  "dft rung (label-TS inputs) and applied to MLTS inputs at deployment, as C1 does per fold",
    "W-final-C23": "C2 = the same trainer on rung 'mlts' (predict_c1.register_mlts, the H table of the config's refs "
                   "mode) with the K unit jsons in $MLT/units/C2; C3 = the same trainer on the train_k mixed data "
                   "(make_mix_atoms / make_mix_xgb: MLTS + label-TS rows, same y, rung mlts_mix; union of the 5 fold "
                   "training lists of the mixed object) with the K unit jsons in $MLT/units/C3; applied to the MLTS "
                   "inputs (rung mlts) for the E replay",
    "W-com-errors": "a committee optimisation that raises is retried once in place, then written as status "
                    "'pipeline_error' (no stalled member); a dead server writes no record and the member exits 1 "
                    "(infra retry); each ORCA run has a 2 h timeout (MLTS_W_TIMEOUT_S)",
    "W-final-gate": "gate M-1' (antisymmetry, train_atoms gate_m1 / gate_m1_sh) on 200 random training pairs of every "
                    "final atom fit (no test pairs exist); XGB is antisymmetric by construction (0.5 [h(z) - h(-dx, m)])",
    "W-com-start": "committee start = the main-engine A-2 TS of the reaction ($MLT/ade/jobs/<job_id>, the FT_k TS that "
                   "E scored) re-optimised by each of FT_all_s0..2 (A-1 OptTS, MLIP initial Hessian, Recalc "
                   "analytic:30, TightOpt); delta_com = max over the forming pairs of (max_j - min_j) of the three "
                   "optimised distances; a reaction with fewer than 3 ok members has no delta_com (counted)",
    "W-com-set": "committee reactions = the ends of the E common P_main para pairs (the pairs E graded); FT_all saw "
                 "these reactions' G data, so delta_com here is in-sample for the MLIP (recorded limit)",
    "W-threshold": "warning threshold = the 95th percentile of delta_com over the E reactions (a TS whose committee "
                   "spread is in the top 5 % of D1); pair delta = max of its two ends; reported with the error ratio "
                   "above / below it, Spearman rho per target and the MAE by delta quartile",
    "W-cost": "deployment cost per reaction = median A-4 core-h and GPU-min of the main engine (E geom table; the "
              "allocated cost of the GPU jobs, mlts_autode A2-cost) + median HF EDA core-h of the H records when "
              "present + 2 committee members (GPU and serial ORCA core-h); target <= 1 core-h + <= 3 GPU-min (spec 0-5)",
}
__doc__ += "\n" + "\n".join(f"  {k}: {v}" for k, v in DECISIONS.items())


def log(msg):
    print(f"[W {time.time() - T0:7.0f}s] {msg}", flush=True)


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T0)
    return WALL_H - (time.time() - t0) / 3600.0


def ch_dir(smoke=False):
    return W_DIR / ("smoke" if smoke else "channel")


# ================================================================ environments
def child(env, args):
    """this file with args in an inputs_mlts environment (fu2 | espley | reactot | d1 | feat) as a subprocess."""
    import inputs_mlts as IM
    sh = f'{IM.ENVS[env]} && {IM.env_common()} MLT_IN_ENV={env} && exec "$PY" "$@"'
    log(f"child ({env}): {' '.join(str(a) for a in args)}")
    return subprocess.run(["bash", "-c", sh, "_", str(THIS)] + [str(a) for a in args]).returncode


def mlts_env_cmd(args):
    """this file in MLTENV=mlts (jobs/env.sh) as a subprocess (the MLIP bridge needs the mlts env)."""
    cmd = ["bash", "-c", 'cd "$0" && export MLTENV=mlts && source jobs/env.sh && exec "$PY" "$@"', str(HERE), str(THIS)]
    return subprocess.run(cmd + [str(a) for a in args]).returncode


# ================================================================ config of the final channel model
def primary_config():
    """(K, primary name, config dict, REP8 {t: (arm, model)}) from K_report + E_report (E resolved the REP8)."""
    K = rj(K_REPORT)
    if not K or K.get("complete") is not True:
        M.not_ready(f"{K_REPORT} missing / incomplete")
    E = rj(E_REPORT)
    if not E or E.get("complete") is not True:
        M.not_ready(f"{E_REPORT} missing / incomplete (W after E)")
    prim = K.get("primary")
    import eval_v2 as EV2
    cfgs = EV2._cfg_list(K)
    if "C1" not in cfgs:
        cfgs["C1"] = dict(name="C1", refs=None, table=None, units_dir=None, app=False, rung=None,
                          predictions=str(M.MLT / "preds" / "C1_predictions.parquet"))
    if prim not in cfgs:
        M.stop(f"primary {prim} not in the K configs")
    r8 = ((E.get("representatives") or {}).get(prim) or {}).get("REP8")
    if not r8:
        M.stop(f"E_report has no REP8 for the primary {prim}")
    rep8 = {t: tuple(v) for t, v in r8.items()}
    c = dict(cfgs[prim])
    c["raw"] = (K.get("configs") or {}).get(prim) if isinstance(K.get("configs"), dict) else None
    return K, prim, c, rep8


def train_spec(prim, c):
    """(training rung, unit rung name, units dir) of the primary config: C1 = the LADDER dft units; C2 = rung mlts,
    units $MLT/units/C2 (train_k layout); C3 = rung mlts_mix (train_k make_mix_*: MLTS + label-TS inputs, same y),
    units $MLT/units/C3."""
    if prim.startswith("C1"):
        return "dft", "dft", M.LAD / "units"
    if prim.startswith("C2"):
        return "mlts", c.get("rung") or "mlts", Path(c.get("units_dir") or (M.MLT / "units" / prim))
    if prim.startswith("C3"):
        return "mlts_mix", c.get("rung") or "mlts_mix", Path(c.get("units_dir") or (M.MLT / "units" / prim))
    return None, None, None


def mix_data(kind, mod, arm):
    """W-final-C23: the C3 training data = train_k.make_mix_atoms / make_mix_xgb over rung mlts + rung dft (both rungs
    registered in this process); -> (mixed data, the MLTS data for the deployment replay)."""
    import train_k as TK
    TK._geo2_import()
    if kind == "atoms":
        import predict_atoms as PA
        TK.X.TA, TK.X.PA = mod, PA
        Dm, Dd = mod.get_data("mlts", arm), mod.get_data("dft", arm)
        return TK.make_mix_atoms(Dm, Dd), Dm
    TK.X.TX = mod
    Dm, Dd = mod.get_data("mlts", arm), mod.get_data("dft", arm)
    return TK.make_mix_xgb(Dm, Dd), Dm


def register(mod, kind, refs, app_root=None):
    """rung 'mlts' in train_atoms / train_xgb (predict_c1.register_mlts) with the H table of the refs mode."""
    import inputs_mlts as IM
    import predict_c1 as PC
    if app_root:
        IM._OPT["root"] = Path(app_root)
    S = IM.src_cfg("mlts")
    if not Path(S["table"]).is_file():
        return None
    reg = PC.register_mlts(mod, kind, "mlts")
    if refs == "min" and S.get("table_min") and Path(S["table_min"]).is_file():
        mod.GEO_INPUTS["mlts"] = S["table_min"]
        reg.update(table=str(S["table_min"]), table_sha256=M.sha256(S["table_min"]))
    return reg


def unit_recs(units_dir, rung_u, arm, model):
    recs = []
    for k in range(5):
        p = Path(units_dir) / f"L__{rung_u}__{arm}__{model}__f{k}.json"
        if not p.is_file():
            M.not_ready(f"{p} missing")
        recs.append(json.loads(p.read_text()))
    return recs


def mode_index(vals):
    """most frequent value; ties -> the smallest (grid order)."""
    from collections import Counter
    c = Counter(int(v) for v in vals)
    best = max(c.values())
    return min(v for v, n in c.items() if n == best)


def e_pairs():
    import pandas as pd
    if not E_PAIRS.is_file():
        return None
    return pd.read_csv(E_PAIRS, dtype={"pair_id": str})


def torch_save(path, obj):
    import torch
    path = M.assert_write_ok(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ================================================================ atom part (env_d1fu2, GPU)
def part_atoms(arm, model, rung, rung_u, units_dir, smoke=False, refs="min"):
    import inputs_mlts as IM
    IM.ensure_env("fu2")
    out = ch_dir(smoke)
    pj = out / f"part_{arm}_{model}.json"
    if pj.is_file():
        log(f"{pj.name} exists")
        return 0
    if str(LAD_DIR) not in sys.path:
        sys.path.insert(0, str(LAD_DIR))
    import numpy as np
    import pandas as pd
    import train_atoms as TA
    FT, Mo, F = TA.FT, TA.Mo, TA.F
    FT.require_gpu()
    reg = register(TA, "atoms", refs)
    if rung in ("mlts", "mlts_mix") and reg is None:
        M.not_ready("rung mlts table missing")
    Dmix = None
    if rung == "mlts_mix":
        D, Dmix = mix_data("atoms", TA, arm)                       # Dmix: the MLTS rung (E replay inputs)
    else:
        D = TA.get_data(rung, arm)
    recs = unit_recs(units_dir, rung_u, arm, model)
    gis = [int(r["chosen"]["grid_index"]) for r in recs]
    eps = [int(r["chosen"]["best_epoch"]) for r in recs]
    gi = mode_index(gis)
    cell = Mo.GRID[model][gi]
    if TA.norm(recs[gis.index(gi)]["chosen"]["cell"]) != TA.norm(cell):
        M.stop(f"{model} GRID[{gi}] != the unit json chosen cell")
    ep = int(np.median(eps))
    rows = np.unique(np.concatenate([D.split(k)["train"] for k in range(5)]))
    seeds = list(F.FINAL_SEEDS)
    if smoke:
        rows, ep, seeds = rows[:1500], 2, seeds[:1]
    fitf, gatef = TA.arm_fns(arm)
    sh = arm == "SH"
    gate_rows = np.random.default_rng(SEED).choice(rows, size=min(N_GATE, len(rows)), replace=False)
    # deployment path for P replay: the same fits on the E pairs with the MLTS inputs (rung mlts)
    Dg, rows_g, pid_g = None, None, None
    EP = e_pairs()
    if EP is not None:
        if rung == "mlts":
            Dg = D
        elif rung == "mlts_mix":
            Dg = Dmix
        elif register(TA, "atoms", refs) is not None:
            try:
                Dg = TA.get_data("mlts", arm)
            except SystemExit as e:                                 # noqa: PERF203  (subset tables: layout differs)
                log(f"rung mlts data not loadable for the E replay set ({e.code})")
                Dg = None
        if Dg is None:                                             # no MLTS inputs yet: the training rung (tests)
            Dg = D
        ok = [p for p in EP.pair_id if p in Dg.pid_row and Dg.usable_pair[Dg.pid_row[p]]]
        rows_g = np.array([Dg.pid_row[p] for p in ok], dtype=int)
        pid_g = ok
    lay = dict(elements=[int(z) for z in D.elements], bin_cols=list(D.bin_cols), num_cols=list(D.num_cols),
               glob_cols=list(D.trep["glob_cols"]), token_version=D.trep.get("token_version"),
               ext_used=bool(D.trep.get("ext_used")), h_cols=list(D.hinfo["h_cols"]), h_targets=list(H9))
    files, gates, P_tr, P_g = [], [], [], []
    for s in seeds:
        t1 = time.time()
        r = fitf(D, model, cell, TA.VARIANT, rows, None, seed=s, n_epochs=ep, log=None)
        g = gatef(r["model"], D, r["std"], gate_rows, seed=s)
        if not g["ok"]:
            M.stop(f"gate M-1' failed ({arm} {model} seed {s}): {g}")
        st = dict(mu=r["std"].mu, sd=r["std"].sd, gmu=r["std"].gmu, gsd=r["std"].gsd, sy=r["sy"])
        if sh:
            st.update(sr=r["sr"], beta=r["beta"])
        p = out / f"W__{rung}__{arm}__{model}__all__s{s - F.SEED}.pt"
        torch_save(p, dict(state=r["model"].state_dict(), cell=cell, std=st, rung=rung, arm=arm, model=model,
                           variant=TA.VARIANT, epochs=ep, seed=s, layout=lay, d_in=int(D.d_in), d_glob=int(D.d_glob),
                           TARGETS=TARGETS, n_train=int(len(rows)), smoke=bool(smoke), code=CODE_VERSION))
        files.append(dict(file=p.name, sha256=M.sha256(p), seed=s, seconds=time.time() - t1))
        gates.append(g)
        P_tr.append(TA.final_kcal(D, r, rows))
        if rows_g is not None and len(rows_g):
            T_g = r["std"].tensors(Dg)
            P_g.append(TA.final_kcal(Dg, dict(r, T=T_g), rows_g))
            del T_g
        log(f"{arm} {model} seed {s}: {ep} epochs on {len(rows)} pairs ({time.time() - t1:.0f} s), gate {g['ok']}")
        del r
    Ytr = np.stack(P_tr).mean(0)
    tmae = dict(zip(TARGETS, np.abs(Ytr - D.y[rows]).mean(0).tolist()))
    rep_e = None
    if P_g:
        Y = np.stack(P_g)
        beta = np.asarray(st["beta"], float) if sh else np.zeros(len(TARGETS))          # same pairs -> same beta
        base = beta * Dg.dH[rows_g] if sh else np.zeros((len(rows_g), len(TARGETS)))
        Fr = pd.concat([pd.DataFrame(dict(pair_id=pid_g, arm=arm, model=model, target=t, yhat=Y[:, :, j].mean(0),
                                          yhat_seed_sd=Y[:, :, j].std(0), base=base[:, j]))
                        for j, t in enumerate(TARGETS)], ignore_index=True)
        pq = out / f"final_on_E_{arm}_{model}.parquet"
        M.write_parquet(pq, Fr)
        rep_e = dict(path=str(pq), sha256=M.sha256(pq), n_pairs=len(pid_g), input_rung=Dg.rung)
    M.write_json(pj, dict(kind="atoms", arm=arm, model=model, rung=rung, unit_rung=rung_u, units_dir=str(units_dir),
                          cell=cell, grid_index=gi, fold_grid_index=gis, fold_best_epoch=eps, epochs=ep,
                          n_train=int(len(rows)), seeds=seeds, files=files, gates=gates, train_mae=tmae, layout=lay,
                          registered=reg, final_on_E=rep_e, smoke=bool(smoke), versions=FT.versions(),
                          written=M.now(), decisions={k: DECISIONS[k] for k in ("W-final-pairs", "W-final-cell",
                                                                               "W-final-gate")}))
    log(f"atoms part {arm} {model} -> {pj}")
    return 0


# ================================================================ XGB (env_espley) — shared with predict_v2
def xgb_mods():
    if str(LAD_DIR) not in sys.path:
        sys.path.insert(0, str(LAD_DIR))
    import train_xgb as TX
    for p in TX.WORKER_PATHS:
        if p not in sys.path:
            sys.path.insert(0, p)
    return TX, TX.D1T


def xgb_fit_rows(D, rows, cells, targets):
    """the LADDER final XGB fit (train_xgb.prep_fit + d1_train._xgb_fit) on pair-table rows `rows` with the grid index
    per target -> dict(models {t: est}, scale, beta, sy, base_train, kc_train)."""
    import numpy as np
    TX, D1T = xgb_mods()
    gate_rows = rows[:1]
    fin = TX.prep_fit(D, rows, dict(train=rows), gate_rows)
    sc = D1T.Scale(D.X[D.ra[rows]], D.X[D.rb[rows]], D.y[rows, 0])
    zd, zm = sc.z(D.X[D.ra[rows]], D.X[D.rb[rows]])
    if not (np.array_equal(zd, fin["zd"]) and np.array_equal(zm, fin["zm"])):
        M.stop("reconstructed d1_train.Scale does not reproduce prep_fit z")
    models, kc = {}, {}
    for t in targets:
        ti = TARGETS.index(t)
        est = D1T._xgb_fit(fin["zd"], fin["zm"], fin["ys"][ti], TX.GRID[int(cells[t])])
        models[t] = est
        kc[t] = np.asarray(D1T._xgb_g(est, zd, zm) * fin["sy"][ti], float)
    return dict(models=models, scale=sc, beta=np.asarray(fin["beta"], float), sy=list(fin["sy"]),
                base_train=fin["base"]["train"], kc_train=kc, fin_scale=fin["scale"])


def xgb_predict(est, sc, beta, sy, t, Xa, Xb, dH, arm):
    """kcal prediction of one target for reaction feature rows Xa, Xb (+ the SH baseline beta * dH)."""
    import numpy as np
    TX, D1T = xgb_mods()
    zd, zm = sc.z(Xa, Xb)
    ti = TARGETS.index(t)
    kc = np.asarray(D1T._xgb_g(est, zd, zm) * sy[ti], float)
    on = arm == "SH" and ti in TX.H_IDX
    return kc + (beta[ti] * dH[:, ti] if on else 0.0)


def save_scale(path, sc):
    import numpy as np
    path = M.assert_write_ok(Path(path))
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, sd=sc.sd, mu=sc.mu, s=sc.s, kd=sc.kd, km=sc.km, sy=np.array([sc.sy]))
    os.replace(tmp, path)


def load_scale(path):
    import numpy as np
    TX, D1T = xgb_mods()
    z = np.load(path)
    sc = D1T.Scale.__new__(D1T.Scale)
    sc.sd, sc.mu, sc.s, sc.kd, sc.km, sc.sy = z["sd"], z["mu"], z["s"], z["kd"].astype(bool), z["km"].astype(bool), \
        float(z["sy"][0])
    return sc


def part_xgb(arm, rung, rung_u, units_dir, targets, smoke=False, refs="min"):
    import inputs_mlts as IM
    IM.ensure_env("espley")
    out = ch_dir(smoke)
    pj = out / f"part_{arm}_XGB.json"
    if pj.is_file():
        log(f"{pj.name} exists")
        return 0
    import numpy as np
    import pandas as pd
    TX, D1T = xgb_mods()
    reg = register(TX, "xgb", refs)
    if rung in ("mlts", "mlts_mix") and reg is None:
        M.not_ready("rung mlts table missing")
    Dmix = None
    if rung == "mlts_mix":
        D, Dmix = mix_data("xgb", TX, arm)
    else:
        D = TX.get_data(rung, arm)
    recs = unit_recs(units_dir, rung_u, arm, "XGB")
    targets = [t for t in targets.split(",") if t]
    if smoke:
        targets = targets[:1]
    cells = {t: mode_index([r["chosen"][t]["grid_index"] for r in recs]) for t in targets}
    rows = np.unique(np.concatenate([D.split(k)["train"] for k in range(5)]))
    if smoke:
        rows = rows[:1500]
    t1 = time.time()
    fit = xgb_fit_rows(D, rows, cells, targets)
    files, tmae = {}, {}
    for t in targets:
        p = out / f"W__{rung}__{arm}__XGB__all__{t}.json"
        tmp = p.with_name(f".{p.stem}.tmp{os.getpid()}.json")
        M.assert_write_ok(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fit["models"][t].save_model(str(tmp))
        os.replace(tmp, p)
        files[t] = dict(file=p.name, sha256=M.sha256(p), grid_index=cells[t], cell=TX.GRID[cells[t]],
                        fold_grid_index=[int(r["chosen"][t]["grid_index"]) for r in recs])
        ti = TARGETS.index(t)
        tmae[t] = float(np.abs(fit["base_train"][:, ti] + fit["kc_train"][t] - D.y[rows, ti]).mean())
    sp = out / f"W__{rung}__{arm}__XGB__all__scale.npz"
    save_scale(sp, fit["scale"])
    rep_e = None
    EP = e_pairs()
    if EP is not None:
        Dg = D if rung == "mlts" else Dmix if rung == "mlts_mix" else (
            TX.get_data("mlts", arm) if register(TX, "xgb", refs) is not None else None)
        Dg = D if Dg is None else Dg                               # no MLTS inputs yet: the training rung (tests)
        if Dg is not None:
            ok = [p for p in EP.pair_id if p in Dg.pid_row and Dg.usable_pair[Dg.pid_row[p]]]
            rg = np.array([Dg.pid_row[p] for p in ok], dtype=int)
            if list(Dg.feat_cols) != list(D.feat_cols):
                M.stop("XGB feature columns of rung mlts differ from the training rung")
            fr = []
            for t in targets:
                yh = xgb_predict(fit["models"][t], fit["scale"], fit["beta"], fit["sy"], t, Dg.X[Dg.ra[rg]],
                                 Dg.X[Dg.rb[rg]], Dg.dH[rg], arm)
                fr.append(pd.DataFrame(dict(pair_id=ok, arm=arm, model="XGB", target=t, yhat=yh, yhat_seed_sd=0.0)))
            pq = out / f"final_on_E_{arm}_XGB.parquet"
            M.write_parquet(pq, pd.concat(fr, ignore_index=True))
            rep_e = dict(path=str(pq), sha256=M.sha256(pq), n_pairs=len(ok), input_rung=Dg.rung)
    M.write_json(pj, dict(kind="xgb", arm=arm, model="XGB", rung=rung, unit_rung=rung_u, units_dir=str(units_dir),
                          targets=targets, files=files, scale=dict(file=sp.name, sha256=M.sha256(sp)),
                          beta=fit["beta"].tolist(), sy=fit["sy"], feat_cols=list(D.feat_cols), n_train=int(len(rows)),
                          train_mae=tmae, registered=reg, final_on_E=rep_e, smoke=bool(smoke), seconds=time.time() - t1,
                          versions=TX.versions(), written=M.now(),
                          decisions={k: DECISIONS[k] for k in ("W-final-pairs", "W-final-cell", "W-final-gate")}))
    log(f"XGB part {arm} {targets} -> {pj} ({time.time() - t1:.0f} s)")
    return 0


# ================================================================ channel (orchestrator, any env)
def cmd_channel(smoke=False):
    K, prim, c, rep8 = primary_config()
    out = ch_dir(smoke)
    card = out / "MODEL_CARD.json"
    if card.is_file() and not smoke:
        log(f"{card} exists")
        return 0
    raw = c.get("raw") or {}
    if isinstance(raw.get("final"), dict):                         # a K-trained final model (not produced today)
        parts = raw["final"]
        rung = "mlts"
    else:
        rung, rung_u, units_dir = train_spec(prim, c)
        if rung is None:
            M.stop(f"primary {prim}: no final-model rule (C1 / C2 / C3)")
        refs = c.get("refs") or "min"
        combos = sorted({tuple(v) for v in rep8.values()})
        if smoke:
            combos = [x for x in combos if x[1] != "XGB"][:1] + [x for x in combos if x[1] == "XGB"][:1]
        for a, m in combos:
            if m == "XGB":
                tg = ",".join(t for t in TARGETS if tuple(rep8[t]) == (a, "XGB"))
                rc = child("espley", ["_xgb", a, rung, rung_u, units_dir, tg, refs] + (["--smoke"] if smoke else []))
            else:
                rc = child("fu2", ["_atoms", a, m, rung, rung_u, units_dir, refs] + (["--smoke"] if smoke else []))
            if rc != 0:
                return rc
        parts = {f"{a}|{m}": str(out / f"part_{a}_{m}.json") for a, m in combos}
    recs = {k: rj(v) for k, v in parts.items()}
    miss = [k for k, v in recs.items() if not v]
    if miss:
        M.not_ready(f"parts missing: {miss}")
    thr = applicability(c, rep8)
    M.write_json(card, dict(
        phase="DEPLOY-V2 W-1 final channel model", config=prim, trained_rung=rung, refs=c.get("refs") or "min",
        REP8={t: list(v) for t, v in rep8.items()}, disp="D3_int (H-0; the model disp is kept as a fallback)",
        parts=recs, part_files=parts, applicability=thr, e_grade=(rj(E_REPORT) or {}).get("grade"),
        smoke=bool(smoke), decisions=DECISIONS, written=M.now(), code=CODE_VERSION))
    log(f"MODEL_CARD -> {card}")
    return 0


def applicability(c, rep8):
    """out-of-range rule (DEPLOY E_ad precedent): per target the 95th percentile of the primary config's test-pair
    yhat_seed_sd of its REP8 choice (atom models only; XGB has one fit = no seed spread)."""
    try:
        import numpy as np
        import pandas as pd
        P = pd.read_parquet(c["predictions"], columns=["arm", "model", "target", "yhat_seed_sd"])
    except Exception as e:                                          # noqa: BLE001
        return dict(note=f"no thresholds ({type(e).__name__})")
    th = {}
    for t in TARGETS:
        a, m = rep8[t]
        q = P[(P.arm == a) & (P.model == m) & (P.target == t)].yhat_seed_sd.to_numpy(float)
        th[t] = float(np.percentile(q, THR_PCT)) if m != "XGB" and len(q) else None
    return dict(rule=f"{THR_PCT:.0f}th percentile of the test yhat_seed_sd of the REP8 model per target (atom models; "
                     "XGB: none); out_of_range = yhat_seed_sd > threshold or an element outside the 9", thresholds=th)


# ================================================================ committee (mlts env, GPU)
def committee_reactions(limit=None, ids=None):
    if ids:                                                         # tests: explicit rxn ids
        return sorted(int(x) for x in str(ids).split(",") if x)
    EP = e_pairs()
    if EP is None:
        M.not_ready(f"{E_PAIRS} missing (E first)")
    q = EP[EP.para_main.astype(str).str.lower() == "true"]
    rx = sorted(set(q.a_rxn.astype(int)) | set(q.b_rxn.astype(int)))
    return rx[:int(limit)] if limit else rx


def start_ts(rxn, root=None):
    """(symbols, pos, formed pairs) of the main-engine A-2 TS, or None."""
    d = Path(root or M.MLT) / "ade" / "jobs" / M.job_of(rxn)
    tr = rj(d / "ts_result.json")
    if not (tr and (d / "ts.xyz").is_file() and (d / ".done_ts").is_file()):
        return None
    s, p = M.read_xyz(d / "ts.xyz")
    pairs = [tuple(int(x) for x in q) for q in tr.get("formed_pairs_ts") or []]
    return (s, p, pairs) if pairs else None


def opt_records(model, rxns, out_dir, starts, workers=None, log_every=1):
    """re-optimise each reaction's TS with one MLTS member (mlts_bridge server) -> recs/<rxn>.json; returns counts."""
    import numpy as np
    import mlts_bridge as B
    spec = B.resolve_spec(model)
    rec_dir = M.assert_write_ok(Path(out_dir) / "recs")
    rec_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rxns if not (rec_dir / f"{r}.json").is_file()]
    cnt = dict(todo=len(todo), done=0, errors=0)
    if not todo:
        return cnt, True
    h = B.start_server(spec)
    work_root = B.local_tmp() / f"mlts_wcom_{B.jobtag()}_{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=True)
    mu, it, st = threading.Lock(), iter(todo), dict(halt=False, timed_out=False, server_dead=False)

    def alive():
        return h.proc is None or h.proc.poll() is None

    def attempt(rx, t1):
        rec = dict(rxn_id=int(rx), model=B.spec_name(spec), time=M.now())
        s0 = starts.get(rx)
        if s0 is None:
            rec.update(status="no_start")
        elif not B.covers(spec, s0[0]):
            rec.update(status="not_covered")
        else:
            syms, pos, pairs = s0
            shutil.rmtree(work_root / str(rx), ignore_errors=True)
            r = B.optimize_ts(spec, syms, pos, pairs, work_root / str(rx), h.socket, tight=True,
                              timeout_s=COM_TIMEOUT_S)
            po = np.asarray(r["pos_A"], float)
            rec.update({k: r.get(k) for k in ("status", "converged", "n_cycles", "n_imag", "imag_cm", "gpu_s",
                                              "n_ext_calls", "gmax_final_eh_bohr")})
            rec.update(pairs=[list(p) for p in pairs], d_start=B.dists(pos, pairs), d_opt=B.dists(po, pairs),
                       pos_A=np.round(po, 6).tolist(), symbols=list(syms))
        rec["wall_s"] = time.time() - t1
        return rec

    def loop():
        while True:
            with mu:
                if st["halt"] or time_left_h() < RESERVE_H:
                    st["timed_out"] = st["timed_out"] or time_left_h() < RESERVE_H
                    return
                if not alive():                                     # W-com-errors: a dead server stops the member
                    st["halt"], st["server_dead"] = True, True
                    return
                rx = next(it, None)
            if rx is None:
                return
            t1 = time.time()
            rec, err = None, None
            try:
                for _ in (1, 2):                                    # W-com-errors: one retry in place
                    try:
                        rec = attempt(rx, t1)
                        break
                    except Exception as e:                          # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"[:400]
                        print(f"[W] committee error {rx}: {err}\n{traceback.format_exc(limit=4)}", file=sys.stderr,
                              flush=True)
                        if not alive():
                            break
                if rec is None and alive():                         # closed as a record: never stalls the member
                    rec = dict(rxn_id=int(rx), model=B.spec_name(spec), time=M.now(), status="pipeline_error",
                               error=err, wall_s=time.time() - t1)
                if rec is not None:
                    M.write_text(rec_dir / f"{rx}.json", json.dumps(rec, default=M._json_default) + "\n")
                    with mu:
                        cnt["done"] += 1
                    if log_every:
                        log(f"{model} {rx}: {rec.get('status')} conv {rec.get('converged')} ({rec['wall_s']:.0f} s)")
                if err:
                    with mu:
                        cnt["errors"] += 1
                        if cnt["errors"] > MAX_ERR:
                            st["halt"] = True
            finally:
                shutil.rmtree(work_root / str(rx), ignore_errors=True)
    nw = int(workers or os.environ.get("MLT_WORKERS") or max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or 4) - 2))
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max(1, min(nw, len(todo)))) as ex:
            for f in [ex.submit(loop) for _ in range(max(1, min(nw, len(todo))))]:
                f.result()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        h.shutdown()
    cnt["halt"], cnt["timed_out"], cnt["server_dead"] = st["halt"], st["timed_out"], st["server_dead"]
    left = [r for r in rxns if not (rec_dir / f"{r}.json").is_file()]
    return dict(cnt, left=len(left)), not left


def members(models=None):
    """committee members (W-2) = the FT-fin runs FT_all_s<i> trained --with-al (finetune.parse_run -> FT_all_s<i>_al);
    MLTS_W_MEMBERS overrides the default for tests only."""
    dflt = os.environ.get("MLTS_W_MEMBERS") or "FT_all_s0_al,FT_all_s1_al,FT_all_s2_al"
    return [x for x in (models or dflt).split(",") if x]


def model_available(model):
    """(ok, why): a fine-tuned spec resolves to a weights path even before finetune.py wrote it -> check the file."""
    import mlts_bridge as B
    try:
        spec = B.resolve_spec(model)
    except Exception as e:                                          # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    w = spec.get("weights")
    if w and not Path(w).is_file():
        return False, f"weights {w} missing"
    return True, ""


def cmd_committee(i, n=3, models=None, limit=None, ids=None):
    if os.environ.get("MLTENV") != "mlts":
        return mlts_env_cmd(["committee", str(i), str(n)] + (["--models", models] if models else [])
                            + (["--limit", str(limit)] if limit else []) + (["--ids", ids] if ids else []))
    i = M.array_task(i)
    mem = members(models)
    if len(mem) != int(n) or not 0 <= i < len(mem):
        raise SystemExit(f"committee {i} of {n}: members {mem}")
    model = mem[i]
    ok, why = model_available(model)
    if not ok:
        M.not_ready(f"committee member {model} not available ({why})")
    rx = committee_reactions(limit, ids)
    starts = {r: start_ts(r) for r in rx}
    cnt, complete = opt_records(model, rx, W_DIR / "committee" / model, starts)
    log(f"committee {model}: {len(rx)} reactions, {cnt}")
    if cnt.get("server_dead"):
        return 1                                                    # infra: the launcher retries once
    if cnt.get("halt"):
        return 3
    return 0 if complete else 2


# ================================================================ report (reactot)
def delta_com(models=None, rxns=None, base=None):
    """{rxn: delta_com (A) or None}, per-member status counts."""
    import numpy as np
    base = Path(base or (W_DIR / "committee"))
    mem = members(models)
    out, st = {}, {}
    for r in rxns:
        ds, ok = [], True
        for m in mem:
            J = rj(base / m / "recs" / f"{r}.json")
            s = (J or {}).get("status", "missing")
            st[s] = st.get(s, 0) + 1
            if s != "ok" or not J.get("d_opt"):
                ok = False
                continue
            ds.append(np.asarray(J["d_opt"], float))
        out[r] = float(np.max(np.ptp(np.stack(ds), axis=0))) if ok and len(ds) == len(mem) else None
    return out, st


def cmd_report(models=None, limit=None, ids=None, smoke=False):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    card = rj(ch_dir(smoke) / "MODEL_CARD.json")
    if not card:
        M.not_ready("MODEL_CARD.json missing (final_w.py channel)")
    rx = committee_reactions(limit, ids)
    mem = members(models)
    have = {m: sum(1 for r in rx if (W_DIR / "committee" / m / "recs" / f"{r}.json").is_file()) for m in mem}
    if any(v < len(rx) for v in have.values()):
        M.not_ready(f"committee records incomplete: {have} of {len(rx)}")
    dc, st = delta_com(models, rx)
    v = np.array([x for x in dc.values() if x is not None], float)
    thr = float(np.percentile(v, THR_PCT)) if len(v) else None
    M.write_csv(W_DIR / "delta_com.csv", pd.DataFrame(dict(rxn_id=list(dc), delta_com=list(dc.values()))))
    E = pd.read_parquet(E_ERR)
    E = E[E.para_main.astype(bool)]
    E["dc_a"] = E.a_rxn.map(dc).astype(float)
    E["dc_b"] = E.b_rxn.map(dc).astype(float)
    E["dc_pair"] = np.fmax(E.dc_a.to_numpy(float), E.dc_b.to_numpy(float))
    rel = {}
    for t in PRIMARY9:
        q = E[(E.target == t) & np.isfinite(E.dc_pair)]
        ae = np.abs(q.err_mlts.to_numpy(float))
        x = q.dc_pair.to_numpy(float)
        r = dict(n=int(len(q)), spearman_rho=float(spearmanr(x, ae).statistic) if len(q) > 2 else None)
        if len(q) >= 8:
            qs = np.percentile(x, [25, 50, 75])
            b = np.digitize(x, qs)
            r["mae_by_quartile"] = [float(ae[b == k].mean()) if (b == k).any() else None for k in range(4)]
            r["quartile_edges_A"] = qs.tolist()
        if thr is not None and len(q):
            hi = x > thr
            r.update(n_above=int(hi.sum()), mae_above=float(ae[hi].mean()) if hi.any() else None,
                     mae_below=float(ae[~hi].mean()) if (~hi).any() else None)
            r["ratio_above_below"] = (r["mae_above"] / r["mae_below"]) if r.get("mae_above") and r.get("mae_below") \
                else None
        rel[t] = r
    cost = {}
    try:
        import eval_v2 as EV2
        gp, _ = EV2.geom_paths("main")
        Gt = pd.read_csv(gp)

        def cst(col):                                               # allocated when logged (A2-cost), else own
            own = pd.to_numeric(Gt[col], errors="coerce")
            return pd.to_numeric(Gt[f"{col}_alloc"], errors="coerce").fillna(own) if f"{col}_alloc" in Gt else own
        cost.update(a_core_h_median=float(cst("core_h").median()), a_gpu_min_median=float(cst("gpu_min").median()),
                    a_core_h_own_median=float(pd.to_numeric(Gt.core_h, errors="coerce").median()))
    except Exception as e:                                          # noqa: BLE001
        cost["a_note"] = f"no E geom table ({type(e).__name__})"
    hf = M.MLT / "hf" / "mlts" / "hf_records.parquet"
    if hf.is_file():
        H = pd.read_parquet(hf)
        col = next((c for c in ("coreh_total", "core_h", "core_h_total") if c in H.columns), None)
        cost["hf_core_h_median"] = float(H[col].median()) if col else None
    com = [rj(W_DIR / "committee" / m / "recs" / f"{r}.json") or {} for m in mem for r in rx]
    com_g = [float(x.get("gpu_s") or 0.0) for x in com]
    com_c = [float(x.get("wall_s") or 0.0) / 3600.0 for x in com]   # one serial ORCA ExtOpt worker per optimisation
    cost["committee_gpu_min_median_per_member"] = float(np.median(com_g)) / 60.0 if com_g else None
    cost["committee_core_h_median_per_member"] = float(np.median(com_c)) if com_c else None
    tot_c = (cost.get("a_core_h_median") or 0.0) + (cost.get("hf_core_h_median") or 0.0) + \
        2 * (cost.get("committee_core_h_median_per_member") or 0.0)
    tot_g = (cost.get("a_gpu_min_median") or 0.0) + 2 * (cost.get("committee_gpu_min_median_per_member") or 0.0)
    cost.update(total_core_h=tot_c, total_gpu_min=tot_g, target=dict(core_h=1.0, gpu_min=3.0),
                within_target=bool(tot_c <= 1.0 and tot_g <= 3.0),
                rule=DECISIONS["W-cost"] + "; committee adds 2 members' GPU time and serial ORCA core-h (member 0 = "
                                           "the deployed TS search)")
    rep = dict(phase="DEPLOY-V2 W (spec 13 W-1, W-2)", complete=True, n_todo=0, stop=None,
               est=dict(units_total=3, unit_wall_h=None, workers=1),
               channel=dict(config=card.get("config"), trained_rung=card.get("trained_rung"), REP8=card.get("REP8"),
                            parts={k: dict(kind=v.get("kind"), cell=v.get("cell"), epochs=v.get("epochs"),
                                           n_train=v.get("n_train"), train_mae=v.get("train_mae"),
                                           smoke=v.get("smoke")) for k, v in (card.get("parts") or {}).items()},
                            applicability=card.get("applicability"), card=str(ch_dir(smoke) / "MODEL_CARD.json"),
                            smoke=bool(smoke)),
               committee=dict(members=mem, n_reactions=len(rx), n_with_delta=int(len(v)), status_counts=st,
                              delta_com_quantiles_A={p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
                              if len(v) else None, warn_threshold_A=thr, rule=DECISIONS["W-threshold"],
                              table=str(W_DIR / "delta_com.csv")),
               relation=rel, cost=cost, decisions=DECISIONS, written=M.now(), code=CODE_VERSION)
    M.write_json(W_REPORT, rep)
    log(f"W report: threshold {thr}, n {len(v)} -> {W_REPORT}")
    return 0


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="final_w.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("channel")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("committee")
    p.add_argument("i")
    p.add_argument("n", nargs="?", default="3")
    p.add_argument("--models", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None, help="tests: comma-separated rxn ids instead of the E reactions")
    p = sub.add_parser("report")
    p.add_argument("--models", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("--smoke", action="store_true", help="tests: the smoke MODEL_CARD")
    p = sub.add_parser("_atoms")
    for a in ("arm", "model", "rung", "rung_u", "units_dir", "refs"):
        p.add_argument(a)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("_xgb")
    for a in ("arm", "rung", "rung_u", "units_dir", "targets", "refs"):
        p.add_argument(a)
    p.add_argument("--smoke", action="store_true")
    o = ap.parse_args(argv)
    if o.cmd == "channel":
        return cmd_channel(o.smoke)
    if o.cmd == "committee":
        return cmd_committee(o.i, o.n, o.models, o.limit, o.ids)
    if o.cmd == "report":
        return cmd_report(o.models, o.limit, o.ids, o.smoke)
    if o.cmd == "_atoms":
        return part_atoms(o.arm, o.model, o.rung, o.rung_u, o.units_dir, o.smoke, o.refs)
    if o.cmd == "_xgb":
        return part_xgb(o.arm, o.rung, o.rung_u, o.units_dir, o.targets, o.smoke, o.refs)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
