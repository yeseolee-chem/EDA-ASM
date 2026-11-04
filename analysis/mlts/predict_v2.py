#!/usr/bin/env python3
r"""predict_v2.py — DEPLOY-V2 (MLTS) Phase P (docs/specs/DEPLOY_V2_MLTS.md 13 P-1 .. P-3): the predictor
SMILES -> autodE-MLTS TS -> inputs -> channel model -> 8 channels + barrier + flags, its self-check and the optional
DFT certificate. Compute nodes only; each part re-executes itself in the environment it needs.

  python predict_v2.py selfcheck [--n-pairs 10] [--n-new 5] [--cores 2] [--workers 6] [--skip-e2e]
        P-2 (mlts env, GPU). (R1) fold replay: the deployment predictor with the primary config's FOLD models on the
        production MLTS inputs of 10 E pairs (20 D1 reactions, disjoint, 2 per dipole, one fold) must reproduce the E
        predictions ($MLT/preds/<primary>_predictions.parquet: atoms <= 1e-4 kcal/mol, XGB refit <= 1e-6);
        (R2) final replay: the predictor with the W final models on the same inputs = final_w's own E-pair predictions
        (final_on_E, <= 1e-4); (E2E) the 20 D1 reactions + 5 SMILES of D1 manifest rows that are not D1 targets (one
        per dipole) from SMILES: mlts_autode.run_one (FT_all_s0, the A-3 engine) -> the inputs_mlts chain on
        $MLT/p/selfcheck (the H-phase commands, targets = this list) -> committee members FT_all_s1/s2 from that TS
        (delta_com) -> the final channel model -> pairs (D1: the 10 pairs; new: (new, its family reference of the
        same core / series)) -> 8 channels + barrier (pair dd and absolute = reference label + dd) + flags (TS gate,
        n_imag, delta_com > W threshold, out of range). -> $MLT/p/selfcheck/{selfcheck.json, predictions.parquet}.
        Exit 0 written (pass or not: P_report.selfcheck_pass), 2 not ready.
  python predict_v2.py cert <i|@I> <n> [--n-pairs 100]
        P-3 (reactot, CPU): label-level EnGrad (mlt_common.orca_grad_input = the D1 'grad' keywords, serial ORCA) at
        the main-engine MLTS TS of the ends of 100 seeded E P_main para pairs (<= 200 reactions), slice i of n ->
        $MLT/p/cert/recs/<rxn_id>.json (max force component / atomic norm, Eh/bohr). Exit 0 / 2 (unfinished).
  python predict_v2.py predict --jobs <job_id>:<ref_rxn>,... | --job-json f [--tag T]
        P-1 (mlts env, GPU): SMILES -> autodE-MLTS (FT_all_s0_al) -> inputs -> final channel model -> 8 channels +
        barrier + flags (TS gate, n_imag, delta_com, out of range) -> $MLT/p/run/<T>/{e2e.json, predictions.parquet}
  python predict_v2.py report    -> results/P_report.json {complete, selfcheck_pass, selfcheck, cert{force
                                    quantiles, relation to the E channel errors}}; exit 0 / 2.
  python predict_v2.py replay --src dft|mlts [--n 20] [--fold k]
        the predictor with fold models on rung inputs vs the reference predictions (dft: LADDER L_predictions =
        the predictor test on label-TS inputs; mlts: = R1). -> $MLT/p/replay_<src>.json
  internal: _pred <job.json> (atoms: env_d1fu2; xgb: env_espley), _im <inputs_mlts args> (inputs_mlts with the
  targets of <root>/targets.csv, in the env each H command needs).

Decisions (for DECISIONS.md; also in P_report.json)"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
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

CODE_VERSION = "mlts-p-1"
REPO = M.REPO
P_DIR = M.MLT / "p"
SC_DIR = P_DIR / "selfcheck"
CERT_DIR = P_DIR / "cert"
P_REPORT = M.RES / "P_report.json"
K_REPORT = M.RES / "K_report.json"
E_REPORT = M.RES / "E_report.json"
W_REPORT = M.RES / "W_report.json"
CARD = M.MLT / "w" / "channel" / "MODEL_CARD.json"
ARMS_JSON = M.MLT / "pipeline" / "arms.json"
E_PAIRS = M.MLT / "eval" / "E_pairs.csv"
E_ERR = M.MLT / "eval" / "E_pair_errors.parquet"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
TARGETS = list(M.TARGETS)
H9 = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
H_IDX = [TARGETS.index(t) for t in H9]
SEED = 20261104
TOL_ATOM, TOL_XGB = 1e-4, 1e-6
FMAX_THR = 0.002                                   # Eh/bohr (G-4 active-learning threshold)
WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
RESERVE_H = float(os.environ.get("MLT_RESERVE_H", "2"))
LABEL_KEYS = dict(d1="d1_kcal", d2="d2_kcal", elst="elst_dft", pauli="pauli_dft", oi="oi_dft", disp="disp_dft",
                  cpcm="cpcm_dft", cds="cds_dft", barrier="barrier_kcal")
IM_ENV = dict(finalize="d1", meta="d1", x182="feat", x182_merge="feat", atoms="feat", tokens="feat",
              hf_inputs="reactot", hf_run="reactot", hf_build="reactot", d3="reactot", build="reactot", h2="reactot")
T0 = time.time()

DECISIONS = {
    "P-replay": "'E 예측과 일치' is checked where it can be exact: (R1) the deployment predictor (token npz + X182 + H "
                "table -> fu2 models.build + the saved std; XGB = the LADDER final fit refitted with the fold unit's "
                "cells) with the primary config's fold models on the production MLTS inputs vs the E prediction rows "
                "(atoms <= 1e-4 kcal/mol = the DP-2 fp32 bound, XGB <= 1e-6); (R2) the same predictor with the W "
                "final models vs final_w's own predictions on those pairs (<= 1e-4)",
    "P-pairs": "10 E common P_main para pairs with 20 distinct reactions, 2 per dipole, from the fold holding most E "
               "pairs (one XGB refit per arm), seed 20261104; disjoint reactions so the E2E run has 20 TSs",
    "P-new": "'D1에 없는 SMILES 5개' = D1 manifest rows that are not among the 9,266 D1 targets (no label: failed or "
             "excluded in the D1 label run) and whose rxn_smiles is not a target's, one per dipole (seeded); each is "
             "paired with its family reference (the most frequent S_folds reference b_rxn of the same core and series), "
             "whose inputs are the production MLTS inputs; absolute values = the reference's label + the predicted dd",
    "P-chain": "inputs on $MLT/p/selfcheck through the unchanged inputs_mlts H commands; inputs_mlts.targets is replaced "
               "in-process by the selfcheck list (the G2 pattern: import, replace one function) because the H code "
               "accepts only the 9,266 D1 targets; refs = the primary config's refs mode, computed on this small set "
               "(min over its own reactions; production min uses all 9,266 -> the D1 E2E values are descriptive)",
    "P-flags": "TS gate = the three A-2 stage markers; n_imag = MLIP modes < -50 cm-1 (ts_result imag_freqs_cm); "
               "delta_com = committee spread (P-com: FT_all_s0..2_al re-optimising the deployed TS) > W warn_threshold; "
               "out_of_range = yhat_seed_sd > the MODEL_CARD applicability threshold or an element outside the 9",
    "P-pass": "selfcheck_pass = R1 pass and R2 pass and every one of the 25 reactions reached a terminal status (TS "
              "fail / input fail / predicted) without a pipeline error and every reaction with an ok TS and inputs was "
              "predicted (finite) and >= 1 of the 5 new SMILES was predicted (P-pass2)",
    "P-cert": "P-3 run (recommended): 100 seeded E P_main para pairs, their ends (<= 200 reactions) at the main-engine "
              "TS; force = -gradient of the label-level EnGrad; the threshold and the relation use the max per-atom "
              "|F| (the G-4 selection rule), the max component quantiles reported too; Spearman rho of the pair max "
              f"force (max of the ends) with |error| per target, MAE above / below {FMAX_THR} Eh/bohr",
    "P-pass2": "the new SMILES must run to the end: every one reaches a terminal TS status and at least one of them is "
               "predicted (finite 9 targets) — a self-check where every new reaction fails its TS search does not pass",
    "P-engine": "engine A' (G2 TS -> MLTS) needs a G2 xTB TS; a reaction without $G2/g2_final/<rxn>/.done (every new "
                "SMILES) runs with engine A in the predictor (recorded per reaction); A'' runs as A (no label-level "
                "refine in P)",
    "P-com": "delta_com in P = the W-2 definition: all three FT_all_s<i>_al re-optimise the deployed TS "
             "(final_w.delta_com), compared with the W warn threshold",
    "P-1": "`predict --jobs <job_id>:<ref_rxn_id>,...` / `--job-json <file>` = the SMILES -> TS -> inputs -> 8 channels "
           "+ barrier + flags chain (the selfcheck E2E code) on $MLT/p/run/<tag>: manifest-format rows (rxn_smiles, "
           "kind, charge, ...) paired with a production reference reaction (absolute = reference label + dd)",
}
__doc__ += "\n" + "\n".join(f"  {k}: {v}" for k, v in DECISIONS.items())


def log(msg):
    print(f"[P {time.time() - T0:7.0f}s] {msg}", flush=True)


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T0)
    return WALL_H - (time.time() - t0) / 3600.0


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def child(env, args, extra_env=None):
    import inputs_mlts as IM
    sh = f'{IM.ENVS[env]} && {IM.env_common()} MLT_IN_ENV={env} && exec "$PY" "$@"'
    e = dict(os.environ, **(extra_env or {}))
    e.pop("MLT_IN_ENV", None)
    log(f"child ({env}): {' '.join(str(a) for a in args)[:200]}")
    return subprocess.run(["bash", "-c", sh, "_", str(THIS)] + [str(a) for a in args], env=e).returncode


# ================================================================ sources and configs
def src_paths(kind, root=None, refs="min"):
    """input files of a source: dft (LADDER label-TS inputs) or mlts (production root or a selfcheck root)."""
    if kind == "dft":
        return dict(name="dft", tokens=str(M.LAD / "fu2" / "tokens"), x182=str(M.LAD / "X182_d1_dft.parquet"),
                    table=str(M.LAD / "inputs" / "dft.parquet"), t_report=str(LAD_RES / "fu2" / "T_report.json"))
    R = Path(root or M.MLT)
    rep = M.RES if R == M.MLT else R / "reports"
    tab = R / "inputs" / ("mlts_min.parquet" if refs == "min" else "mlts.parquet")
    if not tab.is_file():
        tab = R / "inputs" / "mlts.parquet"
    return dict(name=f"mlts:{R}", tokens=str(R / "fu2_mlts" / "tokens"), x182=str(R / "inputs" / "X182_mlts.parquet"),
                table=str(tab), t_report=str(rep / "fu2_mlts" / "T_report.json"))


def primary():
    """(primary name, config, REP8, refs) from K_report / E_report."""
    import eval_v2 as EV2
    K = rj(K_REPORT)
    E = rj(E_REPORT)
    if not (K and K.get("complete") and E and E.get("complete")):
        M.not_ready("K_report / E_report missing or incomplete")
    prim = K["primary"]
    cfgs = EV2._cfg_list(K)
    c = cfgs.get(prim) or dict(name=prim, predictions=str(M.MLT / "preds" / f"{prim}_predictions.parquet"), refs=None)
    rep8 = {t: tuple(v) for t, v in E["representatives"][prim]["REP8"].items()}
    return prim, c, rep8, c.get("refs") or "min"


def fold_unit_files(prim, c, arm, model, k):
    """the fold-k unit model files of the primary config (C1 = LADDER dft units; C2 / C3 = the train_k units
    $MLT/units/<config>/L__<rung>__..., C3 rung mlts_mix)."""
    if prim.startswith("C1"):
        d, ru = M.LAD / "units", "dft"
    else:
        d = Path(c.get("units_dir") or (M.MLT / "units" / prim))
        ru = c.get("rung") or ("mlts_mix" if prim.startswith("C3") else "mlts")
    base = d / f"L__{ru}__{arm}__{model}__f{k}"
    return base, [str(base) + f"__s{j}.pt" for j in range(5)]


# ================================================================ predictor parts (children)
def part_job(kind, arm, model, mode, sources, pairs, out, **kw):
    return dict(kind=kind, arm=arm, model=model, mode=mode, sources=sources, pairs=pairs, out=str(out), **kw)


def _rxn_source(rids, sources, need, pin=None):
    """rxn -> its source: pin {rxn: source index} when given (E2E: a reaction never falls back to another root),
    else the first source that has the reaction (token npz for atoms, X182 row for XGB)."""
    import pandas as pd
    cand = {r: ([sources[int(pin[str(r)])]] if pin and str(r) in pin else sources) for r in rids}
    out = {}
    if need == "tokens":
        for r in rids:
            for s in cand[r]:
                if (Path(s["tokens"]) / f"{r}.npz").is_file():
                    out[r] = s
                    break
        return out
    tabs = {}
    for s in sources:
        if Path(s["x182"]).is_file():
            tabs[s["name"]] = set(pd.read_parquet(s["x182"], columns=["rxn_id"]).rxn_id.astype(int))
    for r in rids:
        for s in cand[r]:
            if r in tabs.get(s["name"], ()):
                out[r] = s
                break
    return out


def _h10(rids, src_of):
    """H10 (10 columns, c_ghost 0) and D3_int per reaction from each reaction's source table."""
    import numpy as np
    import pandas as pd
    cache, H, d3 = {}, np.full((len(rids), len(TARGETS)), np.nan), np.full(len(rids), np.nan)
    for k, r in enumerate(rids):
        s = src_of.get(r)
        if s is None:
            continue
        if s["table"] not in cache:
            cache[s["table"]] = pd.read_parquet(s["table"]).assign(rxn_id=lambda d: d.rxn_id.astype(int)).set_index("rxn_id")
        T = cache[s["table"]]
        if r in T.index:
            H[k, :] = 0.0
            for j, t in zip(H_IDX, H9):
                H[k, j] = float(T.at[r, f"H_{t}"]) if f"H_{t}" in T.columns else np.nan
            if "D3_int" in T.columns:
                d3[k] = float(T.at[r, "D3_int"])
    return H, d3


def run_pred_atoms(job):
    import inputs_mlts as IM
    IM.ensure_env("fu2", script=str(THIS), argv=["_pred", job["_path"]])
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(REPO / "analysis" / "deploy"))
    import data_lf as DL                                           # fill_token_arrays (= the fu2 Data loop)
    FT, Mo, F = DL.FT, DL.Mo, DL.F
    files = [p for p in job["files"] if Path(p).is_file()]
    if len(files) != len(job["files"]):
        M.not_ready(f"model files missing: {sorted(set(job['files']) - set(files))[:3]}")
    objs = [torch.load(p, map_location=FT.DEV) for p in files]
    lay = objs[0].get("layout")
    if lay is None:                                                # LADDER fold units: the rung T_report layout
        tr = M.read_json(job["t_report"])
        lay = dict(elements=[int(z) for z in tr["elements"]], bin_cols=tr["bin_cols"], num_cols=tr["num_cols"],
                   glob_cols=tr["glob_cols"])
    pairs = job["pairs"]
    rids = sorted({int(p[1]) for p in pairs} | {int(p[2]) for p in pairs})
    src = _rxn_source(rids, job["sources"], "tokens", job.get("pin"))
    have = [r for r in rids if r in src]
    paths = [Path(src[r]["tokens"]) / f"{r}.npz" for r in have]
    arr, _ = DL.fill_token_arrays(paths, lay["elements"], len(lay["bin_cols"]), len(lay["num_cols"]))
    X = FT.Data.__new__(FT.Data)
    for k in ("onehot", "bin", "num", "pos", "mask", "etype", "frag"):
        setattr(X, k, arr[k])
    X.n = X.mask.sum(1).astype(int)
    X.d_in = len(lay["elements"]) + len(lay["bin_cols"]) + len(lay["num_cols"])
    row = {r: k for k, r in enumerate(have)}
    H10, _ = _h10(have, src)
    sh = job["arm"] == "SH"
    X.glob = np.concatenate([arr["glob"], H10[:, H_IDX]], 1) if sh else arr["glob"]
    X.d_glob = X.glob.shape[1]
    keep = [p for p in pairs if int(p[1]) in row and int(p[2]) in row and np.isfinite(X.glob[row[int(p[1])]]).all()
            and np.isfinite(X.glob[row[int(p[2])]]).all()]
    ra = np.array([row[int(p[1])] for p in keep], dtype=int)
    rb = np.array([row[int(p[2])] for p in keep], dtype=int)
    Y = []
    for o in objs:
        model = Mo.build(job["model"], o["cell"], X.d_in, len(TARGETS), X.d_glob, o.get("variant", "A+G"),
                         F.DROPOUT).to(FT.DEV)
        model.load_state_dict(o["state"])
        st = FT.Std.__new__(FT.Std)
        for k in ("mu", "sd", "gmu", "gsd", "sy"):
            setattr(st, k, np.asarray(o["std"][k], float))
        T = st.tensors(X)
        ys = FT.predict(model, T, ra, rb) if len(ra) else np.zeros((0, len(TARGETS)))
        if sh:
            yk = ys * np.asarray(o["std"]["sr"], float) + np.asarray(o["std"]["beta"], float) * (H10[ra] - H10[rb])
        else:
            yk = ys * st.sy
        Y.append(yk)
        del model, T
    Y = np.stack(Y)
    beta = np.asarray(objs[0]["std"]["beta"], float) if sh else np.zeros(len(TARGETS))
    base = beta * (H10[ra] - H10[rb]) if sh else np.zeros((len(ra), len(TARGETS)))
    D = pd.concat([pd.DataFrame(dict(pair_id=[p[0] for p in keep], arm=job["arm"], model=job["model"], target=t,
                                     yhat=Y[:, :, j].mean(0), yhat_seed_sd=Y[:, :, j].std(0), base=base[:, j]))
                   for j, t in enumerate(TARGETS)], ignore_index=True)
    M.write_parquet(Path(job["out"]), D)
    miss = sorted({p[0] for p in pairs} - {p[0] for p in keep})
    M.write_json(Path(job["out"]).with_suffix(".json"), dict(n_pairs=len(keep), missing=miss, n_models=len(objs),
                                                             files=files, layout_elements=lay["elements"]))
    return 0


def run_pred_xgb(job):
    import inputs_mlts as IM
    IM.ensure_env("espley", script=str(THIS), argv=["_pred", job["_path"]])
    import numpy as np
    import pandas as pd
    import final_w as FW
    TX, D1T = FW.xgb_mods()
    arm, targets = job["arm"], job["targets"]
    if job["mode"] == "final":
        import xgboost
        part = M.read_json(job["part"])
        out_dir = Path(job["part"]).parent
        models = {}
        for t in targets:
            est = xgboost.XGBRegressor()
            est.load_model(str(out_dir / part["files"][t]["file"]))
            models[t] = est
        sc = FW.load_scale(out_dir / part["scale"]["file"])
        beta, sy, feat = np.asarray(part["beta"], float), part["sy"], part["feat_cols"]
    else:                                                          # fold replay: the LADDER final fit of the unit
        rec = M.read_json(job["unit_json"])
        if job.get("register_rung") == "mlts":
            FW.register(TX, "xgb", job.get("refs", "min"))
        if job["train_rung"] == "mlts_mix":                       # C3: the mixed training data (W-final-C23)
            D = FW.mix_data("xgb", TX, arm)[0]
        else:
            D = TX.get_data(job["train_rung"], arm)
        rows = D.split(int(job["fold"]))["train"]
        cells = {t: int(rec["chosen"][t]["grid_index"]) for t in targets}
        fit = FW.xgb_fit_rows(D, rows, cells, targets)
        models, sc, beta, sy, feat = fit["models"], fit["scale"], fit["beta"], fit["sy"], list(D.feat_cols)
    pairs = job["pairs"]
    rids = sorted({int(p[1]) for p in pairs} | {int(p[2]) for p in pairs})
    src = _rxn_source(rids, job["sources"], "x182", job.get("pin"))
    cache, Xr = {}, {}
    H10, _ = _h10(rids, src)
    for k, r in enumerate(rids):
        s = src.get(r)
        if s is None:
            continue
        if s["x182"] not in cache:
            cache[s["x182"]] = pd.read_parquet(s["x182"]).assign(rxn_id=lambda d: d.rxn_id.astype(int)).set_index("rxn_id")
        x = cache[s["x182"]].loc[r, feat[:182]].to_numpy(float)
        Xr[r] = np.r_[x, H10[k, H_IDX]] if arm == "SH" else x
    hk = {r: k for k, r in enumerate(rids)}
    keep = [p for p in pairs if int(p[1]) in Xr and int(p[2]) in Xr and np.isfinite(Xr[int(p[1])]).all()
            and np.isfinite(Xr[int(p[2])]).all()]
    Xa = np.stack([Xr[int(p[1])] for p in keep]) if keep else np.zeros((0, len(feat)))
    Xb = np.stack([Xr[int(p[2])] for p in keep]) if keep else np.zeros((0, len(feat)))
    dH = np.stack([H10[hk[int(p[1])]] - H10[hk[int(p[2])]] for p in keep]) if keep else np.zeros((0, len(TARGETS)))
    fr = [pd.DataFrame(dict(pair_id=[p[0] for p in keep], arm=arm, model="XGB", target=t,
                            yhat=FW.xgb_predict(models[t], sc, beta, sy, t, Xa, Xb, dH, arm), yhat_seed_sd=0.0))
          for t in targets]
    M.write_parquet(Path(job["out"]), pd.concat(fr, ignore_index=True))
    M.write_json(Path(job["out"]).with_suffix(".json"), dict(n_pairs=len(keep), targets=targets, mode=job["mode"]))
    return 0


def cmd_pred(path):
    job = M.read_json(path)
    job["_path"] = str(path)
    return run_pred_atoms(job) if job["kind"] == "atoms" else run_pred_xgb(job)


def predict(mode, pairs, sources, work, prim=None, c=None, rep8=None, fold=None, card=None, pin=None):
    """the deployed predictor: per REP8 (arm, model) one child part -> {target: frame (pair_id, yhat, yhat_seed_sd)}
    and the part frames. mode 'final' (MODEL_CARD parts) | 'fold' (the primary config's fold-k units).
    pin: {rxn_id: source index} (a pinned reaction is read from that source only)."""
    import pandas as pd
    _orig_part_job = part_job

    def part_job_(*a, **kw):                                       # every child job carries the pin
        return dict(_orig_part_job(*a, **kw), pin={str(k): int(v) for k, v in (pin or {}).items()})
    work = M.assert_write_ok(Path(work))
    work.mkdir(parents=True, exist_ok=True)
    combos = sorted({tuple(v) for v in rep8.values()})
    parts = {}
    for a, m in combos:
        tg = [t for t in TARGETS if t in rep8 and tuple(rep8[t]) == (a, m)]
        out = work / f"part_{mode}_{a}_{m}.parquet"
        c1 = bool(prim and prim.startswith("C1"))
        if m == "XGB":
            if mode == "final":
                job = part_job_("xgb", a, m, mode, sources, pairs, out, targets=tg,
                                part=str(Path(card["part_files"][f"{a}|{m}"])))
            else:
                base, _ = fold_unit_files(prim, c, a, m, fold)
                tr = "dft" if c1 else ("mlts_mix" if str(prim).startswith("C3") else "mlts")
                job = part_job_("xgb", a, m, mode, sources, pairs, out, targets=tg, unit_json=str(base) + ".json",
                                fold=int(fold), train_rung=tr, register_rung=None if c1 else "mlts",
                                refs=(c or {}).get("refs") or "min")
        else:
            if mode == "final":
                pj = rj(card["part_files"][f"{a}|{m}"])
                files = [str(Path(card["part_files"][f"{a}|{m}"]).parent / f["file"]) for f in pj["files"]]
                job = part_job_("atoms", a, m, mode, sources, pairs, out, files=files)
            else:                                                  # fold units: the training rung's token layout
                base, files = fold_unit_files(prim, c, a, m, fold)
                job = part_job_("atoms", a, m, mode, sources, pairs, out, files=files,
                                t_report=str(LAD_RES / "fu2" / "T_report.json") if c1 else
                                src_paths("mlts")["t_report"])
        jp = work / f"job_{mode}_{a}_{m}.json"
        M.write_json(jp, job)
        rc = child("fu2" if job["kind"] == "atoms" else "espley", ["_pred", jp])
        if rc != 0:
            raise SystemExit(rc)
        parts[(a, m)] = pd.read_parquet(out)
    sel = {}
    for t in rep8:
        a, m = rep8[t]
        q = parts[(a, m)]
        sel[t] = q[q.target == t].assign(pair_id=lambda d: d.pair_id.astype(str)).set_index("pair_id")
    return sel, parts


# ================================================================ replay (R1 / R2)
def pick_pairs(n_pairs=10, fold=None):
    """n disjoint E common P_main para pairs, n/5 per dipole, from one fold (DECISIONS P-pairs)."""
    import numpy as np
    import pandas as pd
    if not E_PAIRS.is_file():
        M.not_ready(f"{E_PAIRS} missing (E first)")
    EP = pd.read_csv(E_PAIRS, dtype={"pair_id": str})
    EP = EP[EP.para_main.astype(str).str.lower() == "true"]
    if fold is None:
        fold = int(EP.fold.value_counts().idxmax())
    q = EP[EP.fold == int(fold)]
    rng = np.random.default_rng(SEED)
    pick, used = [], set()
    per = max(1, n_pairs // len(M.DIPOLES))
    for d in M.DIPOLES:
        g = q[q.dipole == d]
        for i in rng.permutation(len(g)):
            r = g.iloc[int(i)]
            if int(r.a_rxn) in used or int(r.b_rxn) in used:
                continue
            pick.append((str(r.pair_id), int(r.a_rxn), int(r.b_rxn)))
            used |= {int(r.a_rxn), int(r.b_rxn)}
            if sum(1 for p in pick if p[0] in set(g.pair_id)) >= per:
                break
    return pick[:n_pairs], int(fold)


def reference_rows(pred_path, pairs, rep8, prim_rung=None):
    import pandas as pd
    P = pd.read_parquet(pred_path, columns=["pair_id", "arm", "model", "target", "yhat", "y"])
    P["pair_id"] = P.pair_id.astype(str)
    ids = {p[0] for p in pairs}
    out = {}
    for t in rep8:
        a, m = rep8[t]
        out[t] = P[(P.pair_id.isin(ids)) & (P.target == t) & (P.arm == a) & (P.model == m)].set_index("pair_id")
    return out


def cmp(sel, ref, pairs, rep8):
    import numpy as np
    worst, n, missing = {}, 0, []
    for t in rep8:
        for p in pairs:
            if p[0] not in sel[t].index or p[0] not in ref[t].index:
                missing.append((p[0], t))
                continue
            d = abs(float(sel[t].at[p[0], "yhat"]) - float(ref[t].at[p[0], "yhat"]))
            m = rep8[t][1]
            worst[m] = max(worst.get(m, 0.0), d)
            n += 1
    ok = not missing and all(v <= (TOL_XGB if m == "XGB" else TOL_ATOM) for m, v in worst.items())
    return dict(pass_=bool(ok), n_compared=n, max_abs_by_model=worst, missing=missing[:20],
                tol=dict(atoms=TOL_ATOM, xgb=TOL_XGB), any_nan=bool(any(not np.isfinite(v) for v in worst.values())))


def cmd_replay(src="mlts", n=20, fold=None, out=None, smoke=False):
    """the predictor with fold models on rung inputs vs the reference predictions (dft: L_predictions; mlts: E);
    src final = R2 (the W final models vs final_on_E; --smoke: the smoke card, its trained targets only)."""
    import pandas as pd
    if src == "final":
        card = rj((M.MLT / "w" / "smoke" / "MODEL_CARD.json") if smoke else CARD)
        if not card:
            M.not_ready("MODEL_CARD.json missing")
        prim, c, rep8, refs = primary()
        if smoke:
            have = {}
            for k, v in card["parts"].items():
                a, m = k.split("|")
                for t in (v.get("targets") or TARGETS):
                    if tuple(rep8.get(t, ())) == (a, m):
                        have[t] = (a, m)
            rep8 = have
        pairs, fold = pick_pairs(int(n), fold)
        res = dict(src=src, fold=fold, pairs=pairs, rep8={t: list(v) for t, v in rep8.items()},
                   **final_replay(pairs, card, rep8, refs, P_DIR / "replay_final"), written=M.now())
        M.write_json(Path(out or (P_DIR / "replay_final.json")), res)
        log(f"replay final: pass {res.get('pass_')}, max |d| {res.get('max_abs_by_model')}")
        return res
    if src == "dft":
        p = str(REPO / "analysis" / "geo2")
        if p not in sys.path:
            sys.path.insert(0, p)
        import eval_geo2 as EG                                     # REP8 of the dft units (unit records only)
        _, r8, _ = EG.reps(EG.load_units(EG.BASE_UNITS))
        rep8 = {t: tuple(r8[("dft", t)]) for t in TARGETS}
        prim, c = "C1", dict(name="C1", refs=None)
        L = pd.read_parquet(LAD_RES / "L_predictions.parquet", columns=["pair_id", "fold", "rung", "target", "arm",
                                                                         "model"])
        L = L[(L.rung == "dft") & (L.target == "oi") & (L.arm == "S") & (L.model == "AW")]
        fold = int(fold if fold is not None else 0)
        S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "a_rxn", "b_rxn"], dtype={"pair_id": str}).set_index("pair_id")
        ids = sorted(L[L.fold == fold].pair_id.astype(str))[:int(n)]
        pairs = [(p, int(S.at[p, "a_rxn"]), int(S.at[p, "b_rxn"])) for p in ids]
        sources = [src_paths("dft")]
        ref_path = LAD_RES / "L_predictions.parquet"
    else:
        prim, c, rep8, refs = primary()
        pairs, fold = pick_pairs(int(n), fold)
        sources = [src_paths("mlts", refs=refs)]
        ref_path = Path(c["predictions"])
    sel, _ = predict("fold", pairs, sources, P_DIR / f"replay_{src}", prim=prim, c=c, rep8=rep8, fold=fold)
    ref = reference_rows(ref_path, pairs, rep8)
    if src == "dft":
        L2 = pd.read_parquet(ref_path, columns=["pair_id", "rung", "arm", "model", "target", "yhat", "y"])
        L2 = L2[L2.rung == "dft"]
        L2["pair_id"] = L2.pair_id.astype(str)
        ref = {t: L2[(L2.pair_id.isin({p[0] for p in pairs})) & (L2.target == t) & (L2.arm == rep8[t][0])
                     & (L2.model == rep8[t][1])].set_index("pair_id") for t in TARGETS}
    res = dict(src=src, fold=fold, n_pairs=len(pairs), pairs=pairs, rep8={t: list(v) for t, v in rep8.items()},
               **cmp(sel, ref, pairs, rep8), written=M.now())
    M.write_json(Path(out or (P_DIR / f"replay_{src}.json")), res)
    log(f"replay {src}: pass {res['pass_']}, max |d| {res['max_abs_by_model']}")
    return res


def final_replay(pairs, card, rep8, refs, work=None):
    """R2: predictor(final models, the inputs final_w used: production MLTS, or the training rung when W ran without
    MLTS inputs) vs final_on_E of the W parts."""
    import pandas as pd
    ref, rungs = {}, set()
    for t in rep8:
        a, m = rep8[t]
        pj = rj(card["part_files"][f"{a}|{m}"]) or {}
        fe = pj.get("final_on_E") or {}
        if not fe.get("path") or not Path(fe["path"]).is_file():
            return dict(pass_=False, reason=f"final_on_E of {m}·{a} missing (W channel ran before E pairs / inputs?)")
        rungs.add(fe.get("input_rung") or "mlts")
        F_ = pd.read_parquet(fe["path"])
        ref[t] = F_[F_.target == t].assign(pair_id=lambda d: d.pair_id.astype(str)).set_index("pair_id")
    if len(rungs) != 1:
        return dict(pass_=False, reason=f"final_on_E parts on different input rungs {sorted(rungs)}")
    src = [src_paths("dft")] if rungs == {"dft"} else [src_paths("mlts", refs=refs)]
    sel, _ = predict("final", pairs, src, work or (P_DIR / "replay_final"), rep8=rep8, card=card)
    return dict(cmp(sel, ref, pairs, rep8), input_rung=sorted(rungs)[0])


# ================================================================ E2E from SMILES
def new_reactions(n_new=5):
    """D1 manifest rows that are not D1 targets (DECISIONS P-new) + their family reference b_rxn."""
    import numpy as np
    import pandas as pd
    man = pd.read_csv(M.D1RUN / "d1_manifest.csv", dtype=str)
    T = M.targets()
    tj, ts = set(T.job_id), set(T.rxn_smiles.astype(str))
    cand = man[~man.job_id.isin(tj) & ~man.rxn_smiles.isin(ts)]
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "ref_kind", "core", "series", "b_rxn"])
    S = S[S.ref_kind == "reference"]
    rng = np.random.default_rng(SEED + 1)
    out = []
    for d in M.DIPOLES:
        g = cand[cand.dipole == d]
        for i in rng.permutation(len(g)):
            r = g.iloc[int(i)]
            core = f"{r.dipole}/{r.skeleton}"
            q = S[(S.core == core) & (S.series == r.series)]
            if q.empty:
                continue
            out.append(dict(job_id=r.job_id, rxn_id=M.rxn_of(r.job_id), dipole=d, core=core, series=r.series,
                            rxn_smiles=r.rxn_smiles, ref_b=int(q.b_rxn.mode().iloc[0])))
            break
    return out[:n_new]


def run_im(root, cmd, args):
    """one inputs_mlts H command on the selfcheck root with the selfcheck targets (child in that command's env)."""
    return child(IM_ENV[cmd], ["_im", cmd] + [str(a) for a in args] + ["--root", str(root)])


def cmd_im(argv):
    """in-process inputs_mlts with targets() = <root>/targets.csv (DECISIONS P-chain)."""
    import inputs_mlts as IM
    cmd = argv[0]
    if not os.environ.get("MLT_IN_ENV"):
        IM.ensure_env(IM_ENV[cmd], script=str(THIS), argv=["_im"] + list(argv))
    k = argv.index("--root")
    root = Path(argv[k + 1])
    with open(root / "targets.csv", newline="") as fh:
        tl = sorted(((int(r["rxn_id"]), r["job_id"]) for r in csv.DictReader(fh)), key=lambda x: x[1])
    orig = IM.targets
    IM.targets = lambda S: list(tl) if S["name"] == "mlts" else orig(S)
    return IM.main(list(argv))


def e2e(pairs, news, prim, c, rep8, refs, card, cores=2, workers=6, root=None, extra_rows=None):
    """the SMILES -> 8 channels chain on a root (default the selfcheck root): pairs = D1 pairs (pair_id, a, b) whose
    two ends run from SMILES; news = new reactions {job_id, rxn_id, ref_b, ...} paired with a production reference;
    extra_rows = {job_id: manifest-format row} for reactions outside the D1 manifest (P-1 `predict`)."""
    import numpy as np
    import pandas as pd
    import final_w as FW
    R = Path(root or SC_DIR)
    arms = rj(ARMS_JSON) or {}
    engine = arms.get("engine") or "A"
    man = {}
    with open(M.D1RUN / "d1_manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            man[r["job_id"]] = r
    man.update(extra_rows or {})
    rx = sorted({p[1] for p in pairs} | {p[2] for p in pairs}) + [n["rxn_id"] for n in news]
    jobs = [M.job_of(r) for r in rx]
    M.write_csv(R / "targets.csv", pd.DataFrame(dict(rxn_id=rx, job_id=jobs)))
    rec = dict(engine=engine, n_reactions=len(rx), steps=[], root=str(R))
    # 1. TS (FT_all_s0_al, the A-3 engine) from SMILES
    import mlts_autode as MA
    import mlts_bridge as B
    m0 = FW.members()[0]
    ok, why = FW.model_available(m0)
    if not ok:
        M.not_ready(f"deployment MLTS {m0} not available ({why})")
    eng = "A" if engine == "App" else engine
    spec = B.resolve_spec(m0)
    h = B.start_server(spec)
    st = {}
    mu = threading.Lock()
    it = iter(jobs)

    def loop():
        while True:
            with mu:
                j = next(it, None)
            if j is None or time_left_h() < RESERVE_H:
                return
            t1 = time.time()
            e_j = eng
            if eng == "Ap" and not (M.G2_FINAL / str(M.rxn_of(j)) / ".done").is_file():
                e_j = "A"                                           # P-engine: no G2 TS for a new SMILES (see note)
            try:
                if not (R / "ade" / "jobs" / j).is_dir():
                    MA.run_one(man[j]["rxn_smiles"], j, m0, e_j, R / "ade", cores, socket=h.socket, job_row=man[j])
                s, w = MA.stage_state(R / "ade" / "jobs" / j)
            except Exception as e:                                  # noqa: BLE001
                s, w = "error", f"{type(e).__name__}: {e}"[:200]
            with mu:
                st[j] = dict(status=s, reason=w, wall_s=time.time() - t1, engine=e_j)
            log(f"E2E TS {j}: {s} {w} (engine {e_j})")
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as ex:
            for f in [ex.submit(loop) for _ in range(max(1, min(workers, len(jobs))))]:
                f.result()
    finally:
        h.shutdown()
    rec["ts"] = st
    if engine == "App":
        rec["note_app"] = "A'' (label-level refine) is not run in the selfcheck: the A TS feeds the chain"
    if any(v.get("engine") != eng for v in st.values()):
        rec["note_engine"] = ("engine A' needs a G2 xTB TS; reactions without $G2/g2_final/<rxn>/.done (the new SMILES) "
                              "ran with engine A (P-engine)")
    # 2. inputs chain
    steps = [("finalize", ["mlts", 0, 1, 1]), ("meta", ["mlts"]), ("x182", ["mlts", 0, 1]), ("x182_merge", ["mlts"]),
             ("atoms", ["mlts", 0, 1]), ("tokens", ["mlts"]), ("hf_inputs", ["mlts"]), ("hf_run", ["mlts", 0, 1]),
             ("hf_build", ["mlts"]), ("d3", ["mlts"]), ("build", ["mlts", "--refs", refs])]
    for cmd, args in steps:
        rc = run_im(R, cmd, args)
        if rc == 2 and cmd in ("finalize", "hf_run", "hf_build"):
            rc = run_im(R, cmd, args)                                # one more pass (unfinished slices)
        rec["steps"].append(dict(cmd=cmd, rc=rc))
        if rc not in (0,):
            rec["chain_stop"] = f"{cmd} exit {rc}"
            break
    # 3. committee spread from the deployed TS: every member re-optimises it (= W-com-start / final_w.delta_com)
    starts = {r: FW.start_ts(r, root=R) for r in rx}
    mem = FW.members()
    for m in mem:
        okm, why = FW.model_available(m)
        if okm:
            FW.opt_records(m, rx, R / "committee" / m, starts)
    dcom, _ = FW.delta_com(None, rx, base=R / "committee")
    rec["delta_com"] = dcom
    # 4. predictions (final model): D1 pairs + (new, reference)
    allp = list(pairs) + [(f"NEW_{n['job_id']}", n["rxn_id"], n["ref_b"]) for n in news]
    sources = [src_paths("mlts", root=R, refs=refs), src_paths("mlts", refs=refs)]
    pin = {r: 0 for r in rx}                                       # selfcheck reactions: the selfcheck root only
    pin.update({n["ref_b"]: 1 for n in news if n["ref_b"] not in pin})   # references of new pairs: production
    pred_rows, pred_err = [], None
    if not rec.get("chain_stop"):
        try:
            sel, _ = predict("final", allp, sources, R / "pred", rep8=rep8, card=card, pin=pin)
            rids = sorted({p[1] for p in allp} | {p[2] for p in allp})
            srcmap = {r: sources[pin.get(r, 0)] for r in rids}
            _, d3v = _h10(rids, srcmap)
            d3 = dict(zip(rids, d3v))
            lab = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
            thr = ((card.get("applicability") or {}).get("thresholds") or {})
            for p in allp:
                for t in [x for x in TARGETS if x in sel]:
                    q = sel[t]
                    yh = float(q.at[p[0], "yhat"]) if p[0] in q.index else float("nan")
                    sd = float(q.at[p[0], "yhat_seed_sd"]) if p[0] in q.index else float("nan")
                    src = f"{rep8[t][1]}·{rep8[t][0]}"
                    if t == "disp" and np.isfinite(d3.get(p[1], np.nan)) and np.isfinite(d3.get(p[2], np.nan)):
                        yh, src = d3[p[1]] - d3[p[2]], "D3_int"
                    lb = lab.get(int(p[2]), {}).get(LABEL_KEYS.get(t, ""), float("nan"))
                    pred_rows.append(dict(pair_id=p[0], a_rxn=p[1], b_rxn=p[2], target=t, yhat=yh, yhat_seed_sd=sd,
                                          source=src, abs_pred_a=(lb + yh) if t in LABEL_KEYS else None,
                                          out_of_range=bool(thr.get(t) is not None and np.isfinite(sd) and sd > thr[t])))
        except SystemExit as e:
            pred_err = f"predict exit {e.code}"
        except Exception as e:                                      # noqa: BLE001
            pred_err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    PR = pd.DataFrame(pred_rows)
    if len(PR):
        M.write_parquet(R / "predictions.parquet", PR)
    rec["predict_error"] = pred_err
    # 5. flags per reaction
    thr_d = (rj(W_REPORT) or {}).get("committee", {}).get("warn_threshold_A")
    els = {"H", "C", "N", "O", "F", "Si", "S", "Cl", "Br"}
    flags = {}
    for r in rx:
        j = M.job_of(r)
        d = R / "ade" / "jobs" / j
        tsr = rj(d / "ts_result.json") or {}
        imag = [float(v) for v in tsr.get("imag_freqs_cm") or []]
        syms = M.read_xyz(d / "ts.xyz")[0] if (d / "ts.xyz").is_file() else []
        gate = st.get(j, {}).get("status")
        dc = dcom.get(r)
        fl = dict(ts_gate=gate, ts_reason=st.get(j, {}).get("reason"), n_imag=sum(1 for v in imag if v < -50.0),
                  delta_com=dc, delta_com_warn=bool(dc is not None and thr_d is not None and dc > thr_d),
                  elements_outside=sorted(set(syms) - els))
        if len(PR):
            q = PR[(PR.a_rxn == r) | (PR.b_rxn == r)]
            fl["predicted"] = bool(len(q) and np.isfinite(q.yhat.to_numpy(float)).all())
            fl["out_of_range_targets"] = sorted(set(q[q.out_of_range].target))
        flags[str(r)] = fl
    rec["flags"] = flags
    # D1 comparison with E (descriptive) and terminal-status check
    if len(PR) and E_ERR.is_file():
        Ee = pd.read_parquet(E_ERR)
        z = PR.merge(Ee[["pair_id", "target", "yhat_mlts", "y"]], on=["pair_id", "target"], how="inner")
        rec["d1_vs_E"] = {t: dict(n=int((z.target == t).sum()),
                                  median_abs_diff=float((z[z.target == t].yhat - z[z.target == t].yhat_mlts).abs().median())
                                  if (z.target == t).any() else None) for t in H9}
    terminal = all(v.get("status") in ("ok", "fail") for v in st.values()) and len(st) == len(jobs)
    stf = R / "inputs" / ("status_mlts_min.csv" if refs == "min" else "status_mlts.csv")
    usable = set()
    if stf.is_file():
        S_ = pd.read_csv(stf, keep_default_na=False)
        usable = {int(r) for r, s in zip(S_.rxn_id, S_.fail_stage.astype(str)) if s == ""}
    usable |= {n["ref_b"] for n in news}                           # production references (H passed in production)
    rec["n_inputs_usable"] = len(usable & set(rx))
    need_p = [p for p in allp if p[1] in usable and p[2] in usable]
    pred_ok = pred_err is None and not rec.get("chain_stop") and all(
        len(PR) and np.isfinite(PR[PR.pair_id == p[0]].yhat.to_numpy(float)).all()
        and (PR.pair_id == p[0]).sum() == len(rep8) for p in need_p)
    rec["n_pairs_needing_prediction"] = len(need_p)
    newp = []
    for n in news:                                                 # P-pass: the new SMILES must run to the end
        q = PR[PR.pair_id == f"NEW_{n['job_id']}"] if len(PR) else PR
        newp.append(dict(job_id=n["job_id"], ts=st.get(n["job_id"] if n["job_id"] in st else M.job_of(n["rxn_id"]),
                                                          {}).get("status"),
                         predicted=bool(len(q) == len(rep8) and np.isfinite(q.yhat.to_numpy(float)).all())))
    rec["new_reactions_outcome"] = newp
    rec["n_new_predicted"] = sum(1 for x in newp if x["predicted"])
    rec.update(terminal=terminal, predicted_all_ok_ts=pred_ok,
               pass_=bool(terminal and pred_ok and not any(v.get("status") == "error" for v in st.values())
                          and (not news or rec["n_new_predicted"] >= 1)))
    return rec


# ================================================================ commands
def cmd_selfcheck(n_pairs=10, n_new=5, cores=2, workers=6, skip_e2e=False):
    if os.environ.get("MLTENV") != "mlts" and not skip_e2e:
        cmd = ["bash", "-c", 'cd "$0" && export MLTENV=mlts && source jobs/env.sh && exec "$PY" "$@"', str(HERE),
               str(THIS), "selfcheck", "--n-pairs", str(n_pairs), "--n-new", str(n_new), "--cores", str(cores),
               "--workers", str(workers)]
        return subprocess.run(cmd).returncode
    card = rj(CARD)
    if not card:
        M.not_ready(f"{CARD} missing (final_w.py channel)")
    if not (rj(W_REPORT) or {}).get("complete"):
        M.not_ready("W_report.json missing / incomplete")
    prim, c, rep8, refs = primary()
    pairs, fold = pick_pairs(n_pairs)
    out = dict(phase="DEPLOY-V2 P-2 selfcheck", primary=prim, refs=refs, fold=fold, pairs=pairs,
               rep8={t: list(v) for t, v in rep8.items()}, decisions=DECISIONS)
    out["R1_fold_replay"] = cmd_replay("mlts", n_pairs, fold, out=SC_DIR / "R1.json")
    out["R2_final_replay"] = final_replay(pairs, card, rep8, refs)
    news = new_reactions(n_new)
    out["new_reactions"] = news
    if skip_e2e:
        out["E2E"] = dict(pass_=None, note="skipped (--skip-e2e)")
    else:
        out["E2E"] = e2e(pairs, news, prim, c, rep8, refs, card, cores, workers)
    out["selfcheck_pass"] = bool(out["R1_fold_replay"].get("pass_") and out["R2_final_replay"].get("pass_")
                                 and out["E2E"].get("pass_"))
    out["written"] = M.now()
    M.write_json(SC_DIR / "selfcheck.json", out)
    log(f"selfcheck: R1 {out['R1_fold_replay'].get('pass_')}, R2 {out['R2_final_replay'].get('pass_')}, "
        f"E2E {out['E2E'].get('pass_')} -> {out['selfcheck_pass']}")
    return 0


def cert_sample(n_pairs=100):
    import numpy as np
    import pandas as pd
    if not E_PAIRS.is_file():
        M.not_ready(f"{E_PAIRS} missing (E first)")
    EP = pd.read_csv(E_PAIRS, dtype={"pair_id": str})
    EP = EP[EP.para_main.astype(str).str.lower() == "true"].sort_values("pair_id")
    rng = np.random.default_rng(SEED + 9)
    q = EP.iloc[sorted(rng.choice(len(EP), size=min(int(n_pairs), len(EP)), replace=False))]
    return q, sorted(set(q.a_rxn.astype(int)) | set(q.b_rxn.astype(int)))


def cmd_cert(i, n, n_pairs=100, ids=None):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import numpy as np
    import gen_dft as GD
    import final_w as FW
    i, n = M.array_task(i), int(n)
    rx = sorted(int(x) for x in ids.split(",")) if ids else cert_sample(n_pairs)[1]     # ids: tests only
    mine = rx[i::n]
    rec_dir = M.assert_write_ok(CERT_DIR / "recs")
    rec_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in mine if not (rec_dir / f"{r}.json").is_file()]
    log(f"cert slice {i}/{n}: {len(mine)} reactions, {len(todo)} to do, {cpus()} serial ORCA workers")
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"mlts_cert_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.getpid()}"
    local.mkdir(parents=True, exist_ok=True)
    mu, it = threading.Lock(), iter(todo)
    env = dict(os.environ)

    def loop():
        while True:
            with mu:
                if time_left_h() < RESERVE_H:
                    return
                r = next(it, None)
            if r is None:
                return
            s0 = FW.start_ts(r)
            out = dict(rxn_id=int(r), time=M.now(), level=M.LABEL_GRAD_LINE.strip())
            if s0 is None:
                out.update(status="no_ts")
            else:
                syms, pos, _ = s0
                res = GD.engrad(local / str(r), M.orca_grad_input(syms, pos, nprocs=1), syms, pos, env, name="c")
                if res["ok"]:
                    g = np.asarray(res["grad"], float)
                    out.update(status="ok", energy_eh=res["energy_eh"], fmax_comp=float(np.abs(g).max()),
                               fmax_atom=float(np.linalg.norm(g, axis=1).max()), frms=float(np.sqrt((g ** 2).mean())),
                               core_h=res["att"]["core_h"])
                else:
                    out.update(status="fail", reason=res.get("reason"), core_h=res["att"]["core_h"])
            shutil.rmtree(local / str(r), ignore_errors=True)
            M.write_text(rec_dir / f"{r}.json", json.dumps(out, default=M._json_default) + "\n")
            log(f"cert {r}: {out['status']} fmax {out.get('fmax_comp')}")
    from concurrent.futures import ThreadPoolExecutor
    nw = max(1, min(cpus(), len(todo)))
    try:
        with ThreadPoolExecutor(nw) as ex:
            for f in [ex.submit(loop) for _ in range(nw)]:
                f.result()
    finally:
        shutil.rmtree(local, ignore_errors=True)
    left = [r for r in mine if not (rec_dir / f"{r}.json").is_file()]
    return 0 if not left else 2


def cmd_report(n_pairs=100):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    sc = rj(SC_DIR / "selfcheck.json")
    if not sc:
        M.not_ready("selfcheck.json missing (predict_v2.py selfcheck)")
    q, rx = cert_sample(n_pairs)
    recs = {r: rj(CERT_DIR / "recs" / f"{r}.json") for r in rx}
    miss = [r for r, v in recs.items() if v is None]
    if miss:
        M.not_ready(f"cert records missing for {len(miss)} of {len(rx)} reactions")
    fm = {r: v.get("fmax_atom") for r, v in recs.items() if v.get("status") == "ok"}   # P-cert: per-atom norm (= G-4)
    v = np.array(list(fm.values()), float)
    vc = np.array([x.get("fmax_comp") for x in recs.values() if x.get("status") == "ok"], float)
    rel = {}
    if E_ERR.is_file() and len(v):
        E = pd.read_parquet(E_ERR)
        E = E[E.pair_id.isin(set(q.pair_id))]
        E["f_pair"] = np.fmax(E.a_rxn.map(fm).astype(float).to_numpy(), E.b_rxn.map(fm).astype(float).to_numpy())
        for t in H9:
            z = E[(E.target == t) & np.isfinite(E.f_pair)]
            ae, f = np.abs(z.err_mlts.to_numpy(float)), z.f_pair.to_numpy(float)
            hi = f > FMAX_THR
            rel[t] = dict(n=int(len(z)), spearman_rho=float(spearmanr(f, ae).statistic) if len(z) > 2 else None,
                          n_above=int(hi.sum()), mae_above=float(ae[hi].mean()) if hi.any() else None,
                          mae_below=float(ae[~hi].mean()) if (~hi).any() else None)
    cert = dict(n_pairs=int(len(q)), n_reactions=len(rx), n_ok=int(len(v)),
                status={s: sum(1 for x in recs.values() if x.get("status") == s) for s in ("ok", "fail", "no_ts")},
                fmax_atom_quantiles={p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 90, 95, 99)} if len(v)
                else None, fmax_comp_quantiles={p: float(np.percentile(vc, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
                if len(vc) else None, frac_above_threshold=float((v > FMAX_THR).mean()) if len(v) else None,
                threshold_eh_bohr=FMAX_THR, threshold_on="max per-atom |F| (G-4 selection rule)", relation=rel,
                core_h_median=float(np.median([x.get("core_h") or 0 for x in recs.values()])), rule=DECISIONS["P-cert"])
    rep = dict(phase="DEPLOY-V2 P (spec 13 P-1 .. P-3)", complete=True, n_todo=0, stop=None,
               est=dict(units_total=1, unit_wall_h=None, workers=1), selfcheck_pass=bool(sc.get("selfcheck_pass")),
               selfcheck={k: sc.get(k) for k in ("primary", "refs", "fold", "R1_fold_replay", "R2_final_replay",
                                                 "new_reactions")},
               e2e={k: (sc.get("E2E") or {}).get(k) for k in ("pass_", "engine", "n_reactions", "steps", "chain_stop",
                                                             "terminal", "predicted_all_ok_ts", "predict_error",
                                                             "d1_vs_E", "flags", "note")},
               cert=cert, decisions=DECISIONS, written=M.now(), code=CODE_VERSION)
    M.write_json(P_REPORT, rep)
    log(f"P report: selfcheck_pass {rep['selfcheck_pass']}, cert ok {cert['n_ok']} -> {P_REPORT}")
    return 0


def cmd_predict(jobs=None, job_json=None, tag="run", cores=2, workers=6):
    """P-1: SMILES -> autodE-MLTS TS -> inputs -> 8 channels + barrier + flags for reactions given as D1 manifest job
    ids (`--jobs T01234:1000017,...` = job : reference rxn_id) or manifest-format rows (`--job-json f`: a list of
    {job_id, rxn_smiles, kind, charge, ..., ref_b}). Runs the selfcheck E2E chain (final channel model, committee,
    flags) on $MLT/p/run/<tag> -> e2e.json + predictions.parquet (pair NEW_<job_id> = (reaction, reference))."""
    if os.environ.get("MLTENV") != "mlts":
        a = ["predict", "--tag", tag, "--cores", str(cores), "--workers", str(workers)]
        a += (["--jobs", jobs] if jobs else []) + (["--job-json", job_json] if job_json else [])
        return subprocess.run(["bash", "-c", 'cd "$0" && export MLTENV=mlts && source jobs/env.sh && exec "$PY" "$@"',
                               str(HERE), str(THIS)] + a).returncode
    card = rj(CARD)
    if not card:
        M.not_ready(f"{CARD} missing (final_w.py channel)")
    prim, c, rep8, refs = primary()
    man = {}
    with open(M.D1RUN / "d1_manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            man[r["job_id"]] = r
    news, extra = [], {}
    for spec_ in [x for x in str(jobs or "").split(",") if x.strip()]:
        j, _, ref = spec_.partition(":")
        if j not in man or not ref:
            raise SystemExit(f"--jobs {spec_!r}: want <manifest job_id>:<reference rxn_id>")
        news.append(dict(job_id=j, rxn_id=M.rxn_of(j), ref_b=int(ref), dipole=man[j].get("dipole")))
    if job_json:
        for r in M.read_json(job_json):
            j = str(r["job_id"])
            extra[j] = {k: str(v) for k, v in r.items() if k != "ref_b"}
            news.append(dict(job_id=j, rxn_id=M.rxn_of(j), ref_b=int(r["ref_b"]), dipole=r.get("dipole")))
    if not news:
        raise SystemExit("predict: give --jobs and / or --job-json")
    root = M.assert_write_ok(P_DIR / "run" / str(tag))
    root.mkdir(parents=True, exist_ok=True)
    rec = e2e([], news, prim, c, rep8, refs, card, cores, workers, root=root, extra_rows=extra)
    rec.update(primary=prim, refs=refs, rep8={t: list(v) for t, v in rep8.items()}, written=M.now(),
               decisions={k: DECISIONS[k] for k in ("P-1", "P-flags", "P-engine", "P-com")})
    M.write_json(root / "e2e.json", rec)
    log(f"predict {tag}: {rec.get('n_new_predicted')}/{len(news)} predicted -> {root}")
    return 0


def cmd_pairs(root, pair_spec, tag="pairs", smoke=False, refs=None):
    """P-1 API: the final channel model on pairs of reactions whose inputs were built under `root` (the inputs_mlts
    H layout; the production root when omitted) -> $MLT/p/pairs/<tag>.parquet (pair, target, yhat incl. disp =
    D3_int, yhat_seed_sd, source, out_of_range)."""
    import numpy as np
    import pandas as pd
    card = rj((M.MLT / "w" / "smoke" / "MODEL_CARD.json") if smoke else CARD)
    if not card:
        M.not_ready("MODEL_CARD.json missing (final_w.py channel)")
    rep8 = {t: tuple(v) for t, v in card["REP8"].items()}
    if smoke:                                                       # the smoke card trained a subset of the combos
        have = {}
        for k, v in card["parts"].items():
            a, m = k.split("|")
            for t in (v.get("targets") or TARGETS):
                if rep8.get(t) == (a, m):
                    have[t] = (a, m)
        rep8 = have
    refs = refs or card.get("refs") or "min"
    pairs = [(f"{tag}_{k}", int(x.split(":")[0]), int(x.split(":")[1])) for k, x in enumerate(pair_spec.split(","))]
    src = [src_paths("mlts", root=root, refs=refs)]
    sel, _ = predict("final", pairs, src, P_DIR / "pairs" / tag, rep8=rep8, card=card)
    rids = sorted({p[1] for p in pairs} | {p[2] for p in pairs})
    _, d3v = _h10(rids, {r: src[0] for r in rids})
    d3 = dict(zip(rids, d3v))
    thr = ((card.get("applicability") or {}).get("thresholds") or {})
    rows = []
    for p in pairs:
        for t in rep8:
            q = sel[t]
            yh = float(q.at[p[0], "yhat"]) if p[0] in q.index else float("nan")
            sd = float(q.at[p[0], "yhat_seed_sd"]) if p[0] in q.index else float("nan")
            s = f"{rep8[t][1]}·{rep8[t][0]}"
            if t == "disp" and np.isfinite(d3.get(p[1], np.nan)) and np.isfinite(d3.get(p[2], np.nan)):
                yh, s = d3[p[1]] - d3[p[2]], "D3_int"
            rows.append(dict(pair_id=p[0], a_rxn=p[1], b_rxn=p[2], target=t, yhat=yh, yhat_seed_sd=sd, source=s,
                             out_of_range=bool(thr.get(t) is not None and np.isfinite(sd) and sd > thr[t])))
    D = pd.DataFrame(rows)
    out = M.write_parquet(P_DIR / "pairs" / f"{tag}.parquet", D)
    log(f"pairs {tag}: {len(pairs)} pairs, {int(np.isfinite(D.yhat).sum())} finite of {len(D)} -> {out}")
    print(D.pivot_table(index="pair_id", columns="target", values="yhat").round(4).to_string())
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_im"]:
        return cmd_im(argv[1:])
    ap = argparse.ArgumentParser(prog="predict_v2.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("selfcheck")
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--n-new", type=int, default=5)
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-e2e", action="store_true")
    p = sub.add_parser("cert")
    p.add_argument("i")
    p.add_argument("n", nargs="?", default="1")
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument("--ids", default=None, help="tests: comma-separated rxn ids instead of the E sample")
    p = sub.add_parser("report")
    p.add_argument("--n-pairs", type=int, default=100)
    p = sub.add_parser("replay")
    p.add_argument("--src", default="mlts", choices=("dft", "mlts", "final"))
    p.add_argument("--smoke", action="store_true", help="tests: src final with the W smoke card")
    p.add_argument("--n", type=int, default=10, help="pairs")
    p.add_argument("--fold", type=int, default=None)
    p = sub.add_parser("pairs")
    p.add_argument("--root", default=None)
    p.add_argument("--pairs", required=True, help="a:b,a:b (rxn ids)")
    p.add_argument("--tag", default="pairs")
    p.add_argument("--refs", default=None)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--jobs", default=None, help="<manifest job_id>:<reference rxn_id>,...")
    p.add_argument("--job-json", default=None, help="list of manifest-format rows with ref_b")
    p.add_argument("--tag", default="run")
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("_pred")
    p.add_argument("job")
    sub.add_parser("_im")
    o = ap.parse_args(argv)
    if o.cmd == "selfcheck":
        return cmd_selfcheck(o.n_pairs, o.n_new, o.cores, o.workers, o.skip_e2e)
    if o.cmd == "cert":
        return cmd_cert(o.i, o.n, o.n_pairs, o.ids)
    if o.cmd == "report":
        return cmd_report(o.n_pairs)
    if o.cmd == "replay":
        r = cmd_replay(o.src, o.n, o.fold, smoke=o.smoke)
        return 0 if r.get("pass_") else 3
    if o.cmd == "pairs":
        return cmd_pairs(o.root, o.pairs, o.tag, o.smoke, o.refs)
    if o.cmd == "predict":
        return cmd_predict(o.jobs, o.job_json, o.tag, o.cores, o.workers)
    if o.cmd == "_pred":
        return cmd_pred(o.job)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
