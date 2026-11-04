#!/usr/bin/env python3
r"""train_k.py — DEPLOY-V2 (MLTS) Phase K channel models C2 / C3 + the C1 inner validation, and the K selection
(docs/specs/DEPLOY_V2_MLTS.md 10; results/PREREG_V2.md 4-6). Compute nodes only (jobs/mlt.slurm through the launcher, or
jobs/sbx.sh); every command re-executes itself in the environment it needs (inputs_mlts.ensure_env): atom units
env_d1fu2 (GPU), XGB units env_espley (CPU), plan / select / status reactot.

CLI (run from analysis/mlts; exit 0 done, 2 not ready / unfinished (same command again), 3 STOP)
  plan                        -> results/K_plan.json {complete, n_todo, stop, est, refs, inputs, rules, units[{arm, unit,
                                 n, cpus, mem_gb, gpu, config, est_h}]}; written before any K training; rewritten only while
                                 no K unit exists (a different plan afterwards = STOP)
  c2  <unit|glob> <i|@I> <n>  C2 units: LADDER training on the MLTS inputs (rung 'mlts'); seed sets s0 = C2/, s1 = C2_s1/,
                                 s2 = C2_s2/
  c3  <unit|glob> <i|@I> <n>  C3 units: label-TS inputs + MLTS inputs mixed on the training side (same y), validation /
                                 test on the MLTS inputs (rung 'mlts_mix')
  c1v <unit|glob> <i|@I> <n>  C1 inner validation: the LADDER dft unit's chosen tuning fit re-run, applied to the MLTS
                                 inputs of its inner_val pairs
  select                      -> $MLT/preds/{C2,C2_s1,C2_s2,C3}_predictions.parquet (L_predictions columns),
                                 $MLT/preds/K_inner_val.parquet, results/K_val.csv, results/K_report.json {complete,
                                 primary (C1|C2|C3), configs, inner_val {C1, C2, C3: oi + cpcm para MAE}, delta_H,
                                 leakage_C2 {pass_}, reps, ...}
  status                      plan progress (prints only)
  c1pred <atoms|xgb|collect> [--refs min|own]
                              C1 (no retraining) through predict_c1.py on the H table of a refs mode (default = the
                                 registered prereg_v2.json criteria.refs.primary -> $MLT/preds/C1_predictions.parquet,
                                 results/C1_report.json {refs, table}; the other mode -> C1_<mode>_*; decision K-15);
                                 atoms needs a GPU (env_d1fu2), xgb CPU (env_espley), collect reactot
  <unit> = '<config>/L__<rung>__<arm>__<model>__f<k>' exactly as listed in K_plan.json; a glob selects several plan units
  of the sub-command, sliced [i::n] (the plan lists one unit per entry with n = 1).
  Test mode: MLT_K_TEST_ROOT=<dir under $MLT/tmp> moves K_plan / K_report / units / preds below <dir>; only there may
  `plan` take the test options --standin dft (the LADDER dft rung files registered as rung 'mlts'), --refs, --configs,
  --arms, --models, --folds, --epochs, --cells, --final-seeds (stored in the plan; units are then flagged smoke).

What is trained (LADDER code imported and called, never its CLI: GA-13 makes train_atoms / train_xgb run / collect STOP
on their code sha; analysis/d1_ladder/train_atoms.py = TA, train_xgb.py = TX, analysis/geo2/deploy = PA / PX / DC)
  rung 'mlts'  registered in this process only: TA/TX.RUNGS['mlts'] = {fu2_res results/fu2_mlts, tokens $MLT/fu2_mlts/
               tokens | x182 $MLT/inputs/X182_mlts.parquet, tag mlts}, GEO_INPUTS['mlts'] = the rung table of the
               pre-registered reference mode (PREREG_V2 6: min -> $MLT/inputs/mlts_min.parquet, own -> mlts.parquet)
               (the predict_c1 / GEO2b pattern). Labels F_pairs_cr2.csv, split T1 (FU2 files), grids, std rules, gates,
               seeds: LADDER's (TA.run_unit / TX.run_unit unchanged).
  in-process replacements (nothing on disk changes): lad_common.assert_path -> writes only below the K units root;
               TA/TX.fingerprint + a 'k' block (config, seed set, refs, table sha, prereg), minus the GEO2b-only keys,
               rules.seeds = the seeds really used; capture wrappers TA.final_kcal / TX.prep_fit / TX.run_parallel /
               TA/TX.write_json keep the chosen tuning fit's inner_val predictions (bit-equal to the unit's val_kcal_mae,
               else STOP) and write the sidecars before the LADDER unit json.
  C2           TA.run_unit / TX.run_unit on rung mlts. s1 / s2: fu2_common.SEED (tuning seed) and the final seeds + 1000 s
               (PREREG_V2 5). XGB is seed free (random_state has no effect: subsample = colsample = 1, hist; GroupKFold
               deterministic), so it is trained once and its predictions are copied into C2_s1 / C2_s2.
  C3           a mixed data object: reaction rows [MLTS | label TS], pair rows [MLTS copies | label copies '<id>#L'],
               same y. Training = MLTS copies of the fold-train pairs usable in rung mlts + label copies of those usable
               in rung dft (final fit: T_main; tuning: inner_train); inner_val and test = MLTS copies only. XGB inner
               GroupKFold folds: each rung's own (identical) partition, a label copy follows its pair's inner fold.
  C1v          per LADDER dft unit: TA fit (fit / fit_sh) with the unit's chosen cell, seed, inner_train and early stopping
               on its dft inner_val (= the tuning fit LADDER did not save), the fitted std / beta applied unchanged to the
               MLTS inputs of the inner_val pairs (geo2 deploy rule DP-4: both ends usable in rung mlts). XGB: TX.prep_fit
               + _task on inner fold 0 with the per-target chosen cell (must reproduce val_kcal_mae exactly). Also
               (descriptive) the unit's final models on the same MLTS inner_val inputs, and a dft-test regression of the
               final models (atoms <= 1e-4, XGB <= 1e-6 vs the LADDER unit parquet, else STOP).
  select       REP / REP8 per config (LADDER 13-1 / eval_geo2.reps on the config's own val_kcal_mae; C1 = LADDER dft);
               primary = min over C1, C2, C3 of the pooled inner-validation para MAE(oi) + MAE(cpcm) of the REP8 models
               (P_main pos p, (fold, pair) rows common to the three configs; ties C1 -> C2 -> C3); test pairs are never
               read for the choice. delta_H (PREREG_V2 5) from the C2 seed sets' test predictions (descriptive).
Outputs: $MLT/units/<config>/<LADDER unit files> + <name>__val.parquet + <name>__train_pairs.parquet + <name>__k.json
(C2 / C3; done = __k.json) or <name>.json + <name>__val.parquet (C1v; done = <name>.json).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)
import inputs_mlts as IM  # noqa: E402          (paths, ensure_env, _clean; stdlib only at import)

CODE_VERSION = "mlts-k-1"
THIS = Path(__file__).resolve()
REPO = M.REPO
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
GEO2_DIR = REPO / "analysis" / "geo2"
GEO2_DEPLOY = GEO2_DIR / "deploy"
DEPLOY_W = REPO / "analysis" / "deploy" / "w"
LAD_UNITS = M.LAD / "units"
LAD_INPUTS = M.LAD / "inputs"
FU2_S_SPLITS = REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json"
T0 = time.time()
JOB_T0 = float(os.environ.get("MLT_JOB_T0") or T0)
WALL_H, RESERVE_H = 48.0, 2.0


def _test_root():
    v = os.environ.get("MLT_K_TEST_ROOT")
    if not v:
        return None
    p = Path(os.path.abspath(v))
    tmp = Path(os.path.abspath(M.MLT / "tmp"))
    if tmp not in p.parents:
        raise SystemExit(f"MLT_K_TEST_ROOT {p} must lie below {tmp} (test roots only)")
    return p


TEST_ROOT = _test_root()
UNITS = (TEST_ROOT / "units") if TEST_ROOT else M.MLT / "units"
PREDS = (TEST_ROOT / "preds") if TEST_ROOT else M.MLT / "preds"
KRES = (TEST_ROOT / "results") if TEST_ROOT else M.RES
K_PLAN = KRES / "K_plan.json"
K_REPORT = KRES / "K_report.json"
K_VAL = KRES / "K_val.csv"
K_IV = PREDS / "K_inner_val.parquet"
PREREG_MD, PREREG_JSON = M.RES / "PREREG_V2.md", M.RES / "prereg_v2.json"
ARMS_JSON = M.MLT / "pipeline" / "arms.json"
C1_PRED = PREDS / "C1_predictions.parquet"                 # train_k.py c1pred (registered refs), predict_c1 layout
C1_REPS = M.MLT / "preds" / "C1_representatives.json"

TARGETS = list(M.TARGETS)                          # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier (= LADDER)
PRIMARY9 = [t for t in TARGETS if t != "c_ghost"]
MAIN = ("oi", "cpcm")
ARMS = ("S", "SH")
ATOM_MODELS = ("AW", "GNN", "TF")
MODELS = ATOM_MODELS + ("XGB",)
NF = 5
SEED_STEP = 1000                                   # PREREG_V2 5: seed set s = tuning and final seeds + 1000 s
LAB = "#L"                                         # C3: pair id suffix of the label-TS copies
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]
CONFIGS = {   # config -> sub-command, model rung, seed set
    "C2": dict(sub="c2", rung="mlts", seed_set=0),
    "C2_s1": dict(sub="c2", rung="mlts", seed_set=1),
    "C2_s2": dict(sub="c2", rung="mlts", seed_set=2),
    "C3": dict(sub="c3", rung="mlts_mix", seed_set=0),
    "C1v": dict(sub="c1v", rung="dft", seed_set=0),
}
PRIMARY_ORDER = ("C1", "C2", "C3")
DECISIONS = {
    "K-1 reuse": "LADDER train_atoms.run_unit / train_xgb.run_unit imported and called (never the LADDER CLI, GA-13); "
                 "rung 'mlts' registered in-process (RUNGS + GEO_INPUTS, predict_c1 / GEO2b pattern); in-process "
                 "replacements only: lad_common.assert_path (writes only below the K units root), fingerprint (+ 'k' "
                 "block, minus GEO2b keys, true seeds), capture wrappers for the chosen tuning fit's inner_val "
                 "predictions (bit-equal to val_kcal_mae, else STOP)",
    "K-2 refs": "K_plan refs = prereg_v2.json criteria.refs.primary (PREREG_V2 6; min -> $MLT/inputs/mlts_min.parquet); "
                "the same table for training, validation and test of C2 / C3 / C1v; its sha256 is fixed in K_plan "
                "(a unit STOPs if the table changed)",
    "K-3 C2 seeds": "s0 = the C2 unit; s1, s2 = tuning seed fu2_common.SEED + 1000 s and final seeds + 1000 s (patched "
                    "in-process for the unit); folders C2/, C2_s1/, C2_s2/ with LADDER unit names; XGB seed free "
                    "(subsample = colsample = 1, hist, deterministic GroupKFold: random_state has no effect) -> trained "
                    "once, C2 XGB predictions copied into C2_s1 / C2_s2",
    "K-4 C3 mix": "mixed data object: reaction rows [MLTS | label TS], pair rows [MLTS | label copies '#L'], same y; "
                  "final training = MLTS copies of the fold-train pairs usable in mlts + label copies of those usable "
                  "in dft; tuning inner_train likewise; inner_val + test = MLTS copies only (validation on deployment "
                  "inputs, comparable with C1 / C2); XGB inner folds = each rung's own GroupKFold partition (label copy "
                  "follows its pair); no cross pairs (a_label, b_mlts); leak checks by base pair id and reaction id",
    "K-5 C1v": "C1 inner validation = the LADDER dft unit's chosen tuning fit re-run (cell, seed 20261019, inner_train, "
               "early stopping on dft inner_val) applied to the MLTS inputs of the inner_val pairs (dft std / beta "
               "unchanged, DP-4 usable rule); not the saved final models, which were trained on those inner_val pairs "
               "(in-sample; reported as variant final_models); XGB replica must reproduce val_kcal_mae exactly, the "
               "atom replica's deviation (GPU nondeterminism) is recorded, not gated",
    "K-6 primary": "REP / REP8 per config = LADDER 13-1 / eval_geo2.reps on the config's own 5-fold val_kcal_mae (C1 = "
                   "LADDER dft REP8); primary = min over C1, C2, C3 of pooled inner-val para MAE(oi) + MAE(cpcm), REP8 "
                   "models, P_main pos p, (fold, pair) rows common to the three configs; ties C1 -> C2 -> C3; test "
                   "pairs never used",
    "K-7 delta_H": "PREREG_V2 5: H (DEPLOY_MODEL_W 7, eval_w.family_points / h_table / h_value) of the C2 REP8 models "
                   "(s0 choice fixed for s1, s2) on each seed set's test predictions; delta_H = max - min, rel = / mean; "
                   "per-target delta para MAE; descriptive only",
    "K-8 units": "one K_plan entry per unit (n = 1) so tests.py c2_k resolves every unit by name; atoms GPU 1 + 8 CPU "
                 "(24 GB; C3 / C1v 48 GB), XGB 32 CPU / 64 GB (C3 96 GB), C1v XGB 16 CPU / 32 GB; entries ordered by "
                 "estimated hours (LADDER dft wall times; C3 x 2)",
    "K-9 cache": "done = the K record (C2 / C3 <name>__k.json, C1v <name>.json, written last); a LADDER unit json "
                 "already present is accepted when its fingerprint equals the current one except 'gpu' (a rerun on "
                 "another GPU type does not STOP), else STOP",
    "K-10 C2-K": "<name>__train_pairs.parquet (pair_id = S_folds id, copy mlts | label) next to every C2 / C3 unit; C1v "
                 "json carries train_pair_ids (the fold's T_main = superset of its fits' pairs)",
    "K-11 plan": "K_plan.json is (re)written only while no K unit exists; a different plan afterwards = STOP",
    "K-12 App": "arms.json is recorded in K_plan; K trains on the main A-4 inputs only (an A'' arm is not trained here)",
    "K-13 test": "test mode = MLT_K_TEST_ROOT below $MLT/tmp; stand-in 'dft' registers the LADDER dft rung files as rung "
                 "mlts (C2 = LADDER retrain, C3 = dft + dft); limits stored in the test plan, units flagged smoke",
    "K-14 report": "K_report.configs = what E / W / P read, one entry per evaluated configuration: C1 (registered refs), "
                   "C1_<other refs> (descriptive), C2, C3 with predictions = a path string, refs, table, units_dir "
                   "($MLT/units/<config>), rung (C3 = mlts_mix); C2_s1 / C2_s2 only under seed_sets (delta_H; not "
                   "evaluated again in E); C1v under inner_val_units. C3's final model is trained by final_w.py "
                   "(train_k make_mix_*), not here.",
    "K-15 C1 refs": "C1 predictions on the pre-registered refs mode (PREREG_V2 6: min primary -> mlts_min.parquet; own "
                    "secondary): train_k.py c1pred atoms|xgb|collect [--refs m] wraps predict_c1.py (imported "
                    "unchanged; in-process only src_cfg('mlts').table = the mode's table and the mode's output names); "
                    "the report records refs / table so E can verify the registration.",
}


def log(msg):
    print(f"[K {time.time() - T0:7.0f}s] {msg}", flush=True)


def stop(msg):
    M.stop(msg)


def not_ready(msg):
    M.not_ready(msg)


def jdump(path, obj):
    return M.write_json(path, IM._clean(obj))


def code_sha():
    return M.sha256(THIS)


def reserve_left():
    return WALL_H - (time.time() - JOB_T0) / 3600.0


# ================================================================ units / plan helpers
def parse_unit(s):
    """'<config>/L__<rung>__<arm>__<model>__f<k>' -> (config, unit dict, LADDER-style name)."""
    cfg, sep, name = str(s).partition("/")
    parts = name.split("__")
    if not sep or cfg not in CONFIGS or len(parts) != 5 or parts[0] != "L" or not parts[4].startswith("f"):
        raise SystemExit(f"bad K unit {s!r} (want <config>/L__<rung>__<arm>__<model>__f<k>, config in {list(CONFIGS)})")
    _, rung, arm, model, f = parts
    u = dict(rung=rung, arm=arm, model=model, fold=int(f[1:]))
    if rung != CONFIGS[cfg]["rung"] or arm not in ARMS or model not in MODELS or not 0 <= u["fold"] < NF:
        raise SystemExit(f"bad K unit {s!r}: rung {rung} (config {cfg} needs {CONFIGS[cfg]['rung']}), arm, model or fold")
    return cfg, u, name


def unit_name(cfg, arm, model, k):
    return f"{cfg}/L__{CONFIGS[cfg]['rung']}__{arm}__{model}__f{k}"


def done_path(cfg, name):
    return UNITS / cfg / (f"{name}.json" if cfg == "C1v" else f"{name}__k.json")


def unit_done(x):
    cfg, _, name = parse_unit(x["unit"])
    return done_path(cfg, name).is_file()


def any_unit_files():
    return UNITS.is_dir() and any(UNITS.glob("*/L__*"))


def load_plan():
    if not K_PLAN.is_file():
        not_ready(f"{K_PLAN} missing (train_k.py plan)")
    P = M.read_json(K_PLAN)
    if not P.get("complete"):
        not_ready(f"{K_PLAN} incomplete")
    if (TEST_ROOT is None) != (P.get("test") is None):
        stop(f"{K_PLAN}: test plan used outside its test root (or a production plan inside one)")
    return P


# ================================================================ plan
def prereg_refs(o):
    """PREREG_V2 6: the pre-registered reference mode of the MLTS inputs (refs.primary)."""
    if o.refs and TEST_ROOT is None:
        stop("--refs is a test option: production reads results/prereg_v2.json criteria.refs.primary")
    if not (PREREG_JSON.is_file() and PREREG_MD.is_file()):
        if TEST_ROOT is not None and o.refs:
            return o.refs, dict(source="test option --refs (no PREREG_V2 yet)", md_sha256=None)
        not_ready("results/PREREG_V2.md / prereg_v2.json missing (Phase R first)")
    J = M.read_json(PREREG_JSON)
    md = M.sha256(PREREG_MD)
    if J.get("md_sha256") != md:
        stop("PREREG_V2.md sha256 differs from prereg_v2.json md_sha256")
    refs = ((J.get("criteria") or {}).get("refs") or {}).get("primary")
    if refs not in ("min", "own"):
        stop(f"prereg_v2.json criteria.refs.primary = {refs!r}, expected min | own")
    info = dict(source="results/prereg_v2.json criteria.refs.primary", md_sha256=md, json_sha256=M.sha256(PREREG_JSON),
                registered=refs)
    if o.refs and o.refs != refs:
        info["test_override"] = o.refs
        refs = o.refs
    return refs, info


def input_files(refs, standin):
    """the rung-mlts files (production: Phase H outputs; test stand-in 'dft': the LADDER dft rung files)."""
    if standin == "dft":
        if TEST_ROOT is None:
            stop("--standin is a test option")
        table, tokens = LAD_INPUTS / "dft.parquet", M.LAD / "fu2" / "tokens"
        fu2_res, x182, tag = LAD_RES / "fu2", M.LAD / "X182_d1_dft.parquet", "dft"
    elif standin:
        stop(f"unknown stand-in {standin!r} (dft)")
    else:
        S = IM.src_cfg("mlts")
        table = S["table_min"] if refs == "min" else S["table"]
        tokens, fu2_res, x182, tag = S["fu2_scr"] / "tokens", S["fu2_res"], S["x182"], S["tag"]
    miss = [str(p) for p in (table, fu2_res / "T_report.json", x182, LAD_INPUTS / "dft.parquet") if not p.is_file()]
    if not tokens.is_dir():
        miss.append(str(tokens))
    if miss:
        not_ready(f"rung mlts inputs missing: {miss} (Phase H: inputs_mlts.py build mlts --refs {refs})")
    return dict(table=str(table), table_sha256=M.sha256(table), tokens=str(tokens), fu2_res=str(fu2_res),
                t_report_sha256=M.sha256(fu2_res / "T_report.json"), x182=str(x182), x182_sha256=M.sha256(x182),
                tag=tag, refs=refs, standin=standin, label_table=str(LAD_INPUTS / "dft.parquet"),
                label_table_sha256=M.sha256(LAD_INPUTS / "dft.parquet"))


def ladder_hours():
    out = {}
    for p in (LAD_RES / "L_report.json", LAD_RES / "L_xgb_report.json"):
        J = M.read_json(p) if p.is_file() else {}
        for n, v in (J.get("units") or {}).items():
            out[n] = float(v.get("wall_seconds") or 0.0) / 3600.0
    return out


DEFAULT_H = dict(GNN=3.5, AW=0.8, TF=0.45, XGB=0.3)


def est_hours(cfg, arm, model, k, lh):
    name = f"L__dft__{arm}__{model}__f{k}"
    base = lh.get(name) or DEFAULT_H[model]
    if cfg == "C1v":
        if model == "XGB":
            return 0.15
        try:
            r = M.read_json(LAD_UNITS / f"{name}.json")
            sec = float(r["tune"][int(r["chosen"]["index"])]["seconds"])
        except (OSError, KeyError, ValueError, TypeError, IndexError):
            sec = base * 3600.0 / 6.0
        return sec / 3600.0 + 0.25
    return (2.0 * base + 0.2) if cfg == "C3" else (base + 0.1)


def build_plan(o):
    if TEST_ROOT is None and any(v is not None for v in (o.configs, o.arms, o.models, o.folds, o.epochs, o.cells,
                                                          o.final_seeds, o.standin, o.refs)):
        stop("plan options are test options (set MLT_K_TEST_ROOT below $MLT/tmp)")
    refs, prereg = prereg_refs(o)
    inputs = input_files(refs, o.standin)
    cfgs = [c for c in (o.configs.split(",") if o.configs else list(CONFIGS))]
    bad = [c for c in cfgs if c not in CONFIGS]
    arms = o.arms.split(",") if o.arms else list(ARMS)
    models = o.models.split(",") if o.models else list(MODELS)
    folds = [int(x) for x in o.folds.split(",")] if o.folds else list(range(NF))
    if bad or set(arms) - set(ARMS) or set(models) - set(MODELS) or set(folds) - set(range(NF)):
        stop(f"bad --configs / --arms / --models / --folds ({bad})")
    limited = any(v is not None for v in (o.configs, o.arms, o.models, o.folds, o.epochs, o.cells, o.final_seeds,
                                           o.standin, o.refs))
    if limited and TEST_ROOT is None:
        stop("plan options other than none are test options (set MLT_K_TEST_ROOT below $MLT/tmp)")
    lh = ladder_hours()
    units = []
    for cfg in cfgs:
        for arm in arms:
            for model in models:
                if CONFIGS[cfg]["seed_set"] > 0 and model == "XGB":
                    continue                                        # K-3: XGB seed free, copied from C2
                for k in folds:
                    gpu = model != "XGB"
                    if gpu:
                        cpus, mem = 8, (48 if cfg in ("C3", "C1v") else 24)
                    else:
                        cpus, mem = (16, 32) if cfg == "C1v" else (32, 96 if cfg == "C3" else 64)
                    units.append(dict(arm=CONFIGS[cfg]["sub"], unit=unit_name(cfg, arm, model, k), n=1, cpus=cpus,
                                      mem_gb=mem, gpu=int(gpu), config=cfg,
                                      est_h=round(est_hours(cfg, arm, model, k, lh), 3)))
    units.sort(key=lambda x: (-x["est_h"], x["unit"]))
    test = None
    if TEST_ROOT is not None:
        test = dict(root=str(TEST_ROOT), standin=o.standin, epochs=o.epochs, final_seeds=o.final_seeds,
                    cells=[int(c) for c in o.cells.split(",")] if o.cells else None)
    hs = sorted(x["est_h"] for x in units)
    arms_json = M.read_json(ARMS_JSON) if ARMS_JSON.is_file() else None
    plan = dict(spec="DEPLOY_V2_MLTS.md 10 (K: C2 / C3 + the C1 inner validation C1v); PREREG_V2 4-6", code=CODE_VERSION,
                code_sha256=code_sha(), written=IM.now(), complete=True, n_todo=len(units), stop=None,
                est=dict(units_total=len(units), unit_wall_h=hs[len(hs) // 2] if hs else 0.0, workers=10,
                         gpu_h_total=round(sum(x["est_h"] for x in units if x["gpu"]), 2),
                         cpu_h_total=round(sum(x["est_h"] * x["cpus"] for x in units if not x["gpu"]), 2),
                         basis="LADDER dft unit wall hours (A10 / 32-CPU XGB), C3 x 2, C1v = the chosen tuning fit"),
                refs=refs, prereg=prereg, inputs=inputs, arms_json=arms_json, configs=CONFIGS, seed_step=SEED_STEP,
                test=test, rules=DECISIONS, units=units,
                launcher="each entry: train_k.py <arm> <unit> @I <n> (env reactot; the script re-executes itself in "
                         "env_d1fu2 (gpu 1) / env_espley); done = every entry's K record, then train_k.py select")
    return plan


PLAN_KEYS = ("refs", "inputs", "test", "units", "configs", "seed_step")


def cmd_plan(o):
    plan = build_plan(o)
    if K_PLAN.is_file():
        old = M.read_json(K_PLAN)
        same = all(json.dumps(IM._clean(old.get(k)), sort_keys=True) == json.dumps(IM._clean(plan.get(k)), sort_keys=True)
                   for k in PLAN_KEYS)
        if same:
            log(f"{K_PLAN} exists and is unchanged ({len(old['units'])} units)")
            return 0
        if any_unit_files():
            diff = [k for k in PLAN_KEYS if json.dumps(IM._clean(old.get(k)), sort_keys=True)
                    != json.dumps(IM._clean(plan.get(k)), sort_keys=True)]
            stop(f"K plan changed ({diff}) after K units were written under {UNITS} (K-11: no silent re-plan)")
    jdump(K_PLAN, plan)
    n_gpu = sum(x["gpu"] for x in plan["units"])
    log(f"plan -> {K_PLAN}: {len(plan['units'])} units ({n_gpu} GPU), refs {plan['refs']}, est GPU-h "
        f"{plan['est']['gpu_h_total']}, CPU-h {plan['est']['cpu_h_total']}")
    return 0


def cmd_status(o):
    P = load_plan()
    by = {}
    for x in P["units"]:
        c = by.setdefault(x["config"], [0, 0])
        c[0] += 1
        c[1] += int(unit_done(x))
    for c, (n, d) in by.items():
        print(f"{c:6s} {d}/{n} done")
    print(f"K_report: {'present' if K_REPORT.is_file() else 'absent'}")
    return 0


# ================================================================ LADDER code in this process
class _NS:
    pass


X = _NS()          # loaded modules: TA / FT / F / PA / DC (atoms), TX / D1T / PX / DC (xgb)
ORIG = {}
CTX = {}           # the unit being trained (fingerprint 'k' block, seeds)
CAP = dict(on=False)


def _geo2_import():
    for p in (str(GEO2_DEPLOY), str(GEO2_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def k_assert_path(p):
    """lad_common.assert_path replacement: LADDER code may write only below the K units root."""
    q = Path(os.path.abspath(p))
    root = Path(os.path.abspath(UNITS))
    if q == root or root in q.parents:
        M.assert_write_ok(q)
        return q
    print(f"STOP: K write path {q} outside {root}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def register(mod, kind, plan):
    """rung 'mlts' in train_atoms / train_xgb of this process (predict_c1.register_mlts with the K plan's table)."""
    inp = plan["inputs"]
    if M.sha256(inp["table"]) != inp["table_sha256"]:
        stop(f"{inp['table']} changed after K_plan.json (K-2)")
    if M.sha256(inp["label_table"]) != inp["label_table_sha256"]:
        stop(f"{inp['label_table']} changed after K_plan.json")
    if kind == "atoms":
        mod.RUNGS["mlts"] = dict(fu2_res=Path(inp["fu2_res"]), tokens=Path(inp["tokens"]), tag=inp["tag"])
    else:
        mod.RUNGS["mlts"] = dict(fu2_res=Path(inp["fu2_res"]), x182=Path(inp["x182"]), tag=inp["tag"])
    mod.GEO_INPUTS["mlts"] = Path(inp["table"])
    if list(mod.TARGETS) != TARGETS or tuple(mod.ARMS) != ARMS:
        stop(f"LADDER TARGETS / ARMS {mod.TARGETS} {mod.ARMS} differ from train_k")


def wrap_fingerprint(orig, kind):
    def fp(D, u, sp, *a, **k):
        d = orig(D, u, sp, *a, **k)
        if CTX.get("cfg") and getattr(D, "rung", None) in ("mlts", "mlts_mix"):
            for key in ("prereg_geo2b", "d0d1_code", "rules_d0d1"):
                d.pop(key, None)
            if kind == "atoms" and isinstance(d.get("rules"), dict):
                d["rules"]["seeds"] = list(CTX["seeds"])
                d["rules"]["tune_seed"] = int(CTX["tune_seed"])
            d["k"] = CTX["fp"]
            d = json.loads(json.dumps(IM._clean(d)))
        return d
    return fp


def setup(kind, plan):
    _geo2_import()
    if kind == "atoms":
        import predict_atoms as PA
        X.PA, X.TA, X.FT, X.F, X.Mo, X.DC = PA, PA.TA, PA.FT, PA.F, PA.Mo, PA.DC
        mod = X.TA
        X.FT.require_gpu()
    else:
        import predict_xgb as PX
        X.PX, X.TX, X.D1T, X.DC = PX, PX.TX, PX.D1T, PX.DC
        mod = X.TX
    register(mod, kind, plan)
    L = mod.L
    if not ORIG.get("assert_path"):
        ORIG["assert_path"] = L.assert_path
        L.assert_path = k_assert_path                      # lad_common.write_atomic / save_torch / _check_root
        ORIG["fingerprint"] = mod.fingerprint
        mod.fingerprint = wrap_fingerprint(mod.fingerprint, kind)
        ORIG["write_json"] = mod.write_json
        mod.write_json = _write_json_hook
        if kind == "atoms":
            ORIG["final_kcal"] = mod.final_kcal
            mod.final_kcal = _final_kcal_cap
            ORIG["seed"], ORIG["final_seeds"] = int(X.F.SEED), list(X.F.FINAL_SEEDS)
        else:
            ORIG["prep_fit"], ORIG["run_parallel"] = mod.prep_fit, mod.run_parallel
            mod.prep_fit, mod.run_parallel = _prep_fit_cap, _run_parallel_cap
    return mod


def _final_kcal_cap(D, r, rows):
    import numpy as np
    y = ORIG["final_kcal"](D, r, rows)
    if CAP.get("on") and CAP.get("iva") is not None and len(rows) == len(CAP["iva"]) \
            and np.array_equal(np.asarray(rows), CAP["iva"]):
        beta = np.asarray(r["beta"], float).copy() if getattr(D, "arm", "S") == "SH" else np.zeros(len(TARGETS))
        CAP["items"].append(dict(yhat=np.array(y, dtype=float, copy=True), beta=beta, best_epoch=r.get("best_epoch")))
    return y


def _prep_fit_cap(D, rows, evals, gate_rows):
    out = ORIG["prep_fit"](D, rows, evals, gate_rows)
    if CAP.get("on"):
        CAP["prep"].append(out)
    return out


def _run_parallel_cap(calls):
    res = ORIG["run_parallel"](calls)
    if CAP.get("on"):
        CAP["par"].append(res)
    return res


def _write_json_hook(path, obj):
    """the LADDER unit json is written last by run_unit: the K sidecars (inner_val predictions of the chosen tuning fit,
    training pairs) go first, so a unit json never exists without them."""
    if CAP.get("on") and CAP.get("js") is not None and Path(os.path.abspath(path)) == CAP["js"]:
        (_sidecars_atoms if CAP["kind"] == "atoms" else _sidecars_xgb)(obj)
    return ORIG["write_json"](path, obj)


def base_id(q):
    q = str(q)
    return q[:-len(LAB)] if q.endswith(LAB) else q


def _val_frame(D, rows, u, Y, base, beta, rung, variant, cfg, extra=None):
    """inner-validation prediction rows (PRED_COLS + config / seed_set / variant), pair ids = S_folds ids."""
    import numpy as np
    d = X.DC.pred_frame(D, rows, u, np.asarray(Y, float), base, beta, rung)
    d["pair_id"] = d.pair_id.map(base_id)
    d["config"] = cfg
    d["seed_set"] = int(CONFIGS[cfg]["seed_set"])
    d["variant"] = variant
    d["set"] = "inner_val"
    for k, v in (extra or {}).items():
        d[k] = v
    return d


def _train_frame(D, rows):
    import numpy as np
    import pandas as pd
    ids = np.asarray(D.P.index)[np.asarray(rows, dtype=int)].astype(str)
    return pd.DataFrame(dict(pair_id=[base_id(q) for q in ids], copy=["label" if q.endswith(LAB) else "mlts" for q in ids]))


def _sidecars_atoms(obj):
    import numpy as np
    D, iva, u, cfg = CAP["D"], CAP["iva"], CAP["u"], CAP["cfg"]
    items = CAP["items"]
    if len(items) != len(obj.get("tune") or []):
        stop(f"{CAP['name']}: {len(items)} captured tuning predictions for {len(obj.get('tune') or [])} tuning cells")
    best = int(obj["chosen"]["index"])
    it = items[best]
    vk = np.abs(it["yhat"] - D.y[iva]).mean(0)
    want = [float(obj["val_kcal_mae"][t]) for t in TARGETS]
    if [float(v) for v in vk] != want:
        stop(f"{CAP['name']}: captured inner_val predictions do not reproduce val_kcal_mae ({vk.tolist()} vs {want})")
    beta = it["beta"]
    base = beta[None, :] * D.dH[iva] if D.arm == "SH" else np.zeros((len(iva), len(TARGETS)))
    V = _val_frame(D, iva, u, it["yhat"][None], base, beta, u["rung"], "tune", cfg,
                   dict(cell_index=best, best_epoch=int(obj["chosen"]["best_epoch"])))
    M.write_parquet(CAP["valp"], V)
    M.write_parquet(CAP["trainp"], _train_frame(D, CAP["sp"]["train"]))
    CAP["val_ok"] = True


def _sidecars_xgb(obj):
    import numpy as np
    TX = X.TX
    D, sp, u, cfg = CAP["D"], CAP["sp"], CAP["u"], CAP["cfg"]
    n_in = len(sp["inner"])
    if len(CAP["prep"]) < n_in + 1 or len(CAP["par"]) < 1:
        stop(f"{CAP['name']}: captured {len(CAP['prep'])} prep_fit / {len(CAP['par'])} run_parallel calls")
    fits, tun, idx = CAP["prep"][:n_in], CAP["par"][0], CAP["idx"]
    jobs = sorted([(j, ti, ci) for ti in range(len(TARGETS)) for j in range(n_in) for ci in idx],
                  key=lambda x: -TX.cost(TX.GRID[x[2]]))                       # = train_xgb.run_unit tuning order
    if len(tun) != len(jobs):
        stop(f"{CAP['name']}: {len(tun)} captured tuning fits, expected {len(jobs)}")
    pos = {jb: p for p, jb in enumerate(jobs)}
    val = sp["inner"][0]["val"]
    if not np.array_equal(val, sp["inner_val"]):
        stop(f"{CAP['name']}: inner fold 0 validation rows differ from inner_val")
    Y = np.zeros((len(val), len(TARGETS)))
    cells = {}
    for ti, t in enumerate(TARGETS):
        gi = int(obj["chosen"][t]["grid_index"])
        kc = tun[pos[(0, ti, gi)]][0]["val"]
        Y[:, ti] = fits[0]["base"]["val"][:, ti] + kc
        mae = float(np.mean(np.abs(Y[:, ti] - D.y[val, ti])))
        if mae != float(obj["val_kcal_mae"][t]):
            stop(f"{CAP['name']} {t}: captured inner fold 0 predictions give {mae!r}, val_kcal_mae {obj['val_kcal_mae'][t]!r}")
        cells[t] = gi
    beta = np.asarray(fits[0]["beta"], float)
    V = _val_frame(D, val, u, Y[None], fits[0]["base"]["val"], beta, u["rung"], "tune", cfg)
    V["cell_index"] = V.target.map(cells)
    M.write_parquet(CAP["valp"], V)
    M.write_parquet(CAP["trainp"], _train_frame(D, sp["train"]))
    CAP["val_ok"] = True


# ================================================================ C3 mixed data objects
def _pad(a, nmax, axes):
    import numpy as np
    w = [(0, 0)] * a.ndim
    for ax in axes:
        w[ax] = (0, nmax - a.shape[ax])
    return np.pad(a, w)


def _mix_common(self, Dm, Dd):
    """shared parts of the C3 objects (K-4): pair table [MLTS | label copies], y, reaction rows, usable filter."""
    import numpy as np
    import pandas as pd
    if list(map(str, Dm.P.index)) != list(map(str, Dd.P.index)):
        stop("C3: the two rungs' pair tables differ (same label file expected)")
    if not np.array_equal(Dm.y, Dd.y, equal_nan=True):
        stop("C3: labels y differ between rung mlts and rung dft")
    if Dm.split_name != Dd.split_name or Dm.sha["s_splits"] != Dd.sha["s_splits"]:
        stop("C3: split files differ between the rungs")
    if list(Dm.hinfo["h_cols"]) != list(Dd.hinfo["h_cols"]):
        stop(f"C3: H columns differ {Dm.hinfo['h_cols']} vs {Dd.hinfo['h_cols']}")
    nr_m = len(Dm.rids)
    self.parts = (Dm, Dd)
    self.Np = len(Dm.P)
    self.nr_m = nr_m
    P2 = Dd.P.copy()
    P2.index = pd.Index([f"{q}{LAB}" for q in Dd.P.index], name=Dm.P.index.name)
    self.P = pd.concat([Dm.P, P2])
    self.y = np.concatenate([Dm.y, Dd.y])
    self.pid_row = {p: k for k, p in enumerate(self.P.index)}
    self.ra = np.r_[Dm.ra, np.where(Dd.ra >= 0, Dd.ra + nr_m, -1)]
    self.rb = np.r_[Dm.rb, np.where(Dd.rb >= 0, Dd.rb + nr_m, -1)]
    self.has_tok = np.r_[Dm.has_tok, Dd.has_tok]
    self.usable_pair = np.r_[Dm.usable_pair, Dd.usable_pair]
    self.usable_rxn = np.r_[Dm.usable_rxn, Dd.usable_rxn]
    self.dH = np.concatenate([Dm.dH, Dd.dH])
    self.H10 = np.concatenate([Dm.H10, Dd.H10])
    self.rids = np.r_[Dm.rids, Dd.rids]                        # reaction id of every reaction row (duplicates)
    self.is_d0 = np.zeros(len(self.P), dtype=bool)
    self.rung, self.arm, self.labels = "mlts_mix", Dm.arm, Dm.labels
    self.split_name = Dm.split_name
    self.linfo = Dm.linfo
    self.hinfo = dict(Dm.hinfo, label_rung=Dd.hinfo)
    self.srep = getattr(Dm, "srep", None)
    self.folds = Dm.folds
    self.pair_list = sorted(list(Dm.pair_list) + [f"{q}{LAB}" for q in Dd.pair_list])
    self.inputs = dict(mlts=Dm.inputs, label=Dd.inputs, rule=DECISIONS["K-4 C3 mix"])
    self.sha = {k: (Dm.sha[k] if Dm.sha[k] == Dd.sha.get(k) else f"mlts:{Dm.sha[k]}|label:{Dd.sha.get(k)}")
                for k in Dm.sha}
    self.opened = list(Dm.opened) + list(Dd.opened)


def _mix_split_checks(self, k, out):
    """K-4 leak checks of a mixed split: base pair ids / reaction ids of the test pairs never in training, no label copy
    of an inner validation pair on the inner training side."""
    import numpy as np
    bid = lambda rows: {base_id(q) for q in np.asarray(self.P.index)[np.asarray(rows, dtype=int)]}   # noqa: E731
    rx = lambda rows: set(self.rids[self.ra[rows]].tolist()) | set(self.rids[self.rb[rows]].tolist())  # noqa: E731
    if bid(out["test"]) & bid(out["train"]):
        stop(f"C3 fold {k}: a test pair (MLTS or label copy) in training")
    if rx(out["test"]) & rx(out["train"]):
        stop(f"C3 fold {k}: a test reaction in training (by reaction id)")
    if bid(out["inner_val"]) & bid(out["inner_train"]):
        stop(f"C3 fold {k}: an inner validation pair (or its label copy) on the inner training side")
    for j, f in enumerate(out.get("inner") or []):
        if bid(f["val"]) & bid(f["train"]):
            stop(f"C3 fold {k} inner {j}: a validation pair (or its label copy) on the inner training side")


def make_mix_atoms(Dm, Dd):
    import numpy as np
    TA = X.TA

    class MixAtoms(TA.LadData):
        """C3 atom data (K-4): fu2 train.Data / LadData attributes over [MLTS | label TS] reaction rows."""

        def __init__(self, Dm, Dd):                            # LadData.__init__ (one rung) is not called
            X.PA.layout_check(Dd, Dm)
            _mix_common(self, Dm, Dd)
            nmax = max(Dm.nmax, Dd.nmax)
            self.nmax = nmax
            for a in ("onehot", "bin", "num", "pos"):
                setattr(self, a, np.concatenate([_pad(getattr(Dm, a), nmax, (1,)), _pad(getattr(Dd, a), nmax, (1,))]))
            for a in ("mask", "frag"):
                setattr(self, a, np.concatenate([_pad(getattr(Dm, a), nmax, (1,)), _pad(getattr(Dd, a), nmax, (1,))]))
            self.etype = np.concatenate([_pad(Dm.etype, nmax, (1, 2)), _pad(Dd.etype, nmax, (1, 2))])
            self.n = np.r_[Dm.n, Dd.n]
            self.trep, self.elements, self.num_cols, self.bin_cols = Dm.trep, Dm.elements, Dm.num_cols, Dm.bin_cols
            self.d_in = Dm.d_in
            self.row = dict(Dm.row)                            # rxn id -> MLTS reaction row (label rows: + nr_m)
            self.glob = np.concatenate([Dm.glob, Dd.glob])
            self.d_glob = self.glob.shape[1]
            if Dm.d_glob != Dd.d_glob:
                stop("C3: global token widths differ")
            self.s_splits_sha = self.sha["s_splits"]
            self.pair_list_sha = TA.C.list_sha([(q,) for q in self.pair_list])

        def split(self, a, b=None):
            k = int(a if b is None else b)
            Dm, Dd = self.parts
            sm, sd = Dm.split(k), Dd.split(k)
            Np = self.Np
            out = dict(train=np.r_[sm["train"], sd["train"] + Np], test=sm["test"],
                       inner_train=np.r_[sm["inner_train"], sd["inner_train"] + Np], inner_val=sm["inner_val"])
            _mix_split_checks(self, k, out)
            ids = {key: [str(q) for q in np.asarray(self.P.index)[out[key]]] for key in out}
            ls = TA.C.list_sha
            out.update(dropped=dict(mlts=sm["dropped"], label=sd["dropped"]),
                       pairs_sha=ls([(q,) for q in ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                     + ids["inner_val"]]),
                       val_sha=ls([(q,) for q in ids["inner_val"]]), test_sha=ls([(q,) for q in ids["test"]]),
                       n_label=dict(train=int(len(sd["train"])), inner_train=int(len(sd["inner_train"]))))
            return out

    return MixAtoms(Dm, Dd)


def make_mix_xgb(Dm, Dd):
    import numpy as np
    TX = X.TX

    class MixX(TX.XData):
        """C3 XGB data (K-4): XData attributes over [MLTS | label TS] reaction rows."""

        def __init__(self, Dm, Dd):                            # XData.__init__ (one rung) is not called
            if list(Dm.feat_cols) != list(Dd.feat_cols) or list(Dm.glob_cols) != list(Dd.glob_cols):
                stop("C3: XGB feature columns differ between rung mlts and rung dft")
            _mix_common(self, Dm, Dd)
            self.glob_cols = Dm.glob_cols
            self.X = np.concatenate([Dm.X, Dd.X])
            self.feat_cols, self.feat_cols_sha = Dm.feat_cols, Dm.feat_cols_sha
            self.pid = self.P.index.to_numpy()
            self.core = np.r_[Dm.core, Dd.core]
            self.pair_list_sha = TX.C.list_sha([(q,) for q in self.pair_list])
            self._splits = {}

        def split(self, a, b=None):
            k = int(a if b is None else b)
            if k in self._splits:
                return self._splits[k]
            Dm, Dd = self.parts
            sm, sd = Dm.split(k), Dd.split(k)
            Np = self.Np
            if len(sm["inner"]) != len(sd["inner"]):
                stop("C3: inner fold counts differ")
            inner = []
            for fm, fd in zip(sm["inner"], sd["inner"]):     # the two rungs' GroupKFold partitions (same lists)
                inner.append(dict(train=np.r_[fm["train"], fd["train"] + Np], val=fm["val"],
                                  dropped=dict(mlts=fm["dropped"], label=fd["dropped"])))
            out = dict(train=np.r_[sm["train"], sd["train"] + Np], test=sm["test"],
                       inner_train=np.r_[sm["inner_train"], sd["inner_train"] + Np], inner_val=sm["inner_val"],
                       inner=inner)
            if not (np.array_equal(inner[0]["train"], out["inner_train"]) and np.array_equal(inner[0]["val"],
                                                                                             out["inner_val"])):
                stop(f"C3 fold {k}: inner fold 0 differs from inner_train / inner_val")
            for j, f in enumerate(inner):                      # label copies follow their pair's inner fold (cores)
                cv = set(self.core[f["val"]].tolist())
                if set(self.core[f["train"]].tolist()) & cv:
                    stop(f"C3 fold {k} inner {j}: a core on both sides")
            _mix_split_checks(self, k, out)
            ids = {key: [str(q) for q in self.pid[out[key]]] for key in ("train", "test", "inner_train", "inner_val")}
            ls = TX.C.list_sha
            out.update(dropped=dict(mlts=sm["dropped"], label=sd["dropped"]),
                       pairs_sha=ls([(q,) for q in ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                     + ids["inner_val"]]),
                       val_sha=ls([(q,) for q in ids["inner_val"]]), test_sha=ls([(q,) for q in ids["test"]]),
                       inner_sha=ls([(q,) for f in inner for q in [str(z) for z in self.pid[f["train"]]] + ["|"]
                                     + [str(z) for z in self.pid[f["val"]]] + ["||"]]),
                       n_label=dict(train=int(len(sd["train"])), inner_train=int(len(sd["inner_train"]))))
            self._splits[k] = out
            return out

    return MixX(Dm, Dd)


# ================================================================ C2 / C3 units
def _unit_ctx(plan, cfg, u, seeds, tune_seed):
    test = plan.get("test") or {}
    inp = plan["inputs"]
    CTX.clear()
    CTX.update(cfg=cfg, seeds=list(seeds), tune_seed=int(tune_seed),
               fp=dict(code=CODE_VERSION, config=cfg, seed_set=int(CONFIGS[cfg]["seed_set"]), seed_step=SEED_STEP,
                       tune_seed=int(tune_seed), final_seeds=list(seeds), refs=plan["refs"],
                       mlts_table_sha256=inp["table_sha256"], mlts_t_report_sha256=inp["t_report_sha256"],
                       label_table_sha256=inp["label_table_sha256"] if cfg == "C3" else None,
                       prereg_md_sha256=(plan.get("prereg") or {}).get("md_sha256"), standin=inp.get("standin"),
                       test=test or None))


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


def _write_k(x, cfg, name, root, rec, js, kind, cached, seeds, tune_seed):
    valp, trainp = root / f"{name}__val.parquet", root / f"{name}__train_pairs.parquet"
    pq = root / f"{name}.parquet"
    if not (valp.is_file() and trainp.is_file() and pq.is_file()):
        stop(f"{js} exists without its K sidecars {valp.name} / {trainp.name} (K-9; delete the unit's files to retrain)")
    if M.sha256(pq) != rec.get("parquet_sha256"):
        stop(f"{pq} changed after its json")
    fp = rec.get("fingerprint") or {}
    K = dict(unit=x["unit"], config=cfg, name=name, kind=kind, ladder_json=str(js), ladder_json_sha256=M.sha256(js),
             parquet=str(pq), parquet_sha256=rec.get("parquet_sha256"), val=str(valp), val_sha256=M.sha256(valp),
             train_pairs=str(trainp), train_pairs_sha256=M.sha256(trainp), fingerprint_k=fp.get("k"),
             seeds=list(seeds) if seeds is not None else None, tune_seed=tune_seed, smoke=bool(rec.get("smoke")),
             wall_seconds=rec.get("wall_seconds"), gpu=fp.get("gpu"), host=rec.get("host"), cached=bool(cached),
             job=os.environ.get("SLURM_JOB_ID"), task=os.environ.get("SLURM_ARRAY_TASK_ID"), code=CODE_VERSION,
             code_sha256=code_sha(), written=IM.now())
    jdump(done_path(cfg, name), K)
    return K


def run_c23(x, plan):
    """one C2 / C3 unit through TA.run_unit / TX.run_unit (K-1, K-3, K-4)."""
    import numpy as np
    cfg, u, name = parse_unit(x["unit"])
    kind = "xgb" if u["model"] == "XGB" else "atoms"
    mod = X.TX if kind == "xgb" else X.TA
    test = plan.get("test") or {}
    root = UNITS / cfg
    if done_path(cfg, name).is_file():
        log(f"{x['unit']}: done (K record present)")
        return 0
    if kind == "xgb":
        D = mod.get_data("mlts", u["arm"]) if cfg != "C3" else make_mix_xgb(mod.get_data("mlts", u["arm"]),
                                                                           mod.get_data("dft", u["arm"]))
    else:
        D = mod.get_data("mlts", u["arm"]) if cfg != "C3" else make_mix_atoms(mod.get_data("mlts", u["arm"]),
                                                                             mod.get_data("dft", u["arm"]))
    s = int(CONFIGS[cfg]["seed_set"])
    if kind == "atoms":
        seed0, fs0 = ORIG["seed"], ORIG["final_seeds"]
        tune_seed = seed0 + SEED_STEP * s
        seeds = [v + SEED_STEP * s for v in fs0][:int(test.get("final_seeds") or len(fs0))]
    else:
        tune_seed, seeds = None, None
    _unit_ctx(plan, cfg, u, seeds or [], tune_seed or 0)
    sp = D.split(u["fold"])
    js = Path(os.path.abspath(root / f"{name}.json"))
    if js.is_file():                                           # K-9: LADDER json present, K record missing
        if kind == "atoms":
            X.F.SEED = tune_seed
            try:
                now = mod.fingerprint(D, u, sp, bool(test))
            finally:
                X.F.SEED = ORIG["seed"]
        else:
            idx = sorted({int(c) for c in test["cells"]}) if test.get("cells") else list(range(len(mod.GRID)))
            now = mod.fingerprint(D, u, sp, idx, bool(test))
        old = M.read_json(js)
        if _no_gpu(old.get("fingerprint")) != _no_gpu(now):
            dk = sorted(k for k in set(old.get("fingerprint") or {}) | set(now)
                        if k != "gpu" and (old.get("fingerprint") or {}).get(k) != now.get(k))
            stop(f"{js} has another fingerprint (keys {dk}): other code / inputs (K-9)")
        K = _write_k(x, cfg, name, root, old, js, kind, True, seeds, tune_seed)
        log(f"{x['unit']}: cached LADDER unit accepted ({K['gpu']})")
        return 0
    CAP.clear()
    CAP.update(on=True, kind=kind, D=D, u=u, cfg=cfg, sp=sp, js=js, name=name, iva=np.asarray(sp["inner_val"]),
               items=[], prep=[], par=[], valp=root / f"{name}__val.parquet",
               trainp=root / f"{name}__train_pairs.parquet", val_ok=False)
    t0 = time.time()
    try:
        if kind == "atoms":
            X.F.SEED = tune_seed                               # K-3 (s0: unchanged)
            cells = test.get("cells")
            rec = mod.run_unit(D, u, root=root, n_epochs=int(test.get("epochs") or X.F.MAX_EPOCHS), seeds=seeds,
                               cells=cells, log=50, smoke=bool(test))
        else:
            cells = test.get("cells")
            CAP["idx"] = sorted({int(c) for c in cells}) if cells else list(range(len(mod.GRID)))
            rec = mod.run_unit(D, u, root=root, cells=cells, smoke=bool(test))
    finally:
        if kind == "atoms":
            X.F.SEED = ORIG["seed"]
        CAP["on"] = False
    if not CAP.get("val_ok"):
        stop(f"{x['unit']}: run_unit returned without the K sidecars (cached LADDER json written elsewhere?)")
    K = _write_k(x, cfg, name, root, rec, js, kind, False, seeds, tune_seed)
    ch = rec.get("chosen") if kind == "atoms" else {t: (rec.get("chosen") or {}).get(t, {}).get("grid_index") for t in MAIN}
    log(f"{x['unit']}: trained ({time.time() - t0:.0f} s, chosen {ch}) val oi "
        f"{rec['val_kcal_mae']['oi']:.4f} cpcm {rec['val_kcal_mae']['cpcm']:.4f} -> {done_path(cfg, name).name}")
    return 0


# ================================================================ C1v units (C1 inner validation)
def _ladder_unit(u):
    name = f"L__dft__{u['arm']}__{u['model']}__f{u['fold']}"
    js, pq = LAD_UNITS / f"{name}.json", LAD_UNITS / f"{name}.parquet"
    if not (js.is_file() and pq.is_file()):
        not_ready(f"{js} / {pq} missing")
    rec = M.read_json(js)
    if rec.get("smoke"):
        stop(f"{name}: smoke unit")
    if M.sha256(pq) != rec.get("parquet_sha256"):
        stop(f"{pq} changed after its json")
    return name, js, pq, rec


def run_c1v_atoms(x, plan):
    import numpy as np
    import pandas as pd
    import torch
    TA, FT, F, Mo, DC, PA = X.TA, X.FT, X.F, X.Mo, X.DC, X.PA
    cfg, u, name = parse_unit(x["unit"])
    test = plan.get("test") or {}
    root = UNITS / cfg
    if done_path(cfg, name).is_file():
        log(f"{x['unit']}: done")
        return 0
    lname, ljs, lpq, rec = _ladder_unit(u)
    if not TA.gates_ok(rec):
        stop(f"{lname}: a gate M-1' not ok")
    arm, model, k = u["arm"], u["model"], u["fold"]
    Dd, Dg = TA.get_data("dft", arm), TA.get_data("mlts", arm)
    lay = PA.layout_check(Dd, Dg)
    sp = Dd.split(k)
    fpd = DC.fingerprint_diff(rec.get("fingerprint"), ORIG["fingerprint"](Dd, u, sp), ("code_sha256", "gpu"),
                              ("train_atoms.py",))
    if not fpd["ok"]:
        stop(f"{lname}: fingerprint differs beyond code_sha256 (train_atoms.py) / gpu: {fpd}")
    if rec.get("val_pairs_sha256") != sp["val_sha"] or rec.get("test_pairs_sha256") != sp["test_sha"]:
        stop(f"{lname}: validation / test pairs differ from the current split")
    itr, iva, tr, te = sp["inner_train"], sp["inner_val"], sp["train"], sp["test"]
    ci = int(rec["chosen"]["index"])
    cell, trec = rec["chosen"]["cell"], rec["tune"][ci]
    if TA.norm(trec["cell"]) != TA.norm(cell):
        stop(f"{lname}: chosen cell != tune[{ci}].cell")
    sh = arm == "SH"
    fitf, _ = TA.arm_fns(arm)
    n_ep = int(test.get("epochs") or F.MAX_EPOCHS)
    t0 = time.time()
    r = fitf(Dd, model, cell, TA.VARIANT, itr, iva, seed=F.SEED, n_epochs=n_ep, log=50)   # the unit's tuning fit
    leak = TA.leak_check(Dd, r["std"], itr, (iva, te), te)
    yd = TA.final_kcal(Dd, r, iva)
    keep, rows_g = DC.deploy_rows(Dd, Dg, iva)
    dy = DC.check_same_pairs(Dd, Dg, iva, keep, rows_g)
    Tg = r["std"].tensors(Dg)
    yg = TA.final_kcal(Dg, dict(r, T=Tg), rows_g)
    beta = np.asarray(r["beta"], float) if sh else np.zeros(len(TARGETS))
    vk = np.abs(yd - Dd.y[iva]).mean(0)
    replica = dict(best_epoch=int(r["best_epoch"]), best_epoch_ladder=int(trec["best_epoch"]),
                   epochs_run=len(r["history"]), seconds=float(r["seconds"]), n_epochs_cap=n_ep,
                   val_kcal_mae=dict(zip(TARGETS, vk.tolist())), val_kcal_mae_ladder=trec["val_kcal_mae"],
                   max_abs_val_diff=float(max(abs(vk[j] - float(trec["val_kcal_mae"][t])) for j, t in enumerate(TARGETS))),
                   beta_equal_ladder=bool(not sh or np.array_equal(beta, np.asarray(trec["beta"], float))),
                   rule="re-run of the chosen tuning fit (cell, seed, inner_train, early stopping on dft inner_val); "
                        "GPU training is not bit-reproducible across devices: recorded, not gated (K-5)")
    V = [_val_frame(Dg, rows_g, u, yg[None], beta * Dg.dH[rows_g] if sh else np.zeros((len(rows_g), len(TARGETS))),
                    beta, "mlts", "tune_replica", cfg),
         _val_frame(Dd, iva, u, yd[None], beta * Dd.dH[iva] if sh else np.zeros((len(iva), len(TARGETS))), beta, "dft",
                    "tune_replica_dftin", cfg)]
    del r, Tg
    # descriptive: the unit's saved final models (trained on T_main incl. these inner_val pairs) + dft test regression
    std = TA.StdSH(Dd, tr) if sh else FT.Std(Dd, tr)
    T_d, T_g = std.tensors(Dd), std.tensors(Dg)
    Yte, Yg = [], []
    for s in F.FINAL_SEEDS:
        p = LAD_UNITS / f"{lname}__s{s - F.SEED}.pt"
        if not p.is_file():
            not_ready(f"{p} missing")
        obj = torch.load(p, map_location=FT.DEV)
        eq = PA.std_equal(std, obj["std"], sh)
        if TA.norm(obj.get("cell")) != TA.norm(cell) or obj.get("rung") != "dft" or not all(eq.values()):
            stop(f"{p}: cell / rung / saved std differ ({eq})")
        mdl = Mo.build(model, cell, Dd.d_in, len(TARGETS), Dd.d_glob, TA.VARIANT, F.DROPOUT).to(FT.DEV)
        mdl.load_state_dict(obj["state"])
        rr = dict(model=mdl, T=T_d, sy=std.sy, std=std)
        Yte.append(TA.final_kcal(Dd, rr, te))
        Yg.append(TA.final_kcal(Dg, dict(rr, T=T_g), rows_g))
        del mdl, obj, rr
    beta_f = np.asarray(std.beta, float) if sh else np.zeros(len(TARGETS))
    mine = DC.pred_frame(Dd, te, u, np.stack(Yte), beta_f * Dd.dH[te] if sh else np.zeros((len(te), len(TARGETS))),
                         beta_f, "dft")
    reg = DC.compare_unit(mine, pd.read_parquet(lpq))
    status = DC.gate_status(reg["max_abs_yhat"], allow_fp32=True)
    if status == "fail" or not (reg["same_pairs"] and reg["same_order"]):
        stop(f"{lname}: saved final models do not reproduce the LADDER unit parquet ({status}, {reg})")
    V.append(_val_frame(Dg, rows_g, u, np.stack(Yg), beta_f * Dg.dH[rows_g] if sh else
                        np.zeros((len(rows_g), len(TARGETS))), beta_f, "mlts", "final_models", cfg))
    del T_d, T_g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _write_c1v(x, cfg, name, root, u, lname, ljs, rec, Dd, sp, V, dict(
        replica=replica, regression_final_models=dict(status=status, **reg), leak=leak, layout_ok=bool(lay.get("ok")),
        n_inner_val=int(len(iva)), n_deploy=int(len(rows_g)), n_not_usable_in_mlts=int((~keep).sum()),
        y_dft_vs_mlts_max_abs=dy, fingerprint_diff=fpd, gpu=FT.versions().get("gpu"), seconds=time.time() - t0))


def run_c1v_xgb(x, plan):
    import numpy as np
    import pandas as pd
    TX, D1T, DC, PX = X.TX, X.D1T, X.DC, X.PX
    cfg, u, name = parse_unit(x["unit"])
    root = UNITS / cfg
    if done_path(cfg, name).is_file():
        log(f"{x['unit']}: done")
        return 0
    lname, ljs, lpq, rec = _ladder_unit(u)
    if not rec.get("gates_ok"):
        stop(f"{lname}: gates not ok")
    arm, k = u["arm"], u["fold"]
    Dd, Dg = TX.get_data("dft", arm), TX.get_data("mlts", arm)
    if list(Dd.feat_cols) != list(Dg.feat_cols):
        stop(f"XGB feature columns of rung mlts differ from rung dft ({arm})")
    sp = Dd.split(k)
    fpd = DC.fingerprint_diff(rec.get("fingerprint"), ORIG["fingerprint"](Dd, u, sp, list(range(len(TX.GRID)))),
                              ("code_sha256",), ("train_xgb.py",))
    if not fpd["ok"]:
        stop(f"{lname}: fingerprint differs beyond code_sha256 (train_xgb.py): {fpd}")
    if rec.get("val_pairs_sha256") != sp["val_sha"] or rec.get("test_pairs_sha256") != sp["test_sha"]:
        stop(f"{lname}: validation / test pairs differ from the current split")
    t0 = time.time()
    f0 = sp["inner"][0]
    tr0, va0, tr, te = f0["train"], f0["val"], sp["train"], sp["test"]
    gp = PX.gate_pairs(lname, te)
    pf = TX.prep_fit(Dd, tr0, dict(val=va0), gp)
    fin = TX.prep_fit(Dd, tr, dict(test=te), gp)
    NT = len(TARGETS)
    on = [arm == "SH" and j in TX.H_IDX for j in range(NT)]
    for ti, t in enumerate(TARGETS):
        tt = rec["tune"][t]
        if float(tt["beta_inner"][0]) != float(pf["beta"][ti]) or float(tt["sy_inner"][0]) != float(pf["sy"][ti]):
            stop(f"{lname} {t}: inner fold 0 beta / s_y differ from the unit json")
    sc0 = D1T.Scale(Dd.X[Dd.ra[tr0]], Dd.X[Dd.rb[tr0]], Dd.y[tr0, 0])
    scf = D1T.Scale(Dd.X[Dd.ra[tr]], Dd.X[Dd.rb[tr]], Dd.y[tr, 0])
    for sc, ev, rows in ((sc0, pf["evals"]["val"], va0), (scf, fin["evals"]["test"], te)):
        z = sc.z(Dd.X[Dd.ra[rows]], Dd.X[Dd.rb[rows]])
        if not (np.array_equal(z[0], ev[0]) and np.array_equal(z[1], ev[1])):
            stop(f"{lname}: the reconstructed d1_train.Scale does not reproduce prep_fit's z")
    keep, rows_g = DC.deploy_rows(Dd, Dg, va0)
    dy = DC.check_same_pairs(Dd, Dg, va0, keep, rows_g)
    zg0 = sc0.z(Dg.X[Dg.ra[rows_g]], Dg.X[Dg.rb[rows_g]])
    zgf = scf.z(Dg.X[Dg.ra[rows_g]], Dg.X[Dg.rb[rows_g]])
    cells = {t: int(rec["chosen"][t]["grid_index"]) for t in TARGETS}
    calls = [(pf["zd"], pf["zm"], pf["ys"][ti], pf["sy"][ti], TX.GRID[cells[t]], dict(val=pf["evals"]["val"], mlts=zg0))
             for ti, t in enumerate(TARGETS)]
    calls += [(fin["zd"], fin["zm"], fin["ys"][ti], fin["sy"][ti], TX.GRID[cells[t]],
               dict(test=fin["evals"]["test"], mlts=zgf)) for ti, t in enumerate(TARGETS)]
    res = TX.run_parallel(calls)
    yd, yg = np.zeros((len(va0), NT)), np.zeros((len(rows_g), NT))
    yte, ygf = np.zeros((len(te), NT)), np.zeros((len(rows_g), NT))
    b0, bf = np.asarray(pf["beta"], float), np.asarray(fin["beta"], float)
    bg0 = np.stack([b0[j] * Dg.dH[rows_g, j] if on[j] else np.zeros(len(rows_g)) for j in range(NT)], 1)
    bgf = np.stack([bf[j] * Dg.dH[rows_g, j] if on[j] else np.zeros(len(rows_g)) for j in range(NT)], 1)
    bad = []
    for ti, t in enumerate(TARGETS):
        kc, kf = res[ti][0], res[NT + ti][0]
        yd[:, ti] = pf["base"]["val"][:, ti] + kc["val"]
        mae = float(np.mean(np.abs(yd[:, ti] - Dd.y[va0, ti])))
        if mae != float(rec["tune"][t]["val_kcal_mae"]):
            bad.append((t, mae, rec["tune"][t]["val_kcal_mae"]))
        yg[:, ti] = bg0[:, ti] + kc["mlts"]
        yte[:, ti] = fin["base"]["test"][:, ti] + kf["test"]
        ygf[:, ti] = bgf[:, ti] + kf["mlts"]
    if bad:
        stop(f"{lname}: the inner fold 0 refit does not reproduce val_kcal_mae exactly: {bad[:3]}")
    mine = DC.pred_frame(Dd, te, u, yte[None], fin["base"]["test"], bf, "dft")
    reg = DC.compare_unit(mine, pd.read_parquet(lpq))
    status = DC.gate_status(reg["max_abs_yhat"], allow_fp32=False)
    if status != "exact" or not (reg["same_pairs"] and reg["same_order"]):
        stop(f"{lname}: the final refit does not reproduce the LADDER unit parquet ({status}, {reg})")
    V = [_val_frame(Dg, rows_g, u, yg[None], bg0, b0, "mlts", "tune_replica", cfg),
         _val_frame(Dd, va0, u, yd[None], pf["base"]["val"], b0, "dft", "tune_replica_dftin", cfg),
         _val_frame(Dg, rows_g, u, ygf[None], bgf, bf, "mlts", "final_models", cfg)]
    for v in V:
        v["cell_index"] = v.target.map(cells)
    replica = dict(val_kcal_mae=dict(zip(TARGETS, np.abs(yd - Dd.y[va0]).mean(0).tolist())),
                   val_kcal_mae_ladder={t: rec["tune"][t]["val_kcal_mae"] for t in TARGETS}, max_abs_val_diff=0.0,
                   exact=True, rule="inner fold 0 refit with the per-target chosen cell (CPU deterministic, = val_kcal_mae)")
    return _write_c1v(x, cfg, name, root, u, lname, ljs, rec, Dd, sp, V, dict(
        replica=replica, regression_final_models=dict(status=status, **reg), n_inner_val=int(len(va0)),
        n_deploy=int(len(rows_g)), n_not_usable_in_mlts=int((~keep).sum()), y_dft_vs_mlts_max_abs=dy,
        fingerprint_diff=fpd, n_jobs=TX.N_JOBS, seconds=time.time() - t0))


def _write_c1v(x, cfg, name, root, u, lname, ljs, rec, Dd, sp, V, info):
    import numpy as np
    import pandas as pd
    Vd = pd.concat(V, ignore_index=True)
    if not np.isfinite(Vd.yhat.to_numpy(float)).all():
        stop(f"{x['unit']}: non-finite inner-validation predictions")
    valp = root / f"{name}__val.parquet"
    M.write_parquet(valp, Vd)
    train_ids = sorted({str(q) for q in np.asarray(Dd.P.index)[sp["train"]]})
    J = dict(unit=u, k_unit=x["unit"], config=cfg, name=name, kind="xgb" if u["model"] == "XGB" else "atoms",
             fold=int(u["fold"]), ladder_unit=lname, ladder_json=str(ljs), ladder_json_sha256=M.sha256(ljs),
             ladder_val_kcal_mae=rec.get("val_kcal_mae"), chosen=rec.get("chosen"), val=str(valp),
             val_sha256=M.sha256(valp), val_pairs_sha256=sp["val_sha"], test_pairs_sha256=sp["test_sha"],
             fingerprint=dict(s_splits=Dd.sha["s_splits"], inputs_table=Dd.sha["inputs_table"], labels=Dd.sha["labels"],
                              k=dict(code=CODE_VERSION, config=cfg, refs=CTX.get("refs"),
                                     mlts_table_sha256=CTX.get("table_sha256"))),
             train_pair_ids=train_ids, train_pairs_note="the fold's T_main (LADDER final fit; the replica fit used its "
                                                         "inner_train subset)",
             smoke=bool(CTX.get("test")), code=CODE_VERSION, code_sha256=code_sha(), host=socket.gethostname(),
             job=os.environ.get("SLURM_JOB_ID"), written=IM.now(), rule=DECISIONS["K-5 C1v"], **info)
    jdump(done_path(cfg, name), J)
    rp = info["replica"]
    log(f"{x['unit']}: C1 inner validation written ({info.get('n_deploy')} MLTS inner_val pairs; replica max |dval| "
        f"{rp['max_abs_val_diff']:.3g}; final-model regression {info['regression_final_models']['status']})")
    return 0


# ================================================================ unit command
def cmd_unit(sub, pat, i, n):
    plan = load_plan()
    i = M.array_task(i)
    n = int(n)
    cands = [x for x in plan["units"] if x["arm"] == sub and (x["unit"] == pat or fnmatch.fnmatchcase(x["unit"], pat))]
    if not cands:                                              # a command that can never succeed: STOP, not resubmit
        print(f"STOP: no K_plan unit of '{sub}' matches {pat!r}", file=sys.stderr)
        return 3
    if not 0 <= i < n:
        print(f"slice {i} of {n}?", file=sys.stderr)
        return 2
    sl = cands[i::n]
    kinds = {"xgb" if parse_unit(x["unit"])[1]["model"] == "XGB" else "atoms" for x in sl}
    if len(kinds) != 1:
        stop("a slice mixes atom and XGB units (different environments); select one kind")
    kind = kinds.pop()
    todo = [x for x in sl if not unit_done(x)]
    if not todo:
        log(f"{sub} {pat} [{i}/{n}]: {len(sl)} units already done")
        return 0
    IM.ensure_env("fu2" if kind == "atoms" else "espley")
    setup(kind, plan)
    CTX["refs"], CTX["table_sha256"], CTX["test"] = plan["refs"], plan["inputs"]["table_sha256"], plan.get("test")
    log(f"{sub} {pat} [{i}/{n}]: {len(todo)} of {len(sl)} units to run ({kind}, refs {plan['refs']}, "
        f"table {Path(plan['inputs']['table']).name}{', TEST ' + json.dumps(plan['test']) if plan.get('test') else ''})")
    for x in todo:
        if reserve_left() < RESERVE_H:
            not_ready(f"less than {RESERVE_H} h of the {WALL_H} h wall left: not starting {x['unit']}")
        refs, tsha, tst = CTX.get("refs"), CTX.get("table_sha256"), CTX.get("test")
        if sub == "c1v":
            (run_c1v_xgb if kind == "xgb" else run_c1v_atoms)(x, plan)
        else:
            run_c23(x, plan)
        CTX.update(refs=refs, table_sha256=tsha, test=tst)
    return 0


# ================================================================ select (reactot)
def _gates_ok(rec, kind):
    if kind == "xgb":
        return bool(rec.get("gates_ok")) and all((rec.get("final") or {}).get(t, {}).get("gate_M1", {}).get("ok")
                                                 for t in TARGETS)
    fits = list(rec.get("tune") or []) + list(rec.get("final") or [])
    return bool(fits) and all(f["gate_M1"]["ok"] for f in fits)


def load_records(plan):
    test = plan.get("test")
    R = {}
    for x in plan["units"]:
        cfg, u, name = parse_unit(x["unit"])
        root = UNITS / cfg
        kind = "xgb" if u["model"] == "XGB" else "atoms"
        e = dict(x=x, cfg=cfg, u=u, name=name, kind=kind)
        if cfg == "C1v":
            J = M.read_json(root / f"{name}.json")
            if M.sha256(J["val"]) != J["val_sha256"]:
                stop(f"{J['val']} changed after its K record")
            if bool(J.get("smoke")) != bool(test):
                stop(f"{x['unit']}: smoke flag {J.get('smoke')} in a {'test' if test else 'production'} plan")
            if (J.get("fingerprint") or {}).get("k", {}).get("mlts_table_sha256") != plan["inputs"]["table_sha256"]:
                stop(f"{x['unit']}: made on another MLTS table")
            e.update(rec=J, val=J["val"], train_ids=J["train_pair_ids"])
        else:
            K = M.read_json(done_path(cfg, name))
            js = Path(K["ladder_json"])
            if M.sha256(js) != K["ladder_json_sha256"] or M.sha256(K["parquet"]) != K["parquet_sha256"] \
                    or M.sha256(K["val"]) != K["val_sha256"] or M.sha256(K["train_pairs"]) != K["train_pairs_sha256"]:
                stop(f"{x['unit']}: a unit file changed after its K record")
            rec = M.read_json(js)
            if bool(rec.get("smoke")) != bool(test):
                stop(f"{x['unit']}: smoke flag {rec.get('smoke')} in a {'test' if test else 'production'} plan")
            if not _gates_ok(rec, kind):
                stop(f"{x['unit']}: a gate M-1' not ok")
            fk = (rec.get("fingerprint") or {}).get("k") or {}
            if fk.get("config") != cfg or fk.get("refs") != plan["refs"] \
                    or fk.get("mlts_table_sha256") != plan["inputs"]["table_sha256"]:
                stop(f"{x['unit']}: fingerprint k block {fk} does not match the plan")
            e.update(rec=rec, krec=K, parquet=K["parquet"], val=K["val"], train=K["train_pairs"])
        R[x["unit"]] = e
    return R


def fu2_test_lists():
    sp = Path(M.read_json(FU2_S_SPLITS)["pairs"]).parent
    return {k: [str(q) for q in M.read_json(sp / f"T1__f{k}.json")["test"]] for k in range(NF)}


def build_preds(R, test_lists):
    import numpy as np
    import pandas as pd
    out, info = {}, {}
    for cfg in ("C2", "C2_s1", "C2_s2", "C3"):
        es = [e for e in R.values() if e["cfg"] == cfg]
        if not es:
            continue
        parts = [pd.read_parquet(e["parquet"])[PRED_COLS] for e in es]
        copied = []
        if CONFIGS[cfg]["seed_set"] > 0:                       # K-3: XGB seed free -> the C2 XGB units
            for e in R.values():
                if e["cfg"] == "C2" and e["kind"] == "xgb":
                    parts.append(pd.read_parquet(e["parquet"])[PRED_COLS])
                    copied.append(e["name"])
        P = pd.concat(parts, ignore_index=True)
        P["pair_id"] = P.pair_id.astype(str)
        P["fold"] = P.fold.astype(int)
        if P.duplicated(["pair_id", "arm", "model", "target"]).any():
            stop(f"{cfg}: duplicated prediction rows (a pair in two folds?)")
        if not np.isfinite(P.yhat.to_numpy(float)).all():
            stop(f"{cfg}: non-finite predictions")
        bad_lists, outside = [], 0
        for k, g in P[P.target == TARGETS[0]].groupby("fold"):
            sets = {(a, m): frozenset(s.pair_id) for (a, m), s in g.groupby(["arm", "model"])}
            if len(set(sets.values())) != 1:
                bad_lists.append(int(k))
            outside += int((~g.pair_id.isin(set(test_lists[int(k)]))).sum())
        if bad_lists or outside:
            stop(f"{cfg}: test-pair lists differ between arms / models in folds {bad_lists}; {outside} rows outside "
                 f"the fold's FU2 test list")
        out[cfg] = P
        info[cfg] = dict(n_rows=int(len(P)), n_pairs=int(P.pair_id.nunique()),
                         n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in P.groupby("fold")},
                         rungs=sorted(set(P.rung)), models=sorted(set(P.model)), arms=sorted(set(P.arm)),
                         xgb_copied_from_C2=copied)
    return out, info


def pmain_para():
    import pandas as pd
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "pos", "in_P_main"])
    S["pair_id"] = S.pair_id.astype(str)
    m = S.in_P_main.astype(str).str.lower().eq("true") & S.pos.fillna("").astype(str).eq("p")
    return set(S.pair_id[m])


def reps_from(R, cfg, test):
    """REP / REP8 of a config on its own units' val_kcal_mae (LADDER 13-1 / eval_geo2.reps)."""
    import pandas as pd
    import eval_geo2 as EG
    rows = []
    for e in R.values():
        if e["cfg"] != cfg:
            continue
        for t, v in (e["rec"].get("val_kcal_mae") or {}).items():
            rows.append(dict(rung=cfg, arm=e["u"]["arm"], model=e["u"]["model"], fold=e["u"]["fold"], target=t,
                             val=float(v)))
    U = pd.DataFrame(rows)
    nf = U.groupby(["rung", "arm", "model", "target"]).fold.nunique()
    if not test and (nf != NF).any():
        stop(f"{cfg}: units without 5 folds: {nf[nf != NF].index.tolist()[:5]}")
    rep, rep8, _ = EG.reps(U)
    return ({a: {t: rep[(cfg, a, t)] for t in TARGETS if (cfg, a, t) in rep} for a in ARMS},
            {t: list(rep8[(cfg, t)]) for t in TARGETS if (cfg, t) in rep8})


def reps_c1(R, test):
    """C1 = LADDER dft REP / REP8 (eval_geo2.reps on L_report + L_xgb_report); test plans: the C1v units' LADDER
    val_kcal_mae of the units present."""
    import pandas as pd
    import eval_geo2 as EG
    if not test:
        rep, rep8, _ = EG.reps(EG.load_units(EG.BASE_UNITS))
        src = "LADDER L_report.json + L_xgb_report.json (eval_geo2.reps)"
    else:
        rows = [dict(rung="dft", arm=e["u"]["arm"], model=e["u"]["model"], fold=e["u"]["fold"], target=t, val=float(v))
                for e in R.values() if e["cfg"] == "C1v" for t, v in (e["rec"].get("ladder_val_kcal_mae") or {}).items()]
        if not rows:
            return None, None, "no C1v units"
        rep, rep8, _ = EG.reps(pd.DataFrame(rows))
        src = "TEST: LADDER val_kcal_mae of the C1v units in the test plan"
    REP = {a: {t: rep[("dft", a, t)] for t in TARGETS if ("dft", a, t) in rep} for a in ARMS}
    REP8 = {t: list(rep8[("dft", t)]) for t in TARGETS if ("dft", t) in rep8}
    if C1_REPS.is_file() and not test:
        C1r = (M.read_json(C1_REPS).get("REP8") or {}).get("dft") or {}
        if {t: list(v) for t, v in C1r.items()} != REP8:
            stop(f"C1 REP8 differs from {C1_REPS}")
    return REP, REP8, src


def inner_val_scores(IV, rep8s, test):
    """K-6: pooled inner-val para MAE per config (REP8 models), (fold, pair) rows common to the configs."""
    import numpy as np
    pm = pmain_para()
    sel = {}
    if rep8s.get("C1"):
        sel["C1"] = ("C1v", "tune_replica", rep8s["C1"])
    for c in ("C2", "C3"):
        if rep8s.get(c):
            sel[c] = (c, "tune", rep8s[c])
    detail = {c: {} for c in sel}
    for t in PRIMARY9:
        rows = {}
        for c, (key, var, r8) in sel.items():
            if t not in r8:
                continue
            a, mo = r8[t]
            d = IV[(IV.config == key) & (IV.variant == var) & (IV.arm == a) & (IV.model == mo) & (IV.target == t)]
            d = d[d.pair_id.isin(pm)]
            if d.duplicated(["fold", "pair_id"]).any():
                stop(f"{c} {t}: duplicated inner-validation rows")
            rows[c] = d.set_index(["fold", "pair_id"])
        if not rows:
            continue
        common = None
        for d in rows.values():
            common = set(d.index) if common is None else common & set(d.index)
        common = sorted(common)
        for c, d in rows.items():
            q = d.loc[common] if common else d.iloc[:0]
            e = np.abs(q.yhat.to_numpy(float) - q.y.to_numpy(float))
            detail[c][t] = dict(model=list(sel[c][2][t]), n=int(len(q)), n_own=int(len(d)),
                                mae=float(e.mean()) if len(e) else None)
    score = {}
    for c in sel:
        v = [detail[c].get(t, {}).get("mae") for t in MAIN]
        score[c] = float(sum(v)) if all(x is not None for x in v) else None
    ok = {c: s for c, s in score.items() if s is not None}
    primary = min(ok, key=lambda c: (ok[c], PRIMARY_ORDER.index(c))) if ok else None
    return score, detail, primary


def c1_final_models_score(IV, rep8):
    """descriptive: the C1 dft units' saved final models (in-sample on inner_val) on the MLTS inner_val inputs."""
    import numpy as np
    if not rep8:
        return None
    pm = pmain_para()
    out = {}
    for t in MAIN:
        if t not in rep8:
            return None
        a, mo = rep8[t]
        d = IV[(IV.config == "C1v") & (IV.variant == "final_models") & (IV.arm == a) & (IV.model == mo) & (IV.target == t)]
        d = d[d.pair_id.isin(pm)]
        out[t] = dict(n=int(len(d)), mae=float(np.abs(d.yhat - d.y).mean()) if len(d) else None)
    s = [out[t]["mae"] for t in MAIN]
    return dict(per_target=out, oi_cpcm=float(sum(s)) if all(v is not None for v in s) else None,
                note="in-sample (the final models were trained on these inner_val pairs); not used for selection (K-5)")


def delta_h(preds, rep8):
    """K-7 / PREREG_V2 5 on the C2 seed sets' test predictions (REP8 of C2 fixed)."""
    import numpy as np
    import pandas as pd
    have = [c for c in ("C2", "C2_s1", "C2_s2") if c in preds]
    if len(have) < 2 or not rep8:
        return dict(available=False, seed_sets=have, why="fewer than two C2 seed sets or no REP8")
    if str(DEPLOY_W) not in sys.path:
        sys.path.insert(0, str(DEPLOY_W))
    import eval_w as EW
    ET = EW.ET
    S = ET.s_folds()
    H, per, mae, fams0, s_t = {}, {}, {}, None, None

    def pick(P, choice):
        return pd.concat([P[(P.target == t) & (P.arm == choice[t][0]) & (P.model == choice[t][1])] for t in TARGETS],
                         ignore_index=True)

    for c in have:
        P = preds[c]
        sel = pick(P, rep8)
        A = ET.frame(sel, c, S)
        fams = EW.family_points(A, c)
        if fams0 is None:
            fams0 = fams
            if not fams:
                return dict(available=False, seed_sets=have, why="no evaluation family (>= 12 points incl. H)")
            pts = pd.concat([fr.assign(fam=f) for f, fr in fams.items()])
            s_t = {t: float(pts[f"y_{t}"].std(ddof=0)) for t in EW.PRIMARY}
        elif sorted(fams) != sorted(fams0) or any(list(fams[f].index) != list(fams0[f].index) for f in fams0):
            stop(f"delta_H: {c} evaluation families / points differ from C2")
        Ht = EW.h_table(fams, s_t)
        H[c], per[c] = EW.h_value(Ht)
        q = A[A.in_P_main.astype(bool) & (A.pos == "p")]
        mae[c] = {t: float(np.abs(g.yhat - g.y).mean()) for t, g in q.groupby("target")}
    hv = np.array([H[c] for c in have])
    d = float(hv.max() - hv.min())
    dm = {t: float(max(mae[c][t] for c in have) - min(mae[c][t] for c in have)) for t in PRIMARY9}
    by_model = {}                                              # descriptive: every arm x model on the same families
    combos = sorted({(a, mo) for a, mo in zip(preds[have[0]].arm, preds[have[0]].model)})
    for a, mo in combos:
        hs = {}
        for c in have:
            P = preds[c]
            q = P[(P.arm == a) & (P.model == mo)]
            if set(q.target) != set(TARGETS):
                break
            A = ET.frame(q, f"{c}:{a}:{mo}", S)
            fm = EW.family_points(A, f"{c}:{a}:{mo}")
            if sorted(fm) != sorted(fams0):
                break
            hs[c] = EW.h_value(EW.h_table(fm, s_t))[0]
        if len(hs) == len(have):
            v = np.array(list(hs.values()))
            by_model[f"{a}/{mo}"] = dict(H=hs, delta_H=float(v.max() - v.min()), delta_H_rel=float((v.max() - v.min())
                                                                                                / v.mean()))
    return dict(available=True, seed_sets=have, H=H, H_per_target=per, delta_H=d, delta_H_rel=d / float(hv.mean()),
                para_mae=mae, delta_mae=dm, rep8=rep8, by_model=by_model, n_families=len(fams0),
                n_points=int(sum(len(v) for v in fams0.values())), s_t=s_t,
                definition="PREREG_V2 5: H = mean_t mean_f r_ft / s_t (DEPLOY_MODEL_W 7; eval_w.family_points / h_table "
                           "/ h_value), C2 REP8 models, seed sets s0 / s1 / s2 (tuning + final seeds + 1000 s); XGB "
                           "units are seed free (copied)")


def leakage(R, test_lists):
    """C-2 for the K units: training pairs (sidecar / C1v train_pair_ids) vs fold-k test pairs and fold-k reactions."""
    import pandas as pd
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "a_rxn", "b_rxn"])
    ab = {str(p): (int(a), int(b)) for p, a, b in zip(S.pair_id, S.a_rxn, S.b_rxn)}
    T = M.targets()
    rfold = dict(zip(T.rxn_id.astype(int), T.fold_T1.astype(int)))
    tests = {k: set(v) for k, v in test_lists.items()}
    n_over, overlaps, n_units = 0, [], 0
    for e in R.values():
        k = int(e["u"]["fold"])
        ids = e["train_ids"] if e["cfg"] == "C1v" else [str(q) for q in pd.read_parquet(e["train"]).pair_id]
        bad = sorted({p for p in ids if p not in ab or p in tests[k] or rfold.get(ab[p][0]) == k
                      or rfold.get(ab[p][1]) == k})
        n_units += 1
        if bad:
            n_over += len(bad)
            overlaps.append(dict(unit=e["x"]["unit"], n=len(bad), first=bad[:10]))
    return dict(pass_=bool(n_units and not n_over), n_units=n_units, n_overlap=n_over, overlaps=overlaps[:20],
                rule="training pair ids (S_folds ids, label copies mapped to their pair) never in the fold's FU2 test "
                     "list and never touching a fold-k reaction (mlt_common.targets fold_T1)")


def c1_reports():
    """{refs mode: C1 report} of train_k.py c1pred (results/C1_report.json = the registered refs mode, C1_<mode>_report
    .json = the other one), test root first."""
    out = {}
    for d in dict.fromkeys((KRES, M.RES)):
        for p in sorted(Path(d).glob("C1*_report.json")):
            J = M.read_json(p)
            if J.get("refs") and J["refs"] not in out:
                out[J["refs"]] = dict(J, _path=str(p))
    return out


def k_configs(R, plan, files):
    """K-14: K_report.configs = the configurations E / W / P read: {name: {name, predictions (path string), refs,
    table, units_dir, rung, ...}}: C1 (the registered refs mode), C1_<other mode> (descriptive), C2, C3; the C2 seed
    sets s1 / s2 only under seed_sets (delta_H), the C1v inner-validation units under inner_val_units."""
    configs, seed_sets = {}, {}
    for c in ("C2", "C2_s1", "C2_s2", "C3"):
        if c not in files:
            continue
        ent = dict(name=c, predictions=files[c]["path"], predictions_sha256=files[c]["sha256"],
                   n_rows=files[c].get("n_rows"), n_pairs=files[c].get("n_pairs"), rung=CONFIGS[c]["rung"],
                   seed_set=CONFIGS[c]["seed_set"], refs=plan["refs"], table=plan["inputs"]["table"],
                   units_dir=str(UNITS / c), n_units=sum(1 for e in R.values() if e["cfg"] == c),
                   gpu_h=float(sum((e["rec"].get("wall_seconds") or e["rec"].get("seconds") or 0) for e in R.values()
                                   if e["cfg"] == c and e["kind"] == "atoms") / 3600),
                   final="trained by final_w.py channel (W-final-C23)" if c == "C3" else None)
        (seed_sets if CONFIGS[c]["seed_set"] > 0 else configs)[c] = ent
    reg = plan["refs"]
    for refs, J in sorted(c1_reports().items(), key=lambda kv: kv[0] != reg):
        name = "C1" if refs == reg else f"C1_{refs}"
        p = J.get("predictions")
        configs[name] = dict(name=name, predictions=p, predictions_sha256=J.get("sha256"), refs=refs,
                             table=J.get("table"), units_dir=str(LAD_UNITS), rung="dft", report=J.get("_path"),
                             descriptive=name != "C1",
                             source="train_k.py c1pred -> predict_c1 (LADDER dft units, no retraining)")
    if "C1" not in configs:                                    # c1pred not run yet: path only (E checks refs itself)
        configs["C1"] = dict(name="C1", predictions=str(C1_PRED) if C1_PRED.is_file() else None, refs=None, table=None,
                             units_dir=str(LAD_UNITS), rung="dft", note="no C1 report with refs (train_k.py c1pred)")
    iv = dict(C1v=dict(units_dir=str(UNITS / "C1v"), n_units=sum(1 for e in R.values() if e["cfg"] == "C1v"),
                       rule=DECISIONS["K-5 C1v"]))
    return configs, seed_sets, iv


def cmd_c1pred(step, refs=None):
    """K-15: predict_c1 (C1, no retraining) on the H table of a refs mode: the registered primary (prereg_v2.json
    criteria.refs.primary) -> $MLT/preds/C1_predictions.parquet + results/C1_report.json; the other mode ->
    C1_<mode>_predictions.parquet + C1_<mode>_report.json (descriptive). predict_c1.py is imported unchanged; in this
    process only inputs_mlts.src_cfg('mlts').table points at the mode's table (min = mlts_min.parquet) and the parts /
    output paths get the mode's name."""
    import predict_c1 as PC
    J = M.read_json(PREREG_JSON) if PREREG_JSON.is_file() else {}
    reg = ((J.get("criteria") or {}).get("refs") or {}).get("primary")
    if reg not in ("min", "own"):
        if TEST_ROOT is None:
            not_ready("results/prereg_v2.json (criteria.refs.primary) missing: Phase R first")
        reg = "min"
    refs = refs or reg
    if refs not in ("min", "own"):
        stop(f"--refs {refs!r} (min | own)")
    tag = "" if refs == reg else f"_{refs}"
    orig = IM.src_cfg

    def src_cfg(name):
        S = orig(name)
        if name == "mlts" and refs == "min":
            S = dict(S, table=S["table_min"], refs="min")
        return S
    IM.src_cfg = src_cfg
    PC.PREDS = PREDS
    PC.PARTS = PREDS / f"parts_{refs}"
    PC.C1_PRED = PREDS / f"C1{tag}_predictions.parquet"
    PC.C1_REPS = PREDS / "C1_representatives.json" if TEST_ROOT is not None else PC.C1_REPS
    PC.C1_REPORT = KRES / f"C1{tag}_report.json"
    log(f"c1pred {step}: refs {refs} (registered {reg}), table {src_cfg('mlts')['table']}, -> {PC.C1_PRED.name}")
    if step == "atoms":
        return PC.cmd_atoms("mlts")
    if step == "xgb":
        return PC.cmd_xgb("mlts")
    if step != "collect":
        stop(f"c1pred step {step!r} (atoms | xgb | collect)")
    rc = PC.cmd_collect()
    if rc == 0 and PC.C1_REPORT.is_file():
        rep = M.read_json(PC.C1_REPORT)
        tab = Path(src_cfg("mlts")["table"])
        rep.update(refs=refs, refs_registered=reg, primary_refs=refs == reg, table=str(tab), table_sha256=M.sha256(tab),
                   prereg_json_sha256=M.sha256(PREREG_JSON) if PREREG_JSON.is_file() else None,
                   rule=DECISIONS["K-15 C1 refs"])
        jdump(PC.C1_REPORT, rep)
    return rc


def cmd_select(o):
    IM.ensure_env("reactot")
    import numpy as np
    import pandas as pd
    _geo2_import()
    plan = load_plan()
    test = plan.get("test")
    est = dict(units_total=len(plan["units"]), unit_wall_h=plan["est"]["unit_wall_h"], workers=plan["est"]["workers"])
    todo = [x["unit"] for x in plan["units"] if not unit_done(x)]
    base = dict(spec="DEPLOY_V2_MLTS.md 10 (K)", code=CODE_VERSION, code_sha256=code_sha(), written=IM.now(),
                plan=str(K_PLAN), plan_sha256=M.sha256(K_PLAN), refs=plan["refs"], test=test, est=est)
    if todo:
        jdump(K_REPORT, dict(base, complete=False, n_todo=len(todo), stop=None, todo=todo[:30]))
        not_ready(f"{len(todo)} K units not finished, e.g. {todo[:3]}")
    if not test:
        inp = plan["inputs"]
        if M.sha256(inp["table"]) != inp["table_sha256"]:
            stop(f"{inp['table']} changed after K_plan.json")
    R = load_records(plan)
    tl = fu2_test_lists()
    preds, pinfo = build_preds(R, tl)
    # L-9: one validation list per config x fold (C2 / C3: the LADDER unit json; C1v: its record)
    bad = []
    for c in sorted({e["cfg"] for e in R.values()}):
        for k in range(NF):
            vs = {e["rec"].get("val_pairs_sha256") for e in R.values() if e["cfg"] == c and e["u"]["fold"] == k}
            if len(vs) > 1:
                bad.append(f"{c} f{k}")
    if bad:
        stop(f"validation-pair lists differ between units of {bad} (L-9)")
    IV = pd.concat([pd.read_parquet(e["val"]) for e in R.values()], ignore_index=True)
    IV["pair_id"] = IV.pair_id.astype(str)
    IV["fold"] = IV.fold.astype(int)
    reps = {}
    for c in ("C2", "C3"):
        if any(e["cfg"] == c for e in R.values()):
            REP, REP8 = reps_from(R, c, bool(test))
            reps[c] = dict(REP=REP, REP8=REP8, source=f"{c} units' val_kcal_mae (LADDER 13-1, eval_geo2.reps)")
    if any(e["cfg"] == "C1v" for e in R.values()) or not test:
        REP, REP8, src = reps_c1(R, bool(test))
        if REP8 is not None:
            reps["C1"] = dict(REP=REP, REP8=REP8, source=src)
    rep8s = {c: v["REP8"] for c, v in reps.items()}
    score, detail, primary = inner_val_scores(IV, rep8s, bool(test))
    if primary is None:
        stop("no configuration has an inner-validation score for both oi and cpcm")
    c1fm = c1_final_models_score(IV, rep8s.get("C1"))
    dH = delta_h(preds, rep8s.get("C2"))
    lk = leakage(R, tl)
    rep1 = [e["rec"]["replica"] for e in R.values() if e["cfg"] == "C1v"]
    c1rep = dict(n_units=len(rep1), max_abs_val_diff=max((r["max_abs_val_diff"] for r in rep1), default=None),
                 n_best_epoch_equal=sum(1 for r in rep1 if "best_epoch" in r and r["best_epoch"] == r["best_epoch_ladder"]),
                 n_atom_units=sum(1 for r in rep1 if "best_epoch" in r),
                 units={e["x"]["unit"]: dict(max_abs_val_diff=e["rec"]["replica"]["max_abs_val_diff"],
                                             best_epoch=e["rec"]["replica"].get("best_epoch"),
                                             best_epoch_ladder=e["rec"]["replica"].get("best_epoch_ladder"),
                                             regression=e["rec"]["regression_final_models"]["status"])
                        for e in R.values() if e["cfg"] == "C1v"})
    # ---- outputs (stop on leakage before anything is published)
    rep = dict(base, complete=True, n_todo=0, stop=None, primary=primary, inner_val=score, inner_val_detail=detail,
               inner_val_rule=DECISIONS["K-6 primary"], c1_final_models_inner_val=c1fm, c1_replica=c1rep, reps=reps,
               delta_H=dH, leakage_C2=lk, decisions=DECISIONS)
    if not lk["pass_"]:
        rep.update(stop="C-2 leak in K units (training pairs touch the held-out fold)", complete=True)
        jdump(K_REPORT, rep)
        stop(f"C-2 leak in K units: {lk['n_overlap']} pairs ({lk['overlaps'][:2]})")
    files = {}
    for c, P in preds.items():
        p = PREDS / f"{c}_predictions.parquet"
        M.write_parquet(p, P)
        files[c] = dict(path=str(p), sha256=M.sha256(p), **pinfo[c])
    M.write_parquet(K_IV, IV)
    vrows = [dict(config=e["cfg"], seed_set=CONFIGS[e["cfg"]]["seed_set"], arm=e["u"]["arm"], model=e["u"]["model"],
                  fold=e["u"]["fold"], target=t, val_kcal_mae=float(v), n_val=e["rec"].get("n_val") or
                  e["rec"].get("n_inner_val"), val_pairs_sha256=e["rec"].get("val_pairs_sha256"))
             for e in R.values() if e["cfg"] != "C1v" for t, v in (e["rec"].get("val_kcal_mae") or {}).items()]
    vrows += [dict(config="C1v", seed_set=0, arm=e["u"]["arm"], model=e["u"]["model"], fold=e["u"]["fold"], target=t,
                   val_kcal_mae=float(v), n_val=e["rec"].get("n_deploy"), val_pairs_sha256=e["rec"].get("val_pairs_sha256"))
              for e in R.values() if e["cfg"] == "C1v"
              for t, v in (e["rec"]["replica"].get("val_kcal_mae") or {}).items()]
    M.write_csv(K_VAL, pd.DataFrame(vrows))
    configs, seed_sets, iv_units = k_configs(R, plan, files)
    rep.update(configs=configs, seed_sets=seed_sets, inner_val_units=iv_units,
               predictions={c: v["path"] for c, v in files.items()},
               inner_val_predictions=dict(path=str(K_IV), sha256=M.sha256(K_IV)),
               k_val=dict(path=str(K_VAL), sha256=M.sha256(K_VAL)),
               configs_rule=DECISIONS["K-14 report"])
    jdump(K_REPORT, rep)
    log(f"select: primary {primary} (inner-val oi + cpcm para MAE {json.dumps({c: round(v, 4) for c, v in score.items() if v is not None})}); "
        f"delta_H {dH.get('delta_H')}; leakage pass {lk['pass_']} -> {K_REPORT}")
    return 0


# ================================================================ main
def parser():
    ap = argparse.ArgumentParser(prog="train_k.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan")
    p.add_argument("--standin", default=None, help="TEST: dft = the LADDER dft rung files registered as rung mlts")
    p.add_argument("--refs", default=None, choices=("min", "own"), help="TEST: override the pre-registered refs")
    p.add_argument("--configs", default=None)
    p.add_argument("--arms", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--folds", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cells", default=None)
    p.add_argument("--final-seeds", dest="final_seeds", type=int, default=None)
    for s in ("c2", "c3", "c1v"):
        q = sub.add_parser(s)
        q.add_argument("unit")
        q.add_argument("i")
        q.add_argument("n")
    sub.add_parser("select")
    sub.add_parser("status")
    q = sub.add_parser("c1pred")
    q.add_argument("step", choices=("atoms", "xgb", "collect"))
    q.add_argument("--refs", default=None, choices=("min", "own"))
    return ap


def main(argv=None):
    o = parser().parse_args(argv)
    if o.cmd == "plan":
        return cmd_plan(o)
    if o.cmd in ("c2", "c3", "c1v"):
        return cmd_unit(o.cmd, o.unit, o.i, o.n)
    if o.cmd == "c1pred":
        return cmd_c1pred(o.step, o.refs)
    if o.cmd == "select":
        return cmd_select(o)
    if o.cmd == "status":
        return cmd_status(o)
    parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
