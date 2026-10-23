#!/usr/bin/env python3
r"""evaluate.py — D1-REF·LADDER evaluation (docs/specs/D1LADDER.md 4-5, 11, 13, 14-1, 16 E). sbatch only. Nothing in the
E1 part was run when it was written (login node): every path is untested until it runs inside a job (start with
`e1-selftest`, which needs no Phase M output).

LINE (spec 4-5; the unchanged contract of tests/run_tests.py L-10, C-11 / C-18): per target t, beta_t = sum(dH_t * y_t) /
sum(dH_t^2) on exactly the given pairs, no intercept (antisymmetric; = D1-HF Phase A report.py 'a'), c_ghost beta = 0
(no HF counterpart); prediction beta * dH. line_beta(dH, y) -> beta [10] in TARGETS order; line_predict(beta, dH).
The module top level needs numpy and lad_common only (pandas is optional there), so L-10 can import it in any env.

CLI (submit from analysis/d1_pair through jobs/py.slurm with PYENV=espley, script ../d1_ladder/evaluate.py; exit 0 =
done, 2 = not ready / usage, 3 = STOP)
  python evaluate.py e1 [--stage 1] [--labels P] [--no-figures]
        Phase E1 (spec 11, 13; stage 1 = rungs dft, g1) -> results/E1_metrics.csv, E1_bends.csv, E1_bend_rates.json,
        E1_goal.csv, E1_compare.csv, E1_label_effect.csv, E1_report.json (provenance, REP, gates, verdicts, decisions)
        and results/figures/figL1_ladder, figL2_targets, figL3_bends (.png, .pdf, .csv). --stage 2 (E2, spec 12): exit 2.
        Exit 2 also while the FU2 Phase E code is not merged and committed in this worktree (spec 11, E1-2).
  python evaluate.py e1-selftest [--no-figures]
        the same pipeline on SYNTHETIC predictions over the real pair universe (S_folds.csv, the FU2 T1 split files,
        F_pairs_cr.csv as 'new' labels, the real FU2 / rev 9 frames for the label effect) -> $LAD/e1_selftest/ (the
        E1_* files, figures/, E1_selftest.json with the checks); exit 3 when a check fails. Never writes results/.
        The only command that may use the unmerged d1-pair copy of the FU2 Phase E code (E1-2).

Inputs (read only)
  results/L_predictions.parquet + L_report.json + L_val.csv (train_atoms.py collect --stage 1 --with-xgb; columns pair_id,
  ref_kind, pos, fold, rung, arm, model, target, y, yhat, base, beta, yhat_seed_sd), $LAD/units/
  L__<rung>__<arm>__<model>__f<k>.json (val_kcal_mae per target, fingerprint), the Phase P files through inputs.py
  (P_report, pair sets P_dft / P_g1 / P_01, fold copies, pairs_frame dH with h = cr), the label file (default
  $LAD/F_pairs_cr2.csv: V_report all_pass and outputs sha256), D1PAIR S_folds.csv and F_pairs.csv (original labels),
  results/PREREG_D1PAIR_ADD3.md / prereg_d1pair_add3.json, FU2 Phase E fu2/evaluate.py + results/fu2/E_tests.json, FU2
  M_predictions.parquet and the rev 9 / FU1 unit parquets (through FU2 load_all), FU2 E_representative.json /
  E_metrics.csv when their E_provenance.json record matches the imported evaluate.py.

What E1 computes (spec 11, 13)
  models     AW, GNN, TF, XGB (unit predictions), REP (spec 13-1: per rung x arm x target the model with the lowest mean
             validation kcal MAE over the 5 outer folds, ties AW -> GNN -> TF -> XGB, unit jsons only), ENS4 (equal-weight
             mean of the 4 test predictions), LINE (line_beta on the fold's outer-train T_main pairs of P_<rung>, spec 4-5),
             null (0).
  sets       pair_set own = P_<rung>, P_01 = P_dft & P_g1; inside each the FU2 metrics_tab evaluation sets (P_main
             reference = primary, A_main, P_sens, P_main_no_d0_overlap) x positions (all, p, m, o, direct).
  metrics    FU2 metrics_tab (n, MAE, RMSE; R2_fam median / quartiles, rho error median on P_main x all via
             d1_eval.family_eval) + para MAE 95 % CI (P_main x p rows) = d1_eval.boot_diff(e, 0, core), 2,000 core
             resamples, seed 20261019.
  bends      FU2 bends_tab (hammett_eval.calls on [DFT, prediction], para families >= 12 points incl. H) per rung x arm x
             model x pair set; d1_eval.bend_rates (sensitivity, false-positive rate, V sensitivity, Clopper-Pearson) on
             all / hidden_V / hidden_bend families.
  goal       spec 13-2 on REP for oi AND cpcm, own set, point estimates: para MAE <= 0.13, sensitivity >= 0.7, FPR <= 0.1;
             'surely' = passed and para MAE CI upper bound <= 0.13; ENS4 secondary; AW / GNN / TF / XGB descriptive; all 9
             targets descriptive (stage 1 R2_fam >= 0.5, beats null); spec 13-3 stage 1 = 'dft 통과 / 실패' only.
  compare    spec 13-4 with FU2 compare (d1_eval.boot_diff, d = MAE(x) - MAE(b)): SH vs S per rung x model, dft vs g1
             per arm x model on P_01, REP vs LINE per rung x arm; eval P_main and P_main para; 10 targets.
  label eff. g1·S new-label results vs FU2 (A+G configs, FU2 REP) and rev 9 arm C (C-XGB, C-rep) original-label results
             for d1, d2, barrier (MAE, para MAE, R2_fam, rho error, bend calls), the FU2 / rev 9 predictions also scored
             on the new labels; DFT truth bend calls original vs new labels (bends_tab = hammett_eval.calls).
  STOP       spec 16 E: every model / arm of a rung must have exactly the rung's test pairs (fold, pair_id) per target
             (FU2 pair_sets_gate), identical P_01 lists across rungs, FU2 pairs = LADDER g1 pairs for the label effect.

Decisions (user 2026-10-22: every judgment = the recommended option; also in E1_report.json 'decisions')
  E1-1  env = env_espley (d1_pair jobs/env.sh PYENV=espley: numpy 1.26.0, pandas 2.1.1, pyarrow 17.0.0 = the FU2 E-T
        scoring versions, + scipy (Clopper-Pearson) and matplotlib 3.8.4 (figures); rev 9 d1_eval ran there). Wrapper =
        analysis/d1_pair/jobs/py.slurm (train_xgb.py XG-2 precedent); no new wrapper file.
  E1-2  FU2 Phase E evaluate.py is imported by file path as module _lad_fu2_evaluate. `e1` uses only this worktree's
        analysis/d1_pair/fu2/evaluate.py (spec 11: E1 after FU2 Phase E is committed and merged): it, fu2/tests/
        test_eval.py and results/fu2/E_tests.json must be tracked and unchanged since HEAD here, else exit 2 (merge
        first). `e1-selftest` may fall back to the d1-pair worktree's copy. fu2_common, d1_eval, hammett_eval (and
        d1pair_common through lad_common) are imported from THIS worktree first, so both copies run the same scoring
        code; sys.path is restored after the import. Gate (STOP), both commands: E_tests.json next to the imported file
        has gate_pass, evaluate_sha256 = its sha256 and test_sha256 = sha256 of the test_eval.py next to it (= FU2
        require_review), and the reused files (REUSED) equal the E_tests.json code_sha256 pins (matched by
        repo-relative path). Recorded, not gated: E-T ran in env_d1fu2, E1 runs in env_espley (E1-1). Source path, copy,
        sha256, git states: E1_report.json.
  E1-3  REP from the unit jsons' val_kcal_mae before any prediction file is read (pandas.read_parquet is blocked for
        L_predictions.parquet and $LAD/units while it runs; FU2 D7 pattern); the unit values are cross-checked against
        L_report.json and L_val.csv; one validation-pair list per rung x fold (L-9). Recorded, not changed (train_atoms
        MA-6n): atom val_kcal_mae is the best-epoch value on the early-stopping pairs, XGB's the fold-0 score of a cell
        chosen by the 4-fold mean.
  E1-4  LINE and null are arm-independent reference rows (spec 4-5): listed under both arms with identical values
        (arm_independent = True) so the tables are rectangular; computed once; in E1_compare the dft vs g1 LINE row is
        listed once with arm '-'; figL3 draws LINE once per rung and leaves null out (degenerate at FPR 0, sens 0).
  E1-5  LINE beta per fold from inputs.fold_lists train (outer-train T_main restricted to P_<rung>) and inputs.pairs_frame
        dH (h = cr) with line_beta; it must equal every SH unit's base and beta (<= 1e-9 kcal/mol, relative 1e-9) and
        every S row must carry base = beta = 0, else STOP (spec 4-2 / 4-5, L-2).
  E1-6  pair_set own / P_01 x the FU2 evaluation sets x positions; the para CI only on P_main x p rows (spec 11), NaN
        elsewhere; mae_null / mae_rel_null = the null row of the same rung x arm x pair set x target x set x position.
  E1-7  JSON NaN (bend rates with a zero DFT denominator, Clopper-Pearson of n = 0) is written as null.
  E1-8  spec 13-2 with NaN sensitivity / FPR (no DFT-bent / DFT-straight family) = criterion not met (d1_eval.stages
        rule); bend rates = the 'all' population (d1_eval.bend_rates, no family filter; ADD3 5-2 / PR-15). Rev 9 stage 0
        (B < null and a Holm test against arm A) has no LADDER counterpart (no arm A; FU2 E-4 precedent): pre-registered
        in ADD3 5-2 (PR-14) as not computable, column stage0_rev9 = 'not computable'; its descriptive substitute
        beats_null_point = P_main MAE < null P_main MAE (point); descriptive stage 1 = R2_fam median >= 0.5.
  E1-9  13-4 sign = FU2 D5 (b = the first named): SH vs S -> b = SH, x = S; dft vs g1 -> b = dft, x = g1; REP vs LINE ->
        b = REP, x = LINE; d = MAE(x) - MAE(b) > 0 = b better (= what HF / the DFT geometry / the model adds). Models
        AW, GNN, TF, XGB, REP, ENS4 (+ LINE for dft vs g1); REP may be a different model on each side (source_b /
        source_x). Pair sets: own for SH vs S and REP vs LINE, P_01 for dft vs g1.
  E1-10 label effect on the FU2 evaluation pairs (T1 test pairs with atom data) = LADDER g1 test pairs (STOP if not);
        FU2 REP = its E_representative.json only when E_provenance.json records it for the imported evaluate.py (else
        omitted, noted); the FU2 / rev 9 frames scored against the new labels too (trained_on = original); DFT calls
        on 4 scopes (every S_folds P_main reference para pair; the test pairs of P_dft, P_g1, P_01), all 9 bend targets.
        Check (STOP, e1 and selftest): for the 6 EDA channels (spec 0-1: unchanged by Q / V) max |new - original dd|
        <= 1e-12 (= L-11 tolerance) and the DFT bend / V calls identical, in every scope. FU2 E_metrics.csv cross-check
        (max |difference|) recorded.
  E1-11 figL1 on P_01 (spec 14-1 '1단계는 P_01'); figL2 on P_01 too (common pairs, as figL1); figL3 on the own sets (the
        13-2 set). Style = rev 9 d1_figures.style() / palette (the validated reference categorical palette in fixed
        slots: AW 1, GNN 2, XGB 3 (= rev 9 arm C XGB), LINE 5, TF 6, ENS4 7, REP 8; null = muted gray; rungs in figL2:
        dft slot 1, g1 slot 2) + secondary encodings (line style / fill = arm, marker = rung); one y axis per panel.
  E1-12 E1 needs ADD3 committed and unchanged (md sha256 = json md_sha256), every unit fingerprint with that add3, the
        label file sha256 and the P_report rung table sha256, the label file = the P_report labels entry, the P_report
        builds current (inputs.p_report per rung), and ADD3 json line_functions.sha256 = line_source_sha256() of this
        file (the LINE functions that Phase C L-10 tested; ADD3 PR-13), else STOP.
  E1-13 e1-selftest: synthetic predictions (AW of dft·SH exact, noise elsewhere), synthetic dH (SH base = line_beta x dH
        by construction), a validation table with planned REP choices and two exact ties; P_g1 = the FU2 pairs, P_dft =
        P_g1 minus every 50th pair plus 20 pairs outside it. Checks: REP rule and ties, read guard, pair-list gate catches
        a missing pair, ENS4 mean, null para MAE = mean |y|, CI order, goal passes for the exact model, 13-3 'dft 통과',
        compare sign, LINE check, unchanged EDA channels keep their DFT bend calls, FU2 E_metrics reproduced.

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 incl. array elements,
LADDER <= 4 jobs while D1-G2 runs) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`)
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_pair; L=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs
  selftest (now; no Phase M output needed):
  sbatch -p cpu1,cpu2 -J lad_e1t --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
         --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/evaluate.py e1-selftest
  E1 (after `train_atoms.py collect --stage 1 --with-xgb` passed; with its job id as dependency if still queued):
  sbatch -p cpu1,cpu2 -J lad_e1 --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
         --export=ALL,PYENV=espley [--dependency=afterok:<collect job> --kill-on-invalid-dep=yes] \
         jobs/py.slurm ../d1_ladder/evaluate.py e1 --stage 1
  (--output must be given: py.slurm's default log folder is the read-only d1_pair scratch.) Rerun = the same command
  (every output is recomputed deterministically and written atomically).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair trees (spec 15)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

try:
    import pandas as pd  # noqa: E402
except ImportError:                   # line_beta / line_predict need numpy only (tests/run_tests.py L-10)
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lad_common as L  # noqa: E402

TARGETS = list(L.DP.TARGETS)                             # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
NO_HF = ("c_ghost",)


def line_beta(dH, y):
    """beta [len(TARGETS)]: zero-intercept least squares per target column on the given pairs (rows); c_ghost and any
    column with sum(dH^2) == 0 give 0. dH, y: arrays [n_pairs, len(TARGETS)] (NaN rows must be removed by the caller)."""
    dH, y = np.asarray(dH, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if dH.shape != y.shape or dH.ndim != 2 or dH.shape[1] != len(TARGETS):
        raise ValueError(f"line_beta: dH {dH.shape} / y {y.shape}, need [n, {len(TARGETS)}]")
    beta = np.zeros(len(TARGETS), dtype=np.float64)
    for j, t in enumerate(TARGETS):
        if t in NO_HF:
            continue
        den = float((dH[:, j] ** 2).sum())
        beta[j] = float((dH[:, j] * y[:, j]).sum() / den) if den > 0 else 0.0
    return beta


def line_predict(beta, dH):
    """LINE prediction beta * dH [n_pairs, len(TARGETS)]."""
    return np.asarray(dH, dtype=np.float64) * np.asarray(beta, dtype=np.float64)[None, :]


LINE_FUNCS = ("line_beta", "line_predict")


def line_source_sha256(path=None):
    """sha256 of the source of the module-level line_beta and line_predict of `path` (default: this file):
    ast.get_source_segment of each def, joined by one newline, in LINE_FUNCS order (= prereg.py line_source, ADD3
    json line_functions, PR-13); None if one is missing."""
    import ast
    import hashlib
    src = Path(path or __file__).read_text(encoding="utf-8")
    segs = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in LINE_FUNCS}
    if set(segs) != set(LINE_FUNCS):
        return None
    return hashlib.sha256("\n".join(segs[k] for k in LINE_FUNCS).encode()).hexdigest()


# ================================================================ Phase E1 (spec 11, 13): constants
STAGE_RUNGS = {1: ("dft", "g1")}                         # E1 = stage 1 (spec 10, 11); E2 (g2) = spec 12, later
ARMS = ("S", "SH")
MODELS4 = ("AW", "GNN", "TF", "XGB")                     # spec 4-3; also the REP tie order (spec 13-1)
MODEL_ORDER = {m: i for i, m in enumerate(MODELS4)}
MODELS_E1 = MODELS4 + ("REP", "ENS4", "LINE", "null")    # spec 11 model axis
ARM_FREE = ("LINE", "null")                              # spec 4-5 reference rows (no training): under both arms (E1-4)
CMP_MODELS = MODELS4 + ("REP", "ENS4")
NF = 5
DD = [f"dd_{t}" for t in TARGETS]
H_TARGETS = [t for t in TARGETS if t not in NO_HF]       # the 9 H values (spec 4-1)
PRIMARY9 = list(L.DP.PRIMARY_TARGETS)                    # 8 channels + barrier
GOAL_TARGETS = ("oi", "cpcm")                            # spec 13-2
GOAL = dict(mae=0.13, sens=0.7, fpr=0.1)                 # spec 13-2 (= d1pair_common STAGE2_*, asserted)
STAGE1_R2 = 0.5                                          # rev 9 stage 1 (descriptive, spec 13-2 last line)
STAGE0_NOTE = ("not computable: rev 9 stage 0 = B < null and a Holm test against arm A (d1_eval.stages); LADDER has no "
               "arm A (FU2 E-4 precedent); pre-registered in ADD3 5-2 (PR-14); substitute = beats_null_point")
EDA6 = ("elst", "pauli", "oi", "disp", "cpcm", "cds")     # spec 0-1: the EDA channels Q / V do not change
EDA_TOL = 1e-12                                          # = tests/run_tests.py TOL['L11']
CHANGED = ("d1", "d2", "barrier")                        # the dd Q / V change (spec 0-1); label effect (spec 11)
N_BOOT, SEED = 2000, 20261019                            # spec 11 (= d1pair_common N_BOOT / SEED, asserted)
BASE_TOL = 1e-9                                          # E1-5
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                             # = train_atoms.PRED_COLS
BOOL_COLS = ("in_P_main", "in_P_sens", "in_A_main", "in_T_main", "d0_overlap", "hidden_V_fam", "hidden_bend_fam")
MET_KEY = ["rung", "arm", "model", "pair_set", "target", "set", "pos"]
MET_KEY_NM = ["rung", "arm", "pair_set", "target", "set", "pos"]
UNITS = L.LAD / "units"
PRED = L.RES / "L_predictions.parquet"
L_REPORT = L.RES / "L_report.json"
L_VAL = L.RES / "L_val.csv"
ADD3_MD = L.RES / "PREREG_D1PAIR_ADD3.md"
ADD3_JSON = L.RES / "prereg_d1pair_add3.json"
DEFAULT_LABELS = L.LAD / "F_pairs_cr2.csv"               # spec 4-1 truth (V-4)
ORIG_LABELS = L.DP.RES / "F_pairs.csv"
S_FOLDS = L.DP.RES / "S_folds.csv"
V_REPORT = L.RES / "V_report.json"
SELFTEST = L.LAD / "e1_selftest"
D1PAIR_TREE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair")
FU2_EVAL_HERE = L.P_DIR / "fu2" / "evaluate.py"          # after the d1-pair merge (spec 11)
FU2_EVAL_D1PAIR = D1PAIR_TREE / "analysis" / "d1_pair" / "fu2" / "evaluate.py"
FU2_MODULE = "_lad_fu2_evaluate"
REUSED = ("analysis/d1_pair/d1_eval.py", "analysis/d1_pair/hammett_eval.py", "analysis/d1_pair/d1pair_common.py",
          "analysis/d1_pair/fu2/fu2_common.py", "analysis/d1_pair/d1_figures.py",
          "analysis/espley_xtb_repro/rev5_common.py", "analysis/espley_xtb_repro/rev6_common.py")
COLOR_SLOT = {"AW": 0, "GNN": 1, "XGB": 2, "LINE": 4, "TF": 5, "ENS4": 6, "REP": 7}
RUNG_SLOT = {"dft": 0, "g1": 1, "g2": 2}
RUNG_MARK = {"dft": "o", "g1": "s", "g2": "^"}
DECISIONS = {
    "E1-1": "env_espley via analysis/d1_pair/jobs/py.slurm PYENV=espley (numpy 1.26.0 / pandas 2.1.1 / pyarrow 17.0.0 "
            "as FU2 E-T, scipy, matplotlib 3.8.4)",
    "E1-2": "FU2 evaluate.py imported by path as _lad_fu2_evaluate; e1: only this worktree's merged copy with evaluate.py, "
            "test_eval.py and E_tests.json committed (else exit 2); selftest may use d1-pair's; reused modules from this "
            "worktree; E_tests.json gate_pass + evaluate_sha256 + test_sha256 + code pins, else STOP",
    "E1-3": "REP from unit json val_kcal_mae before any prediction file is read (read_parquet blocked); cross-checked "
            "with L_report.json / L_val.csv; L-9 one validation list per rung x fold; MA-6n asymmetry recorded",
    "E1-4": "LINE / null arm-independent, listed under both arms (arm_independent), computed once",
    "E1-5": "LINE beta = line_beta on fold_lists train (outer-train T_main of P_<rung>), dH from pairs_frame (h = cr); "
            "= SH unit base / beta (1e-9) and S base = beta = 0, else STOP",
    "E1-6": "pair_set own / P_01 x FU2 metrics_tab sets x positions; para CI on P_main x p only",
    "E1-7": "JSON NaN -> null",
    "E1-8": "13-2: bend rates 'all' population; NaN sens / FPR = not met; rev 9 stage 0 not computable (no arm A; "
            "pre-registered in ADD3 5-2, column stage0_rev9); stage 1 and beats-null descriptive",
    "E1-9": "13-4 sign FU2 D5: b = first named (SH, dft, REP), d = MAE(x) - MAE(b) > 0 = b better; P_main and para",
    "E1-10": "label effect on FU2 pairs = LADDER g1 pairs (STOP otherwise); FU2 REP only with a matching E_provenance; "
             "FU2 / rev 9 also scored on new labels; DFT calls on 4 scopes, all 9 bend targets; STOP if an EDA channel "
             "dd moves > 1e-12 or its DFT calls change",
    "E1-11": "figL1 / figL2 on P_01, figL3 on own sets; d1_figures style and palette, fixed slots, secondary encodings",
    "E1-12": "ADD3 committed + unit fingerprints (add3, labels, inputs_table) + P_report labels / builds current + ADD3 "
             "line_functions.sha256 = this file's LINE source",
    "E1-13": "e1-selftest on synthetic predictions over the real pair universe -> $LAD/e1_selftest",
}


# ================================================================ small helpers
def _not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _need_pandas():
    if pd is None:
        L.stop("pandas is not importable here; E1 needs pandas + pyarrow + scipy + matplotlib (env_espley, E1-1)")


def _log(t0, msg):
    print(f"[e1 {time.time() - t0:7.0f}s] {msg}", flush=True)


def _clean(o):
    """JSON-able copy: numpy -> python, NaN / inf -> None (E1-7), tuples -> lists, Path -> str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _write_json(path, obj):
    return L.write_json(path, _clean(obj))


def _write_csv(path, df):
    return L.write_atomic(path, lambda t: df.to_csv(t, index=False))


def _fin(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _git_state(p):
    """None if p is tracked by git and unchanged since HEAD (in its own worktree), else the problem."""
    p = Path(p)

    def g(*a):
        return subprocess.run(["git", "-C", str(p.parent), *a], capture_output=True, text=True)
    if g("ls-files", "--error-unmatch", str(p)).returncode != 0:
        return "not committed"
    if g("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
        return "modified since its commit"
    return None


def _repo_rel(p):
    s = str(p)
    i = s.find("/analysis/")
    return s[i + 1:] if i >= 0 else None


def _env():
    out = dict(prefix=sys.prefix, python=platform.python_version(), host=platform.node(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), numpy=np.__version__,
               pandas=getattr(pd, "__version__", None))
    for name in ("pyarrow", "scipy", "matplotlib"):
        m = sys.modules.get(name)
        out[name] = getattr(m, "__version__", None) if m is not None else None
    return out


def _expect_stop(fn):
    """True if fn() ends in a STOP (SystemExit 3)."""
    try:
        fn()
    except SystemExit as e:
        return e.code == 3
    return False


# ================================================================ FU2 Phase E code (E1-2)
_FU2: dict = {}


def fu2_eval(allow_unmerged=False):
    """FU2 Phase E fu2/evaluate.py (public API: compare, bends_tab, metrics_tab, annotate, eval_pairs, set_mask,
    pair_sets_gate, load_all) imported by file path under a private name -> (module, info). allow_unmerged = False
    (e1): only this worktree's merged and committed copy (spec 11, E1-2); True (e1-selftest): the d1-pair copy too."""
    if "mod" in _FU2:
        return _FU2["mod"], _FU2["info"]
    if allow_unmerged:
        src = FU2_EVAL_HERE if FU2_EVAL_HERE.is_file() else FU2_EVAL_D1PAIR
        if not src.is_file():
            _not_ready(f"FU2 Phase E code not found ({FU2_EVAL_HERE} after the d1-pair merge, or {FU2_EVAL_D1PAIR})")
    else:
        src = FU2_EVAL_HERE
    et = src.parent.parent / "results" / "fu2" / "E_tests.json"
    tf = src.parent / "tests" / "test_eval.py"
    states = {str(p): (_git_state(p) if p.is_file() else "missing") for p in (src, tf, et)}
    if not allow_unmerged:
        unmerged = [f"{p} {s}" for p, s in states.items() if s]
        if unmerged:
            _not_ready("E1 runs on the FU2 Phase E code merged into this worktree and committed (spec 11 'E1은 FU2 "
                       "Phase E가 commit된 뒤 merge해서 한다'): " + "; ".join(unmerged))
    fdir = str(L.P_DIR / "fu2")
    if fdir not in sys.path:
        sys.path.insert(0, fdir)
    import fu2_common as F2  # noqa: E402      (this worktree's copies first: the FU2 file binds to them)
    import d1_eval as E9  # noqa: E402
    import hammett_eval as HE  # noqa: E402
    root = L.REPO.resolve()
    for m in (F2, E9, HE, L.DP):
        mp = Path(m.__file__).resolve()
        if root not in mp.parents:
            L.stop(f"module {m.__name__} was imported from {mp}, not from this worktree ({root})")
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(FU2_MODULE, str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[FU2_MODULE] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    if not (mod.F is F2 and mod.E9 is E9 and mod.HE is HE and mod.C is L.DP):
        L.stop("FU2 evaluate.py is bound to other copies of fu2_common / d1_eval / hammett_eval / d1pair_common")
    sha = L.sha256(src)
    info = dict(source=str(src), copy=("this worktree (d1-pair merged)" if src == FU2_EVAL_HERE
                                       else "d1-pair worktree (before the merge)"),
                sha256=sha, git_state=states[str(src)] or "committed, unchanged", e_tests=str(et),
                git_states={p: (s or "committed, unchanged") for p, s in states.items()}, merged_required=not allow_unmerged)
    if not et.is_file():
        L.stop(f"{et} missing: the FU2 Phase E code to reuse is not certified by E-T")
    t = json.loads(et.read_text())
    tsha = L.sha256(tf) if tf.is_file() else None
    info.update(e_tests_sha256=L.sha256(et), e_tests_gate_pass=bool(t.get("gate_pass")),
                e_tests_evaluate_sha256=t.get("evaluate_sha256"), e_tests_prefix=t.get("prefix"),
                test_eval=str(tf), test_eval_sha256=tsha, e_tests_test_sha256=t.get("test_sha256"))
    if not t.get("gate_pass") or t.get("evaluate_sha256") != sha:
        L.stop(f"{src} (sha256 {sha[:12]}) is not the evaluate.py that E-T passed (E_tests.json evaluate_sha256 "
               f"{str(t.get('evaluate_sha256'))[:12]}, gate_pass {t.get('gate_pass')})")
    if tsha is None or t.get("test_sha256") != tsha:
        L.stop(f"{tf} (sha256 {str(tsha)[:12]}) is not the test_eval.py that wrote {et} (test_sha256 "
               f"{str(t.get('test_sha256'))[:12]}; FU2 require_review rule)")
    pins = {}
    for p, s in (t.get("code_sha256") or {}).items():
        rel = _repo_rel(p)
        if rel:
            pins[rel] = s
    reused, bad = {}, []
    for rel in REUSED:
        now = L.sha256(L.REPO / rel)
        reused[rel] = dict(sha256=now, e_t_pin=pins.get(rel), equal=now == pins.get(rel))
        if now != pins.get(rel):
            bad.append(rel)
    info["reused_code"] = reused
    if bad:
        L.stop(f"reused scoring code differs from the FU2 E-T pins ({et} code_sha256): {bad}")
    C = mod.C
    consts = (C.N_BOOT, C.SEED, C.STAGE2_MAE, C.STAGE2_SENS, C.STAGE2_FPR, tuple(C.STAGE2_TARGETS), C.STAGE1_R2)
    if consts != (N_BOOT, SEED, GOAL["mae"], GOAL["sens"], GOAL["fpr"], GOAL_TARGETS, STAGE1_R2):
        L.stop(f"d1pair_common bootstrap / stage constants {consts} differ from spec 11 / 13-2")
    _FU2.update(mod=mod, info=info)
    return mod, info


# ================================================================ inputs and gates
def unit_name(r, a, m, k):
    return f"L__{r}__{a}__{m}__f{k}"


def label_info(path):
    """the label file (spec 4-1 truth = $LAD/F_pairs_cr2.csv: V_report all_pass and outputs sha256)."""
    path = Path(os.path.abspath(path))
    if not path.is_file():
        _not_ready(f"label file {path} missing")
    sha = L.sha256(path)
    prov = "not the spec 4-1 truth file: provenance not checked"
    if path == Path(os.path.abspath(DEFAULT_LABELS)):
        rep = json.loads(V_REPORT.read_text()) if V_REPORT.is_file() else {}
        rec = (rep.get("outputs") or {}).get(str(L.LAD / path.name))
        if not rep.get("all_pass") or rec != sha:
            L.stop(f"{path}: results/V_report.json not all_pass or its outputs sha256 differs "
                   f"({str(rec)[:12]} vs {sha[:12]})")
        prov = "V_report.json all_pass, outputs sha256 equal"
    elif path == Path(os.path.abspath(ORIG_LABELS)):
        prov = "original D1PAIR F_pairs.csv"
    return dict(path=str(path), sha256=sha, provenance=prov)


def add3_check():
    """E1-12: the pre-registration ADD3 (spec 9) committed and unchanged, md sha256 = json md_sha256."""
    bad = []
    for p in (ADD3_MD, ADD3_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        else:
            st = _git_state(p)
            if st:
                bad.append(f"{p.name} {st}")
    if bad:
        L.stop("ADD3 (spec 9): " + "; ".join(bad))
    rec = json.loads(ADD3_JSON.read_text())
    md = L.sha256(ADD3_MD)
    if rec.get("md_sha256") != md:
        L.stop(f"{ADD3_MD.name} sha256 != {ADD3_JSON.name} md_sha256")
    pin = (rec.get("line_functions") or {}).get("sha256")
    mine = line_source_sha256()
    if not pin or pin != mine:
        L.stop(f"LINE functions {list(LINE_FUNCS)} of {Path(__file__).resolve()} (source sha256 {str(mine)[:12]}) != "
               f"{ADD3_JSON.name} line_functions.sha256 {str(pin)[:12]} (ADD3 PR-13: the LINE that Phase C L-10 tested)")
    return dict(md=str(ADD3_MD), md_sha256=md, json=str(ADD3_JSON), json_sha256=L.sha256(ADD3_JSON),
                line_functions=dict(sha256=mine, add3=pin, equal=True))


def val_table(rungs, labels_sha, add3_sha, table_sha):
    """E1-3: per-target validation kcal MAE of every unit json (spec 10 / 13-1) + unit records; each json is checked
    against the current files (labels, ADD3, rung input table) and L-9 (one validation-pair list per rung x fold)."""
    rows, units, missing, bad = [], {}, [], []
    for r in rungs:
        for a in ARMS:
            for m in MODELS4:
                for k in range(NF):
                    x = unit_name(r, a, m, k)
                    p = UNITS / f"{x}.json"
                    if not p.is_file():
                        missing.append(x)
                        continue
                    rec = json.loads(p.read_text())
                    fp = rec.get("fingerprint") or {}
                    v = rec.get("val_kcal_mae")
                    why = []
                    if rec.get("smoke"):
                        why.append("smoke unit")
                    if not isinstance(v, dict) or set(v) != set(TARGETS) or not all(_fin(v[t]) for t in TARGETS):
                        why.append("val_kcal_mae missing / not per target / not finite")
                    for key, want in (("labels", labels_sha), ("add3", add3_sha), ("inputs_table", table_sha[r]),
                                      ("rung", r), ("arm", a)):
                        if fp.get(key) != want:
                            why.append(f"fingerprint {key} {str(fp.get(key))[:16]} != {str(want)[:16]}")
                    if why:
                        bad.append(f"{x}: {why}")
                        continue
                    units[x] = dict(key=(r, a, m, k), val={t: float(v[t]) for t in TARGETS},
                                    json_sha256=L.sha256(p), parquet_sha256=rec.get("parquet_sha256"),
                                    n_test=rec.get("n_test"), n_val=rec.get("n_val"),
                                    val_pairs_sha256=rec.get("val_pairs_sha256"))
                    rows += [dict(rung=r, arm=a, model=m, fold=k, target=t, val_kcal_mae=float(v[t])) for t in TARGETS]
    if missing:
        _not_ready(f"{len(missing)} unit jsons missing under {UNITS}: {missing[:6]}")
    if bad:
        L.stop("unit jsons do not match the current inputs (spec 10 / 16 M):\n  " + "\n  ".join(bad[:10]))
    l9 = [f"{r} f{k}" for r in rungs for k in range(NF)
          if len({units[unit_name(r, a, m, k)]["val_pairs_sha256"] for a in ARMS for m in MODELS4}) != 1]
    if l9:
        L.stop(f"validation-pair lists differ between the units of {l9} (L-9: REP would compare different pairs)")
    return pd.DataFrame(rows), units


def pick_rep(V):
    """spec 13-1: per rung x arm x target the model with the lowest mean over the 5 outer folds of val_kcal_mae (float64,
    fold order); exact ties -> AW, GNN, TF, XGB. Validation records only (no test score)."""
    rep, means = {}, {}
    for (r, a, t), g in V.groupby(["rung", "arm", "target"], sort=False):
        mm = {}
        for m, h in g.groupby("model", sort=False):
            h = h.sort_values("fold")
            v = h.val_kcal_mae.to_numpy(float)
            if h.fold.astype(int).tolist() != list(range(NF)) or not np.isfinite(v).all():
                L.stop(f"REP: {r} {a} {m} {t}: validation MAE not finite or folds {h.fold.tolist()}")
            mm[m] = float(np.mean(v))
        if set(mm) != set(MODELS4):
            L.stop(f"REP: {r} {a} {t}: models {sorted(mm)} != {list(MODELS4)}")
        rep[(r, a, t)] = sorted(mm, key=lambda q: (mm[q], MODEL_ORDER[q]))[0]
        means[(r, a, t)] = mm
    return rep, means


def rep_json(rep, means):
    out = {}
    for (r, a, t), m in rep.items():
        out.setdefault(r, {}).setdefault(a, {})[t] = dict(model=m, mean_val_kcal_mae=means[(r, a, t)])
    return out


@contextlib.contextmanager
def no_test_scores(blocked):
    """E1-3: pandas.read_parquet refuses the test-prediction files (L_predictions.parquet, $LAD/units/*) while REP is
    chosen (FU2 D7 pattern); a refused read is a STOP before anything is read."""
    orig = pd.read_parquet
    roots = [Path(os.path.abspath(p)) for p in blocked]

    def guarded(path, *a, **kw):
        q = Path(os.path.abspath(str(path)))
        if any(q == b or b in q.parents for b in roots):
            L.stop(f"test-score file read while the representatives are chosen (spec 13-1): {q}")
        return orig(path, *a, **kw)
    pd.read_parquet = guarded
    try:
        yield
    finally:
        pd.read_parquet = orig


def check_l_report(units, rungs, labels):
    """L_report.json / L_val.csv of `train_atoms.py collect --with-xgb` agree with the unit jsons and the predictions
    file (read before the predictions; no test score is read here)."""
    for p in (L_REPORT, L_VAL, PRED):
        if not p.is_file():
            _not_ready(f"{p} missing (train_atoms.py collect --stage 1 --with-xgb)")
    rep = json.loads(L_REPORT.read_text())
    bad = []
    if rep.get("labels") != str(labels):
        bad.append(f"labels {rep.get('labels')} != {labels}")
    if "XGB" not in (rep.get("models") or []):
        bad.append("collected without --with-xgb")
    lack = sorted(set(rungs) - set(rep.get("rungs") or []))
    if lack:
        bad.append(f"rungs {rep.get('rungs')} lack {lack}")
    if rep.get("predictions_sha256") != L.sha256(PRED):
        bad.append("L_predictions.parquet changed after L_report.json")
    if rep.get("val_csv_sha256") != L.sha256(L_VAL):
        bad.append("L_val.csv changed after L_report.json")
    ru = rep.get("units") or {}
    for x, u in units.items():
        got = (ru.get(x) or {}).get("val_kcal_mae") or {}
        if any(got.get(t) != u["val"][t] for t in TARGETS):
            bad.append(f"{x}: val_kcal_mae differs from L_report.json")
    Lv = pd.read_csv(L_VAL)
    Lv = Lv[Lv.rung.isin(list(rungs))]
    vd = {(r, a, m, int(k), t): float(v) for r, a, m, k, t, v in
          zip(Lv.rung, Lv.arm, Lv.model, Lv.fold, Lv.target, Lv.val_kcal_mae)}
    for x, u in units.items():
        r, a, m, k = u["key"]
        for t in TARGETS:
            w = vd.get((r, a, m, k, t))
            if w is None or abs(w - u["val"][t]) > 1e-12 * max(1.0, abs(u["val"][t])):
                bad.append(f"{x} {t}: L_val.csv {w} != unit json {u['val'][t]}")
                break
    if bad:
        L.stop("L_report.json / L_val.csv / L_predictions.parquet disagree with the unit jsons:\n  " + "\n  ".join(bad[:8]))
    return dict(l_report_sha256=L.sha256(L_REPORT), l_val_sha256=L.sha256(L_VAL), predictions_sha256=L.sha256(PRED),
                stage=rep.get("stage"), n_units=rep.get("n_units"), models=rep.get("models"))


def prep_S(S):
    """S_folds.csv indexed by pair_id; boolean columns as bool; ref_kind / pos NaN -> '' (= train_atoms load_labels)."""
    S = S.copy()
    S["pair_id"] = S.pair_id.astype(str)
    if S.pair_id.duplicated().any():
        L.stop("S_folds.csv: duplicated pair_id")
    for c in BOOL_COLS:
        if S[c].dtype != bool:
            v = S[c].astype(str).map({"True": True, "False": False})
            if v.isna().any():
                L.stop(f"S_folds.csv column {c} is not boolean")
            S[c] = v.astype(bool)
    S["ref_kind"] = S.ref_kind.fillna("").astype(str)
    S["pos"] = S.pos.fillna("").astype(str)
    return S.set_index("pair_id")


def read_lab(path):
    """pair_id + dd_<10 targets> with the pandas default float parser (= inputs.read_labels / train_atoms.load_labels)."""
    Y = pd.read_csv(path, usecols=["pair_id"] + DD)
    Y["pair_id"] = Y.pair_id.astype(str)
    if Y.pair_id.duplicated().any():
        L.stop(f"{path}: duplicated pair_id")
    return Y.set_index("pair_id")


def ref_set(tests_r, rung):
    """{(fold, pair_id)} of a rung's test lists; a pair in two test folds = STOP."""
    ref, seen = set(), {}
    for k, ids in tests_r.items():
        for p in ids:
            if p in seen:
                L.stop(f"{rung}: pair {p} is a test pair of folds {seen[p]} and {k}")
            seen[p] = k
            ref.add((int(k), str(p)))
    return ref


def real_inputs(IN, rep_p, rungs, labels):
    """S_folds, labels (new + original), pair sets, fold lists and dH (h = cr) from the Phase P files (inputs.py API)."""
    if L.sha256(IN.S_FOLDS) != (rep_p.get("inputs") or {}).get(str(IN.S_FOLDS)):
        L.stop(f"{IN.S_FOLDS} sha256 differs from results/P_report.json")
    S = prep_S(pd.read_csv(IN.S_FOLDS, dtype={"pair_id": str}))
    Y, Yold = read_lab(labels), read_lab(ORIG_LABELS)
    for nm, T in (("labels", Y), ("original labels", Yold)):
        if set(T.index) != set(S.index):
            L.stop(f"{nm}: pair ids differ from S_folds.csv")
    lab = (rep_p.get("labels") or {}).get(str(labels)) or {}
    if not lab.get("ids_ok") or lab.get("sha256") != L.sha256(labels):
        L.stop(f"label file {labels}: sha256 / ids differ from results/P_report.json labels (Phase P state changed)")
    sets = {n: set(IN.read_set(n, rep_p).pair_id.astype(str)) for n in ("P_dft", "P_g1", "P_01")}
    tests, trains, dH = {}, {}, {}
    for r in rungs:
        tests[r], trains[r] = {}, {}
        for k in range(NF):
            fl = IN.fold_lists(r, k, rep_p)
            tests[r][k] = [str(p) for p in fl["test"]]
            trains[r][k] = [str(p) for p in fl["train"]]
        PF = IN.pairs_frame(r, label_path=labels, h="cr", rep=rep_p, subset="usable")
        ids = PF.pair_id.astype(str).to_numpy()
        if set(ids.tolist()) != sets[f"P_{r}"]:
            L.stop(f"inputs.pairs_frame({r}) pairs != P_{r}")
        D = pd.DataFrame(index=pd.Index(ids, name="pair_id"))
        for t in TARGETS:
            D[t] = PF[f"dH_{t}"].to_numpy(float) if t in H_TARGETS else 0.0
        if not np.isfinite(D.to_numpy(float)).all():
            L.stop(f"non-finite dH in P_{r} (rung {r} input table)")
        dH[r] = D
    return dict(rungs=tuple(rungs), S=S, Y=Y, Yold=Yold, sets=sets, tests=tests, trains=trains, dH=dH)


def load_predictions(rungs):
    P = pd.read_parquet(PRED)
    miss = [c for c in PRED_COLS if c not in P.columns]
    if miss:
        L.stop(f"{PRED} lacks columns {miss}")
    P = P[P.rung.isin(list(rungs))][PRED_COLS].copy()
    P["pair_id"] = P.pair_id.astype(str)
    P["fold"] = P.fold.astype(int)
    return P.reset_index(drop=True)


# ================================================================ frames
def model_frames(P, rungs):
    """{(rung, arm, model): pair_id, fold, target, y, yhat, base, beta, source} of the unit predictions."""
    fr = {}
    keep = ["pair_id", "fold", "target", "y", "yhat", "base", "beta"]
    for (r, a, m), q in P.groupby(["rung", "arm", "model"], sort=False):
        if r in rungs:
            fr[(r, a, m)] = q[keep].assign(source=m).reset_index(drop=True)
    want = {(r, a, m) for r in rungs for a in ARMS for m in MODELS4}
    if set(fr) != want:
        L.stop(f"L_predictions.parquet rung x arm x model differ from spec 4 (stage rungs x S, SH x AW, GNN, TF, XGB): "
               f"{sorted(set(fr) ^ want)}")
    return fr


def check_pair_attrs(P, S):
    """L_predictions ref_kind / pos = S_folds.csv (the evaluation uses the S_folds columns through FU2 annotate)."""
    a = S.loc[P.pair_id.to_numpy(), ["ref_kind", "pos"]]
    n = int(((a.ref_kind.to_numpy() != P.ref_kind.fillna("").astype(str).to_numpy())
             | (a.pos.to_numpy() != P.pos.fillna("").astype(str).to_numpy())).sum())
    if n:
        L.stop(f"L_predictions.parquet: {n} rows whose ref_kind / pos differ from S_folds.csv")
    return n


def line_frames(ctx):
    """E1-5 (spec 4-5): LINE = line_beta on the fold's outer-train T_main pairs of P_<rung>, applied to its test pairs
    -> {rung: (frame, {fold: {target: beta}})}."""
    Y, out = ctx["Y"], {}
    for r in ctx["rungs"]:
        D, parts, betas = ctx["dH"][r], [], {}
        for k in range(NF):
            tr, te = ctx["trains"][r][k], ctx["tests"][r][k]
            miss = sorted((set(tr) | set(te)) - set(D.index))
            if miss:
                L.stop(f"LINE {r} fold {k}: {len(miss)} pairs without dH (not in P_{r}): {miss[:3]}")
            if not len(te):
                continue
            dtr, ytr = D.loc[tr, TARGETS].to_numpy(float), Y.loc[tr, DD].to_numpy(float)
            if not (np.isfinite(dtr).all() and np.isfinite(ytr).all()):
                L.stop(f"LINE {r} fold {k}: non-finite dH / y on the training pairs")
            beta = line_beta(dtr, ytr)
            pred = line_predict(beta, D.loc[te, TARGETS].to_numpy(float))
            yte = Y.loc[te, DD].to_numpy(float)
            betas[k] = dict(zip(TARGETS, beta.tolist()))
            for j, t in enumerate(TARGETS):
                parts.append(pd.DataFrame(dict(pair_id=list(te), fold=int(k), target=t, y=yte[:, j], yhat=pred[:, j],
                                               base=pred[:, j], beta=float(beta[j]), source="LINE")))
        out[r] = (pd.concat(parts, ignore_index=True), betas)
    return out


def derived_frames(fr, lines, rep, rungs):
    """+ ENS4 (equal-weight mean of the 4 test predictions), REP (spec 13-1 choice per target), LINE, null (= 0)."""
    for r in rungs:
        line = lines[r][0]
        null = line.assign(yhat=0.0, base=np.nan, beta=np.nan, source="0")
        for a in ARMS:
            q = pd.concat([fr[(r, a, m)] for m in MODELS4], ignore_index=True)
            e = q.groupby(["pair_id", "target"], sort=False).agg(
                fold=("fold", "first"), fold_min=("fold", "min"), fold_max=("fold", "max"), y=("y", "first"),
                y_min=("y", "min"), y_max=("y", "max"), yhat=("yhat", "mean"), n=("yhat", "size")).reset_index()
            if (e.n != len(MODELS4)).any() or (e.fold_min != e.fold_max).any() or (e.y_min != e.y_max).any():
                L.stop(f"ENS4 {r} {a}: a pair / target without exactly one prediction per model, one fold and one y")
            fr[(r, a, "ENS4")] = e[["pair_id", "fold", "target", "y", "yhat"]].assign(base=np.nan, beta=np.nan,
                                                                                     source="ENS4")
            parts = []
            for t in TARGETS:
                f = fr[(r, a, rep[(r, a, t)])]
                parts.append(f[f.target == t])
            fr[(r, a, "REP")] = pd.concat(parts, ignore_index=True)
            fr[(r, a, "LINE")] = line
            fr[(r, a, "null")] = null
    return fr


def gate_pairs(EV, fr, ref, Y, sets, rungs):
    """spec 16 E: every model / arm of a rung has exactly the rung's test pairs (fold, pair_id) per target, no
    duplicate, finite yhat, y = the label file (FU2 pair_sets_gate, 1e-9); on P_01 every rung has the same list."""
    out = {}
    for r in rungs:
        frames = {f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS_E1}
        out[r] = EV.pair_sets_gate(frames, ref[r], Y)
    if len(rungs) > 1:
        p01 = sets["P_01"]
        r01 = {r: {(k, p) for k, p in ref[r] if p in p01} for r in rungs}
        if len({frozenset(v) for v in r01.values()}) != 1:
            cnt = {r: len(v) for r, v in r01.items()}
            L.stop(f"spec 16 E: the P_01 test-pair lists differ between rungs ({cnt})")
        frames = {f"{r}|{a}|{m}|P_01": f[f.pair_id.isin(p01)] for (r, a, m), f in fr.items()}
        out["P_01"] = EV.pair_sets_gate(frames, r01[rungs[0]], Y)
    return out


def check_line(fr, lines, rungs):
    """E1-5: the SH units' baseline = LINE (same pairs, same beta), S rows without a baseline; STOP otherwise."""
    rec, bad = {}, []
    for r in rungs:
        Lr = lines[r][0].set_index(["pair_id", "target"])[["yhat", "beta"]].rename(
            columns={"yhat": "line_base", "beta": "line_beta"})
        for m in MODELS4:
            j = fr[(r, "SH", m)].set_index(["pair_id", "target"])[["base", "beta"]].join(Lr, how="left")
            lb = j.line_beta.to_numpy(float)
            db = np.abs(j.base.to_numpy(float) - j.line_base.to_numpy(float))
            dbeta = np.abs(j.beta.to_numpy(float) - lb) / np.maximum(1.0, np.abs(lb))
            mb = float(db.max()) if len(db) and np.isfinite(db).all() else float("inf")
            mbe = float(dbeta.max()) if len(dbeta) and np.isfinite(dbeta).all() else float("inf")
            s = fr[(r, "S", m)]
            nz = int(((s.base != 0) | (s.beta != 0)).sum())
            key = f"{r}|{m}"
            rec[key] = dict(sh_base_max_abs=mb, sh_beta_max_rel=mbe, s_rows_with_baseline=nz)
            if mb > BASE_TOL or mbe > BASE_TOL or nz:
                bad.append(f"{key}: {rec[key]}")
    if bad:
        L.stop("LINE (line_beta on the outer-train T_main pairs) differs from the SH units' baseline, or S rows carry a "
               "baseline (spec 4-2 / 4-5, L-2):\n  " + "\n  ".join(bad))
    return rec


def ps_name(r, ps):
    return f"P_{r}" if ps == "own" else ps


def annotate_all(EV, fr, S, p01, rungs):
    """{(rung, arm, model, pair_set): frame + S_folds columns (FU2 annotate)}; LINE / null once (arm S, E1-4)."""
    A = {}
    for (r, a, m), f in fr.items():
        if m in ARM_FREE and a != ARMS[0]:
            continue
        d = EV.annotate(f, S)
        A[(r, a, m, "own")] = d
        if len(rungs) > 1:
            A[(r, a, m, "P_01")] = d[d.pair_id.isin(p01)]
    return A


# ================================================================ metrics, bends, goal
def metrics_e1(EV, A):
    """E1_metrics: FU2 metrics_tab per rung x arm x model x pair set + the para MAE CI of spec 11 (E1-6)."""
    E9, parts = EV.E9, []
    for (r, a, m, ps), d in A.items():
        Mt = EV.metrics_tab(d, by=("source",))
        pm = (EV.set_mask(d, "P_main") & (d.pos == "p")).to_numpy()
        cis = {}
        for t, q in d[pm].groupby("target", sort=False):
            e = (q.yhat - q.y).to_numpy(float)
            cores = q.core.to_numpy()
            mae_b, lo, hi, _ = E9.boot_diff(e, np.zeros(len(e)), cores, n_boot=N_BOOT, seed=SEED)
            cis[t] = (int(len(np.unique(cores))), mae_b, lo, hi)
        is_p = ((Mt["set"] == "P_main") & (Mt.pos == "p")).to_numpy()
        nc, clo, chi = [], [], []
        for t, ok, mae in zip(Mt.target.to_numpy(), is_p, Mt.mae.to_numpy(float)):
            c = cis.get(t) if ok else None
            if c is not None and abs(c[1] - mae) > 1e-12:
                L.stop(f"para MAE of boot_diff {c[1]} != metrics_tab {mae} ({r} {a} {m} {ps} {t})")
            nc.append(c[0] if c else np.nan)
            clo.append(c[2] if c else np.nan)
            chi.append(c[3] if c else np.nan)
        Mt = Mt.assign(rung=r, arm=a, model=m, pair_set=ps, pair_set_name=ps_name(r, ps),
                       arm_independent=m in ARM_FREE, n_cores=nc, mae_ci_lo=clo, mae_ci_hi=chi)
        parts.append(Mt)
        if m in ARM_FREE:
            parts.append(Mt.assign(arm=ARMS[1]))
    Mt = pd.concat(parts, ignore_index=True)
    nl = Mt[Mt.model == "null"][MET_KEY_NM + ["mae"]].rename(columns={"mae": "mae_null"})
    Mt = Mt.merge(nl, on=MET_KEY_NM, how="left", validate="m:1")
    Mt["mae_rel_null"] = Mt.mae / Mt.mae_null
    cols = ["rung", "arm", "model", "source", "arm_independent", "pair_set", "pair_set_name", "target", "set", "pos", "n",
            "mae", "rmse", "mae_null", "mae_rel_null", "n_cores", "mae_ci_lo", "mae_ci_hi", "n_fam", "r2_fam_median",
            "r2_fam_q1", "r2_fam_q3", "rho_err_median"]
    return Mt[cols]


def met_index(Mt):
    keys = [tuple(x) for x in Mt[MET_KEY].itertuples(index=False, name=None)]
    recs = Mt.to_dict("records")
    MI = dict(zip(keys, recs))
    if len(MI) != len(recs):
        L.stop("E1_metrics: duplicated rung x arm x model x pair set x target x set x pos rows")
    return MI


def bends_e1(EV, A, S, rep):
    """E1_bends (FU2 bends_tab) + bend rates (d1_eval.bend_rates: all / hidden_V / hidden_bend families) per rung x arm
    x model x pair set; key 'rung|arm|model|pair_set'."""
    E9, parts, rates = EV.E9, [], {}
    hv = sorted(S.loc[S.hidden_V_fam, "fam"].astype(str).unique().tolist())
    hb = sorted(S.loc[S.hidden_bend_fam, "fam"].astype(str).unique().tolist())
    for (r, a, m, ps), d in A.items():
        Bd = EV.bends_tab(d, rung=r, arm=a, model=m, pair_set=ps, pair_set_name=ps_name(r, ps), split="T1")
        src = {t: (rep[(r, a, t)] if m == "REP" else ("0" if m == "null" else m)) for t in TARGETS}
        Bd.insert(3, "source", Bd.target.map(src))
        br = dict(all=E9.bend_rates(Bd, "T1"), hidden_V=E9.bend_rates(Bd, "T1", hv),
                  hidden_bend=E9.bend_rates(Bd, "T1", hb))
        keys = [(a, Bd)] + ([(ARMS[1], Bd.assign(arm=ARMS[1]))] if m in ARM_FREE else [])
        for aa, bb in keys:
            parts.append(bb)
            rates[f"{r}|{aa}|{m}|{ps}"] = br
    return pd.concat(parts, ignore_index=True), rates, dict(hidden_V=hv, hidden_bend=hb)


def goal_e1(MI, rates, rungs):
    """spec 13-2 (REP primary, ENS4 secondary, single models descriptive) + spec 13-3 for stage 1 (dft only)."""
    rows, verdict = [], {}
    for r in rungs:
        for a in ARMS:
            for m in ("REP", "ENS4") + MODELS4:
                role = {"REP": "primary", "ENS4": "secondary"}.get(m, "descriptive")
                per = {}
                for t in PRIMARY9:
                    pp = MI[(r, a, m, "own", t, "P_main", "p")]
                    pa = MI[(r, a, m, "own", t, "P_main", "all")]
                    br = rates[f"{r}|{a}|{m}|own"]["all"][t]
                    mae, hi, sens, fpr = pp["mae"], pp["mae_ci_hi"], br["sens"], br["fpr"]
                    c_mae = bool(_fin(mae) and mae <= GOAL["mae"])
                    c_sens = bool(_fin(sens) and sens >= GOAL["sens"])
                    c_fpr = bool(_fin(fpr) and fpr <= GOAL["fpr"])
                    ok = bool(c_mae and c_sens and c_fpr)
                    sure = bool(ok and _fin(hi) and hi <= GOAL["mae"])
                    per[t] = (ok, sure)
                    r2 = pa["r2_fam_median"]
                    rows.append(dict(
                        level="target", rung=r, arm=a, model=m, role=role, source=pp["source"], target=t,
                        judged=bool(t in GOAL_TARGETS and role != "descriptive"), pair_set_name=f"P_{r}",
                        n_para=pp["n"], para_mae=mae, para_mae_ci_lo=pp["mae_ci_lo"], para_mae_ci_hi=hi,
                        n_cores=pp["n_cores"], n_fam=br["n_fam"], n_bent_dft=br["n_bent_dft"], sens=sens,
                        sens_ci_lo=br["sens_ci"][0], sens_ci_hi=br["sens_ci"][1], n_straight_dft=br["n_straight_dft"],
                        fpr=fpr, fpr_ci_lo=br["fpr_ci"][0], fpr_ci_hi=br["fpr_ci"][1], crit_mae=c_mae,
                        crit_sens=c_sens, crit_fpr=c_fpr, passed=ok, surely_passed=sure, mae_all=pa["mae"],
                        mae_null_all=pa["mae_null"],
                        beats_null_point=bool(_fin(pa["mae"]) and _fin(pa["mae_null"]) and pa["mae"] < pa["mae_null"]),
                        r2_fam_median=r2, stage1_descriptive=(bool(r2 >= STAGE1_R2) if _fin(r2) else None),
                        stage0_rev9=STAGE0_NOTE, conclusion=None))
                ok2 = bool(all(per[t][0] for t in GOAL_TARGETS))
                sure2 = bool(all(per[t][1] for t in GOAL_TARGETS))
                rows.append(dict(level="rung_arm", rung=r, arm=a, model=m, role=role, target="+".join(GOAL_TARGETS),
                                 judged=role != "descriptive", pair_set_name=f"P_{r}", passed=ok2, surely_passed=sure2,
                                 conclusion=None))
                verdict[f"{r}|{a}|{m}"] = dict(role=role, passed=ok2, surely_passed=sure2)
    concl = {}
    if "dft" in rungs:
        ok = [a for a in ARMS if verdict[f"dft|{a}|REP"]["passed"]]
        concl = dict(rule="spec 13-3, stage 1: dft part only (dft·S or dft·SH passes with REP -> dft 통과)",
                     dft_passed=bool(ok), passed_arms=ok, conclusion_ko="dft 통과" if ok else "dft 실패",
                     conclusion_en="dft passes" if ok else "dft fails",
                     surely_passed_arms=[a for a in ARMS if verdict[f"dft|{a}|REP"]["surely_passed"]],
                     secondary_ens4_passed_arms=[a for a in ARMS if verdict[f"dft|{a}|ENS4"]["passed"]],
                     note="g1 is not part of the 13-3 table (descriptive); 13-3 a / b need g2 (E2, spec 12)")
        rows.append(dict(level="13-3", rung="dft", arm="+".join(ok) if ok else "-", model="REP", role="primary",
                         target="+".join(GOAL_TARGETS), judged=True, pair_set_name="P_dft", passed=bool(ok),
                         surely_passed=bool(concl["surely_passed_arms"]),
                         conclusion=concl["conclusion_ko"] + " / " + concl["conclusion_en"]))
    return pd.DataFrame(rows), verdict, concl


# ================================================================ compare (spec 13-4)
def compare_e1(EV, fr, S, ref, p01, rep, rungs):
    """E1_compare: FU2 compare (d1_eval.boot_diff, 2,000 core resamples, seed 20261019), d = MAE(x) - MAE(b) (E1-9)."""
    rows = []

    def src(r, a, m, t):
        return rep[(r, a, t)] if m == "REP" else ("0" if m == "null" else m)

    def evalsets(rf):
        return {"P_main": EV.eval_pairs(S, rf), "P_main_para": EV.eval_pairs(S, rf, para=True)}

    def add(comp, kb, kx, ps, psn, sets, arm_label=None):
        for evn, pairs in sets.items():
            R = EV.compare(fr[kx], fr[kb], pairs, targets=TARGETS, n_boot=N_BOOT, seed=SEED)
            for x in R.itertuples(index=False):
                rows.append(dict(comparison=comp, rung_b=kb[0], arm_b=arm_label or kb[1], model_b=kb[2],
                                 source_b=src(kb[0], kb[1], kb[2], x.target), rung_x=kx[0], arm_x=arm_label or kx[1],
                                 model_x=kx[2], source_x=src(kx[0], kx[1], kx[2], x.target), pair_set=ps,
                                 pair_set_name=psn, eval=evn, target=x.target, n_pairs=int(x.n_pairs),
                                 n_cores=int(x.n_cores), mae_x=x.mae_x, mae_b=x.mae_b, d=x.d, ci_lo=x.ci_lo,
                                 ci_hi=x.ci_hi, p=x.p, folds_b_lower=x.folds_b_lower, n_folds=x.n_folds))
    own = {r: evalsets(ref[r]) for r in rungs}
    for r in rungs:
        for m in CMP_MODELS:
            add("SH vs S (what HF adds)", (r, "SH", m), (r, "S", m), "own", f"P_{r}", own[r])
    if "dft" in rungs and "g1" in rungs:
        com = evalsets({(k, p) for k, p in ref["dft"] if p in p01})
        for a in ARMS:
            for m in CMP_MODELS:
                add("dft vs g1 (what the xTB geometry loses)", ("dft", a, m), ("g1", a, m), "P_01", "P_01", com)
        add("dft vs g1 (what the xTB geometry loses)", ("dft", "S", "LINE"), ("g1", "S", "LINE"), "P_01", "P_01", com,
            arm_label="-")
    for r in rungs:
        for a in ARMS:
            add("REP vs LINE (what the model adds on the calibration line)", (r, a, "REP"), (r, a, "LINE"), "own",
                f"P_{r}", own[r])
    out = pd.DataFrame(rows)
    out.insert(1, "sign", "d = MAE(x) - MAE(b); d > 0: b better (FU2 D5: b = the first named)")
    return out


# ================================================================ label effect (spec 11)
def _relabel(d, Y, targets):
    """a copy of a prediction frame with y of `targets` replaced by the dd of label table Y."""
    d = d.copy()
    for t in targets:
        m = (d.target == t).to_numpy()
        d.loc[m, "y"] = Y.loc[d.loc[m, "pair_id"].to_numpy(), f"dd_{t}"].to_numpy(float)
    return d


def _summ(EV, d, targets):
    """per target: P_main MAE (all, para), R2_fam / rho error (FU2 metrics_tab), bend calls and rates (bends_tab)."""
    d = d[d.target.isin(list(targets))]
    Mt = EV.metrics_tab(d, by=("source",))
    Bd = EV.bends_tab(d, split="T1")
    br = EV.E9.bend_rates(Bd, "T1")
    out = {}
    for t in targets:
        qa = Mt[(Mt.target == t) & (Mt["set"] == "P_main") & (Mt.pos == "all")]
        qp = Mt[(Mt.target == t) & (Mt["set"] == "P_main") & (Mt.pos == "p")]
        if len(qa) != 1 or len(qp) != 1:
            L.stop(f"label effect: {len(qa)} / {len(qp)} metrics rows for {t} (one model frame expected)")
        a, p = qa.iloc[0], qp.iloc[0]
        q = Bd[Bd.target == t]
        b = br[t]
        out[t] = dict(source=a["source"], n=int(a["n"]), mae=float(a["mae"]), rmse=float(a["rmse"]), n_para=int(p["n"]),
                      mae_para=float(p["mae"]), n_fam=a["n_fam"], r2_fam_median=a["r2_fam_median"],
                      rho_err_median=a["rho_err_median"], n_fam_bends=int(len(q)),
                      n_bent_dft=int(q.bent_dft.astype(bool).sum()), n_bent_pred=int(q.bent_pred.astype(bool).sum()),
                      n_bent_both=int((q.bent_dft.astype(bool) & q.bent_pred.astype(bool)).sum()), sens=b["sens"],
                      fpr=b["fpr"], n_V_dft=int(q.V_dft.astype(bool).sum()), n_V_pred=int(q.V_pred.astype(bool).sum()))
    return out


def _fu2_prov_ok(d, name, eval_sha):
    p, pv = d / name, d / "E_provenance.json"
    if not (p.is_file() and pv.is_file()):
        return False
    rec = json.loads(pv.read_text()).get(name) or {}
    return rec.get("sha256") == L.sha256(p) and rec.get("evaluate_sha256") == eval_sha


def fu2_outputs(EV, info):
    """FU2 E_representative.json and E_metrics.csv whose E_provenance.json record names the imported evaluate.py (the
    results/fu2 folder next to it first, then this worktree's, then d1-pair's) -> (rep | None, info, E_metrics path)."""
    cands = []
    for d in (Path(info["source"]).parent.parent / "results" / "fu2", L.P_DIR / "results" / "fu2",
              D1PAIR_TREE / "analysis" / "d1_pair" / "results" / "fu2"):
        if d not in cands:
            cands.append(d)
    rep, rinfo, mpath = None, dict(note="no FU2 E_representative.json with a matching E_provenance.json record: FU2 "
                                        "REP rows omitted"), None
    for d in cands:
        if _fu2_prov_ok(d, "E_representative.json", info["sha256"]):
            r = json.loads((d / "E_representative.json").read_text())
            if set(r) == set(TARGETS) and set(r.values()) <= set(EV.CONFIG_NAMES):
                rep = {t: r[t] for t in TARGETS}
                rinfo = dict(path=str(d / "E_representative.json"), sha256=L.sha256(d / "E_representative.json"),
                             provenance="E_provenance.json record = this file and the imported evaluate.py")
                break
    for d in cands:
        if _fu2_prov_ok(d, "E_metrics.csv", info["sha256"]):
            mpath = d / "E_metrics.csv"
            break
    return rep, rinfo, mpath


def fu2_crosscheck(EV, Lf, names, mpath):
    """FU2 metrics_tab on the FU2 frames (original labels, d1 / d2 / barrier) recomputed here vs FU2 E_metrics.csv."""
    if mpath is None:
        return dict(status="no FU2 E_metrics.csv with a matching E_provenance.json record: not compared")
    X = pd.read_csv(mpath)
    mine = pd.concat([EV.metrics_tab(Lf[n][Lf[n].target.isin(list(CHANGED))].assign(model=n), by=("model", "source"))
                      for n in names], ignore_index=True)
    J = mine.merge(X, on=["model", "source", "target", "set", "pos"], how="inner", suffixes=("", "_fu2"))
    out = dict(status="compared", file=str(mpath), sha256=L.sha256(mpath), n_rows_mine=int(len(mine)),
               n_rows_compared=int(len(J)))
    for c in ("n", "mae", "rmse", "r2_fam_median", "rho_err_median"):
        a, b = J[c].to_numpy(float), J[f"{c}_fu2"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        out[f"max_abs_diff_{c}"] = float(np.max(np.abs(a[both] - b[both]))) if both.any() else None
        out[f"n_nan_pattern_differs_{c}"] = int((np.isfinite(a) != np.isfinite(b)).sum())
    return out


def dft_calls(EV, S, Yold, Ynew, scopes):
    """DFT truth bend calls, original vs new labels (spec 11): bends_tab (hammett_eval.calls) with y = original dd and
    yhat = new dd on the scope's P_main reference para pairs (families >= 12 points incl. H)."""
    CH = list(EV.E9.CH)
    base = S[S.in_P_main & (S.ref_kind == "reference") & (S.pos == "p") & S.sigma.notna()]
    rows = []
    for name, ids in scopes.items():
        b = base if ids is None else base[base.index.isin(ids)]
        pid = b.index.to_numpy()
        long = pd.concat([pd.DataFrame(dict(pair_id=pid, target=t, y=Yold.loc[pid, f"dd_{t}"].to_numpy(float),
                                            yhat=Ynew.loc[pid, f"dd_{t}"].to_numpy(float))) for t in CH],
                         ignore_index=True)
        fin = np.isfinite(long.y.to_numpy(float)) & np.isfinite(long.yhat.to_numpy(float))
        n_bad = {t: int((~fin & (long.target == t).to_numpy()).sum()) for t in CH}
        long = long[fin].reset_index(drop=True)
        Bd = EV.bends_tab(EV.annotate(long, S), scope=name)
        for t in CH:
            q = Bd[Bd.target == t]
            bo, bn = q.bent_dft.astype(bool), q.bent_pred.astype(bool)
            vo, vn = q.V_dft.astype(bool), q.V_pred.astype(bool)
            dd = np.abs(Ynew.loc[pid, f"dd_{t}"].to_numpy(float) - Yold.loc[pid, f"dd_{t}"].to_numpy(float))
            dd = dd[np.isfinite(dd)]
            rows.append(dict(part="dft_calls", labels="original vs new", trained_on=None, family="DFT truth",
                             model="DFT", scope=name, target=t, n_para_pairs=int(len(pid)),
                             n_nonfinite_dropped=n_bad[t], n_fam_bends=int(len(q)),
                             n_bent_orig=int(bo.sum()), n_bent_new=int(bn.sum()), n_bent_both=int((bo & bn).sum()),
                             n_orig_only=int((bo & ~bn).sum()), n_new_only=int((~bo & bn).sum()),
                             n_V_orig=int(vo.sum()), n_V_new=int(vn.sum()), n_V_changed=int((vo != vn).sum()),
                             n_pairs_dd_changed=int((dd > 0).sum()),
                             max_abs_dd_change=float(dd.max()) if len(dd) else np.nan,
                             families_changed=";".join(sorted(q.fam[bo != bn].astype(str).tolist()))))
    return rows


def _family(EV, n):
    if n in EV.CONFIG_NAMES or n == "REP":
        return "FU2"
    return "rev 9 arm C" if n.startswith("C-") else "null"


def label_effect(EV, info, ctx, A, ref):
    """E1_label_effect (spec 11, E1-10): part 'model' (d1 / d2 / barrier per model and label set) + part 'dft_calls'."""
    rungs, S, Ynew, Yold = ctx["rungs"], ctx["S"], ctx["Y"], ctx["Yold"]
    rows, meta = [], {}
    if "g1" in rungs:
        for m in ("AW", "GNN", "TF", "XGB", "REP", "ENS4", "null"):
            for t, v in _summ(EV, A[("g1", "S", m, "own")], CHANGED).items():
                rows.append(dict(part="model", labels="new", trained_on="new", family="LADDER g1·S", model=m,
                                 target=t, **v))
        rep2, rinfo, mpath = fu2_outputs(EV, info)
        S2, Lf, ref2 = EV.load_all(rep2)
        meta.update(fu2_representatives=rinfo, fu2_rep=rep2)
        if set(ref2) != set(ref["g1"]):
            L.stop(f"spec 16 E: FU2 test pairs with atom data ({len(ref2)}) != LADDER g1 test pairs ({len(ref['g1'])}): "
                   "the label-effect comparison would use different pairs")
        names = [n for n in ("AW-A+G", "GNN-A+G", "TF-A+G", "REP", "C-XGB", "C-rep", "null") if n in Lf]
        for n in names:
            d = Lf[n]
            for lab, dd in (("original", d), ("new", _relabel(d, Ynew, CHANGED))):
                for t, v in _summ(EV, dd, CHANGED).items():
                    rows.append(dict(part="model", labels=lab, trained_on="original", family=_family(EV, n), model=n,
                                     target=t, **v))
        meta["fu2_models"] = names
        meta["fu2_metrics_crosscheck"] = fu2_crosscheck(EV, Lf, names, mpath)
    scopes = {"P_main (every S_folds P_main reference pair)": None}
    for r in rungs:
        scopes[f"P_{r} test pairs"] = {p for _, p in ref[r]}
    if "dft" in rungs and "g1" in rungs:
        scopes["P_01 test pairs"] = {p for _, p in ref["dft"] if p in ctx["sets"]["P_01"]}
    dc = dft_calls(EV, S, Yold, Ynew, scopes)
    # E1-10 check: the EDA channels are unchanged by Q / V (spec 0-1), so their DFT truth calls must not move
    bad = [f"{r['scope']} {r['target']}: max |dd change| {r['max_abs_dd_change']}, calls changed "
           f"{r['n_orig_only'] + r['n_new_only']} bend / {r['n_V_changed']} V"
           for r in dc if r["target"] in EDA6
           and ((_fin(r["max_abs_dd_change"]) and r["max_abs_dd_change"] > EDA_TOL)
                or r["n_orig_only"] + r["n_new_only"] + r["n_V_changed"] > 0)]
    if bad:
        L.stop("label effect: the new labels change an EDA channel (spec 0-1: only d1 / d2 / barrier change; L-11):\n  "
               + "\n  ".join(bad[:8]))
    rows += dc
    meta["dft_call_scopes"] = list(scopes)
    meta["eda_channels_unchanged"] = dict(targets=list(EDA6), tol=EDA_TOL, ok=True, n_rows_checked=sum(
        1 for r in dc if r["target"] in EDA6))
    return pd.DataFrame(rows), meta


# ================================================================ figures (spec 14-1, stage 1)
def _plot():
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        L.stop(f"matplotlib is not importable ({e}): run e1 in env_espley (E1-1)")
    import d1_figures as DF
    if L.REPO.resolve() not in Path(DF.__file__).resolve().parents:
        L.stop(f"d1_figures imported from {DF.__file__}, not from this worktree")
    return DF.style(), DF


def _color(DF, m):
    return DF.MUTED if m == "null" else DF.SLOT[COLOR_SLOT[m]]


def _save(fig, figdir, name, df):
    for ext in ("pdf", "png"):
        L.write_atomic(figdir / f"{name}.{ext}", lambda t, e=ext: fig.savefig(t, format=e, bbox_inches="tight"))
    _write_csv(figdir / f"{name}.csv", df)
    return [str(figdir / f"{name}.{e}") for e in ("pdf", "png", "csv")]


def figL1(plt, DF, C, MI, rungs, figdir):
    """oi / cpcm para MAE vs rung on P_01 (models: SH solid filled, S dashed open; LINE, null; goal 0.13; 95 % CI)."""
    from matplotlib.lines import Line2D
    series = [(m, a) for m in ("AW", "GNN", "TF", "XGB", "ENS4", "REP") for a in ("SH", "S")] + \
        [("LINE", "S"), ("null", "S")]
    x0 = np.arange(len(rungs), dtype=float)
    w = 0.6 / len(series)
    rows = []
    fig, axs = plt.subplots(1, len(GOAL_TARGETS), figsize=(C.FIG_W, C.FIG_W * 0.36), constrained_layout=True,
                            squeeze=False)
    for ax, t in zip(axs[0], GOAL_TARGETS):
        for i, (m, a) in enumerate(series):
            recs = [MI[(r, a, m, "P_01", t, "P_main", "p")] for r in rungs]
            v = np.array([q["mae"] for q in recs], dtype=float)
            lo = np.array([q["mae_ci_lo"] for q in recs], dtype=float)
            hi = np.array([q["mae_ci_hi"] for q in recs], dtype=float)
            x = x0 + (i - (len(series) - 1) / 2) * w
            col = _color(DF, m)
            free = m in ARM_FREE
            ls = ":" if free else ("-" if a == "SH" else "--")
            mk = {"LINE": "D", "null": "X"}.get(m, "o")
            face = col if (free or a == "SH") else "white"
            err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
            ax.errorbar(x, v, yerr=err, fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, zorder=2)
            ax.plot(x, v, color=col, ls=ls, lw=1.0, marker=mk, ms=3.5, mfc=face, mec=col, mew=0.8, zorder=3)
            for r, q in zip(rungs, recs):
                rows.append(dict(target=t, rung=r, model=m, arm="-" if free else a, source=q["source"], pair_set="P_01",
                                 n_para=q["n"], n_cores=q["n_cores"], para_mae=q["mae"], ci_lo=q["mae_ci_lo"],
                                 ci_hi=q["mae_ci_hi"], goal=GOAL["mae"]))
        ax.axhline(GOAL["mae"], color=DF.INK2, lw=0.8, ls="-.", zorder=1)
        ax.set_xticks(x0, list(rungs))
        ax.set_xlim(-0.6, len(rungs) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_title(DF.TLAB[t])
        ax.set_xlabel("rung")
        ax.set_ylabel("para ΔΔ MAE (kcal/mol)")
        ax.grid(axis="x", visible=False)
    hs = [Line2D([], [], color=_color(DF, m), marker="o", ls="-", lw=1.0, ms=3.5, label=m)
          for m in ("AW", "GNN", "TF", "XGB", "ENS4", "REP")]
    hs += [Line2D([], [], color=DF.INK2, ls="-", lw=1.0, marker="o", ms=3.5, label="SH (structure + HF): solid, filled"),
           Line2D([], [], color=DF.INK2, ls="--", lw=1.0, marker="o", ms=3.5, mfc="white",
                  label="S (structure only): dashed, open"),
           Line2D([], [], color=_color(DF, "LINE"), ls=":", lw=1.0, marker="D", ms=3.5, label="LINE (β·ΔH)"),
           Line2D([], [], color=_color(DF, "null"), ls=":", lw=1.0, marker="X", ms=3.5, label="null (ΔΔ = 0)"),
           Line2D([], [], color=DF.INK2, ls="-.", lw=0.8, label="goal 0.13 kcal/mol")]
    axs[0, -1].legend(handles=hs, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("P_01 test pairs, P_main para pairs; bars = 95 % CI (core bootstrap, 2,000)", fontsize=C.FONT_PT)
    out = _save(fig, figdir, "figL1_ladder", pd.DataFrame(rows))
    plt.close(fig)
    return out


def figL2(plt, DF, C, MI, rungs, figdir):
    """9 targets: para MAE / null para MAE of REP per rung x arm (P_01)."""
    combos = [(r, a) for r in rungs for a in ARMS]
    x = np.arange(len(PRIMARY9), dtype=float)
    w = 0.84 / len(combos)
    rows = []
    fig, ax = plt.subplots(figsize=(C.FIG_W, C.FIG_W * 0.32), constrained_layout=True)
    for i, (r, a) in enumerate(combos):
        v = []
        for t in PRIMARY9:
            q = MI[(r, a, "REP", "P_01", t, "P_main", "p")]
            rel = q["mae"] / q["mae_null"] if (_fin(q["mae_null"]) and q["mae_null"] > 0) else np.nan
            v.append(rel)
            rows.append(dict(rung=r, arm=a, target=t, source=q["source"], pair_set="P_01", n_para=q["n"],
                             para_mae=q["mae"], para_mae_null=q["mae_null"], rel_null=rel))
        col = DF.SLOT[RUNG_SLOT[r]]
        sh = a == "SH"
        ax.bar(x + (i - (len(combos) - 1) / 2) * w, v, width=w * 0.88, color=col, alpha=1.0 if sh else 0.45,
               hatch=None if sh else "////", edgecolor=col, linewidth=0.4, label=f"{r} · {a}")
    ax.axhline(1.0, color=DF.MUTED, lw=0.6)
    ax.set_xticks(x, [DF.TLAB[t] for t in PRIMARY9])
    ax.set_ylabel("para MAE / null para MAE")
    ax.set_title("REP, P_01 test pairs, P_main para pairs (1 = null, ΔΔ = 0)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    out = _save(fig, figdir, "figL2_targets", pd.DataFrame(rows))
    plt.close(fig)
    return out


def figL3(plt, DF, C, rates, rungs, figdir):
    """oi / cpcm bend-call sensitivity vs false-positive rate per rung x arm x model (own sets), goal region shaded."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    models = ("AW", "GNN", "TF", "XGB", "ENS4", "REP", "LINE")
    rows = []
    fig, axs = plt.subplots(1, len(GOAL_TARGETS), figsize=(C.FIG_W, C.FIG_W * 0.4), constrained_layout=True,
                            squeeze=False)
    for ax, t in zip(axs[0], GOAL_TARGETS):
        ax.add_patch(Rectangle((0.0, GOAL["sens"]), GOAL["fpr"], 1.05 - GOAL["sens"], facecolor=DF.GRID,
                               edgecolor=DF.INK2, lw=0.6, ls="--", zorder=0))
        for r in rungs:
            for a in ARMS:
                for m in models:
                    free = m in ARM_FREE
                    if free and a != ARMS[0]:
                        continue
                    br = rates[f"{r}|{a}|{m}|own"]["all"][t]
                    rows.append(dict(target=t, rung=r, arm="-" if free else a, model=m, pair_set=f"P_{r}",
                                     n_fam=br["n_fam"], n_bent_dft=br["n_bent_dft"], sens=br["sens"],
                                     sens_ci_lo=br["sens_ci"][0], sens_ci_hi=br["sens_ci"][1],
                                     n_straight_dft=br["n_straight_dft"], fpr=br["fpr"], fpr_ci_lo=br["fpr_ci"][0],
                                     fpr_ci_hi=br["fpr_ci"][1]))
                    if not (_fin(br["sens"]) and _fin(br["fpr"])):
                        continue
                    col = _color(DF, m)
                    filled = free or a == "SH"
                    ax.scatter([br["fpr"]], [br["sens"]], s=24, marker=RUNG_MARK[r],
                               facecolors=col if filled else "white", edgecolors="white" if filled else col,
                               linewidths=0.8, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("false-positive rate (DFT-straight families called bent)")
        ax.set_ylabel("sensitivity (DFT-bent families called bent)")
        ax.set_title(DF.TLAB[t])
    hs = [Line2D([], [], color=_color(DF, m), marker="o", ls="none", ms=4.5, label=m) for m in models]
    hs += [Line2D([], [], color=DF.INK2, marker=RUNG_MARK[r], ls="none", ms=4.5, label=f"rung {r}") for r in rungs]
    hs += [Line2D([], [], color=DF.INK2, marker="o", ls="none", ms=4.5, label="SH: filled"),
           Line2D([], [], color=DF.INK2, marker="o", ls="none", ms=4.5, mfc="white", label="S: open"),
           Patch(facecolor=DF.GRID, edgecolor=DF.INK2, ls="--", lw=0.6, label="goal: sens ≥ 0.7, FPR ≤ 0.1")]
    axs[0, -1].legend(handles=hs, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Bend calls (hammett_eval.calls), own pair sets, P_main para families (≥ 12 points incl. H)",
                 fontsize=C.FONT_PT)
    out = _save(fig, figdir, "figL3_bends", pd.DataFrame(rows))
    plt.close(fig)
    return out


# ================================================================ the E1 pipeline
def run_e1(EV, info, ctx, P, rep, means, out, figures=True):
    """gates -> frames -> metrics -> bends -> goal -> compare -> label effect -> files under `out` (+ figures)."""
    t0 = time.time()
    rungs, S, Y, sets = ctx["rungs"], ctx["S"], ctx["Y"], ctx["sets"]
    p01 = sets["P_01"]
    ref = {r: ref_set(ctx["tests"][r], r) for r in rungs}
    fr = model_frames(P, rungs)
    gm = {r: EV.pair_sets_gate({f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS4}, ref[r], Y)
          for r in rungs}
    check_pair_attrs(P, S)
    lines = line_frames(ctx)
    fr = derived_frames(fr, lines, rep, rungs)
    ga = gate_pairs(EV, fr, ref, Y, sets, rungs)
    lchk = check_line(fr, lines, rungs)
    _log(t0, "gates passed: test-pair lists (spec 16 E), LINE = SH baseline (E1-5)")
    A = annotate_all(EV, fr, S, p01, rungs)
    Mt = metrics_e1(EV, A)
    MI = met_index(Mt)
    _log(t0, f"metrics: {len(Mt)} rows")
    Bd, rates, fams = bends_e1(EV, A, S, rep)
    _log(t0, f"bends: {len(Bd)} rows")
    goal, verdict, concl = goal_e1(MI, rates, rungs)
    Cmp = compare_e1(EV, fr, S, ref, p01, rep, rungs)
    _log(t0, f"compare: {len(Cmp)} rows")
    LE, le_meta = label_effect(EV, info, ctx, A, ref)
    _log(t0, f"label effect: {len(LE)} rows")
    outputs = {}
    for name, df in (("E1_metrics.csv", Mt), ("E1_bends.csv", Bd), ("E1_goal.csv", goal), ("E1_compare.csv", Cmp),
                     ("E1_label_effect.csv", LE)):
        p = _write_csv(out / name, df)
        outputs[str(p)] = L.sha256(p)
    p = _write_json(out / "E1_bend_rates.json", dict(
        key="rung|arm|model|pair_set (pair_set own = P_<rung>, P_01)", rates=rates, families=fams,
        representatives=rep_json(rep, means),
        note="d1_eval.bend_rates on FU2 bends_tab rows: sens = DFT-bent families called bent / DFT-bent, fpr = DFT-"
             "straight called bent / DFT-straight, V_sens likewise for V calls; *_ci = d1_eval.clopper (Clopper-Pearson "
             "95 %); null = zero denominator (E1-7); LINE / null are arm-independent (both arms listed, E1-4)"))
    outputs[str(p)] = L.sha256(p)
    if figures:
        plt, DF = _plot()
        figdir = out / "figures"
        for f in figL1(plt, DF, EV.C, MI, rungs, figdir) + figL2(plt, DF, EV.C, MI, rungs, figdir) + \
                figL3(plt, DF, EV.C, rates, rungs, figdir):
            outputs[f] = L.sha256(f)
        _log(t0, f"figures written to {figdir}")
    gates = dict(models=gm, all=ga, n_test_pairs={r: len(ref[r]) for r in rungs},
                 n_P01_test_pairs=len({p for _, p in ref[rungs[0]] if p in p01}),
                 n_pairs_in_sets={n: len(v) for n, v in sets.items()})
    return dict(fr=fr, Mt=Mt, MI=MI, Bd=Bd, rates=rates, goal=goal, verdict=verdict, concl=concl, Cmp=Cmp, LE=LE,
                le_meta=le_meta, gates=gates, line_check=lchk, line_betas={r: lines[r][1] for r in rungs},
                outputs=outputs, seconds=time.time() - t0, ref=ref)


def cmd_e1(stage, labels, figures=True):
    if stage not in STAGE_RUNGS:
        print(f"usage: evaluate.py e1 --stage 1 (stage {stage}: E2 = spec 12, not written yet)", file=sys.stderr)
        return 2
    t0 = time.time()
    rungs = STAGE_RUNGS[stage]
    EV, info = fu2_eval()
    import inputs as IN
    lab = label_info(labels)
    labels = Path(lab["path"])
    rep_p = IN.p_report()
    for r in rungs:
        IN.p_report(r)                                   # rung build passed and current, table = P_report's
    tables = {r: rep_p["rungs"][r]["table_sha256"] for r in rungs}
    add3 = add3_check()
    for p in (PRED, L_REPORT, L_VAL):
        if not p.is_file():
            _not_ready(f"{p} missing (train_atoms.py collect --stage 1 --with-xgb)")
    with no_test_scores([PRED, UNITS]):                  # E1-3: REP from validation records only
        V, units = val_table(rungs, lab["sha256"], add3["md_sha256"], tables)
        rep, means = pick_rep(V)
    _log(t0, "REP: " + ", ".join(f"{r}·{a}·{t}={m}" for (r, a, t), m in rep.items()))
    lrep = check_l_report(units, rungs, labels)
    ctx = real_inputs(IN, rep_p, rungs, labels)
    P = load_predictions(rungs)
    res = run_e1(EV, info, ctx, P, rep, means, L.RES, figures)
    me = Path(__file__).resolve()
    ins = [PRED, L_REPORT, L_VAL, labels, ORIG_LABELS, IN.S_FOLDS, IN.P_REPORT, ADD3_MD, ADD3_JSON,
           Path(info["source"]), Path(info["e_tests"])] + [Path(rep_p["sets"][n]["file"]) for n in ("P_dft", "P_g1", "P_01")]
    doc = dict(
        phase="E1", stage=stage, rungs=list(rungs), command=" ".join(sys.argv[1:]), date=time.strftime("%F %T"),
        evaluate_py=dict(path=str(me), sha256=L.sha256(me), git_state=_git_state(me) or "committed, unchanged"),
        fu2_evaluate=info, labels=lab, add3=add3, l_report=lrep, units=units,
        inputs_sha256={str(p): L.sha256(p) for p in ins},
        representatives=rep_json(rep, means),
        rep_rule="spec 13-1: lowest mean over the 5 outer folds of the unit json val_kcal_mae, ties AW -> GNN -> TF -> "
                 "XGB; chosen before any prediction file was read (E1-3); train_atoms MA-6n selection asymmetry noted",
        gates=res["gates"], line_check=res["line_check"], line_betas=res["line_betas"], verdict=res["verdict"],
        conclusion_13_3=res["concl"], label_effect=res["le_meta"], outputs=res["outputs"],
        goal_rule=dict(targets=list(GOAL_TARGETS), para_mae_max=GOAL["mae"], sens_min=GOAL["sens"], fpr_max=GOAL["fpr"],
                       set="own pair set (P_<rung>), P_main reference para pairs, point estimate; surely = CI upper <= "
                           "0.13; REP primary, ENS4 secondary, single models descriptive",
                       bend_population="d1_eval.bend_rates 'all' (no family filter; ADD3 5-2 / PR-15)",
                       stage0_rev9=STAGE0_NOTE, stage1_rev9=f"R2_fam median (P_main, all) >= {STAGE1_R2}, descriptive"),
        env=_env(), decisions=DECISIONS, seconds=time.time() - t0)
    _write_json(L.RES / "E1_report.json", doc)
    for k, v in res["verdict"].items():
        if v["role"] != "descriptive":
            print(f"[E1] 13-2 {k}: passed {v['passed']} surely {v['surely_passed']}", flush=True)
    print(f"[E1] 13-3 stage 1: {res['concl'].get('conclusion_ko')} ({res['concl'].get('conclusion_en')})", flush=True)
    return 0


# ================================================================ selftest (E1-13)
def selftest_inputs():
    """synthetic E1 inputs over the real pair universe (read only): P_g1 = the FU2 pairs (both ends in the FU2
    T_manifest), P_dft = P_g1 minus every 50th pair plus 20 pairs outside it; synthetic dH; predictions with the AW of
    dft·SH exact; a validation table with planned REP choices and two exact ties."""
    import fu2_common as F2
    rng = np.random.default_rng(L.SEED_M)
    rungs = STAGE_RUNGS[1]
    S = prep_S(pd.read_csv(S_FOLDS, dtype={"pair_id": str}))
    newp = L.LAD / "F_pairs_cr.csv"
    if not newp.is_file():
        newp = ORIG_LABELS
    Y, Yold = read_lab(newp), read_lab(ORIG_LABELS)
    have = set(pd.read_csv(L.DP.RES / "fu2" / "T_manifest.csv", usecols=["rxn_id"]).rxn_id.astype(int))
    ids = S.index.to_numpy()
    use = (S.a_rxn.astype(int).isin(have) & S.b_rxn.astype(int).isin(have)).to_numpy()
    g1 = sorted(ids[use].tolist())
    p_g1 = set(g1)
    finite = (np.isfinite(Y.loc[ids, DD].to_numpy(float)).all(1) & np.isfinite(Yold.loc[ids, DD].to_numpy(float)).all(1))
    p_dft = (p_g1 - set(g1[::50])) | set(ids[~use & finite].tolist()[:20])
    sets = {"P_dft": p_dft, "P_g1": p_g1, "P_01": p_dft & p_g1}
    folds = {k: json.loads((F2.SPLITS / f"T1__f{k}.json").read_text()) for k in range(NF)}
    tests, trains, dH = {}, {}, {}
    for r in rungs:
        keep = sets[f"P_{r}"]
        tests[r] = {k: [str(p) for p in folds[k]["test"] if p in keep] for k in range(NF)}
        trains[r] = {k: [str(p) for p in folds[k]["train"] if p in keep] for k in range(NF)}
        u = sorted(keep)
        d = 0.8 * Y.loc[u, DD].to_numpy(float) + rng.normal(0.0, 0.5, (len(u), len(TARGETS)))
        d[:, TARGETS.index("c_ghost")] = 0.0
        dH[r] = pd.DataFrame(d, index=pd.Index(u, name="pair_id"), columns=TARGETS)
    noise = {"AW": 0.25, "GNN": 0.3, "TF": 0.35, "XGB": 0.4}
    parts = []
    for r in rungs:
        for a in ARMS:
            for m in MODELS4:
                for k in range(NF):
                    te, tr = tests[r][k], trains[r][k]
                    if not te:
                        continue
                    yv = Y.loc[te, DD].to_numpy(float)
                    if a == "SH":
                        beta = line_beta(dH[r].loc[tr, TARGETS].to_numpy(float), Y.loc[tr, DD].to_numpy(float))
                        base = line_predict(beta, dH[r].loc[te, TARGETS].to_numpy(float))
                    else:
                        beta, base = np.zeros(len(TARGETS)), np.zeros_like(yv)
                    if (r, a, m) == ("dft", "SH", "AW"):
                        yh = yv.copy()
                    else:
                        yh = yv + rng.normal(0.0, noise[m], yv.shape)
                    att = S.loc[te, ["ref_kind", "pos"]]
                    for j, t in enumerate(TARGETS):
                        parts.append(pd.DataFrame(dict(
                            pair_id=te, ref_kind=att.ref_kind.to_numpy(), pos=att.pos.to_numpy(), fold=int(k), rung=r,
                            arm=a, model=m, target=t, y=yv[:, j], yhat=yh[:, j], base=base[:, j], beta=float(beta[j]),
                            yhat_seed_sd=0.0)))
    P = pd.concat(parts, ignore_index=True)[PRED_COLS]
    vrows, expect = [], {}
    for r in rungs:
        for a in ARMS:
            for t in TARGETS:
                for m in MODELS4:
                    for k in range(NF):
                        v = 1.0 + 0.1 * MODEL_ORDER[m] + float(rng.uniform(0.0, 0.01))
                        if (r, a) == ("dft", "SH") and m == "AW":
                            v = 0.1
                        if (r, a) == ("g1", "SH") and m == "TF":
                            v = 0.2
                        if (r, a, t) == ("g1", "S", "oi") and m in ("AW", "GNN"):
                            v = 0.5
                        if (r, a, t) == ("g1", "S", "cpcm") and m in ("GNN", "XGB"):
                            v = 0.5
                        vrows.append(dict(rung=r, arm=a, model=m, fold=k, target=t, val_kcal_mae=v))
                if (r, a) == ("g1", "SH"):
                    expect[(r, a, t)] = "TF"
                elif (r, a, t) == ("g1", "S", "cpcm"):
                    expect[(r, a, t)] = "GNN"
                else:
                    expect[(r, a, t)] = "AW"
    ctx = dict(rungs=rungs, S=S, Y=Y, Yold=Yold, sets=sets, tests=tests, trains=trains, dH=dH)
    return ctx, P, pd.DataFrame(vrows), expect, dict(new_labels=str(newp), new_labels_sha256=L.sha256(newp))


def cmd_selftest(figures=True):
    t0 = time.time()
    EV, info = fu2_eval(allow_unmerged=True)
    ctx, P, V, expect, sinfo = selftest_inputs()
    checks = {}
    with no_test_scores([PRED, UNITS]):
        rep, means = pick_rep(V)
        checks["guard_blocks_test_scores"] = _expect_stop(
            lambda: pd.read_parquet(UNITS / "e1_selftest_probe.parquet"))
    checks["rep_rule_and_ties"] = bool(set(rep) == set(expect) and all(rep[k] == v for k, v in expect.items()))
    ref0 = ref_set(ctx["tests"]["dft"], "dft")
    q0 = P[(P.rung == "dft") & (P.arm == "S") & (P.model == "AW")]
    EV.pair_sets_gate({"selftest": q0}, ref0, ctx["Y"])
    checks["gate_detects_a_missing_pair"] = _expect_stop(
        lambda: EV.pair_sets_gate({"selftest": q0.iloc[1:]}, ref0, ctx["Y"]))
    res = run_e1(EV, info, ctx, P, rep, means, SELFTEST, figures)
    S, Y, MI, Mt = ctx["S"], ctx["Y"], res["MI"], res["Mt"]
    f = res["fr"][("dft", "S", "ENS4")]
    row = f.iloc[0]
    q = P[(P.rung == "dft") & (P.arm == "S") & (P.pair_id == row["pair_id"]) & (P.target == row["target"])]
    checks["ens4_is_the_mean"] = bool(len(q) == 4 and abs(float(q.yhat.mean()) - float(row["yhat"])) <= 1e-12)
    pids = [p for _, p in res["ref"]["dft"] if S.at[p, "in_P_main"] and S.at[p, "ref_kind"] == "reference"
            and S.at[p, "pos"] == "p"]
    want = float(np.mean(np.abs(Y.loc[pids, "dd_oi"].to_numpy(float))))
    got = MI[("dft", "S", "null", "own", "oi", "P_main", "p")]["mae"]
    checks["null_para_mae_is_mean_abs_y"] = bool(abs(got - want) <= 1e-12)
    ci = Mt[Mt.mae_ci_lo.notna()]
    checks["ci_rows_present_and_ordered"] = bool(len(ci) > 0 and (ci.mae_ci_lo <= ci.mae_ci_hi).all())
    n_out = int(((ci.mae < ci.mae_ci_lo - 1e-12) | (ci.mae > ci.mae_ci_hi + 1e-12)).sum())
    vd = res["verdict"].get("dft|SH|REP") or {}
    checks["goal_passes_for_the_exact_model"] = bool(vd.get("passed") and vd.get("surely_passed"))
    checks["conclusion_13_3_dft_pass"] = bool(res["concl"].get("dft_passed"))
    Cm = res["Cmp"]
    c = Cm[Cm.comparison.str.startswith("SH vs S") & (Cm.rung_b == "dft") & (Cm.model_b == "AW")
           & (Cm["eval"] == "P_main") & (Cm.target == "oi")]
    checks["compare_sign"] = bool(len(c) == 1 and c.d.iloc[0] > 0 and c.mae_b.iloc[0] == 0.0)
    checks["line_equals_sh_baseline"] = bool(all(v["sh_base_max_abs"] <= 1e-12 and v["s_rows_with_baseline"] == 0
                                                 for v in res["line_check"].values()))
    LE = res["LE"]
    dc = LE[(LE.part == "dft_calls") & ~LE.target.isin(list(CHANGED))]
    checks["unchanged_channels_keep_dft_calls"] = bool(len(dc) > 0 and ((dc.n_orig_only + dc.n_new_only) == 0).all())
    xc = res["le_meta"].get("fu2_metrics_crosscheck") or {}
    if xc.get("status") == "compared":
        checks["fu2_E_metrics_reproduced"] = bool(xc.get("n_rows_compared", 0) > 0
                                                  and (xc.get("max_abs_diff_mae") or 0.0) <= 1e-9
                                                  and xc.get("n_nan_pattern_differs_mae", 1) == 0)
    if figures:
        figs = [k for k in res["outputs"] if "/figures/" in k]
        checks["figures_written"] = bool(len(figs) == 9 and all(Path(k).is_file() for k in figs))
    ok = all(checks.values())
    doc = dict(ok=ok, checks=checks, n_metrics_rows_with_mae_outside_ci=n_out, synthetic=sinfo,
               representatives=rep_json(rep, means), expected_rep={"|".join(k): v for k, v in expect.items()},
               verdict=res["verdict"], conclusion_13_3=res["concl"], gates=res["gates"], line_check=res["line_check"],
               label_effect=res["le_meta"], fu2_evaluate=info, outputs=res["outputs"], env=_env(),
               evaluate_py_sha256=L.sha256(Path(__file__).resolve()), line_source_sha256=line_source_sha256(),
               seconds=time.time() - t0,
               note="synthetic predictions: numbers are not results; checks only (E1-13)")
    _write_json(SELFTEST / "E1_selftest.json", doc)
    for k, v in checks.items():
        print(f"[selftest] {k}: {'ok' if v else 'FAIL'}", flush=True)
    print(f"[selftest] {'all checks passed' if ok else 'FAILED'} ({time.time() - t0:.0f} s) -> {SELFTEST}", flush=True)
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description="D1-LADDER evaluation (docs/specs/D1LADDER.md 11): e1 | e1-selftest")
    ap.add_argument("cmd", choices=("e1", "e1-selftest"))
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    _need_pandas()
    if a.cmd == "e1":
        return cmd_e1(a.stage, a.labels, figures=not a.no_figures)
    return cmd_selftest(figures=not a.no_figures)


if __name__ == "__main__":
    sys.exit(main())
