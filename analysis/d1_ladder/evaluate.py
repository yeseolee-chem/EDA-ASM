#!/usr/bin/env python3
r"""evaluate.py — D1-REF·LADDER evaluation (docs/specs/D1LADDER.md 4-5, 11, 12, 13, 14-1, 16 E). sbatch only. Nothing in
the E1 or E2 part was run when it was written (login node): every path is untested until it runs inside a job (start
with `e1-selftest` / `e2-selftest`, which need no Phase M output).

LINE (spec 4-5; the unchanged contract of tests/run_tests.py L-10, C-11 / C-18): per target t, beta_t = sum(dH_t * y_t) /
sum(dH_t^2) on exactly the given pairs, no intercept (antisymmetric; = D1-HF Phase A report.py 'a'), c_ghost beta = 0
(no HF counterpart); prediction beta * dH. line_beta(dH, y) -> beta [10] in TARGETS order; line_predict(beta, dH).
The module top level needs numpy and lad_common only (pandas is optional there), so L-10 can import it in any env.

CLI (submit from analysis/d1_pair through jobs/py.slurm with PYENV=espley, script ../d1_ladder/evaluate.py; exit 0 =
done, 2 = not ready / usage, 3 = STOP)
  python evaluate.py e1 [--stage 1] [--labels P] [--no-figures]
        Phase E1 (spec 11, 13; stage 1 = rungs dft, g1) -> results/E1_metrics.csv, E1_bends.csv, E1_bend_rates.json,
        E1_goal.csv, E1_compare.csv, E1_label_effect.csv, E1_report.json (provenance, REP, gates, verdicts, decisions)
        and results/figures/figL1_ladder, figL2_targets, figL3_bends (.png, .pdf, .csv). --stage 2: exit 2 (E2 = the
        command e2, E2-1). Exit 2 also while the FU2 Phase E code is not merged and committed in this worktree (spec 11,
        E1-2).
  python evaluate.py e1-selftest [--no-figures]
        the same pipeline on SYNTHETIC predictions over the real pair universe (S_folds.csv, the FU2 T1 split files,
        F_pairs_cr.csv as 'new' labels, the real FU2 / rev 9 frames for the label effect) -> $LAD/e1_selftest/ (the
        E1_* files, figures/, E1_selftest.json with the checks); exit 3 when a check fails. Never writes results/.
        The selftests are the only commands that may use the unmerged d1-pair copy of the FU2 Phase E code (E1-2).
  python evaluate.py e2 [--labels P] [--no-figures]
        Phase E2 (spec 12, 13, 14-1; rungs dft, g1, g2; inputs of `train_atoms.py collect --stage 2 --with-xgb`) ->
        results/E2_metrics.csv, E2_bends.csv, E2_bend_rates.json, E2_goal.csv (13-2, 6 rung x arm), E2_interpretation.json
        (13-3 a / b / c), E2_compare.csv (13-4 incl. rung vs rung on P_012), E2_coverage.json, E2_same_ts.csv,
        E2_report.json (provenance, gates, decisions) and results/figures/figL1_ladder, figL2_targets, figL3_bends,
        figL4_examples (.png, .pdf, .csv; they replace E1's stage-1 versions, spec 14-1). Exit 2 = an input missing /
        stale (E2-3), or no results/E1_report.json of `e1 --stage 1` on the same dft unit files (E2-14: 13-3 uses the
        stage-1 dft verdict; E1 writes the same figure names, so E1 must finish before E2), 3 = STOP (spec 16 E pair
        lists, pre-registration, unit code_sha256 (ADD3 11 E, E2-15), coverage / class counts, dft verdict or dft REP
        != E1's).
  python evaluate.py e2-selftest [--no-figures]
        E2 on the E1-13 synthetic inputs + a synthetic g2 rung (E2-13) -> $LAD/e2_selftest/ (E2_* files, figures/,
        E2_selftest.json); the coverage / same-TS class counts are also recomputed on the REAL Phase P sets and compared
        with the supplement. Needs no Phase M2 output; exit 3 when a check fails. Never writes results/.

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
  E2 selftest (now; no Phase M2 output needed): the selftest line with -J lad_e2t and `e2-selftest`.
  E2 (after `train_atoms.py collect --stage 2 --with-xgb`, and always after E1: e2 exits 2 without a current
  results/E1_report.json, E2-14). E1 then E2 in one job (py.slurm ' ::: ' runs e2 only if e1 exited 0):
  sbatch -p cpu1,cpu2 -J lad_e12 --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
         --export=ALL,PYENV=espley --dependency=afterok:<collect job> --kill-on-invalid-dep=yes \
         jobs/py.slurm ../d1_ladder/evaluate.py e1 --stage 1 ::: ../d1_ladder/evaluate.py e2
  (E1 already run on the same dft units: the same line with -J lad_e2 and only `../d1_ladder/evaluate.py e2`.) Do not
  rerun E1 after E2: it rewrites figL1-L3 with the stage-1 versions (then rerun e2).

GEO2 Phase E, model part (docs/specs/GEO2.md 9-2, 10; added 2026-10-30, role LADB; decisions E3-1 .. E3-8 + E3-2b in
DECISIONS3, also in analysis/geo2/results/DECISIONS_tests.md; nothing was run). e1 / e2 / their selftests are unchanged.
  python evaluate.py e3 [--labels P]
        the E2 computation on rung geo: rungs dft (pair set P_023), g2 (own P_g2, P_23, P_023), geo (own P_geo, P_23,
        P_023); inputs = the LADDER E2 inputs + train_atoms.py collect --stage 3 --with-xgb (results/
        L_predictions_geo.parquet, L_report_geo.json, L_val_geo.csv) + train_xgb.py check --stage 3
        (L_xgb_report_geo.json) + the GEO2 Phase P files (P_build_geo.json, P_geo_report.json, P_geo_sets.json,
        $GEO2/inputs/geo.parquet) + PREREG_GEO2 -> analysis/geo2/results/E3_metrics.csv (E2_metrics.csv format; AW, GNN,
        TF, XGB, REP, ENS4, LINE, null), E3_goal.csv (LADDER 13-2 incl. the bend part + REP8 rows + bend_passed),
        E3_bends.csv, E3_bend_rates.json, E3_compare.csv, E3_report.json (provenance, REP / REP8, 'goal_reached' =
        GEO2 10-1, E2 / eval_geo2 cross-checks). Exit 2 = an input missing / stale, 3 = STOP (pair lists, sets,
        pre-registration, the E2 cross-check E3-7).
  python evaluate.py e3-selftest
        the same pipeline with geo := a copy of g2 (needs no GEO2 output) -> analysis/geo2/results/e3_selftest/
        (E3_* + E3_selftest.json with the checks); exit 3 when a check fails.
  sbatch (from analysis/d1_pair): sbatch -p cpu1,cpu2 -J geo2_e3 --time=48:00:00 --cpus-per-task=4 --mem=32G \
         --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out --export=ALL,PYENV=espley \
         jobs/py.slurm ../d1_ladder/evaluate.py e3        (or e3-selftest; analysis/geo2/RUNBOOK_tests.md)

What E2 computes (spec 12, 13; decisions E2-1 .. E2-15 in DECISIONS2, recorded in E2_report.json)
  the E1 pipeline (REP, ENS4, LINE, null; val_table, pick_rep, line_frames, derived_frames, metrics_e1, bends_e1, goal_e1
  reused unchanged) on rungs dft, g1, g2 with pair sets own and P_012; + 13-3 a / b / c (exact sentences, checked
  verbatim against spec 13-3 and ADD3 5-3), 13-4 on P_012, coverage (supplement 4) and the same-TS split (supplement 5:
  primary same_ts_g1, second same_ts_dft), both recomputed from the Phase P / D1-G2 files and compared with the counts
  pre-registered in prereg_d1pair_add3.json 'supplement'. 13-3 is applied with the stage-1 dft verdict (supplement 7):
  E2's dft REP choices and dft·S / dft·SH REP verdicts must equal results/E1_report.json on the same dft unit files
  (E2-14). Every unit fingerprint code_sha256 = the ADD3 json code.fingerprint_keys (ADD3 11 E, E2-15). No label
  effect (E1 only) and no narrative report.
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


def real_inputs(IN, rep_p, rungs, labels, set_names=("P_dft", "P_g1", "P_01")):
    """S_folds, labels (new + original), pair sets, fold lists and dH (h = cr) from the Phase P files (inputs.py API).
    set_names: the pair sets read (E1: the default; E2: E2_SETS, which adds P_g2 and P_012)."""
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
    sets = {n: set(IN.read_set(n, rep_p).pair_id.astype(str)) for n in set_names}
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


def annotate_all(EV, fr, S, p01, rungs, common_name="P_01"):
    """{(rung, arm, model, pair_set): frame + S_folds columns (FU2 annotate)}; LINE / null once (arm S, E1-4).
    The common set p01 is stored under common_name (E1: P_01; E2: P_012)."""
    A = {}
    for (r, a, m), f in fr.items():
        if m in ARM_FREE and a != ARMS[0]:
            continue
        d = EV.annotate(f, S)
        A[(r, a, m, "own")] = d
        if len(rungs) > 1:
            A[(r, a, m, common_name)] = d[d.pair_id.isin(p01)]
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
    X = pd.read_csv(mpath, keep_default_na=False, na_values=[""])   # model "null" must stay a string
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


def figL1(plt, DF, C, MI, rungs, figdir, ps="P_01"):
    """oi / cpcm para MAE vs rung on the common set ps (E1: P_01, E2: P_012) (models: SH solid filled, S dashed open;
    LINE, null; goal 0.13; 95 % CI)."""
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
            recs = [MI[(r, a, m, ps, t, "P_main", "p")] for r in rungs]
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
                rows.append(dict(target=t, rung=r, model=m, arm="-" if free else a, source=q["source"], pair_set=ps,
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
    fig.suptitle(f"{ps} test pairs, P_main para pairs; bars = 95 % CI (core bootstrap, 2,000)", fontsize=C.FONT_PT)
    out = _save(fig, figdir, "figL1_ladder", pd.DataFrame(rows))
    plt.close(fig)
    return out


def figL2(plt, DF, C, MI, rungs, figdir, ps="P_01"):
    """9 targets: para MAE / null para MAE of REP per rung x arm (common set ps; E1: P_01, E2: P_012)."""
    combos = [(r, a) for r in rungs for a in ARMS]
    x = np.arange(len(PRIMARY9), dtype=float)
    w = 0.84 / len(combos)
    rows = []
    fig, ax = plt.subplots(figsize=(C.FIG_W, C.FIG_W * 0.32), constrained_layout=True)
    for i, (r, a) in enumerate(combos):
        v = []
        for t in PRIMARY9:
            q = MI[(r, a, "REP", ps, t, "P_main", "p")]
            rel = q["mae"] / q["mae_null"] if (_fin(q["mae_null"]) and q["mae_null"] > 0) else np.nan
            v.append(rel)
            rows.append(dict(rung=r, arm=a, target=t, source=q["source"], pair_set=ps, n_para=q["n"],
                             para_mae=q["mae"], para_mae_null=q["mae_null"], rel_null=rel))
        col = DF.SLOT[RUNG_SLOT[r]]
        sh = a == "SH"
        ax.bar(x + (i - (len(combos) - 1) / 2) * w, v, width=w * 0.88, color=col, alpha=1.0 if sh else 0.45,
               hatch=None if sh else "////", edgecolor=col, linewidth=0.4, label=f"{r} · {a}")
    ax.axhline(1.0, color=DF.MUTED, lw=0.6)
    ax.set_xticks(x, [DF.TLAB[t] for t in PRIMARY9])
    ax.set_ylabel("para MAE / null para MAE")
    ax.set_title(f"REP, {ps} test pairs, P_main para pairs (1 = null, ΔΔ = 0)")
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
        print(f"usage: evaluate.py e1 --stage 1 (stage {stage}: E2 = spec 12 is the command 'evaluate.py e2', E2-1)",
              file=sys.stderr)
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


# ================================================================ Phase E2 (spec 12, 13, 14-1; stage 2): constants
E2_RUNGS = ("dft", "g1", "g2")                           # spec 12: E2 = spec 11 on every rung + the P_012 comparison
E2_SETS = ("P_dft", "P_g1", "P_01", "P_g2", "P_012")     # Phase P sets read (P_01 only enters the coverage table)
COMMON2 = "P_012"                                        # spec 13-4 / 14-1: the stage-2 common set (= P_01 & P_g2)
RUNG_PAIRS2 = (("dft", "g1", "dft vs g1 (what the xTB geometry loses)"),
               ("dft", "g2", "dft vs g2 (what the deployable SMILES -> xTB input loses)"),
               ("g1", "g2", "g1 vs g2 (what the SMILES-built TS loses against the relaxed label TS)"))
RUNG_LS = {"dft": "-", "g1": "--", "g2": ":"}            # figL4 secondary encoding (colour slot + marker + line style)
SUPP_MD = L.RES / "PREREG_D1PAIR_ADD3_G2.md"             # the stage-2 supplement (ADD3 10)
SPEC_MD = L.REPO / "docs" / "specs" / "D1LADDER.md"
C_COMPARE = L.G2_DIR / "results" / "C_compare.csv"       # D1-G2 Phase C per-reaction same-TS judgments (supplement 5)
SAME_TS_COLS = ("same_ts_g1", "same_ts_dft")             # supplement 5 / PS-7 order: primary first (D1-G2 C-13)
SAME_TS_CLASSES = ("same", "mixed", "different", "unclassified")
DROP_CATS = ("g2_not_ok", "not_target", "no_tokens_or_182", "no_h", "both_usable")   # supplement 4 / PS-8 order
SPLIT_MODELS = ("REP", "ENS4") + MODELS4 + ARM_FREE      # supplement 5: REP, ENS4, the 4 models, LINE, null
SELFTEST2 = L.LAD / "e2_selftest"
E1_REPORT = L.RES / "E1_report.json"                     # the stage-1 dft verdict (E2-14)
TRAINER = {m: ("train_xgb" if m == "XGB" else "train_atoms") for m in MODELS4}   # ADD3 json code.fingerprint_keys
RULES_13_3 = (                                           # spec 13-3 = ADD3 5-3, verbatim (checked at run time, E2-2)
    ("a", "g2·S 또는 g2·SH 통과", "배포 가능한 입력(SMILES → xTB)으로 목표에 도달했다"),
    ("b", "g2 둘 다 실패, dft·S 또는 dft·SH 통과",
     "모델은 목표에 닿을 수 있으나 xTB 구조의 정밀도가 병목이다. 구조를 DFT 수준으로 다듬는 경로를 시험할 근거가 있다"),
    ("c", "dft 둘 다 실패", "이 입력과 이 모델 네 개로는 구조가 정답이어도 목표에 닿지 않는다"))
G1_NOTE = ("g1 is not part of the 13-3 table (spec 13-3 / ADD3 5-3: 'g1 결과는 이 표에 넣지 않는다'); explanatory only "
           "(what the xTB geometry loses)")
COV_RULE = ("supplement 4 / PS-8 (fixed before training): coverage = P_main pairs (S_folds in_P_main) of the T1 test lists "
            "(FU2 split files 'test', folds 0-4) that are in the rung's set; pairs outside P_g2 count as not predictable "
            "(not filled in, not in the 13-2 scores); reason per pair = the first failing end category in DROP_CATS order")
SAME_TS_RULE = ("supplement 5 / PS-7 (fixed before training): reaction = D1-G2 C_compare.csv same_ts_g1 (primary; D1-G2 "
                "C-13: G2 TS vs G1 TS, both xTB) and same_ts_dft (second; G2 TS vs the label DFT TS); pair (a, b) = same "
                "(both ends same TS) / mixed (one end) / different (neither) / unclassified (an end without a judgment); "
                "descriptive only: not used for 13-2, 13-3 or REP; no tests between classes")
DECISIONS2 = {
    "E2-1": "new command 'e2' (+ 'e2-selftest'); 'e1 --stage 2' still exits 2 and names 'e2'; e1 --stage 1 unchanged: E2 "
            "reuses the E1 functions, four of which got a keyword argument whose default is the E1 literal (real_inputs "
            "set_names, annotate_all common_name, figL1 / figL2 ps); line_beta / line_predict untouched (ADD3 PR-13)",
    "E2-2": "pre-registration: add3_check (E1-12) + the supplement (json 'supplement' present, md committed and unchanged, "
            "sha256 = supplement.md_sha256; same_ts_split columns / primary / source as pinned; C_compare.csv committed "
            "with the pinned sha256) + the P_report pair sets and rung tables = json inputs_sha256 + the 13-3 rows of this "
            "file verbatim in the spec and ADD3 5-3; STOP otherwise",
    "E2-3": "inputs = train_atoms.py collect --stage 2 --with-xgb: L_report stage 2, rungs dft g1 g2, XGB, predictions / "
            "L_val sha256 current and no unit json newer than L_report.json, else exit 2 (missing / stale); then E1-3 "
            "unchanged (REP from the unit json validation MAE before any prediction file is read; check_l_report STOPs)",
    "E2-4": "rungs dft, g1, g2; pair sets own (P_<rung>, the 13-2 set) and P_012 (spec 13-4 / 14-1, stage 2); P_01 stays "
            "E1's; metrics / bends / para CI / NaN rules = E1-6 .. E1-8",
    "E2-5": "spec 16 E gates: per rung every model / arm (incl. REP, ENS4, LINE, null) has exactly the rung's test pairs "
            "(FU2 pair_sets_gate); on P_012 the three rungs' lists are identical and every frame has exactly that list; "
            "LINE = SH baseline (E1-5) on g2 too",
    "E2-6": "13-2 = goal_e1 on the 6 rung x arm (REP primary, ENS4 secondary, single models descriptive; surely = passed "
            "and para MAE CI upper <= 0.13); goal_e1's stage-1 '13-3' row is replaced by the stage-2 13-3 row",
    "E2-7": "13-3 = the first matching row (a, b, c) of spec 13-3 on the REP verdicts of g2·S, g2·SH, dft·S, dft·SH; the "
            "exact Korean sentence; g1 excluded (explanatory rows); ENS4 and 'surely' versions descriptive",
    "E2-8": "13-4: SH vs S per rung x model (own); rung vs rung per arm x model on P_012 for dft vs g1, dft vs g2, g1 vs g2 "
            "(b = the first named; + LINE with arm '-'); REP vs LINE per rung x arm (own); sign E1-9; P_main and para",
    "E2-9": "coverage = supplement 4 / PS-8 (COV_RULE) on the fold copies (sha256 = P_report) by fold, position and "
            "dipole, P_g2 missing reasons in DROP_CATS order; the recomputed counts must equal the supplement json and "
            "the evaluated P_main test pairs of every rung must equal its coverage count, else STOP",
    "E2-10": "same-TS split = supplement 5 / PS-7 (SAME_TS_RULE): primary same_ts_g1, second same_ts_dft as pre-registered "
             "(the task text called same_ts_dft primary; the committed supplement and json 'primary' win, both are "
             "descriptive); g2 test pairs in P_main; REP, ENS4, AW, GNN, TF, XGB, LINE, null; 9 targets; pos all / p; n, "
             "cores, MAE, RMSE per class and 'all' ('all' = E2_metrics, checked); class counts = supplement json, else STOP",
    "E2-11": "figL1 / figL2 on P_012 (dft, g1, g2), figL3 on the own sets, figL4 = the first 3 rev 9 fig3 families (FU2 "
             "fig3_families order) with a >= 12-point para series for oi and cpcm on P_012 in every rung: DFT truth (ink) "
             "vs REP of arm SH per rung (slots dft 1, g1 2, g2 3 + marker + line style); same file names as E1 in "
             "results/figures (E2 replaces the stage-1 versions, spec 14-1)",
    "E2-12": "E1 cross-check (record only): E2's dft / g1 own metrics, verdicts and REP vs results/E1_metrics.csv / "
             "E1_report.json; the dft REP and the dft verdict are gated by E2-14",
    "E2-13": "e2-selftest: the E1-13 synthetic inputs + a synthetic g2 rung (P_g2 = P_g1 minus every 13th sorted pair from "
             "the 4th) with planned REP choices and a tie -> $LAD/e2_selftest; the coverage / class counts of the REAL "
             "Phase P sets are compared with the supplement there too; E2-14 / E2-15 rules on constructed records, "
             "E2-15 also on the real unit jsons present",
    "E2-14": "13-3 with the stage-1 dft verdict (supplement 7 '1단계 dft 판정과 함께'): results/E1_report.json (phase E1, "
             "stage 1) must exist and list the same dft unit json / parquet sha256 as this run, else exit 2 (run / rerun "
             "e1 --stage 1 first; this also keeps E1 from replacing E2's figures); E2's dft REP choices (before any "
             "prediction is read) and dft·S / dft·SH REP passed / surely (after 13-2, before any output is written) must "
             "equal E1's, else STOP; recorded in E2_interpretation.json 'stage1_dft_e1'",
    "E2-15": "ADD3 11 E for every unit (supplement 7: unchanged for stage 2): fingerprint code_sha256 = prereg_d1pair_add3"
             ".json code.fingerprint_keys of its trainer (train_xgb for XGB, train_atoms otherwise), else STOP (val_table, "
             "reused unchanged from E1, checks add3 / labels / inputs_table / rung / arm only)",
}


def _log2(t0, msg):
    print(f"[e2 {time.time() - t0:7.0f}s] {msg}", flush=True)


def _close(a, b, tol=1e-12):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol


# ================================================================ E2 pre-registration and readiness (E2-2, E2-3)
def rules_verbatim():
    """E2-2: every RULES_13_3 row is a verbatim table row '| k | condition | sentence |' of spec 13-3 and ADD3 5-3."""
    out, bad = {}, []
    for p in (SPEC_MD, ADD3_MD):
        lines = set(p.read_text(encoding="utf-8").splitlines()) if p.is_file() else set()
        ok = {k: f"| {k} | {c} | {s} |" in lines for k, c, s in RULES_13_3}
        out[str(p)] = ok
        bad += [f"{p.name} row {k}" for k, v in ok.items() if not v]
    if bad:
        L.stop("the 13-3 rows of evaluate.py are not verbatim rows of the spec / ADD3: " + ", ".join(bad))
    return out


def stage2_prereg(rep_p, rungs, set_names):
    """E2-2: the stage-2 supplement (ADD3 10) committed and as recorded, C_compare.csv as pinned, the P_report pair sets
    and rung tables = ADD3 json inputs_sha256 (the trainers' gate list); STOP otherwise."""
    rec = json.loads(ADD3_JSON.read_text())
    s = rec.get("supplement")
    if not isinstance(s, dict):
        L.stop(f"{ADD3_JSON.name} has no 'supplement' key: the stage-2 supplement (ADD3 10) is not registered")
    bad = []
    if s.get("md") != _repo_rel(SUPP_MD):
        bad.append(f"supplement.md {s.get('md')} != {_repo_rel(SUPP_MD)}")
    if not SUPP_MD.is_file():
        bad.append(f"{SUPP_MD.name} missing")
    else:
        st = _git_state(SUPP_MD)
        if st:
            bad.append(f"{SUPP_MD.name} {st}")
        elif L.sha256(SUPP_MD) != s.get("md_sha256"):
            bad.append(f"{SUPP_MD.name} sha256 != {ADD3_JSON.name} supplement.md_sha256")
    sts = s.get("same_ts_split") or {}
    if list(sts.get("columns") or []) != list(SAME_TS_COLS) or sts.get("primary") != SAME_TS_COLS[0]:
        bad.append(f"supplement same_ts_split columns {sts.get('columns')} / primary {sts.get('primary')} != "
                   f"{list(SAME_TS_COLS)} / {SAME_TS_COLS[0]}")
    if sts.get("source") != _repo_rel(C_COMPARE):
        bad.append(f"supplement same_ts_split source {sts.get('source')} != {_repo_rel(C_COMPARE)}")
    if not C_COMPARE.is_file():
        bad.append(f"{C_COMPARE} missing")
    else:
        st = _git_state(C_COMPARE)
        if st:
            bad.append(f"{C_COMPARE.name} {st}")
        if L.sha256(C_COMPARE) != sts.get("source_sha256"):
            bad.append(f"{C_COMPARE.name} sha256 {L.sha256(C_COMPARE)[:12]} != supplement "
                       f"{str(sts.get('source_sha256'))[:12]}")
    ins = rec.get("inputs_sha256") or {}
    pre = {}
    for n in set_names:
        e = (rep_p.get("sets") or {}).get(n) or {}
        f, sha = e.get("file"), e.get("sha256")
        pre[str(f)] = sha
        if f is None or ins.get(f) != sha:
            bad.append(f"pair set {n} ({f}): P_report sha256 {str(sha)[:12]} != ADD3 json inputs_sha256 "
                       f"{str(ins.get(f))[:12]}")
    for r in rungs:
        e = (rep_p.get("rungs") or {}).get(r) or {}
        f, sha = e.get("table"), e.get("table_sha256")
        pre[str(f)] = sha
        if f is None or ins.get(f) != sha:
            bad.append(f"rung table {r} ({f}): P_report sha256 {str(sha)[:12]} != ADD3 json inputs_sha256 "
                       f"{str(ins.get(f))[:12]}")
    if bad:
        L.stop("stage-2 pre-registration (ADD3 10, supplement 7):\n  " + "\n  ".join(bad))
    return dict(record=s, md=str(SUPP_MD), md_sha256=L.sha256(SUPP_MD), md_git_state="committed, unchanged",
                json_sha256=L.sha256(ADD3_JSON), supplement_commit=s.get("commit"),
                c_compare=dict(path=str(C_COMPARE), sha256=L.sha256(C_COMPARE), git_state="committed, unchanged"),
                prereg_inputs_equal=pre)


def stage2_ready(rungs):
    """E2-3: `train_atoms.py collect --stage 2 --with-xgb` outputs present and current, else exit 2 (missing / stale)."""
    for p in (PRED, L_REPORT, L_VAL):
        if not p.is_file():
            _not_ready(f"{p} missing (train_atoms.py collect --stage 2 --with-xgb)")
    rep = json.loads(L_REPORT.read_text())
    why = []
    if int(rep.get("stage") or 0) < 2:
        why.append(f"L_report.json stage {rep.get('stage')} (not a stage-2 collect)")
    lack = sorted(set(rungs) - set(rep.get("rungs") or []))
    if lack:
        why.append(f"L_report.json rungs {rep.get('rungs')} lack {lack}")
    if "XGB" not in (rep.get("models") or []):
        why.append("collected without --with-xgb")
    if rep.get("predictions_sha256") != L.sha256(PRED):
        why.append("L_predictions.parquet differs from L_report.json (a collect running or interrupted?)")
    if rep.get("val_csv_sha256") != L.sha256(L_VAL):
        why.append("L_val.csv differs from L_report.json")
    t_rep = L_REPORT.stat().st_mtime
    newer = [x for r in rungs for a in ARMS for m in MODELS4 for k in range(NF) for x in [unit_name(r, a, m, k)]
             if (UNITS / f"{x}.json").is_file() and (UNITS / f"{x}.json").stat().st_mtime > t_rep]
    if newer:
        why.append(f"{len(newer)} unit jsons newer than L_report.json (e.g. {newer[:3]}): rerun the collect")
    if why:
        _not_ready("stage-2 collect output missing or stale: " + "; ".join(why))
    return dict(l_report_stage=rep.get("stage"), l_report_rungs=rep.get("rungs"), l_report_mtime=t_rep,
                rule="E2-3: exit 2 unless the stage-2 collect outputs are present and current")


def _code_fp_problems(fps, keys):
    """E2-15: problems of {unit: (model, fingerprint code_sha256)} against ADD3 json code.fingerprint_keys."""
    bad = []
    for x, (m, got) in sorted(fps.items()):
        tr = TRAINER[m]
        want = keys.get(tr)
        if not isinstance(want, dict) or not want:
            bad.append(f"{x}: ADD3 json code.fingerprint_keys has no '{tr}' entry")
        elif got != want:
            have = got if isinstance(got, dict) else {}
            diff = sorted(k for k in set(want) | set(have) if have.get(k) != want.get(k))
            bad.append(f"{x}: fingerprint code_sha256 != ADD3 json code.fingerprint_keys.{tr} (files {diff})")
    return bad


def code_fingerprints(names):
    """E2-15 (ADD3 11 E): the fingerprint code_sha256 of every unit json in `names` (unit_name strings) = ADD3 json
    code.fingerprint_keys of its trainer (train_xgb for XGB, train_atoms otherwise); STOP otherwise."""
    keys = (json.loads(ADD3_JSON.read_text()).get("code") or {}).get("fingerprint_keys") or {}
    fps = {}
    for x in names:
        rec = json.loads((UNITS / f"{x}.json").read_text())
        fps[x] = (x.split("__")[3], (rec.get("fingerprint") or {}).get("code_sha256"))
    bad = _code_fp_problems(fps, keys)
    if bad:
        L.stop("unit fingerprints: code_sha256 differs from the pre-registered code (ADD3 11 E, E2-15):\n  "
               + "\n  ".join(bad[:10]))
    return dict(rule="ADD3 11 E: unit fingerprint code_sha256 = prereg_d1pair_add3.json code.fingerprint_keys of the "
                     "trainer (train_xgb for XGB, train_atoms otherwise)", n_units=len(fps),
                n_by_trainer={tr: sum(1 for m, _ in fps.values() if TRAINER[m] == tr)
                              for tr in sorted(set(TRAINER.values()))},
                fingerprint_keys=keys, equal=True)


def _unit_files(u):
    return (u or {}).get("json_sha256"), (u or {}).get("parquet_sha256")


def _stage1_problems(s1, verdict=None, rep=None):
    """E2-14: differences between the stage-1 record s1 (e1_representatives {arm: {target: model}}, e1_verdict
    {'dft|<arm>|REP': {passed, surely_passed}}) and this run's dft REP choices / dft REP verdicts."""
    bad = []
    if rep is not None:
        for a in ARMS:
            for t in TARGETS:
                m1 = (s1["e1_representatives"].get(a) or {}).get(t)
                if m1 != rep[("dft", a, t)]:
                    bad.append(f"REP dft·{a}·{t}: E1 {m1} != E2 {rep[('dft', a, t)]}")
    if verdict is not None:
        for a in ARMS:
            k = f"dft|{a}|REP"
            e1 = tuple(bool(s1["e1_verdict"][k][q]) for q in ("passed", "surely_passed"))
            e2 = tuple(bool(verdict[k][q]) for q in ("passed", "surely_passed"))
            if e1 != e2:
                bad.append(f"{k}: E1 passed / surely {e1} != E2 {e2}")
    return bad


def stage1_dft(units, rep):
    """E2-14 (supplement 7: '13-3 해석 표 전체(a·b·c)를 1단계 dft 판정과 함께 적용'; spec 11 before 12): the stage-1 dft
    verdict = results/E1_report.json of `e1 --stage 1`. It must exist and list the same dft unit files (unit json and
    parquet sha256) as this run, else exit 2 (run / rerun E1 first); its dft REP choices must equal this run's (the same
    validation records), else STOP. The dft·S / dft·SH REP verdicts are compared after 13-2 (check_stage1_dft)."""
    if not E1_REPORT.is_file():
        _not_ready(f"{E1_REPORT} missing: run `evaluate.py e1 --stage 1` before e2 (E2-14: 13-3 uses the stage-1 dft "
                   "verdict, supplement 7; E1 writes the same figure files and must not run after E2)")
    r1 = json.loads(E1_REPORT.read_text())
    why = []
    if r1.get("phase") != "E1" or r1.get("stage") != 1 or "dft" not in (r1.get("rungs") or []):
        why.append(f"phase {r1.get('phase')} stage {r1.get('stage')} rungs {r1.get('rungs')} (not an e1 --stage 1 report)")
    u1 = r1.get("units") or {}
    dft = sorted(x for x, u in units.items() if u["key"][0] == "dft")
    stale = [x for x in dft if _unit_files(u1.get(x)) != _unit_files(units[x])]
    if not dft or stale:
        why.append(f"{len(stale)} of {len(dft)} dft units differ from the ones E1 evaluated (e.g. {stale[:3]})")
    v1 = r1.get("verdict") or {}
    lack = [f"dft|{a}|REP" for a in ARMS if not isinstance(v1.get(f"dft|{a}|REP"), dict)]
    if lack:
        why.append(f"verdict lacks {lack}")
    if why:
        _not_ready(f"{E1_REPORT} is not the current stage-1 dft verdict (rerun `evaluate.py e1 --stage 1`, then e2; "
                   "E2-14): " + "; ".join(why))
    rp1 = (r1.get("representatives") or {}).get("dft") or {}
    s1 = dict(path=str(E1_REPORT), sha256=L.sha256(E1_REPORT), date=r1.get("date"),
              slurm_job=(r1.get("env") or {}).get("slurm_job"),
              e1_predictions_sha256=(r1.get("l_report") or {}).get("predictions_sha256"),
              e1_conclusion_13_3=(r1.get("conclusion_13_3") or {}).get("conclusion_ko"),
              e1_representatives={a: {t: ((rp1.get(a) or {}).get(t) or {}).get("model") for t in TARGETS} for a in ARMS},
              e1_verdict={f"dft|{a}|REP": dict(passed=bool(v1[f"dft|{a}|REP"].get("passed")),
                                                surely_passed=bool(v1[f"dft|{a}|REP"].get("surely_passed")))
                          for a in ARMS},
              n_dft_units_equal=len(dft),
              g1_units_differ_from_e1=sorted(x for x, u in units.items()
                                             if u["key"][0] == "g1" and _unit_files(u1.get(x)) != _unit_files(u)))
    bad = _stage1_problems(s1, rep=rep)
    if bad:
        L.stop("dft REP differs from E1's on the same dft unit files (E2-14):\n  " + "\n  ".join(bad[:10]))
    return s1


def check_stage1_dft(s1, verdict, rep):
    """E2-14: this run's dft·S / dft·SH REP verdicts (passed, surely) and dft REP = the stage-1 ones, else STOP."""
    bad = _stage1_problems(s1, verdict=verdict, rep=rep)
    if bad:
        L.stop("the dft verdict / REP differs from the stage-1 one (results/E1_report.json) on the same dft units "
               "(E2-14: 13-3 is applied with the stage-1 dft verdict, supplement 7):\n  " + "\n  ".join(bad))
    return dict(s1, status="13-3 uses the stage-1 dft verdict: E2's recomputed dft REP and dft·S / dft·SH REP verdicts "
                           "equal results/E1_report.json", equal=True)


# ================================================================ coverage and same-TS classes (E2-9, E2-10)
def _tri(v):
    s = str(v).strip().lower()
    return True if s == "true" else (False if s == "false" else None)


def read_c_compare():
    """D1-G2 C_compare.csv -> {status: {rxn_id: g2_status}, same: {column: {rxn_id: True / False / None}}}."""
    Cc = pd.read_csv(C_COMPARE, usecols=["rxn_id", "g2_status"] + list(SAME_TS_COLS), dtype=str)
    rx = Cc.rxn_id.astype(int).tolist()
    if len(set(rx)) != len(rx):
        L.stop(f"{C_COMPARE}: duplicated rxn_id")
    return dict(status=dict(zip(rx, Cc.g2_status.fillna("").astype(str).tolist())),
                same={c: dict(zip(rx, [_tri(v) for v in Cc[c].tolist()])) for c in SAME_TS_COLS})


def pair_classes(S, cc):
    """supplement 5: per pair (index = S_folds pair_id) and judgment column: same (both ends' reactions = the same TS),
    different (neither), mixed (one), unclassified (an end without a judgment)."""
    a, b = S.a_rxn.astype(int).tolist(), S.b_rxn.astype(int).tolist()
    out = {}
    for col in SAME_TS_COLS:
        j = cc["same"][col]
        cls = []
        for x_, y_ in zip(a, b):
            x, y = j.get(x_), j.get(y_)
            cls.append("unclassified" if x is None or y is None else
                       ("same" if (x and y) else ("different" if not (x or y) else "mixed")))
        out[col] = cls
    return pd.DataFrame(out, index=S.index)


def test_lists_all(IN, rep_p):
    """the unrestricted T1 test lists of the fold copies ($LAD/inputs/fu2_splits, sha256 = P_report folds) -> {pair: k}."""
    files = (rep_p.get("folds") or {}).get("files") or {}
    out = {}
    for k in range(NF):
        p = IN.SPLITS_COPY / f"T1__f{k}.json"
        want = files.get(str(p))
        if not p.is_file() or want is None or L.sha256(p) != want:
            L.stop(f"{p}: missing or sha256 != results/P_report.json folds")
        for q in map(str, json.loads(p.read_text())["test"]):
            if q in out:
                L.stop(f"{q} is a test pair of folds {out[q]} and {k}")
            out[q] = k
    return out


def coverage_e2(S, sets, test_of, g2tab, cc, set_names=E2_SETS):
    """E2_coverage (spec 12; COV_RULE): P_main test pairs and the share in each set, by fold / position / dipole; P_g2
    missing reasons -> (coverage, frame of the P_g2 P_main test pairs)."""
    B = S[S.in_P_main.to_numpy() & S.index.isin(list(test_of))]
    fold = np.array([test_of[p] for p in B.index], dtype=int)
    para = (B.pos == "p").to_numpy()
    pos = B.pos.astype(str).tolist()
    dip = B.dipole.astype(str).tolist()

    def frac(a, b):
        return float(a / b) if b else None

    def part(m, sel):
        sel = np.asarray(sel, dtype=bool)
        return dict(n=int((m & sel).sum()), of=int(sel.sum()), frac=frac(int((m & sel).sum()), int(sel.sum())))
    cov = dict(rule=COV_RULE, n_P_main_test=int(len(B)), n_P_main_test_para=int(para.sum()), sets={})
    for n in set_names:
        m = np.asarray(B.index.isin(list(sets[n])), dtype=bool)
        cov["sets"][n] = dict(
            n=int(m.sum()), frac=frac(int(m.sum()), len(B)), n_para=int((m & para).sum()),
            frac_para=frac(int((m & para).sum()), int(para.sum())), n_not_predictable=int((~m).sum()),
            n_not_predictable_para=int((~m & para).sum()),
            by_fold={str(k): dict(n=int((m & (fold == k)).sum()), of=int((fold == k).sum())) for k in range(NF)},
            by_pos={(p or "(none)"): part(m, [x == p for x in pos]) for p in sorted(set(pos))},
            by_dipole={d: dict(all=part(m, [x == d for x in dip]),
                               para=part(m, np.asarray([x == d for x in dip], dtype=bool) & para))
                       for d in sorted(set(dip))})
    st = cc["status"]
    tt = {int(r): (bool(a), bool(b), bool(h)) for r, a, b, h in
          zip(g2tab.rxn_id, g2tab.has_tok, g2tab.has_x182, g2tab.has_h)}

    def cat(r):                                          # = prereg.py coverage_same_ts (PS-8)
        r = int(r)
        if st.get(r, "") != "ok":
            return "g2_not_ok"
        t = tt.get(r)
        if t is None:
            return "not_target"
        if not (t[0] and t[1]):
            return "no_tokens_or_182"
        return "ok" if t[2] else "no_h"
    mg = np.asarray(B.index.isin(list(sets["P_g2"])), dtype=bool)
    O = B[~mg]
    reasons = []
    for a_, b_ in zip(O.a_rxn, O.b_rxn):
        ca, cb = cat(a_), cat(b_)
        reasons.append(next((x for x in DROP_CATS[:-1] if x in (ca, cb)), "both_usable"))
    op = (O.pos == "p").tolist()
    cov["P_g2_missing_by_reason"] = {x: sum(1 for v in reasons if v == x) for x in DROP_CATS}
    cov["P_g2_missing_by_reason_para"] = {x: sum(1 for v, q in zip(reasons, op) if v == x and q) for x in DROP_CATS}
    cov["P_g2_missing_para"] = int(sum(op))
    return cov, B[mg]


def same_ts_counts(G, cls):
    """class counts (all, para) of the P_g2 P_main test pairs per judgment column (= supplement 5 table)."""
    gp = (G.pos == "p").tolist()
    out = {}
    for col in SAME_TS_COLS:
        c = cls.loc[G.index, col].tolist()
        out[col] = dict(all={k: sum(1 for v in c if v == k) for k in SAME_TS_CLASSES},
                        para={k: sum(1 for v, q in zip(c, gp) if v == k and q) for k in SAME_TS_CLASSES})
    return out


def coverage_vs_prereg(cov, counts, s):
    """E2-9 / E2-10: recomputed coverage and class counts vs the supplement json (structural, fixed before training)."""
    pc, ps = s.get("coverage") or {}, s.get("same_ts_split") or {}
    bad = []
    for k in ("n_P_main_test", "n_P_main_test_para", "P_g2_missing_by_reason", "P_g2_missing_para"):
        if _clean(cov.get(k)) != pc.get(k):
            bad.append(f"coverage {k}: {cov.get(k)} != supplement {pc.get(k)}")
    if set(cov["sets"]) != set(pc.get("sets") or {}):
        bad.append(f"coverage sets {sorted(cov['sets'])} != supplement {sorted(pc.get('sets') or {})}")
    for n, rec in (pc.get("sets") or {}).items():
        mine = cov["sets"].get(n) or {}
        for k in ("n", "n_para", "by_fold"):
            if _clean(mine.get(k)) != rec.get(k):
                bad.append(f"coverage {n} {k}: {mine.get(k)} != supplement {rec.get(k)}")
    cl = ps.get("classes") or {}
    for col in SAME_TS_COLS:
        if _clean(counts.get(col)) != cl.get(col):
            bad.append(f"same-TS classes {col}: {counts.get(col)} != supplement {cl.get(col)}")
    c0 = counts[SAME_TS_COLS[0]]
    if (ps.get("n_pairs"), ps.get("n_pairs_para")) != (sum(c0["all"].values()), sum(c0["para"].values())):
        bad.append(f"same-TS population {sum(c0['all'].values())} / {sum(c0['para'].values())} != supplement "
                   f"{ps.get('n_pairs')} / {ps.get('n_pairs_para')}")
    return bad


def structural_e2(IN, rep_p, S, sets, supp=None):
    """coverage + same-TS pair classes from the Phase P / D1-G2 files only (no prediction is read) -> (doc, classes,
    problems); with supp (json 'supplement') the counts are compared with the pre-registered ones."""
    test_of = test_lists_all(IN, rep_p)
    g2tab = IN.read_table("g2", rep_p)
    cc = read_c_compare()
    cls = pair_classes(S, cc)
    cov, G = coverage_e2(S, sets, test_of, g2tab, cc)
    counts = same_ts_counts(G, cls)
    bad = coverage_vs_prereg(cov, counts, supp) if supp is not None else None
    doc = dict(
        definition=COV_RULE,
        test_lists=dict(dir=str(IN.SPLITS_COPY), n_pairs=len(test_of), check="sha256 = results/P_report.json folds"),
        coverage=cov,
        same_ts_classes=dict(rule=SAME_TS_RULE, source=str(C_COMPARE), source_sha256=L.sha256(C_COMPARE),
                             columns=list(SAME_TS_COLS), primary=SAME_TS_COLS[0],
                             population="P_g2 pairs among the P_main test pairs", n_pairs=int(len(G)),
                             n_pairs_para=int((G.pos == "p").sum()), counts=counts),
        prereg_crosscheck=(dict(status="not compared (synthetic sets)") if bad is None else
                           dict(status="compared with prereg_d1pair_add3.json supplement.coverage / same_ts_split",
                                ok=not bad, problems=bad)))
    return doc, cls, (bad or [])


# ================================================================ E2 gates, compare, 13-3, same-TS split
def common_lists(ref, common, rungs, name=COMMON2):
    """spec 16 E: the rungs' test-pair lists (fold, pair_id) restricted to the common set are identical, else STOP."""
    rc = {r: {(k, p) for k, p in ref[r] if p in common} for r in rungs}
    if len({frozenset(v) for v in rc.values()}) != 1:
        cnt = {r: len(v) for r, v in rc.items()}
        L.stop(f"spec 16 E: the {name} test-pair lists differ between rungs ({cnt})")
    return rc[rungs[0]]


def gate_pairs_e2(EV, fr, ref, Y, common, rungs):
    """E2-5: per rung every model / arm has exactly the rung's test pairs (FU2 pair_sets_gate); on P_012 one list."""
    out = {r: EV.pair_sets_gate({f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS_E1}, ref[r], Y)
           for r in rungs}
    rc = common_lists(ref, common, rungs)
    frames = {f"{r}|{a}|{m}|{COMMON2}": f[f.pair_id.isin(common)] for (r, a, m), f in fr.items() if r in rungs}
    out[COMMON2] = EV.pair_sets_gate(frames, rc, Y)
    return out


def _src(rep, r, a, m, t):
    return rep[(r, a, t)] if m == "REP" else ("0" if m == "null" else m)


def _cmp(EV, fr, rep, rows, comp, kb, kx, ps, psn, sets, arm_label=None):
    """one 13-4 comparison with FU2 compare (E1-9 sign; = compare_e1's add), rows appended."""
    for evn, pairs in sets.items():
        R = EV.compare(fr[kx], fr[kb], pairs, targets=TARGETS, n_boot=N_BOOT, seed=SEED)
        for x in R.itertuples(index=False):
            rows.append(dict(comparison=comp, rung_b=kb[0], arm_b=arm_label or kb[1], model_b=kb[2],
                             source_b=_src(rep, kb[0], kb[1], kb[2], x.target), rung_x=kx[0], arm_x=arm_label or kx[1],
                             model_x=kx[2], source_x=_src(rep, kx[0], kx[1], kx[2], x.target), pair_set=ps,
                             pair_set_name=psn, eval=evn, target=x.target, n_pairs=int(x.n_pairs),
                             n_cores=int(x.n_cores), mae_x=x.mae_x, mae_b=x.mae_b, d=x.d, ci_lo=x.ci_lo,
                             ci_hi=x.ci_hi, p=x.p, folds_b_lower=x.folds_b_lower, n_folds=x.n_folds))


def compare_e2(EV, fr, S, ref, common, rep, rungs):
    """E2_compare (spec 13-4, E2-8): SH vs S, rung vs rung on P_012, REP vs LINE; FU2 compare, d = MAE(x) - MAE(b)."""
    rows = []

    def evalsets(rf):
        return {"P_main": EV.eval_pairs(S, rf), "P_main_para": EV.eval_pairs(S, rf, para=True)}
    own = {r: evalsets(ref[r]) for r in rungs}
    for r in rungs:
        for m in CMP_MODELS:
            _cmp(EV, fr, rep, rows, "SH vs S (what HF adds)", (r, "SH", m), (r, "S", m), "own", f"P_{r}", own[r])
    com = evalsets(common_lists(ref, common, rungs))
    for b, x, label in RUNG_PAIRS2:
        for a in ARMS:
            for m in CMP_MODELS:
                _cmp(EV, fr, rep, rows, label, (b, a, m), (x, a, m), COMMON2, COMMON2, com)
        _cmp(EV, fr, rep, rows, label, (b, "S", "LINE"), (x, "S", "LINE"), COMMON2, COMMON2, com, arm_label="-")
    for r in rungs:
        for a in ARMS:
            _cmp(EV, fr, rep, rows, "REP vs LINE (what the model adds on the calibration line)", (r, a, "REP"),
                 (r, a, "LINE"), "own", f"P_{r}", own[r])
    out = pd.DataFrame(rows)
    out.insert(1, "sign", "d = MAE(x) - MAE(b); d > 0: b better (FU2 D5: b = the first named)")
    return out


def interpret_13_3(verdict, model="REP", key="passed"):
    """spec 13-3 (= ADD3 5-3): the first matching row from the top on the 13-2 verdicts (key) of `model` for g2·S, g2·SH,
    dft·S, dft·SH; g1 is not part of the table."""
    g2 = [a for a in ARMS if verdict[f"g2|{a}|{model}"][key]]
    dft = [a for a in ARMS if verdict[f"dft|{a}|{model}"][key]]
    cond = {"a": bool(g2), "b": (not g2) and bool(dft), "c": not dft}
    k, c, s = next(x for x in RULES_13_3 if cond[x[0]])
    return dict(model=model, verdict_key=key, rule=k, condition_ko=c, conclusion_ko=s, conditions=cond,
                passed_arms={"g2": g2, "dft": dft})


def _goal_detail(MI, rates, verdict, r, a, m="REP"):
    v = verdict[f"{r}|{a}|{m}"]
    out = dict(rung=r, arm=a, model=m, pair_set_name=f"P_{r}", passed=v["passed"], surely_passed=v["surely_passed"])
    for t in GOAL_TARGETS:
        pp = MI[(r, a, m, "own", t, "P_main", "p")]
        br = rates[f"{r}|{a}|{m}|own"]["all"][t]
        out[t] = dict(source=pp["source"], n_para=pp["n"], para_mae=pp["mae"], para_mae_ci_lo=pp["mae_ci_lo"],
                      para_mae_ci_hi=pp["mae_ci_hi"], sens=br["sens"], n_bent_dft=br["n_bent_dft"], fpr=br["fpr"],
                      n_straight_dft=br["n_straight_dft"])
    return out


def interpretation_e2(verdict, MI, rates, concl1, verb):
    """E2_interpretation (spec 13-3, E2-7)."""
    main = interpret_13_3(verdict)
    return dict(
        rule_source="spec 13-3 = ADD3 5-3: the first matching row from the top; stage 2 applies the whole table with the "
                    "stage-1 dft verdict (supplement 7): the dft verdict used here is recomputed on the same dft units "
                    "and must equal results/E1_report.json (E2-14, 'stage1_dft_e1')",
        rule=main["rule"], condition_ko=main["condition_ko"], conclusion_ko=main["conclusion_ko"],
        passed_arms=main["passed_arms"],
        table=[dict(rule=k, condition_ko=c, conclusion_ko=s, condition_met=main["conditions"][k])
               for k, c, s in RULES_13_3],
        judged_on="REP (spec 13-1, validation MAE only); spec 13-2 on the own set (P_g2, P_dft): oi and cpcm para MAE <= "
                  "0.13, sensitivity >= 0.7, FPR <= 0.1, point estimates; NaN sensitivity / FPR = not met (E1-8)",
        rows=[_goal_detail(MI, rates, verdict, r, a) for r in ("g2", "dft") for a in ARMS],
        g1_explanatory=dict(note=G1_NOTE, rows=[_goal_detail(MI, rates, verdict, "g1", a) for a in ARMS]),
        secondary_ens4=dict(note="ENS4 (spec 13-1 / 13-2 secondary): the same table on the ENS4 verdicts; not the "
                                 "conclusion", **interpret_13_3(verdict, "ENS4")),
        descriptive_surely=dict(note="the same table on REP 'surely passed' (para MAE CI upper <= 0.13), descriptive",
                                **interpret_13_3(verdict, "REP", "surely_passed")),
        stage1_dft=dict(rule=concl1.get("rule"), conclusion_ko=concl1.get("conclusion_ko"),
                        passed_arms=concl1.get("passed_arms")),
        verbatim_rows=verb)


def goal_row_13_3(interp):
    pa = interp["passed_arms"]
    return dict(level="13-3", rung="g2+dft", arm=f"g2: {'+'.join(pa['g2']) or '-'}; dft: {'+'.join(pa['dft']) or '-'}",
                model="REP", role="primary", target="+".join(GOAL_TARGETS), judged=True, pair_set_name="P_g2 / P_dft",
                passed=None, surely_passed=None, conclusion=f"13-3 {interp['rule']}: {interp['conclusion_ko']}")


def same_ts_split(EV, A, cls, rep, MI, primary, rung="g2"):
    """E2_same_ts (spec 12, SAME_TS_RULE; descriptive): the rung's P_main test pairs split by pair class per judgment
    column; per class and 'all': n, cores, MAE, RMSE (9 targets, pos all / p). 'all' must equal E2_metrics."""
    E9, rows, bad = EV.E9, [], []
    for a in ARMS:
        for m in SPLIT_MODELS:
            free = m in ARM_FREE
            d = A[(rung, ARMS[0] if free else a, m, "own")]
            d = d[(EV.set_mask(d, "P_main") & d.target.isin(PRIMARY9)).to_numpy()]
            miss = sorted(set(d.pair_id) - set(cls.index))
            if miss:
                L.stop(f"same-TS split: {len(miss)} pairs without a class (e.g. {miss[:3]})")
            e_all = (d.yhat - d.y).to_numpy(float)
            cores = d.core.to_numpy()
            pa = (d.pos == "p").to_numpy()
            tm = {t: (d.target == t).to_numpy() for t in PRIMARY9}
            for col in SAME_TS_COLS:
                c = cls.loc[d.pair_id.to_numpy(), col].tolist()
                cm = {k: np.fromiter((v == k for v in c), dtype=bool, count=len(c)) for k in SAME_TS_CLASSES}
                for t in PRIMARY9:
                    for pos in ("all", "p"):
                        mp = (tm[t] & pa) if pos == "p" else tm[t]
                        n_sum = 0
                        for k in ("all",) + SAME_TS_CLASSES:
                            mk = mp if k == "all" else (mp & cm[k])
                            n_sum += int(mk.sum()) if k != "all" else 0
                            e = e_all[mk]
                            rows.append(dict(split_column=col, role="primary" if col == primary else "second",
                                             ts_class=k, rung=rung, arm=a, model=m, source=_src(rep, rung, a, m, t),
                                             arm_independent=free, target=t, set="P_main", pos=pos,
                                             n_pairs=int(mk.sum()), n_cores=int(len(pd.unique(cores[mk]))),
                                             mae=E9.mae(e), rmse=E9.rmse(e)))
                        ref_mae = MI[(rung, a, m, "own", t, "P_main", pos)]["mae"]
                        if n_sum != int(mp.sum()) or not _close(E9.mae(e_all[mp]), ref_mae):
                            bad.append(f"{col} {a} {m} {t} {pos}: classes {n_sum} / all {int(mp.sum())}, MAE "
                                       f"{E9.mae(e_all[mp])} vs E2_metrics {ref_mae}")
    if bad:
        L.stop("same-TS split inconsistent with E2_metrics:\n  " + "\n  ".join(bad[:8]))
    return pd.DataFrame(rows)


# ================================================================ figure figL4 (spec 14-1)
def figL4(plt, DF, C, EV, A, rep, rungs, figdir, ps=COMMON2, arm="SH"):
    """rev 9 fig3 families (FU2 fig3_families order; the first 3 with a para series of >= 12 points incl. H for oi and
    cpcm on ps in every rung) x oi / cpcm: DFT truth vs REP (arm SH) of every rung; y = -dd / 1.3642, H at the origin."""
    from matplotlib.lines import Line2D
    E9, HE = EV.E9, EV.HE

    def para(r, f, t):
        d = A[(r, arm, "REP", ps)]
        q = d[(EV.set_mask(d, "P_main") & (d.pos == "p") & d.sigma.notna() & (d.fam == f) & (d.target == t)).to_numpy()]
        return q.sort_values("pair_id", kind="mergesort")

    def ser(q, col):
        return HE.family_series(E9.hammett_frame(q, col), q.fam.iloc[0], col, C.MIN_X_FAM) if len(q) else None
    fams = [f for f in EV.fig3_families()
            if all(ser(para(r, f, t), "y") is not None for r in rungs for t in GOAL_TARGETS)][:3]
    cols = ["fam", "target", "pair_set", "arm", "X", "sigma_p", "y_dft"] + [f"y_rep_{r}" for r in rungs] + \
        [f"rep_{r}" for r in rungs]
    rows = []
    fig, axs = plt.subplots(len(GOAL_TARGETS), max(1, len(fams)), figsize=(C.FIG_W, C.FIG_W * 0.5),
                            constrained_layout=True, squeeze=False)
    for j, f in enumerate(fams):
        for i, t in enumerate(GOAL_TARGETS):
            ax = axs[i, j]
            x, y, nm = ser(para(rungs[0], f, t), "y")
            pred = {}
            for r in rungs:
                q = para(r, f, t)
                sy, sp = ser(q, "y"), ser(q, "pred")
                if sy is None or sp is None or sy[2] != nm or sp[2] != nm or \
                        not np.allclose(sy[1], y, rtol=0.0, atol=1e-9):
                    L.stop(f"figL4: {f} / {t}: the para points or the DFT values differ between rungs on {ps}")
                pred[r] = sp[1]
            o = np.argsort(x, kind="mergesort")
            for r in rungs:
                col = DF.SLOT[RUNG_SLOT[r]]
                ax.plot(x[o], pred[r][o], color=col, ls=RUNG_LS[r], lw=1.0, marker=RUNG_MARK[r], ms=3.0, mfc=col,
                        mec="white", mew=0.4, zorder=2)
            ax.scatter(x, y, s=12, color=DF.INK, linewidths=0, zorder=4)
            ax.text(0.02, 0.98, "REP " + ", ".join(f"{r} {rep[(r, arm, t)]}" for r in rungs), transform=ax.transAxes,
                    ha="left", va="top", fontsize=C.FONT_PT - 2, color=DF.INK2)
            if i == 0:
                ax.set_title(f, fontsize=C.FONT_PT - 1)
            if j == 0:
                ax.set_ylabel(f"{DF.TLAB[t]}: −ΔΔ/1.3642")
            if i == len(GOAL_TARGETS) - 1:
                ax.set_xlabel("σp")
            for k, (n, a_, b_) in enumerate(zip(nm, x, y)):
                rows.append(dict(fam=f, target=t, pair_set=ps, arm=arm, X=n, sigma_p=float(a_), y_dft=float(b_),
                                 **{f"y_rep_{r}": float(pred[r][k]) for r in rungs},
                                 **{f"rep_{r}": rep[(r, arm, t)] for r in rungs}))
    if not fams:
        for ax in axs.ravel():
            ax.set_axis_off()
        axs[0, 0].text(0.5, 0.5, f"no rev 9 fig3 family with a para series (>= {C.MIN_X_FAM} points incl. H) for oi "
                                 f"and cpcm on {ps}", ha="center", va="center", transform=axs[0, 0].transAxes,
                       color=DF.INK2)
    hs = [Line2D([], [], color=DF.INK, marker="o", ls="none", ms=3.5, label="DFT (truth)")]
    hs += [Line2D([], [], color=DF.SLOT[RUNG_SLOT[r]], ls=RUNG_LS[r], lw=1.0, marker=RUNG_MARK[r], ms=3.0,
                  label=f"REP {r}·{arm}") for r in rungs]
    axs[0, -1].legend(handles=hs, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{ps} test pairs (common to {', '.join(rungs)}), P_main para pairs; y = −ΔΔ/1.3642 (log k units), H "
                 f"at the origin", fontsize=C.FONT_PT)
    out = _save(fig, figdir, "figL4_examples", pd.DataFrame(rows, columns=cols))
    plt.close(fig)
    return out, fams


# ================================================================ E1 cross-check (E2-12, record only)
def e1_crosscheck(Mt, verdict, rep):
    pm, pr = L.RES / "E1_metrics.csv", L.RES / "E1_report.json"
    if not (pm.is_file() and pr.is_file()):
        return dict(status="no E1 output in results/: not compared")
    X = pd.read_csv(pm, dtype={"pos": str, "set": str, "pair_set": str, "model": str, "source": str},
                    keep_default_na=False, na_values=[""])           # model 'null' must stay a string
    X = X[X.pair_set == "own"]
    mine = Mt[(Mt.pair_set == "own") & Mt.rung.isin(list(STAGE_RUNGS[1]))]
    J = mine.merge(X, on=MET_KEY, how="inner", suffixes=("", "_e1"))
    a, b = J.mae.to_numpy(float), J.mae_e1.to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    r1 = json.loads(pr.read_text())
    v1, rp1 = r1.get("verdict") or {}, r1.get("representatives") or {}
    vd = sorted(k for k, v in v1.items() if k in verdict and (bool(v.get("passed")), bool(v.get("surely_passed")))
                != (bool(verdict[k]["passed"]), bool(verdict[k]["surely_passed"])))
    rd = sorted(f"{r}|{a_}|{t}" for (r, a_, t), m in rep.items()
                if r in rp1 and (((rp1.get(r) or {}).get(a_) or {}).get(t) or {}).get("model") != m)
    return dict(status="compared (record only)", e1_metrics=str(pm), e1_metrics_sha256=L.sha256(pm),
                e1_report_sha256=L.sha256(pr), e1_predictions_sha256=(r1.get("l_report") or {}).get("predictions_sha256"),
                n_rows_e2=int(len(mine)), n_rows_compared=int(len(J)),
                max_abs_diff_mae=float(np.max(np.abs(a[both] - b[both]))) if both.any() else None,
                n_nan_pattern_differs=int((np.isfinite(a) != np.isfinite(b)).sum()), verdicts_differ=vd,
                representatives_differ=rd)


# ================================================================ the E2 pipeline
def run_e2(EV, ctx, P, rep, means, out, cov_doc, cls, verb, figures=True, stage1=None):
    """gates -> frames -> metrics -> bends -> 13-2 -> (stage-1 dft check) -> 13-3 -> 13-4 -> same-TS split -> files under
    `out` (+ figures). stage1 = stage1_dft() record (e2; E2-14), None in e2-selftest (no E1 of the synthetic run)."""
    t0 = time.time()
    rungs, S, Y, sets = ctx["rungs"], ctx["S"], ctx["Y"], ctx["sets"]
    common = sets[COMMON2]
    ref = {r: ref_set(ctx["tests"][r], r) for r in rungs}
    evc = {}
    for r in rungs:                                      # E2-9: evaluated P_main test pairs = the coverage count
        q = S.loc[[p for _, p in ref[r]]]
        n, n_p = int(q.in_P_main.sum()), int((q.in_P_main & (q.pos == "p")).sum())
        c = cov_doc["coverage"]["sets"][f"P_{r}"]
        evc[r] = dict(n_P_main_test=n, n_P_main_test_para=n_p, coverage_n=c["n"], coverage_n_para=c["n_para"])
        if (n, n_p) != (c["n"], c["n_para"]):
            L.stop(f"{r}: evaluated P_main test pairs {n} / para {n_p} != coverage {c['n']} / {c['n_para']} (E2-9)")
    fr = model_frames(P, rungs)
    gm = {r: EV.pair_sets_gate({f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS4}, ref[r], Y)
          for r in rungs}
    check_pair_attrs(P, S)
    lines = line_frames(ctx)
    fr = derived_frames(fr, lines, rep, rungs)
    ga = gate_pairs_e2(EV, fr, ref, Y, common, rungs)
    lchk = check_line(fr, lines, rungs)
    _log2(t0, f"gates passed: test-pair lists per rung and on {COMMON2} (spec 16 E), LINE = SH baseline (E1-5)")
    A = annotate_all(EV, fr, S, common, rungs, common_name=COMMON2)
    Mt = metrics_e1(EV, A)
    MI = met_index(Mt)
    _log2(t0, f"metrics: {len(Mt)} rows")
    Bd, rates, fams = bends_e1(EV, A, S, rep)
    _log2(t0, f"bends: {len(Bd)} rows")
    goal, verdict, concl1 = goal_e1(MI, rates, rungs)
    s1chk = (check_stage1_dft(stage1, verdict, rep) if stage1 is not None else
             dict(status="no stage-1 record (e2-selftest: synthetic predictions, no E1 of them)", equal=None))
    interp = interpretation_e2(verdict, MI, rates, concl1, verb)
    interp["stage1_dft_e1"] = s1chk
    goal = pd.concat([goal[goal.level != "13-3"], pd.DataFrame([goal_row_13_3(interp)])], ignore_index=True)
    Cmp = compare_e2(EV, fr, S, ref, common, rep, rungs)
    _log2(t0, f"compare: {len(Cmp)} rows")
    ST = same_ts_split(EV, A, cls, rep, MI, SAME_TS_COLS[0])
    _log2(t0, f"same-TS split: {len(ST)} rows")
    outputs = {}
    for name, df in (("E2_metrics.csv", Mt), ("E2_bends.csv", Bd), ("E2_goal.csv", goal), ("E2_compare.csv", Cmp),
                     ("E2_same_ts.csv", ST)):
        p = _write_csv(out / name, df)
        outputs[str(p)] = L.sha256(p)
    for name, obj in (
            ("E2_bend_rates.json", dict(
                key=f"rung|arm|model|pair_set (pair_set own = P_<rung>, {COMMON2})", rates=rates, families=fams,
                representatives=rep_json(rep, means),
                note="d1_eval.bend_rates on FU2 bends_tab rows (as E1_bend_rates.json): sens = DFT-bent families called "
                     "bent / DFT-bent, fpr = DFT-straight called bent / DFT-straight, V_sens likewise; *_ci = "
                     "Clopper-Pearson 95 %; null = zero denominator (E1-7); LINE / null arm-independent (E1-4)")),
            ("E2_interpretation.json", interp),
            ("E2_coverage.json", dict(cov_doc, evaluated=evc))):
        p = _write_json(out / name, obj)
        outputs[str(p)] = L.sha256(p)
    fams4 = None
    if figures:
        plt, DF = _plot()
        figdir = out / "figures"
        f4, fams4 = figL4(plt, DF, EV.C, EV, A, rep, rungs, figdir)
        for f in figL1(plt, DF, EV.C, MI, rungs, figdir, ps=COMMON2) + figL2(plt, DF, EV.C, MI, rungs, figdir,
                                                                           ps=COMMON2) + \
                figL3(plt, DF, EV.C, rates, rungs, figdir) + f4:
            outputs[f] = L.sha256(f)
        _log2(t0, f"figures written to {figdir} (figL4 families: {fams4})")
    gates = dict(models=gm, all=ga, n_test_pairs={r: len(ref[r]) for r in rungs},
                 n_common_test_pairs=len(common_lists(ref, common, rungs)), common_set=COMMON2,
                 n_pairs_in_sets={n: len(v) for n, v in sets.items()}, evaluated_vs_coverage=evc)
    return dict(fr=fr, Mt=Mt, MI=MI, Bd=Bd, rates=rates, goal=goal, verdict=verdict, concl1=concl1, interp=interp,
                Cmp=Cmp, ST=ST, gates=gates, line_check=lchk, line_betas={r: lines[r][1] for r in rungs},
                outputs=outputs, fams4=fams4, seconds=time.time() - t0, ref=ref, stage1=s1chk)


def cmd_e2(labels, figures=True):
    t0 = time.time()
    rungs = E2_RUNGS
    EV, info = fu2_eval()
    import inputs as IN
    verb = rules_verbatim()
    lab = label_info(labels)
    labels = Path(lab["path"])
    rep_p = IN.p_report()
    for r in rungs:
        IN.p_report(r)                                   # rung build passed and current, table = P_report's
    tables = {r: rep_p["rungs"][r]["table_sha256"] for r in rungs}
    add3 = add3_check()
    supp = stage2_prereg(rep_p, rungs, E2_SETS)
    ready = stage2_ready(rungs)
    with no_test_scores([PRED, UNITS]):                  # E1-3: REP from validation records only
        V, units = val_table(rungs, lab["sha256"], add3["md_sha256"], tables)
        cfp = code_fingerprints(sorted(units))           # E2-15 (ADD3 11 E)
        rep, means = pick_rep(V)
    _log2(t0, "REP: " + ", ".join(f"{r}·{a}·{t}={m}" for (r, a, t), m in rep.items()))
    lrep = check_l_report(units, rungs, labels)
    s1 = stage1_dft(units, rep)                          # E2-14: exit 2 without a current E1; REP = E1's, else STOP
    ctx = real_inputs(IN, rep_p, rungs, labels, set_names=E2_SETS)
    cov_doc, cls, bad = structural_e2(IN, rep_p, ctx["S"], ctx["sets"], supp["record"])
    if bad:
        L.stop("coverage / same-TS class counts differ from the pre-registered supplement (E2-9 / E2-10):\n  "
               + "\n  ".join(bad[:10]))
    _log2(t0, f"coverage P_g2 {cov_doc['coverage']['sets']['P_g2']['n']} / {cov_doc['coverage']['n_P_main_test']} "
              f"P_main test pairs (= supplement)")
    P = load_predictions(rungs)
    res = run_e2(EV, ctx, P, rep, means, L.RES, cov_doc, cls, verb, figures, stage1=s1)
    e1x = e1_crosscheck(res["Mt"], res["verdict"], rep)
    me = Path(__file__).resolve()
    ins = [PRED, L_REPORT, L_VAL, labels, ORIG_LABELS, IN.S_FOLDS, IN.P_REPORT, ADD3_MD, ADD3_JSON, SUPP_MD, SPEC_MD,
           C_COMPARE, E1_REPORT, Path(info["source"]), Path(info["e_tests"])] + \
        [Path(rep_p["sets"][n]["file"]) for n in E2_SETS] + [IN.table_path(r) for r in rungs]
    it = res["interp"]
    doc = dict(
        phase="E2", stage=2, rungs=list(rungs), command=" ".join(sys.argv[1:]), date=time.strftime("%F %T"),
        evaluate_py=dict(path=str(me), sha256=L.sha256(me), git_state=_git_state(me) or "committed, unchanged",
                         line_source_sha256=line_source_sha256()),
        fu2_evaluate=info, labels=lab, add3=add3, supplement={k: v for k, v in supp.items() if k != "record"},
        ready=ready, l_report=lrep, units=units, code_fingerprints=cfp, stage1_dft_e1=res["stage1"],
        inputs_sha256={str(p): L.sha256(p) for p in ins},
        representatives=rep_json(rep, means),
        rep_rule="spec 13-1: lowest mean over the 5 outer folds of the unit json val_kcal_mae, ties AW -> GNN -> TF -> "
                 "XGB; chosen before any prediction file was read (E1-3); train_atoms MA-6n selection asymmetry noted",
        gates=res["gates"], line_check=res["line_check"], line_betas=res["line_betas"], verdict=res["verdict"],
        conclusion_13_3=dict(rule=it["rule"], conclusion_ko=it["conclusion_ko"], passed_arms=it["passed_arms"]),
        coverage_P_g2={k: cov_doc["coverage"]["sets"]["P_g2"][k] for k in ("n", "frac", "n_para", "frac_para")},
        same_ts=dict(primary=SAME_TS_COLS[0], second=SAME_TS_COLS[1],
                     counts=cov_doc["same_ts_classes"]["counts"]),
        figL4_families=res["fams4"], e1_crosscheck=e1x, outputs=res["outputs"],
        goal_rule=dict(targets=list(GOAL_TARGETS), para_mae_max=GOAL["mae"], sens_min=GOAL["sens"], fpr_max=GOAL["fpr"],
                       set="own pair set (P_<rung>; g2: P_g2), P_main reference para pairs, point estimate; surely = CI "
                           "upper <= 0.13; REP primary, ENS4 secondary, single models descriptive",
                       bend_population="d1_eval.bend_rates 'all' (no family filter; ADD3 5-2 / PR-15)",
                       stage0_rev9=STAGE0_NOTE, stage1_rev9=f"R2_fam median (P_main, all) >= {STAGE1_R2}, descriptive"),
        env=_env(), decisions=DECISIONS2, decisions_e1_reused=DECISIONS, seconds=time.time() - t0)
    _write_json(L.RES / "E2_report.json", doc)
    for k, v in res["verdict"].items():
        if v["role"] != "descriptive":
            print(f"[E2] 13-2 {k}: passed {v['passed']} surely {v['surely_passed']}", flush=True)
    print(f"[E2] 13-3 {it['rule']}: {it['conclusion_ko']}", flush=True)
    print(f"[E2] coverage P_g2: {doc['coverage_P_g2']}", flush=True)
    return 0


# ================================================================ e2-selftest (E2-13)
def selftest_inputs_e2():
    """the E1-13 synthetic inputs + a synthetic g2 rung: P_g2 = P_g1 minus every 13th sorted pair from the 4th, P_012 =
    P_01 & P_g2; dH = 0.6 y + N(0, 0.8); noisy predictions (no exact model); validation table: g2·SH -> XGB, g2·S oi ->
    TF (exact tie TF = XGB), other g2·S targets -> AW."""
    import fu2_common as F2
    ctx, P, V, expect, sinfo = selftest_inputs()
    rng = np.random.default_rng(L.SEED_M + 2)
    S, Y = ctx["S"], ctx["Y"]
    g1 = sorted(ctx["sets"]["P_g1"])
    p_g2 = set(g1) - set(g1[3::13])
    sets = dict(ctx["sets"], P_g2=p_g2, P_012=ctx["sets"]["P_01"] & p_g2)
    folds = {k: json.loads((F2.SPLITS / f"T1__f{k}.json").read_text()) for k in range(NF)}
    te2 = {k: [str(p) for p in folds[k]["test"] if str(p) in p_g2] for k in range(NF)}
    tr2 = {k: [str(p) for p in folds[k]["train"] if str(p) in p_g2] for k in range(NF)}
    u = sorted(p_g2)
    d = 0.6 * Y.loc[u, DD].to_numpy(float) + rng.normal(0.0, 0.8, (len(u), len(TARGETS)))
    d[:, TARGETS.index("c_ghost")] = 0.0
    dH = pd.DataFrame(d, index=pd.Index(u, name="pair_id"), columns=TARGETS)
    noise = {"AW": 0.3, "GNN": 0.35, "TF": 0.4, "XGB": 0.45}
    parts = []
    for a in ARMS:
        for m in MODELS4:
            for k in range(NF):
                te, tr = te2[k], tr2[k]
                if not te:
                    continue
                yv = Y.loc[te, DD].to_numpy(float)
                if a == "SH":
                    beta = line_beta(dH.loc[tr, TARGETS].to_numpy(float), Y.loc[tr, DD].to_numpy(float))
                    base = line_predict(beta, dH.loc[te, TARGETS].to_numpy(float))
                else:
                    beta, base = np.zeros(len(TARGETS)), np.zeros_like(yv)
                yh = yv + rng.normal(0.0, noise[m], yv.shape)
                att = S.loc[te, ["ref_kind", "pos"]]
                for j, t in enumerate(TARGETS):
                    parts.append(pd.DataFrame(dict(
                        pair_id=te, ref_kind=att.ref_kind.to_numpy(), pos=att.pos.to_numpy(), fold=int(k), rung="g2",
                        arm=a, model=m, target=t, y=yv[:, j], yhat=yh[:, j], base=base[:, j], beta=float(beta[j]),
                        yhat_seed_sd=0.0)))
    P2 = pd.concat([P] + parts, ignore_index=True)[PRED_COLS]
    vrows, expect = [], dict(expect)
    for a in ARMS:
        for t in TARGETS:
            for m in MODELS4:
                for k in range(NF):
                    v = 1.0 + 0.1 * MODEL_ORDER[m] + float(rng.uniform(0.0, 0.01))
                    if a == "SH" and m == "XGB":
                        v = 0.3
                    if (a, t) == ("S", "oi") and m in ("TF", "XGB"):
                        v = 0.6
                    vrows.append(dict(rung="g2", arm=a, model=m, fold=k, target=t, val_kcal_mae=v))
            expect[("g2", a, t)] = "XGB" if a == "SH" else ("TF" if t == "oi" else "AW")
    V2 = pd.concat([V, pd.DataFrame(vrows)], ignore_index=True)
    ctx2 = dict(ctx, rungs=E2_RUNGS, sets=sets, tests=dict(ctx["tests"], g2=te2), trains=dict(ctx["trains"], g2=tr2),
                dH=dict(ctx["dH"], g2=dH))
    return ctx2, P2, V2, expect, dict(sinfo, g2="P_g2 = P_g1 minus every 13th sorted pair from the 4th; dH = 0.6 y + "
                                                "N(0, 0.8); noise AW 0.3, GNN 0.35, TF 0.4, XGB 0.45")


def cmd_selftest_e2(figures=True):
    t0 = time.time()
    EV, info = fu2_eval(allow_unmerged=True)
    import inputs as IN
    checks = {}
    verb = rules_verbatim()
    checks["rules_13_3_verbatim"] = all(all(v.values()) for v in verb.values())
    ctx, P, V, expect, sinfo = selftest_inputs_e2()
    with no_test_scores([PRED, UNITS]):
        rep, means = pick_rep(V)
    checks["rep_rule_and_ties"] = bool(set(rep) == set(expect) and all(rep[k] == v for k, v in expect.items()))

    def vv(g2s, g2sh, ds, dsh):
        keys = (("g2", "S"), ("g2", "SH"), ("dft", "S"), ("dft", "SH"))
        return {f"{r}|{a}|REP": dict(passed=p, surely_passed=p) for (r, a), p in zip(keys, (g2s, g2sh, ds, dsh))}
    cases = {(True, False, False, False): "a", (False, True, True, True): "a", (False, False, True, False): "b",
             (False, False, False, True): "b", (False, False, False, False): "c"}
    checks["rules_13_3_first_match"] = all(interpret_13_3(vv(*k))["rule"] == v for k, v in cases.items())
    rep_p = IN.p_report()                                # the REAL Phase P sets: structural numbers vs the supplement
    real_sets = {n: set(IN.read_set(n, rep_p).pair_id.astype(str)) for n in E2_SETS}
    supp = json.loads(ADD3_JSON.read_text()).get("supplement") or {}
    cov_real, cls, bad_real = structural_e2(IN, rep_p, ctx["S"], real_sets, supp)
    checks["coverage_and_classes_equal_prereg"] = not bad_real
    cov_syn, _, _ = structural_e2(IN, rep_p, ctx["S"], ctx["sets"], None)
    ref0 = {r: ref_set(ctx["tests"][r], r) for r in E2_RUNGS}
    common = ctx["sets"][COMMON2]
    drop = sorted(x for x in ref0["g2"] if x[1] in common)[0]
    bad_ref = dict(ref0, g2=ref0["g2"] - {drop})
    checks["common_gate_detects_a_missing_pair"] = _expect_stop(lambda: common_lists(bad_ref, common, E2_RUNGS))
    res = run_e2(EV, ctx, P, rep, means, SELFTEST2, cov_syn, cls, verb, figures)
    S, Y, MI, verdict = ctx["S"], ctx["Y"], res["MI"], res["verdict"]
    vd = verdict.get("dft|SH|REP") or {}
    checks["goal_passes_for_the_exact_model"] = bool(vd.get("passed") and vd.get("surely_passed"))
    checks["g2_fails_with_noise"] = not any(verdict[f"g2|{a}|REP"]["passed"] for a in ARMS)
    checks["interpretation_rule_b"] = res["interp"]["rule"] == "b" and res["interp"]["conclusion_ko"] == RULES_13_3[1][2]
    Cm = res["Cmp"]
    c = Cm[(Cm.comparison == RUNG_PAIRS2[1][2]) & (Cm.arm_b == "SH") & (Cm.model_b == "AW") & (Cm["eval"] == "P_main")
           & (Cm.target == "oi")]
    checks["rung_compare_sign_P012"] = bool(len(c) == 1 and c.d.iloc[0] > 0 and c.mae_b.iloc[0] == 0.0
                                            and c.pair_set.iloc[0] == COMMON2)
    pids = [p for _, p in res["ref"]["g2"] if S.at[p, "in_P_main"] and S.at[p, "pos"] == "p"]
    want = float(np.mean(np.abs(Y.loc[pids, "dd_cpcm"].to_numpy(float))))
    got = MI[("g2", "SH", "null", "own", "cpcm", "P_main", "p")]["mae"]
    checks["null_para_mae_is_mean_abs_y_g2"] = bool(abs(got - want) <= 1e-12)
    n_rows = len(ARMS) * len(SPLIT_MODELS) * len(SAME_TS_COLS) * len(PRIMARY9) * 2 * (1 + len(SAME_TS_CLASSES))
    checks["same_ts_rows_complete"] = bool(len(res["ST"]) == n_rows)
    checks["line_equals_sh_baseline"] = bool(all(v["sh_base_max_abs"] <= 1e-12 and v["s_rows_with_baseline"] == 0
                                                 for v in res["line_check"].values()))
    g = res["goal"]
    checks["goal_rows_6_rung_arm"] = bool(len(g[(g.level == "rung_arm") & (g.model == "REP")]) == 6
                                          and (g.level == "13-3").sum() == 1)
    # E2-14 on constructed stage-1 records: equal -> no problem; a flipped dft verdict or dft REP -> problem / STOP
    s1 = dict(e1_verdict={f"dft|{a}|REP": dict(passed=bool(verdict[f"dft|{a}|REP"]["passed"]),
                                                surely_passed=bool(verdict[f"dft|{a}|REP"]["surely_passed"]))
                          for a in ARMS},
              e1_representatives={a: {t: rep[("dft", a, t)] for t in TARGETS} for a in ARMS})
    flip_v = dict(s1, e1_verdict=dict(s1["e1_verdict"], **{"dft|S|REP": dict(
        passed=not s1["e1_verdict"]["dft|S|REP"]["passed"],
        surely_passed=s1["e1_verdict"]["dft|S|REP"]["surely_passed"])}))
    other = "GNN" if rep[("dft", "SH", "oi")] != "GNN" else "AW"
    flip_r = dict(s1, e1_representatives=dict(s1["e1_representatives"],
                                              SH=dict(s1["e1_representatives"]["SH"], oi=other)))
    checks["stage1_dft_rule"] = bool(not _stage1_problems(s1, verdict, rep) and _stage1_problems(flip_v, verdict, rep)
                                     and _stage1_problems(flip_r, verdict, rep)
                                     and _stage1_problems(flip_r, rep=rep) and not _stage1_problems(flip_r, verdict))
    checks["stage1_dft_stop_on_difference"] = bool(_expect_stop(lambda: check_stage1_dft(flip_v, verdict, rep))
                                                   and check_stage1_dft(s1, verdict, rep)["equal"] is True)
    # E2-15 on constructed fingerprints, then on the REAL unit jsons present now (no false STOP before the real run)
    fkeys = (json.loads(ADD3_JSON.read_text()).get("code") or {}).get("fingerprint_keys") or {}
    ka = dict(fkeys.get("train_atoms") or {})
    checks["code_fp_rule"] = bool(ka and not _code_fp_problems({"a": ("AW", dict(ka))}, fkeys)
                                  and _code_fp_problems({"b": ("TF", {**ka, "train_atoms.py": "0" * 64})}, fkeys)
                                  and _code_fp_problems({"c": ("XGB", dict(ka))}, fkeys)
                                  and _code_fp_problems({"d": ("GNN", None)}, fkeys))
    present = [x for x in (unit_name(r, a, m, k) for r in E2_RUNGS for a in ARMS for m in MODELS4 for k in range(NF))
               if (UNITS / f"{x}.json").is_file()]
    cfp = {}
    checks["code_fp_real_units_match"] = bool(present) and not _expect_stop(
        lambda: cfp.update(code_fingerprints(present)))
    if figures:
        figs = [k for k in res["outputs"] if "/figures/" in k]
        checks["figures_written"] = bool(len(figs) == 12 and all(Path(k).is_file() for k in figs))
        checks["figL4_has_families"] = bool(res["fams4"])
    ok = all(checks.values())
    doc = dict(ok=ok, checks=checks, coverage_prereg_problems=bad_real, synthetic=sinfo,
               code_fingerprints_real_units=dict(cfp, n_present=len(present)),
               representatives=rep_json(rep, means), expected_rep={"|".join(k): v for k, v in expect.items()},
               verdict=verdict, interpretation=res["interp"], gates=res["gates"], line_check=res["line_check"],
               coverage_real=cov_real, fu2_evaluate=info, outputs=res["outputs"], figL4_families=res["fams4"],
               env=_env(), evaluate_py_sha256=L.sha256(Path(__file__).resolve()),
               line_source_sha256=line_source_sha256(), seconds=time.time() - t0,
               note="synthetic predictions: numbers are not results; checks only (E2-13)")
    _write_json(SELFTEST2 / "E2_selftest.json", doc)
    for k, v in checks.items():
        print(f"[e2-selftest] {k}: {'ok' if v else 'FAIL'}", flush=True)
    print(f"[e2-selftest] {'all checks passed' if ok else 'FAILED'} ({time.time() - t0:.0f} s) -> {SELFTEST2}",
          flush=True)
    return 0 if ok else 3


# ================================================================ Phase E3 (GEO2 docs/specs/GEO2.md 9-2, 10): rung geo
GEO2_RES = L.REPO / "analysis" / "geo2" / "results"           # the E3 write root (E3-1)
E3_SELFTEST_DIR = GEO2_RES / "e3_selftest"
GEO2_SCR = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))   # = geo2_common.GEO2 (read only here)
GEO_TABLE = GEO2_SCR / "inputs" / "geo.parquet"               # feat_geo.py p (= train_atoms.inputs_path('geo'))
P_BUILD_GEO = GEO2_RES / "P_build_geo.json"
P_GEO_REPORT = GEO2_RES / "P_geo_report.json"
P_GEO_SETS = GEO2_RES / "P_geo_sets.json"
PREREG_GEO2_MD = GEO2_RES / "PREREG_GEO2.md"
PREREG_GEO2_JSON = GEO2_RES / "prereg_geo2.json"
PRED_GEO = L.RES / "L_predictions_geo.parquet"                # train_atoms.py collect --stage 3 --with-xgb (GA-3)
L_REPORT_GEO = L.RES / "L_report_geo.json"
L_VAL_GEO = L.RES / "L_val_geo.csv"
L_XGB_REPORT_GEO = L.RES / "L_xgb_report_geo.json"            # train_xgb.py check --stage 3
E2_REPORT_F, E2_METRICS_F = L.RES / "E2_report.json", L.RES / "E2_metrics.csv"
E2_GOAL_F, E2_RATES_F = L.RES / "E2_goal.csv", L.RES / "E2_bend_rates.json"
EVAL_GEO2_REPORT = GEO2_RES / "E_report.json"                 # eval_geo2.py run (cross-check, record only)
EVAL_GEO2_SELFTEST = GEO2_RES / "selftest" / "E_report.json"  # eval_geo2.py selftest
E3_LADDER_RUNGS = ("dft", "g2")
E3_RUNGS = ("dft", "g2", "geo")
E3_LAYOUT = {"dft": ("P_023",), "g2": ("own", "P_23", "P_023"), "geo": ("own", "P_23", "P_023")}   # E3-3
E3_COMMON = {"P_23": ("g2", "geo"), "P_023": ("dft", "g2", "geo")}
E3_GOAL_RUNGS = ("g2", "geo")
GEO_SET_NAMES = ("P_geo", "P_23", "P_023", "P_g2", "P_dft")   # = feat_geo.py SET_NAMES (P_geo_sets.json keys)
REP8_ARMS = ("SH", "S")                                        # GEO2 9-1: REP8 ties SH before S, then MODELS4 order
E3_PAIRS = (("g2", "geo", "P_23", "g2 vs geo (what the refined TS changes against the G2 TS)"),
            ("dft", "geo", "P_023", "dft vs geo (what the refined TS still loses against the label TS)"))
E3_NUM = ("n", "mae", "rmse", "mae_null", "mae_rel_null", "n_cores", "mae_ci_lo", "mae_ci_hi", "n_fam", "r2_fam_median",
          "r2_fam_q1", "r2_fam_q3", "rho_err_median")
E3_TOL = 1e-12
DECISIONS3 = {
    "E3-1": "new commands e3 / e3-selftest; e1 / e2 / their selftests unchanged (no existing function edited; E3 reuses "
            "fu2_eval, label_info, add3_check, val_table, pick_rep, check_l_report, real_inputs, load_predictions, "
            "model_frames, check_pair_attrs, line_frames, derived_frames, check_line, metrics_e1, met_index, bends_e1, "
            "goal_e1, common_lists, _cmp, ref_set as they are); outputs under analysis/geo2/results (GEO2 write root; "
            "nothing in analysis/d1_ladder/results is written); no figures (GEO2 11 makes figG2_*)",
    "E3-2": "LADDER rungs dft / g2: inputs.p_report() without a rung and every Phase P read with that report (the "
            "P_build fingerprints hold the d1ladder worktree paths and the pre-GEO2 inputs.py sha256, D1-G3 F-13); the "
            "builds are checked by table sha256 = P_build_<rung>.json outputs = P_report rungs table_sha256",
    "E3-2b": "(review fix 2026-10-30) results/P_report.json 'inputs' is keyed by the absolute path of the worktree that "
             "ran inputs.py pairs (d1ladder), while real_inputs / inputs.pairs_frame look S_folds.csv up by this "
             "worktree's inputs.S_FOLDS: cmd_e3 hands them a copy of the report whose S_folds entry is re-keyed to "
             "str(inputs.S_FOLDS) when a recorded key has the same repo-relative part (_repo_rel, D1-G3 F-13) and its "
             "sha256 = the current file, else STOP; results/P_report.json, real_inputs and e1 / e2 are unchanged",
    "E3-3": "rungs dft (P_023 only), g2 (own = P_g2, P_23, P_023), geo (own = P_geo, P_23, P_023); P_geo / P_23 / P_023 "
            "from results/P_geo_sets.json (sha256 = P_geo_report outputs and per-list sha256; P_g2 / P_dft = the LADDER "
            "set files; P_23 = P_g2 & P_geo; P_023 = P_dft & P_23; P_geo = both ends usable in the geo table), else STOP",
    "E3-4": "geo: predictions = results/L_predictions_geo.parquet (rung geo only, LADDER columns); REP / REP8 from the "
            "geo unit jsons (val_table, before any prediction is read), = L_report_geo.json / L_val_geo.csv, XGB units = "
            "L_xgb_report_geo.json, every geo unit fingerprint prereg_geo2 = PREREG_GEO2.md sha256 (committed, "
            "unchanged, LADDER result snapshot unchanged), else STOP; geo LINE = line_beta on the T1 outer-train pairs "
            "of P_geo with dH from the geo table H_<t>, checked = the SH units' baseline (E1-5)",
    "E3-5": "the E2 computation per rung x arm x model x pair set (metrics_e1 incl. the para CI, bends_e1, goal_e1 13-2 on "
            "the own set: REP primary, ENS4 secondary, single models descriptive) + REP8 rows (GEO2 9-1: per rung x "
            "target the lowest 5-fold mean val_kcal_mae over 2 arms x 4 models, ties SH before S, then AW, GNN, TF, XGB) "
            "+ column bend_passed (sensitivity >= 0.7 and FPR <= 0.1): the bend part of LADDER 13-2 that GEO2 10-1 "
            "'목표 도달' takes from E3 (the MAE part is eval_geo2 goal_mae; E3 reports both)",
    "E3-6": "E3_compare (descriptive, FU2 compare): SH vs S (geo own), g2 vs geo on P_23, dft vs geo on P_023 (per arm x "
            "AW / GNN / TF / XGB / REP / ENS4 + LINE; b = the first named, d = MAE(x) - MAE(b)), REP vs LINE (geo own)",
    "E3-7": "E2 cross-check: the g2 own rows of E3_metrics = results/E2_metrics.csv (every number, 1e-12 relative), g2 "
            "bend rates = E2_bend_rates.json, g2 goal rows = E2_goal.csv, dft / g2 REP = E2_report.json (STOP in e3, a "
            "check in e3-selftest); eval_geo2 E_report.json REP8 / goal_mae REP8 compared when present (record only)",
    "E3-8": "e3-selftest: geo := a copy of g2 (predictions, validation records, P_geo = P_g2, folds, dH) -> "
            "analysis/geo2/results/e3_selftest/; checks: geo rows = g2 rows (metrics on own / P_23 / P_023, goal, REP, "
            "REP8, bend rates), g2 vs geo compare d = 0, the E2 cross-check, REP8 / goal_mae = eval_geo2 selftest",
}


def _log3(t0, msg):
    print(f"[e3 {time.time() - t0:7.0f}s] {msg}", flush=True)


def _assert_e3(p):
    q = Path(os.path.abspath(p))
    root = Path(os.path.abspath(GEO2_RES))
    if not (q == root or root in q.parents):
        L.stop(f"E3 write path {q} outside {root} (E3-1)")
    return q


def _write_e3(path, fn):
    """atomic write under analysis/geo2/results only (per-process temporary name)."""
    path = _assert_e3(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fn(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _csv3(path, df):
    return _write_e3(path, lambda t: df.to_csv(t, index=False))


def _json3(path, obj):
    text = json.dumps(_clean(obj), indent=1)
    return _write_e3(path, lambda t: Path(t).write_text(text))


def _list_sha(ids):
    """= feat_geo.py list_sha / the GEO2 interface: sha256 of '\\n'.join(sorted pair ids)."""
    import hashlib
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode()).hexdigest()


def _nanf(v):
    """float (NaN for None / ''), or None when v is not a number (then compared as text)."""
    if v is None:
        return float("nan")
    if isinstance(v, str):
        if v == "":
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _num_close(a, b, tol=E3_TOL):
    x, y = _nanf(a), _nanf(b)
    if x is None or y is None:
        return str(a) == str(b)
    if np.isnan(x) or np.isnan(y):
        return bool(np.isnan(x) and np.isnan(y))
    return abs(x - y) <= tol * max(1.0, abs(y))


def _same_obj(a, b, tol=E3_TOL, path="", out=None):
    """differences (list of str) between two json-like objects; numbers within tol, NaN = None."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: key only on one side")
            else:
                _same_obj(a[k], b[k], tol, f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _same_obj(x, y, tol, f"{path}[{i}]", out)
    elif not _num_close(a, b, tol):
        out.append(f"{path}: {a} != {b}")
    return out


# ---------------------------------------------------------------- E3 inputs and gates
def ladder_builds(IN, rep_p, rungs):
    """E3-2: LADDER rung builds by sha256 (P_build_<rung>.json all_pass, outputs = the table = P_report table_sha256)."""
    out = {}
    for r in rungs:
        bp, tp = IN.build_report_path(r), IN.table_path(r)
        br = json.loads(bp.read_text()) if bp.is_file() else {}
        sha = L.sha256(tp) if tp.is_file() else None
        want = ((rep_p.get("rungs") or {}).get(r) or {}).get("table_sha256")
        if not br.get("all_pass") or br.get("stop") or (br.get("outputs") or {}).get(str(tp)) != sha or want != sha:
            L.stop(f"{tp}: sha256 {sha} != {bp.name} outputs / results/P_report.json rungs.{r} (LADDER Phase P changed)")
        out[r] = dict(table=str(tp), table_sha256=sha, build_report=str(bp), build_report_sha256=L.sha256(bp),
                      rule=DECISIONS3["E3-2"])
    return out


def f13_inputs(IN, rep_p):
    """E3-2b: (copy of rep_p, record). The P_report 'inputs' entry of S_folds.csv as real_inputs / inputs._read_s_folds
    look it up (key str(IN.S_FOLDS)): native key -> rep_p as it is; otherwise the recorded key(s) whose repo-relative
    part equals that of IN.S_FOLDS must all carry the sha256 of the current file, and the copy gets that sha256 under
    str(IN.S_FOLDS); no such key or another sha256 = STOP. rep_p itself is not modified."""
    k0 = str(IN.S_FOLDS)
    inp = dict(rep_p.get("inputs") or {})
    cur = L.sha256(IN.S_FOLDS) if Path(IN.S_FOLDS).is_file() else None
    if k0 in inp:
        return rep_p, dict(native_key=True, key=k0, sha256=inp[k0], rule=DECISIONS3["E3-2b"])
    rel = _repo_rel(k0)
    hits = {k: v for k, v in inp.items() if rel is not None and _repo_rel(k) == rel}
    vals = sorted({str(v) for v in hits.values()})
    if not hits or cur is None or vals != [cur]:
        L.stop(f"results/P_report.json inputs: no entry for {k0} and the repo-relative entries {sorted(hits)} "
               f"(sha256 {vals}) do not equal the current file (sha256 {cur}) (E3-2b, D1-G3 F-13)")
    rep = dict(rep_p, inputs=dict(inp, **{k0: cur}))
    return rep, dict(native_key=False, key=k0, repo_relative=rel, recorded_keys=sorted(hits), sha256=cur,
                     rule=DECISIONS3["E3-2b"])


def e2_record():
    """E3-7: the LADDER E2 outputs (read only), evaluated on the current results/L_predictions.parquet."""
    for p in (E2_REPORT_F, E2_METRICS_F, E2_GOAL_F, E2_RATES_F, PRED, L_REPORT):
        if not p.is_file():
            _not_ready(f"{p} missing (LADDER E2 outputs)")
    r = json.loads(E2_REPORT_F.read_text())
    if r.get("phase") != "E2":
        L.stop(f"{E2_REPORT_F}: phase {r.get('phase')} (not an E2 report)")
    if (r.get("l_report") or {}).get("predictions_sha256") != L.sha256(PRED):
        L.stop("results/L_predictions.parquet differs from the one E2 evaluated (E2_report.json l_report)")
    return dict(report=r, sha256={str(p): L.sha256(p) for p in (E2_REPORT_F, E2_METRICS_F, E2_GOAL_F, E2_RATES_F)})


def prereg_geo2_check():
    """E3-4: PREREG_GEO2.md + prereg_geo2.json committed and unchanged, md sha256 = json, rungs ['geo'], and every
    LADDER result file of the json snapshot unchanged (GEO2 14 R·M3: an existing LADDER result changed = STOP)."""
    bad = []
    for p in (PREREG_GEO2_MD, PREREG_GEO2_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        else:
            st = _git_state(p)
            if st:
                bad.append(f"{p.name} {st}")
    if bad:
        L.stop("PREREG_GEO2 (GEO2 8-3): " + "; ".join(bad))
    rec = json.loads(PREREG_GEO2_JSON.read_text())
    md = L.sha256(PREREG_GEO2_MD)
    if rec.get("md_sha256") != md:
        L.stop(f"{PREREG_GEO2_MD.name} sha256 != {PREREG_GEO2_JSON.name} md_sha256")
    if list(rec.get("rungs") or []) != ["geo"]:
        L.stop(f"{PREREG_GEO2_JSON.name} rungs {rec.get('rungs')} != ['geo']")
    snap = rec.get("ladder_results_sha256") or {}
    if not isinstance(snap, dict) or not snap:
        L.stop(f"{PREREG_GEO2_JSON.name} has no ladder_results_sha256 snapshot")
    changed = sorted(k for k, s in snap.items() if not (L.REPO / k).is_file() or L.sha256(L.REPO / k) != s)
    if changed:
        L.stop(f"LADDER result files changed since the GEO2 pre-registration (GEO2 14): {changed[:8]}")
    return dict(md=str(PREREG_GEO2_MD), md_sha256=md, json=str(PREREG_GEO2_JSON), json_sha256=L.sha256(PREREG_GEO2_JSON),
                n_ladder_snapshot=len(snap), ladder_results_unchanged=True)


def e3_ready():
    """E3-4: the stage-3 collect outputs present and current, else exit 2 (= E2-3 for the geo files)."""
    for p in (PRED_GEO, L_REPORT_GEO, L_VAL_GEO, L_XGB_REPORT_GEO):
        if not p.is_file():
            _not_ready(f"{p} missing (train_xgb.py check --stage 3, then train_atoms.py collect --stage 3 --with-xgb)")
    rep = json.loads(L_REPORT_GEO.read_text())
    why = []
    if int(rep.get("stage") or 0) != 3:
        why.append(f"L_report_geo.json stage {rep.get('stage')} (not a stage-3 collect)")
    if list(rep.get("rungs") or []) != ["geo"]:
        why.append(f"L_report_geo.json rungs {rep.get('rungs')}")
    if "XGB" not in (rep.get("models") or []):
        why.append("collected without --with-xgb")
    if rep.get("predictions_sha256") != L.sha256(PRED_GEO):
        why.append("L_predictions_geo.parquet differs from L_report_geo.json")
    if rep.get("val_csv_sha256") != L.sha256(L_VAL_GEO):
        why.append("L_val_geo.csv differs from L_report_geo.json")
    t_rep = L_REPORT_GEO.stat().st_mtime
    newer = [x for a in ARMS for m in MODELS4 for k in range(NF) for x in [unit_name("geo", a, m, k)]
             if (UNITS / f"{x}.json").is_file() and (UNITS / f"{x}.json").stat().st_mtime > t_rep]
    if newer:
        why.append(f"{len(newer)} geo unit jsons newer than L_report_geo.json (e.g. {newer[:3]}): rerun the collect")
    if why:
        _not_ready("stage-3 collect output missing or stale: " + "; ".join(why))
    return dict(l_report_geo_stage=rep.get("stage"), l_report_geo_rungs=rep.get("rungs"), l_report_geo_mtime=t_rep)


def geo_unit_prereg(units, md_sha):
    """E3-4 / GA-8: every geo unit json records the current PREREG_GEO2.md sha256 (key prereg_geo2) and is no smoke."""
    bad, n = [], 0
    for x, u in units.items():
        if u["key"][0] != "geo":
            continue
        n += 1
        rec = json.loads((UNITS / f"{x}.json").read_text())
        if (rec.get("fingerprint") or {}).get("prereg_geo2") != md_sha or rec.get("smoke"):
            bad.append(x)
    if n != len(ARMS) * len(MODELS4) * NF or bad:
        L.stop(f"geo units: {n} found (need {len(ARMS) * len(MODELS4) * NF}); without the current PREREG_GEO2 md "
               f"sha256 or smoke: {bad[:6]} (GEO2 8-4: every unit trained under the pre-registration)")
    return dict(n_units=n, prereg_geo2_md_sha256=md_sha, rule="unit fingerprint prereg_geo2 = PREREG_GEO2.md sha256")


def check_l_report_geo(units, labels, md_sha):
    """E3-4: L_report_geo.json / L_val_geo.csv / L_xgb_report_geo.json agree with the geo unit jsons (= check_l_report
    for the stage-3 collect; read before the predictions)."""
    rep = json.loads(L_REPORT_GEO.read_text())
    bad = []
    if rep.get("labels") != str(labels):
        bad.append(f"labels {rep.get('labels')} != {labels}")
    if (rep.get("geo2") or {}).get("prereg_geo2_md_sha256") != md_sha:
        bad.append("L_report_geo.json geo2.prereg_geo2_md_sha256 != PREREG_GEO2.md")
    ru = rep.get("units") or {}
    if set(ru) != set(units):
        bad.append(f"L_report_geo.json units ({len(ru)}) != the geo unit jsons ({len(units)})")
    for x, u in units.items():
        got = (ru.get(x) or {}).get("val_kcal_mae") or {}
        if any(got.get(t) != u["val"][t] for t in TARGETS):
            bad.append(f"{x}: val_kcal_mae differs from L_report_geo.json")
    Lv = pd.read_csv(L_VAL_GEO)
    vd = {(r, a, m, int(k), t): float(v) for r, a, m, k, t, v in
          zip(Lv.rung, Lv.arm, Lv.model, Lv.fold, Lv.target, Lv.val_kcal_mae)}
    for x, u in units.items():
        r, a, m, k = u["key"]
        for t in TARGETS:
            w = vd.get((r, a, m, k, t))
            if w is None or abs(w - u["val"][t]) > 1e-12 * max(1.0, abs(u["val"][t])):
                bad.append(f"{x} {t}: L_val_geo.csv {w} != unit json {u['val'][t]}")
                break
    xr = json.loads(L_XGB_REPORT_GEO.read_text())
    if not xr.get("ok"):
        bad.append(f"L_xgb_report_geo.json ok {xr.get('ok')}")
    if xr.get("labels") != str(labels):
        bad.append(f"L_xgb_report_geo.json labels {xr.get('labels')} != {labels}")
    xu = xr.get("units") or {}
    for x, u in units.items():
        if u["key"][2] != "XGB":
            continue
        e = xu.get(x) or {}
        if e.get("json_sha256") != u["json_sha256"] or e.get("parquet_sha256") != u["parquet_sha256"]:
            bad.append(f"{x}: not in L_xgb_report_geo.json or changed since")
    if bad:
        L.stop("L_report_geo.json / L_val_geo.csv / L_xgb_report_geo.json disagree with the geo unit jsons:\n  "
               + "\n  ".join(bad[:8]))
    return dict(l_report_geo_sha256=L.sha256(L_REPORT_GEO), l_val_geo_sha256=L.sha256(L_VAL_GEO),
                l_xgb_report_geo_sha256=L.sha256(L_XGB_REPORT_GEO), predictions_geo_sha256=L.sha256(PRED_GEO),
                n_units=len(units), models=rep.get("models"))


def load_predictions_geo():
    G = pd.read_parquet(PRED_GEO)
    miss = [c for c in PRED_COLS if c not in G.columns]
    if miss:
        L.stop(f"{PRED_GEO} lacks columns {miss}")
    if set(G.rung.astype(str)) != {"geo"}:
        L.stop(f"{PRED_GEO} holds rungs {sorted(set(G.rung.astype(str)))}, not geo only")
    G = G[PRED_COLS].copy()
    G["pair_id"] = G.pair_id.astype(str)
    G["fold"] = G.fold.astype(int)
    return G.reset_index(drop=True)


def geo_inputs(IN, rep_p, S, ladder_sets):
    """E3-3 / E3-4: the GEO2 Phase P files of the geo rung (read only) -> sets P_geo / P_23 / P_023, the T1 test and
    outer-train lists restricted to P_geo (fold copies, sha256 = P_report), dH (geo table H_<t>, h = cr)."""
    for p in (P_BUILD_GEO, P_GEO_REPORT, P_GEO_SETS, GEO_TABLE):
        if not p.is_file():
            _not_ready(f"{p} missing (GEO2 Phase P: feat_geo.py p / p_report)")
    b = json.loads(P_BUILD_GEO.read_text())
    tsha = L.sha256(GEO_TABLE)
    if not b.get("all_pass") or b.get("stop") or (b.get("outputs") or {}).get(str(GEO_TABLE)) != tsha:
        L.stop(f"{GEO_TABLE} sha256 {tsha[:12]} != {P_BUILD_GEO.name} outputs, or the build did not pass")
    pr = json.loads(P_GEO_REPORT.read_text())
    ssha = L.sha256(P_GEO_SETS)
    bad = []
    if pr.get("stop"):
        bad.append(f"{P_GEO_REPORT.name} stop: {pr.get('stop_reasons')}")
    if (pr.get("outputs") or {}).get(str(P_GEO_SETS)) != ssha:
        bad.append(f"{P_GEO_SETS.name} sha256 != {P_GEO_REPORT.name} outputs")
    J = json.loads(P_GEO_SETS.read_text())
    lists = {}
    for k in GEO_SET_NAMES:
        if not isinstance(J.get(k), list):
            L.stop(f"{P_GEO_SETS.name} lacks the list {k}")
        lists[k] = [str(x) for x in J[k]]
        if len(set(lists[k])) != len(lists[k]):
            bad.append(f"{k}: duplicated pair ids")
        want = (pr.get("sha256") or {}).get(k)
        if want is not None and want != _list_sha(lists[k]):
            bad.append(f"{k}: sha256 != {P_GEO_REPORT.name} sha256.{k}")
    sets = {k: set(v) for k, v in lists.items()}
    for k in ("P_g2", "P_dft"):
        if sets[k] != ladder_sets[k]:
            bad.append(f"{k} of {P_GEO_SETS.name} ({len(sets[k])}) != the LADDER set file ({len(ladder_sets[k])})")
    if sets["P_23"] != sets["P_g2"] & sets["P_geo"]:
        bad.append("P_23 != P_g2 & P_geo")
    if sets["P_023"] != sets["P_dft"] & sets["P_23"]:
        bad.append("P_023 != P_dft & P_g2 & P_geo")
    T = pd.read_parquet(GEO_TABLE)
    T = T.assign(rxn_id=T.rxn_id.astype(int))
    if T.rxn_id.duplicated().any():
        bad.append(f"{GEO_TABLE}: duplicated rxn_id")
    usable = set(T.rxn_id[T.usable.astype(bool)].tolist())
    ar, br_ = S.a_rxn.astype(int).to_numpy(), S.b_rxn.astype(int).to_numpy()
    pg = {p for p, x, y in zip(S.index, ar, br_) if int(x) in usable and int(y) in usable}
    if pg != sets["P_geo"]:
        bad.append(f"P_geo ({len(sets['P_geo'])}) != pairs with both ends usable in {GEO_TABLE.name} ({len(pg)})")
    if bad:
        L.stop("GEO2 pair sets / geo table inconsistent (E3-3):\n  " + "\n  ".join(bad))
    files = (rep_p.get("folds") or {}).get("files") or {}
    tests, trains = {}, {}
    for k in range(NF):
        p = IN.SPLITS_COPY / f"T1__f{k}.json"
        want = files.get(str(p))
        if not p.is_file() or want is None or L.sha256(p) != want:
            L.stop(f"{p}: missing or sha256 != results/P_report.json folds")
        rec = json.loads(p.read_text())
        tests[k] = [str(q) for q in rec["test"] if str(q) in sets["P_geo"]]
        trains[k] = [str(q) for q in rec["train"] if str(q) in sets["P_geo"]]
    Ti = T.set_index("rxn_id")
    ids = [p for p in S.index if p in sets["P_geo"]]
    a = S.loc[ids, "a_rxn"].astype(int).to_numpy()
    b_ = S.loc[ids, "b_rxn"].astype(int).to_numpy()
    D = pd.DataFrame(index=pd.Index(ids, name="pair_id"))
    for t in TARGETS:
        D[t] = (Ti.loc[a, f"H_{t}"].to_numpy(float) - Ti.loc[b_, f"H_{t}"].to_numpy(float)) if t in H_TARGETS else 0.0
    if not np.isfinite(D.to_numpy(float)).all():
        L.stop("non-finite dH in P_geo (geo input table)")
    info = dict(table=str(GEO_TABLE), table_sha256=tsha, p_build_geo_sha256=L.sha256(P_BUILD_GEO),
                p_geo_report_sha256=L.sha256(P_GEO_REPORT), p_geo_sets_sha256=ssha,
                n={k: len(v) for k, v in sets.items()}, sha256={k: _list_sha(v) for k, v in lists.items()},
                rule=DECISIONS3["E3-3"])
    return dict(sets={k: sets[k] for k in ("P_geo", "P_23", "P_023")}, tests=tests, trains=trains, dH=D, info=info)


# ---------------------------------------------------------------- E3 selections, frames, tables
def pick_rep8(V, rungs):
    """GEO2 9-1 REP8: per rung x target the (arm, model) of the 8 with the lowest mean over the 5 outer folds of
    val_kcal_mae (pick_rep's mean); ties SH before S, then AW, GNN, TF, XGB. Validation records only."""
    rep8, means8 = {}, {}
    for (r, t), g in V[V.rung.isin(list(rungs))].groupby(["rung", "target"], sort=False):
        mm = {}
        for (a, m), h in g.groupby(["arm", "model"], sort=False):
            h = h.sort_values("fold")
            v = h.val_kcal_mae.to_numpy(float)
            if h.fold.astype(int).tolist() != list(range(NF)) or not np.isfinite(v).all():
                L.stop(f"REP8: {r} {a} {m} {t}: validation MAE not finite or folds {h.fold.tolist()}")
            mm[(a, m)] = float(np.mean(v))
        if set(mm) != {(a, m) for a in ARMS for m in MODELS4}:
            L.stop(f"REP8: {r} {t}: arm x model {sorted(mm)} incomplete")
        rep8[(r, t)] = sorted(mm, key=lambda q: (mm[q], REP8_ARMS.index(q[0]), MODEL_ORDER[q[1]]))[0]
        means8[(r, t)] = mm
    return rep8, means8


def rep8_json(rep8, means8):
    return {f"{r}|{t}": dict(arm=v[0], model=v[1], mean_val_kcal_mae={f"{a}|{m}": x for (a, m), x in means8[(r, t)].items()})
            for (r, t), v in rep8.items()}


def gate_pairs_e3(EV, fr, ref, Y, sets, rungs):
    """spec 16 E / E2-5 on E3: per rung every model / arm has exactly the rung's test pairs; on P_23 and P_023 the
    rungs' lists are identical and every frame has exactly that list."""
    out = {r: EV.pair_sets_gate({f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS_E1}, ref[r], Y)
           for r in rungs}
    for name, rr in E3_COMMON.items():
        rc = common_lists(ref, sets[name], rr, name=name)
        frames = {f"{r}|{a}|{m}|{name}": f[f.pair_id.isin(sets[name])] for (r, a, m), f in fr.items() if r in rr}
        out[name] = EV.pair_sets_gate(frames, rc, Y)
    return out


def annotate_e3(EV, fr, S, sets):
    """{(rung, arm, model, pair_set): annotated frame} on the E3_LAYOUT pair sets; LINE / null once (arm S, E1-4)."""
    A = {}
    for (r, a, m), f in fr.items():
        if m in ARM_FREE and a != ARMS[0]:
            continue
        d = EV.annotate(f, S)
        for ps in E3_LAYOUT[r]:
            A[(r, a, m, ps)] = d if ps == "own" else d[d.pair_id.isin(sets[ps])]
    return A


def add_bend_passed(G):
    """E3-5: bend_passed = sensitivity >= 0.7 and FPR <= 0.1 (target rows); rung_arm rows: oi and cpcm both."""
    G = G.copy()
    key = {}
    for r in G.itertuples(index=False):
        if r.level == "target":
            key[(r.rung, r.arm, r.model, r.target)] = bool(r.crit_sens) and bool(r.crit_fpr)
    bp = []
    for r in G.itertuples(index=False):
        if r.level == "target":
            bp.append(key[(r.rung, r.arm, r.model, r.target)])
        elif r.level == "rung_arm":
            bp.append(all(key.get((r.rung, r.arm, r.model, t), False) for t in GOAL_TARGETS))
        else:
            bp.append(None)
    G["bend_passed"] = bp
    return G


def goal_rep8(MI, rates, rep8, rungs):
    """E3-5: LADDER 13-2 on the REP8 choice per target (GEO2 10-1 '목표 도달' = passed for oi and cpcm)."""
    rows, verdict = [], {}
    for r in rungs:
        per, detail = {}, {}
        for t in PRIMARY9:
            a, m = rep8[(r, t)]
            pp = MI[(r, a, m, "own", t, "P_main", "p")]
            pa = MI[(r, a, m, "own", t, "P_main", "all")]
            br = rates[f"{r}|{a}|{m}|own"]["all"][t]
            mae, hi, sens, fpr = pp["mae"], pp["mae_ci_hi"], br["sens"], br["fpr"]
            c_mae = bool(_fin(mae) and mae <= GOAL["mae"])
            c_sens = bool(_fin(sens) and sens >= GOAL["sens"])
            c_fpr = bool(_fin(fpr) and fpr <= GOAL["fpr"])
            ok = bool(c_mae and c_sens and c_fpr)
            sure = bool(ok and _fin(hi) and hi <= GOAL["mae"])
            per[t] = dict(passed=ok, surely=sure, bend=bool(c_sens and c_fpr), mae=c_mae)
            detail[t] = dict(arm=a, model=m, para_mae=mae, para_mae_ci_lo=pp["mae_ci_lo"], para_mae_ci_hi=hi, n_para=pp["n"],
                             sens=sens, n_bent_dft=br["n_bent_dft"], fpr=fpr, n_straight_dft=br["n_straight_dft"])
            r2 = pa["r2_fam_median"]
            rows.append(dict(
                level="target", rung=r, arm=a, model="REP8", role="primary (GEO2 9-1 / 10-1)", source=m, target=t,
                judged=bool(t in GOAL_TARGETS), pair_set_name=f"P_{r}", n_para=pp["n"], para_mae=mae,
                para_mae_ci_lo=pp["mae_ci_lo"], para_mae_ci_hi=hi, n_cores=pp["n_cores"], n_fam=br["n_fam"],
                n_bent_dft=br["n_bent_dft"], sens=sens, sens_ci_lo=br["sens_ci"][0], sens_ci_hi=br["sens_ci"][1],
                n_straight_dft=br["n_straight_dft"], fpr=fpr, fpr_ci_lo=br["fpr_ci"][0], fpr_ci_hi=br["fpr_ci"][1],
                crit_mae=c_mae, crit_sens=c_sens, crit_fpr=c_fpr, passed=ok, surely_passed=sure, mae_all=pa["mae"],
                mae_null_all=pa["mae_null"],
                beats_null_point=bool(_fin(pa["mae"]) and _fin(pa["mae_null"]) and pa["mae"] < pa["mae_null"]),
                r2_fam_median=r2, stage1_descriptive=(bool(r2 >= STAGE1_R2) if _fin(r2) else None),
                stage0_rev9=STAGE0_NOTE, conclusion=None, bend_passed=bool(c_sens and c_fpr)))
        v = dict(passed=all(per[t]["passed"] for t in GOAL_TARGETS), surely_passed=all(per[t]["surely"] for t in GOAL_TARGETS),
                 bend_passed=all(per[t]["bend"] for t in GOAL_TARGETS), mae_passed=all(per[t]["mae"] for t in GOAL_TARGETS),
                 choices={t: list(rep8[(r, t)]) for t in PRIMARY9}, detail={t: detail[t] for t in GOAL_TARGETS})
        verdict[f"{r}|REP8"] = v
        rows.append(dict(level="rung", rung=r, arm="; ".join(f"{t}: {rep8[(r, t)][0]}" for t in GOAL_TARGETS),
                         model="REP8", role="primary (GEO2 9-1 / 10-1)",
                         source="; ".join(f"{t}: {rep8[(r, t)][1]}" for t in GOAL_TARGETS), target="+".join(GOAL_TARGETS),
                         judged=True, pair_set_name=f"P_{r}", passed=v["passed"], surely_passed=v["surely_passed"],
                         crit_mae=v["mae_passed"], bend_passed=v["bend_passed"],
                         conclusion=("GEO2 10-1 목표 도달" if v["passed"] else "GEO2 10-1 목표 미도달")
                         + f" (MAE part {v['mae_passed']}, bend part {v['bend_passed']})"))
    return pd.DataFrame(rows), verdict


def compare_e3(EV, fr, S, ref, sets, rep):
    """E3_compare (E3-6, descriptive): FU2 compare, d = MAE(x) - MAE(b) (E1-9 sign)."""
    rows = []

    def evalsets(rf):
        return {"P_main": EV.eval_pairs(S, rf), "P_main_para": EV.eval_pairs(S, rf, para=True)}
    own = evalsets(ref["geo"])
    for m in CMP_MODELS:
        _cmp(EV, fr, rep, rows, "SH vs S (what HF adds)", ("geo", "SH", m), ("geo", "S", m), "own", "P_geo", own)
    for b, x, ps, label in E3_PAIRS:
        com = evalsets(common_lists(ref, sets[ps], E3_COMMON[ps], name=ps))
        for a in ARMS:
            for m in CMP_MODELS:
                _cmp(EV, fr, rep, rows, label, (b, a, m), (x, a, m), ps, ps, com)
        _cmp(EV, fr, rep, rows, label, (b, "S", "LINE"), (x, "S", "LINE"), ps, ps, com, arm_label="-")
    for a in ARMS:
        _cmp(EV, fr, rep, rows, "REP vs LINE (what the model adds on the calibration line)", ("geo", a, "REP"),
             ("geo", a, "LINE"), "own", "P_geo", own)
    out = pd.DataFrame(rows)
    out.insert(1, "sign", "d = MAE(x) - MAE(b); d > 0: b better (FU2 D5: b = the first named)")
    return out


def model_table(MI):
    """oi / cpcm / barrier para MAE + CI per rung (g2, geo) x pair set (own, P_23) x arm x model (GEO2 11 sentence 4)."""
    rows = []
    for r in E3_GOAL_RUNGS:
        for ps in ("own", "P_23"):
            for a in ARMS:
                for m in MODELS4 + ("REP", "ENS4"):
                    for t in ("oi", "cpcm", "barrier"):
                        x = MI.get((r, a, m, ps, t, "P_main", "p"))
                        if x:
                            rows.append(dict(rung=r, pair_set=ps, pair_set_name=x["pair_set_name"], arm=a, model=m,
                                             source=x["source"], target=t, n=x["n"], para_mae=x["mae"],
                                             ci_lo=x["mae_ci_lo"], ci_hi=x["mae_ci_hi"]))
    return rows


def run_e3(EV, ctx, P, rep, rep8):
    """gates -> frames -> metrics -> bends -> 13-2 (REP, ENS4, models, REP8) -> compare; nothing is written here."""
    t0 = time.time()
    rungs, S, Y, sets = ctx["rungs"], ctx["S"], ctx["Y"], ctx["sets"]
    ref = {r: ref_set(ctx["tests"][r], r) for r in rungs}
    fr = model_frames(P, rungs)
    gm = {r: EV.pair_sets_gate({f"{r}|{a}|{m}": fr[(r, a, m)] for a in ARMS for m in MODELS4}, ref[r], Y)
          for r in rungs}
    check_pair_attrs(P, S)
    lines = line_frames(ctx)
    fr = derived_frames(fr, lines, rep, rungs)
    ga = gate_pairs_e3(EV, fr, ref, Y, sets, rungs)
    lchk = check_line(fr, lines, rungs)
    _log3(t0, "gates passed: test-pair lists per rung and on P_23 / P_023 (spec 16 E), LINE = SH baseline (E1-5)")
    A = annotate_e3(EV, fr, S, sets)
    Mt = metrics_e1(EV, A)
    MI = met_index(Mt)
    _log3(t0, f"metrics: {len(Mt)} rows")
    Bd, rates, fams = bends_e1(EV, A, S, rep)
    _log3(t0, f"bends: {len(Bd)} rows")
    goal, verdict, _ = goal_e1(MI, rates, E3_GOAL_RUNGS)
    g8, verdict8 = goal_rep8(MI, rates, rep8, E3_GOAL_RUNGS)
    goal = pd.concat([add_bend_passed(goal), g8], ignore_index=True)
    Cmp = compare_e3(EV, fr, S, ref, sets, rep)
    _log3(t0, f"compare: {len(Cmp)} rows")
    gates = dict(models=gm, all=ga, n_test_pairs={r: len(ref[r]) for r in rungs},
                 n_common_test_pairs={n: len(common_lists(ref, sets[n], rr, name=n)) for n, rr in E3_COMMON.items()},
                 n_pairs_in_sets={n: len(v) for n, v in sets.items()})
    return dict(fr=fr, Mt=Mt, MI=MI, Bd=Bd, rates=rates, fams=fams, goal=goal, verdict=verdict, verdict8=verdict8,
                Cmp=Cmp, gates=gates, line_check=lchk, line_betas={r: lines[r][1] for r in rungs}, ref=ref,
                seconds=time.time() - t0)


# ---------------------------------------------------------------- E3 cross-checks and selftest checks
def e2_crosscheck_e3(res, rep, e2):
    """E3-7: E3's g2 own rows = the LADDER E2 outputs; dft / g2 REP = E2_report.json."""
    bad, rec = [], {}
    R = e2["report"].get("representatives") or {}
    for (r, a, t), m in sorted(rep.items()):
        if r in E3_LADDER_RUNGS:
            m2 = (((R.get(r) or {}).get(a) or {}).get(t) or {}).get("model")
            if m2 != m:
                bad.append(f"REP {r}|{a}|{t}: E3 {m} != E2 {m2}")
    keycols = {c: str for c in MET_KEY + ["source", "pair_set_name"]}
    X = pd.read_csv(E2_METRICS_F, dtype=keycols, keep_default_na=False, na_values=[""], float_precision="round_trip")
    X = X[(X.rung == "g2") & (X.pair_set == "own")]
    Mt = res["Mt"]
    mine = Mt[(Mt.rung == "g2") & (Mt.pair_set == "own")]
    J = mine.merge(X, on=MET_KEY, how="outer", suffixes=("", "_e2"), indicator=True)
    n_only = int((J["_merge"] != "both").sum())
    if n_only:
        bad.append(f"E2_metrics g2 own: {n_only} rows not in both E3 and E2")
    Jb = J[J["_merge"] == "both"]
    ncol = {}
    for c in E3_NUM + ("source",):
        nb = int(sum(not _num_close(x, y) for x, y in zip(Jb[c].tolist(), Jb[f"{c}_e2"].tolist())))
        ncol[c] = nb
        if nb:
            bad.append(f"E2_metrics g2 own column {c}: {nb} rows differ")
    rec["metrics"] = dict(n_rows_e3=int(len(mine)), n_rows_e2=int(len(X)), n_not_in_both=n_only, n_differing=ncol)
    BR = json.loads(E2_RATES_F.read_text()).get("rates") or {}
    keys = sorted(k for k in res["rates"] if k.startswith("g2|") and k.endswith("|own"))
    rb = []
    for k in keys:
        if k not in BR:
            rb.append(f"{k}: not in E2_bend_rates.json")
            continue
        rb += _same_obj(_clean(res["rates"][k]), BR[k], path=k)
    bad += rb[:20]
    rec["bend_rates"] = dict(n_keys=len(keys), n_problems=len(rb))
    gk = ["level", "rung", "arm", "model", "target"]
    G2 = pd.read_csv(E2_GOAL_F, dtype={c: str for c in gk}, keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    G2 = G2[(G2.rung == "g2") & G2.level.isin(["target", "rung_arm"])]
    G3 = res["goal"]
    G3 = G3[(G3.rung == "g2") & G3.level.isin(["target", "rung_arm"]) & (G3.model != "REP8")]    # REP8: E3 only
    JG = G3.merge(G2, on=gk, how="outer", suffixes=("", "_e2"), indicator=True)
    gb = [f"goal rows not in both: {int((JG['_merge'] != 'both').sum())}"] if (JG["_merge"] != "both").any() else []
    JGb = JG[JG["_merge"] == "both"]
    for c in ("passed", "surely_passed", "para_mae", "para_mae_ci_hi", "sens", "fpr"):
        for x, y, lv, ar, mo, tg in zip(JGb[c], JGb[f"{c}_e2"], JGb.level, JGb.arm, JGb.model, JGb.target):
            if not _num_close(x, y):
                gb.append(f"goal {lv} {ar} {mo} {tg} {c}: E3 {x} != E2 {y}")
    bad += gb[:20]
    rec["goal"] = dict(n_rows_e3=int(len(G3)), n_rows_e2=int(len(G2)), n_problems=len(gb))
    return dict(ok=not bad, problems=bad[:40], **rec, rule=DECISIONS3["E3-7"])


def eval_geo2_crosscheck(path, rep8, MI, rung="geo"):
    """E3-7 (record only): eval_geo2.py E_report.json rep8 and goal_mae REP8 = E3's REP8 choice and its para MAE."""
    if not path.is_file():
        return dict(status=f"{path} absent: not compared", equal=None)
    E = json.loads(path.read_text())
    r8 = E.get("rep8") or {}
    diff = [f"{r}|{t}: E3 {list(v)} != eval_geo2 {r8.get(f'{r}|{t}')}" for (r, t), v in sorted(rep8.items())
            if list(r8.get(f"{r}|{t}") or []) != list(v)]
    gm = (E.get("goal_mae") or {}).get("REP8") or {}
    dm = {}
    for t in GOAL_TARGETS:
        a, m = rep8[(rung, t)]
        mine = MI[(rung, a, m, "own", t, "P_main", "p")]["mae"]
        other = gm.get(t)
        dm[t] = dict(e3=mine, eval_geo2=other, abs_diff=(abs(mine - float(other)) if other is not None else None))
    eq = bool(not diff and all(v["abs_diff"] is not None and v["abs_diff"] <= 1e-9 for v in dm.values()))
    return dict(status="compared", file=str(path), sha256=L.sha256(path), rep8_differences=diff, goal_mae_REP8=dm,
                equal=eq)


def selftest_e3_checks(res, rep, rep8, e2x, egx):
    """E3-8: with geo = a copy of g2 every geo number must equal the g2 one."""
    checks = {}
    Mt = res["Mt"]
    k = ["arm", "model", "pair_set", "target", "set", "pos"]
    bad = []
    for ps in ("own", "P_23", "P_023"):
        a = Mt[(Mt.rung == "geo") & (Mt.pair_set == ps)]
        b = Mt[(Mt.rung == "g2") & (Mt.pair_set == ps)]
        J = a.merge(b, on=k, how="outer", suffixes=("", "_g2"), indicator=True)
        if (J["_merge"] != "both").any() or not len(a):
            bad.append(f"{ps}: rows not in both ({int((J['_merge'] != 'both').sum())}) or empty")
        for c in E3_NUM + ("source",):
            nb = int(sum(not _num_close(x, y) for x, y in zip(J[c].tolist(), J[f"{c}_g2"].tolist())))
            if nb:
                bad.append(f"{ps} {c}: {nb} rows differ")
    checks["metrics_geo_equal_g2"] = not bad
    checks["rep_geo_equal_g2"] = all(rep[("geo", a_, t)] == rep[("g2", a_, t)] for a_ in ARMS for t in TARGETS)
    checks["rep8_geo_equal_g2"] = all(rep8[("geo", t)] == rep8[("g2", t)] for t in TARGETS)
    rb = []
    for key, v in res["rates"].items():
        if key.startswith("geo|"):
            other = "g2|" + key[len("geo|"):]
            rb += [f"{key}: no {other}"] if other not in res["rates"] else \
                _same_obj(_clean(v), _clean(res["rates"][other]), tol=0.0, path=key)
    checks["bend_rates_geo_equal_g2"] = not rb
    G = res["goal"]
    gk = ["level", "arm", "model", "target"]
    Ga, Gb = G[G.rung == "geo"], G[G.rung == "g2"]
    JG = Ga.merge(Gb, on=gk, how="outer", suffixes=("", "_g2"), indicator=True)
    gbad = int((JG["_merge"] != "both").sum())
    for c in ("passed", "surely_passed", "bend_passed", "para_mae", "sens", "fpr", "source"):
        gbad += int(sum(not _num_close(x, y, 0.0) for x, y in zip(JG[c].tolist(), JG[f"{c}_g2"].tolist())))
    checks["goal_geo_equal_g2"] = bool(gbad == 0 and len(Ga) > 0)
    checks["goal_rep8_rung_rows"] = bool(((G.level == "rung") & (G.model == "REP8")).sum() == len(E3_GOAL_RUNGS))
    C = res["Cmp"]
    c = C[C.comparison == E3_PAIRS[0][3]]
    checks["compare_g2_vs_geo_zero"] = bool(len(c) > 0 and (c.d.abs() == 0).all() and (c.ci_lo == 0).all()
                                            and (c.ci_hi == 0).all())
    checks["e2_crosscheck"] = bool(e2x.get("ok"))
    checks["eval_geo2_selftest_equal"] = bool(egx.get("equal"))
    return checks, dict(metrics_problems=bad[:20], bend_rate_problems=rb[:20], goal_problems=gbad)


# ---------------------------------------------------------------- the E3 command
def cmd_e3(labels, selftest=False):
    """E3 (GEO2 9-2): the E2 computation on rung geo (+ dft / g2 on the common sets) -> analysis/geo2/results/E3_*;
    selftest: geo = a copy of g2 -> analysis/geo2/results/e3_selftest/ (E3-8)."""
    t0 = time.time()
    EV, info = fu2_eval()
    import inputs as IN
    lab = label_info(labels)
    labels = Path(lab["path"])
    rep_p = IN.p_report()                                # E3-2: no rung argument (cross-worktree build fingerprints)
    rep_p, f13 = f13_inputs(IN, rep_p)                   # E3-2b: S_folds entry by repo-relative path (copy)
    builds = ladder_builds(IN, rep_p, E3_LADDER_RUNGS)
    add3 = add3_check()
    e2 = e2_record()
    tables = {r: builds[r]["table_sha256"] for r in E3_LADDER_RUNGS}
    pre = ready = None
    if not selftest:
        pre = prereg_geo2_check()
        ready = e3_ready()
        if not GEO_TABLE.is_file():
            _not_ready(f"{GEO_TABLE} missing")
        tables["geo"] = L.sha256(GEO_TABLE)
    with no_test_scores([PRED, PRED_GEO, UNITS]):        # E1-3 / E3-4: REP and REP8 from validation records only
        V, units = val_table(E3_LADDER_RUNGS if selftest else E3_RUNGS, lab["sha256"], add3["md_sha256"], tables)
        if selftest:
            V = pd.concat([V, V[V.rung == "g2"].assign(rung="geo")], ignore_index=True)
            gfp = dict(status="e3-selftest: geo = a copy of g2 (no geo units)")
        else:
            gfp = geo_unit_prereg(units, pre["md_sha256"])
        rep, means = pick_rep(V)
        rep8, means8 = pick_rep8(V, E3_GOAL_RUNGS)
    _log3(t0, "REP8: " + ", ".join(f"{r}·{t}={a}/{m}" for (r, t), (a, m) in rep8.items()))
    lad_units = {x: u for x, u in units.items() if u["key"][0] in E3_LADDER_RUNGS}
    geo_units = {x: u for x, u in units.items() if u["key"][0] == "geo"}
    lrep = check_l_report(lad_units, E3_LADDER_RUNGS, labels)
    lrep_geo = None if selftest else check_l_report_geo(geo_units, labels, pre["md_sha256"])
    ctx = real_inputs(IN, rep_p, E3_LADDER_RUNGS, labels, set_names=("P_dft", "P_g2"))
    if selftest:
        sg = set(ctx["sets"]["P_g2"])
        geo = dict(sets=dict(P_geo=sg, P_23=sg & ctx["sets"]["P_g2"], P_023=ctx["sets"]["P_dft"] & sg),
                   tests={k: list(v) for k, v in ctx["tests"]["g2"].items()},
                   trains={k: list(v) for k, v in ctx["trains"]["g2"].items()}, dH=ctx["dH"]["g2"].copy(),
                   info=dict(mode="e3-selftest: geo = a copy of g2 (E3-8)", n=len(sg)))
    else:
        geo = geo_inputs(IN, rep_p, ctx["S"], ctx["sets"])
    ctx = dict(ctx, rungs=E3_RUNGS, sets=dict(ctx["sets"], **geo["sets"]), tests=dict(ctx["tests"], geo=geo["tests"]),
               trains=dict(ctx["trains"], geo=geo["trains"]), dH=dict(ctx["dH"], geo=geo["dH"]))
    P = load_predictions(E3_LADDER_RUNGS)
    G = P[P.rung == "g2"].assign(rung="geo") if selftest else load_predictions_geo()
    P = pd.concat([P, G], ignore_index=True)
    res = run_e3(EV, ctx, P, rep, rep8)
    e2x = e2_crosscheck_e3(res, rep, e2)
    egx = eval_geo2_crosscheck(EVAL_GEO2_SELFTEST if selftest else EVAL_GEO2_REPORT, rep8, res["MI"])
    if not selftest and not e2x["ok"]:
        L.stop("E3 is not the E2 computation on the LADDER g2 rung (E3-7):\n  " + "\n  ".join(e2x["problems"][:10]))
    out = E3_SELFTEST_DIR if selftest else GEO2_RES
    outputs = {}
    for name, df in (("E3_metrics.csv", res["Mt"]), ("E3_bends.csv", res["Bd"]), ("E3_goal.csv", res["goal"]),
                     ("E3_compare.csv", res["Cmp"])):
        p = _csv3(out / name, df)
        outputs[str(p)] = L.sha256(p)
    p = _json3(out / "E3_bend_rates.json", dict(
        key="rung|arm|model|pair_set (pair_set own = P_<rung>, P_23, P_023)", rates=res["rates"], families=res["fams"],
        representatives=rep_json(rep, means), rep8=rep8_json(rep8, means8),
        note="d1_eval.bend_rates on FU2 bends_tab rows (as E2_bend_rates.json); null = zero denominator (E1-7)"))
    outputs[str(p)] = L.sha256(p)
    v8 = res["verdict8"].get("geo|REP8") or {}
    me = Path(__file__).resolve()
    ins = [PRED, L_REPORT, L_VAL, labels, ORIG_LABELS, IN.S_FOLDS, IN.P_REPORT, ADD3_MD, ADD3_JSON, E2_REPORT_F,
           E2_METRICS_F, E2_GOAL_F, E2_RATES_F, Path(info["source"]), Path(info["e_tests"])] + \
        [Path(rep_p["sets"][n]["file"]) for n in ("P_dft", "P_g2")] + [IN.table_path(r) for r in E3_LADDER_RUNGS]
    if not selftest:
        ins += [PRED_GEO, L_REPORT_GEO, L_VAL_GEO, L_XGB_REPORT_GEO, PREREG_GEO2_MD, PREREG_GEO2_JSON, P_BUILD_GEO,
                P_GEO_REPORT, P_GEO_SETS, GEO_TABLE]
    doc = dict(
        phase="E3-selftest" if selftest else "E3", spec="docs/specs/GEO2.md 9-2, 10", rungs=list(E3_RUNGS),
        layout={r: list(v) for r, v in E3_LAYOUT.items()}, command=" ".join(sys.argv[1:]), date=time.strftime("%F %T"),
        evaluate_py=dict(path=str(me), sha256=L.sha256(me), git_state=_git_state(me) or "committed, unchanged",
                         line_source_sha256=line_source_sha256()),
        fu2_evaluate=info, labels=lab, add3=add3, ladder_builds=builds, p_report_inputs=f13, e2=e2["sha256"],
        prereg_geo2=pre, ready=ready,
        geo_unit_prereg=gfp, l_report=lrep, l_report_geo=lrep_geo, geo_inputs=geo["info"],
        inputs_sha256={str(p): L.sha256(p) for p in ins if Path(p).is_file()},
        representatives=rep_json(rep, means), rep8=rep8_json(rep8, means8),
        rep_rule="spec 13-1 (REP) and GEO2 9-1 (REP8): lowest mean over the 5 outer folds of the unit json val_kcal_mae; "
                 "REP ties AW -> GNN -> TF -> XGB, REP8 ties SH before S then that order; validation records only "
                 "(E1-3)",
        gates=res["gates"], line_check=res["line_check"], line_betas=res["line_betas"], verdict=res["verdict"],
        verdict_rep8=res["verdict8"],
        goal_reached=dict(rule="GEO2 10-1 '목표 도달' (shown separately from the main verdict): geo REP8 oi and cpcm "
                               "para MAE <= 0.13 (own P_geo, point estimate; = eval_geo2 goal_mae) and the E3 bend "
                               "criteria sensitivity >= 0.7, FPR <= 0.1 (LADDER 13-2) for both",
                          reached=v8.get("passed"), mae_part=v8.get("mae_passed"), bend_part=v8.get("bend_passed"),
                          surely=v8.get("surely_passed"), detail=v8.get("detail")),
        model_table=model_table(res["MI"]), e2_crosscheck=e2x, eval_geo2_crosscheck=egx, outputs=outputs,
        goal_rule=dict(targets=list(GOAL_TARGETS), para_mae_max=GOAL["mae"], sens_min=GOAL["sens"], fpr_max=GOAL["fpr"],
                       set="own pair set (P_<rung>), P_main reference para pairs, point estimate; surely = CI upper <= "
                           "0.13; REP primary, ENS4 secondary, single models descriptive; REP8 = the GEO2 10-1 goal",
                       bend_population="d1_eval.bend_rates 'all' (no family filter; ADD3 5-2 / PR-15)"),
        env=_env(), decisions=DECISIONS3, decisions_e1_reused=DECISIONS, seconds=time.time() - t0)
    if selftest:
        checks, detail = selftest_e3_checks(res, rep, rep8, e2x, egx)
        ok = all(checks.values())
        doc.update(ok=ok, checks=checks, check_detail=detail,
                   note="e3-selftest: geo = a copy of g2; the geo numbers are g2's, not results (E3-8)")
        _json3(out / "E3_selftest.json", doc)
        for kk, v in checks.items():
            print(f"[e3-selftest] {kk}: {'ok' if v else 'FAIL'}", flush=True)
        print(f"[e3-selftest] {'all checks passed' if ok else 'FAILED'} ({time.time() - t0:.0f} s) -> {out}", flush=True)
        return 0 if ok else 3
    _json3(out / "E3_report.json", doc)
    for kk, v in res["verdict"].items():
        if v["role"] != "descriptive":
            print(f"[E3] 13-2 {kk}: passed {v['passed']} surely {v['surely_passed']}", flush=True)
    print(f"[E3] REP8 geo: {v8.get('choices')} passed {v8.get('passed')} (MAE {v8.get('mae_passed')}, bends "
          f"{v8.get('bend_passed')}); E2 cross-check ok {e2x['ok']}; eval_geo2 cross-check {egx.get('equal')}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="D1-LADDER evaluation (docs/specs/D1LADDER.md 11, 12): e1 | e1-selftest | "
                                             "e2 | e2-selftest; GEO2 (docs/specs/GEO2.md 9-2): e3 | e3-selftest")
    ap.add_argument("cmd", choices=("e1", "e1-selftest", "e2", "e2-selftest", "e3", "e3-selftest"))
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    _need_pandas()
    if a.cmd == "e1":
        return cmd_e1(a.stage, a.labels, figures=not a.no_figures)
    if a.cmd == "e2":
        return cmd_e2(a.labels, figures=not a.no_figures)
    if a.cmd == "e2-selftest":
        return cmd_selftest_e2(figures=not a.no_figures)
    if a.cmd in ("e3", "e3-selftest"):                  # GEO2 9-2 (E3-1): no figures
        return cmd_e3(a.labels, selftest=a.cmd == "e3-selftest")
    return cmd_selftest(figures=not a.no_figures)


if __name__ == "__main__":
    sys.exit(main())
