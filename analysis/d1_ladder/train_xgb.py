#!/usr/bin/env python3
r"""train_xgb.py — D1-REF·LADDER Phase M1 / M2 XGB pair model (docs/specs/D1LADDER.md 4-2 .. 4-4, 10, 12): rev 9 arm C XGB
'as is' (train_ml_single.GRIDS["XGB"], antisymmetric 1/2 [h(dx, xbar) - h(-dx, xbar)], inner GroupKFold(4) by core) on the
rung x arm units  L__<rung>__<arm>__XGB__f<k>  (T1 folds 0..4). CPU only, sbatch only, env = env_espley (the rev 9 arm C
env). Nothing in this file was run when it was written (login node): every path is untested until it runs inside a job.

CLI (submit from analysis/d1_pair with jobs/py.slurm PYENV=espley, script ../d1_ladder/train_xgb.py; exit 0 = done,
2 = not ready / units missing / usage, 3 = STOP)
  python train_xgb.py list [--stage 1|2]                 unit names in run order (stage 1: 20, stage 2: 10)
  python train_xgb.py smoke [--stage 1] [--labels P]     = run --smoke
  python train_xgb.py run --smoke [--stage 1] [--labels P]
        spec 8 smoke: fold 0, every rung x arm of the stage (stage 1 = dft, g1), grid cells first + last (smallest /
        largest), all 10 targets, inner GroupKFold(4), final fit, gate M-1' + leak checks on every fit ->
        results/C_smoke_xgb.json (incl. a stage CPU-hour estimate). Bypasses the ADD3 gate; writes only $LAD/smoke_xgb
        (removed first, never read by run / check); refuses $LAD/units.
  python train_xgb.py run <i|@I|A> <n> [--stage 1|2] [--labels P]
        units [i::n] of the stage (A = $SLURM_ARRAY_TASK_ID; py.slurm replaces @I). Exit 3 unless the ADD3 gate passes
        (= train_atoms.gate_add3). Idempotent: a unit whose $LAD/units/<unit>.json exists is skipped after its
        fingerprint is compared (mismatch -> exit 3, spec 10).
  python train_xgb.py check [--stage 1|2] [--labels P]
        every XGB unit of the rungs of stages <= stage: json + parquet present, parquet sha256, not smoke, gates ok,
        per-target val_kcal_mae, test / validation pair lists = the current split, fingerprint = the CURRENT one
        (train_atoms.py collect --with-xgb leaves the XGB fingerprints to this file) -> results/L_xgb_report.json.
        Run it before `train_atoms.py collect --with-xgb`.
  python train_xgb.py l6
        L-6 helper for tests/ (spec 8): g1 x S fold 0 with the ORIGINAL F_pairs.csv, full grid, into $LAD/smoke_xgb_l6
        (no ADD3, never $LAD/units) vs the rev 9 units $D1PAIR_SCRATCH/M/units/T1/T1__C__XGB__<t>__f0.{json,parquet}:
        yhat <= 1e-6, y <= 1e-12, same best cell and inner CV MAE (<= 1e-9), rev 9 test pairs missing here = exactly the
        pairs dropped as not usable -> results/C_l6_xgb.json (exit 3 when it fails).

GEO2 stage 3 (docs/specs/GEO2.md 8-1; added 2026-10-30; decisions XG-G1 .. XG-G10 at the end of this docstring; the
  train_atoms.py GA-* rules apply line for line): rung 'geo' = STAGE_RUNGS[3] only (10 units L__geo__<arm>__XGB__f<k>).
  Stage 1 / 2 unit lists, gates and outputs are unchanged; but this file's sha256 is no longer the one that wrote the
  LADDER results, so smoke / run / check with --stage 1|2 and l6 without --stage 3 STOP before writing anything
  (XG-G9); only `list --stage 1|2` is usable here.
  python train_xgb.py list --stage 3                    the 10 geo XGB units
  python train_xgb.py smoke --stage 3 [--labels P]      -> results/C_smoke_xgb_geo.json, scratch $LAD/smoke_xgb_geo
  python train_xgb.py run <i|@I|A> <n> --stage 3        gates = train_atoms stage 3 (PREREG_GEO2 + LADDER result
        snapshot, GEO2 Phase P, analysis/geo2/results/C_tests_geo.json); units -> $LAD/units/L__geo__*__XGB__* (new)
  python train_xgb.py check --stage 3                   the geo XGB units only -> results/L_xgb_report_geo.json
  python train_xgb.py l6 --stage 3                      the L-6 helper re-run (GEO2 8-2 regression 2) without touching
        the LADDER outputs: -> results/C_l6_xgb_geo.json, scratch $LAD/smoke_xgb_l6_geo (same comparison as l6)
  geo inputs: X182 $GEO2/X182_d1_geo.parquet, reports analysis/geo2/results/fu2_geo, table $GEO2/inputs/geo.parquet.

Rules (spec 4-2, 4-3; rev 9 d1_train.py arm C unchanged unless named)
  pairs / folds  FU2 T1 split files $D1PAIR_SCRATCH/fu2/splits/T1__f<k>.json (sha256 = FU2 S_splits.json; their train /
                 test lists are re-derived here from S_folds.csv with the d1_train.masks arm C rule and must be equal):
                 train = outer-train T_main, test = held-out P_sens u A_main. Pairs whose a or b is not usable in the rung
                 are dropped from every list exactly as train_atoms.py does (usable reaction = T_manifest row + input-table
                 row + finite H 9 (+ finite 182 when the table carries them)) -> the same P_<rung> as AW / GNN / TF.
  inner CV       GroupKFold(4) by core (S_folds.csv) on the UNFILTERED outer-train list (= rev 9
                 d1_train.inner_folds(d1_train.masks(u)[0])), then the usable filter on each side; fold 0 is asserted to
                 be the split file inner_train / inner_val (the validation pairs of the atom models, L-9).
  model          d1_train.Scale on the fit's training pairs (z = [dx / s_delta, (xbar - mu) / s], columns with s < 1e-12
                 dropped; y / s_y, s_y = std(y) ddof 0, no mean removed), d1_train._xgb_fit (clone of
                 train_ml_single.GRIDS["XGB"][0] + set_params, trained both ways (z, y) and (-dx, -y)),
                 d1_train._xgb_g (1/2 [h(dx) - h(-dx)]), cells = d1_train.grid_cells("XGB") (12). kcal = g * s_y with
                 rev 9's dtype semantics (float32 * float), so g1 x S fold 0 with the original labels reproduces rev 9.
  per target     each of the 10 targets is tuned independently (rev 9 unit = one target): best cell = lowest mean inner
                 MAE in kcal/mol (ties: grid order, np.argmin), final fit on all usable training pairs (T_main).
  arm S          inputs = the rung's 182 global features (X182 file, parquet column order = rev 9), target y = dd_<t> of the
                 label file (default $LAD/F_pairs_cr2.csv); base = beta = 0.
  arm SH         inputs = [182 | H 9] (H in d1 d2 elst pauli oi disp cpcm cds barrier order, from $LAD/inputs/<rung>.parquet);
                 per target r_t = y_t - beta_t * dH_t, beta_t = sum(dH_t y_t) / sum(dH_t^2) without intercept, fitted ONLY
                 on the fit's own training pairs (tuning: the inner-train side; final: T_main); c_ghost beta = 0 (r = y);
                 Scale / s_y on r; inner CV score = mean |beta dH + g s_y - y| (kcal); final prediction = beta dH + g s_y.
  validation     val_kcal_mae per target (spec 13-1) = the chosen cell's inner fold 0 score (fit on inner_train, scored on
                 inner_val = the atom models' validation pairs); every fold x cell score is stored.
  gate M-1'      on EVERY fit (tuning and final), 200 random test pairs (rev 9 fit_final rng rule, seeded by the unit
                 name), inputs only, swapped reaction inputs recomputed from x in float64: |g(a,b) + g(b,a)| and |g(a,a)| of
                 the final prediction incl. the baseline <= 1e-4 kcal/mol (FU2 / spec 10) and of the model part <= 1e-6
                 (rev 9 gate M-1). A failure is exit 3 before the unit is written (spec 16 M).
  L-2 / L-8      asserted per fit: beta / scale pairs = the fit's training pairs, none of them a validation or test pair, no
                 test reaction among the fit's reactions.

Inputs (Phase P contract = train_atoms.py's; inputs.py did not exist when this was written)
  $LAD/inputs/<rung>.parquet   rxn_id (unique) + exactly one complete set H_<t> or hf_<t> (t = the 9 H targets); optional,
                 cross-checked when present: the 182 columns under T_report glob_cols names (= the X182 file, 1e-12 relative),
                 token_sha256 or tok_sha256 (the name inputs.py writes; = T_manifest), usable (= the rule above; inputs.py
                 usable = has_tok & has_x182 & has_h is the same rule). inputs.py's frame API (pairs_frame / fold_lists)
                 is not used, so that XGB and train_atoms.py read the Phase P files the same way (collect cross-checks).
  rung files     dft: analysis/d1_ladder/results/fu2/{T_report.json, T_manifest.csv, Q_report.json, S_splits.json} +
                 $LAD/X182_d1_dft.parquet; g1: analysis/d1_pair/results/fu2 + $D1PAIR_SCRATCH/X182_d1.parquet; g2:
                 analysis/d1_g2/results/fu2 + $G2/X182_d1_g2.parquet. T_report / Q_report checked as train_atoms.load_tokens
                 does (versions, sha256 links, token layout); X182 sha256 = T_report x182_sha256 (= the token glob source),
                 path = Q_report overrides FU2_X182 when set; columns = T_report glob_cols. A usable reaction without a finite
                 X182 row is a STOP (P_<rung> would differ from the atom models').
  labels         --labels (default $LAD/F_pairs_cr2.csv): provenance and the L-11 invariant exactly as train_atoms.py
                 (V_report / Q_report all_pass + outputs sha256; the original = FU2 S_splits F_pairs sha256; only dd_d1 /
                 dd_d2 / dd_barrier may differ from F_pairs.csv).
  L-4            every input path per role in the unit json ('inputs'); files opened through Python while loading
                 ('opened_inputs', sys.addaudithook; pyarrow parquet reads appear only in 'inputs'). Rung-independent roles:
                 labels*, split* (FU2 split files, S_splits.json, S_folds.csv).

Cache ($LAD/units/<unit>.parquet, then <unit>.json last)
  fingerprint keys = train_atoms.py's where they apply: code d1-ladder-2, code_sha256 (this file, lad_common.py,
  d1_train.py, d1pair_common.py, train_ml_single.py, fu2_common.py, fu2/tokens.py), rung, arm, model, inputs_table,
  labels, pair_list (P_<rung>), split_pairs, s_splits, t_report, t_manifest, q_report, add3, grid, h_cols, rules; plus
  XGB-only: inner_folds, s_folds, x182, feat_cols, estimator (repr of GRIDS["XGB"][0]) and versions (scikit-learn,
  xgboost, numpy, pandas, joblib, pyarrow, python). No torch / CUDA / GPU keys (CPU; spec 10 'XGB version'). A cached unit
  with another fingerprint -> exit 3. json keys read by train_atoms collect: parquet_sha256, val_kcal_mae {target: kcal},
  val_pairs_sha256, chosen, n_train, n_test, n_val, dropped, beta, wall_seconds.

Outputs
  $LAD/units/L__<rung>__<arm>__XGB__f<k>.{parquet,json}; $LAD/smoke_xgb/ (smoke), $LAD/smoke_xgb_l6/ (l6)
  parquet columns = train_atoms PRED_COLS: pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, base (S 0),
  beta (S 0), yhat_seed_sd (0.0: one fit); results/C_smoke_xgb.json, results/C_l6_xgb.json, results/L_xgb_report.json.

sbatch (login node: one-shot commands only). Stage 1 = 20 units = 4 rung-arms x 5 folds; per unit 10 targets x 4 inner
folds x 12 cells + 10 final fits = 490 XGB fits. Estimate from rev 9 (T1__C__XGB__<t>__f<k>: 147-176 s wall per target on
48 CPUs with 12 fits in parallel): ~0.2 CPU-h per target -> ~2 CPU-h per unit, ~40-45 CPU-h for stage 1 (SH: 9 more
columns, a few % more); 4 tasks x 32 CPUs ~ 20-40 min per task (smoke measures it: C_smoke_xgb.json).
  Phase M1 = ONE combined plan with train_atoms.py (review M-2 / M-3), written out in train_atoms.py's docstring: JX
  (this array, %4) -> JXC (check, afterok JX) -> JM (atom array, %4, afterany JXC) -> one collect --with-xgb (afterok
  JM and JXC). Peak 4 LADDER tasks; MaxSubmit N + 10 <= 20; start only when no other LADDER task runs or waits.
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_pair; L=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs
  JX=$(sbatch --parsable -p cpu1,cpu2 -J lad_xgb --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-3%4 \
       --output=$L/%x_%A_%a.out --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py run @I 4); echo "JX=$JX"
  JXC=$(sbatch --parsable -p cpu1,cpu2 -J lad_xchk --time=48:00:00 --cpus-per-task=2 --mem=16G --output=$L/%x_%j.out \
       --export=ALL,PYENV=espley --dependency=afterok:${JX:?} --kill-on-invalid-dep=yes \
       jobs/py.slurm ../d1_ladder/train_xgb.py check --stage 1); echo "JXC=$JXC"
  (--output must be given: py.slurm's default log folder is the read-only d1_pair scratch.)
  Phase C: smoke and l6 run inside tests/run_tests.py cpu (checks smoke / L-6); stand-alone if needed:
  sbatch -p cpu1,cpu2 -J lad_xsmoke --time=48:00:00 --cpus-per-task=32 --mem=64G --output=$L/%x_%j.out \
         --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py smoke
  Resubmission: a task that died -> the JX line with --array=<k> (finished units are skipped), then JXC.
  GroupKFold risk: XData.split STOPs when GroupKFold(4) fold 0 regenerated on this node differs from the split file
  (numpy argsort tie order can depend on the CPU, FU2 C8); if that happens, resubmit on the node type of the rev 9 / FU2
  export runs (rev 9 units: n112 / n115) rather than changing the folds.

Decisions (user 2026-10-22: every judgment = the recommended option; recorded here and in the unit json 'decisions')
  XG-1  env = env_espley (analysis/d1_pair/jobs/env.sh PYENV=espley; scikit-learn 1.3.1, xgboost 3.2.0, numpy 1.26.0,
        pandas 2.1.1 = the versions in every rev 9 T1__C__XGB unit json). It has no torch, so train_atoms.py (imports
        torch and fu2/train.py) cannot be imported: its shared rules (unit names, PRED_COLS, label / split / input-table
        loading, usable-pair filter, beta_fit, ADD3 gate, cache-root guard, fingerprint key names, L-4 audit hook) are
        re-implemented here line for line; train_atoms collect --with-xgb cross-checks the test-pair lists and the
        validation-pair sha256 of both files' units (spec 16 E, L-9), so a divergence is a STOP, not a silent difference.
  XG-2  wrapper: jobs/lad.slurm has no env_espley branch and jobs/gpu.slurm is env_d1fu2; no wrapper file is added (task
        scope). Jobs run through analysis/d1_pair/jobs/py.slurm (PYENV=espley, submitted from analysis/d1_pair, script
        ../d1_ladder/train_xgb.py, --output into $LAD/logs) = the g0f0.py precedent (d1_pair jobs/g.slurm).
  XG-3  rev 9 reused by import (d1_train.Scale, _xgb_fit, _xgb_g, grid_cells, d1pair_common N_INNER / ANTISYM_*);
        d1_train.data() / masks() / prep() are NOT called: they read the g1 X182_d1.parquet and the original F_pairs.csv
        (L-4 for dft / g2, new labels); the masks arm C rule is re-derived from S_folds.csv and asserted.
  XG-4  inner folds on the unfiltered outer-train list, then the usable filter (fold 0 = the atom models' validation pairs
        by construction and assertion; identical to rev 9 when nothing is dropped; core-disjoint, asserted).
  XG-5  one unit = one rung x arm x fold with all 10 targets, each tuned on its own (rev 9 per-target units, regrouped to
        the spec 10 unit name); the 10 per-target best cells are recorded under 'chosen'.
  XG-6  SH residual, beta and score exactly as spec 4-2 (no intercept, fit pairs only, c_ghost 0, score in kcal); S uses
        y itself (no 0 * dH arithmetic).
  XG-7  validation kcal MAE = the chosen cell's inner fold 0 score (the same pairs and the same 'tuning fit on inner_train'
        as train_atoms MA-6).
  XG-8  gate M-1' on every fit; both tolerances (FU2 1e-4 on the final prediction, rev 9 1e-6 on the model part).
  XG-9  yhat_seed_sd = 0.0 (one fit, estimator random_state 42 as rev 9; SD over one value); 'n_seeds': 1 in the json.
  XG-10 X182 source = the spec 4-1 file of the rung, tied to the tokens by T_report x182_sha256 (the token glob is that
        row, fu2/tokens.py) and to Q_report overrides FU2_X182; the input-table 182 columns, when present, only cross-checked.
  XG-11 ADD3 need = train_atoms' (label file + the trained rungs' input tables); the X182 file is in the fingerprint (and is
        checked by the gate too if Phase A lists it in inputs_sha256).
  XG-12 smoke = cells first + last with their real n_estimators (boosting has no epochs; the full cells are cheap), in
        $LAD/smoke_xgb (not $LAD/smoke: train_atoms smoke deletes that folder and the two smokes may run concurrently).
  XG-13 parallel = joblib (loky) over target x inner fold x cell (480 fits), then the 10 final fits; every fit n_jobs = 1
        (the GRIDS estimator), so results do not depend on the worker count; costliest cells are dispatched first.
  XG-14 run order = rung x arm x fold, sliced [i::n] (n = 4: 5 units per task); array cap: see XG-18 (superseded the
        earlier "%1 + K = 4 - R" text, review M-2).
  XG-15 l6 = a self-check command (L-6) that tests/ can call; it needs Phase P's g1 table (usability) like every run.
Review fixes 2026-10-23 (FIXER; recommended option each time)
  XG-16 check records json_sha256 / parquet_sha256 per unit; train_atoms collect --with-xgb requires that report
        current (review M-4).
  XG-17 run: ADD3 gate opens no file of another rung's column only (L-4), P_report consistency (tables + label file =
        inputs.py pairs), stage 2 needs C_tests.json g2_verified (= train_atoms MA-15 .. MA-17).
  XG-18 M1 plan: XGB array %4 first, then the atom array; one collect (review M-2 / M-3; spec 15 at most 4 jobs).
GEO2 additions 2026-10-30 (docs/specs/GEO2.md 8-1; role LADA; also in analysis/geo2/results/DECISIONS_lad.md)
  XG-G1 rung list and paths only (= train_atoms GA-1): STAGE_RUNGS[3] = ("geo",), RUNG_ORDER + "geo", RUNGS["geo"] =
        fu2_res analysis/geo2/results/fu2_geo, x182 $GEO2/X182_d1_geo.parquet, tag geo; inputs_path("geo") =
        $GEO2/inputs/geo.parquet. GRIDS["XGB"], the fits and the scoring are unchanged.
  XG-G2 stage 1 / 2 paths unchanged (except the XG-G9 refusal); stage-3 outputs carry "_geo" (res_out):
        C_smoke_xgb_geo.json, L_xgb_report_geo.json, C_l6_xgb_geo.json; scratch $LAD/smoke_xgb_geo, $LAD/smoke_xgb_l6_geo.
  XG-G3 run --stage 3 gates = train_atoms gate_prereg_geo2 (gate_add3 rule on PREREG_GEO2.md / prereg_geo2.json +
        LADDER result snapshot), gate_p_geo, gate_c_geo (re-implemented line for line: train_atoms imports torch).
  XG-G4 geo units are written to $LAD/units only after the PREREG_GEO2 gate (_check_root rung check).
  XG-G5 fingerprint: geo units add prereg_geo2 (sha256 of PREREG_GEO2.md, None in smoke) = train_atoms GA-8.
  XG-G6 check --stage 3 covers the geo rung only (train_atoms collect --stage 3 --with-xgb reads its report).
  XG-G7 l6 --stage 3 = the same L-6 run and comparison into the _geo outputs (regression 2 must not rewrite
        results/C_l6_xgb.json); l6 without --stage 3 is unchanged.
  XG-G8 no absolute repo path enters a unit fingerprint; the LADDER result snapshot uses repo-relative keys (F-13).
        Every existing LADDER XGB unit records the pre-GEO2 train_xgb.py sha256 (79ddc291...), so no stage-1 / 2 unit
        matches this file's fingerprint: stage-1 / 2 run / check are unusable here by construction (XG-G9).
  XG-G9 (review fix 2026-10-30) guard_ladder_write: smoke / run / check with --stage 1|2 and l6 without --stage 3 STOP
        (exit 3) before any write or folder removal when CODE_SHA != results/L_xgb_report.json code_sha256 (the code of
        the LADDER XGB results; no such file = a fresh LADDER tree, old behaviour). Before this fix `check --stage 1|2`
        flagged all 30 LADDER units and still overwrote the tracked results/L_xgb_report.json with ok false.
  XG-G10 (review fix 2026-10-30) gate_c_geo = train_atoms GA-14 (C_tests_geo.json code_sha256 of train_atoms.py,
        train_xgb.py, inputs.py, lad_common.py = the files now).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair, fu2 and espley_xtb_repro trees
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")      # ... nor from the joblib workers (they inherit the env)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------- L-4: files this process opens for reading (= train_atoms)
_OPEN_LOG: list = []
_SKIP_PREFIX = tuple(sorted({p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, "/proc", "/sys", "/dev", "/etc",
                                         "/usr", "/lib", "/lib64", "/opt") if p}))
_SKIP_EXT = (".py", ".pyc", ".so", ".pth", ".typed")


def _audit(event, args):
    """'open' audit events of reads outside the interpreter / system trees (never raises)."""
    if event != "open":
        return
    try:
        a = tuple(args) + (None, None, None)
        p, mode, flags = a[0], a[1], a[2]
        if p is None or isinstance(p, int):
            return
        if mode is not None:
            if any(c in str(mode) for c in "wax+"):
                return
        elif isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
            return
        s = os.path.abspath(os.fsdecode(p))
        if s.endswith(_SKIP_EXT) or s.startswith(_SKIP_PREFIX) or "/site-packages/" in s:
            return
        _OPEN_LOG.append(s)
    except Exception:          # noqa: BLE001  (an audit hook must never raise into the caller)
        return


sys.addaudithook(_audit)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402          (paths, write_atomic / write_json / sha256 / assert_path, stop)

FU2_DIR = L.P_DIR / "fu2"
sys.path.insert(0, str(FU2_DIR))
import fu2_common as F  # noqa: E402          (FU2 constants and paths; numpy-level, loads in env_espley)
import tokens as FTok  # noqa: E402           (TOKEN_VERSION, BIN_COLS, NUM_BASE, NUM_EXT; numpy / pandas only)
import atoms as FA  # noqa: E402              (ATOM_VERSION_D1PAIR; numpy / pandas only at import)
import d1_train as D1T  # noqa: E402          (rev 9: Scale, _xgb_fit, _xgb_g, grid_cells)
import train_ml_single as TMS  # noqa: E402   (rev 6: GRIDS["XGB"]; on sys.path through d1pair_common)

C = F.C
CODE_VERSION = "d1-ladder-2"                                  # spec 10 (= train_atoms.py)
MODEL = "XGB"
ARMS = ("S", "SH")
SPLIT, NF = "T1", 5
FOLD_KEYS = ("train", "test", "inner_train", "inner_val")
STAGE_RUNGS = {1: ("dft", "g1"), 2: ("g2",), 3: ("geo",)}     # stage 3 = GEO2 (docs/specs/GEO2.md 8-1, XG-G1)
RUNG_ORDER = ("dft", "g1", "g2", "geo")
GEO_STAGE = 3
TARGETS = list(F.TARGETS)                                     # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
NT = len(TARGETS)
H_TARGETS = [t for t in TARGETS if t != "c_ghost"]            # spec 4-1: the 9 HF values
H_IDX = [TARGETS.index(t) for t in H_TARGETS]
H_PREFIXES = ("H_", "hf_")
CHANGED_DD = ("d1", "d2", "barrier")                          # Q / V change only these (spec 0-1, L-11)
UNITS = L.LAD / "units"
SMOKE = L.LAD / "smoke_xgb"
L6_DIR = L.LAD / "smoke_xgb_l6"
INPUTS = L.LAD / "inputs"
DEFAULT_LABELS = L.LAD / "F_pairs_cr2.csv"
ORIG_LABELS = C.RES / "F_pairs.csv"
FU2_S_SPLITS = F.FU2_RES / "S_splits.json"
S_FOLDS = C.RES / "S_folds.csv"
ADD3_MD = L.RES / "PREREG_D1PAIR_ADD3.md"
ADD3_JSON = L.RES / "prereg_d1pair_add3.json"
# GEO2 stage 3 (docs/specs/GEO2.md 6-1, 8-1; read only here; = train_atoms.py GEO2 constants)
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))
GEO2_RES = L.REPO / "analysis" / "geo2" / "results"
GEO_INPUTS = {"geo": GEO2 / "inputs" / "geo.parquet"}         # the other rungs keep $LAD/inputs/<rung>.parquet
PREREG_GEO2_MD = GEO2_RES / "PREREG_GEO2.md"
PREREG_GEO2_JSON = GEO2_RES / "prereg_geo2.json"
C_TESTS_GEO = GEO2_RES / "C_tests_geo.json"
C_GEO_TRAINER_CODE = ("train_atoms.py", "train_xgb.py", "inputs.py", "lad_common.py")   # = train_atoms (XG-G10)
P_BUILD_GEO = GEO2_RES / "P_build_geo.json"
P_GEO_REPORT = GEO2_RES / "P_geo_report.json"
P_GEO_SETS = GEO2_RES / "P_geo_sets.json"
P_SET_KEYS = ("P_geo", "P_23", "P_023", "P_g2", "P_dft")
SMOKE_GEO = L.LAD / "smoke_xgb_geo"
L6_DIR_GEO = L.LAD / "smoke_xgb_l6_geo"
REV9_UNITS = D1T.UNITS / "T1"                                 # rev 9 cache: T1__C__XGB__<t>__f<k>.{json,parquet}
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                                  # = train_atoms.PRED_COLS (L_predictions.parquet)
RUNGS = {   # spec 4-1 columns: token reports (usable reactions) + the 182 global features of the rung
    "dft": dict(fu2_res=L.RES / "fu2", x182=L.LAD / "X182_d1_dft.parquet", tag="dft"),
    "g1": dict(fu2_res=F.FU2_RES, x182=F.X182, tag=""),
    "g2": dict(fu2_res=L.G2_DIR / "results" / "fu2", x182=L.G2 / "X182_d1_g2.parquet", tag="g2"),
    "geo": dict(fu2_res=GEO2_RES / "fu2_geo", x182=GEO2 / "X182_d1_geo.parquet", tag="geo"),      # GEO2 8-1
}
GRID = D1T.grid_cells("XGB")                                  # 12 cells, sorted-key product (rev 9 order)
ESTIMATOR = repr(TMS.GRIDS["XGB"][0])
N_JOBS = int(os.environ.get("LAD_XGB_NJOBS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
WORKER_PATHS = [str(L.P_DIR), str(C.ESP_DIR)]                 # d1_train, train_ml_single for the joblib workers
# hashed once at import: the code this process runs (fu2 train.py review F3 rule)
CODE_SHA = {k: L.sha256(Path(p).resolve()) for k, p in (
    ("train_xgb.py", __file__), ("lad_common.py", L.__file__), ("d1_train.py", D1T.__file__),
    ("d1pair_common.py", C.__file__), ("train_ml_single.py", TMS.__file__), ("fu2/fu2_common.py", F.__file__),
    ("fu2/tokens.py", FTok.__file__))}
DECISIONS = {k: v for k, v in (
    ("XG-1", "env_espley (rev 9 arm C env); train_atoms.py not importable (torch) -> its shared rules re-implemented"),
    ("XG-2", "jobs via analysis/d1_pair/jobs/py.slurm PYENV=espley (g0f0.py precedent); no new wrapper"),
    ("XG-3", "rev 9 primitives imported (Scale, _xgb_fit, _xgb_g, grid_cells); d1_train.data/masks/prep not called"),
    ("XG-4", "inner GroupKFold(4) by core on the unfiltered outer-train list, then the usable filter; fold 0 = split file"),
    ("XG-5", "one unit = rung x arm x fold, 10 targets tuned independently"),
    ("XG-6", "SH: r = y - beta dH, beta no intercept on the fit pairs, c_ghost 0; score |beta dH + g - y| kcal"),
    ("XG-7", "val_kcal_mae = chosen cell's inner fold 0 score"),
    ("XG-8", "gate M-1' on every fit: final incl. baseline <= 1e-4, model part <= 1e-6"),
    ("XG-9", "yhat_seed_sd = 0.0 (one fit, random_state 42)"),
    ("XG-10", "X182 = spec 4-1 file of the rung, sha256 = T_report x182_sha256"),
    ("XG-11", "ADD3 need = label file + trained rungs' input tables (= train_atoms)"),
    ("XG-12", "smoke: fold 0, cells first + last, real n_estimators, $LAD/smoke_xgb"),
    ("XG-13", "joblib over target x inner fold x cell; fits n_jobs 1 (worker count does not change results)"),
    ("XG-14", "run order rung x arm x fold, [i::n]"),
    ("XG-15", "l6 self-check command for L-6"),
    ("XG-16", "check records json / parquet sha256 per unit (train_atoms collect --with-xgb requires it)"),
    ("XG-17", "run: ADD3 gate skips files of other rungs only; P_report consistency; stage 2 needs C g2_verified"),
    ("XG-18", "M1: XGB array %4 first, then the atom array, one collect"))}
_GATE: dict = {}


# ================================================================ small helpers (= train_atoms.py)
def stop(msg):
    L.stop(msg)


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _abs(p):
    return Path(os.path.abspath(p))


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


def write_json(path, obj):
    return L.write_json(path, _jsonable(obj))


def norm(obj):
    """the value as it reads back from json (tuples -> lists, numpy -> python); fingerprints are compared this way."""
    return json.loads(json.dumps(_jsonable(obj)))


def inputs_path(rung):
    return GEO_INPUTS[rung] if rung in GEO_INPUTS else INPUTS / f"{rung}.parquet"


def res_out(name, stage):
    """results/<name> for stages 1 / 2; results/<stem>_geo<suffix> for stage 3 (XG-G2 = train_atoms.res_out)."""
    p = L.RES / name
    return p.with_name(f"{p.stem}_geo{p.suffix}") if int(stage) == GEO_STAGE else p


def collect_rungs(stage):
    """rungs of check: every rung of the stages <= stage (LADDER); stage 3 = the geo rung only (= train_atoms)."""
    return tuple(STAGE_RUNGS[GEO_STAGE]) if int(stage) == GEO_STAGE else rungs_upto(stage)


def uname(u):
    return f"L__{u['rung']}__{u['arm']}__{MODEL}__f{u['fold']}"


def unit_list(stage=1, rungs=None):
    rungs = STAGE_RUNGS[int(stage)] if rungs is None else rungs
    us = [dict(rung=r, arm=a, model=MODEL, fold=k) for r in rungs for a in ARMS for k in range(NF)]
    return sorted(us, key=lambda u: (RUNG_ORDER.index(u["rung"]), ARMS.index(u["arm"]), u["fold"]))


def rungs_upto(stage):
    return tuple(r for s in sorted(STAGE_RUNGS) if s <= int(stage) for r in STAGE_RUNGS[s])


def opened_summary(paths):
    """L-4 record: per directory the number of files opened; the file names outside directories with > 20 files."""
    c = Counter(os.path.dirname(p) for p in paths)
    big = {d for d, n in c.items() if n > 20}
    return dict(dirs=dict(sorted(c.items())), files=sorted({p for p in paths if os.path.dirname(p) not in big}))


def _ids(D, rows):
    return [str(q) for q in D.pid[np.asarray(rows, dtype=int)]]


def _num(df):
    """DataFrame -> float64 array; None / NA -> NaN (pd.to_numeric), anything non-numeric raises."""
    return df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def _maxdiff(a, b):
    """max |a - b| with equal NaN patterns required (inf when the NaN patterns differ)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


def versions():
    from importlib.metadata import version
    out = {k: version(k) for k in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib", "pyarrow")}
    out["python"] = platform.python_version()
    return out


def cost(cell):
    return float(cell["n_estimators"]) * float(cell["max_depth"])


# ================================================================ inputs
def versions_expected(rung):
    tag = RUNGS[rung]["tag"]
    sfx = f"-{tag}" if tag else ""
    return FTok.TOKEN_VERSION + sfx, FA.ATOM_VERSION_D1PAIR + sfx


def load_meta(rung):
    """the rung's token reports (= train_atoms.load_tokens checks, without loading the npz files: the atom models verify
    every token sha256) + the X182 file tied to them -> token reaction ids in T_manifest row order."""
    cfg = RUNGS[rung]
    res = Path(cfg["fu2_res"])
    tpath, mpath, qpath, spath = (res / n for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json"))
    miss = [str(p) for p in (tpath, mpath, qpath, spath) if not p.is_file()]
    if miss:
        not_ready(f"rung {rung}: token files missing {miss}")
    trep, qrep = json.loads(tpath.read_text()), json.loads(qpath.read_text())
    if trep.get("stop") or qrep.get("stop"):
        stop(f"rung {rung}: T_report / Q_report stop")
    tv, av = versions_expected(rung)
    num_want = list(FTok.NUM_BASE) + (list(FTok.NUM_EXT) if trep.get("ext_used") else [])
    sha = dict(t_report=L.sha256(tpath), t_manifest=L.sha256(mpath), q_report=L.sha256(qpath),
               s_splits_rung=L.sha256(spath))
    checks = (("token_version", trep.get("token_version") == tv), ("atom_version", qrep.get("atom_version") == av),
              ("q_report_sha256", trep.get("q_report_sha256") == sha["q_report"]),
              ("splits_sha256 = rung S_splits.json", trep.get("splits_sha256") == sha["s_splits_rung"]),
              ("splits_sha256 = FU2 S_splits.json", trep.get("splits_sha256") == L.sha256(FU2_S_SPLITS)),
              ("manifest_sha256", trep.get("manifest_sha256") == sha["t_manifest"]),
              ("bin_cols", list(trep.get("bin_cols") or []) == list(FTok.BIN_COLS)),
              ("num_cols", list(trep.get("num_cols") or []) == num_want),
              ("glob_cols 182", len(trep.get("glob_cols") or []) == 182))
    bad = [k for k, ok in checks if not ok]
    if bad:
        stop(f"rung {rung}: token report does not match the current inputs / expected layout: {bad}")
    x182 = _abs(cfg["x182"])
    ov = (qrep.get("overrides") or {}).get("FU2_X182")
    if ov is not None and _abs(ov) != x182:
        stop(f"rung {rung}: Q_report overrides FU2_X182 {ov} != {x182} (spec 4-1)")
    if ov is None and x182 != _abs(F.X182):
        stop(f"rung {rung}: Q_report has no FU2_X182 override, so its tokens were built from {F.X182}, not {x182}")
    if not x182.is_file():
        not_ready(f"rung {rung}: {x182} missing")
    xsha = L.sha256(x182)
    if xsha != trep.get("x182_sha256"):
        stop(f"rung {rung}: {x182} sha256 {xsha[:12]} != T_report x182_sha256 {str(trep.get('x182_sha256'))[:12]}")
    man = pd.read_csv(mpath)
    rids = man.rxn_id.astype(int).to_numpy()
    if len(set(rids.tolist())) != len(rids):
        stop(f"{mpath}: duplicated rxn_id")
    return dict(trep=trep, rids=rids, row={int(r): k for k, r in enumerate(rids)}, x182=x182,
                glob_cols=list(trep["glob_cols"]), man_sha=dict(zip(rids.tolist(), man.sha256.astype(str))),
                sha=dict(sha, x182=xsha),
                inputs=dict(t_report=str(tpath), t_manifest=str(mpath), q_report=str(qpath), s_splits_rung=str(spath),
                            x182=str(x182), n_tokens=int(len(rids))))


def load_x182(meta):
    """the rung's X182 parquet -> [token rows, 182] float64 in T_report glob_cols (= parquet) order; NaN rows where absent."""
    X = pd.read_parquet(meta["x182"])
    if "rxn_id" not in X.columns:
        stop(f"{meta['x182']}: no rxn_id column")
    X = X.assign(rxn_id=X.rxn_id.astype(int))
    if X.rxn_id.duplicated().any():
        stop(f"{meta['x182']}: duplicated rxn_id")
    X = X.set_index("rxn_id")
    if list(X.columns) != meta["glob_cols"]:
        stop(f"{meta['x182']}: columns differ from T_report glob_cols (order included)")
    return X.reindex(meta["rids"]).to_numpy(dtype=float)


def load_inputs(rung, meta, A):
    """$LAD/inputs/<rung>.parquet -> H 9 per token row (NaN where absent), usable reactions, info (= train_atoms.load_inputs;
    the optional 182 table columns are compared with the X182 file rows A, which are the token glob by T_report)."""
    path = inputs_path(rung)
    if not path.is_file():
        not_ready(f"{path} missing (Phase P)")
    X = pd.read_parquet(path)
    if "rxn_id" not in X.columns:
        stop(f"{path}: no rxn_id column (columns {list(X.columns)[:30]})")
    X = X.assign(rxn_id=X.rxn_id.astype(int))
    if X.rxn_id.duplicated().any():
        stop(f"{path}: duplicated rxn_id")
    sets = [pre for pre in H_PREFIXES if all(f"{pre}{t}" in X.columns for t in H_TARGETS)]
    if len(sets) != 1:
        stop(f"{path}: need exactly one complete set of H columns {[p + '<t>' for p in H_PREFIXES]} for {H_TARGETS}; "
             f"found sets {sets}; columns {list(X.columns)[:40]} ... (Phase P contract, train_atoms.py docstring)")
    hcols = [f"{sets[0]}{t}" for t in H_TARGETS]
    X = X.set_index("rxn_id")
    rids, nr = meta["rids"], len(meta["rids"])
    common = [int(r) for r in rids if int(r) in X.index]
    rows = np.array([meta["row"][r] for r in common], dtype=int)
    H9 = np.full((nr, len(H_TARGETS)), np.nan)
    in_tab = np.zeros(nr, dtype=bool)
    H9[rows] = _num(X.loc[common, hcols])
    in_tab[rows] = True
    ok182 = np.ones(nr, dtype=bool)
    gcols = meta["glob_cols"]
    x182 = "columns absent: not cross-checked"
    if all(c in X.columns for c in gcols):
        G = _num(X.loc[common, gcols])
        fin = np.isfinite(G).all(1)
        ok182[rows] = fin
        g0 = A[rows]
        dev = np.abs(G - g0) > 1e-12 * np.maximum(1.0, np.abs(g0))
        nbad = int((dev & np.isfinite(G)).any(1).sum())
        if nbad:
            stop(f"{path}: {nbad} reactions whose 182 columns differ from {meta['x182']} (> 1e-12 relative)")
        x182 = "equal to the rung X182 file (= token glob)"
    tsha = "column absent: not cross-checked"
    for col in ("token_sha256", "tok_sha256"):          # train_atoms contract name / the name inputs.py writes
        if col in X.columns:
            bad = [r for r in common if str(X.at[r, col]) != meta["man_sha"][r]]
            if bad:
                stop(f"{path}: {col} differs from T_manifest for {len(bad)} reactions, e.g. {bad[:5]}")
            tsha = f"{col} equal to T_manifest"
    usable = in_tab & np.isfinite(H9).all(1) & ok182
    uchk = "column absent: not cross-checked"
    if "usable" in X.columns:
        theirs = np.zeros(nr, dtype=bool)
        theirs[rows] = X.loc[common, "usable"].astype(bool).to_numpy()
        if (theirs != usable).any():
            stop(f"{path}: 'usable' differs from token + table row + finite H (+182) for "
                 f"{int((theirs != usable).sum())} token reactions")
        uchk = "equal"
    info = dict(path=str(path), sha256=L.sha256(path), n_rows=int(len(X)), n_token_rows_in_table=len(common),
                n_table_rows_without_token=int(len(X) - len(common)), h_cols=hcols, x182_check=x182,
                token_sha256_check=tsha, usable_check=uchk, n_usable=int(usable.sum()),
                n_nonfinite_h=int((in_tab & ~np.isfinite(H9).all(1)).sum()))
    return H9, usable, info


def label_provenance(path, sha):
    """= train_atoms.label_provenance: a passed Q / V output or the original D1PAIR file (tests: others, recorded)."""
    if path == _abs(ORIG_LABELS):
        want = json.loads(FU2_S_SPLITS.read_text()).get("inputs_sha256", {}).get("F_pairs")
        if sha != want:
            stop(f"{path} sha256 {sha[:12]} != FU2 S_splits.json inputs_sha256.F_pairs {str(want)[:12]}")
        return "D1PAIR F_pairs.csv (sha256 = FU2 S_splits.json inputs_sha256.F_pairs)"
    reports = {_abs(L.LAD / "F_pairs_cr2.csv"): L.RES / "V_report.json",
               _abs(L.LAD / "F_pairs_cr.csv"): L.RES / "Q_report.json"}
    rp = reports.get(path)
    if rp is None:
        return "unverified (not a Q / V output, not the original; tests only — the ADD3 gate covers production)"
    rep = json.loads(rp.read_text()) if rp.is_file() else {}
    if not rep.get("all_pass"):
        stop(f"{path}: {rp.name} is missing or not all_pass")
    rec = (rep.get("outputs") or {}).get(str(L.LAD / path.name))
    if rec != sha:
        stop(f"{path}: sha256 differs from {rp.name} outputs ({str(rec)[:12]} vs {sha[:12]})")
    return f"{rp.name} all_pass, outputs sha256 equal"


def load_labels(labels):
    """= train_atoms.load_labels: label file -> pair table (index pair_id) + info; L-11 invariant on every load."""
    path = _abs(labels)
    if not path.is_file():
        not_ready(f"label file {path} missing")
    sha = L.sha256(path)
    prov = label_provenance(path, sha)
    P = pd.read_csv(path)
    need = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos"] + [f"dd_{t}" for t in TARGETS]
    miss = [c for c in need if c not in P.columns]
    if miss:
        stop(f"{path}: columns missing {miss}")
    if P.pair_id.duplicated().any():
        stop(f"{path}: duplicated pair_id")
    l11 = "original labels"
    if path != _abs(ORIG_LABELS):
        O = pd.read_csv(ORIG_LABELS)
        if list(P.columns) != list(O.columns) or P.pair_id.tolist() != O.pair_id.tolist():
            stop(f"{path}: columns / pair order differ from {ORIG_LABELS}")
        ch = {f"dd_{t}" for t in CHANGED_DD}
        other = [c for c in O.columns if not c.startswith("dd_") and not P[c].equals(O[c])]
        dmax = {t: _maxdiff(P[f"dd_{t}"], O[f"dd_{t}"]) for t in TARGETS if f"dd_{t}" not in ch}
        if other or max(dmax.values()) > 1e-12:
            stop(f"{path}: differs from F_pairs.csv outside dd_d1 / dd_d2 / dd_barrier: columns {other}, dd {dmax}")
        l11 = dict(non_dd_columns_identical=True, unchanged_dd_max_abs=dmax)
    P = P.assign(pair_id=P.pair_id.astype(str), ref_kind=P.ref_kind.fillna("").astype(str),
                 pos=P.pos.fillna("").astype(str), a_rxn=P.a_rxn.astype(int), b_rxn=P.b_rxn.astype(int))
    return P.set_index("pair_id"), dict(path=str(path), sha256=sha, provenance=prov, l11=l11, n_pairs=int(len(P)))


_FOLDS: dict = {}


def load_folds():
    """the FU2 T1 split files (sha256 = FU2 S_splits.json; rung independent) + S_folds.csv (sha256 = S_splits
    inputs_sha256.S_folds): core per pair (the rev 9 inner GroupKFold groups) and the d1_train.masks arm C check."""
    if _FOLDS:
        return _FOLDS
    srep = json.loads(FU2_S_SPLITS.read_text())
    s_sha = L.sha256(S_FOLDS)
    if s_sha != (srep.get("inputs_sha256") or {}).get("S_folds"):
        stop(f"{S_FOLDS} sha256 differs from {FU2_S_SPLITS} inputs_sha256.S_folds")
    S = pd.read_csv(S_FOLDS)
    for col in ("in_T_main", "in_P_sens", "in_A_main"):
        if S[col].dtype != bool:
            stop(f"{S_FOLDS}: column {col} is {S[col].dtype}, not bool")
    if S.pair_id.duplicated().any() or S.core.isna().any():
        stop(f"{S_FOLDS}: duplicated pair_id or missing core")
    S = S.assign(pair_id=S.pair_id.astype(str))
    folds, files = {}, {}
    for k in range(NF):
        name = f"{SPLIT}__f{k}.json"
        p = F.SPLITS / name
        if not p.is_file() or L.sha256(p) != srep["units"][name]["sha256"]:
            stop(f"{p} missing or changed after {FU2_S_SPLITS}")
        rec = json.loads(p.read_text())
        # d1_train.masks(u), u = T1 arm C fold k: test = fold == k & (P_sens | A_main), train = fold != k & T_main (S order)
        te = S.loc[(S.fold_T1 == k) & (S.in_P_sens | S.in_A_main), "pair_id"].tolist()
        tr = S.loc[(S.fold_T1 != k) & S.in_T_main, "pair_id"].tolist()
        if tr != list(rec["train"]) or te != list(rec["test"]):
            stop(f"{p}: train / test lists differ from the d1_train.masks arm C rule on {S_FOLDS}")
        folds[k] = rec
        files[f"split_T1_f{k}"] = str(p)
    _FOLDS.update(srep=srep, folds=folds, files=files, core=dict(zip(S.pair_id, S.core)),
                  s_splits_sha=L.sha256(FU2_S_SPLITS), s_folds_sha=s_sha)
    return _FOLDS


_BASE: dict = {}


def rung_base(rung, labels):
    """everything of a rung that both arms share (cached per process; = train_atoms.rung_base without the tokens)."""
    key = (rung, str(_abs(labels)))
    if key in _BASE:
        return _BASE[key]
    if rung not in RUNGS:
        stop(f"unknown rung {rung}")
    t0, i0 = time.time(), len(_OPEN_LOG)
    meta = load_meta(rung)
    A = load_x182(meta)
    H9, ur, hinfo = load_inputs(rung, meta, A)
    nofeat = ur & ~np.isfinite(A).all(1)
    if nofeat.any():
        stop(f"rung {rung}: {int(nofeat.sum())} usable reactions without a finite row in {meta['x182']}, e.g. "
             f"{meta['rids'][nofeat][:5].tolist()} (P_{rung} would differ from the atom models')")
    P, linfo = load_labels(labels)
    fo = load_folds()
    fold_ids = {q for f in fo["folds"].values() for k in FOLD_KEYS for q in f[k]}
    miss = sorted(fold_ids - set(P.index))
    if miss:
        stop(f"{len(miss)} split pair ids are not in {linfo['path']}: {miss[:5]}")
    y = P[[f"dd_{t}" for t in TARGETS]].to_numpy(float)
    row = meta["row"]
    ra = np.array([row.get(int(r), -1) for r in P.a_rxn])
    rb = np.array([row.get(int(r), -1) for r in P.b_rxn])
    has_tok = (ra >= 0) & (rb >= 0)
    usable_pair = has_tok & ur[np.maximum(ra, 0)] & ur[np.maximum(rb, 0)]
    H10 = np.zeros((len(meta["rids"]), NT))
    for jh, j in enumerate(H_IDX):
        H10[:, j] = np.where(ur, H9[:, jh], 0.0)                  # c_ghost column stays 0 (beta 0)
    dH = np.zeros_like(y)
    dH[usable_pair] = H10[ra[usable_pair]] - H10[rb[usable_pair]]
    pid = P.index.to_numpy()
    core = np.array([fo["core"].get(q) for q in pid], dtype=object)
    if any(core[k] is None for k in range(len(pid)) if pid[k] in fold_ids):
        stop(f"a split pair has no core in {S_FOLDS}")
    in_folds = np.isin(pid, sorted({q for f in fo["folds"].values() for q in f["train"] + f["test"]}))
    pair_list = sorted(pid[usable_pair & in_folds].tolist())
    inputs = dict(meta["inputs"], inputs_table=hinfo["path"], labels=linfo["path"], labels_original_l11=(
        str(ORIG_LABELS) if linfo["path"] != str(_abs(ORIG_LABELS)) else None), split_s_splits=str(FU2_S_SPLITS),
        split_s_folds=str(S_FOLDS), **fo["files"])
    sha = dict(meta["sha"], inputs_table=hinfo["sha256"], labels=linfo["sha256"], s_splits=fo["s_splits_sha"],
               s_folds=fo["s_folds_sha"])
    b = dict(rids=meta["rids"], glob_cols=meta["glob_cols"], x182=A, H9=H9, H10=H10, usable_rxn=ur, hinfo=hinfo, P=P,
             linfo=linfo, folds=fo["folds"], y=y, pid=pid, pid_row={p: k for k, p in enumerate(pid)}, core=core, ra=ra,
             rb=rb, has_tok=has_tok, usable_pair=usable_pair, dH=dH, pair_list=pair_list,
             pair_list_sha=C.list_sha([(q,) for q in pair_list]), inputs=inputs, sha=sha, labels_path=linfo["path"],
             opened=list(_OPEN_LOG[i0:]), load_seconds=time.time() - t0)
    print(f"[data] rung {rung}: {len(meta['rids'])} token reactions, {hinfo['n_usable']} usable, {len(pair_list)} pairs "
          f"in P_{rung} ({int(usable_pair.sum())} usable of {len(P)}), labels {Path(linfo['path']).name} "
          f"({b['load_seconds']:.0f} s)", flush=True)
    _BASE[key] = b
    return b


class XData:
    """one rung x arm: pair table, y, reaction rows (T_manifest order) of a / b, features X (S: 182; SH: 182 + H 9), H10 /
    dH (SH baseline), the usable-pair filter and the split lists (= train_atoms.LadData without the tokens)."""

    _KEYS = ("rids", "glob_cols", "H10", "usable_rxn", "hinfo", "P", "linfo", "folds", "y", "pid", "pid_row", "core",
             "ra", "rb", "has_tok", "usable_pair", "dH", "pair_list", "pair_list_sha", "inputs", "sha", "opened")

    def __init__(self, rung, arm, labels=DEFAULT_LABELS):
        if arm not in ARMS:
            stop(f"unknown arm {arm}")
        b = rung_base(rung, labels)
        for k in self._KEYS:
            setattr(self, k, b[k])
        self.rung, self.arm, self.labels = rung, arm, b["labels_path"]
        if arm == "S":
            self.X, cols = b["x182"], list(b["glob_cols"])
        else:
            self.X = np.concatenate([b["x182"], b["H10"][:, H_IDX]], 1)
            cols = list(b["glob_cols"]) + [f"H_{t}" for t in H_TARGETS]
        self.feat_cols = cols
        self.feat_cols_sha = C.list_sha([(c,) for c in cols])
        self._splits = {}

    def split(self, a, b=None):
        """split(k) or split("T1", k) -> {train, test, inner_train, inner_val (usable pair-table rows), inner (4 rev 9
        inner folds after the filter), dropped, list sha256}; leak checks (= train_atoms LadData.split + inner folds)."""
        split, k = (SPLIT, a) if b is None else (a, b)
        if split != SPLIT:
            stop(f"split {split}: LADDER uses T1 only (spec 4-3)")
        k = int(k)
        if k in self._splits:
            return self._splits[k]
        rec = self.folds[k]
        out, dropped = {}, {}
        for key in FOLD_KEYS:
            rows = np.array([self.pid_row[q] for q in rec[key]], dtype=int)
            keep = self.usable_pair[rows]
            out[key] = rows[keep]
            dropped[key] = dict(no_tokens=int((~self.has_tok[rows]).sum()), not_usable=int((~keep).sum()))
        st = {key: set(out[key].tolist()) for key in FOLD_KEYS}
        if st["test"] & st["train"]:
            stop(f"{self.rung} fold {k}: test pair in train")
        if st["inner_train"] | st["inner_val"] != st["train"] or st["inner_train"] & st["inner_val"]:
            stop(f"{self.rung} fold {k}: inner folds do not partition train")
        rx = lambda r: set(self.ra[r].tolist()) | set(self.rb[r].tolist())          # noqa: E731
        if rx(out["test"]) & rx(out["train"]):
            stop(f"{self.rung} fold {k}: test reaction in train")
        # rev 9 d1_train.inner_folds: GroupKFold(4), group = core, on the unfiltered outer-train list (S order)
        from sklearn.model_selection import GroupKFold
        tr_all = np.array([self.pid_row[q] for q in rec["train"]], dtype=int)
        inner = []
        for j, (ia, ib) in enumerate(GroupKFold(C.N_INNER).split(tr_all, groups=self.core[tr_all])):
            itr_all, iva_all = tr_all[ia], tr_all[ib]
            if j == 0 and (_ids(self, itr_all) != list(rec["inner_train"]) or _ids(self, iva_all) != list(rec["inner_val"])):
                stop(f"{self.rung} fold {k}: GroupKFold(4) fold 0 differs from the split file inner_train / inner_val")
            if set(self.core[itr_all].tolist()) & set(self.core[iva_all].tolist()):
                stop(f"{self.rung} fold {k} inner {j}: a core on both sides")
            itr, iva = itr_all[self.usable_pair[itr_all]], iva_all[self.usable_pair[iva_all]]
            if not len(itr) or not len(iva) or set(itr.tolist()) & set(iva.tolist()):
                stop(f"{self.rung} fold {k} inner {j}: empty or overlapping inner sides")
            inner.append(dict(train=itr, val=iva, dropped=dict(train=int(len(itr_all) - len(itr)),
                                                              val=int(len(iva_all) - len(iva)))))
        if not (np.array_equal(inner[0]["train"], out["inner_train"]) and np.array_equal(inner[0]["val"], out["inner_val"])):
            stop(f"{self.rung} fold {k}: filtered inner fold 0 differs from the filtered split file lists (L-9)")
        ids = {key: _ids(self, out[key]) for key in FOLD_KEYS}
        out.update(inner=inner, dropped=dropped,
                   pairs_sha=C.list_sha([(q,) for q in ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"]
                                         + ["|"] + ids["inner_val"]]),
                   val_sha=C.list_sha([(q,) for q in ids["inner_val"]]), test_sha=C.list_sha([(q,) for q in ids["test"]]),
                   inner_sha=C.list_sha([(q,) for f in inner for q in _ids(self, f["train"]) + ["|"] + _ids(self, f["val"])
                                         + ["||"]]))
        self._splits[k] = out
        return out


_DATA: dict = {}


def get_data(rung, arm, labels=DEFAULT_LABELS):
    key = (rung, arm, str(_abs(labels)))
    if key not in _DATA:
        _DATA[key] = XData(rung, arm, labels)
    return _DATA[key]


# ================================================================ fits (spec 4-2; rev 9 d1_train.prep / _xgb_cell / fit_final)
def beta_fit(dH, y):
    """per target beta_t = sum(dH_t * y_t) / sum(dH_t^2) (no intercept); c_ghost 0. dH, y [n, 10] (fit pairs only).
    = train_atoms.beta_fit (copied: train_atoms imports torch)."""
    dH, y = np.asarray(dH, float), np.asarray(y, float)
    num, den = (dH * y).sum(0), (dH * dH).sum(0)
    beta = np.zeros(NT)
    for j in H_IDX:
        if not (np.isfinite(den[j]) and den[j] > 0 and np.isfinite(num[j])):
            stop(f"beta for {TARGETS[j]}: sum(dH^2) = {den[j]} on the fit pairs")
        beta[j] = num[j] / den[j]
    return beta


def prep_fit(D, rows, evals, gate_rows):
    """rev 9 d1_train.prep for one fit on pair-table rows `rows` (its training pairs), per target: beta (SH) on their dH / y,
    r = y - beta dH (S, c_ghost: r = y), d1_train.Scale on their x with s_y = std(r), ys = r / s_y; z and baseline of each
    eval set; z and baseline of the gate pairs as (a, b), (b, a), (a, a) recomputed from x."""
    rows = np.asarray(rows, dtype=int)
    xa, xb = D.X[D.ra[rows]], D.X[D.rb[rows]]
    y, dH = D.y[rows], D.dH[rows]
    beta = beta_fit(dH, y) if D.arm == "SH" else np.zeros(NT)
    on = [D.arm == "SH" and j in H_IDX for j in range(NT)]
    R = [y[:, j] - beta[j] * dH[:, j] if on[j] else y[:, j].copy() for j in range(NT)]
    scs = [D1T.Scale(xa, xb, r) for r in R]
    sc = scs[0]
    for s in scs[1:]:
        if not (np.array_equal(s.kd, sc.kd) and np.array_equal(s.km, sc.km)):
            stop("Scale columns differ between targets of one fit")
    sy = [float(s.sy) for s in scs]
    if not all(np.isfinite(v) and v > 0 for v in sy):
        stop(f"s_y not positive / finite on the fit pairs: {sy}")
    zd, zm = sc.z(xa, xb)
    ev, base = {}, {}
    for name, er in evals.items():
        er = np.asarray(er, dtype=int)
        ev[name] = sc.z(D.X[D.ra[er]], D.X[D.rb[er]])
        base[name] = np.stack([beta[j] * D.dH[er, j] if on[j] else np.zeros(len(er)) for j in range(NT)], 1)
    ga, gb = D.ra[gate_rows], D.rb[gate_rows]
    for name, (p, q) in (("ab", (ga, gb)), ("ba", (gb, ga)), ("aa", (ga, ga))):
        ev[name] = sc.z(D.X[p], D.X[q])
        base[name] = np.stack([beta[j] * (D.H10[p, j] - D.H10[q, j]) if on[j] else np.zeros(len(p)) for j in range(NT)], 1)
    return dict(rows=rows, beta_rows=rows, beta=beta, sy=sy, ys=[R[j] / sy[j] for j in range(NT)], zd=zd, zm=zm, evals=ev,
                base=base, rxn_rows=np.unique(np.r_[D.ra[rows], D.rb[rows]]),
                scale=dict(n_dropped_delta=int((~sc.kd).sum()), n_dropped_xbar=int((~sc.km).sum())))


def leak_check(D, pf, others, test_rows):
    """L-2 / L-8 (spec 16 M): beta / scaling pairs = the fit's training pairs, disjoint from its validation / test pairs; no
    test reaction among the fit's reactions."""
    used = set(np.asarray(pf["rows"]).tolist())
    if pf["beta_rows"] is not pf["rows"] or set(np.asarray(pf["beta_rows"]).tolist()) != used:
        stop("L-2: beta pairs differ from the fit's training pairs")
    for o in others:
        if used & set(np.asarray(o).tolist()):
            stop("L-2 / L-8: a validation / test pair is among the fit's scaling / beta pairs")
    trx = set(D.ra[test_rows].tolist()) | set(D.rb[test_rows].tolist())
    if trx & set(np.asarray(pf["rxn_rows"]).tolist()):
        stop("L-8: a test reaction is among the fit's scaling reactions")
    out = dict(n_fit_pairs=len(used), n_scale_rxns=int(len(pf["rxn_rows"])), ok=True)
    if D.arm == "SH":
        out["beta_pairs_sha256"] = C.list_sha([(q,) for q in _ids(D, pf["beta_rows"])])
    return out


def _task(paths, zd, zm, ys, sy, params, evals):
    """one rev 9 XGB fit (joblib worker): d1_train._xgb_fit on (zd, zm, ys) with the cell `params`, then for every eval set
    (zd', zm') the kcal prediction d1_train._xgb_g(m, zd', zm') * s_y (float32 * float as rev 9, returned as float64)."""
    import sys as _s
    import time as _t
    _s.dont_write_bytecode = True
    for p in paths:
        if p not in _s.path:
            _s.path.insert(0, p)
    import numpy as _np
    import d1_train as _D1T
    t0 = _t.time()
    m = _D1T._xgb_fit(zd, zm, ys, params)
    out = {k: _np.asarray(_D1T._xgb_g(m, a, b) * sy, dtype=float) for k, (a, b) in evals.items()}
    return out, _t.time() - t0


def run_parallel(calls):
    """calls = [(zd, zm, ys, sy, params, evals)] -> [(kcal dict, seconds)] in order (worker count does not change results)."""
    if N_JOBS <= 1:
        return [_task(WORKER_PATHS, *c) for c in calls]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=N_JOBS, max_nbytes="200K")(delayed(_task)(WORKER_PATHS, *c) for c in calls)


def gate_eval(kc, base, j):
    """gate M-1' of one fit and target: final prediction (baseline + model, kcal) and the model part alone."""
    kab, kba, kaa = kc["ab"], kc["ba"], kc["aa"]
    gab, gba, gaa = base["ab"][:, j] + kab, base["ba"][:, j] + kba, base["aa"][:, j] + kaa
    anti, selfm = float(np.abs(gab + gba).max()), float(np.abs(gaa).max())
    m_anti, m_self = float(np.abs(kab + kba).max()), float(np.abs(kaa).max())
    ok = anti <= F.ANTISYM_TOL and selfm <= F.ANTISYM_TOL and m_anti <= C.ANTISYM_TOL and m_self <= C.ANTISYM_TOL
    return dict(n=int(len(kab)), anti_max=anti, self_max=selfm, model_anti_max=m_anti, model_self_max=m_self, ok=bool(ok))


# ================================================================ gates / provenance (= train_atoms.py)
def _foreign_to(path, rungs):
    """= train_atoms._foreign_to: `path` belongs to another rung's spec 4-1 column for every rung of this process."""
    import inputs as IN
    return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)


def gate_add3(need, rungs=()):
    """spec 9 (= train_atoms.gate_add3 = d1_split.gate_add1 / fu2 train.gate_add2); exit 3 otherwise. need = absolute input
    paths that prereg_d1pair_add3.json inputs_sha256 must list; files of other rungs' columns only are not opened (L-4)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (ADD3_MD, ADD3_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(ADD3_JSON.read_text())
        if L.sha256(ADD3_MD) != rec.get("md_sha256"):
            bad.append(f"{ADD3_MD.name} sha256 differs from {ADD3_JSON.name} md_sha256")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need) - set(ins))
        if lack:
            bad.append(f"{ADD3_JSON.name} inputs_sha256 lacks {lack}")
        skipped = []
        for path, s in ins.items():
            if path not in need and _foreign_to(path, rungs):
                skipped.append(path)
                continue
            if not Path(path).is_file() or L.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
        _GATE["add3_not_opened_other_rungs"] = skipped
    if bad:
        print("PREREG ADD3 gate (spec 9: commit the addendum before any Phase M training):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        sys.exit(3)
    _GATE["add3"] = L.sha256(ADD3_MD)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def gate_p(rungs, labels):
    """= train_atoms.gate_p (MA-16): P_report all_pass, rung tables + label file sha256 = what inputs.py pairs recorded."""
    p = L.RES / "P_report.json"
    if not p.is_file():
        not_ready(f"{p} missing (inputs.py pairs)")
    rep = json.loads(p.read_text())
    if rep.get("running"):
        not_ready(f"{p} is being written")
    bad = [] if rep.get("all_pass") and not rep.get("stop") else [f"P_report all_pass {rep.get('all_pass')}"]
    for r in rungs:
        want = ((rep.get("rungs") or {}).get(r) or {}).get("table_sha256")
        if want is None or not inputs_path(r).is_file() or L.sha256(inputs_path(r)) != want:
            bad.append(f"{inputs_path(r)} sha256 != P_report rungs.{r}.table_sha256 ({want})")
    lp = str(_abs(labels))
    lab = (rep.get("labels") or {}).get(lp) or {}
    if not lab.get("ids_ok") or not Path(lp).is_file() or L.sha256(lp) != lab.get("sha256"):
        bad.append(f"label file {lp}: sha256 / ids differ from P_report labels ({lab.get('sha256')})")
    if bad:
        stop("Phase P state changed since results/P_report.json (rerun inputs.py build / pairs): " + "; ".join(bad))
    return dict(sha256=L.sha256(p), rungs=list(rungs), labels=lp)


def gate_c_stage2():
    """= train_atoms.gate_c_stage2 (MA-17): stage 2 needs results/C_tests.json all_pass with g2_verified."""
    c = json.loads((L.RES / "C_tests.json").read_text()) if (L.RES / "C_tests.json").is_file() else {}
    if not (c.get("all_pass") and c.get("g2_verified")):
        stop(f"stage 2 needs results/C_tests.json all_pass with g2_verified (rerun tests with the g2 table): "
             f"all_pass {c.get('all_pass')}, g2_verified {c.get('g2_verified')}")
    return dict(c_tests_sha256=L.sha256(L.RES / "C_tests.json"))


# ---------------------------------------------------------------- GEO2 stage 3 gates (= train_atoms.py GA-4 .. GA-7)
def ladder_results_changed(rec):
    """= train_atoms.ladder_results_changed: snapshot paths (repo-relative) whose file is missing or changed now."""
    snap = rec.get("ladder_results_sha256")
    if not isinstance(snap, dict) or not snap:
        return ["prereg_geo2.json has no ladder_results_sha256 snapshot"]
    bad = []
    for rel, s in sorted(snap.items()):
        p = L.REPO / rel
        if not p.is_file() or L.sha256(p) != s:
            bad.append(rel)
    return bad


def gate_prereg_geo2(need, rungs=()):
    """= train_atoms.gate_prereg_geo2 (gate_add3 rule on PREREG_GEO2.md + prereg_geo2.json, json rungs = ["geo"],
    inputs_sha256 ⊇ need, listed files unchanged, LADDER result snapshot unchanged); exit 3 otherwise."""
    def git(*a):
        return subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (PREREG_GEO2_MD, PREREG_GEO2_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(PREREG_GEO2_JSON.read_text())
        if L.sha256(PREREG_GEO2_MD) != rec.get("md_sha256"):
            bad.append(f"{PREREG_GEO2_MD.name} sha256 differs from {PREREG_GEO2_JSON.name} md_sha256")
        if list(rec.get("rungs") or []) != list(STAGE_RUNGS[GEO_STAGE]):
            bad.append(f"{PREREG_GEO2_JSON.name} rungs {rec.get('rungs')} != {list(STAGE_RUNGS[GEO_STAGE])}")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need) - set(ins))
        if lack:
            bad.append(f"{PREREG_GEO2_JSON.name} inputs_sha256 lacks {lack}")
        skipped = []
        for path, s in ins.items():
            if path not in need and _foreign_to(path, rungs):
                skipped.append(path)
                continue
            if not Path(path).is_file() or L.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
        _GATE["prereg_geo2_not_opened_other_rungs"] = skipped
        changed = ladder_results_changed(rec)
        if changed:
            bad.append(f"LADDER result files changed since the pre-registration (spec 14): {changed[:8]}")
    if bad:
        print("PREREG_GEO2 gate (GEO2 spec 8-3: commit PREREG_GEO2.md + prereg_geo2.json before any stage-3 training):"
              "\n  " + "\n  ".join(bad), file=sys.stderr, flush=True)
        sys.exit(3)
    _GATE["prereg_geo2"] = L.sha256(PREREG_GEO2_MD)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def gate_p_geo(rungs, labels):
    """= train_atoms.gate_p_geo: P_build_geo.json all_pass + table sha256, P_geo_report.json stop false, P_geo_sets.json
    lists + |P_23| >= 0.98 |P_g2|, label file = results/P_report.json labels entry; exit 2 / 3."""
    for p in (P_BUILD_GEO, P_GEO_REPORT, P_GEO_SETS, L.RES / "P_report.json"):
        if not p.is_file():
            not_ready(f"{p} missing (GEO2 Phase P: feat_geo.py p geo / p_report)")
    b = json.loads(P_BUILD_GEO.read_text())
    pr = json.loads(P_GEO_REPORT.read_text())
    if b.get("running") or pr.get("running"):
        not_ready(f"{P_BUILD_GEO.name} / {P_GEO_REPORT.name} is being written")
    bad = [] if b.get("all_pass") and not b.get("stop") else [f"P_build_geo all_pass {b.get('all_pass')}, stop "
                                                              f"{b.get('stop')}"]
    for r in rungs:
        t = inputs_path(r)
        want = (b.get("outputs") or {}).get(str(t))
        if want is None or not t.is_file() or L.sha256(t) != want:
            bad.append(f"{t} sha256 != {P_BUILD_GEO.name} outputs ({want})")
    if pr.get("stop"):
        bad.append(f"{P_GEO_REPORT.name} stop: {pr.get('stop_reasons') or pr.get('stop_reason')}")
    sets = json.loads(P_GEO_SETS.read_text())
    miss = [k for k in P_SET_KEYS if not isinstance(sets.get(k), list)]
    if miss:
        bad.append(f"{P_GEO_SETS.name} lacks the lists {miss}")
    elif len(set(sets["P_23"])) < 0.98 * len(set(sets["P_g2"])):
        bad.append(f"|P_23| {len(set(sets['P_23']))} < 0.98 |P_g2| {len(set(sets['P_g2']))} (spec 7 STOP)")
    lp = str(_abs(labels))
    rep = json.loads((L.RES / "P_report.json").read_text())
    lab = (rep.get("labels") or {}).get(lp) or {}
    if not lab.get("ids_ok") or not Path(lp).is_file() or L.sha256(lp) != lab.get("sha256"):
        bad.append(f"label file {lp}: sha256 / ids differ from results/P_report.json labels ({lab.get('sha256')})")
    if bad:
        stop("GEO2 Phase P state differs from what training needs (rerun feat_geo.py p / p_report): " + "; ".join(bad))
    return dict(p_build_geo_sha256=L.sha256(P_BUILD_GEO), p_geo_report_sha256=L.sha256(P_GEO_REPORT),
                p_geo_sets_sha256=L.sha256(P_GEO_SETS), sha256=L.sha256(P_BUILD_GEO), rungs=list(rungs), labels=lp,
                n_sets={k: len(sets[k]) for k in P_SET_KEYS})


def gate_c_geo():
    """= train_atoms.gate_c_geo: analysis/geo2/results/C_tests_geo.json all_pass and (XG-G10) its code_sha256 of the
    trainer files = the files now."""
    c = json.loads(C_TESTS_GEO.read_text()) if C_TESTS_GEO.is_file() else {}
    if not c.get("all_pass"):
        stop(f"stage 3 needs {C_TESTS_GEO} all_pass (tests/run_tests.py geo, GEO2 spec 8-2): all_pass "
             f"{c.get('all_pass')}")
    rec = c.get("code_sha256") if isinstance(c.get("code_sha256"), dict) else {}
    diff = [k for k in C_GEO_TRAINER_CODE if rec.get(k) != L.sha256(HERE / k)]
    if diff:
        stop(f"{C_TESTS_GEO.name} code_sha256 differs from the code now for {diff}: the C tests did not run on the code "
             f"that would train (XG-G10; rerun tests/run_tests.py geo gpu + cpu + merge)")
    return dict(c_tests_geo_sha256=L.sha256(C_TESTS_GEO), code_checked=list(C_GEO_TRAINER_CODE))


def guard_ladder_write(stage, what):
    """XG-G9 (= train_atoms GA-13): smoke / run / check with --stage 1|2 and l6 without --stage 3 write LADDER files
    (results/C_smoke_xgb.json, L_xgb_report.json, C_l6_xgb.json, $LAD/smoke_xgb, $LAD/smoke_xgb_l6, $LAD/units). STOP
    (exit 3) before any write unless CODE_SHA = results/L_xgb_report.json code_sha256, the code that wrote the LADDER
    XGB results (no such file = a fresh LADDER tree: the old behaviour). Stage 3 is not affected; `list` never calls
    this."""
    if int(stage) == GEO_STAGE:
        return
    ref = L.RES / "L_xgb_report.json"
    if not ref.is_file():
        return
    try:
        rec = json.loads(ref.read_text()).get("code_sha256")
    except (OSError, ValueError, AttributeError) as e:
        stop(f"{what} --stage {stage}: {ref} unreadable ({type(e).__name__}); refusing to write LADDER files (XG-G9)")
    rec = rec if isinstance(rec, dict) else {}
    diff = sorted(k for k in set(rec) | set(CODE_SHA) if rec.get(k) != CODE_SHA.get(k))
    if diff:
        stop(f"{what} --stage {stage} refused: this code differs from the code of the LADDER XGB results (results/"
             f"L_xgb_report.json code_sha256: {diff}); with the GEO2 additions the stage-1 / 2 commands other than list "
             f"would rewrite or invalidate LADDER files (GEO2 8-1 / 13 / 14; XG-G9)")


def _check_root(root, smoke, rung=None):
    r = _abs(root)
    u = _abs(UNITS)
    inside = r == u or u in r.parents
    if smoke and inside:
        stop("--smoke / l6 never write to $LAD/units")
    if inside and not _GATE.get("add3") and not (rung in GEO_INPUTS and _GATE.get("prereg_geo2")):
        stop("writing $LAD/units needs the ADD3 gate (train_xgb.py run)")
    if inside and rung in GEO_INPUTS and not _GATE.get("prereg_geo2"):          # XG-G4
        stop("writing geo units to $LAD/units needs the PREREG_GEO2 gate (train_xgb.py run --stage 3)")
    L.assert_path(r / "probe")
    return r


def fingerprint(D, u, sp, idx, smoke=False):
    fp = _fingerprint_dict(D, u, sp, idx, smoke)
    if D.rung in GEO_INPUTS:                    # XG-G5 (= train_atoms GA-8)
        fp["prereg_geo2"] = None if smoke else (L.sha256(PREREG_GEO2_MD) if PREREG_GEO2_MD.is_file() else None)
    return norm(fp)


def _fingerprint_dict(D, u, sp, idx, smoke=False):
    sh = D.arm == "SH"
    return dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=D.rung, arm=D.arm, model=MODEL,
        inputs_table=D.sha["inputs_table"], labels=D.sha["labels"], pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"],
        inner_folds=sp["inner_sha"], s_splits=D.sha["s_splits"], s_folds=D.sha["s_folds"], t_report=D.sha["t_report"],
        t_manifest=D.sha["t_manifest"], q_report=D.sha["q_report"], x182=D.sha["x182"],
        add3=None if smoke else (L.sha256(ADD3_MD) if ADD3_MD.is_file() else None),
        grid=[GRID[i] for i in idx], estimator=ESTIMATOR, feat_cols=D.feat_cols_sha,
        h_cols=D.hinfo["h_cols"] if sh else None,
        rules=dict(inner=f"GroupKFold({C.N_INNER}) by core (S_folds.csv) on the unfiltered split-file outer-train list, "
                         "then the usable-pair filter; fold 0 = split file inner_train / inner_val",
                   scale="d1_train.Scale on the fit's training pairs; y / std(y) without mean removal",
                   select="per target lowest mean inner MAE (kcal), ties grid order; final fit on all training pairs",
                   val="chosen cell's inner fold 0 MAE (kcal)", seeds="one fit, GRIDS estimator random_state",
                   antisym=dict(final_tol=F.ANTISYM_TOL, model_tol=C.ANTISYM_TOL, n=C.ANTISYM_N),
                   sh=("beta = sum(dH dy) / sum(dH^2) no intercept on the fit pairs, c_ghost 0; target r = y - beta dH; "
                       "score |beta dH + g - y|") if sh else None),
        versions=versions())


# ================================================================ one unit
def run_unit(D, u, root=UNITS, cells=None, smoke=False):
    """tune (per target: grid cells x 4 inner GroupKFold folds) -> final (per target best cell, all training pairs) ->
    test predictions + gate M-1' on every fit for one unit L__<rung>__<arm>__XGB__f<k>."""
    root = _check_root(root, smoke, getattr(D, "rung", None))
    root.mkdir(parents=True, exist_ok=True)
    if (u["rung"], u["arm"]) != (D.rung, D.arm) or u.get("model") != MODEL:
        stop(f"unit {u} does not match the data ({D.rung}, {D.arm}, {MODEL})")
    name = uname(u)
    idx = list(range(len(GRID))) if cells is None else sorted({int(c) for c in cells})
    sp = D.split(u["fold"])
    fp = fingerprint(D, u, sp, idx, smoke)
    js = root / f"{name}.json"
    if js.is_file():
        old = json.loads(js.read_text())
        ofp = old.get("fingerprint") or {}
        if ofp != fp:
            stop(f"cache {js} has a different fingerprint: keys {sorted(k for k in set(fp) | set(ofp) if ofp.get(k) != fp.get(k))}")
        print(f"  {name}: cached", flush=True)
        return old
    t_unit = time.time()
    tr, te, inner = sp["train"], sp["test"], sp["inner"]
    if min(len(tr), len(te), *[len(f["train"]) for f in inner], *[len(f["val"]) for f in inner]) == 0:
        stop(f"{name}: empty split list")
    # gate pairs: 200 random test pairs (rev 9 d1_train.fit_final rng rule, seeded by the unit name)
    rng = np.random.default_rng(C.SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10 ** 6)
    gp = te[rng.choice(len(te), size=min(C.ANTISYM_N, len(te)), replace=False)]
    # ---- tuning
    fits = []
    for f in inner:
        pf = prep_fit(D, f["train"], dict(val=f["val"]), gp)
        pf["leak"] = leak_check(D, pf, (f["val"], te), te)
        fits.append(pf)
    jobs = sorted([(j, ti, ci) for ti in range(NT) for j in range(len(fits)) for ci in idx],
                  key=lambda x: -cost(GRID[x[2]]))
    print(f"  {name}: {len(jobs)} tuning fits on {N_JOBS} workers ({len(D.feat_cols)} features)", flush=True)
    res = run_parallel([(fits[j]["zd"], fits[j]["zm"], fits[j]["ys"][ti], fits[j]["sy"][ti], GRID[ci], fits[j]["evals"])
                        for j, ti, ci in jobs])
    mae = np.full((NT, len(fits), len(idx)), np.nan)
    gmax = {t: dict(anti_max=0.0, self_max=0.0, model_anti_max=0.0, model_self_max=0.0) for t in TARGETS}
    tune_s = []
    for (j, ti, ci), (kc, sec) in zip(jobs, res):
        f, c = fits[j], idx.index(ci)
        yv = D.y[inner[j]["val"], ti]
        mae[ti, j, c] = float(np.mean(np.abs(f["base"]["val"][:, ti] + kc["val"] - yv)))
        g = gate_eval(kc, f["base"], ti)
        if not g["ok"]:
            stop(f"gate M-1' failed ({name}, {TARGETS[ti]}, inner {j}, cell {GRID[ci]}): {g}")
        for k in gmax[TARGETS[ti]]:
            gmax[TARGETS[ti]][k] = max(gmax[TARGETS[ti]][k], g[k])
        tune_s.append(sec)
    if not np.isfinite(mae).all():
        stop(f"{name}: non-finite inner MAE")
    tune, best = {}, {}
    for ti, t in enumerate(TARGETS):
        means = mae[ti].mean(axis=0)                                # = rev 9 pf.mean(axis=0), shape [cells]
        b = int(np.argmin(means))
        best[t] = idx[b]
        cell = GRID[idx[b]]
        edge = {}
        for k, v in cell.items():
            vals = sorted({c[k] for c in GRID})
            edge[k] = bool(len(vals) > 1 and v in (vals[0], vals[-1]))
        tune[t] = dict(grid_index=idx, grid_mae=means.tolist(), fold_mae=mae[ti].tolist(), best=dict(cell=cell, grid_index=idx[b]),
                       inner_cv_mae=float(means[b]), inner_fold_mae=mae[ti][:, b].tolist(), val_kcal_mae=float(mae[ti, 0, b]),
                       edge_hits=edge, beta_inner=[float(f["beta"][ti]) for f in fits],
                       sy_inner=[float(f["sy"][ti]) for f in fits], gate_max=gmax[t])
    # ---- final
    fin = prep_fit(D, tr, dict(test=te, train=tr), gp)
    fin["leak"] = leak_check(D, fin, (te,), te)
    fres = run_parallel([(fin["zd"], fin["zm"], fin["ys"][ti], fin["sy"][ti], GRID[best[t]], fin["evals"])
                         for ti, t in enumerate(TARGETS)])
    yhat = np.zeros((len(te), NT))
    final = {}
    for ti, (t, (kc, sec)) in enumerate(zip(TARGETS, fres)):
        g = gate_eval(kc, fin["base"], ti)
        if not g["ok"]:
            stop(f"gate M-1' failed ({name}, {t}, final): {g}")
        yhat[:, ti] = fin["base"]["test"][:, ti] + kc["test"]
        ptr = fin["base"]["train"][:, ti] + kc["train"]
        final[t] = dict(cell=GRID[best[t]], grid_index=best[t], beta=float(fin["beta"][ti]), sy=float(fin["sy"][ti]),
                        train_mae=float(np.mean(np.abs(ptr - D.y[tr, ti]))), gate_M1=g, seconds=float(sec))
    if not np.isfinite(yhat).all():
        stop(f"{name}: non-finite test predictions")
    beta = np.asarray(fin["beta"], float)
    base = fin["base"]["test"]
    pid = D.pid[te].astype(str)
    rk, ps = D.P.ref_kind.to_numpy()[te], D.P.pos.to_numpy()[te]
    parts = [pd.DataFrame(dict(pair_id=pid, ref_kind=rk, pos=ps, fold=int(u["fold"]), rung=u["rung"], arm=u["arm"],
                               model=MODEL, target=t, y=D.y[te, j], yhat=yhat[:, j], base=base[:, j], beta=float(beta[j]),
                               yhat_seed_sd=0.0)) for j, t in enumerate(TARGETS)]
    pred = pd.concat(parts, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    L.write_atomic(pq, lambda p: pred.to_parquet(p, index=False))
    rec = dict(unit=u, name=name, model=MODEL, fingerprint=fp, smoke=bool(smoke), gates_ok=True, n_seeds=1,
               chosen={t: dict(cell=GRID[best[t]], grid_index=best[t]) for t in TARGETS}, tune=tune, final=final,
               val_kcal_mae={t: tune[t]["val_kcal_mae"] for t in TARGETS}, val_pairs_sha256=sp["val_sha"],
               test_pairs_sha256=sp["test_sha"], n_val=int(len(sp["inner_val"])), beta=beta.tolist(), n_train=int(len(tr)),
               n_test=int(len(te)), n_inner_train=int(len(sp["inner_train"])), n_inner_val=int(len(sp["inner_val"])),
               inner_folds=[dict(n_train=int(len(f["train"])), n_val=int(len(f["val"])), dropped=f["dropped"],
                                 scale=pf["scale"], leak=pf["leak"]) for f, pf in zip(inner, fits)],
               final_scale=fin["scale"], final_leak=fin["leak"], dropped=sp["dropped"], n_gate_pairs=int(len(gp)),
               n_features=len(D.feat_cols), feature_cols=D.feat_cols if D.arm == "SH" else "T_report glob_cols",
               parquet_sha256=L.sha256(pq), inputs=D.inputs, inputs_table=D.hinfo, labels=D.linfo,
               opened_inputs=opened_summary(D.opened), fit_seconds=dict(tune=tune_s, final=[final[t]["seconds"] for t in TARGETS]),
               cpu_seconds=float(sum(tune_s) + sum(final[t]["seconds"] for t in TARGETS)), n_jobs=N_JOBS,
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), decisions=DECISIONS)
    write_json(js, rec)
    print(f"  {name}: done ({rec['wall_seconds']:.0f} s wall, {rec['cpu_seconds'] / 3600:.2f} CPU-h) "
          f"val {json.dumps({t: round(v, 4) for t, v in rec['val_kcal_mae'].items()})}", flush=True)
    return rec


# ================================================================ commands
def cmd_smoke(stage, labels):
    """spec 8 smoke for XGB -> results/C_smoke_xgb.json; writes only $LAD/smoke_xgb.
    Stage 3 (XG-G2): results/C_smoke_xgb_geo.json, scratch $LAD/smoke_xgb_geo."""
    guard_ladder_write(stage, "smoke")                 # XG-G9: before the scratch folder is removed
    smoke_dir = SMOKE_GEO if int(stage) == GEO_STAGE else SMOKE
    L.assert_path(smoke_dir / "probe")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    cells = [0, len(GRID) - 1]
    out = dict(versions=versions(), n_jobs=N_JOBS, labels=str(_abs(labels)), stage=int(stage), cells=[GRID[c] for c in cells],
               configs={}, code=CODE_VERSION, code_sha256=CODE_SHA, decisions=DECISIONS)
    ok, stage_cpu = True, 0.0
    for rung in STAGE_RUNGS[int(stage)]:
        for arm in ARMS:
            D = get_data(rung, arm, labels)
            u = dict(rung=rung, arm=arm, model=MODEL, fold=0)
            t0 = time.time()
            rec = run_unit(D, u, root=smoke_dir, cells=cells, smoke=True)
            g_ok = bool(rec["gates_ok"]) and all(rec["final"][t]["gate_M1"]["ok"] for t in TARGETS)
            ok &= g_ok
            ts, fs = rec["fit_seconds"]["tune"], rec["fit_seconds"]["final"]
            unit_cpu = float(np.mean(ts)) * len(GRID) * C.N_INNER * NT + float(np.mean(fs)) * NT   # full grid, 1 unit
            stage_cpu += unit_cpu * NF
            out["configs"][uname(u)] = dict(
                seconds=time.time() - t0, cpu_seconds=rec["cpu_seconds"], tune_fit_seconds_mean=float(np.mean(ts)),
                final_fit_seconds_mean=float(np.mean(fs)), unit_cpu_hours_full_grid=unit_cpu / 3600,
                gate_max={t: rec["tune"][t]["gate_max"] for t in TARGETS},
                gate_final={t: rec["final"][t]["gate_M1"] for t in TARGETS}, chosen=rec["chosen"],
                val_kcal_mae=rec["val_kcal_mae"], n_train=rec["n_train"], n_test=rec["n_test"],
                n_inner_train=rec["n_inner_train"], n_inner_val=rec["n_inner_val"], inner_folds=rec["inner_folds"],
                dropped=rec["dropped"], beta=rec["beta"], n_features=rec["n_features"], pair_list_n=len(D.pair_list),
                opened_dirs=rec["opened_inputs"]["dirs"], ok=g_ok)
            print(f"[smoke] {uname(u)}: ok {g_ok} {out['configs'][uname(u)]['seconds']:.0f} s", flush=True)
    out["stage_cpu_hours_estimate"] = stage_cpu / 3600
    out["stage_wall_hours_estimate_4x32"] = stage_cpu / 3600 / (4 * 32)
    out["ok"] = bool(ok)
    write_json(res_out("C_smoke_xgb.json", stage), out)
    return 0 if ok else 3


def cmd_run(i, n, stage, labels):
    guard_ladder_write(stage, "run")                   # XG-G9
    i = os.environ.get("SLURM_ARRAY_TASK_ID", "") if str(i) == "A" else str(i)
    if not (i.isdigit() and str(n).isdigit() and 0 <= int(i) < int(n)):
        print(f"usage: run <i|A> <n> with 0 <= i < n (got i={i!r}, n={n!r}; '@I' is replaced by jobs/py.slurm)",
              file=sys.stderr)
        return 2
    us = unit_list(stage)[int(i)::int(n)]
    if not us:
        print(f"[M] slice {i} of {n}: no units", flush=True)
        return 0
    rs = sorted({u["rung"] for u in us})
    need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}
    if int(stage) == GEO_STAGE:                        # XG-G3: PREREG_GEO2, GEO2 Phase P, C_tests_geo
        if set(rs) - set(STAGE_RUNGS[GEO_STAGE]):
            stop(f"stage 3 holds the geo rung only, got {rs}")
        pre = gate_prereg_geo2(need, rs)
        gp = gate_p_geo(rs, labels)
        gc = gate_c_geo()
        print(f"[M] PREREG_GEO2 ok (git {pre['git_head']}, md {_GATE['prereg_geo2'][:12]}), GEO2 P ok "
              f"({gp['p_build_geo_sha256'][:12]}, sets {gp['n_sets']}), C_tests_geo ok "
              f"({gc['c_tests_geo_sha256'][:12]}), {versions()}, n_jobs {N_JOBS}, slice {i}/{n}: {len(us)} units",
              flush=True)
    else:
        add3 = gate_add3(need, rs)
        gp = gate_p(rs, labels)
        if int(stage) == 2:
            gate_c_stage2()
        print(f"[M] ADD3 ok (git {add3['git_head']}), P_report ok ({gp['sha256'][:12]}), {versions()}, n_jobs {N_JOBS}, "
              f"slice {i}/{n}: {len(us)} units", flush=True)
    for u in us:
        t0 = time.time()
        D = get_data(u["rung"], u["arm"], labels)
        rec = run_unit(D, u, root=UNITS)
        print(f"[M] {uname(u)} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_check(stage, labels):
    """every XGB unit of the rungs up to the stage is present, unchanged and current -> results/L_xgb_report.json.
    Stage 3 (XG-G6): the geo XGB units only -> results/L_xgb_report_geo.json."""
    guard_ladder_write(stage, "check")                 # XG-G9: never rewrite the LADDER results/L_xgb_report.json
    rungs = collect_rungs(stage)
    us = unit_list(rungs=rungs)
    missing = [uname(u) for u in us if not ((UNITS / f"{uname(u)}.json").is_file()
                                            and (UNITS / f"{uname(u)}.parquet").is_file())]
    if missing:
        print(f"GATE: {len(missing)} XGB units missing: {missing[:8]}", file=sys.stderr)
        return 2
    bad, units = [], {}
    full = list(range(len(GRID)))
    for u in us:
        name = uname(u)
        r = json.loads((UNITS / f"{name}.json").read_text())
        pq = UNITS / f"{name}.parquet"
        if L.sha256(pq) != r.get("parquet_sha256"):
            bad.append(f"{name}: parquet changed after its json")
        if r.get("smoke") or not r.get("gates_ok"):
            bad.append(f"{name}: smoke unit or gate not ok")
        if not isinstance(r.get("val_kcal_mae"), dict) or set(r["val_kcal_mae"]) != set(TARGETS):
            bad.append(f"{name}: val_kcal_mae missing or not per target")
        D = get_data(u["rung"], u["arm"], labels)
        sp = D.split(u["fold"])
        now = fingerprint(D, u, sp, full)
        old = r.get("fingerprint") or {}
        if old != now:
            bad.append(f"{name}: fingerprint keys differ {sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))}")
        q = pd.read_parquet(pq)
        if list(q.columns) != PRED_COLS:
            bad.append(f"{name}: parquet columns {list(q.columns)} != PRED_COLS")
        want = _ids(D, sp["test"])
        for t in TARGETS:
            if q.loc[q.target == t, "pair_id"].astype(str).tolist() != want:
                bad.append(f"{name}: test pairs of {t} differ from the current split")
                break
        if r.get("val_pairs_sha256") != sp["val_sha"]:
            bad.append(f"{name}: validation pairs differ from the current split (L-9)")
        units[name] = dict(json_sha256=L.sha256(UNITS / f"{name}.json"), parquet_sha256=L.sha256(pq),   # MA-14 of atoms
                           chosen=r.get("chosen"), val_kcal_mae=r.get("val_kcal_mae"), beta=r.get("beta"),
                           n_train=r.get("n_train"), n_test=r.get("n_test"), n_val=r.get("n_val"), dropped=r.get("dropped"),
                           inner_cv_mae={t: r["tune"][t]["inner_cv_mae"] for t in TARGETS},
                           edge_hits={t: r["tune"][t]["edge_hits"] for t in TARGETS},
                           gate_max=dict(anti=max([r["tune"][t]["gate_max"]["anti_max"] for t in TARGETS]
                                                  + [r["final"][t]["gate_M1"]["anti_max"] for t in TARGETS]),
                                         self=max([r["tune"][t]["gate_max"]["self_max"] for t in TARGETS]
                                                  + [r["final"][t]["gate_M1"]["self_max"] for t in TARGETS])),
                           cpu_seconds=r.get("cpu_seconds"), wall_seconds=r.get("wall_seconds"), host=r.get("host"))
    rep = dict(stage=int(stage), rungs=list(rungs), n_units=len(us), labels=str(_abs(labels)), bad=bad, ok=not bad,
               code=CODE_VERSION, code_sha256=CODE_SHA, versions=versions(), units=units,
               cpu_hours=float(sum((v["cpu_seconds"] or 0) for v in units.values()) / 3600))
    write_json(res_out("L_xgb_report.json", stage), rep)
    print(f"[check] {len(us)} XGB units, {len(bad)} problems" + ("" if not bad else ": " + "; ".join(bad[:5])), flush=True)
    return 3 if bad else 0


def cmd_l6(stage=1):
    """L-6 (spec 8): g1 x S fold 0 with the original labels = the rev 9 C-XGB units (<= 1e-6) -> results/C_l6_xgb.json.
    stage 3 (XG-G7, GEO2 8-2 regression 2): the same run and comparison -> results/C_l6_xgb_geo.json, scratch
    $LAD/smoke_xgb_l6_geo (the LADDER C_l6_xgb.json and $LAD/smoke_xgb_l6 are not touched)."""
    guard_ladder_write(stage, "l6")                    # XG-G9: l6 without --stage 3 would rewrite C_l6_xgb.json
    u = dict(rung="g1", arm="S", model=MODEL, fold=0)
    l6_dir = L6_DIR_GEO if int(stage) == GEO_STAGE else L6_DIR
    L.assert_path(l6_dir / "probe")
    if l6_dir.exists():
        shutil.rmtree(l6_dir)
    D = get_data("g1", "S", ORIG_LABELS)
    sp = D.split(0)
    rec = run_unit(D, u, root=l6_dir, smoke=True)
    mine = pd.read_parquet(l6_dir / f"{uname(u)}.parquet")
    dropped_test = sorted(set(map(str, D.folds[0]["test"])) - set(_ids(D, sp["test"])))
    train_identical = (sp["dropped"]["train"]["not_usable"] == 0
                       and all(f["dropped"]["train"] == 0 and f["dropped"]["val"] == 0 for f in sp["inner"]))
    out = dict(unit=uname(u), labels=str(_abs(ORIG_LABELS)), rev9_units=str(REV9_UNITS), dropped_test=dropped_test,
               train_identical=bool(train_identical), versions=versions(), targets={})
    ok = bool(train_identical)
    for t in TARGETS:
        base = REV9_UNITS / f"T1__C__XGB__{t}__f0"
        rj = json.loads(base.with_suffix(".json").read_text())
        rp = pd.read_parquet(base.with_suffix(".parquet"))
        rp = rp.assign(pair_id=rp.pair_id.astype(str))[["pair_id", "y", "yhat"]]
        m = mine.loc[mine.target == t, ["pair_id", "y", "yhat"]].merge(rp, on="pair_id", how="outer", suffixes=("", "_rev9"),
                                                                        indicator=True)
        both = m[m._merge == "both"]
        only_mine = sorted(m.loc[m._merge == "left_only", "pair_id"].tolist())
        only_rev9 = sorted(m.loc[m._merge == "right_only", "pair_id"].tolist())
        dy = float(np.abs(both.y - both.y_rev9).max()) if len(both) else float("inf")
        dyh = float(np.abs(both.yhat - both.yhat_rev9).max()) if len(both) else float("inf")
        best_eq = norm(rec["chosen"][t]["cell"]) == norm(rj.get("best_params"))
        dcv = abs(rec["tune"][t]["inner_cv_mae"] - float(rj.get("inner_cv_mae", np.nan)))
        dtr = abs(rec["final"][t]["train_mae"] - float(rj.get("train_mae", np.nan)))
        ok_t = bool(not only_mine and only_rev9 == dropped_test and dy <= 1e-12 and dyh <= 1e-6 and best_eq and dcv <= 1e-9)
        ok &= ok_t
        out["targets"][t] = dict(n_both=int(len(both)), only_mine=only_mine, only_rev9=only_rev9, max_abs_y=dy,
                                 max_abs_yhat=dyh, best_equal=bool(best_eq), best=rec["chosen"][t]["cell"],
                                 best_rev9=rj.get("best_params"), inner_cv_mae_diff=dcv, train_mae_diff=dtr, ok=ok_t)
    out["ok"] = bool(ok)
    write_json(res_out("C_l6_xgb.json", stage), out)
    print(f"[l6] ok {ok}: " + json.dumps({t: (v["max_abs_yhat"], v["best_equal"]) for t, v in out["targets"].items()}),
          flush=True)
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=True)
    ap.add_argument("cmd", choices=("list", "smoke", "run", "check", "l6"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(uname(u) for u in unit_list(a.stage)))
        return 0
    if a.cmd == "smoke" or (a.cmd == "run" and a.smoke):
        return cmd_smoke(a.stage, a.labels)
    if a.cmd == "run":
        if len(a.rest) != 2:
            print("usage: train_xgb.py run <i|A> <n> [--stage 1|2] [--labels P]  |  run --smoke", file=sys.stderr)
            return 2
        return cmd_run(a.rest[0], a.rest[1], a.stage, a.labels)
    if a.cmd == "check":
        return cmd_check(a.stage, a.labels)
    if a.cmd == "l6":
        return cmd_l6(a.stage)
    return 2


if __name__ == "__main__":
    sys.exit(main())
