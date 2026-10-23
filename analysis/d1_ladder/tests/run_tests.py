#!/usr/bin/env python3
r"""tests/run_tests.py — D1-REF·LADDER Phase C: checks L-1 .. L-11 and the smoke test (docs/specs/D1LADDER.md 8)
-> results/C_tests.json (one entry per check: status / pass, numbers, details). sbatch only. Nothing in this file was
run when it was written (login node, no python allowed): every path is untested until it runs inside a job.

CLI (exit 0 = every check passed on the final inputs (labels final, P_report ready), 2 = nothing failed but a check is
waiting / pending / error, 3 = a check failed = spec 16 C STOP). Every check runs the production code (train_atoms.py, train_xgb.py, inputs.py, FU2 / rev 9).
  python tests/run_tests.py gpu [--labels P]   GPU part: env_d1fu2 (jobs/gpu.slurm), fu2 train.require_gpu; atom models
        AW / GNN / TF (train_atoms.py), the FU2 data path (L-6) and inputs.LadderData -> results/C_tests_gpu.json + merge
  python tests/run_tests.py cpu [--labels P]   CPU part: env_espley (analysis/d1_pair/jobs/py.slurm PYENV=espley); XGB
        (train_xgb.py), the inputs.py frame API, the rung-table provenance -> results/C_tests_cpu.json + merge
  python tests/run_tests.py merge              results/C_tests_{gpu,cpu}.json -> results/C_tests.json (any env)
  python tests/run_tests.py l4child <atoms|xgb> <rung> <labels> <out.json>   internal: one fresh L-4 training process
  --labels P overrides the label file of the label-independent checks (default: see C-3); L-6 / L-10 always use the
  original D1PAIR F_pairs.csv, L-11 always F_pairs_cr2.csv.

Which part runs which check (both parts are needed; the merge combines them per check)
  check  gpu part (env_d1fu2, GPU node)                              cpu part (env_espley, CPU node)
  L-1    AW / GNN / TF x S / SH x rung (noisy float64 copies)         XGB x S / SH x rung (GRID[0] fit, float64 z)
  L-2    beta pairs of every SH fit in run_unit (spy)                  beta pairs of every SH prep_fit in run_unit (spy)
  L-3    beta_fit (y = 2 dH), StdSH + final_sh with a zero model       beta_fit, prep_fit baseline with a zero model
  L-4    child process per rung: train_atoms run_unit (AW, S + SH)     child process per rung: train_xgb run_unit (S + SH)
  L-5    LadData SH global H / H10 / dH vs table (+ inputs API)        XData X / H10 / dH vs table, table provenance (+ API)
  L-6    LadData + Std tensors (sha256), s_y = FU2 unit T1__*__AG__f0  train_xgb.py l6 (= rev 9 T1__C__XGB__<t>__f0)
  L-7    FU2 loss_fn (S) / loss_sh (SH) on the true y                  - (XGB has no identity penalty)
  L-8    Std / StdSH statistics per fold + run_unit spy (FU2 C7 / C8)  inner GroupKFold folds per fold + prep_fit spy
  L-9    validation pairs of the 3 atom models (spy) + val sha256      XGB inner fold 0 pairs + val sha256; merge: equal
  L-10   LINE (train_atoms.beta_fit) on D1-HF Phase A pairs / folds    LINE (train_xgb.beta_fit) + inputs.pairs_frame
  L-11   LadData y (+ inputs.LadderData y) vs cr2 / F_pairs.csv        XData y (+ inputs.pairs_frame) vs cr2 / F_pairs.csv
  smoke  train_atoms.py smoke (stage 1)                                train_xgb.py smoke (stage 1)
Every check of the gpu part runs inside the GPU job (the part imports torch / FU2 and require_gpu gates it); the ones that
actually need the GPU are L-1, L-2, L-4, L-8 / L-9 (spy run_unit), L-6 (FU2 tensors) and smoke.

Inputs that must exist (else the check reports 'waiting', never 'fail'; g2 is 'skipped' while its table is absent)
  every rung check   $LAD/inputs/<rung>.parquet + results/P_build_<rung>.json all_pass with that table's sha256
                     (inputs.py build g1 now; build dft after Phase V passed and the F0 STOP is decided; g2: stage 2)
  inputs API parts   results/P_report.json all_pass (inputs.py pairs; needs both builds -> waits for V)
  L-11, final label  $LAD/F_pairs_cr2.csv + results/V_report.json all_pass with its sha256 (Phase V collect)
  L-6                $LAD/inputs/g1.parquet (usable filter), $D1PAIR_SCRATCH/fu2/units/T1__{AW,GNN,TF}__AG__f0.*,
                     $D1PAIR_SCRATCH/M/units/T1/T1__C__XGB__<t>__f0.* (read only)
  L-10               analysis/d1_hf/results/A_summary.csv (read only)
How 'waiting' works: resolve() checks those files before a check runs; a rung / label / P_report that is not ready
makes the sub-result 'waiting' with the reason, and a production function that exits 2 (its not-ready exit) is recorded
as 'waiting' too. Before V the label-independent checks run on F_pairs_cr.csv (Phase Q passed) and are marked
provisional (labels.final = false); results/C_tests.json all_pass needs a rerun of both parts with the final labels, dft
ready and P_report present.

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 counts array
elements; this needs N + 3 <= 20) and LADDER <= 4 running jobs while D1-G2 runs; sinfo for idle partitions)
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder
  sinfo -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -o "%.9P %.6t %.10G %C"; sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"
  JG=$(sbatch --parsable -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 -J lad_ctest_g --time=48:00:00 \
       --cpus-per-task=4 --mem=32G jobs/gpu.slurm tests/run_tests.py gpu); echo "JG=$JG"
  cd ../d1_pair
  JC=$(sbatch --parsable -p cpu1,cpu2 -J lad_ctest_c --time=48:00:00 --cpus-per-task=32 --mem=64G \
       --output=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs/%x_%j.out --export=ALL,PYENV=espley \
       jobs/py.slurm ../d1_ladder/tests/run_tests.py cpu); echo "JC=$JC"
  cd ../d1_ladder
  sbatch -p cpu1,cpu2 -J lad_ctest_m --time=48:00:00 --cpus-per-task=1 --mem=2G --dependency=afterany:${JG:?}:${JC:?} \
       --export=ALL,LADENV=reactot jobs/lad.slurm tests/run_tests.py merge
  (py.slurm's default log folder is the read-only d1_pair scratch, hence --output. Each part also merges when it ends;
  the dependent merge job makes the final C_tests.json independent of which part finished last.)

Decisions (user 2026-10-22: every judgment = the recommended option; also in C_tests.json 'decisions')
  C-1  two parts by environment (no env has both torch and xgboost: env_d1fu2 lacks xgboost, env_espley lacks torch),
       merged per check; a check that both parts own passes only when both pass (L-7: gpu only).
  C-2  statuses pass / fail / waiting (an input phase not finished) / pending (a part has not run) / skipped (g2 table
       absent; spec 8 'g2 skipped and reported'); exit 3 on any fail, 2 on any waiting / pending, 0 otherwise.
  C-3  labels: final = $LAD/F_pairs_cr2.csv when results/V_report.json all_pass and lists its sha256; otherwise the
       label-independent checks use $LAD/F_pairs_cr.csv (Q_report all_pass, sha256) marked provisional (else the
       original F_pairs.csv); all_pass needs the final labels; L-11 waits for cr2.
  C-4  the loaders tested are the ones training uses (train_atoms.LadData, train_xgb.XData); the inputs.py loader API
       (LadderData, pairs_frame, reaction_frame, read_set) is checked for the same values when P_report exists.
  C-5  L-1 atom models = FU2 C1 pattern: a 1-epoch fit of grid cell 0 on 1,024 fold-0 training pairs, N(0, 0.3^2) on
       every weight (FU2 tests noisy64), float64 eval; final prediction (S: predict x s_y; SH: train_atoms.final_sh =
       beta dH + residual) on 200 fold-0 test pairs; plus the production gate (gate_m1 / gate_m1_sh) on that model;
       non-vacuity median |g| >= 1e-2 (FU2 C1) and median |beta dH| > 0 (SH). XGB: GRID[0] (smallest cell) fit on the
       fold-0 training pairs per target, swapped inputs recomputed from x in float64 (train_xgb.prep_fit), final
       prediction incl. baseline <= 1e-4, the production gate_eval too.
  C-6  L-2 / L-8 / L-9 spy on the production run_unit (atoms: every model, cell 0, 1 epoch, 1 seed; XGB: cell 0) in
       $LAD/tests (smoke=True, never $LAD/units), plus split-level checks for all 5 folds.
  C-7  L-4: one fresh python process per tool x rung (the train_atoms / train_xgb audit hook from its import on, plus
       the unit's explicit input paths), classified with inputs.check_paths / rung_roots. Rung-independent pair lists
       are allowed for every rung: FU2 S_splits.json, splits/T1__f<k>.json, splits/pairs.parquet (pair ids only; spec
       4-3 uses the same folds for every rung; train_atoms / train_xgb read the FU2 originals). Interpreter / system /
       non-data paths and this test's own outputs are ignored; any other unclassified data path = fail.
  C-8  L-5: 50 random usable reactions and 50 usable pairs per rung (rng SEED_M + 50 + rung index): trainer inputs
       (SH global H columns, H10, dH) = rung table H_<t> (1e-12); plus (cpu) the table's own provenance on the sample
       and on every row: H0 = labels_d1_hf(_g2) values (1e-12), 6 channels unchanged, H - H0 of d1 / d2 / barrier =
       reference-energy change x 627.5094740631 (1e-9), and the trainers' usable reactions = the table's.
  C-9  L-6 FU2 unit = the cached T1__{AW,GNN,TF}__AG__f0 (json final s_y, __s0.pt std, parquet y); unit tensors =
       FU2 Data (default FU2 roots) scaled with the .pt statistics; loader = train_atoms LadData (g1, S, original
       labels). Gates: the loader's raw arrays = FU2 Data's (sha256), the loader's arrays scaled with the unit's
       statistics = the unit tensors (sha256 per tensor x, glob, pos, mask, etype, frag, n; float32 on CPU), the
       loader's fu2 Std on its fold-0 training pairs = the unit's statistics (1e-12), s_y = json final s_y (1e-12), y
       and test pairs = the unit parquet; the sha256 of the fully recomputed tensors is reported (a 1-ulp statistic
       difference from another CPU could flip one float32 element without any loader difference). Split lists, ends
       and y for all 5 folds, Std + tensors (training and inner-training pairs) of fold 0 vs FU2 Data on this node.
       XGB = train_xgb.py l6 (C_l6_xgb.json: yhat <= 1e-6, same best cell).
  C-10 L-7 atoms only; S = fu2 loss_fn, SH = train_atoms.loss_sh, on the fold-0 training pairs, plus exact-identity and
       barrier-perturbation controls (FU2 C9).
  C-11 L-10 LINE = train_atoms.beta_fit / train_xgb.beta_fit (the beta of the SH baseline and of the spec 4-5 LINE row;
       evaluate.py does not exist yet, its LINE rows must use the same function) on the D1-HF Phase A pairs (S_folds
       in_P_main & reference, both ends with all 9 original H) and folds (beta from fold_T1 != k, applied to fold k),
       original H = rung table H0_<t> (+ inputs.pairs_frame h='orig' when P_report exists); gate: n, n_para and para
       MAE within 1e-6 of A_summary.csv (mae_all and beta reported).
  C-12 L-11: independent parse of the label files (pandas float_precision='round_trip') by pair_id; all 10 targets vs
       F_pairs_cr2.csv, elst pauli oi disp cpcm cds c_ghost vs F_pairs.csv, 1e-12.
  C-13 smoke = train_atoms.cmd_smoke / train_xgb.cmd_smoke (stage 1) once dft and g1 are ready; before that the same
       run_unit settings on the ready rungs into $LAD/tests (status waiting); $LAD/units listing unchanged before/after.
  C-14 scratch $LAD/tests (removed per sub-run); partial results results/C_tests_{gpu,cpu}.json, merged C_tests.json;
       json written atomically with unique temp names (lad_common.write_atomic uses one fixed .tmp name).
  C-15 wrappers: gpu part = jobs/gpu.slurm (env_d1fu2); cpu part = analysis/d1_pair/jobs/py.slurm PYENV=espley, not
       jobs/lad.slurm: lad.slurm offers only LADENV=d1 / reactot and neither has xgboost (reactot: scikit-learn 1.7.2,
       no xgboost), while train_xgb.py and the L-6 rev 9 reproduction need env_espley (= train_xgb XG-1 / XG-2);
       lad.slurm (reactot) runs the merge. No wrapper file was added.
Review fixes 2026-10-23 (FIXER; recommended option each time)
  C-16 (review C-1) C_tests.json records stage_verified, g2_verified and all_pass_scope; a skipped g2 still does not
       block stage 1, but train_atoms / train_xgb run --stage 2 require g2_verified (a rerun with the g2 table).
  C-17 (review C-2) L-3 XGB also runs the production train_xgb.run_unit on synthetic data with a zero model
       (run_parallel -> 0): stored yhat = base = beta dH (S 0), stored inner scores / val_kcal_mae = the LINE MAE, 1e-10.
  C-18 (review C-3) L-10 (cpu part) also tests analysis/d1_ladder/evaluate.py line_beta(dH, y) (contract above) and
       waits while evaluate.py does not exist: Phase C cannot pass before the LINE function of E1 exists.
  C-19 (review C-4) merge re-checks that each part's code / labels / tables / P_report / V_report / ready rungs still
       equal the files now; otherwise 'pending' (rerun).
  C-20 (review C-5) a part exits 0 only when all its checks pass on the final inputs (labels final, P_report ready);
       an aborted part (import / no GPU) marks its checks 'error' (exit 2, listed under 'errors'), not 'fail'.
  C-21 (review C-6) SH non-vacuity also on the residual-model part alone (median |model part| >= 1e-2).
  C-22 (review C-7) the L-4 child imports the trainer before any other project module (its audit hook first); the
       classifier is inputs.check_paths with the corrected roots (inputs P23), so the split whitelist is redundant.
  C-23 (review C-8) L-7 SH also checks the training path: one fit_sh epoch with epoch_batches / loss_sh spied (loss
       arguments = the batch, sr / sy = StdSH, baseline flipped with the oriented pair, r sr + base = oriented y).
  C-24 (review C-9) L-5 provenance against independent sources: e_ref_orig = labels_d1_hf, g rungs e_ref = key minimum
       over the table, dft Q references = the source job's labels_d1_hf value.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair, fu2, d1_hf, espley trees

import contextlib  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                  # analysis/d1_ladder/tests
LADC = HERE.parent                                      # analysis/d1_ladder (code)
if str(LADC) not in sys.path:
    sys.path.insert(0, str(LADC))
if __name__ == "__main__" and sys.argv[1:2] == ["l4child"] and sys.argv[2:3] in (["atoms"], ["xgb"]):
    # L-4 child (review C-7): import the trainer first, as in production where it is __main__, so its audit hook sees
    # every later open (lad_common / d1pair_common / numpy imports included)
    __import__({"atoms": "train_atoms", "xgb": "train_xgb"}[sys.argv[2]])
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lad_common as L  # noqa: E402

DP = L.DP
TESTS_VERSION = "d1-ladder-c1"
RUNG_ORDER = ("dft", "g1", "g2")
STAGE1 = ("dft", "g1")
ARMS = ("S", "SH")
TARGETS = list(DP.TARGETS)                              # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
H9 = list(DP.PRIMARY_TARGETS)                           # the 9 HF values (spec 4-1)
HIDX = [TARGETS.index(t) for t in H9]
IGHOST = TARGETS.index("c_ghost")
CH6 = ("elst", "pauli", "oi", "disp", "cpcm", "cds")
UNCHANGED = [t for t in TARGETS if t not in ("d1", "d2", "barrier")]
CHECKS = [f"L-{i}" for i in range(1, 12)] + ["smoke"]
PARTS_OF = {c: ("gpu", "cpu") for c in CHECKS}
PARTS_OF["L-7"] = ("gpu",)
TOL = dict(L1=1e-4, L3=1e-10, L5=1e-12, L5_reref=1e-9, L6=1e-12, L6_xgb=1e-6, L7=1e-3, L7_exact=1e-12, L7_pert=1e-9,
           L8_stats=1e-12, L8_fed_x=1e-5, L10=1e-6, L11=1e-12, nonvacuity_median_abs_g=1e-2)
CR2 = L.LAD / "F_pairs_cr2.csv"
CR = L.LAD / "F_pairs_cr.csv"
ORIG = DP.RES / "F_pairs.csv"
S_FOLDS = DP.RES / "S_folds.csv"
A_SUMMARY = L.HF_RES / "A_summary.csv"
FU2_UNITS = DP.SCRATCH / "fu2" / "units"
FU2_SPLITS = DP.SCRATCH / "fu2" / "splits"
FU2_S_SPLITS = DP.RES / "fu2" / "S_splits.json"
SCR = L.LAD / "tests"
OUT = L.RES / "C_tests.json"
PART_FILE = {p: L.RES / f"C_tests_{p}.json" for p in ("gpu", "cpu")}
H_PARQUET = {"dft": L.D1HF_LABELS, "g1": L.D1HF_LABELS, "g2": L.G2_DIR / "results" / "labels_d1_hf_g2.parquet"}
SPLIT_WHITELIST = [FU2_S_SPLITS, FU2_SPLITS / "pairs.parquet"] + [FU2_SPLITS / f"T1__f{k}.json" for k in range(5)]
DATA_ROOTS = ("/gpfs/tmp_cpu2/yeseo1ee/", "/gpfs/home1/yeseo1ee/projects/", str(L.REPO) + os.sep)
P_WAIT = "results/P_report.json missing or not all_pass (inputs.py pairs; it needs build dft, i.e. Phase V)"
CODE_FILES = {"tests/run_tests.py": Path(__file__).resolve(), "inputs.py": LADC / "inputs.py",
              "train_atoms.py": LADC / "train_atoms.py", "train_xgb.py": LADC / "train_xgb.py",
              "lad_common.py": LADC / "lad_common.py"}
DECISIONS = {
    "C-1": "two parts by env (gpu: env_d1fu2 / torch; cpu: env_espley / xgboost), merged per check",
    "C-2": "pass / fail / waiting / pending / skipped; exit 3 fail, 2 waiting or pending, 0 all pass",
    "C-3": "final labels F_pairs_cr2.csv (V_report); before V label-independent checks on F_pairs_cr.csv (provisional)",
    "C-4": "trainers' loaders tested (train_atoms.LadData, train_xgb.XData) + inputs.py API when P_report exists",
    "C-5": "L-1: FU2 C1 pattern (1 epoch, noisy64, float64) on the final prediction; XGB GRID[0] fit, float64 z",
    "C-6": "L-2 / L-8 / L-9 spy on production run_unit (cell 0, 1 epoch, 1 seed) in $LAD/tests + all-fold split checks",
    "C-7": "L-4 fresh child process per tool x rung; FU2 split files (pair lists) allowed for every rung; unknown = fail",
    "C-8": "L-5: 50 reactions / 50 pairs; trainer inputs = table (1e-12); table provenance (H0, channels, dRef x KCAL)",
    "C-9": "L-6: FU2 units T1__{AW,GNN,TF}__AG__f0 (json s_y, .pt std, parquet y); tensor sha256 with the unit's std, "
           "loader std 1e-12; XGB = train_xgb l6",
    "C-10": "L-7 atoms only (S loss_fn, SH loss_sh) + exact-identity / perturbation controls",
    "C-11": "L-10 LINE = beta_fit on D1-HF Phase A pairs / folds with table H0; gate n, n_para, para MAE 1e-6",
    "C-12": "L-11 round_trip parse by pair_id; 10 targets vs cr2, 7 unchanged vs F_pairs.csv, 1e-12",
    "C-13": "smoke = cmd_smoke of both trainers (stage 1); partial run_unit smoke while dft waits; units unchanged",
    "C-14": "scratch $LAD/tests; partial C_tests_{gpu,cpu}.json; atomic json with unique temp names",
    "C-15": "gpu part jobs/gpu.slurm; cpu part d1_pair/jobs/py.slurm PYENV=espley (lad.slurm envs lack xgboost); "
            "merge via lad.slurm reactot",
    "C-16": "stage_verified / g2_verified / all_pass_scope; stage-2 training requires g2_verified",
    "C-17": "L-3 XGB: production run_unit with a zero model on synthetic data (yhat = beta dH, scores = LINE MAE)",
    "C-18": "L-10 cpu: evaluate.py line_beta tested; waiting while evaluate.py is absent",
    "C-19": "merge re-checks code / labels / tables / P_report / V_report / ready rungs of each part",
    "C-20": "exit 0 only on final inputs; abort = 'error' (exit 2), not 'fail'",
    "C-21": "SH non-vacuity on the model part alone too",
    "C-22": "L-4 child imports the trainer first; inputs.check_paths roots corrected",
    "C-23": "L-7 SH through fit_sh (epoch_batches / loss_sh spies)",
    "C-24": "L-5 reference energies vs labels_d1_hf / key minimum / Q source (g2 P20 own: constrained rows = orig)",
    "C-25": "smoke guard = this process writes nothing under $LAD/units (audit hook); dir diff = concurrent training (info)",
}
RANK = {"skipped": 0, "pass": 1, "waiting": 2, "pending": 3, "error": 4, "fail": 5}


# ================================================================ small helpers
def now():
    return time.strftime("%F %T")


def combine(statuses):
    """the worst status (fail > error > pending > waiting > pass); skipped entries do not count. 'error' = the part
    could not run its checks (environment / GPU missing; review C-5), not a spec 16 C STOP."""
    s = [x for x in statuses if x != "skipped"]
    return max(s, key=RANK.__getitem__) if s else "skipped"


class Waiting(Exception):
    """an input phase has not finished: the check reports 'waiting', not 'fail'."""


class _Tee(io.TextIOBase):
    """stderr copy, so STOP / NOT READY messages of the production code land in the check result."""

    def __init__(self, orig):
        self.orig, self.parts = orig, []

    def write(self, s):
        self.orig.write(s)
        self.parts.append(s)
        return len(s)

    def flush(self):
        self.orig.flush()

    def fileno(self):                    # libraries that write to fd 2 directly (loky, C extensions) bypass the copy
        return self.orig.fileno()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.orig, "encoding", "utf-8")

    def tail(self, n=3000):
        return "".join(self.parts)[-n:]


def guard(fn, *a, **k):
    """run fn -> its result dict (must carry 'status'); Waiting -> waiting; exit 2 -> waiting (not ready);
    any other exit / exception -> fail with the message."""
    t0 = time.time()
    tee = _Tee(sys.stderr)
    try:
        with contextlib.redirect_stderr(tee):
            res = fn(*a, **k)
        if not isinstance(res, dict) or "status" not in res:
            res = dict(status="fail", error=f"check returned {type(res).__name__}, not a result dict")
    except Waiting as e:
        res = dict(status="waiting", why=str(e))
    except SystemExit as e:
        code = e.code
        res = dict(status="waiting" if code == 2 else "fail", exit_code=code if isinstance(code, int) else str(code),
                   why=("exit 2 inside the production code = not ready (an input of this check is missing)"
                        if code == 2 else "exit inside the production code (STOP; message in stderr_tail)"),
                   stderr_tail=tee.tail())
    except Exception as e:                                             # noqa: BLE001
        res = dict(status="fail", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-6000:],
                   stderr_tail=tee.tail())
    res.setdefault("seconds", round(time.time() - t0, 1))
    return res


def _js(o):
    """json-safe copy: numpy -> python, non-finite floats -> 'nan' / 'inf' strings (jq-readable), sets sorted."""
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [_js(v) for v in sorted(o, key=str)]
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, np.ndarray):
        return _js(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path, obj):
    """atomic json write under $LAD / results (lad_common.assert_path), unique temp name."""
    path = L.assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_js(obj), f, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_json(p):
    p = Path(p)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def sha(p):
    return L.sha256(p) if Path(p).is_file() else None


def maxdiff(a, b):
    """max |a - b| with equal NaN patterns required (inf when shapes or NaN patterns differ)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


def maxrel(a, b):
    """max |a - b| / max(1, |b|), NaN patterns equal (inf otherwise)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float((np.abs(a[m] - b[m]) / np.maximum(1.0, np.abs(b[m]))).max()) if m.any() else 0.0


def arr_sha(x):
    import hashlib
    x = np.ascontiguousarray(x)
    return hashlib.sha256(f"{x.dtype}|{x.shape}|".encode() + x.tobytes()).hexdigest()


def tensor_sha(T):
    """sha256 per tensor of a fu2 Std.tensors dict (moved to CPU)."""
    return {k: arr_sha(v.detach().cpu().numpy()) for k, v in sorted(T.items())}


def ids_of(index_like, rows):
    arr = np.asarray(index_like)
    return [str(q) for q in arr[np.asarray(rows, dtype=int)]]


def rx_of(D, rows):
    rows = np.asarray(rows, dtype=int)
    return set(np.asarray(D.ra)[rows].tolist()) | set(np.asarray(D.rb)[rows].tolist())


def table_path(r):
    return L.LAD / "inputs" / f"{r}.parquet"


def read_table(r):
    T = pd.read_parquet(table_path(r))
    return T.assign(rxn_id=T.rxn_id.astype(int)).set_index("rxn_id")


def _git_head():
    try:
        return L.git("rev-parse", "HEAD").strip()
    except Exception:                                                  # noqa: BLE001
        return None


def code_shas():
    return {k: sha(p) for k, p in CODE_FILES.items()}


def _units_snapshot():
    u = L.LAD / "units"
    if not u.is_dir():
        return None
    return sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(u))


class _UnitWrites:
    """C-25 (2026-10-23): the smoke guard 'smoke never writes $LAD/units' judged on THIS process's own writes (audit
    hook: open with a write mode / O_CREAT, rename / replace / remove / mkdir under $LAD/units), not on the whole
    directory, because stage-1 training writes units there at the same time (the old before == after snapshot test
    failed on those concurrent writes; it is still recorded as units_dir_unchanged, information only)."""
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def __init__(self):
        self.on, self.hits, self.installed = False, [], False
        self.root = os.path.realpath(str(L.LAD / "units"))

    def _under(self, p):
        try:
            return os.path.realpath(os.fsdecode(p)).startswith(self.root + os.sep) or \
                os.path.realpath(os.fsdecode(p)) == self.root
        except (TypeError, ValueError):
            return False

    def hook(self, ev, args):
        if not self.on:
            return
        try:
            if ev == "open" and len(args) >= 3:
                p, mode, flags = args[0], args[1], args[2]
                w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                    (isinstance(flags, int) and bool(flags & self.WRITE_FLAGS))
                if w and isinstance(p, (str, bytes, os.PathLike)) and self._under(p):
                    self.hits.append(("open", os.fsdecode(p)))
            elif ev in ("os.rename", "os.replace", "os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree", "shutil.move"):
                for p in args[:2]:
                    if isinstance(p, (str, bytes, os.PathLike)) and self._under(p):
                        self.hits.append((ev, os.fsdecode(p)))
        except Exception:                                              # noqa: BLE001 (never break the test run)
            pass

    def start(self):
        if not self.installed:
            sys.addaudithook(self.hook)
            self.installed = True
        self.hits, self.on = [], True

    def stop(self):
        self.on = False
        return list(self.hits)


UNIT_WRITES = _UnitWrites()


def _smoke_units_verdict(res, before, after):
    """C-25: fail only on writes by this process under $LAD/units; the directory diff is information."""
    hits = UNIT_WRITES.stop()
    res["units_dir_unchanged"] = before == after
    if before != after and before is not None and after is not None:
        b, a = {x[0]: x for x in before}, {x[0]: x for x in after}
        res["units_dir_changed_by_others"] = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))[:40]
    res["units_written_by_smoke"] = hits[:40]
    res["units_guard"] = "C-25: this process's writes under $LAD/units (audit hook); directory diff = concurrent training"
    if hits:
        res["status"] = "fail"
    return res


def _rmtree(p):
    p = Path(p)
    L.assert_path(p / "probe")
    if p.exists():
        shutil.rmtree(p)
    return p


def _cuda_free(torch):
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fu2_tests():
    """FU2's own test module (analysis/d1_pair/fu2/tests/run_tests.py) for its helpers (noisy64), under another name."""
    import importlib.util
    p = L.P_DIR / "fu2" / "tests" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("fu2_tests_run_tests", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Ctx:
    """per-part context: modules, resolved state (rungs / labels / P_report), spy results."""

    def __init__(self, part, **kw):
        self.part, self.spy, self.check = part, {}, None
        self.__dict__.update(kw)

    @property
    def IN(self):
        if "_IN" not in self.__dict__:
            import inputs as _inputs
            self.__dict__["_IN"] = _inputs
        return self.__dict__["_IN"]

    @property
    def p_ready(self):
        return bool(self.state["p_report"]["ready"])


# ================================================================ state: which inputs are ready
def resolve(labels_arg=None):
    rungs = {}
    for r in RUNG_ORDER:
        tab, br = table_path(r), L.RES / f"P_build_{r}.json"
        if not tab.is_file():
            rungs[r] = dict(status="skipped" if r == "g2" else "waiting",
                            why=f"{tab} missing (Phase P: inputs.py build {r}"
                                + ("; needs D1-G2 Phase F / H, stage 2)" if r == "g2" else ")"))
            continue
        rep, s = read_json(br), L.sha256(tab)
        if not rep.get("all_pass") or (rep.get("outputs") or {}).get(str(tab)) != s:
            rungs[r] = dict(status="waiting", why=f"{br.name}: all_pass {rep.get('all_pass')}, stop {rep.get('stop')}, "
                                                  f"table sha256 recorded: {(rep.get('outputs') or {}).get(str(tab)) == s}")
            continue
        rungs[r] = dict(status="ready", why="", table=str(tab), table_sha256=s, build_report_sha256=L.sha256(br))

    def passed(rep, p):
        return bool(rep.get("all_pass") and not rep.get("stop") and p.is_file()
                    and (rep.get("outputs") or {}).get(str(p)) == L.sha256(p))
    vrep, qrep = read_json(L.RES / "V_report.json"), read_json(L.RES / "Q_report.json")
    cr2_ok, cr_ok = passed(vrep, CR2), passed(qrep, CR)
    if labels_arg:
        lab = Path(os.path.abspath(labels_arg))
        final, why = bool(lab == CR2 and cr2_ok), "--labels"
    elif cr2_ok:
        lab, final, why = CR2, True, "F_pairs_cr2.csv: V_report.json all_pass with its sha256"
    elif cr_ok:
        lab, final, why = CR, False, "provisional: Phase V not passed, F_pairs_cr.csv (Q_report all_pass) used"
    else:
        lab, final, why = ORIG, False, "provisional: neither V nor Q outputs verified, original F_pairs.csv used"
    pr = read_json(L.RES / "P_report.json")
    return dict(rungs=rungs, labels=dict(path=str(lab), sha256=sha(lab), final=final, why=why, cr2_ready=cr2_ok,
                                         cr_ready=cr_ok, v_report_sha256=sha(L.RES / "V_report.json")),
                p_report=dict(ready=bool(pr.get("all_pass") and not pr.get("stop")), sha256=sha(L.RES / "P_report.json")))


def per_rung(ctx, fn, rungs=RUNG_ORDER):
    subs = {}
    for r in rungs:
        rs = ctx.state["rungs"][r]
        if rs["status"] != "ready":
            subs[r] = dict(status=rs["status"], why=rs["why"])
            continue
        print(f"  [{ctx.part} {ctx.check}] rung {r}  {now()}", flush=True)
        subs[r] = guard(fn, r)
    return dict(status=combine(s["status"] for s in subs.values()), rungs=subs)


def _need_cr2(ctx):
    if not ctx.state["labels"]["cr2_ready"]:
        raise Waiting("F_pairs_cr2.csv missing or results/V_report.json not all_pass with its sha256 (Phase V collect)")


def _api(ctx, fn):
    """inputs.py API sub-check: waiting while P_report is missing."""
    if not ctx.p_ready:
        return dict(status="waiting", why=P_WAIT)
    return guard(fn)


# ================================================================ L-3 (synthetic, both parts)
def _synth(seed):
    rng = np.random.default_rng(seed)
    nr, npair = 60, 400
    H10 = np.zeros((nr, len(TARGETS)))
    H10[:, HIDX] = rng.normal(size=(nr, len(HIDX))) * 3.0
    ra = rng.integers(0, nr, npair)
    rb = (ra + rng.integers(1, nr, npair)) % nr
    dH = H10[ra] - H10[rb]
    y2 = 2.0 * dH
    y2[:, IGHOST] = rng.normal(size=npair) * 0.1
    yn = 2.0 * dH + rng.normal(0.0, 0.3, size=dH.shape)
    yn[:, IGHOST] = rng.normal(size=npair) * 0.1
    return dict(rng=rng, nr=nr, npair=npair, H10=H10, ra=ra, rb=rb, dH=dH, y2=y2, yn=yn, tr=np.arange(300),
                te=np.arange(300, npair))


def _beta_checks(beta_fit, s):
    b2 = np.asarray(beta_fit(s["dH"], s["y2"]), float)
    bn = np.asarray(beta_fit(s["dH"], s["yn"]), float)
    lsq = np.array([np.linalg.lstsq(s["dH"][:, [j]], s["yn"][:, j], rcond=None)[0][0] for j in HIDX])
    out = dict(beta_y_eq_2dH_minus_2_max=float(np.abs(b2[HIDX] - 2.0).max()), beta_c_ghost=float(b2[IGHOST]),
               beta_noise_vs_lstsq_no_intercept_rel=float(np.max(np.abs(bn[HIDX] - lsq) / np.abs(lsq))),
               beta_noise_c_ghost=float(bn[IGHOST]))
    out["ok"] = bool(out["beta_y_eq_2dH_minus_2_max"] <= TOL["L3"] and b2[IGHOST] == 0.0 and bn[IGHOST] == 0.0
                     and out["beta_noise_vs_lstsq_no_intercept_rel"] <= TOL["L3"])
    return out


def l3_atoms(ctx):
    TA, FT, torch = ctx.TA, ctx.FT, ctx.torch
    s = _synth(ctx.F.SEED + 3)
    a = _beta_checks(TA.beta_fit, s)
    rng, nr, nmax = s["rng"], s["nr"], 4
    D = FT.Data.__new__(FT.Data)                       # synthetic fu2 Data (only the attributes Std / batch / final_sh use)
    D.onehot = np.zeros((nr, nmax, 2))
    D.onehot[:, :, 0] = 1.0
    D.bin = rng.integers(0, 2, size=(nr, nmax, 1)).astype(float)
    D.num = rng.normal(size=(nr, nmax, 2))
    D.pos = rng.normal(size=(nr, nmax, 3))
    D.mask = np.ones((nr, nmax), dtype=bool)
    D.mask[:, -1] = rng.random(nr) < 0.5
    D.n = D.mask.sum(1).astype(int)
    D.etype = np.zeros((nr, nmax, nmax), dtype=np.int8)
    D.frag = np.zeros((nr, nmax), dtype=np.int8)
    D.glob = rng.normal(size=(nr, 3))
    D.y, D.ra, D.rb, D.dH, D.H10, D.arm = s["yn"], s["ra"], s["rb"], s["dH"], s["H10"], "SH"
    tr, te = s["tr"], s["te"]
    std = TA.StdSH(D, tr)
    beta_ref = np.asarray(TA.beta_fit(s["dH"][tr], s["yn"][tr]), float)
    T = std.tensors(D, dtype=torch.float64, device=torch.device("cpu"))

    class Zero(torch.nn.Module):
        """residual model with output 0 for every pair."""

        def forward(self, R, ia, ib):
            return torch.zeros((len(ia), len(TARGETS)), dtype=R["x"].dtype, device=R["x"].device)

    z = Zero()
    fin = TA.final_sh(z, T, D, std, s["ra"][te], s["rb"][te])
    want = std.beta * s["dH"][te]
    fin_s = FT.predict(z, T, s["ra"][te], s["rb"][te]) * std.sy
    c = dict(stdsh_beta_equals_beta_fit=bool(np.array_equal(std.beta, beta_ref)),
             stdsh_beta_rows_are_fit_rows=bool(set(np.asarray(std.beta_rows).tolist()) == set(tr.tolist())),
             zero_residual_final_minus_beta_dH_max=maxdiff(fin, want),
             zero_residual_S_final_max=float(np.abs(fin_s).max()),
             residual_target_minus_y_minus_beta_dH_max=maxdiff(std.r_tr, s["yn"][tr] - std.beta * s["dH"][tr]),
             baseline_median_abs=float(np.median(np.abs(want[:, HIDX]))))
    ok = (a["ok"] and c["stdsh_beta_equals_beta_fit"] and c["stdsh_beta_rows_are_fit_rows"]
          and c["zero_residual_final_minus_beta_dH_max"] <= TOL["L3"] and c["zero_residual_S_final_max"] <= TOL["L3"]
          and c["residual_target_minus_y_minus_beta_dH_max"] <= TOL["L3"] and c["baseline_median_abs"] > 0)
    return dict(status="pass" if ok else "fail", tol=TOL["L3"], beta=a, atoms=c,
                data="synthetic: 60 reactions, 400 pairs, H ~ N(0, 3^2), y = 2 dH (+ N(0, 0.3^2)), c_ghost random")


def l3_xgb(ctx):
    TX = ctx.TX
    s = _synth(ctx.F.SEED + 3)
    a = _beta_checks(TX.beta_fit, s)
    rng, tr, te = s["rng"], s["tr"], s["te"]
    X = rng.normal(size=(s["nr"], 6))
    gate = te[:50]
    out, ok = {}, a["ok"]
    for arm in ARMS:
        Dx = types.SimpleNamespace(X=X, ra=s["ra"], rb=s["rb"], y=s["yn"], dH=s["dH"], H10=s["H10"], arm=arm)
        pf = TX.prep_fit(Dx, tr, dict(val=te), gate)
        beta = np.asarray(pf["beta"], float)
        on = [arm == "SH" and j in HIDX for j in range(len(TARGETS))]
        exp = np.asarray(TX.beta_fit(s["dH"][tr], s["yn"][tr]), float) if arm == "SH" else np.zeros(len(TARGETS))
        want = np.stack([beta[j] * s["dH"][te, j] if on[j] else np.zeros(len(te)) for j in range(len(TARGETS))], 1)
        yhat0 = pf["base"]["val"] + np.zeros_like(pf["base"]["val"])          # residual model output 0
        ga, gb = s["ra"][gate], s["rb"][gate]
        want_ab = np.stack([beta[j] * (s["H10"][ga, j] - s["H10"][gb, j]) if on[j] else np.zeros(len(ga))
                            for j in range(len(TARGETS))], 1)
        resid = np.stack([pf["ys"][j] * pf["sy"][j] for j in range(len(TARGETS))], 1)
        want_r = np.stack([s["yn"][tr, j] - beta[j] * s["dH"][tr, j] if on[j] else s["yn"][tr, j]
                           for j in range(len(TARGETS))], 1)
        e = dict(beta_equal_expected=bool(np.array_equal(beta, exp)),
                 zero_residual_final_minus_beta_dH_max=maxdiff(yhat0, want),
                 gate_base_ab_minus_beta_dH_max=maxdiff(pf["base"]["ab"], want_ab),
                 gate_base_ba_plus_ab_max=maxdiff(pf["base"]["ba"], -pf["base"]["ab"]),
                 gate_base_aa_max=float(np.abs(pf["base"]["aa"]).max()),
                 residual_target_rel=maxrel(resid, want_r), beta_rows_are_rows=bool(pf["beta_rows"] is pf["rows"]))
        e["ok"] = bool(e["beta_equal_expected"] and e["zero_residual_final_minus_beta_dH_max"] <= TOL["L3"]
                       and e["gate_base_ab_minus_beta_dH_max"] <= TOL["L3"] and e["gate_base_ba_plus_ab_max"] <= TOL["L3"]
                       and e["gate_base_aa_max"] <= TOL["L3"] and e["residual_target_rel"] <= TOL["L3"]
                       and e["beta_rows_are_rows"])
        ok &= e["ok"]
        out[arm] = e
    ru = _l3_xgb_run_unit(TX, ctx.F.SEED + 33)
    ok &= ru["ok"]
    return dict(status="pass" if ok else "fail", tol=TOL["L3"], beta=a, xgb_prep_fit=out, xgb_run_unit=ru,
                data="synthetic: 60 reactions (6 features), 400 pairs, y = 2 dH (+ N(0, 0.3^2))")


def _l3_xgb_run_unit(TX, seed):
    """review C-2: the production train_xgb.run_unit on a synthetic XData with the XGB model replaced by a zero model
    (run_parallel -> 0 for every eval set): parquet yhat must be base = beta_final dH (S: 0) and every stored inner score
    mean |beta_j dH - y| on inner fold j's validation pairs (beta_j from that fold's training pairs), 1e-10."""
    rng = np.random.default_rng(seed)
    nr, ntr, nte = 60, 300, 100
    H10 = np.zeros((nr, len(TARGETS)))
    H10[:, HIDX] = rng.normal(size=(nr, len(HIDX))) * 3.0
    a_tr = rng.integers(0, 40, ntr)                                   # training pairs on reactions 0..39, test on 40..59
    b_tr = (a_tr + rng.integers(1, 40, ntr)) % 40                     # (no test reaction is a fit reaction, L-8)
    a_te = 40 + rng.integers(0, 20, nte)
    b_te = 40 + (a_te - 40 + rng.integers(1, 20, nte)) % 20
    ra, rb = np.r_[a_tr, a_te], np.r_[b_tr, b_te]
    dH = H10[ra] - H10[rb]
    y = 2.0 * dH + rng.normal(0.0, 0.3, size=dH.shape)
    y[:, IGHOST] = rng.normal(size=len(ra)) * 0.1
    tr, te = np.arange(ntr), np.arange(ntr, ntr + nte)
    fo = np.arange(ntr) % 4
    inner = [dict(train=tr[fo != j], val=tr[fo == j], dropped=dict(train=0, val=0)) for j in range(4)]
    sp = dict(train=tr, test=te, inner_train=inner[0]["train"], inner_val=inner[0]["val"], inner=inner,
              dropped={k: dict(no_tokens=0, not_usable=0) for k in ("train", "test", "inner_train", "inner_val")},
              pairs_sha="synthetic", val_sha="synthetic-val", test_sha="synthetic-test", inner_sha="synthetic-inner")
    pid = np.array([f"S{i:04d}" for i in range(len(ra))], dtype=object)
    X0 = rng.normal(size=(nr, 6))
    real = TX.run_parallel

    def zero(calls):                                                  # zero model output, same call contract
        return [({k: np.zeros(len(zd)) for k, (zd, zm) in c[5].items()}, 0.0) for c in calls]
    out, ok = {}, True
    TX.run_parallel = zero
    try:
        for arm in ARMS:
            X = X0 if arm == "S" else np.concatenate([X0, H10[:, HIDX]], 1)
            D = types.SimpleNamespace(
                rung="g1", arm=arm, X=X, ra=ra, rb=rb, y=y, dH=dH, H10=H10, pid=pid,
                P=pd.DataFrame(dict(ref_kind="reference", pos="p"), index=pid),
                sha={k: "synthetic" for k in ("inputs_table", "labels", "s_splits", "s_folds", "t_report", "t_manifest",
                                              "q_report", "x182")},
                pair_list_sha="synthetic", feat_cols=[f"x{i}" for i in range(X.shape[1])], feat_cols_sha="synthetic",
                hinfo=dict(h_cols=[f"H_{t}" for t in H9]), inputs={}, linfo={}, opened=[])
            D.split = lambda a_, b_=None, sp=sp: sp
            u = dict(rung="g1", arm=arm, model=TX.MODEL, fold=0)
            root = _rmtree(SCR / "l3_xgb" / arm)
            rec = TX.run_unit(D, u, root=root, cells=[0], smoke=True)
            pq = pd.read_parquet(root / f"{TX.uname(u)}.parquet")
            bf = np.asarray(TX.beta_fit(dH[tr], y[tr]), float) if arm == "SH" else np.zeros(len(TARGETS))
            dy, db, dmae, dval = 0.0, 0.0, 0.0, 0.0
            for j, t in enumerate(TARGETS):
                q = pq[pq.target == t]
                on = arm == "SH" and j in HIDX
                want = bf[j] * dH[te, j] if on else np.zeros(len(te))
                dy = max(dy, maxdiff(q["yhat"].to_numpy(float), want))
                db = max(db, maxdiff(q["base"].to_numpy(float), want))
                folds = []
                for f in inner:
                    bj = np.asarray(TX.beta_fit(dH[f["train"]], y[f["train"]]), float)[j] if on else 0.0
                    folds.append(float(np.mean(np.abs(bj * dH[f["val"], j] - y[f["val"], j]))))
                dmae = max(dmae, abs(float(rec["tune"][t]["grid_mae"][0]) - float(np.mean(folds))))
                dval = max(dval, abs(float(rec["val_kcal_mae"][t]) - folds[0]))
            e = dict(yhat_minus_beta_dH_max=dy, base_minus_beta_dH_max=db, inner_score_minus_line_mae_max=dmae,
                     val_kcal_mae_minus_fold0_line_mae_max=dval,
                     unit_beta_equals_beta_fit=bool(np.array_equal(np.asarray(rec["beta"], float), bf)),
                     ids_are_test=pq[pq.target == TARGETS[0]].pair_id.astype(str).tolist() == [str(p) for p in pid[te]])
            e["ok"] = bool(max(dy, db, dmae, dval) <= TOL["L3"] and e["unit_beta_equals_beta_fit"] and e["ids_are_test"])
            ok &= e["ok"]
            out[arm] = e
    finally:
        TX.run_parallel = real
    return dict(out, ok=bool(ok), method="train_xgb.run_unit (cells [0], smoke root $LAD/tests/l3_xgb) with "
                                         "run_parallel returning 0: the stored prediction is the baseline alone")


# ================================================================ L-4 (child process per tool x rung)
def l4_child(tool, rung, labels, out):
    """one fresh training process: import the trainer first (its audit hook), load S + SH, run a 1-cell smoke unit."""
    if tool == "atoms":
        import train_atoms as M
        model = "AW"
    elif tool == "xgb":
        import train_xgb as M
        model = "XGB"
    else:
        print(f"unknown tool {tool}", file=sys.stderr)
        return 2
    root = _rmtree(SCR / "l4" / f"{tool}_{rung}")
    explicit = set()
    for arm in ARMS:
        D = M.get_data(rung, arm, labels)
        u = dict(rung=rung, arm=arm, model=model, fold=0)
        if tool == "atoms":
            M.run_unit(D, u, root=root, n_epochs=1, seeds=M.F.FINAL_SEEDS[:1], cells=[0], log=None, smoke=True)
        else:
            M.run_unit(D, u, root=root, cells=[0], smoke=True)
        explicit |= {v for v in D.inputs.values() if isinstance(v, str) and v.startswith("/")}
    rec = dict(tool=tool, rung=rung, labels=str(labels), opened=sorted(set(M._OPEN_LOG)), n_open_events=len(M._OPEN_LOG),
               explicit=sorted(explicit), prefixes=[sys.prefix, sys.base_prefix, sys.exec_prefix],
               python=sys.executable, finished=now())
    write_json(out, rec)
    print(f"[l4child] {tool} {rung}: {len(rec['opened'])} opened paths, {len(explicit)} explicit", flush=True)
    return 0


def _classify(IN, rung, rec):
    pre = tuple(sorted({os.path.abspath(p).rstrip(os.sep) + os.sep for p in (rec.get("prefixes") or []) if p}))
    own = os.path.abspath(SCR) + os.sep
    white = {os.path.abspath(p) for p in SPLIT_WHITELIST}
    roots = IN.rung_roots()

    def lab(p):
        q, best = Path(p), None
        for root, lb in roots:
            if q == root or root in q.parents:
                if best is None or len(str(root)) > len(str(best[0])):
                    best = (root, lb)
        return None if best is None else best[1]
    opened, explicit = set(rec.get("opened") or []), set(rec.get("explicit") or [])
    counts, ignored, foreign, unknown, white_hit, data, n_own = Counter(), Counter(), [], [], set(), [], 0
    for p in sorted(opened | explicit):
        q, rp = os.path.abspath(p), os.path.realpath(p)
        if pre and (q.startswith(pre) or rp.startswith(pre)):
            ignored["interpreter"] += 1
            continue
        if q.startswith(own) or rp.startswith(own):
            n_own += 1
            continue
        if not (q.startswith(DATA_ROOTS) or rp.startswith(DATA_ROOTS)):
            ignored["/" + (q.split(os.sep)[1] if q.count(os.sep) > 1 else q)] += 1
            continue
        if q in white or rp in white:
            white_hit.add(q)
            continue
        lb, use = lab(q), q
        if lb is None:
            lb, use = lab(rp), rp
        if lb is None:
            unknown.append(q)
            continue
        counts[lb] += 1
        data.append(use)
        if lb != "shared" and rung not in lb.split("+"):
            foreign.append([q, lb])
    off = IN.check_paths(rung, data)
    tab = os.path.abspath(table_path(rung))
    n_rung = sum(v for k, v in counts.items() if rung in k.split("+") and k != "shared")
    res = dict(n_opened=len(opened), n_explicit=len(explicit), by_column=dict(counts), n_rung_column_paths=n_rung,
               n_foreign=len(foreign), foreign=foreign[:50], n_unknown=len(unknown), unknown=unknown[:50],
               rung_independent_allowed=sorted(white_hit), ignored_non_data=dict(ignored), n_own_outputs=n_own,
               table_opened=tab in opened, table_listed=tab in (opened | explicit),
               inputs_check_paths=dict(ok=bool(off["ok"]), n_foreign=len(off["foreign"]), foreign=off["foreign"][:20],
                                       n_unknown=len(off["unknown"]), unknown=off["unknown"][:20]))
    if rung == "g2":
        res["n_g1_or_dft_paths"] = sum(v for k, v in counts.items() if k != "shared" and "g2" not in k.split("+"))
    ok = (not foreign and not unknown and off["ok"] and not off["unknown"] and res["table_listed"] and n_rung > 0
          and len(opened) > 0)
    res["status"] = "pass" if ok else "fail"
    return res


def l4_check(ctx, tool):
    IN = ctx.IN

    def one(r):
        out = SCR / "l4" / f"{tool}_{r}.json"
        if out.exists():
            out.unlink()
        cmd = [sys.executable, str(Path(__file__).resolve()), "l4child", tool, r, str(ctx.labels), str(out)]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(LADC))
        print(p.stdout[-2000:], flush=True)
        if p.returncode != 0 or not out.is_file():
            return dict(status="waiting" if p.returncode == 2 else "fail", child_exit=p.returncode,
                        child_cmd=" ".join(cmd), stderr_tail=p.stderr[-3000:], stdout_tail=p.stdout[-1500:])
        res = _classify(IN, r, read_json(out))
        res.update(child_seconds=round(time.time() - t0, 1), child_record=str(out))
        return res
    res = per_rung(ctx, one)
    res.update(rule="each rung's training process opens only that rung's spec 4-1 column (+ rung-independent labels, "
                    "folds, reports); g2: no G1 / DFT path", rung_independent_files=[str(p) for p in SPLIT_WHITELIST])
    return res


# ================================================================ L-5
def _l5_trainer(D, Xin, glob_cols, r, seed):
    Tb = read_table(r)
    hcols = [f"H_{t}" for t in H9]
    rng = np.random.default_rng(seed)
    rids = np.asarray(D.rids).astype(int)
    ur = np.nonzero(np.asarray(D.usable_rxn))[0]
    up = np.nonzero(np.asarray(D.usable_pair))[0]
    if not len(ur) or not len(up):
        return dict(status="fail", why="no usable reaction / pair"), np.zeros(0, dtype=int)
    pick = rng.choice(ur, size=min(50, len(ur)), replace=False)
    rid = rids[pick]
    want = Tb.loc[rid, hcols].to_numpy(float)
    Xin = np.asarray(Xin)
    pp = rng.choice(up, size=min(50, len(up)), replace=False)
    a, b = rids[np.asarray(D.ra)[pp]], rids[np.asarray(D.rb)[pp]]
    want_dh = Tb.loc[a, hcols].to_numpy(float) - Tb.loc[b, hcols].to_numpy(float)
    tok = set(rids.tolist())
    tab_u = set(Tb.index[Tb.usable.astype(bool)].tolist())
    mine = set(rids[np.asarray(D.usable_rxn)].tolist())
    res = dict(n_reactions=int(len(pick)), n_pairs=int(len(pp)), n_input_columns=int(Xin.shape[1]),
               global_H_columns_vs_table=maxdiff(Xin[pick][:, 182:], want),
               H10_vs_table=maxdiff(np.asarray(D.H10)[pick][:, HIDX], want),
               global182_vs_table_rel=maxrel(Xin[pick][:, :182], Tb.loc[rid, list(glob_cols)].to_numpy(float)),
               dH_vs_table=maxdiff(np.asarray(D.dH)[pp][:, HIDX], want_dh),
               dH_c_ghost_max=float(np.abs(np.asarray(D.dH)[pp][:, IGHOST]).max()),
               usable_reactions_equal_table=mine == (tab_u & tok), table_usable_have_tokens=tab_u <= tok,
               n_usable_reactions=len(mine), n_usable_pairs=int(len(up)))
    ok = (res["n_input_columns"] == 191 and res["global_H_columns_vs_table"] <= TOL["L5"]
          and res["H10_vs_table"] <= TOL["L5"] and res["global182_vs_table_rel"] <= TOL["L5"]
          and res["dH_vs_table"] <= TOL["L5"] and res["dH_c_ghost_max"] == 0.0 and res["usable_reactions_equal_table"]
          and res["table_usable_have_tokens"])
    res["status"] = "pass" if ok else "fail"
    return res, rid


def _l5_api(ctx, r, D, pid, rid):
    """inputs.py: P_<rung> set file = the trainer's usable pairs; reaction_frame H = the trainer's H10."""
    IN = ctx.IN

    def f():
        sset = set(IN.read_set(r).pair_id.astype(str))
        mine = set(np.asarray(pid).astype(str)[np.asarray(D.usable_pair)].tolist())
        RF = IN.reaction_frame(r, "SH", "cr")
        rows = [int(np.nonzero(np.asarray(D.rids).astype(int) == q)[0][0]) for q in rid]
        dh = maxdiff(RF.loc[rid, [f"H_{t}" for t in H9]].to_numpy(float), np.asarray(D.H10)[rows][:, HIDX])
        usable_rf = set(RF.index.astype(int).tolist()) == set(np.asarray(D.rids).astype(int)[np.asarray(D.usable_rxn)].tolist())
        ok = sset == mine and dh <= TOL["L5"] and usable_rf
        return dict(status="pass" if ok else "fail", set_file_equals_trainer_pairs=sset == mine, n_set=len(sset),
                    n_trainer=len(mine), reaction_frame_H_vs_trainer=dh, reaction_frame_rows_equal_usable=usable_rf)
    return _api(ctx, f)


def _l5_provenance(r, seed):
    """the rung table itself: H0 = labels_d1_hf(_g2), channels unchanged, H - H0 = reference change x KCAL."""
    Tb = read_table(r)
    Hs = pd.read_parquet(H_PARQUET[r])
    Hs = Hs[Hs.geom == r].assign(rxn_id=lambda d: d.rxn_id.astype(int)).set_index("rxn_id")
    hh = Tb[Tb.has_h.astype(bool)]
    rng = np.random.default_rng(seed)
    pick = rng.choice(hh.index.to_numpy(), size=min(50, len(hh)), replace=False)

    def diffs(sub):
        e1 = (sub.e_ref1_hf_orig_eh - sub.e_ref1_hf_eh).to_numpy(float) * L.KCAL
        e2 = (sub.e_ref2_hf_orig_eh - sub.e_ref2_hf_eh).to_numpy(float) * L.KCAL
        out = dict(n=int(len(sub)), n_h_changed=int(sub.h_changed.astype(bool).sum()),
                   H0_vs_labels_d1_hf=maxdiff(sub[[f"H0_{t}" for t in H9]].to_numpy(float),
                                              Hs.loc[sub.index, H9].to_numpy(float)),
                   channels_H_equal_H0=all(np.array_equal(sub[f"H_{t}"].to_numpy(float), sub[f"H0_{t}"].to_numpy(float))
                                           for t in CH6),
                   d1_change_vs_ref1=maxdiff((sub.H_d1 - sub.H0_d1).to_numpy(float), e1),
                   d2_change_vs_ref2=maxdiff((sub.H_d2 - sub.H0_d2).to_numpy(float), e2),
                   barrier_change_vs_refs=maxdiff((sub.H_barrier - sub.H0_barrier).to_numpy(float), e1 + e2))
        out["ok"] = bool(out["H0_vs_labels_d1_hf"] <= TOL["L5"] and out["channels_H_equal_H0"]
                         and max(out["d1_change_vs_ref1"], out["d2_change_vs_ref2"], out["barrier_change_vs_refs"])
                         <= TOL["L5_reref"])
        return out
    s50, full = diffs(hh.loc[pick]), diffs(hh)
    refs = _l5_refs(Tb, Hs, r)
    return dict(status="pass" if s50["ok"] and full["ok"] and refs["ok"] else "fail", source=str(H_PARQUET[r]),
                sample50=s50, all_rows_with_h=full, reference_energies=refs)


def _l5_refs(Tb, Hs, r):
    """review C-9: the table's reference energies against independent sources. Every row with a reference energy:
    e_ref<k>_hf_orig_eh = labels_d1_hf(_g2) e_frag<k>_rel_eh of that row (exact); g rungs: e_ref<k>_hf_eh = the lowest
    e_ref<k>_hf_orig_eh of its ref<k>_key over the whole table (both fragment columns, exact); dft: src own -> new =
    orig, src Q -> new = labels_d1_hf e_frag<src_frag>_rel_eh of the source job (exact), src V -> finite."""
    Hj = Hs.reset_index().set_index("job_id")
    out, ok = {}, True
    long = []
    for k in (1, 2):
        sub = Tb[np.isfinite(Tb[f"e_ref{k}_hf_orig_eh"].to_numpy(float))]
        d = maxdiff(sub[f"e_ref{k}_hf_orig_eh"].to_numpy(float), Hs.loc[sub.index, f"e_frag{k}_rel_eh"].to_numpy(float))
        out[f"orig{k}_vs_labels_d1_hf"] = d
        ok &= d == 0.0
        long.append(pd.DataFrame(dict(key=sub[f"ref{k}_key"].astype(str).to_numpy(),
                                      e=sub[f"e_ref{k}_hf_orig_eh"].to_numpy(float))))
    if r != "dft":
        emin = pd.concat(long).groupby("key").e.min()
        # g2 built with P20 option (c) (inputs.py --g2-cons-own, P_build_g2.json g2_q_keys == 'own', decision
        # 2026-10-23): rows of constrained keys ('|fixed:' suffix, constrained rows only) keep their own reference
        own_mode = r == "g2" and (read_json(L.RES / "P_build_g2.json") or {}).get("g2_q_keys") == "own"
        out["g2_cons_own"] = bool(own_mode)
        for k in (1, 2):
            sub = Tb[np.isfinite(Tb[f"e_ref{k}_hf_orig_eh"].to_numpy(float))]
            keys = sub[f"ref{k}_key"].astype(str)
            cons = keys.str.contains("|fixed:", regex=False).to_numpy() if own_mode else np.zeros(len(sub), bool)
            new = sub[f"e_ref{k}_hf_eh"].to_numpy(float)
            d = maxdiff(new[~cons], emin.loc[keys[~cons]].to_numpy(float))
            out[f"new{k}_vs_key_min"] = d
            ok &= d == 0.0
            if own_mode:
                dc = maxdiff(new[cons], sub[f"e_ref{k}_hf_orig_eh"].to_numpy(float)[cons])
                out[f"new{k}_constrained_own_vs_orig"] = dict(n=int(cons.sum()), max_abs=dc)
                ok &= dc == 0.0
    else:
        for k in (1, 2):
            sub = Tb[np.isfinite(Tb[f"e_ref{k}_hf_orig_eh"].to_numpy(float))]
            src = sub[f"ref{k}_src"].astype(str)
            own, q, v = sub[src == "own"], sub[src == "Q"], sub[src == "V"]
            dq = maxdiff(q[f"e_ref{k}_hf_eh"].to_numpy(float),
                         np.array([Hj.at[j, f"e_frag{int(f)}_rel_eh"] for j, f in
                                   zip(q[f"ref{k}_src_job"].astype(str), q[f"ref{k}_src_frag"])], dtype=float)) \
                if len(q) else 0.0
            out[f"ref{k}"] = dict(n_own=int(len(own)), n_Q=int(len(q)), n_V=int(len(v)),
                                  own_new_equals_orig=maxdiff(own[f"e_ref{k}_hf_eh"].to_numpy(float),
                                                              own[f"e_ref{k}_hf_orig_eh"].to_numpy(float)),
                                  Q_new_vs_source_labels_d1_hf=dq,
                                  V_finite=bool(np.isfinite(v[f"e_ref{k}_hf_eh"].to_numpy(float)).all()),
                                  other_src=sorted(set(src) - {"own", "Q", "V"}))
            ok &= (out[f"ref{k}"]["own_new_equals_orig"] == 0.0 and dq == 0.0 and out[f"ref{k}"]["V_finite"]
                   and not out[f"ref{k}"]["other_src"])
    out["ok"] = bool(ok)
    return out


def l5_atoms(ctx):
    TA = ctx.TA

    def one(r):
        D = TA.get_data(r, "SH", ctx.labels)
        Ds = TA.get_data(r, "S", ctx.labels)
        res, rid = _l5_trainer(D, D.glob, D.trep["glob_cols"], r, L.SEED_M + 50 + RUNG_ORDER.index(r))
        s_ok = bool(Ds.d_glob == 182 and np.array_equal(Ds.usable_pair, D.usable_pair)
                    and np.array_equal(np.asarray(Ds.glob), np.asarray(D.glob)[:, :182]))
        res["S_arm"] = dict(d_glob=int(Ds.d_glob), same_usable_pairs_and_182_as_SH=s_ok)
        res["inputs_api"] = _l5_api(ctx, r, D, D.P.index, rid)
        res["status"] = combine([res["status"], "pass" if s_ok else "fail", res["inputs_api"]["status"]])
        return res
    return per_rung(ctx, one)


def l5_xgb(ctx):
    TX = ctx.TX

    def one(r):
        D = TX.get_data(r, "SH", ctx.labels)
        Ds = TX.get_data(r, "S", ctx.labels)
        res, rid = _l5_trainer(D, D.X, D.glob_cols, r, L.SEED_M + 50 + RUNG_ORDER.index(r))
        s_ok = bool(np.asarray(Ds.X).shape[1] == 182 and np.array_equal(Ds.usable_pair, D.usable_pair)
                    and np.array_equal(np.asarray(Ds.X), np.asarray(D.X)[:, :182], equal_nan=True))
        res["S_arm"] = dict(n_features=int(np.asarray(Ds.X).shape[1]), same_usable_pairs_and_182_as_SH=s_ok)
        res["table_provenance"] = guard(_l5_provenance, r, L.SEED_M + 60 + RUNG_ORDER.index(r))
        res["inputs_api"] = _l5_api(ctx, r, D, D.pid, rid)
        res["status"] = combine([res["status"], "pass" if s_ok else "fail", res["table_provenance"]["status"],
                                 res["inputs_api"]["status"]])
        return res
    return per_rung(ctx, one)


# ================================================================ L-6
def l6_atoms(ctx):
    TA, FT, Mo, torch = ctx.TA, ctx.FT, ctx.Mo, ctx.torch
    g1 = ctx.state["rungs"]["g1"]
    if g1["status"] != "ready":
        raise Waiting(g1["why"])
    cpu = torch.device("cpu")
    D0 = FT.Data()                                     # FU2's own data on the default FU2 roots
    Dl = TA.get_data("g1", "S", ORIG)
    arrays = {}
    for nm in ("rids", "n", "onehot", "bin", "num", "pos", "mask", "etype", "frag", "glob"):
        a, b = np.asarray(getattr(Dl, nm)), np.asarray(getattr(D0, nm))
        arrays[nm] = dict(equal=bool(a.dtype == b.dtype and a.shape == b.shape and arr_sha(a) == arr_sha(b)),
                          shape=list(a.shape), dtype=str(a.dtype))
    meta = dict(elements=list(Dl.elements) == list(D0.elements), num_cols=list(Dl.num_cols) == list(D0.num_cols),
                bin_cols=list(Dl.bin_cols) == list(D0.bin_cols), d_in=Dl.d_in == D0.d_in, d_glob=Dl.d_glob == D0.d_glob)
    folds, s_l, s_0 = {}, {}, {}
    for k in range(5):
        sl, s0 = Dl.split(k), D0.split("T1", k)
        s_l[k], s_0[k] = sl, s0
        f = {}
        for key in ("train", "test", "inner_train", "inner_val"):
            il, i0 = ids_of(Dl.P.index, sl[key]), ids_of(D0.P.index, s0[key])
            f[key] = dict(n=len(il), n_fu2=len(i0), same=il == i0)
        ends = all(np.array_equal(np.asarray(getattr(Dl, e))[sl[key]], np.asarray(getattr(D0, e))[s0[key]])
                   for e in ("ra", "rb") for key in ("train", "test")) \
            if f["train"]["same"] and f["test"]["same"] else False
        f.update(reaction_rows_equal=bool(ends), y_train_max_abs=maxdiff(Dl.y[sl["train"]], D0.y[s0["train"]]),
                 y_test_max_abs=maxdiff(Dl.y[sl["test"]], D0.y[s0["test"]]))
        folds[str(k)] = f
    tens, st_final = {}, None
    for stage, key in (("final", "train"), ("tuning", "inner_train")):
        st_l, st_0 = FT.Std(Dl, s_l[0][key]), FT.Std(D0, s_0[0][key])
        par = {p: maxdiff(getattr(st_l, p), getattr(st_0, p)) for p in ("mu", "sd", "gmu", "gsd", "sy")}
        Tl = st_l.tensors(Dl, dtype=torch.float32, device=cpu)
        sh_l = tensor_sha(Tl)
        del Tl
        T0 = st_0.tensors(D0, dtype=torch.float32, device=cpu)
        sh_0 = tensor_sha(T0)
        del T0
        tens[stage] = dict(std_max_abs=par, tensor_sha256_equal={k: sh_l[k] == sh_0[k] for k in sh_l},
                           tensor_sha256={k: v[:16] for k, v in sh_l.items()})
        if stage == "final":
            st_final = (st_l, st_0, sh_l)
    st_l, st_0, sh_l = st_final
    te_ids = ids_of(Dl.P.index, s_l[0]["test"])
    units, units_ok = {}, True
    for m in Mo.MODELS:
        name = f"T1__{m}__AG__f0"
        js, pt, pq = FU2_UNITS / f"{name}.json", FU2_UNITS / f"{name}__s0.pt", FU2_UNITS / f"{name}.parquet"
        rec = json.loads(js.read_text())
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        sp_ = obj["std"]
        dpt = {p: maxdiff(np.asarray(sp_[p], dtype=float), getattr(st_l, p)) for p in ("mu", "sd", "gmu", "gsd", "sy")}
        dsy = max(maxdiff(np.asarray(f["sy"], dtype=float), st_l.sy) for f in rec["final"])
        st_u, st_lu = copy.copy(st_0), copy.copy(st_l)         # the unit's own scaling statistics (.pt) on both data
        for p in ("mu", "sd", "gmu", "gsd", "sy"):
            setattr(st_u, p, np.asarray(sp_[p], dtype=float))
            setattr(st_lu, p, np.asarray(sp_[p], dtype=float))
        Tu = st_u.tensors(D0, dtype=torch.float32, device=cpu)
        sh_u = tensor_sha(Tu)
        del Tu
        Tlu = st_lu.tensors(Dl, dtype=torch.float32, device=cpu)
        sh_lu = tensor_sha(Tlu)
        del Tlu
        fp = rec.get("fingerprint") or {}
        fpc = dict(t_report=fp.get("t_report") == Dl.sha["t_report"], t_manifest=fp.get("t_manifest") == Dl.sha["t_manifest"],
                   q_report=fp.get("q_report") == Dl.sha["q_report"], s_splits=fp.get("s_splits") == Dl.sha["s_splits"])
        q = pd.read_parquet(pq)
        q = q.assign(pair_id=q.pair_id.astype(str))
        same_ids, yd = True, 0.0
        for j, t in enumerate(TARGETS):
            qt = q[q.target == t]
            same_ids &= qt.pair_id.tolist() == te_ids
            yd = max(yd, maxdiff(qt["y"].to_numpy(float), Dl.y[s_l[0]["test"], j]) if len(qt) == len(te_ids) else float("inf"))
        counts = dict(n_train=rec.get("n_train") == len(s_l[0]["train"]), n_test=rec.get("n_test") == len(s_l[0]["test"]),
                      n_inner_train=rec.get("n_inner_train") == len(s_l[0]["inner_train"]),
                      n_inner_val=rec.get("n_inner_val") == len(s_l[0]["inner_val"]))
        e = dict(pt_std_vs_loader_max_abs=dpt, json_final_sy_vs_loader_max_abs=dsy,
                 loader_tensors_with_unit_std_sha256_equal_unit={k: sh_lu[k] == sh_u[k] for k in sh_u},
                 info_loader_tensors_recomputed_std_sha256_equal_unit={k: sh_l[k] == sh_u[k] for k in sh_u},
                 fingerprint_inputs_equal=fpc, parquet_test_ids_equal=bool(same_ids), parquet_y_max_abs=yd,
                 counts_equal=counts, gpu=fp.get("gpu"))
        e["ok"] = bool(max(dpt.values()) <= TOL["L6"] and dsy <= TOL["L6"]
                       and all(e["loader_tensors_with_unit_std_sha256_equal_unit"].values())
                       and all(fpc.values()) and same_ids and yd <= TOL["L6"] and all(counts.values()))
        units_ok &= e["ok"]
        units[name] = e
    api = _api(ctx, lambda: _l6_inputs_loader(ctx, D0))
    ok = (all(v["equal"] for v in arrays.values()) and all(meta.values())
          and all(f[key]["same"] for f in folds.values() for key in ("train", "test", "inner_train", "inner_val"))
          and all(f["reaction_rows_equal"] and f["y_train_max_abs"] <= TOL["L6"] and f["y_test_max_abs"] <= TOL["L6"]
                  for f in folds.values())
          and all(max(v["std_max_abs"].values()) <= TOL["L6"] and all(v["tensor_sha256_equal"].values())
                  for v in tens.values()) and units_ok)
    del D0
    _cuda_free(torch)
    return dict(status=combine(["pass" if ok else "fail", api["status"]]), tol=TOL["L6"],
                loader="train_atoms.get_data('g1', 'S', original F_pairs.csv)", fu2="fu2 train.Data() (FU2 roots)",
                arrays=arrays, meta=meta, folds=folds, std_and_tensors=tens, fu2_units=units, inputs_LadderData=api)


def _l6_inputs_loader(ctx, D0):
    DL = ctx.IN.LadderData("g1", "S", ORIG, "orig")
    arrays = {nm: bool(arr_sha(np.asarray(getattr(DL, nm))) == arr_sha(np.asarray(getattr(D0, nm))))
              for nm in ("rids", "n", "onehot", "bin", "num", "pos", "mask", "etype", "frag", "glob")}
    folds = {}
    for k in range(5):
        a, b = DL.split("T1", k), D0.split("T1", k)
        folds[str(k)] = {key: ids_of(DL.P.index, a[key]) == ids_of(D0.P.index, b[key])
                         for key in ("train", "test", "inner_train", "inner_val")}
        folds[str(k)]["y_train_max_abs"] = maxdiff(DL.y[a["train"]], D0.y[b["train"]])
    ok = all(arrays.values()) and all(v for f in folds.values() for kk, v in f.items() if kk != "y_train_max_abs") \
        and all(f["y_train_max_abs"] <= TOL["L6"] for f in folds.values())
    out = dict(status="pass" if ok else "fail", arrays_equal_fu2=arrays, folds=folds,
               y_fu2_max_abs_diff=getattr(DL, "y_fu2_max_abs_diff", None))
    del DL
    return out


def l6_xgb(ctx):
    g1 = ctx.state["rungs"]["g1"]
    if g1["status"] != "ready":
        raise Waiting(g1["why"])
    rc = ctx.TX.cmd_l6()
    p = L.RES / "C_l6_xgb.json"
    rep = read_json(p)
    tg = rep.get("targets") or {}
    summ = {t: dict(ok=v.get("ok"), max_abs_yhat=v.get("max_abs_yhat"), max_abs_y=v.get("max_abs_y"),
                    best_equal=v.get("best_equal"), best=v.get("best"), inner_cv_mae_diff=v.get("inner_cv_mae_diff"),
                    train_mae_diff=v.get("train_mae_diff"), n_both=v.get("n_both"),
                    n_only_rev9=len(v.get("only_rev9") or []), n_only_mine=len(v.get("only_mine") or []))
            for t, v in tg.items()}
    yh = [v["max_abs_yhat"] for v in summ.values() if isinstance(v.get("max_abs_yhat"), (int, float))]
    ok = bool(rc == 0 and rep.get("ok") is True and set(tg) == set(TARGETS) and all(v["ok"] for v in summ.values())
              and len(yh) == len(TARGETS) and max(yh) <= TOL["L6_xgb"])
    return dict(status="pass" if ok else "fail", tol=TOL["L6_xgb"], exit=rc, file=str(p), sha256=sha(p),
                train_identical=rep.get("train_identical"), n_dropped_test=len(rep.get("dropped_test") or []),
                rev9_units=rep.get("rev9_units"), targets=summ)


# ================================================================ L-7 (atoms only)
def l7_atoms(ctx):
    TA, FT, F, torch = ctx.TA, ctx.FT, ctx.F, ctx.torch

    def T(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)

    def one(r):
        out, ok = {}, True
        for arm in ARMS:
            D = TA.get_data(r, arm, ctx.labels)
            rows = D.split(0)["train"]
            y = D.y[rows]
            yex = y.copy()
            yex[:, F.I_BAR] = yex[:, F.I_CH9].sum(1)                  # exact identity
            resid = y[:, F.I_BAR] - y[:, F.I_CH9].sum(1)
            if arm == "S":
                st = FT.Std(D, rows)
                sy = st.sy
                ys = y / sy
                _, l1, pen = FT.loss_fn(T(ys), T(ys), T(sy))
                e_ex = yex / sy
                _, _, pen_ex = FT.loss_fn(T(e_ex), T(e_ex), T(sy))
                e2 = e_ex.copy()
                e2[:, F.I_BAR] += 0.7
                _, _, pen_bar = FT.loss_fn(T(e2), T(e_ex), T(sy))
                expect = 0.7
                e = dict(loss="fu2 train.loss_fn")
            else:
                st = TA.StdSH(D, rows)
                sy, sr, base = st.sy, st.sr, st.base_tr
                rs = st.r_tr / sr
                _, l1, pen = TA.loss_sh(T(rs), T(rs), T(sr), T(base), T(sy))
                pen_np = float(np.mean(np.abs(resid)) / sy[F.I_BAR])
                rex = (yex - base) / sr
                _, _, pen_ex = TA.loss_sh(T(rex), T(rex), T(sr), T(base), T(sy))
                r2 = rex.copy()
                r2[:, F.I_BAR] += 0.7
                _, _, pen_bar = TA.loss_sh(T(r2), T(rex), T(sr), T(base), T(sy))
                expect = float(0.7 * sr[F.I_BAR] / sy[F.I_BAR])
                _, _, pen_nob = TA.loss_sh(T(rex), T(rex), T(sr), T(np.zeros_like(base)), T(sy))
                e = dict(loss="train_atoms.loss_sh", penalty_true_y_vs_numpy_final=abs(float(pen) - pen_np),
                         info_penalty_if_baseline_omitted=float(pen_nob), training_path=_l7_sh_path(ctx, D, rows))
            e.update(n_pairs=int(len(rows)), penalty_term_true_y=float(F.LAMBDA * pen), penalty_true_y=float(pen),
                     l1_true_y=float(l1), exact_identity_pen=float(pen_ex), pert_barrier_pen=float(pen_bar),
                     pert_barrier_expect=expect, s_barrier=float(sy[F.I_BAR]),
                     label_identity_residual_kcal=dict(mean=float(np.abs(resid).mean()), max=float(np.abs(resid).max())))
            e["ok"] = bool(e["penalty_term_true_y"] <= TOL["L7"] and e["l1_true_y"] == 0.0
                           and e["exact_identity_pen"] <= TOL["L7_exact"]
                           and abs(e["pert_barrier_pen"] - expect) <= TOL["L7_pert"]
                           and (arm == "S" or (e["penalty_true_y_vs_numpy_final"] <= TOL["L7_exact"]
                                               and e["training_path"]["ok"])))
            ok &= e["ok"]
            out[arm] = e
        return dict(status="pass" if ok else "fail", tol=TOL["L7"], arms=out)
    return per_rung(ctx, one)


def _l7_sh_path(ctx, D, rows):
    """review C-8: the SH loss as fit_sh really calls it. One epoch of train_atoms.fit_sh (AW cell 0, 512 fold-0 training
    pairs) with fu2 epoch_batches and train_atoms.loss_sh spied: per batch the loss gets r = yb[:, :10] and base =
    yb[:, 10:] of the batch, sr / sy = the fit's StdSH values, base = beta (H(a) - H(b)) of the batch's ORIENTED pair
    (sign flipped with the residual), and r * sr + base = the oriented label y (float32 tensors: relative 1e-6)."""
    TA, FT, F, Mo, torch = ctx.TA, ctx.FT, ctx.F, ctx.Mo, ctx.torch
    rng = np.random.default_rng(F.SEED + 77)
    sub = np.sort(rng.choice(rows, size=min(512, len(rows)), replace=False))
    nt = len(TARGETS)
    seen_b, seen_l = [], []
    real_eb, real_ls = FT.epoch_batches, TA.loss_sh

    def eb(rng_, ra_all, rb_all, ytr, bs):
        for ra, rb, yb in real_eb(rng_, ra_all, rb_all, ytr, bs):
            seen_b.append((np.asarray(ra).copy(), np.asarray(rb).copy(), yb.detach().double().cpu().numpy()))
            yield ra, rb, yb

    def ls(yhat, r, sr, base, sy):
        seen_l.append(tuple(x.detach().double().cpu().numpy() for x in (r, sr, base, sy)))
        return real_ls(yhat, r, sr, base, sy)
    FT.epoch_batches, TA.loss_sh = eb, ls
    try:
        fr = TA.fit_sh(D, "AW", Mo.GRID["AW"][0], TA.VARIANT, sub, None, seed=F.SEED, n_epochs=1)
    finally:
        FT.epoch_batches, TA.loss_sh = real_eb, real_ls
    st = fr["std"]
    lab = {}
    for p in sub:
        a, b = int(D.ra[p]), int(D.rb[p])
        lab.setdefault((a, b), []).append(D.y[p])
        lab.setdefault((b, a), []).append(-D.y[p])
    f32 = lambda x: np.asarray(np.float32(x), dtype=float)                 # noqa: E731
    rel = lambda a, b: float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max()) if np.size(b) else 0.0  # noqa: E731
    d_args = d_sr = d_base = d_y = 0.0
    for (ra, rb, yb), (r, sr, base, sy) in zip(seen_b, seen_l):
        d_args = max(d_args, float(np.abs(r - yb[:, :nt]).max()), float(np.abs(base - yb[:, nt:]).max()))
        d_sr = max(d_sr, float(np.abs(sr - f32(st.sr)).max()), float(np.abs(sy - f32(st.sy)).max()))
        d_base = max(d_base, rel(base, st.beta * (D.H10[ra] - D.H10[rb])))
        kc = r * f32(st.sr) + base
        for i in range(len(ra)):
            cands = lab.get((int(ra[i]), int(rb[i])), [])
            d_y = max(d_y, min((rel(kc[i], c) for c in cands), default=float("inf")))
    orig = {(int(D.ra[p]), int(D.rb[p])) for p in sub}
    n_sw = sum(int((int(a), int(b)) not in orig) for ra, rb, _ in seen_b for a, b in zip(ra, rb))
    out = dict(n_batches=len(seen_b), n_loss_calls=len(seen_l), n_pairs=int(len(sub)), loss_args_vs_batch_max=d_args,
               sr_sy_vs_std_max=d_sr, base_vs_oriented_beta_dH_rel=d_base, final_vs_oriented_y_rel=d_y,
               n_swapped_rows=int(n_sw))
    out["ok"] = bool(len(seen_b) == len(seen_l) > 0 and d_args == 0.0 and d_sr == 0.0 and d_base <= 1e-6 and d_y <= 1e-5
                     and out["n_swapped_rows"] > 0)
    return out


# ================================================================ L-1
def l1_atoms(ctx):
    TA, FT, F, Mo, torch = ctx.TA, ctx.FT, ctx.F, ctx.Mo, ctx.torch
    fu2t = _fu2_tests()

    def one(r):
        res, ok = {}, True
        for ai, arm in enumerate(ARMS):
            D = TA.get_data(r, arm, ctx.labels)
            sp = D.split(0)
            rng = np.random.default_rng(F.SEED + 11 + 10 * RUNG_ORDER.index(r) + ai)
            sub = rng.choice(sp["train"], size=min(1024, len(sp["train"])), replace=False)
            pick = rng.choice(sp["test"], size=min(F.ANTISYM_N, len(sp["test"])), replace=False)
            fitf, gatef = (FT.fit, FT.gate_m1) if arm == "S" else (TA.fit_sh, TA.gate_m1_sh)
            for i, m in enumerate(TA.MODELS):
                fr = fitf(D, m, Mo.GRID[m][0], TA.VARIANT, sub, None, seed=F.SEED, n_epochs=1)
                st = fr["std"]
                mod = fu2t.noisy64(fr["model"], F.SEED + 1000 * i + 100 * ai)
                T64 = st.tensors(D, dtype=torch.float64)
                a_, b_ = np.asarray(D.ra)[pick], np.asarray(D.rb)[pick]
                if arm == "S":
                    def g(x, y):
                        return FT.predict(mod, T64, x, y, bs=32) * st.sy
                else:
                    def g(x, y):
                        return TA.final_sh(mod, T64, D, st, x, y, bs=32)
                gab, gba, gaa = g(a_, b_), g(b_, a_), g(a_, a_)
                e = dict(anti_max=float(np.abs(gab + gba).max()), self_max=float(np.abs(gaa).max()),
                         median_abs_g=float(np.median(np.abs(gab))))
                if arm == "SH":
                    base = st.beta * (D.H10[a_] - D.H10[b_])
                    e["median_abs_baseline"] = float(np.median(np.abs(base[:, HIDX])))
                    # review C-6: the residual-model part alone must be non-vacuous too (beta dH meets 1e-2 by itself)
                    e["median_abs_model_part"] = float(np.median(np.abs(FT.predict(mod, T64, a_, b_, bs=32) * st.sr)))
                pg = gatef(mod, D, st, sp["test"], seed=F.SEED + i)
                e["production_gate"] = pg
                e["ok"] = bool(e["anti_max"] <= TOL["L1"] and e["self_max"] <= TOL["L1"]
                               and e["median_abs_g"] >= TOL["nonvacuity_median_abs_g"] and pg["ok"]
                               and (arm == "S" or (e["median_abs_baseline"] > 0 and e["median_abs_model_part"]
                                                   >= TOL["nonvacuity_median_abs_g"])))
                ok &= e["ok"]
                res[f"{arm}/{m}"] = e
                del fr, mod, T64
                _cuda_free(torch)
        return dict(status="pass" if ok else "fail", tol=TOL["L1"], n_pairs=int(F.ANTISYM_N), configs=res,
                    method="1-epoch fit (cell 0, 1,024 fold-0 training pairs), FU2 noisy64 (N(0, 0.3^2)), float64; "
                           "final prediction (SH incl. beta dH) on fold-0 test pairs")
    return per_rung(ctx, one)


def l1_xgb(ctx):
    TX = ctx.TX

    def one(r):
        res, ok = {}, True
        for ai, arm in enumerate(ARMS):
            D = TX.get_data(r, arm, ctx.labels)
            sp = D.split(0)
            rng = np.random.default_rng(TX.C.SEED + 11 + 10 * RUNG_ORDER.index(r) + ai)
            pick = rng.choice(sp["test"], size=min(TX.C.ANTISYM_N, len(sp["test"])), replace=False)
            pf = TX.prep_fit(D, sp["train"], dict(test=pick), pick)
            outs = TX.run_parallel([(pf["zd"], pf["zm"], pf["ys"][ti], pf["sy"][ti], TX.GRID[0], pf["evals"])
                                    for ti in range(len(TARGETS))])
            per_t, gab_all, kab_all = {}, [], []
            for ti, t in enumerate(TARGETS):
                kc, base = outs[ti][0], pf["base"]
                fab, fba, faa = base["ab"][:, ti] + kc["ab"], base["ba"][:, ti] + kc["ba"], base["aa"][:, ti] + kc["aa"]
                ge = TX.gate_eval(kc, base, ti)
                per_t[t] = dict(anti_max=float(np.abs(fab + fba).max()), self_max=float(np.abs(faa).max()),
                                model_anti_max=float(np.abs(kc["ab"] + kc["ba"]).max()),
                                model_self_max=float(np.abs(kc["aa"]).max()),
                                ab_equals_test_eval=bool(np.array_equal(kc["ab"], kc["test"])), production_gate=ge)
                gab_all.append(fab)
                kab_all.append(kc["ab"])
            med = float(np.median(np.abs(np.stack(gab_all))))
            med_m = float(np.median(np.abs(np.stack(kab_all))))            # review C-6: model part alone
            okc = bool(all(e["anti_max"] <= TOL["L1"] and e["self_max"] <= TOL["L1"] and e["production_gate"]["ok"]
                           for e in per_t.values()) and med >= TOL["nonvacuity_median_abs_g"]
                       and med_m >= TOL["nonvacuity_median_abs_g"])
            ok &= okc
            res[f"{arm}/XGB"] = dict(ok=okc, median_abs_g=med, median_abs_model_part=med_m, cell=TX.GRID[0],
                                     n_pairs=int(len(pick)), targets=per_t)
        return dict(status="pass" if ok else "fail", tol=TOL["L1"], configs=res,
                    method="GRID[0] fit on the fold-0 training pairs per target; (a,b), (b,a), (a,a) recomputed from x "
                           "in float64 (prep_fit); final = beta dH + g")
    return per_rung(ctx, one)


# ================================================================ spy runs (L-2, L-8, L-9)
def spy_atoms(ctx, r):
    """production train_atoms.run_unit for every arm x model (cell 0, 1 epoch, 1 seed) with fit / fit_sh spied."""
    TA, FT, F, torch = ctx.TA, ctx.FT, ctx.F, ctx.torch
    root = _rmtree(SCR / "spy_atoms" / r)
    calls = []
    real = {"S": FT.fit, "SH": TA.fit_sh}

    def make(arm):
        f0 = real[arm]

        def spy(Dx, m, cell, v, trr, val=None, **kw):
            res = f0(Dx, m, cell, v, trr, val, **kw)
            st = res["std"]
            rr = np.unique(np.r_[np.asarray(Dx.ra)[trr], np.asarray(Dx.rb)[trr]])
            five = rr[:5]
            x = res["T"]["x"]
            xn = x[torch.as_tensor(five, device=x.device)][:, :, -len(Dx.num_cols):].double().cpu().numpy()
            calls.append(dict(arm=arm, model=m, tr=np.asarray(trr).copy(), val=None if val is None else np.asarray(val).copy(),
                              pair_rows=np.asarray(st.pair_rows).copy(), rxn_rows=np.asarray(st.rxn_rows).copy(),
                              mu=np.array(st.mu, copy=True), sd=np.array(st.sd, copy=True), gmu=np.array(st.gmu, copy=True),
                              gsd=np.array(st.gsd, copy=True), sy=np.array(st.sy, copy=True),
                              sr=np.array(st.sr, copy=True) if arm == "SH" else None,
                              beta=np.array(st.beta, copy=True) if arm == "SH" else None,
                              beta_rows=np.asarray(st.beta_rows).copy() if arm == "SH" else None, five=five, xn=xn))
            return res
        return spy
    data = dict(D={}, sp={}, units={})
    FT.fit, TA.fit_sh = make("S"), make("SH")
    try:
        for arm in ARMS:
            D = TA.get_data(r, arm, ctx.labels)
            sp = D.split(0)
            data["D"][arm], data["sp"][arm] = D, sp
            for m in TA.MODELS:
                u = dict(rung=r, arm=arm, model=m, fold=0)
                n0 = len(calls)
                rec = TA.run_unit(D, u, root=root, n_epochs=1, seeds=F.FINAL_SEEDS[:1], cells=[0], log=None, smoke=True)
                pq = pd.read_parquet(root / f"{TA.uname(u)}.parquet")
                data["units"][(arm, m)] = dict(calls=calls[n0:], rec=rec, pq=pq)
                _cuda_free(torch)
    finally:
        FT.fit, TA.fit_sh = real["S"], real["SH"]
    return data


def spy_xgb(ctx, r):
    """production train_xgb.run_unit for both arms (cell 0) with prep_fit spied."""
    TX = ctx.TX
    root = _rmtree(SCR / "spy_xgb" / r)
    calls = []
    real = TX.prep_fit

    def spy(Dx, rows, evals, gate_rows):
        pf = real(Dx, rows, evals, gate_rows)
        first = not any(c["arm"] == Dx.arm for c in calls)
        c = dict(arm=Dx.arm, rows=np.asarray(rows).copy(), evals={k: np.asarray(v).copy() for k, v in evals.items()},
                 beta=np.asarray(pf["beta"], dtype=float).copy(), beta_rows_is_rows=pf["beta_rows"] is pf["rows"],
                 beta_rows=np.asarray(pf["beta_rows"]).copy(), rxn_rows=np.asarray(pf["rxn_rows"]).copy(),
                 sy=[float(v) for v in pf["sy"]])
        if first and "val" in pf["evals"]:
            c["z_val"] = tuple(np.array(z, copy=True) for z in pf["evals"]["val"])
        calls.append(c)
        return pf
    data = dict(D={}, sp={}, units={})
    TX.prep_fit = spy
    try:
        for arm in ARMS:
            D = TX.get_data(r, arm, ctx.labels)
            sp = D.split(0)
            data["D"][arm], data["sp"][arm] = D, sp
            u = dict(rung=r, arm=arm, model="XGB", fold=0)
            n0 = len(calls)
            rec = TX.run_unit(D, u, root=root, cells=[0], smoke=True)
            pq = pd.read_parquet(root / f"{TX.uname(u)}.parquet")
            data["units"][arm] = dict(calls=calls[n0:], rec=rec, pq=pq)
    finally:
        TX.prep_fit = real
    return data


def run_spies(ctx, fn):
    ctx.spy = {}
    for r in RUNG_ORDER:
        if ctx.state["rungs"][r]["status"] != "ready":
            continue
        holder = {}

        def f(r=r, holder=holder):
            holder["data"] = fn(ctx, r)
            return dict(status="pass")
        print(f"[C {ctx.part}] spy run_unit rung {r}  {now()}", flush=True)
        g = guard(f)
        ctx.spy[r] = dict(status=g["status"], err={k: v for k, v in g.items() if k != "status"}, data=holder.get("data"))


def _spy_of(ctx, r):
    s = ctx.spy.get(r)
    if s is None:
        return None, dict(status="fail", why="spy run missing for a ready rung")
    if s["status"] != "pass" or s["data"] is None:
        return None, dict(status=s["status"] if s["status"] != "pass" else "fail", spy_run=s["err"])
    return s["data"], None


def _parquet_base(pq, D, te, pid, beta_f, arm):
    te_list = ids_of(pid, te)
    ids_ok, dbase, dbeta = True, 0.0, 0.0
    for j, t in enumerate(TARGETS):
        qt = pq[pq.target == t]
        ids_ok &= qt.pair_id.astype(str).tolist() == te_list
        want = beta_f[j] * np.asarray(D.dH)[te, j] if arm == "SH" else np.zeros(len(te))
        dbase = max(dbase, maxdiff(qt["base"].to_numpy(float), want))
        dbeta = max(dbeta, float(np.abs(qt["beta"].to_numpy(float) - beta_f[j]).max()) if len(qt) else float("inf"))
    return dict(parquet_ids_are_test=bool(ids_ok), parquet_base_minus_beta_dH_max=dbase,
                parquet_beta_minus_final_beta_max=dbeta)


def l2_atoms(ctx):
    TA = ctx.TA

    def one(r):
        data, err = _spy_of(ctx, r)
        if err:
            return err
        res, ok = {}, True
        for (arm, m), u in data["units"].items():
            D, sp, calls = data["D"][arm], data["sp"][arm], u["calls"]
            e = dict(n_fit_calls=len(calls))
            if len(calls) != 2:
                e["ok"] = False
                res[f"{arm}/{m}"] = e
                ok = False
                continue
            ct, cf = calls
            e.update(tuning_train_is_inner_train=bool(np.array_equal(ct["tr"], sp["inner_train"])),
                     tuning_val_is_inner_val=bool(ct["val"] is not None and np.array_equal(ct["val"], sp["inner_val"])),
                     final_train_is_train=bool(np.array_equal(cf["tr"], sp["train"])), final_val_none=cf["val"] is None)
            te_ids = set(ids_of(D.P.index, sp["test"]))
            okb = True
            if arm == "SH":
                for tag, c, others in (("tuning", ct, set(ids_of(D.P.index, sp["inner_val"])) | te_ids),
                                       ("final", cf, te_ids)):
                    bids = set(ids_of(D.P.index, c["beta_rows"]))
                    want = np.asarray(TA.beta_fit(D.dH[c["tr"]], D.y[c["tr"]]), float)
                    e[tag] = dict(n_beta_pairs=len(bids), beta_pairs_equal_fit_pairs=set(c["beta_rows"].tolist())
                                  == set(c["tr"].tolist()), beta_equal_recomputed=bool(np.array_equal(c["beta"], want)),
                                  n_beta_pairs_in_validation_or_test=len(bids & others))
                    okb &= (e[tag]["beta_pairs_equal_fit_pairs"] and e[tag]["beta_equal_recomputed"]
                            and e[tag]["n_beta_pairs_in_validation_or_test"] == 0)
                beta_f = cf["beta"]
                e["unit_json_beta_equals_final_fit"] = bool(np.array_equal(np.asarray(u["rec"]["beta"], float), beta_f))
                okb &= e["unit_json_beta_equals_final_fit"]
            else:
                beta_f = np.zeros(len(TARGETS))
            e.update(_parquet_base(u["pq"], D, sp["test"], D.P.index, beta_f, arm))
            e["ok"] = bool(okb and e["tuning_train_is_inner_train"] and e["tuning_val_is_inner_val"]
                           and e["final_train_is_train"] and e["final_val_none"] and e["parquet_ids_are_test"]
                           and e["parquet_base_minus_beta_dH_max"] == 0.0 and e["parquet_beta_minus_final_beta_max"] == 0.0)
            ok &= e["ok"]
            res[f"{arm}/{m}"] = e
        return dict(status="pass" if ok else "fail", units=res)
    res = per_rung(ctx, one)
    res["rule"] = "beta of every SH fit = beta_fit on exactly that fit's training pairs; no validation / test pair_id"
    return res


def l2_xgb(ctx):
    TX = ctx.TX

    def one(r):
        data, err = _spy_of(ctx, r)
        if err:
            return err
        res, ok = {}, True
        for arm, u in data["units"].items():
            D, sp, calls = data["D"][arm], data["sp"][arm], u["calls"]
            nin = len(sp["inner"])
            e = dict(n_prep_fit_calls=len(calls), expected=nin + 1)
            if len(calls) != nin + 1:
                e["ok"] = False
                res[arm] = e
                ok = False
                continue
            te_ids = set(ids_of(D.pid, sp["test"]))
            seq, fits = True, []
            for j, c in enumerate(calls):
                if j < nin:
                    f = sp["inner"][j]
                    seq &= bool(np.array_equal(c["rows"], f["train"]) and np.array_equal(c["evals"].get("val"), f["val"]))
                    others = set(ids_of(D.pid, f["val"])) | te_ids
                else:
                    seq &= bool(np.array_equal(c["rows"], sp["train"]) and np.array_equal(c["evals"].get("test"), sp["test"])
                                and np.array_equal(c["evals"].get("train"), sp["train"]))
                    others = te_ids
                want = np.asarray(TX.beta_fit(D.dH[c["rows"]], D.y[c["rows"]]), float) if arm == "SH" \
                    else np.zeros(len(TARGETS))
                bids = set(ids_of(D.pid, c["beta_rows"]))
                fits.append(dict(fit="final" if j == nin else f"inner{j}", beta_rows_is_rows=c["beta_rows_is_rows"],
                                 beta_equal_expected=bool(np.array_equal(c["beta"], want)),
                                 n_beta_pairs_in_validation_or_test=len(bids & others),
                                 n_fit_rxns_in_test=len(set(c["rxn_rows"].tolist()) & rx_of(D, sp["test"]))))
            beta_f = calls[-1]["beta"]
            e.update(call_sequence_ok=bool(seq), fits=fits,
                     unit_json_beta_equals_final_fit=bool(np.array_equal(np.asarray(u["rec"]["beta"], float), beta_f)),
                     **_parquet_base(u["pq"], D, sp["test"], D.pid, beta_f, arm))
            e["ok"] = bool(seq and all(f["beta_rows_is_rows"] and f["beta_equal_expected"]
                                       and f["n_beta_pairs_in_validation_or_test"] == 0 and f["n_fit_rxns_in_test"] == 0
                                       for f in fits)
                           and e["unit_json_beta_equals_final_fit"] and e["parquet_ids_are_test"]
                           and e["parquet_base_minus_beta_dH_max"] == 0.0 and e["parquet_beta_minus_final_beta_max"] == 0.0)
            ok &= e["ok"]
            res[arm] = e
        return dict(status="pass" if ok else "fail", units=res)
    res = per_rung(ctx, one)
    res["rule"] = "beta of every SH prep_fit = beta_fit on exactly its training pairs; no validation / test pair_id"
    return res


# ================================================================ L-8 (FU2 C7 / C8)
def _l8_atoms_splits(ctx, r):
    TA, FT = ctx.TA, ctx.FT
    out, ok = {}, True
    for arm in ARMS:
        D = TA.get_data(r, arm, ctx.labels)
        core = D.P.core.astype(str).to_numpy()
        for k in range(5):
            s = D.split(k)
            te_rx = rx_of(D, s["test"])
            c = {}
            for stage, rows, others in (("final", s["train"], {"test": s["test"]}),
                                        ("tuning", s["inner_train"], {"test": s["test"], "inner_val": s["inner_val"]})):
                st = FT.Std(D, rows) if arm == "S" else TA.StdSH(D, rows)
                pr = set(np.asarray(st.pair_rows).tolist())
                c[stage] = dict(stat_rxns_in_test=len(set(np.asarray(st.rxn_rows).tolist()) & te_rx),
                                stat_rxns_outside_fit=len(set(np.asarray(st.rxn_rows).tolist()) - rx_of(D, rows)),
                                stat_pairs_equal_fit=pr == set(np.asarray(rows).tolist()),
                                **{f"stat_pairs_in_{n}": len(pr & set(np.asarray(o).tolist())) for n, o in others.items()})
                if arm == "SH":
                    c[stage]["beta_pairs_equal_fit"] = set(np.asarray(st.beta_rows).tolist()) == pr
            it, iv = set(s["inner_train"].tolist()), set(s["inner_val"].tolist())
            trs, tes = set(s["train"].tolist()), set(s["test"].tolist())
            c["c8"] = dict(cover=(it | iv) == trs, inner_overlap=len(it & iv), val_in_test=len(iv & tes),
                           val_core_in_inner_train=len(set(core[sorted(iv)].tolist()) & set(core[sorted(it)].tolist())),
                           val_fraction=len(iv) / max(1, len(trs)), test_in_train=len(trs & tes),
                           test_rxn_in_train=len(rx_of(D, s["train"]) & te_rx))
            okk = all(v["stat_rxns_in_test"] == 0 and v["stat_rxns_outside_fit"] == 0 and v["stat_pairs_equal_fit"]
                      and all(v[x] == 0 for x in v if x.startswith("stat_pairs_in_")) and v.get("beta_pairs_equal_fit", True)
                      for kk, v in c.items() if kk != "c8")
            c8 = c["c8"]
            okk &= (c8["cover"] and c8["inner_overlap"] == 0 and c8["val_in_test"] == 0 and c8["val_core_in_inner_train"] == 0
                    and 0.15 <= c8["val_fraction"] <= 0.35 and c8["test_in_train"] == 0 and c8["test_rxn_in_train"] == 0)
            c["ok"] = bool(okk)
            ok &= c["ok"]
            out[f"{arm}/f{k}"] = c
    return ok, out


def _l8_atoms_spy(ctx, data):
    C = ctx.C
    out, ok = {}, True
    for (arm, m), u in data["units"].items():
        D, sp = data["D"][arm], data["sp"][arm]
        te_rx = rx_of(D, sp["test"])
        for tag, c in zip(("tuning", "final"), u["calls"]):
            tr = c["tr"]
            rr = np.unique(np.r_[np.asarray(D.ra)[tr], np.asarray(D.rb)[tr]])
            vals = D.num[rr][D.mask[rr]]
            mu, sd = vals.mean(0), vals.std(0)
            sd[sd < C.SCALE_EPS] = 1.0
            gg = np.asarray(D.glob)[rr]
            gmu, gsd = gg.mean(0), gg.std(0)
            gsd[gsd < C.SCALE_EPS] = 1.0
            d = dict(rxn_rows_equal=bool(np.array_equal(c["rxn_rows"], rr)), mu=maxdiff(c["mu"], mu), sd=maxdiff(c["sd"], sd),
                     gmu=maxdiff(c["gmu"], gmu), gsd=maxdiff(c["gsd"], gsd), sy=maxdiff(c["sy"], D.y[tr].std(0)),
                     stat_rxns_in_test=len(set(rr.tolist()) & te_rx))
            if arm == "SH":
                d["sr"] = maxdiff(c["sr"], (D.y[tr] - c["beta"] * D.dH[tr]).std(0))
            five = c["five"]
            want = ((D.num[five] - mu) / sd)[D.mask[five]]
            got = c["xn"][D.mask[five]]
            d["fed_x_rel"] = float((np.abs(got - want) / np.maximum(1.0, np.abs(want))).max())
            stats = [d[x] for x in ("mu", "sd", "gmu", "gsd", "sy", "sr") if x in d]
            d["ok"] = bool(d["rxn_rows_equal"] and max(stats) <= TOL["L8_stats"] and d["stat_rxns_in_test"] == 0
                           and d["fed_x_rel"] <= TOL["L8_fed_x"])
            ok &= d["ok"]
            out[f"{arm}/{m}/{tag}"] = d
    return ok, out


def l8_atoms(ctx):
    def one(r):
        ok1, splits = _l8_atoms_splits(ctx, r)
        data, err = _spy_of(ctx, r)
        if err:
            return dict(status=combine(["pass" if ok1 else "fail", err["status"]]), folds=splits, run_unit=err)
        ok2, spy = _l8_atoms_spy(ctx, data)
        return dict(status="pass" if ok1 and ok2 else "fail", folds=splits, run_unit=spy)
    res = per_rung(ctx, one)
    res["tol"] = dict(stats=TOL["L8_stats"], fed_x_rel=TOL["L8_fed_x"])
    return res


def l8_xgb(ctx):
    TX = ctx.TX

    def one(r):
        out, ok = {}, True
        for arm in ARMS:
            D = TX.get_data(r, arm, ctx.labels)
            for k in range(5):
                s = D.split(k)
                trs, tes = set(s["train"].tolist()), set(s["test"].tolist())
                te_rx = rx_of(D, s["test"])
                inner = []
                for f in s["inner"]:
                    a, b = set(f["train"].tolist()), set(f["val"].tolist())
                    inner.append(dict(cover=(a | b) == trs, overlap=len(a & b), val_in_test=len(b & tes),
                                      core_overlap=len(set(np.asarray(D.core)[f["val"]].tolist())
                                                       & set(np.asarray(D.core)[f["train"]].tolist())),
                                      fit_rxns_in_test=len(rx_of(D, f["train"]) & te_rx)))
                e = dict(test_in_train=len(trs & tes), test_rxn_in_train=len(rx_of(D, s["train"]) & te_rx), inner=inner)
                e["ok"] = bool(e["test_in_train"] == 0 and e["test_rxn_in_train"] == 0
                               and all(x["cover"] and x["overlap"] == 0 and x["val_in_test"] == 0 and x["core_overlap"] == 0
                                       and x["fit_rxns_in_test"] == 0 for x in inner))
                ok &= e["ok"]
                out[f"{arm}/f{k}"] = e
        data, err = _spy_of(ctx, r)
        if err:
            return dict(status=combine(["pass" if ok else "fail", err["status"]]), folds=out, run_unit=err)
        spy = {}
        for arm, u in data["units"].items():
            D, sp, calls = data["D"][arm], data["sp"][arm], u["calls"]
            c0 = calls[0] if calls else None
            if c0 is None or "z_val" not in c0:
                spy[arm] = dict(ok=False, why="first prep_fit call (inner fold 0) not captured")
                ok = False
                continue
            rows, val, beta = c0["rows"], c0["evals"]["val"], c0["beta"]
            on = [arm == "SH" and j in HIDX for j in range(len(TARGETS))]
            R = [D.y[rows, j] - beta[j] * D.dH[rows, j] if on[j] else D.y[rows, j].copy() for j in range(len(TARGETS))]
            sc = TX.D1T.Scale(D.X[np.asarray(D.ra)[rows]], D.X[np.asarray(D.rb)[rows]], R[0])
            zd, zm = sc.z(D.X[np.asarray(D.ra)[val]], D.X[np.asarray(D.rb)[val]])
            e = dict(scale_from_fit_pairs_reproduces_val_z=bool(np.array_equal(zd, c0["z_val"][0])
                                                                and np.array_equal(zm, c0["z_val"][1])),
                     sy_max_abs=max(abs(float(np.std(R[j])) - c0["sy"][j]) for j in range(len(TARGETS))),
                     fit_rxns_in_test=[len(set(c["rxn_rows"].tolist()) & rx_of(D, sp["test"])) for c in calls])
            e["ok"] = bool(e["scale_from_fit_pairs_reproduces_val_z"] and e["sy_max_abs"] <= TOL["L8_stats"]
                           and not any(e["fit_rxns_in_test"]))
            ok &= e["ok"]
            spy[arm] = e
        return dict(status="pass" if ok else "fail", folds=out, run_unit=spy)
    return per_rung(ctx, one)


# ================================================================ L-9
def l9_atoms(ctx):
    C = ctx.C

    def one(r):
        data, err = _spy_of(ctx, r)
        if err:
            return err
        from sklearn.model_selection import GroupKFold
        DS, DH = data["D"]["S"], data["D"]["SH"]
        want = ids_of(DS.P.index, data["sp"]["S"]["inner_val"])
        want_sha = C.list_sha([(q,) for q in want])
        units = {}
        for (arm, m), u in data["units"].items():
            ct = u["calls"][0] if u["calls"] else None
            got = ids_of(data["D"][arm].P.index, ct["val"]) if ct is not None and ct["val"] is not None else None
            units[f"{arm}/{m}"] = dict(tuning_validation_pairs_equal=got == want,
                                       unit_val_pairs_sha256_equal=u["rec"].get("val_pairs_sha256") == want_sha)
        usable = set(np.asarray(DS.P.index).astype(str)[np.asarray(DS.usable_pair)].tolist())
        folds, val_sha = {}, {}
        for k in range(5):
            sS, sH = DS.split(k), DH.split(k)
            rec = DS.folds[k]
            cores = DS.P.loc[list(rec["train"]), "core"].to_numpy()
            _, b = next(iter(GroupKFold(C.N_INNER).split(rec["train"], groups=cores)))
            filt = [q for q in rec["inner_val"] if q in usable]
            folds[str(k)] = dict(S_equal_SH=sS["val_sha"] == sH["val_sha"],
                                 equals_split_file_inner_val_filtered=ids_of(DS.P.index, sS["inner_val"]) == filt,
                                 n=int(len(sS["inner_val"])), n_split_file=len(rec["inner_val"]),
                                 info_groupkfold_first_fold_equals_split_file=[rec["train"][i] for i in b]
                                 == list(rec["inner_val"]))
            val_sha[str(k)] = sS["val_sha"]
        ok = all(all(v.values()) for v in units.values()) and \
            all(f["S_equal_SH"] and f["equals_split_file_inner_val_filtered"] for f in folds.values())
        return dict(status="pass" if ok else "fail", units=units, folds=folds, val_sha=val_sha, n_val_f0=len(want))
    res = per_rung(ctx, one)
    res["rule"] = ("validation pairs = the first inner fold of the outer training side (FU2 split inner_val, P_<rung> "
                   "filter) for AW / GNN / TF (here) and XGB (cpu part); merge compares the sha256 per rung x fold")
    return res


def l9_xgb(ctx):
    C = ctx.C

    def one(r):
        data, err = _spy_of(ctx, r)
        if err:
            return err
        DS, DH = data["D"]["S"], data["D"]["SH"]
        want = ids_of(DS.pid, data["sp"]["S"]["inner_val"])
        want_sha = C.list_sha([(q,) for q in want])
        units = {}
        for arm, u in data["units"].items():
            c0 = u["calls"][0] if u["calls"] else None
            got = ids_of(data["D"][arm].pid, c0["evals"]["val"]) if c0 is not None and "val" in c0["evals"] else None
            units[arm] = dict(inner_fold0_validation_pairs_equal=got == want,
                              unit_val_pairs_sha256_equal=u["rec"].get("val_pairs_sha256") == want_sha)
        usable = set(np.asarray(DS.pid).astype(str)[np.asarray(DS.usable_pair)].tolist())
        folds, val_sha = {}, {}
        for k in range(5):
            sS, sH = DS.split(k), DH.split(k)
            rec = DS.folds[k]
            filt = [q for q in rec["inner_val"] if q in usable]
            folds[str(k)] = dict(S_equal_SH=sS["val_sha"] == sH["val_sha"],
                                 inner0_val_equals_inner_val=bool(np.array_equal(sS["inner"][0]["val"], sS["inner_val"])),
                                 equals_split_file_inner_val_filtered=ids_of(DS.pid, sS["inner_val"]) == filt,
                                 n=int(len(sS["inner_val"])))
            val_sha[str(k)] = sS["val_sha"]
        ok = all(all(v.values()) for v in units.values()) and all(all(v for kk, v in f.items() if kk != "n")
                                                                  for f in folds.values())
        return dict(status="pass" if ok else "fail", units=units, folds=folds, val_sha=val_sha, n_val_f0=len(want))
    return per_rung(ctx, one)


# ================================================================ L-10
def _line(beta_fit, fold, y10, dH10):
    pred, betas = np.full_like(y10, np.nan), {}
    for k in sorted(np.unique(fold).tolist()):
        tr = fold != k
        b = np.asarray(beta_fit(dH10[tr], y10[tr]), dtype=float)
        pred[~tr] = b * dH10[~tr]
        betas[int(k)] = b
    return pred, betas


def _vs_A(A, r, pos, y10, pred, betas):
    para = np.asarray(pos) == "p"
    rows, ok = {}, True
    for t in H9:
        j = TARGETS.index(t)
        a = A[(A.geom == r) & (A.target == t)]
        if len(a) != 1:
            rows[t] = dict(ok=False, why=f"{len(a)} A_summary rows for {r} / {t}")
            ok = False
            continue
        a = a.iloc[0]
        e = np.abs(pred[:, j] - y10[:, j])
        bA = json.loads(a["beta_by_fold"])
        x = dict(n=int(len(y10)), n_A=int(a["n"]), n_para=int(para.sum()), n_para_A=int(a["n_para"]),
                 mae_para=float(e[para].mean()), mae_para_A=float(a["mae_para"]), mae_all=float(e.mean()),
                 mae_all_A=float(a["mae_all"]))
        x["d_mae_para"] = abs(x["mae_para"] - x["mae_para_A"])
        x["d_mae_all"] = abs(x["mae_all"] - x["mae_all_A"])
        x["beta_max_abs_diff"] = (max(abs(float(betas[int(k)][j]) - float(v)) for k, v in bA.items())
                                  if set(int(k) for k in bA) == set(betas) else float("inf"))
        x["ok"] = bool(x["n"] == x["n_A"] and x["n_para"] == x["n_para_A"] and x["d_mae_para"] <= TOL["L10"])
        ok &= x["ok"]
        rows[t] = x
    dm = [v["d_mae_para"] for v in rows.values() if "d_mae_para" in v]
    return dict(status="pass" if ok else "fail", max_d_mae_para=max(dm) if dm else None, targets=rows)


_EV: dict = {}


def _evaluate_line_beta():
    """review C-3: the LINE function of analysis/d1_ladder/evaluate.py (spec 8 lists evaluate.py as Phase C code; spec
    4-5 LINE). Contract for its author: evaluate.line_beta(dH, y) -> beta [10] (TARGETS order) fitted on exactly the
    given pairs, no intercept, c_ghost 0; the E1 LINE rows must be beta * dH with this function. Imported by file path
    under the name lad_evaluate (FU2 Phase E may bring an analysis/d1_pair/fu2/evaluate.py that is earlier on
    sys.path). Missing file -> Waiting (C_tests cannot reach all_pass without it)."""
    if "f" in _EV:
        return _EV["f"]
    p = LADC / "evaluate.py"
    if not p.is_file():
        raise Waiting(f"{p} not written yet: L-10 must test its LINE function evaluate.line_beta(dH, y) (C-11)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lad_evaluate", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["lad_evaluate"] = m
    spec.loader.exec_module(m)
    f = getattr(m, "line_beta", None)
    if not callable(f):
        raise RuntimeError(f"{p} has no callable line_beta(dH, y) (contract in tests/run_tests.py C-11)")
    _EV["f"] = f
    return f


def l10_part(ctx, beta_fit, load_labels, who, with_evaluate=False):
    A = pd.read_csv(A_SUMMARY)
    P, linfo = load_labels(ORIG)
    S = pd.read_csv(S_FOLDS, dtype={"pair_id": str})
    S = S[S.in_P_main.astype(bool) & (S.ref_kind == "reference")].reset_index(drop=True)
    dd = [f"dd_{t}" for t in TARGETS]

    def one(r):
        Tb = read_table(r)
        h0 = Tb[[f"H0_{t}" for t in H9]]
        okr = set(h0.index[np.isfinite(h0.to_numpy(float)).all(1)].tolist())
        q = S[S.a_rxn.astype(int).isin(okr) & S.b_rxn.astype(int).isin(okr)].reset_index(drop=True)
        y10 = P.loc[q.pair_id.astype(str), dd].to_numpy(float)
        dH10 = np.zeros_like(y10)
        for t in H9:
            dH10[:, TARGETS.index(t)] = (h0.loc[q.a_rxn.astype(int), f"H0_{t}"].to_numpy(float)
                                         - h0.loc[q.b_rxn.astype(int), f"H0_{t}"].to_numpy(float))
        pred, betas = _line(beta_fit, q.fold_T1.to_numpy(), y10, dH10)
        res = _vs_A(A, r, q["pos"].to_numpy(), y10, pred, betas)
        res.update(beta_fit=who, source="rung table H0_<t> (= labels_d1_hf values) x original F_pairs.csv",
                   n_pairs_P_main_reference=int(len(S)), n_pairs_with_H=int(len(q)))

        def api():
            pf = ctx.IN.pairs_frame(r, ORIG, "orig", subset="has_h")
            pf = pf[pf.in_P_main.astype(bool) & (pf.ref_kind == "reference")].reset_index(drop=True)
            y2 = pf[dd].to_numpy(float)
            d2 = np.zeros_like(y2)
            for t in H9:
                d2[:, TARGETS.index(t)] = pf[f"dH_{t}"].to_numpy(float)
            p2, b2 = _line(beta_fit, pf.fold_T1.to_numpy(), y2, d2)
            rr = _vs_A(A, r, pf["pos"].to_numpy(), y2, p2, b2)
            same = pf.pair_id.astype(str).tolist() == q.pair_id.astype(str).tolist()
            rr.update(same_pairs_as_direct=same, pred_vs_direct_max=maxdiff(p2[:, HIDX], pred[:, HIDX]) if same
                      else float("inf"))
            if not (same and rr["pred_vs_direct_max"] <= TOL["L10"]):
                rr["status"] = "fail"
            return rr
        res["inputs_pairs_frame"] = _api(ctx, api)
        st = [res["status"], res["inputs_pairs_frame"]["status"]]
        if with_evaluate:
            def ev():
                pe, be = _line(_evaluate_line_beta(), q.fold_T1.to_numpy(), y10, dH10)
                rr = _vs_A(A, r, q["pos"].to_numpy(), y10, pe, be)
                rr["pred_vs_beta_fit_max"] = maxdiff(pe[:, HIDX], pred[:, HIDX])
                if rr["pred_vs_beta_fit_max"] > TOL["L10"]:
                    rr["status"] = "fail"
                return rr
            res["evaluate_line"] = guard(ev)
            st.append(res["evaluate_line"]["status"])
        res["status"] = combine(st)
        return res
    res = per_rung(ctx, one, rungs=STAGE1)
    res.update(tol=TOL["L10"], reference=str(A_SUMMARY), reference_sha256=sha(A_SUMMARY), labels=linfo.get("path"),
               rule="D1-HF Phase A: P_main reference pairs, truth = beta x HF dd without intercept, beta from "
                    "fold_T1 != k applied to fold k; gate n, n_para, para MAE")
    return res


# ================================================================ L-11
def _label_frames():
    Fn = pd.read_csv(CR2, float_precision="round_trip", dtype={"pair_id": str}).set_index("pair_id")
    Fo = pd.read_csv(ORIG, float_precision="round_trip", dtype={"pair_id": str}).set_index("pair_id")
    return Fn, Fo


def _l11_compare(y, ids, Fn, Fo):
    y = np.asarray(y, dtype=float)
    if len(set(ids) - set(Fn.index)) or len(set(ids) - set(Fo.index)):
        return dict(status="fail", why="loader pair ids not in the label files")
    d_new = {t: maxdiff(y[:, j], Fn.loc[ids, f"dd_{t}"].to_numpy(float)) for j, t in enumerate(TARGETS)}
    d_old = {t: maxdiff(y[:, TARGETS.index(t)], Fo.loc[ids, f"dd_{t}"].to_numpy(float)) for t in UNCHANGED}
    n_ch = {t: int((np.abs(y[:, TARGETS.index(t)] - Fo.loc[ids, f"dd_{t}"].to_numpy(float)) > TOL["L11"]).sum())
            for t in ("d1", "d2", "barrier")}
    ok = max(d_new.values()) <= TOL["L11"] and max(d_old.values()) <= TOL["L11"]
    return dict(status="pass" if ok else "fail", n_pairs=len(ids), max_abs_vs_F_pairs_cr2=d_new,
                max_abs_unchanged_vs_F_pairs=d_old, info_n_pairs_changed_vs_original=n_ch)


def l11_atoms(ctx):
    _need_cr2(ctx)
    Fn, Fo = _label_frames()

    def one(r):
        D = ctx.TA.get_data(r, "S", CR2)
        res = _l11_compare(D.y, [str(q) for q in D.P.index], Fn, Fo)
        res.update(provenance=D.linfo.get("provenance"), l11_invariant_on_load=D.linfo.get("l11"))

        def api():
            DL = ctx.IN.LadderData(r, "S", CR2, "cr")
            out = _l11_compare(DL.y, [str(q) for q in DL.P.index], Fn, Fo)
            del DL
            return out
        res["inputs_LadderData"] = _api(ctx, api)
        res["status"] = combine([res["status"], res["inputs_LadderData"]["status"]])
        return res
    res = per_rung(ctx, one)
    res.update(tol=TOL["L11"], labels=str(CR2), labels_sha256=sha(CR2))
    return res


def l11_xgb(ctx):
    _need_cr2(ctx)
    Fn, Fo = _label_frames()

    def one(r):
        D = ctx.TX.get_data(r, "S", CR2)
        res = _l11_compare(D.y, [str(q) for q in D.pid], Fn, Fo)
        res.update(provenance=D.linfo.get("provenance"), l11_invariant_on_load=D.linfo.get("l11"))

        def api():
            pf = ctx.IN.pairs_frame(r, CR2, "cr", subset="usable")
            return _l11_compare(pf[[f"dd_{t}" for t in TARGETS]].to_numpy(float), pf.pair_id.astype(str).tolist(), Fn, Fo)
        res["inputs_pairs_frame"] = _api(ctx, api)
        res["status"] = combine([res["status"], res["inputs_pairs_frame"]["status"]])
        return res
    res = per_rung(ctx, one)
    res.update(tol=TOL["L11"], labels=str(CR2), labels_sha256=sha(CR2))
    return res


# ================================================================ smoke
def smoke_atoms(ctx):
    TA, F, Mo, torch = ctx.TA, ctx.F, ctx.Mo, ctx.torch
    before = _units_snapshot()
    UNIT_WRITES.start()                                    # C-25
    ready = [r for r in STAGE1 if ctx.state["rungs"][r]["status"] == "ready"]
    if len(ready) == len(STAGE1):
        rc = TA.cmd_smoke(1, str(ctx.labels))
        p = L.RES / "C_smoke_atoms.json"
        rep = read_json(p)
        cf = {n: dict(ok=c.get("ok"), anti_max=max([g["anti_max"] for g in c.get("gate_M1_tune") or []]
                                                    + [(c.get("gate_M1_final") or {}).get("anti_max", float("inf"))]),
                      self_max=max([g["self_max"] for g in c.get("gate_M1_tune") or []]
                                   + [(c.get("gate_M1_final") or {}).get("self_max", float("inf"))]),
                      seconds=c.get("seconds"), gpu_mem_peak_gb=c.get("gpu_mem_peak_gb"), d_glob=c.get("d_glob"),
                      n_train=c.get("n_train"), n_test=c.get("n_test")) for n, c in (rep.get("configs") or {}).items()}
        ok = bool(rc == 0 and rep.get("ok") is True and len(cf) == len(STAGE1) * len(ARMS) * len(TA.MODELS))
        res = dict(status="pass" if ok else "fail", mode="train_atoms.cmd_smoke(stage 1)", exit=rc, file=str(p),
                   sha256=sha(p), configs=cf, stage_upper_bound_gpu_hours=rep.get("stage_upper_bound_gpu_hours"))
    else:
        root = _rmtree(SCR / "smoke_atoms_partial")
        cf, ok = {}, True
        for r in ready:
            for arm in ARMS:
                D = TA.get_data(r, arm, ctx.labels)
                for m in TA.MODELS:
                    u = dict(rung=r, arm=arm, model=m, fold=0)
                    rec = TA.run_unit(D, u, root=root, n_epochs=2, seeds=F.FINAL_SEEDS[:1],
                                      cells=[0, len(Mo.GRID[m]) - 1], log=None, smoke=True)
                    g_ok = all(t["gate_M1"]["ok"] for t in rec["tune"]) and all(f["gate_M1"]["ok"] for f in rec["final"])
                    ok &= bool(g_ok)
                    cf[TA.uname(u)] = dict(ok=bool(g_ok), anti_max=max([t["gate_M1"]["anti_max"] for t in rec["tune"]]
                                                                       + [f["gate_M1"]["anti_max"] for f in rec["final"]]))
                    _cuda_free(torch)
        waits = {r: ctx.state["rungs"][r]["why"] for r in STAGE1 if r not in ready}
        res = dict(status="fail" if not ok else "waiting", mode="partial (same run_unit settings as cmd_smoke) on the "
                   "ready rungs", ready=ready, waiting_for=waits, configs=cf)
    after = _units_snapshot()
    return _smoke_units_verdict(res, before, after)        # C-25


def smoke_xgb(ctx):
    TX = ctx.TX
    before = _units_snapshot()
    UNIT_WRITES.start()                                    # C-25
    ready = [r for r in STAGE1 if ctx.state["rungs"][r]["status"] == "ready"]
    if len(ready) == len(STAGE1):
        rc = TX.cmd_smoke(1, str(ctx.labels))
        p = L.RES / "C_smoke_xgb.json"
        rep = read_json(p)
        cf = {n: dict(ok=c.get("ok"), seconds=c.get("seconds"), cpu_seconds=c.get("cpu_seconds"),
                      anti_max=max([v.get("anti_max", float("inf")) for v in (c.get("gate_max") or {}).values()]
                                   + [v.get("anti_max", float("inf")) for v in (c.get("gate_final") or {}).values()]),
                      unit_cpu_hours_full_grid=c.get("unit_cpu_hours_full_grid"))
              for n, c in (rep.get("configs") or {}).items()}
        ok = bool(rc == 0 and rep.get("ok") is True and len(cf) == len(STAGE1) * len(ARMS))
        res = dict(status="pass" if ok else "fail", mode="train_xgb.cmd_smoke(stage 1)", exit=rc, file=str(p),
                   sha256=sha(p), configs=cf, stage_cpu_hours_estimate=rep.get("stage_cpu_hours_estimate"))
    else:
        root = _rmtree(SCR / "smoke_xgb_partial")
        cf, ok = {}, True
        for r in ready:
            for arm in ARMS:
                D = TX.get_data(r, arm, ctx.labels)
                u = dict(rung=r, arm=arm, model="XGB", fold=0)
                rec = TX.run_unit(D, u, root=root, cells=[0, len(TX.GRID) - 1], smoke=True)
                g_ok = bool(rec.get("gates_ok")) and all(rec["final"][t]["gate_M1"]["ok"] for t in TARGETS)
                ok &= g_ok
                cf[TX.uname(u)] = dict(ok=g_ok, cpu_seconds=rec.get("cpu_seconds"))
        waits = {r: ctx.state["rungs"][r]["why"] for r in STAGE1 if r not in ready}
        res = dict(status="fail" if not ok else "waiting", mode="partial (cells first + last, = cmd_smoke) on the ready "
                   "rungs", ready=ready, waiting_for=waits, configs=cf)
    after = _units_snapshot()
    return _smoke_units_verdict(res, before, after)        # C-25


# ================================================================ parts and merge
def new_part(name):
    return dict(part=name, tests_version=TESTS_VERSION, started=now(), host=platform.node(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), python=sys.version.split()[0], prefix=sys.prefix,
                git_head=_git_head(), code_sha256=code_shas(), checks={}, decisions=DECISIONS, tol=TOL)


def run_check(ctx, part, name, fn, *a):
    print(f"[C {ctx.part}] {name} start {now()}", flush=True)
    ctx.check = name
    res = guard(fn, ctx, *a)
    res["pass"] = res["status"] == "pass"
    part["checks"][name] = res
    print(f"[C {ctx.part}] {name}: {res['status'].upper()} ({res.get('seconds')} s)", flush=True)
    write_json(PART_FILE[ctx.part], dict(part, running=True))


def finish(part):
    part["finished"] = now()
    part["status"] = combine([(part["checks"].get(c) or {}).get("status", "pending")
                              for c in CHECKS if part["part"] in PARTS_OF[c]])
    st = part.get("state") or {}
    part["final_inputs"] = bool((st.get("labels") or {}).get("final") and (st.get("p_report") or {}).get("ready"))
    part["running"] = False
    write_json(PART_FILE[part["part"]], part)
    merge()
    print(f"[C {part['part']}] part status {part['status'].upper()} (final inputs {part['final_inputs']})", flush=True)
    if part["status"] == "fail":
        return 3
    return 0 if part["status"] == "pass" and part["final_inputs"] else 2     # review C-5: 0 only on the final inputs


def abort(part, msg):
    """the part could not run (environment / GPU): every check 'error' (exit 2), not a spec 16 C failure (review C-5)."""
    for c in CHECKS:
        if part["part"] in PARTS_OF[c]:
            part["checks"].setdefault(c, {"status": "error", "pass": False, "why": msg})
    part["abort"] = msg
    print(f"ERROR (not a check failure): {msg}", file=sys.stderr, flush=True)
    return finish(part)


def run_gpu(labels_arg):
    part = new_part("gpu")
    try:
        import torch
        import train_atoms as TA
    except Exception as e:                                             # noqa: BLE001
        return abort(part, f"import torch / train_atoms failed (the gpu part needs env_d1fu2): {type(e).__name__}: {e}")
    try:
        TA.FT.require_gpu()
    except SystemExit as e:
        return abort(part, f"fu2 train.require_gpu exit {e.code} (env_d1fu2 on a CUDA node required)")
    ctx = Ctx("gpu", TA=TA, FT=TA.FT, F=TA.F, Mo=TA.Mo, C=TA.C, torch=torch)
    ctx.state = resolve(labels_arg)
    ctx.labels = Path(ctx.state["labels"]["path"])
    part.update(state=ctx.state, versions=TA.FT.versions(), trainer_code_sha256=TA.CODE_SHA, device=str(TA.FT.DEV))
    write_json(PART_FILE["gpu"], dict(part, running=True))
    run_check(ctx, part, "L-4", l4_check, "atoms")
    run_check(ctx, part, "L-3", l3_atoms)
    run_check(ctx, part, "L-6", l6_atoms)
    run_check(ctx, part, "L-11", l11_atoms)
    run_check(ctx, part, "L-5", l5_atoms)
    run_check(ctx, part, "L-10", l10_part, TA.beta_fit, TA.load_labels, "train_atoms.beta_fit")
    run_check(ctx, part, "L-7", l7_atoms)
    run_check(ctx, part, "L-1", l1_atoms)
    run_spies(ctx, spy_atoms)
    part["spy_runs"] = {r: dict(status=v["status"], **({} if v["status"] == "pass" else {"error": v["err"]}))
                        for r, v in ctx.spy.items()}
    run_check(ctx, part, "L-2", l2_atoms)
    run_check(ctx, part, "L-8", l8_atoms)
    run_check(ctx, part, "L-9", l9_atoms)
    ctx.spy = {}
    _cuda_free(torch)
    run_check(ctx, part, "smoke", smoke_atoms)
    return finish(part)


def run_cpu(labels_arg):
    part = new_part("cpu")
    try:
        import train_xgb as TX
    except Exception as e:                                             # noqa: BLE001
        return abort(part, f"import train_xgb failed (the cpu part needs env_espley): {type(e).__name__}: {e}")
    ctx = Ctx("cpu", TX=TX, F=TX.F, C=TX.C)
    ctx.state = resolve(labels_arg)
    ctx.labels = Path(ctx.state["labels"]["path"])
    part.update(state=ctx.state, versions=TX.versions(), trainer_code_sha256=TX.CODE_SHA, n_jobs=TX.N_JOBS)
    write_json(PART_FILE["cpu"], dict(part, running=True))
    run_check(ctx, part, "L-4", l4_check, "xgb")
    run_check(ctx, part, "L-3", l3_xgb)
    run_check(ctx, part, "L-11", l11_xgb)
    run_check(ctx, part, "L-5", l5_xgb)
    run_check(ctx, part, "L-10", l10_part, TX.beta_fit, TX.load_labels, "train_xgb.beta_fit", True)
    run_check(ctx, part, "L-1", l1_xgb)
    run_spies(ctx, spy_xgb)
    part["spy_runs"] = {r: dict(status=v["status"], **({} if v["status"] == "pass" else {"error": v["err"]}))
                        for r, v in ctx.spy.items()}
    run_check(ctx, part, "L-2", l2_xgb)
    run_check(ctx, part, "L-8", l8_xgb)
    run_check(ctx, part, "L-9", l9_xgb)
    ctx.spy = {}
    run_check(ctx, part, "L-6", l6_xgb)
    run_check(ctx, part, "smoke", smoke_xgb)
    return finish(part)


def _l9_cross(pe):
    g, c = (pe.get("gpu") or {}).get("rungs") or {}, (pe.get("cpu") or {}).get("rungs") or {}
    out, bad, n = {}, [], 0
    for r in RUNG_ORDER:
        a, b = (g.get(r) or {}).get("val_sha"), (c.get(r) or {}).get("val_sha")
        if not a or not b:
            continue
        eq = {k: a.get(k) == b.get(k) for k in sorted(set(a) | set(b))}
        out[r] = eq
        n += len(eq)
        bad += [f"{r} f{k}" for k, v in eq.items() if not v]
    return dict(status="fail" if bad else ("pass" if n else "pending"), folds=out, mismatches=bad,
                rule="atom-model validation pairs (FU2 split inner_val, P_<rung> filter) = XGB inner GroupKFold fold 0 "
                     "pairs: list sha256 per rung x fold")


def _consistency(parts):
    g, c = parts.get("gpu") or {}, parts.get("cpu") or {}
    if not g or not c:
        return dict(ok=False, why="a part has not run")
    gs, cs = g.get("state") or {}, c.get("state") or {}
    lab = (gs.get("labels") or {}).get("sha256") == (cs.get("labels") or {}).get("sha256")
    tabs = {r: ((gs.get("rungs") or {}).get(r) or {}).get("table_sha256") == ((cs.get("rungs") or {}).get(r) or {}).get("table_sha256")
            for r in RUNG_ORDER}
    code = g.get("code_sha256") == c.get("code_sha256")
    pr = (gs.get("p_report") or {}).get("sha256") == (cs.get("p_report") or {}).get("sha256")
    ok = bool(lab and all(tabs.values()) and code and pr)
    return dict(ok=ok, same_labels=lab, same_tables=tabs, same_code=code, same_p_report=pr,
                why="" if ok else "the two parts ran on different inputs or code: rerun the older part")


def _current(parts):
    """review C-4: each part's recorded code / labels / tables / P_report / ready rungs still equal the files now."""
    code_now, cur = code_shas(), resolve()
    out, why = {}, []
    for p, d in parts.items():
        if not d:
            continue
        st = d.get("state") or {}
        lab = st.get("labels") or {}
        rungs = st.get("rungs") or {}
        c = dict(code=d.get("code_sha256") == code_now,
                 labels=bool(lab.get("path")) and sha(lab["path"]) == lab.get("sha256"),
                 tables=all(sha(v.get("table")) == v.get("table_sha256") for v in rungs.values()
                            if v.get("status") == "ready"),
                 p_report=sha(L.RES / "P_report.json") == (st.get("p_report") or {}).get("sha256"),
                 ready_rungs={r for r, v in rungs.items() if v.get("status") == "ready"}
                 == {r for r, v in cur["rungs"].items() if v.get("status") == "ready"},
                 v_report=sha(L.RES / "V_report.json") == lab.get("v_report_sha256"))
        out[p] = c
        why += [f"part {p}: {k} changed since it ran (rerun the part)" for k, v in c.items() if not v]
    return dict(ok=not why, parts=out, why=why)


def merge(verbose=False):
    parts = {p: read_json(f) for p, f in PART_FILE.items()}
    checks = {}
    for c in CHECKS:
        pe = {}
        for p in PARTS_OF[c]:
            r = (parts[p].get("checks") or {}).get(c) if parts[p] else None
            pe[p] = r if r is not None else dict(status="pending", why=f"the {p} part has not run this check "
                                                                       f"(results/C_tests_{p}.json)")
        checks[c] = dict(status=combine(v["status"] for v in pe.values()), parts=pe)
    cross = _l9_cross(checks["L-9"]["parts"])
    checks["L-9"]["cross_part"] = cross
    checks["L-9"]["status"] = combine([checks["L-9"]["status"], cross["status"]])
    for v in checks.values():
        v["pass"] = v["status"] == "pass"
    cons = _consistency(parts)
    st = [(parts[p].get("state") or {}) for p in PART_FILE if parts[p]]
    labels_final = bool(len(st) == len(PART_FILE) and all((s.get("labels") or {}).get("final") for s in st))
    reasons = [f"part {p} has not run" for p in PART_FILE if not parts[p]]
    s0 = st[0] if st else {}
    if st and not labels_final:
        reasons.append(f"labels {(s0.get('labels') or {}).get('path')} are provisional: final = F_pairs_cr2.csv after "
                       f"Phase V (results/V_report.json all_pass)")
    for r, v in (s0.get("rungs") or {}).items():
        if v.get("status") == "waiting":
            reasons.append(f"{r}: {v.get('why')}")
        elif v.get("status") == "skipped":
            reasons.append(f"{r} skipped: {v.get('why')}")
    if st and not (s0.get("p_report") or {}).get("ready"):
        reasons.append(P_WAIT)
    if st and len(st) == len(PART_FILE) and not cons["ok"]:
        reasons.append(cons["why"])
    cur = _current(parts)
    reasons += cur["why"]
    status = combine([v["status"] for v in checks.values()] + ["pass" if cons["ok"] else "pending"]
                     + ["pass" if labels_final else "waiting"] + ["pass" if cur["ok"] else "pending"])
    all_pass = status == "pass"
    # review C-1: a skipped g2 does not count against all_pass (stage 1), so say which stage was verified; stage 2
    # training (train_atoms / train_xgb run --stage 2) requires g2_verified
    g2_ready = bool(len(st) == len(PART_FILE)
                    and all(((s.get("rungs") or {}).get("g2") or {}).get("status") == "ready" for s in st))
    est = read_json(L.RES / "M1_gpu_estimate.json")
    out = dict(all_pass=all_pass, status=status, stage_verified=(2 if g2_ready else 1) if all_pass else None,
               g2_verified=bool(all_pass and g2_ready),
               all_pass_scope=("rungs dft, g1, g2" if g2_ready else "stage 1: rungs dft, g1 (g2 not tested)"),
               failed=[c for c, v in checks.items() if v["status"] == "fail"],
               errors=[c for c, v in checks.items() if v["status"] == "error"],
               not_final=[c for c, v in checks.items() if v["status"] in ("waiting", "pending")], waiting_reasons=reasons,
               labels=s0.get("labels"), labels_final=labels_final, rungs=s0.get("rungs"), p_report=s0.get("p_report"),
               consistency=cons, current=cur, checks=checks,
               parts={p: dict(file=str(PART_FILE[p]), sha256=sha(PART_FILE[p]), **{k: d.get(k) for k in (
                   "status", "started", "finished", "running", "host", "slurm_job", "git_head", "python", "prefix",
                   "abort", "spy_runs", "versions", "device", "n_jobs")}) if d else None for p, d in parts.items()},
               gpu_part_checks=[c for c in CHECKS if "gpu" in PARTS_OF[c]], cpu_part_checks=[c for c in CHECKS if "cpu" in PARTS_OF[c]],
               checks_needing_the_gpu=["L-1", "L-2", "L-4", "L-6", "L-8", "L-9", "smoke"],
               gpu_estimate=dict(file=str(L.RES / "M1_gpu_estimate.json"), stage1_gpu_hours=est.get("stage1_gpu_hours"),
                                 per_model_unit_hours={m: (v or {}).get("mean_h") for m, v in
                                                       (est.get("per_model_unit_hours") or {}).items()}) if est else None,
               decisions=DECISIONS, tol=TOL, tests_version=TESTS_VERSION, code_sha256=code_shas(), merged=now())
    write_json(OUT, out)
    print(f"[C merge] status {status.upper()}: " + ", ".join(f"{c} {v['status']}" for c, v in checks.items()), flush=True)
    if verbose:
        for x in reasons:
            print(f"  waiting: {x}", flush=True)
    return 3 if out["failed"] else (0 if out["all_pass"] else 2)


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    cmd = a[0]
    if cmd == "l4child":
        if len(a) != 5:
            print("usage: run_tests.py l4child <atoms|xgb> <rung> <labels> <out.json>", file=sys.stderr)
            return 2
        return l4_child(a[1], a[2], a[3], Path(a[4]))
    if cmd == "merge":
        return merge(verbose=True)
    if cmd in ("gpu", "cpu"):
        labels = None
        if "--labels" in a:
            i = a.index("--labels")
            if i + 1 >= len(a):
                print("--labels needs a path", file=sys.stderr)
                return 2
            labels = a[i + 1]
        return run_gpu(labels) if cmd == "gpu" else run_cpu(labels)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
