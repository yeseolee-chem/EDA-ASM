#!/usr/bin/env python3
r"""prereg.py — D1-REF·LADDER Phase A (docs/specs/D1LADDER.md 9): the pre-registration addendum ADD3 for stage 1
(rungs dft, g1), generated from result files only -> results/PREREG_D1PAIR_ADD3.md (Korean, the ADD1 / ADD2 style) and
results/prereg_d1pair_add3.json. sbatch only (jobs/lad.slurm, LADENV=reactot). Nothing in this file was run when it was
written (login node, no python allowed): every path is untested until it runs inside a job.

CLI (submit from analysis/d1_ladder; exit 0 = done, 2 = not ready (an input phase or the Phase C commit is missing;
nothing written), 3 = refused (a STOP condition, or the result of `check` failed; nothing written))
  python prereg.py write   preconditions below -> md (written first) + json (written last: it holds md_sha256). If both
                           files are already tracked by git and unchanged since HEAD, nothing is rewritten (a committed
                           pre-registration is frozen, spec 9 / 15): `write` then runs `check` and returns its exit code.
                           A tracked ADD3 file that was modified (or one of the two missing) -> exit 3, nothing written.
  python prereg.py check [--stage 1|2]
                           the train_atoms.py / train_xgb.py gate_add3 conditions for a stage-<stage> process, mirrored
                           (the trainers cannot be imported in reactot: torch / xgboost): both files tracked and unchanged
                           since HEAD, sha256(md) = json md_sha256, json inputs_sha256 lists the 'need' paths of the stage
                           (stage 1: label file + dft / g1 tables; stage 2: label file + g2 table), every listed file that
                           is not of another rung's column only (MA-15) still has its sha256. --stage 2 also: json
                           'supplement' present, results/PREREG_D1PAIR_ADD3_G2.md committed, unchanged, sha256 = the
                           recorded one, and the trainers' MA-16 (P_report) / MA-17 (C_tests g2_verified) conditions.
                           Also prints whether HEAD is contained in origin/d1-ladder (information only).
  python prereg.py supplement
                           stage 2 (ADD3 md section 10, PR-5 / PR-10): preconditions below -> the new md
                           results/PREREG_D1PAIR_ADD3_G2.md (Korean, written first) and the committed json extended by the
                           8 listed inputs_sha256 / inputs_roles entries + a new top-level key 'supplement' (written last,
                           atomically; every existing entry and key byte-identical, verified). Nothing is committed: the
                           exact git commands are printed and must be run at once (window: until that commit the json differs
                           from HEAD, so every train_atoms.py / train_xgb.py run and evaluate.py e1 that STARTS exits 3 at its
                           ADD3 gate; running processes are not affected; a task that hit the gate wrote nothing; refused
                           while a lad_m1 / lad_xgb / lad_e1 task is pending, PS-11). If the self-check after the json write
                           fails or raises, the committed json bytes are written back and the new md removed (exit 3; the
                           files are then = HEAD again). A committed supplement is frozen (supplement = check --stage 2); an
                           uncommitted one from an earlier run is regenerated from the committed json.

Preconditions of `write` (all must hold; exit 2 = not ready, 3 = STOP; the reasons are printed, no file is written)
  V   results/V_report.json: all_pass is True, stop falsy; outputs contains str($LAD/F_pairs_cr2.csv) and
      str($LAD/refs_cr2.parquet); every outputs path exists with its recorded sha256. Before V collect (V_report missing
      or not all_pass) the absent F_pairs_cr2.csv / refs_cr2.parquet are 'not ready' (exit 2), not a STOP.
  P   results/P_report.json: running falsy, all_pass is True, stop falsy; fu2_check.ok, dft_vs_g1.ok;
      rungs.dft / rungs.g1 table_sha256 = sha256 of $LAD/inputs/<rung>.parquet now, and results/P_build_<rung>.json
      all_pass with outputs[table] = that sha256; sets.P_dft / P_g1 / P_01: file present, sha256 = the file now;
      labels[str($LAD/F_pairs_cr2.csv)]: ids_ok and sha256 = the file now.
  C   results/C_tests.json (stage 1): failed == [] (else STOP); all_pass is True, status == "pass", stage_verified in
      (1, 2), labels_final is True, errors == [], not_final == [], checks[L-1 .. L-11, smoke].status == "pass",
      current.ok and consistency.ok; labels.path == str($LAD/F_pairs_cr2.csv) and labels.sha256 = the file now;
      rungs.dft / rungs.g1 status "ready" with table_sha256 = the table now; p_report.ready and p_report.sha256 =
      results/P_report.json now; code_sha256 = the 5 files of tests/run_tests.py CODE_FILES now (else rerun Phase C).
      g2 may be skipped (all_pass_scope "stage 1"); g2_verified is not required for stage 1.
  also  Q_report.json all_pass; start_report.json stop false; no L__* file in $LAD/units (spec 16 A: no training before
      the ADD3 commit); the trainers' gate literals present in train_atoms.py and train_xgb.py (file names, json keys,
      'need' rule, default label / input paths); the FU2 atom grid parsed from fu2/models.py = ADD2 json models.grid and
      the XGB grid parsed from train_ml_single.GRIDS["XGB"] = rev 9 prereg_d1pair.json grids.XGB; the MA-6n note found
      in the train_atoms.py docstring; every gated input present; the gated code (CODE_ATOMS / CODE_XGB), the 5 Phase C
      code files (C_CODE = tests/run_tests.py CODE_FILES) and results/C_tests.json tracked by git and unchanged since
      HEAD (spec 8: Phase C is committed before Phase A; the recorded code_commit = that HEAD). Other d1_ladder files
      (evaluate.py receives the E1 code in parallel, refs_*, g0f0, jobs) are not required clean; their sha256 and git
      state are recorded. prereg.py itself is committed together with the ADD3 files. LINE pin (PR-13): the source of
      evaluate.py line_beta + line_predict (the functions L-10 tests) must equal their HEAD version (else exit 2).

Gate contract (train_atoms.py gate_add3 lines 796-832 = train_xgb.py gate_add3 lines 851-886; cmd_run builds
  need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}): files results/PREREG_D1PAIR_ADD3.md and
  results/prereg_d1pair_add3.json; json keys read: md_sha256 (= sha256 of the md) and inputs_sha256 {absolute
  unresolved path (os.path.abspath): sha256}; need for stage 1 = /gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv,
  .../inputs/dft.parquet, .../inputs/g1.parquet. Every other json key is a record (not read by the gate).

Decisions (user 2026-10-22: every judgment = the recommended option; also in the json 'decisions')
  PR-1  env reactot (pandas / pyarrow); the trainers are not imported (train_atoms needs torch + fu2 train, train_xgb
        xgboost): their gate literals are checked in their source text and their fingerprint code_sha256 naming is
        mirrored (CODE_ATOMS / CODE_XGB below).
  PR-2  gated (inputs_sha256) = what a stage-1 result depends on and what stays fixed through stage 2: the label file,
        the two rung tables, P_dft / P_g1 / P_01, the original F_pairs.csv (L-11 check in every load), the FU2 T1 split
        files + S_splits.json + S_folds.csv, the token reports and X182 file of dft and g1, and the code files in the
        trainers' fingerprint code_sha256. NOT gated, recorded under not_gated_sha256: P_report.json and C_tests*.json
        (rewritten by stage-2 'pairs' / tests: gating them would block the g2 gate without any stage-1 input changing),
        V / Q / start / F0 / G0 reports, inputs.py, tests, evaluate.py (E1 code, written separately), refs_*, g0f0,
        prereg.py, job wrappers, fu2/atoms.py (imported for a constant only, in no fingerprint; D1-G2 merges may touch it).
  PR-3  refusal conditions = the preconditions above (exact keys).
  PR-4  the Phase C state must be committed first (spec 8); code_commit = HEAD at write time.
  PR-5  a committed ADD3 is never rewritten by `write`; write on a committed ADD3 = check. The md is frozen for good; the
        only later edit of the json is the stage-2 supplement of md section 10 (new inputs_sha256 / inputs_roles entries
        for the g2 files + a new 'supplement' key, every existing entry byte-identical; command `supplement`, below).
  PR-6  grids read from the code by ast (fu2/models.py GRID evaluated with only `dict`; train_ml_single.GRIDS["XGB"] as a
        literal, cells = d1_train.grid_cells rule: sorted keys, itertools.product) and cross-checked with the ADD2 json /
        rev 9 prereg json; a mismatch = refused. Training constants from fu2_common / d1pair_common (importable here).
  PR-7  FU2 Phase E outputs: the first of (analysis/d1_pair/results/fu2 of this worktree, the d1-pair worktree) whose
        E_check.json is ok with E_metrics.csv present AND whose E_check inputs_sha256 of .../results/fu2/
        M_predictions.parquet = this branch's M_report.json predictions_sha256; absent -> only M_report.json (recorded).
        The git state of the E_* files in their worktree is recorded and printed ('commit 전' while untracked).
  PR-8  the reviewer's post-hoc numbers (oi para 0.65 / 0.26) are constants quoted from spec 9 (no file in the repo).
  PR-9  changed counts recomputed from the files: reactions / molecules = refs_cr2.parquet rows with src != 'own';
        pairs = F_pairs.csv vs F_pairs_cr2.csv (|dd difference| > 1e-9, round_trip parse).
  PR-10 stage 2 (spec 9 / 12 'ADD3 보완판'): the unit fingerprints carry sha256(PREREG_D1PAIR_ADD3.md) and collect
        compares it, so the md is never edited; the supplement is a new md, and the json gets only the new
        inputs_sha256 / inputs_roles entries listed in md section 10 (stage2_supplement_paths) + a 'supplement' key.
  PR-11 expected pair-list sha256 per rung (trainer rule: sorted P_<rung> pair ids in the T1 train / test lists,
        d1pair_common.list_sha) recorded for the Phase E fingerprint check.
  PR-12 md first, json last (atomic each); a self-check re-reads both after writing.
  PR-13 evaluate.py (spec 8 Phase C code) is not required committed (E1 code written in parallel) and C_tests.json does
        not pin it; instead the json 'line_functions' records sha256 of the source of line_beta + line_predict (ast
        segments joined by one newline = evaluate.line_source_sha256), which must equal the HEAD version; E1 STOPs when
        its own LINE source differs (md section 9 lists this as a deviation from spec 8).
  PR-14 spec 13-2 'stages 0 and 1 (rev 9) descriptive': rev 9 stage 0 is a Holm test against arm A (d1_eval.stages), which
        LADDER does not have (FU2 E-4 reports stages 1 and 2 only): pre-registered as not computable; the descriptive
        substitute is 'beats null' (REP P_main MAE < null P_main MAE, point estimate); stage 1 = R2_fam median >= 0.5.
  PR-15 13-2 bend rates population pre-registered: d1_eval.bend_rates with no family filter (all para families of the
        rung's own-set test pairs that FU2 bends_tab keeps), truth = hammett_eval.calls on the F_pairs_cr2.csv dd; a NaN
        rate (zero DFT denominator) = criterion not met; hidden_V / hidden_bend rates descriptive only.
  PR-16 md text: result numbers only from files; rule values are quotes of the spec; reviewer numbers are quotes of
        spec 0-1 / 9 (constants below, source named in place).

Stage 2: `supplement` (spec 12; ADD3 md section 10). Written 2026-10-23 on the login node, never run here: untested until
  it runs inside a job (sbatch -p cpu1,cpu2 -J lad_A2 --time=48:00:00 --cpus-per-task=2 --mem=16G
  --export=ALL,LADENV=reactot jobs/lad.slurm prereg.py supplement, from analysis/d1_ladder).
  Preconditions (exit 2 = not ready, 3 = refused; nothing written in either case):
    ADD3   PREREG_D1PAIR_ADD3.md tracked, = HEAD, sha256 = ADD3_MD_SHA256 (3 otherwise); the HEAD json = the stage-1 json
           (sha256 ADD3_JSON_SHA256, no 'supplement'); json on disk = HEAD, or = an uncommitted earlier supplement of it
           (regenerated); its stage2_plan.supplement_entries = stage2_supplement_paths() (paths, roles, order) and none
           of them listed yet; the trainers' gate literals (GATE_LITERALS + STAGE2_LITERALS) present.
    build  results/P_build_g2.json all_pass and current (inputs._build_ok: fingerprint recomputed = recorded, table
           sha256 = outputs) with g2_q_keys == 'own' in the report, its fingerprint and references.g2_mode (P20 option
           (c), inputs.py build g2 --g2-cons-own), references.n_constrained_replaced == 0; while the D1-G2 F_report has
           stop true: fingerprint accept_f0_stop true and g2_gates.F_stop_accepted.decision_sha256 = F_DECISION.md now
           (inputs.py build g2 --accept-g2f-stop, commit 11aeef90; both flags: jobs/p_stage2.sh, cdbaa3ec).
    queue  (PS-11) no pending (PD / CF) task named lad_m1 / lad_xgb / lad_e1 of this user (squeue on the compute node,
           checked again right before the md write): such a task could start inside the commit window and exit 3 at its
           ADD3 gate (exit 2 while one is pending; a failed squeue only warns and is recorded).
    same TS the D1-G2 C_summary.json same_ts_downstream.column (C-13) = SAME_TS_COLS[0] (PS-7; 3 otherwise).
    P      results/P_report.json not running, all_pass, stop falsy, rungs_built has g2; rungs dft / g1 / g2 table_sha256
           = the tables now; rungs.g2.build_report_sha256 = P_build_g2.json now; sets P_dft, P_g1, P_01, P_g2, P_012 at
           inputs.set_path with sha256 = the files now; P_dft / P_g1 / P_01 sha256 = the ADD3 json entries; labels[
           F_pairs_cr2.csv] ids_ok and sha256 now; fu2_check.ok, dft_vs_g1.ok.
    C      results/C_tests.json failed == [] (else 3); all_pass, status pass, stage_verified == 2, g2_verified,
           all_pass_scope "rungs dft, g1, g2", labels_final, errors == [], not_final == [], every check pass, current.ok,
           consistency.ok (tests/run_tests.py merge keys); labels = F_pairs_cr2.csv now; rungs dft / g1 / g2 ready with
           the tables now; p_report ready with P_report.json now; code_sha256 = C_CODE now; every per-rung check result
           that has a g2 entry: g2 status pass; C_tests.json and the C_CODE files tracked and = HEAD (spec 8 / PR-4).
    D1-G2  = inputs._g2_ready (F_report / H_report present and complete, H_report stop false, its output of
           labels_d1_hf_g2.parquet matched by the repo-relative path with that file's sha256, the D1-G2 files committed:
           G2_COMMITTED), except the F_report stop, accepted only when F_DECISION.md is committed and quotes the
           F_report gate_failure numbers (n, n_g2_ok, rate %) and every stop reason is the F gate rate (PS-6);
           origin/d1-g2 (as last fetched) contained in HEAD (spec 12 'origin/d1-g2를 merge한다'; 2 otherwise).
    units  no L__g2__* entry in $LAD/units (spec 16 A: no stage-2 training before the supplement commit).
    gate   the 8 entries exist (else 2) with the columns 'g2' (P_012: 'dft+g1+g2'); the extended json passes the
           gate_add3 content conditions for a stage-1 process (rungs dft, g1) and a g2 process (need = label file + g2
           table) on the files now (3 otherwise).
  Decisions (user 2026-10-22 / 23: every judgment = the recommended option; json supplement.decisions = SUPP_DECISIONS)
  PS-1 .. PS-12 below.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees (d1_pair, fu2, espley)

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402
import inputs as IN  # noqa: E402            (paths, RUNGS, check_paths; numpy / pandas only at import)

DP = L.DP
FU2_DIR = L.P_DIR / "fu2"
sys.path.insert(0, str(FU2_DIR))
import fu2_common as F  # noqa: E402         (FU2 training constants; d1pair_common only, loads in every env)

CODE_VERSION = "d1-ladder-a1"
ADD3_MD = L.RES / "PREREG_D1PAIR_ADD3.md"                 # = train_atoms.ADD3_MD = train_xgb.ADD3_MD
ADD3_JSON = L.RES / "prereg_d1pair_add3.json"             # = train_atoms.ADD3_JSON = train_xgb.ADD3_JSON
STAGE = 1
STAGE_RUNGS = ("dft", "g1")
ARMS = ("S", "SH")
ATOM_MODELS = ("AW", "GNN", "TF")
MODELS4 = ("AW", "GNN", "TF", "XGB")
NF = 5
TARGETS = list(DP.TARGETS)                                # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
H9 = list(DP.PRIMARY_TARGETS)                             # the 9 HF values / primary targets
CHANGED = ("d1", "d2", "barrier")
CHECKS = [f"L-{i}" for i in range(1, 12)] + ["smoke"]
TOL_PAIR = 1e-9

CR2 = L.LAD / "F_pairs_cr2.csv"
ORIG = DP.RES / "F_pairs.csv"
REFS_CR2 = L.LAD / "refs_cr2.parquet"
V_REPORT = L.RES / "V_report.json"
Q_REPORT = L.RES / "Q_report.json"
P_REPORT = L.RES / "P_report.json"
C_TESTS = L.RES / "C_tests.json"
START = L.RES / "start_report.json"
F0_REPORT = L.RES / "F0_report.json"
F0_DECISION = L.RES / "F0_DECISION.md"
G0_REPORT = L.RES / "G0_report.json"
V_TREP = L.RES / "V_targets_report.json"
UNITS = L.LAD / "units"
SPEC = L.REPO / "docs" / "specs" / "D1LADDER.md"
A_SUMMARY = L.HF_RES / "A_summary.csv"
A_REPORT = L.HF_RES / "A_report.json"
FU2_RES = DP.RES / "fu2"                                  # FU2 results of this branch (= 2d745d23)
FU2_E_DIRS = (FU2_RES, Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/analysis/d1_pair/results/fu2"))
FU2_EVAL = (FU2_DIR / "evaluate.py",
            Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/analysis/d1_pair/fu2/evaluate.py"))
ADD2_JSON = DP.RES / "prereg_d1pair_add2.json"
REV9_JSON = DP.RES / "prereg_d1pair.json"
PREV = {"PREREG_D1PAIR.md": DP.RES / "PREREG_D1PAIR.md", "PREREG_D1PAIR_ADD1.md": DP.RES / "PREREG_D1PAIR_ADD1.md",
        "PREREG_D1PAIR_ADD2.md": DP.RES / "PREREG_D1PAIR_ADD2.md"}

# the trainers' fingerprint code_sha256 (train_atoms.CODE_SHA / train_xgb.CODE_SHA), same keys
CODE_ATOMS = {"train_atoms.py": HERE / "train_atoms.py", "lad_common.py": HERE / "lad_common.py",
              "fu2/models.py": FU2_DIR / "models.py", "fu2/train.py": FU2_DIR / "train.py",
              "fu2/fu2_common.py": FU2_DIR / "fu2_common.py", "fu2/tokens.py": FU2_DIR / "tokens.py",
              "d1pair_common.py": L.P_DIR / "d1pair_common.py"}
CODE_XGB = {"train_xgb.py": HERE / "train_xgb.py", "lad_common.py": HERE / "lad_common.py",
            "d1_train.py": L.P_DIR / "d1_train.py", "d1pair_common.py": L.P_DIR / "d1pair_common.py",
            "train_ml_single.py": DP.ESP_DIR / "train_ml_single.py", "fu2/fu2_common.py": FU2_DIR / "fu2_common.py",
            "fu2/tokens.py": FU2_DIR / "tokens.py"}
# tests/run_tests.py CODE_FILES (C_tests.json code_sha256)
C_CODE = {"tests/run_tests.py": HERE / "tests" / "run_tests.py", "inputs.py": HERE / "inputs.py",
          "train_atoms.py": HERE / "train_atoms.py", "train_xgb.py": HERE / "train_xgb.py",
          "lad_common.py": HERE / "lad_common.py"}
LADDER_CODE = {n: HERE / n for n in ("lad_common.py", "inputs.py", "train_atoms.py", "train_xgb.py", "evaluate.py",
                                     "refs_q.py", "refs_v.py", "g0f0.py", "tests/run_tests.py", "jobs/gpu.slurm",
                                     "jobs/lad.slurm")}
AUX_CODE = {"fu2/atoms.py": FU2_DIR / "atoms.py", "fu2/splits.py": FU2_DIR / "splits.py",
            "d1_eval.py": L.P_DIR / "d1_eval.py", "hammett_eval.py": L.P_DIR / "hammett_eval.py",
            "d1_pair/jobs/py.slurm": L.P_DIR / "jobs" / "py.slurm", "d1_hf/build.py": L.HF_DIR / "build.py",
            "d1_hf/hf_common.py": L.HF_DIR / "hf_common.py"}
# literals that the trainers' ADD3 gate depends on (both files)
GATE_LITERALS = ('ADD3_MD = L.RES / "PREREG_D1PAIR_ADD3.md"', 'ADD3_JSON = L.RES / "prereg_d1pair_add3.json"',
                 'DEFAULT_LABELS = L.LAD / "F_pairs_cr2.csv"', 'INPUTS = L.LAD / "inputs"',
                 'return INPUTS / f"{rung}.parquet"',
                 'need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}',
                 'if L.sha256(ADD3_MD) != rec.get("md_sha256"):', 'ins = rec.get("inputs_sha256") or {}',
                 'git("ls-files", "--error-unmatch", str(p))', 'git("diff", "--quiet", "HEAD", "--", str(p))',
                 'return Path(os.path.abspath(p))', 'lack = sorted(set(need) - set(ins))',
                 # the skip rule that cmd_check mirrors (MA-15) and the per-file verification
                 'if path not in need and _foreign_to(path, rungs):',
                 'return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)',
                 'if not Path(path).is_file() or L.sha256(path) != s:')
LINE_FUNCS = ("line_beta", "line_predict")                # evaluate.py LINE functions (L-10; = evaluate.LINE_FUNCS)
# spec 0-1 (the reviewer's HF pre-check): the parts that no result file holds, quoted (PR-16)
REVIEWER_PRECHECK_QUOTE = dict(
    source="docs/specs/D1LADDER.md 0-1 (reviewer-supplied numbers; the counts below are not in Q_report.json)",
    flagged_share_pct=7.1, cyclooctyne_D=135, cyclooctyne_Dp=122,
    caveat_ko="dipole 221건 중 일부는 Coley가 일부러 고른 입체 호환 형태일 수 있다")
REVIEWER_POSTHOC = dict(
    source="docs/specs/D1LADDER.md 9 (reviewer-supplied numbers; no computation file in the repository)",
    note="post-hoc exploration after seeing D1-HF Phase A and FU2 results (original labels); not pre-registered",
    items=[dict(what="HF@G1 + FU2 linear combination", target="oi", subset="para", mae_kcal=0.65),
           dict(what="HF@DFT + FU2 linear combination", target="oi", subset="para", mae_kcal=0.26,
                reading="same target / subset as the first item (spec text lists it right after)")])
STAGE2_RULE = dict(targets=["oi", "cpcm"], para_mae_max=0.13, bend_sensitivity_min=0.7, false_positive_rate_max=0.1)
DECISIONS = {
    "PR-1": "env reactot; trainers not imported: gate literals checked in their source, code_sha256 naming mirrored",
    "PR-2": "gated = label file, 2 rung tables, P_dft/P_g1/P_01, F_pairs.csv, FU2 T1 split files + S_splits + S_folds, "
            "token reports + X182 of dft/g1, trainer fingerprint code; P_report / C_tests / reports / other code recorded "
            "only (they change in stage 2 or after ADD3)",
    "PR-3": "refusal = V_report / P_report / C_tests stage-1 keys + consistency with the files now (docstring)",
    "PR-4": "Phase C committed first (spec 8); code_commit = HEAD at write",
    "PR-5": "a committed ADD3 is frozen; write on it = check; the md never changes, the json only receives the stage-2 "
            "supplement entries of md section 10 (existing entries unchanged)",
    "PR-6": "grids parsed from fu2/models.py and train_ml_single.py (ast), cross-checked with ADD2 / rev 9 prereg json",
    "PR-7": "FU2 Phase E outputs from the first dir with E_check ok and E_check M_predictions sha256 = this branch's "
            "M_report predictions_sha256 (this worktree, then the d1-pair worktree); E_* git state recorded",
    "PR-8": "reviewer post-hoc numbers quoted from spec 9 (no file)",
    "PR-9": "changed reactions / molecules from refs_cr2.parquet (src != own); pairs from F_pairs.csv vs F_pairs_cr2.csv",
    "PR-10": "stage 2: this md stays byte-identical (unit fingerprints carry its sha256); supplement = new md + new "
             "inputs_sha256 / inputs_roles entries (md section 10 list) + a new 'supplement' key",
    "PR-11": "expected trainer pair-list sha256 per rung recorded",
    "PR-12": "md first, json last; self-check after writing",
    "PR-13": "evaluate.py not required committed and not pinned by C_tests.json: json line_functions = sha256 of the "
             "line_beta + line_predict source (= HEAD version, else not ready); E1 STOPs on a different LINE source",
    "PR-14": "rev 9 stage 0 (Holm test vs arm A) not computable in LADDER (no arm A; FU2 E-4 precedent): descriptive "
             "substitute beats null (REP P_main MAE < null, point); stage 1 = R2_fam median >= 0.5, descriptive",
    "PR-15": "13-2 bend rates = d1_eval.bend_rates over all para families of the own-set test pairs (FU2 bends_tab), truth "
             "= hammett_eval.calls on F_pairs_cr2.csv dd; NaN rate = not met; hidden_V / hidden_bend descriptive",
    "PR-16": "result numbers from files; rule values quoted from the spec; reviewer numbers quoted from spec 0-1 / 9",
}


class Refuse(Exception):
    def __init__(self, code, reasons):
        super().__init__("; ".join(reasons))
        self.code, self.reasons = int(code), list(reasons)


# ================================================================ small helpers
def _now():
    return time.strftime("%F %T")


def _json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _sha(p):
    return L.sha256(p) if p is not None and Path(p).is_file() else None


def _ab(p):
    return str(Path(os.path.abspath(p)))


def _git_rc(*a):
    return subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True).returncode


def _git_out(*a):
    r = subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _tracked(p):
    return _git_rc("ls-files", "--error-unmatch", str(p)) == 0


def _clean(p):
    return _git_rc("diff", "--quiet", "HEAD", "--", str(p)) == 0


def _committed(p):
    return Path(p).is_file() and _tracked(p) and _clean(p)


def _last_commit(p):
    return _git_out("log", "-1", "--format=%h", "--", str(p))


def _disp(p):
    """$LAD / $D1PAIR_SCRATCH / $G2 / repo-relative display form of a path."""
    s = _ab(p)
    for root, tag in ((L.LAD, "$LAD"), (DP.SCRATCH, "$D1PAIR_SCRATCH"), (L.G2, "$G2"), (L.REPO, "")):
        r = _ab(root)
        if s == r or s.startswith(r + os.sep):
            rest = s[len(r) + 1:]
            return f"{tag}/{rest}" if tag else rest
    return s


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _f(x, d=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(v) else f"{v:.{d}f}"


def _n(x):
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "–"


def _pct(x, d=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(v) else f"{100 * v:.{d}f} %"


def _q(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(n=int(len(a)), p50=float(np.median(a)), p90=float(np.quantile(a, 0.9)), max=float(a.max()),
                mean=float(a.mean()))


def _vc(s):
    return {str(k): int(v) for k, v in pd.Series(s).value_counts().items()}


def _assign(src, name):
    """the value node of the module-level `name = ...` assignment in source text `src` (None if absent)."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def _doc_items(doc, prefixes=("MA", "XG")):
    """numbered decision items of a docstring: lines '  <P>-<n>[x]  text' + continuation lines indented >= 6 spaces."""
    pat = re.compile(r"^  ((?:%s)-\d+[a-z]?)\s+(.*)$" % "|".join(prefixes))
    items, cur = {}, None
    for line in doc.splitlines():
        m = pat.match(line)
        if m:
            cur = m.group(1)
            items[cur] = m.group(2).strip()
            continue
        if cur and re.match(r"^ {6,}\S", line):
            items[cur] += " " + line.strip()
            continue
        cur = None
    return items


def _id_span(keys):
    """'<first>–<last> (n개)' of decision ids ordered by number ('P6b_graphs' -> P6b, 'V10_imag' -> V10, 'MA-6n')."""
    def key(s):
        m = re.match(r"^([A-Za-z]+-?)(\d+)([a-z]*)$", s)
        return (int(m.group(2)), m.group(3)) if m else (10 ** 9, s)
    ids = sorted({str(k).split("_")[0] for k in (keys or [])}, key=key)
    return f"{ids[0]}–{ids[-1]} ({len(ids)}개)" if ids else "–"


def _git_state_at(p):
    """git state of p in its own worktree: 'committed' (tracked, = HEAD), 'modified' or 'untracked'."""
    p = Path(p)
    def rc(*a):
        return subprocess.run(["git", "-C", str(p.parent), *a], capture_output=True, text=True).returncode
    if rc("ls-files", "--error-unmatch", str(p)) != 0:
        return "untracked"
    return "committed" if rc("diff", "--quiet", "HEAD", "--", str(p)) == 0 else "modified"


GIT_KO = {"committed": "commit됨(HEAD와 같음)", "modified": "commit 뒤 수정됨", "untracked": "commit 안 됨",
          "missing": "파일 없음"}


def line_source(text):
    """sha256 of the source of the module-level line_beta and line_predict of an evaluate.py text (ast segments joined by
    one newline, LINE_FUNCS order; = evaluate.line_source_sha256); None if one is missing."""
    segs = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name in LINE_FUNCS}
    if set(segs) != set(LINE_FUNCS):
        return None
    return hashlib.sha256("\n".join(segs[k] for k in LINE_FUNCS).encode()).hexdigest()


def line_pin():
    """PR-13: the LINE source of evaluate.py now and in HEAD (the functions L-10 tests and E1 uses)."""
    ev = HERE / "evaluate.py"
    rel = "analysis/d1_ladder/evaluate.py"
    try:
        now = line_source(ev.read_text(encoding="utf-8")) if ev.is_file() else None
    except SyntaxError as e:
        raise Refuse(3, [f"{ev}: syntax error ({e}); Phase C L-10 imports this file"])
    r = subprocess.run(["git", "-C", str(L.HERE), "show", f"HEAD:{rel}"], capture_output=True)   # bytes: UTF-8 text
    try:
        head = line_source(r.stdout.decode("utf-8")) if r.returncode == 0 and r.stdout else None
    except (SyntaxError, UnicodeDecodeError):
        head = None
    return dict(file=rel, functions=list(LINE_FUNCS), sha256=now, sha256_head=head, equal_head=bool(now and now == head),
                evaluate_sha256=_sha(ev), evaluate_git_state=_git_state_at(ev) if ev.is_file() else "missing",
                rule="sha256 of ast.get_source_segment of the module-level defs, joined by one newline, in the order "
                     "line_beta, line_predict (= evaluate.line_source_sha256); E1 STOPs when its own value differs")


def start_0_3_record():
    """start condition 0-3 from git: the merge commit in HEAD's history whose second parent is the D1-HF commit, and
    whether its first (d1-g2) parent already contained the D1-HF results."""
    hf = L.COMMITS["d1hf"]
    full = _git_out("rev-parse", "--verify", f"{hf}^{{commit}}")
    found = None
    for line in (_git_out("log", "--merges", "--format=%H %P", "HEAD") or "").splitlines():
        h, *par = line.split()
        if full and full in par[1:]:
            found = (h, par[0])
            break
    rec = dict(d1hf_commit=hf, merge_commit=None, first_parent=None, first_parent_had_d1hf=None, merge_subject=None)
    if found:
        rec.update(merge_commit=found[0][:8], first_parent=found[1][:8],
                   first_parent_had_d1hf=_git_rc("merge-base", "--is-ancestor", full, found[1]) == 0,
                   merge_subject=_git_out("log", "-1", "--format=%s", found[0]))
    return rec


def need_paths(rungs=STAGE_RUNGS, labels=CR2):
    """the trainers' cmd_run 'need' set for these rungs (default labels)."""
    return {_ab(labels)} | {_ab(IN.table_path(r)) for r in rungs}


# ================================================================ preconditions
def check_v(b2, b3, ok):
    v = _json(V_REPORT)
    if v is None:
        b2.append("results/V_report.json missing (Phase V collect has not run)")
        return None
    if v.get("stop"):
        b3.append(f"V_report.json stop: {v.get('stop_reason')}")
        return v
    if v.get("all_pass") is not True:
        b2.append(f"V_report.json all_pass {v.get('all_pass')}")
        return v
    outs = v.get("outputs") or {}
    for p in (CR2, REFS_CR2):
        if str(p) not in outs:
            b3.append(f"V_report.json outputs lack {p}")
    bad = [p for p, s in outs.items() if not Path(p).is_file() or L.sha256(p) != s]
    if bad:
        b3.append(f"V_report.json outputs missing or changed since collect: {bad[:5]}")
    if not any(x.startswith("V_report") for x in b3):
        ok.append("V_report all_pass, outputs (F_pairs_cr2.csv, refs_cr2.parquet, ...) unchanged")
    return v


def check_p(b2, b3, ok):
    p = _json(P_REPORT)
    if p is None:
        b2.append("results/P_report.json missing (inputs.py pairs)")
        return None
    if p.get("running"):
        b2.append("results/P_report.json is being written (inputs.py pairs running)")
        return p
    if p.get("stop"):
        b3.append(f"P_report.json stop: {p.get('stop_reasons') or p.get('stop_reason')}")
        return p
    if p.get("all_pass") is not True:
        b2.append(f"P_report.json all_pass {p.get('all_pass')}")
        return p
    n0 = len(b3)
    for r in STAGE_RUNGS:
        tab = IN.table_path(r)
        rr = (p.get("rungs") or {}).get(r) or {}
        now = _sha(tab)
        if rr.get("table_sha256") is None or now != rr.get("table_sha256"):
            b3.append(f"P_report rungs.{r}.table_sha256 {rr.get('table_sha256')} != {tab} now ({now})")
        br = _json(IN.build_report_path(r)) or {}
        if not br.get("all_pass") or (br.get("outputs") or {}).get(str(tab)) != now:
            b3.append(f"{IN.build_report_path(r).name}: all_pass {br.get('all_pass')} or outputs[{tab}] != the table now")
    if not (p.get("fu2_check") or {}).get("ok"):
        b3.append("P_report fu2_check.ok is not true (P_g1 != FU2 pair list, spec 16 P)")
    if not (p.get("dft_vs_g1") or {}).get("ok"):
        b3.append(f"P_report dft_vs_g1.ok is not true ({p.get('dft_vs_g1')}, spec 16 P)")
    for name in ("P_dft", "P_g1", "P_01"):
        s = (p.get("sets") or {}).get(name)
        if not s or not s.get("file") or _sha(s["file"]) != s.get("sha256"):
            b3.append(f"P_report sets.{name} missing or its file changed")
    lab = (p.get("labels") or {}).get(str(CR2)) or {}
    if not lab.get("ids_ok") or lab.get("sha256") != _sha(CR2):
        b3.append(f"P_report labels[{CR2}]: ids_ok {lab.get('ids_ok')}, sha256 {lab.get('sha256')} != the file now")
    if len(b3) == n0:
        ok.append("P_report all_pass, fu2_check ok, dft_vs_g1 ok; tables, sets, labels = the files now")
    return p


def check_c(b2, b3, ok):
    c = _json(C_TESTS)
    if c is None:
        b2.append("results/C_tests.json missing (Phase C)")
        return None
    if c.get("failed"):
        b3.append(f"C_tests.json failed checks {c.get('failed')} (spec 16 C STOP)")
        return c
    keys = dict(all_pass=c.get("all_pass") is True, status_pass=c.get("status") == "pass",
                stage_verified=c.get("stage_verified") in (1, 2), labels_final=c.get("labels_final") is True,
                no_errors=not c.get("errors"), none_not_final=not c.get("not_final"),
                every_check_pass=all((((c.get("checks") or {}).get(k)) or {}).get("status") == "pass" for k in CHECKS),
                current_ok=(c.get("current") or {}).get("ok") is True,
                consistency_ok=(c.get("consistency") or {}).get("ok") is True)
    n0 = len(b2)
    if not all(keys.values()):
        b2.append(f"C_tests.json not passed for stage 1: {[k for k, v in keys.items() if not v]} "
                  f"(status {c.get('status')}, not_final {c.get('not_final')}, errors {c.get('errors')})")
    lab = c.get("labels") or {}
    if lab.get("path") != str(CR2) or lab.get("sha256") != _sha(CR2):
        b2.append(f"C_tests.json labels {lab.get('path')} / {str(lab.get('sha256'))[:12]} != {CR2} now (rerun Phase C)")
    for r in STAGE_RUNGS:
        rr = (c.get("rungs") or {}).get(r) or {}
        if rr.get("status") != "ready" or rr.get("table_sha256") != _sha(IN.table_path(r)):
            b2.append(f"C_tests.json rungs.{r}: status {rr.get('status')}, table_sha256 != the table now (rerun Phase C)")
    pr = c.get("p_report") or {}
    if not pr.get("ready") or pr.get("sha256") != _sha(P_REPORT):
        b2.append("C_tests.json p_report not ready or its sha256 != results/P_report.json now (rerun Phase C)")
    code_now = {k: _sha(p) for k, p in C_CODE.items()}
    diff = sorted(k for k in set(code_now) | set(c.get("code_sha256") or {})
                  if (c.get("code_sha256") or {}).get(k) != code_now.get(k))
    if diff:
        b2.append(f"C_tests.json code_sha256 differs from the code now: {diff} (rerun Phase C)")
    if len(b2) == n0:
        ok.append("C_tests.json all_pass (stage 1), every check pass, labels final = F_pairs_cr2.csv, tables / P_report / "
                  "code = the files now")
    return c


def check_other(b2, b3, ok):
    q = _json(Q_REPORT)
    if not q or q.get("all_pass") is not True or q.get("stop"):
        b3.append("results/Q_report.json missing or not all_pass")
    s = _json(START)
    if not s or s.get("stop"):
        b3.append("results/start_report.json missing or stop")
    for p in (F0_REPORT, F0_DECISION, G0_REPORT, V_TREP, A_SUMMARY, A_REPORT, FU2_RES / "M_report.json", SPEC, ADD2_JSON,
              REV9_JSON, ORIG):
        if not Path(p).is_file():
            b3.append(f"{p} missing")
    # Phase V outputs: absent before V collect = not ready (check_v STOPs when V_report all_pass lists them but they are
    # missing or changed)
    for p in (REFS_CR2, CR2):
        if not Path(p).is_file():
            b2.append(f"{p} missing (Phase V collect has not written it)")
    lp = line_pin()
    if not lp["equal_head"]:
        b2.append(f"evaluate.py line_beta / line_predict source {str(lp['sha256'])[:12]} != the HEAD version "
                  f"{str(lp['sha256_head'])[:12]} (commit the LINE functions and rerun Phase C L-10, PR-13)")
    if UNITS.is_dir():
        names = sorted(e.name for e in os.scandir(UNITS) if e.name.startswith("L__"))
        if names:
            b3.append(f"{len(names)} Phase M unit files in $LAD/units before the ADD3 commit (spec 16 A): {names[:5]}")
    for f in (HERE / "train_atoms.py", HERE / "train_xgb.py"):
        txt = f.read_text(encoding="utf-8")
        miss = [s for s in GATE_LITERALS if s not in txt]
        if miss:
            b3.append(f"{f.name}: ADD3 gate literals not found (the gate changed; regenerate this contract): {miss}")
    # the code that trains (gated) and the code + result of Phase C must be in the recorded commit; other d1_ladder files
    # (evaluate.py gets the E1 code in parallel, refs_*, g0f0, jobs) are recorded with their git state only
    unc = [n for n, p in {**C_CODE, **{f"gated:{k}": v for k, v in {**CODE_ATOMS, **CODE_XGB}.items()},
                          "results/C_tests.json": C_TESTS}.items() if not _committed(p)]
    if unc:
        b2.append(f"not committed or modified since HEAD (commit the Phase C state first, spec 8): {unc}")
    if not b3 and not unc and lp["equal_head"]:
        ok.append("Q / start reports pass; no $LAD/units L__ files; gate literals present in both trainers; gated code, "
                  "Phase C code and C_tests.json committed; evaluate.py LINE source = HEAD")


def preconditions():
    b2, b3, ok = [], [], []
    v = check_v(b2, b3, ok)
    p = check_p(b2, b3, ok)
    c = check_c(b2, b3, ok)
    check_other(b2, b3, ok)
    if b3:
        raise Refuse(3, b3 + b2)
    if b2:
        raise Refuse(2, b2)
    return v, p, c, ok


# ================================================================ content: code-derived design values
def grids():
    """FU2 atom grid (fu2/models.py GRID) and rev 9 XGB grid (train_ml_single.GRIDS["XGB"] -> d1_train.grid_cells)."""
    mp = FU2_DIR / "models.py"
    src = mp.read_text(encoding="utf-8")
    node = _assign(src, "GRID")
    if node is None:
        raise Refuse(3, [f"{mp}: no module-level GRID"])
    G = eval(ast.get_source_segment(src, node), {"__builtins__": {}, "dict": dict})   # noqa: S307 (own source)
    add2 = _json(ADD2_JSON) or {}
    if G != ((add2.get("models") or {}).get("grid")):
        raise Refuse(3, [f"fu2/models.py GRID {G} != ADD2 json models.grid (spec 4-3 'FU2 M-2 as is')"])
    tp = DP.ESP_DIR / "train_ml_single.py"
    s2 = tp.read_text(encoding="utf-8")
    gn = _assign(s2, "GRIDS")
    val = None
    if isinstance(gn, ast.Dict):
        for k, v in zip(gn.keys, gn.values):
            if isinstance(k, ast.Constant) and k.value == "XGB":
                val = v
    if not (isinstance(val, ast.Tuple) and len(val.elts) == 2):
        raise Refuse(3, [f"{tp}: GRIDS['XGB'] = (estimator, grid) not found"])
    est = ast.get_source_segment(s2, val.elts[0])
    raw = ast.literal_eval(val.elts[1])
    gv = {k.split("__")[-1]: list(v) for k, v in raw.items()}
    rev9 = ((_json(REV9_JSON) or {}).get("grids") or {}).get("XGB")
    if gv != rev9:
        raise Refuse(3, [f"train_ml_single GRIDS['XGB'] {gv} != rev 9 prereg_d1pair.json grids.XGB {rev9}"])
    d1t = (L.P_DIR / "d1_train.py").read_text(encoding="utf-8")
    if "ks = sorted(g)" not in d1t or "itertools.product(*[g[k] for k in ks])" not in d1t:
        raise Refuse(3, ["d1_train.grid_cells no longer the sorted-key product (cell order unknown)"])
    ks = sorted(gv)
    cells = [dict(zip(ks, v)) for v in itertools.product(*[gv[k] for k in ks])]
    return dict(atoms=G, atoms_source=f"{_disp(mp)} GRID (= ADD2 json models.grid)",
                xgb_values=gv, xgb_cells=cells, xgb_estimator=est,
                xgb_source=f"{_disp(tp)} GRIDS['XGB'] (prefix removed; = rev 9 prereg grids.XGB), cells = "
                           "d1_train.grid_cells('XGB') (sorted keys, itertools.product)")


def constants():
    return dict(lambda_identity=F.LAMBDA, lr=F.LR, weight_decay=F.WD, dropout=F.DROPOUT, batch_pairs=F.BATCH,
                max_epochs=F.MAX_EPOCHS, patience=F.PATIENCE, tune_seed=F.SEED, final_seeds=list(F.FINAL_SEEDS),
                gate_m1_tol_kcal=F.ANTISYM_TOL, gate_m1_pairs=F.ANTISYM_N, xgb_model_part_tol_kcal=DP.ANTISYM_TOL,
                xgb_gate_pairs=DP.ANTISYM_N, n_inner=DP.N_INNER, bootstrap_seed=DP.SEED,
                source="analysis/d1_pair/fu2/fu2_common.py (atoms), analysis/d1_pair/d1pair_common.py (XGB)")


def trainer_notes():
    ta = (HERE / "train_atoms.py").read_text(encoding="utf-8")
    tx = (HERE / "train_xgb.py").read_text(encoding="utf-8")
    ma = _doc_items(ast.get_docstring(ast.parse(ta), clean=False) or "", ("MA",))
    xg = _doc_items(ast.get_docstring(ast.parse(tx), clean=False) or "", ("XG",))
    if "MA-6n" not in ma:
        raise Refuse(3, ["train_atoms.py docstring: MA-6n note not found (Phase A must record it)"])
    return dict(ma=ma, xg=xg, ma6n=ma["MA-6n"], xg7=xg.get("XG-7"), ma6=ma.get("MA-6"))


def unit_names(rungs=STAGE_RUNGS):
    return [f"L__{r}__{a}__{m}__f{k}" for r in rungs for a in ARMS for m in MODELS4 for k in range(NF)]


def unit_counts(G):
    """units and GPU trainings (one per grid cell for tuning + one per final seed, per atom unit; spec 4-4)."""
    seeds = len(F.FINAL_SEEDS)
    n_cells = {m: len(G[m]) for m in ATOM_MODELS}
    names = unit_names()
    n_atoms = sum(1 for u in names if u.split("__")[3] in ATOM_MODELS)
    per_rung_gpu = len(ARMS) * NF * sum(n_cells[m] + seeds for m in ATOM_MODELS)
    return dict(stage1=len(names), stage1_atoms=n_atoms, stage1_xgb=len(names) - n_atoms,
                gpu_stage1=len(STAGE_RUNGS) * per_rung_gpu, stage2=len(ARMS) * len(MODELS4) * NF, gpu_stage2=per_rung_gpu,
                n_cells=n_cells, n_final_seeds=seeds,
                rule="atom unit = n_cells tuning fits + n_final_seeds final fits; stage 2 = rung g2 x 2 arms")


# ================================================================ content: results
def q_summary(q):
    au, ra, q6 = q.get("audit") or {}, q.get("reassembly") or {}, q.get("Q6") or {}
    keep = ("reactions", "molecules", "molecules_by_kind", "rows_by_kind", "rows_constrained",
            "dipolarophile_rows_cyclooctyne", "by_dipole", "by_skeleton")
    hp = q.get("hf_precheck") or {}
    return dict(
        file=_disp(Q_REPORT), sha256=_sha(Q_REPORT), git_head=q.get("git_head"), code_version=q.get("code_version"),
        gates_ok={k: (v or {}).get("ok") for k, v in (q.get("gates") or {}).items()},
        audit={t: {k: (au.get(t) or {}).get(k) for k in keep} for t in ("0.05", "0.5", "1", "2")},
        dev_of_rows_above_0_05=au.get("dev_of_rows_above_0.05"),
        reassembly={k: ra.get(k) for k in ("n_rows_changed", "by_changed", "n_fragments_changed",
                                           "n_fragments_changed_by_kind", "n_constrained_fragments_changed",
                                           "delta_d1", "delta_d2", "delta_barrier", "n_source_molecules")},
        pairs={t: {k: (q6.get(t) or {}).get(k) for k in ("n_changed", "n_abs_gt", "in_P_main", "by_pos_P_main",
                                                         "abs_change")} for t in CHANGED},
        pairs_n_rows_rewritten=q6.get("n_rows_rewritten"), pairs_output=q6.get("output"),
        pairs_output_sha256=q6.get("output_sha256"),
        azide_D_para={k: (q6.get("azide_D_para") or {}).get(k) for k in ("csv", "n", "n_changed_dd_barrier",
                                                                          "n_changed_dd_d2")},
        keys={k: (q.get("keys") or {}).get(k) for k in ("n_keys", "n_keys_by_kind", "n_keys_constrained",
                                                       "n_rows_constrained", "n_keys_multi_row")},
        g_vs_e={k: (q.get("g_vs_e") or {}).get(k) for k in ("n_keys_compared", "n_argmin_differ",
                                                           "n_argmin_differ_material", "material_by_kind")},
        hf_precheck=dict(source=hp.get("source"), sha256=hp.get("sha256"), reviewer=hp.get("reviewer"),
                         replication_P_all=((hp.get("same_keys") or {}).get("P_all_reactions")),
                         replication_all_targets=((hp.get("same_keys") or {}).get("all_targets"))),
        decisions=q.get("decisions"), outputs=q.get("outputs"))


def v_summary(v):
    tr = _json(V_TREP) or {}
    de = v.get("decide") or {}
    d = v.get("dft") or {}
    pv = v.get("pairs_vs_cr") or {}
    return dict(
        file=_disp(V_REPORT), sha256=_sha(V_REPORT), git_head=v.get("git_head"), code_version=v.get("code_version"),
        finished=v.get("finished"),
        targets={k: tr.get(k) for k in ("n_keys", "n_a", "n_b", "n_c", "n_a_and_b", "n_constrained_keys",
                                        "n_constrained_excluded_from_a", "n_targets")},
        n_targets=v.get("n_targets"), groups=v.get("groups"),
        n_keys_replaced=v.get("n_keys_replaced"), n_fragments_replaced_by_v=v.get("n_fragments_replaced_by_v"),
        n_rows_changed_vs_cr=v.get("n_rows_changed_vs_cr"), n_rows_changed_vs_orig=v.get("n_rows_changed_vs_orig"),
        pairs_vs_cr=dict(n_pairs_touching_v=pv.get("n_pairs_touching_v"),
                         untouched_pairs_identical=pv.get("untouched_pairs_identical"),
                         dd={t: {k: (x or {}).get(k) for k in ("n_changed_vs_cr", "in_P_main", "abs_change")}
                             for t, x in (pv.get("dd") or {}).items()}),
        dft={k: d.get(k) for k in ("n_started", "n_fail", "rate", "rate_if_d1_tol", "n_imag_fail_within_d1_tol",
                                   "statuses")},
        gates_ok={k: (x or {}).get("ok") for k, x in (v.get("gates") or {}).items()},
        trial=dict(core_h_total=de.get("trial_core_h_total"), estimate_used=(de.get("estimate") or {}).get("used"),
                   limit=(de.get("estimate") or {}).get("limit"), go=de.get("go")),
        core_h=v.get("core_h"), decisions=v.get("decisions"), outputs=v.get("outputs"),
        cyclooctyne_csv=v.get("cyclooctyne_csv"))


def cr2_totals():
    R = pd.read_parquet(REFS_CR2, columns=["job_id", "rxn_id", "frag", "key", "src"])
    ch = R[R.src.astype(str) != "own"]
    mol = ch.dropna(subset=["key"]).drop_duplicates("key")
    kind = mol.key.astype(str).str.split(":").str[0]
    return dict(source=_disp(REFS_CR2), sha256=_sha(REFS_CR2), n_reference_rows=int(len(R)),
                n_fragments_changed=int(len(ch)), fragments_by_src=_vc(ch.src.astype(str)),
                n_reactions_changed=int(ch.rxn_id.nunique()), n_molecules_changed=int(len(mol)),
                molecules_by_kind=_vc(kind),
                molecules_by_src={s: int(ch[ch.src.astype(str) == s].key.nunique()) for s in sorted(set(ch.src.astype(str)))},
                rule="rows of refs_cr2.parquet whose final reference is not the reaction's own (src Q or V)")


def pair_changes():
    O = pd.read_csv(ORIG, float_precision="round_trip")
    N = pd.read_csv(CR2, float_precision="round_trip")
    if O.pair_id.astype(str).tolist() != N.pair_id.astype(str).tolist():
        raise Refuse(3, [f"{CR2} pair order differs from {ORIG}"])
    pm = O.in_P_main.astype(str).str.lower().eq("true").to_numpy()
    para = O.pos.fillna("").astype(str).eq("p").to_numpy()
    rk = O.ref_kind.fillna("").astype(str)
    out, other = {}, {}
    anyc = np.zeros(len(O), dtype=bool)
    for t in TARGETS:
        d = N[f"dd_{t}"].to_numpy(float) - O[f"dd_{t}"].to_numpy(float)
        ch = np.abs(np.nan_to_num(d, nan=0.0)) > TOL_PAIR
        if t in CHANGED:
            out[t] = dict(n_changed=int(ch.sum()), in_P_main=int((ch & pm).sum()), para=int((ch & para).sum()),
                          para_in_P_main=int((ch & para & pm).sum()), by_ref_kind=_vc(rk[ch]),
                          abs_change=_q(np.abs(d[ch])), n_abs_gt={f"{x:g}": int((np.abs(np.nan_to_num(d)) > x).sum())
                                                                  for x in (0.05, 0.5, 1.0, 2.0)})
            anyc |= ch
        else:
            other[t] = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0
    out["any"] = dict(n_changed=int(anyc.sum()), in_P_main=int((anyc & pm).sum()),
                      para_in_P_main=int((anyc & para & pm).sum()))
    return dict(original=_disp(ORIG), original_sha256=_sha(ORIG), new=_disp(CR2), new_sha256=_sha(CR2),
                n_pairs=int(len(O)), n_P_main=int(pm.sum()), n_P_main_para=int((pm & para).sum()), changed=out,
                unchanged_targets_max_abs=other, tol=TOL_PAIR)


def pair_sets(p, names=("P_dft", "P_g1", "P_01")):
    fold_ids = set()
    for k in range(NF):
        rec = json.loads((IN.FU2_SPLITS / f"T1__f{k}.json").read_text(encoding="utf-8"))
        fold_ids |= set(map(str, rec["train"])) | set(map(str, rec["test"]))
    out = {}
    for name in names:
        s = (p.get("sets") or {})[name]
        ids = set(pd.read_csv(s["file"], usecols=["pair_id"], dtype={"pair_id": str}).pair_id)
        pl = sorted(ids & fold_ids)
        out[name] = dict(
            file=_disp(s["file"]), path=_ab(s["file"]), sha256=s.get("sha256"), n=s.get("n"),
            by_ref_kind=s.get("by_ref_kind"), para=s.get("para"), para_by_ref_kind=s.get("para_by_ref_kind"),
            by_pos=s.get("by_pos"), in_P_main=s.get("in_P_main"), in_P_main_para=s.get("in_P_main_para"),
            in_P_sens=s.get("in_P_sens"), in_A_main=s.get("in_A_main"), in_T_main=s.get("in_T_main"),
            folds_T1=s.get("folds_T1"),
            dropped={k: (s.get("dropped") or {}).get(k) for k in ("n", "by_reason", "n_used_in_T1", "used_by_reason")},
            dH_changed_by_rereferencing=s.get("dH_changed_by_rereferencing"),
            n_in_T1_lists=len(pl), pair_list_sha256_trainer_rule=DP.list_sha([(q,) for q in pl]))
    return out


def p_summary(p):
    return dict(
        file=_disp(P_REPORT), sha256=_sha(P_REPORT), finished=p.get("finished"), git_head=p.get("git_head"),
        rungs={r: {k: ((p.get("rungs") or {}).get(r) or {}).get(k) for k in (
            "table", "table_sha256", "build_report", "build_report_sha256", "counts", "references", "P_a_max_abs",
            "reassembly", "f0")} for r in STAGE_RUNGS},
        fu2_check={k: (p.get("fu2_check") or {}).get(k) for k in ("ok", "n_pairs", "n_fu2_used", "n_P_g1",
                                                                  "n_only_in_P_g1", "n_only_in_fu2", "pairs_sha256")},
        dft_vs_g1=p.get("dft_vs_g1"), h_changed_by_rereferencing=p.get("h_changed_by_rereferencing"),
        labels=p.get("labels"), folds_files=(p.get("folds") or {}).get("files"), decisions=p.get("decisions"))


def c_summary(c):
    parts = {}
    for k, d in (c.get("parts") or {}).items():
        d = d or {}
        parts[k] = {x: d.get(x) for x in ("file", "sha256", "status", "slurm_job", "host", "git_head", "python",
                                          "prefix", "versions", "device", "finished")}
    extra = {n: _sha(L.RES / n) for n in ("C_tests_gpu.json", "C_tests_cpu.json", "C_smoke_atoms.json",
                                          "C_smoke_xgb.json", "C_l6_xgb.json", "M1_gpu_estimate.json")}
    return dict(file=_disp(C_TESTS), sha256=_sha(C_TESTS), committed=_committed(C_TESTS),
                last_commit=_last_commit(C_TESTS),
                **{k: c.get(k) for k in ("all_pass", "status", "stage_verified", "g2_verified", "all_pass_scope",
                                         "labels_final", "failed", "errors", "not_final", "tests_version", "merged",
                                         "code_sha256", "gpu_estimate")},
                labels={k: (c.get("labels") or {}).get(k) for k in ("path", "sha256", "final")},
                rungs={r: {k: ((c.get("rungs") or {}).get(r) or {}).get(k) for k in ("status", "why", "table_sha256")}
                       for r in ("dft", "g1", "g2")},
                p_report=c.get("p_report"),
                checks={k: (((c.get("checks") or {}).get(k)) or {}).get("status") for k in CHECKS},
                l10_tol=(c.get("tol") or {}).get("L10"),
                parts=parts, files_sha256=extra, decisions=c.get("decisions"))


def hf_phase_a():
    A = pd.read_csv(A_SUMMARY)
    rep = _json(A_REPORT) or {}
    out = {}
    for g in ("dft", "g1"):
        sub = A[A.geom == g].set_index("target")
        out[g] = {t: {k: (None if pd.isna(sub.at[t, k]) else (float(sub.at[t, k]) if k not in ("band_para",)
                                                              else str(sub.at[t, k])))
                      for k in ("n", "n_para", "mae_all", "mae_para", "null_mae_para", "beta_mean",
                                "fu2_gnn_ag_mae_para", "band_para") if k in sub.columns}
                  for t in H9 if t in sub.index}
    return dict(file=_disp(A_SUMMARY), sha256=_sha(A_SUMMARY), report=_disp(A_REPORT), report_sha256=_sha(A_REPORT),
                fit=rep.get("fit"), note=rep.get("note"), baseline=rep.get("baseline"),
                n_pairs_P_main=rep.get("n_pairs_P_main"), labels="original (labels_d1_hf.parquet H, F_pairs.csv)",
                summary=out)


FU2_MODELS_MD = ("AW-A+G", "GNN-A+G", "TF-A+G", "C-XGB", "REP", "null")
FU2_MODELS_ALL = ("AW-A", "AW-A+G", "GNN-A", "GNN-A+G", "TF-A", "TF-A+G", "REP", "C-XGB", "C-rep", "XGBW", "null")


def fu2_results():
    M = _json(FU2_RES / "M_report.json") or {}
    units = M.get("units") or {}
    m_ag = {}
    for m in ATOM_MODELS:
        for k in range(NF):
            u = units.get(f"T1__{m}__AG__f{k}") or {}
            m_ag[f"{m}-A+G f{k}"] = dict(cell=(u.get("chosen") or {}).get("cell"),
                                         best_epoch=(u.get("chosen") or {}).get("best_epoch"),
                                         n_train=u.get("n_train"), n_test=u.get("n_test"))
    pred_sha = M.get("predictions_sha256")
    out = dict(m_report=_disp(FU2_RES / "M_report.json"), m_report_sha256=_sha(FU2_RES / "M_report.json"),
               m_predictions_sha256=pred_sha, m_n_units=M.get("n_units"), units_A_G=m_ag,
               labels="original (F_pairs.csv)", e_present=False, e_searched=[str(d) for d in FU2_E_DIRS], e_rejected=[])
    # PR-7: E_check ok, E_metrics present, and the M_predictions that E-0 hashed = this branch's FU2 Phase M predictions
    # (E_check keys are absolute paths of the worktree E ran in: matched by the repo-relative tail)
    tail = "/analysis/d1_pair/results/fu2/M_predictions.parquet"
    for d in FU2_E_DIRS:
        chk = _json(d / "E_check.json")
        if not (chk and chk.get("ok") and (d / "E_metrics.csv").is_file()):
            out["e_rejected"].append(dict(dir=str(d), why="no E_check.json ok with E_metrics.csv"))
            continue
        got = sorted({s for p, s in (chk.get("inputs_sha256") or {}).items() if str(p).endswith(tail)})
        if not pred_sha or got != [pred_sha]:
            out["e_rejected"].append(dict(dir=str(d), why=f"E_check M_predictions sha256 {got} != this branch's "
                                                          f"M_report predictions_sha256 {pred_sha}"))
            continue
        break
    else:
        return out
    E = pd.read_csv(d / "E_metrics.csv")
    br = (_json(d / "E_bend_rates.json") or {}).get("rates") or {}
    mets = {}
    for m in FU2_MODELS_ALL:
        mets[m] = {}
        for pos in ("p", "all"):
            sub = E[(E["model"] == m) & (E["set"] == "P_main") & (E["pos"] == pos)]
            mets[m][pos] = {str(t): dict(n=int(r["n"]), mae=float(r["mae"]), rmse=float(r["rmse"]),
                                         source=str(r["source"]),
                                         r2_fam_median=(None if pd.isna(r["r2_fam_median"]) else float(r["r2_fam_median"])))
                            for t, r in sub.groupby("target").first().iterrows()}
    bends = {m: {t: {k: (((br.get(m) or {}).get("all") or {}).get(t) or {}).get(k)
                     for k in ("n_fam", "sens", "sens_ci", "n_bent_dft", "fpr", "fpr_ci", "n_straight_dft")}
                 for t in H9} for m in FU2_MODELS_ALL}
    descr = {}
    for m in FU2_MODELS_MD:
        r = {}
        for t in STAGE2_RULE["targets"]:
            mae = ((mets.get(m) or {}).get("p") or {}).get(t, {}).get("mae")
            b = bends.get(m, {}).get(t, {})
            sens, fpr = b.get("sens"), b.get("fpr")
            r[t] = dict(para_mae=mae, sens=sens, fpr=fpr,
                        meets=bool(mae is not None and sens is not None and fpr is not None
                                   and mae <= STAGE2_RULE["para_mae_max"] and sens >= STAGE2_RULE["bend_sensitivity_min"]
                                   and fpr <= STAGE2_RULE["false_positive_rate_max"]))
        r["both"] = all(r[t]["meets"] for t in STAGE2_RULE["targets"])
        descr[m] = r
    files = {e.name: L.sha256(Path(e.path)) for e in sorted(os.scandir(d), key=lambda e: e.name)
             if e.name.startswith("E_") and e.is_file()}
    git_state = {n: _git_state_at(d / n) for n in files}
    rep = _json(d / "E_representative.json") or {}
    out.update(e_present=True, e_dir=str(d), e_files_git_state=git_state,
               e_files_git_counts={s: sum(1 for v in git_state.values() if v == s) for s in sorted(set(git_state.values()))},
               e_predictions_match=True,
               e_representative_not_A_G={t: m for t, m in rep.items() if not str(m).endswith("A+G")},
               e_check=dict(ok=chk.get("ok"), n_gates=chk.get("n_gates"),
                                                         n_fail=chk.get("n_fail"), date=chk.get("date"),
                                                         git_head=chk.get("git_head"),
                                                         evaluate_sha256=chk.get("evaluate_sha256"),
                                                         n_P_main_test_pairs=chk.get("n_P_main_test_pairs"),
                                                         n_P_main_cores=chk.get("n_P_main_cores")),
               e_primary_verdict=_json(d / "E_primary_verdict.json"), e_representative=_json(d / "E_representative.json"),
               metrics_P_main=mets, bends_all=bends, stage2_rule_descriptive=descr, e_files_sha256=files)
    return out


def deviations(v, p):
    s = _json(START) or {}
    f0 = _json(F0_REPORT) or {}
    gr, f1 = f0.get("gate_rates") or {}, f0.get("F1") or {}
    n_done = gr.get("n_g0_done") or f0.get("n_targets")
    vd = v.get("decisions") or {}
    s02 = s.get("0-2") or {}
    ci = s02.get("core_identical") or {}
    thr = gr.get("threshold")
    rate_f1 = f1.get("rate")
    rate_wo = (f1.get("n_without_B6_only") / n_done) if (n_done and f1.get("n_without_B6_only") is not None) else None
    rate_b6 = (gr.get("rate") or {}).get("B6")

    def over(x):
        return None if (x is None or thr is None) else bool(x > thr)
    cls = f1.get("classes") or {}
    by_class = {k: sum(int(n) for c, n in cls.items() if pat in c)
                for k, pat in (("engine_gate", "engine_gate"), ("parse_fA", "parse:fA"), ("B6", ":B6:"))}
    txt = F0_DECISION.read_text(encoding="utf-8")

    def md_count(pat):
        m = re.search(r"`" + pat + r"[^`]*`\s+([\d,]+)", txt)
        return int(m.group(1).replace(",", "")) if m else None
    md_counts = {k: md_count(pat) for k, pat in (("engine_gate", re.escape("ext:B1:engine_gate1")),
                                                  ("parse_fA", re.escape("ext:B1:parse:fA")),
                                                  ("B6", re.escape("ext:B6:no_imaginary_frequency")))}
    tol_txt = " ".join(str(x) for x in ((v.get("dft") or {}).get("note_d1_tol"), vd.get("V10_imag")) if x)
    mt = re.search(r"(-?\d+(?:\.\d+)?)\s*cm-1", tol_txt)
    return dict(
        start_0_2=dict(decision=s02.get("decision"), changed=s02.get("changed"), deletions=s02.get("deletions"),
                       core_identical=ci, n_core=len(ci), n_core_identical=sum(1 for x in ci.values() if x is True)),
        start_0_3=dict(**start_0_3_record(),
                       results_diff_vs_6b9fbeb2=(s.get("0-3") or {}).get("results_diff_vs_6b9fbeb2"),
                       labels_d1_hf_sha256=(s.get("0-3") or {}).get("labels_d1_hf_sha256")),
        f0_f1=dict(report=_disp(F0_REPORT), report_sha256=_sha(F0_REPORT), decision_md=_disp(F0_DECISION),
                   decision_md_sha256=_sha(F0_DECISION), decision_text=txt, threshold=thr,
                   n_g0_done=n_done, n_F1=f1.get("n"), rate_F1=rate_f1, n_B6_only=f1.get("n_B6_only"),
                   n_without_B6_only=f1.get("n_without_B6_only"), rate_without_B6=rate_wo, rate_B6=rate_b6,
                   F1_over=over(rate_f1), without_B6_over=over(rate_wo), B6_over=over(rate_b6),
                   classes=cls, report_counts_by_class=by_class, decision_md_counts_by_class=md_counts,
                   decision_md_counts_equal_report=all(md_counts[k] == by_class[k] for k in by_class),
                   build_dft_f0=((p.get("rungs") or {}).get("dft") or {}).get("f0")),
        v_decisions={k: vd.get(k) for k in ("V7_rr", "V8_xtb", "V10_imag", "V12_repro", "V13_v4", "V17_name")},
        v_dft_sensitivity=dict(rate=(v.get("dft") or {}).get("rate"),
                               rate_if_d1_tol=(v.get("dft") or {}).get("rate_if_d1_tol"),
                               d1_tol_cm1=(float(mt.group(1)) if mt else None)))


def code_records():
    gated = {**CODE_ATOMS, **CODE_XGB}
    rec = dict(
        gated={k: dict(path=_disp(p), sha256=_sha(p)) for k, p in sorted(gated.items())},
        fingerprint_keys=dict(train_atoms={k: _sha(p) for k, p in CODE_ATOMS.items()},
                              train_xgb={k: _sha(p) for k, p in CODE_XGB.items()}),
        ladder_not_gated={k: _sha(p) for k, p in LADDER_CODE.items() if k not in ("lad_common.py", "train_atoms.py",
                                                                                     "train_xgb.py")},
        ladder_git_state={k: ("committed, = HEAD" if _committed(p) else ("tracked, modified" if _tracked(p) else "untracked"))
                          for k, p in LADDER_CODE.items()},
        aux_not_gated={k: _sha(p) for k, p in AUX_CODE.items()},
        generator={"prereg.py": _sha(Path(__file__).resolve())},
        fu2_phase_e_evaluate={_disp(p) if str(p).startswith(str(L.REPO)) else str(p): _sha(p)
                              for p in FU2_EVAL if Path(p).is_file()})
    return rec


def gated_inputs(p):
    """inputs_sha256: absolute unresolved path -> (role, sha256), in a fixed order (PR-2)."""
    G = {}

    def add(path, role):
        G.setdefault(_ab(path), role)
    add(CR2, "labels: F_pairs_cr2.csv (spec 4-1 y; need)")
    for r in STAGE_RUNGS:
        add(IN.table_path(r), f"input table {r} (need)")
    for name in ("P_dft", "P_g1", "P_01"):
        add(p["sets"][name]["file"], f"pair set {name}")
    add(ORIG, "original F_pairs.csv (L-11 invariant on every label load)")
    add(IN.FU2_S_SPLITS, "FU2 S_splits.json (split index)")
    add(IN.S_FOLDS, "S_folds.csv (folds, cores; XGB inner GroupKFold)")
    for k in range(NF):
        add(IN.FU2_SPLITS / f"T1__f{k}.json", f"T1 fold {k} split file")
    for r in STAGE_RUNGS:
        cfg = IN.RUNGS[r]
        res = Path(cfg["fu2_res"])
        for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json"):
            add(res / n, f"tokens {r}: {n}")
        add(cfg["x182"], f"global 182 features {r}")
    for k, pth in {**CODE_ATOMS, **CODE_XGB}.items():
        add(pth, f"code {k} (trainer fingerprint code_sha256)")
    out = {}
    miss = []
    for path, role in G.items():
        s = _sha(path)
        if s is None:
            miss.append(path)
        out[path] = dict(role=role, sha256=s, column=_column(path))
    if miss:
        raise Refuse(3, [f"gated inputs missing: {miss}"])
    return out


def stage2_supplement_paths():
    """md section 10 / PR-5: the only entries the stage-2 supplement adds to inputs_sha256 (g2 column + P_012)."""
    cfg = IN.RUNGS["g2"]
    res = Path(cfg["fu2_res"])
    out = [(IN.table_path("g2"), "input table g2 (need of a g2 process)"), (IN.set_path("P_g2"), "pair set P_g2"),
           (IN.set_path("P_012"), "pair set P_012")]
    out += [(res / n, f"tokens g2: {n}") for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json")]
    out.append((Path(cfg["x182"]), "global 182 features g2"))
    return out


def _column(path):
    """spec 4-1 column of a path by inputs.check_paths (dft / g1 / g2 / dft+g1 / shared / unclassified)."""
    roots = IN.rung_roots()
    q = Path(os.path.abspath(path))
    best = None
    for root, lab in roots:
        if q == root or root in q.parents:
            if best is None or len(str(root)) > len(str(best[0])):
                best = (root, lab)
    return best[1] if best else "unclassified (verified by every process)"


def not_gated_records():
    files = [P_REPORT, C_TESTS, V_REPORT, Q_REPORT, START, F0_REPORT, F0_DECISION, G0_REPORT, V_TREP,
             L.RES / "V_decide.json", L.RES / "V_summary.md", L.RES / "labels_d1_cr2.parquet", L.RES / "labels_d1_cr.parquet",
             L.LAD / "labels_d1_cr2.json", REFS_CR2, L.LAD / "F_pairs_cr.csv", L.LAD / "Q_keys.parquet",
             IN.build_report_path("dft"), IN.build_report_path("g1"), A_SUMMARY, A_REPORT, L.D1HF_LABELS,
             FU2_RES / "M_report.json", FU2_RES / "M_predictions.parquet", ADD2_JSON, REV9_JSON, SPEC,
             *PREV.values()]
    return {_disp(f): _sha(f) for f in files}


# ================================================================ markdown
def _tab(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def _cell(c):
    return f"L{c.get('n_layers')}·폭 {c.get('width')}" if isinstance(c, dict) else "–"


def make_md(D):
    g, k, Q, V, T, PC, PS, PSUM, C, HF, FU, DEV, CODE, GI, NOTES = (
        D["grids"], D["constants"], D["q"], D["v"], D["cr2_totals"], D["pair_changes"], D["pair_sets"], D["p"],
        D["c_tests"], D["hf_phase_a"], D["fu2"], D["deviations"], D["code"], D["inputs"], D["notes"])
    md = []
    a = md.append
    a(f"# PREREG ADD3 — D1PAIR 사전 등록 추가본 (D1-REF·LADDER 1단계: dft·g1, {D['date']})")
    a("")
    a(f"이 문서는 `PREREG_D1PAIR.md`(commit {D['prev']['PREREG_D1PAIR.md']}), `PREREG_D1PAIR_ADD1.md`"
      f"(commit {D['prev']['PREREG_D1PAIR_ADD1.md']}), `PREREG_D1PAIR_ADD2.md`(commit {D['prev']['PREREG_D1PAIR_ADD2.md']})"
      "에 더하는 추가본이다. 셋은 바꾸지 않는다.")
    a(f"근거: 명령서 `docs/specs/D1LADDER.md` 9절(sha256 `{D['spec_sha256']}`). 이 문서와 json은 "
      "`analysis/d1_ladder/prereg.py write`가 결과 파일만 읽어 만들었다. 결과 수치는 모두 결과 파일에서 옮겼다. 규칙의 기준값은 "
      "명령서를 인용했고, 검토자 수치는 명령서 0-1·9절 인용이며 그 자리에 출처를 적었다(생성 코드 sha256은 json "
      "`code.generator`).")
    a("범위: **1단계(rung `dft`·`g1`)**. 2단계(`g2`)는 10절의 보완 규칙으로 따로 등록한다.")
    a("")
    a("Phase M1 학습(`train_atoms.py run`, `train_xgb.py run`)은 이 문서와 `prereg_d1pair_add3.json`이 commit된 뒤에만 돈다. "
      "두 학습 코드의 `gate_add3`가 매 실행 전에 확인한다.")
    a("- 두 파일이 git에 있고 HEAD 뒤로 바뀌지 않았는지")
    a("- md sha256이 json `md_sha256`과 같은지")
    a(f"- json `inputs_sha256`에 정답 파일과 학습하는 rung의 입력표가 있는지, 그리고 적힌 **{len(GI)}개 파일**의 sha256이 "
      "지금 파일과 같은지(다른 rung 열에만 속한 파일은 열지 않는다, L-4)")
    a("")
    a("하나라도 다르면 exit 3이다. 모든 unit 지문(`add3`)에 이 md의 sha256이 들어가고, collect는 그 값을 다시 비교한다.")
    a("")
    a(f"**지금까지 본 것** (작성 {D['written']}):")
    a(f"- Phase M1 학습은 한 번도 돌지 않았다(`$LAD/units`의 `L__` 파일 {D['n_units_before']}개).")
    a("- Phase C 검사 코드(`tests/run_tests.py`)가 계산하는 MAE는 L-3(합성 자료), L-10(원래 라벨·원래 H의 LINE = D1-HF "
      "Phase A 재현), smoke의 검증 짝 MAE(2 epoch, 새 정답)뿐이다(코드 기준). 새 정답의 시험 짝 점수는 계산하지도 보지도 "
      "않았다.")
    a("- 원래 라벨로 이미 본 결과(D1-HF Phase A, FU2 g1·S, 검토자 사후 탐색·사전 점검)는 7절에 그대로 적는다.")
    a("")

    # 1
    a("## 1. 입력 고정")
    a("")
    a(f"**정답:** `{PC['new']}` sha256 `{PC['new_sha256']}` (V_report.json all_pass, outputs에 같은 sha256; "
      "P_report.json `labels` ids_ok). 원래 라벨 `analysis/d1_pair/results/F_pairs.csv`는 비교용이다.")
    a("")
    a(f"**코드:** commit `{D['git_head']}` (Phase C 결과가 commit된 HEAD; 학습 코드, Phase C 코드 5개, `C_tests.json`이 "
      "HEAD와 같음을 확인했다. 다른 d1_ladder 파일의 git 상태는 json `code.ladder_git_state`). 아래 파일은 두 학습 코드의 지문 "
      "`code_sha256`에 들어가는 코드이고, `inputs_sha256`에도 있어 gate가 매 실행 전에 확인한다.")
    a("")
    md.extend(_tab(["파일", "sha256"], [[f"`{v['path']}`", f"`{v['sha256']}`"] for v in CODE["gated"].values()]))
    a("")
    a("gate가 확인하지 않는 코드(`inputs.py`, `evaluate.py`, `tests/run_tests.py`, `refs_q.py`, `refs_v.py`, `g0f0.py`, "
      "job 파일, `fu2/atoms.py`, D1-HF·rev 9 평가 코드, FU2 Phase E `evaluate.py`)의 sha256은 json `code`에 적는다. "
      "d1_ladder `evaluate.py`의 E1 코드는 학습과 무관하게 따로 작성·commit되므로 gate 밖에 둔다(E1 규칙은 5·6절에 고정, "
      "LINE 함수는 3-5절·9절 7번의 소스 sha256으로 고정).")
    a("")
    a("**입력 파일** (gate가 확인; 열 = 명령서 4-1의 rung 열, `inputs.check_paths`):")
    a("")
    md.extend(_tab(["파일", "역할", "열", "sha256"],
                   [[f"`{_disp(pth)}`", x["role"], x["column"], f"`{x['sha256']}`"] for pth, x in GI.items()
                    if not x["role"].startswith("code ")]))
    a("")
    a("토큰 파일(반응별 npz)의 sha256은 각 rung의 `T_manifest.csv`에 있고, 학습 코드가 읽을 때마다 확인한다. "
      "`P_report.json`, `C_tests.json`, Q·V 보고 등은 2단계에서 다시 쓰이므로 gate에 넣지 않고 json `not_gated_sha256`에 적는다.")
    a("")

    # 2
    a("## 2. Phase Q·V — 기준 구조 규칙과 결과")
    a("")
    a("### 2-1. 규칙 (명령서 2·3절, 결과를 본 뒤 바꾸지 않음)")
    a("- **분자 키(Q-2):** 반응 SMILES에서 나눈 dipole·dipolarophile의 RDKit 정규 isomeric SMILES. 거울상 이성질체는 한 "
      "분자(사전순 앞의 SMILES가 키), E/Z·cis/trans 부분입체 이성질체는 다른 분자. Coley 규칙이 이면각을 고정한(constrained) "
      "dipole 기준은 키에 고정 이면각 상태를 붙여 같은 상태끼리만 묶는다. 같은 키 안의 원소 구성·결합 그래프가 같음을 "
      "확인했다.")
    a("- **재조립(Q-4):** 키마다 라벨에 실제로 쓰인 기준 단일점 에너지의 최저 E_min을 고르고, E_orig − E_min > 0.05 "
      "kcal/mol인 반응만 E_min과 그 좌표로 바꾼다. d1·d2·barrier만 기존 조립 함수로 다시 계산하고, TS 에너지·EDA 6채널·"
      "c_ghost는 그대로다(gate Q-a–Q-e).")
    a("- **확인 계산(V-1–V-4):** 대상 = (a) dev > 0.5 kcal/mol 반응이 있는 분자, (b) cyclooctyne 계열 dipolarophile(골격 "
      "D, Dp) 전부, (c) 대조군 20개(`default_rng(20261023)`, dipolarophile 10·dipole 10); constrained dipole 키는 제외. "
      "형태 탐색 = ETKDGv3 300개(randomSeed 20261023) + D1 기준에서 출발한 randomize-and-relax 300개 → "
      "`xtb --opt tight --alpb water --gfn 2` → 중복 제거(무거운 원자 RMSD < 0.1 Å 또는 ΔE < 0.05 kcal/mol) → xTB 하위 "
      "5개 + D1 기준 형태. DFT = D1 autodE 반응물과 같은 B3LYP-D3BJ/def2-SVP SMD(water) Opt + Freq(허수 진동수는 한 번 "
      "밀어 다시 최적화, 남으면 버림), 라벨 수준 단일점(라벨 `frag*_rel` 입력의 좌표만 바꿈), HF/6-31G* 단일점.")
    a("- **최종 기준(V-4):** 분자마다 min(Q의 E_min, V 형태)을 라벨 수준 단일점 에너지로 고른다. V 형태가 0.05 kcal/mol "
      "넘게 낮으면 바꾸고 Q-4·Q-5 규칙으로 다시 조립한다 → `F_pairs_cr2.csv`(이후 모든 단계의 정답). 원래 라벨은 지우지 "
      "않는다.")
    a("")
    a("### 2-2. 결과")
    au = Q["audit"]
    a(f"- **Q 감사:** dev > 0.05 / 0.5 / 1 / 2 kcal/mol인 반응 {_n(au['0.05']['reactions'])} / {_n(au['0.5']['reactions'])} / "
      f"{_n(au['1']['reactions'])} / {_n(au['2']['reactions'])}개, 영향받는 분자 {_n(au['0.05']['molecules'])} / "
      f"{_n(au['0.5']['molecules'])} / {_n(au['1']['molecules'])} / {_n(au['2']['molecules'])}개. gate "
      + ", ".join(f"{x} {'통과' if y else '실패'}" for x, y in Q["gates_ok"].items()) + ".")
    ra = Q["reassembly"]
    a(f"- **Q 재조립:** 반응 {_n(ra['n_rows_changed'])}개(조각 {_n(ra['n_fragments_changed'])}개, 분자 "
      f"{_n(ra['n_source_molecules'])}개)의 기준을 바꿨다. Δbarrier 중앙값 {_f((ra.get('delta_barrier') or {}).get('p50'))}, "
      f"최대 {_f((ra.get('delta_barrier') or {}).get('max'), 2)} kcal/mol.")
    vg = V["groups"] or {}
    tr = V["targets"]
    a(f"- **V 대상:** 분자 {_n(V['n_targets'])}개 = (a) {_n(tr.get('n_a'))}, (b) {_n(tr.get('n_b'))} (겹침 "
      f"{_n(tr.get('n_a_and_b'))}), (c) {_n(tr.get('n_c'))}; constrained dipole 키 {_n(tr.get('n_constrained_keys'))}개 제외"
      f"(그중 dev > 0.5인 것 {_n(tr.get('n_constrained_excluded_from_a'))}개).")
    rows = []
    for gk, gname in (("a", "(a) dev > 0.5"), ("b", "(b) cyclooctyne 계열"), ("c", "(c) 대조군")):
        x = vg.get(gk) or {}
        lo = x.get("lowering_of_lower") or {}
        rows.append([gname, _n(x.get("n")), _n(x.get("n_lower_gt_0_05")), _pct(x.get("frac_lower")), _f(lo.get("p50")),
                     _f(lo.get("max"))])
    md.extend(_tab(["V 대상", "분자", "더 낮은 형태 (> 0.05)", "비율", "낮아진 양 중앙값", "최대"], rows))
    d = V["dft"]
    a("")
    a(f"- **V 결과:** 기준이 바뀐 분자 키 {_n(V['n_keys_replaced'])}개, V 기준을 쓰게 된 조각 "
      f"{_n(V['n_fragments_replaced_by_v'])}개, Q 대비 바뀐 반응 {_n(V['n_rows_changed_vs_cr'])}개. DFT 최적화 실패 "
      f"{_n(d.get('n_fail'))}/{_n(d.get('n_started'))} ({_pct(d.get('rate'))}). gate "
      + ", ".join(f"{x} {'통과' if y else '실패'}" for x, y in V["gates_ok"].items()) + ".")
    a(f"- **최종(cr2, 원래 라벨 대비):** 기준이 바뀐 반응 {_n(T['n_reactions_changed'])}개 (V_report "
      f"`n_rows_changed_vs_orig` {_n(V['n_rows_changed_vs_orig'])}), 조각 {_n(T['n_fragments_changed'])}개, 분자 "
      f"{_n(T['n_molecules_changed'])}개 ("
      + ", ".join(f"{x} {_n(y)}" for x, y in T["molecules_by_kind"].items()) + "; 출처 "
      + ", ".join(f"{x} {_n(y)}" for x, y in T["molecules_by_src"].items()) + ").")
    rows = [[f"dd_{t}", _n(c['n_changed']), _n(c['in_P_main']), _n(c['para_in_P_main']),
             _f((c.get('abs_change') or {}).get('p50')), _f((c.get('abs_change') or {}).get('max'), 2)]
            for t, c in PC["changed"].items() if t in CHANGED]
    c_any = PC["changed"]["any"]
    rows.append(["셋 중 하나라도", _n(c_any["n_changed"]), _n(c_any["in_P_main"]), _n(c_any["para_in_P_main"]), "", ""])
    a("")
    um = PC["unchanged_targets_max_abs"]
    umax = float(max(um.values())) if um else 0.0
    a(f"바뀐 짝(전체 {_n(PC['n_pairs'])}짝, |Δ| > {PC['tol']:g} kcal/mol; 다른 {len(um)}개 타깃의 최대 차이 "
      f"{umax:.1e} kcal/mol):")
    a("")
    md.extend(_tab(["타깃", "바뀐 짝", "P_main", "P_main para", "바뀐 양 중앙값", "최대"], rows))
    a("")
    a("### 2-3. 검토자의 기준 구조 HF 사전 점검 (명령서 0-1절)")
    rv = (Q["hf_precheck"].get("reviewer") or {})
    rp = (Q["hf_precheck"].get("replication_P_all") or {})
    rq = REVIEWER_PRECHECK_QUOTE
    try:
        share = rv["flagged"] / rv["P_all_reactions"]
    except (KeyError, TypeError, ZeroDivisionError):
        share = None
    a(f"- 검토자: 짝에 쓰인 {_n(rv.get('P_all_reactions'))}반응 중 {_n(rv.get('flagged'))}반응({_pct(share)}; 명령서 "
      f"인용 {rq['flagged_share_pct']} %)이 같은 분자의 최저 기준보다 1 kcal/mol 넘게 높은 기준을 썼다. 편차 중앙값 "
      f"{rv.get('median_dev')} kcal/mol. dipolarophile {_n(rv.get('dipolarophile_rows'))}건 중 cyclooctyne(D "
      f"{rq['cyclooctyne_D']}, Dp {rq['cyclooctyne_Dp']}; 명령서 인용)이 "
      f"{_pct(rv.get('cyclooctyne_share_of_dipolarophile'), 0)}다. dipole {_n(rv.get('dipole_rows'))}건 — "
      f"명령서 인용: \"{rq['caveat_ko']}\".")
    a(f"- 출처: Q_report.json `hf_precheck.reviewer`(수치); D·Dp 개수, 비율 인용값과 dipole 단서는 {rq['source']}.")
    a(f"- Q의 재현(같은 키, `labels_d1_hf.parquet` geom=dft): HF 판정 {_n(rp.get('hf_flag'))}반응, DFT 판정 "
      f"{_n(rp.get('dft_flag'))}반응, 반응 단위 일치율 {_pct(rp.get('agreement'))}, HF 편차 중앙값 "
      f"{_f(rp.get('hf_median_dev_flagged_reactions'), 2)} kcal/mol.")
    a("")

    # 3
    a("## 3. LADDER 설계 (명령서 4절 전체)")
    a("")
    a("### 3-1. 단계(rung)마다 입력은 모두 그 구조에서 계산한다")
    tabs = {r: (PSUM["rungs"].get(r) or {}).get("table_sha256") for r in STAGE_RUNGS}
    md.extend(_tab(["", "`dft`", "`g1`", "`g2` (2단계)"], [
        ["역할", "상한(구조가 정답일 때), 배포 불가", "구조를 xTB로 바꾸면 잃는 것", "배포 성능 (SMILES → xTB)"],
        ["TS 좌표", "라벨 TS (D1-HF geom=dft EDA 입력, `$LAD/dft_final/<rxn>/ts.xyz`)", "`G1_ROOT_D1F/<rxn>/ts.xyz`",
         "`$G2/g2_final/<rxn>/ts.xyz`"],
        ["반응물 좌표(구조 feature)", "반응별 라벨 기준 구조 (dft `frag*_rel` 입력)", "G1 `rel1/rel2.xyz`", "G2 `rel1/rel2.xyz`"],
        ["`ts.hess` (B6)", "새 계산: 라벨 TS 고정 ORCA `! XTB2 ALPB(water) Freq` (G0)", "기존", "D1-G2"],
        ["원자 토큰 (FU2)", "F0 `$LAD/fu2/tokens` (fu2-tokens-3-dft)", "`$D1PAIR_SCRATCH/fu2/tokens` (fu2-tokens-3)",
         "D1-G2 Phase F"],
        ["전역 182", "`$LAD/X182_d1_dft.parquet`", "`$D1PAIR_SCRATCH/X182_d1.parquet`", "`$G2/X182_d1_g2.parquet`"],
        ["HF/6-31G* 9값 (H)", "`labels_d1_hf.parquet` geom=dft + V HF 단일점", "같은 파일 geom=g1", "`labels_d1_hf_g2.parquet`"],
        ["입력표", f"`$LAD/inputs/dft.parquet` `{str(tabs['dft'])[:16]}…`",
         f"`$LAD/inputs/g1.parquet` `{str(tabs['g1'])[:16]}…`", "2단계"]]))
    a("")
    a("- 구조 feature(토큰, 182)는 모든 rung에서 반응별 기준 구조를 그대로 쓴다(기존 파이프라인과 같음).")
    a("- **H 재조립:** H의 d1·d2·barrier는 rung 안에서 같은 기준 규칙으로 다시 조립한다.")
    a("  - `dft`: Q·V가 고른 최종 기준 구조의 HF 에너지(`$LAD/refs_cr2.parquet` `e_hf_eh`: 자기 기준 → labels_d1_hf "
      "geom=dft 그 행 값, Q 출처 → 출처 행 값, V 출처 → V의 HF 단일점).")
    a("  - `g1`(2단계 `g2`도 같음): 그 rung의 반응별 기준 구조 중 분자 키(Q-2)마다 HF `e_frag{k}_rel_eh`가 가장 낮은 것"
      "(문턱 없음, 동점은 job_id가 작은 것). 배포 때 반응물마다 기준을 한 번만 계산하는 것과 같다.")
    a("  - 조립은 D1-HF `build.py`(`parse_rxn`)에 기준 에너지만 바꿔 넣는다. gate P-a: 원래 기준을 넣으면 "
      "`labels_d1_hf.parquet`가 1e-9 안에서 재현된다. H의 EDA 6채널은 그대로다.")
    a("- H = HF 값 9개 `d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier` (kcal/mol, 반응별).")
    a("- 한 rung의 학습·평가는 그 rung 열의 파일만 읽는다(L-4). 정답 y는 모든 rung에서 `F_pairs_cr2.csv`다.")
    a("")
    a("### 3-2. 입력 두 가지 (arm)")
    md.extend(_tab(["arm", "원자 모델 입력", "XGB 입력", "학습 대상", "예측"], [
        ["**S** (구조만)", "원자 토큰 + 전역 182 (FU2 변형 \"A+G\")", "182 (rev 9 arm C)", "ΔΔ", "모델 출력"],
        ["**SH** (구조 + HF)", "원자 토큰 + 전역 (182 + H 9)", "182 + H 9", "잔차 r = ΔΔ − β·ΔH", "β·ΔH + 모델 출력"]]))
    a("")
    a("**SH의 기준선(검량선):**")
    a("- 타깃 t마다 ΔH_t = H_t(a) − H_t(b), β_t = Σ ΔH_t·Δy_t ÷ Σ ΔH_t², **절편 없음**(절편이 있으면 반대칭이 깨진다).")
    a("- β는 **그 학습에 쓰는 짝으로만** 맞춘다: 튜닝 때는 안쪽 학습 쪽, 최종 모델은 바깥 학습 쪽 전체(T_main: 기준 짝 + "
      "anchor 짝). c_ghost는 HF 대응 값이 없으므로 β = 0.")
    a("")
    a("**SH의 학습 규칙(FU2 규칙에서 이것만 바꾼다):**")
    a("- 출력 = 잔차. 표준화는 학습 쪽 잔차 표준편차(ddof 0)로 나누기만 한다(평균은 빼지 않는다). 전역 H 9열은 182열처럼 "
      "그 학습의 학습 반응으로 표준화한다.")
    a(f"- 손실 = 잔차 MAE(표준화, 10 타깃) + λ·|ŷ_barrier − Σ 9채널 ŷ| / s_barrier, λ = {k['lambda_identity']}. 벌점은 "
      "**기준선을 더한 최종 예측 ŷ**로 계산하고, s_barrier는 원래 y의 학습 쪽 표준편차다.")
    a("- 튜닝·early stopping 지표 = 최종 예측 ŷ의 검증 MAE를 **원래 y의 학습 쪽 표준편차**로 나눈 값의 9 주 타깃 평균"
      "(FU2와 같은 단위).")
    a("- (a,b)/(b,a) 순서 뒤집기는 잔차와 기준선을 함께 뒤집는다(FU2 `epoch_batches`).")
    a("- XGB: 타깃마다 r_t를 학습한다. 안쪽 CV 점수 = |β·ΔH + ĝ − y|의 평균(kcal).")
    a("")
    a("### 3-3. 모델 네 개")
    ag = g["atoms"]
    md.extend(_tab(["모델", "구조·grid·학습 규칙"], [
        ["AW", "FU2 M-2 그대로, 변형 A+G; grid " + ", ".join(_cell(c) for c in ag["AW"])],
        ["GNN", "FU2 M-2 그대로, 변형 A+G; grid " + ", ".join(_cell(c) for c in ag["GNN"])],
        ["TF", "FU2 M-2 그대로, 변형 A+G (4 head); grid " + ", ".join(_cell(c) for c in ag["TF"])],
        ["XGB", "rev 9 arm C XGB 그대로: " + ", ".join(f"{x} {y}" for x, y in g["xgb_values"].items())
         + f" ({len(g['xgb_cells'])}칸), ½[h(Δx) − h(−Δx)], 양방향 학습, 안쪽 `GroupKFold({k['n_inner']})`(core), "
           f"`{g['xgb_estimator']}`"]]))
    a("")
    a(f"- 원자 모델 구조·토큰 열·반대칭은 ADD2 2절과 같다(json `ladder.models.atoms_fu2_m2`). 출처: {g['atoms_source']}; "
      f"{g['xgb_source']}.")
    a(f"- 원자 모델 학습: AdamW(lr {k['lr']}, weight decay {k['weight_decay']}), cosine(T_max {k['max_epochs']}), dropout "
      f"{k['dropout']}, batch {k['batch_pairs']}짝, 최대 {k['max_epochs']} epoch, patience {k['patience']}, 튜닝 seed "
      f"{k['tune_seed']}, 최종 seed {', '.join(map(str, k['final_seeds']))}의 평균(seed 표준편차 저장). 칸 = 검증 지표 최저"
      "(동점은 grid 순서), 최종 = 바깥 학습 쪽 전부를 best epoch만큼.")
    mrs = re.search(r"random_state\s*=\s*(\d+)", g["xgb_estimator"] or "")
    a(f"- XGB: 타깃마다 따로 튜닝(안쪽 MAE 평균 최저, 동점은 grid 순서), random_state {mrs.group(1) if mrs else '–'} 한 번 "
      "학습(위 estimator), yhat_seed_sd = 0.")
    a(f"- gate M-1′: 학습한 모든 모델에서 시험 짝 {k['gate_m1_pairs']}개(입력만)를 float64로 다시 계산해 기준선을 포함한 "
      f"최종 예측의 |g(a,b) + g(b,a)|, |g(a,a)| ≤ {k['gate_m1_tol_kcal']:g} kcal/mol (XGB는 모델 부분도 ≤ "
      f"{k['xgb_model_part_tol_kcal']:g}).")
    a("- 입력 금지(FU2와 같음): σ, 계열·core 번호, X 이름.")
    a("")
    a("### 3-4. 짝·분할·학습 단위")
    a(f"- 분할: FU2 T1 split 파일(`d1_train.masks` arm C + 첫 안쪽 GroupKFold({k['n_inner']}) fold) fold 0–{NF - 1}. 학습 = "
      "학습 쪽 T_main, 시험 = 시험 쪽 P_sens ∪ A_main, 주 지표 = P_main. 검증 짝(대표 고르기, 13-1) = 첫 안쪽 fold"
      "(`inner_val`), 4 모델 공통(L-9).")
    a("- 한 rung에서 a·b가 사용 가능(토큰 + 입력표 행 + 유한한 H 9 + 유한한 182)하지 않은 짝은 모든 목록에서 뺀다 → 두 arm은 "
      "같은 `P_<rung>`을 쓴다(4절).")
    uc = D["unit_counts"]
    a(f"- 단위 이름 `L__<rung>__<arm>__<model>__f<k>`; 1단계 = {len(STAGE_RUNGS)} rung × {len(ARMS)} arm × "
      f"{len(MODELS4)} 모델 × {NF} fold = **{uc['stage1']} unit**(원자 {uc['stage1_atoms']} + XGB {uc['stage1_xgb']}), "
      f"GPU 학습 {uc['gpu_stage1']}회(원자 unit마다 튜닝 칸 수 + 최종 seed {uc['n_final_seeds']}개; 칸 수 "
      + ", ".join(f"{m} {n}" for m, n in uc["n_cells"].items()) + f"). 2단계 = {uc['stage2']} unit, GPU "
      f"{uc['gpu_stage2']}회.")
    a("- 정답이 바뀌었으므로 모든 rung·arm을 새로 학습한다. FU2·rev 9 결과는 재사용하지 않고 \"원래 라벨\" 비교로만 보고한다.")
    a("- 캐시 지문(명령서 10절): 입력표·짝 라벨·짝 목록·split 목록 sha256, 이 md의 sha256, grid, 코드 버전 `d1-ladder-2`와 "
      "코드 sha256, torch·CUDA·GPU(원자) / scikit-learn·xgboost(XGB) 버전. 다른 지문의 캐시를 만나면 exit 3.")
    a("")
    a("### 3-5. 참고 행 (학습 없음)")
    a("- **null:** ΔΔ = 0.")
    lp = D["line_pin"]
    a("- **LINE:** β·ΔH만 쓴다. β는 바깥 학습 쪽 T_main에서 위 규칙으로 구한다(`evaluate.line_beta`·`line_predict`, L-10이 "
      f"D1-HF Phase A 재현으로 확인). 두 함수의 소스 sha256 `{lp['sha256']}`(= HEAD 판; json `line_functions`)을 여기 "
      "고정한다. E1은 자기 소스가 이 값과 다르면 멈춘다(9절 7번).")
    a("")

    # 4
    a("## 4. 짝 집합")
    rows = []
    for name, s in PS.items():
        f = s.get("folds_T1") or {}
        rows.append([name, _n(s["n"]), _n((s.get("by_ref_kind") or {}).get("reference")),
                     _n((s.get("by_ref_kind") or {}).get("anchor")), _n(s.get("para")), _n(s.get("in_P_main")),
                     _n(s.get("in_P_main_para")),
                     " / ".join(_n((f.get(str(i)) or {}).get("test")) for i in range(NF)),
                     f"{_n((s.get('dropped') or {}).get('n'))} ({_n((s.get('dropped') or {}).get('n_used_in_T1'))})",
                     f"`{s['sha256']}`"])
    md.extend(_tab(["집합", "짝", "기준", "anchor", "para", "P_main", "P_main para", "시험 짝 fold 0–4", "빠진 짝 (T1에 쓰인 것)",
                    "sha256"], rows))
    a("")
    fc, dg = PSUM["fu2_check"], PSUM["dft_vs_g1"] or {}
    a(f"- P_g1 = FU2가 실제로 쓴 짝 목록: {'같음' if fc.get('ok') else '다름'} (FU2 {_n(fc.get('n_fu2_used'))}, P_g1 "
      f"{_n(fc.get('n_P_g1'))}). P_dft / P_g1 = {_n(dg.get('n_P_dft'))} / {_n(dg.get('n_P_g1'))} = {_f(dg.get('ratio'), 4)} "
      f"(하한 {dg.get('limit')}).")
    a("- 판정(13-2)은 그 rung의 자기 집합(`P_dft`, `P_g1`), rung 사이 비교(13-4)는 `P_01`(2단계에서 `P_012`)에서 한다.")
    a("- 학습 코드 규칙의 짝 목록 sha256(지문 `pair_list`와 비교할 값): "
      + ", ".join(f"{n} `{s['pair_list_sha256_trainer_rule'][:16]}…` ({_n(s['n_in_T1_lists'])}짝)" for n, s in PS.items()
                  if n != "P_01") + ".")
    hc = PSUM.get("h_changed_by_rereferencing") or {}
    for r in STAGE_RUNGS:
        x = hc.get(r) or {}
        bv = x.get("by_value") or {}
        a(f"- H 재조립으로 바뀐 값({r}): 반응 {_n(x.get('n_reactions'))}개; "
          + ", ".join(f"{t} {_n((bv.get(t) or {}).get('n_changed'))}개(> 0.05: {_n((bv.get(t) or {}).get('n_gt_0_05'))})"
                      for t in CHANGED) + ".")
    a("")

    # 5
    a("## 5. 판정 규칙 (명령서 13절, 고정)")
    a("")
    a("### 5-1. 대표 (REP)")
    a("- rung × arm × 타깃마다 4 모델 중 **검증 kcal MAE**(바깥 fold 5개 평균)가 가장 낮은 모델. 동점이면 AW → GNN → TF → XGB "
      "순서의 앞 것. 시험 점수로 고르지 않는다.")
    a("- ENS4 = 4 모델 시험 예측의 같은 비중 평균(조정값 없음). 보조로 쓴다.")
    a("- **고르기의 비대칭(학습 전 기록, `train_atoms.py` MA-6n):** 원자 모델의 검증 kcal MAE는 early stopping과 칸 고르기에 "
      "쓴 바로 그 `inner_val` 짝에서의 best epoch 값이라 낙관적이다. XGB의 값은 안쪽 fold 4개 평균으로 고른 칸의 fold 0 "
      "점수다. 명령서 4-3이 FU2 early stopping을 고정하므로 코드는 바꾸지 않고, REP가 원자 모델 쪽으로 기울 수 있음을 "
      "결과 해석에 적는다. 원문: “" + NOTES["ma6n"] + "”")
    a("")
    a(f"### 5-2. 단계 2 판정 (주 판정, rung × arm; 1단계는 {len(STAGE_RUNGS) * len(ARMS)}개)")
    r2 = STAGE2_RULE
    a(f"- REP 기준. **oi와 cpcm 모두** para 짝 MAE ≤ {r2['para_mae_max']} kcal/mol, 꺾임 민감도 ≥ "
      f"{r2['bend_sensitivity_min']}, 위양성률 ≤ {r2['false_positive_rate_max']}이면 통과.")
    a(f"- 집합은 그 rung의 자기 집합, 점 추정으로 판정한다. CI 상한까지 ≤ {r2['para_mae_max']}이면 \"확실히 통과\"로 따로 "
      "표시한다.")
    a("- **para 짝 MAE:** 그 rung 자기 집합의 시험 짝(fold 5개 합) 중 P_main 기준 para 짝(`set_mask` P_main, pos p)의 MAE. "
      f"CI = core 단위 재추출 {DP.N_BOOT:,}회, seed {DP.SEED}(`boot_diff`, 비교 상대 오차 0).")
    a("- **꺾임 민감도·위양성률이 계산되는 계열(학습 전 고정):** `d1_eval.bend_rates`를 **계열 제한 없이**(fams=None) 쓴다. "
      "계열은 FU2 `bends_tab`이 그 rung 자기 집합의 시험 짝 중 P_main 기준 para 짝(σ 있음)으로 만드는 계열 전부이고, 점이 "
      f"{DP.MIN_X_FAM}개 이상(H 포함)인 것만 남는다. 꺾임 판정은 `hammett_eval.calls`로 DFT 정답(`F_pairs_cr2.csv` dd)과 "
      "REP 예측에 같은 방식으로 한다. 민감도 = DFT에서 꺾인 계열 중 예측도 꺾인 비율, 위양성률 = DFT에서 곧은 계열 중 예측이 "
      "꺾인 비율(Clopper–Pearson 95 % CI 함께). DFT 쪽 분모가 0이라 값이 없으면(NaN) 그 기준은 **미달**로 본다(rev 9 "
      "`d1_eval.stages`와 같음). hidden_V·hidden_bend 계열만의 비율은 설명용이다.")
    a("- 근거(rev 9 PREREG 5-3, 그대로 인용): 숨은 꺾임 크기 중앙값이 궤도 0.47, 용매 0.45 kcal/mol이다. 모의실험(σp 18점)에서 "
      "잡음 SD ≤ 0.35 × 꺾임 크기 → SD ≤ 약 0.16 → MAE ≈ 0.8 × SD ≤ 약 0.13.")
    a("- oi·cpcm 정답은 Q·V에서 바뀌지 않는다(Q-b).")
    a("- 보조 판정: ENS4에 같은 기준. 개별 모델의 통과는 **설명용**(가장 좋은 것을 고르면 사후 선택이 된다). 단계 0·1(rev 9 "
      "기준)과 9 타깃 전부의 수치도 설명용으로 함께 적는다.")
    a(f"  - **단계 1(rev 9):** P_main 전체 위치의 계열 R²_fam 중앙값 ≥ {DP.STAGE1_R2}. 설명용.")
    a("  - **단계 0(rev 9)은 계산할 수 없다(학습 전 고정):** rev 9 단계 0은 대표 모델(arm B)이 null보다 MAE가 낮고 arm A 대비 "
      "Holm 검정이 유의할 것이다(`d1_eval.stages`). LADDER에는 arm A가 없다(FU2 E-4도 단계 1·2만 보고). 대신 설명용으로 "
      "**\"null보다 나음\"** = REP의 P_main 전체 위치 MAE < null MAE(점 추정)를 9 타깃마다 적는다.")
    a("")
    a("### 5-3. 해석 규칙 (위에서부터 처음 맞는 것 하나)")
    md.extend(_tab(["#", "조건", "결론 문장"], [
        ["a", "g2·S 또는 g2·SH 통과", "배포 가능한 입력(SMILES → xTB)으로 목표에 도달했다"],
        ["b", "g2 둘 다 실패, dft·S 또는 dft·SH 통과", "모델은 목표에 닿을 수 있으나 xTB 구조의 정밀도가 병목이다. 구조를 DFT "
                                                   "수준으로 다듬는 경로를 시험할 근거가 있다"],
        ["c", "dft 둘 다 실패", "이 입력과 이 모델 네 개로는 구조가 정답이어도 목표에 닿지 않는다"]]))
    a("")
    a("- 1단계(E1)에서는 dft 부분만 적용해 \"dft 통과 / 실패\"만 적는다. g1 결과는 이 표에 넣지 않는다(\"xTB로 바꾸면 잃는 몫\"의 "
      "설명용).")
    a("")
    a("### 5-4. 설명용 비교 (보정 없음, FU2 `compare`, d = MAE(상대) − MAE(기준))")
    a("- 같은 rung·모델에서 SH 대 S (HF가 더하는 몫)")
    a("- 같은 arm·모델에서 rung 사이 (P_01, 2단계에서 P_012)")
    a("- REP 대 LINE (모델이 검량선 위에 더하는 몫)")
    a("")

    # 6
    a("## 6. 지표 (명령서 11절, E1)")
    a("- 평가 함수는 FU2 Phase E 코드(`fu2/evaluate.py`: `bends_tab`, `compare`)와 그것이 부르는 `d1_eval.boot_diff`·"
      "`clopper`·`family_eval`, `hammett_eval.calls`를 그대로 쓴다(FU2 Phase E가 d1-pair에 commit된 뒤 merge).")
    a("- `results/E1_metrics.csv`: rung × arm × 모델(AW, GNN, TF, XGB, REP, ENS4, LINE, null) × 타깃 × 집합(자기 집합, P_01) × "
      "위치(all, p, m, o, direct) → n, MAE, RMSE, para MAE의 95 % CI, R²_fam 중앙값, ρ 오차. para MAE의 CI = core 단위 "
      f"재추출 {DP.N_BOOT:,}회, seed {k['bootstrap_seed']}, `boot_diff`에 비교 상대 오차 0을 넣어 MAE 자체의 CI.")
    a("- `results/E1_bends.csv`, `E1_bend_rates.json`: 꺾임 민감도·위양성률(Clopper–Pearson).")
    a("- `results/E1_goal.csv`: 5절 판정. `results/E1_compare.csv`: 5-4 비교.")
    a("- **라벨 교정 효과(설명용)** `results/E1_label_effect.csv`: g1·S의 새 라벨 결과 대 FU2·rev 9 원래 라벨 결과(d1·d2·barrier "
      "MAE, 계열 R²_fam, 꺾임 판정 수), DFT 정답 자체의 꺾임 판정 수 변화(`hammett_eval.calls`, 원래 대 새 라벨).")
    a("- 그림 `figL1`–`figL3`의 1단계 판. 예측 `results/L_predictions.parquet`(pair_id, ref_kind, pos, fold, rung, arm, model, "
      "target, y, yhat, base, beta, yhat_seed_sd), 검증 `results/L_val.csv`.")
    a("")

    # 7
    a("## 7. 이미 본 결과 (원래 라벨)")
    a("")
    a(f"### 7-1. D1-HF Phase A — LINE(정답 ΔΔ = β × HF ΔΔ, 절편 없음, β는 학습 fold), P_main {_n(HF['n_pairs_P_main'])}짝")
    a(f"`{HF['file']}` sha256 `{HF['sha256']}`. 사전 등록되지 않은 설명용 결과다(A_report: \"{HF.get('note')}\").")
    a("")
    hs = HF["summary"]
    rows = []
    for t in H9:
        x0, x1 = (hs.get("dft") or {}).get(t) or {}, (hs.get("g1") or {}).get(t) or {}
        rows.append([t, _n(x0.get("n_para")), _f(x0.get("mae_para")), _f(x1.get("mae_para")), _f(x0.get("null_mae_para")),
                     _f(x0.get("fu2_gnn_ag_mae_para")), _f(x0.get("beta_mean")), _f(x1.get("beta_mean"))])
    md.extend(_tab(["타깃", "para 짝", "LINE dft para MAE", "LINE g1 para MAE", "null para", "FU2 GNN-A+G para (같은 짝)",
                    "β dft", "β g1"], rows))
    a("")
    a("### 7-2. FU2 g1·S (원래 라벨)")
    a(f"- Phase M: `{FU['m_report']}` sha256 `{FU['m_report_sha256']}` ({_n(FU['m_n_units'])} unit, 예측 sha256 "
      f"`{FU['m_predictions_sha256']}`). A+G unit이 고른 칸:")
    rows = []
    for m in ATOM_MODELS:
        rows.append([f"{m}-A+G"] + [f"{_cell((FU['units_A_G'].get(f'{m}-A+G f{i}') or {}).get('cell'))} · ep "
                                     f"{(FU['units_A_G'].get(f'{m}-A+G f{i}') or {}).get('best_epoch')}" for i in range(NF)])
    md.extend(_tab(["구성"] + [f"fold {i}" for i in range(NF)], rows))
    a("")
    if FU.get("e_present"):
        ec = FU["e_check"]
        pv = FU.get("e_primary_verdict") or {}
        a(f"- Phase E: `{FU['e_dir']}` (E_check ok, gate {ec.get('n_gates')}개 실패 {ec.get('n_fail')}, {ec.get('date')}, "
          f"git {str(ec.get('git_head'))[:10]}, evaluate.py `{str(ec.get('evaluate_sha256'))[:16]}…`). 주 비교 "
          f"(ADD2 4절): {pv.get('n_better')}개 타깃에서 원자 대표가 C-XGB보다 나음 → \"{pv.get('verdict_ko')}\". "
          f"P_main 시험 짝 {_n(ec.get('n_P_main_test_pairs'))}, core {_n(ec.get('n_P_main_cores'))}.")
        gc = FU.get("e_files_git_counts") or {}
        n_e = sum(gc.values())
        a(f"- E-0이 해시한 `M_predictions.parquet` sha256 = 이 브랜치 `M_report.json` `predictions_sha256`"
          f"(`{str(FU['m_predictions_sha256'])[:16]}…`)임을 확인했다. E_* 파일 {n_e}개의 git 상태(그 worktree): "
          + ", ".join(f"{GIT_KO.get(s, s)} {n}개" for s, n in gc.items())
          + (" — **commit 전** 파일의 수치다(d1-pair에 commit된 뒤 merge, 6절)." if gc.get("untracked") or gc.get("modified")
             else "."))
        a("")
        nag = FU.get("e_representative_not_A_G") or {}
        rep_note = ("모든 타깃이 A+G" if not nag else "A+G가 아닌 타깃: " + ", ".join(f"{t} {m}" for t, m in nag.items()))
        a(f"P_main para 짝 MAE (kcal/mol). REP = FU2 ADD2 3절 규칙의 대표(`E_representative.json`: {rep_note}):")
        a("")
        rows = []
        for m in FU2_MODELS_MD:
            mm = (FU["metrics_P_main"].get(m) or {}).get("p") or {}
            rows.append([m] + [_f((mm.get(t) or {}).get("mae")) for t in H9])
        md.extend(_tab(["모델"] + H9, rows))
        a("")
        a("oi·cpcm의 단계 2 수치(설명용, 원래 라벨, g1은 5-3 표에 넣지 않는 rung이다):")
        a("")
        rows = []
        for m in FU2_MODELS_MD:
            dsc = FU["stage2_rule_descriptive"].get(m) or {}
            rows.append([m] + [x for t in ("oi", "cpcm") for x in (_f((dsc.get(t) or {}).get("para_mae")),
                                                                    _f((dsc.get(t) or {}).get("sens"), 2),
                                                                    _f((dsc.get(t) or {}).get("fpr"), 2))]
                        + ["통과" if dsc.get("both") else "미달"])
        md.extend(_tab(["모델", "oi para MAE", "oi 민감도", "oi 위양성률", "cpcm para MAE", "cpcm 민감도", "cpcm 위양성률",
                        f"기준({STAGE2_RULE['para_mae_max']} / {STAGE2_RULE['bend_sensitivity_min']} / "
                        f"{STAGE2_RULE['false_positive_rate_max']})"], rows))
        a("")
        a("(민감도·위양성률은 FU2 `E_bend_rates.json`의 `all` 계열 = 5-2절과 같은 모집단.)")
    else:
        a(f"- Phase E 결과 없음(찾은 곳: {', '.join(FU['e_searched'])}; 쓰지 않은 이유: "
          + "; ".join(f"{x['dir']}: {x['why']}" for x in FU.get("e_rejected") or []) + "). 시험 점수는 이 문서에 없다.")
    a("")
    a("### 7-3. 검토자의 사후 탐색 (사전 등록 아님)")
    for it in REVIEWER_POSTHOC["items"]:
        a(f"- {it['what']}: {it['target']} {it['subset']} MAE {it['mae_kcal']} kcal/mol")
    a(f"- 출처: {REVIEWER_POSTHOC['source']}. 원래 라벨·원래 H 기준이다.")
    a("")
    a("### 7-4. 검토자의 기준 구조 HF 사전 점검 — 2-3절.")
    a("")
    a("### 7-5. Phase C에서 나온 수치")
    l10 = C.get("l10_tol")
    a(f"- L-10({C['checks'].get('L-10')}): 원래 라벨·원래 H의 LINE이 7-1 표(dft·g1 para MAE)를 "
      f"{(format(l10, 'g') if isinstance(l10, (int, float)) else '–')} 안에서 재현했다(C_tests.json `tol.L10`). "
      f"L-6({C['checks'].get('L-6')}): g1·S 원래 라벨의 입력 텐서·y·s_y와 XGB fold 0 예측이 FU2·rev 9 unit과 같다. "
      "smoke: 2 epoch 학습의 검증 짝 MAE(새 정답)는 `C_smoke_atoms.json`·`C_smoke_xgb.json`에 있다. 시험 짝 점수는 없다.")
    a("")

    # 8
    a("## 8. Phase C 결과")
    a(f"`{C['file']}` sha256 `{C['sha256']}` (commit {C.get('last_commit')}), tests `{C.get('tests_version')}`, merge "
      f"{C.get('merged')}: all_pass {C.get('all_pass')}, 범위 \"{C.get('all_pass_scope')}\", stage_verified "
      f"{C.get('stage_verified')}, g2_verified {C.get('g2_verified')}, 정답 `{_disp(C['labels'].get('path') or '')}`.")
    a("")
    md.extend(_tab(CHECKS, [[C["checks"].get(x) for x in CHECKS]]))
    a("")
    parts = C.get("parts") or {}
    a("- 검사 job: " + "; ".join(f"{p} job {v.get('slurm_job')} ({v.get('host')}, git {str(v.get('git_head'))[:8]}, "
                                 f"{(v.get('versions') or {}).get('gpu') or (v.get('versions') or {}).get('xgboost') or ''})"
                                 for p, v in parts.items()) + ".")
    ge = C.get("gpu_estimate") or {}
    a(f"- 예상 GPU 시간(FU2 unit 기록): 1단계 {_f(ge.get('stage1_gpu_hours'), 1)} GPU-시간 (unit당 "
      + ", ".join(f"{m} {_f(h, 2)}" for m, h in (ge.get("per_model_unit_hours") or {}).items()) + ").")
    a("")

    # 9
    a("## 9. 계획과 다르게 한 점·해석 (지금까지)")
    s02, s03, f0, vd = DEV["start_0_2"], DEV["start_0_3"], DEV["f0_f1"], DEV["v_decisions"]
    nci, nc = s02.get("n_core_identical"), s02.get("n_core")
    a(f"1. **시작 조건 0-2:** `analysis/d1_pair`가 `{L.COMMITS['d1pair']}`과 다른 파일은 "
      f"{', '.join('`' + x + '`' for x in (s02.get('changed') or []))}이고 삭제는 {_n(s02.get('deletions'))}개, FU2 핵심 코드 "
      f"{_n(nc)}개 중 {_n(nci)}개가 같다(`start_report.json` `0-2`). 결정 원문: \"{s02.get('decision')}\".")
    rd = s03.get("results_diff_vs_6b9fbeb2")
    rd_txt = "없음(바이트 동일)" if rd == [] else f"{rd}"
    if s03.get("merge_commit"):
        why = ("d1-g2 쪽 부모 `{}`에 D1-HF 최종 결과(`{}`)가 없어서".format(s03.get("first_parent"), s03.get("d1hf_commit"))
               if s03.get("first_parent_had_d1hf") is False else
               "d1-g2 쪽 부모 `{}`의 D1-HF 포함 여부: {}.".format(s03.get("first_parent"), s03.get("first_parent_had_d1hf")))
        a(f"2. **시작 조건 0-3:** {why} d1-hf(`{s03.get('d1hf_commit')}`)를 d1-ladder에 merge했다(merge commit "
          f"`{s03.get('merge_commit')}`, \"{s03.get('merge_subject')}\"). `analysis/d1_hf/results`와 "
          f"`{s03.get('d1hf_commit')}`의 차이: {rd_txt}; `labels_d1_hf.parquet` sha256 "
          f"`{str(s03.get('labels_d1_hf_sha256'))[:16]}…`(`start_report.json` `0-3`).")
    else:
        a(f"2. **시작 조건 0-3:** HEAD 이력에서 `{s03.get('d1hf_commit')}`를 merge한 commit을 찾지 못했다. "
          f"`analysis/d1_hf/results`와의 차이: {rd_txt}(`start_report.json` `0-3`).")
    thr = f0.get("threshold")
    thr_t = _pct(thr, 0)

    def cmp_t(flag):
        return "–" if flag is None else ("> " + thr_t if flag else "≤ " + thr_t)
    rc_, mc_ = f0.get("report_counts_by_class") or {}, f0.get("decision_md_counts_by_class") or {}
    a(f"3. **F0 F-1 (`F0_DECISION.md`, sha256 `{str(f0.get('decision_md_sha256'))[:16]}…`):** 전역 182가 빠진 반응 "
      f"{_n(f0.get('n_F1'))}/{_n(f0.get('n_g0_done'))} = {_pct(f0.get('rate_F1'), 2)} {cmp_t(f0.get('F1_over'))}. 명령서 "
      f"16절이 F-1과 B6을 따로 적으므로 F-1에서 B6 실패({_n(f0.get('n_B6_only'))}개, {_pct(f0.get('rate_B6'), 2)} "
      f"{cmp_t(f0.get('B6_over'))}: B6 자기 gate {'실패' if f0.get('B6_over') else '통과'})를 빼고 센다: "
      f"{_n(f0.get('n_without_B6_only'))}/{_n(f0.get('n_g0_done'))} = {_pct(f0.get('rate_without_B6'), 2)} "
      f"{cmp_t(f0.get('without_B6_over'))} → "
      + ("진행(`inputs.py build dft --accept-f0-stop`). 빠진 반응은 dft rung의 사용 가능 반응에서 제외된다."
         if f0.get("without_B6_over") is False else "이 해석으로도 기준을 넘는다(진행 근거 없음)."))
    if not f0.get("decision_md_counts_equal_report"):
        a(f"   - `F0_DECISION.md` 본문의 분류 수(engine gate {_n(mc_.get('engine_gate'))}, parse:fA "
          f"{_n(mc_.get('parse_fA'))}, B6 {_n(mc_.get('B6'))})는 `F0_report.json` `F1.classes`(engine gate "
          f"{_n(rc_.get('engine_gate'))}, parse:fA {_n(rc_.get('parse_fA'))}, B6 {_n(rc_.get('B6'))})와 다르다. B6를 뺀 합계는 "
          f"md {_n((mc_.get('engine_gate') or 0) + (mc_.get('parse_fA') or 0))}, 보고서 "
          f"{_n((rc_.get('engine_gate') or 0) + (rc_.get('parse_fA') or 0))}이다. 이 문서는 보고서 수치를 쓴다.")
    a("4. **Phase V 구현 결정(V_report.json `decisions`; 설명 뒤 원문):**")
    tol = DEV["v_dft_sensitivity"].get("d1_tol_cm1")
    tol_t = f"{tol:g} cm⁻¹" if isinstance(tol, (int, float)) else "V_report `dft.note_d1_tol`의 값"
    for key, ko in (("V7_rr", "randomize-and-relax 시작마다 고정 seed를 넣었다(D1/autodE는 seed 없음). 재현성을 위한 이탈이며 "
                              "알고리즘·인자는 같다."),
                    ("V8_xtb", "입체 서명을 확인할 수 없는 분자는 실패로 두고 V 기준을 쓰지 않는다(명령서 V-3 '입체가 키와 다르면 "
                               "버림'을 적용하는 방법)."),
                    ("V10_imag", "명령서 V-3 그대로: 다시 최적화한 뒤 허수 진동수가 하나라도 남으면 실패. D1 TS 단계 허용치"
                                 f"({tol_t})를 썼을 때의 실패율은 민감도로만 보고한다(실패율 "
                                 f"{_pct(DEV['v_dft_sensitivity'].get('rate'))}, 허용치 적용 시 "
                                 f"{_pct(DEV['v_dft_sensitivity'].get('rate_if_d1_tol'))})."),
                    ("V12_repro", "시험 묶음의 기준 구조 라벨·HF 입력을 그대로 다시 계산해 재현됨을 확인하는 검사를 더했다."),
                    ("V17_name", "명령서 8절의 `refs.py`(Q·V)를 `refs_q.py`(Q)와 `refs_v.py`(V)로 나눴다.")):
        if vd.get(key):
            a(f"   - {key}: {ko} (원문: \"{vd[key]}\")")
    cpu = (C.get("parts") or {}).get("cpu") or {}
    cv = cpu.get("versions") or {}
    env_name = Path(str(cpu.get("prefix"))).name if cpu.get("prefix") else "–"
    a(f"5. **XGB 환경:** `{env_name}`(scikit-learn {cv.get('scikit-learn', '–')}, xgboost {cv.get('xgboost', '–')}; "
      "C_tests.json `parts.cpu`). 이 환경에는 torch가 없어 `train_xgb.py`가 `train_atoms.py`의 공통 규칙을 옮겨 쓰고, collect가 "
      "두 쪽의 시험·검증 짝 목록을 대조한다. 원문(XG-1): \"" + str(NOTES["xg"].get("XG-1")) + "\"")
    a("6. **대표 고르기의 비대칭(MA-6n):** 5-1절.")
    lp = D["line_pin"]
    a(f"7. **`evaluate.py`의 commit (명령서 8절 Phase C 코드):** ADD3는 `evaluate.py`가 commit되어 있기를 요구하지 않는다(지금 "
      f"git 상태: {GIT_KO.get(lp['evaluate_git_state'], lp['evaluate_git_state'])}, sha256 `{lp['evaluate_sha256']}`). "
      "E1 코드를 학습과 따로 쓰고 있기 때문이다. "
      "`C_tests.json`은 `evaluate.py`를 고정하지 않으므로, L-10이 시험한 LINE 함수(`line_beta`·`line_predict`)의 소스 sha256 "
      f"`{lp['sha256']}`을 json `line_functions`에 적었다. 이 값은 HEAD 판과 같다. E1은 자기 소스가 이 값과 다르면 멈추고, 쓴 "
      "`evaluate.py`의 sha256을 `E1_report.json`에 적는다.")
    if FU.get("e_present"):
        gc = FU.get("e_files_git_counts") or {}
        if gc.get("untracked") or gc.get("modified"):
            a(f"8. **FU2 Phase E 수치(7-2절)는 commit 전 파일이다:** `{FU['e_dir']}`의 E_* 파일 git 상태 "
              + ", ".join(f"{GIT_KO.get(s, s)} {n}개" for s, n in gc.items()) + ". 설명용으로만 옮겼고 판정에 쓰지 않는다.")
    a("")
    P_dec = (PSUM.get("decisions") or {})
    a("명령서가 정하지 않아 정한 값은 학습 코드 docstring의 결정 "
      f"{_id_span(NOTES['ma'])}, {_id_span(NOTES['xg'])}, `inputs.py` {_id_span(P_dec)}, `refs_q.py` "
      f"{_id_span(Q.get('decisions'))}, `refs_v.py` {_id_span(V.get('decisions'))}, `tests/run_tests.py` "
      f"{_id_span(C.get('decisions'))}에 있다. json `trainer_decisions`, `q.decisions`, `v.decisions`, `p_report.decisions`, "
      "`c_tests.decisions`에 그대로 옮겼다.")
    a("")

    # 10
    a("## 10. 2단계(g2) 보완 규칙 (이 문서는 바꾸지 않는다)")
    a("- **이 md는 바이트 하나도 바꾸지 않는다:** 1단계 unit 지문의 `add3`가 이 md의 sha256이고 collect가 그것을 다시 "
      "비교하기 때문이다. 2단계 학습 전에 보완판 `results/PREREG_D1PAIR_ADD3_G2.md`를 새 파일로 쓴다.")
    a("- **commit된 이 json에 허용되는 수정은 하나뿐이다:** 아래 g2 파일을 `inputs_sha256`·`inputs_roles`에 **새 항목으로 "
      "더하고**, 새 최상위 키 `supplement`(보완판 md 경로·sha256·commit)를 더한다. 이미 있는 항목과 키는 그대로 둔다(1단계 "
      "process의 gate는 g2 열에만 속한 파일을 열지 않고, 여러 rung에 걸친 `P_012.csv`는 바뀌지 않는 한 통과한다). 더할 항목:")
    for pth, role in stage2_supplement_paths():
        a(f"  - `{_disp(pth)}` ({role})")
    a("- 학습 코드의 gate는 이 md와 json만 읽고 보완판 md는 확인하지 않는다. 보완판을 만드는 명령(`prereg.py`에 추가할 "
      "`supplement`, 아직 없음)이 보완판의 commit을 확인하고, E2가 보완판 commit·sha256을 다시 확인해야 한다.")
    a("- 시작 조건(명령서 12절): D1-G2 `X182_d1_g2.parquet`, `$G2/fu2/` 토큰, `labels_d1_hf_g2.parquet`가 각 gate를 통과하고 "
      "commit됨, `origin/d1-g2` merge. 그 뒤 P(g2 입력표, H 재조립, P_g2, P_012) → C(g2 검사, `g2_verified`) → 보완판 → "
      f"M2({D['unit_counts']['stage2']} unit) → E2.")
    a("- g2 H 재조립의 constrained dipole 기준: 명령서 Q-2 그대로 '묶지 않음'(고정 이면각 상태를 키에 붙임; `inputs.py build g2 "
      "--g2-q-keys`). 이 키는 D1(DFT) 기준 구조로 정해지므로 g2 H에 DFT 정보가 일부 들어간다(P20). 보완판에서 확정·기록한다.")
    a("- g2는 G2가 라벨과 다른 TS를 찾은 짝도 그대로 학습·평가하고, D1-G2 Phase C의 \"같은 TS\" 판정에 따른 나눔은 설명용으로만 "
      "보고한다. 적용 범위(P_main 시험 짝 중 g2로 예측 가능한 비율)를 보고한다.")
    a("")

    # 11
    a("## 11. 멈춤 규칙 (명령서 16절)")
    a("- **A:** ADD3 commit 전 학습 시도(`gate_add3`가 exit 3).")
    a("- **M:** gate M-1′ 실패; 캐시 지문 불일치; 튜닝·β에 시험 짝 사용(L-2·L-8 assert).")
    a("- **E:** 같은 집합에서 비교하는 모델 사이 시험 짝 목록 불일치; unit 지문의 `add3`·`code_sha256`이 이 문서(json "
      "`code.fingerprint_keys`)와 다름.")
    a("- **원칙:** 결과를 본 뒤 Q·V 규칙, rung·arm·모델·grid·β 규칙·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.")
    a("")
    return "\n".join(md) + "\n"


# ================================================================ json
def make_json(D, md_sha):
    GI = D["inputs"]
    return _jsonable(dict(
        date=D["date"], written=D["written"], code_version=CODE_VERSION, stage=STAGE, rungs=list(STAGE_RUNGS),
        spec=_disp(SPEC), spec_sha256=D["spec_sha256"], spec_committed=_committed(SPEC),
        base_prereg_commit=D["prev"]["PREREG_D1PAIR.md"], add1_commit=D["prev"]["PREREG_D1PAIR_ADD1.md"],
        add2_commit=D["prev"]["PREREG_D1PAIR_ADD2.md"],
        md=_disp(ADD3_MD), md_sha256=md_sha,
        code_commit=D["git_head"],
        code_commit_clean=dict(
            gated_and_phase_c=all(_committed(p) for p in {**C_CODE, **CODE_ATOMS, **CODE_XGB}.values()) and _committed(C_TESTS),
            not_committed_not_gated=sorted(k for k, p in LADDER_CODE.items() if not _committed(p)),
            rule="the gated code, the 5 Phase C code files and C_tests.json must equal HEAD (checked before writing); "
                 "other d1_ladder files are recorded with their git state (code.ladder_git_state)"),
        line_functions=D["line_pin"],
        inputs_sha256={p: x["sha256"] for p, x in GI.items()},
        inputs_roles={p: dict(role=x["role"], column=x["column"]) for p, x in GI.items()},
        gate=dict(code=["analysis/d1_ladder/train_atoms.py gate_add3 (lines 796-832)",
                        "analysis/d1_ladder/train_xgb.py gate_add3 (lines 851-886)"],
                  keys_read=["md_sha256", "inputs_sha256"],
                  need_stage1=sorted(need_paths()),
                  rule="both files tracked + unchanged since HEAD; sha256(md) = md_sha256; need ⊆ inputs_sha256; every "
                       "listed file unchanged (files only in other rungs' columns are skipped, MA-15)"),
        not_gated_sha256=D["not_gated"], code=D["code"],
        labels=dict(path=str(CR2), sha256=_sha(CR2), original=str(ORIG), original_sha256=_sha(ORIG),
                    rule="y = dd_<t> of F_pairs_cr2.csv; the original labels only for comparison (L-6, L-10, E1 label effect)"),
        q=D["q"], v=D["v"], cr2_totals=D["cr2_totals"], pair_changes=D["pair_changes"],
        ladder=dict(
            rungs={"dft": dict(role="upper bound (label TS), not deployable", ts="label TS (D1-HF geom=dft EDA input coords)",
                               reactants="per-reaction label references (dft frag*_rel inputs)",
                               hess="ORCA ! XTB2 ALPB(water) Freq at the label TS (G0)", tokens="$LAD/fu2/tokens fu2-tokens-3-dft",
                               x182="$LAD/X182_d1_dft.parquet", H="labels_d1_hf geom=dft + V HF single points (refs_cr2)",
                               table=_ab(IN.table_path("dft"))),
                   "g1": dict(role="what xTB structures lose", ts="G1_ROOT_D1F/<rxn>/ts.xyz", reactants="G1 rel1/rel2.xyz",
                              hess="existing", tokens="$D1PAIR_SCRATCH/fu2/tokens fu2-tokens-3",
                              x182="$D1PAIR_SCRATCH/X182_d1.parquet", H="labels_d1_hf geom=g1",
                              table=_ab(IN.table_path("g1"))),
                   "g2": dict(role="deployable (SMILES -> xTB)", stage=2, ts="$G2/g2_final/<rxn>/ts.xyz",
                              reactants="G2 rel1/rel2.xyz", hess="D1-G2", tokens="D1-G2 Phase F",
                              x182="$G2/X182_d1_g2.parquet", H="labels_d1_hf_g2.parquet")},
            structure_features="per-reaction references in every rung (existing pipeline)",
            h_reassembly=dict(
                values=H9, changed=list(CHANGED),
                dft="final Q / V reference HF energy (refs_cr2.parquet e_hf_eh: own -> labels_d1_hf dft row, Q -> source "
                    "row, V -> V HF single point)",
                g1_g2="per Q-2 key the lowest rung HF e_frag{k}_rel_eh among the rung's references (no threshold, ties "
                      "smallest job_id)",
                assembly="D1-HF build.py parse_rxn with only the reference energies replaced; gate P-a (original references "
                         "reproduce labels_d1_hf.parquet within 1e-9); EDA channels verbatim"),
            rung_isolation="a rung's training / evaluation reads only that rung's column files (L-4)",
            arms=dict(S=dict(atoms="atom tokens + global 182 (FU2 variant A+G)", xgb="182 (rev 9 arm C)", target="dd",
                             prediction="model output"),
                      SH=dict(atoms="atom tokens + global (182 + H 9)", xgb="182 + H 9", target="r = dd - beta * dH",
                              prediction="beta * dH + model output")),
            sh_baseline=dict(beta="beta_t = sum(dH_t dy_t) / sum(dH_t^2), no intercept", fit_pairs="the fit's own training "
                             "pairs only (tuning: inner train; final: outer train T_main = reference + anchor pairs)",
                             c_ghost_beta=0.0),
            sh_training=dict(output="residual / SD(train residual), ddof 0, no mean removed",
                             loss=f"mean |rhat - r| (standardised, 10 targets) + {F.LAMBDA} * mean |yhat_bar - sum_9 yhat| / "
                                  "s_bar on the final prediction; s_bar = SD of the original dd_barrier (training pairs)",
                             metric="mean over the 9 primary targets of MAE(final yhat, y) / SD(original y, training pairs)",
                             flip="(a, b) <-> (b, a) flips residual and baseline together (fu2 epoch_batches)",
                             xgb="per target r_t; inner CV score = mean |beta dH + g - y| (kcal)"),
            models=dict(atoms=list(ATOM_MODELS), xgb="XGB", variant="A+G",
                        grid=dict(D["grids"]["atoms"], XGB=D["grids"]["xgb_cells"]),
                        xgb_grid_values=D["grids"]["xgb_values"], xgb_estimator=D["grids"]["xgb_estimator"],
                        grid_sources=dict(atoms=D["grids"]["atoms_source"], xgb=D["grids"]["xgb_source"]),
                        atoms_fu2_m2={k: ((_json(ADD2_JSON) or {}).get("models") or {}).get(k)
                                      for k in ("architecture", "antisymmetry", "token_columns", "forbidden_inputs")},
                        xgb_rule="1/2 [h(dx) - h(-dx)], trained both ways, inner GroupKFold(4) by core, per-target tuning "
                                 "(lowest mean inner MAE, ties grid order), one fit random_state 42"),
            training_constants=D["constants"],
            splits=dict(split="T1", folds=list(range(NF)), files=[_ab(IN.FU2_SPLITS / f"T1__f{i}.json") for i in range(NF)],
                        train="outer-train T_main", test="held-out P_sens | A_main", primary="P_main",
                        validation="first inner fold (inner_val), the same for the 4 models (L-9)",
                        usable_filter="pairs whose a or b is not usable in the rung (token + table row + finite H 9 + "
                                      "finite 182) dropped from every list; S and SH share P_<rung>"),
            forbidden_inputs=["sigma", "family / core ids", "X names"],
            units=dict(stage1=D["units"], n_stage1=len(D["units"]), counts=D["unit_counts"],
                       name="L__<rung>__<arm>__<model>__f<k>"),
            reference_rows=dict(null="dd = 0", LINE="beta * dH, beta from the outer-train T_main (evaluate.line_beta)"),
            fingerprint="input table, label file, pair list, split lists sha256, ADD3 md sha256, grid, code d1-ladder-2 + "
                        "code sha256, torch / CUDA / GPU (atoms) or scikit-learn / xgboost versions (XGB); mismatch -> exit 3"),
        pair_sets=D["pair_sets"], p_report=D["p"],
        decision_rules=dict(
            representative=dict(rule="rung x arm x target: lowest validation kcal MAE (mean over the 5 outer folds) among "
                                     "AW, GNN, TF, XGB; ties AW -> GNN -> TF -> XGB; test scores never used",
                                source=dict(atoms="unit json val_kcal_mae (MA-6)", xgb="unit json val_kcal_mae (XG-7)"),
                                asymmetry_note_MA_6n=D["notes"]["ma6n"]),
            ENS4="equal-weight mean of the 4 models' test predictions (no tuning), secondary",
            stage2=dict(STAGE2_RULE, on="REP", set="the rung's own set (P_dft, P_g1; stage 2 P_g2)", estimate="point",
                        certain_pass=f"CI upper bound <= {STAGE2_RULE['para_mae_max']} marked separately",
                        n_rung_arms_stage1=len(STAGE_RUNGS) * len(ARMS),
                        para_mae=f"test pairs of the own set (5 folds pooled), set_mask P_main x pos p; CI = boot_diff "
                                 f"with a zero comparator, {DP.N_BOOT} core resamples, seed {DP.SEED}",
                        bend_population=dict(
                            function="d1_eval.bend_rates(Bd, 'T1', fams=None): no family filter (PR-15)",
                            families="FU2 bends_tab families of the own-set test pairs: in_P_main, reference, pos p, sigma "
                                     f"known, >= {DP.MIN_X_FAM} points incl. H",
                            truth="hammett_eval.calls on the F_pairs_cr2.csv dd (oi / cpcm = the original labels, Q-b) and "
                                  "on the REP prediction",
                            sensitivity="DFT-bent families called bent / DFT-bent", fpr="DFT-straight families called "
                                                                                       "bent / DFT-straight",
                            nan="zero DFT denominator -> NaN -> criterion not met (rev 9 d1_eval.stages)",
                            descriptive_only=["hidden_V", "hidden_bend"]),
                        secondary="ENS4 with the same rule", single_models="descriptive only",
                        also_descriptive=dict(
                            stage1=f"rev 9 stage 1: R2_fam median (P_main, all positions) >= {DP.STAGE1_R2}",
                            stage0="not computable (rev 9 stage 0 = B < null and Holm test vs arm A; LADDER has no arm A; "
                                   "FU2 E-4 precedent); substitute beats_null = REP P_main MAE < null P_main MAE (point) "
                                   "(PR-14)",
                            targets="all 9 primary targets")),
            rationale="rev 9 PREREG 5-3: hidden bend size median orbital 0.47, solvent 0.45 kcal/mol; simulation (18 sigma_p "
                      "points) noise SD <= 0.35 x bend size -> SD <= ~0.16 -> MAE ~ 0.8 x SD <= ~0.13",
            interpretation=[dict(id="a", condition="g2-S or g2-SH passes",
                                 conclusion="배포 가능한 입력(SMILES → xTB)으로 목표에 도달했다"),
                            dict(id="b", condition="both g2 fail, dft-S or dft-SH passes",
                                 conclusion="모델은 목표에 닿을 수 있으나 xTB 구조의 정밀도가 병목이다. 구조를 DFT 수준으로 다듬는 "
                                            "경로를 시험할 근거가 있다"),
                            dict(id="c", condition="both dft fail",
                                 conclusion="이 입력과 이 모델 네 개로는 구조가 정답이어도 목표에 닿지 않는다")],
            stage1="E1 applies only the dft part ('dft pass / fail'); g1 is descriptive (not in the table)",
            comparisons=dict(function="fu2/evaluate.py compare, d = MAE(other) - MAE(reference), uncorrected",
                             list=["SH vs S (same rung, model)", "rung vs rung (same arm, model; P_01, later P_012)",
                                   "REP vs LINE"])),
        metrics=dict(
            functions="fu2/evaluate.py bends_tab, compare; d1_eval.boot_diff / clopper / family_eval; hammett_eval.calls",
            E1_metrics=dict(file="results/E1_metrics.csv", rows="rung x arm x model (AW, GNN, TF, XGB, REP, ENS4, LINE, null) "
                            "x target x set (own, P_01) x position (all, p, m, o, direct)",
                            columns=["n", "MAE", "RMSE", "para MAE 95 % CI", "R2_fam median", "rho error"],
                            ci=dict(unit="core", n=2000, seed=DP.SEED, how="boot_diff with a zero-error comparator")),
            E1_bends=["results/E1_bends.csv", "results/E1_bend_rates.json (Clopper-Pearson)"],
            E1_goal="results/E1_goal.csv (section 13)", E1_compare="results/E1_compare.csv (13-4)",
            E1_label_effect="results/E1_label_effect.csv: g1-S new labels vs FU2 / rev 9 original (d1 d2 barrier MAE, "
                            "R2_fam, bend calls) + DFT truth bend calls original vs new (hammett_eval.calls)",
            figures=["figL1", "figL2", "figL3 (stage-1 versions)"],
            predictions="results/L_predictions.parquet (pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, "
                        "base, beta, yhat_seed_sd)"),
        seen_before_add3=dict(phase_M_units=D["n_units_before"], checked=D["written"],
                              phase_C_mae=["L-3 synthetic", "L-10 LINE with original labels / H (= D1-HF Phase A)",
                                           "smoke validation-pair MAE (2 epochs, new labels)"],
                              new_label_test_scores=False, d1hf_phase_a=D["hf_phase_a"], fu2_g1_S=D["fu2"],
                              reviewer_posthoc=REVIEWER_POSTHOC,
                              reviewer_hf_precheck=D["q"]["hf_precheck"]),
        c_tests=D["c_tests"],
        deviations=D["deviations"],
        trainer_decisions=dict(train_atoms=D["notes"]["ma"], train_xgb=D["notes"]["xg"]),
        stage2_plan=dict(supplement_md="results/PREREG_D1PAIR_ADD3_G2.md (new file)",
                         this_md="never modified (unit fingerprints carry its sha256; collect compares it)",
                         json="the only allowed edit of this committed json: new inputs_sha256 / inputs_roles entries "
                              "(supplement_entries) + a new top-level 'supplement' key; every existing entry and key "
                              "byte-identical (PR-5)",
                         supplement_entries={_ab(pth): role for pth, role in stage2_supplement_paths()},
                         supplement_md_check="not read by the trainers' gate_add3: the supplement command (not written yet) "
                                             "and E2 must check its commit and sha256",
                         command="prereg.py supplement (to be written for stage 2)",
                         g2_constrained_keys="spec Q-2 literal ('not grouped'; inputs.py build g2 --g2-q-keys), DFT-derived "
                                             "key suffix noted (P20); confirmed in the supplement"),
        requirements_checked=D["checked"], decisions=DECISIONS,
        env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
                 slurm_job=os.environ.get("SLURM_JOB_ID"), prefix=sys.prefix)))


# ================================================================ commands
def collect_content(v, p, c, checked):
    head = L.git("rev-parse", "HEAD").strip()
    n_units = len([e.name for e in os.scandir(UNITS) if e.name.startswith("L__")]) if UNITS.is_dir() else 0
    D = dict(date=time.strftime("%Y-%m-%d"), written=_now(), git_head=head, spec_sha256=_sha(SPEC),
             prev={k: _last_commit(pth) for k, pth in PREV.items()}, n_units_before=n_units, checked=checked)
    D["grids"] = grids()
    D["constants"] = constants()
    D["notes"] = trainer_notes()
    D["units"] = unit_names()
    D["unit_counts"] = unit_counts(D["grids"]["atoms"])
    D["line_pin"] = line_pin()
    D["q"] = q_summary(_json(Q_REPORT))
    D["v"] = v_summary(v)
    D["cr2_totals"] = cr2_totals()
    D["pair_changes"] = pair_changes()
    D["pair_sets"] = pair_sets(p)
    D["p"] = p_summary(p)
    D["c_tests"] = c_summary(c)
    D["hf_phase_a"] = hf_phase_a()
    D["fu2"] = fu2_results()
    D["deviations"] = deviations(v, p)
    D["code"] = code_records()
    D["inputs"] = gated_inputs(p)
    D["not_gated"] = not_gated_records()
    if not set(need_paths()) <= set(D["inputs"]):
        raise Refuse(3, [f"internal: need {sorted(need_paths())} not all in inputs_sha256"])
    return D


def _foreign_to(path, rungs):
    """= train_atoms._foreign_to (line 790-793) = train_xgb._foreign_to (845-848), MA-15."""
    return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)


def gate_mirror(rec, rungs, need):
    """the content conditions of train_atoms.gate_add3 (lines 812-826) = train_xgb.gate_add3 (866-880) for a process that
    trains `rungs` with `need` (cmd_run: label file + the rungs' input tables), on json record `rec` and the files now;
    the git conditions (both ADD3 files tracked and = HEAD) are the callers'. A stage-1 process of one rung verifies a
    subset of what rungs (dft, g1) together verify (its need and its non-foreign paths are subsets)."""
    bad = []
    if L.sha256(ADD3_MD) != rec.get("md_sha256"):
        bad.append(f"{ADD3_MD.name} sha256 differs from {ADD3_JSON.name} md_sha256")
    ins = rec.get("inputs_sha256") or {}
    lack = sorted(set(need) - set(ins))
    if lack:
        bad.append(f"{ADD3_JSON.name} inputs_sha256 lacks {lack}")
    verified, skipped = [], []
    for path, s in ins.items():
        if path not in need and _foreign_to(path, rungs):
            skipped.append(path)
            continue
        verified.append(path)
        if not Path(path).is_file() or L.sha256(path) != s:
            bad.append(f"input {path} changed or missing")
    return dict(ok=not bad, rungs=list(rungs), need=sorted(need), bad=bad, n_verified=len(verified),
                n_skipped=len(skipped), verified=verified, skipped=skipped)


def cmd_check(stage=1):
    """the gate_add3 conditions of a stage-<stage> process (train_atoms.py 796-832 / train_xgb.py 851-886), mirrored;
    stage 2 also the supplement md and the trainers' MA-16 / MA-17 run gates; exit 0 / 3."""
    stage = int(stage)
    rungs = STAGE_RUNGS if stage == 1 else STAGE2_RUNGS
    bad = []
    for p in (ADD3_MD, ADD3_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif not _tracked(p):
            bad.append(f"{p.name} not committed")
        elif not _clean(p):
            bad.append(f"{p.name} modified since its commit")
    n_listed = 0
    if not bad:
        rec = json.loads(ADD3_JSON.read_text(encoding="utf-8"))
        m = gate_mirror(rec, rungs, need_paths(rungs))
        bad += m["bad"]
        n_listed = m["n_verified"]
        if stage == 2:
            bad += supplement_md_problems(rec)
            bad += stage2_run_gates()
    head = _git_out("rev-parse", "HEAD")
    pushed = _git_rc("merge-base", "--is-ancestor", "HEAD", "origin/d1-ladder") == 0
    if bad:
        print(f"PREREG ADD3 check FAILED for stage {stage} (the trainers' gates would exit 3):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        return 3
    print(f"[check] ADD3 gate conditions pass for stage {stage} (rungs {', '.join(rungs)}; HEAD {head}; {n_listed} listed "
          f"files verified; HEAD in origin/d1-ladder: {pushed}; md sha256 {L.sha256(ADD3_MD)}"
          + (f"; supplement md sha256 {L.sha256(SUPP_MD)}" if stage == 2 else "") + ")", flush=True)
    return 0


def cmd_write():
    have = [p for p in (ADD3_MD, ADD3_JSON) if p.is_file()]
    tracked = [p for p in (ADD3_MD, ADD3_JSON) if _tracked(p)]
    if len(have) == 2 and len(tracked) == 2 and all(_clean(p) for p in tracked):
        print("[write] ADD3 md and json are committed and unchanged: frozen (spec 9 / 15), nothing rewritten; running check",
              flush=True)
        return cmd_check()
    if tracked:
        print(f"REFUSED: {[p.name for p in tracked]} tracked by git but modified or incomplete; a committed pre-registration "
              f"is never regenerated (restore it with git, or write a new addendum)", file=sys.stderr, flush=True)
        return 3
    try:
        v, p, c, checked = preconditions()
        D = collect_content(v, p, c, checked)
        md = make_md(D)
        md_sha = hashlib.sha256(md.encode()).hexdigest()          # = sha256 of the file write_atomic writes (str.encode)
        J = make_json(D, md_sha)                                   # everything computed before the first write
    except Refuse as e:
        print(f"{'REFUSED' if e.code == 3 else 'NOT READY'} (exit {e.code}), nothing written:\n  " + "\n  ".join(e.reasons),
              file=sys.stderr, flush=True)
        return e.code
    L.write_atomic(ADD3_MD, md)
    if L.sha256(ADD3_MD) != md_sha:
        print("STOP: the written md sha256 differs from the sha256 of its text", file=sys.stderr, flush=True)
        return 3
    L.write_json(ADD3_JSON, J)
    # self-check (PR-12): the files read back as the gate reads them
    rec = json.loads(ADD3_JSON.read_text(encoding="utf-8"))
    bad = []
    if rec.get("md_sha256") != L.sha256(ADD3_MD):
        bad.append("md_sha256 != sha256(md) after writing")
    ins = rec.get("inputs_sha256") or {}
    if not need_paths() <= set(ins):
        bad.append(f"inputs_sha256 lacks {sorted(need_paths() - set(ins))}")
    bad += [f"{q} changed while writing" for q, s in ins.items() if _sha(q) != s]
    if bad:
        print("STOP: self-check after writing failed: " + "; ".join(bad), file=sys.stderr, flush=True)
        return 3
    print(f"[write] {ADD3_MD.name} ({md_sha[:16]}) + {ADD3_JSON.name}: {len(ins)} gated files, stage 1 rungs "
          f"{', '.join(STAGE_RUNGS)}, code commit {D['git_head'][:10]}. Commit both files (and prereg.py) before Phase M1.",
          flush=True)
    return 0


# ================================================================ stage 2: the ADD3 supplement (md section 10)
SUPP_VERSION = "d1-ladder-a2"
SUPP_MD = L.RES / "PREREG_D1PAIR_ADD3_G2.md"
STAGE2_RUNGS = ("g2",)
# the committed stage-1 ADD3 (cae9922b): the md never changes (unit fingerprints carry its sha256); the json is extended
# exactly once, from this state
ADD3_MD_SHA256 = "daa82585c0ab06653b582524d9fbb35a99dddff9f5c0380e3f8e6b4514cd5667"
ADD3_JSON_SHA256 = "a8258d7376d301a7b4e4d614957791af87a4d50e7abc4a1cdf755d101fc89a13"
G2_RES = IN.G2_RES                                        # analysis/d1_g2/results (merged from origin/d1-g2)
G2_FU2 = Path(IN.RUNGS["g2"]["fu2_res"])
H_PARQUET_G2 = Path(IN.RUNGS["g2"]["h_parquet"])
F_REPORT_G2 = G2_RES / "F_report.json"
H_REPORT_G2 = G2_RES / "H_report.json"
F_DECISION_G2 = G2_RES / "F_DECISION.md"
C_SUMMARY_G2 = G2_RES / "C_summary.json"
C_COMPARE_G2 = G2_RES / "C_compare.csv"
# inputs._g2_ready's committed files + the other D1-G2 files this supplement reads
G2_COMMITTED = (F_REPORT_G2, H_REPORT_G2, H_PARQUET_G2, G2_FU2 / "T_report.json", G2_FU2 / "T_manifest.csv",
                G2_FU2 / "Q_report.json", G2_FU2 / "S_splits.json", F_DECISION_G2, C_SUMMARY_G2, C_COMPARE_G2)
SETS2 = ("P_dft", "P_g1", "P_01", "P_g2", "P_012")
# primary = D1-G2 Phase C's downstream same-TS judgment (spec 12 defers to it; C_summary same_ts_downstream.column, C-13,
# consumer 'D1LADDER 12 E2'); second = the literal G2-vs-DFT column (named in the stage-2 task text), descriptive too
SAME_TS_COLS = ("same_ts_g1", "same_ts_dft")
SAME_TS_KO = {"same_ts_g1": "G2 TS 대 G1 TS(라벨 TS를 xTB 표면으로 옮긴 구조; 둘 다 xTB)",
              "same_ts_dft": "G2 TS 대 라벨(DFT) TS"}
WINDOW_JOBS = ("lad_m1", "lad_xgb", "lad_e1")             # job names whose tasks run an ADD3 gate at start (PS-11)
SAME_TS_CLASSES = ("same", "mixed", "different", "unclassified")
DROP_CATS = ("g2_not_ok", "not_target", "no_tokens_or_182", "no_h", "both_usable")
DROP_KO = {"g2_not_ok": "G2 TS 없음(D1-G2 G 실패)", "not_target": "대상 반응 아님", "no_tokens_or_182":
           "토큰·182 없음(D1-G2 F gate 실패, 3절)", "no_h": "HF 9값 없음(D1-G2 H)", "both_usable": "두 반응 모두 사용 가능(불일치)"}
# stage-2 run gate literals of both trainers (their sha256 is gated too; quoted in md 6-2)
STAGE2_LITERALS = ('STAGE_RUNGS = {1: ("dft", "g1"), 2: ("g2",)}', 'rs = sorted({u["rung"] for u in us})',
                   'add3 = gate_add3(need, rs)', 'if int(stage) == 2:', 'gate_c_stage2()',
                   'if not (c.get("all_pass") and c.get("g2_verified")):')
GATE_QUOTE = ('rs = sorted({u["rung"] for u in us})',
              'need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}',
              'add3 = gate_add3(need, rs)', 'gp = gate_p(rs, labels)', 'if int(stage) == 2:', '    gate_c_stage2()',
              '# gate_add3:', 'ins = rec.get("inputs_sha256") or {}', 'lack = sorted(set(need) - set(ins))',
              'for path, s in ins.items():', '    if path not in need and _foreign_to(path, rungs):',
              '        skipped.append(path)', '        continue',
              '    if not Path(path).is_file() or L.sha256(path) != s:',
              '        bad.append(f"input {path} changed or missing")',
              '# _foreign_to:', 'return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)',
              '# gate_c_stage2 (MA-17):', 'if not (c.get("all_pass") and c.get("g2_verified")):')
STAGE1_RESULTS = tuple(L.RES / n for n in ("L_predictions.parquet", "L_report.json", "L_val.csv", "L_xgb_report.json",
                                            "E1_report.json", "E1_metrics.csv", "E1_goal.csv"))
SUPP_DECISIONS = {
    "PS-1": "one run writes the supplement md first, then the extended json atomically (os.replace); nothing is committed: "
            "the exact git commands are printed; until that commit every starting trainer run / evaluate e1 exits 3 at "
            "the ADD3 gate (window risk printed; running processes unaffected; a gated-out task wrote nothing); a failed "
            "or raising self-check after the json write puts the committed json bytes back and removes the new md",
    "PS-2": "base = the committed stage-1 json (git show HEAD, sha256 ADD3_JSON_SHA256); it must re-serialise byte for "
            "byte (json.dumps indent 1); new entries appended in the md section 10 order, 'supplement' appended last; "
            "removing the additions must give the committed bytes (checked before and after writing)",
    "PS-3": "refusal = docstring 'Stage 2' preconditions (P_build_g2 'own', P_report with g2 / P_g2 / P_012, C_tests "
            "g2_verified + every check + g2 sub-results pass + committed, D1-G2 files committed, F stop only as the "
            "F_DECISION.md deviation, no L__g2__ unit, both gate mirrors pass on the extended json)",
    "PS-4": "supplement.commit = HEAD at write time (the code and committed inputs the supplement was made from); the commit "
            "that adds the md + json comes after (git log -1 -- results/PREREG_D1PAIR_ADD3_G2.md)",
    "PS-5": "P20 = option (c) (decision 2026-10-23, inputs.py P20): g2 rows of constrained-dipole keys keep their own G2 "
            "reference, so the DFT-derived '|fixed:' state never enters a g2 value; other keys per-key minimum (spec 4-1); "
            "g2_q_keys = 'own'. Limitation (recorded deviation): the kept rows are chosen by the Q_keys 'constrained' "
            "flag = the D1 label ref_port.constrained (D1 / DFT pipeline), not by G2's own port result",
    "PS-6": "D1-G2 Phase F STOP (gate 1.36 % > 1 %) accepted as a deviation (analysis/d1_g2/results/F_DECISION.md, "
            "committed; its quoted numbers must equal F_report gate_failure; the g2 build accepted it with "
            "--accept-g2f-stop and recorded the decision sha256 = the file now); the failed reactions are not usable in g2",
    "PS-7": "same-TS split for E2 (descriptive only): reaction = D1-G2 C_compare.csv same_ts_g1 (spec 12: 'D1-G2 Phase C의 "
            "같은 TS 판정' = its decision C-13, which names D1LADDER 12 E2 as a consumer: G2 vs G1, both xTB); pair = same "
            "(both ends) / mixed / different / unclassified (an end without a judgment); the literal G2-vs-DFT column "
            "same_ts_dft (named in the stage-2 task text) is reported the same way as a second descriptive split",
    "PS-8": "coverage = P_main pairs of the T1 test lists (folds 0-4) that are in the rung's set; pairs outside P_g2 count "
            "as not predictable (not filled in); reasons per pair = first failing end category (DROP_CATS order)",
    "PS-9": "Phase C (g2) committed first (spec 8 / PR-4 analogue): C_tests.json + the 5 C_CODE files tracked and = HEAD",
    "PS-10": "a committed supplement is frozen (supplement = check --stage 2); an uncommitted earlier supplement on disk is "
             "regenerated from the committed json; any other modification of the json -> refused",
    "PS-11": "commit window guard: not ready (exit 2) while this user has a pending (PD / CF) task named lad_m1, lad_xgb "
             "or lad_e1 (the jobs whose start runs an ADD3 gate: train_atoms / train_xgb run, evaluate e1); other names "
             "are not detected; a failed squeue only warns (recorded)",
    "PS-12": "origin/d1-g2 as last fetched must be contained in HEAD (spec 12 start condition; not ready otherwise)",
}


def _head_bytes(p):
    """the HEAD version of a tracked file (bytes), None if absent."""
    rel = Path(p).resolve().relative_to(L.REPO.resolve()).as_posix()
    r = subprocess.run(["git", "-C", str(L.HERE), "show", f"HEAD:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _without_supplement(obj, paths):
    """a copy of a (supplemented) json record without 'supplement' and without the inputs entries of `paths`."""
    o = json.loads(json.dumps(obj))
    o.pop("supplement", None)
    for k in ("inputs_sha256", "inputs_roles"):
        d = o.get(k)
        if isinstance(d, dict):
            for q in paths:
                d.pop(q, None)
    return o


def supplement_md_problems(rec):
    """stage-2 check: json 'supplement' present, its md committed, unchanged and with the recorded sha256."""
    s = rec.get("supplement")
    if not isinstance(s, dict):
        return [f"{ADD3_JSON.name} has no 'supplement' key (prereg.py supplement, then commit its two files)"]
    bad = []
    if s.get("md") != _disp(SUPP_MD):
        bad.append(f"supplement.md {s.get('md')} != {_disp(SUPP_MD)}")
    if not SUPP_MD.is_file():
        bad.append(f"{SUPP_MD.name} missing")
    elif not _tracked(SUPP_MD):
        bad.append(f"{SUPP_MD.name} not committed")
    elif not _clean(SUPP_MD):
        bad.append(f"{SUPP_MD.name} modified since its commit")
    elif L.sha256(SUPP_MD) != s.get("md_sha256"):
        bad.append(f"{SUPP_MD.name} sha256 != {ADD3_JSON.name} supplement.md_sha256")
    return bad


def stage2_run_gates(labels=CR2):
    """the other run gates of a stage-2 process: train_atoms.gate_p / gate_c_stage2 (MA-16 / MA-17) for rung g2."""
    bad = []
    p = _json(P_REPORT) or {}
    if p.get("running") or not p.get("all_pass") or p.get("stop"):
        bad.append(f"P_report.json running {p.get('running')}, all_pass {p.get('all_pass')}, stop {p.get('stop')} (MA-16)")
    for r in STAGE2_RUNGS:
        want = ((p.get("rungs") or {}).get(r) or {}).get("table_sha256")
        if want is None or _sha(IN.table_path(r)) != want:
            bad.append(f"{IN.table_path(r)} sha256 != P_report rungs.{r}.table_sha256 ({want}) (MA-16)")
    lab = (p.get("labels") or {}).get(_ab(labels)) or {}
    if not lab.get("ids_ok") or _sha(labels) != lab.get("sha256"):
        bad.append(f"label file {labels}: sha256 / ids differ from P_report labels (MA-16)")
    c = _json(C_TESTS) or {}
    if not (c.get("all_pass") and c.get("g2_verified")):
        bad.append(f"C_tests.json all_pass {c.get('all_pass')}, g2_verified {c.get('g2_verified')} (MA-17)")
    return bad


def window_tasks():
    """PS-11: this user's pending (PD / CF) tasks named in WINDOW_JOBS (squeue, run inside the job) -> (hits | None when
    the query failed, record)."""
    import getpass
    try:
        user = getpass.getuser()
    except Exception:                                              # noqa: BLE001
        user = ""
    cmd = ["squeue", "-h", "-r", "-u", user, "-t", "PD,CF", "-o", "%i %j"]
    rec = dict(command=" ".join(cmd), job_names=list(WINDOW_JOBS), checked=_now())
    if not user:
        return None, dict(rec, ok=False, error="no user name")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, dict(rec, ok=False, error=f"{type(e).__name__}: {e}")
    if r.returncode != 0:
        return None, dict(rec, ok=False, error=(r.stderr or "").strip()[:300], rc=r.returncode)
    hit = []
    for line in r.stdout.splitlines():
        part = line.split(None, 1)
        if len(part) == 2 and part[1].strip() in WINDOW_JOBS:
            hit.append(f"{part[0]} {part[1].strip()}")
    return hit, dict(rec, ok=True, n_pending=len(hit), pending=hit[:20])


def supplement_base():
    """-> (mode, HEAD json bytes, json bytes on disk): 'fresh' (disk = HEAD = the stage-1 json), 'pending' (an earlier
    supplement run wrote md + json, not committed: regenerated from HEAD) or 'frozen' (HEAD json has the supplement)."""
    for p in (ADD3_MD, ADD3_JSON):
        if not p.is_file():
            raise Refuse(2, [f"{p.name} missing (stage 1: prereg.py write, then commit)"])
        if not _tracked(p):
            raise Refuse(2, [f"{p.name} not committed (commit the stage-1 ADD3 first)"])
    if not _clean(ADD3_MD) or L.sha256(ADD3_MD) != ADD3_MD_SHA256:
        raise Refuse(3, [f"{ADD3_MD.name} differs from HEAD or from its stage-1 sha256 {ADD3_MD_SHA256[:16]}: it must never "
                         f"change (unit fingerprints carry it); restore it with git checkout HEAD -- {_disp(ADD3_MD)}"])
    head, disk = _head_bytes(ADD3_JSON), ADD3_JSON.read_bytes()
    if head is None:
        raise Refuse(3, [f"git show HEAD:{_disp(ADD3_JSON)} failed"])
    hobj = json.loads(head.decode("utf-8"))
    if "supplement" in hobj:
        if disk != head or not _clean(ADD3_JSON):
            raise Refuse(3, [f"the committed {ADD3_JSON.name} carries a supplement and was modified since: a committed "
                             f"supplement is never rewritten (restore it with git checkout HEAD -- {_disp(ADD3_JSON)})"])
        return "frozen", head, disk
    if hashlib.sha256(head).hexdigest() != ADD3_JSON_SHA256 or hobj.get("md_sha256") != ADD3_MD_SHA256:
        raise Refuse(3, [f"HEAD {ADD3_JSON.name} is not the stage-1 ADD3 json (sha256 {ADD3_JSON_SHA256[:16]}, md_sha256 "
                         f"{ADD3_MD_SHA256[:16]}): the only allowed edit is this supplement (PR-5)"])
    if _head_bytes(SUPP_MD) is not None:                 # only staged (git add, no commit) = still pending
        raise Refuse(3, [f"{SUPP_MD.name} is in HEAD but HEAD {ADD3_JSON.name} has no 'supplement' key "
                         "(inconsistent; resolve by hand)"])
    if disk == head and _clean(ADD3_JSON):
        return "fresh", head, disk
    plan = list(((hobj.get("stage2_plan") or {}).get("supplement_entries") or {}))
    try:
        dobj = json.loads(disk.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        dobj = None
    if not (isinstance(dobj, dict) and "supplement" in dobj
            and json.dumps(_without_supplement(dobj, plan), indent=1).encode("utf-8") == head):
        raise Refuse(3, [f"{ADD3_JSON.name} modified since its commit by something other than a supplement run: restore it "
                         f"(git checkout HEAD -- {_disp(ADD3_JSON)}) and rerun"])
    return "pending", head, disk


def check_d1g2(b2, b3, ok):
    """D1-G2 readiness = inputs._g2_ready (with --accept-g2f-stop: the F_report STOP passes when its reasons are the F
    gate rate only and F_DECISION.md is committed), here also requiring that F_DECISION.md quotes the F_report numbers
    (PS-6), plus origin/d1-g2 in HEAD (PS-12). -> info dict."""
    n0 = len(b2) + len(b3)
    info, reps = {}, {}
    for p in (F_REPORT_G2, H_REPORT_G2):
        r = _json(p)
        if r is None:
            b2.append(f"{_disp(p)} missing (D1-G2 Phase F / H)")
            continue
        reps[p.name] = r
        info[p.name] = dict(sha256=_sha(p), complete=r.get("complete"), stop=r.get("stop"),
                            stop_reasons=r.get("stop_reasons"))
        if not r.get("complete"):
            b2.append(f"D1-G2 {p.name}: complete {r.get('complete')}")
    h = reps.get("H_report.json") or {}
    if h.get("stop"):
        b3.append(f"D1-G2 H_report.json stop {h.get('stop_reasons')} (spec 16 '2단계': wait for D1-G2)")
    if h:
        outs = h.get("outputs") or {}
        rel = str(H_PARQUET_G2.relative_to(L.REPO))
        hit = [k for k in outs if k == str(H_PARQUET_G2) or k.endswith(os.sep + rel)]
        info["H_report_output_key"] = hit
        if len(hit) != 1 or outs.get(hit[0]) != _sha(H_PARQUET_G2):
            b3.append(f"D1-G2 H_report outputs: {len(hit)} entries end with {rel} (need exactly one with the sha256 of "
                      f"{_disp(H_PARQUET_G2)} now)")
    f = reps.get("F_report.json") or {}
    fdev = None
    if f.get("stop"):
        gf = f.get("gate_failure") or {}
        txt = F_DECISION_G2.read_text(encoding="utf-8") if F_DECISION_G2.is_file() else ""
        try:
            nums = [f"{int(gf['n']):,}", f"{int(gf['n_g2_ok']):,}", f"{100 * float(gf['rate']):.2f} %"]
        except (KeyError, TypeError, ValueError):
            nums = []
        reasons = [str(x) for x in (f.get("stop_reasons") or [])]
        covered = bool(txt and nums and all(x in txt for x in nums) and reasons
                       and all(x.startswith("F gate failure") for x in reasons))
        fdev = dict(accepted=covered, decision=_disp(F_DECISION_G2), decision_sha256=_sha(F_DECISION_G2),
                    decision_commit=_last_commit(F_DECISION_G2), numbers_quoted_in_decision=nums, stop_reasons=reasons,
                    gate_failure=gf, threshold=f.get("threshold"), counts=f.get("counts"), n_g2_ok=f.get("n_g2_ok"),
                    n_targets=f.get("n_targets"), versions=f.get("versions"), decision_text=txt)
        if not covered:
            b3.append(f"D1-G2 F_report.json stop {reasons}: not the STOP that F_DECISION.md accepted (numbers {nums} quoted "
                      f"there: {[x in txt for x in nums]}; spec 16 '2단계': wait for D1-G2)")
    info["f_deviation"] = fdev
    unc = [_disp(p) for p in G2_COMMITTED if not _committed(p)]
    info["not_committed"] = unc
    if unc:
        b2.append(f"D1-G2 files not committed or modified since HEAD (spec 12; inputs._g2_ready): {unc}")
    info["last_commit"] = {_disp(p): _last_commit(p) for p in G2_COMMITTED}
    info["origin_d1g2_in_head"] = _git_rc("merge-base", "--is-ancestor", "origin/d1-g2", "HEAD") == 0
    info["origin_d1g2"] = _git_out("rev-parse", "--short", "origin/d1-g2")
    if not info["origin_d1g2_in_head"]:
        b2.append(f"origin/d1-g2 ({info['origin_d1g2']}, as last fetched) is not contained in HEAD: merge it first "
                  "(spec 12 start condition, PS-12)")
    if len(b2) + len(b3) == n0:
        ok.append("D1-G2 F_report / H_report complete (F stop = the F_DECISION.md deviation), H output = "
                  "labels_d1_hf_g2.parquet, D1-G2 files committed, origin/d1-g2 in HEAD")
    return info


def check_build_g2(b2, b3, ok):
    """P_build_g2.json passed, current (inputs._build_ok) and built with the P20 option (c) mode 'own'; a D1-G2 F_report
    STOP accepted by that build (--accept-g2f-stop) with the F_DECISION.md that is committed now."""
    good, br, code = IN._build_ok("g2")
    redo = ("rebuild: sbatch -p cpu1,cpu2 jobs/p_stage2.sh from analysis/d1_ladder (dft / g1 --force byte-identical, "
            "then inputs.py build g2 --accept-g2f-stop --g2-cons-own, then inputs.py pairs), then Phase C with g2")
    if not good:
        (b3 if code == 3 else b2).append(f"g2 build: {br} ({redo})")
        return None
    ref = br.get("references") or {}
    modes = dict(report=br.get("g2_q_keys"), fingerprint=(br.get("fingerprint") or {}).get("g2_q_keys"),
                 references=ref.get("g2_mode"))
    n0 = len(b3)
    if any(v != "own" for v in modes.values()):
        b3.append(f"P_build_g2.json g2_q_keys {modes} != 'own' (P20 decision 2026-10-23 = option (c); {redo})")
    if ref.get("n_constrained_replaced") not in (0, None):
        b3.append(f"P_build_g2.json references.n_constrained_replaced {ref.get('n_constrained_replaced')} != 0 with mode "
                  "'own'")
    n2 = len(b2)
    if (_json(F_REPORT_G2) or {}).get("stop"):
        # F_DECISION.md is not in inputs._fingerprint: an edit after the build is caught here (PS-6)
        acc = (br.get("g2_gates") or {}).get("F_stop_accepted") or {}
        if (br.get("fingerprint") or {}).get("accept_f0_stop") is not True or not acc:
            b2.append("P_build_g2.json was not built with --accept-g2f-stop although the D1-G2 F_report has stop true "
                      f"({redo})")
        elif acc.get("decision_sha256") != _sha(F_DECISION_G2):
            b2.append(f"P_build_g2.json g2_gates.F_stop_accepted.decision_sha256 {str(acc.get('decision_sha256'))[:12]} != "
                      f"{_disp(F_DECISION_G2)} now ({str(_sha(F_DECISION_G2))[:12]}; {redo})")
    if len(b3) == n0 and len(b2) == n2:
        ok.append("P_build_g2.json all_pass, current (fingerprint), g2_q_keys 'own' (P20 option (c)); a D1-G2 F stop "
                  "only as accepted by the build with the F_DECISION.md of now")
    return br


def check_p2(b2, b3, ok, hobj):
    """P_report.json with the g2 rung and the sets P_g2 / P_012; stage-1 sets = the ADD3 json entries."""
    p = _json(P_REPORT)
    if p is None:
        b2.append("results/P_report.json missing (inputs.py pairs)")
        return None
    if p.get("running"):
        b2.append("results/P_report.json is being written (inputs.py pairs running)")
        return None
    if p.get("stop"):
        b3.append(f"P_report.json stop: {p.get('stop_reasons') or p.get('stop_reason')}")
        return None
    if p.get("all_pass") is not True:
        b2.append(f"P_report.json all_pass {p.get('all_pass')}")
        return None
    if "g2" not in (p.get("rungs_built") or []):
        b2.append(f"P_report.json rungs_built {p.get('rungs_built')} has no g2 ({(p.get('rungs_not_built') or {}).get('g2')}; "
                  "sbatch -p cpu1,cpu2 jobs/p_stage2.sh: build g2 --accept-g2f-stop --g2-cons-own, then pairs)")
        return None
    n0 = len(b2) + len(b3)
    for r in IN.RUNG_ORDER:
        rr = (p.get("rungs") or {}).get(r) or {}
        now = _sha(IN.table_path(r))
        if rr.get("table_sha256") is None or rr.get("table_sha256") != now:
            b3.append(f"P_report rungs.{r}.table_sha256 {rr.get('table_sha256')} != {IN.table_path(r)} now ({now})")
    if ((p.get("rungs") or {}).get("g2") or {}).get("build_report_sha256") != _sha(IN.build_report_path("g2")):
        b2.append("P_report.json was made from another P_build_g2.json (rerun inputs.py pairs)")
    ins = hobj.get("inputs_sha256") or {}
    for name in SETS2:
        s = (p.get("sets") or {}).get(name) or {}
        if not s.get("file") or _sha(s["file"]) != s.get("sha256") or _ab(s["file"]) != _ab(IN.set_path(name)):
            b3.append(f"P_report sets.{name} missing, not at {IN.set_path(name)} or its file changed")
        elif name in ("P_dft", "P_g1", "P_01") and ins.get(_ab(s["file"])) != s.get("sha256"):
            b3.append(f"{name}: P_report sha256 != the ADD3 json entry (stage-1 set changed: the stage-1 gate would exit 3)")
    lab = (p.get("labels") or {}).get(str(CR2)) or {}
    if not lab.get("ids_ok") or lab.get("sha256") != _sha(CR2):
        b3.append(f"P_report labels[{CR2}]: ids_ok {lab.get('ids_ok')}, sha256 != the file now")
    if not (p.get("fu2_check") or {}).get("ok"):
        b3.append("P_report fu2_check.ok is not true (spec 16 P)")
    if not (p.get("dft_vs_g1") or {}).get("ok"):
        b3.append("P_report dft_vs_g1.ok is not true (spec 16 P)")
    if len(b2) + len(b3) == n0:
        ok.append("P_report all_pass with rungs dft, g1, g2 and sets P_dft, P_g1, P_01, P_g2, P_012 (= the files now; "
                  "stage-1 sets = the ADD3 json)")
    return p


def check_c2(b2, b3, ok):
    """C_tests.json passed with the g2 rung (tests/run_tests.py merge keys) and committed. -> (c, g2 sub-statuses)."""
    c = _json(C_TESTS)
    if c is None:
        b2.append("results/C_tests.json missing (Phase C)")
        return None, {}
    if c.get("failed"):
        b3.append(f"C_tests.json failed checks {c.get('failed')} (spec 16 C STOP)")
        return c, {}
    keys = dict(all_pass=c.get("all_pass") is True, status_pass=c.get("status") == "pass",
                stage_verified_2=c.get("stage_verified") == 2, g2_verified=c.get("g2_verified") is True,
                scope_g2=c.get("all_pass_scope") == "rungs dft, g1, g2", labels_final=c.get("labels_final") is True,
                no_errors=not c.get("errors"), none_not_final=not c.get("not_final"),
                every_check_pass=all((((c.get("checks") or {}).get(k)) or {}).get("status") == "pass" for k in CHECKS),
                current_ok=(c.get("current") or {}).get("ok") is True,
                consistency_ok=(c.get("consistency") or {}).get("ok") is True)
    n0 = len(b2)
    if not all(keys.values()):
        b2.append(f"C_tests.json not passed with g2: {[k for k, v in keys.items() if not v]} (status {c.get('status')}, "
                  f"scope {c.get('all_pass_scope')}; rerun Phase C with the g2 table: tests gpu + cpu + merge)")
    lab = c.get("labels") or {}
    if lab.get("path") != str(CR2) or lab.get("sha256") != _sha(CR2):
        b2.append("C_tests.json labels != F_pairs_cr2.csv now (rerun Phase C)")
    for r in IN.RUNG_ORDER:
        rr = (c.get("rungs") or {}).get(r) or {}
        if rr.get("status") != "ready" or rr.get("table_sha256") != _sha(IN.table_path(r)):
            b2.append(f"C_tests.json rungs.{r}: status {rr.get('status')}, table_sha256 != the table now (rerun Phase C)")
    pr = c.get("p_report") or {}
    if not pr.get("ready") or pr.get("sha256") != _sha(P_REPORT):
        b2.append("C_tests.json p_report not ready or its sha256 != results/P_report.json now (rerun Phase C)")
    code_now = {k: _sha(q) for k, q in C_CODE.items()}
    diff = sorted(k for k in set(code_now) | set(c.get("code_sha256") or {})
                  if (c.get("code_sha256") or {}).get(k) != code_now.get(k))
    if diff:
        b2.append(f"C_tests.json code_sha256 differs from the code now: {diff} (rerun Phase C)")
    g2sub = {}
    for k in CHECKS:
        for part, res in ((((c.get("checks") or {}).get(k)) or {}).get("parts") or {}).items():
            rr = res.get("rungs") if isinstance(res, dict) else None
            if isinstance(rr, dict) and "g2" in rr:
                g2sub[f"{k}/{part}"] = (rr.get("g2") or {}).get("status")
    if not g2sub or any(v != "pass" for v in g2sub.values()):
        b2.append(f"C_tests.json per-rung g2 results not all pass: {g2sub or 'none'}")
    unc = [n for n, q in {**C_CODE, "results/C_tests.json": C_TESTS}.items() if not _committed(q)]
    if unc:
        b2.append(f"not committed or modified since HEAD (commit the Phase C (g2) state first, spec 8 / PS-9): {unc}")
    if len(b2) == n0:
        ok.append("C_tests.json all_pass, stage_verified 2, g2_verified, every check pass (g2 sub-results pass), labels / "
                  "tables / P_report / code = the files now, committed")
    return c, g2sub


def supplement_preconditions(hobj):
    b2, b3, ok = [], [], []
    plan = (hobj.get("stage2_plan") or {}).get("supplement_entries") or {}
    want = [(_ab(pth), role) for pth, role in stage2_supplement_paths()]
    if list(plan.items()) != want:
        b3.append(f"stage2_supplement_paths() {want} != the committed json stage2_plan.supplement_entries {list(plan.items())} "
                  "(md section 10 list; paths, roles and order must be equal)")
    dup = [q for q, _ in want if q in (hobj.get("inputs_sha256") or {}) or q in (hobj.get("inputs_roles") or {})]
    if dup:
        b3.append(f"already in the committed json inputs: {dup} (existing entries are never changed)")
    for f in (HERE / "train_atoms.py", HERE / "train_xgb.py"):
        txt = f.read_text(encoding="utf-8")
        lines = {x.strip() for x in txt.splitlines()}
        miss = [s for s in GATE_LITERALS + STAGE2_LITERALS if s not in txt]
        miss += [s for s in GATE_QUOTE if not s.startswith("#") and s.strip() not in lines]
        if miss:
            b3.append(f"{f.name}: gate literals not found (the gate changed; regenerate this contract): {miss}")
    if UNITS.is_dir():
        g2u = sorted(e.name for e in os.scandir(UNITS) if e.name.startswith("L__g2__"))
        if g2u:
            b3.append(f"{len(g2u)} stage-2 unit files in $LAD/units before the supplement commit (spec 16 A): {g2u[:5]}")
    hit, queue = window_tasks()
    if hit:
        b2.append(f"{len(hit)} pending task(s) named {list(WINDOW_JOBS)} would start inside the commit window and exit 3 "
                  f"at their ADD3 gate: {hit[:5]} (submit the supplement when none is pending, PS-11)")
    elif hit is None:
        print(f"[supplement] warning: queue check failed ({queue.get('error')}): pending lad_m1 / lad_xgb / lad_e1 tasks "
              "not checked (PS-11)", flush=True)
    else:
        ok.append(f"no pending {'/'.join(WINDOW_JOBS)} task (commit window, PS-11)")
    g2 = check_d1g2(b2, b3, ok)
    br = check_build_g2(b2, b3, ok)
    p = check_p2(b2, b3, ok, hobj)
    c, g2sub = check_c2(b2, b3, ok)
    entries = []
    p012 = _ab(IN.set_path("P_012"))
    for pth, role in stage2_supplement_paths():
        q = _ab(pth)
        s = _sha(pth)
        col = _column(pth)
        if s is None:
            b2.append(f"{_disp(pth)} missing ({role})")
        exp = "dft+g1+g2" if q == p012 else "g2"
        if col != exp:
            b3.append(f"{_disp(pth)}: column {col} != {exp} (inputs.rung_roots changed; the stage-1 skip rule depends on it)")
        entries.append(dict(path=q, disp=_disp(pth), role=role, column=col, sha256=s))
    if b3:
        raise Refuse(3, b3 + b2)
    if b2:
        raise Refuse(2, b2)
    return dict(g2=g2, br=br, p=p, c=c, g2sub=g2sub, entries=entries, ok=ok, queue=queue)


def _tri(v):
    s = str(v).strip().lower()
    return True if s == "true" else (False if s == "false" else None)


def coverage_same_ts(p):
    """PS-7 / PS-8: coverage of the P_main test pairs per set, reasons for P_g2, same-TS classes of the P_g2 ones."""
    S = pd.read_csv(IN.S_FOLDS, usecols=["pair_id", "a_rxn", "b_rxn", "pos", "in_P_main"], dtype={"pair_id": str})
    test = {}
    for k in range(NF):
        rec = json.loads((IN.FU2_SPLITS / f"T1__f{k}.json").read_text(encoding="utf-8"))
        for q in map(str, rec["test"]):
            test[q] = k
    pm = S.in_P_main.astype(str).str.lower().eq("true")
    B = S[pm & S.pair_id.isin(set(test))].copy()
    B["fold"] = B.pair_id.map(test).astype(int)
    para = B.pos.fillna("").astype(str).eq("p")
    ids = {n: set(pd.read_csv((p.get("sets") or {})[n]["file"], usecols=["pair_id"], dtype={"pair_id": str}).pair_id)
           for n in SETS2}

    def frac(a, b):
        return float(a / b) if b else None
    cov = dict(rule=SUPP_DECISIONS["PS-8"], n_P_main_test=int(len(B)), n_P_main_test_para=int(para.sum()), sets={})
    for n, s in ids.items():
        m = B.pair_id.isin(s)
        cov["sets"][n] = dict(n=int(m.sum()), frac=frac(int(m.sum()), len(B)), n_para=int((m & para).sum()),
                              frac_para=frac(int((m & para).sum()), int(para.sum())),
                              by_fold={str(k): dict(n=int((m & (B.fold == k)).sum()), of=int((B.fold == k).sum()))
                                       for k in range(NF)})
    T = pd.read_parquet(IN.table_path("g2"), columns=["rxn_id", "has_tok", "has_x182", "has_h", "usable"])
    Cc = pd.read_csv(C_COMPARE_G2, usecols=["rxn_id", "g2_status"] + list(SAME_TS_COLS), dtype=str)
    st = dict(zip(Cc.rxn_id.astype(int), Cc.g2_status.fillna("").astype(str)))
    tt = {int(r): (bool(a), bool(b), bool(h)) for r, a, b, h in zip(T.rxn_id, T.has_tok, T.has_x182, T.has_h)}

    def cat(r):
        r = int(r)
        if st.get(r, "") != "ok":
            return "g2_not_ok"
        t = tt.get(r)
        if t is None:
            return "not_target"
        if not (t[0] and t[1]):
            return "no_tokens_or_182"
        return "ok" if t[2] else "no_h"
    out = B[~B.pair_id.isin(ids["P_g2"])]
    reasons = {}
    for a_, b_ in zip(out.a_rxn, out.b_rxn):
        ca, cb = cat(a_), cat(b_)
        r = next((x for x in DROP_CATS[:-1] if x in (ca, cb)), "both_usable")
        reasons[r] = reasons.get(r, 0) + 1
    cov["P_g2_missing_by_reason"] = {x: reasons.get(x, 0) for x in DROP_CATS}
    cov["P_g2_missing_para"] = int((~B.pair_id.isin(ids["P_g2"]) & para).sum())
    same = {col: {int(r): _tri(v) for r, v in zip(Cc.rxn_id, Cc[col])} for col in SAME_TS_COLS}
    G = B[B.pair_id.isin(ids["P_g2"])]
    gpara = G.pos.fillna("").astype(str).eq("p").to_numpy()
    usable = set(T.rxn_id[T.usable.astype(bool)].astype(int))
    sts = dict(rule=SUPP_DECISIONS["PS-7"], source=_disp(C_COMPARE_G2), source_sha256=_sha(C_COMPARE_G2),
               columns=list(SAME_TS_COLS), primary=SAME_TS_COLS[0], population="P_g2 pairs among the P_main test pairs",
               n_pairs=int(len(G)), n_pairs_para=int(gpara.sum()), classes={}, reactions_usable={})
    for col in SAME_TS_COLS:
        cls = []
        for a_, b_ in zip(G.a_rxn, G.b_rxn):
            x, y = same[col].get(int(a_)), same[col].get(int(b_))
            cls.append("unclassified" if x is None or y is None else
                       ("same" if (x and y) else ("different" if not (x or y) else "mixed")))
        cls = np.asarray(cls, dtype=object)
        sts["classes"][col] = dict(all={k: int((cls == k).sum()) for k in SAME_TS_CLASSES},
                                   para={k: int(((cls == k) & gpara).sum()) for k in SAME_TS_CLASSES})
        vals = [same[col].get(r) for r in usable]
        sts["reactions_usable"][col] = dict(n=len(vals), same=sum(1 for v in vals if v is True),
                                            different=sum(1 for v in vals if v is False),
                                            unclassified=sum(1 for v in vals if v is None))
    return cov, sts


def build_summary(br):
    ref, tok, tab = br.get("references") or {}, br.get("tokens") or {}, br.get("table") or {}
    return dict(
        file=_disp(IN.build_report_path("g2")), sha256=_sha(IN.build_report_path("g2")),
        git_state=_git_state_at(IN.build_report_path("g2")), finished=br.get("finished"), git_head=br.get("git_head"),
        code_version=br.get("code_version"), g2_q_keys=br.get("g2_q_keys"),
        fingerprint_g2_q_keys=(br.get("fingerprint") or {}).get("g2_q_keys"),
        table={k: tab.get(k) for k in ("n_targets", "n_tok", "n_x182", "n_h", "n_usable", "n_tok_without_h",
                                       "n_tok_without_x182", "channels_unchanged")},
        h_changed=tab.get("h_changed"),
        references={k: ref.get(k) for k in ("g2_mode", "n_constrained_kept_own", "dev_kcal_kept_own",
                                            "n_rows_constrained_keys", "n_constrained_replaced", "n_fragments_with_energy",
                                            "n_keys",
                                            "n_keys_multi_row", "n_fragments_replaced", "n_jobs_replaced",
                                            "n_fragments_dev_gt_0_05", "n_fragments_dev_gt_1", "by_kind", "dev_kcal",
                                            "source", "sha256", "rule")},
        reference_graphs={k: (br.get("reference_graphs") or {}).get(k) for k in ("ok", "n_checked", "n_vs_key_fail")},
        p_a={k: (br.get("P-a") or {}).get(k) for k in ("ok", "max_abs_gated", "n_rows")},
        tokens={k: tok.get(k) for k in ("token_version", "atom_version", "n_tokens", "n_x182_rows", "elements_same_as_g1",
                                        "t_report_sha256", "manifest_sha256", "q_report_sha256", "s_splits_sha256",
                                        "x182_sha256")},
        h_source=br.get("h_source"), g2_gates=br.get("g2_gates"), outputs=br.get("outputs"))


def d1g2_summary(info):
    cs = _json(C_SUMMARY_G2) or {}
    ov, dn = cs.get("overall") or {}, cs.get("same_ts_downstream") or {}
    c13 = [str(x) for x in (cs.get("decisions") or []) if str(x).startswith("C-13")]
    rule = cs.get("rule") or {}
    return dict(info, c_summary=dict(
        file=_disp(C_SUMMARY_G2), sha256=_sha(C_SUMMARY_G2),
        overall={k: ov.get(k) for k in ("g2_ok", "g2_eq_dft", "g2_eq_g1", "g1_ok_of_g2_ok")},
        downstream={k: dn.get(k) for k in ("column", "literal_column", "decision", "consumers", "classified",
                                           "literal_rule_for_reference", "level_shift_baseline_g1_vs_dft")},
        rule=dict(rmsd_A=rule.get("rmsd_A"), dform_A=rule.get("dform_A"), section=rule.get("section")),
        decision_c13=c13[0] if c13 else None))


def stage1_state():
    names = sorted(e.name for e in os.scandir(UNITS)) if UNITS.is_dir() else []
    s1 = [n for n in names if n.endswith(".json") and n.startswith(("L__dft__", "L__g1__"))]
    return dict(n_stage1_unit_json=len(s1), n_stage1_units=len(unit_names()),
                results={_disp(f): _sha(f) for f in STAGE1_RESULTS})


def supplement_content(hobj, pre):
    D = dict(date=time.strftime("%Y-%m-%d"), written=_now(), git_head=L.git("rev-parse", "HEAD").strip(),
             branch=_git_out("rev-parse", "--abbrev-ref", "HEAD"), add3_md_sha256=ADD3_MD_SHA256,
             add3_md_commit=_last_commit(ADD3_MD), base_json_sha256=ADD3_JSON_SHA256,
             base_json_commit=_last_commit(ADD3_JSON), generator_sha256=_sha(Path(__file__).resolve()),
             generator_git=_git_state_at(Path(__file__).resolve()), entries=pre["entries"], checked=pre["ok"],
             queue=pre.get("queue"))
    # PR-12 / PS-3: the extended json's gate content, in memory, for a stage-1 and a g2 process
    rec = dict(md_sha256=hobj.get("md_sha256"), inputs_sha256=dict(hobj.get("inputs_sha256") or {}))
    rec["inputs_sha256"].update({e["path"]: e["sha256"] for e in pre["entries"]})
    D["mirror"] = {"stage1": gate_mirror(rec, STAGE_RUNGS, need_paths()),
                   "stage2": gate_mirror(rec, STAGE2_RUNGS, need_paths(STAGE2_RUNGS))}
    bad = [f"{s}: {x}" for s, m in D["mirror"].items() for x in m["bad"]]
    if bad:
        raise Refuse(3, ["the extended json would not pass the trainers' gate_add3: " + "; ".join(bad[:10])])
    p = pre["p"]
    D["build"] = build_summary(pre["br"])
    D["pair_sets"] = pair_sets(p, SETS2)
    D["p_report"] = dict(file=_disp(P_REPORT), sha256=_sha(P_REPORT), git_state=_git_state_at(P_REPORT),
                         finished=p.get("finished"), git_head=p.get("git_head"), rungs_built=p.get("rungs_built"),
                         fu2_check_ok=(p.get("fu2_check") or {}).get("ok"), dft_vs_g1=p.get("dft_vs_g1"))
    D["h_changed_g2"] = (p.get("h_changed_by_rereferencing") or {}).get("g2")
    D["coverage"], D["same_ts"] = coverage_same_ts(p)
    D["c_tests"] = c_summary(pre["c"])
    D["c_g2_sub"] = pre["g2sub"]
    D["d1g2"] = d1g2_summary(pre["g2"])
    col = ((D["d1g2"].get("c_summary") or {}).get("downstream") or {}).get("column")
    if col != SAME_TS_COLS[0]:
        raise Refuse(3, [f"D1-G2 C_summary.json same_ts_downstream.column {col!r} != the primary split column "
                         f"{SAME_TS_COLS[0]!r} (PS-7 follows D1-G2 C-13; regenerate this rule)"])
    D["units"] = unit_names(STAGE2_RUNGS)
    D["unit_counts"] = ((hobj.get("ladder") or {}).get("units") or {}).get("counts") or {}
    D["stage1_state"] = stage1_state()
    recs = [H_PARQUET_G2, F_REPORT_G2, H_REPORT_G2, F_DECISION_G2, C_SUMMARY_G2, C_COMPARE_G2, G2_FU2 / "Q_manifest.csv",
            IN.build_report_path("g2"), P_REPORT, C_TESTS, L.RES / "C_tests_gpu.json", L.RES / "C_tests_cpu.json",
            HERE / "inputs.py", HERE / "tests" / "run_tests.py", HERE / "evaluate.py", Path(__file__).resolve()]
    D["records"] = {_disp(f): dict(sha256=_sha(f), git=_git_state_at(f) if Path(f).is_file() else "missing")
                    for f in recs}
    return D


def _decision_quote(txt):
    """the '- 결정:' bullet of F_DECISION.md (with its continuation lines), else the whole text."""
    out, on = [], False
    for line in (txt or "").splitlines():
        if line.startswith("- 결정"):
            on = True
            out.append(line[2:].strip())
            continue
        if on:
            if line.startswith("  ") and line.strip():
                out.append(line.strip())
                continue
            break
    return " ".join(out) if out else " ".join(x.strip() for x in (txt or "").splitlines() if x.strip())


def make_supp_md(D):
    E, B, PS, COV, ST, C, G = (D["entries"], D["build"], D["pair_sets"], D["coverage"], D["same_ts"], D["c_tests"],
                               D["d1g2"])
    uc, M1, M2 = D["unit_counts"], D["mirror"]["stage1"], D["mirror"]["stage2"]
    md = []
    a = md.append
    a(f"# PREREG ADD3 보완판 — D1PAIR 사전 등록 추가본 (D1-REF·LADDER 2단계: g2, {D['date']})")
    a("")
    a(f"이 문서는 `PREREG_D1PAIR_ADD3.md`(commit {D['add3_md_commit']}, sha256 `{D['add3_md_sha256']}`) 10절의 2단계 보완 "
      "규칙에 따라 새 파일로 쓴 보완판이다. ADD3 md는 한 바이트도 바꾸지 않았다(sha256과 git 상태를 다시 확인했다).")
    a(f"`prereg_d1pair_add3.json`(commit {D['base_json_commit']}, 보완 전 sha256 `{D['base_json_sha256']}`)에는 10절이 허용한 "
      f"것만 더했다: `inputs_sha256`·`inputs_roles`의 새 항목 {len(E)}개(10절 목록, 같은 순서)와 새 최상위 키 `supplement`(이 md의 "
      "경로·sha256·commit). 이미 있는 항목과 키는 그대로다(더한 것을 빼고 다시 쓰면 commit된 json과 바이트 단위로 같음을 쓰기 "
      "전과 쓴 뒤에 확인했다).")
    a(f"이 문서와 json 추가분은 `analysis/d1_ladder/prereg.py supplement`가 결과 파일만 읽어 만들었다(코드 commit "
      f"`{D['git_head']}`, 생성 코드 sha256 `{D['generator_sha256']}`, git 상태 {GIT_KO.get(D['generator_git'], D['generator_git'])}). "
      "결과 수치는 모두 결과 파일에서 옮겼고, 규칙은 명령서와 ADD3를 인용했다.")
    a("범위: **2단계(rung `g2`, 명령서 12절)**. 1단계 규칙은 바꾸지 않는다(7절).")
    a("")
    a("M2 학습(`train_atoms.py run --stage 2`, `train_xgb.py run --stage 2`)은 이 문서와 json 추가분이 commit된 뒤에만 돈다. "
      "두 학습 코드가 매 실행 전에 확인하는 것은 ADD3 md·json(`gate_add3`), `P_report.json`(MA-16), `C_tests.json`의 "
      "`g2_verified`(MA-17)다(6-2절). 학습 코드는 이 보완판을 읽지 않으므로, 이 보완판의 commit·sha256은 "
      "`prereg.py check --stage 2`와 E2가 확인한다.")
    a("")
    s1 = D["stage1_state"]
    have = {k: v for k, v in (s1.get("results") or {}).items() if v}
    a(f"**지금까지 본 것** (작성 {D['written']}):")
    a("- 2단계 학습은 한 번도 돌지 않았다(`$LAD/units`의 `L__g2__` 파일 0개).")
    a(f"- 1단계 학습: unit json {_n(s1.get('n_stage1_unit_json'))}/{_n(s1.get('n_stage1_units'))}개. "
      + ("1단계 예측·평가 파일(`L_predictions.parquet`, `L_report.json`, `E1_*`)은 아직 없다."
         if not have else "있는 1단계 결과 파일: " + ", ".join(f"`{k}`" for k in have)
         + ". 이 보완판이 새로 정한 것(2·3·4·5절)은 g2 구조·입력에 관한 결정이고 1단계 점수에 따라 고른 것이 아니다."))
    a("- g2에서 지금까지 계산한 것은 구조와 입력뿐이다: D1-G2 Phase G(TS), C(구조 비교, 같은 TS 판정), F(feature·토큰), "
      "H(HF), LADDER Phase P(g2 입력표, P_g2, P_012), Phase C 검사의 g2 항목(" + ", ".join(sorted(D["c_g2_sub"])) + "). "
      "명령서 8절의 smoke는 rung dft·g1이다. 이 명령은 g2 예측 파일을 열지 않으며 g2 시험 짝 점수는 이 문서에 없다.")
    a("")

    # 1
    a("## 1. 2단계가 더하는 입력 (g2 열)")
    a("")
    a("**gate에 더한 항목** (json `inputs_sha256`·`inputs_roles`; 열 = 명령서 4-1의 rung 열, `inputs.check_paths`):")
    a("")
    md.extend(_tab(["파일", "역할", "열", "sha256"], [[f"`{e['disp']}`", e["role"], e["column"], f"`{e['sha256']}`"]
                                                    for e in E]))
    a("")
    a("- g2 process의 `need`(학습 코드 `cmd_run`: 정답 파일 + 학습하는 rung의 입력표): "
      + ", ".join(f"`{_disp(x)}`" for x in M2["need"]) + ".")
    a(f"- 보완된 json으로 학습 코드 gate의 내용 조건을 지금 파일에 그대로 적용한 결과: g2 process는 {_n(M2['n_verified'])}개를 "
      f"확인하고(정답·공유 파일·코드·`P_012.csv`·g2 열) dft·g1 열에만 속한 {_n(M2['n_skipped'])}개는 열지 않는다(MA-15, L-4). "
      f"1단계 process(dft·g1)는 {_n(M1['n_verified'])}개를 확인하고 g2 열에만 속한 {_n(M1['n_skipped'])}개는 열지 않는다. "
      "`P_012.csv`는 세 rung에 걸친 파일이라 1단계 process도 확인한다(바뀌면 1단계 학습도 exit 3). 둘 다 통과. git 조건(두 "
      "파일이 commit되어 HEAD와 같음)은 commit 뒤에 참이 된다.")
    a("- 토큰 npz(반응별)의 sha256은 g2 `T_manifest.csv`에 있고 학습 코드가 읽을 때마다 확인한다(1단계와 같음).")
    a(f"- 코드: commit `{D['git_head']}`. 학습 코드·FU2 코드(두 학습 코드의 지문 `code_sha256`)는 ADD3 json 항목 그대로이고 g2 "
      "process의 gate도 그 sha256을 확인한다(위 결과에 포함).")
    a("")
    a("**기록만 하는 g2 관련 파일** (gate 밖; json `supplement.records`):")
    a("")
    md.extend(_tab(["파일", "sha256", "git"], [[f"`{k}`", f"`{v['sha256']}`", GIT_KO.get(v["git"], v["git"])]
                                             for k, v in D["records"].items()]))
    a("")
    t, pa, tk = B["table"], B["p_a"], B["tokens"]
    a(f"**g2 입력표** `$LAD/inputs/g2.parquet` (`P_build_g2.json` sha256 `{B['sha256']}`, 작성 {B.get('finished')}, git "
      f"{str(B.get('git_head'))[:10]}): 대상 {_n(t.get('n_targets'))}반응 중 토큰 {_n(t.get('n_tok'))}, 182 "
      f"{_n(t.get('n_x182'))}, H 9값 {_n(t.get('n_h'))}, 사용 가능 {_n(t.get('n_usable'))}. gate P-a(원래 기준으로 "
      f"`labels_d1_hf_g2.parquet` 재현): {'통과' if pa.get('ok') else '실패'}, 최대 차이 {pa.get('max_abs_gated')} "
      f"({_n(pa.get('n_rows'))}행). 토큰 `{tk.get('token_version')}` / `{tk.get('atom_version')}`, 원소 목록이 g1과 같음: "
      f"{tk.get('elements_same_as_g1')}. EDA 6채널은 재조립으로 바뀌지 않았다: {t.get('channels_unchanged')}.")
    hc = (B.get("h_changed") or {})
    bv = hc.get("by_value") or {}
    a(f"- H 재조립으로 바뀐 값(g2): 반응 {_n(hc.get('n_reactions'))}개(사용 가능 {_n(hc.get('n_reactions_usable'))}); "
      + ", ".join(f"{x} {_n((bv.get(x) or {}).get('n_changed'))}개(> 0.05: {_n((bv.get(x) or {}).get('n_gt_0_05'))})"
                  for x in CHANGED) + ".")
    a("")

    # 2
    r = B["references"]
    a("## 2. g2 H 재조립의 constrained dipole 기준 (P20, 확정)")
    a("- ADD3 10절은 명령서 Q-2 그대로(고정 이면각 상태를 키에 붙여 같은 상태끼리만 묶음, `--g2-q-keys`)를 적고, 그 상태가 "
      "D1(DFT) 기준 구조에서 정해지므로 g2 H에 DFT 정보가 일부 들어간다는 점(P20)을 이 보완판에서 확정·기록한다고 했다.")
    a("- **결정 (2026-10-23, 사용자 규칙 '모든 판단 = 추천안'; `inputs.py` P20, commit 04c9f7e9): 선택지 (c), `inputs.py "
      "build g2 --g2-cons-own`.** constrained dipole 키에 속한 g2 행은 기준을 바꾸지 않는다(반응마다 자기 G2 기준 구조의 HF "
      "에너지). 나머지 키는 명령서 4-1대로 키마다 g2 기준 구조 HF 에너지의 최저값을 쓴다(문턱 없음, 동점은 job_id가 작은 것).")
    a("- **이유:** Q-2 키의 `|fixed:` 접미사(고정 이면각 상태)는 D1(DFT) 기준 구조에서 계산된다. 이것으로 g2 기준을 묶으면 "
      "DFT에서만 얻는 정보가 g2 H에 들어간다(명령서 4-1: g2가 DFT 파일을 하나라도 읽으면 DFT 정보가 새는 것). 배포 때(SMILES → "
      "xTB)는 이 상태를 알 수 없으므로 반응마다 자기 G2 기준을 쓰는 것이 배포와 같다. 쓰지 않은 선택지: (a) 명령서 Q-2 키 "
      "그대로(DFT 유래 상태 사용), (b) g2만 SMILES 수준 키로 묶기(명령서에 없는 새 키).")
    a("- **한계 (이탈로 기록):** constrained 키에는 constrained 행만 들어 있으므로 `|fixed:` 상태 자체는 g2 값에 들어가지 "
      "않는다. 그러나 어느 g2 행을 그대로 둘지는 `$LAD/Q_keys.parquet`의 행별 `constrained` 표시로 고르고, 이 표시는 D1 라벨의 "
      "`ref_port.constrained`(D1, 곧 DFT 흐름의 기준 구조 결과)다. G2 자신의 기준 구조 결과(G2 job의 `ref_result.json` "
      "`port.constrained`)는 commit된 입력이 아니어서 읽지 않았다. 따라서 'constrained인가'라는 DFT 유래 정보 한 가지가 행 "
      "선택에 남는다: D1에서는 자유인데 G2에서는 constrained인 행은 자유 키의 최저값으로 바뀌고, 반대 경우는 자기 기준을 "
      "유지한다. 두 표시가 다른 g2 행의 수는 세지 않았다(G2 job 폴더를 읽어야 함).")
    dv = r.get("dev_kcal") or {}
    dk = r.get("dev_kcal_kept_own") or {}
    a(f"- 기록(`P_build_g2.json`): `g2_q_keys` = `{B.get('g2_q_keys')}`, 지문 `g2_q_keys` = `{B.get('fingerprint_g2_q_keys')}`, "
      f"`references.g2_mode` = `{r.get('g2_mode')}`. constrained 키의 기준 행 {_n(r.get('n_rows_constrained_keys'))}개 중 키 "
      f"최저값이 더 낮았지만 자기 기준을 유지한 조각 {_n(r.get('n_constrained_kept_own'))}개(키 최저값과의 차 중앙값 "
      f"{_f(dk.get('p50'))}, 최대 {_f(dk.get('max'), 2)} kcal/mol), constrained 조각 교체 "
      f"{_n(r.get('n_constrained_replaced'))}개. 바뀐 조각 {_n(r.get('n_fragments_replaced'))}개(반응 "
      f"{_n(r.get('n_jobs_replaced'))}개; " + ", ".join(f"{x} {_n(y)}" for x, y in (r.get("by_kind") or {}).items())
      + f"), 바뀐 양 중앙값 {_f(dv.get('p50'))}, 최대 {_f(dv.get('max'), 2)} kcal/mol; 분자 키 {_n(r.get('n_keys'))}개. "
      f"기준 구조 그래프 검사(P6b): {(B.get('reference_graphs') or {}).get('ok')}.")
    a("- 이 결정은 g2 입력표를 만들기 전에 정했다(g2 학습·평가 결과 없음).")
    a("")

    # 3
    fd = G.get("f_deviation") or {}
    gf, cn = fd.get("gate_failure") or {}, fd.get("counts") or {}
    comp = gf.get("components") or {}
    a("## 3. D1-G2 Phase F 멈춤의 수용 (이탈)")
    a(f"- `analysis/d1_g2/results/F_report.json`(sha256 `{(G.get('F_report.json') or {}).get('sha256')}`): G2 ok "
      f"{_n(cn.get('g2_ok'))}반응 중 전역 182 {_n(cn.get('features_ok'))}, Q 사용 {_n(cn.get('q_use'))}, 토큰 "
      f"{_n(cn.get('tokens_ok'))}. gate 실패 {_n(gf.get('n'))}/{_n(gf.get('n_g2_ok'))} = {_pct(gf.get('rate'), 2)} > "
      f"{_pct(gf.get('threshold'), 0)} → stop (D1-G2 명령서 14절 F; 구성 feature {_n(comp.get('features'))}, Q 제외 "
      f"{_n(comp.get('Q_excluded'))}, T 실패 {_n(comp.get('T_bad'))}).")
    a(f"- 결정 `{fd.get('decision')}`(sha256 `{fd.get('decision_sha256')}`, commit {fd.get('decision_commit')}) 원문: "
      f"“{_decision_quote(fd.get('decision_text'))}”")
    a("- 명령서 16절 '2단계: D1-G2 F·H gate 실패 → 그쪽 해결까지 대기'에서 벗어난다. D1-G2의 결정으로 해결된 것으로 보고 "
      "진행하며 이탈로 기록한다. 실패 반응은 g2에서 토큰·182가 없어 사용 가능 반응이 아니다(명령서 7절). 그 반응이 들어간 짝은 "
      "P_g2에서 빠지고, 4절 적용 범위에서 따로 센다(`no_tokens_or_182`).")
    fa = ((B.get("g2_gates") or {}).get("F_stop_accepted") or {})
    a("- g2 입력표는 `inputs.py build g2 --accept-g2f-stop --g2-cons-own`(commit 11aeef90·04c9f7e9; `jobs/p_stage2.sh`, "
      "cdbaa3ec)으로 만들었다. `--accept-g2f-stop`은 멈춤 이유가 F gate 비율뿐이고 `F_DECISION.md`가 commit되어 있을 때만 "
      f"이 멈춤을 통과시키고 `P_build_g2.json` `g2_gates.F_stop_accepted`에 그 파일의 sha256(`{fa.get('decision_sha256')}`)을 "
      "남긴다. 이 명령은 그 값이 지금 파일의 sha256과 같음을 확인했다(`F_DECISION.md`는 build 지문에 들어 있지 않음).")
    hi = G.get("H_report.json") or {}
    a(f"- D1-G2 `H_report.json`: complete {hi.get('complete')}, stop {hi.get('stop')}; `labels_d1_hf_g2.parquet` sha256 = "
      f"H_report outputs의 값(경로 키 `{(G.get('H_report_output_key') or ['–'])[0]}`, 저장소 상대 경로로 맞춤, inputs.py P21). "
      f"D1-G2 파일 {len(G2_COMMITTED)}개 commit됨(HEAD와 같음). `origin/d1-g2`가 HEAD에 들어 있음: "
      f"{G.get('origin_d1g2_in_head')}.")
    a(f"- `inputs.py build g2`의 D1-G2 확인 기록(`P_build_g2.json` `g2_gates`): "
      f"`{json.dumps(_jsonable(B.get('g2_gates')), ensure_ascii=False)[:600]}`.")
    a("")

    # 4
    a("## 4. 짝 집합과 적용 범위")
    rows = []
    for name, s in PS.items():
        f = s.get("folds_T1") or {}
        rows.append([name, _n(s["n"]), _n((s.get("by_ref_kind") or {}).get("reference")),
                     _n((s.get("by_ref_kind") or {}).get("anchor")), _n(s.get("para")), _n(s.get("in_P_main")),
                     _n(s.get("in_P_main_para")),
                     " / ".join(_n((f.get(str(i)) or {}).get("test")) for i in range(NF)),
                     f"{_n((s.get('dropped') or {}).get('n'))} ({_n((s.get('dropped') or {}).get('n_used_in_T1'))})",
                     f"`{s['sha256']}`"])
    md.extend(_tab(["집합", "짝", "기준", "anchor", "para", "P_main", "P_main para", "시험 짝 fold 0–4", "빠진 짝 (T1에 쓰인 것)",
                    "sha256"], rows))
    a("")
    a("- 판정(13-2)은 g2 자기 집합 `P_g2`, rung 사이 비교(13-4)는 `P_012` = P_01 ∩ P_g2에서 한다(ADD3 4절·5-4).")
    a("- 학습 코드 규칙의 짝 목록 sha256(지문 `pair_list`와 비교할 값): "
      + ", ".join(f"{n} `{PS[n]['pair_list_sha256_trainer_rule'][:16]}…` ({_n(PS[n]['n_in_T1_lists'])}짝)"
                  for n in ("P_g2",)) + ".")
    a(f"- **적용 범위 (명령서 12절, 정의 학습 전 고정):** T1 fold 0–4 시험 목록(FU2 split 파일 `test`)의 짝 중 P_main 짝"
      f"(`S_folds.csv` `in_P_main`) {_n(COV['n_P_main_test'])}짝(para {_n(COV['n_P_main_test_para'])}) 가운데 그 집합에 "
      "든 비율. g2에서 실패한 반응은 배포 때도 예측할 수 없으므로 P_g2 밖의 짝은 '예측 불가'로 세고, g2 성적(13-2)에는 "
      "넣지 않는다(빈 예측을 null 등으로 채우지 않음).")
    a("")
    rows = []
    for n, x in COV["sets"].items():
        rows.append([n, _n(x["n"]), _pct(x["frac"]), _n(x["n_para"]), _pct(x["frac_para"]),
                     " / ".join(f"{_n(v['n'])}/{_n(v['of'])}" for v in x["by_fold"].values())])
    md.extend(_tab(["집합", "P_main 시험 짝", "비율", "para", "para 비율", "fold 0–4"], rows))
    a("")
    mr = COV["P_g2_missing_by_reason"]
    a(f"- P_g2에 없는 P_main 시험 짝 {_n(sum(mr.values()))}개(para {_n(COV['P_g2_missing_para'])})의 이유(두 반응 중 먼저 걸리는 "
      "것): " + ", ".join(f"{DROP_KO[x]} {_n(mr.get(x))}" for x in DROP_CATS) + ".")
    a("")

    # 5
    sc = (G.get("c_summary") or {})
    ov, dn, ru = sc.get("overall") or {}, sc.get("downstream") or {}, sc.get("rule") or {}

    def kn(x):
        x = x or {}
        return f"{_n(x.get('k'))}/{_n(x.get('n'))} ({_pct(x.get('frac'), 2)})"
    a("## 5. 같은 TS에 따른 나눔 (E2, 설명용)")
    a("- 명령서 12절: g2는 G2가 라벨과 다른 TS를 찾은 짝도 그대로 학습·평가한다(배포 때 구별할 수 없다). 나눔은 설명용이고 "
      "13-2 판정, 13-3 해석, REP 고르기에 쓰지 않는다.")
    a(f"- **나눔 규칙 (학습 전 고정):** 반응별 판정 = D1-G2 Phase C `{ST['source']}`(sha256 `{ST['source_sha256']}`)의 "
      f"`{ST['primary']}`: {SAME_TS_KO[ST['primary']]}, {ru.get('section') or 'VALIDATION_SPEC 6.4'} 규칙(무거운 원자 RMSD ≤ "
      f"{ru.get('rmsd_A')} Å, 형성 결합 거리 차 ≤ {ru.get('dform_A')} Å). 짝 (a, b)는 두 반응이 모두 같은 TS면 `same`, "
      "하나만 같으면 `mixed`, 둘 다 다르면 `different`, 한쪽이라도 판정이 비면 `unclassified`.")
    a(f"- **이 열을 쓰는 이유:** 명령서 12절은 'D1-G2 Phase C의 \"같은 TS\" 판정'을 따르라고 한다. D1-G2 Phase C는 결정 "
      f"C-13으로 그 판정(쓰는 곳에 'D1LADDER 12 E2 (same TS split)'가 적혀 있음)을 `{dn.get('column')}`로 정했다. "
      f"`same_ts_dft`는 xTB·DFT 수준 차이를 넘어 비교하므로 같은 TS도 대부분 '다름'으로 판정한다(같은 TS를 xTB로 옮긴 G1 "
      f"대 DFT 기준선 {kn(dn.get('level_shift_baseline_g1_vs_dft'))}, G2 = DFT TS {kn(ov.get('g2_eq_dft'))}).")
    a("- **E2 보고:** P_g2 시험 짝(fold 0–4 합) 중 P_main 짝을 이 네 부류로 나눠, 부류마다 REP·ENS4·4 모델·LINE·null의 MAE"
      "(9 주 타깃; 전체 위치와 para)를 적는다. 부류 사이 검정이나 기준 판정은 하지 않는다.")
    a(f"- **두 번째 나눔:** `{SAME_TS_COLS[1]}`({SAME_TS_KO[SAME_TS_COLS[1]]}; 2단계 작업 지시 문구에 적힌 열)도 같은 방식으로 "
      "네 부류로 나눠 함께 보고한다. 둘 다 설명용이고 어느 것도 판정에 쓰지 않는다.")
    a(f"- D1-G2 Phase C 비율(`C_summary.json` `overall`): G2 = DFT TS {kn(ov.get('g2_eq_dft'))}, G2 = G1 "
      f"{kn(ov.get('g2_eq_g1'))}.")
    a("")
    rows = []
    for col in SAME_TS_COLS:
        cl = ST["classes"][col]
        rows.append([f"`{col}`"] + [f"{_n(cl['all'][k])} ({_n(cl['para'][k])})" for k in SAME_TS_CLASSES])
    md.extend(_tab(["반응 판정 열", "same", "mixed", "different", "unclassified"], rows))
    a("")
    ru2 = ST["reactions_usable"]
    a(f"(P_g2의 P_main 시험 짝 {_n(ST['n_pairs'])}개, 괄호 = para {_n(ST['n_pairs_para'])}개. g2 사용 가능 반응의 판정: "
      + "; ".join(f"`{c}` 같음 {_n(v['same'])} / 다름 {_n(v['different'])} / 없음 {_n(v['unclassified'])}"
                  for c, v in ru2.items()) + ". 부류 크기는 구조만으로 정해지는 수이고 점수가 아니다.)")
    a("")

    # 6
    a("## 6. M2 학습 단위와 gate")
    a(f"- 단위 `L__g2__<arm>__<model>__f<k>` = g2 × S·SH × AW·GNN·TF·XGB × fold 0–4 = **{len(D['units'])} unit**(원자 "
      f"{sum(1 for u in D['units'] if u.split('__')[3] in ATOM_MODELS)} + XGB "
      f"{sum(1 for u in D['units'] if u.split('__')[3] == 'XGB')}; ADD3 json `ladder.units.counts.stage2` = "
      f"{uc.get('stage2')}), GPU 학습 {uc.get('gpu_stage2')}회(원자 unit마다 튜닝 칸 수 + 최종 seed "
      f"{uc.get('n_final_seeds')}개).")
    a("- 명령: `train_xgb.py run <i> <n> --stage 2` → `train_xgb.py check --stage 2` → `train_atoms.py run <i> <n> --stage 2` → "
      "`train_atoms.py collect --stage 2 --with-xgb`(1단계 unit까지 모아 `L_predictions.parquet`).")
    a("")
    a("### 6-1. 학습 규칙 — ADD3 3절 그대로")
    a("- arm·모델·grid·β 규칙·SH 학습 규칙·학습 상수·분할(T1 fold 0–4)·검증 짝(`inner_val`, L-9)·gate M-1′·캐시 지문은 ADD3 "
      "3절과 같다. unit 지문 `add3`는 ADD3 md의 sha256(바뀌지 않음)이고, 이 보완판의 sha256은 지문에 들어가지 않는다.")
    a("")
    a("### 6-2. 매 실행 전 gate (학습 코드 원문, `train_atoms.py` = `train_xgb.py`; 두 파일의 sha256도 gate가 확인)")
    a("")
    a("```python")
    for x in GATE_QUOTE:
        a(x)
    a("```")
    a("")
    a("- 2단계 process: `rs` = [`g2`], `need` = 정답 파일 + `$LAD/inputs/g2.parquet`. 보완 전 json에는 g2 입력표가 없어 "
      "`lack`이 생기므로 2단계 학습은 이 보완 commit 전에는 exit 3이다(명령서 16절 A).")
    a("")

    # 7
    a("## 7. 1단계 규칙은 모두 그대로 적용한다")
    r2 = STAGE2_RULE
    a("- ADD3의 1단계 규칙이 g2에 바뀌지 않고 적용된다: 3절(rung 열 분리 L-4, arm S·SH, β 규칙, SH 학습 규칙, 모델 네 개와 "
      f"grid, 학습 상수, 짝·분할·검증 짝, 캐시 지문), 5절(REP·ENS4, 13-2 기준 {'·'.join(r2['targets'])} 모두 para MAE ≤ "
      f"{r2['para_mae_max']}·민감도 ≥ {r2['bend_sensitivity_min']}·위양성률 ≤ {r2['false_positive_rate_max']}, 꺾임 모집단, "
      "단계 0·1 설명용), 6절(지표), 11절(멈춤 규칙). 결과를 본 뒤 이 규칙을 바꾸지 않는다.")
    a("- g2에 맞춰 읽는 곳(ADD3가 이미 적은 것): 13-2 판정 집합 = P_g2(json `decision_rules.stage2.set`); 13-3 해석 표 전체"
      "(a·b·c)를 1단계 dft 판정과 함께 적용; 13-4 rung 사이 비교 = P_012; E2 = 명령서 11절 지표 + P_012 비교(명령서 12절) + "
      "적용 범위(4절) + 같은 TS 나눔(5절).")
    a("- 이 보완판이 새로 정한 것은 2절(P20), 3절(F 이탈 수용), 4절의 적용 범위 정의, 5절의 나눔 규칙뿐이다. 모두 g2 학습 전, "
      "g2 점수 없이 정했다.")
    a("- E2는 시작할 때 이 보완판이 commit되어 있고 sha256이 json `supplement.md_sha256`과 같은지 확인해야 한다"
      "(`prereg.py check --stage 2`가 같은 확인과 학습 코드 gate의 내용 조건을 확인한다).")
    a("")

    # 8
    a("## 8. Phase C 결과 (g2 포함)")
    a(f"`{C['file']}` sha256 `{C['sha256']}` (commit {C.get('last_commit')}), tests `{C.get('tests_version')}`, merge "
      f"{C.get('merged')}: all_pass {C.get('all_pass')}, 범위 \"{C.get('all_pass_scope')}\", stage_verified "
      f"{C.get('stage_verified')}, g2_verified {C.get('g2_verified')}, 정답 `{_disp((C.get('labels') or {}).get('path') or '')}`.")
    a("")
    md.extend(_tab(CHECKS, [[C["checks"].get(x) for x in CHECKS]]))
    a("")
    a("- g2 하위 결과: " + ", ".join(f"{k} {v}" for k, v in D["c_g2_sub"].items()) + ".")
    parts = C.get("parts") or {}
    a("- 검사 job: " + "; ".join(f"{q} job {(v or {}).get('slurm_job')} ({(v or {}).get('host')}, git "
                                 f"{str((v or {}).get('git_head'))[:8]})" for q, v in parts.items()) + ".")
    a("")

    # 9
    a("## 9. 계획과 다르게 한 점·해석 (2단계)")
    a("1. **P20:** ADD3 10절은 명령서 Q-2 그대로(`--g2-q-keys`)를 적었고 확정을 이 보완판에 미뤘다. 선택지 (c)로 확정했다(2절). "
      "한계: 그대로 둘 행은 D1 라벨의 `ref_port.constrained`(DFT 흐름) 표시로 고른다(2절 '한계').")
    a("2. **D1-G2 Phase F 멈춤 수용:** 명령서 16절 '2단계 대기'에서 벗어난다(3절, D1-G2 `F_DECISION.md`).")
    a(f"3. **같은 TS 판정 열:** 명령서 12절이 따르라고 한 D1-G2 Phase C 판정(C-13)대로 `{SAME_TS_COLS[0]}`을 나눔 규칙으로 "
      f"두었다. 2단계 작업 지시 문구는 `{SAME_TS_COLS[1]}`였으나 이 열은 수준 차이 때문에 같은 TS도 거의 모두 '다름'으로 "
      "판정하므로 두 번째 설명용 나눔으로 함께 보고한다(5절).")
    a("4. **Phase C 먼저 commit:** 1단계(명령서 8절, ADD3 PR-4)와 같이 g2를 포함한 `C_tests.json`과 Phase C 코드 5개가 commit된 "
      "상태에서만 이 보완판을 썼다.")
    a("")

    # 10
    a("## 10. 멈춤 규칙 (명령서 16절, 2단계)")
    a("- **A:** 이 보완판·json 추가분 commit 전 2단계 학습 시도(`gate_add3`가 exit 3: json에 g2 입력표가 없거나 json이 HEAD와 "
      "다름).")
    a(f"- **2단계:** D1-G2 F·H gate 실패 → 그쪽 해결까지 대기(F의 {_pct(gf.get('rate'), 2)} 멈춤은 3절의 이탈로 수용; 다른 "
      "F·H 멈춤은 대기).")
    a("- **M·E·원칙:** ADD3 11절 그대로(gate M-1′, 캐시 지문, 시험 짝 목록, 결과를 본 뒤 규칙 변경 금지).")
    a("")
    return "\n".join(md) + "\n"


def supp_record(D, md_sha):
    """the json 'supplement' value (new top-level key; PR-5)."""
    C = D["c_tests"]
    return dict(
        md=_disp(SUPP_MD), md_sha256=md_sha, commit=D["git_head"],
        commit_meaning="HEAD when the supplement was written (the code and committed inputs it was generated from); the "
                       "md and this json are committed together right after: git log -1 --format=%H -- "
                       + _disp(SUPP_MD),
        written=D["written"], date=D["date"], code_version=SUPP_VERSION, stage=2, rungs=list(STAGE2_RUNGS),
        generator=dict(path=_disp(Path(__file__).resolve()), sha256=D["generator_sha256"], git_state=D["generator_git"]),
        add3_md_sha256=D["add3_md_sha256"], add3_md_commit=D["add3_md_commit"], base_json_sha256=D["base_json_sha256"],
        base_json_commit=D["base_json_commit"],
        entries_added={e["path"]: dict(role=e["role"], column=e["column"], sha256=e["sha256"]) for e in D["entries"]},
        gate=dict(code=["analysis/d1_ladder/train_atoms.py gate_add3 (lines 796-832), cmd_run 1057-1062, "
                        "gate_c_stage2 859-865", "analysis/d1_ladder/train_xgb.py gate_add3 (lines 851-886), cmd_run "
                        "1115-1120, gate_c_stage2 911-917"],
                  keys_read=["md_sha256", "inputs_sha256"], need_stage2=D["mirror"]["stage2"]["need"],
                  mirror={s: {k: m[k] for k in ("ok", "rungs", "need", "n_verified", "n_skipped", "skipped")}
                          for s, m in D["mirror"].items()},
                  rule="stage-2 process: rs = ['g2'], need = label file + $LAD/inputs/g2.parquet; files of the dft / g1 "
                       "columns only are not opened (MA-15); MA-16 P_report; MA-17 C_tests all_pass + g2_verified"),
        g2_q_keys="own", p20=dict(option="c", flag="inputs.py build g2 --g2-cons-own", decision_commit="04c9f7e9",
                                  references=D["build"]["references"],
                                  limitation="kept rows chosen by Q_keys 'constrained' = D1 label ref_port.constrained "
                                             "(D1 / DFT pipeline), not G2's own port result; the '|fixed:' state itself "
                                             "never enters a g2 value; rows where the two flags differ not counted"),
        f_stop_build=dict(flag="inputs.py build g2 --accept-g2f-stop", commit="11aeef90",
                          job="jobs/p_stage2.sh (cdbaa3ec)",
                          accepted=(D["build"].get("g2_gates") or {}).get("F_stop_accepted")),
        queue_check=D.get("queue"),
        f_deviation={k: v for k, v in (D["d1g2"].get("f_deviation") or {}).items() if k != "decision_text"},
        d1g2={k: v for k, v in D["d1g2"].items() if k != "f_deviation"},
        coverage=D["coverage"], same_ts_split=D["same_ts"],
        units_stage2=D["units"], unit_counts={k: D["unit_counts"].get(k) for k in ("stage2", "gpu_stage2")},
        pair_sets={n: {k: s.get(k) for k in ("path", "sha256", "n", "in_P_main", "in_P_main_para", "n_in_T1_lists",
                                             "pair_list_sha256_trainer_rule")} for n, s in D["pair_sets"].items()},
        build={k: v for k, v in D["build"].items() if k not in ("references",)}, p_report=D["p_report"],
        h_changed_g2=D["h_changed_g2"],
        c_tests=dict(file=C.get("file"), sha256=C.get("sha256"), committed=C.get("committed"),
                     last_commit=C.get("last_commit"), all_pass=C.get("all_pass"), status=C.get("status"),
                     stage_verified=C.get("stage_verified"), g2_verified=C.get("g2_verified"),
                     all_pass_scope=C.get("all_pass_scope"), checks=C.get("checks"), g2_sub=D["c_g2_sub"],
                     code_sha256=C.get("code_sha256")),
        stage1_state=D["stage1_state"], records=D["records"],
        decision_rules_stage2=dict(set="P_g2 (13-2, own set)", interpretation="13-3 a / b / c with the stage-1 dft verdict",
                                   comparisons="13-4 on P_012", e2="spec 11 metrics + P_012 comparison (spec 12) + "
                                   "coverage + same-TS split (descriptive)", stage1_rules="ADD3 sections 3, 5, 6, 11 unchanged"),
        requirements_checked=D["checked"], decisions=SUPP_DECISIONS,
        env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
                 slurm_job=os.environ.get("SLURM_JOB_ID"), prefix=sys.prefix))


def make_supp_json(head, D, md_sha):
    """the committed json + the 8 entries + 'supplement' -> (text, record); PS-2 invariants checked in memory."""
    base_txt = head.decode("utf-8")
    base = json.loads(base_txt)
    if json.dumps(base, indent=1) != base_txt:
        raise Refuse(3, [f"the committed {ADD3_JSON.name} does not re-serialise byte for byte (json.dumps indent 1): it "
                         "cannot be extended without rewriting existing entries"])
    obj = json.loads(base_txt)
    paths = [e["path"] for e in D["entries"]]
    for e in D["entries"]:
        obj["inputs_sha256"][e["path"]] = e["sha256"]
        obj["inputs_roles"][e["path"]] = dict(role=e["role"], column=e["column"])
    obj["supplement"] = _jsonable(supp_record(D, md_sha))
    txt = json.dumps(obj, indent=1)
    back = json.loads(txt)
    bad = []
    if json.dumps(_without_supplement(back, paths), indent=1) != base_txt:
        bad.append("removing the additions does not give the committed json byte for byte")
    if list(back)[:len(base)] != list(base) or set(back) - set(base) != {"supplement"}:
        bad.append("top-level keys changed (only 'supplement' may be added, last)")
    for k in ("inputs_sha256", "inputs_roles"):
        if list(back[k])[:len(base[k])] != list(base[k]) or list(back[k])[len(base[k]):] != paths:
            bad.append(f"{k}: existing entries reordered or the additions differ from md section 10")
    if any(back[k] != base[k] for k in base if k not in ("inputs_sha256", "inputs_roles")) or \
            any(back[k][q] != base[k][q] for k in ("inputs_sha256", "inputs_roles") for q in base[k]):
        bad.append("an existing entry changed")
    if bad:
        raise Refuse(3, bad)
    return txt, back


def _supp_selfcheck(txt, head, D):
    """PR-12 / PS-2: the files read back as the gate reads them -> list of problems."""
    back = ADD3_JSON.read_bytes()
    bad = [] if back == txt.encode() else ["the json on disk differs from the text written"]
    r2 = json.loads(back.decode("utf-8"))
    if (r2.get("supplement") or {}).get("md_sha256") != L.sha256(SUPP_MD):
        bad.append("supplement.md_sha256 != sha256(md) after writing")
    if json.dumps(_without_supplement(r2, [e["path"] for e in D["entries"]]), indent=1).encode("utf-8") != head:
        bad.append("the json without the additions != the committed json")
    for s, rungs in (("stage 1", STAGE_RUNGS), ("stage 2", STAGE2_RUNGS)):
        bad += [f"{s}: {x}" for x in gate_mirror(r2, rungs, need_paths(rungs))["bad"]]
    return bad


def _restore_committed(head):
    """PS-1: the committed json bytes back on disk (= HEAD) and the new, untracked supplement md removed -> message."""
    js, md, err = _disp(ADD3_JSON), _disp(SUPP_MD), []
    ok = False
    try:
        if not (ADD3_JSON.is_file() and ADD3_JSON.read_bytes() == head):
            L.write_atomic(ADD3_JSON, head)
        ok = ADD3_JSON.read_bytes() == head
    except Exception as e:                                       # noqa: BLE001
        err.append(f"json: {type(e).__name__}: {e}")
    try:
        if SUPP_MD.is_file() and not _tracked(SUPP_MD):
            SUPP_MD.unlink()
    except Exception as e:                                       # noqa: BLE001
        err.append(f"md: {type(e).__name__}: {e}")
    if ok and not err:
        return f"{js} restored to the committed bytes (= HEAD) and {md} removed: nothing to commit, the gates pass again"
    return (f"RESTORE INCOMPLETE ({'; '.join(err) or 'json != HEAD bytes'}): run at once on the login node: cd {L.REPO} "
            f"&& git checkout HEAD -- {js} && rm -f {md}")


def cmd_supplement():
    try:
        mode, head, disk = supplement_base()
    except Refuse as e:
        print(f"{'REFUSED' if e.code == 3 else 'NOT READY'} (exit {e.code}), nothing written:\n  " + "\n  ".join(e.reasons),
              file=sys.stderr, flush=True)
        return e.code
    if mode == "frozen":
        print("[supplement] the committed ADD3 json already carries the supplement: frozen (PS-10), nothing rewritten; "
              "running check --stage 2", flush=True)
        return cmd_check(2)
    if mode == "pending":
        print(f"[supplement] note: {SUPP_MD.name} / {ADD3_JSON.name} of an earlier supplement run are on disk but not "
              "committed; both are regenerated from the committed json (PS-10)", flush=True)
    try:
        hobj = json.loads(head.decode("utf-8"))
        pre = supplement_preconditions(hobj)
        D = supplement_content(hobj, pre)
        md = make_supp_md(D)
        md_sha = hashlib.sha256(md.encode()).hexdigest()          # = sha256 of the file write_atomic writes
        txt, rec = make_supp_json(head, D, md_sha)                # everything computed before the first write
    except Refuse as e:
        print(f"{'REFUSED' if e.code == 3 else 'NOT READY'} (exit {e.code}), nothing written:\n  " + "\n  ".join(e.reasons),
              file=sys.stderr, flush=True)
        if mode == "pending":
            print(f"  note: {ADD3_JSON.name} on disk is still the uncommitted supplement of an earlier run (every starting "
                  f"trainer run exits 3): commit it with its md, or restore: git checkout HEAD -- {_disp(ADD3_JSON)}; "
                  f"rm -f {_disp(SUPP_MD)}", file=sys.stderr, flush=True)
        return e.code
    if ADD3_JSON.read_bytes() != disk or _head_bytes(ADD3_JSON) != head:
        print(f"STOP: {ADD3_JSON.name} or HEAD changed while the supplement was computed; nothing written (rerun)",
              file=sys.stderr, flush=True)
        return 3
    hit, _ = window_tasks()                                      # PS-11 again, right before the window opens
    if hit:
        print(f"NOT READY (exit 2), nothing written: {len(hit)} pending task(s) named {list(WINDOW_JOBS)} appeared while "
              f"the supplement was computed: {hit[:5]} (PS-11)", file=sys.stderr, flush=True)
        return 2
    md_rel, js_rel = _disp(SUPP_MD), _disp(ADD3_JSON)
    print(f"[supplement] writing {SUPP_MD.name}, then {ADD3_JSON.name}: from the json write until its commit every "
          "trainer run that starts exits 3 at gate_add3", flush=True)
    L.write_atomic(SUPP_MD, md)                                  # md first: no gate reads it
    if L.sha256(SUPP_MD) != md_sha:
        print(f"STOP: the written md sha256 differs from the sha256 of its text; the json was not written ("
              f"{_restore_committed(head)})", file=sys.stderr, flush=True)
        return 3
    try:
        L.write_atomic(ADD3_JSON, txt)                           # json last, one os.replace
        bad = _supp_selfcheck(txt, head, D)
    except Exception as e:                                       # noqa: BLE001  (PS-1: never leave the json modified)
        bad = [f"{type(e).__name__}: {e}"]
    if bad:
        print("STOP (exit 3): self-check after writing failed: " + "; ".join(bad) + "\n  " + _restore_committed(head),
              file=sys.stderr, flush=True)
        return 3
    branch = D.get("branch") or "d1-ladder"
    if D.get("generator_git") != "committed":                    # the generator goes into the same commit (stage-1 rule)
        js_rel += " " + _disp(Path(__file__).resolve())
    print(f"[supplement] {SUPP_MD.name} ({md_sha[:16]}) + {ADD3_JSON.name}: +{len(D['entries'])} inputs entries, key "
          f"'supplement'; gate mirror: stage 1 {D['mirror']['stage1']['n_verified']} verified / "
          f"{D['mirror']['stage1']['n_skipped']} skipped, stage 2 {D['mirror']['stage2']['n_verified']} verified / "
          f"{D['mirror']['stage2']['n_skipped']} skipped; code commit {D['git_head'][:10]}.", flush=True)
    print("COMMIT BOTH NOW (login node). Until this commit the json differs from HEAD, so every train_atoms.py / "
          "train_xgb.py run (any stage) and evaluate.py e1 that STARTS exits 3 at its ADD3 gate ('modified since its "
          "commit'); running processes are not affected (the gate runs once at start); a task that hit it wrote no unit: "
          "resubmit it after the commit.\n"
          f"  cd {L.REPO}\n"
          f"  git add {md_rel} {js_rel}\n"
          f"  git commit -m \"D1LADDER Phase A stage 2: PREREG ADD3 supplement (g2; prereg.py supplement, md sha256 "
          f"{md_sha[:16]})\" -m \"Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\" -- "
          f"{md_rel} {js_rel}\n"
          f"  git push origin {branch}\n"
          f"then: python prereg.py check --stage 2 (sbatch) before M2.", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("write", "check", "supplement"))
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2), help="check: stage 1 (dft, g1) or 2 (g2)")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        return cmd_write()
    if a.cmd == "supplement":
        return cmd_supplement()
    return cmd_check(a.stage)


if __name__ == "__main__":
    sys.exit(main())
