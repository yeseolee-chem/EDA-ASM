#!/usr/bin/env python3
r"""D1-REF·LADDER Phase G0 (docs/specs/D1LADDER.md 5) and Phase F0 (6): the DFT-rung structure folders and the xTB
inputs computed on them. sbatch only; nothing in this file was run when it was written (login node), so every path is
untested until it runs inside a job.

G0  $LAD/dft_final/<rxn_id>/ for the 9,266 targets (lad_common.targets(); each must be D1-HF geom=dft ok), in the G1
    folder format the feature and FU2 code read (= d1_pair/g1_final/<rxn_id>/). Readers checked: xtb_slice.geom_marker /
    geom_files / ts_fragments (.done, ts.xyz, rel1.xyz, rel2.xyz), ext_features.block_b6 (result.json imag_freqs_cm,
    imag_mode_forming_share + work/ts.hess), ext_features.inputs_fp (bytes of ts.xyz, rel1.xyz, rel2.xyz, result.json),
    d1_features.root_ids (.done / .fail_*, all-digit folder names), fu2/atoms.py (ts_fragments only). work/ts.out is
    read only by d1_g1.py (G audits) and r6_recover.py, neither of which runs here, so it is kept gzipped.
      ts.xyz         D1-HF dft EDA input coordinates (d1_hf/inputs/<T>/hf/eda.inp via hf_common.parse_inp), label order
      rel1.xyz       d1_hf/inputs/<T>/hf/frag1_rel.inp coordinates (frag1 = role1 = dipole)
      rel2.xyz       d1_hf/inputs/<T>/hf/frag2_rel.inp coordinates (frag2 = role2 = dipolarophile)
      work/ts.hess   ORCA 6.1.1 '! XTB2 ALPB(water) Freq' at the fixed label TS (d1_g2/g2_autode.FREQ_HEADER with nprocs 1),
                     charge = label charge1 + charge2, mult 1; the g2_autode.finalize path: orca_direct.write_inp / run /
                     imag_and_mode, run_ts.forming_share (D1 code copy, imported through g2_autode, not copied)
      work/ts.inp, work/ts.out.gz
      result.json    imag_freqs_cm (every negative frequency, ascending), n_imag, imag_mode_forming_share (most negative
                     mode, the one ext_features.block_b6 takes; null when n_imag = 0), lowest_freq_cm, ... (the G1 keys that
                     apply: charge1/2, ts_header, orca_bin, e_ts_eh, ts_run_hours, hess_geom_max_dev_A, hess_shift_A,
                     formed_pairs_label, d_form_dft, status, fail_reasons, wall_s, core_h)
      .done | .fail_<reason>   written last inside $LAD/dft_final/<rxn_id>.tmp, which is then os.replace'd to <rxn_id>
    No top-level ts.hess (g1_final has none; the g2_final copy is a D1-G2 addition) and no other ORCA by-product (quota).
    Gates (spec 5 / 16 G0): atom order (worker: ts.xyz elements and coordinates = D1_JOBS/<T>/ts.xyz <= 1e-6 A, (1) tags =
    input_meta A_idx, fragment sizes = n_f1 / n_f2, role1 / role2 = dipole / dipolarophile, rel1 / rel2 = D1_JOBS
    ref_dipole.xyz / ref_dipolarophile.xyz (order, <= 1e-6 A), rel1 + rel2 element multisets = the TS (1) + (2) atoms,
    charges = label; g0_report: xtb_slice.ts_fragments(label, "g1", meta) on the folder returns a namespace) failing for
    > 0.5 % of the targets = STOP; Freq failures (ORCA not terminated, .hess missing / unreadable / wrong size / $atoms
    symbols or geometry != ts.xyz, unexpected errors) > 1 % = STOP; n_imag 0 or >= 2 is recorded, not a failure; B6 not
    computable (ext_features.block_b6 itself, run on every G0 folder in g0_report) > 1 % of the .done folders = STOP
    (spec 5 last sentence; F0 confirms it from status_B6).
F0  the D1-G2 Phase F path (phase_fh.py F part = the T-1 / T-2 root paths) with root = $LAD/dft_final:
      f0_slice   d1_features.py runroot $LAD/dft_final $LAD/F0/out $LAD/F0/ids_all.csv i n --labels --meta --prof
      f0_merge   d1_features.py mergeroot $LAD/F0/out $LAD/X182_d1_dft.parquet $LAD/F0/mergeroot_report.json
                 (column order = X182_d1; F-1 = mergeroot stop: columns != X182_d1 or a non-finite value)
      f0_atoms / f0_gate / f0_tokens   fu2/atoms.py root run / gate / tokens (Q-a <= 1e-6, Q-b, Q-c, T-a unchanged) with
                 D1PAIR_PHASE=FROOT, G1_ROOT=$LAD/dft_final, FU2_SCR=$LAD/fu2, FU2_RES=analysis/d1_ladder/results/fu2,
                 FU2_X182=$LAD/X182_d1_dft.parquet, FU2_ATOM_VERSION=fu2-atoms-3-dft, FU2_TOKEN_VERSION=fu2-tokens-3-dft,
                 ESPLEY_OUT=$LAD/F0/fu2_xtb, R5_SCRATCH=$LAD/F0/fu2_r5 (set, not read by atoms.py)
      f0_report  results/F0_report.json + results/F0_tokens_manifest.csv (byte copy of results/fu2/T_manifest.csv)
    Inputs of both: the verbatim G2 inputs $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv, $G2/prof (read only).

CLI (exit 0 = done, 2 = not ready / incomplete (resubmit the missing part), 3 = STOP or hard error; every command writes
results/G0_steps/<step>.json or results/F0_steps/<step>.json; a rerun overwrites its own step record)
  G0 worker: jobs/lad.slurm, LADENV=d1 (the D1 run env: ORCA 6.1.1, D1 code copy), submit from analysis/d1_ladder:
    g0 <i|A> <n> [retry]  slice i of n (A = $SLURM_ARRAY_TASK_ID) of the targets in job_id order; one serial ORCA per
                          reaction, $SLURM_CPUS_PER_TASK reactions at a time (threads), node-local work dirs. A reaction
                          with .done / .fail_* is skipped; 'retry' first moves this slice's .fail_freq_* / .fail_error_*
                          folders to $LAD/dft_final_prev/<rxn_id>.<time> (explicit only; never automatic).
  everything else: analysis/d1_pair jobs/g.slurm with --export=ALL,PYENV=venv5,D1PAIR_PHASE=F (test_t.env_f is checked),
  submit from analysis/d1_pair, script ../d1_ladder/g0f0.py:
    g0_report             live scan of the 9,266 folders + _g0check child (ts_fragments + ext_features.block_b6 on every
                          .done folder, root env = $LAD/dft_final + G2 inputs) -> results/G0_status.csv, G0_report.json
    f0_prep               (after a passing G0 report) $LAD/F0/ids_all.csv (all target rxn_ids) + results/F0_estimate.json
                          (core-h expected for F0 from the D1PAIR G1 F / Q logs)
    f0_estimate           rewrite results/F0_estimate.json only
    f0_slice <i|@I|A> <n> runroot slice i of n (g.slurm replaces @I by $SLURM_ARRAY_TASK_ID); skipped when its 4 slice
                          files exist; refused if slices of another n exist
    f0_merge              mergeroot (skipped when the report matches X182_d1_dft.parquet); n_done must = the G0 .done count
    f0_atoms <i> <n>      fu2/atoms.py root run i n (per-reaction cache in atoms.py)
    f0_gate               one 'root run 0 1' retry pass if <rid>.err.json remain (until one exits 0), then root gate
    f0_tokens             root tokens (refused after a Q STOP)
    f0_report             results/F0_report.json, results/F0_tokens_manifest.csv
    f0_rest               f0_merge, f0_atoms 0 1, f0_gate, f0_tokens (stops at the first non-zero), then always f0_report
  internal: _g0check <ids.csv> <out.json>  (child of g0_report)

sbatch chain (6 tasks; at most 2 running at once, so that V (2) + G0 / F0 (2) stay within the LADDER cap of 4, spec 15).
Right before submitting (login node, one-shot commands): N = `squeue -u $USER -h -r | wc -l` must leave room: N + 6 <= 20
(else submit the G0 part (3 tasks) now and the F0 part (3 tasks, without the JG0R dependency) once G0R has passed);
`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`. D1-G2 jobs keep priority. (Fixer 2026-10-22: 2 slices %2 instead of 4 %4.)
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder; L=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs
  JG0=$(sbatch --parsable -p cpu1,cpu2 -J lad_g0 --time=48:00:00 --cpus-per-task=16 --mem=32G --array=0-1%2 \
       --output=$L/%x_%A_%a.out --export=ALL,LADENV=d1 jobs/lad.slurm g0f0.py g0 A 2); echo "JG0=$JG0"
  JG0R=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J lad_g0r --time=48:00:00 --cpus-per-task=16 --mem=32G \
       --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterany:${JG0:?} \
       jobs/g.slurm ../d1_ladder/g0f0.py g0_report ::: ../d1_ladder/g0f0.py f0_prep); echo "JG0R=$JG0R"
  JFS=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J lad_f0s --time=48:00:00 --cpus-per-task=32 --mem=64G \
       --array=0-1%2 --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F \
       --dependency=afterok:${JG0R:?} --kill-on-invalid-dep=yes jobs/g.slurm ../d1_ladder/g0f0.py f0_slice @I 2); echo "JFS=$JFS"
  JFR=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J lad_f0r --time=48:00:00 --cpus-per-task=48 --mem=96G \
       --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JFS:?} \
       --kill-on-invalid-dep=yes jobs/g.slurm ../d1_ladder/g0f0.py f0_rest); echo "JFR=$JFR"
  (--output must be given for g.slurm: its default log folder is the read-only d1_pair scratch.)
Resubmission (all steps idempotent, writes atomic): a G0 task that died or exited 2 -> the same JG0 line with
--array=<k> (finished reactions are skipped), then JG0R..JFR again. JG0R exit 2 = G0 incomplete (read G0_report.json
missing / not_started); exit 3 = G0 STOP: report, nothing else. A dead f0_slice -> resubmit that slice (--array=<k>), then
JFR; JFR died -> the same JFR line without --dependency. JFR exit 3 = F0 STOP (results/F0_report.json stop_reasons).

Decisions (user 2026-10-22: every branch = the recommended option; recorded in G0_report / F0_report "decisions"):
  G0-1 targets = lad_common.targets() (9,266) and they must be exactly the hf_status.csv geom=dft ok job_ids (else STOP).
  G0-2 coordinates come from the D1-HF dft inputs (spec 5); they are cross-checked against the label files in D1_JOBS
       (order + 1e-6 A) rather than taken from them, so both sources must agree.
  G0-3 ORCA with %pal nprocs 1 (serial, no MPI: wall = core time; reactions run in parallel instead), %maxcore =
       config_g2.yaml maxcore_mb (as g2_autode.finalize). The written ts.xyz is read back and its coordinates go into the
       ORCA input and into forming_share, so result.json equals what ext_features.block_b6 recomputes from the folder.
  G0-4 n_imag 0 / >= 2: .done (spec 5). The share is taken from the most negative mode, as block_b6 does.
  G0-5 rates: order and Freq over the 9,266 targets; B6 over the .done folders; unexpected errors count as Freq
       failures (conservative). The B6 > 1 % STOP is applied already at G0 with the F0 code itself (block_b6), so F0 is
       not run when it would STOP; F0 re-checks it from status_B6 and STOPs on a G0 / F0 B6 disagreement.
  G0-6 operational guards (no method change): up to two reactions of a task run alone first (environment canary); the
       task stops (exit 3) only when both fail Freq / error, or on a write error; a single failure stays that reaction's
       .fail_* (a rerun without 'retry' skips it). A task stops scheduling when >= 20 and > 25 % of its finished
       reactions failed Freq / error (exit 3), when 44 h have passed (exit 2, resubmit), or when the slice's bytes
       (including what earlier runs of the slice wrote) exceed D1LAD_G0_BUDGET_GB / n (default 7 GB in total; expected
       ~5 GB: mean ts.hess ~0.43 MB + ts.out.gz ~0.07 MB per reaction) -> exit 3 'budget' (a resubmission cannot get
       around it; raise the variable explicitly). (Fixer 2026-10-22: one canary, a per-run budget and exit 2 before.)
  F0-1 inputs = the verbatim G2 inputs (sha256 = results/G_inputs_report.json of D1-G2; label rows compared with
       labels_d1.json in g0_report); the D1PAIR ones cover only 8,659 of the 9,266 rows; T-1 / T-2 passed with them.
  F0-2 D1G2_SCRATCH=$LAD only in the environment of the d1_features.py children (runroot -> _rootrun, mergeroot): its only
       use is the ROOT_WRITE_OK guard (d1_features.py G2_SCRATCH / assert_root_write_ok; the other guard is
       d1pair_common.assert_write_ok, which does not cover /gpfs/tmp_cpu2/yeseo1ee/d1_ladder). The job env and this
       process keep the default so test_t / g2_common keep their own paths; nothing is written under $G2.
  F0-3 STOP (spec 16 F0), every rate over n_done = the G0 .done folders (= the reactions F0 processes): F-1 = every
       mergeroot not-ok row = every .done reaction missing from X182_d1_dft (xtb_status, geom, B1..B6 incl. the B1
       engine gate and B6-only failures, non-finite), as the D1PAIR F-1 ('all 182 columns finite'; fixer 2026-10-22:
       B6-only rows were left out before); B6 = status_B6 not ok / skipped (its own 1 % gate on top);
       Q-a = Q_manifest qa_ok false (atoms.py definition, incl. compute failures); T-a = T_report bad rows.
       Any > 1 % = STOP. Spec conflict, reading recorded in F0_report gate_rates.effective_limits: spec 6 applies the FU2
       Q-a / T-a gates 'as is', so their own verdicts also stop (Q_report stop = Q-a > QA_MAX_FAIL = 42 reactions
       absolute ~ 0.45 % of 9,266 / any Q-b / Q-c; T_report stop = any T-a row): the effective Q-a / T-a limits are
       stricter than 1 %. Also STOP: mergeroot stop, a step exit 3, inconsistent counts, and a token layout (num_cols,
       bin_cols, glob_cols, ext_used, bond_cap) different from the D1PAIR g1 rung's T_report (spec 4-2 / 4-3: same model
       inputs; e.g. FU2 Q-1 dropping mu_abs / f_plus / f_minus). The D1-G2 combined rate (targets without a token) is
       reported for information.
  F0-4 layout: f0_slice as a 2-task array %2 (reaction-parallel xtb, XTB_THREADS=1 from g_env.sh), then one 48-core job
       for merge / atoms (one slice, 48 workers) / gate / tokens / report (MaxSubmit: the whole chain is 6 tasks).
  F0-6 step(): a SystemExit inside a step body (lad_common.stop, load_d1_labels) is recorded in the step json with its
       code, so f0_rest still reaches f0_report (fixer 2026-10-22).
  F0-5 estimate from the D1PAIR logs (results/F0_estimate.json): Phase F 1024506 (8 x 32 cores) + merge 1024507,
       FU2 Q 1029859 (2 x 48 cores) + gate / tokens 1030201 (8 cores), scaled per reaction to the G0 .done count.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                           # no __pycache__ in analysis/d1_*, the D1 code copy

import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402

DP = L.DP
T0 = float(os.environ.get("D1LADDER_JOB_T0") or time.time())
STOP_AFTER_H = 44.0

# ---------------------------------------------------------------- inputs (read only)
HF_INPUTS = L.HF_SCR / "inputs"                          # <T>/hf/{eda,frag1_rel,frag2_rel}.inp (D1-HF geom=dft)
HF_STATUS = L.HF_RES / "hf_status.csv"                   # job_id, rxn_id, geom, status, reason
G2_INPUTS = dict(labels=L.G2 / "G" / "labels_d1_g2.json", meta=L.G2 / "G" / "input_meta_d1_g2.csv", prof=L.G2 / "prof")
G2_INPUTS_REPORT = L.G2_DIR / "results" / "G_inputs_report.json"
G2_T_REPORT = L.G2_DIR / "results" / "T_report.json"
START_REPORT = L.RES / "start_report.json"
# ---------------------------------------------------------------- G0
DFT_FINAL = L.LAD / "dft_final"
DFT_PREV = L.LAD / "dft_final_prev"
G0_DIR = L.LAD / "G0"
G0_STEPS = L.RES / "G0_steps"
G0_STATUS = L.RES / "G0_status.csv"
G0_REPORT = L.RES / "G0_report.json"
LEVEL = ("dft rung: label TS (D1-HF geom=dft eda.inp coordinates, fixed) + per-reaction label references (frag*_rel.inp);"
         " ts.hess = ORCA 6.1.1 XTB2 ALPB(water) Freq")
ORDER_MAX, FREQ_MAX, B6_MAX = 0.005, 0.01, 0.01          # spec 16 G0 (+ spec 5 last sentence)
TOL_TS_A, TOL_REL_A, TOL_RT_A, TOL_HESS_GEOM_A = 1e-6, 1e-6, 5e-9, 1e-4   # TOL_HESS_GEOM_A = ext_features.TOL_HESS_GEOM_A
ABORT_MIN, ABORT_FRAC = 20, 0.25
BUDGET_GB = float(os.environ.get("D1LAD_G0_BUDGET_GB", "7"))
# ---------------------------------------------------------------- F0
F0_DIR = L.LAD / "F0"
F0_OUT = F0_DIR / "out"
F0_SLICES = F0_OUT / "xtb" / "f_slices"                 # d1_features runroot: <out>/xtb/f_slices/slice<i>of<n>_*
F0_IDS = F0_DIR / "ids_all.csv"
F0_MREP = F0_DIR / "mergeroot_report.json"
F0_G1P = F0_OUT / "xtb" / "xtb_features_g1_d1.parquet"   # written by mergeroot
F0_EXTP = F0_OUT / "xtb" / "xtb_features_ext_g1_d1.parquet"
X182_DFT = L.LAD / "X182_d1_dft.parquet"
FU2_SCR, FU2_RES = L.LAD / "fu2", L.RES / "fu2"
VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-dft", FU2_TOKEN_VERSION="fu2-tokens-3-dft")   # spec 6
SLICE_FILES = ("g1.parquet", "ext.parquet", "root.json", "versions.json")
F0_STEPS = L.RES / "F0_steps"
RETRY_REC = F0_STEPS / "f0_atoms_retry.json"
F0_REPORT = L.RES / "F0_report.json"
F0_TOK_MAN = L.RES / "F0_tokens_manifest.csv"
F0_EST = L.RES / "F0_estimate.json"
F0_RATE_MAX = 0.01                                       # spec 16 F0
ATOMS_ENV_KEYS = ("D1PAIR_PHASE", "G1_ROOT", "ESPLEY_LABELS", "ESPLEY_META", "ESPLEY_PROF", "ESPLEY_OUT", "R5_SCRATCH",
                  "FU2_SCR", "FU2_RES", "FU2_X182", "FU2_ATOM_VERSION", "FU2_TOKEN_VERSION")
Q_KEYS = ("n_target", "n_ok", "n_compute_fail", "n_qa_not_reproduced", "n_qa_fail", "qa_max_fail", "n_qb_fail",
          "n_excluded_ext_fail", "ext_pass_rate", "ext_used", "n_use", "fail_reasons", "qa_maxdiff_max",
          "qb_maxdiff_max", "atom_version", "stop", "stop_reason", "manifest_sha256")
T_KEYS = ("token_version", "n_reactions", "n_excluded_from_Q", "num_cols", "ext_used", "gate_Ta", "stop",
          "manifest_sha256", "q_report_sha256", "x182_sha256")
T_LAYOUT_KEYS = ("num_cols", "bin_cols", "glob_cols", "ext_used", "bond_cap")   # = D1PAIR fu2-tokens-3 T_report
# D1PAIR G1 logs for the F0 estimate (spec 6): the runs that made X182_d1 and the fu2-atoms-3 / tokens
EST_LOGS = {"F_run": (DP.LOGS, [f"d1f_run_1024506_{k}.out" for k in range(8)], ("d1_features.py run ",)),
            "F_merge": (DP.LOGS, ["d1f_merge_1024507.out"], ("d1_features.py merge",)),
            "Q_run": (DP.SCRATCH / "fu2" / "logs", [f"d1fu2_Q_1029859_{k}.out" for k in range(2)], ("fu2/atoms.py run ",)),
            "QT": (DP.SCRATCH / "fu2" / "logs", ["d1fu2_QT_1030201.out"], ("fu2/atoms.py gate", "fu2/tokens.py build"))}
RE_CPUS = re.compile(r"\bcpus=(\d+)")
RE_EXIT = re.compile(r"^=== exit (-?\d+) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*$")
RE_CMD = re.compile(r"^=== (.+?)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s*$")
RE_NRX = re.compile(r"^\[(?:F slice|Q) \d+/\d+\] (\d+) rxns")

G0_DECISIONS = [
    "G0-1 targets = lad_common.targets() (9,266) = exactly the hf_status.csv geom=dft ok job_ids, else STOP",
    "G0-2 ts / rel coordinates from the D1-HF dft inputs (spec 5), cross-checked against D1_JOBS ts.xyz / ref_dipole.xyz "
    "/ ref_dipolarophile.xyz (order + 1e-6 A), tags vs input_meta A_idx, roles, element multisets, charges",
    "G0-3 ORCA %pal nprocs 1 (serial, wall = core time), %maxcore = config_g2.yaml maxcore_mb; ts.xyz read back and used "
    "for the ORCA input and forming_share (result.json = what ext_features.block_b6 recomputes)",
    "G0-4 n_imag 0 or >= 2 -> .done and recorded; share from the most negative mode (block_b6 rule)",
    "G0-5 order / Freq rates over the 9,266 targets, B6 over the .done folders (block_b6 on every folder at G0); "
    "unexpected errors count as Freq failures",
    "G0-6 operational guards only: two canary reactions per task (stop only if both fail Freq/error, or a write "
    "error), abort at >= 20 and > 25 % Freq/error failures, 44 h, slice byte budget incl. earlier runs "
    "D1LAD_G0_BUDGET_GB (default 7 GB total) -> exit 3",
    "folder = g1_final format (.done, ts.xyz, rel1.xyz, rel2.xyz, result.json, work/ts.hess) + work/ts.inp, "
    "work/ts.out.gz; no top-level ts.hess, no other ORCA by-products (quota); ts.out read by no F0 code"]
F0_DECISIONS = [
    "F0-1 inputs = verbatim G2 inputs ($G2/G/labels_d1_g2.json, input_meta_d1_g2.csv, $G2/prof; sha256 = D1-G2 "
    "G_inputs_report.json), read only",
    "F0-2 D1G2_SCRATCH=$LAD only for the d1_features.py children (the ROOT_WRITE_OK guard); job env unchanged",
    "F0-3 rates over n_done (G0 .done folders): F-1 (every reaction missing from X182_d1_dft, B6-only included, as "
    "D1PAIR F-1), B6 (status_B6 not ok/skipped), Q-a (Q_manifest qa_ok false), T-a (T_report bad); any > 1 % = STOP; "
    "plus the unchanged FU2 verdicts (Q_report stop = Q-a > 42 absolute / Q-b / Q-c, T_report stop = any T-a: spec 6 "
    "'as is', so the effective Q-a / T-a limits are stricter than 1 %), mergeroot stop, step exit 3, inconsistent "
    "counts, G0/F0 B6 disagreement, token layout != D1PAIR g1 T_report",
    "F0-4 f0_slice = 2-task array %2; merge / atoms (0 of 1, 48 workers) / gate / tokens / report in one 48-core job",
    "F0-6 step() records a SystemExit inside a step (f0_rest always reaches f0_report)",
    "F0-5 core-h estimate from D1PAIR logs (F 1024506 + merge 1024507, Q 1029859 + gate/tokens 1030201), per reaction "
    "x the G0 .done count"]


# ================================================================ helpers
class Stop(Exception):
    """(stage, message, exit code): 2 = not ready / incomplete, 3 = STOP (spec 16) or a hard error."""

    def __init__(self, stage, msg, rc=3):
        super().__init__(stage, msg)
        self.stage, self.msg, self.rc = stage, msg, rc


class Fail(Exception):
    """a per-reaction G0 failure; the message is the reason (order_* | freq_*)."""


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clean(o):
    """JSON without NaN / Infinity (readable with jq); numpy scalars / arrays, tuples, sets and paths -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, Path):
        return str(o)
    return o


def git_head():
    try:
        return L.git("rev-parse", "--short", "HEAD").strip() or None
    except Exception:                                       # noqa: BLE001 — informational
        return None


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def step(group, name, body):
    """body(rec) -> exit code (0 / 2 / 3); results/<group>_steps/<name>.json (with core_h_alloc = wall x cpus)."""
    out = L.RES / f"{group}_steps" / f"{name}.json"
    rec = dict(step=name, group=group, argv=sys.argv[1:], started=now(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_job=os.environ.get("SLURM_ARRAY_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), cpus=cpus(), git=git_head(),
               python=sys.executable)
    t0 = time.time()
    try:
        rc = int(body(rec) or 0)
    except Stop as e:
        rec.update(fail_stage=e.stage, fail=str(e.msg)[:4000])
        rc = e.rc
    except SystemExit as e:                                # L.stop / load_d1_labels / L.targets exit inside a body
        code = e.code
        rc = 0 if code in (None, 0) else (code if isinstance(code, int) else 3)
        rec.update(fail_stage="exit", fail=f"SystemExit({code!r}) inside the step (message on stderr)"[:4000])
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed step, recorded
        rec.update(fail_stage="error", fail=f"{type(e).__name__}: {e}"[:4000], traceback=traceback.format_exc()[-4000:])
        rc = 3
    rec.update(rc=rc, finished=now(), wall_s=round(time.time() - t0, 1))
    rec["core_h_alloc"] = rec["wall_s"] * cpus() / 3600.0
    L.write_json(out, clean(rec))
    short = {k: v for k, v in rec.items() if k not in ("traceback",)}
    print(json.dumps(clean(short), default=str)[:8000], flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[{name}] exit {rc} -> {out}", flush=True)
    return rc


def array_k(s):
    if str(s).upper() in ("A", "@I"):
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            raise SystemExit(f"{s}: no SLURM_ARRAY_TASK_ID (submit with --array or give the slice number)")
        return int(v)
    return int(s)


def marker(d):
    """'.done', the first sorted '.fail_*' or None for a folder (one small folder)."""
    d = Path(d)
    if (d / ".done").is_file():
        return ".done"
    if not d.is_dir():
        return None
    f = sorted(p.name for p in d.glob(".fail_*"))
    return f[0] if f else None


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def qstats(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(float(x))], dtype=float)
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), min=float(a.min()), p05=float(np.quantile(a, 0.05)), median=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)), max=float(a.max()))


def folder_bytes(d):
    return int(sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file()))


# ================================================================ shared checks
def start_ok(rec):
    s = read_json(START_REPORT)
    rec["start_report"] = dict(stop=s.get("stop"), git_head=s.get("git_head"))
    if s.get("stop") is not False:
        raise Stop("start", f"{START_REPORT}: stop = {s.get('stop')!r} (spec 1: lad_common.py start must pass)", 3)


def hf_dft_ok():
    with open(HF_STATUS, newline="") as f:
        return {r["job_id"] for r in csv.DictReader(f) if r["geom"] == "dft" and r["status"] == "ok"}


def target_rows(rec):
    """lad_common.targets() (labels_d1.json checks inside), which must equal the D1-HF geom=dft ok job_ids."""
    tg = L.targets()
    ok = hf_dft_ok()
    ids = {d["job_id"] for d in tg}
    rec["targets"] = dict(n=len(tg), hf_dft_ok=len(ok), hf_status=str(HF_STATUS), hf_status_sha256=L.sha256(HF_STATUS))
    if ids != ok:
        raise Stop("targets", f"targets ({len(ids)}) != hf_status geom=dft ok ({len(ok)}): only targets "
                              f"{sorted(ids - ok)[:10]}, only D1-HF {sorted(ok - ids)[:10]}", 3)
    return tg


def g2_inputs_check(rec):
    """the verbatim G2 inputs (read only): sha256 = the D1-G2 prep_g report, prof folder present."""
    rep = read_json(G2_INPUTS_REPORT)
    want = rep.get("outputs") or {}
    got = {k: L.sha256(G2_INPUTS[k]) for k in ("labels", "meta")}
    rec["g2_inputs"] = dict({k: dict(path=str(G2_INPUTS[k]), sha256=got[k]) for k in got},
                            prof=str(G2_INPUTS["prof"]), report=str(G2_INPUTS_REPORT), report_stop=rep.get("stop"))
    bad = [k for k in got if want.get(str(G2_INPUTS[k])) != got[k]]
    if bad or rep.get("stop") is not False or not G2_INPUTS["prof"].is_dir():
        raise Stop("inputs", f"G2 inputs {bad} differ from {G2_INPUTS_REPORT} (or report stop / prof missing)", 3)
    return got


def read_meta():
    with open(G2_INPUTS["meta"], newline="") as f:
        return {int(r["rxn_id"]): sorted(int(x) for x in str(r["A_idx"]).split()) for r in csv.DictReader(f)}


# ================================================================ G0 worker (LADENV=d1)
def env_g0(rec):
    bad = []
    d1py = os.environ.get("D1_PY")
    if not d1py or Path(sys.executable).resolve() != Path(d1py).resolve():
        bad.append(f"python {sys.executable} is not $D1_PY {d1py!r} (jobs/lad.slurm with LADENV=d1)")
    orca = os.environ.get("ORCA_BIN", "")
    want_orca = Path("/home1/yeseo1ee/orca_6_1_1_avx2/orca")
    if not (orca and os.access(orca, os.X_OK)) or Path(orca).resolve() != want_orca.resolve():
        bad.append(f"ORCA_BIN={orca!r} (need executable {want_orca})")
    if os.environ.get("D1LADDER_SCRATCH") != str(L.LAD):
        bad.append(f"D1LADDER_SCRATCH={os.environ.get('D1LADDER_SCRATCH')!r} (jobs/lad.slurm sets {L.LAD})")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        bad.append(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} (d1 env.sh sets 1)")
    rec["env"] = {k: os.environ.get(k) for k in ("D1_PY", "ORCA_BIN", "D1LADDER_SCRATCH", "OMP_NUM_THREADS", "TMPDIR",
                                                 "SLURM_CPUS_PER_TASK", "LADENV")}
    if bad:
        raise Stop("environment", "; ".join(bad), 3)
    start_ok(rec)


def g0_modules(rec):
    """hf_common (parse_inp) and the D1-G2 finalize code path: g2_autode.FREQ_HEADER / CONFIG / label_pairs and the D1
    code copy modules it imports (orca_direct, run_ts, pilot_common). Importing g2_autode only rebinds autodE helpers in
    this process (no file is written)."""
    for p in (L.HF_DIR, L.G2_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import hf_common as H
    import g2_autode as GA
    import yaml
    cfg = yaml.safe_load(GA.CONFIG.read_text())
    OD, RT, pc = GA.OD, GA.RT, GA.pc
    pilot = Path(GA.G.PILOT).resolve()
    files = {m.__name__: str(Path(m.__file__).resolve()) for m in (OD, RT, pc, GA, H)}
    rec["code"] = dict(files, freq_header=GA.FREQ_HEADER, maxcore_mb=cfg["maxcore_mb"], pilot=str(pilot))
    if any(Path(files[m.__name__]).parent != pilot for m in (OD, RT, pc)):
        raise Stop("code", f"orca_direct / run_ts / pilot_common not from the D1 code copy {pilot}: {files}", 3)
    return SimpleNamespace(H=H, GA=GA, OD=OD, RT=RT, pc=pc, cfg=cfg)


def atoms_of(p):
    """(symbols, (n,3) coordinates, tags) of a parse_inp result."""
    a = p["atoms"]
    return [x[0] for x in a], np.array([[x[2], x[3], x[4]] for x in a], dtype=float).reshape(-1, 3), [x[1] for x in a]


def order_checks(lab, T, rid, eda, r1, r2, meta, M):
    """the worker part of the atom-order gate (module docstring); raises Fail('order_*'); returns the deviations."""
    n, q1, q2 = int(lab["n_atoms"]), int(lab["charge1"]), int(lab["charge2"])
    syms, xyz, tags = atoms_of(eda)
    lts = M.pc.read_xyz(DP.D1_JOBS / T / "ts.xyz")
    if list(lts[0]) != syms or len(syms) != n:
        raise Fail("order_ts_elements_vs_label")
    dts = float(np.abs(np.asarray(lts[1], dtype=float) - xyz).max())
    if not dts <= TOL_TS_A:
        raise Fail("order_ts_coordinates_vs_label")
    if set(tags) != {1, 2}:
        raise Fail("order_ts_tags")
    A = [k for k, t in enumerate(tags) if t == 1]
    if A != meta.get(rid):
        raise Fail("order_ts_tags_vs_meta")
    if len(A) != int(lab["n_f1"]) or n - len(A) != int(lab["n_f2"]):
        raise Fail("order_fragment_sizes")
    if (lab.get("role1"), lab.get("role2")) != ("dipole", "dipolarophile") or \
            (lab.get("rel1_file"), lab.get("rel2_file")) != ("ref_dipole.xyz", "ref_dipolarophile.xyz"):
        raise Fail("order_roles")
    if eda["charge"] != q1 + q2 or eda["mult"] != 1:
        raise Fail("order_charge_ts")
    fr = eda.get("frag") or {}
    for k, q in ((1, q1), (2, q2)):
        if f"FRAG{k}_C" in fr and int(fr[f"FRAG{k}_C"]) != q:
            raise Fail(f"order_charge_frag{k}")
    devs = dict(ts_vs_label_max_A=dts)
    for k, (r, ref, q, tv) in enumerate(((r1, "ref_dipole.xyz", q1, 1), (r2, "ref_dipolarophile.xyz", q2, 2)), 1):
        s, x, _ = atoms_of(r)
        if r["charge"] != q or r["mult"] != int(fr.get(f"FRAG{k}_M", 1)):
            raise Fail(f"order_charge_rel{k}")
        lr = M.pc.read_xyz(DP.D1_JOBS / T / ref)
        if list(lr[0]) != s:
            raise Fail(f"order_rel{k}_elements_vs_label")
        d = float(np.abs(np.asarray(lr[1], dtype=float) - x).max())
        if not d <= TOL_REL_A:
            raise Fail(f"order_rel{k}_coordinates_vs_label")
        if Counter(s) != Counter(syms[j] for j in range(n) if tags[j] == tv):
            raise Fail(f"order_rel{k}_elements_vs_ts")
        devs[f"rel{k}_vs_label_max_A"] = d
    devs.update(n_f1=len(A), n_f2=n - len(A), A_idx_eq_meta=True, rel_elements_eq_ts=True)
    return syms, xyz, devs


def hess_atoms(path):
    """$atoms symbols and coordinates [A] of an ORCA .hess file (the ext_features.parse_hess $atoms parse)."""
    lines = Path(path).read_text().splitlines()
    sec = {ln.strip(): i for i, ln in enumerate(lines) if ln.startswith("$")}
    a = sec["$atoms"]
    na = int(lines[a + 1].split()[0])
    rows = [lines[a + 2 + m].split() for m in range(na)]
    xyz = np.array([[float(x.replace("D", "E").replace("d", "e")) for x in r[2:5]] for r in rows]) / DP.R5.BOHR
    return [r[0] for r in rows], xyz


def g0_one(lab, meta, ctx):
    """one reaction -> (rxn_id, status, category); never raises. Category: skip | ok | order | freq | error |
    not_started | write_error."""
    rid, T = int(lab["rxn_id"]), lab["job_id"]
    out = DFT_FINAL / str(rid)
    if marker(out) is not None:
        return rid, "skip", "skip"
    if ctx.abort.is_set():
        return rid, "not_started_abort", "not_started"
    if time.time() - T0 > STOP_AFTER_H * 3600:
        return rid, "not_started_time", "not_started"
    with ctx.lock:
        if ctx.bytes > ctx.budget:
            return rid, "not_started_budget", "not_started"
    M = ctx.mods
    t0 = time.time()
    q1, q2 = int(lab["charge1"]), int(lab["charge2"])
    res = dict(rxn_id=rid, job_id=T, geom="dft", level=LEVEL, charge1=q1, charge2=q2,
               label_files={k: lab.get(k) for k in ("ts_file", "rel1_file", "rel2_file")},
               roles=[lab.get("role1"), lab.get("role2")], host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    reasons = []
    tmp, lw = DFT_FINAL / f"{rid}.tmp", ctx.local / str(rid)
    try:
        L.assert_path(tmp)
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "work").mkdir(parents=True)
        if lw.exists():
            shutil.rmtree(lw)
        lw.mkdir(parents=True)
        src = {k: HF_INPUTS / T / "hf" / f"{k}.inp" for k in ("eda", "frag1_rel", "frag2_rel")}
        res["sources"] = {k: dict(path=str(p), sha256=L.sha256(p)) for k, p in src.items()}
        eda, r1, r2 = (M.H.parse_inp(src[k].read_text()) for k in ("eda", "frag1_rel", "frag2_rel"))
        ts_syms, ts_xyz, res["checks"] = order_checks(lab, T, rid, eda, r1, r2, meta, M)
        # the files the feature code reads (label atom order)
        M.pc.write_xyz(tmp / "ts.xyz", ts_syms, ts_xyz, f"DFT label TS rxn {rid} ({T}; D1-HF geom=dft eda.inp)")
        for k, r in enumerate((r1, r2), 1):
            s, x, _ = atoms_of(r)
            M.pc.write_xyz(tmp / f"rel{k}.xyz", s, x, f"DFT label reference {k} rxn {rid} ({T}; D1-HF frag{k}_rel.inp)")
        syms, X = M.pc.read_xyz(tmp / "ts.xyz")                 # = what xtb_slice / ext_features read
        X = np.asarray(X, dtype=float)
        if list(syms) != ts_syms or not float(np.abs(X - ts_xyz).max()) <= TOL_RT_A:
            raise Fail("order_ts_xyz_readback")
        # ts.hess: XTB2 ALPB(water) Freq at the fixed label TS (g2_autode.finalize path, nprocs 1)
        header = M.GA.FREQ_HEADER.format(n=1, m=M.cfg["maxcore_mb"])
        inp = lw / "ts.inp"
        M.OD.write_inp(inp, header, syms, X, charge=q1 + q2, mult=1)
        t1 = time.time()
        outp = M.OD.run(M.cfg, inp)
        res["freq_wall_s"] = time.time() - t1
        txt = outp.read_text(errors="replace") if outp.is_file() else ""
        shutil.copyfile(inp, tmp / "work" / "ts.inp")
        if outp.is_file():
            with open(outp, "rb") as fi, gzip.open(tmp / "work" / "ts.out.gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
        err = lw / "ts.err"
        if err.is_file() and err.stat().st_size:
            res["orca_stderr_tail"] = err.read_text(errors="replace")[-2000:]
        ver = M.H.RE_VER.search(txt)
        res.update(ts_header=header, orca_bin=M.OD.orca_bin(M.cfg), orca_version=ver.group(1) if ver else None,
                   ts_terminated=M.OD.terminated(txt), e_ts_eh=M.OD.last_fspe(txt), ts_run_hours=M.OD.run_hours(txt))
        hess = lw / "ts.hess"
        if not (M.OD.terminated(txt) and hess.is_file()):
            raise Fail("freq_not_terminated")
        try:
            freqs, modes = M.OD.parse_hess(hess)
            imag, mode = M.OD.imag_and_mode(hess)
            hs, hx = hess_atoms(hess)
        except Exception as e:                             # noqa: BLE001
            res["hess_parse_error"] = f"{type(e).__name__}: {e}"[:300]
            raise Fail("freq_hess_unreadable")
        n = len(syms)
        if freqs.size != 3 * n or modes.shape != (3 * n, 3 * n):
            raise Fail("freq_hess_size")
        if list(hs) != list(syms):
            raise Fail("freq_hess_atoms")
        shift = hx - X
        dev = float(np.abs(shift - shift.mean(axis=0)).max())
        res.update(hess_geom_max_dev_A=dev, hess_shift_A=shift.mean(axis=0).tolist())
        if not dev <= TOL_HESS_GEOM_A:
            raise Fail("freq_hess_geometry")
        lp = M.GA.label_pairs(lab)
        heavy = [k for k, s in enumerate(syms) if s != "H"]
        share = None if mode is None else float(M.RT.forming_share(X, mode, lp, heavy))
        nz = freqs[freqs != 0.0]
        res.update(imag_freqs_cm=[float(v) for v in imag], n_imag=len(imag), imag_mode_forming_share=share,
                   lowest_freq_cm=float(nz.min()) if nz.size else None, n_zero_freq=int((freqs == 0.0).sum()),
                   n_freq=int(freqs.size), formed_pairs_label=[list(p) for p in lp],
                   d_form_dft=[float(np.linalg.norm(X[a] - X[b])) for a, b in lp],
                   imag_note=None if len(imag) == 1 else f"n_imag = {len(imag)} at the label TS (recorded, spec 5)")
        shutil.copyfile(hess, tmp / "work" / "ts.hess")
    except Fail as e:
        reasons.append(str(e))
    except Exception as e:                                 # noqa: BLE001 — recorded, the folder gets .fail_error_*
        res["traceback"] = traceback.format_exc()[-4000:]
        reasons.append(f"error_{type(e).__name__}")
    finally:
        shutil.rmtree(lw, ignore_errors=True)
    cat = "ok" if not reasons else reasons[0].split("_")[0]
    try:
        if reasons:                                         # a failed folder carries no structure files
            for f in ("ts.xyz", "rel1.xyz", "rel2.xyz", "work/ts.hess"):
                (tmp / f).unlink(missing_ok=True)
        res.update(status="ok" if not reasons else "fail:" + reasons[0], fail_reasons=reasons,
                   wall_s=time.time() - t0, core_h=(time.time() - t0) / 3600.0, written=now())
        (tmp / "result.json").write_text(json.dumps(clean(res), indent=1))
        mk = ".done" if not reasons else ".fail_" + re.sub(r"[^A-Za-z0-9_]+", "_", reasons[0])
        (tmp / mk).write_text(now() + "\n" + "\n".join(reasons) + "\n")
        os.replace(tmp, out)
        b = folder_bytes(out)
    except Exception as e:                                 # noqa: BLE001 — disk / quota: stop this task
        ctx.abort.set()
        print(f"  {rid}: WRITE ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return rid, f"write_error:{type(e).__name__}", "write_error"
    with ctx.lock:
        ctx.bytes += b
        ctx.n_fin += 1
        if cat in ("freq", "error"):
            ctx.n_bad += 1
            if ctx.n_bad >= ABORT_MIN and ctx.n_bad > ABORT_FRAC * ctx.n_fin:
                ctx.abort.set()
    return rid, res["status"], cat


def cmd_g0(i, n, retry=False):
    def body(rec):
        env_g0(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        tg = target_rows(rec)
        g2_inputs_check(rec)
        meta = read_meta()
        mods = g0_modules(rec)
        mine = tg[i::n]
        L.assert_path(DFT_FINAL).mkdir(parents=True, exist_ok=True)
        moved = []
        if retry:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for d in mine:
                f = DFT_FINAL / str(d["rxn_id"])
                mk = marker(f)
                if mk and mk.startswith((".fail_freq", ".fail_error")):
                    dst = L.assert_path(DFT_PREV / f"{d['rxn_id']}.{stamp}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(L.assert_path(f), dst)
                    moved.append(int(d["rxn_id"]))
        todo, have = [], []
        for d in mine:
            f = DFT_FINAL / str(d["rxn_id"])
            if marker(f) is None:
                todo.append(d)
            else:
                have.append(f)
        W = cpus()
        local = Path(os.environ.get("TMPDIR") or "/tmp") / f"d1lad_g0_{os.getpid()}"
        # G0-6 byte budget over the whole slice: what earlier runs of this slice already wrote counts (resubmission)
        b0 = sum(folder_bytes(f) for f in have)
        ctx = SimpleNamespace(lock=threading.Lock(), bytes=b0, budget=BUDGET_GB * 1e9 / n, abort=threading.Event(),
                              n_fin=0, n_bad=0, local=local, mods=mods)
        rec.update(slice=dict(i=i, n=n, n_slice=len(mine), n_todo=len(todo), workers=W, retry_moved=moved,
                              budget_bytes=ctx.budget, bytes_before=b0, local=str(local)))
        print(f"[g0 {i}/{n}] {len(mine)} rxns, {len(todo)} to do, {W} workers, budget {ctx.budget / 1e9:.2f} GB "
              f"({b0 / 1e9:.3f} GB already written)", flush=True)
        cnt, st_cnt = Counter(), Counter()
        local.mkdir(parents=True, exist_ok=True)
        try:
            if todo:
                # G0-6 canary: up to two reactions run alone; the task stops only when both fail Freq / error (an
                # environment problem) or on a write error; a single failure is recorded as that reaction's .fail_*
                canary = []
                for d in todo[:2]:
                    rid, st, cat = g0_one(d, meta, ctx)
                    cnt[cat] += 1
                    st_cnt[st if cat not in ("ok", "skip") else cat] += 1
                    canary.append(dict(rxn_id=rid, status=st, category=cat))
                    print(f"  canary {rid}: {st}", flush=True)
                    if cat not in ("freq", "error"):
                        break
                rec["canary"] = canary
                bad_c = [c for c in canary if c["category"] in ("freq", "error", "write_error")]
                if canary[-1]["category"] == "write_error" or (len(canary) == 2 and len(bad_c) == 2):
                    rec.update(counts=dict(cnt), statuses=dict(st_cnt))
                    raise Stop("canary", f"canary reactions of the slice failed: {canary} (environment, disk?); fix the "
                                         "cause, then rerun with 'retry' (moves this slice's .fail_freq / .fail_error "
                                         "folders). A failure tied to one reaction is kept as its .fail_* and skipped "
                                         "by a rerun without 'retry'", 3)
                with ThreadPoolExecutor(max_workers=W) as ex:
                    futs = [ex.submit(g0_one, d, meta, ctx) for d in todo[len(canary):]]
                    for k, fu in enumerate(as_completed(futs), 1):
                        try:
                            rid, st, cat = fu.result()
                        except Exception as e:                # noqa: BLE001 — a bug in g0_one: stop scheduling
                            ctx.abort.set()
                            rid, st, cat = None, f"crash:{type(e).__name__}:{e}", "error"
                        cnt[cat] += 1
                        st_cnt[st if cat not in ("ok", "skip") else cat] += 1
                        if cat not in ("ok", "skip", "not_started") or k % 200 == 0:
                            print(f"  {rid}: {st}  [{k}/{len(futs)}] {dict(cnt)}", flush=True)
        finally:
            shutil.rmtree(local, ignore_errors=True)
        remaining = [int(d["rxn_id"]) for d in mine if marker(DFT_FINAL / str(d["rxn_id"])) is None]
        rec.update(counts=dict(cnt), statuses=dict(st_cnt.most_common(40)), bytes_written=ctx.bytes - b0,
                   bytes_slice_total=ctx.bytes, n_remaining=len(remaining), remaining=remaining[:200],
                   aborted=ctx.abort.is_set())
        print(f"[g0 {i}/{n}] {dict(cnt)}; remaining {len(remaining)}; slice total {ctx.bytes / 1e9:.3f} GB", flush=True)
        if ctx.abort.is_set():
            raise Stop("abort", f"task stopped scheduling ({dict(cnt)}): Freq / error failures >= {ABORT_MIN} and "
                                f"> {ABORT_FRAC:.0%}, or a write error; read the .fail_* result.json files", 3)
        if st_cnt.get("not_started_budget"):
            raise Stop("budget", f"slice byte budget {ctx.budget / 1e9:.2f} GB reached ({ctx.bytes / 1e9:.3f} GB incl. "
                                 f"earlier runs; {st_cnt['not_started_budget']} reactions not started): check the folder "
                                 "sizes and the quota, then resubmit with a larger D1LAD_G0_BUDGET_GB (explicit)", 3)
        return 2 if remaining else 0
    return step("G0", f"g0_{i:02d}of{n:02d}", body)


# ================================================================ g0_report (venv5, d1_pair jobs/g.slurm F env)
_TT = {}


def tt():
    """analysis/d1_g2/test_t.py (run / now / git_head / clean / env_f / Fail / P_DIR), imported on first use."""
    if not _TT:
        if str(L.G2_DIR) not in sys.path:
            sys.path.insert(0, str(L.G2_DIR))
        import test_t
        _TT["m"] = test_t
    return _TT["m"]


def env_f(rec):
    TT = tt()
    try:
        TT.env_f(rec)                                      # d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F
    except TT.Fail as e:
        raise Stop("environment", str(e.args[1] if len(e.args) > 1 else e), 3)


def child(rec, name, argv, cwd, env=None, ok=(0,)):
    """one child interpreter (test_t.run); exit code recorded; an exit outside ok raises Stop (2 for 2, else 3)."""
    rc = tt().run(argv, cwd, env)
    rec.setdefault("exit", {})[name] = rc
    if rc not in ok:
        raise Stop(name, f"{name}: exit {rc}", 2 if rc == 2 else 3)
    return rc


def g0check_env():
    """environment of the _g0check child: the feature-code roots on $LAD/dft_final and the G2 inputs."""
    return dict(os.environ, G1_ROOT=str(DFT_FINAL), ESPLEY_LABELS=str(G2_INPUTS["labels"]),
                ESPLEY_META=str(G2_INPUTS["meta"]), ESPLEY_PROF=str(G2_INPUTS["prof"]),
                ESPLEY_OUT=str(G0_DIR / "gate_xtb"), R5_SCRATCH=str(G0_DIR / "gate_r5"), PYTHONDONTWRITEBYTECODE="1")


_G0C = {}


def _g0c_init():
    import xtb_slice as xs
    _G0C["lab"] = {int(d["rxn_id"]): d for d in json.loads(Path(os.environ["ESPLEY_LABELS"]).read_text())}
    _G0C["meta"] = xs.load_meta()


def _g0c_one(rid):
    """ts_fragments (the order gate) and ext_features.block_b6 (the F0 B6 code) on one G0 folder."""
    import ext_features as ef
    import xtb_slice as xs
    row = dict(rxn_id=int(rid), tsf=None, b6=None)
    try:
        g = xs.ts_fragments(_G0C["lab"][int(rid)], "g1", _G0C["meta"])
    except Exception as e:                                 # noqa: BLE001
        row["tsf"] = f"error:{type(e).__name__}:{e}"[:200]
        return row
    if isinstance(g, str):
        row["tsf"] = g
        return row
    row["tsf"] = "ok"
    try:
        v, qc = ef.block_b6(SimpleNamespace(g=g, rid=int(rid)))
        row.update(b6="ok", nu=float(v["nu_imag"]), share=float(v["mode_share"]), mode_async=float(v["mode_async"]),
                   qc={k: float(x) for k, x in qc.items()})
    except ef.BlockFail as e:
        row.update(b6=str(e.detail)[:200], qc={k: (float(x) if isinstance(x, (int, float)) else str(x))
                                              for k, x in (e.qc or {}).items()})
    except Exception as e:                                 # noqa: BLE001
        row["b6"] = f"error:{type(e).__name__}:{e}"[:200]
    return row


def _g0check(ids_csv, out_json):
    """child of g0_report (g0check_env): exit 3 unless the rev 4-6 modules picked up the root environment; then
    ts_fragments + block_b6 for the listed rxn_ids in a spawn pool -> <out_json>."""
    import multiprocessing as mp
    import ext_features as ef  # noqa: F401
    import fragmenter as fr
    import r6_partition as RP
    import rev5_common as R5
    import xtb_slice as xs
    want = dict(G1_ROOT=DFT_FINAL, ESPLEY_LABELS=G2_INPUTS["labels"], ESPLEY_META=G2_INPUTS["meta"],
                ESPLEY_PROF=G2_INPUTS["prof"])
    got = dict(xs_g1=(xs.GEOM_ROOT["g1"], "G1_ROOT"), R5_G1_ROOT=(R5.G1_ROOT, "G1_ROOT"),
               xs_LABELS=(xs.LABELS_PATH, "ESPLEY_LABELS"), R5_LABELS=(R5.LABELS, "ESPLEY_LABELS"),
               xs_META=(xs.META, "ESPLEY_META"), R5_META=(R5.META, "ESPLEY_META"), RP_PROF=(RP.PROF, "ESPLEY_PROF"),
               R5_PROF=(R5.PROF, "ESPLEY_PROF"))
    bad = {k: (str(p), str(want[v])) for k, (p, v) in got.items() if Path(p) != Path(want[v])}
    si = (os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), R5.BONDI.get("Si"), R5.GFN2_VALENCE.get("Si"))
    if bad or xs.PARTITION_MODE != "label" or si != ("1", 1.11, 14, 2.10, 4):
        print(f"STOP: _g0check modules not on the root environment: {bad}, partition {xs.PARTITION_MODE}, Si {si}",
              file=sys.stderr)
        return 3
    with open(ids_csv, newline="") as f:
        ids = [int(r["rxn_id"]) for r in csv.DictReader(f)]
    W = max(1, cpus())
    print(f"[_g0check] {len(ids)} folders, {W} workers, root {xs.GEOM_ROOT['g1']}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(W, initializer=_g0c_init) as pool:
        rows = pool.map(_g0c_one, ids, chunksize=16)
    L.write_json(out_json, clean(dict(ids_csv=str(ids_csv), ids_sha256=L.sha256(ids_csv), n=len(rows),
                                      env={k: os.environ.get(k) for k in want}, partition=xs.PARTITION_MODE,
                                      wall_s=time.time() - t0, rows=rows)))
    print(f"[_g0check] tsf {Counter(r['tsf'] for r in rows).most_common(8)} b6 "
          f"{Counter(r['b6'] for r in rows).most_common(8)} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def cmd_g0_report():
    def body(rec):
        import pandas as pd
        env_f(rec)
        start_ok(rec)
        tg = target_rows(rec)
        g2sha = g2_inputs_check(rec)
        # the G2 label rows the feature code reads = the label rows (verbatim)
        g2lab = {int(d["rxn_id"]): d for d in json.loads(G2_INPUTS["labels"].read_text())}
        canon = lambda d: json.dumps(d, sort_keys=True)          # noqa: E731 — NaN-safe row equality
        diff = [int(d["rxn_id"]) for d in tg if int(d["rxn_id"]) not in g2lab or canon(g2lab[int(d["rxn_id"])]) != canon(d)]
        if diff or len(g2lab) != len(tg):
            raise Stop("inputs", f"{G2_INPUTS['labels']}: {len(diff)} target rows differ from labels_d1.json "
                                 f"({diff[:10]}), {len(g2lab)} rows vs {len(tg)} targets", 3)
        rep = dict(spec="docs/specs/D1LADDER.md 5 (STOP: 16 G0)", written=now(), decisions=G0_DECISIONS,
                   thresholds=dict(order=ORDER_MAX, freq=FREQ_MAX, b6=B6_MAX), root=str(DFT_FINAL),
                   inputs=dict(hf_inputs=str(HF_INPUTS), g2=rec["g2_inputs"], labels_rows_equal_labels_d1=True))
        # live scan
        res, marks, missing = {}, {}, []
        for d in tg:
            r = int(d["rxn_id"])
            mk = marker(DFT_FINAL / str(r))
            if mk is None:
                missing.append(r)
                continue
            marks[r] = mk
            res[r] = read_json(DFT_FINAL / str(r) / "result.json")
        steps = {p.stem: read_json(p) for p in sorted(G0_STEPS.glob("g0_*of*.json"))}     # the g0 task records
        rep["steps"] = {k: dict(rc=v.get("rc"), finished=v.get("finished"), slurm_job=v.get("slurm_job"),
                                cpus=v.get("cpus"), wall_s=v.get("wall_s"), core_h_alloc=v.get("core_h_alloc"),
                                counts=v.get("counts"), fail=v.get("fail")) for k, v in steps.items()}
        if missing:
            rep.update(complete=False, n_missing=len(missing), missing=missing[:500], stop=False,
                       note="G0 incomplete: resubmit the g0 tasks whose slice has missing reactions, then g0_report")
            L.write_json(G0_REPORT, clean(rep))
            raise Stop("incomplete", f"{len(missing)} targets without .done / .fail_* under {DFT_FINAL}", 2)
        done = sorted(r for r, m in marks.items() if m == ".done")
        ids_csv = L.write_atomic(G0_DIR / "gate_ids.csv", "rxn_id\n" + "".join(f"{r}\n" for r in done))
        gate_json = G0_DIR / "gate.json"
        child(rec, "g0check", [THIS, "_g0check", ids_csv, gate_json], HERE, g0check_env())
        gate = json.loads(gate_json.read_text())
        if gate.get("ids_sha256") != L.sha256(ids_csv) or gate.get("n") != len(done):
            raise Stop("g0check", f"{gate_json} is not from this run", 3)
        gr = {int(x["rxn_id"]): x for x in gate["rows"]}
        rows = []
        for d in tg:
            r, mk = int(d["rxn_id"]), marks[int(d["rxn_id"])]
            x, g = res.get(r) or {}, gr.get(r) or {}
            if mk == ".done":
                tsf = g.get("tsf")
                cat, reason = ("ok", "") if tsf == "ok" else ("order", f"order_ts_fragments:{tsf}")
            else:
                reason = (x.get("fail_reasons") or [mk[len(".fail_"):]])[0]
                cat = reason.split("_")[0] if reason.split("_")[0] in ("order", "freq", "error") else "error"
            imag = x.get("imag_freqs_cm")
            rows.append(dict(rxn_id=r, status="ok" if cat == "ok" else "fail", reason=reason, n_imag=x.get("n_imag"),
                             lowest_freq_cm=x.get("lowest_freq_cm"), core_h=x.get("core_h"), job_id=d["job_id"],
                             category=cat, marker=mk,
                             imag_freqs_cm=";".join(f"{v:.4f}" for v in imag) if isinstance(imag, list) else None,
                             imag_mode_forming_share=x.get("imag_mode_forming_share"),
                             ts_fragments=g.get("tsf") if mk == ".done" else None,
                             b6_check=g.get("b6") if mk == ".done" else None, freq_wall_s=x.get("freq_wall_s"),
                             hess_geom_max_dev_A=x.get("hess_geom_max_dev_A")))
        S = pd.DataFrame(rows)
        L.write_atomic(G0_STATUS, lambda t: S.to_csv(t, index=False))
        n_t, n_done = len(S), len(done)
        okd = S[S.marker == ".done"]
        n_order = int((S.category == "order").sum())
        n_freq = int(S.category.isin(["freq", "error"]).sum())
        b6_bad = okd[(okd.ts_fragments == "ok") & (okd.b6_check != "ok")]
        rates = dict(order=n_order / n_t, freq=n_freq / n_t, b6=len(b6_bad) / n_done if n_done else float("nan"))
        stop = []
        if rates["order"] > ORDER_MAX:
            stop.append(f"atom-order gate failures {n_order}/{n_t} = {rates['order']:.4f} > {ORDER_MAX} (spec 16 G0)")
        if rates["freq"] > FREQ_MAX:
            stop.append(f"Freq failures {n_freq}/{n_t} = {rates['freq']:.4f} > {FREQ_MAX} (spec 16 G0)")
        if not n_done or rates["b6"] > B6_MAX:
            stop.append(f"B6 not computable {len(b6_bad)}/{n_done} = {rates['b6']:.4f} > {B6_MAX} (spec 5)")
        ni = Counter(int(v) for v in okd.n_imag.dropna())
        ex = next((res[r] for r in done if res.get(r)), {})
        rep.update(complete=True, counts=dict(targets=n_t, done=n_done, ok=int((S.status == "ok").sum()),
                                              fail_order=n_order, fail_freq=int((S.category == "freq").sum()),
                                              fail_error=int((S.category == "error").sum()),
                                              ts_fragments_fail=int((okd.ts_fragments != "ok").sum()),
                                              b6_not_computable=len(b6_bad)),
                   rates=rates, fail_reasons=dict(Counter(S.loc[S.status != "ok", "reason"]).most_common(60)),
                   fail_rxn={str(r.rxn_id): r.reason for r in S[S.status != "ok"].itertuples()},
                   n_imag={"0": ni.get(0, 0), "1": ni.get(1, 0), "2": ni.get(2, 0),
                           ">=3": sum(v for k, v in ni.items() if k >= 3)},
                   lowest_freq_cm=qstats(okd.lowest_freq_cm), lowest_freq_cm_n_imag1=qstats(
                       okd.loc[okd.n_imag == 1, "lowest_freq_cm"]),
                   forming_share=qstats(okd.imag_mode_forming_share), n_share_below_0p5=int(
                       (okd.imag_mode_forming_share.astype(float) < 0.5).sum()),
                   b6_not_computable=dict(n=len(b6_bad), rate=rates["b6"], threshold=B6_MAX,
                                          reasons=dict(Counter(b6_bad.b6_check).most_common(20)),
                                          rxn=[int(r) for r in b6_bad.rxn_id][:500]),
                   ts_fragments_fail=dict(Counter(okd.loc[okd.ts_fragments != "ok", "ts_fragments"]).most_common(20)),
                   b6_qc_max=dict(nu_diff_cm=max([abs(x["qc"].get("qc_b6_nu_diff_cm", 0.0)) for x in gate["rows"]
                                                  if x.get("b6") == "ok"] or [0.0]),
                                  share_diff=max([abs(x["qc"].get("qc_b6_mode_share_diff", 0.0)) for x in gate["rows"]
                                                  if x.get("b6") == "ok"] or [0.0])),
                   freq=dict(header=ex.get("ts_header"), orca_bin=ex.get("orca_bin"), orca_version=ex.get("orca_version"),
                             wall_s=qstats(okd.freq_wall_s), hess_geom_max_dev_A=qstats(okd.hess_geom_max_dev_A)),
                   core_h=dict(per_rxn_sum=float(S.core_h.dropna().sum()), per_rxn=qstats(S.core_h),
                               alloc_g0_steps=float(sum((v.get("core_h_alloc") or 0.0) for v in steps.values()))),
                   bytes=dict(total=int(sum(folder_bytes(DFT_FINAL / str(r)) for r in done)),
                              note="dft_final .done folders (failed folders are small)"),
                   status_csv=str(G0_STATUS), status_csv_sha256=L.sha256(G0_STATUS),
                   done_ids_sha256=DP.list_sha([(r,) for r in done]), gate_json=str(gate_json),
                   gate_json_sha256=L.sha256(gate_json), g2_inputs_sha256=g2sha,
                   stop=bool(stop), stop_reasons=stop)
        L.write_json(G0_REPORT, clean(rep))
        rec["report"] = {k: rep.get(k) for k in ("counts", "rates", "n_imag", "stop", "stop_reasons")}
        return 3 if stop else 0
    return step("G0", "g0_report", body)


# ================================================================ F0 (venv5, d1_pair jobs/g.slurm F env)
_G0S = {}


def prior_f0(rec):
    """F0 prerequisites -> {ids, done, fail}: start report, D1-G2 Phase T F_ready (root paths validated with the G2
    inputs), G2 inputs sha256, G0 passed (G0_report complete / no STOP, G0_status.csv unchanged) and a live scan of
    $LAD/dft_final equal to it. Stop 2 (not ready) / 3."""
    if _G0S:
        rec.update(g0=_G0S["summary"], g2_inputs=_G0S["g2_inputs"])
        return _G0S
    start_ok(rec)
    t = read_json(G2_T_REPORT)
    if t.get("F_ready") is not True or t.get("all_pass") is not True:
        raise Stop("phase_T", f"{G2_T_REPORT}: F_ready {t.get('F_ready')!r} (the D1-G2 root paths must be validated)", 3)
    g2_inputs_check(rec)
    g2i = rec["g2_inputs"]                                  # cached only once every check below has passed
    rep = read_json(G0_REPORT)
    if not rep:
        raise Stop("g0", f"{G0_REPORT} missing (g0_report)", 2)
    if rep.get("stop"):
        raise Stop("g0", f"G0 STOP: {rep.get('stop_reasons')}", 3)
    if not rep.get("complete"):
        raise Stop("g0", "G0 incomplete (G0_report.json complete false)", 2)
    if L.sha256(G0_STATUS) != rep.get("status_csv_sha256"):
        raise Stop("g0", "G0_status.csv changed after G0_report.json (rerun g0_report)", 2)
    tg = target_rows(rec)
    ids = sorted(int(d["rxn_id"]) for d in tg)
    done, fail, open_ = set(), {}, []
    for r in ids:
        mk = marker(DFT_FINAL / str(r))
        if mk == ".done":
            done.add(r)
        elif mk:
            fail[r] = mk
        else:
            open_.append(r)
    if open_ or DP.list_sha([(r,) for r in sorted(done)]) != rep.get("done_ids_sha256"):
        raise Stop("g0", f"live scan of {DFT_FINAL} differs from G0_report ({len(open_)} without marker, {len(done)} "
                         f".done): rerun g0_report", 2)
    summary = dict(n_targets=len(ids), n_done=len(done), n_fail=len(fail), G0_report_sha256=L.sha256(G0_REPORT),
                   fail_markers=dict(Counter(fail.values()).most_common(30)))
    _G0S.update(ids=ids, done=done, fail=fail, summary=summary, g2_inputs=g2i)
    rec["g0"] = summary
    return _G0S


def prior_all(rec):
    env_f(rec)
    return prior_f0(rec)


def ids_text(ids):
    return "rxn_id\n" + "".join(f"{r}\n" for r in sorted(ids))


def inputs_args():
    return [a for k in ("labels", "meta", "prof") for a in (f"--{k}", G2_INPUTS[k])]


def feat_env():
    """d1_features.py children: D1G2_SCRATCH=$LAD so its ROOT_WRITE_OK guard accepts $LAD (and refuses $G2)."""
    return dict(os.environ, D1G2_SCRATCH=str(L.LAD), PYTHONDONTWRITEBYTECODE="1")


def atoms_env():
    """= phase_fh.atoms_env with the DFT-rung roots."""
    return dict(os.environ, D1PAIR_PHASE="FROOT", G1_ROOT=str(DFT_FINAL), ESPLEY_OUT=str(F0_DIR / "fu2_xtb"),
                R5_SCRATCH=str(F0_DIR / "fu2_r5"), ESPLEY_LABELS=str(G2_INPUTS["labels"]),
                ESPLEY_META=str(G2_INPUTS["meta"]), ESPLEY_PROF=str(G2_INPUTS["prof"]), FU2_SCR=str(FU2_SCR),
                FU2_RES=str(FU2_RES), FU2_X182=str(X182_DFT), PYTHONDONTWRITEBYTECODE="1", **VERSIONS)


def ts_of(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def log_segments(path):
    """(cpus from the job header, [(command, wall_s, rc)], n rxns of the first '[F slice ..] / [Q ..] N rxns' line)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    m = RE_CPUS.search(lines[0]) if lines else None
    segs, cur, nrx = [], None, None
    for ln in lines:
        e = RE_EXIT.match(ln)
        if e:
            if cur:
                segs.append((cur[0], ts_of(e.group(2)) - cur[1], int(e.group(1))))
            cur = None
            continue
        c = RE_CMD.match(ln)
        if c:
            cur = (c.group(1).strip(), ts_of(c.group(2)))
            continue
        k = RE_NRX.match(ln)
        if k and nrx is None:
            nrx = int(k.group(1))
    return (int(m.group(1)) if m else None), segs, nrx


def estimate(n_done):
    """F0 core-h from the D1PAIR G1 F / Q logs (allocated core-h = wall x cpus), per reaction x n_done."""
    fr = read_json(DP.RES / "F_report.json")
    qr = read_json(DP.RES / "fu2" / "Q_report.json")
    n_ref = dict(F=fr.get("n_g1_done"), Q=qr.get("n_target"))
    out, total = {}, 0.0
    for name, (d, files, pre) in EST_LOGS.items():
        ch, n, rcs, used = 0.0, 0, [], []
        for f in files:
            c, segs, nrx = log_segments(d / f)
            for cmd, w, rc in segs:
                if cmd.startswith(pre):
                    ch += w * (c or 0) / 3600.0
                    rcs.append(rc)
                    used.append(dict(file=f, cmd=cmd, wall_s=w, cpus=c, rc=rc))
            n += nrx or 0
        ref = n_ref["F"] if name.startswith("F") else n_ref["Q"]
        per = ch / (n if name.endswith("_run") else ref) if (n if name.endswith("_run") else ref) else float("nan")
        est = per * n_done
        total += est
        out[name] = dict(core_h_logs=ch, n_rxn_logs=n if name.endswith("_run") else None, n_ref=ref,
                         core_h_per_rxn=per, est_core_h=est, all_exit_0=all(r == 0 for r in rcs) and bool(rcs),
                         segments=used)
    return dict(spec="docs/specs/D1LADDER.md 6 (estimate before F0 starts)", written=now(), n_done=n_done,
                rule="allocated core-h (wall x cpus) of the D1PAIR G1 F / Q runs per reaction x the G0 .done count",
                n_ref=n_ref, n_match=dict(F=out["F_run"]["n_rxn_logs"] == n_ref["F"],
                                          Q=out["Q_run"]["n_rxn_logs"] == n_ref["Q"]),
                parts=out, total_core_h=total,
                plan="f0_slice 2 x 32 cores (array %2) + f0_rest 1 x 48 cores; G0 2 x 16 + g0_report 16 cores")


def cmd_f0_prep():
    def body(rec):
        g = prior_all(rec)
        txt = ids_text(g["ids"])
        if F0_IDS.is_file():
            if F0_IDS.read_text() != txt:
                raise Stop("ids", f"{F0_IDS} differs from the target rxn_ids (use a fresh $LAD/F0)", 3)
            rec["ids"] = "exists, equal"
        else:
            L.write_atomic(F0_IDS, txt)
        est = estimate(len(g["done"]))
        L.write_json(F0_EST, clean(est))
        rec["estimate"] = dict(total_core_h=est["total_core_h"], n_match=est["n_match"],
                               parts={k: v["est_core_h"] for k, v in est["parts"].items()})
        return 0
    return step("F0", "f0_prep", body)


def cmd_f0_estimate():
    def body(rec):
        g = prior_all(rec)
        est = estimate(len(g["done"]))
        L.write_json(F0_EST, clean(est))
        rec["estimate"] = dict(total_core_h=est["total_core_h"], n_match=est["n_match"])
        return 0
    return step("F0", "f0_estimate", body)


def cmd_f0_slice(i, n):
    def body(rec):
        g = prior_all(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        if not F0_IDS.is_file() or F0_IDS.read_text() != ids_text(g["ids"]):
            raise Stop("ids", f"{F0_IDS} missing or not the target list (run f0_prep)", 2)
        if not F0_EST.is_file():
            raise Stop("estimate", f"{F0_EST} missing (run f0_prep: the estimate comes before F0)", 2)
        other = sorted({p.name.split("of")[1][:2] for p in F0_SLICES.glob("slice*_root.json")} - {f"{n:02d}"})
        if other:
            raise Stop("slices", f"{F0_SLICES} has slices of n = {other}: mergeroot needs all slices of one n", 3)
        tag = f"slice{i:02d}of{n:02d}"
        have = [f for f in SLICE_FILES if (F0_SLICES / f"{tag}_{f}").is_file()]
        rec["slice"] = dict(tag=tag, dir=str(F0_SLICES), present=have)
        if len(have) == len(SLICE_FILES):
            rec["skipped"] = "slice outputs exist"
            return 0
        child(rec, "runroot", [L.P_DIR / "d1_features.py", "runroot", DFT_FINAL, F0_OUT, F0_IDS, i, n] + inputs_args(),
              L.P_DIR, feat_env())
        return 0
    return step("F0", f"f0_slice_{i:02d}of{n:02d}", body)


def merged():
    """the mergeroot report if it belongs to the current X182_d1_dft.parquet (sha256) and ids_all.csv, else None."""
    if not (F0_MREP.is_file() and X182_DFT.is_file()):
        return None
    m = read_json(F0_MREP)
    if (m.get("x182") or {}).get("sha256") != L.sha256(X182_DFT) or m.get("ids_csv") != str(F0_IDS):
        return None
    return m


def need_merged(rec):
    m = merged()
    if m is None:
        raise Stop("f0_merge", "no mergeroot report for the current X182_d1_dft.parquet (run f0_merge)", 2)
    if m.get("stop"):
        raise Stop("f0_merge", "mergeroot report stop (columns != X182_d1 or non-finite values: F-1)", 3)
    rec["x182"] = dict(path=str(X182_DFT), sha256=m["x182"]["sha256"], n=m["x182"]["n"])
    return m


def cmd_f0_merge():
    def body(rec):
        g = prior_all(rec)
        m = merged()
        if m is None:
            child(rec, "mergeroot", [L.P_DIR / "d1_features.py", "mergeroot", F0_OUT, X182_DFT, F0_MREP], L.P_DIR,
                  feat_env(), ok=(0, 3))
            m = merged()
            if m is None:
                raise Stop("mergeroot", f"{F0_MREP} does not match {X182_DFT} after mergeroot", 3)
        else:
            rec["skipped"] = "mergeroot report matches X182_d1_dft.parquet"
        rec["mergeroot"] = {k: m.get(k) for k in ("n_slices", "n_requested", "n_done", "n_feat", "F_fail_rate", "x182",
                                                  "F1_all_finite", "stop")}
        if m.get("n_done") != len(g["done"]) or m.get("n_requested") != len(g["ids"]):
            raise Stop("mergeroot", f"n_done {m.get('n_done')} / n_requested {m.get('n_requested')} != G0 .done "
                                    f"{len(g['done'])} / targets {len(g['ids'])}", 3)
        if m.get("stop"):
            raise Stop("mergeroot", "mergeroot report stop (F-1: columns != X182_d1 or non-finite values)", 3)
        return 0
    return step("F0", "f0_merge", body)


def atoms_counts():
    a = FU2_SCR / "atoms"
    return dict(npz=len(list(a.glob("*.npz"))), fail=len(list(a.glob("*.fail.json"))), err=len(list(a.glob("*.err.json"))))


def cmd_f0_atoms(i, n):
    def body(rec):
        prior_all(rec)
        need_merged(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        env = atoms_env()
        rec["child_env"] = {k: env[k] for k in ATOMS_ENV_KEYS}
        child(rec, "atoms_run", [L.P_DIR / "fu2" / "atoms.py", "root", "run", i, n], L.P_DIR, env)
        rec["atoms"] = atoms_counts()
        return 0
    return step("F0", f"f0_atoms_{i:02d}of{n:02d}", body)


def cmd_f0_gate():
    err = sorted((FU2_SCR / "atoms").glob("*.err.json"))
    if err and read_json(RETRY_REC).get("rc") != 0:        # no retry pass exited 0 yet (phase_fh f_gate rule)
        def retry(rec):
            prior_all(rec)
            need_merged(rec)
            rec["err_before"] = [p.name for p in err][:500]
            rec["n_err_before"] = len(err)
            child(rec, "atoms_retry", [L.P_DIR / "fu2" / "atoms.py", "root", "run", 0, 1], L.P_DIR, atoms_env())
            rec["atoms"] = atoms_counts()
            return 0
        rc = step("F0", "f0_atoms_retry", retry)
        if rc:
            return rc

    def body(rec):
        prior_all(rec)
        need_merged(rec)
        rec["atoms"] = atoms_counts()
        rc = child(rec, "atoms_gate", [L.P_DIR / "fu2" / "atoms.py", "root", "gate"], L.P_DIR, atoms_env(), ok=(0, 2, 3))
        q = FU2_RES / "Q_report.json"
        if rc == 2:
            rec["note"] = "atoms incomplete (atoms.py gate exit 2): run f0_atoms"
        elif q.is_file():
            r = read_json(q)
            rec["Q_report"] = {k: r.get(k) for k in Q_KEYS}
        return rc
    return step("F0", "f0_gate", body)


def cmd_f0_tokens():
    def body(rec):
        prior_all(rec)
        need_merged(rec)
        q = FU2_RES / "Q_report.json"
        if not q.is_file():
            raise Stop("Q", f"no {q} (run f0_gate)", 2)
        r = read_json(q)
        if r.get("stop"):
            raise Stop("Q", f"Q gate STOP: {r.get('stop_reason')}", 3)
        rc = child(rec, "tokens_build", [L.P_DIR / "fu2" / "atoms.py", "root", "tokens"], L.P_DIR, atoms_env(), ok=(0, 3))
        t = FU2_RES / "T_report.json"
        if t.is_file():
            tr = read_json(t)
            rec["T_report"] = {k: tr.get(k) for k in T_KEYS}
        return rc
    return step("F0", "f0_tokens", body)


def reason_class(w):
    """a failure reason without its numeric tail (e.g. ext:B1:engine_gate1:1.4e-06 -> ext:B1:engine_gate1)."""
    w = re.sub(r":[-+]?\d[\d.eE+-]*$", "", str(w))
    return w if len(w) <= 70 else w[:70] + "..."


def cmd_f0_report():
    def body(rec):
        import pandas as pd
        g = prior_all(rec)
        done, n_done = g["done"], len(g["done"])
        rep = dict(spec="docs/specs/D1LADDER.md 6 (STOP: 16 F0)", written=now(), decisions=F0_DECISIONS,
                   threshold=F0_RATE_MAX, root=str(DFT_FINAL), n_targets=len(g["ids"]), n_g0_done=n_done,
                   g0_not_done=dict(Counter(g["fail"].values()).most_common()), inputs=rec.get("g2_inputs"),
                   versions=VERSIONS, child_env_features=dict(D1G2_SCRATCH=str(L.LAD)),
                   child_env_atoms={k: atoms_env()[k] for k in ATOMS_ENV_KEYS}, xtb_threads=os.environ.get("XTB_THREADS"),
                   core_h_estimate=read_json(F0_EST) or None, g0=g["summary"])
        stop, missing = [], []
        steps = {p.stem: read_json(p) for p in sorted(F0_STEPS.glob("*.json")) if p.stem != "f0_report"}
        rep["steps"] = {k: dict(rc=v.get("rc"), finished=v.get("finished"), slurm_job=v.get("slurm_job"),
                                slurm_array_task=v.get("slurm_array_task"), cpus=v.get("cpus"), wall_s=v.get("wall_s"),
                                exit=v.get("exit"), fail=v.get("fail"), skipped=v.get("skipped"))
                        for k, v in steps.items()}
        gate_rec = steps.get("f0_gate") or {}
        superseded = [k for k, v in steps.items() if k == "f0_atoms_retry" and v.get("rc") != 0
                      and gate_rec.get("rc") == 0 and str(gate_rec.get("started") or "") >= str(v.get("finished") or "~")]
        rep["superseded_steps"] = superseded
        stop += [f"step {k} exit 3: {v.get('fail_stage')}: {v.get('fail')}" for k, v in steps.items()
                 if v.get("rc") == 3 and k not in superseded]
        rep["core_h_actual"] = dict(alloc_f0_steps=float(sum((v.get("core_h_alloc") or 0.0) for v in steps.values())),
                                    note="wall x cpus of the latest record of each step (a rerun overwrites its record)")
        # ---- features (F-1, B6)
        feat_fail = f1_fail = b6_fail = x_ids = None
        m = merged()
        if m is None:
            missing.append(f"{F0_MREP} for the current {X182_DFT} (f0_merge)")
        else:
            not_ok = {int(r): w for r, w in (m.get("not_ok") or {}).items()}
            root_miss = {r: w for r, w in not_ok.items() if str(w).startswith("root:")}
            feat_fail = {r: w for r, w in not_ok.items() if not str(w).startswith("root:")}
            X1 = pd.read_parquet(X182_DFT)
            X0_cols = list(pd.read_parquet(DP.X182_D1).columns)
            x_ids = set(int(r) for r in X1.rxn_id)
            cols_equal = list(X1.columns) == X0_cols and len(X0_cols) == 183
            E = pd.read_parquet(F0_EXTP, columns=["rxn_id", "ext_status"] + [f"status_B{k}" for k in range(1, 7)])
            G1 = pd.read_parquet(F0_G1P, columns=["rxn_id", "xtb_status"])
            E["rxn_id"] = E.rxn_id.astype(int)
            st = E.set_index("rxn_id")
            b6_fail = {int(r): str(s) for r, s in st.status_B6.items() if str(s) not in ("ok", "skipped")}
            b6_only = {r for r in b6_fail if all(str(st.at[r, f"status_B{k}"]) == "ok" for k in range(1, 6))}
            # F-1 as in D1PAIR (REV9: all 182 columns finite) = every .done reaction missing from X182_d1_dft, B6-only
            # failures included; B6 keeps its own 1 % gate on top (fixer 2026-10-22: B6-only rows were excluded before)
            f1_fail = dict(feat_fail)
            S0 = pd.read_csv(G0_STATUS)
            g0_b6 = set(int(r) for r in S0.loc[(S0.marker == ".done") & (S0.ts_fragments == "ok")
                                               & (S0.b6_check != "ok"), "rxn_id"])
            # G0 prediction (block_b6 on the folders) vs F0 status_B6, on the rows whose B6 block really ran in F0
            # (not geometry-skipped, not a lost record 'error:...'; those still count as B6 failures above)
            f0_ran = set(int(r) for r, s in st.status_B6.items() if str(s) != "skipped" and not str(s).startswith("error:"))
            b6_agree = (g0_b6 & f0_ran) == (set(b6_fail) & f0_ran)
            f0_geom_ok = f0_ran
            rep["features"] = dict(
                x182=dict(path=str(X182_DFT), sha256=m["x182"]["sha256"], n=len(X1), columns_equal_X182_d1=cols_equal,
                          X182_d1=str(DP.X182_D1)),
                mergeroot_report=dict(path=str(F0_MREP), sha256=L.sha256(F0_MREP)), n_requested=m.get("n_requested"),
                n_done=m.get("n_done"), n_feat=m.get("n_feat"), n_slices=m.get("n_slices"),
                F1_all_finite=m.get("F1_all_finite"), n_feature_fail=len(feat_fail),
                feature_fail_classes=dict(Counter(reason_class(w) for w in feat_fail.values()).most_common(40)),
                feature_fail_rxn=dict(sorted(feat_fail.items())), xtb_status=m.get("xtb_status"),
                g1_rows=len(G1), ext_rows=len(E), versions=m.get("versions"), si_decision=m.get("si_decision"),
                sha256=m.get("sha256"))
            rep["F1"] = dict(n=len(f1_fail), rate=len(f1_fail) / n_done if n_done else None,
                             rule="every G0 .done reaction without a finite 182-feature row in X182_d1_dft (mergeroot "
                                  "not-ok: xtb_status, ext geom / B1..B6, non-finite), as D1PAIR F-1",
                             n_B6_only=len(b6_only), n_without_B6_only=len(f1_fail) - len(b6_only & set(f1_fail)),
                             classes=dict(Counter(reason_class(w) for w in f1_fail.values()).most_common(40)),
                             n_B1_engine_gate=sum(1 for w in f1_fail.values() if "B1:engine_gate" in str(w)))
            rep["B6_not_computed"] = dict(n=len(b6_fail), rate=len(b6_fail) / n_done if n_done else None,
                                          n_B6_only=len(b6_only), reasons=dict(Counter(b6_fail.values()).most_common(20)),
                                          rxn=sorted(b6_fail)[:500], g0_prediction_equal=bool(b6_agree),
                                          n_g0_predicted=len(g0_b6))
            if m.get("stop") or not cols_equal or not m.get("F1_all_finite"):
                stop.append(f"F-1 (mergeroot): stop {m.get('stop')}, columns_equal_X182_d1 {cols_equal}, "
                            f"F1_all_finite {m.get('F1_all_finite')}")
            if (m.get("n_done") != n_done or set(root_miss) != set(g["fail"]) or x_ids != done - set(feat_fail)
                    or set(int(r) for r in G1.rxn_id) != done):
                stop.append(f"inconsistent feature counts: n_done {m.get('n_done')} vs G0 .done {n_done}, root: "
                            f"{len(root_miss)} vs G0 not done {len(g['fail'])}, X182 rows {len(x_ids)} vs "
                            f"{n_done - len(feat_fail)}, g1 rows {len(G1)}")
            if not b6_agree:
                stop.append(f"B6 status differs between G0 (block_b6 on the folders: {len(g0_b6 & f0_geom_ok)}) and F0 "
                            f"(status_B6: {len(b6_fail)})")
        # ---- Q (Q-a, Q-b, Q-c)
        qa_fail = q_excl = None
        qf, qm = FU2_RES / "Q_report.json", FU2_RES / "Q_manifest.csv"
        if not (qf.is_file() and qm.is_file()):
            missing.append(f"{qf} / {qm} (f0_gate)")
        else:
            q = read_json(qf)
            Q = pd.read_csv(qm)
            q_excl = sorted(int(r) for r in Q.loc[Q.use.astype(str) != "True", "rxn_id"])
            qa_fail = sorted(int(r) for r in Q.loc[Q.qa_ok.astype(str) != "True", "rxn_id"])
            ov = q.get("overrides") or {}
            rep["Q"] = dict({k: q.get(k) for k in Q_KEYS}, n_qa_fail_manifest=len(qa_fail), qa_fail=qa_fail[:500],
                            n_excluded=len(q_excl), excluded=q_excl[:500], report_sha256=L.sha256(qf),
                            manifest_sha256_file=L.sha256(qm), overrides=ov)
            if q.get("stop"):
                stop.append(f"Q gate STOP (atoms.py gate, unchanged): {q.get('stop_reason')}")
            want = dict(FU2_X182=str(X182_DFT), FU2_ATOM_VERSION=VERSIONS["FU2_ATOM_VERSION"], G1_ROOT=str(DFT_FINAL),
                        FU2_SCR=str(FU2_SCR), FU2_RES=str(FU2_RES))
            if any(ov.get(k) != v for k, v in want.items()) or q.get("manifest_sha256") != L.sha256(qm):
                stop.append(f"Q_report is not from this run: overrides {ov}, manifest sha256 {q.get('manifest_sha256')}")
            if x_ids is not None and set(int(r) for r in Q.rxn_id) != x_ids:
                stop.append(f"Q_manifest rows ({len(Q)}) != X182_d1_dft rows ({len(x_ids)})")
        # ---- T (T-a)
        t_bad = tok_ok = None
        tf, tm = FU2_RES / "T_report.json", FU2_RES / "T_manifest.csv"
        if not (tf.is_file() and tm.is_file()):
            missing.append(f"{tf} / {tm} (f0_tokens)")
        else:
            t = read_json(tf)
            T = pd.read_csv(tm)
            t_bad = sorted(int(b["rxn_id"]) for b in (t.get("bad") or []))
            no_file = [int(r) for r, s in zip(T.rxn_id, T.sha256)
                       if not (FU2_SCR / "tokens" / f"{int(r)}.npz").is_file()
                       or L.sha256(FU2_SCR / "tokens" / f"{int(r)}.npz") != s]
            tok_ok = set(int(r) for r in T.rxn_id) - set(t_bad) - set(no_file)
            L.write_atomic(F0_TOK_MAN, tm.read_bytes())
            rep["T"] = dict({k: t.get(k) for k in T_KEYS}, n_bad=len(t_bad), bad=t.get("bad"),
                            n_token_files_missing_or_changed=len(no_file), token_files_missing_or_changed=no_file[:100],
                            report_sha256=L.sha256(tf), manifest=dict(path=str(tm), sha256=L.sha256(tm)),
                            tokens_dir=str(FU2_SCR / "tokens"),
                            F0_tokens_manifest=dict(path=str(F0_TOK_MAN), sha256=L.sha256(F0_TOK_MAN)))
            if t.get("stop"):
                stop.append(f"T gate STOP (tokens.py, unchanged): gate_Ta {t.get('gate_Ta')}")
            if no_file:
                stop.append(f"{len(no_file)} token npz missing or not equal to T_manifest.csv")
            if t.get("token_version") != VERSIONS["FU2_TOKEN_VERSION"] or (
                    X182_DFT.is_file() and t.get("x182_sha256") != L.sha256(X182_DFT)) or (
                    qf.is_file() and t.get("q_report_sha256") != L.sha256(qf)):
                stop.append("T_report is not from this run (token version / X182 / Q_report sha256)")
            # the dft rung must have the token layout of the g1 rung (D1PAIR fu2-tokens-3): same model inputs (spec 4-2,
            # 4-3); a dropped extension (FU2 Q-1: ext pass rate < 0.99 -> no mu_abs / f_plus / f_minus) is a STOP
            tref_p = DP.RES / "fu2" / "T_report.json"
            tref = read_json(tref_p)
            lay = {k: (t.get(k), tref.get(k)) for k in T_LAYOUT_KEYS}
            lay_bad = [k for k, (a, b) in lay.items() if a != b]
            rep["T"]["layout_vs_d1pair_g1"] = dict(
                ref=str(tref_p), ref_sha256=L.sha256(tref_p) if tref_p.is_file() else None, differ=lay_bad,
                elements=dict(dft=t.get("elements"), g1=tref.get("elements"),
                              note="reported only: the element one-hot is built per rung from its own T_report"),
                ext_pass_rate=(rep.get("Q") or {}).get("ext_pass_rate"))
            if not tref or lay_bad:
                stop.append(f"token layout differs from the D1PAIR g1 rung ({tref_p}) in {lay_bad or 'all (missing)'}: "
                            "the rungs would not share the model input layout (spec 4-2 / 4-3)")
        # ---- spec 16 F0 rates
        rep["counts"] = dict(targets=len(g["ids"]), g0_done=n_done, features_ok=len(x_ids) if x_ids is not None else None,
                             q_use=(len(x_ids) - len(q_excl)) if (x_ids is not None and q_excl is not None) else None,
                             tokens_ok=len(tok_ok) if tok_ok is not None else None)
        if not missing:
            comp = {"F-1": len(f1_fail), "B6": len(b6_fail), "Q-a": len(qa_fail), "T-a": len(t_bad)}
            rate = {k: (v / n_done if n_done else float("nan")) for k, v in comp.items()}
            over = [k for k, v in rate.items() if not v <= F0_RATE_MAX]
            qmax = (rep.get("Q") or {}).get("qa_max_fail")
            rep["gate_rates"] = dict(n=comp, n_g0_done=n_done, rate=rate, threshold=F0_RATE_MAX, over=over,
                                     effective_limits=dict(
                                         F_1="> 1 % of the G0 .done reactions (all missing from X182_d1_dft)",
                                         B6="> 1 % of the G0 .done reactions",
                                         Q_a=f"> {qmax} reactions (FU2 QA_MAX_FAIL, absolute: "
                                             f"{(qmax / n_done if (qmax is not None and n_done) else float('nan')):.4%} of "
                                             f"{n_done}) by the unchanged atoms.py gate, or > 1 %",
                                         T_a="any bad row by the unchanged tokens.py gate, or > 1 %",
                                         reading="spec 6 applies FU2 Q-a / T-a 'as is' (their own STOP verdicts) and "
                                                 "spec 16 adds > 1 %: both are applied, so Q-a and T-a stop earlier "
                                                 "than 1 % (recorded reading of a spec conflict)"),
                                     combined_info=dict(n=n_done - len(tok_ok), rate=(n_done - len(tok_ok)) / n_done
                                                        if n_done else None, rule="G0 .done without a token npz "
                                                        "(D1-G2 phase_fh F definition; information only)"))
            stop += [f"{k} failures {comp[k]}/{n_done} = {rate[k]:.4f} > {F0_RATE_MAX} (spec 16 F0)" for k in over]
        sha = {}
        for p in (X182_DFT, F0_MREP, F0_G1P, F0_EXTP, F0_IDS, F0_EST, G0_STATUS, G0_REPORT, qf, qm, tf, tm, F0_TOK_MAN):
            if Path(p).is_file():
                sha[str(p)] = L.sha256(p)
        rep.update(sha256=sha, missing=missing, complete=not missing, stop=bool(stop), stop_reasons=stop)
        L.write_json(F0_REPORT, clean(rep))
        rec["report"] = {k: rep.get(k) for k in ("counts", "gate_rates", "complete", "stop", "stop_reasons", "missing")}
        return 3 if stop else (2 if missing else 0)
    return step("F0", "f0_report", body)


def cmd_f0_rest():
    seq = [("f0_merge", cmd_f0_merge), ("f0_atoms 0 1", lambda: cmd_f0_atoms(0, 1)), ("f0_gate", cmd_f0_gate),
           ("f0_tokens", cmd_f0_tokens)]
    rc = 0
    for name, fn in seq:
        r = fn()
        print(f"[f0_rest] {name}: exit {r}  ({now()})", flush=True)
        if r:
            rc = r
            break
    r = cmd_f0_report()
    print(f"[f0_rest] f0_report: exit {r}", flush=True)
    return max(rc, r)


# ================================================================ CLI
def main(a):
    cmd, n = (a[0] if a else ""), len(a)
    if cmd == "g0" and n in (3, 4) and (n == 3 or a[3] == "retry"):
        return cmd_g0(array_k(a[1]), int(a[2]), retry=n == 4)
    if cmd == "_g0check" and n == 3:
        return _g0check(a[1], a[2])
    if cmd == "f0_slice" and n == 3:
        return cmd_f0_slice(array_k(a[1]), int(a[2]))
    if cmd == "f0_atoms" and n == 3:
        return cmd_f0_atoms(array_k(a[1]), int(a[2]))
    plain = {"g0_report": cmd_g0_report, "f0_prep": cmd_f0_prep, "f0_estimate": cmd_f0_estimate,
             "f0_merge": cmd_f0_merge, "f0_gate": cmd_f0_gate, "f0_tokens": cmd_f0_tokens, "f0_report": cmd_f0_report,
             "f0_rest": cmd_f0_rest}
    if cmd in plain and n == 1:
        return plain[cmd]()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
