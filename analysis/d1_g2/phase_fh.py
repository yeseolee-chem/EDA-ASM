#!/usr/bin/env python3
r"""D1-G2 Phase F (docs/specs/D1G2.md 8) and Phase H (9) on every G2-ok reaction; STOP rules of spec 14 F / H.
Run through sbatch only. No method, threshold or gate is changed: this module only drives the paths Phase T validated
(test_t.py t1 / t2 --inputs g2, t3) and Phase S ran on $G2/g2_final (s_smoke.py f / h), with test_t.run as the child
launcher and the s_smoke child environments, and writes the step records and the two gate reports.
(User 2026-10-22: no final report -> Phase R / RUN_REPORT_G2.md is not made here; F_report.json / H_report.json are the
spec 8 / 9 / 14 gate records.)

Every command first checks: its job environment (test_t.env_f for F = d1_pair jobs/g.slurm with PYENV=venv5,
D1PAIR_PHASE=F; test_t.env_hf for H = d1_hf jobs/hf.slurm), the Phase T gate (results/T_report.json F_ready for F,
all_pass for H; results/S_report.json: see 'Phase S' below), and that Phase G is finished: results/G_report.json
all_finished and no STOP, G_status.csv = the sha256 in G_report, and a live scan of $G2/g2_final in which every
X_targets rxn has .done or .fail_<reason> (ok count = G_status 'ok'). Each command writes results/F_steps/<step>.json or
results/H_steps/<step>.json (children's exit codes, host, job, times; a rerun of a step overwrites its record).
Exit codes: 0 done; 2 not ready / incomplete (resubmit the missing part later); 3 STOP (spec 14) or a hard error
(environment, inconsistent inputs, unexpected child exit) -- read the step json. The only automatic retries are the
atoms.py err pass in f_gate (until one pass exits 0) and run.py's own SlowConv retry; nothing else is retried.

CLI
  F (submit from analysis/d1_pair: jobs/g.slurm, --export=ALL,PYENV=venv5,D1PAIR_PHASE=F):
    f_slice <i> <n>    d1_features.py runroot $G2/g2_final $G2/F/out $G2/F/ids_all.csv i n --labels --meta --prof (the
                       G2 inputs $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv, $G2/prof); ids_all.csv = all 9,266
                       X_targets rxn_ids, so a G2 failure is listed by mergeroot as root:<marker>. Skipped when the slice's
                       g1 / ext parquets and root / versions records exist; refused if slices of another n exist.
    f_merge            d1_features.py mergeroot $G2/F/out $G2/X182_d1_g2.parquet $G2/F/mergeroot_report.json (needs all
                       slices of one n; column order = X182_d1, sha256 in the report); skipped when the report's x182
                       sha256 = the parquet's. n_done must equal the G2 .done count.
    f_atoms <i> <n>    fu2/atoms.py root run i n in the FU2 child environment (below); per-reaction cache in atoms.py.
    f_gate             if <rid>.err.json remain ($G2/fu2/atoms; atoms.py: retryable, recomputed on a rerun) and no retry
                       pass has exited 0 yet: one pass 'root run 0 1' (step f_atoms_retry; at most one per f_gate call);
                       then fu2/atoms.py root gate (Q gates unchanged; exit 2 = atoms incomplete, 3 = STOP). A failed
                       retry record is superseded in f_report by a later f_gate record with exit 0.
    f_tokens           fu2/atoms.py root tokens (T-a gate unchanged; exit 3 = STOP); refused after a Q STOP.
    f_report           results/F_report.json + results/F_tokens_manifest.csv (byte copy of results/fu2/T_manifest.csv).
    f_all              f_slice 0..3 of 4, f_merge, f_atoms 0..3 of 4, f_gate, f_tokens (stops at the first non-zero
                       step), then always f_report; exit = the worst of them.
    FU2 child environment (= s_smoke.py f): D1PAIR_PHASE=FROOT, G1_ROOT=$G2/g2_final, ESPLEY_LABELS / META / PROF = the
    G2 inputs, ESPLEY_OUT=$G2/F/fu2_xtb, R5_SCRATCH=$G2/F/fu2_r5 (set; atoms do not read them), FU2_SCR=$G2/fu2,
    FU2_RES=analysis/d1_g2/results/fu2, FU2_X182=$G2/X182_d1_g2.parquet, FU2_ATOM_VERSION=fu2-atoms-3-g2,
    FU2_TOKEN_VERSION=fu2-tokens-3-g2.
  H (submit from analysis/d1_hf: jobs/hf.slurm; child environment = s_smoke.py h with the full-run roots:
     D1HF_SCRATCH=$G2 (inputs $G2/inputs/<T>/g2, runs $G2/runs/g2, logs $G2/logs), D1HF_RES=analysis/d1_g2/results,
     D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs (read only), D1HF_G2_ROOT=$G2/g2_final):
    h_inputs [W]       prep.py geominputs g2 <all 9,266 job_ids> -> results/g2_inputs.csv; the status-ok job_ids are cut
                       into nb = W * ceil(n_ok / (1000 W)) batches (<= 1,000 each; sorted by (n_atoms, job_id) and dealt
                       round robin = the D1HF prep.batches rule; each file sorted by job_id) -> results/H_batches/H_bNN.csv
                       (job_id, n_atoms) + plan.json (worker k of W owns the batches j with j % W == k). Default W = 7.
                       Skipped when plan.json exists for this W and its files / g2_inputs.csv are unchanged; refused when
                       a plan exists for another W or other inputs.
    h_batch <k|A> <W>  worker k of W (A = $SLURM_ARRAY_TASK_ID): spec 14 H guard first (no new run when the cumulative
                       gate failure rate over the current plan's batch checks already written is > 1 %), then ONE run.py
                       run <its batches as one list, results/H_batches/H_w<k>of<W>.csv, largest n_atoms first> g2 hf
                       $SLURM_CPUS_PER_TASK (one thread pool per job = balanced cores; the order changes no result), then
                       build.py check <batch> g2 for each of its batches (-> results/M_checks/H_bNN.json). The D1HF M
                       checkpoint verdict (any single gate failure / ok < 90 %) is recorded only: D1G2 spec 14 H replaces
                       it with the 1 % rule, so build.py gate_prev is not used. Exit 3 = cumulative rate > 1 % after the
                       run (spec 14 H STOP) or a run.py hard error.
    h_build            build.py allgeom - g2 -> $G2/labels_d1_hf_g2.json, results/labels_d1_hf_g2.parquet, H_status.csv.
    h_report           results/H_report.json (needs a successful h_build whose H_status.csv is unchanged).

Gate failure rates (spec 14 F, H: "the existing gate failure rate > 1 % = STOP"), defined here:
  F  n_g2_ok = targets with .done in g2_final. gate_fail = G2-ok reactions without a token npz that passed every
     existing gate = (a) features: F_ok false in d1_features mergeroot (xtb_status / ext_status not ok, incl. the B1
     engine gate, or a non-finite value among the 182) + (b) Q: X182_d1_g2 rows with Q_manifest use = false (compute
     failure ts_fragments / h_map / err, Q-a not reproduced, Q-b, failed extension self-check when ext_used) + (c) T:
     T_report 'bad' rows (T-a). gate_fail_rate = gate_fail / n_g2_ok. STOP if > 0.01, or Q_report stop (Q-a > 42 =
     QA_MAX_FAIL, Q-b, Q-c), or T_report stop, or mergeroot stop (columns != X182_d1 / non-finite), or a step exit 3,
     or inconsistent counts. NB: D1PAIR Phase F on G1 lost 96 / 8,433 = 1.14 % at (a) alone (B1 engine gate).
     This combined definition (compute failures included) is a choice made here, not confirmed by the user; F_report
     also gives gate_failure.component_rates and Q n_excluded_compute_fail / n_excluded_gate_fail for other readings.
  H  finished = H_status ok + fail + gate_fail + eda_table_unexplained + parse_fail (a run folder with .done or .fail);
     gate_fail = gate_fail + eda_table_unexplained + parse_fail (build.py gates g1_asm, g2_sum, g3_seam, g_enuc,
     g_bond_ref, g_scf, g_setting, orca, the EDA table rows, the parser). gate_fail_rate = gate_fail / finished.
     STOP if > 0.01, or (decision here, same 1 % rule on the D1HF input check) prep.py geominputs status not ok for
     > 1 % of the G2-ok targets (expected 0: g2_autode re-maps every TS to the label order). The D1HF M batch
     checkpoint verdicts (build.py check: ok / finished < 90 %, any single gate failure) are reported, not a STOP.
  'fail' (SCF not converged after the one SlowConv retry, ORCA error) is reported separately (run_fail), not a gate.
  Phase S: results/S_report.json all_pass is false only through S-1 flagging xtb ref/scan_*/scan.inp files (no '!'
  line, no banned word) as ORCA inputs. prior() lets exactly that through (recorded as prior_gates.S_report.open: NOT
  accepted in writing by the user) and STOPs on any other S failure (S-2..S-5 not passed, or another S-1 problem).
  F and H are independent: H does not wait for F, so an F STOP does not hold H (scancel H to hold it).

sbatch plan (from analysis/d1_g2 of this worktree; logs in $G2/logs). MaxSubmit 20 counts array elements; the plan
adds W + 4 tasks (status 1 + F 1 + H inputs 1 + H workers W + H build/report 1). Right before submitting (login node,
one-shot commands): N = `squeue -u $USER -h | wc -l` (every queued task, incl. any test job), W = min(7, 20 - N - 4)
(W >= 1, else wait until tasks drain; the same $W goes into h_inputs, --array and h_batch), P = the ':'-joined G pool
job ids still in squeue (none left: drop --dependency from JS), and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`. Every
dependency uses ${X:?}, so a refused sbatch stops the lines after it instead of passing an empty id. Running at most
W + 1 at once (F + W workers).
  L=/gpfs/tmp_cpu2/yeseo1ee/d1_g2/logs; W=7; P=1033504:1033505:1033506:1033507:1033508:1033509:1033510:1033511:1033512
  # 0. Phase G gate: g2_check.py status after every pool job ended (exit 0 = all targets finished, no STOP)
  JS=$(sbatch --parsable -p cpu1,cpu2 -J d1g2_gst --time=48:00:00 --cpus-per-task=1 --mem=4G --export=ALL,G2ENV=d1 \
       --dependency=afterany:${P:?} jobs/g2.slurm g2_check.py status); echo "JS=$JS"
  # 1. F: one 48-core job (D1PAIR F: 8 slices x 32 cores, 2-4 min each; Q: 4 slices x 48 cores, 96G) ~ 1 h
  JF=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J d1g2_F --time=48:00:00 --cpus-per-task=48 --mem=96G \
       --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JS:?} \
       --kill-on-invalid-dep=yes jobs/g.slurm ../d1_g2/phase_fh.py f_all); echo "JF=$JF"
  # 2. H inputs + batches (single process, ~10-20 min); H does not wait for F (independent results)
  JHI=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J d1g2_Hin --time=48:00:00 --cpus-per-task=2 --mem=8G \
       --output=$L/%x_%j.out --dependency=afterok:${JS:?} --kill-on-invalid-dep=yes \
       jobs/hf.slurm ../d1_g2/phase_fh.py h_inputs ${W:?}); echo "JHI=$JHI"
  # 3. H workers: W x 48 serial ORCA (D1HF Md: 916 rxns / 48 cores / 3.4 h, MaxRSS 13 GB; %maxcore 3500 x 48 = 168 GB)
  #    ~ 8,600 x 0.15 core-h / (48 W) cores ~ 4 h for W = 7
  JHB=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J d1g2_Hb --time=48:00:00 --cpus-per-task=48 --mem=180G \
       --array=0-$((W-1)) --output=$L/%x_%A_%a.out --dependency=afterok:${JHI:?} --kill-on-invalid-dep=yes \
       jobs/hf.slurm ../d1_g2/phase_fh.py h_batch A ${W:?}); echo "JHB=$JHB"
  # 4. H labels + report after every worker ended (any state)
  JHR=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J d1g2_Hrep --time=48:00:00 --cpus-per-task=2 --mem=16G \
       --output=$L/%x_%j.out --dependency=afterok:${JHI:?},afterany:${JHB:?} --kill-on-invalid-dep=yes \
       jobs/hf.slurm ../d1_g2/phase_fh.py h_build ::: ../d1_g2/phase_fh.py h_report); echo "JHR=$JHR"
Resubmission (48 h wall / node failure; every step is idempotent, writes are atomic):
  * JS exit 2 (a pool job ended before every target finished): JF / JHI / JHB / JHR are cancelled by
    --kill-on-invalid-dep. Resubmit the unfinished G pool job(s) with their original command, then steps 0-4 with JS
    afterany on the new pool ids. JS exit 3 = Phase G STOP: report, nothing else.
  * JF died: the same JF command without --dependency (finished slices / merge skipped, atoms cached per reaction,
    gate / tokens / report rerun deterministically). JF exit 3 = F STOP (results/F_report.json): report.
  * JHI died: the JHI command without --dependency (same W), then JHB and JHR (both were cancelled).
  * worker k died or exited 2 (run.py: reaction killed / error, or not started after D1HF_STOP_AFTER_H = 45 h):
      (cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J d1g2_Hb$k --time=48:00:00 --cpus-per-task=48 --mem=180G \
       --output=$L/%x_%j.out jobs/hf.slurm ../d1_g2/phase_fh.py h_batch $k $W)      (the plan's W)
    then JHR again with --dependency=afterany:<that job>. .done reactions are skipped, calculations with a good .out.gz
    are kept, .fail reactions are not retried (D1HF rule: one SlowConv retry inside run.py only).
    Worker exit 3 = spec 14 H STOP (cumulative gate failure rate > 1 %, H_steps/h_batch_w<k>of<W>.json 'h_gate') or
    a hard error (fail_stage): report, do not resubmit.
  * Re-planning (another W / new inputs): move results/H_batches away by hand, then h_inputs moves results/M_checks
    aside with it (M_checks.prev_<time>); a check file older than plan.json is never read as the current plan's.
  * JHR can be rerun at any time without a dependency (rebuild + report only).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                           # no __pycache__ in analysis/d1_g2, d1_pair, d1_hf

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402
import test_t as TT  # noqa: E402                      (run / now / git_head / clean / env_f / env_hf / Fail, P_DIR, H_DIR)

DP = G.DP
P_DIR, H_DIR = TT.P_DIR, TT.H_DIR
INPUTS_G2 = dict(labels=G.G2 / "G" / "labels_d1_g2.json", meta=G.G2 / "G" / "input_meta_d1_g2.csv", prof=G.G2 / "prof")
VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-g2", FU2_TOKEN_VERSION="fu2-tokens-3-g2")      # spec 8
# ---------------------------------------------------------------- F
F_DIR = G.G2 / "F"
F_OUT = F_DIR / "out"
F_SLICES = F_OUT / "xtb" / "f_slices"                   # d1_features runroot: <out>/xtb/f_slices/slice<i>of<n>_*
F_IDS = F_DIR / "ids_all.csv"
F_MREP = F_DIR / "mergeroot_report.json"
X182_G2 = G.G2 / "X182_d1_g2.parquet"
FU2_SCR, FU2_RES = G.G2 / "fu2", G.RES / "fu2"
N_SLICES = 4                                            # features and atoms slices of f_all (each slice = all cores)
SLICE_FILES = ("g1.parquet", "ext.parquet", "root.json", "versions.json")
F_RATE_MAX = 0.01                                       # spec 14 F
F_STEPS = G.RES / "F_steps"
RETRY_REC = F_STEPS / "f_atoms_retry.json"
ATOMS_ENV_KEYS = ("D1PAIR_PHASE", "G1_ROOT", "ESPLEY_LABELS", "ESPLEY_META", "ESPLEY_PROF", "ESPLEY_OUT", "R5_SCRATCH",
                  "FU2_SCR", "FU2_RES", "FU2_X182", "FU2_ATOM_VERSION", "FU2_TOKEN_VERSION")
Q_KEYS = ("n_target", "n_ok", "n_compute_fail", "n_qa_not_reproduced", "n_qa_fail", "qa_max_fail", "n_qb_fail",
          "n_excluded_ext_fail", "ext_pass_rate", "ext_used", "n_use", "fail_reasons", "qa_maxdiff_max",
          "qb_maxdiff_max", "atom_version", "stop", "stop_reason", "manifest_sha256")
T_KEYS = ("token_version", "n_reactions", "n_excluded_from_Q", "num_cols", "ext_used", "gate_Ta", "stop",
          "manifest_sha256", "q_report_sha256", "x182_sha256")
# ---------------------------------------------------------------- H
HF_KEYS = ("D1HF_SCRATCH", "D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT")
H_RUNS = G.G2 / "runs" / "g2"                           # = hf_common RUNS / g2 with D1HF_SCRATCH = $G2
H_LIST_ALL = G.G2 / "H" / "targets_job.csv"
H_BATCHES = G.RES / "H_batches"
H_PLAN = H_BATCHES / "plan.json"
M_CHECKS = G.RES / "M_checks"                           # = build.py check output with D1HF_RES = analysis/d1_g2/results
H_STEPS = G.RES / "H_steps"
H_STATUS = G.RES / "H_status.csv"
H_LABELS_JSON = G.G2 / "labels_d1_hf_g2.json"
H_PARQUET = G.RES / "labels_d1_hf_g2.parquet"
BATCH_MAX = 1000                                        # D1HF M lists (<= 1,000 per checkpoint)
W_DEFAULT = 7
H_RATE_MAX = 0.01                                       # spec 14 H
H_FIN = ("ok", "fail", "gate_fail", "eda_table_unexplained", "parse_fail")
H_GATE = ("gate_fail", "eda_table_unexplained", "parse_fail")
H_OPEN = ("not_done", "incomplete", "not_started")
D1HF_COREH_PILOT = 0.106                                # spec 12: D1HF pilot median core-h per reaction (5 calculations)
# Phase S: the one S-1 problem form let through (s_smoke.audit_inputs on an xtb detailed input: no '!' line at all, no
# banned word reported); s_smoke keeps at most 30 problems, so a full list (>= 30) cannot be verified -> STOP
S1_XTB_SCAN = re.compile(r"^T\d+/(?:[^/]+/)*scan\.inp: '!' lines ''$")
S1_LIST_MAX = 30

F_DEF = ("n_g2_ok = targets with .done in $G2/g2_final; gate_fail = G2-ok reactions without a token npz that passed "
         "every existing gate = (a) features: F_ok false in d1_features mergeroot (xtb_status / ext_status not ok incl. "
         "the B1 engine gate, or a non-finite value among the 182) + (b) Q: X182_d1_g2 rows with Q_manifest use = false "
         "(compute failure, Q-a not reproduced, Q-b, extension self-check when ext_used) + (c) T: T_report bad (T-a); "
         "gate_fail_rate = gate_fail / n_g2_ok; STOP (spec 14 F) if > 0.01, or Q_report / T_report / mergeroot stop, or "
         "a step exit 3, or inconsistent counts")
H_DEF = ("finished = H_status ok + fail + gate_fail + eda_table_unexplained + parse_fail; gate_fail = gate_fail + "
         "eda_table_unexplained + parse_fail (build.py gates g1_asm, g2_sum, g3_seam, g_enuc, g_bond_ref, g_scf, "
         "g_setting, orca, EDA table rows, parser); gate_fail_rate = gate_fail / finished; STOP (spec 14 H) if > 0.01, "
         "or prep.py geominputs status not ok for > 1 % of the G2-ok targets (decision: the same 1 % rule on the D1HF "
         "input check); the D1HF M batch checkpoint verdicts (build.py check) are reported only (spec 14 H replaces "
         "them); 'fail' (SCF / ORCA run failure) is reported as run_fail, not a gate")


class Stop(Exception):
    """(stage, message, exit code): 2 = not ready / incomplete, 3 = STOP (spec 14) or a hard error."""

    def __init__(self, stage, msg, rc=3):
        super().__init__(stage, msg)
        self.stage, self.msg, self.rc = stage, msg, rc


def child(rec, name, argv, cwd, env=None, ok=(0,)):
    """one child interpreter (test_t.run: this python, output to the job log); exit code recorded; an exit outside
    ok raises Stop (rc 2 for a child exit 2, else 3)."""
    rc = TT.run(argv, cwd, env)
    rec.setdefault("exit", {})[name] = rc
    if rc not in ok:
        raise Stop(name, f"{name}: exit {rc}", 2 if rc == 2 else 3)
    return rc


def step(group, name, body):
    """run body(rec) -> exit code (0 / 2 / 3); results/<group>_steps/<name>.json; returns the exit code."""
    out = G.RES / f"{group}_steps" / f"{name}.json"
    rec = dict(step=name, group=group, argv=sys.argv[1:], started=TT.now(), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
               cpus=os.environ.get("SLURM_CPUS_PER_TASK"), git=TT.git_head(), python=sys.executable)
    t0 = time.time()
    try:
        rc = int(body(rec) or 0)
    except Stop as e:
        rec.update(fail_stage=e.stage, fail=str(e.msg)[:4000])
        rc = e.rc
    except TT.Fail as e:                                   # test_t.env_f / env_hf
        rec.update(fail_stage=str(e.args[0]), fail=str(e.args[1] if len(e.args) > 1 else "")[:4000])
        rc = 3
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed step, recorded
        rec.update(fail_stage="error", fail=f"{type(e).__name__}: {e}"[:4000], traceback=traceback.format_exc()[-4000:])
        rc = 3
    rec.update(rc=rc, finished=TT.now(), wall_s=round(time.time() - t0, 1))
    G.write_json(out, TT.clean(rec))
    short = {k: v for k, v in rec.items() if k not in ("traceback", "input_fail")}
    print(json.dumps(TT.clean(short), default=str)[:8000], flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[{name}] exit {rc} -> {out}", flush=True)
    return rc


# ================================================================ shared checks
_G2 = {}                                                # g2_ready() result, once per process


def targets_df():
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    X["rxn_id"] = X.rxn_id.astype(int)
    return X


def g2_ready(rec):
    """Phase G finished and not stopped (results/G_report.json of g2_check.py status + a live scan of g2_final)
    -> {X, ok (set of rxn_id), fail {rxn_id: marker}, summary}; Stop 2 (not ready) / 3 (G STOP) otherwise."""
    if _G2:
        rec["g2"] = _G2["summary"]
        return _G2
    rf, sf = G.RES / "G_report.json", G.RES / "G_status.csv"
    if not (rf.is_file() and sf.is_file()):
        raise Stop("g2", "results/G_report.json / G_status.csv missing (g2_check.py status after the pool jobs)", 2)
    rep = json.loads(rf.read_text())
    if rep.get("stop"):
        raise Stop("g2", f"Phase G STOP (spec 14 G): {rep.get('stop')}", 3)
    if not rep.get("all_finished"):
        raise Stop("g2", f"Phase G not finished ({rep.get('n_finished')} / {rep.get('n_targets')})", 2)
    if G.sha256(sf) != rep.get("status_csv_sha256"):
        raise Stop("g2", "results/G_status.csv changed after G_report.json (rerun g2_check.py status)", 2)
    X = targets_df()
    ok, fail, open_ = set(), {}, []
    for r in X.rxn_id:
        d = G.G2_FINAL / str(r)
        if (d / ".done").is_file():
            ok.add(int(r))
            continue
        m = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
        if m:
            fail[int(r)] = m[0]
        else:
            open_.append(int(r))
    if open_:
        raise Stop("g2", f"{len(open_)} targets without .done / .fail_* in {G.G2_FINAL} (first {open_[:10]})", 2)
    S = pd.read_csv(sf)
    n_ok_csv = int((S.status == "ok").sum())
    if n_ok_csv != len(ok) or len(S) != len(X):
        raise Stop("g2", f"G_status.csv ({len(S)} rows, {n_ok_csv} ok) != live scan ({len(X)} targets, {len(ok)} .done):"
                         f" rerun g2_check.py status", 2)
    summary = dict(n_targets=len(X), n_ok=len(ok), n_fail=len(fail), G_report_sha256=G.sha256(rf),
                   G_status_sha256=rep.get("status_csv_sha256"), G_overall_stop=rep.get("overall_stop"),
                   fail_markers=dict(Counter(fail.values()).most_common(30)))
    _G2.update(X=X, ok=ok, fail=fail, summary=summary)
    rec["g2"] = summary
    return _G2


def s_open(s):
    """Phase S (spec 5 / 14 S) -> (None = passed | note of the one let-through form, [STOP reasons]). Let through:
    S-1 failed only because s_smoke flagged xtb ref/scan_*/scan.inp files (no '!' line, no banned word) as ORCA inputs,
    with S-2..S-5 passed. That acceptance is NOT recorded from the user: the note says so."""
    ch = s.get("checks") or {}
    bad = [f"{k} pass_ = {(ch.get(k) or {}).get('pass_')!r}" for k in ("S-2", "S-3", "S-4", "S-5")
           if (ch.get(k) or {}).get("pass_") is not True]
    s1 = ch.get("S-1") or {}
    if s1.get("pass_") is True:
        return None, bad
    probs = [str(p) for p in s1.get("problems") or []]
    other = [p for p in probs if not S1_XTB_SCAN.match(p)]
    if not s1 or not probs or other or len(probs) >= S1_LIST_MAX or not s1.get("n_inputs"):
        bad.append(f"S-1 failed beyond the xtb scan.inp form: {len(probs)} problems, other {other[:5]}")
        return None, bad
    return (f"S-1 false only through {len(probs)} xtb scan.inp files flagged as ORCA inputs ({probs[:3]}); Phase S "
            f"STOP not accepted in writing by the user (no decision recorded)"), bad


def prior(rec, key):
    """the Phase T gate (spec 4; test_t.py report [F]) must have passed: results/T_report.json <key> is true; Phase S
    (results/S_report.json) must have passed except the s_open() form, which is recorded (prior_gates.S_report.open)."""
    f, fs = G.RES / "T_report.json", G.RES / "S_report.json"
    t = json.loads(f.read_text()) if f.is_file() else {}
    s = json.loads(fs.read_text()) if fs.is_file() else {}
    note, s_bad = s_open(s)
    if not fs.is_file():
        s_bad = ["results/S_report.json missing"] + s_bad
    rec["prior_gates"] = dict(T_report=dict(all_pass=t.get("all_pass"), F_ready=t.get("F_ready"), gate=t.get("gate"),
                                            written=t.get("written"), sha256=G.sha256(f) if f.is_file() else None),
                              S_report=dict(all_pass=s.get("all_pass"), sha256=G.sha256(fs) if fs.is_file() else None,
                                            checks={k: v.get("pass_") for k, v in (s.get("checks") or {}).items()},
                                            open=note, stop=s_bad))
    if t.get(key) is not True:
        raise Stop("phase_T", f"results/T_report.json {key} = {t.get(key)!r} (spec 4: Phase T must pass first)", 3)
    if s.get("all_pass") is not True and (s_bad or note is None):
        raise Stop("phase_S", f"results/S_report.json all_pass = {s.get('all_pass')!r}: {s_bad} (spec 5 / 14 S)", 3)


def env_F(rec):
    TT.env_f(rec)
    prior(rec, "F_ready")
    return g2_ready(rec)


def env_H(rec):
    TT.env_hf(rec)
    prior(rec, "all_pass")
    return g2_ready(rec)


def worst(*rc):
    return max(int(r) for r in rc)


# ================================================================ F (spec 8)
def f_ids(g):
    """$G2/F/ids_all.csv = every X_targets rxn_id (runroot records its sha256; a different existing file = STOP)."""
    txt = "rxn_id\n" + "".join(f"{r}\n" for r in sorted(int(x) for x in g["X"].rxn_id))
    if F_IDS.is_file():
        if F_IDS.read_text() != txt:
            raise Stop("ids", f"{F_IDS} differs from the X_targets rxn_ids (use a fresh $G2/F)", 3)
    else:
        G.write_atomic(F_IDS, txt)
    return F_IDS


def f_inputs_args():
    return [a for k in ("labels", "meta", "prof") for a in (f"--{k}", INPUTS_G2[k])]


def cmd_f_slice(i, n):
    def body(rec):
        g = env_F(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        other = sorted({p.name.split("of")[1][:2] for p in F_SLICES.glob("slice*_root.json")} - {f"{n:02d}"})
        if other:
            raise Stop("slices", f"{F_SLICES} has slices of n = {other}: mergeroot needs all slices of one n", 3)
        tag = f"slice{i:02d}of{n:02d}"
        have = [f for f in SLICE_FILES if (F_SLICES / f"{tag}_{f}").is_file()]
        rec["slice"] = dict(tag=tag, dir=str(F_SLICES), present=have)
        if len(have) == len(SLICE_FILES):
            rec["skipped"] = "slice outputs exist"
            return 0
        ids = f_ids(g)
        child(rec, "runroot", [P_DIR / "d1_features.py", "runroot", G.G2_FINAL, F_OUT, ids, i, n] + f_inputs_args(),
              P_DIR)
        return 0
    return step("F", f"f_slice_{i:02d}of{n:02d}", body)


def merged():
    """the mergeroot report if it belongs to the current X182_d1_g2.parquet (sha256) and ids_all.csv, else None."""
    if not (F_MREP.is_file() and X182_G2.is_file()):
        return None
    m = json.loads(F_MREP.read_text())
    if (m.get("x182") or {}).get("sha256") != G.sha256(X182_G2) or m.get("ids_csv") != str(F_IDS):
        return None
    return m


def need_merged(rec):
    m = merged()
    if m is None:
        raise Stop("f_merge", "no mergeroot report for the current X182_d1_g2.parquet (run f_merge)", 2)
    if m.get("stop"):
        raise Stop("f_merge", "mergeroot report stop (columns != X182_d1 or non-finite values)", 3)
    rec["x182"] = dict(path=str(X182_G2), sha256=m["x182"]["sha256"], n=m["x182"]["n"])
    return m


def cmd_f_merge():
    def body(rec):
        g = env_F(rec)
        m = merged()
        if m is None:
            child(rec, "mergeroot", [P_DIR / "d1_features.py", "mergeroot", F_OUT, X182_G2, F_MREP], P_DIR)
            m = merged()
            if m is None:
                raise Stop("mergeroot", f"{F_MREP} does not match {X182_G2} after mergeroot", 3)
        else:
            rec["skipped"] = "mergeroot report matches X182_d1_g2.parquet"
        rec["mergeroot"] = {k: m.get(k) for k in ("n_slices", "n_requested", "n_done", "n_feat", "F_fail_rate", "x182",
                                                  "F1_all_finite", "stop")}
        if m.get("n_done") != len(g["ok"]) or m.get("n_requested") != len(g["X"]):
            raise Stop("mergeroot", f"n_done {m.get('n_done')} / n_requested {m.get('n_requested')} != G2 .done "
                                    f"{len(g['ok'])} / targets {len(g['X'])}", 3)
        if m.get("stop"):
            raise Stop("mergeroot", "mergeroot report stop (columns != X182_d1 or non-finite values)", 3)
        return 0
    return step("F", "f_merge", body)


def atoms_env():
    """= s_smoke.py f (FU2 child environment) with the full-run roots."""
    return dict(os.environ, D1PAIR_PHASE="FROOT", G1_ROOT=str(G.G2_FINAL), ESPLEY_OUT=str(F_DIR / "fu2_xtb"),
                R5_SCRATCH=str(F_DIR / "fu2_r5"), ESPLEY_LABELS=str(INPUTS_G2["labels"]),
                ESPLEY_META=str(INPUTS_G2["meta"]), ESPLEY_PROF=str(INPUTS_G2["prof"]), FU2_SCR=str(FU2_SCR),
                FU2_RES=str(FU2_RES), FU2_X182=str(X182_G2), PYTHONDONTWRITEBYTECODE="1", **VERSIONS)


def atoms_counts():
    a = FU2_SCR / "atoms"
    return dict(npz=len(list(a.glob("*.npz"))), fail=len(list(a.glob("*.fail.json"))), err=len(list(a.glob("*.err.json"))))


def cmd_f_atoms(i, n):
    def body(rec):
        env_F(rec)
        need_merged(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        env = atoms_env()
        rec["child_env"] = {k: env[k] for k in ATOMS_ENV_KEYS}
        child(rec, "atoms_run", [P_DIR / "fu2" / "atoms.py", "root", "run", i, n], P_DIR, env)
        rec["atoms"] = atoms_counts()
        return 0
    return step("F", f"f_atoms_{i:02d}of{n:02d}", body)


def step_rec(path):
    """a step json (results/<group>_steps/<step>.json) or {} when missing / unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cmd_f_gate():
    err = sorted((FU2_SCR / "atoms").glob("*.err.json"))
    if err and step_rec(RETRY_REC).get("rc") != 0:         # no retry pass exited 0 yet (a failed / not-ready one reruns)
        def retry(rec):
            env_F(rec)
            need_merged(rec)
            rec["err_before"] = [p.name for p in err][:500]
            rec["n_err_before"] = len(err)
            child(rec, "atoms_retry", [P_DIR / "fu2" / "atoms.py", "root", "run", 0, 1], P_DIR, atoms_env())
            rec["atoms"] = atoms_counts()
            return 0
        rc = step("F", "f_atoms_retry", retry)
        if rc:
            return rc

    def body(rec):
        env_F(rec)
        need_merged(rec)
        rec["atoms"] = atoms_counts()
        rc = child(rec, "atoms_gate", [P_DIR / "fu2" / "atoms.py", "root", "gate"], P_DIR, atoms_env(), ok=(0, 2, 3))
        q = FU2_RES / "Q_report.json"
        if rc == 2:
            rec["note"] = "atoms incomplete (atoms.py gate exit 2): run f_atoms for every slice"
        elif q.is_file():
            r = json.loads(q.read_text())
            rec["Q_report"] = {k: r.get(k) for k in Q_KEYS}
        return rc
    return step("F", "f_gate", body)


def cmd_f_tokens():
    def body(rec):
        env_F(rec)
        need_merged(rec)
        q = FU2_RES / "Q_report.json"
        if not q.is_file():
            raise Stop("Q", f"no {q} (run f_gate)", 2)
        r = json.loads(q.read_text())
        if r.get("stop"):
            raise Stop("Q", f"Q gate STOP: {r.get('stop_reason')}", 3)
        rc = child(rec, "tokens_build", [P_DIR / "fu2" / "atoms.py", "root", "tokens"], P_DIR, atoms_env(), ok=(0, 3))
        t = FU2_RES / "T_report.json"
        if t.is_file():
            tr = json.loads(t.read_text())
            rec["T_report"] = {k: tr.get(k) for k in T_KEYS}
        return rc
    return step("F", "f_tokens", body)


def cmd_f_report():
    def body(rec):
        g = env_F(rec)
        ok_ids = g["ok"]
        n_g2 = len(ok_ids)
        rep = dict(spec="docs/specs/D1G2.md 8 (STOP: 14 F)", written=TT.now(), gate_rule=F_DEF, threshold=F_RATE_MAX,
                   n_targets=len(g["X"]), n_g2_ok=n_g2, n_g2_not_ok=len(g["fail"]),
                   g2_not_ok_markers=dict(Counter(g["fail"].values()).most_common()),
                   inputs={k: str(v) for k, v in INPUTS_G2.items()}, versions=VERSIONS,
                   child_env_atoms={k: atoms_env()[k] for k in ATOMS_ENV_KEYS})
        stop, missing = [], []
        steps = {p.stem: json.loads(p.read_text()) for p in sorted(F_STEPS.glob("*.json")) if p.stem != "f_report"}
        rep["steps"] = {k: dict(rc=v.get("rc"), finished=v.get("finished"), slurm_job=v.get("slurm_job"),
                                exit=v.get("exit"), fail=v.get("fail"), skipped=v.get("skipped")) for k, v in steps.items()}
        gate_rec = steps.get("f_gate") or {}
        superseded = [k for k, v in steps.items() if k == "f_atoms_retry" and v.get("rc") != 0
                      and gate_rec.get("rc") == 0 and str(gate_rec.get("started") or "") >= str(v.get("finished") or "~")]
        rep["superseded_steps"] = {k: f"exit {steps[k].get('rc')} at {steps[k].get('finished')}, followed by f_gate exit 0 "
                                      f"started {gate_rec.get('started')}" for k in superseded}
        rep["prior_gates"] = rec.get("prior_gates")
        stop += [f"step {k} exit 3: {v.get('fail_stage')}: {v.get('fail')}" for k, v in steps.items()
                 if v.get("rc") == 3 and k not in superseded]
        # (a) features
        feat_fail, x_ids = None, None
        m = merged()
        if m is None:
            missing.append(f"{F_MREP} for the current {X182_G2} (f_merge)")
        else:
            not_ok = {int(r): w for r, w in (m.get("not_ok") or {}).items()}
            root_miss = {r: w for r, w in not_ok.items() if str(w).startswith("root:")}
            feat_fail = {r: w for r, w in not_ok.items() if not str(w).startswith("root:")}
            X1 = pd.read_parquet(X182_G2)
            X0_cols = list(pd.read_parquet(DP.X182_D1).columns)
            x_ids = set(int(r) for r in X1.rxn_id)
            cols_equal = list(X1.columns) == X0_cols and len(X0_cols) == 183
            rep["features"] = dict(x182=dict(path=str(X182_G2), sha256=m["x182"]["sha256"], n=len(X1),
                                             columns_equal_X182_d1=cols_equal, X182_d1=str(DP.X182_D1)),
                                   mergeroot_report=dict(path=str(F_MREP), sha256=G.sha256(F_MREP)),
                                   n_requested=m.get("n_requested"), n_done=m.get("n_done"), n_feat=m.get("n_feat"),
                                   n_slices=m.get("n_slices"), F1_all_finite=m.get("F1_all_finite"),
                                   n_feature_fail=len(feat_fail), feature_fail_rate=len(feat_fail) / n_g2 if n_g2 else None,
                                   feature_fail_reasons=dict(Counter(feat_fail.values()).most_common(60)),
                                   feature_fail_rxn=dict(sorted(feat_fail.items())),
                                   xtb_status=m.get("xtb_status"), ext_status_n=len(m.get("ext_status") or {}),
                                   versions=m.get("versions"), si_decision=m.get("si_decision"))
            if m.get("stop") or not cols_equal or not m.get("F1_all_finite"):
                stop.append(f"mergeroot: stop {m.get('stop')}, columns_equal_X182_d1 {cols_equal}, "
                            f"F1_all_finite {m.get('F1_all_finite')}")
            if (m.get("n_done") != n_g2 or set(root_miss) != set(g["fail"]) or x_ids != ok_ids - set(feat_fail)):
                stop.append(f"inconsistent feature counts: n_done {m.get('n_done')} vs G2 ok {n_g2}, root: {len(root_miss)}"
                            f" vs G2 not ok {len(g['fail'])}, X182 rows {len(x_ids)} vs {n_g2 - len(feat_fail)}")
        # (b) Q
        q_excl = None
        qf, qm = FU2_RES / "Q_report.json", FU2_RES / "Q_manifest.csv"
        if not (qf.is_file() and qm.is_file()):
            missing.append(f"{qf} / {qm} (f_gate)")
        else:
            q = json.loads(qf.read_text())
            Q = pd.read_csv(qm)
            q_excl = sorted(int(r) for r in Q.loc[Q.use.astype(str) != "True", "rxn_id"])     # bool column as text
            ov = q.get("overrides") or {}
            q_cf = int(((Q.use.astype(str) != "True") & (Q.status.astype(str) != "ok")).sum())   # no atom values
            rep["Q"] = dict({k: q.get(k) for k in Q_KEYS}, n_excluded=len(q_excl), excluded=q_excl,
                            n_excluded_compute_fail=q_cf, n_excluded_gate_fail=len(q_excl) - q_cf,
                            report_sha256=G.sha256(qf), manifest_sha256_file=G.sha256(qm), overrides=ov)
            if q.get("stop"):
                stop.append(f"Q gate STOP: {q.get('stop_reason')}")
            want = dict(FU2_X182=str(X182_G2), FU2_ATOM_VERSION=VERSIONS["FU2_ATOM_VERSION"], G1_ROOT=str(G.G2_FINAL))
            if any(ov.get(k) != v for k, v in want.items()) or q.get("manifest_sha256") != G.sha256(qm):
                stop.append(f"Q_report is not from this run: overrides {ov}, manifest sha256 {q.get('manifest_sha256')}")
            if x_ids is not None and set(int(r) for r in Q.rxn_id) != x_ids:
                stop.append(f"Q_manifest rows ({len(Q)}) != X182_d1_g2 rows ({len(x_ids)})")
        # (c) T
        tok_ok, t_bad = None, None
        tf, tm = FU2_RES / "T_report.json", FU2_RES / "T_manifest.csv"
        if not (tf.is_file() and tm.is_file()):
            missing.append(f"{tf} / {tm} (f_tokens)")
        else:
            t = json.loads(tf.read_text())
            T = pd.read_csv(tm)
            t_bad = sorted(int(b["rxn_id"]) for b in (t.get("bad") or []))
            no_file = [int(r) for r, s in zip(T.rxn_id, T.sha256)
                       if not (FU2_SCR / "tokens" / f"{int(r)}.npz").is_file()
                       or G.sha256(FU2_SCR / "tokens" / f"{int(r)}.npz") != s]
            tok_ok = set(int(r) for r in T.rxn_id) - set(t_bad) - set(no_file)
            G.write_atomic(G.RES / "F_tokens_manifest.csv", tm.read_bytes())
            rep["T"] = dict({k: t.get(k) for k in T_KEYS}, n_bad=len(t_bad), bad=t.get("bad"),
                            n_token_files_missing_or_changed=len(no_file), token_files_missing_or_changed=no_file[:100],
                            report_sha256=G.sha256(tf), manifest=dict(path=str(tm), sha256=G.sha256(tm)),
                            F_tokens_manifest=dict(path=str(G.RES / "F_tokens_manifest.csv"),
                                                   sha256=G.sha256(G.RES / "F_tokens_manifest.csv")))
            if t.get("stop"):
                stop.append(f"T gate STOP: gate_Ta {t.get('gate_Ta')}")
            if no_file:
                stop.append(f"{len(no_file)} token npz missing or not equal to T_manifest.csv")
            if t.get("token_version") != VERSIONS["FU2_TOKEN_VERSION"] or (
                    X182_G2.is_file() and t.get("x182_sha256") != G.sha256(X182_G2)) or (
                    qf.is_file() and t.get("q_report_sha256") != G.sha256(qf)):
                stop.append("T_report is not from this run (token version / X182 / Q_report sha256)")
        # gate failure rate
        rep["counts"] = dict(targets=len(g["X"]), g2_ok=n_g2, features_ok=len(x_ids) if x_ids is not None else None,
                             q_use=(len(x_ids) - len(q_excl)) if (x_ids is not None and q_excl is not None) else None,
                             tokens_ok=len(tok_ok) if tok_ok is not None else None)
        if not missing:
            gate_fail = n_g2 - len(tok_ok)
            parts = dict(features=len(feat_fail), Q_excluded=len(q_excl), T_bad=len(t_bad))
            rate = gate_fail / n_g2 if n_g2 else float("nan")
            rep["gate_failure"] = dict(n=gate_fail, n_g2_ok=n_g2, rate=rate, threshold=F_RATE_MAX, components=parts,
                                       component_rates={k: v / n_g2 for k, v in parts.items()} if n_g2 else None,
                                       stop=bool(not n_g2 or rate > F_RATE_MAX))
            if gate_fail != sum(parts.values()):
                stop.append(f"inconsistent gate counts: {gate_fail} != {parts}")
            if not n_g2 or rate > F_RATE_MAX:
                stop.append(f"F gate failure rate {gate_fail}/{n_g2} = {rate:.4f} > {F_RATE_MAX} (spec 14 F)")
        rep.update(missing=missing, complete=not missing, stop=bool(stop), stop_reasons=stop)
        G.write_json(G.RES / "F_report.json", TT.clean(rep))
        rec["report"] = {k: rep.get(k) for k in ("counts", "gate_failure", "complete", "stop", "stop_reasons", "missing")}
        return 3 if stop else (2 if missing else 0)
    return step("F", "f_report", body)


def cmd_f_all():
    n = N_SLICES
    seq = [(f"f_slice {i} {n}", (lambda i=i: cmd_f_slice(i, n))) for i in range(n)] + [("f_merge", cmd_f_merge)] + \
          [(f"f_atoms {i} {n}", (lambda i=i: cmd_f_atoms(i, n))) for i in range(n)] + \
          [("f_gate", cmd_f_gate), ("f_tokens", cmd_f_tokens)]
    rc = 0
    for name, fn in seq:
        r = fn()
        print(f"[f_all] {name}: exit {r}  ({TT.now()})", flush=True)
        if r:
            rc = r
            break
    r = cmd_f_report()
    print(f"[f_all] f_report: exit {r}", flush=True)
    return worst(rc, r)


# ================================================================ H (spec 9)
def env_h():
    """= s_smoke.py h child environment with the full-run roots."""
    return dict(os.environ, D1HF_SCRATCH=str(G.G2), D1HF_RES=str(G.RES), D1HF_TEMPLATES=str(TT.HF_SCR / "inputs"),
                D1HF_G2_ROOT=str(G.G2_FINAL), PYTHONDONTWRITEBYTECODE="1")


def load_plan():
    return json.loads(H_PLAN.read_text()) if H_PLAN.is_file() else None


def plan_problems(plan):
    bad = []
    gi = G.RES / "g2_inputs.csv"
    if not gi.is_file() or G.sha256(gi) != plan.get("g2_inputs_sha256"):
        bad.append("results/g2_inputs.csv differs from the plan")
    for name, sha in (plan.get("files") or {}).items():
        f = H_BATCHES / name
        if not f.is_file() or G.sha256(f) != sha:
            bad.append(f"{name} differs from the plan")
    return bad


def cmd_h_inputs(W):
    def body(rec):
        g = env_H(rec)
        if W < 1:
            raise Stop("args", f"W = {W}?", 3)
        plan = load_plan()
        if plan is not None:
            bad = plan_problems(plan)
            if plan.get("W") != W or bad:
                raise Stop("plan", f"{H_PLAN} exists for W = {plan.get('W')} ({bad}); W = {W} asked: move "
                                   f"results/H_batches away before planning again", 3)
            rec["skipped"] = "plan exists for this W, inputs unchanged"
            rec["plan"] = {k: plan.get(k) for k in ("W", "nb", "n_ok", "sizes", "max_size")}
            return 0
        X = g["X"]
        jobs = sorted(X.job_id)
        lst = G.write_atomic(H_LIST_ALL, "job_id\n" + "".join(f"{T}\n" for T in jobs))
        env = env_h()
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        child(rec, "geominputs", [H_DIR / "prep.py", "geominputs", "g2", lst], H_DIR, env)
        gi_f = G.RES / "g2_inputs.csv"
        gi = pd.read_csv(gi_f, dtype=str, keep_default_na=False)
        if sorted(gi.job_id) != jobs:
            raise Stop("geominputs", f"{gi_f}: {len(gi)} rows != the {len(jobs)} targets", 3)
        st = dict(zip(gi.job_id, gi.status))
        rid = dict(zip(X.job_id, X.rxn_id))
        g2ok = sorted(T for T in jobs if rid[T] in g["ok"])
        okj = sorted(T for T in jobs if st[T] == "ok")
        stray = sorted(set(okj) - set(g2ok))
        if stray:
            raise Stop("geominputs", f"status ok without a G2 .done: {stray[:10]}", 3)
        if not okj:
            raise Stop("geominputs", "no reaction with g2 inputs", 3)
        input_fail = {T: st[T] for T in g2ok if st[T] != "ok"}
        nb = W * math.ceil(len(okj) / (BATCH_MAX * W))
        D = X[X.job_id.isin(okj)][["job_id", "n_atoms"]].sort_values(["n_atoms", "job_id"]).reset_index(drop=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        if any(H_BATCHES.glob("H_*.csv")):                # files of an unfinished earlier planning: kept, moved aside
            prev = G.assert_path(H_BATCHES.with_name(f"H_batches.prev_{ts}"))
            os.replace(H_BATCHES, prev)
            rec["previous_batches_moved_to"] = str(prev)
        if M_CHECKS.is_dir() and any(M_CHECKS.glob("*.json")):     # checks of an earlier plan: kept, moved aside
            prev = G.assert_path(M_CHECKS.with_name(f"M_checks.prev_{ts}"))
            os.replace(M_CHECKS, prev)
            rec["previous_checks_moved_to"] = str(prev)
        files, sizes = {}, []
        for j in range(nb):
            b = D.iloc[j::nb].sort_values("job_id")
            name = f"H_b{j:02d}.csv"
            G.write_atomic(H_BATCHES / name, lambda t, b=b: b.to_csv(t, index=False))
            files[name] = G.sha256(H_BATCHES / name)
            sizes.append(len(b))
        plan = dict(W=W, nb=nb, n_ok=len(okj), n_g2_ok=len(g2ok), sizes=sizes, max_size=max(sizes), files=files,
                    assignment={str(k): [f"H_b{j:02d}.csv" for j in range(nb) if j % W == k] for k in range(W)},
                    g2_inputs_sha256=G.sha256(gi_f), g2_inputs_status=dict(Counter(gi.status).most_common()),
                    n_input_fail_g2_ok=len(input_fail), input_fail=input_fail, written=TT.now(),
                    rule=f"status-ok job_ids sorted by (n_atoms, job_id), dealt round robin into nb = W * ceil(n / "
                         f"({BATCH_MAX} W)) batches (D1HF prep.batches rule), each file sorted by job_id; worker k of W "
                         f"owns the batches j with j % W == k")
        G.write_json(H_PLAN, TT.clean(plan))                 # last: the plan exists only with all its files
        rec["plan"] = {k: plan[k] for k in ("W", "nb", "n_ok", "n_g2_ok", "sizes", "max_size", "g2_inputs_status",
                                            "n_input_fail_g2_ok")}
        rec["input_fail"] = input_fail
        return 0
    return step("H", "h_inputs", body)


def h_checks(plan):
    """build.py check records (results/M_checks/H_bNN.json) of the current plan's batches; a file older than plan.json
    is from an earlier plan and is not read -> (records, missing, stale, cumulative spec 14 H rate over them)."""
    t_plan = H_PLAN.stat().st_mtime
    recs, missing, stale, fin, gate = {}, [], [], 0, 0
    for name in sorted(plan["files"]):
        f = M_CHECKS / f"{Path(name).stem}.json"
        if not f.is_file():
            missing.append(name)
            continue
        if f.stat().st_mtime < t_plan:
            stale.append(name)
            continue
        c = json.loads(f.read_text())
        bs = {str(s): int(v) for s, v in (c.get("by_status") or {}).items()}
        nf, ng = sum(bs.get(s, 0) for s in H_FIN), sum(bs.get(s, 0) for s in H_GATE)
        recs[name] = dict(n=c.get("n"), by_status=bs, n_finished=nf, n_ok=c.get("n_ok"), n_gate_fail=ng,
                          d1hf_checkpoint_stop=c.get("stop"), sha256=G.sha256(f))
        fin, gate = fin + nf, gate + ng
    rate = gate / fin if fin else None
    cum = dict(n_checks=len(recs), n_finished=fin, n_gate_fail=gate, rate=rate, threshold=H_RATE_MAX,
               stop=bool(fin and rate > H_RATE_MAX))
    return recs, missing, stale, cum


def cmd_h_batch(k, W):
    def body(rec):
        env_H(rec)
        plan = load_plan()
        if plan is None:
            raise Stop("plan", f"no {H_PLAN} (run h_inputs)", 2)
        bad = plan_problems(plan)
        if plan.get("W") != W or bad or not 0 <= k < W:
            raise Stop("plan", f"worker {k} of {W} vs plan W = {plan.get('W')}; {bad}", 3)
        mine = plan["assignment"][str(k)]
        rec["batches"] = mine
        if not mine:
            rec["note"] = "no batch for this worker"
            return 0
        B = pd.concat([pd.read_csv(H_BATCHES / b, dtype={"job_id": str}) for b in mine], ignore_index=True)
        U = B.sort_values(["n_atoms", "job_id"], ascending=[False, True]).reset_index(drop=True)
        ulist = G.write_atomic(H_BATCHES / f"H_w{k}of{W}.csv", lambda t: U.to_csv(t, index=False))
        rec["n_reactions"] = len(U)
        env = env_h()
        rec["child_env"] = {k2: env[k2] for k2 in HF_KEYS}
        cum0 = h_checks(plan)[3]                           # spec 14 H: no new run after the 1 % rule already failed
        rec["h_gate_before"] = cum0
        if cum0["stop"]:
            raise Stop("h_gate", f"spec 14 H STOP: cumulative gate failure rate {cum0['n_gate_fail']}/"
                                 f"{cum0['n_finished']} > {H_RATE_MAX} over {cum0['n_checks']} batch checks: no new run", 3)
        nw = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        rc_run = child(rec, "run", [H_DIR / "run.py", "run", ulist, "g2", "hf", nw], H_DIR, env, ok=(0, 2, 3))
        checks = {}
        for b in mine:                                    # D1HF M checkpoint: exit 3 = its own rule, recorded only
            stem = Path(b).stem
            r = child(rec, f"check_{stem}", [H_DIR / "build.py", "check", H_BATCHES / b, "g2"], H_DIR, env, ok=(0, 3))
            c = json.loads((M_CHECKS / f"{stem}.json").read_text())
            checks[b] = dict(exit=r, n=c.get("n"), by_status=c.get("by_status"), n_finished=c.get("n_finished"),
                             n_ok=c.get("n_ok"), d1hf_checkpoint_stop=c.get("stop"))
        rec["checks"] = checks
        cum = h_checks(plan)[3]
        rec["h_gate_after"] = cum
        if cum["stop"]:
            rec["h_gate"] = (f"spec 14 H STOP: cumulative gate failure rate {cum['n_gate_fail']}/{cum['n_finished']} > "
                             f"{H_RATE_MAX} over {cum['n_checks']} batch checks")
            return 3
        return rc_run
    return step("H", f"h_batch_w{k}of{W}", body)


def cmd_h_build():
    def body(rec):
        env_H(rec)
        if load_plan() is None:
            raise Stop("plan", f"no {H_PLAN} (run h_inputs)", 2)
        env = env_h()
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        child(rec, "allgeom", [H_DIR / "build.py", "allgeom", "-", "g2"], H_DIR, env)
        rec["outputs"] = {str(p): G.sha256(p) for p in (H_LABELS_JSON, H_PARQUET, H_STATUS)}
        return 0
    return step("H", "h_build", body)


def coreh_by_job():
    """{job_id: core-h} = sum of timing.tsv wall_seconds (serial ORCA: wall = core time, = build.py coreh_total)."""
    out = {}
    if not H_RUNS.is_dir():
        return out
    for d in H_RUNS.iterdir():
        f = d / "timing.tsv"
        if not f.is_file():
            continue
        lines = f.read_text().splitlines()
        if not lines or "wall_seconds" not in lines[0].split("\t"):
            continue
        j, s = lines[0].split("\t").index("wall_seconds"), 0.0
        for ln in lines[1:]:
            t = ln.split("\t")
            try:
                s += float(t[j])
            except (IndexError, ValueError):
                pass
        out[d.name] = s / 3600
    return out


def qstats(v):
    s = pd.Series(list(v), dtype=float)
    return dict(n=int(s.notna().sum()), sum=float(s.sum()), median=float(s.median()) if len(s) else None,
                mean=float(s.mean()) if len(s) else None, p90=float(s.quantile(0.9)) if len(s) else None,
                max=float(s.max()) if len(s) else None)


def cmd_h_report():
    def body(rec):
        g = env_H(rec)
        plan = load_plan()
        bf = H_STEPS / "h_build.json"
        b = json.loads(bf.read_text()) if bf.is_file() else {}
        if plan is None or b.get("rc") != 0 or not H_STATUS.is_file() or \
                (b.get("outputs") or {}).get(str(H_STATUS)) != G.sha256(H_STATUS):
            raise Stop("h_build", "run h_build first (no plan, h_build not ok, or H_status.csv changed since)", 2)
        X = g["X"]
        rid = dict(zip(X.job_id, X.rxn_id))
        g2ok = sorted(T for T in X.job_id if rid[T] in g["ok"])
        gi = pd.read_csv(G.RES / "g2_inputs.csv", dtype=str, keep_default_na=False)
        gst = dict(zip(gi.job_id, gi.status))
        S = pd.read_csv(H_STATUS, dtype=str, keep_default_na=False)
        lab = json.loads(H_LABELS_JSON.read_text())
        by = S.status.value_counts().to_dict()
        fin = S[S.status.isin(H_FIN)]
        gate = S[S.status.isin(H_GATE)]
        inok = {T for T in g2ok if gst.get(T) == "ok"}
        open_ = S[S.status.isin(H_OPEN) & S.job_id.isin(inok)]
        input_fail = {T: gst.get(T) for T in g2ok if gst.get(T) != "ok"}
        rate = len(gate) / len(fin) if len(fin) else float("nan")
        in_rate = len(input_fail) / len(g2ok) if g2ok else float("nan")
        checks, ck_missing, ck_stale, ck_cum = h_checks(plan)
        ck_d1hf = [f"{n}: {c['d1hf_checkpoint_stop']}" for n, c in checks.items() if c.get("d1hf_checkpoint_stop")]
        ck_missing = ck_missing + ck_stale                 # a check older than plan.json = not done for this plan
        workers = {}
        for k in range(int(plan["W"])):
            f = H_STEPS / f"h_batch_w{k}of{plan['W']}.json"
            w = json.loads(f.read_text()) if f.is_file() else {}
            workers[str(k)] = dict(rc=w.get("rc"), slurm_job=w.get("slurm_job"), host=w.get("host"),
                                   started=w.get("started"), finished=w.get("finished"), wall_s=w.get("wall_s"),
                                   exit=w.get("exit"), fail=w.get("fail"), batches=w.get("batches"),
                                   h_gate=w.get("h_gate"),
                                   before_plan=bool(w) and str(w.get("started") or "") < str(plan.get("written") or ""))
        ch = coreh_by_job()
        st_of = dict(zip(S.job_id, S.status))
        core = dict(total=float(sum(ch.values())), n_run_folders=len(ch),
                    by_status={s: qstats([v for T, v in ch.items() if st_of.get(T) == s])
                               for s in sorted({st_of.get(T, "?") for T in ch})},
                    ok_coreh_total=qstats(r.get("coreh_total") for r in lab),
                    reference=f"D1HF pilot median {D1HF_COREH_PILOT} core-h per reaction (spec 12)")
        stop = []
        if len(fin) and rate > H_RATE_MAX:
            stop.append(f"H gate failure rate {len(gate)}/{len(fin)} = {rate:.4f} > {H_RATE_MAX} (spec 14 H)")
        if g2ok and in_rate > H_RATE_MAX:
            stop.append(f"g2 input check failed for {len(input_fail)}/{len(g2ok)} = {in_rate:.4f} G2-ok targets > "
                        f"{H_RATE_MAX}")
        if len(lab) != by.get("ok", 0):
            stop.append(f"labels_d1_hf_g2.json rows {len(lab)} != H_status ok {by.get('ok', 0)}")
        complete = not len(open_) and not ck_missing and len(fin) > 0
        rep = dict(spec="docs/specs/D1G2.md 9 (STOP: 14 H)", written=TT.now(), gate_rule=H_DEF, threshold=H_RATE_MAX,
                   n_targets=len(X), n_g2_ok=len(g2ok), n_inputs_ok=len(inok), by_status=by, n_finished=len(fin),
                   n_ok=by.get("ok", 0),
                   gate_failure=dict(n=len(gate), n_finished=len(fin), rate=rate, by_status=gate.status.value_counts()
                                     .to_dict(), reasons=dict(Counter(gate.reason).most_common(40)),
                                     jobs=gate.job_id.tolist()[:300], stop=bool(len(fin) and rate > H_RATE_MAX)),
                   run_fail=dict(n=int((S.status == "fail").sum()),
                                 reasons=dict(Counter(S.reason[S.status == "fail"]).most_common(20))),
                   input_fail=dict(n=len(input_fail), rate=in_rate, by_status=dict(Counter(input_fail.values())),
                                   jobs=dict(list(input_fail.items())[:300])),
                   no_input_reasons=dict(Counter(S.reason[S.status == "no_input"]).most_common()),
                   open=dict(n=len(open_), by_status=open_.status.value_counts().to_dict(),
                             jobs=open_.job_id.tolist()[:300]),
                   checkpoints=dict(n_batches=len(plan["files"]), missing=ck_missing, stale=ck_stale, cumulative=ck_cum,
                                    d1hf_rule_stopped=ck_d1hf, d1hf_rule_note="build.py check verdict (D1HF M rule: "
                                    "ok < 90 % or any gate failure) reported only; D1G2 spec 14 H = the 1 % rule",
                                    batches=checks),
                   prior_gates=rec.get("prior_gates"),
                   workers=workers, plan={k: plan.get(k) for k in ("W", "nb", "n_ok", "sizes", "max_size")},
                   core_h=core,
                   outputs={str(p): (G.sha256(p) if p.is_file() else None) for p in (H_LABELS_JSON, H_PARQUET, H_STATUS)},
                   child_env={k: env_h()[k] for k in HF_KEYS},
                   complete=complete, stop=bool(stop), stop_reasons=stop)
        G.write_json(G.RES / "H_report.json", TT.clean(rep))
        rec["report"] = {k: rep.get(k) for k in ("n_g2_ok", "n_inputs_ok", "by_status", "n_finished", "complete", "stop",
                                                 "stop_reasons")}
        rec["report"]["gate_failure_rate"] = rate
        rec["report"]["core_h_total"] = core["total"]
        return 3 if stop else (0 if complete else 2)
    return step("H", "h_report", body)


# ================================================================ CLI
def array_k(s):
    if s.upper() in ("A", "@I"):
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            raise SystemExit("h_batch A: no SLURM_ARRAY_TASK_ID (submit with --array or give k)")
        return int(v)
    return int(s)


def main(a):
    cmd, n = (a[0] if a else ""), len(a)
    if cmd == "f_slice" and n == 3:
        return cmd_f_slice(int(a[1]), int(a[2]))
    if cmd == "f_atoms" and n == 3:
        return cmd_f_atoms(int(a[1]), int(a[2]))
    if cmd == "h_inputs" and n in (1, 2):
        return cmd_h_inputs(int(a[1]) if n == 2 else W_DEFAULT)
    if cmd == "h_batch" and n == 3:
        return cmd_h_batch(array_k(a[1]), int(a[2]))
    plain = {"f_merge": cmd_f_merge, "f_gate": cmd_f_gate, "f_tokens": cmd_f_tokens, "f_report": cmd_f_report,
             "f_all": cmd_f_all, "h_build": cmd_h_build, "h_report": cmd_h_report}
    if cmd in plain and n == 1:
        return plain[cmd]()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
