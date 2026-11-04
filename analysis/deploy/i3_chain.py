#!/usr/bin/env python3
r"""i3_chain.py — DEPLOY-MODEL 14-3 I-3 chain smoke (docs/specs/DEPLOY_MODEL.md 14-2, 14-3).

10 D1 reactions (5 dipole kinds x 2 = 5 P_main para reference pairs, g2_deploy.py i3-prep) from SMILES through the
deployment path: g2_deploy.py (autodE ts -> stereo -> ref -> finalize_lf) -> M (meta_lf) -> F (feat_lf) -> T (tokens_lf)
-> predict_tf.py run --model final. Every write stays under one root <R> (default $DEP/i3; dry run $DEP/i3_dev) and
analysis/deploy/results/I3_chain*.json; production $DEP outputs are only read. One command per step (sbatch chain).
Exit 0 done (subset exceptions recorded), 2 not ready / unfinished (resubmit the same command), 3 STOP.

  step      command (submit from analysis/deploy unless noted)            job wrapper / env
  prep      i3_chain.py prep [--root R]                                     jobs/cpu_d1.slurm (D1 run env)
  g2        i3_chain.py g2 [--root R] [--n-par 10] [--cores 4] [--expect-final]   jobs/cpu_d1.slurm, n_par x cores CPUs
  fake-g2   i3_chain.py fake-g2 --root $DEP/i3_dev [--cores 4]              jobs/cpu_d1.slurm (DRY RUN ONLY, never R=$DEP/i3)
  lm        i3_chain.py lm [--root R]                                       jobs/cpu_d1.slurm
  ft        ../deploy/i3_chain.py ft [--root R] [--steps slice,merge,atoms,gate,tokens]
                                     from analysis/d1_pair: jobs/g.slurm, --export=ALL,PYENV=venv5,D1PAIR_PHASE=F
  predict   i3_chain.py predict [--root R]                                  jobs/cpu_py.slurm DEPENV=d1fu2 (CPU is enough)
  report    i3_chain.py report [--root R]                                   jobs/cpu_py.slurm DEPENV=d1fu2
  selftest  i3_chain.py selftest [--root R]                                 any env with numpy (subset-rule unit checks)
Real I-3 = jobs/i3_chain.sh (one-shot, login node): prep -> g2 (40 CPUs, 64G) -> lm -> ft (10 CPUs) -> predict
(afterok + kill-on-invalid-dep) -> report (afterany), all --time=48:00:00, logs R/logs/i3_*. Dry run (done once on
$DEP/i3_dev): prep -> fake-g2 -> [g2 --expect-final: g2_deploy pool wiring, every reaction skipped] -> lm -> ft ->
predict -> report (-> analysis/deploy/results/I3_chain_dry.json).

Wiring. The parent command runs in the job env with the production dep_common paths and starts every DEPLOY script
as a child with i3_env(R): all DEP_* variables of the job removed except DEP_JOB_T0 / DEP_STOP_AFTER_H, then
  DEP_SCRATCH=R (dep_common write guard = R and R/results only; feat_lf F_DIR = R/F), DEP_RESULTS=R/results,
  DEP_LF_FINAL=R/g2lf_final, DEP_LABELS_LF=R/labels_lf.json, DEP_META_LF=R/input_meta_lf.csv, DEP_X182_LF=R/X182_lf.parquet,
  DEP_FU2_LF=R/fu2_lf, DEP_TOK_LF=R/fu2_lf/tokens, DEP_PROF_EMPTY=R/prof_empty (empty), DEP_I_DIR=R/I,
  DEP_MODEL_DIR=$DEP/model and DEP_UNITS=$DEP/units (production, read only: outside the child write guard).
  $G2 (g2_final, jobs), $LAD, the D1 manifest keep their defaults (read only). g2_deploy.py sets D1G2_SCRATCH=R itself.
Step records: R/results/i3_steps/<step>.json (argv, exit codes, subset exceptions with their reason). report ->
R/results/I3_chain.json + analysis/deploy/results/I3_chain.json (R = $DEP/i3) or I3_chain_dry.json (other roots).

Subset mode (10 reactions; nothing in the existing DEPLOY files is changed - unchanged commands first, overrides only
here, each recorded with its reason):
  L  finalize_lf.py report --jobs R/i3_list.csv (its subset mode: n_g2_ok == 8,616 skipped). Real run: L-1 (freq /
     stage failures counted against the LADDER G2 run; 0.5 % of 10 < 1 reaction), L-2 (permutation of $G2/g2_final
     within 1e-6 A) and L-4 (|d nu| <= 1 cm-1) cannot hold for a new, non-deterministic autodE run by design -> recorded,
     replaced by the report's LADDER comparison; a .done folder with missing files / keys stays a STOP. Dry run (copied
     D1-G2 work folders = the LADDER structures): every L gate must pass.
  M  meta_lf.py run unchanged when L passed and every result.json source_dir exists; otherwise _meta = meta_lf.cmd_run
     with (a) the L condition 'complete + same subset' (failed L gates recorded) and (b) build_i3 = meta_lf.build with a
     source fallback: g2_deploy.py finalises from the node-local work folder ($TMPDIR, deleted after keep_small), so
     result.json source_dir no longer exists and meta_lf.build would STOP on every deployed reaction (finding, reported);
     the fallback reads R/jobs/<job_id> (keep_small copy) and requires the sha256 of every input finalize_lf recorded.
  F  feat_lf.py slice 0 1 / merge unchanged. F-3 (feature failures <= 2 %) is not judgeable on 10 (2 % of 10 < 1):
     a merge STOP whose only reasons are F-3 is accepted; the failed reactions have no X182 row (their pairs are
     excluded by predict_tf with the reason). F-1 / non-finite stay STOP. F-2 (redetection) is reported.
  T  tokens_lf.py atoms 0 1 / gate / tokens unchanged, except: after an accepted F-3 or Q-a subset exception they run
     through _tok (tokens_lf with feat_lf.need_f_report / tokens_lf.q_check replaced by versions that relax only that
     rate rule). Q-a <= 0.5 % (0.5 % of 10 < 1) not judgeable; FU2's own gate (Q-a > 42, Q-b, Q-c) stays STOP. T element
     check (T_report elements == all 9) cannot hold without Cl / Br / F / Si / S among 10 reactions: accepted when the
     elements are a subset of the 9 (= predict_tf's own input check), and so is tokens_lf's d_in check, which it
     computes from that element count (dry run: 5 elements -> 25; the model one-hot keeps the 9 -> 29); every other T
     check stays STOP (ft --force-tok runs the _tok path without an exception: dry-run test only). Known subset
     fragility (STOP, not relaxed): FU2 drops the 3 ext columns when < 99 % of the reactions pass the ext checks
     (2 / 8,504 failed in production) - with 10 reactions one ext failure would make d_in 26 != 29.
  P  predict_tf.py run R/fu2_lf/tokens R/X182_lf.parquet R/i3_pairs.csv R/results/I3_predictions.parquet --model final
     --t-report R/results/fu2_lf/T_report.json; and the same pairs on the production lf inputs ($DEP/fu2_lf/tokens,
     $DEP/X182_lf.parquet) -> I3_predictions_lfprod.parquet (value reference for re-found TSs).
  report  per-step status, counts, output formats; LADDER comparison per reaction: new ts / rel1 / rel2 vs
     $G2/g2_final (L-2 rule: atom permutation within 1e-6 A) and, in autodE order, Kabsch RMSD vs $G2/jobs/<T> (the
     same structures as $G2/g2_final, permutation checked); re-found TS = identical (L-2) or same element order + same
     formed pairs + same A_idx + heavy-atom RMSD <= 0.05 A (decision: recommended threshold, raw values reported); for
     re-found TSs the X182 row, token and prediction differences vs the production lf inputs.
Decisions (recommended options, recorded): selection = g2_deploy.py i3-prep unchanged (seed 20261019; outputs keep
its names i3_manifest.csv / i3_list.csv / i3_pairs.csv); g2 pool 10 x 4 cores (D1-G2 used 4 cores per reaction);
predict on CPU (env_d1fu2); TS templates = g2_deploy default copy of $G2/ts_templates (they include the templates the
D1-G2 run saved, so re-finding the LADDER TS is favoured, as in deployment); dry-run fake stage = g2_deploy.cmd_one with
the three autodE stages replaced by a copy of $G2/jobs/<T> (inputs.tar.gz unpacked so keep_small re-packs it),
finalised from a node-local work folder exactly like g2_deploy (ORCA XTB2 Freq is run).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402            (stdlib only)

DEP_PROD = Path("/gpfs/tmp_cpu2/yeseo1ee/deploy")         # = dep_common default DEP (production, read only here)
DEFAULT_ROOT = DEP_PROD / "i3"
G_STATUS = C.REPO / "analysis" / "d1_g2" / "results" / "G_status.csv"
D1PAIR_DIR = C.REPO / "analysis" / "d1_pair"
KEEP_DEP = ("DEP_JOB_T0", "DEP_STOP_AFTER_H")
N_RXN, N_PAIRS, N_DIPOLES = 10, 5, 5
# gate constants of the existing DEPLOY code (asserted against the modules in the children that import them)
L1_MAX = 0.005                     # finalize_lf.L1_MAX_FREQ_FAIL
F3_MAX = 0.02                      # feat_lf.F3_MAX
QA_RATE_MAX = 0.005                # tokens_lf.QA_RATE_MAX
SAME_TS_RMSD_HEAVY = 0.05          # A, re-found TS rule (report)
TOL_PERM_A = 1e-6                  # L-2
H_KCAL = 627.5094740631
SRC_KEYS = ("ts.xyz", "ts_result.json", "ref_dipole.xyz", "ref_dipolarophile.xyz", "ref_result.json",
            "stereo_result.json")
TOKEN_KEYS = ("rxn_id", "Z", "bin", "num", "pos", "edges", "etype", "glob", "n_bonds_rel", "token_version")
OUT_COLS = ["pair_id", "a_rxn", "b_rxn", "target", "yhat", "yhat_sd", "out_of_range"]       # = predict_tf.OUT_COLS
READ_KEYS = ("rxn_id", "charge1", "charge2", "role1", "role2", "n_atoms", "n_f1", "n_f2", "status", "alt_used",
             "flag_foreign_bond", "flag_async", "formed_pairs_ts", "d1_kcal", "d2_kcal", "e_bond_kcal", "e_ab_eh",
             "e_frag1_dist_eh", "e_frag2_dist_eh", "e_frag1_rel_eh", "e_frag2_rel_eh", "elst_dft", "pauli_dft", "oi_dft",
             "disp_dft", "cpcm_dft", "cds_dft")                                                 # = meta_lf.READ_KEYS
DFT_KEYS = READ_KEYS[13:]

L_WHY = {
    "L1": "L-1 counts against the LADDER G2 run (G_status ok, its stage markers): a reaction the new autodE run fails "
          "(any stage or the freq) counts as a freq failure, and 0.5 % of 10 reactions is < 1 reaction - not judgeable "
          "on the I-3 subset (n_g2_ok == 8,616 is already skipped by finalize_lf report --jobs)",
    "L2": "L-2 asks every new ts / rel xyz to be an atom permutation of $G2/g2_final within 1e-6 A - true by "
          "construction for the D1-G2 work folders (Phase L), not for a new non-deterministic autodE run; replaced by "
          "the report's LADDER comparison",
    "L4": "L-4 asks |d nu_imag| <= 1 cm-1 vs $G2/g2_final for >= 99 % of the reactions - same reason as L-2",
}


# ================================================================ helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _abs(p) -> Path:
    return Path(os.path.abspath(str(p)))


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(HERE), capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:                                                 # noqa: BLE001
        return None


def check_root(root) -> Path:
    """R must be a strict sub-folder $DEP/i3* (never $DEP itself or a production folder)."""
    r = _abs(root)
    if r.parent != DEP_PROD or not r.name.startswith("i3"):
        C.stop(f"--root {r}: must be {DEP_PROD}/i3* (e.g. {DEFAULT_ROOT}, {DEP_PROD}/i3_dev)")
    return r


def check_parent():
    """parent commands run with the production dep_common paths (the children get the overrides)."""
    if _abs(C.DEP) != DEP_PROD or _abs(C.RES) != _abs(HERE / "results"):
        C.stop(f"run without DEP_SCRATCH / DEP_RESULTS overrides (dep_common DEP {C.DEP}, RES {C.RES})")


def check_child(root):
    if _abs(C.DEP) != _abs(root) or _abs(C.RES) != _abs(root / "results"):
        C.stop(f"child without the i3 environment: DEP {C.DEP}, RES {C.RES}, root {root}")


def overrides(root):
    r = _abs(root)
    return {"DEP_SCRATCH": r, "DEP_RESULTS": r / "results", "DEP_LF_FINAL": r / "g2lf_final",
            "DEP_LABELS_LF": r / "labels_lf.json", "DEP_META_LF": r / "input_meta_lf.csv",
            "DEP_X182_LF": r / "X182_lf.parquet", "DEP_FU2_LF": r / "fu2_lf", "DEP_TOK_LF": r / "fu2_lf" / "tokens",
            "DEP_PROF_EMPTY": r / "prof_empty", "DEP_I_DIR": r / "I", "DEP_MODEL_DIR": DEP_PROD / "model",
            "DEP_UNITS": DEP_PROD / "units"}


def i3_env(root, base=None):
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items() if not k.startswith("DEP_") or k in KEEP_DEP}
    env.update({k: str(v) for k, v in overrides(root).items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def marker(d):
    """'.done', the first sorted '.fail_*' name, or None (= finalize_lf.marker)."""
    d = Path(d)
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def run_mode(root):
    return "dry_run" if (Path(root) / "results" / "i3_steps" / "fake-g2.json").is_file() else "real"


def selection(root):
    """(manifest frame, job ids of the list, pairs frame, job_id -> rxn_id)."""
    import pandas as pd
    root = Path(root)
    for f in ("i3_manifest.csv", "i3_list.csv", "i3_pairs.csv"):
        if not (root / f).is_file():
            C.not_ready(f"{root / f} missing (i3_chain.py prep)")
    man = pd.read_csv(root / "i3_manifest.csv", keep_default_na=False, dtype=str)
    ids = pd.read_csv(root / "i3_list.csv", dtype=str).job_id.tolist()
    P = pd.read_csv(root / "i3_pairs.csv", dtype=str)
    rk = {r.job_id: int(float(r.meta_rxn_id)) for r in man.itertuples()}
    return man, ids, P, rk


def check_env(kind):
    if kind == "d1":
        d1 = os.environ.get("D1_PY")
        if not d1 or _abs(Path(sys.executable).resolve()) != _abs(Path(d1).resolve()):
            C.stop(f"needs the D1 run env (jobs/cpu_d1.slurm); python {sys.executable}, D1_PY {d1}")
    elif kind == "venv5":
        if os.environ.get("D1PAIR_PHASE") != "F" or os.environ.get("PYENV") != "venv5":
            C.stop("needs analysis/d1_pair/jobs/g.slurm with --export=ALL,PYENV=venv5,D1PAIR_PHASE=F "
                   f"(D1PAIR_PHASE={os.environ.get('D1PAIR_PHASE')}, PYENV={os.environ.get('PYENV')})")
    elif kind == "d1fu2":
        if "env_d1fu2" not in sys.executable:
            C.stop(f"needs env_d1fu2 (jobs/cpu_py.slurm DEPENV=d1fu2 or jobs/gpu_fu2.slurm); python {sys.executable}")


_CUR = {}


class Step:
    """the record of one parent command -> R/results/i3_steps/<name>.json (written at the end, also on STOP)."""

    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.rec = dict(step=name, root=str(root), mode=run_mode(root), started=now(), host=socket.gethostname(),
                        slurm_job=os.environ.get("SLURM_JOB_ID"), cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
                        python=sys.executable, git=git_head(), overrides={k: str(v) for k, v in overrides(root).items()},
                        commands=[], subset_exceptions=[], notes=[])
        self.t0 = time.time()
        self.done = False
        _CUR["step"] = self

    def run(self, label, argv, env, cwd=HERE):
        argv = [str(a) for a in argv]
        t = time.time()
        print(f"=== [{now()}] {label}: {' '.join(argv)}", flush=True)
        r = subprocess.run(argv, env=env, cwd=str(cwd))
        print(f"=== [{now()}] {label}: exit {r.returncode} ({time.time() - t:.0f} s)", flush=True)
        self.rec["commands"].append(dict(label=label, argv=argv, rc=r.returncode, wall_s=round(time.time() - t, 1)))
        return r.returncode

    def subset(self, gate, why, detail=None):
        self.rec["subset_exceptions"].append(dict(gate=gate, why=why, detail=detail))
        print(f"[I-3 subset] {gate}: {why}", flush=True)

    def finish(self, rc, status=None):
        if status is None:
            status = {0: "ok_subset" if self.rec["subset_exceptions"] else "ok", 2: "not_ready"}.get(rc, "stop")
        self.rec.update(status=status, rc=int(rc), ended=now(), wall_s=round(time.time() - self.t0, 1))
        C.write_json(self.root / "results" / "i3_steps" / f"{self.name}.json", self.rec)
        self.done = True
        print(f"[i3_chain {self.name}] {status} (exit {rc})", flush=True)
        return int(rc)


def step_ok(root, name):
    r = read_json(Path(root) / "results" / "i3_steps" / f"{name}.json") or {}
    return r.get("status") in ("ok", "ok_subset")


# ================================================================ gate classification (subset rules)
def classify_L(rep, mode):
    """L_report.json -> dict(failed, soft, hard, why). Real run: L-1 / L-2 / L-4 failures are subset / design
    exceptions (L_WHY); dry run: none allowed. Format problems of .done folders are always hard."""
    failed = [g for g in ("L1", "L2", "L4") if not (rep.get(g) or {}).get("pass")]
    hard = []
    if not rep.get("complete"):
        hard.append("L_report incomplete")
    if rep.get("subset") is None:
        hard.append("L_report is not a --jobs subset report")
    l1 = rep.get("L1") or {}
    if l1.get("n_done_incomplete"):
        hard.append(f"L-1: {l1['n_done_incomplete']} .done folders with missing files / keys: {l1.get('done_incomplete')}")
    if l1.get("n_done_g2_not_ok"):
        hard.append(f"L-1: {l1['n_done_g2_not_ok']} .done reactions that are not G2 ok (selection is G2 ok only)")
    if mode == "dry_run":
        hard += [f"{g} failed on copied D1-G2 work folders (the LADDER structures): bug" for g in failed]
        soft = []
    else:
        soft = list(failed)
    return dict(failed=failed, soft=soft, hard=hard, why={g: L_WHY[g] for g in soft})


def classify_F(rep):
    """F_report.json STOP reasons -> (soft, hard): F-3 only is soft when F3_MAX x n < 1 (one failure exceeds it)."""
    reasons = list(rep.get("stop_reasons") or [])
    n = int(rep.get("n_done") or 0)
    soft = [r for r in reasons if r.startswith("F-3:") and n > 0 and n * F3_MAX < 1]
    return soft, [r for r in reasons if r not in soft]


def classify_Q(rec):
    """tokens_lf q_check record (T_lf_check.json 'gate') -> (soft, hard): only the 0.5 % rate rule is soft when
    QA_RATE_MAX x n < 1; FU2's own gate (Q_report stop), atom version and a failed gate child stay hard."""
    reasons = list(rec.get("stop_reasons") or [])
    n = int(rec.get("n_target") or 0)
    soft = [r for r in reasons if r.startswith("Q-a failures") and n > 0 and n * QA_RATE_MAX < 1]
    hard = [r for r in reasons if r not in soft]
    if rec.get("gate_rc") not in (None, 0):          # the FU2 child gate (Q-a > 42, Q-b, Q-c); the rate rule is parent-side
        hard.append(f"FU2 gate child exit {rec.get('gate_rc')}")
    if rec.get("retry_rc") not in (None, 0):
        hard.append(f"atoms retry pass exit {rec.get('retry_rc')}")
    return soft, hard


def classify_T(rec):
    """tokens_lf T_lf_check.json 'tokens' record -> (soft, hard): the element check is soft when the elements are a
    non-empty subset of the 9 model elements; d_in is soft when it differs only through that element count
    (tokens_lf.t_check computes d_in = len(T_report elements) + 6 + 14; the model one-hot always has the 9);
    every other check and a failed tokens child stay hard."""
    checks = rec.get("checks") or {}
    failed = [k for k, v in checks.items() if not v]
    el = [int(z) for z in rec.get("elements") or []]
    soft = ["elements"] if "elements" in failed and el and set(el) <= set(C.ELEMENTS) else []
    if soft and "d_in" in failed and checks.get("bin_cols") and checks.get("num_cols") and \
            int(rec.get("d_in") or -1) - len(el) + len(C.ELEMENTS) == 29:
        soft.append("d_in")
    hard = [f"T check {k}" for k in failed if k not in soft]
    if rec.get("tokens_rc") not in (None, 0):
        hard.append(f"tokens child exit {rec.get('tokens_rc')}")
    if not checks:
        hard.append("no T checks recorded")
    return soft, hard


# ================================================================ prep
def cost_estimate(ids, n_par, cores):
    """autodE cost of the selected reactions from the D1-G2 run (G_status.csv core_h_total = wall x 4 cores)."""
    import numpy as np
    import pandas as pd
    st = pd.read_csv(G_STATUS, dtype={"job_id": str})
    s = st[st.job_id.isin(ids)].copy()
    cols = ["core_h_total", "core_h_ts", "core_h_stereo", "core_h_ref", "core_h_freq"]
    s["wall_h"] = s.core_h_total / s.cores
    ok = st[st.status == "ok"]
    wall_max = float(s.wall_h.max())
    waves = math.ceil(len(s) / max(1, n_par))
    per = s[["job_id", "rxn_id", "n_atoms", "dipole", "status", "cores"] + cols + ["wall_h"]].to_dict("records")
    return dict(
        source=str(G_STATUS), source_sha256=C.sha256(G_STATUS), n=int(len(s)),
        d1g2_core_h_sum=float(s.core_h_total.sum()), d1g2_core_h_by_stage={c: float(s[c].sum()) for c in cols[1:]},
        d1g2_wall_h=dict(max=wall_max, sum=float(s.wall_h.sum()), median=float(s.wall_h.median())),
        d1g2_all_ok_core_h=dict(n=int(len(ok)), median=float(ok.core_h_total.median()),
                                p90=float(np.quantile(ok.core_h_total, 0.9)), max=float(ok.core_h_total.max())),
        per_reaction=per,
        i3=dict(n_par=n_par, cores=cores, alloc_cpus=n_par * cores, waves=waves,
                wall_h_expected=wall_max if waves == 1 else float(s.wall_h.sum() / n_par),
                alloc_core_h_expected=n_par * cores * (wall_max if waves == 1 else float(s.wall_h.sum() / n_par)),
                used_core_h_expected=float(s.core_h_total.sum()),
                note="autodE is not deterministic: a new run can take another path (other conformers / template "
                     "misses); the D1-G2 cost of the same reactions is the expectation, not a bound. finalize_lf "
                     "freq is included in core_h_total (core_h_freq); F / T / predict add < 0.01 core-h per reaction."))


def cmd_prep(root, n_par=10, cores=4):
    st = Step(root, "prep")
    for d in (root, root / "logs", root / "results", root / "prof_empty"):
        C.assert_write_ok(d)
        d.mkdir(parents=True, exist_ok=True)
    if any((root / "prof_empty").iterdir()):
        C.stop(f"{root / 'prof_empty'} is not empty")
    rc = st.run("g2_deploy i3-prep", [sys.executable, HERE / "g2_deploy.py", "i3-prep", "--out", root], i3_env(root))
    if rc:
        return st.finish(rc)
    man, ids, P, rk = selection(root)
    jobs = sorted(set(P.a_job) | set(P.b_job))
    chk = dict(n_pairs=len(P) == N_PAIRS, n_dipoles=P.dipole.nunique() == N_DIPOLES, n_rxn=len(jobs) == N_RXN,
               list_eq_pairs=sorted(ids) == jobs, manifest_eq_list=sorted(man.job_id) == jobs,
               kind_d1=bool((man.kind == "D1").all()), mapped_smiles=bool((man.mapped_smiles.str.strip() != "").all()),
               rxn_ids=all(int(P.a_rxn[k]) == rk[P.a_job[k]] and int(P.b_rxn[k]) == rk[P.b_job[k]] for k in range(len(P))),
               neutral=bool((man.charge.astype(int) == 0).all()))
    import pandas as pd
    S = pd.read_csv(D1PAIR_DIR / "results" / "S_folds.csv", dtype=str).set_index("pair_id")
    pr = [dict(pair_id=p.pair_id, dipole=p.dipole, a_job=p.a_job, a_rxn=int(p.a_rxn), b_job=p.b_job, b_rxn=int(p.b_rxn),
               ref_kind=S.loc[p.pair_id, "ref_kind"], pos=S.loc[p.pair_id, "pos"], Xb=S.loc[p.pair_id, "Xb"],
               core=S.loc[p.pair_id, "core"], skeleton=S.loc[p.pair_id, "skeleton"], fold_T1=S.loc[p.pair_id, "fold_T1"])
          for p in P.itertuples()]
    st.rec.update(checks=chk, pairs=pr, jobs=jobs,
                  files={f: C.sha256(root / f) for f in ("i3_manifest.csv", "i3_list.csv", "i3_pairs.csv")},
                  cost_estimate=cost_estimate(jobs, n_par, cores))
    return st.finish(0 if all(chk.values()) else 3)


# ================================================================ g2 (real) and fake-g2 (dry run)
def per_reaction_markers(root, ids, rk):
    out = {}
    for T in ids:
        d = root / "g2lf_final" / str(rk[T])
        res = read_json(d / "result.json") or {}
        dt = read_json(root / "jobs" / T / "deploy_times.json") or {}
        out[T] = dict(rxn_id=rk[T], marker=marker(d), status=res.get("status"), fail_reasons=res.get("fail_reasons"),
                      deploy_status=dt.get("status"), wall_s=dt.get("wall_s"), core_h=dt.get("core_h"),
                      fake_stage=dt.get("fake_stage", False))
    return out


def cmd_g2(root, n_par, cores, expect_final=False):
    check_env("d1")
    st = Step(root, "g2")
    if not step_ok(root, "prep"):
        C.not_ready("prep not done")
    man, ids, P, rk = selection(root)
    missing = [T for T in ids if marker(root / "g2lf_final" / str(rk[T])) is None]
    st.rec.update(n_par=n_par, cores=cores, n_missing_before=len(missing), expect_final=expect_final)
    if expect_final and missing:
        C.stop(f"--expect-final: {len(missing)} reactions have no final marker (g2 would start autodE): {missing}")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    if missing and cpus and n_par * cores > cpus:
        st.rec["notes"].append(f"n_par x cores = {n_par * cores} > SLURM_CPUS_PER_TASK {cpus} (oversubscribed)")
    rc = st.run("g2_deploy pool", [sys.executable, HERE / "g2_deploy.py", "pool", root / "i3_list.csv", n_par, cores,
                                   "--manifest", root / "i3_manifest.csv", "--out", root], i3_env(root))
    rx = per_reaction_markers(root, ids, rk)
    st.rec["reactions"] = rx
    from collections import Counter
    st.rec["markers"] = dict(Counter(str(v["marker"]) for v in rx.values()))
    if rc == 0 and any(v["marker"] is None for v in rx.values()):
        rc = 2
    return st.finish(rc)


def cmd_fake_g2(root, cores):
    check_env("d1")
    if root.name == DEFAULT_ROOT.name:
        C.stop(f"fake-g2 is the dry run only: refusing {root} (the real I-3 root must come from g2_deploy.py)")
    st = Step(root, "fake-g2")
    st.rec["mode"] = "dry_run"
    if not step_ok(root, "prep"):
        C.not_ready("prep not done")
    man, ids, P, rk = selection(root)
    rc = st.run("fake g2 stage", [sys.executable, THIS, "_fake_g2", "--root", root, "--cores", cores], i3_env(root))
    rx = per_reaction_markers(root, ids, rk)
    st.rec.update(reactions=rx, cores=cores, source=str(C.G2_JOBS),
                  note="ts / stereo / ref stages NOT run: $G2/jobs/<T> copied into a node-local work folder, then "
                       "finalize_lf + keep_small exactly as g2_deploy.cmd_one")
    if rc == 0 and any(v["marker"] is None for v in rx.values()):
        rc = 2
    return st.finish(rc)


def child_fake_g2(root, cores):
    """= g2_deploy.cmd_one per reaction with stage_ts / stage_stereo / stage_ref replaced by a copy of the D1-G2 work
    folder $G2/jobs/<T> (+ its inputs.tar.gz unpacked, so keep_small re-packs the same ORCA inputs)."""
    import tarfile
    import yaml
    import g2_deploy as GD
    check_child(root)
    M = GD.setup(root)
    G, GA, FL, root = M["G"], M["GA"], M["FL"], M["root"]
    import pandas as pd
    manifest = root / "i3_manifest.csv"
    GD.check_manifest(manifest)
    ids = [str(x) for x in pd.read_csv(root / "i3_list.csv", dtype=str).job_id]
    fin, jobs = root / "g2lf_final", root / "jobs"
    out = {}
    for T in ids:
        job = GA.pc.load_job(T, manifest)
        rk = FL.rxn_key(job)
        if FL.marker(fin / str(rk)) is not None:
            out[T] = "skip" + str(FL.marker(fin / str(rk)))
            continue
        src = C.G2_JOBS / T
        if not src.is_dir():
            C.not_ready(f"{src} missing")
        os.environ["SLURM_CPUS_PER_TASK"] = str(cores)
        os.environ["OMP_NUM_THREADS"] = str(cores)
        local = _abs(os.environ.get("TMPDIR", "/tmp")) / f"d1dep_fake_{os.getpid()}"
        cfg = yaml.safe_load(Path(GA.CONFIG).read_text())
        cfg["scratch"] = str(local)
        local.mkdir(parents=True, exist_ok=True)
        (local / "config_g2_local.yaml").write_text(yaml.safe_dump(cfg))
        jd = GA.pc.job_dir(cfg, T)
        try:
            for p in sorted(src.iterdir()):
                if not p.is_file():
                    continue
                if p.name == "inputs.tar.gz":
                    with tarfile.open(p) as t:
                        for m in t.getmembers():
                            if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts:
                                t.extract(m, jd)
                else:
                    shutil.copy2(p, jd / p.name)
            os.chdir(local)
            t0 = time.time()
            s = FL.finalize_lf(job, jd, fin, cores)
            w = time.time() - t0
            (jd / "deploy_times.json").write_text(json.dumps(dict(
                job_id=T, rxn_key=str(rk), status=s, cores=cores, wall_s=dict(finalize=w),
                core_h=dict(finalize=w * cores / 3600), manifest=str(manifest), manifest_sha256=C.sha256(manifest),
                root=str(root), code="analysis/deploy/i3_chain.py _fake_g2", code_sha256=C.sha256(THIS),
                fake_stage=True, source=str(src),
                note="I-3 dry run: ts / stereo / ref = the D1-G2 work folder (not run); finalize_lf run"), indent=1))
            GA.keep_small(jd, jobs / T)
        finally:
            os.chdir("/")
            shutil.rmtree(local, ignore_errors=True)
        out[T] = s
        print(f"  {T} -> {rk}: {s}  ({w:.1f} s)", flush=True)
    print(json.dumps(out, indent=1), flush=True)
    return 0


# ================================================================ lm
def cmd_lm(root):
    check_env("d1")
    st = Step(root, "lm")
    mode = run_mode(root)
    if not (step_ok(root, "g2") or step_ok(root, "fake-g2")):
        C.not_ready("neither g2 nor fake-g2 finished ok")
    man, ids, P, rk = selection(root)
    miss = [T for T in ids if marker(root / "g2lf_final" / str(rk[T])) is None]
    if miss:
        C.not_ready(f"no final marker for {miss}")
    env = i3_env(root)
    rcL = st.run("finalize_lf report --jobs", [sys.executable, HERE / "finalize_lf.py", "report", "--jobs",
                                               root / "i3_list.csv"], env)
    lrep = read_json(root / "results" / "L_report.json")
    if rcL == 2 or not lrep or not lrep.get("complete"):
        return st.finish(2 if rcL in (0, 2) else rcL)
    if rcL not in (0, 3):
        return st.finish(3)
    cls = classify_L(lrep, mode)
    st.rec["L"] = dict(rc=rcL, all_pass=lrep.get("all_pass"), **cls)
    if cls["hard"]:
        st.rec["notes"].append("L hard failures: " + "; ".join(cls["hard"]))
        return st.finish(3)
    for g in cls["soft"]:
        st.subset(g, L_WHY[g], {k: v for k, v in (lrep.get(g) or {}).items() if not isinstance(v, (list, dict))})
    done = [rk[T] for T in ids if marker(root / "g2lf_final" / str(rk[T])) == ".done"]
    src_missing = [r for r in done if not Path((read_json(root / "g2lf_final" / str(r) / "result.json") or {})
                                              .get("source_dir", "/nonexistent")).is_dir()]
    st.rec.update(n_done=len(done), n_source_dir_missing=len(src_missing))
    if lrep.get("all_pass") and not src_missing:
        st.rec["M_path"] = "meta_lf.py run (unchanged)"
        rcM = st.run("meta_lf run --jobs", [sys.executable, HERE / "meta_lf.py", "run", "--jobs", root / "i3_list.csv"], env)
    else:
        st.rec["M_path"] = "i3_chain.py _meta (meta_lf.cmd_run with the I-3 L condition + source fallback)"
        if src_missing:
            st.subset("M:source_dir", "g2_deploy.py finalises from the node-local work folder ($TMPDIR, removed after "
                      "keep_small): result.json source_dir is gone, meta_lf.build would STOP; the keep_small copy "
                      "<root>/jobs/<job_id> is read instead with every recorded input sha256 checked (DEPLOY finding: "
                      "the deployment path cannot run meta_lf.py unchanged)", dict(n=len(src_missing)))
        rcM = st.run("i3 _meta", [sys.executable, THIS, "_meta", "--root", root], env)
    mrep = read_json(root / "results" / "M_report.json") or {}
    st.rec["M"] = {k: mrep.get(k) for k in ("n_done_folders", "n_records", "n_problems", "n_charge_nonzero",
                                            "read_keys_missing", "n_pairs_not_across_partition", "nan_roundtrip_ok",
                                            "all_pass", "stop", "i3_override")}
    return st.finish(rcM)


def build_i3(ML, lf_root, man, rids, jobs_root):
    """= meta_lf.build (meta_lf.py 64-108) with one change: when result.json source_dir is gone (g2_deploy node-local
    work folder) the keep_small copy jobs_root/<job_id> is read, and every input sha256 finalize_lf recorded must match.
    Returns build's 5 lists + {rxn_id: source used}."""
    recs, meta, charges, prob, cross, used = [], [], [], [], [], {}
    for rid in rids:
        d = Path(lf_root) / str(rid)
        try:
            res = json.loads((d / "result.json").read_text())
            src = Path(res["source_dir"])
            sha = res["inputs_sha256"]
            how = "source_dir"
            if not src.is_dir():
                src, how = Path(jobs_root) / str(res["job_id"]), "jobs_copy"
                bad = [f for f, s in sha.items() if not (src / f).is_file() or C.sha256(src / f) != s]
                if bad or "ts.xyz" not in sha:
                    raise ValueError(f"{src}: inputs differ from the sha256 finalize_lf recorded: {bad}")
            for f in ("ts_result.json", "ref_result.json"):
                if C.sha256(src / f) != sha.get(f):
                    raise ValueError(f"{src / f} changed since finalize_lf (sha256)")
            tsr = json.loads((src / "ts_result.json").read_text())
            refr = json.loads((src / "ref_result.json").read_text())
            n = int((d / "ts.xyz").read_text().split()[0])
            job = str(res["job_id"])
            m = man.get(job)
            if m is None or int(m["meta_rxn_id"]) != int(rid):
                raise ValueError(f"manifest row for {job} missing or meta_rxn_id != {rid}")
            if res.get("status") != "ok" or int(res["rxn_id"]) != int(rid):
                raise ValueError(f"result.json status {res.get('status')} / rxn_id {res.get('rxn_id')}")
            if sorted(int(i) for i in tsr["A_idx"]) != [int(i) for i in res["A_idx"]] or \
                    ML.fp_string(tsr["formed_pairs_ts"]) != ML.fp_string(res["formed_pairs_ts"]):
                raise ValueError("result.json A_idx / formed_pairs_ts != ts_result.json")
            rec = ML.make_record(rid, res, tsr, refr, n, job)
            if not (0 < rec["n_f1"] < n) or any(not (0 <= int(i) < n) for p in res["formed_pairs_ts"] for i in p):
                raise ValueError(f"n_f1 {rec['n_f1']} / pair indices out of range (n_atoms {n})")
        except Exception as e:                                        # noqa: BLE001
            prob.append(dict(rxn_id=int(rid), problem=f"{type(e).__name__}: {e}"))
            continue
        used[int(rid)] = dict(how=how, path=str(src))
        q = dict(rxn_id=int(rid), job_id=job, manifest_charge=int(m["charge"]),
                 dipole_smiles_charge=ML.formal_charge(m["dipole_smiles"]),
                 dipolarophile_smiles_charge=ML.formal_charge(m["dipolarophile_smiles"]),
                 ts_result_charge_dipole=tsr.get("charge_dipole"), ts_result_charge_dipolarophile=tsr.get("charge_dipolarophile"))
        q["nonzero"] = any(v is None or int(v) != 0 for k, v in q.items() if k not in ("rxn_id", "job_id"))
        charges.append(q)
        A = set(int(i) for i in res["A_idx"])
        cross.append(all((int(i) in A) != (int(j) in A) for i, j in res["formed_pairs_ts"]))
        recs.append(rec)
        meta.append(dict(rxn_id=int(rid), A_idx=" ".join(str(int(i)) for i in res["A_idx"])))
    return recs, meta, charges, prob, cross, used


def child_meta(root):
    """= meta_lf.cmd_run (meta_lf.py 124-197) with (a) the L condition 'L_report complete for this subset' (the failed
    L gates must be subset / design exceptions, classify_L) and (b) build_i3 instead of meta_lf.build."""
    import pandas as pd
    import meta_lf as ML
    check_child(root)
    FL = ML.FL
    if FL.L1_MAX_FREQ_FAIL != L1_MAX or tuple(ML.READ_KEYS) != READ_KEYS:
        C.stop("finalize_lf / meta_lf constants changed (L1_MAX_FREQ_FAIL / READ_KEYS)")
    jobs = pd.read_csv(root / "i3_list.csv", dtype=str).job_id.tolist()
    lrep_p = C.RES / "L_report.json"
    lrep = C.read_json(lrep_p) if lrep_p.is_file() else None
    want = sorted(jobs)
    if not lrep or not lrep.get("complete") or lrep.get("subset") != want:
        C.not_ready(f"{lrep_p}: missing, incomplete or subset {None if not lrep else lrep.get('subset')} != {want}")
    cls = classify_L(lrep, run_mode(root))
    if cls["hard"]:
        C.stop("L: " + "; ".join(cls["hard"]))
    tg = FL.load_targets(jobs)
    tg_rid = {int(t["rxn_id"]) for t in tg}
    rids = sorted(int(p.parent.name) for p in C.LF_FINAL.glob("*/.done")
                  if p.parent.name.isdigit() and int(p.parent.name) in tg_rid)
    st = pd.read_csv(C.RES / "L_status.csv")
    st_done = sorted(int(r) for r in st.rxn_id[st.lf_marker == ".done"])
    if rids != st_done:
        C.not_ready(f"{C.LF_FINAL} .done set ({len(rids)}) != L_status.csv .done set ({len(st_done)}): rerun report")
    man = FL.load_manifest()
    recs, meta, charges, prob, cross, used = build_i3(ML, C.LF_FINAL, man, rids, root / "jobs")
    Q = pd.DataFrame(charges)
    nz = Q[Q.nonzero] if len(Q) else Q
    key_missing = sorted({k for r in recs for k in ML.READ_KEYS if k not in r})
    fmt_bad = [r["rxn_id"] for r in recs if not ML.FP_RE.match(r["formed_pairs_ts"])]
    s_g2 = ML.schema_vs(ML.G2_LABELS, recs)
    s_d0 = ML.schema_vs(ML.D0_LABELS, recs)
    cmp = {}
    if isinstance(s_g2, tuple):
        s_g2, g2lab = s_g2
        L = {int(d["rxn_id"]): d for d in g2lab}
        both = [r for r in recs if r["rxn_id"] in L]
        cmp = dict(n_both=len(both), n_atoms_equal=sum(r["n_atoms"] == int(L[r["rxn_id"]]["n_atoms"]) for r in both),
                   n_f1_equal=sum(r["n_f1"] == int(L[r["rxn_id"]]["n_f1"]) for r in both),
                   formed_pairs_ts_equal=sum(r["formed_pairs_ts"] == L[r["rxn_id"]]["formed_pairs_ts"] for r in both),
                   alt_used_equal=sum(r["alt_used"] == int(L[r["rxn_id"]]["alt_used"]) for r in both),
                   note="label values (DFT TS, label atom order) vs lf (G2 TS, autodE order): information only")
        del g2lab, L
    if isinstance(s_d0, tuple):
        s_d0 = s_d0[0]
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 4 (I-3 subset, i3_chain.py _meta)", lf_root=str(C.LF_FINAL), subset=want,
               n_done_folders=len(rids), n_records=len(recs), n_problems=len(prob), problems=prob[:50],
               n_charge_nonzero=len(nz), charge_nonzero=nz.to_dict("records")[:50] if len(nz) else [],
               charge_sources=["manifest charge", "RDKit formal charge of manifest dipole_smiles / dipolarophile_smiles",
                               "ts_result.json charge_dipole / charge_dipolarophile"],
               read_keys=list(ML.READ_KEYS), read_keys_missing=key_missing, dft_keys_nan=list(ML.DFT_KEYS),
               n_formed_pairs_ts_format_bad=len(fmt_bad), formed_pairs_ts_format_bad=fmt_bad[:20],
               n_pairs_not_across_partition=sum(not x for x in cross),
               schema_vs_labels_d1_g2=s_g2, schema_vs_labels_all=s_d0, values_vs_labels_d1_g2=cmp,
               l_report_sha256=C.sha256(lrep_p), manifest_sha256=C.sha256(C.D1_MANIFEST), written=time.strftime("%F %T"),
               i3_override=dict(l_report_all_pass=lrep.get("all_pass"), l_gates_failed=cls["failed"], why=cls["why"],
                                sources={str(k): v for k, v in used.items()},
                                n_source_fallback=sum(v["how"] == "jobs_copy" for v in used.values()),
                                code="analysis/deploy/i3_chain.py build_i3 / child_meta (= meta_lf build / cmd_run)",
                                code_sha256=C.sha256(THIS)))
    stop = []
    if len(nz):
        stop.append(f"{len(nz)} reactions with a reactant charge != 0")
    if prob:
        stop.append(f"{len(prob)} .done folders whose record cannot be built consistently")
    if key_missing or fmt_bad:
        stop.append(f"record keys missing {key_missing} / formed_pairs_ts format bad {len(fmt_bad)}")
    rep["stop"] = stop
    rep["all_pass"] = not stop
    if not stop:
        recs.sort(key=lambda r: r["rxn_id"])
        C.write_json(C.LABELS_LF, recs)
        C.write_csv(C.META_LF, pd.DataFrame(meta, columns=["rxn_id", "A_idx"]).sort_values("rxn_id"))
        back = json.loads(C.LABELS_LF.read_text())
        rep["nan_roundtrip_ok"] = all(isinstance(b[k], float) and math.isnan(b[k]) for b in back[:50] for k in ML.DFT_KEYS)
        rep["outputs"] = {str(C.LABELS_LF): C.sha256(C.LABELS_LF), str(C.META_LF): C.sha256(C.META_LF)}
        if not rep["nan_roundtrip_ok"]:
            rep["stop"] = stop = ["DFT keys do not read back as NaN"]
            rep["all_pass"] = False
    C.write_json(C.RES / "M_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("n_done_folders", "n_records", "n_problems", "n_charge_nonzero",
                                          "read_keys_missing", "n_pairs_not_across_partition", "all_pass", "stop")}),
          flush=True)
    if stop:
        C.stop("; ".join(stop))
    return 0


# ================================================================ ft
FT_STEPS = ("slice", "merge", "atoms", "gate", "tokens")
GATE_STEP = {"F-3": "merge", "Q-a": "gate", "T:elements": "tokens"}
F3_WHY = ("feature failure rate <= 2 % (F-3) is not judgeable on 10 reactions (one failure = 10 %): accepted; the "
          "failed reactions have no X182 row and predict_tf excludes their pairs with the reason")
QA_WHY = ("Q-a failure rate <= 0.5 % (tokens_lf) is not judgeable on 10 reactions: accepted; FU2's own gate (Q-a > 42, "
          "Q-b, Q-c) passed; the failed reactions are not tokenised (Q_manifest use = False)")
T_WHY = ("T_report elements == all 9 model elements (and tokens_lf's d_in computed from them) cannot hold for 10 "
         "reactions without each of F / Si / S / Cl / Br: accepted as elements subset of the 9 (= predict_tf input "
         "check; the model one-hot has the 9, d_in 29); bin / num / glob columns, ext, token version, T-a must pass")


def cmd_ft(root, steps, force_tok=False):
    check_env("venv5")
    steps = [s for s in FT_STEPS if s in steps]
    prev = read_json(root / "results" / "i3_steps" / "ft.json") or {}
    st = Step(root, "ft")
    st.rec.update(steps=steps, force_tok=force_tok,
                  calls=(prev.get("calls") or []) + [dict(steps=steps, started=st.rec["started"], force_tok=force_tok,
                                                          slurm_job=st.rec["slurm_job"])],
                  steps_done=[s for s in prev.get("steps_done") or [] if s not in steps],
                  subset_exceptions=[x for x in prev.get("subset_exceptions") or [] if GATE_STEP.get(x.get("gate")) not in steps])
    for k in ("F", "Q", "T"):
        if prev.get(k) is not None and {"F": "merge", "Q": "gate", "T": "tokens"}[k] not in steps:
            st.rec[k] = prev[k]
    m = read_json(root / "results" / "M_report.json") or {}
    if not (step_ok(root, "lm") and m.get("all_pass") and (root / "labels_lf.json").is_file()):
        C.not_ready("lm not done (M_report all_pass, labels_lf.json)")
    env = i3_env(root)
    res = root / "results"
    py = sys.executable
    has = lambda g: any(x.get("gate") == g for x in st.rec["subset_exceptions"])      # noqa: E731

    def f_subset():
        r = read_json(res / "F_report.json")
        if not r or not r.get("stop"):
            return False
        soft, hard = classify_F(r)
        if hard:
            st.rec["notes"].append(f"F STOP: {hard}")
            C.stop(f"F_report STOP: {hard}")
        if not has("F-3"):
            st.subset("F-3", F3_WHY, dict(reasons=soft, not_ok=r.get("not_ok")))
        return True

    def q_subset():
        g = (read_json(res / "T_lf_check.json") or {}).get("gate") or {}
        if not (g.get("stop") or g.get("i3_subset_relaxed")):
            return False
        soft, hard = classify_Q(g)
        if hard:
            st.rec["notes"].append(f"Q STOP: {hard}")
            C.stop(f"Q gate STOP: {hard}")
        if not has("Q-a"):
            st.subset("Q-a", QA_WHY, dict(reasons=soft or g.get("i3_subset_relaxed")))
        return True

    if "slice" in steps:
        rc = st.run("feat_lf slice 0 1", [py, HERE / "feat_lf.py", "slice", 0, 1], env)
        if rc:
            return st.finish(rc)
        st.rec["steps_done"].append("slice")
    if "merge" in steps:
        rc = st.run("feat_lf merge", [py, HERE / "feat_lf.py", "merge"], env)
        r = read_json(res / "F_report.json") or {}
        if rc == 3 and r.get("stop"):
            f_subset()                                                # STOP unless F-3 on the subset only
        elif rc:
            return st.finish(rc)
        st.rec["F"] = {k: r.get(k) for k in ("n_done", "n_feat", "F_fail_rate", "reasons", "stop_reasons")}
        st.rec["F"]["F2_n"] = (r.get("F2") or {}).get("n")
        st.rec["steps_done"].append("merge")
    fs = (f_subset() if set(steps) & {"atoms", "gate", "tokens"} else False) or force_tok

    def tok(cmd, args):
        if fs:
            return st.run(f"i3 _tok {cmd} ({'forced' if force_tok else 'F-3 subset'})",
                          [py, THIS, "_tok", cmd, "--root", root], env)
        return st.run(f"tokens_lf {cmd}", [py, HERE / "tokens_lf.py", cmd] + list(args), env)

    if "atoms" in steps:
        rc = tok("atoms", [0, 1])
        if rc:
            return st.finish(rc)
        st.rec["steps_done"].append("atoms")
    if "gate" in steps:
        rc = tok("gate", [])
        g = (read_json(res / "T_lf_check.json") or {}).get("gate") or {}
        if rc == 3 and g.get("stop"):
            q_subset()                                                # STOP unless the Q-a rate rule only
        elif rc:
            return st.finish(rc)
        elif g.get("i3_subset_relaxed"):
            q_subset()
        st.rec["Q"] = {k: g.get(k) for k in ("n_target", "n_ok", "n_qa_fail", "n_qb_fail", "n_use", "ext_used",
                                             "ext_pass_rate", "stop", "stop_reasons", "i3_subset_relaxed")}
        st.rec["steps_done"].append("gate")
    if "tokens" in steps:
        qs = q_subset()
        if fs or qs:
            rc = st.run("i3 _tok tokens (subset)", [py, THIS, "_tok", "tokens", "--root", root], env)
        else:
            rc = st.run("tokens_lf tokens", [py, HERE / "tokens_lf.py", "tokens"], env)
        t = (read_json(res / "T_lf_check.json") or {}).get("tokens") or {}
        if rc == 3 and t:
            soft, hard = classify_T(t)
            if hard:
                st.rec["notes"].append(f"T STOP: {hard}")
                return st.finish(3)
            st.subset("T:elements", T_WHY, dict(relaxed=soft, elements=t.get("elements"), d_in_tokens_lf=t.get("d_in"),
                                                checks=t.get("checks")))
        elif rc:
            return st.finish(rc)
        st.rec["T"] = {k: t.get(k) for k in ("checks", "elements", "d_in", "n_reactions", "n_bad", "stop_reasons")}
        st.rec["steps_done"].append("tokens")
    st.rec["complete"] = set(FT_STEPS) <= set(st.rec["steps_done"])
    return st.finish(0)


def tok_overrides(FLm, TL, D):
    """(need_f_report_i3, q_check_i3) for the modules feat_lf (FLm), tokens_lf (TL), dep_common (D)."""
    orig_q = TL.q_check

    def need_f_report_i3():
        """= feat_lf.need_f_report (feat_lf.py 401-411) accepting a STOP whose only reasons are subset F-3."""
        if not (FLm.F_REPORT.is_file() and D.X182_LF.is_file()):
            D.not_ready(f"{FLm.F_REPORT} / {D.X182_LF} missing (feat_lf.py merge)")
        r = D.read_json(FLm.F_REPORT)
        if r.get("stop"):
            soft, hard = classify_F(r)
            if hard:
                D.stop(f"F_report.json STOP: {hard}")
            print(f"[I-3 subset] F_report STOP accepted (F-3 not judgeable on {r.get('n_done')} reactions): {soft}",
                  flush=True)
        if (r.get("x182") or {}).get("sha256") != D.sha256(D.X182_LF):
            D.not_ready(f"{D.X182_LF} is not the one of {FLm.F_REPORT} (rerun feat_lf.py merge)")
        return r

    def q_check_i3(qf):
        rec = orig_q(qf)
        if rec.get("stop"):
            soft, hard = classify_Q(dict(rec, gate_rc=None))
            if soft and not hard:
                print(f"[I-3 subset] Q rate rule relaxed on {rec.get('n_target')} reactions: {soft}", flush=True)
                rec = dict(rec, stop=False, stop_reasons=[], i3_subset_relaxed=soft)
        return rec

    return need_f_report_i3, q_check_i3


def child_tok(root, cmd):
    """tokens_lf <cmd> with feat_lf.need_f_report / tokens_lf.q_check relaxed for the subset rate rules only."""
    import tokens_lf as TL
    check_child(root)
    FLm, D = TL.FL, TL.D
    if FLm.F3_MAX != F3_MAX or TL.QA_RATE_MAX != QA_RATE_MAX or tuple(D.ELEMENTS) != tuple(C.ELEMENTS):
        C.stop("feat_lf.F3_MAX / tokens_lf.QA_RATE_MAX changed")
    FLm.need_f_report, TL.q_check = tok_overrides(FLm, TL, D)
    if cmd == "atoms":
        return TL.cmd_atoms(0, 1)
    if cmd == "gate":
        return TL.cmd_gate()
    if cmd == "tokens":
        return TL.cmd_tokens()
    C.stop(f"unknown _tok command {cmd}")


# ================================================================ predict
def cmd_predict(root):
    check_env("d1fu2")
    st = Step(root, "predict")
    if not (step_ok(root, "ft") and (read_json(root / "results" / "i3_steps" / "ft.json") or {}).get("complete")):
        C.not_ready("ft not done (all of slice, merge, atoms, gate, tokens)")
    res = root / "results"
    tok, x182, pairs, trep = root / "fu2_lf" / "tokens", root / "X182_lf.parquet", root / "i3_pairs.csv", \
        res / "fu2_lf" / "T_report.json"
    miss = [str(p) for p in (tok, x182, pairs, trep) if not p.exists()]
    if miss:
        C.not_ready(f"missing {miss}")
    env = i3_env(root)
    rc = st.run("predict_tf run --model final (I-3 inputs)",
                [sys.executable, HERE / "predict_tf.py", "run", tok, x182, pairs, res / "I3_predictions.parquet",
                 "--model", "final", "--t-report", trep], env)
    if rc:
        return st.finish(rc)
    rc2 = st.run("predict_tf run --model final (production lf inputs, same pairs)",
                 [sys.executable, HERE / "predict_tf.py", "run", DEP_PROD / "fu2_lf" / "tokens", DEP_PROD / "X182_lf.parquet",
                  pairs, res / "I3_predictions_lfprod.parquet", "--model", "final", "--t-report",
                  HERE / "results" / "fu2_lf" / "T_report.json"], env)
    if rc2:
        st.rec["notes"].append(f"reference prediction on the production lf inputs exit {rc2} (comparison only)")
    rep = read_json(res / "I3_predictions.report.json") or {}
    st.rec["predict"] = dict(pairs=rep.get("pairs"), n_reactions=rep.get("n_reactions"),
                             excluded_reactions=rep.get("excluded_reactions"), n_models=len(rep.get("models") or []),
                             thresholds_source=rep.get("thresholds_source"), seconds=rep.get("seconds"),
                             device=(rep.get("versions") or {}).get("gpu"))
    return st.finish(0)


# ================================================================ report
def read_xyz(path):
    """= pilot_common.read_xyz: (symbols, [n, 3] coordinates in A)."""
    import numpy as np
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.asarray(xyz, dtype=float)


def match_perm(a, b, tol=TOL_PERM_A):
    """= finalize_lf.match_perm (L-2): p with b[p[i]] == a[i] (same element, |dx| <= tol), a bijection."""
    import numpy as np
    sa, xa = list(a[0]), np.asarray(a[1], dtype=float)
    sb, xb = list(b[0]), np.asarray(b[1], dtype=float)
    if len(sa) != len(sb):
        return None, None, "n_atoms"
    if sorted(sa) != sorted(sb):
        return None, None, "elements"
    D = np.linalg.norm(xa[:, None] - xb[None], axis=-1)
    eb = np.array(sb)
    p = []
    for k, s in enumerate(sa):
        c = np.where((D[k] <= tol) & (eb == s))[0]
        if len(c) != 1:
            return None, float(D[k][eb == s].min()), f"atom_{k}_candidates_{len(c)}"
        p.append(int(c[0]))
    if len(set(p)) != len(p):
        return None, None, "not_bijective"
    return p, float(max(D[k, p[k]] for k in range(len(p)))), ""


def kabsch_rmsd(X, Y):
    """RMSD (A) of X onto Y after the optimal proper rotation + translation (= fu2 atoms.kabsch_rmsd)."""
    import numpy as np
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((R @ Xc.T).T - Yc) ** 2).sum(1).mean()))


def _fp(pairs):
    return sorted(tuple(sorted(int(x) for x in p)) for p in pairs or [])


def compare_rxn(root, T, rid, mode):
    """LADDER comparison of one reaction (module docstring 'report')."""
    import numpy as np
    new_d, lad_d, old_d, prod_d = root / "g2lf_final" / str(rid), C.G2_FINAL / str(rid), C.G2_JOBS / T, C.LF_FINAL / str(rid)
    nres = read_json(new_d / "result.json") or {}
    r = dict(job_id=T, rxn_id=rid, lf_marker=marker(new_d), lf_status=nres.get("status"),
             ladder_marker=marker(lad_d), production_lf_marker=marker(prod_d))
    if r["lf_marker"] != ".done":
        return r
    lres, pres = read_json(lad_d / "result.json") or {}, read_json(prod_d / "result.json") or {}
    files = dict(ts=("ts.xyz", "ts.xyz", "ts.xyz"), rel1=("rel1.xyz", "rel1.xyz", "ref_dipole.xyz"),
                 rel2=("rel2.xyz", "rel2.xyz", "ref_dipolarophile.xyz"))
    for k, (fn, fl, fo) in files.items():
        new, lad, old = read_xyz(new_d / fn), read_xyz(lad_d / fl), read_xyz(old_d / fo)
        p, dev, why = match_perm(new, lad)
        r[f"l2_{k}"] = "ok" if p is not None else why
        r[f"l2_{k}_dev_A"] = dev
        p0, dev0, why0 = match_perm(old, lad)
        r[f"ladder_jobs_vs_g2_final_{k}"] = "ok" if p0 is not None else why0
        same_order = list(new[0]) == list(old[0])
        r[f"{k}_same_element_order"] = same_order
        if same_order:
            heavy = [i for i, s in enumerate(new[0]) if s != "H"]
            r[f"{k}_rmsd_all_A"] = kabsch_rmsd(new[1], old[1])
            r[f"{k}_rmsd_heavy_A"] = kabsch_rmsd(new[1][heavy], old[1][heavy])
            r[f"{k}_max_abs_dx_A"] = float(np.abs(new[1] - old[1]).max())
    r["same_formed_pairs"] = _fp(nres.get("formed_pairs_ts")) == _fp(pres.get("formed_pairs_ts") or
                                                                     (read_json(old_d / "ts_result.json") or {}).get("formed_pairs_ts"))
    r["same_A_idx"] = sorted(nres.get("A_idx") or []) == sorted(pres.get("A_idx") or
                                                                (read_json(old_d / "ts_result.json") or {}).get("A_idx") or [])
    ni, li = nres.get("imag_freqs_cm") or [], lres.get("imag_freqs_cm") or []
    r["imag1_cm"], r["ladder_imag1_cm"] = (ni[0] if ni else None), (li[0] if li else None)
    r["d_imag1_cm"] = float(ni[0] - li[0]) if ni and li else None
    en, el = nres.get("e_ts_freq_eh"), lres.get("e_ts_freq_eh")
    r["d_e_ts_kcal"] = float((en - el) * H_KCAL) if en is not None and el is not None else None
    r["identical_L2"] = all(r[f"l2_{k}"] == "ok" for k in files)
    r["refound_ts"] = bool(r["l2_ts"] == "ok" or (r["ts_same_element_order"] and r["same_formed_pairs"] and r["same_A_idx"]
                                                   and r.get("ts_rmsd_heavy_A", 9e9) <= SAME_TS_RMSD_HEAVY))
    r["refs_same"] = all(r[f"l2_{k}"] == "ok" or r.get(f"{k}_rmsd_heavy_A", 9e9) <= SAME_TS_RMSD_HEAVY for k in ("rel1", "rel2"))
    if mode == "dry_run":                                     # byte parity with the production Phase L folder
        r["production_lf_same_bytes"] = {f: (new_d / f).is_file() and (prod_d / f).is_file() and
                                         C.sha256(new_d / f) == C.sha256(prod_d / f)
                                         for f in ("ts.xyz", "rel1.xyz", "rel2.xyz", "work/ts.hess", "work/ts.inp")}
    return r


def value_parity(root, rids):
    """X182 rows and token arrays of R vs the production lf inputs, per reaction (same atom order required)."""
    import numpy as np
    import pandas as pd
    out = {}
    X = pd.read_parquet(root / "X182_lf.parquet").set_index("rxn_id") if (root / "X182_lf.parquet").is_file() else None
    Xp = pd.read_parquet(DEP_PROD / "X182_lf.parquet")
    Xp = Xp[Xp.rxn_id.isin(rids)].set_index("rxn_id")
    for rid in rids:
        r = {}
        if X is not None and rid in X.index and rid in Xp.index:
            d = (X.loc[rid] - Xp.loc[rid, X.columns]).abs().astype(float)
            r.update(x182_max_abs=float(d.max()), x182_max_col=str(d.idxmax()), x182_n_cols_gt_1e6=int((d > 1e-6).sum()))
        else:
            r["x182"] = f"row lf_i3={X is not None and rid in X.index} production={rid in Xp.index}"
        a, b = root / "fu2_lf" / "tokens" / f"{rid}.npz", DEP_PROD / "fu2_lf" / "tokens" / f"{rid}.npz"
        if a.is_file() and b.is_file():
            ta, tb = np.load(a), np.load(b)
            if ta["Z"].tolist() == tb["Z"].tolist():
                r.update({f"token_{k}_max_abs": float(np.abs(ta[k].astype(float) - tb[k].astype(float)).max())
                          for k in ("num", "pos", "glob", "bin")})
                r["token_edges_equal"] = bool(np.array_equal(ta["edges"], tb["edges"]) and np.array_equal(ta["etype"], tb["etype"]))
            else:
                r["token"] = "atom order differs"
        else:
            r["token"] = f"file lf_i3={a.is_file()} production={b.is_file()}"
        out[int(rid)] = r
    return out


def formats(root, ids, rk, mode):
    """output format checks of every stage -> {name: dict(ok, ...)}."""
    import numpy as np
    import pandas as pd
    res = root / "results"
    f = {}
    # g2 work folders (g2_deploy keep_small) + finals (finalize_lf)
    need_done = ("ts.xyz", "ts_result.json", "ref_dipole.xyz", "ref_dipolarophile.xyz", "ref_result.json",
                 "stereo_result.json", ".done_ts", ".done_stereo", ".done_ref")
    jobs = {}
    for T in ids:
        jd, mk = root / "jobs" / T, marker(root / "g2lf_final" / str(rk[T]))
        miss = [x for x in ("deploy_times.json", "inputs.tar.gz") if not (jd / x).is_file()]
        if mk == ".done":
            miss += [x for x in need_done if not (jd / x).is_file()]
        elif mk is None:
            miss.append("g2lf_final marker")
        else:
            st_mk = [p.name for p in jd.glob(".fail_*")] + [p.name for p in jd.glob(".done_*")]
            if not st_mk:
                miss.append("stage markers")
        jobs[T] = miss
    f["g2_jobs"] = dict(ok=not any(jobs.values()), missing={k: v for k, v in jobs.items() if v})
    S = pd.read_csv(res / "L_status.csv") if (res / "L_status.csv").is_file() else None
    if S is not None:
        d = S[S.lf_marker == ".done"]
        f["g2lf_final"] = dict(ok=bool(len(S) == len(ids) and S.lf_marker.notna().all() and (d.done_complete == True).all()),  # noqa: E712
                               markers=S.lf_marker.fillna("missing").value_counts().to_dict(),
                               done_problems=d.loc[d.done_complete != True, ["rxn_id", "done_problems"]].to_dict("records"))  # noqa: E712
    else:
        f["g2lf_final"] = dict(ok=False, why="no L_status.csv")
    lab = read_json(root / "labels_lf.json")
    if lab is not None:
        n_done = sum(marker(root / "g2lf_final" / str(rk[T])) == ".done" for T in ids)
        f["labels_lf"] = dict(ok=bool(len(lab) == n_done and all(set(READ_KEYS) <= set(x) for x in lab)
                                      and all(isinstance(x[k], float) and math.isnan(x[k]) for x in lab for k in DFT_KEYS)),
                              n=len(lab), n_done=n_done)
        mt = pd.read_csv(root / "input_meta_lf.csv")
        f["input_meta_lf"] = dict(ok=list(mt.columns) == ["rxn_id", "A_idx"] and len(mt) == len(lab), n=len(mt))
    else:
        f["labels_lf"] = dict(ok=False, why="no labels_lf.json")
    if (root / "X182_lf.parquet").is_file():
        import pyarrow.parquet as pq
        X = pd.read_parquet(root / "X182_lf.parquet")
        ref = list(pq.read_schema(C.G2 / "X182_d1_g2.parquet").names)
        fr = read_json(res / "F_report.json") or {}
        f["X182_lf"] = dict(ok=bool(list(X.columns) == ref and len(ref) == 183 and np.isfinite(X[ref[1:]].to_numpy(float)).all()
                                    and len(X) == fr.get("n_feat")), n=len(X), columns_equal_X182_d1_g2=list(X.columns) == ref)
    else:
        f["X182_lf"] = dict(ok=False, why="no X182_lf.parquet")
    tdir = root / "fu2_lf" / "tokens"
    tm = res / "fu2_lf" / "T_manifest.csv"
    if tdir.is_dir() and tm.is_file():
        T_ = pd.read_csv(tm)
        bad = {}
        for rid in T_.rxn_id.astype(int):
            t = np.load(tdir / f"{rid}.npz")
            n = len(t["Z"])
            prob = [k for k in TOKEN_KEYS if k not in t.files]
            if not prob:
                if t["bin"].shape != (n, 6) or t["num"].shape != (n, 14) or t["pos"].shape != (n, 3) or t["glob"].shape != (182,):
                    prob.append(f"shapes bin {t['bin'].shape} num {t['num'].shape} pos {t['pos'].shape} glob {t['glob'].shape}")
                if str(t["token_version"][0]) != C.TOKEN_VERSION:
                    prob.append(f"token_version {t['token_version'][0]}")
                if int((t["etype"] == 2).sum()) != 2 or t["edges"].shape != (len(t["etype"]), 2):
                    prob.append("edges")
                if not set(int(z) for z in t["Z"]) <= set(C.ELEMENTS):
                    prob.append("elements")
                if int(t["rxn_id"]) != rid:
                    prob.append("rxn_id")
            if prob:
                bad[int(rid)] = prob
        f["tokens"] = dict(ok=not bad and len(T_) > 0, n=int(len(T_)), problems=bad)
    else:
        f["tokens"] = dict(ok=False, why="no tokens / T_manifest.csv")
    pp = res / "I3_predictions.parquet"
    if pp.is_file():
        O = pd.read_parquet(pp)
        rep = read_json(res / "I3_predictions.report.json") or {}
        ex = pd.read_csv(res / "I3_predictions.excluded.csv", dtype={"pair_id": str})
        from collections import Counter
        per = Counter(O.pair_id)
        tg = list(C.TARGETS)
        chk = dict(columns=list(O.columns) == OUT_COLS,
                   targets=all(sorted(g.target) == sorted(tg) for _, g in O.groupby("pair_id")),
                   rows=len(O) == 10 * len(per) and all(v == 10 for v in per.values()),
                   yhat_finite=bool(np.isfinite(O.yhat.to_numpy(float)).all()),
                   yhat_sd_ok=bool((np.isfinite(O.yhat_sd.to_numpy(float)) & (O.yhat_sd.to_numpy(float) >= 0)).all()),
                   out_of_range_bool=str(O.out_of_range.dtype) == "boolean" and not O.out_of_range.isna().any(),
                   ids_int=str(O.a_rxn.dtype).startswith("int") and str(O.b_rxn.dtype).startswith("int"),
                   excluded_columns=list(ex.columns) == ["pair_id", "a_rxn", "b_rxn", "reason"],
                   n_models_5=len(rep.get("models") or []) == 5 and rep.get("mode") == "final",
                   counts=(rep.get("pairs") or {}).get("n_predicted") == len(per) and
                          (rep.get("pairs") or {}).get("n_predicted", 0) + (rep.get("pairs") or {}).get("n_excluded", 0) == N_PAIRS)
        f["predictions"] = dict(ok=all(chk.values()) and len(per) > 0, checks=chk, n_pairs=len(per), n_rows=len(O),
                                n_excluded=len(ex), excluded=ex.to_dict("records"))
    else:
        f["predictions"] = dict(ok=False, why="no I3_predictions.parquet")
    return f


def cmd_report(root):
    import numpy as np
    import pandas as pd
    mode = run_mode(root)
    res = root / "results"
    steps = {p.stem: read_json(p) for p in sorted((res / "i3_steps").glob("*.json")) if not p.name.endswith(".abort.json")}
    aborts = {p.name: read_json(p) for p in sorted((res / "i3_steps").glob("*.abort.json"))}
    order = ["prep", "fake-g2" if mode == "dry_run" else "g2", "lm", "ft", "predict"]
    man, ids, P, rk = selection(root)
    out = dict(spec="docs/specs/DEPLOY_MODEL.md 14-3 I-3 (chain smoke)", code="analysis/deploy/i3_chain.py",
               code_sha256=C.sha256(THIS), git=git_head(), root=str(root), mode=mode, written=now(),
               steps={k: dict(status=(steps.get(k) or {}).get("status", "missing"), rc=(steps.get(k) or {}).get("rc"),
                              started=(steps.get(k) or {}).get("started"), wall_s=(steps.get(k) or {}).get("wall_s"),
                              slurm_job=(steps.get(k) or {}).get("slurm_job"),
                              commands=[(c["label"], c["rc"], c["wall_s"]) for c in (steps.get(k) or {}).get("commands", [])],
                              subset_exceptions=(steps.get(k) or {}).get("subset_exceptions", []),
                              notes=(steps.get(k) or {}).get("notes", []))
                      for k in order + sorted(set(steps) - set(order))},
               selection=(steps.get("prep") or {}).get("pairs"), cost_estimate=(steps.get("prep") or {}).get("cost_estimate"),
               aborted_attempts={k: dict(status=v.get("status"), ended=v.get("ended"), python=v.get("python"))
                                 for k, v in aborts.items() if v})
    g2r = (steps.get(order[1]) or {})
    rx = g2r.get("reactions") or {}
    ch = {T: (v.get("core_h") or {}) for T, v in rx.items()}
    out["cost_actual"] = dict(
        step=order[1], job_wall_h=(g2r.get("wall_s") or 0) / 3600, n_par=g2r.get("n_par"), cores=g2r.get("cores"),
        alloc_core_h=((g2r.get("wall_s") or 0) / 3600 * (g2r.get("n_par") or 0) * (g2r.get("cores") or 0)) or None,
        used_core_h_sum=float(sum(sum(v.values()) for v in ch.values())) if ch else None,
        used_core_h_by_stage={s: float(sum(v.get(s, 0.0) for v in ch.values())) for s in ("ts", "stereo", "ref", "finalize")},
        per_reaction={T: dict(core_h=v, marker=rx[T].get("marker")) for T, v in ch.items()},
        note="deploy_times.json core_h = stage wall x cores (g2_deploy.cmd_one); the dry run's fake-g2 has finalize only")
    rep = lambda p: read_json(res / p) or {}                               # noqa: E731
    L, M, F, Q, T, TC = rep("L_report.json"), rep("M_report.json"), rep("F_report.json"), rep("fu2_lf/Q_report.json"), \
        rep("fu2_lf/T_report.json"), rep("T_lf_check.json")
    PR = rep("I3_predictions.report.json")
    out["counts"] = dict(
        reactions=len(ids), pairs=len(P),
        g2_markers=(steps.get(order[1]) or {}).get("reactions") and
        dict(pd.Series([v["marker"] for v in steps[order[1]]["reactions"].values()]).fillna("missing").value_counts()),
        L=dict(complete=L.get("complete"), all_pass=L.get("all_pass"), markers=L.get("lf_markers"),
               L1=(L.get("L1") or {}).get("pass"), L2=(L.get("L2") or {}).get("pass"), L4=(L.get("L4") or {}).get("pass"),
               L4_d_imag_cm=(L.get("L4") or {}).get("d_imag_cm")),
        M=dict(n_records=M.get("n_records"), all_pass=M.get("all_pass"), path=(steps.get("lm") or {}).get("M_path"),
               n_source_fallback=(M.get("i3_override") or {}).get("n_source_fallback")),
        F=dict(n_done=F.get("n_done"), n_feat=F.get("n_feat"), F1=(F.get("x182") or {}).get("columns_equal_X182_d1_g2"),
               F2_n=(F.get("F2") or {}).get("n"), F3_n=(F.get("F3") or {}).get("n"), not_ok=F.get("not_ok"),
               stop_reasons=F.get("stop_reasons"), core_s_per_rxn=(F.get("F4") or {}).get("total_core_s_per_rxn")),
        Q=dict(n_target=Q.get("n_target"), n_ok=Q.get("n_ok"), n_qa_fail=Q.get("n_qa_fail"), n_qb_fail=Q.get("n_qb_fail"),
               n_use=Q.get("n_use"), ext_used=Q.get("ext_used"), stop=Q.get("stop")),
        T=dict(n_reactions=T.get("n_reactions"), elements=T.get("elements"), num_cols=T.get("num_cols"),
               checks=(TC.get("tokens") or {}).get("checks"), gate_Ta=T.get("gate_Ta")),
        predict=dict(pairs=PR.get("pairs"), n_reactions=PR.get("n_reactions"), excluded=PR.get("excluded_reactions")))
    out["formats"] = formats(root, ids, rk, mode)
    # ---- LADDER comparison
    cmpr = [compare_rxn(root, T, rk[T], mode) for T in ids]
    done = [c for c in cmpr if c["lf_marker"] == ".done"]
    out["ladder_comparison"] = dict(
        rule=f"identical = L-2 (ts, rel1, rel2 atom permutations of $G2/g2_final within {TOL_PERM_A} A); re-found TS = "
             f"identical or (same element order as $G2/jobs/<T>/ts.xyz, same formed pairs, same A_idx, heavy-atom Kabsch "
             f"RMSD <= {SAME_TS_RMSD_HEAVY} A); $G2/jobs/<T> = $G2/g2_final up to the permutation (checked per reaction)",
        n_done=len(done), n_identical=sum(c["identical_L2"] for c in done), n_refound_ts=sum(c["refound_ts"] for c in done),
        n_refs_same=sum(c["refs_same"] for c in done),
        ladder_jobs_eq_g2_final=all(c.get(f"ladder_jobs_vs_g2_final_{k}") == "ok" for c in done for k in ("ts", "rel1", "rel2")),
        reactions=cmpr)
    refound = [c["rxn_id"] for c in done if c["refound_ts"]]
    out["value_parity_production_lf"] = dict(
        note="re-found TSs only: R X182 / tokens vs the production lf inputs of the same reaction ($DEP/X182_lf.parquet, "
             "$DEP/fu2_lf/tokens); for a TS found again by a new autodE run small differences are expected",
        reactions=value_parity(root, refound) if refound else {})
    lab = read_json(root / "labels_lf.json") or []
    if lab:                                      # parent process: C.LABELS_LF / C.META_LF = the production Phase M files
        prod = {int(d["rxn_id"]): d for d in read_json(C.LABELS_LF) or []}
        pm = pd.read_csv(C.META_LF, dtype={"A_idx": str}).set_index("rxn_id").A_idx.to_dict()
        im = pd.read_csv(root / "input_meta_lf.csv", dtype={"A_idx": str}).set_index("rxn_id").A_idx.to_dict()
        out["value_parity_production_lf"]["labels_record_equal"] = {
            str(d["rxn_id"]): json.dumps(d, sort_keys=True) == json.dumps(prod.get(int(d["rxn_id"])), sort_keys=True)
            for d in lab}
        out["value_parity_production_lf"]["meta_A_idx_equal"] = {str(k): pm.get(k) == v for k, v in im.items()}
    pp, pq_ = res / "I3_predictions.parquet", res / "I3_predictions_lfprod.parquet"
    if pp.is_file():
        O = pd.read_parquet(pp)
        rows = []
        for pid, g in O.groupby("pair_id"):
            a, b = int(g.a_rxn.iloc[0]), int(g.b_rxn.iloc[0])
            ca = next(c for c in cmpr if c["rxn_id"] == a)
            cb = next(c for c in cmpr if c["rxn_id"] == b)
            rows.append(dict(pair_id=pid, a_rxn=a, b_rxn=b, both_refound=bool(ca.get("refound_ts") and cb.get("refound_ts")),
                             yhat={t: float(v) for t, v in zip(g.target, g.yhat)},
                             yhat_sd={t: float(v) for t, v in zip(g.target, g.yhat_sd)},
                             n_out_of_range=int(g.out_of_range.fillna(False).sum())))
        if pq_.is_file():
            R0 = pd.read_parquet(pq_)
            m = O.merge(R0[["pair_id", "target", "yhat"]], on=["pair_id", "target"], how="left", suffixes=("", "_prod"))
            for r in rows:
                g = m[m.pair_id == r["pair_id"]]
                d = (g.yhat - g.yhat_prod).abs()
                r["max_abs_dyhat_vs_production_lf"] = None if d.isna().all() else float(d.max())
        out["pairs"] = rows
    # ---- verdict
    hard = []
    for k in order:
        s = (steps.get(k) or {}).get("status", "missing")
        if s not in ("ok", "ok_subset"):
            hard.append(f"step {k}: {s}")
    bad_fmt = [k for k, v in out["formats"].items() if not v.get("ok")]
    hard += [f"format {k}" for k in bad_fmt]
    if mode == "dry_run":
        if out["ladder_comparison"]["n_identical"] != len(done):
            hard.append("dry run: not every structure identical to $G2/g2_final")
        nb = [c["rxn_id"] for c in done if not all((c.get("production_lf_same_bytes") or {}).values())]
        out["dry_run_production_lf_bytes_differ"] = nb
    out["subset_exceptions"] = [dict(step=k, **x) for k in order for x in (steps.get(k) or {}).get("subset_exceptions", [])]
    out["i3_pass"] = not hard and bool(out.get("pairs"))
    out["i3_fail_reasons"] = hard if hard else ([] if out.get("pairs") else ["no pair predicted"])
    out["all_5_pairs_predicted"] = len(out.get("pairs") or []) == N_PAIRS
    out["decisions"] = [
        "selection = g2_deploy.py i3-prep unchanged (seed 20261019), file names i3_manifest / i3_list / i3_pairs.csv",
        "g2: pool 10 x 4 cores in one job (D1-G2: 4 cores per reaction); TS templates = g2_deploy default copy",
        "subset gates: L-1/L-2/L-4 (real run), F-3, Q-a rate, T elements as in the module docstring; others STOP",
        "M: i3 override when result.json source_dir is gone (g2_deploy node-local work folder) - keep_small copy + sha256",
        f"re-found TS rule: heavy-atom Kabsch RMSD <= {SAME_TS_RMSD_HEAVY} A + same formed pairs + same A_idx",
        "predict: env_d1fu2 on CPU; reference prediction on the production lf inputs for the same pairs"]
    name = "I3_chain.json" if root.name == DEFAULT_ROOT.name else "I3_chain_dry.json"
    C.write_json(res / "I3_chain.json", out)
    C.write_json(HERE / "results" / name, out)
    print(json.dumps(dict(mode=mode, i3_pass=out["i3_pass"], fail=out["i3_fail_reasons"], steps={k: v["status"] for k, v in
                          out["steps"].items()}, formats={k: v.get("ok") for k, v in out["formats"].items()},
                          ladder={k: v for k, v in out["ladder_comparison"].items() if k != "reactions"},
                          subset=[(x["step"], x["gate"]) for x in out["subset_exceptions"]],
                          written=[str(res / "I3_chain.json"), str(HERE / "results" / name)]), indent=1, default=str))
    return 0 if out["i3_pass"] else 3


# ================================================================ selftest
def cmd_selftest(root):
    import numpy as np
    ok = {}
    L = dict(complete=True, subset=["T1"], all_pass=False, L1={"pass": True, "n_done_incomplete": 0},
             L2={"pass": False}, L4={"pass": False})
    c = classify_L(L, "real")
    ok["L_real_soft"] = c["soft"] == ["L2", "L4"] and not c["hard"]
    c = classify_L(L, "dry_run")
    ok["L_dry_hard"] = len(c["hard"]) == 2 and not c["soft"]
    c = classify_L(dict(L, L1={"pass": False, "n_done_incomplete": 1, "done_incomplete": [1]}), "real")
    ok["L_format_hard"] = any("missing files" in h for h in c["hard"])
    ok["L_not_subset_hard"] = bool(classify_L(dict(L, subset=None), "real")["hard"])
    ok["L_pass_clean"] = classify_L(dict(L, all_pass=True, L2={"pass": True}, L4={"pass": True}), "dry_run") == \
        dict(failed=[], soft=[], hard=[], why={})
    s, h = classify_F(dict(n_done=10, stop_reasons=["F-3: 1 / 10 = 0.1000 feature failures (other than redetection) > 0.02"]))
    ok["F3_subset_soft"] = len(s) == 1 and not h
    s, h = classify_F(dict(n_done=100, stop_reasons=["F-3: 3 / 100 = 0.0300 ..."]))
    ok["F3_judgeable_hard"] = not s and len(h) == 1
    s, h = classify_F(dict(n_done=10, stop_reasons=["F-1: X182_lf columns != ...", "F-3: 1 / 10 ..."]))
    ok["F1_hard"] = len(h) == 1 and h[0].startswith("F-1") and len(s) == 1
    s, h = classify_Q(dict(n_target=10, stop_reasons=["Q-a failures 1 / 10 > 0.005"], gate_rc=0))
    ok["Q_rate_soft"] = len(s) == 1 and not h
    s, h = classify_Q(dict(n_target=10, stop_reasons=["Q_report stop: ['Q-b: 1 ...']", "Q-a failures 1 / 10 > 0.005"],
                           gate_rc=3))
    ok["Q_fu2_hard"] = len(h) >= 1
    s, h = classify_Q(dict(n_target=300, stop_reasons=["Q-a failures 2 / 300 > 0.005"], gate_rc=0))
    ok["Q_judgeable_hard"] = not s and len(h) == 1
    chk = dict(elements=False, bin_cols=True, num_cols=True, ext_used=True, d_in=False, glob_cols=True,
               token_version=True, x182_sha256=True, gate_Ta=True)
    s, h = classify_T(dict(checks=chk, elements=[1, 6, 7, 8], d_in=24, tokens_rc=0))
    ok["T_elements_din_soft"] = s == ["elements", "d_in"] and not h
    s, h = classify_T(dict(checks=dict(chk, num_cols=False), elements=[1, 6, 7, 8], d_in=21, tokens_rc=0))
    ok["T_num_hard"] = s == ["elements"] and sorted(h) == ["T check d_in", "T check num_cols"]
    s, h = classify_T(dict(checks=chk, elements=[1, 6, 53], d_in=23, tokens_rc=0))
    ok["T_iodine_hard"] = not s and sorted(h) == ["T check d_in", "T check elements"]
    s, h = classify_T(dict(checks=dict(chk, elements=True, d_in=True), elements=list(C.ELEMENTS), d_in=29, tokens_rc=0))
    ok["T_all_pass"] = not s and not h
    rng = np.random.default_rng(C.SEED)
    syms = ["C", "C", "N", "O", "H", "H", "H"]
    X = rng.normal(size=(7, 3)) * 2
    perm = rng.permutation(7)
    b = ([syms[i] for i in perm], X[perm])
    p, dev, why = match_perm((syms, X), b)
    ok["perm_found"] = p is not None and all(perm[p[i]] == i for i in range(7)) and dev == 0.0
    p, dev, why = match_perm((syms, X + 1e-5), b)
    ok["perm_tol"] = p is None
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    ok["kabsch_rot"] = kabsch_rmsd(X @ Rz.T + 3.0, X) < 1e-10
    ok["kabsch_mirror_not_aligned"] = kabsch_rmsd(X * np.array([1, 1, -1]), X) > 1e-3       # proper rotations only
    env = i3_env(DEP_PROD / "i3_dev", base=dict(PATH="/bin", DEP_G2="/x", DEP_JOB_T0="1", DEP_SCRATCH="/y"))
    ok["env"] = (env["DEP_SCRATCH"] == str(DEP_PROD / "i3_dev") and "DEP_G2" not in env and env["DEP_JOB_T0"] == "1"
                 and env["DEP_MODEL_DIR"] == str(DEP_PROD / "model") and env["DEP_RESULTS"] == str(DEP_PROD / "i3_dev" / "results"))
    # _tok overrides on stub modules (relax only the subset rate rules)
    import types
    tmp = Path(root) / "results" / "_selftest_tmp"
    C.assert_write_ok(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    x, fr, qf = tmp / "x182.bin", tmp / "F_report.json", tmp / "Q.json"
    x.write_bytes(b"x182")
    Ds = types.SimpleNamespace(X182_LF=x, read_json=C.read_json, stop=C.stop, not_ready=C.not_ready, sha256=C.sha256)
    nf, qc = tok_overrides(types.SimpleNamespace(F_REPORT=fr), types.SimpleNamespace(q_check=C.read_json), Ds)
    print("[selftest] _tok overrides: the next 'STOP: F_report.json STOP' and 'NOT READY' lines are expected", flush=True)

    def code_of(f):
        try:
            f()
            return 0
        except C.Stop as e:
            return e.code
    good_x = dict(sha256=C.sha256(x))
    C.write_json(fr, dict(stop=True, n_done=10, stop_reasons=["F-3: 1 / 10 = 0.1000 feature failures"], x182=good_x))
    ok["tok_F3_subset_accepted"] = code_of(nf) == 0
    C.write_json(fr, dict(stop=True, n_done=10, stop_reasons=["F-1: X182_lf columns != ..."], x182=good_x))
    ok["tok_F1_stop"] = code_of(nf) == 3
    C.write_json(fr, dict(stop=False, n_done=10, stop_reasons=[], x182=dict(sha256="0" * 64)))
    ok["tok_F_sha_not_ready"] = code_of(nf) == 2
    C.write_json(qf, dict(stop=True, n_target=10, stop_reasons=["Q-a failures 1 / 10 > 0.005"]))
    r = qc(qf)
    ok["tok_Q_rate_relaxed"] = r["stop"] is False and r["i3_subset_relaxed"] == ["Q-a failures 1 / 10 > 0.005"]
    C.write_json(qf, dict(stop=True, n_target=10, stop_reasons=["Q_report stop: ['Q-b: 1']", "Q-a failures 1 / 10 > 0.005"]))
    ok["tok_Q_fu2_kept"] = qc(qf)["stop"] is True
    C.write_json(qf, dict(stop=True, n_target=400, stop_reasons=["Q-a failures 3 / 400 > 0.005"]))
    ok["tok_Q_judgeable_kept"] = qc(qf)["stop"] is True
    shutil.rmtree(tmp)                                                # this test's own folder
    sd = Path(root) / "results" / "i3_steps"
    n1, n2 = "_selftest_abort", "_selftest_abort2"
    st = Step(root, n1)
    st.rec["commands"].append(dict(label="x", argv=[], rc=0, wall_s=0.0))
    _record_abort(root, n1, 2, None)
    r1 = read_json(sd / f"{n1}.json") or {}
    _CUR.clear()
    _record_abort(root, n2, 3, "tb")
    r2 = read_json(sd / f"{n2}.abort.json") or {}
    ok["abort_mid_step"] = r1.get("status") == "not_ready" and r1.get("rc") == 2 and len(r1.get("commands") or []) == 1
    ok["abort_pre_step"] = r2.get("status") == "stop" and not (sd / f"{n2}.json").exists()
    for p in (sd / f"{n1}.json", sd / f"{n2}.abort.json"):           # this test's own files
        if p.is_file():
            p.unlink()
    print("[selftest] root guard: the next 4 'STOP: --root ...' lines are expected", flush=True)
    bad = []
    for r in (DEP_PROD, DEP_PROD / "g2lf_final", DEP_PROD / "x3", Path("/tmp/i3")):
        try:
            check_root(r)
            bad.append(str(r))
        except C.Stop:
            pass
    ok["root_guard"] = not bad
    rep = dict(written=now(), checks=ok, all_pass=all(ok.values()), code_sha256=C.sha256(THIS))
    C.write_json(Path(root) / "results" / "I3_selftest.json", rep)
    print(json.dumps(rep, indent=1))
    return 0 if rep["all_pass"] else 3


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="i3_chain.py", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("prep", "g2", "fake-g2", "lm", "ft", "predict", "report", "selftest",
                                    "_fake_g2", "_meta", "_tok"))
    ap.add_argument("sub", nargs="?", default=None, help="_tok: atoms | gate | tokens")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n-par", type=int, default=10)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--expect-final", action="store_true", help="g2: STOP instead of starting autodE for a reaction "
                                                                "without a final marker (wiring test)")
    ap.add_argument("--steps", default=",".join(FT_STEPS))
    ap.add_argument("--force-tok", action="store_true", help="ft: run atoms / gate / tokens through _tok even without a "
                                                             "subset exception (dry-run test of the override path)")
    a = ap.parse_args(argv)
    if (a.sub is not None) != (a.cmd == "_tok"):
        ap.error("the second positional argument is the _tok command only")
    root = check_root(a.root)
    if a.cmd.startswith("_"):
        if a.cmd == "_fake_g2":
            return child_fake_g2(root, a.cores)
        if a.cmd == "_meta":
            return child_meta(root)
        return child_tok(root, a.sub)
    check_parent()
    if a.cmd == "selftest":
        return cmd_selftest(root)
    try:
        if a.cmd == "prep":
            return cmd_prep(root, a.n_par, a.cores)
        if a.cmd == "g2":
            return cmd_g2(root, a.n_par, a.cores, a.expect_final)
        if a.cmd == "fake-g2":
            return cmd_fake_g2(root, a.cores)
        if a.cmd == "lm":
            return cmd_lm(root)
        if a.cmd == "ft":
            steps = [s for s in a.steps.split(",") if s]
            if not set(steps) <= set(FT_STEPS):
                C.stop(f"--steps {steps}: subset of {FT_STEPS}")
            return cmd_ft(root, steps, a.force_tok)
        if a.cmd == "predict":
            return cmd_predict(root)
        if a.cmd == "report":
            return cmd_report(root)
    except SystemExit as e:                                           # dep_common STOP (3) / not ready (2)
        _record_abort(root, a.cmd, e.code if isinstance(e.code, int) else 3, None)
        raise
    except Exception:                                                 # noqa: BLE001
        _record_abort(root, a.cmd, 3, traceback.format_exc())
        raise
    return 2


def _record_abort(root, cmd, code, tb):
    """a parent command that left through STOP / not ready / an exception before Step.finish: finish its record
    <cmd>.json (the commands run so far are kept); a failure before the record existed (wrong job env) goes to
    <cmd>.abort.json and never replaces the record of an earlier run."""
    if cmd == "report":
        return
    st = _CUR.get("step")
    try:
        if st is not None and not st.done:
            st.rec["aborted"] = dict(code=code, traceback=tb)
            st.finish(code)
        elif st is None:
            C.write_json(Path(root) / "results" / "i3_steps" / f"{cmd}.abort.json",
                         dict(step=cmd, root=str(root), status={2: "not_ready"}.get(code, "stop"), rc=code, ended=now(),
                              aborted=dict(code=code, traceback=tb), host=socket.gethostname(), python=sys.executable,
                              slurm_job=os.environ.get("SLURM_JOB_ID")))
    except Exception:                                                 # noqa: BLE001
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
