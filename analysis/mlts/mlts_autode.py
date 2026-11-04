#!/usr/bin/env python3
r"""mlts_autode.py — DEPLOY-V2 (MLTS) Phase A (docs/specs/DEPLOY_V2_MLTS.md 6 A-2, A-3, A-4): autodE-MLTS, modelled on
analysis/d1_g2/g2_autode.py. Compute nodes only (jobs/mlt.slurm or jobs/sbx.sh); every sub-command re-executes itself
in the environment it needs (ensure_env), so any MLTENV can start it.

What is the D1 code, unchanged (imported, never copied; $PILOT = d1_env/EDA-ASM/D1_build/d1_autode_pilot):
  run_ts.run_autode / analyse / forming_share, stereo_check.py (run as the D1 script), make_reference.coley_port /
  dipolarophile_check, pilot_common (configure_autode called as is, markers, xyz io), orca_direct, d1_run/orca_parse.py
  --infra; for A'' analysis/d1_g3 opt.d1() / g3l.g3l_text / g3l.inhess_evidence (the G3L input).
What changes, in the autodE worker process only (decisions A2-* below; autodE files untouched):
  * the ORCA keyword sets (after pilot_common.configure_autode): every hmethod calculation is '! <runtype> ExtOpt' + the
    '%method ProgExt' block of mlts_bridge.ext_method_block (no functional / basis / RIJCOSX / def2/J / dispersion);
    Opt and OptTS add TightOpt; low_opt stays LooseOpt + MaxIter 10; Freq -> EnGrad + the MLIP analytic Hessian
    written as <calc>.hess (A2-freq; --freq numfreq = ORCA NumFreq through ExtOpt, measurement only);
  * ORCA.add_solvent_keyword / print_solvent are no-ops (the solvent is in the MLTS PES); the reaction keeps
    solvent_name='water' (xTB conformers and templates as in D1);
  * ORCA.execute: OptTS calcs run the A-1 scheme (InHess Read = MLIP analytic Hessian at the start, restarted with a
    new MLIP Hessian every 30 cycles, 100 cycles in total, Trust -0.1, D1 modify_internal blocks; DECISIONS A-1);
  * ORCA.generate_input_for writes ExtOpt inputs serial (no %pal; autodE n_cores still drives the xTB conformer pool);
  * ORCA._energy_from / ORCAOptimiser.last_energy_change read the last token of 'FINAL SINGLE POINT ENERGY' (ExtOpt
    prints 'FINAL SINGLE POINT ENERGY (From external program) <E>'; autodE 1.4.5 reads token 4 = '(From');
  * reference step: make_reference.dft_refine -> mlts_refine (MLTS Opt TightOpt + MLIP Hessian G, orig_energy=None
    so the autodE dipole and the alt conformer go through the same refine, as G2); Coley rule and thresholds unchanged;
  * TS template folder = $MLT/ts_templates (D1 / G2 template folders are neither read nor written).

Engines (arms): A = autodE-MLTS (above); Ap = the G2 TS ($G2/jobs/<T>/ts.xyz, G2 ok = $G2/g2_final/<rxn>/.done) ->
mlts_bridge.optimize_ts (A-1), G2's reactants MLTS-optimised, then run_ts.analyse, stereo and the same reference step;
App = A's MLTS TS -> label-level ORCA OptTS (g3l.g3l_text: the D1 L0 OptTS + Freq, InHess Read of A's ts.hess = the
MLTS analytic Hessian at that TS, no Recalc_Hess, 8 MPI ranks), then run_ts.analyse with the DFT frequencies and the
D1 stereo check; A's (MLTS) references are kept.

Layout (A-2; contract): <out_root>/jobs/<job_id>/ = ts.xyz, ts_result.json (formed_pairs_ts, A_idx, B_idx, checks incl.
flag_async / no_foreign_bond, charge_dipole / charge_dipolarophile), ts.hess (MLIP analytic Hessian at ts.xyz; App:
DFT Freq), r_dipole_autode.xyz, r_dipolarophile_autode.xyz, ref_dipole.xyz, ref_dipolarophile.xyz, ref_result.json
(alt_used), stereo_result.json, stereo.log, energies.csv / methods.txt / product.xyz (A), timing.json {wall_s, core_h,
gpu_s, n_ext_calls, ...}, inputs.tar.gz (every .inp of the reaction), .done_ts/.done_stereo/.done_ref or .fail_<stage>.
The folder is published atomically (os.replace) from the node-local work folder ${TMPDIR:-/tmp}/mlts_<jobtag>_<job_id>
BEFORE that folder is deleted (DECISIONS: no repeat of the g2_deploy $TMPDIR bug). A published folder = final (skip).
Roots: main engine $MLT/ade (inputs_mlts default R/ade/jobs, R = $MLT); A when A'' is the main engine $MLT/ade_pre;
A-3 arms $MLT/a3/<A|Ap|App>/ade; A-4 A'' arm (--root R) R/ade; smoke $MLT/a2smoke/<model>/ade. MLTS_A_BASE moves
every root (tests: $MLT/tmp/atest), reports then go to <base>/reports instead of analysis/mlts/results.

CLI (from analysis/mlts; exit 0 done, 2 incomplete / not ready (same command again), 3 STOP). i n = slice i of n ('@I').
  run <list> [i n] [--model M] [--engine A|Ap|App|A,Ap] [--list csv] [--out R] [--workers w] [--cores c] [--freq F]
        GPU job: one MLTS server per model (mlts_bridge, lazily, node-local socket) + w autodE workers (D1 env
        subprocesses, c cores each). <list>: a3 (default arms A,Ap; --list or $MLT/pipeline/lists/a3.csv; model
        FT_<fold>_al), d1:<k> (D1 targets of T1 fold k; model FT_<k>_al, fold -1 FT_0_al), test:<T..,rxn..>, ids:<csv>.
        --engine App runs A into the A'' source root.                                                       [mlts]
  refine_dft <list> [i n] [--root R] [--src S] [--list csv]
        A'' for the list (CPU, $SLURM_CPUS_PER_TASK // 8 ORCA runs of 8 ranks): a3 -> $MLT/a3/App/ade from
        $MLT/a3/A/ade; d1:<k> without --root -> $MLT/ade from $MLT/ade_pre (A'' main); with --root R -> R/ade from
        $MLT/ade (A'' extra arm).                                                                             [d1]
  a3_report [--list csv]        -> results/A3_report.json {complete, arms{A,Ap,App}{...}, rule_inputs, est}     [d1]
  status [--root R]             -> results/A4_report.json (R/reports/ with --root) {complete, folds{k: ...}}  [any]
  smoke --model M [--n N]       -> results/A2_smoke.json {pass, inp_checks (C-3 wording)}                       [mlts]
  one <job_id> --model SPEC --engine A|Ap --out R --cores c --socket S [--tag t] [--freq F]   (internal worker) [d1]
Python API (contract): run_one(rxn_smiles, job_id, model, engine, out_root, cores, socket=None, job_row=None, ...)
  -> Path(<out_root>/jobs/<job_id>) in the A-2 layout (autodE is process-global: one reaction per process).

Decisions (for results/DECISIONS.md; also in every slice summary as DECISIONS):"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                  # nothing compiled into the read-only code trees

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket as _socket  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)

CODE_VERSION = "mlts-a-1"
DECISIONS = {
    "A2-keywords": "autodE-MLTS keyword sets (after pilot_common.configure_autode, this process only): low_opt "
                   "'LooseOpt ExtOpt' + MaxIter 10, opt 'Opt TightOpt ExtOpt', opt_ts 'OptTS TightOpt ExtOpt' + "
                   "%geom InHess Read / InHessName \"mlts_inhess.hess\" / Trust -0.1 / MaxIter 30, grad 'EnGrad ExtOpt', "
                   "hess 'EnGrad ExtOpt' (+ MLIP Hessian file), sp / low_sp 'SP ExtOpt'; every set carries the "
                   "mlts_bridge.ext_method_block %method block (socket + per-reaction call log). No functional, basis, "
                   "RI, dispersion or solvent word; CPCM/SMD writers are no-ops; solvent_name='water' kept (xTB).",
    "A2-optts": "OptTS = DECISIONS A-1: initial Hessian = MLIP analytic Hessian (InHess Read), a new MLIP Hessian and an "
                "ORCA restart every 30 cycles (Recalc analytic:30), 100 cycles in total, Trust -0.1, the D1 forming-bond "
                "modify_internal blocks (autodE added_internals), TightOpt. The first segment is autodE's own input; "
                "later segments = the same text with the current geometry and MaxIter = cycles left (<= 30); "
                "<calc>.out = the segment outputs concatenated, <calc>.xyz = the final geometry, <calc>.hess = the MLIP "
                "Hessian at the final geometry (autodE reads the TS modes from it, as from the D1 'OptTS Freq').",
    "A2-freq": "Freq route = MLIP analytic Hessian written where autodE reads it (<calc>.hess; the ORCA run of a hess "
               "calc is one 'EnGrad ExtOpt' call for energy / termination) instead of ORCA NumFreq through ExtOpt. "
               "Measured per Hessian (MACE-OMOL-0, shared RTX 3090, tests/a_freq_bench.py): 30-atom label TS analytic "
               "12.1 GPU-s through the server vs NumFreq 19.7 GPU-s + 96 s wall (176 calls); 8 atoms 4.8 vs 4.9 GPU-s "
               "+ 44 s wall; same imaginary frequency (0.01 cm-1). Per reaction (30-31 atoms) 8-12 Hessians = 86-88 % "
               "of the GPU time, so NumFreq would add ~50 GPU-s and ~10 min wall. --freq numfreq kept (measurement only).",
    "A2-serial": "ExtOpt inputs are written with n_cores = 1 (no %pal block: no MPI start for E/G calls, one GPU "
                 "server call per step); autodE Config.n_cores = the worker's cores for the xTB conformer pool.",
    "A2-energy": "ORCA ExtOpt prints 'FINAL SINGLE POINT ENERGY (From external program) <E>'; autodE 1.4.5 reads "
                 "line.split()[4] (= '(From') -> ValueError. ORCA._energy_from and ORCAOptimiser.last_energy_change "
                 "read the last token instead (identical for plain ORCA lines); in-process only.",
    "A2-ref": "reference step = make_reference.coley_port with dft_refine -> mlts_refine (MLTS 'Opt TightOpt ExtOpt' "
              "+ MLIP Hessian thermo at 298.15 K; E = MLTS energy at the optimised geometry, no separate SP) for both "
              "the autodE dipole (orig_energy=None, G2 style) and the alt conformer; Coley rule and thresholds "
              "(RMSD > 0.05 A, dG < -0.239 kcal/mol) unchanged; the dipolarophile check unchanged.",
    "A2-config": "base config = $D1RUN/config_run.yaml (the D1 label flow: gates, OptTS Trust/MaxIter, maxcore 3000, "
                 "xtb 6.7.1) read only, with scratch -> the node-local work folder; TS templates "
                 "$MLT/ts_templates/<model> (A2-templates).",
    "A2-local": "node-local work folder ${TMPDIR:-/tmp}/mlts_<jobtag>_<job_id> (autodE temp dirs and xtb scratch "
                "inside it via TMPDIR); the A-2 folder (everything listed in DECISIONS Phase H + timing.json + "
                "inputs.tar.gz of every .inp) is published atomically BEFORE the work folder is deleted; a worker "
                "killed by the per-reaction timeout is published by the parent with .fail_ts 'timeout_worker'.",
    "A2-retry": "one infra retry of the ts stage (A and A'): on any StageFail of attempt 1 with infra evidence = the "
                "MLTS server not answering or its error count rising during the attempt, plus for an 'autodE "
                "exception' the D1 rule (d1_run/orca_parse.py --infra); the server process is restarted by the GPU "
                "job's parent (<= 3 restarts per model; a model whose server keeps dying stops the slice with exit 1 = "
                "infra retry, not a STOP).",
    "A2-crash": "a non-StageFail exception in the ts stage is retried once in place, then closed as .fail_ts "
                "'pipeline_error: ...'; a worker that exits without a published folder (not killed by a signal) is "
                "counted per (out_root, reaction, arm) in $MLT/a/crashes/ and the 2nd such exit is published by the "
                "parent as .fail_<first unfinished stage> 'pipeline_error' (a reaction never keeps a slice at exit 2).",
    "A2-stereo": "a D1 stereo_check.py crash (no .done_stereo / .fail_stereo) is closed as .fail_stereo 'stereo_check "
                 "rc N: no marker (crash)' so A-3 / A-4 never wait on a half-finished folder.",
    "A2-charge": "every MLIP Hessian is computed with the molecule's charge (autodE calc.molecule.charge; A' the "
                 "manifest charge); D1 is neutral, the P predictor may not be.",
    "A2-templates": "TS templates in $MLT/ts_templates/<model> (one folder per MLTS model: no template of one fold model "
                    "guides another fold's search; within a model results still depend on run order, as in D1 / G2).",
    "A2-one-server": "a GPU job runs one MLTS server at a time (MLTS_ONE_SERVER=1): tasks are model-major and a task "
                     "of the next model waits until the running tasks of the current model end and its server is "
                     "stopped (two OMOL-size servers can exhaust a 24 GB card in a Hessian).",
    "A2-term": "SIGTERM (wall / scancel) stops the workers from claiming new reactions before the job exits.",
    "A2-run-one": "run_one without a socket starts a server on a socket path unique to the call (pid + random), so "
                  "concurrent calls never attach to or stop each other's server.",
    "A2-workers": "one GPU job = one server per model (lazily started, stopped when its reactions are done) + w worker "
                  "subprocesses of c cores. c = 2 (measured: ~60 % of a reaction's wall is serial ORCA ExtOpt steps and "
                  "GPU Hessian waits; the xTB pool is the only parallel part); w = min((cpus - 1) // c, A-3 "
                  "est.workers) with est.workers = median over the A-3 slices of 0.7 x (reaction wall / reaction GPU "
                  "time), 6 before A-3 (zero-shot MACE-OMOL-0 on the 3090: wall / GPU 3.2-3.6 -> 2-3 workers).",
    "A2-cost": "per reaction: own core_h = worker wall x worker cores; own gpu_s = server evaluation time of the "
               "reaction's E/G calls (client log) + its Hessians; ALLOCATED (the A-3 / budget basis) = the parent's "
               "wall of the reaction x job CPUs / workers (core-h) and x 1 GPU / workers (GPU-min), logged per task in "
               "$MLT/a/alloc/<list>_<arm>.jsonl; Ap adds the G2 TS cost (G_status core_h_ts + core_h_stereo); App = "
               "label-level OptTS wall x 8 ranks + A's (allocated) core-h, GPU = A's; medians over the attempted "
               "reactions (pre-start failures: Ap without a G2 TS, App whose A failed, are left out).",
    "A3-metrics": "A-3 per arm vs the label TS ($LAD/dft_final): same_ts = rmsd.same_ts (0.10 A / 0.02 A, sorted "
                  "forming distances), |dd| = max over the label forming pairs of |d_arm - d_label| after mapping the "
                  "arm's atoms onto the label order (same order: direct; else heavy-graph isomorphism of minimal RMSD); "
                  "same_ts_frac and success over all listed reactions (a failure counts as not same); dd / RMSD medians "
                  "over the same-TS reactions (and over all ok); cost medians over all finished reactions.",
    "App-input": "A'' input = g3l.g3l_text(opt.d1(), A's ts.xyz, manifest charge, A's formed_pairs_ts) (the G3L text: "
                 "D1 L0 OptTS Freq header, InHess Read / InHessName \"g2.hess\", no Recalc_Hess, %pal 8); the file "
                 "named g2.hess is A's ts.hess (MLTS analytic Hessian at that geometry, $atoms checked <= 1e-4 A); "
                 "D1G3_SCRATCH points to a shadow under $MLT (nothing written to $G3); one infra rerun as gen_dft G-2.",
    "Ap-start": "A' start = $G2/jobs/<T>/ts.xyz with its own formed_pairs_ts (autodE order, = $G2/g2_final/<rxn>/ts.xyz "
                "except the 0.08 % re-mapped reactions); a reaction without G2 .done fails as no_g2_ts; A' TS gate = "
                "optimize_ts converged + run_ts.analyse (autodE frequencies of the MLIP Hessian).",
}
__doc__ += "\n" + "\n".join(f"  {k}: {v}" for k, v in DECISIONS.items())

# ================================================================ paths and constants
BASE = Path(os.environ.get("MLTS_A_BASE") or str(M.MLT))
IS_PROD = Path(os.path.abspath(BASE)) == Path(os.path.abspath(M.MLT))
ADE_MAIN = BASE / "ade"                       # out_root of the A-4 main engine (H reads $MLT/ade/jobs)
ADE_PRE = BASE / "ade_pre"                    # A when A'' is the main engine
A3_ROOT = BASE / "a3"                         # a3/<arm>/ade
AWORK = BASE / "a"                            # slices/, logs/, d1g3_shadow/
SMOKE_ROOT = BASE / "a2smoke"
REPORTS = M.RES if IS_PROD else BASE / "reports"
TS_TEMPLATES = Path(os.environ.get("MLTS_TS_TEMPLATES") or str(BASE / "ts_templates"))
LISTS = M.MLT / "pipeline" / "lists"
MANIFEST = M.D1RUN / "d1_manifest.csv"
G_STATUS = M.ANALYSIS / "d1_g2" / "results" / "G_status.csv"
ARMS = ("A", "Ap", "App")
STAGES = ("ts", "stereo", "ref")
SEG_CYCLES, MAXITER_TS, TRUST_TS = 30, 100, -0.1          # = mlts_bridge.D1_OPTTS (A-1)
INHESS = "mlts_inhess.hess"
OPTTS_BLOCK = (f'\n%geom\nInHess Read\nInHessName "{INHESS}"\nTrust {TRUST_TS}\nMaxIter {SEG_CYCLES}\nend')
FREQ_DEFAULT = os.environ.get("MLTS_FREQ", "analytic")
CORES_DEFAULT = int(os.environ.get("MLTS_CORES", "2"))      # A2-workers (measured: serial ORCA steps + GPU waits)
WORKERS_CAP = 6                                            # before A-3 has measured the GPU share
WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
RESERVE_H = float(os.environ.get("MLT_RESERVE_H", "2"))
RXN_TIMEOUT_H = float(os.environ.get("MLTS_RXN_TIMEOUT_H", "8"))
MAX_SERVER_RESTARTS = 3
MAX_PIPELINE_ERR = 20
APP_NPROCS = 8
KEEP_SUFFIX = (".xyz", ".json", ".csv", ".txt", ".log", ".hess", ".jsonl")
LEVEL_A = "autodE 1.4.5 from SMILES (D1 flow), hmethod = ORCA 6.1.1 ExtOpt + MLTS (mlts_bridge server)"
LEVEL_REF = "MLTS: ORCA Opt TightOpt ExtOpt + MLIP analytic Hessian (autodE thermo, 298.15 K)"
A3_BUDGET = {"A": (0.6, 3.0), "Ap": (0.6, 3.0), "App": (1.0, 3.0)}   # spec 0-5 (= pipeline.A3_BUDGET)
N_D1 = 9266
T_START = time.time()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[mlts_autode {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T_START)
    return WALL_H - (time.time() - t0) / 3600.0


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def jobtag():
    """MLTS_RUN_TAG, else <jobid>[_<task>] (stable across the steps of one job: node-local names and the autodE
    calculation identities stay the same, so a killed sandbox step resumes)."""
    if os.environ.get("MLTS_RUN_TAG"):
        return re.sub(r"[^A-Za-z0-9_.-]", "_", os.environ["MLTS_RUN_TAG"])
    j, a = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{j}_{a}" if j and a else (j or f"pid{os.getpid()}")


def local_base():
    p = Path(os.environ.get("MLTS_LOCAL") or os.environ.get("TMPDIR") or "/tmp")
    return p if p.is_dir() and os.access(p, os.W_OK) else Path("/tmp")


def run_dir(tag, job_id, kind="mlts", arm="A"):
    """node-local work folder ${TMPDIR:-/tmp}/mlts_<jobtag>_<job_id> (arm A; other arms get a _<arm> suffix: two
    arms of one reaction may run at the same time in one job)."""
    return local_base() / (f"{kind}_{tag}_{job_id}" + ("" if arm == "A" else f"_{arm}"))


def socket_path(tag, name):
    d = local_base() / f"mlts_{tag}"
    p = d / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))[:40]}.sock"
    if len(str(p)) > 100:                                         # AF_UNIX limit (108 bytes)
        h = hashlib.sha1(str(p).encode()).hexdigest()[:12]
        d = Path("/tmp") / f"mlts_{h}"
        p = d / "s.sock"
    return p


def jdump(path, obj):
    return M.write_json(path, obj)


def local_json(path, obj):
    """atomic JSON write inside a node-local work folder (published later by copy_out, which is guarded)."""
    p = Path(path)
    t = p.with_name(f".{p.name}.tmp{os.getpid()}")
    t.write_text(json.dumps(obj, indent=1, default=M._json_default) + "\n")
    os.replace(t, p)
    return p


def rj(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def stage_state(d):
    """('ok' | 'fail' | 'incomplete', reason) from the stage markers of an A-2 folder."""
    d = Path(d)
    for st in STAGES:
        f = d / f".fail_{st}"
        if f.is_file():
            txt = f.read_text(errors="replace").strip()
            return "fail", f"{st}_" + (txt.split(":")[0].strip() if txt else "fail")
        if not (d / f".done_{st}").is_file():
            return "incomplete", f"{st}_not_run"
    return "ok", ""


def is_final(d):
    return Path(d).is_dir() and not Path(d).name.startswith(".")


# ================================================================ environments
def env_cmd(name, args):
    """argv running `python <this file> <args>` in MLTENV=name (jobs/env.sh)."""
    return ["bash", "-c", f'cd "$0" && export MLTENV={name} && source jobs/env.sh && exec "$PY" "$@"', str(HERE),
            str(THIS), *[str(a) for a in args]]


def ensure_env(name):
    """re-execute this command in MLTENV=name unless already there (inputs_mlts.ensure_env pattern)."""
    if os.environ.get("MLTENV") == name:
        return
    if os.environ.get("MLTS_A_REEXEC") == name:
        raise SystemExit(f"[mlts_autode] re-exec into MLTENV={name} did not take effect")
    sys.stdout.flush()
    sys.stderr.flush()
    env = dict(os.environ, MLTS_A_REEXEC=name)
    cmd = env_cmd(name, sys.argv[1:])
    os.execvpe("bash", cmd, env)


def have_autode():
    try:
        import importlib.util
        return importlib.util.find_spec("autode") is not None
    except Exception:                                            # noqa: BLE001
        return False


# ================================================================ lists, rows, models
_T = {}


def targets():
    if "T" not in _T:
        _T["T"] = M.targets()
    return _T["T"]


def manifest_rows(job_ids=None):
    want = None if job_ids is None else set(job_ids)
    out = {}
    with open(MANIFEST, newline="") as fh:
        for r in csv.DictReader(fh):
            if want is None or r["job_id"] in want:
                out[r["job_id"]] = r
    return out


def _rowdict(r):
    return dict(rxn_id=int(r.rxn_id), job_id=str(r.job_id), fold_T1=int(r.fold_T1), n_atoms=int(r.n_atoms),
                dipole=str(r.dipole), formed_pairs_ts=str(r.formed_pairs_ts))


def build_a3_list(path):
    """the pipeline.cmd_lists A-3 rule (60 fold-k test reactions per fold, round-robin over the dipoles present,
    numpy default_rng(SEED + 3)); used only when no launcher list exists and this is not the production base."""
    import numpy as np
    import pandas as pd
    T = targets()
    cols = ["rxn_id", "job_id", "dipole", "fold_T1", "n_atoms"]
    rng = np.random.default_rng(M.SEED + 3)
    parts = []
    for k in range(5):
        avail = {dn: int(((T.fold_T1 == k) & (T.dipole == dn)).sum()) for dn in M.DIPOLES}
        quota, left = {dn: 0 for dn in M.DIPOLES}, 60
        while left > 0 and any(quota[d] < avail[d] for d in M.DIPOLES):
            for dn in M.DIPOLES:
                if left > 0 and quota[dn] < avail[dn]:
                    quota[dn] += 1
                    left -= 1
        for dn in M.DIPOLES:
            g = T[(T.fold_T1 == k) & (T.dipole == dn)].sort_values("rxn_id")
            if quota[dn]:
                parts.append(g.iloc[sorted(rng.choice(len(g), size=quota[dn], replace=False))][cols])
    A = pd.concat(parts).sort_values(["fold_T1", "rxn_id"])
    M.write_csv(path, A)
    return path


def a3_list_path(list_path=None):
    if list_path:
        return Path(list_path)
    p = LISTS / "a3.csv"
    if p.is_file() or IS_PROD:
        return p
    q = BASE / "lists" / "a3.csv"
    if not q.is_file():
        build_a3_list(q)
    return q


def read_id_csv(path):
    with open(path, newline="") as fh:
        rd = list(csv.DictReader(fh))
    ids = []
    for r in rd:
        if r.get("rxn_id"):
            ids.append(int(float(r["rxn_id"])))
        elif r.get("job_id"):
            ids.append(M.rxn_of(r["job_id"]))
    return ids


def resolve_list(spec, list_path=None):
    """[row dict] of a list spec (module docstring)."""
    T = targets()
    s = str(spec)
    if s == "a3":
        p = a3_list_path(list_path)
        if not p.is_file():
            M.not_ready(f"A-3 list {p} missing (written by the launcher stage 'lists')")
        ids = set(read_id_csv(p))
        S = T[T.rxn_id.isin(ids)]
    elif s.startswith("d1:"):
        S = T[T.fold_T1 == int(s.split(":", 1)[1])]
    elif s.startswith("test:"):
        ids = set()
        for t in s.split(":", 1)[1].split(","):
            t = t.strip()
            if t:
                ids.add(M.rxn_of(t) if t.upper().startswith("T") else int(t))
        S = T[T.rxn_id.isin(ids)]
    elif s.startswith("ids:"):
        S = T[T.rxn_id.isin(set(read_id_csv(s.split(":", 1)[1])))]
    else:
        raise SystemExit(f"unknown list {s!r} (a3 | d1:<k> | test:<ids> | ids:<csv>)")
    return [_rowdict(r) for r in S.itertuples()]


def slice_rows(rows, i, n):
    """deterministic slice i of n: largest first, round-robin (balanced walls)."""
    rows = sorted(rows, key=lambda r: (-int(r["n_atoms"]), int(r["rxn_id"])))
    return rows[int(i)::int(n)]


def model_for(row, override=None):
    if override:
        return override
    k = int(row["fold_T1"])
    return f"FT_{k if k >= 0 else 0}_al"


def arms_of(engine, list_spec):
    if engine:
        return [e.strip() for e in str(engine).split(",") if e.strip()]
    return ["A", "Ap"] if list_spec == "a3" else ["A"]


def out_root_for(list_spec, arm, out=None, main_engine=None):
    """A-2 out_root (…/ade) of an arm run by `run`."""
    if out:
        return Path(out)
    if list_spec == "a3":
        return A3_ROOT / ("A" if arm == "App" else arm) / "ade"
    if arm == "App" or main_engine == "App":
        return ADE_PRE
    return ADE_MAIN


def app_roots(list_spec, root=None, src=None):
    """(source A root, destination root) of refine_dft."""
    if list_spec == "a3":
        s = Path(src) if src else A3_ROOT / "A" / "ade"
        d = (Path(root) / "ade") if root else A3_ROOT / "App" / "ade"
        return s, d
    if root:
        return (Path(src) if src else ADE_MAIN), Path(root) / "ade"
    return (Path(src) if src else ADE_PRE), ADE_MAIN


# ================================================================ D1 modules (autodE worker / D1 env only)
_D = {}


def d1():
    """pilot_common, run_ts, make_reference, orca_direct (the D1 code copy), mlts_bridge, numpy, yaml."""
    if _D:
        return _D
    import numpy as np
    import yaml
    import mlts_bridge as MB                                   # before the D1 paths (no name clashes either way)
    for p in (str(M.D1RUN), str(M.PILOT)):
        if p not in sys.path:
            sys.path.insert(1, p)
    import pilot_common as pc
    import run_ts as RT
    import make_reference as MR
    import orca_direct as OD
    _D.update(np=np, yaml=yaml, MB=MB, pc=pc, RT=RT, MR=MR, OD=OD, orig_configure=pc.configure_autode,
              orig_refine=MR.dft_refine)
    return _D


def base_config(local):
    D = d1()
    cfg = D["yaml"].safe_load(Path(M.D1_CONFIG).read_text())
    cfg["scratch"] = str(local)
    return cfg


# ================================================================ in-process autodE hooks (worker)
_CTX = dict(installed=False)


def _append(path, line):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def mlip_hessian(syms, pos, charge=0):
    """MLIP analytic Hessian (Eh/bohr^2) through the server (molecular charge passed on: A2-charge); one line in the
    reaction's Hessian log."""
    MB = d1()["MB"]
    t = time.time()
    H, dt = MB.server_hessian(_CTX["sock"], list(syms), d1()["np"].asarray(pos, float), int(charge or 0))
    _append(_CTX["hess_log"], f"{(time.time() - t) * 1e3:.2f} {dt * 1e3:.2f} {len(syms)} 2\n")
    return H


def write_xyz_plain(path, syms, pos, title=""):
    body = "".join(f"{s:<3}{x:16.10f}{y:16.10f}{z:16.10f}\n" for s, (x, y, z) in zip(syms, pos))
    Path(path).write_text(f"{len(syms)}\n{title}\n{body}")


def _seg_text(text0, pos, maxiter):
    """autodE's OptTS input with the current geometry (autodE print_coordinates format) and MaxIter = maxiter in the
    A-1 block (the first 'MaxIter' line after 'InHess Read')."""
    lines = text0.split("\n")
    k = next(i for i, ln in enumerate(lines) if ln.startswith("*xyz"))
    n = len(pos)
    coords = [f"{lines[k + 1 + j].split()[0]:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}" for j, (x, y, z) in enumerate(pos)]
    if lines[k + 1 + n].strip() != "*":
        raise RuntimeError("OptTS input: coordinate block does not have the expected atom count")
    out = lines[:k + 1] + coords + lines[k + 1 + n:]
    ih = next(i for i, ln in enumerate(out) if ln.strip() == "InHess Read")
    im = next(i for i in range(ih, len(out)) if re.match(r"^\s*MaxIter\s+\d+\s*$", out[i]))
    out[im] = f"MaxIter {int(maxiter)}"
    return "\n".join(out)


def _optts_segments(calc, run_external):
    """the A-1 OptTS of one autodE opt_ts calculation, inside autodE's temporary folder (cwd)."""
    D = d1()
    np, MB = D["np"], D["MB"]
    name = calc.name
    inp0, out0 = Path(calc.input.filename), Path(calc.output.filename)
    text0 = inp0.read_text()
    syms = [a.label for a in calc.molecule.atoms]
    q = int(calc.molecule.charge or 0)
    cur = np.asarray(calc.molecule.coordinates, float)
    t0 = time.time()
    cycles, segs = 0, []
    while True:
        k = len(segs) + 1
        smax = min(SEG_CYCLES, MAXITER_TS - cycles)
        MB.write_orca_hess(INHESS, syms, cur, mlip_hessian(syms, cur, q))
        base = name if k == 1 else f"{name}_s{k}"
        if k > 1:
            Path(base + ".inp").write_text(_seg_text(text0, cur, smax))
        run_external(params=[calc.method.path, base + ".inp"], output_filename=base + ".out")
        r = MB.parse_run(Path("."), base, syms, cur)
        segs.append(dict(seg=k, n_cycles=r["n_cycles"], converged=r["converged"], terminated=r["terminated"],
                         max_iter_hit=r["max_iter_hit"], inhess_read=r["inhess_read"][:1], errors=r["errors"][:2]))
        cycles += int(r["n_cycles"])
        cur = np.asarray(r["pos_A"], float)
        if r["converged"] or r["n_cycles"] < smax or cycles >= MAXITER_TS:
            break
    if len(segs) > 1:
        os.replace(out0, Path(f"{name}_s1.out"))
        with open(out0, "w") as fo:
            for s in segs:
                b = name + "_s1" if s["seg"] == 1 else f"{name}_s{s['seg']}"
                if s["seg"] > 1:
                    fo.write(f"\n# ---- mlts_autode: OptTS segment {s['seg']} (restart with a new MLIP Hessian)\n")
                fo.write(Path(b + ".out").read_text(errors="replace"))
    write_xyz_plain(f"{name}.xyz", syms, cur, f"{name} MLTS OptTS final")
    hess = f"{name}.hess"
    done = False
    if _CTX["freq"] == "numfreq" and segs[-1]["converged"]:
        MB.run_orca_ext(syms, cur, "NumFreq", _CTX["spec"], _CTX["sock"], Path("."), base=f"{name}_nf",
                        charge=int(calc.molecule.charge), mult=int(calc.molecule.mult))
        _copy_ext_calls(Path("mlts_ext.log"))
        if Path(f"{name}_nf.hess").is_file():
            shutil.copy(f"{name}_nf.hess", hess)
            done = True
    if not done:                                                 # always the MLIP Hessian at the final geometry
        MB.write_orca_hess(hess, syms, cur, mlip_hessian(syms, cur, q))
    return dict(calc=name, n_segments=len(segs), n_cycles=cycles, converged=bool(segs[-1]["converged"]),
                wall_s=time.time() - t0, segs=segs, n_atoms=len(syms), final_pos=np.round(cur, 7).tolist())


def _copy_ext_calls(p):
    """E/G calls logged by a run_orca_ext run in this folder -> the reaction's call log (GPU accounting)."""
    if p.is_file():
        _append(_CTX["ext_log"], p.read_text())
        p.unlink()


def _exec(self, calc):
    from autode.wrappers.keywords import HessianKeywords, OptTSKeywords
    kw = calc.input.keywords
    if isinstance(kw, OptTSKeywords):
        from autode.config import Config
        from autode.utils import run_external, work_in_tmp_dir
        here, rec = os.getcwd(), {}

        @work_in_tmp_dir(filenames_to_copy=calc.input.filenames, kept_file_exts=Config.ORCA.copied_output_exts)
        def go():
            rec.update(_optts_segments(calc, run_external))
        go()
        rec["hess_path"] = os.path.join(here, f"{calc.name}.hess")
        _append(_CTX["optts_log"], json.dumps(rec) + "\n")
        return None
    _CTX["orig_execute"](self, calc)
    if _CTX["freq"] == "analytic" and isinstance(kw, HessianKeywords):
        out = Path(calc.output.filename)
        if out.is_file() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="replace")[-4000:]:
            syms = [a.label for a in calc.molecule.atoms]
            pos = d1()["np"].asarray(calc.molecule.coordinates, float)
            d1()["MB"].write_orca_hess(out.with_suffix(".hess"), syms, pos,
                                       mlip_hessian(syms, pos, int(calc.molecule.charge or 0)))
    return None


def _gen(self, calc):
    n = calc.n_cores
    calc.n_cores = 1
    try:
        return _CTX["orig_gen"](self, calc)
    finally:
        calc.n_cores = n


def _energy_from(self, calc):
    from autode.exceptions import CouldNotGetProperty
    from autode.values import PotentialEnergy
    if calc.output.filename.endswith(".hess"):
        return PotentialEnergy(0.0)
    for line in reversed(calc.output.file_lines):
        if "FINAL SINGLE POINT ENERGY" in line:
            return PotentialEnergy(line.split()[-1], units="Ha")
    raise CouldNotGetProperty(name="energy")


def _last_energy_change(self):
    import numpy as np
    from autode.values import PotentialEnergy
    e = [PotentialEnergy(ln.split()[-1], units="Ha") for ln in self._lines if "FINAL SINGLE POINT ENERGY" in ln]
    if len(e) < 2:
        return PotentialEnergy(np.inf)
    return e[-1] - e[-2]


def mlts_configure(cfg, cores=None):
    """pilot_common.configure_autode unchanged, then the MLTS changes (module docstring)."""
    D = d1()
    ade = D["orig_configure"](cfg, cores)
    from autode.wrappers.keywords import KeywordsSet, MaxOptCycles
    import autode.wrappers.ORCA as AO
    blk = "\n" + D["MB"].ext_method_block(_CTX["spec"], _CTX["sock"], log=_CTX["ext_log"]).rstrip("\n")
    hess = ["EnGrad", "ExtOpt", blk] if _CTX["freq"] == "analytic" else ["NumFreq", "ExtOpt", blk]
    ade.Config.ORCA.keywords = KeywordsSet(
        low_opt=["LooseOpt", "ExtOpt", MaxOptCycles(10), blk], grad=["EnGrad", "ExtOpt", blk],
        low_sp=["SP", "ExtOpt", blk], opt=["Opt", "TightOpt", "ExtOpt", blk],
        opt_ts=["OptTS", "TightOpt", "ExtOpt", OPTTS_BLOCK, blk], hess=hess, sp=["SP", "ExtOpt", blk])
    if not _CTX.get("patched"):
        _CTX.update(orig_execute=AO.ORCA.execute, orig_gen=AO.ORCA.generate_input_for, patched=True)
        AO.ORCA.add_solvent_keyword = lambda self, molecule, keywords: None
        AO.ORCA.print_solvent = lambda self, inp_file, molecule, keywords: None
        AO.ORCA.execute = _exec
        AO.ORCA.generate_input_for = _gen
        AO.ORCA._energy_from = _energy_from
        AO.ORCAOptimiser.last_energy_change = property(_last_energy_change)
    tdir = template_dir(_CTX["spec"])
    tdir.mkdir(parents=True, exist_ok=True)
    ade.Config.ts_template_folder_path = str(tdir)
    return ade


def template_dir(spec):
    """A2-templates: one autodE TS template folder per MLTS model under $MLT/ts_templates (a model's templates never
    guide another model's search: no cross-fold flow, results independent of other folds' run order)."""
    name = d1()["MB"].spec_name(spec) if isinstance(spec, dict) else str(spec)
    return TS_TEMPLATES / re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:80]


def mlts_refine(ade, mol, temp):
    """replaces make_reference.dft_refine (A2-ref)."""
    from autode.methods import get_hmethod
    h = get_hmethod()
    t = time.time()
    mol.optimise(method=h)
    e_opt = float(mol.energy)
    mol.calc_thermo(method=h, temp=temp)
    e = float(mol.energy)
    g = float(mol.g_cont) if mol.g_cont is not None else None
    return dict(e_opt_eh=e_opt, e_sp_eh=e, g_cont_eh=g, G_eh=(e + g) if g is not None else None, level=LEVEL_REF,
                wall_s=time.time() - t)


def install(spec, sock, local, freq):
    D = d1()
    _CTX.update(spec=spec, sock=str(sock), ext_log=str(local / "mlts_ext.log"), hess_log=str(local / "mlts_hess.log"),
                optts_log=str(local / "mlts_optts.jsonl"), freq=freq, installed=True)
    D["pc"].configure_autode = mlts_configure             # run_ts / make_reference look it up at call time
    D["MR"].dft_refine = mlts_refine


# ================================================================ stages (node-local job folder jd)
def infra_evidence(jd, sock, n_err0, orca=True):
    """A2-retry: infra evidence of a failed attempt. orca (the D1 rule, only for an 'autodE exception'):
    d1_run/orca_parse.py --infra on the ts folder; always: the MLTS server not answering or its error count rising
    during the attempt (a server failure while autodE builds TS guesses makes autodE drop the guess, so the stage can
    end as 'no transition state' / a TS check instead of an exception)."""
    ev = []
    D = d1()
    if orca:
        r = subprocess.run([sys.executable, str(M.D1RUN / "orca_parse.py"), "--infra", str(jd / "ts")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ev.append("orca_parse_infra")
    st = D["MB"].health(sock)
    if st is None:
        ev.append("server_down")
    elif int(st.get("n_err") or 0) > int(n_err0 or 0):
        ev.append(f"server_errors_{int(st.get('n_err') or 0) - int(n_err0 or 0)}")
    return ev


def wait_server(sock, timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if d1()["MB"].health(sock):
            return True
        time.sleep(5)
    return False


def _n_err(sock):
    st = d1()["MB"].health(sock)
    return int(st.get("n_err") or 0) if st else 0


def stage_ts_A(cfg, job, jd, spec, sock, freq):
    """run_ts.main for kind D1 (run_autode + analyse), with the D1 retry_infra_once rule (A2-retry)."""
    D = d1()
    pc, RT = D["pc"], D["RT"]
    if pc.done(jd, "ts"):
        return True
    fr, _ = pc.import_repo_modules(cfg)
    res = dict(job_id=job["job_id"], kind=job["kind"], ts_id=job.get("ts_id"), rxn_smiles=job["rxn_smiles"],
               engine=cfg["ts_stage"]["engine"], arm="A", level=LEVEL_A, model=D["MB"].spec_name(spec), spec=spec,
               freq_route=freq, code=CODE_VERSION, attempts=[])
    for att in (1, 2):
        t0, n0 = time.time(), _n_err(sock)
        try:
            got = RT.run_autode(cfg, job, jd, res, False)
            res["wall_s"] = time.time() - t0
            RT.analyse(cfg, job, jd, res, got, fr)
        except RT.StageFail as e:
            res["wall_s"] = time.time() - t0
            ev = infra_evidence(jd, sock, n0, orca=str(e).startswith("autodE exception"))
            res["attempts"].append(dict(attempt=att, wall_s=res["wall_s"], fail=str(e), infra=ev))
            if att == 1 and ev:
                if "server_down" in ev:
                    wait_server(sock)
                continue                                         # D1 retry_infra_once: same input, once
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", ("infra: " if ev else "") + str(e))
            return False
        except Exception as e:                                   # noqa: BLE001  (A2-crash: not a StageFail)
            res["wall_s"] = time.time() - t0
            ev = infra_evidence(jd, sock, n0, orca=False)
            res["attempts"].append(dict(attempt=att, wall_s=res["wall_s"], fail=f"{type(e).__name__}: {e}"[:400],
                                        infra=ev, traceback=traceback.format_exc()[-2000:]))
            if att == 1:
                if "server_down" in ev:
                    wait_server(sock)
                continue
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", f"pipeline_error: {type(e).__name__}: {e}"[:300])
            return False
        res["attempts"].append(dict(attempt=att, wall_s=res["wall_s"], fail=None))
        pc.write_json(jd / "ts_result.json", res)
        pc.mark_done(jd, "ts")
        return True
    return False


def ade_imag_mode(syms, pos, H, charge):
    """autodE's imaginary frequencies and first vibrational mode of a Cartesian Hessian (Eh/bohr^2), as A gets them."""
    np = d1()["np"]
    import autode as ade
    from autode.hessians import Hessian
    mol = ade.Molecule(atoms=[ade.Atom(s, *map(float, c)) for s, c in zip(syms, pos)], charge=int(charge), mult=1)
    mol.hessian = Hessian(np.asarray(H, float), atoms=mol.atoms, units="Ha a0^-2").to("Ha Å^-2")
    imag = [float(v) for v in (mol.imaginary_frequencies or [])]
    try:
        mode = np.array(mol.normal_mode(6), dtype=float)
    except Exception:                                            # noqa: BLE001
        mode = None
    return imag, mode


def stage_ts_Ap(cfg, job, jd, spec, sock):
    """A': the G2 TS -> mlts_bridge.optimize_ts; G2's reactants MLTS-optimised; run_ts.analyse (Ap-start)."""
    D = d1()
    pc, RT, MB, np = D["pc"], D["RT"], D["MB"], D["np"]
    if pc.done(jd, "ts"):
        return True
    fr, _ = pc.import_repo_modules(cfg)
    T, rid = job["job_id"], int(float(job["meta_rxn_id"]))
    g2j, g2f = M.G2_JOBS / T, M.G2_FINAL / str(rid)
    res = dict(job_id=T, kind=job["kind"], ts_id=job.get("ts_id"), rxn_smiles=job["rxn_smiles"], engine="orca",
               arm="Ap", level="A': G2 TS -> ORCA ExtOpt OptTS (MLTS, A-1)", model=MB.spec_name(spec), spec=spec,
               code=CODE_VERSION, start=str(g2j / "ts.xyz"))
    t0 = time.time()
    if not (g2f / ".done").is_file() or not (g2j / "ts.xyz").is_file() or not (g2j / "ts_result.json").is_file():
        res.update(wall_s=0.0, fail="no G2 TS")
        pc.write_json(jd / "ts_result.json", res)
        pc.mark_fail(jd, "ts", "no_g2_ts: $G2 has no .done TS for this reaction")
        return False
    res["attempts"] = []
    q = int(job["charge"])
    for att in (1, 2):                                           # A2-retry (A' too): one infra / crash retry
        n0 = _n_err(sock)
        try:
            ade = pc.configure_autode(cfg)
            syms, pos0 = pc.read_xyz(g2j / "ts.xyz")
            tsr0 = json.loads((g2j / "ts_result.json").read_text())
            pairs = [tuple(int(x) for x in p) for p in tsr0["formed_pairs_ts"]]
            shutil.rmtree(jd / "ap_optts", ignore_errors=True)
            r = MB.optimize_ts(spec, syms, pos0, pairs, jd / "ap_optts", sock, start_hess="analytic", tight=True,
                               final_hessian=False)
            res["optimize_ts"] = {k: r.get(k) for k in ("status", "converged", "n_cycles", "n_segments", "n_ext_calls",
                                                        "gpu_s", "n_hess_calls", "recalc", "start_hess", "tight",
                                                        "inhess_read", "e_final_eh", "gmax_final_eh_bohr", "orca_wall_s",
                                                        "wall_s", "max_iter_hit")}
            res["g2_formed_pairs_ts"] = [list(p) for p in pairs]
            if r["status"] != "ok":
                raise RT.StageFail(f"MLTS OptTS {r['status']} (from the G2 TS)")
            pos = np.asarray(r["pos_A"], float)
            H = mlip_hessian(syms, pos, q)
            imag, mode = ade_imag_mode(syms, pos, H, q)
            reacs = []
            wd = jd / "ap_reactants"
            wd.mkdir(exist_ok=True)
            os.chdir(wd)
            from autode.methods import get_hmethod
            for f in ("r_dipole_autode.xyz", "r_dipolarophile_autode.xyz"):
                s, x = pc.read_xyz(g2j / f)
                mol = ade.Molecule(name=f"g2{f.split('_')[1]}", atoms=[ade.Atom(a, *map(float, c)) for a, c in zip(s, x)],
                                   charge=0, mult=1, solvent_name=cfg["ts_stage"]["solvent"])
                mol.optimise(method=get_hmethod())
                reacs.append(([a.label for a in mol.atoms], np.array([a.coord for a in mol.atoms], dtype=float)))
            res["wall_s"] = time.time() - t0
            got = dict(ts=(list(syms), pos), reacs=reacs, prod=None, imag=imag, mode=mode)
            RT.analyse(cfg, job, jd, res, got, fr)
            D["MB"].write_orca_hess(jd / "ts.hess", syms, pos, H)
        except RT.StageFail as e:
            res["wall_s"] = time.time() - t0
            ev = infra_evidence(jd, sock, n0, orca=False)
            res["attempts"].append(dict(attempt=att, fail=str(e), infra=ev))
            if att == 1 and ev:
                if "server_down" in ev:
                    wait_server(sock)
                continue
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", ("infra: " if ev else "") + str(e))
            return False
        except Exception as e:                                   # noqa: BLE001  (A2-crash: e.g. a server error)
            res["wall_s"] = time.time() - t0
            ev = infra_evidence(jd, sock, n0, orca=False)
            res["attempts"].append(dict(attempt=att, fail=f"{type(e).__name__}: {e}"[:400], infra=ev,
                                        traceback=traceback.format_exc()[-2000:]))
            if att == 1:
                if "server_down" in ev:
                    wait_server(sock)
                continue
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", f"pipeline_error: {type(e).__name__}: {e}"[:300])
            return False
        res["attempts"].append(dict(attempt=att, fail=None))
        pc.write_json(jd / "ts_result.json", res)
        pc.mark_done(jd, "ts")
        return True
    return False


def stage_stereo(cfg_path, manifest, job, jd):
    pc = d1()["pc"]
    if pc.done(jd, "stereo"):
        return True
    r = subprocess.run([sys.executable, str(M.D1RUN / "stereo_check.py"), "--job", job["job_id"], "--config",
                        str(cfg_path), "--manifest", str(manifest)], cwd=str(M.PILOT), capture_output=True, text=True)
    (jd / "stereo.log").write_text(r.stdout + r.stderr)
    if not pc.done(jd, "stereo") and not (jd / ".fail_stereo").is_file():
        # A2-stereo: D1 stereo_check.py has no try/except around RDKit; a crash writes no marker -> close the stage
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
        pc.mark_fail(jd, "stereo", f"stereo_check rc {r.returncode}: no marker (crash) {tail[0][:200]}")
    return pc.done(jd, "stereo")


def stage_ref(cfg, job, jd):
    """make_reference.main's D1 branch with orig_energy=None (both dipole conformers through mlts_refine; A2-ref)."""
    D = d1()
    pc, MR = D["pc"], D["MR"]
    if pc.done(jd, "ref"):
        return True
    fr, _ = pc.import_repo_modules(cfg)
    ts_res = json.loads((jd / "ts_result.json").read_text())
    res = dict(job_id=job["job_id"], kind=job["kind"], level=LEVEL_REF)
    t0 = time.time()
    try:
        orig = pc.read_xyz(jd / "r_dipole_autode.xyz")
        r_dph = pc.read_xyz(jd / "r_dipolarophile_autode.xyz")
        port, alt = MR.coley_port(cfg, jd, jd / "ref", ts_res, fr, orig, None, False)
        res["port"] = port
        if "abort" in port and port.get("abort", "").startswith("ring count changed"):
            raise MR.StageFail(port["abort"])
        ref_dip = alt if port["use_alt"] else orig
        res.update(source="coley_port_mlts", alt_used=int(port["use_alt"]))
        res["dipolarophile_check"] = MR.dipolarophile_check(fr, jd, ts_res, job, r_dph)
        if res["dipolarophile_check"].get("ok") is False:
            raise MR.StageFail("dipolarophile configuration at the TS differs from its reference")
    except MR.StageFail as e:
        res.update(wall_s=time.time() - t0)
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", str(e))
        return False
    except Exception as e:                                       # noqa: BLE001
        res.update(wall_s=time.time() - t0, traceback=traceback.format_exc())
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", f"{type(e).__name__}: {e}")
        return False
    pc.write_xyz(jd / "ref_dipole.xyz", *ref_dip, f"{job['job_id']} MLTS dipole reference (alt_used={res['alt_used']})")
    pc.write_xyz(jd / "ref_dipolarophile.xyz", *r_dph, f"{job['job_id']} MLTS dipolarophile reference")
    res.update(wall_s=time.time() - t0)
    pc.write_json(jd / "ref_result.json", res)
    pc.mark_done(jd, "ref")
    return True


def final_ts_hess(jd, local, charge=0):
    """A: ts.hess = the MLIP Hessian autodE used at the final TS (OptTS log, geometry hash) or a new one."""
    D = d1()
    np, pc, MB = D["np"], D["pc"], D["MB"]
    if (jd / "ts.hess").is_file() or not (jd / "ts.xyz").is_file():
        return "exists" if (jd / "ts.hess").is_file() else "no_ts"
    syms, pos = pc.read_xyz(jd / "ts.xyz")
    pos = np.asarray(pos, float)
    lp = Path(_CTX["optts_log"])
    for ln in (lp.read_text().splitlines() if lp.is_file() else [])[::-1]:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        hp, fp = Path(r.get("hess_path") or ""), r.get("final_pos")
        if not (fp and hp.is_file() and np.shape(fp) == pos.shape):
            continue
        fp = np.asarray(fp, float)
        if float(np.abs((fp - fp.mean(0)) - (pos - pos.mean(0))).max()) < 1e-5:   # autodE translates the final TS
            H = MB.OI.read_hess(hp)["hessian"]                                   # (Cartesian Hessian: invariant)
            MB.write_orca_hess(jd / "ts.hess", syms, pos, H)                     # $atoms = ts.xyz
            return "optts_translated" if float(np.abs(fp - pos).max()) >= 1e-5 else "optts"
    MB.write_orca_hess(jd / "ts.hess", syms, pos, mlip_hessian(syms, pos, charge))
    return "recomputed"


def log_sums(path):
    """(n calls, wall_s, gpu_s) of a client / Hessian log (wall_ms eval_ms n_atoms flag per line)."""
    n, w, g = 0, 0.0, 0.0
    p = Path(path)
    if p.is_file():
        for ln in p.read_text().splitlines():
            t = ln.split()
            if len(t) >= 2:
                try:
                    w += float(t[0]) / 1e3
                    g += float(t[1]) / 1e3
                    n += 1
                except ValueError:
                    pass
    return n, w, g


def copy_out(jd, dest, extra=()):
    """publish the A-2 folder: top-level small files + markers + extra files + inputs.tar.gz of every .inp under jd
    (atomic: <dest>.tmp -> os.replace)."""
    dest = Path(dest)
    tmp = dest.parent / f".{dest.name}.tmp{os.getpid()}"
    M.assert_write_ok(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if Path(jd).is_dir():
        for p in Path(jd).iterdir():
            if p.is_file() and (p.suffix in KEEP_SUFFIX or p.name.startswith((".done", ".fail"))):
                shutil.copy2(p, tmp / p.name)
    for p in extra:
        p = Path(p)
        if p.is_file():
            shutil.copy2(p, tmp / p.name)
    n_inp = 0
    with tarfile.open(tmp / "inputs.tar.gz", "w:gz") as t:
        if Path(jd).is_dir():
            for p in sorted(Path(jd).rglob("*.inp")):
                if p.is_file():
                    t.add(p, arcname=str(p.relative_to(jd)))
                    n_inp += 1
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return n_inp


def write_manifest(path, row):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


def _work_one(job, spec, engine, out_root, cores, sock, tag, freq):
    """one reaction in this (D1-env) process -> the published A-2 folder."""
    D = d1()
    pc = D["pc"]
    T = job["job_id"]
    out = Path(out_root) / "jobs" / T
    if is_final(out):
        return out
    local = run_dir(tag, T, arm=engine)
    local.mkdir(parents=True, exist_ok=True)
    (local / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(local / "tmp")                    # autodE mkdtemp + xtb scratch inside the work folder
    import tempfile
    tempfile.tempdir = str(local / "tmp")
    os.environ["SLURM_CPUS_PER_TASK"] = str(cores)               # pilot_common.n_cores() -> autodE n_cores
    os.environ["OMP_NUM_THREADS"] = str(cores)
    cfg = base_config(local)
    cfg_path = local / "config_mlts.yaml"
    cfg_path.write_text(D["yaml"].safe_dump(cfg))
    manifest = local / "manifest.csv"
    write_manifest(manifest, job)
    install(spec, sock, local, freq)
    jd = pc.job_dir(cfg, T)
    t_all, times, published = time.time(), {}, False
    tim = dict(job_id=T, engine=engine, model=D["MB"].spec_name(spec), spec=spec, cores=int(cores), freq_route=freq,
               host=_socket.gethostname(), job=jobtag(), started=now(), code=CODE_VERSION, local=str(local))
    prev = rj(local / "timing_partial.json") or {}
    try:
        t = time.time()
        ok = (stage_ts_A(cfg, job, jd, spec, sock, freq) if engine == "A" else stage_ts_Ap(cfg, job, jd, spec, sock))
        times["ts"] = time.time() - t
        if ok:
            t = time.time()
            ok = stage_stereo(cfg_path, manifest, job, jd)
            times["stereo"] = time.time() - t
        if ok:
            t = time.time()
            ok = stage_ref(cfg, job, jd)
            times["ref"] = time.time() - t
        os.chdir(str(local))
        if pc.done(jd, "ts"):
            t = time.time()
            tim["ts_hess"] = final_ts_hess(jd, local, int(float(job.get("charge") or 0))) if engine == "A" \
                else "optimize_ts_final"
            times["ts_hess"] = time.time() - t
        n_e, w_e, g_e = log_sums(_CTX["ext_log"])
        n_h, w_h, g_h = log_sums(_CTX["hess_log"])
        tsr = rj(jd / "ts_result.json") or {}
        ap = tsr.get("optimize_ts") or {}
        wall = time.time() - t_all + float(prev.get("wall_s") or 0.0)
        g_ap = float(ap.get("gpu_s") or 0.0)
        tim.update(wall_s=wall, core_h=wall * int(cores) / 3600.0, gpu_s=g_e + g_h + g_ap,
                   n_ext_calls=n_e + int(ap.get("n_ext_calls") or 0), eval_gpu_s=g_e, n_hess=n_h, hess_gpu_s=g_h,
                   ext_client_wall_s=w_e, hess_wall_s=w_h, ap_optimize_gpu_s=g_ap, stage_wall_s=times,
                   wall_s_earlier_steps=float(prev.get("wall_s") or 0.0), finished=now(),
                   status="ok" if all(pc.done(jd, s) for s in STAGES) else "fail")
        if engine == "Ap":
            gs = g2_cost().get(T) or {}
            tim.update(g2_core_h_ts=gs.get("core_h_ts"), g2_core_h_stereo=gs.get("core_h_stereo"),
                       core_h_deploy=tim["core_h"] + float(gs.get("core_h_ts") or 0) + float(gs.get("core_h_stereo") or 0))
        if engine == "A":
            tim["n_optts_calcs"] = sum(1 for _ in Path(_CTX["optts_log"]).read_text().splitlines()) \
                if Path(_CTX["optts_log"]).is_file() else 0
        tim["n_inputs"] = sum(1 for p in jd.rglob("*.inp") if p.is_file())
        pc.write_json(jd / "timing.json", tim)
        extra = [p for p in (local / "mlts_optts.jsonl",) if p.is_file()]
        copy_out(jd, out, extra)
        published = True
    finally:
        os.chdir("/")
        if not published:
            local_json(local / "timing_partial.json", dict(wall_s=time.time() - t_all + float(prev.get("wall_s") or 0)))
        elif os.environ.get("MLTS_KEEP_LOCAL") != "1":
            shutil.rmtree(local, ignore_errors=True)
    return out


_G2C = {}


def g2_cost():
    if "c" not in _G2C:
        c = {}
        if G_STATUS.is_file():
            with open(G_STATUS, newline="") as fh:
                for r in csv.DictReader(fh):
                    c[r["job_id"]] = {k: float(r[k]) for k in ("core_h_ts", "core_h_stereo", "core_h_total")
                                      if r.get(k) not in (None, "")}
        _G2C["c"] = c
    return _G2C["c"]


# ================================================================ servers (GPU job parent, MLTENV=mlts)
def _spawn_server(spec, sock, log_path):
    """mlts_bridge.start_server in this env, or the mlts env through jobs/env.sh (for run_one in the D1 env)."""
    import mlts_bridge as MB
    Path(sock).parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: F401
        return MB.start_server(spec, sock_path=str(sock), log_path=str(log_path))
    except ImportError:
        pass
    M.assert_write_ok(log_path)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", "-c", 'cd "$0" && export MLTENV=mlts && source jobs/env.sh && exec "$PY" "$@"', str(HERE),
           str(HERE / "mlts_bridge.py"), "serve", "--spec", json.dumps(spec), "--socket", str(sock), "--parent-pid",
           str(os.getpid())]
    lf = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    lf.close()
    t0 = time.time()
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc {proc.returncode} (log {log_path})")
        if Path(sock).exists() and MB.health(str(sock), timeout=2.0):
            return MB.ServerHandle(spec, str(sock), proc, str(log_path))
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"server not ready after 1200 s (log {log_path})")


class Servers:
    """one server per model name, started on first use, restarted when its process died (<= 3 times), stopped when
    release() says no task of that model is left (A2-workers)."""

    def __init__(self, tag, logdir):
        self.tag, self.logdir = tag, Path(logdir)
        self.h, self.restarts, self.stats = {}, Counter(), {}
        self.mu = threading.Lock()
        self.locks = {}

    def get(self, name, spec):
        with self.mu:
            lk = self.locks.setdefault(name, threading.Lock())
        with lk:
            h = self.h.get(name)
            if h is not None and h.proc is not None and h.proc.poll() is None:
                return h.socket
            if h is not None:
                if self.restarts[name] >= MAX_SERVER_RESTARTS:
                    raise RuntimeError(f"server {name} died {self.restarts[name]} times")
                self.restarts[name] += 1
                log(f"server {name} not running (rc {h.proc.poll() if h.proc else None}): restart {self.restarts[name]}")
            sock = socket_path(self.tag, name)
            t = time.time()
            self.h[name] = _spawn_server(spec, sock, self.logdir / f"server_{self.tag}_{name}.log")
            log(f"server {name} up at {sock} in {time.time() - t:.1f} s")
            return self.h[name].socket

    def monitor(self, stop):
        while not stop.wait(15):
            with self.mu:
                items = list(self.h.items())
            for name, h in items:
                if h.proc is not None and h.proc.poll() is not None and not getattr(h, "released", False):
                    try:
                        self.get(name, h.spec)
                    except Exception as e:                       # noqa: BLE001
                        log(f"server {name} restart failed: {e}")

    def release(self, name):
        with self.mu:
            h = self.h.get(name)
        if h is None:
            return
        h.released = True
        self.stats[name] = {k: v for k, v in (h.stats() or {}).items() if k != "spec"}
        h.shutdown()

    def close(self):
        for name in list(self.h):
            if not getattr(self.h[name], "released", False):
                self.release(name)


# ================================================================ the GPU job: run
def _publish_timeout(local, dest, T, why, kind="timeout_worker"):
    """parent side: a worker killed by the per-reaction timeout (kind timeout_worker) or crashing twice (kind
    pipeline_error, A2-crash) -> A-2 folder with .fail_<first unfinished stage> (no D1 code needed)."""
    jd = local / "jobs" / T
    jd.mkdir(parents=True, exist_ok=True)
    if not any((jd / f".fail_{s}").is_file() for s in STAGES):
        st = next((s for s in STAGES if not (jd / f".done_{s}").is_file()), "ref")
        (jd / f".fail_{st}").write_text(f"{kind}: {why}")
    local_json(jd / "timing.json", dict(job_id=T, status="fail", reason=f"{kind}: {why}", finished=now()))
    copy_out(jd, dest)
    shutil.rmtree(local, ignore_errors=True)


KILLED_RC = (143, 137, -15, -9, 130, -2)                # SIGTERM / SIGKILL / SIGINT: the job is ending, not a crash


def crash_path(dest, arm):
    """A2-crash: per (out_root, reaction, arm) count of worker exits without a published folder (shared FS, so it
    survives job boundaries)."""
    h = hashlib.sha1(str(Path(os.path.abspath(dest)).parent).encode()).hexdigest()[:10]
    return AWORK / "crashes" / f"{h}_{Path(dest).name}.{arm}.json"


def note_crash(dest, arm, rc, log_path):
    p = crash_path(dest, arm)
    J = rj(p) or dict(dest=str(dest), arm=arm, n=0, events=[])
    J["n"] = int(J.get("n") or 0) + 1
    J["events"] = (J.get("events") or [])[-5:] + [dict(rc=rc, job=jobtag(), host=_socket.gethostname(), time=now(),
                                                       log=str(log_path))]
    jdump(p, J)
    return J["n"]


def alloc_path(list_spec, arm):
    return AWORK / "alloc" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', list_spec)}_{arm}.jsonl"


def read_alloc(list_spec, arm):
    """A2-cost (allocated): {job_id: last record} of the per-task allocation log."""
    out = {}
    p = alloc_path(list_spec, arm)
    if p.is_file():
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
                out[r["job_id"]] = r
            except (ValueError, KeyError):
                continue
    return out


def measured_workers():
    """A2-workers: A-3's measured recommendation (A3_report est.workers), else WORKERS_CAP."""
    w = ((rj(REPORTS / "A3_report.json") or {}).get("est") or {}).get("workers")
    try:
        return max(1, int(w)) if w else WORKERS_CAP
    except (TypeError, ValueError):
        return WORKERS_CAP


def cmd_run(list_spec, i="0", n="1", model=None, engine=None, list_path=None, out=None, workers=None, cores=None,
            freq=None):
    ensure_env("mlts")
    import mlts_bridge as MB
    i, n = M.array_task(i), int(n)
    freq = freq or FREQ_DEFAULT
    arms = arms_of(engine, list_spec)
    main_engine = "App" if "App" in arms else None
    run_arms = sorted({("A" if a == "App" else a) for a in arms}, key=lambda a: ("A", "Ap").index(a))
    if list_spec != "a3" and len(run_arms) > 1:
        raise SystemExit(f"run {list_spec}: one engine per out_root ({arms}); only the a3 list runs A and Ap together")
    rows = slice_rows(resolve_list(list_spec, list_path), i, n)
    tasks = []
    for r in rows:
        for a in run_arms:
            root = out_root_for(list_spec, a, out, main_engine)
            tasks.append(dict(row=r, arm=a, root=root, model=model_for(r, model)))
    specs = {}
    for t in tasks:
        if t["model"] not in specs:
            try:
                specs[t["model"]] = MB.resolve_spec(t["model"])
            except Exception as e:                               # noqa: BLE001
                M.not_ready(f"model {t['model']}: {e}")
            w = specs[t["model"]].get("weights")
            if w and not Path(os.path.expandvars(str(w))).is_file():
                M.not_ready(f"model {t['model']}: weights {w} missing")
    todo = [t for t in tasks if not is_final(t["root"] / "jobs" / t["row"]["job_id"])]
    todo.sort(key=lambda t: (t["model"], -t["row"]["n_atoms"], t["row"]["rxn_id"], t["arm"]))
    cores = int(cores or CORES_DEFAULT)
    nw = int(workers or os.environ.get("MLTS_WORKERS") or min(max(1, (cpus() - 1) // cores), measured_workers()))
    nw = max(1, min(nw, len(todo) or 1))
    tag = jobtag()
    logdir = AWORK / "logs" / re.sub(r"[^A-Za-z0-9_.-]", "_", list_spec)
    summ_p = AWORK / "slices" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', list_spec)}_{'-'.join(arms)}_{i:02d}of{n:02d}.json"
    log(f"run {list_spec} slice {i}/{n}: {len(rows)} reactions, arms {arms} ({run_arms} here), {len(tasks)} tasks, "
        f"{len(todo)} to do, {nw} workers x {cores} cores, freq {freq}, models {sorted(specs)}, tag {tag}, "
        f"{time_left_h():.1f} h left")
    left = Counter(t["model"] for t in todo)
    srv = Servers(tag, AWORK / "logs")
    stop = threading.Event()
    mon = threading.Thread(target=srv.monitor, args=(stop,), daemon=True)
    mon.start()
    mu = threading.Lock()
    cv = threading.Condition(mu)
    pos = dict(i=0)
    running = Counter()
    one_server = os.environ.get("MLTS_ONE_SERVER", "1") != "0"      # A2-one-server (24 GB cards)
    counts, per, errs = Counter(), [], []
    st = dict(halt="", timed_out=False, server_dead=False)
    t0 = time.time()
    import signal

    def _term(*_):                                               # A2-term: stop claiming, then exit (finally: servers)
        st["halt"] = st["halt"] or "SIGTERM"
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _term)

    def claim():
        """next task; with one_server a task of another model waits until the running tasks of the current model are
        done (one MLTS server on the GPU at a time: two OMOL-size servers can exhaust a 24 GB card in a Hessian)."""
        with cv:
            while True:
                if st["halt"]:
                    return None
                if time_left_h() < RESERVE_H:
                    st["timed_out"] = True
                    return None
                if pos["i"] >= len(todo):
                    return None
                t = todo[pos["i"]]
                if one_server and any(c > 0 and m != t["model"] for m, c in running.items()):
                    cv.wait(timeout=30)
                    continue
                pos["i"] += 1
                running[t["model"]] += 1
                return t

    def one(t):
        T, a, name = t["row"]["job_id"], t["arm"], t["model"]
        dest = t["root"] / "jobs" / T
        if is_final(dest):
            return "skip"
        try:
            sock = srv.get(name, specs[name])
        except RuntimeError as e:                                # server died MAX_SERVER_RESTARTS times: infra
            with mu:
                st["server_dead"] = True
                st["halt"] = st["halt"] or f"MLTS server {name}: {e}"
            return "error_server"
        lp = logdir / f"{T}.{a}.log"
        M.assert_write_ok(lp)
        lp.parent.mkdir(parents=True, exist_ok=True)
        args = ["one", T, "--model", json.dumps(specs[name]), "--engine", a, "--out", str(t["root"]), "--cores",
                str(cores), "--socket", sock, "--tag", tag, "--freq", freq]
        env = dict(os.environ, SLURM_CPUS_PER_TASK=str(cores), OMP_NUM_THREADS=str(cores))
        env.pop("MLTS_A_REEXEC", None)
        tt = time.time()
        with open(lp, "a") as fo:
            fo.write(f"=== {now()} {a} {T} model {name} job {tag}\n")
            fo.flush()
            p = subprocess.Popen(env_cmd("d1", args), stdout=fo, stderr=subprocess.STDOUT, env=env)
            try:
                rc = p.wait(timeout=RXN_TIMEOUT_H * 3600)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                _publish_timeout(run_dir(tag, T, arm=a), dest, T, f"> {RXN_TIMEOUT_H} h")
                rc = "timeout"
        wall_p = time.time() - tt
        if not is_final(dest) and rc not in KILLED_RC and not st["halt"]:
            n_cr = note_crash(dest, a, rc, lp)                   # A2-crash: the 2nd unpublished exit closes it
            if n_cr >= 2:
                _publish_timeout(run_dir(tag, T, arm=a), dest, T, f"worker rc {rc}, {n_cr} crashes (log {lp})",
                                 kind="pipeline_error")
        if is_final(dest):
            s, why = stage_state(dest)
            tim = rj(dest / "timing.json") or {}
            rec = dict(job_id=T, arm=a, model=name, status=s, reason=why, wall_s=tim.get("wall_s"),
                       gpu_s=tim.get("gpu_s"), n_atoms=t["row"]["n_atoms"], wall_parent_s=wall_p,
                       alloc_core_h=wall_p * cpus() / max(1, nw) / 3600.0, alloc_gpu_min=wall_p / max(1, nw) / 60.0,
                       workers=nw, cpus=cpus(), job=tag, time=now())
            with mu:
                per.append(rec)
                ap = alloc_path(list_spec, a)
                M.assert_write_ok(ap)
                ap.parent.mkdir(parents=True, exist_ok=True)
                _append(ap, json.dumps(rec, default=M._json_default) + "\n")
            return s
        with mu:
            errs.append(dict(job_id=T, arm=a, rc=rc, log=str(lp)))
        return f"error_rc{rc}"

    def loop():
        while True:
            t = claim()
            if t is None:
                return
            try:
                s = one(t)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                s = f"error_{type(e).__name__}"
                with mu:
                    errs.append(dict(job_id=t["row"]["job_id"], arm=t["arm"], error=f"{type(e).__name__}: {e}"[:300]))
            with cv:
                counts[s] += 1
                left[t["model"]] -= 1
                running[t["model"]] -= 1
                if s.startswith("error"):
                    st["halt"] = st["halt"] or ("" if len(errs) <= MAX_PIPELINE_ERR else
                                                f"more than {MAX_PIPELINE_ERR} pipeline errors")
                if left[t["model"]] == 0:                        # released before the next model's tasks start
                    srv.release(t["model"])
                cv.notify_all()
            log(f"{t['row']['job_id']} {t['arm']}: {s} ({sum(counts.values())}/{len(todo)}, {time.time() - t0:.0f} s)")
    try:
        with ThreadPoolExecutor(nw) as ex:
            for f in [ex.submit(loop) for _ in range(nw)]:
                f.result()
    finally:
        stop.set()
        srv.close()
    wall = time.time() - t0
    rest = [t for t in tasks if not is_final(t["root"] / "jobs" / t["row"]["job_id"])]
    busy = sum(float(v.get("eval_s") or 0) + float(v.get("hess_s") or 0) for v in srv.stats.values())
    g = [p["gpu_s"] for p in per if p.get("gpu_s")]
    w = [p["wall_s"] for p in per if p.get("wall_s")]
    ratio = sorted(float(a) / float(b) for a, b in zip(w, g) if b) if g else []
    rec_w = max(1, int(0.7 * ratio[len(ratio) // 2])) if ratio else None
    summ = dict(list=list_spec, slice=i, n_slices=n, arms=arms, run_arms=run_arms, model_override=model, freq=freq,
                n_rows=len(rows), n_tasks=len(tasks), n_todo_at_start=len(todo), counts=dict(counts), errors=errs[:50],
                n_left=len(rest), workers=nw, cores_per_worker=cores, cpus=cpus(), wall_s=wall, server=srv.stats,
                server_restarts=dict(srv.restarts), gpu_busy_s=busy, gpu_busy_frac=busy / wall if wall else None,
                wall_over_gpu_median=ratio[len(ratio) // 2] if ratio else None, recommended_workers=rec_w,
                per_task=per, halt=st["halt"], timed_out=st["timed_out"], host=_socket.gethostname(), job=tag,
                finished=now(), code=CODE_VERSION, DECISIONS=DECISIONS)
    jdump(summ_p, summ)
    log(f"run {list_spec} {i}/{n} done: {dict(counts)}, {len(rest)} left, GPU busy {busy:.0f} s of {wall:.0f} s "
        f"wall, recommended workers {rec_w}")
    if st["halt"]:
        return 1 if st["server_dead"] else 3                    # a server that keeps dying = infra (launcher retry)
    return 0 if not rest else 2


def cmd_one(job_id, model, engine, out, cores, sock, tag, freq, job_json=None):
    """internal worker (D1 env): one reaction (A or Ap)."""
    if not have_autode():
        ensure_env("d1")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))      # finally blocks run (partial wall kept for a resume)
    job = json.loads(Path(job_json).read_text()) if job_json else manifest_rows([job_id]).get(job_id)
    if job is None:
        raise SystemExit(f"{job_id} not in {MANIFEST}")
    MB = d1()["MB"]
    spec = MB.resolve_spec(json.loads(model) if str(model).strip().startswith("{") else model)
    t = time.time()
    p = _work_one(job, spec, engine, Path(out), int(cores), sock, tag, freq)
    s, why = stage_state(p)
    log(f"{job_id} {engine}: {s} {why} ({time.time() - t:.0f} s) -> {p}")
    return 0


def run_one(rxn_smiles, job_id, model, engine="A", out_root=None, cores=4, socket=None, job_row=None, tag=None,
            freq=None):
    """contract API: one reaction -> Path(<out_root>/jobs/<job_id>) in the A-2 layout. engine A or Ap (App: use
    refine_dft on A's folder). In the D1 env it runs in this process (autodE Config is global: one reaction at a
    time); elsewhere it runs `one` in the D1 env. Without socket a server for model is started for this call."""
    out_root = Path(out_root or ADE_MAIN)
    job = dict(job_row) if job_row else manifest_rows([job_id]).get(job_id)
    if job is None:
        raise ValueError(f"{job_id}: no manifest row (pass job_row for a reaction outside the D1 manifest)")
    if rxn_smiles and str(job.get("rxn_smiles")) != str(rxn_smiles):
        raise ValueError(f"{job_id}: rxn_smiles differs from the manifest row")
    import mlts_bridge as MB
    spec = MB.resolve_spec(model)
    tag = tag or jobtag()
    h = None
    if socket is None:                                           # A2-run-one: a socket of this call only (never shared)
        import secrets
        own = f"{tag}_ro{os.getpid()}_{secrets.token_hex(3)}"
        h = _spawn_server(spec, socket_path(own, MB.spec_name(spec)), AWORK / "logs" / f"server_{own}_run_one.log")
        socket = h.socket
    try:
        if have_autode() and os.environ.get("MLTENV") == "d1":
            return _work_one(job, spec, engine, out_root, int(cores), socket, tag, freq or FREQ_DEFAULT)
        jp = local_base() / f"mlts_{tag}_{job_id}.job.json"
        jp.write_text(json.dumps(job))
        args = ["one", job_id, "--model", json.dumps(spec), "--engine", engine, "--out", str(out_root), "--cores",
                str(cores), "--socket", str(socket), "--tag", tag, "--freq", freq or FREQ_DEFAULT, "--job-json", str(jp)]
        subprocess.run(env_cmd("d1", args), env=dict(os.environ, SLURM_CPUS_PER_TASK=str(cores)))
        jp.unlink(missing_ok=True)
        return out_root / "jobs" / job_id
    finally:
        if h is not None:
            h.shutdown()


# ================================================================ A'': refine_dft (D1 env, CPU)
_G3 = {}


def d1_g3():
    """(D, opt, g3l) of analysis/d1_g3 with D1G3_SCRATCH = a shadow under $MLT (App-input)."""
    if _G3:
        return _G3["D"], _G3["O"], _G3["L"]
    shadow = AWORK / "d1g3_shadow"
    os.environ["D1G3_SCRATCH"] = str(shadow)
    p = str(M.ANALYSIS / "d1_g3")
    if p not in sys.path:
        sys.path.insert(1, p)
    import g3l as G3L
    O = G3L.O
    if Path(os.path.abspath(G3L.C.G3)) != Path(os.path.abspath(shadow)):
        M.stop(f"d1_g3 G3 root {G3L.C.G3} is not the shadow {shadow}")
    if int(O.PARTB_NPROCS) != APP_NPROCS:
        M.stop(f"opt.PARTB_NPROCS {O.PARTB_NPROCS} != {APP_NPROCS}")
    _G3.update(D=O.d1(), O=O, L=G3L)
    return _G3["D"], _G3["O"], _G3["L"]


RE_FREQ_OUT = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", re.M)
RE_TIMING = re.compile(r"^(.{3,60}?)\s+\.\.\.\s+(\d+\.\d+)\s+sec", re.M)


def app_unit(row, job, src, dest, tag, D, O, G3L, cfg_base):
    """A'' of one reaction -> dest/jobs/<T> (A-2 layout). Returns the status string."""
    Dd = d1()
    np, pc, RT, OD, MB = Dd["np"], Dd["pc"], Dd["RT"], Dd["OD"], Dd["MB"]
    T = row["job_id"]
    s, out = Path(src) / "jobs" / T, Path(dest) / "jobs" / T
    if is_final(out):
        return "skip"
    if not is_final(s):
        return "no_src"
    local = run_dir(tag, T, "mlts_app")
    if local.exists():
        shutil.rmtree(local)
    cfg = dict(cfg_base, scratch=str(local))
    cfg_path = local / "config_mlts.yaml"
    jd = local / "jobs" / T
    jd.mkdir(parents=True)
    cfg_path.write_text(Dd["yaml"].safe_dump(cfg))
    manifest = local / "manifest.csv"
    write_manifest(manifest, job)
    a_tim = rj(s / "timing.json") or {}
    tim = dict(job_id=T, engine="App", source=str(s), host=_socket.gethostname(), job=tag, started=now(),
               code=CODE_VERSION, a_core_h=a_tim.get("core_h"), a_gpu_s=a_tim.get("gpu_s"), a_wall_s=a_tim.get("wall_s"))
    t_all = time.time()
    try:
        state, why = stage_state(s)
        keep = [p for p in s.iterdir() if p.is_file() and p.name not in ("timing.json", "inputs.tar.gz")]
        if state != "ok":
            for p in keep:
                shutil.copy2(p, jd / p.name)
            tim.update(status="fail", reason=f"a_{why}", core_h=float(a_tim.get("core_h") or 0.0),
                       gpu_s=float(a_tim.get("gpu_s") or 0.0), wall_s=0.0, dft_core_h=0.0, n_dft_cycles=0)
            local_json(jd / "timing.json", tim)
            copy_out(jd, out)
            return "fail_a"
        for p in keep:
            if p.name in ("ref_dipole.xyz", "ref_dipolarophile.xyz", "ref_result.json", ".done_ref",
                          "r_dipole_autode.xyz", "r_dipolarophile_autode.xyz", "energies.csv", "methods.txt"):
                shutil.copy2(p, jd / p.name)
        syms, xyz = pc.read_xyz(s / "ts.xyz")
        tsr = json.loads((s / "ts_result.json").read_text())
        pairs = [tuple(int(x) for x in p) for p in tsr["formed_pairs_ts"]]
        q = int(job["charge"])
        hs = s / "ts.hess"
        hh = MB.OI.read_hess(hs)
        dmax = float(np.abs(np.asarray(hh["pos_bohr"], float) * M.BOHR2A - np.asarray(xyz, float)).max())
        if list(hh["symbols"]) != list(syms) or dmax > 1e-4:
            raise RuntimeError(f"A's ts.hess does not belong to ts.xyz (max |dx| {dmax:.2e} A)")
        text = G3L.g3l_text(D, list(syms), np.asarray(xyz, float), q, [list(p) for p in pairs])[0]
        work = local / "dft"
        runs, txt = [], ""
        for k in (1, 2):
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            (work / "ts.inp").write_text(text)
            shutil.copyfile(hs, work / G3L.HESS_NAME)
            t0 = time.time()
            with open(work / "ts.out", "w") as fo, open(work / "ts.err", "w") as fe:
                p = subprocess.run([str(D["orca"]), "ts.inp"], cwd=str(work), stdout=fo, stderr=fe, env=D["env"])
            wall = time.time() - t0
            txt = (work / "ts.out").read_text(errors="replace")
            term = "ORCA TERMINATED NORMALLY" in txt
            infra = (not term) and (p.returncode < 0 or not txt.strip()
                                    or ("SCF ITERATIONS" not in txt and "GEOMETRY OPTIMIZATION CYCLE" not in txt))
            runs.append(dict(run=k, rc=p.returncode, wall_s=wall, core_h=wall * APP_NPROCS / 3600, infra=infra))
            if not infra:
                break
        dft_ch = sum(r["core_h"] for r in runs)
        term = "ORCA TERMINATED NORMALLY" in txt
        conv = bool(OD.opt_converged(txt))
        ih = G3L.inhess_evidence(txt)
        n_cyc = txt.count("GEOMETRY OPTIMIZATION CYCLE")
        n_grad = txt.count("CARTESIAN GRADIENT")
        tmg = {m.group(1).strip(): float(m.group(2)) for m in RE_TIMING.finditer(txt.rsplit("Timings for individual modules", 1)[-1])} \
            if "Timings for individual modules" in txt else {}
        app = dict(runs=runs, terminated=term, converged=conv, n_opt_cycles=n_cyc, n_gradients=n_grad, inhess=ih,
                   module_timings_s=tmg, header=text.split("*xyz")[0], hess_source=str(hs), nprocs=APP_NPROCS)
        res = dict(job_id=T, kind=job["kind"], ts_id=job.get("ts_id"), rxn_smiles=job["rxn_smiles"], engine="orca",
                   arm="App", level="A'': A's MLTS TS -> label-level ORCA OptTS (G3L input, InHess = MLTS Hessian)",
                   code=CODE_VERSION, a_source=str(s), app=app)
        ok = False
        try:
            if runs[-1]["infra"]:
                raise RT.StageFail("app_pipeline_infra")
            if ih.get("not_read"):
                raise RT.StageFail("app_inhess_not_read")
            if not conv:
                raise RT.StageFail("app_optts_not_converged" if term else "app_orca_not_terminated")
            if not (work / "ts.hess").is_file() or not (work / "ts.xyz").is_file():
                raise RT.StageFail("app_no_freq")
            s2, x2 = pc.read_xyz(work / "ts.xyz")
            if list(s2) != list(syms):
                raise RT.StageFail("app_atom_order_changed")
            imag, mode = OD.imag_and_mode(work / "ts.hess")
            reacs = [pc.read_xyz(s / "r_dipole_autode.xyz"), pc.read_xyz(s / "r_dipolarophile_autode.xyz")]
            fr, _ = pc.import_repo_modules(cfg)
            got = dict(ts=(list(syms), np.asarray(x2, float)), reacs=reacs, prod=None, imag=imag, mode=mode)
            RT.analyse(cfg, job, jd, res, got, fr)
            ok = True
        except RT.StageFail as e:
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", str(e))
        if ok:
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_done(jd, "ts")
            shutil.copy(work / "ts.hess", jd / "ts.hess")
            if not stage_stereo(cfg_path, manifest, job, jd):
                ok = False
            elif not (jd / ".done_ref").is_file():
                pc.mark_fail(jd, "ref", "a_ref_missing")
        shutil.copy(work / "ts.inp", jd / "app_ts.inp")
        wall = time.time() - t_all
        tim.update(status="ok" if all(pc.done(jd, x) for x in STAGES) else "fail", wall_s=wall, dft_core_h=dft_ch,
                   n_dft_cycles=n_cyc, n_dft_gradients=n_grad, dft_converged=conv,
                   core_h=dft_ch + float(a_tim.get("core_h") or 0.0), gpu_s=float(a_tim.get("gpu_s") or 0.0),
                   n_ext_calls=int(a_tim.get("n_ext_calls") or 0), finished=now())
        local_json(jd / "timing.json", tim)
        copy_out(jd, out, [work / "ts.out"] if not ok else [])
        return tim["status"]
    finally:
        if os.environ.get("MLTS_KEEP_LOCAL") != "1" and is_final(out):
            shutil.rmtree(local, ignore_errors=True)


def cmd_refine_dft(list_spec, i="0", n="1", root=None, src=None, list_path=None):
    ensure_env("d1")
    i, n = M.array_task(i), int(n)
    srcr, dest = app_roots(list_spec, root, src)
    rows = slice_rows(resolve_list(list_spec, list_path), i, n)
    man = manifest_rows([r["job_id"] for r in rows])
    D, O, G3L = d1_g3()
    d1()
    local0 = local_base()
    cfg_base = base_config(local0)
    slots = max(1, cpus() // APP_NPROCS)
    tag = jobtag()
    todo = [r for r in rows if not is_final(dest / "jobs" / r["job_id"])]
    log(f"refine_dft {list_spec} slice {i}/{n}: {len(rows)} reactions, {len(todo)} to do, {slots} slots x {APP_NPROCS} "
        f"ranks, {srcr} -> {dest}, {time_left_h():.1f} h left")
    mu = threading.Lock()
    it = iter(todo)
    cnt, errs = Counter(), []
    st = dict(timed_out=False)
    t0 = time.time()

    def loop():
        while True:
            with mu:
                if time_left_h() < RESERVE_H:
                    st["timed_out"] = True
                    return
                r = next(it, None)
            if r is None:
                return
            try:
                s = app_unit(r, man[r["job_id"]], srcr, dest, tag, D, O, G3L, cfg_base)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                s = f"error_{type(e).__name__}"
                with mu:
                    errs.append(dict(job_id=r["job_id"], error=f"{type(e).__name__}: {e}"[:300]))
            with mu:
                cnt[s] += 1
            log(f"refine_dft {r['job_id']}: {s} ({sum(cnt.values())}/{len(todo)}, {time.time() - t0:.0f} s)")
    with ThreadPoolExecutor(slots) as ex:
        for f in [ex.submit(loop) for _ in range(slots)]:
            f.result()
    rest = [r for r in rows if not is_final(dest / "jobs" / r["job_id"])]
    jdump(AWORK / "slices" / f"refine_{re.sub(r'[^A-Za-z0-9_.-]', '_', list_spec)}_{i:02d}of{n:02d}.json",
          dict(list=list_spec, slice=i, n_slices=n, src=str(srcr), dest=str(dest), counts=dict(cnt), errors=errs[:50],
               n_left=len(rest), slots=slots, wall_s=time.time() - t0, timed_out=st["timed_out"], finished=now(),
               job=tag, DECISIONS={k: DECISIONS[k] for k in ("App-input", "A2-cost")}))
    log(f"refine_dft {list_spec} {i}/{n}: {dict(cnt)}, {len(rest)} left")
    if len(errs) > MAX_PIPELINE_ERR:
        return 3
    return 0 if not rest else 2


# ================================================================ reports
def du_gb(p):
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return tot / 1e9


def _median(v):
    return _pct(v, 50)


def _pct(v, q):
    """numpy-default (linear) percentile of the non-None values (None if empty)."""
    v = sorted(float(x) for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _dist(pos, p):
    import numpy as np
    return float(np.linalg.norm(np.asarray(pos[p[0]], float) - np.asarray(pos[p[1]], float)))


def pair_dd(lab, arm, lab_pairs, arm_pairs):
    """max over the label forming pairs of |d_arm - d_label| with the arm's atoms mapped onto the label order
    (A3-metrics); (dd, how)."""
    import numpy as np
    ls, lp = lab
    as_, ap = arm
    lpp = sorted(tuple(sorted(p)) for p in lab_pairs)
    app_ = sorted(tuple(sorted(p)) for p in arm_pairs)
    if list(ls) == list(as_) and lpp == app_:
        return max(abs(_dist(ap, p) - _dist(lp, p)) for p in lpp), "direct"
    rm = d1()["MB"].rmsd_mod()
    fr = rm.fr
    from networkx.algorithms.isomorphism import GraphMatcher
    a = (list(ls), np.asarray(lp, float))
    b = (list(as_), np.asarray(ap, float))
    Ga, Gb = fr.heavy_graph(*a)[0], fr.heavy_graph(*b)[0]
    best, bm = np.inf, None
    for k, m in enumerate(GraphMatcher(Ga, Gb, node_match=lambda x, y: x["lab"] == y["lab"]).isomorphisms_iter()):
        ia = list(m)
        ib = [m[x] for x in ia]
        for mirror in (1.0, -1.0):
            r = rm.kabsch_rmsd(a[1][ia], b[1][ib] * np.array([mirror, 1.0, 1.0]))
            if r < best:
                best, bm = r, dict(m)
        if k + 1 >= 5000:
            break
    if bm is None:
        return None, "no_map"
    inv = {v: k for k, v in bm.items()}
    mapped = sorted(tuple(sorted((inv.get(i), inv.get(j)))) for i, j in app_ if i in inv and j in inv)
    if mapped != lpp:
        return None, "pairs_differ"
    return max(abs(_dist(ap, (bm[i], bm[j])) - _dist(lp, (i, j))) for i, j in lpp), "mapped"


def arm_record(row, arm, folder, labs, alloc=None):
    """A-3 / E record of one reaction and arm. alloc (A2-cost allocated): {arm: {job_id: alloc record}} of the GPU
    jobs (A, Ap); A'' adds A's allocated cost to its DFT core-h."""
    rec = dict(rxn_id=row["rxn_id"], job_id=row["job_id"], fold_T1=row["fold_T1"], dipole=row["dipole"],
               n_atoms=row["n_atoms"], arm=arm)
    if not is_final(folder):
        rec.update(status="missing")
        return rec
    s, why = stage_state(folder)
    tim = rj(folder / "timing.json") or {}
    ch = tim.get("core_h_deploy") if arm == "Ap" else tim.get("core_h")
    rec.update(status=s, reason=why, core_h=ch, core_h_own=tim.get("core_h"),
               gpu_min=(float(tim["gpu_s"]) / 60.0) if tim.get("gpu_s") is not None else None,
               wall_s=tim.get("wall_s"), n_ext_calls=tim.get("n_ext_calls"), n_hess=tim.get("n_hess"),
               disk_gb=du_gb(folder), n_dft_cycles=tim.get("n_dft_cycles"), n_dft_gradients=tim.get("n_dft_gradients"),
               dft_core_h=tim.get("dft_core_h"))
    # A2-cost: pre-start failures (A' without a G2 TS, A'' whose A failed) cost nothing of this arm: not in the medians
    rec["pre_start_fail"] = bool(s == "fail" and ((arm == "Ap" and str(why).startswith("ts_no_g2_ts"))
                                                   or (arm == "App" and str(tim.get("reason") or "").startswith("a_"))))
    al = (alloc or {}).get("A" if arm == "App" else arm, {}).get(row["job_id"])
    if al:
        ac, ag = float(al.get("alloc_core_h") or 0.0), float(al.get("alloc_gpu_min") or 0.0)
        if arm == "Ap":
            ac += float(tim.get("g2_core_h_ts") or 0.0) + float(tim.get("g2_core_h_stereo") or 0.0)
        elif arm == "App":
            ac += float(tim.get("dft_core_h") or 0.0)
        rec.update(core_h_alloc=ac, gpu_min_alloc=ag)
    if s != "ok":
        return rec
    pc, MB = d1()["pc"], d1()["MB"]
    ts = pc.read_xyz(folder / "ts.xyz")
    tsr = rj(folder / "ts_result.json") or {}
    own = [tuple(int(x) for x in p) for p in tsr.get("formed_pairs_ts") or []]
    lab_ts = M.label_ts(row["rxn_id"])
    lp = M.formed_pairs(labs[row["rxn_id"]])
    if not lab_ts.is_file() or not own:
        rec.update(compare="no_label_ts" if not lab_ts.is_file() else "no_pairs")
        return rec
    L = pc.read_xyz(lab_ts)
    d_lab = [_dist(L[1], p) for p in lp]
    d_arm = [_dist(ts[1], p) for p in own]
    same, rmsd, dform = MB.same_ts((list(L[0]), L[1]), (list(ts[0]), ts[1]), d_lab, d_arm)
    dd, how = pair_dd(L, ts, lp, own)
    imag = [float(v) for v in tsr.get("imag_freqs_cm") or []]
    rec.update(same_ts=bool(same), rmsd_heavy=rmsd, dform_sorted=dform, dd_max=dd, dd_how=how,
               n_imag=sum(1 for v in imag if v < -50.0), imag_freqs_cm=imag[:3], d_label=d_lab, d_arm=d_arm)
    return rec


def arm_summary(recs, arm):
    """cost medians (A2-cost) over the attempted reactions (pre-start failures out) on the ALLOCATED cost (the GPU
    job's cores and GPU time shared over its workers: what deployment pays) when recorded, else the reaction's own
    cost; the own medians are reported next to them."""
    n = len(recs)
    fin = [r for r in recs if r["status"] in ("ok", "fail")]
    att = [r for r in fin if not r.get("pre_start_fail")]
    ok = [r for r in recs if r["status"] == "ok"]
    same = [r for r in ok if r.get("same_ts")]
    reasons = Counter(r.get("reason") for r in recs if r["status"] == "fail")
    ch = _median((r.get("core_h_alloc") if r.get("core_h_alloc") is not None else r.get("core_h")) for r in att)
    gm = _median((r.get("gpu_min_alloc") if r.get("gpu_min_alloc") is not None else r.get("gpu_min")) for r in att)
    b = A3_BUDGET[arm]
    return dict(n=n, n_finished=len(fin), n_attempted=len(att), n_pre_start_fail=len(fin) - len(att), n_ok=len(ok),
                n_missing=n - len(fin), cost_basis="allocated" if any(r.get("core_h_alloc") is not None for r in att)
                else "own", gpu_min_own_median=_median(r.get("gpu_min") for r in att),
                success=len(ok) / n if n else None, same_ts_frac=len(same) / n if n else None,
                same_ts_frac_of_ok=len(same) / len(ok) if ok else None,
                dd_median=_median(r.get("dd_max") for r in same), dd_median_all_ok=_median(r.get("dd_max") for r in ok),
                dd_p90_all_ok=_pct([r.get("dd_max") for r in ok], 90),
                rmsd_median=_median(r.get("rmsd_heavy") for r in same),
                rmsd_median_all_ok=_median(r.get("rmsd_heavy") for r in ok),
                nimag1_frac_of_ok=(sum(1 for r in ok if r.get("n_imag") == 1) / len(ok)) if ok else None,
                core_h_median=ch, gpu_min_median=gm, wall_median=_median(r.get("wall_s") for r in att),
                core_h_own_median=_median(r.get("core_h") for r in att),
                dft_cycles_median=_median(r.get("n_dft_cycles") for r in fin if r.get("n_dft_cycles")),
                dft_gradients_median=_median(r.get("n_dft_gradients") for r in fin if r.get("n_dft_gradients")),
                disk_gb_per_rxn=(sum(r.get("disk_gb") or 0 for r in fin) / len(fin)) if fin else None,
                fail_reasons=dict(reasons.most_common(20)),
                in_budget=(ch is not None and gm is not None and ch <= b[0] and gm <= b[1]), budget=dict(core_h=b[0],
                                                                                                          gpu_min=b[1]))


def slice_summaries(prefix):
    out = []
    d = AWORK / "slices"
    if d.is_dir():
        for p in sorted(d.glob(f"{prefix}*.json")):
            j = rj(p)
            if j:
                out.append(j)
    return out


def cmd_a3_report(list_path=None):
    ensure_env("d1")
    rows = resolve_list("a3", list_path)
    T = targets()
    labs = {int(r.rxn_id): str(r.formed_pairs_ts) for r in T.itertuples()}
    arms, recs_all = {}, {}
    alloc = {a: read_alloc("a3", a) for a in ("A", "Ap")}
    for arm in ARMS:
        root = A3_ROOT / arm / "ade" / "jobs"
        recs = [arm_record(r, arm, root / r["job_id"], labs, alloc) for r in rows]
        recs_all[arm] = recs
        arms[arm] = arm_summary(recs, arm)
    n_todo = sum(a["n_missing"] for a in arms.values())
    sl = slice_summaries("a3_")
    rw = [s.get("recommended_workers") for s in sl if s.get("recommended_workers")]
    workers = int(_median(rw)) if rw else WORKERS_CAP
    walls = {a: arms[a]["wall_median"] for a in ("A", "Ap")}
    unit_wall_h = max([w for w in walls.values() if w is not None] or [0.3 * 3600]) / 3600.0
    order = sorted(ARMS, key=lambda a: (-(arms[a]["same_ts_frac"] or 0), arms[a]["core_h_median"] or 1e9))
    rep = dict(phase="A-3 (spec 6)", complete=n_todo == 0, n_todo=n_todo, stop=None, n_list=len(rows),
               list=str(a3_list_path(list_path)), arms=arms,
               disk_gb_per_rxn=arms["A"]["disk_gb_per_rxn"],
               rule_inputs=dict(budget=A3_BUDGET, stop_below=0.70, limit_below=0.85, tie_same=0.025,
                                app_extra_arm_core_h=1.0, ranking_same_then_cost=order,
                                in_budget={a: arms[a]["in_budget"] for a in ARMS},
                                same_ts_frac={a: arms[a]["same_ts_frac"] for a in ARMS}),
               est=dict(units_total=N_D1, unit_wall_h=unit_wall_h, workers=workers, cores_per_worker=CORES_DEFAULT,
                        unit_wall_h_by_arm={a: (walls[a] / 3600.0 if walls[a] else None) for a in walls},
                        slices=[dict(slice=s.get("slice"), gpu_busy_frac=s.get("gpu_busy_frac"),
                                     wall_over_gpu_median=s.get("wall_over_gpu_median"), workers=s.get("workers"))
                                for s in sl]),
               cost_rule=DECISIONS["A2-cost"], metrics_rule=DECISIONS["A3-metrics"], written=now(), code=CODE_VERSION)
    M.write_csv(REPORTS / "A3_status.csv", __import__("pandas").DataFrame(
        [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
         for a in ARMS for r in recs_all[a]]))
    jdump(REPORTS / "A3_report.json", rep)
    log("A3: " + "; ".join(f"{a} n={arms[a]['n']} ok={arms[a]['n_ok']} same={arms[a]['same_ts_frac']} "
                           f"core_h={arms[a]['core_h_median']} gpu_min={arms[a]['gpu_min_median']}" for a in ARMS))
    return 0 if rep["complete"] else 2


def cmd_status(root=None):
    R = Path(root) if root else BASE
    jobs = R / "ade" / "jobs"
    rep_dir = REPORTS if (root is None or Path(os.path.abspath(R)) == Path(os.path.abspath(M.MLT))) and IS_PROD \
        else R / "reports"
    T = targets()
    folds = {}
    n_todo = 0
    for k in sorted(int(x) for x in T.fold_T1.unique()):
        S = T[T.fold_T1 == k]
        c, rs = Counter(), Counter()
        for job in S.job_id:
            d = jobs / str(job)
            if not is_final(d):
                c["missing"] += 1
                continue
            s, why = stage_state(d)
            c["ok" if s == "ok" else "fail"] += 1
            if s != "ok":
                rs[why] += 1
        n_todo += c["missing"]
        folds[str(k)] = dict(complete=c["missing"] == 0, n=int(len(S)), n_ok=c["ok"], n_fail=c["fail"],
                             n_missing=c["missing"], reasons=dict(rs.most_common(30)))
    rep = dict(phase="A-4 (spec 6)", root=str(R), complete=n_todo == 0, n_todo=n_todo, stop=None, folds=folds,
               n_ok=sum(f["n_ok"] for f in folds.values()), n_fail=sum(f["n_fail"] for f in folds.values()),
               est=dict(units_total=int(len(T))), written=now(), code=CODE_VERSION)
    jdump(rep_dir / "A4_report.json", rep)
    log(f"status {R}: ok {rep['n_ok']}, fail {rep['n_fail']}, missing {n_todo}")
    return 0 if rep["complete"] else 2


# ================================================================ smoke / C-3 input check
BANNED = re.compile(r"(?i)(?<![A-Za-z0-9])(B3LYP\S*|PBE0?|BP86|BLYP|TPSS\w*|r2SCAN\S*|wB97\S*|M06\S*|HF-3c|RHF|UHF|HF|"
                    r"DFT|XTB\d?|GFN\d?\S*|def2[-/]\S+|6-31\S*|cc-pV\S+|SVP|TZVP\S*|RIJCOSX|RIJK|RI-J|D3BJ|D3|D4|"
                    r"CPCM\S*|C-PCM|SMD\S*|ALPB\S*|GBSA\S*|COSMO\S*)(?![A-Za-z0-9])")
NEED = re.compile(r"(?im)^!.*(?<![A-Za-z0-9])ExtOpt(?![A-Za-z0-9])")


def inp_check_text(text):
    """C-3 on one ORCA input: (ok, banned words, has ExtOpt). Quoted strings (ProgExt / Ext_Params paths) and the
    coordinate block are not words of the method."""
    head = text.split("*xyz")[0]
    head = re.sub(r'"[^"]*"', '""', head)
    bad = sorted({m.group(1) for m in BANNED.finditer(head)})
    ext = bool(NEED.search(head))
    return (not bad and ext), bad, ext


def c3_check(folders, n=50, seed=None):
    """C-3: up to n random ORCA inputs (.inp whose first line starts with '!') out of the inputs.tar.gz of the given
    A-2 folders."""
    items = []
    for f in folders:
        tp = Path(f) / "inputs.tar.gz"
        if not tp.is_file():
            continue
        with tarfile.open(tp) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".inp"):
                    txt = t.extractfile(m).read().decode(errors="replace")
                    first = next((ln for ln in txt.splitlines() if ln.strip()), "")
                    if first.lstrip().startswith("!"):
                        items.append((f"{Path(f).name}/{m.name}", txt))
    rng = random.Random(M.SEED if seed is None else seed)
    pick = rng.sample(items, min(int(n), len(items))) if items else []
    bad = []
    for name, txt in pick:
        ok, words, ext = inp_check_text(txt)
        if not ok:
            bad.append(dict(input=name, banned=words, extopt=ext))
    return dict(pass_=bool(pick) and not bad, n_orca_inputs=len(items), n_sampled=len(pick), n_bad=len(bad),
                bad=bad[:20], banned_pattern=BANNED.pattern, required="'!' line contains ExtOpt",
                keyword_lines=sorted({next(ln for ln in t.splitlines() if ln.strip()).strip()[:80] for _, t in pick})[:20])


def cmd_smoke(model, n_rxn=3, workers=None, cores=None):
    """A-2 smoke: autodE-MLTS (arm A) on the n smallest G2-ok D1 targets (distinct dipoles first) with model, then
    the C-3 wording check of every ORCA input -> A2_smoke.json {pass, inp_checks}."""
    ensure_env("mlts")
    T = targets()
    ok_g2 = {p.parent.name for p in M.G2_FINAL.glob("*/.done")}
    S = T[T.rxn_id.astype(str).isin(ok_g2)].sort_values(["n_atoms", "rxn_id"])
    pick, seen = [], set()
    for r in S.itertuples():
        if r.dipole not in seen and len(pick) < int(n_rxn):
            pick.append(str(r.job_id))
            seen.add(r.dipole)
    root = SMOKE_ROOT / re.sub(r"[^A-Za-z0-9_.-]", "_", str(model)) / "ade"
    spec = "test:" + ",".join(pick)
    rc = cmd_run(spec, "0", "1", model=model, engine="A", out=str(root), workers=workers or min(len(pick), 4),
                 cores=cores)
    folders = [root / "jobs" / j for j in pick]
    res = {}
    for j, f in zip(pick, folders):
        s, why = stage_state(f) if is_final(f) else ("missing", "")
        tim = rj(f / "timing.json") or {}
        res[j] = dict(status=s, reason=why, wall_s=tim.get("wall_s"), gpu_s=tim.get("gpu_s"), core_h=tim.get("core_h"),
                      n_ext_calls=tim.get("n_ext_calls"), n_hess=tim.get("n_hess"), n_inputs=None)
    chk = c3_check([f for f in folders if is_final(f)], n=50)
    n_ok = sum(1 for v in res.values() if v["status"] == "ok")
    complete = all(v["status"] != "missing" for v in res.values())
    rep = {"pass": bool(chk["pass_"] and n_ok >= 1 and complete), "complete": complete, "inp_checks": chk,
           "model": str(model), "reactions": res, "n_ok": n_ok, "run_rc": rc, "root": str(root), "written": now(),
           "code": CODE_VERSION, "DECISIONS": DECISIONS}
    jdump(REPORTS / "A2_smoke.json", rep)
    log(f"smoke {model}: pass {rep['pass']} (inputs {chk['n_sampled']}/{chk['n_orca_inputs']} bad {chk['n_bad']}, "
        f"ok {n_ok}/{len(pick)})")
    if not complete:
        return 2
    return 0 if chk["pass_"] else 3


# ================================================================ CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="mlts_autode.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("list")
    r.add_argument("i", nargs="?", default="0")
    r.add_argument("n", nargs="?", default="1")
    r.add_argument("--model")
    r.add_argument("--engine")
    r.add_argument("--list", dest="list_path")
    r.add_argument("--out")
    r.add_argument("--workers", type=int)
    r.add_argument("--cores", type=int)
    r.add_argument("--freq", choices=("analytic", "numfreq"))
    o = sub.add_parser("one")
    o.add_argument("job_id")
    o.add_argument("--model", required=True)
    o.add_argument("--engine", default="A", choices=("A", "Ap"))
    o.add_argument("--out", required=True)
    o.add_argument("--cores", type=int, default=CORES_DEFAULT)
    o.add_argument("--socket", required=True)
    o.add_argument("--tag", default=None)
    o.add_argument("--freq", default=FREQ_DEFAULT, choices=("analytic", "numfreq"))
    o.add_argument("--job-json")
    f = sub.add_parser("refine_dft")
    f.add_argument("list")
    f.add_argument("i", nargs="?", default="0")
    f.add_argument("n", nargs="?", default="1")
    f.add_argument("--root")
    f.add_argument("--src")
    f.add_argument("--list", dest="list_path")
    a = sub.add_parser("a3_report")
    a.add_argument("--list", dest="list_path")
    s = sub.add_parser("status")
    s.add_argument("--root")
    m = sub.add_parser("smoke")
    m.add_argument("--model", required=True)
    m.add_argument("--n", type=int, default=3)
    m.add_argument("--workers", type=int)
    m.add_argument("--cores", type=int)
    x = ap.parse_args(argv)
    if x.cmd == "run":
        return cmd_run(x.list, x.i, x.n, model=x.model, engine=x.engine, list_path=x.list_path, out=x.out,
                       workers=x.workers, cores=x.cores, freq=x.freq)
    if x.cmd == "one":
        return cmd_one(x.job_id, x.model, x.engine, x.out, x.cores, x.socket, x.tag or jobtag(), x.freq, x.job_json)
    if x.cmd == "refine_dft":
        return cmd_refine_dft(x.list, x.i, x.n, root=x.root, src=x.src, list_path=x.list_path)
    if x.cmd == "a3_report":
        return cmd_a3_report(x.list_path)
    if x.cmd == "status":
        return cmd_status(x.root)
    if x.cmd == "smoke":
        return cmd_smoke(x.model, x.n, x.workers, x.cores)
    return 2


if __name__ == "__main__":
    sys.exit(main())
