"""Stage definitions and runners of the GNN3 input pipeline (thin wrappers; standard library only).

Every stage runs the ORIGINAL scripts of ``analysis/deploy`` as child processes, unchanged, with the environment the
published inputs were built with (CH3 jobs 1070562-1070568, logs in /gpfs/tmp_cpu2/yeseo1ee/ch3/logs): all ``DEP_*``
variables point into one fresh root, so nothing outside it is written. This module adds no computation; it only fixes
the order, the arguments, the environment and the record of exit codes.

Environments (the interpreter that runs this module must be the one the stage needs):

    stage        job wrapper (submit directory)                         interpreter
    ts-search    analysis/deploy/jobs/cpu_d1.slurm (analysis/deploy)    D1 run env: python 3.9.5 venv, autodE 1.4.5,
    finalize                                                            rdkit, xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f
    metadata                                                            (/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh)
    features     analysis/d1_pair/jobs/g.slurm (analysis/d1_pair),      venv5: python 3.10.14, tblite 0.7.0, xtb 6.7.1
    tokens       --export=ALL,PYENV=venv5,D1PAIR_PHASE=F                (edcfbbe), the Si patch (site_si)

Scope: reactions of the D1 set (the 9,266 targets of analysis/d1_g2/results/G_status.csv). ``finalize`` (report)
and ``metadata`` read that target list and the D1 manifest, so a reaction outside D1 can get a TS
(``ts-search``) but not the later stages without a code change (CODE_MAP.md, problems found).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .. import constants as K

DEPLOY_DIR = K.REPOSITORY / "analysis" / "deploy"
D1PAIR_DIR = K.REPOSITORY / "analysis" / "d1_pair"
G_STATUS = K.REPOSITORY / "analysis" / "d1_g2" / "results" / "G_status.csv"
#: The published inputs (read only): $CH3/lf and its reports.
PRODUCTION_ROOT = K.LF_INPUTS
PRODUCTION_REPORTS = K.REPOSITORY / "analysis" / "ch3" / "results" / "lf_inputs"
#: DEP_* variables passed through unchanged (job start time, time budget); every other DEP_* is replaced.
KEEP_DEP = ("DEP_JOB_T0", "DEP_STOP_AFTER_H")


@dataclass(frozen=True)
class Stage:
    """One stage: its original commands (scripts relative to analysis/deploy) and the environment it needs."""

    name: str
    purpose: str
    environment: str                     # "d1" or "venv5"
    working_directory: Path              # = the submit directory of the original job


STAGES: dict[str, Stage] = {
    "ts-search": Stage("ts-search", "SMILES -> autodE TS (+ stereo / reference checks) -> finalize (g2_deploy.py pool)",
                       "d1", DEPLOY_DIR),
    "finalize": Stage("finalize", "ORCA XTB2 frequency job at the existing g2 TS; report with gates L-1..L-4",
                      "d1", DEPLOY_DIR),
    "metadata": Stage("metadata", "per-reaction records for the feature code (meta_lf.py run)", "d1", DEPLOY_DIR),
    "features": Stage("features", "xTB features -> X182_lf.parquet (feat_lf.py slice 0 1, merge)", "venv5", D1PAIR_DIR),
    "tokens": Stage("tokens", "per-atom xTB values -> 29-wide atom tokens (tokens_lf.py atoms 0 1, gate, tokens)",
                    "venv5", D1PAIR_DIR),
}
ORDER = ("finalize", "metadata", "features", "tokens")


def root_environment(root: Path, base: dict | None = None) -> dict:
    """Child environment: every input / output path of the deploy code inside ``root`` (= i3_chain.i3_env).

    Writes go to root/g2lf_final, root/labels_lf.json, root/input_meta_lf.csv, root/F, root/X182_lf.parquet,
    root/fu2_lf and root/results (the reports); the D1-G2 folders, the D1 manifest and the label files are only read.
    """
    root = Path(os.path.abspath(root))
    env = {k: v for k, v in dict(os.environ if base is None else base).items()
           if not k.startswith("DEP_") or k in KEEP_DEP}
    env.update(DEP_SCRATCH=str(root), DEP_RESULTS=str(root / "results"), DEP_LF_FINAL=str(root / "g2lf_final"),
               DEP_LABELS_LF=str(root / "labels_lf.json"), DEP_META_LF=str(root / "input_meta_lf.csv"),
               DEP_X182_LF=str(root / "X182_lf.parquet"), DEP_FU2_LF=str(root / "fu2_lf"),
               DEP_TOK_LF=str(root / "fu2_lf" / "tokens"), DEP_PROF_EMPTY=str(root / "prof_empty"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


def stage_commands(name: str, root: Path, job_list: Path, cores: int = 4, manifest: Path | None = None,
                   n_parallel: int = 1) -> list[list[str]]:
    """The original command lines of a stage (script paths absolute; run with the stage's interpreter)."""
    deploy = str(DEPLOY_DIR)
    job_list = str(Path(job_list).resolve())
    if name == "ts-search":
        if manifest is None:
            raise ValueError("ts-search needs --manifest (D1 manifest format, see g2_deploy.py)")
        return [[f"{deploy}/g2_deploy.py", "pool", job_list, str(n_parallel), str(cores), "--manifest",
                 str(Path(manifest).resolve()), "--out", str(Path(root).resolve())]]
    if name == "finalize":
        return [[f"{deploy}/finalize_lf.py", "run", "0", "1", str(cores), "--jobs", job_list],
                [f"{deploy}/finalize_lf.py", "report", "--jobs", job_list]]
    if name == "metadata":
        return [[f"{deploy}/meta_lf.py", "run", "--jobs", job_list]]
    if name == "features":
        return [[f"{deploy}/feat_lf.py", "slice", "0", "1"], [f"{deploy}/feat_lf.py", "merge"]]
    if name == "tokens":
        return [[f"{deploy}/tokens_lf.py", "atoms", "0", "1"], [f"{deploy}/tokens_lf.py", "gate"],
                [f"{deploy}/tokens_lf.py", "tokens"]]
    raise ValueError(f"unknown stage {name}")


def check_interpreter(stage: Stage) -> None:
    """Refuse to run a stage in the wrong job environment (the original scripts would fail later, less clearly)."""
    if stage.environment == "d1":
        d1 = os.environ.get("D1_PY")
        if not d1 or Path(sys.executable).resolve() != Path(d1).resolve():
            raise SystemExit(f"STOP: stage {stage.name} needs the D1 run env (analysis/deploy/jobs/cpu_d1.slurm); "
                             f"python {sys.executable}, D1_PY {d1}")
    elif os.environ.get("D1PAIR_PHASE") != "F" or os.environ.get("PYENV") != "venv5":
        raise SystemExit(f"STOP: stage {stage.name} needs analysis/d1_pair/jobs/g.slurm with "
                         "--export=ALL,PYENV=venv5,D1PAIR_PHASE=F")


def job_rows(job_list: Path) -> list[dict]:
    """G_status rows (job_id, rxn_id, dipole, status, ...) of the job ids in ``job_list`` (csv with a job_id column)."""
    with open(job_list, newline="") as handle:
        wanted = [row["job_id"] for row in csv.DictReader(handle)]
    with open(G_STATUS, newline="") as handle:
        status = {row["job_id"]: row for row in csv.DictReader(handle)}
    missing = [j for j in wanted if j not in status]
    if missing:
        raise SystemExit(f"STOP: job ids not in {G_STATUS} (D1 targets only): {missing[:10]}")
    return [status[j] for j in wanted]


def prepare_root(root: Path) -> None:
    """Create the root, its results folder and the EMPTY profile folder the feature code requires."""
    root = Path(root)
    for folder in (root, root / "results", root / "prof_empty", root / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    if any((root / "prof_empty").iterdir()):
        raise SystemExit(f"STOP: {root / 'prof_empty'} must be empty")


def reuse_finalized(root: Path, job_list: Path, source_root: Path) -> dict:
    """Copy finished ``finalize`` folders (``source_root/g2lf_final/<rxn_id>``) into ``root`` so that finalize_lf.py
    skips them (it skips every folder with a .done marker). Use this to avoid rerunning the ORCA frequency job, whose
    result is not bit-for-bit reproducible (the B6 features nu_imag / mode_share / mode_async move by ~1e-7)."""
    copied, present = [], []
    for row in job_rows(job_list):
        source, target = Path(source_root) / "g2lf_final" / row["rxn_id"], Path(root) / "g2lf_final" / row["rxn_id"]
        if target.exists():
            present.append(row["rxn_id"])
            continue
        if not (source / ".done").is_file():
            raise SystemExit(f"STOP: {source} has no .done marker")
        temporary = target.with_name(f".{target.name}.copy")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(source, temporary)                     # hard links (ts.hess) become plain copies
        os.replace(temporary, target)
        copied.append(row["rxn_id"])
    return dict(source=str(source_root), copied=copied, already_present=present)


def _tokens_subset_exception(root: Path) -> tuple[list, list]:
    """Classify an exit 3 of ``tokens_lf.py tokens`` with the deploy subset rule (i3_chain.classify_T, unchanged):
    T_report elements that are a subset of the 9 model elements (a small set of reactions has no Cl / Br / Si ...)
    are accepted; every other failed check stays a STOP. Returns (accepted, refused)."""
    sys.path.insert(0, str(DEPLOY_DIR))
    import i3_chain                                            # noqa: E402  (stdlib only at import)
    record = json.loads((Path(root) / "results" / "T_lf_check.json").read_text()).get("tokens") or {}
    return i3_chain.classify_T(record)


def run_stage(name: str, root: Path, job_list: Path, cores: int = 4, manifest: Path | None = None,
              n_parallel: int = 1, reuse_from: Path | None = None) -> int:
    """Run one stage into ``root``; returns 0 (done), 2 (not ready: resubmit) or 3 (STOP).

    Writes ``root/results/gnn3_steps/<stage>.json`` with every command line, exit code and accepted exception.
    """
    stage = STAGES[name]
    check_interpreter(stage)
    root = Path(os.path.abspath(root))
    prepare_root(root)
    record = dict(stage=name, purpose=stage.purpose, root=str(root), job_list=str(job_list), cores=cores,
                  host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), python=sys.executable,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), commands=[], accepted_exceptions=[])
    if name == "finalize" and reuse_from is not None:
        record["reused"] = reuse_finalized(root, job_list, Path(reuse_from))
    env = root_environment(root)
    rc = 0
    for argv in stage_commands(name, root, job_list, cores, manifest, n_parallel):
        start = time.time()
        print(f"=== [{time.strftime('%H:%M:%S')}] {name}: {' '.join(argv)}", flush=True)
        rc = subprocess.run([sys.executable] + argv, env=env, cwd=str(stage.working_directory)).returncode
        print(f"=== exit {rc} ({time.time() - start:.0f} s)", flush=True)
        record["commands"].append(dict(argv=argv, rc=rc, wall_seconds=round(time.time() - start, 1)))
        if rc == 3 and name == "tokens" and argv[1] == "tokens":
            accepted, refused = _tokens_subset_exception(root)
            if accepted and not refused:
                record["accepted_exceptions"].append(dict(
                    check=accepted, rule="i3_chain.classify_T: elements are a subset of the 9 model elements "
                                         "(the model one-hot keeps all 9, so the token width stays 29)"))
                rc = 0
            else:
                record["refused"] = refused
        if rc != 0:
            break
    record.update(rc=rc, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = root / "results" / "gnn3_steps" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return rc


def select_reactions(n_per_dipole: int, seed: int, token_manifest: Path, global_table_ids: set) -> list[dict]:
    """A seeded choice of D1 reactions for checks: ``n_per_dipole`` per dipole among the reactions that are G2 ok and
    have a published token (token checks ok) and a 182-feature row. Dipoles in sorted order, one numpy Generator.

    Returns the G_status rows (job_id, rxn_id, dipole, ...) of the chosen reactions, sorted by job_id.
    """
    import numpy as np
    with open(token_manifest, newline="") as handle:
        token_ok = {int(r["rxn_id"]) for r in csv.DictReader(handle)
                    if r.get("n_ok", "True") in ("True", "true", "1") and r.get("finite", "True") in ("True", "true", "1")
                    and int(r.get("n_formed", 2)) == 2}
    with open(G_STATUS, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok" and int(r["rxn_id"]) in token_ok
                and int(r["rxn_id"]) in global_table_ids]
    rng = np.random.default_rng(seed)
    chosen = []
    for dipole in sorted({r["dipole"] for r in rows}):
        candidates = sorted((r for r in rows if r["dipole"] == dipole), key=lambda r: int(r["rxn_id"]))
        for k in sorted(rng.choice(len(candidates), size=n_per_dipole, replace=False).tolist()):
            chosen.append(candidates[k])
    return sorted(chosen, key=lambda r: r["job_id"])
