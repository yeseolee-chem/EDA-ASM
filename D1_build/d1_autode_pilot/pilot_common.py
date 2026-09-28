#!/usr/bin/env python3
"""pilot_common.py — config, paths, markers, autodE configuration shared by every stage.

Every stage is idempotent: it writes its result atomically and a `.done_<stage>` marker
last; a rerun finds the marker and exits 0. A stage that cannot succeed writes
`.fail_<stage>` with the reason, and later stages refuse to start (run_job.sh checks).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
EH_TO_KCAL = 627.5094740631          # same constant as label_true/scripts/_common.py
STAGES = ("ts", "ref", "inputs", "sp", "label")


# ---------------------------------------------------------------- config / manifest
def load_config(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("D1P_CONFIG", HERE / "config.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_job(job_id: str, manifest: str | Path | None = None) -> dict:
    m = pd.read_csv(manifest or os.environ.get("D1P_MANIFEST", HERE / "pilot_manifest.csv"),
                    keep_default_na=False)
    row = m[m.job_id == job_id]
    if len(row) != 1:
        raise SystemExit(f"job {job_id} not in manifest")
    return row.iloc[0].to_dict()


def job_dir(cfg: dict, job_id: str) -> Path:
    d = Path(cfg["scratch"]) / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_path(cfg: dict, *parts) -> Path:
    return Path(cfg["repo_root"]).joinpath(*parts)


def import_repo_modules(cfg: dict):
    """fragmenter (graph partition) and stage3_parse (method-① parser) straight from the repo."""
    for sub in ("analysis/espley_xtb_repro", "label_true/scripts"):
        p = str(repo_path(cfg, sub))
        if p not in sys.path:
            sys.path.insert(0, p)
    import fragmenter            # noqa: F401
    import stage3_parse          # noqa: F401
    return sys.modules["fragmenter"], sys.modules["stage3_parse"]


# ---------------------------------------------------------------- markers / io
def done(jd: Path, stage: str) -> bool:
    return (jd / f".done_{stage}").is_file()


def failed(jd: Path, stage: str) -> str | None:
    f = jd / f".fail_{stage}"
    return f.read_text() if f.is_file() else None


def mark_done(jd: Path, stage: str):
    (jd / f".fail_{stage}").unlink(missing_ok=True)
    (jd / f".done_{stage}").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


def mark_fail(jd: Path, stage: str, reason: str):
    (jd / f".fail_{stage}").write_text(reason)
    print(f"[FAIL {stage}] {reason}", file=sys.stderr)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return str(x)


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0]); xyz.append([float(v) for v in t[1:4]])
    import numpy as np
    return syms, np.array(xyz)


def write_xyz(path, syms, xyz, title=""):
    body = "\n".join(f"{s:<2} {x:15.8f} {y:15.8f} {z:15.8f}" for s, (x, y, z) in zip(syms, xyz))
    Path(path).write_text(f"{len(syms)}\n{title}\n{body}\n")


def xyz_block(syms, xyz) -> str:
    return "\n".join([str(len(syms)), ""] + [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(syms, xyz)]) + "\n"


def n_cores() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# ---------------------------------------------------------------- autodE
def configure_autode(cfg: dict, cores: int | None = None):
    """Set autodE to the Coley 2023 protocol (Sci. Data 10:66, doi:10.1038/s41597-023-01977-8).

    ORCA engine: B3LYP D3BJ def2-SVP (low_opt/opt/opt_ts/grad/hess/low_sp), def2-TZVP (sp),
    SMD(water) via autodE's CPCM(water) + %cpcm smd block, OptTS block mirroring Coley's G16 flags.
    G16 engine: Coley's keyword lists verbatim (input_generation.py determine_keywords).
    """
    import autode as ade
    from autode.values import Allocation
    from autode.wrappers.keywords import implicit_solvent_types as solv

    t = cfg["ts_stage"]
    ade.Config.n_cores = cores or n_cores()
    ade.Config.max_core = Allocation(cfg["maxcore_mb"], units="MB")
    ade.Config.hmethod_conformers = bool(t["hmethod_conformers"])
    ade.Config.lcode = "xtb"
    ade.Config.XTB.path = cfg["xtb_bin"]

    if t["engine"] == "orca":
        ade.Config.hcode = "orca"
        ade.Config.ORCA.path = cfg["orca_bin"]
        kw = ade.Config.ORCA.keywords
        kw.set_functional(t["functional"])
        kw.set_dispersion(t["dispersion"])
        kw.set_opt_basis_set(t["opt_basis"])
        kw.low_sp.basis_set = t["opt_basis"]
        kw.sp.basis_set = t["sp_basis"]
        kw.opt_ts = [k for k in kw.opt_ts._list if "%geom" not in str(k)] + [t["orca_optts_block"]]
        ade.Config.ORCA.implicit_solvation_type = solv.smd
    elif t["engine"] == "g16":
        if not cfg.get("g16_bin"):
            raise SystemExit("engine g16 requested but g16_bin is not set")
        ade.Config.hcode = "g16"
        ade.Config.G16.path = cfg["g16_bin"]
        f, lo, hi, disp, iop = t["functional"], t["opt_basis"].replace("-", "").lower(), \
            t["sp_basis"].replace("-", "").lower(), t["g16_disp"], t["g16_iop"]
        K = ade.Config.G16.keywords
        K.low_opt = ade.SinglePointKeywords([f, lo, "Opt=(loose, maxcycles=10)", disp, iop])
        K.grad = ade.SinglePointKeywords([f, lo, "Force(NoStep)", disp, iop])
        K.opt = ade.SinglePointKeywords([f, lo, "Opt", disp, iop])
        K.opt_ts = ade.SinglePointKeywords([f, lo, "Freq", disp, t["g16_ts"], iop])
        K.hess = ade.SinglePointKeywords([f, lo, "Freq", disp, iop])
        K.sp = ade.SinglePointKeywords([f, hi, disp, iop])
    else:
        raise SystemExit(f"unknown engine {t['engine']}")
    return ade


def hmethod_keywords_summary() -> dict:
    """What autodE will actually write (recorded in every result.json)."""
    import autode as ade
    code = ade.Config.hcode
    K = getattr(ade.Config, code.upper()).keywords
    return {k: str(getattr(K, k)) for k in ("low_opt", "opt", "opt_ts", "grad", "hess", "sp", "low_sp")}
