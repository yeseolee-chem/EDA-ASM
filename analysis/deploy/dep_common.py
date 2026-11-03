"""DEPLOY-MODEL common paths, constants and helpers (docs/specs/DEPLOY_MODEL.md 2).

Light on purpose (stdlib only at import): it is imported by the D1 run env (python 3.9, phases L / M), venv5 (F / T / Q)
and env_d1fu2 (C / K / E / W / I). Writes are allowed only under $DEP and analysis/deploy/results (write guard).
Every path can be overridden by an environment variable (tests only); the defaults are the spec paths."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = Path(os.environ.get("DEP_RESULTS", HERE / "results"))

DEP = Path(os.environ.get("DEP_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/deploy"))
G2 = Path(os.environ.get("DEP_G2", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2"))
G2_JOBS = Path(os.environ.get("DEP_G2_JOBS", G2 / "jobs"))
G2_FINAL = Path(os.environ.get("DEP_G2_FINAL", G2 / "g2_final"))
LAD = Path(os.environ.get("DEP_LAD", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))
D1_RUN = Path(os.environ.get("DEP_D1_RUN", "/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run"))
D1_MANIFEST = Path(os.environ.get("DEP_D1_MANIFEST", D1_RUN / "d1_manifest.csv"))

LF_FINAL = Path(os.environ.get("DEP_LF_FINAL", DEP / "g2lf_final"))
LABELS_LF = Path(os.environ.get("DEP_LABELS_LF", DEP / "labels_lf.json"))
META_LF = Path(os.environ.get("DEP_META_LF", DEP / "input_meta_lf.csv"))
X182_LF = Path(os.environ.get("DEP_X182_LF", DEP / "X182_lf.parquet"))
FU2_LF = Path(os.environ.get("DEP_FU2_LF", DEP / "fu2_lf"))
TOK_LF = Path(os.environ.get("DEP_TOK_LF", FU2_LF / "tokens"))
PROF_EMPTY = Path(os.environ.get("DEP_PROF_EMPTY", DEP / "prof_empty"))
UNITS = Path(os.environ.get("DEP_UNITS", DEP / "units"))
MODEL_DIR = Path(os.environ.get("DEP_MODEL_DIR", DEP / "model"))

RUNG, ARM, MODEL = "lf", "S", "TF"
TARGETS = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier")   # = fu2_common.TARGETS
PRIMARY = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")             # 9 main targets (G-C)
ELEMENTS = (1, 6, 7, 8, 9, 14, 16, 17, 35)
TOKEN_VERSION = "fu2-tokens-3-lf"
ATOM_VERSION = "fu2-atoms-3-lf"
SEED = 20261019
N_BOOT = 2000

WRITE_OK = (DEP, RES)


class Stop(SystemExit):
    """STOP (exit 3) or not ready (exit 2), printed to stderr."""


def stop(msg: str, code: int = 3):
    print(f"STOP: {msg}" if code == 3 else f"NOT READY: {msg}", file=sys.stderr, flush=True)
    raise Stop(code)


def not_ready(msg: str):
    stop(msg, 2)


def check_targets():
    """TARGETS / ELEMENTS against the FU2 code (needs analysis/d1_pair/fu2 importable; envs with numpy)."""
    sys.path.insert(0, str(REPO / "analysis" / "d1_pair" / "fu2"))
    sys.path.insert(0, str(REPO / "analysis" / "d1_pair"))
    import fu2_common as F  # noqa: E402
    if tuple(F.TARGETS) != TARGETS:
        stop(f"TARGETS {TARGETS} != fu2_common.TARGETS {tuple(F.TARGETS)}")
    return True


def assert_write_ok(path) -> Path:
    p = Path(path).resolve()
    for root in WRITE_OK:
        r = Path(root).resolve()
        if p == r or r in p.parents:
            return p
    stop(f"write outside {[str(x) for x in WRITE_OK]}: {p}")


def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_bytes(path, data: bytes) -> Path:
    p = assert_write_ok(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(p)
    with open(t, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)
    return p


def write_text(path, text: str) -> Path:
    return write_bytes(path, text.encode())


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj) -> Path:
    return write_text(path, json.dumps(obj, indent=1, default=_json_default) + "\n")


def write_parquet(path, df) -> Path:
    p = assert_write_ok(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(p)
    df.to_parquet(t, index=False)
    os.replace(t, p)
    return p


def write_csv(path, df) -> Path:
    p = assert_write_ok(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(p)
    df.to_csv(t, index=False)
    os.replace(t, p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text())


def array_task(i) -> int:
    """'@I' / 'A' -> $SLURM_ARRAY_TASK_ID, else int(i)."""
    if str(i) in ("@I", "A"):
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            stop("array index requested but SLURM_ARRAY_TASK_ID is not set", 2)
        return int(v)
    return int(i)
