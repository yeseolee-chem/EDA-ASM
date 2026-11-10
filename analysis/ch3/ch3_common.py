"""CH3-MODEL common paths, constants and helpers (docs/specs/CH3_MODEL.md 5).

Light on purpose (stdlib only at import, numpy on demand): imported by reactot (S0), env_d1fu2 (V, C, K, W, I) and
env_espley (E). Writes are allowed only under $CH3 and analysis/ch3/results (write guard); the regenerated deploy inputs
are written by the unchanged deploy code with DEP_SCRATCH=$CH3/lf, DEP_RESULTS=analysis/ch3/results/lf_inputs (spec 3).
Every path can be overridden by an environment variable (tests only); the defaults are the spec paths."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = Path(os.environ.get("CH3_RESULTS", HERE / "results"))
CH3 = Path(os.environ.get("CH3_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/ch3"))

LF = CH3 / "lf"                                   # DEP_SCRATCH of the regenerated deploy inputs (spec 3)
LF_RES = RES / "lf_inputs"                        # DEP_RESULTS of the regeneration (deploy reports land here)
UNITS = CH3 / "units"
MODEL_DIR = CH3 / "model"
SMOKE = CH3 / "smoke"
C3R = CH3 / "c3r"
LOGS = CH3 / "logs"

DEPLOY_SHA = "2f9ffa35511ca59dc74f02744edc2a9d5ff31afd"
DEPLOY_DIR = REPO / "analysis" / "deploy"
DEPLOY_RES = DEPLOY_DIR / "results"               # committed deploy reports (read only)
G2 = Path(os.environ.get("CH3_G2", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2"))
LAD = Path(os.environ.get("CH3_LAD", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))
D1PAIR = Path(os.environ.get("CH3_D1PAIR", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair"))
D1_RUN = Path(os.environ.get("CH3_D1_RUN", "/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run"))
LABELS = LAD / "F_pairs_cr2.csv"

TARGETS10 = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier")   # = fu2_common.TARGETS
TARGETS3 = ("strain", "inter", "solv", "barrier")
GROUPS = {"strain": ("d1", "d2"), "inter": ("elst", "pauli", "oi", "disp", "c_ghost"), "solv": ("cpcm", "cds"),
          "barrier": ("barrier",)}
M_ROWS = tuple(tuple(1.0 if t in GROUPS[g] else 0.0 for g in TARGETS3) for t in TARGETS10)   # 10 x 4 (spec 5)
I_BAR3, I_CH9_3, I_PRIM3 = 3, [0, 1, 2], [0, 1, 2, 3]   # fu2_common constants in ch3 mode (spec 5)
RUNG3, ARM, MODEL = "lf3", "S", "TF"
SEED = 20261019
N_BOOT = 2000

WRITE_OK = (CH3, RES)


def M():
    """10 -> 4 conversion matrix (TARGETS10 rows, TARGETS3 columns), float64."""
    import numpy as np
    return np.asarray(M_ROWS, dtype=np.float64)


class Stop(SystemExit):
    """STOP (exit 3) or not ready (exit 2), printed to stderr."""


def stop(msg: str, code: int = 3):
    print(f"STOP: {msg}" if code == 3 else f"NOT READY: {msg}", file=sys.stderr, flush=True)
    raise Stop(code)


def not_ready(msg: str):
    stop(msg, 2)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
        if isinstance(o, np.bool_):
            return bool(o)
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
