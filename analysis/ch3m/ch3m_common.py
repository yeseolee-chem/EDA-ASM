"""CH3M common paths, constants and helpers (docs/specs/CH3M_MODELS.md 3).

Stdlib only at import (numpy on demand): imported by reactot / env_d1fu2 (S0, XP, TF · GNN, C, collect) and env_espley
(XGB, E). Writes are allowed only under $CH3M and analysis/ch3m/results (write guard). The ch3 constants (TARGETS10,
TARGETS3, M, seeds) come from analysis/ch3/ch3_common.py (read only)."""
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
CH3_DIR = REPO / "analysis" / "ch3"
sys.path.insert(0, str(CH3_DIR))
import ch3_common as C3  # noqa: E402

RES = Path(os.environ.get("CH3M_RESULTS", HERE / "results"))
CH3M = Path(os.environ.get("CH3M_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/ch3m"))
UNITS = CH3M / "units"
SMOKE = CH3M / "smoke"
CR = CH3M / "cr"
XPACK = CH3M / "xgb_pack"
MODEL_DIR = CH3M / "model"
LOGS = CH3M / "logs"

CH3 = C3.CH3                                       # /gpfs/tmp_cpu2/yeseo1ee/ch3 (read only)
CH3_LF = C3.LF                                     # regenerated lf inputs
CH3_RES = C3.RES                                   # analysis/ch3/results (committed ch3 results, read only)
DEPLOY_RES = C3.DEPLOY_RES
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
LAD, G2, D1PAIR, LABELS = C3.LAD, C3.G2, C3.D1PAIR, C3.LABELS

CH3_SHA = "6a4f59c26a3c4db0302feed3d159d51597b330e9"
TARGETS10, TARGETS3 = list(C3.TARGETS10), list(C3.TARGETS3)
M_ROWS, GROUPS = C3.M_ROWS, C3.GROUPS
SEED, N_BOOT = C3.SEED, C3.N_BOOT
ARM = "S"
RUNGS = {10: "lf10", 3: "lf3"}
MODELS = ("TF", "GNN", "XGB")
LABEL_MODELS = ("dep", "TF10", "TF3", "GNN10", "GNN3", "XGB10", "XGB3")
CANDIDATES = ("TF3", "GNN10", "GNN3", "XGB10", "XGB3")        # spec 8-3, tie order
NF = 5

WRITE_OK = (CH3M, RES)


def M():
    return C3.M()


def uname(rung, model, fold):
    return f"D__{rung}__{ARM}__{model}__f{fold}"


def label_of(rung, model):
    """'lf10' + 'GNN' -> 'GNN10'."""
    return f"{model}{'10' if rung == 'lf10' else '3'}"


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


def write_npz(path, **arrays) -> Path:
    import numpy as np
    p = assert_write_ok(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(f".{p.stem}.{os.getpid()}.tmp.npz")
    np.savez(t, **arrays)
    os.replace(t, p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text())


def array_task(i) -> int:
    if str(i) in ("@I", "A"):
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            stop("array index requested but SLURM_ARRAY_TASK_ID is not set", 2)
        return int(v)
    return int(i)
