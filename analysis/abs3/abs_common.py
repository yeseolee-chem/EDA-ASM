"""ABS3 common paths, constants and helpers (docs/specs/ABS3_MODELS.md 3).

Stdlib only at import (numpy on demand): imported by env_d1fu2 (S0, LS, TF · GNN, C, collect) and env_espley (XGB, E).
Writes are allowed only under $ABS3 and analysis/abs3/results (write guard; STOP otherwise). The ch3 constants
(TARGETS10, TARGETS3, M, seed, bootstrap count) come from analysis/ch3/ch3_common.py (read only). Every path can be
overridden by an environment variable (developer dry runs only); the defaults are the spec paths."""
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
CH3M_DIR = REPO / "analysis" / "ch3m"
sys.path.insert(0, str(CH3_DIR))
import ch3_common as C3  # noqa: E402

RES = Path(os.environ.get("ABS3_RESULTS", HERE / "results"))
ABS3 = Path(os.environ.get("ABS3_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/abs3"))
UNITS = ABS3 / "units"
SMOKE = ABS3 / "smoke"
PACK = ABS3 / "pack"
COLLECT = ABS3 / "collect"
LOGS = Path("/gpfs/tmp_cpu2/yeseo1ee/abs3/logs")

CH3_LF = C3.LF                                      # $CH3/lf, regenerated lf inputs (read only)
CH3_RES = C3.RES                                    # analysis/ch3/results (read only)
CH3M_RES = CH3M_DIR / "results"                     # analysis/ch3m/results (read only)
CH3M_SCRATCH = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3m")
DEPLOY_RES = C3.DEPLOY_RES
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
D1PAIR_RES = REPO / "analysis" / "d1_pair" / "results"
LAD, D1PAIR = C3.LAD, C3.D1PAIR
LABELS = C3.LABELS                                  # $LAD/F_pairs_cr2.csv (pair dd_<t>)
LABELS_JSON = LAD / "labels_d1_cr2.json"            # per-reaction labels (spec 4)
LABELS_PQ = LAD_RES / "labels_d1_cr2.parquet"       # per-reaction attributes (dipole, skeleton, series, attach)
V_REPORT = LAD_RES / "V_report.json"
S_FOLDS = D1PAIR_RES / "S_folds.csv"
S_SPLITS = D1PAIR_RES / "fu2" / "S_splits.json"
SPLIT_FILES = [D1PAIR / "fu2" / "splits" / f"T1__f{k}.json" for k in range(5)]
K3_PRED = CH3_RES / "K3_predictions.parquet"
KM_PRED = CH3M_RES / "KM_predictions.parquet"
EM_NOISE = CH3M_RES / "EM_noise.csv"
EM_METRICS = CH3M_RES / "EM_metrics.csv"
EM_RANK = CH3M_RES / "EM_rank.csv"
PACK_NPZ = PACK / "lf_abs.npz"
PACK_JSON = PACK / "lf_abs.json"

CH3M_SHA = "36b8e17b4098288a99a9c87c0f69b8e89aa9a0bb"
TARGETS10, TARGETS3 = list(C3.TARGETS10), list(C3.TARGETS3)
M_ROWS, GROUPS = C3.M_ROWS, C3.GROUPS
SEED, N_BOOT = C3.SEED, C3.N_BOOT
# spec 4: label key of each of the 10 targets in labels_d1_cr2.json (= V_report.json pairs.code.d1_pairs)
LABEL_KEYS = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
              "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
              "barrier": "barrier_kcal"}
ARM = "S"
RUNG = "lf3"                                        # unit names A__lf3__S__<model>__f<k> (spec 0-2)
PAIR_RUNG = "abs3"                                  # rung of the assembled pair predictions (spec 7)
MODELS = ("TF", "GNN", "XGB")
ABS_LABEL = {"TF": "A-TF", "GNN": "A-GNN", "XGB": "A-XGB"}
PAIR_LABEL = {"TF": "TF3", "GNN": "GNN3", "XGB": "XGB3"}
LABEL_MODELS = ("A-TF", "A-GNN", "A-XGB", "TF3", "GNN3", "XGB3")
NF = 5
FOLD_KEYS = ("train", "test", "inner_train", "inner_val")
RXN_COLS = ["rxn_id", "fold", "model", "target", "y", "yhat", "yhat_seed_sd", "dipole", "skeleton", "series", "attach"]
PAIR_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                        # = deploy K_predictions / CH3M KM_predictions

NN_TRAINER_FILES = ("train_abs.py", "abs_model.py", "abs_data.py", "abs_common.py")     # C_tests trainer_code_sha256
XGB_TRAINER_FILES = ("xgb_abs.py", "abs_xgbcore.py", "abs_pairs.py", "abs_common.py")

WRITE_OK = (ABS3, RES)


def M():
    return C3.M()


def uname(model, fold):
    return f"A__{RUNG}__{ARM}__{model}__f{fold}"


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


def sha256_array(a) -> str:
    """sha256 of an array's dtype, shape and C-order bytes (LS-5: pack arrays vs abs_data arrays)."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype.kind == "U":
        a = np.ascontiguousarray(a.astype(str))
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


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


def clean(o):
    """json-safe copy: numpy scalars / arrays -> python, non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            o = float(o)
    except ImportError:
        pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def norm(o):
    """the value as it reads back from json (fingerprints are compared this way)."""
    return json.loads(json.dumps(clean(o)))


def write_json(path, obj) -> Path:
    return write_text(path, json.dumps(clean(obj), indent=1, ensure_ascii=False) + "\n")


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


def code_sha(files):
    """{relative name: sha256} of files under analysis/ (hashed at import time by the callers)."""
    out = {}
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = REPO / "analysis" / f
        out[str(p.relative_to(REPO / "analysis")) if REPO / "analysis" in p.parents else str(p)] = sha256(p)
    return out
