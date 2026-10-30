"""GEO2 common paths and helpers (docs/specs/GEO2.md). Python 3.9+ (D1 run env), reactot and env_d1fu2 compatible.

Writes only under $GEO2 (= $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2) and analysis/geo2/results.
Every other root is read only. The roots can be overridden by environment variables (tests only)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))
GEO1 = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))            # GEO1 scratch (read only)
GEO1_RES = REPO / "analysis" / "geo" / "results"                                    # GEO1 results (read only)
G2_FINAL = Path(os.environ.get("GEO2_G2_FINAL", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final"))
D1_JOBS = Path(os.environ.get("GEO2_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
D1_LABELS_G2 = Path(os.environ.get("GEO2_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
GEO_FINAL = GEO2 / "geo_final"
COV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, As=1.19, Se=1.20,
           Br=1.20, I=1.39)                                                         # Cordero et al. 2008 (A)
SEEDS = (0, 1, 2)


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    q = Path(os.path.abspath(p))
    for root in (GEO2, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $GEO2 / analysis/geo2/results")


def write_atomic(path, data):
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=_jsonable))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], dtype=float)


def xyz_text(syms, X, comment):
    return f"{len(syms)}\n{comment}\n" + "".join(f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n"
                                                 for s, (x, y, z) in zip(syms, np.asarray(X, float)))


def marker(d):
    d = Path(d)
    if (d / ".done").is_file():
        return ".done"
    f = sorted(d.glob(".fail_*")) if d.is_dir() else []
    return f[0].name if f else None


def parse_pairs(s):
    return sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(s).split())


def dmat(X):
    X = np.asarray(X, float)
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def form_d(X, pairs):
    D = dmat(X)
    return np.array([D[i, j] for i, j in pairs])


def kabsch_rmsd(P, Q, reflect=True):
    """RMSD of P onto Q (same order) after the best proper (or, with reflect, improper) superposition."""
    P, Q = np.asarray(P, float) - np.mean(P, 0), np.asarray(Q, float) - np.mean(Q, 0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    V = Vt.T
    s = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    best = np.inf
    for dd in ((s, -s) if reflect else (s,)):
        R = V @ np.diag([1.0, 1.0, dd]) @ U.T
        if not reflect and np.linalg.det(R) < 0:
            continue
        best = min(best, float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean())))
    return best
