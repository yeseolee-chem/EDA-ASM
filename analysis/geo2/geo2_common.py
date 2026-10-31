"""GEO2 common paths and helpers (docs/specs/GEO2.md). Python 3.9+ (D1 run env), reactot and env_d1fu2 compatible.

Writes only under $GEO2 (= $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2b; GEO2b) and analysis/geo2/results.
Every other root is read only. The roots can be overridden by environment variables (tests only).
GEO2b decisions (results/DECISIONS.md): A-G2c = the G2 input of both datasets is G2c ($G2C/{d1,d0}/g2_final, the
select_g2c.py build); D0D1 = D0 joins D1 (uid D0_<rid>, label TS = the Coley profile TS, D0 folds of d0_folds.csv)."""
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
RES = Path(os.environ.get("GEO2_RESULTS", str(HERE / "results")))                    # override: tests only
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))
GEO1 = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))          # GEO1b scratch (read only)
GEO1_RES = Path(os.environ.get("GEO2_GEO1_RES", str(REPO / "analysis" / "geo1b" / "results")))  # GEO1b results (r/o)
# ---- G2 input = G2c (decision A-G2c): the select_g2c.py build roots, read only (= geo1b geo_common D1_G2_FINAL /
# D0_G2_FINAL). Each .done folder = a byte copy of the chosen run (orig / r1 / r2) + g2c.json {rxn_id, src, chosen}.
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
G2_FINAL = Path(os.environ.get("GEO2_G2_FINAL", str(G2C / "d1" / "g2_final")))      # D1: $G2C/d1/g2_final/<rid>
G2_FINAL_D0 = Path(os.environ.get("GEO2_G2_FINAL_D0", str(G2C / "d0" / "g2_final")))  # D0: $G2C/d0/g2_final/<rid>
G2C_RES = Path(os.environ.get("G2C_RESULTS", str(REPO.parent / "eda-asm-prediction-g2c" / "analysis" / "g2c"
                                                  / "results")))                    # = geo1b geo_common G2C_RES
G2C_SELECT = G2C_RES / "C_select.csv"                                               # src, rxn_id, chosen, ...
G2C_RUNS = ("orig", "r1", "r2")
D1_JOBS = Path(os.environ.get("GEO2_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
D1_LABELS_G2 = Path(os.environ.get("GEO2_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
# ---- D0 (decision D0D1, design step 5; read only; = what GEO1b data.py reads)
D0_LABELS = Path(os.environ.get("GEO2_D0_LABELS", str(REPO / "labels_all.json")))
D0_PROF = Path(os.environ.get("GEO2_D0_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/"
                                              "full_dataset_profiles"))           # <rid>/<ts_file> = label TS
D0_EXCLUDE = {3090, 3766, 4252, 3400, 5783}                                        # labels_all non-ok (geo_common)
D0_DIR = GEO1 / "d0"
D0_FOLDS = D0_DIR / "d0_folds.csv"                     # rxn_id, component, fold (0-4; -9 = D-1 overlap), twin_of, overlap
D0_PAIRS_CSV = D0_DIR / "D0_pairs.csv"                 # D0 pairs (pair_id D#####, ref_kind d0, dd_*, edge_kind, ok)
D0_TWINS = D0_DIR / "d0_twins.csv"                     # D0 reactions with a stereo-free D1 twin (m1 = a T1 fold -1 twin)
D0_PAIRS_REPORT = GEO1_RES / "D0_pairs_report.json"    # stop false + outputs sha256 (DECISIONS_PAIR P-stop reader rule)
D0_FOLD_OVERLAP = -9
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


def d0_job_id(rid):
    """the job_id of a D0 reaction in the combined GEO2b label rows (design step 6) = the D0-G2 job name."""
    return f"D0G2_{int(rid):04d}"


def report_output_ok(report, path):
    """reader rule of the GEO1b D0 files (DECISIONS_PAIR P-stop, = geo1b geo_common.load_d0_folds): the report says
    stop false and records under 'outputs' (keyed by absolute path; matched by file name) the sha256 of the file now on
    disk. -> None if fine, else the problem (str)."""
    try:
        rep = json.loads(Path(report).read_text())
    except (OSError, ValueError) as e:
        return f"{report} unreadable ({type(e).__name__})"
    if rep.get("stop") is not False:
        return f"{report}: stop = {rep.get('stop')!r} (must be false)"
    if not Path(path).is_file():
        return f"{path} missing"
    outs = [v for k, v in (rep.get("outputs") or {}).items() if Path(k).name == Path(path).name]
    if len(outs) != 1 or (outs[0] or {}).get("sha256") != sha256(path):
        return f"{path} sha256 differs from the one recorded in {Path(report).name} outputs"
    return None


def csv_text(rows, fieldnames):
    """rows (dicts) -> csv text (for write_atomic; csv module only, so the D1 run env needs no pandas)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(fieldnames), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
