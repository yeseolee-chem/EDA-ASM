"""GEO1b common paths and helpers (docs/specs/GEO1B.md; copied from analysis/geo/geo_common.py, G1 removed). Python 3.9 compatible (D1 run env) and the GPU env (env_d1fu2).
GEO1b decisions A-G2c (G2 inputs = the G2c build roots, G2-raw baseline from results/G2RAW_baseline.json) and D0D1
(D0 folds from $GEO1B/d0/d0_folds.csv); judgment calls in results/DECISIONS_GB.md."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True                  # never write .pyc into the read-only D1 code copy (as c_compare.py)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))

# ---- G2 inputs = G2c (decision A-G2c, user 2026-10-30): the select_g2c.py build roots, read only. Each .done folder is
# a byte copy of the chosen run (orig / r1 / r2) + g2c.json {rxn_id, src, chosen, source, energies_eh, rule}.
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))                   # = select_g2c.G2C
D0G2 = Path(os.environ.get("D0G2_SCRATCH", str(G2C / "d0")))                               # D0 G2 root = $G2C/d0
D0_G2_FINAL = D0G2 / "g2_final"                                                             # $G2C/d0/g2_final/<rid>
D1_G2_FINAL = G2C / "d1" / "g2_final"                                                       # $G2C/d1/g2_final/<rid>
G2C_RES = Path(os.environ.get("G2C_RESULTS", str(REPO.parent / "eda-asm-prediction-g2c" / "analysis" / "g2c"
                                                  / "results")))                           # = select_g2c.RES
G2C_SELECT = G2C_RES / "C_select.csv"                                                       # src, rxn_id, chosen, ...
G2C_RUNS = ("orig", "r1", "r2")                                                             # = select_g2c.RUNS

# ---- D0 folds (decision D0D1, design step 2: analysis/geo1b/d0_pairs.py build)
D0_DIR = GEO / "d0"
D0_FOLDS = D0_DIR / "d0_folds.csv"                         # rxn_id, component, fold (0-4; -9 overlap), twin_of, overlap
D0_TWINS = D0_DIR / "d0_twins.csv"                         # every D0 reaction with a stereo-free D1 twin (P-twin-table)
D0_PAIRS_REPORT = RES / "D0_pairs_report.json"             # stop false + outputs sha256 of d0_folds.csv (P-stop)
D0_FOLD_OVERLAP = -9                                       # = d0_pairs.FOLD_OVERLAP (a D-1 overlap, in no set)

# ---- D1 (read only)
D1_G2_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")         # D1-G2 labels / meta only (its g2_final = G2c run 'orig')
D1_G2_SINGLE_RUN = D1_G2_ROOT / "g2_final"                 # single-run D1-G2 TSs: tests.py U-8b regression ONLY
#                                                            (GB-U8r), never a model input (inputs = D1_G2_FINAL, G2c)
D1_LABELS_G2 = D1_G2_ROOT / "G" / "labels_d1_g2.json"      # 9,266 D1 label rows (prep_g.py inputs)
D1_META_G2 = D1_G2_ROOT / "G" / "input_meta_d1_g2.csv"     # rxn_id, A_idx (frag1 = dipole, 0-based)
D1_PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
D1_JOBS = D1_PAIR / "d1_jobs"
D1_MANIFEST = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv")
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
S_FOLDS = REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"

# ---- D0 (read only)
D0_LABELS = REPO / "labels_all.json"
D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
D0_META = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv")
D0_EXCLUDE = {3090, 3766, 4252, 3400, 5783}                # labels_all non-ok (= g1_geom.EXCLUDE)

RMSD_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis")
D1_FOLD_COUNTS = {0: 1874, 1: 1879, 2: 1503, 3: 1435, 4: 2195, -1: 380}   # C_compare rows per T1 fold (checked in I)
# G2-raw baseline (ALL para P_main pairs, D1 test folds 0-4, G2 input as is): no longer hard-coded (A-G2c). It is
# computed from the records BEFORE training by `data.py baseline` and frozen in results/G2RAW_baseline.json, which
# tests.py U-8b and evaluate.py read (load_g2raw_baseline). That file is a freeze, not an independent check (same code,
# same records). The independent regression of metrics.py and of the record assembly is the single-run part of U-8b
# (GB-U8r): the single-run D1-G2 TSs (D1_G2_SINGLE_RUN = GEO1's G2 input) placed on the same D1 test records must
# reproduce the GEO1B.md 3-2 numbers below (= GEO1 analysis/geo/results E_rxn.csv 8,425 rows, E_pairs.csv para rows).
G2RAW_BASELINE = RES / "G2RAW_baseline.json"
G2RAW_SINGLE_RUN_REF = dict(n=1667, share=0.1338, median=0.0218, n_test=8425, g2="single-run D1-G2 (GEO1B.md 3-2)")
G2RAW_TOL = dict(share=0.005, median=0.001)                # GEO1 U-8 tolerances (median in A); n, n_test exact
ELEM = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]   # type index = position
TYPE = {e: i for i, e in enumerate(ELEM)}


def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], dtype=float)


def done(folder):
    return (Path(folder) / ".done").is_file()


def read_if_done(folder):
    return read_xyz(Path(folder) / "ts.xyz") if done(folder) else None


def parse_pairs(s):
    """'1-2 4-8' -> [(1, 2), (4, 8)] (0-based atom indices, each pair sorted, list sorted) = g2_autode.label_pairs."""
    return sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(s).split())


def dmat(x):
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def form_d(x, pairs):
    D = dmat(x)
    return [float(D[i, j]) for i, j in pairs]


def load_meta(path):
    m = pd.read_csv(path, dtype={"A_idx": str})
    return {int(r.rxn_id): sorted(int(v) for v in str(r.A_idx).split()) for r in m.itertuples()}


def load_d1_labels():
    return json.loads(D1_LABELS_G2.read_text())


def load_d0_labels():
    L = json.loads(D0_LABELS.read_text())
    ok = sorted((d for d in L if int(d["rxn_id"]) not in D0_EXCLUDE), key=lambda d: int(d["rxn_id"]))
    assert all(d["status"] == "ok" for d in ok), "a non-excluded D0 label is not status ok"
    return ok


def load_d0_smiles():
    df = pd.read_csv(D0_SMILES)
    return dict(zip(df.rxn_id.astype(int), df.rxn_smiles))


def kabsch_align(mob, ref, reflect=True):
    """mob aligned onto ref (same atom order). Returns (aligned, rmsd, reflected). With reflect=True the improper
    solution is taken when its RMSD is lower (mirror-image TS of an achiral system = the same TS)."""
    mob, ref = np.asarray(mob, float), np.asarray(ref, float)
    mc, rc = mob.mean(0), ref.mean(0)
    X, Y = mob - mc, ref - rc
    U, S, Vt = np.linalg.svd(X.T @ Y)
    V = Vt.T
    s = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    best = None
    for dd in (s, -s):                       # dd = s -> best proper rotation; dd = -s -> best improper (mirror)
        R = V @ np.diag([1.0, 1.0, dd]) @ U.T
        improper = bool(np.linalg.det(R) < 0)
        if improper and not reflect:
            continue
        A = X @ R.T + rc
        r = float(np.sqrt(((A - ref) ** 2).sum(1).mean()))
        if best is None or r < best[1] - 1e-12:
            best = (A, r, improper)
    return best


COV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, As=1.19, Se=1.20,
           Br=1.20, I=1.39)                     # Cordero et al. 2008 covalent radii (A)
COV_F = 1.25                                    # bond if d < 1.25 (r_i + r_j)
SPD_MAX = 9


def bond_types(syms, xyz, pairs):
    """0 none / 1 covalent (from the INPUT geometry) / 2 forming pair. Never computed from the label geometry."""
    D = dmat(xyz)
    r = np.array([COV[s] for s in syms])
    B = (D < COV_F * (r[:, None] + r[None, :])).astype(np.int8)
    np.fill_diagonal(B, 0)
    for i, j in pairs:
        B[i, j] = B[j, i] = 2                   # forming bond (overrides a covalent call)
    return B


def spd_matrix(B):
    """shortest-path hop counts over bond types 1 and 2, clipped at SPD_MAX (unreachable -> SPD_MAX)."""
    n = len(B)
    S = np.where(B > 0, 1, 99).astype(np.int32)
    np.fill_diagonal(S, 0)
    for k in range(n):                          # Floyd-Warshall (n <= ~70)
        S = np.minimum(S, S[:, k:k + 1] + S[k:k + 1, :])
    return np.minimum(S, SPD_MAX).astype(np.int8)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """decision U-0: records.pkl is written in the D1 run env (numpy 2.x: numpy._core) and read in the GPU env
    (numpy 1.26: numpy.core); map the module names so either env can load it."""
    def find_class(self, module, name):
        if module.startswith("numpy._core") and not hasattr(np, "_core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_records():
    with open(GEO / "data" / "records.pkl", "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def load_g2raw_baseline():
    """results/G2RAW_baseline.json (data.py baseline; decision A-G2c), or None if it has not been computed."""
    return json.loads(G2RAW_BASELINE.read_text()) if G2RAW_BASELINE.is_file() else None


def load_d0_folds():
    """$GEO1B/d0/d0_folds.csv (d0_pairs.py build) -> ({rxn_id: dict(component, fold, twin_of, overlap)}, problems).
    Reader checks (DECISIONS_PAIR P-stop): D0_pairs_report.json has stop false and records the sha256 of this file;
    overlap rows have fold -9 and no twin, every other row fold 0-4 and a component. A non-empty problem list = STOP."""
    bad = []
    if not D0_FOLDS.is_file():
        return {}, [f"{D0_FOLDS} missing (run d0_pairs.py build)"]
    try:
        rep = json.loads(D0_PAIRS_REPORT.read_text())
    except (OSError, ValueError) as e:
        rep = {}
        bad.append(f"{D0_PAIRS_REPORT} unreadable ({type(e).__name__})")
    if rep.get("stop") is not False:
        bad.append(f"{D0_PAIRS_REPORT}: stop = {rep.get('stop')!r} (must be false)")
    outs = [v for k, v in (rep.get("outputs") or {}).items() if Path(k).name == D0_FOLDS.name]
    if len(outs) != 1 or outs[0].get("sha256") != sha256(D0_FOLDS):
        bad.append(f"{D0_FOLDS} sha256 differs from the one recorded in {D0_PAIRS_REPORT.name}")
    F = pd.read_csv(D0_FOLDS, dtype=str, keep_default_na=False)
    if list(F.columns) != ["rxn_id", "component", "fold", "twin_of", "overlap"]:
        return {}, bad + [f"{D0_FOLDS}: columns {list(F.columns)}"]
    out = {}
    for r in F.itertuples(index=False):
        rid, fold = int(r.rxn_id), int(r.fold)
        twin = int(r.twin_of) if r.twin_of != "" else None
        if r.overlap not in ("True", "False") or rid in out:
            bad.append(f"d0_folds.csv rxn {rid}: overlap {r.overlap!r} or duplicate row")
            continue
        ov = r.overlap == "True"
        if (ov and (fold != D0_FOLD_OVERLAP or twin is not None)) or \
                (not ov and (fold not in (0, 1, 2, 3, 4) or r.component == "")):
            bad.append(f"d0_folds.csv rxn {rid}: overlap {ov}, fold {fold}, component {r.component!r}, twin {twin}")
        out[rid] = dict(component=r.component, fold=fold, twin_of=twin, overlap=ov)
    return out, bad


def load_d0_twins():
    """$GEO1B/d0/d0_twins.csv (d0_pairs.py build; DECISIONS_PAIR P-twin-table) -> ({D0 rxn_id: dict(kind, twin_of, m1,
    bind = T1 folds of the twins that set the D0 fold, ns = T1 folds of every stereo-free D1 twin)}, problems). Its
    sha256 must be the one D0_pairs_report.json records under 'outputs' (same reader check as load_d0_folds). Moved
    here from tests.py (GB-twin-m1) so data.py, tests.py U-9b and evaluate.py read it the same way."""
    p = D0_TWINS
    try:
        rep = json.loads(D0_PAIRS_REPORT.read_text())
    except (OSError, ValueError):
        rep = {}
    outs = [v for k, v in (rep.get("outputs") or {}).items() if Path(k).name == p.name]
    if not p.is_file() or len(outs) != 1 or outs[0].get("sha256") != sha256(p):
        return {}, [f"{p} missing or its sha256 differs from the one in {D0_PAIRS_REPORT.name}"]

    def folds(s):                                   # '<D1 rxn_id>:<T1 fold> ...' -> {T1 fold}
        return {int(x.rsplit(":", 1)[1]) for x in str(s).split()}
    F = pd.read_csv(p, dtype=str, keep_default_na=False)
    out, bad = {}, []
    for r in F.itertuples(index=False):
        rid = int(r.rxn_id)
        if rid in out or r.m1 not in ("True", "False"):
            bad.append(f"d0_twins.csv rxn {rid}: duplicate row or m1 {r.m1!r}")
            continue
        out[rid] = dict(kind=r.kind, twin_of=int(r.twin_of) if r.twin_of != "" else None, m1=r.m1 == "True",
                        bind=folds(r.bind), ns=folds(r.twins_ns))
    return out, bad


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def write_csv(df, path, **kw):
    """df.to_csv(path, **kw) via <path>.<pid>.tmp + os.replace: a killed job never leaves a truncated csv (GB-atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, **kw)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def core_fold_map():
    """D1 core 'dipole/skeleton' (e.g. 'nitrile oxide/A') -> T1 outer fold (S_folds.csv fold_T1, one value per core;
    -1 = core never a test core in D1PAIR T1)."""
    S = pd.read_csv(S_FOLDS, usecols=["core", "fold_T1"]).drop_duplicates()
    assert S.groupby("core").fold_T1.nunique().max() == 1, "a core has two T1 folds"
    return dict(zip(S.core, S.fold_T1.astype(int)))


def d1_core_table():
    """C_compare (all 9,266 D1 rows): rxn_id -> (core 'dipole/skeleton', T1 fold, None). GEO1b reads no G1-derived
    column (same_ts_g1 is never read); the third slot stays None so the record layout is unchanged."""
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton"])
    fm = core_fold_map()
    return {int(r.rxn_id): (f"{r.dipole}/{r.skeleton}", int(fm.get(f"{r.dipole}/{r.skeleton}", -9)), None)
            for r in C.itertuples()}
