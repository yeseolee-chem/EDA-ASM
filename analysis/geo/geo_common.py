"""GEO-1 common paths and helpers (docs/specs/GEO1.md). Python 3.9 compatible (D1 run env) and the GPU env (env_d1fu2)."""
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
GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))
D0G2 = Path(os.environ.get("D0G2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2"))

# ---- D1 (read only)
D1_G2_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")
D1_G2_FINAL = D1_G2_ROOT / "g2_final"
D1_LABELS_G2 = D1_G2_ROOT / "G" / "labels_d1_g2.json"      # 9,266 D1 label rows (prep_g.py inputs)
D1_META_G2 = D1_G2_ROOT / "G" / "input_meta_d1_g2.csv"     # rxn_id, A_idx (frag1 = dipole, 0-based)
D1_PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
D1_JOBS = D1_PAIR / "d1_jobs"
D1_G1_FINAL = D1_PAIR / "g1_final"
D1_MANIFEST = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv")
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
S_FOLDS = REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"

# ---- D0 (read only)
D0_LABELS = REPO / "labels_all.json"
D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
D0_META = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv")
D0_G1_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
D0_EXCLUDE = {3090, 3766, 4252, 3400, 5783}                # labels_all non-ok (= g1_geom.EXCLUDE)

RMSD_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis")
D1_FOLD_COUNTS = {0: 1874, 1: 1879, 2: 1503, 3: 1435, 4: 2195, -1: 380}   # C_compare rows per T1 fold (checked in I)
G2RAW_PRIMARY = dict(n=840, share=0.1595, median=0.0155)                  # C_compare d_g2 vs d_dft, label-length rule
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


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
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
    """C_compare (all 9,266 D1 rows): rxn_id -> (core 'dipole/skeleton', T1 fold, same_ts_g1 as True/False/None)."""
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton", "same_ts_g1"])
    fm = core_fold_map()
    out = {}
    for r in C.itertuples():
        core = f"{r.dipole}/{r.skeleton}"
        s = r.same_ts_g1
        s = True if s in (True, "True") else (False if s in (False, "False") else None)
        out[int(r.rxn_id)] = (core, int(fm.get(core, -9)), s)
    return out
