"""ESPB (docs/specs/ESPB.md): paths, constants, grids and small I/O helpers shared by the analysis/espb scripts.

Every step reads its paths here. Imported in env_espley (P, A, B, C), env_d1fu2 (E) and reactot (F, R): standard library
only at import time. The sha256 / write_atomic / write_json / read_json helpers are copied from analysis/gnn3r2/gr_common.py
(write_atomic also takes a callable(tmp_path), as rev5_common.write_atomic, for the copied training code). The D1PAIR
constants below are written again with the same values; s0_start.py checks them against d1pair_common by import.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                   # analysis/espb
REPO = HERE.parents[1]                                   # $WT
RES = HERE / "results"

# ---------------------------------------------------------------- roots (spec header)
EB = Path(os.environ.get("ESPB_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espb"))
G2R = Path("/gpfs/tmp_cpu2/yeseo1ee/g2r")
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")
ESP_DIR = REPO / "analysis" / "espley_xtb_repro"         # r6_nn, train_ml_single (read only)
D1PAIR_DIR = REPO / "analysis" / "d1_pair"               # d1pair_common (read only; checked in S0)
GNN3_PACKAGE = REPO / "gnn3"                             # gnn3.metrics / gnn3.channels (read only)

# ---------------------------------------------------------------- GNN3R2 inputs (read only)
GR2_RES = REPO / "analysis" / "gnn3r2" / "results"
PAIRS = GR2_RES / "F_pairs_r.csv"
S_FOLDS = GR2_RES / "S_folds_r.csv"
S_SPLITS = GR2_RES / "S_splits_r.json"
SPLIT_DIR = GR2_RES / "splits"
LABEL_REPORT = GR2_RES / "B_label_report.json"
GNN3_CV = GR2_RES / "cv"                                 # gnn3_fold<k>.parquet (GNN3R2 CV predictions, committed)
GNN3_METRICS = GR2_RES / "eval" / "metrics.csv"
X182 = G2R / "lf" / "X182_lf.parquet"
LABELS_CR2 = LAD / "labels_d1_cr2.json"

# ---------------------------------------------------------------- outputs
P_DIR = EB / "P"
X46 = P_DIR / "X46.parquet"
P_REACTIONS = RES / "P_reactions.csv"
P_INNER = RES / "P_inner_folds.csv"
P_REPORT = RES / "P_report.json"
UNITS_EA = EB / "units" / "EA"
UNITS_EB = EB / "units" / "EB"
NN_WORK = EB / "nn"
PREREG_MD = RES / "PREREG_ESPB.md"
PREREG_JSON = RES / "prereg_espb.json"

# ---------------------------------------------------------------- data expectations (P gates)
N_PAIRS, N_REACTIONS, N_TEST = 8344, 8594, 8175
TEST_SIZES = {-1: 169, 0: 1682, 1: 1705, 2: 1414, 3: 1344, 4: 2030}
N_FOLDS = 5
# 10 per-reaction labels (= d1pair_common.D1_TCOL, LABEL_CHANNELS order of gnn3.constants)
D1_TCOL = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}
CHANNELS = ["strain", "inter", "solv", "barrier"]       # pair targets (= gnn3.constants.TARGETS)
TARGETS_EA = ["d1", "d2", "inter", "solv", "barrier"]   # E-A per-reaction targets (5-1)
TARGETS_EB = list(CHANNELS)                              # E-B pair targets (5-2)
MODELS = ["Ridge", "KRR", "SVR", "NN2", "NN4"]           # Espley models (no XGB, 0-3)
ARMS = ["EA", "EB"]

# ---------------------------------------------------------------- seeds and D1PAIR constants (same values, S0-checked)
SEED = 20261019                                          # = d1pair_common.SEED (bootstraps, M-1 sample, P-9 sample)
CV_SEED = 22                                             # = d1pair_common.CV_SEED (NN cv stage)
TUNE_SEED = 23                                           # = d1pair_common.TUNE_SEED = r6_nn.TUNE_SEED
NN_FINAL_SEEDS = (22, 23, 14, 1, 2)                      # X-10 (= train_ml_single.SEEDS)
N_INNER = 4                                              # = d1pair_common.N_INNER
SCALE_EPS = 1e-12                                        # = d1pair_common.SCALE_EPS
ANTISYM_TOL = 1e-6                                       # = d1pair_common.ANTISYM_TOL (gate M-1, kcal/mol)
ANTISYM_N = 200                                          # = d1pair_common.ANTISYM_N
GAMMAS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]   # = d1pair_common.GAMMAS (E-B kernels)
SVR_C = [1, 10, 100, 1000]                               # = d1pair_common.SVR_C
SVR_EPS = [0.05, 0.1, 0.5]                               # = d1pair_common.SVR_EPS
NN_SIZE = {"NN2": 2, "NN4": 4}                           # = d1pair_common.NN_SIZE
EB_RIDGE_ALPHA = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]   # = d1pair_common.grid_values("Ridge")["alpha"]
EB_KRR_ALPHA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]               # = d1pair_common.grid_values("KRR_rbf")["alpha"]

# ---------------------------------------------------------------- E-A grids (5-1: Espley ESI Table S3 U train_ml_single.GRIDS)
_TENTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EA_GRIDS = {
    "Ridge": {"regressor__ridge__alpha": [1e-4, 1e-3, 0.01] + _TENTHS + [1, 10, 100, 1000]},
    "KRR": {"regressor__kernelridge__alpha": [1e-5, 1e-4, 1e-3, 0.01] + _TENTHS + [1],
            "regressor__kernelridge__gamma": [None, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 1]},
    "SVR": {"regressor__svr__C": [1, 10, 30, 50, 100, 1000],
            "regressor__svr__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
            "regressor__svr__epsilon": [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]},
}
EA_GRID_CELLS = {"Ridge": 16, "KRR": 154, "SVR": 240}

# ---------------------------------------------------------------- evaluation (8-2)
BOOT_N, BOOT_SEED = 2000, 20261019


def model_name(arm: str, model: str) -> str:
    """'EA-Ridge' ... 'EB-NN4' (the model keys of the comparison tables)"""
    return f"{arm}-{model}"


def path_setup():
    """r6_nn / train_ml_single (analysis/espley_xtb_repro) and the gnn3 package (gnn3/gnn3) on sys.path."""
    for p in (str(ESP_DIR), str(GNN3_PACKAGE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path | str, data) -> None:
    """str / bytes, or a callable(tmp_path) that writes the file; tmp file then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    if callable(data):
        os.close(handle)
        data(temporary)
    else:
        with os.fdopen(handle, "wb") as out:
            out.write(data.encode() if isinstance(data, str) else data)
    os.replace(temporary, path)


def write_json(path: Path | str, obj) -> None:
    write_atomic(path, json.dumps(obj, indent=1, default=str, allow_nan=True) + "\n")


write_json_safe = write_json                             # name used by the copied D1PAIR code


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def list_sha(rows) -> str:
    """sha256 of a list of tuples, one comma-joined line each (= d1pair_common.list_sha)."""
    return hashlib.sha256(("\n".join(",".join(map(str, r)) for r in rows) + "\n").encode()).hexdigest()


def split_unit(k: int) -> dict:
    """GNN3R2 T1__f<k>.json: train / test / inner_train / inner_val pair ids"""
    return read_json(SPLIT_DIR / f"T1__f{k}.json")
