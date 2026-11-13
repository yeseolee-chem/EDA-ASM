"""GNN3R (docs/specs/GNN3R.md): paths, decisions and small I/O helpers shared by the analysis/gnn3r scripts.

Every step reads its paths from here (spec 2). Standard library only: imported in reactot (A, B), env_espley (S) and
env_d1fu2 (D, K, E, W). ``dataset()`` is the one place that builds the GNN3 training data (spec 6).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # analysis/gnn3r
REPO = HERE.parents[1]                                   # $WT
RES = HERE / "results"

# ---------------------------------------------------------------- roots (spec header, 2)
GR = Path(os.environ.get("GNN3R_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/gnn3r"))
G2R = Path("/gpfs/tmp_cpu2/yeseo1ee/g2r")
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")
D1PAIR_SCRATCH = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
D1_RUN = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run")

# ---------------------------------------------------------------- inputs (read only, spec 2 table)
TOKENS = G2R / "lf" / "fu2_lf" / "tokens"
TOKEN_REPORTS = REPO / "analysis" / "g2r" / "results" / "lf" / "fu2_lf"     # T_report.json, T_manifest.csv, Q_report.json
X182 = G2R / "lf" / "X182_lf.parquet"
LABELS_LF = G2R / "lf" / "labels_lf.json"
INPUT_META_LF = G2R / "lf" / "input_meta_lf.csv"
USABLE = REPO / "analysis" / "g2r" / "results" / "U_usable.csv"
TARGETS = REPO / "analysis" / "g2r" / "results" / "X_targets.csv"
G2R_MANIFEST = REPO / "analysis" / "g2r" / "results" / "g2r.json"
LABELS_CR2 = LAD / "labels_d1_cr2.json"
F_PAIRS_CR2 = LAD / "F_pairs_cr2.csv"
V_REPORT = REPO / "analysis" / "d1_ladder" / "results" / "V_report.json"
ROWATTR_ALL = D1PAIR_SCRATCH / "P" / "rowattr.pkl.all.pkl"
ROWATTR_KEPT = D1PAIR_SCRATCH / "P" / "rowattr.pkl"
D1_JOBS = D1PAIR_SCRATCH / "d1_jobs"
D1PAIR_CODE = REPO / "analysis" / "d1_pair"
D1PAIR_RES = D1PAIR_CODE / "results"
SIGMA = D1PAIR_CODE / "sigma_hansch1991.csv"
PAIRS_D1 = D1PAIR_RES / "pairs_d1.csv"
ANCHOR_D1 = D1PAIR_RES / "pairs_d1_anchor.csv"
P4_STEREO = D1PAIR_RES / "P4_stereo.csv"
P4_DROP = D1PAIR_RES / "P4_drop.csv"
S_FOLDS = D1PAIR_RES / "S_folds.csv"
REF_REACTIONS = REPO / "docs" / "specs" / "gnn3r_ref" / "ref_reactions_est.csv"
REF_PAIRS = REPO / "docs" / "specs" / "gnn3r_ref" / "ref_pairs_est.csv"
GNN3_PACKAGE = REPO / "gnn3"                             # import root of the gnn3 package (gnn3/gnn3)

# ---------------------------------------------------------------- outputs
ROWATTR_R = GR / "A" / "rowattr_r.pkl"
P4_CACHE = GR / "B" / "p4_cache.pkl"
B_RERUN = GR / "B" / "rerun"
CV = GR / "cv"
FINAL = GR / "final"
PAIRS_R = RES / "F_pairs_r.csv"
S_FOLDS_R = RES / "S_folds_r.csv"
SPLIT_DIR = RES / "splits"
S_SPLITS_R = RES / "S_splits_r.json"
LABEL_REPORT = RES / "B_label_report.json"
B_OUTPUT_NAMES = ("F_pairs_r.csv", "B_reactions.csv", "B_row_status.csv")      # compared by B-R

# ---------------------------------------------------------------- decisions and expected values (spec 0-2, 0-3, 1)
N_TARGETS = 9252                                         # D-1
N_USABLE = 8617                                          # D-2
N_NOT_USABLE = N_TARGETS - N_USABLE                      # 635 (B-7)
F = ["dipole", "skeleton", "series", "cfg", "orient", "face", "c3"]      # family key (D-6, = d1_pairs.F)
NEED_DIPOLES = ("nitrone", "azomethine ylide", "carbonyl ylide")         # c3 rows (build_rowattr.py `need`)
NEED_SKELETONS = ("AppE", "AppZ", "B", "Bp")
XB_RANK = {"Me": 0, "CO2Me": 1, "tBu": 2, "CHO": 3, "Ts": 4, "NMe2": 5}   # D-7 anchor b order; anything else 6
XB_RANK_OTHER = 6
POS_RANK = {"p": 0, "m": 1, "o": 2, "direct": 3}
EXPECT_REACTIONS, EXPECT_PAIRS = 8604, 8355              # 0-3 local estimate
RANGE_REACTIONS, RANGE_PAIRS = (8432, 8776), (8188, 8522)   # B-6 (+-2 %)
EXPECT_TEST = {-1: 170, 0: 1686, 1: 1706, 2: 1414, 3: 1344, 4: 2035}     # spec 5 (local estimate, reported only)
N_FOLD_CORES = 50                                        # S_folds.csv cores (spec 5-1)
N_INNER = 4                                              # GroupKFold(4), group = core (D-9)


def sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path | str, data: bytes | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "wb") as out:
        out.write(data.encode() if isinstance(data, str) else data)
    os.replace(temporary, path)


def write_json(path: Path | str, obj) -> None:
    write_atomic(path, json.dumps(obj, indent=1, default=str, allow_nan=True) + "\n")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def copy_atomic(source: Path | str, target: Path | str) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(handle)
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def import_gnn3():
    """Put the gnn3 package (gnn3/gnn3) on sys.path, as gnn3/scripts/*.py do."""
    if str(GNN3_PACKAGE) not in sys.path:
        sys.path.insert(0, str(GNN3_PACKAGE))


def import_d1pair():
    """Put analysis/d1_pair on sys.path (d1pair_common adds analysis/espley_xtb_repro for fragmenter)."""
    if str(D1PAIR_CODE) not in sys.path:
        sys.path.insert(0, str(D1PAIR_CODE))


def dataset():
    """The GNN3R training data (spec 6): gnn3.data.PairDataset on the g2r inputs and the GNN3R pairs / splits."""
    import_gnn3()
    from gnn3.data import PairDataset
    return PairDataset(token_dir=TOKENS, report_dir=TOKEN_REPORTS, global_table=X182,
                       pair_labels=RES / "F_pairs_r.csv", split_dir=RES / "splits", split_report=RES / "S_splits_r.json",
                       pair_attributes=RES / "S_folds_r.csv", label_report=RES / "B_label_report.json")
