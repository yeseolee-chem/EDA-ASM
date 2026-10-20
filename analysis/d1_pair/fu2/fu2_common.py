"""FU2 constants and paths (docs/specs/D1PAIR_FU2.md). Imports only d1pair_common (numpy-level), so it loads in every env
(venv5 for Q, env_espley for T / splits, env_d1fu2 for M / C)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # analysis/d1_pair/fu2
sys.path.insert(0, str(HERE.parent))
import d1pair_common as C  # noqa: E402

FU2_RES = C.RES / "fu2"
FU2_SCR = C.SCRATCH / "fu2"
ATOMS = FU2_SCR / "atoms"                                 # Q: <rxn_id>.npz
TOKENS = FU2_SCR / "tokens"                               # T: <rxn_id>.npz
SPLITS = FU2_SCR / "splits"                               # T1 / LODO / LOSO pair lists exported from d1_train.masks
UNITS = FU2_SCR / "units"                                 # M cache (not written before ADD2)
SMOKE = FU2_SCR / "smoke"                                 # C smoke test scratch (never read by M)
X182 = C.SCRATCH / "X182_d1.parquet"

CODE_VERSION = "d1pair-fu2-1"
TARGETS = list(C.TARGETS)                                 # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
PRIMARY = list(C.PRIMARY_TARGETS)                         # 8 channels + barrier (validation metric, main comparison)
CHANNELS9 = [t for t in TARGETS if t != "barrier"]        # identity: barrier = sum of these 9
I_BAR = TARGETS.index("barrier")
I_CH9 = [TARGETS.index(t) for t in CHANNELS9]
I_PRIM = [TARGETS.index(t) for t in PRIMARY]
SEED = C.SEED

# Q: extension values pass rate below which |mu|, f+, f- are dropped from every model (Q-1)
EXT_MIN_PASS = 0.99
QA_TOL = 1e-6                                             # Q-a: 30 recomputed columns vs X182_d1
QA_MAX_FAIL = 42                                          # 0.5 % of 8,337
QB_TOL = 1e-3                                             # Q-b: fragment charge sums
BOND_CAP = 8                                              # T-1 graph distance to the nearest reacting atom, capped

# M-1 training rules
LAMBDA = 0.1
LR, WD, DROPOUT, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 0.1, 64, 300, 30
FINAL_SEEDS = [SEED + i for i in range(5)]
ANTISYM_TOL = 1e-4                                        # gate M-1' / C1 (kcal/mol, float64 recomputation)
ANTISYM_N = 200


def assert_path(path):
    """FU2 writes only under $FU2_SCR or results/fu2 (spec 규칙: never M/units, M/units_fu, PREREG / ADD1, rev 4-6)."""
    import os
    C.assert_write_ok(path)
    p = Path(os.path.abspath(path))
    if not any(p == r or r in p.parents for r in (FU2_SCR.resolve(), FU2_RES.resolve(), FU2_SCR, FU2_RES)):
        print(f"STOP: FU2 write path {p} is outside {FU2_SCR} and {FU2_RES}", file=sys.stderr)
        sys.exit(3)
    return p


def write_atomic(path, data):
    assert_path(path)
    return C.write_atomic(path, data)


def write_json(path, obj):
    assert_path(path)
    return C.write_json_safe(path, obj)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())
