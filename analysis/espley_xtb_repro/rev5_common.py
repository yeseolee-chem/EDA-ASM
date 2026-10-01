#!/usr/bin/env python3
"""rev5_common.py — shared conventions of rev 5 (docs/specs/REV5_FEATURES_FIGURES.md).

Single source of truth for paths, the extended feature blocks B1..B6 (column names, produced by ext_features.py),
the arms, and every constant that is fixed before any rev 5 result exists (lockbox, selection rule, bootstrap,
Delta-epsilon floor, scan steps, contact / penetration parameters). Import it; do not copy its constants.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ---------------------------------------------------------------- paths
SCRATCH = Path(os.environ.get("R5_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev5"))
XTB_ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
G1_ROOT = Path(os.environ.get("G1_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1"))
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
META = Path(os.environ.get("ESPLEY_META", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv"))
ESP_DATA = Path(os.environ.get("ESPLEY_REPO_DATA", "/gpfs/tmp_cpu2/yeseo1ee/espley_compare"))
LABELS = Path(os.environ.get("ESPLEY_LABELS") or REPO / "labels_all.json")
FEAT_G1 = XTB_ROOT / "xtb_features_g1.parquet"          # rev 4 G1 features (never modified)
FEAT_EXT = XTB_ROOT / "xtb_features_ext_g1.parquet"     # rev 5 blocks B1..B6 (ext_features.py)
RES4 = HERE / "results_rev4"
RES5 = HERE / "results_rev5"
ROWS4 = RES4 / "rows_rev4.csv"
ROWS5 = RES5 / "rows_rev5.csv"
LOCKBOX = RES5 / "lockbox_ids.csv"
PREREG5B_JSON = RES5 / "prereg_rev5b.json"              # EXT_SEL blocks, written by select_blocks.py (C-3)

# ---------------------------------------------------------------- engines
XTB_VERSION = "6.7.1 (edcfbbe)"
TBLITE_VERSION = "0.7.0"
EH2KCAL = 627.5094740631
EH2EV = 27.211386245988
BOHR = 1.8897259886                                     # bohr per Angstrom

# ---------------------------------------------------------------- feature blocks (fixed before any rev 5 result)
FRAGS = ("dA", "dB", "rA", "rB")          # A = frag1 = dipole, B = dipolarophile; d = TS-geometry fragment, r = relaxed
REACT = ("dip_a", "dip_mid", "dip_b", "dph_a", "dph_b")   # xtb_slice.py definition; bond ad (a-a) is the shorter one
BLOCKS = {
    # B1 orbital overlap (tblite GFN2, ALPB water)
    "B1": ["mo_pauli_S2_occ", "mo_pauli_S2_front", "mo_oi_AB", "mo_oi_BA", "mo_oi_tot",
           "mo_S_HA_LB", "mo_S_HB_LA", "mo_dE_HA_LB", "mo_dE_HB_LA", "mo_S2dE_HA_LB", "mo_S2dE_HB_LA",
           "mo_S2dE_win_AB", "mo_S2dE_win_BA"]
          + [f"eps_{o}_{f}" for f in FRAGS for o in ("H", "L")],
    # B2 inter-fragment contacts (geometry only)
    "B2": [f"bm_{p}_{r}" for p in ("hh", "hH", "HH") for r in ("025", "035", "050")]
          + ["vdw_pen_sum", "vdw_n_075", "vdw_n_085", "vdw_n_100", "dmin_nf_1", "dmin_nf_2"]
          + [f"rdf_hh_{b}" for b in ("240", "260", "280", "300", "320", "340", "360", "380")],
    # B3 frozen-fragment multipole electrostatics + penetration proxies (xtb --json)
    "B3": ["mp_E_qmu", "mp_E_mumu", "mp_E_qTheta", "pen_damp_10", "pen_damp_20", "pen_ovl_10", "pen_ovl_20"],
    # B4 reactivity indices and response (xtb --vipea / --vfukui / D4 block; GEDT from the TS single point)
    "B4": [f"{k}_{f}" for f in FRAGS for k in ("mu", "eta", "omega")] + ["dN_rel", "dN_dist"]
          + [f"fk_{k}_{a}" for a in REACT for k in ("p", "m", "0")]
          + [f"alpha_{a}" for a in REACT] + [f"alpha_mol_{f}" for f in FRAGS] + ["gedt"],
    # B5 distance-sensitivity scan of fragment B along u (xtb single points)
    "B5": [f"scan_{k}_{x}" for x in ("Eint", "pauli", "oi", "elst") for k in ("slope", "curv", "m10", "p10")],
    # B6 TS character (Hammond): G1 hess/result.json
    # B6 product features (xtb_dErxn, prog_ad, prog_be) dropped by user decision 2026-10-01 (B-7 STOP: product
    # graph / mapping failures in 121 of 4,839 rows); TS-character features only
    "B6": ["nu_imag", "mode_share", "mode_async"],
}
BLOCK_ORDER = ("B1", "B2", "B3", "B4", "B5", "B6")
EXT_COLS = [c for b in BLOCK_ORDER for c in BLOCKS[b]]
assert len(EXT_COLS) == len(set(EXT_COLS)) == 109, len(EXT_COLS)

# B1
DE_FLOOR_EV = 1.0                     # Delta-epsilon floor for S^2 / Delta-epsilon terms
FRONT = 3                             # HOMO..HOMO-2 and LUMO..LUMO+2 windows
# B2
BM_RHO = {"025": 0.25, "035": 0.35, "050": 0.50}                   # Angstrom
VDW_S = {"075": 0.75, "085": 0.85, "100": 1.00}
RDF_EDGES = np.round(np.arange(2.4, 4.0 + 1e-9, 0.2), 1)          # 9 edges -> 8 bins
BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "Br": 1.85, "S": 1.80, "I": 1.98}
# B3
PEN_ALPHA = {"10": 1.0, "20": 2.0}                                # 1/Angstrom
GFN2_VALENCE = {"H": 1, "C": 4, "N": 5, "O": 6, "F": 7, "Cl": 7, "Br": 7, "S": 6, "I": 7}
# B5
SCAN_DELTAS = (-0.10, -0.05, 0.05, 0.10)                          # Angstrom, fragment B moved along u
SCAN_GATE_EH = 1e-6                                               # delta = 0 recomputation vs xtb_slice
# B-0 gates
GATE_E_EH = 1e-6
GATE_S_BLOCK = 1e-10
GATE_ORTHO = 1e-8
SMOKE_RXNS = (20, 105, 2434, 2721, 3452)
# B-7 gates
MAX_EXT_FAIL = 0.01

# ---------------------------------------------------------------- arms
BASE = "ESPLEY73"
TARGETS9 = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft",
            "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft"]
SEL_TARGETS = ["dft_elst_dft", "dft_pauli_dft", "dft_oi_dft"]      # selection criterion = mean dev-CV NMAE of these

# ---------------------------------------------------------------- Phase C / D constants (pre-registered)
SEED = 20261001
LOCKBOX_FRAC = 0.15                   # rows_rev5 -> lockbox via np.random.default_rng(SEED)
N_OUTER = 5                           # dev CV: KFold(5, shuffle=True, random_state=SEED)
N_INNER = 5                           # nested GridSearchCV: KFold(5, shuffle=True, random_state=SEED)
SEL_MIN_REL = 0.02                    # forward selection: relative criterion decrease >= 2 %
SEL_MIN_FOLDS = 4                     # ... and lower in >= 4 of the 5 outer folds
N_BOOT = 10_000                       # reaction-level bootstrap (rng SEED)
PROTO_A_SEEDS = [22, 23, 14, 1, 2]
LC_FRACS = [round(0.1 * k, 1) for k in range(1, 11)]               # learning-curve train fractions (D-4)

# ---------------------------------------------------------------- figure style (Phase E)
COLORS = {"Espley": "#2a78d6", "EXT_SEL": "#eb6834", "ESPLEY73": "#1baf7a", "ESPLEY46": "#eda100"}
MM = 1 / 25.4
FIG_W1, FIG_W2 = 89 * MM, 183 * MM
FONT_PT = 7


# ---------------------------------------------------------------- helpers
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, data):
    """str -> text, bytes -> binary, callable(tmp_path) -> writer; then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if callable(data):
        data(tmp)
    elif isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data)
    os.replace(tmp, path)


def write_json(path, obj):
    write_atomic(path, json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else
                                  (x.tolist() if hasattr(x, "tolist") else str(x))))


def nmae(y, p):
    """MAE / mean absolute deviation of the targets (as train_ml_single.mets)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    mad = float(np.mean(np.abs(y - y.mean())))
    return float(np.mean(np.abs(y - p))) / mad if mad > 0 else float("nan")


def ext_sel_blocks():
    """EXT_SEL block list, fixed by C-3 (prereg_rev5b.json); raises before C-3 exists."""
    return list(json.loads(PREREG5B_JSON.read_text())["ext_sel_blocks"])


def arm_columns(arm, base_cols):
    """Feature columns of an arm name: ESPLEY46/54/73 (base_cols = train_ml_single.FEATURE_SETS), BASE+Bk,
    BASE+Bj+Bk (forward-selection candidates, '+'-joined), EXT_ALL, EXT_SEL."""
    if arm in base_cols:
        return list(base_cols[arm])
    if arm == "EXT_ALL":
        blocks = list(BLOCK_ORDER)
    elif arm == "EXT_SEL":
        blocks = ext_sel_blocks()
    elif arm.startswith("BASE+"):
        blocks = arm.split("+")[1:]
    else:
        raise KeyError(arm)
    return list(base_cols[BASE]) + [c for b in BLOCK_ORDER if b in blocks for c in BLOCKS[b]]


def nadeau_bengio(d, n_test, n_train):
    """Corrected resampled t-test (Nadeau & Bengio 2003) on J paired differences d (e.g. per-seed MAE differences).
    Returns (mean, t, two-sided p, 95 % CI) with variance inflated by (1/J + n_test/n_train), df = J - 1."""
    from scipy import stats
    d = np.asarray(d, float)
    J = len(d)
    var = float(np.var(d, ddof=1))
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    m = float(d.mean())
    t = m / se if se > 0 else float("inf")
    p = float(2 * stats.t.sf(abs(t), J - 1))
    h = float(stats.t.ppf(0.975, J - 1) * se)
    return dict(mean=m, t=float(t), p=p, ci95=(m - h, m + h), J=J, n_test=float(n_test), n_train=float(n_train))
