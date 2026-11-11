"""Every constant of the GNN3 model in one place: name, value, unit and where the value came from.

The values are copied from the code that produced the published GNN3 results (CH3M units D__lf3__S__GNN__f{0..4}).
Changing any of them changes the model; they are documented here, not tuned here.

Source abbreviations used below:
    fu2_common = analysis/d1_pair/fu2/fu2_common.py   (training rules, "M-1")
    models     = analysis/d1_pair/fu2/models.py       (network definition)
    fu2_train  = analysis/d1_pair/fu2/train.py        (fit loop, standardisation)
    ch3_common = analysis/ch3/ch3_common.py           (3-channel targets)
    data_ch3   = analysis/ch3/data_ch3.py             (3-channel index constants)
    d1pair     = analysis/d1_pair/d1pair_common.py
    dep_common = analysis/deploy/dep_common.py        (token layout of the "lf" inputs)
    tokens     = analysis/d1_pair/fu2/tokens.py       (atom-token columns)
    eval_ch3   = analysis/ch3/eval_ch3.py             (evaluation definitions)
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- targets
#: The 10 per-reaction labels of the D1 data set, in the column order of the label file (ch3_common.TARGETS10).
#: All are energies in kcal/mol. "c_ghost" (bsse_shift_kcal) = interaction energy from the separate fragment single
#: points minus the EDA bonding energy e_bond: the ghost-basis (BSSE) shift plus the move of the fragments into the TS
#: cavity; d1 + d2 + e_bond + c_ghost = barrier exactly. The 6 EDA terms reproduce e_bond to < 0.1 kcal/mol (ORCA
#: table closure), so the 10 labels add up to the barrier within 0.1 kcal/mol (max 0.096).
LABEL_CHANNELS: tuple[str, ...] = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier")

#: The 4 quantities GNN3 predicts, in output-column order (ch3_common.TARGETS3).
TARGETS: tuple[str, ...] = ("strain", "inter", "solv", "barrier")

#: Which labels add up to each predicted channel (ch3_common.GROUPS).
CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "strain": ("d1", "d2"),
    "inter": ("elst", "pauli", "oi", "disp", "c_ghost"),
    "solv": ("cpcm", "cds"),
    "barrier": ("barrier",),
}

#: Output column of the barrier (fu2_common.I_BAR in 3-channel mode, data_ch3.MODE3).
BARRIER_INDEX: int = 3
#: Output columns whose sum should equal the barrier (fu2_common.I_CH9 in 3-channel mode).
CHANNEL_INDICES: list[int] = [0, 1, 2]
#: Output columns averaged into the validation metric (fu2_common.I_PRIM in 3-channel mode): all four.
VALIDATION_INDICES: list[int] = [0, 1, 2, 3]

# --------------------------------------------------------------------------- training rules (fu2_common, "M-1")
#: Weight of the identity penalty |barrier - (strain + inter + solv)| / s_barrier in the loss (fu2_common.LAMBDA).
IDENTITY_PENALTY_WEIGHT: float = 0.1
#: AdamW learning rate and weight decay (fu2_common.LR, WD).
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
#: Dropout probability used in every dropout layer (fu2_common.DROPOUT).
DROPOUT: float = 0.1
#: Pairs per optimisation step (fu2_common.BATCH).
BATCH_SIZE_PAIRS: int = 64
#: Length of the cosine learning-rate schedule and the maximum number of epochs (fu2_common.MAX_EPOCHS).
MAX_EPOCHS: int = 300
#: Early stopping: stop when the validation metric has not improved for this many epochs (fu2_common.PATIENCE).
PATIENCE_EPOCHS: int = 30
#: Seed of the tuning fits (fu2_common.SEED = d1pair.SEED).
BASE_SEED: int = 20261019
#: Seeds of the five final fits whose predictions are averaged (fu2_common.FINAL_SEEDS).
FINAL_SEEDS: list[int] = [BASE_SEED + i for i in range(5)]
#: Feature columns whose standard deviation on the training reactions is below this are not scaled (d1pair.SCALE_EPS).
SCALE_EPSILON: float = 1e-12

#: Hyper-parameter grid of the GNN (models.GRID["GNN"]): number of message-passing layers x hidden width.
GRID: list[dict[str, int]] = [dict(n_layers=layers, width=width) for layers in (4, 6) for width in (64, 128)]

# --------------------------------------------------------------------------- network geometry (models.GNN defaults)
#: Number of Gaussian radial-basis functions that encode an inter-atomic distance (models.GNN n_rbf).
N_RADIAL_BASIS: int = 32
#: The radial-basis centres span [0, this] in angstrom (models.GNN d_max).
RADIAL_BASIS_MAX_DISTANCE_ANGSTROM: float = 4.0
#: Number of edge types: 0 = no bond (masked out), 1 = covalent bond, 2 = forming bond of the TS (models.MPLayer).
N_EDGE_TYPES: int = 3
#: Additive attention mask for padded atoms; finite so that fully padded rows give no NaN (models.NEG).
ATTENTION_MASK_VALUE: float = -1e9
#: The model variant of the published results: atom tokens + the 182 global features ("A+G", models.VARIANTS).
VARIANT: str = "A+G"

# --------------------------------------------------------------------------- inference / checks
#: Pairs per forward pass at prediction time (fu2_train.predict bs).
PREDICT_BATCH_PAIRS: int = 256
#: Pairs per forward pass in the float64 antisymmetry check (fu2_train.gate_m1).
ANTISYMMETRY_BATCH_PAIRS: int = 32
#: Number of random test pairs used by the antisymmetry check (fu2_common.ANTISYM_N).
ANTISYMMETRY_CHECK_PAIRS: int = 200
#: Largest allowed |f(a,b) + f(b,a)| and |f(a,a)| in kcal/mol (fu2_common.ANTISYM_TOL).
ANTISYMMETRY_TOLERANCE_KCAL: float = 1e-4

# --------------------------------------------------------------------------- input layout ("lf" atom tokens)
#: Atomic numbers of the one-hot element block, in column order (dep_common.ELEMENTS).
ELEMENTS: tuple[int, ...] = (1, 6, 7, 8, 9, 14, 16, 17, 35)
#: Binary per-atom flags (tokens.BIN_COLS); the first one, frag_B, is also the fragment id (0 = dipole, 1 = dipolarophile).
BINARY_COLUMNS: tuple[str, ...] = ("frag_B", "dip_a", "dip_mid", "dip_b", "dph_a", "dph_b")
#: Numeric per-atom features (tokens.NUM_BASE + tokens.NUM_EXT).
NUMERIC_COLUMNS: tuple[str, ...] = ("bond_dist", "q_rel", "q_frag", "q_TS", "dq_dist", "dq_int", "v_rel", "v_frag", "v_TS",
                                    "dv_dist", "dv_int", "mu_abs", "f_plus", "f_minus")
#: Width of one atom token = one-hot elements + binary flags + numeric features (= 29).
ATOM_TOKEN_WIDTH: int = len(ELEMENTS) + len(BINARY_COLUMNS) + len(NUMERIC_COLUMNS)
#: Number of global (whole-reaction) features (the "182 columns").
N_GLOBAL_FEATURES: int = 182
#: Token / atom versions of the "lf" inputs (dep_common.TOKEN_VERSION, ATOM_VERSION).
TOKEN_VERSION: str = "fu2-tokens-3-lf"
ATOM_VERSION: str = "fu2-atoms-3-lf"
#: Tolerance of the token glob vs the 182-column table check, relative to max(1, |value|) (data_lf.GLOB_REL_TOL).
GLOBAL_TABLE_RELATIVE_TOLERANCE: float = 1e-12

# --------------------------------------------------------------------------- splits
#: Number of outer cross-validation folds of the T1 split (data_lf.NF).
N_FOLDS: int = 5
#: Names of the four pair lists of a fold (data_lf.FOLD_KEYS).
FOLD_LISTS: tuple[str, ...] = ("train", "test", "inner_train", "inner_val")

# --------------------------------------------------------------------------- evaluation (eval_ch3)
#: |y| at or above which the sign of a pair is scored and a pair enters the cause analysis (eval_ch3.SIGN_MIN), kcal/mol.
SIGN_THRESHOLD_KCAL: float = 1.0
#: Minimum number of pairs in a cell (dipole x cfg x series x position x substituent) for the cell-mean analysis.
CELL_MIN_PAIRS: int = 4
#: Minimum number of points (substituents + the H reference) in a family for the ranking analysis.
RANK_MIN_POINTS: int = 8
#: |Δy| bins of the pairwise ordering accuracy (eval_ch3.RANK_BINS), kcal/mol: RT ln 2, 1, 2 at 298 K.
RANK_BINS: tuple[tuple[float, float, str], ...] = ((0.41, 1.0, "0.41-1"), (1.0, 2.0, "1-2"), (2.0, float("inf"), ">2"))
#: Substituent positions of the evaluation tables (eval_ch3.POSITIONS); "p" = para is the headline position.
POSITIONS: tuple[str, ...] = ("all", "p", "m", "o", "direct")
#: Core bootstrap: number of resamples and seed (eval_geo2.N_BOOT / SEED).
BOOTSTRAP_RESAMPLES: int = 2000
BOOTSTRAP_SEED: int = 20261019
#: Applicability threshold = this percentile of the test-pair seed spread (eval_ch3.AD_PCT).
APPLICABILITY_PERCENTILE: float = 95.0

# --------------------------------------------------------------------------- default file locations (override by env)
SCRATCH = Path(os.environ.get("GNN3_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/gnn3"))
#: Regenerated "lf" inputs: atom tokens (fu2_lf/tokens/<rxn_id>.npz) and the 182-column table (X182_lf.parquet).
LF_INPUTS = Path(os.environ.get("GNN3_LF_INPUTS", "/gpfs/tmp_cpu2/yeseo1ee/ch3/lf"))
#: Reports of the token build (T_report.json, T_manifest.csv, Q_report.json) committed in the repository.
REPOSITORY = Path(__file__).resolve().parents[2]
TOKEN_REPORTS = Path(os.environ.get("GNN3_TOKEN_REPORTS", REPOSITORY / "analysis/ch3/results/lf_inputs/fu2_lf"))
#: Pair labels: one row per pair, dd_<label> = label(a) - label(b) for the 10 labels.
PAIR_LABELS = Path(os.environ.get("GNN3_PAIR_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv"))
#: T1 cross-validation split files T1__f<k>.json and the report holding their sha256.
SPLIT_DIR = Path(os.environ.get("GNN3_SPLIT_DIR", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits"))
SPLIT_REPORT = REPOSITORY / "analysis/d1_pair/results/fu2/S_splits.json"
#: Per-pair attributes used in evaluation (core, family, P_main / P_sens membership, in_T_main).
PAIR_ATTRIBUTES = REPOSITORY / "analysis/d1_pair/results/S_folds.csv"
#: Report proving the label file is the validated one (its "outputs" block holds the label sha256).
LABEL_REPORT = REPOSITORY / "analysis/d1_ladder/results/V_report.json"
#: The published GNN3 cross-validation units and predictions (read only; used by the equivalence checks).
ORIGINAL_UNITS = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3m/units")
ORIGINAL_RESULTS = REPOSITORY / "analysis/ch3m/results"
