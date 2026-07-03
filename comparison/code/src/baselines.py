"""Pluggable Δ-baseline interface.

A *baseline* maps geometry + xTB cache → 4-channel kcal/mol prediction:
    {E_strain, V_elst, Pauli, E_oi}

Three variants:
  - `geom6` : the existing 6-d geometric descriptors (RMSD, vdW, 1/r, C6).
              Mirrors src/eda_asm/asr_v1/baseline_physics.py contents, but
              re-implemented locally so the v2 module does NOT modify any
              existing code (per spec §1 "기존 파일 수정 금지").
  - `xtb`   : xTB-derived scalar features only. Per-reaction:
                E_int^xtb, dipole_norm_complex/A/B, dipole_int,
                HOMO/LUMO/gap of complex/A/B, sum_q_A.
  - `xtb+geom6` : feature-union of the above two.

Each baseline trains a per-channel ridge (sklearn Ridge, α grid-searched)
on the train fold and predicts on test/OOD. Train fold labels are the
only fit signal — strict leakage prevention.
"""
from __future__ import annotations

from dataclasses import dataclass

import ase.io
import numpy as np
import pandas as pd
from ase.data import vdw_radii
from sklearn.linear_model import Ridge


# Channels predicted by the baseline (E_disp is excluded per spec §5).
TARGET_CHANNELS = ["E_strain", "V_elst", "Pauli", "E_oi"]

# Column names in the canonical ADF parquet for those channels.
TARGET_COL_MAP = {
    "E_strain": "E_strain_kcal",
    "V_elst":   "V_elst_kcal",
    "Pauli":    "Pauli_kcal",
    "E_oi":     "E_orb_kcal",
}

# Grimme D2 C6 (kcal/mol)·Å⁶ — same table the existing baseline_physics.py uses.
_C6_TABLE = {
    1: 0.14, 5: 3.13, 6: 1.65, 7: 1.11, 8: 0.70, 9: 0.61,
    14: 9.23, 15: 7.84, 16: 5.57, 17: 5.07, 34: 12.47, 35: 12.47, 53: 31.50,
}


def _pairwise(coords: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def _kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape != B.shape:
        n = min(len(A), len(B))
        if n == 0:
            return 0.0
        Ac = A[:n] - A[:n].mean(axis=0)
        Bc = B[:n] - B[:n].mean(axis=0)
        return float(np.sqrt(((Ac - Bc) ** 2).sum() / n))
    Ac = A - A.mean(axis=0); Bc = B - B.mean(axis=0)
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_mat = Vt.T @ D @ U.T
    return float(np.sqrt((((Ac @ R_mat) - Bc) ** 2).sum() / len(A)))


def geom6_descriptors(reaction) -> np.ndarray:
    """Return the 6-d geometric descriptor vector. R := geometry_fragA (TS-frozen
    fragment A) acts as a reactant-side proxy when the full R complex xyz isn't
    available; similarly for P := fragB. This matches the in-repo baseline's
    approximation for reactions without separate R/P complexes.
    """
    ts = ase.io.read(reaction.ts_xyz)
    a = ase.io.read(reaction.frag_a_xyz)
    b = ase.io.read(reaction.frag_b_xyz)
    R_pos = a.get_positions()    # proxy
    P_pos = b.get_positions()    # proxy
    TS_pos = ts.get_positions()
    Z_TS = ts.numbers

    d1 = _kabsch_rmsd(R_pos, TS_pos)
    d2 = _kabsch_rmsd(P_pos, TS_pos)

    dist_TS = _pairwise(TS_pos)
    iu = np.triu_indices_from(dist_TS, k=1)
    rij = dist_TS[iu]

    vdW = np.array([vdw_radii[int(z)] for z in Z_TS])
    vdW_pairs = (vdW[:, None] + vdW[None, :])[iu]
    over = np.maximum(vdW_pairs - rij, 0.0)
    d3 = float(np.exp(over / 0.3).sum())

    d4 = float((1.0 / rij).sum())

    C6 = np.array([_C6_TABLE.get(int(z), 1.0) for z in Z_TS])
    C6_pairs = np.sqrt(C6[:, None] * C6[None, :])[iu]
    d5 = float(-(C6_pairs / rij**6).sum())

    d6 = float(len(Z_TS))
    return np.array([d1, d2, d3, d4, d5, d6], dtype=float)


def xtb_features_vector(xtb_row: dict) -> np.ndarray:
    """Pack a per-reaction xtb dict (XtbResult.to_dict()) into a feature vector.
    Returns NaN-filled vector when xtb failed; downstream sklearn ridge replaces
    NaNs with column means (handled by RidgeBaseline below).
    """
    keys = [
        "E_int_kcal", "E_complex_kcal", "E_fragA_kcal", "E_fragB_kcal",
        "dipole_complex_norm", "dipole_fragA_norm", "dipole_fragB_norm",
        "dipole_int",
        "HOMO_complex", "LUMO_complex", "gap_complex",
        "HOMO_fragA", "LUMO_fragA", "gap_fragA",
        "HOMO_fragB", "LUMO_fragB", "gap_fragB",
        "sum_q_A_frag_atoms", "n_atoms",
    ]
    return np.array([xtb_row.get(k, np.nan) for k in keys], dtype=float)


XTB_FEATURE_NAMES = [
    "E_int_kcal", "E_complex_kcal", "E_fragA_kcal", "E_fragB_kcal",
    "dipole_complex_norm", "dipole_fragA_norm", "dipole_fragB_norm",
    "dipole_int",
    "HOMO_complex", "LUMO_complex", "gap_complex",
    "HOMO_fragA", "LUMO_fragA", "gap_fragA",
    "HOMO_fragB", "LUMO_fragB", "gap_fragB",
    "sum_q_A_frag_atoms", "n_atoms",
]


@dataclass
class RidgeBaseline:
    """Per-channel ridge: feature matrix → 4 channels."""
    variant: str  # "geom6" | "xtb" | "xtb+geom6"
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)

    def fit_predict(
        self,
        feat_train: np.ndarray, y_train: np.ndarray,
        feat_test: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Returns predictions (n_test, 4) and metadata dict."""
        # Impute NaNs with column-mean from train.
        col_mean = np.nanmean(feat_train, axis=0)
        feat_train = np.where(np.isnan(feat_train), col_mean[None, :], feat_train)
        feat_test = np.where(np.isnan(feat_test), col_mean[None, :], feat_test)
        # Standardise features.
        mu, sigma = feat_train.mean(axis=0), feat_train.std(axis=0) + 1e-8
        Xtr = (feat_train - mu) / sigma
        Xte = (feat_test - mu) / sigma

        preds = np.zeros((feat_test.shape[0], y_train.shape[1]))
        meta = {"variant": self.variant, "alpha_per_channel": {}}
        for ch_idx, ch in enumerate(TARGET_CHANNELS):
            # Tiny inner CV to pick alpha.
            best_alpha, best_score = None, np.inf
            for a in self.alphas:
                ridge = Ridge(alpha=a, fit_intercept=True)
                # Simple holdout: last 20% of train fold.
                n = len(Xtr); split = max(1, int(0.8 * n))
                ridge.fit(Xtr[:split], y_train[:split, ch_idx])
                pred = ridge.predict(Xtr[split:])
                score = float(np.mean((pred - y_train[split:, ch_idx]) ** 2))
                if score < best_score:
                    best_score = score; best_alpha = a
            ridge = Ridge(alpha=best_alpha, fit_intercept=True)
            ridge.fit(Xtr, y_train[:, ch_idx])
            preds[:, ch_idx] = ridge.predict(Xte)
            meta["alpha_per_channel"][ch] = best_alpha
        return preds, meta


def build_feature_matrix(
    rids: list[str],
    variant: str,
    geom6_map: dict[str, np.ndarray],
    xtb_map: dict[str, dict],
) -> np.ndarray:
    rows = []
    for rid in rids:
        parts = []
        if variant in ("geom6", "xtb+geom6"):
            parts.append(geom6_map[rid])
        if variant in ("xtb", "xtb+geom6"):
            parts.append(xtb_features_vector(xtb_map.get(rid, {})))
        rows.append(np.concatenate(parts))
    return np.stack(rows, axis=0)
