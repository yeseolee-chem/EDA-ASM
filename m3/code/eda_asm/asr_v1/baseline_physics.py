"""Cheap physics-inspired per-reaction baseline for Δ-learning.

Implements:
  - ``compute_descriptors(R, TS, P)`` → 6-vector of deterministic, geometry-
    only descriptors (no atom mapping, no charges, no QM).
  - ``LinearBaseline`` — ridge regression fit per fold on the train labels,
    used to predict the 5-channel ASR baseline. The ML head then learns the
    residual y - baseline.

Per-channel descriptor → label correspondence (rough physical intent):
  d1, d2 = Kabsch RMSD(R↔TS), RMSD(P↔TS)
            → E_strain (geometric distortion proxies)
  d3     = Σ_pairs exp(-(r_ij - r_vdW_sum) / 0.3) at TS
            → Pauli (short-contact repulsion proxy)
  d4     = Σ_pairs 1/r_ij at TS
            → V_elst (compactness ↔ electrostatic strength)
  d5     = -Σ_pairs √(C6_i C6_j) / r_ij^6 at TS  (negative)
            → E_disp (Grimme D2-style dispersion)
  d6     = n_atoms at TS
            → intercept-like scale factor (couples to every channel)

All sums are over UNORDERED pairs i<j to avoid double counting.
Distances in Å. The C6 table uses Grimme D2-style atomic C6 coefficients
(Hartree·Bohr⁶) converted to kcal·Å⁶/mol via × 627.509 × 0.52918⁶.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import vdw_radii


# Grimme D2 atomic C6 coefficients [J·nm⁶/mol] from PCCP 2006, 8, 5287.
# Converted to (kcal/mol)·Å⁶ : × 1000 / 4.184  (J/mol → kcal/mol)
#                              × 10⁶          (nm⁶ → Å⁶ : 10⁻⁶ → factor in denominator)
# Net: J/mol·nm⁶ × (1/4184) kcal/J × 10⁶ Å⁶/nm⁶ = J/mol·nm⁶ × 239.006 ≈ kcal/mol·Å⁶
# We list the converted values directly:
_C6_TABLE_KCAL_A6 = {
    1:   0.14,
    5:   3.13,   # B
    6:   1.65,
    7:   1.11,
    8:   0.70,
    9:   0.61,
    14:  9.23,   # Si
    15:  7.84,
    16:  5.57,
    17:  5.07,
    34: 12.47,   # Se (close to Br for our use)
    35: 12.47,
    53: 31.50,
}


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix; diagonal forced to +inf so 1/r and
    exp() don't divide by zero. Shape (n, n)."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """RMSD of A → B after centroid + rotation alignment.

    Atom order in A and B must already correspond. We do NOT do permutation
    matching — for the dipolar set R has shape r0+r1 and TS has its own
    autodE-derived ordering, so the per-element rotation alignment is the
    best we can do without atom mapping; the resulting RMSD is an upper
    bound on the true rearrangement extent, which is still a useful
    monotonic proxy for "how much did the system move".
    """
    if A.shape != B.shape:
        return float("nan")
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_mat = Vt.T @ D @ U.T
    Ac_rot = Ac @ R_mat
    return float(np.sqrt(((Ac_rot - Bc) ** 2).sum() / len(A)))


def compute_descriptors(R: Atoms, TS: Atoms, P: Atoms) -> np.ndarray:
    """Return shape (6,) float32 descriptor vector. Deterministic, no fitting."""
    R_pos = R.get_positions()
    TS_pos = TS.get_positions()
    P_pos = P.get_positions()
    Z_TS = TS.numbers

    # 1, 2 — strain proxies
    d1 = kabsch_rmsd(R_pos, TS_pos)
    d2 = kabsch_rmsd(P_pos, TS_pos)

    # Pairwise quantities at TS
    dist_TS = _pairwise_distances(TS_pos)  # (n, n), Å, diag = inf
    iu = np.triu_indices_from(dist_TS, k=1)
    rij = dist_TS[iu]

    # 3 — Pauli short-contact proxy: only pairs closer than vdW sum
    vdW = np.array([vdw_radii[int(z)] for z in Z_TS])
    vdW_sum = vdW[:, None] + vdW[None, :]
    vdW_pairs = vdW_sum[iu]
    over = np.maximum(vdW_pairs - rij, 0.0)               # ≥ 0
    d3 = float(np.exp(over / 0.3).sum() - len(rij))       # exp(0)=1 baseline removed

    # 4 — 1/r Coulomb proxy (positive)
    d4 = float((1.0 / rij).sum())

    # 5 — D2-style dispersion proxy (negative)
    c6 = np.array([_C6_TABLE_KCAL_A6.get(int(z), 0.0) for z in Z_TS])
    c6_pairs = np.sqrt(c6[:, None] * c6[None, :])[iu]
    d5 = -float((c6_pairs / (rij ** 6 + 1e-9)).sum())

    # 6 — system size
    d6 = float(len(TS))

    return np.array([d1, d2, d3, d4, d5, d6], dtype=np.float32)


# ===== Ridge baseline per-channel =============================================


class LinearBaseline:
    """Per-channel ridge regression: label_c ≈ β_0c + Σ β_kc · d_k.

    Use ``fit(D_train, Y_train)`` on a train fold then ``predict(D_all)`` for
    the full N (train + val) so the baseline value of every reaction is
    available for both the loss and the final prediction.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.W_: np.ndarray | None = None     # (d_in + 1, 5)
        self.d_in_: int | None = None
        self.d_mean_: np.ndarray | None = None
        self.d_std_: np.ndarray | None = None

    def fit(self, D_train: np.ndarray, Y_train: np.ndarray) -> "LinearBaseline":
        """Fit ridge regression on (N_train, d_in) descriptors and (N_train, 5) labels."""
        D = np.asarray(D_train, dtype=np.float64)
        Y = np.asarray(Y_train, dtype=np.float64)
        # Z-score descriptors per dimension for numerical stability.
        mean = D.mean(axis=0)
        std = D.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        Dn = (D - mean) / std
        # Augment with intercept column.
        X = np.concatenate([Dn, np.ones((Dn.shape[0], 1))], axis=1)
        n_feat = X.shape[1]
        # Ridge: W = (XᵀX + αI)^-1 Xᵀ Y; do not regularize intercept.
        reg = self.alpha * np.eye(n_feat)
        reg[-1, -1] = 0.0
        W = np.linalg.solve(X.T @ X + reg, X.T @ Y)         # (d_in+1, 5)
        self.W_ = W.astype(np.float32)
        self.d_in_ = D.shape[1]
        self.d_mean_ = mean.astype(np.float32)
        self.d_std_ = std.astype(np.float32)
        return self

    def predict(self, D: np.ndarray) -> np.ndarray:
        """Return (N, 5) baseline predictions in kcal/mol."""
        assert self.W_ is not None, "LinearBaseline not fitted"
        Dn = (np.asarray(D, dtype=np.float32) - self.d_mean_) / self.d_std_
        X = np.concatenate([Dn, np.ones((Dn.shape[0], 1), dtype=np.float32)], axis=1)
        return X @ self.W_                                      # (N, 5)

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "W": self.W_,
            "d_in": self.d_in_,
            "d_mean": self.d_mean_,
            "d_std": self.d_std_,
        }

    def load_state_dict(self, s: dict) -> None:
        self.alpha = float(s["alpha"])
        self.W_ = np.asarray(s["W"], dtype=np.float32)
        self.d_in_ = int(s["d_in"])
        self.d_mean_ = np.asarray(s["d_mean"], dtype=np.float32)
        self.d_std_ = np.asarray(s["d_std"], dtype=np.float32)


DESCRIPTOR_NAMES = (
    "rmsd_RT_A", "rmsd_PT_A",
    "pauli_overlap_proxy",
    "inv_r_sum_TS",
    "dispersion_D2_proxy",
    "n_atoms",
)
N_DESCRIPTORS = 6
