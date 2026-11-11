"""From the 10 activation-energy labels to the 3 channels (+ barrier) that GNN3 predicts.

The activation strain model splits the barrier of a cycloaddition into the energy needed to distort the two reactants
into their transition-state shapes (strain) and the energy they gain by interacting in that shape (interaction). The
energy decomposition analysis (EDA) further splits the interaction into electrostatics, Pauli repulsion, orbital
interaction and dispersion, and the solvent model adds two solvation terms. GNN3 predicts three coarse channels:

    strain  = d1 + d2                               (distortion of the dipole + of the dipolarophile)
    inter   = elst + pauli + oi + disp + c_ghost    (gas-phase interaction incl. the counterpoise shift)
    solv    = cpcm + cds                            (electrostatic + non-electrostatic SMD solvation)
    barrier = barrier                               (predicted directly, not as the sum)

so that strain + inter + solv = barrier up to the label closure error (< 0.1 kcal/mol, the identity the loss penalises).
All values are electronic energies in kcal/mol.
"""
from __future__ import annotations

import numpy as np

from .constants import CHANNEL_GROUPS, LABEL_CHANNELS, TARGETS


def channel_matrix() -> np.ndarray:
    """The 10 x 4 summation matrix M with M[i, j] = 1 when label i belongs to channel j.

    Returns
    -------
    numpy.ndarray, shape [10, 4], float64
        Rows follow ``LABEL_CHANNELS``, columns follow ``TARGETS``.

    Notes
    -----
    Built element by element from ``CHANNEL_GROUPS`` exactly like ``ch3_common.M_ROWS`` so that ``y10 @ M`` gives the
    same floating-point numbers as the original pipeline.

    Examples
    --------
    >>> channel_matrix().shape
    (10, 4)
    """
    rows = tuple(tuple(1.0 if label in CHANNEL_GROUPS[target] else 0.0 for target in TARGETS) for label in LABEL_CHANNELS)
    return np.asarray(rows, dtype=np.float64)


def labels_to_channels(labels_kcal: np.ndarray) -> np.ndarray:
    """Sum 10-label values into the 4 GNN3 targets.

    Parameters
    ----------
    labels_kcal : numpy.ndarray, shape [n, 10]
        Values in ``LABEL_CHANNELS`` order (kcal/mol); per-reaction labels or pair differences ΔΔ alike.

    Returns
    -------
    numpy.ndarray, shape [n, 4]
        strain, inter, solv, barrier in kcal/mol.

    Examples
    --------
    >>> labels_to_channels(np.eye(10))[0]   # d1 contributes only to strain
    array([1., 0., 0., 0.])
    """
    labels_kcal = np.asarray(labels_kcal, dtype=np.float64)
    if labels_kcal.shape[-1] != len(LABEL_CHANNELS):
        raise ValueError(f"expected {len(LABEL_CHANNELS)} label columns, got {labels_kcal.shape[-1]}")
    return labels_kcal @ channel_matrix()
