"""Native reaction-dataset loaders (non-Halo8).

Both loaders expose a uniform ReactionRecord interface (R / TS / P + metadata)
so the downstream ADF EDA-ASM input-prep step can treat them identically.

Datasets:
    QMrxn20Loader              — von Rudorff et al. 2020 (E2 / SN2)
    DipolarCycloadditionLoader — Stuyver / Jorner / Coley 2023 ([3+2])
"""
from .base import HARTREE_TO_EV, KCAL_PER_MOL_TO_EV, Geometry, ReactionRecord, read_xyz
from .dipolar_cycloaddition import DipolarCycloadditionLoader
from .qmrxn20 import QMrxn20Loader

__all__ = [
    "HARTREE_TO_EV",
    "KCAL_PER_MOL_TO_EV",
    "Geometry",
    "ReactionRecord",
    "read_xyz",
    "DipolarCycloadditionLoader",
    "QMrxn20Loader",
]
