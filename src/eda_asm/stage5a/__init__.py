"""Stage 5-A: unimolecular fragmentation preprocessor (v2 clean rewrite).

Implements the six-pattern classifier (P0_BIMOL, P1_OPEN, P2_CLOSED,
P3_TETHER, P4_DISSOC, P5_HSHIFT) from ``stage_5A_v2_clean.md``.

Inputs are the 400 selected reactions in
``outputs/phase1/selected_reactions.csv``; outputs are written to
``outputs/stage5a/``.
"""
from .classify import (
    classify_reaction,
    detect_bond_changes,
    detect_h_migration,
)
from .fragmenters import (
    fragment_P0,
    fragment_P1,
    fragment_P2,
    fragment_P3,
    fragment_P4,
    fragment_P5_hshift,
)
from .pipeline import process_one_reaction
from .types import FragmentSpec, FragmentationResult

__all__ = [
    "FragmentSpec",
    "FragmentationResult",
    "classify_reaction",
    "detect_bond_changes",
    "detect_h_migration",
    "fragment_P0",
    "fragment_P1",
    "fragment_P2",
    "fragment_P3",
    "fragment_P4",
    "fragment_P5_hshift",
    "process_one_reaction",
]
