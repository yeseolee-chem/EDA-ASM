"""ASR (Activation-Strain Reactive) prediction — v1 (dipolar POC).

Implements ASR_Revised_Plan_v2_DipolarPOC §5: predict the 5-component
ASM/EDA decomposition (E_strain, Pauli, V_elst, E_orb, E_disp) directly
from (R, P) geometries using a frozen NequIP backbone + light prediction
head.

This is the v1 baseline. v2 will overwrite the head designs while keeping
the data + backbone interface stable.
"""

from .backbone import NequIPFeatureExtractor
from .data import (
    ASR_COMPONENTS,
    ASR_COMPONENT_SIGNS,
    AsrSample,
    load_label_table,
    iter_reaction_pairs,
)
from .models import BaselineB0, ModelM1, SignConstrainedHead

__all__ = [
    "NequIPFeatureExtractor",
    "ASR_COMPONENTS",
    "ASR_COMPONENT_SIGNS",
    "AsrSample",
    "load_label_table",
    "iter_reaction_pairs",
    "BaselineB0",
    "ModelM1",
    "SignConstrainedHead",
]


def __getattr__(name):
    # Lazy import so users without mace-torch can still use the NequIP path.
    if name == "MACEOFFFeatureExtractor":
        from .backbone_maceoff import MACEOFFFeatureExtractor
        return MACEOFFFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
