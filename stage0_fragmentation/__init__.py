"""Stage-0: BE-matrix-based fragment partitioning for ASM-EDA.

See ``stage0_fragmentation_spec.md`` (project root) for the full algorithm
and references (FlowER / Ugi-Dugundji BE matrix, ASM protocol, etc.).
"""
from .api import run_fragmentation
from .be_matrix import build_be_matrix, validate_be_matrix
from .types import FragmentationResult

__all__ = [
    "FragmentationResult",
    "build_be_matrix",
    "run_fragmentation",
    "validate_be_matrix",
]
