"""Dataclasses for the Stage-0 fragmentation API."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FragmentationResult:
    """Output of run_fragmentation.

    Attributes
    ----------
    fragments : list[set[int]]
        Each entry is the set of atom indices belonging to one fragment.
        For ASM-EDA the convention is two fragments; rearrangement cases may
        carry only one (entire molecule) and rely on ``fallback_strategy``.
    migrating_atoms : list[dict]
        Atoms that lose at least one bond and gain at least one bond between
        R and P with balanced bond-order changes. Each entry has keys
        ``atom`` (int), ``from`` (list[int]) and ``to`` (list[int]).
    reactive_bonds : list[tuple[int, int]]
        Sorted, unique (i, j) pairs (i < j) whose bond order differs between
        R and P (i.e. nonzero off-diagonal ΔBE).
    cap_sites : dict[int, list[tuple[int, int]]]
        Mapping fragment index → list of (anchor_atom_in_fragment,
        partner_atom_in_other_fragment) tuples needing H caps.
    is_pure_rearrangement : bool
        True when removing reactive bonds + migrating atoms still leaves the
        molecular graph connected (i.e. no clean two-component split exists).
    fallback_strategy : str | None
        One of {"migration_clustering", "user_hint", "strain_only"} when
        rearrangement was triggered, else None.
    notes : list[str]
        Human-readable annotations from intermediate steps (helpful when
        ``verbose=True`` runs report unusual conditions).
    """

    fragments: list[set[int]]
    migrating_atoms: list[dict] = field(default_factory=list)
    reactive_bonds: list[tuple[int, int]] = field(default_factory=list)
    cap_sites: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    is_pure_rearrangement: bool = False
    fallback_strategy: str | None = None
    notes: list[str] = field(default_factory=list)
