"""Result dataclasses for Stage 5-A fragmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FragmentSpec:
    atom_indices: np.ndarray            # (n_f,) int — indices into the parent molecule
    role: str                           # 'reactive_A' | 'reactive_B' | 'tether' | 'whole'
    multiplicity: int                   # 1 (closed-shell singlet) or 2 (doublet radical)
    cap_attachment: list[tuple[int, int, np.ndarray]] | None = None
    # cap_attachment entries: (orig_neighbour_idx, frag_local_idx_of_cap_H, h_position_xyz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_indices": [int(x) for x in self.atom_indices.tolist()],
            "role": self.role,
            "multiplicity": int(self.multiplicity),
            "cap_attachment": (
                [
                    {
                        "neighbour_in_parent": int(orig),
                        "cap_idx_in_fragment": int(loc),
                        "h_position": [float(c) for c in pos.tolist()],
                    }
                    for (orig, loc, pos) in self.cap_attachment
                ]
                if self.cap_attachment
                else None
            ),
        }


@dataclass(slots=True)
class FragmentationResult:
    pattern: str                                       # P0_BIMOL | P1_OPEN | P2_CLOSED | P3_TETHER
    fragments: list[FragmentSpec] = field(default_factory=list)
    cap_h_positions: np.ndarray | None = None          # (M, 3) — all cap H positions concatenated
    confidence: float = 1.0
    notes: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "fragments": [f.to_dict() for f in self.fragments],
            "n_fragments": len(self.fragments),
            "cap_h_positions": (
                [[float(v) for v in row] for row in self.cap_h_positions.tolist()]
                if self.cap_h_positions is not None and len(self.cap_h_positions) > 0
                else None
            ),
            "confidence": float(self.confidence),
            "notes": self.notes,
            "debug": self.debug,
        }
