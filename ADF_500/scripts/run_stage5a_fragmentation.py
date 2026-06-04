"""Stage 5-A driver: classify the 400 selected reactions into P0/P1/P2/P3
and write per-reaction fragmentation results.

Usage:
    python scripts/run_stage5a_fragmentation.py
        [--selected outputs/phase1/selected_reactions.csv]
        [--output outputs/stage5a]
        [--xyz]      # also write R/TS/P xyz + per-fragment xyz files

Outputs under ``--output``:
    fragmentation_summary.json        — one entry per reaction (compact)
    pattern_distribution.json         — counts and tertile/source breakdown
    per_reaction/<reaction_id>/result.json
    per_reaction/<reaction_id>/{R,TS,P}.xyz                 (if --xyz)
    per_reaction/<reaction_id>/<role>_{R,TS,P}.xyz           (if --xyz)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from eda_asm.phase1.halo8_io import _ELEMENTS
from eda_asm.phase1.paths import SELECTED_CSV
from eda_asm.stage5a.loader import ReactionFrames, load_reaction_frames
from eda_asm.stage5a.pipeline import process_one_reaction
from eda_asm.stage5a.types import FragmentationResult


def _symbol(z: int) -> str:
    z = int(z)
    return _ELEMENTS[z] if 0 < z < len(_ELEMENTS) else f"Z{z}"


def _write_xyz(
    path: Path,
    numbers: np.ndarray,
    positions: np.ndarray,
    comment: str = "",
) -> None:
    lines = [f"{len(numbers)}", comment]
    for z, (x, y, zc) in zip(numbers, positions):
        lines.append(f"{_symbol(int(z)):2s} {x:14.8f} {y:14.8f} {zc:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def _write_xyz_fragment(
    path: Path,
    parent_numbers: np.ndarray,
    parent_positions: np.ndarray,
    atom_indices: np.ndarray,
    cap_meta: list[tuple[int, int, np.ndarray]] | None,
    comment: str,
) -> None:
    """Emit a fragment xyz at the given parent geometry, appending cap Hs.

    ``cap_meta`` entries are ``(parent_neighbour_idx, cap_local_idx, h_pos)``
    where ``h_pos`` was computed at R geometry. When we re-emit the
    fragment at TS or P geometry, we recompute the cap H position from
    the same parent neighbour so the cap moves with the cut bond.
    """
    atoms = list(map(int, atom_indices))
    cap_count = len(cap_meta) if cap_meta else 0
    lines = [f"{len(atoms) + cap_count}", comment]
    for a in atoms:
        x, y, z = parent_positions[a]
        lines.append(f"{_symbol(int(parent_numbers[a])):2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    if cap_meta:
        # Recompute cap H position at this geometry from the same f→t bond
        # direction we used at R. We don't have ``f_idx`` here directly —
        # cap_meta stores only the parent ``t_idx``. The fragment-local
        # atom adjacent to that t_idx is whichever fragment atom has the
        # original bond to t_idx in R; we resolve it by nearest atom in
        # this fragment at R geometry, which is unambiguous given how
        # caps were placed.
        for (t_idx, _local_cap_idx, _h_R) in cap_meta:
            # Find the fragment atom whose distance to t_idx at R is smallest;
            # that atom IS the f_idx (it was directly bonded in R).
            # We use parent_positions for the t_idx lookup at this geometry.
            atom_pos = parent_positions[atoms]
            d = np.linalg.norm(atom_pos - parent_positions[t_idx], axis=1)
            f_local = int(np.argmin(d))
            f_idx = atoms[f_local]
            vec = parent_positions[t_idx] - parent_positions[f_idx]
            n = float(np.linalg.norm(vec))
            unit = vec / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
            h_pos = parent_positions[f_idx] + 1.09 * unit
            lines.append(f"H  {h_pos[0]:14.8f} {h_pos[1]:14.8f} {h_pos[2]:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def _process(
    frames: ReactionFrames,
    out_dir: Path,
    write_xyz: bool,
) -> dict:
    try:
        result, dbg = process_one_reaction(
            frames.numbers,
            frames.positions_R,
            frames.positions_P,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "reaction_id": frames.reaction_id,
            "source": frames.source,
            "error": f"{type(e).__name__}: {e}",
        }

    rxn_dir = out_dir / "per_reaction" / frames.reaction_id
    rxn_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "reaction_id": frames.reaction_id,
        "source": frames.source,
        "n_atoms": int(frames.n_atoms),
        "energy_R": frames.energy_R,
        "energy_TS": frames.energy_TS,
        "energy_P": frames.energy_P,
        "activation_energy": frames.energy_TS - frames.energy_R,
        "ts_frame_idx": frames.ts_frame_idx,
        "frame_index_first": frames.frame_index_first,
        "frame_index_last": frames.frame_index_last,
        "result": result.to_dict(),
        "debug": dbg,
    }
    (rxn_dir / "result.json").write_text(json.dumps(record, indent=2))

    if write_xyz:
        comment = f"{frames.reaction_id} natoms={frames.n_atoms}"
        _write_xyz(rxn_dir / "R.xyz", frames.numbers, frames.positions_R, f"{comment} frame=R")
        _write_xyz(rxn_dir / "TS.xyz", frames.numbers, frames.positions_TS, f"{comment} frame=TS")
        _write_xyz(rxn_dir / "P.xyz", frames.numbers, frames.positions_P, f"{comment} frame=P")
        for frag in result.fragments:
            for label, pos in (("R", frames.positions_R),
                               ("TS", frames.positions_TS),
                               ("P", frames.positions_P)):
                _write_xyz_fragment(
                    rxn_dir / f"{frag.role}_{label}.xyz",
                    frames.numbers,
                    pos,
                    frag.atom_indices,
                    frag.cap_attachment,
                    comment=f"{frames.reaction_id} {frag.role} mult={frag.multiplicity} frame={label}",
                )

    # Compact summary entry
    return {
        "reaction_id": frames.reaction_id,
        "source": frames.source,
        "n_atoms": int(frames.n_atoms),
        "pattern": result.pattern,
        "n_fragments": len(result.fragments),
        "confidence": result.confidence,
        "notes": result.notes,
        "fragment_atoms": [
            [int(x) for x in frag.atom_indices.tolist()] for frag in result.fragments
        ],
        "fragment_roles": [frag.role for frag in result.fragments],
        "fragment_multiplicities": [int(frag.multiplicity) for frag in result.fragments],
        "n_caps": (
            int(len(result.cap_h_positions))
            if result.cap_h_positions is not None
            else 0
        ),
        "n_bond_changes": dbg["n_bond_changes"],
        "core_atoms": dbg["core_atoms"],
        "activation_energy": frames.energy_TS - frames.energy_R,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=SELECTED_CSV,
        help="Path to selected_reactions.csv (default: outputs/phase1/selected_reactions.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage5a"),
        help="Output directory (default: outputs/stage5a)",
    )
    parser.add_argument(
        "--xyz",
        action="store_true",
        help="Also write R/TS/P and per-fragment xyz files per reaction",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_reaction").mkdir(parents=True, exist_ok=True)

    cache = args.output / "frames_cache.pkl"
    if cache.exists():
        print(f"[stage5a] loading frames from cache {cache}", flush=True)
        frames_by_id = pickle.loads(cache.read_bytes())
    else:
        print(f"[stage5a] loading frames for {args.selected} …", flush=True)
        t0 = time.time()
        frames_by_id = load_reaction_frames(args.selected)
        print(f"[stage5a] loaded {len(frames_by_id)} reactions in {time.time()-t0:.1f}s",
              flush=True)
        cache.write_bytes(pickle.dumps(frames_by_id))
        print(f"[stage5a] cached frames to {cache}", flush=True)

    summary: list[dict] = []
    errors: list[dict] = []
    for rid in sorted(frames_by_id):
        rec = _process(frames_by_id[rid], args.output, write_xyz=args.xyz)
        if "error" in rec:
            errors.append(rec)
        else:
            summary.append(rec)

    (args.output / "fragmentation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if errors:
        (args.output / "errors.json").write_text(json.dumps(errors, indent=2))

    # Distribution stats
    df = pd.DataFrame(summary)
    pattern_counts = df["pattern"].value_counts().to_dict()
    by_source = {
        src: df[df["source"] == src]["pattern"].value_counts().to_dict()
        for src in sorted(df["source"].unique())
    }
    confidence_mean = df.groupby("pattern")["confidence"].mean().to_dict()
    n_bond_change_hist = {
        pat: {str(k): int(v) for k, v in Counter(sub["n_bond_changes"].astype(int)).items()}
        for pat, sub in df.groupby("pattern")
    }
    dist = {
        "n_reactions": len(summary),
        "n_errors": len(errors),
        "pattern_counts": pattern_counts,
        "pattern_counts_by_source": by_source,
        "mean_confidence_by_pattern": confidence_mean,
        "n_bond_changes_by_pattern": n_bond_change_hist,
    }
    (args.output / "pattern_distribution.json").write_text(json.dumps(dist, indent=2))

    print("[stage5a] done.")
    print(f"  total processed : {len(summary)} (errors={len(errors)})")
    print(f"  pattern counts  : {pattern_counts}")
    print(f"  by source       : {json.dumps(by_source, indent=2)}")
    print(f"  outputs in      : {args.output}")


if __name__ == "__main__":
    main()
