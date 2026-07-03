"""Load R/TS/P frames for the 400 selected reactions from Halo8.

Reads ``outputs/phase1/selected_reactions.csv`` for the trajectory IDs,
``data/halo8_index/index.parquet`` for the (R, TS, P) frame indices and
the source DB file, then streams the matching DBs once each to extract
the three positions arrays per reaction.

Returns one ``ReactionFrames`` per reaction, indexed by ``reaction_id``.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eda_asm.phase1.halo8_io import (
    decode_data,
    decode_numbers,
    decode_positions,
    parse_trajectory_id,
)
from eda_asm.phase1.paths import DB_FILES, INDEX_PARQUET


@dataclass(slots=True)
class ReactionFrames:
    reaction_id: str
    source: str
    numbers: np.ndarray            # (N,)
    positions_R: np.ndarray        # (N, 3)
    positions_TS: np.ndarray       # (N, 3)
    positions_P: np.ndarray        # (N, 3)
    energy_R: float
    energy_TS: float
    energy_P: float
    n_atoms: int
    total_charge: float
    ts_frame_idx: int
    frame_index_first: int
    frame_index_last: int


def load_reaction_frames(
    selected_csv: Path,
    index_parquet: Path | None = None,
    db_files: list[Path] | None = None,
) -> dict[str, ReactionFrames]:
    if index_parquet is None:
        index_parquet = INDEX_PARQUET
    if db_files is None:
        db_files = DB_FILES

    selected = pd.read_csv(selected_csv)
    selected_ids = set(selected["reaction_id"].tolist())
    index_df = pd.read_parquet(index_parquet)
    index_df = index_df[index_df["reaction_id"].isin(selected_ids)].copy()
    if len(index_df) != len(selected_ids):
        missing = selected_ids - set(index_df["reaction_id"])
        raise RuntimeError(f"{len(missing)} selected ids not in index parquet: "
                           f"first few: {list(missing)[:3]}")

    # Group target frame indices by source DB.
    targets: dict[int, dict[str, set[int]]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for r in index_df.itertuples(index=False):
        ridx = int(r.frame_index_first)
        tsidx = int(r.ts_frame_idx)
        pidx = int(r.frame_index_last)
        targets[int(r.source_db_idx)][r.reaction_id] = {ridx, tsidx, pidx}
        meta[r.reaction_id] = {
            "source": r.source,
            "ridx": ridx,
            "tsidx": tsidx,
            "pidx": pidx,
            "energy_R": float(r.energy_R),
            "energy_TS": float(r.energy_TS),
            "energy_P": float(r.energy_P),
            "total_charge": float(r.total_charge),
            "n_atoms": int(r.n_atoms_max),
        }

    # Per-reaction frame storage: traj_id -> {frame_idx: positions}
    coords: dict[str, dict[int, np.ndarray]] = {tid: {} for tid in selected_ids}
    numbers_lookup: dict[str, np.ndarray] = {}

    for db_idx, db_path in enumerate(db_files, start=1):
        wanted = targets.get(db_idx)
        if not wanted:
            continue
        remaining = sum(len(v) for v in wanted.values())
        wanted = {k: set(v) for k, v in wanted.items()}  # copy, consumed below
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT natoms, numbers, positions, data FROM systems")
        while remaining > 0:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            for natoms, nb, pb, db_blob in rows:
                data = decode_data(db_blob)
                traj, frame_idx = parse_trajectory_id(str(data["dand_id"]))
                want_frames = wanted.get(traj)
                if not want_frames or frame_idx not in want_frames:
                    continue
                coords[traj][frame_idx] = decode_positions(pb, int(natoms))
                if traj not in numbers_lookup:
                    numbers_lookup[traj] = decode_numbers(nb)
                want_frames.discard(frame_idx)
                remaining -= 1
                if remaining == 0:
                    break
        conn.close()

    out: dict[str, ReactionFrames] = {}
    for rid in selected_ids:
        m = meta[rid]
        frames = coords[rid]
        if m["ridx"] not in frames or m["tsidx"] not in frames or m["pidx"] not in frames:
            raise RuntimeError(
                f"missing frames for {rid}: have {sorted(frames)}, "
                f"need {sorted({m['ridx'], m['tsidx'], m['pidx']})}"
            )
        out[rid] = ReactionFrames(
            reaction_id=rid,
            source=m["source"],
            numbers=numbers_lookup[rid],
            positions_R=frames[m["ridx"]],
            positions_TS=frames[m["tsidx"]],
            positions_P=frames[m["pidx"]],
            energy_R=m["energy_R"],
            energy_TS=m["energy_TS"],
            energy_P=m["energy_P"],
            n_atoms=m["n_atoms"],
            total_charge=m["total_charge"],
            ts_frame_idx=m["tsidx"],
            frame_index_first=m["ridx"],
            frame_index_last=m["pidx"],
        )
    return out
