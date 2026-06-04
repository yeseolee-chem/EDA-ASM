"""Stage 3.2 — Pre-compute bond changes for every Halo8 trajectory.

Reads the index from Stage 3.1 (knows R / TS frame indices for each
trajectory), then streams the source DBs again to pull just the required
two frames worth of positions per trajectory and computes bond changes.
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bonds import bond_changes
from .halo8_io import (
    decode_data,
    decode_numbers,
    decode_positions,
    parse_trajectory_id,
)
from .logging_setup import get_logger, log_header
from .paths import BOND_CHANGES_PARQUET, DB_FILES, INDEX_PARQUET, ensure_dirs


def _needed_frames(index_df: pd.DataFrame) -> dict[int, dict[str, set[int]]]:
    """Return {db_idx: {traj_id: {R_idx, TS_idx, P_idx}}}."""
    needed: dict[int, dict[str, set[int]]] = defaultdict(dict)
    for r in index_df.itertuples(index=False):
        ridx = int(r.frame_index_first)
        tsidx = int(r.ts_frame_idx)
        pidx = int(r.frame_index_last)
        needed[int(r.source_db_idx)][r.reaction_id] = {ridx, tsidx, pidx}
    return needed


def _collect_positions(
    db_path: Path,
    targets: dict[str, set[int]],
) -> dict[str, dict[int, np.ndarray]]:
    """Extract positions for the requested (traj, frame) pairs from one DB."""
    out: dict[str, dict[int, np.ndarray]] = {tid: {} for tid in targets}
    remaining = sum(len(v) for v in targets.values())
    if remaining == 0:
        return out
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT energy, natoms, numbers, positions, data FROM systems")
    while remaining > 0:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for energy, natoms, nb, pb, db_ in rows:
            data = decode_data(db_)
            traj, frame_idx = parse_trajectory_id(str(data["dand_id"]))
            wanted = targets.get(traj)
            if not wanted or frame_idx not in wanted:
                continue
            positions = decode_positions(pb, int(natoms))
            out[traj][frame_idx] = positions
            wanted.discard(frame_idx)
            remaining -= 1
            if remaining == 0:
                break
    conn.close()
    return out


def run(
    index_parquet: Path | None = None,
    db_files: list[Path] | None = None,
    output_parquet: Path | None = None,
) -> Path:
    ensure_dirs()
    log = get_logger("phase1.stage3_2")
    log_header(log, "3.2 Bond-change pre-computation")
    if index_parquet is None:
        index_parquet = INDEX_PARQUET
    if output_parquet is None:
        output_parquet = BOND_CHANGES_PARQUET
    if db_files is None:
        db_files = DB_FILES

    index_df = pd.read_parquet(index_parquet)
    log.info("Loaded index: %d trajectories", len(index_df))

    needed = _needed_frames(index_df)
    numbers_lookup = {
        r.reaction_id: np.asarray(r.atomic_numbers, dtype=int)
        for r in index_df.itertuples(index=False)
    }

    records: list[dict] = []
    failures = 0
    t0 = time.time()
    for db_idx, db_path in enumerate(db_files, start=1):
        targets = needed.get(db_idx, {})
        if not targets:
            continue
        log.info(
            "Collecting positions from %s (%d trajectories, %d frames)",
            db_path.name,
            len(targets),
            sum(len(v) for v in targets.values()),
        )
        # Targets is consumed in-place by _collect_positions; copy first.
        targets_copy = {k: set(v) for k, v in targets.items()}
        positions = _collect_positions(db_path, targets_copy)
        for traj_id, frames in positions.items():
            row = index_df.loc[index_df["reaction_id"] == traj_id].iloc[0]
            ridx = int(row.frame_index_first)
            tsidx = int(row.ts_frame_idx)
            pos_R = frames.get(ridx)
            pos_TS = frames.get(tsidx)
            if pos_R is None or pos_TS is None:
                failures += 1
                continue
            numbers = numbers_lookup[traj_id]
            try:
                bc = bond_changes(numbers, pos_R, pos_TS)
            except Exception as e:  # noqa: BLE001
                log.warning("bond_changes failed for %s: %s", traj_id, e)
                failures += 1
                continue
            records.append(
                {
                    "reaction_id": traj_id,
                    "source": row.source,
                    "n_atoms_max": int(row.n_atoms_max),
                    "n_heavy_atoms": int(row.n_heavy_atoms),
                    "activation_energy": float(row.activation_energy),
                    "bonds_R": [list(b) for b in bc["bonds_R"]],
                    "bonds_TS": [list(b) for b in bc["bonds_TS"]],
                    "bonds_broken": bc["bonds_broken"],
                    "bonds_formed": bc["bonds_formed"],
                    "n_bond_changes": int(bc["n_bond_changes"]),
                    "n_components_R": int(bc["n_components_R"]),
                }
            )
        log.info(
            "  cumulative records=%d, failures=%d, elapsed=%.1fs",
            len(records),
            failures,
            time.time() - t0,
        )

    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)
    log.info("Wrote %s with %d rows (failures=%d)", output_parquet, len(df), failures)

    # Spec validation: n_bond_changes >= 1 for every reaction.
    n_zero = int((df["n_bond_changes"] == 0).sum())
    log.info("trajectories with n_bond_changes == 0: %d", n_zero)
    if n_zero:
        log.warning(
            "  these are likely indexing or geometry artifacts; they will be excluded"
            " from sampling pools that require bond_changes >= 1"
        )

    # Histogram of bond changes (Halo8 paper Fig.6 reference).
    hist = df["n_bond_changes"].value_counts().sort_index()
    log.info("bond-change histogram (count by n_bond_changes):\n%s", hist.to_string())
    return output_parquet
