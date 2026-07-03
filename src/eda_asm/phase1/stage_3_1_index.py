"""Stage 3.1 — Halo8 indexing.

Stream every row of Halo_1.db .. Halo_10.db and aggregate per-trajectory
metadata (source, atomic numbers, energies, frame count, ...). Output a
single Parquet table at data/halo8_index/index.parquet.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .halo8_io import (
    assign_source,
    iter_index_rows,
    parse_trajectory_id,
    _formula_from_numbers,
)
from .logging_setup import get_logger, log_header
from .paths import DB_FILES, INDEX_PARQUET, ensure_dirs

MIN_FRAMES = 20  # CLAUDE.md drop rule for short trajectories


@dataclass(slots=True)
class _TrajAgg:
    db_idx: int
    source: str
    numbers: np.ndarray
    natoms: int
    charge: float
    formula: str
    frames: dict[int, float] = field(default_factory=dict)


def _build_records(aggs: dict[str, _TrajAgg]) -> list[dict]:
    rows: list[dict] = []
    for traj_id, a in aggs.items():
        if not a.frames:
            continue
        frame_indices = sorted(a.frames)
        n_snapshots = len(frame_indices)
        # R = frame 0; if missing pick the smallest index.
        r_idx = 0 if 0 in a.frames else frame_indices[0]
        e_R = a.frames[r_idx]
        ts_idx = max(a.frames, key=lambda k: a.frames[k])
        e_TS = a.frames[ts_idx]
        ts_pos = frame_indices.index(ts_idx)
        last_idx = frame_indices[-1]
        e_P = a.frames[last_idx]
        n_heavy = int(np.sum(a.numbers != 1))
        rows.append(
            {
                "reaction_id": traj_id,
                "source": a.source,
                "n_atoms_max": a.natoms,
                "n_heavy_atoms": n_heavy,
                "atomic_numbers": a.numbers.tolist(),
                "n_snapshots": n_snapshots,
                "frame_index_first": r_idx,
                "frame_index_last": last_idx,
                "ts_frame_idx": ts_idx,
                "ts_position_in_sorted": ts_pos,
                "energy_R": float(e_R),
                "energy_TS": float(e_TS),
                "energy_P": float(e_P),
                "activation_energy": float(e_TS - e_R),
                "ea_relative_to_P": float(e_TS - e_P),
                "total_charge": float(a.charge),
                "formula": a.formula,
                "source_db_idx": a.db_idx,
            }
        )
    return rows


def run(db_files: list[Path] | None = None, output_parquet: Path | None = None) -> Path:
    ensure_dirs()
    log = get_logger("phase1.stage3_1")
    log_header(log, "3.1 Halo8 indexing", min_frames=MIN_FRAMES)
    if db_files is None:
        db_files = DB_FILES
    if output_parquet is None:
        output_parquet = INDEX_PARQUET

    aggs: dict[str, _TrajAgg] = {}
    total_rows = 0
    t0 = time.time()
    for db_idx, db_path in enumerate(db_files, start=1):
        if not db_path.exists():
            log.error("DB missing: %s", db_path)
            continue
        log.info("Streaming %s (%.1f GB)", db_path.name, db_path.stat().st_size / 2**30)
        rows_in_db = 0
        t_db = time.time()
        for rid, did, energy, natoms, charge, numbers, _data in iter_index_rows(db_path):
            traj, frame_idx = parse_trajectory_id(did)
            agg = aggs.get(traj)
            if agg is None:
                src = assign_source(traj, numbers)
                agg = _TrajAgg(
                    db_idx=db_idx,
                    source=src,
                    numbers=numbers,
                    natoms=int(natoms),
                    charge=float(charge),
                    formula=_formula_from_numbers(numbers),
                )
                aggs[traj] = agg
            agg.frames[frame_idx] = float(energy)
            rows_in_db += 1
        total_rows += rows_in_db
        log.info(
            "  %s: %d rows, %d trajectories so far, %.1fs",
            db_path.name,
            rows_in_db,
            len(aggs),
            time.time() - t_db,
        )

    log.info(
        "Streamed %d total rows across %d trajectories in %.1fs",
        total_rows,
        len(aggs),
        time.time() - t0,
    )

    rows = _build_records(aggs)
    df = pd.DataFrame(rows)
    log.info(
        "Built index dataframe: %d rows, source distribution:\n%s",
        len(df),
        df["source"].value_counts().to_string(),
    )

    # Drop trajectories with too few frames (note: nothing in spec says hard
    # drop here, but Phase 2 needs at least an interior TS so we annotate).
    df["short_traj"] = df["n_snapshots"] < MIN_FRAMES
    df["interior_ts"] = (df["ts_frame_idx"] != df["frame_index_first"]) & (
        df["ts_frame_idx"] != df["frame_index_last"]
    )

    n_short = int(df["short_traj"].sum())
    n_no_interior = int((~df["interior_ts"]).sum())
    log.info("short trajectories (<%d frames): %d", MIN_FRAMES, n_short)
    log.info("trajectories without interior TS: %d", n_no_interior)

    df.to_parquet(output_parquet, index=False)
    log.info("Wrote index parquet to %s", output_parquet)
    return output_parquet
