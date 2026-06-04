"""Stage 3.4 — Extract R, TS, and IRC 25/50/75% snapshots for selected reactions.

For each selected trajectory:
1. Load all frames, sort by frame index.
2. R = frame 0, TS = max-energy frame.
3. Walk R..TS, compute cumulative pairwise RMSD as arc length.
4. Pick the snapshots closest to 25%, 50%, 75% of total arc length (use the
   real snapshot, no linear interpolation per spec).
5. Persist 5-point bundle to outputs/phase1/.tmp/<reaction_id>.npz.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .halo8_io import FrameRow, fetch_frames_multi
from .logging_setup import get_logger, log_header
from .paths import DB_FILES, SELECTED_CSV, TMP_DIR, ensure_dirs

ZETA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _frame_to_frame_rmsd(positions: np.ndarray) -> np.ndarray:
    """Cumulative arc length using simple frame-to-frame RMSD.

    positions: (T, N, 3). Returns (T,) cumulative array starting from 0.
    Within a single trajectory the atom ordering is constant, so a direct
    RMSD (no alignment) is the right notion of arc length.
    """
    diff = np.diff(positions, axis=0)
    rmsd = np.sqrt((diff * diff).sum(-1).mean(-1))  # (T-1,)
    cum = np.zeros(positions.shape[0])
    cum[1:] = np.cumsum(rmsd)
    return cum


def _select_5_indices(cum_arc: np.ndarray, ts_pos: int) -> list[int]:
    """Return indices in the original frame array for ζ = 0, 0.25, 0.5, 0.75, 1.0.

    cum_arc: cumulative arc length along R..ts_pos (length ts_pos+1).
    ts_pos:  index (in the sorted frames) of the TS frame.
    Returns a list of original frame-array indices (length 5, monotonic).
    """
    total = cum_arc[ts_pos]
    if total <= 0:
        # Degenerate: all zero arc — fallback to evenly spaced indices.
        idxs = np.linspace(0, ts_pos, 5).round().astype(int).tolist()
        return idxs
    targets = ZETA_GRID * total
    idxs: list[int] = []
    for t in targets:
        diff = np.abs(cum_arc[: ts_pos + 1] - t)
        idxs.append(int(np.argmin(diff)))
    # Ensure strictly monotone non-decreasing (it should already be).
    for i in range(1, 5):
        if idxs[i] < idxs[i - 1]:
            idxs[i] = idxs[i - 1]
    return idxs


def _stack_field(rows: list[FrameRow], picked: list[int], key: str, default_shape) -> np.ndarray | None:
    arrs = []
    for i in picked:
        v = rows[i].data.get(key)
        if v is None:
            return None
        arrs.append(np.asarray(v))
    return np.stack(arrs, axis=0)


def _process_one(traj_id: str, rows: list[FrameRow], log) -> dict | None:
    if len(rows) < 5:
        log.warning("skip %s: only %d frames", traj_id, len(rows))
        return None
    # find TS (max energy)
    energies = np.array([r.energy for r in rows])
    ts_pos = int(np.argmax(energies))
    if ts_pos == 0 or ts_pos == len(rows) - 1:
        log.warning("skip %s: TS at boundary (pos=%d / %d)", traj_id, ts_pos, len(rows))
        return None

    # R is the first frame
    if rows[0].frame_idx != 0:
        # Halo8 frames usually start at 0 but if not we accept the smallest.
        log.warning("traj %s starts at frame %d, treating as R", traj_id, rows[0].frame_idx)

    positions = np.stack([r.positions for r in rows], axis=0)
    cum = _frame_to_frame_rmsd(positions)
    picked_idx = _select_5_indices(cum, ts_pos)

    # Sanity: monotone energy R→TS (ignore tiny wiggles)
    e_RtoTS = energies[: ts_pos + 1]
    if e_RtoTS[-1] <= e_RtoTS[0]:
        log.warning("traj %s: TS energy not above R energy (R=%.4f TS=%.4f)", traj_id, e_RtoTS[0], e_RtoTS[-1])

    # Heuristic dip detection: any interior frame more than 0.1 eV below R?
    n_dip = int(np.sum(e_RtoTS < e_RtoTS[0] - 0.1))
    if n_dip:
        log.info("traj %s: %d interior frames dip > 0.1 eV below R", traj_id, n_dip)

    picked = [rows[i] for i in picked_idx]

    # Verify atom order/elements consistent.
    base_numbers = picked[0].numbers
    for r in picked[1:]:
        if not np.array_equal(r.numbers, base_numbers):
            log.error("atom order changed within trajectory %s — aborting", traj_id)
            return None

    coords = np.stack([r.positions for r in picked], axis=0)
    forces = np.stack(
        [(r.forces if r.forces is not None else np.zeros_like(r.positions)) for r in picked],
        axis=0,
    )
    energies_5 = np.array([r.energy for r in picked])
    homo = np.array([r.data.get("HOMO_level", np.nan) for r in picked])
    lumo = np.array([r.data.get("LUMO_level", np.nan) for r in picked])
    homo_idx = np.array([r.data.get("HOMO_idx", -1) for r in picked])
    lumo_idx = np.array([r.data.get("LUMO_idx", -1) for r in picked])
    natoms = base_numbers.shape[0]
    mulliken = np.stack(
        [np.asarray(r.data.get("Mulliken_charges", np.zeros(natoms))) for r in picked]
    )
    lowdin = np.stack(
        [np.asarray(r.data.get("Lowdin_charges", np.zeros(natoms))) for r in picked]
    )
    dipole = np.stack(
        [np.asarray(r.data.get("Dipole_moment", np.zeros(3))) for r in picked]
    )
    dispersion = np.array([r.data.get("Dispersion_correction", np.nan) for r in picked])

    return {
        "reaction_id": traj_id,
        "frame_indices": np.array([r.frame_idx for r in picked], dtype=int),
        "ts_pos_in_sorted": ts_pos,
        "n_frames_total": len(rows),
        "numbers": base_numbers.astype(int),
        "coords_5pts": coords,
        "energies_5pts": energies_5,
        "forces_5pts": forces,
        "homo_5pts": homo,
        "lumo_5pts": lumo,
        "homo_idx_5pts": homo_idx,
        "lumo_idx_5pts": lumo_idx,
        "mulliken_5pts": mulliken,
        "lowdin_5pts": lowdin,
        "dipole_5pts": dipole,
        "dispersion_5pts": dispersion,
        "dand_ids": np.array([r.dand_id for r in picked], dtype=object),
        "charge": picked[0].charge,
        "zeta_values": ZETA_GRID,
        "arc_length_total": float(cum[ts_pos]),
    }


def run(
    selected_csv: Path | None = None,
    db_files: list[Path] | None = None,
    tmp_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    ensure_dirs()
    log = get_logger("phase1.stage3_4")
    log_header(log, "3.4 5-point extraction")
    if selected_csv is None:
        selected_csv = SELECTED_CSV
    if tmp_dir is None:
        tmp_dir = TMP_DIR
    if db_files is None:
        db_files = DB_FILES

    selected = pd.read_csv(selected_csv)
    log.info("Selected: %d reactions", len(selected))

    # group reaction_ids by source DB so we stream each DB exactly once.
    by_db: dict[int, set[str]] = defaultdict(set)
    for r in selected.itertuples(index=False):
        by_db[int(r.source_db_idx)].add(r.reaction_id)

    written: list[str] = []
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for db_idx, traj_ids in by_db.items():
        db_path = db_files[db_idx - 1]
        log.info(
            "Loading frames for %d trajectories from %s",
            len(traj_ids),
            db_path.name,
        )
        # fetch_frames_multi is a single full-DB scan; do not call per-traj.
        traj_to_rows = fetch_frames_multi(db_path, traj_ids)
        for traj_id in traj_ids:
            rows = traj_to_rows.get(traj_id)
            if not rows:
                failures.append((traj_id, "no rows returned"))
                continue
            try:
                bundle = _process_one(traj_id, rows, log)
            except Exception as e:  # noqa: BLE001
                failures.append((traj_id, f"{type(e).__name__}: {e}"))
                continue
            if bundle is None:
                failures.append((traj_id, "process_one returned None"))
                continue
            out = tmp_dir / f"{traj_id}.npz"
            np.savez(out, **bundle)
            written.append(traj_id)
        log.info(
            "  cumulative written=%d failures=%d elapsed=%.1fs",
            len(written),
            len(failures),
            time.time() - t0,
        )

    log.info("Stage 3.4 complete: written=%d failures=%d", len(written), len(failures))
    if failures:
        for tid, reason in failures[:20]:
            log.warning("  failed %s: %s", tid, reason)
    return tmp_dir, written
