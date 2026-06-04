"""Extract 5-point ζ bundles in R/pre_TS/TS/post_TS/P scheme.

Run once with system python (3.10+). Writes:
    outputs/stage5b/zeta_bundles/<rxn_id>.npz

Each .npz contains:
    reaction_id
    frame_indices       (5,) int
    ts_pos_in_sorted    int — position of TS in the sorted frames array
    n_frames_total      int
    numbers             (N,) int — atomic numbers
    symbols             (N,) str
    coords_5pts         (5, N, 3) float — Cartesian coordinates
    energies_5pts       (5,) float — Halo8 reference energies (eV)
    forces_5pts         (5, N, 3) float
    arc_total_R_to_P    float
    arc_R_to_TS         float
    halo_db_idx         int (1..10)
    zeta_labels         ["R", "pre_TS", "TS", "post_TS", "P"]
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/src")
from eda_asm.phase1.halo8_io import fetch_frames_multi  # type: ignore


REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
DB_DIR = Path("/home1/yeseo1ee/projects/ts_prediction_project/data")
SELECTED_CSV = REPO / "outputs" / "phase1" / "selected_reactions.csv"
RXN_LIST = REPO / "outputs" / "stage5b" / "rxn_id_list.txt"
OUT_DIR = REPO / "outputs" / "stage5b" / "zeta_bundles"

ZETA_LABELS = np.array(["R", "pre_TS", "TS", "post_TS", "P"])

_NUM2SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
            17: "Cl", 35: "Br", 53: "I"}


def arc_length_cum(positions: np.ndarray) -> np.ndarray:
    """Cumulative Cartesian arc-length along a (T, N, 3) trajectory."""
    if positions.shape[0] < 2:
        return np.zeros(positions.shape[0])
    diff = np.diff(positions, axis=0)              # (T-1, N, 3)
    step = np.sqrt((diff * diff).sum(axis=-1)).mean(axis=-1)  # (T-1,)
    cum = np.zeros(positions.shape[0])
    cum[1:] = np.cumsum(step)
    return cum


def select_5_zeta_indices(arc: np.ndarray, ts_pos: int) -> list:
    n = len(arc)
    arc_TS = arc[ts_pos]
    arc_P = arc[-1]
    pre_pos = int(np.argmin(np.abs(arc[:ts_pos + 1] - arc_TS / 2)))
    post_pos = ts_pos + int(np.argmin(np.abs(arc[ts_pos:] - (arc_TS + arc_P) / 2)))
    return [0, pre_pos, ts_pos, post_pos, n - 1]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load rxn list and source_db_idx
    with open(RXN_LIST) as f:
        rxn_list = [line.strip() for line in f if line.strip()]
    print(f"rxn list: {len(rxn_list)}")

    rxn_db: dict[str, int] = {}
    with open(SELECTED_CSV) as f:
        for row in csv.DictReader(f):
            rid = row["reaction_id"]
            try:
                rxn_db[rid] = int(row["source_db_idx"])
            except (KeyError, ValueError):
                pass

    # Group by DB for one scan per DB
    by_db: dict[int, set] = defaultdict(set)
    for rid in rxn_list:
        db_idx = rxn_db.get(rid)
        if db_idx is None:
            print(f"  [WARN] no source_db_idx for {rid}")
            continue
        by_db[db_idx].add(rid)

    # ts_frame_idx for each reaction (from selected_reactions.csv)
    ts_frame_idx_map: dict[str, int] = {}
    with open(SELECTED_CSV) as f:
        for row in csv.DictReader(f):
            try:
                ts_frame_idx_map[row["reaction_id"]] = int(row["ts_frame_idx"])
            except (KeyError, ValueError):
                pass

    t0_all = time.time()
    n_written = 0
    n_skipped = 0
    n_failed = 0
    for db_idx in sorted(by_db.keys()):
        ids = by_db[db_idx]
        dbp = DB_DIR / f"Halo_{db_idx}.db"
        t0 = time.time()
        print(f"\n=== Halo_{db_idx}.db — {len(ids)} trajectories ===")
        traj_to_rows = fetch_frames_multi(dbp, ids)
        print(f"  scan {time.time()-t0:.1f}s, got {sum(1 for v in traj_to_rows.values() if v)} non-empty")

        for rid in ids:
            rows = traj_to_rows.get(rid)
            if not rows:
                print(f"  [FAIL] {rid}: no rows")
                n_failed += 1
                continue
            rows.sort(key=lambda r: r.frame_idx)

            ts_idx = ts_frame_idx_map.get(rid)
            ts_pos = None
            if ts_idx is not None:
                for i, r in enumerate(rows):
                    if r.frame_idx == ts_idx:
                        ts_pos = i
                        break
            if ts_pos is None:
                energies = np.array([r.energy for r in rows])
                ts_pos = int(np.argmax(energies[1:-1])) + 1

            positions = np.stack([r.positions for r in rows], axis=0)
            arc = arc_length_cum(positions)
            picked = select_5_zeta_indices(arc, ts_pos)

            picked_rows = [rows[i] for i in picked]
            numbers = picked_rows[0].numbers.astype(int)
            symbols = np.array([_NUM2SYM.get(int(n), "?") for n in numbers])

            bundle = {
                "reaction_id": rid,
                "frame_indices": np.array([r.frame_idx for r in picked_rows], dtype=int),
                "ts_pos_in_sorted": int(ts_pos),
                "n_frames_total": int(len(rows)),
                "numbers": numbers,
                "symbols": symbols,
                "coords_5pts": np.stack([r.positions for r in picked_rows], axis=0),
                "energies_5pts": np.array([r.energy for r in picked_rows]),
                "forces_5pts": np.stack(
                    [r.forces if r.forces is not None else np.zeros_like(r.positions)
                     for r in picked_rows], axis=0),
                "arc_total_R_to_P": float(arc[-1]),
                "arc_R_to_TS": float(arc[ts_pos]),
                "halo_db_idx": int(db_idx),
                "zeta_labels": ZETA_LABELS,
            }
            np.savez(OUT_DIR / f"{rid}.npz", **bundle)
            n_written += 1
        print(f"  written so far: {n_written}, failed: {n_failed}, time: {time.time()-t0_all:.0f}s")

    print(f"\n=== DONE  written={n_written} failed={n_failed} skipped={n_skipped} ===")
    print(f"total time: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
