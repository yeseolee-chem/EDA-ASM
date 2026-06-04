"""Count how many Halo8 trajectories are multi-molecular at tight bond cutoff.

For each of the ~19,176 trajectories:
- R = frame_index_first (typically 0)
- P = frame_index_last
- detect bonds with Cordero × TIGHT_TOL
- count connected components in R and in P
- a trajectory is "multi-molecular" if components_R >= 2 OR components_P >= 2

Streams each Halo_*.db once (≈ 5 minutes total).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.bonds import covalent_radius  # noqa: E402
from eda_asm.phase1.halo8_io import (  # noqa: E402
    decode_data,
    decode_numbers,
    decode_positions,
    parse_trajectory_id,
)
from eda_asm.phase1.paths import DB_FILES, INDEX_PARQUET  # noqa: E402

TIGHT_TOL = 1.10


def detect_tight(numbers: np.ndarray, positions: np.ndarray) -> set[tuple[int, int]]:
    n = len(numbers)
    radii = np.array([covalent_radius(int(z)) for z in numbers])
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    bonds: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            if d < 1e-3:
                continue
            if d < TIGHT_TOL * (radii[i] + radii[j]):
                bonds.add((i, j))
    return bonds


def n_components(numbers: np.ndarray, positions: np.ndarray) -> int:
    bonds = detect_tight(numbers, positions)
    g = nx.Graph()
    g.add_nodes_from(range(len(numbers)))
    g.add_edges_from(bonds)
    return nx.number_connected_components(g)


def main() -> int:
    import sqlite3

    index = pd.read_parquet(INDEX_PARQUET)
    print(f"Indexed trajectories: {len(index)}")

    # Per-trajectory we need (frame_index_first, frame_index_last).
    needed: dict[str, dict[str, int]] = {}
    for r in index.itertuples(index=False):
        needed[r.reaction_id] = {
            "first": int(r.frame_index_first),
            "last": int(r.frame_index_last),
            "source": r.source,
            "db_idx": int(r.source_db_idx),
            "n_atoms": int(r.n_atoms_max),
        }

    # Per-DB targets: {db_idx: {traj_id: {first_idx, last_idx}}}
    by_db: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
    for tid, info in needed.items():
        by_db[info["db_idx"]][tid] = {"first": info["first"], "last": info["last"]}

    # Per-trajectory captured frames: {tid: {"R": positions, "P": positions, "numbers": numbers}}
    captured: dict[str, dict[str, np.ndarray]] = {}

    t0 = time.time()
    for db_idx, db_path in enumerate(DB_FILES, start=1):
        if db_idx not in by_db:
            continue
        targets = by_db[db_idx]
        n_target_frames = sum(2 for _ in targets)
        print(
            f"[{db_path.name}] scanning for {len(targets)} trajectories "
            f"({n_target_frames} frames needed)..."
        )
        t_db = time.time()
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT energy, natoms, numbers, positions, data FROM systems")
        found = 0
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            for energy, natoms, nb, pb, db_ in rows:
                data = decode_data(db_)
                did = str(data["dand_id"])
                traj, frame_idx = parse_trajectory_id(did)
                want = targets.get(traj)
                if want is None:
                    continue
                if frame_idx == want["first"]:
                    if traj not in captured:
                        captured[traj] = {}
                    captured[traj]["R"] = decode_positions(pb, int(natoms))
                    captured[traj]["numbers"] = decode_numbers(nb)
                    found += 1
                elif frame_idx == want["last"]:
                    if traj not in captured:
                        captured[traj] = {}
                    captured[traj]["P"] = decode_positions(pb, int(natoms))
                    captured[traj].setdefault("numbers", decode_numbers(nb))
                    found += 1
        conn.close()
        print(f"  done in {time.time() - t_db:.1f}s, captured frames this db = {found}")

    print(f"\nCaptured frames for {len(captured)} trajectories in {time.time() - t0:.1f}s")

    # Now classify
    results: list[dict] = []
    skipped = 0
    for tid, info in needed.items():
        frames = captured.get(tid)
        if not frames or "R" not in frames or "P" not in frames:
            skipped += 1
            continue
        numbers = frames["numbers"]
        try:
            n_R = n_components(numbers, frames["R"])
            n_P = n_components(numbers, frames["P"])
        except Exception:
            skipped += 1
            continue
        results.append(
            {
                "reaction_id": tid,
                "source": info["source"],
                "n_atoms": info["n_atoms"],
                "n_components_R": n_R,
                "n_components_P": n_P,
                "is_multimolecular": n_R >= 2 or n_P >= 2,
            }
        )

    bimol = [r for r in results if r["is_multimolecular"]]
    print(f"\n=== Cordero × {TIGHT_TOL} cutoff applied to ALL Halo8 ===")
    print(f"Total classified: {len(results)} (skipped: {skipped})")
    print(f"Multi-molecular (R ≥ 2 OR P ≥ 2): {len(bimol)} "
          f"({len(bimol) / len(results) * 100:.1f}%)")

    from collections import Counter
    by_source_all = Counter(r["source"] for r in results)
    by_source_bimol = Counter(r["source"] for r in bimol)
    print()
    print(f"{'source':16} {'total':>8} {'bimol':>8}  {'rate':>8}")
    for src in sorted(set(by_source_all) | set(by_source_bimol)):
        tot = by_source_all.get(src, 0)
        bim = by_source_bimol.get(src, 0)
        rate = bim / tot * 100 if tot else 0
        print(f"{src:16} {tot:>8} {bim:>8}  {rate:>7.1f}%")

    r_only = sum(1 for r in bimol if r["n_components_R"] >= 2 and r["n_components_P"] < 2)
    p_only = sum(1 for r in bimol if r["n_components_P"] >= 2 and r["n_components_R"] < 2)
    both = sum(1 for r in bimol if r["n_components_R"] >= 2 and r["n_components_P"] >= 2)
    print(f"\nR ≥ 2 only (associative): {r_only}")
    print(f"P ≥ 2 only (dissociative): {p_only}")
    print(f"Both R ≥ 2 AND P ≥ 2:     {both}")

    out_path = ROOT / "outputs" / "phase1" / "halo8_multimolecular_all_tight110.parquet"
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"\nWrote full list to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
