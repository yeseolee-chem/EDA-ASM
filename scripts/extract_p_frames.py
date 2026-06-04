"""One-shot helper: extract the product-side (P) snapshot for each selected
reaction, so the Phase 1.5 review tool can show R / TS / P side-by-side.

P is the trajectory's **last frame** (per Halo8 dataset convention).

Output: outputs/phase1/.tmp_p/<reaction_id>.npz
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.halo8_io import fetch_frames_multi  # noqa: E402
from eda_asm.phase1.paths import DB_FILES, SELECTED_CSV, TMP_DIR  # noqa: E402

OUT_DIR = TMP_DIR.parent / ".tmp_p"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTED_CSV)

    by_db: dict[int, set[str]] = defaultdict(set)
    for r in selected.itertuples(index=False):
        by_db[int(r.source_db_idx)].add(r.reaction_id)

    written = 0
    failed: list[str] = []
    t0 = time.time()
    for db_idx, traj_ids in by_db.items():
        db_path = DB_FILES[db_idx - 1]
        print(f"[{db_path.name}] scanning for {len(traj_ids)} trajectories")
        multi = fetch_frames_multi(db_path, traj_ids)
        for tid, frames in multi.items():
            if not frames:
                failed.append(tid)
                continue
            energies = np.array([f.energy for f in frames])
            ts_pos = int(np.argmax(energies))
            p_frame = frames[-1]
            np.savez(
                OUT_DIR / f"{tid}.npz",
                p_positions=p_frame.positions,
                p_forces=(p_frame.forces if p_frame.forces is not None else np.zeros_like(p_frame.positions)),
                p_energy=p_frame.energy,
                p_dand_id=p_frame.dand_id,
                p_frame_idx=p_frame.frame_idx,
                p_homo=p_frame.data.get("HOMO_level", np.nan),
                p_lumo=p_frame.data.get("LUMO_level", np.nan),
                p_dipole=np.asarray(p_frame.data.get("Dipole_moment", [0.0, 0.0, 0.0])),
                p_mulliken=np.asarray(p_frame.data.get("Mulliken_charges", np.zeros(p_frame.natoms))),
                p_selection_method=np.array("last_frame"),
                p_ts_frame_idx=frames[ts_pos].frame_idx,
                p_n_total_frames=len(frames),
            )
            written += 1
        print(f"  cumulative written={written}  failed={len(failed)}  elapsed={time.time()-t0:.1f}s")

    print(f"Done. Wrote {written} P-bundles to {OUT_DIR}")
    if failed:
        print(f"Failed for {len(failed)} reactions: {failed[:10]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
