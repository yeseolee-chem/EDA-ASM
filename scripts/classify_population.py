"""Run the stage5a classifier on the full Halo8 population (19,176 reactions)
to compare the pattern distribution against the 400-sample distribution.

For each trajectory, we need the R-frame and P-frame positions (TS isn't
needed by the classifier). We stream the 10 source DB files once each,
collecting only those two frames per reaction.

Writes ``outputs/stage5a/population_classification.json`` and
``outputs/stage5a/distribution_compare.json``.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eda_asm.phase1.halo8_io import (
    decode_data,
    decode_numbers,
    decode_positions,
    parse_trajectory_id,
)
from eda_asm.phase1.paths import DB_FILES, INDEX_PARQUET
from eda_asm.stage5a.classify import classify_reaction


def main(out_dir: Path) -> None:
    idx_df = pd.read_parquet(INDEX_PARQUET)
    print(f"population size: {len(idx_df)} reactions")

    # Bucket needed frames per DB
    targets: dict[int, dict[str, set[int]]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for r in idx_df.itertuples(index=False):
        ridx = int(r.frame_index_first)
        pidx = int(r.frame_index_last)
        targets[int(r.source_db_idx)][r.reaction_id] = {ridx, pidx}
        meta[r.reaction_id] = {
            "source": r.source,
            "ridx": ridx,
            "pidx": pidx,
            "n_atoms": int(r.n_atoms_max),
        }

    # Stream each DB once, collecting R and P frame positions per traj
    coords: dict[str, dict[int, np.ndarray]] = {tid: {} for tid in meta}
    numbers_lookup: dict[str, np.ndarray] = {}
    t0 = time.time()
    for db_idx, db_path in enumerate(DB_FILES, start=1):
        wanted = targets.get(db_idx)
        if not wanted:
            continue
        remaining = sum(len(v) for v in wanted.values())
        wanted = {k: set(v) for k, v in wanted.items()}
        print(
            f"[{db_idx}/{len(DB_FILES)}] {db_path.name}: "
            f"{len(wanted)} trajectories, {remaining} frames to grab",
            flush=True,
        )
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
        print(f"    elapsed={time.time()-t0:.1f}s", flush=True)

    # Classify each reaction
    print(f"\nclassifying {len(meta)} reactions...", flush=True)
    classifications: list[dict] = []
    errors: list[dict] = []
    pattern_counter: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    t1 = time.time()
    for i, (rid, m) in enumerate(meta.items()):
        if i and i % 2000 == 0:
            print(f"  {i}/{len(meta)} done  ({time.time()-t1:.1f}s)", flush=True)
        try:
            frames = coords[rid]
            if m["ridx"] not in frames or m["pidx"] not in frames:
                errors.append({"rid": rid, "error": "missing R or P frame"})
                continue
            pat, dbg = classify_reaction(
                numbers_lookup[rid],
                frames[m["ridx"]],
                frames[m["pidx"]],
            )
            n_h = len(dbg.get("migrating_H_atoms", []) or [])
            n_hal = len(dbg.get("migrating_halogen_atoms", []) or [])
            classifications.append({
                "rxn_id": rid,
                "source": m["source"],
                "n_atoms": m["n_atoms"],
                "pattern": pat,
                "n_H_migrating": n_h,
                "n_halogen_migrating": n_hal,
                "n_polyvalent_migrating": len(dbg.get("polyvalent_migrating_atoms", []) or []),
                "n_rearranging": len(dbg.get("rearranging_atoms", []) or []),
                "n_bond_changes": len(dbg.get("bonds_broken", []) or [])
                                  + len(dbg.get("bonds_formed", []) or []),
            })
            pattern_counter[pat] += 1
            by_source[m["source"]][pat] += 1
        except Exception as e:  # noqa: BLE001
            errors.append({"rid": rid, "error": f"{type(e).__name__}: {e}"})

    print(f"  done in {time.time()-t1:.1f}s\n", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "population_classification.json").write_text(json.dumps(classifications, indent=2))
    if errors:
        (out_dir / "population_errors.json").write_text(json.dumps(errors, indent=2))

    # Compare with the 400 sample distribution
    sample = json.loads((out_dir / "fragmentation_summary.json").read_text())
    sample_counter: Counter = Counter(r["pattern"] for r in sample)
    sample_by_source: dict[str, Counter] = defaultdict(Counter)
    for r in sample:
        sample_by_source[r["source"]][r["pattern"]] += 1

    all_patterns = sorted(set(pattern_counter) | set(sample_counter))
    pop_total = sum(pattern_counter.values())
    sam_total = sum(sample_counter.values())
    print("=== Pattern distribution: population vs 400 sample ===")
    print(f"{'pattern':<15s}  {'POP %':>8s}  {'SAMPLE %':>9s}  {'POP n':>7s}  {'SAMPLE n':>9s}  {'Δ pp':>7s}")
    rows = []
    for pat in all_patterns:
        pop_pct = 100 * pattern_counter[pat] / pop_total if pop_total else 0
        sam_pct = 100 * sample_counter[pat] / sam_total if sam_total else 0
        delta = sam_pct - pop_pct
        print(f"{pat:<15s}  {pop_pct:7.2f}%  {sam_pct:8.2f}%  {pattern_counter[pat]:>7d}  {sample_counter[pat]:>9d}  {delta:+7.2f}")
        rows.append({
            "pattern": pat,
            "population_count": int(pattern_counter[pat]),
            "population_pct": round(pop_pct, 3),
            "sample_count": int(sample_counter[pat]),
            "sample_pct": round(sam_pct, 3),
            "delta_pp": round(delta, 3),
        })

    print()
    print("=== By source ===")
    for src in sorted(by_source):
        ps = by_source[src]
        ss = sample_by_source.get(src, Counter())
        pt = sum(ps.values()); st = sum(ss.values())
        print(f"\n[{src}] population total={pt}, sample total={st}")
        for pat in all_patterns:
            if not ps.get(pat) and not ss.get(pat):
                continue
            pp = 100 * ps.get(pat, 0) / pt if pt else 0
            sp = 100 * ss.get(pat, 0) / st if st else 0
            print(f"  {pat:<14s}  pop={pp:6.2f}% ({ps.get(pat,0):>5})  sample={sp:6.2f}% ({ss.get(pat,0):>3})  Δ={sp-pp:+5.2f}pp")

    dist = {
        "population_total": pop_total,
        "sample_total": sam_total,
        "n_errors": len(errors),
        "rows": rows,
        "pattern_counts_by_source_population": {
            src: dict(c) for src, c in by_source.items()
        },
        "pattern_counts_by_source_sample": {
            src: dict(c) for src, c in sample_by_source.items()
        },
    }
    (out_dir / "distribution_compare.json").write_text(json.dumps(dist, indent=2))
    print(f"\nwrote {out_dir/'population_classification.json'} ({len(classifications)} entries)")
    print(f"wrote {out_dir/'distribution_compare.json'}")
    if errors:
        print(f"errors: {len(errors)} (see population_errors.json)")


if __name__ == "__main__":
    main(Path("outputs/stage5a"))
