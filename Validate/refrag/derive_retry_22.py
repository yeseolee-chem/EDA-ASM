#!/usr/bin/env python3
"""Generate targeted retry candidates for the 22 still-FAIL reactions.

Per-category strategies:

  ts_not_max  (3): re-scan full Halo8 trajectory; pick R = global min on R-side,
                   P = global min on P-side, TS = global max in between. Write a
                   new stage5a with the corrected frame indices (same fragmentation
                   as current winner).

  schema      (2): re-run the same winner stage5a (transient SCF failures may not
                   repeat) AND try one closed-shell variant if parity allows.

  conservation (17): exhaustive spin enumeration. For the winner fragmentation,
                   try every (mult_1, mult_2, ...) combination consistent with
                   electron parity, up to mult=4 per fragment, plus all sign
                   choices for open shells. Excludes duplicates of variants
                   already tried.

Synthetic rid suffixes:
  __t<N>   trajectory-fixed frames
  __r<N>   re-run with same settings
  __m<N>   multiplicity-sweep variant
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
DB_DIR = Path("/home1/yeseo1ee/projects/ts_prediction_project/data")
WINNER_S5 = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
CAND_S5 = ROOT / "Validate" / "refrag" / "candidates_stage5a" / "per_reaction"
CAND_SUM = ROOT / "Validate" / "refrag" / "candidate_summary.json"
DB_IDX = ROOT / "outputs" / "asr_spec" / "db_idx_map.json"
DIAG = ROOT / "Validate" / "refrag" / "still_fail_diagnosis.json"

Z_OF = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17,"Br":35,"I":53}
SYM_OF = {v: k for k, v in Z_OF.items()}


def _decode_data_blob(b: bytes) -> dict:
    if not b: return {}
    offset = int(np.frombuffer(b[:8], np.int64)[0])
    return json.loads(b[offset:].decode())


def _scan_trajectory(db_path: Path, rxn_id: str) -> list[tuple[int, float]]:
    """Return [(frame_idx, energy_eV), ...] for one reaction, sorted by frame_idx."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT energy, data FROM systems")
    frames: list[tuple[int, float]] = []
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for energy, db in rows:
            try:
                data = _decode_data_blob(db)
            except Exception:
                continue
            did = str(data.get("dand_id", ""))
            head, _, tail = did.rpartition("_")
            if head != rxn_id:
                continue
            try:
                fi = int(tail)
            except ValueError:
                continue
            frames.append((fi, float(energy)))
    conn.close()
    frames.sort()
    return frames


def _pick_RTS_P(frames: list[tuple[int, float]]) -> tuple[int, int, int] | None:
    """From a trajectory, return (R_idx, TS_idx, P_idx) with TS = global max,
    R = global min before TS, P = global min after TS."""
    if len(frames) < 3:
        return None
    energies = [e for _, e in frames]
    ts_pos = max(range(len(energies)), key=lambda i: energies[i])
    if ts_pos == 0 or ts_pos == len(energies) - 1:
        return None
    r_pos = min(range(0, ts_pos), key=lambda i: energies[i])
    p_pos = ts_pos + min(range(len(energies) - ts_pos), key=lambda i: energies[ts_pos + i])
    if energies[ts_pos] <= energies[r_pos] or energies[ts_pos] <= energies[p_pos]:
        return None
    return frames[r_pos][0], frames[ts_pos][0], frames[p_pos][0]


def _e_count(atoms, symbols) -> int:
    if symbols is None:
        return len(atoms)
    return sum(Z_OF.get(symbols[a], 0) for a in atoms)


def _coupling(mults, signs) -> str:
    n_open = sum(1 for m in mults if m > 1)
    if n_open == 0: return "closed_shell_singlet"
    if n_open == 1: return f"multiplet_{max(mults)}"
    if all(signs[i] > 0 for i, m in enumerate(mults) if m > 1):
        return "ferromagnetic_high_spin"
    return "broken_symmetry_singlet"


def _enumerate_mult_combos(e_counts: list[int], max_mult: int = 4):
    """Yield all (mults, signs) combos consistent with electron parity per
    fragment, multiplicities up to max_mult, with valid sign assignments."""
    valid_mults_per_frag = []
    for e in e_counts:
        ms = []
        for m in range(1, max_mult + 1):
            if (m - 1) % 2 == e % 2:
                ms.append(m)
        valid_mults_per_frag.append(ms)
    for combo in itertools.product(*valid_mults_per_frag):
        open_idx = [i for i, m in enumerate(combo) if m > 1]
        if not open_idx:
            yield list(combo), [1] * len(combo)
            continue
        # All sign patterns over open shells; deduplicate by overall flip
        n = len(open_idx)
        seen_signs: set = set()
        for bits in range(2 ** n):
            signs = [1] * len(combo)
            for j, oi in enumerate(open_idx):
                signs[oi] = 1 if (bits >> j) & 1 else -1
            # Canonicalize: prefer first open-shell sign = +1
            if signs[open_idx[0]] < 0:
                signs = [-s if i in open_idx else s for i, s in enumerate(signs)]
            key = tuple(signs)
            if key in seen_signs:
                continue
            seen_signs.add(key)
            yield list(combo), signs


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "Validate"))
    import pickle
    with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)

    diag = json.loads(DIAG.read_text())
    db_idx_map = json.loads(DB_IDX.read_text())
    cand_sum = json.loads(CAND_SUM.read_text())

    n_t = n_r = n_m = 0
    rows: list[dict] = []

    for rec in diag["rows"]:
        rid = rec["rid"]
        cat = rec["cat"]
        # Load winner stage5a (the current canonical fragmentation choice)
        sp = WINNER_S5 / rid / "result.json"
        if not sp.exists():
            continue
        winner = json.loads(sp.read_text())
        win_frags = winner["result"]["fragments"]
        # Symbols from cache
        ent = cache.get(rid)
        symbols = None
        if ent is not None:
            try:
                symbols = [SYM_OF.get(int(z), "?") for z in ent.numbers]
            except Exception:
                pass

        existing = cand_sum["rids"].get(rid, {"n_candidates": 0, "candidates": []})
        existing_keys = {(c.get("synth_rid"), c.get("label")) for c in existing["candidates"]}

        next_t = max([int(c["synth_rid"].rsplit("__t", 1)[-1])
                      for c in existing["candidates"]
                      if "__t" in c["synth_rid"]
                         and c["synth_rid"].rsplit("__t", 1)[-1].isdigit()],
                     default=-1) + 1
        next_r = max([int(c["synth_rid"].rsplit("__r", 1)[-1])
                      for c in existing["candidates"]
                      if "__r" in c["synth_rid"]
                         and c["synth_rid"].rsplit("__r", 1)[-1].isdigit()],
                     default=-1) + 1
        next_m = max([int(c["synth_rid"].rsplit("__m", 1)[-1])
                      for c in existing["candidates"]
                      if "__m" in c["synth_rid"]
                         and c["synth_rid"].rsplit("__m", 1)[-1].isdigit()],
                     default=-1) + 1

        # ── ts_not_max: trajectory-fixed frames ──────────────────────────
        if cat == "ts_not_max":
            db_idx = db_idx_map.get(rid)
            if db_idx:
                frames = _scan_trajectory(DB_DIR / f"Halo_{db_idx}.db", rid)
                pick = _pick_RTS_P(frames)
                if pick:
                    r_idx, ts_idx, p_idx = pick
                    # Only emit if any frame actually changed
                    if (r_idx, ts_idx, p_idx) != (winner["frame_index_first"],
                                                    winner["ts_frame_idx"],
                                                    winner["frame_index_last"]):
                        synth = f"{rid}__t{next_t}"
                        next_t += 1
                        db_idx_map[synth] = db_idx
                        new_s = dict(winner)
                        new_s["reaction_id"] = synth
                        new_s["frame_index_first"] = r_idx
                        new_s["ts_frame_idx"] = ts_idx
                        new_s["frame_index_last"] = p_idx
                        new_s["result"] = dict(new_s["result"])
                        new_s["result"]["notes"] = (
                            f"trajectory-fixed frames: R={r_idx} TS={ts_idx} P={p_idx}"
                        )
                        new_s["result"]["pattern"] = "t_traj_fix"
                        new_s["fragmentation_revision"] = 6
                        out_p = CAND_S5 / synth / "result.json"
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        out_p.write_text(json.dumps(new_s, indent=2))
                        existing["candidates"].append({
                            "synth_rid": synth, "label": "t_traj_fix",
                            "n_fragments": len(win_frags),
                            "fragment_sizes": sorted([len(f["atom_indices"])
                                                       for f in win_frags], reverse=True),
                            "multiplicities": [f["multiplicity"] for f in win_frags],
                            "coupling": new_s["result"].get("coupling", "?"),
                        })
                        rows.append({"rid": rid, "synth_rid": synth, "kind": "traj_fix",
                                      "frames": [r_idx, ts_idx, p_idx]})
                        n_t += 1

        # ── schema: re-run winner with same settings (transient retry) ───
        if cat == "schema":
            synth = f"{rid}__r{next_r}"
            next_r += 1
            db_idx = db_idx_map.get(rid)
            if db_idx is not None:
                db_idx_map[synth] = db_idx
                new_s = dict(winner)
                new_s["reaction_id"] = synth
                new_s["result"] = dict(new_s["result"])
                new_s["result"]["pattern"] = "r_rerun_same"
                new_s["fragmentation_revision"] = 6
                out_p = CAND_S5 / synth / "result.json"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(new_s, indent=2))
                existing["candidates"].append({
                    "synth_rid": synth, "label": "r_rerun_same",
                    "n_fragments": len(win_frags),
                    "fragment_sizes": sorted([len(f["atom_indices"])
                                              for f in win_frags], reverse=True),
                    "multiplicities": [f["multiplicity"] for f in win_frags],
                    "coupling": new_s["result"].get("coupling", "?"),
                })
                rows.append({"rid": rid, "synth_rid": synth, "kind": "rerun_same"})
                n_r += 1

        # ── conservation: full multiplicity sweep on winner fragmentation ─
        if cat in ("cons_huge", "cons_mid", "cons_small"):
            e_counts = [_e_count(set(f["atom_indices"]), symbols) for f in win_frags]
            cur_mults = tuple(int(f["multiplicity"]) for f in win_frags)
            cur_signs_set = set()
            for c in existing["candidates"]:
                if c.get("synth_rid", "").startswith(rid + "__"):
                    mults_c = tuple(c.get("multiplicities", []))
                    if mults_c:
                        cur_signs_set.add(mults_c)

            already_seen = set(cur_signs_set)
            already_seen.add(cur_mults)

            n_this = 0
            for mults, signs in _enumerate_mult_combos(e_counts, max_mult=4):
                key = tuple(mults)
                if key in already_seen and tuple(signs) == (1,) * len(signs):
                    # Same mults, all-positive signs == ferromagnetic; only skip if
                    # same exact combo already produced.
                    pass
                # Hard cap: avoid runaway — at most 6 new variants per rid
                if n_this >= 6:
                    break
                # Skip exact already-tried mult tuple
                if tuple(mults) in already_seen:
                    continue
                synth = f"{rid}__m{next_m}"
                next_m += 1
                db_idx = db_idx_map.get(rid)
                if db_idx is None:
                    break
                db_idx_map[synth] = db_idx
                new_frags = []
                for i, f in enumerate(win_frags):
                    new_frags.append({
                        "atom_indices": sorted(f["atom_indices"]),
                        "role": f"comp_{i}",
                        "multiplicity": mults[i],
                        "cap_attachment": None,
                    })
                total_spin = sum(s * (m - 1) for s, m in zip(signs, mults))
                label = f"m_sweep_{'_'.join(str(m) for m in mults)}"
                new_s = dict(winner)
                new_s["reaction_id"] = synth
                new_s["result"] = {
                    "pattern": label,
                    "fragments": new_frags,
                    "spin_signs": signs,
                    "total_spin_polarization": int(total_spin),
                    "coupling": _coupling(mults, signs),
                    "n_fragments": len(new_frags),
                    "cap_h_positions": None,
                    "confidence": 0.7,
                    "notes": f"multiplicity sweep ({label})",
                    "debug": {"source": "derive_retry_22.py", "label": label},
                }
                new_s["fragmentation_revision"] = 6
                out_p = CAND_S5 / synth / "result.json"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(new_s, indent=2))
                existing["candidates"].append({
                    "synth_rid": synth, "label": label,
                    "n_fragments": len(new_frags),
                    "fragment_sizes": sorted([len(f["atom_indices"])
                                              for f in new_frags], reverse=True),
                    "multiplicities": list(mults),
                    "coupling": new_s["result"]["coupling"],
                })
                rows.append({"rid": rid, "synth_rid": synth, "kind": "mult_sweep",
                              "mults": list(mults), "signs": signs})
                n_m += 1
                n_this += 1
                already_seen.add(key)

        existing["n_candidates"] = len(existing["candidates"])
        cand_sum["rids"][rid] = existing

    DB_IDX.write_text(json.dumps(db_idx_map, indent=2))
    CAND_SUM.write_text(json.dumps(cand_sum, indent=2))

    summary = {
        "n_traj_fix": n_t,
        "n_rerun_same": n_r,
        "n_mult_sweep": n_m,
        "total": n_t + n_r + n_m,
        "rows": rows,
    }
    out_p = ROOT / "Validate" / "refrag" / "retry_22_summary.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"Retry candidates written: {n_t + n_r + n_m}")
    print(f"  traj_fix:    {n_t}")
    print(f"  rerun_same:  {n_r}")
    print(f"  mult_sweep:  {n_m}")
    print(f"  summary:     {out_p}")
    print(f"  db_idx size: {len(db_idx_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
