"""Rebuild xtb_extra cache with fallback source-dir resolver.

Differences vs v1 (`build_xtb_extra_cache.py`):
  - Always uses `compute_xtb_extra_from_status` (index-based, no need for
    geometry_fragA.xyz / geometry_fragB.xyz files).
  - For each rid, walks a list of candidate roots looking for any directory
    that contains both `status.json` and `ts.xyz`. Original ADF source_dirs
    (`ADF_250/adf_outputs/.../<rid>/`) are tried first; if missing (archive
    paths deleted), falls back to `runs/orca_eda_label/staging/<rid>/`,
    `runs/orca_recompute/inputs/<rid>/`, `rerun/.../<rid>/`, etc.

Writes:
  analysis/exp_6arm_redesign_v2/xtb_cache/xtb_extra_v2.parquet

Rationale for full rebuild: the index-based slicing produces inter-fragment
WBO values that are correct even when fragA/fragB atoms aren't contiguous
in ts.xyz (the v1 code assumed contiguity). Re-computing the 638 already-OK
reactions ensures uniform semantics across the 789-row cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from src.xtb_features import compute_xtb_extra_from_status, compute_xtb_extra_from_orca_inp  # noqa: E402


# Candidate roots (relative to REPO). Order = priority: try first match wins.
CANDIDATE_ROOTS = [
    # Original ADF outputs (the "canonical" source_dir from orca_eda_labels.parquet)
    "ADF_250/adf_outputs",
    "ADF_dipolar_expand",
    # ORCA recompute inputs (mirror of ADF source for ORCA SP)
    "runs/orca_recompute/inputs",
    # ORCA EDA label staging (only has status.json + ts.xyz, no frag xyzs — but that's fine)
    "runs/orca_eda_label/staging",
    "runs/orca_eda_label/inputs",
    # AL re-runs / retries
    "rerun",
    "retry",
    "work_fix_fail_19",
    "fix_fail_19",
]


def _index_rid_dirs(repo: Path) -> dict[str, list[Path]]:
    """Walk each candidate root once; map rid → list of directories containing status.json."""
    rid_to_dirs: dict[str, list[Path]] = {}
    for sr in CANDIDATE_ROOTS:
        base = repo / sr
        if not base.exists():
            continue
        # rglob status.json (fast: only matches dirs that have it)
        for p in base.rglob("status.json"):
            rid = p.parent.name
            rid_to_dirs.setdefault(rid, []).append(p.parent)
    return rid_to_dirs


def _resolve_source_dir(rid: str, primary_source_dir: str, rid_index: dict[str, list[Path]]) -> Path | None:
    """Return a directory that has BOTH status.json and ts.xyz, preferring the
    primary_source_dir from orca_eda_labels.parquet when it exists."""
    repo = REPO

    def _ok(d: Path) -> bool:
        return (d / "status.json").exists() and (d / "ts.xyz").exists()

    # 1) primary path
    pri = (repo / primary_source_dir) if not Path(primary_source_dir).is_absolute() else Path(primary_source_dir)
    if _ok(pri):
        return pri

    # 2) candidate roots
    for d in rid_index.get(rid, []):
        if _ok(d):
            return d
    return None


def _resolve_orca_inp(rid: str) -> Path | None:
    """Fallback to ORCA EDA input file containing ts geometry + frag indices.

    Used when status.json + ts.xyz can't be found anywhere, but the ORCA
    recompute pipeline already wrote an eda.inp.
    """
    p = REPO / "runs/orca_recompute/inputs" / rid / "eda.inp"
    return p if p.exists() else None


def _worker(args: tuple) -> dict:
    mode, rid, *rest = args
    if mode == "status":
        status_path, ts_path = rest
        res = compute_xtb_extra_from_status(rid=rid, ts_xyz=ts_path, status_json=status_path)
        used = str(Path(status_path).parent)
    elif mode == "orca_inp":
        (eda_inp,) = rest
        res = compute_xtb_extra_from_orca_inp(rid=rid, eda_inp=eda_inp)
        used = str(Path(eda_inp).parent)
    else:
        return {"rid": rid, "ok": False, "fail_reason": f"unknown_mode:{mode}", "source_dir_used": ""}
    dd = res.to_dict()
    dd["source_dir_used"] = used
    dd["mode"] = mode
    return dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orca-labels", type=Path,
                    default=REPO / "runs/orca_recompute/results/orca_eda_labels.parquet")
    ap.add_argument("--out", type=Path,
                    default=HERE / "xtb_cache/xtb_extra_v2.parquet")
    ap.add_argument("--workers", type=int,
                    default=min(16, max(1, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    orca = pd.read_parquet(args.orca_labels)
    if args.limit:
        orca = orca.head(args.limit).copy()

    # idempotency
    have = set()
    if args.out.exists() and not args.rebuild:
        prev = pd.read_parquet(args.out)
        have = set(prev["rid"].to_list())
        print(f"[xtb-extra-v2] cache exists with {len(have)} entries; skipping those.")
    else:
        prev = pd.DataFrame()

    todo = orca[~orca["reaction_id"].isin(have)].copy()
    print(f"[xtb-extra-v2] {len(todo)} rids to compute; building rid → dirs index...", flush=True)

    rid_index = _index_rid_dirs(REPO)
    print(f"[xtb-extra-v2] indexed {len(rid_index)} unique rids across {len(CANDIDATE_ROOTS)} roots.", flush=True)

    # Resolve source dirs sequentially first (small, fast — just stat checks).
    # Two-stage resolution: status.json+ts.xyz first; if missing, fall back to
    # ORCA EDA input file (runs/orca_recompute/inputs/<rid>/eda.inp).
    payload = []
    unresolved = []
    via_status = 0
    via_orca_inp = 0
    for _, r in todo.iterrows():
        rid = r["reaction_id"]
        sd = _resolve_source_dir(rid, r["source_dir"], rid_index)
        if sd is not None:
            payload.append(("status", rid, str(sd / "status.json"), str(sd / "ts.xyz")))
            via_status += 1
            continue
        # fallback: ORCA EDA input file
        orca_inp = _resolve_orca_inp(rid)
        if orca_inp is not None:
            payload.append(("orca_inp", rid, str(orca_inp)))
            via_orca_inp += 1
            continue
        unresolved.append(rid)
    print(f"[xtb-extra-v2] resolved via status.json: {via_status}")
    print(f"[xtb-extra-v2] resolved via ORCA eda.inp: {via_orca_inp}")
    print(f"[xtb-extra-v2] unresolved: {len(unresolved)}")
    if unresolved[:5]:
        print(f"[xtb-extra-v2] unresolved sample: {unresolved[:5]}")

    results: list[dict] = []
    # Record unresolved as failures (so the parquet has rows for them too)
    for rid in unresolved:
        results.append({
            "rid": rid, "ok": False, "fail_reason": "unresolved_source_dir",
            "source_dir_used": "", "mode": "none",
        })

    if payload:
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, p): p[1] for p in payload}
            done = 0
            for fut in as_completed(futs):
                rid = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"rid": rid, "ok": False,
                                    "fail_reason": f"worker_exc:{e}",
                                    "source_dir_used": ""})
                done += 1
                if done % 25 == 0 or done == len(futs):
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (len(futs) - done)
                    print(f"[xtb-extra-v2] {done}/{len(futs)}  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)

    new_df = pd.DataFrame(results)
    out_df = pd.concat([prev, new_df], ignore_index=True) if len(prev) else new_df
    out_df.to_parquet(args.out, index=False)
    n_ok = int(out_df["ok"].sum())
    print(f"[xtb-extra-v2] wrote {args.out}  ok={n_ok}/{len(out_df)}")
    fail_reasons = out_df.loc[~out_df["ok"], "fail_reason"].value_counts().head(10).to_dict()
    print(f"[xtb-extra-v2] top fail reasons: {fail_reasons}")


if __name__ == "__main__":
    main()
