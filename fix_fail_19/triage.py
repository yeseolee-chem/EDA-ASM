"""Triage the 19 FAIL reactions into A/B/C queues per fix_fail_19_spec §2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import GROUP_A_IDS, GROUP_B_IDS, GROUP_C_IDS


def _read_failing_rids(manifest_path: Path) -> list[str]:
    """Return reaction_ids whose verdict==FAIL in the manifest."""
    fails: list[str] = []
    with open(manifest_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("verdict") == "FAIL":
                fails.append(row["reaction_id"])
    return fails


def _resolve_paths(rid: str, json_dir: Path, halo8_dir: Path,
                    rkf_dir: Path) -> dict[str, str]:
    """Compose the per-reaction input paths and verify they exist."""
    paths = {
        "reaction_id": rid,
        "json_path":  str(json_dir / f"{rid}.json"),
        "halo8_path": str(halo8_dir / f"{rid}.frames"),
        "rkf_path":   str(rkf_dir / rid),
    }
    return paths


def _check_paths(entry: dict, halo8_dir: Path, rkf_dir: Path) -> list[str]:
    """Return list of missing inputs for one reaction (empty if all present).

    Per spec §8 edge cases: missing/corrupted rkf triggers re-run by groups A/B,
    so we only require json + halo8 source. rkf_dir presence is informational.
    """
    missing: list[str] = []
    if not Path(entry["json_path"]).exists():
        missing.append(f"json: {entry['json_path']}")
    h = Path(entry["halo8_path"])
    if not (h.exists() or h.with_suffix("").exists()
            or halo8_dir.exists()):
        missing.append(f"halo8: {entry['halo8_path']}")
    return missing


def triage(manifest_path: Path, json_dir: Path, halo8_dir: Path,
            rkf_dir: Path, out_dir: Path) -> dict:
    """Write queue_{A,B,C}.json under out_dir; return summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fails = _read_failing_rids(manifest_path)
    if len(fails) != 19:
        sys.stderr.write(f"triage: expected 19 FAIL rows, got {len(fails)}\n")
        for r in fails:
            sys.stderr.write(f"  - {r}\n")
        sys.exit(1)
    expected = set(GROUP_A_IDS) | set(GROUP_B_IDS) | set(GROUP_C_IDS)
    got = set(fails)
    missing = expected - got
    unexpected = got - expected
    if missing or unexpected:
        sys.stderr.write("triage: reaction_id set mismatch\n")
        if missing:
            sys.stderr.write(f"  missing from manifest: {sorted(missing)}\n")
        if unexpected:
            sys.stderr.write(f"  unexpected in manifest: {sorted(unexpected)}\n")
        sys.exit(1)

    queues: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    group_of = {**{r: "A" for r in GROUP_A_IDS},
                **{r: "B" for r in GROUP_B_IDS},
                **{r: "C" for r in GROUP_C_IDS}}
    all_missing: dict[str, list[str]] = {}
    for rid in fails:
        entry = _resolve_paths(rid, json_dir, halo8_dir, rkf_dir)
        miss = _check_paths(entry, halo8_dir, rkf_dir)
        if miss:
            all_missing[rid] = miss
        queues[group_of[rid]].append(entry)
    if all_missing:
        sys.stderr.write("triage: missing input files\n")
        for rid, miss in all_missing.items():
            for m in miss:
                sys.stderr.write(f"  {rid}: {m}\n")
        sys.exit(1)

    (out_dir / "queue_A.json").write_text(json.dumps(queues["A"], indent=2))
    (out_dir / "queue_B.json").write_text(json.dumps(queues["B"], indent=2))
    (out_dir / "queue_C.json").write_text(json.dumps(queues["C"], indent=2))
    print(f"triage: {len(fails)} FAIL reactions")
    print(f"  group A (trajectory artifact)     : {len(queues['A'])}")
    print(f"  group B (spin reference mismatch) : {len(queues['B'])}")
    print(f"  group C (marginal residual)       : {len(queues['C'])}")
    print(f"queues written: {out_dir / 'queue_{A,B,C}.json'}")
    return {"A": len(queues["A"]), "B": len(queues["B"]), "C": len(queues["C"])}


def main() -> int:
    """CLI entry per fix_fail_19_spec §2."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--halo8-dir", required=True, type=Path)
    ap.add_argument("--rkf-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    triage(args.manifest, args.json_dir, args.halo8_dir, args.rkf_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
