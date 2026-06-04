"""Group C — relaxed τ_fail (no new ADF), marginal-tag classification."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config


def classify(res_cons: float, cfg: Config) -> tuple[str, bool]:
    """Return (verdict, marginal_tag) per spec §5 table."""
    if res_cons <= cfg.tau_pass:
        return "PASS", False
    if res_cons <= cfg.tau_warn:
        return "WARN", False
    if res_cons <= cfg.tau_fail_relaxed:
        return "WARN", True
    return "FAIL", True


def process_one(entry: dict, manifest_path: Path, out_dir: Path,
                 cfg: Config) -> dict:
    """One reaction — read residual from manifest, apply relaxed thresholds."""
    rid = entry["reaction_id"]
    res_cons = None
    with open(manifest_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("reaction_id") == rid:
                try:
                    res_cons = float(row["max_abs_res_cons_kcal"])
                except (KeyError, ValueError, TypeError):
                    pass
                break
    if res_cons is None:
        out = {
            "reaction_id": rid, "max_abs_res_cons_kcal": None,
            "new_verdict": "FAIL", "marginal_tag": "true",
            "action_taken": "relax_tau",
            "error": "residual missing from manifest",
        }
    else:
        verdict, marginal = classify(res_cons, cfg)
        out = {
            "reaction_id": rid,
            "max_abs_res_cons_kcal": res_cons,
            "new_verdict": verdict,
            "marginal_tag": "true" if marginal else "false",
            "action_taken": "relax_tau",
        }
    out_path = out_dir / "C" / f"{rid}_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> int:
    """CLI entry per fix_fail_19_spec §5."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    cfg = Config()
    queue = json.loads(args.queue.read_text())
    summary = {"PASS": 0, "WARN_clean": 0, "WARN_marginal": 0, "FAIL": 0}
    for entry in queue:
        r = process_one(entry, args.manifest, args.out_dir, cfg)
        v = r["new_verdict"]; m = r.get("marginal_tag") == "true"
        if v == "PASS":
            summary["PASS"] += 1
        elif v == "WARN" and not m:
            summary["WARN_clean"] += 1
        elif v == "WARN" and m:
            summary["WARN_marginal"] += 1
        else:
            summary["FAIL"] += 1
    print(f"group_c: PASS={summary['PASS']}  WARN(clean)={summary['WARN_clean']}  "
          f"WARN(marginal)={summary['WARN_marginal']}  FAIL={summary['FAIL']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
