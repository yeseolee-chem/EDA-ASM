"""Dry-run report for the ADF retry of recoverable failures.

Reads ADF_250/adf_outputs/parsed/failures.csv (produced by
inventory_labels.py), filters to recoverable failure_reasons, and:

  1. verifies each reaction directory + its 5 input decks are present
  2. classifies the retry strategy per reaction
       full_restart           — all 5 .in files exist, just re-run
       scf_aid_then_restart   — patch SCF block, then re-run
  3. estimates wallclock from previously-successful neighbors
  4. writes two manifest files for downstream consumers:
       outputs/asr_v1/retry/recoverable_dirs.txt   — abs paths, one per line
       outputs/asr_v1/retry/scf_aid_targets.txt    — subset that needs the
                                                     SCF-aid patch first

NO ADF calls. NO writes outside outputs/asr_v1/retry/.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

RECOVERABLE = {
    "scf_not_converged", "failed_step",
    "eda_incomplete", "fragment_energy_missing",
    "no_output_file", "no_status_file", "invalid_status",
}
SCF_AID_REASONS = {"scf_not_converged", "failed_step"}

REQUIRED_INPUTS = ["c1_fragA_ts.in", "c2_fragB_ts.in", "c3_eda.in",
                   "c4_fragA_opt.in", "c5_fragB_opt.in"]


def _wallclock_distribution(adf_root: Path) -> dict:
    """Pull wallclock_s from every successful reaction → stats per family."""
    by_fam: dict[str, list[int]] = {}
    for status_p in adf_root.glob("batch_*/*/status.json"):
        try:
            s = json.loads(status_p.read_text())
        except Exception:
            continue
        if s.get("exit_code") != 0:
            continue
        w = s.get("wallclock_s")
        if not isinstance(w, int) or w <= 0:
            continue
        fam = s["reaction_id"].split("_")[0]
        by_fam.setdefault(fam, []).append(w)
    out = {}
    for fam, vals in by_fam.items():
        out[fam] = {
            "n": len(vals),
            "median_s": int(statistics.median(vals)),
            "mean_s": int(statistics.mean(vals)),
            "max_s": max(vals),
        }
    return out


def _find_rxn_dir(adf_root: Path, rxn_id: str) -> Path | None:
    matches = list(adf_root.glob(f"batch_*/{rxn_id}"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--failures",
                    default="ADF_250/adf_outputs/parsed/failures.csv")
    ap.add_argument("--adf-root", type=Path,
                    default=Path("ADF_250/adf_outputs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/asr_v1/retry"))
    ap.add_argument("--family", default="dipolar",
                    help="only consider this family (default: dipolar). "
                         "Use 'all' to include all families.")
    args = ap.parse_args()

    df = pd.read_csv(args.failures)
    if args.family != "all":
        df = df[df["family"] == args.family]
    df = df[df["failure_reason"].isin(RECOVERABLE)].reset_index(drop=True)

    wall_stats = _wallclock_distribution(args.adf_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_restart_dirs: list[str] = []
    scf_aid_dirs: list[str] = []
    skipped_missing_inputs: list[tuple[str, list[str]]] = []

    for _, r in df.iterrows():
        rid = r["reaction_id"]
        rdir = _find_rxn_dir(args.adf_root, rid)
        if rdir is None:
            rows.append({"reaction_id": rid, "status": "DIR_MISSING",
                         "strategy": "—", "detail": ""})
            continue
        missing = [n for n in REQUIRED_INPUTS if not (rdir / n).is_file()]
        if missing:
            skipped_missing_inputs.append((rid, missing))
            rows.append({"reaction_id": rid, "status": "INPUTS_MISSING",
                         "strategy": "—", "detail": ",".join(missing)})
            continue
        if r["failure_reason"] in SCF_AID_REASONS:
            strategy = "scf_aid_then_restart"
            scf_aid_dirs.append(str(rdir.resolve()))
        else:
            strategy = "full_restart"
        full_restart_dirs.append(str(rdir.resolve()))
        rows.append({"reaction_id": rid, "family": r["family"],
                     "status": "READY", "strategy": strategy,
                     "failure_reason": r["failure_reason"],
                     "detail": str(r.get("detail", ""))})

    # Write the consumer-facing manifests
    (args.out_dir / "recoverable_dirs.txt").write_text(
        "\n".join(full_restart_dirs) + "\n"
    )
    (args.out_dir / "scf_aid_targets.txt").write_text(
        "\n".join(scf_aid_dirs) + "\n"
    )
    report = pd.DataFrame(rows)
    report.to_csv(args.out_dir / "report.csv", index=False)

    fam = args.family
    wall = wall_stats.get(fam, {})
    print(f"\n=== Retry dry-run report  (family={fam}) ===")
    print(f"\nrecoverable candidates: {len(df)}")
    print(f"  → READY for retry        : {len(full_restart_dirs)}")
    print(f"      (of which need SCF aid: {len(scf_aid_dirs)})")
    print(f"  → INPUTS_MISSING (skip)  : {len(skipped_missing_inputs)}")

    if wall:
        med, mean, mx = wall["median_s"], wall["mean_s"], wall["max_s"]
        n_ready = len(full_restart_dirs)
        print(f"\n{fam} wallclock per reaction (from {wall['n']} prior successes):")
        print(f"  median = {med//60} min,  mean = {mean//60} min,  max = {mx//60} min")
        total_serial_h = n_ready * mean / 3600
        print(f"\nProjected total compute (one ADF per reaction):")
        print(f"  serial         : {total_serial_h:5.1f} GPU/CPU-hours wall")
        for par in (4, 8, 12, 16):
            print(f"  {par:2d}-way parallel : {total_serial_h/par:5.1f} h wall")
    else:
        print(f"\n[warn] no wallclock stats for family={fam}")

    print(f"\nbreakdown by failure_reason × strategy:")
    print(report.groupby(["failure_reason", "strategy"], dropna=False)
                .size().to_string())

    print(f"\nWrote:")
    print(f"  {args.out_dir/'recoverable_dirs.txt'}   ({len(full_restart_dirs)} dirs)")
    print(f"  {args.out_dir/'scf_aid_targets.txt'}    ({len(scf_aid_dirs)} dirs)")
    print(f"  {args.out_dir/'report.csv'}              ({len(rows)} rows)")
    print("\nNext steps (run only after license arrives + verified):")
    print(f"  1. python scripts/asr_v1/patch_scf_aids.py --dry-run")
    print(f"  2. python scripts/asr_v1/patch_scf_aids.py             # actually patch")
    print(f"  3. sbatch scripts/asr_v1/submit_adf_retry.sh")


if __name__ == "__main__":
    main()
