#!/usr/bin/env python
"""Write per-reaction status.json after the C1-C5 chain finishes.

Per ASR_ADF_Computation_Spec_v1.0 §8.2. Parses each *.out file to determine
calc_status (converged / not_converged / failed / missing), then writes the
status.json contract that Label Extraction consumes.

Exit code semantics:
  status.json.exit_code = 0  iff every required calc is `converged`
  status.json.exit_code = 1  if any required calc missed / failed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NORMAL_TERMINATION = re.compile(r"\bNORMAL TERMINATION\b", re.IGNORECASE)
NOT_CONVERGED = re.compile(r"(scf.*not.*converged|not converged|did not converge)",
                           re.IGNORECASE)
ERROR_MARKERS = re.compile(r"(error|fatal|sigsegv|invalid)", re.IGNORECASE)


def classify_out(out_path: Path) -> str:
    if not out_path.is_file():
        return "missing"
    try:
        text = out_path.read_text(errors="replace")
    except Exception:
        return "missing"
    if NORMAL_TERMINATION.search(text):
        if NOT_CONVERGED.search(text):
            return "not_converged"
        return "converged"
    if NOT_CONVERGED.search(text):
        return "not_converged"
    if ERROR_MARKERS.search(text):
        return "failed"
    if out_path.stat().st_size < 500:
        return "missing"
    return "failed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rid", required=True)
    p.add_argument("--rxn-dir", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--functional", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--frag-method", required=True)
    p.add_argument("--atoms-a", required=True, help="JSON list of fragA indices")
    p.add_argument("--atoms-b", required=True)
    p.add_argument("--charge-a", required=True, type=int)
    p.add_argument("--charge-b", required=True, type=int)
    p.add_argument("--mult-a", required=True, type=int)
    p.add_argument("--mult-b", required=True, type=int)
    p.add_argument("--total-charge", required=True, type=int)
    p.add_argument("--dataset-delta-Ea", required=True, type=float)
    p.add_argument("--atom-permutation", required=True, help='JSON list or "null"')
    p.add_argument("--single-atom-a", type=int, default=0)
    p.add_argument("--single-atom-b", type=int, default=0)
    args = p.parse_args()

    rxn_dir = Path(args.rxn_dir).resolve()

    calc_status: dict[str, str] = {}
    output_files: dict[str, str] = {}
    for tag, fname in [("c1_fragA_ts", "c1_fragA_ts.out"),
                       ("c2_fragB_ts", "c2_fragB_ts.out"),
                       ("c3_eda", "c3_eda.out"),
                       ("c4_fragA_opt", "c4_fragA_opt.out"),
                       ("c5_fragB_opt", "c5_fragB_opt.out")]:
        if tag == "c4_fragA_opt" and args.single_atom_a:
            calc_status[tag] = "n/a_single_atom"
            continue
        if tag == "c5_fragB_opt" and args.single_atom_b:
            calc_status[tag] = "n/a_single_atom"
            continue
        out = rxn_dir / fname
        calc_status[tag] = classify_out(out)
        output_files[tag] = fname

    # AMS version
    ams_version = "unknown"
    for out_name in ("c1_fragA_ts.out", "c2_fragB_ts.out", "c3_eda.out"):
        p_out = rxn_dir / out_name
        if p_out.is_file():
            try:
                txt = p_out.read_text(errors="replace")
                m = re.search(r"AMS\s+(\d{4}\.\d+)", txt)
                if m:
                    ams_version = m.group(1)
                    break
            except Exception:
                pass

    required_ok = all(
        v in ("converged", "n/a_single_atom") for v in calc_status.values()
    )
    exit_code = 0 if required_ok else 1

    try:
        start_dt = args.start
        end_dt = args.end
        from datetime import datetime
        wall = int((datetime.fromisoformat(end_dt.replace("Z", "+00:00"))
                    - datetime.fromisoformat(start_dt.replace("Z", "+00:00"))).total_seconds())
    except Exception:
        wall = 0

    payload = {
        "reaction_id": args.rid,
        "submitted_utc": args.start,
        "completed_utc": args.end,
        "wallclock_s": wall,
        "exit_code": exit_code,
        "retry_count": 0,
        "ams_version": ams_version,
        "workflow_tool": "manual",
        "functional": args.functional,
        "basis": args.basis,
        "fragment_method": args.frag_method,
        "fragment_atoms_a": json.loads(args.atoms_a),
        "fragment_atoms_b": json.loads(args.atoms_b),
        "fragment_charge_a": args.charge_a,
        "fragment_charge_b": args.charge_b,
        "fragment_mult_a": args.mult_a,
        "fragment_mult_b": args.mult_b,
        "total_charge": args.total_charge,
        "atom_permutation": (None if args.atom_permutation in ("null", "None")
                             else json.loads(args.atom_permutation)),
        "dataset_delta_Ea": args.dataset_delta_Ea,
        "calc_status": calc_status,
        "output_files": output_files,
    }
    (rxn_dir / "status.json").write_text(json.dumps(payload, indent=2))
    print(f"status.json written for {args.rid} exit_code={exit_code} "
          f"calc_status={calc_status}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
