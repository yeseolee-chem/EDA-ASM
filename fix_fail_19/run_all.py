"""Run triage → group A → group B → group C → manifest_v2 (per spec §6)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config, GROUP_A_IDS, GROUP_B_IDS, GROUP_C_IDS
from .group_a_reendpoint import process_one as proc_a
from .group_b_spinsweep import process_one as proc_b
from .group_c_relax import process_one as proc_c
from .triage import triage


MANIFEST_EXTRA_COLS = [
    "group", "action_taken", "new_verdict", "new_max_abs_res_cons_kcal",
    "endpoint_drift_kcal_R", "endpoint_drift_kcal_P",
    "s2_supermol_TS", "winning_coupling", "marginal_tag", "exclude_reason",
]


def _validate_revalidate(rid: str, asr_v2_dir: Path) -> str:
    """Run the project validator on the re-endpointed JSON; return new verdict.

    Falls back to FAIL on any error or if the validator isn't reachable.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Validate"))
        from validate_asr import (derive, check1_schema, check3_topology,
                                    check4_conservation, check5_signs,
                                    aggregate, Config as VCfg)  # type: ignore
        raw = json.loads((asr_v2_dir / f"{rid}.json").read_text())
        d = derive(rid, raw)
        cfg = VCfg()
        issues = (check1_schema(d) + check3_topology(d, cfg)
                  + check4_conservation(d, cfg) + check5_signs(d))
        return aggregate(issues)
    except Exception:
        return "FAIL"


def run_all(manifest: Path, json_dir: Path, halo8_dir: Path, rkf_dir: Path,
             out_dir: Path) -> None:
    """End-to-end pipeline per spec §6."""
    cfg = Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) triage
    triage(manifest, json_dir, halo8_dir, rkf_dir, out_dir)
    qA = json.loads((out_dir / "queue_A.json").read_text())
    qB = json.loads((out_dir / "queue_B.json").read_text())
    qC = json.loads((out_dir / "queue_C.json").read_text())

    # 2) group A
    results_A: dict[str, dict] = {}
    for entry in qA:
        results_A[entry["reaction_id"]] = proc_a(entry, halo8_dir, out_dir, cfg)
    # 3) re-validate pending A
    for rid, r in results_A.items():
        if r["new_verdict"] == "PENDING_REVALIDATE":
            v = _validate_revalidate(rid, out_dir / "A" / "asr_v2")
            r["new_verdict"] = "PASS" if v != "FAIL" else "EXCLUDED"
            if r["new_verdict"] == "EXCLUDED":
                r["exclude_reason"] = "re-validator returned FAIL"

    # 4) group B
    results_B: dict[str, dict] = {}
    for entry in qB:
        results_B[entry["reaction_id"]] = proc_b(entry, halo8_dir, out_dir, cfg)

    # 5) demoted_to_C → append to queue_C for relax
    qC_combined = list(qC)
    for rid, r in results_B.items():
        if r["new_verdict"] == "demoted_to_C":
            qC_combined.append({"reaction_id": rid, "json_path": "",
                                  "halo8_path": "", "rkf_path": ""})

    # 6) group C
    results_C: dict[str, dict] = {}
    for entry in qC_combined:
        results_C[entry["reaction_id"]] = proc_c(entry, manifest, out_dir, cfg)

    # 7) merge into manifest_v2
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    fieldnames = list(rows[0].keys()) + [c for c in MANIFEST_EXTRA_COLS
                                           if c not in rows[0]]
    fail_to_group = ({rid: "A" for rid in GROUP_A_IDS}
                     | {rid: "B" for rid in GROUP_B_IDS}
                     | {rid: "C" for rid in GROUP_C_IDS})

    for row in rows:
        rid = row["reaction_id"]
        if rid not in fail_to_group:
            for c in MANIFEST_EXTRA_COLS:
                row.setdefault(c, "")
            continue
        g = fail_to_group[rid]
        row["group"] = g
        rA = results_A.get(rid, {})
        rB = results_B.get(rid, {})
        rC = results_C.get(rid, {})
        # Group precedence: A overrides if applicable, then B, then C
        action = rA.get("action_taken") or rB.get("action_taken") or rC.get("action_taken") or ""
        verdict = (rA.get("new_verdict") if g == "A"
                   else rB.get("new_verdict") if g == "B"
                   else rC.get("new_verdict"))
        # If group B was demoted_to_C, the actual final verdict comes from rC
        if g == "B" and rB.get("new_verdict") == "demoted_to_C":
            verdict = rC.get("new_verdict") or "FAIL"
        row["action_taken"] = action
        row["new_verdict"] = verdict or ""
        row["new_max_abs_res_cons_kcal"] = (
            rB.get("new_max_abs_res_cons_kcal") or rC.get("max_abs_res_cons_kcal") or "")
        row["endpoint_drift_kcal_R"] = rA.get("drift_R") if rA else ""
        row["endpoint_drift_kcal_P"] = rA.get("drift_P") if rA else ""
        row["s2_supermol_TS"] = rB.get("s2_supermol_TS") if rB else ""
        row["winning_coupling"] = rB.get("winning_coupling") if rB else ""
        row["marginal_tag"] = rC.get("marginal_tag", "false")
        row["exclude_reason"] = rA.get("exclude_reason", "") if rA else ""

    manifest_v2 = out_dir / "manifest_v2.csv"
    with open(manifest_v2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    # Summary stats
    A_pass = sum(1 for r in results_A.values() if r["new_verdict"] == "PASS")
    A_excl = sum(1 for r in results_A.values() if r["new_verdict"] == "EXCLUDED")
    B_pass = sum(1 for r in results_B.values() if r["new_verdict"] == "PASS")
    B_warn = sum(1 for r in results_B.values() if r["new_verdict"] == "WARN")
    B_demo = sum(1 for r in results_B.values() if r["new_verdict"] == "demoted_to_C")
    C_pass = sum(1 for r in results_C.values() if r["new_verdict"] == "PASS")
    C_warn_m = sum(1 for r in results_C.values()
                   if r["new_verdict"] == "WARN" and r.get("marginal_tag") == "true")
    C_warn_c = sum(1 for r in results_C.values()
                   if r["new_verdict"] == "WARN" and r.get("marginal_tag") == "false")
    C_fail = sum(1 for r in results_C.values() if r["new_verdict"] == "FAIL")
    recovered = (A_pass + B_pass + B_warn + C_pass + C_warn_c + C_warn_m)
    print(f"group A: PASS / EXCLUDED  = {A_pass} / {A_excl}")
    print(f"group B: PASS / WARN / demoted_to_C = {B_pass} / {B_warn} / {B_demo}")
    print(f"group C: PASS / WARN(marginal) / FAIL = {C_pass} / {C_warn_m + C_warn_c} / {C_fail}")
    print(f"total FAIL → recovered = {recovered}")
    print(f"manifest_v2 written: {manifest_v2}")


def main() -> int:
    """CLI entry per fix_fail_19_spec §6."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--halo8-dir", required=True, type=Path)
    ap.add_argument("--rkf-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    run_all(args.manifest, args.json_dir, args.halo8_dir, args.rkf_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
