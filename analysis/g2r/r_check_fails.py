#!/usr/bin/env python3
"""D1-G2R 5-5 (docs/specs/D1G2R.md): the 19 excluded_check reactions (TS check / partition failures, D-4) -> which
check failed and its values, from $G2R/jobs/<job_id>/ts_result.json (when present) and the .fail_ts text. No compute. The failed check names come from the .fail_ts text
('TS check failed: a, b' as written by run_ts.analyse; 'partition:<status>' -> partition); under the G2 gate
(extra_imag_tol_cm set, config_g2.yaml) one_imag is informational, so a False one_imag goes to other_false_checks.

    python analysis/g2r/r_check_fails.py          stdlib only; after Phase X (results/X_targets.csv)

Output results/R_check_fails.csv (empty cell = no value). These reactions are not included anywhere.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

N_CHECK = 19
SPEC_CHECKS = ("one_imag", "imag_below_min", "extra_imag_small", "no_foreign_bond", "not_product_like",
               "forming_bond_present", "imag_mode_on_forming_bonds")
PREFIX = "TS check failed: "
COLS = ["job_id", "rxn_id", "g2_marker", "fail_ts_text", "failed_checks", "text_checks_false_in_ts_result",
        "other_false_checks", "partition_status",
        "n_imag", "imag_freqs_cm", "formed_d1_A", "formed_d2_A", "imag_mode_forming_share", "foreign",
        "flag_async", "included"]


def cell(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def main() -> int:
    with open(C.RES / "X_targets.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["group"] == "excluded_check"]
    out = []
    for r in rows:
        jd = C.G2R_JOBS / r["job_id"]
        tr = C.read_json(jd / "ts_result.json") if (jd / "ts_result.json").is_file() else {}
        fail = (jd / ".fail_ts").read_text().strip() if (jd / ".fail_ts").is_file() else ""
        ck = tr.get("checks") or {}
        fd = tr.get("formed_d") or [None, None]
        part = (tr.get("partition") or {}).get("status")
        if fail.startswith(PREFIX):                          # authoritative: run_ts.analyse builds this text
            failed = [k.strip() for k in fail[len(PREFIX):].split(",") if k.strip()]
        else:                                                # G2 gate (extra_imag_tol_cm set): one_imag is not hard
            soft = {"one_imag"} if "extra_imag_small" in ck else set()
            failed = [k for k in SPEC_CHECKS if ck.get(k) is False and k not in soft]
        if (part is not None and part != "ok") or fail.startswith("partition"):
            failed.append("partition")
        out.append(dict(job_id=r["job_id"], rxn_id=r["rxn_id"], g2_marker=r["g2_marker"], fail_ts_text=fail,
                        failed_checks=" ".join(failed),
                        text_checks_false_in_ts_result=all(ck.get(k) is False for k in failed if k != "partition"),
                        other_false_checks=" ".join(k for k, v in ck.items()
                                                    if v is False and k not in failed and not k.startswith("flag_")),
                        partition_status=part, n_imag=tr.get("n_imag"), imag_freqs_cm=tr.get("imag_freqs_cm"),
                        formed_d1_A=fd[0] if len(fd) > 0 else None, formed_d2_A=fd[1] if len(fd) > 1 else None,
                        imag_mode_forming_share=tr.get("imag_mode_forming_share"), foreign=tr.get("foreign"),
                        flag_async=ck.get("flag_async"), included=False))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    w.writeheader()
    w.writerows({k: cell(v) for k, v in o.items()} for o in out)
    C.write_atomic(C.RES / "R_check_fails.csv", buf.getvalue())
    counts = {}
    for o in out:
        for k in (o["failed_checks"].split() or ["(none)"]):
            counts[k] = counts.get(k, 0) + 1
    print(json.dumps(dict(n=len(out), want=N_CHECK, failed_check_counts=counts), indent=1))
    return 0 if len(out) == N_CHECK else 3


if __name__ == "__main__":
    sys.exit(main())
