"""Stage 3.5 — Case A/B/C classification.

- Case A: n_components_R >= 2  (bimolecular, R already split)
- Case B: n_components_R == 1 AND n_bond_changes in [2, 3]  (simple unimolecular)
- Case C: n_components_R == 1 AND n_bond_changes >= 4         (concerted multi-bond)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .logging_setup import get_logger, log_header
from .paths import CASE_JSON, SELECTED_CSV, BOND_CHANGES_PARQUET, ensure_dirs


def classify_one(n_components_R: int, n_bond_changes: int) -> tuple[str, str]:
    if n_components_R >= 2:
        return "A", f"n_components_R={n_components_R}"
    if n_bond_changes in (2, 3):
        return "B", f"unimolecular, {n_bond_changes} bond changes"
    if n_bond_changes >= 4:
        return "C", f"unimolecular, {n_bond_changes} bond changes (concerted)"
    return "C", f"unimolecular but only {n_bond_changes} bond change(s); flagged"


def run(
    selected_csv: Path | None = None,
    bond_changes_parquet: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    ensure_dirs()
    log = get_logger("phase1.stage3_5")
    log_header(log, "3.5 Case A/B/C classification")
    if selected_csv is None:
        selected_csv = SELECTED_CSV
    if bond_changes_parquet is None:
        bond_changes_parquet = BOND_CHANGES_PARQUET
    if output_json is None:
        output_json = CASE_JSON

    selected = pd.read_csv(selected_csv)
    bc = pd.read_parquet(bond_changes_parquet).set_index("reaction_id")

    out: dict[str, dict] = {}
    counts = {"A": 0, "B": 0, "C": 0}
    for rid in selected["reaction_id"]:
        if rid not in bc.index:
            log.warning("missing bond-change entry for %s — defaulting to C", rid)
            out[rid] = {"case": "C", "rationale": "missing bond-change record"}
            counts["C"] += 1
            continue
        row = bc.loc[rid]
        case, rationale = classify_one(int(row["n_components_R"]), int(row["n_bond_changes"]))
        out[rid] = {
            "case": case,
            "rationale": rationale,
            "n_components_R": int(row["n_components_R"]),
            "n_bond_changes": int(row["n_bond_changes"]),
        }
        counts[case] += 1

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    log.info("Case distribution: A=%d B=%d C=%d", counts["A"], counts["B"], counts["C"])
    log.info("Wrote %s", output_json)
    return {"counts": counts, "by_reaction": out}
