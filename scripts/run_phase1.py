"""Top-level Phase 1 orchestrator.

Runs Stages 3.1 — 3.8 sequentially, stopping at the gates:
  - Gate 1 (after Stage 3.3): user reviews sampling_report.html
  - Gate 2 (after Stage 3.5): user picks Case-C policy
  - Gate 3 (after Stage 3.8): user fills manual_review_log.json

Resumes only the stages whose outputs are missing.

Usage:
  python scripts/run_phase1.py --to 3.3       # stop at gate 1
  python scripts/run_phase1.py --case-c manual --to 3.8  # stop at gate 3
  python scripts/run_phase1.py --finalize     # run 3.9 + 3.10 after manual review
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.paths import (  # noqa: E402
    BOND_CHANGES_PARQUET,
    CASE_JSON,
    FRAGMENTS_AUTO_JSON,
    INDEX_PARQUET,
    PHASE1_H5,
    REVIEW_DIR,
    SAMPLING_REPORT_HTML,
    SELECTED_CSV,
    ensure_dirs,
)
from eda_asm.phase1.logging_setup import get_logger  # noqa: E402

STAGES_BEFORE_GATES = ["3.1", "3.2", "3.3"]  # gate 1 after 3.3
STAGES_GATE2 = ["3.4", "3.5"]                # gate 2 after 3.5
STAGES_GATE3 = ["3.6", "3.7", "3.8"]         # gate 3 after 3.8
STAGES_FINAL = ["3.9", "3.10"]


def stage_outputs_exist(stage: str) -> bool:
    return {
        "3.1": INDEX_PARQUET.exists(),
        "3.2": BOND_CHANGES_PARQUET.exists(),
        "3.3": SELECTED_CSV.exists() and SAMPLING_REPORT_HTML.exists(),
        "3.4": True,  # checked per-reaction npz inside the stage
        "3.5": CASE_JSON.exists(),
        "3.6": FRAGMENTS_AUTO_JSON.exists(),
        "3.7": FRAGMENTS_AUTO_JSON.exists(),
        "3.8": REVIEW_DIR.exists() and any(REVIEW_DIR.glob("*.html")),
        "3.9": False,  # always rerun on --finalize
        "3.10": PHASE1_H5.exists(),
    }[stage]


def run_stage(stage: str, *, case_c: str = "manual", seed: int = 42) -> None:
    log = get_logger("phase1.driver")
    log.info("[driver] launching stage %s", stage)
    if stage == "3.1":
        from eda_asm.phase1.stage_3_1_index import run
        run()
    elif stage == "3.2":
        from eda_asm.phase1.stage_3_2_bond_changes import run
        run()
    elif stage == "3.3":
        from eda_asm.phase1.stage_3_3_sampling import run as run_sample
        from eda_asm.phase1.sampling_report import build as build_report
        result = run_sample(seed=seed)
        build_report(
            selected=result["selected"],
            population=result["population_df"],
            quotas=result["quotas"],
            cell_log=result["cell_log"],
        )
    elif stage == "3.4":
        from eda_asm.phase1.stage_3_4_five_points import run
        run()
    elif stage == "3.5":
        from eda_asm.phase1.stage_3_5_classify import run
        run()
    elif stage == "3.6":
        from eda_asm.phase1.stage_3_6_frag_caseA import run
        run()
    elif stage == "3.7":
        from eda_asm.phase1.stage_3_7_frag_caseB import run
        run()
    elif stage == "3.8":
        from eda_asm.phase1.stage_3_8_review_queue import run
        run(case_c_strategy=case_c)
    elif stage == "3.9":
        from eda_asm.phase1.stage_3_9_final_fragments import run
        run()
    elif stage == "3.10":
        from eda_asm.phase1.stage_3_10_h5 import run
        run()
    else:
        raise ValueError(f"unknown stage {stage}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=None, help="last stage to run before stopping (e.g. 3.3, 3.8)")
    ap.add_argument("--from-stage", default=None, help="first stage to run (e.g. 3.4)")
    ap.add_argument("--case-c", choices=["manual", "exclude"], default="manual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finalize", action="store_true", help="run 3.9 + 3.10 only")
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    args = ap.parse_args()

    ensure_dirs()
    if args.finalize:
        run_stage("3.9")
        run_stage("3.10")
        return

    all_stages = STAGES_BEFORE_GATES + STAGES_GATE2 + STAGES_GATE3
    start = args.from_stage or "3.1"
    end = args.to or "3.8"
    run = False
    for s in all_stages:
        if s == start:
            run = True
        if not run:
            continue
        if not args.force and stage_outputs_exist(s):
            print(f"[skip] {s} (outputs already exist)")
        else:
            run_stage(s, case_c=args.case_c, seed=args.seed)
        if s == end:
            break


if __name__ == "__main__":
    main()
