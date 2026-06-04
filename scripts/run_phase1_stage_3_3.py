"""Driver for Stage 3.3 — stratified sampling + report HTML."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.sampling_report import build as build_report  # noqa: E402
from eda_asm.phase1.stage_3_3_sampling import DEFAULT_SEED, run as sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    result = sample(seed=args.seed)
    html = build_report(
        selected=result["selected"],
        population=result["population_df"],
        quotas=result["quotas"],
        cell_log=result["cell_log"],
    )
    print(f"\nReport written to: {html}")


if __name__ == "__main__":
    main()
