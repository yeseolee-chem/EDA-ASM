"""Thin wrapper around ``train_cv.py --model b0`` for the backbone-comparison
spec CLI (ASR_Backbone_Comparison_Spec_v1.0 §10).

Existing driver code in ``train_cv.py`` already runs 5-fold × 5-ensemble
B0 training; this script just rewrites sys.argv so the spec's per-model
binaries work as written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr_v1.yaml")
    ap.add_argument("--features", default=None,
                    help="override feature_cache path from the config")
    ap.add_argument("--out", default=None,
                    help="output dir (defaults to outputs/asr_v1/b0)")
    args = ap.parse_args()

    forwarded = ["train_cv.py", "--config", args.config, "--model", "b0"]
    if args.features is not None:
        forwarded += ["--features", args.features]
    if args.out is not None:
        forwarded += ["--output-dir", args.out]

    # Defer to the existing driver.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.argv = forwarded
    import train_cv                                                # noqa: F401
    train_cv.main()


if __name__ == "__main__":
    main()
