"""Thin wrapper around ``train_cv.py --model m1`` for the backbone-comparison
spec CLI (ASR_Backbone_Comparison_Spec_v1.0 §10).
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
                    help="output dir (defaults to outputs/asr_v1/m1)")
    args = ap.parse_args()

    forwarded = ["train_cv.py", "--config", args.config, "--model", "m1"]
    if args.features is not None:
        forwarded += ["--features", args.features]
    if args.out is not None:
        forwarded += ["--output-dir", args.out]

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.argv = forwarded
    import train_cv                                                # noqa: F401
    train_cv.main()


if __name__ == "__main__":
    main()
