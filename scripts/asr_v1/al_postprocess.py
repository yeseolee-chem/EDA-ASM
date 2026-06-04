"""After the round's ADF batch finishes, parse the per-reaction status.json +
.out files under the round's adf_inputs/ and append new labels to the master
asr_labels.parquet.

Writes:
  outputs/asr_v1/al/round_<R>/round_labels.parquet  — new labels only
  ADF_250/adf_outputs/parsed/asr_labels.parquet     — overwritten with
        (old + new) labels merged on reaction_id (new wins on dup, which
        should not happen because AL picks were unlabeled)
  ADF_250/adf_outputs/parsed/asr_labels.before_round<R>.parquet — backup
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--round-dir", default=None)
    ap.add_argument("--labels-parquet",
                    default="ADF_250/adf_outputs/parsed/asr_labels.parquet")
    args = ap.parse_args()

    repo = Path.cwd()
    round_dir = Path(args.round_dir or f"outputs/asr_v1/al/round_{args.round:02d}")
    adf_inputs_root = round_dir / "adf_inputs"
    if not adf_inputs_root.is_dir():
        raise FileNotFoundError(f"missing {adf_inputs_root}")

    # Parse the round's ADF outputs into a parquet via the existing extractor.
    round_parquet = round_dir / "round_labels.parquet"
    print(f"[post] parsing {adf_inputs_root} → {round_parquet}")
    res = subprocess.run(
        [
            sys.executable, "ADF_250/scripts/adf/extract_asr_labels.py",
            "--adf-root", str(adf_inputs_root),
            "--out", str(round_parquet),
        ],
        check=False, capture_output=True, text=True,
    )
    print((res.stdout or "")[-1500:])
    if res.returncode != 0:
        print((res.stderr or "")[-2000:])
        raise RuntimeError("extract_asr_labels.py failed")

    new_df = pd.read_parquet(round_parquet)
    print(f"[post] new labels this round: {len(new_df)}  by family: "
          f"{new_df['family'].value_counts().to_dict()}")

    # Backup and merge into the master parquet.
    master = Path(args.labels_parquet)
    backup = master.with_name(f"asr_labels.before_round{args.round:02d}_"
                              f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    shutil.copy(master, backup)
    old_df = pd.read_parquet(master)
    overlap = set(old_df["reaction_id"]) & set(new_df["reaction_id"])
    if overlap:
        print(f"[post] WARNING: {len(overlap)} ids in both old + new (keeping new)")
    merged = pd.concat([
        old_df[~old_df["reaction_id"].isin(new_df["reaction_id"])],
        new_df,
    ], ignore_index=True)
    merged.to_parquet(master)
    print(f"[post] merged labels: {len(old_df)} + {len(new_df)} = "
          f"{len(merged)} → {master}  (backup {backup.name})")


if __name__ == "__main__":
    main()
