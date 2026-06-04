#!/usr/bin/env python
"""Aggregate per-reaction asr_label.json files into a single parquet.

Walks `ADF_800/runs/<reaction_id>/asr_label.json`, joins with manifest.csv
(reaction metadata) and fragments.parquet (partition method/charges), and
writes `ADF_800/parsed/asr_labels.parquet`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path,
                   default=REPO / "ADF_800" / "runs")
    p.add_argument("--manifest", type=Path,
                   default=REPO / "ADF_800" / "manifest.csv")
    p.add_argument("--fragments", type=Path,
                   default=REPO / "data" / "fragments" / "v1" / "fragments.parquet")
    p.add_argument("--seed-csv", type=Path,
                   default=REPO / "data" / "selection" / "initial_seed_v1"
                   / "selected_reactions.csv")
    p.add_argument("--out", type=Path,
                   default=REPO / "ADF_800" / "parsed" / "asr_labels.parquet")
    args = p.parse_args()

    rows: list[dict] = []
    for rxn_dir in sorted(args.runs_root.iterdir()):
        if not rxn_dir.is_dir():
            continue
        label_path = rxn_dir / "asr_label.json"
        if not label_path.is_file():
            continue
        d = json.loads(label_path.read_text())
        d["reaction_id"] = rxn_dir.name
        rows.append(d)

    if not rows:
        print("no asr_label.json files found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame.from_records(rows)
    print(f"loaded {len(df)} asr_label.json files")

    if args.manifest.is_file():
        manifest = pd.read_csv(args.manifest)
        df = df.merge(manifest, on="reaction_id", how="left",
                      suffixes=("", "_manifest"))
    if args.seed_csv.is_file():
        seed = pd.read_csv(args.seed_csv)[
            ["reaction_id", "source", "smiles_r", "smiles_p", "n_heavy_atoms",
             "quartile", "delta_Ea"]
        ]
        df = df.merge(seed, on="reaction_id", how="left",
                      suffixes=("", "_seed"))
    if args.fragments.is_file():
        frag = pd.read_parquet(args.fragments)[
            ["reaction_id", "partition_method", "partition_status",
             "fragment_charge_a", "fragment_charge_b",
             "min_interfragment_dist_ts"]
        ]
        df = df.merge(frag, on="reaction_id", how="left",
                      suffixes=("", "_frag"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")
    print(f"wrote {args.out}")
    print(f"by family: {df['family'].value_counts().to_dict() if 'family' in df else '(no family column)'}")
    print(f"columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
