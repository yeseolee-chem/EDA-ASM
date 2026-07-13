"""Stage 2 (v8) — MACE-OFF23_medium feature extraction for the v8 cohort.

Iterates over reaction_id in outputs/v8_review/labels/labels_v8_5channel.parquet
(799 reactions after review). Geometries are already cleaned and staged as
outputs/v8_review/raw_geoms/{rid}/{R,TS,P}.xyz, so no per-family switching is
needed — just ase.io.read the three files directly.

Frozen MACE-OFF23_medium via eda_asm.asr_v1.backbone_maceoff.MACEOFFFeatureExtractor.
Per-atom invariant features (256-d). One .pt per reaction with keys {R,TS,P},
each holding {feat: [n_atoms, 256] float32, z: list[int], pos: [n_atoms, 3] float32}.

Output layout:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8/{rid}.pt
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8/_progress.jsonl

Sharding: --shard S --nshards N processes reactions where row_index % N == S.
Idempotent: skips a rid whose output .pt already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor

LABELS = REPO / "outputs/v8_review/labels/labels_v8_5channel.parquet"
GEOM_ROOT = REPO / "outputs/v8_review/raw_geoms"

OUT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium_v8")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "_progress.jsonl"


def load_triple(rid: str):
    d = GEOM_ROOT / rid
    R = ase.io.read(str(d / "R.xyz"))
    TS = ase.io.read(str(d / "TS.xyz"))
    P = ase.io.read(str(d / "P.xyz"))
    return R, TS, P


def run_one(fe: MACEOFFFeatureExtractor, rid: str) -> dict:
    R, TS, P = load_triple(rid)
    feat_R = fe.extract(R)
    feat_TS = fe.extract(TS)
    feat_P = fe.extract(P)
    return {
        "reaction_id": rid,
        "R":  {"z": R.get_atomic_numbers().tolist(),
               "pos": R.get_positions().astype(np.float32),
               "feat": feat_R.cpu().numpy().astype(np.float32)},
        "TS": {"z": TS.get_atomic_numbers().tolist(),
               "pos": TS.get_positions().astype(np.float32),
               "feat": feat_TS.cpu().numpy().astype(np.float32)},
        "P":  {"z": P.get_atomic_numbers().tolist(),
               "pos": P.get_positions().astype(np.float32),
               "feat": feat_P.cpu().numpy().astype(np.float32)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="medium",
                    help="MACE-OFF23 size: small|medium|large (default medium)")
    ap.add_argument("--shard", type=int, default=0,
                    help="This worker's shard index in [0, nshards)")
    ap.add_argument("--nshards", type=int, default=1,
                    help="Total number of shards (row_index %% nshards == shard)")
    args = ap.parse_args()

    if args.nshards < 1:
        raise ValueError("--nshards must be >= 1")
    if not (0 <= args.shard < args.nshards):
        raise ValueError(f"--shard {args.shard} out of range for --nshards {args.nshards}")

    df = pd.read_parquet(LABELS)
    print(f"[{time.strftime('%H:%M:%S')}] cohort = {len(df)} reactions (v8)")
    if "family" in df.columns:
        print(df.family.value_counts())

    # Shard by row index (deterministic given fixed parquet ordering)
    df = df.reset_index(drop=True)
    mask = (np.arange(len(df)) % args.nshards) == args.shard
    subset = df[mask].reset_index(drop=True)
    print(f"[{time.strftime('%H:%M:%S')}] shard {args.shard}/{args.nshards} -> "
          f"{len(subset)} reactions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] loading MACE-OFF23 {args.model_size} on {device}")
    fe = MACEOFFFeatureExtractor(model_size=args.model_size, device=device)
    print(f"[{time.strftime('%H:%M:%S')}] feature_dim = {fe.feature_dim}")

    done = fail = skipped = 0
    t0 = time.time()
    with open(PROGRESS, "a") as pf:
        for i, row in subset.iterrows():
            rid = row.reaction_id
            out = OUT_DIR / f"{rid}.pt"
            if out.exists():
                skipped += 1
                continue
            try:
                d = run_one(fe, rid)
                # Atomic write: to tmp, then rename
                tmp = out.with_suffix(".pt.tmp")
                torch.save(d, tmp)
                tmp.rename(out)
                done += 1
                pf.write(json.dumps({"rid": rid, "shard": args.shard,
                                     "status": "ok"}) + "\n")
                pf.flush()
            except Exception as e:
                fail += 1
                pf.write(json.dumps({"rid": rid, "shard": args.shard,
                                     "err": f"{type(e).__name__}: {e}"}) + "\n")
                pf.flush()
            if (done + fail + skipped) % 25 == 0:
                elapsed = time.time() - t0
                rate = (done + fail) / max(elapsed, 1e-6)
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{done+fail+skipped}/{len(subset)} "
                      f"done={done} skipped={skipped} fail={fail} "
                      f"rate={rate:.2f}/s")

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE  elapsed={elapsed:.1f}s "
          f"done={done} skipped={skipped} fail={fail} of {len(subset)}")


if __name__ == "__main__":
    main()
