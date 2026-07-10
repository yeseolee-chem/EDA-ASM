"""Stage 2 v7 — MACE-OFF23_medium feature extraction for the 776-reaction v7
cohort. Idempotent: skips rxns whose .pt already exists AND has complete
R/TS/P feat blocks. Re-extracts anything incomplete.

Uses the pretrained MACE-OFF23_medium backbone at
    /home1/yeseo1ee/.cache/mace/MACE-OFF23_medium.model
(NOT the halo8 fine-tune in progress — that's separate).

Output: /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt
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
sys.path.insert(0, str(REPO / "pipeline_rebuild"))

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor
from stage2_mace_features import load_triple  # reuse raw loader

LABELS_V7 = REPO / "labels/orca/orca_eda_labels_v7.parquet"
OUT_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
PROGRESS  = OUT_DIR / "_progress_v7.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_complete(pt: Path) -> bool:
    try:
        d = torch.load(str(pt), weights_only=False, map_location="cpu")
    except Exception:
        return False
    for k in ("R", "TS", "P"):
        if k not in d or not isinstance(d[k], dict) or "feat" not in d[k]:
            return False
    return True


def run_one(fe, rid, fam):
    R, TS, P = load_triple(rid, fam)
    feat_R  = fe.extract(R)
    feat_TS = fe.extract(TS)
    feat_P  = fe.extract(P)
    return {
        "reaction_id": rid,
        "family":      fam,
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
    ap.add_argument("--model-size", default="medium")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_parquet(LABELS_V7)
    # Round-robin shard split
    df = df.reset_index(drop=True)
    df = df[df.index % args.nshards == args.shard].reset_index(drop=True)
    print(f"[{time.strftime('%H:%M:%S')}] shard {args.shard}/{args.nshards} -> {len(df)} rxns")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] loading MACE-OFF23 {args.model_size} on {device}")
    fe = MACEOFFFeatureExtractor(model_size=args.model_size, device=device)
    print(f"[{time.strftime('%H:%M:%S')}] feature_dim = {fe.feature_dim}")

    done = fail = skip = 0
    t0 = time.time()
    with open(PROGRESS, "a") as pf:
        for i, row in df.iterrows():
            rid, fam = row.reaction_id, row.family
            out = OUT_DIR / f"{rid}.pt"
            if out.exists() and is_complete(out):
                skip += 1
                continue
            try:
                d = run_one(fe, rid, fam)
                tmp = out.with_suffix(".pt.tmp")
                torch.save(d, tmp)
                tmp.rename(out)   # atomic
                done += 1
            except Exception as e:
                fail += 1
                pf.write(json.dumps({"rid": rid, "family": fam,
                                     "err": f"{type(e).__name__}: {e}"}) + "\n")
                pf.flush()
            if (done + fail + skip) % 25 == 0:
                elapsed = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {done+fail+skip}/{len(df)} "
                      f"done={done} fail={fail} skip={skip} elapsed={elapsed:.0f}s")

    print(f"[{time.strftime('%H:%M:%S')}] DONE  done={done} fail={fail} skip={skip}")


if __name__ == "__main__":
    main()
