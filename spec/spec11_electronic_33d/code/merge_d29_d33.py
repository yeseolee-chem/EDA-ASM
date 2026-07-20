"""SPEC_11 - merge d29..d33 shard parquets into a single canonical parquet.

Reads shards from /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d29_33_v9_chunks/
Writes         spec/spec11_electronic_33d/data/descriptors_d29_d33.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
CHUNK_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d29_33_v9_chunks")
OUT = REPO / "spec/spec11_electronic_33d/data/descriptors_d29_d33.parquet"


def main():
    shards = sorted(CHUNK_DIR.glob("shard_*.parquet"))
    print(f"[merge_d29_d33] {len(shards)} shard files")
    if not shards:
        raise SystemExit("no shard parquets to merge")
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)
    # keep last successful row per rid; failures with error still recorded
    df = df.sort_values("scf_ok", ascending=False).drop_duplicates("reaction_id",
                                                                    keep="first")
    df = df.sort_values("reaction_id").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    n_ok = int(df["scf_ok"].sum())
    print(f"wrote {OUT}  rows={len(df)}  scf_ok={n_ok}")
    for col in ["d29", "d30", "d31", "d32", "d33"]:
        vals = df.loc[df["scf_ok"], col].astype(float)
        print(f"  {col}: n={len(vals)}  mean={vals.mean():+.3e}  "
              f"std={vals.std():+.3e}  min={vals.min():+.3e}  max={vals.max():+.3e}")


if __name__ == "__main__":
    main()
