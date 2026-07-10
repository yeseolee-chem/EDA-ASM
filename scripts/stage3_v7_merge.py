"""Merge shard parquets from stage3_v7_descriptors into descriptors_v7.parquet."""
from pathlib import Path
import pandas as pd

CHUNK_DIR   = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v7_chunks")
OUT_PARQUET = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v7.parquet")

parts = sorted(CHUNK_DIR.glob("shard_*.parquet"))
if not parts:
    raise SystemExit(f"no shard parquets in {CHUNK_DIR}")
dfs = [pd.read_parquet(p) for p in parts]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="reaction_id", keep="last")
df.to_parquet(OUT_PARQUET, index=False)
n_err = df["error"].notna().sum() if "error" in df.columns else 0
print(f"merged {len(df)} rows ({n_err} with error) -> {OUT_PARQUET}")
