"""Merge d25 shard parquets, produce clean output."""
from pathlib import Path
import pandas as pd

CHUNK_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d25_refR_chunks")
OUT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/spec/spec05_d25_sum/data/descriptors_d25_refR.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

parts = sorted(CHUNK_DIR.glob("shard_*.parquet"))
dfs = [pd.read_parquet(p) for p in parts]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="reaction_id", keep="last")
df.to_parquet(OUT, index=False)
n_ok = int(df["scf_ok"].sum())
print(f"merged {len(df)} rxns ({n_ok} ok, {len(df) - n_ok} failed) -> {OUT}")
if "error" in df.columns:
    from collections import Counter
    errs = df[df["scf_ok"] == False]
    if len(errs):
        print("error types:", dict(Counter(str(e).split(":")[0] for e in errs.get("error", []))))
        print("sample failures:", errs.head(5).reaction_id.tolist())
# Physics sanity: pearson(d25, y_strain)
labels = pd.read_parquet("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/labels/orca/orca_eda_labels_v7.parquet")
merged = df[df["scf_ok"] == True].merge(labels[["reaction_id", "E_strain_kcal"]], on="reaction_id")
if len(merged) > 10:
    r = merged["d25"].corr(merged["E_strain_kcal"])
    print(f"pearson(d25, E_strain) = {r:+.3f}  (expect > 0)")
