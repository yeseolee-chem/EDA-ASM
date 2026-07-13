"""Merge d26/27/28 shard parquets + sanity check pearson vs targets."""
from pathlib import Path
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
CHUNK_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/d26_28_v9_chunks")
OUT = REPO / "spec/spec05_d25_sum/data/descriptors_channel_proxies.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

parts = sorted(CHUNK_DIR.glob("shard_*.parquet"))
dfs = [pd.read_parquet(p) for p in parts]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="reaction_id", keep="last")
df.to_parquet(OUT, index=False)
n_ok = int(df["scf_ok"].sum())
print(f"merged {len(df)} rxns ({n_ok} ok, {len(df) - n_ok} failed) -> {OUT}")

# Physics sanity: pearson(d26, y_elst), pearson(d27, y_Pauli), pearson(d28, y_oi)
labels = pd.read_parquet(REPO / "outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet")
merged = df[df["scf_ok"]].merge(
    labels[["reaction_id", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal", "E_strain_kcal"]],
    on="reaction_id")
if len(merged) > 10:
    print(f"pearson(d26, V_elst) = {merged.d26.corr(merged.V_elst_kcal):+.3f}  (expect + since same sign convention: attractive elst has both -)")
    print(f"pearson(d27, Pauli)  = {merged.d27.corr(merged.Pauli_kcal):+.3f}  (expect > 0)")
    print(f"pearson(d28, E_orb)  = {merged.d28.corr(merged.E_orb_kcal):+.3f}  (expect < 0 since more overlap -> more orbital stabilisation = negative E_orb)")
