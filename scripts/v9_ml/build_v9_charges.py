"""Build v9 charges parquet matching orca_eda_charges_v7 schema."""
import json, pandas as pd
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
LABELS = REPO/"outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet"
OUT = REPO/"labels/orca/orca_eda_charges_v9.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]

df = pd.read_parquet(LABELS)
rows=[]
for _, r in df.iterrows():
    rid = r["reaction_id"]
    fam = _fam(rid)
    fA=0; fB=0; tot=0
    if fam in ("qmrxn20_sn2","qmrxn20_e2"):
        fB=-1; tot=-1
    if rid in ("dipolar_004594","dipolar_005435"):
        fA=-1; fB=0; tot=-1
    rows.append({"reaction_id":rid, "family":fam,
                 "total_charge":tot,
                 "fragment_charge_a":fA, "fragment_charge_b":fB,
                 "fragment_mult_a":1, "fragment_mult_b":1})
pd.DataFrame(rows).to_parquet(OUT, index=False)
print(f"wrote {OUT}  n={len(rows)}")
