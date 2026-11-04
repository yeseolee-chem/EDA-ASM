"""tests/d0plus_coldiff.py — sandbox only, READ ONLY (D0-plus): per column, how many D0G1 / D0G3 rows of the test-root
plan_d0.parquet differ from the production plan (pos_A compared as float arrays, other values as JSON); the first
differing values of each column. Prints JSON."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gen_dft as G  # noqa: E402

PROD_PQ = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/plan/plan_d0.parquet")


def g13(df):
    return df[df.group.isin(["D0G1", "D0G3"])].sort_values("struct_id").reset_index(drop=True)


a, b = g13(pd.read_parquet(G.PLAN_D0_PQ)), g13(pd.read_parquet(PROD_PQ))
out = {}
for c in b.columns:
    if c == "pos_A":
        d = [i for i, (x, y) in enumerate(zip(a[c], b[c])) if not np.array_equal(np.asarray(x, float), np.asarray(y, float))]
    else:
        d = [i for i, (x, y) in enumerate(zip(a[c], b[c])) if json.dumps(x, default=str) != json.dumps(y, default=str)]
    if d:
        out[c] = dict(n=len(d), kinds=a.loc[d, "kind"].value_counts().to_dict(),
                      first=[dict(struct_id=a.struct_id[i], test=str(a[c][i])[:300], prod=str(b[c][i])[:300]) for i in d[:3]])
print(json.dumps(dict(n=len(a), columns_differing=out), indent=1, default=str))
