#!/usr/bin/env python3
"""diag_p4.py — diagnostic summary of results/P4_stereo.csv (no gate): how far from the ring plane the mismatched
substituents are, by comparison kind, core and stereo_index difference."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1pair_common as C
df = pd.read_csv(C.RES / "P4_stereo.csv")
rows = []
for r in df.itertuples():
    if not isinstance(r.detail, str) or not r.detail.startswith("["):
        continue
    for c in json.loads(r.detail):
        rows.append(dict(pair_id=r.pair_id, ref_kind=r.ref_kind, core=r.core, fam=r.fam, sdiff=r.stereo_a != r.stereo_b,
                         kind=c["kind"], pos=c["pos"], d_a=c["d_a"], d_b=c["d_b"], same=c["same"], same_local=c["same_local"]))
X = pd.DataFrame(rows)
X["dmin"] = np.minimum(X.d_a.abs(), X.d_b.abs())
out = {"n_comparisons": len(X), "n_mismatch": int((~X.same).sum())}
for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
    m = X[(~X.same) & (X.dmin >= thr)]
    out[f"mismatch_dmin_ge_{thr}"] = dict(n=len(m), pairs=int(m.pair_id.nunique()), by_kind=m.kind.value_counts().to_dict(),
                                         sdiff=int(m.sdiff.sum()), cores=m.core.value_counts().head(12).to_dict())
out["mismatch_dmin_quantiles"] = X.loc[~X.same, "dmin"].quantile([.5, .9, .99, 1]).round(4).to_dict()
out["match_dmin_quantiles"] = X.loc[X.same, "dmin"].quantile([.01, .1, .5]).round(4).to_dict()
out["by_pos_kind"] = X.groupby(["pos", "kind"]).agg(n=("same", "size"), mis=("same", lambda s: int((~s).sum())),
                                                    dmed=("dmin", "median")).reset_index().to_dict("records")
big = X[(~X.same) & (X.dmin >= 0.3)]
out["big_examples"] = big.head(40).to_dict("records")
rf = df[df.status == "ring_fail"]
out["ring_fail_reasons"] = rf.detail.str.split(":").str[0].value_counts().to_dict()
out["ring_fail_detail_top"] = rf.detail.value_counts().head(10).to_dict()
amb = df[df.status == "x_site_ambiguous"]
out["x_site_ambiguous"] = dict(n=len(amb), n_diff_pos=amb.n_diff_pos.value_counts().to_dict(), cores=amb.core.value_counts().head(10).to_dict(),
                               pos=amb.pos.value_counts().to_dict())
print(json.dumps(out, indent=1, default=str))
