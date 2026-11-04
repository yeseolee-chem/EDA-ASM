#!/usr/bin/env python3
"""ewps_probe.py — read-only probe for the EWPS implementer (eval_v2 / final_w / predict_v2 / summary_v2): which
python packages each environment has, and the shapes of the upstream files E / W / P / S read. Writes nothing."""
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
for e in ("espley", "fu2", "d1", "feat"):
    if e in sys.argv:
        import inputs_mlts as IM
        IM.ensure_env(e)
out = dict(python=sys.version.split()[0], prefix=sys.prefix, env=os.environ.get("MLT_IN_ENV") or os.environ.get("MLTENV"))
for m in ("numpy", "pandas", "scipy", "matplotlib", "networkx", "pyarrow", "xgboost", "sklearn", "torch", "rdkit"):
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out[m] = f"missing ({type(e).__name__})"
if "files" in sys.argv:
    import pandas as pd
    import mlt_common as M
    R = M.REPO / "analysis"
    K = pd.read_parquet(R / "deploy" / "results" / "K_predictions.parquet")
    out["lf"] = dict(cols=list(K.columns), n=len(K), rungs=sorted(K.rung.unique()), arms=sorted(K.arm.unique()),
                     models=sorted(K.model.unique()), n_pairs=int(K.pair_id.nunique()))
    L = pd.read_parquet(R / "d1_ladder" / "results" / "L_predictions.parquet", columns=["pair_id", "rung", "arm", "model"])
    out["L"] = {r: int(g.pair_id.nunique()) for r, g in L.groupby("rung")}
    out["mlt_ls"] = sorted(p.name for p in M.MLT.iterdir())
    for d in ("preds", "pipeline", "models", "inputs", "eval", "w"):
        p = M.MLT / d
        out[f"mlt_{d}"] = sorted(x.name for x in p.iterdir())[:40] if p.is_dir() else None
    T = pd.read_parquet(M.LAD / "inputs" / "dft.parquet")
    out["dft_table_cols_head"] = list(T.columns)[:12] + ["..."] + list(T.columns)[-30:]
    lab = json.loads(M.LABELS.read_text())
    byid = {int(d["rxn_id"]): d for d in lab}
    F = pd.read_csv(M.LAD / "F_pairs_cr2.csv", usecols=["pair_id", "a_rxn", "b_rxn", "dd_disp", "dd_oi"])
    q = F.head(2000)
    da = [byid[int(a)].get("disp_dft") for a in q.a_rxn]
    db = [byid[int(b)].get("disp_dft") for b in q.b_rxn]
    import numpy as np
    d = np.asarray(da, float) - np.asarray(db, float)
    out["dd_disp_sign"] = dict(max_abs_a_minus_b=float(np.nanmax(np.abs(d - q.dd_disp))),
                               max_abs_b_minus_a=float(np.nanmax(np.abs(-d - q.dd_disp))), keys=sorted(lab[0])[:60])
print(json.dumps(out, indent=1, default=str))
