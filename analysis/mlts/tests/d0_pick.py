"""tests/d0_pick.py — sandbox only: print small test structures of $MLT/dft/plan/plan_d0.parquet (and the D1 plan when
present) as JSON: the smallest d0_ts_near n1 row, d0_frag, d0_react_disp, and the smallest charged d0_frag /
d0_react_disp. Writes nothing."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

import gen_dft as G  # noqa: E402

out = {}
if G.PLAN_D0_PQ.is_file():
    P = pd.read_parquet(G.PLAN_D0_PQ, columns=["struct_id", "group", "kind", "n_atoms", "charge", "d0_id", "rxn_ids",
                                                "dipoles", "mol_id"])
    P = P.sort_values(["n_atoms", "struct_id"])
    for kind in ("d0_ts_near", "d0_ts_imag", "d0_frag", "d0_react_ref", "d0_react_disp"):
        x = P[(P.kind == kind) & P.struct_id.str.endswith(("_n1", "_i", "_f1", "_f2", "_ref", "_d1"))]
        if len(x):
            r = x.iloc[0]
            out[kind] = dict(struct_id=r.struct_id, n_atoms=int(r.n_atoms), charge=int(r.charge), d0_id=int(r.d0_id),
                             dipoles=r.dipoles, rxn_ids=r.rxn_ids[:60])
        y = x[x.charge != 0]
        if len(y):
            r = y.iloc[0]
            out[kind + "_charged"] = dict(struct_id=r.struct_id, n_atoms=int(r.n_atoms), charge=int(r.charge),
                                          d0_id=int(r.d0_id), dipoles=r.dipoles)
    s = P[(P.kind == "d0_frag") & (P.rxn_ids != "")]
    if len(s):
        r = s.iloc[0]
        out["d0_frag_shared_d1"] = dict(struct_id=r.struct_id, n_atoms=int(r.n_atoms), d0_id=int(r.d0_id),
                                        dipoles=r.dipoles, n_rxn_ids=len(r.rxn_ids.split(",")))
if G.PLAN_PQ.is_file():
    Q = pd.read_parquet(G.PLAN_PQ, columns=["struct_id", "group", "kind", "n_atoms"]).sort_values(["n_atoms", "struct_id"])
    out["d1"] = {k: Q[Q.kind == k].struct_id.iloc[0] for k in ("ts_near", "frag", "react_ref") if (Q.kind == k).any()}
print(json.dumps(out, indent=1))
