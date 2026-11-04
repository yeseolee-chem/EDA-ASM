"""tests/d0_plan_check.py — sandbox only, READ ONLY: compare a new D0 plan (argv[1] root) with an older one (argv[2] root):
struct_id set difference, the review cases (3270 skipped, 174 / 175 f1 tagged, the resonance molecule skipped), tag
counts by kind, d0_class.csv overlap columns. Prints JSON."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import pandas as pd  # noqa: E402

new, old = Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None
cols = ["struct_id", "group", "kind", "rxn_ids", "folds", "dipoles", "d0_id", "d0_class", "mol_key"]
P = pd.read_parquet(new / "dft/plan/plan_d0.parquet", columns=cols)
out = dict(n=len(P), by_kind=P.kind.value_counts().to_dict(),
           with_rxn_ids={k: int((x.rxn_ids != "").sum()) for k, x in P.groupby("kind")})
sel = P[P.d0_id.isin([174, 175, 3270]) & P.kind.isin(["d0_frag", "d0_ts_near"])]
out["cases"] = [dict(struct_id=r.struct_id, n_rxn_ids=len([x for x in r.rxn_ids.split(",") if x]), folds=r.folds,
                     dipoles=r.dipoles) for r in sel.itertuples()]
out["resonance_mol_rows"] = int(P.mol_key.str.contains(r"C=\[N\+\]\(C\)\[CH-\]C\(C\)=O", regex=True).sum())
C = pd.read_csv(new / "dft/plan/d0_class.csv", keep_default_na=False)
out["class_csv_3270"] = C[C.d0_id == 3270][["d0_id", "overlap", "overlap_d1", "overlap_d1_ident", "planned",
                                             "dipole_mol_n_d1_rxns"]].to_dict("records")
out["class_csv_cols"] = list(C.columns)
out["n_dipole_mol_shared_d1_planned"] = int(((C.dipole_mol_n_d1_rxns > 0) & (C.planned.astype(str) == "True")).sum())
if old:
    Q = pd.read_parquet(old / "dft/plan/plan_d0.parquet", columns=cols)
    a, b = set(P.struct_id), set(Q.struct_id)
    out["vs_old"] = dict(only_new=sorted(a - b)[:20], n_only_new=len(a - b), only_old=sorted(b - a)[:20],
                         n_only_old=len(b - a))
    m = P.merge(Q, on="struct_id", suffixes=("", "_o"))
    for c in ("rxn_ids", "folds", "dipoles"):
        d = m[m[c] != m[c + "_o"]]
        out["vs_old"][f"n_{c}_changed"] = int(len(d))
        out["vs_old"][f"{c}_changed_by_kind"] = d.kind.value_counts().to_dict()
        out["vs_old"][f"{c}_changed_first"] = d.struct_id.head(6).tolist()
print(json.dumps(out, indent=1, default=str))
