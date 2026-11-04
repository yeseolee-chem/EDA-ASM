"""tests/d0plus_rows.py — sandbox only, READ ONLY (D0-plus): compare the test-root D0 plan ($MLT_SCRATCH) with the
production plan /gpfs/tmp_cpu2/yeseo1ee/mlts/dft/plan/plan_d0.parquet: D0G1 / D0G3 rows (struct ids, every column,
geometry hash), then the D0G2 selection (training only in g2_plan_d0.json 'rxns', D0-gauss), the holdout list file and
its checks (eval in pool, distinct reactant pairs, sibling closure, disjoint from the training units and their groups,
class, d1val, sha256 in g2_plan_d0.json / plan_d0_meta.json / d0_holdout_ref.json / results copy), d0_holdout_ref.json
(reference = Coley TS, start = G2c xTB TS) and the G_plan_d0.json cap check. Prints JSON."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402
import gen_d0 as D  # noqa: E402

PROD_PQ = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/plan/plan_d0.parquet")
P, Q = pd.read_parquet(G.PLAN_D0_PQ), pd.read_parquet(PROD_PQ)


def g13(df):
    return df[df.group.isin(["D0G1", "D0G3"])].sort_values("struct_id").reset_index(drop=True)


def val(v, root):
    """arrays -> float lists; the $MLT root prefix (hess_src of the run-step Hessian cache) -> '$MLT' (root-relative:
    the only column that differs between a test root and production, tests/d0plus_coldiff.py)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    return v.replace(root, "$MLT") if isinstance(v, str) else v


def row_hash(df, root):
    cols = sorted(df.columns)
    return [hashlib.sha256(json.dumps({c: val(r[c], root) for c in cols}, sort_keys=True, default=str).encode()).hexdigest()
            for r in df.to_dict("records")]


def geom_hash(df):
    h = hashlib.sha256()
    for r in df.itertuples(index=False):
        h.update(f"{r.struct_id}|{r.symbols}|".encode())
        h.update(np.asarray(r.pos_A, np.float64).tobytes())
    return h.hexdigest()


a, b = g13(P), g13(Q)
ha, hb = row_hash(a, str(M.MLT)), row_hash(b, "/gpfs/tmp_cpu2/yeseo1ee/mlts")
out = dict(test_plan=str(G.PLAN_D0_PQ), prod_plan=str(PROD_PQ), prod_sha256=M.sha256(PROD_PQ),
           d0g1_d0g3=dict(n_test=len(a), n_prod=len(b), same_struct_ids=list(a.struct_id) == list(b.struct_id),
                          same_columns=list(P.columns) == list(Q.columns),
                          same_dtypes=dict(P.dtypes.astype(str)) == dict(Q.dtypes.astype(str)),
                          n_rows_differing_all_columns_root_relative=int(sum(x != y for x, y in zip(ha, hb)))
                          if len(ha) == len(hb) else None,
                          rows_sha256_test=hashlib.sha256("".join(ha).encode()).hexdigest(),
                          rows_sha256_prod=hashlib.sha256("".join(hb).encode()).hexdigest(),
                          geometry_sha256_test=geom_hash(a), geometry_sha256_prod=geom_hash(b),
                          seeds_equal=bool((a.seed.values == b.seed.values).all()) if len(a) == len(b) else False,
                          by_group_kind={f"{g}/{k}": int(n) for (g, k), n in a.groupby(["group", "kind"]).size().items()}),
           d0g2_rows_test=int((P.group == "D0G2").sum()), d0g2_rows_prod=int((Q.group == "D0G2").sum()))
G2 = P[P.group == "D0G2"]
roles = [json.loads(s).get("role") for s in G2.disp]
out["d0g2_rows_by_role"] = dict(Counter(roles))
g2p = M.read_json(G.G2_PLAN_D0)
pool = M.read_json(G.G2_POOL_D0)["ranked"]
pool_ids = {int(u["rxn_id"]) for u in pool}
pool_by = {int(u["rxn_id"]): u for u in pool}
train = list(g2p["rxns"])
tids = {int(u["rxn_id"]) for u in train}
H = pd.read_csv(D.HOLD_CSV, keep_default_na=False)
ev = [int(x) for x, r in zip(H.d0_id, H.role) if r == "eval"]
sb = [int(x) for x, r in zip(H.d0_id, H.role) if r == "sibling"]
hids = ev + sb
pc = Counter(u["d0_class"] for u in pool)
tc = Counter(u["d0_class"] for u in train)
hc = Counter(pool_by[x]["d0_class"] for x in ev if x in pool_by)
out["selection"] = dict(n_g2=g2p.get("n_g2"), n_holdout=g2p.get("n_holdout"), n_units=g2p.get("n_units"),
                        n_g2_source=g2p.get("n_g2_source"), pinned_n_g2=g2p.get("pinned_n_g2"), n_train=len(train),
                        train_roles=dict(Counter(u.get("role") for u in train)),
                        no_listed_reaction_in_rxns=not (tids & set(hids)),
                        train_is_rank_prefix=[int(u["rank"]) for u in train] == list(range(len(train))),
                        per_class={s: dict(pool=pc[s], train=tc.get(s, 0), train_expected=round(len(train) * pc[s] / len(pool), 1),
                                           holdout=hc.get(s, 0)) for s in sorted(pc)},
                        train_n_atoms_median=float(np.median([u["n_atoms"] for u in train])) if train else None,
                        hold_n_atoms_median=float(np.median([pool_by[x]["n_atoms"] for x in ev])) if ev else None,
                        pool_n_atoms_median=float(np.median([u["n_atoms"] for u in pool])))
C = pd.read_csv(G.D0_CLASS_CSV, keep_default_na=False)
cmap = dict(zip(C.d0_id.astype(int), C.d0_class.astype(str)))
planned = sorted(int(r) for r, p in zip(C.d0_id, C.planned) if str(p) == "True")
S = pd.read_csv(D.D0_SMILES, usecols=["rxn_id", "rxn_smiles"])
smi0 = dict(zip(S.rxn_id.astype(int), S.rxn_smiles))
pair = {r: D.pair_ident(smi0[r]) for r in planned}
by_pair = {}
for r in planned:
    by_pair.setdefault(pair[r], set()).add(r)
listed = set(hids)
meta = M.read_json(G.PLAN_D0_META)
sha = M.sha256(D.HOLD_CSV)
R = M.read_json(D.HOLD_REF)
out["holdout_file"] = dict(path=str(D.HOLD_CSV), columns=list(H.columns), n_rows=len(H), n_eval=len(ev), n_sibling=len(sb),
                           sha256=sha, eval_by_class=dict(sorted(Counter(cmap[x] for x in ev).items())),
                           sibling_by_class=dict(sorted(Counter(cmap[x] for x in sb).items())),
                           eval_all_in_pool=set(ev) <= pool_ids, listed_all_planned=listed <= set(planned),
                           eval_distinct_pairs=len({pair[x] for x in ev}),
                           closure_missing=sorted(m for x in listed for m in by_pair[pair[x]] if m not in listed)[:10],
                           every_sibling_shares_an_eval_pair=all(pair[x] in {pair[e] for e in ev} for x in sb),
                           disjoint_from_train=not (listed & tids),
                           train_groups_hit=len({pair[x] for x in listed} & {pair[t] for t in tids}),
                           class_matches_d0_class_csv=all(cmap[i] == c for i, c in zip(H.d0_id.astype(int), H.d0_class)),
                           in_d1val=sorted(listed & set(D.d1val_ids())),
                           sha_in_g2_plan=(g2p.get("holdout") or {}).get("sha256") == sha,
                           sha_in_plan_meta=meta.get("holdout_sha256") == sha,
                           sha_in_ref_json=R.get("holdout_sha256") == sha,
                           results_copy_identical=D.HOLD_CSV_RES.is_file() and M.sha256(D.HOLD_CSV_RES) == sha,
                           g2_plan_holdout_ids_match=sorted((g2p.get("holdout") or {}).get("ids") or []) == sorted(ev),
                           g2_plan_sibling_ids_match=sorted((g2p.get("holdout") or {}).get("sibling_ids") or []) == sorted(sb),
                           first_rows=H.head(2).to_dict("records"))
ref_ok = all(u["reference"]["xyz"] == pool_by[u["d0_id"]]["label_ts"] and u["starts"]["xtb_g2c"] == pool_by[u["d0_id"]]["start"]
             for u in R["units"])
out["holdout_ref"] = dict(path=str(D.HOLD_REF), n=R["n"], n_siblings=R.get("n_siblings"), sha_matches=R["holdout_sha256"] == sha,
                          ids_match=sorted(u["d0_id"] for u in R["units"]) == sorted(ev),
                          reference_is_coley_ts_and_start_is_g2c=ref_ok,
                          first_unit={k: v for k, v in R["units"][0].items() if k != "pairs"})
J = M.read_json(G.G_PLAN_D0_JSON)
out["cap_check"] = {k: v for k, v in J["budget"]["cap_check"].items() if k not in ("rule",)}
out["holdout_allocation"] = (J.get("d0g2_selection") or {}).get("holdout_allocation")
if out["holdout_allocation"]:
    for k in ("rule", "sib_rule", "by_class"):
        out["holdout_allocation"].pop(k, None)
print(json.dumps(out, indent=1, default=str))
