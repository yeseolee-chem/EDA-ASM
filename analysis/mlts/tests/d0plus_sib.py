"""tests/d0plus_sib.py — sandbox only, READ ONLY (D0-plus review item 'sibling reactions'): on a d0plustest root
($MLT_SCRATCH with a D0 plan + holdout), how many holdout reactions share their reactant pair (D0-ident molecule
identities of the reactant side, stereo-free) with other D0 reactions (planned / pool / training D0G2), with a D1 target,
or with a Phase I D1-design reaction (d1_validation jobs label.json / ts_result.json, ts_id D1-NNNN-rK); and what a
group-aware holdout would cost. Prints JSON."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402
import gen_d0 as D  # noqa: E402


def pair_of(smi):
    s = str(smi).split(">>")
    if len(s) != 2:
        return None
    xs = [D.mol_ident(x) for x in s[0].split(".")]
    return None if None in xs else ".".join(sorted(xs))


S = pd.read_csv(D.D0_SMILES, usecols=["rxn_id", "rxn_smiles"])
smi0 = dict(zip(S.rxn_id.astype(int), S.rxn_smiles))
C = pd.read_csv(G.D0_CLASS_CSV, keep_default_na=False)
planned = sorted(int(r) for r, p in zip(C.d0_id, C.planned) if str(p) == "True")
cls = dict(zip(C.d0_id.astype(int), C.d0_class.astype(str)))
pair0 = {r: pair_of(smi0[r]) for r in planned}
grp = defaultdict(set)
for r, p in pair0.items():
    grp[p].add(r)
labs1 = M.read_json(M.LABELS)
pair1 = Counter(pair_of(d["rxn_smiles"]) for d in labs1)
rx1 = {D.rxn_ident(d["rxn_smiles"]) for d in labs1}
# Phase I D1-design reactions
des, n_meta = [], 0
for j in sorted(p for p in (M.D1VAL / "jobs").iterdir() if p.is_dir()):
    for n in ("label.json", "ts_result.json"):
        f = j / n
        if f.is_file():
            try:
                m = M.read_json(f)
            except Exception:                                                  # noqa: BLE001
                continue
            n_meta += 1
            if re.fullmatch(r"D1-\d+-r\d+", str(m.get("ts_id", ""))) and m.get("rxn_smiles"):
                des.append(dict(job=j.name, ts_id=m["ts_id"], smi=m["rxn_smiles"]))
            break
for d in sorted((M.D1VAL / "v7").glob("*")):
    f = M.D1VAL / "pilot_copy" / "jobs" / d.name / "label.json"
    if f.is_file():
        m = M.read_json(f)
        if re.fullmatch(r"D1-\d+-r\d+", str(m.get("ts_id", ""))) and m.get("rxn_smiles"):
            des.append(dict(job="v7_" + d.name, ts_id=m["ts_id"], smi=m["rxn_smiles"]))
des_ts = sorted({d["ts_id"] for d in des})
des_rx = {D.rxn_ident(d["smi"]) for d in des}
des_pair = {pair_of(d["smi"]) for d in des}
rx0 = {r: D.rxn_ident(smi0[r]) for r in planned}
g2p = M.read_json(G.G2_PLAN_D0)
pool = M.read_json(G.G2_POOL_D0)["ranked"]
pool_ids = {int(u["rxn_id"]) for u in pool}
train = [int(u["rxn_id"]) for u in g2p["rxns"] if u.get("role", "train") == "train"]
hold = [int(x) for x in pd.read_csv(D.HOLD_CSV).d0_id] if D.HOLD_CSV.is_file() else []
tset = set(train)


def sib(r):
    return sorted(grp[pair0[r]] - {r})


hs = {r: sib(r) for r in hold}
out = dict(n_planned=len(planned), n_pair_groups=len(grp),
           group_size=dict(sorted(Counter(len(v) for v in grp.values()).items())),
           n_planned_in_groups_ge2=sum(len(v) for v in grp.values() if len(v) > 1),
           d1_pairs=len(pair1), n_planned_with_d1_pair_sibling=sum(pair0[r] in pair1 for r in planned),
           n_pool_with_d1_pair_sibling=sum(pair0.get(r) in pair1 for r in pool_ids),
           d1design=dict(n_meta=n_meta, n_rows=len(des), ts_ids=des_ts[:40], n_ts_ids=len(des_ts),
                         n_planned_same_reaction=sum(rx0[r] in des_rx for r in planned),
                         planned_same_reaction=sorted(r for r in planned if rx0[r] in des_rx)[:20],
                         n_planned_same_pair=sum(pair0[r] in des_pair for r in planned),
                         n_pool_same_reaction=sum(rx0.get(r) in des_rx for r in pool_ids),
                         n_pool_same_pair=sum(pair0.get(r) in des_pair for r in pool_ids),
                         design_rxn_is_d1_target=sum(x in rx1 for x in des_rx)),
           holdout=dict(n=len(hold), n_with_d0_sibling=sum(bool(v) for v in hs.values()),
                        n_siblings=len({x for v in hs.values() for x in v}),
                        n_siblings_in_training_d0g2=len({x for v in hs.values() for x in v} & tset),
                        n_siblings_also_holdout=len({x for v in hs.values() for x in v} & set(hold)),
                        n_with_d1_pair_sibling=sum(pair0[r] in pair1 for r in hold),
                        n_same_reaction_as_d1design=sum(rx0[r] in des_rx for r in hold),
                        n_same_pair_as_d1design=sum(pair0[r] in des_pair for r in hold)),
           train=dict(n=len(train), n_groups=len({pair0[r] for r in train})))
# group-aware candidates: pool units outside the training groups, no D1 pair sibling, no D1-design pair
tg = {pair0[r] for r in train}
cand = [u for u in pool if int(u["rxn_id"]) not in tset and pair0[int(u["rxn_id"])] not in tg
        and pair0[int(u["rxn_id"])] not in pair1 and pair0[int(u["rxn_id"])] not in des_pair]
cg = defaultdict(set)
for u in cand:
    cg[u["d0_class"]].add(pair0[int(u["rxn_id"])])
pc = Counter(u["d0_class"] for u in pool)
out["group_aware"] = {s: dict(pool=pc[s], cand_units=sum(u["d0_class"] == s for u in cand), cand_groups=len(cg[s]),
                              planned=sum(cls[r] == s for r in planned),
                              planned_groups=len({pair0[r] for r in planned if cls[r] == s}))
                      for s in sorted(pc)}
print(json.dumps(out, indent=1, default=str))
