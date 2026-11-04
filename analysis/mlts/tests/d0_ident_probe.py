"""tests/d0_ident_probe.py — sandbox only, READ ONLY (prints JSON, writes nothing): resonance-invariant identity checks
for the D0 extension (review defect 1 / 2).
  - RDKit standard InChI on the reviewer's resonance pair
  - D0 Q-key molecules vs D1 cr-key molecules: raw SMILES-part string vs stereo-free standard InChI
  - D0 ok reactions vs D1 targets: gen_d0.canon_rxn string vs InChI reaction key
  - the 24 d1_validation D0 reactions: reaction + reactant-molecule identity vs D1 (Phase I tag rule check)
  - Phase I existing.extxyz rows: group patterns x kind x tag state (the D1-design validation rows)"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402
import gen_d0 as D  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
out = {}


def inchi(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(m)
    try:
        x = Chem.MolToInchi(m)
    except Exception:  # noqa: BLE001
        return None
    return x or None


def rxn_key(smi):
    sides = []
    for side in smi.split(">>"):
        xs = [inchi(s) for s in side.split(".")]
        if None in xs:
            return None
        sides.append(".".join(sorted(xs)))
    return ">>".join(sides)


out["pair"] = [inchi("C=[N+](C)[CH-]C(C)=O"), inchi("[CH2-][N+](C)=CC(C)=O")]
T = M.targets().set_index("rxn_id")
labs1 = M.read_json(M.LABELS)
d1smi = defaultdict(set)
for d in labs1:
    for k in (1, 2):
        d1smi[D.smi_part(d["cr"][f"key{k}"])].add(int(d["rxn_id"]))
d1id = defaultdict(set)
nfail1 = 0
for s, rx in d1smi.items():
    x = inchi(s)
    if x is None:
        nfail1 += 1
        continue
    d1id[x] |= rx
L = {int(d["rxn_id"]): d for d in M.read_json(D.D0_LABELS)}
ok = sorted(r for r, d in L.items() if d.get("status") == "ok")
Qk = {int(d["rxn_id"]): (d.get("cr") or {}) for d in M.read_json(D.D0_Q)}
d0smi = defaultdict(list)
for r in ok:
    for k in (1, 2):
        d0smi[D.smi_part(Qk[r][f"key{k}"])].append((r, k))
raw = {s for s in d0smi if s in d1smi}
idhit, nfail0, idonly = set(), 0, []
for s, occ in d0smi.items():
    x = inchi(s)
    if x is None:
        nfail0 += 1
        continue
    if x in d1id:
        idhit.add(s)
        if s not in raw:
            rx = sorted(d1id[x])
            idonly.append(dict(d0_smi=s, n_occ=len(occ), d0_rxns=sorted({o[0] for o in occ})[:10], n_d1_rxns=len(rx),
                               d1_smi=sorted(t for t in d1smi if inchi(t) == x)[:3],
                               folds=sorted({int(T.at[i, "fold_T1"]) for i in rx})))
out["molecules"] = dict(n_d1_smiles=len(d1smi), n_d1_inchi=len(d1id), inchi_fail_d1=nfail1, n_d0_smiles=len(d0smi),
                        inchi_fail_d0=nfail0, shared_raw=len(raw), shared_inchi=len(idhit),
                        raw_not_inchi=sorted(raw - idhit)[:10], inchi_only=idonly)
S = pd.read_csv(D.D0_SMILES, usecols=["rxn_id", "rxn_smiles"])
smi0 = dict(zip(S.rxn_id.astype(int), S.rxn_smiles))
c1, k1 = defaultdict(set), defaultdict(set)
for d in labs1:
    c1[D.canon_rxn(d["rxn_smiles"])].add(int(d["rxn_id"]))
    kk = rxn_key(d["rxn_smiles"])
    if kk:
        k1[kk].add(int(d["rxn_id"]))
cs, ks, kfail = {}, {}, 0
for r in ok:
    c = D.canon_rxn(smi0[r])
    if c in c1:
        cs[r] = sorted(c1[c])
    kk = rxn_key(smi0[r])
    if kk is None:
        kfail += 1
    elif kk in k1:
        ks[r] = sorted(k1[kk])
out["reactions"] = dict(canon=sorted(cs), inchi=sorted(ks), inchi_only={str(r): ks[r] for r in ks if r not in cs},
                        canon_only=sorted(set(cs) - set(ks)), inchi_fail=kfail,
                        n_d1_rxn_keys=len(k1), d1_rxn_key_fail=len(labs1) - sum(len(v) for v in k1.values()))
val = D.d1val_ids()
vr = {}
for r in val:
    mols = []
    for k in (1, 2):
        s = D.smi_part(Qk[r][f"key{k}"])
        x = inchi(s)
        mols.append(dict(k=k, raw_d1=len(d1smi.get(s, ())), inchi_d1=len(d1id.get(x, ())) if x else None))
    vr[str(r)] = dict(rxn_inchi_d1=ks.get(r, []), rxn_canon_d1=cs.get(r, []), mols=mols)
out["d1val"] = dict(ids=val, mismatch={k: v for k, v in vr.items() if v["rxn_inchi_d1"] != v["rxn_canon_d1"]
                                       or any(m["raw_d1"] != m["inchi_d1"] for m in v["mols"])},
                    n_rxn_match=sum(bool(v["rxn_inchi_d1"]) for v in vr.values()),
                    n_mol_match=sum(bool(m["inchi_d1"]) for v in vr.values() for m in v["mols"]))
al = M.read_extxyz("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/existing.extxyz")
c = Counter()
ex = defaultdict(set)
for a in al:
    g = str(a.info.get("group", ""))
    pat = re.sub(r"\d+", "#", g)
    st = "rx" if a.info.get("rxn_ids") else ("dip" if a.info.get("dipoles") else "none")
    c[f"{pat}|{a.info.get('kind')}|{st}"] += 1
    if g.startswith("D1-"):
        ex["D1-"].add(g)
out["phase_i"] = dict(n=len(al), by=dict(sorted(c.items())), d1_groups=sorted(ex["D1-"]),
                      info_keys=sorted(al[0].info.keys()))
print(json.dumps(out, indent=1, default=str))
