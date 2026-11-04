"""tests/d0_ikey_check.py — sandbox only, READ ONLY: planned D0 molecules (argv[1] root plan) that match a D1 molecule
by the InChIKey first block (connectivity only: also ignores charge / protonation) but not by the gen_d0 identity
(full stereo-free standard InChI) — i.e. what a looser key would add. Prints JSON."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
import mlt_common as M  # noqa: E402
import gen_d0 as D  # noqa: E402
from rdkit import Chem  # noqa: E402


def ikey(x):
    try:
        return Chem.InchiToInchiKey(x).split("-")[0] if x and x.startswith("InChI=") else None
    except Exception:  # noqa: BLE001
        return None


P = pd.read_parquet(Path(sys.argv[1]) / "dft/plan/plan_d0.parquet", columns=["group", "d0_id"])
planned = sorted(set(P.loc[P.group == "D0G1", "d0_id"].astype(int)))
Qk = {int(d["rxn_id"]): (d.get("cr") or {}) for d in M.read_json(D.D0_Q)}
d1 = defaultdict(set)
for d in M.read_json(M.LABELS):
    for k in (1, 2):
        d1[D.mol_ident(D.smi_part(d["cr"][f"key{k}"]))].add(int(d["rxn_id"]))
d1k = defaultdict(set)
for x in d1:
    d1k[ikey(x)].add(x)
hits = {}
for r in planned:
    for k in (1, 2):
        s = D.smi_part(Qk[r][f"key{k}"])
        x = D.mol_ident(s)
        if x in d1:
            continue
        h = d1k.get(ikey(x))
        if h:
            e = hits.setdefault(s, dict(d0_inchi=x, d1_inchi=sorted(h)[:2], n_occ=0, rxns=[]))
            e["n_occ"] += 1
            e["rxns"] = sorted(set(e["rxns"]) | {r})[:6]
print(json.dumps(dict(n_planned=len(planned), n_ikey_only=len(hits), hits=hits), indent=1))
