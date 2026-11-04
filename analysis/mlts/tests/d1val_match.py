"""tests/d1val_match.py — sandbox only, report: d1_validation jobs whose ts_id starts with 'D1-' (Phase I rows with
kinds d0_* and group D1-NNNN-r*) against the 9,266 D1 targets with the stereo-free canonical reaction rule
(gen_d0.canon_rxn) and with stereo. Prints JSON; writes nothing."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402
import gen_d0 as D  # noqa: E402

T = M.targets().set_index("rxn_id")
c0, c1 = {}, {}
for d in M.read_json(M.LABELS):
    for st, C in ((False, c0), (True, c1)):
        c = D.canon_rxn(d["rxn_smiles"], stereo=st)
        if c:
            C.setdefault(c, []).append(int(d["rxn_id"]))
out = {}
for j in sorted((M.D1VAL / "jobs").iterdir()):
    p = j / "label.json"
    if not p.is_file():
        continue
    m = json.loads(p.read_text())
    tid = str(m.get("ts_id") or "")
    if not tid.startswith("D1-") or not m.get("rxn_smiles"):
        continue
    a, b = D.canon_rxn(m["rxn_smiles"]), D.canon_rxn(m["rxn_smiles"], stereo=True)
    hit, hit_s = c0.get(a, []), c1.get(b, [])
    out.setdefault(tid, dict(jobs=[], stereo_free=hit, isomeric=hit_s,
                             folds=sorted({int(T.at[x, "fold_T1"]) for x in hit}),
                             dipoles=sorted({str(T.at[x, "dipole"]) for x in hit})))["jobs"].append(j.name)
print(json.dumps(dict(n=len(out), n_stereo_free_match=sum(bool(v["stereo_free"]) for v in out.values()),
                      n_isomeric_match=sum(bool(v["isomeric"]) for v in out.values()), by_ts_id=out), indent=1))
