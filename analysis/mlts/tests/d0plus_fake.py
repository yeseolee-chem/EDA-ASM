"""tests/d0plus_fake.py — sandbox only, TEST ROOTS ONLY (D0-plus / D0-gauss): synthetic label-level records in the test
root so `collect --d0` and finetune.py exercise the holdout code path. Refuses unless $MLT_SCRATCH is under
.../tmp/d0plustest.
  - for the first 2 evaluation reactions and the first sibling of d0_holdout.csv and the first training D0G2 unit of
    g2_plan_d0.json: records of the plan rows d0g1_<rid>_n1 / _n2 / _i and d0g3_<rid>_f1 / _f2 (plan geometry; existing
    records are never overwritten)
  - a D0G2 unit (unit.json status ok + steps.json: 2 g2_path + 1 ts_final_g2 steps around the G2c start) for the
    training unit and for the first evaluation reaction (= a manual `run_g2 --d0 --only` of a holdout reaction: its
    rows must be marked holdout by collect and dropped by finetune)
  - the 3 rows of the first D0 molecule whose representative d0_id is a listed reaction (d0_react_ref / _disp: must be
    KEPT by finetune)
  - $MLT/dft/gen_G4.extxyz: al_ts structures with d0_id of an evaluation reaction and of the sibling (dropped in *_al
    runs) and of the training reaction (kept)
Energies = sum(finetune.ATOM_REF_EH) + 1e-3 Eh x N + noise, gradients ~ N(0, 2e-3) Eh/bohr (seeded); every record has
synthetic=True. Prints the ids written."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402
import finetune as F  # noqa: E402

if "/tmp/d0plustest" not in str(M.MLT):
    raise SystemExit(f"refusing: MLT_SCRATCH {M.MLT} is not a d0plustest root")
rng = np.random.default_rng(7)
Z = {s: z for z, s in enumerate(["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
                                  "S", "Cl", "Ar"])}
Z["Br"] = 35


def energy(syms):
    return float(sum(F.ATOM_REF_EH[Z[s]] for s in syms) + 1e-3 * len(syms) + rng.normal(0, 1e-4))


def grad(n):
    return rng.normal(0, 2e-3, (n, 3)).tolist()


def record(r):
    syms = str(r["symbols"]).split()
    pos = np.asarray(r["pos_A"], float).reshape(-1, 3)
    return dict(struct_id=r["struct_id"], group=r["group"], kind=r["kind"], rxn_ids=r["rxn_ids"], folds=r["folds"],
                dipoles=r["dipoles"], d0_id=int(r["d0_id"]), d0_class=r["d0_class"], charge=int(r["charge"]),
                n_atoms=len(syms), symbols=syms, pos_A=pos.tolist(), energy_eh=energy(syms), grad_eh_bohr=grad(len(syms)),
                status="ok", core_h=0.0, disp={}, synthetic=True, note="tests/d0plus_fake.py")


P = G.load_plan(d0=True)
g2p = M.read_json(G.G2_PLAN_D0)
pool = {int(u["rxn_id"]): u for u in M.read_json(G.G2_POOL_D0)["ranked"]}
H = pd.read_csv(G.PLAN_DIR / "d0_holdout.csv", keep_default_na=False)
ev = [int(x) for x, r in zip(H.d0_id, H.role) if r == "eval"]
sb = [int(x) for x, r in zip(H.d0_id, H.role) if r == "sibling"]
hold = [pool[x] for x in ev[:2]]
sib = sb[:1]
train = list(g2p["rxns"])[:1]
hset = set(ev) | set(sb)
out = dict(holdout=[int(u["rxn_id"]) for u in hold], sibling=sib, train=[int(u["rxn_id"]) for u in train], records=[],
           units=[])
for rid in [int(u["rxn_id"]) for u in hold + train] + sib:
    sids = [f"d0g1_{rid}_n1", f"d0g1_{rid}_n2", f"d0g1_{rid}_i", f"d0g3_{rid}_f1", f"d0g3_{rid}_f2"]
    for r in P[P.struct_id.isin(sids)].to_dict("records"):
        p = G.rec_path(r["group"], r["struct_id"])
        if not p.is_file():
            M.write_json(p, record(r))
            out["records"].append(r["struct_id"])
for u in hold[:1] + train:
    rid = int(u["rxn_id"])
    ud = G.RUNS / "D0G2" / str(rid)
    if not (ud / "unit.json").is_file():
        syms, x0 = M.read_xyz(u["start"])
        x0 = np.asarray(x0, float)
        steps = [dict(cycle=k + 1, kind="ts_final_g2" if k == 2 else "g2_path", energy_eh=energy(syms),
                      pos_A=(x0 + rng.normal(0, 0.02, x0.shape) * (2 - k) / 2).tolist(), grad_eh_bohr=grad(len(syms)))
                 for k in range(3)]
        M.write_json(ud / "steps.json", dict(rxn_id=rid, symbols=list(syms), n_steps=3, steps=steps))
        M.write_json(ud / "unit.json", dict(rxn_id=rid, struct_id=u["struct_id"], d0_id=rid, d0_class=u["d0_class"],
                                            run_group="D0G2", status="ok", core_h=0.0, n_steps=3, n_atoms=len(syms),
                                            converged_to_label=None, synthetic=True, note="tests/d0plus_fake.py"))
        out["units"].append(rid)
mol = P[(P.kind == "d0_react_ref") & P.d0_id.isin(sorted(hset))]
if len(mol):
    mid = mol.mol_id.iloc[0]
    for r in P[P.mol_id == mid].to_dict("records"):
        p = G.rec_path(r["group"], r["struct_id"])
        if not p.is_file():
            M.write_json(p, record(r))
            out["records"].append(r["struct_id"])
    out["molecule"] = dict(mol_id=mid, rep_d0_id=int(mol.d0_id.iloc[0]))
al = []
sib_row = P[(P.d0_id == sib[0]) & (P.kind == "d0_ts_near")].iloc[0] if sib else None
for tag, rid, syms, x0, dips, q in (
        [("hold", int(hold[0]["rxn_id"]), *M.read_xyz(hold[0]["start"]), hold[0]["dipoles"], int(hold[0]["charge"])),
         ("train", int(train[0]["rxn_id"]), *M.read_xyz(train[0]["start"]), train[0]["dipoles"], int(train[0]["charge"]))]
        + ([("sib", sib[0], str(sib_row.symbols).split(), np.asarray(sib_row.pos_A, float).reshape(-1, 3),
             sib_row.dipoles, int(sib_row.charge))] if sib else [])):
    x = np.asarray(x0, float) + rng.normal(0, 0.03, (len(syms), 3))
    al.append(M.ext_atoms(syms, x, energy(syms), grad(len(syms)), struct_id=f"g4_test_{tag}_{rid}", kind="al_ts",
                          source="test_g4", rxn_ids="", folds="", dipoles=dips, level="label", group="G4", d0_id=rid,
                          charge=q))
M.write_extxyz(G.DFT / "gen_G4.extxyz", al)
out["g4"] = [a.info["struct_id"] for a in al]
print(json.dumps(out, indent=1))
