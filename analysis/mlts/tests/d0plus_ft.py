"""tests/d0plus_ft.py — sandbox only; writes only under a d0plustest root (D0-plus finetune checks). Input I = the
production $MLT/dft/existing.extxyz (read only, in-process override, as tests/d0_ft_dry.py --with-real-I); D0G1-3 and
G4 come from the test root (collect --d0 on synthetic + copied records, tests/d0plus_fake.py). argv[1]:
  dry    `finetune.py data --dry FT_0 FT_all_s0 FT_LODO_1 FTpilot_MACE-OMOL-0_0 FT_0_al FT_all_s0_al` (nothing written)
  neg    negative control: the holdout drop disabled in-process (Pool._drop_holdout -> no-op), `data --dry FT_0`
         must STOP in the C-2 check (exit 3); prints the exit code
  neg_sid  negative control for the independent C-2 check: the info-based D0 reaction lookup disabled in-process
         (finetune.d0id_of -> -1, so nothing is dropped); the struct-id check must STOP (exit 3)
  neg_g4   a G4 row without d0_id and without D1 rxn_ids added ($MLT/dft/gen_G4_neg.extxyz); `data --dry FT_0_al` must
         STOP (exit 3)
  al_dup   a G4 exact duplicate (d0_id of an evaluation reaction) of a kept base D0G1 row ($MLT/dft/gen_G4_dup.extxyz);
         `data --dry FT_0 FT_0_al`: the base row stays in FT_0 and leaves FT_0_al (hold_al)
  build  a real `finetune.py data FT_0` on the test root ($MLT/ft/data/fold0, $MLT_RESULTS/ft/FT_0_data.json) and the
         d0_holdout record of the data json"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import mlt_common as M  # noqa: E402
import finetune as F  # noqa: E402

if "/tmp/d0plustest" not in str(M.MLT) or "/tmp/d0plustest" not in str(M.RES):
    raise SystemExit("refusing: not a d0plustest root")
REAL_I = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/existing.extxyz")
F.INPUTS = (("I", REAL_I),) + tuple(x for x in F.INPUTS if x[0] != "I")
mode = sys.argv[1]
if mode == "dry":
    sys.exit(F.main(["data", "--dry", "FT_0", "FT_all_s0", "FT_LODO_1", "FTpilot_MACE-OMOL-0_0", "FT_0_al",
                     "FT_all_s0_al"]))
if mode == "neg":
    F.Pool._drop_holdout = lambda self, rows: rows
    try:
        rc = F.main(["data", "--dry", "FT_0"])
    except SystemExit as e:
        rc = e.code
    print(json.dumps(dict(negative_control="holdout drop disabled", exit=rc, expected=3, ok=rc == 3)))
    sys.exit(0 if rc == 3 else 1)
if mode == "neg_sid":
    F.d0id_of = lambda a, inp: -1
    try:
        rc = F.main(["data", "--dry", "FT_0"])
    except SystemExit as e:
        rc = e.code
    print(json.dumps(dict(negative_control="info d0_id lookup disabled (struct-id check only)", exit=rc, expected=3,
                          ok=rc == 3)))
    sys.exit(0 if rc == 3 else 1)
if mode in ("neg_g4", "al_dup"):
    import pandas as pd  # noqa: E402
    H = pd.read_csv(F.D0_HOLDOUT, keep_default_na=False)
    hset = {int(x) for x in H.d0_id}
    ev0 = int(next(x for x, r in zip(H.d0_id, H.role) if r == "eval"))
    al = M.read_extxyz(F.AL_INPUT[1])
    base = next(a for a in M.read_extxyz(F.DFT / "gen_D0G1.extxyz") if int(a.info["d0_id"]) not in hset)
    syms, f = base.get_chemical_symbols(), np.asarray(base.arrays["REF_forces"], float) / M.EHBOHR2EVA
    if mode == "neg_g4":
        x = base.positions + np.random.default_rng(3).normal(0, 0.05, base.positions.shape)
        al.append(M.ext_atoms(syms, x, float(base.info["e_eh"]), (-f).tolist(), struct_id="g4_test_noid", kind="al_ts",
                              source="test_g4", rxn_ids="", folds="", dipoles="", level="label", group="G4"))
        p = F.DFT / "gen_G4_neg.extxyz"
    else:
        al.append(M.ext_atoms(syms, base.positions, float(base.info["e_eh"]), (-f).tolist(), struct_id="g4_test_dup",
                              kind="al_ts", source="test_g4", rxn_ids="", folds="", dipoles=base.info.get("dipoles"),
                              level="label", group="G4", d0_id=ev0, charge=int(base.info.get("charge", 0) or 0)))
        p = F.DFT / "gen_G4_dup.extxyz"
    M.write_extxyz(p, al)
    F.AL_INPUT = ("G4", p)
    runs = ["FT_0_al"] if mode == "neg_g4" else ["FT_0", "FT_0_al"]
    try:
        rc = F.main(["data", "--dry"] + runs)
    except SystemExit as e:
        rc = e.code
    want = 3 if mode == "neg_g4" else 0
    print(json.dumps(dict(case=mode, base_row=base.info["struct_id"], exit=rc, expected=want, ok=rc == want)))
    sys.exit(0 if rc == want else 1)
if mode == "build":
    rc = F.main(["data", "FT_0"])
    j = M.read_json(M.RES / "ft" / "FT_0_data.json")
    print(json.dumps(dict(exit=rc, data_json=str(M.RES / "ft" / "FT_0_data.json"), d0_holdout=j.get("d0_holdout"),
                          d0_holdout_sha256=j.get("d0_holdout_sha256"),
                          inputs_D0holdout=j["inputs"].get("D0holdout"),
                          leak={k: j["leakage_C2"].get(k) for k in ("pass_", "kept_holdout_ts_rows",
                                                                     "kept_holdout_molecule_rows", "d0_holdout_sha256")},
                          counts={k: v["n"] for k, v in j["counts"].items()}), indent=1))
    sys.exit(rc)
raise SystemExit("mode dry | neg | neg_sid | neg_g4 | al_dup | build")
