"""tests/d0_fake_g2.py — sandbox only, TEST ROOTS ONLY: write a synthetic D0G2 unit (runs/D0G2/<rid>/unit.json +
steps.json, status 'test_synthetic') from the real D0G1 records of <rid> (d0g1_<rid>_n1 as a path step, d0g1_<rid>_lab
as the final step) so `collect --d0` and finetune exercise the D0G2 code path. Refuses to write under the production
$MLT (MLT_SCRATCH must contain '/tmp/')."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402

rid = int(sys.argv[1])
if "/tmp/" not in str(M.MLT):
    raise SystemExit(f"refusing: MLT_SCRATCH {M.MLT} is not a test root")
recs = [G._rj(G.RUNS / "D0G1" / f"d0g1_{rid}_{s}.json") for s in ("n1", "lab")]
if any(r is None or r.get("status") != "ok" for r in recs):
    raise SystemExit("need ok records d0g1_<rid>_n1 and d0g1_<rid>_lab")
u = next(x for x in M.read_json(G.G2_POOL_D0)["ranked"] if int(x["rxn_id"]) == rid)
steps = [dict(cycle=k + 1, kind=("ts_final_g2" if k == 1 else "g2_path"), energy_eh=r["energy_eh"], pos_A=r["pos_A"],
              grad_eh_bohr=r["grad_eh_bohr"]) for k, r in enumerate(recs)]
ud = G.RUNS / "D0G2" / str(rid)
M.write_json(ud / "steps.json", dict(rxn_id=rid, symbols=recs[0]["symbols"], n_steps=2, steps=steps))
M.write_json(ud / "unit.json", dict(rxn_id=rid, struct_id=u["struct_id"], d0_id=rid, d0_class=u["d0_class"],
                                    run_group="D0G2", status="test_synthetic", core_h=0.0, n_steps=2, n_atoms=u["n_atoms"],
                                    converged_to_label=None, note="synthetic test unit (tests/d0_fake_g2.py)"))
print("wrote", ud)
