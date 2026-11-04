"""tests/d0_fake_start.py — sandbox only, TEST ROOTS ONLY: write runs/D0G2/<rid>/starts.json (no unit.json) for the
first selected D0G2 unit of g2_plan_d0.json, i.e. an OptTS that looks in flight, so the plan_d0 re-plan guard can be
exercised; with --clear, remove that file again. Refuses to write under the production $MLT."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402

if "/tmp/" not in str(M.MLT):
    raise SystemExit(f"refusing: MLT_SCRATCH {M.MLT} is not a test root")
sel = M.read_json(G.G2_PLAN_D0)["rxns"]
if not sel:
    raise SystemExit("g2_plan_d0.json selects no units")
rid = int(sel[0]["rxn_id"])
p = G.RUNS / "D0G2" / str(rid) / "starts.json"
if "--clear" in sys.argv:
    M.assert_write_ok(p)
    p.unlink(missing_ok=True)
    print("removed", p)
else:
    M.write_json(p, [dict(start=G._now(), job="test_fake_start")])
    print("wrote", p)
