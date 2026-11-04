#!/usr/bin/env python3
"""tests/a_api.py — the contract API mlts_autode.run_one(rxn_smiles, job_id, model, engine, out_root, cores) called
from the mlts env (no socket: run_one starts its own server; the reaction runs as `one` in the D1 env). Stand-in model:
zero-shot MACE-OMOL-0. Writes only under $MLT/tmp/atest/api. Run:
  MLTS_A_BASE=$MLT/tmp/atest SBX_GPU=1 SBX_C=3 SBX_MEM=16G bash jobs/sbx.sh mlts tests/a_api.py [job_id]"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLTS_A_BASE", "/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/atest")
import mlt_common as M  # noqa: E402
import mlts_autode as A  # noqa: E402

job = sys.argv[1] if len(sys.argv) > 1 else "T00851"
row = A.manifest_rows([job])[job]
out_root = Path(os.environ["MLTS_A_BASE"]) / "api" / "ade"
t = time.time()
p = A.run_one(row["rxn_smiles"], job, "MACE-OMOL-0", "A", out_root=out_root, cores=2, tag="atest_api")
st = A.stage_state(p)
need = ["ts.xyz", "ts_result.json", "ts.hess", "ref_dipole.xyz", "ref_dipolarophile.xyz", "ref_result.json",
        "stereo_result.json", "timing.json", "inputs.tar.gz", ".done_ts", ".done_stereo", ".done_ref"]
miss = [f for f in need if not (p / f).is_file()]
tim = json.loads((p / "timing.json").read_text())
rep = dict(path=str(p), expected=str(out_root / "jobs" / job), state=st, missing=miss, wall_s=time.time() - t,
           timing={k: tim.get(k) for k in ("wall_s", "core_h", "gpu_s", "n_ext_calls", "n_hess")},
           ok=(p == out_root / "jobs" / job and st[0] == "ok" and not miss))
M.write_json(Path(os.environ["MLTS_A_BASE"]) / "reports" / "a_api.json", rep)
print(json.dumps(rep), flush=True)
sys.exit(0 if rep["ok"] else 1)
