#!/usr/bin/env python3
"""tests/integ_kcons.py — Integrate step: the K_report.json consumers (review EWPS 1 / 2 / 5, CRK 3) read the train_k
test report ($MLT/tmp/ktest/results/K_report.json, rewritten by `MLT_K_TEST_ROOT=$MLT/tmp/ktest train_k.py select`):
eval_v2._cfg_list -> existing prediction files + refs; final_w.train_spec / predict_v2.fold_unit_files -> the unit
files train_k wrote ($MLT/tmp/ktest/units/<config>/L__<rung>__...). Read only. reactot env, CPU:
  bash jobs/sbx.sh reactot tests/integ_kcons.py   (or as one step of tests/integ_jobs.py)"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import eval_v2 as EV  # noqa: E402
import final_w as FW  # noqa: E402
import predict_v2 as PV  # noqa: E402

KT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/ktest")
OK = []


def check(c, what):
    OK.append(bool(c))
    print(("PASS " if c else "FAIL ") + what, flush=True)


K = json.loads((KT / "results" / "K_report.json").read_text())
cfgs = EV._cfg_list(K)
check({"C2", "C3"} <= set(cfgs) and not any(n.startswith("C2_s") for n in cfgs),
      f"configs evaluated: {sorted(cfgs)} (seed sets only under seed_sets)")
for n in ("C2", "C3"):
    c = cfgs[n]
    check(Path(c["predictions"]).is_file() and c["refs"] in ("own", "min") and c["table"],
          f"{n}: predictions a file path ({Path(c['predictions']).name}), refs {c['refs']}, table set")
    rung, rung_u, ud = FW.train_spec(n, dict(K["configs"][n]))
    j = Path(ud) / f"L__{rung_u}__SH__TF__f0.json"
    check(j.is_file() and rung == ("mlts_mix" if n == "C3" else "mlts"), f"{n}: final_w unit json {j} (rung {rung})")
    base, pts = PV.fold_unit_files(n, dict(K["configs"][n]), "SH", "TF", 0)
    check(Path(str(base) + ".json").is_file() and Path(pts[0]).is_file(), f"{n}: predict_v2 fold unit {base.name}")
check(K.get("primary") in cfgs and isinstance(K["configs"][K["primary"]].get("predictions"), str),
      f"primary {K.get('primary')}: predictions recorded as a path string")
print(f"{sum(OK)}/{len(OK)} checks pass")
sys.exit(0 if all(OK) else 1)
