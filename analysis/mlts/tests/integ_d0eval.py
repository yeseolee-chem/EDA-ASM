#!/usr/bin/env python3
"""tests/integ_d0eval.py — Integrate step: `ts_geom.py d0_eval` (decision D0-eval) on a test root with a fake
d0_holdout_ref.json (3 small neutral D0 pool reactions + 1 charged one; the real list is written by
`gen_d0.py plan_d0 --n-holdout 100`, not yet run). Stand-in model: $MLT/ft/BENCH_MACE-OMOL-0_b16 (no FT_all model
exists yet). Compute nodes only:  SBX_GPU=1 bash jobs/sbx.sh mlts tests/integ_d0eval.py
Writes only below /gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/integtest/d0eval."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/integtest/d0eval")
REAL = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
STAND = str(REAL / "ft" / "BENCH_MACE-OMOL-0_b16" / "BENCH_MACE-OMOL-0_b16.model")
HERE = Path(__file__).resolve().parents[1]
OK = []


def check(c, what):
    OK.append(bool(c))
    print(("PASS " if c else "FAIL ") + what, flush=True)


def tg(*a):
    r = subprocess.run([sys.executable, str(HERE / "ts_geom.py"), "d0_eval", *a], capture_output=True, text=True)
    out = "\n".join(x for x in (r.stdout + r.stderr).splitlines() if not x.startswith("[mlts_bridge]"))
    print(f"$ ts_geom.py d0_eval {' '.join(a)[:100]} -> rc {r.returncode}\n{out[-1500:]}", flush=True)
    return r.returncode


def main():
    assert str(ROOT).startswith(str(REAL / "tmp"))
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    pool = json.loads((REAL / "dft" / "plan" / "g2_pool_d0.json").read_text())["ranked"]
    neu = sorted((u for u in pool if int(u["charge"]) == 0), key=lambda u: int(u["n_atoms"]))[:3]
    chg = next(u for u in pool if int(u["charge"]) != 0)
    units = [dict(d0_id=int(u["d0_id"]), d0_class=u["d0_class"], n_atoms=int(u["n_atoms"]), charge=int(u["charge"]),
                  pairs=u["pairs"], pool_rank=int(u["rank"]), reference=dict(kind="coley_g16_ts", xyz=u["label_ts"]),
                  starts=dict(xtb_g2c=u["start"], reference=u["label_ts"])) for u in neu + [chg]]
    ref = ROOT / "d0_holdout_ref.json"
    ref.write_text(json.dumps(dict(holdout_sha256="test", n=len(units), units=units)))
    com = ["--models", f"{STAND},{STAND}", "--root", str(ROOT / "d0eval"), "--ref", str(ref)]
    out = ROOT / "D0_eval_report.json"
    rc0 = tg("report", *com, "--out", str(out))
    R0 = json.loads(out.read_text())
    check(rc0 == 2 and R0["complete"] is False and R0["n_units"] == 4, "report before any run: incomplete (exit 2)")
    rc1 = tg("run", "0", "2", *com, "--workers", "3")          # task 0 of 2 = the first model (8 items: 2 starts x 4)
    rc2 = tg("run", "1", "2", *com, "--workers", "3")
    rc3 = tg("report", *com, "--out", str(out))
    R = json.loads(out.read_text())
    m = R["per_model"].get(STAND) or {}
    check(rc1 == 0 and rc2 == 0 and rc3 == 0 and R["complete"], f"two run tasks + report: complete ({rc1}, {rc2}, {rc3})")
    lab, g2 = m.get("label") or {}, m.get("g2") or {}
    check(lab.get("n_records") == 4 and g2.get("n_records") == 4 and R["n_charged_not_covered"] == 1
          and (lab.get("status_counts") or {}).get("not_covered") == 1,
          f"4 records per start, the charged reaction not covered ({lab.get('status_counts')}, {g2.get('status_counts')})")
    check(lab.get("n_attempted") == 3 and lab.get("dd_median_A") is not None and "label" in R["pooled"]
          and R["floor_rmsd_A"] == 0.009 and "by_class" in R["pooled"]["label"],
          f"FT-2 table on the Coley reference: |dd| median {lab.get('dd_median_A')}, same TS {lab.get('same_ts_frac')}, "
          f"g2 start |dd| median {g2.get('dd_median_A')}")
    rec = json.loads(next((ROOT / "d0eval" / Path(STAND).name / "g2" / "recs").glob("*.json")).read_text())
    check(rec.get("tight") == "tight" and rec.get("reference") in ("coley_g16_ts", None) and rec.get("set") == "d0",
          f"records: TightOpt, set d0 ({rec.get('status')}, {rec.get('start_kind')})")
    rc4 = tg("run", "0", "2", *com)
    check(rc4 == 0, "rerun skips every recorded item (exit 0)")
    print(f"{sum(OK)}/{len(OK)} checks pass")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
