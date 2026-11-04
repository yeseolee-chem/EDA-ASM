"""tests/d0plus_setup.py — sandbox only, TEST ROOTS ONLY (D0-plus): fill the test root $MLT_SCRATCH (must be under
/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/d0plustest/, $MLT_RESULTS too) with COPIES of production files (production is only
read). argv[1]:
  d0  $MLT/dft/plan/{plan_d0.parquet, g2_plan_d0.json, g2_pool_d0.json, d0_class.csv, plan_d0_meta.json},
      $MLT_RESULTS/{G_plan_d0.json, G0_estimate_d0.json, G0_estimate.json} (repo results), and every D0G1 / D0G3
      record json on disk now ($MLT/dft/runs/D0G1|D0G3/*.json), so the test plan sees 'D0 records exist' as production
  d1  $MLT/dft/plan/{plan.parquet, plan_meta.json, g2_plan.json}, $MLT_RESULTS/{G_plan.json, G0_estimate.json}, the
      first 40 G1 + 20 G3 record jsons (plain `collect` smoke)
Optional argv[2] (d0 mode) = at most this many record jsons per group (first by name; default all).
Existing destination files are not overwritten. Prints a JSON manifest (sources with sha256)."""
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/d0plustest")
PROD = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
REPO_RES = Path(__file__).resolve().parent.parent / "results"
for root in (M.MLT, M.RES):
    try:
        Path(root).resolve().relative_to(BASE)
    except ValueError:
        raise SystemExit(f"refusing: {root} is not under {BASE}")
if Path(M.MLT).resolve() == PROD.resolve():
    raise SystemExit("refusing: production root")
mode = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
pairs = []
if mode == "d0":
    for f in ("plan_d0.parquet", "g2_plan_d0.json", "g2_pool_d0.json", "d0_class.csv", "plan_d0_meta.json"):
        pairs.append((PROD / "dft" / "plan" / f, M.MLT / "dft" / "plan" / f))
    for f in ("G_plan_d0.json", "G0_estimate_d0.json", "G0_estimate.json"):
        pairs.append((REPO_RES / f, M.RES / f))
    for g in ("D0G1", "D0G3"):
        for p in sorted((PROD / "dft" / "runs" / g).glob("*.json"))[:lim]:
            pairs.append((p, M.MLT / "dft" / "runs" / g / p.name))
elif mode == "d1":
    for f in ("plan.parquet", "plan_meta.json", "g2_plan.json"):
        pairs.append((PROD / "dft" / "plan" / f, M.MLT / "dft" / "plan" / f))
    for f in ("G_plan.json", "G0_estimate.json"):
        pairs.append((REPO_RES / f, M.RES / f))
    for g, k in (("G1", 40), ("G3", 20)):
        for p in sorted((PROD / "dft" / "runs" / g).glob("*.json"))[:k]:
            pairs.append((p, M.MLT / "dft" / "runs" / g / p.name))
else:
    raise SystemExit("mode d0 | d1")
out = []
for src, dst in pairs:
    M.assert_write_ok(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    new = not dst.exists()
    if new:
        tmp = dst.with_name(f".{dst.name}.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    out.append(dict(src=str(src), dst=str(dst), copied=new, sha256=M.sha256(dst)))
print(json.dumps(dict(mode=mode, mlt=str(M.MLT), res=str(M.RES), n=len(out), n_copied=sum(x["copied"] for x in out),
                      files=[x for x in out if not x["dst"].endswith(".json") or "/runs/" not in x["dst"]],
                      n_records=sum("/runs/" in x["dst"] for x in out)), indent=1))
