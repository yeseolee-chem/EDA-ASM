#!/usr/bin/env python3
"""tests/integ_g4d0.py — Integrate step: the G-4 D0 rows of g4_al.py (decision G4-d0) on a test root. Compute nodes
only, one step per call (each in its own env):
  bash jobs/sbx.sh reactot tests/integ_g4d0.py setup     test inputs: copies of the D0 pool / plans + a fake holdout
  bash jobs/sbx.sh reactot tests/integ_g4d0.py plan      g4_plan (D1 + D0 rows) + the holdout guards (exit 2 / 3)
  SBX_GPU=1 bash jobs/sbx.sh mlts tests/integ_g4d0.py opt    one D0 row: opt_model (stand-in MLTS) -> record
  bash jobs/sbx.sh reactot tests/integ_g4d0.py run       run G4 --only d0_<id> (label-level EnGrad)
  bash jobs/sbx.sh reactot tests/integ_g4d0.py collect   collect on a plan reduced to that row: extxyz info d0_id
Stand-in model (no fine-tuned MLTS exists yet): $MLT/ft/BENCH_MACE-OMOL-0_b16 for all five entries.
Writes only below /gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/integtest/g4."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/integtest/g4")
SRC, TMLT, TRES = ROOT / "src", ROOT / "mlt", ROOT / "res"
REAL = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
STAND = str(REAL / "ft" / "BENCH_MACE-OMOL-0_b16" / "BENCH_MACE-OMOL-0_b16.model")
MODELS = ",".join([STAND] * 5)
MAXA = 22                                     # small reactions only (30 neutral D0 pool reactions <= 22 atoms)
HERE = Path(__file__).resolve().parents[1]
os.environ.update(MLT_SCRATCH=str(TMLT), MLT_RESULTS=str(TRES), MLT_G4_SRC=str(SRC), MLT_G4_SRC_RES=str(TRES))
OK = []


def check(c, what):
    OK.append(bool(c))
    print(("PASS " if c else "FAIL ") + what, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def g4(*a):
    r = subprocess.run([sys.executable, str(HERE / "g4_al.py"), *a], capture_output=True, text=True, env=os.environ)
    print(f"$ g4_al.py {' '.join(a)[:120]} -> rc {r.returncode}\n{(r.stdout + r.stderr)[-1500:]}", flush=True)
    return r.returncode


def setup():
    assert str(ROOT).startswith("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/")
    shutil.rmtree(ROOT, ignore_errors=True)
    pl = SRC / "dft" / "plan"
    pl.mkdir(parents=True)
    for p in (TMLT, TRES / "ft"):
        p.mkdir(parents=True)
    for f in ("g2_pool_d0.json", "g2_plan.json"):
        shutil.copy(REAL / "dft" / "plan" / f, pl / f)
    shutil.copy(HERE / "results" / "G0_estimate.json", TRES / "G0_estimate.json")     # the G-0 EnGrad cost fit
    pool = json.loads((pl / "g2_pool_d0.json").read_text())["ranked"]
    small = [u for u in pool if int(u["charge"]) == 0 and int(u["n_atoms"]) <= MAXA]
    train, ev, sib = small[:3], small[3:6], small[6:8]
    with open(pl / "d0_holdout.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["d0_id", "d0_class", "n_atoms", "reason", "role"])
        for u in ev:
            w.writerow([u["d0_id"], u["d0_class"], u["n_atoms"], "test eval", "eval"])
        for u in sib:
            w.writerow([u["d0_id"], u["d0_class"], u["n_atoms"], "test sibling", "sibling"])
    plan = dict(rxns=[dict(u, role="train") for u in train],
                holdout=dict(sha256=sha(pl / "d0_holdout.csv"), ids=[int(u["d0_id"]) for u in ev],
                             sibling_ids=[int(u["d0_id"]) for u in sib]))
    (pl / "g2_plan_d0.json").write_text(json.dumps(plan))
    (ROOT / "setup.json").write_text(json.dumps(dict(n_small=len(small), train=[int(u["d0_id"]) for u in train],
                                                     eval=[int(u["d0_id"]) for u in ev],
                                                     sib=[int(u["d0_id"]) for u in sib])))
    print(f"setup: {len(small)} neutral D0 pool reactions <= {MAXA} atoms; train {len(train)}, eval {len(ev)}, sib "
          f"{len(sib)}")


def plan():
    S = json.loads((ROOT / "setup.json").read_text())
    rc = g4("g4_plan", "--models", MODELS, "--allow-unchecked", "--n-per-model", "2", "--n-d0-per-model", "4",
            "--max-atoms", str(MAXA), "--force")
    P = json.loads((TMLT / "g4" / "plan.json").read_text())
    R = json.loads((TRES / "G4_plan.json").read_text())
    d0 = [r for r in P["rxns"] if r.get("set") == "d0"]
    ids = [int(r["d0_id"]) for r in d0]
    bad = set(ids) & set(S["eval"] + S["sib"] + S["train"])
    check(rc == 0 and R.get("complete") and R.get("n_d0") == len(d0) and len(d0) == min(20, S["n_small"] - 8),
          f"g4_plan: {len(d0)} D0 rows (4 per model x 5), n_d1 {R.get('n_d1')}, report n_d0 {R.get('n_d0')}")
    check(not bad and len(ids) == len(set(ids)), f"no holdout / sibling / D0G2 training reaction drawn, no duplicate ({bad})")
    check(all(int(r["charge"]) == 0 and r["dipole"] == r["d0_class"] and r["fold_T1"] == -9 for r in d0),
          "D0 rows neutral, dipole = D0 class, fold -9")
    check(R["predicted_core_h"].get("d0", 0) > 0 and R["est"]["opt"]["units_total"] == len(P["rxns"]),
          f"cost / est cover the D0 rows (d0 {R['predicted_core_h'].get('d0')} core-h)")
    hc = SRC / "dft" / "plan" / "d0_holdout.csv"
    keep = hc.read_bytes()
    hc.unlink()
    rc2 = g4("g4_plan", "--models", MODELS, "--allow-unchecked", "--n-per-model", "2", "--max-atoms", str(MAXA), "--force")
    hc.write_bytes(keep + b"999999,x,1,edited,eval\r\n")
    rc3 = g4("g4_plan", "--models", MODELS, "--allow-unchecked", "--n-per-model", "2", "--max-atoms", str(MAXA), "--force")
    hc.write_bytes(keep)
    check(rc2 == 2 and rc3 == 3, f"holdout list missing -> exit 2 (not ready); edited -> exit 3 (D0-hold-fix) ({rc2}, {rc3})")
    rc4 = g4("g4_plan", "--models", MODELS, "--allow-unchecked", "--n-per-model", "2", "--n-d0-per-model", "4",
             "--max-atoms", str(MAXA))
    check(rc4 == 0, f"restored list -> the same plan is up to date ({rc4})")


def opt():
    sys.path.insert(0, str(HERE))
    import g4_al as A
    P = A.load_plan()
    r = min((x for x in P["rxns"] if x.get("set") == "d0"), key=lambda x: int(x["n_atoms"]))
    (ROOT / "row.json").write_text(json.dumps(r))
    info = A.opt_model(P, int(r["k"]), [r], {}, {}, 1)
    rec = json.loads(A.row_opt_rec(P, r).read_text())
    print(json.dumps({k: rec.get(k) for k in ("status", "dd_max", "same_ts", "n_imag", "tight", "set", "d0_id",
                                              "d0_class", "n_cycles", "wall_s")}), flush=True)
    check(rec.get("set") == "d0" and int(rec.get("d0_id")) == int(r["d0_id"]) and rec.get("tight") == "tight"
          and rec.get("status") in ("ok", "not_converged", "fail") and "/d0_g2/" in str(A.row_opt_rec(P, r)),
          f"D0 row optimised with TightOpt -> record in k<k>_<model>/d0_g2 ({rec.get('status')}, server alive "
          f"{info.get('server_alive_at_end')})")


def run():
    r = json.loads((ROOT / "row.json").read_text())
    rc = g4("run", "G4", "0", "1", "--only", f"d0_{int(r['d0_id'])}")
    sys.path.insert(0, str(HERE))
    import g4_al as A
    er = json.loads(A.row_engrad_rec(r).read_text()) if A.row_engrad_rec(r).is_file() else {}
    check(rc in (0, 2) and er.get("struct_id") == f"g4d0_{int(r['d0_id'])}" and er.get("status") == "ok",
          f"label-level EnGrad of the D0 row ({er.get('status')}, {er.get('struct_id')}, core-h {er.get('core_h')})")


def collect():
    r = json.loads((ROOT / "row.json").read_text())
    p = TMLT / "g4" / "plan.json"
    P = json.loads(p.read_text())
    P["rxns"] = [x for x in P["rxns"] if x.get("set") == "d0" and int(x["d0_id"]) == int(r["d0_id"])]
    p.write_text(json.dumps(P))
    rc = g4("collect", "--g4")
    sys.path.insert(0, str(HERE))
    import mlt_common as M
    R = json.loads((TRES / "G4_report.json").read_text())
    ext = TMLT / "dft" / "gen_G4.extxyz"
    al = M.read_extxyz(ext) if ext.is_file() and ext.stat().st_size else []
    inf = al[0].info if al else {}
    check(rc == 0 and R.get("complete") and R.get("n_planned_d0") == 1, f"collect complete on the reduced plan "
                                                                         f"(selected {R.get('n_selected')})")
    if al:
        check(str(inf.get("d0_id")) == str(r["d0_id"]) and str(inf.get("rxn_ids") or "") == "" and
              str(inf.get("folds") or "") == "" and str(inf.get("dipoles")) == str(r["d0_class"]),
              f"selected D0 structure: d0_id {inf.get('d0_id')}, rxn_ids "
                                              f"{inf.get('rxn_ids')!r}, dipoles {inf.get('dipoles')!r}")
    else:
        print(f"INFO: max |F| below 0.002 Eh/bohr: nothing selected ({R.get('n_selected')})")


if __name__ == "__main__":
    {"setup": setup, "plan": plan, "opt": opt, "run": run, "collect": collect}[sys.argv[1]]()
    print(f"{sum(OK)}/{len(OK)} checks pass")
    sys.exit(0 if all(OK) else 1)
