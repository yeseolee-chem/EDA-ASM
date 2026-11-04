"""tests/d0plus_fix.py — sandbox only, TEST ROOTS ONLY (D0-plus review fixes): exit codes + files changed for the
pin / holdout-fixity / cap / in-flight / replace guards. $MLT_SCRATCH = a planned d0plustest root R (gen_d0 plan_d0
--n-near 2 --n-g2 200 --n-holdout 100 done, $MLT_RESULTS = R/results). Modes:
  idem          on R itself: the same gen_d0 command, `gen_dft plan_d0 --n-near 2 --n-g2 200`, `gen_dft plan_d0
                --n-near 2` (auto + pin) -> all exit 0 'up to date', no file changed (sha256 + mtime)
  fast          on a clone R_fast (plan dir + results + 20 records per group copied): early STOPs (pin, --n-holdout
                without --replace-holdout, bad arguments, missing / edited holdout file for plan_d0, collect --d0 and
                finetune, edited results copy), then the restored file is up to date
  heavy1        on a clone R_heavy1: --n-g2 3000 (cap STOP, nothing written); in-flight guard (starts.json without
                unit.json for a training unit, gen_d0 --n-g2 199 -> exit 2, nothing written)
  heavy2        on a clone R_heavy2: --replace-holdout refused after FT data with D0 inputs exist (fake data json);
                allowed without (50 evaluation reactions), then --n-holdout 100 --replace-holdout reproduces R's sha
  ftcheck       (internal) import finetune, call d0_holdout() on $MLT_SCRATCH; exit 0 / its SystemExit code
Prints JSON (one record per scenario: command, exit, expected, ok, files changed)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/d0plustest")
for root in (M.MLT, M.RES):
    Path(root).resolve().relative_to(BASE)          # raises outside the test base
mode = sys.argv[1]
if mode == "ftcheck":
    import finetune as F                            # noqa: E402
    try:
        ids, meta = F.d0_holdout()
        print(json.dumps(dict(n=len(ids), sha=(meta.get("sha256") or "")[:12], roles=meta.get("by_role"))))
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)
R = Path(M.MLT)


def snap(root):
    fs = sorted((root / "dft" / "plan").glob("*")) + sorted(p for p in (root / "results").rglob("*") if p.is_file())
    return {str(p.relative_to(root)): (M.sha256(p), p.stat().st_mtime_ns) for p in fs if p.is_file()}


def run(root, args, env_extra=None, script=None):
    env = dict(os.environ, MLT_SCRATCH=str(root), MLT_RESULTS=str(root / "results"), **(env_extra or {}))
    cmd = [sys.executable, str(HERE / (script or args[0]))] + (args if script else args[1:])
    p = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True)
    tail = [ln for ln in (p.stdout + p.stderr).strip().splitlines() if ln.startswith(("[STOP]", "[NOT READY]", "[plan_d0]"))]
    return p.returncode, tail[-1][:400] if tail else (p.stdout + p.stderr).strip()[-300:]


out = []


def case(root, name, args, expect, script=None, exclude=()):
    s0 = snap(root)
    rc, msg = run(root, args, script=script)
    s1 = snap(root)
    ch = sorted(k for k in set(s0) | set(s1) if s0.get(k) != s1.get(k) and k not in exclude)
    out.append(dict(case=name, cmd=" ".join(args), exit=rc, expected=expect, ok=rc == expect, files_changed=ch[:8],
                    n_changed=len(ch), msg=msg))
    return rc, ch


def put(p, obj):
    """write a clone file (the clone is under BASE but not under $MLT_SCRATCH, so not via mlt_common's guard)."""
    p = Path(p)
    p.parent.resolve().relative_to(BASE)            # raises outside the test base
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def clone(tag):
    C = BASE / f"{R.name}_{tag}"
    if C.exists():
        shutil.rmtree(C)
    for sub in ("dft/plan", "results"):
        shutil.copytree(R / sub, C / sub)
    for g in ("D0G1", "D0G3"):
        (C / "dft" / "runs" / g).mkdir(parents=True, exist_ok=True)
        for p in sorted((R / "dft" / "runs" / g).glob("*.json"))[:20]:
            shutil.copyfile(p, C / "dft" / "runs" / g / p.name)
    return C


PLAN0 = ["gen_d0.py", "plan_d0", "--n-near", "2", "--n-g2", "200", "--n-holdout", "100"]
if mode == "idem":
    for nm, a in (("same gen_d0 command", PLAN0),
                  ("gen_dft explicit 200", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "200"]),
                  ("gen_dft auto (pin)", ["gen_dft.py", "plan_d0", "--n-near", "2"])):
        case(R, nm, a, 0)
elif mode == "fast":
    C = clone("fast")
    H, HR = C / "dft" / "plan" / "d0_holdout.csv", C / "results" / "d0_holdout.csv"
    case(C, "pin: gen_dft --n-g2 170", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "170"], 3)
    case(C, "pin: gen_dft --n-g2 0", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "0"], 3)
    case(C, "--n-holdout 50 without --replace-holdout", PLAN0[:-1] + ["50"], 3)
    case(C, "bad --n-g2", ["gen_d0.py", "plan_d0", "--n-g2", "abc"], 3)
    case(C, "bad --n-near", ["gen_d0.py", "plan_d0", "--n-near", "3"], 3)
    case(C, "--unpin with --n-g2", ["gen_d0.py", "plan_d0", "--unpin", "--n-g2", "5"], 3)
    case(C, "argparse error", ["gen_d0.py", "plan_d0", "--n-holdout", "x"], 3)
    txt = H.read_text()
    H.rename(H.with_suffix(".bak"))
    case(C, "file missing: gen_dft plan_d0 --n-g2 200", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "200"], 3,
         exclude=("dft/plan/d0_holdout.bak",))
    case(C, "file missing: collect --d0", ["gen_dft.py", "collect", "--d0"], 3, exclude=("dft/plan/d0_holdout.bak",))
    case(C, "file missing: finetune d0_holdout", ["ftcheck"], 3, script="tests/d0plus_fix.py",
         exclude=("dft/plan/d0_holdout.bak",))
    H.with_suffix(".bak").rename(H)
    lines = txt.splitlines(keepends=True)
    a, b = lines[1].split(",", 1), lines[2].split(",", 1)            # swap the d0_ids of the first two rows
    put(H, "".join([lines[0], b[0] + "," + a[1], a[0] + "," + b[1]] + lines[3:]))
    case(C, "same-length edit: gen_dft plan_d0 --n-g2 200", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "200"],
         3)
    case(C, "same-length edit: collect --d0", ["gen_dft.py", "collect", "--d0"], 3)
    case(C, "same-length edit: finetune d0_holdout", ["ftcheck"], 3, script="tests/d0plus_fix.py")
    put(H, txt)
    rtxt = HR.read_text()
    put(HR, rtxt.replace(",eval\n", ",sibling\n", 1))
    case(C, "results copy edited: plan_d0", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "200"], 3)
    put(HR, rtxt)
    case(C, "restored: gen_dft plan_d0 --n-g2 200 up to date", ["gen_dft.py", "plan_d0", "--n-near", "2", "--n-g2", "200"],
         0)
    case(C, "restored: finetune d0_holdout", ["ftcheck"], 0, script="tests/d0plus_fix.py")
elif mode == "heavy1":
    C = clone("heavy1")
    case(C, "cap: --n-g2 3000", ["gen_d0.py", "plan_d0", "--n-near", "2", "--n-g2", "3000"], 3)
    g2p = M.read_json(C / "dft" / "plan" / "g2_plan_d0.json")
    rid = int(g2p["rxns"][0]["rxn_id"])
    st = C / "dft" / "runs" / "D0G2" / str(rid) / "starts.json"
    put(st, dict(test="in-flight unit (tests/d0plus_fix.py)"))
    case(C, f"in-flight unit {rid}: gen_d0 --n-g2 199", ["gen_d0.py", "plan_d0", "--n-near", "2", "--n-g2", "199"], 2)
    st.unlink()
elif mode == "heavy2":
    C = clone("heavy2")
    sha0 = M.sha256(C / "dft" / "plan" / "d0_holdout.csv")
    fj = C / "results" / "ft" / "FT_0_data.json"
    put(fj, dict(test="fake data json with D0 inputs (tests/d0plus_fix.py)", inputs={"D0G1": dict(present=True)},
                 d0_holdout_sha256=sha0))
    case(C, "replace after FT data built", PLAN0[:-1] + ["50", "--replace-holdout"], 3)
    fj.unlink()
    rc, ch = case(C, "replace without FT data (50)", PLAN0[:-1] + ["50", "--replace-holdout"], 0)
    H = M.read_json(C / "dft" / "plan" / "g2_plan_d0.json")["holdout"]
    out[-1].update(n_eval=H["n"], n_listed=H["n_listed"], sha=(H["sha256"] or "")[:12])
    case(C, "replace back to 100", PLAN0 + ["--replace-holdout"], 0)
    sha2 = M.sha256(C / "dft" / "plan" / "d0_holdout.csv")
    out[-1].update(sha=sha2[:12], same_as_original=sha2 == sha0)
else:
    raise SystemExit("mode idem | fast | heavy1 | heavy2 | ftcheck")
print(json.dumps(dict(mode=mode, root=str(R), all_ok=all(x["ok"] for x in out), cases=out), indent=1))
