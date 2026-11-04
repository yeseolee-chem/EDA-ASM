#!/usr/bin/env python3
"""tests/integ_jobs.py — Integrate step: the module quick tests in ONE test job (the GPU sandbox 1061635 reached its
SLURM step limit), each step a subprocess with its own environment, outputs only below $MLT/tmp/integtest:
  sbatch -p cpu1,cpu2 -J mlt_test_integ --time=48:00:00 -c 16 --mem=64G --comment=mltp:test:integ:manual \
         --export=ALL,MLTENV=reactot jobs/mlt.slurm tests/integ_jobs.py cpu [step ...]
  -> $MLT/tmp/integtest/jobs/summary_<group>.json (rc + output tail per step)"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REAL = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
IT = REAL / "tmp" / "integtest"
KT = REAL / "tmp" / "ktest"
SMOKE_TAR = IT / "a" / "a2smoke" / "MACE-OMOL-0" / "ade" / "jobs" / "T00312" / "inputs.tar.gz"
PY = sys.executable


def env(**kw):
    e = dict(os.environ)
    e.update({k: str(v) for k, v in kw.items()})
    return e


def _prereg_check():
    import hashlib
    d = IT / "r" / "res"
    md, js = d / "PREREG_V2.md", d / "prereg_v2.json"
    J = json.loads(js.read_text())
    ok = J.get("md_sha256") == hashlib.sha256(md.read_bytes()).hexdigest()
    t0 = md.stat().st_mtime
    r = subprocess.run([PY, str(HERE / "prereg_v2.py"), "write", "--allow-partial", "--from-results", str(HERE / "results")],
                       env=env(MLT_RESULTS=d), capture_output=True, text=True, cwd=HERE)
    ok2 = r.returncode == 0 and md.stat().st_mtime == t0
    refs = ((J.get("criteria") or {}).get("refs") or {}).get("primary")
    print(f"prereg: md_sha256 matches {ok}; rerun rc {r.returncode}, not rewritten {ok2}; refs.primary {refs}; "
          f"C1 inner-validation text: {'tuning fit' in md.read_text() or '조정' in md.read_text()}")
    return 0 if ok and ok2 and refs == "min" else 1


def _copy_res(src, dst):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return 0


STEPS = {
    "syntax": ([PY, "tests/integ_syntax.py"], {}),
    "pipe": ([PY, "tests/test_pipeline.py"], dict(PIPETEST_ROOT=IT / "pipetest")),
    "dry": ([PY, "pipeline.py", "dry"], {}),
    "kselect": ([PY, "train_k.py", "select"], dict(MLT_K_TEST_ROOT=KT)),
    "kcons": ([PY, "tests/integ_kcons.py"], {}),
    "c2k": ([PY, "tests.py", "c2_k", "--units-dir", str(KT / "units"), "--src-results", str(KT / "results")],
            dict(MLT_RESULTS=IT / "cr" / "res")),
    "c3files": ([PY, "tests.py", "c3_files", str(SMOKE_TAR)], {}),
    "prereg": ([PY, "prereg_v2.py", "write", "--allow-partial", "--from-results", str(HERE / "results")],
               dict(MLT_RESULTS=IT / "r" / "res")),
    "prereg_check": (_prereg_check, {}),
    "s_copy": (lambda: _copy_res(REAL / "tmp" / "ewpstest" / "results", IT / "s" / "res"), {}),
    "summary": ([PY, "summary_v2.py", "write", "--stop-file", "no"], dict(MLT_RESULTS=IT / "s" / "res")),
    "eselftest": ([PY, "eval_v2.py", "selftest"], dict(MLT_SCRATCH=IT / "ewps", MLT_RESULTS=IT / "ewps" / "res")),
    # GPU (MLTENV=mlts): predict_v2.e2e on a COPY of the EWPS fixture (stand-in A-2 folders, zero-shot committee)
    "e_copy": (lambda: _copy_res(REAL / "tmp" / "ewpstest", IT / "ewpstest"), {}),
    "e2e": ([PY, "tests/ewps_e2e.py"], dict(MLT_SCRATCH=IT / "ewpstest", MLT_RESULTS=IT / "ewpstest" / "results",
                                            MLTS_W_MEMBERS="MACE-OMOL-0,AIMNet2")),
}
GROUPS = dict(cpu=["syntax", "pipe", "dry", "kselect", "kcons", "c2k", "c3files", "prereg", "prereg_check", "s_copy",
                   "summary", "eselftest"],
              gpu=["e_copy", "e2e"])


def main(argv):
    group = argv[0] if argv else "cpu"
    names = argv[1:] or GROUPS[group]
    out = IT / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    for d in (IT / "cr" / "res", IT / "r" / "res", IT / "ewps" / "res"):
        d.mkdir(parents=True, exist_ok=True)
    rep = dict(group=group, host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"), steps={})
    for n in names:
        cmd, e = STEPS[n]
        t = time.time()
        print(f"===== {n} {time.strftime('%H:%M:%S')}", flush=True)
        if callable(cmd):
            try:
                rc, tail = cmd(), ""
            except Exception as ex:                                     # noqa: BLE001
                rc, tail = 1, f"{type(ex).__name__}: {ex}"
        else:
            try:
                r = subprocess.run(cmd, cwd=HERE, env=env(**e), capture_output=True, text=True, timeout=3600)
                rc, tail = r.returncode, (r.stdout + "\n" + r.stderr)[-6000:]
            except subprocess.TimeoutExpired:
                rc, tail = "timeout", ""
        print(tail[-3000:], flush=True)
        print(f"===== {n} rc {rc} ({time.time() - t:.0f} s)", flush=True)
        rep["steps"][n] = dict(rc=rc, wall_s=round(time.time() - t, 1), tail=tail[-4000:])
        (out / f"summary_{group}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v["rc"] for k, v in rep["steps"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
