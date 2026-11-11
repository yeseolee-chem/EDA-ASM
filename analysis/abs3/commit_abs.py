#!/usr/bin/env python3
"""commit_abs.py — ABS3 10: git add + commit of one phase's files from a job (git from the reactot PATH of the job
scripts). Stdlib only.

  python commit_abs.py <phase> [<path> ...]     phase in S0 LS C PREREG K E S; default paths = the phase's files

Only paths under analysis/abs3/ and files below 50 MB (spec 10); never $ABS3. No-op when nothing is staged. Only the
phase's paths are committed. Retries while another git process holds the index lock. Author yeseo1ee
<yeseolele@gmail.com> + the co-author trailer."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402

R = "analysis/abs3/results"
PHASE_FILES = {
    "S0": [f"{R}/S0_start.json"],
    "LS": [f"{R}/LS_report.json"],
    "C": [f"{R}/C_tests.json", f"{R}/C_part_gpu.json", f"{R}/C_part_xgb.json", f"{R}/C_overfit.json",
          f"{R}/C8_smoke.json", f"{R}/DECISIONS.md"],
    "PREREG": [f"{R}/PREREG_ABS3.md", f"{R}/prereg_abs3.json", f"{R}/EA_selftest.json", f"{R}/DECISIONS.md"],
    "K": [f"{R}/KA_predictions.parquet", f"{R}/KA_pairs.parquet", f"{R}/KA_val.csv", f"{R}/KA_report.json"],
    "E": [f"{R}/EA_abs.csv", f"{R}/ED_compare.csv", f"{R}/EC_cancel.csv", f"{R}/EA_metrics.csv", f"{R}/EA_rank.csv",
          f"{R}/EA_cause.csv", f"{R}/EA_identity.csv", f"{R}/EA_report.json"],
    "S": [f"{R}/SUMMARY_ABS3.md", f"{R}/DECISIONS.md"],
}
MAX_BYTES = 50 * 1024 * 1024
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


def git(*a, check=False):
    for _ in range(12):
        r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
        if r.returncode == 0 or "index.lock" not in (r.stderr or ""):
            break
        time.sleep(10)
    if check and r.returncode != 0:
        C.stop(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r


def summary(phase):
    def j(name):
        p = C.RES / name
        return C.read_json(p) if p.is_file() else {}
    if phase == "S0":
        return f"start conditions all_pass {j('S0_start.json').get('all_pass')}"
    if phase == "LS":
        ls = j("LS_report.json")
        return f"reaction labels / sets all_pass {ls.get('all_pass')} ({(ls.get('LS4') or {}).get('n_reactions')} reactions)"
    if phase == "C":
        return f"code tests all_pass {j('C_tests.json').get('all_pass')}"
    if phase == "PREREG":
        return f"pre-registration before K (md sha256 {str(j('prereg_abs3.json').get('md_sha256'))[:12]})"
    if phase == "K":
        k = j("KA_report.json")
        return f"K: {k.get('n_units')} units, {(k.get('gpu_hours') or 0):.2f} GPU-h, XGB {(k.get('xgb_cpu_hours') or 0):.1f} CPU-h, " \
               f"C-P {all(v.get('ok') for v in (k.get('C_P') or {}).values()) and bool(k.get('C_P'))}"
    if phase == "E":
        s = (j("EA_report.json").get("summary") or {})
        marks = {lab: "/".join(v["mark"] for kk, v in d.items() if kk.startswith("p/")) for lab, d in (s.get("q_d") or {}).items()}
        return f"E: best absolute model {s.get('best_abs_model')}; para marks {json.dumps(marks, ensure_ascii=False)}"
    if phase == "S":
        return "SUMMARY_ABS3.md"
    return ""


def main(argv):
    if not argv or argv[0] not in PHASE_FILES:
        raise SystemExit(__doc__)
    phase, paths = argv[0], argv[1:] or PHASE_FILES[argv[0]]
    base = (C.REPO / "analysis" / "abs3").resolve()
    big, add = [], []
    for p in paths:
        ap = (C.REPO / p).resolve()
        if not (ap == base or base in ap.parents):
            C.stop(f"commit_abs: {p} is outside analysis/abs3")
        if not ap.exists():
            continue
        for f in ([ap] if ap.is_file() else [f for f in ap.rglob("*") if f.is_file()]):
            if f.stat().st_size > MAX_BYTES:
                big.append(str(f.relative_to(C.REPO)))
            elif not any(x.startswith(".") and x.endswith(".tmp") for x in f.parts):
                add.append(str(f.relative_to(C.REPO)))
    if not add:
        print(f"[commit {phase}] no file", flush=True)
        return 0
    git("add", "--", *add, check=True)
    if git("diff", "--cached", "--quiet", "--", *add).returncode == 0:
        print(f"[commit {phase}] nothing staged", flush=True)
        return 0
    msg = f"ABS3 {phase}: {summary(phase)}"
    if big:
        msg += "\n\nNot committed (> 50 MB, spec 10): " + ", ".join(big)
    msg += f"\n\n{TRAILER}"
    git("-c", "user.name=yeseo1ee", "-c", "user.email=yeseolele@gmail.com", "commit", "-q", "-m", msg, "--", *add,
        check=True)
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(json.dumps(dict(phase=phase, head=head, n_files=len(add), big=big, message=msg.splitlines()[0]),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
