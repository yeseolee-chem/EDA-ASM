#!/usr/bin/env python3
"""commit_phase.py — CH3-MODEL 13: git add + commit of one phase's files from a job (git from the reactot PATH that
jobs/cpu_ch3.slurm / gpu_ch3.slurm activate). Stdlib only.

  python commit_phase.py <phase> [<path> ...]     phase in S0 V C PREREG K E W I; default paths = the phase's files

Only paths under analysis/ch3/ (never $CH3 outputs, CLAUDE.md) and below 50 MB (spec 13); a file over 50 MB is left out
and listed in the commit message. No-op (exit 0) when nothing is staged. Retries while another git process holds
the index lock. Author yeseo1ee <yeseolele@gmail.com> + the Claude co-author trailer (user rule)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402

R = "analysis/ch3/results"
PHASE_FILES = {
    "S0": [f"{R}/S0_start.json", "analysis/ch3", f"{R}/DECISIONS.md"],
    "V": [f"{R}/V_inputs.json", f"{R}/lf_inputs", f"{R}/DECISIONS.md"],
    "C": [f"{R}/C_tests.json", f"{R}/C_part_cpu.json", f"{R}/C3r_regress.json", f"{R}/C8_smoke.json",
          f"{R}/DECISIONS.md"],
    "PREREG": [f"{R}/PREREG_CH3.md", f"{R}/prereg_ch3.json", f"{R}/E3_selftest.json"],
    "K": [f"{R}/K3_predictions.parquet", f"{R}/K3_val.csv", f"{R}/K3_report.json"],
    "E": [f"{R}/E3_compare.csv", f"{R}/E3_metrics.csv", f"{R}/E3_rank.csv", f"{R}/E3_cause.csv", f"{R}/E3_ad.json",
          f"{R}/E3_report.json"],
    "W": [f"{R}/W3_report.json", f"{R}/MODEL_CARD_ch3.json"],
    "I": [f"{R}/I3_selftest.json"],
}
MAX_BYTES = 50 * 1024 * 1024
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


def git(*a, check=False):
    for k in range(12):
        r = subprocess.run(["git", "-C", str(C3.REPO), *a], capture_output=True, text=True)
        if r.returncode == 0 or "index.lock" not in (r.stderr or ""):
            break
        time.sleep(10)
    if check and r.returncode != 0:
        C3.stop(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r


def summary(phase):
    """one line from the phase's report (what the commit carries)."""
    def j(name):
        p = C3.RES / name
        return C3.read_json(p) if p.is_file() else {}
    if phase == "S0":
        s = j("S0_start.json")
        return f"start conditions all_pass {s.get('all_pass')} (failed {s.get('failed')})"
    if phase == "V":
        v = j("V_inputs.json")
        return f"regenerated lf inputs, V all_pass {v.get('all_pass')} (failed {v.get('failed')})"
    if phase == "C":
        c = j("C_tests.json")
        return f"code tests all_pass {c.get('all_pass')}"
    if phase == "PREREG":
        p = j("prereg_ch3.json")
        return f"pre-registration before K (md sha256 {str(p.get('md_sha256'))[:12]})"
    if phase == "K":
        k = j("K3_report.json")
        return f"K3 cross-validation: {k.get('n_units')} units, {k.get('n_test_pairs')} test pairs, " \
               f"{(k.get('gpu_hours') or 0):.2f} GPU-h"
    if phase == "E":
        e = j("E3_report.json")
        a = e.get("adoption") or {}
        return f"E3 ch3 vs lf_sum: adopted {a.get('adopted')} (failed {a.get('failed')})"
    if phase == "W":
        w = j("W3_report.json")
        s = w.get("selection") or {}
        return f"final model: cell {s.get('cell')}, {s.get('n_epochs')} epochs, {w.get('n_train_pairs')} pairs"
    if phase == "I":
        i = j("I3_selftest.json")
        return f"predictor checks I-1/I-2 all_pass {i.get('all_pass_I1_I2')}"
    return ""


def main(argv):
    if not argv or argv[0] not in PHASE_FILES:
        raise SystemExit(__doc__)
    phase, paths = argv[0], argv[1:] or PHASE_FILES[argv[0]]
    big, add = [], []
    for p in paths:
        ap = (C3.REPO / p).resolve()
        if not (ap == (C3.REPO / "analysis" / "ch3").resolve() or (C3.REPO / "analysis" / "ch3").resolve() in ap.parents):
            C3.stop(f"commit_phase: {p} is outside analysis/ch3")
        if not ap.exists():
            continue
        files = [ap] if ap.is_file() else [f for f in ap.rglob("*") if f.is_file()]
        for f in files:
            if f.stat().st_size > MAX_BYTES:
                big.append(str(f.relative_to(C3.REPO)))
            elif not any(x.startswith(".") and x.endswith(".tmp") for x in f.parts):
                add.append(str(f.relative_to(C3.REPO)))
    if add:
        git("add", "--", *add, check=True)
    if git("diff", "--cached", "--quiet").returncode == 0:
        print(f"[commit {phase}] nothing staged", flush=True)
        return 0
    msg = f"CH3-MODEL {phase}: {summary(phase)}"
    if big:
        msg += "\n\nNot committed (> 50 MB, spec 13): " + ", ".join(big)
    msg += f"\n\n{TRAILER}"
    git("-c", "user.name=yeseo1ee", "-c", "user.email=yeseolele@gmail.com", "commit", "-q", "-m", msg, check=True)
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(json.dumps(dict(phase=phase, head=head, n_files=len(add), big=big, message=msg.splitlines()[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
