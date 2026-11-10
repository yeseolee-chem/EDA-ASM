#!/usr/bin/env python3
"""commit_m.py — CH3M 11: git add + commit of one phase's files from a job (git from the reactot PATH of the job
scripts). Stdlib only.

  python commit_m.py <phase> [<path> ...]     phase in S0 XP C PREREG K E W I S; default paths = the phase's files

Only paths under analysis/ch3m/ and files below 50 MB (spec 11); never $CH3M. No-op when nothing is staged. Retries
while another git process holds the index lock. Author yeseo1ee <yeseolele@gmail.com> + the co-author trailer."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402

R = "analysis/ch3m/results"
PHASE_FILES = {
    "S0": [f"{R}/S0_start.json"],
    "XP": [f"{R}/XP_report.json", f"{R}/DECISIONS.md"],
    "C": [f"{R}/C_tests.json", f"{R}/C_part_cpu.json", f"{R}/CR_tf.json", f"{R}/CR_gnn.json", f"{R}/CR_gnn2.json",
          f"{R}/CR_xgb.json", f"{R}/C8_smoke_nn.json", f"{R}/C8_smoke_xgb.json", f"{R}/C_deviation.json",
          f"{R}/DECISIONS.md"],
    "PREREG": [f"{R}/PREREG_CH3M.md", f"{R}/prereg_ch3m.json", f"{R}/EM_selftest.json", f"{R}/DECISIONS.md"],
    "K": [f"{R}/KM_predictions.parquet", f"{R}/KM_val.csv", f"{R}/KM_report.json"],
    "E": [f"{R}/EM_3v10.csv", f"{R}/EM_noise.csv", f"{R}/EM_choice.csv", f"{R}/EM_cancel.csv", f"{R}/EM_metrics.csv",
          f"{R}/EM_rank.csv", f"{R}/EM_cause.csv", f"{R}/EM_identity.csv", f"{R}/EM_ad.json", f"{R}/EM_report.json"],
    "W": [f"{R}/WM_report.json", f"{R}/MODEL_CARD_ch3m.json"],
    "I": [f"{R}/IM_selftest.json"],
    "S": [f"{R}/SUMMARY_CH3M.md", f"{R}/DECISIONS.md"],
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
    if phase == "XP":
        return f"XGB input packs ok {j('XP_report.json').get('ok')}"
    if phase == "C":
        c = j("C_tests.json")
        return f"code tests all_pass {c.get('all_pass')}" + (" (C-RG exception recorded)" if c.get("C_deviation") else "")
    if phase == "PREREG":
        return f"pre-registration before K (md sha256 {str(j('prereg_ch3m.json').get('md_sha256'))[:12]})"
    if phase == "K":
        k = j("KM_report.json")
        return f"K: {k.get('n_units')} units, {(k.get('gpu_hours') or 0):.2f} GPU-h, XGB {(k.get('xgb_cpu_hours') or 0):.1f} CPU-h, " \
               f"C-XB {(k.get('C_XB') or {}).get('ok')}"
    if phase == "E":
        ch = j("EM_report.json").get("choice") or {}
        return f"E: replacement {ch.get('chosen')} (final model {ch.get('final_model')})"
    if phase == "W":
        w = j("WM_report.json")
        s = w.get("selection") or {}
        return f"final model {w.get('candidate')}: cell {s.get('cell')}, {s.get('n_epochs')} epochs, {w.get('n_train_pairs')} pairs"
    if phase == "I":
        return f"predictor checks I-1/I-2 all_pass {j('IM_selftest.json').get('all_pass_I1_I2')}"
    if phase == "S":
        return "SUMMARY_CH3M.md"
    return ""


def main(argv):
    if not argv or argv[0] not in PHASE_FILES:
        raise SystemExit(__doc__)
    phase, paths = argv[0], argv[1:] or PHASE_FILES[argv[0]]
    base = (C.REPO / "analysis" / "ch3m").resolve()
    big, add = [], []
    for p in paths:
        ap = (C.REPO / p).resolve()
        if not (ap == base or base in ap.parents):
            C.stop(f"commit_m: {p} is outside analysis/ch3m")
        if not ap.exists():
            continue
        for f in ([ap] if ap.is_file() else [f for f in ap.rglob("*") if f.is_file()]):
            if f.stat().st_size > MAX_BYTES:
                big.append(str(f.relative_to(C.REPO)))
            elif not any(x.startswith(".") and x.endswith(".tmp") for x in f.parts):
                add.append(str(f.relative_to(C.REPO)))
    if add:
        git("add", "--", *add, check=True)
    if git("diff", "--cached", "--quiet").returncode == 0:
        print(f"[commit {phase}] nothing staged", flush=True)
        return 0
    msg = f"CH3M {phase}: {summary(phase)}"
    if big:
        msg += "\n\nNot committed (> 50 MB, spec 11): " + ", ".join(big)
    msg += f"\n\n{TRAILER}"
    git("-c", "user.name=yeseo1ee", "-c", "user.email=yeseolele@gmail.com", "commit", "-q", "-m", msg, check=True)
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(json.dumps(dict(phase=phase, head=head, n_files=len(add), big=big, message=msg.splitlines()[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
