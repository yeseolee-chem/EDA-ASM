"""ADF queue feeder per ASR_Fragmentation_Spec.md section 6.

Watches outputs/stage5a/review_log.json. For each reaction whose user_decision
is `accepted` (status=accepted) or `modified` (status=modified), if a spec
result file does not yet exist at outputs/asr_spec/<rxn_id>.json, submit the
spec-compliant runner `scripts/run_asr_spec.py`.

The runner internally runs 11 ADF jobs (BP86/D3BJ/TZ2P/Good) and writes a
spec-section-7 schema JSON.

Concurrency is bounded by MAX_CONCURRENT. Idempotent: skipped if already done.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
REVIEW_LOG = REPO / "outputs/stage5a/review_log.json"
SPEC_OUT_DIR = REPO / "outputs/asr_spec"
QUEUE_LOG = REPO / "outputs/asr_spec/queue_log.jsonl"

AMSBIN = os.environ.get("AMSBIN")
DEFAULT_CONCURRENT = 4
POLL_SECONDS = 30


def already_done(rxn_id: str) -> bool:
    fp = SPEC_OUT_DIR / f"{rxn_id}.json"
    if not fp.exists():
        return False
    try:
        d = json.loads(fp.read_text())
        # Only treat as done if AUTO_ACCEPT_CANDIDATE / MANUAL_REVIEW / passed
        # (NOT "FAILED" — those should be retried)
        return d.get("status_at_queue") in ("AUTO_ACCEPT_CANDIDATE",
                                              "MANUAL_REVIEW_REQUIRED")
    except Exception:
        return False


def submit_job(rxn_id: str, log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{rxn_id}.log"
    env = os.environ.copy()
    env["NSCM"] = "1"
    env["PYTHONPATH"] = f"{REPO}/src:{REPO}:" + env.get("PYTHONPATH", "")
    cmd = [
        "nice", "-n", "10",
        f"{AMSBIN}/amspython", "scripts/run_asr_spec.py", "--rxn_id", rxn_id,
    ]
    fh = open(log_file, "ab")
    fh.write(f"\n[{time.strftime('%H:%M:%S')}] START {rxn_id}\n".encode())
    fh.flush()
    return subprocess.Popen(cmd, cwd=str(REPO), env=env,
                             stdout=fh, stderr=subprocess.STDOUT)


def load_accepted(cutoff_iso: str | None = None) -> list[str]:
    """Read review_log.json; return rxn_ids whose status is accepted/modified.
    Optionally filter to those reviewed AFTER cutoff_iso."""
    if not REVIEW_LOG.exists():
        return []
    try:
        d = json.loads(REVIEW_LOG.read_text())
    except Exception:
        return []
    out = []
    for rid, v in d.items():
        if v.get("review_status") not in ("accepted", "modified"):
            continue
        if cutoff_iso and v.get("review_completed_at", "") < cutoff_iso:
            continue
        out.append(rid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_CONCURRENT)
    ap.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    ap.add_argument("--cutoff-iso", default=None,
                    help="Only process accepts after this ISO timestamp")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log-dir", type=Path,
                    default=REPO / "outputs/asr_spec/logs")
    args = ap.parse_args()

    if not AMSBIN:
        print("ERROR: AMSBIN not set. Source amsbashrc.sh first.", file=sys.stderr)
        sys.exit(1)

    print(f"[feeder] AMSBIN={AMSBIN}")
    print(f"[feeder] runner=scripts/run_asr_spec.py (BP86/D3BJ/TZ2P/Good)")
    print(f"[feeder] max_concurrent={args.max_concurrent}  poll={args.poll_seconds}s")
    if args.cutoff_iso:
        print(f"[feeder] cutoff_iso={args.cutoff_iso}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    SPEC_OUT_DIR.mkdir(parents=True, exist_ok=True)

    running: dict[str, subprocess.Popen] = {}
    while True:
        # Reap finished
        for rid in list(running):
            rc = running[rid].poll()
            if rc is not None:
                msg = f"[{time.strftime('%H:%M:%S')}] DONE {rid} rc={rc}"
                print(msg, flush=True)
                with open(QUEUE_LOG, "a") as f:
                    f.write(json.dumps({"rxn_id": rid, "rc": rc,
                                          "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n")
                running.pop(rid)

        # Submit new
        accepted = load_accepted(cutoff_iso=args.cutoff_iso)
        for rid in accepted:
            if len(running) >= args.max_concurrent:
                break
            if rid in running:
                continue
            if already_done(rid):
                continue
            print(f"[{time.strftime('%H:%M:%S')}] SUBMIT {rid}", flush=True)
            running[rid] = submit_job(rid, args.log_dir)

        if args.once:
            break
        time.sleep(args.poll_seconds)

    # Wait for stragglers
    for rid, proc in running.items():
        proc.wait()


if __name__ == "__main__":
    main()
