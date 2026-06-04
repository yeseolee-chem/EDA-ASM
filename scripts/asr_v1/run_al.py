"""Orchestrate 5 rounds of active learning over the Δ-learning M1 backbone.

Per round R:
  1. ``al_train_and_select.py`` — train Δ-M1 ensemble + pick top-K
  2. ``al_prepare_round.py``    — fragment + generate ADF inputs for the picks
  3. ``sbatch --wait`` the round's submit_round_adf.sh (CPU array)
  4. ``al_postprocess.py``      — parse new ADF labels, merge into master parquet

This script is designed to run as a single long-walltime Slurm job on a
GPU partition (training needs GPU; ``--wait`` idles cheaply during ADF).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], log_prefix: str) -> None:
    print(f"\n[{log_prefix}] running: {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed with rc={res.returncode}")


def _sbatch_wait(submit_script: str) -> int:
    print(f"\n[orch] sbatch --wait {submit_script}")
    sys.stdout.flush()
    # --wait: blocks until the submitted job terminates.
    res = subprocess.run(
        ["sbatch", "--wait", submit_script],
        check=False, capture_output=True, text=True,
    )
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    m = re.search(r"Submitted batch job (\d+)", res.stdout)
    job_id = int(m.group(1)) if m else -1
    if res.returncode != 0:
        raise RuntimeError(f"sbatch --wait failed: rc={res.returncode}")
    return job_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--end-round", type=int, default=5)
    ap.add_argument("--n-pick", type=int, default=40)
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--pool", default="outputs/asr_v1/al/pool_features.pt")
    ap.add_argument("--config", default="configs/asr_v1_maceoff_delta_n250.yaml")
    args = ap.parse_args()

    log = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rounds": []}
    log_path = Path("outputs/asr_v1/al/run_summary.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for R in range(args.start_round, args.end_round + 1):
        round_dir = Path(f"outputs/asr_v1/al/round_{R:02d}")
        round_dir.mkdir(parents=True, exist_ok=True)
        round_log = {"round": R, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"\n========== AL round {R}/{args.end_round} ==========")

        # 1) train + select
        _run([
            sys.executable, "scripts/asr_v1/al_train_and_select.py",
            "--config", args.config,
            "--round", str(R),
            "--pool", args.pool,
            "--round-dir", str(round_dir),
            "--n-pick", str(args.n_pick),
            "--n-ensemble", str(args.n_ensemble),
        ], log_prefix=f"r{R}-select")

        # 2) prepare ADF inputs
        _run([
            sys.executable, "scripts/asr_v1/al_prepare_round.py",
            "--round", str(R),
            "--round-dir", str(round_dir),
        ], log_prefix=f"r{R}-prep")

        # 3) wait for ADF batch
        submit_script = round_dir / "submit_round_adf.sh"
        job_id = _sbatch_wait(str(submit_script))
        round_log["adf_job_id"] = job_id

        # 4) parse + merge labels
        _run([
            sys.executable, "scripts/asr_v1/al_postprocess.py",
            "--round", str(R),
            "--round-dir", str(round_dir),
        ], log_prefix=f"r{R}-post")

        round_log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Read picks_summary for headline stats
        sp = round_dir / "picks_summary.json"
        if sp.exists():
            ps = json.loads(sp.read_text())
            round_log["picks_summary"] = {
                k: ps[k] for k in (
                    "n_labeled_at_start", "n_pool_after_filter", "n_pick",
                    "fragments_ok", "fragments_warning", "fragments_failed",
                ) if k in ps
            }
        log["rounds"].append(round_log)
        log_path.write_text(json.dumps(log, indent=2))
        print(f"[orch] round {R} done. Summary so far: {log_path}")

    log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_path.write_text(json.dumps(log, indent=2))
    print("\n[orch] all rounds done.")


if __name__ == "__main__":
    main()
