#!/usr/bin/env amspython
"""Run one fragmentation candidate (synthetic rid) through ADF.

Points STAGE5A_DIR at candidates_stage5a/, OUT_DIR at candidate_results/,
and deletes the /tmp workdir after a successful write to keep disk usage
bounded.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
for p in (ROOT, ROOT / "ADF_500" / "scripts", ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import run_asr_spec  # type: ignore

run_asr_spec.STAGE5A_DIR = ROOT / "Validate" / "refrag" / "candidates_stage5a"
run_asr_spec.OUT_DIR = ROOT / "Validate" / "refrag" / "candidate_results"

# Frame loader matches the dand_id 'head' against rxn_id. Our synthetic rids
# are '<orig>__c<N>'; strip the suffix so the Halo8 lookup uses the parent rid.
_orig_load_3_frames = run_asr_spec.load_3_frames

def _patched_load_3_frames(rxn_id, stage5a):
    """Strip '__cN' candidate suffix when fetching Halo8 frames."""
    base = rxn_id.split("__", 1)[0]
    return _orig_load_3_frames(base, stage5a)

run_asr_spec.load_3_frames = _patched_load_3_frames


def _cleanup_workdir(rxn_id: str) -> None:
    """Remove the /tmp workdir to keep disk usage bounded."""
    wd = Path("/tmp/yeseo1ee/asr_spec") / rxn_id
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    import argparse, json, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_id", required=True)
    args = ap.parse_args()
    rid = args.rxn_id
    out_path = run_asr_spec.OUT_DIR / f"{rid}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("status_at_queue") not in (None, "FAILED"):
                print(f"[SKIP] {rid} already done")
                _cleanup_workdir(rid)
                sys.exit(0)
        except Exception:
            pass
    t0 = time.time()
    try:
        result = run_asr_spec.run_one(rid)
    except Exception as e:
        result = {
            "reaction_id": rid,
            "schema_version": "asr_spec_v1_candidate",
            "status_at_queue": "FAILED",
            "manual_review_reasons": [f"ERROR: {e}"],
            "traceback": traceback.format_exc(),
        }
    result["fragmentation_revision"] = 3
    result["candidate_wall_sec"] = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("status_at_queue")
    print(f"[OK] {rid}: {status} ({time.time()-t0:.0f}s)")
    if status not in (None, "FAILED"):
        _cleanup_workdir(rid)
