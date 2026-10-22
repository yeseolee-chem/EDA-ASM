"""D1-G2 common paths, constants and helpers (docs/specs/D1G2.md).

Writes only under $G2 (= $D1G2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/d1_g2) and analysis/d1_g2/results; the D1 run,
its code, the D1 job folders and the d1_pair / d1_hf scratch are read only (spec 13)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
G2 = Path(os.environ.get("D1G2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2"))
G2_FINAL, G2_JOBS, LOGS = G2 / "g2_final", G2 / "jobs", G2 / "logs"
sys.path.insert(0, str(REPO / "analysis" / "d1_pair"))
import d1pair_common as DP  # noqa: E402              (load_d1_labels, DPP, D1_RUN, D1_JOBS, D1_STATUS, ...)

D1_RUN = DP.D1_RUN                                      # /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run (read only)
D1_ENV = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env")
D1_REPO = D1_ENV / "EDA-ASM"                            # runtime code copy of the D1 run (config_run.yaml repo_root)
PILOT = D1_REPO / "D1_build" / "d1_autode_pilot"
MANIFEST = D1_RUN / "d1_manifest.csv"
MANIFEST_SHA = "d6abae6446d3fd294db706cebcbc290771f712803033e12695cb5a75bcee06f5"
LEDGER = D1_ENV.parent / "d1_ledger"                    # summary/<bundle>.small.tar, bundles.tsv
UBAI_JOBS = DP.UBAI_JOBS
HF_RES = REPO / "analysis" / "d1_hf" / "results"        # D1HF (read only)


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    """writes only under $G2 or analysis/d1_g2/results (spec 13); exit 3 otherwise."""
    q = Path(os.path.abspath(p))
    for root in (G2, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $G2 / analysis/d1_g2/results")


def write_atomic(path, data):
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def targets():
    """spec 1: labels_d1.json rows (d1pair_common.load_d1_labels checks) whose skeleton is not in d1pair_common.DPP."""
    return sorted((d for d in DP.load_d1_labels() if d["skeleton"] not in DP.DPP), key=lambda d: d["job_id"])
