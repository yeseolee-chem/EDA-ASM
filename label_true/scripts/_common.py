"""Shared utilities."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

BUNDLE_ROOT = Path(__file__).resolve().parent.parent

# ---- Constants from SPEC ----

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}

COVALENT_FACTOR_PRIMARY = 1.25
COVALENT_FACTOR_FALLBACK = 1.20
EH_TO_KCAL = 627.5094740631


def load_config():
    # config.yaml carries non-ASCII comments; do not depend on the job's LANG.
    with open(BUNDLE_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Nurion addition: let a single job override host-shaped knobs without
    # editing the shared config.yaml. Several PBS jobs run concurrently out of
    # the same bundle directory, so a job that rewrote config.yaml to set its
    # own parallelism would corrupt every sibling job's view. Method parameters
    # are deliberately NOT overridable here.
    for key, env, cast in (("parallel_rxns", "B3EDA_PARALLEL_RXNS", int),
                           ("max_retries",   "B3EDA_MAX_RETRIES",   int),
                           ("orca_bin",      "B3EDA_ORCA_BIN",      str)):
        v = os.environ.get(env)
        if v:
            cfg[key] = cast(v)
    return cfg


def read_template(name):
    return (BUNDLE_ROOT / "templates" / name).read_text()


def orca_terminated_normally(path):
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
        return "ORCA TERMINATED NORMALLY" in tail
    except Exception:
        return False


def read_xyz(path):
    import numpy as np
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def formula(syms):
    from collections import Counter
    c = Counter(syms)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}",
          file=sys.stderr, flush=True)


def run_stage(cmd, log_path=None):
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    kwargs = {}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["stdout"] = open(log_path, "w")
        kwargs["stderr"] = subprocess.STDOUT
    rc = subprocess.call(cmd, shell=isinstance(cmd, str), **kwargs)
    if rc != 0:
        log(f"FAILED (rc={rc}): {cmd}")
        sys.exit(rc)
