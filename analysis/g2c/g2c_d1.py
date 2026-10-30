#!/usr/bin/env python3
"""G2c track (docs/specs/GEO1B.md 6): one more independent run of the D1-G2 protocol per D1 reaction into a new
scratch root ($D1G2_SCRATCH = $G2C/d1_r<k>), compute unchanged (analysis/d1_g2/g2_autode.py imported), with the
D1-G2 _graph repair active exactly as analysis/geo/d0g2.py does (GEO1 decision Z-graph). D1 run env, sbatch only
(analysis/g2c/jobs/g2c.slurm).

  python g2c_d1.py list                                  -> $G2C/lists/d1_targets.csv (job_id, rxn_id; g2_common.targets())
  python g2c_d1.py one <job_id> <cores>                  -> $D1G2_SCRATCH/g2_final/<rxn_id>/ (.done or .fail_*)
  python g2c_d1.py pool <list.csv> <n_par> <cores> [i n] (slice i of n; stops starting new reactions after
                                                          $G2C_STOP_AFTER_H h, default 44; idempotent: finished = skip)
autodE draws a fresh numpy RandomState for every conformer (autode/conformers/conf_gen.py), so a repeat run is an
independent sample of the same protocol. Nothing here reads a label for selection; finalize maps atom order to the
label as D1-G2 did (rung g2 format).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIG_D1G2 = "/gpfs/tmp_cpu2/yeseo1ee/d1_g2"
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
ROOT = os.path.abspath(os.environ.get("D1G2_SCRATCH", ""))
if os.path.dirname(ROOT) != str(G2C) or not os.path.basename(ROOT).startswith("d1_r") or ROOT == ORIG_D1G2:
    print(f"STOP: D1G2_SCRATCH={ROOT!r} must be {G2C}/d1_r<k> (never the D1-G2 scratch)", file=sys.stderr)
    sys.exit(3)
sys.path.insert(0, str(REPO / "analysis" / "d1_g2"))
import pandas as pd  # noqa: E402
import g2_common as G  # noqa: E402
import g2_autode as GA  # noqa: E402
import g2_refinalize as GR  # noqa: E402  (sets GA._graph = GR._graph_fixed, the D1-G2 Phase G repair)

assert GA._graph is GR._graph_fixed, "the D1-G2 _graph repair (g2_refinalize) is not active"
assert os.path.abspath(str(G.G2)) == ROOT, "g2_common.G2 is not the D1G2_SCRATCH root"
STOP_H = float(os.environ.get("G2C_STOP_AFTER_H", "44"))


def cmd_list():
    out = G2C / "lists" / "d1_targets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(job_id=d["job_id"], rxn_id=int(d["rxn_id"])) for d in G.targets()]
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"{len(rows)} D1 targets -> {out}", flush=True)
    return 0


def cmd_one(T, cores):
    return GA.cmd_one(T, cores)


def cmd_pool(lst, n_par, cores, i=0, m=1):
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(m)]
    lab = {d["job_id"]: d for d in G.targets()}
    t0 = time.time()
    print(f"[g2c_d1 pool] root {ROOT}: {len(ids)} reactions, {n_par} at a time x {cores} cores", flush=True)

    def one(T):
        fin = G.G2_FINAL / str(lab[T]["rxn_id"])
        if (fin / ".done").is_file() or list(fin.glob(".fail_*")):
            return T, "skip"
        if time.time() - t0 > STOP_H * 3600:
            return T, "not_started"
        log = G.LOGS / "rxn" / f"{T}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores)],
                               stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        fs = [p.name for p in fin.glob(".*")] if fin.is_dir() else []
        return T, (fs[0] if fs else f"error rc={r.returncode}")

    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for fu in as_completed([ex.submit(one, T) for T in ids]):
            T, s = fu.result()
            k = ".fail" if s.startswith(".fail") else s
            res[k] = res.get(k, 0) + 1
            print(f"  {T}: {s}", flush=True)
    print(f"[g2c_d1 pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["list"]:
        sys.exit(cmd_list())
    if a[:1] == ["one"] and len(a) == 3:
        sys.exit(cmd_one(a[1], a[2]))
    if a[:1] == ["pool"] and len(a) in (4, 6):
        sys.exit(cmd_pool(*a[1:]))
    raise SystemExit(__doc__)
