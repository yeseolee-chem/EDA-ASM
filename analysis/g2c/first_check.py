#!/usr/bin/env python3
"""G2c first check (docs/specs/GEO1B.md 5-3 step 5, report only; STOP if different).
For each rerun root, the first 2 reactions (job_id order) with a .done final folder: the set of ORCA '!' lines over all
.inp files in jobs/<T>/inputs.tar.gz must equal the set in the original run of the same reaction (d1_g2 / d0_g2).
D1 run env or any python3, sbatch only.   python first_check.py d1_r1 [d1_r2 d0_r1 d0_r2]
Writes results/C_first_check.json (merged over calls)."""
from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
G2C = Path("/gpfs/tmp_cpu2/yeseo1ee/g2c")
ORIG = {"d1": Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2"), "d0": Path("/gpfs/tmp_cpu2/yeseo1ee/d0_g2")}


def bang(tar):
    out = set()
    with tarfile.open(tar) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".inp"):
                for ln in t.extractfile(m).read().decode(errors="replace").splitlines():
                    if ln.strip().startswith("!"):
                        out.add(" ".join(ln.split()))
    return sorted(out)


def check(rootname):
    src = rootname.split("_")[0]
    root = G2C / rootname
    jobs = sorted(p.name for p in (root / "jobs").iterdir() if (p / "inputs.tar.gz").is_file()) if (root / "jobs").is_dir() else []
    rows = []
    for T in jobs:
        a, b = root / "jobs" / T / "inputs.tar.gz", ORIG[src] / "jobs" / T / "inputs.tar.gz"
        if not b.is_file():
            continue
        if not any((root / "jobs" / T).glob(".done_ts")) or not any((ORIG[src] / "jobs" / T).glob(".done_ts")):
            continue                                       # both runs must have reached the TS stage (else no '!' lines)
        la, lb = bang(a), bang(b)
        rows.append(dict(job_id=T, same=la == lb, new=la, orig=lb))
        if len(rows) == 2:
            break
    return dict(root=rootname, n_checked=len(rows), all_same=bool(rows) and all(r["same"] for r in rows), rows=rows)


def main(roots):
    out = HERE / "results" / "C_first_check.json"
    rep = json.loads(out.read_text()) if out.is_file() else {}
    for r in roots:
        rep[r] = check(r)
        print(r, rep[r]["n_checked"], "all_same" if rep[r]["all_same"] else "DIFFERENT/none", flush=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out)
    bad = [r for r in roots if rep[r]["n_checked"] == 2 and not rep[r]["all_same"]]
    return 3 if bad else 0


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
