#!/usr/bin/env python3
"""D1-G2 output compaction (quota: 289 / 300 GB on 2026-10-22). sbatch only (jobs/g2.slurm), after every Phase G
pool job and g2_refinalize.py have ended.

  python g2_compact.py run [workers]   # -> results/G_compact.json

For every final folder $G2/g2_final/<rxn_id>/ (all-digit name, .done):
  * ts.hess and work/ts.hess (finalize wrote two identical copies) -> one inode: work/ts.hess is replaced by a hardlink
    of ts.hess after a byte comparison (a mismatch is recorded and left alone);
  * work/ts.out -> work/ts.out.gz (gzip; the plain file is removed only after the decompressed bytes compare equal).
Readers of these folders: the 182-feature code reads work/ts.hess and result.json (ext_features.parse_hess and its
result.json self-check), FU2 atoms and D1-HF read ts.xyz / rel1.xyz / rel2.xyz / result.json; nothing reads
work/ts.out. Bytes of ts.hess are unchanged, so every reader sees the same data. Idempotent; *.tmp folders skipped.
"""
from __future__ import annotations

import filecmp
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402


def one(name):
    d = G.G2_FINAL / name
    out = dict(link="skip", gz="skip", saved=0)
    a, b, o = d / "ts.hess", d / "work" / "ts.hess", d / "work" / "ts.out"
    if a.is_file() and b.is_file():
        if os.stat(a).st_ino == os.stat(b).st_ino:
            out["link"] = "already"
        elif filecmp.cmp(a, b, shallow=False):
            tmp = b.with_name("ts.hess.lnk")
            if tmp.exists():
                tmp.unlink()
            os.link(a, tmp)
            out["saved"] += b.stat().st_size
            os.replace(tmp, b)
            out["link"] = "linked"
        else:
            out["link"] = "differ"
    if o.is_file():
        gz, tmp = o.with_name("ts.out.gz"), o.with_name("ts.out.gz.tmp")
        raw = o.read_bytes()
        with gzip.open(tmp, "wb") as f:
            f.write(raw)
        with gzip.open(tmp, "rb") as f:
            same = f.read() == raw
        if same:
            os.replace(tmp, gz)
            out["saved"] += len(raw) - gz.stat().st_size
            o.unlink()
            out["gz"] = "gzipped"
        else:
            tmp.unlink()
            out["gz"] = "verify_failed"
    elif (d / "work" / "ts.out.gz").is_file():
        out["gz"] = "already"
    return name, out


def cmd_run(workers=8):
    names = sorted(p.parent.name for p in G.G2_FINAL.glob("*/.done") if p.parent.name.isdigit())
    print(f"[compact] {len(names)} .done folders, {workers} workers", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        for name, r in ex.map(one, names, chunksize=64):
            res[name] = r
    cnt = lambda k: {v: sum(r[k] == v for r in res.values()) for v in sorted({r[k] for r in res.values()})}  # noqa: E731
    rep = dict(n=len(res), link=cnt("link"), gz=cnt("gz"), saved_GB=sum(r["saved"] for r in res.values()) / 1e9,
               problems={n: r for n, r in res.items() if r["link"] == "differ" or r["gz"] == "verify_failed"})
    G.write_json(G.RES / "G_compact.json", rep)
    print(rep, flush=True)
    return 3 if rep["problems"] else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        sys.exit(cmd_run(*(sys.argv[2:3] or [8])))
    raise SystemExit(__doc__)
