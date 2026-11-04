#!/usr/bin/env python3
"""tests/nur_verify_all.py — sandbox only, READ-ONLY on outputs: inv_nurion verify for every done tar (parallel) + a
frame count / partial ase read of the collected $MLT/dft/nurion.extxyz.gz vs results/N_report.json. Prints JSON."""
import argparse
import contextlib
import gzip
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inv_nurion as N  # noqa: E402
import mlt_common as C  # noqa: E402


def one(stem):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = N.cmd_verify(argparse.Namespace(tar=stem, root=None))
    return stem, rc, json.loads(buf.getvalue())


def collected():
    rep = C.read_json(N.RES_N)
    n, lines = 0, 0
    head = []
    with gzip.open(N.COLLECT_EXT, "rt") as f:
        while True:
            ln = f.readline()
            if not ln:
                break
            k = int(ln.split()[0])
            block = [ln] + [f.readline() for _ in range(k + 1)]
            if n < 300:
                head.extend(block)
            n += 1
            lines += k + 2
    tmp = C.MLT / "tmp" / "nurtest" / "collect_head.extxyz"
    tmp.write_text("".join(head))
    al = C.read_extxyz(tmp)
    tmp.unlink()
    ok_tags = all(isinstance(a.info["rxn_ids"], str) and isinstance(a.info["folds"], str) for a in al)
    return dict(frames=n, report_n=rep["n_structures"], frames_eq_report=n == rep["n_structures"],
                head_read=len(al), head_tags_str=ok_tags, head_kinds=sorted({a.info["kind"] for a in al}))


def main():
    rows, _ = N.tar_states()
    done = [r["tar"] for r in rows if r["state"] == "done"]
    out = {}
    with ProcessPoolExecutor(max_workers=min(13, len(done))) as ex:
        fc = ex.submit(collected)
        for stem, rc, r in ex.map(one, done):
            out[stem] = dict(rc=rc, n=r["n_read"], kinds_match=r["kinds_match"], problems=r["problems"],
                             hess_label=r["hess_label"], hess_other_ok=r["hess_other"] == r["hess_other_report"])
        out["collect"] = fc.result()
    print(json.dumps(out, indent=1))
    return 0 if all(v.get("rc", 0) == 0 for v in out.values()) and out["collect"]["frames_eq_report"] else 3


if __name__ == "__main__":
    sys.exit(main())
