#!/usr/bin/env python3
"""tests/nur_probe.py <tar> <job> [<job> ...] — READ-ONLY probe of OptTS .00k.hess files (sandbox only; writes nothing
on GPFS): for every TS_*_optts*_orca.out of the jobs, the %geom block of the input echo, and for each .hess of that
run its geometry vs the run's steps (which cycle it belongs to, Kabsch) and its max |dH| vs the TS-guess Freq."""
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import inv_nurion as N  # noqa: E402
import mlt_common as C  # noqa: E402
import orca_io as O  # noqa: E402


def main():
    tar, jobs = sys.argv[1], set(sys.argv[2:])
    reg = N.registry()
    stem, ent = N.resolve(tar, reg)
    out = {}
    tmp = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        with tarfile.open(N.UPLOAD / ent["file"], "r|*") as tf:
            for m in tf:
                mm = N.RE_UNIT.match(m.name)
                if not mm or mm.group(2) not in jobs or mm.group(3) != "archive.tgz":
                    if out and len(out) == len(jobs):
                        break
                    continue
                j = mm.group(2)
                d = tmp / j
                arc = tarfile.open(fileobj=io.BytesIO(tf.extractfile(m).read()), mode="r:gz")
                inner = arc.extractfile("ts.tar").read() if "ts.tar" in arc.getnames() else None
                src = tarfile.open(fileobj=io.BytesIO(inner), mode="r:") if inner else arc
                for x in src.getmembers():
                    if x.isfile() and "/transition_states/" in x.name and "/truncated/" not in x.name and \
                            (x.name.endswith((".hess", "_orca.out")) and ("optts" in x.name or x.name.endswith("_hess_orca.hess")
                                                                        or x.name.endswith("_hess_orca.out"))):
                        p = d / Path(x.name).name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(src.extractfile(x).read())
                res = {}
                guess = {f.name: O.read_hess(f) for f in d.glob("*_hess_orca.hess")}
                for f in sorted(d.glob("TS_*_optts*_orca.out")):
                    echo = O.input_echo(f)
                    geom = re.findall(r"(?is)%geom.*?end", echo)
                    steps, info = O.read_out_steps(f)
                    base = f.name[:-4]
                    r = dict(geom=[" ".join(g.split()) for g in geom], n_steps=len(steps), inhess=bool(re.search(r"(?i)inhess", echo)))
                    hs = {}
                    for h in sorted(d.glob(base + "*.hess")):
                        hh = O.read_hess(h)
                        pos = hh["pos_bohr"] * C.BOHR2A
                        dev = [N.kabsch(pos, s["pos_A"])[1] for s in steps]
                        k = int(np.argmin(dev)) if dev else None
                        g = {gn: float(np.abs(hh["hessian"] - gv["hessian"]).max()) for gn, gv in guess.items()
                             if gv["hessian"].shape == hh["hessian"].shape}
                        gd = {gn: N.kabsch(pos, gv["pos_bohr"] * C.BOHR2A)[1] for gn, gv in guess.items()
                              if len(gv["symbols"]) == len(pos)}
                        fr = N.freqs_cm(hh["hessian"], hh["masses"])
                        hs[h.name] = dict(closest_cycle=(k + 1) if k is not None else None,
                                          dev_closest_A=float(dev[k]) if dev else None,
                                          dev_step1_A=float(dev[0]) if dev else None, max_dH_vs_guess=g,
                                          geom_dev_vs_guess_A=gd, n_imag=int((fr < -50).sum()))
                    r["hess"] = hs
                    res[f.name] = r
                out[j] = res
                shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out, indent=1, default=C._json_default))


if __name__ == "__main__":
    main()
