#!/usr/bin/env python3
"""tests/nur_prof.py — profile reading an inv_nurion extxyz.gz (sandbox only; writes only $MLT/tmp/nurtest)."""
import cProfile
import gzip
import pstats
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as C  # noqa: E402

src = Path(sys.argv[1])
tmp = C.MLT / "tmp" / "nurtest" / "prof_plain.extxyz"
with gzip.open(src, "rb") as f, open(tmp, "wb") as g:
    shutil.copyfileobj(f, g)
from ase.io import read  # noqa: E402
t0 = time.time()
a = read(str(tmp), index=":", format="extxyz")
print("plain ase read", len(a), round(time.time() - t0, 2), "s")
t0 = time.time()
a = read(str(src), index=":", format="extxyz")
print("gz ase read", len(a), round(time.time() - t0, 2), "s")
pr = cProfile.Profile()
pr.enable()
a = read(str(tmp), index=":200", format="extxyz")
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
tmp.unlink()
