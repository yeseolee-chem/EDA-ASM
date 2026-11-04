#!/usr/bin/env python3
"""tests/nur_corefold.py — READ-ONLY: which non-target D1 manifest rows have no core fold (sandbox only)."""
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inv_nurion as N  # noqa: E402

ctx = N.Ctx()
st = Counter()
ex = {}
for j, r in ctx.d1m.items():
    if j in ctx.job_rxn:
        continue
    core = f"{r['dipole']}/{r['skeleton']}"
    k = (r.get("kind"), core, core in ctx.cf)
    st[k] += 1
    ex.setdefault(k, j)
for k, v in sorted(st.items(), key=lambda x: -x[1]):
    print(v, k, ex[k])
print("cores with folds:", len(ctx.cf), sorted(ctx.cf)[:40])
