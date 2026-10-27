#!/usr/bin/env python3
"""Decision U-7: diagnostic reruns of the spec U-7 overfit test (same 32 D1 training G1 samples, dropout 0), varying only
the learning rate / number of steps; report only, no gate.  GPU env:  python diag_u7.py
Writes results/U7_diag.json."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
import tests  # noqa: E402
import train as T  # noqa: E402


def main():
    recs = T.load_records()
    tr, _, _ = T.split(recs, 0)
    items = [(r, "g1") for r in tr if r["src"] == "D1" and r["xyz_g1"] is not None][:32]     # = tests.data U-7
    out = {}
    for name, lr, steps in (("spec_lr1e-3_2000", 1e-3, 2000), ("lr2e-4_2000", 2e-4, 2000),
                            ("lr1e-3_8000", 1e-3, 8000), ("lr2e-4_8000", 2e-4, 8000)):
        start, end = tests.overfit(items, steps, lr=lr)
        out[name] = dict(lr=lr, steps=steps, start_L1=start, end_L1=end, ratio=end / start)
        print(name, out[name], flush=True)
        GC.write_json(GC.RES / "U7_diag.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
