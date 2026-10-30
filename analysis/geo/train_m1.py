#!/usr/bin/env python3
"""GEO2 Phase Y: one extra GEO1 refiner for the D1 reactions of T1 fold -1 (docs/specs/GEO2.md 3).

The 380 D1 reactions of the six T1 fold -1 cores are never a GEO1 test fold, so no GEO1 model left them out. Their
pairs are LADDER training pairs of every fold, so the geo rung needs their structures from a model that did not see them.
  python train_m1.py plan          -> results/Y_plan.json (config = most frequent R_select choice, ties c0 < c1 < c2 < c3;
                                      epochs = median of the 5 chosen best epochs + 1, rounded down)
  python train_m1.py final <seed>  -> $GEO/preds/final_fm1_s<seed>.pkl (seeds 0, 1, 2)
Training set = every D1 reaction with fold >= 0 + D0 (overlap False); no inner validation (fixed epochs, as 'final').
GPU env (jobs/gpu.slurm). Uses train.py unchanged (fit / predict / train_items / eval_items / load_records).
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
import train as T  # noqa: E402


def plan():
    sel = [json.loads((GC.RES / f"R_select_f{k}.json").read_text()) for k in range(5)]
    cnt = Counter(s["chosen"] for s in sel)
    top = max(cnt.values())
    cfg = sorted(c for c, v in cnt.items() if v == top)[0]
    ep = int(np.floor(np.median([s["best_ep"] + 1 for s in sel])))
    out = dict(cfg=cfg, epochs=ep, chosen=[s["chosen"] for s in sel], best_ep=[s["best_ep"] for s in sel],
               rule="cfg = most frequent R_select choice (ties: lowest name); epochs = floor(median(best_ep + 1))")
    GC.write_json(GC.RES / "Y_plan.json", out)
    print(out, flush=True)
    return 0


def final(seed):
    p = json.loads((GC.RES / "Y_plan.json").read_text())
    recs = T.load_records()
    test = [r for r in recs if r["src"] == "D1" and r["fold"] == -1]
    train = [r for r in recs if (r["src"] == "D1" and r["fold"] >= 0) or (r["src"] == "D0" and not r["overlap"])]
    assert test and not ({r["uid"] for r in test} & {r["uid"] for r in train}), "fold -1 leak"
    ema, arch, n_pass, _ = T.fit(T.train_items(train), None, p["cfg"], int(seed), fixed_epochs=int(p["epochs"]),
                                 tag=f"final_fm1_s{seed}")
    P = T.predict(ema.model(arch), T.eval_items(test), n_pass)
    out = GC.GEO / "preds" / f"final_fm1_s{seed}.pkl"
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(P, f, protocol=4)
    tmp.replace(out)
    print(f"fold -1: {len(P)} reactions -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["plan"]:
        sys.exit(plan())
    if len(a) == 2 and a[0] == "final":
        sys.exit(final(a[1]))
    raise SystemExit(__doc__)
