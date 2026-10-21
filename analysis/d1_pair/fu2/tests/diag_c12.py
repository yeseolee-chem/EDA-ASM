"""FU2 Phase C — C12 diagnostic for AW (user decision 2026-10-21: diagnose before deciding; nothing is changed).

  python fu2/tests/diag_c12.py         # GPU node, env_d1fu2 -> results/fu2/C12_diag.json

C12 failed for AW only (smallest cell L=1 / width 64: training MAE / null = 0.066 > 0.05 after 300 epochs on 64 pairs,
dropout 0, batch 8). Same protocol (production train.fit, variant A, T1 fold 0 training pairs), 3 random 64-pair subsets:
  * all 4 AW grid cells at 300 epochs (the smallest = the C12 condition; depth vs width)
  * the smallest cell at 1,000 epochs (training budget; the production cosine has T_max = 300 epochs, and past T_max
    PyTorch CosineAnnealingLR follows a periodic cosine (lr back to 1e-3 at epoch 600, ~2.5e-4 at 1,000) -> also run
    with T_max = 1,000 via a local override)
Training L1 at epochs 100 / 200 / 300 (/ 1,000) is recorded to see whether the fit is still improving.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fu2_common as F  # noqa: E402
import models as Mo  # noqa: E402
import train as T  # noqa: E402


def ratio(D, r, q):
    yh = T.predict(r["model"], r["T"], D.ra[q], D.rb[q])
    ys = D.y[q] / r["sy"]
    per = np.abs(yh - ys).mean(0) / np.maximum(np.abs(ys).mean(0), 1e-12)
    return float(np.abs(yh - ys).mean() / np.abs(ys).mean()), dict(zip(F.TARGETS, per.round(4).tolist()))


def curve(r, eps=(100, 200, 300, 1000)):
    h = {x["epoch"]: x["l1"] for x in r["history"]}
    return {str(e): h[e] for e in eps if e in h}


def main():
    T.require_gpu()
    t0 = time.time()
    D = T.Data()
    tr = D.split("T1", 0)["train"]
    out = dict(protocol="train.fit, variant A, dropout 0, batch 8, T1 fold 0 training pairs, 64-pair subsets",
               provenance=dict(code_sha256=T.CODE_SHA, q_report=D.q_report_sha, t_report=D.t_report_sha,
                               token_version=D.trep.get("token_version")),
               threshold=0.05, gpu=T.versions()["gpu"], subsets={})
    for s in range(3):
        q = np.random.default_rng(F.SEED + 500 + s).choice(tr, size=64, replace=False)
        rec = {}
        for ci, cell in enumerate(Mo.GRID["AW"]):
            r = T.fit(D, "AW", cell, "A", q, None, seed=F.SEED, n_epochs=300, dropout=0.0, batch_size=8)
            rt, per = ratio(D, r, q)
            rec[f"cell{ci}_L{cell['n_layers']}_W{cell['width']}_300ep"] = dict(ratio=rt, ratio_per_target=per, l1_curve=curve(r),
                                                                               seconds=r["seconds"])
            print(f"[subset {s}] AW {cell} 300 ep: ratio {rt:.4f}", flush=True)
        # smallest cell, 1,000 epochs: (a) production scheduler (T_max = 300: periodic cosine), (b) T_max = 1,000
        r = T.fit(D, "AW", Mo.GRID["AW"][0], "A", q, None, seed=F.SEED, n_epochs=1000, dropout=0.0, batch_size=8)
        rt, per = ratio(D, r, q)
        rec["cell0_1000ep_Tmax300"] = dict(ratio=rt, ratio_per_target=per, l1_curve=curve(r), seconds=r["seconds"])
        print(f"[subset {s}] AW cell0 1000 ep (T_max 300): ratio {rt:.4f}", flush=True)
        old = F.MAX_EPOCHS
        try:
            F.MAX_EPOCHS = 1000                                      # local override for this diagnostic run only
            r = T.fit(D, "AW", Mo.GRID["AW"][0], "A", q, None, seed=F.SEED, n_epochs=1000, dropout=0.0, batch_size=8)
        finally:
            F.MAX_EPOCHS = old
        rt, per = ratio(D, r, q)
        rec["cell0_1000ep_Tmax1000"] = dict(ratio=rt, ratio_per_target=per, l1_curve=curve(r), seconds=r["seconds"])
        print(f"[subset {s}] AW cell0 1000 ep (T_max 1000): ratio {rt:.4f}", flush=True)
        out["subsets"][str(s)] = rec
    out["seconds"] = time.time() - t0
    F.write_json(F.FU2_RES / "C12_diag.json", out)
    print(json.dumps({s: {k: round(v["ratio"], 4) for k, v in rec.items()} for s, rec in out["subsets"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
