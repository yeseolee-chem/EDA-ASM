#!/usr/bin/env python3
"""Interim learning-curve check of the finished Phase R tune runs (user question 2026-10-28: are 300 epochs still too
few?). Reads $GEO/runs/tune_*.history.json; report only, no gate.  python diag_curves.py
Writes results/R_curves_interim.json and prints a table."""
from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo")) / "runs"
AT = (0, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299)
WIN = ((0, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300))
LR, WARM = 2e-4, 0.05


def lr_at(ep, n_ep):                                   # train.py schedule at the end of epoch ep (steps cancel)
    w = WARM * n_ep
    s = ep + 1
    return LR * s / w if s < w else LR * 0.5 * (1 + math.cos(math.pi * (s - w) / (n_ep - w)))


def ma(x, k=11):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def slope100(x):
    t = np.arange(len(x))
    return float(np.polyfit(t, x, 1)[0] * 100)


def one(p):
    H = json.loads(p.read_text())
    h = H["hist"]
    n = len(h)
    keys = [k for k in h[0] if k not in ("epoch", "wall_s", "val_primary", "val_rxn_median")]
    vp = np.array([e["val_primary"] for e in h])
    vm = np.array([e["val_rxn_median"] for e in h])
    n_pairs = int(np.lcm.reduce([Fraction(float(v)).limit_denominator(500).denominator for v in vp]))
    vp_ma, vm_ma = ma(vp), ma(vm)
    out = dict(
        tag=p.name.replace(".history.json", ""), cfg=H["cfg"], n_pass=H["n_pass"], n_train=H["n_train"], gpu=H["gpu"],
        n_epochs=n, best_ep=H["best_ep"], best=H["best"], n_val_pairs_inferred=n_pairs,
        epoch_s=float(np.median(np.diff([e["wall_s"] for e in h]))),
        val_primary_max=float(vp.max()), val_primary_argmax=int(vp.argmax()),
        val_primary_ma11_max=float(vp_ma.max()), val_primary_ma11_argmax=int(vp_ma.argmax()),
        val_primary_last=float(vp[-1]), val_rxn_median_last=float(vm[-1]),
        val_rxn_median_min=float(vm.min()), val_rxn_median_argmin=int(vm.argmin()),
        val_rxn_median_ma11_min=float(vm_ma.min()), val_rxn_median_ma11_argmin=int(vm_ma.argmin()),
        epochs_within_1pair_of_max=[int(i) for i in np.where(vp >= vp.max() - 1.0 / n_pairs - 1e-12)[0]],
        last50=dict(val_primary_mean=float(vp[-50:].mean()), val_primary_sd=float(vp[-50:].std()),
                    val_primary_slope_per100=slope100(vp[-50:]), val_rxn_median_mean=float(vm[-50:].mean()),
                    val_rxn_median_slope_per100=slope100(vm[-50:]),
                    **{f"train_{k}_slope_per100": slope100([e[k] for e in h[-50:]]) for k in keys}),
        windows={f"{a}-{b - 1}": dict(val_primary=float(vp[a:b].mean()), val_rxn_median=float(vm[a:b].mean()),
                                      **{f"train_{k}": float(np.mean([e[k] for e in h[a:b]])) for k in keys})
                 for a, b in WIN if a < n},
        at={str(i): dict(lr=lr_at(i, n), val_primary=float(vp[i]), val_rxn_median=float(vm[i]),
                         **{f"train_{k}": float(h[i][k]) for k in keys}) for i in AT if i < n},
    )
    return out


def main():
    R = [one(p) for p in sorted(RUNS.glob("tune_*.history.json"))]
    tmp = HERE / "results" / f"R_curves_interim.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(dict(note="interim, finished tune runs only (report only)", runs=R), indent=1))
    os.replace(tmp, HERE / "results" / "R_curves_interim.json")
    for r in R:
        print(f"\n== {r['tag']} ({r['cfg']}, n_pass {r['n_pass']}, {r['gpu']}, {r['epoch_s']:.0f} s/ep, "
              f"n_train {r['n_train']}, val pairs ~{r['n_val_pairs_inferred']}) best_ep {r['best_ep']} best {r['best']}")
        print(f"  val_primary max {r['val_primary_max']:.3f} @ {r['val_primary_argmax']}, ma11 max "
              f"{r['val_primary_ma11_max']:.3f} @ {r['val_primary_ma11_argmax']}, last {r['val_primary_last']:.3f}; "
              f"rxn_median min {r['val_rxn_median_min']:.4f} @ {r['val_rxn_median_argmin']}, last "
              f"{r['val_rxn_median_last']:.4f}")
        print("  last50:", {k: round(v, 5) for k, v in r["last50"].items()})
        print("  window        " + "  ".join(f"{k:>12s}" for k in next(iter(r["windows"].values()))))
        for w, d in r["windows"].items():
            print(f"  {w:>12s}  " + "  ".join(f"{v:12.5f}" for v in d.values()))
        print("  epoch   " + "  ".join(f"{k:>12s}" for k in r["at"]["0"]))
        for i, d in r["at"].items():
            print(f"  {i:>5s}   " + "  ".join(f"{v:12.5g}" for v in d.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
