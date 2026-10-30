#!/usr/bin/env python3
"""Phase U: GEO1b unit tests (GPU env).  python tests.py synth   (U-1..U-6, U-7s; no records needed)
                                          python tests.py data    (U-0, U-7, U-8b, U-9b, U-10; needs records.pkl)
Writes results/U_tests_<mode>.json; exit 3 if any test fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
import train as T  # noqa: E402
from metrics import evaluate_preds  # noqa: E402
from model import GeoRefiner  # noqa: E402

RES = {}


def check(name, ok, **info):
    RES[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def fake_rec(n, rng, uid, rid):
    syms = list(rng.choice(["C", "N", "O", "H"], n))
    x = rng.normal(0.0, 1.6, (n, 3))
    pairs = np.array([[0, 1], [2, 3]])
    B = GC.bond_types(syms, x, pairs)
    return dict(uid=uid, src="D1", src_id=0, rxn_id=rid, core="fake/A", fold=0, n=n,
                types=np.array([GC.TYPE[s] for s in syms]), lab=x + rng.normal(0, 0.05, x.shape),
                frag=np.array([1] * (n // 2) + [2] * (n - n // 2)), form=np.array([1, 1, 1, 1] + [0] * (n - 4)),
                pairs=pairs, charge=0, same=None, overlap=False, xyz_g2=x, lab_on_g2=x + 0.03, bond_g2=B,
                spd_g2=GC.spd_matrix(B), xyz_g1=None, map_rmsd_g2=0.1)


def batch_of(exs, double=True):
    """T.collate, then (double=True) x0 / y rebuilt in float64 from the examples (collate stores float32)."""
    b = T.collate(exs)
    if double:
        N = b["x0"].shape[1]
        for key in ("x0", "y"):
            z = torch.zeros((len(exs), N, 3), dtype=torch.float64)
            for i, e in enumerate(exs):
                z[i, :len(e["types"])] = torch.as_tensor(e[key], dtype=torch.float64)
            b[key] = z.to(T.DEV)
    return b


def model_random_head(double=True):
    torch.manual_seed(1)
    m = GeoRefiner(**T.SIZE["S"], drop=0.0).to(T.DEV).eval()
    torch.nn.init.normal_(m.head.w.weight, std=0.5)
    return m.double() if double else m


def synth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    R, t = T.rand_rot(rng), np.array([3.0, -1.0, 2.0])
    _, r1, f1 = GC.kabsch_align(X, X @ R.T + t)
    M = (X @ R.T + t) * np.array([-1.0, 1.0, 1.0])
    _, r2, f2 = GC.kabsch_align(X, M, reflect=True)
    _, r3, f3 = GC.kabsch_align(X, M, reflect=False)
    check("U-1", r1 < 1e-8 and not f1 and r2 < 1e-8 and f2 and not f3, rmsd_rot=r1, rmsd_mirror=r2)
    recs = [fake_rec(n, rng, f"D1_{i}", i) for i, n in enumerate((14, 23))]
    exs = [T.make_example((r, "g2"), False, 0.0, "mix", rng) for r in recs]
    m0 = GeoRefiner(**T.SIZE["S"]).to(T.DEV).eval().double()
    b = batch_of(exs)
    with torch.no_grad():
        out = m0(b, 2)
    check("U-2", float((out - b["x0"]).abs().max()) < 1e-10, max_diff=float((out - b["x0"]).abs().max()))
    m = model_random_head()
    e = exs[0]
    with torch.no_grad():
        o1 = m(batch_of([e]), 2)[0].cpu().numpy()
        err = 0.0
        for refl in (False, True):
            Rm = T.rand_rot(rng) @ (np.diag([-1.0, 1.0, 1.0]) if refl else np.eye(3))
            e2 = dict(e, x0=e["x0"] @ Rm.T + t)
            o2 = m(batch_of([e2]), 2)[0].cpu().numpy()
            err = max(err, float(np.abs(o2 - (o1 @ Rm.T + t)).max()))
    check("U-3", err < 1e-8, max_err=err)
    n = len(e["types"])
    p = rng.permutation(n)
    inv = np.argsort(p)
    e3 = dict(e, types=e["types"][p], frag=e["frag"][p], form=e["form"][p], spd=e["spd"][np.ix_(p, p)],
              bond=e["bond"][np.ix_(p, p)], x0=e["x0"][p], y=e["y"][p], pairs=inv[e["pairs"]])
    with torch.no_grad():
        o3 = m(batch_of([e3]), 2)[0].cpu().numpy()
    check("U-4", float(np.abs(o3 - o1[p]).max()) < 1e-8, max_err=float(np.abs(o3 - o1[p]).max()))
    with torch.no_grad():
        ob = m(batch_of(exs), 2).cpu().numpy()
        oa = [m(batch_of([x]), 2)[0].cpu().numpy() for x in exs]
    err5 = max(float(np.abs(ob[i, :len(x["types"])] - oa[i]).max()) for i, x in enumerate(exs))
    check("U-5", err5 < 1e-8, max_err=err5)
    qs = np.array([T.sample_q(rng) for _ in range(100000)])
    mid = qs[(qs > 0) & (qs < 1)]
    check("U-6", abs((qs == 1).mean() - 0.1) < 0.01 and abs((qs == 0).mean() - 0.8) < 0.01
          and mid.min() >= 0.4 and mid.max() <= 0.6, q1=(qs == 1).mean(), q0=(qs == 0).mean())
    # U-7s: a learnable equivariant target (uniform 3 % expansion about the centroid), 32 random molecules
    items = []
    for i in range(32):
        r = fake_rec(int(rng.integers(12, 30)), rng, f"D1_{100 + i}", 100 + i)
        x = r["xyz_g2"]
        r.update(lab_on_g2=x.mean(0) + 1.03 * (x - x.mean(0)))
        items.append((r, "g2"))
    start, end = overfit(items, 600)
    check("U-7s", end < 0.3 * start, start_L1=start, end_L1=end)


def overfit(items, n_steps, lr=1e-3):
    """train a dropout-free S model on the items (input = q 1, no noise); mean coordinate L1 before / after."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = GeoRefiner(**T.SIZE["S"], drop=0.0).to(T.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    def l1():
        P = T.predict(m, items, 2)
        return float(np.mean([np.abs(P[r["uid"]] - r[f"lab_on_{k}"]).mean() for r, k in items]))
    start, step = l1(), 0
    while step < n_steps:
        for _, b in T.batches(items, True, 0.0, "q1", rng, True):
            m.train()
            loss, _ = T.loss_fn(m(b, 2), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return start, l1()


def data():
    recs = T.load_records()
    rep = json.loads((GC.RES / "D_dataset_report.json").read_text())
    check("U-0", len(recs) == rep["n"] and rep.get("g1_used") is False, n=len(recs))
    # U-7 overfit (report only, as GEO1 decision U-7): 32 D1 training G2 samples
    tr, _, _ = T.split(recs, 0)
    items = [(r, "g2") for r in tr if r["src"] == "D1" and r["xyz_g2"] is not None][:32]
    start, end = overfit(items, 2000)
    RES["U-7"] = dict(ok=True, report_only=True, start_L1=start, end_L1=end, ratio=end / start)
    print(f"[REPORT] U-7 start {start:.4f} end {end:.4f} ratio {end / start:.3f}", flush=True)
    # U-8b metric regression on the G2-raw baseline, ALL para P_main pairs
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in (0, 1, 2, 3, 4) and r["xyz_g2"] is not None]
    mt = evaluate_preds({r["uid"]: r["xyz_g2"] for r in test}, test, with_rmsd=False)
    ex = GC.G2RAW_PRIMARY
    C = pd.read_csv(GC.C_COMPARE, usecols=["rxn_id", "d_dft_1", "d_dft_2", "d_g2_1", "d_g2_2"]).set_index("rxn_id")
    dev = 0.0
    for r in test[:300]:
        dl = sorted(GC.form_d(r["lab"], r["pairs"]))
        dg = sorted(GC.form_d(r["xyz_g2"], r["pairs"]))
        c = C.loc[r["rxn_id"]]
        dev = max(dev, float(np.abs(np.array(dl) - np.sort([c.d_dft_1, c.d_dft_2])).max()),
                  float(np.abs(np.array(dg) - np.sort([c.d_g2_1, c.d_g2_2])).max()))
    check("U-8b", mt["n_primary"] == ex["n"] and abs(mt["primary_share"] - ex["share"]) <= 0.005
          and abs(mt["pair_pe_median_para"] - ex["median"]) <= 0.001 and dev <= 1e-4 and len(test) == 8425,
          n=mt["n_primary"], share=mt["primary_share"], median=mt["pair_pe_median_para"], max_dist_dev=dev,
          n_test=len(test))
    # U-9b leakage + G1-free + every G2 input used
    bad = []
    if any(r.get("xyz_g1") is not None or r.get("same") is not None for r in recs):
        bad.append("a record carries G1 data (xyz_g1 / same)")
    for k in range(5):
        tr, va, te = T.split(recs, k)
        tr_ids = {r["uid"] for r in tr}
        if tr_ids & {r["uid"] for r in te + va}:
            bad.append(f"fold {k}: test/inner-val reaction in train")
        if any(r["src"] == "D0" and r["overlap"] for r in tr):
            bad.append(f"fold {k}: overlapping D0 in train")
        it = T.train_items(tr)
        if any(kind != "g2" for _, kind in it):
            bad.append(f"fold {k}: a non-G2 training sample")
        if len(it) != sum(r["xyz_g2"] is not None for r in tr):
            bad.append(f"fold {k}: train items {len(it)} != train records with G2")
        for r, kind in T.eval_items(te)[:200]:
            e = T.make_example((r, kind), False, 0.0, "mix", np.random.default_rng(0))
            if not np.allclose(e["x0"] + e["center"], r["xyz_g2"], atol=1e-10, rtol=0):
                bad.append(f"fold {k}: test input != G2 for {r['uid']}")
            syms = [GC.ELEM[t] for t in r["types"]]
            if not np.array_equal(GC.bond_types(syms, r["xyz_g2"], r["pairs"]), r["bond_g2"]):
                bad.append(f"fold {k}: bond graph not from the input for {r['uid']}")
    check("U-9b", not bad, problems=bad[:20])
    # U-10 mapped RMSD defined for >= 95 % of D1 G2 inputs (descriptive bins only)
    v = np.array([r["map_rmsd_g2"] for r in recs if r["src"] == "D1" and r["xyz_g2"] is not None], float)
    check("U-10", np.isfinite(v).mean() >= 0.95, finite=float(np.isfinite(v).mean()),
          median=float(np.nanmedian(v)), n=int(len(v)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"synth": synth, "data": data}.get(mode, lambda: sys.exit(__doc__))()
    GC.write_json(GC.RES / f"U_tests_{mode}.json", RES)
    sys.exit(0 if all(v["ok"] for v in RES.values()) else 3)
