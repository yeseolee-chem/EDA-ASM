"""FU2 Phase C — automatic code checks C1-C12 (docs/specs/D1PAIR_FU2.md, Phase C) + C13 / C14 (user review F5)
-> results/fu2/C_tests.json.

  python fu2/tests/run_tests.py         # GPU node (gpu4 / gpu5, RTX A6000), env_d1fu2 (fu2/jobs/fu2.slurm PYENV=fu2);
                                        # exit 3 if any check fails

Every check runs the production code (fu2/train.py Data / Std / batch / orient / epoch_batches / loss_fn / fit /
run_unit / gate_m1, fu2/models.py). Invariance checks (C1-C5, C10) use float64 copies of briefly trained models with
random noise added to every parameter (the properties must hold for any weights; non-vacuity and negative controls are
recorded and gated). Tolerances in kcal/mol unless stated.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fu2_common as F  # noqa: E402
import models as Mo  # noqa: E402
import train as T  # noqa: E402

C = F.C
TOL = 1e-4
RES = {}
COV_Z = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 17: 1.02, 35: 1.20, 16: 1.05, 53: 1.39, 14: 1.11}  # fragmenter.COV + Si


def record(name, ok, **kw):
    RES[name] = dict(ok=bool(ok), **kw)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {json.dumps(kw, default=str)[:1500]}", flush=True)


def trained(D, m, v, cell, rows):
    r = T.fit(D, m, cell, v, rows, None, seed=F.SEED, n_epochs=1)
    return r["model"], r["std"]


def noisy64(model, seed, scale=0.3):
    """float64 eval copy with N(0, scale^2) added to every parameter."""
    mm = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        for k, p in enumerate(mm.parameters()):
            g = torch.Generator(device=p.device).manual_seed(seed + k)
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype, device=p.device))
    return mm


def g_of(model, R, ia, ib, sy):
    with torch.no_grad():
        return model(R, ia, ib).cpu().numpy() * sy


def clone_R(R):
    return {k: v.clone() for k, v in R.items()}


def pad_R(R, nmax, rng):
    """pad the atom axis to nmax with random finite x / pos values (mask False, no edges): masking must hide them."""
    u, n = R["mask"].shape
    p = nmax - n
    dev, dt = R["x"].device, R["x"].dtype
    rx = torch.tensor(rng.normal(size=(u, p, R["x"].shape[2])), dtype=dt, device=dev)
    rp = torch.tensor(rng.normal(size=(u, p, 3)) * 10, dtype=dt, device=dev)
    z = lambda t, shape: torch.zeros(shape, dtype=t.dtype, device=t.device)   # noqa: E731
    et = torch.cat([torch.cat([R["etype"], z(R["etype"], (u, n, p))], 2), z(R["etype"], (u, p, nmax))], 1)
    return dict(x=torch.cat([R["x"], rx], 1), pos=torch.cat([R["pos"], rp], 1), mask=torch.cat([R["mask"], z(R["mask"], (u, p))], 1),
                etype=et, frag=torch.cat([R["frag"], z(R["frag"], (u, p))], 1), glob=R["glob"])


def main():
    T.require_gpu()
    t_start = time.time()
    rng = np.random.default_rng(F.SEED)
    D = T.Data()
    sp = D.split("T1", 0)
    tr, te = sp["train"], sp["test"]
    sub = rng.choice(tr, size=min(1024, len(tr)), replace=False)
    configs = [(m, v) for m in Mo.MODELS for v in Mo.VARIANTS]
    models = {}
    for i, (m, v) in enumerate(configs):
        mod, std = trained(D, m, v, Mo.GRID[m][0], sub)
        models[(m, v)] = (noisy64(mod, F.SEED + 1000 * i), std)
    T64 = {}

    def tens(std):
        if id(std) not in T64:
            T64[id(std)] = std.tensors(D, dtype=torch.float64)
        return T64[id(std)]

    pick = rng.choice(te, size=min(200, len(te)), replace=False)
    # ---- C1 antisymmetry g(b,a) = -g(a,b), g(a,a) = 0 (float64, 200 test pairs)
    out = {}
    for (m, v), (mod, std) in models.items():
        Tt = tens(std)
        ra, rb = D.ra[pick], D.rb[pick]
        gab, gba, gaa = (T.predict(mod, Tt, x, y, bs=32) * std.sy for x, y in ((ra, rb), (rb, ra), (ra, ra)))
        out[f"{m}/{v}"] = dict(anti_max=float(np.abs(gab + gba).max()), self_max=float(np.abs(gaa).max()),
                               median_abs_g=float(np.median(np.abs(gab))))
    record("C1", all(o["anti_max"] <= TOL and o["self_max"] <= TOL and o["median_abs_g"] >= 1e-2 for o in out.values()),
           n_pairs=len(pick), tol=TOL, nonvacuity_min_median_abs_g=1e-2, configs=out)

    # ---- C2 atom order, C3 rotation + translation, C4 padding, C5 batch independence (+ negative controls)
    p30 = pick[:30]
    big = int(np.argmax(D.n))
    c = {k: {} for k in ("C2", "C3", "C4", "C5")}
    for (m, v), (mod, std) in models.items():
        Tt = tens(std)
        d = dict(C2=0.0, C3=0.0, C4=0.0, C5=0.0)
        neg2 = neg3 = 0
        for q in p30:
            a, b = D.ra[q], D.rb[q]
            R, ia, ib = T.batch(Tt, np.array([a]), np.array([b]))
            g0 = g_of(mod, R, ia, ib, std.sy)
            j = int(ia[0]); n = int(R["mask"][j].sum())
            for side in (ia, ib):
                js = int(side[0]); ns = int(R["mask"][js].sum())
                perm = torch.as_tensor(rng.permutation(ns), device=R["x"].device)
                Rp = clone_R(R)
                for k in ("x", "pos", "mask", "frag"):
                    Rp[k][js, :ns] = R[k][js, :ns][perm]
                Rp["etype"][js, :ns, :ns] = R["etype"][js, :ns, :ns][perm][:, perm]
                d["C2"] = max(d["C2"], float(np.abs(g_of(mod, Rp, ia, ib, std.sy) - g0).max()))
                Q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device=R["x"].device))
                if torch.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                Rr = clone_R(R)
                Rr["pos"][js, :ns] = R["pos"][js, :ns] @ Q.T + torch.randn(3, dtype=torch.float64, device=Q.device) * 10
                d["C3"] = max(d["C3"], float(np.abs(g_of(mod, Rr, ia, ib, std.sy) - g0).max()))
            # negative controls on side a: x permuted alone (tokens no longer match positions / edges); non-rigid scaling
            perm = torch.as_tensor(rng.permutation(n), device=R["x"].device)
            Rn = clone_R(R); Rn["x"][j, :n] = R["x"][j, :n][perm]
            neg2 += float(np.abs(g_of(mod, Rn, ia, ib, std.sy) - g0).max()) > 1e-3
            Rs = clone_R(R); Rs["pos"][j, :n] = R["pos"][j, :n] * 1.05
            neg3 += float(np.abs(g_of(mod, Rs, ia, ib, std.sy) - g0).max()) > 1e-3
            # C4: production batch with the largest reaction in it, and explicit padding with random finite values
            Rb, iab, ibb = T.batch(Tt, np.array([a, big]), np.array([b, big]))
            d["C4"] = max(d["C4"], float(np.abs(g_of(mod, Rb, iab, ibb, std.sy)[:1] - g0).max()),
                          float(np.abs(g_of(mod, pad_R(R, D.nmax, rng), ia, ib, std.sy) - g0).max()))
            others = rng.choice(tr, size=63, replace=False)
            R5, i5a, i5b = T.batch(Tt, np.r_[D.ra[others], a], np.r_[D.rb[others], b])
            d["C5"] = max(d["C5"], float(np.abs(g_of(mod, R5, i5a, i5b, std.sy)[-1:] - g0).max()))
        for k in c:
            c[k][f"{m}/{v}"] = d[k]
        c["C2"][f"{m}/{v}/neg_control_changed_frac"] = neg2 / len(p30)
        c["C3"][f"{m}/{v}/neg_control_changed_frac"] = neg3 / len(p30)
    for name in ("C2", "C3", "C4", "C5"):
        diffs = {k: v for k, v in c[name].items() if not k.endswith("neg_control_changed_frac")}
        negs = {k: v for k, v in c[name].items() if k.endswith("neg_control_changed_frac")}
        record(name, max(diffs.values()) <= TOL and all(v >= 0.9 for v in negs.values()), n_pairs=len(p30), tol=TOL,
               max_diff=diffs, negative_controls=negs or None)

    # ---- C13 (review F5): changing only the global token changes A+G outputs and leaves A outputs unchanged
    c13 = {}
    rng13 = np.random.default_rng(F.SEED + 13)                      # own generator: C1-C12 keep their original samples
    for (m, v), (mod, std) in models.items():
        Tt = tens(std)
        n_changed, dmax = 0, 0.0
        for q in p30:
            R, ia, ib = T.batch(Tt, np.array([D.ra[q]]), np.array([D.rb[q]]))
            g0 = g_of(mod, R, ia, ib, std.sy)
            Rg = clone_R(R)
            Rg["glob"][ia] = R["glob"][ia] + torch.tensor(rng13.normal(size=R["glob"].shape[1]), dtype=R["glob"].dtype,
                                                          device=R["glob"].device)
            d = float(np.abs(g_of(mod, Rg, ia, ib, std.sy) - g0).max())
            dmax = max(dmax, d)
            n_changed += d > 1e-6
        c13[f"{m}/{v}"] = dict(max_diff=dmax, changed_frac=n_changed / len(p30))
    # the production global token = the reaction's own X182_d1 row (after un-standardising), not constant
    X182 = pd.read_parquet(F.X182).set_index("rxn_id")[D.trep["glob_cols"]]
    st0 = models[configs[0]][1]
    Tg = tens(st0)["glob"].cpu().numpy() * st0.gsd + st0.gmu
    rows10 = rng13.choice(len(D.rids), size=10, replace=False)
    glob_rel = max(float(np.max(np.abs(Tg[k] - X182.loc[int(D.rids[k])].to_numpy(float))
                                / np.maximum(1.0, np.abs(X182.loc[int(D.rids[k])].to_numpy(float))))) for k in rows10)
    glob_nonconst = bool(np.ptp(Tg, axis=0).max() > 0)
    record("C13", all((c13[f"{m}/A"]["max_diff"] <= 1e-12) and (c13[f"{m}/A+G"]["changed_frac"] >= 0.9) for m in Mo.MODELS)
           and glob_rel <= 1e-9 and glob_nonconst, n_pairs=len(p30), tol_A=1e-12, AG_changed_threshold=1e-6, configs=c13,
           glob_vs_X182_rel_maxdiff=glob_rel, glob_nonconstant=glob_nonconst)

    # ---- C6 label alignment: loader y = F_pairs dd (a - b) and the training batches' y signs follow the order swap
    Fp = pd.read_csv(C.RES / "F_pairs.csv").set_index("pair_id")
    q50 = rng.choice(tr, size=50, replace=False)
    n_ok = 0
    dd = {}
    for q in q50:
        f = Fp.loc[D.P.index[q]]
        same_rxn = int(f.a_rxn) == int(D.rids[D.ra[q]]) and int(f.b_rxn) == int(D.rids[D.rb[q]])
        yv = np.array([float(f[f"dd_{t}"]) for t in F.TARGETS])
        n_ok += bool(same_rxn and np.allclose(D.y[q], yv, rtol=0, atol=1e-12))
        dd[(int(f.a_rxn), int(f.b_rxn))] = yv
    sy0 = models[configs[0]][1].sy
    ytr = torch.tensor(D.y[q50] / sy0, dtype=torch.float64)
    seen, n_swap, n_keep, n_sign_ok = 0, 0, 0, 0
    for ra, rb, y in T.epoch_batches(np.random.default_rng(F.SEED), D.ra[q50], D.rb[q50], ytr, 16):
        for x_a, x_b, yy in zip(ra, rb, y.numpy() * sy0):
            key = (int(D.rids[x_a]), int(D.rids[x_b]))
            seen += 1
            if key in dd:
                n_keep += 1; n_sign_ok += bool(np.allclose(yy, dd[key], rtol=0, atol=1e-9))
            elif key[::-1] in dd:
                n_swap += 1; n_sign_ok += bool(np.allclose(yy, -dd[key[::-1]], rtol=0, atol=1e-9))
    record("C6", n_ok == 50 and n_sign_ok == 50 and seen == 50 and n_swap >= 1 and n_keep >= 1, loader_ok=n_ok,
           batch_sign_ok=n_sign_ok, n=50, n_swapped=n_swap, n_kept=n_keep)

    # ---- C7 / C8: statistics only from training rows; tuning validation = first inner fold (core-disjoint)
    c7, c8 = {}, {}
    from sklearn.model_selection import GroupKFold
    core = D.P.core.to_numpy()
    for k in range(5):
        s = D.split("T1", k)
        rx = lambda rows: set(D.ra[rows]) | set(D.rb[rows])           # noqa: E731
        for stage, rows in (("final", s["train"]), ("tuning", s["inner_train"])):
            st = T.Std(D, rows)
            c7[f"f{k}/{stage}"] = dict(stat_rxns_in_test=len(set(st.rxn_rows) & rx(s["test"])),
                                       stat_pairs_in_test=len(set(st.pair_rows) & set(s["test"])),
                                       stat_rxns_outside_train=len(set(st.rxn_rows) - rx(s["train"])))
        rec = F.read_json(F.SPLITS / f"T1__f{k}.json")
        st8 = {key: set(rec[key]) for key in ("train", "test", "inner_train", "inner_val")}
        cores = D.P.loc[rec["train"], "core"].to_numpy()
        _, b_ = next(iter(GroupKFold(C.N_INNER).split(rec["train"], groups=cores)))
        c8[f"f{k}"] = dict(cover=st8["inner_train"] | st8["inner_val"] == st8["train"],
                           inner_overlap=len(st8["inner_train"] & st8["inner_val"]),
                           val_in_test=len(st8["inner_val"] & st8["test"]),
                           val_core_in_inner_train=len(set(D.P.loc[rec["inner_val"], "core"]) &
                                                       set(D.P.loc[rec["inner_train"], "core"])),
                           val_fraction=len(st8["inner_val"]) / len(st8["train"]),
                           info_groupkfold_recomputed_equal=[rec["train"][i] for i in b_] == rec["inner_val"])
    calls, real_fit = [], T.fit

    def spy(Dx, m, cell, v, trr, val=None, **kw):
        kw["n_epochs"] = 1
        r = real_fit(Dx, m, cell, v, trr, val, **kw)
        calls.append(dict(tr=np.asarray(trr).copy(), val=None if val is None else np.asarray(val).copy(),
                          std_pairs=r["std"].pair_rows.copy(), mu=r["std"].mu.copy(), sd=r["std"].sd.copy(),
                          x=r["T"]["x"][:, :, -len(D.num_cols):].double().cpu().numpy(),
                          r=r if val is None else None))                      # final fit kept for C14
        return r

    T.fit = spy
    c14 = {}
    try:
        with tempfile.TemporaryDirectory(dir=F.FU2_SCR) as tmp:
            u14 = dict(split="T1", model="AW", variant="A", fold=0)
            T.run_unit(D, u14, root=Path(tmp), n_epochs=1, seeds=F.FINAL_SEEDS[:1], cells=[0], log=None)
            # ---- C14 (review F5): the unit parquet as written: y = F_pairs dd by pair_id, yhat = predict x s_y
            pq = pd.read_parquet(Path(tmp) / f"{T.uname(u14)}.parquet")
            rf = calls[-1]["r"]
            yh = T.predict(rf["model"], rf["T"], D.ra[te], D.rb[te]) * rf["sy"]
            want = pd.DataFrame(yh, columns=F.TARGETS, index=D.P.index[te]).stack().rename("want")
            want.index.names = ["pair_id", "target"]
            got = pq.set_index(["pair_id", "target"])
            jn = got.join(want, how="outer")
            dd = Fp[[f"dd_{t}" for t in F.TARGETS]].rename(columns=lambda c: c[3:]).stack().rename("dd")
            dd.index.names = ["pair_id", "target"]
            jn = jn.join(dd, how="left")
            rk = pq.drop_duplicates("pair_id").set_index("pair_id").ref_kind
            c14 = dict(n_rows=len(pq), n_expected=len(te) * len(F.TARGETS),
                       rows_unmatched=int(jn.want.isna().sum() + jn.yhat.isna().sum() + jn.y.isna().sum() + jn.dd.isna().sum()),
                       y_vs_F_pairs_maxdiff=float(np.abs(jn.y - jn.dd).max(skipna=False)),
                       yhat_vs_predict_maxdiff=float(np.abs(jn.yhat - jn.want).max(skipna=False)),
                       ref_kind_equal_F_pairs=bool((rk == Fp.loc[rk.index, "ref_kind"]).all()),
                       unit_columns_ok=bool((pq.fold == u14["fold"]).all() and (pq.split == u14["split"]).all()
                                            and (pq.model == u14["model"]).all() and (pq.variant == u14["variant"]).all()),
                       seed_sd_zero_for_one_seed=bool((pq.yhat_seed_sd == 0).all()),
                       pair_ids_equal_test=sorted(pq.pair_id.unique()) == sorted(D.P.index[te]))
    finally:
        T.fit = real_fit
    record("C14", c14.get("n_rows") == c14.get("n_expected") and c14.get("rows_unmatched") == 0
           and c14.get("y_vs_F_pairs_maxdiff", 1) <= 1e-12 and c14.get("yhat_vs_predict_maxdiff", 1) <= 1e-6
           and c14.get("pair_ids_equal_test", False) and c14.get("ref_kind_equal_F_pairs", False)
           and c14.get("unit_columns_ok", False), tol_y=1e-12, tol_yhat=1e-6, **c14)
    tune_call, final_call = calls[0], calls[1]
    stat_ok = []
    for cl, rows in ((tune_call, sp["inner_train"]), (final_call, sp["train"])):
        rr = np.unique(np.r_[D.ra[rows], D.rb[rows]])
        vals = D.num[rr][D.mask[rr]]
        mu, sd = vals.mean(0), vals.std(0)
        sd[sd < C.SCALE_EPS] = 1.0
        five = rr[:5]
        fed = cl["x"][five][D.mask[five]]
        want = ((D.num[five] - mu) / sd)[D.mask[five]]
        stat_ok.append(dict(rows_equal=set(cl["std_pairs"]) == set(rows), mu_maxdiff=float(np.abs(cl["mu"] - mu).max()),
                            sd_maxdiff=float(np.abs(cl["sd"] - sd).max()), fed_x_maxdiff=float((np.abs(fed - want) / np.maximum(1.0, np.abs(want))).max())))
    c8["run_unit"] = dict(tuning_val_eq_inner_val=sorted(tune_call["val"]) == sorted(sp["inner_val"]),
                          tuning_train_eq_inner_train=sorted(tune_call["tr"]) == sorted(sp["inner_train"]),
                          tuning_val_in_test=len(set(tune_call["val"]) & set(te)),
                          final_val_is_none=final_call["val"] is None,
                          final_train_eq_train=sorted(final_call["tr"]) == sorted(tr))
    ok7 = all(v["stat_rxns_in_test"] == 0 and v["stat_pairs_in_test"] == 0 and v["stat_rxns_outside_train"] == 0
              for v in c7.values()) and all(s["rows_equal"] and s["mu_maxdiff"] <= 1e-12 and s["sd_maxdiff"] <= 1e-12
                                            and s["fed_x_maxdiff"] <= 1e-5 for s in stat_ok)   # relative to max(1, |x|): x is fed in float32
    record("C7", ok7, folds=c7, run_unit_statistics=dict(tuning=stat_ok[0], final=stat_ok[1]))
    ok8 = all(v["cover"] and v["inner_overlap"] == 0 and v["val_in_test"] == 0 and v["val_core_in_inner_train"] == 0
              and 0.15 <= v["val_fraction"] <= 0.35 for kk, v in c8.items() if kk.startswith("f")) and \
        all(c8["run_unit"][k] for k in ("tuning_val_eq_inner_val", "tuning_train_eq_inner_train", "final_val_is_none",
                                         "final_train_eq_train")) and c8["run_unit"]["tuning_val_in_test"] == 0
    record("C8", ok8, checks=c8, note="GroupKFold recomputation on this node is information only (tie order may depend "
                                      "on the CPU's numpy sort kernel); splits.py used d1_train.inner_folds itself")

    # ---- C9 identity penalty: true y (spec: penalty term <= 1e-3) + exact-identity / perturbation / unit checks
    rows = tr
    sy = torch.tensor(sy0, dtype=torch.float64)
    ys = torch.tensor(D.y[rows] / sy0, dtype=torch.float64)
    loss, l1, pen = T.loss_fn(ys, ys, sy)
    resid = D.y[rows, F.I_BAR] - D.y[rows][:, F.I_CH9].sum(1)
    yex = ys.clone()
    yex[:, F.I_BAR] = (ys * sy)[:, F.I_CH9].sum(1) / sy[F.I_BAR]
    _, _, pen_ex = T.loss_fn(yex, yex, sy)
    e = torch.zeros_like(yex)
    e[:, F.I_BAR] = 0.7
    _, _, pen_bar = T.loss_fn(yex + e, yex, sy)
    ie = F.TARGETS.index("elst")
    e2 = torch.zeros_like(yex); e2[:, ie] = 0.3
    _, _, pen_el = T.loss_fn(yex + e2, yex, sy)
    yr = torch.tensor(np.random.default_rng(F.SEED).normal(size=ys.shape), dtype=torch.float64)
    lr_, l1r, penr = T.loss_fn(yr, ys, sy)
    kc = yr.numpy() * sy0
    pen_np = float(np.mean(np.abs(kc[:, F.I_BAR] - kc[:, F.I_CH9].sum(1))) / sy0[F.I_BAR])
    checks9 = dict(penalty_term_true_y=float(F.LAMBDA * pen), penalty_true_y=float(pen), l1_true_y=float(l1),
                   exact_identity_pen=float(pen_ex), pert_barrier_pen=float(pen_bar), pert_elst_pen=float(pen_el),
                   pert_elst_expect=float(0.3 * sy0[ie] / sy0[F.I_BAR]), random_pen=float(penr), random_pen_numpy=pen_np,
                   loss_minus_l1_lambda_pen=float(lr_ - l1r - F.LAMBDA * penr),
                   random_l1_minus_numpy=float(l1r - np.abs(yr.numpy() - ys.numpy()).mean()))
    ok9 = (checks9["penalty_term_true_y"] <= 1e-3 and checks9["l1_true_y"] == 0.0 and checks9["exact_identity_pen"] <= 1e-12
           and abs(checks9["pert_barrier_pen"] - 0.7) <= 1e-9
           and abs(checks9["pert_elst_pen"] - checks9["pert_elst_expect"]) <= 1e-9
           and abs(checks9["random_pen"] - pen_np) <= 1e-12 and abs(checks9["loss_minus_l1_lambda_pen"]) <= 1e-12
           and abs(checks9["random_l1_minus_numpy"]) <= 1e-12)
    record("C9", ok9, tol=1e-3, unit="standardised barrier units (|kcal residual| / s_barrier)", n_pairs=len(rows),
           label_identity_residual_kcal=dict(mean=float(np.abs(resid).mean()), p99=float(np.quantile(np.abs(resid), 0.99)),
                                             max=float(np.abs(resid).max())), s_barrier=float(sy0[F.I_BAR]), **checks9)

    # ---- C10 AW locality (L = 1) + positive control + E = sum eps in the model output
    awm, astd = trained(D, "AW", "A", dict(n_layers=1, width=64), sub)
    aw = noisy64(awm, F.SEED + 77)
    At = tens(astd)
    n_tested, dmax, pos_ctrl, n_pos = 0, 0.0, 0, 0
    for q in rng.choice(tr, size=60, replace=False):
        a = D.ra[q]
        R, ia, ib = T.batch(At, np.array([a]), np.array([a]))
        j = int(ia[0]); n = int(R["mask"][j].sum())
        P0 = R["pos"][j, :n]
        dist = torch.cdist(P0, P0)
        i = int(torch.argmax((dist > 5.0).sum(1)))
        far = torch.nonzero(dist[i] > 5.0).ravel()
        if len(far) == 0:
            continue
        n_tested += 1
        with torch.no_grad():
            e0 = aw.atom_contrib(R)[j, i].cpu().numpy() * astd.sy
            Rm = clone_R(R)
            u = (P0[far] - P0[i]) / (P0[far] - P0[i]).norm(dim=1, keepdim=True)
            step = torch.rand(len(far), 1, dtype=P0.dtype, device=P0.device) * 3 + 0.5
            Rm["pos"][j, far] = P0[far] + u * step                     # strictly outward: stays beyond 5 A of atom i
            assert bool((torch.cdist(Rm["pos"][j, :n], Rm["pos"][j, :n])[i, far] > 5.0).all())
            e1 = aw.atom_contrib(Rm)[j, i].cpu().numpy() * astd.sy
            near = torch.nonzero((dist[i] < 4.0) & (dist[i] > 0)).ravel()
            if len(near):
                n_pos += 1
                Rq = clone_R(R)
                dv = torch.randn(3, dtype=P0.dtype, device=P0.device)
                Rq["pos"][j, int(near[0])] = P0[int(near[0])] + 0.2 * dv / dv.norm()
                pos_ctrl += float(np.abs(aw.atom_contrib(Rq)[j, i].cpu().numpy() * astd.sy - e0).max()) > 1e-6
        dmax = max(dmax, float(np.abs(e1 - e0).max()))
    emax = 0.0
    for q in p30:
        a, b = D.ra[q], D.rb[q]
        R, ia, ib = T.batch(At, np.array([a]), np.array([b]))
        with torch.no_grad():
            eps = aw.atom_contrib(R).sum(1)
            emax = max(emax, float(np.abs((aw(R, ia, ib) - (eps[ia] - eps[ib])).cpu().numpy() * astd.sy).max()))
    record("C10", n_tested >= 20 and dmax <= 1e-6 and emax <= 1e-6 and n_pos >= 20 and pos_ctrl / max(1, n_pos) >= 0.9,
           n_reactions=n_tested, eps_i_max_diff=dmax, g_minus_sum_eps_max=emax, tol=1e-6,
           positive_control=dict(n=n_pos, changed_frac=pos_ctrl / max(1, n_pos)))

    # ---- C11 GNN edges (all reactions): 2 'formed' = formed_pairs_ts, covalent = the reactants' bonds (recounted
    #      independently from rel1 / rel2), every H has one covalent edge to its owner; fragment sums of q/v_rel
    n_bad, examples, reasons = 0, [], {}
    for k, rid in enumerate(D.rids):
        a = np.load(F.ATOMS / f"{rid}.npz")
        et = D.etype[k]
        n = int(a["n"])
        formed = {tuple(sorted(map(int, p))) for p in a["formed"]}
        f_et = {(i, j) for i, j in zip(*np.nonzero(np.triu(et == 2)))}
        c_et = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(et == 1)))}
        nb_rel = 0
        for Zk, Pk in (("rel1_Z", "rel1_pos"), ("rel2_Z", "rel2_pos")):
            r = np.array([COV_Z[int(z)] for z in a[Zk]])
            Dm = np.linalg.norm(a[Pk][:, None] - a[Pk][None], axis=-1)
            adj = Dm < 1.25 * (r[:, None] + r[None])
            np.fill_diagonal(adj, False)
            nb_rel += int(np.triu(adj, 1).sum())
        frag, rel_idx, Zs, P = a["frag"], a["rel_idx"], a["ts_Z"], a["ts_pos"]
        map_ok = True
        for (i, j) in c_et:
            if frag[i] != frag[j]:
                map_ok = False; break
            Zk, Pk = ("rel1_Z", "rel1_pos") if frag[i] == 0 else ("rel2_Z", "rel2_pos")
            ri, rj = int(rel_idx[i]), int(rel_idx[j])
            dist = float(np.linalg.norm(a[Pk][ri] - a[Pk][rj]))
            if not dist < 1.25 * (COV_Z[int(a[Zk][ri])] + COV_Z[int(a[Zk][rj])]):
                map_ok = False; break
        h_ok = True
        deg = {i: [] for i in range(n)}
        for i, j in c_et:
            deg[i].append(j); deg[j].append(i)
        for h in np.flatnonzero(Zs == 1):
            heavy = [x for x in range(n) if Zs[x] != 1 and frag[x] == frag[h]]
            owner = min(heavy, key=lambda x: float(np.linalg.norm(P[h] - P[x])))
            if deg[h] != [owner]:
                h_ok = False; break
        sums = max(abs(float(a["q_rel"][frag == 0].sum() - a["q_gs1"].sum())), abs(float(a["q_rel"][frag == 1].sum() - a["q_gs2"].sum())),
                   abs(float(a["v_rel"][frag == 0].sum() - a["v_gs1"].sum())), abs(float(a["v_rel"][frag == 1].sum() - a["v_gs2"].sum())))
        checks = dict(formed=f_et == formed and len(f_et) == 2, cov_eq_bonds=c_et == {tuple(map(int, b)) for b in a["bonds"]},
                      cov_count=len(c_et) == nb_rel == int(a["n_bonds_rel"][0]), disjoint=not (f_et & c_et),
                      symmetric=bool((et == et.T).all()), mapped_edges_are_bonds=map_ok, h_one_edge_to_owner=h_ok,
                      frag_sums=sums <= 1e-9)
        if not all(checks.values()):
            n_bad += 1
            for kk, vv in checks.items():
                if not vv:
                    reasons[kk] = reasons.get(kk, 0) + 1
            if len(examples) < 5:
                examples.append(int(rid))
    record("C11", n_bad == 0, n_reactions=len(D.rids), n_bad=n_bad, failing_checks=reasons, examples=examples)

    # ---- C12 learnability: 64 training pairs, 300 epochs -> training MAE < 5 % of null (gating: dropout 0, batch 8)
    c12 = {}
    q64 = rng.choice(tr, size=64, replace=False)
    for m in Mo.MODELS:
        for tag, kw in (("gate", dict(dropout=0.0, batch_size=8)), ("info_M1_settings", dict(dropout=F.DROPOUT, batch_size=F.BATCH))):
            r = T.fit(D, m, Mo.GRID[m][0], "A", q64, None, seed=F.SEED, n_epochs=300, **kw)
            yh = T.predict(r["model"], r["T"], D.ra[q64], D.rb[q64])
            ys64 = D.y[q64] / r["sy"]
            per_t = np.abs(yh - ys64).mean(0) / np.maximum(np.abs(ys64).mean(0), 1e-12)
            c12[f"{m}/{tag}"] = dict(train_mae=float(np.abs(yh - ys64).mean()), null_mae=float(np.abs(ys64).mean()),
                                     ratio=float(np.abs(yh - ys64).mean() / np.abs(ys64).mean()),
                                     ratio_per_target=dict(zip(F.TARGETS, per_t.round(4).tolist())), seconds=r["seconds"], **kw)
    # information only (C12 diagnostic, user decision 2026-10-21): the same 64 pairs, AW larger width / longer schedule
    old = F.MAX_EPOCHS
    for tag, cell, ne, tmax in (("AW/info_L1W128_300ep", dict(n_layers=1, width=128), 300, old),
                                ("AW/info_L1W64_1000ep_Tmax1000", dict(n_layers=1, width=64), 1000, 1000)):
        try:
            F.MAX_EPOCHS = tmax                                      # cosine horizon for this information run only
            r = T.fit(D, "AW", cell, "A", q64, None, seed=F.SEED, n_epochs=ne, dropout=0.0, batch_size=8)
        finally:
            F.MAX_EPOCHS = old
        yh = T.predict(r["model"], r["T"], D.ra[q64], D.rb[q64])
        ys64 = D.y[q64] / r["sy"]
        c12[tag] = dict(ratio=float(np.abs(yh - ys64).mean() / np.abs(ys64).mean()), cell=cell, epochs=ne, T_max=tmax,
                        seconds=r["seconds"])
    record("C12", all(c12[f"{m}/gate"]["ratio"] < 0.05 for m in Mo.MODELS), threshold=0.05, runs=c12,
           note="gating runs: dropout 0, batch 8 (300 epochs x 8 steps; the spec fixes neither); info runs: M-1 settings "
                "and two AW diagnostics on the same 64 pairs (not gating)")

    allok = all(v["ok"] for v in RES.values())
    out = dict(all_pass=bool(allok), tests=RES, versions=T.versions(), device=str(T.DEV), seconds=time.time() - t_start,
               t_report_sha256=F.sha256(F.FU2_RES / "T_report.json"), q_report_sha256=F.sha256(F.FU2_RES / "Q_report.json"),
               splits_sha256=F.sha256(F.FU2_RES / "S_splits.json"), code_version=F.CODE_VERSION,
               code_sha256={**T.CODE_SHA, "tests/run_tests.py": F.sha256(Path(__file__))})
    F.write_json(F.FU2_RES / "C_tests.json", out)
    print(json.dumps(dict(all_pass=allok, failed=[k for k, v in RES.items() if not v["ok"]]), indent=1))
    return 0 if allok else 3


if __name__ == "__main__":
    sys.exit(main())
