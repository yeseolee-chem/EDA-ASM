#!/usr/bin/env python3
r"""predict_atoms.py — GEO2b DEPLOY, atom-token models AW / GNN / TF (analysis/geo2/results/DECISIONS_deploy.md DP-1 .. DP-3).
The 30 LADDER dft units L__dft__<arm>__<model>__f<k> ($LAD/units: json + parquet + the 5 final models __s0..4.pt) are
applied (1) to their own DFT test inputs = regression gate against analysis/d1_ladder/results/L_predictions.parquet, and
only when that passes (2) to the geo-rung inputs of the same test pairs (deployment, rung label 'dft_on_geo').
GPU, env_d1fu2 (torch 2.2.1+cu118; analysis/geo2/jobs/m3_gpu.slurm). Nothing here was run when it was written.

  python predict_atoms.py run
        -> analysis/geo2/results/DEPLOY_regress_atoms.json (gate, per-unit checks, token compatibility) and, if the
           gate passes, $GEO2B/deploy/parts/atoms.parquet (PRED_COLS, rung dft_on_geo). Exit 0 done, 2 not ready,
           3 STOP (gate failed: no part written). A rerun recomputes everything (deterministic) and replaces both.

How a prediction is produced (LADDER code imported, not copied; analysis/d1_ladder/train_atoms.py = TA, fu2/train.py = FT)
  data      TA.get_data(rung, arm) (train_atoms.py 840-851) -> TA.LadData (778-834): rung dft = $LAD/fu2/tokens +
            $LAD/inputs/dft.parquet + F_pairs_cr2.csv + FU2 T1 splits; rung geo = $GEO2B/fu2_geo/tokens +
            $GEO2B/inputs/geo.parquet + F_pairs_d0d1.csv (D1 rows = F_pairs_cr2.csv bytes) + T1D0 splits. Arm SH global
            token = [X182 | H 9 of the rung table] (train_atoms.py 794).
  scaling   the fit's training pairs = Dd.split(k)['train'] (the unit's T_main, dft data); FT.Std (fu2/train.py 140-160)
            or TA.StdSH (train_atoms.py 867-880: + beta = TA.beta_fit on the same pairs, residual SD s_r) recomputed and
            required EQUAL (np.array_equal) to the std dict saved in every __s<j>.pt (train_atoms.py 1343-1347). The dft
            std (token / global mean + SD, s_y, s_r, beta) is then applied unchanged to the geo inputs: std.tensors(Dg)
            (fu2/train.py 155-160) standardises the geo tokens and the geo [X182 | H 9] with the dft training statistics.
  model     fu2 models.build(model, cell = unit json chosen.cell, d_in, 10, d_glob, 'A+G') (models.py 283-284) +
            load_state_dict of each saved seed model.
  output    TA.final_kcal(D, r, rows) (train_atoms.py 978-982): S = FT.predict (fu2/train.py 185-193, eval, float32,
            bs 256) x s_y; SH = TA.final_sh (892-894) = FT.predict x s_r + beta x (H(a) - H(b)) of D (dft H for the
            regression, geo H for deployment). yhat = mean over the 5 seed models, yhat_seed_sd = their SD (ddof 0),
            base = beta x dH (SH) or 0 (S) = train_atoms.run_unit 1349-1358. The dft test pairs keep their order; the
            deployed pairs are those whose both ends are usable in the geo rung (deploy_common.deploy_rows).
Gate (DP-2): max |yhat - L_predictions yhat| over the 30 units <= 1e-6 kcal/mol ('exact'), else <= 1e-4 ('within_fp32_bound',
  the FU2 gate M-1' float tolerance; the measured batch-composition sensitivity bs 32 vs 256 of seed 0 is recorded), else
  STOP. y, base, beta and the unit parquet are compared too; unit fingerprints may differ only in code_sha256
  (train_atoms.py) and gpu. The LADDER dft units all ran on NVIDIA A10: submit to an A10 partition (gpu2, gpu6).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair / d1_ladder trees

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_common as DC  # noqa: E402

sys.path.insert(0, str(DC.LAD_DIR))
import train_atoms as TA  # noqa: E402     (LADDER trainer: LadData, StdSH, final_kcal, final_sh, fingerprint, ...)
import torch  # noqa: E402

FT, Mo, F = TA.FT, TA.Mo, TA.F
if list(TA.TARGETS) != DC.TARGETS or tuple(TA.ARMS) != DC.ARMS or tuple(TA.MODELS) != DC.ATOM_MODELS:
    DC.stop(f"train_atoms TARGETS / ARMS / MODELS {TA.TARGETS} {TA.ARMS} {TA.MODELS} differ from deploy_common")
BS_SMALL = 32                                       # batch-composition sensitivity (seed 0 only; DP-2)
LAYOUT_KEYS = ("elements", "num_cols", "bin_cols", "d_in", "d_glob")


def norm(o):
    return TA.norm(o)


def layout_check(Dd, Dg):
    """DP-3: the dft and geo data objects must have the same token / global layout (else the dft models cannot read the
    geo tokens): element one-hot order, numeric / binary token columns, input width, global token width + names."""
    rec = {k: dict(dft=norm(getattr(Dd, k)), geo=norm(getattr(Dg, k))) for k in LAYOUT_KEYS}
    rec["glob_cols"] = dict(equal=list(Dd.trep["glob_cols"]) == list(Dg.trep["glob_cols"]), n=len(Dd.trep["glob_cols"]))
    rec["ext_used"] = dict(dft=bool(Dd.trep.get("ext_used")), geo=bool(Dg.trep.get("ext_used")))
    rec["h_cols"] = dict(dft=Dd.hinfo["h_cols"], geo=Dg.hinfo["h_cols"])
    rec["token_version"] = dict(dft=Dd.trep.get("token_version"), geo=Dg.trep.get("token_version"))
    bad = [k for k in LAYOUT_KEYS if rec[k]["dft"] != rec[k]["geo"]]
    if not rec["glob_cols"]["equal"]:
        bad.append("glob_cols")
    if rec["ext_used"]["dft"] != rec["ext_used"]["geo"]:
        bad.append("ext_used")
    if [c.split("_", 1)[1] for c in rec["h_cols"]["dft"]] != [c.split("_", 1)[1] for c in rec["h_cols"]["geo"]]:
        bad.append("h_cols targets")
    if bad:
        DC.stop(f"token / global layout of rung geo differs from rung dft ({Dd.arm}): {bad}")
    rec["ok"] = True
    return rec


def predict_bs(D, r, rows, bs):
    """= TA.final_kcal with another batch size (FT.predict / TA.final_sh bs argument); seed 0 sensitivity only."""
    if D.arm == "S":
        return FT.predict(r["model"], r["T"], D.ra[rows], D.rb[rows], bs=bs) * r["sy"]
    return TA.final_sh(r["model"], r["T"], D, r["std"], D.ra[rows], D.rb[rows], bs=bs)


def std_equal(std, saved, sh):
    keys = ("mu", "sd", "gmu", "gsd", "sy") + (("sr", "beta") if sh else ())
    out = {}
    for k in keys:
        a, b = np.asarray(saved[k], dtype=float), np.asarray(getattr(std, k), dtype=float)
        out[k] = bool(a.shape == b.shape and np.array_equal(a, b))
    return out


def unit_run(Dd, Dg, u, Lp):
    """one LADDER dft atom unit -> (record incl. the regression comparison, geo-input deploy rows, kept dft test rows)."""
    name = TA.uname(u)
    js, pq = TA.UNITS / f"{name}.json", TA.UNITS / f"{name}.parquet"
    if not (js.is_file() and pq.is_file()):
        DC.not_ready(f"{js} / {pq} missing")
    rec = json.loads(js.read_text())
    if rec.get("smoke") or not TA.gates_ok(rec):
        DC.stop(f"{name}: smoke unit or a gate M-1' not ok")
    if TA.L.sha256(pq) != rec.get("parquet_sha256"):
        DC.stop(f"{pq} changed after its json")
    sp = Dd.split(u["fold"])
    fpd = DC.fingerprint_diff(rec.get("fingerprint"), TA.fingerprint(Dd, u, sp), ("code_sha256", "gpu"),
                              ("train_atoms.py",))
    if not fpd["ok"]:
        DC.stop(f"{name}: fingerprint differs beyond code_sha256 (train_atoms.py) / gpu: {fpd}")
    if rec.get("test_pairs_sha256") != sp["test_sha"] or rec.get("val_pairs_sha256") != sp["val_sha"]:
        DC.stop(f"{name}: test / validation pairs differ from the current split")
    sh = u["arm"] == "SH"
    tr, te = sp["train"], sp["test"]
    std = TA.StdSH(Dd, tr) if sh else FT.Std(Dd, tr)
    cell = rec["chosen"]["cell"]
    keep, rows_g = DC.deploy_rows(Dd, Dg, te)
    dy_pairs = DC.check_same_pairs(Dd, Dg, te, keep, rows_g)
    Yd, Yg, s0_small, eq_all, pt_info = [], [], None, {}, []
    T_d = T_g = None
    for s in F.FINAL_SEEDS:
        p = TA.UNITS / f"{name}__s{s - F.SEED}.pt"
        if not p.is_file():
            DC.not_ready(f"{p} missing")
        obj = torch.load(p, map_location=FT.DEV)
        meta_ok = (norm(obj.get("cell")) == norm(cell) and norm(obj.get("unit")) == norm(u) and obj.get("rung") == "dft"
                   and obj.get("arm") == u["arm"] and obj.get("variant") == TA.VARIANT)
        if not meta_ok:
            DC.stop(f"{p}: cell / unit / rung / arm / variant differ from {name}.json")
        eq = std_equal(std, obj["std"], sh)
        if not all(eq.values()):
            DC.stop(f"{p}: saved std differs from the std recomputed on the unit's training pairs: {eq}")
        for k, v in eq.items():
            eq_all[k] = eq_all.get(k, True) and v
        if T_d is None:
            T_d, T_g = std.tensors(Dd), std.tensors(Dg)
        model = Mo.build(u["model"], cell, Dd.d_in, len(TA.TARGETS), Dd.d_glob, TA.VARIANT, F.DROPOUT).to(FT.DEV)
        model.load_state_dict(obj["state"])
        r = dict(model=model, T=T_d, sy=std.sy, std=std)
        Yd.append(TA.final_kcal(Dd, r, te))
        if s0_small is None:
            s0_small = DC.maxdiff(predict_bs(Dd, r, te, BS_SMALL), Yd[0])
        Yg.append(TA.final_kcal(Dg, dict(r, T=T_g), rows_g))
        pt_info.append(dict(file=p.name, sha256=TA.L.sha256(p)))
        del model, obj, r
    beta = np.asarray(std.beta, dtype=float) if sh else np.zeros(len(TA.TARGETS))
    if not np.array_equal(beta, np.asarray(rec.get("beta"), dtype=float)):
        DC.stop(f"{name}: beta differs from the unit json")
    base_d = beta * Dd.dH[te] if sh else np.zeros((len(te), len(TA.TARGETS)))
    base_g = beta * Dg.dH[rows_g] if sh else np.zeros((len(rows_g), len(TA.TARGETS)))
    mine = DC.pred_frame(Dd, te, u, np.stack(Yd), base_d, beta, DC.SRC_RUNG)
    dep = DC.pred_frame(Dg, rows_g, u, np.stack(Yg), base_g, beta, DC.RUNG_DEPLOY)
    c_unit = DC.compare_unit(mine, pd.read_parquet(pq))
    c_lad = DC.compare_unit(mine, DC.ladder_rows(Lp, u["arm"], u["model"], u["fold"]))
    fin_g = np.isfinite(dep.yhat.to_numpy(float)).all()
    if not fin_g:
        DC.stop(f"{name}: non-finite deploy predictions")
    del T_d, T_g
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = dict(unit=u, cell=cell, best_epoch=rec["chosen"].get("best_epoch"), n_test=int(len(te)),
               n_deploy=int(len(rows_g)), n_not_usable_in_geo=int((~keep).sum()), vs_l_predictions=c_lad,
               vs_unit_parquet=c_unit, batch_bs32_vs_bs256_seed0=s0_small, std_equal_saved=eq_all,
               y_dft_vs_geo_max_abs=dy_pairs, fingerprint_diff=fpd, unit_gpu=(rec.get("fingerprint") or {}).get("gpu"),
               unit_host=rec.get("host"), pt=pt_info)
    return out, dep, te[keep]


def token_compat(Dd, Dg, rids):
    """DP-3 per reaction (the ends of the deployed pairs): atom count, element counts, role-flag counts (bin columns),
    covalent / formed edge counts equal between the dft and the geo token; per numeric token column the shift of the
    reaction mean (atom-order free) geo - dft."""
    rids = sorted(int(r) for r in rids)
    rd = np.array([Dd.row[r] for r in rids], dtype=int)
    rg = np.array([Dg.row[r] for r in rids], dtype=int)
    n_eq = Dd.n[rd] == Dg.n[rg]
    el_eq = (Dd.onehot[rd].sum(1) == Dg.onehot[rg].sum(1)).all(1)
    bin_eq = (Dd.bin[rd].sum(1) == Dg.bin[rg].sum(1)).all(1)

    def edges(D, rows, e):
        return (D.etype[rows] == e).sum((1, 2)) // 2
    cov_eq = edges(Dd, rd, 1) == edges(Dg, rg, 1)
    form_eq = edges(Dd, rd, 2) == edges(Dg, rg, 2)

    def rmean(D, rows):
        m = D.mask[rows][..., None].astype(float)
        return (D.num[rows] * m).sum(1) / np.maximum(m.sum(1), 1.0)
    md, mg = rmean(Dd, rd), rmean(Dg, rg)
    shift = {}
    for j, c in enumerate(Dd.num_cols):
        a, b = md[:, j], mg[:, j]
        sd = float(a.std())
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        shift[c] = dict(mean_dft=float(a.mean()), mean_geo=float(b.mean()), sd_dft=sd,
                        mean_abs_diff=float(np.abs(b - a).mean()), mad_over_sd_dft=float(np.abs(b - a).mean() / sd)
                        if sd > 0 else float("nan"), pearson_r=r)
    bad = ~(n_eq & el_eq & bin_eq)
    return dict(n_reactions=len(rids), n_atoms_equal=int(n_eq.sum()), element_counts_equal=int(el_eq.sum()),
                role_flag_counts_equal=int(bin_eq.sum()), covalent_edge_count_equal=int(cov_eq.sum()),
                formed_edge_count_equal=int(form_eq.sum()),
                composition_differs=[int(x) for x in np.asarray(rids)[bad][:20]], n_composition_differs=int(bad.sum()),
                num_col_reaction_mean_shift=shift,
                note="per reaction; atom order is not compared (the pooled models are permutation invariant)")


def cmd_run():
    t0 = time.time()
    FT.require_gpu()                                 # env_d1fu2 + CUDA (exit 3 otherwise)
    Lp, lp_info = DC.load_ladder_dft(DC.ATOM_MODELS)
    DC.log(t0, f"L_predictions dft atom rows {len(Lp)}; {FT.versions()}")
    units, deps, layout, deployed_rx, kept = {}, [], {}, set(), {}
    data = {}
    for arm in TA.ARMS:
        Dd = TA.get_data(DC.SRC_RUNG, arm)
        Dg = TA.get_data(DC.IN_RUNG, arm)
        data[arm] = (Dd, Dg)
        layout[arm] = layout_check(Dd, Dg)
        for model in TA.MODELS:
            for k in range(TA.NF):
                u = dict(rung=DC.SRC_RUNG, arm=arm, model=model, fold=k)
                name = TA.uname(u)
                rec, dep, te_kept = unit_run(Dd, Dg, u, Lp)
                units[name] = rec
                deps.append(dep)
                kept[name] = sorted(np.asarray(Dd.P.index)[te_kept].astype(str).tolist())
                deployed_rx |= set(Dd.P.a_rxn.to_numpy()[te_kept].tolist()) | set(Dd.P.b_rxn.to_numpy()[te_kept].tolist())
                c = rec["vs_l_predictions"]
                DC.log(t0, f"{name}: test {rec['n_test']}, deploy {rec['n_deploy']}, |dyhat| vs L "
                           f"{c['max_abs_yhat']:.3g} (unit parquet {rec['vs_unit_parquet']['max_abs_yhat']:.3g}), "
                           f"bs32 {rec['batch_bs32_vs_bs256_seed0']:.3g}")
    # one deployed pair list per fold for every unit (spec 16 E analogue)
    bad_lists = [f"f{k}" for k in range(TA.NF)
                 if len({tuple(v) for n, v in kept.items() if n.endswith(f"__f{k}")}) != 1]
    if bad_lists:
        DC.stop(f"deployed pair lists differ between atom units of {bad_lists}")
    tok = token_compat(*data["S"], deployed_rx)
    vs_l = [v["vs_l_predictions"] for v in units.values()]
    vs_u = [v["vs_unit_parquet"] for v in units.values()]
    dmax = max(v["max_abs_yhat"] for v in vs_l)
    pairs_ok = all(v["same_pairs"] and v["same_order"] for v in vs_l + vs_u)
    others = {c: max(v[f"max_abs_{c}"] for v in vs_l + vs_u) for c in ("y", "base", "beta")}
    status = DC.gate_status(max(dmax, max(v["max_abs_yhat"] for v in vs_u)), allow_fp32=True)
    passed = bool(status != "fail" and pairs_ok and all(x <= DC.Y_TOL for x in others.values()))
    gpu_now = FT.versions().get("gpu")
    gate = dict(status=status, passed=passed, max_abs_yhat_vs_l_predictions=dmax,
                max_abs_yhat_vs_unit_parquet=max(v["max_abs_yhat"] for v in vs_u),
                max_abs_yhat_seed_sd=max(v["max_abs_yhat_seed_sd"] for v in vs_l),
                max_abs_y_base_beta=others, same_pairs_and_order=pairs_ok,
                batch_bs32_vs_bs256_seed0_max=max(v["batch_bs32_vs_bs256_seed0"] for v in units.values()),
                tol_exact=DC.TOL_EXACT, tol_fp32_bound=DC.TOL_FP32, gpu=gpu_now,
                unit_gpus=sorted({str(v["unit_gpu"]) for v in units.values()}),
                same_gpu_type_as_units=bool({str(v["unit_gpu"]) for v in units.values()} == {str(gpu_now)}),
                rule="DP-2: 30 dft atom units recomputed from the saved models on their DFT test inputs vs "
                     "L_predictions.parquet / unit parquet: exact <= 1e-6 kcal/mol; else within_fp32_bound <= 1e-4 "
                     "(fu2 ANTISYM_TOL); y / base / beta <= 1e-12; same pairs and order; else fail (no deploy part)")
    out = dict(phase="GEO2b DEPLOY atoms (DP-1 .. DP-3)", gate=gate, layout=layout, token_compat=tok,
               l_predictions=lp_info, units=units, versions=FT.versions(), env=DC.env_info(), code_sha256=DC.code_sha(),
               train_atoms_code_sha256=TA.CODE_SHA, decisions=str(DC.DECISIONS_MD),
               decisions_sha256=DC.sha256(DC.DECISIONS_MD) if DC.DECISIONS_MD.is_file() else None)
    if not passed:
        out.update(deploy=None, seconds=time.time() - t0)
        DC.write_json(DC.REG_ATOMS, out)
        DC.stop(f"atom regression gate failed: {status}, max |dyhat| {dmax:.3g}, pairs ok {pairs_ok}, y/base/beta {others}"
                f" -> {DC.REG_ATOMS} (no deploy part written)")
    D = pd.concat(deps, ignore_index=True)[DC.PRED_COLS]
    if D.duplicated(["pair_id", "arm", "model", "target"]).any():
        DC.stop("duplicated deploy rows (a pair in two folds?)")
    DC.write_parquet(DC.PART_ATOMS, D)
    out["deploy"] = dict(part=str(DC.PART_ATOMS), sha256=DC.sha256(DC.PART_ATOMS), n_rows=int(len(D)),
                         n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in D.groupby("fold")},
                         rung=DC.RUNG_DEPLOY)
    out["seconds"] = time.time() - t0
    DC.write_json(DC.REG_ATOMS, out)
    DC.log(t0, f"gate {status} (max |dyhat| {dmax:.3g}); {len(D)} deploy rows -> {DC.PART_ATOMS}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        sys.exit(cmd_run())
    raise SystemExit(__doc__)
