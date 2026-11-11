#!/usr/bin/env python3
r"""tests_abs.py — ABS3 Phase C (docs/specs/ABS3_MODELS.md 5). Exit 0 all pass, 2 not ready, 3 STOP.

  python tests_abs.py gpu     env_d1fu2 on RTX 3090: C-E (TF, GNN), C-O (TF, GNN), C-8 smoke (TF, GNN) with C-T and
                              C-5, C-P on the smoke reaction predictions -> results/C_part_gpu.json
  python tests_abs.py xgb     env_espley (32 CPUs): the XGB inner folds of every fold (pair rule -> reactions; disjoint),
                              C-O (XGB), C-X, C-8 smoke (XGB) with C-T and C-5, C-P -> results/C_part_xgb.json
  python tests_abs.py merge   both parts -> results/C_tests.json (all_pass, trainer_code_sha256), C_overfit.json,
                              C8_smoke.json

C-5 runs inside every fit (train_abs.leak_check / xgb_abs.leak_check: STOP on a leak); the parts record it.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import abs_pairs as AP  # noqa: E402

T3 = list(C.TARGETS3)
GPU_PART_FILES = tuple(C.NN_TRAINER_FILES) + ("tests_abs.py", "abs_pairs.py", "abs_pack.py")
XGB_PART_FILES = tuple(C.XGB_TRAINER_FILES) + ("tests_abs.py", "abs_data.py", "abs_pack.py")


def files_sha(names):
    return {f: C.sha256(HERE / f) for f in names}


def provenance():
    """the inputs a C part ran on (review 11-11: merge rejects parts of an earlier round)."""
    import os
    ls = C.RES / "LS_report.json"
    return dict(pack_npz_sha256=C.sha256(C.PACK_NPZ) if C.PACK_NPZ.is_file() else None,
                ls_report_sha256=C.sha256(ls) if ls.is_file() else None, slurm_job=os.environ.get("SLURM_JOB_ID"))


def clear_part(name):
    p = C.RES / name
    if p.is_file():
        C.assert_write_ok(p)
        p.unlink()


def pack_pairs(k_list=(0,)):
    if not (C.PACK_NPZ.is_file() and C.PACK_JSON.is_file()):
        C.not_ready(f"{C.PACK_NPZ} missing (abs_pack.py)")
    z = np.load(C.PACK_NPZ, allow_pickle=False)
    return pd.concat([AP.pack_test_pairs(z, k) for k in k_list], ignore_index=True)


def cp_on(pq, label):
    rx = pd.read_parquet(pq)
    pairs = pack_pairs((0,))
    a = AP.assemble(rx, pairs, label)
    out = AP.cp_check(a, rx, pairs, AP.labels_truth(C.LABELS, C.M()))
    out["n_seed_cols"] = len([c for c in rx.columns if c.startswith("yhat_s") and c[6:].isdigit()])
    out["seed_sd_finite"] = bool(np.isfinite(a.yhat_seed_sd).all()) if out["n_seed_cols"] else None
    return out


# ================================================================ GPU part (env_d1fu2)
def c_e(A, TA, model):
    """C-E: same seed -> AbsNet.enc = models.build (parameters and reaction vectors identical); optimiser parameters
    exclude the pair head and the TF cross block."""
    import torch
    Mo, FT, AM = TA.Mo, TA.FT, TA.AM
    out = {}
    sp = A.split(0)
    std = TA.AD.AbsStd(A, sp["train"])
    T = std.tensors(A.D)
    rows = np.asarray(sp["test"][:48], dtype=int)
    for ci in (0, len(TA.grid(model)) - 1):
        cell = TA.grid(model)[ci]
        FT.seed_all(TA.F.SEED + ci)
        a = AM.build(model, cell, A.d_in, A.d_glob, TA.VARIANT, TA.F.DROPOUT).to(TA.DEV)
        FT.seed_all(TA.F.SEED + ci)
        m = Mo.build(model, cell, A.d_in, 4, A.d_glob, TA.VARIANT, TA.F.DROPOUT).to(TA.DEV)
        sa, sm = a.enc.state_dict(), m.state_dict()
        same_keys = list(sa) == list(sm)
        pmax = max(float((sa[k] - sm[k]).abs().max()) for k in sa) if same_keys else float("inf")
        a.eval()
        m.eval()
        with torch.no_grad():
            R, ia, _ = FT.batch(T, rows, rows)
            va = a.rvec(R)[ia]
            vm = (m.readout(R) if model == "GNN" else m.pooled(*m.tokens(R)))[ia]
            vmax = float((va - vm).abs().max())
            outw = int(a(R, ia).shape[1])
        used = [k for k, _ in a.used_named_parameters()]
        unused = a.unused_parameter_names()
        opt_ids = {id(p) for p in a.used_parameters()}
        checks = dict(enc_state_equal=same_keys and pmax == 0.0, rvec_equal=vmax == 0.0, out_width_4=outw == 4,
                      no_head_or_cross_in_optimiser=not any(k.startswith(("enc.head.", "enc.cross.")) for k in used),
                      head_unused_only=all(k.startswith(("enc.head.", "enc.cross.")) for k in unused) and bool(unused),
                      new_head_trained=any(k.startswith("head.") for k in used),
                      optimiser_params_are_used=len(opt_ids) == len(used),
                      d_r=a.d_r == AM.expected_dr(model, int(cell["width"]), A.d_glob))
        out[f"cell{ci}"] = dict(cell=cell, ok=bool(all(checks.values())), checks=checks, max_param_diff=pmax,
                                max_rvec_diff=vmax, d_r=a.d_r, n_used=len(used), unused=unused)
    return dict(ok=all(v["ok"] for v in out.values()), cells=out)


def c_t_nn(A, TA, rec, root):
    """C-T (TF / GNN) on a smoke unit: 4 outputs, standardisation from the training reactions only, identity penalty in
    kcal, validation metric = mean of the 4 targets, unit record targets / columns."""
    import torch
    u = rec["unit"]
    sp = A.split(u["fold"])
    pt = torch.load(root / f"{TA.uname(u)}__s0.pt", map_location="cpu")
    ref = TA.AD.AbsStd(A, sp["train"]).as_dict()
    std_eq = all(np.array_equal(np.asarray(pt["std"][k], float), np.asarray(ref[k], float)) for k in ref)
    # the same statistics from a set that also holds a test reaction differ (the check can fail)
    alt = TA.AD.AbsStd(A, np.r_[sp["train"], sp["test"][:1]]).as_dict()
    sensitive = not all(np.array_equal(np.asarray(alt[k]), np.asarray(ref[k])) for k in ("mu", "gmu", "ymu"))
    # loss: penalty on kc = mu + s * yhat (kcal), divided by s_barrier
    g = np.random.default_rng(0)
    yh, y = g.normal(size=(7, 4)), g.normal(size=(7, 4))
    sy, mu = np.abs(g.normal(size=4)) + 0.5, g.normal(size=4)
    loss, l1, pen = TA.loss_abs(*(torch.tensor(v, dtype=torch.float64) for v in (yh, y, sy, mu)))
    kc = mu + yh * sy
    pen_np = float((np.abs(kc[:, 3] - kc[:, :3].sum(1)) / sy[3]).mean())
    l1_np = float(np.abs(yh - y).mean())
    loss_ok = abs(float(pen) - pen_np) <= 1e-12 and abs(float(l1) - l1_np) <= 1e-12 and \
        abs(float(loss) - (l1_np + TA.F.LAMBDA * pen_np)) <= 1e-12
    vm = all(len(t["val_per_target"]) == 4 and abs(t["val"] - float(np.mean(t["val_per_target"]))) <= 1e-12
             for t in rec["tune"])
    P = pd.read_parquet(root / f"{TA.uname(u)}.parquet")
    checks = dict(fu2_constants_4_targets=TA.DC3.current_mode() == TA.DC3.MODE3, model_output_4=all(t["out_dim"] == 4 for t in rec["tune"]),
                  pt_targets_4=list(pt["TARGETS"]) == T3, pt_kind_AbsNet=pt.get("kind") == "AbsNet",
                  std_from_training_reactions=bool(std_eq), std_check_sensitive=bool(sensitive),
                  loss_penalty_kcal=bool(loss_ok), val_metric_mean4=bool(vm), record_targets_4=list(rec["targets"]) == T3,
                  prediction_targets_4=sorted(P.target.unique()) == sorted(T3),
                  prediction_columns=list(P.columns) == ["rxn_id", "fold", "model", "target", "y", "yhat", "yhat_seed_sd",
                                                         "yhat_s0"],
                  y_equals_y4=bool(np.array_equal(
                      P.pivot(index="rxn_id", columns="target", values="y")[T3].to_numpy(),
                      A.y4[[A.row[int(r)] for r in sorted(P.rxn_id.unique())]])))
    return dict(ok=bool(all(checks.values())), checks=checks, loss_detail=dict(pen=float(pen), pen_numpy=pen_np))


def cmd_gpu():
    clear_part("C_part_gpu.json")
    import train_abs as TA
    TA.FT.require_gpu()
    A = TA.AD.get_data()
    out = dict(part="gpu", written=C.now(), versions=TA.FT.versions(), code_sha256=files_sha(GPU_PART_FILES),
               provenance=provenance(), C_E={}, C_O={}, C_8={}, C_T={}, C_P={})
    for m in TA.NN_MODELS:
        out["C_E"][m] = c_e(A, TA, m)
        print(f"[C-E] {m}: {out['C_E'][m]['ok']}", flush=True)
    for m in TA.NN_MODELS:
        out["C_O"][m] = TA.overfit(A, m)
        print(f"[C-O] {m}: ok {out['C_O'][m]['ok']} ratio {out['C_O'][m]['ratio']}", flush=True)
    for m in TA.NN_MODELS:
        rec, pq, sec = TA.smoke(A, m)
        leak = all(t["leak"]["ok"] for t in rec["tune"] + rec["final"])
        out["C_8"][m] = dict(ok=bool(leak and len(rec["tune"]) == 2 and len(rec["final"]) == 1), seconds=sec,
                             cells=[t["cell"] for t in rec["tune"]], out_dims=[t["out_dim"] for t in rec["tune"]],
                             leak_C5=bool(leak), n_train=rec["n_train"], n_test=rec["n_test"],
                             n_inner_train=rec["n_inner_train"], n_inner_val=rec["n_inner_val"])
        out["C_T"][m] = c_t_nn(A, TA, rec, pq.parent)
        out["C_P"][m] = cp_on(pq, C.ABS_LABEL[m])
        print(f"[C-8/C-T/C-P] {m}: {out['C_8'][m]['ok']} / {out['C_T'][m]['ok']} / {out['C_P'][m]['ok']}", flush=True)
    out["ok"] = bool(all(v["ok"] for k in ("C_E", "C_O", "C_8", "C_T", "C_P") for v in out[k].values()))
    C.write_json(C.RES / "C_part_gpu.json", out)
    print(json.dumps(C.clean(dict(ok=out["ok"], **{k: {m: v["ok"] for m, v in out[k].items()}
                                                 for k in ("C_E", "C_O", "C_8", "C_T", "C_P")})), indent=1))
    return 0 if out["ok"] else 3


# ================================================================ XGB part (env_espley)
def cmd_xgb():
    clear_part("C_part_xgb.json")
    import xgb_abs as XA
    P = XA.PackAbs()
    out = dict(part="xgb", written=C.now(), versions=XA.TX.versions(), n_jobs=XA.N_JOBS,
               code_sha256=files_sha(XGB_PART_FILES), provenance=provenance())
    inner = {}
    for k in range(C.NF):
        sp = P.split(k)
        inner[str(k)] = [dict(n_train=int(len(f["train"])), n_val=int(len(f["val"])), n_pairs=f["n_pairs"]) for f in sp["inner"]]
    out["inner_folds"] = dict(ok=True, folds=inner,
                              rule="pair XGB rule -> reactions; disjoint per inner fold, fold 0 = TF · GNN inner sets "
                                   "(PackAbs.split STOPs otherwise)")
    out["C_O"] = XA.overfit(P)
    print(f"[C-O] XGB: ok {out['C_O']['ok']} ratio {out['C_O']['ratio']}", flush=True)
    out["C_X"] = XA.determinism(P)
    print(f"[C-X] {out['C_X']}", flush=True)
    rec, pq, sec = XA.smoke(P)
    leak = all(f["leak"]["ok"] for f in rec["inner_folds"]) and rec["final_leak"]["ok"]
    out["C_8"] = dict(ok=bool(leak), seconds=sec, cpu_seconds=rec["cpu_seconds"], chosen=rec["chosen"], leak_C5=bool(leak),
                      n_train=rec["n_train"], n_test=rec["n_test"], inner_folds=rec["inner_folds"])
    out["C_T"] = XA.ct_checks(P, rec, pq)
    out["C_P"] = cp_on(pq, C.ABS_LABEL["XGB"])
    out["ok"] = bool(out["C_O"]["ok"] and out["C_X"]["ok"] and out["C_8"]["ok"] and out["C_T"]["ok"] and out["C_P"]["ok"])
    C.write_json(C.RES / "C_part_xgb.json", out)
    print(json.dumps(C.clean(dict(ok=out["ok"], C_O=out["C_O"]["ok"], C_X=out["C_X"]["ok"], C_8=out["C_8"]["ok"],
                                  C_T=out["C_T"], C_P=out["C_P"]["ok"])), indent=1))
    return 0 if out["ok"] else 3


# ================================================================ merge
def _need(name):
    p = C.RES / name
    if not p.is_file():
        C.not_ready(f"{p} missing")
    return C.read_json(p)


def cmd_merge():
    g, x = _need("C_part_gpu.json"), _need("C_part_xgb.json")
    nn_now, xgb_now = files_sha(C.NN_TRAINER_FILES), files_sha(C.XGB_TRAINER_FILES)
    prov_now = provenance()
    same_inputs = all((p.get("provenance") or {}).get(k) == prov_now[k] and prov_now[k] is not None
                      for p in (g, x) for k in ("pack_npz_sha256", "ls_report_sha256"))
    checks = {
        "C-E encoder identity (TF, GNN)": all(v["ok"] for v in g["C_E"].values()),
        "C-O overfit 64 reactions (TF, GNN, XGB)": all(v["ok"] for v in g["C_O"].values()) and bool(x["C_O"]["ok"]),
        "C-T wiring (TF, GNN, XGB)": all(v["ok"] for v in g["C_T"].values()) and bool(x["C_T"]["ok"]),
        "C-5 leak, every fit (smoke + overfit; STOP inside fits)": all(v["leak_C5"] for v in g["C_8"].values())
        and bool(x["C_8"]["leak_C5"]) and all(v["leak"]["ok"] for v in g["C_O"].values()) and bool(x["C_O"]["leak"]["ok"]),
        "C-X XGB determinism": bool(x["C_X"]["ok"]),
        "C-P pair ΔΔ assembly (TF, GNN, XGB smoke)": all(v["ok"] for v in g["C_P"].values()) and bool(x["C_P"]["ok"]),
        "C-8 smoke (TF, GNN, XGB)": all(v["ok"] for v in g["C_8"].values()) and bool(x["C_8"]["ok"]),
        "XGB inner folds (pair rule -> reactions) disjoint": bool(x["inner_folds"]["ok"]),
        "parts ran the current trainer / test code": g.get("code_sha256") == files_sha(GPU_PART_FILES)
        and x.get("code_sha256") == files_sha(XGB_PART_FILES),
        "parts ran on the current pack and LS report": bool(same_inputs),
    }
    rec = dict(spec="docs/specs/ABS3_MODELS.md 5", written=C.now(), all_pass=bool(all(checks.values())), checks=checks,
               trainer_code_sha256=dict(train_abs=nn_now, xgb_abs=xgb_now),
               parts_sha256={n: C.sha256(C.RES / n) for n in ("C_part_gpu.json", "C_part_xgb.json")},
               versions=dict(gpu=g.get("versions"), xgb=x.get("versions")))
    ov = dict(spec="docs/specs/ABS3_MODELS.md 5 C-O", written=C.now(),
              models={**{m: {k: v[k] for k in ("cell", "n", "epochs", "dropout", "train_mae", "std_64", "ratio", "threshold",
                                                "seconds", "ok") if k in v} for m, v in g["C_O"].items()},
                      "XGB": {k: x["C_O"][k] for k in ("cell", "n", "train_mae", "std_64", "ratio", "threshold", "ok")}},
              ok=bool(checks["C-O overfit 64 reactions (TF, GNN, XGB)"]))
    sm = dict(spec="docs/specs/ABS3_MODELS.md 5 C-8", written=C.now(),
              models={**g["C_8"], "XGB": x["C_8"]}, C_T={**g["C_T"], "XGB": x["C_T"]}, C_P={**g["C_P"], "XGB": x["C_P"]},
              C_E=g["C_E"], C_X=x["C_X"], ok=bool(checks["C-8 smoke (TF, GNN, XGB)"]))
    C.write_json(C.RES / "C_overfit.json", ov)
    C.write_json(C.RES / "C8_smoke.json", sm)
    C.write_json(C.RES / "C_tests.json", rec)
    print(json.dumps(C.clean(dict(all_pass=rec["all_pass"], checks=checks)), indent=1, ensure_ascii=False))
    return 0 if rec["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    fn = dict(gpu=cmd_gpu, xgb=cmd_xgb, merge=cmd_merge).get(a[0] if len(a) == 1 else "")
    if fn is None:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn())
