#!/usr/bin/env python3
r"""tests_ch3.py — CH3-MODEL Phase C (docs/specs/CH3_MODEL.md 6). Exit 0 all pass, 2 not ready, 3 STOP.

  python tests_ch3.py cpu      C-Y, C-D, C-5 (static, every fold) on CPU (env_d1fu2) -> results/C_part_cpu.json
  python tests_ch3.py merge    C_part_cpu.json + results/C3r_regress.json (C-3′, train_ch3.py c3r) + results/C8_smoke.json
                               (C-8 + C-T, train_ch3.py smoke) + C-4 / C-5 records of every GPU fit -> results/C_tests.json
                               (all_pass, trainer_code_sha256 of the files train_ch3 run / final check)

C-Y  Ch3Data.y == y10 @ M bit for bit, == the explicit group sums of F_pairs_cr2 dd_* (<= 1e-12), finite on every
     usable fold pair; |strain + inter + solv - barrier| max and distribution (STOP when the max > 0.2)
C-D  Ch3Data input arrays / pairs / splits == DeployData('lf') (np.array_equal; list sha256 equal), y10 == DeployData y
C-5  standardisation statistics only from the fit's training pairs / reactions (train_tf.leak_check on fu2 Std) for the
     tuning fit and the final fit of every fold and for split('all'); no test reaction among them
C-3′ / C-8 / C-T / C-4: GPU (train_ch3.py c3r / smoke); merge reads their records and checks every fit's gate.
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
import ch3_common as C3  # noqa: E402

ARRAYS = ("onehot", "bin", "num", "pos", "mask", "etype", "frag", "n", "glob", "ra", "rb", "has_tok", "usable_rxn",
          "usable_pair", "rids")
IDENT_MAX = 0.2


def c_y(D):
    y_mm = D.y10 @ C3.M()
    bit = bool(np.array_equal(D.y, y_mm, equal_nan=True))
    L = pd.read_csv(C3.LABELS)
    L["pair_id"] = L.pair_id.astype(str)
    L = L.set_index("pair_id").reindex(D.P.index)
    explicit = np.stack([L[[f"dd_{t}" for t in C3.GROUPS[g]]].sum(axis=1, min_count=len(C3.GROUPS[g])).to_numpy(float)
                         for g in C3.TARGETS3], 1)
    fin10 = np.isfinite(D.y10).all(1)
    dexp = float(np.abs(D.y[fin10] - explicit[fin10]).max())
    nan_same = bool(np.isfinite(D.y[fin10]).all() and np.isfinite(explicit[fin10]).all())
    n_nan10 = int((~fin10).sum())
    rows = np.array(sorted(D.pid_row[q] for q in D.pair_list), dtype=int)
    fin = bool(np.isfinite(D.y[rows]).all())
    res = D.y[:, 0] + D.y[:, 1] + D.y[:, 2] - D.y[:, 3]
    r_use = np.abs(res[rows])
    r_all = np.abs(res[np.isfinite(res)])
    dist = lambda v: dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.median(v)),             # noqa: E731
                          p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)), max=float(v.max()))
    ok = bit and dexp <= 1e-12 and nan_same and fin and float(r_use.max()) <= IDENT_MAX
    return dict(ok=bool(ok), bitwise_y_eq_y10_at_M=bit, max_abs_diff_explicit_sums=dexp,
                finite_where_y10_finite=nan_same, n_label_rows_with_nan_y10=n_nan10,
                finite_on_usable_fold_pairs=fin, n_usable_fold_pairs=int(len(rows)),
                identity_residual_usable=dist(r_use), identity_residual_all_labels=dist(r_all), stop_max=IDENT_MAX,
                stop=bool(float(r_use.max()) > IDENT_MAX))


def c_d(D, E):
    arr = {k: bool(np.array_equal(getattr(D, k), getattr(E, k))) for k in ARRAYS}
    arr["y10"] = bool(np.array_equal(D.y10, E.y, equal_nan=True))
    arr["P_index"] = bool((D.P.index == E.P.index).all())
    lists = {"pair_list_sha": D.pair_list_sha == E.pair_list_sha}
    for k in list(range(5)) + ["all"]:
        a, b = D.split(k), E.split(k)
        for key in ("train", "test", "inner_train", "inner_val"):
            lists[f"{k}/{key}"] = bool(np.array_equal(a[key], b[key]))
        lists[f"{k}/pairs_sha"] = a["pairs_sha"] == b["pairs_sha"]
    sha = {k: D.sha.get(k) == E.sha.get(k) for k in E.sha}
    return dict(ok=bool(all(arr.values()) and all(lists.values()) and all(sha.values())), arrays=arr, lists=lists,
                inputs_sha256_equal=sha, pair_list_sha=D.pair_list_sha, n_pairs=len(D.pair_list))


def c_5(D, FT, TT):
    out = {}
    for k in list(range(5)) + ["all"]:
        sp = D.split(k)
        tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
        rec = {}
        if k != "all":
            s1 = FT.Std(D, itr)
            rec["tune"] = TT.leak_check(D, s1, itr, (iva, te), te)
        s2 = FT.Std(D, tr)
        rec["final"] = TT.leak_check(D, s2, tr, (te,) if len(te) else (), te)
        trx = set(D.ra[tr].tolist()) | set(D.rb[tr].tolist())
        tex = set(D.ra[te].tolist()) | set(D.rb[te].tolist()) if len(te) else set()
        rec["n_test_reactions_in_train"] = len(trx & tex)
        out[str(k)] = rec
    ok = all(r["n_test_reactions_in_train"] == 0 and all(v.get("ok", True) for v in r.values() if isinstance(v, dict))
             for r in out.values())
    return dict(ok=bool(ok), folds=out)


def cmd_cpu():
    import data_ch3 as DC3
    import train_tf as TT
    D = DC3.get_data("ch3")
    E = DC3.get_data("lf")
    rec = dict(written=C3.now(), C_Y=c_y(D), C_D=c_d(D, E), C_5=c_5(D, DC3.FT, TT))
    rec["ok"] = bool(rec["C_Y"]["ok"] and rec["C_D"]["ok"] and rec["C_5"]["ok"])
    C3.write_json(C3.RES / "C_part_cpu.json", rec)
    print(json.dumps(dict(ok=rec["ok"], C_Y=rec["C_Y"]["ok"], C_D=rec["C_D"]["ok"], C_5=rec["C_5"]["ok"],
                          identity_max=rec["C_Y"]["identity_residual_usable"]["max"]), indent=1))
    if rec["C_Y"]["stop"]:
        C3.stop(f"C-Y: identity residual max {rec['C_Y']['identity_residual_usable']['max']} > {IDENT_MAX}")
    return 0 if rec["ok"] else 3


def _fits_ok(unit):
    fits = list(unit.get("tune") or []) + list(unit.get("final") or [])
    g = bool(fits) and all(f["gate_M1"]["ok"] for f in fits)
    lk = bool(fits) and all((f.get("leak") or {}).get("ok") for f in fits)
    anti = max(f["gate_M1"]["anti_max"] for f in fits) if fits else None
    selfm = max(f["gate_M1"]["self_max"] for f in fits) if fits else None
    return g, lk, anti, selfm, len(fits)


def cmd_merge():
    import train_ch3 as TC
    parts = {n: C3.RES / n for n in ("C_part_cpu.json", "C3r_regress.json", "C8_smoke.json")}
    miss = [n for n, p in parts.items() if not p.is_file()]
    if miss:
        C3.not_ready(f"C parts missing: {miss}")
    cpu, c3r, sm = (C3.read_json(p) for p in parts.values())
    unit = C3.read_json(C3.C3R / "D__lf__S__TF__f0.json")
    g3, l3, a3, s3, n3 = _fits_ok(unit)
    sunit = C3.read_json(C3.SMOKE / "D__lf3__S__TF__f0.json")
    gs, ls, as_, ss, ns = _fits_ok(sunit)
    checks = {"C-Y label conversion": cpu["C_Y"]["ok"], "C-D data identity": cpu["C_D"]["ok"],
              "C-3′ 10-target regression vs deploy K fold 0": bool(c3r.get("ok")),
              "C-T ch3 mode": bool((sm.get("CT") or {}).get("ok")),
              "C-4 antisymmetry (every C-3′ and smoke fit)": g3 and gs,
              "C-5 standardisation leak (static, every fold + every GPU fit)": cpu["C_5"]["ok"] and l3 and ls,
              "C-8 smoke": bool(sm.get("C8_ok"))}
    code_now = {f: TC.CODE_SHA[f] for f in TC.TRAINER_FILES}
    code_tested = dict(c3r=(c3r.get("code_sha256") or {}), smoke=(sm.get("code_sha256") or {}))
    same_code = all(all(v.get(f) == code_now[f] for f in TC.TRAINER_FILES) for v in code_tested.values())
    checks["GPU parts ran the current trainer code"] = bool(same_code)
    rec = dict(spec="docs/specs/CH3_MODEL.md 6", written=C3.now(), all_pass=bool(all(checks.values())), checks=checks,
               C3r=dict(checks=c3r.get("checks"), chosen=c3r.get("chosen"), chosen_deploy=c3r.get("chosen_deploy"),
                        best_epoch_diff=c3r.get("best_epoch_diff"), para_mae_rel_diff=c3r.get("para_mae_rel_diff"),
                        max_abs_dyhat=c3r.get("max_abs_dyhat"), gpu=c3r.get("gpu"), gpu_deploy=c3r.get("gpu_deploy")),
               C4=dict(c3r=dict(n_fits=n3, anti_max=a3, self_max=s3), smoke=dict(n_fits=ns, anti_max=as_, self_max=ss)),
               CT=sm.get("CT"), C_Y=dict((k, cpu["C_Y"][k]) for k in ("identity_residual_usable",
                                                                       "identity_residual_all_labels",
                                                                       "max_abs_diff_explicit_sums")),
               trainer_code_sha256=code_now,
               parts_sha256={n: C3.sha256(p) for n, p in parts.items()})
    C3.write_json(C3.RES / "C_tests.json", rec)
    print(json.dumps(dict(all_pass=rec["all_pass"], checks=checks), indent=1, ensure_ascii=False))
    return 0 if rec["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["cpu"]:
        sys.exit(cmd_cpu())
    if a == ["merge"]:
        sys.exit(cmd_merge())
    print(__doc__)
    sys.exit(2)
