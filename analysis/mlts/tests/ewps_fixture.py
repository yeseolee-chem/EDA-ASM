#!/usr/bin/env python3
"""ewps_fixture.py — synthetic stand-in inputs for eval_v2 / final_w / predict_v2 / summary_v2 tests (EWPS implementer).
Everything is written under the test root $MLT_SCRATCH (must be .../mlts/tmp/ewpstest) and $MLT_RESULTS; production
files are only read. reactot env (pandas). NOT real MLTS results: the 'MLTS' predictions are the LADDER dft-rung test
predictions plus seeded noise (C1: 1.0 x, C2: 0.6 x the dft error SD per target), 2 % of the reactions dropped from C1/C2
(coverage), D3_int = label disp + N(0, 0.01); K_report / arms.json / prereg written as stand-ins.

  python tests/ewps_fixture.py e        K / preds / arms / inputs stand-ins (run prereg_v2.py write first)
  python tests/ewps_fixture.py reports  stand-in A3 / A4 / X / FT / W / P reports for summary_v2 (after eval_v2 run)
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

if "ewpstest" not in str(M.MLT) or "ewpstest" not in str(M.RES):
    raise SystemExit(f"refusing: MLT_SCRATCH / MLT_RESULTS must point at the ewpstest root ({M.MLT}, {M.RES})")
PROD_PREDS = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/preds")


def cmd_e():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(20261104)
    L = pd.read_parquet(M.REPO / "analysis" / "d1_ladder" / "results" / "L_predictions.parquet")
    D = L[L.rung == "dft"].copy()
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "a_rxn", "b_rxn"])
    rx = sorted(set(S.a_rxn) | set(S.b_rxn))
    drop = set(rng.choice(rx, size=int(0.02 * len(rx)), replace=False).tolist())
    bad = set(S.pair_id[S.a_rxn.isin(drop) | S.b_rxn.isin(drop)].astype(str))
    sd = D.assign(e=D.yhat - D.y).groupby("target").e.std().to_dict()
    for name, f in (("C1", 1.0), ("C2", 0.6)):
        P = D[~D.pair_id.astype(str).isin(bad)].copy()
        P["rung"] = "mlts"
        # noise per (pair, target), identical for every arm / model of a pair (keeps LINE / base untouched)
        key = P.pair_id.astype(str) + "|" + P.target
        u = {k: rng.normal() for k in sorted(set(key))}
        P["yhat"] = P.yhat + f * P.target.map(sd) * key.map(u)
        M.write_parquet(M.MLT / "preds" / f"{name}_predictions.parquet", P)
    reps = json.loads((PROD_PREDS / "C1_representatives.json").read_text())
    M.write_json(M.MLT / "preds" / "C1_representatives.json", reps)
    rep_mlts = dict(REP=reps["REP"]["mlts"], REP8=reps["REP8"]["mlts"])
    K = dict(complete=True, n_todo=0, stop=None, primary="C1",
             configs={"C1": dict(refs="min", note="stand-in: LADDER dft predictions + 1.0 x noise"),
                      "C2": dict(refs="min", note="stand-in: LADDER dft predictions + 0.6 x noise")},
             inner_val={"C1": 0.40, "C2": 0.30}, delta_H=dict(note="stand-in"), leakage_C2={"pass_": True},
             reps={"C2": rep_mlts}, stand_in=True)
    M.write_json(M.RES / "K_report.json", K)
    M.write_json(M.MLT / "pipeline" / "arms.json", dict(engine="A", extra_arm_App=False, app_root=None, app_usable=None,
                                                         written=M.now(), writer="ewps_fixture (stand-in)"))
    lab = json.loads(M.LABELS.read_text())
    T = pd.DataFrame(dict(rxn_id=[int(d["rxn_id"]) for d in lab],
                          D3_int=[float(d["disp_dft"]) + rng.normal(0, 0.01) for d in lab]))
    T = T[~T.rxn_id.isin(drop)]
    M.write_parquet(M.MLT / "inputs" / "mlts_min.parquet", T)
    print(json.dumps(dict(dropped_rxn=len(drop), dropped_pairs=len(bad), sd=sd), indent=1, default=float))


def cmd_geom():
    """synthetic per-reaction structure table for all 9,266 targets (97 % ok, 85 % same TS, |dd| lognormal); the meta
    json matches the real folder count of the test root so eval_v2.geom_fresh accepts it."""
    import numpy as np
    import pandas as pd
    import eval_v2 as EV2
    rng = np.random.default_rng(7)
    T = M.targets()
    n = len(T)
    ok = rng.random(n) < 0.97
    same = rng.random(n) < 0.85
    dd = np.where(same, rng.lognormal(np.log(0.003), 0.8, n), rng.lognormal(np.log(0.08), 0.7, n))
    G = pd.DataFrame(dict(rxn_id=T.rxn_id, job_id=T.job_id, fold_T1=T.fold_T1, dipole=T.dipole, n_atoms=T.n_atoms,
                          arm="A", status=np.where(ok, "ok", "fail"), reason=np.where(ok, "", "ts_stand_in"),
                          compare="", same_ts=np.where(ok, same, None), rmsd_heavy=np.where(ok, dd * 0.6, np.nan),
                          dform_sorted=np.where(ok, dd, np.nan), dd_max=np.where(ok, dd, np.nan), dd_how="direct",
                          n_imag=np.where(ok, 1, np.nan), core_h=rng.lognormal(np.log(0.4), 0.3, n),
                          gpu_min=rng.lognormal(np.log(2.0), 0.3, n)))
    c, j = EV2.geom_paths("main")
    M.write_csv(c, G)
    M.write_json(j, dict(tag="main", root=str(M.MLT), engine="A", n_folders=EV2.count_folders(M.MLT / "ade" / "jobs"),
                         n_records=n, a4_sha256=EV2.fsha(M.RES / "A4_report.json"), csv_sha256=M.sha256(c),
                         written=M.now(), code="ewps_fixture stand-in"))
    print(f"synthetic geometry: {n} rows, ok {int(ok.sum())}, same {int((ok & same).sum())}")


def cmd_reports():
    E = json.loads((M.RES / "E_report.json").read_text())
    arms = dict(A=dict(n=300, n_ok=291, same_ts_frac=0.86, same_ts_frac_of_ok=0.886, dd_median=0.0031, dd_p90_all_ok=0.041,
                       rmsd_median=0.008, core_h_median=0.42, gpu_min_median=2.1, nimag1_frac_of_ok=0.97),
                Ap=dict(n=300, n_ok=280, same_ts_frac=0.74, dd_median=0.0035, rmsd_median=0.009, core_h_median=0.38,
                        gpu_min_median=1.6),
                App=dict(n=300, n_ok=285, same_ts_frac=0.88, dd_median=0.0011, rmsd_median=0.004, core_h_median=0.95,
                         gpu_min_median=2.1, dft_gradients_median=6))
    M.write_json(M.RES / "A3_report.json", dict(complete=True, arms=arms, stand_in=True))
    M.write_json(M.RES / "A4_report.json", dict(complete=True, n_ok=9001, n_fail=265, stand_in=True,
                                                folds={str(k): dict(complete=True) for k in range(-1, 5)}))
    M.write_json(M.RES / "X_report.json", dict(complete=True, stand_in=True, per_dipole={
        d: dict(pass_a=(d != "carbonyl ylide"), label=dict(dd_median_A=0.006 if d != "carbonyl ylide" else 0.014,
                                                            same_ts_frac=0.93 if d != "carbonyl ylide" else 0.81))
        for d in M.DIPOLES}))
    M.write_json(M.RES / "FT_report.json", dict(complete=True, stand_in=True, base="MACE-OMOL-0",
                                                ft2_a=dict(dd_median_A=0.004, dd_p90_A=0.02, rmsd_median_A=0.01,
                                                           same_ts_frac=0.93, nimag1_frac=0.97, pass_=True)))
    if not (M.RES / "P_report.json").is_file():
        M.write_json(M.RES / "P_report.json", dict(complete=True, stand_in=True, selfcheck_pass=False,
                                                   selfcheck=dict(R1_fold_replay=dict(pass_=None),
                                                                  R2_final_replay=dict(pass_=True)),
                                                   e2e=dict(pass_=None), cert=dict(n_ok=2, n_reactions=2,
                                                                                    fmax_comp_quantiles={"50": 0.0247},
                                                                                    threshold_eh_bohr=0.002,
                                                                                    frac_above_threshold=1.0)))
    print(json.dumps(dict(E_grade=E.get("grade"), written=["A3", "A4", "X", "FT", "P"]), indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["e"]:
        cmd_e()
    elif a == ["geom"]:
        cmd_geom()
    elif a == ["reports"]:
        cmd_reports()
    else:
        raise SystemExit(__doc__)
