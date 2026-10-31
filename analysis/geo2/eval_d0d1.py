#!/usr/bin/env python3
"""GEO2b Phase E for decision D0D1 (design step 13): the geo rung trained once on D1 + D0 pairs (split T1D0), evaluated
descriptively. D1 and D0 are reported separately and never pooled; there is no comparison verdict and no g2 / dft column
(the GEO2.md 10 geo-vs-g2 verdict is dropped by the user decision D0D1, 2026-10-31).

  python eval_d0d1.py run        -> results/ E_metrics.csv, E3_metrics.csv, E3_bends.csv, E3_bend_rates.json,
                                    E3_goal.csv, E_app.csv, E_bin.csv (D1) + E_d0_metrics.csv (D0) + E_report.json
  python eval_d0d1.py selftest   -> results/selftest_d0d1/ (the same files + E_selftest.json): geo := the LADDER g2
                                    predictions and unit records (D1; must reproduce the E2 numbers) + synthetic D0
                                    predictions on the real D0 pairs; exit 3 when a check fails
Exit 0 = done, 2 = an input missing or the GEO1b D0 build still preliminary (E-9), 3 = STOP (e.g. D0_pairs.csv rebuilt
after the splits: E-9). env_espley (analysis/geo2/jobs/m3_cpu.slurm M3ENV=espley or analysis/d1_pair/jobs/py.slurm
PYENV=espley, = LADDER evaluate.py E1-1: the FU2 Phase E scoring code, scipy for the bend CIs; RUNBOOK_geo2b.md E).

Selections (all from the geo unit validation records, which are D1 inner-validation pairs only in T1D0):
  REP·S / REP·SH (LADDER 13-1), REP8 (GEO2.md 9-1), ENS4·S / ENS4·SH, and every model x arm; the D0 rows use the same
  choices (no D0 validation exists).
D1 (T1 test pairs; the LADDER E2 computation, reused unchanged from analysis/d1_ladder/evaluate.py):
  E_metrics.csv    eval_geo2.metrics on own_geo (P_main reference pairs with a geo prediction): REP8, REP·SH, REP·S,
                   ENS4·SH, ENS4·S x 10 targets x position; para core-bootstrap CI (2,000, seed 20261019)
  E3_metrics.csv   LADDER metrics_e1 (E2_metrics.csv format): AW, GNN, TF, XGB, REP, ENS4, LINE, null x S / SH, pair set
                   own (P_geo), sets P_main / A_main / P_sens / P_main_no_d0_overlap x positions
  E3_bends.csv, E3_bend_rates.json   FU2 bends_tab + d1_eval.bend_rates (LADDER bends_e1)
  E3_goal.csv      LADDER 13-2 (goal_e1: REP primary, ENS4 secondary, models descriptive) + bend_passed + REP8 rows:
                   oi and cpcm para MAE <= 0.13, bend sensitivity >= 0.7, false-positive rate <= 0.1 (point estimates;
                   'surely' = CI upper <= 0.13). The only absolute target; descriptive, no comparison
  E_app.csv, E_bin.csv   eval_geo2.app_metrics / by_class on own_geo (bins = GEO1b E_rxn.csv, worse end of the pair)
D0 (the fold-held-out D0 test pairs, D0_pairs.csv ok rows):
  E_d0_metrics.csv sel x target x edge_kind (all, sibling, dipole, leaf) x subset (all, no_m1_twin): n, MAE, RMSE,
                   component-bootstrap 95 % CI of the MAE (component = 'D0/<component>'), null MAE (mean |y|)
Decisions (results/DECISIONS_ZE.md): E-* entries. Assumptions about the trainer output: see check_predictions().
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import eval_geo2 as EG  # noqa: E402     (metrics / select / boot_diff / app_metrics / by_class / pairs_frame / reps)
import geo2_common as C  # noqa: E402

LAD_DIR = REPO / "analysis" / "d1_ladder"
GEO_PRED, GEO_UNITS = EG.GEO_PRED, EG.GEO_UNITS                 # train_atoms.py collect --stage 3 --with-xgb
OUT_SELFTEST = C.RES / "selftest_d0d1"
RUNG = "geo"
TARGETS, MODELS, ARMS = EG.TARGETS, EG.MODELS, ("S", "SH")     # ARMS in LADDER order (EG.ARMS = REP8 tie order)
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]                                     # = train_atoms.PRED_COLS (LADDER L_predictions.parquet)
D0_REF_KIND = "d0"                                               # D0_pairs.csv ref_kind / pos (DECISIONS_PAIR P-layout)
S_SPLITS = C.RES / "S_splits_d0d1.json"                          # = d0d1_splits.S_SPLITS (d0.d0_pairs_sha256; E-9)
EDGE_KINDS = ("sibling", "dipole", "leaf")
Y_TOL = 1e-9                                                     # = FU2 Y_TOL (y = label dd)
E2_TOL, LINE_TOL = 1e-12, 1e-8                                   # selftest: E2 reproduction (LINE = SH base: E1-5 1e-9)
N_BOOT, SEED = EG.N_BOOT, EG.SEED
D0_SELS = ([f"{m}·{a}" for a in ARMS for m in MODELS] + ["REP·S", "REP·SH", "REP8", "ENS4·S", "ENS4·SH", "null"])
DECISIONS_E = {
    "E-1": "thin eval_d0d1.py instead of 'eval_geo2.py run --rungs geo' and instead of a new evaluate.py e3: it imports "
           "analysis/d1_ladder/evaluate.py unchanged and calls its E2 functions (fu2_eval, prep_S, read_lab, label_info, "
           "model_frames, check_pair_attrs, ref_set, derived_frames, gate_pairs, check_line, annotate_all, metrics_e1, "
           "met_index, bends_e1, goal_e1) on rung geo, and eval_geo2 metrics / reps / select / app_metrics / by_class / "
           "goal; LADDER outputs are untouched (no LADDER file is written or edited)",
    "E-2": "REP / REP8 / ENS4 from the geo unit records val_kcal_mae (L_report_geo.json + L_xgb_report_geo.json; T1D0 "
           "inner validation = D1 only) before the prediction file is read; the D0 rows use the same choices",
    "E-3": "LINE of the geo rung = the SH units' baseline column (base = beta x dH as trained; T1D0 fits beta on its own "
           "training pairs), checked equal across the 4 SH models and zero for S (LADDER check_line); no input-table "
           "recomputation (the stage-3 LINE of the trainer is the reference); null = 0",
    "E-4": "D1 rows = every non-'d0' row (pair ids of S_folds.csv); D0 rows = ref_kind 'd0' (pair ids of D0_pairs.csv); "
           "the two id sets must agree, unknown ids = STOP; a D1 row fold must be its S_folds.csv fold_T1 (review R-5)",
    "E-5": "D0 evaluation pairs = predicted D0 pairs with D0_pairs.csv ok true (label eligibility); predicted not-ok pairs "
           "are counted and left out; the pair fold must equal d0_folds.csv of both ends (else STOP: leakage)",
    "E-6": "D0 CI = component bootstrap (component = D0_pairs.csv core 'D0/<component>', 2,000 resamples, seed 20261019; "
           "= eval_geo2.boot_diff(e, 0, component) with the resampling weights cached per component list)",
    "E-7": "D0 subsets all and no_m1_twin (neither end has a stereo-free D1 twin of T1 fold -1, d0_twins.csv m1; "
           "DECISIONS_PAIR P-y-m1 recommendation); edge kinds all / sibling / dipole / leaf",
    "E-8": "no verdict, no g2 / dft rows or columns; eval_geo2.py run (the GEO2 comparison) is disabled under D0D1",
    "E-9": "run scores only the final D0 build the energy models were trained on: D0_pairs_report.json g2_source 'g2c' "
           "(else not ready, exit 2) and D0_pairs.csv sha256 = S_splits_d0d1.json d0.d0_pairs_sha256 (else STOP: rebuilt "
           "after the splits / training); selftest reads the D0 files without these two rules (DECISIONS_ZE 연결 S-4)",
}


def _not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _jdump(obj):
    return json.dumps(_clean(obj), indent=1)


def _clean(o):
    """JSON-able copy: numpy -> python, NaN / inf -> None, tuples -> lists, Path -> str (= LADDER evaluate._clean)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _csv(out, name, df, outputs):
    p = C.write_atomic(out / name, df.to_csv(index=False))
    outputs[str(p)] = C.sha256(p)


def _json(out, name, obj, outputs):
    p = C.write_atomic(out / name, _jdump(obj))
    outputs[str(p)] = C.sha256(p)


# ---------------------------------------------------------------- unit records -> REP / REP8 (before any test score)
def load_units_geo(paths, rung=RUNG):
    """unit records of the geo rung (report 'units' {name: {val_kcal_mae}}) -> rows rung, arm, model, fold, target, val.
    Names are split on '__': rung = 2nd token, arm / model / f<k> = the last three (robust to an extra split token);
    a unit listed in two reports must carry the same values; every arm x model x target needs folds 0-4."""
    rows, seen = [], {}
    for p in paths:
        if not Path(p).is_file():
            _not_ready(f"{p} missing (train_atoms.py collect --stage 3 --with-xgb / train_xgb.py check --stage 3)")
        for name, u in (json.loads(Path(p).read_text()).get("units") or {}).items():
            tok = name.split("__")
            if len(tok) < 5 or tok[1] != rung:
                continue
            arm, model, f = tok[-3], tok[-2], tok[-1]
            v = u.get("val_kcal_mae") or {}
            if not v:
                continue
            key = (arm, model, int(f[1:]))
            vv = {t: float(v[t]) for t in TARGETS if t in v}
            if key in seen and seen[key] != vv:
                C.stop(f"unit {name}: val_kcal_mae differs between the unit reports")
            if key in seen:
                continue
            seen[key] = vv
            rows += [dict(rung=rung, arm=arm, model=model, fold=key[2], target=t, val=x) for t, x in vv.items()]
    U = pd.DataFrame(rows, columns=["rung", "arm", "model", "fold", "target", "val"])
    bad = [f"{a}|{m}|{t}" for a in ARMS for m in MODELS for t in TARGETS
           if sorted(U[(U.arm == a) & (U.model == m) & (U.target == t)].fold.tolist()) != [0, 1, 2, 3, 4]]
    if bad or not np.isfinite(U.val.to_numpy(float)).all():
        C.stop(f"geo unit records: arm|model|target without exactly folds 0-4 or non-finite val: {bad[:6]}")
    return U


def unit_json_check(U, rung=RUNG):
    """$LAD/units/L__<rung>__<arm>__<model>__f<k>.json, when present: same val_kcal_mae, no smoke unit (else STOP)."""
    sys.path.insert(0, str(LAD_DIR))
    import lad_common as L  # noqa: E402
    n, bad = 0, []
    for (a, m, k), g in U.groupby(["arm", "model", "fold"]):
        p = L.LAD / "units" / f"L__{rung}__{a}__{m}__f{k}.json"
        if not p.is_file():
            continue
        n += 1
        r = json.loads(p.read_text())
        v = r.get("val_kcal_mae") or {}
        if r.get("smoke") or any(abs(float(v.get(t, np.nan)) - x) > 0 or not np.isfinite(float(v.get(t, np.nan)))
                                 for t, x in zip(g.target, g.val)):
            bad.append(p.name)
    if bad:
        C.stop(f"unit jsons disagree with the collected unit records (or smoke units): {bad[:6]}")
    return dict(n_unit_jsons_compared=n, n_units=int(len(U.groupby(["arm", "model", "fold"]))))


def rep_tables(U):
    rep, rep8, mval = EG.reps(U)
    out = {f"{r}|{a}|{t}": dict(model=m, mean_val={mo: float(mval[(r, a, mo, t)]) for mo in MODELS})
           for (r, a, t), m in rep.items()}
    out8 = {f"{r}|{t}": dict(arm=v[0], model=v[1]) for (r, t), v in rep8.items()}
    return rep, rep8, out, out8


# ---------------------------------------------------------------- predictions
def check_predictions(G, d1_ids, d0_ids, d1_fold=None):
    """Assumptions about L_predictions_geo.parquet (stage-3 collect with the T1D0 split; LADDER agent): columns
    include the LADDER PRED_COLS; rung 'geo' only; D1 rows = pair ids of S_folds.csv with ref_kind / pos of S_folds;
    D0 rows = ref_kind 'd0' (pos 'd0') with the pair ids D##### of D0_pairs.csv; fold = the outer fold 0-4 of the
    unit (D1: = S_folds.csv fold_T1 when d1_fold {pair_id: fold_T1} is given: T1 test = fold_T1 == k, inputs.py /
    FU2 evaluate D6); one row per pair x arm x model x target. -> (D1 rows, D0 rows) or STOP."""
    miss = [c for c in PRED_COLS if c not in G.columns]
    if miss:
        C.stop(f"{GEO_PRED.name} lacks columns {miss} (columns {list(G.columns)})")
    G = G[PRED_COLS].copy()
    G["pair_id"], G["fold"] = G.pair_id.astype(str), G.fold.astype(int)
    G["ref_kind"], G["pos"] = G.ref_kind.fillna("").astype(str), G.pos.fillna("").astype(str)
    if set(G.rung.astype(str)) != {RUNG}:
        C.stop(f"{GEO_PRED.name} holds rungs {sorted(set(G.rung.astype(str)))}, not '{RUNG}' only")
    is0 = (G.ref_kind == D0_REF_KIND).to_numpy()
    in0, in1 = G.pair_id.isin(d0_ids).to_numpy(), G.pair_id.isin(d1_ids).to_numpy()
    bad = int((is0 != in0).sum() + (~is0 & ~in1).sum())
    if bad:
        C.stop(f"{bad} prediction rows whose ref_kind 'd0' does not match D0_pairs.csv ids, or D1 rows whose pair id is "
               f"not in S_folds.csv (e.g. {G.pair_id[(is0 != in0) | (~is0 & ~in1)].head(3).tolist()})")
    if not set(G.fold) <= {0, 1, 2, 3, 4}:
        C.stop(f"prediction folds {sorted(set(G.fold))} (must be 0-4)")
    if d1_fold is not None:                                      # the D1 test lists are not re-derived from the split
        g1 = G[~is0]
        wf = (g1.pair_id.map(d1_fold).fillna(-99).astype(int) != g1.fold).to_numpy()
        if wf.any():
            C.stop(f"{int(wf.sum())} D1 prediction rows whose fold != S_folds.csv fold_T1 (a T1 test pair in another "
                   f"fold: leak or wrong split), e.g. {g1.pair_id[wf].head(3).tolist()}")
    if G.duplicated(["pair_id", "arm", "model", "target"]).any():
        C.stop("duplicated prediction rows (pair x arm x model x target)")
    if set(zip(G.arm, G.model)) != {(a, m) for a in ARMS for m in MODELS} or set(G.target) != set(TARGETS):
        C.stop(f"arm x model {sorted(set(zip(G.arm, G.model)))} / targets {sorted(set(G.target))} incomplete")
    return G[~is0].reset_index(drop=True), G[is0].reset_index(drop=True)


# ---------------------------------------------------------------- D1 (LADDER E2 functions on rung geo)
def line_from_sh(fr):
    """E-3: LINE of the geo rung = the AW·SH base column (check_line then requires every SH model to carry the same
    base / beta and the S rows none)."""
    f = fr[(RUNG, "SH", "AW")]
    line = f.assign(yhat=f.base.to_numpy(float), source="LINE")[["pair_id", "fold", "target", "y", "yhat", "base",
                                                                  "beta", "source"]]
    betas = {int(k): {t: float(g[g.target == t].beta.iloc[0]) for t in TARGETS if (g.target == t).any()}
             for k, g in line.groupby("fold")}
    return line.reset_index(drop=True), betas


def add_bend_passed(G):
    """bend_passed = sensitivity >= 0.7 and FPR <= 0.1 (target rows); rung_arm rows: oi and cpcm both (geo2 E3-5)."""
    import evaluate as LE
    G = G.copy()
    key = {(r.rung, r.arm, r.model, r.target): bool(r.crit_sens) and bool(r.crit_fpr)
           for r in G.itertuples(index=False) if r.level == "target"}
    G["bend_passed"] = [key[(r.rung, r.arm, r.model, r.target)] if r.level == "target" else
                        (all(key.get((r.rung, r.arm, r.model, t), False) for t in LE.GOAL_TARGETS)
                         if r.level == "rung_arm" else None) for r in G.itertuples(index=False)]
    return G


def goal_rep8(MI, rates, rep8, rungs=(RUNG,)):
    """LADDER 13-2 on the REP8 choice per target (GEO2.md 9-1): rows + {rung|REP8: verdict} (descriptive under D0D1)."""
    import evaluate as LE
    rows, verdict = [], {}
    for r in rungs:
        per, detail = {}, {}
        for t in LE.PRIMARY9:
            a, m = rep8[(r, t)]
            pp, pa = MI[(r, a, m, "own", t, "P_main", "p")], MI[(r, a, m, "own", t, "P_main", "all")]
            br = rates[f"{r}|{a}|{m}|own"]["all"][t]
            mae, hi, sens, fpr = pp["mae"], pp["mae_ci_hi"], br["sens"], br["fpr"]
            c_mae = bool(LE._fin(mae) and mae <= LE.GOAL["mae"])
            c_sens = bool(LE._fin(sens) and sens >= LE.GOAL["sens"])
            c_fpr = bool(LE._fin(fpr) and fpr <= LE.GOAL["fpr"])
            ok = bool(c_mae and c_sens and c_fpr)
            sure = bool(ok and LE._fin(hi) and hi <= LE.GOAL["mae"])
            per[t] = dict(passed=ok, surely=sure, bend=bool(c_sens and c_fpr), mae=c_mae)
            detail[t] = dict(arm=a, model=m, para_mae=mae, para_mae_ci_lo=pp["mae_ci_lo"], para_mae_ci_hi=hi,
                             n_para=pp["n"], sens=sens, n_bent_dft=br["n_bent_dft"], fpr=fpr,
                             n_straight_dft=br["n_straight_dft"])
            r2 = pa["r2_fam_median"]
            rows.append(dict(
                level="target", rung=r, arm=a, model="REP8", role="primary (GEO2.md 9-1 REP8)", source=m, target=t,
                judged=bool(t in LE.GOAL_TARGETS), pair_set_name=f"P_{r}", n_para=pp["n"], para_mae=mae,
                para_mae_ci_lo=pp["mae_ci_lo"], para_mae_ci_hi=hi, n_cores=pp["n_cores"], n_fam=br["n_fam"],
                n_bent_dft=br["n_bent_dft"], sens=sens, sens_ci_lo=br["sens_ci"][0], sens_ci_hi=br["sens_ci"][1],
                n_straight_dft=br["n_straight_dft"], fpr=fpr, fpr_ci_lo=br["fpr_ci"][0], fpr_ci_hi=br["fpr_ci"][1],
                crit_mae=c_mae, crit_sens=c_sens, crit_fpr=c_fpr, passed=ok, surely_passed=sure, mae_all=pa["mae"],
                mae_null_all=pa["mae_null"],
                beats_null_point=bool(LE._fin(pa["mae"]) and LE._fin(pa["mae_null"]) and pa["mae"] < pa["mae_null"]),
                r2_fam_median=r2, stage1_descriptive=(bool(r2 >= LE.STAGE1_R2) if LE._fin(r2) else None),
                stage0_rev9=LE.STAGE0_NOTE, conclusion=None, bend_passed=bool(c_sens and c_fpr)))
        gt = LE.GOAL_TARGETS
        v = dict(passed=all(per[t]["passed"] for t in gt), surely_passed=all(per[t]["surely"] for t in gt),
                 bend_passed=all(per[t]["bend"] for t in gt), mae_passed=all(per[t]["mae"] for t in gt),
                 choices={t: list(rep8[(r, t)]) for t in LE.PRIMARY9}, detail={t: detail[t] for t in gt})
        verdict[f"{r}|REP8"] = v
        rows.append(dict(level="rung", rung=r, arm="; ".join(f"{t}: {rep8[(r, t)][0]}" for t in gt), model="REP8",
                         role="primary (GEO2.md 9-1 REP8)", source="; ".join(f"{t}: {rep8[(r, t)][1]}" for t in gt),
                         target="+".join(gt), judged=True, pair_set_name=f"P_{r}", passed=v["passed"],
                         surely_passed=v["surely_passed"], crit_mae=v["mae_passed"], bend_passed=v["bend_passed"],
                         conclusion=f"LADDER 13-2 goal {'reached' if v['passed'] else 'not reached'} (MAE part "
                                    f"{v['mae_passed']}, bend part {v['bend_passed']}); descriptive, no comparison"))
    return pd.DataFrame(rows), verdict


def d1_part(P1, rep, rep8):
    """the LADDER E2 computation on the geo D1 test rows -> dict of tables + gates (nothing is written here)."""
    sys.path.insert(0, str(LAD_DIR))
    import evaluate as LE  # noqa: E402     (unchanged LADDER file; E1/E2 functions)
    t0 = time.time()
    EV, info = LE.fu2_eval()
    lab = LE.label_info(LE.DEFAULT_LABELS)
    S = LE.prep_S(pd.read_csv(LE.S_FOLDS, dtype={"pair_id": str}))
    Y = LE.read_lab(lab["path"])
    LE.check_pair_attrs(P1, S)
    rungs = (RUNG,)
    fr = LE.model_frames(P1, rungs)
    base = fr[(RUNG, "S", "AW")]
    base = base[base.target == TARGETS[0]]
    tests = {int(k): g.pair_id.astype(str).tolist() for k, g in base.groupby("fold")}
    ref = {RUNG: LE.ref_set(tests, RUNG)}
    gm = EV.pair_sets_gate({f"{RUNG}|{a}|{m}": fr[(RUNG, a, m)] for a in ARMS for m in MODELS}, ref[RUNG], Y)
    lines = {RUNG: line_from_sh(fr)}
    fr = LE.derived_frames(fr, lines, rep, rungs)
    ga = LE.gate_pairs(EV, fr, ref, Y, {}, rungs)
    lchk = LE.check_line(fr, lines, rungs)
    A = LE.annotate_all(EV, fr, S, None, rungs)
    Mt = LE.metrics_e1(EV, A)
    MI = LE.met_index(Mt)
    Bd, rates, fams = LE.bends_e1(EV, A, S, rep)
    goal, verdict, _ = LE.goal_e1(MI, rates, rungs)
    g8, verdict8 = goal_rep8(MI, rates, rep8, rungs)
    goal = pd.concat([add_bend_passed(goal), g8], ignore_index=True)
    F = EG.pairs_frame()
    own = EG.common_pairs(P1, F, rungs)
    M = EG.metrics(P1, F, rep, rep8, rungs, own, "own_geo")
    app = EG.app_metrics(P1, F, rep, rep8, own)
    binr = EG.by_class(P1, F, rep, rep8, own)
    gmae = EG.goal(M)
    cross = {}                                           # E_metrics (eval_geo2) vs E3 (LADDER) on the same pairs
    for sel, a, m in [("REP8", None, None)] + [(f"{m}·{a}", a, m) for m in ("REP", "ENS4") for a in ARMS]:
        for t in LE.GOAL_TARGETS:
            aa, mm = rep8[(RUNG, t)] if sel == "REP8" else (a, m)
            x = MI[(RUNG, aa, mm, "own", t, "P_main", "p")]
            y = M[(M.sel == sel) & (M.target == t) & (M.pos == "p")]
            cross[f"{sel}|{t}"] = dict(e3=x["mae"], e_metrics=float(y.mae.iloc[0]), n_e3=int(x["n"]), n_e=int(y.n.iloc[0]))
    cross_ok = all(abs(v["e3"] - v["e_metrics"]) <= 1e-9 and v["n_e3"] == v["n_e"] for v in cross.values())
    v8 = verdict8[f"{RUNG}|REP8"]
    goal_rep = {
        "REP8": dict(role="primary (GEO2.md 9-1)", passed=v8["passed"], surely_passed=v8["surely_passed"],
                     mae_part=v8["mae_passed"], bend_part=v8["bend_passed"], choices=v8["choices"], detail=v8["detail"]),
        **{f"{m}·{a}": dict(role=verdict[f"{RUNG}|{a}|{m}"]["role"], passed=verdict[f"{RUNG}|{a}|{m}"]["passed"],
                            surely_passed=verdict[f"{RUNG}|{a}|{m}"]["surely_passed"])
           for a in ARMS for m in ("REP", "ENS4") + MODELS}}
    return dict(Mt=Mt, MI=MI, Bd=Bd, rates=rates, fams=fams, goal=goal, verdict=verdict, verdict8=verdict8, M=M,
                app=app, bin=binr, goal_mae=gmae, goal_rep=goal_rep, cross=dict(ok=cross_ok, rows=cross),
                gates=dict(models=gm, all=ga, n_test_pairs=len(ref[RUNG])), line_check=lchk,
                line_betas=lines[RUNG][1], n_own=len(own), n_own_para=int(F.set_index("pair_id").pos.reindex(own).eq("p")
                                                                          .sum()),
                fu2_evaluate=info, labels=lab, seconds=time.time() - t0)


# ---------------------------------------------------------------- D0
def load_d0(final=True):
    """D0_pairs.csv / d0_folds.csv / d0_twins.csv under the reader rule P-stop (D0_pairs_report.json stop false and
    its outputs sha256 = each file) -> (pairs frame indexed by pair_id, {rxn_id: fold}, m1 rxn ids, info). final (run;
    E-9): the report must be the final build (g2_source 'g2c', else exit 2) and D0_pairs.csv = the one the splits
    recorded (S_splits_d0d1.json d0.d0_pairs_sha256, else STOP); checked first, so a preliminary report is exit 2."""
    split_rec = None
    if final:
        try:
            src = json.loads(C.D0_PAIRS_REPORT.read_text()).get("g2_source")
        except (OSError, ValueError) as e:
            _not_ready(f"{C.D0_PAIRS_REPORT} unreadable ({type(e).__name__})")
        if src != "g2c":
            _not_ready(f"{C.D0_PAIRS_REPORT.name} g2_source {src!r}: the preliminary GEO1b D0 build (the final one comes "
                       f"with the geo1b -> geo2b merge M1)")
        if not S_SPLITS.is_file() or not C.D0_PAIRS_CSV.is_file():
            _not_ready(f"{S_SPLITS} or {C.D0_PAIRS_CSV} missing (d0d1_splits.py build)")
        want = (json.loads(S_SPLITS.read_text()).get("d0") or {}).get("d0_pairs_sha256")
        now_ = C.sha256(C.D0_PAIRS_CSV)
        if want != now_:
            C.stop(f"{C.D0_PAIRS_CSV} sha256 {now_[:12]} != {S_SPLITS.name} d0.d0_pairs_sha256 {str(want)[:12]}: D0 pairs "
                   f"rebuilt after the splits / training cannot be scored (E-9)")
        split_rec = {str(S_SPLITS): C.sha256(S_SPLITS)}
    bad = [w for w in (C.report_output_ok(C.D0_PAIRS_REPORT, p) for p in (C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS)) if w]
    if bad:
        C.stop("D0 inputs: " + "; ".join(bad))
    D = pd.read_csv(C.D0_PAIRS_CSV, dtype={"pair_id": str})
    need = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "core", "edge_kind", "ok"] + [f"dd_{t}" for t in TARGETS]
    if [c for c in need if c not in D.columns] or D.pair_id.duplicated().any():
        C.stop(f"{C.D0_PAIRS_CSV.name}: columns {list(D.columns)} lack {need} or duplicated pair ids")
    D["ok"] = D.ok.astype(str).map({"True": True, "False": False})
    if D.ok.isna().any() or (D.ref_kind.astype(str) != D0_REF_KIND).any() or not set(D.edge_kind) <= set(EDGE_KINDS):
        C.stop(f"{C.D0_PAIRS_CSV.name}: ok not boolean, ref_kind not 'd0' or edge kinds {sorted(set(D.edge_kind))}")
    Fo = pd.read_csv(C.D0_FOLDS, dtype=str, keep_default_na=False)
    folds = {int(r): int(f) for r, f in zip(Fo.rxn_id, Fo.fold)}
    Tw = pd.read_csv(C.D0_TWINS, dtype=str, keep_default_na=False)
    m1 = {int(r) for r, x in zip(Tw.rxn_id, Tw.m1) if x == "True"} if len(Tw) else set()
    info = {str(p): C.sha256(p) for p in (C.D0_PAIRS_CSV, C.D0_FOLDS, C.D0_TWINS, C.D0_PAIRS_REPORT)}
    info.update(split_rec or {})
    return D.set_index("pair_id"), folds, m1, info


def d0_frames(P0, D, folds, rep, rep8):
    """D0 test rows -> ({sel: frame pair_id, fold, target, y, yhat, source}, checks). STOP when an arm x model x target
    has another (fold, pair) list, y != D0_pairs.csv dd (1e-9), a non-finite yhat, a pair in two folds, or a pair fold
    != d0_folds.csv of its ends (E-5); not-ok pairs are counted and left out."""
    probs, ref = [], None
    for (a, m, t), g in P0.groupby(["arm", "model", "target"]):
        s = frozenset(zip(g.fold.astype(int), g.pair_id))
        ref = s if ref is None else ref
        if s != ref:
            probs.append(f"{a}|{m}|{t}: (fold, pair) list differs ({len(ref - s)} missing, {len(s - ref)} extra)")
        dy = np.abs(g.y.to_numpy(float) - D.loc[g.pair_id, f"dd_{t}"].to_numpy(float))
        if not np.isfinite(dy).all() or (len(dy) and dy.max() > Y_TOL):
            probs.append(f"{a}|{m}|{t}: y differs from D0_pairs.csv dd_{t} (max {np.nanmax(dy):.3g})")
        if not np.isfinite(g.yhat.to_numpy(float)).all():
            probs.append(f"{a}|{m}|{t}: non-finite yhat")
    if ref is not None:
        pf = {}
        for k, p in ref:
            pf.setdefault(p, set()).add(k)
        two = sorted(p for p, ks in pf.items() if len(ks) > 1)
        wrong = sorted(p for p, ks in pf.items() if len(ks) == 1 and (folds.get(int(D.at[p, "a_rxn"])) != next(iter(ks))
                                                                   or folds.get(int(D.at[p, "b_rxn"])) != next(iter(ks))))
        if two or wrong:
            probs.append(f"D0 pairs in two folds {two[:3]} or tested in another fold than d0_folds.csv {wrong[:3]}")
    if probs or ref is None:
        C.stop("D0 test rows: " + ("; ".join(probs[:10]) if probs else "none"))
    keep = D.ok.reindex(P0.pair_id).to_numpy(bool)
    n_not_ok = len({p for p, k in zip(P0.pair_id, keep) if not k})
    P0 = P0[keep]
    cols = ["pair_id", "fold", "target", "y", "yhat"]
    fr = {f"{m}·{a}": P0[(P0.arm == a) & (P0.model == m)][cols].assign(source=m) for a in ARMS for m in MODELS}
    def rows_of(sel, t):
        f = fr[sel]
        return f[f.target == t]
    for a in ARMS:
        fr[f"REP·{a}"] = pd.concat([rows_of(f"{rep[(RUNG, a, t)]}·{a}", t) for t in TARGETS], ignore_index=True)
        q = P0[P0.arm == a].groupby(["pair_id", "target"], sort=False).agg(
            fold=("fold", "first"), y=("y", "first"), yhat=("yhat", "mean"), n=("model", "nunique")).reset_index()
        if (q.n != len(MODELS)).any():
            C.stop(f"ENS4·{a} (D0): a pair / target without {len(MODELS)} model predictions")
        fr[f"ENS4·{a}"] = q[cols].assign(source="ENS4")
    fr["REP8"] = pd.concat([rows_of(f"{rep8[(RUNG, t)][1]}·{rep8[(RUNG, t)][0]}", t)
                            .assign(source=f"{rep8[(RUNG, t)][1]}·{rep8[(RUNG, t)][0]}") for t in TARGETS],
                           ignore_index=True)
    fr["null"] = fr["AW·S"].assign(yhat=0.0, source="0")
    return fr, dict(n_pairs_predicted=len(ref), n_not_ok_left_out=n_not_ok, n_pairs_evaluated=len(ref) - n_not_ok)


_W: dict = {}


def boot_mae_ci(e, cores, n_boot=N_BOOT, seed=SEED):
    """MAE and its core (component) bootstrap 95 % CI = eval_geo2.boot_diff(e, 0, cores)[0:3]; the resampling weights
    depend only on (number of cores, seed), so they are cached."""
    ae = np.abs(np.asarray(e, float))
    uc, inv = np.unique(cores, return_inverse=True)
    k = len(uc)
    if (k, n_boot, seed) not in _W:
        rng = np.random.default_rng(seed)
        _W[(k, n_boot, seed)] = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    w = _W[(k, n_boot, seed)]
    dstar = (w @ np.bincount(inv, ae)) / (w @ np.bincount(inv)) - (w @ np.bincount(inv, np.zeros(len(ae)))) / \
        (w @ np.bincount(inv))
    lo, hi = np.percentile(dstar, [2.5, 97.5])
    return float(ae.mean() - 0.0), float(lo), float(hi)


def d0_metrics(fr, D, m1):
    """E_d0_metrics.csv rows: sel x target x edge_kind (all + kinds) x subset (all, no_m1_twin)."""
    rows = []
    a_m1 = D.a_rxn.astype(int).isin(m1) | D.b_rxn.astype(int).isin(m1)
    for sel in D0_SELS:
        f = fr[sel]
        for t in TARGETS:
            q = f[f.target == t]
            Dq = D.loc[q.pair_id]
            e, y = (q.yhat - q.y).to_numpy(float), q.y.to_numpy(float)
            src = sorted(set(q.source))
            for ek in ("all",) + EDGE_KINDS:
                me = np.ones(len(q), bool) if ek == "all" else (Dq.edge_kind == ek).to_numpy()
                for sub in ("all", "no_m1_twin"):
                    m = me & (np.ones(len(q), bool) if sub == "all" else ~a_m1.reindex(q.pair_id).to_numpy(bool))
                    r = dict(sel=sel, arm=sel.split("·")[1] if "·" in sel else ("S+SH" if sel == "REP8" else "-"),
                             model=sel.split("·")[0], source="; ".join(src), target=t, edge_kind=ek, subset=sub,
                             n=int(m.sum()), n_components=int(Dq.core[m].nunique()), n_folds=int(q.fold[m].nunique()))
                    if m.any():
                        mae, lo, hi = boot_mae_ci(e[m], Dq.core.to_numpy()[m])
                        mn = float(np.abs(y[m]).mean())
                        r.update(mae=mae, rmse=float(np.sqrt(np.mean(e[m] ** 2))), mae_ci_lo=lo, mae_ci_hi=hi,
                                 mae_null=mn, mae_rel_null=mae / mn if mn > 0 else np.nan)
                    rows.append(r)
    cols = ["sel", "arm", "model", "source", "target", "edge_kind", "subset", "n", "n_components", "n_folds", "mae",
            "rmse", "mae_ci_lo", "mae_ci_hi", "mae_null", "mae_rel_null"]
    return pd.DataFrame(rows).reindex(columns=cols)


# ---------------------------------------------------------------- pipeline
def evaluate(U, G, out, mode, extra=None):
    """REP from U (before G is looked at by the callers' design), D1 then D0, write every output under out."""
    t0 = time.time()
    rep, rep8, rep_out, rep8_out = rep_tables(U)
    sys.path.insert(0, str(LAD_DIR))
    import evaluate as LE  # noqa: E402
    Sf = pd.read_csv(LE.S_FOLDS, usecols=["pair_id", "fold_T1"], dtype={"pair_id": str})
    S_ids = set(Sf.pair_id)
    D, folds, m1, d0_info = load_d0(final=mode == "run")                 # E-9: the final D0 build only in run
    P1, P0 = check_predictions(G, S_ids, set(D.index), dict(zip(Sf.pair_id, Sf.fold_T1.astype(int))))
    d1 = d1_part(P1, rep, rep8)
    fr0, d0chk = d0_frames(P0, D, folds, rep, rep8)
    D0M = d0_metrics(fr0, D, m1)
    outputs = {}
    for name, df in (("E_metrics.csv", d1["M"]), ("E3_metrics.csv", d1["Mt"]), ("E3_bends.csv", d1["Bd"]),
                     ("E3_goal.csv", d1["goal"]), ("E_app.csv", d1["app"]), ("E_bin.csv", d1["bin"]),
                     ("E_d0_metrics.csv", D0M)):
        _csv(out, name, df, outputs)
    _json(out, "E3_bend_rates.json", dict(key="rung|arm|model|pair_set (own = P_geo, D1 T1 test pairs)",
                                          rates=d1["rates"], families=d1["fams"],
                                          note="d1_eval.bend_rates on FU2 bends_tab rows (= E2_bend_rates.json format)"),
          outputs)
    h = D0M[(D0M.edge_kind == "all") & (D0M.subset == "all") & D0M.sel.isin(["REP8", "REP·S", "REP·SH", "ENS4·S",
                                                                               "ENS4·SH"])]
    d0_head = {f"{r.sel}|{r.target}": dict(n=r.n, mae=r.mae, ci=[r.mae_ci_lo, r.mae_ci_hi], mae_null=r.mae_null)
               for r in h.itertuples() if r.target in ("oi", "cpcm", "barrier")}
    report = dict(
        mode=mode, spec="docs/specs/GEO2.md 9 read with GEO1B.md 4-3 and decision D0D1 (design step 13)",
        verdict="none (user decision D0D1: no comparison model, no geo-vs-g2 verdict; descriptive results)",
        pooled=False, note="D1 and D0 are separate tables (E_* / E3_* = D1, E_d0_metrics = D0); never pooled",
        date=time.strftime("%F %T"), command=" ".join(sys.argv[1:]),
        inputs=dict(predictions=str(GEO_PRED), units=[str(p) for p in GEO_UNITS], d0=d0_info, **(extra or {})),
        n=dict(d1_test_pairs=d1["gates"]["n_test_pairs"], d1_own_geo=d1["n_own"], d1_own_geo_para=d1["n_own_para"],
               **{f"d0_{k}": v for k, v in d0chk.items()},
               d0_by_edge_kind={str(k): int(v) for k, v in D.loc[sorted(set(fr0["AW·S"].pair_id))].edge_kind
                                .value_counts().items()}),
        representatives=rep_out, rep8=rep8_out,
        rep_rule="REP: per arm x target the lowest 5-fold mean val_kcal_mae of AW, GNN, TF, XGB (ties in that order); "
                 "REP8: over both arms (ties SH before S, then that order); validation = D1 inner-validation pairs of "
                 "T1D0; the D0 rows use the same choices",
        goal_d1=dict(rule="LADDER 13-2 (oi and cpcm: para MAE <= 0.13, bend sensitivity >= 0.7, FPR <= 0.1; own P_geo "
                          "T1 test pairs, point estimates; surely = para MAE CI upper <= 0.13); the only absolute "
                          "target, descriptive", by_selection=d1["goal_rep"], goal_mae_eval_geo2=d1["goal_mae"],
                     e_metrics_vs_e3=d1["cross"]),
        d0_headline=d0_head, gates=d1["gates"], line_check=d1["line_check"], line_betas=d1["line_betas"],
        fu2_evaluate=d1["fu2_evaluate"], labels_d1=d1["labels"], decisions=DECISIONS_E, outputs=outputs,
        seconds=dict(total=time.time() - t0, d1=d1["seconds"]))
    return report, dict(rep=rep, rep8=rep8, d1=d1, fr0=fr0, D0M=D0M, D=D, m1=m1, P0=P0, P1=P1, folds=folds)


def _stop3(fn, *a, **k):
    """eval_geo2 helpers STOP with SystemExit('STOP: ...') = exit 1; re-raised as exit 3 (this file's STOP code)."""
    try:
        return fn(*a, **k)
    except SystemExit as e:
        if isinstance(e.code, str) and e.code.startswith("STOP"):
            print(e.code, file=sys.stderr, flush=True)
            sys.exit(3)
        raise


def cmd_run():
    U = load_units_geo(GEO_UNITS)
    uj = unit_json_check(U)
    rep_geo = json.loads(GEO_UNITS[0].read_text())
    if not GEO_PRED.is_file():
        _not_ready(f"{GEO_PRED} missing (train_atoms.py collect --stage 3 --with-xgb)")
    if not EG.GEO1B_RXN.is_file():                      # E_bin bins; without it every pair would be silently 'na'
        _not_ready(f"{EG.GEO1B_RXN} missing (the final GEO1b E1 results must be in this worktree)")
    psha = C.sha256(GEO_PRED)
    if rep_geo.get("predictions_sha256") not in (None, psha):
        C.stop(f"{GEO_PRED.name} changed after {GEO_UNITS[0].name} (predictions_sha256)")
    load_d0(final=True)                                 # E-9 before anything is written (exit 2 / 3)
    G = pd.read_parquet(GEO_PRED)
    extra = dict(predictions_sha256=psha, unit_reports={str(p): C.sha256(p) for p in GEO_UNITS if p.is_file()},
                 geo1b_e_rxn={str(EG.GEO1B_RXN): C.sha256(EG.GEO1B_RXN)},
                 unit_jsons=uj, l_report_geo=dict(stage=rep_geo.get("stage"), rungs=rep_geo.get("rungs"),
                                                  labels=rep_geo.get("labels"), models=rep_geo.get("models")))
    _json(C.RES, "E_report.json", dict(mode="run", status="running (or killed / stopped): not a finished evaluation",
                                       started=time.strftime("%F %T")), {})    # the final report is written last
    report, _ = _stop3(evaluate, U, G, C.RES, "run", extra)
    outputs = report["outputs"]
    _json(C.RES, "E_report.json", report, outputs)
    g = report["goal_d1"]["by_selection"]["REP8"]
    print(json.dumps(dict(n=report["n"], rep8_goal=dict(passed=g["passed"], mae_part=g["mae_part"],
                                                        bend_part=g["bend_part"]), d0_headline=report["d0_headline"]),
                     indent=1, default=float), flush=True)
    return 0


# ---------------------------------------------------------------- selftest (geo := LADDER g2; synthetic D0)
D0_NOISE = {"AW": 0.0, "GNN": 0.3, "TF": 0.5, "XGB": 0.8}          # SH arm; S arm = 1.5 x (AW·S nonzero)


def synthetic_d0(D, folds, seed=SEED):
    """synthetic D0 prediction rows on the real ok D0 pairs: AW·SH exact (yhat = y), the others y + N(0, s)."""
    rng = np.random.default_rng(seed)
    ok = D[D.ok]
    fk = np.array([folds[int(a)] for a in ok.a_rxn])
    rows = []
    for a in ARMS:
        for m in MODELS:
            s = D0_NOISE[m] * (1.0 if a == "SH" else 1.5) + (0.2 if (a, m) == ("S", "AW") else 0.0)
            for t in TARGETS:
                y = ok[f"dd_{t}"].to_numpy(float)
                rows.append(pd.DataFrame(dict(pair_id=ok.index.to_numpy(), ref_kind=D0_REF_KIND, pos=D0_REF_KIND,
                                              fold=fk, rung=RUNG, arm=a, model=m, target=t, y=y,
                                              yhat=y + rng.normal(0.0, s, len(y)) if s else y.copy(), base=0.0,
                                              beta=0.0, yhat_seed_sd=0.0)))
    return pd.concat(rows, ignore_index=True)


def _num_close(a, b, tol, rel=True):
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if np.isnan(x) or np.isnan(y):
        return bool(np.isnan(x) and np.isnan(y))
    return abs(x - y) <= tol * (max(1.0, abs(y)) if rel else 1.0)


def _expect_stop(fn):
    try:
        fn()
    except SystemExit as e:
        return e.code == 3
    return False


def selftest_checks(res, report):
    sys.path.insert(0, str(LAD_DIR))
    import evaluate as LE  # noqa: E402
    ck, det = {}, {}
    rep, rep8, d1 = res["rep"], res["rep8"], res["d1"]
    # the 'run' unit readers on the real LADDER g2 files: load_units_geo = eval_geo2.load_units, unit jsons agree
    Ug = load_units_geo(EG.BASE_UNITS, rung="g2")
    Ue = EG.load_units(EG.BASE_UNITS)
    Ue = Ue[Ue.rung == "g2"].drop_duplicates(["arm", "model", "fold", "target"])
    kk = ["arm", "model", "fold", "target"]
    Jg = Ug.merge(Ue, on=kk, how="outer", suffixes=("", "_e"), indicator=True)
    uj = unit_json_check(Ug, rung="g2")
    ck["run_unit_readers_on_g2"] = bool((Jg["_merge"] == "both").all() and len(Jg) == 8 * 5 * len(TARGETS)
                                        and (Jg.val == Jg.val_e).all() and uj["n_unit_jsons_compared"] == 40)
    det["run_unit_readers_on_g2"] = dict(rows=int(len(Jg)), unit_jsons=uj)
    E2 = json.loads(LE.L.RES.joinpath("E2_report.json").read_text())["representatives"]["g2"]
    ck["rep_equal_E2_g2"] = all(rep[(RUNG, a, t)] == E2[a][t]["model"] for a in ARMS for t in TARGETS)
    exp8 = {t: min(((E2[a][t]["mean_val_kcal_mae"][m], EG.ARMS.index(a), MODELS.index(m)), (a, m))
                   for a in ARMS for m in MODELS)[1] for t in TARGETS}
    ck["rep8_equal_E2_means"] = all(tuple(rep8[(RUNG, t)]) == exp8[t] for t in TARGETS)
    # E_metrics (eval_geo2) own_geo = E2_metrics g2 own P_main p (as eval_geo2 selftest; 1e-9)
    EM = pd.read_csv(LE.L.RES / "E2_metrics.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    EMp = EM[(EM.pair_set == "own") & (EM.set == "P_main") & (EM.pos == "p") & (EM.rung == "g2")]
    M, bad = d1["M"], []
    for s_my, mo, arm in (("REP·SH", "REP", "SH"), ("REP·S", "REP", "S"), ("ENS4·SH", "ENS4", "SH"), ("ENS4·S", "ENS4", "S")):
        for t in TARGETS:
            a = M[(M.sel == s_my) & (M.target == t) & (M.pos == "p")]
            b = EMp[(EMp.arm == arm) & (EMp.model == mo) & (EMp.target == t)]
            if len(a) != 1 or len(b) != 1 or int(a.n.iloc[0]) != int(b.n.iloc[0]) or any(
                    abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) > 1e-9 for c in ("mae", "mae_ci_lo", "mae_ci_hi")):
                bad.append(f"{s_my} {t}")
    ck["E_metrics_equal_E2"], det["E_metrics"] = not bad, bad[:10]
    # E3_metrics geo own = E2_metrics g2 own (every row and number; LINE within 1e-8 absolute, E-3)
    num = ["n", "mae", "rmse", "mae_null", "mae_rel_null", "n_cores", "mae_ci_lo", "mae_ci_hi", "n_fam", "r2_fam_median",
           "r2_fam_q1", "r2_fam_q3", "rho_err_median"]
    k = ["arm", "model", "target", "set", "pos"]
    Mt = d1["Mt"]
    J = Mt.merge(EM[(EM.rung == "g2") & (EM.pair_set == "own")], on=k, how="outer", suffixes=("", "_e2"), indicator=True)
    nb = {"rows_not_in_both": int((J["_merge"] != "both").sum())}
    Jb = J[J["_merge"] == "both"]
    for c in num + ["source"]:
        isl = (Jb.model == "LINE").to_numpy()
        nb[c] = int(sum(not (_num_close(x, y, LINE_TOL, rel=False) if li else _num_close(x, y, E2_TOL))
                        for x, y, li in zip(Jb[c].tolist(), Jb[f"{c}_e2"].tolist(), isl)))
    ck["E3_metrics_equal_E2"], det["E3_metrics"] = all(v == 0 for v in nb.values()), nb
    # bend rates and goal rows = E2 (LINE bend calls recorded only)
    BR = json.loads((LE.L.RES / "E2_bend_rates.json").read_text())["rates"]
    rb, rl = [], []
    for key, v in d1["rates"].items():
        other = "g2|" + key[len(f"{RUNG}|"):]
        diff = [] if _same(_clean(v), BR.get(other)) else [key]
        (rl if "|LINE|" in key else rb).extend(diff)
    ck["bend_rates_equal_E2"], det["bend_rates"], det["bend_rates_LINE_differ"] = not rb, rb[:10], rl
    G2 = pd.read_csv(LE.L.RES / "E2_goal.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    G2 = G2[(G2.rung == "g2") & G2.level.isin(["target", "rung_arm"])]
    G3 = d1["goal"][d1["goal"].level.isin(["target", "rung_arm"]) & (d1["goal"].model != "REP8")]
    JG = G3.merge(G2, on=["level", "arm", "model", "target"], how="outer", suffixes=("", "_e2"), indicator=True)
    gb = int((JG["_merge"] != "both").sum()) + sum(
        int(sum(not _num_close(x, y, E2_TOL) for x, y in zip(JG[c].tolist(), JG[f"{c}_e2"].tolist())))
        for c in ("passed", "surely_passed", "para_mae", "para_mae_ci_hi", "sens", "fpr"))
    ck["goal_equal_E2"], det["goal_problems"] = gb == 0 and len(G3) > 0, gb
    ck["goal_rep8_rows"] = bool(((d1["goal"].level == "rung") & (d1["goal"].model == "REP8")).sum() == 1)
    ck["e_metrics_vs_e3"] = bool(d1["cross"]["ok"])
    # D0 (synthetic): AW·SH exact, ENS4 = mean, null = mean |y|, REP / REP8 = the chosen model rows, CI = boot_diff
    D0M, fr0 = res["D0M"], res["fr0"]
    x = D0M[D0M.sel == "AW·SH"]
    ck["d0_exact_model_zero"] = bool(len(x) and (x.mae.fillna(0).abs().max() == 0) and (x.mae_ci_hi.fillna(0).abs().max() == 0))
    e = fr0["ENS4·SH"].merge(pd.concat([fr0[f"{m}·SH"] for m in MODELS]).groupby(["pair_id", "target"]).yhat.mean()
                             .rename("mean4").reset_index(), on=["pair_id", "target"])
    ck["d0_ens4_mean"] = bool(len(e) and np.allclose(e.yhat, e.mean4, rtol=0, atol=1e-12))
    nl = D0M[(D0M.sel == "null") & (D0M.n > 0)]
    ck["d0_null_is_mean_abs_y"] = bool(len(nl) and np.allclose(nl.mae, nl.mae_null, rtol=0, atol=1e-12))
    okr = True
    for t in TARGETS:
        for a in ARMS:
            a_ = D0M[(D0M.sel == f"REP·{a}") & (D0M.target == t)].mae.to_numpy()
            b_ = D0M[(D0M.sel == f"{rep[(RUNG, a, t)]}·{a}") & (D0M.target == t)].mae.to_numpy()
            okr &= bool(np.allclose(a_, b_, equal_nan=True, rtol=0, atol=0))
        a8, m8 = rep8[(RUNG, t)]
        okr &= bool(np.allclose(D0M[(D0M.sel == "REP8") & (D0M.target == t)].mae.to_numpy(),
                                D0M[(D0M.sel == f"{m8}·{a8}") & (D0M.target == t)].mae.to_numpy(), equal_nan=True,
                                rtol=0, atol=0))
    ck["d0_rep_rows_equal_chosen"] = okr
    w = D0M[(D0M.subset == "all") & (D0M.edge_kind != "all")].groupby(["sel", "target"]).n.sum()
    al = D0M[(D0M.subset == "all") & (D0M.edge_kind == "all")].set_index(["sel", "target"]).n
    ck["d0_edge_kinds_add_up"] = bool((w.reindex(al.index) == al).all())
    q = D0M[D0M.n > 0]
    ck["d0_ci_brackets_mae"] = bool(((q.mae_ci_lo <= q.mae + 1e-12) & (q.mae <= q.mae_ci_hi + 1e-12)).all())
    D, bd = res["D"], []
    for sel, t in (("GNN·SH", "oi"), ("XGB·S", "cpcm"), ("REP8", "barrier")):
        f = fr0[sel][fr0[sel].target == t]
        a_ = boot_mae_ci((f.yhat - f.y).to_numpy(float), D.loc[f.pair_id].core.to_numpy())
        b_ = EG.boot_diff((f.yhat - f.y).to_numpy(float), np.zeros(len(f)), D.loc[f.pair_id].core.to_numpy())[:3]
        bd.append(max(abs(u - v) for u, v in zip(a_, b_)))
    ck["d0_boot_equal_eval_geo2"], det["d0_boot_max_diff"] = bool(max(bd) <= 1e-12), max(bd)
    # negative checks: each must STOP (exit 3)
    P0, folds, rep_, rep8_ = res["P0"], res["folds"], rep, rep8
    p = P0.pair_id.iloc[0]
    t1 = P0.copy()
    t1.loc[t1.pair_id == p, "fold"] = (int(P0.fold.iloc[0]) + 1) % 5
    t2 = P0.copy()
    t2.loc[t2.index[0], "y"] += 0.01
    t3 = P0.drop(index=P0.index[0])
    ck["neg_d0_wrong_fold_stops"] = _expect_stop(lambda: d0_frames(t1, D, folds, rep_, rep8_))
    ck["neg_d0_wrong_y_stops"] = _expect_stop(lambda: d0_frames(t2, D, folds, rep_, rep8_))
    ck["neg_d0_missing_row_stops"] = _expect_stop(lambda: d0_frames(t3, D, folds, rep_, rep8_))
    G = pd.concat([res["P1"], P0], ignore_index=True)
    t4 = G.copy()
    t4.loc[t4.index[-1], "ref_kind"] = "reference"
    S_ids = set(res["P1"].pair_id)
    ck["neg_ref_kind_mismatch_stops"] = _expect_stop(lambda: check_predictions(t4, S_ids, set(D.index)))
    t5 = G[~((G.arm == "SH") & (G.model == "TF"))]
    ck["neg_missing_model_stops"] = _expect_stop(lambda: check_predictions(t5, S_ids, set(D.index)))
    Sf = pd.read_csv(LE.S_FOLDS, usecols=["pair_id", "fold_T1"], dtype={"pair_id": str})
    t6 = G.copy()
    t6.loc[t6.index[0], "fold"] = (int(t6.fold.iloc[0]) + 1) % 5          # a D1 row (P1 first) in another fold
    ck["neg_d1_wrong_fold_stops"] = _expect_stop(lambda: check_predictions(
        t6, S_ids, set(D.index), dict(zip(Sf.pair_id, Sf.fold_T1.astype(int)))))
    EV, _ = LE.fu2_eval()
    f1 = res["P1"]
    fr = LE.model_frames(f1, (RUNG,))
    b0 = fr[(RUNG, "S", "AW")]
    b0 = b0[b0.target == TARGETS[0]]
    ref = LE.ref_set({int(k): g.pair_id.tolist() for k, g in b0.groupby("fold")}, RUNG)
    fr[(RUNG, "SH", "GNN")] = fr[(RUNG, "SH", "GNN")].iloc[1:]
    Y = LE.read_lab(LE.DEFAULT_LABELS)
    ck["neg_d1_missing_pair_stops"] = _expect_stop(lambda: EV.pair_sets_gate(
        {f"{RUNG}|{a}|{m}": fr[(RUNG, a, m)] for a in ARMS for m in MODELS}, ref, Y))
    return ck, det


def _same(a, b, tol=E2_TOL):
    """json-like equality, numbers within tol (relative), NaN / None equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return _num_close(a, b, tol)


def cmd_selftest():
    t0 = time.time()
    U = EG.load_units(EG.BASE_UNITS)
    U = U[U.rung == "g2"].assign(rung=RUNG)
    L0 = pd.read_parquet(EG.BASE_PRED)
    G1 = L0[L0.rung == "g2"].assign(rung=RUNG)
    D, folds, m1, _ = load_d0(final=False)              # E-9 rules are for run (the preliminary D0 build is fine here)
    G = pd.concat([G1, synthetic_d0(D, folds)], ignore_index=True)
    report, res = _stop3(evaluate, U, G, OUT_SELFTEST, "selftest",
                         dict(selftest="geo := LADDER g2 predictions / units (L_predictions.parquet, L_report.json, "
                                       "L_xgb_report.json); D0 = synthetic predictions on the real ok D0 pairs"))
    ck, det = selftest_checks(res, report)
    report.update(selftest=dict(checks=ck, detail=det, all_pass=all(ck.values()), seconds=time.time() - t0))
    _json(OUT_SELFTEST, "E_selftest.json", report, report["outputs"])
    for k, v in ck.items():
        print(f"[selftest] {k}: {'ok' if v else 'FAIL'}", flush=True)
    print(f"[selftest] {'all checks passed' if all(ck.values()) else 'FAILED'} ({time.time() - t0:.0f} s) "
          f"-> {OUT_SELFTEST}", flush=True)
    return 0 if all(ck.values()) else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["run"]:
        sys.exit(cmd_run())
    if a == ["selftest"]:
        sys.exit(cmd_selftest())
    raise SystemExit(__doc__)
