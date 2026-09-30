#!/usr/bin/env python3
"""espley_fairness_ablation.py — REV4 §4-3: how much of the gap to Espley 2024 is geometry information?

Espley ds3 (3,510 rows): their 46 AM1 features and their DFT targets, their 80/10/10 split (test = the FIRST half of
the 20 % hold-out, seeds 22/23/14/1/2; replication against their ml_results.pkl is asserted), our pipeline
(train_ml_single._make_pipe: StandardScaler(X) -> est, y standardised), Espley-style tuning: GridSearchCV
(KFold 5, rs 23, MAE) ONCE on the seed-23 training rows, then the best estimator is refit on every seed's training
rows and scored on its test rows. One self-contained port of the user's original three files (common.py, the
ablation, the extra role arm); our features come from the rev 4 G1 parquet (xTB geometry).

arm                         features                                          targets              models
A_esp46_ourpipe             Espley46                                          5 Espley             KRR, SVR
C_esp46_roleAM1_role        Espley46 (AM1 distortions role-swapped)           role d1/d2           KRR, SVR
C0_esp46_role_noswapfeat    Espley46                                          role d1/d2           KRR, SVR
B_g1_esp46_plus_G1geom11    Espley46 + 11 distances, G1 (xTB geometry)        5 Espley             KRR, SVR
O_g1_ours46_tune23          our ESPLEY46, G1                                  5 Espley + role      KRR, SVR  (extra)
role d1/d2 = Espley's distortion_energy_1/2_dft re-assigned to dipole / dipolarophile with the swap vector from our
labels (|d1-t2| + |d2-t1| < |d1-t1| + |d2-t2|); the same vector swaps distortion_energy_1/2_am1 in the role features.

Rows (ABL_ROWS):
  common (default; DEVIATION from the original) every (arm, target, model) uses the rows usable by ALL arms — Espley
         features, an ok label (role targets), G1 xtb_status ok with NaN-free ESPLEY46, every target non-NaN — so the
         arms stay comparable. n and the dropped rxn ids are in every JSON and in
         results_rev4/espley_geometry_ablation_rows.json.
  own    the original rule: each arm on its own usable rows = ok label ∩ arm features ∩ target non-NaN (G1 arms also
         need G1). The original's first factor was "rev 3 xtb ok"; rev 3 was ok for every labelled rxn, so "ok label"
         selects the same rows. -> separate files (_ownrows).
(Also: refit on a clone of the best estimator — identical for KRR / SVR; n_jobs from $ESPLEY_NJOBS; GridSearchCV
error_score="raise" as compare_espley.our_pipeline — a failed grid fit stops the job instead of scoring NaN silently.)

  python espley_fairness_ablation.py list                   # job index, slice, done
  python espley_fairness_ablation.py run [--slice I] [--arm A ..] [--target T ..] [--model M ..]
  python espley_fairness_ablation.py aggregate
run: the filtered job list, then jobs[I::N_SLICES] -> $R5_SCRATCH/phaseA/ablation[_ownrows]/<arm>__<target>__<model>.json
(atomic; $ABL_SCRATCH overrides the directory). An existing JSON is skipped; one made from other inputs or rows is
refused (move it away).
aggregate -> results_rev4/espley_geometry_ablation[_ownrows].csv (mae = mean of the 5 seeds, se = mean per-seed
std(|err|)/sqrt(n_test) as Espley, sd_seeds, best params, n), _per_seed.csv, _rows.json; exit 1 if a JSON is
missing or stale.
Inputs: $ESPLEY_REPO_DATA, $ESPLEY_LABELS, $ESPLEY_FEAT_G1 (default $ESPLEY_OUT/xtb_features_g1.parquet).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, GRIDS, _make_pipe  # noqa: E402
from rev5_common import SCRATCH as R5_SCRATCH  # noqa: E402

warnings.filterwarnings("ignore")

ESP = Path(os.environ.get("ESPLEY_REPO_DATA", "/gpfs/tmp_cpu2/yeseo1ee/espley_compare"))
ESP_FEAT = ESP / "feature_selection/_f_selection/tt/manual_tt_solvent.pkl"
ESP_RES = ESP / "machine_learning/tt/solvent/ml_results.pkl"
LABELS = Path(os.environ.get("ESPLEY_LABELS") or HERE.parent.parent / "labels_all.json")
ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
FEAT = {"g1": Path(os.environ.get("ESPLEY_FEAT_G1") or ROOT / "xtb_features_g1.parquet")}   # parquet `geom` tag = key
GEOM_NOTE = {"none": "Espley AM1 features only", "g1": "xTB geometry (G1)"}
# ABL_ROWS=common (default, DEVIATION: one row set for every arm) | own (the original rule per arm: ok label, arm
# features and target non-NaN; G1 arms also need G1) — own reproduces the REV4 §0-3 numbers, separate files (_ownrows)
ROWS_MODE = os.environ.get("ABL_ROWS", "common")
assert ROWS_MODE in ("common", "own"), ROWS_MODE
SUFFIX = "" if ROWS_MODE == "common" else "_ownrows"
_ABL = Path(os.environ.get("ABL_SCRATCH") or R5_SCRATCH / "phaseA" / "ablation")
OUT = _ABL.with_name(_ABL.name + SUFFIX)             # <dir> (common) / <dir>_ownrows (own)
RES = HERE / "results_rev4"
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))
N_SLICES = 9                                          # = r4_ablation.sh --array=0-8
SEEDS = [22, 23, 14, 1, 2]
TUNE_SEED = 23
TGT = ["distortion_energy_1_dft", "distortion_energy_2_dft", "interaction_energies_dft", "e_barrier_dft",
       "q_barrier_dft"]
ROLE = ["dipole_role", "dipolarophile_role"]
GEO11 = ["dist_R_dip_ab", "dist_R_dip_bc", "dist_R_dip_ac", "dist_R_dph_ab", "dist_TS_dip_ab", "dist_TS_dip_bc",
         "dist_TS_dip_ac", "dist_TS_dph_ab", "dist_TS_form_ad", "dist_TS_form_be", "dist_TS_diag_ae"]
OURS46 = FEATURE_SETS["ESPLEY46"]
assert set(GEO11) <= set(OURS46)
MODELS = ["KRR_rbf", "SVR_rbf"]
ARMS = {  # arm: (feature blocks, targets, models, geometry of our features, extra)
    "A_esp46_ourpipe": (["esp46"], TGT, MODELS, "none", False),
    "C_esp46_roleAM1_role": (["esp46_role"], ROLE, MODELS, "none", False),
    "C0_esp46_role_noswapfeat": (["esp46"], ROLE, MODELS, "none", False),
    "B_g1_esp46_plus_G1geom11": (["esp46", "g1_geo11"], TGT, MODELS, "g1", False),     # REV4 §4-3 B_g1
    "O_g1_ours46_tune23": (["g1_ours46"], TGT + ROLE, MODELS, "g1", True),             # extra, not in §4-3
}
JOBS = [(a, t, m) for m in MODELS for a, (_, tg, ms, _, _) in ARMS.items() if m in ms for t in tg]  # KRR first
DEVIATION = ("rows = intersection of the usable rows of every (arm, target): Espley features, ok label, G1 "
             "xtb_status ok with NaN-free ESPLEY46, all targets non-NaN; the original used each arm's own rows")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    os.replace(tmp, path)


def write_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def espley_split(n, seed):
    """Espley's _perform_train_test_split: test = the FIRST half of the 20 % hold-out (= compare_espley)."""
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed)
    te, _ = train_test_split(rest, test_size=0.5, random_state=seed)
    return tr, te


def check_split(esp):
    """The split must reproduce the test targets stored in Espley's ml_results.pkl (as compare_espley.prep)."""
    res = pd.read_pickle(ESP_RES)
    for t in TGT:
        row = res[res.model_target == f"svr_{t}"].iloc[0]
        assert [int(s) for s in row.random_state] == SEEDS, (t, list(row.random_state))
        for s_i, s in enumerate(row.random_state):
            _, te = espley_split(len(esp), int(s))
            y = np.asarray(row.y_test_true[s_i], float).ravel()
            assert y.size == len(te) and np.allclose(esp[t].values[te], y, atol=1e-6), \
                f"split replication failed: {t} seed {s} (stored n_test {y.size}, ours {len(te)})"


def ours(g, ids):
    """Our ESPLEY46 (⊇ GEO11) on geometry g in Espley row order; NaN where the rxn is absent or xtb_status != ok."""
    p = FEAT[g]
    if not p.is_file():
        sys.exit(f"GATE {g}: {p} missing (r4_feat_aggregate.sh GEOM={g})")
    df = pd.read_parquet(p)
    tags = sorted(set(df["geom"].astype(str))) if "geom" in df else ["<no geom column>"]
    lacking = [c for c in ["rxn_id", "xtb_status"] + OURS46 if c not in df]
    bad = [f"geom tags {tags} != ['{g}']"] if tags != [g] else []
    if lacking:
        bad.append(f"columns missing: {lacking}")
    elif df.rxn_id.duplicated().any():
        bad.append(f"{int(df.rxn_id.duplicated().sum())} duplicate rxn_id")
    if bad:
        sys.exit(f"GATE {g} ({p}): " + "; ".join(bad))
    F = df.assign(rxn_id=df.rxn_id.astype(int)).set_index("rxn_id").reindex(ids)
    x = F[OURS46].to_numpy(dtype=float, copy=True)
    x[~F["xtb_status"].eq("ok").to_numpy()] = np.nan
    return pd.DataFrame(x, columns=OURS46)


def load():
    esp = pd.read_pickle(ESP_FEAT).reset_index(drop=True)
    rm = [c for c in esp.columns if any(t in c for t in ("contributions", "reactant", "structure", "path",
                                                           "reaction_number", "sum_distortion_energies"))]
    X = esp.drop(columns=rm)
    X = X[[c for c in X.columns if "_dft" not in c]]
    assert X.shape[1] == 46, X.shape
    ids = esp.reaction_number.astype(int).to_numpy()
    assert len(set(ids)) == len(ids), "duplicate reaction_number in Espley ds3"
    check_split(esp)
    lab = {int(r["rxn_id"]): r for r in json.load(open(LABELS))}
    have_lab = np.array([i in lab and lab[i]["status"] == "ok" for i in ids])
    d1o = np.array([float(lab[i]["d1_kcal"]) if h else np.nan for i, h in zip(ids, have_lab)])
    d2o = np.array([float(lab[i]["d2_kcal"]) if h else np.nan for i, h in zip(ids, have_lab)])
    t1, t2 = esp.distortion_energy_1_dft.to_numpy(float), esp.distortion_energy_2_dft.to_numpy(float)
    sw = have_lab & (np.abs(d1o - t2) + np.abs(d2o - t1) < np.abs(d1o - t1) + np.abs(d2o - t2))
    XR = X.copy()
    a1, a2 = X.distortion_energy_1_am1.to_numpy().copy(), X.distortion_energy_2_am1.to_numpy().copy()
    XR["distortion_energy_1_am1"] = np.where(sw, a2, a1); XR["distortion_energy_2_am1"] = np.where(sw, a1, a2)
    blocks = {"esp46": X.to_numpy(float), "esp46_role": XR.to_numpy(float)}
    for g in FEAT:
        f = ours(g, ids)
        blocks[f"{g}_geo11"], blocks[f"{g}_ours46"] = f[GEO11].to_numpy(), f.to_numpy()
    Y = {t: esp[t].to_numpy(float) for t in TGT}
    # without an ok label the role is unknown -> NaN (the original kept index order there; those rows had no
    # xTB features, so it never used them)
    Y["dipole_role"] = np.where(have_lab, np.where(sw, t2, t1), np.nan)
    Y["dipolarophile_role"] = np.where(have_lab, np.where(sw, t1, t2), np.nan)

    esp_ok = ~np.isnan(blocks["esp46"]).any(axis=1)
    own = {(a, t): esp_ok & ~np.isnan(np.hstack([blocks[b] for b in bl])).any(axis=1) & ~np.isnan(Y[t])
           for a, (bl, tg, _, _, _) in ARMS.items() for t in tg}
    common = np.logical_and.reduce(list(own.values()))       # DEVIATION: one row set for every arm
    if not common.any():
        sys.exit("GATE: no Espley ds3 row is usable by every arm")
    why = {"espley_feature_nan": ~esp_ok, "no_ok_label": ~have_lab,
           **{f"{g}_not_ok_or_nan": np.isnan(blocks[f"{g}_ours46"]).any(axis=1) for g in FEAT},
           **{f"nan_{t}": np.isnan(Y[t]) for t in TGT}}
    rows = dict(n_espley=len(ids), n_common=int(common.sum()),
                rows_sha256=hashlib.sha256(" ".join(map(str, ids[common])).encode()).hexdigest(),
                deviation=DEVIATION, split_replicated=True, n_role_swapped=int(sw[common].sum()),
                dropped={int(i): [k for k, m in why.items() if m[j]] for j, i in enumerate(ids) if not common[j]},
                own={f"{a}__{t}": dict(n=int(m.sum()), dropped_by_intersection=ids[m & ~common].tolist())
                     for (a, t), m in own.items()})
    inputs = {k: dict(path=str(p), sha256=sha256(p)) for k, p in
              [("espley_features", ESP_FEAT), ("labels", LABELS)] + [(f"feat_{g}", FEAT[g]) for g in FEAT]}
    # the original `have` (rev 3 xtb ok) -> ok label: the same rows, rev 3 was ok for every labelled rxn
    orig = {k: have_lab & m for k, m in own.items()}
    return dict(n=len(ids), blocks=blocks, Y=Y, common=common, orig=orig, rows=rows, rows_sha256=rows["rows_sha256"],
                inputs=inputs)


def fingerprint(o):
    return o["rows_sha256"], {k: v["sha256"] for k, v in o["inputs"].items()}


def run_one(D, a, t, m):
    f = OUT / f"{a}__{t}__{m}.json"
    if f.exists():
        if fingerprint(json.loads(f.read_text())) != fingerprint(D):
            sys.exit(f"{f} exists but was made from other inputs / rows — move it away")
        print(f"skip: {f.name}", flush=True)
        return
    bl, _, _, geom, extra = ARMS[a]
    Xa, y, n = np.hstack([D["blocks"][b] for b in bl]), D["Y"][t], D["n"]
    ok = D["common"] if ROWS_MODE == "common" else D["orig"][(a, t)]
    t0 = time.time()
    tr23, _ = espley_split(n, TUNE_SEED); tr23 = tr23[ok[tr23]]
    est, grid = GRIDS[m]
    gs = GridSearchCV(_make_pipe(est), grid, cv=KFold(5, shuffle=True, random_state=TUNE_SEED),
                      scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise").fit(Xa[tr23], y[tr23])
    per = {}
    for s in SEEDS:
        tr, te = espley_split(n, s); tr, te = tr[ok[tr]], te[ok[te]]
        e = np.abs(clone(gs.best_estimator_).fit(Xa[tr], y[tr]).predict(Xa[te]) - y[te])
        per[s] = dict(mae=float(e.mean()), se=float(e.std() / np.sqrt(len(e))), n=int(len(e)), n_train=int(len(tr)))
    own = D["rows"]["own"][f"{a}__{t}"]
    r = dict(arm=a, target=t, model=m, our_geom=geom, extra=extra, feature_blocks=bl, n_features=int(Xa.shape[1]),
             best=str(gs.best_params_), best_params=gs.best_params_, cv_mae_tune=float(-gs.best_score_),
             mae=float(np.mean([v["mae"] for v in per.values()])), se=float(np.mean([v["se"] for v in per.values()])),
             per_seed=per, n_tune=int(len(tr23)), n_rows=int(ok.sum()), n_rows_own=own["n"],
             dropped_by_intersection=own["dropped_by_intersection"], rows_sha256=D["rows_sha256"],
             inputs=D["inputs"], rows_mode=ROWS_MODE, deviation=DEVIATION if ROWS_MODE == "common" else None, tune_seed=TUNE_SEED, n_jobs=N_JOBS,
             sklearn=sklearn.__version__, sec=time.time() - t0)
    write_json(f, r)
    print(f"{a:26s} {t:26s} {m:8s} MAE {r['mae']:.3f}  n {r['n_rows']} ({r['sec']:.0f}s) {r['best']}", flush=True)


def run(arms, targets, models, sl):
    jobs = [(a, t, m) for a, t, m in JOBS
            if (not arms or a in arms) and (not targets or t in targets) and (not models or m in models)]
    if sl is not None:
        jobs = jobs[sl::N_SLICES]
    print(f"{len(jobs)} jobs: {jobs}", flush=True)
    if not jobs:
        return
    D = load()
    R = D["rows"]
    print(f"rows usable by every arm: {R['n_common']}/{R['n_espley']} (sha256 {R['rows_sha256'][:12]}); "
          f"dropped {len(R['dropped'])}: {sorted(R['dropped'])[:30]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for a, t, m in jobs:
        run_one(D, a, t, m)
    print("ALL DONE", flush=True)


def aggregate():
    D = load()
    arm_i = {a: k for k, a in enumerate(ARMS)}
    rows, seeds, missing, stale = [], [], [], []
    for a, t, m in sorted(JOBS, key=lambda j: (arm_i[j[0]], ARMS[j[0]][1].index(j[1]), MODELS.index(j[2]))):
        f = OUT / f"{a}__{t}__{m}.json"
        if not f.is_file():
            missing.append(f.name); continue
        r = json.loads(f.read_text())
        if fingerprint(r) != fingerprint(D):
            stale.append(f.name); continue
        ps = pd.DataFrame([dict(seed=int(s), **v) for s, v in r["per_seed"].items()]).rename(columns={"n": "n_test"})
        assert sorted(ps.seed) == sorted(SEEDS), (f.name, ps.seed.tolist())
        g = ARMS[a][3]
        key = dict(arm=a, our_geom=g, geom_note=GEOM_NOTE[g], extra=ARMS[a][4], target=t, model=m)
        rows.append(dict(key, mae=r["mae"], se=r["se"], sd_seeds=float(ps.mae.std()), n_rows=r["n_rows"],
                         n_tune=r["n_tune"], n_test_mean=float(ps.n_test.mean()), best_params=r["best"],
                         cv_mae_tune=r["cv_mae_tune"], sec=r["sec"]))
        seeds.append(ps.assign(**key)[list(key) + ["seed", "mae", "se", "n_test", "n_train"]])
    if missing or stale:
        sys.exit(f"aggregate: {len(missing)} missing {missing[:10]}, {len(stale)} made from other inputs / rows "
                 f"{stale[:10]} — (re)run r4_ablation.sh first")
    RES.mkdir(exist_ok=True)
    tab = pd.DataFrame(rows)
    write_csv(tab.round(4), RES / f"espley_geometry_ablation{SUFFIX}.csv")
    write_csv(pd.concat(seeds, ignore_index=True).round(4), RES / f"espley_geometry_ablation{SUFFIX}_per_seed.csv")
    arms = {a: dict(feature_blocks=bl, targets=tg, models=ms, our_geom=g, extra=x)
            for a, (bl, tg, ms, g, x) in ARMS.items()}
    write_json(RES / f"espley_geometry_ablation{SUFFIX}_rows.json",
               dict(D["rows"], inputs=D["inputs"], arms=arms, seeds=SEEDS, tune_seed=TUNE_SEED))
    print(f"rows: {D['rows']['n_common']}/{D['rows']['n_espley']}; wrote {RES}/espley_geometry_ablation*.csv, _rows.json")
    for m in MODELS:
        p = tab[tab.model == m].pivot(index="arm", columns="target", values="mae")
        p = p.reindex(index=[a for a in ARMS if a in p.index], columns=[t for t in TGT + ROLE if t in p.columns])
        print(f"\n=== {m}: test MAE, mean of seeds {SEEDS} (kcal/mol) ===\n{p.round(3).to_string()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--slice", type=int, help=f"run jobs[slice::{N_SLICES}] of the filtered list")
    r.add_argument("--arm", nargs="+", choices=list(ARMS))
    r.add_argument("--target", nargs="+", choices=TGT + ROLE)
    r.add_argument("--model", nargs="+", choices=MODELS)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, (arm, t, m) in enumerate(JOBS):
            done = "done" if (OUT / f"{arm}__{t}__{m}.json").is_file() else ""
            print(f"{k:3d}  slice {k % N_SLICES:2d}  {arm:26s} {t:26s} {m:8s} {done}")
    elif a.cmd == "run":
        if a.slice is not None and not 0 <= a.slice < N_SLICES:
            sys.exit(f"--slice {a.slice} out of range 0..{N_SLICES - 1}")
        run(a.arm, a.target, a.model, a.slice)
    else:
        aggregate()


if __name__ == "__main__":
    main()
