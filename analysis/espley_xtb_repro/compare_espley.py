#!/usr/bin/env python3
"""compare_espley.py — rev 4 comparison with Espley et al. 2024 (ds3, [3+2]) on their reactions, targets and splits.

Held equal:
  reactions   the 3,510 reactions of Espley's ds3 ML set (their `reaction_number` = Coley reaction id = our rxn_id),
              in their row order
  targets     their DFT target values (B3LYP-D3(BJ)/def2-TZVP SMD(water), Gaussian), so every side is scored against
              the same numbers
  splits      their protocol exactly: train_test_split(test_size=0.2, rs=seed), then the first half of the remaining
              20 % is the test set; seeds 22/23/14/1/2. Verified against the test targets stored in their ml_results.pkl.
  test rows   every side (Espley, ours G1) is scored on the INTERSECTION of the rows usable by all sides = labelled
              (ok label) AND G1-usable (xtb_status ok, NaN-free ESPLEY46/73): 3,327 of 3,510, asserted in plot
              (N_SCORED); the dropped rxn ids are reported. G1 also TRAINS on those rows (Espley stored models /
              retrains: their rows).
  metric      test MAE averaged over the 5 seeds; error bar = their definition, std(|error|)/sqrt(n_test) per seed,
              averaged over seeds
What differs: the geometry (Espley AM1-optimised; ours G1 GFN2-xTB/ALPB re-optimised from the DFT TS and references),
the features (46 AM1 vs xTB ESPLEY46 / ESPLEY73) and the model family / tuning.
MAIN (fixed a priori, no best-of): G1 · ESPLEY46 · KRR vs Espley SVR (their published best). d1/d2 are compared by role
(dipole / dipolarophile): Espley's distortion_energy_1/2 follow a reactant index, so their 46 AM1 features (AM1
distortions re-assigned by the same swap vector) are retrained on the role targets (a) with their own protocol, whose
re-run is checked against their hps.pkl on their index d1/d2 (flag espley_protocol_replicated), and (b) with our pipeline.
APPENDIX: best-of-models per side; index-based d1/d2.

  ESPLEY_GEOM=g1 ESPLEY_FEAT=<parquet> python compare_espley.py prep       # checks + row masks
  ESPLEY_GEOM=g1 ESPLEY_FEAT=<parquet> python compare_espley.py train <k>  # k 0-4 Espley targets, 5-6 role d1/d2
  python compare_espley.py espley_role theirs|ours    # Espley 46 AM1 feat. on role d1/d2: their protocol / our pipeline
  python compare_espley.py plot                       # intersection scoring, tables, figures -> results_rev4/
Scratch: $R4_SCRATCH/compare/{espley,g1}/. prep, train and espley_role skip existing outputs; every write is atomic.
The Espley outputs in compare/espley/ do not depend on our geometry.
"""
import ast
import json
import numbers
import os
import pickle
import sys
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, GRIDS, _make_pipe, sha256  # noqa: E402

ESP = Path(os.environ.get("ESPLEY_REPO_DATA", "/gpfs/tmp_cpu2/yeseo1ee/espley_compare"))
ESP_DS = ESP / "feature_selection/_f_selection/tt/manual_tt_solvent.pkl"
ESP_RES = ESP / "machine_learning/tt/solvent/ml_results.pkl"
ESP_HPS = ESP / "hyperparameter_tuning/tt/solvent/hps.pkl"
OUT = Path(os.environ.get("R4_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev4")) / "compare"
GEOM = os.environ.get("ESPLEY_GEOM") or None
FEAT = Path(os.environ["ESPLEY_FEAT"]) if os.environ.get("ESPLEY_FEAT") else None
LABELS = Path(os.environ.get("ESPLEY_LABELS") or HERE.parent.parent / "labels_all.json")
RES = HERE / "results_rev4"
GEOMS = ("g1",)                                   # our geometry; the parquet `geom` tag is the geometry name
N_SCORED = 3327                                   # rev 4 scored rows (labelled AND G1-usable), asserted in plot()
SEEDS = [22, 23, 14, 1, 2]
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))
# Espley target -> (display name, our label key, xTB pre-ML feature)
# Espley's distortion_energy_1/2 follow their own reactant index, which is the dipole / dipolarophile in ~69 % of rows
# and swapped in ~31 % (prep() measures this): index-based, appendix only. The main comparison uses ROLE_TARGETS.
TARGETS = {
    "distortion_energy_1_dft": ("Distortion 1 (index)", "d1_kcal", "xtb_dist_dipole_kcal"),
    "distortion_energy_2_dft": ("Distortion 2 (index)", "d2_kcal", "xtb_dist_dipolarophile_kcal"),
    "interaction_energies_dft": ("Interaction", "eint_spe_kcal", "xtb_interaction_kcal"),
    "e_barrier_dft": ("ΔE‡", "barrier_kcal", "xtb_e_barrier_kcal"),
    "q_barrier_dft": ("ΔG‡", None, "xtb_e_barrier_kcal"),
}
ESP_MODELS = {"ridge": "Ridge", "krr": "KRR", "svr": "SVR", "2_st_nn": "2-layer NN", "4_st_nn": "4-layer NN"}
# Espley stores the interaction energy with the opposite sign (positive = stabilising)
SIGN = {"interaction_energies_dft": -1.0}
# Espley's d1/d2 numbers re-assigned to dipole / dipolarophile by the swap vector of prep()
ROLE_TARGETS = {"dipole_distortion_role": ("Dipole distortion (role-consistent)", 0),
                "dipolarophile_distortion_role": ("Dipolarophile distortion (role-consistent)", 1)}
TKEYS = list(TARGETS) + list(ROLE_TARGETS)
INDEX_D = ["distortion_energy_1_dft", "distortion_energy_2_dft"]
MAIN_TARGETS = list(ROLE_TARGETS) + ["interaction_energies_dft", "e_barrier_dft", "q_barrier_dft"]
LABEL = {**{t: v[0] for t, v in TARGETS.items()}, **{t: v[0] for t, v in ROLE_TARGETS.items()}}
SHORT = {"dipole_distortion_role": "Dipole\ndistortion *", "dipolarophile_distortion_role": "Dipolarophile\ndistortion *",
         "interaction_energies_dft": "Interaction", "e_barrier_dft": "ΔE‡", "q_barrier_dft": "ΔG‡"}
# pre-ML xTB estimate per target (xTB gives an electronic barrier only: none for ΔG‡)
PRE_X = {**{t: v[2] for t, v in TARGETS.items()}, "q_barrier_dft": None,
         "dipole_distortion_role": "xtb_dist_dipole_kcal", "dipolarophile_distortion_role": "xtb_dist_dipolarophile_kcal"}
ARMS = {"ESPLEY46": FEATURE_SETS["ESPLEY46"], "ESPLEY73": FEATURE_SETS["ESPLEY73"]}
MAIN_ARM, MAIN_MODEL, MAIN_ESP_MODEL = "ESPLEY46", "KRR", "SVR"     # pre-registered (REV4 4-1): no best-of selection
OUR_MODELS = [m.replace("_rbf", "") for m in GRIDS]

ESP_SIDE = "Espley (AM1 geometry, 46 AM1 feat.)"
OUR_SIDE = {"g1": "Ours G1 (xTB geometry)"}
PROTO = {"stored": "stored test predictions", "theirs": "their protocol (retrained)", "ours": "our pipeline"}
KEY = ["side", "protocol", "arm", "model", "target"]

# Espley's own protocol, from their code (github.com/the-grayson-group/distortion-interaction_ML, branch master).
#  hyperparameter_tuning/hyp_tuning.py — not among the downloaded repo files: prep() fetches it to HYP_SRC and
#  espley_role theirs parses it (check_their_grid) against THEIR_TUNE; 547 lines, sha256 HYP_SHA256 on 2026-09-30:
#    :100-106   tuning X keeps sum_distortion_energies_am1 (47 columns; ml_analysis.py:139 drops it -> 46);
#               which of the two reproduces hps.pkl is tested, see TUNE_DROPS
#    :126-127   split with random_state=23 hard-coded: tuned once, on the seed-23 training split
#    :172-176, :184-185   StandardScaler fit on X_train only; y raw ('just_X', used at :355)
#    :303-308   models: 'krr' KernelRidge(kernel='poly') over alpha, gamma; 'svr' SVR(kernel='rbf') over
#               gamma, epsilon, C, coef0, degree
#    :309-320   hp_values, their tuning grid (ESI Table S3): alpha :309, gamma per model :312 (picked at :343-344),
#               epsilon :313, C :314, coef0 :315, degree :316
#    :354-355   GridSearchCV(model, hp_dict, cv=5, scoring='neg_mean_absolute_error').fit(X_train, y_train)
#    :363       hps.pkl[<model>_<target>] = {'b_hps': best_params_, 'train_mae': |best_score_|}
#  machine_learning/ml_analysis.py ($ESPLEY_REPO_DATA):
#    :139-145, :194-199   46 features: tag drop incl. sum_distortion_energies, then every '_dft' column
#    :168-169   per-seed split; :215-218 StandardScaler fit on that seed's X_train, y raw
#    :355-360, :375   KernelRidge(kernel='rbf') / SVR(kernel='rbf').set_params(**hps[...]['b_hps']) — KRR is tuned
#                     with the poly kernel and run with rbf; kept as they did
#    :408, :464   fit on the whole training split, predict X_test; :479-487 SE = std(|err|)/sqrt(n)
HYP_URL = ("https://raw.githubusercontent.com/the-grayson-group/distortion-interaction_ML/master/"
           "hyperparameter_tuning/hyp_tuning.py")
HYP_SHA256 = "d16407304eb9152a7753eeea54824593b52fce8054f0e958fe2213b0656af16d"
HYP_SRC = OUT / "espley" / "hyp_tuning.py"
TUNE_SEED = 23
THEIR_TUNE = {
    "KRR": (KernelRidge(kernel="poly"), {"alpha": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                         "gamma": [None, 0.1, 0.5, 0.9]}),
    "SVR": (SVR(kernel="rbf"), {"gamma": ["auto", "scale"], "epsilon": [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
                                "C": [1, 30, 50], "coef0": [0, 1], "degree": [1, 2, 3]}),
}
THEIR_ML = {"KRR": lambda: KernelRidge(kernel="rbf"), "SVR": lambda: SVR(kernel="rbf")}
DROP_ML = ("contributions", "reactant", "structure", "path", "reaction_number", "sum_distortion_energies")
DROP_TUNE = DROP_ML[:-1]
# tuning columns, tried in this order on their index d1/d2 before any role target is fitted: the first whose re-run
# gives the hps.pkl params of SVR and KRR on both is used for the role targets; none -> TUNE_DROPS[0], flagged
TUNE_DROPS = (DROP_TUNE, DROP_ML)


def espley_split(n, seed):
    """Their _perform_train_test_split: X_test is the FIRST half of the 20 % hold-out."""
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed)
    te, _va = train_test_split(rest, test_size=0.5, random_state=seed)
    return tr, te


def espley_X(esp, swap=None, drop=DROP_ML):
    """Espley's feature columns: tag drop (ml_analysis.py:139-145; hyp_tuning.py:100-106 keeps sum_distortion_energies)
    then every '_dft' column (ml_analysis.py:194-199). swap (bool per row): the two AM1 distortions re-assigned to
    dipole / dipolarophile like the role targets."""
    X = esp[[c for c in esp.columns if not any(tag in c for tag in drop) and "_dft" not in c]].copy()
    if swap is not None:
        a1, a2 = X["distortion_energy_1_am1"].to_numpy().copy(), X["distortion_energy_2_am1"].to_numpy().copy()
        X["distortion_energy_1_am1"], X["distortion_energy_2_am1"] = np.where(swap, a2, a1), np.where(swap, a1, a2)
    return X


def target_values(t, esp, P):
    if t in TARGETS:
        return esp[t].values.astype(float)
    t1, t2 = esp["distortion_energy_1_dft"].values, esp["distortion_energy_2_dft"].values
    sw = P["swap"]
    y = np.where(sw == 1, t2, t1) if ROLE_TARGETS[t][1] == 0 else np.where(sw == 1, t1, t2)
    return np.where(np.isnan(sw), np.nan, y)


def side_name(g, arm):
    return f"{OUR_SIDE[g]}, {arm[6:]} feat."


def load_esp():
    return pd.read_pickle(ESP_DS).reset_index(drop=True)


def write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _native(x):
    return x.item() if hasattr(x, "item") else str(x)


def need_geom():
    if GEOM not in GEOMS or FEAT is None:
        sys.exit(f"set ESPLEY_GEOM (one of {GEOMS}) and ESPLEY_FEAT; got {GEOM!r}, {FEAT}")
    return GEOM


def load_feat():
    df = pd.read_parquet(FEAT)
    tags = sorted(set(df["geom"].astype(str))) if "geom" in df else ["<no geom column>"]
    if tags != [GEOM]:
        sys.exit(f"GATE: ESPLEY_GEOM={GEOM} expects geom tag {GEOM!r}, {FEAT} has {tags}")
    df = df.assign(rxn_id=df["rxn_id"].astype(int))
    if df["rxn_id"].duplicated().any():
        sys.exit(f"GATE: duplicate rxn_id in {FEAT}")
    return df.set_index("rxn_id")


def load_common():
    f = OUT / "espley" / "prep.pkl"
    if not f.exists():
        sys.exit(f"{f} missing: run `compare_espley.py prep` first")
    return pickle.loads(f.read_bytes())


def load_prep(g, feat=None):
    """OUT/<g>/prep.pkl; exits unless it was made from feat (default: the file it records) as that file is now."""
    f = OUT / g / "prep.pkl"
    if not f.exists():
        sys.exit(f"{f} missing: run `ESPLEY_GEOM={g} ESPLEY_FEAT=... compare_espley.py prep` first")
    P = pickle.loads(f.read_bytes())
    feat = Path(P["feat"]) if feat is None else feat
    if P["feat"] != str(feat.resolve()) or not feat.exists() or sha256(feat) != P["feat_sha256"]:
        sys.exit(f"{f} was made from {P['feat']} (sha256 {P['feat_sha256'][:12]}), not from {feat} as it is now: "
                 f"move {f.parent} away")
    return P


def same_feat(df, P, what):
    """Our outputs carry the sha256 of the feature parquet they were trained on; it must be the one of prep.pkl."""
    got = sorted(set(df["feat_sha256"].astype(str))) if "feat_sha256" in df else ["<none>"]
    if got != [P["feat_sha256"]]:
        sys.exit(f"{what} was made from features sha256 {got}, {P['geom']}/prep.pkl from {P['feat_sha256']}: "
                 "move it away")


def fetch_hyp():
    """Their hyp_tuning.py (not among the downloaded repo files) -> HYP_SRC; None when offline."""
    if not HYP_SRC.exists():
        try:
            data = urllib.request.urlopen(HYP_URL, timeout=60).read()
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: cannot fetch {HYP_URL} ({type(e).__name__}: {e}); grid not checked against the source")
            return None
        write_atomic(HYP_SRC, lambda f: f.write_bytes(data))
    return HYP_SRC


def check_their_grid(path):
    """THEIR_TUNE vs `models` / `hp_values` parsed (ast, nothing executed) from their hyp_tuning.py."""
    if path is None:
        return dict(verified=None, reason=f"{HYP_URL} not available")
    try:
        node = {a.targets[0].id: a for a in ast.walk(ast.parse(path.read_text()))
                if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name)
                and a.targets[0].id in ("models", "hp_values") and isinstance(a.value, ast.Dict)}
        hp = ast.literal_eval(node["hp_values"].value)
        found = {}
        for k, v in zip(node["models"].value.keys, node["models"].value.values):
            m = ast.literal_eval(k)
            spec = dict(zip((ast.literal_eval(x) for x in v.keys), v.values))
            found[m] = dict(est=spec["model"].func.id,
                            kw={a.arg: ast.literal_eval(a.value) for a in spec["model"].keywords},
                            grid={h: hp[h][m] if isinstance(hp[h], dict) else hp[h]
                                  for h in ast.literal_eval(spec["hp"])})
    except Exception as e:  # noqa: BLE001
        return dict(verified=False, file=str(path), reason=f"parse failed: {type(e).__name__}: {e}")
    want = {m.lower(): dict(est=type(est).__name__, kw=dict(kernel=est.kernel), grid=grid)
            for m, (est, grid) in THEIR_TUNE.items()}
    got = {m: found.get(m) for m in want}
    return dict(verified=got == want, file=str(path), sha256=sha256(path), sha256_2026_09_30=HYP_SHA256,
                lines=dict(models=node["models"].lineno, hp_values=node["hp_values"].lineno),
                parsed=got, transcribed=want)


def _same(a, b):
    num = lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool)  # noqa: E731
    return bool(np.isclose(a, b)) if num(a) and num(b) else a == b


def hps_in_grid():
    """Every KRR / SVR b_hps in their hps.pkl must be a point of THEIR_TUNE (guards the transcription offline)."""
    hps = pd.read_pickle(ESP_HPS)
    off = {}
    for mt, v in hps.items():
        m = mt.split("_", 1)[0].upper()
        if m not in THEIR_TUNE:
            continue
        grid, b = THEIR_TUNE[m][1], v["b_hps"]
        if set(b) != set(grid) or not all(any(_same(x, gv) for gv in grid[k]) for k, x in b.items()):
            off[mt] = b
    return hps, off


def footnote(fig, notes, width, fontsize):
    """Wrapped italic notes under the axes (tight_layout keeps the axes above them)."""
    lines = [ln for note in notes for ln in textwrap.wrap(note, width)]
    for i, line in enumerate(lines):
        fig.text(0.5, 0.012 + 0.022 * (len(lines) - 1 - i), line, ha="center", va="bottom", fontsize=fontsize,
                 style="italic")
    fig.tight_layout(rect=[0, 0.03 + 0.022 * len(lines), 1, 1])


def frame(rows, ids, y, yhat, **kw):
    return pd.DataFrame(dict(row=rows, rxn_id=ids[rows], y=y, yhat=yhat)).assign(**kw)


def seed_row(e, **kw):
    return dict(**kw, mae=float(e.mean()), se=float(e.std() / np.sqrt(len(e))), n_test=len(e))


def our_pipeline(X, y, ok, n, s, est, grid):
    """Per-seed nested GridSearchCV (GRIDS / _make_pipe, X and y scaled) on the usable training rows of seed s."""
    tr, te = espley_split(n, s)
    tr, te = tr[ok[tr]], te[ok[te]]
    gs = GridSearchCV(_make_pipe(est), grid, cv=KFold(5, shuffle=True, random_state=s),
                      scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
    gs.fit(X[tr], y[tr])
    return te, gs.best_estimator_.predict(X[te]), gs.best_params_


def their_protocol(Xt, Xm, y, n, model):
    """Espley's protocol (see THEIR_TUNE): tune once on the seed-23 training split (X scaled, y raw), then per seed
    refit the ML-time estimator with those params -> (best params, CV MAE, {seed: (test rows, predictions)})."""
    ok = ~np.isnan(y)
    tr23 = espley_split(n, TUNE_SEED)[0]
    tr23 = tr23[ok[tr23]]                              # their X_train order: cv=5 is an unshuffled KFold on it
    est, grid = THEIR_TUNE[model]
    gs = GridSearchCV(est, grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=N_JOBS)   # error_score as theirs
    gs.fit(StandardScaler().fit_transform(Xt[tr23]), y[tr23])
    out = {}
    for s in SEEDS:
        tr, te = espley_split(n, s)
        tr, te = tr[ok[tr]], te[ok[te]]
        sc = StandardScaler().fit(Xm[tr])
        m = THEIR_ML[model]().set_params(**gs.best_params_).fit(sc.transform(Xm[tr]), y[tr])
        out[s] = (te, m.predict(sc.transform(Xm[te])))
    return gs.best_params_, float(-gs.best_score_), out


def prep_common():
    """Geometry-independent checks -> OUT/espley/{prep.pkl, espley_stored_preds.parquet}."""
    f, fp = OUT / "espley" / "prep.pkl", OUT / "espley" / "espley_stored_preds.parquet"
    fetch_hyp()
    if f.exists() and fp.exists():
        print("exists:", f)
        return pickle.loads(f.read_bytes())
    esp = load_esp()
    res = pd.read_pickle(ESP_RES)
    lab = {int(r["rxn_id"]): r for r in json.load(open(LABELS))}
    ids = esp["reaction_number"].astype(int).values
    n = len(esp)
    print(f"Espley ds3 rows: {n}, unique reaction_number: {len(set(ids))}")
    assert len(set(ids)) == n, "duplicate reaction_number in Espley ds3"
    x46, x47 = list(espley_X(esp).columns), list(espley_X(esp, drop=DROP_TUNE).columns)
    assert len(x46) == 46, (len(x46), x46)
    assert [c for c in x47 if c not in x46] == ["sum_distortion_energies_am1"], x47

    # 1. id mapping + label agreement (their Gaussian DFT vs our ORCA labels)
    have_lab = np.array([i in lab and lab[i]["status"] == "ok" for i in ids])
    print(f"in our labels (ok): {have_lab.sum()}/{n}; without: {[int(i) for i in ids[~have_lab]]}")
    agree = {}
    for t, (_, key, _) in TARGETS.items():
        if key is None:
            continue
        m = have_lab
        ours = np.array([lab[i][key] if lab.get(i) else np.nan for i in ids])[m]
        d = ours - SIGN.get(t, 1.0) * esp[t].values[m]
        agree[t] = dict(n=int(m.sum()), mae=float(np.nanmean(np.abs(d))), bias=float(np.nanmean(d)),
                        r=float(np.corrcoef(ours, esp[t].values[m])[0, 1]),
                        n_gt1=int((np.abs(d) > 1).sum()))
        print(f"  label agreement {t:26s} MAE {agree[t]['mae']:.3f}  bias {agree[t]['bias']:+.3f}  "
              f"r {agree[t]['r']:.4f}  |d|>1: {agree[t]['n_gt1']}")

    # 1b. d1/d2 index vs role: our frag1 is always the dipole; Espley's _1/_2 is a reactant index
    t1, t2 = esp["distortion_energy_1_dft"].values, esp["distortion_energy_2_dft"].values
    d1o = np.array([lab[i]["d1_kcal"] if have_lab[k] else np.nan for k, i in enumerate(ids)])
    d2o = np.array([lab[i]["d2_kcal"] if have_lab[k] else np.nan for k, i in enumerate(ids)])
    sw = np.abs(d1o - t2) + np.abs(d2o - t1) < np.abs(d1o - t1) + np.abs(d2o - t2)
    swap = np.where(have_lab, sw.astype(float), np.nan)
    dip, dpo = np.where(sw, t2, t1), np.where(sw, t1, t2)
    role = dict(n_swapped=int(np.nansum(swap)), n=int(have_lab.sum()),
                mae_d1_raw=float(np.nanmean(np.abs(d1o - t1))), mae_d2_raw=float(np.nanmean(np.abs(d2o - t2))),
                mae_d1_role=float(np.nanmean(np.abs(d1o - dip))), mae_d2_role=float(np.nanmean(np.abs(d2o - dpo))),
                n_ambiguous=int(np.sum(have_lab & (np.abs(t1 - t2) < 0.5))))
    # Espley's own AM1 distortions follow the same index as their DFT columns (hence the role re-assignment)
    a1, a2 = esp["distortion_energy_1_am1"].values, esp["distortion_energy_2_am1"].values
    role["am1_same_index_r"] = float(np.corrcoef(a1, t1)[0, 1])
    role["am1_swapped_index_r"] = float(np.corrcoef(a1, t2)[0, 1])
    m = have_lab
    role["am1_role_r_dipole"] = float(np.corrcoef(np.where(sw, a2, a1)[m], dip[m])[0, 1])
    role["am1_role_r_dipolarophile"] = float(np.corrcoef(np.where(sw, a1, a2)[m], dpo[m])[0, 1])
    print("  d1/d2 index vs role:", json.dumps(role, indent=1))

    # 2. split replication for every stored model/target: our split reproduces their stored test targets;
    # 3. their stored per-row test predictions (y in te order), all models
    split_ok, stored = {}, []
    for _, row in res.iterrows():
        mt = row.model_target
        model = next(k for k in sorted(ESP_MODELS, key=len, reverse=True) if mt.startswith(k + "_"))
        t = mt[len(model) + 1:]
        assert t in TARGETS, mt
        ok = []
        for s_i, s in enumerate(row.random_state):
            y = np.asarray(row.y_test_true[s_i], float).ravel()
            p = np.asarray(row.y_test_pred_values[s_i], float).ravel()
            _, te = espley_split(n, int(s))
            ok.append(bool(len(y) == len(te) == len(p) and np.allclose(esp[t].values[te], y, atol=1e-6)))
            if ok[-1]:
                stored.append(frame(te, ids, y, p, seed=int(s), target=t, side=ESP_SIDE, geom="am1", arm="AM1-46",
                                    protocol=PROTO["stored"], model=ESP_MODELS[model]))
        split_ok[mt] = ok
        print(f"  split replication {mt:36s} seeds {list(row.random_state)} -> {ok}")
    assert all(all(v) for v in split_ok.values()), "split replication failed"

    C = dict(ids=ids, n=n, have_lab=have_lab, agree=agree, split_ok=split_ok, swap=swap, role=role,
             esp46=x46, esp_tune=x47, labels=str(LABELS.resolve()), labels_sha256=sha256(LABELS))
    write_atomic(fp, lambda p: pd.concat(stored, ignore_index=True).to_parquet(p, index=False))
    write_atomic(f, lambda p: p.write_bytes(pickle.dumps(C)))
    print("wrote", f, fp)
    return C


def prep():
    C = prep_common()
    g = need_geom()
    f = OUT / g / "prep.pkl"
    if f.exists():
        load_prep(g, FEAT)
        print("exists:", f)
        return
    esp, feat = load_esp(), load_feat()
    ids = C["ids"]
    cols = sorted(set(sum(ARMS.values(), [])))
    lacking = [c for c in cols + [c for c in PRE_X.values() if c] + ["xtb_status"] if c not in feat]
    if lacking:
        sys.exit(f"GATE: columns missing in {FEAT}: {lacking}")
    F = feat.reindex(ids)                                   # rows in Espley order; NaN where we have no row
    st = F["xtb_status"].fillna("absent").astype(str).to_numpy()
    nan = F[cols].isna().any(axis=1).to_numpy()
    have_feat = (st == "ok") & ~nan
    status = np.where(st != "ok", st, np.where(nan, "ok_but_nan_feature", "ok"))
    print(f"[{g}] {FEAT}: usable {have_feat.sum()}/{len(ids)}; not usable: "
          f"{pd.Series(status[~have_feat]).value_counts().to_dict()}")
    xpre = {t: F[c].to_numpy(float) for t, c in PRE_X.items() if c}
    pre = {}
    for t, x in xpre.items():
        y = SIGN.get(t, 1.0) * target_values(t, esp, C)
        m = have_feat & ~np.isnan(y)
        pre[t] = float(np.mean(np.abs(x[m] - y[m])))
    # our features are role-based: on the swapped rows the xTB dipole strain tracks their _2, not their _1
    t1, t2 = esp["distortion_energy_1_dft"].values, esp["distortion_energy_2_dft"].values
    xd, ms = F["xtb_dist_dipole_kcal"].to_numpy(float), C["have_lab"] & have_feat & (C["swap"] == 1)
    role_xtb = dict(swapped_rows_corr_xtbdipole_vs_t1=float(np.corrcoef(xd[ms], t1[ms])[0, 1]),
                    swapped_rows_corr_xtbdipole_vs_t2=float(np.corrcoef(xd[ms], t2[ms])[0, 1]))
    P = dict(geom=g, feat=str(FEAT.resolve()), feat_sha256=sha256(FEAT), ids=ids, have_feat=have_feat,
             status=status, xpre=xpre, pre_xtb=pre, role_xtb=role_xtb)
    write_atomic(f, lambda p: p.write_bytes(pickle.dumps(P)))
    print(f"[{g}] pre-ML xTB MAE (own usable rows):", {k: round(v, 2) for k, v in pre.items()}, role_xtb)
    print("wrote", f)


def train(k):
    g = need_geom()
    t = TKEYS[k]
    out, out_p = OUT / g / f"ours_per_seed_{k}.csv", OUT / g / f"ours_preds_{k}.parquet"
    P = load_prep(g, FEAT)
    if out.exists() and out_p.exists():
        same_feat(pd.read_parquet(out_p), P, out_p)
        print("exists:", out); return
    C = load_common()
    esp, feat = load_esp(), load_feat()
    ids, n = C["ids"], C["n"]
    y_all = target_values(t, esp, C)
    # train (and score) on the rows usable by every side = labelled AND G1-usable (the rev 4 rows, plot() asserts)
    assert np.array_equal(P["ids"], ids)
    ok = C["have_lab"] & P["have_feat"] & ~np.isnan(y_all)
    rows, preds = [], []
    for arm, cols in ARMS.items():
        X_all = feat.reindex(ids)[cols].to_numpy(float)      # rows in Espley order; NaN where we lack features
        side = side_name(g, arm)
        for name, (est, grid) in GRIDS.items():
            model = name.replace("_rbf", "")
            for s in SEEDS:
                te, p, best = our_pipeline(X_all, y_all, ok, n, s, est, grid)
                e = np.abs(p - y_all[te])
                kw = dict(side=side, protocol=PROTO["ours"], geom=g, arm=arm, model=model, target=t, seed=s,
                          feat_sha256=P["feat_sha256"])
                rows.append(seed_row(e, **kw, n_test_espley=len(espley_split(n, s)[1]), best_params=json.dumps(best)))
                preds.append(frame(te, ids, y_all[te], p, **kw))
                print(f"{g} {t} {arm} {name} seed {s}: MAE {e.mean():.3f} (n={len(e)})", flush=True)
    write_atomic(out_p, lambda f: pd.concat(preds, ignore_index=True).to_parquet(f, index=False))
    write_atomic(out, lambda f: pd.DataFrame(rows).to_csv(f, index=False))      # CSV last: marks a finished k


def espley_role(mode):
    """Espley's 46 AM1 features (AM1 distortions re-assigned by the swap vector) on the role targets.
    theirs: their protocol, SVR + KRR; first re-run on their index targets (unswapped) against hps.pkl and their
            stored test predictions, which fixes the tuning columns (TUNE_DROPS; none -> TUNE_DROPS[0], flagged),
            then on the role targets -> role_theirs.json. ours: our pipeline, all GRIDS models."""
    if mode not in ("theirs", "ours"):
        sys.exit(f"espley_role theirs|ours, got {mode!r}")
    d = OUT / "espley"
    out, out_p, out_j = d / f"role_{mode}_per_seed.csv", d / f"role_{mode}_preds.parquet", d / "role_theirs.json"
    if out.exists() and out_p.exists():
        print("exists:", out); return
    C, esp = load_common(), load_esp()
    ids, n = C["ids"], C["n"]
    sw = C["swap"] == 1                           # no label -> NaN -> not swapped; its role target is NaN anyway
    Xr = espley_X(esp, sw).to_numpy(float)
    rows, preds = [], []

    def record(t, model, proto, s, te, p, y, best):
        e = np.abs(p - y[te])
        kw = dict(side=ESP_SIDE, protocol=proto, geom="am1", arm="AM1-46", model=model, target=t, seed=s)
        rows.append(seed_row(e, **kw, n_test_espley=len(espley_split(n, s)[1]), best_params=json.dumps(best)))
        preds.append(frame(te, ids, y[te], p, **kw))
        print(f"espley {mode} {t} {model} seed {s}: MAE {e.mean():.3f} (n={len(e)})", flush=True)

    if mode == "ours":
        for t in ROLE_TARGETS:
            y = target_values(t, esp, C)
            for name, (est, grid) in GRIDS.items():
                for s in SEEDS:
                    te, p, best = our_pipeline(Xr, y, ~np.isnan(y), n, s, est, grid)
                    record(t, name.replace("_rbf", ""), PROTO["ours"], s, te, p, y, best)
    else:
        hps, off = hps_in_grid()
        chk = check_their_grid(fetch_hyp())
        print("grid check:", {k: chk.get(k) for k in ("verified", "lines", "sha256", "reason")},
              "| hps.pkl b_hps off THEIR_TUNE:", off or "none", flush=True)
        if chk["verified"] is False or off:
            sys.exit("STOP: THEIR_TUNE does not match their hyp_tuning.py / hps.pkl — fix the transcription")
        res = pd.read_pickle(ESP_RES).set_index("model_target")
        Xm_idx = espley_X(esp).to_numpy(float)
        rep = dict(grid_check=chk, hps_off_grid=off, ml_features=C["esp46"], attempts={}, replicated=False,
                   tuning_features=None, tuned={}, replication={})
        # 1. tuning columns: replication on their index targets, same tuned params as hps.pkl (+ test predictions vs
        #    ml_results.pkl, reported), TUNE_DROPS in order; no role target is fitted before this is fixed
        for drop in TUNE_DROPS:
            cols = list(espley_X(esp, drop=drop).columns)
            Xt_idx = espley_X(esp, None, drop).to_numpy(float)
            fits, att = {}, {}
            for t in INDEX_D:
                y = esp[t].values.astype(float)
                for m in THEIR_TUNE:
                    best, cv, per = their_protocol(Xt_idx, Xm_idx, y, n, m)
                    fits[(t, m)] = (best, cv, per, y)
                    mt, ref = f"{m.lower()}_{t}", hps[f"{m.lower()}_{t}"]
                    row, rp = res.loc[mt], {}
                    for s_i, s in enumerate(row.random_state):
                        te, p = per[int(s)]
                        ps = np.asarray(row.y_test_pred_values[s_i], float).ravel()
                        rp[int(s)] = dict(max_abs_pred_diff=float(np.max(np.abs(p - ps))),
                                          mae_rerun=float(np.mean(np.abs(p - y[te]))),
                                          mae_stored=float(np.mean(np.abs(ps - y[te]))))
                    eq = set(best) == set(ref["b_hps"]) and all(_same(v, ref["b_hps"][k]) for k, v in best.items())
                    att.setdefault(t, {})[m] = dict(
                        params_rerun=best, params_hps_pkl=ref["b_hps"], params_equal=eq,
                        cv_mae_rerun=cv, cv_mae_hps_pkl=float(ref["train_mae"]), per_seed=rp)
                    print(f"replication [{len(cols)} tuning feat.] {mt}: params equal {eq} ({best} vs "
                          f"{ref['b_hps']}); max |Δpred| {max(v['max_abs_pred_diff'] for v in rp.values()):.2e}",
                          flush=True)
            ok = all(v["params_equal"] for a in att.values() for v in a.values())
            rep["attempts"][str(len(cols))] = dict(features=cols, params_equal=ok, replication=att)
            if ok:
                break
        else:
            # PREREG_REV4 (fixed before any result): not a STOP — tune on the columns of their hyp_tuning.py (47,
            # TUNE_DROPS[0]) and publish with espley_protocol_replicated = False
            drop = TUNE_DROPS[0]
            cols = list(espley_X(esp, drop=drop).columns)
            att = rep["attempts"][str(len(cols))]["replication"]
            Xt_idx = espley_X(esp, None, drop).to_numpy(float)
            fits = {}
            for t in INDEX_D:
                y = esp[t].values.astype(float)
                for m in THEIR_TUNE:
                    fits[(t, m)] = (*their_protocol(Xt_idx, Xm_idx, y, n, m), y)
            print(f"NOT REPLICATED with any of {sorted(rep['attempts'])} tuning column sets: using {len(cols)} "
                  f"(their hyp_tuning.py), flagged espley_protocol_replicated=False", flush=True)
        rep.update(replicated=bool(ok), tuning_features=cols, replication=att)
        for (t, m), (best, cv, per, y) in fits.items():
            rep["tuned"].setdefault(t, {})[m] = dict(best_params=best, cv_mae=cv)
            for s, (te, p) in per.items():
                record(t, m, PROTO["theirs"], s, te, p, y, best)
        # 2. role targets, tuned on the same columns (AM1 distortions re-assigned; their sum is swap-invariant)
        Xt_role = espley_X(esp, sw, drop).to_numpy(float)
        for t in ROLE_TARGETS:
            y = target_values(t, esp, C)
            for m in THEIR_TUNE:
                best, cv, per = their_protocol(Xt_role, Xr, y, n, m)
                rep["tuned"].setdefault(t, {})[m] = dict(best_params=best, cv_mae=cv)
                for s, (te, p) in per.items():
                    record(t, m, PROTO["theirs"], s, te, p, y, best)
    write_atomic(out_p, lambda f: pd.concat(preds, ignore_index=True).to_parquet(f, index=False))
    if mode == "theirs":
        write_atomic(out_j, lambda f: f.write_text(json.dumps(rep, indent=1, default=_native)))
    write_atomic(out, lambda f: pd.DataFrame(rows).to_csv(f, index=False))      # CSV last: marks completion


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = load_common()
    P = {g: load_prep(g) for g in GEOMS}       # each checked against its feature parquet as that file is now
    ids, n = C["ids"], C["n"]
    for g in GEOMS:
        assert np.array_equal(P[g]["ids"], ids), g
    # test rows: the intersection of the rows usable by every side (Espley: all rows; role targets need our label)
    usable = C["have_lab"] & np.logical_and.reduce([P[g]["have_feat"] for g in GEOMS])
    dropped = []
    for i in np.flatnonzero(~usable):
        why = [] if C["have_lab"][i] else ["no_label"]
        why += [f"{g}:{P[g]['status'][i]}" for g in GEOMS if not P[g]["have_feat"][i]]
        dropped.append(dict(rxn_id=int(ids[i]), reasons=why))
    print(f"scored rows = intersection: {usable.sum()}/{n}; dropped {len(dropped)}:",
          [r["rxn_id"] for r in dropped][:50])
    if int(usable.sum()) != N_SCORED:
        sys.exit(f"GATE: {int(usable.sum())} scored rows (labelled AND G1-usable), expected the rev 4 {N_SCORED}")

    d = OUT / "espley"
    ours = {OUT / g / f"ours_preds_{k}.parquet": g for g in GEOMS for k in range(len(TKEYS))}
    files = ([d / "espley_stored_preds.parquet", d / "role_theirs_preds.parquet", d / "role_ours_preds.parquet"]
             + list(ours))
    miss = [str(f) for f in files + [d / "role_theirs.json"] if not f.exists()]
    if miss:
        sys.exit(f"missing outputs (run prep / train / espley_role first): {miss}")
    frames = {f: pd.read_parquet(f) for f in files}
    for f, g in ours.items():
        same_feat(frames[f], P[g], f)
    rep = json.loads((d / "role_theirs.json").read_text())
    if not rep.get("replicated"):
        print(f"NOTE: {d / 'role_theirs.json'}: their protocol re-run does not reproduce hps.pkl — the role numbers "
              "of their protocol are published with espley_protocol_replicated=False", flush=True)
    preds = pd.concat(frames.values(), ignore_index=True)
    preds = preds[usable[preds["row"].to_numpy()]].copy()

    # gate: every side is scored on exactly the intersection test rows of each seed, against the same y
    n_te = {s: len(espley_split(n, s)[1]) for s in SEEDS}
    want = {s: np.sort(espley_split(n, s)[1][usable[espley_split(n, s)[1]]]) for s in SEEDS}
    for key, sub in preds.groupby(KEY + ["seed"]):
        if not np.array_equal(np.sort(sub["row"].to_numpy()), want[key[-1]]):
            sys.exit(f"GATE: test rows of {key} differ from the intersection ({len(sub)} vs {len(want[key[-1]])})")
    yy = preds.groupby(["target", "seed", "row"])["y"].agg(["min", "max"])
    ydev = float((yy["max"] - yy["min"]).max())
    if ydev > 1e-6:
        sys.exit(f"GATE: sides disagree on the target values (max |Δy| {ydev:.2e})")

    preds["ae"] = (preds["yhat"] - preds["y"]).abs()
    per = (preds.groupby(KEY + ["seed"])["ae"]
           .agg(mae="mean", se=lambda a: float(np.std(a.to_numpy()) / np.sqrt(len(a))), n_test="size")
           .reset_index())
    per["n_test_espley"] = per["seed"].map(n_te)
    per["geom"] = per["side"].map(preds.drop_duplicates("side").set_index("side")["geom"])
    agg = (per.groupby(KEY).agg(mae=("mae", "mean"), se=("se", "mean"), sd=("mae", "std"),
                                n_seeds=("seed", "nunique"), n_test=("n_test", "mean")).reset_index())
    G = agg.set_index(KEY).sort_index()
    PS = per.set_index(KEY + ["seed"])["mae"].sort_index()

    def ps(k):
        return PS.loc[k].reindex(SEEDS)

    def esp_key(t):                  # Espley SVR: stored predictions; role targets retrained with their protocol
        return ESP_SIDE, PROTO["stored"] if t in TARGETS else PROTO["theirs"], "AM1-46", MAIN_ESP_MODEL, t

    def our_key(gg, t):
        return side_name(gg, MAIN_ARM), PROTO["ours"], MAIN_ARM, MAIN_MODEL, t

    side_order = [ESP_SIDE] + [side_name(gg, a) for gg in GEOMS for a in ARMS]

    def order(df):
        return (df.assign(_t=df.target.map(TKEYS.index), _s=df.side.map(side_order.index),
                          _p=df.protocol.map(list(PROTO.values()).index))
                .sort_values(["_t", "_s", "_p", "model"]).drop(columns=["_t", "_s", "_p"]))

    esp = load_esp()
    pre = {}
    for gg in GEOMS:
        pre[gg] = {}
        for t, x in P[gg]["xpre"].items():
            y = SIGN.get(t, 1.0) * target_values(t, esp, C)
            pre[gg][t] = float(np.mean(np.abs(x[usable] - y[usable])))

    # MAIN: G1 · ESPLEY46 · KRR vs Espley SVR (both fixed a priori)
    main = []
    for t in MAIN_TARGETS:
        a, b = ps(esp_key(t)), ps(our_key("g1", t))
        e, o1 = G.loc[esp_key(t)], G.loc[our_key("g1", t)]
        main.append(dict(target=LABEL[t], target_key=t,
                         espley=f"{ESP_SIDE} · {MAIN_ESP_MODEL} · {esp_key(t)[1]}",
                         espley_mae=e.mae, espley_se=e.se,
                         # retrained rows: their protocol re-run reproduces hps.pkl on index d1/d2 (plot gates on it)
                         espley_protocol_replicated=rep["replicated"] if t in ROLE_TARGETS else None,
                         g1=f"{side_name('g1', MAIN_ARM)} · {MAIN_MODEL}", g1_mae=o1.mae, g1_se=o1.se,
                         paired_diff_espley_minus_g1=float((a - b).mean()), paired_diff_sd=float((a - b).std()),
                         mae_ratio_espley_over_g1=float((a / b).mean()),
                         pre_ml_xtb_g1_mae=pre["g1"].get(t),
                         n_test_per_seed=o1.n_test, n_test_espley_per_seed=float(np.mean(list(n_te.values())))))
    main = pd.DataFrame(main)

    cols = ["mae", "se", "sd", "n_seeds", "n_test"]
    role_rows = ([(ESP_SIDE, PROTO["theirs"], "AM1-46", m) for m in THEIR_TUNE]
                 + [(ESP_SIDE, PROTO["ours"], "AM1-46", m) for m in OUR_MODELS]
                 + [our_key(gg, None)[:4] for gg in GEOMS])
    role = pd.DataFrame([dict(target=LABEL[t], target_key=t, side=s, protocol=p, arm=a, model=m,
                              **G.loc[(s, p, a, m, t)][cols].to_dict())
                         for t in ROLE_TARGETS for s, p, a, m in role_rows])

    best = agg.loc[agg.groupby(["side", "protocol", "target"])["mae"].idxmin()].copy()
    best["n_models_compared"] = [G.loc[(s, p)].xs(t, level="target").shape[0]
                                 for s, p, t in best[["side", "protocol", "target"]].itertuples(index=False)]
    best = order(best.assign(target_label=best.target.map(LABEL)))

    idx_rows = []
    for t in INDEX_D:
        fixed = [(esp_key(t), "fixed: Espley's published best"),
                 ((ESP_SIDE, PROTO["theirs"], "AM1-46", MAIN_ESP_MODEL, t), "their protocol re-run (replication)"),
                 (our_key("g1", t), "fixed: pre-registered G1 · ESPLEY46 · KRR")]
        for k, sel in fixed:
            idx_rows.append(dict(target=LABEL[t], target_key=t, selection=sel, **dict(zip(KEY[:4], k[:4])),
                                 **G.loc[k][cols].to_dict()))
        for r in best[best.target == t].itertuples(index=False):
            idx_rows.append(dict(target=LABEL[t], target_key=t, selection="best of models",
                                 side=r.side, protocol=r.protocol, arm=r.arm, model=r.model,
                                 **{c: getattr(r, c) for c in cols}))
    idx = pd.DataFrame(idx_rows)

    R = C["role"]
    n_use = int(usable.sum())
    for name, df in (("main", main), ("role", role), ("appendix_best", best), ("index_d1d2_appendix", idx),
                     ("per_seed", order(per))):
        write_atomic(RES / f"espley_compare_{name}.csv", lambda f, df=df: df.round(4).to_csv(f, index=False))
    checks = dict(
        inputs=dict(espley_repo=str(ESP), labels=C["labels"], labels_sha256=C["labels_sha256"],
                    features={gg: dict(file=P[gg]["feat"], sha256=P[gg]["feat_sha256"]) for gg in GEOMS}),
        main_rule=dict(ours=f"G1 · {MAIN_ARM} · {MAIN_MODEL}", espley=f"{MAIN_ESP_MODEL}: stored test predictions; "
                       "role targets retrained with their protocol (re-run reproduces hps.pkl on index d1/d2)",
                       selection="fixed a priori, no best-of"),
        label_agreement=C["agree"], split_replicated=C["split_ok"], d1d2_index_vs_role=C["role"],
        d1d2_xtb_dipole_block={gg: P[gg]["role_xtb"] for gg in GEOMS},
        espley_features=dict(ml=C["esp46"], their_protocol_tuning=rep["tuning_features"]),
        intersection=dict(n_espley=n, n_with_our_label=int(C["have_lab"].sum()),
                          n_usable={gg: int(P[gg]["have_feat"].sum()) for gg in GEOMS}, n_scored=n_use,
                          n_dropped=len(dropped), dropped=dropped),
        test_rows_identical_across_sides=True, max_abs_y_diff_across_sides=ydev,
        pre_ml_xtb_mae_on_scored_rows=pre, their_protocol=rep)
    write_atomic(RES / "espley_compare_checks.json",
                 lambda f: f.write_text(json.dumps(checks, indent=1, default=_native)))

    colors = {"esp": "#7f7f7f", "g1": "#08519c"}
    geo = ("Espley: 46 AM1 features on AM1-optimised geometries; ours G1: GFN2-xTB features on GFN2-xTB/ALPB geometries "
           "re-optimised from the DFT TS and references.")
    same_note = (f"Held equal: ds3 reactions, Espley's DFT targets (Gaussian B3LYP-D3(BJ)/def2-TZVP SMD(water)), 80/10/10 "
                 f"splits and seeds, metric; every bar is scored on the same {n_use:,}/{n:,} reactions usable by all "
                 f"sides ({len(dropped)} dropped, listed in espley_compare_checks.json).")

    # main figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.6), gridspec_kw={"width_ratios": [1.6, 1]})
    x, w = np.arange(len(MAIN_TARGETS)), 0.36
    bars = [("espley", f"Espley · AM1 geometry · 46 AM1 feat. · {MAIN_ESP_MODEL}", colors["esp"]),
            ("g1", f"Ours G1 · xTB geometry · {MAIN_ARM} · {MAIN_MODEL}", colors["g1"])]
    for i, (c, lab, col) in enumerate(bars):
        v, e = main[f"{c}_mae"].to_numpy(), main[f"{c}_se"].to_numpy()
        off = (i - (len(bars) - 1) / 2) * w
        ax.bar(x + off, v, w, yerr=e, capsize=3, color=col, edgecolor="black", lw=0.5, label=lab)
        for j, t in enumerate(MAIN_TARGETS):
            star = " *" if c == "espley" and t in ROLE_TARGETS else ""
            ax.text(x[j] + off, v[j] + e[j] + 0.05, f"{v[j]:.2f}{star}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in MAIN_TARGETS])
    ax.set_ylabel("test MAE (kcal/mol) — models fixed a priori\nmean of 5 seeds, error bar = SE (Espley's definition)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Same ds3 reactions · same DFT targets · identical 80/10/10 test rows (seeds 22/23/14/1/2)\n"
                 "· different geometry (ours G1 xTB vs Espley AM1)", fontsize=10.5)
    for j, t in enumerate(MAIN_TARGETS):      # paired per seed: identical test rows, so each line is a paired comparison
        a, b = ps(esp_key(t)), ps(our_key("g1", t))
        for s in SEEDS:
            ax2.plot([j - 0.15, j + 0.15], [a[s], b[s]], color="gray", lw=0.8, alpha=0.7)
        ax2.scatter(np.full(5, j - 0.15), a.to_numpy(), color=colors["esp"], edgecolor="black", s=28, zorder=3,
                    label=bars[0][1] if j == 0 else None)
        ax2.scatter(np.full(5, j + 0.15), b.to_numpy(), color=colors["g1"], edgecolor="black", s=28, zorder=3,
                    label=bars[1][1] if j == 0 else None)
        ax2.text(j, max(a.max(), b.max()) + 0.12, f"÷{(a / b).mean():.1f}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(MAIN_TARGETS))); ax2.set_xticklabels([SHORT[t] for t in MAIN_TARGETS], fontsize=8.5)
    ax2.set_ylabel("test MAE per seed (kcal/mol)")
    ax2.set_title(f"Paired by seed: Espley {MAIN_ESP_MODEL} vs G1 · {MAIN_ARM} · {MAIN_MODEL}; ÷ = mean MAE ratio",
                  fontsize=10.5)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.12); ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=7.5, loc="upper left")
    footnote(fig, [same_note, f"Differs: geometry and features — {geo}",
                   "Models fixed a priori (no best-of; appendix table for that): Espley SVR = their published best "
                   "(Interaction, ΔE‡, ΔG‡: their stored test predictions), ours KRR on ESPLEY46.",
                   f"* Dipole / dipolarophile = role-consistent targets (Espley's distortion_energy_1/2 put the "
                   f"dipolarophile first in {R['n_swapped']:,}/{R['n']:,} reactions); Espley SVR retrained on them with "
                   "their own protocol (their grid, tuned once on seed 23, X-only scaling, AM1 distortions re-assigned "
                   "by the same swap; the re-run reproduces their hps.pkl on their index d1/d2). Index-based d1/d2: "
                   "appendix."], width=205, fontsize=7.6)
    write_atomic(RES / "espley_compare_main.png", lambda f: fig.savefig(f, dpi=150, format="png"))
    plt.close(fig)

    # role figure: Espley 46 AM1 retrained two ways vs ours, on the role targets
    ents = [((ESP_SIDE, PROTO["theirs"], "AM1-46", "SVR"), "Espley 46 AM1 · SVR · their protocol", "#7f7f7f"),
            ((ESP_SIDE, PROTO["theirs"], "AM1-46", "KRR"), "Espley 46 AM1 · KRR · their protocol", "#a8a8a8"),
            ((ESP_SIDE, PROTO["ours"], "AM1-46", "KRR"), "Espley 46 AM1 · KRR · our pipeline", "#d9d9d9"),
            (our_key("g1", None)[:4], f"Ours G1 · xTB geometry · {MAIN_ARM} · {MAIN_MODEL}", colors["g1"])]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x, w = np.arange(len(ROLE_TARGETS)), 0.19
    for i, (k, lab, col) in enumerate(ents):
        r = [G.loc[k + (t,)] for t in ROLE_TARGETS]
        v, e = np.array([q.mae for q in r]), np.array([q.se for q in r])
        off = (i - (len(ents) - 1) / 2) * w
        ax.bar(x + off, v, w, yerr=e, capsize=3, color=col, edgecolor="black", lw=0.5, label=lab)
        for j in range(len(x)):
            ax.text(x[j] + off, v[j] + e[j] + 0.04, f"{v[j]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[t] for t in ROLE_TARGETS])
    ax.set_ylabel("test MAE (kcal/mol), mean of 5 seeds ± SE")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Role-consistent distortions on Espley's ds3 rows and splits · different geometry", fontsize=10.5)
    rp = rep["replication"]
    stored = [G.loc[esp_key(t)].mae for t in INDEX_D]
    rerun = [G.loc[(ESP_SIDE, PROTO["theirs"], "AM1-46", MAIN_ESP_MODEL, t)].mae for t in INDEX_D]
    maxd = max(v["max_abs_pred_diff"] for t in rp for v in rp[t][MAIN_ESP_MODEL]["per_seed"].values())
    footnote(fig, [f"Targets: Espley's distortion_energy_1/2 re-assigned to dipole / dipolarophile "
                   f"({R['n_swapped']:,}/{R['n']:,} swapped); their AM1 distortion features re-assigned by the same "
                   "swap vector.",
                   f"Their protocol: their SVR / KRR grids (hyp_tuning.py), tuned once on the seed-23 training split "
                   f"on {len(rep['tuning_features'])} features (the set whose re-run gives their hps.pkl), "
                   "StandardScaler on X only, refit per seed on 46. Our pipeline: nested per-seed GridSearchCV, X and "
                   "y scaled.",
                   f"Check, their protocol re-run on their index targets vs their stored SVR (same rows): d1 "
                   f"{rerun[0]:.2f} vs {stored[0]:.2f}, d2 {rerun[1]:.2f} vs {stored[1]:.2f}; max |Δprediction| "
                   f"{maxd:.1e} kcal/mol; tuned SVR / KRR params = hps.pkl: yes.",
                   f"Geometry differs — {geo}",
                   f"Scored on the same {n_use:,}/{n:,} reactions as the main figure."], width=165, fontsize=7.4)
    write_atomic(RES / "espley_compare_role.png", lambda f: fig.savefig(f, dpi=150, format="png"))
    plt.close(fig)

    pd.set_option("display.width", 250)
    print("MAIN\n", main.round(3).to_string(index=False))
    print("ROLE\n", role.round(3).to_string(index=False))
    print("INDEX d1/d2 (appendix)\n", idx.round(3).to_string(index=False))
    print("their protocol:", json.dumps({k: rep.get(k) for k in ("hps_off_grid", "tuned")}, indent=1, default=_native))
    print("wrote", sorted(str(p) for p in RES.glob("espley_compare_*")))


if __name__ == "__main__":
    {"prep": lambda: prep(), "train": lambda: train(int(sys.argv[2])),
     "espley_role": lambda: espley_role(sys.argv[2]), "plot": lambda: plot()}[sys.argv[1]]()
