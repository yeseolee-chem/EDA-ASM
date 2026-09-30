#!/usr/bin/env python3
"""compare_espley.py — like-for-like comparison with Espley et al. 2024 (ds3, [3+2]).

Everything that can be held equal is held equal:
  reactions   the 3,510 reactions of Espley's ds3 ML set (their `reaction_number`
              = Coley reaction id = our rxn_id), in their row order
  targets     their DFT target values (B3LYP-D3(BJ)/def2-TZVP SMD(water), Gaussian),
              so both sides are scored against the same numbers
  splits      their protocol exactly: train_test_split(test_size=0.2, rs=seed), then
              the first half of the remaining 20 % is the test set; seeds 22/23/14/1/2.
              Verified against the test targets stored in their ml_results.pkl.
  metric      test MAE averaged over the 5 seeds; error bar = their definition,
              std(|error|)/sqrt(n_test) per seed, averaged over seeds
  selection   "best model" per target on both sides (lowest mean test MAE among the
              models each side tried)
What differs: the features (theirs: 46 AM1 features on AM1-optimised geometries;
ours: xTB ESPLEY46 / ESPLEY73 on the DFT geometries) and the model family / tuning.

  python compare_espley.py prep          # validate mapping, labels, splits -> prep.pkl
  python compare_espley.py train <k>     # k 0-4 Espley targets, 5-6 role-consistent d1/d2 (ours only)
  python compare_espley.py plot          # figure + table
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, GRIDS, _make_pipe  # noqa: E402

ESP = Path(os.environ.get("ESPLEY_REPO_DATA", "/gpfs/tmp_cpu2/yeseo1ee/espley_compare"))
OUT = ESP / "results"
FEAT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb")) / "xtb_features.parquet"
LABELS = HERE.parent.parent / "labels_all.json"
RES = HERE / "results"
SEEDS = [22, 23, 14, 1, 2]
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))
# Espley target -> (display name, our label key, xTB pre-ML feature)
# Espley's distortion_energy_1/2 follow their own reactant index, which is the dipole /
# dipolarophile in ~69 % of rows and swapped in ~31 % (prep() measures this); both sides
# are trained and scored on these raw columns, so they are named by index, not by role.
TARGETS = {
    "distortion_energy_1_dft": ("Distortion 1", "d1_kcal", "xtb_dist_dipole_kcal"),
    "distortion_energy_2_dft": ("Distortion 2", "d2_kcal", "xtb_dist_dipolarophile_kcal"),
    "interaction_energies_dft": ("Interaction", "eint_spe_kcal", "xtb_interaction_kcal"),
    "e_barrier_dft": ("ΔE‡", "barrier_kcal", "xtb_e_barrier_kcal"),
    "q_barrier_dft": ("ΔG‡", None, "xtb_e_barrier_kcal"),
}
ESP_MODELS = {"ridge": "Ridge", "krr": "KRR", "svr": "SVR", "2_st_nn": "2-layer NN", "4_st_nn": "4-layer NN"}
# Espley stores the interaction energy with the opposite sign (positive = stabilising)
SIGN = {"interaction_energies_dft": -1.0}
# ours only (no Espley counterpart): Espley's d1/d2 numbers re-assigned to dipole / dipolarophile
ROLE_TARGETS = {"dipole_distortion_role": ("Dipole distortion (role-consistent)", 0),
                "dipolarophile_distortion_role": ("Dipolarophile distortion (role-consistent)", 1)}
TKEYS = list(TARGETS) + list(ROLE_TARGETS)
ARMS = {"ESPLEY46": FEATURE_SETS["ESPLEY46"], "ESPLEY73": FEATURE_SETS["ESPLEY73"]}


def espley_split(n, seed):
    """Their _perform_train_test_split: X_test is the FIRST half of the 20 % hold-out."""
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed)
    te, _va = train_test_split(rest, test_size=0.5, random_state=seed)
    return tr, te


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    esp = pd.read_pickle(ESP / "feature_selection/_f_selection/tt/manual_tt_solvent.pkl").reset_index(drop=True)
    res = pd.read_pickle(ESP / "machine_learning/tt/solvent/ml_results.pkl")
    feat = pd.read_parquet(FEAT).set_index("rxn_id")
    lab = {int(r["rxn_id"]): r for r in json.load(open(LABELS))}
    ids = esp["reaction_number"].astype(int).values
    n = len(esp)
    print(f"Espley ds3 rows: {n}, unique reaction_number: {len(set(ids))}")

    # 1. id mapping + label agreement (their Gaussian DFT vs our ORCA labels)
    have_lab = np.array([i in lab and lab[i]["status"] == "ok" for i in ids])
    have_feat = np.array([i in feat.index and feat.loc[i, "xtb_status"] == "ok" for i in ids])
    print(f"in our labels (ok): {have_lab.sum()}/{n}; in our xTB features (ok): {have_feat.sum()}/{n}")
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
    # Espley's own AM1 distortions: same index convention as their DFT columns?
    a1, a2 = esp["distortion_energy_1_am1"].values, esp["distortion_energy_2_am1"].values
    role["am1_same_index_r"] = float(np.corrcoef(a1, t1)[0, 1])
    role["am1_swapped_index_r"] = float(np.corrcoef(a1, t2)[0, 1])
    # their features are role-based (…_di_… = 3-atom dipole, …_dp_… = 2-atom dipolarophile);
    # does the dipole-feature block track their _1 target or the true dipole strain?
    xd = np.array([feat.loc[i, "xtb_dist_dipole_kcal"] if have_feat[k] else np.nan for k, i in enumerate(ids)])
    ms = have_lab & have_feat & sw
    role["swapped_rows_corr_xtbdipole_vs_t1"] = float(np.corrcoef(xd[ms], t1[ms])[0, 1])
    role["swapped_rows_corr_xtbdipole_vs_t2"] = float(np.corrcoef(xd[ms], t2[ms])[0, 1])
    print("  d1/d2 index vs role:", json.dumps(role, indent=1))

    # 2. split replication: our split must reproduce their stored test targets exactly
    split_ok = {}
    for t in TARGETS:
        row = res[res.model_target == f"svr_{t}"].iloc[0]
        ok = []
        for s_i, s in enumerate(row.random_state):
            _, te = espley_split(n, int(s))
            ok.append(bool(np.allclose(esp[t].values[te], np.asarray(row.y_test_true[s_i]).ravel(), atol=1e-9)))
        split_ok[t] = ok
        print(f"  split replication {t:26s} seeds {list(row.random_state)} -> {ok}")
    assert all(all(v) for v in split_ok.values()), "split replication failed"

    # 3. Espley's own per-seed test metrics, all models
    esp_rows = []
    for _, row in res.iterrows():
        mt = row.model_target
        model = next(k for k in sorted(ESP_MODELS, key=len, reverse=True) if mt.startswith(k + "_"))
        t = mt[len(model) + 1:]
        for s_i, s in enumerate(row.random_state):
            y = np.asarray(row.y_test_true[s_i]).ravel(); p = np.asarray(row.y_test_pred_values[s_i]).ravel()
            _, te = espley_split(n, int(s))            # y is in te order (checked in step 2)
            e = np.abs(y - p)[have_feat[te]]           # score on exactly the rows ours is scored on
            esp_rows.append(dict(side="Espley (AM1, 46 feat.)", arm="AM1-46", model=ESP_MODELS[model], target=t,
                                 seed=int(s), mae=float(e.mean()), se=float(e.std() / np.sqrt(len(e))), n_test=len(e),
                                 n_test_espley=len(y), mae_all_rows=float(np.abs(y - p).mean())))
    print("reactions without our xTB features:", [int(i) for i in ids[~have_feat]])
    pd.DataFrame(esp_rows).to_csv(OUT / "espley_per_seed.csv", index=False)

    # 4. pre-ML baselines on the same 3,510 rows
    pre = {}
    for t, (_, _, xf) in TARGETS.items():
        m = have_feat
        x = np.array([feat.loc[i, xf] if have_feat[k] else np.nan for k, i in enumerate(ids)])[m]
        pre[t] = float(np.mean(np.abs(x - SIGN.get(t, 1.0) * esp[t].values[m])))
    pickle.dump(dict(ids=ids, have_feat=have_feat, agree=agree, split_ok=split_ok, pre_xtb=pre, n=n,
                     swap=swap, role=role), open(OUT / "prep.pkl", "wb"))
    print("pre-ML xTB MAE:", {k: round(v, 2) for k, v in pre.items()})


def target_values(t, esp, P):
    if t in TARGETS:
        return esp[t].values
    t1, t2 = esp["distortion_energy_1_dft"].values, esp["distortion_energy_2_dft"].values
    sw = P["swap"]
    y = np.where(sw == 1, t2, t1) if ROLE_TARGETS[t][1] == 0 else np.where(sw == 1, t1, t2)
    return np.where(np.isnan(sw), np.nan, y)


def train(k):
    t = TKEYS[k]
    out = OUT / f"ours_per_seed_{k}.csv"
    if out.exists():
        print("exists:", out); return
    P = pickle.load(open(OUT / "prep.pkl", "rb"))
    esp = pd.read_pickle(ESP / "feature_selection/_f_selection/tt/manual_tt_solvent.pkl").reset_index(drop=True)
    feat = pd.read_parquet(FEAT).set_index("rxn_id")
    ids, n = P["ids"], P["n"]
    y_all = target_values(t, esp, P)
    rows = []
    for arm, cols in ARMS.items():
        X_all = feat.reindex(ids)[cols].values          # rows in Espley order; NaN where we lack features
        ok = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
        for name, (est, grid) in GRIDS.items():
            for s in SEEDS:
                tr, te = espley_split(n, s)
                tr = tr[ok[tr]]
                te_ok = te[ok[te]]
                gs = GridSearchCV(_make_pipe(est), grid, cv=KFold(5, shuffle=True, random_state=s),
                                  scoring="neg_mean_absolute_error", n_jobs=N_JOBS, error_score="raise")
                gs.fit(X_all[tr], y_all[tr])
                e = np.abs(gs.best_estimator_.predict(X_all[te_ok]) - y_all[te_ok])
                rows.append(dict(side=f"Ours (xTB, {arm[6:]} feat.)", arm=arm, model=name.replace("_rbf", ""),
                                 target=t, seed=s, mae=float(e.mean()), se=float(e.std() / np.sqrt(len(e))),
                                 n_test=len(e), n_test_espley=len(te)))
                print(f"{t} {arm} {name} seed {s}: MAE {e.mean():.3f} (n={len(e)}/{len(te)})", flush=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = pickle.load(open(OUT / "prep.pkl", "rb"))
    esp = pd.read_csv(OUT / "espley_per_seed.csv")
    ours = pd.concat([pd.read_csv(OUT / f"ours_per_seed_{k}.csv") for k in range(len(TKEYS))])
    allr = pd.concat([esp, ours])
    g = allr.groupby(["side", "arm", "model", "target"]).agg(mae=("mae", "mean"), se=("se", "mean"),
                                                            sd=("mae", "std"), n_seeds=("seed", "nunique")).reset_index()
    best = g.loc[g.groupby(["side", "target"]).mae.idxmin()].copy()
    tbl = best.pivot(index="target", columns="side", values=["mae", "se", "model"])
    sides = ["Espley (AM1, 46 feat.)", "Ours (xTB, 46 feat.)", "Ours (xTB, 73 feat.)"]
    colors = {"Espley (AM1, 46 feat.)": "#7f7f7f", "Ours (xTB, 46 feat.)": "#6baed6", "Ours (xTB, 73 feat.)": "#08519c"}
    tlist = list(TARGETS)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.9), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(tlist)); w = 0.26
    for i, s in enumerate(sides):
        v = [tbl.loc[t, ("mae", s)] for t in tlist]; e = [tbl.loc[t, ("se", s)] for t in tlist]
        b = ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=3, color=colors[s], edgecolor="black", lw=0.5, label=s)
        for j, t in enumerate(tlist):
            ax.text(x[j] + (i - 1) * w, v[j] + e[j] + 0.05, f"{v[j]:.2f}\n{tbl.loc[t, ('model', s)]}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([TARGETS[t][0] for t in tlist])
    ax.set_ylabel("test MAE (kcal/mol) — best model per side\nmean of 5 seeds, error bar = SE (Espley's definition)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Same ds3 reactions · same DFT targets · identical 80/10/10 test rows (seeds 22/23/14/1/2)",
                 fontsize=10.5)

    # paired per-seed view: identical test set per seed, so each line is a paired comparison
    e_b = best.set_index(["side", "target"])
    for j, t in enumerate(tlist):
        a = esp[(esp.model == e_b.loc[("Espley (AM1, 46 feat.)", t), "model"]) & (esp.target == t)].set_index("seed").mae
        o = ours[(ours.arm == "ESPLEY73") & (ours.model == e_b.loc[("Ours (xTB, 73 feat.)", t), "model"])
                 & (ours.target == t)].set_index("seed").mae
        for s in SEEDS:
            ax2.plot([j - 0.15, j + 0.15], [a[s], o[s]], color="gray", lw=0.8, alpha=0.7)
        ax2.scatter(np.full(5, j - 0.15), a[SEEDS], color=colors[sides[0]], edgecolor="black", s=28, zorder=3,
                    label=sides[0] if j == 0 else None)
        ax2.scatter(np.full(5, j + 0.15), o[SEEDS], color=colors[sides[2]], edgecolor="black", s=28, zorder=3,
                    label=sides[2] if j == 0 else None)
        ax2.text(j, max(a.max(), o.max()) + 0.12, f"÷{(a[SEEDS] / o[SEEDS]).mean():.1f}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(tlist))); ax2.set_xticklabels([TARGETS[t][0] for t in tlist], rotation=20, ha="right")
    ax2.set_ylabel("test MAE per seed (kcal/mol)")
    ax2.set_title("Paired by seed (same test reactions); ÷ = mean MAE ratio", fontsize=10.5)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1); ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=8, loc="upper left")
    R = P["role"]
    notes = ["Held equal: reactions and test rows (3,509 of 3,510; one reaction has no xTB features and is dropped "
             "from both sides), targets (Espley's Gaussian B3LYP-D3(BJ)/def2-TZVP SMD(water) values),",
             "splits, seeds, metric, best-of-models selection.   Differs: features — Espley 46 AM1 features on "
             "AM1-optimised geometries; ours xTB single points on the DFT geometries (an upper bound).",
             f"Distortion 1/2 = Espley's reactant index: the dipolarophile, not the dipole, is '1' in "
             f"{R['n_swapped']:,}/{R['n']:,} reactions ({R['n_swapped'] / R['n']:.0%}). Both sides use role-based "
             "(dipole / dipolarophile) features, so both carry the same handicap on these two targets."]
    for i, line in enumerate(notes):
        fig.text(0.5, 0.058 - 0.024 * i, line, ha="center", va="bottom", fontsize=8.2, style="italic")
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    out = RES / "figures" / "espley_vs_ours_ds3.png"
    fig.savefig(out, dpi=150); print("wrote", out)

    rows = []
    for t in tlist:
        r = {"target": TARGETS[t][0]}
        for s in sides:
            r[f"{s} model"] = tbl.loc[t, ("model", s)]
            r[f"{s} MAE"] = round(float(tbl.loc[t, ("mae", s)]), 3)
            r[f"{s} SE"] = round(float(tbl.loc[t, ("se", s)]), 3)
        # xTB gives an electronic barrier only, so there is no pre-ML baseline for ΔG‡
        r["pre-ML xTB MAE"] = None if t == "q_barrier_dft" else round(P["pre_xtb"][t], 2)
        rows.append(r)
    for t in ROLE_TARGETS:                     # ours only: role-consistent d1/d2, no Espley counterpart
        r = {"target": ROLE_TARGETS[t][0]}
        for s in sides[1:]:
            b = best[(best.side == s) & (best.target == t)].iloc[0]
            r[f"{s} model"] = b.model; r[f"{s} MAE"] = round(float(b.mae), 3); r[f"{s} SE"] = round(float(b.se), 3)
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "espley_vs_ours_ds3.csv", index=False)
    g.round(4).to_csv(RES / "espley_vs_ours_ds3_all_models.csv", index=False)
    allr.round(4).to_csv(RES / "espley_vs_ours_ds3_per_seed.csv", index=False)
    json.dump(dict(label_agreement=P["agree"], d1d2_index_vs_role=P["role"], split_replicated=P["split_ok"],
                   pre_ml_xtb_mae=P["pre_xtb"], dropped_no_features=[int(i) for i in P["ids"][~P["have_feat"]]]),
              open(RES / "espley_vs_ours_ds3_checks.json", "w"), indent=1)
    print(pd.DataFrame(rows).to_string(index=False))
    print("label agreement:", json.dumps(P["agree"], indent=1))
    print("d1/d2 index vs role:", json.dumps(P["role"], indent=1))


if __name__ == "__main__":
    {"prep": lambda: prep(), "train": lambda: train(int(sys.argv[2])), "plot": lambda: plot()}[sys.argv[1]]()
