"""run_all.py — D1 통계 분석 일괄 실행 (QM 없음, 수 분).

python run_all.py --labels LABELS_D1.json --d0 labels_all.json --out OUTDIR
                  [--manifest d1_manifest.csv --status d1_status.csv --status-col CLS]
                  [--sigma sigma_hansch1991.csv]

출력(OUTDIR): qc.json, pairs_H.csv, pairs_aryl.csv, sel_regio.csv, sel_exoendo.csv,
  budget.csv, zeta.csv, corr_dd.csv, transfer.csv, lowrank.csv, joint_lowrank.csv, pca.csv,
  selectivity.csv, regio_known.csv, fingerprint_direct.csv, sign_consistency.csv, presence_proxy.csv,
  aryl_lowrank.csv, domain.json, [missing_logit.csv], [hammett.csv, sigma_vs_rank2.csv], headline.json
기존 파일을 덮어쓴다(분석은 결정론적: seed 고정).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1s_prep import *          # noqa: F401,F403
from d1s_lowrank import als, additive, subst_only, cv, matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--d0", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--manifest"); ap.add_argument("--status"); ap.add_argument("--status-col", default="cls")
ap.add_argument("--sigma")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
H = {}
def w(df, name, index=True):
    df.to_csv(OUT / name, index=index)

# ------------------------------------------------------------- A. load + QC
df = load_labels(a.labels)
qc = dict(n_rows=len(df), n_parent=int(df.is_parent.sum()),
          q_badref=int(df.q_badref.sum()), q_stepwise=int(df.q_stepwise.sum()), q_bad=int(df.q_bad.sum()),
          closure_max=float(df.closure.abs().max()),
          orient_counts=df.orient.value_counts().to_dict(),
          n_gates_failed=int((~df.gates_ok).sum()),
          n_dipole=df.dipole.value_counts().to_dict(), n_skeleton=df.skeleton.value_counts().to_dict(),
          barrier_neg_frac=float((df.barrier_kcal < 0).mean()),
          bad_rows=df.loc[df.q_bad, ["job_id", "ts_id", "d1_kcal", "d2_kcal", "formed_d1", "formed_d2", "barrier_kcal"]].round(3).to_dict("records"))
json.dump(qc, open(OUT / "qc.json", "w"), indent=1, ensure_ascii=False, default=str)
H.update(n_rows=len(df), q_bad=qc["q_bad"])

# ------------------------------------------------------------- B. pairs
P = pairs_vs_H(df); P.to_csv(OUT / "pairs_H.csv", index=False)
Pc = P[~P.q_bad].copy()
H.update(n_pairs=len(P), n_pairs_clean=len(Pc), n_children_no_parent=int((~df.is_parent).sum() - len(P)))
big = Pc[Pc.barrier_kcal.abs() > 15][["job_id", "parent_job", "core", "R", "barrier_kcal", "d1_kcal", "d2_kcal"]]
w(big, "inspect_large_ddbarrier.csv", index=False)

# ------------------------------------------------------------- C. covariance budget
rows = []
def add(name, D):
    b = budget(D); b.update(set=name, n=len(D), sd_dE=D.barrier_kcal.std()); rows.append(b)
add("ALL pairs", Pc)
for s in ["dipole", "dipolarophile"]:
    add(f"side={s}", Pc[Pc.side == s])
    for at in ["direct", "p", "m", "o"]:
        add(f"side={s}, attach={at}", Pc[(Pc.side == s) & (Pc.attach == at)])
for dp in sorted(Pc.dipole.unique()):
    add(f"dipole={dp}", Pc[Pc.dipole == dp])
add("D1 absolute (between all TS)", df[~df.q_bad])
d0 = pd.DataFrame(json.load(open(a.d0)))
d0 = d0[(~d0.flag_foreign_bond.astype(bool)) & (d0.oi_dft < 0)].copy()
d0["bsse_shift_kcal"] = d0.eint_spe_kcal - d0.e_bond_kcal
add("D0 absolute (between all TS)", d0)
B = pd.DataFrame(rows).set_index("set"); w(B, "budget.csv")
H["budget_all_pairs"] = {k: round(float(B.loc["ALL pairs", k]), 3) for k in ["d1", "d2", "elst", "Pauli", "OI", "disp", "CPCM", "CDS", "bsse"]}
Pc["dstrain"] = Pc.d1_kcal + Pc.d2_kcal
Pc["dint"] = Pc[["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]].sum(axis=1)
H["r_strain_barrier"] = round(float(np.corrcoef(Pc.dstrain, Pc.barrier_kcal)[0, 1]), 3)
H["r_eint_barrier"] = round(float(np.corrcoef(Pc.dint, Pc.barrier_kcal)[0, 1]), 3)

# ------------------------------------------------------------- D. TS-position coupling
T = CH + ["bsse_shift_kcal", "barrier_kcal"]
X1, X2 = Pc[["zeta"]].values, Pc[["zeta", "asyn"]].values
zr = []
for c in T:
    y = Pc[c].values
    m1, m2 = LinearRegression().fit(X1, y), LinearRegression().fit(X2, y)
    zr.append(dict(ch=CHN[c], r_dzeta=np.corrcoef(Pc.zeta, y)[0, 1], slope_per_0p1A=m1.coef_[0] * 0.1,
                   R2_dzeta=m1.score(X1, y), R2_dzeta_dasyn=m2.score(X2, y), sd=y.std()))
Z = pd.DataFrame(zr).set_index("ch"); w(Z, "zeta.csv")
H["r_dzeta_barrier"] = round(float(Z.loc["ΔE‡", "r_dzeta"]), 3)
w(Pc[CH].rename(columns=CHN).corr(), "corr_dd.csv")

# ------------------------------------------------------------- E. transferability + low rank
lr = []
for side in ["dipole", "dipolarophile"]:
    for lab, sub in [("direct", Pc[(Pc.side == side) & (Pc.attach == "direct")]), ("all80", Pc[Pc.side == side])]:
        for c in T:
            if c == "bsse_shift_kcal":
                continue
            M, mask, piv = matrix(sub, c)
            r = dict(side=side, set=lab, ch=CHN[c], n_ctx=M.shape[0], n_sub=M.shape[1], fill=mask.mean(), zero_MAE=np.abs(M[mask]).mean())
            r["R2_subst_only"], r["MAE_subst_only"] = cv(M, mask, subst_only)
            r["R2_additive"], r["MAE_additive"] = cv(M, mask, additive)
            for k in (1, 2, 3):
                r[f"R2_rank{k}"], r[f"MAE_rank{k}"] = cv(M, mask, lambda MM, mm, k=k: als(MM, mm, k)[0])
            # leave-one-core-out transfer of a scalar substituent constant
            pred = np.full(len(sub), np.nan)
            for core in sub.core.unique():
                tr = (sub.core != core).values
                mu = sub[tr].groupby("R")[c].mean()
                pred[~tr] = sub.loc[~tr, "R"].map(mu).values
            ok = ~np.isnan(pred); y = sub[c].values
            r["LOCO_R2_scalar"] = 1 - np.mean((y[ok] - pred[ok]) ** 2) / np.var(y[ok])
            lr.append(r)
L = pd.DataFrame(lr); w(L, "lowrank.csv", index=False)
q = L[(L.side == "dipole") & (L.set == "direct") & (L.ch == "ΔE‡")].iloc[0]
H["lowrank_dipole_direct_dE"] = {k: round(float(q[k]), 3) for k in ["R2_subst_only", "R2_additive", "R2_rank1", "R2_rank2", "R2_rank3", "LOCO_R2_scalar"]}
jl = []
for side in ["dipole", "dipolarophile"]:
    for lab, sub in [("direct", Pc[(Pc.side == side) & (Pc.attach == "direct")]), ("all80", Pc[Pc.side == side])]:
        blocks = []
        for c in CH:
            piv = sub.pivot_table(index="ctx", columns="R", values=c, aggfunc="mean")
            piv = piv / np.nanstd(piv.values); piv.index = [f"{CHN[c]}|{i}" for i in piv.index]; blocks.append(piv)
        Bm = pd.concat(blocks); M = Bm.values; mask = ~np.isnan(M); M = np.nan_to_num(M)
        r = dict(side=side, set=lab, rows=M.shape[0], n_sub=M.shape[1])
        r["R2_subst_only"] = cv(M, mask, subst_only)[0]; r["R2_additive"] = cv(M, mask, additive)[0]
        for k in (1, 2, 3, 4, 5):
            r[f"R2_rank{k}"] = cv(M, mask, lambda MM, mm, k=k: als(MM, mm, k)[0])[0]
        jl.append(r)
w(pd.DataFrame(jl), "joint_lowrank.csv", index=False)
Zs = (Pc[CH] - Pc[CH].mean()) / Pc[CH].std()
pca = PCA().fit(Zs.values); sc = pca.transform(Zs.values)
pc = pd.DataFrame(pca.components_.T, index=[CHN[c] for c in CH], columns=[f"PC{i+1}" for i in range(len(CH))])
pc.loc["explained_var_ratio"] = pca.explained_variance_ratio_
pc.loc["r_with_dzeta"] = [np.corrcoef(sc[:, i], Pc.zeta)[0, 1] for i in range(len(CH))]
pc.loc["r_with_ddE"] = [np.corrcoef(sc[:, i], Pc.barrier_kcal)[0, 1] for i in range(len(CH))]
w(pc, "pca.csv")
H["pca_evr_top3"] = [round(float(v), 3) for v in pca.explained_variance_ratio_[:3]]

# ------------------------------------------------------------- F. selectivity
R = sibling_pairs(df[df.orient.isin(["a", "b"])], ["cell_id", "face", "stereo_index"], "orient", "a", "b")
dB = df[df.face.isin(["exo", "endo"])]
E = sibling_pairs(dB, ["cell_id", "orient"], "face", "exo", "endo")
R.to_csv(OUT / "sel_regio.csv", index=False); E.to_csv(OUT / "sel_exoendo.csv", index=False)
def summ(S, name):
    res = dict(set=name, n=len(S), sd=S.barrier_kcal.std()); res.update(budget(S))
    win = np.sign(S.barrier_kcal)
    for c in BUD:
        res["acc_" + CHN[c]] = (np.sign(S[c]) == win).mean()
    res["acc_strain"] = (np.sign(S.d1_kcal + S.d2_kcal) == win).mean()
    ei = S[["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]].sum(axis=1)
    res["acc_Eint"] = (np.sign(ei) == win).mean()
    res["acc_earlierTS"] = (np.sign(-S.zeta) == win).mean()
    return res
rows = [summ(R, "regio a−b (all)")] + [summ(R[R.dipole == d], f"regio a−b {d}") for d in sorted(R.dipole.unique())]
rows += [summ(E, "exo−endo (all)")] + [summ(E[E.dipole == d], f"exo−endo {d}") for d in sorted(E.dipole.unique())]
S = pd.DataFrame(rows).set_index("set"); w(S, "selectivity.csv")
R2s = R[R.series == "R2"].copy(); R2s["a_pref"] = R2s.barrier_kcal < 0
R2s["skel_grp"] = np.where(R2s.skeleton.isin(["A", "B", "C"]), "A+B+C", R2s.skeleton)
known = R2s.groupby(["dipole", "skel_grp"]).a_pref.agg(["mean", "size"]); w(known, "regio_known.csv")
H["regio_a_pref_ABC"] = {d: round(float(known.loc[(d, "A+B+C"), "mean"]), 3) for d in known.index.get_level_values(0).unique() if (d, "A+B+C") in known.index}
H["regio_acc_strain"] = round(float(S.loc["regio a−b (all)", "acc_strain"]), 3)
H["regio_acc_Eint"] = round(float(S.loc["regio a−b (all)", "acc_Eint"]), 3)
H["exo_pref_frac"] = round(float((E.barrier_kcal < 0).mean()), 3)
H["exo_acc_d2"] = round(float(S.loc["exo−endo (all)", "acc_d2"]), 3)
H["n_regio_pairs"], H["n_exoendo_pairs"] = len(R), len(E)

# ------------------------------------------------------------- G. fingerprints
rng = np.random.default_rng(0)
fp = []
for side in ["dipole", "dipolarophile"]:
    s = Pc[(Pc.side == side) & (Pc.attach == "direct")]
    for Xs, g in s.groupby("X"):
        r = dict(side=side, X=Xs, n=len(g), n_core=g.core.nunique())
        cores = g.core.unique()
        for c in T:
            r[CHN[c]] = g[c].median()
            byc = {k: g.loc[g.core == k, c].values for k in cores}
            bs = [np.median(np.concatenate([byc[k] for k in rng.choice(cores, len(cores))])) for _ in range(300)]
            r[CHN[c] + "_lo"], r[CHN[c] + "_hi"] = np.percentile(bs, [2.5, 97.5])
        fp.append(r)
FP = pd.DataFrame(fp); w(FP, "fingerprint_direct.csv", index=False)
cons = Pc[Pc.attach == "direct"].groupby(["side", "X"]).apply(lambda g: (np.sign(g.barrier_kcal) == np.sign(g.barrier_kcal.median())).mean())
w(cons.rename("sign_consistency").to_frame(), "sign_consistency.csv")
H["sign_consistency_median"] = {s: round(float(cons.loc[s].median()), 2) for s in ["dipole", "dipolarophile"]}

# ------------------------------------------------------------- H. presence proxy (without status file)
c = df[~df.is_parent].groupby(["dipole", "skeleton", "series", "R"]).size().rename("n").reset_index()
sib = c.groupby(["dipole", "skeleton", "series"]).n.max().rename("n_sib")
allR = sorted(df.loc[~df.is_parent, "R"].unique())
G = pd.DataFrame([(*k, Rr, ns) for k, ns in sib.items() for Rr in allR], columns=["dipole", "skeleton", "series", "R", "n_sib"]).merge(c, how="left").fillna({"n": 0})
G["present_frac"] = G.n / G.n_sib; w(G, "presence_proxy.csv", index=False)
H["presence_proxy_by_dipole"] = {k: round(float(v), 3) for k, v in G.groupby("dipole").apply(lambda g: g.n.sum() / g.n_sib.sum()).items()}

# ------------------------------------------------------------- I. aryl vs Ph
A = pairs_vs_Ph(df); A.to_csv(OUT / "pairs_aryl.csv", index=False)
ar = []
for side in ["dipole", "dipolarophile"]:
    for pos in ["p", "m", "o"]:
        s = A[(A.side == side) & (A.pos == pos)]
        for c in ["barrier_kcal", "elst_dft", "pauli_dft", "oi_dft", "d1_kcal", "d2_kcal"]:
            M, mask, piv = matrix(s, c, columns="X")
            r = dict(side=side, pos=pos, ch=CHN[c], n_ctx=M.shape[0], sd=M[mask].std())
            r["R2_subst_only"] = cv(M, mask, subst_only)[0]
            r["R2_rank1"] = cv(M, mask, lambda MM, mm: als(MM, mm, 1)[0])[0]
            r["R2_rank2"] = cv(M, mask, lambda MM, mm: als(MM, mm, 2)[0])[0]
            ar.append(r)
w(pd.DataFrame(ar), "aryl_lowrank.csv", index=False)
H["n_aryl_pairs"] = len(A)

# ------------------------------------------------------------- J. domain D1 vs D0
mu, sd = d0[CH].mean(), d0[CH].std()
Z0 = ((d0[CH] - mu) / sd).values; Z1 = ((df.loc[~df.q_bad, CH] - mu) / sd).values
nn = NearestNeighbors(n_neighbors=2).fit(Z0)
thr = np.percentile(nn.kneighbors(Z0)[0][:, 1], 95)
d10 = nn.kneighbors(Z1, n_neighbors=1)[0][:, 0]
dom = dict(d0_nn95=float(thr), d1_frac_beyond=float((d10 > thr).mean()))
json.dump(dom, open(OUT / "domain.json", "w"), indent=1)
H["d1_frac_outside_d0"] = round(dom["d1_frac_beyond"], 3)

# ------------------------------------------------------------- K. optional: missingness with status
if a.manifest and a.status:
    import statsmodels.formula.api as smf
    man = pd.read_csv(a.manifest); st = pd.read_csv(a.status)
    m = man.merge(st[["job_id", a.status_col]], on="job_id", how="left")
    m["ok"] = (m[a.status_col] == "ok").astype(int)
    m["R"] = np.where(m.series == "R1", m.R1, m.R2); m["X"] = m.R.astype(str).str.replace(r"^[pmo]-", "", regex=True)
    tab = m.groupby(["dipole", "X"]).ok.mean().unstack(0); w(tab, "ok_rate_dipole_X.csv")
    w(m.groupby(["dipole", a.status_col]).size().unstack(fill_value=0), "status_by_dipole.csv")
    fit = smf.logit("ok ~ C(dipole) + C(skeleton) + C(X) + C(attach) + C(series)", data=m).fit(disp=0, maxiter=200)
    w(pd.DataFrame(dict(coef=fit.params, se=fit.bse, p=fit.pvalues)), "missing_logit.csv")
    H["ok_rate_overall"] = round(float(m.ok.mean()), 3)

# ------------------------------------------------------------- L. optional: Hammett with literature sigma
if a.sigma:
    sg = pd.read_csv(a.sigma).set_index("X")      # columns: sigma_m, sigma_p, F, R  (Hansch-Leo-Taft 1991 table values)
    hm = []
    for side in ["dipole", "dipolarophile"]:
        for pos, scol in [("p", "sigma_p"), ("m", "sigma_m")]:
            s = A[(A.side == side) & (A.pos == pos)].join(sg, on="X")
            s = s.dropna(subset=[scol])
            for c in ["barrier_kcal"] + CH:
                for ctx, g in s.groupby("ctx"):
                    if len(g) < 8:
                        continue
                    lr1 = LinearRegression().fit(g[[scol]], g[c]); lr2 = LinearRegression().fit(g[["F", "R"]], g[c])
                    hm.append(dict(side=side, pos=pos, ch=CHN[c], ctx=ctx, n=len(g), rho=lr1.coef_[0], R2_sigma=lr1.score(g[[scol]], g[c]),
                                   rho_F=lr2.coef_[0], rho_R=lr2.coef_[1], R2_FR=lr2.score(g[["F", "R"]], g[c])))
    HM = pd.DataFrame(hm); w(HM, "hammett.csv", index=False)
    # learned rank-2 substituent vectors (direct, both sides) vs (F, R): canonical correlation
    from sklearn.cross_decomposition import CCA
    cc = []
    for side in ["dipole", "dipolarophile"]:
        sub = Pc[(Pc.side == side) & (Pc.attach == "direct")]
        for c in ["barrier_kcal"] + CH:
            M, mask, piv = matrix(sub, c)
            _, U, V = als(M, mask, 2, iters=150)
            Vd = pd.DataFrame(V, index=piv.columns).join(sg[["F", "R"]], how="inner").dropna()
            if len(Vd) < 8:
                continue
            cca = CCA(n_components=2).fit(Vd[[0, 1]].values, Vd[["F", "R"]].values)
            u, v = cca.transform(Vd[[0, 1]].values, Vd[["F", "R"]].values)
            cc.append(dict(side=side, ch=CHN[c], n_sub=len(Vd), can_r1=np.corrcoef(u[:, 0], v[:, 0])[0, 1], can_r2=np.corrcoef(u[:, 1], v[:, 1])[0, 1]))
    w(pd.DataFrame(cc), "sigma_vs_rank2.csv", index=False)

json.dump(H, open(OUT / "headline.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(H, indent=1, ensure_ascii=False))
