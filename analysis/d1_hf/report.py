#!/usr/bin/env python3
"""D1-HF reports (docs/specs/D1HF.md sections 6, 8, 9). Run through sbatch only.

  python report.py a     # Phase A (spec 8): results/A_pairs.csv, A_summary.csv, A_report.json, figA_parity.png
  python report.py p     # Phase P report -> results/P_REPORT.md, results/P_metrics.json (exit 3 = the P decision fails)

Reads results/hf_status.csv, results/labels_d1_hf.parquet (build.py all), results/P_sample.csv, results/X_manifest.csv,
results/S1_compare.csv, $HF/runs/<geom>/<T>/{timing,attempts}.tsv and the label values of labels_d1.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402

LAB_KEYS = dict(d1="d1_kcal", d2="d2_kcal", elst="elst_dft", pauli="pauli_dft", oi="oi_dft", disp="disp_dft",
                cpcm="cpcm_dft", cds="cds_dft", barrier="barrier_kcal")
EDA_KEYS = ["bond", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
GATES = ["g1_asm", "g2_sum", "g3_seam", "g_enuc", "g_setting", "orca", "g_bond_ref", "g_scf", "g2_sum_spec", "g3_seam_spec"]
NODE_CORES = 48                                                    # spec 10: one job = one node slot of 48 cores


def fit_line(x, y):
    """y = a + b x (least squares), Pearson r."""
    b, a = np.polyfit(x, y, 1)
    return dict(n=int(len(x)), r=float(np.corrcoef(x, y)[0, 1]), slope=float(b), intercept=float(a))


def cmd_p():
    P = pd.read_csv(H.RES / "P_sample.csv", dtype={"job_id": str})
    st = pd.read_csv(H.RES / "hf_status.csv", dtype={"job_id": str})
    L = pd.read_parquet(H.RES / "labels_d1_hf.parquet")
    X = pd.read_csv(H.RES / "X_manifest.csv", dtype={"job_id": str}, keep_default_na=False)
    s1 = pd.read_csv(H.RES / "S1_compare.csv")
    ids = set(P.job_id)
    sd = st[(st.geom == "dft") & st.job_id.isin(ids)]
    n_ok, n = int((sd.status == "ok").sum()), len(ids)
    reasons = sd[sd.status != "ok"].groupby(["status", "reason"]).size().reset_index(name="n").values.tolist()
    att = []
    for geom in ("dft", "g1"):
        for T in ids:
            f = H.RUNS / geom / T / "attempts.tsv"
            if f.is_file():
                att.append(pd.read_csv(f, sep="\t").assign(job_id=T, geom=geom))
    A = pd.concat(att, ignore_index=True) if att else pd.DataFrame(columns=["attempt"])
    n_retry = int((A.attempt == 2).sum())
    Ld = L[(L.geom == "dft") & L.job_id.isin(ids)].merge(X[["job_id", "n_atoms"]].astype({"n_atoms": int}), on="job_id")
    gate_rate = {g: float(Ld[g].mean()) for g in GATES if g in Ld}
    # cost
    tim = []
    for T in ids:
        f = H.RUNS / "dft" / T / "timing.tsv"
        if f.is_file():
            tim.append(pd.read_csv(f, sep="\t").assign(job_id=T))
    Tm = pd.concat(tim, ignore_index=True)
    med = (Tm.groupby(["job_id", "calc"]).wall_seconds.sum() / 3600).groupby("calc").median().to_dict()
    b, la = np.polyfit(np.log(Ld.n_atoms), np.log(Ld.coreh_total), 1)
    a = float(np.exp(la))
    allx = X[~X.x_fail.astype(bool)].n_atoms.astype(float)
    est = float((a * allx ** b).sum())
    cost = dict(median_coreh_per_calc=med, median_coreh_per_reaction=float(Ld.coreh_total.median()),
                fit=dict(a=a, b=float(b), model="coreh_total = a * n_atoms^b (log-log least squares, ok pilot rows)"),
                m_estimate_coreh_all_targets=est,
                m_wall_hours={k: est / (NODE_CORES * k) for k in (1, 5, 10)},
                m_wall_note="ideal packing of 48-core jobs; k = number of such jobs running at once (MaxJobs 10)")
    # relation to the label (descriptive)
    lab = {d["job_id"]: d for d in H.targets()}
    rel = {}
    for k, lk in LAB_KEYS.items():
        q = Ld[["job_id", k]].dropna()
        y = np.array([float(lab[T][lk]) for T in q.job_id])
        rel[k] = fit_line(q[k].to_numpy(float), y)
    rel_note = "label = intercept + slope * HF (reaction level, P ok rows, DFT TS)"
    # G1 structure effect (descriptive)
    Lg = L[(L.geom == "g1") & L.job_id.isin(ids)].set_index("job_id")
    g1 = {}
    if len(Lg):
        Dd = L[(L.geom == "dft")].set_index("job_id").loc[Lg.index]
        for k in EDA_KEYS:
            dlt = Lg[k] - Dd[k]
            g1[k] = dict(n=int(len(dlt)), mae=float(dlt.abs().mean()), r=float(np.corrcoef(Lg[k], Dd[k])[0, 1]))
    g1_st = st[(st.geom == "g1") & st.job_id.isin(ids)].status.value_counts().to_dict()
    s1_ok = bool(s1.ok.all())
    decide = dict(ok_rate=n_ok / n, ok_rate_ge_95=n_ok / n >= 0.95,
                  ok_rows_all_gates=bool(len(Ld) and all(Ld[g].all() for g in GATES if not g.endswith("_spec") and g in Ld)),
                  s1_pass=s1_ok)
    decide["pass"] = decide["ok_rate_ge_95"] and decide["ok_rows_all_gates"] and s1_ok
    met = dict(n_sample=n, n_ok=n_ok, failures=reasons, n_retry=n_retry, gate_pass_rate_ok_rows=gate_rate, cost=cost,
               relation_to_label=dict(note=rel_note, by_target=rel), g1=dict(status=g1_st, hf_g1_minus_hf_dft=g1),
               decision=decide)
    H.write_json(H.RES / "P_metrics.json", met)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["# D1-HF Phase P — 파일럿 100반응 보고", "",
          f"파일럿 {n}개 중 {n_ok}개 ok({n_ok / n:.0%}), 재시도 {n_retry}회. 판정(ok ≥ 95 %, ok 행 gate 100 %, S-1 통과): "
          f"**{'통과' if decide['pass'] else '미달'}**. 수치 출처: `results/P_metrics.json`.", "",
          "## 성공과 실패", "", f"- ok {n_ok} / {n}", f"- 실패 이유별: {reasons if reasons else '없음'}",
          f"- 재시도(SlowConv + MaxIter 500): {n_retry}회", "",
          "## gate 통과율 (ok 행)", "", "| gate | 통과율 |", "|---|---|"]
    md += [f"| {g} | {fmt(v)} |" for g, v in gate_rate.items()]
    md += ["", "- 판정 gate는 라벨과 같은 기준(g2·g3 ≤ 0.11 kcal/mol)이다. `*_spec`은 명령서 표의 기준(0.01, 1e-3)으로 본 값이며 "
           "보고만 한다.", "", "## 비용", "", "| 계산 | 코어시간 중앙값 |", "|---|---|"]
    md += [f"| {c} | {fmt(v)} |" for c, v in med.items()]
    md += ["", f"- 반응당 코어시간 중앙값 {fmt(cost['median_coreh_per_reaction'])}; 맞춤 coreh = {a:.3g} × 원자 수^{b:.3g}",
           f"- 본계산(대상 {len(allx)}개 전부) 예상 {est:,.0f} 코어시간; 48코어 job k개를 동시에 돌릴 때 기간 "
           + ", ".join(f"k={k}: {v:.1f} h" for k, v in cost["m_wall_hours"].items()), "",
           "## 정답과의 관계 (설명용; 라벨 = 절편 + 기울기 × HF, 반응 단위)", "",
           "| 타깃 | n | r | 기울기 | 절편 |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['r']:.3f} | {v['slope']:.3f} | {v['intercept']:.3f} |" for k, v in rel.items()]
    md += ["", "## 구조 효과 (설명용; HF@G1 − HF@DFT-TS, EDA 1개)", "", f"- G1 EDA 상태: {g1_st}", "",
           "| 항목 | n | MAE (kcal/mol) | r |", "|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['mae']:.3f} | {v['r']:.3f} |" for k, v in g1.items()]
    md += ["", "## 판정", "", f"- ok 비율 {n_ok / n:.1%} (기준 ≥ 95 %): {decide['ok_rate_ge_95']}",
           f"- ok 행 판정 gate 100 %: {decide['ok_rows_all_gates']}", f"- S-1 통과: {s1_ok}",
           f"- 결과: **{'통과' if decide['pass'] else '미달'}**"]
    H.write_atomic(H.RES / "P_REPORT.md", "\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return 0 if decide["pass"] else 3


A_TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier"]
D1PAIR_RES = H.REPO / "analysis" / "d1_pair" / "results"


def band(m):
    return "<=0.13" if m <= 0.13 else ("<=0.26" if m <= 0.26 else ("0.26-0.6" if m < 0.6 else ">=0.6"))


def cmd_a():
    """Phase A (spec 8; descriptive, not pre-registered): D1PAIR P_main reference pairs (S_folds.csv + F_pairs.csv),
    HF dd = HF(a_rxn) - HF(b_rxn) (DFT-TS HF values), truth = dd_<t>; per target truth = alpha + beta * HF dd fitted on
    fold_T1 != k and applied to fold k (5 folds). Baseline: FU2 GNN-A+G on the same pairs (results/fu2/M_predictions)."""
    S = pd.read_csv(D1PAIR_RES / "S_folds.csv")
    Fp = pd.read_csv(D1PAIR_RES / "F_pairs.csv", usecols=["pair_id"] + [f"dd_{t}" for t in A_TARGETS])
    S = S.merge(Fp, on="pair_id", validate="1:1")
    S = S[S.in_P_main & (S.ref_kind == "reference")].copy()
    L = pd.read_parquet(H.RES / "labels_d1_hf.parquet")
    hf = L[L.geom == "dft"].set_index("rxn_id")
    miss = sorted((set(S.a_rxn) | set(S.b_rxn)) - set(hf.index))
    S = S[S.a_rxn.isin(hf.index) & S.b_rxn.isin(hf.index)]
    G = pd.read_parquet(D1PAIR_RES / "fu2" / "M_predictions.parquet")
    G = G[(G.model == "GNN") & (G.variant == "A+G")].set_index(["pair_id", "target"]).yhat
    rows, summ = [], []
    for t in A_TARGETS:
        d = S[["pair_id", "fold_T1", "pos", "fam", "core"]].assign(
            y=S[f"dd_{t}"].to_numpy(float), x=hf.loc[S.a_rxn, t].to_numpy(float) - hf.loc[S.b_rxn, t].to_numpy(float))
        d["pred"] = np.nan
        fit = {}
        for k in sorted(d.fold_T1.unique()):
            tr, te = d.fold_T1 != k, d.fold_T1 == k
            b, a = np.polyfit(d.x[tr], d.y[tr], 1)
            d.loc[te, "pred"] = a + b * d.x[te]
            fit[int(k)] = (float(a), float(b))
        d["gnn_ag"] = [G.get((p, t), np.nan) for p in d.pair_id]
        para = d.pos == "p"
        e, eg = (d.pred - d.y).abs(), (d.gnn_ag - d.y).abs()
        both = d.gnn_ag.notna()
        r = dict(target=t, n=len(d), n_para=int(para.sum()), mae_all=float(e.mean()), mae_para=float(e[para].mean()),
                 null_mae_all=float(d.y.abs().mean()), null_mae_para=float(d.y[para].abs().mean()),
                 r_hf_vs_truth=float(np.corrcoef(d.x, d.y)[0, 1]), r_pred_vs_truth=float(np.corrcoef(d.pred, d.y)[0, 1]),
                 alpha_mean=float(np.mean([v[0] for v in fit.values()])),
                 beta_mean=float(np.mean([v[1] for v in fit.values()])),
                 fu2_gnn_ag_n=int(both.sum()), fu2_gnn_ag_mae_all=float(eg[both].mean()),
                 fu2_gnn_ag_mae_para=float(eg[both & para].mean()),
                 hf_mae_para_same_pairs=float(e[both & para].mean()))
        r["band_para"] = band(r["mae_para"])
        summ.append(r)
        rows.append(d.assign(target=t))
    A = pd.concat(rows, ignore_index=True)
    Sm = pd.DataFrame(summ)
    H.write_atomic(H.RES / "A_pairs.csv", lambda t: A.to_csv(t, index=False))
    H.write_atomic(H.RES / "A_summary.csv", lambda t: Sm.to_csv(t, index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300})
    fig, axs = plt.subplots(3, 3, figsize=(183 / 25.4, 183 / 25.4), constrained_layout=True)
    for ax, t in zip(axs.ravel(), A_TARGETS):
        q = A[(A.target == t) & (A.pos == "p")]
        lo, hi = float(np.nanmin(np.r_[q.x, q.y])), float(np.nanmax(np.r_[q.x, q.y]))
        ax.plot([lo, hi], [lo, hi], color="#898781", lw=0.6)
        ax.scatter(q.x, q.y, s=3, color="#2a78d6", linewidths=0, alpha=0.6)
        s = Sm.set_index("target").loc[t]
        ax.set_title(f"{t}: r {s.r_hf_vs_truth:.2f}, para MAE {s.mae_para:.2f} ({s.band_para})")
        ax.set_xlabel("HF/6-31G* ΔΔ (kcal/mol)")
        ax.set_ylabel("label ΔΔ (kcal/mol)")
    H.write_atomic(H.RES / "figA_parity.png", lambda p: fig.savefig(p, format="png", bbox_inches="tight"))
    plt.close(fig)
    H.write_json(H.RES / "A_report.json", dict(n_pairs=len(S), missing_rxn=miss[:20], n_missing_rxn=len(miss),
                                               note="descriptive, not pre-registered; bands only, no decision",
                                               baseline="FU2 GNN-A+G yhat (5 seeds mean) from results/fu2/M_predictions"
                                                        ".parquet, same pairs (the pair without atom data is missing)"))
    print(Sm.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["a"]:
        sys.exit(cmd_a())
    if a[:1] == ["p"]:
        sys.exit(cmd_p())
    raise SystemExit(__doc__)
