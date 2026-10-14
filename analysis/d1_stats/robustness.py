"""robustness.py — SPEC_D1_ALL B4: headline 숫자의 강건성 변형 · 코어 클러스터 부트스트랩 CI · dipole 층별 · 지문 BH-FDR · 게이트 B.

python robustness.py LABELS RESULTS_DIR [N_BOOT=1000]
  RESULTS_DIR = run_all.py 출력 폴더 (headline.json 의 기본값과 같은 정의를 여기서 다시 계산한다)
출력 (RESULTS_DIR):
  robustness_headline.csv   metric × {base, incl_stepwise, alt_same, lowest_sibling, no_carbonyl_ylide} + base CI (core bootstrap)
  robustness_by_dipole.csv  metric × dipole (base 정의), 전체와 부호가 다르면 sign_differs=True
  fingerprint_signtest_bh.csv  (side, X, channel): 코어별 중앙값의 부호 검정 → Benjamini–Hochberg q (FDR 0.05)
  gate_b.json               부호가 기본값과 다른 (metric, variant) 목록 — STOP 아님, 결론 문구를 약하게 할 대상
변형 정의 (SPEC B4):
  incl_stepwise   q_stepwise 행을 포함 (q_badref 는 계속 제외)
  alt_same        부모 · 자식의 alt_used 가 같은 짝만 (짝 지표에만)
  lowest_sibling  (cell, regio, face) 안에서 ΔΔE‡ 가 가장 낮은 짝 하나만 (q5q7.py 와 같은 규칙; 짝 지표에만)
  no_carbonyl_ylide  carbonyl ylide 를 뺌 (B8)
부트스트랩은 짝이 아니라 코어(dipole × 골격) 단위로 다시 뽑는다 (같은 코어의 짝은 독립이 아니다). seed 고정.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, pairs_vs_H, sibling_pairs, budget, BUD, CHN

LAB, RD = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEED = 20261014
INT = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]
df0 = load_labels(LAB)


def build(df):
    """pairs + selectivity siblings with the q_bad column of df (variants change q_bad before this)."""
    P = pairs_vs_H(df)
    Pc = P[~P.q_bad].copy()
    R = sibling_pairs(df[df.orient.isin(["a", "b"])], ["cell_id", "face", "stereo_index"], "orient", "a", "b")
    E = sibling_pairs(df[df.face.isin(["exo", "endo"])], ["cell_id", "orient"], "face", "exo", "endo")
    for D in (Pc, R, E):
        if len(D):
            D["strain"] = D.d1_kcal + D.d2_kcal
            D["Eint"] = D[INT].sum(axis=1)
    return Pc, R, E


def pair_metrics(Pc):
    out = {}
    if len(Pc) < 10:
        return out
    y = Pc.barrier_kcal
    out["r_strain_barrier"] = np.corrcoef(Pc.strain, y)[0, 1]
    out["r_eint_barrier"] = np.corrcoef(Pc.Eint, y)[0, 1]
    out["r_dzeta_barrier"] = np.corrcoef(Pc.zeta, y)[0, 1]
    for k, v in budget(Pc).items():
        out[f"share_{k}"] = v
    return out


def sel_metrics(R, E):
    out = {}
    if len(R) >= 10:
        w = np.sign(R.barrier_kcal)
        out["regio_acc_strain"] = (np.sign(R.strain) == w).mean()
        out["regio_acc_Eint"] = (np.sign(R.Eint) == w).mean()
        R2s = R[(R.series == "R2") & R.skeleton.isin(["A", "B", "C"])]          # = run_all.py regio_a_pref_ABC
        for d, g in R2s.groupby("dipole"):
            out[f"regio_a_pref_ABC|{d}"] = (g.barrier_kcal < 0).mean()
    if len(E) >= 10:
        out["exo_pref_frac"] = (E.barrier_kcal < 0).mean()
        out["exo_acc_d2"] = (np.sign(E.d2_kcal) == np.sign(E.barrier_kcal)).mean()
    return out


def all_metrics(Pc, R, E):
    m = pair_metrics(Pc); m.update(sel_metrics(R, E)); return m


FRAC = ("regio_acc", "regio_a_pref", "exo_")       # fractions in [0, 1]: "sign" = which side of 0.5


def side(k, v):
    return np.sign(v - 0.5) if k.startswith(FRAC) else np.sign(v)


# ------------------------------------------------------------------ variants
base_df = df0.copy()
Pc, R, E = build(base_df)
variants = {"base": (Pc, R, E)}
v = df0.copy(); v["q_bad"] = v.q_badref
variants["incl_stepwise"] = build(v)
alt = df0.set_index("job_id").alt_used
Pa = Pc[Pc.job_id.map(alt).values == Pc.parent_job.map(alt).values]
variants["alt_same"] = (Pa, R, E)                      # pair-level variant; selectivity unchanged
c = Pc.copy(); c["key"] = c.cell_id + "|" + c.regio + "|" + c.face.fillna("")
variants["lowest_sibling"] = (c.loc[c.groupby("key").barrier_kcal.idxmin()], R, E)
nc = lambda D: D[D.dipole != "carbonyl ylide"]
variants["no_carbonyl_ylide"] = (nc(Pc), nc(R), nc(E))
vals = {name: all_metrics(*t) for name, t in variants.items()}
pair_only = {"alt_same", "lowest_sibling"}

# ------------------------------------------------------------------ core cluster bootstrap (base)
rng = np.random.default_rng(SEED)
def boot(fn, D, B=NB):
    cores = D.core.unique(); byc = {k: x for k, x in D.groupby("core")}; bs = []
    for _ in range(B):
        bs.append(fn(pd.concat([byc[k] for k in rng.choice(cores, len(cores))])))
    return pd.DataFrame(bs)
bp = boot(pair_metrics, Pc)
bsR = boot(lambda D: sel_metrics(D, E.iloc[:0]), R) if len(R) else pd.DataFrame()
bsE = boot(lambda D: sel_metrics(R.iloc[:0], D), E) if len(E) else pd.DataFrame()
bs = pd.concat([bp, bsR, bsE], axis=1)
rows = []
for k in vals["base"]:
    r = dict(metric=k)
    for name in variants:
        r[name] = vals[name].get(k, np.nan)
    if k in bs:
        r["ci_lo"], r["ci_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
    r["sign_flip_variants"] = ";".join(n for n in variants if n != "base" and not np.isnan(r[n]) and side(k, r[n]) != side(k, r["base"])
                                       and not (n in pair_only and k.startswith(("regio_", "exo_"))))
    rows.append(r)
H = pd.DataFrame(rows); H.to_csv(os.path.join(RD, "robustness_headline.csv"), index=False)

# ------------------------------------------------------------------ by dipole
rows = []
for dp in sorted(df0.dipole.unique()):
    m = all_metrics(Pc[Pc.dipole == dp], R[R.dipole == dp] if len(R) else R, E[E.dipole == dp] if len(E) else E)
    for k, val in m.items():
        b = vals["base"].get(k, np.nan)
        rows.append(dict(metric=k, dipole=dp, value=val, overall=b, sign_differs=bool(side(k, val) != side(k, b))))
pd.DataFrame(rows).to_csv(os.path.join(RD, "robustness_by_dipole.csv"), index=False)

# ------------------------------------------------------------------ fingerprint sign tests with BH FDR
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
CHS = ["d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "barrier_kcal"]
rows = []
for (side, X), g in Pc[Pc.attach == "direct"].groupby(["side", "X"]):
    med = g.groupby("core")[CHS].median()                   # one value per core (cores are the independent units)
    for ch in CHS:
        x = med[ch].values; x = x[x != 0]
        if len(x) < 3:
            continue
        k = int((x > 0).sum())
        rows.append(dict(side=side, X=X, channel=CHN[ch], n_core=len(x), n_pos=k, median=float(np.median(med[ch])),
                         p=binomtest(k, len(x), 0.5).pvalue))
F = pd.DataFrame(rows)
if len(F):
    F["q_bh"] = multipletests(F.p, alpha=0.05, method="fdr_bh")[1]
    F["sig_fdr05"] = F.q_bh < 0.05
F.to_csv(os.path.join(RD, "fingerprint_signtest_bh.csv"), index=False)

flips = [dict(metric=r.metric, variants=r.sign_flip_variants) for r in H.itertuples() if r.sign_flip_variants]
gb = dict(n_metrics=len(H), n_sign_flips=len(flips), sign_flips=flips,
          n_fingerprint_tests=len(F), n_fingerprint_sig_fdr05=int(F.sig_fdr05.sum()) if len(F) else 0,
          n_boot=NB, seed=SEED, note="게이트 B: 부호가 바뀌는 항목은 STOP 하지 않고 보고 (결론 문구를 약하게 할 대상)")
json.dump(gb, open(os.path.join(RD, "gate_b.json"), "w"), indent=1, ensure_ascii=False)
print(H.round(3).to_string(index=False)); print(json.dumps(gb, indent=1, ensure_ascii=False))
