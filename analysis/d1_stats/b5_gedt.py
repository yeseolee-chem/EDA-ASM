"""b5_gedt.py — SPEC_D1_ALL B5 GEDT (Domingo 핵심 변수): 기존 eda.out 의 복합체 전하로 GEDT 를 만들고 판정한다 (새 QM 없음).

python b5_gedt.py LABELS ARCHIVE_SCAN_CSV RESULTS_DIR [FEAT_EXT_PARQUET] [N_BOOT=1000]
  ARCHIVE_SCAN_CSV  d1_archive_scan.py 출력 (ok 행: mull/loew_frag0 · tag1 합 · a_idx_meta · a_idx_tag)
  FEAT_EXT_PARQUET  Phase C xtb_features_ext_d1_g1.parquet (job_id, gedt = xTB B4 블록, G1 기하). 없으면 gedt_xtb 는 빈칸
GEDT 정의: 복합체(TS) 원자 전하를 조각 1(= eda.inp 의 (1) 원자 = role1 = dipole, 조각 분할은 라벨과 같은
fragmenter.partition — a_idx_meta == a_idx_tag 로 확인)에 대해 더한 값 − charge1.
  양수 = 전자가 dipole → dipolarophile 로 (FEDF), 음수 = dipolarophile → dipole (REDF). xTB 의 b_ct · gedt 와 같은 부호.
  Mulliken · Löwdin (B3LYP-D3(BJ)/def2-TZVP, SMD 물, DFT TS). Domingo 는 NPA 전하를 쓴다 — 방법이 다르다는 것을 함께 적는다.
출력 (RESULTS_DIR):
  gedt_d1.csv          job_id, gedt_mulliken, gedt_loewdin, gedt_xtb, 자체 검사 열 (frag0 − tag1 합, a_idx 일치)
  gedt_partial.csv     짝(자식 − H 부모, q_bad 제외) 에서 dipole · 쪽별: y = ΔΔE‡ 또는 ΔΔE‡(용매화 제외 = − ΔΔCPCM − ΔΔCDS),
                       g = |ΔGEDT| (SPEC 의 주 변수) 또는 Δ|GEDT| (|GEDT 자식| − |GEDT 부모|, 보조), s = ΔΔ변형 (d1 + d2).
                       r(y,g), r(y,s), 부분상관 r(y,g | s), r(y,s | g), R²(g), R²(s), 코어 부트스트랩 95 % CI
  gedt_sign_slopes.csv dipole 별, 자식 GEDT 부호(FEDF / REDF) 별 ΔΔE‡ ~ Δ|GEDT| 기울기 (CI), 비 = FEDF / REDF
  gedt.json            검사 요약, 판정 초안 (GEDT 가 변형보다 장벽을 더 잘 설명하는 dipole 수), 방법 차이 메모
판정 문구는 사람이 STATS_REPORT 에 쓴다 (세 문구 중 하나 + 제약: HN3 대리, 범함수, 용매). 이 스크립트는 숫자만 낸다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, pairs_vs_H  # noqa: E402

LAB, SCAN, RD = sys.argv[1:4]
FEXT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
NB = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
SEED = 20261014
df = load_labels(LAB)
A = pd.read_csv(SCAN, dtype={"a_idx_meta": str, "a_idx_tag": str})
A = A[A["class"] == "ok"].set_index("job_id")
q1 = df.set_index("job_id").charge1.astype(float)
G = pd.DataFrame(index=df.job_id)
G["gedt_mulliken"] = A.mull_frag0.reindex(G.index) - q1
G["gedt_loewdin"] = A.loew_frag0.reindex(G.index) - q1
G["chk_mull_frag0_minus_tag1"] = (A.mull_frag0 - A.mull_tag1_sum).reindex(G.index)
G["chk_loew_frag0_minus_tag1"] = (A.loew_frag0 - A.loew_tag1_sum).reindex(G.index)
G["chk_a_idx_meta_eq_tag"] = (A.a_idx_meta == A.a_idx_tag).reindex(G.index)
G["gedt_xtb"] = np.nan
if FEXT and os.path.isfile(FEXT):
    X = pd.read_parquet(FEXT, columns=["job_id", "gedt", "status_B4"])
    X = X[X.status_B4 == "ok"].set_index("job_id").gedt
    G["gedt_xtb"] = X.reindex(G.index)
G.reset_index().rename(columns={"index": "job_id"}).to_csv(os.path.join(RD, "gedt_d1.csv"), index=False)
n_at = A.n_atoms.reindex(G.index).fillna(0)
tol = 1e-4 + 5e-7 * n_at                                    # 6-decimal atomic charges summed
chk = dict(n_rows=len(G), n_with_charges=int(G.gedt_mulliken.notna().sum()), n_xtb=int(G.gedt_xtb.notna().sum()),
           n_frag0_tag1_mismatch=int(((G.chk_mull_frag0_minus_tag1.abs() > tol) | (G.chk_loew_frag0_minus_tag1.abs() > tol)).sum()),
           n_a_idx_meta_ne_tag=int((G.chk_a_idx_meta_eq_tag == False).sum()),          # noqa: E712
           max_abs_frag0_minus_tag1=float(np.nanmax(np.abs(G[["chk_mull_frag0_minus_tag1", "chk_loew_frag0_minus_tag1"]].to_numpy(float)))) if len(A) else None)

# ------------------------------------------------------------------ pairs
P = pairs_vs_H(df)
P = P[~P.q_bad].copy()
for m in ("mulliken", "loewdin", "xtb"):
    g = G[f"gedt_{m}"]
    c, p = P.job_id.map(g), P.parent_job.map(g)
    P[f"gc_{m}"] = c
    P[f"dabs_{m}"] = c.abs() - p.abs()                       # Δ|GEDT|
    P[f"absd_{m}"] = (c - p).abs()                           # |ΔGEDT|
P["strain"] = P.d1_kcal + P.d2_kcal
P["y_all"] = P.barrier_kcal
P["y_nosolv"] = P.barrier_kcal - P.cpcm_dft - P.cds_dft
rng = np.random.default_rng(SEED)


def stats(D, y, g, s):
    d = D[[y, g, s]].dropna()
    if len(d) < 10 or d[g].std() == 0 or d[s].std() == 0:
        return None
    r = d.corr().to_numpy()
    ryg, rys, rgs = r[0, 1], r[0, 2], r[1, 2]
    den = lambda a: np.sqrt(max(1e-12, 1 - a * a))
    return dict(n=len(d), r_y_g=ryg, r_y_s=rys, r_g_s=rgs, pr_y_g__s=(ryg - rys * rgs) / (den(rys) * den(rgs)),
                pr_y_s__g=(rys - ryg * rgs) / (den(ryg) * den(rgs)), R2_g=ryg ** 2, R2_s=rys ** 2)


def boot(D, fn, keys):
    cores = D.core.unique(); byc = {k: x for k, x in D.groupby("core")}; bs = []
    if len(cores) < 2:
        return {}
    for _ in range(NB):
        v = fn(pd.concat([byc[k] for k in rng.choice(cores, len(cores))]))
        if v is not None:
            bs.append([v[k] for k in keys])
    if not bs:
        return {}
    lo, hi = np.nanpercentile(np.array(bs, float), [2.5, 97.5], axis=0)
    return {f"{k}_ci": [float(a), float(b)] for k, a, b in zip(keys, lo, hi)}


rows = []
subsets = [("all", "all", P)]
subsets += [(dp, "all", g) for dp, g in P.groupby("dipole")]
subsets += [(dp, sd, g) for (dp, sd), g in P.groupby(["dipole", "side"])]
subsets += [("azide", "HN3 + vinyl-X (R2, 골격 A)", P[(P.dipole == "azide") & (P.side == "dipolarophile") & (P.skeleton == "A")])]
KEYS = ["pr_y_g__s", "pr_y_s__g", "R2_g", "R2_s"]
for m in ("mulliken", "loewdin", "xtb"):
    for gname in ("dabs", "absd"):
        for yname in ("y_all", "y_nosolv"):
            for dp, sd, D in subsets:
                st = stats(D, yname, f"{gname}_{m}", "strain")
                if st is None:
                    continue
                st.update(boot(D, lambda x: stats(x, yname, f"{gname}_{m}", "strain"), KEYS))
                rows.append(dict(charges=m, gedt_term={"dabs": "Δ|GEDT|", "absd": "|ΔGEDT|"}[gname],
                                 barrier={"y_all": "ΔΔE‡", "y_nosolv": "ΔΔE‡ − ΔΔ(CPCM+CDS)"}[yname],
                                 dipole=dp, side=sd, **st))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(RD, "gedt_partial.csv"), index=False)

# ------------------------------------------------------------------ sign-dependent slopes (FEDF vs REDF)
def slope(D, y, g):
    d = D[[y, g]].dropna()
    if len(d) < 10 or d[g].var() == 0:
        return None
    return dict(n=len(d), slope=float(np.polyfit(d[g], d[y], 1)[0]))


srows = []
for m in ("mulliken", "loewdin", "xtb"):
    for yname in ("y_all", "y_nosolv"):
        for dp, D in [("all", P)] + list(P.groupby("dipole")):
            out = dict(charges=m, barrier=yname, dipole=dp)
            for lab, mask in (("FEDF", D[f"gc_{m}"] > 0), ("REDF", D[f"gc_{m}"] < 0)):
                st = slope(D[mask], yname, f"dabs_{m}")
                if st is None:
                    continue
                out[f"n_{lab}"], out[f"slope_{lab}"] = st["n"], st["slope"]
                ci = boot(D[mask], lambda x: slope(x, yname, f"dabs_{m}"), ["slope"])
                if ci:
                    out[f"slope_{lab}_ci"] = ci["slope_ci"]
            if "slope_FEDF" in out and "slope_REDF" in out and out["slope_REDF"] != 0:
                out["ratio_FEDF_over_REDF"] = out["slope_FEDF"] / out["slope_REDF"]
            srows.append(out)
pd.DataFrame(srows).to_csv(os.path.join(RD, "gedt_sign_slopes.csv"), index=False)

sub = lambda term: R[(R.charges == "mulliken") & (R.gedt_term == term) & (R.barrier == "ΔΔE‡") & (R.side == "all") & (R.dipole != "all")] if len(R) else R
prim, sec = sub("|ΔGEDT|"), sub("Δ|GEDT|")                   # SPEC B5 (1): ΔΔE‡ ~ |ΔGEDT| is the primary term
better = prim[prim.R2_g > prim.R2_s].dipole.tolist() if len(prim) else []
summ = dict(checks=chk, n_pairs_clean=len(P), n_pairs_with_gedt=int(P.dabs_mulliken.notna().sum()),
            primary="Mulliken, |ΔGEDT|, ΔΔE‡, dipole 별 (side 전체); 보조 Δ|GEDT|",
            dipoles_gedt_R2_gt_strain_R2=better, n_dipoles=int(prim.dipole.nunique()) if len(prim) else 0,
            dipoles_gedt_R2_gt_strain_R2_secondary=sec[sec.R2_g > sec.R2_s].dipole.tolist() if len(sec) else [],
            fedf_frac_children=float((P.gc_mulliken > 0).mean()) if P.gc_mulliken.notna().any() else None,
            method_note="DFT Mulliken / Löwdin 원자 전하 (SMD 물, 실제 TS). Domingo 2023 은 NPA (기체상). xTB GEDT 는 G1 기하의 GFN2 Mulliken.",
            n_boot=NB, seed=SEED)
json.dump(summ, open(os.path.join(RD, "gedt.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
if len(prim):
    print(prim[["dipole", "n", "r_y_g", "r_y_s", "pr_y_g__s", "pr_y_s__g", "R2_g", "R2_s"]].round(3).to_string(index=False))
