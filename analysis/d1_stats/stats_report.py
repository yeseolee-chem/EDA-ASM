"""stats_report.py — results_final/ 의 표와 출력에서 STATS_REPORT.md 를 만든다 (SPEC_D1_ALL §7 · §8 순서). 숫자는 파일에서 옮기기만 한다.

python stats_report.py RESULTS_DIR OUT_MD
없는 파일은 "없음(단계 미실행)"으로 적는다. 해석 문구(B5 판정 세 문구 중 하나, B2 MNAR 결론 등)는 사람이 쓰는 자리로 비워 둔다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
L = []


def p(*a):
    L.extend(a)


def jl(name):
    f = RD / name
    return json.load(open(f)) if f.is_file() else None


def csv(name, **kw):
    f = RD / name
    return pd.read_csv(f, **kw) if f.is_file() else None


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{pre}[{i}]."))
    else:
        out[pre[:-1]] = d
    return out


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.3f}"
    return str(v).replace("|", "\\|")                     # cell text such as |ΔGEDT| or ctx keys


def table(df, n=60, cols=None):
    if df is None or not len(df):
        return ["(표 없음)"]
    d = df[cols] if cols else df
    if len(d) > n:
        d = d.head(n)
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    if len(df) > n:
        out.append(f"\n({len(df)}행 중 {n}행만; 전체는 파일)")
    return out


def txt(name, n=60):
    f = RD / name
    if not f.is_file():
        return ["(출력 없음: 단계 미실행)"]
    lines = f.read_text(errors="replace").splitlines()
    return ["```"] + lines[:n] + (["… (생략, 전체는 " + name + ")"] if len(lines) > n else []) + ["```"]


p("# STATS_REPORT — D1 확정 데이터 통계 (SPEC_D1_ALL Phase B)", "",
  f"- 결과 폴더: `{RD}` · 생성: stats_report.py (숫자는 파일에서 옮김)", "")
p("## 1. 입력과 게이트", "")
f = RD / "INPUTS.md"
p(*(f.read_text().splitlines() if f.is_file() else ["(INPUTS.md 없음)"]), "")
ga = jl("gate_a.json")
p(f"- 게이트 A: {'통과' if ga and ga.get('passed') else '실패/미실행'}; " + (json.dumps({k: ga[k] for k in ('n_rows', 'exclusions', 'closure_max', 'orient_counts', 'large_ddbarrier_pairs', 'large_pairs_alt_used_differs') if k in ga}, ensure_ascii=False) if ga else ""), "")

p("## 2. Headline: 중간 파일 → 확정, 강건성 변형, 코어 부트스트랩 CI", "")
hf, hi = jl("headline.json"), json.load(open(HERE / "reference_interim" / "headline.json"))
if hf:
    a, b = flat(hi), flat(hf)
    rows = [dict(key=k, interim=a.get(k), final=b.get(k)) for k in sorted(set(a) | set(b))]
    p(*table(pd.DataFrame(rows), n=200), "")
rb = csv("robustness_headline.csv")
p("### 강건성 (B4: 기본 / 변형별 / CI)", "", *table(rb, n=100), "")
gb = jl("gate_b.json")
p(f"- 게이트 B: 부호가 바뀐 항목 {gb['n_sign_flips'] if gb else '미실행'} — " + (json.dumps(gb.get("sign_flips"), ensure_ascii=False) if gb else ""),
  f"- 지문 부호 검정: {gb['n_fingerprint_tests'] if gb else ''}개 중 BH-FDR 0.05 유의 {gb['n_fingerprint_sig_fdr05'] if gb else ''}", "")
bd = csv("robustness_by_dipole.csv")
if bd is not None:
    p("### dipole 별 (전체와 부호가 다른 칸만)", "", *table(bd[bd.sign_differs.astype(str) == "True"], n=80), "")

p("## 3. B1b: dipole 별 E_int 상관 · 저랭크 CI", "")
p(*table(csv("q5_dipole_eint.csv"), n=60), "", *txt("b1b_q5q7.txt", 80), "")
cb = csv("ci_lowrank_boot.csv")
if cb is not None:
    num = cb.select_dtypes("number")
    grp = [c for c in cb.columns if c not in num.columns]
    s = cb.groupby(grp)[list(num.columns)].quantile([0.025, 0.5, 0.975]).unstack() if grp else num.quantile([0.025, 0.5, 0.975])
    p("저랭크 CV R² 코어 부트스트랩 (2.5 / 50 / 97.5 %):", "", "```", s.round(3).to_string(), "```", "")

p("## 4. B2: 결측 원인", "")
p(*table(csv("status_by_dipole.csv"), n=40), "")
ml = csv("missing_logit.csv")
if ml is not None:
    num = [c for c in ml.columns if ml[c].dtype.kind == "f"]
    key = "coef" if "coef" in ml.columns else (num[0] if num else None)
    if key:
        ml = ml.reindex(ml[key].abs().sort_values(ascending=False).index)
    p("로지스틱 계수 상위:", "", *table(ml, n=25), "")
p("지목 칸 (azomethine ylide × 전자 부족, carbonyl ylide R2):", "", *table(csv("missing_cause_cells.csv"), n=80), "")
mc = jl("missing_cause.json")
p("```", json.dumps(mc, indent=1, ensure_ascii=False) if mc else "(missing_cause.json 없음)", "```", "",
  "- 결론 (사람이 쓴다): 지목 칸의 no_ts 가 고개 없음(barrierless)이면 MNAR — Methods 에 실패율과 함께.", "")

p("## 5. B3: 문헌 σ Hammett", "")
hm = csv("hammett.csv")
if hm is None:
    p("- σ 표(sigma_hansch1991.csv)를 받지 못해 보류 (STOP 아님).", "")
else:
    p(*table(hm, n=60), "", *table(csv("sigma_vs_rank2.csv"), n=20), "")

p("## 6. B5: Houk vs Domingo", "")
p(*table(csv("q2_azide_table.csv"), n=60), "", *txt("b5_q2.txt", 60), "")
gj = jl("gedt.json")
p("GEDT 검사 · 요약:", "", "```", json.dumps(gj, indent=1, ensure_ascii=False) if gj else "(gedt.json 없음)", "```", "")
gp = csv("gedt_partial.csv")
if gp is not None:
    for term, what in (("|ΔGEDT|", "주 변수 (SPEC B5)"), ("Δ|GEDT|", "보조")):
        prim = gp[(gp.charges == "mulliken") & (gp.gedt_term == term) & (gp.side.isin(["all", "HN3 + vinyl-X (R2, 골격 A)"]))]
        p(f"부분상관 (Mulliken, {term} = {what}; ΔΔE‡ 와 용매화 제외):", "",
          *table(prim, n=40, cols=[c for c in ["barrier", "dipole", "side", "n", "r_y_g", "r_y_s", "pr_y_g__s", "pr_y_s__g", "R2_g", "R2_s", "pr_y_g__s_ci", "pr_y_s__g_ci"] if c in prim.columns]), "")
p("GEDT 부호별 기울기:", "", *table(csv("gedt_sign_slopes.csv"), n=40), "")
asy = RD / "inflection" / "asymmetry_slopes.csv"
p("σ 비대칭 (inflection.py):", "", *(table(pd.read_csv(asy), n=40) if asy.is_file() else ["- σ 표가 없어 보류"]), "",
  "- 판정 (사람이 쓴다; 셋 중 하나 + 제약 HN3 대리 · B3LYP · 물 SMD): \"D1(물)은 Houk 쪽\" / \"용매화 제외 시 Domingo 쪽\" / \"GEDT가 변형보다 장벽을 더 잘 설명함/못함\"", "")

p("## 7. B6: σ 순서 채널 변곡점", "")
isum = RD / "inflection" / "inflection_summary.csv"
p(*(table(pd.read_csv(isum), n=40) if isum.is_file() else ["- σ 표가 없어 보류"]), "", *txt("b6_inflection.txt", 40), "")

p("## 8. B7 메틸 · cis/trans, B8 carbonyl ylide 민감도", "")
p(*table(csv("q5_methyl.csv"), n=80), "", *txt("b7_methyl.txt", 40), "")
p(*table(csv("q4_sensitivity.csv"), n=40), "", *txt("b8_missing.txt", 60), "")

p("## 9. 한계와 미해결 (사람이 쓴다)", "")
tmp = OUT.with_name(OUT.name + ".tmp")
tmp.write_text("\n".join(map(str, L)) + "\n", encoding="utf-8")
os.replace(tmp, OUT)
print(f"wrote {OUT} ({len(L)} lines)")
