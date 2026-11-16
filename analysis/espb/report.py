#!/usr/bin/env python3
"""ESPB Phase R (docs/specs/ESPB.md 10): results/RUN_REPORT_ESPB.md (Korean) and the manifest results/espb.json.

    python report.py [--pushed <sha>]       PYENV=reactot (jobs/py.slurm, submitted from analysis/espb), after figures.py

Reads only results/ (S0, COPY, P, prereg, C, E reports and tables, figures) and the unit json files under $EB/units
(gates A-1 .. A-4 / B-1 .. B-4, wall time). --pushed records the HEAD that was pushed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eb_common as C  # noqa: E402
import prereg  # noqa: E402

FIGS = {"fig1_primary": "주 비교: P_main에서 타깃별 GNN3와 ESPB의 MAE(core bootstrap 95 % 구간), all과 para",
        "fig2_parity": "P_main barrier ΔΔ의 DFT 대 예측(GNN3, ESPB), y = x와 ±1 kcal/mol 띠",
        "fig3_candidates": "타깃별 후보 10개의 P_main MAE와 inner 점수 순위, GNN3 선과 0 예측 선",
        "fig4_positions": "P_main 위치(p, m, o, direct)별 MAE, GNN3와 ESPB"}


def n(v) -> str:
    return f"{int(v):,}"


def f3(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}"


def units():
    out = []
    for arm, d in (("EA", C.UNITS_EA), ("EB", C.UNITS_EB)):
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            out.append(dict(arm=arm, unit=f.stem, model=r["model"], target=r["target"], fold=r["fold"],
                            seconds=r.get("seconds"), inner=r.get("inner_cv_mae"),
                            ok=bool(r.get("gate_A2_finite", r.get("gate_B2_finite", False))
                                    and r.get("gate_A4_seeds", r.get("gate_B4_seeds", False))
                                    and r.get("gate_M1", True)),
                            m1=(r.get("antisym") or {}).get("max_sum")))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pushed", default=None)
    args = ap.parse_args()
    R = C.RES
    s0, cp, p, c, e = (C.read_json(R / n_) for n_ in ("S0_start.json", "COPY.json", "P_report.json", "C_report.json",
                                                      "E_report.json"))
    pr = C.read_json(C.PREREG_JSON)
    U = units()
    ea, eb = U[U.arm == "EA"], U[U.arm == "EB"]
    prim = pd.read_csv(R / "E_primary.csv")
    sec = pd.read_csv(R / "E_secondary.csv")
    met = pd.read_csv(R / "E_metrics.csv")
    rank = pd.read_csv(R / "E_rank.csv")
    sel = pd.read_csv(R / "C_selection.csv")
    nat = pd.read_csv(R / "E_native.csv")
    pre = pd.read_csv(R / "E_preml.csv")
    figs_ok = all((R / "figures" / f"{f}.{x}").is_file() for f in FIGS for x in ("pdf", "png", "csv"))
    head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    try:
        prereg.gate()
        g_ok = True
    except SystemExit:
        g_ok = False
    L = ["# ESPB 실행 보고 — Espley-type baseline을 D1 짝 ΔΔ task에 맞춰 학습하고 GNN3R2와 비교", "",
         f"- 지시서 `docs/specs/ESPB.md` · 브랜치 `espb` · 결과 commit `{head[:8]}`"
         + (f" · push한 HEAD `{args.pushed}`" if args.pushed else ""),
         f"- scratch `{C.EB}` (units, NN 작업 폴더), 결과 `analysis/espb/results/`, manifest `results/espb.json`, "
         "사전 등록 `results/PREREG_ESPB.md`",
         "- 대상: GNN3R2 데이터셋(반응 8,594, 짝 8,344, test 짝 8,175), 같은 core 5-fold 분할, 같은 채점 코드(`gnn3.metrics`).", "",
         "## 1. 단계별 판정", "", "| 단계 | 판정 | 비고 |", "|---|---|---|",
         f"| 1 시작 조건 | {'통과' if s0['all_pass'] else 'STOP'} | 1-1 … 1-9, 상수 확인 2-3 |",
         f"| 2 복사 | 통과 | 사본 {len(cp['files'])}개, diff `COPY_DIFF.patch` |",
         f"| P 데이터 표 | {'통과' if p['all_pass'] else 'STOP'} | P-1 … P-9 |",
         f"| G 사전 등록 | {'통과' if g_ok else 'STOP'} | commit 뒤 학습, 등록 sha256 {len(pr['sha256'])}개 = 현재 파일 |",
         f"| A 학습 (E-A) | {'통과' if len(ea) == 125 and ea.ok.all() else 'STOP'} | unit {len(ea)}/125, A-1 … A-4 |",
         f"| B 학습 (E-B) | {'통과' if len(eb) == 100 and eb.ok.all() else 'STOP'} | unit {len(eb)}/100, B-1 … B-4, M-1 최대 "
         f"{eb.m1.max():.1e} kcal/mol |",
         f"| C 모으기·선택 | {'통과' if c['all_pass'] else 'STOP'} | C-1 … C-4 |",
         f"| E 평가·비교 | {'통과' if e['all_pass'] else 'STOP'} | E-0 (GNN3R2 metrics 재현, 최대 차 "
         f"{max(e['gates']['E-0']['max_abs'].values()):.1e}) |",
         f"| F 그림 | {'통과' if figs_ok else 'STOP'} | 4개 × PDF·PNG·CSV |", ""]
    # 2. primary
    L += ["## 2. 주 비교 (사전 등록, P_main · 위치 all)", "",
          "| 타깃 | MAE GNN3 | MAE ESPB (고른 후보) | d = ESPB − GNN3 (95 % CI) | p | Holm p | 해석 |", "|---|---|---|---|---|---|---|"]
    for r in prim.itertuples():
        L.append(f"| {r.target} | {f3(r.mae_gnn3)} | {f3(r.mae_espb)} ({r.espb_candidate}) | {r.d:+.3f} "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p:.3g} | {r.p_holm:.3g} | {r.reading} |")
    L += ["", f"- n = {n(prim.n.iloc[0])} 기준 짝, core {prim.n_cores.iloc[0]}개. core bootstrap {C.BOOT_N}번, seed "
          f"{C.BOOT_SEED}. d < 0이면 ESPB가 낮다.", ""]
    # 3. secondary
    L += ["## 3. 보조 비교 (설명용)", "", "**ESPB 대 GNN3** (Holm은 각 행 묶음의 타깃 4개 안에서)", "",
          "| 집합 | 부분 | 타깃 | n | MAE GNN3 | MAE ESPB | d (95 % CI) | Holm p |", "|---|---|---|---|---|---|---|---|"]
    s1 = sec[sec.comparison == "ESPB vs GNN3"]
    for r in s1.itertuples():
        L.append(f"| {r.eval_set} | {r.subset} | {r.target} | {n(r.n)} | {f3(r.mae_versus)} | {f3(r.mae_model)} | "
                 f"{r.d:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p_holm:.3g} |")
    z = sec[(sec.comparison == "zero predictor") & sec.model.isin(["GNN3", "ESPB"])]
    L += ["", "**0을 예측하는 기준 대비** (P_main; 개선 = MAE(0) − MAE)", "",
          "| 부분 | 타깃 | MAE(0) | GNN3 MAE · 개선 | ESPB MAE · 개선 |", "|---|---|---|---|---|"]
    for (sub, t), g in z.groupby(["subset", "target"], sort=False):
        gg = g.set_index("model")
        L.append(f"| {sub} | {t} | {f3(gg.mae_versus.iloc[0])} | {f3(gg.loc['GNN3', 'mae_model'])} · "
                 f"{f3(gg.loc['GNN3', 'improvement'])} | {f3(gg.loc['ESPB', 'mae_model'])} · {f3(gg.loc['ESPB', 'improvement'])} |")
    sl = sec[(sec.comparison == "shrinkage slope") & sec.model.isin(["GNN3", "ESPB"])]
    L += ["", "**수축 기울기** (P_main barrier, y에 대한 yhat의 OLS 기울기; 1이면 수축 없음)", "",
          "| series | GNN3 | ESPB |", "|---|---|---|"]
    for sub, g in sl.groupby("subset", sort=False):
        gg = g.set_index("model")
        L.append(f"| {sub} | {f3(gg.loc['GNN3', 'slope'])} | {f3(gg.loc['ESPB', 'slope'])} |")
    rk = rank[rank.model.isin(["GNN3", "ESPB"])]
    L += ["", "**계열 안 장벽 순위** (`M.ranking_table`, P_sens_ref)", "",
          "| 범위 | 모델 | 계열 수 | top-1 (무작위) | top-3 (무작위) | Spearman 중앙값 |", "|---|---|---|---|---|---|"]
    for r in rk.itertuples():
        L.append(f"| {r.scope} | {r.model} | {r.n_families} | {f3(r.lowest_top1)} ({f3(r.lowest_top1_random)}) | "
                 f"{f3(r.lowest_top3)} ({f3(r.lowest_top3_random)}) | {f3(r.spearman_median)} |")
    sa = met[(met.model.isin(["GNN3", "ESPB"])) & (met.eval_set == "P_main") & (met.pos.isin(["all", "p"]))]
    L += ["", "**부호 정확도** (|y| ≥ 1 kcal/mol 짝, P_main)", "", "| 위치 | 타깃 | GNN3 | ESPB |", "|---|---|---|---|"]
    for (pos, t), g in sa.groupby(["pos", "target"], sort=False):
        gg = g.set_index("model")
        L.append(f"| {pos} | {t} | {f3(gg.loc['GNN3', 'sign_acc'])} | {f3(gg.loc['ESPB', 'sign_acc'])} |")
    # 4. candidates
    L += ["", "## 4. 후보 10개", "", "| 타깃 | 후보 | inner 점수 | 순위 | 선택 | test MAE P_main all | para |",
          "|---|---|---|---|---|---|---|"]
    mm_ = met[(met.eval_set == "P_main")].set_index(["model", "target", "pos"])
    for r in sel.itertuples():
        a = mm_.loc[(r.candidate, r.target, "all"), "mae"]
        b = mm_.loc[(r.candidate, r.target, "p"), "mae"]
        L.append(f"| {r.target} | {r.candidate} | {f3(r.score)} | {r.rank} | {'✔' if r.chosen else ''} | {f3(a)} | {f3(b)} |")
    # 5. Espley-side metrics
    L += ["", "## 5. Espley 쪽 지표", "", "**E-A 반응별 절대 test MAE** (fold별 MAE의 평균; % = MAE / test 범위)", "",
          "| 모델 | " + " | ".join(["d1", "d2", "inter", "solv", "Int", "barrier"]) + " |", "|---" * 7 + "|"]
    for m, g in nat.groupby("model", sort=False):
        gg = g.set_index("target")
        L.append(f"| {m} | " + " | ".join(f"{f3(gg.loc[t, 'mae'])} ({gg.loc[t, 'mae_pct_range']:.1f} %)"
                                         for t in ["d1", "d2", "inter", "solv", "Int", "barrier"]) + " |")
    L += ["", "- GNN3는 짝 모델이라 반응별 절대값을 내지 않아 이 표에 없다.", "",
          "**ML 이전 xTB 대 DFT** (부호 규약 P-8: xtb_e_barrier = xtb_sum_distortion + xtb_interaction)", "",
          "| 수준 | 타깃 | xTB | n | MAE | 평균 부호 오차 | r |", "|---|---|---|---|---|---|---|"]
    for r in pre.itertuples():
        L.append(f"| {r.level} | {r.target} | `{r.xtb}` | {n(r.n)} | {f3(r.mae)} | {f3(r.mean_signed)} | {f3(r.r)} |")
    # 6. changes, 7. figures, 8. decisions / gates
    L += ["", "## 6. Espley 원본에서 바꾼 점과 하지 않은 것", "", "| # | Espley 2024 원본 (ds3) | ESPB | 이유 |", "|---|---|---|---|"]
    L += [f"| {x['id']} | {x['espley']} | {x['espb']} | {x['why']} |" for x in pr["changes"]]
    L += ["", "| 하지 않은 것 | 이유 |", "|---|---|"] + [f"| {x['item']} | {x['why']} |" for x in pr["excluded"]]
    L += ["", "## 7. 그림 (`results/figures/`)", ""] + [f"- `{f}` — {d}" for f, d in FIGS.items()]
    pg = p["gates"]
    L += ["", "## 8. 결정 · 편차 · gate 값", "", "- 결정·편차 전체: `results/DECISIONS.md`.",
          f"- P-7 반응별 라벨 닫힘 최대 |barrier − 9개 합| = {pg['P-7']['max_abs']:.3f} kcal/mol.",
          f"- P-8 xTB 규약: |xtb_e_barrier − (xtb_sum_distortion + xtb_interaction)| 최대 "
          f"{pg['P-8']['max_abs_barrier_minus_dist_plus_int']:.2e}, |xtb_sum_distortion − (dipole + dipolarophile)| 최대 "
          f"{pg['P-8']['max_abs_sum_distortion_minus_parts']:.2e}.",
          f"- P-9 효과 없는 hyperparameter: Ridge tol {pg['P-9']['ridge_tol_max_abs']:.1e}, RBF SVR coef0·degree "
          f"{pg['P-9']['svr_coef0_degree_max_abs']:.1e} (≤ 1e-10).",
          f"- 학습 시간: E-A {ea.seconds.sum() / 3600:.1f} CPU-job-h(unit 합), E-B {eb.seconds.sum() / 3600:.1f} h.", ""]
    C.write_atomic(R / "RUN_REPORT_ESPB.md", "\n".join(L) + "\n")
    files = sorted(q for q in R.rglob("*") if q.is_file() and q.name != "espb.json")
    man = dict(spec="docs/specs/ESPB.md", branch="espb", results_commit=head, pushed=args.pushed,
               written=time.strftime("%Y-%m-%d %H:%M:%S"), roots=dict(espb=str(C.EB), results="analysis/espb/results"),
               inputs={str(q): C.sha256(q) for q in (C.PAIRS, C.S_FOLDS, C.S_SPLITS, C.LABEL_REPORT, C.GNN3_METRICS, C.X182,
                                                    C.LABELS_CR2) + tuple(C.SPLIT_DIR / f"T1__f{k}.json" for k in range(5))
                       + tuple(C.GNN3_CV / f"gnn3_fold{k}.parquet" for k in range(5))},
               code={str(q.relative_to(C.REPO)): C.sha256(q) for q in sorted(C.HERE.glob("*.py")) + sorted((C.HERE / "jobs").glob("*"))},
               results={str(q.relative_to(C.REPO)): C.sha256(q) for q in files},
               versions=dict(env_espley=s0["checks"]["1-8"]["env_espley"]["versions"]),
               slurm_jobs=dict(S0=s0.get("slurm_job"), P=p.get("slurm_job"), C=c.get("slurm_job"), E=e.get("slurm_job"),
                               report=os.environ.get("SLURM_JOB_ID")),
               chosen=c["chosen"], primary=prim.to_dict("records"))
    C.write_json(R / "espb.json", man)
    print(f"wrote {R / 'RUN_REPORT_ESPB.md'} ({len(L)} lines) and {R / 'espb.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
