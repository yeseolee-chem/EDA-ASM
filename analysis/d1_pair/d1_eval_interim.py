#!/usr/bin/env python3
"""d1_eval_interim.py — PROVISIONAL T1 evaluation without SVR (user request 2026-10-20, while the SVR units still run).

  python d1_eval_interim.py      # results/interim_T1_noSVR/*  (exit 3 if a non-SVR T1 / PB unit is missing or failed M-1)

Same functions as Phase E (d1_eval.py): representative model, metrics, core cluster bootstrap + Holm, stages, bend
calls. Differences: only the T1 split (+ arm P and B_ref, which are T1), SVR units left out, and the representative of
B / C chosen among the 5 finished models (Ridge, KRR, XGB, NN-2, NN-4) by the pre-registered rule (lowest mean inner-CV
MAE). The final Phase E (all 6 models, all splits, after collect) supersedes everything written here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("D1PAIR_MODELS", None)
import d1pair_common as C  # noqa: E402
import d1_eval as E  # noqa: E402
import d1_train as T  # noqa: E402

OUT = C.RES / "interim_T1_noSVR"
EXCLUDE = ("SVR",)


def gather():
    recs, parts, missing, m1 = [], [], [], []
    for g in ("T1", "PB"):
        for u in T.unit_list(g):
            if u["model"] in EXCLUDE:
                continue
            f = T.UNITS / g / f"{T.uname(u)}.json"
            if not (f.is_file() and f.with_suffix(".parquet").is_file()):
                missing.append(T.uname(u)); continue
            rec = json.loads(f.read_text())
            if rec.get("gate_M1") is False:
                m1.append(T.uname(u))
            recs.append({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in rec.items() if k != "grid_mae"})
            parts.append(pd.read_parquet(f.with_suffix(".parquet")))
    return pd.DataFrame(recs), pd.concat(parts, ignore_index=True), missing, m1


def r(x, t):
    return None if x is None or not np.isfinite(x) else round(float(x), 3 if t in C.FINE3 else 2)


def main():
    U, M, missing, m1 = gather()
    if missing or m1:
        print(f"STOP: missing {missing[:10]} / M-1 failed {m1[:10]}", file=sys.stderr)
        return 3
    S = pd.read_csv(C.RES / "S_folds.csv")
    F = pd.read_csv(C.RES / "F_pairs.csv", usecols=["pair_id"] + [f"dd_{t}" for t in C.TARGETS])
    S = S.merge(F, on="pair_id", how="left", validate="1:1").set_index("pair_id")
    f0r = json.loads((C.RES / "f0_report.json").read_text())
    f0 = pd.read_parquet(C.RES / "f0_d1.parquet")
    L, reps = E.rep_table(S, M, U, f0, f0r)
    Mt = E.metrics(L, U)
    Pt = E.primary_tests(L)
    St = E.secondary_tests(L, reps)
    St = St[St.split == "T1"]
    Bd = E.bends(L, splits=("T1",))
    br = E.bend_rates(Bd, "T1")
    Sg = E.stages(Mt, Pt, br)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in (("metrics", Mt), ("primary_tests", Pt), ("secondary_tests", St), ("bends", Bd), ("stages", Sg),
                     ("units", U)):
        C.write_atomic(OUT / f"interim_{name}.csv", lambda t, d=df: d.to_csv(t, index=False))
    C.write_atomic(OUT / "interim_predictions_T1_noSVR.parquet", lambda t: M.to_parquet(t, index=False))
    rep = {f"{sp}|{a}|{t}": m for (sp, a, t), m in sorted(reps.items())}
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(written=time.strftime("%Y-%m-%d %H:%M"), git_head=head, excluded_models=list(EXCLUDE), n_units=len(U),
                representative=rep, bend_rates_T1=br, note="provisional; superseded by the final Phase E")
    C.write_json_safe(OUT / "interim.json", meta)

    # ---- readable summary (numbers copied from the CSVs above)
    def mae(arm, t, pos="all", sset="P_main"):
        q = Mt[(Mt.split == "T1") & (Mt.arm == arm) & Mt.rep & (Mt.set == sset) & (Mt.target == t) & (Mt.pos == pos)]
        return float(q.mae.iloc[0]) if len(q) else np.nan
    lines = ["# 잠정 T1 결과 — SVR 제외 (PROVISIONAL)", "",
             f"작성 {meta['written']} · git {head} · `d1_eval_interim.py` (계산 노드). **SVR unit이 아직 계산 중이라 6개 모델 중 5개"
             f"(Ridge, KRR, XGB, NN-2, NN-4)만으로 대표 모델을 골랐다.** 최종 결과(Phase E, `results/E_*`)가 이 파일을 대체한다. "
             f"사전 등록(`PREREG_D1PAIR.md`)은 이 결과를 보기 전에 commit되었고 바뀌지 않는다.", "",
             f"시험 짝: T1 바깥 5 fold를 합친 P_main 기준 짝 {int(L[(L.arm == 'null') & (L.target == 'barrier') & L.in_P_main & (L.ref_kind == 'reference')].shape[0])}개 "
             f"(44 core), anchor 시험 짝은 따로.", "",
             "## 주 비교 — B 대표 모델 대 A (시험 짝 MAE, kcal/mol; core cluster bootstrap 2,000, Holm 9개)", "",
             "| 타깃 | B 대표 | null | A (f0 차이) | B | C | P | B−ref (KRR) | A − B [95 % CI] | Holm p | B가 낮은 fold |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in C.PRIMARY_TARGETS:
        p = Pt[Pt.target == t].iloc[0]
        lines.append(f"| {t} | {reps.get(('T1', 'B', t))} | {r(mae('null', t), t)} | {r(mae('A', t), t)} | {r(mae('B', t), t)} | "
                     f"{r(mae('C', t), t)} | {r(mae('P', t), t)} | {r(mae('B_ref', t), t)} | {r(p.d_A_minus_B, t)} "
                     f"[{r(p.ci_lo, t)}, {r(p.ci_hi, t)}] | {p.p_holm:.2g} | {int(p.folds_B_lower)}/{int(p.n_folds)} |")
    lines += ["", f"c_ghost (주 비교 밖): null {r(mae('null', 'c_ghost'), 'c_ghost')}, A {r(mae('A', 'c_ghost'), 'c_ghost')}, "
              f"B {r(mae('B', 'c_ghost'), 'c_ghost')} ({reps.get(('T1', 'B', 'c_ghost'))}), C {r(mae('C', 'c_ghost'), 'c_ghost')}.", "",
              "## 위치별 MAE — B 대표 모델 (괄호: null)", "", "| 타깃 | p | m | o | direct | anchor 짝 |", "|---|---|---|---|---|---|"]
    for t in C.PRIMARY_TARGETS:
        cells = [f"{r(mae('B', t, pos), t)} ({r(mae('null', t, pos), t)})" for pos in C.POSITIONS]
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {r(mae('B', t, 'all', 'A_main'), t)} |")
    lines += ["", "## 단계 판정 (사전 등록 기준, 잠정)", "",
              "| 타깃 | 단계 0 (B < null, A − B CI > 0, Holm p < 0.05) | R²_fam 중앙값 | 단계 1 (≥ 0.5) | para 짝 MAE | 꺾임 민감도 | 위양성률 | 단계 2 |",
              "|---|---|---|---|---|---|---|---|"]
    for row in Sg.itertuples():
        f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {row.target} | {row.stage0} | {f(row.r2_fam_median)} | {row.stage1} | {f(row.mae_para_B, 2)} | "
                     f"{f(row.bend_sens, 2)} | {f(row.bend_fpr, 2)} | {'' if row.stage2 is None else row.stage2} |")
    lines += ["", "## 꺾임 판정 일치 (T1 시험 para 계열, B 대표 모델; Clopper–Pearson 95 %)", "",
              "| 채널 | DFT 꺾임 계열 | 민감도 [CI] | DFT 직선 계열 | 위양성률 [CI] | DFT V | V 민감도 |", "|---|---|---|---|---|---|---|"]
    for c, b in br.items():
        g = lambda v: "–" if v is None or not np.isfinite(v) else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {c} | {b['n_bent_dft']} | {g(b['sens'])} [{g(b['sens_ci'][0])}, {g(b['sens_ci'][1])}] | {b['n_straight_dft']} | "
                     f"{g(b['fpr'])} [{g(b['fpr_ci'][0])}, {g(b['fpr_ci'][1])}] | {b['n_V_dft']} | {g(b['V_sens'])} |")
    lines += ["", "## 참고", "", "- disp는 b_disp feature가 분산 채널과 거의 같은 양이라 헤드라인으로 쓰지 않는다(§3-1).",
              "- 모든 수치는 같은 폴더의 `interim_*.csv`에서 옮겼다. 모델별 결과는 `interim_metrics.csv`(rep=False 행 포함).",
              "- SVR이 끝나면 대표 모델이 바뀔 수 있고, 그에 따라 B 열·검정·단계 판정도 바뀔 수 있다."]
    C.write_atomic(OUT / "INTERIM_T1_noSVR.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
