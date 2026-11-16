#!/usr/bin/env python3
"""ESPB Phase G (docs/specs/ESPB.md 4): pre-registration before any A / B training job.

    python prereg.py write      # results/PREREG_ESPB.md, results/prereg_espb.json, results/COPY.json, results/COPY_DIFF.patch
    python prereg.py check      # the gate below, as a command (exit 3 if not satisfied)

gate() (called by ea_train.py / eb_train.py run): PREREG_ESPB.md and prereg_espb.json are tracked in HEAD with no
uncommitted change, and every sha256 registered in prereg_espb.json (eb_common.py, ea_train.py, eb_train.py, collect.py,
compare.py and the Phase P outputs) equals the file now; else exit 3. `write` needs a passed Phase P (P_report.json).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eb_common as C  # noqa: E402

REGISTERED = [HERE / "eb_common.py", HERE / "ea_train.py", HERE / "eb_train.py", HERE / "collect.py", HERE / "compare.py",
              C.X46, C.P_REACTIONS, C.P_INNER, C.P_REPORT]
COPIES = [("analysis/d1_pair/d1_f0.py", "analysis/espb/ea_train.py"),
          ("analysis/d1_pair/d1_train.py", "analysis/espb/eb_train.py"),
          ("analysis/d1_pair/jobs/env.sh", "analysis/espb/jobs/env.sh"),
          ("analysis/d1_pair/jobs/py.slurm", "analysis/espb/jobs/py.slurm"),
          ("analysis/gnn3r2/gr_common.py", "analysis/espb/eb_common.py")]          # helpers only (sha256, write_*, read_json)
CHANGES = [
    ("X-1", "Coley [3+2], ωB97X-D/def2-TZVP", "GNN3R2 데이터셋 8,594 반응 / 8,344 짝, cr2 B3LYP-D3(BJ)/def2-TZVP SMD(water)", "데이터 동일"),
    ("X-2", "AM1(Gaussian) descriptor 46개", "GFN2-xTB/ALPB(water) ESPLEY46 (DIST11 + MULLIKEN15 + VALENCE15 + E5); APT 대신 Wiberg valence, q_barrier 없음", "사용자 결정 2026-11-16"),
    ("X-3", "AM1 TS·GS", "g2r xTB TS ($G2R/lf/X182_lf.parquet, GNN3와 같은 파일)", "입력 정보 수준 동일"),
    ("X-4", "무작위 80/10/10, seed 22/23/14/1/2", "GNN3R2 core 단위 outer fold 5개 (T1__f<k>.json)", "분할 동일"),
    ("X-5", "seed 23 train에서 KFold(5) GridSearchCV 한 번", "outer fold마다 GroupKFold(4, group = core) = GNN3 inner split", "튜닝 조건 동일"),
    ("X-6", "ESI Table S3 grid, KRR polynomial(표)/RBF(본문)", "ESI Table S3 ∪ rev 4 grid, KRR RBF; 효과 없는 값(Ridge tol, RBF SVR coef0·degree) 제외(P-9)", "원본보다 좁지 않은 grid"),
    ("X-7", "y 원값", "y 표준화(rev 4 _make_pipe), NN은 r6_nn 그대로", "rev 4에서 원본을 이긴 설정"),
    ("X-8", "타깃 5개(dipole·dipolarophile distortion, Interaction, ΔE‡, ΔG‡)", "E-A: d1, d2, inter, solv, barrier (Interaction = inter + solv로 나눔, ΔG‡ 없음)", "우리 3채널"),
    ("X-9", "반응별 절대값", "짝 ΔΔ: E-A = f(a) − f(b), E-B = 반대칭 짝 모델(D1PAIR arm C)", "task 동일"),
    ("X-10", "seed별 분할의 평균 MAE", "NN 최종 예측 = seed 22, 23, 14, 1, 2 예측의 평균", "GNN3 5-seed 평균과 맞춤"),
]
EXCLUDED = [("AM1 원본 descriptor 계산", "사용자 결정 2026-11-16"), ("feature 효과와 구조 효과를 나누는 arm", "사용자 결정 2026-11-16"),
            ("XGB", "Espley 모델이 아님"), ("GNN3 재학습", "GNN3R2 CV 예측(commit됨)을 그대로 씀"),
            ("이전 결과 삭제", "대체하는 이전 판 없음")]


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def gate() -> None:
    problems = []
    for f in (C.PREREG_MD, C.PREREG_JSON):
        rel = str(f.relative_to(C.REPO))
        if git("ls-files", "--error-unmatch", rel)[0] != 0:
            problems.append(f"{rel} not committed")
        elif git("diff", "--quiet", "HEAD", "--", rel)[0] != 0:
            problems.append(f"{rel} changed after the commit")
    if C.PREREG_JSON.is_file():
        for path, sha in C.read_json(C.PREREG_JSON)["sha256"].items():
            if not Path(path).is_file() or C.sha256(path) != sha:
                problems.append(f"{path} sha256 differs from the registration")
    else:
        problems.append("prereg_espb.json missing")
    if problems:
        print(f"STOP (pre-registration gate): {problems}", file=sys.stderr)
        sys.exit(3)


def copy_records():
    rows, patch = [], []
    for orig, copy in COPIES:
        r = subprocess.run(["git", "-C", str(C.REPO), "diff", "--no-index", "--no-color", "--", orig, copy],
                           capture_output=True, text=True)
        patch.append(r.stdout)
        rows.append(dict(original=orig, copy=copy, original_sha256=C.sha256(C.REPO / orig), copy_sha256=C.sha256(C.REPO / copy),
                         original_in_origin_gnn3r2=git("cat-file", "-e", f"origin/gnn3r2:{orig}")[0] == 0,
                         diff_lines=sum(1 for l in r.stdout.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))))
    C.write_atomic(C.RES / "COPY_DIFF.patch", "".join(patch))
    C.write_json(C.RES / "COPY.json", dict(spec="docs/specs/ESPB.md 2-2 / 2-3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
                                           files=rows, patch_sha256=C.sha256(C.RES / "COPY_DIFF.patch")))
    return rows


def md(shas, copies) -> str:
    L = ["# ESPB 사전 등록 (Phase G)", "",
         f"- 지시서: `docs/specs/ESPB.md` · 작성 {time.strftime('%Y-%m-%d %H:%M:%S')} · 학습(A·B) 제출 전에 commit한다.",
         "- 학습 스크립트(`ea_train.py`, `eb_train.py`)는 이 파일과 `prereg_espb.json`이 HEAD에 commit되어 있고, json에 적힌 "
         "sha256이 현재 파일과 같을 때만 돈다(`prereg.gate`).", "",
         "## 1. Espley 원본에서 바꾸는 점 (지시서 0-2)", "", "| # | Espley 2024 원본 (ds3) | ESPB | 이유 |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CHANGES]
    L += ["", "## 2. 하지 않는 것 (0-3)", "", "| 항목 | 이유 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in EXCLUDED]
    L += ["", "## 3. 모델 · grid · 전처리 · seed (5절)", "",
          "**E-A (각각 예측하고 빼기, `ea_train.py`)** — 반응별 타깃 d1, d2, inter, solv, barrier. "
          "`train_ml_single._make_pipe(est)` = X StandardScaler → est, y 표준화. GridSearchCV(cv = P-6의 core inner fold 4개, "
          "neg MAE, refit=False), 최적 = 평균 inner MAE 최저(동률: grid 순서 앞). 최종 fit = 학습 반응 전부.", "",
          "| 모델 | grid | 칸 |", "|---|---|---|"]
    for m, g in C.EA_GRIDS.items():
        L.append(f"| {m} | " + " × ".join(f"{k.split('__')[-1]} {v}" for k, v in g.items()) + f" | {C.EA_GRID_CELLS[m]} |")
    L += ["", "- NN-2 / NN-4: `r6_nn` 그대로. tune = seed 23, inner fold 1 train → val (Hyperband `r6_nn.HB`, "
          "`HP_STORAGE`). cv = seed 22, inner fold마다 새 프로세스로 fit_predict(validation_data = 그 inner val) → OOF. "
          f"final = seed {list(C.NN_FINAL_SEEDS)} 각각 새 프로세스, 학습 반응 전부, validation_data = inner fold 1 val(history "
          "전용), 예측 = 다섯 seed 평균.",
          "- 짝 예측: strain = [f_d1 + f_d2](a) − [f_d1 + f_d2](b), inter·solv·barrier = f(a) − f(b). NN은 seed별 ΔΔ의 평균, "
          "표준편차 = yhat_seed_sd.", "",
          "**E-B (차이를 직접 학습, `eb_train.py`)** — 짝 타깃 strain, inter, solv, barrier(채널 ΔΔ). `d1_train` 그대로: "
          "z = [Δx / s_Δ, (x̄ − μ) / s], y / s_y, 반대칭 Ridge(Δx만, 절편 없음) · KRR · SVR(반대칭 곱 커널, 양방향) · NN(공유 h).",
          f"- grid = `d1_train.grid_cells`: Ridge alpha {C.EB_RIDGE_ALPHA} (8칸); KRR gamma_Δ × gamma_x̄ ∈ {C.GAMMAS}², alpha "
          f"{C.EB_KRR_ALPHA} (384칸); SVR gamma_Δ × gamma_x̄ ∈ {C.GAMMAS}², C {C.SVR_C}, epsilon {C.SVR_EPS} (768칸).",
          f"- 튜닝 = GroupKFold({C.N_INNER}, core) (= P-6), 최적 = 평균 inner MAE 최저. NN: tune(inner fold 1, seed 23), "
          f"cv(inner 4개, seed 22, 한 프로세스), final = seed {list(C.NN_FINAL_SEEDS)} 평균. gate M-1(반대칭) ≤ {C.ANTISYM_TOL} "
          f"kcal/mol, test 짝 {C.ANTISYM_N}개, NN은 seed마다.", "",
          "## 4. 대표 선택 규칙 (7-2, test 점수 미사용)", "",
          "- 후보: 타깃마다 {EA, EB} × {Ridge, KRR, SVR, NN2, NN4} = 10개.",
          "- 점수 = inner ΔΔ MAE: outer fold k마다 inner fold 4개의 'inner val 짝 전체에 대한 MAE'를 평균하고, 그 값을 5개 fold에 "
          "대해 평균. E-A는 inner OOF 반응 예측으로 inner val 짝의 ΔΔ를 만들어 계산, E-B는 최적 칸의 inner fold별 MAE(NN은 cv 단계).",
          "- 고르기 = 점수 최저. 동률이면 EA → EB, Ridge → KRR → SVR → NN2 → NN4 순서의 앞 것.",
          "- `collect.py`는 json과 OOF 행만 읽어 `C_selection.csv`를 먼저 쓰고(sha256 기록), 그 뒤에 test 예측을 읽는다(C-4).", "",
          "## 5. 비교 (8-2)", "",
          "- **주 비교(확증):** 집합 P_main(기준 짝), 위치 all, 타깃 4개. d = MAE(ESPB) − MAE(GNN3). "
          f"`gnn3.metrics.bootstrap_mae_difference(err_ESPB, err_GNN3, core)`, core bootstrap {C.BOOT_N}번, seed {C.BOOT_SEED}, "
          "95 % CI와 양측 p. **Holm 보정 범위 = 이 4개 타깃.**",
          "- **보조(설명용):** P_main para, test_all(all, para), P_main 위치(p, m, o, direct), series(R1, R2); 후보 10개 각각 대 "
          "GNN3(P_main all, para); 0 예측 기준 MAE와 개선폭; 수축 기울기(P_main barrier, yhat의 y에 대한 OLS 기울기, series별). "
          "Holm은 각 묶음(같은 비교·집합·부분집합의 타깃 4개) 안에서 따로 표시한다.",
          "- 표: `M.error_table`, `M.ranking_table`, `M.cause_agreement`, `M.breakdown_table`(GNN3, ESPB); Espley 쪽 "
          "`E_native.csv`(반응별 절대 MAE와 % range), `E_preml.csv`(ML 이전 xTB).",
          "- 채점 코드 동일성 E-0: GNN3R2 CV 예측을 같은 코드로 다시 채점한 표 = `analysis/gnn3r2/results/eval/metrics.csv` (≤ 1e-12).", "",
          "## 6. 등록 파일 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    L += [f"| `{p}` | `{s}` |" for p, s in shas.items()]
    L += ["", "## 7. 복사 기록 (2-2, 2-3)", "", "| 원본 | 사본 | 원본 sha256 | 사본 sha256 | 바뀐 줄 |", "|---|---|---|---|---|"]
    L += [f"| `{c['original']}` | `{c['copy']}` | `{c['original_sha256'][:12]}…` | `{c['copy_sha256'][:12]}…` | {c['diff_lines']} |"
          for c in copies]
    L += ["", "diff 전체: `results/COPY_DIFF.patch`. 결정·편차: `results/DECISIONS.md`.", ""]
    return "\n".join(L)


def cmd_write() -> int:
    p = C.read_json(C.P_REPORT) if C.P_REPORT.is_file() else {}
    if not p.get("all_pass"):
        print("STOP: Phase P did not pass (results/P_report.json)", file=sys.stderr)
        return 3
    copies = copy_records()
    shas = {str(f): C.sha256(f) for f in REGISTERED}
    rec = dict(spec="docs/specs/ESPB.md 4", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), changes=[dict(id=a, espley=b, espb=c, why=d) for a, b, c, d in CHANGES],
               excluded=[dict(item=a, why=b) for a, b in EXCLUDED],
               ea=dict(grids={m: {k: v for k, v in g.items()} for m, g in C.EA_GRIDS.items()}, cells=C.EA_GRID_CELLS,
                       targets=C.TARGETS_EA, preprocessing="train_ml_single._make_pipe (X and y standardised)",
                       nn=dict(tune_seed=C.TUNE_SEED, cv_seed=C.CV_SEED, final_seeds=list(C.NN_FINAL_SEEDS))),
               eb=dict(grid="d1_train.grid_cells", ridge_alpha=C.EB_RIDGE_ALPHA, krr_alpha=C.EB_KRR_ALPHA, gammas=C.GAMMAS,
                       svr_c=C.SVR_C, svr_eps=C.SVR_EPS, cells=dict(Ridge=8, KRR=384, SVR=768), targets=C.TARGETS_EB,
                       preprocessing="d1_train.Scale", nn=dict(tune_seed=C.TUNE_SEED, cv_seed=C.CV_SEED,
                                                              final_seeds=list(C.NN_FINAL_SEEDS)),
                       antisym_tol=C.ANTISYM_TOL, antisym_n=C.ANTISYM_N),
               selection=dict(candidates=[C.model_name(a, m) for a in C.ARMS for m in C.MODELS],
                              score="mean over 5 outer folds of the mean over 4 inner folds of the inner-val pair MAE",
                              tie_order="EA, EB; Ridge, KRR, SVR, NN2, NN4", order_enforced="C_selection.csv before test rows"),
               primary=dict(set="P_main", pos="all", targets=C.CHANNELS, statistic="MAE(ESPB) - MAE(GNN3)",
                            test="gnn3.metrics.bootstrap_mae_difference, core bootstrap", n=C.BOOT_N, seed=C.BOOT_SEED,
                            holm="the 4 targets"),
               secondary=dict(sets=["P_main para", "test_all all", "test_all para", "P_main p/m/o/direct", "P_main R1/R2"],
                              candidates_vs_gnn3=True, zero_predictor=True, shrinkage_slope="P_main barrier per series",
                              holm="within each group of 4 targets, reported separately"),
               sha256=shas, copies=copies)
    C.write_json(C.PREREG_JSON, rec)
    C.write_atomic(C.PREREG_MD, md(shas, copies))
    print(json.dumps(dict(written=[str(C.PREREG_MD), str(C.PREREG_JSON)], sha256=shas), indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["write"]:
        sys.exit(cmd_write())
    if a == ["check"]:
        gate()
        print("pre-registration gate: ok")
        sys.exit(0)
    raise SystemExit(__doc__)
