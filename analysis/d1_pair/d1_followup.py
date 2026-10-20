#!/usr/bin/env python3
"""d1_followup.py — D1PAIR FU1 (docs/specs/D1PAIR_FU1.md): T2 stop checks, PREREG addendum, learning curve, wide XGB.

  python d1_followup.py checkK           # gates K-a (30 cached fingerprints), K-b (unit lists vs 5df20b6), K inventory
  python d1_followup.py checkKc          # gate K-c: no T2 unit json written after the K-2 cancel time
  python d1_followup.py add1             # results/PREREG_D1PAIR_ADD1.md + prereg_d1pair_add1.json (commit them next)
  python d1_followup.py list LC|XW
  python d1_followup.py run LC|XW <i> <n>
  python d1_followup.py collect LC|XW    # results/followup/LC_* / XW_* (+ LC_curve.png)
  python d1_followup.py report           # results/followup/SUMMARY_FU1_tables.md (tables for SUMMARY_FU1.md)

LC (exploratory): arm C, KRR and XGB (PREREG grids), 10 targets, T1 outer folds 0-4, training fraction f in
{0.25, 0.5, 0.75} x repeat r in {0, 1, 2}: the training-side cores of fold k (T_main training pairs; the anchor-only cores
included) sorted by name, numpy.random.default_rng(20261019 + 100 k + r).permutation, the first ceil(f * n) cores kept
(nested in f), all their training pairs used; test pairs = the T1 unit's. f = 1.0 = the finished T1 units (fingerprint
re-checked). Tuning / scaling / antisymmetry / gate M-1 = d1_train (masks, tune_grid, fit_final).
XW (exploratory): arm C, model XGBW, 10 targets x T1 folds = 50 units; grid n_estimators {500, 1000, 2000, 4000} x
learning_rate {0.05, 0.1, 0.2} x max_depth {5, 7, 9} (key-sorted product, as d1_train.grid_cells), the rev 6 XGB estimator
otherwise (d1_train._xgb_fit / _xgb_g, both-ways training, 1/2 [h(dx) - h(-dx)]).
Cache $D1PAIR_SCRATCH/M/units_fu/{LC,XW}/; fingerprint = the d1_train items + chosen-core sha256 + FU_CODE_VERSION.
Gates: PREREG (d1_split.gate_prereg) and the FU1 addendum (d1_split.gate_add1) before any LC / XW run or collect.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("D1PAIR_MODELS", None)
import d1pair_common as C  # noqa: E402
import d1_train as T  # noqa: E402

FU_CODE_VERSION = "d1pair-fu-1"
FU = T.M_DIR / "units_fu"
OUT = C.RES / "followup"
K_TIME = "2026-10-20 08:16:27"                 # K-2: cancel of the T2-running tasks (KST, login node `date`)
K_CANCELLED = ["1025347_0 (d1m_nn, NN-2/NN-4, in run T2)", "1025512_0 (d1m_svrH, SVR helper, in run T2)",
               "1025512_1 (d1m_svrH, in run T2)", "1025512_2 (d1m_svrH, in run T2)",
               "1025351 d1m_collect + 1025352 d1e_all (stale chain: depended on the cancelled tasks)"]
K_RESUBMITTED = ["1025589 d1m_svrH2 array 0-2 (SVR helpers, same slices, run LOSO ::: LODO ::: T1, reverse order)",
                 "1025589_2 was cancelled at once to free a slot for the K check (1025592) and resubmitted as 1025594 (array 2)"]
K_LEFT = ["1025512_3 d1m_svrH (pending; its first command `run T2` is skipped by SKIP_GROUPS, then LOSO / LODO / T1)",
          "1025348_0-1 d1m_grid, 1025350_0-3 d1m_svr (not at T2 yet; T2 is their last command and is skipped)"]
FRACS = [0.25, 0.5, 0.75]
REPS = [0, 1, 2]
LC_MODELS = ["KRR", "XGB"]
XW_GRID = {"n_estimators": [500, 1000, 2000, 4000], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [5, 7, 9]}
TARGET_MAE = {"barrier": 0.16, "d1": 0.19, "d2": 0.04, "elst": 0.21, "pauli": 0.41, "oi": 0.29, "cpcm": 0.22, "cds": 0.01,
              "disp": 0.05}           # FU1 L-3 (post hoc: median bend size x 0.28, review 2026-10-20)
N_JOBS = T.N_JOBS


def tu(model, t, k):
    return dict(group="T1", split="T1", arm="C", model=model, target=t, fold=k)


# ================================================================ Phase K checks
def cmd_checkK():
    OUT.mkdir(parents=True, exist_ok=True)
    # K-b: unit lists vs 5df20b6
    old_src = subprocess.run(["git", "-C", str(HERE), "show", "5df20b6:analysis/d1_pair/d1_train.py"], capture_output=True,
                             text=True, check=True).stdout
    import importlib.util
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "d1_train_5df20b6.py"
        p.write_text(old_src)
        spec = importlib.util.spec_from_file_location("d1_train_5df20b6", p)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        kb = {g: [old.uname(u) for u in old.unit_list(g)] == [T.uname(u) for u in T.unit_list(g)] for g in T.GROUPS}
    # K-a: 30 finished units (T1, LODO, LOSO, PB; every model), only the stored fingerprint is read (no score is opened)
    done = []
    for g in ("T1", "LODO", "LOSO", "PB"):
        for u in T.unit_list(g):
            if (T.UNITS / g / f"{T.uname(u)}.json").is_file():
                done.append(u)
    rng = np.random.default_rng(C.SEED)
    by_model = {m: [u for u in done if u["model"] == m] for m in C.MODELS}
    pick = []
    for m, us in by_model.items():                      # every model present, the rest at random
        if us:
            pick.append(us[int(rng.integers(len(us)))])
    rest = [u for u in done if u not in pick]
    pick += [rest[i] for i in rng.choice(len(rest), size=30 - len(pick), replace=False)]
    ka = []
    for u in pick:
        tr, te = T.masks(u)
        stored = json.loads((T.UNITS / u["group"] / f"{T.uname(u)}.json").read_text())["fingerprint"]
        ka.append(dict(unit=T.uname(u), group=u["group"], model=u["model"], same=T.fingerprint(u, tr, te) == stored))
    # inventory
    rows = []
    for g in T.GROUPS:
        for u in T.unit_list(g):
            rows.append(dict(group=g, arm=u["arm"], model=u["model"], done=(T.UNITS / g / f"{T.uname(u)}.json").is_file()))
    inv = pd.DataFrame(rows).groupby(["group", "arm", "model"]).done.agg(["sum", "count"]).reset_index() \
        .rename(columns={"sum": "n_done", "count": "n_units"})
    C.write_atomic(OUT / "K_inventory.csv", lambda t: inv.to_csv(t, index=False))
    rep = dict(gate_Ka=dict(n=len(ka), all_same=all(r["same"] for r in ka), models=sorted({r["model"] for r in ka}),
                            groups=sorted({r["group"] for r in ka}), rows=ka),
               gate_Kb=dict(per_group=kb, all_same=all(kb.values())), k_time=K_TIME, cancelled=K_CANCELLED,
               resubmitted=K_RESUBMITTED, left_running=K_LEFT, skip_groups=sorted(T.skipped()),
               t2_units_done=int(inv.loc[inv.group == "T2", "n_done"].sum()), inventory_sha256=C.sha256(OUT / "K_inventory.csv"))
    rep["stop"] = not (rep["gate_Ka"]["all_same"] and rep["gate_Kb"]["all_same"])
    C.write_json_safe(OUT / "K_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "gate_Ka"} | {"gate_Ka_all_same": rep["gate_Ka"]["all_same"]}, indent=1))
    return 3 if rep["stop"] else 0


def cmd_checkKc():
    t0 = time.mktime(time.strptime(K_TIME, "%Y-%m-%d %H:%M:%S"))
    if time.time() < t0 + 3600:                             # FU1 K-4: the check covers the full hour after K-2
        print(f"gate K-c: wait until 1 h after {K_TIME}", file=sys.stderr)
        return 2
    new = sorted(p.name for p in (T.UNITS / "T2").glob("*.json") if p.stat().st_mtime > t0)
    rep = json.loads((OUT / "K_report.json").read_text())
    rep["gate_Kc"] = dict(checked=time.strftime("%Y-%m-%d %H:%M:%S"), new_t2_units_after_k_time=new, ok=not new)
    rep["stop"] = bool(rep["stop"] or new)
    C.write_json_safe(OUT / "K_report.json", rep)
    print(json.dumps(rep["gate_Kc"], indent=1))
    return 3 if new else 0


# ================================================================ Phase A addendum
def cmd_add1():
    kr = json.loads((OUT / "K_report.json").read_text())
    if kr.get("stop") or not (kr["gate_Ka"]["all_same"] and kr["gate_Kb"]["all_same"]) or not (kr.get("gate_Kc") or {}).get("ok"):
        print("STOP: Phase K gates (K-a, K-b, K-c) must pass before the addendum", file=sys.stderr)
        return 3
    evals = sorted(p.name for p in C.RES.glob("E_*")) + sorted(p.name for p in C.RES.glob("M_*"))
    lodo_loso_outputs = [p for p in evals]
    if lodo_loso_outputs:
        print(f"STOP: evaluation outputs already exist in results/: {lodo_loso_outputs}", file=sys.stderr)
        return 3
    interim = json.loads((C.RES / "interim_T1_noSVR" / "interim.json").read_text())
    md = f"""# PREREG ADD1 — D1PAIR 사전 등록 추가본 (FU1, 2026-10-20)

이 문서는 `PREREG_D1PAIR.md`(commit 10f99d8d, 학습 전)에 더하는 추가본이다. 원래 사전 등록은 바꾸지 않는다.
근거: `docs/specs/D1PAIR_FU1.md`, 잠정 T1 결과(`results/interim_T1_noSVR/`, SVR 제외, git {interim['git_head']}).

## A-1. 결정과 기록

- **F1** T2를 즉시 중단한다(사용자 결정 2026-10-20). 이미 끝난 T2 unit {kr['t2_units_done']}개는 지우지 않고 보관하되 분석하지 않는다
  (`$D1PAIR_SCRATCH/M/SKIP_GROUPS` = T2; 취소: {'; '.join(kr['cancelled'])}; 다시 제출: {'; '.join(kr['resubmitted'])}).
  gate K-a(캐시 지문 30개) {'통과' if kr['gate_Ka']['all_same'] else '실패'}, K-b(unit 목록) {'통과' if kr['gate_Kb']['all_same'] else '실패'}.
- **F2** learning curve를 돌린다(arm C, 탐색). **F3** XGB 탐색 범위를 넓힌다(arm C, 모델 XGBW, 탐색).
- **D-1** T2 중단(이탈, `results/DEVIATIONS.md`).
- **LODO·LOSO 결과는 이 commit 전에 계산·열람하지 않았다.** `results/`에 Phase E 산출물(`E_*`, `M_*`)이 없음을 확인했다
  (확인 시각 {time.strftime('%Y-%m-%d %H:%M')}). LODO·LOSO unit은 학습 중이며 캐시(scratch)에만 있다. 예외로 기록할 것:
  상태 점검 중 학습 로그의 마지막 줄 출력에서 LOSO unit 하나(arm B, NN-4, elst, fold 4)의 시험 MAE 한 값이 화면에 한 번
  보였다. 집계·비교(B 대 C 포함)·판정은 계산하지도 보지도 않았다. 이후 점검은 개수만 보도록 바꿨다.

## A-2. 확증 비교 추가

- T1의 "C < B"는 결과를 본 뒤 나온 관찰이라 확증이 아니다. **아직 보지 않은 LODO와 LOSO에서** 확인한다.
- 비교: split(LODO, LOSO) × 9 타깃(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier)마다 arm C 대표 모델 대 arm B 대표 모델.
  대표 모델 = PREREG 규칙(6개 모델 중 바깥 fold 평균 안쪽 CV MAE 최저, SVR 포함).
- 지표: P_main 시험 짝 MAE 차이 d = MAE(B) − MAE(C). 95 % CI와 양측 p: 그 split의 시험 짝에서 core 단위 cluster bootstrap 2,000회
  (seed 20261019; p = min(1, 2 min((1 + #{{d* ≤ 0}})/(B + 1), (1 + #{{d* ≥ 0}})/(B + 1)))). 18개에 Holm 보정. 가설 B − C > 0;
  "C가 낫다" 확증 = Holm p < 0.05 이고 CI 하한 > 0.
- SVR까지 모두 끝난 뒤 Phase E와 함께 계산한다(`d1_eval.add1_tests` → `results/E_add1_tests.csv`).

## A-3. 탐색 분석의 설계와 판독 규칙 (L, X)

결과는 **탐색**이며 PREREG의 대표 모델·arm·단계 판정을 바꾸지 않는다.

### Phase L — learning curve
- arm C, 모델 KRR·XGB(PREREG grid), 타깃 10개, T1 바깥 fold 0–4; 학습 비율 f ∈ {{0.25, 0.5, 0.75}} × 반복 r ∈ {{0, 1, 2}}
  (900 unit); f = 1.0 = 끝난 T1 unit(fingerprint 재확인).
- core 단위 축소: fold k의 학습 쪽 core(기준 짝 없는 core 포함)를 이름순 정렬 → `numpy.random.default_rng(20261019 + 100 k + r)
  .permutation` → 앞에서 ceil(f × core 수)개(같은 (k, r)에서 0.25 ⊂ 0.5 ⊂ 0.75). 고른 core의 학습 짝 전부. 시험 짝 = T1 unit과 같음.
  학습 core < 4면 STOP. 튜닝·스케일·반대칭·gate M-1 = d1_train.
- 판독: 모델 × 타깃 × 비율마다 학습 짝 수, P_main 시험 MAE(전체, para; 반복 평균·범위), 안쪽 CV MAE.
  **판정:** f = 0.5 → 1.0의 para MAE 감소율이 ≥ 10 %면 "데이터를 더 늘리면 아직 좋아진다", < 3 %면 "포화 — feature가 한계",
  그 사이 "불확실". 감소율 ≥ 3 %일 때만 참고 외삽: f = 0.5, 0.75, 1.0의 평균 para MAE에 MAE = a·n^(−b)를 log-log 최소제곱으로
  맞추고, 채널별 목표 MAE(사후 정의, 꺾임 크기 중앙값 × 0.28: {', '.join(f'{k} {v}' for k, v in TARGET_MAE.items())})에 닿는
  학습 짝 수. 외삽은 참고용.

### Phase X — XGB 넓은 grid
- arm C, 모델 XGBW, 타깃 10개 × T1 fold 5 = 50 unit. grid 36칸: n_estimators {{500, 1000, 2000, 4000}} × learning_rate
  {{0.05, 0.1, 0.2}} × max_depth {{5, 7, 9}}; 나머지는 rev 6 XGB 설정(random_state 42, n_jobs 1, tree_method hist). 양방향 학습,
  ½[h(Δx) − h(−Δx)]. 안쪽 GroupKFold(4), 최저 평균 안쪽 MAE, 같으면 grid 순서. gate M-1.
- 판독: 같은 시험 짝에서 C/XGB 대 C/XGBW의 P_main MAE(전체, para), 안쪽 CV MAE, 차이(XGB − XGBW)의 core bootstrap 95 % CI(2,000);
  Hammett 수준(R²_fam 중앙값, 채널별 꺾임 민감도·위양성률, 목표 MAE 대비 배수); 고른 값 분포와 grid 끝 걸림.
  **어떤 한 값이든 50 %를 넘는 unit이 그 값의 grid 최댓값(4000, 0.2, 9)을 고르면 STOP**(보고 후 사용자 결정; 2026-10-20 사용자
  지시에 따라 추천 옵션으로 진행).
"""
    C.write_atomic(C.RES / "PREREG_D1PAIR_ADD1.md", md)
    C.write_json_safe(C.RES / "prereg_d1pair_add1.json",
                      dict(date=time.strftime("%Y-%m-%d"), spec="docs/specs/D1PAIR_FU1.md", base_prereg_commit="10f99d8d",
                           md_sha256=C.sha256(C.RES / "PREREG_D1PAIR_ADD1.md"), decisions=["F1", "F2", "F3", "D-1"],
                           t2_units_done=kr["t2_units_done"], gate_Ka=kr["gate_Ka"]["all_same"], gate_Kb=kr["gate_Kb"]["all_same"],
                           add1_tests=dict(splits=["LODO", "LOSO"], targets=C.PRIMARY_TARGETS, metric="MAE(B) - MAE(C), P_main",
                                           bootstrap=dict(n=C.N_BOOT, seed=C.SEED, unit="core"), correction="Holm over 18"),
                           LC=dict(arm="C", models=LC_MODELS, fracs=FRACS, reps=REPS, rule={">=0.10": "still improving",
                                   "<0.03": "saturated", "else": "uncertain"}, target_mae=TARGET_MAE),
                           XW=dict(arm="C", model="XGBW", grid=XW_GRID, stop="> 50 % of units at the grid maximum of any parameter")))
    print(md)
    return 0


# ================================================================ LC / XW units
def lc_units():
    return [dict(kind="LC", model=m, target=t, fold=k, frac=f, rep=r) for m in LC_MODELS for t in C.TARGETS
            for k in range(5) for f in FRACS for r in REPS]


def xw_units():
    return [dict(kind="XW", model="XGBW", target=t, fold=k) for t in C.TARGETS for k in range(5)]


def fu_name(u):
    if u["kind"] == "LC":
        return f"C__{u['model']}__{u['target']}__f{u['fold']}__frac{u['frac']}__rep{u['rep']}"
    return f"C__XGBW__{u['target']}__f{u['fold']}"


def lc_cores(tr, k, r, f):
    S = T.data()[0]
    cores = sorted(S.loc[tr, "core"].unique())
    perm = list(np.random.default_rng(C.SEED + 100 * k + r).permutation(np.array(cores, dtype=object)))
    keep = perm[:math.ceil(f * len(cores))]
    return sorted(keep), len(cores)


def xw_cells():
    ks = sorted(XW_GRID)
    return [dict(zip(ks, v)) for v in itertools.product(*[XW_GRID[k] for k in ks])]


def fu_fingerprint(u, base_fp, cores):
    return hashlib.sha256(json.dumps(dict(u=u, base=base_fp, cores=C.list_sha([(c,) for c in cores]), fu=FU_CODE_VERSION,
                                          grid=repr(xw_cells()) if u["kind"] == "XW" else None), sort_keys=True).encode()).hexdigest()


def run_fu(u):
    S = T.data()[0]
    tunit = tu("XGB" if u["kind"] == "XW" else u["model"], u["target"], u["fold"])
    tr, te = T.masks(tunit)
    cores = None
    if u["kind"] == "LC":
        cores, ncore = lc_cores(tr, u["fold"], u["rep"], u["frac"])
        if len(cores) < C.N_INNER:
            raise SystemExit(f"STOP: {fu_name(u)} keeps {len(cores)} training cores (< 4)")
        tr = tr[S.loc[tr, "core"].isin(cores).to_numpy()]
    mu = dict(tunit, model=u["model"])                      # d1_train unit view (XGBW = XGB code path)
    fp = fu_fingerprint(u, T.fingerprint(tunit, tr, te), cores or [])
    d = FU / u["kind"]
    f = d / f"{fu_name(u)}.json"
    if f.is_file() and f.with_suffix(".parquet").is_file():
        rec = json.loads(f.read_text())
        if rec.get("fingerprint") != fp:
            raise SystemExit(f"STOP: cached {f.name} from other inputs")
        if rec.get("gate_M1") is False:
            raise SystemExit(f"STOP: gate M-1 {f.stem} (cached failure)")
        return "cached"
    t0 = time.time()
    if u["kind"] == "LC":
        cells, mean, pf = T.tune_grid(tunit, tr)
    else:
        cells = xw_cells()
        from joblib import Parallel, delayed
        per = []
        for itr, ite in T.inner_folds(tr):
            sc, (zd, zm, ys, _), (td, tm) = T.prep(itr, ite, tunit)
            yte = T.target_y(tunit, ite)
            per.append(Parallel(n_jobs=N_JOBS)(delayed(T._xgb_cell)(zd, zm, ys, td, tm, yte, sc.sy, c) for c in cells))
        pf = np.asarray(per)
        mean = pf.mean(axis=0)
    b = int(np.argmin(mean))
    best = cells[b]
    gte, gtr, anti, scl = T.fit_final(mu, tr, te, best)
    y_te = S.loc[te, f"dd_{u['target']}"].to_numpy(float)
    y_tr = S.loc[tr, f"dd_{u['target']}"].to_numpy(float)
    m1 = anti["max_sum"] <= C.ANTISYM_TOL and anti["max_zero"] <= C.ANTISYM_TOL
    pm = S.loc[te, "in_P_main"].to_numpy()
    rec = dict(u, fingerprint=fp, n_train=len(tr), n_test=len(te), n_cores_kept=len(cores) if cores else None,
               cores_kept=cores, best_params=best, inner_cv_mae=float(mean[b]), inner_fold_mae=pf[:, b].tolist(),
               edge_max={k: bool(v == max(c[k] for c in cells)) for k, v in best.items()}, antisym=anti, scale=scl,
               gate_M1=bool(m1), train_mae=float(np.mean(np.abs(gtr - y_tr))),
               test_mae_P_main=float(np.mean(np.abs(gte - y_te)[pm])), seconds=time.time() - t0, host=os.uname().nodename,
               versions=T.versions())
    pred = pd.DataFrame(dict(pair_id=S.loc[te, "pair_id"].to_numpy(), kind=u["kind"], model=u["model"], target=u["target"],
                             fold=u["fold"], frac=u.get("frac", 1.0), rep=u.get("rep", 0), y=y_te, yhat=gte))
    C.write_atomic(f.with_suffix(".parquet"), lambda t: pred.to_parquet(t, index=False))
    C.write_json_safe(f, rec)
    if not m1:
        raise SystemExit(f"STOP: gate M-1 {fu_name(u)}: {anti}")
    return f"done ({time.time() - t0:.0f} s)"


def gates():
    import d1_split as SP
    SP.gate_prereg()
    SP.gate_add1()


def cmd_run(kind, i, n):
    gates()
    us = (lc_units() if kind == "LC" else xw_units())[i::n]
    print(f"[FU {kind} {i}/{n}] {len(us)} units, n_jobs {N_JOBS}", flush=True)
    for u in us:
        try:
            st = run_fu(u)
        except SystemExit as e:
            print(f"  {fu_name(u)}: {e.code}", flush=True)
            return 3
        print(f"  {fu_name(u)}: {st}", flush=True)
    return 0


# ================================================================ collect / readout
def load_fu(kind):
    recs, parts, missing = [], [], []
    for u in (lc_units() if kind == "LC" else xw_units()):
        f = FU / kind / f"{fu_name(u)}.json"
        if not f.is_file():
            missing.append(fu_name(u)); continue
        recs.append(json.loads(f.read_text()))
        parts.append(pd.read_parquet(f.with_suffix(".parquet")))
    return recs, (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), missing


def t1_unit(model, t, k):
    """(rec, pred) of a finished T1 C unit with its fingerprint re-checked (f = 1.0 of LC, baseline of XW)."""
    u = tu(model, t, k)
    f = T.UNITS / "T1" / f"{T.uname(u)}.json"
    if not f.is_file():
        raise SystemExit(f"STOP: T1 unit {f.name} missing")
    rec = json.loads(f.read_text())
    tr, te = T.masks(u)
    if T.fingerprint(u, tr, te) != rec["fingerprint"]:
        raise SystemExit(f"STOP: T1 unit {f.name} fingerprint differs")
    return rec, pd.read_parquet(f.with_suffix(".parquet"))


def sets(S):
    P = S[S.in_P_main & (S.ref_kind == "reference")]
    return set(P.pair_id), set(P.loc[P.pos == "p", "pair_id"])


def cmd_collect(kind):
    gates()
    import d1_eval as E
    S = T.data()[0]
    allm, para = sets(S)
    recs, pred, missing = load_fu(kind)
    if missing:
        print(f"GATE: {len(missing)} {kind} units missing: {missing[:5]}", file=sys.stderr)
        return 2
    if any(r.get("gate_M1") is False for r in recs):
        print("STOP: gate M-1 failed in a follow-up unit", file=sys.stderr); return 3
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if kind == "LC":
        for r in recs:
            p = pred[(pred.model == r["model"]) & (pred.target == r["target"]) & (pred.fold == r["fold"]) &
                     (pred.frac == r["frac"]) & (pred.rep == r["rep"])]
            rows.append(dict(model=r["model"], target=r["target"], fold=r["fold"], frac=r["frac"], rep=r["rep"],
                             n_train=r["n_train"], n_cores=r["n_cores_kept"], inner_cv_mae=r["inner_cv_mae"],
                             abs_err_sum_all=float((p.yhat - p.y).abs()[p.pair_id.isin(allm)].sum()),
                             n_all=int(p.pair_id.isin(allm).sum()), abs_err_sum_para=float((p.yhat - p.y).abs()[p.pair_id.isin(para)].sum()),
                             n_para=int(p.pair_id.isin(para).sum())))
        for m in LC_MODELS:
            for t in C.TARGETS:
                for k in range(5):
                    rec, p = t1_unit(m, t, k)
                    rows.append(dict(model=m, target=t, fold=k, frac=1.0, rep=0, n_train=rec["n_train"], n_cores=None,
                                     inner_cv_mae=rec["inner_cv_mae"],
                                     abs_err_sum_all=float((p.yhat - p.y).abs()[p.pair_id.isin(allm)].sum()),
                                     n_all=int(p.pair_id.isin(allm).sum()),
                                     abs_err_sum_para=float((p.yhat - p.y).abs()[p.pair_id.isin(para)].sum()),
                                     n_para=int(p.pair_id.isin(para).sum())))
        U = pd.DataFrame(rows)
        C.write_atomic(OUT / "LC_units.csv", lambda t: U.to_csv(t, index=False))
        # pooled over the 5 folds per (model, target, frac, rep), then mean / range over the repeats
        g = U.groupby(["model", "target", "frac", "rep"]).agg(n_train=("n_train", "mean"), inner_cv_mae=("inner_cv_mae", "mean"),
                                                              ea=("abs_err_sum_all", "sum"), na=("n_all", "sum"),
                                                              ep=("abs_err_sum_para", "sum"), np_=("n_para", "sum")).reset_index()
        g["mae_all"], g["mae_para"] = g.ea / g.na, g.ep / g.np_
        cur = g.groupby(["model", "target", "frac"]).agg(n_train=("n_train", "mean"), mae_all=("mae_all", "mean"),
                                                        mae_all_min=("mae_all", "min"), mae_all_max=("mae_all", "max"),
                                                        mae_para=("mae_para", "mean"), mae_para_min=("mae_para", "min"),
                                                        mae_para_max=("mae_para", "max"),
                                                        inner_cv_mae=("inner_cv_mae", "mean"), n_rep=("rep", "nunique")).reset_index()
        C.write_atomic(OUT / "LC_curve.csv", lambda t: cur.to_csv(t, index=False))
        jud = []
        for (m, t), d in cur.groupby(["model", "target"]):
            d = d.set_index("frac")
            red = (d.loc[0.5, "mae_para"] - d.loc[1.0, "mae_para"]) / d.loc[0.5, "mae_para"]
            verdict = "still improving" if red >= 0.10 else ("saturated" if red < 0.03 else "uncertain")
            row = dict(model=m, target=t, mae_para_050=float(d.loc[0.5, "mae_para"]), mae_para_100=float(d.loc[1.0, "mae_para"]),
                       reduction_050_to_100=float(red), verdict=verdict, target_mae=TARGET_MAE.get(t))
            if red >= 0.03 and t in TARGET_MAE:
                x = np.log(d.loc[[0.5, 0.75, 1.0], "n_train"].to_numpy(float))
                y = np.log(d.loc[[0.5, 0.75, 1.0], "mae_para"].to_numpy(float))
                slope, icpt = np.polyfit(x, y, 1)
                b = -slope
                row.update(fit_b=float(b), fit_a=float(np.exp(icpt)),
                           n_for_target=float(np.exp((icpt - np.log(TARGET_MAE[t])) / b)) if b > 0 else None,
                           n_now=float(d.loc[1.0, "n_train"]))
            jud.append(row)
        J = pd.DataFrame(jud)
        C.write_atomic(OUT / "LC_verdicts.csv", lambda t: J.to_csv(t, index=False))
        lc_figure(cur, J)
        print(J.to_string())
        return 0
    # ---- XW vs XGB
    for r in recs:
        rows.append(dict(target=r["target"], fold=r["fold"], inner_cv_mae=r["inner_cv_mae"], **{f"best_{k}": v for k, v in
                                                                                          r["best_params"].items()},
                         **{f"edge_{k}": v for k, v in r["edge_max"].items()}, seconds=r["seconds"]))
    U = pd.DataFrame(rows)
    C.write_atomic(OUT / "XW_units.csv", lambda t: U.to_csv(t, index=False))
    edge_frac = {k: float(U[f"edge_{k}"].mean()) for k in XW_GRID}
    cmp_rows, frames = [], {}
    Sx = S.reset_index().set_index("pair_id")
    for name in ("XGB", "XGBW"):
        parts = []
        for t in C.TARGETS:
            for k in range(5):
                if name == "XGB":
                    rec, p = t1_unit("XGB", t, k)
                else:
                    p = pred[(pred.target == t) & (pred.fold == k)]
                parts.append(p[["pair_id", "target", "y", "yhat"]].assign(fold=k))
        frames[name] = pd.concat(parts, ignore_index=True)
    inner = {n: pd.DataFrame([dict(target=t, inner=(np.mean([t1_unit("XGB", t, k)[0]["inner_cv_mae"] for k in range(5)])
                                                    if n == "XGB" else U[U.target == t].inner_cv_mae.mean())) for t in C.TARGETS])
             for n in ("XGB", "XGBW")}
    hm = {}
    for name, fr in frames.items():
        Lx = fr.join(Sx[["ref_kind", "in_P_main", "pos", "core", "fam", "sigma", "Xb", "hidden_V_fam", "hidden_bend_fam"]],
                     on="pair_id").assign(split="T1", arm="B", rep=True, model=name)     # arm/rep: selector for d1_eval.bends
        Bd = E.bends(Lx, splits=("T1",))
        hm[name] = (Lx, E.bend_rates(Bd, "T1"))
    for t in C.TARGETS:
        a = frames["XGB"][frames["XGB"].target == t].set_index("pair_id")
        b = frames["XGBW"][frames["XGBW"].target == t].set_index("pair_id")
        for sname, ids in (("all", allm), ("para", para)):
            idx = sorted(set(a.index) & set(b.index) & ids)
            ea, eb = (a.loc[idx, "yhat"] - a.loc[idx, "y"]).to_numpy(), (b.loc[idx, "yhat"] - b.loc[idx, "y"]).to_numpy()
            d, lo, hi, p = E.boot_diff(ea, eb, Sx.loc[idx, "core"].to_numpy())
            row = dict(target=t, set=sname, n=len(idx), mae_XGB=float(np.mean(np.abs(ea))), mae_XGBW=float(np.mean(np.abs(eb))),
                       d_XGB_minus_XGBW=d, ci_lo=lo, ci_hi=hi, p=p)
            if sname == "para":
                for name in ("XGB", "XGBW"):
                    Lx, br = hm[name]
                    fe = E.family_eval(Lx[Lx.target == t])
                    row[f"r2_fam_median_{name}"] = float(fe.r2_fam.median()) if len(fe) else np.nan
                    if t in br:
                        row[f"bend_sens_{name}"], row[f"bend_fpr_{name}"] = br[t]["sens"], br[t]["fpr"]
                    row[f"x_target_{name}"] = (row[f"mae_{name}"] / TARGET_MAE[t]) if t in TARGET_MAE else np.nan
            for name in ("XGB", "XGBW"):
                row[f"inner_cv_mae_{name}"] = float(inner[name].set_index("target").loc[t, "inner"])
            cmp_rows.append(row)
    Cm = pd.DataFrame(cmp_rows)
    C.write_atomic(OUT / "XW_compare.csv", lambda t: Cm.to_csv(t, index=False))
    rep = dict(n_units=len(U), edge_max_fraction=edge_frac, best_value_counts={k: U[f"best_{k}"].value_counts().to_dict()
                                                                                for k in XW_GRID},
               stop_edge=bool(max(edge_frac.values()) > 0.5), stop_rule="> 50 % of units at the grid maximum of any parameter")
    C.write_json_safe(OUT / "XW_report.json", rep)
    print(Cm.to_string()); print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop_edge"] else 0


def lc_figure(cur, J):
    import d1_figures as Fg
    plt = Fg.style()
    fig, axs = plt.subplots(2, 5, figsize=(C.FIG_W, C.FIG_W * 0.42), constrained_layout=True)
    for ax, t in zip(axs.ravel(), C.TARGETS):
        for m, col in (("KRR", Fg.SLOT[0]), ("XGB", Fg.SLOT[1])):
            d = cur[(cur.model == m) & (cur.target == t)].sort_values("frac")
            ax.fill_between(d.n_train, d.mae_para_min, d.mae_para_max, color=col, alpha=0.12, linewidth=0)
            ax.plot(d.n_train, d.mae_para, color=col, lw=1.2, marker="o", ms=3, label=m)
        if t in TARGET_MAE:
            ax.axhline(TARGET_MAE[t], color=Fg.MUTED, lw=0.6)
        ax.set_xscale("log"); ax.set_title(Fg.TLAB[t])
    for ax in axs[1]:
        ax.set_xlabel("training pairs")
    for ax in axs[:, 0]:
        ax.set_ylabel("para ΔΔ MAE (kcal/mol)")
    axs[0, -1].legend(loc="upper right")
    Fg.FIG = OUT
    cap = ("Learning curve (exploratory): arm C, T1, para test-pair MAE vs number of training pairs "
           "(training cores subsampled at 25 / 50 / 75 %, 3 repeats, band = range; 100 % = the T1 units). Gray line = the "
           "post-hoc target MAE (median bend size x 0.28); any extrapolation to it is for reference only.")
    C.write_atomic(OUT / "LC_curve_CAPTION.md", cap + "\n")
    Fg.save(fig, "LC_curve", cur, "Learning curve (exploratory): arm C, T1, para test-pair MAE vs number of training pairs "
            "(training cores subsampled at 25 / 50 / 75 %, 3 repeats, band = range; 100 % = the T1 units). Gray line = the "
            "post-hoc target MAE (median bend size x 0.28); any extrapolation to it is for reference only.")
    plt.close(fig)


def md_table(df):
    """plain markdown table (no tabulate dependency); floats to 3 decimals."""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def cmd_report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# FU1 tables (for SUMMARY_FU1.md)", ""]
    k = json.loads((OUT / "K_report.json").read_text())
    lines += ["## K", "", f"- gate K-a: {k['gate_Ka']['all_same']} ({k['gate_Ka']['n']} units, models {k['gate_Ka']['models']})",
              f"- gate K-b: {k['gate_Kb']['all_same']} {k['gate_Kb']['per_group']}", f"- gate K-c: {k.get('gate_Kc')}",
              f"- T2 units done (kept, not analysed): {k['t2_units_done']}", ""]
    for f in ("LC_verdicts.csv", "LC_curve.csv", "XW_compare.csv"):
        p = OUT / f
        if p.is_file():
            lines += [f"## {f}", "", md_table(pd.read_csv(p)), ""]
    if (OUT / "XW_report.json").is_file():
        lines += ["## XW_report.json", "", "```", (OUT / "XW_report.json").read_text(), "```"]
    C.write_atomic(OUT / "SUMMARY_FU1_tables.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "checkK":
        sys.exit(cmd_checkK())
    if cmd == "checkKc":
        sys.exit(cmd_checkKc())
    if cmd == "add1":
        sys.exit(cmd_add1())
    if cmd == "list":
        for u in (lc_units() if a[1] == "LC" else xw_units()):
            print(fu_name(u))
        sys.exit(0)
    if cmd == "run":
        sys.exit(cmd_run(a[1], int(a[2]), int(a[3])))
    if cmd == "collect":
        sys.exit(cmd_collect(a[1]))
    if cmd == "report":
        sys.exit(cmd_report())
    raise SystemExit(__doc__)
