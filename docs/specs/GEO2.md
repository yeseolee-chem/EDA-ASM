# GEO2 — 보정 구조(geo)로 에너지 8채널 + 장벽 예측: LADDER에 rung `geo` 추가, AW·GNN·TF·XGB 4개 모두 학습 — Claude Code 실행 지시서

**저장소:** GEO1이 끝나 commit·push된 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1`에서 `geo1` HEAD를 기준으로 새 브랜치 **`geo2`**를 만든다(`git switch -c geo2`). 이 worktree에는 `analysis/geo`(GEO1), `analysis/d1_ladder`(LADDER), `analysis/d1_g3`(D1-G3), `analysis/d1_g2`, `analysis/d1_hf`, `analysis/d1_pair`가 모두 있다.
**이 파일:** `docs/specs/GEO2.md`로 commit한다.
**새 코드:** `analysis/geo2/` · **결과:** `analysis/geo2/results/` · **scratch:** `$GEO2=/gpfs/tmp_cpu2/yeseo1ee/geo2`
**LADDER 확장:** `analysis/d1_ladder/`의 학습·입력·검사 코드에 rung `geo`(stage 3)를 **추가만** 한다. 기존 rung(dft, g1, g2)의 동작과 결과 파일은 바꾸지 않는다(9절 gate).

**순서:** 0 → Y(fold −1 추가 GEO1 모델) → Z(geo 구조 폴더) → T(회귀 검사) → G(xTB Freq) → F(feature·토큰) → H(HF EDA) → P(입력표·짝) → R(사전 등록) → M3(geo 학습: AW·GNN·TF·XGB) → E(평가) → S(요약 후 **멈춤**)
- Y는 GPU, Z 이후는 CPU라서 Y가 도는 동안 T 검사를 먼저 해도 된다(Z는 Y가 끝나야 한다).
- 14절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 이 문서의 코드는 **로컬에서 실제 데이터 또는 가짜 데이터로 돌려 본 것**이다(0-3절). 기존 코드를 재사용하라고 한 부분(6–8절)은 그 코드를 읽고 맞추되, 바꾼 내용은 `results/DECISIONS.md`에 적는다. 계산 설정·판정 기준은 바꾸지 않는다.

---

## 0. 무엇을, 왜

### 0-1. 전체 파이프라인과 이 명령서의 위치

| 단계 | 입력 → 출력 | 상태 |
|---|---|---|
| 1 | SMILES → G2 TS (autodE, xTB) | 완료 (D1-G2) |
| 2 | G2 TS → 보정 TS (GEO1, Uni-Mol+ 차용) | 완료 (GEO1). 시험 fold 짝 비율 0.14 → 0.35 |
| **3** | **보정 TS → 에너지 (짝 ΔΔ 10타깃: d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost, barrier)** | **이 명령서** |

- 에너지 모델은 LADDER의 모델을 그대로 쓴다. 반응 짝의 차이(ΔΔ)를 예측한다.
- **"가장 좋았던 모델"의 정의:** LADDER에서 가장 좋은 결과는 **dft·SH**였다(DFT 구조 위 xTB feature + HF/6-31G\* 9값, REP oi para 0.098, cpcm 0.059; ENS4는 "확실히 통과").
  - REP는 하나의 모델이 아니다. 타깃마다 AW·GNN·TF·XGB 중 **검증 MAE가 가장 낮은 것**을 고르는 규칙(LADDER 13-1)이다.
  - 그래서 이 명령서는 **같은 학습 규칙으로 AW·GNN·TF·XGB 4개를 모두 geo 입력으로 학습**하고, 같은 규칙으로 고른다.

| 항목 | 내용 |
|---|---|
| 모델 | **AW, GNN, TF** (원자 토큰 모델, GPU) + **XGB** (CPU). 4개 모두 학습한다 |
| 입력 방식 (arm) | **S**(구조 feature만) + **SH**(구조 + HF 9값). LADDER와 같다 |
| 분할 | T1 fold 0–4. 학습·튜닝·seed·epoch 규칙은 LADDER 4-2·4-3 그대로 |
| 학습 단위 | 2 arm × 4 모델 × 5 fold = **40 unit** (atom 30 GPU + XGB 10 CPU) |
| 왜 다시 학습하나 | 1단계(GEO1) 구조 오차를 에너지 모델이 일부 메울 수 있도록, **예측 구조를 입력으로** 학습한다. DFT 구조로 학습한 모델에 예측 구조를 넣는 방식(학습 없이 적용)은 하지 않는다 |
| 고르기 | 타깃마다 4 모델(REP) 또는 2 arm × 4 모델(REP8) 중 검증 MAE가 가장 낮은 것. 4 모델 각각의 결과도 모두 보고한다 |

### 0-2. rung `geo`의 정의 (g2와 **TS 좌표만** 다르다)

| | `g2` (기존) | **`geo` (새로)** |
|---|---|---|
| TS 좌표 | `$G2/g2_final/<rxn>/ts.xyz` | **GEO1 예측 구조**: 그 반응을 시험 fold로 둔 GEO1 final 모델 seed 3개의 좌표 평균. fold −1 반응은 Y 단계의 추가 모델 |
| 반응물 기준 구조 | G2 `rel1.xyz`, `rel2.xyz` | **같음** (byte copy) |
| `ts.hess` (feature B6) | D1-G2 | **새 계산:** geo TS에서 `! XTB2 ALPB(water) Freq` (최적화 없음; LADDER G0와 같은 방식) |
| 전역 182 · FU2 토큰 | D1-G2 Phase F | **새 계산** (같은 코드, 루트 = geo 폴더) |
| HF/6-31G\* 9값 (H) | D1-G2 Phase H | **새 계산** (같은 runner, 'g2' geometry slot에 geo 루트) |
| H의 d1·d2·barrier 기준 | rung 안 분자 키별 최저 HF 기준(`refs="min"`) | 같음 |

- 반응물 기준 구조를 G2와 같게 두는 이유: **TS 보정의 효과만** 분리해서 보기 위해서다.
- 예측 구조가 상식 검사(5-2절)를 통과하지 못하면 G2 TS를 쓴다. 배포 때도 쓸 수 있는 규칙이고, 표시를 남긴다.

### 0-3. 이 명령서의 코드가 어디까지 확인되었나 (작성자, 2026-10-30)

| 코드 | 확인 |
|---|---|
| `eval_geo2.py` | **실제 LADDER 결과로 selftest 통과:** g2 예측을 geo로 복사해 넣으면 REP가 `E2_report.json`과 같고, g2의 REP·S/SH, ENS4·S/SH para MAE·CI 40행이 `E2_metrics.csv`와 1e-9 안에서 같으며, 차이는 0이다 |
| `build_geo.py` | 가짜 데이터(491반응)로 실행. GEO1 `E_rxn.csv`와 형성 결합 오차 차이 5e-11 Å, 재실행 시 모두 kept, 상식 검사 3종 실패 경로 확인 |
| `train_m1.py` | GEO1 학습 코드(작은 모델)로 실행: fold −1만 예측 파일에 들어감 |
| 6–8절 재사용 지시 | 코드를 읽고 설계함. HPC에서만 돌 수 있어 실행하지 못했다. T·C 검사가 그 검증이다 |

### 0-4. 이미 아는 기준값 (P_23 = g2와 geo 모두 있는 P_main 기준 짝, para; g2 REP8 기준)

| 타깃 | g2 (현재 배포 가능) | dft (상한, 배포 불가) |
|---|---|---|
| oi | 1.038 | 0.098 |
| cpcm | 0.315 | 0.059 |
| barrier | 0.397 | 0.110 |

- 이 값은 selftest에서 나온 값이다. geo가 g2와 같은 짝을 가지면 P_23 = P_g2(6,401짝)다.
- 같은 형태(same) 짝에서 g2 oi는 0.63, 다른 형태 짝에서는 1.2–2.1이다. 구조 보정은 같은 형태 짝에서 주로 효과가 있을 것으로 예상한다(판정에는 쓰지 않음).

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | GEO1이 끝났다: `analysis/geo/results/E_report.json`, `E_rxn.csv`, `R_select_f{0..4}.json`이 있고 commit·push되어 있다. `$GEO/preds/final_f{k}_s{0,1,2}.pkl`(k = 0–4) 15개가 있다 |
| 0-2 | LADDER 결과가 있다: `analysis/d1_ladder/results/L_predictions.parquet`, `L_report.json`, `L_xgb_report.json`, `E2_report.json`, `E2_metrics.csv` |
| 0-3 | D1-G3 코드가 있다: `analysis/d1_g3/feat.py`, `g3_common.py` (재사용 원본) |
| 0-4 | `$G2/g2_final/<rxn>/{ts.xyz, rel1.xyz, rel2.xyz}` (8,616 `.done`), `analysis/d1_g2/results/labels_d1_hf_g2.parquet` |
| 0-5 | 디스크 여유 30 GB 이상 (G 폴더 약 5 GB, HF 실행 폴더, 토큰) |
| 0-6 | `python analysis/geo2/eval_geo2.py selftest`가 exit 0 (`results/selftest/E_report.json`의 `selftest.all_pass` true). 아니면 코드나 LADDER 결과가 이 명령서 작성 때와 다른 것이므로 STOP |

- **폴더 만들기:** `sbatch --time=48:00:00 --output=/gpfs/home1/yeseo1ee/geo2_init_%j.out --wrap "mkdir -p /gpfs/tmp_cpu2/yeseo1ee/geo2/logs"`
- 환경:
  - CPU(ORCA, xtb, feature): D1 run 환경 (`analysis/geo/jobs/cpu.slurm`)
  - FU2 atoms, feature 자식 프로세스: `analysis/d1_pair/jobs/g.slurm` (`PYENV=venv5`, `D1PAIR_PHASE=F`), D1-G3 feat.py와 같다
  - HF: `analysis/d1_hf/jobs/hf.slurm`
  - GPU 학습(AW·GNN·TF), GEO1 pickle 읽기: `analysis/d1_ladder/jobs/gpu.slurm` / `analysis/geo/jobs/gpu.slurm` (env_d1fu2)
  - XGB 학습: LADDER와 같은 env (`analysis/d1_pair/jobs/py.slurm`, `PYENV=espley`)
  - eval: `espley` 또는 reactot

## 2. 공통 모듈 `analysis/geo2/geo2_common.py` (그대로 작성)

```python
"""GEO2 common paths and helpers (docs/specs/GEO2.md). Python 3.9+ (D1 run env), reactot and env_d1fu2 compatible.

Writes only under $GEO2 (= $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2) and analysis/geo2/results.
Every other root is read only. The roots can be overridden by environment variables (tests only)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))
GEO1 = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))            # GEO1 scratch (read only)
GEO1_RES = REPO / "analysis" / "geo" / "results"                                    # GEO1 results (read only)
G2_FINAL = Path(os.environ.get("GEO2_G2_FINAL", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final"))
D1_JOBS = Path(os.environ.get("GEO2_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
D1_LABELS_G2 = Path(os.environ.get("GEO2_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
GEO_FINAL = GEO2 / "geo_final"
COV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, As=1.19, Se=1.20,
           Br=1.20, I=1.39)                                                         # Cordero et al. 2008 (A)
SEEDS = (0, 1, 2)


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    q = Path(os.path.abspath(p))
    for root in (GEO2, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $GEO2 / analysis/geo2/results")


def write_atomic(path, data):
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=_jsonable))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], dtype=float)


def xyz_text(syms, X, comment):
    return f"{len(syms)}\n{comment}\n" + "".join(f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n"
                                                 for s, (x, y, z) in zip(syms, np.asarray(X, float)))


def marker(d):
    d = Path(d)
    if (d / ".done").is_file():
        return ".done"
    f = sorted(d.glob(".fail_*")) if d.is_dir() else []
    return f[0].name if f else None


def parse_pairs(s):
    return sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(s).split())


def dmat(X):
    X = np.asarray(X, float)
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def form_d(X, pairs):
    D = dmat(X)
    return np.array([D[i, j] for i, j in pairs])


def kabsch_rmsd(P, Q, reflect=True):
    """RMSD of P onto Q (same order) after the best proper (or, with reflect, improper) superposition."""
    P, Q = np.asarray(P, float) - np.mean(P, 0), np.asarray(Q, float) - np.mean(Q, 0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    V = Vt.T
    s = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    best = np.inf
    for dd in ((s, -s) if reflect else (s,)):
        R = V @ np.diag([1.0, 1.0, dd]) @ U.T
        if not reflect and np.linalg.det(R) < 0:
            continue
        best = min(best, float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean())))
    return best
```

## 3. Phase Y — fold −1 반응용 GEO1 추가 모델 (`analysis/geo/train_m1.py`, GPU)

- **왜:** T1 fold −1의 6개 core(380반응)는 GEO1에서 한 번도 시험 fold가 아니었다. 그래서 그 반응을 보지 않은 GEO1 모델이 없다. 그런데 이 반응들의 짝은 LADDER의 모든 fold에서 **학습 짝**으로 쓰인다. 학습 입력이 "모델이 이미 본 구조"(정답에 너무 가까운 구조)가 되면 안 되므로, 이 반응들을 뺀 모델을 하나 더 학습한다.
- **설정(결과 보기 전에 고정):**
  - 설정 = R_select에서 가장 많이 뽑힌 것. 동률이면 이름 순서(c0 < c1 < c2 < c3)
  - epoch = 5개 fold에서 고른 (best_ep + 1)의 중앙값, 내림
  - 학습 = fold ≥ 0인 D1 전부 + D0(겹침 제외). 내부 검증 없음(정해진 epoch). seed 0, 1, 2
- **GEO1 `train.py`를 바꾸지 않는다.** `fit`·`predict`의 인자가 현재 `train.py`와 다르면(GEO1에서 300 epoch, select 규칙을 바꾼 경우), 현재 `train.py`의 `final` 명령이 부르는 방식과 똑같이 부르도록 맞추고 기록한다.
- 실행: `train_m1.py plan`(CPU 가능) → `train_m1.py final 0|1|2`(GPU 3 job)
- 출력: `$GEO/preds/final_fm1_s{0,1,2}.pkl`, `analysis/geo/results/Y_plan.json`
- 이 파일은 GEO1 쪽(`analysis/geo/`)에 둔다. GEO1의 다른 파일은 바꾸지 않는다.

`analysis/geo/train_m1.py` (그대로 작성):

```python
#!/usr/bin/env python3
"""GEO2 Phase Y: one extra GEO1 refiner for the D1 reactions of T1 fold -1 (docs/specs/GEO2.md 3).

The 380 D1 reactions of the six T1 fold -1 cores are never a GEO1 test fold, so no GEO1 model left them out. Their
pairs are LADDER training pairs of every fold, so the geo rung needs their structures from a model that did not see them.
  python train_m1.py plan          -> results/Y_plan.json (config = most frequent R_select choice, ties c0 < c1 < c2 < c3;
                                      epochs = median of the 5 chosen best epochs + 1, rounded down)
  python train_m1.py final <seed>  -> $GEO/preds/final_fm1_s<seed>.pkl (seeds 0, 1, 2)
Training set = every D1 reaction with fold >= 0 + D0 (overlap False); no inner validation (fixed epochs, as 'final').
GPU env (jobs/gpu.slurm). Uses train.py unchanged (fit / predict / train_items / eval_items / load_records).
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
import train as T  # noqa: E402


def plan():
    sel = [json.loads((GC.RES / f"R_select_f{k}.json").read_text()) for k in range(5)]
    cnt = Counter(s["chosen"] for s in sel)
    top = max(cnt.values())
    cfg = sorted(c for c, v in cnt.items() if v == top)[0]
    ep = int(np.floor(np.median([s["best_ep"] + 1 for s in sel])))
    out = dict(cfg=cfg, epochs=ep, chosen=[s["chosen"] for s in sel], best_ep=[s["best_ep"] for s in sel],
               rule="cfg = most frequent R_select choice (ties: lowest name); epochs = floor(median(best_ep + 1))")
    GC.write_json(GC.RES / "Y_plan.json", out)
    print(out, flush=True)
    return 0


def final(seed):
    p = json.loads((GC.RES / "Y_plan.json").read_text())
    recs = T.load_records()
    test = [r for r in recs if r["src"] == "D1" and r["fold"] == -1]
    train = [r for r in recs if (r["src"] == "D1" and r["fold"] >= 0) or (r["src"] == "D0" and not r["overlap"])]
    assert test and not ({r["uid"] for r in test} & {r["uid"] for r in train}), "fold -1 leak"
    ema, arch, n_pass, _ = T.fit(T.train_items(train), None, p["cfg"], int(seed), fixed_epochs=int(p["epochs"]),
                                 tag=f"final_fm1_s{seed}")
    P = T.predict(ema.model(arch), T.eval_items(test), n_pass)
    out = GC.GEO / "preds" / f"final_fm1_s{seed}.pkl"
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(P, f, protocol=4)
    tmp.replace(out)
    print(f"fold -1: {len(P)} reactions -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["plan"]:
        sys.exit(plan())
    if len(a) == 2 and a[0] == "final":
        sys.exit(final(a[1]))
    raise SystemExit(__doc__)
```


## 4. Phase Z — geo 구조 폴더 (`analysis/geo2/build_geo.py`)

### 4-1. 규칙
- 반응마다 TS 하나: 그 반응을 시험 fold로 둔 GEO1 final 모델 seed 3개의 좌표 평균(GEO1 평가와 같은 앙상블). fold −1은 Y의 3개 seed 평균.
- `rel1.xyz`, `rel2.xyz` = G2 폴더에서 byte copy.
- 폴더 형식은 D1-G3 Phase O 출력(`<m>_final/<rxn>/`)과 같다: `ts.xyz`, `rel1.xyz`, `rel2.xyz`, `result.json`, `.done`. 그래서 6절의 G 단계(`feat.py` `g_one`)가 그대로 읽는다.

### 4-2. 상식 검사 (실패하면 G2 TS로 대체하고 `fallback: true`)
| 검사 | 기준 |
|---|---|
| 가까운 원자 | 모든 원자 쌍 d ≥ 0.60 × (r_i + r_j) (Cordero 공유 결합 반지름) |
| 결합 그래프 | 형성 결합을 뺀 원자 쌍에서, G2에서 결합(d < 1.10 Σr)이던 것이 1.40 Σr를 넘지 않고, 결합이 아니던 것(d > 1.40 Σr)이 1.10 Σr 아래로 오지 않음 |
| 형성 결합 길이 | 1.4–4.0 Å |

### 4-3. gate (STOP)
- GEO1 `E_rxn.csv`의 반응별 형성 결합 오차(`e_short_model`, `e_long_model`)를 이 앙상블 좌표로 다시 계산한 값과 비교해 최대 차이 ≤ 1e-6 Å. **GEO1 평가에 쓴 바로 그 구조인지** 확인하는 검사다.
  - `E_rxn.csv` 열 이름이 다르면(GEO1에서 `evaluate.py`를 고친 경우) 같은 뜻의 열을 찾아 맞추고 기록한다.
- 대체(fallback) 비율 ≤ 2 %.
- 같은 반응이 두 fold에서 예측되면 STOP.

`analysis/geo2/build_geo.py` (그대로 작성; GPU 환경을 CPU 파티션에서. `sbatch -p cpu1,cpu2 --cpus-per-task=4 --mem=32G ../geo/jobs/gpu.slurm ../geo2/build_geo.py build` 형식):

```python
#!/usr/bin/env python3
"""GEO2 Phase Z: the geo-rung structure folders $GEO2/geo_final/<rxn_id>/ from the GEO1 test predictions
(docs/specs/GEO2.md 4). GPU env on a CPU partition (jobs/gpu.slurm of analysis/geo; it reads the GEO1 pickles).

  python build_geo.py build     -> $GEO2/geo_final/<rxn_id>/{ts.xyz, rel1.xyz, rel2.xyz, result.json, .done}
                                   + results/Z_status.csv + results/Z_report.json   (exit 3 = STOP, spec 4-3)
Each D1 reaction gets exactly one TS: the 3-seed mean of the GEO1 'final' model of the T1 fold whose test set holds it
(folds 0-4: final_f<k>_s<seed>.pkl; fold -1: final_fm1_s<seed>.pkl of train_m1.py). rel1 / rel2 = byte copies of the
G2 reference structures (the g2 rung's). A structure that fails the sanity rules (4-2) is replaced by the G2 TS
(deployable fallback, flagged). Idempotent: an existing .done folder whose sources are unchanged is kept.
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo2_common as C  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
MIN_CONTACT = 0.60          # d_ij >= 0.60 (r_i + r_j) for every pair
BOND_IN, BOND_OUT = 1.10, 1.40   # G2 bond (d < 1.10 sum r) must stay < 1.40 sum r; G2 non-bond (> 1.40) stays > 1.10
FORM_RANGE = (1.4, 4.0)     # forming distances (A)
FALLBACK_MAX = 0.02         # spec 4-3
E_RXN_TOL = 1e-6            # A, forming-distance errors vs GEO1 E_rxn.csv


def load_preds():
    """uid -> (mean xyz, source tag, pkl sha256 list). Folds 0-4 and fold -1; seed sets must agree."""
    out = {}
    files = {f"f{k}": [C.GEO1 / "preds" / f"final_f{k}_s{s}.pkl" for s in C.SEEDS] for k in FOLDS}
    files["fm1"] = [C.GEO1 / "preds" / f"final_fm1_s{s}.pkl" for s in C.SEEDS]
    for tag, fs in files.items():
        miss = [str(f) for f in fs if not f.is_file()]
        if miss:
            C.stop(f"GEO1 prediction files missing: {miss}")
        parts = []
        for f in fs:
            with open(f, "rb") as fh:
                parts.append(pickle.load(fh))
        if any(set(p) != set(parts[0]) for p in parts):
            C.stop(f"{tag}: the seed files hold different reaction sets")
        shas = [C.sha256(f) for f in fs]
        for u in parts[0]:
            if u in out:
                C.stop(f"{u} predicted by two folds ({out[u][1]} and {tag})")
            out[u] = (np.mean([np.asarray(p[u], float) for p in parts], axis=0), tag, shas)
    return out


def sanity(syms, X, Xg2, pairs):
    """-> (ok, reason, info). X = geo TS, Xg2 = G2 TS (same atom order = label order)."""
    if X.shape != Xg2.shape or not np.isfinite(X).all():
        return False, "shape_or_nonfinite", {}
    r = np.array([C.COV[s] for s in syms])
    S = r[:, None] + r[None, :]
    D, Dg = C.dmat(X), C.dmat(Xg2)
    iu = np.triu_indices(len(syms), 1)
    ratio = (D / S)[iu]
    info = dict(min_contact_ratio=float(ratio.min()))
    if ratio.min() < MIN_CONTACT:
        return False, "close_contact", info
    fp = {tuple(p) for p in pairs}
    mask = np.array([(i, j) not in fp for i, j in zip(*iu)])
    rg, rx = (Dg / S)[iu][mask], (D / S)[iu][mask]
    broke = int(((rg < BOND_IN) & (rx > BOND_OUT)).sum())
    made = int(((rg > BOND_OUT) & (rx < BOND_IN)).sum())
    info.update(n_bonds_broken=broke, n_bonds_made=made)
    if broke or made:
        return False, "bond_graph_changed", info
    d = C.form_d(X, pairs)
    info["d_form"] = d.tolist()
    if d.min() < FORM_RANGE[0] or d.max() > FORM_RANGE[1]:
        return False, "forming_distance_range", info
    return True, "", info


def e_sorted(X, lab, pairs):
    dl = C.form_d(lab, pairs)
    o = np.argsort(dl)
    return (C.form_d(X, pairs) - dl)[o]


def build():
    t0 = time.time()
    labs = json.loads(C.D1_LABELS_G2.read_text())                       # 9,266 D1 targets (job_id, rxn_id, pairs)
    P = load_preds()
    erxn_path = C.GEO1_RES / "E_rxn.csv"
    erxn = {}
    if erxn_path.is_file():
        with open(erxn_path, newline="") as f:
            for row in csv.DictReader(f):
                erxn[int(row["rxn_id"])] = (float(row["e_short_model"]), float(row["e_long_model"]))
    else:
        C.stop(f"{erxn_path} missing (GEO1 Phase E must be finished)")
    rows, n_fb, e_dev, n_e = [], 0, 0.0, 0
    for d in labs:
        rid, T = int(d["rxn_id"]), d["job_id"]
        uid = f"D1_{rid}"
        g2 = C.G2_FINAL / str(rid)
        row = dict(rxn_id=rid, job_id=T, status="", source="", fallback=False, reason="")
        if C.marker(g2) != ".done":
            row.update(status="no_g2")
            rows.append(row)
            continue
        if uid not in P:
            row.update(status="no_geo1_prediction")
            rows.append(row)
            continue
        X, tag, shas = P[uid]
        syms_l, lab = C.read_xyz(C.D1_JOBS / T / "ts.xyz")
        syms_g, Xg2 = C.read_xyz(g2 / "ts.xyz")
        if list(syms_g) != list(syms_l):
            C.stop(f"{rid}: G2 ts.xyz element order != label")
        pairs = C.parse_pairs(d["formed_pairs_ts"])
        if rid in erxn:                                                  # consistency with the GEO1 evaluation
            dev = float(np.abs(e_sorted(X, lab, pairs) - np.array(erxn[rid])).max())
            e_dev, n_e = max(e_dev, dev), n_e + 1
        ok, reason, info = sanity(syms_l, X, Xg2, pairs)
        Xuse = X if ok else Xg2
        n_fb += 0 if ok else 1
        out = C.GEO_FINAL / str(rid)
        src = {"g2_rel1": C.sha256(g2 / "rel1.xyz"), "g2_rel2": C.sha256(g2 / "rel2.xyz"), "pkl": shas}
        if C.marker(out) == ".done":
            old = json.loads((out / "result.json").read_text())
            if old.get("sources") == src and old.get("fallback") == (not ok):
                row.update(status="kept", source=tag, fallback=not ok, reason=reason)
                rows.append(row)
                continue
            C.stop(f"{out} exists with other sources: move it aside explicitly (never automatic)")
        tmp = C.GEO_FINAL / f"{rid}.tmp"
        C.assert_path(tmp)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        (tmp / "ts.xyz").write_text(C.xyz_text(syms_l, Xuse, f"geo TS rxn {rid} ({'G2 fallback: ' + reason if not ok else 'GEO1 ' + tag + ' 3-seed mean'})"))
        for k in (1, 2):
            shutil.copyfile(g2 / f"rel{k}.xyz", tmp / f"rel{k}.xyz")
        heavy = [i for i, s in enumerate(syms_l) if s != "H"]
        res = dict(rxn_id=rid, job_id=T, geom="geo", source=f"GEO1 {tag} seeds {list(C.SEEDS)} mean", fold_tag=tag,
                   fallback=not ok, fallback_reason=reason, sanity=info, sources=src,
                   formed_pairs_label=[list(p) for p in pairs], d_form_geo=C.form_d(X, pairs).tolist(),
                   d_form_used=C.form_d(Xuse, pairs).tolist(), d_form_g2=C.form_d(Xg2, pairs).tolist(),
                   d_form_label=C.form_d(lab, pairs).tolist(),
                   rmsd_heavy_label_geo=C.kabsch_rmsd(X[heavy], lab[heavy]),
                   rmsd_heavy_label_g2=C.kabsch_rmsd(Xg2[heavy], lab[heavy]),
                   status="ok", written=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (tmp / "result.json").write_text(json.dumps(res, indent=1, default=C._jsonable))
        (tmp / ".done").write_text(res["written"] + "\n")
        if out.exists():
            shutil.rmtree(C.assert_path(out))
        tmp.replace(out)
        row.update(status="ok", source=tag, fallback=not ok, reason=reason)
        rows.append(row)
    n_ok = sum(r["status"] in ("ok", "kept") for r in rows)
    stops = []
    if n_e == 0 or e_dev > E_RXN_TOL:
        stops.append(f"forming-distance errors differ from GEO1 E_rxn.csv (max {e_dev:.2e} A over {n_e} rxns)")
    if n_ok and n_fb / n_ok > FALLBACK_MAX:
        stops.append(f"fallback rate {n_fb / n_ok:.3f} > {FALLBACK_MAX}")
    C.RES.mkdir(parents=True, exist_ok=True)
    with open(C.RES / "Z_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    rep = dict(n_targets=len(labs), n_folders=n_ok, by_status=dict(Counter(r["status"] for r in rows)),
               by_source=dict(Counter(r["source"] for r in rows if r["source"])), n_fallback=n_fb,
               fallback_reasons=dict(Counter(r["reason"] for r in rows if r["fallback"])),
               e_rxn_check=dict(n=n_e, max_abs_dev_A=e_dev, tol=E_RXN_TOL), stop=bool(stops), stop_reasons=stops,
               wall_s=time.time() - t0)
    C.write_json(C.RES / "Z_report.json", rep)
    print(json.dumps(rep, indent=1, default=str), flush=True)
    return 3 if stops else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        sys.exit(build())
    raise SystemExit(__doc__)
```


## 5. Phase T — 회귀 검사 (계산 전, 하나라도 실패하면 STOP)

| # | 검사 | 기준 | 근거 코드 |
|---|---|---|---|
| T-2 | feature 경로: `$LAD/dft_final`의 3반응 폴더를 복사해 6절 G·F 코드로 다시 계산 | 토큰 sha256 = `analysis/d1_ladder/results/F0_tokens_manifest.csv`, 182열 = `$LAD/X182_d1_dft.parquet` (≤ 1e-8) | D1-G3 `feat.py t2_g`·`t2_f` 그대로 |
| T-3 | HF 경로: 3반응(T00003, T00019, T00614)의 dft 좌표로 6절 H 코드 | `labels_d1_hf.parquet`(geom = dft)와 ≤ 1e-6 kcal/mol | D1-G3 `feat.py t3` 그대로 |
| T-4 | 평가 코드 | `eval_geo2.py selftest` all_pass (1절 0-6) | 이 문서 |

## 6. Phase G·F·H·P — geo 입력 (`analysis/geo2/feat_geo.py`)

### 6-1. 만드는 방법
- **`analysis/d1_g3/feat.py`를 `analysis/geo2/feat_geo.py`로 복사**하고, `g3_common.py`를 `analysis/geo2/g3c.py`로 복사한다. 이 두 파일은 D1-G3에서 검토·실행된 코드다.
- `g3c.py`에서 바꾸는 것:
  - `G3` → `$GEO2`(환경 변수 `GEO2_SCRATCH`, 기본 `/gpfs/tmp_cpu2/yeseo1ee/geo2`)
  - `RES` → `analysis/geo2/results`
  - `METHODS = ("geo",)`
  - 나머지(LAD, G2, HF_DIR, HF_SCR, P_DIR, KCAL, SEED 상수)는 그대로
- `feat_geo.py`에서 바꾸는 것:

| 부분 | 바꾸는 내용 |
|---|---|
| 대상 목록 | `part_a`, `need_o`, `o_state`, `in_scope`, `u1_*`, `base_of`의 Part A·Phase O·U-1 논리를 지우고 **`geo_base()`**로 바꾼다: `$GEO2/geo_final/<rxn>/.done`인 반응 전부(job_id 순서). 전제: `results/Z_report.json` stop false |
| 경로 | `paths(m)`: `final = $GEO2/geo_final`, `feat = $GEO2/geo_feat`. 나머지는 원래 규칙(`F_geo`, `X182_d1_geo.parquet`, `fu2_geo`, `results/fu2_geo`, `hf_geo`, `results/hf_geo`, `inputs/geo.parquet`, `results/P_build_geo.json`). 버전 `fu2-atoms-3-geo`, `fu2-tokens-3-geo` |
| G (xTB Freq) | 약 8,600반응이므로 **slice 인자**를 더한다: `g <i\|A> <n>` (LADDER `g0f0.py g0`와 같은 방식). `g_one`, `run_pool`, `order_g`는 그대로 쓴다. 출처 폴더 = `geo_final`, 쓰는 폴더 = `geo_feat` |
| F (feature·토큰) | `g0f0.py`의 `f0_slice <i> <n>` → `f0_rest`(merge, atoms, gate, tokens) 구조로 나눈다: `f_slice <i\|@I> <n>`, `f_atoms <i> <n>`, `f_rest`. 자식 명령(`d1_features.py runroot/mergeroot`, `fu2/atoms.py root run/gate/tokens`)과 환경 변수 규칙(`feat_env`, `atoms_env_m`)은 `feat.py` 그대로 |
| H (HF EDA) | 8,600반응 × 5계산이므로 **D1-G2 `analysis/d1_g2/phase_fh.py`의 H 구조**(입력 준비 → 작업자 i/n 배치 실행 → build → report)로 나눈다: `h_inputs`, `h_batch <i> <n>`, `h_build`, `h_report`. runner 호출과 환경 변수(`env_h(m)`: `D1HF_SCRATCH=$GEO2/hf_geo`, `D1HF_RES=analysis/geo2/results/hf_geo`, `D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs`(읽기 전용), `D1HF_G2_ROOT=$GEO2/geo_final`)는 `feat.py` 그대로 |
| P (입력표) | `cmd_p` 그대로 (`inputs.RUNGS["geo"] = rung_cfg("geo")`를 실행 중에 등록; `refs="min"`; constrained key는 자기 기준 유지(F-7b)). 출력 `$GEO2/inputs/geo.parquet`, `results/P_build_geo.json` |
| T 검사 | `t2_g`, `t2_f`, `t3` 그대로 (작업 폴더만 `$GEO2/t2`, `$GEO2/t3`) |
| report | `results/F_report.json` + `results/F_G_status.csv` |

- **HF 추가 gate (새로):** geo의 `rel1`/`rel2`는 G2와 같은 좌표이므로, `frag1_rel`·`frag2_rel` HF 에너지(`e_frag1_rel_eh`, `e_frag2_rel_eh`)가 `labels_d1_hf_g2.parquet`와 **반응마다 ≤ 1e-6 Eh**여야 한다. 0.5 %를 넘는 반응이 어긋나면 STOP. HF 경로가 같은 좌표에서 같은 값을 내는지 확인하는 무료 검사다.
- **pilot:** H는 먼저 50반응(job_id 순서 첫 50개)을 돌려 반응당 코어시간을 재고 전체를 예상한다. **2,500코어시간 이하면** 나머지를 자동 진행하고, 넘으면 STOP. 참고: D1-G2 H는 8,616반응에 1,611코어시간(반응당 평균 0.187)이었다.

### 6-2. STOP 기준 (D1-G3 / LADDER G0·F0와 같음)
| 단계 | STOP |
|---|---|
| G | 원자 순서 실패 > 0.5 %; Freq 실패 > 1 %; B6 계산 불가 > 1 % (`ext_features.block_b6`로 확인) |
| F | F-1(182열 non-finite) > 1 %; Q-a > 1 %; T-a > 1 %; FU2 gate 자체의 STOP |
| H | 실행 실패 > 1 %; rel 에너지 gate 실패 > 0.5 % |
| P | gate P-a 실패 |

- n_imag가 0개이거나 2개 이상인 것은 실패가 아니다. 기록만 한다(LADDER G0-4와 같음; 예측 구조는 xTB 정류점이 아니다).

### 6-3. 실행 예 (로그인 노드는 sbatch만; 각 줄 전에 `squeue -u $USER -h -r | wc -l`로 MaxSubmit 20 여유 확인)
```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo2; L=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs
# G: 4 slices × 16 workers (D1 run env; feat.py G step와 같은 job 파일 형식으로 jobs/geo2.slurm을 만든다 = d1_g3/jobs/g3.slurm 복사, G3ENV → GEO2ENV)
sbatch -p cpu1,cpu2 -J geo2_g --time=48:00:00 --cpus-per-task=16 --mem=32G --array=0-3 --output=$L/%x_%A_%a.out \
  --export=ALL,GEO2ENV=d1 jobs/geo2.slurm feat_geo.py g A 4
# F: d1_pair g.slurm (PYENV=venv5, D1PAIR_PHASE=F), submit from analysis/d1_pair
(cd ../d1_pair && sbatch -p cpu1,cpu2 -J geo2_fs --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-3 \
  --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm ../geo2/feat_geo.py f_slice @I 4)
# H: d1_hf hf.slurm, 5 workers
(cd ../d1_hf && sbatch -p cpu1,cpu2 -J geo2_h --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-4 \
  --output=$L/%x_%A_%a.out jobs/hf.slurm ../geo2/feat_geo.py h_batch @I 5)
```
- `g.slurm`, `hf.slurm`의 기본 로그 폴더는 읽기 전용 scratch이므로 `--output`을 꼭 준다(D1-G3와 같음).
- 실제 job 파일 이름과 인자는 복사한 코드에 맞춘다.

## 7. Phase P 보고와 짝 집합

- `results/P_geo_report.json`:
  - P_geo = a·b 모두 geo 입력이 usable인 짝
  - **P_23 = P_g2 ∩ P_geo** (주 비교 집합), **P_023 = P_dft ∩ P_g2 ∩ P_geo** (회복률 집합)
  - 집합별 짝 수(기준·anchor·para), fold별 수, P_g2에 있는데 P_geo에 없는 짝과 이유
- **STOP:** |P_23| < 0.98 × |P_g2|. geo는 g2 위에 만든 것이므로 거의 같아야 한다.

## 8. Phase R·M3 — geo로 AW·GNN·TF·XGB 학습 (LADDER stage 3)

### 8-1. LADDER 코드에 rung `geo` 추가 (기본 동작은 그대로)

| 파일 | 추가 |
|---|---|
| `train_atoms.py` | `STAGE_RUNGS[3] = ("geo",)`, `RUNG_ORDER`에 `"geo"`, `RUNGS["geo"] = dict(fu2_res=<repo>/analysis/geo2/results/fu2_geo, tokens=$GEO2/fu2_geo/tokens, tag="geo")`, `inputs_path("geo") = $GEO2/inputs/geo.parquet` (다른 rung은 그대로 `$LAD/inputs/<rung>.parquet`) |
| `train_xgb.py` | `STAGE_RUNGS[3]`, `RUNGS["geo"] = dict(fu2_res=…/fu2_geo, x182=$GEO2/X182_d1_geo.parquet, tag="geo")`, `inputs_path` 같은 규칙 |
| `inputs.py` | `RUNGS["geo"]` = `feat_geo.rung_cfg("geo")`와 같은 값 (실행 중 등록이 아니라 고정) |
| 사전 등록 gate | stage 3은 ADD3 대신 **`analysis/geo2/results/PREREG_GEO2.md` + `prereg_geo2.json`**을 확인한다(`gate_add3`와 같은 방식: commit됨, sha256 일치, `inputs_sha256`에 geo 입력표·라벨 파일이 있음) |
| stage 3 검사 gate | `results/C_tests.json` 대신 `analysis/geo2/results/C_tests_geo.json` all_pass를 확인한다 |
| collect | `collect --stage 3`은 **geo 행만** `results/L_predictions_geo.parquet`, `L_report_geo.json`, `L_xgb_report_geo.json`에 쓴다. 기존 `L_predictions.parquet`, `L_report.json`, `L_xgb_report.json`은 건드리지 않는다 |
| unit | `$LAD/units/L__geo__<arm>__<model>__f<k>.*` 새 파일만 만든다. 기존 unit 파일은 열기만 한다 |

- 쓰기 허용 경로(`lad_common.assert_path`)에 `$GEO2`가 필요하면 추가한다. 기본값은 그대로 둔다.
- 모델 구조·grid·학습 규칙(`fu2/models.py`, `fu2/train.py`, `train_ml_single.GRIDS["XGB"]`)은 **한 글자도 바꾸지 않는다.** rung 목록과 경로만 추가한다.
- 절대 경로가 다른 worktree에서 돌기 때문에 지문(fingerprint) 비교가 어긋나면, D1-G3 결정 F-13처럼 **repo 상대 경로로 비교**하고 기록한다.

### 8-2. 검사 `C_tests_geo.json` (`tests/run_tests.py`를 rung geo로; 하나라도 실패하면 STOP)

| # | 검사 | 기준 |
|---|---|---|
| L-1 | 반대칭: 4 모델 × 2 arm, rung geo | ≤ 1e-4 kcal/mol |
| L-2, L-3, L-7, L-8, L-9 | LADDER와 같음, rung geo | LADDER 기준 그대로 |
| L-4 | 단계 분리: geo 학습 process가 연 입력 파일 | geo 열의 파일(`$GEO2/...`)과 공통 파일(라벨, fold)만. dft·g1 입력과 G1 폴더 경로 0개 |
| L-5 | H 일치: 무작위 50반응의 H 9값 = geo 입력표 | 1e-12 |
| L-11 | y = `F_pairs_cr2.csv` | 1e-12 |
| **회귀 1** | `train_atoms.py list --stage 1`, `--stage 2`의 unit 목록이 수정 전과 같다 | 문자열 같음 |
| **회귀 2** | L-6, L-10을 수정한 코드로 다시 돌림 | LADDER 기준 그대로 통과 |
| **회귀 3** | g2·S XGB fold 0을 수정한 코드로 다시 학습·예측 | `L_predictions.parquet` g2·S·XGB fold 0과 ≤ 1e-6 |
| smoke | rung geo, fold 0: AW·GNN·TF는 구성마다 2 epoch(`train_atoms.py smoke`), XGB는 `train_xgb.py smoke` | 4 모델 모두 gate M-1′ 통과 |

### 8-3. 사전 등록 `PREREG_GEO2.md` + `prereg_geo2.json` (학습 전에 commit·push)

**담을 것:**
1. 0-2절 rung `geo` 정의, 4절 상식 검사와 대체 규칙, Y 설정
2. 학습 규칙은 LADDER 4-2·4-3 그대로: arm S·SH, AW·GNN·TF·XGB, grid, β 규칙, seed 5개, 300 epoch, patience 30, T1 fold
3. 짝 집합과 sha256(P_geo, P_23, P_023), 라벨 파일 `F_pairs_cr2.csv` sha256, geo 입력표 sha256
4. 9–10절의 지표와 판정 규칙 (REP·REP8 정의 포함)
5. **이미 본 결과도 그대로 적는다:** GEO1 시험 결과(짝 비율 0.142 → 0.347 등), 작성자의 에너지 환산 예측(같은 형태 짝 oi −30 % 안팎, 전체 −10 % 안팎; 추정치)
6. 알려진 한계: 1단계 학습 데이터의 fold 교차 의존(10-3절)

### 8-4. 학습 실행
- 단위: **2 arm(S, SH) × 4 모델(AW, GNN, TF, XGB) × 5 fold = 40 unit.** AW·GNN·TF는 GPU(30 unit; unit마다 튜닝 4칸 + seed 5개, LADDER와 같음), XGB는 CPU(10 unit). **4개 모두 학습해야 E로 간다.** 하나라도 빠지면 REP를 고를 수 없으므로 STOP.
- 예상: LADDER rung 하나가 14(g2)–49(dft) GPU시간이었다. smoke로 다시 예상해서 **80 GPU시간을 넘으면 STOP**.
- LADDER와 같은 명령, `--stage 3`:
```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/d1_ladder
sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 -J geo2_m3 --time=48:00:00 --array=0-3 \
  --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%A_%a.out jobs/gpu.slurm train_atoms.py run @I 4 --stage 3
(cd ../d1_pair && sbatch -p cpu1,cpu2 -J geo2_x3 --time=48:00:00 --cpus-per-task=32 --mem=64G \
  --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py run 0 1 --stage 3)
# 끝나면: train_atoms.py collect --stage 3 --with-xgb
```
- 실제 인자는 LADDER 코드의 CLI에 맞춘다.

## 9. Phase E — 평가 (`analysis/geo2/eval_geo2.py run`)

### 9-1. 선택(selection) 정의 — 모두 검증 MAE로만 고른다
| 이름 | 정의 |
|---|---|
| REP·SH, REP·S | LADDER 13-1: rung × arm × 타깃마다 4 모델 중 5 fold 평균 검증 kcal MAE가 가장 낮은 것. 동점이면 AW → GNN → TF → XGB |
| **REP8** | rung × 타깃마다 2 arm × 4 모델(8개) 중 검증 MAE가 가장 낮은 것. 동점이면 SH 먼저, 그다음 모델 순서. **주 판정에 쓴다** |
| ENS4·SH, ENS4·S | 4 모델 시험 예측의 같은 비중 평균 |

- 검증 MAE는 unit json의 `val_kcal_mae`(5 fold 평균)다. 시험 점수로 고르지 않는다.

### 9-2. 출력
| 파일 | 내용 |
|---|---|
| `E_metrics.csv` | 짝 집합(P_23, P_023, 각 rung 자기 집합) × rung × 선택 × 10 타깃 × 위치 → n, MAE, para는 core 단위 95 % CI(2,000회, seed 20261019) |
| `E_compare.csv` | geo − g2 (P_23), geo − dft (P_023): 선택 × 타깃, para MAE 차이와 core bootstrap CI(LADDER `boot_diff`) |
| `E_recovery.csv` | 회복률 = (MAE_g2 − MAE_geo) ÷ (MAE_g2 − MAE_dft), P_023 para |
| `E_class.csv` | REP8, 짝의 TS 분류(same / mixed / different; C_compare `same_ts_g1`)별 para MAE |
| `E_app.csv` | barrier 응용 지표: 같은 계열 안 순서 정답률(실제 차이 ≥ 0.41 / 1.0 / 1.36 kcal/mol = 2배 / 약 5배 / 10배), 계열 Spearman 중앙값, 위치 선택성 변화 MAE·방향 정답률 |
| `E_report.json` | 짝 수, REP·REP8 목록(타깃별로 4 모델 중 무엇이 골라졌는지), 판정 |

- **4 모델 각각의 결과:** `eval_geo2.py`는 REP·REP8·ENS4를 낸다. 모델별(AW, GNN, TF, XGB) para MAE와 CI는 `evaluate.py`의 E2 계산을 rung geo에 적용한 `e3`가 낸다(`E3_metrics.csv`, E2_metrics와 같은 형식).
- LADDER 단계 2 목표(oi·cpcm para MAE ≤ 0.13 + 꺾임 민감도 ≥ 0.7 + 위양성률 ≤ 0.1)의 꺾임 부분은 `evaluate.py`의 E2 계산을 rung geo에 적용해서 낸다(`e3` 명령을 E2와 같은 방식으로 추가; 출력 `E3_goal.csv`, `E3_bends.csv`). MAE 부분은 `eval_geo2.py`의 `goal_mae`다.

`analysis/geo2/eval_geo2.py` (그대로 작성):

```python
#!/usr/bin/env python3
"""GEO2 Phase E: energy (8 EDA channels + c_ghost + barrier) of the geo rung against dft / g1 / g2 (docs/specs/GEO2.md 9).

  python eval_geo2.py selftest   # g2 predictions relabelled as rung 'geo': must reproduce E2 numbers exactly, diff = 0
  python eval_geo2.py run        # results/E_metrics.csv, E_compare.csv, E_recovery.csv, E_class.csv, E_app.csv,
                                 # E_report.json
Reads only; any pandas + pyarrow + numpy + scipy env (LADDER evaluate env 'espley' or reactot).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
BASE_PRED = LAD_RES / "L_predictions.parquet"
BASE_UNITS = (LAD_RES / "L_report.json", LAD_RES / "L_xgb_report.json")
GEO_PRED = LAD_RES / "L_predictions_geo.parquet"                       # stage-3 collect (spec 8-1)
GEO_UNITS = (LAD_RES / "L_report_geo.json", LAD_RES / "L_xgb_report_geo.json")
E2_REPORT, E2_METRICS = LAD_RES / "E2_report.json", LAD_RES / "E2_metrics.csv"
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
MAIN = ("oi", "cpcm")
MODELS = ("AW", "GNN", "TF", "XGB")                 # LADDER 13-1 tie order
ARMS = ("SH", "S")                                  # REP8 tie order: SH first (spec 9-1)
N_BOOT, SEED = 2000, 20261019                       # = d1pair_common N_BOOT / SEED (LADDER spec 11)
GOAL_MAE = 0.13                                     # LADDER 13-2
APP_T = (0.41, 1.0, 1.36)                           # RT ln2, 1 kcal/mol, RT ln10 at 298 K (kcal/mol)


# ---------------------------------------------------------------- statistics (= d1_eval.boot_diff, copied verbatim)
def boot_diff(dx, db, cores, n_boot=N_BOOT, seed=SEED):
    """d = MAE(x) - MAE(b) with the cores resampled: (d, ci_lo, ci_hi, p)."""
    ex, eb = np.abs(dx), np.abs(db)
    uc, inv = np.unique(cores, return_inverse=True)
    sx, sb, cnt = np.bincount(inv, ex), np.bincount(inv, eb), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(uc)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    num = w @ cnt
    dstar = (w @ sx) / num - (w @ sb) / num
    d = ex.mean() - eb.mean()
    lo, hi = np.percentile(dstar, [2.5, 97.5])
    p = min(1.0, 2 * min((1 + (dstar <= 0).sum()) / (n_boot + 1), (1 + (dstar >= 0).sum()) / (n_boot + 1)))
    return float(d), float(lo), float(hi), float(p)


# ---------------------------------------------------------------- inputs
def load_units(paths):
    """unit json records (L_report / L_xgb_report 'units') -> rows rung, arm, model, fold, target, val."""
    rows = []
    for p in paths:
        for name, u in json.loads(Path(p).read_text())["units"].items():
            _, rung, arm, model, f = name.split("__")
            for t, v in (u.get("val_kcal_mae") or {}).items():
                rows.append(dict(rung=rung, arm=arm, model=model, fold=int(f[1:]), target=t, val=float(v)))
    U = pd.DataFrame(rows)
    n = U.groupby(["rung", "arm", "model", "target"]).fold.nunique()
    if (n != 5).any():
        raise SystemExit(f"STOP: units without 5 folds: {n[n != 5].index.tolist()[:5]}")
    return U


def reps(U):
    """REP (LADDER 13-1: per rung x arm x target the lowest 5-fold mean val kcal MAE, ties AW -> GNN -> TF -> XGB) and
    REP8 (per rung x target over both arms; ties SH before S, then the model order)."""
    m = U.groupby(["rung", "arm", "model", "target"]).val.mean()
    rep, rep8 = {}, {}
    for (rung, arm, t), s in m.groupby(level=["rung", "arm", "target"]):
        v = {k[2]: float(x) for k, x in s.items()}
        rep[(rung, arm, t)] = min((v[mo], MODELS.index(mo), mo) for mo in v)[2]
    for (rung, t), s in m.groupby(level=["rung", "target"]):
        v = {(k[1], k[2]): float(x) for k, x in s.items()}
        rep8[(rung, t)] = min((x, ARMS.index(a), MODELS.index(mo), a, mo) for (a, mo), x in v.items())[3:]
    return rep, rep8, m


def pairs_frame():
    F = pd.read_csv(F_PAIRS)
    keep = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main", "core", "dipole", "skeleton", "series", "fam",
            "cfg", "orient", "face", "c3", "Xb"]
    F = F[[c for c in keep if c in F.columns]].copy()
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "same_ts_g1"]).set_index("rxn_id").same_ts_g1
    C = C.map({True: True, False: False, "True": True, "False": False})
    sa, sb = C.reindex(F.a_rxn).values, C.reindex(F.b_rxn).values
    F["cls"] = np.where((sa == True) & (sb == True), "same",                         # noqa: E712
                        np.where((sa == False) & (sb == False), "different", "mixed"))  # noqa: E712
    return F


def load_preds(selftest):
    L = pd.read_parquet(BASE_PRED)
    if selftest:
        G = L[L.rung == "g2"].copy()
        G["rung"] = "geo"
    else:
        G = pd.read_parquet(GEO_PRED)
        if set(G.rung) != {"geo"} or list(G.columns) != list(L.columns):
            raise SystemExit("STOP: L_predictions_geo.parquet must hold rung 'geo' only with the LADDER columns")
    return pd.concat([L, G], ignore_index=True)


def select(L, rung, sel, t, rep, rep8):
    """per-pair test prediction of one selection: ('REP', arm) | ('REP8',) | ('ENS4', arm) | (model, arm)."""
    d = L[(L.rung == rung) & (L.target == t)]
    if sel[0] == "REP8":
        arm, mo = rep8[(rung, t)]
        d = d[(d.arm == arm) & (d.model == mo)]
    elif sel[0] == "REP":
        d = d[(d.arm == sel[1]) & (d.model == rep[(rung, sel[1], t)])]
    elif sel[0] == "ENS4":
        d = d[d.arm == sel[1]].groupby("pair_id").agg(yhat=("yhat", "mean"), y=("y", "first"), n=("model", "nunique"))
        if (d.n != 4).any():
            raise SystemExit(f"STOP: ENS4 {rung} {sel} {t}: a pair without 4 model predictions")
        return d[["y", "yhat"]]
    else:
        d = d[(d.arm == sel[1]) & (d.model == sel[0])]
    if d.pair_id.duplicated().any():
        raise SystemExit(f"STOP: duplicated pair predictions {rung} {sel} {t}")
    return d.set_index("pair_id")[["y", "yhat"]]


SELS = [("REP8",), ("REP", "SH"), ("REP", "S"), ("ENS4", "SH"), ("ENS4", "S")]
SEL_NAME = {("REP8",): "REP8", ("REP", "SH"): "REP·SH", ("REP", "S"): "REP·S", ("ENS4", "SH"): "ENS4·SH",
            ("ENS4", "S"): "ENS4·S"}


def common_pairs(L, F, rungs):
    """P_main reference pairs predicted in every rung of 'rungs' (all targets, all models)."""
    ref = set(F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].pair_id)
    s = ref
    for r in rungs:
        s = s & set(L[L.rung == r].pair_id)
    return sorted(s)


# ---------------------------------------------------------------- tables
def metrics(L, F, rep, rep8, rungs, pset, name):
    Fi = F.set_index("pair_id")
    rows = []
    for rung in rungs:
        for sel in SELS:
            for t in TARGETS:
                d = select(L, rung, sel, t, rep, rep8).reindex(pset)
                if d.yhat.isna().any():
                    raise SystemExit(f"STOP: {rung} {sel} {t} lacks predictions on {name}")
                e = (d.yhat - d.y).values
                pos, core = Fi.pos.reindex(pset).values, Fi.core.reindex(pset).values
                for p in ("all", "p", "m", "o", "direct"):
                    m = np.ones(len(e), bool) if p == "all" else pos == p
                    if not m.any():
                        continue
                    r = dict(pair_set=name, rung=rung, sel=SEL_NAME[sel], target=t, pos=p, n=int(m.sum()),
                             mae=float(np.abs(e[m]).mean()))
                    if p == "p":
                        _, lo, hi, _ = boot_diff(e[m], np.zeros(m.sum()), core[m])
                        r.update(n_cores=int(len(set(core[m]))), mae_ci_lo=lo, mae_ci_hi=hi)
                    rows.append(r)
    return pd.DataFrame(rows)


def compare(L, F, rep, rep8, x, b, pset, name):
    """d = MAE(x) - MAE(b), para pairs of pset, core bootstrap (LADDER compare convention)."""
    Fi = F.set_index("pair_id")
    para = [p for p in pset if Fi.at[p, "pos"] == "p"]
    rows = []
    for sel in SELS:
        for t in TARGETS:
            dx = select(L, x, sel, t, rep, rep8).reindex(para)
            db = select(L, b, sel, t, rep, rep8).reindex(para)
            ex, eb = (dx.yhat - dx.y).values, (db.yhat - db.y).values
            d, lo, hi, p = boot_diff(ex, eb, Fi.core.reindex(para).values)
            rows.append(dict(pair_set=name, x=x, b=b, sel=SEL_NAME[sel], target=t, n=len(para), mae_x=float(np.abs(ex).mean()),
                             mae_b=float(np.abs(eb).mean()), d=d, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


def recovery(M):
    """(MAE_g2 - MAE_geo) / (MAE_g2 - MAE_dft) on P_023 para (REP8 and per-arm REP)."""
    m = M[(M.pair_set == "P_023") & (M.pos == "p")].pivot_table(index=["sel", "target"], columns="rung", values="mae")
    m["recovery"] = (m["g2"] - m["geo"]) / (m["g2"] - m["dft"])
    return m.reset_index()


def by_class(L, F, rep, rep8, pset):
    Fi = F.set_index("pair_id")
    para = [p for p in pset if Fi.at[p, "pos"] == "p"]
    rows = []
    for rung in ("dft", "g1", "g2", "geo"):
        if rung not in set(L.rung):
            continue
        for t in TARGETS:
            d = select(L, rung, ("REP8",), t, rep, rep8).reindex(para)
            e = np.abs(d.yhat - d.y).values
            for cls in ("same", "mixed", "different"):
                m = Fi.cls.reindex(para).values == cls
                rows.append(dict(rung=rung, sel="REP8", target=t, cls=cls, n=int(m.sum()), mae=float(e[m].mean())))
    return pd.DataFrame(rows)


def app_metrics(L, F, rep, rep8, pset):
    """barrier application metrics (post hoc in REPORT_D1LADDER 3; here fixed before the geo results): within-family
    order accuracy for true |gap| >= 0.41 / 1.0 / 1.36 kcal/mol, family Spearman median, regio change MAE / direction."""
    from scipy.stats import spearmanr
    Fi = F.set_index("pair_id")
    rows = []
    for rung in ("dft", "g1", "g2", "geo"):
        if rung not in set(L.rung):
            continue
        for sel in (("REP8",), ("ENS4", "SH"), ("ENS4", "S")):
            d = select(L, rung, sel, "barrier", rep, rep8).reindex(pset).join(Fi, how="left")
            acc = {t: [] for t in APP_T}
            rho = []
            for _, h in d.groupby("fam"):
                y, p = np.r_[h.y.values, 0.0], np.r_[h.yhat.values, 0.0]   # the family reference (H) = 0
                if len(y) >= 8:
                    rho.append(spearmanr(y, p)[0])
                i, j = np.triu_indices(len(y), 1)
                dy, dp = y[i] - y[j], p[i] - p[j]
                for t in APP_T:
                    k = np.abs(dy) >= t
                    acc[t].append(np.sign(dy[k]) == np.sign(dp[k]))
            key = ["dipole", "skeleton", "series", "cfg", "face", "c3", "Xb", "pos"]
            G = d.assign(o=d.orient.fillna("_"))
            rg = []
            for kk, h in G.groupby(key, dropna=False):
                if len(h) == 2 and h.o.nunique() == 2:
                    h = h.sort_values("o")
                    rg.append((kk[-1], h.y.iloc[0] - h.y.iloc[1], h.yhat.iloc[0] - h.yhat.iloc[1]))
            rg = pd.DataFrame(rg, columns=["pos", "dy", "dp"])
            e = (rg.dp - rg.dy).abs()
            big = rg.dy.abs() >= APP_T[0]
            rows.append(dict(rung=rung, sel=SEL_NAME[sel], n_pairs=len(d),
                             **{f"order_acc_{t}": float(np.concatenate(acc[t]).mean()) for t in APP_T},
                             spearman_fam_median=float(np.median(rho)) if rho else np.nan, regio_n=len(rg),
                             regio_mae=float(e.mean()) if len(rg) else np.nan,
                             regio_dir_acc_2x=float((np.sign(rg.dp[big]) == np.sign(rg.dy[big])).mean()) if big.any()
                             else np.nan))
    return pd.DataFrame(rows)


def verdict(C):
    """spec 10-1: REP8, P_23 para, geo vs g2: both oi and cpcm ci_hi < 0 -> improved; one -> partial; none -> not."""
    c = C[(C.pair_set == "P_23") & (C.sel == "REP8") & (C.target.isin(MAIN))].set_index("target")
    k = int((c.ci_hi < 0).sum())
    return dict(rule="REP8 para P_23: ci_hi(MAE_geo - MAE_g2) < 0 for oi and cpcm", n_main_improved=k,
                verdict={2: "개선", 1: "부분", 0: "불가"}[k],
                detail={t: {kk: float(c.at[t, kk]) for kk in ("mae_x", "mae_b", "d", "ci_lo", "ci_hi")} for t in MAIN})


def goal(M):
    """LADDER 13-2 MAE part (oi, cpcm para MAE <= 0.13) for geo; bend sensitivity / FPR come from evaluate.py e3 (spec 9-2)."""
    m = M[(M.pair_set == "own_geo") & (M.pos == "p") & (M.target.isin(MAIN)) & (M.rung == "geo")]
    return {s: {t: float(g[g.target == t].mae.iloc[0]) for t in MAIN} | {"mae_ok": bool((g.mae <= GOAL_MAE).all())}
            for s, g in m.groupby("sel")}


# ---------------------------------------------------------------- selftest checks against E2
def selftest_checks(M, C, rep):
    E2 = json.loads(E2_REPORT.read_text())["representatives"]
    bad = []
    for (rung, arm, t), mo in rep.items():
        if rung in E2 and E2[rung][arm][t]["model"] != mo:
            bad.append(f"REP {rung} {arm} {t}: {mo} != E2 {E2[rung][arm][t]['model']}")
    EM = pd.read_csv(E2_METRICS)
    EM = EM[(EM.pair_set == "own") & (EM.set == "P_main") & (EM.pos == "p")]
    own = M[(M.pair_set == "own_g2") & (M.pos == "p") & (M.rung == "g2")]
    n_cmp = 0
    for s_my, s_e2, arm in (("REP·SH", "REP", "SH"), ("REP·S", "REP", "S"), ("ENS4·SH", "ENS4", "SH"),
                            ("ENS4·S", "ENS4", "S")):
        for t in TARGETS:
            a = own[(own.sel == s_my) & (own.target == t)]
            b = EM[(EM.rung == "g2") & (EM.arm == arm) & (EM.model == s_e2) & (EM.target == t)]
            if len(a) != 1 or len(b) != 1:
                bad.append(f"missing row {s_my} {t}")
                continue
            n_cmp += 1
            if int(a.n.iloc[0]) != int(b.n.iloc[0]) or abs(a.mae.iloc[0] - b.mae.iloc[0]) > 1e-9 or \
                    abs(a.mae_ci_lo.iloc[0] - b.mae_ci_lo.iloc[0]) > 1e-9 or abs(a.mae_ci_hi.iloc[0] - b.mae_ci_hi.iloc[0]) > 1e-9:
                bad.append(f"E2 mismatch {s_my} {t}: {a[['n', 'mae', 'mae_ci_lo', 'mae_ci_hi']].values} vs "
                           f"{b[['n', 'mae', 'mae_ci_lo', 'mae_ci_hi']].values}")
    z = C[(C.pair_set == "P_23")]
    if (z.d.abs() > 0).any() or (z.ci_lo != 0).any() or (z.ci_hi != 0).any():
        bad.append("geo == g2 copy but some difference != 0")
    return dict(n_e2_rows_compared=n_cmp, problems=bad[:30], all_pass=not bad)


# ---------------------------------------------------------------- main
def main(mode):
    selftest = mode == "selftest"
    L = load_preds(selftest)
    F = pairs_frame()
    U = load_units(BASE_UNITS)
    if selftest:
        G = U[U.rung == "g2"].copy()
        G["rung"] = "geo"
        U = pd.concat([U, G], ignore_index=True)
    else:
        U = pd.concat([U, load_units(GEO_UNITS)], ignore_index=True)
    rep, rep8, mval = reps(U)
    P23 = common_pairs(L, F, ("g2", "geo"))
    P023 = common_pairs(L, F, ("dft", "g2", "geo"))
    own = {r: common_pairs(L, F, (r,)) for r in ("dft", "g1", "g2", "geo")}
    M = pd.concat([metrics(L, F, rep, rep8, ("g2", "geo"), P23, "P_23"),
                   metrics(L, F, rep, rep8, ("dft", "g2", "geo"), P023, "P_023")]
                  + [metrics(L, F, rep, rep8, (r,), own[r], f"own_{r}") for r in own], ignore_index=True)
    C = pd.concat([compare(L, F, rep, rep8, "geo", "g2", P23, "P_23"),
                   compare(L, F, rep, rep8, "geo", "dft", P023, "P_023")], ignore_index=True)
    out = RES / ("selftest" if selftest else "")
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "E_metrics.csv", index=False)
    C.to_csv(out / "E_compare.csv", index=False)
    recovery(M).to_csv(out / "E_recovery.csv", index=False)
    by_class(L, F, rep, rep8, P023).to_csv(out / "E_class.csv", index=False)
    app_metrics(L, F, rep, rep8, P023).to_csv(out / "E_app.csv", index=False)
    rep_out = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rep.items()}
    rep8_out = {f"{k[0]}|{k[1]}": list(v) for k, v in rep8.items()}
    report = dict(mode=mode, n_pairs=dict(P_23=len(P23), P_023=len(P023), **{f"own_{r}": len(v) for r, v in own.items()}),
                  n_para=dict(P_23=int(F.set_index("pair_id").pos.reindex(P23).eq("p").sum()),
                              P_023=int(F.set_index("pair_id").pos.reindex(P023).eq("p").sum())),
                  representatives=rep_out, rep8=rep8_out, verdict=verdict(C), goal_mae=goal(M))
    if selftest:
        report["selftest"] = selftest_checks(M, C, rep)
    (out / "E_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("n_pairs", "verdict")} | ({"selftest": report.get("selftest")}
                                                                        if selftest else {}), indent=1, default=float))
    return 3 if selftest and not report["selftest"]["all_pass"] else 0


if __name__ == "__main__":
    if sys.argv[1:] in (["selftest"], ["run"]):
        sys.exit(main(sys.argv[1]))
    raise SystemExit(__doc__)
```


## 10. 판정 (결과 보기 전에 고정; PREREG_GEO2에 그대로 적는다)

### 10-1. 주 판정 — 배포 파이프라인(G2 → 보정 구조 → 에너지)이 G2 직접 예측보다 나은가
- 집합 P_23, para P_main 기준 짝, **REP8**(타깃마다 2 arm × 4 모델 중 검증 MAE 최저), d = MAE_geo − MAE_g2, core bootstrap 95 % CI.

| 판정 | 조건 |
|---|---|
| **개선** | oi와 cpcm **둘 다** CI 상한 < 0 |
| 부분 | 둘 중 하나만 CI 상한 < 0 |
| 불가 | 둘 다 아님 |

- **목표 도달**(따로 표시): geo의 REP8 oi·cpcm para MAE ≤ 0.13이고 E3 꺾임 기준도 통과.

### 10-2. 설명용 (판정 아님)
- 10 타깃 전부의 차이와 CI, 회복률, TS 분류별 결과, 응용 지표, REP·SH/REP·S/ENS4별 결과
- 4 모델(AW, GNN, TF, XGB) 각각의 geo 대 g2 결과 (`E3_metrics.csv`). 개별 모델의 통과는 설명용으로만 적는다(가장 좋은 것을 고르면 사후 선택이 된다; LADDER 13-2와 같음)

### 10-3. 꼭 적을 한계
- **1단계 학습 데이터의 fold 교차 의존:** fold k를 시험할 때, 학습 짝의 구조를 만든 GEO1 모델(다른 fold의 final 모델)은 fold k 반응의 DFT 구조를 학습에 썼다. 시험 짝의 구조와 에너지 라벨은 쓰지 않았으므로 직접 누수는 아니다. 완전히 막으려면 1단계를 중첩 교차검증으로 약 20회 더 학습해야 하며, 이번에는 하지 않았다.
- TS 형태 불일치(G2가 다른 형태를 찾은 반응)는 이 파이프라인이 고치지 않는다.
- 반응물 기준 구조는 G2(xTB) 그대로다.
- 예측 구조는 xTB 정류점이 아니다(n_imag 분포를 보고).

## 11. Phase S — `results/SUMMARY_GEO2.md` (한국어) 후 **멈춤**

- 맨 앞 네 문장:
  1. 주 판정(11-1)과 oi·cpcm의 g2 → geo MAE와 차이 CI
  2. 회복률(oi, cpcm, barrier)
  3. TS 같은 형태 짝에서의 결과(설명용)
  4. 타깃별로 고른 모델(REP8)과 4 모델 각각의 oi·cpcm 결과
- 절: 시작 조건 → Y → Z → T → G·F·H·P → R(commit hash) → M3 → E → 판정 → 비용 → 한계
- 그림 `results/figures/`:
  - `figG2_ladder`: oi·cpcm·barrier para MAE 대 rung(dft, g1, g2, geo), REP8, CI 막대
  - `figG2_class`: TS 분류별 MAE (g2 대 geo)
  - `figG2_models`: oi·cpcm para MAE, 4 모델 × 2 arm, g2 대 geo
- commit·push 후 사용자 지시를 기다린다.

## 12. 계산량·상한

| 항목 | 예상 | 상한 (넘으면 STOP) |
|---|---|---|
| Y: GEO1 추가 모델 3 seed | GEO1 final 1회 × 3 | 30 GPU시간 |
| G: xTB Freq | 약 15코어시간 (LADDER G0: 9,266반응 14코어시간) | — |
| F: feature·토큰 | 수십 코어시간 (LADDER F0 규모) | — |
| H: HF EDA | 약 1,600코어시간 (D1-G2 H 기준) | 2,500코어시간 (pilot 50반응으로 예상) |
| M3: 학습 (AW·GNN·TF GPU, XGB CPU) | 15–50 GPU시간 + XGB CPU 약 10코어시간 (LADDER rung 하나: g2 14, dft 49 GPU시간; GNN이 대부분) | 80 GPU시간 (smoke로 예상) |

## 13. HPC 규칙
- python, ORCA, xtb, RDKit은 모두 `sbatch`(`--time=48:00:00`)로만 돌린다. 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만 한다.
- MaxJobs 10, MaxSubmit 20(배열 원소 포함).
- **읽기만 하는 것:** GEO1(`$GEO`, `analysis/geo/results`; Y의 새 파일 제외), LADDER 결과 파일과 기존 unit, D1-G2, D1-HF, D1-G3, D1 run·job 폴더, `labels_d1.json`, `F_pairs*.csv`
- **새 파일은 여기에만 쓴다:** `analysis/geo2/`, `$GEO2`, `analysis/geo/train_m1.py`·`$GEO/preds/final_fm1_*`·`analysis/geo/results/Y_plan.json`, LADDER의 `*_geo.*` 새 결과 파일과 `$LAD/units/L__geo__*`, 8-1절의 코드 추가
- 원자적 쓰기, 멱등. 수치는 파일에서 옮긴다.
- 결과를 본 뒤 판정·선택 규칙·짝 집합을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

## 14. 결정 기록과 STOP

- 명령서에 없는 갈림길이 생기면 추천안을 고르고 `results/DECISIONS.md`에 적은 뒤 진행한다.

| 단계 | STOP 조건 |
|---|---|
| 0 | 1절 시작 조건 중 하나라도 아님 (특히 0-6 selftest) |
| Y | 예상 > 30 GPU시간; fold −1 반응이 학습 목록에 섞임 |
| Z | `E_rxn.csv` 재현 실패; 대체 > 2 %; 한 반응이 두 fold에서 예측됨 |
| T | T-2–T-4 중 하나라도 실패 |
| G·F·H·P | 6-2절 기준; H 예상 > 2,500코어시간 |
| P | \|P_23\| < 0.98 × \|P_g2\| |
| R·M3 | PREREG commit 전 학습 시도; 8-2 검사 실패; 예상 > 80 GPU시간; 캐시 지문 불일치; 40 unit 중 하나라도 없음; 기존 LADDER 결과 파일이 바뀜(sha256 전후 비교) |
| E | selftest 실패; 비교하는 rung 사이 시험 짝 목록 불일치 |
| 언제든 | 라벨·기존 결과 파일을 바꿔야 할 것 같음; 이 문서에 없는 계산 방법 변경이 필요해 보임 |

STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.
