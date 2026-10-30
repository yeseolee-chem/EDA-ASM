# GEO1B — G1 없이 G2 전부로 구조 보정 다시 학습(GEO1b) → 에너지(GEO2b), 그리고 G2 형태 전체 재탐색(G2c) — Claude Code 실행 지시서

**저장소·worktree**
- **A 트랙(GPU → 에너지):** GEO1·GEO2를 돌린 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1`을 쓴다. 브랜치 `geo1b`(`origin/geo1`에서)를 만들고, GEO1b가 끝나면 `geo2b`(`geo1b`에서)를 만든다.
- **B 트랙(CPU, G2c):** 새 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c`, 브랜치 `g2c`(`origin/geo1`에서)를 쓴다.
  - 이유: G2c 계산 job은 반응마다 코드를 다시 읽는다. A 트랙에서 브랜치를 바꿔도 돌고 있는 CPU job에 영향이 없도록 worktree를 나눈다.

**이 파일:** `docs/specs/GEO1B.md`로 `geo1b`와 `g2c` 두 브랜치에 모두 commit한다.

**scratch 경로**

| 이름 | 경로 |
|---|---|
| `$GEO1B` | `/gpfs/tmp_cpu2/yeseo1ee/geo1b` |
| `$GEO2B` | `/gpfs/tmp_cpu2/yeseo1ee/geo2b` |
| `$G2C` | `/gpfs/tmp_cpu2/yeseo1ee/g2c` |

- 기존 `$GEO`(`…/geo`), `$GEO2`(`…/geo2`), `d1_g2`, `d0_g2`는 **읽기만** 한다. 아무것도 지우지 않는다.

**순서:** 0 → 두 트랙을 동시에 진행한다.
- **A 트랙:** D → U → R → E1·E2 → 요약 → GEO2b (GEO2.md 전체) → 요약
- **B 트랙 (0 직후 시작):** 준비 → 재실행 4개 root → build → report → **멈춤**

**멈추는 조건과 보고 원칙**
- 7절의 STOP 조건에서만 멈추고 묻는다.
- 버그를 찾으면 **고치기 전에** 보고한다.
- 계산 설정과 판정 규칙은 바꾸지 않는다. 바꾼 코드는 그 결과 폴더의 `DECISIONS.md`에 적는다.

---

## 0. 무엇을, 왜

### 0-1. 사용자 결정 (2026-10-30)

| # | 결정 |
|---|---|
| 1 | **G1은 앞으로 없는 것으로 한다.** 학습, 평가, 판정 어디에서도 G1 좌표와 G1에서 나온 값을 읽지 않는다. 여기에는 `C_compare.same_ts_g1`, `d1_pair/g1_final`, `espley_xtb_g1`이 포함된다. GEO2b 에너지 비교에서도 LADDER `g1` rung을 뺀다 |
| 2 | 구조 보정 모델은 **G2 입력 전부**로 학습한다. 같은/다른 형태로 걸러내지 않는다. "쉬운 데이터만 골랐다"는 비판을 막기 위해서다 |
| 3 | 기존 GEO2는 **중단**한다. G1을 학습에 쓴 GEO1 구조를 기반으로 하기 때문이다 |
| 4 | **G2c**(G2 TS 형태 재탐색)는 시범 없이 **D1·D0 전체**를 처음부터 계산한다 |
| 5 | 최종 판정은 **에너지**(GEO2b, GEO2.md 10절 규칙 그대로)로 한다. GEO1b 구조 지표는 설명용이다 |
| 6 | 학습 설정은 **GEO1 실제 실행값 그대로** 쓴다: MAX_EP 300, patience 없음, AdamW 2e-4, warmup 5 % 후 cosine (horizon 300 epoch), R-select 11-epoch 창, R-once launcher, R-lpt. epoch와 LR은 바꾸지 않는다(0-4) |

### 0-2. GEO1 → GEO1b에서 바뀌는 것

| 항목 | GEO1 | **GEO1b** |
|---|---|---|
| 학습 예제 | G1 → 라벨 (전부) + G2 → 라벨 (`same_ts_g1 == True`만) | **G2 → 라벨, G2가 있는 학습 반응 전부** (D1 학습 fold + D0, D0 겹침 5개 제외) |
| 기록의 `same` / `xyz_g1` | C_compare / G1 폴더에서 읽음 | 항상 `None` (읽지 않음) |
| 주 지표 (내부 검증·시험) | **같은 형태** para 짝(840)의 짝 오차 ≤ 0.005 Å 비율 | **모든** para P_main 짝(시험 1,667짝)의 짝 오차 ≤ 0.005 Å 비율 |
| 형태별 분석 | same / mixed / different (G1 기반) | **G2 대 라벨의 대칭 고려 무거운 원자 RMSD 구간**으로 나눔: low ≤ 0.25 < mid ≤ 0.75 < high (Å). 설명용으로만 씀 |
| ablation | no_d0, q1_only, r0 | 없음 |
| 판정 | 구조 (pass / partial / fail) | 구조는 판정 없음. **GEO2b 에너지로 판정** |
| 모델, 손실, 가짜 궤적, 설정 격자 c0–c3, seed 3개, 앙상블 | — | **그대로** |

### 0-3. G2c (B 트랙)
- **목적:** G2의 TS 형태가 라벨과 어긋나는 문제를 구조 보정 모델 대신 **입력 단계에서** 줄일 수 있는지 본다.
- **방법:**
  - 같은 G2 절차(autodE, XTB2 ALPB(water), 코드 그대로)를 반응마다 **2번 더** 독립 실행한다. autodE는 형태를 만들 때마다 새 난수를 쓴다(`conf_gen.py`의 `np.random.RandomState()`).
  - 기존 1번과 새 2번을 합친 3개 결과 중 **TS 에너지가 가장 낮은 것**을 고른다.
  - 고를 때 라벨을 보지 않으므로 배포에서도 쓸 수 있는 규칙이다.
- **이번 범위:** 계산, 선택, 비교 보고까지만 한다. G2c로 GEO1을 다시 학습할지는 그 결과를 보고 사용자가 정한다.

### 0-4. epoch와 LR을 바꾸지 않는 이유
- GEO1 R-curves 점검(DECISIONS "R-curves") 결과:
  - 검증 반응 오차는 약 110–130 epoch부터 평평했다.
  - 학습 손실은 300 epoch까지 절반으로 줄었지만, 그 감소가 검증으로 이어지지 않았다.
- 형태 불일치는 "같은 입력에 정답이 여러 개"인 문제다. 학습을 더 오래 하거나 LR을 바꾸면 학습 데이터의 형태 선택을 외울 뿐, 시험 반응의 형태를 맞히지 못한다.
- 따라서 이번에는 학습 데이터만 바꿔서 그 효과를 깨끗하게 본다.

### 0-5. 이 명령서의 코드가 어디까지 확인되었나 (작성자, 2026-10-30, `origin/geo1` 00eea4f 코드 기준)

| 코드 | 확인 |
|---|---|
| `geo1b.patch` (`geo_common.py`, `train.py`, `tests.py`, `r_launch.py`, `jobs/*.slurm`) | GEO1 원본에 `git apply --check`가 통과했고, 적용 결과가 작성자가 실행한 파일과 바이트 단위로 같았다 |
| GEO1b `tests.py synth` | U-1–U-6, U-7s 모두 PASS (U-7s L1 0.0378 → 0.0066) |
| `train.py` tune · select · final, `evaluate.py`, `train_m1.py` plan · final, `rmsd_b.py` | 가짜 기록 192개로 끝까지 실행했다 (tune 5 fold × c0·c2, final 15, 평가, fold −1 모델, RMSD 보고). `train_items`는 G2가 있는 학습 기록 수와 같고 종류는 모두 `g2`였다. `train_m1.py final`은 `GEO_MAX_EP ≠ 300`이면 멈춘다 |
| `r_launch.py` | `tasks()` = 41개 (tune 20, select 5, final 15, eval 1). 이름은 모두 `gb_`로 시작한다 |
| `select_g2c.py` check · build · report | 가짜 폴더(D1 12, D0 5 × 3 run)로 실행했다. 가장 낮은 에너지 선택, 동률 처리, 원래 run이 실패한 반응 보충, 폴더 통째 복사, 재실행 시 건너뛰기, 보고를 확인했다 |
| `geo2b.patch` (`eval_geo2.py`, `geo2_common.py`, `build_geo.py`) | 작성자의 GEO2 코드에 적용한 뒤 **실제 LADDER 결과로** `eval_geo2.py selftest`를 통과했다 (E2 40행 일치, 차이 0) |
| `data.py`, `g2c_d1.py`, D0 재실행 | HPC 경로와 autodE가 필요해 실행하지 못했다. U-8b·U-9b·U-10 검사와 G2c 진행 확인이 그 검증 역할을 한다 |

---

## 1. 0단계: GEO2 중단, worktree, 폴더, 디스크

### 1-1. GEO2 중단
1. `squeue -u yeseo1ee -o "%i %j %T %M"`로 GEO2 작업으로 낸 job을 모두 찾는다(Y의 `train_m1`, launcher, Z·G·F·H 등). GEO2 `DECISIONS.md`에 적힌 job id도 대조한다.
2. 찾은 job을 모두 `scancel`한다. GEO1 job은 이미 끝났어야 한다.
3. 브랜치 `geo2`에 `analysis/geo2/results/STOPPED.md`를 쓴다. 내용:
   - 날짜
   - 사유: "사용자 결정 2026-10-30: G1 미사용. GEO1(G1 학습) 구조 기반이라 중단, GEO2b로 대체"
   - 취소한 job 목록
   - 그때까지 끝난 단계
4. commit하고 `git push origin geo2`. `geo2` 브랜치가 없으면(GEO2가 시작되지 않았으면) 이 항목은 건너뛰고 그 사실을 기록한다.
5. `$GEO2`와 `$GEO/preds/final_fm1_*`은 지우지 않는다. 이후 읽지도 않는다.

### 1-2. 브랜치와 worktree
```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
git fetch origin
git switch -c geo1b origin/geo1
git worktree add /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c -b g2c origin/geo1
```

### 1-3. 폴더 (login node에서는 sbatch만)
```bash
sbatch -p cpu1,cpu2 --time=48:00:00 --cpus-per-task=1 --mem=1G -J gb_init \
  --output=/gpfs/home1/yeseo1ee/gb_init_%j.out \
  --wrap "mkdir -p /gpfs/tmp_cpu2/yeseo1ee/geo1b/logs /gpfs/tmp_cpu2/yeseo1ee/geo2b/logs /gpfs/tmp_cpu2/yeseo1ee/g2c/logs /gpfs/tmp_cpu2/yeseo1ee/g2c/lists; \
          df -h /gpfs/tmp_cpu2; du -sh /gpfs/tmp_cpu2/yeseo1ee/d1_g2 /gpfs/tmp_cpu2/yeseo1ee/d0_g2 /gpfs/tmp_cpu2/yeseo1ee/geo /gpfs/tmp_cpu2/yeseo1ee/geo2; \
          sinfo -o '%P %c %m %D %a'"
```

### 1-4. 디스크 확인 (STOP 조건)
- **필요량 = 2 × (`d1_g2` 크기 + `d0_g2` 크기) + 40 GB.** 앞부분은 G2c 재실행 4개 root, 40 GB는 GEO1b와 GEO2b의 여유분이다.
- 사용자 할당량 여유가 필요량보다 작으면 **STOP**한다. 그때 사용자에게 다음 선택지를 정리해서 묻는다:
  - G1 폴더(`d1_pair/g1_final`, `espley_xtb_g1`) 삭제. 사용자가 G1을 쓰지 않기로 했지만, **자동으로 지우지 않는다.**
  - 기존 `$GEO2` scratch 삭제
  - G2c `jobs/` 폴더를 tar로 압축
- 1-3의 `sinfo` 출력은 G2c job 크기를 정할 때 쓴다(5-3).

### 1-5. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | `geo1b` HEAD = `origin/geo1` 00eea4f(또는 그 뒤 commit). `analysis/geo/`의 GEO1 코드와 결과가 있다 |
| 0-2 | `$GEO/data/records.pkl`, `analysis/geo/results/D_dataset_report.json`이 있다. GEO1 D 보고와 개수를 비교하는 데만 쓴다(n 14,526; D1 G2 8,616; D0 G2 4,488) |
| 0-3 | `/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final`, `/gpfs/tmp_cpu2/yeseo1ee/d0_g2/g2_final`, `rmsd.py`(`RMSD_DIR`)에 `mapped_heavy_rmsd(a, b, cap=5000)`가 있다. 이 함수는 무거운 원자에 대해 그래프 동형(같은 원소끼리 대응)과 거울상을 모두 따져 가장 작은 RMSD를 준다 |
| 0-4 | 디스크 조건(1-4) |


## 2. GEO1b 코드 (`analysis/geo1b/`, 브랜치 `geo1b`)

### 2-1. GEO1 코드 복사 → 정리 → patch
```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
cp -r analysis/geo analysis/geo1b
rm -rf analysis/geo1b/results analysis/geo1b/__pycache__
mkdir analysis/geo1b/results
rm analysis/geo1b/inventory.py analysis/geo1b/d0g2.py analysis/geo1b/diag_u7.py analysis/geo1b/diag_interim.py
# 아래 2-2 블록을 docs/specs/geo1b.patch 로 저장한 뒤:
git apply --check docs/specs/geo1b.patch && git apply docs/specs/geo1b.patch
# 2-3 ~ 2-7 파일을 그대로 쓴다 (data.py, metrics.py, evaluate.py, rmsd_b.py, train_m1.py)
grep -n 'G1_FINAL\|G1_ROOT\|"same_ts_g1"\|, "g1")' analysis/geo1b/*.py      # 출력 0줄이어야 한다
```
- 남는 파일: `geo_common.py`, `model.py`(GEO1과 같음), `train.py`, `metrics.py`, `evaluate.py`, `data.py`, `tests.py`, `r_launch.py`, `val_baseline.py`, `diag_curves.py`, `rmsd_b.py`, `train_m1.py`, `jobs/gpu.slurm`, `jobs/cpu.slurm`
- `git apply --check`가 실패하면(origin/geo1 뒤에 GEO1 코드가 바뀐 경우) patch의 각 변경을 손으로 옮기고 `DECISIONS.md`에 적는다. 바꾸는 내용은 다음과 같다:
  - **경로:** 기본 scratch를 `geo1b`로 바꾼다.
  - **G1 경로 상수 삭제:** 남은 코드가 실수로 쓰면 바로 오류가 나도록 한다.
  - **`d1_core_table`:** `same_ts_g1`을 읽지 않는다.
  - **G2 그대로 기준값:** 전체 para 1,667짝, 비율 0.1338, 중앙값 0.0218 Å.
  - **`train_items`:** G2 전부를 쓰고 G1은 쓰지 않는다.
  - **`train.py`:** ablation 명령을 없앤다.
  - **`tests.py`:** U-7s 합성 예제를 g2 칸으로 옮기고, `data()`를 U-8b·U-9b·U-10으로 바꾼다.
  - **`r_launch.py`:** 기본 경로를 바꾸고, ablation을 없애고, job 이름을 `gb_`로 시작하게 한다.
  - **slurm 파일:** `GEO_SCRATCH`, 로그 경로, job 이름을 바꾼다.

### 2-2. `docs/specs/geo1b.patch` (그대로 저장)
```diff
--- a/analysis/geo1b/geo_common.py
+++ b/analysis/geo1b/geo_common.py
@@ -1,4 +1,4 @@
-"""GEO-1 common paths and helpers (docs/specs/GEO1.md). Python 3.9 compatible (D1 run env) and the GPU env (env_d1fu2)."""
+"""GEO1b common paths and helpers (docs/specs/GEO1B.md; copied from analysis/geo/geo_common.py, G1 removed). Python 3.9 compatible (D1 run env) and the GPU env (env_d1fu2)."""
 from __future__ import annotations
 
 import hashlib
@@ -15,7 +15,7 @@
 HERE = Path(__file__).resolve().parent
 REPO = HERE.parents[1]
 RES = HERE / "results"
-GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))
+GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))
 D0G2 = Path(os.environ.get("D0G2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2"))
 
 # ---- D1 (read only)
@@ -25,7 +25,6 @@
 D1_META_G2 = D1_G2_ROOT / "G" / "input_meta_d1_g2.csv"     # rxn_id, A_idx (frag1 = dipole, 0-based)
 D1_PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
 D1_JOBS = D1_PAIR / "d1_jobs"
-D1_G1_FINAL = D1_PAIR / "g1_final"
 D1_MANIFEST = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv")
 C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
 S_FOLDS = REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
@@ -36,12 +35,11 @@
 D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
 D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
 D0_META = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv")
-D0_G1_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
 D0_EXCLUDE = {3090, 3766, 4252, 3400, 5783}                # labels_all non-ok (= g1_geom.EXCLUDE)
 
 RMSD_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis")
 D1_FOLD_COUNTS = {0: 1874, 1: 1879, 2: 1503, 3: 1435, 4: 2195, -1: 380}   # C_compare rows per T1 fold (checked in I)
-G2RAW_PRIMARY = dict(n=840, share=0.1595, median=0.0155)                  # C_compare d_g2 vs d_dft, label-length rule
+G2RAW_PRIMARY = dict(n=1667, share=0.1338, median=0.0218)   # GEO1b: ALL para P_main pairs, test folds 0-4, G2 raw
 ELEM = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]   # type index = position
 TYPE = {e: i for i, e in enumerate(ELEM)}
 
@@ -185,13 +183,9 @@
 
 
 def d1_core_table():
-    """C_compare (all 9,266 D1 rows): rxn_id -> (core 'dipole/skeleton', T1 fold, same_ts_g1 as True/False/None)."""
-    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton", "same_ts_g1"])
+    """C_compare (all 9,266 D1 rows): rxn_id -> (core 'dipole/skeleton', T1 fold, None). GEO1b reads no G1-derived
+    column (same_ts_g1 is never read); the third slot stays None so the record layout is unchanged."""
+    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton"])
     fm = core_fold_map()
-    out = {}
-    for r in C.itertuples():
-        core = f"{r.dipole}/{r.skeleton}"
-        s = r.same_ts_g1
-        s = True if s in (True, "True") else (False if s in (False, "False") else None)
-        out[int(r.rxn_id)] = (core, int(fm.get(core, -9)), s)
-    return out
+    return {int(r.rxn_id): (f"{r.dipole}/{r.skeleton}", int(fm.get(f"{r.dipole}/{r.skeleton}", -9)), None)
+            for r in C.itertuples()}
--- a/analysis/geo1b/train.py
+++ b/analysis/geo1b/train.py
@@ -1,6 +1,6 @@
 #!/usr/bin/env python3
-"""Phase R: GEO-1 training.
-  python train.py smoke <k> | tune <k> <cfg> | select <k> | final <k> <seed> | ablation <k> <no_d0|q1_only|r0>"""
+"""Phase R: GEO1b training (docs/specs/GEO1B.md 3; = analysis/geo/train.py except train_items: all G2, no G1, no ablation).
+  python train.py smoke <k> | tune <k> <cfg> | select <k> | final <k> <seed>"""
 from __future__ import annotations
 
 import json
@@ -64,10 +64,8 @@
     for r in recs:
         if r["src"] == "D0" and not use_d0:
             continue
-        if r["xyz_g2"] is not None and r["same"] is True:
+        if r["xyz_g2"] is not None:                       # GEO1b: every G2 input, no form filter, no G1
             S.append((r, "g2"))
-        if r["xyz_g1"] is not None:
-            S.append((r, "g1"))
     return S
 
 
@@ -259,14 +257,6 @@
         P = predict(ema.model(arch), eval_items(te_r), n_pass)
         dump_atomic(P, GC.GEO / "preds" / f"final_f{k}_s{seed}.pkl")
         return 0
-    if cmd == "ablation":
-        name = a[2]
-        kw = dict(no_d0=dict(use_d0=False), q1_only=dict(q_mode="q1"), r0=dict(n_pass_override=1))[name]
-        items = train_items(full, use_d0=kw.pop("use_d0", True))
-        ema, arch, n_pass, _ = fit(items, None, cfg, 0, fixed_epochs=ep, tag=f"abl_{name}_f{k}", **kw)
-        P = predict(ema.model(arch), eval_items(te_r), n_pass)
-        dump_atomic(P, GC.GEO / "preds" / f"abl_{name}_f{k}.pkl")
-        return 0
     raise SystemExit(__doc__)
 
 
--- a/analysis/geo1b/tests.py
+++ b/analysis/geo1b/tests.py
@@ -1,6 +1,6 @@
 #!/usr/bin/env python3
-"""Phase U: GEO-1 unit tests (GPU env).  python tests.py synth   (U-1..U-6, U-7s; no records needed)
-                                          python tests.py data    (U-0, U-7, U-8, U-9; needs records.pkl)
+"""Phase U: GEO1b unit tests (GPU env).  python tests.py synth   (U-1..U-6, U-7s; no records needed)
+                                          python tests.py data    (U-0, U-7, U-8b, U-9b, U-10; needs records.pkl)
 Writes results/U_tests_<mode>.json; exit 3 if any test fails."""
 from __future__ import annotations
 
@@ -34,8 +34,8 @@
     return dict(uid=uid, src="D1", src_id=0, rxn_id=rid, core="fake/A", fold=0, n=n,
                 types=np.array([GC.TYPE[s] for s in syms]), lab=x + rng.normal(0, 0.05, x.shape),
                 frag=np.array([1] * (n // 2) + [2] * (n - n // 2)), form=np.array([1, 1, 1, 1] + [0] * (n - 4)),
-                pairs=pairs, charge=0, same=True, overlap=False, xyz_g2=x, lab_on_g2=x + 0.03, bond_g2=B,
-                spd_g2=GC.spd_matrix(B), xyz_g1=None)
+                pairs=pairs, charge=0, same=None, overlap=False, xyz_g2=x, lab_on_g2=x + 0.03, bond_g2=B,
+                spd_g2=GC.spd_matrix(B), xyz_g1=None, map_rmsd_g2=0.1)
 
 
 def batch_of(exs, double=True):
@@ -107,8 +107,8 @@
     for i in range(32):
         r = fake_rec(int(rng.integers(12, 30)), rng, f"D1_{100 + i}", 100 + i)
         x = r["xyz_g2"]
-        r.update(xyz_g1=x, lab_on_g1=x.mean(0) + 1.03 * (x - x.mean(0)), bond_g1=r["bond_g2"], spd_g1=r["spd_g2"])
-        items.append((r, "g1"))
+        r.update(lab_on_g2=x.mean(0) + 1.03 * (x - x.mean(0)))
+        items.append((r, "g2"))
     start, end = overfit(items, 600)
     check("U-7s", end < 0.3 * start, start_L1=start, end_L1=end)
 
@@ -138,13 +138,14 @@
 def data():
     recs = T.load_records()
     rep = json.loads((GC.RES / "D_dataset_report.json").read_text())
-    check("U-0", len(recs) == rep["n"], n=len(recs))
-    # U-7 overfit
+    check("U-0", len(recs) == rep["n"] and rep.get("g1_used") is False, n=len(recs))
+    # U-7 overfit (report only, as GEO1 decision U-7): 32 D1 training G2 samples
     tr, _, _ = T.split(recs, 0)
-    items = [(r, "g1") for r in tr if r["src"] == "D1" and r["xyz_g1"] is not None][:32]
+    items = [(r, "g2") for r in tr if r["src"] == "D1" and r["xyz_g2"] is not None][:32]
     start, end = overfit(items, 2000)
-    check("U-7", end < 0.5 * start, start_L1=start, end_L1=end)
-    # U-8 metric regression on the G2-raw baseline
+    RES["U-7"] = dict(ok=True, report_only=True, start_L1=start, end_L1=end, ratio=end / start)
+    print(f"[REPORT] U-7 start {start:.4f} end {end:.4f} ratio {end / start:.3f}", flush=True)
+    # U-8b metric regression on the G2-raw baseline, ALL para P_main pairs
     test = [r for r in recs if r["src"] == "D1" and r["fold"] in (0, 1, 2, 3, 4) and r["xyz_g2"] is not None]
     mt = evaluate_preds({r["uid"]: r["xyz_g2"] for r in test}, test, with_rmsd=False)
     ex = GC.G2RAW_PRIMARY
@@ -156,11 +157,14 @@
         c = C.loc[r["rxn_id"]]
         dev = max(dev, float(np.abs(np.array(dl) - np.sort([c.d_dft_1, c.d_dft_2])).max()),
                   float(np.abs(np.array(dg) - np.sort([c.d_g2_1, c.d_g2_2])).max()))
-    check("U-8", mt["n_primary"] == ex["n"] and abs(mt["primary_share"] - ex["share"]) <= 0.005
-          and abs(mt["pair_pe_median_same_para"] - ex["median"]) <= 0.001 and dev <= 1e-4,
-          n=mt["n_primary"], share=mt["primary_share"], median=mt["pair_pe_median_same_para"], max_dist_dev=dev)
-    # U-9 leakage
+    check("U-8b", mt["n_primary"] == ex["n"] and abs(mt["primary_share"] - ex["share"]) <= 0.005
+          and abs(mt["pair_pe_median_para"] - ex["median"]) <= 0.001 and dev <= 1e-4 and len(test) == 8425,
+          n=mt["n_primary"], share=mt["primary_share"], median=mt["pair_pe_median_para"], max_dist_dev=dev,
+          n_test=len(test))
+    # U-9b leakage + G1-free + every G2 input used
     bad = []
+    if any(r.get("xyz_g1") is not None or r.get("same") is not None for r in recs):
+        bad.append("a record carries G1 data (xyz_g1 / same)")
     for k in range(5):
         tr, va, te = T.split(recs, k)
         tr_ids = {r["uid"] for r in tr}
@@ -168,8 +172,11 @@
             bad.append(f"fold {k}: test/inner-val reaction in train")
         if any(r["src"] == "D0" and r["overlap"] for r in tr):
             bad.append(f"fold {k}: overlapping D0 in train")
-        if any(kind == "g2" and r["same"] is not True for r, kind in T.train_items(tr)):
-            bad.append(f"fold {k}: non-same G2 sample in train")
+        it = T.train_items(tr)
+        if any(kind != "g2" for _, kind in it):
+            bad.append(f"fold {k}: a non-G2 training sample")
+        if len(it) != sum(r["xyz_g2"] is not None for r in tr):
+            bad.append(f"fold {k}: train items {len(it)} != train records with G2")
         for r, kind in T.eval_items(te)[:200]:
             e = T.make_example((r, kind), False, 0.0, "mix", np.random.default_rng(0))
             if not np.allclose(e["x0"] + e["center"], r["xyz_g2"], atol=1e-10, rtol=0):
@@ -177,7 +184,11 @@
             syms = [GC.ELEM[t] for t in r["types"]]
             if not np.array_equal(GC.bond_types(syms, r["xyz_g2"], r["pairs"]), r["bond_g2"]):
                 bad.append(f"fold {k}: bond graph not from the input for {r['uid']}")
-    check("U-9", not bad, problems=bad[:20])
+    check("U-9b", not bad, problems=bad[:20])
+    # U-10 mapped RMSD defined for >= 95 % of D1 G2 inputs (descriptive bins only)
+    v = np.array([r["map_rmsd_g2"] for r in recs if r["src"] == "D1" and r["xyz_g2"] is not None], float)
+    check("U-10", np.isfinite(v).mean() >= 0.95, finite=float(np.isfinite(v).mean()),
+          median=float(np.nanmedian(v)), n=int(len(v)))
 
 
 if __name__ == "__main__":
--- a/analysis/geo1b/r_launch.py
+++ b/analysis/geo1b/r_launch.py
@@ -6,7 +6,7 @@
 file exists or whose job name is queued is never submitted again; a job that ended without its output is resubmitted
 once (infra), a second loss blocks its dependants and writes results/STOP_R.md.
   sbatch -p cpu1,cpu2 --time=48:00:00 --cpus-per-task=1 --mem=2G --dependency=afterok:<smoke> \
-         --wrap "source .../conda.sh; conda activate reactot; cd analysis/geo; python r_launch.py"
+         --wrap "source .../conda.sh; conda activate reactot; cd analysis/geo1b; python r_launch.py"
 Writes results/R_budget.json, results/HPC_efficiency.csv, $GEO/run_r/state.json."""
 from __future__ import annotations
 
@@ -19,11 +19,11 @@
 
 HERE = Path(__file__).resolve().parent
 RES = HERE / "results"
-GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))
+GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))
 RUNS, PREDS, LOGS, STATE = GEO / "runs", GEO / "preds", GEO / "logs", GEO / "run_r" / "state.json"
 GPU_P, CPU_P = "gpu1,gpu2,gpu3,gpu5,gpu6", "cpu1,cpu2"
 MAX_SUBMIT, GPU_H_MAX, POLL_S, STOP_AFTER_H = 20, 200.0, 120, 46.0
-FOLDS, CFGS, SEEDS, ABL = (0, 1, 2, 3, 4), ("c0", "c1", "c2", "c3"), (0, 1, 2), ("no_d0", "q1_only", "r0")
+FOLDS, CFGS, SEEDS, ABL = (0, 1, 2, 3, 4), ("c0", "c1", "c2", "c3"), (0, 1, 2), ()   # GEO1b: no ablation
 USER = os.environ.get("USER", "yeseo1ee")
 
 
@@ -87,21 +87,21 @@
     T = []
     for k in FOLDS:
         for cfg in CFGS:
-            T.append(dict(name=f"geo_tune_f{k}_{cfg}", gpu=True, args=["train.py", "tune", str(k), cfg], env=env,
+            T.append(dict(name=f"gb_tune_f{k}_{cfg}", gpu=True, args=["train.py", "tune", str(k), cfg], env=env,
                           out=RUNS / f"tune_f{k}_{cfg}.history.json", deps=[]))
     for k in FOLDS:
-        T.append(dict(name=f"geo_sel_f{k}", gpu=False, args=["train.py", "select", str(k)], env=env,
-                      out=RES / f"R_select_f{k}.json", deps=[f"geo_tune_f{k}_{cfg}" for cfg in CFGS]))
+        T.append(dict(name=f"gb_sel_f{k}", gpu=False, args=["train.py", "select", str(k)], env=env,
+                      out=RES / f"R_select_f{k}.json", deps=[f"gb_tune_f{k}_{cfg}" for cfg in CFGS]))
     for k in FOLDS:
         for s in SEEDS:
-            T.append(dict(name=f"geo_final_f{k}_s{s}", gpu=True, args=["train.py", "final", str(k), str(s)], env=env,
-                          out=PREDS / f"final_f{k}_s{s}.pkl", deps=[f"geo_sel_f{k}"]))
+            T.append(dict(name=f"gb_final_f{k}_s{s}", gpu=True, args=["train.py", "final", str(k), str(s)], env=env,
+                          out=PREDS / f"final_f{k}_s{s}.pkl", deps=[f"gb_sel_f{k}"]))
         if k in c["abl_folds"]:
             for a in ABL:
-                T.append(dict(name=f"geo_abl_{a}_f{k}", gpu=True, args=["train.py", "ablation", str(k), a], env=env,
-                              out=PREDS / f"abl_{a}_f{k}.pkl", deps=[f"geo_sel_f{k}"]))
-    T.append(dict(name="geo_eval", gpu=False, args=["evaluate.py"], env=env, out=RES / "E_report.json",
-                  deps=[t["name"] for t in T if t["name"].startswith(("geo_final", "geo_abl"))]))
+                T.append(dict(name=f"gb_abl_{a}_f{k}", gpu=True, args=["train.py", "ablation", str(k), a], env=env,
+                              out=PREDS / f"abl_{a}_f{k}.pkl", deps=[f"gb_sel_f{k}"]))
+    T.append(dict(name="gb_eval", gpu=False, args=["evaluate.py"], env=env, out=RES / "E_report.json",
+                  deps=[t["name"] for t in T if t["name"].startswith(("gb_final", "gb_abl"))]))
     return T
 
 
@@ -153,7 +153,7 @@
     dep = [f"--begin=now+{ONCE_MIN}minutes"] if begin else ([f"--dependency=afterany:{me}"] if me else [])
     env = "--export=ALL,GEO_R_ONCE=1" if ONCE else "--export=ALL"
     try:
-        out = sh(["sbatch", "--parsable", "-p", CPU_P, "-J", "geo_rlaunch", "--time=48:00:00", "--cpus-per-task=1",
+        out = sh(["sbatch", "--parsable", "-p", CPU_P, "-J", "gb_rlaunch", "--time=48:00:00", "--cpus-per-task=1",
                   "--mem=2G", f"--output={LOGS}/%x_%j.out", env, *dep, "--wrap", RELAUNCH])
         print(f"[r_launch] relaunched as {out.strip()}", flush=True)
         return True
--- a/analysis/geo1b/jobs/gpu.slurm
+++ b/analysis/geo1b/jobs/gpu.slurm
@@ -1,10 +1,10 @@
 #!/bin/bash
-#SBATCH --job-name=geo_gpu
+#SBATCH --job-name=gb_gpu
 #SBATCH --time=48:00:00
 #SBATCH --cpus-per-task=4
 #SBATCH --mem=32G
-#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo/logs/%x_%j.out
-# submit from analysis/geo/:  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 jobs/gpu.slurm <script.py> <args...>
+#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out
+# submit from analysis/geo1b/:  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 jobs/gpu.slurm <script.py> <args...>
 # CPU-only use (select / evaluate):  sbatch -p <cpu partition> jobs/gpu.slurm <script.py> <args...>
 set -uo pipefail
 cd "${SLURM_SUBMIT_DIR:?}" || exit 2
@@ -14,9 +14,9 @@
 [ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
 export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
 export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
-export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
+export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo1b D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
 gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
-echo "[geo_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD 2>/dev/null)" \
+echo "[gb_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD 2>/dev/null)" \
      "$($PY --version 2>&1) gpu=${gpu:-none} $(date '+%F %T')"
 "$PY" "$@"; rc=$?
 echo "=== exit $rc $(date '+%F %T')"
--- a/analysis/geo1b/jobs/cpu.slurm
+++ b/analysis/geo1b/jobs/cpu.slurm
@@ -1,18 +1,18 @@
 #!/bin/bash
-#SBATCH --job-name=geo_cpu
+#SBATCH --job-name=gb_cpu
 #SBATCH --time=48:00:00
 #SBATCH --cpus-per-task=4
 #SBATCH --mem=16G
-#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo/logs/%x_%j.out
-# submit from analysis/geo/:  sbatch [-p ... --cpus-per-task=N --mem=M -J name --array=...] jobs/cpu.slurm <script.py> <args...>
+#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out
+# submit from analysis/geo1b/:  sbatch [-p ... --cpus-per-task=N --mem=M -J name --array=...] jobs/cpu.slurm <script.py> <args...>
 # "@I" -> $SLURM_ARRAY_TASK_ID.  D1G2_SCRATCH: $D0G2 unless GEO_D1G2_SCRATCH is exported (only for check Z-1).
 set -uo pipefail
 cd "${SLURM_SUBMIT_DIR:?}" || exit 2
 source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
 PY="${D1_PY:?}"
-export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
+export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo1b D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
 export D1G2_SCRATCH="${GEO_D1G2_SCRATCH:-$D0G2_SCRATCH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 D1G2_JOB_T0=$(date +%s)
-echo "[geo_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
+echo "[gb_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
      "git=$(git rev-parse --short HEAD 2>/dev/null) D1G2_SCRATCH=$D1G2_SCRATCH $($PY -V 2>&1) $(date '+%F %T')"
 args=(); for a in "$@"; do args+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); done
 "$PY" "${args[@]}"; rc=$?
```

### 2-3. `analysis/geo1b/data.py` (그대로 작성)
```python
#!/usr/bin/env python3
"""GEO1b Phase D: records without G1 -> $GEO1B/data/records.pkl + results/D_dataset_report.json (D1 run env).

Same as analysis/geo/data.py except (docs/specs/GEO1B.md 3-2):
  * no G1 input is read anywhere: every record has xyz_g1 = None and same = None (layout unchanged for train/model code);
  * each record with a G2 input gets map_rmsd_g2 = rmsd.mapped_heavy_rmsd(G2 TS, label TS) (heavy atoms, graph
    isomorphisms and mirror images; inf -> nan). It is used only for descriptive bins in the evaluation, never in training.
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

bond_types, spd_matrix = GC.bond_types, GC.spd_matrix


def canon_rxn(smi, stereo=False):
    """unchanged from analysis/geo/data.py (decision D-1)."""
    from rdkit import Chem
    out = []
    for side in smi.split(">>"):
        ms = []
        for s in side.split("."):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            m = Chem.RemoveHs(m)
            if not stereo:
                Chem.RemoveStereochemistry(m)
            ms.append(Chem.MolToSmiles(m))
        out.append(".".join(sorted(ms)))
    return ">>".join(out)


def d0_overlap_ids():
    """unchanged from analysis/geo/data.py."""
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False)
    s1 = man[man.kind == "D1"].rxn_smiles
    c1 = [canon_rxn(s) for s in s1]
    d1 = {c for c in c1 if c is not None}
    d1s = {c for c in (canon_rxn(s, stereo=True) for s in s1) if c is not None}
    smi = GC.load_d0_smiles()
    c0 = {rid: canon_rxn(s) for rid, s in smi.items()}
    c0s = {rid: canon_rxn(s, stereo=True) for rid, s in smi.items()}
    stats = dict(n_d1_smiles=int(len(s1)), n_d1_parsed=sum(c is not None for c in c1), n_d1_canon=len(d1),
                 n_d1_canon_with_stereo=len(d1s), n_d0_canon_none=sum(v is None for v in c0.values()),
                 n_overlap_with_stereo=sum(c is not None and c in d1s for c in c0s.values()))
    return {rid for rid, c in c0.items() if c is not None and c in d1}, stats


def map_rmsd(g2, lab):
    r = float(rm.mapped_heavy_rmsd(g2, lab))
    return r if math.isfinite(r) else float("nan")


def make_rec(src, rid, core, fold, lab, g2, A_idx, pairs, charge, overlap, log):
    syms, lx = lab
    n = len(syms)
    if any(s not in GC.TYPE for s in syms):
        log.append(f"{src} {rid}: element outside ELEM -> dropped")
        return None
    rec = dict(uid=f"{src}_{rid}", src=src, src_id=0 if src == "D1" else 1, rxn_id=int(rid), core=core, fold=int(fold),
               n=n, types=np.array([GC.TYPE[s] for s in syms], np.int64), lab=lx.astype(np.float64),
               frag=np.array([1 if i in set(A_idx) else 2 for i in range(n)], np.int64),
               form=np.array([1 if any(i in p for p in pairs) else 0 for i in range(n)], np.int64),
               pairs=np.array(pairs, np.int64), charge=int(charge), same=None, overlap=bool(overlap),
               xyz_g2=None, xyz_g1=None, map_rmsd_g2=float("nan"))
    if g2 is None:
        return rec
    if list(g2[0]) != list(syms):
        log.append(f"{src} {rid}: g2 element order differs -> input dropped")
        return rec
    x = g2[1].astype(np.float64)
    al, r, refl = GC.kabsch_align(lx, x, reflect=True)
    B = bond_types(syms, x, pairs)
    rec.update(xyz_g2=x, lab_on_g2=al, rmsd_g2=r, refl_g2=refl, bond_g2=B, spd_g2=spd_matrix(B),
               map_rmsd_g2=map_rmsd(g2, (syms, lx)))
    return rec


def main():
    log, recs = [], []
    T1 = GC.d1_core_table()                          # rxn_id -> (core, fold, None)
    meta1 = GC.load_meta(GC.D1_META_G2)
    for d in GC.load_d1_labels():
        rid = int(d["rxn_id"])
        core, fold, _ = T1[rid]
        if rid not in meta1:
            log.append(f"D1 {rid}: no A_idx in input_meta_d1_g2.csv -> dropped")
            continue
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        r = make_rec("D1", rid, core, fold, lab, GC.read_if_done(GC.D1_G2_FINAL / str(rid)), meta1[rid],
                     GC.parse_pairs(d["formed_pairs_ts"]), int(d["charge1"]) + int(d["charge2"]), False, log)
        if r is not None:
            recs.append(r)
    meta0 = GC.load_meta(GC.D0_META)
    ov, ov_stats = d0_overlap_ids()
    if ov_stats["n_d1_parsed"] < 9000:
        print(f"STOP: only {ov_stats['n_d1_parsed']} D1 reaction SMILES parsed for the overlap check", file=sys.stderr)
        return 3
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
        r = make_rec("D0", rid, "D0", -1, lab, GC.read_if_done(GC.D0G2 / "g2_final" / str(rid)), meta0[rid],
                     GC.parse_pairs(L["formed_pairs_ts"]), int(L["charge1"]) + int(L["charge2"]), rid in ov, log)
        if r is not None:
            recs.append(r)
    out = GC.GEO / "data" / "records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    tmp.replace(out)
    df = pd.DataFrame([dict(src=r["src"], fold=r["fold"], g2=r["xyz_g2"] is not None, overlap=r["overlap"],
                            refl_g2=bool(r.get("refl_g2")), map_rmsd=r["map_rmsd_g2"]) for r in recs])
    cnt = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}   # noqa: E731
    n_bad_order = sum("element order differs" in m for m in log)

    def mq(g):
        v = g.map_rmsd[g.g2].to_numpy(float)
        f = v[np.isfinite(v)]
        return dict(n_g2=int(g.g2.sum()), n_nan=int(np.isnan(v).sum()),
                    q10_50_90=[float(x) for x in np.quantile(f, [0.1, 0.5, 0.9])] if len(f) else None,
                    share_le=dict((str(t), float((f <= t).mean())) for t in (0.25, 0.5, 0.75, 1.0)) if len(f) else None)
    GC.write_json(GC.RES / "D_dataset_report.json", dict(
        n=len(recs), g1_used=False, overlap_check=ov_stats, log=log[:200], n_log=len(log),
        n_element_order_differs=n_bad_order,
        by_src={s: dict(n=int(len(g)), g2=int(g.g2.sum()), map_rmsd_g2=mq(g)) for s, g in df.groupby("src")},
        d0_overlap=int(df.overlap.sum()), d1_fold_counts=cnt(df[df.src == "D1"].fold),
        reflected=dict(g2=int(df.refl_g2.sum())), records=str(out), sha256=GC.sha256(out)))
    if n_bad_order > 0.01 * len(recs):
        print(f"STOP: element order differs for {n_bad_order} inputs (> 1 %)", file=sys.stderr)
        return 3
    d1 = df[(df.src == "D1") & df.g2]
    if d1.map_rmsd.isna().mean() > 0.05:
        print(f"STOP: mapped RMSD undefined (graph mismatch) for {d1.map_rmsd.isna().mean():.3f} of D1 G2 inputs",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 2-4. `analysis/geo1b/metrics.py` (그대로 작성)
```python
"""GEO1b metrics (docs/specs/GEO1B.md 4-1). Bond identity across a pair = rank by LABEL length (short, long), as GEO1.
Primary (inner validation and test) = share of ALL para P_main reference pairs with pair error <= 0.005 A. No pair
class, no G1-derived flag. map_rmsd_g2 (records) only labels descriptive bins."""
from __future__ import annotations

import numpy as np
import pandas as pd

import geo_common as GC

_FP = None
BINS = (0.25, 0.75)                                       # mapped RMSD (G2 raw vs label, A): low <= 0.25 < mid <= 0.75 < high


def f_pairs():
    global _FP
    if _FP is None:
        F = pd.read_csv(GC.F_PAIRS, usecols=["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main"])
        _FP = F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].copy()
    return _FP


def rxn_errors(pred, lab, pairs):
    dl = np.array([np.linalg.norm(lab[i] - lab[j]) for i, j in pairs])
    dp = np.array([np.linalg.norm(pred[i] - pred[j]) for i, j in pairs])
    o = np.argsort(dl)
    return dp[o] - dl[o]                                  # (e_short, e_long)


def rmsd_bin(v):
    if v is None or not np.isfinite(v):
        return "na"
    return "low" if v <= BINS[0] else ("mid" if v <= BINS[1] else "high")


def pair_bin(a, b):
    """the worse of the two reactions' bins (na if either is na)."""
    order = dict(low=0, mid=1, high=2)
    if "na" in (a, b):
        return "na"
    return max((a, b), key=lambda s: order[s])


def evaluate_preds(P, recs, with_rmsd=True):
    """P: uid -> predicted xyz (any frame); recs: records of the evaluated reactions (D1 only)."""
    R = {r["rxn_id"]: r for r in recs if r["src"] == "D1" and r["uid"] in P}
    E, rows = {}, []
    for rid, r in R.items():
        e = rxn_errors(P[r["uid"]], r["lab"], r["pairs"])
        E[rid] = e
        row = dict(rxn_id=rid, core=r["core"], e_short=float(e[0]), e_long=float(e[1]), abs_max=float(np.abs(e).max()),
                   map_rmsd_g2=float(r.get("map_rmsd_g2", np.nan)), bin=rmsd_bin(r.get("map_rmsd_g2")))
        if with_rmsd:
            heavy = r["types"] != 0
            row["rmsd_heavy"] = GC.kabsch_align(P[r["uid"]][heavy], r["lab"][heavy], reflect=True)[1]
        rows.append(row)
    F = f_pairs()
    F = F[F.a_rxn.isin(E) & F.b_rxn.isin(E)].copy()
    F["pe"] = [float(np.max(np.abs(E[a] - E[b]))) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["core"] = [R[a]["core"] for a in F.a_rxn]
    F["dipole"] = [R[a]["core"].split("/")[0] for a in F.a_rxn]
    F["bin"] = [pair_bin(rmsd_bin(R[a].get("map_rmsd_g2")), rmsd_bin(R[b].get("map_rmsd_g2")))
                for a, b in zip(F.a_rxn, F.b_rxn)]
    prim = F[F.pos == "p"]
    D = pd.DataFrame(rows)
    if not with_rmsd:
        D["rmsd_heavy"] = np.nan
    nan = float("nan")
    return dict(primary_share=float((prim.pe <= 0.005).mean()) if len(prim) else nan, n_primary=int(len(prim)),
                all_pairs_share=float((F.pe <= 0.005).mean()) if len(F) else nan, n_pairs=int(len(F)),
                pair_pe_median_para=float(prim.pe.median()) if len(prim) else nan,
                rxn_form_abs_median=float(np.median(np.abs(np.concatenate([D.e_short, D.e_long])))) if len(D) else nan,
                rmsd_heavy_median=float(D.rmsd_heavy.median()) if len(D) else nan, table_pairs=F, table_rxn=D)
```

### 2-5. `analysis/geo1b/evaluate.py` (그대로 작성)
```python
#!/usr/bin/env python3
"""GEO1b Phase E1: structure evaluation on the D1 test folds (G2 input only), descriptive (docs/specs/GEO1B.md 4-1).
The decision is made on energy in GEO2b; here only numbers and CIs. GPU env on a CPU partition.  python evaluate.py
Writes results/E_main.csv, E_pairs.csv, E_rxn.csv, E_by_bin.csv, E_by_dipole.csv, E_report.json."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds  # noqa: E402

B_BOOT, SEED, THR = 2000, 20261026, 0.005
SEEDS = (0, 1, 2)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def preds_for(name, folds):
    """uid -> xyz for the given test folds; None if a fold file is missing. 'model' = 3-seed coordinate mean (GEO1)."""
    P = {}
    for k in folds:
        if name == "model":
            S = [GC.GEO / "preds" / f"final_f{k}_s{s}.pkl" for s in SEEDS]
            if not all(p.is_file() for p in S):
                return None
            parts = [load_pkl(p) for p in S]
            assert all(set(q) == set(parts[0]) for q in parts), f"seed prediction sets differ in fold {k}"
            P.update({u: np.mean([q[u] for q in parts], axis=0) for u in parts[0]})
        else:                                             # seed<s>
            p = GC.GEO / "preds" / f"final_f{k}_s{name[4:]}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
    return P


def core_counts(F):
    """per core over ALL para pairs: (#pairs with pe <= THR, #pairs)."""
    prim = F[F.pos == "p"]
    g = prim.groupby("core").pe
    return pd.DataFrame(dict(ok=g.apply(lambda s: int((s <= THR).sum())), n=g.size()))


def boot_diff(Fa, Fb, rng):
    """core bootstrap of share(a), share(b), share(a) - share(b) on the same para pairs."""
    A, B = core_counts(Fa), core_counts(Fb)
    assert A.index.equals(B.index) and (A.n == B.n).all(), "para pair sets differ between the compared targets"
    cores = A.index.to_numpy()
    out = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(cores), len(cores), replace=True)
        na = A.n.to_numpy()[pick].sum()
        sa, sb = A.ok.to_numpy()[pick].sum() / na, B.ok.to_numpy()[pick].sum() / na
        out.append((sa, sb, sa - sb))
    q = np.percentile(np.array(out), [2.5, 97.5], axis=0)
    return dict(ci_a=list(q[:, 0]), ci_b=list(q[:, 1]), ci_diff=list(q[:, 2]))


def main():
    recs = GC.load_records()
    if any(r.get("xyz_g1") is not None or r.get("same") is not None for r in recs):
        print("STOP: a record carries G1 data (xyz_g1 / same); GEO1b records must be G1-free", file=sys.stderr)
        return 3
    folds = [0, 1, 2, 3, 4]
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in folds and r["xyz_g2"] is not None]
    targets = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
    for name in ("model",) + tuple(f"seed{s}" for s in SEEDS):
        P = preds_for(name, folds)
        if P is not None:
            targets[name] = P
    if "model" not in targets or len(targets["model"]) != len(test):
        print("STOP: model predictions missing for some test reactions", file=sys.stderr)
        return 3
    res, tabs = {}, {}
    for name, P in targets.items():
        m = evaluate_preds(P, test)
        tabs[name] = m
        res[name] = {k: v for k, v in m.items() if not k.startswith("table_")}
    rng = np.random.default_rng(SEED)
    ci = {"model_vs_g2_raw": boot_diff(tabs["model"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)}
    for s in SEEDS:
        if f"seed{s}" in tabs:
            ci[f"seed{s}_vs_g2_raw"] = boot_diff(tabs[f"seed{s}"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)
    GC.RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).T.to_csv(GC.RES / "E_main.csv")
    tabs["model"]["table_pairs"].merge(tabs["g2_raw"]["table_pairs"][["pair_id", "pe"]], on="pair_id",
                                       suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_pairs.csv", index=False)
    tabs["model"]["table_rxn"].merge(tabs["g2_raw"]["table_rxn"][["rxn_id", "e_short", "e_long", "rmsd_heavy"]],
                                     on="rxn_id", suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_rxn.csv", index=False)
    rows_b, rows_d = [], []
    for name in ("model", "g2_raw"):
        F, D = tabs[name]["table_pairs"], tabs[name]["table_rxn"]
        for (pos, b), g in F.groupby(["pos", "bin"]):
            rows_b.append(dict(target=name, level="pair", pos=pos, bin=b, n=len(g),
                               share=float((g.pe <= THR).mean()), median=float(g.pe.median())))
        for b, g in D.groupby("bin"):
            rows_b.append(dict(target=name, level="rxn", pos="", bin=b, n=len(g), share=float("nan"),
                               median=float(np.median(np.abs(np.concatenate([g.e_short, g.e_long]))))))
        for dip, g in F[F.pos == "p"].groupby("dipole"):
            rows_d.append(dict(target=name, dipole=dip, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
    pd.DataFrame(rows_b).to_csv(GC.RES / "E_by_bin.csv", index=False)
    pd.DataFrame(rows_d).to_csv(GC.RES / "E_by_dipole.csv", index=False)
    ex = GC.G2RAW_PRIMARY
    g2ok = (res["g2_raw"]["n_primary"] == ex["n"] and abs(res["g2_raw"]["primary_share"] - ex["share"]) <= 0.005
            and abs(res["g2_raw"]["pair_pe_median_para"] - ex["median"]) <= 0.001)
    GC.write_json(GC.RES / "E_report.json", dict(
        decision="none here: structure is descriptive; GEO2b decides on energy", primary=res, ci=ci, n_test=len(test),
        g2_raw_expected=ex, g2_raw_matches_expected=bool(g2ok), thr_A=THR, n_boot=B_BOOT, g1_used=False))
    d = ci["model_vs_g2_raw"]["ci_diff"]
    print(f"model {res['model']['primary_share']:.3f} vs g2_raw {res['g2_raw']['primary_share']:.3f} "
          f"(all para pairs, n={res['model']['n_primary']}), diff CI [{d[0]:+.3f}, {d[1]:+.3f}]", flush=True)
    if not g2ok:
        print("STOP: the G2-raw baseline does not reproduce GC.G2RAW_PRIMARY (data or metric changed)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 2-6. `analysis/geo1b/rmsd_b.py` (그대로 작성)
```python
#!/usr/bin/env python3
"""GEO1b Phase E2 (D1 run env, CPU): symmetry-aware heavy-atom RMSD and sanity of the test predictions, report only
(docs/specs/GEO1B.md 4-2).   python rmsd_b.py <n_proc>
Reads $GEO1B/data/records.pkl and $GEO1B/preds/final_f{k}_s{0,1,2}.pkl (3-seed mean = the evaluated model).
Writes results/E_rmsd.csv (one row per test reaction) and results/E_rmsd.json (summary).
  map_g2    = rmsd.mapped_heavy_rmsd(G2 raw, label)      (= records map_rmsd_g2; recomputed as a check)
  map_model = rmsd.mapped_heavy_rmsd(model, label)
  map_move  = rmsd.mapped_heavy_rmsd(model, G2 raw)      (> 0.30 A = autodE's distinct-conformer threshold)
  sanity    = the GEO2 build_geo rules (close contact, bond graph change vs G2, forming distance 1.4-4.5 A)."""
from __future__ import annotations

import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import rmsd_bin  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

MIN_CONTACT, BOND_IN, BOND_OUT, FORM = 0.60, 1.10, 1.40, (1.4, 4.5)   # build_geo rules; 4.5 A = GEO1 decision I-1
MOVE_A = 0.30


def mapped(a, b):
    r = float(rm.mapped_heavy_rmsd(a, b))
    return r if math.isfinite(r) else float("nan")


def sanity(syms, X, Xg2, pairs):
    r = np.array([GC.COV[s] for s in syms])
    S = r[:, None] + r[None, :]
    D, Dg = GC.dmat(X), GC.dmat(Xg2)
    iu = np.triu_indices(len(syms), 1)
    if (D / S)[iu].min() < MIN_CONTACT:
        return "close_contact"
    fp = {tuple(sorted(p)) for p in pairs}
    mask = np.array([(i, j) not in fp for i, j in zip(*iu)])
    rg, rx = (Dg / S)[iu][mask], (D / S)[iu][mask]
    if ((rg < BOND_IN) & (rx > BOND_OUT)).any() or ((rg > BOND_OUT) & (rx < BOND_IN)).any():
        return "bond_graph_changed"
    d = GC.form_d(X, [tuple(p) for p in pairs])
    if min(d) < FORM[0] or max(d) > FORM[1]:
        return "forming_distance_range"
    return ""


def one(a):
    uid, rid, core, syms, lab, g2, mod, pairs, map_rec = a
    L, G, M = (syms, lab), (syms, g2), (syms, mod)
    return dict(uid=uid, rxn_id=rid, core=core, map_g2_rec=map_rec, map_g2=mapped(G, L), map_model=mapped(M, L),
                map_move=mapped(M, G), sanity=sanity(syms, mod, g2, pairs))


def main(n_proc):
    recs = GC.load_records()
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in range(5) and r["xyz_g2"] is not None]
    P = {}
    for k in range(5):
        parts = []
        for s in (0, 1, 2):
            with open(GC.GEO / "preds" / f"final_f{k}_s{s}.pkl", "rb") as f:
                parts.append(pickle.load(f))
        P.update({u: np.mean([np.asarray(q[u], float) for q in parts], axis=0) for u in parts[0]})
    jobs = [(r["uid"], r["rxn_id"], r["core"], [GC.ELEM[t] for t in r["types"]], r["lab"], r["xyz_g2"],
             P[r["uid"]], r["pairs"].tolist(), r["map_rmsd_g2"]) for r in test]
    with Pool(int(n_proc)) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    D = pd.DataFrame(rows)
    D["bin"] = [rmsd_bin(v) for v in D.map_g2]
    GC.RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(GC.RES / "E_rmsd.csv", index=False)

    def summ(g):
        f = lambda c: g[c].dropna()                                   # noqa: E731
        return dict(n=int(len(g)), median_map_g2=float(f("map_g2").median()), median_map_model=float(f("map_model").median()),
                    share_le_025=dict(g2=float((f("map_g2") <= 0.25).mean()), model=float((f("map_model") <= 0.25).mean())),
                    share_le_050=dict(g2=float((f("map_g2") <= 0.5).mean()), model=float((f("map_model") <= 0.5).mean())),
                    share_moved_gt_030=float((f("map_move") > MOVE_A).mean()),
                    n_nan=dict(g2=int(g.map_g2.isna().sum()), model=int(g.map_model.isna().sum())))
    dev = (D.map_g2 - D.map_g2_rec).abs().max()
    out = dict(all=summ(D), by_bin={b: summ(g) for b, g in D.groupby("bin")},
               sanity_fail_share=float((D.sanity != "").mean()), sanity_reasons=D.sanity[D.sanity != ""].value_counts().to_dict(),
               map_g2_vs_records_max_dev=float(dev) if np.isfinite(dev) else None, move_threshold_A=MOVE_A)
    GC.write_json(GC.RES / "E_rmsd.json", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
```

### 2-7. `analysis/geo1b/train_m1.py` (GEO2b Y 단계용, 그대로 작성)
```python
#!/usr/bin/env python3
"""GEO2b Phase Y: one extra GEO1b refiner for the D1 reactions of T1 fold -1 (docs/specs/GEO2.md 3 + GEO1B.md 5).
Run with the R environment of GEO1b: --export=ALL,GEO_MAX_EP=300,GEO_PATIENCE=100000 (same LR horizon as 'final').

The 380 D1 reactions of the six T1 fold -1 cores are never a GEO1 test fold, so no GEO1 model left them out. Their
pairs are LADDER training pairs of every fold, so the geo rung needs their structures from a model that did not see them.
  python train_m1.py plan          -> results/Y_plan.json (config = most frequent R_select choice, ties c0 < c1 < c2 < c3;
                                      epochs = median of the 5 chosen best epochs + 1, rounded down)
  python train_m1.py final <seed>  -> $GEO/preds/final_fm1_s<seed>.pkl (seeds 0, 1, 2)
Training set = every D1 reaction with fold >= 0 + D0 (overlap False); no inner validation (fixed epochs, as 'final').
GPU env (analysis/geo1b/jobs/gpu.slurm). Uses GEO1b train.py unchanged (fit / predict / train_items / eval_items / load_records).
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
    if T.MAX_EP != 300:
        raise SystemExit(f"STOP: GEO_MAX_EP must be 300 (GEO1b R setting), got {T.MAX_EP}")
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

### 2-8. commit
- `docs/specs/GEO1B.md`, `docs/specs/geo1b.patch`, `analysis/geo1b/`을 commit하고 `git push -u origin geo1b`.
- `analysis/geo1b/results/DECISIONS.md`를 새로 만든다. 첫 줄에는 "GEO1의 R-user·R-once·R-select·R-lpt·U-0·D-1·I-1 결정을 그대로 이어받음"이라고 쓴다.

---

## 3. GEO1b 실행 (A 트랙)

모든 명령은 `analysis/geo1b/`에서 낸다.
- **CPU (D1 run env):** `jobs/cpu.slurm`
- **GPU env:** `jobs/gpu.slurm`
- **파티션:** GEO1과 같다(GPU `gpu1,gpu2,gpu3,gpu5,gpu6`, CPU `cpu1,cpu2`).

### 3-1. D: 기록 만들기 (G1 없음)
```bash
sbatch -p cpu1,cpu2 --cpus-per-task=4 --mem=32G -J gb_data jobs/cpu.slurm data.py
```
`results/D_dataset_report.json`을 다음 표와 대조한다.

| 항목 | 기대 | 다르면 |
|---|---|---|
| `n` | 14,526 (GEO1 D와 같음) | STOP |
| `by_src.D1.g2` / `by_src.D0.g2` | 8,616 / 4,488 | STOP |
| `g1_used` | false | STOP |
| `d0_overlap` | 5 | STOP |
| D1 `map_rmsd_g2.n_nan` / `n_g2` | ≤ 0.05 (코드가 STOP) | STOP |
| D1 `map_rmsd_g2` 분포 | 보고만 함(10·50·90 % 지점, ≤ 0.25 / 0.5 / 0.75 / 1.0 Å 비율) | — |

- 시간: 무거운 원자 그래프 동형 RMSD 13,104개를 계산한다. 대칭이 많은 분자에서 느릴 수 있고, 48시간 안에 끝나야 한다.

### 3-2. U: 검사
```bash
sbatch -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J gb_u_synth jobs/gpu.slurm tests.py synth
sbatch -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J gb_u_data  jobs/gpu.slurm tests.py data
```

| 검사 | 기준 |
|---|---|
| U-1–U-6, U-7s | GEO1과 같음 (작성자 로컬에서 모두 PASS) |
| U-0 | 기록 수 = D 보고, `g1_used` false |
| U-7 | **보고만** 한다(GEO1 결정 U-7과 같음). 학습 G2 32개, 2,000 step의 L1 시작 → 끝 비율 |
| U-8b | G2 그대로: 시험 반응 8,425개; 모든 para 짝 **1,667**; 비율 **0.1338 ± 0.005**; 중앙값 **0.0218 ± 0.001 Å**; 형성 거리 대 C_compare ≤ 1e-4 Å |
| U-9b | fold마다 시험·내부 검증 반응이 학습에 0개; 겹치는 D0가 학습에 0개; 학습 예제는 모두 `g2`; 예제 수 = G2가 있는 학습 기록 수; G1 데이터가 있는 기록 0개; 시험 입력 = G2; 결합 그래프 = 입력에서 계산 |
| U-10 | D1 G2 입력 중 대칭 고려 RMSD가 정의되는 비율 ≥ 0.95 |

U-7을 뺀 나머지 중 하나라도 FAIL이면 STOP한다.

### 3-3. R: 학습 (launcher, GEO1과 같은 방식)
```bash
sbatch -p cpu1,cpu2 --time=48:00:00 --cpus-per-task=1 --mem=2G -J gb_rlaunch \
  --output=/gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/%x_%j.out \
  --export=ALL,GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo1b,GEO_R_MAX_EP=300,GEO_R_PATIENCE=100000,GEO_R_ONCE=1 \
  --wrap "source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh && conda activate reactot && \
          cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo1b && PYTHONDONTWRITEBYTECODE=1 python r_launch.py"
```
- launcher가 tune 20개 → select 5개 → final 15개 → `gb_eval`(evaluate.py) 순서로 낸다. smoke는 없다(`GEO_R_MAX_EP`가 있으면 R_budget은 사용자 값을 쓴다).
- launcher는 대기열 개수를 셀 때 **사용자의 모든 job**(G2c 포함)을 센다. 그래서 MaxSubmit 20을 넘지 않는다.
- 슬롯 배분은 6절을 따른다.
- select의 주 지표는 이제 **모든 para 짝** 비율이다. 이 사실을 `DECISIONS.md`에 적는다.

### 3-4. E1·E2: 평가 (설명용)
- **E1:** launcher의 `gb_eval`이 `evaluate.py`를 실행해 다음 파일을 만든다: `E_main.csv`, `E_pairs.csv`, `E_rxn.csv`, `E_by_bin.csv`, `E_by_dipole.csv`, `E_report.json`.
  - G2 그대로 기준값이 GC.G2RAW_PRIMARY와 다르면 exit 3 → STOP.
  - `E_rxn.csv`의 열 `e_short_model`과 `e_long_model`은 GEO2b `build_geo.py`가 대조할 때 쓴다.
- **E2:** E1이 끝나면 실행한다.
  ```bash
  sbatch -p cpu1,cpu2 --cpus-per-task=16 --mem=32G -J gb_rmsd jobs/cpu.slurm rmsd_b.py 16
  ```
  - `E_rmsd.csv`, `E_rmsd.json`을 만든다.
  - **`sanity_fail_share` > 0.02이면 STOP.** GEO2b `build_geo.py`의 대체 비율 상한(2 %)을 넘으므로 어차피 거기서 멈춘다. 그 전에 실패 이유별 개수를 보고한다.

### 3-5. GEO1b 요약 (`results/SUMMARY_GEO1B.md`) → commit, push → 바로 4절
- **표 1 (모든 para 짝):** 비율과 중앙값, 모델 대 G2 그대로, 차이의 core bootstrap 95 % CI. seed 0, 1, 2 각각의 값도 적는다.
- **표 2:** 반응 단위 형성 결합 오차 중앙값(짧은 결합, 긴 결합 부호 있는 값). 대칭 고려 RMSD 중앙값과 ≤ 0.25 / 0.5 Å 비율(E2).
- **표 3:** G2 대 라벨 RMSD 구간(low, mid, high)별 결과. 짝 비율과 반응 오차, 그리고 모델이 형태를 옮긴 비율(`share_moved_gt_030`).
- **표 4:** dipole별 결과.
- GPU 시간(`HPC_efficiency.csv`).
- **판정 문장은 쓰지 않는다.** 판정은 GEO2b가 한다.
- GEO1 숫자와 비교하지 않는다. GEO1은 G1을 썼기 때문이다.

---

## 4. GEO2b: GEO1b 구조로 에너지 (A 트랙, 브랜치 `geo2b`)

### 4-1. 브랜치와 코드
```bash
git switch -c geo2b geo1b
git checkout geo2 -- docs/specs/GEO2.md analysis/geo2/geo2_common.py analysis/geo2/build_geo.py analysis/geo2/eval_geo2.py
# 4-2 블록을 docs/specs/geo2b.patch 로 저장한 뒤:
git apply --check docs/specs/geo2b.patch && git apply docs/specs/geo2b.patch
grep -n '"g1", "g2"\|"same_ts_g1"\|C_COMPARE' analysis/geo2/eval_geo2.py      # 0줄이어야 한다
```
- **`geo2` 브랜치가 없거나 위 파일이 없는 경우:** `docs/specs/GEO2.md`(사용자가 가진 판)의 코드 블록에서 세 파일을 만든다.
- **`train_m1.py`는 `geo2`에서 가져오지 않는다.** `analysis/geo1b/train_m1.py`(2-7)를 쓴다.
- **`git apply --check`가 실패한 경우:** `geo2` 브랜치에서 그 파일이 고쳐진 것이다. patch의 변경을 손으로 옮긴다. 옮길 내용:
  - `geo2_common`: 세 경로를 `geo2b`, `geo1b`, `analysis/geo1b/results`로 바꾼다.
  - `build_geo`: `FORM_RANGE` 상한을 4.5 Å로 바꾼다(GEO1 결정 I-1과 맞춤).
  - `eval_geo2`:
    - `g1` rung을 뺀다.
    - 짝 분류를 C_compare `same_ts_g1` 대신 GEO1b `E_rxn.csv`의 `bin`으로 한다. 짝의 구간은 두 반응 중 더 나쁜 쪽이고, 없으면 `na`다.
    - 출력 파일 이름을 `E_class.csv`에서 `E_bin.csv`로 바꾼다.
  - 바꾼 뒤 `eval_geo2.py selftest`가 통과해야 한다.

### 4-2. `docs/specs/geo2b.patch` (그대로 저장)
```diff
--- a/analysis/geo2/geo2_common.py
+++ b/analysis/geo2/geo2_common.py
@@ -1,6 +1,6 @@
 """GEO2 common paths and helpers (docs/specs/GEO2.md). Python 3.9+ (D1 run env), reactot and env_d1fu2 compatible.
 
-Writes only under $GEO2 (= $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2) and analysis/geo2/results.
+Writes only under $GEO2 (= $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2b; GEO2b) and analysis/geo2/results.
 Every other root is read only. The roots can be overridden by environment variables (tests only)."""
 from __future__ import annotations
 
@@ -16,9 +16,9 @@
 HERE = Path(__file__).resolve().parent
 REPO = HERE.parents[1]
 RES = HERE / "results"
-GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))
-GEO1 = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))            # GEO1 scratch (read only)
-GEO1_RES = REPO / "analysis" / "geo" / "results"                                    # GEO1 results (read only)
+GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))
+GEO1 = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))          # GEO1b scratch (read only)
+GEO1_RES = REPO / "analysis" / "geo1b" / "results"                                  # GEO1b results (read only)
 G2_FINAL = Path(os.environ.get("GEO2_G2_FINAL", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final"))
 D1_JOBS = Path(os.environ.get("GEO2_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
 D1_LABELS_G2 = Path(os.environ.get("GEO2_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
--- a/analysis/geo2/build_geo.py
+++ b/analysis/geo2/build_geo.py
@@ -27,7 +27,7 @@
 FOLDS = (0, 1, 2, 3, 4)
 MIN_CONTACT = 0.60          # d_ij >= 0.60 (r_i + r_j) for every pair
 BOND_IN, BOND_OUT = 1.10, 1.40   # G2 bond (d < 1.10 sum r) must stay < 1.40 sum r; G2 non-bond (> 1.40) stays > 1.10
-FORM_RANGE = (1.4, 4.0)     # forming distances (A)
+FORM_RANGE = (1.4, 4.5)     # forming distances (A); 4.5 = GEO1 decision I-1 (11 asynchronous label TSs 4.02-4.36 A)
 FALLBACK_MAX = 0.02         # spec 4-3
 E_RXN_TOL = 1e-6            # A, forming-distance errors vs GEO1 E_rxn.csv
 
--- a/analysis/geo2/eval_geo2.py
+++ b/analysis/geo2/eval_geo2.py
@@ -1,8 +1,9 @@
 #!/usr/bin/env python3
-"""GEO2 Phase E: energy (8 EDA channels + c_ghost + barrier) of the geo rung against dft / g1 / g2 (docs/specs/GEO2.md 9).
+"""GEO2b Phase E: energy (8 EDA channels + c_ghost + barrier) of the geo rung against dft / g2 (docs/specs/GEO2.md 9,
+GEO1B.md 5-2: no g1 rung, pair bins from GEO1b E_rxn.csv instead of the G1-derived same_ts_g1 class).
 
   python eval_geo2.py selftest   # g2 predictions relabelled as rung 'geo': must reproduce E2 numbers exactly, diff = 0
-  python eval_geo2.py run        # results/E_metrics.csv, E_compare.csv, E_recovery.csv, E_class.csv, E_app.csv,
+  python eval_geo2.py run        # results/E_metrics.csv, E_compare.csv, E_recovery.csv, E_bin.csv, E_app.csv,
                                  # E_report.json
 Reads only; any pandas + pyarrow + numpy + scipy env (LADDER evaluate env 'espley' or reactot).
 """
@@ -20,7 +21,8 @@
 RES = HERE / "results"
 LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
 F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"
-C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
+GEO1B_RXN = REPO / "analysis" / "geo1b" / "results" / "E_rxn.csv"            # rxn_id -> bin (G2 vs label mapped RMSD)
+BIN_ORDER = {"low": 0, "mid": 1, "high": 2}
 BASE_PRED = LAD_RES / "L_predictions.parquet"
 BASE_UNITS = (LAD_RES / "L_report.json", LAD_RES / "L_xgb_report.json")
 GEO_PRED = LAD_RES / "L_predictions_geo.parquet"                       # stage-3 collect (spec 8-1)
@@ -87,11 +89,12 @@
     keep = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main", "core", "dipole", "skeleton", "series", "fam",
             "cfg", "orient", "face", "c3", "Xb"]
     F = F[[c for c in keep if c in F.columns]].copy()
-    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "same_ts_g1"]).set_index("rxn_id").same_ts_g1
-    C = C.map({True: True, False: False, "True": True, "False": False})
-    sa, sb = C.reindex(F.a_rxn).values, C.reindex(F.b_rxn).values
-    F["cls"] = np.where((sa == True) & (sb == True), "same",                         # noqa: E712
-                        np.where((sa == False) & (sb == False), "different", "mixed"))  # noqa: E712
+    if GEO1B_RXN.is_file():
+        B = pd.read_csv(GEO1B_RXN, usecols=["rxn_id", "bin"]).set_index("rxn_id").bin
+        ba, bb = B.reindex(F.a_rxn).fillna("na").values, B.reindex(F.b_rxn).fillna("na").values
+    else:
+        ba = bb = np.array(["na"] * len(F))
+    F["cls"] = ["na" if "na" in (x, y) else max((x, y), key=BIN_ORDER.get) for x, y in zip(ba, bb)]
     return F
 
 
@@ -193,13 +196,13 @@
     Fi = F.set_index("pair_id")
     para = [p for p in pset if Fi.at[p, "pos"] == "p"]
     rows = []
-    for rung in ("dft", "g1", "g2", "geo"):
+    for rung in ("dft", "g2", "geo"):
         if rung not in set(L.rung):
             continue
         for t in TARGETS:
             d = select(L, rung, ("REP8",), t, rep, rep8).reindex(para)
             e = np.abs(d.yhat - d.y).values
-            for cls in ("same", "mixed", "different"):
+            for cls in ("low", "mid", "high", "na"):
                 m = Fi.cls.reindex(para).values == cls
                 rows.append(dict(rung=rung, sel="REP8", target=t, cls=cls, n=int(m.sum()), mae=float(e[m].mean())))
     return pd.DataFrame(rows)
@@ -211,7 +214,7 @@
     from scipy.stats import spearmanr
     Fi = F.set_index("pair_id")
     rows = []
-    for rung in ("dft", "g1", "g2", "geo"):
+    for rung in ("dft", "g2", "geo"):
         if rung not in set(L.rung):
             continue
         for sel in (("REP8",), ("ENS4", "SH"), ("ENS4", "S")):
@@ -307,7 +310,7 @@
     rep, rep8, mval = reps(U)
     P23 = common_pairs(L, F, ("g2", "geo"))
     P023 = common_pairs(L, F, ("dft", "g2", "geo"))
-    own = {r: common_pairs(L, F, (r,)) for r in ("dft", "g1", "g2", "geo")}
+    own = {r: common_pairs(L, F, (r,)) for r in ("dft", "g2", "geo")}
     M = pd.concat([metrics(L, F, rep, rep8, ("g2", "geo"), P23, "P_23"),
                    metrics(L, F, rep, rep8, ("dft", "g2", "geo"), P023, "P_023")]
                   + [metrics(L, F, rep, rep8, (r,), own[r], f"own_{r}") for r in own], ignore_index=True)
@@ -318,7 +321,7 @@
     M.to_csv(out / "E_metrics.csv", index=False)
     C.to_csv(out / "E_compare.csv", index=False)
     recovery(M).to_csv(out / "E_recovery.csv", index=False)
-    by_class(L, F, rep, rep8, P023).to_csv(out / "E_class.csv", index=False)
+    by_class(L, F, rep, rep8, P023).to_csv(out / "E_bin.csv", index=False)
     app_metrics(L, F, rep, rep8, P023).to_csv(out / "E_app.csv", index=False)
     rep_out = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rep.items()}
     rep8_out = {f"{k[0]}|{k[1]}": list(v) for k, v in rep8.items()}
```

### 4-3. GEO2.md를 실행할 때의 바꿔 읽기 (나머지 규칙, 검사, 판정은 GEO2.md 그대로)

| GEO2.md에 적힌 것 | GEO2b에서는 |
|---|---|
| `analysis/geo` (GEO1 코드, 결과) | `analysis/geo1b` |
| `$GEO` = `/gpfs/tmp_cpu2/yeseo1ee/geo` | `$GEO1B` = `/gpfs/tmp_cpu2/yeseo1ee/geo1b` |
| `$GEO2` = `/gpfs/tmp_cpu2/yeseo1ee/geo2` | `$GEO2B` = `/gpfs/tmp_cpu2/yeseo1ee/geo2b`. `feat_geo.py`, `g3c.py` 복사본의 scratch 상수도 모두 이 경로로 한다 |
| 1절 0-1 시작 조건 (GEO1 산출물) | `analysis/geo1b/results/{E_report.json, E_rxn.csv, R_select_f0..4.json}`, `$GEO1B/preds/final_f{k}_s{0,1,2}.pkl` 15개 |
| 3절 Y: `analysis/geo/train_m1.py`, `analysis/geo/jobs/gpu.slurm` | `analysis/geo1b/train_m1.py`, `analysis/geo1b/jobs/gpu.slurm`. **`--export=ALL,GEO_MAX_EP=300,GEO_PATIENCE=100000` 필수** (final과 같은 LR 일정. 없으면 스크립트가 멈춘다) |
| 4절 Z: `build_geo.py`를 `analysis/geo/jobs/gpu.slurm`로 | `analysis/geo1b/jobs/gpu.slurm` (`GEO_SCRATCH`=geo1b)로 `cd analysis/geo2; sbatch … ../geo1b/jobs/gpu.slurm build_geo.py build`. `GEO2_SCRATCH`를 따로 주지 않으면 4-2의 기본값 `geo2b`를 쓴다 |
| 9절 `E_class.csv` (same / mixed / different) | `E_bin.csv` (low / mid / high / na). 설명용이다 |
| 9·11절 rung `g1` 비교와 그림의 g1 | 뺀다 (dft, g2, geo만) |
| 11절 5번 "이미 본 결과" (GEO1 숫자, 에너지 환산 예측) | GEO1b 결과(3-5 표 1, 2)로 바꾼다. GEO1 숫자와 예측치는 쓰지 않는다 |
| 0-4 기준값 (g2 REP8 P_23 para: oi 1.038, cpcm 0.315, barrier 0.397) | 그대로 |
| 10절 판정 (REP8, P_23 para, oi·cpcm CI_hi < 0) | **그대로.** 이것이 이번 실험의 최종 판정이다 |

- GEO2b `DECISIONS.md`의 첫 줄: "GEO2.md를 GEO1B.md 4절의 바꿔 읽기로 실행. G1 미사용".
- 요약은 GEO2.md 11절 형식으로 `analysis/geo2/results/SUMMARY_GEO2B.md`에 쓴다. commit하고 `git push -u origin geo2b`.

---

## 5. G2c: G2 TS 형태 전체 재탐색 (B 트랙, worktree `eda-asm-prediction-g2c`, 브랜치 `g2c`)

### 5-1. 규칙 (결과를 보기 전에 정함. 라벨과 G1을 읽지 않으므로 배포 가능)

| 항목 | 내용 |
|---|---|
| 후보 | 반응마다 run 3개: `orig`(기존 D1-G2 / D0-G2 final 폴더), `r1`, `r2`(새 독립 실행). `.done`인 것만 후보 |
| 새 실행 | D1-G2 / D0-G2 절차와 코드를 그대로 쓴다(autodE `calculate_reaction_profile`과 TS 형태 탐색 `find_lowest_energy_ts_conformer`, XTB2 ALPB(water), stereo, 기준 구조, final). 바뀌는 것은 scratch root뿐이다. D1은 `_graph` 수정(g2_refinalize)을 켠 상태로 돌린다(GEO1 결정 Z-graph, D0-G2와 같음) |
| 고르기 | `result.json`의 `e_ts_freq_eh`가 가장 낮은 run. 이 값은 마지막 TS에서 XTB2 ALPB(water) Freq로 구한 에너지이고, 모든 run에서 같은 ORCA 입력 헤더로 계산된다. 1e-7 Eh 안의 차이는 동률로 보고 orig < r1 < r2 순으로 고른다 |
| G2c 폴더 | 고른 run의 final 폴더를 **통째로** byte copy한다(ts.xyz, rel1, rel2, ts.hess, work/, result.json)에 `g2c.json`을 더한다. TS, 기준 구조, hess가 한 run에서 나오므로 서로 맞다 |
| 위치 | `$G2C/d1/g2_final/<rxn_id>/`, `$G2C/d0/g2_final/<rxn_id>/` (G2와 같은 형식이라 나중에 rung으로 바로 쓸 수 있다) |

### 5-2. 코드 (`g2c` worktree에 작성. `analysis/geo/d0g2.py`는 GEO1 원본을 그대로 쓴다)

`analysis/g2c/g2c_d1.py`
```python
#!/usr/bin/env python3
"""G2c track (docs/specs/GEO1B.md 6): one more independent run of the D1-G2 protocol per D1 reaction into a new
scratch root ($D1G2_SCRATCH = $G2C/d1_r<k>), compute unchanged (analysis/d1_g2/g2_autode.py imported), with the
D1-G2 _graph repair active exactly as analysis/geo/d0g2.py does (GEO1 decision Z-graph). D1 run env, sbatch only
(analysis/g2c/jobs/g2c.slurm).

  python g2c_d1.py list                                  -> $G2C/lists/d1_targets.csv (job_id, rxn_id; g2_common.targets())
  python g2c_d1.py one <job_id> <cores>                  -> $D1G2_SCRATCH/g2_final/<rxn_id>/ (.done or .fail_*)
  python g2c_d1.py pool <list.csv> <n_par> <cores> [i n] (slice i of n; stops starting new reactions after
                                                          $G2C_STOP_AFTER_H h, default 44; idempotent: finished = skip)
autodE draws a fresh numpy RandomState for every conformer (autode/conformers/conf_gen.py), so a repeat run is an
independent sample of the same protocol. Nothing here reads a label for selection; finalize maps atom order to the
label as D1-G2 did (rung g2 format).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIG_D1G2 = "/gpfs/tmp_cpu2/yeseo1ee/d1_g2"
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
ROOT = os.path.abspath(os.environ.get("D1G2_SCRATCH", ""))
if os.path.dirname(ROOT) != str(G2C) or not os.path.basename(ROOT).startswith("d1_r") or ROOT == ORIG_D1G2:
    print(f"STOP: D1G2_SCRATCH={ROOT!r} must be {G2C}/d1_r<k> (never the D1-G2 scratch)", file=sys.stderr)
    sys.exit(3)
sys.path.insert(0, str(REPO / "analysis" / "d1_g2"))
import pandas as pd  # noqa: E402
import g2_common as G  # noqa: E402
import g2_autode as GA  # noqa: E402
import g2_refinalize as GR  # noqa: E402  (sets GA._graph = GR._graph_fixed, the D1-G2 Phase G repair)

assert GA._graph is GR._graph_fixed, "the D1-G2 _graph repair (g2_refinalize) is not active"
assert os.path.abspath(str(G.G2)) == ROOT, "g2_common.G2 is not the D1G2_SCRATCH root"
STOP_H = float(os.environ.get("G2C_STOP_AFTER_H", "44"))


def cmd_list():
    out = G2C / "lists" / "d1_targets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(job_id=d["job_id"], rxn_id=int(d["rxn_id"])) for d in G.targets()]
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"{len(rows)} D1 targets -> {out}", flush=True)
    return 0


def cmd_one(T, cores):
    return GA.cmd_one(T, cores)


def cmd_pool(lst, n_par, cores, i=0, m=1):
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(m)]
    lab = {d["job_id"]: d for d in G.targets()}
    t0 = time.time()
    print(f"[g2c_d1 pool] root {ROOT}: {len(ids)} reactions, {n_par} at a time x {cores} cores", flush=True)

    def one(T):
        fin = G.G2_FINAL / str(lab[T]["rxn_id"])
        if (fin / ".done").is_file() or list(fin.glob(".fail_*")):
            return T, "skip"
        if time.time() - t0 > STOP_H * 3600:
            return T, "not_started"
        log = G.LOGS / "rxn" / f"{T}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores)],
                               stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        fs = [p.name for p in fin.glob(".*")] if fin.is_dir() else []
        return T, (fs[0] if fs else f"error rc={r.returncode}")

    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for fu in as_completed([ex.submit(one, T) for T in ids]):
            T, s = fu.result()
            k = ".fail" if s.startswith(".fail") else s
            res[k] = res.get(k, 0) + 1
            print(f"  {T}: {s}", flush=True)
    print(f"[g2c_d1 pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["list"]:
        sys.exit(cmd_list())
    if a[:1] == ["one"] and len(a) == 3:
        sys.exit(cmd_one(a[1], a[2]))
    if a[:1] == ["pool"] and len(a) in (4, 6):
        sys.exit(cmd_pool(*a[1:]))
    raise SystemExit(__doc__)
```

`analysis/g2c/select_g2c.py`
```python
#!/usr/bin/env python3
"""G2c selection and report (docs/specs/GEO1B.md 6-4). D1 run env (rmsd.py: networkx + fragmenter), sbatch only.

  python select_g2c.py check              -> results/C_progress.json (markers per run root; exit 3 if a root is unfinished)
  python select_g2c.py build              -> $G2C/{d1,d0}/g2_final/<rxn_id>/ + results/C_select.csv, C_select.json
  python select_g2c.py report <n_proc>    -> results/C_rxn.csv, C_report.json   (D1 only, descriptive)

Rule (fixed before any result; deployable: reads no label and no G1):
  candidates = the '.done' final folders of the same reaction in run orig (D1-G2 / D0-G2), r1, r2;
  chosen     = lowest result.json e_ts_freq_eh (XTB2 ALPB(water) Freq energy at the final TS, the same ORCA header in
               every run); |dE| <= 1e-7 Eh counts as a tie, broken orig < r1 < r2;
  G2c folder = byte copy of the whole chosen folder (TS, rel1, rel2, ts.hess, work/, result.json) + g2c.json.
'report' compares orig and G2c with the D1 label TS (symmetry-aware heavy-atom RMSD, forming-bond errors)."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
ORIG = {"d1": Path(os.environ.get("G2C_ORIG_D1", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final")),
        "d0": Path(os.environ.get("G2C_ORIG_D0", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2/g2_final"))}
RUNS = ("orig", "r1", "r2")
TIE_EH, EH2KCAL = 1e-7, 627.509474
D1_LABELS = Path(os.environ.get("G2C_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
D1_JOBS = Path(os.environ.get("G2C_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
RMSD_DIR = Path(os.environ.get("G2C_RMSD_DIR",
                               "/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis"))
N_BOOT, SEED = 2000, 20261030


def root(src, run):
    return ORIG[src] if run == "orig" else G2C / f"{src}_{run}" / "g2_final"


def marker(d):
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def targets(src):
    if src == "d1":
        return sorted(int(d["rxn_id"]) for d in json.loads(D1_LABELS.read_text()))
    lab = G2C / "d0_r1" / "G" / "labels_d0_g2.json"           # written by d0g2.py manifest (D0 targets, 5,260)
    return sorted(int(d["rxn_id"]) for d in json.loads(lab.read_text()))


def cmd_check():
    out, bad = {}, []
    for src in ("d1", "d0"):
        T = targets(src)
        for run in ("r1", "r2"):
            m = [marker(root(src, run) / str(r)) for r in T]
            n_mark = sum(x is not None for x in m)
            out[f"{src}_{run}"] = dict(n_targets=len(T), n_marked=n_mark, n_done=sum(x == ".done" for x in m))
            if n_mark < len(T):
                bad.append(f"{src}_{run}: {len(T) - n_mark} reactions without a marker")
    write_json(RES / "C_progress.json", dict(runs=out, unfinished=bad))
    print(json.dumps(out, indent=1), flush=True)
    return 3 if bad else 0


def energy(d):
    if marker(d) != ".done":
        return None
    e = json.loads((d / "result.json").read_text()).get("e_ts_freq_eh")
    return float(e) if e is not None and math.isfinite(float(e)) else None


def choose(E):
    ok = {k: v for k, v in E.items() if v is not None}
    if not ok:
        return None
    emin = min(ok.values())
    return next(k for k in RUNS if k in ok and ok[k] <= emin + TIE_EH)


def cmd_build():
    if cmd_check():
        print("STOP: a rerun root is unfinished (results/C_progress.json)", file=sys.stderr)
        return 3
    rows = []
    for src in ("d1", "d0"):
        for rid in targets(src):
            E = {run: energy(root(src, run) / str(rid)) for run in RUNS}
            nodone_e = [run for run in RUNS if marker(root(src, run) / str(rid)) == ".done" and E[run] is None]
            ch = choose(E)
            row = dict(src=src, rxn_id=rid, chosen=ch or "", n_done=sum(v is not None for v in E.values()),
                       **{f"e_{k}": E[k] for k in RUNS}, done_without_energy=" ".join(nodone_e),
                       dE_vs_orig_kcal=(E[ch] - E["orig"]) * EH2KCAL if ch and E["orig"] is not None else None)
            rows.append(row)
            if not ch:
                continue
            out = G2C / src / "g2_final" / str(rid)
            meta = dict(rxn_id=rid, src=src, chosen=ch, source=str(root(src, ch) / str(rid)), energies_eh=E,
                        rule="lowest e_ts_freq_eh over .done runs orig/r1/r2, tie 1e-7 Eh -> orig<r1<r2")
            if marker(out) == ".done" and (out / "g2c.json").is_file():
                old = json.loads((out / "g2c.json").read_text())
                if old.get("chosen") == ch and old.get("energies_eh") == E:
                    continue
                print(f"STOP: {out} exists with another choice; move it aside explicitly", file=sys.stderr)
                return 3
            tmp = out.with_name(f"{rid}.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(root(src, ch) / str(rid), tmp)
            (tmp / "g2c.json").write_text(json.dumps(meta, indent=1))
            if out.exists():
                shutil.rmtree(out)
            tmp.replace(out)
    D = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "C_select.csv", index=False)
    summ = {}
    for src, g in D.groupby("src"):
        o = g[g.e_orig.notna()]
        ch = g[g.chosen != ""]
        summ[src] = dict(n_targets=int(len(g)), n_g2_orig_done=int(len(o)), n_g2c_done=int(len(ch)),
                         gained=int(((g.chosen != "") & g.e_orig.isna()).sum()),
                         chosen=ch.chosen.value_counts().to_dict(),
                         changed_share_of_orig_done=float((o.chosen != "orig").mean()) if len(o) else None,
                         dE_kcal_quantiles=[float(x) for x in o.dE_vs_orig_kcal.dropna().quantile([0.1, 0.5, 0.9])]
                         if len(o) else None,
                         n_done_without_energy=int((g.done_without_energy != "").sum()))
    write_json(RES / "C_select.json", dict(summary=summ, written=time.strftime("%Y-%m-%dT%H:%M:%S")))
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return 0


# ---------------------------------------------------------------- report (D1, descriptive)
def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], float)


def _init():
    sys.path.insert(0, str(RMSD_DIR))
    global rm
    import rmsd as rm  # noqa: F401


def errs(x, lab, pairs):
    dl = np.array([np.linalg.norm(lab[1][i] - lab[1][j]) for i, j in pairs])
    dx = np.array([np.linalg.norm(x[1][i] - x[1][j]) for i, j in pairs])
    o = np.argsort(dl)
    return (dx - dl)[o]


def one(a):
    rid, job, pairs, ch = a
    lab = read_xyz(D1_JOBS / job / "ts.xyz")
    o = read_xyz(root("d1", "orig") / str(rid) / "ts.xyz") if ch is not None and \
        marker(root("d1", "orig") / str(rid)) == ".done" else None
    c = read_xyz(G2C / "d1" / "g2_final" / str(rid) / "ts.xyz")

    def m(a_, b_):
        r = float(rm.mapped_heavy_rmsd(a_, b_))
        return r if math.isfinite(r) else float("nan")
    eo = errs(o, lab, pairs) if o else [np.nan, np.nan]
    ec = errs(c, lab, pairs)
    return dict(rxn_id=rid, chosen=ch, map_orig=m(o, lab) if o else np.nan, map_g2c=m(c, lab),
                map_g2c_vs_orig=m(c, o) if o and ch != "orig" else (0.0 if o else np.nan),
                e_short_orig=eo[0], e_long_orig=eo[1], e_short_g2c=ec[0], e_long_g2c=ec[1])


def cmd_report(n_proc):
    S = pd.read_csv(RES / "C_select.csv")
    S = S[(S.src == "d1") & (S.chosen.fillna("") != "")]
    lab = {int(d["rxn_id"]): d for d in json.loads(D1_LABELS.read_text())}
    jobs = [(int(r.rxn_id), lab[int(r.rxn_id)]["job_id"],
             sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(lab[int(r.rxn_id)]["formed_pairs_ts"]).split()),
             r.chosen) for r in S.itertuples()]
    with Pool(int(n_proc), initializer=_init) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    D = pd.DataFrame(rows)
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton"])
    D = D.merge(C.assign(core=C.dipole + "/" + C.skeleton)[["rxn_id", "core"]], on="rxn_id", how="left")
    D.to_csv(RES / "C_rxn.csv", index=False)
    both = D[D.map_orig.notna() & D.map_g2c.notna()]
    rng = np.random.default_rng(SEED)
    cores = both.core.unique()
    by = {c: g for c, g in both.groupby("core")}
    diffs = {t: [] for t in (0.25, 0.5)}
    for _ in range(N_BOOT):
        g = pd.concat([by[c] for c in rng.choice(cores, len(cores), replace=True)])
        for t in diffs:
            diffs[t].append(float((g.map_g2c <= t).mean() - (g.map_orig <= t).mean()))
    fb = lambda g, k: float(np.nanmedian(np.abs(np.r_[g[f"e_short_{k}"], g[f"e_long_{k}"]])))   # noqa: E731
    rep = dict(n=int(len(D)), n_both=int(len(both)), changed_share=float((both.chosen != "orig").mean()),
               moved_conformer_share=float((both.map_g2c_vs_orig > 0.30).mean()),
               map_median=dict(orig=float(both.map_orig.median()), g2c=float(both.map_g2c.median())),
               share_le={str(t): dict(orig=float((both.map_orig <= t).mean()), g2c=float((both.map_g2c <= t).mean()),
                                      diff_ci=[float(x) for x in np.percentile(diffs[t], [2.5, 97.5])]) for t in diffs},
               form_abs_median=dict(orig=fb(both, "orig"), g2c=fb(both, "g2c")),
               n_boot=N_BOOT, seed=SEED, note="descriptive; no gate. Bins/thresholds fixed in GEO1B.md 6-4.")
    write_json(RES / "C_report.json", rep)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["check"]:
        sys.exit(cmd_check())
    if a == ["build"]:
        sys.exit(cmd_build())
    if a[:1] == ["report"] and len(a) == 2:
        sys.exit(cmd_report(a[1]))
    raise SystemExit(__doc__)
```

`analysis/g2c/jobs/g2c.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=g2c
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/g2c/logs/%x_%j.out
# G2c reruns / selection (docs/specs/GEO1B.md 6). D1 run env. Submit from the folder of the script you run:
#   D1 rerun : cd analysis/g2c; sbatch -J g2c_d1_r1 -p <P> --cpus-per-task=<C> --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r1 \
#              jobs/g2c.slurm g2c_d1.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/lists/d1_targets.csv <n_par> <cores> <i> <n>
#   D0 rerun : cd analysis/geo;  sbatch -J g2c_d0_r1 -p <P> --cpus-per-task=<C> --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1 \
#              ../g2c/jobs/g2c.slurm d0g2.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1/d0_manifest.csv <n_par> <cores> <i> <n>
#   select   : cd analysis/g2c; sbatch -J g2c_sel -p <P> --cpus-per-task=16 --export=ALL,G2C_ROOT=none jobs/g2c.slurm select_g2c.py report 16
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
: "${G2C_ROOT:?G2C_ROOT must be exported (a $G2C/<src>_r<k> root, or none for select)}"
export G2C_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/g2c PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 G2C_STOP_AFTER_H=44 D0G2_STOP_AFTER_H=44
if [ "$G2C_ROOT" != none ]; then
  case "$G2C_ROOT" in /gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r[12]|/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r[12]) ;; *) echo "STOP: bad G2C_ROOT $G2C_ROOT"; exit 3;; esac
  export D1G2_SCRATCH="$G2C_ROOT" D0G2_SCRATCH="$G2C_ROOT" GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D1G2_JOB_T0=$(date +%s)
fi
echo "[g2c] host=$(hostname) job=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} root=$G2C_ROOT git=$(git rev-parse --short HEAD 2>/dev/null)" \
     "$($PY -V 2>&1) $(date '+%F %T')"
"$PY" "$@"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
```
- commit할 것: `docs/specs/GEO1B.md`, `analysis/g2c/`. 그다음 `git push -u origin g2c`.
- **G2c job이 도는 동안 이 worktree의 `analysis/d1_g2/`, `analysis/geo/d0g2.py`, `analysis/g2c/g2c_d1.py`를 고치지 않는다.** 반응마다 이 파일들을 다시 읽기 때문이다.

### 5-3. 준비와 실행
1. **목록과 manifest** (`g2c` worktree에서)

   ```bash
   cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c/analysis/g2c
   sbatch -p cpu1,cpu2 --cpus-per-task=1 --mem=4G -J g2c_list --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r1 jobs/g2c.slurm g2c_d1.py list
   cd ../geo
   for k in 1 2; do
     sbatch -p cpu1,cpu2 --cpus-per-task=1 --mem=4G -J g2c_d0man_r$k --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r$k \
       ../g2c/jobs/g2c.slurm d0g2.py manifest
   done
   ```

   기대값 (다르면 STOP):
   - `lists/d1_targets.csv` 9,266행
   - `d0_r{1,2}/d0_manifest.csv` 각 5,260행
   - `d0_r{1,2}/G/labels_d0_g2.json` 각 5,260행

2. **job 크기:** GEO1 Phase Z(D0-G2)의 배치를 기준으로 한다. 그때는 task 하나에 32 CPU, pool 4 × 8코어였다(GEO1 결정 Z-4).
   - 1-3의 `sinfo`에서 노드가 더 크면 한 job을 노드 하나 전체로 잡는다(`--cpus-per-task` = 노드 코어 수, `n_par` = 코어 수 ÷ 8, `cores` = 8). 그러면 같은 슬롯으로 더 많이 계산한다.
   - 정한 값을 `analysis/g2c/results/DECISIONS.md`에 적는다.
3. **slice 개수:** root마다 slice 개수 n은 각 job이 약 40시간 안에 끝나도록 정한다.
   - 반응당 평균 코어시간: D1 0.336(C_compare), D0 1.37(GEO1 Z).
   - pool은 44시간이 지나면 새 반응을 시작하지 않는다. 남은 반응은 같은 명령을 다시 내면 이어서 한다(끝난 반응은 건너뜀).
4. **순서:** D1 r1 → D1 r2 → D0 r1 → D0 r2. 동시 job 수는 6절을 따른다.

   ```bash
   # D1 (g2c 폴더에서), slice i = 0..n-1
   sbatch -p <P> --cpus-per-task=<C> --mem=<M> -J g2c_d1_r1_$i --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d1_r1 \
     jobs/g2c.slurm g2c_d1.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/lists/d1_targets.csv <n_par> 8 $i <n>
   # D0 (geo 폴더에서)
   sbatch -p <P> --cpus-per-task=<C> --mem=<M> -J g2c_d0_r1_$i --export=ALL,G2C_ROOT=/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1 \
     ../g2c/jobs/g2c.slurm d0g2.py pool /gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1/d0_manifest.csv <n_par> 8 $i <n>
   ```

5. **첫 검사 (멈추지 않고 보고만):** 각 root에서 처음 끝난 반응 2개의 `jobs/<T>/inputs.tar.gz` 안 `.inp`의 `!` 줄 집합이 기존 `d1_g2` 또는 `d0_g2`의 같은 반응과 같은지 본다. 다르면 STOP한다. 환경이 바뀐 것이다.
6. **진행 확인:** `select_g2c.py check`를 하루 한 번 sbatch로 돌린다. 각 root의 성공률을 기존 G2와 비교해 보고한다(D1 0.930, D0 0.853). 파이프라인 오류(`error_*`, `missing`)가 5 %를 넘으면 STOP한다.

### 5-4. 선택과 보고 (4개 root가 모두 끝난 뒤)
```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c/analysis/g2c
sbatch -p cpu1,cpu2 --cpus-per-task=2 --mem=8G -J g2c_build --export=ALL,G2C_ROOT=none jobs/g2c.slurm select_g2c.py build
sbatch -p cpu1,cpu2 --cpus-per-task=16 --mem=32G -J g2c_report --dependency=afterok:<build> --export=ALL,G2C_ROOT=none \
  jobs/g2c.slurm select_g2c.py report 16
```
- `build` 출력: `C_select.csv`, `C_select.json`.
  - D1·D0별: 바뀐 비율(orig가 아닌 run이 뽑힌 비율), 보충된 반응 수, 에너지 차이 분포.
- `report` 출력: `C_rxn.csv`, `C_report.json`. D1만, 설명용이다.
  - 라벨 대비 대칭 고려 RMSD: 중앙값, ≤ 0.25 / 0.5 Å 비율. orig 대 G2c, 차이의 core bootstrap 95 % CI.
  - 형태가 실제로 바뀐 비율(G2c 대 orig RMSD > 0.30 Å).
  - 형성 결합 오차 중앙값.
- 요약은 `analysis/g2c/results/SUMMARY_G2C.md`에 쓴다. commit하고 push한 뒤 **멈춘다.** G2c로 구조 보정을 다시 학습할지(GEO1c)는 사용자가 정한다.

---

## 6. HPC 규칙과 슬롯 배분
- sbatch만 쓴다. 모든 job은 `--time=48:00:00`. login node에서는 편집, git, squeue, scancel, sbatch만 한다. `du`, `sinfo`, `sacct`도 sbatch 안에서 돌린다.
- **MaxJobs 10, MaxSubmit 20.** A 트랙(GPU 학습, GEO2b)을 먼저 채운다:

| 시기 | G2c가 대기열에 둘 수 있는 job 수 (실행 + 대기) |
|---|---|
| GEO1b R이 도는 동안 | **3** |
| GEO1b E 이후 ~ GEO2b의 CPU 단계(G·F·H) 시작 전 | 6 |
| GEO2b의 G·F·H가 도는 동안 | 3 |
| GEO2b M3(GPU) 이후 | 6 |

- G2c job 제출도 sbatch로만 한다. 슬롯을 지키기 위해 GEO1 R-once와 같은 1-CPU launcher를 써도 된다. 쓰면 `DECISIONS.md`에 적는다.
- 같은 job을 두 번 내지 않는다. 이름이 대기열에 있으면 내지 않는다.

---

## 7. STOP 조건 (여기서만 멈추고 묻는다)

| 단계 | 조건 |
|---|---|
| 0 | 시작 조건 1-5 중 하나라도 아님. 디스크 부족(1-4). GEO2 job 취소 실패 |
| D | n, G2 수, 겹침이 3-1 표와 다름. `g1_used` ≠ false. 원소 순서 불일치 > 1 %. 대칭 고려 RMSD 미정의 > 5 % |
| U | U-0, U-1–U-6, U-7s, U-8b, U-9b, U-10 중 FAIL |
| R | launcher의 `STOP_R.md` (같은 task가 두 번 잃어버려짐). NaN 손실. 시험 반응이 학습 목록에 들어감 |
| E1 | G2 그대로 기준값 불일치(exit 3). 모델 예측이 빠진 시험 반응 |
| E2 | `sanity_fail_share` > 0.02 |
| GEO2b | GEO2.md 14절의 STOP과 같음(바꿔 읽기 적용). selftest 실패 |
| G2c | 목록·manifest 행 수가 다름. 첫 검사의 입력 줄이 다름. 파이프라인 오류 > 5 %. `check`나 `build`가 exit 3 |
| 공통 | 버그 발견(고치기 전 보고). 예상 비용(9절)의 2배를 넘음 |

---

## 8. 보고
- 각 단계가 끝날 때마다 결과 파일, `DECISIONS.md`, `HPC_efficiency.csv`(sacct)를 commit하고 push한다.
- 사용자에게는 세 번 보고한다. 각 보고는 짧게, 표 위주로 한다:
  - **GEO1b 요약 (3-5):** 끝나면 바로 GEO2b로 넘어간다.
  - **GEO2b 판정:** GEO2.md 10절 문장 그대로 쓴다.
  - **G2c 요약 (5-4):** 보고 후 멈춘다.
- 숫자에는 n과 CI를 같이 적는다. 구조 지표로 판정 문장을 쓰지 않는다.

---

## 9. 예상 비용 (작성자 추정)

| 단계 | 근거 | 추정 |
|---|---|---|
| GEO1b D | RMSD 13,104개 | CPU 수 시간 |
| GEO1b R | GEO1 tune 247 + final 103 GPU시간 × 학습 예제 비율 0.675 (13,104 ÷ 19,409), ablation 없음 | **약 240 GPU시간**. 동시 6–7개면 약 2일 |
| GEO1b E1, E2 | — | CPU 수 시간 |
| GEO2b | GEO2.md 12절 (Y는 위 비율 적용 약 15 GPU시간) | GEO2.md와 같음 (H 상한 2,500 코어시간, M3 상한 80 GPU시간) |
| G2c D1 | 9,266 × 0.336 코어시간 × 2 run | **약 6,200 코어시간** |
| G2c D0 | 7,201 코어시간(GEO1 Z 실측) × 2 run | **약 14,400 코어시간** |
| G2c build, report | — | 약 50 코어시간 |
| 합계 CPU (G2c) | | **약 20,700 코어시간.** 동시 3–6 job, 노드 크기에 따라 약 3–7일 |
