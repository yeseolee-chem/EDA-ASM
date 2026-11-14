# GNN3R2 — c3 판정 오류 2건을 고쳐 짝을 다시 만들고 GNN3를 다시 학습 — Claude Code 실행 지시서

**저장소:** `origin/gnn3r` HEAD(`4c8709c`)에서 새 브랜치 **`gnn3r2`**를 만든다. 새 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r2`(이하 `$WT`).
**이 파일:** `docs/specs/GNN3R2.md`로 commit한다.
**새 코드:** `analysis/gnn3r2/` · **결과:** `analysis/gnn3r2/results/` · **scratch:** `$GR2=/gpfs/tmp_cpu2/yeseo1ee/gnn3r2`
**읽기만 (X 단계 전까지):** 이전 결과 GNN3R(`analysis/gnn3r/`, `$GR=/gpfs/tmp_cpu2/yeseo1ee/gnn3r`), `$G2R`, `$LAD`, `$D1PAIR_SCRATCH`, D1 run, `gnn3/` 패키지, `analysis/` 아래 기존 코드. **기존 코드는 고치지 않는다.** `analysis/gnn3r/`의 코드를 `analysis/gnn3r2/`로 복사하고, 2-3절 표의 곳만 바꾼다.

**순서:** 1 시작 조건 → 2 브랜치·복사 → A(c3 다시 판정) → B(짝 + GNN3R과 대조) → S(분할) → D(데이터 확인) → K(교차검증, GPU) → E(평가) → W(최종 모델, GPU) → R(보고·commit·push) → X(이전 GNN3R 삭제) → **멈춤**
- 12절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산과 삭제는 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출, git, `squeue`·`scancel`만 한다.
- 바꾸는 것은 **c3 판정 규칙 하나**다. 나머지 규칙은 모두 GNN3R(`docs/specs/GNN3R.md` 0-2절 D-1 … D-11) 그대로다.

---

## 0. 무엇을, 왜

### 0-1. 목표

1. c3(C3 치환기 X와 옆 C4 기준 치환기가 고리의 같은 면인지)를 **구조식 기반의 안정한 규칙**으로 다시 판정한다. 그 결과로 c3가 틀렸던 반응 2개(T10104, T10116)를 바로잡는다.
2. 바로잡은 행 속성으로 짝을 다시 만들어 데이터셋을 고정하고(`results/F_pairs_r.csv`), GNN3를 5겹 교차검증 + 최종 모델로 다시 학습한다.
3. 끝나면 이전 GNN3R(scratch, worktree, 브랜치, `analysis/gnn3r/`)을 지운다(X).

### 0-2. 무엇이 틀렸나 (GNN3R 결과 검토, 2026-11-14)

| 반응 | 골격 | X | GNN3R c3 | 실제 배치 |
|---|---|---|---|---|
| T10104 | azomethine ylide / B′ / endo, cfg t | o-SMe | cis | **trans** |
| T10116 | azomethine ylide / B′ / endo, cfg t | o-tBu | cis | **trans** |

- **원인:** 기존 c3 = 생성물 전체 분자의 MMFF 배치 **하나**(seed 7)에서 X와 C4 기준 치환기가 고리 평균면의 같은 쪽인지 본다. B′처럼 C4가 붐비고 X가 ortho 치환 아릴이면, 배치에 따라 답이 뒤집힌다. T10104는 seed 7, 1, 2, 3, 42에서 각각 cis, trans, trans, trans, trans로 나왔다.
- **실제 배치의 근거:** X를 CH₃로 바꾼 뼈대(core)를 구조식(입체 표기 포함, 거울상 동일 취급)으로 비교하면, 두 반응의 뼈대는 같은 면(azomethine ylide / B′ / endo)의 trans 67개와 같고, cis 5개와 다르다.
- **GNN3R에 준 영향:**
  - T10116이 cis 계열(cfg t, endo, cis)의 anchor 기준으로 뽑혔다.
  - 실제로 cis인 5개(T10047 o-Br, T10056 o-CHO, T10067 o-Cl, T10092 o-OH, T10111 o-Ts)가 기준과 입체가 달라 P-4에서 모두 버려졌다.
  - T10104 → T10116 짝은 둘 다 trans라 비교 자체는 맞지만, 계열이 틀렸다.
- **나머지는 맞다:** 데이터셋 안 아릴·anchor 짝 2,018개 모두 a와 b의 뼈대 입체가 같다. c3가 있는 2,653행 중 다른 행의 c3도 모두 맞다.

### 0-3. 고친 규칙 (D-12)

- **비유:** 붐비는 손님이 경계선에 걸쳐 앉으면 사진 한 장으로는 어느 쪽인지 헷갈린다. 손님을 작은 표지(CH₃)로 바꿔 같은 의자에 두고 카메라 5대로 본다.
- **규칙 `c3_core`:**
  1. C3, X의 첫 원자(xs), C4, C4 기준 치환기(ms4)를 기존과 똑같이 찾는다: `backbone_in_product`, `c3_orientation`의 원자 선택, B′는 GNN3R의 메틸 규칙(`c3_orientation_bp`).
  2. **core 만들기:** 생성물에서 X 가지(xs에서 C3를 거치지 않고 닿는 원자) 중 xs만 남기고 모두 지운다. xs는 sp3 탄소(CH₃: 원자번호 6, 전하 0, 방향족 아님, 암시 수소)로 바꾼다. C3–xs는 단일 결합이다. 편집 전에 Kekulé화하고, 다른 원자의 입체 표기는 그대로 둔다.
  3. **판정:** core에 수소를 붙이고, seed (7, 11, 23, 42, 97) 각각으로 `EmbedMolecule` + `MMFFOptimizeMolecule`. 고리 법선(5원자 SVD)과 판정(u·v > 0 → cis)은 기존과 같다.
  4. **5개가 모두 같아야 한다.** 그 값이 c3다. 다르면 "?"(STOP).
  5. C3에 치환기가 없는 X = H 행은 c3 = ""(GNN3R DEC-1과 같음).
- **적용 범위:** `need` 행(nitrone·azomethine ylide·carbonyl ylide, series R1, skeleton AppE·AppZ·B·Bp) 중 C3 치환기가 있는 **2,653행 전부**. B′만이 아니라 같은 규칙을 모두에 쓴다.
- **로컬 확인** (d1 생성물 SMILES, RDKit):
  - 서로 다른 core 72개, 모든 core에서 seed 5개의 답이 같음.
  - GNN3R c3와 2,651행 같음. 바뀌는 행은 **T10104, T10116 두 개(cis → trans)뿐**.

### 0-4. 기대 변화

| 짝 | 변화 |
|---|---|
| T10104 → T10116 | **빠짐** |
| T10104 → T07195, T10116 → T07195 | **추가.** trans 계열(cfg t, endo, trans)에 Ph가 없어 anchor 계열이고, b = T07195(m-Me, D-7 순위 1위)는 그대로 |
| T10047, T10067, T10092, T10111 → T10056 | **추가.** cis 계열의 사용 가능 행 5개, b = T10056(o-CHO, D-7 순서: Me·CO2Me·tBu 없음 → CHO) |

새 짝 6개가 모두 P-4를 통과할 때의 기대값:

| 항목 | GNN3R | GNN3R2 기대 |
|---|---|---|
| 반응 | 8,589 | **8,594** (+5: T10047, T10056, T10067, T10092, T10111) |
| 짝 | 8,339 | **8,344** |
| 기준 짝 / anchor 짝 | 6,978 / 1,361 | 6,978 / 1,366 |
| fold 1 test 짝 (azomethine ylide/B′는 fold 1) | 1,700 | 1,705 |
| `p4_dropped` 행 | 16 | 11 |

- 그 밖의 짝은 모두 GNN3R과 **바이트 단위로 같은 값**이어야 한다(G-1). pair_id 번호는 정렬 때문에 달라질 수 있으므로 (a_job, b_job)으로 비교한다.
- 새 짝 6개는 P-4에서 버려질 수 있다(D-8 규칙 그대로). 버려진 짝만큼 위 수가 줄어든다.

### 0-5. 그대로 두는 결정

| # | 내용 |
|---|---|
| D-1 … D-11 | GNN3R과 같다: 대상 9,252, g2r 입력, cr2 라벨, 짝 규칙, anchor 순서, fold 표, 평가 집합, GNN3 상수, RTX 3090 |
| D-8 | P-4 v2 mismatch는 버린다. 나머지 P-4 버림 11짝(T01318, T09972, T10078, T10523, T10560, T10596, T10683, T10719, T10744, T10748, T10756)은 이번 수정과 무관하며 그대로 둔다 |
| D-13 | 이전 GNN3R은 새 결과가 모두 통과하고 push된 뒤 지운다(X). 사용자 방침(2026-11-13): 혼동 방지를 위해 이전 것은 끝난 뒤 삭제 |

---

## 1. 시작 조건 (하나라도 아니면 STOP) → `results/S0_start.json`

| # | 조건 |
|---|---|
| 1-1 | `origin/gnn3r` = `4c8709c` 또는 그 뒤. `analysis/gnn3r/results/RUN_REPORT_GNN3R.md`의 모든 단계가 통과 |
| 1-2 | GNN3R 파일 sha256: `analysis/gnn3r/results/F_pairs_r.csv` `5597ad11558efb1bc3fc6322f99b5920026166e161a1df383662c6905b95a467`, `B_row_status.csv` `9207e71631e5e8883c9b6403c09bafa219601143cbc44a6310455b8cad514efe`, `$GR/A/rowattr_r.pkl` `f0e7fe41001a933d20af756be5db724eb6db33ef40b36109c88ed2657b7e81e3`. `$GR/B/p4_cache.pkl`이 있고 반응 8,605개 이상이 들어 있다 |
| 1-3 | GNN3R 시작 조건 1-2 … 1-10(`docs/specs/GNN3R.md` 1절)이 그대로 성립한다: g2r 입력, cr2 라벨, rowattr, D1PAIR 결과·코드 sha256, D1 jobs, 참조 목록, 환경 |
| 1-4 | `PYENV=reactot`의 RDKit 판 = `2026.03.2`(GNN3R `A_report.json`과 같음) |
| 1-5 | `$GR2`가 없거나 비어 있다. 디스크 여유 20 GB 이상 |

---

## 2. 브랜치 · 복사 · 경로

### 2-1. 브랜치

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction        # 주 저장소 (실제 경로 확인)
git fetch origin gnn3r:refs/remotes/origin/gnn3r
git worktree add -b gnn3r2 /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r2 origin/gnn3r
sbatch -p cpu1,cpu2 --time=48:00:00 --wrap "mkdir -p $GR2/logs $GR2/A $GR2/B $GR2/cv $GR2/final $GR2/ref_gnn3r"
```
- 이 지시서를 commit한다.
- 제출할 때 항상 `--output=$GR2/logs/%x_%j.out`(배열은 `%x_%A_%a.out`)으로 덮어쓴다.

### 2-2. 복사 (CPU job 하나, `analysis/gnn3r2/setup_r2.py`)

| 원본 | 사본 | 확인 |
|---|---|---|
| `analysis/gnn3r/*.py`, `analysis/gnn3r/jobs/` (`delete_old.py` 제외) | `analysis/gnn3r2/` | 파일별 sha256을 `results/COPY.json`에 기록 |
| `analysis/gnn3r/results/{F_pairs_r.csv, B_row_status.csv, B_reactions.csv, S_folds_r.csv, eval/metrics.csv, RUN_REPORT_GNN3R.md, DELETE_old_gnn3.json}` | `analysis/gnn3r2/ref_gnn3r/` | sha256 = 원본 (commit함. X 단계 뒤에도 대조가 다시 돌도록) |
| `$GR/A/rowattr_r.pkl` | `$GR2/ref_gnn3r/rowattr_r_gnn3r.pkl` | sha256 `f0e7fe41…` |
| `$GR/B/p4_cache.pkl` | `$GR2/B/p4_cache.pkl` | sha256 = 원본. P-4 링 정보는 DFT TS 파일에서만 나오고 c3와 무관하므로 그대로 다시 쓴다 |

### 2-3. 사본에서 바꿀 곳 (이 밖의 수정은 실행 전에 `results/DECISIONS.md`에 적고 보고)

| 파일 | 바꿀 곳 |
|---|---|
| `gr_common.py` | `GR` = 환경 변수 `GNN3R2_SCRATCH`, 기본값 `/gpfs/tmp_cpu2/yeseo1ee/gnn3r2`(예전 `GNN3R_SCRATCH`는 읽지 않음). `REF_GNN3R = HERE / "ref_gnn3r"`, `ROWATTR_GNN3R = GR / "ref_gnn3r" / "rowattr_r_gnn3r.pkl"`, `CORE_SEEDS = (7, 11, 23, 42, 97)`, `EXPECT_C3_CHANGED = {"T10104": ("cis", "trans"), "T10116": ("cis", "trans")}`, `EXPECT_REACTIONS, EXPECT_PAIRS = 8594, 8344`, `RANGE_REACTIONS, RANGE_PAIRS = (8422, 8766), (8177, 8511)`(± 2 %), `EXPECT_TEST = {-1: 169, 0: 1682, 1: 1705, 2: 1414, 3: 1344, 4: 2030}`. 문서 문자열의 지시서 이름 |
| `rowattr_r.py` | `c3_core`(0-3절)를 더하고 3절 규칙대로 적용. A-3, A-5를 3절 표대로 바꾼다. 출력 `results/A_c3_core.csv` |
| `pairs_r.py` | 로직 변경 없음 (경로는 `gr_common`에서 옴) |
| `s0_start.py` | 1절 조건 |
| `report_r.py`, `gnn3r_manifest.py` | 이름 GNN3R2, manifest 파일 `results/gnn3r2.json`, 보고에 c3 수정과 GNN3R 대조(4-2절) 추가 |
| 새 파일 `diff_gnn3r.py` | 4-2절 G-1 … G-3 |
| 새 파일 `delete_gnn3r.py` | 11절 |
| `splits_r.py`, `check_data.py`, `train_cv_r.py`, `train_final_r.py`, `check_cv.py`, `evaluate_r.py`, `jobs/` | 바꾸지 않음 |

---

## 3. Phase A — c3 다시 판정 (`analysis/gnn3r2/rowattr_r.py`, `PYENV=reactot`, CPU)

1. GNN3R과 같이 `ROWATTR_ALL`에서 9,252행(job_id 순)을 고른다.
2. A-2(복사한 `c3_orientation`이 저장된 c3를 재현)를 GNN3R과 같이 확인한다.
3. `need` 행 중 C3 치환기가 있는 행 전부에 `c3_core`를 적용한다. X = H 행은 ""다. 다른 열은 손대지 않는다.
4. 출력: `$GR2/A/rowattr_r.pkl`, `results/A_c3_core.csv`(job_id, c3_gnn3r, c3_core, core SMILES(입체 표기 포함 정규형), seed별 판정 5개), `results/A_report.json`.

| # | 검사 | 실패 시 |
|---|---|---|
| A-1 | GNN3R A-1과 같음 (9,252행, job_id 집합 = `X_targets.csv`) | STOP |
| A-2 | GNN3R A-2와 같음 (`need` 2,667행 차이 0) | STOP |
| A-3 | C3 치환기가 있는 `need` 행 2,653개: `c3_core` ∈ {cis, trans}, 행마다 seed 5개 판정이 같음. X = H 행 14개는 "". 서로 다른 core 수를 기록한다(로컬 72). 서로 거울상인 core는 c3가 같다 | STOP |
| A-4 | GNN3R A-4와 같음 (참조 목록 `c3_est` 356행과 같음) | STOP |
| A-5 | GNN3R `rowattr_r_gnn3r.pkl`과 비교: 행·열이 같고, c3를 뺀 모든 열이 같다. c3가 다른 행 = `EXPECT_C3_CHANGED`와 **정확히** 같다(T10104, T10116: cis → trans) | 다른 행이 더 있거나 적으면 STOP (목록 첨부) |
| A-6 | GNN3R A-6과 같음 (F 안 Ph 중복 0) | STOP |

---

## 4. Phase B — 짝 = 데이터셋 고정 (`PYENV=reactot`, CPU)

### 4-1. 짝 만들기

`analysis/gnn3r2/pairs_r.py build` → `pairs_r.py rerun`(B-R). 규칙과 검사 B-0 … B-7, B-R은 GNN3R 4절 그대로다. 입력 rowattr만 `$GR2/A/rowattr_r.pkl`이다. P-4는 복사한 캐시를 쓴다(새로 계산할 반응 0 기대).

### 4-2. GNN3R과 대조 (`analysis/gnn3r2/diff_gnn3r.py`) → `results/G_diff_gnn3r.csv`, `results/G_report.json`

- **바뀌는 계열 T:** 아릴 행(pos p·m·o) 중 F = (azomethine ylide, Bp, R1, t, orient, endo, c3)이고 c3 ∈ {cis, trans}인 두 계열. orient는 T10104의 값이다.
- 직접 치환기 짝(b = H)은 H 후보가 c3로 좁혀지지 않으므로(H 행 c3 = "") 바뀌지 않는다.

| # | 검사 | 실패 시 |
|---|---|---|
| G-1 | a가 T에 속하지 않는 짝: GNN3R(`ref_gnn3r/F_pairs_r.csv`)과 (a_job, b_job) 집합이 같고, 짝마다 ref_kind, fam, core, pos, Xb, sigma, p4_v2, dd 10개가 **완전히 같다** | STOP (다른 짝 목록 첨부) |
| G-2 | a가 T에 속하는 짝 ⊆ 0-4절의 추가 짝 6개. 6개 중 빠진 짝은 모두 P-4 v2 = mismatch로 버려진 것이다. T10104 → T10116 짝은 없다 | STOP |
| G-3 | `B_row_status.csv`가 GNN3R과 다른 행 ⊆ {T10047, T10056, T10067, T10092, T10111, T10104, T10116} (status 또는 fam) | STOP |
| G-4 | 반응·짝 수 = 0-4절 기대값 − (G-2에서 P-4로 빠진 만큼). 기준 짝 6,978 그대로 | STOP |

- G-1 … G-4가 모두 통과해야 `results/B_label_report.json`(`all_pass` true)을 쓴다. `pairs_r.py`가 B 검사 뒤 이 파일을 쓰는 구조라면, G가 실패할 때 지운다.

---

## 5. Phase S — 분할 · 평가 속성

`analysis/gnn3r2/splits_r.py`(`PYENV=espley`). 규칙과 검사 S-1 … S-4는 GNN3R 5절 그대로다.
- 기대 test 크기: f0 1,682 · **f1 1,705** · f2 1,414 · f3 1,344 · f4 2,030 · 늘 학습 169. P-4로 빠진 짝이 있으면 f1만 그만큼 줄어든다.
- fold 1을 뺀 fold의 test 목록은 GNN3R과 (a_job, b_job) 기준으로 같아야 한다(S-5, 다르면 STOP).

---

## 6. Phase D — 데이터 확인

`analysis/gnn3r2/check_data.py`(env_d1fu2, CPU). 검사 D-1 … D-4는 GNN3R 6절 그대로다(`gr_common.dataset()`이 `analysis/gnn3r2/results`를 읽는다).

---

## 7. Phase K — 교차검증 학습 (RTX 3090)

모든 fold를 다시 학습한다(fold 1의 test와 다른 fold들의 학습 집합이 바뀜). 제출 전 `squeue -u $USER -h | wc -l` + 5 ≤ 20 확인.

```
sbatch --array=0-4%5 -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3r2_cv \
  --output=$GR2/logs/%x_%A_%a.out --export=ALL,GNN3_SCRATCH=$GR2,GNN3R2_SCRATCH=$GR2 \
  gnn3/jobs/run.slurm analysis/gnn3r2/train_cv_r.py --fold-from-array --out $GR2/cv
```
검사 K-1 … K-3(`analysis/gnn3r2/check_cv.py`)은 GNN3R 7절 그대로다. 결과는 `$GR2/cv`에만 쓴다(`$GR/cv`에 쓰면 STOP).

---

## 8. Phase E — 평가

`analysis/gnn3r2/evaluate_r.py`(env_d1fu2, CPU) → `results/eval/`. 검사 E-1: 평가한 짝 수 = 짝 수 − fold −1 짝 수.

---

## 9. Phase W — 최종 모델 (RTX 3090)

`analysis/gnn3r2/train_final_r.py`. K 배열이 끝난 뒤 `--dependency=afterok:<배열 job id>`, `-p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3r2_final --output=$GR2/logs/%x_%j.out --export=ALL,GNN3_SCRATCH=$GR2,GNN3R2_SCRATCH=$GR2`. 검사 W-1(`n_train_pairs` = 짝 수, RTX 3090)은 GNN3R 9절 그대로다.

---

## 10. Phase R — 보고 · manifest · commit · push

1. `results/RUN_REPORT_GNN3R2.md` (한국어, 표):
   - **c3 수정:** 바뀐 행 표(job_id, X, c3 이전 → 이후, core SMILES), 규칙 요약, core 수, seed 일치.
   - **데이터셋:** 반응·짝 수와 0-4절 기대값 대조, 새 짝 6개의 P-4 결과, G-1 … G-4 판정.
   - **학습:** GNN3R 10절과 같은 표.
   - **성능:** `results/eval/metrics.csv`의 P_main·test_all, 4 target의 MAE·r. 옆에 GNN3R 값(`ref_gnn3r/metrics.csv`)을 참고로 둔다.
2. `results/gnn3r2.json`: `gnn3r.json`과 같은 형태로, 경로를 `$GR2`와 `analysis/gnn3r2`로 바꾸고 c3 규칙(D-12)을 더한다.
3. commit: `analysis/gnn3r2/` 코드·job·results·`ref_gnn3r/`. `.pt`와 scratch 파일은 commit하지 않는다. `git push origin gnn3r2`. push한 HEAD를 보고에 적어 한 번 더 commit·push.

---

## 11. Phase X — 이전 GNN3R 삭제 (D-13)

**조건 (하나라도 아니면 STOP):**
- 1 … 10절 검사가 모두 통과했다.
- `origin/gnn3r2`가 push되어 있다.
- `git merge-base --is-ancestor origin/gnn3r origin/gnn3r2`가 참이다(커밋 기록이 사라지지 않음).

| 대상 | 방법 |
|---|---|
| `$GR = /gpfs/tmp_cpu2/yeseo1ee/gnn3r` 전체 (A, B, cv, final `.pt`, logs) | `analysis/gnn3r2/delete_gnn3r.py`를 sbatch로 실행. 지우기 전 파일 수·바이트를 기록하고, 지운 뒤 없음을 확인 |
| worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r` | 로그인 노드 git: `git worktree remove …` |
| 브랜치 `gnn3r` | `git branch -D gnn3r`, `git push origin --delete gnn3r` |
| `gnn3r2` 안의 `analysis/gnn3r/` 폴더 | `git rm -r analysis/gnn3r` (대조용 사본은 `analysis/gnn3r2/ref_gnn3r/`에 있음, 기록은 git history에 남음) |
| 남김 | `docs/specs/GNN3R.md`, `docs/specs/gnn3r_ref/`(사본 코드가 읽음), `gnn3/` 패키지, `$GR2/ref_gnn3r/` |

- 기록 `results/DELETE_old_gnn3r.json`(요청 문구, 대상별 전후 크기, git 명령, 남긴 것)을 commit·push하고 **멈춘다.**

---

## 12. STOP 조건

| 단계 | 조건 |
|---|---|
| 1 | 1-1 … 1-5 |
| 2 | 복사 sha256 불일치, 2-3절 밖의 코드 수정이 필요함 |
| A | A-1 … A-6. 특히 A-5(c3가 바뀐 행이 T10104·T10116 두 개가 아님) |
| B | B-0 … B-7, B-R, G-1 … G-4 |
| S | S-1 … S-5 |
| D | D-1 … D-4 |
| K | K-2, K-3, 같은 fold 두 번 실패, `$GR` 아래에 쓰려 함 |
| E · W | E-1, K 재확인 실패, W-1 |
| X | 11절 조건 |
| 공통 | 기존 코드(`analysis/gnn3r/` 포함)나 GNN3 상수를 바꿔야만 진행할 수 있음 |

### 자원

| 단계 | 자원 | 예상 |
|---|---|---|
| A, B | cpu1·cpu2, 4코어, 16 GB (`analysis/d1_pair/jobs/py.slurm`, `--export=ALL,D1PAIR_CODE=$WT/analysis/d1_pair,PYENV=…`) | A 몇 분(같은 정규 core SMILES는 한 번만 계산해도 됨, 로컬 72개), B 1시간 안(P-4 캐시 재사용) |
| S, D, E | CPU | 몇 분 |
| K | RTX 3090 × 5 | fold당 약 1–1.5 GPU-시간 (GNN3R: 0.99–1.43) |
| W | RTX 3090 × 1 | 약 1 GPU-시간 (GNN3R: 0.92) |
