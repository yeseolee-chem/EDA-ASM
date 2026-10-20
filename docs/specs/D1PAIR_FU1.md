# D1PAIR FU1 — T2 중단 · learning curve · XGB 탐색 범위 확장 (Claude Code 실행 지시서)

**대상:** 브랜치 `d1-pair` `5df20b6` (`analysis/d1_pair/`). 새 브랜치를 만들지 않고 이어서 commit한다.
**새 폴더:** 코드 `analysis/d1_pair/` (새 파일 `d1_followup.py`) · 결과 `analysis/d1_pair/results/followup/` · 캐시 `$D1PAIR_SCRATCH/M/units_fu/`
**이 파일:** `docs/specs/D1PAIR_FU1.md`로 commit한다. 지시서 rev 9(`D1PAIR_REV9.md`)의 규칙은 여기서 바꾸는 것 말고는 그대로다.
**보고:** Phase마다 보고한다. STOP이 나오면 다음 Phase로 가지 않고 사용자에게 묻는다. 버그를 찾으면 고치기 전에 보고한다.
**순서:** K(T2 중단, 지금 바로) → A(사전 등록 추가본 commit) → L(learning curve) → X(XGB 넓은 grid) → R(보고)

---

## 0. 사용자 결정 (2026-10-20)

| # | 결정 | 이유 (잠정 T1 결과, `results/interim_T1_noSVR/`) |
|---|---|---|
| F1 | **T2를 즉시 중단한다.** 이미 끝난 T2 unit은 지우지 않고 보관하되 분석하지 않는다 | T1(다른 계열의 V를 학습에 포함)에서도 숨은 V 10건 중 C 1건, B 3건만 V로 판정 → V를 뺀 T2는 정보가 없다. T2는 arm B만 학습한다 |
| F2 | **learning curve를 돌린다** (arm C) | 데이터를 늘리면 좋아지는지, feature가 한계인지 판단 |
| F3 | **XGB 탐색 범위를 넓힌다** (arm C) | XGB unit 100개 모두 n_estimators 500(최대), 97개가 learning_rate 0.1(최대), 42개가 max_depth 7(최대)을 고름 |

**기본값으로 넣은 것** (바꾸려면 Phase A commit 전에 사용자에게 묻는다): LODO·LOSO에서 "C가 B보다 낫다"를 확증 비교로 사전 등록한다(§A-2). learning curve와 넓은 XGB는 **탐색**으로 표시한다.

---

## Phase K — T2 즉시 중단

### K-1. 새 T2 unit이 시작되지 않게 한다 (코드, 로그인 노드에서 편집만)
`d1_train.py`에 다음만 더한다. **fingerprint에 들어가는 것(CODE_VERSION, grid_cells, versions, unit_list)은 바꾸지 않는다.**
```python
SKIP_FILE = M_DIR / "SKIP_GROUPS"          # one group name per line; user decision 2026-10-20: T2

def skipped():
    return set(SKIP_FILE.read_text().split()) if SKIP_FILE.is_file() else set()
```
- `cmd_run` 맨 앞(`gate_prereg` 전): `if group in skipped(): print(f"[M {group}] skipped (SKIP_GROUPS, user 2026-10-20)"); return 0`
- `_worker` 맨 앞: `if u["group"] in skipped(): return uname(u), "skipped", 0.0` (이미 대기열에 들어간 unit도 시작하지 않게)
- 그다음 `$D1PAIR_SCRATCH/M/SKIP_GROUPS`에 `T2` 한 줄을 원자적으로 쓴다.
- 실행 중인 job의 다음 명령(`:::` 뒤)과 spawn된 하위 프로세스는 디스크의 새 코드를 읽으므로, 이 변경은 추가만 하고 기존 동작을 바꾸지 않아야 한다.

### K-2. 지금 T2를 계산 중인 task를 멈춘다
1. `squeue -u $USER`와 각 task 로그(`$D1PAIR_SCRATCH/logs/<이름>_<jobid>_<task>.out`)의 마지막 `=== d1_train.py run <G>` 줄로, **지금 T2를 돌리는 task**를 찾는다(마지막 `run T2` 뒤에 `=== exit`가 없는 task). 표로 보고한다.
2. 그 task만 `scancel <jobid>_<task>`. 진행 중이던 T2 unit만 사라진다(끝난 unit은 캐시에 남는다).
   - **SVR helper(`d1m_svrH`, 역순 `T2 ::: LOSO ::: LODO ::: T1`)**는 T2부터 시작하므로, T2 중이면 취소하고 같은 slice로 T2만 뺀 명령을 다시 제출한다:
     `sbatch --array=<같은 원소> ... -J d1m_svrH2 --export=...,D1PAIR_MODELS=SVR,D1PAIR_ORDER=reverse jobs/py.slurm d1_train.py run LOSO @I 4 ::: d1_train.py run LODO @I 4 ::: d1_train.py run T1 @I 4`
     (`--cpus-per-task`, `--mem`, 파티션, export는 원래 helper 제출과 같게. 원래 명령은 `jobs/submit_M.sh` 끝 주석에 있다.)
   - 아직 T2에 이르지 않은 task(NN·grid·SVR 본 배열은 T2가 마지막)는 그대로 둔다. K-1 덕분에 T2에 이르면 바로 건너뛴다.
3. **collect·Phase E 사슬을 다시 건다.** 기존 `d1m_collect`, `d1e_all`은 의존 job이 취소되면 `--kill-on-invalid-dep`로 함께 취소되므로, 남아 있으면 `scancel`하고 새로 제출한다:
   `col=$(sbatch --parsable --dependency=afterany:<남은 모든 M job id들> ... jobs/py.slurm d1_train.py collect)` → `sbatch --dependency=afterok:$col ... jobs/py.slurm d1_eval.py all ::: d1_figures.py`
   (`afterany`: collect가 스스로 빠진 unit을 확인하고 exit 2로 사슬을 멈춘다.) **단, 이 사슬은 Phase A commit 뒤에 제출한다.**

### K-3. collect·평가·그림에서 T2를 뺀다
- `cmd_collect`: `skipped()`에 있는 group은 unit 목록과 누락 검사에서 빼고, `M_report.json`에 `skipped_groups: ["T2"]`, `skipped_reason`, `n_skipped_units_on_disk`(이미 끝난 T2 unit 수)를 적는다. 끝난 T2 unit은 `M_predictions.parquet`에 넣지 않는다.
- `d1_eval.py`: `input_gate`의 기대 unit 목록에서 skipped group을 뺀다. `secondary_tests`의 `"T2: B vs A"`, `cmd_all`의 T2 대 T1 표, `bends(..., splits)`는 T2가 skipped면 건너뛰고 `E.json`에 `t2: "stopped by user decision 2026-10-20 (FU1 F1)"`을 적는다.
- `d1_figures.py`: T2가 skipped면 `fig7`을 그리지 않고 `figures/fig7_vblind_NOT_RUN.txt`에 사유를 쓴다.

### K-4. 검사 (모두 sbatch에서)
- **gate K-a (캐시 지문):** 바뀐 `d1_train.py`로 이미 끝난 unit 30개(T1·LODO·LOSO·PB, 모든 모델 포함, seed 20261019로 고름)의 fingerprint를 다시 계산해 저장된 json과 **모두 같아야 한다.** 다르면 코드 변경을 되돌리고 **STOP**.
- **gate K-b (unit 목록):** `d1_train.py list <G>`가 모든 group에서 변경 전 commit(`5df20b6`)과 같다.
- **gate K-c (중단 확인):** K-2 이후 1시간 안에 새로 생긴 T2 unit json이 없다(수정 시각으로 확인). 있으면 원인 보고.
- 보고: `results/followup/K_inventory.csv` — group × arm × model별 끝난 unit 수(T2 포함), 취소한 task, 다시 제출한 job id.
- `results/DEVIATIONS.md`에 적는다: **D-1** T2 중단(사용자 결정 2026-10-20, 이유 §0 F1), 끝난 T2 unit 수, 분석하지 않음.

---

## Phase A — 사전 등록 추가본 (L·X 학습 전, LODO·LOSO 평가 전)

`results/PREREG_D1PAIR_ADD1.md` + `prereg_d1pair_add1.json`을 쓰고 **commit·push한다.** `d1_followup.py`는 이 commit이 있고 내용 sha256이 같을 때만 실행한다(`d1_split.gate_prereg`와 같은 방식).

### A-1. 담을 것
- §0의 결정 F1–F3, D-1.
- **LODO·LOSO 결과(모델 점수, 예측, 판정)를 이 commit 전에 계산하거나 열지 않았다는 기록.** `results/`에 LODO·LOSO 평가 산출물이 없음을 확인해 적는다. 이미 있다면 commit하지 말고 **STOP**.

### A-2. 확증 비교 추가 (기본값, 사용자 확인 항목)
- T1의 "C < B"는 결과를 본 뒤 나온 것이라 확증이 아니다. **아직 보지 않은 LODO와 LOSO에서** 확인한다.
- 비교: split(LODO, LOSO) × 9 타깃(8채널 + barrier)마다 **arm C 대표 모델 대 arm B 대표 모델**. 대표 모델 규칙은 PREREG 그대로(6개 모델 중 안쪽 CV MAE 최저, SVR 포함).
- 지표: P_main 시험 짝 MAE 차이(B − C). 95 % CI와 양측 p: 그 split의 시험 짝에서 **core 단위 cluster bootstrap** 2,000회(seed 20261019). 18개(9 × 2 split)에 **Holm 보정**. 가설: B − C > 0.
- 이 비교는 SVR까지 모두 끝난 뒤 Phase E와 함께 계산한다.

### A-3. 탐색 분석의 설계·판독 규칙 (L, X)
아래 Phase L·X 설계와 판독 규칙을 그대로 적는다. 결과는 **탐색**이며 PREREG의 대표 모델·arm·단계 판정을 바꾸지 않는다.

---

## Phase L — learning curve (탐색)

### L-1. 설계
- arm **C**, 모델 **KRR, XGB**(PREREG grid 그대로), 타깃 10개, **T1 바깥 fold 0–4**.
- 학습 비율 f ∈ {0.25, 0.5, 0.75}, 반복 r ∈ {0, 1, 2}. **f = 1.0은 이미 끝난 T1 unit**(`T1__C__KRR__*`, `T1__C__XGB__*`)을 그대로 쓴다(다시 계산하지 않음, 캐시 fingerprint로 확인).
- **core 단위로 줄인다:** fold k의 학습 쪽 core(T_main 학습 짝에 있는 core, 기준 짝이 없는 core 4개 포함)를 이름순 정렬 → `numpy.random.default_rng(20261019 + 100*k + r).permutation` → 앞에서 ceil(f × core 수)개. 같은 (k, r)에서 0.25 ⊂ 0.5 ⊂ 0.75가 되게 같은 순서를 쓴다. 고른 core의 학습 짝은 모두 쓴다.
- 시험 짝은 T1 unit과 **같다**(fold k의 P_sens ∪ A_main). 학습 core가 4개 미만이면 **STOP**(GroupKFold(4) 불가).
- 튜닝·스케일·반대칭·gate M-1은 `d1_train`과 같다(`masks`로 T1 학습·시험 짝을 얻고 학습 짝만 줄인 뒤 `tune_grid`, `fit_final`을 그대로 부른다).
- unit 수: 2 모델 × 10 타깃 × 5 fold × 3 비율 × 3 반복 = **900**.

### L-2. 실행
- `d1_followup.py run LC <i> <n>` / `list LC` / `collect LC`. 캐시 `$D1PAIR_SCRATCH/M/units_fu/LC/C__<model>__<target>__f<k>__frac<f>__rep<r>.{json,parquet}`. fingerprint = d1_train과 같은 항목 + 고른 core 목록 sha256 + `FU_CODE_VERSION = "d1pair-fu-1"`.
- `sbatch --array=0-3 -p cpu1,cpu2 -J d1fu_lc --cpus-per-task=48 --mem=160G --export=ALL,PYENV=espley,D1PAIR_WORKERS=1,D1PAIR_NJOBS=48 jobs/py.slurm d1_followup.py run LC @I 4` (KRR·XGB unit은 지금 2–3분이므로 task당 약 10시간 안쪽 예상).

### L-3. 판독 (`results/followup/LC_*.csv`, `LC_curve.png`)
- 모델 × 타깃 × 비율마다: 학습 짝 수, P_main 시험 MAE(전체, para; 반복 3개 평균과 범위), 안쪽 CV MAE.
- **판정 규칙 (Phase A에 고정):** f = 0.5 → 1.0의 para MAE 감소율(KRR·XGB 각각)이
  - ≥ 10 %: **"데이터를 더 늘리면 아직 좋아진다"**
  - < 3 %: **"포화 — feature가 한계"**
  - 그 사이: **"불확실"**
- 참고 외삽(감소율 ≥ 3 %일 때만): f = 0.5, 0.75, 1.0의 평균 para MAE에 MAE = a·n^(−b)를 log-log 최소제곱으로 맞추고, 채널별 목표 MAE에 닿는 학습 짝 수를 적는다. 목표는 아래 표(사후 정의, 2026-10-20 점검 `claude/review-d1pair-interim-5df20b6.md`): 꺾임 크기 중앙값 × 0.28. **외삽은 참고용**이라고 그림 설명에 적는다.

| barrier | d1 | d2 | elst | pauli | oi | cpcm | cds | disp |
|---|---|---|---|---|---|---|---|---|
| 0.16 | 0.19 | 0.04 | 0.21 | 0.41 | 0.29 | 0.22 | 0.01 | 0.05 |

---

## Phase X — XGB 넓은 grid (탐색)

### X-1. 설계
- arm **C**, 모델 이름 **`XGBW`**(기존 `XGB` 캐시·fingerprint와 섞이지 않게), 타깃 10개, **T1 바깥 fold 0–4** → **50 unit**.
- grid(36칸): `n_estimators` {500, 1000, 2000, 4000} × `learning_rate` {0.05, 0.1, 0.2} × `max_depth` {5, 7, 9}. 나머지 설정은 rev 6 `train_ml_single.GRIDS["XGB"][0]` 그대로(random_state 42, n_jobs 1, tree_method hist).
- 학습·예측은 `d1_train._xgb_fit`, `_xgb_g`를 그대로 쓴다(양방향 학습, ½[h(Δx) − h(−Δx)]). 안쪽 GroupKFold(4), 최저 평균 안쪽 MAE, 같으면 grid 순서(키 이름순 곱, `d1_train.grid_cells`와 같은 구성).
- gate M-1 그대로.

### X-2. 실행
- `d1_followup.py run XW <i> <n>`. 캐시 `$D1PAIR_SCRATCH/M/units_fu/XW/`.
- `sbatch --array=0-3 -p cpu1,cpu2 -J d1fu_xw --cpus-per-task=48 --mem=160G --export=ALL,PYENV=espley,D1PAIR_WORKERS=1,D1PAIR_NJOBS=48 jobs/py.slurm d1_followup.py run XW @I 4` (칸 수와 나무 수가 늘어 unit당 수십 분 예상. 48시간에 걸리면 같은 명령으로 다시 제출, 캐시가 끝난 unit을 건너뛴다).
- 제출 순서: L과 X를 함께 제출해도 된다. 둘 다 SVR 학습과 노드를 나눠 쓰므로 `squeue` 개수 규칙(MaxJobs 10, MaxSubmit 20)을 지킨다.

### X-3. 판독 (`results/followup/XW_*.csv`)
- 같은 시험 짝에서 기존 `C/XGB` 대 `C/XGBW`: P_main MAE(전체, para), 안쪽 CV MAE, 차이(XGB − XGBW)의 core bootstrap 95 % CI(2,000회).
- Hammett 수준(T1 시험 para 계열, 부록 D `hammett_eval.calls`): R²_fam 중앙값, 채널별 꺾임 민감도·위양성률, 위 L-3 표의 목표 MAE 대비 배수.
- 고른 값 분포와 grid 끝 걸림. **50 % 넘는 unit이 여전히 끝(4000, 0.2, 9)에 걸리면 보고하고 STOP**(더 넓히기 전에 사용자에게 묻는다).

---

## Phase R — 보고 `results/followup/SUMMARY_FU1.md` (한국어)
- 맨 앞 세 문장: T2 중단 처리 결과, learning curve 판정, 넓은 XGB의 효과.
- 절: K(취소·재제출 표, gate K-a·b·c, D-1) → A(추가 사전 등록 commit hash) → L(표·그림·판정·외삽) → X(표·끝 걸림) → 이탈 → 한계.
- 반드시 적을 한계: L·X는 탐색이다. T1은 이미 결과를 본 분할이다. L의 외삽은 참고용이다. SVR이 끝나기 전 결과다.
- 수치는 모두 `results/followup/` 파일에서 옮긴다.

---

## 규칙
- **HPC:** 모든 python은 `sbatch`(`#SBATCH --time=48:00:00`). 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만. 로그인 노드에서 python·루프·`nohup`·폴러 금지. 제출 전 `squeue -u $USER -h | wc -l`.
- idempotent, 원자적 쓰기(`C.write_atomic`, `C.write_json_safe`). BLAS 스레드 1.
- **바꾸지 않는 것:** `results/PREREG_D1PAIR.md`·`prereg_d1pair.json`, 기존 `M/units/` 캐시, `CODE_VERSION`, 기존 grid(`d1pair_common`의 상수), f0. 새 캐시는 `M/units_fu/`에만 쓴다.
- LODO·LOSO 평가는 Phase A commit 뒤에만. L·X는 T1만 쓴다.
- σ, 계열·core 번호, X 이름은 입력에 넣지 않는다(PREREG 그대로).

## STOP

| Phase | 조건 |
|---|---|
| K | gate K-a(캐시 지문) 또는 K-b(unit 목록) 불일치; K-2 뒤 새 T2 unit이 생김 |
| A | Phase A 전에 LODO·LOSO 평가 산출물이 이미 있음 |
| L | 학습 core < 4; f = 1.0 캐시 unit이 없거나 fingerprint 불일치; gate M-1 실패 |
| X | gate M-1 실패; 50 % 넘는 unit이 넓힌 grid의 끝에 걸림 |
