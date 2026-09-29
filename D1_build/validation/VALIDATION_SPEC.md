# VALIDATION SPEC — D1 본실험 전 전수 검증 (ORCA 단독)

작성 2026-09-29 · 실행: Claude Code (UBAI) · 위치: `D1_build/validation/`

선행 문서:
- `D1_build/d1_autode_pilot/SPEC_D1_pilot.md`: 파일럿 SPEC. 이 문서에 없는 규칙은 모두 그 SPEC을 따른다.
- `D1_build/D1_PLAN.md` §11: 파일럿 결과.

---

## 0. 한눈에 보기 (Claude Code가 할 일, 순서대로)

1. §2 규칙을 읽는다. 이 문서의 판정 기준(§7)은 **사전 등록**이므로 실행 후 바꾸지 않는다.
2. **Phase F:** 파일럿 코드를 고친다(F1–F6). 단위 테스트를 통과시킨 뒤 S0를 다시 돌려 PASS를 확인한다.
3. **Phase V:** 실험 V1–V9의 task 목록을 만들고, worker pool 하나로 한 번에 돌린다(§6).
4. **중간 점검:** 우선순위 0–2(V1, V2a, V5)가 끝나면 중간 보고를 낸다. §8의 STOP 조건에 걸리면 queue를 멈춘다.
5. **최종 보고:** 모든 task가 끝나면 `VALIDATION_REPORT.md`를 만든다. A1–A10 판정과 §8 결정표를 채운다.
6. 커밋하고 사용자에게 보고한다. **D1 본실험은 시작하지 않는다.**

예상 규모(추정, 실측 아님):
- task 약 145개, 약 1,100 core-hour
- 동시 10 job × 8코어로 벽시계 약 15–25시간

---

## 1. 배경: 무엇을 검증하나

### 1.1 파일럿이 남긴 질문

| 파일럿 결과 | 남은 질문 |
|---|---|
| A3 = 0.003 kcal/mol | 구조가 같으면 라벨이 같다. **이미 확인됨** |
| A5: D0 반응을 SMILES부터 다시 돌리면 채널이 최대 약 10 kcal/mol 다름 (형성 결합 0.06–0.12 Å 차이) | 이 차이가 **엔진** 때문인가, **다른 TS를 찾은 것** 때문인가? |
| A5는 barrier로만, 2개로만 판정했음 | 채널 기준으로, 충분한 표본으로 다시 판정해야 함 |
| J07이 허수 gate에서 탈락 | 우리 gate가 D0를 만든 autodE 규칙보다 엄격했음 (F1) |
| J10의 ΔG‡ −4.22 | 서로 다른 기준 구조를 비교한 계산 착오 (F2) |

### 1.2 교수님 명제를 검증 가능한 가설로 바꾸기

교수님 명제는 **"입력(inp)이 같으면 결과는 완전히 같아야 한다"**이다.

- **같은 프로그램에 같은 입력을 주는 경우**라면 이 명제는 옳아야 하고, V1에서 직접 검증한다.
- **ORCA와 G16 사이**에서는 "같은 입력"이 성립하지 않는다. 키워드 이름이 같아도 실제로 푸는 식과 수치 근사가 다르다.

| 항목 | Coley D0 (G16) | 파일럿 (ORCA 6.1.1, autodE가 작성) | 근거 |
|---|---|---|---|
| "B3LYP"의 국소 상관 | VWN-III | **VWN-V** (ORCA 기본). G16과 같은 것은 `B3LYP/G` | ORCA 6.1.1 manual §3.3; Hertwig & Koch 1997 |
| 2전자 적분 | 정확 적분 (기본) | **RIJCOSX + def2/J** 근사 | ORCA manual §2.8; Neese et al. 2009 |
| 수치 적분 격자 | G16 기본 | ORCA DefGrid2 (기본) | 각 프로그램 설명서 |
| SMD | 같은 모델, G16식 cavity 이산화 | 같은 모델, ORCA의 Gaussian-charge 이산화 | Marenich et al. 2009; Garcia-Ratés & Neese 2020 |
| 최적화기와 수렴 기준 | G16 기본 | ORCA NormalOpt (TolMaxG 3e-4, TolMaxD 4e-3 a.u.) | ORCA manual §4.1 Table 4.1 |
| **TS 탐색의 출발 구조** | autodE 1.2 무작위 conformer | autodE 1.4.5 무작위 conformer. **seed가 고정되지 않음** | autodE 1.4.5 소스: `molecule.py` ETKDG `randomSeed` 미지정, `conf_gen.py` `np.random.RandomState()`; Coley 2023 Fig. 10 |

EDA 단일점 계산(라벨)은 D0와 D1 모두 **같은 ORCA 입력**을 쓴다. A3가 보여 주듯 구조가 같으면 라벨이 같다. 따라서 차이는 **TS 구조가 어디서 왔느냐**에서만 생긴다.

**가설**

| id | 가설 | 검증 실험 | 예측 |
|---|---|---|---|
| H0 | 같은 ORCA 입력 → 같은 결과 (교수님 명제) | V1 | 차이 ≤ 1e-6 Eh |
| H1 | A5 차이는 엔진(위 표의 구현 차이)에서 온다 | V2 (Coley TS에서 출발해 ORCA로 재최적화) | 참이면 TS가 크게 이동 |
| H2 | A5 차이는 autodE 무작위 탐색이 다른 TS를 찾은 데서 온다 | V3 (같은 반응을 ORCA로 반복 실행) | 참이면 반복 간 차이가 크고 V2 이동은 작음 |

사전 기대는 H2 쪽이다. 구현 차이(VWN, RI, 격자)는 보통 작기 때문이다. 다만 **이것은 추정이므로 실험으로 결정한다.**

---

## 2. 규칙 (파일럿 SPEC §2 전부 + 아래)

- **사전 등록.** §7의 임계값과 공식은 `validation/config_val.yaml`에 **실행 전에** 고정하고, 그 커밋 해시를 보고서에 적는다. 결과를 본 뒤 기준을 바꾸고 싶으면 원래 기준의 판정을 그대로 두고 대안 판정을 따로 병기한다.
- **본실험 금지.** D1 2,846행 제출은 사용자 승인 전까지 하지 않는다.
- **method 변경 금지.** V2의 L1/L2 수준은 진단 전용이다. 본실험 수준(L0)을 바꾸는 결정은 사용자가 한다.
- **읽기 전용 대상:** `labels_all.json`, `label_true/**`, D0 raw data, 파일럿 scratch(`/gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot`). 파일럿 산출물은 **복사해서** 쓴다.
- **scratch:** `/gpfs/tmp_cpu2/yeseo1ee/d1_validation` (링크 `D1_build/validation/scratch`)
- **HPC:** 모든 계산은 sbatch, `--time=48:00:00`. 로그인 노드에서는 one-shot 명령만 쓴다. MaxSubmit이 20이므로 worker pool(10) + 분석 job(1)만 제출한다.
- **autodE 프로토콜 그대로 (사용자 결정 2026-09-29).** D0·D1 라벨의 정의는 "반응당 autodE **1회** 실행이 찾은 최저 에너지 TS (+ Coley 기준 구조 후처리)"다. autodE 소스(v1.2.3, v1.4.5)의 선택 규칙과 같다. 따라서:
  - V3/V6b의 반복 실행은 **측정 전용**이다. 여러 번 돌려 가장 낮은 값을 고르는 식(best-of-N)으로 라벨을 바꾸지 않는다. 그렇게 하면 1회 실행인 D0보다 D1이 체계적으로 낮은 쪽으로 치우친다.
  - autodE seed를 고정하는 패치를 하지 않는다. autodE에는 seed 설정이 없다. 재현성은 **저장된 구조**(ts.xyz, ref_*.xyz)로 보장한다. Coley도 구조 파일로 데이터를 정의했다.
  - gate는 autodE의 선택을 바꾸지 않고 **수용 여부만** 판정한다. 그 수용 규칙이 D0보다 엄격하지 않은지 V9(D0 전수)와 F1(autodE 허수 규칙)로 확인한다.
- **실패 처리:** task 실패는 자동 재시도하지 않는다. 과학적 실패(TS 없음, gate 탈락)는 `.fail_*`로 기록하고 "완료"로 센다. 인프라 실패(크래시, wall)만 재제출 대상이다.

---

## 3. Phase F — 코드 수정 (파일럿 번들 기준)

`D1_build/d1_autode_pilot/`을 **수정해서** 쓴다. 변경은 모두 config로 켜고 끌 수 있게 하고, 파일럿 결과를 재현하는 기본값도 함께 남긴다.

| id | 수정 | 내용 | 근거 |
|---|---|---|---|
| F1 | 허수 gate를 autodE 규칙으로 정렬 | `run_ts.analyse`의 `one_imag`(= 허수 1개)를 둘로 바꾼다. **hard:** `imag[0] ≤ min_imag_cm(−40)` 그리고 `all(v > extra_imag_tol_cm(−50) for v in imag[1:])`. **flag:** 허수가 2개 이상이면 `flag_small_extra_imag`. config `gates.extra_imag_tol_cm: -50` | autodE 1.2.0–1.2.3과 1.4.5 `transition_state.py::optimise`가 모두 "Had small imaginary modes"(추가 모드가 전부 > −50 cm⁻¹이면 그대로 수용). D0를 만든 규칙과 같음 |
| F2 | 보고서의 ΔE‡/ΔG‡ 기준 보정 | `alt_used == 1`이면 `autodE_dE_ref = autodE_dE − port.dEsp_alt_minus_orig`, `autodE_dG_ref = autodE_dG − port.dG_alt_minus_orig`. Coley `G_act`와는 `dG_ref`로 비교 | 파일럿 J10: 1.94 + 4.54 = 6.48 vs G_act 6.16 |
| F3 | RDKit 경고 억제 | `stereo_utils` import 시 `RDLogger.DisableLog('rdApp.warning')` | S0 로그 730 KB 대부분이 이 경고 |
| F4 | SP별 비용 기록 | 각 `.out`의 `TOTAL RUN TIME` × nprocs를 파싱해 `label.json`의 `sp_coreh = {eda: …, frag1_dist: …}`에 넣는다 | 계산별 비용 분해 |
| F5 | 구조 보존 | `report.py`가 job마다 `ts.xyz`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`, `product.xyz`를 `results/geometries/<job>/`로 복사 (커밋 대상, 작음) | 사후 RMSD 검토 |
| F6 | 새 job kind | `D0_reopt_L0`, `D0_reopt_L2` (§6.3): `run_ts.py`는 Coley TS를 출발점으로 ORCA OptTS+Freq를 실행하고, `make_reference.py`는 replay처럼 Coley 기준 구조를 쓴다. `D0_reopt_L0`만 추가 단계 `refopt`(§6.3)를 갖는다 | V2 |

**Phase F 완료 조건**
1. 파일럿 로컬 테스트(`tests/`)가 모두 통과한다.
2. F1 단위 테스트:
   - 가짜 허수 목록 `[-360.7, -14.9]` → 통과 + flag
   - `[-360.7, -60]` → 탈락
   - `[-30]` → 탈락
3. F2 단위 테스트: 파일럿 `labels_d1_pilot.json`에서 J10 `dG_ref − G_act`가 +0.32 ± 0.01로 나온다.
4. **S0 재실행 PASS** (새 scratch, `submit_pilot.sh`의 S0만 단독 제출).

---

## 4. Phase V — 실험 목록

| id | 질문 | 방법 | 표본 | 비용 추정 (core-h) | 판정 |
|---|---|---|---|---|---|
| **V1** | H0: 같은 입력이면 같은 결과인가 | (a) 파일럿 J08, J11의 SP 입력 5개를 새 폴더에서 그대로 재실행 → 7개 출력의 에너지 비교<br>(b) `eda.inp`만 `%pal nprocs 4`로 바꿔 재실행<br>(c) `D0_reopt_L0` 2개를 **같은 입력으로 두 번** 실행 → TS 구조 비교 | 2+2+2 | 약 25 | A6 |
| **V2a** | Coley TS에서 ORCA 기울기는 0에 가까운가 | Coley TS 구조에서 EnGrad를 L0/L1/L2로 계산 → max, RMS 기울기 | 24 × 3 | 약 40 | 진단 |
| **V2b** | H1: 엔진을 바꾸면 TS가 얼마나 움직이고 라벨이 얼마나 바뀌나 | Coley TS에서 출발해 ORCA OptTS+Freq(L0 24개, L2 8개) → 기준 구조는 Coley 것 고정 → SP 5개 → 라벨 | 24 + 8 | 약 350 | **A5-eng** |
| **V2c** | 기준 구조까지 ORCA로 최적화하면 d1/d2/barrier가 얼마나 바뀌나 | V2b L0 job 안에서 Coley 기준 구조 2개를 L0로 Opt → `frag*_rel` 재계산 | 24 | 약 25 | 진단 |
| **V3** | H2: 같은 반응을 다시 돌리면 TS/라벨이 얼마나 흩어지나 | `D0_control`을 반응당 3회 독립 실행 (seed 미고정이 곧 독립). job id `V3_<rxn>_r1..r3`, 각자 별도 scratch 폴더. r1이 V4를 겸함 | 8 × 3 | 약 260 | **A5-run** |
| **V4** | (원래 A5) SMILES부터 전체 체인 → D0 재현도 | `D0_control` 24개. V3의 1회차가 이 중 8개를 겸함 | 24 | 약 170 (추가분) | **A5-e2e** |
| **V5** | A3, A4를 규모 있게 | `D0_replay` 24개 (Coley 구조·기준) + Coley 규칙 포트 진단 | 24 | 약 75 | **A3, A4** |
| **V6** | 성공률·비용·D1 재현성 | (a) D1 새 추출 10행: `sample_pilot.py`와 같은 층화(패널당 2), seed 20260930, 파일럿 10행 제외<br>(b) D1 반복: J00, J04, J08, J09를 새 job id로 2회 더 (파일럿 1회 + 2회 = 3회)<br>(c) J07: 파일럿 job 폴더를 validation scratch로 **복사**해 autodE checkpoint를 살리고, `.fail_ts`를 지운 뒤 F1 규칙으로 재실행 | 10 + 8 + 1 | 약 210 | **A1**, A5-run(D1), A10 |
| **V7** | J03는 정말 배리어가 없나 | 형성 결합 2개를 1.6–3.4 Å, 10×10 격자로 2D relaxed scan (L0) → minimax 경로 | 1 (100점) | 약 10 | **A10** |
| **V8** | gate는 나쁜 입력을 잡아내나 | 음성 대조 N1–N8 (§6.7). 대부분 QM 없이 텍스트/구조 주입 | 8종 | 약 1 | **A7** |
| **V9** | gate는 D0의 정상 TS를 떨어뜨리지 않나 | D0 ok 5,260건 전부에 TS gate 적용 (QM 없음) | 5,260 | 약 1 | **A8** |

---

## 5. D0 검증 표본 선정 (V2, V3, V4, V5 공통)

```text
pool   = labels_all status=="ok", charge1==charge2==0, not flag_async, n_atoms <= 35
strata = S1: 추적 이면각 있음 & alt_used==1   (allyl형, Coley가 기준 구조를 교체)
         S2: 추적 이면각 있음 & alt_used==0
         S3: 추적 없음 & dipole 골격 비방향족   (propargyl/allenyl, 선형)
         S4: 추적 없음 & dipole 골격 방향족     (고리형 mesoionic)
         "추적 이면각" = evidence/cfg_validation_D0.csv 의 n_track > 0
         방향족 여부 = stereo_utils.dipole_backbone의 골격 원자 중 aromatic이 있는가
강제 포함 = rxn 20 (S1), rxn 105 (S4)
추출   = stratum당 6개 (강제 포함분 포함), seed 20260930  → 24개
V3/V2b-L2 부분집합 = stratum당 2개 (20, 105 포함), seed 20260930  → 8개
```

로컬 점검 결과 n_atoms ≤ 32 조건에서도 풀 크기는 S1 192, S2 102, S3 187, S4 81이라 충분하다. 선정 결과(`val_d0_set.csv`: rxn_id, stratum, subset8 여부)는 커밋한다.

**한계:** 표본이 35원자 이하로 제한된다(D0 중앙값은 44원자). 큰 분자에서 엔진 효과가 달라지는지는 이 검증 범위 밖이다. 보고서에 명시한다.

---

## 6. 구현

### 6.1 폴더

```
D1_build/validation/
  VALIDATION_SPEC.md          이 문서
  config_val.yaml             파일럿 config 복사 + scratch 변경 + gates.extra_imag_tol_cm + §7 사전 등록 블록
  make_tasks.py               → val_d0_set.csv, val_manifest.csv (체인형 job), tasks.csv (모든 task)
  worker.sh                   worker pool (§6.2)
  run_task.sh                 task_id → 해당 체인/스크립트 실행
  val_grad.py  val_scan_j03.py  val_determinism.py
  analysis/  validation_report.py  negative_controls.py  gate_audit_d0.py  rmsd.py  stats.py
  results/   VALIDATION_REPORT.md, *.csv, geometries/ (커밋)
  scratch -> /gpfs/tmp_cpu2/yeseo1ee/d1_validation
```

### 6.2 worker pool (20 제한 안에서 145 task 실행)

- **job array:** `worker.sh`를 `--array=0-9%10`로 제출한다. 각 worker는 8코어, 32 GB, 48 h다.
- **claim 방식:** 각 worker는 `tasks.csv`를 우선순위 순으로 훑으며 task를 **claim**한다.
  - claim은 `mkdir $Q/claims/<task_id>`의 원자성으로 한다.
  - heartbeat를 두고, owner job이 squeue에서 사라진 경우에만 claim을 인수한다.
  - **이 로직은 `label_true/scripts/smd_relabel_worker.sh`(v5)의 claim/heartbeat/takeover 구현을 그대로 가져온다.** 이 클러스터에서 검증된 코드다.
- **실행:** claim한 task를 `run_task.sh <task_id>`로 실행한다. 종료 코드가 0이거나 과학적 실패(`.fail_*` 존재)면 `$Q/done/<id>`, 인프라 실패면 `$Q/failed/<id>`에 기록한다.
- **정지 스위치:** 매 claim 전에 `$Q/STOP` 파일이 있으면 새 task를 잡지 않고 종료한다(§8 STOP용).
- **분석:** `validation_report.py`를 별도 sbatch(1코어)로 `--dependency=afterany:<worker array>`로 걸어 둔다.
- **wall에 걸린 경우:** `worker.sh`를 다시 제출하면 남은 task를 이어서 처리한다(멱등).

**우선순위** (작을수록 먼저):

| 우선순위 | task |
|---|---|
| 0 | V1 |
| 1 | V2a |
| 2 | V5 |
| 3 | V2b-L0 (V2c 포함), V1c |
| 4 | V7, V6c(J07) |
| 5 | V2b-L2 |
| 6 | V3 부분집합 8 × 3회 |
| 7 | V4 나머지 16 |
| 8 | V6a |
| 9 | V6b |

### 6.3 ORCA 입력 수준 (V2)

**L0 = 파일럿 TS 수준, 글자 그대로.**
- 파일럿 scratch의 autodE optts 입력(예: `jobs/J10/ts/J10/transition_states/*_optts_orca.inp`)에서 머리부(좌표 앞까지)를 복사한다.
- 공백과 `%pal`, `%maxcore` 값을 뺀 나머지가 아래와 한 글자라도 다르면 STOP하고 보고한다.

```
! OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J  CPCM(Water)
%geom
Calc_Hess true
Recalc_Hess 30
Trust -0.1
MaxIter 100
end
%cpcm
smd true
SMDsolvent "water"
end
%output
xyzfile=True
end
%scf
maxiter 250
end
%output
Print[P_Hirshfeld] = 1
end
% maxcore
3000
%pal nprocs 8
end
```
(autodE 1.4.5 + pilot config이 로컬에서 실제로 생성한 헤더. 공백, `%maxcore` 값, `%pal` 값은 비교에서 제외한다.)

**L1** = L0에서 `B3LYP` → `B3LYP/G` (G16과 같은 VWN-III). V2a에서만 쓴다.

**L2** = G16을 ORCA 안에서 최대한 흉내 낸 수준. `! OptTS Freq B3LYP/G NORI D3BJ def2-SVP DefGrid3 TightSCF CPCM(Water)` + L0와 같은 블록.
- `NORI`: RI를 모두 끈다(ORCA manual §2.8 Table 2.44). `NOCOSX`는 RIJONX로 바뀔 뿐이라 쓰지 않는다.
- SMD 이산화 차이는 ORCA 안에서 없앨 수 없다. 보고서에 명시한다.

**EnGrad (V2a)** = 각 수준에서 `OptTS Freq`를 `EnGrad`로 바꾸고 `%geom` 블록을 뺀다. `.engrad`를 파싱해 max|g|, RMS g (Eh/bohr)를 구한다.

**출력 확인 (수준마다 필수, 보고서에 기록)**
1. VWN 버전이 L0는 V, L1/L2는 III로 출력 문구에 나오는지. 실제 문자열을 찾아 적는다.
2. L2 출력에 RIJCOSX/COSX/RI-J가 활성화되어 있지 않은지.
3. D3(BJ) 파라미터(s6, a1, s8, a2) 출력이 세 수준에서 동일한지.
4. SMD CDS 항과 ε 78.3550이 있는지.

**재최적화 출력 파싱**
- 최종 구조: `<base>.xyz`
- 진동수: `<base>.hess`의 `$vibrational_frequencies`
- 허수 모드 벡터: `$normal_modes`의 6번 열
- **파서 검증:** 파일럿 J10의 autodE TS `.hess`를 파싱해 첫 허수가 −186.75 cm⁻¹(autodE가 기록한 값)로 나와야 한다.

이 결과를 `run_ts.analyse(got=…)`에 넣어 파일럿과 **같은 TS gate**(F1 적용)를 통과시킨다. 반응물 구조로는 Coley의 plain `r*.xyz`를 쓴다.

**V2c `refopt` 단계 (`D0_reopt_L0`만)**
1. `labels_all`의 `rel1_file`, `rel2_file`을 L0 Opt 수준(`! Opt B3LYP RIJCOSX D3BJ def2-SVP def2/J CPCM(Water)` + smd 블록)으로 최적화한다.
2. 최적화된 두 구조로 `frag{1,2}_rel.inp`를 **repo 빌더 텍스트 그대로** 다시 만들어 SP를 돌린다.
3. 결과를 `label_refopt.json`에 저장한다. 채널 6개는 변하지 않으므로 d1, d2, barrier만 비교한다.

### 6.4 TS 비교: 원자 대응 + 거울상 고려 RMSD

atom 순서가 다른 두 TS(Coley 대 autodE, 반복 실행끼리)를 비교할 때 쓴다.
- 대응은 무거운 원자 그래프의 동형사상으로 찾는다.
- 생성물 입체가 미지정이면 거울상 TS가 나올 수 있으므로 거울상도 함께 비교한다.

```python
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher
import fragmenter as fr   # repo analysis/espley_xtb_repro

def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))

def mapped_heavy_rmsd(a, b, cap=5000):
    """a, b = (syms, xyz ndarray). Min heavy-atom RMSD over graph isomorphisms and mirror images."""
    Ga = fr.heavy_graph(*a)[0]; Gb = fr.heavy_graph(*b)[0]
    best = np.inf
    gm = GraphMatcher(Ga, Gb, node_match=lambda x, y: x["lab"] == y["lab"])
    for k, m in enumerate(gm.isomorphisms_iter()):
        ia = list(m); ib = [m[i] for i in ia]
        for mirror in (1.0, -1.0):
            Q = b[1][ib] * np.array([mirror, 1.0, 1.0])
            best = min(best, kabsch_rmsd(a[1][ia], Q))
        if k + 1 >= cap:
            break
    return best
```

**단위 테스트:**
1. 같은 구조 → 0.
2. 원자 순서를 섞은 구조 → 0.
3. 거울상 → 0.
4. D0 rxn 20 TS와 파일럿 J12(replay) TS → 0.

**"같은 TS"의 정의:** 매핑 RMSD ≤ 0.10 Å이고, 형성 결합 두 길이 차이가 모두 ≤ 0.02 Å.

### 6.5 통계

```python
def pooled_sd(groups):                     # groups: 반응별 반복값 배열 리스트
    num = sum(((g - g.mean()) ** 2).sum() for g in groups if len(g) > 1)
    den = sum(len(g) - 1 for g in groups if len(g) > 1)
    return float(np.sqrt(num / den))

# 평균 편향의 95% CI: t 분포 (n-1 자유도).
# 비율의 CI: 반응 단위 bootstrap 10,000회, seed 20260930.
# 성공률의 CI: Wilson 95%.
```

### 6.6 V7 (J03 2D scan)

1. **구조 확보:** 파일럿 scratch `jobs/J03/ts/J03/output/`의 생성물 구조를 복사한다.
2. **형성 결합 원자 찾기:** `stereo_utils.template_match(product_xyz, mapped_product_smiles)`로 map 번호를 구조 인덱스로 옮긴다.
3. **scan 입력 (ORCA L0 Opt 수준):**

   ```
   %geom Scan
     B i j = 1.60, 3.40, 10
     B k l = 1.60, 3.40, 10
   end end
   ```

   100개 제약 최적화이고, 결과는 `<base>.relaxscanact.dat`에 나온다.
4. **판정:** 격자에서 minimax 경로(각 단계 최대 에너지를 최소화하는 경로, 4-이웃 Dijkstra)를 (1.6, 1.6)에서 (3.4, 3.4)까지 구한다.
   - 경로 최댓값이 반응물 쪽 끝점이고 경로 위에 내부 극대가 없으면 → **안장점 없음 확정**.
   - 내부 극대가 있으면 → 그 격자점에서 OptTS를 시도한다. 성공하면 "autodE 탐색 실패"로 재분류한다.

### 6.7 V8 음성 대조 (심어 놓은 결함을 gate가 잡는가)

모두 **복사본**에 결함을 넣고, 해당 gate 함수를 호출해 실패하는지 본다. 원본은 건드리지 않는다.

| id | 심는 결함 | 잡아야 할 gate |
|---|---|---|
| N1 | `eda.inp`의 `SMD(water)` → `CPCM(water)` (옛 버그 재현) | `check_method_lines`, audit `eda_inp_smd_count` |
| N2 | `eda.inp`의 `FRAG1_C 0` → `1` (전하 오류) | 전자수 홀짝 또는 audit charge/mult 불일치 |
| N3 | D0 rxn 3090 (foreign bond, 제외된 반응)을 replay 분석 | `no_foreign_bond` |
| N4 | TS 대신 생성물 구조 + 임의 모드 벡터 | `forming_in_range` 또는 `imag_mode_on_forming_bonds` |
| N5 | 출력 하나에서 `ORCA TERMINATED NORMALLY` 삭제 | `run_sp` 판정, uniformity |
| N6 | 출력 하나에서 `SMD CDS free energy correction energy` 줄 삭제 | uniformity `cds`, audit `no_smd_cds` |
| N7 | rxn 20 replay에서 기준 구조를 `_alt` 대신 plain으로 | 포트 진단 `use_alt` 불일치 → A4 판정에서 검출. d1이 약 4.6 변하는 것도 확인 |
| N8 | 입체 가능한 C=C dipolarophile을 가진 D0 반응 1개에서 기준 구조의 cis/trans를 뒤집음 (이면각 180° 회전 후 재최적화 없이) | `dipolarophile_check` |

- N8에 쓸 반응은 `dipolarophile_check`가 `checked=True`를 내는 D0 반응 중 seed로 고른다.
- **판정(A7):** 8/8을 잡아야 한다.

### 6.8 V9 D0 전수 gate 감사 (QM 없음)

`labels_all` ok 5,260건과 제외된 5건에 run_ts TS gate를 모두 적용한다.
- 적용 gate: 분할, 형성 결합 audit, foreign, 형성 결합 범위, regio flag, 모드 비중. 모드는 `TS_imag_mode.xyz`에서 얻는다.
- 허수 개수 gate는 D0 진동수 파일이 없으므로 **적용 불가**로 보고한다.

**기대:**
- ok 5,260건의 hard gate 탈락 0건 (탈락이 있으면 한 건씩 원인을 적는다).
- foreign bond로 제외된 3건(3090, 3766, 4252)은 `no_foreign_bond`로 탈락해야 한다.
- 로컬 사전 점검에서 모드 비중 최소 0.546, regio 역전 4건(flag만)을 확인했다.

---

## 7. 판정 기준 (사전 등록 — `config_val.yaml`에 고정)

**모델 오차 기준값 `MAE_ref`** (rev 3 ESPLEY73+KRR, `analysis/espley_xtb_repro/results/SUMMARY.md` 표에서 읽어 확인한다):

| target | barrier | d1 | d2 | elst | pauli | oi | cpcm | cds | disp |
|---|---|---|---|---|---|---|---|---|---|
| MAE_ref (kcal/mol) | 1.45 | 1.12 | 0.82 | 1.52 | 2.02 | 1.16 | 1.01 | 0.23 | 1.25* |

\* disp는 ESPLEY73에서 feature와 해석적으로 같아 MAE가 0.01이므로, 실제 예측인 ESPLEY46의 1.25를 쓴다(SUMMARY rev 3).

**기준을 MAE_ref에 묶는 이유:** 데이터 생성 과정이 라벨에 넣는 잡음이 모델 오차보다 충분히 작아야, D0와 D1을 합쳐도 학습된 관계가 흔들리지 않는다.

| id | 판정 | 계산 | PASS 조건 |
|---|---|---|---|
| **A1** | D1 성공률 | 파일럿 10행 + V6a 10행 = 20행, 라벨 ok 비율 | ≥ 16/20 **그리고** 실패가 모두 "화학적 원인(안장점 없음)"이거나 "gate"로 분류됨. 파이프라인 버그 0건. Wilson CI 보고 |
| **A2** | gate 전수 통과 | 모든 task에서 생성된 라벨 | 100% (gate 목록은 파일럿 SPEC §8 A2 + F1) |
| **A3** | 구조가 같으면 라벨이 같다 | V5 24개, 9 target의 \|라벨 − labels_all\| | 24/24 모두 max ≤ 0.1 kcal/mol |
| **A4** | 기준 구조 규칙 재현 | V5 포트 진단의 `use_alt` vs Coley `alt_used` | ≥ 22/24 일치. 불일치는 모두 경계 사례: \|dG − (−0.239)\| < 0.5 kcal/mol 또는 \|RMSD − 0.05\| < 0.03 Å |
| **A5-eng** | 엔진 효과 (H1) | V2b-L0 24개, target t별 Δ_t = 라벨 − labels_all (TS만 ORCA, 기준 구조는 Coley) | 모든 t에 대해: (i) \|mean Δ_t\| ≤ 0.2·MAE_ref,t **또는** mean의 95% CI가 0을 포함<br>(ii) RMS Δ_t ≤ 0.5·MAE_ref,t<br>그리고 TS 매핑 RMSD 중앙값 ≤ 0.05 Å |
| **A5-run** | 반복 실행 잡음 (H2) | V3 8 × 3회 (+ V6b D1 4 × 3회), target별 pooled SD_run,t, NF_t = SD_run,t / MAE_ref,t | **보고 전용 (PASS/FAIL 없음).** 단 NF_t > 1인 채널이 있으면 §8 결정 사항 |
| **A5-e2e** | SMILES부터 전체 체인으로 D0 재현 (원래 A5의 재정의) | V4 24개, Δ_t = control − labels_all | (i) 편향: A5-eng (i)과 같은 기준<br>(ii) 설명되지 않는 초과 분산이 없음: R_t = RMS(Δ_t)² / (RMS_eng,t² + 2·SD_run,t²)이고, bootstrap 95% CI 하한 > 1.0이면 FAIL<br>원래 기준(\|Δbarrier\| ≤ 1.0 비율)은 참고용으로 병기 |
| **A6** | 결정성 (H0) | V1 | (a) 같은 입력: 7개 출력 모두 \|ΔE\| ≤ 1e-6 Eh, 9 라벨 \|Δ\| ≤ 0.001 kcal/mol<br>(b) nprocs 1 vs 4: \|ΔE\| ≤ 1e-6 Eh<br>(c) 같은 OptTS 입력 두 번 (8코어 MPI): 좌표 최대 차 ≤ 1e-4 Å. (c)는 병렬 합산 순서 때문에 미세하게 어긋날 수 있다. 1e-4 초과 ~ 0.01 Å 이하면 FAIL이 아니라 크기만 보고하고, 0.01 Å를 넘으면 FAIL |
| **A7** | 음성 대조 | V8 | 8/8 검출 |
| **A8** | D0 전수 gate 감사 | V9 | ok 5,260건 hard gate 탈락 0 (또는 전부 원인 기재), foreign 3/3 검출 |
| **A9** | 비용 | 파일럿 + V6 D1 행의 core-hour | 평균·95% CI와 2,846행 외삽을 보고 (판정 없음) |
| **A10** | 개별 실패 해소 | V6c, V7 | J07이 F1 적용 후 ok. J03은 2D scan에서 안장점 없음 확정 (또는 재분류 후 처리) |

**TS 일치율 (A5 보조, 보고 전용)**
- V3 반복 쌍 중 "같은 TS"(§6.4)인 비율
- V4 control TS가 Coley TS와 같은 비율
- 반복 3회 중 가장 낮은 에너지 TS가 Coley TS와 같은 비율

**새 허수 규칙 영향 (보고 전용):** V2b와 V3/V4의 ORCA Hessian 전체에서 추가 허수 모드(−50 < ν < 0)가 있는 TS의 수와 크기 분포.

---

## 8. 결정표 (최종 보고서에 채울 것)

| 결과 | 해석 | 조치 (제안; 채택은 사용자) |
|---|---|---|
| A3, A6(a)(b), A7 중 하나라도 FAIL | 파이프라인 결함 또는 비결정성 | **STOP.** `$Q/STOP`을 만들고 보고. 본실험 불가 |
| A6 PASS, A5-eng PASS, A5-e2e PASS | H0 성립, 엔진 영향 작음 → A5 차이는 TS 탐색 편차 (H2) | **GO (ORCA L0).** 단 A5-run 결과는 따로 판단 (아래) |
| A5-eng FAIL (L0), L2 결과는 기준 충족 | 차이의 원인이 VWN 또는 RI 설정 | 선택지: (a) 본실험 TS 수준을 L2 설정으로 변경 (비용↑), (b) D0 TS를 L0로 재최적화해 균질화. 사용자 결정 |
| A5-eng FAIL (L0, L2 모두) | SMD 이산화처럼 ORCA 안에서 못 없애는 차이 | 선택지: (a) G16이 가능한 곳(예: KISTI)에서 D1 TS 탐색, (b) D0 전체를 ORCA로 재최적화 후 재라벨. 사용자 결정 |
| A5-e2e FAIL, A5-eng PASS | 엔진도 반복 잡음도 설명하지 못하는 차이 | autodE 1.2→1.4.5 차이 가능성. **STOP 후 보고.** 원인 조사 필요 |
| A5-run: 어떤 채널이든 NF > 1 | TS 한 점 라벨이 "어느 TS가 잡혔나"에 크게 좌우됨 (D0에도 해당) | 본실험 차단은 아님. 다만 논문의 라벨 정의·잡음 하한 서술을 결정해야 함: TS conformer 여럿의 최저·평균, 또는 고정 반응 좌표 지점 라벨 (Bickelhaupt & Houk 2017). 사용자 결정 |
| A1 FAIL | 성공률 부족 | 실패 유형별 대책 후 재검증 |
| A2, A4, A8, A10 FAIL | 해당 gate나 규칙 문제 | 원인 보고, 수정안 제시, 해당 부분만 재검증 |

**중간 점검 (우선순위 0–2 완료 시):** A6, A3(V5), V2a 요약을 `interim_report.md`로 낸다. A6이나 A3가 FAIL이면 즉시 `$Q/STOP`.

---

## 9. 실행 순서 (명령)

```bash
# 0) Phase F: 코드 수정 → 로컬(컴퓨트 노드 sbatch) 단위 테스트 → S0 단독 재실행
cd D1_build/d1_autode_pilot
sbatch -p <idle> -o <scratch>/logs/s0_%j.log --export=ALL,PILOT_DIR=$PWD,D1P_CONFIG=../validation/config_val.yaml s0_preflight.sh

# 1) 사전 등록: config_val.yaml(임계값) + val_d0_set.csv를 커밋. 해시를 기록.
#    make_tasks.py도 python이므로 sbatch로 실행.
cd ../validation
git add config_val.yaml val_d0_set.csv tasks.csv val_manifest.csv && git commit -m "validation: pre-registered criteria and task list"

# 2) 제출 (one-shot): worker 10 + 분석 1 = 11 task
squeue -u $USER -h | wc -l            # ≤ 9 필요
W=$(sbatch --parsable -p <idle> --array=0-9%10 -o scratch/logs/w_%A_%a.log --export=ALL,VAL_DIR=$PWD worker.sh)
sbatch -p <idle> --dependency=afterany:$W -o scratch/logs/report_%j.log --export=ALL,VAL_DIR=$PWD run_report.sh

# 3) 모니터링: one-shot만
ls scratch/queue/done | wc -l; ls scratch/queue/failed; cat scratch/interim_report.md
```

- worker가 48 h wall에 걸리면 같은 명령으로 worker만 다시 제출한다. 분석 job도 다시 건다.

---

## 10. 보고서 형식 (`VALIDATION_REPORT.md` + 사용자 답변)

1. **사전 등록 해시**와 실행 job 번호.
2. **A1–A10 판정표.** 값, 95% CI, PASS/FAIL, FAIL 원인.
3. **가설 판정 요약 (H0/H1/H2).**
   - V1 차이값
   - V2a 기울기: 수준별 max/RMS와, L0→L1→L2에서 기울기가 줄어드는지
   - V2b: TS 이동량과 라벨 Δ 분포 (target별 mean, RMS, CI)
   - V3: SD_run, NF, TS 일치율
4. **교수님 명제에 대한 답:** 한 문단. 데이터로만 쓴다.
   - "같은 ORCA 입력 → 차이 x Eh"
   - "Coley 구조에서 ORCA 재최적화 → 이동 y Å, 채널 z kcal/mol"
   - "반복 실행 → SD w"
5. **§8 결정표**를 채운 것과 권고 (GO / 조건부 / STOP).
6. **비용과 본실험 일정 외삽.**
7. **한계:** 35원자 이하 표본, SMD 이산화 차이는 ORCA로 격리 불가, D0 진동수 부재.

---

## 11. 참고 문헌 (DOI 확인됨)

- Stuyver, Jorner & Coley, *Sci. Data* **10**, 66 (2023). doi:10.1038/s41597-023-01977-8. D0 프로토콜, 재현성(Fig. 10), 실패율 12.3%.
- Young, Silcock, Sterling & Duarte, *Angew. Chem. Int. Ed.* **60**, 4266–4274 (2021). doi:10.1002/anie.202011941. autodE. (권·쪽은 Coley 2023 참고문헌 18로 확인)
- Hertwig & Koch, *Chem. Phys. Lett.* (1997). doi:10.1016/S0009-2614(97)00207-8. B3LYP의 VWN 매개화 모호성.
- Neese, Wennmohs, Hansen & Becker, *Chem. Phys.* **356**, 98 (2009). doi:10.1016/j.chemphys.2008.10.036. RIJCOSX.
- Garcia-Ratés & Neese, *J. Comput. Chem.* (2020). doi:10.1002/jcc.26139. ORCA CPCM/SMD의 cavity와 Gaussian-charge 방식.
- Marenich, Cramer & Truhlar, *J. Phys. Chem. B* (2009). doi:10.1021/jp810292n. SMD.
- Fernández & Bickelhaupt, *Chem. Soc. Rev.* **43**, 4953 (2014). doi:10.1039/c4cs00055b. TS에서 dΔE_strain/dζ = −dΔE_int/dζ. TS 한 점 분석의 위험.
- Bickelhaupt & Houk, *Angew. Chem. Int. Ed.* **56**, 10070 (2017). doi:10.1002/anie.201701486. 반응 좌표를 핵심 기하 변수에 투영해 일관되게 비교할 것.

**문서 (동료 심사 아님, 사실 확인용)**
- ORCA 6.1.1 Manual: §3.3 (B3LYP vs B3LYP/G), §2.8 Table 2.44 (NORI/NOCOSX), §4.1 Table 4.1 (최적화 수렴 기준)
- autodE 소스: v1.2.0–1.2.3, v1.4.5 `transition_states/transition_state.py`, `species/molecule.py`, `conformers/conf_gen.py`
