# D1-REF·LADDER — 반응물 기준 구조 통일(재조립 + 확인 계산) → 구조 3단계 × 입력 2가지 × 모델 4개 — Claude Code 실행 지시서

**이 문서는 이전 `docs/specs/D1LADDER.md`(2026-10-22 첫 판)를 대체한다.** 첫 판이 이미 commit되었으면 이 파일로 덮어쓰고, commit 메시지에 "rev 2: Phase Q·V 추가, 라벨 교정 반영"을 적는다. 첫 판으로 시작한 작업이 있으면 멈추고 무엇을 했는지 보고한다.

**저장소:** D1-G2 Phase T 통과 commit이 올라간 `d1-g2` HEAD에서 새 브랜치 **`d1-ladder`**를 만든다. 이 HEAD에는 `analysis/d1_pair/`(FU2 포함, `2d745d23`과 같음), `analysis/d1_hf/`(`6b9fbeb2`), `analysis/d1_g2/`가 모두 있다.
**새 코드:** `analysis/d1_ladder/` · **결과:** `analysis/d1_ladder/results/` · **scratch:** `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/` (이하 `$LAD`)
**순서:** 0(시작 조건) → **Q(기준 구조 감사·재조립)** → **V(기준 구조 확인 계산)** → G0(DFT 구조 폴더) → F0(DFT 구조 위 xTB 입력) → P(입력표·짝) → C(코드·검사) → A(사전 등록 ADD3) → M1(dft·g1 학습) → E1(1단계 평가·보고) → [D1-G2 F·H 끝나면] M2(g2 학습) → E2(전체 평가) → R(보고)
**사용자 승인으로 멈추는 곳은 없다.** 16절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.

---

## 0. 무엇을, 왜

### 0-1. 기준 구조 문제 (Phase Q·V)

- **라벨의 출발점(기준 구조):**
  - d1·d2 = (TS 모양 그대로 떼어 낸 조각 에너지) − (반응물 기준 구조 에너지)
  - 장벽 = TS − 기준 구조
- **문제:** D1은 반응마다 반응물 형태(conformer)를 따로 찾았다.
  - dipolarophile 기준 구조는 autodE 반응물 그대로다(`make_reference.py`: Coley `_alt` 규칙은 dipole에만 적용).
  - autodE 형태 탐색은 seed가 고정되어 있지 않다.
  - 그래서 **같은 분자인데 반응마다 다른 형태가 기준으로** 잡혔다.
- **검토자 사전 점검** (`labels_d1_hf.parquet` geom=dft의 `e_frag*_rel_eh`, 즉 라벨 기준 좌표에서 계산한 HF 에너지):
  - 짝에 쓰인 7,410반응 중 523반응(7.1 %)이 같은 분자의 최저 기준보다 1 kcal/mol 넘게 높은 기준을 썼다. 편차 중앙값은 3.3 kcal/mol이다.
  - dipolarophile 313건 중 cyclooctyne(D 135, Dp 122)이 82 %다.
  - dipole 221건 중 일부는 Coley가 일부러 고른 입체 호환 형태일 수 있다.
- **원칙:** 반응 속도는 가장 낮은 반응물 형태를 기준으로 한 장벽이 정한다(Curtin–Hammett; Seeman, Chem. Rev. 1983, 83, 83, doi:10.1021/cr00054a001). Coley 절차도 가장 낮은 형태를 기준으로 쓴다(Stuyver, Jorner & Coley, Sci. Data 2023, 10, 66, doi:10.1038/s41597-023-01977-8).
- **바뀌는 것:** 같은 분자는 모든 반응에서 **같은 최저 기준**을 쓴다. 바뀌는 라벨은 **d1·d2·barrier뿐**이다. TS 에너지와 EDA 채널(elst, pauli, oi, disp, cpcm, cds)은 바뀌지 않는다.
- **원래 라벨은 지우지 않는다.** 새 라벨 파일을 따로 만들고 둘 다 보고한다.

### 0-2. LADDER (Phase G0 이후)

- D1-HF Phase A는 검량선 하나(정답 ΔΔ = β × HF ΔΔ)만 시험했다. 완성 모델(검량선 + 남은 차이를 모델이 학습)은 돌리지 않았으므로, 목표 도달 여부는 아직 모른다.
- 이 명령서는 구조 세 단계마다 모델 네 개를 같은 짝·분할·규칙으로 학습하고, 사전 등록한 단계 2 기준으로 **목표 도달 여부**를 판정한다. 정답은 교정한 라벨이다.

| 단계 (rung) | 구조 | 역할 |
|---|---|---|
| `dft` | 라벨 TS 그 자체 (B3LYP-D3BJ/def2-SVP) | 상한: 구조가 정답일 때 이 입력과 모델로 어디까지 가나. 배포 불가 |
| `g1` | DFT TS에서 출발해 xTB로 다시 최적화 | 구조를 xTB로 바꾸면 잃는 것 |
| `g2` | 반응 SMILES에서 xTB만으로 찾은 TS | 실제로 쓸 입력 (배포 성능) |

## 1. 시작 조건

| # | 조건 | 확인 |
|---|---|---|
| 0-1 | D1-G2 Phase T(T-1·T-2·T-3) 통과가 commit되어 있다. 기존 feature·토큰 코드의 **루트 인자**를 이 명령서도 쓴다 | `d1-g2` 로그와 `results/T_*` |
| 0-2 | `analysis/d1_pair/`가 `2d745d23`과 같다 | `git diff --stat 2d745d23 -- analysis/d1_pair` 비어 있음 |
| 0-3 | D1-HF 결과가 `6b9fbeb2`와 같다 | `labels_d1_hf.parquet` sha256 |
| 0-4 | `labels_d1.json` sha256 = `1eaa4171…`(9,496행) | `d1pair_common.load_d1_labels()` |

하나라도 아니면 STOP.

---

## 2. Phase Q — 기준 구조 감사와 재조립 (새 양자 계산 없음)

### Q-1. 입력
- 라벨: `labels_d1.json`(DFT). 기준 단일점 에너지 키(`e_frag1_rel_eh`, `e_frag2_rel_eh` 또는 같은 뜻의 키)와, d1·d2·barrier를 조립하는 기존 함수(라벨을 만든 코드, `label_true/scripts/_common.py` 계열)를 찾아 Q 보고에 적는다.
- 반응별 기준 구조 기록: D1 job 폴더의 `ref_result.json`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`(경로는 `d1pair_common`의 D1 run·job 경로 규칙을 따른다).
- 반응 SMILES: D1-G2 `results/X_targets.csv`.

### Q-2. 같은 분자 판정
- 분자 키 = RDKit 정규 isomeric SMILES(반응 SMILES에서 dipole·dipolarophile을 분리).
  - 거울상 이성질체는 같은 분자로 묶는다(물 SMD는 비키랄이라 에너지가 같다). 거울상 SMILES 둘 중 사전순으로 앞의 것을 키로 쓴다.
  - E/Z, cis/trans처럼 서로 다른 부분입체 이성질체는 다른 분자다.
- **입체 고정 기준은 묶지 않는다.** `ref_result.json`에서 Coley 규칙이 이면각을 고정해(constrained) 고른 dipole 기준은 키에 고정 이면각 상태를 더한다(같은 상태끼리만 묶음). 이 기준은 TS와 호환되도록 일부러 고른 것이다.
- 확인: 같은 키 안에서 기준 구조의 원소 구성·결합 그래프가 같다. 다르면 STOP.

### Q-3. 감사 (DFT 에너지)
- 분자 키마다 반응별 기준 에너지(라벨 수준)를 모으고, 최저와의 차이 dev를 구한다.
- 보고 `results/Q_audit.csv`(분자 키, 조각 종류, 반응 수, 최저, 최대, 범위, constrained 여부) + `results/Q_report.json`:
  - dev > 0.05·0.5·1·2 kcal/mol인 반응 수(골격·dipole별)
  - 영향받는 분자 수
  - 검토자 HF 사전 점검(0-1절)과의 일치(반응 단위 dev > 1 판정의 일치율)
  - Coley가 `_alt`를 판정할 때 쓴 G(E_sp + G_cont)가 기록에 있으면, G 기준 최저와 E 기준 최저가 다른 분자 수

### Q-4. 재조립 규칙
- 분자 키마다 **최저 에너지 기준 E_min**을 고른다. 고르는 에너지는 라벨에 실제로 쓰인 기준 단일점 에너지다.
- 반응의 원래 기준 E_orig가 **E_orig − E_min > 0.05 kcal/mol**이면 E_min과 그 좌표로 바꾼다. 0.05 이하는 그대로 둔다(SCF 재현 편차 수준).
- d1·d2·barrier는 **기존 조립 함수**에 바뀐 기준 에너지만 넣어 다시 계산한다. 다른 값은 건드리지 않는다.
- 출력: `$LAD/labels_d1_cr.json` + `results/labels_d1_cr.parquet`(열: rxn_id, 원래·새 d1·d2·barrier, 바뀐 조각, 바뀐 양, 기준을 가져온 rxn_id).

### Q-5. gate (하나라도 실패하면 STOP)
| # | 검사 | 기준 |
|---|---|---|
| Q-a | 원래 기준 에너지를 넣으면 원래 라벨이 나온다 | 전 행 d1·d2·barrier 1e-9 kcal/mol |
| Q-b | EDA 채널(elst, pauli, oi, disp, cpcm, cds), c_ghost | 전 행 원래 값과 같음 |
| Q-c | 바뀐 행의 Δd(해당 조각)와 Δbarrier가 기준 에너지 차이와 같음 | 1e-9 |
| Q-d | 항등식 잔차(barrier − 9채널 합)가 원래와 같음 | 1e-9 |
| Q-e | 새 기준 ≤ 원래 기준 (모든 행) | 예외 0 |

### Q-6. 짝 라벨
- `F_pairs.csv`의 dd_*를 새 라벨로 다시 계산한다 → `$LAD/F_pairs_cr.csv`. 짝 목록·fold(`S_folds.csv`)는 바꾸지 않는다.
- 보고: dd_d1·dd_d2·dd_barrier가 바뀐 짝 수(골격·dipole·위치별), 바뀐 양 분포. 예시로 azide/D para 계열의 원래 대 새 dd_barrier·dd_d2 표를 넣는다.

---

## 3. Phase V — 기준 구조 확인 계산 (리뷰어 대비: "찾은 것 중 최저 = 진짜 최저인가")

### V-1. 대상
- **(a)** Q-3에서 dev > 0.5 kcal/mol인 반응이 하나라도 있는 분자
- **(b)** cyclooctyne 계열 dipolarophile 전부(골격 D, Dp)
- **(c)** 대조군: (a)·(b)에 없는 분자에서 무작위 20개 (`numpy.random.default_rng(20261023)`, dipolarophile 10 · dipole 10)
- **제외:** constrained dipole 기준(Q-2). 보고에 수를 적는다.
- 목록 `results/V_targets.csv`. 예상 분자 수는 약 150–250개다(검토자 추정).

### V-2. 형태(conformer) 탐색 (xTB)
- 시작 구조:
  - ① RDKit ETKDGv3 300개 (`randomSeed = 20261023`)
  - ② 그 분자의 D1 기준 구조들에서 출발한 randomize-and-relax 300개 (autodE의 RR와 같은 방식; Coley 규칙 4단계와 같은 도구)
  - ETKDG가 실패하면 ②만 쓰고 기록한다.
- 최적화: `xtb --opt tight --alpb water --gfn 2`. 결합 그래프가 바뀐 구조는 버린다.
- 중복 제거: 무거운 원자 RMSD < 0.1 Å(거울상·대칭 고려) 또는 ΔE < 0.05 kcal/mol.
- 남길 것: xTB 에너지 하위 **5개** + 그 분자의 D1 기준 구조 형태들(이미 DFT 값이 있으므로 다시 계산하지 않음).

### V-3. DFT
- **최적화 + 진동수:** D1 autodE가 반응물 최적화에 쓴 것과 같은 수준·키워드(B3LYP-D3BJ/def2-SVP, SMD(water); D1 run config의 `opt`·`hess` 키워드를 그대로 가져온다), ORCA 6.1.1. 허수 진동수가 있으면 그 방향으로 한 번 밀어 다시 최적화하고, 그래도 있으면 버린다.
- **라벨 수준 단일점:** 그 분자 조각의 라벨 `frag*_rel` 입력 파일을 복사해 **좌표만** 바꾼다(D1-HF가 입력을 다룬 방식과 같음). 키워드·기저·용매·전하·다중도는 그대로다.
- **HF 단일점 (dft rung H 값용):** D1-HF와 같은 HF/6-31G\* 입력 규칙으로 같은 좌표에서 계산한다.
- 결합 그래프·입체(E/Z, 고리 융합)가 키와 다르면 버린다.

### V-4. 결정과 최종 라벨
- 분자마다 **최종 기준 = min(Q의 E_min, V에서 찾은 형태들)**. 라벨 수준 단일점 에너지로 고른다.
- V 형태가 Q의 E_min보다 0.05 kcal/mol 넘게 낮으면 기준을 바꾸고, Q-4·Q-5와 같은 규칙으로 다시 조립한다 → **`$LAD/labels_d1_cr2.json`, `results/labels_d1_cr2.parquet`, `$LAD/F_pairs_cr2.csv`**. 이것이 이후 모든 단계의 정답이다.
- 보고 `results/V_report.json`·`V_summary.md`:
  - 대상별로 더 낮은 형태를 찾은 분자 수와 낮아진 양 분포
  - 대조군에서 더 낮은 형태를 찾은 비율
  - cyclooctyne 부모 분자(D, Dp)의 형태별 에너지 표
  - 코어시간

### V-5. 시험 묶음과 비용 (자동 진행)
- 먼저 10개 분자를 돈다: cyclooctyne, 3,3-dimethylcyclooctyne, 그리고 (a)에서 무작위 8개(seed 20261023). 전체 예상 코어시간을 보고한다.
- 예상이 **5,000 코어시간 이하**이면 나머지를 자동으로 진행한다. 넘으면 STOP.
- 참고(검토자 추정): 형태 하나당 최적화 + 진동수 + 단일점은 약 1 코어시간이다. 근거는 D1 파일럿 단계별 평균으로, ref 단계 0.5, 단일점 5개(sp 단계) 2.8 코어시간/반응이다. 분자 약 200개 × 5형태 ≈ 1,000 코어시간이다.

---

## 4. LADDER 설계

### 4-1. 단계마다 입력은 모두 그 구조에서 계산한다

| | `dft` | `g1` | `g2` |
|---|---|---|---|
| TS 좌표 | 라벨 TS (D1-HF `geom=dft` EDA 입력 좌표) | `G1_ROOT_D1F/<rxn>/ts.xyz` | `$G2/g2_final/<rxn>/ts.xyz` |
| 반응물 좌표 (구조 feature용) | 반응별 라벨 기준 구조 (D1-HF dft `frag*_rel` 입력 좌표) | G1 `rel1/rel2.xyz` | G2 `rel1/rel2.xyz` |
| `ts.hess` (feature B6) | **새 계산:** 라벨 TS에서 ORCA `! XTB2 ALPB(water) Freq` (최적화 없음) | 기존 | D1-G2 |
| 원자 토큰 (FU2) | **새 계산 (F0)** | 기존 `$D1PAIR_SCRATCH/fu2/tokens/` | D1-G2 Phase F |
| 전역 182 feature | **새 계산 (F0)** `$LAD/X182_d1_dft.parquet` | 기존 `X182_d1.parquet` | D1-G2 `$G2/X182_d1_g2.parquet` |
| HF/6-31G\* 9값 (H) | `labels_d1_hf.parquet` `geom == dft` + V-3 HF 단일점 | 같은 파일 `geom == g1` | D1-G2 `labels_d1_hf_g2.parquet` |

- **구조 feature(토큰, 182)는 모든 rung에서 반응별 기준 구조를 그대로 쓴다**(기존 파이프라인과 같음).
- **H의 에너지 값 중 d1·d2·barrier는 rung 안에서 같은 기준 규칙으로 다시 조립한다:**
  - `dft`: Q·V가 고른 최종 기준 구조의 HF 에너지
  - `g1`, `g2`: 그 rung의 반응별 기준 구조들 중 분자 키(Q-2)마다 HF 에너지가 가장 낮은 것. 배포할 때는 반응물마다 기준을 한 번만 계산하므로 같은 분자에 같은 기준이 쓰인다.
  - 조립은 D1-HF 조립 함수(`build.py`)에 기준 에너지만 바꿔 넣는다. gate P-a에서 원래 기준을 넣으면 `labels_d1_hf.parquet`가 1e-9 안에서 재현되어야 한다.
  - H의 EDA 채널 값은 그대로다.
- 한 rung의 학습·평가는 **그 rung 열의 파일만** 읽는다(gate L-4). `g2`가 G1·DFT 파일을 하나라도 읽으면 DFT 정보가 새는 것이다.
- H = HF 값 9개: `d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier` (kcal/mol, 반응별).
- **정답 y:** `F_pairs_cr2.csv`(V-4). 원래 라벨은 비교용으로만 쓴다.

### 4-2. 입력 두 가지 (arm)

| arm | 원자 모델 입력 | XGB 입력 | 학습 대상 | 예측 |
|---|---|---|---|---|
| **S** (구조만) | 원자 토큰 + 전역 182 (FU2 변형 "A+G") | 182 (rev 9 arm C) | ΔΔ | 모델 출력 |
| **SH** (구조 + HF) | 원자 토큰 + 전역 (182 + H 9) | 182 + H 9 | 잔차 r = ΔΔ − β·ΔH | β·ΔH + 모델 출력 |

**SH의 기준선 (검량선):**
- 타깃 t마다 ΔH_t = H_t(a) − H_t(b).
- β_t = Σ ΔH_t·Δy_t ÷ Σ ΔH_t², **절편 없음**. 절편이 있으면 반대칭이 깨진다.
- β는 **그 학습에 쓰는 짝으로만** 맞춘다. 튜닝 때는 안쪽 학습 쪽, 최종 모델은 바깥 학습 쪽 전체(T_main: 기준 짝 + anchor 짝)다.
- c_ghost는 HF 대응 값이 없으므로 β = 0이다.

**SH의 학습 규칙 (FU2 규칙을 이렇게만 바꾼다):**
- 출력 = 잔차. 표준화는 학습 쪽 잔차 표준편차로 나누기만 한다(평균은 빼지 않는다).
- 손실 = 잔차 MAE(표준화 단위) + λ·|ŷ_barrier − Σ 9채널 ŷ| / s_barrier, λ = 0.1. 벌점은 **기준선을 더한 최종 예측 ŷ**로 계산하고, s_barrier는 원래 y의 학습 쪽 표준편차다.
- 튜닝·early stopping 지표 = 최종 예측 ŷ의 검증 MAE를 **원래 y의 학습 쪽 표준편차**로 나눈 값의 9 주 타깃 평균. FU2와 같은 단위다.
- XGB: 타깃마다 r_t를 학습한다. 안쪽 CV 점수 = |β·ΔH + ĝ − y|의 평균(kcal)이다.

### 4-3. 모델 네 개

| 모델 | 구조·grid·학습 규칙 |
|---|---|
| AW | FU2 M-2 그대로, 변형 A+G |
| GNN | FU2 M-2 그대로, 변형 A+G |
| TF | FU2 M-2 그대로, 변형 A+G |
| XGB | rev 9 arm C XGB 그대로 (`train_ml_single.GRIDS["XGB"]`, ½[h(dx) − h(−dx)], 안쪽 `GroupKFold(4)`) |

- 짝·분할: `d1_train.masks(u)` 또는 `fu2.train.Data.split`와 같은 T1 fold 0–4. 학습 = 학습 쪽 T_main, 시험 = 시험 쪽 P_sens ∪ A_main, 주 지표 = P_main.
- 원자 모델은 seed 5개(20261019 + 0..4) 평균, 최대 300 epoch, patience 30. FU2와 같다.
- 입력 금지(FU2와 같음): σ, 계열·core 번호, X 이름.

### 4-4. 학습 단위
- 정답이 바뀌었으므로(d1·d2·barrier) **모든 rung·arm을 새로 학습한다.** FU2·rev 9 결과는 재사용하지 않고, "원래 라벨" 비교로만 보고한다.

| rung × arm | 단계 | 단위 |
|---|---|---|
| dft·S, dft·SH, g1·S, g1·SH | 1단계 | 4 × 4 모델 × 5 fold = 80 |
| g2·S, g2·SH | 2단계 | 2 × 4 × 5 = 40 |

GPU 학습 횟수 = 6 rung-arm × 3 원자 모델 × 5 fold × 9회(튜닝 4 + seed 5) = **810회**(FU2의 3배). 1단계 540회, 2단계 270회.

### 4-5. 참고 행 (학습 없음)
- **null:** ΔΔ = 0.
- **LINE:** β·ΔH만 쓴다. β는 바깥 학습 쪽 T_main에서 구한다.

---

## 5. Phase G0 — DFT 구조 폴더

- 대상: D1-HF `geom=dft` ok 반응 9,266개.
- `$LAD/dft_final/<rxn_id>/`를 G1 폴더와 **같은 형식**으로 만든다.
  - `ts.xyz` ← D1-HF dft EDA 입력 좌표 (`hf_common.parse_inp`)
  - `rel1.xyz`, `rel2.xyz` ← 반응별 dft `frag1_rel`, `frag2_rel` 입력 좌표 (4-1: 구조 feature는 반응별 기준 그대로)
  - `work/ts.hess` ← 아래 Freq 계산
  - `result.json` ← feature 코드가 읽는 키를 G1 형식대로 채운다
- Freq: ORCA 6.1.1, `! XTB2 ALPB(water) Freq`, 좌표 = 라벨 TS(고정), 전하·다중도 = 라벨. G1 `ts.hess`와 같은 parse(`ext_features.parse_hess`)로 읽혀야 한다.
- **원자 순서 gate:** `ts.xyz` 원소 순서 = 라벨 TS, rel1 + rel2 원소 = TS 원소(조각 대응 `xtb_slice.ts_fragments(label, …)`이 오류 없이 돈다).
- 출력: `results/G0_status.csv`(rxn_id, status, 이유, n_imag, 최소 진동수, 코어시간), `results/G0_report.json`.
- 참고: xTB 입장에서 라벨 TS는 정류점이 아니므로 허수 진동수가 0개 또는 2개 이상일 수 있다. 실패로 치지 않고 기록한다. 이 때문에 feature B6가 계산되지 않는 반응이 1 %를 넘으면 STOP.

## 6. Phase F0 — DFT 구조 위 xTB 입력

D1-G2 Phase F와 **같은 경로**(T-1·T-2 루트 인자)를 루트 = `$LAD/dft_final`로 돌린다.
- 전역 182 → `$LAD/X182_d1_dft.parquet` (열 순서 = `X182_d1.parquet`, F-1 gate 그대로)
- FU2 원자 값·토큰 → `$LAD/fu2/` (버전 `fu2-atoms-3-dft`, `fu2-tokens-3-dft`). FU2 Q-a(30열 재현 ≤ 1e-6), Q-b(조각 전하 합), T-a(토큰 수 = 원자 수, NaN 없음)를 그대로 적용한다.
- 출력: `results/F0_report.json`(반응 수, 실패 이유, sha256), `results/F0_tokens_manifest.csv`.
- 실행: sbatch 배열, `XTB_THREADS=1`, 반응 단위 병렬. 시작 전에 G1 Q·F 로그로 예상 코어시간을 보고한다.
- Phase V와 G0·F0는 서로 독립이다. 자리가 있으면 같이 돌려도 된다.

## 7. Phase P — 입력표와 짝 집합

- 단계마다 반응 입력표 `$LAD/inputs/<rung>.parquet`(rxn_id, 토큰 경로·sha256, 182열, H 9열; H의 d1·d2·barrier는 4-1 규칙으로 다시 조립한 값)를 만든다.
- **gate P-a:** 원래 기준 에너지를 넣으면 D1-HF 값(`labels_d1_hf.parquet`)이 1e-9 안에서 재현된다(dft·g1).
- **사용 가능 반응:** 토큰 + 182 + H가 모두 있는 반응.
- **사용 가능 짝:** a·b가 둘 다 사용 가능한 짝.
- 짝 집합:
  - P_dft, P_g1 (1단계), P_g2 (2단계): 각 단계 자기 집합
  - P_01 = P_dft ∩ P_g1
  - P_012 = P_01 ∩ P_g2
- 확인: P_g1이 FU2가 실제로 쓴 짝 목록(`$D1PAIR_SCRATCH/fu2/splits/pairs.parquet`)과 같은지. 다르면 STOP.
- 보고 `results/P_report.json`: 집합별 짝 수(기준·anchor·para), fold별 수, 빠진 짝과 이유, 기준 구조 재조립으로 바뀐 H 값 수(rung별).

## 8. Phase C — 코드와 검사

**코드:** `analysis/d1_ladder/`(`lad_common.py`, `refs.py`(Q·V), `inputs.py`, `train_atoms.py`, `train_xgb.py`, `evaluate.py`, `tests/`)
- FU2·rev 9·D1-HF 코드는 import해서 쓴다. 바꿔야 하면 기본값이 기존 동작인 인자만 추가하고 L-6으로 확인한다.
- **라벨 파일 인자:** 학습·평가 코드는 짝 라벨 경로를 인자로 받는다(기본 = `F_pairs_cr2.csv`). L-6·L-10은 원래 라벨(`F_pairs.csv`)로 돈다.

**검사:** `tests/`를 sbatch로 돌린다(GPU 검사는 GPU 노드). 결과 `results/C_tests.json`. **하나라도 실패하면 STOP.**

| # | 검사 | 통과 기준 |
|---|---|---|
| L-1 | 반대칭, 기준선 포함 최종 예측: 4 모델 × 2 arm × 3 rung에서 g(b,a) = −g(a,b), g(a,a) = 0 | ≤ 1e-4 kcal/mol (float64 재계산) |
| L-2 | β 누출: β를 맞춘 짝 = 그 학습의 학습 짝 | 검증·시험 pair_id 0개 (assert) |
| L-3 | 기준선 덧셈: 합성 데이터에서 잔차 모델 출력을 0으로 두면 예측 = β·ΔH; y = 2·ΔH이면 β = 2 | 1e-10 |
| L-4 | 단계 분리: 학습 process가 연 입력 파일 경로를 기록 | 각 rung에서 4-1 그 열의 파일만. g2에서 G1·DFT 경로 0개 |
| L-5 | H 일치: 무작위 50반응의 H 9값 = 그 rung 입력표 값 | 1e-12 |
| L-6 | 회귀 (g1·S 설정 + **원래 라벨**): loader가 만드는 입력 텐서·y·s_y = FU2 unit | 텐서 sha256 같음, s_y 1e-12. XGB g1·S fold 0 예측 = rev 9 C-XGB unit, ≤ 1e-6 |
| L-7 | 항등식 벌점: 정답 y(기준선 + 잔차)를 넣으면 | ≤ 1e-3 |
| L-8 | 스케일·튜닝 누출 (FU2 C7·C8과 같음) | 시험 쪽 0개 |
| L-9 | 대표 고르기 검증 짝: 4 모델이 같은 검증 짝(바깥 학습 쪽의 첫 안쪽 fold)을 쓴다 | pair_id 목록 같음 |
| L-10 | LINE 함수 + **원래 라벨·원래 H**: D1-HF Phase A의 짝·fold를 넣음 | `analysis/d1_hf/results/A_summary.csv`의 dft·g1 para MAE와 1e-6 안에서 같음 |
| L-11 | 새 정답: loader의 y가 `F_pairs_cr2.csv`와 같고, elst·pauli·oi·disp·cpcm·cds·c_ghost는 `F_pairs.csv`와도 같음 | 1e-12 |
| smoke | fold 0, 구성마다 2 epoch, rung dft·g1 | gate M-1′ 통과 |

- 통과하면 코드, `C_tests.json`, 예상 GPU 시간(FU2 unit 기록으로 추정)을 **commit·push하고 보고한 뒤 Phase A로 간다.**

## 9. Phase A — 사전 등록 ADD3 (학습 전)

`analysis/d1_ladder/results/PREREG_D1PAIR_ADD3.md` + `prereg_d1pair_add3.json`을 쓰고 commit·push한다. 학습 코드는 이 commit과 sha256을 확인한 뒤에만 돈다(`d1_split.gate_add1` 방식).

**담을 것:**
- Q·V 규칙과 결과 요약(바뀐 반응·짝 수), 정답 파일 `F_pairs_cr2.csv` sha256
- 4절 전체(rung·arm·모델·grid·β 규칙·H 재조립 규칙)
- 짝 집합과 sha256
- 13절 판정 규칙
- 11절 지표

**이미 본 결과도 그대로 적는다:**
- D1-HF Phase A(LINE의 dft·g1 성적, 원래 라벨)
- FU2 g1·S 결과(원래 라벨)
- 검토자의 사후 탐색(HF@G1 + FU2 선형 결합 oi para 0.65, HF@DFT + FU2 0.26)
- 검토자의 기준 구조 HF 사전 점검(0-1절)

기존 PREREG·ADD1·ADD2는 바꾸지 않는다.

## 10. Phase M1 — 1단계 학습 (dft·S, dft·SH, g1·S, g1·SH)

- 단위 이름: `L__<rung>__<arm>__<model>__f<k>`
- 캐시: `$LAD/units/`. json의 입력 지문 = rung 입력표 sha256, 짝 라벨 sha256, 짝 목록 sha256, ADD3 sha256, grid, 코드 버전 `d1-ladder-2`, torch·CUDA·GPU, XGB 버전. 지문이 다른 캐시를 만나면 exit 3.
- 원자 모델은 env `env_d1fu2`(GPU), XGB는 rev 9 arm C와 같은 env(CPU)를 쓴다.
- 각 unit json에 **검증 짝 kcal MAE를 타깃별로** 저장한다(대표 고르기용, 13-1).
- 예측: `results/L_predictions.parquet` — pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, base(β·ΔH, S는 0), beta, yhat_seed_sd.
- gate M-1′: FU2와 같다. 기준선을 포함한 최종 예측으로 검사한다.

## 11. Phase E1 — 1단계 평가

평가 함수는 FU2 Phase E 코드(`fu2/evaluate.py`: `bends_tab`, `compare`)와 그것이 부르는 `d1_eval.boot_diff`·`clopper`·`family_eval`, `hammett_eval.calls`를 그대로 쓴다.
- 그 코드가 아직 `d1-pair`에 없으면 학습(M1)은 먼저 하고, E1은 FU2 Phase E가 commit된 뒤 merge해서 한다.

출력:
- `results/E1_metrics.csv`: rung × arm × 모델(AW, GNN, TF, XGB, REP, ENS4, LINE, null) × 타깃 × 집합(자기 집합, P_01) × 위치(all, p, m, o, direct) → n, MAE, RMSE, para MAE의 95 % CI, R²_fam 중앙값, ρ 오차.
  - para MAE의 CI는 core 단위 재추출 2,000회, seed 20261019이다. `boot_diff`에 비교 상대 오차 0을 넣어 MAE 자체의 CI를 낸다.
- `results/E1_bends.csv`, `E1_bend_rates.json`: 민감도·위양성률(Clopper–Pearson).
- `results/E1_goal.csv`: 13절 판정.
- `results/E1_compare.csv`: 13-4 비교.
- **라벨 교정 효과 (설명용):** g1·S의 새 라벨 결과 대 FU2·rev 9 원래 라벨 결과(d1·d2·barrier의 MAE, 계열 R²_fam, 꺾임 판정 수), 그리고 DFT 정답 자체의 꺾임 판정 수 변화(`hammett_eval.calls`, 원래 대 새 라벨). → `results/E1_label_effect.csv`.
- 그림 `figL1`–`figL3`의 1단계 판.
- 보고 `results/SUMMARY_LADDER_stage1.md`(한국어). 맨 앞 세 문장은 Q·V 결과(바뀐 반응 수), dft 판정, g1 판정이다. 그다음 D1-G2 F·H를 기다린다.

## 12. Phase M2·E2 — g2 (D1-G2 Phase F·H 통과 뒤)

- 시작 조건: D1-G2의 `X182_d1_g2.parquet`, `$G2/fu2/` 토큰, `labels_d1_hf_g2.parquet`이 각 gate를 통과하고 commit되어 있다. `origin/d1-g2`를 merge한다.
- P(g2 입력표, H 재조립, P_g2, P_012) → M(g2·S, g2·SH, 10절과 같은 규칙) → E2(11절 + P_012 비교) 순서로 한다.
- g2는 **G2가 라벨과 다른 TS를 찾은 짝도 그대로 학습·평가한다.** 배포 때는 구별할 수 없기 때문이다.
  - 설명용으로만 D1-G2 Phase C의 "같은 TS" 판정에 따라 짝 오차를 나눠 따로 보고한다.
- 적용 범위(coverage)를 보고한다: P_main 시험 짝 중 g2로 예측 가능한 비율. g2에서 실패한 반응은 배포 때도 예측할 수 없다.

## 13. 판정 규칙 (ADD3에 고정)

### 13-1. 대표 (REP)
- rung × arm × 타깃마다 4 모델 중 **검증 kcal MAE**(바깥 fold 5개 평균)가 가장 낮은 모델을 고른다.
- 동점이면 AW → GNN → TF → XGB 순서의 앞 것이다. 시험 점수로 고르지 않는다.
- ENS4 = 4 모델 시험 예측의 같은 비중 평균이다(조정값 없음). 보조로 쓴다.

### 13-2. 단계 2 판정 (주 판정, rung × arm 6개)
- REP 기준. **oi와 cpcm 모두** 다음을 만족하면 통과다.
  - para 짝 MAE ≤ 0.13 kcal/mol
  - 꺾임 민감도 ≥ 0.7
  - 위양성률 ≤ 0.1
- 집합은 그 rung의 자기 집합이고, 점 추정으로 판정한다. CI 상한까지 ≤ 0.13이면 "확실히 통과"로 따로 표시한다.
- 근거(rev 9 PREREG 5-3, 그대로 인용): 숨은 꺾임 크기 중앙값이 궤도 0.47, 용매 0.45 kcal/mol이다. 모의실험(σp 18점)에서 잡음 SD ≤ 0.35 × 꺾임 크기 → SD ≤ 약 0.16 → MAE ≈ 0.8 × SD ≤ 약 0.13.
- 참고: oi·cpcm 정답은 Q·V에서 바뀌지 않는다(Q-b).
- 보조 판정: ENS4에 같은 기준을 적용한다. 개별 모델의 통과는 **설명용**으로만 적는다(가장 좋은 것을 고르면 사후 선택이 된다).
- 단계 0·1(rev 9 기준)과 9개 타깃 전부의 수치는 설명용으로 함께 적는다.

### 13-3. 해석 규칙 (위에서부터 처음 맞는 것 하나)

| # | 조건 | 결론 문장 |
|---|---|---|
| a | g2·S 또는 g2·SH 통과 | 배포 가능한 입력(SMILES → xTB)으로 목표에 도달했다 |
| b | g2 둘 다 실패, dft·S 또는 dft·SH 통과 | 모델은 목표에 닿을 수 있으나 xTB 구조의 정밀도가 병목이다. 구조를 DFT 수준으로 다듬는 경로를 시험할 근거가 있다 |
| c | dft 둘 다 실패 | 이 입력과 이 모델 네 개로는 구조가 정답이어도 목표에 닿지 않는다 |

- 1단계(E1)에서는 dft 부분만 적용해 "dft 통과 / 실패"만 적는다.
- g1 결과는 이 표에 넣지 않는다. "xTB로 바꾸면 잃는 몫"의 설명용이다.

### 13-4. 설명용 비교 (보정 없음, `compare`, d = MAE(상대) − MAE(기준))
- 같은 rung·모델에서 SH 대 S (HF가 더하는 몫)
- 같은 arm·모델에서 rung 사이 (P_01, 뒤에 P_012)
- REP 대 LINE (모델이 검량선 위에 더하는 몫)

## 14. 그림과 보고 (Phase R)

### 14-1. 그림 (`results/figures/`)
| 그림 | 내용 |
|---|---|
| `figQ1_refs` | 분자별 기준 에너지 범위(원래), 골격·dipole별. cyclooctyne 부모 분자의 형태별 에너지(V 포함) |
| `figQ2_pairs` | azide/D·nitrone/D para 계열의 dd_barrier·dd_d2, 원래 대 새 라벨 (σp 축) |
| `figL1_ladder` | oi·cpcm para MAE(y) 대 rung(x: dft, g1, g2). 모델별 선, SH 실선, S 점선, LINE·null 표시, 0.13 가로선, 공통 집합(P_012, 1단계는 P_01), CI 막대 |
| `figL2_targets` | 9 타깃 para MAE ÷ null, rung × arm (REP) |
| `figL3_bends` | oi·cpcm의 민감도 대 위양성률. rung × arm × 모델, 목표 영역(≥ 0.7, ≤ 0.1) 표시 |
| `figL4_examples` | rev 9 fig3 계열 3개 × oi·cpcm: DFT 정답 대 REP(SH) 예측 (dft, g1, g2) |

### 14-2. `results/SUMMARY_LADDER.md` (한국어)

맨 앞 네 문장:
1. Q·V로 바뀐 반응·짝 수
2. rung × arm 6개의 단계 2 판정
3. 13-3 해석 규칙의 결론 문장
4. g2 적용 범위

절 순서: 시작 조건 → Q → V → G0 → F0 → P → C → A(commit hash) → M(단위 표, gate) → E(판정, 9 타깃 표, 꺾임, 비교, 라벨 교정 효과, 같은 TS 나눔) → 이탈 → 비용 → 한계.

꼭 적을 한계:
- 기준 구조를 분자별 최저로 바꿨다(Coley 절차의 원칙과 같고, 원래 라벨도 함께 보고).
- constrained dipole 기준은 그대로 두었다.
- TS 형태(conformer) 선택의 무작위성은 이 명령서에서 다루지 않았다(D1 검증 V3: 다시 돌리면 같은 TS 0.50).
- T1은 이미 여러 번 본 분할이다.
- dft rung은 배포할 수 없다(라벨 TS를 입력으로 씀).
- LODO·LOSO는 이 명령서에 없다.
- autodE seed가 고정되지 않아 G2는 다시 돌리면 달라질 수 있다.
- 꺾임 판정 근거 모의실험은 점마다 독립인 잡음을 가정했다.

수치는 모두 파일에서 옮긴다.

## 15. 규칙 (HPC)

- python, ORCA, xtb, RDKit은 모두 `sbatch`(`--time=48:00:00`)로만 돌린다. 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만 한다.
- MaxJobs 10, MaxSubmit 20(배열 원소 포함). **D1-G2 job이 먼저다.** 이 명령서는 동시에 최대 4개를 쓰고, D1-G2가 쉬는 동안만 더 쓴다.
- GPU 파티션은 `sinfo`로 확인하고 FU2와 같은 방식(GPU 종류 고정 안 함)으로 쓴다. 비결정성은 허용하고 기록한다.
- 읽기만 하는 것:
  - `labels_d1.json`
  - D1 run·job 폴더(`ref_result.json`, 기준 구조 xyz, 입력 파일 포함)
  - `d1_pair/` scratch와 결과(FU2 units 포함)
  - `d1_hf/` scratch와 결과
  - `$G2/`
- 새 파일은 `analysis/d1_ladder/`, `$LAD/`, 이 명령서에만 쓴다. 원래 라벨 파일을 덮어쓰지 않는다.
- 원자적 쓰기, 멱등.
- 결과를 본 뒤 Q·V 규칙, rung·arm·모델·grid·β 규칙·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

## 16. STOP

| 단계 | 조건 |
|---|---|
| 0 | 1절 시작 조건 중 하나라도 아님 |
| Q | 기준 에너지 키나 조립 함수를 찾을 수 없음; 같은 키 안에서 원소·결합 그래프가 다름; Q-a–Q-e 중 하나라도 실패 |
| V | 시험 묶음 기준 전체 예상 > 5,000 코어시간; DFT 최적화 실패(허수 진동수 포함) > 10 %; 라벨 단일점 입력을 좌표만 바꿔 만들 수 없음 |
| G0 | 원자 순서 gate 실패 > 0.5 %; Freq 실패 > 1 % |
| F0 | Q-a·T-a·F-1 실패 > 1 %; B6 계산 불가 > 1 % |
| P | P-a 실패; P_g1 ≠ FU2 짝 목록; P_dft가 P_g1보다 2 % 넘게 적음 |
| C | L-1–L-11, smoke 중 하나라도 실패 |
| A | ADD3 commit 전 학습 시도 |
| M | gate M-1′ 실패; 캐시 지문 불일치; 튜닝·β에 시험 짝 사용 |
| E | 같은 집합에서 비교하는 모델 사이 시험 짝 목록 불일치 |
| 2단계 | D1-G2 F·H gate 실패(D1-G2의 STOP) → 그쪽 해결까지 대기 |
| 언제든 | 이 문서에 없는 방법 변경이 필요해 보임 |

STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.
