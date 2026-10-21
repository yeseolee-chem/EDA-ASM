# FU2 코드 검토 — commit d1e4e3d8 (Phase Q + T + C 코드)

검토 대상: `analysis/d1_pair/fu2/{atoms, tokens, splits, models, train, fu2_common}.py`, `fu2/tests/run_tests.py`, `results/fu2/{Q_report, T_report, S_splits, C_review}`.

방법:
1. 코드 전부를 명령서(D1PAIR_FU2.md)와 한 줄씩 대조했다.
2. 가짜 반응 40개로 실제 `models.py`·`train.py` 함수를 CPU에서 따로 돌려 성질을 확인했다.
3. 코드를 쓰지 않은 별도 검토자 1명이 버그만 찾았다.

## 1. 결론

- **모델·학습 코드(models.py, train.py)는 명령서와 일치하고, 막는 버그나 정보 누출은 없다.** (확신 0.9)
- **고칠 것이 하나 있다: 원자 대응.** 반응물에서 "모양이 같은 원자"(페닐의 두 ortho C, CF₃의 F 3개, NO₂의 O 2개, tBu·SiMe₃의 C 등)는 TS 원자와 짝지을 때 위치를 보지 않고 첫 번째로 찾은 대응을 쓴다. 그래서 해당 원자의 q_rel, v_rel, Δq_dist, Δv_dist가 반대편 원자 값일 수 있다.
  - 토큰 숫자 14열 중 이 4열, 그리고 대칭 원자에만 해당한다. q_frag·q_TS·간선은 영향이 없다.
  - 영향 크기는 아직 모른다(확신 0.6로 "결과에 의미 있게 영향"). 하지만 고치는 데 xTB 재계산이 필요 없어 싸다. **입력이 바뀌므로 ADD2(입력 sha256 고정) 전에 고친다.**
- `C_tests.json`(C1–C12)과 `C_smoke.json`이 아직 commit에 없다. **승인은 아래 수정 후 두 파일이 all_pass / ok일 때 유효하다.**

## 2. 확인한 것

| 항목 | 방법 | 결과 |
|---|---|---|
| 명령서 일치 (토큰, 그래프, 학습 규칙, 세 모델 구조, grid, 손실, 증강) | 코드 대조 | 일치. 명령서가 정하지 않은 값(GNN 거리 RBF 0–4 Å, TF 0–10 Å, pooling 모양 등)은 무난한 선택 |
| Q-a 재현 | Q_report | 30열 최대 차 0, Q-b 7e-8, 확장 8,336/8,337 통과(1 반응 제외) |
| T-a | T_report | 8,336 반응, NaN 없음, 형성 간선 모두 2개, 원자 없는 짝 297개(feature 없는 기준 짝 296 + 1) |
| 학습·시험 짝 = arm C | splits.py | `d1_train.masks(u)`·`inner_folds(tr)[0]`을 그대로 씀 → C-XGB와 같은 짝. 원자 없어 빠지는 짝은 T1 f0 시험 1개 |
| 반대칭·원자 순서·회전·batch 독립 | 가짜 데이터, 3 모델 × 2 변형 × 4칸 = 24개, float64, 무작위 가중치 | g(a,b)+g(b,a) = 0, g(a,a) = 0(정확히), 원자 순서 1e-10, 회전 1e-9, batch 4e-9 |
| AW 국소성 | 같은 방법 | L = 1: 5 Å 밖 원자를 움직여도 ε_i 변화 0. L = 2: 변함(수용 범위 10 Å, 예상대로) |
| 손실의 항등식 벌점 | numpy로 따로 계산 | 차 3e-17 |
| 순서 바꾸기 증강 | `epoch_batches` | 120/120 짝에서 y 부호가 같이 바뀜 |
| 학습 가능성 | 가짜 데이터 60 epoch | 세 모델 모두 학습 MAE가 null의 9–19 %로 내려감 |
| 정답 빈 값 | F_pairs.csv | 쓰는 짝(P_sens, A_main)에서 10 타깃 모두 0 |
| 누출 | 코드 대조 + 별도 검토자 | 표준화(원자 feature·전역 토큰·y) = 그 학습의 학습 짝 반응만. 튜닝·early stopping = 첫 안쪽 fold(core 겹침 0). gate M-1′은 시험 짝 입력만 씀 |
| 저장 예측 | 코드 대조 | pair_id·y·yhat 정렬 맞음, 둘 다 kcal/mol |
| GPU 메모리 (추정) | 별도 검토자 | GNN L = 6·폭 128 학습 최대 약 10 GB, 예측 bs 256 약 5 GB → A10(24 GB)·A6000(48 GB) 안 |

## 3. 학습 전에 고칠 것 (ADD2 전)

| # | 무엇 | 왜 | 어떻게 |
|---|---|---|---|
| F1 | **대칭 원자 대응을 위치로 정하기** (꼭) | `fragmenter.partition`은 같은 원자 집합의 첫 VF2 대응을 그대로 쓴다(fragmenter.py 135–156). 반응물 그래프의 대칭(automorphism)을 위치로 가리지 않는다. 그래서 syn ortho C의 Δq_dist가 q_frag(syn) − q_rel(anti)가 될 수 있다. Phase E 해석 검사("작용기 원자의 Δq를 0으로")도 흐려진다 | 반응물 heavy 그래프의 automorphism 중 **반응 원자 5개를 고정한 것**을 모두 나열한다. 각 대응으로 TS 조각과 반응물의 heavy 원자를 Kabsch 정렬해 RMSD가 가장 작은 것을 쓴다(symmetry-corrected RMSD와 같은 생각). H는 지금 규칙(바뀐 heavy 원자의 owner H 평균). 저장된 npz(q_gs1·q_gs2, rel*_pos, ts_pos)에서 다시 만들고 xTB는 다시 돌리지 않는다. `ATOM_VERSION = fu2-atoms-2`. 반응 원자가 고정이라 Q-a는 그대로 통과해야 한다. automorphism이 10,000개를 넘으면 STOP |
| F2 | **대응 검사 gate Q-c** (꼭) | 지금 검사(Q-a, C11)는 반응 원자 5개만 보거나, 대응을 만든 같은 규칙을 다시 확인한다(순환). 나머지 원자의 대응은 아무도 독립적으로 확인하지 않는다 | `Q_report`에 추가: 원소 일치(Z_TS = Z_rel, 하나라도 틀리면 STOP), 대응이 바뀐 반응·원자 수, 바뀐 원자의 \|Δq_dist\| 변화 분포, 조각별 RMSD(전·후) |
| F3 | 캐시 지문에 코드 포함 | 지문에는 코드 버전 문자열 `d1pair-fu2-1`만 있다. 고친 뒤 문자열을 안 올리면 옛 결과를 조용히 다시 쓴다 | 지문에 models.py·train.py·fu2_common.py·tokens.py의 sha256과 `S_splits.json` sha256을 넣는다(안쪽 fold 목록까지 고정) |
| F4 | 발산하면 멈추기 | 검증 MAE가 NaN이면 best_epoch = 0 → 최종 학습이 조용히 1 epoch만 돈다 | 손실·검증 값이 NaN/inf이거나 best_epoch = 0이면 STOP |
| F5 | 검사 2개 추가 | "A+G"가 전역 토큰을 실제로 쓰는지, 저장 파일이 맞는지는 아무 검사도 확인하지 않는다 | **C13:** 전역 토큰만 바꾸면 A+G 출력은 변하고 A는 그대로(≤ 1e-12). **C14:** C8의 run_unit 결과 parquet을 다시 읽어 y = F_pairs dd(pair_id 기준), yhat = predict × s_y(≤ 1e-6) 확인 |

고친 뒤 다시 할 것: Q gate(Q-c 포함) → T build → C1–C14 + smoke → commit·push → 검토 → Phase A.

## 4. ADD2에 꼭 적을 것 (Phase A)

- **대표 구성:** rev 9 `d1_eval.representatives`와 같은 방식으로 고른다.
  - 타깃마다, 고른 칸의 안쪽 검증 MAE를 kcal/mol로 바꾼다(× 그 fold의 s_y).
  - 바깥 fold 5개 평균이 가장 낮은 구성을 쓴다.
  - 동점이면 AW → GNN → TF, A → A+G 순으로 정한다.
- **M-4(LODO·LOSO)의 "모델마다 대표 하나":** 그 모델의 두 변형 중, 칸을 고를 때 쓴 값(9 타깃 표준화 평균 검증 MAE)의 5 fold 평균이 낮은 쪽.
- **알려진 점:** 최종 학습은 튜닝 때 고른 epoch 수를 그대로 쓴다. 다만 짝이 약 1.33배라 optimizer step도 약 1.33배가 된다(명령서 규칙 그대로; 고치지 않고 적어 둔다).

## 5. 아직 없는 코드 (평가 전에 따로 검토)

- Phase E 평가 코드(C-XGB와 같은 시험 짝, bootstrap, Holm, 꺾임 판정).
- M-4 실행 코드.
- 대표 구성 고르기 코드.

이 셋은 결론을 정하는 코드라, 실행 전에 한 번 더 검토받는다.

## 6. Claude Code에 보낼 지시 (그대로 붙여 넣기)

```
FU2 코드 검토 결과 (사용자 검토, 2026-10-21) — results/fu2/REVIEW_FU2_CODE_d1e4e3d8.md 참조.
Phase A로 가기 전에 F1–F5를 고치고, C_review.md 2절에 적은 뒤 다시 멈춰라.
F1 대칭 원자 대응: 반응물 heavy 그래프 automorphism 중 반응 원자 5개 고정인 것을 모두 나열, TS 조각과
   Kabsch RMSD 최소인 대응 사용(H = 바뀐 heavy 원자의 owner H 평균). 저장된 npz에서 재계산(xTB 재실행 없음),
   ATOM_VERSION fu2-atoms-2, Q-a 그대로 통과해야 함, automorphism > 10,000이면 STOP.
F2 gate Q-c: 원소 일치(틀리면 STOP), 대응 바뀐 반응·원자 수, 바뀐 원자의 |Δq_dist| 변화 분포, 조각별 RMSD 전·후 보고.
F3 캐시 지문에 models.py·train.py·fu2_common.py·tokens.py sha256 + S_splits.json sha256.
F4 손실·검증 NaN/inf 또는 best_epoch = 0이면 STOP.
F5 C13(전역 토큰을 바꾸면 A+G만 변함, A는 ≤ 1e-12), C14(run_unit parquet: y = F_pairs dd by pair_id, yhat = predict × s_y ≤ 1e-6).
그다음 Q gate → T build → C1–C14 + smoke → commit·push (C_tests.json, C_smoke.json 포함) → 사용자에게 알리고 멈춤.
ADD2(Phase A)에 대표 구성 규칙(REVIEW 4절)을 그대로 적는다. Phase E·M-4·대표 고르기 코드는 실행 전 따로 검토받는다.
```

## 7. 추가 — C12 AW 실패 (commit 06457e82)

- **결과:** 64짝을 외우는 정도(학습 MAE ÷ null)는 AW 0.066(기준 0.05 넘김), GNN 0.006, TF 0.003. AW에서 가장 나쁜 타깃은 cpcm 0.13, cds 0.12, d2 0.09다. C1–C11과 smoke는 모두 통과했다(smoke GPU 최대 8.3 GB, Phase M 상한 약 135 A10 GPU-시간).
- **판단:** 버그보다는 "외우는 속도가 느린 것"일 가능성이 크다(확신 0.8).
  - 버그(정답이 어긋남, 기울기가 끊김)라면 보통 null 근처(≈ 1.0)에 머문다. AW는 1.0에서 0.066까지 내려갔다.
  - 반대칭·국소성·정답 정렬 검사(C1–C11)를 모두 통과했다.
  - AW는 원자 기여의 단순 합이고, 가장 작은 칸(1층, 폭 64, 2,400 step)이라 셋 중 가장 제약이 크다.
  - 5 % 기준은 명령서에서 정한 임의의 값이다.
- **진단 (기준은 바꾸지 않음; 판정 규칙을 미리 정함):**
  1. AW 같은 설정(dropout 0, batch 8)으로 3,000 epoch.
  2. AW 가장 큰 칸(L = 2, 폭 128), 300 epoch.
  - 둘 중 하나라도 0.05 미만이면 "속도 문제, 버그 아님"으로 C_review에 기록하고 진행한다.
  - 둘 다 0.05 위에서 평평해지면 AW 구현을 다시 점검한다(STOP).
  - 이 진단은 F1 수정 후 C를 다시 돌릴 때 같이 돌린다. 토큰이 바뀌므로 C12는 어차피 다시 돌려야 한다.
- **진행 범위:** F1–F5 수정 → Q gate → T → C1–C14 + C12 진단 → push → 멈춤. ADD2와 Phase M은 아직 하지 않는다. F1이 토큰 값을 바꾸므로, 지금 학습하면 다시 해야 한다.

참고문헌: Meli & Biggin, "spyrmsd: symmetry-corrected RMSD calculations in Python", *J. Cheminform.* 12 (2020), DOI 10.1186/s13321-020-00455-2.

## 8. 재검토 — commit c2838ba1 (F1–F5 반영)

**결론:** F1–F5가 지적대로 들어갔고 막는 문제는 없다. 남은 결정은 하나다: 거울상 → **반사 허용(O(3)) 채택 권장** (확신 0.85).

| 항목 | 확인 (코드 + 결과 파일) | 판정 |
|---|---|---|
| F1 대칭 대응 (`atoms.sym_remap`) | 반응 원자 5개를 고유 표시로 고정한 automorphism × 저장 대응 합성, heavy 원자 Kabsch RMSD 최소, 동점은 기존 유지, H는 owner 규칙, bijection·공유결합 간선 불변 확인, automorphism > 10,000이면 STOP(최대 12). xTB 재계산 없이 변환, 실패 0 | 맞음 |
| F1 효과 | 대응이 바뀐 반응 2,414 (heavy 원자 7,763). 바뀐 조각의 RMSD가 크게 줄어듦(A 831, B 1,110 조각에서 > 0.1 Å, 중앙값 0.56 / 0.44 Å), 나빠진 조각 0 → 실제로 위치가 틀린 대응이었다. 값 변화는 작음: \|Δq_dist\| 중앙값 0.0004, p99 0.024, 최대 0.048 e | 고칠 가치 있었음 |
| F2 Q-c | 원소 불일치 0, 반응 원자 재대응 0 (둘 다 STOP 조건), Q-a 0, Q-b 7e-8 | 통과 |
| F3 지문 | 코드 5개 파일 sha256(import 때 한 번) + S_splits sha256. `Data()`가 토큰·Q·분할 버전 확인, collect가 모든 unit 지문을 현재와 비교 | 맞음 |
| F4 발산 | 학습 손실·검증 NaN/inf, best_epoch < 1이면 exit 3 | 맞음 |
| F5 C13 / C14 | C13: A 변형 차 0, A+G 30/30 변함, 전역 토큰 = X182 행. C14: 17,130행, y = F_pairs dd(차 0), yhat = predict × s_y(차 0) | 통과 |
| C12 | AW 가장 작은 칸만 0.066. 같은 64짝에서 폭 128은 0.008, 1,000 epoch(T_max 1,000)는 0.0008 → 용량·학습량 문제, 버그 아님. 사용자 결정 "무시" | 받아들임 (ADD2에 편차로 적기) |

**거울상 결정 — O(3) 권장.**
- 비유: 거울상 이형태는 왼손·오른손 장갑과 같다. 회전만 허용하면 왼손 장갑을 오른손에 맞추느라 엄지를 새끼손가락에 대게 된다(대칭 원자에서 반대편 원자를 고름). 반사를 허용하면 엄지와 엄지가 맞는다.
- 근거: 같은 분자의 거울상 구조는 대응 원자마다 xTB 전하·결합 차수가 같다(해밀토니안과 등방성 용매 ALPB가 반사에 대해 변하지 않음). 그래서 TS 조각이 반응물 이형태의 거울상일 때는 반사를 허용한 대응이 "같은 자리의 같은 원자"를 준다.
- 진짜 입체중심이 있어도 같은 원자의 전하는 거울상에서 같다. 그래서 O(3)를 써서 값이 틀어지는 경우는 없다.
- 영향: 대응이 바뀌는 조각은 A 539, B 1,031이다. q_rel 변화는 중앙값 0.0002, p99 0.012, 최대 0.039 e로 작다. 반응 원자와 Q-a는 영향이 없다.
- 비용: npz 변환만 다시 하면 된다(xTB 없음).
- 덧붙임: B 조각에서 v2 RMSD > 1 Å가 1,901개다. 대부분 거울상 맞춤으로 설명된다(1,697). O(3)를 쓴 뒤에도 1 Å를 넘는 조각은 회전 이성질체 차이다. 이때 Δq_dist에는 이형태 변화도 섞이는데, 라벨 d1·d2도 같은 기준(완화된 반응물)이라 정의가 일관된다. 남는 조각 수와 예시 5개만 보고받는다.

**작은 메모:** collect는 지문에서 GPU 이름만 빼고 비교한다. 그래서 env_d1fu2가 아닌 환경에서 돌리면 STOP한다(조용히 틀리지는 않음). collect도 같은 환경에서 돌린다.

**Claude Code에 보낼 지시:**
```
사용자 검토 (c2838ba1): F1–F5 반영 확인. C12 AW 작은 칸 실패는 사용자 결정대로 무시하고 ADD2에 편차로 기록.
결정: 원자 대응 Kabsch에 반사 허용(O(3)) — mirror_diag의 O3 선택을 sym_remap 규칙으로 채택.
 - sym_remap: 후보마다 RMSD = min(고유 회전, 비고유 회전); 동점(1e-6 Å)은 기존 대응 유지; 반응 원자 고정 그대로.
 - ATOM_VERSION fu2-atoms-3 (fu2-atoms-2 npz에서 변환, xTB 없음), TOKEN_VERSION fu2-tokens-3.
 - Q gate 다시 (Q-a·Q-b·Q-c). Q-c는 O(3) RMSD 기준. v2 대비 대응이 바뀐 조각·원자 수, |Δq_dist| 변화 분포,
   O(3) RMSD > 1 Å 조각 수와 예시 5개(rxn_id, 조각, 추정 원인) 보고.
 - T build → C1–C14 + smoke 다시 → commit·push.
그다음 Phase A: PREREG_D1PAIR_ADD2에 REVIEW 4절(대표 구성 규칙 등) + 원자 대응 규칙(대칭 + O(3)) + C12 결정(편차)
 + Q·T·분할 sha256을 적고 commit·push → 사용자 확인 뒤 Phase M.
```

참고문헌: Kabsch, "A discussion of the solution for the best rotation to relate two sets of vectors", *Acta Cryst.* A34 (1978), DOI 10.1107/S0567739478001680.

## 9. 재검토 — commit 35313e3a (반사 허용 + Phase C 재실행) → **승인**

**판단: 코드 승인. Phase A(ADD2)로 간다.** (확신 0.9)

| 항목 | 확인 | 판정 |
|---|---|---|
| 반사 허용 구현 | 후보마다 min(고유, 비고유) Kabsch RMSD. 출발점은 언제나 원래 VF2 대응(*_v1)이라 v1·v2 파일 어느 쪽에서 변환해도 같은 결과. 고유 회전 규칙의 선택도 기록됨 | 맞음 |
| 자기 검사 | mirror_diag가 같은 O(3) 선택을 따로 다시 계산 → 불일치 0 (0이 아니면 STOP) | 통과 |
| Q gate | Q-a 0, Q-b 7e-8, 원소 불일치 0, 반응 원자 재대응 0, automorphism 최대 12 | 통과 |
| 효과 | VF2 대비 바뀐 반응 2,280. 1 Å 넘게 어긋나는 조각: A 175 → 103, B 1,901 → 510 (고유 회전 규칙 대비). 나빠진 조각 0. 전하 변화는 작음 (\|Δq_dist\| p99 0.024 e) | 기대대로 |
| C1–C11, C13, C14 | 통과 (C14 y·yhat 차 0) | 통과 |
| C12 | AW 가장 작은 칸 0.064 (폭 128은 0.0075, 1,000 epoch는 0.0007) | 사용자 결정대로 판정 제외 |
| smoke | gate M-1′ 0, 상한 약 132 A10 GPU-시간 | 통과 |

**남은 작은 것 (막지 않음):** 반사 허용 뒤에도 1 Å 넘게 어긋나는 조각(A 103, B 510)의 예시 5개는 아직 보고되지 않았다. 회전 이성질체 차이일 것으로 본다. ADD2의 "알려진 점"에 수만 적고, Phase E 해석 그림에서 이 반응들을 표시한다.

### ADD2(PREREG_D1PAIR_ADD2)에 꼭 들어갈 것

1. **입력 고정:** code commit 35313e3a, 코드 sha256 5개(models, train, fu2_common, tokens, d1pair_common), Q_report·Q_manifest·T_report·T_manifest·S_splits·pairs.parquet sha256, atom/token 버전 fu2-atoms-3 / fu2-tokens-3.
2. **모델·학습 규칙:** 명령서 M-1–M-3 그대로 (3 모델 × 2 변형 × grid 4칸, λ 0.1, AdamW 1e-3, cosine 300, dropout 0.1, batch 64, patience 30, 튜닝 = 첫 안쪽 fold, 최종 seed 5개 평균). GPU = RTX A6000 (gpu4/gpu5). GPU 비결정성 허용.
3. **대표 구성 (주 비교에 쓰는 원자 모델):** 타깃마다 고른 칸의 안쪽 검증 MAE를 kcal/mol로 바꿔(× 그 fold의 s_y) 바깥 fold 5개 평균이 가장 낮은 구성. 동점이면 AW → GNN → TF, A → A+G. 시험 점수는 보지 않는다.
4. **주 비교:** 9 타깃(8채널 + barrier; barrier는 모델이 직접 낸 값)마다 d = MAE(C-XGB) − MAE(원자 대표). P_main 시험 짝, 원자 없는 짝은 양쪽에서 똑같이 뺀다. core cluster bootstrap 2,000회 (seed 20261019), Holm 9개. 판정: Holm p < 0.05이고 CI 하한 > 0인 타깃이 5개 이상이면 "원자 입력이 낫다", 2–4개면 "일부 채널에서만", 1개 이하면 "이득 없음". C-XGB = 기존 arm C의 XGB(원래 grid)다.
5. **부 비교 (보정 없음):** 모델 사이, 변형 사이(A+G 대 A, A+G 대 C-XGB), para 짝, C 대표(SVR이 끝나면 6개 중), XGBW, LODO·LOSO.
6. **M-4:** 모델마다 대표 변형 하나 = 9 타깃 표준화 평균 검증 MAE의 5 fold 평균이 낮은 변형. LODO·LOSO도 같은 규칙으로 칸을 다시 튜닝한다.
7. **Hammett 수준 (부):** rev 9 부록 D 규칙(`hammett_eval.calls`)과 같게. 채널별 목표 MAE(0.28 × 꺾임 크기)는 사후 정의임을 적는다.
8. **계획과 다르게 한 점:** ① C12 AW 가장 작은 칸 실패를 판정에서 뺌(용량 문제, 진단 근거). ② 원자 대응을 대칭 + 반사 허용 규칙으로 고침(F1, O(3)). ③ 확장 계산 실패 1 반응 제외 → T1 f0 시험 1짝 빠짐. ④ 최종 학습 step 약 1.33배 (규칙 그대로).
9. **멈춤 규칙:** 학습 중 gate M-1′ 실패, NaN, 지문 불일치면 STOP. 시험 결과를 본 뒤 grid·규칙을 바꾸지 않는다. Phase E 코드는 실행 전에 따로 검토받는다.

### Claude Code에 보낼 지시

```
사용자 검토 (35313e3a): 코드 승인. Phase A로 진행.
results/PREREG_D1PAIR_ADD2.md + prereg_d1pair_add2.json 작성 — REVIEW_FU2_CODE_d1e4e3d8.md 9절 "ADD2에 꼭 들어갈 것" 1–9를 모두 담는다
(기존 PREREG·ADD1은 바꾸지 않음). commit·push 후 멈추고 알림 → 사용자 확인 → Phase M 제출(gpu4/gpu5 A6000).
```
