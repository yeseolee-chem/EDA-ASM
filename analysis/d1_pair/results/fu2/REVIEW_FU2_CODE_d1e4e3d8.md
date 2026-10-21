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

참고문헌: Meli & Biggin, "spyrmsd: symmetry-corrected RMSD calculations in Python", *J. Cheminform.* 12 (2020), DOI 10.1186/s13321-020-00455-2.
