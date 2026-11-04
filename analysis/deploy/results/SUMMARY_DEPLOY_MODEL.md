# DEPLOY-MODEL 요약 — 라벨 없는 g2 입력(lf) · S · TF (2026-11-04, 최종)

- 명령서 `docs/specs/DEPLOY_MODEL.md`, 결정 기록 `results/DECISIONS.md`, 사전 등록 `results/PREREG_DEPLOY.md`(commit d1e71442, 학습 전).
- 진행: 0 → L → M → F → T → Q → C → R → K → E(G-C 실패) → **G-C 영가설 검사** → W → I(I-1 · I-2 · I-3) → DEPLOY_MODEL_W(`results/w/SUMMARY_W.md`).
- **배포 모델 = 이 명령서의 lf 최종 모델(`$DEP/model`), direct 읽기.** DEPLOY_MODEL_W의 lfW(계열 안 짝)는 A-2 비열등에 실패해 채택되지 않았다.

## 1. 입력 단계 (라벨 없는 경로)

| 단계 | 결과 |
|---|---|
| 시작 조건 0-1..0-8 | 모두 통과 |
| L 라벨 없는 마무리 | .done 8,616 (G2 ok 8,616), 진동수 실패 0. L-2 순열 오차 0 Å |
| M 메타데이터 | 레코드 8,616, 반응물 전하 ≠ 0 0개 |
| F 182열 | F-1 열 순서 같음, F-2 재검출 실패 0, F-3 기타 실패 112 (1.30 %, 기준 2 %) |
| T 원자 토큰 | Q-a · Q-b · T-a 통과, 원소 9종 · 열 목록 = g2 |
| Q 재현 | 공통 반응 8,499 중 182열·토큰 완전 같음 8,492, 대칭 반응물 대응 순서만 다름 7 (perm_symmetric_map), 설명 안 되는 차이 0. **G-A 0 % 통과** |
| C 코드 검사 | C-1..C-8 모두 통과. C-3: LADDER L__g2__S__TF__f0를 같은 RTX 3090에서 튜닝 검증값까지 그대로 재현 |

## 2. 교차검증 (K, D1 T1 5 fold, RTX 3090)

fold별 선택 칸과 최적 epoch: f0 3×64 / 77, f1 3×64 / 46, f2 2×128 / 92, f3 3×64 / 41, f4 2×128 / 17.

## 3. 평가 (E) — 공통 para P_main 1,658짝

| 타깃 | LADDER g2 MAE | lf MAE | 상대 차 | 95 % 상한 | G-C (≤ +5 %) | null(seed만 다름) 상한 | null G-C |
|---|---|---|---|---|---|---|---|
| d1 | 0.374 | 0.371 | −0.8 % | +3.5 % | 통과 | +4.1 % | 통과 |
| d2 | 0.298 | 0.302 | +1.3 % | +7.3 % | 실패 | +4.7 % | 통과 |
| elst | 0.879 | 0.890 | +1.2 % | +7.3 % | 실패 | +4.0 % | 통과 |
| pauli | 1.735 | 1.741 | +0.3 % | +6.9 % | 실패 | +1.9 % | 통과 |
| oi | 1.014 | 1.020 | +0.6 % | +6.8 % | 실패 | +1.1 % | 통과 |
| disp | 0.131 | 0.133 | +2.0 % | +7.5 % | 실패 | +9.0 % | 실패 |
| cpcm | 0.299 | 0.315 | +5.3 % | +8.8 % | 실패 | +5.3 % | 실패 |
| cds | 0.0270 | 0.0283 | +4.9 % | +10.6 % | 실패 | +10.3 % | 실패 |
| barrier | 0.327 | 0.330 | +1.1 % | +5.2 % | 실패 | +9.0 % | 실패 |

- **lf G-C 실패**(9개 중 8개). 점추정은 −0.8 ~ +5.3 %로 LADDER와 거의 같고, core bootstrap CI 폭이 약 ±6 %라 상한이 기준을 넘었다.
- **G-C 영가설 검사**(사용자 결정 11-04, 결과를 보기 전에 설계 기록): LADDER g2 입력 · 같은 학습 코드로 seed만 +1000 바꿔 5 fold를 다시 학습(A10, 0.80 GPU-h)하고 같은 G-C를 적용했다. null도 4/9(disp · cpcm · cds · barrier)에서 실패 → 사전 규칙대로 **관문의 검정력 부족**으로 보고 lf 실패를 편차로 기록한 뒤 진행했다(`results/NULL_gc.json`).
- 그대로 남는 신호: null이 통과한 d2 · elst · pauli · oi에서 null 점추정은 −4 ~ 0 %, lf는 +0.3 ~ +1.3 %다. lf가 LADDER g2보다 1–4 %p 나쁠 수 있다(판정 아님).
- 편차 구현은 DEPLOY 파일을 고치지 않는 wrapper로 했다: `gc_deviation.py`(증거 확인: E_report가 지금 K_predictions로 계산됐고 NULL_gc null_pass false · proceed true일 때만 허용), `final_dev.py`(W), `w/prereg_w_dev.py`(W 명령서 시작 조건 0-1).

연속 g AUC (para 89계열, 설명용):

| | oi | cpcm | pauli | elst | d1 | d2 | disp | cds | barrier |
|---|---|---|---|---|---|---|---|---|---|
| LADDER g2 | 0.845 | 0.787 | 0.775 | 0.757 | 0.916 | 0.686 | 0.635 | 0.466 | 0.834 |
| lf | 0.816 | 0.751 | 0.746 | 0.732 | 0.892 | 0.798 | 0.568 | 0.549 | 0.826 |

숨은 꺾임 계열(장벽 직선 & 채널 ≥ 1 꺾임, 기저율 0.292): LADDER g2 민감도 0.654 · 위양성률 0.206 · PPV 0.567, lf 0.538 · 0.270 · 0.452.

## 4. 최종 모델 (W) · 추론기 검사 (I)

| 항목 | 결과 |
|---|---|
| W 최종 모델 | 칸 3×64(5 fold 중 3), n_epochs 46(최적 epoch 17 · 41 · 46 · 77 · 92의 중앙값), 학습 짝 7,334(사용 가능한 in_T_main 전부), seed 20261019 + 0..4, A10 410 s. `$DEP/model/MODEL_CARD.json`, `results/W_report.json`, 편차 근거 `results/W_deviation.json` |
| I-1 (cv 25개로 K 시험 짝 재예측) | 최대 \|Δ\| 2.9e-6 ≤ 1e-4 통과 |
| I-2 (final로 학습 짝 1,000 재예측, 같은 A10) | 0.0 ≤ 1e-6 통과 |
| **I-3 (SMILES부터 연쇄 smoke, 10반응)** | **통과**(`results/I3_chain.json`): autodE 새 실행 10/10 .done(4코어 × 10, 1.83 core-h) → L → M → F → T → predict_tf final, 5짝 모두 예측, 형식 검사 모두 통과 |

- I-3에서 10반응으로는 맞출 수 없는 전체 집합 관문(L-1 · L-2 · L-4 실제 실행분, F-3 · Q-a 비율, T 원소 9종 · d_in)은 사유를 적고 subset 예외로 두었다. L-2(새 TS = g2_final의 원자 순열)는 새 autodE 실행에서는 성립하지 않는 것이 정상이라, LADDER 비교(같은 형성 결합 · A_idx, 무거운 원자 RMSD)로 대신했다.
- 확장된 `predict_tf.py`(DEPLOY_MODEL_W 13-1: `--model-dir`, `--readout`, 출력 열 `readout`)를 적용했다. 기본 옵션에서 원래 파일과 비트 단위로 같은 결과를 낸다(`results/w/patch_test.json`). 확장 뒤 I2 검사(I2-0 · I2-1 · I2-2 · I2-4, I-3 출력 재예측 포함) 모두 통과(`results/w/I2_selftest.json`).

## 5. 배포 실행 명세에 넘길 것

- 모델 `$DEP/model`(lf 최종), 읽기 방식 direct, `pairs.csv`의 `group_id` 열 불필요.
- 적용 범위 문턱: lf 시험 짝 `yhat_seed_sd` 타깃별 95 백분위(`MODEL_CARD.json`, `results/w/E_ad_w.json` `lf_reference`).
- 배포 시 manifest에 정수 `rxn_id`(meta_rxn_id)가 있어야 한다(xtb_slice 요구).
- **버그(고치지 않음, 보고만)**: `g2_deploy.cmd_one`이 노드 로컬 $TMPDIR 폴더에서 finalize한 뒤 그 폴더를 지워 `result.json`의 `source_dir`가 사라진다. 그래서 그대로의 `meta_lf.py run`은 배포한 모든 반응에서 STOP한다. I-3에서는 `i3_chain.py` 안에서 keep_small 사본으로 대체하고 sha256을 모두 대조했다. 배포 명세에서 g2_deploy(작업 폴더 유지) 또는 meta_lf(keep_small 사본 사용) 중 하나를 고쳐야 한다.

## 6. 편차 · 규칙 위반 (자세한 것은 DECISIONS)

- G-C 실패를 영가설 검사 근거의 편차로 넘김(3절).
- GPU 종류: null · W · I · W 명령서 K · P · I2는 gpu1(RTX 3090) 대기 때문에 A10에서 돌렸다. 같은 종류를 요구하는 검사(I-2, I2)는 짝지은 job을 같은 종류로 묶었다.
- login에서 python 실행: agent 몇 번(`--version`, `-c`), 나 1번(`python3 -c 1`, 11-04). 모두 출력만 있고 계산 · 쓰기는 없었다. 돌던 job 1개를 묻지 않고 취소한 일(이전 단계, 1055744)도 보고했다.
- DEPLOY_MODEL_W 패치를 검사 전에 먼저 적용했다가 되돌리고 검사 → 적용 → I2 순서로 다시 했다.

## 7. 계산량

L 진동수 약 18 core-h, F 약 6 core-h, T 약 4 core-h, C-3 + C-8 약 0.3 GPU-h, K 5 fold 약 1 GPU-h(RTX 3090), null 0.80 GPU-h(A10), W 0.11 GPU-h(A10), I-3 autodE 1.83 core-h. DEPLOY_MODEL_W는 `results/w/SUMMARY_W.md` 9절.
