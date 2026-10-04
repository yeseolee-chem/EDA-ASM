# rev 6 SUMMARY — Espley 2024 모델과 우리 모델의 공정 비교 + 우리 8채널 모델

명세: `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md`. 사전 등록: `PREREG_REV6.md` / `prereg_rev6.json`(commit `6f4afcb9`, 모든 rev 6 학습 전).
수치는 모두 이 폴더(`results_rev6/`)의 파일에서 옮겼다.

rev 6은 두 모델을 같은 조건에서 비교했다. Espley 2024 모델은 그들의 AM1 46 feature와 그들의 코드를 썼고, XGB를 추가했다. 우리 모델은 xTB 182 feature와 우리 파이프라인을 썼다. 반응(C, 3,386개), 분할(seed 22/23/14/1/2), 정답(우리 SMD(water) 라벨)은 양쪽이 같다.
사전 등록한 주 비교는 각 쪽 대표 모델끼리 했고, Holm 보정을 적용했다. 우리 모델의 test MAE가 d1, d2, Int, ΔE‡ 네 타깃 모두에서 낮았다. 차이는 +0.42–+1.06 kcal/mol, Holm p ≤ 4.7e-4이고, 다섯 seed 모두에서 같은 방향이었다.
우리 8채널 모델(대표 모델은 대부분 KRR)의 test MAE는 다음과 같다(kcal/mol): elst 3.16, Pauli 5.72, OI 3.52, disp 0.425, CPCM 1.26, CDS 0.206, d1 1.54, d2 1.19.

## 1. 주 비교 (확증적, `B_primary_tests.csv`)

- 대상: 행 C(3,386).
- 대표 모델: §3-3 규칙대로 test 점수가 아니라 검증 MAE가 가장 낮은 모델을 골랐다.
  - ESP-C는 네 타깃 모두 XGB, OUR-O는 네 타깃 모두 KRR이다.
  - Espley 쪽은 test MAE가 가장 낮은 모델(test best-of)과 그들의 `y_train_CV` 규칙으로 고른 모델도 모두 XGB였다(`B_representative.json`).
- 검정: 차이 d = MAE(ESP-C) − MAE(OUR-O)를 seed별로 구하고 Nadeau–Bengio 보정 t-검정을 했다(J = 5). 네 타깃에 Holm 보정을 적용했다.

| 타깃 | ESP-C (XGB) MAE | OUR-O (KRR) MAE | d (95 % CI) | p | Holm p | OUR-O가 낮은 seed |
|---|---:|---:|---|---:|---:|---:|
| d1 (dipole 변형) | 2.42 | 1.54 | +0.88 (0.69, 1.08) | 2.3e-4 | 4.7e-4 | 5/5 |
| d2 (dipolarophile 변형) | 1.62 | 1.19 | +0.42 (0.36, 0.49) | 6.1e-5 | 1.8e-4 | 5/5 |
| Int (eint_spe) | 2.26 | 1.24 | +1.03 (0.92, 1.14) | 1.3e-5 | 5.3e-5 | 5/5 |
| ΔE‡ | 2.76 | 1.70 | +1.06 (0.81, 1.31) | 3.0e-4 | 4.7e-4 | 5/5 |

- 보조 지표(`B_summary.csv`)는 아래와 같다. 순서는 ESP-C XGB → OUR-O KRR이다.

| 타깃 | NMAE | r² | \|오차\| ≤ 1 kcal/mol 비율 |
|---|---|---|---|
| d1 | 0.436 → 0.277 | 0.786 → 0.906 | 0.273 → 0.446 |
| d2 | 0.364 → 0.269 | 0.843 → 0.906 | 0.430 → 0.555 |
| Int | 0.474 → 0.258 | 0.749 → 0.918 | 0.309 → 0.541 |
| ΔE‡ | 0.427 → 0.263 | 0.802 → 0.921 | 0.260 → 0.384 |

- seed별 test 행: 334–341, 학습 행: 2,705–2,710. 모든 arm·모델이 같은 행을 썼다(`B.json` rows_identical, Phase S sha256과 일치).
- 그림: `figures/fig1_primary`, `fig3_parity`, `fig4_seed_pairs`.

## 2. 튜닝 조건을 맞춘 비교 (설명적, `B_secondary_tests.csv`)

feature만 다르고 학습·튜닝 방식은 같은 두 쌍이다. 두 쌍 모두 다섯 seed 전부에서 OUR 쪽 MAE가 낮았다.

| 쌍 | 타깃 | ESP 쪽 MAE | OUR 쪽 MAE | d (95 % CI) | p |
|---|---|---:|---:|---|---:|
| ESP-O(XGB) 대 OUR-O(KRR), 우리 파이프라인 | d1 | 2.34 | 1.54 | +0.80 (0.55, 1.06) | 9.3e-4 |
| | d2 | 1.60 | 1.19 | +0.41 (0.35, 0.47) | 3.9e-5 |
| | Int | 2.24 | 1.24 | +1.01 (0.89, 1.13) | 2.0e-5 |
| | ΔE‡ | 2.75 | 1.70 | +1.06 (0.81, 1.30) | 3.0e-4 |
| ESP-C(XGB) 대 OUR-C(NN-4), Espley 프로토콜 | d1 | 2.42 | 1.59 | +0.83 (0.57, 1.09) | 8.7e-4 |
| | d2 | 1.62 | 1.21 | +0.40 (0.32, 0.48) | 1.4e-4 |
| | Int | 2.26 | 1.25 | +1.01 (0.95, 1.07) | 1.0e-6 |
| | ΔE‡ | 2.76 | 1.83 | +0.92 (0.61, 1.23) | 1.2e-3 |

OUR-C의 대표 모델은 네 타깃 모두 NN-4, ESP-O는 모두 XGB다.

## 3. 모델별 전체 (`B_summary.csv`, `fig2_models`)

모델 6개(Ridge, KRR, SVR, XGB, NN-2, NN-4) × arm 4개의 seed 평균 MAE를 모두 공개한다. 같은 모델끼리 ESP-C와 OUR-O를 비교하면 24쌍(모델 6 × 타깃 4) 모두 OUR-O가 낮았다. 모든 쌍이 다섯 seed 전부에서 같은 방향이고, 95 % CI가 0 위에 있다. 타깃별 차이 범위는 다음과 같다.

| 타깃 | 가장 작은 차이 | 가장 큰 차이 |
|---|---|---|
| d1 | +0.78 (XGB) | +1.42 (Ridge) |
| d2 | +0.34 (XGB) | +1.09 (KRR) |
| Int | +0.91 (XGB) | +1.68 (Ridge) |
| ΔE‡ | +0.87 (XGB) | +1.72 (Ridge) |

ESP-O 대 OUR-O도 같은 경향이다(+0.33 ~ +1.72). 네 arm의 모든 쌍은 CSV에 있다.

**NN이 두 프로토콜에서 같은 이유.** 우리 파이프라인은 seed마다 Hyperband로 튜닝했다. 그런데 다섯 seed 모두에서 Espley 프로토콜의 seed 23 단일 튜닝과 똑같은 하이퍼파라미터가 골라졌다(예: 2층 = dropout 0.2, 뉴런 64/32, l2 1e-3, lr 1e-3). NN 코드도 같다(P-2에서 비트 단위로 확인). 그래서 같은 feature에서는 두 프로토콜의 NN 예측이 비트 단위로 같다. ESP-C와 ESP-O의 NN, OUR-C와 OUR-O의 NN은 표에서 같은 값이다.

## 4. Σ채널 (설명적, `B_sigma_per_seed.csv`, `fig6_sumchannels`)

OUR-O의 채널별 대표 모델을 더해 Int와 ΔE‡를 만들었다.
- Int̂ = Σ6̂ + ĉ_ghost
- ΔE‡̂ = d1̂ + d2̂ + Int̂

채널별 대표 모델은 다음과 같다(`B_representative.json`).
- KRR: elst, Pauli, OI, disp, CDS, d1, d2
- NN-4: CPCM, c_ghost

| 타깃 | Σ채널 MAE | OUR-O 직접(KRR) MAE | Σ − 직접 (95 % CI) | p | ESP-C 대 Σ |
|---|---:|---:|---|---:|---|
| Int | 1.32 | 1.24 | +0.08 (0.04, 0.13) | 6.9e-3 | +0.95 (0.83, 1.06) |
| ΔE‡ | 1.77 | 1.70 | +0.07 (0.01, 0.14) | 0.038 | +0.99 (0.74, 1.23) |

8채널로 나눠 예측한 뒤 더하면, 직접 예측보다 0.07–0.08 kcal/mol 나빴다. 그래도 Espley 모델보다는 약 1 kcal/mol 낮았다.

## 5. 8채널 (Phase A, `A_summary.csv`, `A_representative.json`, `fig5_channels`)

주 결과는 행 C, §2-3 분할, OUR-O(182 feature, 우리 파이프라인)로 얻었다. Espley에는 이 채널을 맞히는 모델이 없으므로 이것은 비교가 아니라 우리 모델의 결과다. ESP-O(Espley 46 feature × 우리 파이프라인)는 참고열로만 둔다.

| 채널 | OUR-O 대표 | MAE | NMAE | r² | ESP-O 대표(참고) | ESP-O MAE |
|---|---|---:|---:|---:|---|---:|
| d1 | KRR | 1.54 | 0.277 | 0.906 | XGB | 2.34 |
| d2 | KRR | 1.19 | 0.269 | 0.906 | XGB | 1.60 |
| elst | KRR | 3.16 | 0.269 | 0.910 | XGB | 5.71 |
| Pauli | KRR | 5.72 | 0.249 | 0.922 | XGB | 9.36 |
| OI | KRR | 3.52 | 0.252 | 0.917 | XGB | 5.59 |
| disp | KRR | 0.425 | 0.119 | 0.982 | XGB | 1.483 |
| CPCM | NN-4 | 1.26 | 0.385 | 0.833 | XGB | 2.81 |
| CDS | KRR | 0.206 | 0.346 | 0.867 | XGB | 0.355 |
| (c_ghost) | NN-4 | 0.233 | 0.474 | 0.715 | — | — |

**부록**(`A_appendix_*`)은 우리 라벨 전체에서 했다.
- 행: 우리 라벨 ok ∩ R_feat ∩ R_hyg = 4,963. Espley 반응이 아닌 행도 들어 있다.
- 분할: Protocol A(학습 3,970 / test 497).
- 대표 모델의 MAE:

| 채널 | 대표 모델 | MAE |
|---|---|---:|
| d1 | KRR | 1.70 |
| d2 | KRR | 1.25 |
| elst | SVR | 3.55 |
| Pauli | KRR | 6.45 |
| OI | KRR | 3.98 |
| disp | KRR | 0.470 |
| CPCM | NN-4 | 1.54 |
| CDS | KRR | 0.222 |

행과 분할이 달라 주 결과와 직접 비교하지 않는다.

## 6. 되살리기와 편향 점검

**Phase R** (`R_plan.md`, `R2_report.json`, `R2_by_reason.csv`, `R5_report.json`, `fig7_recovery`)

- **결과:** rev 4 G1 실패 400개 가운데 127개를 되살렸다. Espley 반응은 182개 중 63개다. gate는 rev 4와 같은 값을 그대로 썼고, R_plan에 없는 설정은 쓰지 않았다. gate 기록이 빠진 행도 없다.
- **처음 성공한 방법(127개):**

| 방법 | 행 수 |
|---|---:|
| 형성 결합 고정 사전 최적화(`constrained_pre`) | 49 |
| 참조 그래프 유지 제약 최적화(`constrained_ref`) | 28 |
| OptTS 재시작 | 20 |
| 다른 Coley 반응물 파일로 출발(`alt_start`) | 19 |
| 정확한 Hessian | 7 |
| 모드 방향 변위 | 4 |

- **남은 실패(273개):**

| 사유 | 행 수 |
|---|---:|
| 첫 허수 gate | 172 |
| OptTS 미수렴 | 38 |
| 외부 결합 | 35 |
| mode share | 28 |

- **partition만 실패했던 4개:** 라벨 A_idx를 적용하자 외부 결합 감사가 처음 돌았고, 4개 모두 그 감사에서 걸려 남았다. DFT TS 재탐색은 5,260개 모두 라벨 A_idx와 같았다.
- **되살린 TS와 DFT TS의 차이**(괄호 안은 원래 성공 TS):

| 지표 | 중앙값 | p90 |
|---|---:|---:|
| heavy RMSD (Å) | 0.40 (0.19) | 1.26 (0.51) |
| 짧은 형성 결합 \|Δd\| (Å) | 0.10 (0.07) | 0.37 (0.21) |

  되살린 TS는 원래 성공 TS보다 DFT TS에서 더 멀다.
- **R-4:** rxn 4994는 B1 gate 허용치 1e-5 Eh로 통과했다(`R4_rxn4994.json`). 다른 블록 값은 rev 5와 같다(B5만 6e-10 차이).
- **행 흐름(Espley 반응):** R_esp 3,510 → R_lab 3,509 → R_feat 3,390 → C 3,386(hygiene 4 제외).
  - C_all = 3,390, C_nocref = 3,375.
  - C에 들어간 되살린 행: 64개(1.9 %).
    - 내역: constrained_pre 21, alt_start 16, optts_restart 12, constrained_ref 11, exact_hess 2, displace_mode 1, gate_tol_user_decision 1.
  - Espley 쪽 feature는 3,510개 전부에 있다. 공통 반응이 줄어든 것은 우리 쪽 실패 때문이다(119개).

**편향 점검** (§2-2, `B_bias_check.csv`)

- **방법:**
  - 대상은 ESP-C 모델이다.
  - 빠진 행 D(우리 쪽 실패 119개 중 test에 든 50개 반응)와 C 행(1,397개 반응)의 Espley |오차|를 비교했다.
  - 반응마다 seed 평균을 낸 뒤, 반응 단위 bootstrap으로 95 % CI를 구했다.
- **대표 모델(XGB)의 D − C:**

| 타깃 | D − C (95 % CI) |
|---|---|
| d1 | +0.21 (−0.37, 0.84) |
| d2 | −0.15 (−0.53, 0.26) |
| Int | +0.36 (−0.48, 1.45) |
| ΔE‡ | +0.52 (−0.23, 1.32) |

  네 타깃 모두 CI가 0을 포함한다.
- **ΔE‡에서는 경향이 보인다:** 여섯 모델 모두 점추정이 양수이고, KRR(+0.78; 0.03, 1.55)과 SVR(+1.00; 0.17, 1.85)은 CI가 0 위에 있다. 즉 우리가 잃은 반응은 Espley 모델에게도 ΔE‡가 더 어려운 반응일 수 있다.
- **해석:** 주 비교는 C에서만 했으므로 이 결과는 C의 대표성에 대한 한계다.

**민감도** (`B_secondary_tests.csv`, Holm 없음)

| 조건 | 대표 모델 | d1 | d2 | Int | ΔE‡ |
|---|---|---|---|---|---|
| C_all(hygiene 행 포함, 다시 학습) | ESP-C XGB 대 OUR-O KRR | +0.84 (0.62, 1.06) | +0.44 (0.35, 0.52) | +0.97 (0.85, 1.09) | +1.08 (0.87, 1.29) |
| C_nocref(`constrained_ref` 11행을 빼고 다시 학습) | ESP-C XGB 대 OUR-O 대표 | +0.83 (0.59, 1.08) | +0.43 (0.36, 0.50) | +0.95 (0.88, 1.01) | +1.07 (0.79, 1.34) |

- C_nocref에서 OUR-O 대표 모델은 d1 NN-4, d2 KRR, Int SVR, ΔE‡ KRR이다.
- 같은 모델로 test에서 그 행만 빼고 다시 채점해도 결론은 같다.

## 7. 이탈 사항

1. **명세 §3-1의 서술과 그들의 코드가 다르다.**
   - 그들의 `hyp_tuning.py`는 KRR·SVR뿐 아니라 Ridge(alpha, tol)와 RF도 튜닝하고, NN은 keras_tuner Hyperband로 튜닝한다.
   - 공개된 `hps.pkl`·`ml_results.pkl`에는 RF가 없다.
   - 그들의 코드를 고치지 않았으므로 RF도 돌았지만, 6개 모델에 들지 않아 결과를 수집하지 않았다.
2. **keras_tuner를 1.4.7로 바꿨다**(사용자 결정, P-1 STOP 뒤).
   - 그들의 requirements인 1.4.5는 keras 3.3.3에서 Hyperband 체크포인트를 저장하지 못한다("filename must end in .weights.h5").
   - 나머지 고정 버전은 그대로다.
   - 1.4.5로 돌다 멈춘 첫 P-1 튜닝은 `P1/hyp_kt145_crashed`에 보관했다.
3. **NN은 비트 단위로 재현되지 않는다**(사용자 결정으로 수용, P-1).
   - 그들의 저장 예측과 비교하면 반응별 |차|의 중앙값이 0.18–1.34 kcal/mol이고, seed별 MAE 차이는 ±0.22 kcal/mol 이내다.
   - NN 하이퍼파라미터는 12개 키 중 9개가 다르다(keras_tuner 1.4.7 재실행).
   - sklearn 예측은 3.9e-11 안에서, sklearn 하이퍼파라미터는 그대로 같다.
   - 우리 환경 안에서는 결정적이다. P-2의 같은 노드 재실행과 NN 재구현이 50/50 완전히 일치했다.
4. **데이터셋에서 `q_barrier_dft`(ΔG‡)를 뺐다.** 우리 라벨에 ΔG‡가 없기 때문이다.
5. **`constrained_ref` 민감도는 다시 학습해서 했다**(C_nocref). 명세 R-2의 "빼 본다"를 그렇게 읽었고, 다시 채점한 값도 적었다.
6. **스레드와 병렬 실행이 명세 §12와 다르다.**
   - 모든 rev 6 학습은 BLAS 1, TensorFlow intra-op 2 / inter-op 1로 돌렸다.
   - 한 할당 안에서 학습 단위 여러 개를 별도 프로세스로 동시에 돌렸다. 명세 §12는 "병렬은 n_jobs로만"이라고 했다.
7. **R-1:** partition 재탐색은 더는 gate가 아니다. partition만 실패했던 4행은 외부 결합 감사에서 걸렸다.
8. **P-2 튜닝 비교와 우리 파이프라인 학습이 동시에 돌았다**(사전 등록 §8 이탈 9).
   - P-2 튜닝 비교가 끝나기 전에 우리 파이프라인 학습(그들의 코드를 쓰지 않음)을 시작했다.
   - Espley 프로토콜 arm은 P-2 통과 뒤에만 돌렸다.
   - P-2 튜닝 비교의 판정 방식도 실행 뒤에 바꿨다.
     - 처음에는 `train_mae`가 정확히 같아야 했다. xgboost import에 따른 BLAS 차이(1e-15)로 STOP이 났다.
     - 명세 P-2 허용치인 1e-8로 바꿔 다시 비교했다(최대 차이 1.0e-14). 튜닝값은 모두 같았다.
9. **그림 글꼴:** Arial이 없어 DejaVu Sans를 썼다(`figures/figures.json`).
10. **HPC 규칙 위반 (자진 보고).** 셋 다 계산은 없었다.
    - 로그인 노드에서 빈 `python3 -c 1`을 한 번 실행했다.
    - 로그인 노드에서 `python3 --version`을 한 번 실행했다.
    - 로그인 노드 명령 하나에 `sleep 2`가 들어갔다.

## 8. 한계

- **남은 실패와 편향.** 남은 실패 273행(Espley 반응 119행)에서는 우리 모델이 feature를 만들 수 없다.
  - 편향 점검에서 ΔE‡는 빠진 행이 Espley 모델에게도 더 어렵다는 경향을 보였다(§6).
  - 결과는 C(3,386)에 대한 것이다.
- **Espley 프로토콜의 seed 23 튜닝 누출(Espley에 유리).** seed 23 train 행이 다른 seed의 test와 겹친다.
  - test 334–341행 가운데 260–284행이 겹친다(`S.json` test_row_audit).
  - 우리 파이프라인은 겹침이 0이다.
- **배포 성능이 아니다.** 양쪽 모두 구조 출발점이 Coley DFT TS다(Espley AM1 TS, 우리 G1 xTB TS). SMILES에서 출발하는 배포 성능은 이 실험이 재지 않았다.
- **NN 재현 정도.** §7-3의 내용이 그대로 한계다.
- **grid 경계 일치**(`B_summary.csv` edge_hits). grid 끝에 걸린 값이 많다.

  | 방식 | 모델 | grid 끝에 걸린 값 |
  |---|---|---|
  | Espley 프로토콜 | Ridge | alpha와 tol이 모든 타깃에서 끝(ESP-C는 alpha 1.0 상한, OUR-C는 alpha 0.01 하한, tol은 둘 다 0.1 하한) |
  | Espley 프로토콜 | KRR | alpha 1.0 상한(ESP-C의 d1·d2·ΔE‡, OUR-C의 d1) |
  | 우리 파이프라인 | SVR | 15개 튜닝값 중 8–10개(OUR-O; 주로 epsilon 하한, gamma 1e-3) |
  | 두 방식 모두 | XGB | 값이 둘인 grid라 늘 끝 |

  grid를 넓혔다면 일부 모델의 MAE가 달라졌을 수 있다. 대표 모델 선택과 결론은 사전 등록한 grid 안에서의 것이다.
- **`constrained_ref` 행.** 참조 결합 그래프를 제약으로 유지한 행은 C에 11개 있다. 빼고 다시 학습해도 결론은 같았다(§6).
- **그들 방식의 특이점.** 대표 모델 규칙의 Espley 쪽 검증 MAE 가운데 KRR 값은 poly 커널로 튜닝한 점수이고, 실제 학습은 rbf로 했다(그들 코드 그대로).
  - 그들의 `y_train_CV` 규칙으로 골라도 대표 모델은 같다(XGB).
- **라벨.** Σ6채널과 e_bond의 차이는 최대 0.105 kcal/mol이다(ORCA EDA 표 반올림, `S_label_checks.json`). Int ≠ Σ6이고 c_ghost가 그 차이다(평균 0.078, sd 0.667).

## 9. 산출물 (`results_rev6/`)

- **사양·계획:**
  - `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md`
  - `R_plan.md`
  - `PREREG_REV6.md`, `prereg_rev6.json`
- **Phase R:** `R_inventory.*`, `R_rows.csv`, `R2_attempts.csv`, `R2_by_reason.csv`, `R2_report.json`, `R3_*.json`, `R4_rxn4994.json`, `R5_report.json`, `R_rows_final.csv`
- **Phase P:** `P1_*`, `P2_*`, `P3_role_check.json`, `P3_swap.csv`, `P4_their_code_diff.patch`, `P4_patched_sha256.json`
- **Phase S:** `S_rows.csv`, `S_splits.csv`, `S_splits_appendixA.csv`, `S_label_checks.json`, `S.json`
- **Phase B:** `B_per_seed.csv`, `B_summary.csv`, `B_primary_tests.csv`, `B_secondary_tests.csv`, `B_representative.json`, `B_bias_check.csv`, `B_sigma_*`, `B_predictions.parquet`, `B.json`
- **Phase A:** `A_per_seed.csv`, `A_summary.csv`, `A_representative.json`, `A_predictions.parquet`, `A_appendix_*`
- **그림:** `figures/fig1_primary` … `fig7_recovery`(PDF, PNG 300 dpi, 그린 값 CSV)

## 10. 재현

- **코드:** `analysis/espley_xtb_repro/`
  - 공통 상수: `rev6_common.py`
  - Phase R: `r6_recover.py`, `r6_partition.py`, `r6_features.py`
  - Phase P: `r6_espley.py`
  - Phase S: `r6_split.py`
  - Phase B/A: `r6_nn.py`, `r6_train.py`, `r6_espc.py`
  - 집계·그림: `r6_analyze.py`, `r6_figures.py`
- **sbatch 스크립트:** 모두 `--time=48:00:00`이고, 다시 제출하면 끝난 것은 건너뛴다.
  - `r6_R.sh`
  - `r6_F.sh`
  - `r6_P.sh`
  - `r6_run.sh`(MODE = split / units / arm / p2_* / analyze_B / analyze_A / figures)
- **환경:**
  - Espley 고정 환경 `/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley`
    - python 3.11, tensorflow 2.16.1, keras 3.3.3, keras_tuner 1.4.7, scikit-learn 1.3.1, numpy 1.26.0, pandas 2.1.1, xgboost 3.2.0
    - `r6_espley_env.sh`, `r6_espley_env_kt147.sh`
  - Phase R·feature: reactot + rev 5 venv(xtb 6.7.1 edcfbbe, tblite 0.7.0, ORCA 6.1.1)
- **실행 순서:**
  1. `r6_R.sh`(inventory → run → summary)
  2. `r6_F.sh`
  3. `r6_P.sh` p1_ml / p1_hyp / p1_compare / p3
  4. `r6_run.sh` p2_*
  5. `r6_run.sh` split
  6. PREREG를 commit
  7. `r6_run.sh` units(B, A, Call, Nocref, AppA)
  8. `r6_run.sh` arm(ESP-C, OUR-C, ESP-C_all, ESP-C_nocref)
  9. analyze_A, analyze_B
  10. figures
