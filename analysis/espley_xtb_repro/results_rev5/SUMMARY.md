# espley_xtb_repro — rev 5 결과 (2026-10-01): G1 기하 위 확장 feature 블록, lockbox 평가, Espley 비교 그림
G0(DFT oracle geometry) 결과는 2026-09-30 사용자 결정으로 폐기. git ≤ `740ba89`에만 남음.

rev 5는 rev 4의 G1(GFN2-xTB/ALPB(water) 기하) 위에서 세 가지를 했다. 첫째, 위 문장의 폐기를 실행했다(Phase A): 해당 결과
파일·코드 경로·scratch를 지우고, rev 4 분석을 G1과 Espley 행만으로 다시 만들어 `740ba895`와 대조했다. 둘째, rev 4에서 절대 MAE가
가장 컸던 Pauli·OI·elst(NMAE로는 CPCM 0.399, CDS 0.374가 더 크다; `results_rev4/rev4_headline.csv`)를 겨냥해 G1 구조(TS, rel1, rel2,
G1 TS에서 자른 조각)에서만 계산하는 확장 feature 블록 B1–B6(109열)을 만들고(Phase B), 15 % lockbox를 떼어 둔 뒤 dev 행에서만 전진
블록 선별을 해 **EXT_SEL = BASE(ESPLEY73) + B1 + B4 (133 feature)**를 사전 등록으로 고정했다(Phase C). 셋째, lockbox로 최종 평가하고
(D-1, 헤드라인), Espley 2024와 lockbox 정면 비교(D-3(b)), Espley 분할 비교(D-3(a)), 학습 곡선(D-4)을 돌리고, 논문용 그림 8개를
만들었다(Phase E). 새 feature에는 DFT 에너지나 라벨 값을 쓰지 않았다. 다만 G1은 라벨의 DFT TS·참조에서 출발하므로 배포 성능이 아니다(§7).

- 사양 [`REV5_FEATURES_FIGURES.md`](../../../docs/specs/REV5_FEATURES_FIGURES.md). 사전 등록 [`PREREG_REV5a.md`](PREREG_REV5a.md)
  (commit `358b2d10`, 2026-10-01 11:45:28, 블록 선별과 모든 rev 5 학습 전)와 [`PREREG_REV5b.md`](PREREG_REV5b.md) + `prereg_rev5b.json`
  (commit `9c823fff`, 13:25:26, Phase D 전). 상수 원본은 `rev5_common.py`. 브랜치 `espley-rev4`(rev 4 결과 `740ba895`, Phase A `18862409`).
- 타깃: repo 루트 `labels_all.json`(SMD(water), 방법 ① 조립). 행: `rows_rev5.csv` **4,838** = `rows_rev4.csv`(4,839) ∩ 확장 feature ok.
- 엔진: xtb 6.7.1 (edcfbbe) — xtb가 출력하는 값 전부. tblite 0.7.0 — B1 궤도 overlap만.
- 수치는 이 폴더(`results_rev5/`)의 파일에서 옮겼다. 폴더 밖 출처는 그 자리에 경로를 적었다. MAE는 kcal/mol, 소수 둘째 자리
  (disp와 CDS는 셋째 자리), NMAE·r²는 셋째 자리, p는 유효숫자 둘이다. 예외: 선별 기준(§3-1)은 넷째 자리, dev CV 차이 표(§3-3)는
  모든 타깃을 셋째 자리, 배수·비율은 셋째 자리로 쓴다.
- D-2(Protocol A)와 하류 분석은 §5에 있다.

| 블록 | 내용 | 열 | 엔진 |
|---|---|---:|---|
| B1 | 조각 MO overlap 𝒮 = C_Aᵀ S_AB C_B: occ–occ Σ𝒮², occ→virt Σ𝒮²/Δε(양방향, 합), HOMO/LUMO 쌍의 \|𝒮\|·Δε·𝒮²/Δε, 3×3 창 최댓값, 조각·반응물 HOMO/LUMO | 21 | tblite GFN2, ALPB |
| B2 | 조각 간 접촉: Σexp(−r/ρ)(쌍 종류 3 × ρ 3), vdW 침투 합·쌍 수, 형성 결합 밖 최단 접촉 2개, heavy–heavy 거리 히스토그램 8칸 | 23 | 기하 |
| B3 | 조각 밀도 고정 다극 정전기(q–μ, μ–μ, q–Θ)와 침투 대리값 4개 | 7 | xtb `--json` |
| B4 | μ·η·ω(`--vipea`, 구조 4개), ΔN 2개, Fukui f⁺/f⁻/f⁰(반응 원자 5개), D4 α(원자 5개, 분자 4개), GEDT | 39 | xtb |
| B5 | B 조각 평행이동 스캔(δ = ±0.05, ±0.10 Å): E_int, b_pauli, b_oi, b_elst의 기울기·곡률·δ = ∓0.10 값 | 16 | xtb |
| B6 | TS 성격: `nu_imag`, `mode_share`, `mode_async` (생성물 feature 3개는 사용자 결정으로 제외, §6-3) | 3 | G1 `ts.hess`, `result.json` |

---

## 1. 헤드라인 — D-1 lockbox · EXT_SEL · KRR(RBF) (사전 고정)

dev 4,112 rxn 전체로 학습하고(GridSearchCV, `KFold(5, shuffle=True, random_state=20261001)`, X·y 표준화, rev 4 grid), lockbox 726 rxn으로
한 번 채점했다. 95 % CI는 lockbox 반응 단위 bootstrap 10,000회의 percentile 구간이고, 모든 arm·모델·타깃이 같은 재표본 인덱스를 쓴다.
짝지은 차이는 arm − BASE이다(음수 = arm의 MAE가 낮음). BASE = ESPLEY73(73), EXT_SEL = BASE + B1 + B4(133), EXT_ALL = BASE + B1…B6
(182, 선별과 무관한 사전 고정 arm). 출처: `D1_lockbox.csv`, `D1_lockbox_paired.csv`. 그림: `figures/fig8_channel_lockbox`.

| 타깃 | **EXT_SEL MAE [95 % CI]** | NMAE | r² | BASE MAE | **EXT_SEL − BASE [95 % CI]** (상대) | EXT_ALL MAE [95 % CI] | EXT_ALL − BASE [95 % CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.73** [1.63, 1.84] | 0.249 | 0.936 | 1.86 | **−0.13** [−0.22, −0.05] (−7.1 %) | 1.71 [1.60, 1.82] | −0.16 [−0.25, −0.07] |
| d1 (dipole strain) | **1.61** [1.50, 1.72] | 0.274 | 0.912 | 1.73 | **−0.12** [−0.20, −0.05] (−7.1 %) | 1.63 [1.53, 1.74] | −0.10 [−0.18, −0.02] |
| d2 (dipolarophile strain) | **1.22** [1.14, 1.31] | 0.266 | 0.917 | 1.31 | **−0.09** [−0.15, −0.04] (−7.1 %) | 1.22 [1.14, 1.31] | −0.09 [−0.15, −0.03] |
| elst | **3.78** [3.50, 4.07] | 0.271 | 0.907 | 4.20 | **−0.42** [−0.57, −0.27] (−10.0 %) | 3.60 [3.32, 3.89] | −0.60 [−0.78, −0.42] |
| Pauli | **6.66** [6.15, 7.22] | 0.257 | 0.904 | 7.53 | **−0.87** [−1.21, −0.54] (−11.5 %) | 6.49 [6.01, 7.01] | −1.04 [−1.39, −0.69] |
| OI | **3.99** [3.69, 4.30] | 0.259 | 0.908 | 4.51 | **−0.53** [−0.71, −0.34] (−11.6 %) | 3.86 [3.57, 4.17] | −0.66 [−0.87, −0.45] |
| disp | 0.471 [0.437, 0.507] | 0.127 | 0.980 | 0.484 | −0.013 [−0.028, +0.003] (−2.6 %) | 0.471 [0.437, 0.507] | −0.013 [−0.028, +0.002] |
| CPCM | 1.61 [1.49, 1.73] | 0.384 | 0.867 | 1.58 | +0.03 [−0.03, +0.08] (+1.6 %) | 1.54 [1.42, 1.67] | −0.04 [−0.11, +0.03] |
| CDS | 0.211 [0.197, 0.225] | 0.299 | 0.905 | 0.249 | −0.038 [−0.051, −0.026] (−15.4 %) | 0.213 [0.199, 0.227] | −0.036 [−0.050, −0.023] |

### 관찰

1. **겨냥한 세 채널(elst, Pauli, OI)에서 절대 감소폭(kcal/mol)이 가장 크다.**
   - EXT_SEL − BASE는 −0.42 / −0.87 / −0.53이고(상대 −10.0 / −11.5 / −11.6 %), 세 CI 모두 0을 포함하지 않는다.
     bootstrap 10,000회 가운데 차이가 0 이상인 재표본의 비율은 세 채널 모두 0이다(`frac_boot_diff_ge0`).
   - NMAE: elst 0.301 → 0.271, Pauli 0.291 → 0.257, OI 0.293 → 0.259.
   - 상대 감소와 NMAE 감소는 CDS가 가장 크다(−15.4 %, NMAE 0.353 → 0.299). 세 채널은 MAE 규모 자체가 커서 절대 폭이 크다.
2. **ΔE‡, d1, d2, CDS도 CI가 0 아래다.** ΔE‡ −0.13, d1 −0.12, d2 −0.09(셋 모두 −7.1 %), CDS −0.038(−15.4 %).
3. **disp와 CPCM은 개선이 확인되지 않았다(KRR에서 CI가 0을 포함).**
   - disp: −0.013 [−0.028, +0.003]. CI가 0을 포함한다. 부록의 SVR·XGB에서는 CI가 0 아래다(관찰 5).
   - CPCM: EXT_SEL − BASE는 +0.03 [−0.03, +0.08]로 점추정은 EXT_SEL이 높고 CI가 0을 포함한다. EXT_ALL − BASE도 −0.04 [−0.11, +0.03]로
     CI가 0을 포함한다.
   - 같은 용매화 항이라도 CDS는 −15.4 %로 줄었고 CPCM 감소는 확인되지 않았다. 원인은 시험하지 않았다.
4. **EXT_ALL의 점추정은 elst·Pauli·OI에서 EXT_SEL보다 낮다**(3.60 / 6.49 / 3.86 대 3.78 / 6.66 / 3.99). dev CV에서도 같은 방향이었다(§3).
   그러나 EXT_ALL − EXT_SEL의 짝지은 CI는 사전 등록 산출물에 없다. 그래서 두 arm의 차이는 판정하지 않는다.
   EXT_SEL은 사전 등록된 선별 규칙(단계마다 상대 감소 ≥ 2 %, 바깥 fold ≥ 4/5)의 결과이고, EXT_ALL은 그 규칙과 무관한 사전 고정 arm이다.
5. **부록 모델** (`D1_lockbox.csv`, `D1_lockbox_paired.csv`). EXT_SEL − BASE [95 % CI]:

   | 타깃 | Ridge | SVR(RBF) | XGB |
   |---|---:|---:|---:|
   | ΔE‡ | −0.26 [−0.35, −0.17] | −0.22 [−0.30, −0.13] | −0.03 [−0.11, +0.04] |
   | d1 | −0.26 [−0.34, −0.17] | −0.15 [−0.23, −0.07] | −0.08 [−0.14, −0.01] |
   | d2 | −0.16 [−0.22, −0.10] | −0.13 [−0.20, −0.07] | −0.07 [−0.12, −0.01] |
   | elst | −0.46 [−0.66, −0.26] | −0.37 [−0.55, −0.19] | −0.18 [−0.39, +0.02] |
   | Pauli | −1.12 [−1.52, −0.73] | −0.97 [−1.36, −0.58] | −0.84 [−1.21, −0.47] |
   | OI | −0.73 [−0.96, −0.51] | −0.62 [−0.85, −0.39] | −0.19 [−0.41, +0.03] |
   | disp | −0.007 [−0.020, +0.006] | −0.023 [−0.037, −0.008] | −0.018 [−0.035, −0.002] |
   | CPCM | +0.002 [−0.056, +0.058] | +0.009 [−0.048, +0.068] | −0.012 [−0.071, +0.048] |
   | CDS | −0.060 [−0.078, −0.042] | −0.042 [−0.055, −0.029] | −0.034 [−0.046, −0.021] |

   - 겨냥한 세 채널 가운데 Pauli만 네 모델 모두에서 CI가 0 아래다(d1, d2, CDS도 네 모델 모두 0 아래). XGB에서는 ΔE‡, elst, OI, CPCM의
     CI가 0을 포함한다.
   - **KRR 사전 고정에 대한 참고.** EXT_SEL에서 SVR의 lockbox MAE가 KRR보다 점추정으로 낮은 타깃은 9개 가운데 6개다: ΔE‡ 1.64 대 1.73,
     Pauli 6.52 대 6.66, OI 3.91 대 3.99, d2 1.19 대 1.22, elst 3.77 대 3.78, disp 0.463 대 0.471(KRR이 낮은 것은 d1, CPCM, CDS).
     SVR − KRR의 짝지은 CI는 계산하지 않았으므로 판정하지 않는다. SVR은 27개 단위 모두에서 튜닝값 하나 이상이 grid 경계에 걸렸다(§7).
     헤드라인은 사전 등록대로 KRR이다.

---

## 2. Espley 2024 대비

공통 조건: 타깃은 Espley의 DFT 값이다(Interaction은 Espley의 부호, 양수 = 안정화). Dipole / Dipolarophile 변형 타깃은 Espley의
`distortion_energy_1/2`를 rev 4 swap 벡터로 역할 기준(dipole / dipolarophile)에 다시 배정한 값이다. Espley 쪽 feature는 그들의 AM1 46개이고,
역할 타깃에는 AM1 변형 feature를 같은 swap 벡터로 재배열했다. "Espley 행" = Espley ds3 ∩ 라벨 ∩ `rows_rev5` = **3,322 rxn**(`D3a.json` `rows`).
두 쪽은 기하(AM1 대 G1, G1은 DFT TS 출발), feature, 모델, 튜닝이 모두 다르고 G1은 입력 대응 arm이 아니다. 아래 배수는 파이프라인 전체의
차이이지 feature 단독의 효과가 아니다(§7).

### 2-1. D-3(b) lockbox 정면 비교 — EXT_SEL의 1차 Espley 비교 (블록 선별 편향 없음)

- 행: test = lockbox ∩ Espley 행 **502 rxn**, train = dev ∩ Espley 행 **2,820 rxn**. 두 쪽이 같은 train 행으로 학습한다.
- Espley 쪽: 그들의 프로토콜. ESI Table S3 grid, 이 train에서 `GridSearchCV(cv=5)`(그들 코드처럼 shuffle 없음), X만 StandardScaler,
  튜닝 열 47 / 적합 열 46. Interaction, ΔE‡, ΔG‡도 저장 예측이 아니라 이 train에서 다시 학습했다.
  주 비교는 그들이 발표한 최고 모델인 SVR이고, KRR(poly로 튜닝하고 rbf로 실행 — 그들 코드 그대로)은 부록이다.
- 우리 쪽: EXT_SEL KRR, nested GridSearchCV(`KFold(5, shuffle=True, random_state=20261001)`).
- CI: test 502 반응 단위 짝지은 bootstrap 10,000회(rng 20261001).
- 출처 `D3b_summary.csv`, `D3b_paired.csv`. 그림 `figures/fig2_espley_lockbox`.

| 타깃 | Espley SVR (그들의 프로토콜) | **EXT_SEL · KRR** | Espley − EXT_SEL [95 % CI] | Espley ÷ EXT_SEL | r² (Espley / EXT_SEL) |
|---|---:|---:|---:|---:|---:|
| Dipole 변형 (역할) | 2.41 [2.23, 2.59] | **1.58** [1.46, 1.69] | 0.84 [0.64, 1.03] | 1.530 | 0.798 / 0.917 |
| Dipolarophile 변형 (역할) | 1.88 [1.73, 2.03] | **1.11** [1.02, 1.20] | 0.77 [0.61, 0.93] | 1.690 | 0.789 / 0.924 |
| Interaction | 2.26 [2.09, 2.43] | **1.16** [1.07, 1.26] | 1.10 [0.92, 1.28] | 1.945 | 0.721 / 0.922 |
| ΔE‡ | 2.98 [2.76, 3.19] | **1.62** [1.50, 1.74] | 1.36 [1.13, 1.57] | 1.838 | 0.790 / 0.938 |
| ΔG‡ | 2.88 [2.67, 3.09] | **1.66** [1.54, 1.79] | 1.21 [0.99, 1.43] | 1.729 | 0.814 / 0.939 |

- **5개 타깃 모두 CI가 0보다 위다.** bootstrap 10,000회 가운데 차이가 0 이하인 재표본은 없다(`frac_boot_diff_le0` = 0). 배수는 1.530 – 1.945다.
- **부록: Espley KRR(그들의 프로토콜).** MAE 2.84 / 2.14 / 2.53 / 3.23 / 3.35. EXT_SEL과의 차이 1.26 [1.04, 1.49] / 1.03 [0.85, 1.22] /
  1.37 [1.18, 1.56] / 1.61 [1.36, 1.86] / 1.68 [1.41, 1.97]. 타깃 순서는 dipole / dipolarophile / Interaction / ΔE‡ / ΔG‡다.
- EXT_SEL은 lockbox를 쓰기 전에 고정됐다(PREREG_REV5b). 그래서 이 비교에는 블록 선별 편향이 없다. 행 필터(G1 성공)와 기하 출발점의 차이는
  남는다(§7). 겨냥 채널은 lockbox 행이 포함된 rev 4 결과를 보고 정했다(§6-5).

### 2-2. D-3(a) Espley 분할 (rev 4 방식)

- **행.** Espley ds3 3,510 → 라벨 있음 3,509 → rev 4 채점 행 3,327 → ∩ `rows_rev5` = 3,322(`D3a.json` `rows`).
  - 빠진 5건: 336, 634, 1479, 4111은 `rows_rev4.csv`에 없다(rev 4 hygiene). 이 4건은 rev 4 Espley 비교의 3,327행에는 들어 있었다.
    4994는 확장 feature 실패로 `rows_rev5`에 없다.
- **분할.** Espley 방식(test = 20 % hold-out의 첫 절반), seed 22/23/14/1/2. seed당 채점 test는 330 – 338행(평균 333)이다.
  - 우리 모델은 채점 행의 train 부분(2,650 – 2,657행, 평균 2,653.8)으로 학습한다.
  - Espley 저장 모델과 Espley 쪽 재학습은 그들의 train 분할 전체(2,807 – 2,808행)로 학습한다. 학습 행 수는 Espley 쪽이 많다.
  - 모든 쪽을 seed마다 같은 test 행으로 채점했다(`checks.test_rows_identical_across_sides` = true).
- **우리 쪽.** ESPLEY46(rev 4 사전 고정 주 비교의 재실행), ESPLEY73, EXT_SEL. 모두 KRR, seed마다 nested.
- **Espley 쪽.**
  - Espley SVR: Interaction / ΔE‡ / ΔG‡는 그들이 저장한 SVR 예측이고, 역할 d1/d2는 그들의 프로토콜 SVR 재학습이다.
  - 후보 전체: 저장 Ridge / KRR / SVR / 2·4-layer NN(Interaction, ΔE‡, ΔG‡), 그들의 프로토콜 SVR / KRR(역할 d1/d2), Espley 46 feature ×
    우리 파이프라인 Ridge / KRR / SVR / XGB(5개 모두)(`D3a.json` `design.espley_candidates`). rev 4의 빈칸(우리 파이프라인 XGB의 Interaction,
    ΔE‡, ΔG‡)을 채웠다.
  - **가장 강한 Espley 쪽 모델은 5개 타깃 모두 Espley 46 feature × 우리 파이프라인 XGB다**(`D3a.json` `strongest`).
    이 test 행에서 고른 best-of이므로 Espley에 유리한 선택이다.
- **검정.** seed별 MAE 차이(Espley − 우리)에 Nadeau–Bengio 보정 t-검정. J = 5, n_test / n_train = 333 / 2,653.8(우리 학습 행 평균).
  양측 p와 95 % CI를 보고한다.
- **EXT_SEL 주의.** EXT_SEL의 블록은 dev 행에서 골랐고, dev 행은 이 test 행과 겹친다. EXT_SEL의 블록 선별 편향이 없는 Espley 비교는 §2-1이다.
- 출처 `D3a_summary.csv`, `D3a_tests.csv`, `D3a_per_seed.csv`.

seed 평균 test MAE:

| 타깃 | Espley SVR | 우리 ESPLEY46 | 우리 ESPLEY73 | **우리 EXT_SEL** | Espley 최강 (XGB, Espley 46) |
|---|---:|---:|---:|---:|---:|
| Dipole 변형 (역할) | 2.56 | 1.67 | 1.63 | **1.57** | 2.31 |
| Dipolarophile 변형 (역할) | 1.91 | 1.44 | 1.30 | **1.23** | 1.63 |
| Interaction | 2.44 | 1.63 | 1.35 | **1.26** | 2.23 |
| ΔE‡ | 3.04 | 2.01 | 1.82 | **1.66** | 2.81 |
| ΔG‡ | 2.96 | 2.01 | 1.85 | **1.72** | 2.75 |

짝지은 차이 (Espley − 우리) [95 % CI]; Nadeau–Bengio t; p. **대 Espley SVR:**

| 타깃 | ESPLEY46 (rev 4 주 비교) | ESPLEY73 | EXT_SEL |
|---|---|---|---|
| Dipole | 0.89 [0.66, 1.11]; t 10.87; p 4.1e-4 | 0.93 [0.71, 1.15]; 11.73; 3.0e-4 | 0.99 [0.76, 1.22]; 11.85; 2.9e-4 |
| Dipolarophile | 0.47 [0.30, 0.63]; 7.90; 1.4e-3 | 0.61 [0.47, 0.74]; 12.36; 2.5e-4 | 0.68 [0.58, 0.79]; 17.97; 5.6e-5 |
| Interaction | 0.81 [0.65, 0.97]; 14.08; 1.5e-4 | 1.09 [0.92, 1.26]; 18.15; 5.4e-5 | 1.18 [1.02, 1.34]; 20.75; 3.2e-5 |
| ΔE‡ | 1.03 [0.79, 1.28]; 11.64; 3.1e-4 | 1.22 [1.04, 1.40]; 18.59; 4.9e-5 | 1.38 [1.15, 1.60]; 16.87; 7.2e-5 |
| ΔG‡ | 0.95 [0.78, 1.12]; 15.36; 1.0e-4 | 1.11 [0.98, 1.23]; 25.05; 1.5e-5 | 1.24 [1.07, 1.40]; 21.23; 2.9e-5 |

**대 Espley 쪽 최강 모델(XGB):**

| 타깃 | ESPLEY46 | ESPLEY73 | EXT_SEL |
|---|---|---|---|
| Dipole | 0.64 [0.36, 0.91]; t 6.42; p 3.0e-3 | 0.68 [0.38, 0.97]; 6.35; 3.1e-3 | 0.74 [0.44, 1.04]; 6.87; 2.3e-3 |
| Dipolarophile | 0.19 [0.14, 0.24]; 11.16; 3.7e-4 | 0.33 [0.24, 0.42]; 10.35; 4.9e-4 | 0.41 [0.31, 0.50]; 12.22; 2.6e-4 |
| Interaction | 0.59 [0.41, 0.78]; 8.86; 9.0e-4 | 0.88 [0.76, 0.99]; 21.10; 3.0e-5 | 0.96 [0.89, 1.04]; 35.58; 3.7e-6 |
| ΔE‡ | 0.80 [0.71, 0.88]; 25.67; 1.4e-5 | 0.99 [0.80, 1.17]; 15.03; 1.1e-4 | 1.14 [0.98, 1.31]; 19.09; 4.4e-5 |
| ΔG‡ | 0.74 [0.56, 0.91]; 11.46; 3.3e-4 | 0.90 [0.64, 1.15]; 9.75; 6.2e-4 | 1.02 [0.81, 1.24]; 13.39; 1.8e-4 |

1. **rev 4 사전 고정 주 비교(ESPLEY46 KRR 대 Espley SVR)가 재현된다.**
   - 5개 타깃 모두, seed 5/5에서 ESPLEY46이 낮다. p ≤ 1.4e-3(최대 = dipolarophile). 최강 Espley 모델 대비 p ≤ 3.0e-3(최대 = dipole).
   - ESPLEY46의 MAE는 rev 4 값(`D3a.json` `rev4_reference`: 1.6727 / 1.4407 / 1.6322 / 2.0053 / 2.0110)과 소수 둘째 자리까지 같다.
     행이 5개 줄었고(3,327 → 3,322), 나머지 설정은 같다.
2. **EXT_SEL은 Espley SVR보다 0.68 – 1.38 낮고(p ≤ 2.9e-4), 최강 Espley 모델보다 0.41 – 1.14 낮다(p ≤ 2.3e-3).**
   표의 30개 비교(arm 3 × 타깃 5 × 비교 대상 2)에서 모두 seed 5/5로 우리 쪽이 낮다(`seeds_ours_lower` = 5).
3. **우리 쪽은 5개 타깃 모두 ESPLEY46 → ESPLEY73 → EXT_SEL 순으로 낮아진다.** 예: ΔE‡ 2.01 → 1.82 → 1.66, Interaction 1.63 → 1.35 → 1.26.
4. **배수(Espley SVR ÷ EXT_SEL).** seed 평균 MAE의 비는 1.630 / 1.557 / 1.931 / 1.827 / 1.716이다(`mae_ratio_of_means`).
   fig1은 seed별 비의 평균 1.636 / 1.559 / 1.934 / 1.830 / 1.717을 단다(`mae_ratio_mean`).
5. **그림.**
   - `fig1_espley_mae`: 계열 4개 × 타깃 5개 막대, 배수와 p(EXT_SEL 대 Espley SVR) 주석.
   - `fig3_parity`: Espley SVR / EXT_SEL × 타깃 5개, 1,370 rxn(반응별로 test였던 seed의 예측 평균).
   - `fig4_error_ecdf`: |오차| 누적분포(타깃마다 seed별 오차 1,665개). 1 kcal/mol 안의 비율, Espley SVR → EXT_SEL:
     dipole 0.277 → 0.432, dipolarophile 0.371 → 0.532, Interaction 0.289 → 0.527, ΔE‡ 0.244 → 0.404, ΔG‡ 0.241 → 0.385
     (`figures/fig4_error_ecdf.csv`의 `within_1` 행).
   - `fig5_seed_pairs`: 25개 seed 쌍(타깃 5 × seed 5)이 모두 Espley → EXT_SEL로 내려간다(`figures/fig5_seed_pairs.csv`).

### 2-3. D-4 학습 곡선 (Espley `ml_analysis.py` 방식)

- 분할은 Espley 분할, seed 5개다. test 행은 D-3(a)의 채점 행으로 고정했다.
- 부분집합: seed별 train 행(채점 행 ∩ Espley train 분할)을 rxn_id로 인덱싱한다. 그들의 `ml_analysis.py`가 샘플링하는 X_train 순서(분할 후 순서)로
  두고 `sample(frac=f, random_state=seed)`로 뽑는다(f = 0.1 … 1.0). 두 쪽이 같은 부분집합을 쓴다.
- 하이퍼파라미터는 전체 train 튜닝값으로 고정했다.
  - Espley SVR: 역할 타깃은 그들의 프로토콜 재튜닝 값(rev 4 재사용), Interaction / ΔE‡ / ΔG‡는 `hps.pkl` 값.
  - EXT_SEL KRR: D-3(a)의 seed별 튜닝값.
- 기준선(fig6 점선)은 f = 1.0일 때 Espley SVR의 MAE다. 이 SVR은 곡선과 같은 학습 행(평균 2,653.8)으로 학습했다(PREREG_REV5a §5).
- EXT_SEL 블록은 이 test 행과 겹치는 dev 행에서 골랐다(§2-2와 같은 주의). 두 쪽 모두 하이퍼파라미터는 전체 train 튜닝값이어서, 작은 f의 MAE에는
  전체 train의 튜닝 정보가 들어 있다.
- 출처 `D4_learning_curves_summary.csv`, `D4_learning_curves_reach.csv`, `D4_learning_curves_sensitivity.csv`. 그림 `figures/fig6_learning_curves`.

| 타깃 | Espley SVR f = 0.1 | Espley SVR f = 1.0 (기준선) | EXT_SEL f = 0.1 | EXT_SEL f = 1.0 | 기준선 도달 행 수 |
|---|---:|---:|---:|---:|---:|
| Dipole 변형 (역할) | 3.49 | 2.58 | 2.24 | 1.57 | ≤ 265.4 |
| Dipolarophile 변형 (역할) | 2.67 | 1.92 | 1.73 | 1.23 | ≤ 265.4 |
| Interaction | 3.44 | 2.45 | 1.84 | 1.26 | ≤ 265.4 |
| ΔE‡ | 4.13 | 3.05 | 2.35 | 1.66 | ≤ 265.4 |
| ΔG‡ | 4.05 | 2.97 | 2.39 | 1.72 | ≤ 265.4 |

학습 행 수는 f = 0.1에서 평균 265.4(265 – 266), f = 1.0에서 평균 2,653.8이다.

1. **EXT_SEL은 첫 격자점(f = 0.1)에서 이미 Espley SVR의 f = 1.0 MAE보다 낮다(5개 타깃 모두).**
   - 그래서 도달 행 수는 "≤ 265.4"로만 정해진다(`reached` = True, `at_first_point` = True). f < 0.1은 돌리지 않았으므로 더 정확한 값은 모른다.
   - D-3(a)의 Espley SVR(그들의 train 분할 전체로 학습: 2.56 / 1.91 / 2.44 / 3.04 / 2.96)을 기준선으로 잡아도 결과는 같다(`D4_learning_curves_reach.csv`).
2. **부분집합 순서에 대한 민감도.** ds3 행 순서로 뽑으면(`D4_learning_curves_sensitivity.csv`, `D4_learning_curves.json` `sensitivity_frame_order`)
   f = 0.1에서 EXT_SEL은 2.18 / 1.74 / 1.78 / 2.34 / 2.40, Espley SVR은 3.51 / 2.64 / 3.22 / 4.24 / 4.14다.
   f = 1.0은 두 순서가 같은 행이므로 값이 소수 둘째 자리까지 같다(EXT_SEL은 약 1e-12 안, Espley SVR은 SVR이 행 순서에 의존해 최대 4.7e-6 차이).
   도달 판정(첫 격자점에서 도달)은 순서에 따라 바뀌지 않는다.
3. **확인.** f = 1.0의 EXT_SEL MAE는 D-3(a)와 7.2e-12 안에서 같다. 그들의 `ml_analysis.py` 학습 곡선을 그들의 행 전체로 다시 돌린 예측은
   `ml_results.pkl`에 저장된 값과 135개 모두 1.8e-12 안에서 같다(`D4_learning_curves.json` `checks`).

---

## 3. 블록 선별 (C-2, dev 4,112 rxn만)

KRR(RBF)만 썼다. dev 행에 바깥 `KFold(5, shuffle=True, random_state=20261001)`를 두고, 바깥 fold마다 train 안에서 같은 KFold로
GridSearchCV를 돌렸다(nested). 선별 기준은 elst·Pauli·OI의 바깥 fold test NMAE 평균을 다시 5 fold로 평균한 값이다.
채택 조건은 상대 감소 ≥ 2 %이고, 바깥 5 fold 가운데 4 fold 이상에서 fold 기준이 낮아야 한다. job 997696(1 h 21 min).

### 3-1. 선별 경로 (`C2_selection_path.csv`, `prereg_rev5b.json`)

| 단계 | 현재 arm | 후보 | feature 수 | 기준 | 상대 감소 | 낮은 fold | 판정 |
|---|---|---|---:|---:|---:|---:|---|
| 0 | — | BASE | 73 | 0.2951 | — | — | 시작 |
| 1 | BASE | +B1 | 94 | 0.2792 | 5.38 % | 5/5 | 최선 아님 |
| 1 | BASE | +B2 | 96 | 0.2947 | 0.13 % | 3/5 | 최선 아님 |
| 1 | BASE | +B3 | 80 | 0.2928 | 0.78 % | 4/5 | 최선 아님 |
| 1 | BASE | **+B4** | 112 | **0.2746** | **6.95 %** | 5/5 | **채택** |
| 1 | BASE | +B5 | 89 | 0.2941 | 0.34 % | 2/5 | 최선 아님 |
| 1 | BASE | +B6 | 76 | 0.2874 | 2.62 % | 5/5 | 최선 아님 |
| 2 | BASE+B4 | **+B1** | 133 | **0.2677** | **2.50 %** | 5/5 | **채택** |
| 2 | BASE+B4 | +B2 | 135 | 0.2732 | 0.48 % | 4/5 | 최선 아님 |
| 2 | BASE+B4 | +B3 | 119 | 0.2758 | −0.46 % | 1/5 | 최선 아님 |
| 2 | BASE+B4 | +B5 | 128 | 0.2746 | −0.03 % | 3/5 | 최선 아님 |
| 2 | BASE+B4 | +B6 | 115 | 0.2716 | 1.08 % | 5/5 | 최선 아님 |
| 3 | BASE+B1+B4 | +B2 | 156 | 0.2672 | 0.20 % | 3/5 | 최선 아님 |
| 3 | BASE+B1+B4 | +B3 | 140 | 0.2675 | 0.07 % | 4/5 | 최선 아님 |
| 3 | BASE+B1+B4 | +B5 | 149 | 0.2668 | 0.32 % | 4/5 | 최선 아님 |
| 3 | BASE+B1+B4 | +B6 | 136 | 0.2646 | 1.15 % | 5/5 | 최선, 탈락(< 2 %) → 선택 종료 |

- **EXT_SEL = BASE + B1 + B4, 133 feature**(채택 순서 B4 → B1). 최종 기준 0.2677(elst 0.2751, Pauli 0.2615, OI 0.2665).
- 1단계에서 +B6(2.62 %, 5/5)도 채택 조건을 넘었지만 최선이 아니었다. B4를 더한 뒤에는 1.08 %, B1까지 더한 뒤에는 1.15 %로 2 %에 못 미쳤다.

### 3-2. 블록별 dev CV — 세 채널 (`C2_block_cv.csv`; 그림 `figures/fig7_channel_blocks`)

MAE (NMAE), 5 fold 평균. 마지막 열은 NMAE가 BASE보다 낮은 fold 수(elst / Pauli / OI)다.

| arm | feature 수 | elst | Pauli | OI | BASE보다 낮은 fold |
|---|---:|---:|---:|---:|---|
| BASE (= ESPLEY73) | 73 | 4.08 (0.301) | 7.47 (0.290) | 4.58 (0.294) | — |
| +B1 | 94 | 3.91 (0.289) | 7.00 (0.272) | 4.32 (0.277) | 5 / 5 / 5 |
| +B2 | 96 | 4.10 (0.302) | 7.44 (0.289) | 4.57 (0.293) | 2 / 3 / 3 |
| +B3 | 80 | 4.03 (0.298) | 7.43 (0.289) | 4.55 (0.292) | 5 / 3 / 4 |
| +B4 | 112 | 3.85 (0.284) | 6.88 (0.267) | 4.25 (0.273) | 5 / 5 / 5 |
| +B5 | 89 | 4.09 (0.302) | 7.45 (0.289) | 4.54 (0.291) | 3 / 2 / 4 |
| +B6 | 76 | 4.01 (0.296) | 7.24 (0.281) | 4.44 (0.284) | 5 / 5 / 5 |
| **EXT_SEL** (BASE+B1+B4) | 133 | **3.73 (0.275)** | **6.73 (0.262)** | **4.16 (0.267)** | 5 / 5 / 5 |
| EXT_ALL | 182 | 3.68 (0.272) | 6.59 (0.256) | 4.03 (0.259) | 5 / 5 / 5 |

### 3-3. 9개 타깃의 dev CV MAE 차이 (arm − BASE; 괄호 = NMAE가 BASE보다 낮은 fold 수, `C2_block_cv.csv`)

| 타깃 (BASE MAE) | +B1 | +B2 | +B3 | +B4 | +B5 | +B6 | EXT_SEL | EXT_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ΔE‡ (1.94) | −0.058 (5) | +0.031 (0) | −0.004 (2) | −0.146 (5) | +0.031 (0) | −0.006 (3) | −0.140 (5) | −0.178 (5) |
| d1 (1.86) | −0.065 (5) | +0.019 (3) | −0.010 (3) | −0.081 (5) | −0.004 (3) | −0.024 (5) | −0.078 (5) | −0.101 (5) |
| d2 (1.42) | −0.027 (5) | +0.001 (2) | +0.001 (2) | −0.064 (5) | +0.012 (2) | −0.018 (5) | −0.076 (5) | −0.102 (5) |
| elst (4.08) | −0.168 (5) | +0.015 (2) | −0.049 (5) | −0.236 (5) | +0.006 (3) | −0.069 (5) | −0.355 (5) | −0.400 (5) |
| Pauli (7.47) | −0.478 (5) | −0.035 (3) | −0.043 (3) | −0.594 (5) | −0.028 (2) | −0.229 (5) | −0.741 (5) | −0.880 (5) |
| OI (4.58) | −0.257 (5) | −0.011 (3) | −0.026 (4) | −0.327 (5) | −0.036 (4) | −0.143 (5) | −0.421 (5) | −0.547 (5) |
| disp (0.477) | +0.005 (2) | −0.002 (3) | −0.001 (4) | −0.015 (5) | +0.001 (2) | −0.002 (4) | −0.008 (4) | −0.007 (4) |
| CPCM (1.66) | −0.054 (5) | −0.010 (3) | −0.068 (5) | −0.035 (4) | −0.010 (3) | +0.012 (1) | −0.058 (4) | −0.117 (5) |
| CDS (0.261) | −0.008 (5) | +0.004 (0) | −0.001 (4) | −0.036 (5) | −0.001 (3) | −0.002 (3) | −0.034 (5) | −0.035 (5) |

- **B4가 단일 블록 가운데 기여가 가장 크다.** 9개 타깃 가운데 CPCM을 뺀 8개에서 단일 블록 가운데 MAE를 가장 크게 줄였다.
  선별 기준으로는 B4 다음이 B1, 그다음이 B6(feature 3개)이다.
- **B2, B3, B5는 세 채널의 선별 기준을 1 % 미만으로만 바꿨다**(0.13 / 0.78 / 0.34 %). B3는 CPCM에서는 단일 블록 가운데 가장 컸다(−0.068).
- **B5(거리 민감도 스캔) 가설은 지지되지 않았다.** 사양 B5는 기울기·곡률이 G1–DFT 형성 거리 차이를 1차로 보정할 수 있다는 가설을 두었고
  블록 절제로 시험하기로 했다. 결과는 기준 0.34 % 감소, 2/5 fold다.
- dev CV에서도 EXT_ALL이 세 채널 모두 EXT_SEL보다 낮다(NMAE 0.272 / 0.256 / 0.259 대 0.275 / 0.262 / 0.267).

### 3-4. 단일 feature–타깃 상관 (dev 4,112행, Pearson r, `C2_feature_target_r.csv`)

| 순위 | elst | Pauli | OI |
|---:|---|---|---|
| 1 | `mo_pauli_S2_occ` (B1) −0.676 | `mo_pauli_S2_occ` (B1) +0.707 | `mo_pauli_S2_occ` (B1) −0.711 |
| 2 | `vdw_n_085` (B2) −0.633 | `vdw_n_085` (B2) +0.650 | `vdw_n_085` (B2) −0.682 |
| 3 | `bm_hh_035` (B2) −0.603 | `bm_hh_025` (B2) +0.634 | `scan_curv_Eint` (B5) −0.645 |
| 4 | `bm_hh_025` (B2) −0.600 | `bm_hh_035` (B2) +0.628 | `bm_hh_025` (B2) −0.636 |
| 5 | `mo_oi_tot` (B1) −0.579 | `mo_oi_tot` (B1) +0.607 | `bm_hh_035` (B2) −0.636 |
| 6 | `scan_curv_Eint` (B5) −0.576 | `scan_curv_Eint` (B5) +0.591 | `mo_oi_tot` (B1) −0.633 |

- 부호는 라벨의 부호를 따른다(elst와 OI는 음, Pauli는 양). 접촉·overlap이 큰 반응일수록 세 항의 크기도 큰 경향이 있다(상관).
- 블록별 최대 |r|: B3 `pen_ovl_20` 0.480 / 0.470 / 0.466, B6 `nu_imag` 0.372 / 0.414 / 0.428. B4 최대 |r|은 0.36 이하다
  (`alpha_dip_mid`: elst −0.335, Pauli +0.357; `eta_dB`: OI +0.357).
- 비교: rev 4의 대응 단일 xTB 항은 `b_elst` 0.399, `b_pauli` 0.339, `b_oi` 0.362였다(`results_rev4/rev4_pre_ml.csv`, 4,839행이라 행 집합이 다르다).
- **단일 feature 상관과 블록 기여가 어긋난다.** 단일 r이 높은 B2와 B5는 KRR에 거의 기여하지 않았고, 단일 |r|이 0.36 이하인 B4가 가장 크게 기여했다.
  B2·B5의 정보가 BASE(거리, `b_*` 항)와 겹친다는 해석은 가설이며 시험하지 않았다.

### 3-5. 그림

- `fig7_channel_blocks`: elst, Pauli, OI의 dev CV NMAE. BASE, +B1 … +B6, EXT_SEL, EXT_ALL 막대와 바깥 fold 점, BASE 평균 점선.
- `fig8_channel_lockbox`: 같은 비교의 lockbox 판(§1). 9개 타깃의 BASE 대 EXT_SEL lockbox MAE, bootstrap CI, 패널 제목에 Δ [CI].

---

## 4. Phase별 기록

### Phase A — 폐기 실행 (job 997360: 단계 1–3, 약 40분, PHASE_A_OK; grep 997423; commit `18862409`)

- **결과**
  - A-1 재생성(`A_regen_check.json`, PASS): 다시 만든 `results_rev4/` 파일 17개를 `740ba895`의 같은 행과 비교했다. 실패 0, 모든 값 열의
    max |차이| 0(허용 오차 표 1e-9, 절제 1e-3). 남은 행: `espley_compare_role` 14 / 16, `espley_compare_appendix_best` 25 / 39,
    `espley_compare_index_d1d2_appendix` 14 / 20, `espley_compare_per_seed` 485 / 765, `espley_geometry_ablation`(·`_ownrows`) 42 / 68
    (arm A, C, C0, B_g1, O_g1), 절제 seed별 210 / 340. 헤드라인·부록·pre-ML 표는 행 수가 그대로다(9 / 108 / 27 / 36 / 9).
    절제 arm은 `espley_fairness_ablation.py`로 다시 돌렸다(공통 행과 원 규칙 행, KRR·SVR).
    `results_rev4/SUMMARY.md`는 폐기 서술을 지우고 맨 위에 폐기 문장을 두었다.
  - A-2 코드: `xtb_slice.py --geom`의 선택지를 {g1, g2}로, 기본값을 g1로 줄였다. 폐기 기하 전용 Python 스크립트 2개(`verify_geom_dft.py`,
    `phase2_report.py`)와 r4_* 2개(`r4_feat_verify.sh`, `r4_phase2_report.sh`)를 지웠다(commit `18862409`).
  - A-3 scratch(`A_deleted.txt`): 대상 6곳, 파일 112개, 18,656,690 B. 지우기 전에 경로·크기·sha256을 기록했다. rev 4 ablation 디렉터리 두 곳은
    남기고, 폐기 기하를 쓴 arm(B, B_role, O)의 JSON만 지웠다(26 + 26개).
  - A-4 행(`A_rows_check.json`, PASS): G1 parquet만으로 다시 만든 행이 `rows_rev4.csv`와 같은 집합·같은 순서다(4,839; G1 ok 4,860,
    NaN 제외 0, hygiene 제외 21).
  - A-4 grep(`A_grep_check.txt`, PASS): 18건, 모두 허용(폐기 문장 2건 — `results_rev4/SUMMARY.md`, `README.md`; `PREREG_REV4.md` 16건).
    `REV4_XTB_GEOMETRY.md`는 편집 없이(바이트 동일) `docs/specs/`로 옮겨져 검사 범위 밖에 있다. 이 파일에는 패턴에 걸리는 줄이 18개 있다(§6-8).
- **STOP 판정:** Phase A에는 STOP 조건이 없고, 네 검사 모두 PASS다.
- **진행:** Phase B로 갔다.

### Phase B-0 — 엔진 smoke (probe 997265 / 997328: r4 venv에 tblite 0.7.0 설치; smoke 997363)

- **결과** (`B0_smoke.json`, 코드 `r5-ext-2`)
  - 대상 5건 가운데 rxn 20은 G1 기하가 없어 건너뛰었다(rev 4 OptTS 미수렴). 105, 2434, 2721, 3452를 평가하고, 하전 반응 3312(q = 0, +1)와
    3323(q = 0, −2)을 더했다(6건).
  - gate 1 (tblite − xtb, GFN2 ALPB(water) 총에너지, 6건 × 구조 5개): 최대 |ΔE| 5.4e-8 Eh(rxn 2721 TS) < 1e-6.
  - gate 2 (조각 overlap = TS overlap의 해당 블록): 12개 조각 모두 차이 0.0 < 1e-10.
  - gate 3 (|CᵀSC − I|): 최대 3.6e-15(rxn 2721 TS) < 1e-8.
  - 보조 검사: 사중극자 순서 xx,xy,yy,xz,yz,zz, 쌍극자 확인 최대 2.9e-4 au(허용 1e-3).
  - 시간: rxn당 1.10 – 6.83 core-s, 평균 0.000795 core-h/rxn. 4,860건 투영 3.86 core-h.
- **STOP 판정:** 세 gate 모두 통과(`passed` true, `stop_reasons` []).
- **진행:** B-7로 갔다.

### Phase B-7 — 확장 feature 실행과 gate

- **1차 (array 997381, 병합 997382, 코드 `r5-ext-2`) — STOP.**
  - rows_rev4 실패 122 / 4,839 = 2.52 % > 1 %. 121건이 B6 생성물 단계였다: 생성물 그래프 ≠ G1 TS + 형성 결합 66, xtb 최적화 뒤 생성물 heavy 그래프
    변화 54, 최적화 미수렴 1. 나머지 1건은 B1 엔진 gate(rxn 4994, 1.9e-6 Eh)다. 계산량 6.06 core-h.
  - 출처: PREREG_REV5a §6과 병합 로그 `/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/logs/ext_merge.997382.out`. 1차의 `B7_report.json`과 분위 CSV는 2차가 덮어썼다(§6-4).
  - **사용자 결정(2026-10-01): B6 생성물 feature 3개(`xtb_dErxn`, `prog_ad`, `prog_be`)를 뺀다.** B6는 TS 성격 3개만 남고 블록 전체는 109열이다.
- **2차 (array 997666, 병합 997667, 코드 `r5-ext-3`) — 통과** (`B7_report.json`)
  - rows_rev4 실패 1 / 4,839 = 0.02 %(rxn 4994, B1 엔진 gate 1.9e-6 Eh). 병합 후 NaN 0, 비유한 값 0, B5 δ = 0 gate 실패 0.
  - 블록별 실패: B1 1, B2–B6 0. 기하 실패 0.
  - 엔진 일치(ok 행): tblite − xtb 최대 |ΔE| TS 9.2e-7, fA 1.4e-7, fB 1.6e-7 Eh. S 블록 최대 차이 0, |CᵀSC − I| 최대 6.9e-15.
    B5 δ = 0 재계산은 xtb_slice 값과 최대 2.0e-12 Eh(TS), 1.0e-12 Eh(조각) 안에서 같다.
  - core-h: 합계 2.63(B4 1.28, B5 0.91, B3 0.22, B1 0.20, B6 0.016, B2 0.0007). rxn당 평균 1.95 s, 중앙값 1.86 s.
  - **Δε 하한(1.0 eV).** 4,838행에서 occ→virt 합의 하한 적용 쌍은 모두 1,726개(rxn당 평균 0.357, 최대 7)이고, 1개 이상인 반응은 1,336개다.
    HOMO..HOMO−2 × LUMO..LUMO+2 창에서는 1,656개(최대 4), 1개 이상인 반응은 역시 1,336개다.
  - 분포(`B7_feature_quantiles.csv`, 109열의 중앙값·1 %·99 % 분위 등). 예:
    - `mo_pauli_S2_occ` 중앙값 0.0312(0.0105 – 0.0747).
    - `mo_dE_HA_LB` 중앙값 3.03 eV, 1 % 분위 −0.19 eV. `mo_dE_HB_LA` 중앙값 2.12 eV, 1 % 분위 −0.51 eV.
      조각 사이에서 GFN2 HOMO가 상대 조각의 LUMO보다 위에 있는 반응이 있다는 뜻이고, 이 쌍에 하한이 걸린다.
    - `nu_imag` 중앙값 −247.4 cm⁻¹(−420.1 – −60.6).
  - 행: `rows_rev5.csv` = 4,838(sha256 `a89a3d92…`).
  - 배열은 18 slice를 9개 array 원소(원소 i = slice i, i+9)로 돌렸다(§6-2).
- **STOP 판정:** 1차는 실패 비율 조건으로 STOP. 2차는 세 STOP 조건(실패 > 1 %, 병합 후 NaN, B5 gate) 모두 해당 없음.
- **진행:** 사용자 결정 뒤 2차를 돌렸고, 2차 통과 후 Phase C로 갔다.

### Phase C-1 — lockbox (job 997691; PREREG_REV5a commit `358b2d10`)

- **결과** (`lockbox.json`): `rows_rev5` 4,838(rxn_id 오름차순)에서 `np.random.default_rng(20261001)`로 **lockbox 726**(0.1501)을 뽑았다.
  **dev 4,112.** sha256: lockbox `97c05514…`, dev `19a8e212…`. 둘 다 PREREG_REV5a에 기록하고 선별 전에 commit했다.
- **STOP 판정:** 해당 조건 없음.
- **진행:** C-2로 갔다.

### Phase C-2 / C-3 — 선별 (job 997696, 1 h 21 min, git `358b2d10`; PREREG_REV5b commit `9c823fff`)

- **결과:** §3. dev 4,112행만 썼고 lockbox 726행은 X / y에 올리지 않았다(job 로그 `/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/logs/select.997696.out`).
  바깥 fold test 크기 823 / 823 / 822 / 822 / 822. 산출물 `C2_block_cv.csv`(fold별 `C2_block_cv_folds.csv`), `C2_selection_path.csv` / `.json`,
  `C2_feature_target_r.csv`.
- **C-3:** `prereg_rev5b.json`과 `PREREG_REV5b.md`에 EXT_SEL = BASE + B1 + B4(133)를 적고 Phase D 전에 commit했다.
- **STOP 판정:** 해당 조건 없음(채택 블록 2개라 `EXT_SEL = BASE` 경우도 아님).
- **진행:** Phase D로 갔다.

### Phase D-1 — lockbox 평가 (job 997729, wall 1,057 s, git `9c823fff`)

- **결과:** §1. arm 3 × 모델 4 × 타깃 9 = 108행(`D1_lockbox.csv`), 짝지은 차이 72행(`D1_lockbox_paired.csv`).
- **lockbox 보호 검사:** 처음 lockbox를 쓰기 전에 선별 캐시 파일 526개를 모두 읽었다. lockbox id 0개, 나타난 id = dev 4,112개(`D1_lockbox.json` `select_audit`).
  D-3(a), D-3(b), D-4 JSON에도 같은 검사 결과가 있다.
- **STOP 판정:** Phase D에는 STOP 조건이 없다. 사전 등록 gate(PREREG 파일·lockbox sha256 commit 확인)는 통과했다.
- **진행:** 그대로 진행.

### Phase D-2 — Protocol A (job 997730, 2시간 10분, 정상 종료) → §5

### Phase D-3 / D-4 — Espley 비교와 학습 곡선 (job 997731)

- **결과:** §2. D-3(a) wall 851 s, D-3(b) 101 s, D-4 13 s. 코드 `espley_rev5.py`는 commit된 그대로였다(`code_file.committed_unchanged` true).
- **재현 확인:**
  - 역할 d1/d2의 그들의 프로토콜 SVR / KRR 예측은 rev 4 재학습을 다시 썼다(`rev4_replicated` true, 튜닝 열 47).
  - Espley 46 × 우리 파이프라인 역할 예측은 rev 4 파일과 14,032개 모두 1.9e-11 안에서 같다.
  - 모든 쪽의 test 타깃이 seed마다 같다(최대 차이 7.1e-15).
- **STOP 판정:** 해당 조건 없음.
- **진행:** Phase E로 갔다.

### Phase E — 그림 (job 997732; fig7 다시 그림 997743)

`figures_rev5_manifest.json`: 8개 모두 status ok, `layout_warnings` 없음. 글꼴 DejaVu Sans(Arial 없음), 7 pt, 양단 183 mm, PNG 300 dpi + PDF + 같은 이름의 CSV.
색은 `rev5_common.COLORS`(Espley `#2a78d6`, EXT_SEL `#eb6834`, ESPLEY73 `#1baf7a`, ESPLEY46 `#eda100`).

| 그림 | 출처 | 보이는 것 |
|---|---|---|
| `fig1_espley_mae` | D-3(a) | 타깃 5 × 계열 4(Espley SVR, ESPLEY46, ESPLEY73, EXT_SEL) 막대. 5 seed 평균 test MAE, 오차 막대 = seed MAE의 SE, seed 점 5개, 값 라벨. 타깃마다 Espley ÷ EXT_SEL(seed별 비의 평균)과 Nadeau–Bengio p(EXT_SEL 대 Espley SVR). |
| `fig2_espley_lockbox` | D-3(b) | lockbox 502 rxn 정면 비교. Espley SVR(그들의 프로토콜) 대 EXT_SEL KRR, bootstrap 95 % CI. |
| `fig3_parity` | D-3(a) | 2행(Espley SVR / EXT_SEL) × 타깃 5열. x = DFT, y = 반응별로 test였던 seed의 예측 평균(1,370 rxn). y = x선, ±1·±2 kcal/mol 회색 띠, 타깃마다 축 공유, MAE·r² 주석. |
| `fig4_error_ecdf` | D-3(a) | 타깃별 \|오차\| 누적분포, 계열 4개(타깃마다 seed별 오차 1,665개). 1·2 kcal/mol 세로선, 1 kcal/mol 안의 비율 라벨. |
| `fig5_seed_pairs` | D-3(a) | 타깃별 seed 5개의 짝지은 선(Espley SVR → EXT_SEL), 제목에 p. |
| `fig6_learning_curves` | D-4 | 타깃별 test MAE 대 학습 행 수(평균 ± SE), Espley SVR 대 EXT_SEL. 점선 = Espley SVR f = 1.0, 도달 표시 "n ≤ 265". |
| `fig7_channel_blocks` | C-2 | elst, Pauli, OI의 dev CV NMAE: BASE, +B1 … +B6, EXT_SEL, EXT_ALL(fold 점 포함), BASE 평균 점선. |
| `fig8_channel_lockbox` | D-1 | 9개 타깃의 lockbox MAE, BASE 대 EXT_SEL, 짝지은 bootstrap CI(패널 제목에 Δ [CI]). |

- **fig7 다시 그림(997743).** 세로로 돌린 값 라벨이 fold 점과 겹쳐, 라벨 간격을 1.5 pt에서 4 pt로 늘려 fig7만 다시 그렸다.
  `figures_rev5.py`의 이 수정(4줄)은 이 문서를 쓰는 시점에 commit되지 않았다. 나머지 그림은 수정 전 코드(sha256 `38ac315a…`)로 그렸다.
- **STOP 판정:** 해당 조건 없음.

### 하류 분석 (job 997733, 5분, 정상 종료) → §5

---

## 5. D-2 Protocol A와 하류 분석

### 5-1. D-2 Protocol A — rev 4와의 연속성 (잡 997730, 2시간 10분)

`rows_rev5` 4,838 rxn 전체, 80/10/10 분할, seed 22/23/14/1/2, seed마다 nested GridSearchCV(5-fold), X·y 표준화, KRR(RBF).
출처: `D2_protocolA_headline.md` / `.csv`, 전 모델·arm은 `D2_protocolA_table.csv`, 그림 `D2_protocolA_mae_bar_*`, `D2_protocolA_scatter_*`.

| 타깃 | ESPLEY46 (46) | ESPLEY73 (73) | EXT_SEL † (133) | EXT_ALL (182) | EXT_SEL − ESPLEY73 | EXT_ALL − ESPLEY73 |
|---|---:|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | 2.10 ± 0.03 | 1.83 ± 0.06 | 1.71 ± 0.06 | 1.73 ± 0.08 | −0.12 | −0.10 |
| d1 | 1.84 ± 0.10 | 1.78 ± 0.08 | 1.70 ± 0.09 | 1.70 ± 0.10 | −0.08 | −0.08 |
| d2 | 1.51 ± 0.08 | 1.36 ± 0.04 | 1.29 ± 0.05 | 1.28 ± 0.06 | −0.07 | −0.08 |
| elst | 5.15 ± 0.26 | 3.95 ± 0.16 | 3.62 ± 0.15 | 3.59 ± 0.14 | −0.33 | −0.36 |
| Pauli | 8.02 ± 0.39 | 7.19 ± 0.22 | 6.51 ± 0.26 | 6.45 ± 0.25 | −0.69 | −0.74 |
| OI | 4.92 ± 0.19 | 4.41 ± 0.08 | 4.02 ± 0.15 | 3.96 ± 0.10 | −0.38 | −0.45 |
| disp | 1.36 ± 0.05 | 0.47 ± 0.02 | 0.46 ± 0.02 | 0.47 ± 0.02 | −0.01 | −0.01 |
| CPCM | 3.01 ± 0.06 | 1.60 ± 0.07 | 1.55 ± 0.08 | 1.51 ± 0.09 | −0.04 | −0.09 |
| CDS | 0.34 ± 0.01 | 0.26 ± 0.00 | 0.22 ± 0.00 | 0.22 ± 0.01 | −0.04 | −0.04 |

(test MAE ± seed 간 sd, kcal/mol.)

- **† 선별 편향.** EXT_SEL의 블록은 dev 행에서 골랐고, dev 행은 이 표의 test 행과 겹친다. 그래서 EXT_SEL 열은 낙관적일 수 있다.
  편향 없는 값은 §1(D-1 lockbox)이다. 방향은 D-1과 같다: 절대 감소폭은 Pauli, OI, elst 순으로 크다.
- **rev 4 헤드라인과 직접 비교하지 않는다.** 행이 4,839 → 4,838로 한 줄 줄면서 `split_80_10_10`의 무작위 분할이 통째로 바뀌었다.
  예를 들어 ESPLEY73 · KRR barrier는 rev 4 1.92, D-2 1.83인데, 이 차이는 test 행이 달라서 생긴 것이다(feature와 모델은 같다).
  같은 분할 안의 arm 간 차이만 해석한다.

### 5-2. 하류 분석 — G1, EXT_SEL · KRR, D-2 예측 (잡 997733)

출처: `downstream_EXT_SEL/`. D-2의 test fold 예측(5 seed 합집합, 1,985 rxn)을 쓴다.

**전하별 test MAE** (`charge_breakdown_g1.csv`; 중성 1,897, 하전 88 = q₂ −2 70 + q₂ +1 18; q₂ = dipolarophile 전하)

| 타깃 | 전체 | 중성 | 하전 | q₂ = −2 | q₂ = +1 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.71 | 1.70 | 1.92 | 1.82 | 2.40 |
| d1 | 1.70 | 1.67 | 2.45 | 2.03 | 4.43 |
| d2 | 1.29 | 1.25 | 2.11 | 1.05 | 7.02 |
| elst | 3.62 | 3.48 | 6.56 | 5.06 | 13.49 |
| Pauli | 6.51 | 6.37 | 9.33 | 5.69 | 26.19 |
| OI | 4.02 | 3.89 | 6.97 | 4.51 | 18.36 |
| disp | 0.46 | 0.46 | 0.50 | 0.42 | 0.87 |
| CPCM | 1.55 | 1.45 | 3.89 | 3.39 | 6.19 |
| CDS | 0.217 | 0.216 | 0.241 | 0.180 | 0.524 |

- 하전 반응, 특히 q₂ = +1(18건)이 여전히 약점이다(Pauli 26.19, OI 18.36, elst 13.49).

**그룹 분할** (`group_split_g1.csv`; 값 = 그룹 홀드아웃 MAE / random MAE)

| 타깃 | random MAE | dipolarophile 홀드 | dipole 홀드 |
|---|---:|---:|---:|
| barrier | 1.71 | 1.316 | 1.063 |
| d1 | 1.70 | 1.164 | 1.036 |
| d2 | 1.29 | 1.165 | 1.063 |
| elst | 3.62 | 1.148 | 1.038 |
| Pauli | 6.51 | 1.156 | 1.055 |
| OI | 4.02 | 1.171 | 1.036 |
| disp | 0.46 | 1.048 | 1.041 |
| CPCM | 1.55 | 1.046 | 1.007 |
| CDS | 0.22 | 1.139 | 1.102 |

**MMP 지배 채널 일치(`dom_agree`)와 split별 null 기준선** (`split_metrics_g1.csv`, `evaluate_pairs_g1.json`)
- 방법은 rev 4와 같다: MMP 쌍마다 7채널(d1, d2, elst, Pauli, OI, CPCM, disp) 중 Δ가 가장 큰 채널을 참값과 예측값에서 각각 고르고 일치율을 잰다.
  예측은 고정 하이퍼파라미터 KRR(alpha = gamma = 1e-3)의 out-of-fold 값이다(rev 4 그대로, 튜닝 없음).
- null 기준선: (i) **최빈 채널 비율** = 그 split에서 참 지배 채널이 가장 흔한 채널(모든 split에서 Pauli)인 쌍의 비율 = "항상 Pauli"의 일치율;
  (ii) **순열 기준선** = 예측 지배 채널을 split 안에서 섞었을 때의 기대 일치율 Σ_c p_true(c) p_pred(c) (1,000회 순열로도 확인, rng 20261001).

| split | 쌍 | dom_agree | 최빈(Pauli) 기준선 | 순열 기준선 |
|---|---:|---:|---:|---:|
| random | 885 | 0.688 | 0.646 | 0.470 |
| dipole_class | 3,752 | **0.599** | **0.621** | 0.431 |
| loso: *C | 145 | **0.703** | **0.807** | 0.642 |
| loso: *c1ccccc1 | 318 | 0.711 | 0.682 | 0.491 |
| loso: *NC (= *C(=O)NC) | 670 | 0.733 | 0.699 | 0.537 |
| loso: *C#N | 1,075 | 0.768 | 0.757 | 0.598 |
| loso: *OC (= *C(=O)OC) | 658 | 0.757 | 0.731 | 0.555 |
| loso: *C(C)=O | 1,020 | 0.798 | 0.768 | 0.600 |
| loso: *[N-]c1ccccc1 | 86 | 0.791 | 0.779 | 0.597 |
| loso: *[N+]#CC(C)=O | 142 | 0.768 | 0.746 | 0.556 |

- **모든 split에서 순열 기준선보다는 높다**(순열 p = 0.001, 1,000회 순열에서 가능한 최솟값).
- **그러나 최빈 채널 기준선("항상 Pauli")과 비교하면 이점이 작거나 없다.** random에서 +0.042(0.688 대 0.646)이고,
  **dipole_class(−0.022)와 loso *C(−0.104)에서는 "항상 Pauli"보다 낮다.** 나머지 loso split에서는 +0.011 ~ +0.034다.
  rev 4의 같은 지표(ESPLEY73, `results_rev4/downstream_g1/split_metrics_g1.csv`)는 random 0.68, dipole_class 0.583이었다.
- 즉 지배 채널 판정은 채널 MAE 개선에도 불구하고 대부분 "Pauli가 지배한다"는 사전 분포에서 나온다. 반응 쌍 사이의 채널 순위를 예측했다고
  주장하려면 이 기준선을 넘는 split별 근거가 필요하다.
- loso 치환기 10개 가운데 2쌍(*NC / *C(=O)NC, *OC / *C(=O)OC)은 같은 쌍 집합이다(rev 4와 같은 이유). 고유 홀드아웃은 8개다.

---

## 6. 사전 등록·사양과 다른 점

1. **B-0 대상.** 사양의 5건 가운데 rxn 20은 G1 기하가 없어(rev 4 OptTS 미수렴) 평가하지 못했다. 나머지 4건과 하전 반응 2건(3312, 3323)을
   평가했다(PREREG_REV5a §6). 또 B-0은 B6 생성물 feature가 있던 코드(`r5-ext-2`)로 돌았고, 생성물 feature를 뺀 최종 코드(`r5-ext-3`)로는
   smoke를 다시 하지 않았다. 같은 gate는 B-7 2차의 행별 QC 열이 모든 행에서 확인한다(§4 B-7).
2. **B-7 배열 모양.** 사양은 "18 slices %10"이다. 실제로는 MaxSubmit 20(array 원소마다 1)에 맞추려고 `--array=0-8`, 원소 i가 slice i와 i+9를
   차례로 계산했다. slice 수(18)와 결과는 같다.
3. **B6 생성물 feature 제외 (사용자 결정 2026-10-01).** B-7 1차가 STOP 한 뒤(실패 122 / 4,839, 그중 121건이 B6 생성물 단계), 사용자 결정으로
   `xtb_dErxn`, `prog_ad`, `prog_be`를 뺐다. 이 결정은 1차 실행의 수치(블록별 실패 수 등)를 본 뒤에 내려졌다. 그때 확장 블록으로 학습한 모델은
   없었다. 따라서 rev 5의 어떤 arm에도 Hammond 생성물 feature(xTB 반응 에너지, 진행도)는 없고, 사양 B6의 절반만 시험했다.
4. **1차 B-7 기록과 실행 코드.** 2차 병합이 같은 이름의 `B7_report.json`, `B7_feature_quantiles.csv`, 확장 parquet을 덮어썼다(`ext_parquet_write` = replaced).
   1차 수치는 병합 로그(`/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/logs/ext_merge.997382.out`)와 PREREG_REV5a §6에만 남는다.
   - 2차는 commit 전의 작업 트리 코드로 돌았다: HEAD `18862409` + 수정된 `ext_features.py`, `rev5_common.py`(B6 열 112 → 109), `r5_ext_merge.py`.
     이 수정은 뒤에 `358b2d10`으로 commit됐지만, 실행 시점 파일과 commit 파일이 같은지 sha256으로 기록하지 않았다(`B7_report.json`에는 코드 버전이
     없고, job 로그 `r4_env` 줄에는 HEAD만 있다).
   - 같은 방식으로 Phase A(997360, grep 997423), B-0(997363), B-7 1차(997381 / 997382)는 HEAD `740ba895` 위의 미commit rev 5 코드
     (`18862409`에서 처음 commit)로, lockbox(997691)는 HEAD `18862409`에서 돌았다(job 로그 `r4_env` 줄).
5. **lockbox 이전에 lockbox 행을 포함해 계산된 것 (PREREG_REV5a에 공개).**
   - Phase A가 rev 4 분석(헤드라인 표, Espley 비교, 기하 절제 arm A / C / C0 / B_g1 / O_g1 재학습)을 다시 만들었다. 확장 블록은 없다.
   - B-7 feature 분포(`B7_feature_quantiles.csv`)는 lockbox 행을 포함한 4,838행에서 계산했다(서술용).
   - 둘 다 블록 선별에 쓰이지 않았다. 다만 lockbox 반응은 rev 4의 Protocol A test fold와 Espley 비교에 이미 들어 있었다(BASE = ESPLEY73 모델의 성능은 이미 보였다).
6. **D-3(a) Espley 역할 SVR.** 사양은 "(a) 그들의 프로토콜 SVR로 재학습"이다. rev 5는 다시 돌리지 않고 rev 4의 재학습 예측(`role_theirs_preds.parquet`;
   rev 4에서 그들의 프로토콜 재현을 확인한 코드의 출력, `results_rev4/espley_compare_checks.json`)을 다시 썼다. D-4의 역할 타깃 하이퍼파라미터도
   이 값(C 50 / 30, ε 1 / 0.25, gamma auto)이다.
7. **D-4 세부.**
   - 사양의 "Espley 행 순서"는 PREREG_REV5a §4에서 "그들의 `ml_analysis.py` X_train 순서(분할 후 순서)"로 정했고, ds3 행 순서는 민감도로 따로 적었다(§2-3).
   - fig6 기준선은 사양의 "Espley 전체 데이터 MAE" 대신, 사전 등록대로 같은 곡선 학습 행(평균 2,653.8)으로 학습한 Espley SVR의 f = 1.0 MAE다.
     그들의 train 분할 전체(2,807 – 2,808행)로 학습한 D-3(a) 값도 `D4_learning_curves_reach.csv`에 있고, 도달 판정은 같다.
   - 도달 행 수는 첫 격자점에서 이미 도달해 "≤ 265.4"로만 적는다.
   - Espley 쪽은 `ml_analysis.py`처럼 StandardScaler를 train 전체에 맞춘 뒤 부분집합을 자른다. 우리 쪽은 파이프라인을 부분집합에서 다시 맞춘다(X·y 스케일러 포함).
8. **Phase A grep 범위.** `REV4_XTB_GEOMETRY.md`(rev 4 사양)를 편집 없이 `docs/specs/`로 옮겼다. 그래서 A-4 grep이 이 파일의 패턴 일치 18줄을
   보지 않는다(`A_grep_check.txt` 머리말에 공개, 바이트 동일 확인).
9. **그림 선택.**
   - 글꼴은 Arial이 없어 DejaVu Sans를 썼다(사양이 허용). 모든 그림을 양단 183 mm로 그렸다(PREREG_REV5a §5).
   - fig1 오차 막대는 5 seed MAE의 SE이고, Espley 정의 SE는 CSV에 있다(PREREG_REV5a §5). fig1 배수는 seed별 비의 평균이다(§2-2의 4).
   - fig3의 MAE·r² 주석은 반응별 seed 평균 예측(1,370 rxn)으로 계산했다. 그래서 seed별 MAE의 평균(표)과 조금 다르다
     (예: Espley SVR dipole 2.556 대 2.560, EXT_SEL ΔE‡ 1.669 대 1.663; `figures/fig3_parity.csv`).
   - fig4는 타깃마다 5 seed의 오차 1,665개를 합쳐 그렸다.
   - fig7은 사양 팔레트 밖의 회색 두 개(단일 블록 `#c8c8c8`, EXT_ALL `#8f8f8f`)를 쓰고, BASE는 ESPLEY73 색으로 칠했다. fig8의 BASE도 ESPLEY73 색이다.
   - fig7 수정 코드가 commit되지 않았다(§4 Phase E).
10. **rev 4에서 넘어온 기록 문제.** `results_rev4/espley_geometry_ablation_ownrows_rows.json`은 Phase A 재생성 뒤에도 공통 행 기록과 같은 행 집합
    (`n_common` 3,327, `rows_sha256` `4563c182…`)을 담는다(`A_regen_check.json`). rev 4 SUMMARY §4-3의 문제를 고치지 않았다.
11. **HPC 규칙 위반 (자진 보고).**
    - 코드 작성 에이전트 4개가 로그인 노드에서 빈 입력의 `python3`를 한 번씩 실행했다(계산 없음, PREREG_REV5a §6).
    - 이 문서의 검증 에이전트 2개가 로그인 노드에서 각각 `python3 --version`, `python3 -c 1`을 한 번 실행했다(자진 보고, 계산·파일 쓰기 없음).
    - rev 4에서 오케스트레이터가 로그인 노드 대기 루프를 돌렸다(사용자 지적). rev 5는 세션 스케줄러와 단발 확인으로 바꿨다.

---

## 7. 한계, 산출물, 재현

### 한계

- **G1은 배포 성능이 아니다.** G1 TS와 참조는 라벨의 DFT 구조에서 출발한다. conformer와 입체 선택은 DFT 정보다. SMILES에서 출발하는 G2(autodE)는 아직 돌리지 않았다(사용자 결정 대기).
- **행 선택.** 모든 rev 5 결과는 G1 gate 통과 ∩ 확장 feature ok인 4,838 rxn(status ok 5,260 가운데)에서 나왔다. G1 실패 400건, hygiene 21건, 확장 feature 실패 1건(4994)의
  오차는 모른다. Espley 비교도 G1 성공 ∩ hygiene ∩ 확장 feature ok 행(`rows_rev5` ∩ Espley, 3,322 / 라벨 있는 Espley 3,509)으로 제한된다.
  이 필터는 우리 파이프라인에서 나왔고 우리 쪽에 유리할 수 있다.
- **Espley 비교에서 두 쪽은 기하(AM1 대 G1), feature, 모델, 튜닝이 모두 다르다.** rev 4 Phase 4-4 판정(G1을 "입력 대응 arm"으로 선언하지 않음)이 그대로다.
  D-3(a)의 Espley 쪽은 더 많은 행으로 학습했고, "최강" 모델은 test 행에서 골랐다(둘 다 Espley에 유리). D-3(b)는 두 쪽이 같은 2,820행으로 학습한다.
- **선별 편향의 범위.** 선별은 dev 행만 썼다. 그러나 D-3(a) test 행과 D-2 test 행은 dev 행과 겹친다. EXT_SEL의 블록 선별 편향이 없는 값은 D-1(lockbox)과 D-3(b)뿐이다.
- **lockbox 크기.** lockbox는 726행(Espley 행은 502)이고, 85 / 15 분할 하나다. CI는 lockbox 행의 표본 변동만 반영하고 학습 쪽 변동(분할, 튜닝)은 반영하지 않는다.
- **EXT_ALL 대 EXT_SEL은 판정하지 않았다.** EXT_ALL의 점추정이 elst / Pauli / OI에서 낮지만, 짝지은 CI가 없다(§1). 전진 선별은 탐욕적이고 2 % 문턱에서 멈췄다.
- **B6는 TS 성격 3개뿐이다.** 생성물(Hammond) feature는 사용자 결정으로 빠졌다(§6-3).
- **Δε 하한.** 1,336 / 4,838 반응에서 S²/Δε 항의 쌍 하나 이상이 하한(1.0 eV, 사전 고정, 적합하지 않음)에 걸렸다. 조각 사이 HOMO가 상대 LUMO보다 위인 반응도 있다(§4 B-7).
- **GFN2 IP/EA의 절댓값은 계통적으로 어긋날 수 있다.** B4(가장 먼저 채택된 블록)의 μ, η, ω, ΔN은 `--vipea`에서 나온다. ML에서는 상대값으로만 쓰이므로 그대로 두었다.
  B4의 기여가 이 지수들의 화학적 해석을 검증하지는 않는다. B4의 단일 feature 상관은 |r| ≤ 0.36이다(§3-4).
- **B5 가설은 블록 절제에서 지지되지 않았다**(§3-3). 기하 불일치가 Pauli / OI / elst 오차의 원인이라는 rev 4의 가설은 여전히 직접 시험하지 않았다.
- **튜닝 grid 경계.** D-1의 KRR 27개 단위 가운데 6개에서 γ가 grid 하단(3e-4)에 걸렸다(EXT_SEL·EXT_ALL ΔE‡, 세 arm의 disp, EXT_ALL CDS; `D1_lockbox.csv`
  `edge_hits_total`). D-3(b)의 EXT_SEL dipolarophile도 같다. D-3(a)의 EXT_SEL KRR은 seed × 타깃 25개 단위 가운데 6개(dipolarophile seed 22 / 23 / 1 / 2,
  ΔE‡ seed 1, ΔG‡ seed 14)에서 γ = 3e-4였고(`D3a.json` `ours_best_params`; ESPLEY46·ESPLEY73은 0개), 이 값은 D-4에도 그대로 쓰였다.
  D-1 부록 모델은 경계 일치가 더 많다(`edge_hits_total`: Ridge 7 / 27 단위, SVR 27 / 27, XGB 27 / 27). §1의 SVR 대 KRR 비교는 이 점을 감안해야 한다.
  grid는 rev 4 그대로 사전 고정했다.
- **CPCM과 disp.** CPCM 개선은 확인되지 않았고(§1), disp도 KRR에서 CI가 0을 포함한다(부록 SVR·XGB에서는 0 아래). 하전 반응 성능은 하류 분석(§5)에서 본다.

### 산출물 (`results_rev5/`)

- **사전 등록**
  - `PREREG_REV5a.md`: lockbox, dev CV, 선별 규칙, Phase D·E 세부, 이미 알려진 차이(commit `358b2d10`).
  - `PREREG_REV5b.md`: EXT_SEL 고정(commit `9c823fff`).
  - `prereg_rev5b.json`: EXT_SEL 블록·채택 순서·feature 수, 선별 규칙과 경로 요약, 입력 sha256.
- **Phase A**
  - `A_deleted.txt`: 지운 scratch 112개의 경로·크기·sha256(삭제 전 기록)과 상태 줄.
  - `A_regen_check.json`: 재생성한 `results_rev4/` 파일 17개 대 `740ba895` 비교(PASS).
  - `A_rows_check.json`: G1만으로 다시 만든 행 = `rows_rev4.csv`(PASS).
  - `A_grep_check.txt`: A-4 grep 결과(18건 모두 허용, 옮긴 파일 공개).
- **Phase B**
  - `B0_smoke.json`: 엔진 smoke 6건의 gate 값, QC, feature, 시간.
  - `B7_report.json`: B-7 2차의 slice, 블록별 실패·core-h, Δε 하한 수, QC, gate, 분위.
  - `B7_feature_quantiles.csv`: 109열의 n, 비유한 수, 중앙값, 1 %·99 % 분위, 평균, sd, 최소, 최대(4,838행).
  - `rows_rev5.csv`: 행 4,838(rxn_id).
- **Phase C**
  - `lockbox.json`: lockbox 규칙, 수, sha256.
  - `lockbox_ids.csv`: lockbox 726 rxn_id.
  - `dev_ids.csv`: dev 4,112 rxn_id.
  - `C2_block_cv.csv`: arm × 타깃 dev CV 요약(MAE, NMAE, r², BASE 대비 차이, 낮은 fold 수; 114행).
  - `C2_block_cv_folds.csv`: fold별 값(MAE, NMAE, r², 튜닝값, 경계 일치; 570행).
  - `C2_selection_path.csv`: 단계별 후보, 기준, 상대 감소, fold 수, 판정(16행).
  - `C2_selection_path.json`: 규칙·fold 정의를 포함한 선별 전체 기록.
  - `C2_feature_target_r.csv`: 확장 feature 109개 × elst / Pauli / OI의 Pearson r(dev).
- **Phase D**
  - `D1_lockbox.csv`: arm 3 × 모델 4 × 타깃 9의 lockbox MAE·NMAE·r²·CI, 튜닝값, 경계 일치(108행).
  - `D1_lockbox_paired.csv`: EXT_SEL − BASE, EXT_ALL − BASE 짝지은 bootstrap 차이(72행).
  - `D1_lockbox.json`: 설정, 입력 sha256, 선별 캐시 검사, bootstrap 인덱스 sha256, 단위별 시간.
  - `D3a_summary.csv`: D-3(a) 계열별 seed 평균 MAE·SE·r²·NMAE(54행).
  - `D3a_tests.csv`: Nadeau–Bengio 검정 30개(seed별 차이 포함).
  - `D3a_per_seed.csv`: 계열 × 타깃 × seed MAE 등(270행).
  - `D3a_predictions.csv`: 그림용 test 예측(계열 4 × 타깃 5 × seed 5).
  - `D3a_rows.csv`: 채점 행 3,322(rxn_id, Espley 행 번호).
  - `D3a.json`: 행 흐름, 설계, 그들의 프로토콜 재사용, 교차 확인, 최강 모델, 튜닝값, rev 4 기준값, 주석.
  - `D3b_summary.csv`: D-3(b) 쪽별 MAE·CI·NMAE·r²·튜닝값.
  - `D3b_paired.csv`: Espley SVR / KRR − EXT_SEL 짝지은 bootstrap 차이와 배수.
  - `D3b_predictions.csv`: D-3(b) test 예측(502 rxn).
  - `D3b.json`: D-3(b) 설정, 행, 그들의 프로토콜 세부, 단위별 기록.
  - `D4_learning_curves.csv`: 타깃 × 쪽 × f × seed MAE(500행).
  - `D4_learning_curves_summary.csv`: seed 평균·SE(100행).
  - `D4_learning_curves_reach.csv`: 기준선 2개에 대한 도달 판정(10행).
  - `D4_learning_curves_sensitivity.csv`: ds3 행 순서 민감도(500행).
  - `D4_learning_curves.json`: 설계, 하이퍼파라미터와 출처, 재현 확인, 도달, 요약.
- **Phase E** (`figures/`)
  - `fig1_espley_mae` … `fig8_channel_lockbox`: 각각 `.pdf`(벡터), `.png`(300 dpi), `.csv`(그린 값).
  - `figures_rev5_manifest.json`: 그림별 입력 sha256, 코드 sha256, 상태, 배치 경고, 스타일.
- `SUMMARY.md`: 이 문서.
- `D2_protocolA_headline.md` / `.csv`: D-2 Protocol A 헤드라인(KRR, arm 4개 × 9 타깃, MAE ± sd, NMAE, r²); `D2_protocolA_table.csv`: 모든 arm·모델; `D2_protocolA_meta.json`: 입력 sha256과 실행 기록.
- `D2_protocolA_mae_bar_*` / `D2_protocolA_scatter_*`: D-2 그림(arm별 MAE 막대, KRR 산점도).
- `downstream_EXT_SEL/`: 하류 분석(EXT_SEL · KRR, D-2 예측) — `charge_breakdown_g1.csv`, `group_split_g1.csv`, `split_metrics_g1.csv`(MMP dMAE, dom_agree와 null 기준선), `margin_calibration_g1.csv`, `mmp_pairs_v2_g1.csv`, `evaluate_pairs_g1.json`.
- **scratch:** `/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/`(`ext/`, `select/`, `lockbox/`, `espley/`, `protoA/`, job 로그 `logs/`).
  확장 feature parquet은 `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_ext_g1.parquet`(sha256 `35d1a84f…`).

### 재현 (`analysis/espley_xtb_repro/`, 모두 sbatch, `--time=48:00:00`; 단계는 idempotent이고 결과는 원자적으로 쓴다)

1. `r5_probe.sh` — r4 venv에 tblite 0.7.0 설치·확인.
2. `r5_phaseA.sh` — `r5_phase_a.py`(`r5_phaseA_discard.txt`): scratch 기록·삭제 → `make_rows_rev4.py --check-g1-only` → `results_rev4/` 재생성·대조.
   `results_rev4/SUMMARY.md`를 다시 쓴 뒤 `PHASEA_STEPS=4`로 grep 검사.
3. `r5_ext_smoke.sh` — B-0(`ext_features.py smoke`).
4. `sbatch --array=0-8 r5_ext_array.sh` → `sbatch --dependency=afterany:<JID> r5_ext_merge.sh` — B-7(`ext_features.py slice`, `r5_ext_merge.py`).
5. `r5_lockbox.sh` — C-1(`select_blocks.py lockbox`). 이어서 PREREG_REV5a commit.
6. `r5_select.sh` — C-2 / C-3(`select_blocks.py select`). 이어서 PREREG_REV5b commit.
7. `r5_lockbox_eval.sh` — D-1(`final_eval.py`). `r5_protoA.sh` — D-2(`train_ml_single.py`, `aggregate_ml.py`). `r5_espley.sh all` — D-3 / D-4(`espley_rev5.py`).
   세 스크립트는 사전 등록 파일이 commit됐는지 먼저 확인한다(D-1과 D-3 / D-4는 lockbox sha256 기록도 확인한다).
8. `r5_figures.sh` — Phase E(`figures_rev5.py`; `--only fig7`로 한 그림만).
9. `sbatch --dependency=afterok:<PROTOA_JID> r5_downstream.sh` — 하류 분석(`analyze_extra.py`, `evaluate_pairs.py`).

