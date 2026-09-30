# espley_xtb_repro — rev 4 결과 (2026-09-30): xTB 기하 위 feature (G1)
G0(DFT oracle geometry) 결과는 2026-09-30 사용자 결정으로 폐기. git ≤ `740ba89`에만 남음.

사양 [`REV4_XTB_GEOMETRY.md`](../../../docs/specs/REV4_XTB_GEOMETRY.md), 사전 등록 [`PREREG_REV4.md`](PREREG_REV4.md)
(commit `be0a457b`, 2026-09-30 15:47:26). 브랜치 `espley-rev4` (기준 `1ada37aa`).

- **G1 = GFN2-xTB/ALPB(water) 기하.** TS는 ORCA 6.1.1 `! XTB2 ALPB(water) OptTS Freq`로, 참조 두 개는 xtb 6.7.1
  `--opt tight --alpb water`로 다시 최적화했다. 출발점은 라벨의 DFT TS와 DFT 참조다. 남는 DFT 정보는 conformer와 입체 선택뿐이다.
  **G1이 rev 4의 헤드라인이다.** G1은 Espley 2024와 비교하는 arm(“Espley 비교 arm”)이다. 다만 사전 등록 규칙(Phase 4-4)으로는
  “입력 대응 arm”으로 선언하지 않는다(§2 Phase 4-4). DFT TS에서 출발하므로 배포 성능도 아니다(§6).
- 타깃: repo 루트 `labels_all.json`(SMD(water) 재계산, 방법 ① 조립)의 DFT 라벨, status ok 5,260.
- 수치는 이 폴더(`results_rev4/`)의 파일에서 옮겼다. 폴더 밖 출처는 그 자리에 경로를 적었다. MAE는 kcal/mol, 소수 둘째 자리,
  NMAE·r²는 셋째 자리, 비율은 파일에 저장된 자릿수 그대로다. rev 1–3 결과 파일은 사용자 지시(“이전 실험 결과 전부 삭제”)로 지웠다.
  git 이력(≤ `1ada37aa`)에는 남아 있다.

---

## 1. 헤드라인 — G1 · ESPLEY73 · KRR(RBF) (사전 고정)

Protocol A: 80/10/10 분할, seed 22/23/14/1/2, seed마다 nested GridSearchCV(5-fold), X·y 표준화.
n = 4,839 rxn(`rows_rev4.csv`)이다. MAE ± seed 간 sd, NMAE = MAE / 평균절대편차, r².
출처: `rev4_headline.csv` / `.md`.

| 타깃 | MAE | NMAE | r² |
|---|---:|---:|---:|
| ΔE‡ (barrier) | 1.92 ± 0.05 | 0.272 | 0.916 |
| d1 (dipole strain) | 1.77 ± 0.10 | 0.291 | 0.896 |
| d2 (dipolarophile strain) | 1.42 ± 0.07 | 0.304 | 0.875 |
| elst | 3.91 ± 0.14 | 0.291 | 0.897 |
| Pauli | 7.15 ± 0.25 | 0.281 | 0.901 |
| OI | 4.32 ± 0.12 | 0.278 | 0.902 |
| disp | 0.45 ± 0.01 | 0.122 | 0.980 |
| CPCM | 1.56 ± 0.08 | 0.399 | 0.847 |
| CDS | 0.26 ± 0.01 | 0.374 | 0.854 |

### 관찰

1. **MAE가 가장 큰 세 타깃은 Pauli, OI, elst다**(7.15 / 4.32 / 3.91, NMAE 0.281 / 0.278 / 0.291).
   - 대응하는 단일 xTB feature의 pre-ML Pearson r은 `b_pauli` 0.339, `b_oi` 0.362, `b_elst` 0.399다(`rev4_pre_ml.csv`).
   - 해석(가설, 검증하지 않음): G1 TS의 형성 결합 길이는 DFT TS와 중앙값 |Δ| 0.07 / 0.10 Å(짧은 / 긴 결합) 다르다
     (`g1_geometry_summary.json`). 그러면 접촉 거리에 민감한 교환·궤도 항 feature가 라벨을 계산한 기하와 다른 거리에서
     계산된다. 이것이 이 세 타깃 오차의 원인 후보이지만, 이 메커니즘을 따로 시험하지는 않았다.
2. **KRR 사전 고정의 대가(G1 · ESPLEY73).** KRR이 최적이 아닌 타깃은 5개이고, 모두 SVR이 최적이다.
   손실: d2 +0.004, Pauli +0.045, OI +0.004, disp +0.002, CPCM +0.006.

### 부록 (요약; 전체는 `rev4_appendix_by_arm.csv`, `rev4_appendix_by_model.csv`, `rev4_appendix_full.csv`)

KRR(RBF) × arm, test MAE:

| 타깃 | ESPLEY46 | ESPLEY54 | ESPLEY73 |
|---|---:|---:|---:|
| barrier | 2.07 | 1.94 | 1.92 |
| d1 | 1.81 | 1.83 | 1.77 |
| d2 | 1.55 | 1.51 | 1.42 |
| elst | 5.03 | 4.01 | 3.91 |
| Pauli | 8.02 | 7.55 | 7.15 |
| OI | 4.77 | 4.62 | 4.32 |
| disp | 1.35 | 0.47 | 0.45 |
| CPCM | 2.92 | 1.75 | 1.56 |
| CDS | 0.34 | 0.33 | 0.26 |

ESPLEY73 × 모델, test MAE:

| 타깃 | Ridge | KRR | SVR | XGB |
|---|---:|---:|---:|---:|
| barrier | 2.47 | 1.92 | 1.94 | 2.04 |
| d1 | 2.37 | 1.77 | 1.79 | 1.79 |
| d2 | 1.76 | 1.42 | 1.41 | 1.43 |
| elst | 4.80 | 3.91 | 3.93 | 4.35 |
| Pauli | 9.22 | 7.15 | 7.11 | 7.79 |
| OI | 5.67 | 4.32 | 4.32 | 4.62 |
| disp | 0.50 | 0.45 | 0.45 | 0.48 |
| CPCM | 1.79 | 1.56 | 1.56 | 1.66 |
| CDS | 0.36 | 0.26 | 0.26 | 0.28 |

**pre-ML 기준선** (`rev4_pre_ml.csv`): 대응 xTB feature 하나를 그대로 예측값으로 쓴다. 같은 4,839행.
- **MAE:** barrier 7.06, d1 7.64, d2 4.20, CPCM 3.02, CDS 3.49.
- **elst, Pauli, OI는 편향이 커서 MAE가 거의 |bias|와 같다.** 그래서 Pearson r로 본다.
  - MAE: 48.43 / 94.13 / 47.92.
  - 편향: elst와 OI는 양, Pauli는 음(−94.13)이다. OI의 편향은 47.89다.
  - r: elst 0.399, Pauli 0.339, OI 0.362.
- **나머지 타깃의 r:** barrier 0.778, d1 0.840, d2 0.835.
- **disp:** MAE 0.94(r 0.977).

---

## 2. Phase별 기록

### Phase 1-1 — 엔진 smoke test (job 996704 / 996705 / 996711)
대상은 rxn 20, 105와 무작위 3건(2434, 2721, 3452; seed 20260930)이다. 출처는 `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1/_smoke/smoke_report.json`이며,
이 파일은 `results_rev4/` 밖에 있다.

1. **GFN2-xTB와 ALPB(water)가 실제로 적용됐다.** ORCA 출력(`ts.out`)을 그대로 인용한다:
   ```
   Your calculation utilizes the semiempirical GFN2-xTB method
   * xtb version 6.7.1 (edcfbbe) compiled by 'albert@albert-system' on 2024-07-22
   program call               : /home1/yeseo1ee/orca_6_1_1_avx2/otool_xtb ts_XTB.xyz --grad -c 0 -u 0 -P 2 --namespace ts --input ts_XTB.input.tmp --alpb WATER --acc 0.200000
   * Solvation model:               ALPB
   Solvent                        WATER
   Parameter file                 internal GFN2-xTB/ALPB
   :  Hamiltonian                  GFN2-xTB          :
   ```
   ORCA가 부르는 `otool_xtb`는 feature 엔진과 같은 빌드다(xtb 6.7.1, edcfbbe).
2. **`.hess`를 읽을 수 있다.** `$atoms` 블록은 최종 기하를 질량 중심으로 평행이동한 것이다. 이동을 빼면 기하 차이는 최대 2.7e-8 Å이고,
   진동수는 마지막 Freq 블록과 0.005 cm⁻¹ 안에서 같다. 수렴한 4건의 첫 허수 진동수는 −243.5 ~ −329.8 cm⁻¹다.
3. **엔진 일치.** 최적화된 TS에서 xtb 6.7.1 `--grad --alpb water`의 max |g|는 기준(5e-4 Eh/bohr)보다 작다.
   - 수렴한 4건: 105 1.13e-5, 2434 4.53e-5, 2721 3.79e-6, 3452 1.21e-5 Eh/bohr.
   - rxn 20: OptTS가 200 cycle 안에 수렴하지 않았다(max |g| 6.1e-4).

STOP 조건은 해당하지 않아 Phase 1-2로 진행했다.

### Phase 1-2 — G1 기하 전체 (array 996712, 요약 996713, 재실행 996965)

`g1_geometry_summary.json` / `.csv`, 5,260 rxn:

| 항목 | 값 |
|---|---|
| 성공(모든 gate 통과) | **4,860 / 5,260 (92.4 %)** |
| 실패 400건 (gate) | 첫 허수 gate(≤ −40 cm⁻¹) 160, OptTS 미수렴 112, 참조 1 결합 그래프 변화 49, 형성 결합 mode share < 0.5 46, 외부 결합 gate 23, partition 계산 불가 4, 추가 허수 gate(> −50 cm⁻¹) 3, 참조 2 결합 그래프 변화 2, 형성 결합 거리 gate 1 |
| partition 불일치 (G1 ≠ 라벨 `A_idx`) | 0 % |
| TS heavy-atom RMSD, G1 vs DFT (같은 원자 순서) | 중앙값 0.186 Å, 90 % 0.512 Å, 최대 3.064 Å |
| Δd_form, 짧은 / 긴 형성 결합 | 중앙값 +0.025 / +0.058 Å; 절댓값 중앙값 0.073 / 0.100 Å; 절댓값 90 % 0.209 / 0.371 Å |
| 참조 RMSD: rel1 / rel2 | 중앙값 0.098 / 0.032 Å, 90 % 0.246 / 0.197 Å |
| 계산량 | 216.1 core-h |

- 16건의 ORCA 실행은 공유 노드에서 MPI finalize 단계의 bus error(vader BTL)로 죽었다. 모델 학습 전에 같은 입력으로 다시 돌렸고(996965),
  결과는 13건 ok, 2건 OptTS 미수렴, 1건 참조 1 그래프 변화다.
- STOP 판정: 성공률 92.4 % ≥ 90 %, partition 불일치 0 % ≤ 1 %. 따라서 진행했다.

### Phase 2 — feature 계산 (G1 996975, 행 996977)
- **G1 parquet.** 5,260행 가운데 ok 4,860행이고, ok 행의 ESPLEY73 feature에 NaN은 없다.
- **행 (`rows_rev4.json`).** G1 ok 행이 4,860이고 NaN 제외는 0이다. hygiene 규칙으로 21행을 뺐다
  (d1 < 0: 7, d2 < 0: 5, d2 > 50: 9). 최종 n은 **4,839**다.

### Phase 3 — 재학습 (Protocol A, 사전 등록 그대로)
- job:
  - ML smoke 996978: 학습 없음, import·행 게이트만 확인.
  - pack 996988 (G1).
  - array 997145, 집계 997146.
  - 하류 997148 (G1).
- pack 996988은 BLAS 스레드 과다 할당(GridSearchCV worker마다 코어 수만큼 스레드) 때문에 약 2시간 뒤 취소했다.
  - 그 전에 G1의 4개 타깃(barrier, d1, d2, Pauli)이 다중 스레드 BLAS로 끝났다.
  - 나머지 G1 타깃 5개(elst, OI, disp, CPCM, CDS)는 `OMP/OPENBLAS/MKL_NUM_THREADS=1`로 997145에서 다시 돌렸다.
  - 원리상 달라지는 것은 부동소수 합산 순서뿐이다. 다만 단일 스레드 재실행과 비교하지는 않았다(§4-2).
- 원소마다 3 arm × 4 모델 × 5 seed를 돌렸다. seed당 test는 484행이다.

### Phase 3-3 — 하류 분석 (ESPLEY73 · KRR 예측, 행 = `rows_rev4.csv`)

**전하별 test MAE** (`downstream_g1/charge_breakdown_g1.csv`).
- 5 seed test fold의 합집합 1,981 rxn을 썼다: 중성 1,901, 하전 80.
- 하전 80건 가운데 q₂ = −2가 63건, q₂ = +1이 17건이다. q₂는 fragment 2(dipolarophile)의 전하다.

| 타깃 | 중성 | 하전 | q₂ = +1 |
|---|---:|---:|---:|
| barrier | 1.89 | 2.63 | 3.74 |
| d1 | 1.74 | 2.52 | 4.35 |
| d2 | 1.38 | 2.25 | 6.53 |
| elst | 3.79 | 6.63 | 9.45 |
| Pauli | 7.05 | 9.42 | 18.56 |
| OI | 4.25 | 5.89 | 9.97 |
| disp | 0.45 | 0.45 | 0.86 |
| CPCM | 1.46 | 3.79 | 5.31 |
| CDS | 0.26 | 0.24 | 0.46 |

- 하전 반응이 약점이다. q₂ = +1(17건) 쪽의 오차가 특히 크다(d2 6.53, Pauli 18.56).

**그룹 분할** (`downstream_g1/group_split_g1.csv`).
- 방법: seed마다 `GroupShuffleSplit(test 10 %)`로 dipolarophile(dph) 또는 dipole 그룹 전체를 test로 뺐다.
  HP는 헤드라인의 seed별 HP를 그대로 썼다. random 열은 헤드라인 분할이다.
- 표 값: random 열은 MAE, 홀드 열은 random 대비 배수(파일에 저장된 네 자리)다.

| 타깃 | random | dph 홀드 | dipole 홀드 |
|---|---:|---:|---:|
| barrier | 1.92 | 1.2653 | 1.0226 |
| d1 | 1.77 | 1.1572 | 1.0335 |
| d2 | 1.42 | 1.1159 | 1.0212 |
| elst | 3.91 | 1.1681 | 1.0593 |
| Pauli | 7.15 | 1.1524 | 1.0605 |
| OI | 4.32 | 1.1738 | 1.0492 |
| disp | 0.45 | 1.0272 | 1.0632 |
| CPCM | 1.56 | 1.0850 | 1.0270 |
| CDS | 0.26 | 1.1018 | 1.0307 |

- **dipole 홀드.** random 대비 1.0212 – 1.0632배로, 약 1.1배 이내다(타깃 9개).
- **dph 홀드.** random 대비 1.0272 – 1.2653배다(타깃 9개). 절대 증가량 예: barrier +0.51(1.92 → 2.43).

**MMP 쌍** (`downstream_g1/split_metrics_g1.csv`).
- KRR은 α = γ = 1e-3으로 고정했다(튜닝 없음).
- ΔMAE는 쌍 차이의 MAE다. dom_agree는 |Δ|가 가장 큰 채널이 일치하는 비율이다.
- dom_agree 계산에서 disp는 out-of-fold 예측값이다.
- 값은 파일에 소수 셋째 자리로 저장돼 있고, 셋째 자리까지 맞춰 옮겼다.

| 분할 | 쌍 수 | ΔMAE Pauli | ΔMAE elst | ΔMAE OI | dom_agree | τ₉₅ (coverage) |
|---|---:|---:|---:|---:|---:|---:|
| random (5-fold) | 904 | 8.394 | 4.751 | 5.022 | 0.680 | 격자 안에서 없음 |
| dipole_class | 3,757 | 10.825 | 5.795 | 6.558 | 0.583 | 격자 안에서 없음 |
| loso (치환기 10개 = 10행; 2쌍은 같은 행 집합 → 고유 8개) | 86 – 1,078 | 파일 참조 | 파일 참조 | 파일 참조 | 0.654 – 0.791 | 5행 5.5 – 7.0 |

**loso 10행 가운데 2쌍은 같은 행 집합이라 고유 홀드아웃은 8개다.**
- `*C(=O)NC` ≡ `*NC`(671쌍)와 `*C(=O)OC` ≡ `*OC`(660쌍)는 `split_metrics_g1.csv`에서 모든 지표가 같다.
- 원인은 같은 치환기를 다른 단일 결합에서 자른 두 표기가 같은 반응을 고르는 것으로 보인다(`mmp_utils.splits`의 `subs_dip`).
  `evaluate_pairs.py`에서 확인할 항목으로 남긴다.
- τ₉₅에 도달한 loso 5행은 고유 홀드아웃 4개(`*c1ccccc1`, `*C(=O)OC`/`*OC`, `*C(C)=O`, `*[N+]#CC(C)=O`)다.
- loso에서 train 멤버의 예측은 in-sample이라 낙관적이다.

**margin 보정** (`downstream_g1/margin_calibration_g1.csv`).
- 예측 margin ≥ τ인 쌍만 남겼을 때 coverage와 dominant-channel 일치율(agree)을 본다.
- agree(참 margin)은 참 margin으로 같은 τ를 적용한 값이다.

| 분할 | τ | coverage | agree | agree(참 margin) |
|---|---:|---:|---:|---:|
| random | 0 | 1.000 | 0.680 | 0.680 |
| random | 2 | 0.663 | 0.801 | 0.800 |
| random | 4 | 0.486 | 0.870 | 0.876 |
| random | 8 | 0.304 | 0.927 | 0.914 |
| dipole_class | 0 | 1.000 | 0.583 | 0.583 |
| dipole_class | 2 | 0.633 | 0.712 | 0.700 |
| dipole_class | 4 | 0.442 | 0.778 | 0.771 |
| dipole_class | 8 | 0.245 | 0.862 | 0.869 |

- **예측 margin의 agree가 참 margin의 agree를 따라간다.**
  - random과 dipole_class의 τ 격자(0 – 8, 0.5 간격) 전체에서 차이는 0.02 이내다.
  - 그러나 두 분할 모두 τ ≤ 8 안에서 agree 0.95에 도달하지 못한다(최대 0.927 / 0.862). 그래서 `split_metrics_g1.csv`의 τ₉₅ 칸이 비어 있다.

### Phase 4-1 / 4-2 — Espley 비교 (결과는 §3)
- job:
  - Espley 역할 d1/d2 재학습 996868: (a) 그들의 프로토콜 SVR/KRR, (b) 우리 파이프라인 4 모델. 같은 job이 Espley 비교 준비 파일도 썼다.
  - G1 비교 준비 996979.
  - 비교 학습 pack 996989: 우리 G1 × 7 타깃.
  - 채점·표·그림 996995.
- **그들의 프로토콜 재현.**
  - 그들의 index d1/d2에서 다시 튜닝한 SVR·KRR 파라미터가 `hps.pkl`과 같다. 튜닝 열은 47개이고, 그들의 `hyp_tuning.py`를 따른다.
  - 재학습한 test 예측은 그들이 저장한 예측과 ~1e-12 안에서 같다.
  - 그래서 `espley_protocol_replicated = True`다(`espley_compare_checks.json` `their_protocol`).
- **분할 재현.** 그들이 저장한 25개 (모델, 타깃) × 5 seed의 test 타깃이 모두 일치한다(`split_replicated`).

### Phase 4-3 — 기하 정보 절제 (common 996990, own 996991, 집계 996996 / 996997)
- 방법: 튜닝은 seed 23 train에서 한 번 하고, 그 뒤 seed마다 refit한다. 분할은 Espley 방식이다.
- 주 결과는 **모든 arm이 쓸 수 있는 공통 3,327행**이다(`espley_geometry_ablation.csv`).
- KRR(RBF) test MAE:

| arm | d1 (index) | d2 (index) | Interaction | ΔE‡ | ΔG‡ | dipole (역할) | dipolarophile (역할) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: Espley46 (우리 파이프라인) | 2.54 | 2.44 | 2.41 | 3.03 | 2.93 | | |
| B_g1: Espley46 + G1 거리 11 | 2.34 | 2.11 | 2.16 | 2.67 | 2.59 | | |
| C: Espley46 역할 기준 (AM1 변형 feature swap) | | | | | | 2.60 | 1.89 |
| C0: Espley46, feature swap 없이 역할 타깃 | | | | | | 2.75 | 1.99 |
| O_g1: 우리 ESPLEY46 (G1) | 2.72 | 2.69 | 1.63 | 2.06 | 2.06 | 1.67 | 1.44 |

- **G1 거리 11개를 더했을 때의 이득 (A → B_g1).** d1 −0.20, Interaction −0.25, d2 −0.33, ΔG‡ −0.34, ΔE‡ −0.37.
- **O_g1과 §3 주 표의 차이.** O_g1은 seed 23에서 한 번만 튜닝하므로 §3 주 표의 G1 KRR(seed마다 nested)과 조금 다를 수 있다.
  예: ΔE‡ 2.06 vs 2.01.
- **원 규칙 재현 (`ABL_ROWS=own`, `espley_geometry_ablation_ownrows.csv`의 `n_rows` 열).** arm마다 자기 행을 쓴다.
  - AM1 feature만 쓰는 arm(A, C, C0)은 3,509행, G1 arm(B_g1, O_g1)은 3,327행이다.
  - arm A는 d1 2.66, Interaction 2.42, ΔE‡ 3.05로, 사양 §0-3의 로컬 기준값(2.66, 2.43, 3.05)과 같거나 0.01 다르다.

### Phase 4-4 — Espley AM1 TS의 출발 구조 (Bath 996860)
- **아카이브.** BATH-01480 v2(`data_archive_files.zip`, 4,302,424,907 B)에 ds3 AM1 TS 로그 3,663개가 있다.
  BATH-01398 v1은 HTTP 401로 받을 수 없었다. 파일 목록과 로그만 HTTP range 요청으로 받았다.
- **표본.** ML 세트 3,510개 가운데 20건(seed 20260930): 843, 1675, 1816, 2035, 2075, 2173, 2347, 2636, 2695, 3239, 3482, 3554, 4042,
  4182, 4239, 4522, 4564, 4609, 4815, 4947.
- **대응 검증** (`mapping_evidence`): `reaction_number`와 rxn_id가 같고, 로그의 원자 순서가 Coley와 같다. 20건 모두 통과했다.
- 출처는 `espley_am1_start_rmsd.json`(`summary`, `verdict`)과 `espley_am1_start_rmsd.csv`(로그별)다.

| 항목 | 값 |
|---|---|
| Gaussian route (모두 G16 C.01, SMD water) | `opt=(calcfc,ts,noeigen,maxcycles=80,maxstep=3)` 10건, `opt=(calcfc,ts,noeigen,maxcycles=30,maxstep=2)` 10건 |
| 첫 입력 vs Coley DFT TS, mapped heavy RMSD 중앙값 (20건, **규칙상 판정값**) | **0.050 Å** (inf 1건 포함) |
| 같은 값, 유한값 19건의 중앙값 | 4.9e-15 Å |
| 같은 원자 순서 RMSD 중앙값 (20건, inf 없음) | 0.050 Å |
| < 0.01 Å | 10 / 20 |
| AM1 최종 vs DFT TS 중앙값 | 0.356 Å (inf 포함) / 0.355 Å (유한 19건) / 같은 원자 순서 0.356 Å; 최소 0.093 Å |
| 첫 입력 vs AM1 최종 중앙값 | 0.124 Å (< 0.01 Å: 6 / 20) |
| AM1 최종 형성 결합 − DFT, 중앙값 | 짧은 쪽 −0.090 Å, 긴 쪽 −0.225 Å |

**inf 1건.** rxn 4564는 첫 입력과 DFT TS의 heavy-atom 결합 그래프가 동형이 아니어서 mapped RMSD가 inf다.
같은 원자 순서로 재면 0.670 Å다.

**20건은 route로 깨끗하게 둘로 나뉜다.**
- **maxcycles=80 로그 10건: 1단계 AM1 TS 계산으로 보인다.**
  - rxn 843, 1816, 2347, 2636, 3239, 3482, 4042, 4182, 4522, 4609.
  - title이 Coley의 autodE 제목(“Generated by autodE on: 2022-…”)이다.
  - 첫 입력이 Coley DFT TS와 정확히 같다(RMSD 4.5e-16 – 4.9e-15 Å).
  - 30 – 74 step을 돈다.
- **maxcycles=30 로그 10건: 이전 AM1 출력에서 이어 간 재시작으로 보인다.**
  - rxn 1675, 2035, 2075, 2173, 2695, 3554, 4239, 4564, 4815, 4947.
  - title이 `ts_<n>.out`다.
  - 3 – 26 step만 돈다.
  - 첫 입력이 자기 AM1 최종 구조와 거의 같다: 6건은 < 0.01 Å, 8건은 < 0.014 Å. 나머지 두 건(3554, 4239)은 0.136 / 0.268 Å다.
  - 이 로그에서는 원래의 출발 구조를 알 수 없다.
  - 이 10건의 첫 입력은 어떤 Coley 파일과도 0.01 Å 안에서 일치하지 않는다(가장 가까운 Coley 구조까지 0.098 – 1.036 Å).
    이는 재시작 입력이라서이지, AM1이 생성물이나 허수 모드 변위 구조에서 출발했다는 증거가 아니다.

**판정 (사전 등록 규칙 그대로): 중앙값 0.050 Å ≥ 0.01 Å → G1을 “입력 대응 arm”으로 선언하지 않는다.**
- **이 판정은 경계선 위에 있다.**
  - 20개 중앙값은 정확히 10번째 값(4.9e-15 Å, DFT TS 출발)과 11번째 값(0.1008 Å, rxn 4947 재시작 로그)의 한가운데다.
  - inf 1건을 빼고 유한값 19개로 보면 중앙값이 4.9e-15 Å가 되어 규칙을 만족한다.
  - 같은 원자 순서 RMSD(inf 없음)의 중앙값도 0.050 Å이므로 사전 등록 판정은 유지된다. 그러나 로그 한 건이 판정을 가른다.
- **규칙이 실패한 것은 재시작 로그 때문이다. 독립적인 다른 출발 구조의 증거는 아니다.**
  확인할 수 있는 1단계 로그 10건은 모두 DFT TS에서 출발했다.
- `verdict.statement`의 “The start matches none of Coley's files within 0.01 A”는 중앙값에 대한 문장이다. 로그별로는 위와 같이 10 / 10으로 나뉜다.

---

## 3. Espley 2024 대비 — 주 비교 (사전 고정: G1 · ESPLEY46 · KRR 대 Espley SVR)

**비교 조건.**
- 반응: Espley ds3 ML 세트 3,510개 가운데 라벨이 있고 G1을 쓸 수 있는 **3,327 rxn**.
- 타깃: Espley의 DFT 값.
- 분할: Espley 방식, seed 22/23/14/1/2. seed당 채점 test 행은 평균 334이고, 그들의 원래 test는 351행이다.
- 오차 막대: Espley 정의의 SE.
- 역할 기준 d1/d2의 Espley 값은 그들의 프로토콜 (a)로 재학습한 SVR이다. Interaction / ΔE‡ / ΔG‡는 그들이 저장한 SVR test 예측을 같은 행에서 다시 채점했다.
- 출처: `espley_compare_main.csv`, 그림 `espley_compare_main.png`.

| 타깃 | Espley SVR (AM1 기하, AM1 46) | **G1 · ESPLEY46 · KRR** | Espley ÷ G1 | 짝지은 차 (Espley − G1) ± sd | G1이 낮은 seed |
|---|---:|---:|---:|---:|---:|
| Dipole 변형 (역할) | 2.57 ± 0.13 | **1.67 ± 0.08** | 1.540 | 0.90 ± 0.14 | 5 / 5 |
| Dipolarophile 변형 (역할) | 1.91 ± 0.09 | **1.44 ± 0.08** | 1.329 | 0.47 ± 0.10 | 5 / 5 |
| Interaction | 2.45 ± 0.12 | **1.63 ± 0.08** | 1.501 | 0.82 ± 0.09 | 5 / 5 |
| ΔE‡ | 3.04 ± 0.15 | **2.01 ± 0.09** | 1.520 | 1.04 ± 0.16 | 5 / 5 |
| ΔG‡ | 2.97 ± 0.14 | **2.01 ± 0.10** | 1.476 | 0.96 ± 0.11 | 5 / 5 |

pre-ML xTB 기준선(채점 행, G1): dipole 7.87, dipolarophile 4.15, Interaction 5.99, ΔE‡ 6.51. ΔG‡에는 대응 xTB 값이 없다.

1. **사전 고정 비교에서 5개 타깃 모두, 5개 seed 모두 G1이 낮다.** Espley ÷ G1은 1.33 – 1.54다.
   - **Espley 쪽의 가장 강한 역할 모델과 비교해도 G1이 낮다.** 그 모델은 우리 파이프라인 XGB(2.32 / 1.63)이고, G1 KRR은 1.67 / 1.44다
     (`espley_compare_role.csv`, `espley_compare_appendix_best.csv`).
   - 양쪽 best-of 부록(`espley_compare_appendix_best.csv`):
     - G1 최고: ESPLEY46 1.67 / 1.40 / 1.63 / 2.00 / 2.01, ESPLEY73 1.63 / 1.30 / 1.35 / 1.82 / 1.85.
     - Espley 최고: 2.32 / 1.63 / 2.45 / 3.04 / 2.97.
     - 타깃 순서는 모두 dipole / dipolarophile / Interaction / ΔE‡ / ΔG‡다.
2. **두 쪽에서 다른 것.**
   - **기하.** Espley는 AM1로 재최적화한 TS와 GS를 쓴다. G1은 DFT TS와 DFT 참조에서 출발해 GFN2-xTB/ALPB로 재최적화한다.
   - **feature.** AM1 46개 대 xTB `ESPLEY46` 46개.
   - **모델.** SVR 대 KRR.
   - **튜닝·전처리.** 우리는 seed마다 nested GridSearchCV를 돌리고 X·y를 표준화한다.
     Espley 프로토콜은 seed 23 train에서 한 번 튜닝하고(그들의 격자, 47열) X만 표준화한다. 저장 예측은 그들의 원래 모델 출력이다.
   - **학습 행.** Espley 저장 모델과 역할 재학습은 그들의 행으로 학습했다. 우리 모델은 채점 행 3,327개의 train 부분으로 학습했다.
   - **채점 행의 선택.**
     - 채점은 G1이 gate를 통과한 행으로 제한된다. 라벨이 있는 Espley 세트 3,509건 가운데 G1 실패 182건(5.2 %)이 빠졌다
       (첫 허수 gate 76, OptTS 미수렴 44, 참조 1 그래프 변화 28, 형성 결합 mode share 23, 외부 결합 gate 10, 추가 허수 gate 1). 여기에 라벨 없는 1건이 더 빠졌다(`espley_compare_checks.json` `intersection`).
     - 이 필터는 우리 파이프라인에서 나왔고 G1에 유리할 수 있다. xTB 기하가 잘 나오는 반응이 xTB feature도 잘 맞을 가능성이 있다.
     - xTB 기하 모델을 실제로 쓰려면 이 실패들도 처리해야 한다. 따라서 Espley ÷ G1 배수는 G1 성공 부분집합에서만 성립한다.
3. **G1과 Espley의 입력 기하.**
   - Phase 4-4에서 확인할 수 있는 Espley 1단계 AM1 TS 계산 10건은 모두 Coley DFT TS에서 출발했다.
     G1도 DFT TS에서 출발하므로 두 쪽의 출발점이 같을 가능성이 크다.
   - 그러나 사전 등록 규칙은 경계선에서 충족되지 않았다(§2 Phase 4-4). 그래서 G1은 “Espley 비교 arm”으로만 부르고 입력 대응 arm으로 선언하지 않는다.
   - G1은 DFT의 conformer·입체 선택에서 출발하고, 대부분 그 선택을 유지한다(G1 TS vs DFT TS heavy-atom RMSD 중앙값 0.186 Å, 최대 3.064 Å).
     AM1 최종 TS는 DFT TS에서 중앙값 0.356 Å 떨어져 있다(20건; 표본이 달라 참고로만).
   - 지금까지 돌린 arm 가운데서는 G1이 Espley와 가장 가깝게 맞춘 비교다.
     Phase 4-5(Espley AM1 기하 위 xTB feature, 승인 대기)를 돌리면 더 가깝게 맞출 수 있다.
4. **인덱스 기준 d1/d2는 부록이다** (`espley_compare_index_d1d2_appendix.csv`).
   - Espley의 `distortion_energy_1/2` 열은 역할이 아니라 반응물 인덱스를 따른다. 3,509건 가운데 1,102건에서 `_1`이 dipolarophile이다.
   - 역할 배정: 우리 라벨에 가장 가깝게 맞는 쪽을 고른다. 배정 뒤 라벨 차이 MAE는 0.056 / 0.047이다(배정 전 1.79 / 1.79).
   - 역할이 모호한 168행(|d1 − d2| < 0.5 kcal/mol)도 같은 규칙으로 배정했고 따로 빼지 않았다. 이 행들은 잘못 배정돼도 타깃이 0.5 kcal/mol 미만으로만 바뀐다.
   - Espley 자신의 AM1 변형 feature도 같은 인덱스를 따른다(AM1 `_1` 대 DFT `_1` r 0.75, 대 `_2` r 0.35). 반면 우리 xTB feature는 역할 기준이다.
   - 그래서 인덱스 혼합의 영향은 한쪽에만 크다(역할 → 인덱스, test MAE):
     - **G1 KRR:** d1 1.67 → 2.72, d2 1.44 → 2.67.
     - **Espley SVR:** d1 2.57 → 2.53(거의 변화 없음), d2 1.91 → 2.36.
     - 인덱스 타깃에서는 Espley SVR이 G1 KRR보다 낮다. G1이 낮은 seed는 d1 1/5, d2 0/5다.
   - rev 3의 d1/d2 공정성 서술은 철회한다. 인덱스 혼합의 영향이 한쪽에만 크기 때문이다(§5).

---

## 4. 사전 등록·사양과 다른 점

1. **학습 순서: 사전 등록 문구가 사실과 다르다.**
   - job 996868이 PREREG commit(`be0a457b`, 15:47:26)보다 먼저 끝났다.
     - 이 job은 Espley 역할 d1/d2를 (a) 그들의 프로토콜 SVR/KRR과 (b) 우리 파이프라인 Ridge/KRR/SVR/XGB로 재학습했다. HEAD `1ada37aa` 작업 트리에서 돌았고 13:56에 끝났다.
     - 로그(`/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/logs/esp_role.996868.out`)는 모든 모델의 seed별 test MAE를 출력했다(그들의 행, n ≈ 351).
       예: `espley ours dipolarophile_distortion_role XGB seed 22: MAE 1.639 (n=351)`.
   - 따라서 이 결과는 PREREG 작성 전에 볼 수 있었다. PREREG 본문의 “어떤 모델도 학습하지 않았다”와 commit 메시지의
     “before any training”은 이 job에 대해 틀렸다.
   - 주 표의 Espley d1/d2를 (a) SVR로 정한 규칙(PREREG §6)도 이 MAE가 로그에 찍힌 뒤에 commit됐다.
     사양 §4-1 / 4-2는 “Espley SVR”과 (a)(b) 두 재학습을 적었지만, 어느 쪽을 주 표에 쓸지는 정하지 않았다.
   - 선택된 (a) SVR(2.57 / 1.91)은 Espley 쪽에서 가장 강한 모델이 아니다. (b) XGB는 2.32 / 1.63이다.
     G1 KRR(1.67 / 1.44)은 둘보다 모두 낮으므로 §3의 결론은 이 선택에 달려 있지 않다.
   - 같은 job이 Espley 비교 준비 파일도 썼다(`compare/espley/prep.pkl`, 13:43).
   - PREREG보다 먼저 돈 다른 job은 학습이 없는 입력 점검이다: Bath 996860(13:42), G1 비교 준비 996979(15:45), ML smoke 996978.
2. **BLAS 스레드.**
   - G1의 4개 타깃(barrier, d1, d2, Pauli)은 다중 스레드 BLAS로 학습됐다(996988). 나머지 G1 타깃 5개는 단일 스레드로 학습됐다(997145).
   - Phase 4 비교 학습 pack 996989도 스레드 고정(`r4_compare_train.sh` 수정)보다 먼저 돌았다.
   - 원리상 합산 순서만 다르다. 그러나 단일 스레드 재실행과 비교하지는 않았다.
     GridSearchCV에서 점수가 거의 같은 후보가 있으면, 이론적으로 다른 하이퍼파라미터가 뽑힐 수 있다.
3. **절제의 행 기록 파일.** `espley_geometry_ablation_ownrows_rows.json`은 공통 행 기록 `espley_geometry_ablation_rows.json`과 바이트 단위로 같다
   (md5 `3d4c8448…`, `n_common` 3,327, `rows_sha256` `4563c182…`). 즉 원 규칙 행을 따로 기록하지 않았다.
   - 두 파일의 `own` 블록은 arm A의 n을 3,510으로 적는다. 실제 원 규칙 실행(`_ownrows.csv`)은 3,509행을 썼다.
   - 원 규칙 행 수는 `_ownrows.csv`의 `n_rows` 열을 기준으로 한다(§2 Phase 4-3).
4. **MMP loso 중복.** 치환기 10개 가운데 2쌍이 같은 반응 집합이라, 고유 홀드아웃은 8개다(§2 Phase 3-3). 코드 확인이 필요하다.

## 5. rev 3 서술의 정정 (사양 Phase 6)

- **Espley 비교에서 두 쪽의 차이를 feature로만 적은 서술은 정정한다.** 기하도 다르다.
  Espley는 AM1 재최적화 기하이고, rev 4의 우리 쪽은 G1이다(§3-2).
- **rev 3의 d1/d2 공정성 서술은 철회한다.** 인덱스 혼합의 영향은 한쪽에만 크다(§3-4).
- **Espley 비교의 결론은 §3으로 대체한다.** 역할 기준 d1/d2, 사전 고정 모델, G1 기하, G1 성공 행이라는 조건이 붙는다.

## 6. 한계와 사용자 결정 대기

- **G1은 배포 성능이 아니다.** TS와 참조가 라벨의 DFT 구조에서 출발하므로 conformer와 입체 선택은 DFT 정보다.
  DFT 정보 없이 SMILES에서 출발하는 G2가 필요하다.
- **행 선택.**
  - rev 4 행은 G1 gate를 통과한 반응뿐이다(4,860 / 5,260; 실패 400건 = 7.6 %).
  - rev 4의 모든 결과는 이 부분집합에서 나왔다. xTB TS 탐색이 실패하는 반응에서의 오차는 모른다.
  - Espley 비교도 같은 제한을 받는다(§3-2).
- **Phase 4-4 판정은 경계선 위에 있고, 재시작 로그 10건의 원래 출발 구조는 알 수 없다.**
- **loso MMP 지표는 train 멤버의 in-sample 예측을 쓰고, 고유 홀드아웃은 8개다.**
- **사용자 결정 대기 (PREREG §8):**
  - **Phase 4-5:** Espley AM1 기하 위에서 xTB feature를 계산한다. 입력 기하까지 맞춘 비교가 된다.
  - **Phase 5 (G2):** autodE로 SMILES부터 xTB 수준 탐색을 한다. D1 venv(autodE 1.4.5)를 써야 하므로 사용자 결정 뒤에 진행한다.
    D1(`D1_build/`, `d1-build`)은 건드리지 않았다. D1 gate 함수는 `d1-build@d573111e`의 읽기 전용 스냅샷에서 import했다.

## 7. 산출물 (`results_rev4/`)

- **사전 등록·행:** `PREREG_REV4.md`, `rows_rev4.csv`(4,839행), `rows_rev4.json`(단계별 건수, 제외 id).
- **Phase 1:** `g1_geometry_summary.json` / `.csv`(rxn별 status, 허수, RMSD, Δd_form). smoke 결과는 폴더 밖 `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1/_smoke/smoke_report.json`에 있다.
- **Phase 3:** `rev4_headline.md` / `.csv`, `rev4_appendix_by_arm.csv`, `rev4_appendix_by_model.csv`, `rev4_appendix_full.csv`, `rev4_pre_ml.csv`.
  - 그림: `figures/mae_bar_espley73_g1.png`, `figures/scatter_{Ridge,KRR_rbf,SVR_rbf,XGB}_ESPLEY73_g1.png`.
- **Phase 3-3:** `downstream_g1/`: `charge_breakdown_g1`, `group_split_g1`, `split_metrics_g1`, `margin_calibration_g1`, `mmp_pairs_v2_g1`(.csv).
  - `mmp_pairs_v2_g1.csv`의 쌍 목록 (r1, r2)은 4,536쌍이다.
- **Phase 4-1 / 4-2:** `espley_compare_main.csv` / `.png`, `espley_compare_role.csv` / `.png`, `espley_compare_index_d1d2_appendix.csv`, `espley_compare_appendix_best.csv`,
  `espley_compare_per_seed.csv`, `espley_compare_checks.json`(교집합과 제외 id, 라벨 일치, 분할 재현, 인덱스 대 역할, 그들의 프로토콜 재현).
- **Phase 4-3:**
  - 공통 행: `espley_geometry_ablation.csv`, `espley_geometry_ablation_per_seed.csv`, `espley_geometry_ablation_rows.json`(3,327행, sha256 `4563c182…`).
  - 원 규칙 행: `espley_geometry_ablation_ownrows.csv`, `espley_geometry_ablation_ownrows_per_seed.csv`. 행 수는 `n_rows` 열로 본다(A, C, C0 3,509, G1 arm 3,327).
  - `espley_geometry_ablation_ownrows_rows.json`은 공통 행 기록의 사본이다(바이트 동일, §4-3). 원 규칙 행을 기록하지 않는다.
- **Phase 4-4:** `espley_am1_start_rmsd.csv`(로그별: route, title, step 수, RMSD), `espley_am1_start_rmsd.json`(요약, 판정, 아카이브, 대응 검증).
- **job 로그:** `/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/logs/`. scratch 산출물은 `/gpfs/tmp_cpu2/yeseo1ee/{espley_xtb_g1, espley_xtb, espley_rev4}`에 있다.

