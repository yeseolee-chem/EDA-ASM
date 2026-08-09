# PIPELINE — Espley 파이프라인 그대로, 타깃만 2 → 7채널

**원칙** 논문(10.1039/d4dd00224e) 및 레포(the-grayson-group/distortion-interaction_ML)의
구조·모델·평가를 그대로 유지한다. 바꾸는 것은 **타깃 이 초과 그것에 맞는 라벨 단계**,
그리고 Gaussian/AM1이 없으므로 **SQM 층을 GFN2-xTB로 교체**하는 것뿐이다.

전제: ORCA EDA 결과 ~3500개 보유 (eda.inp / eda.out / eda_property.txt 형태).

---

## 1. 채널 정의 (검증 완료)

`eda.out`의 EDA 표에서 Hartree 열로 파싱 (kcal 열은 소수 2자리라 사용 금지).

| 타깃 열 | ORCA 항 | 예시값 (샘플 파일) |
|---|---|---|
| `elst_dft` | Electrostatic Energy | −42.64 |
| `pauli_dft` | Pauli Energy **+ Delta E^0(XC)** | 157.42 − 63.21 = **+94.21** |
| `oi_dft` | Orbital Energy | −63.80 |
| `disp_dft` | Delta Dispersion | −5.77 |
| `cpcm_dft` | Delta CPCM Dielectric | −0.52 |
| `cds_dft` | Delta SMD CDS correction | −2.84 |
| `strain_*_dft` | 별도 SPE (아래 §3) | — |

파생(학습 제외): `eint_dft` = Bond Energy (합 검사용).
하드 게이트: |Bond − Σ7항| < 0.02 kcal/mol (샘플 파일 잔차 −0.0023 ↩).

XC→Pauli 근거: Cárdenas Sabando et al., JCTC 2025, 21, 7920 (10.1021/acs.jctc.5c01003) eq 10.

**컬럼 계약**: 모든 타깃은 `_dft` 접미사. 레포의 `f_select.py`가
`'_dft' in col` 부분문자열로 타깃을 잡으므로 하위 ML 스크립트가 **무수정** 확장된다.

---

## 2. 레포 파일별 처분

| 레포 파일 | 처분 | 이유 |
|---|---|---|
| `diassep.py` | **그대로** | TS→조각 원자 매핑. ds3용 BFS 5원자 라벨링 그대로 사용 |
| `get_energies.py` | **교체** → `orca_eda_parser.py` + `label_builder.py` | GoodVibes CSV 뺄셈 로직 → ORCA 정규식 파싱 + 게이트 |
| SQM 계산층 (Gaussian/AM1) | **교체** → xTB GFN2 | Gaussian 없음. AM1 원리(6열)의 xTB 값 |
| 특징 추출 (Morfeus/cclib) | **부분 교체** | 거리 11개: 기하량이라 그대로. 전하: cclib(Gaussian 파싱) → xTB json/출력 파싱으로 교체 |
| `f_select.py` | 그대로 (+1행 수정) | `VarianceThreshold(0.05)`가 비표준화 특징에서 Mulliken 12개를 삭제하는 버그 → 제거 또는 상대분산 기준으로 교체 |
| `tt_ml/hyp_tuning.py` | 그대로 (+그리드 수정) | ε 상한 1.0인데 이 왔에서 `cpcm_dft`=−0.52 → **튜너가 채널을 통째로 잘라냄**. 채널별 ε ∈ [0.01, 20] 로그스케일 필요 |
| `tt_ml/ml_analysis.py` | **그대로** | `_dft` 계약으로 자동 확장 (타깃 6→N) |
| 설정 yaml | 경로만 수정 | — |

모델 셋(ridge / KRR / SVR / NN 2·4층), 5-seed 평균, test MAE as % of range 보고 → 전부 논문 그대로.

---

## 3. strain — 유일하게 재계산이 필요한 곳

`eda.out`에 조각 SPC 에너지가 **인쇄되지 않는다** (`SPC Fragment 1 .... done`뿐).
단, 역산이 가능하다:

```
E(fragA_dist) + E(fragB_dist) = E(AB) − BondEnergy        ← 필요값 이미 계산
```

따라서 선택지:

| 방식 | 추가 계산 | 얻는 것 |
|---|---|---|
| (a) 이완 조각 SPE ×2 만 | 반응당 2 SPC (~7000개, EDA 대비 훨씬 작음) | `strain_sum_dft` 만 |
| (b) (a) + 변형 조각 SPE ×2 | 반응당 4 SPC | `strain_di_dft`, `strain_dp_dft` (논문의 distortion_1/2에 1:1) |
| (c) 아카이브에 `eda_frag1/2.out`이 남아 있으면 | **0** | (b)와 동일 |

**첫 액션: 3500개 아카이브에 조각 출력 파일이 보존돼 있는지 확인** → 있으면 (c), 없으면 (b) 권장.
논문은 조각별로 나눠 예측하므로 (b/c)가 "논문 그대로"에 부합. 이 경우 타깃 열은
strain 2개 + 상호작용 6개 = 8이 되고, "7채널"은 strain을 합산 보고할 때의 집계다.
주의: 조각 1/2 → dipole/dipolarophile 역할 매핑은 diassep 결과와 단일 소스로 관리
(샘플 파일: frag1 = CCOCNOHHH 9원자, frag2 = CCCHHHH 7원자 → N,O 포함 조각이 dipole).
SPE 레벨은 EDA와 동일: B3LYP D3BJ def2-TZVP CPCM(water)+SMD, TightSCF.

---

## 4. 실행 순서

```
P1. 파서 일괄 실행 (준비 완료, 즉시)
    python orca_eda_parser.py --batch <root> --glob '*/eda.out' --out labels_6ch.parquet
    → ok/FAIL 집계, 합-일관성·정상종료·loose-SCF 카운트가 열로 기록됨

P2. 조각 출력 아카이브 확인 → strain 방식 (a/b/c) 확정 → strain SPE 제출

P3. diassep.py로 3500개 원자 매핑 → 5원자 반응중심 확정

P4. xTB GFN2: 반응당 5구조 (이완 di/dp, 변형 di/dp, TS)
    → 논문 41특징 (거리 11 + 전자 30, AM1→GFN2) + AM1 6열 원리의 xTB 값

P5. labels + features 병합 → manual_tt 형식 pkl → f_select(수정후) → hyp_tuning(ε 그리드 확장) → ml_analysis
    → 논문 Table 1 형식으로 채널별 pre-ML MAE / SVR test MAE / % of range
```

P1의 부산물로 **loose-SCF 발생률**이 3500개 전 샘플에 대해 공짓로 나온다
(샘플 파일 1건 발생) → 비율이 높으면 S01 수렴 감사를 그 부분집합에 집중.

---

## 5. 미결정 (착수 전 확정 필요)

- **D-APT**: 논문 특징에 APT 유래 15개. Gaussian 없으니 (i) xTB Hessian 15,000회 또는 (ii) GFN2 원자 쌍극자로 대체. 권장 (ii) → "논문 그대로"에서 유일하게 벗어나는 지점이므로 논문에 명시.
- **부호·역할 규약 문서**: frag→di/dp 매핑, strain 부호(+), 채널 부호를 한 표으로.
