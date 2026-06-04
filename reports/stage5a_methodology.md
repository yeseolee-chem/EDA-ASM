# Stage 5-A: 500-반응 표본 추출 및 Fragmentation 방법론

생성: 2026-05-12 (Asia/Seoul)
최종 수정: 2026-05-13 (Bickelhaupt-aligned P2 sub-classification, top-tier journal alignment)
범위: ASM-EDA ground truth 계산을 위한 Halo8 데이터셋 표본
관련 파일: [`stage5a_500_sample.xlsx`](stage5a_500_sample.xlsx)

---

## 0. 한눈에 보기

- **모집단**: Halo8 데이터셋 (총 **19,176 reactions**, 약 20.1M frames)
- **최종 표본**: **500 reactions** = 초기 400 + 추가 100
- **분류 체계**: 6개 패턴 (P0_BIMOL / P1_OPEN / P2_CLOSED / P3_TETHER / P4_DISSOC / P5_HSHIFT)
- **P2_CLOSED sub-classification**: P2A_PI_CYCLO (π→π closed-shell) vs P2B_SIGMA_REARRANGE (σ-skeletal, doublet radical fragments) — Fernández & Bickelhaupt 2014 §4.2에 기반
- **Ground truth**: 사용자 검토를 거쳐 accept된 400 reactions (잠금, byte-identical 보존)
- **검증**: 패턴 분포가 모집단과 통계적으로 유사 (rare-pattern은 의도적 floor)
- **Contribution**: D2AF (Chung et al., *Chem. Sci.* 2025, DOI: 10.1039/d4sc07226j)가 distortion-only fragmentation을 자동화한 데 비해, 본 연구는 **interaction energy 채널까지** 자동화하는 6-pattern decision tree와 σ/π aware sub-classification을 제안.

---

## 1. 모집단: Halo8

| 항목 | 값 |
|---|---|
| 데이터셋 출처 | Lee et al., *Sci. Data* **2025**, DOI: 10.1038/s41597-025-05944-3 |
| 저장 형식 | ASE SQLite DB 10개 (`Halo_1.db` ~ `Halo_10.db`) |
| 총 frames | 20,116,288 |
| 총 trajectories (= reactions) | **19,176** |
| 평균 frames / trajectory | ~1,050 |

### 카테고리 분포
| Source | 궤적 수 | 비고 |
|---|---:|---|
| T1x | 9,835 | Transition1x 기반 유기 반응 (할로겐 없음) |
| Halo_F | 3,317 | 불소 치환 |
| Halo_Cl | 3,247 | 염소 치환 |
| Halo_Br | 2,777 | 브로민 치환 |

### Trajectory key
- 각 frame: `dand_id = "<source>_<formula>_rxn<NNNNN>_<frame_idx>"`
- 예: `Halogen_C4FH5N2O_rxn14045_17`
- **Reaction id (trajectory id)** = `dand_id.rsplit("_", 1)[0]`
- 5점 IRC 추출 시 R / TS / P frame:
  - R = `frame_index_first` (보통 0)
  - TS = `ts_frame_idx` (interior maximum-energy frame)
  - P = `frame_index_last` (last frame)

---

## 2. 초기 400 표본 추출 (Phase 1)

`configs/phase1.yaml` 및 `src/eda_asm/phase1/stage_3_3_sampling.py` 참조.

### 2.1 Stratification 축

| 축 | 구간 | 비율 |
|---|---|---|
| **Source** | T1x : Halo_F : Halo_Cl : Halo_Br | 190 : 70 : 70 : 70 |
| **Heavy atom count** | 5 / 6 / 7 / 8 | 균등 (25%씩) |
| **Bond change bin** | "2–3" / "4–6" | 60% : 40% |
| **Eₐ tertile** | low / mid / high | 균등 (33%씩) |

### 2.2 Filter

- `min_frames`: 20 frame 미만 trajectory 제외
- `interior_ts`: TS frame이 trajectory 내부 (R/P가 양 끝점)
- TS 에너지가 R, P 양쪽보다 높을 것

### 2.3 Random seed: **42**

### 2.4 Cell 정의
`{source} | h={n_heavy_atoms} | bond={bond_change_bin} | ea={ea_tertile}` 의 4축 cell 단위로 추출.

---

## 3. Stage 5-A Classifier — Decision Tree

`src/eda_asm/stage5a/classify.py` 참조. **결합 인식 (connectivity)** 은 RDKit `DetermineBonds` 대신 **거리 기반 covalent-radius 검출** (`stage0_fragmentation.bond_detection.detect_bonds_strict`) 사용 — TS의 stretched bond에 강건함.

**Bond order assignment (P2 sub-classification 용)**: R geometry는 stable minimum이므로 stretched bond 문제가 없다. R-frame에 한해 RDKit `rdDetermineBonds.DetermineBonds(mol, charge=0)`를 적용하여 단일·이중·삼중·방향족 결합 차수를 추론한다. RDKit이 실패한 경우 fallback으로 모든 결합을 single로 가정하고 `bond_order_fallback=True` 플래그를 기록 (보수적 가정: P2B로 분류될 가능성이 높아짐 → reviewer가 수동 확인 가능). Aromatic 결합은 π-함유로 취급 (즉 σ-only가 아님).

### Decision Tree

```
Step 1: 반응물 결합 그래프의 connected components ≥ 2?
    YES → P0_BIMOL
    NO  → Step 2

Step 1b: bond change 0?
    YES → P0_BIMOL (단순 conformer)
    NO  → Step 2

Step 2: 생성물 결합 그래프의 connected components ≥ 2?
    YES → P4_DISSOC
    NO  → Step 3

Step 3: H 또는 단가 할로겐 (F/Cl/Br/I, R-degree=1) 이동 검출?
    (= broken ∩ formed 에 속하는 H 또는 halogen 원자)
    YES → P5_HSHIFT
    NO  → Step 4

Step 4: core_atoms = bonds_broken ∪ bonds_formed 의 induced subgraph가
        몇 개의 connected component를 갖는가?

    1 component:
        |core_atoms| ≤ 2 AND n_changes ≤ 2  →  P1_OPEN
        else                                →  P2_CLOSED + P2 sub-classification

    ≥ 2 components:
        조건 A: 양 component 모두 ≥ 2 atoms
        조건 B: 양 component 모두 bonds_broken 의 atoms와 공통 원자 보유
        조건 C: A ∧ B 만족하고 tether path 존재

        모두 만족 → P2_CLOSED + P2 sub-classification (fresh 분류기는 P3_TETHER를 emit 하지 않음)
        그 외     → P2_CLOSED + P2 sub-classification

Step 4b (P2_CLOSED 전용 sub-classification):
    bonds_broken 중 σ-only single 결합 (R-frame 차수 = 1) 이 있는가?
        YES → P2B_SIGMA_REARRANGE  (doublet radical fragments, mult=2)
        NO  → P2A_PI_CYCLO         (closed-shell singlet, mult=1)
```

**조건 B의 근거 (bookmark review에서 추가됨)**: 한쪽 component가 broken bond를 전혀 포함하지 않으면 그것은 단순한 new-bond acceptor(목적지)일 뿐 진정한 reactive partner가 아님. 이런 경우는 P3 cycloaddition이 아니라 group migration에 가까우므로 P2로 회송.

**P3_TETHER 폐기의 근거 (2026-05-13 review round 3)**: Halo8에 분류기를 적용한 결과, "≥ 2 cores + tether" 조건을 통과하는 fresh-classifier 케이스는 거의 모두 σ-skeletal rearrangement (Cope, Claisen, [3,3]-sigmatropic 류, 자주 bicyclic 또는 strained-ring 환경) 였다. 이들은 Fernández & Bickelhaupt 2014 §4.2의 doublet-radical 처리 (P2B)가 화학적으로 맞는 분류이며 closed-shell singlet + tether (Houk-style intramolecular cycloaddition) 가 아니다. 따라서 fresh classifier는 P3_TETHER를 더 이상 emit 하지 않고 모두 P2_CLOSED로 보내 σ/π aware P2 sub-classification으로 처리한다. P3_TETHER 라벨은 **legacy ground-truth** (사용자가 이전 분류기 출력을 accept한 케이스) 에만 잔존한다.

**Step 4b의 근거 (Fernández & Bickelhaupt, *Chem. Soc. Rev.* 2014, §4.2)**: 저자들은 σ-skeletal rearrangement (Cope, Claisen 등 [3,3]-sigmatropic 류) 및 그와 유사하게 σ-bond 절단을 수반하는 unimolecular 변환에 대해 **doublet radical fragment**를 ASM 분석에 사용한다 — 끊어지는 σ-bond의 두 끝을 각각 doublet radical state로 두고, unpaired electron은 끊어진 결합 축 위의 **σ-type sp² hybrid orbital** (즉 σ-sp², not sp²-π)에 위치시킨다. 이것이 σ-bond cleavage가 동반된 unimolecular 변환에 대한 Bickelhaupt-style 정통 처리이다.

반면 Diels–Alder, [3+2] 등 π→π 재구성 (broken bond가 모두 π-component)은 Vermeeren et al. (*Nat. Protoc.* 2020, DOI: 10.1038/s41596-019-0265-0)에서 명시적으로 **closed-shell singlet** fragment로 처리된다.

따라서 broken-bond의 σ/π 성격에 따라 P2A_PI_CYCLO (closed-shell) vs P2B_SIGMA_REARRANGE (doublet)를 구분하는 것이 두 표준 절차의 자연스러운 통합이다.

---

## 4. Migration 분류 (P5의 메타데이터)

`detect_migrating_atoms`는 `bonds_broken ∩ bonds_formed`에 속하는 모든 원자를 4가지 범주로 분류:

| 범주 | 정의 | P5 fragment로 분리? |
|---|---|---|
| **h_migrating** | 원자번호 = 1 (H) | ✓ |
| **halogen_migrating** | 원자번호 ∈ {9, 17, 35, 53} 이고 R-degree = 1 | ✓ |
| **polyvalent_migrating** | R-degree ≥ 2 이고 모든 R 결합이 broken (full partner swap, 예: 양가 O가 두 C 사이를 이동) | ✗ (메타데이터만) |
| **rearranging** | R-degree ≥ 2 이고 부분적인 결합 변화 (예: C 원자가 한 partner를 잃고 다른 partner를 얻음) | ✗ (메타데이터만) |

Polyvalent / rearranging 원자는 별도 fragment로 분리하지 않고 scaffold에 포함되며, 리뷰어가 메타데이터에서 확인 가능.

---

## 5. Pattern별 Fragment 처리 규칙

### P0_BIMOL — 명시적 두 반응물
- 각 반응물 분자를 그대로 fragment로 분리
- 모두 closed-shell singlet (mult=1)
- Cap 없음

### P1_OPEN — 단일 결합 cleavage
- 유일 reactive bond에서 절단
- 두 doublet radical fragment (mult=2 each)
- Cap 없음
- 신뢰도 1.0 (단일 변화) 또는 0.7 (다중 변화 fallback)

### P2_CLOSED — 다중 결합 동시 변화 (서브-분류 적용)

P2_CLOSED는 broken bond의 σ/π 성격에 따라 두 sub-pattern으로 분리한다 (Bickelhaupt 2014 §4.2 근거):

#### P2A_PI_CYCLO — π-cycloaddition (Diels–Alder, [3+2], …)
- **조건**: bonds_broken에 σ-only single 결합이 **없음** (모두 차수 ≥ 2 또는 aromatic)
- 모든 reactive bond (broken ∪ formed)를 R-skeleton에서 제거
- 가장 큰 2개 connected component → `reactive_A`, `reactive_B`
- 두 fragment 모두 **closed-shell singlet (mult = 1)**
- 신뢰도 0.8 (정상) 또는 0.6 (stray 흡수 발생)

#### P2B_SIGMA_REARRANGE — σ-skeletal rearrangement (Cope, Claisen, electrocyclic, …)
- **조건**: bonds_broken에 σ-only single 결합 **≥ 1개**
- 동일하게 모든 reactive bond를 제거하고 두 component로 분할
- 두 fragment 모두 **doublet radical (mult = 2)** — Bickelhaupt-style σ-bond cleavage 처리
- 신뢰도 0.6 (multi-reference 위험 명시; UDFT 또는 broken-symmetry DFT 권장)

#### Stray-component absorption (둘 다에 공통)
- 추가로 발생한 작은 component는 R geometry에서의 **minimum atom-atom 거리** 기준으로 가까운 쪽에 흡수 (`absorb_stray_components` 함수). 흡수가 발생하면 confidence를 한 단계 demote.

#### Ring rearrangement fallback
- 어떤 cut으로도 분리되지 않으면 (ring topology) 단일 whole 분자 + `confidence = 0` (수동 검토 필요).

#### Ground truth 호환성
- 기존 accepted ground truth가 P2_CLOSED에 mult=1을 부여한 케이스에 대해 새 분류기가 P2B로 판단하면 (즉 σ-only broken이 검출되면), **ground truth가 우선**한다 (mult=1 유지). 분류기의 권고는 [`outputs/stage5a/p2_subtype_audit.json`](../outputs/stage5a/p2_subtype_audit.json)에 별도 기록되어 reviewer가 manual override를 결정한다.

### P3_TETHER — legacy 패턴 (fresh classifier에서는 더 이상 emit 하지 않음)

P3_TETHER는 원래 spec (v0) 에서 Houk-style intramolecular π-cycloaddition (diene + dienophile + σ-tether, 모두 closed-shell singlet) 을 캡처하기 위해 도입되었다. 그러나 Halo8의 fresh-classifier 적용 결과 (2026-05-13 review round 3), "≥ 2 cores + 짧은 tether" 조건을 통과하는 실제 케이스는 거의 모두 σ-skeletal rearrangement였다. 이들은 Fernández & Bickelhaupt 2014 §4.2의 doublet-radical 처리 (P2B) 가 canonical하며 closed-shell + tether 처리가 아니다.

**결정**: fresh classifier는 P3_TETHER를 더 이상 emit 하지 않는다. 모든 would-be-P3 케이스는 **P2_CLOSED + σ/π aware sub-classification** 으로 회송되어:
- 모든 broken bond가 π (order ≥ 2) → P2A_PI_CYCLO (mult=1)
- broken bond에 σ-only single이 있으면 → P2B_SIGMA_REARRANGE (mult=2 doublet)

P3_TETHER 라벨은 [`accepted_ground_truth.json`](../outputs/stage5a/accepted_ground_truth.json) 의 legacy 케이스 (사용자가 이전 분류기 출력을 accept했던 P3 cases) 에만 보존되어 있다. Stage 6 GPR features에서는 P3 one-hot은 0이 되지 않도록 legacy 케이스 호환성을 유지하나, 새로운 분류는 모두 P2A/P2B로 표현된다.

**다음 단계**: 향후 진짜 intramolecular π-cycloaddition (broken-bond 없이 formed-bond만 생기는 Diels-Alder 류)이 필요해지면 별도 P_CYCLOADD 패턴으로 도입을 검토할 수 있다. 본 Halo8 표본에서는 그런 케이스가 거의 없어 P3 retiring 결정에 영향이 없다.

### P4_DISSOC — 생성물 분해
- 생성물 bond graph의 각 connected component가 fragment
- **Small molecule merge** (1.5 Å threshold): product geometry에서 단일-원자 fragment 쌍 (예: H₂, HF, HCl, HBr)을 자동 병합. RDKit `DetermineBonds`가 H–H 결합을 종종 놓치는 문제를 거리 기반 후처리로 복구.
- 멀티플리시티는 electron parity (charge = 0 가정)
- 신뢰도 0.95
- **Radical-pair demotion**: 모든 fragment의 mult ≥ 2 인 경우 (homolytic dissociation), 단일 whole 분자로 demoted + `confidence = 0` + `demoted_reason: homolytic_dissociation_multireference` + `recommended_method: CASSCF_or_NEVPT2` (메타데이터 audit trail).

#### P4_DISSOC의 이론적 정당화

P4_DISSOC는 **microscopic reversibility 원리**에 기반한다. ASM-EDA는 IRC 좌표에 대해 시간 가역적이므로 (Vermeeren et al. 2020, §"Calculating PES"; DOI: 10.1038/s41596-019-0265-0), product → reactant 방향의 분석은 reactant → product 방향의 분석과 동등한 정보를 제공한다.

Cycloreversion이나 dissociation 반응의 경우, reactant는 단일 분자이지만 product가 두 개의 분리된 분자이므로, **product-side fragmentation**이 자연스러운 bimolecular ASM 분석을 가능하게 한다. 이 경우 fragment 정의는 product의 connected components와 정확히 일치하며, 추가적 cut 결정이 필요 없다 (highest confidence 0.95).

**한계점**: 모든 fragment가 multiplicity ≥ 2 인 경우 (homolytic dissociation)는 single-determinant KS-DFT EDA로 정확히 기술되기 어렵다 (multi-reference 효과). 이 경우 fragment를 단일 whole 분자로 demote하고 confidence=0으로 표시하여 수동 검토 (CASSCF 또는 NEVPT2 권장)를 강제한다.

### P5_HSHIFT — 단가 원자 이동
- 각 H 및 단가 halogen migrant이 독립 doublet fragment (mult = 2)
- 역할 이름: `migrating_H`, `migrating_F`, `migrating_Cl`, `migrating_Br`, `migrating_I`
- Scaffold = 나머지 모든 atom, multiplicity는 electron parity로 결정
- 신뢰도 0.95

---

## 6. 추가 100 표본 추출 (Addendum)

### 6.1 동기

초기 400 표본의 통계 검정 결과:
- χ² goodness-of-fit: χ² = 14.03, dof = 3, **p = 0.0029** → 모집단과 유의한 차이
- 주요 편향: **T1x P2_CLOSED 16.3pp 결핍** (모집단 25.8% vs 표본 9.5%)
- Rare pattern (P0/P1/P3) 거의 미포함 → 모델이 해당 fragmentation scheme 학습 불가

### 6.2 Allocation (총 +100)

| Pattern | 추가 수 | 출처 | 근거 |
|---|---:|---|---|
| **P0_BIMOL** | 8 | 모집단 random | Rare-pattern **floor** (모집단 0.08%) |
| **P1_OPEN** | 8 | 모집단 random | Rare-pattern **floor** (모집단 0.11%) |
| **P3_TETHER** | 20 | 모집단 random | Rare-pattern **floor** (기존 4개를 24개로) |
| **P2_CLOSED** | 32 | T1x 28 + 기타 4 | T1x **deficit fix** (16.3pp 격차 축소) |
| **P4_DISSOC** | 10 | 모집단 random | Proportional fill |
| **P5_HSHIFT** | 22 | 모집단 random | Proportional fill |

Random seed: **4242** (`scripts/sample_addendum_100.py`)

### 6.3 다축 stratification 권고 근거

ASM-EDA ground truth로서의 표본 설계는 다음 다축 권고에 따랐다:

1. **1차 축 — Pattern (P0–P5)**: 각 패턴이 본질적으로 다른 fragmentation logic을 사용 (multiplicity, fragment 수, EDA 채널 계산 방식 모두 상이). GPR feature에 `pattern_one_hot`이 들어가는 이상, ARD가 패턴별 lengthscale을 학습하려면 패턴별 데이터 셀이 채워져 있어야 함.
2. **2차 축 — n_bond_changes**: strain 크기의 일차 결정 변수. 패턴 내부에서도 EDA 채널 분포의 분산을 확보.
3. **3차 축 — Heavy atom count**: dispersion(E_disp)·long-range orbital(E_orb)의 결정 변수. AIMNet2 Stage 1 모델과의 size-extrapolation에도 영향.
4. **Eₐ는 stratification 축이 아닌 검증 축으로**: 사후에 분위수 분포가 모집단과 일치하는지 확인.

500-sample 단위에서 다축 셀이 너무 잘게 쪼개지지 않도록, 패턴을 1차 축으로 고정하고 다른 축은 (T1x P2의 sub-source 선별을 제외하고는) random fill로 처리.

---

## 7. 검증 — 500 표본 vs 모집단

### 7.1 전체 패턴 분포

| Pattern | 모집단 % | 500 % | Δ pp | 평가 |
|---|---:|---:|---:|---|
| P0_BIMOL | 0.08% | 1.60% | +1.52 | 의도적 floor |
| P1_OPEN | 0.11% | 1.60% | +1.49 | 의도적 floor |
| P2_CLOSED | 20.84% | 19.60% | **−1.24** | sampling noise 이내 (이전 −6.84pp → 개선) |
| P3_TETHER | 0.32% | 2.80% | +2.48 | 의도적 floor |
| P4_DISSOC | 25.94% | 26.20% | +0.26 | 거의 완전 일치 |
| P5_HSHIFT | 52.71% | 48.20% | −4.51 | rare-pattern oversampling의 자연스러운 trade-off |

### 7.2 Source별 (참고)

각 source 안에서도 패턴 비율이 모집단과 ±5pp 이내로 추적됨 (T1x의 P2 격차는 16pp → 6pp 미만으로 축소).

### 7.3 Ground truth 안정성

`outputs/stage5a/accepted_ground_truth.json`에 400개 accepted reactions의 fragmentation 결과가 byte-identical로 snapshot되어 있음. Classifier 코드 업데이트 후 재실행 시에도 자동 복원되어 ground truth는 변하지 않음.

---

## 8. Decision Tree의 점진적 정제 — Methodology Validation

Top-tier journal의 reproducibility 기준을 만족하기 위해, 본 classifier는 7단계의 patch (v0 → 현재)를 거치며 **bookmark-driven failure mode analysis**를 적용하였다. 각 patch는 이전 버전에서 fail한 specific reaction을 ground truth로 삼아 decision rule을 정제하였다 (단순 changelog가 아니라 **methodology validation evidence**).

| 버전 | 주요 변경 | 검증 근거 |
|---|---|---|
| v0 | 4 패턴 (P0–P3) | 초기 spec |
| v1 | +P4 (product dissoc), +P_STRAIN_ONLY | E2 elimination HBr ⓘ HX extrusion 실패 사례 |
| v2 | P5_HSHIFT 분리, monovalent + balanced 조건 | H transfer가 multireference로 잘못 라우팅된 사례 |
| v3 | rearranging atom 개념, balanced 조건 | C 원자 partner-swap이 false migration으로 잡힌 사례 |
| v4 | 비대칭 balance (net loss만 demote), small-mol merge | H₂ extrusion이 분리된 H radical로 잡힌 사례 |
| v2-clean | P_STRAIN_ONLY 폐지, 단순화된 4-step decision tree | 과도한 보수 fallback이 false negative 양산 |
| **현재 (v2-clean + bookmark review + σ/π aware)** | halogen migration → P5, polyvalent/rearranging은 메타데이터, P3 양쪽 cores 모두 broken bond 필요, **P2A/P2B sub-classification** (Bickelhaupt 2014 §4.2) | T1x σ-skeletal rearrangement가 closed-shell로 잘못 처리된 44개 사례 |

각 patch의 상세 spec과 trigger case는 supplementary `stage_5A_patch_v{1,2,3,4}.md` 및 `stage_5A_v2_clean.md` 파일 참조. 현재 분류기는 모든 trigger case를 통과하며, `accepted_ground_truth.json` 400 reactions에 대해 byte-identical 보존된다.

---

## 9. 파일 구조

```
outputs/phase1/
    selected_reactions.csv          ← 500 entries (header + 500 rows)
    additional_selected.csv         ← 100 addendum entries

outputs/stage5a/
    fragmentation_summary.json      ← per-reaction classifier 출력 (500)
    accepted_ground_truth.json      ← 400 accepted reactions 잠금
    population_classification.json  ← 모집단 전체 19,176개 분류
    distribution_compare.json       ← 모집단 vs 표본 통계
    addendum_100_summary.json       ← +100 추출 audit trail
    review_log.json                 ← UI 리뷰 상태 (400 accepted + 100 not_reviewed)
    review_audit.json               ← 리뷰 action 시계열 로그
    frames_cache.pkl                ← R/TS/P frame 캐시 (Halo8 DB 스캔 회피용)
    per_reaction/<rxn_id>/
        result.json                 ← 상세 fragmentation 결과
        R.xyz, TS.xyz, P.xyz        ← 전체 분자 geometry
        <role>_R.xyz, <role>_TS.xyz, <role>_P.xyz   ← fragment별 geometry

reports/
    stage5a_500_sample.xlsx         ← 본 표본 (이 문서의 대상)
    stage5a_methodology.md          ← 본 문서
```

---

## 10. 참고문헌

### Verified (DOI resolves — `curl -sI https://doi.org/<DOI>` 확인)

| # | 출처 | DOI | 역할 |
|---|---|---|---|
| 1 | Lee et al., *Sci. Data* 2025 | [10.1038/s41597-025-05944-3](https://doi.org/10.1038/s41597-025-05944-3) | Halo8 데이터셋 |
| 2 | Fernández & Bickelhaupt, *Chem. Soc. Rev.* 2014 | [10.1039/c4cs00055b](https://doi.org/10.1039/c4cs00055b) | Fragmentation 일반 원리; **§4.2 — P2A/P2B sub-classification의 이론적 근거** |
| 3 | Fernández, Bickelhaupt, Cossío, *Chem. Eur. J.* 2014 | [10.1002/chem.201303874](https://doi.org/10.1002/chem.201303874) | Doublet fragment scheme (P1/P5/P2B의 이론적 backbone) |
| 4 | Bickelhaupt & Houk, *Angew. Chem. Int. Ed.* 2017 | [10.1002/anie.201701486](https://doi.org/10.1002/anie.201701486) | Tether scheme (P3); intramolecular cycloaddition ASM 적용 사례 |
| 5 | Vermeeren et al., *Nat. Protoc.* 2020 | [10.1038/s41596-019-0265-0](https://doi.org/10.1038/s41596-019-0265-0) | ASM-EDA 일반 protocol; π→π closed-shell ASM 정설; microscopic reversibility (P4_DISSOC 정당화) |
| 6 | Fernández, Cossío, Sierra, *Chem. Rev.* 2009 | [10.1021/cr900209c](https://doi.org/10.1021/cr900209c) | Dyotropic 반응 종설 — P5_HSHIFT/P2B의 이론적 backbone (특히 type-I dyotropic의 σ-bond migration analog) |
| 7 | Chung group, D2AF, *Chem. Sci.* 2025 | [10.1039/d4sc07226j](https://doi.org/10.1039/d4sc07226j) | Fragmentation 자동화 선행 연구 (distortion-only); **본 연구의 contribution context** (interaction energy 채널까지 자동화) |

**Note**: Intramolecular cycloaddition에 대한 Houk group의 ASM 응용 사례는 본 문서 작성 시점에서 신뢰할 수 있는 DOI를 자동 검증하지 못해 의도적으로 제외하였다 (modification 문서가 제시한 후보 DOI가 publisher resolver에서 404 반환). 필요 시 출판 단계에서 도서관 access를 통해 출처를 확정하여 보완할 것. 단, 본 분류기의 이론적 정당화는 #2~#6 (모두 verified)만으로 충분히 성립한다.

---

## 11. 다음 단계 (제안)

1. **Inverse-propensity weighting**: Stage 6 GPR 학습 시 표본 점유율의 역수 가중치 적용 → 잔여 표본 편향 보정 (DFT 재계산 없이)
2. **ADF EDA-NOCV 실행**: 500 reactions × 평균 4 SP 계산 ≈ 2,000 ADF jobs
3. **Group migration 확장 (옵션)**: OH/NH₂ 그룹 이동을 P5의 일반화로 처리하는 새 패턴 (현재는 polyvalent_migrating으로 메타데이터에만 표시)
