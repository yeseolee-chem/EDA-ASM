# REV5 — G0 폐기, 채널 feature 확장, Espley 비교 그림

**대상:** `main` `740ba89`, `analysis/espley_xtb_repro/`
**결과 위치:** 새 폴더 `analysis/espley_xtb_repro/results_rev5/`
**보고:** Phase마다 보고하고, STOP 조건이 나오면 다음 Phase로 가지 않는다.

**순서:** Phase A(G0 폐기) → B(새 feature) → C(사전 등록과 선별) → D(최종 평가) → E(그림) → F(문서)

(사용자 제공 명세, 2026-09-30. 원문 그대로 보관.)

---

## 0. 배경 — 확인된 사실 (2026-09-30)

### 0-1. rev 4의 G1 헤드라인 (`results_rev4/rev4_headline.csv`, ESPLEY73 · KRR, n = 4,839)

| 타깃 | MAE | NMAE | pre-ML 단일 feature의 Pearson r |
|---|---:|---:|---:|
| Pauli | 7.15 | 0.281 | `b_pauli` 0.339 |
| OI | 4.32 | 0.278 | `b_oi` 0.362 |
| elst | 3.91 | 0.291 | `b_elst` 0.399 |

- G1 TS와 DFT TS의 형성 결합 길이 차이(절댓값): 짧은 결합 중앙값 0.073 Å, 긴 결합 0.100 Å. 90% 분위는 0.209 / 0.371 Å (`g1_geometry_summary.json`).

### 0-2. Espley 비교 (rev 4 사전 고정 주 비교)

- 결과: 5개 타깃 모두 G1이 이겼고, seed 5/5에서 이겼다.
- 로컬 재계산(Nadeau–Bengio 보정 t-검정): p ≤ 0.0014.
- 가장 강한 Espley 역할 모델(우리 파이프라인 XGB)과 비교해도 p ≤ 0.003.

### 0-3. 엔진 확인 (로컬, rxn 52 r0, 2026-09-30)

**tblite 0.7.0과 xtb 6.7.1(edcfbbe)의 GFN2 에너지가 같다.**
- 기체상: −25.26109588 대 −25.26109587 Eh
- ALPB(water): −25.27468155 대 −25.27468154 Eh

**tblite는 궤도 계수와 AO overlap을 준다.**
- `calc.set("save-integrals", 1)`로 overlap 행렬을 얻는다.
- 검증: CᵀSC = I, 최대 오차 2.4e-15.

**xtb molden을 PySCF로 읽는 경로는 쓰지 않는다.**
- max|CᵀSC − I| = 2.46으로, 규격화가 맞지 않았다.

**xtb 6.7.1 바이너리가 이미 주는 값**
- `--json`: 원자 쌍극자, 원자 사중극자, 궤도 에너지
- D4 블록: 원자별 α(0), 분자 α(0), C6, C8
- `--vfukui`: f⁺, f⁻, f⁰
- `--vipea`: 수직 IP와 EA

### 0-4. 사용자 결정 (2026-09-30)

1. **G0(DFT 기하) 결과와 코드 경로를 폐기한다.** git 이력(≤ `740ba89`)에만 남긴다.
2. **Pauli, OI, elst를 개선하려고 feature를 확장한다.** xTB로 계산할 수 있는 값과 G1 기하에서 뽑을 수 있는 값만 쓴다.
3. **우리 모델과 Espley 모델을 비교하는 논문용 그림을 만든다.**

---

## 1. 규칙

- **HPC:** 저장소 CLAUDE.md 규칙을 따른다.
  - sbatch만 쓰고 `--time=48:00:00`을 쓴다.
  - 로그인 노드에서 python과 루프를 쓰지 않는다.
  - 제출 전에 `squeue -u $USER -h | wc -l`을 확인한다. MaxSubmit 20이고 array 원소는 개별로 센다.
  - 모든 단계는 idempotent여야 하고, 결과는 원자적으로 쓴다.
- **보존:** `labels_all.json`, G1 기하(`espley_xtb_g1/`), G1 feature parquet, `PREREG_REV4.md`는 수정하지 않는다. 사전 등록 문서는 역사 기록이다.
- **엔진**
  - xtb 6.7.1 바이너리: 이 바이너리가 출력하는 값은 모두 이것으로 계산한다.
  - tblite 0.7.0: 궤도 overlap 블록(B1)에만 쓴다.
  - 두 엔진의 동등성은 B-0 gate로 확인한다.
  - PySCF molden 경로는 금지한다(0-3).
- **정보 수준:** 새 feature는 G1 구조(TS, rel1, rel2, G1 TS에서 자른 조각)와 G1 제품 구조(B6)에서만 계산한다. DFT 에너지나 DFT 라벨 값은 쓰지 않는다.
- **선별 편향 방지:** lockbox(C-1)는 Phase D 전까지 어떤 계산에도 쓰지 않는다. 결과를 본 뒤에 arm, 모델, 행을 고르지 않는다.
- **수치:** 모든 수치는 파일에서 직접 읽는다.

---

## Phase A — G0 폐기

### A-1. 결과 파일 (`results_rev4/`)

**삭제 (`git rm`)**
- `downstream_g0/` 폴더 전체
- `figures/*_g0.png`
- `phase2_verify_geom_dft.json`, `phase2_report.json`, `phase2_feature_shift.csv` (G0 → G1 비교)
- `AUDIT_G0.md`

**재생성 (손으로 고치지 말고 스크립트로)**
- `rev4_headline.*`, `rev4_appendix_*.csv`, `rev4_pre_ml.csv`: G0 열을 없애고 `rev4_tables.py`로 다시 만든다.
- `espley_compare_*`: G0 행과 열(`g0_upper_bound_*` 등)을 없애고 다시 만든다.
- `espley_geometry_ablation*`: G0를 쓰는 arm B, B_role, O를 뺀다. A, C, C0, B_g1, O_g1만 남긴다.

**`SUMMARY.md`**
- G0 서술과 표를 모두 지운다.
- 맨 위에 한 줄만 남긴다: "G0(DFT oracle geometry) 결과는 2026-09-30 사용자 결정으로 폐기. git ≤ `740ba89`에만 남음."

### A-2. 코드

- `xtb_slice.py`: `--geom` 선택지를 `{g1, g2}`로 줄이고 기본값을 `g1`로 한다.
- `train_ml_single.py`, `compare_espley.py`, `espley_fairness_ablation.py`, `rev4_tables.py`, `analyze_extra.py`, `evaluate_pairs.py`: G0 분기를 없앤다.
- `verify_geom_dft.py`와 G0 전용 `r4_*.sh`를 삭제한다.

### A-3. scratch

- 삭제: `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g0.parquet`, G0 ML 출력, `compare/g0/`
- 지우기 전에 경로와 sha256을 `results_rev5/A_deleted.txt`에 기록한다.

### A-4. 검증

- 다음 결과가 `PREREG_REV4.md`와 폐기 문장 말고는 0건이어야 한다.
  ```
  grep -rniE "g0|oracle|upper bound|상한|geom dft" analysis/espley_xtb_repro --include=*.py --include=*.sh --include=*.md
  ```
- 행 집합 확인: rev 4 행 규칙의 "G0 ok" 조건은 5,260건 모두 ok였다. 그래서 G1만으로 다시 만든 행은 `rows_rev4.csv`(4,839)와 같아야 한다. 이를 assert하고 결과를 보고한다.
- commit: `espley_xtb_repro: discard G0 (DFT-geometry) results and code paths (user decision 2026-09-30)`

---

## Phase B — 새 feature (G1 구조에서만)

### B-0. 엔진 smoke (sbatch 1건)

- **대상:** rev 4 smoke와 같은 5건(20, 105, 2434, 2721, 3452).
- **tblite 설치:** r4 venv에 `pip install tblite==0.7.0`을 한다. 설치할 수 없으면 STOP.
- **gate 1:** 같은 구조, 같은 전하에서 tblite GFN2 ALPB(water) 총에너지와 xtb 바이너리의 차이가 1e-6 Eh 미만.
- **gate 2:** tblite 조각 계산의 overlap 행렬이 TS 전체 계산 overlap의 해당 블록과 1e-10 안에서 같다(원자 순서 매핑 확인).
- **gate 3:** CᵀSC = I (1e-8).
- **STOP:** gate 하나라도 실패하면 멈춘다. 시간(core-h/rxn)도 보고한다.

### B-1 ~ B-6. feature 블록

모든 블록은 새 스크립트 `ext_features.py`(`--block {B1..B6}`)가 계산한다.
- 입력 구조: rev 4 G1 폴더의 `ts.xyz`, `rel1.xyz`, `rel2.xyz`, `ts.hess`, `result.json`
- 조각: 라벨의 `A_idx`로 G1 TS를 자른 것(xtb_slice의 `fA`, `fB`와 같은 방법)
- 반응 원자: xtb_slice와 같은 정의 — dipole 3개(dip_a, dip_mid, dip_b), dipolarophile 2개(dph_a, dph_b)

#### B1. 궤도 overlap (Pauli, OI) — tblite GFN2, ALPB(water)

**계산**
1. 조각 A, B를 각각 따로 계산한다(q1, q2). ε, C, 점유를 얻는다.
2. TS 전체를 `save-integrals`로 계산해 AO overlap S를 얻는다.
3. 원소별 AO 수는 단일 원자 계산으로 구한다. 합이 전체 nao와 같은지 assert한다.
4. AO를 원자 순서로 A, B에 매핑하고 S_AB를 잘라낸다.
5. MO overlap: 𝒮 = C_Aᵀ S_AB C_B

**feature**

| 이름 | 정의 | 겨냥하는 항 |
|---|---|---|
| `mo_pauli_S2_occ` | Σ_{i∈occA, j∈occB} 𝒮_ij² | Pauli |
| `mo_pauli_S2_front` | 양쪽 HOMO..HOMO−2끼리의 Σ𝒮² | Pauli |
| `mo_oi_AB`, `mo_oi_BA`, `mo_oi_tot` | Σ_{i∈occA, a∈virtB} 𝒮_ia² / Δε_ia, 반대 방향, 합 | OI |
| `mo_S_HA_LB`, `mo_S_HB_LA` | \|𝒮(HOMO_A, LUMO_B)\|, \|𝒮(HOMO_B, LUMO_A)\| | OI |
| `mo_dE_HA_LB`, `mo_dE_HB_LA` | ε_LUMO − ε_HOMO (eV) | OI |
| `mo_S2dE_HA_LB`, `mo_S2dE_HB_LA` | 𝒮² / Δε | OI |
| `mo_S2dE_win_AB`, `mo_S2dE_win_BA` | HOMO..HOMO−2 × LUMO..LUMO+2 창에서 𝒮²/Δε의 최댓값 | OI (형식상 FMO가 아닌 궤도) |
| `eps_H/L_{dA,dB,rA,rB}` | 조각(TS 기하)과 relaxed 반응물의 HOMO/LUMO 에너지 | OI |

- **Δε 하한:** Δε를 max(Δε, 1.0 eV)로 둔다(사전 고정). 하한에 걸린 쌍의 수를 보고한다.
- **근거**
  - 점유–점유 overlap이 클수록 Pauli 반발이 커진다.
  - 궤도 안정화는 S²/Δε로 설명한다 (Vermeeren et al., Nat. Protoc. 2020, doi:10.1038/s41596-019-0265-0).
  - 1,3-쌍극 고리화의 FMO 모델과 distortion (Ess & Houk, JACS 2008, doi:10.1021/ja800009z).
  - 형식상 HOMO/LUMO가 아닌 궤도(예: 아자이드의 HOMO−3, LUMO+2)가 지배할 수 있다 (J. Org. Chem. 2021, 86, 5792, doi:10.1021/acs.joc.1c00239).

#### B2. 조각 간 접촉 (Pauli, elst 침투) — 기하만

G1 TS에서 조각 간 원자쌍(i∈A, j∈B) 전체를 쓴다.

| 이름 | 정의 | 개수 |
|---|---|---:|
| `bm_{hh,hH,HH}_{025,035,050}` | Σ exp(−r_ij/ρ), ρ = 0.25/0.35/0.50 Å, 쌍 종류(heavy–heavy, heavy–H, H–H)별 | 9 |
| `vdw_pen_sum` | Σ max(0, R_i + R_j − r_ij), Bondi 반지름 | 1 |
| `vdw_n_{075,085,100}` | r_ij < s(R_i + R_j)인 쌍의 수, s = 0.75/0.85/1.00 | 3 |
| `dmin_nf_1`, `dmin_nf_2` | 형성 결합 두 쌍을 뺀 가장 짧은 접촉, 두 번째로 짧은 접촉 | 2 |
| `rdf_hh_{240..380}` | heavy–heavy 조각 간 거리 히스토그램, 2.4–4.0 Å, 0.2 Å 폭 | 8 |

- **근거**
  - 교환 반발은 거리에 대해 지수적으로 줄어든다 (Buckingham, Proc. R. Soc. A 1938, doi:10.1098/rspa.1938.0173).
  - van der Waals 반지름 (Bondi, J. Phys. Chem. 1964, doi:10.1021/j100785a001).

#### B3. 다극 정전기 (elst) — xtb 바이너리 `--json`, 조각 밀도 고정

- **입력:** xtb_slice의 조각 단일점 계산(`fA`, `fB`, ALPB)을 `--json`으로 다시 실행해 원자 전하, 원자 쌍극자, 원자 사중극자를 읽는다.
- **feature**
  - `mp_E_qmu`: 전하–쌍극자, 양방향 합
  - `mp_E_mumu`: 쌍극자–쌍극자
  - `mp_E_qTheta`: 전하–사중극자, 양방향 합
  - 모두 Cartesian 다극 상호작용 텐서로 계산한다. 전하–전하 항은 기존 `b_elst`다.
- **침투 대리값 (계수 고정, 적합하지 않음)**
  - `pen_damp_{10,20}` = Σ q_i q_j (1 − e^{−α r_ij}) / r_ij, α = 1.0 / 2.0 Å⁻¹
  - `pen_ovl_{10,20}` = Σ N_i N_j e^{−α r_ij}. N은 GFN2 원자 가전자 수에서 Mulliken 전하를 뺀 값이다.
- **근거**
  - GFN2는 원자 다극자(사중극자까지)로 정전기를 계산한다 (Bannwarth et al., JCTC 2019, doi:10.1021/acs.jctc.8b01176).
  - 짧은 거리에서의 정전기 침투 (Wang et al., JCTC 2015, doi:10.1021/acs.jctc.5b00267).

#### B4. 반응성 지수와 응답 (OI, elst) — xtb 바이너리

**`--vipea` (ALPB water)**
- 대상: rel1, rel2, fA, fB
- IP와 EA에서 μ = −(IP + EA)/2, η = IP − EA, ω = μ²/2η를 구한다.
- 구조 4개 × 3 = 12개
- `dN_rel`, `dN_dist` = (μ_dph − μ_dip)/(η_dip + η_dph): relaxed 쌍, distorted 쌍

**`--vfukui`**
- 대상: fA, fB (TS 기하)
- 반응 원자 5개 × {f⁺, f⁻, f⁰} = 15개

**편극률 (D4 블록)**
- 반응 원자 5개의 원자 α(0)
- fA, fB, rel1, rel2의 분자 α(0)
- 모두 9개

**GEDT**
- TS 단일점의 Mulliken 전하를 dipole 조각 원자에 대해 더한다(기존 TS 계산을 재사용).

**근거**
- electrophilicity (Parr, von Szentpály & Liu, JACS 1999, doi:10.1021/ja983494x)
- hardness (Parr & Pearson, JACS 1983, doi:10.1021/ja00364a005)
- Fukui 함수 (Parr & Yang, JACS 1984, doi:10.1021/ja00326a036)
- D4 편극률 (Caldeweyher et al., JCP 2019, doi:10.1063/1.5090222)
- GEDT (Domingo, Molecules 2016, doi:10.3390/molecules21101319)

**주의:** GFN2의 IP/EA 절댓값은 계통적으로 어긋날 수 있다. ML에서 상대값으로만 쓰이므로 그대로 둔다.

#### B5. 거리 민감도 스캔 (기하 불일치 대응) — xtb 바이너리

**스캔**
- 방향: G1 TS에서 u = (dph_a와 dph_b의 중심) − (dip_a와 dip_b의 중심)을 단위벡터로 만든다.
- B 조각 전체를 δ·u만큼 평행이동한다. δ ∈ {−0.10, −0.05, +0.05, +0.10} Å이고, 조각 내부 기하는 고정한다.
- δ마다 복합체 단일점(ALPB, q1 + q2)을 계산한다. 조각 에너지는 기존 값을 재사용한다.

**δ마다 계산하는 값 X**
- E_int(δ)
- b_pauli(δ), b_oi(δ): xtb_slice의 B_CH8과 같은 항별 분해
- b_elst(δ): 조각 전하를 고정하고 새 거리로 계산

**feature (X ∈ {E_int, b_pauli, b_oi, b_elst}, 모두 16개)**
- `scan_slope_X` = [X(+0.05) − X(−0.05)] / 0.10
- `scan_curv_X` = [X(+0.05) − 2X(0) + X(−0.05)] / 0.05²
- `scan_X_m10`, `scan_X_p10`: δ = −0.10, +0.10의 값

**gate:** δ = 0에서 다시 계산한 값이 xtb_slice 값과 1e-6 Eh(에너지 항) 안에서 같아야 한다. 다르면 STOP.

**근거**
- G1과 DFT의 형성 거리는 0.07–0.10 Å 다르다(0-1).
- ASM/EDA 항은 반응 좌표를 따라 가파르게 변한다 (Fernández & Bickelhaupt, Chem. Soc. Rev. 2014, doi:10.1039/c4cs00055b; Vermeeren et al. 2020).
- 기울기와 곡률을 주면, 모델이 계통적인 거리 차이를 1차로 보정할 수 있다. 이것은 가설이며, Phase C의 블록 ablation이 시험한다.

#### B6. TS 성격 (Hammond)

**G1 `ts.hess`와 `result.json`에서**
- `nu_imag`: 첫 허수 진동수
- `mode_share`: 형성 결합 위 허수 모드 비율(기존 값)
- `mode_async` = |Δr_ad| / (|Δr_ad| + |Δr_be|): 허수 모드 변위를 두 형성 결합에 투영한 값

**제품**
- Coley `p0_*.xyz`에서 출발해 `xtb --opt tight --alpb water`로 최적화한다.
- gate: 최적화한 제품의 heavy-atom 결합 그래프가 Coley 제품과 동형이어야 한다.
- 원자 대응은 `formed_pairs_ts`와 heavy 그래프 동형으로 맞추고 assert한다.

**feature**
- `xtb_dErxn` = E_P − E_rel1 − E_rel2 (kcal/mol)
- `prog_ad`, `prog_be` = r_TS / r_P (두 형성 결합)

**근거:** Hammond, JACS 1955, doi:10.1021/ja01607a027

**정보 수준:** 제품의 출발 구조는 DFT 제품이다. G1의 rel과 같은 수준이다. G2로 갈 때는 autodE 제품으로 바꿔야 한다. 문서에 적는다.

### B-7. 실행과 gate

- **실행:** array(`r5_ext_array.sh`, 18 slices %10). 출력은 `xtb_features_ext_g1.parquet`이고 `rxn_id`로 G1 parquet과 병합한다.
- **STOP 조건**
  - `rows_rev4` 가운데 새 feature 실패가 1%를 넘을 때
  - 병합 후 NaN이 있을 때
  - B5 δ = 0 일치 gate가 실패할 때
- **보고**
  - 블록별 실패 수와 core-h
  - Δε 하한에 걸린 수
  - 블록별 feature의 분포(중앙값, 1%·99% 분위)
- **행:** `rows_rev5.csv` = `rows_rev4` ∩ 새 feature ok

---

## Phase C — 사전 등록과 블록 선별 (개발 집합에서만)

### C-1. PREREG_REV5a (선별 전에 commit)

**lockbox**
- `rows_rev5`에서 15%를 뽑는다(`np.random.default_rng(20261001)`).
- `lockbox_ids.csv`와 sha256을 기록한다.
- Phase D 전까지 어떤 학습에도 쓰지 않는다.

**dev**
- 나머지 85%. 선별은 dev 안의 5-fold CV(`KFold(5, shuffle, rs=20261001)`)로만 한다.
- 바깥 fold마다 train 안에서 nested GridSearchCV를 돌린다(rev 4 grid, Protocol A와 같은 파이프라인).

**모델:** 선별에는 KRR(RBF)만 쓴다.

**후보 arm**
- `BASE` = ESPLEY73
- `BASE` + 블록 하나씩(6개)
- `EXT_ALL` = BASE + B1..B6

**선별 기준:** elst, Pauli, OI 세 타깃의 dev CV NMAE 평균

**전진 블록 선택 규칙**
- 매 단계에서 기준을 가장 많이 줄이는 블록을 더한다.
- 채택 조건: 상대 감소 ≥ 2%, 그리고 바깥 5 fold 중 4 fold 이상에서 감소.
- 조건을 못 맞추면 멈춘다. 결과가 `EXT_SEL`이다.
- 채택된 블록이 하나도 없으면 `EXT_SEL = BASE`로 두고 계속 진행한다(STOP 아님).

**사전 고정 arm:** `EXT_ALL`은 선별과 상관없이 최종 평가에 포함한다.

### C-2. 선별 실행과 보고

- 9 타깃 모두에 대해 블록별 dev CV MAE와 NMAE를 보고한다. fold별 값도 저장한다.
- 선별 경로를 보고한다: 단계별 기준값, 채택 또는 탈락.
- 새 feature 하나와 세 타깃의 Pearson r (dev 행만)을 보고한다.

### C-3. PREREG_REV5b (Phase D 전에 commit)

`EXT_SEL`의 블록 목록과 feature 수를 적는다. 이후 바꾸지 않는다.

---

## Phase D — 최종 평가

### D-1. lockbox (1차, 선별 편향 없음)

- **방법:** dev 전체로 학습(nested 튜닝)하고 lockbox로 채점한다.
- **arm:** BASE, EXT_SEL, EXT_ALL
- **모델:** KRR이 헤드라인이고, Ridge, SVR, XGB는 부록이다.
- **타깃:** 9개
- **지표**
  - MAE, NMAE, r²
  - 반응 단위 bootstrap(10,000회) 95% CI
  - 짝지은 차이 EXT_SEL − BASE의 bootstrap CI

### D-2. Protocol A (rev 4와의 연속성)

- rev 4와 같은 방식이다: 80/10/10, seed 22/23/14/1/2, nested.
- 행: `rows_rev5` 전체
- arm: ESPLEY46, ESPLEY73, EXT_SEL, EXT_ALL
- **주석:** EXT_SEL은 dev 행에서 골랐으므로 D-2 test 행과 겹친다. 편향 없는 값은 D-1이다. 표에 이 주석을 단다.

### D-3. Espley 비교

**(a) Espley 분할 (rev 4 방식)**
- 행: 3,327 ∩ `rows_rev5`
- 우리 쪽: ESPLEY46(rev 4 사전 고정 주 비교를 다시 실행), ESPLEY73, EXT_SEL. 모두 KRR.
- Espley 쪽
  - 저장된 SVR 예측
  - 역할 d1/d2는 (a) 그들의 프로토콜 SVR로 재학습
  - rev 4의 빈칸을 채운다: Espley 46 feature × 우리 파이프라인 XGB를 Interaction, ΔE‡, ΔG‡에도 적용
- 검정: seed별 짝지은 차이에 Nadeau–Bengio 보정 t(J = 5, n_test/n_train은 실제 값)
  - p와 95% CI를 보고한다.
  - 비교 대상은 "Espley SVR"과 "Espley 쪽에서 가장 강한 모델" 두 가지다.

**(b) lockbox 정면 비교 (EXT_SEL의 1차 Espley 비교)**
- test = lockbox ∩ Espley 행
- train = dev ∩ Espley 행
- Espley 쪽: 그들의 프로토콜을 따른다 — ESI Table S3 grid, 이 train에서 GridSearchCV 5-fold, X만 표준화, SVR과 KRR.
- 우리 쪽: EXT_SEL KRR, nested
- 타깃 5개: 역할 dipole, 역할 dipolarophile, Interaction, ΔE‡, ΔG‡
- 지표: 반응 단위 짝지은 bootstrap CI

### D-4. 학습 곡선 (Espley `ml_analysis.py` 방식)

- 분할: Espley 분할, seed 5개
- 부분집합: train을 rxn_id로 인덱싱하고 Espley 행 순서로 둔 뒤 `sample(frac=f, random_state=seed)`로 뽑는다. f ∈ {0.1, …, 0.9, 1.0}
- 양쪽이 **같은 부분집합**을 쓴다.
- 하이퍼파라미터는 전체 train 튜닝값으로 고정한다(Espley 코드와 같은 방식). test 행도 고정한다.
- 대상: Espley SVR 대 EXT_SEL KRR, 타깃 5개

---

## Phase E — 그림 (matplotlib)

### 공통 규격

- **파일:** `results_rev5/figures/`에 PDF(벡터)와 PNG(300 dpi)를 둘 다 저장한다.
- **크기:** 단단 89 mm, 양단 183 mm. 글꼴 Arial(없으면 DejaVu Sans), 7 pt.
- **자료:** 그림마다 그린 값을 같은 이름의 CSV로 저장한다(표 보기).
- **색 (엔티티에 고정, 모든 그림에서 같게)**

  | 계열 | 색 |
  |---|---|
  | Espley | `#2a78d6` |
  | Ours EXT_SEL | `#eb6834` |
  | Ours ESPLEY73 | `#1baf7a` |
  | Ours ESPLEY46 | `#eda100` |

  - 이 4색은 dataviz 검증기에서 명도, 채도, 색각 이상 구분(인접 ΔE 최저 9.1), 정상 시각 하한을 모두 통과했다.
  - 두 색(`#1baf7a`, `#eda100`)은 배경 대비가 3:1 미만이다. 그래서 막대에 값 라벨을 직접 달고 CSV를 둔다.
- **형식 규칙**
  - 이중 y축을 쓰지 않는다.
  - 계열이 2개 이상이면 범례를 둔다.
  - 글자는 검정과 회색 잉크로만 쓴다.
  - 격자는 옅게 한다.
  - 막대 사이에 2 px 간격을 둔다.
- **확인:** 저장한 PNG를 열어 라벨 겹침과 잘림을 직접 확인한다.

### 그림 목록

| 파일 | 내용 | 출처 |
|---|---|---|
| `fig1_espley_mae` | 타깃 5개 × 계열 4개 막대(평균 test MAE, SE 오차 막대, seed 점 5개, 값 라벨). 타깃마다 Espley ÷ EXT_SEL 배수와 p를 주석 | D-3(a) |
| `fig2_espley_lockbox` | lockbox 정면 비교 막대(Espley 대 EXT_SEL), bootstrap 95% CI | D-3(b) |
| `fig3_parity` | 2행(Espley / EXT_SEL) × 타깃 5열. x = DFT, y = 예측(반응별로 test였던 seed의 평균). y = x선, ±1과 ±2 kcal/mol 회색 띠. 타깃마다 축 공유, MAE와 r² 주석 | D-3(a) |
| `fig4_error_ecdf` | 타깃별 \|오차\| 누적분포, 계열 4개. 1과 2 kcal/mol 세로선, 1 kcal/mol 안의 비율을 직접 라벨로 | D-3(a) |
| `fig5_seed_pairs` | 타깃별 seed 5개의 짝지은 선(Espley → EXT_SEL), 제목에 p | D-3(a) |
| `fig6_learning_curves` | 타깃별 test MAE 대 학습 행 수(평균 ± SE). Espley 전체 데이터 MAE에 EXT_SEL이 도달하는 행 수를 주석 | D-4 |
| `fig7_channel_blocks` | elst, Pauli, OI의 dev CV NMAE: BASE, 블록별, EXT_SEL, EXT_ALL (fold 점 포함) | C-2 |
| `fig8_channel_lockbox` | 타깃 9개의 lockbox MAE, BASE 대 EXT_SEL, 짝지은 bootstrap CI | D-1 |

---

## Phase F — 문서

- **`results_rev5/SUMMARY.md`**
  - 헤드라인: D-1 lockbox 표(EXT_SEL · KRR, 타깃 9개, BASE와의 차이, CI)
  - Espley 비교: D-3(a)와 D-3(b). EXT_SEL은 (b)를 1차로 둔다.
  - 선별 경로와 블록별 기여(C-2)
  - 사전 등록과 다른 점
- **하류 분석 (G1, EXT_SEL · KRR)**
  - charge breakdown과 group split
  - MMP `dom_agree`에는 **분할별 null 기준선**을 함께 적는다: 최빈 채널 비율과 순열 기준선.
  - rev 4에서 G1 `dom_agree`는 0.680이었고, 전체 쌍 기준 "항상 Pauli"가 0.671이었다.
- **README와 CLAUDE.md**의 상태 줄을 갱신한다.
- **commit:** `espley_xtb_repro rev 5: extended xTB/geometry feature blocks (EXT_SEL), lockbox evaluation, Espley comparison figures`

---

## 보고 형식 (Phase마다)

1. 실행한 job id
2. 결과 표
3. STOP 조건별 판정
4. 다음 Phase 진행 여부
