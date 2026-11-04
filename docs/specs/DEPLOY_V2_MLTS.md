# DEPLOY-V2 (MLTS) — 라벨 수준 TS를 미세조정 MLIP로 싸게 얻어 8채널을 예측하는 배포 모델 — Claude Code 실행 지시서

**저장소:** `origin/deploy` HEAD(`2f9ffa3`)에서 새 브랜치 **`mlts`**를 만든다. 새 worktree
`/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts` (`git worktree add ... -b mlts origin/deploy`, 로그인 노드에서 git만).
**이 파일:** `docs/specs/DEPLOY_V2_MLTS.md`로 commit한다.
**새 코드:** `analysis/mlts/` · **결과:** `analysis/mlts/results/` · **scratch:** `$MLT=/gpfs/tmp_cpu2/yeseo1ee/mlts`
**읽기만 하는 것:** `$G2=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`, `$DEP=/gpfs/tmp_cpu2/yeseo1ee/deploy`,
`$G3=/gpfs/tmp_cpu2/yeseo1ee/d1_g3`, `$D1PAIR_SCRATCH`, D1 run(`$D1RUN=/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`), `analysis/` 아래 기존 코드 전부.
**기존 코드는 고치지 않는다.** 필요한 함수는 import해서 부른다. 바꿔야 하는 동작은 `analysis/mlts/`의 새 파일에 둔다
(G2가 `g2_autode.py`에서 한 방식: D1 코드는 import, 바꿀 함수만 이 프로세스 안에서 대체).

**순서:** 0 → I(DFT 힘 데이터 목록) → G(DFT 데이터 생성) → Z(기반 MLIP 고르기) → FT(미세조정) → A(TS 엔진) → H(입력)
→ C(코드 검사) → R(사전 등록) → K(채널 모델) → E(평가) → X(LODO 일반화) → W(최종) → P(추론기) → S(요약 후 **멈춤**)
- 14절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출과 git 외에 아무것도 하지 않는다.
  array는 원소 하나가 task 하나다(제출 한도 20, 실행 한도 10). 제출 전에 `squeue -u $USER -h | wc -l`을 확인한다.
- 모든 단위는 원자적으로 쓰고, 결과 파일이 있으면 건너뛴다(48 h에 잘리면 같은 스크립트를 다시 제출).

---

## 0. 무엇을, 왜

### 0-1. 원자료로 확인한 사실 — 채널 모델이 아니라 구조가 병목이다

| 입력 구조 (D1 시험 para) | 라벨 TS 대비 형성 결합 \|Δd\| 중앙값 | 같은 종류 모델의 oi para MAE (kcal/mol) | 출처 |
|---|---|---|---|
| 라벨 DFT TS (B3LYP-D3BJ/def2-SVP, SMD water) | 0 | **0.102** (dft·SH, 1,759짝), 0.175 (dft·S) | `d1_ladder/results/E1_goal.csv` |
| G3L: G2 TS → 라벨 수준 OptTS 재최적화 | **0.0012 Å** (19반응), RMSD 0.006 Å, 라벨 TS로 수렴 18/19 | (구조가 라벨과 같음) | `d1_g3/results/G3L_report.json` |
| g3r: r2SCAN-3c 재최적화 | 0.063 Å | 0.53 (dft·S), 1.19 (dft·SH) (7짝, 설명용) | `d1_g3/results/SUMMARY_G3.md` E-1·E-3 |
| g2: autodE + XTB2 ALPB(water) | 0.186 Å (위와 같은 7짝 표본) | 1.01–1.09 (전체 para) | 같은 곳, `deploy/results/E_metrics.csv` |
| GEO1b: ML 좌표 보정 → dft 모델 | — | 1.615, **d1 5.59** (null 0.56보다 나쁨) | `geo2/results/SUMMARY_DEPLOY.md` |

dft·SH의 모든 채널 (para MAE / 꺾임 민감도 / 위양성률): d1 0.067 / 0.94 / 0.05, elst 0.099 / 0.95 / 0.06,
pauli 0.119 / 1.00 / 0.03, oi 0.102 / 0.96 / 0.02, cpcm 0.060 / 1.00 / 0.01, barrier 0.110 / 0.88 / 0.07.
**구조가 라벨 수준이면 지금 채널 모델로 계열별 꺾임 판정까지 된다.**

**왜 장벽은 xTB 구조로도 맞는데 채널은 안 맞나 (정류점 원리):**
- TS는 총에너지의 정류점이라 ∂E_total/∂r = 0이다. 구조 오차 δr이 장벽에 주는 영향은 2차(∝ δr²)다.
- 개별 채널(elst, Pauli, oi, d1 …)은 TS에서 기울기가 0이 아니다. 오차가 1차(∝ δr)로 들어간다.
- 원자료가 이와 맞는다. g3r 구조(\|Δd\| 0.063 Å)의 HF 짝 잡음 중앙값은 barrier 0.11인데 Pauli 1.68, oi 0.91이다(SUMMARY_G3 E-2).
- 문헌도 맞는다. xTB TS에서 B3LYP 장벽을 Δ학습한 Δ2는 MAE 1.32 kcal/mol이고, 오차가 xTB–DFT 구조 차이와 거의 상관이 없었다(TS · 평형 구조 RMSD 대 절대 오차 Spearman ρ = 0.2, 무거운 원자 ≤ 10개 C·N·O 반응; doi:10.1039/D3SC02408C).
- 그래서 **8채널은 장벽보다 훨씬 정확한 TS 구조를 요구한다.**

결론 — 배포 구조가 갖춰야 할 세 조건:
1. **라벨과 같은 수준의 PES**: 다른 범함수(r2SCAN-3c)는 TS 위치가 0.06 Å 어긋나 이미 부족하다.
2. **그 PES 위의 정류점**: 좌표를 직접 예측하면(GEO1b) 결합마다 작은 오차가 남아 인공 변형 에너지가 쌓인다
   (d1 오차 5.6 kcal/mol; 어림: 결합 20개 × K(Δr)², AMBER C–C K ≈ 310 kcal mol⁻¹ Å⁻², Δr ≈ 0.03 Å이면 약 5.6).
3. **같은 TS 형태**: xTB 재최적화가 라벨 TS에서 출발해도 형태를 바꾸는 반응이 있다
   (para 짝 1,655개 중 same 840 · mixed 562 · different 253, `d1_g3/results/S_report.json`).
   G2는 autodE의 hmethod를 xTB로 바꿔 TS를 최적화했기 때문이다. D1 라벨은 같은 autodE 흐름에서 hmethod = DFT였다.

### 0-2. 설계 — DFT를 흉내 내는 MLIP를 autodE의 hmethod 자리에 넣는다

| 단계 | 내용 | 대체하는 것 |
|---|---|---|
| ① 구조 엔진 **MLTS** | 기반 MLIP를 **라벨 TS 수준(B3LYP-D3BJ/def2-SVP, SMD water)의 에너지·힘**에 미세조정 | DFT (반응당 TS 8.2 코어시간) |
| ② TS 탐색 | **D1과 같은 autodE 1.4.5 흐름**(lmethod = xTB conformer, `hmethod_conformers = False`), hmethod의 ORCA 계산만 `ExtOpt` + MLTS | G2의 `XTB2 ALPB(water)` 자리 |
| ③ 입력 | MLTS TS에서 X182 + 원자 토큰 + HF/6-31G\* EDA 9값 (LADDER dft rung과 같은 코드) | — |
| ④ 채널 모델 | LADDER dft·SH/S 단위 그대로(재학습 없음) 또는 MLTS 입력으로 재학습 — 안쪽 검증으로 고름 | — |
| ⑤ 품질 표시 | MLTS 3-seed 위원회 TS 차이, 허수 진동수, (선택) TS에서 DFT 기울기 1회 | — |

- MLIP는 우리 DFT의 **힘 데이터**에서 PES를 배운다. EDA 라벨은 MLTS 학습에 쓰지 않는다.
- 최적화는 그 PES 위에서 하므로 결과가 정류점이다(조건 2). GEO1b는 좌표를 예측해서 이 성질이 없었다.

### 0-3. 다른 길을 쓰지 않는 이유

| 대안 | 판단 | 근거 |
|---|---|---|
| 범용 MLIP 그대로(AIMNet2, UMA, MACE-OFF) | 단독으로는 부족할 것으로 예상, Z에서 측정 | 학습 수준이 ωB97M 계열이라 라벨(B3LYP/SVP + SMD)과 다르다. 다른 범함수 g3r도 \|Δd\| 0.063 Å로 부족했다. 범용 MLIP는 반응 데이터로 미세조정하지 않으면 TS 탐색에 믿을 수 없고, 힘·에너지 지표가 TS 탐색 성공을 예측하지 못한다(doi:10.1002/advs.202506240). 평형 구조로만 학습한 모델의 TS 에너지 오차는 반응 데이터로 학습한 모델의 약 6.6배(0.495 대 0.075 eV; doi:10.1038/s41597-022-01870-w) |
| ML TS 생성기(React-OT, OA-ReactDiff, TSDiff) | 시작 구조로만 가능 | 생성 구조는 어떤 PES의 정류점도 아니다(조건 2). 보고된 중앙 RMSD는 0.053–0.08 Å이고, 무거운 원자 7개 이하 기체상 집합 기준이다(doi:10.1038/s42256-025-01010-0, 10.1038/s43588-023-00563-7). TSDiff도 DFT 정제가 필요하다(doi:10.1038/s41467-023-44629-6). 다만 같은 구조의 MLIP보다 TS를 더 자주 찾았다는 비교가 있어(doi:10.1002/advs.202506240), A-3 시범에서 형태 불일치가 크면 추가 출발점 후보로 검토한다 |
| 2D 그래프 모델(CGR 등) | 채택 안 함 | 구조 없이 TS 위치를 알 수 없다. CGR 장벽 MAE 2.57 kcal/mol. 같은 연구에서 TS 구조를 넣은 3D 모델은 2.25, 반응물·생성물 구조로는 6.20(doi:10.1021/acs.jpca.2c02614) |
| 값싼 구조(AM1) + ML로 변형·상호작용 직접 예측 | 채택 안 함 | Stuyver [3+2] 집합에서 d1 2.55, d2 2.37, 상호작용 2.46 kcal/mol(Diels–Alder는 ≤ 0.5)(doi:10.1039/D4DD00224E) |
| xTB 구조 + 값싼 EDA Δ학습(g2·SH) | 이미 실패 | g2·SH oi 1.04 ≈ g2·S 1.09 |
| G3L(DFT 재최적화) | 정답이지만 비쌈 | 반응당 2.6 코어시간, 14만 반응이면 약 36만 코어시간 |

### 0-4. 성공 기준 (R에서 그대로 등록)

| 등급 | 조건 (D1 T1 시험, para P_main) |
|---|---|
| 목표 달성 | oi · cpcm para MAE ≤ 0.13 **그리고** d1 · elst · pauli · oi 꺾임 민감도 ≥ 0.7, 위양성률 ≤ 0.1 (LADDER 목표) |
| 부분 개선 | 9개 타깃 모두 para MAE가 lf(현재 배포 모델)의 0.5배 이하 |
| 실패 | 그 외 |

### 0-5. 비용 예산

| 항목 | 상한 | 넘으면 |
|---|---|---|
| G: DFT 데이터 생성 | 5,000 코어시간 (G-0 실측으로 예상) | STOP |
| FT: 미세조정 | fold 5 + 최종 3 seed + LODO 5 + 능동학습 1회 ≈ 15회 × ≤ 12 GPU-h | 회당 24 GPU-h 넘으면 STOP |
| A: autodE-MLTS (D1 9,266반응) | 반응당 중앙값 0.6 코어시간 + 3 GPU-분 (A-3 실측) | STOP |
| H: HF EDA | 반응당 중앙값 0.106 코어시간 (`d1_hf/results/P_REPORT.md`) | — |
| 배포 목표 | 반응당 ≤ 1 코어시간 + ≤ 3 GPU-분 | W에서 보고 |

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | `origin/deploy` = `2f9ffa3` 이후이고 `analysis/{d1_g2,d1_g3,d1_hf,d1_ladder,d1_pair,deploy,geo2}`가 있다 |
| 0-2 | `$LAD/units/L__dft__{S,SH}__{TF,GNN,AW}__f{0..4}.{json,parquet}`와 가중치가 있다 (K의 C1용) |
| 0-3 | `$G2/jobs/<job_id>/ts.xyz` · `ts_result.json`이 ok 8,616반응에 있다 (A′ 대조용) |
| 0-4 | D1 run config(`$D1RUN/config_run.yaml`)와 G2 config(`analysis/d1_g2/config_g2.yaml`)를 읽을 수 있다 |
| 0-5 | ORCA 6.1.1에서 `! ExtOpt`가 동작한다 (C-1에서 확인) |
| 0-6 | D1 라벨 구조 전체의 원소 집합과 전하 · 다중도를 계산해 `results/0_elements.json`에 적는다. 예상: {H, C, N, O, F, Si, S, Cl, Br}(치환기 SiMe3 · Ts · SMe 포함), 모두 전하 0 · 다중도 1. 다르면 STOP. Si를 못 다루는 기반 MLIP는 Z에서 뺀다 |
| 0-7 | GPU partition(gpu3/4/5)에서 새 conda env **`mlts`**를 만들 수 있다 (`reactot`은 고치지 않는다; `conda create --clone reactot` 후 설치) |
| 0-8 | 디스크 여유 200 GB 이상 |

폴더: `sbatch --time=48:00:00 --wrap "mkdir -p $MLT/{logs,data,dft,ft,models,ade,final,inputs,units,preds,tmp}"`.

---

## 2. Phase I — 이미 있는 라벨 수준 DFT 힘 데이터 목록 (`inv_dft.py`, CPU)

- 찾을 곳: `$D1RUN` 아래 autodE 작업 폴더와 tar, `$G3`의 G3L 20반응·Part B 40반응·g3 재최적화, `$LAD`의 반응물 최적화.
- 쓰는 것: ORCA 출력 중 키워드가 **D1 run config의 `opt_ts`·`opt`·`grad`와 같은 수준**(B3LYP, D3BJ, def2-SVP, SMD water, 같은 grid)인 것만.
  최적화 단계마다 좌표 · 에너지 · 기울기(`CARTESIAN GRADIENT` 블록)를 읽는다. `_trj.xyz`와 단계 수가 맞아야 한다.
- 기록: rxn_id, 종류(ts_path, ts_final, react_opt, frag), 원자 수, E (Eh), F (Eh/bohr), 원본 파일 sha256, 키워드 문자열.
- 산출: `results/I_inventory.json`(종류별 구조 수 · 반응 수 · 수준 불일치로 뺀 수), `$MLT/dft/existing.extxyz`.
- 판정 I-1: TS 근방 구조(라벨 TS 형성 결합과의 차이 ≤ 0.3 Å)가 5,000반응 이상에서 50,000개 이상이면 G-2를 건너뛴다.
  아니면 G-2를 한다.

## 3. Phase G — 라벨 수준 DFT 데이터 만들기 (`gen_dft.py`, CPU array)

**수준:** D1 run config의 `grad` 키워드 그대로(`EnGrad`, B3LYP D3BJ def2-SVP, SMD(water) 블록 포함). ORCA 6.1.1.
**반응 범위:** D1 라벨이 있는 모든 반응. 구조마다 반응의 T1 fold를 적어 FT_k가 fold k를 빼게 한다.

| 묶음 | 구조 | 반응당 수 | 목적 |
|---|---|---|---|
| G-1 TS 근방 | 라벨 TS에서 정규 모드 방향 무작위 변위(조화 에너지 0.5–3 kcal/mol). 허수 모드 방향으로 형성 결합 ±0.05–0.15 Å 변위 1개 포함 | 3 | TS 주변 곡률 |
| G-2 경로 | G2 TS → 라벨 수준 OptTS 궤적(G3L과 같은 입력, `analysis/d1_g3` 함수 import) | 400반응(dipole · g1 형태 class로 층화), 궤적 전 단계 | xTB-TS에서 DFT-TS로 가는 길 |
| G-3 반응물 | 라벨 `ref_dipole.xyz` · `ref_dipolarophile.xyz`와 정규 모드 변위 2개씩, TS에서 잘라낸 두 조각 | 고유 분자마다 3, 조각 2 | 반응물 최적화 · 변형 에너지 영역 |

- 정규 모드: 라벨 TS Hessian이 D1 job 폴더에 있으면 그것을 쓰고, 없으면 그 구조에서 xTB Hessian을 쓴다(기록).
- G-0 (먼저): 무작위 50 EnGrad로 코어시간을 재고 전체를 예상한다. **예상 > 5,000 코어시간이면 STOP.**
- 산출: `$MLT/dft/gen_{G1,G2,G3}.extxyz`(에너지 eV, 힘 eV/Å로 변환, `rxn_id`, `fold`, `kind`, `source` 정보 포함), `results/G_report.json`.
- **G-4 능동학습 1회 (FT 1차 뒤):** 학습 쪽 반응 1,000개에서 MLTS로 TS를 최적화하고 그 구조에서 DFT EnGrad 1회.
  최대 힘 > 0.002 Eh/bohr인 구조를 학습에 더해 2차 미세조정한다(비용 약 100 코어시간).

## 4. Phase Z — 기반 MLIP 고르기 (`zero_shot.py`, GPU)

후보 (env에서 쓸 수 있는 것만; 못 쓰면 이유를 적고 뺀다):

| 후보 | 학습 수준 (원 논문) | 비고 |
|---|---|---|
| AIMNet2 | ωB97M-D3/def2-TZVP (14원소) | Si 포함, 전하 지원. TS 벤치마크는 원 논문에 없음 |
| MACE-OMOL-0 | OMol25 (ωB97M-V/def2-TZVPD) | MACE 미세조정 도구 사용 가능 |
| UMA-S | OMol25 등 | HuggingFace 접근 권한 필요. 없으면 뺀다 |
| MACE-OFF24 | SPICE (ωB97M-D3(BJ)/def2-TZVPPD) | **Si 없음** → SiMe3 없는 반응에서 참고값만 |
| GFN2-xTB ALPB(water) | — | 기준선 (= G2 수준) |

- 시험 반응: 학습 쪽에서 dipole별 40개, 모두 200반응.
- 지표:
  - G 데이터에 대한 힘 MAE와 반응 안 상대 에너지 MAE (수준 차이가 있으니 참고)
  - 라벨 TS에서 출발한 TS 최적화의 \|Δd\| · RMSD · 허수 진동수 1개 비율
  - GPU 속도
- **FT 시범:** Z 지표 상위 2개 후보로 fold 0 미세조정을 하고 FT-2 지표(5절)로 하나를 고른다. 같으면 빠른 쪽.
- 예상(사전 기록): 미세조정 전 \|Δd\|는 g3r(0.063 Å)과 비슷하게 부족할 것이다.
- 산출: `results/Z_report.json`, `results/Z_DECISION.md`.

## 5. Phase FT — 미세조정 (`finetune.py`, GPU)

- 학습 데이터: I + G의 구조. **FT_k는 T1 fold k의 반응 구조를 하나도 쓰지 않는다**(누수 검사: rxn_id 교집합 0).
  검증 = 학습 반응 중 10 %(core로 묶어 뽑음).
- 손실: 에너지 + 힘 (기반 MLIP의 미세조정 기본 가중치에서 시작, 검증으로 1회 조정). 원자 기준 에너지는 최소제곱으로 맞춘다.
- 모델: FT_0…FT_4 (평가용), FT_all × 3 seed (배포용, 위원회), FT_LODO_d × 5 (X용).
- **FT-1 (검증, 기록만):** TS 근방 구조의 힘 MAE ≤ 0.6 kcal mol⁻¹ Å⁻¹, 반응 안 상대 에너지 MAE ≤ 0.3 kcal/mol.
  - 근거(어림): TS의 반응 좌표 곡률 약 100 kcal mol⁻¹ Å⁻²에서 힘 오차 0.6이면 TS 위치 오차 약 0.006 Å.
  - 판정에 쓰지 않는 이유: 힘·에너지 지표는 TS 탐색 성공을 예측하지 못했다(doi:10.1002/advs.202506240). 판정은 FT-2 기하로만 한다.
- **FT-2 (기하, 판정):** fold k 시험 반응 전부에 FT_k를 쓴다. ORCA `ExtOpt OptTS`로 두 출발점에서 최적화한다.
  - (a) 라벨 TS에서 출발: \|Δd\| 중앙값 ≤ 0.010 Å, 90 % ≤ 0.030 Å, 무거운 원자 RMSD 중앙값 ≤ 0.02 Å, 라벨과 같은 TS(`rmsd.same_ts`, 0.10 Å / 0.02 Å) ≥ 0.90
  - (b) G2 TS에서 출발: 같은 지표를 보고하고 g1 형태 class(same/different)별로 나눈다
  - MLTS Hessian 허수 진동수 정확히 1개: ≥ 0.95
  - 근거: G3L(같은 수준)은 \|Δd\| 0.0012 Å로 라벨에 수렴했다.
- G-4 뒤 2차 미세조정을 한 번 더 하고 FT-2를 다시 잰다. **2차에서도 FT-2 (a)를 못 넘으면 STOP**(설계가 틀렸다는 뜻이므로 보고).
- 산출: `$MLT/models/FT_{k}.model`, `results/FT_report.json`, `results/FT_geom.csv`(반응별 Δd · RMSD · 허수 진동수).

## 6. Phase A — TS 엔진 (`oet_mlts.py`, `mlts_autode.py`)

### A-1. ORCA ↔ MLTS 다리
- ORCA 6.1 `! ExtOpt` + `%method ProgExt "<wrapper>" Ext_Params "..." end`.
  wrapper는 모델을 한 번만 올리는 **서버 방식**이어야 한다(호출마다 모델을 다시 올리면 수치 Hessian이 수십 분 걸린다).
- 공개 도구 `orca-mlips`(MACE 체크포인트 `--weights-file`, 서버 방식, 해석적 Hessian 덤프)가 우리 모델을 올리면 그것을 쓴다.
  아니면 `orca-external-tools`의 서버/클라이언트 구조를 따라 `oet_mlts.py`를 만든다.
- 검사 (모두 C-1에서):
  - 다리 일치: ORCA `ExtOpt EnGrad`의 E · F = 파이썬 직접 호출 (\|ΔE\| < 1e-6 Eh, \|ΔF\| < 1e-5 Eh/bohr)
  - 최적화기 일치: ORCA `ExtOpt OptTS`와 Sella(ASE)가 같은 MLTS로 라벨 TS 5개에서 같은 TS(형성 결합 차 < 0.005 Å)
  - Hessian: ORCA `NumFreq` 허수 진동수 수 = MLTS 자동미분 Hessian의 허수 진동수 수

### A-2. autodE-MLTS (`mlts_autode.py`, `g2_autode.py`를 본뜸)
- `pilot_common.configure_autode`를 그대로 부른 뒤 ORCA 키워드 세트만 바꾼다:
  `XTB2 ALPB(water)` 자리를 `ExtOpt` + `%method` 블록으로 바꾸고, `Freq`는 `NumFreq`로 바꾼다(또는 해석적 Hessian 덤프 + `InHessName`, C-1 결과로 정함).
- G2처럼 `ORCA.add_solvent_keyword` · `print_solvent`를 이 프로세스 안에서만 no-op으로 만든다. 용매는 MLTS의 PES에 들어 있다.
  - 생성된 `.inp`에 CPCM · SMD 단어가 없는지 검사한다.
- 반응은 `solvent_name="water"`를 유지한다(lmethod xTB conformer는 D1과 같게).
- 반응물 단계: `make_reference.dft_refine` 대신 `mlts_refine`(MLTS Opt tight + NumFreq로 G). Coley 규칙과 기준값은 바꾸지 않는다.
- TS template 폴더 = `$MLT/ts_templates` (D1 · G2 template는 읽지도 쓰지도 않는다).
- 작업 폴더 = 노드 로컬 `${TMPDIR:-/tmp}/mlts_<jobid>_<rxn>`. **지우기 전에** `ts.xyz`, `ts_result.json`, `ref_*.xyz`, `.hess`, 입력 tar를
  `$MLT/ade/jobs/<job_id>/`에 복사한다. 메타데이터는 이 사본에서 만든다(`DECISIONS.md`의 g2_deploy `$TMPDIR` 버그를 되풀이하지 않는다).
- 실행 단위: GPU job 하나 = MLTS 서버 1개(GPU 1장) + autodE 작업자 n개(CPU, xTB conformer). n은 A-3에서 정한다.

### A-3. 시범 (fold마다 60반응, 모두 300; 그 fold의 FT_k 사용)
- 비교 엔진:
  - **A**: autodE-MLTS (A-2)
  - **A′**: G2 TS → ORCA `ExtOpt OptTS` (MLTS)
  - **A″ (정밀-경량)**: A의 MLTS TS → 라벨 수준 ORCA `OptTS`, 첫 Hessian = MLTS 해석적 Hessian(`InHessName`), `Recalc_Hess` 없음.
    DFT 기울기 횟수와 코어시간을 잰다(G3L은 G2 TS에서 21회, 2.6 코어시간).
    - 근거: ML Hessian을 쓰면 TS 최적화 단계가 2–3배 준다(doi:10.1038/s41467-024-52481-5).
    - 미리 적어 두는 예상: MLIP 시작 구조에서 DFT 기울기 4회 미만이 필요했다는 보고가 있다(arXiv 2604.00405, 심사 전).
    - A″는 A의 구조 정확도가 FT-2를 못 넘을 때의 대안 계층이다. 비용이 반응당 1 코어시간 이하이면 K · E에서 별도 arm으로 평가한다.
- 지표: TS 성공률, 라벨과 같은 TS 비율, 같은 TS 반응의 \|Δd\| · RMSD, 반응당 wall · GPU-분 · 코어시간.
- 채택: 같은 TS 비율이 높은 엔진. 같으면 싼 쪽.
  - **같은 TS 비율 < 0.70이면 STOP** (형태 문제를 다른 방법으로 풀어야 함).
  - 0.70–0.85이면 진행하고 형태 불일치를 주된 한계로 기록한다.
- 비용이 0-5 예산을 넘으면 STOP.

### A-4. 본 실행
D1 대상 9,266반응 전부. 반응마다 자기 fold의 FT_k를 쓴다. 실패 반응은 이유와 함께 기록한다(재시도 1회, infra 오류만).

## 7. Phase H — MLTS TS 위의 입력 (`inputs_mlts.py`, CPU)

- 마무리: `analysis/deploy/finalize_lf.py`와 같은 규칙(라벨 의존 없음; `formed_pairs_ts` · `A_idx`는 `ts_result.json`)으로
  `$MLT/final/<rxn_id>/`(G1 폴더 형식)를 만든다.
- X182: `feat_lf.py`와 같은 코드(조각 재검출 `redetect`만, 실패하면 탈락).
- 원자 토큰: `tokens_lf.py`와 같은 코드.
- HF/6-31G\* EDA 9값: LADDER dft rung이 쓴 `d1_hf` 코드와 같은 설정.
- **D3BJ 직접 계산 (물리 채널):** 분산은 구조만의 함수다(doi:10.1063/1.3382344, 10.1002/jcc.21759).
  - B3LYP 매개변수 D3(BJ)로 두 조각 사이의 분산 상호작용 E_disp(TS) − E_disp(조각1) − E_disp(조각2)를 계산해 입력 열 `D3_int`로 더한다.
  - H-0: 라벨 TS 100반응에서 `D3_int`와 라벨 disp 채널을 비교한다. \|차\| 중앙값 ≤ 0.01 kcal/mol이면 disp는 모델 대신 `D3_int`를 그대로 예측값으로 쓴다(E에서 둘 다 보고). 아니면 입력 열로만 쓴다.
- **H-1 재현 검사:** 라벨 TS 50반응을 같은 코드에 넣어 LADDER dft rung 입력(X182 · 토큰 · HF)과 같은지 본다(\|Δ\| ≤ 1e-4). 다르면 STOP.
- **H-2:** 입력 실패율 ≤ 2 %. 넘으면 STOP.

## 8. Phase C — 코드 검사 (`tests.py`, 결과 `results/C_tests.json`)

| # | 검사 |
|---|---|
| C-1 | A-1의 세 검사 (다리 · 최적화기 · Hessian) |
| C-2 | 누수: FT_k 학습 rxn_id ∩ fold k = ∅, 채널 단위 f{k} 학습 짝 ∩ fold k = ∅ |
| C-3 | autodE-MLTS가 만든 `.inp` 무작위 50개에 범함수 · 기저 · CPCM · SMD 단어가 없고 `ExtOpt`가 있다 |
| C-4 | MLTS 단위 변환: 물 · 메탄 · 라벨 TS 3개에서 MLTS E · F를 ORCA DFT EnGrad와 비교해 단위(eV/Å ↔ Eh/bohr) 오류가 없다 |
| C-5 | H-1 재현 검사 |
| C-6 | K의 C1: LADDER dft 단위에 라벨 TS 입력을 넣으면 `L_predictions.parquet`과 같다(최대 차 ≤ 1e-4 kcal/mol) |
| C-7 | 입력 민감도: MLTS TS를 형성 결합 방향으로 ±0.01 Å 밀었을 때 예측 변화가 E-2 잡음 표(SUMMARY_G3)와 같은 크기인지 보고 |

## 9. Phase R — 사전 등록 (`results/PREREG_V2.md`, K 시작 **전에** commit)

- 0-4 등급 기준, 1차 지표, 비교 대상(dft rung 상한, lf), 대표 모델 고르는 규칙(안쪽 검증), 재학습 잡음 대역 계산법.
- "지금까지 본 것": I · G · Z · FT · A · H 결과 (구조 지표만, 채널 결과는 0개).

## 10. Phase K — 채널 모델

| 이름 | 방법 |
|---|---|
| C1 재학습 없음 | LADDER `L__dft__{S,SH}__{TF,GNN,AW}__f{k}`에 fold k의 MLTS 입력을 넣는다. 대표는 LADDER가 고른 것 그대로(REP, REP8) |
| C2 MLTS 입력 재학습 | LADDER와 같은 grid · 분할 · 학습 코드, rung = `mlts` |
| C3 섞어 학습 | 학습 쪽에 라벨 TS 입력과 MLTS 입력을 둘 다 넣는다(같은 y) |

- C1 · C2 · C3 중 1차 모델은 **안쪽 검증 para MAE**(oi + cpcm 합)로 고른다. 시험 결과로 고르지 않는다.
- 재학습 잡음: C2를 seed 3개로 학습해 δ_H(`DEPLOY_MODEL_W_AMEND1.md` 정의)를 계산한다.

## 11. Phase E — 평가 (`eval_v2.py`, LADDER E1과 같은 함수)

- 집합: D1 T1 시험, P_main para (LADDER와 같은 짝). 위치별(o, m, direct)은 부 지표.
- 지표: 9개 타깃 para MAE와 core 부트스트랩 CI, 채널별 꺾임 민감도 · 위양성률, 숨은 꺾임 민감도 · PPV, 연속 G AUC.
- 비교 (같은 짝): dft rung(상한), lf(현재 배포), MLTS 1차 모델, C1 · C2 · C3 전부(기술용).
- 분해: (i) A에서 라벨과 같은 TS를 찾은 반응만 포함한 짝 / 아닌 짝, (ii) dipole별, (iii) g1 형태 class별.
- 산출: `results/E_metrics.csv`, `E_bends.csv`, `E_bend_rates.json`, `E_report.json`, 그림(dft · lf · MLTS의 para MAE 막대, 반응별 \|Δd\| 대 오차 산점도).

## 12. Phase X — D1 밖 일반화 근사: dipole 하나 빼고 미세조정 (LODO)

- dipole d마다 FT_LODO_d: d의 모든 반응 구조를 뺀 G · I 데이터로 미세조정.
- d의 반응에서 FT-2 지표((a) 라벨 TS 출발, (b) G2 TS 출발)를 잰다. autodE는 돌리지 않는다.
- **보고만 한다.** FT-2 (a)를 못 넘는 dipole이 있으면 "새 dipole 계열에는 그 계열 DFT 데이터를 조금 넣어 미세조정해야 함"으로 기록한다.

## 13. Phase W · P — 최종 모델과 추론기

- W-1: FT_all × 3 seed. 채널 모델은 K에서 고른 설정으로 전체 D1 학습. C1이면 LADDER dft 설정을 전체 D1으로 학습한다(최종 단위가 없을 때).
- W-2 위원회 표시: 세 MLTS가 최적화한 TS의 형성 결합 차 최대값 δ_com. E 데이터에서 δ_com과 채널 오차의 관계를 보고 경고 기준을 정한다.
- P-1 `predict_v2.py`: SMILES → autodE-MLTS → 입력 → 8채널 + barrier + 표시(TS gate, 허수 진동수, δ_com, 입력 범위 밖 여부).
- P-2 자체 검사: D1 20반응(E 예측과 일치), D1에 없는 SMILES 5개(끝까지 도는지만).
- P-3 (선택) DFT 증명서: 최종 TS에서 라벨 수준 EnGrad 1회(약 0.1 코어시간). 시험 200반응에서 최대 힘 분포와 채널 오차의 관계를 보고한다.

## 14. STOP

| 조건 |
|---|
| 1절 시작 조건 실패 |
| G-0 예상 > 5,000 코어시간, FT 1회 > 24 GPU-h, A-3 비용이 예산 초과 |
| 2차 미세조정 뒤에도 FT-2 (a) 실패 |
| A-3 같은 TS 비율 < 0.70 |
| H-1 재현 실패, 입력 실패율 > 2 % |
| C-2 누수, C-6 회귀 실패 |
| K · E 결과를 본 뒤 등록 내용을 바꿔야 할 것 같을 때 |

## 15. 보고 (`results/SUMMARY_V2.md`)

- 맨 위 한 줄: 등급(0-4)과 oi · cpcm para MAE, 숨은 꺾임 PPV.
- 구조 표: xTB(g2) · MLTS · G3L · 라벨의 \|Δd\| · RMSD · 같은 TS 비율 · 반응당 비용.
- 채널 표: dft 상한 · lf · MLTS의 9개 타깃 para MAE, 꺾임 민감도 · 위양성률.
- 한계: 형태 불일치 비율, LODO 결과, 배포 비용 실측.
- commit 금지: `$MLT` 아래 산출물(저장소 `CLAUDE.md`).

---

## 부록. 근거 문헌 (2026-11-04 Crossref에서 DOI 확인)

| 쓰는 곳 | 문헌 | DOI |
|---|---|---|
| 라벨 데이터 | Stuyver et al., Reaction profiles for quantum chemistry-computed [3+2] cycloaddition reactions, *Sci Data* 2023 | 10.1038/s41597-023-01977-8 |
| autodE | Young et al., *Angew. Chem. Int. Ed.* 2021, 60, 4266 | 10.1002/anie.202011941 |
| Sella (C-1 최적화기 대조) | Hermes et al., *J. Chem. Theory Comput.* 2022, 18, 6974 | 10.1021/acs.jctc.2c00395 |
| 반응 데이터 없는 MLIP의 TS 한계 | Schreiner et al., Transition1x, *Sci Data* 2022 | 10.1038/s41597-022-01870-w |
| 범용 MLIP는 미세조정 필요, 힘 지표 ≠ TS 성공 | Zhao, Duan et al., *Adv. Sci.* 2025 | 10.1002/advs.202506240 |
| ML Hessian TS 최적화 | NewtonNet Hessian, *Nat. Commun.* 2024 | 10.1038/s41467-024-52481-5 |
| 기반 후보 | AIMNet2, *Chem. Sci.* 2025 · MACE-OFF, *J. Am. Chem. Soc.* 2025 | 10.1039/D4SC08572H · 10.1021/jacs.4c07099 |
| 고리화첨가 TS 영역 MLIP (반응당 약 100–600 구조) | Duarte 그룹, *PCCP* 2022 · *Nat. Commun.* 2024 | 10.1039/D2CP02978B · 10.1038/s41467-024-50418-6 |
| ML TS 생성기 | OA-ReactDiff · React-OT · TSDiff | 10.1038/s43588-023-00563-7 · 10.1038/s42256-025-01010-0 · 10.1038/s41467-023-44629-6 |
| 장벽 Δ학습(xTB TS) | Δ2, *Chem. Sci.* 2023 | 10.1039/D3SC02408C |
| TS 구조가 정보를 가짐 | Spiekermann et al., *J. Phys. Chem. A* 2022 | 10.1021/acs.jpca.2c02614 |
| [3+2] 변형·상호작용 ML 기준 | Espley et al., *Digital Discovery* 2024 | 10.1039/D4DD00224E |
| EDA 항의 범함수 민감도 (Pauli · oi 약 20 %) | *J. Comput. Chem.* 2023/24 | 10.1002/jcc.27244 |
| D3 · BJ 감쇠 | Grimme 2010 · 2011 | 10.1063/1.3382344 · 10.1002/jcc.21759 |
| SMD | Marenich, Cramer, Truhlar 2009 | 10.1021/jp810292n |
| Δ-ML | Ramakrishnan et al. 2015 | 10.1021/acs.jctc.5b00099 |

심사 전 문헌(판정 근거로 쓰지 않고 예상 기록에만 씀): OMol25 · UMA(arXiv 2505.08762, 2506.23971),
MLIP TS 탐색 비교(arXiv 2604.00405). 도구(논문 아님): ORCA 6.1 ExtOpt 문서, `orca-mlips` 1.3.0(PyPI, MIT).
