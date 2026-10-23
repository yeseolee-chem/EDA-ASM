# PREREG ADD3 — D1PAIR 사전 등록 추가본 (D1-REF·LADDER 1단계: dft·g1, 2026-10-23)

이 문서는 `PREREG_D1PAIR.md`(commit 10f99d8d), `PREREG_D1PAIR_ADD1.md`(commit 1de89a0f), `PREREG_D1PAIR_ADD2.md`(commit cf86f727)에 더하는 추가본이다. 셋은 바꾸지 않는다.
근거: 명령서 `docs/specs/D1LADDER.md` 9절(sha256 `e0d873df7b3807ae19d2e887971781fc52956051de38b4ed4bb5e8236555abcb`). 이 문서와 json은 `analysis/d1_ladder/prereg.py write`가 결과 파일만 읽어 만들었다. 결과 수치는 모두 결과 파일에서 옮겼다. 규칙의 기준값은 명령서를 인용했고, 검토자 수치는 명령서 0-1·9절 인용이며 그 자리에 출처를 적었다(생성 코드 sha256은 json `code.generator`).
범위: **1단계(rung `dft`·`g1`)**. 2단계(`g2`)는 10절의 보완 규칙으로 따로 등록한다.

Phase M1 학습(`train_atoms.py run`, `train_xgb.py run`)은 이 문서와 `prereg_d1pair_add3.json`이 commit된 뒤에만 돈다. 두 학습 코드의 `gate_add3`가 매 실행 전에 확인한다.
- 두 파일이 git에 있고 HEAD 뒤로 바뀌지 않았는지
- md sha256이 json `md_sha256`과 같은지
- json `inputs_sha256`에 정답 파일과 학습하는 rung의 입력표가 있는지, 그리고 적힌 **33개 파일**의 sha256이 지금 파일과 같은지(다른 rung 열에만 속한 파일은 열지 않는다, L-4)

하나라도 다르면 exit 3이다. 모든 unit 지문(`add3`)에 이 md의 sha256이 들어가고, collect는 그 값을 다시 비교한다.

**지금까지 본 것** (작성 2026-10-23 16:19:32):
- Phase M1 학습은 한 번도 돌지 않았다(`$LAD/units`의 `L__` 파일 0개).
- Phase C 검사 코드(`tests/run_tests.py`)가 계산하는 MAE는 L-3(합성 자료), L-10(원래 라벨·원래 H의 LINE = D1-HF Phase A 재현), smoke의 검증 짝 MAE(2 epoch, 새 정답)뿐이다(코드 기준). 새 정답의 시험 짝 점수는 계산하지도 보지도 않았다.
- 원래 라벨로 이미 본 결과(D1-HF Phase A, FU2 g1·S, 검토자 사후 탐색·사전 점검)는 7절에 그대로 적는다.

## 1. 입력 고정

**정답:** `$LAD/F_pairs_cr2.csv` sha256 `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` (V_report.json all_pass, outputs에 같은 sha256; P_report.json `labels` ids_ok). 원래 라벨 `analysis/d1_pair/results/F_pairs.csv`는 비교용이다.

**코드:** commit `4316ae3548c9c64f90657b28c2ea6daf9bffb618` (Phase C 결과가 commit된 HEAD; 학습 코드, Phase C 코드 5개, `C_tests.json`이 HEAD와 같음을 확인했다. 다른 d1_ladder 파일의 git 상태는 json `code.ladder_git_state`). 아래 파일은 두 학습 코드의 지문 `code_sha256`에 들어가는 코드이고, `inputs_sha256`에도 있어 gate가 매 실행 전에 확인한다.

| 파일 | sha256 |
|---|---|
| `analysis/d1_pair/d1_train.py` | `414b5cc66762391691b252235c9133f0d193312a8039d35369299a8dfe30027c` |
| `analysis/d1_pair/d1pair_common.py` | `65a6e2da81e4dedf20c51423cbfe9c97e8b8f2272b24454a739a09e8552aeedc` |
| `analysis/d1_pair/fu2/fu2_common.py` | `66dcf72f5b23a015e1750e4b2ac458dd61a4722050a7be24a03b8269a628dfe2` |
| `analysis/d1_pair/fu2/models.py` | `de2065a7d39c1e008b31274ab1b4fc94fa123ce53eda14a3c2f426dbd0d6197a` |
| `analysis/d1_pair/fu2/tokens.py` | `f37207bbf987764cfbadd064ad4f9ffbdb33e54daa7563825b3966d0c5e95514` |
| `analysis/d1_pair/fu2/train.py` | `2f1ab7a06bfec20901cfd9130a9784961070e6477728ade92fed9a1cce729469` |
| `analysis/d1_ladder/lad_common.py` | `e0d64e1e70bc0df60bd9f4befb6cdeebca3effcecc0eb57ab6b50f555ff62afe` |
| `analysis/d1_ladder/train_atoms.py` | `fb00f6046e1b45de4dbea80b80aa2f09ff7ddd11d2dfeaec8d532fe773a528cb` |
| `analysis/espley_xtb_repro/train_ml_single.py` | `05c6b38cdf7803f4a6c6b2d643555db95b5e366ea02714049232add38a55da4c` |
| `analysis/d1_ladder/train_xgb.py` | `79ddc2914ecf8664862ba0a416427d6ea3be03bc80735840208ea57543796d74` |

gate가 확인하지 않는 코드(`inputs.py`, `evaluate.py`, `tests/run_tests.py`, `refs_q.py`, `refs_v.py`, `g0f0.py`, job 파일, `fu2/atoms.py`, D1-HF·rev 9 평가 코드, FU2 Phase E `evaluate.py`)의 sha256은 json `code`에 적는다. d1_ladder `evaluate.py`의 E1 코드는 학습과 무관하게 따로 작성·commit되므로 gate 밖에 둔다(E1 규칙은 5·6절에 고정, LINE 함수는 3-5절·9절 7번의 소스 sha256으로 고정).

**입력 파일** (gate가 확인; 열 = 명령서 4-1의 rung 열, `inputs.check_paths`):

| 파일 | 역할 | 열 | sha256 |
|---|---|---|---|
| `$LAD/F_pairs_cr2.csv` | labels: F_pairs_cr2.csv (spec 4-1 y; need) | shared | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `$LAD/inputs/dft.parquet` | input table dft (need) | dft | `a20802a3eccf8d0d9b4c2020585d8e5aa5b708a8842cee9c346c4a4d834e9a82` |
| `$LAD/inputs/g1.parquet` | input table g1 (need) | g1 | `46833c475e1373f91f81596c5f77420585d56451eab282c16bfb6ee750ebfe52` |
| `$LAD/inputs/sets/P_dft.csv` | pair set P_dft | dft | `c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5` |
| `$LAD/inputs/sets/P_g1.csv` | pair set P_g1 | g1 | `ba08a648d0c0068a2027bafdb213d8e3eed671f2894ff23ef7f50dfef1413156` |
| `$LAD/inputs/sets/P_01.csv` | pair set P_01 | dft+g1 | `f81b99d363bae44104a92e7907a436b8beab6bf8c87204ba6dd2c4e8966ea5be` |
| `analysis/d1_pair/results/F_pairs.csv` | original F_pairs.csv (L-11 invariant on every label load) | shared | `aeaba8f8794358c126a57e2f6829792970d1e5c7d8433100ac3134a56c73f043` |
| `analysis/d1_pair/results/fu2/S_splits.json` | FU2 S_splits.json (split index) | shared | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `analysis/d1_pair/results/S_folds.csv` | S_folds.csv (folds, cores; XGB inner GroupKFold) | shared | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `$D1PAIR_SCRATCH/fu2/splits/T1__f0.json` | T1 fold 0 split file | shared | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `$D1PAIR_SCRATCH/fu2/splits/T1__f1.json` | T1 fold 1 split file | shared | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `$D1PAIR_SCRATCH/fu2/splits/T1__f2.json` | T1 fold 2 split file | shared | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `$D1PAIR_SCRATCH/fu2/splits/T1__f3.json` | T1 fold 3 split file | shared | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `$D1PAIR_SCRATCH/fu2/splits/T1__f4.json` | T1 fold 4 split file | shared | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |
| `analysis/d1_ladder/results/fu2/T_report.json` | tokens dft: T_report.json | dft | `b346ca19cbbe42c837c311d284335dae0b7b9a274275e7912555eb440ea052fe` |
| `analysis/d1_ladder/results/fu2/T_manifest.csv` | tokens dft: T_manifest.csv | dft | `fc2d07a6fc4a553660ef9361d82539383e93fcdb27b3d4e04b2b2cabbec47d68` |
| `analysis/d1_ladder/results/fu2/Q_report.json` | tokens dft: Q_report.json | dft | `7bed93541a26d487f4a415456964bcf302f6b264c15e7a519dee29d287c4b48a` |
| `analysis/d1_ladder/results/fu2/S_splits.json` | tokens dft: S_splits.json | dft | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `$LAD/X182_d1_dft.parquet` | global 182 features dft | dft | `2f90dd3ef13c277b2aabb4a2798d7070a8bc26b8b213290f5c8a9f1bb8d46500` |
| `analysis/d1_pair/results/fu2/T_report.json` | tokens g1: T_report.json | g1 | `673ac6c0efc77b52d18f562a821fa19ed9180ad4bf0f1a1fb6ff011cc0d8a871` |
| `analysis/d1_pair/results/fu2/T_manifest.csv` | tokens g1: T_manifest.csv | g1 | `15df2a7c4eba21e7bb8778dc3fc8772686693138394dece6c88730208c668cc4` |
| `analysis/d1_pair/results/fu2/Q_report.json` | tokens g1: Q_report.json | g1 | `12e6c5c08f17575ddb93abe698560856bd0d2d1587a4260fd7c8d98e45f88307` |
| `$D1PAIR_SCRATCH/X182_d1.parquet` | global 182 features g1 | g1 | `39699f0fea47b77055190cdaf25983f2c0dd9c89d797ba0a349306a05c199596` |

토큰 파일(반응별 npz)의 sha256은 각 rung의 `T_manifest.csv`에 있고, 학습 코드가 읽을 때마다 확인한다. `P_report.json`, `C_tests.json`, Q·V 보고 등은 2단계에서 다시 쓰이므로 gate에 넣지 않고 json `not_gated_sha256`에 적는다.

## 2. Phase Q·V — 기준 구조 규칙과 결과

### 2-1. 규칙 (명령서 2·3절, 결과를 본 뒤 바꾸지 않음)
- **분자 키(Q-2):** 반응 SMILES에서 나눈 dipole·dipolarophile의 RDKit 정규 isomeric SMILES. 거울상 이성질체는 한 분자(사전순 앞의 SMILES가 키), E/Z·cis/trans 부분입체 이성질체는 다른 분자. Coley 규칙이 이면각을 고정한(constrained) dipole 기준은 키에 고정 이면각 상태를 붙여 같은 상태끼리만 묶는다. 같은 키 안의 원소 구성·결합 그래프가 같음을 확인했다.
- **재조립(Q-4):** 키마다 라벨에 실제로 쓰인 기준 단일점 에너지의 최저 E_min을 고르고, E_orig − E_min > 0.05 kcal/mol인 반응만 E_min과 그 좌표로 바꾼다. d1·d2·barrier만 기존 조립 함수로 다시 계산하고, TS 에너지·EDA 6채널·c_ghost는 그대로다(gate Q-a–Q-e).
- **확인 계산(V-1–V-4):** 대상 = (a) dev > 0.5 kcal/mol 반응이 있는 분자, (b) cyclooctyne 계열 dipolarophile(골격 D, Dp) 전부, (c) 대조군 20개(`default_rng(20261023)`, dipolarophile 10·dipole 10); constrained dipole 키는 제외. 형태 탐색 = ETKDGv3 300개(randomSeed 20261023) + D1 기준에서 출발한 randomize-and-relax 300개 → `xtb --opt tight --alpb water --gfn 2` → 중복 제거(무거운 원자 RMSD < 0.1 Å 또는 ΔE < 0.05 kcal/mol) → xTB 하위 5개 + D1 기준 형태. DFT = D1 autodE 반응물과 같은 B3LYP-D3BJ/def2-SVP SMD(water) Opt + Freq(허수 진동수는 한 번 밀어 다시 최적화, 남으면 버림), 라벨 수준 단일점(라벨 `frag*_rel` 입력의 좌표만 바꿈), HF/6-31G* 단일점.
- **최종 기준(V-4):** 분자마다 min(Q의 E_min, V 형태)을 라벨 수준 단일점 에너지로 고른다. V 형태가 0.05 kcal/mol 넘게 낮으면 바꾸고 Q-4·Q-5 규칙으로 다시 조립한다 → `F_pairs_cr2.csv`(이후 모든 단계의 정답). 원래 라벨은 지우지 않는다.

### 2-2. 결과
- **Q 감사:** dev > 0.05 / 0.5 / 1 / 2 kcal/mol인 반응 3,100 / 783 / 633 / 466개, 영향받는 분자 639 / 181 / 143 / 86개. gate Q-a 통과, Q-b 통과, Q-c 통과, Q-d 통과, Q-e 통과.
- **Q 재조립:** 반응 3,100개(조각 3,266개, 분자 639개)의 기준을 바꿨다. Δbarrier 중앙값 0.137, 최대 69.47 kcal/mol.
- **V 대상:** 분자 152개 = (a) 95, (b) 82 (겹침 45), (c) 20; constrained dipole 키 445개 제외(그중 dev > 0.5인 것 86개).
| V 대상 | 분자 | 더 낮은 형태 (> 0.05) | 비율 | 낮아진 양 중앙값 | 최대 |
|---|---|---|---|---|---|
| (a) dev > 0.5 | 95 | 12 | 12.6 % | 0.202 | 1.173 |
| (b) cyclooctyne 계열 | 82 | 18 | 22.0 % | 0.177 | 2.277 |
| (c) 대조군 | 20 | 0 | 0.0 % | – | – |

- **V 결과:** 기준이 바뀐 분자 키 20개, V 기준을 쓰게 된 조각 123개, Q 대비 바뀐 반응 123개. DFT 최적화 실패 16/505 (3.2 %). gate Q-a 통과, Q-b 통과, Q-c 통과, Q-d 통과, Q-e 통과, V-a 통과, V-c 통과, V-e 통과.
- **최종(cr2, 원래 라벨 대비):** 기준이 바뀐 반응 3,131개 (V_report `n_rows_changed_vs_orig` 3,131), 조각 3,299개, 분자 639개 (dipole 432, dipolarophile 207; 출처 Q 619, V 20).

바뀐 짝(전체 8,389짝, |Δ| > 1e-09 kcal/mol; 다른 7개 타깃의 최대 차이 0.0e+00 kcal/mol):

| 타깃 | 바뀐 짝 | P_main | P_main para | 바뀐 양 중앙값 | 최대 |
|---|---|---|---|---|---|
| dd_d1 | 2,463 | 1,738 | 418 | 0.086 | 5.60 |
| dd_d2 | 1,610 | 1,244 | 274 | 0.137 | 8.56 |
| dd_barrier | 3,687 | 2,695 | 615 | 0.121 | 8.56 |
| 셋 중 하나라도 | 3,687 | 2,695 | 615 |  |  |

### 2-3. 검토자의 기준 구조 HF 사전 점검 (명령서 0-1절)
- 검토자: 짝에 쓰인 7,410반응 중 523반응(7.1 %; 명령서 인용 7.1 %)이 같은 분자의 최저 기준보다 1 kcal/mol 넘게 높은 기준을 썼다. 편차 중앙값 3.3 kcal/mol. dipolarophile 313건 중 cyclooctyne(D 135, Dp 122; 명령서 인용)이 82 %다. dipole 221건 — 명령서 인용: "dipole 221건 중 일부는 Coley가 일부러 고른 입체 호환 형태일 수 있다".
- 출처: Q_report.json `hf_precheck.reviewer`(수치); D·Dp 개수, 비율 인용값과 dipole 단서는 docs/specs/D1LADDER.md 0-1 (reviewer-supplied numbers; the counts below are not in Q_report.json).
- Q의 재현(같은 키, `labels_d1_hf.parquet` geom=dft): HF 판정 533반응, DFT 판정 468반응, 반응 단위 일치율 98.7 %, HF 편차 중앙값 3.30 kcal/mol.

## 3. LADDER 설계 (명령서 4절 전체)

### 3-1. 단계(rung)마다 입력은 모두 그 구조에서 계산한다
|  | `dft` | `g1` | `g2` (2단계) |
|---|---|---|---|
| 역할 | 상한(구조가 정답일 때), 배포 불가 | 구조를 xTB로 바꾸면 잃는 것 | 배포 성능 (SMILES → xTB) |
| TS 좌표 | 라벨 TS (D1-HF geom=dft EDA 입력, `$LAD/dft_final/<rxn>/ts.xyz`) | `G1_ROOT_D1F/<rxn>/ts.xyz` | `$G2/g2_final/<rxn>/ts.xyz` |
| 반응물 좌표(구조 feature) | 반응별 라벨 기준 구조 (dft `frag*_rel` 입력) | G1 `rel1/rel2.xyz` | G2 `rel1/rel2.xyz` |
| `ts.hess` (B6) | 새 계산: 라벨 TS 고정 ORCA `! XTB2 ALPB(water) Freq` (G0) | 기존 | D1-G2 |
| 원자 토큰 (FU2) | F0 `$LAD/fu2/tokens` (fu2-tokens-3-dft) | `$D1PAIR_SCRATCH/fu2/tokens` (fu2-tokens-3) | D1-G2 Phase F |
| 전역 182 | `$LAD/X182_d1_dft.parquet` | `$D1PAIR_SCRATCH/X182_d1.parquet` | `$G2/X182_d1_g2.parquet` |
| HF/6-31G* 9값 (H) | `labels_d1_hf.parquet` geom=dft + V HF 단일점 | 같은 파일 geom=g1 | `labels_d1_hf_g2.parquet` |
| 입력표 | `$LAD/inputs/dft.parquet` `a20802a3eccf8d0d…` | `$LAD/inputs/g1.parquet` `46833c475e1373f9…` | 2단계 |

- 구조 feature(토큰, 182)는 모든 rung에서 반응별 기준 구조를 그대로 쓴다(기존 파이프라인과 같음).
- **H 재조립:** H의 d1·d2·barrier는 rung 안에서 같은 기준 규칙으로 다시 조립한다.
  - `dft`: Q·V가 고른 최종 기준 구조의 HF 에너지(`$LAD/refs_cr2.parquet` `e_hf_eh`: 자기 기준 → labels_d1_hf geom=dft 그 행 값, Q 출처 → 출처 행 값, V 출처 → V의 HF 단일점).
  - `g1`(2단계 `g2`도 같음): 그 rung의 반응별 기준 구조 중 분자 키(Q-2)마다 HF `e_frag{k}_rel_eh`가 가장 낮은 것(문턱 없음, 동점은 job_id가 작은 것). 배포 때 반응물마다 기준을 한 번만 계산하는 것과 같다.
  - 조립은 D1-HF `build.py`(`parse_rxn`)에 기준 에너지만 바꿔 넣는다. gate P-a: 원래 기준을 넣으면 `labels_d1_hf.parquet`가 1e-9 안에서 재현된다. H의 EDA 6채널은 그대로다.
- H = HF 값 9개 `d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier` (kcal/mol, 반응별).
- 한 rung의 학습·평가는 그 rung 열의 파일만 읽는다(L-4). 정답 y는 모든 rung에서 `F_pairs_cr2.csv`다.

### 3-2. 입력 두 가지 (arm)
| arm | 원자 모델 입력 | XGB 입력 | 학습 대상 | 예측 |
|---|---|---|---|---|
| **S** (구조만) | 원자 토큰 + 전역 182 (FU2 변형 "A+G") | 182 (rev 9 arm C) | ΔΔ | 모델 출력 |
| **SH** (구조 + HF) | 원자 토큰 + 전역 (182 + H 9) | 182 + H 9 | 잔차 r = ΔΔ − β·ΔH | β·ΔH + 모델 출력 |

**SH의 기준선(검량선):**
- 타깃 t마다 ΔH_t = H_t(a) − H_t(b), β_t = Σ ΔH_t·Δy_t ÷ Σ ΔH_t², **절편 없음**(절편이 있으면 반대칭이 깨진다).
- β는 **그 학습에 쓰는 짝으로만** 맞춘다: 튜닝 때는 안쪽 학습 쪽, 최종 모델은 바깥 학습 쪽 전체(T_main: 기준 짝 + anchor 짝). c_ghost는 HF 대응 값이 없으므로 β = 0.

**SH의 학습 규칙(FU2 규칙에서 이것만 바꾼다):**
- 출력 = 잔차. 표준화는 학습 쪽 잔차 표준편차(ddof 0)로 나누기만 한다(평균은 빼지 않는다). 전역 H 9열은 182열처럼 그 학습의 학습 반응으로 표준화한다.
- 손실 = 잔차 MAE(표준화, 10 타깃) + λ·|ŷ_barrier − Σ 9채널 ŷ| / s_barrier, λ = 0.1. 벌점은 **기준선을 더한 최종 예측 ŷ**로 계산하고, s_barrier는 원래 y의 학습 쪽 표준편차다.
- 튜닝·early stopping 지표 = 최종 예측 ŷ의 검증 MAE를 **원래 y의 학습 쪽 표준편차**로 나눈 값의 9 주 타깃 평균(FU2와 같은 단위).
- (a,b)/(b,a) 순서 뒤집기는 잔차와 기준선을 함께 뒤집는다(FU2 `epoch_batches`).
- XGB: 타깃마다 r_t를 학습한다. 안쪽 CV 점수 = |β·ΔH + ĝ − y|의 평균(kcal).

### 3-3. 모델 네 개
| 모델 | 구조·grid·학습 규칙 |
|---|---|
| AW | FU2 M-2 그대로, 변형 A+G; grid L1·폭 64, L1·폭 128, L2·폭 64, L2·폭 128 |
| GNN | FU2 M-2 그대로, 변형 A+G; grid L4·폭 64, L4·폭 128, L6·폭 64, L6·폭 128 |
| TF | FU2 M-2 그대로, 변형 A+G (4 head); grid L2·폭 64, L2·폭 128, L3·폭 64, L3·폭 128 |
| XGB | rev 9 arm C XGB 그대로: n_estimators [200, 500], max_depth [3, 5, 7], learning_rate [0.03, 0.1] (12칸), ½[h(Δx) − h(−Δx)], 양방향 학습, 안쪽 `GroupKFold(4)`(core), `XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist")` |

- 원자 모델 구조·토큰 열·반대칭은 ADD2 2절과 같다(json `ladder.models.atoms_fu2_m2`). 출처: analysis/d1_pair/fu2/models.py GRID (= ADD2 json models.grid); analysis/espley_xtb_repro/train_ml_single.py GRIDS['XGB'] (prefix removed; = rev 9 prereg grids.XGB), cells = d1_train.grid_cells('XGB') (sorted keys, itertools.product).
- 원자 모델 학습: AdamW(lr 0.001, weight decay 0.0001), cosine(T_max 300), dropout 0.1, batch 64짝, 최대 300 epoch, patience 30, 튜닝 seed 20261019, 최종 seed 20261019, 20261020, 20261021, 20261022, 20261023의 평균(seed 표준편차 저장). 칸 = 검증 지표 최저(동점은 grid 순서), 최종 = 바깥 학습 쪽 전부를 best epoch만큼.
- XGB: 타깃마다 따로 튜닝(안쪽 MAE 평균 최저, 동점은 grid 순서), random_state 42 한 번 학습(위 estimator), yhat_seed_sd = 0.
- gate M-1′: 학습한 모든 모델에서 시험 짝 200개(입력만)를 float64로 다시 계산해 기준선을 포함한 최종 예측의 |g(a,b) + g(b,a)|, |g(a,a)| ≤ 0.0001 kcal/mol (XGB는 모델 부분도 ≤ 1e-06).
- 입력 금지(FU2와 같음): σ, 계열·core 번호, X 이름.

### 3-4. 짝·분할·학습 단위
- 분할: FU2 T1 split 파일(`d1_train.masks` arm C + 첫 안쪽 GroupKFold(4) fold) fold 0–4. 학습 = 학습 쪽 T_main, 시험 = 시험 쪽 P_sens ∪ A_main, 주 지표 = P_main. 검증 짝(대표 고르기, 13-1) = 첫 안쪽 fold(`inner_val`), 4 모델 공통(L-9).
- 한 rung에서 a·b가 사용 가능(토큰 + 입력표 행 + 유한한 H 9 + 유한한 182)하지 않은 짝은 모든 목록에서 뺀다 → 두 arm은 같은 `P_<rung>`을 쓴다(4절).
- 단위 이름 `L__<rung>__<arm>__<model>__f<k>`; 1단계 = 2 rung × 2 arm × 4 모델 × 5 fold = **80 unit**(원자 60 + XGB 20), GPU 학습 540회(원자 unit마다 튜닝 칸 수 + 최종 seed 5개; 칸 수 AW 4, GNN 4, TF 4). 2단계 = 40 unit, GPU 270회.
- 정답이 바뀌었으므로 모든 rung·arm을 새로 학습한다. FU2·rev 9 결과는 재사용하지 않고 "원래 라벨" 비교로만 보고한다.
- 캐시 지문(명령서 10절): 입력표·짝 라벨·짝 목록·split 목록 sha256, 이 md의 sha256, grid, 코드 버전 `d1-ladder-2`와 코드 sha256, torch·CUDA·GPU(원자) / scikit-learn·xgboost(XGB) 버전. 다른 지문의 캐시를 만나면 exit 3.

### 3-5. 참고 행 (학습 없음)
- **null:** ΔΔ = 0.
- **LINE:** β·ΔH만 쓴다. β는 바깥 학습 쪽 T_main에서 위 규칙으로 구한다(`evaluate.line_beta`·`line_predict`, L-10이 D1-HF Phase A 재현으로 확인). 두 함수의 소스 sha256 `2fafcba7536fae30043da7b07b91a7239a0ba2a9a5df064b4045caf76f34a230`(= HEAD 판; json `line_functions`)을 여기 고정한다. E1은 자기 소스가 이 값과 다르면 멈춘다(9절 7번).

## 4. 짝 집합
| 집합 | 짝 | 기준 | anchor | para | P_main | P_main para | 시험 짝 fold 0–4 | 빠진 짝 (T1에 쓰인 것) | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| P_dft | 8,325 | 7,192 | 1,133 | 2,116 | 6,874 | 1,759 | 1,709 / 1,667 / 1,444 / 1,357 / 1,685 | 64 (14) | `c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5` |
| P_g1 | 8,092 | 6,956 | 1,136 | 2,057 | 6,884 | 1,764 | 1,713 / 1,670 / 1,446 / 1,357 / 1,688 | 297 (1) | `ba08a648d0c0068a2027bafdb213d8e3eed671f2894ff23ef7f50dfef1413156` |
| P_01 | 8,079 | 6,946 | 1,133 | 2,052 | 6,874 | 1,759 | 1,709 / 1,667 / 1,444 / 1,357 / 1,685 | 310 (14) | `f81b99d363bae44104a92e7907a436b8beab6bf8c87204ba6dd2c4e8966ea5be` |

- P_g1 = FU2가 실제로 쓴 짝 목록: 같음 (FU2 8,092, P_g1 8,092). P_dft / P_g1 = 8,325 / 8,092 = 1.0288 (하한 0.98).
- 판정(13-2)은 그 rung의 자기 집합(`P_dft`, `P_g1`), rung 사이 비교(13-4)는 `P_01`(2단계에서 `P_012`)에서 한다.
- 학습 코드 규칙의 짝 목록 sha256(지문 `pair_list`와 비교할 값): P_dft `16b6d1f9cf6fbb46…` (8,024짝), P_g1 `f5bd6105a8661f25…` (8,037짝).
- H 재조립으로 바뀐 값(dft): 반응 3,131개; d1 1,992개(> 0.05: 1,676), d2 1,307개(> 0.05: 959), barrier 3,131개(> 0.05: 2,545).
- H 재조립으로 바뀐 값(g1): 반응 8,429개; d1 7,796개(> 0.05: 3,622), d2 8,069개(> 0.05: 1,049), barrier 8,429개(> 0.05: 5,367).

## 5. 판정 규칙 (명령서 13절, 고정)

### 5-1. 대표 (REP)
- rung × arm × 타깃마다 4 모델 중 **검증 kcal MAE**(바깥 fold 5개 평균)가 가장 낮은 모델. 동점이면 AW → GNN → TF → XGB 순서의 앞 것. 시험 점수로 고르지 않는다.
- ENS4 = 4 모델 시험 예측의 같은 비중 평균(조정값 없음). 보조로 쓴다.
- **고르기의 비대칭(학습 전 기록, `train_atoms.py` MA-6n):** 원자 모델의 검증 kcal MAE는 early stopping과 칸 고르기에 쓴 바로 그 `inner_val` 짝에서의 best epoch 값이라 낙관적이다. XGB의 값은 안쪽 fold 4개 평균으로 고른 칸의 fold 0 점수다. 명령서 4-3이 FU2 early stopping을 고정하므로 코드는 바꾸지 않고, REP가 원자 모델 쪽으로 기울 수 있음을 결과 해석에 적는다. 원문: “note for ADD3 (review M-6, no code change: FU2 early stopping on inner_val is fixed by spec 4-3): the atom models' val_kcal_mae is the best-epoch value on the same inner_val pairs that early stopping and the cell choice used (optimistic); XGB's is the fold-0 score of a cell chosen by the 4-fold mean. Phase A must record this selection asymmetry of the spec 13-1 REP rule before training.”

### 5-2. 단계 2 판정 (주 판정, rung × arm; 1단계는 4개)
- REP 기준. **oi와 cpcm 모두** para 짝 MAE ≤ 0.13 kcal/mol, 꺾임 민감도 ≥ 0.7, 위양성률 ≤ 0.1이면 통과.
- 집합은 그 rung의 자기 집합, 점 추정으로 판정한다. CI 상한까지 ≤ 0.13이면 "확실히 통과"로 따로 표시한다.
- **para 짝 MAE:** 그 rung 자기 집합의 시험 짝(fold 5개 합) 중 P_main 기준 para 짝(`set_mask` P_main, pos p)의 MAE. CI = core 단위 재추출 2,000회, seed 20261019(`boot_diff`, 비교 상대 오차 0).
- **꺾임 민감도·위양성률이 계산되는 계열(학습 전 고정):** `d1_eval.bend_rates`를 **계열 제한 없이**(fams=None) 쓴다. 계열은 FU2 `bends_tab`이 그 rung 자기 집합의 시험 짝 중 P_main 기준 para 짝(σ 있음)으로 만드는 계열 전부이고, 점이 12개 이상(H 포함)인 것만 남는다. 꺾임 판정은 `hammett_eval.calls`로 DFT 정답(`F_pairs_cr2.csv` dd)과 REP 예측에 같은 방식으로 한다. 민감도 = DFT에서 꺾인 계열 중 예측도 꺾인 비율, 위양성률 = DFT에서 곧은 계열 중 예측이 꺾인 비율(Clopper–Pearson 95 % CI 함께). DFT 쪽 분모가 0이라 값이 없으면(NaN) 그 기준은 **미달**로 본다(rev 9 `d1_eval.stages`와 같음). hidden_V·hidden_bend 계열만의 비율은 설명용이다.
- 근거(rev 9 PREREG 5-3, 그대로 인용): 숨은 꺾임 크기 중앙값이 궤도 0.47, 용매 0.45 kcal/mol이다. 모의실험(σp 18점)에서 잡음 SD ≤ 0.35 × 꺾임 크기 → SD ≤ 약 0.16 → MAE ≈ 0.8 × SD ≤ 약 0.13.
- oi·cpcm 정답은 Q·V에서 바뀌지 않는다(Q-b).
- 보조 판정: ENS4에 같은 기준. 개별 모델의 통과는 **설명용**(가장 좋은 것을 고르면 사후 선택이 된다). 단계 0·1(rev 9 기준)과 9 타깃 전부의 수치도 설명용으로 함께 적는다.
  - **단계 1(rev 9):** P_main 전체 위치의 계열 R²_fam 중앙값 ≥ 0.5. 설명용.
  - **단계 0(rev 9)은 계산할 수 없다(학습 전 고정):** rev 9 단계 0은 대표 모델(arm B)이 null보다 MAE가 낮고 arm A 대비 Holm 검정이 유의할 것이다(`d1_eval.stages`). LADDER에는 arm A가 없다(FU2 E-4도 단계 1·2만 보고). 대신 설명용으로 **"null보다 나음"** = REP의 P_main 전체 위치 MAE < null MAE(점 추정)를 9 타깃마다 적는다.

### 5-3. 해석 규칙 (위에서부터 처음 맞는 것 하나)
| # | 조건 | 결론 문장 |
|---|---|---|
| a | g2·S 또는 g2·SH 통과 | 배포 가능한 입력(SMILES → xTB)으로 목표에 도달했다 |
| b | g2 둘 다 실패, dft·S 또는 dft·SH 통과 | 모델은 목표에 닿을 수 있으나 xTB 구조의 정밀도가 병목이다. 구조를 DFT 수준으로 다듬는 경로를 시험할 근거가 있다 |
| c | dft 둘 다 실패 | 이 입력과 이 모델 네 개로는 구조가 정답이어도 목표에 닿지 않는다 |

- 1단계(E1)에서는 dft 부분만 적용해 "dft 통과 / 실패"만 적는다. g1 결과는 이 표에 넣지 않는다("xTB로 바꾸면 잃는 몫"의 설명용).

### 5-4. 설명용 비교 (보정 없음, FU2 `compare`, d = MAE(상대) − MAE(기준))
- 같은 rung·모델에서 SH 대 S (HF가 더하는 몫)
- 같은 arm·모델에서 rung 사이 (P_01, 2단계에서 P_012)
- REP 대 LINE (모델이 검량선 위에 더하는 몫)

## 6. 지표 (명령서 11절, E1)
- 평가 함수는 FU2 Phase E 코드(`fu2/evaluate.py`: `bends_tab`, `compare`)와 그것이 부르는 `d1_eval.boot_diff`·`clopper`·`family_eval`, `hammett_eval.calls`를 그대로 쓴다(FU2 Phase E가 d1-pair에 commit된 뒤 merge).
- `results/E1_metrics.csv`: rung × arm × 모델(AW, GNN, TF, XGB, REP, ENS4, LINE, null) × 타깃 × 집합(자기 집합, P_01) × 위치(all, p, m, o, direct) → n, MAE, RMSE, para MAE의 95 % CI, R²_fam 중앙값, ρ 오차. para MAE의 CI = core 단위 재추출 2,000회, seed 20261019, `boot_diff`에 비교 상대 오차 0을 넣어 MAE 자체의 CI.
- `results/E1_bends.csv`, `E1_bend_rates.json`: 꺾임 민감도·위양성률(Clopper–Pearson).
- `results/E1_goal.csv`: 5절 판정. `results/E1_compare.csv`: 5-4 비교.
- **라벨 교정 효과(설명용)** `results/E1_label_effect.csv`: g1·S의 새 라벨 결과 대 FU2·rev 9 원래 라벨 결과(d1·d2·barrier MAE, 계열 R²_fam, 꺾임 판정 수), DFT 정답 자체의 꺾임 판정 수 변화(`hammett_eval.calls`, 원래 대 새 라벨).
- 그림 `figL1`–`figL3`의 1단계 판. 예측 `results/L_predictions.parquet`(pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, base, beta, yhat_seed_sd), 검증 `results/L_val.csv`.

## 7. 이미 본 결과 (원래 라벨)

### 7-1. D1-HF Phase A — LINE(정답 ΔΔ = β × HF ΔΔ, 절편 없음, β는 학습 fold), P_main 6,885짝
`analysis/d1_hf/results/A_summary.csv` sha256 `0f83040f6965318760e1855834145d635579f1c1dcec55359f8da1370cd3bad8`. 사전 등록되지 않은 설명용 결과다(A_report: "descriptive, not pre-registered; bands only, no decision").

| 타깃 | para 짝 | LINE dft para MAE | LINE g1 para MAE | null para | FU2 GNN-A+G para (같은 짝) | β dft | β g1 |
|---|---|---|---|---|---|---|---|
| d1 | 1,764 | 0.175 | 0.376 | 0.581 | 0.226 | 0.713 | 0.531 |
| d2 | 1,764 | 0.102 | 0.296 | 0.430 | 0.167 | 0.833 | 0.551 |
| elst | 1,764 | 0.273 | 1.106 | 1.286 | 0.596 | 0.902 | 0.322 |
| pauli | 1,764 | 0.481 | 2.251 | 2.510 | 1.151 | 0.948 | 0.223 |
| oi | 1,764 | 0.283 | 1.299 | 1.442 | 0.660 | 0.939 | 0.241 |
| disp | 1,764 | 0.102 | 0.198 | 0.145 | 0.086 | 0.732 | 0.559 |
| cpcm | 1,764 | 0.201 | 0.368 | 0.421 | 0.247 | 0.733 | 0.318 |
| cds | 1,764 | 0.000 | 0.036 | 0.032 | 0.028 | 1.000 | 0.588 |
| barrier | 1,764 | 0.321 | 0.581 | 0.740 | 0.229 | 0.725 | 0.464 |

### 7-2. FU2 g1·S (원래 라벨)
- Phase M: `analysis/d1_pair/results/fu2/M_report.json` sha256 `6cb9124aee0e5ef12982d2f9850fcc973adf7448ff91fb0b3689b4dd27495445` (30 unit, 예측 sha256 `ea7b4363ab7b4621af984d539eda89fb32c7edbdcd4ee2fca9df49e70535d29e`). A+G unit이 고른 칸:
| 구성 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| AW-A+G | L2·폭 128 · ep 132 | L2·폭 128 · ep 70 | L2·폭 128 · ep 139 | L1·폭 128 · ep 148 | L2·폭 128 · ep 151 |
| GNN-A+G | L6·폭 128 · ep 71 | L6·폭 64 · ep 83 | L6·폭 64 · ep 36 | L6·폭 128 · ep 100 | L6·폭 64 · ep 126 |
| TF-A+G | L2·폭 128 · ep 129 | L3·폭 128 · ep 90 | L2·폭 64 · ep 52 | L2·폭 64 · ep 141 | L3·폭 64 · ep 78 |

- Phase E: `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/analysis/d1_pair/results/fu2` (E_check ok, gate 23개 실패 0, 2026-10-23 16:13:35, git c6c416c65b, evaluate.py `caf3400fb4ba8b37…`). 주 비교 (ADD2 4절): 6개 타깃에서 원자 대표가 C-XGB보다 나음 → "원자 입력이 182개 feature보다 낫다". P_main 시험 짝 6,884, core 44.
- E-0이 해시한 `M_predictions.parquet` sha256 = 이 브랜치 `M_report.json` `predictions_sha256`(`ea7b4363ab7b4621…`)임을 확인했다. E_* 파일 23개의 git 상태(그 worktree): commit 뒤 수정됨 1개, commit 안 됨 22개 — **commit 전** 파일의 수치다(d1-pair에 commit된 뒤 merge, 6절).

P_main para 짝 MAE (kcal/mol). REP = FU2 ADD2 3절 규칙의 대표(`E_representative.json`: 모든 타깃이 A+G):

| 모델 | d1 | d2 | elst | pauli | oi | disp | cpcm | cds | barrier |
|---|---|---|---|---|---|---|---|---|---|
| AW-A+G | 0.231 | 0.204 | 0.657 | 1.251 | 0.719 | 0.101 | 0.263 | 0.033 | 0.262 |
| GNN-A+G | 0.226 | 0.167 | 0.596 | 1.151 | 0.660 | 0.086 | 0.247 | 0.028 | 0.229 |
| TF-A+G | 0.243 | 0.206 | 0.655 | 1.244 | 0.707 | 0.102 | 0.246 | 0.031 | 0.266 |
| C-XGB | 0.232 | 0.186 | 0.644 | 1.255 | 0.700 | 0.099 | 0.256 | 0.034 | 0.260 |
| REP | 0.226 | 0.167 | 0.657 | 1.251 | 0.719 | 0.086 | 0.247 | 0.028 | 0.229 |
| null | – | – | – | – | – | – | – | – | – |

oi·cpcm의 단계 2 수치(설명용, 원래 라벨, g1은 5-3 표에 넣지 않는 rung이다):

| 모델 | oi para MAE | oi 민감도 | oi 위양성률 | cpcm para MAE | cpcm 민감도 | cpcm 위양성률 | 기준(0.13 / 0.7 / 0.1) |
|---|---|---|---|---|---|---|---|
| AW-A+G | 0.719 | 0.62 | 0.12 | 0.263 | 0.76 | 0.17 | 미달 |
| GNN-A+G | 0.660 | 0.77 | 0.17 | 0.247 | 0.81 | 0.11 | 미달 |
| TF-A+G | 0.707 | 0.69 | 0.14 | 0.246 | 0.76 | 0.13 | 미달 |
| C-XGB | 0.700 | 0.50 | 0.21 | 0.256 | 0.81 | 0.10 | 미달 |
| REP | 0.719 | 0.62 | 0.12 | 0.247 | 0.81 | 0.11 | 미달 |
| null | – | 0.00 | 0.00 | – | 0.00 | 0.00 | 미달 |

(민감도·위양성률은 FU2 `E_bend_rates.json`의 `all` 계열 = 5-2절과 같은 모집단.)

### 7-3. 검토자의 사후 탐색 (사전 등록 아님)
- HF@G1 + FU2 linear combination: oi para MAE 0.65 kcal/mol
- HF@DFT + FU2 linear combination: oi para MAE 0.26 kcal/mol
- 출처: docs/specs/D1LADDER.md 9 (reviewer-supplied numbers; no computation file in the repository). 원래 라벨·원래 H 기준이다.

### 7-4. 검토자의 기준 구조 HF 사전 점검 — 2-3절.

### 7-5. Phase C에서 나온 수치
- L-10(pass): 원래 라벨·원래 H의 LINE이 7-1 표(dft·g1 para MAE)를 1e-06 안에서 재현했다(C_tests.json `tol.L10`). L-6(pass): g1·S 원래 라벨의 입력 텐서·y·s_y와 XGB fold 0 예측이 FU2·rev 9 unit과 같다. smoke: 2 epoch 학습의 검증 짝 MAE(새 정답)는 `C_smoke_atoms.json`·`C_smoke_xgb.json`에 있다. 시험 짝 점수는 없다.

## 8. Phase C 결과
`analysis/d1_ladder/results/C_tests.json` sha256 `eda269b77874e0c591404bde234a00d337314b83ea4224288da05439bbdc6778` (commit 202814de), tests `d1-ladder-c1`, merge 2026-10-23 16:18:36: all_pass True, 범위 "stage 1: rungs dft, g1 (g2 not tested)", stage_verified 1, g2_verified False, 정답 `$LAD/F_pairs_cr2.csv`.

| L-1 | L-2 | L-3 | L-4 | L-5 | L-6 | L-7 | L-8 | L-9 | L-10 | L-11 | smoke |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |

- 검사 job: gpu job 1038438 (n058.hpc, git 825c8017, NVIDIA A10); cpu job 1038439 (n116.hpc, git 825c8017, 3.2.0).
- 예상 GPU 시간(FU2 unit 기록): 1단계 65.5 GPU-시간 (unit당 AW 0.62, GNN 2.30, TF 0.35).

## 9. 계획과 다르게 한 점·해석 (지금까지)
1. **시작 조건 0-2:** `analysis/d1_pair`가 `2d745d23`과 다른 파일은 `analysis/d1_pair/d1_features.py`, `analysis/d1_pair/fu2/atoms.py`이고 삭제는 0개, FU2 핵심 코드 5개 중 5개가 같다(`start_report.json` `0-2`). 결정 원문: "accepted: only the D1-G2 Phase T root-argument additions (T-1 / T-2 passed)".
2. **시작 조건 0-3:** d1-g2 쪽 부모 `0bb682d4`에 D1-HF 최종 결과(`6b9fbeb2`)가 없어서 d1-hf(`6b9fbeb2`)를 d1-ladder에 merge했다(merge commit `2e45b774`, "Merge d1-hf (6b9fbeb2: D1-HF final results) into d1-ladder (D1LADDER start condition 0-3)"). `analysis/d1_hf/results`와 `6b9fbeb2`의 차이: 없음(바이트 동일); `labels_d1_hf.parquet` sha256 `33227c333e08355a…`(`start_report.json` `0-3`).
3. **F0 F-1 (`F0_DECISION.md`, sha256 `2037e1807a8eb965…`):** 전역 182가 빠진 반응 132/9,266 = 1.42 % > 1 %. 명령서 16절이 F-1과 B6을 따로 적으므로 F-1에서 B6 실패(57개, 0.62 % ≤ 1 %: B6 자기 gate 통과)를 빼고 센다: 75/9,266 = 0.81 % ≤ 1 % → 진행(`inputs.py build dft --accept-f0-stop`). 빠진 반응은 dft rung의 사용 가능 반응에서 제외된다.
   - `F0_DECISION.md` 본문의 분류 수(engine gate 69, parse:fA –, B6 57)는 `F0_report.json` `F1.classes`(engine gate 69, parse:fA 6, B6 57)와 다르다. B6를 뺀 합계는 md 69, 보고서 75이다. 이 문서는 보고서 수치를 쓴다.
4. **Phase V 구현 결정(V_report.json `decisions`; 설명 뒤 원문):**
   - V7_rr: randomize-and-relax 시작마다 고정 seed를 넣었다(D1/autodE는 seed 없음). 재현성을 위한 이탈이며 알고리즘·인자는 같다. (원문: "autodE get_simanl_conformer x 300 round robin over the forms + autodE prunes; seeded RandomState(20261023+n)")
   - V8_xtb: 입체 서명을 확인할 수 없는 분자는 실패로 두고 V 기준을 쓰지 않는다(명령서 V-3 '입체가 키와 다르면 버림'을 적용하는 방법). (원문: "--opt tight --alpb water --gfn 2 --chrg q --uhf 0; graph iso to anchor; stereo3d_sig = the key's signature (unique D1 signature, else the Q E_min source's); not checkable -> molecule failed (no V replacement)")
   - V10_imag: 명령서 V-3 그대로: 다시 최적화한 뒤 허수 진동수가 하나라도 남으면 실패. D1 TS 단계 허용치(-50 cm⁻¹)를 썼을 때의 실패율은 민감도로만 보고한다(실패율 3.2 %, 허용치 적용 시 0.0 %). (원문: "any freq < 0 -> one 1.0 A displacement + Opt/Freq; any freq < 0 after that = failure 'imag' (spec V-3); the -50 cm-1 (config extra_imag_tol_cm) count is a reported sensitivity only")
   - V12_repro: 시험 묶음의 기준 구조 라벨·HF 입력을 그대로 다시 계산해 재현됨을 확인하는 검사를 더했다. (원문: "trial anchors rerun unchanged: |dE| <= 0.05 kcal/mol vs label and labels_d1_hf, else STOP")
   - V17_name: 명령서 8절의 `refs.py`(Q·V)를 `refs_q.py`(Q)와 `refs_v.py`(V)로 나눴다. (원문: "refs_v.py (task) instead of refs.py (spec 8)")
5. **XGB 환경:** `env_espley`(scikit-learn 1.3.1, xgboost 3.2.0; C_tests.json `parts.cpu`). 이 환경에는 torch가 없어 `train_xgb.py`가 `train_atoms.py`의 공통 규칙을 옮겨 쓰고, collect가 두 쪽의 시험·검증 짝 목록을 대조한다. 원문(XG-1): "env = env_espley (analysis/d1_pair/jobs/env.sh PYENV=espley; scikit-learn 1.3.1, xgboost 3.2.0, numpy 1.26.0, pandas 2.1.1 = the versions in every rev 9 T1__C__XGB unit json). It has no torch, so train_atoms.py (imports torch and fu2/train.py) cannot be imported: its shared rules (unit names, PRED_COLS, label / split / input-table loading, usable-pair filter, beta_fit, ADD3 gate, cache-root guard, fingerprint key names, L-4 audit hook) are re-implemented here line for line; train_atoms collect --with-xgb cross-checks the test-pair lists and the validation-pair sha256 of both files' units (spec 16 E, L-9), so a divergence is a STOP, not a silent difference."
6. **대표 고르기의 비대칭(MA-6n):** 5-1절.
7. **`evaluate.py`의 commit (명령서 8절 Phase C 코드):** ADD3는 `evaluate.py`가 commit되어 있기를 요구하지 않는다(지금 git 상태: commit됨(HEAD와 같음), sha256 `29b997cc954c66f8ea1ce0654b4f828408756a5cf17c7fabd7a764a44e7aff4b`). E1 코드를 학습과 따로 쓰고 있기 때문이다. `C_tests.json`은 `evaluate.py`를 고정하지 않으므로, L-10이 시험한 LINE 함수(`line_beta`·`line_predict`)의 소스 sha256 `2fafcba7536fae30043da7b07b91a7239a0ba2a9a5df064b4045caf76f34a230`을 json `line_functions`에 적었다. 이 값은 HEAD 판과 같다. E1은 자기 소스가 이 값과 다르면 멈추고, 쓴 `evaluate.py`의 sha256을 `E1_report.json`에 적는다.
8. **FU2 Phase E 수치(7-2절)는 commit 전 파일이다:** `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/analysis/d1_pair/results/fu2`의 E_* 파일 git 상태 commit 뒤 수정됨 1개, commit 안 됨 22개. 설명용으로만 옮겼고 판정에 쓰지 않는다.

명령서가 정하지 않아 정한 값은 학습 코드 docstring의 결정 MA-1–MA-18 (19개), XG-1–XG-18 (18개), `inputs.py` P1–P23 (23개), `refs_q.py` D1–D14 (14개), `refs_v.py` V1–V20 (20개), `tests/run_tests.py` C-1–C-24 (24개)에 있다. json `trainer_decisions`, `q.decisions`, `v.decisions`, `p_report.decisions`, `c_tests.decisions`에 그대로 옮겼다.

## 10. 2단계(g2) 보완 규칙 (이 문서는 바꾸지 않는다)
- **이 md는 바이트 하나도 바꾸지 않는다:** 1단계 unit 지문의 `add3`가 이 md의 sha256이고 collect가 그것을 다시 비교하기 때문이다. 2단계 학습 전에 보완판 `results/PREREG_D1PAIR_ADD3_G2.md`를 새 파일로 쓴다.
- **commit된 이 json에 허용되는 수정은 하나뿐이다:** 아래 g2 파일을 `inputs_sha256`·`inputs_roles`에 **새 항목으로 더하고**, 새 최상위 키 `supplement`(보완판 md 경로·sha256·commit)를 더한다. 이미 있는 항목과 키는 그대로 둔다(1단계 process의 gate는 g2 열에만 속한 파일을 열지 않고, 여러 rung에 걸친 `P_012.csv`는 바뀌지 않는 한 통과한다). 더할 항목:
  - `$LAD/inputs/g2.parquet` (input table g2 (need of a g2 process))
  - `$LAD/inputs/sets/P_g2.csv` (pair set P_g2)
  - `$LAD/inputs/sets/P_012.csv` (pair set P_012)
  - `analysis/d1_g2/results/fu2/T_report.json` (tokens g2: T_report.json)
  - `analysis/d1_g2/results/fu2/T_manifest.csv` (tokens g2: T_manifest.csv)
  - `analysis/d1_g2/results/fu2/Q_report.json` (tokens g2: Q_report.json)
  - `analysis/d1_g2/results/fu2/S_splits.json` (tokens g2: S_splits.json)
  - `$G2/X182_d1_g2.parquet` (global 182 features g2)
- 학습 코드의 gate는 이 md와 json만 읽고 보완판 md는 확인하지 않는다. 보완판을 만드는 명령(`prereg.py`에 추가할 `supplement`, 아직 없음)이 보완판의 commit을 확인하고, E2가 보완판 commit·sha256을 다시 확인해야 한다.
- 시작 조건(명령서 12절): D1-G2 `X182_d1_g2.parquet`, `$G2/fu2/` 토큰, `labels_d1_hf_g2.parquet`가 각 gate를 통과하고 commit됨, `origin/d1-g2` merge. 그 뒤 P(g2 입력표, H 재조립, P_g2, P_012) → C(g2 검사, `g2_verified`) → 보완판 → M2(40 unit) → E2.
- g2 H 재조립의 constrained dipole 기준: 명령서 Q-2 그대로 '묶지 않음'(고정 이면각 상태를 키에 붙임; `inputs.py build g2 --g2-q-keys`). 이 키는 D1(DFT) 기준 구조로 정해지므로 g2 H에 DFT 정보가 일부 들어간다(P20). 보완판에서 확정·기록한다.
- g2는 G2가 라벨과 다른 TS를 찾은 짝도 그대로 학습·평가하고, D1-G2 Phase C의 "같은 TS" 판정에 따른 나눔은 설명용으로만 보고한다. 적용 범위(P_main 시험 짝 중 g2로 예측 가능한 비율)를 보고한다.

## 11. 멈춤 규칙 (명령서 16절)
- **A:** ADD3 commit 전 학습 시도(`gate_add3`가 exit 3).
- **M:** gate M-1′ 실패; 캐시 지문 불일치; 튜닝·β에 시험 짝 사용(L-2·L-8 assert).
- **E:** 같은 집합에서 비교하는 모델 사이 시험 짝 목록 불일치; unit 지문의 `add3`·`code_sha256`이 이 문서(json `code.fingerprint_keys`)와 다름.
- **원칙:** 결과를 본 뒤 Q·V 규칙, rung·arm·모델·grid·β 규칙·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

