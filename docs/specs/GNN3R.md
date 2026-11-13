# GNN3R — 새 g2(g2r) 입력과 다시 만든 짝(반응 약 8,604개)으로 GNN3 다시 학습 — Claude Code 실행 지시서

**저장소:** `origin/g2r` HEAD(`df3deb3`)에서 새 브랜치 **`gnn3r`**를 만든다. 새 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r`(이하 `$WT`).
**이 파일:** `docs/specs/GNN3R.md`. 참조 목록 두 개 `docs/specs/gnn3r_ref/{ref_reactions_est.csv, ref_pairs_est.csv}`와 **같이** commit한다.
**새 코드:** `analysis/gnn3r/` · **결과:** `analysis/gnn3r/results/` · **scratch:** `$GR=/gpfs/tmp_cpu2/yeseo1ee/gnn3r`
**읽기만:** `$G2R=/gpfs/tmp_cpu2/yeseo1ee/g2r`, `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`, `$D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair`, D1 run(`/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`), `gnn3/` 패키지와 `analysis/` 아래 기존 코드 전부. **기존 코드는 고치지 않는다.** import해서 부르고, 바꿀 동작은 `analysis/gnn3r/`에 둔다.

**순서:** 1 시작 조건 → 2 브랜치·폴더 → A(행 속성, B′ c3 수정) → B(짝 = 데이터셋 고정) → S(분할·속성 파일) → D(데이터 확인) → K(교차검증 학습, GPU) → E(평가) → W(최종 모델, GPU) → R(보고·manifest·commit·push) → **멈춤**
- 13절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출, git, `squeue`·`scancel`만 한다.
- GNN3 모델·학습 규칙(`gnn3/gnn3/constants.py`)은 바꾸지 않는다. 바뀌는 것은 **데이터(입력 g2r, 짝 목록, 분할)뿐**이다.

---

## 0. 무엇을, 왜

### 0-1. 목표

1. 새 g2 입력(`$G2R`, 사용 가능 반응 8,617개)과 cr2 라벨로 짝을 0-2절 규칙대로 다시 만들어, **반응 약 8,604개 / 짝 약 8,355개** 데이터셋을 만든다.
2. 데이터셋은 B 단계에서 파일(`results/F_pairs_r.csv`)로 **고정**하고 sha256을 적는다. 이후 모든 단계와 앞으로의 재학습은 이 파일만 읽는다.
3. 이 데이터셋으로 GNN3를 5겹 교차검증(K)하고 평가(E)한 뒤, 최종 모델(W)을 학습한다.

### 0-2. 확정 결정 (사용자, 2026-11-13)

| # | 항목 | 결정 |
|---|---|---|
| D-1 | 대상 | D1 비D″ 라벨 9,266 − `flag_short_forming` 9 − `flag_regio_ambiguous` 5 = **9,252** (D1G2R D-1). `flag_async`, cr2 전 기준 초과(> 20) 행 포함 |
| D-2 | 입력 | **g2r** (`$G2R/lf`). 사용 가능 = `analysis/g2r/results/U_usable.csv`의 `usable` True **8,617**개 (규칙: 대상 ∧ L `.done` ∧ 유한한 X182 행 ∧ 토큰) |
| D-3 | 라벨 | **cr2** (`$LAD/labels_d1_cr2.json`). 짝 라벨 dd_t = y_t(a) − y_t(b), 10개 라벨(`d1pair_common.D1_TCOL`) |
| D-4 | 라벨 기반 행 제외 | **없음.** 기준 초과 > 20과 위생 규칙은 cr2 값에서 0건을 거르므로(위생은 −0.5 kcal/mol 허용) 적용하지 않는다. D1PAIR의 G1 특성 필터(`in_P_feat`)도 적용하지 않는다(g2r 입력이 따로 있음) |
| D-5 | B′ c3 | **수정해서 포함.** B′(skeleton `Bp`) 행의 c3를 C4의 메틸 기준으로 다시 정한다(0-5절). 같은 c3끼리만 짝이 되므로 s1·s2 입체가 섞이지 않는다 |
| D-6 | 짝 규칙 | **D1PAIR 규칙 그대로**(`d1_pairs.build`, `d1_anchor_pairs.build`를 import): 아릴 X → 같은 계열 F의 Ph 반응이 기준, 직접 X → H 반응이 기준. 계열 키 F = (dipole, skeleton, series, cfg, orient, face, c3) |
| D-7 | 기준이 없을 때 (anchor) | 기준이 DFT에 없거나(**D1PAIR anchor 계열**) g2r에서 쓸 수 없으면, 같은 F의 사용 가능 행 가운데 하나를 기준 b로 쓴다. 순서: Xb **Me → CO2Me → tBu → CHO → Ts → NMe2** → 나머지(Xb 이름순). 같으면 위치 p < m < o < direct, 그다음 job_id |
| D-8 | P-4 입체 검사 | v2 mismatch 짝은 버린다 (`d1pair_common.P4_DECISION` = "drop_v2_mismatch (user 2026-10-19)") |
| D-9 | 겹(fold) | 발표된 T1 core → fold 표(`analysis/d1_pair/results/S_folds.csv`의 `fold_T1`)를 그대로 쓴다. −1(anchor만 있는 core 6개)은 늘 학습. test(k) = core의 fold가 k인 짝 전부(기준 + anchor), train(k) = 나머지 전부. 안쪽 검증 = 학습 짝을 core로 묶은 GroupKFold(4)의 첫 분할 |
| D-10 | 평가 집합 | `in_P_main` = `in_P_sens` = 기준 짝 전부, `in_T_main` = 짝 전부 |
| D-11 | 모델 | GNN3 그대로(격자 {4, 6}층 × {64, 128}폭, seed 5개, 4 출력 strain·inter·solv·barrier). RTX 3090에서만 학습 |

### 0-3. 8,604는 어떻게 나왔나 (로컬 추정, 2026-11-13)

**(a) 반응 깔때기**

| 단계 | 빠짐 | 남음 | 근거 |
|---|---|---|---|
| D1 비D″ 라벨 | — | 9,266 | `labels_d1.json` |
| short_forming 9, regio_ambiguous 5 (겹침 0) | 14 | 9,252 | D-1, `analysis/g2r/results/X_targets.csv` |
| g2: TS 없음 487, autodE 오류 108, 입체 그래프 불일치 3, 허수 진동 없음 1 | 599 | 8,653 | `U_usable.csv` `missing_stage` X (D1G2R D-6) |
| g2: TS 검사 16, 조각 분할 3 | 19 | 8,634 | 같음, X 618 = 599 + 19 (D1G2R D-4) |
| g2: 기준 구조 코드 오류 20 중 복구 못 한 17 | 17 | **8,617** | `missing_stage` R (D1G2R D-3, 복구 3) |
| 짝 (아래 (b)) | 13 | **8,604** | 이 지시서 B 단계 |

**(b) 사용 가능하지만 짝이 없는 13개 (로컬 추정)**

| 이유 | 수 | job_id |
|---|---|---|
| P-4 v2 mismatch (D1PAIR `P4_drop.csv`) | 4 | T08439 (CY/Bp m-CN), T09972 (AY/B o-OH), T10679 (nitrone/Bp o-Ac), T10704 (nitrone/Bp o-CO2Me) |
| 계열 없음: D1PAIR에서 어느 계열에도 들지 못함(F 중복, H 후보 둘 이상 등) 6, async라 로컬 추정 계열이 없음 1(T02833) | 7 | T01507, T02374, T02510, T02573, T02833, T11185, T11220 |
| 계열에 사용 가능한 반응이 자기 하나뿐 | 2 | T01073 (AY/A Ac), T09850 (AY/AppE o-CO2Me) |

- T08439·T10679·T10704는 c3가 비어 있던 B′ 행이다. c3 수정 뒤에는 기준이 바뀔 수 있어 서버의 P-4 결과가 달라질 수 있다.

**(c) 8,604의 구성**

| 출처 | 반응 | 설명 |
|---|---|---|
| D1PAIR 계열에서 이미 짝이 되던 반응 | 8,094 | 2026-10-19 D1PAIR 기준 짝 + anchor 짝의 계열, g2r 사용 가능 |
| B′ c3 수정으로 살아난 반응 | 356 | 예전에 c3가 비어 s1·s2가 같은 키 → 중복으로 둘 다 버려짐 (D-5) |
| `flag_async` (D1PAIR에서 제외됐던 행) | 149 | D-1 |
| cr2 전 기준 초과 > 20 (D1PAIR에서 제외됐던 행) | 5 | D-4 |
| **합계** | **8,604** | |

**(d) 짝 8,355 = 기준 짝 6,977 + anchor 짝 1,378, 계열 323개, 기준 반응(b) 154개 + anchor 반응**

| fold_T1 | anchor | 기준 | 합 |
|---|---|---|---|
| −1 (늘 학습) | 170 | 0 | 170 |
| 0 | 269 | 1,417 | 1,686 |
| 1 | 305 | 1,401 | 1,706 |
| 2 | 171 | 1,243 | 1,414 |
| 3 | 36 | 1,308 | 1,344 |
| 4 | 427 | 1,608 | 2,035 |
| **합** | **1,378** | **6,977** | **8,355** |

| 쌍극자 | anchor | 기준 | 합 | | 위치 | anchor | 기준 | 합 |
|---|---|---|---|---|---|---|---|---|
| nitrone | 266 | 1,986 | 2,252 | | p | 328 | 1,839 | 2,167 |
| azide | 260 | 1,536 | 1,796 | | m | 325 | 1,879 | 2,204 |
| nitrile oxide | 44 | 1,663 | 1,707 | | o | 405 | 1,705 | 2,110 |
| azomethine ylide | 488 | 966 | 1,454 | | direct | 320 | 1,554 | 1,874 |
| carbonyl ylide | 320 | 826 | 1,146 | | | | | |

참고: 예전 GNN3 학습 짝(`S_folds.csv` `in_T_main`) 7,970개 / 반응 8,210개. 새 목록과 (a, b)가 같은 짝 6,444개.

**(e) 8,602 → 8,604 정정.** 처음 센 8,602는 B′의 Ph 반응을 **직접(direct) 치환기**로 잘못 넣어 셌다. D1PAIR 규칙(`build_rowattr.py`: X = Ph → `pos = ref_aryl`)에서 Ph는 아릴 계열의 **기준**이지 치환기가 아니다. 규칙대로 하면 azomethine ylide/B′/exo의 Ph 두 개(T01421 trans, T01422 cis)가 기준으로 들어오고(+2 반응), B′ 아릴 짝 290개가 anchor → 기준 짝으로 바뀐다. 이 지시서는 규칙대로 센 **8,604**를 기대값으로 쓴다.

**(f) 서버 수가 조금 다를 수 있는 이유 (그래서 B 단계 허용 범위 ±2 %)**
1. 로컬에는 cfg(TS 쌍극자 배치, `ts_dipole_config`)가 D1PAIR 짝에 들었던 행만 있다. 그 밖의 **510행**(B′ 356 + async 149 + 초과 5)은 계열을 (dipole, skeleton, series, face, 아릴/직접, c3)로 추정했다(`ref_reactions_est.csv`의 `family_basis` = `estimated_family`). 서버는 진짜 F로 묶으므로 일부 계열이 cfg로 갈라질 수 있다.
2. 새로 생긴 짝의 P-4는 DFT TS가 서버에만 있어 로컬에서 돌리지 못했다. 로컬은 D1PAIR `P4_drop.csv`만 반영했다.
3. 직접 치환기의 H 후보가 이제 9,252행(async·초과 H 행 포함)에서 나온다. D1PAIR는 9,035행에서 골랐다. 후보 수가 바뀌는 행이 몇 개 있을 수 있다.

### 0-4. 같은 데이터셋을 보장하는 방법

1. **입력 고정:** 시작 조건(1절)에서 모든 입력 파일과 짝 코드의 sha256을 확인한다. 하나라도 다르면 STOP.
2. **규칙 결정성:** 무작위 요소는 c3의 3D 배치(RDKit `EmbedMolecule(randomSeed=7)` + MMFF, 기존과 같음) 하나뿐이다. A-2(기존 c3와 같음)·A-4(로컬 c3와 같음)로 확인한다. 정렬 순서와 pair_id 순서는 4-5절에 못 박는다. B 단계를 두 번 돌려 출력 sha256이 같아야 한다(B-R).
3. **참조 목록 대조:** 로컬 추정 목록(`docs/specs/gnn3r_ref/`)과 서버 결과를 반응·짝 단위로 대조해 다른 행마다 서버 쪽 이유를 적는다(`results/B_diff_vs_est.csv`). 총수가 허용 범위 밖이면 STOP.
4. **고정 후 재사용:** `results/F_pairs_r.csv`를 commit하고 sha256을 `results/B_label_report.json`과 `results/gnn3r.json`에 적는다. GNN3 데이터 로더(`gnn3.data.load_pair_labels`)는 sha256이 이 보고와 다르면 읽기를 거부한다. 앞으로 재학습은 B를 다시 돌리지 않고 이 파일을 쓴다.

참조 목록 sha256:

| 파일 | 행 | sha256 |
|---|---|---|
| `docs/specs/gnn3r_ref/ref_reactions_est.csv` | 8,604 | `95e69518023a59420f74d59be7687585af83f7eb86030c6f28c4dca1b7ea901b` |
| `docs/specs/gnn3r_ref/ref_pairs_est.csv` | 8,355 | `77b78095cf8fec29d5f47dcd081419442d4e582b85f66ff797a8ea0c0fb16e0f` |

- `ref_reactions_est.csv`: job_id, rxn_id, dipole, skeleton, series, attach, R1, R2, face, source(`pairable_D1PAIR`·`Bp_c3_fix`·`async`·`ex_gt20_restored`), est_group, c3_est(B′ 행만), family_basis
- `ref_pairs_est.csv`: a_job, b_job, ref_kind, est_group, a_rxn, b_rxn

### 0-5. B′ c3 수정 — 무엇이 문제였나

- **비유:** C3의 치환기 X가 고리 면 위인지 아래인지 말하려면 옆 탄소 C4의 "기준 깃발"과 비교해야 한다. 다른 골격에서는 C4에 깃발이 하나라 바로 비교한다. B′에서는 C4에 깃발이 둘(CH3, 다리머리 탄소)이라 예전 코드가 판정을 포기하고 c3를 빈칸으로 두었다. 그래서 서로 다른 두 입체(s1, s2)가 같은 이름표를 받았고, D1PAIR의 중복 제거(`keep=False`)가 둘 다 버렸다.
- **규칙:** `build_rowattr.c3_orientation`과 같다. 단 skeleton이 `Bp`이고 C4의 고리 밖 무거운 원자(`ms4`)가 2개이며 그중 **정확히 하나가 메틸**(탄소, 무거운 이웃이 C4 하나뿐)이면, 그 메틸을 C4의 기준 치환기로 쓴다. 배치(seed 7, MMFF), 고리 법선(SVD), 판정(u·v > 0 → cis)은 그대로다.
- **로컬 결과:** B′ 행 356개(참조 목록 안) 모두 판정됨, cis 161 / trans 195. 수정 뒤 F 안 기준 중복 0.

---

## 1. 시작 조건 (하나라도 아니면 STOP) → `results/S0_start.json`

| # | 조건 |
|---|---|
| 1-1 | `origin/g2r` = `df3deb3` 또는 그 뒤. `analysis/{g2r, d1_pair, d1_ladder}`와 `gnn3/`가 있다 |
| 1-2 | g2r 입력 sha256 = `analysis/g2r/results/g2r.json`의 값: `$G2R/lf/X182_lf.parquet` `66c40316…`, `labels_lf.json` `3f2bf2cc…`, `input_meta_lf.csv` `a3968398…`, `analysis/g2r/results/lf/fu2_lf/T_manifest.csv` `f424d12d…`(8,617행). `T_report.json`의 `manifest_sha256`·`q_report_sha256`이 맞다. `$G2R/lf/fu2_lf/tokens/`에 npz 8,617개 |
| 1-3 | `analysis/g2r/results/U_usable.csv` sha256 `8a4b4f478fea9d9e58979ae4253d3d61c9ccfbae7f975accf16c7d3cfbd96db1`, 9,252행, `usable` True 8,617. `X_targets.csv` sha256 `8e89a3920704519559794b4dba03bca2a9455345861a704d4e14483253b2b7a5`, 9,252행 |
| 1-4 | `$LAD/labels_d1_cr2.json` sha256 `7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755`. `analysis/d1_ladder/results/V_report.json` `all_pass` true. 9,252개 대상 모두 10개 라벨(`D1_TCOL`)이 유한하다. `$LAD/F_pairs_cr2.csv` sha256 `f2fe5cab…`(V_report `outputs`) |
| 1-5 | `$D1PAIR_SCRATCH/P/rowattr.pkl.all.pkl` 9,496행(9,252개 대상 모두 포함), `rowattr.pkl` 9,035행 |
| 1-6 | sha256: `analysis/d1_pair/results/pairs_d1.csv` `227063285d65a05195a0d662df040618ac52bd175718837423ded2343c83ecce`(7,253행), `pairs_d1_anchor.csv` `2121c684bd11a77d4b59128432d3e6e22d3dab23081bd25bd9864e5172a9bb6c`(1,159행), `P4_stereo.csv` `5fb1155b4cc35ad39b4721b7bc8e1900cc5d7c61c9fd7f6b7089e5633035dac5`, `P4_drop.csv` `91a50b8fabf2b94a1c4d499bff50edc323fee1e0af2fd4450fd22b78ba394ee6`(29행), `S_folds.csv` `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7`, `analysis/d1_pair/sigma_hansch1991.csv` `2ba63224928301657f09707cb3be1b0e05a6c69ef1e733b490b9a4ac6261d06f` |
| 1-7 | 짝 코드 sha256: `analysis/d1_pair/d1_pairs.py` `424c2bc24b4ed5111c35a00688a88d43f47dffb2e61e50c5326256f31be603e1`, `d1_anchor_pairs.py` `f979e1e336df850a0b55d6ed762718b035bf2b151468263919b53a5aa40bac18`, `build_rowattr.py` `81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4`, `d1_phase_p.py` `630e0d9bb7f57ea31b542f82e2533965f25aff931ef4d528b5c1f3636e92cefd` |
| 1-8 | `labels_d1.json` sha256 = `d1pair_common.D1_LABELS_SHA`(`1eaa4171…`). 사용 가능 8,617개 모두 `$D1PAIR_SCRATCH/d1_jobs/<job_id>/{ts.xyz, sp/eda.inp}`가 있다(P-4가 읽음) |
| 1-9 | 참조 목록 두 파일의 sha256 = 0-4절 표 |
| 1-10 | 환경: `analysis/d1_pair/jobs/py.slurm` + `PYENV=reactot`(RDKit 판 기록), `PYENV=espley`(sklearn 1.3.1), `gnn3/jobs/run.slurm`(env_d1fu2, `.d1fu2_env_ok`). `sinfo -p gpu1`에 rtx3090 노드가 있다 |
| 1-11 | `$GR`가 없거나 비어 있다. 디스크 여유 20 GB 이상 |

---

## 2. 브랜치 · 폴더 · 경로

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction        # 주 저장소 (실제 경로 확인)
git fetch origin g2r:refs/remotes/origin/g2r
git worktree add -b gnn3r /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r origin/g2r
sbatch -p cpu1,cpu2 --time=48:00:00 --wrap "mkdir -p $GR/logs $GR/A $GR/B $GR/cv $GR/final"
```
- 이 지시서와 참조 목록 두 개를 commit한다(1-9 확인 뒤).
- 기존 job 스크립트의 `#SBATCH --output`은 다른 폴더를 가리킨다. 제출할 때 항상 `--output=$GR/logs/%x_%j.out`(배열은 `%x_%A_%a.out`)으로 덮어쓴다.
- 경로는 `analysis/gnn3r/gr_common.py` 한 곳에 둔다. 모든 단계가 이 모듈에서 경로를 읽는다.

| 이름 | 값 |
|---|---|
| `GR` | `/gpfs/tmp_cpu2/yeseo1ee/gnn3r` |
| `G2R` | `/gpfs/tmp_cpu2/yeseo1ee/g2r` |
| `TOKENS` | `$G2R/lf/fu2_lf/tokens` |
| `TOKEN_REPORTS` | `$WT/analysis/g2r/results/lf/fu2_lf` (T_report.json, T_manifest.csv, Q_report.json) |
| `X182` | `$G2R/lf/X182_lf.parquet` |
| `USABLE` | `$WT/analysis/g2r/results/U_usable.csv` |
| `TARGETS` | `$WT/analysis/g2r/results/X_targets.csv` |
| `LABELS_CR2` | `$LAD/labels_d1_cr2.json` |
| `ROWATTR_ALL` | `$D1PAIR_SCRATCH/P/rowattr.pkl.all.pkl` |
| `RES` | `$WT/analysis/gnn3r/results` |

---

## 3. Phase A — 행 속성과 B′ c3 수정 (`analysis/gnn3r/rowattr_r.py`, `PYENV=reactot`, CPU)

`build_rowattr.py`는 import하면 바로 실행되는 스크립트라 부를 수 없다. 아래 함수만 **글자 그대로 복사**하고, 복사한 원본의 sha256(1-7)을 파일 머리에 적는다: `any_bond_query`, `backbone_in_product`, `termini`, `c3_orientation`(+ `DIP`, `PAT`).

1. `ROWATTR_ALL`을 읽어 `TARGETS`의 job_id 9,252개만 남긴다(행 순서 = job_id 오름차순).
2. `need` = dipole ∈ {nitrone, azomethine ylide, carbonyl ylide} ∧ series = R1 ∧ skeleton ∈ {AppE, AppZ, B, Bp} (`build_rowattr.py`와 같은 식). 9,252행에서 2,667행, 그중 Bp 922행(로컬 집계).
3. 복사한 `c3_orientation`으로 `need` 행 전부의 c3를 다시 계산한다 → 저장된 c3와 비교(A-2).
4. `c3_orientation_bp`(0-5절 규칙: `ms4`가 2개이고 정확히 하나가 메틸이면 그 메틸 사용)를 `need` ∧ skeleton = Bp 행에만 적용해 c3를 바꾼다. 다른 행의 c3는 저장된 값 그대로 둔다.
5. 출력: `$GR/A/rowattr_r.pkl`(9,252행, 열은 `rowattr.pkl.all.pkl`과 같음), `results/A_c3_bp.csv`(job_id, c3_old, c3_new, ms4 원소·차수), `results/A_report.json`(RDKit 판, 수, 검사).

| # | 검사 | 실패 시 |
|---|---|---|
| A-1 | 9,252행, job_id 집합 = `X_targets.csv` | STOP |
| A-2 | 복사한 `c3_orientation`의 재계산 값 = 저장된 c3, `need` 행 100 % | STOP (복사가 원본과 다르거나 RDKit 판이 다름) |
| A-3 | Bp `need` 행 922개 모두 c3 ∈ {cis, trans} | STOP |
| A-4 | 참조 목록에 있는 B′ 356행의 c3 = `ref_reactions_est.csv`의 `c3_est` | STOP |
| A-5 | Bp `need` 행의 c3를 뺀 모든 열이 `rowattr.pkl.all.pkl`과 같다 | STOP |
| A-6 | 9,252행에서 `pos = ref_aryl` 행이 F 안에서 중복되는 경우: 수정 전 수(보고), 수정 뒤 **0** | 수정 뒤 > 0이면 STOP |

---

## 4. Phase B — 짝 만들기 = 데이터셋 고정 (`analysis/gnn3r/pairs_r.py`, `PYENV=reactot`, CPU)

### 4-0. 기존 규칙 재현 (B-0)

`d1_pairs.build(rowattr.pkl, sigma)`와 `d1_anchor_pairs.build(rowattr.pkl, sigma)`(둘 다 import, 수정 없음)를 D1PAIR 입력 그대로 돌려, `pairs_d1.csv`(7,253행)와 `pairs_d1_anchor.csv`(1,159행)의 (pair_id, a_job, b_job)를 순서까지 그대로 다시 만든다. 다르면 STOP.

### 4-1. 계열 만들기 (9,252행, 사용 가능 여부와 무관)

`r` = `$GR/A/rowattr_r.pkl`.
1. `Pref = d1_pairs.build(r, sigma)` → 기준 짝 후보. 아릴: 같은 F의 유일한 Ph(`ref_aryl`) 반응이 b. 직접: dipole·skeleton·face가 같은 H(`ref_direct`) 후보를 orient → cfg → c3 순으로(후보 모두 값이 있을 때만) 좁혀 정확히 1개일 때 b.
2. `A0 = d1_anchor_pairs.build(r, sigma)` → anchor 계열의 **구성원만** 쓴다(그 안의 b 선택은 버린다). 아릴: F에 Ph 행이 아예 없는 계열. 직접: H 후보가 0개인 직접 행을 F로 묶은 계열. F + Xb + pos 중복 행은 둘 다 버림(`keep=False`).
3. `d1_anchor_pairs.py` 본문과 같은 검사: `Pref`와 `A0`의 반응이 겹치지 않는다(B-1).
4. 계열 목록을 만든다. 각 계열 = (종류, 기준 반응 또는 없음, 구성원):
   - 기준 계열: `Pref`의 b마다 하나. 구성원 = 그 b를 가진 a들. 기준 = b.
   - anchor 계열: `A0`의 `fam`마다 하나. 구성원 = a ∪ b. 기준 없음.

### 4-2. 사용 가능 반응으로 줄이고 b 고르기

사용 가능 = `USABLE`의 `usable` True(8,617).
1. 기준 계열, **기준이 사용 가능:** 사용 가능한 구성원 a마다 짝 (a, 기준), `ref_kind = reference`.
2. 기준 계열, **기준이 사용 불가:** 사용 가능한 구성원을 F로 다시 묶는다(직접 계열은 F가 섞일 수 있음). 묶음마다 아래 3을 적용한다.
3. anchor 계열(과 2의 묶음): 사용 가능한 구성원이 2개 이상이면 D-7 순서의 첫 행을 b로, 나머지를 a로 한다. `ref_kind = anchor`. 1개 이하면 짝 없음.
   - 순서 키 = (Xb 순위: Me 0, CO2Me 1, tBu 2, CHO 3, Ts 4, NMe2 5, 그 밖 6; 위치 순위: p 0, m 1, o 2, direct 3; Xb; job_id), 오름차순 첫 행.
4. 기준 계열의 b 선택 규칙은 바꾸지 않는다. 사용 가능하면 언제나 Ph·H가 b다.

### 4-3. P-4 입체 검사

`d1_phase_p.ring_info`, `d1_phase_p.compare`를 import해서(수정 없음) 4-2의 짝 전부에 돌린다. 라벨 레코드는 `labels_d1.json`(`formed_pairs_ts`), TS는 `$D1PAIR_SCRATCH/d1_jobs`. 반응별 결과는 `$GR/B/p4_cache.pkl`에 저장해 다시 쓴다.
- `v2 == "mismatch"`인 짝을 버린다(D-8). `ring_fail`·`ring_elements_differ`·`no_relative_info`는 남기고 수만 센다(D1PAIR와 같음).
- 버린 짝의 a는 다른 b로 다시 짝짓지 않는다.
- D1PAIR와 (a_job, b_job)가 같은 짝은 v2가 `P4_stereo.csv`의 값과 같아야 한다(B-3).

### 4-4. 짝 라벨

`LABELS_CR2`에서 dd_t = y_t(a) − y_t(b), t = `D1_TCOL`의 10개(d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost, barrier), kcal/mol, float64 그대로(반올림 없음).

### 4-5. pair_id와 열

- 짝 정렬 키: (ref_kind: reference 먼저, core, fam, 위치 순위 p·m·o·direct, Xb, a_job). 기준 짝 `GRP00000`부터, anchor 짝 `GRA00000`부터 번호.
- `fam` = F 7개를 "/"로 이은 값(D1PAIR와 같음), anchor 짝은 끝에 "/anchor". `core` = dipole + "/" + skeleton. F는 a의 값.
- `sigma` = pos가 m이면 σm(Xb), 그 밖은 σp(Xb) (`sigma_hansch1991.csv`, 없으면 빈칸).
- `results/F_pairs_r.csv` 열: pair_id, ref_kind, a_job, a_rxn, b_job, b_rxn, core, fam, dipole, skeleton, series, cfg, orient, face, c3, pos, Xb, sigma, anchor_Xb, anchor_pos, p4_v2, dd_d1, dd_d2, dd_elst, dd_pauli, dd_oi, dd_disp, dd_cpcm, dd_cds, dd_c_ghost, dd_barrier. 쓰기: `to_csv(index=False, float_format=None)`, 줄바꿈 `\n`, 임시 파일 → `os.replace`.

### 4-6. 그 밖의 출력

| 파일 | 내용 |
|---|---|
| `results/B_reactions.csv` | 데이터셋 반응: job_id, rxn_id, core, role(a/b), n_pairs, fam |
| `results/B_row_status.csv` | 9,252행 전부: job_id, rxn_id, usable, status, fam. status는 아래 순서로 처음 맞는 것 하나: `not_usable` → `dedup_dropped`(F + Xb + pos 중복) → `ref_duplicated_family` → `H_ambiguous`(H 후보 2개 이상) → `no_family`(그 밖에 계열 없음) → `singleton` → `p4_dropped` → `in_dataset` |
| `results/B_diff_vs_est.csv` | 참조 목록과 다른 반응·짝: kind(reaction/pair), id, in_est, in_server, est_group, server status·fam, family_basis |
| `results/B_report.json` | 수(계열·짝·반응, ref_kind·dipole·pos·fold별), 4-3 P-4 집계, 검사 결과, 입력·출력 sha256 |

### 4-7. 검사

| # | 검사 | 실패 시 |
|---|---|---|
| B-0 | 4-0 재현: (pair_id, a_job, b_job) 순서까지 같음 | STOP |
| B-1 | `Pref`·`A0` 반응 겹침 0 | STOP |
| B-2 | 모든 짝: a, b 사용 가능, a ≠ b, core 같음, a_job 중복 0, a 집합 ∩ b 집합 = ∅. 아릴 기준 짝과 anchor 짝은 a·b의 F 7개가 같고, 직접 기준 짝은 dipole·skeleton·face가 같다(H 행의 c3·orient는 비어 있을 수 있음). F 열의 빈 값은 D1PAIR처럼 `fillna("")` 뒤 비교 | STOP |
| B-3 | D1PAIR와 같은 짝의 P-4 v2 = `P4_stereo.csv` | STOP |
| B-4 | D1PAIR(`F_pairs_cr2.csv`)와 (a_job, b_job)가 같은 짝: dd 10개 차이 ≤ 1e-9 | STOP |
| B-5 | 모든 짝: \|dd_barrier − (나머지 9개 합)\| ≤ 0.2 kcal/mol, 모든 dd 유한 | STOP |
| B-6 | 반응 수 8,432 – 8,776 (8,604 ± 2 %), 짝 수 8,188 – 8,522 (8,355 ± 2 %) | 범위 밖이면 STOP (`B_diff_vs_est.csv` 요약 첨부) |
| B-7 | `B_row_status.csv`: 9,252행, `in_dataset` 수 = 데이터셋 반응 수, `not_usable` = 635 | STOP |
| B-R | 같은 job에서 새 python 프로세스로 B를 한 번 더 돌려 `$GR/B/rerun/`에 쓴다(P-4 캐시 사용 가능) → `F_pairs_r.csv`, `B_reactions.csv`, `B_row_status.csv` sha256이 `results/`의 것과 같음 | STOP |

- 모든 검사가 통과하면 그때만 `results/B_label_report.json`을 쓴다: `{"all_pass": true, "outputs": {"<F_pairs_r.csv 절대 경로>": "<sha256>"}, "gates": {...}}`. (`gnn3.data.load_pair_labels`가 이 형식을 읽는다.)

---

## 5. Phase S — 분할 · 평가 속성 (`analysis/gnn3r/splits_r.py`, `PYENV=espley`, CPU)

1. core → fold 표 = `S_folds.csv`의 (core, fold_T1). core마다 값이 하나다(50개). 짝의 fold_T1 = 그 core의 값.
2. `results/S_folds_r.csv`: pair_id, ref_kind, a_job, a_rxn, b_job, b_rxn, core, dipole, skeleton, series, cfg, fam, pos, Xb, sigma, in_P_main(= ref_kind == reference), in_P_sens(= 같음), in_T_main(= True), fold_T1. 행 순서 = pair_id 순.
3. k = 0 … 4:
   - test = fold_T1 == k인 짝, train = fold_T1 ≠ k인 짝(−1 포함). 둘 다 pair_id 오름차순.
   - (inner_train, inner_val) = `sklearn.model_selection.GroupKFold(4).split(train, groups=core)`의 **첫** 분할(`d1_train.inner_folds`와 같은 규칙), train 순서 유지.
   - `results/splits/T1__f{k}.json` = {"split": "T1", "fold": k, "train", "test", "inner_train", "inner_val"} (pair_id 목록).
4. `results/S_splits_r.json` = {pairs: F_pairs_r.csv 경로·sha256, n_pairs, sklearn 판, units: {"T1__f{k}.json": {n_train, n_test, n_inner_train, n_inner_val, sha256}}}.

기대 test 크기(로컬 추정): f0 1,686 · f1 1,706 · f2 1,414 · f3 1,344 · f4 2,035 · 늘 학습 170.

| # | 검사 | 실패 시 |
|---|---|---|
| S-1 | 모든 짝의 core가 표에 있다 | STOP (새 core) |
| S-2 | fold마다 test 짝·core·반응이 train에 없다 | STOP |
| S-3 | inner_train ∪ inner_val = train, 겹침 0, inner_val의 core는 inner_train에 없다 | STOP |
| S-4 | fold −1이 아닌 짝은 test 목록 정확히 하나에, −1 짝은 test에 없다. 5개 train의 합집합 = 모든 짝 = `in_T_main` | STOP |

---

## 6. Phase D — 데이터 확인 (`analysis/gnn3r/check_data.py`, env_d1fu2, CPU)

`gr_common.dataset()`이 아래 인자로 `gnn3.data.PairDataset`을 만든다. K·W·D가 모두 이 함수를 쓴다.

```
PairDataset(token_dir=TOKENS, report_dir=TOKEN_REPORTS, global_table=X182,
            pair_labels=RES/"F_pairs_r.csv", split_dir=RES/"splits", split_report=RES/"S_splits_r.json",
            pair_attributes=RES/"S_folds_r.csv", label_report=RES/"B_label_report.json")
```

| # | 검사 | 실패 시 |
|---|---|---|
| D-1 | `usable_reaction` True 8,617, rxn_id 집합 = `USABLE`의 사용 가능 집합 | STOP |
| D-2 | `usable_pair` 전부 True, `pair_list` = 모든 pair_id | STOP |
| D-3 | k = 0 … 4 `fold_split(k)`: 모든 목록의 `dropped`가 0, 크기 = `S_splits_r.json` | STOP |
| D-4 | `all_training_pairs()` 성공, 크기 = 짝 수 | STOP |

`results/D_check.json`에 `dataset.sha256`과 검사 결과를 적는다.

---

## 7. Phase K — 교차검증 학습 (`analysis/gnn3r/train_cv_r.py`, env_d1fu2, RTX 3090)

- 내용: `gnn3.training.run_cross_validation_unit(gr_common.dataset(), fold, $GR/cv, log_every=50)`. 격자 4칸을 안쪽 분할로 고르고, 고른 칸으로 seed 5개를 학습해 test 짝을 예측한다. 이미 끝난 fold는 기존 json을 돌려준다(재제출 안전).
- `--fold-from-array`면 fold = `$SLURM_ARRAY_TASK_ID`.
- 제출(저장소 루트에서, 전에 `squeue -u $USER -h | wc -l` + 5 ≤ 20 확인):

```
sbatch --array=0-4%5 -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3r_cv \
  --output=$GR/logs/%x_%A_%a.out --export=ALL,GNN3_SCRATCH=$GR \
  gnn3/jobs/run.slurm analysis/gnn3r/train_cv_r.py --fold-from-array --out $GR/cv
```
- 다섯 fold 모두 같은 GPU 종류(RTX 3090)에서 돈다. gpu1이 차 있으면 기다린다.

| # | 검사 (`analysis/gnn3r/check_cv.py`, CPU) | 실패 시 |
|---|---|---|
| K-1 | `$GR/cv/gnn3_fold{k}.json`·`.parquet` 5쌍, `__seed{j}.pt` 25개 | 빠진 fold만 다시 제출. 두 번째도 실패하면 STOP |
| K-2 | json의 checks(반대칭 등) 전부 통과 | STOP |
| K-3 | fold k parquet의 pair_id 집합 = `T1__f{k}.json` test, 4 target, 값 유한 | STOP |

json·parquet는 `results/cv/`로 복사해 commit한다(.pt는 `$GR`에만).

---

## 8. Phase E — 평가 (`analysis/gnn3r/evaluate_r.py`, env_d1fu2, CPU)

`gnn3/scripts/evaluate.py`의 `main`과 같은 표를 만든다. 다른 점은 입력 경로 두 개뿐이다: `truth, attributes = metrics.channel_truth(RES/"F_pairs_r.csv")`, `sets = metrics.pair_sets(RES/"S_folds_r.csv")`.
- 출력 `results/eval/`: metrics.csv, rank.csv, cause.csv, identity.csv, breakdown.csv, applicability.json.
- 검사 E-1: 평가한 짝 수 = 짝 수 − fold −1 짝 수. 다르면 STOP.

---

## 9. Phase W — 최종 모델 (`analysis/gnn3r/train_final_r.py`, env_d1fu2, RTX 3090)

- 먼저 K-1 … K-3을 다시 확인한다(실패 시 STOP).
- `setting = gnn3.training.choose_final_setting([gnn3_fold0.json … gnn3_fold4.json])` (5 fold 중 가장 많이 고른 칸, 동률은 평균 검증값 → 격자 순서, epoch = 5개 best epoch의 중앙값).
- `gnn3.training.train_final_model(gr_common.dataset(), setting, $GR/final)` → `gnn3_final__seed{j}.pt` 5개, `final_report.json`.
- 제출: K 배열이 끝난 뒤 `--dependency=afterok:<배열 job id>`, `-p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3r_final --output=$GR/logs/%x_%j.out`.
- 검사 W-1: `n_train_pairs` = 짝 수(−1 포함 전부). 다르면 STOP. `final_report.json`과 setting을 `results/final/`로 복사한다.

---

## 10. Phase R — 보고 · manifest · commit · push

1. `results/RUN_REPORT_GNN3R.md` (한국어, 표):
   - 깔때기: 0-3(a)의 기대 수와 실제 수를 나란히. 13개(또는 실제 수) 짝 없는 반응과 status.
   - 데이터셋: 반응·짝·계열 수, ref_kind·dipole·pos·fold별 표, 참조 목록과의 차이 요약(`B_diff_vs_est.csv` 이유별 수).
   - P-4: 검사한 짝, mismatch로 버린 짝, 그 밖 상태별 수.
   - 학습: fold별 고른 칸·epoch·검증 MAE, 최종 setting, GPU·시간.
   - 성능: `results/eval/metrics.csv`에서 P_main·test_all의 4 target MAE·r.
2. `results/gnn3r.json` (데이터셋 manifest, `g2r.json`과 같은 형태): header(spec, branch, results commit, 작성 시각), roots(`$GR`, `$G2R`, cv, final), files(경로 + sha256: F_pairs_r.csv, S_folds_r.csv, S_splits_r.json, splits 5개, B_label_report.json, rowattr_r.pkl, U_usable.csv, X182, T_manifest, labels_d1_cr2.json), counts, rules(0-2절 요약), model(setting, .pt 경로).
3. commit: `analysis/gnn3r/` 코드·job·results 전부. `$GR`·`$G2R`·`$LAD`의 파일과 `.pt`는 commit하지 않는다. `git push origin gnn3r`. push한 HEAD를 `RUN_REPORT_GNN3R.md`에 적어 한 번 더 commit·push.
4. **멈춘다.**

---

## 11. 자원 · 제출 요약

| 단계 | 어디서 | 자원 | 예상 |
|---|---|---|---|
| A | `analysis/d1_pair/jobs/py.slurm`, `PYENV=reactot` | cpu1·cpu2 중 빈 곳, 4코어, 16 GB | 1시간 안 |
| B | 같음 | 4코어, 16 GB | P-4가 TS 약 8,600개를 읽음, 1–3시간 |
| S | `py.slurm`, `PYENV=espley` | 1코어 | 몇 분 |
| D, E, 검사 | `gnn3/jobs/run.slurm` | cpu1·cpu2, 4코어, 32 GB | 몇 분 |
| K | `gnn3/jobs/run.slurm` 배열 0–4 | gpu1 RTX 3090 × 5 | fold당 약 1–1.5 GPU-시간 (예전 7,388짝에 5 fold 합 5.0 GPU-시간) |
| W | 같음 | RTX 3090 × 1 | 약 1–2 GPU-시간 |

- `py.slurm`은 `analysis/d1_pair/`에서 제출하는 스크립트다(`CODE = $SLURM_SUBMIT_DIR`). 새 코드는 `--export=ALL,D1PAIR_CODE=$WT/analysis/d1_pair,PYENV=...`로 넘기고 `$WT/analysis/gnn3r/...` 절대 경로로 부른다.
- A → B → S → D는 `--dependency=afterok`로 이어 제출해도 된다(한 번에 4칸).

---

## 12. 규칙

- 모든 계산은 sbatch, `--time=48:00:00`. 로그인 노드에 상주 프로세스 금지.
- 모든 출력은 임시 파일 → `os.replace`. 출력이 있으면 그 단계를 건너뛴다(B-R의 두 번째 실행은 `$GR/B/rerun/`에 쓴다).
- 기존 코드(`analysis/d1_pair/*`, `gnn3/*`, `analysis/g2r/*`) 수정 금지. 복사한 함수는 `analysis/gnn3r/rowattr_r.py`에만 두고 원본 sha256을 적는다.
- `gnn3/gnn3/constants.py`의 값(격자, seed, epoch, patience, 학습률 등)을 바꾸지 않는다. 격자 밖 튜닝 금지.
- K·W는 RTX 3090에서만. 다른 GPU로 돌린 결과는 쓰지 않는다.
- `$GR`·`$G2R`·`$LAD`·`$D1PAIR_SCRATCH`의 파일은 commit하지 않는다. `results/`에는 표·json·csv·parquet만.
- 제출 전 `squeue -u $USER -h | wc -l`로 칸 확인(배열 원소 하나 = 한 칸, 최대 20).
- 13절 밖의 이상은 기록하고 진행한다. 버그는 고치기 전에 보고한다.

---

## 13. STOP 조건

| 단계 | 조건 |
|---|---|
| 1 | 시작 조건 1-1 … 1-11 중 하나라도 아님 |
| A | A-1 … A-6 |
| B | B-0 … B-7, B-R. 특히 B-6(반응 8,432–8,776, 짝 8,188–8,522 밖) |
| S | S-1 … S-4 |
| D | D-1 … D-4 |
| K | K-2, K-3, 같은 fold 두 번 실패(K-1) |
| E | E-1 |
| W | K 재확인 실패, W-1 |
| 공통 | 기존 코드나 GNN3 상수를 바꿔야만 진행할 수 있음 |
