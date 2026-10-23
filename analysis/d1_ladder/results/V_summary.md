# D1LADDER Phase V — 기준 구조 확인 계산 요약

- 코드 d1-ladder-v1, git 825c8017a2, 작성 2026-10-23 15:48:36
- 판정: **통과** — 정답 파일 `labels_d1_cr2.json`, `F_pairs_cr2.csv`

## V-1 대상
- 분자 키 964개 중 대상 152개: (a) dev > 0.5 kcal/mol 95개, (b) cyclooctyne 계열(D, Dp) 82개 (겹침 45개), (c) 대조군 20개
- 제외: constrained dipole 키 445개 (그중 dev > 0.5인 것 86개)

## V-5 시험 묶음과 비용
- 시험 10개 분자 코어시간 34.9 (할당 기준), 전체 예상 686 코어시간 (한도 5,000) → 자동 진행
- 시험 DFT 최적화 실패율 0.0 %
- 시험 기준 구조 재현 (라벨·HF 단일점 그대로 다시 계산): 최대 |ΔE| 0.0000 / 0.0000 kcal/mol

## 더 낮은 형태를 찾은 분자 (라벨 수준 단일점, E_min − E_V > 0.05 kcal/mol)
| 대상 | 분자 수 | V 형태 있음 | 더 낮음 | > 0.5 | > 1 | > 2 | 낮아진 양 중앙값 | 최대 |
|---|---|---|---|---|---|---|---|---|
| (a) dev > 0.5 | 95 | 80 | 12 | 2 | 1 | 0 | 0.202 | 1.173 |
| (b) cyclooctyne 계열 | 82 | 82 | 18 | 4 | 3 | 1 | 0.177 | 2.277 |
| (c) 대조군 | 20 | 12 | 0 | 0 | 0 | 0 | - | - |

- 대조군에서 더 낮은 형태를 찾은 비율: 0/20 = 0.0 %

## cyclooctyne 부모 분자의 형태별 에너지 (ΔE = E − Q의 E_min, kcal/mol)
| 분자 | 출처 | id | D1 반응 수 | ΔE 라벨 | ΔE HF | 상태 |
|---|---|---|---|---|---|---|
| cyclooctyne | D1 form | f0 T05096/frag2 | 204 | 0.000 | 0.000 | kept (D1 value) |
| cyclooctyne | D1 form | f1 T03871/frag2 | 99 | 2.897 | 3.307 | kept (D1 value) |
| cyclooctyne | D1 form | f2 T00392/frag2 | 5 | 67.402 | 75.126 | kept (D1 value) |
| cyclooctyne | V etkdg | c0 = f0 |  | 0.021 | 0.008 | ok |
| 3,3-dimethylcyclooctyne | D1 form | f0 T06285/frag2 | 494 | 0.000 | 0.000 | kept (D1 value) |
| 3,3-dimethylcyclooctyne | D1 form | f1 T01545/frag2 | 48 | 2.753 | 3.055 | kept (D1 value) |
| 3,3-dimethylcyclooctyne | D1 form | f2 T10920/frag2 | 105 | 2.807 | 3.269 | kept (D1 value) |
| 3,3-dimethylcyclooctyne | D1 form | f3 T00458/frag2 | 1 | 65.778 | 74.485 | kept (D1 value) |
| 3,3-dimethylcyclooctyne | D1 form | f4 T03397/frag2 | 1 | 68.397 | 76.008 | kept (D1 value) |
| 3,3-dimethylcyclooctyne | V etkdg | c0 = f0 |  | 0.017 | -0.042 | ok |
| 3,3-dimethylcyclooctyne | V etkdg | c1 = f0 |  | 0.065 | 0.020 | ok |

## DFT (V-3)
- 최적화 시작 505개, 실패 16개 (3.2 %; 명령서 V-3대로 재최적화 뒤 허수 진동수가 하나라도 남으면 실패), 참고: D1 TS 단계 허용치(−50 cm⁻¹)를 썼다면 0.0 % (그중 허용치 안 16개)
- 상태별: fail:imag 16, ok 489

## 재조립 (V-4)
- 기준이 바뀐 분자 키 20개, V 기준을 쓰게 된 조각 123개, cr 대비 바뀐 반응 123개, 원래 라벨 대비 바뀐 반응 3131개
- gate: Q-a 통과, Q-b 통과, Q-c 통과, Q-d 통과, Q-e 통과, V-a 통과, V-c 통과, V-e 통과
- dd_d1: cr 대비 바뀐 짝 0개 (P_main 0개), 바뀐 양 중앙값 - kcal/mol
- dd_d2: cr 대비 바뀐 짝 117개 (P_main 105개), 바뀐 양 중앙값 0.227 kcal/mol
- dd_barrier: cr 대비 바뀐 짝 117개 (P_main 105개), 바뀐 양 중앙값 0.227 kcal/mol

## 코어시간
- 할당 기준 835.3, 실제 사용 796.4 코어시간 (그룹별 할당: a 439.7, b 711.5, c 38.2)

## 결정 (사용자 지시: 모든 갈림길은 추천안)
- V1_env: targets/chain/decide/collect reactot; check/work/conf/dft = D1 run env (autodE 1.4.5, RDKit 2024.3.2, xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f)
- V2_targets: (a) dev > 0.5; (b) dipolarophile keys with a D / Dp row; constrained excluded; (c) one default_rng(20261023): 10 dipolarophiles then 10 dipoles from the sorted rest
- V3_trial: cyclooctyne + 3,3-dimethylcyclooctyne + 8 of sorted (a) with a fresh default_rng(20261023)
- V4_anchor: lowest label reference with D1-HF orig + hf inputs; all structures in its atom order; its inputs = templates
- V5_forms: D1 references deduped (dE < 0.05 or RMSD < 0.1) = forms, kept with D1 values, xTB-relaxed into the dedupe pool first; DFT on the 5 lowest new search conformers
- V6_etkdg: ETKDGv3 defaults, randomSeed 20261023, 300, no pre-pruning, with H
- V7_rr: autodE get_simanl_conformer x 300 round robin over the forms + autodE prunes; seeded RandomState(20261023+n)
- V8_xtb: --opt tight --alpb water --gfn 2 --chrg q --uhf 0; graph iso to anchor; stereo3d_sig = the key's signature (unique D1 signature, else the Q E_min source's); not checkable -> molecule failed (no V replacement)
- V9_dft_text: autodE ORCA text reproduced byte for byte (checked on D1 run inputs); ORCA run directly
- V10_imag: any freq < 0 -> one 1.0 A displacement + Opt/Freq; any freq < 0 after that = failure 'imag' (spec V-3); the -50 cm-1 (config extra_imag_tol_cm) count is a reported sensitivity only
- V11_sp: label SP = template with coordinates replaced (checked), no retry; HF route A == route B, D1-HF retry rule; valid conformer = both SPs ok; every ORCA output = 6.1.1 (487d211c) or STOP
- V12_repro: trial anchors rerun unchanged: |dE| <= 0.05 kcal/mol vs label and labels_d1_hf, else STOP
- V13_v4: replace iff E_min(Q) - E_V > 0.05; Q-4/Q-5 via refs_q with tag cr2; gates V-a, V-c, V-e added
- V14_hf_refs: HF of final references: own / Q source / V SP -> $LAD/refs_cr2.parquet
- V15_cost: estimate = max(mean x N, n_atoms^3 scaling) of allocated core-h; <= 5000 -> go
- V16_layout: trial 2 tasks %2; main K tasks (K = --max-run, default 4), throttle min(K, 4 - other LADDER tasks); 8 CPUs 48G; chain launcher resubmits; round limit (main from the estimate) -> STOP when hit
- V17_name: refs_v.py (task) instead of refs.py (spec 8)
- V18_lock: per-molecule O_EXCL lock with the SLURM job id; held by a live job -> skipped; dead owner -> taken over
- V19_shared_writes: check.json / STOP written with a per-process temporary name (parallel array tasks)
- V20_mpirun: check requires mpirun on PATH under $MPI_ROOT

## 출력 파일 (sha256)
- `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/labels_d1_cr2.json` 7e5882bce4d47dcd
- `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder/results/labels_d1_cr2.parquet` 696d2ec6ee1b75b5
- `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` f2fe5cab1c61f082
- `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/refs_cr2.parquet` 4890ec9b54967f25
- `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder/results/V_molecules.csv` 90fc8ae1c3560c90
- `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder/results/V_conformers.csv` 4644403a8283a695
- `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_ladder/results/V_cyclooctyne_conformers.csv` 4df128e661646e89
