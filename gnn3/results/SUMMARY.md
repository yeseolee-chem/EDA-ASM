# GNN3_CLEAN 요약

GNN3(CH3M lf3 GNN, 3채널 짝 모델)를 만든 코드를 `gnn3/` 한 폴더로 정리했다. 숫자는 하나도 바뀌지 않았다
(아래 동일성 표). 기존 `analysis/` 코드는 고치지 않았다.

## 새 구조

```
gnn3/
  README.md (영어) · README_ko.md · MODEL_CARD.md · CODE_MAP.md
  gnn3/      constants → channels → data → model → training → predict / metrics (순환 없음), inputs/pipeline (입력 단계 wrapper)
  scripts/   prepare_inputs · train_cv · train_final · predict · evaluate
  jobs/      run.slurm (env_d1fu2) · prepare_inputs.sh (입력 사슬 sbatch 2개)
  tests/     check_start.py (S0) · test_units.py (E-6, 11개) · equivalence.py (E-1 … E-5, E-2b)
  results/   S0_start · E_equivalence · E_parts/ · eval/ (공개 예측 평가표 + 위치·골격별 MAE·R²) · DECISIONS · R_reader_test · SUMMARY
```

## 동일성 검사

전부 통과(`results/E_equivalence.json` all_pass). 어떤 숫자도 바뀌지 않았다.

| 검사 | 비교 | 결과 |
|---|---|---|
| E-1 데이터 | 토큰 · 전역 182 · 라벨 · usable 표시 · 짝 행 · fold 목록(5 + all) · 표준화 통계/텐서 vs 원래 `Ch3Data` | sha256 모두 같음 |
| E-2 옛 `.pt` 25개 | 새 모델에 읽어 fold별 시험 짝 예측 vs `KM_predictions` | 최대 차이 0.0 (비트 동일) |
| E-2b 예측 명령(추가) | `scripts/predict.py`, fold 0 시험 짝 1,590개 | 1.8e-15 (CSV 반올림) |
| E-3 재학습 | RTX 3090, fold 0 처음부터(튜닝 4칸 + seed 5개) | 같은 칸 · best epoch(49/69/37/86, 선택 4×64·49), 검증 차이 0.0, 시험 ŷ 차이 0.0 (비트 동일), 3,208 s (원래 3,210 s) |
| E-4 지표 | EM_metrics 60행 · EM_rank 2 · EA_cause 8 · EM_identity · EM_ad | 소수 4자리 같음 |
| E-5 입력 | D1 10반응(쌍극자당 2, seed 20261111), g2 폴더부터 → 토큰 · X182 vs `$CH3/lf` | finalize 재사용 갈래 0.0; ORCA 재계산 갈래는 Hessian 특성 3개만 ≤ 4.8e-8, 나머지 0.0 (D-6 사전 규칙) |
| E-6 단위 시험 | CPU 11개 | 모두 통과 |

## 이름을 바꾼 주요 항목 20개

| 원래 | 새 이름 |
|---|---|
| `models.GNN` (embed, layers, pool, gproj, head) | `model.GNNPairModel` (atom_embedding, message_layers, attention_pool, global_projection, pair_head) |
| `GNN.readout` | `GNNPairModel.encode_reactions` |
| `models.MPLayer` (etype, edge, w_h, w_e, msg_out, upd, norm) | `model.MessagePassingLayer` (edge_type_embedding, edge_distance_mlp, node_to_message, edge_to_message, message_output, node_update, layer_norm) |
| `models.AttnPool` (`w`) | `model.AttentionPool` (`score`) |
| `rbf` / `pdist` / `mlp` | `radial_basis` / `pairwise_distances` / `make_mlp` |
| `fu2 train.batch` | `model.make_reaction_batch` |
| `fu2 train.predict` | `model.predict_standardized` |
| `fu2 train.fit` | `training.train_one_fit` |
| `loss_fn` | `training.identity_penalty_loss` |
| `epoch_batches` / `orient` | `iterate_training_batches` / `swap_pair_order` |
| `seed_all` | `seed_everything` |
| `gate_m1` / `leak_check` | `check_antisymmetry` / `check_no_leak` |
| `Std` (mu, sd, gmu, gsd, sy) | `data.Standardizer` (numeric_mean, numeric_std, global_mean, global_std, target_scale) |
| `DeployData` + `Ch3Data` (ra, rb, y, y10, usable_rxn) | `data.PairDataset` (rows_a, rows_b, targets_kcal, labels_kcal, usable_reaction) |
| `DeployData.split(k)` / `split("all")` | `PairDataset.fold_split(k)` / `all_training_pairs()` |
| `train_m.run_unit`, unit `D__lf3__S__GNN__f{k}`, `__s{j}.pt` | `training.run_cross_validation_unit`, `gnn3_fold{k}`, `__seed{j}.pt` |
| `choose_final_grid` / `train_final` | `choose_final_setting` / `train_final_model` |
| `LAMBDA`, `I_BAR` / `I_CH9` / `I_PRIM` | `IDENTITY_PENALTY_WEIGHT`, `BARRIER_INDEX` / `CHANNEL_INDICES` / `VALIDATION_INDICES` |
| `eval_ch3.metrics` / `rank_table` / `cause` / `cause_rows` | `metrics.error_table` / `ranking_table` / `cause_agreement` / `cause_table` |
| `eval_geo2.boot_diff` / `eval_m.ad` | `metrics.bootstrap_mae_difference` / `applicability_thresholds` |

옛 `.pt` 25개는 `model.load_weights`의 이름표(키 지도)로 그대로 읽힌다(E-2).

## 옮기지 않은 것

- CH3M 관리 코드: PREREG · C_tests 게이트, 쓰기 보호, code sha256, smoke · 회귀(C-RT/C-RG) · collect · Phase W 명령.
  숫자를 지키는 검사(누출 · 반대칭 · 설정 지문 · 입력 sha256)는 옮겼다.
- 10채널 ↔ 3채널 전환(`use_targets`, `current_mode`, `MODE10`)과 입력 경로 고정(`_bind_env`: DEP_SCRATCH/DEP_RESULTS를
  $CH3/lf에 묶음): GNN3는 늘 3채널이라 채널 설정은 상수로 고정했고, 입력 경로는 `gnn3/constants.py`에 둔다.
- 다른 모델(TF, AW, XGB, 절대값), TF/graph read-out, TF10 최종 모델 자체검사(I-1/I-2).
- 다른 모델과의 비교 함수(boot_rel, compare, q1, q2 …).
- 입력 단계 코드(finalize_lf · meta_lf · feat_lf · tokens_lf · fu2 atoms/tokens · espley xTB 코드 · D1 pilot):
  지시서대로 복사하지 않고 그대로 호출한다.

## 발견한 문제 (기록만, 고치지 않음; 자세히 CODE_MAP.md 5절)

1. 입력 사슬의 TS 탐색 뒤 단계(finalize run · report, meta_lf)가 D1-G2 대상 9,266개(G_status.csv; run · meta_lf는 D1
   manifest 행도)만 받는다 → D1 밖 새 SMILES는 finalize 첫 명령에서 멈춰 기존 코드로 입력을 끝까지 만들 수 없다(P-1). 배포 경로 g2_deploy는 node-local 폴더를 지워
   meta_lf가 STOP한다(P-2); 유일한 새 반응 사슬 i3_chain은 지워진 `$DEP/i3*` root만 받는다(P-3).
2. g2.slurm이 `D1G2_SCRATCH`를 운영 폴더로 고정한다(P-4).
3. ORCA 진동수 재계산은 비트 단위로 재현되지 않는다: `nu_imag` ~1e-7(E-5 rerun 4.8e-8), `mode_share` · `mode_async`
   ~1e-11 (P-5).
4. 기존 코드에는 GNN3로 새 반응을 예측하는 명령이 없다(predict_tf는 TF/10 타깃 고정, predict_m에는 run 없음) →
   `scripts/predict.py`로 채움(P-6).
5. final_m이 게이트 전에 WM_gate.json을 쓴다(P-7); 최종 모델 산출물 이름에 후보 이름이 없어 겹친다(P-8);
   predict_m import가 다른 DEP 환경에서 STOP(P-9).
6. 작은 것: choose_final_grid 홀수 검사 없음(P-10), require_gpu 안내가 a6000(P-11), 같은 이름 ANTISYM_TOL 두 값(P-12),
   final fit의 cosine T_max 300(P-13), rank_families 중복 Xb 처리(P-14).
7. 지시서의 "cyclooctyne D R² 0.13"은 커밋된 출처가 없었다. 다시 계산하니 전체 시험 짝 장벽 R² 0.129(n 541),
   P_main 0.027(n 458)이다(P-15). MODEL_CARD_ch3m.json은 GNN3가 아니라 TF10 카드다(P-16).
8. g2 실패율은 정확히 7.0 %(650 / 9,266), 토큰까지 잃는 비율은 8.25 %.

## 독자 시험

`gnn3/`만 읽을 수 있는 새 에이전트로 2번(`results/R_reader_test.json`).

- 1회: 9/10. 9번(새 SMILES 짝 예측 명령)이 "문서에 없음".
  - 고친 곳: README §7을 명령 순서로 다시 썼다(manifest 열, LIST/PAIRS 파일, ts-search, 입력 사슬, predict, job_id↔rxn_id, 어디서 멈추는지, 어느 fold 네트워크를 쓸지).
  - 혼동 9건도 고쳤다: DFT 수준(SVP 구조 / TZVP 에너지), c_ghost 설명 통일, anchor 짝 쓰임, core 46 vs 44, 위생 조건, 원자 특성 단위, cr2 수준, T_max, MODEL_CARD 문구.
- 2회(새 에이전트): **10/10**, "문서에 없음" 없음.
  - 남은 작은 혼동 6건도 고쳤다: 라벨 합 오차, P_sens/T_main 정의, 결합 종류 3개, 문턱 보정 방식, 스크리닝 범위, 데이터가 폴더 밖에 있는 이유.

## 문서 검증 (push 전)

workflow 하나로 문서의 모든 주장을 코드 · 결과 파일과 대조했다. 점검 6갈래(MODEL_CARD 숫자, README vs 코드, §7 명령,
한/영 일치, CODE_MAP 줄 번호 · 문제 목록, SUMMARY · 결과 일치) 뒤에 갈래마다 반박 검증 에이전트를 붙였다.
- 확인된 23건은 모두 고쳤다. 숫자 결과는 바뀌지 않았고, 문서 · docstring 문장만 고쳤다. 주요 수정:
  - fold에 들어간 core는 44개이고, 언제나 학습에 있는 core는 6개다.
  - g2를 다시 돌렸을 때 같은 TS는 10개 중 2개였다("약 절반"이 아님).
  - 토큰을 얻지 못한 비율은 8.2 %다.
  - meta 효과의 SD 설명, ortho MAE 범위, 제외 파일 이름(`predictions.excluded.csv`).
  - §7: 경로와 ROOT 만들기, D1-G2 대상의 정의, 멈추는 지점(finalize run), core 기준 fold 고르기, E-2b가 GPU 결과라는 점.
  - P-1 · P-5 · P-9 · P-12의 세부 사항, D-6 수치(측정 7.4e-7, 상한 8.6e-7).
- 반박되어 고치지 않은 것 2건: 한국어판 표현 생략으로, 뜻은 바뀌지 않았다.

## 사용 자원

| 항목 | 사용 |
|---|---|
| GPU (RTX 3090, gpu1) | 약 0.91 GPU-h: E-3 0.89, E-2 0.01, E-2b 0.01, S0 시작 검사 4 s |
| CPU | 0.5 core-h 미만: E-1 · E-4 · E-6, E-5 입력 사슬 두 갈래(ORCA 진동수 10반응 21 s, 특성·토큰 각 약 35 s × 8 core), 평가 |

## HEAD

push 시점 HEAD: `05ec49c339e7f17bfbc56e9ee3d8af26f4145fe9` (origin/gnn3-clean). 이 줄을 더한 commit이 그 위에 하나 더 있다.
