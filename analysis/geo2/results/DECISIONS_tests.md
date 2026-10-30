# GEO2 결정 기록 — role LADB (8-2 검사 `C_tests_geo.json` + 9-2 `evaluate.py e3`)

명령서 `docs/specs/GEO2.md` 8-2, 9-2, 10. 이 역할이 쓴 파일: `analysis/d1_ladder/tests/run_tests.py`(mode `geo`),
`analysis/d1_ladder/evaluate.py`(명령 `e3`, `e3-selftest`), `analysis/geo2/RUNBOOK_tests.md`, 이 파일.
아무것도 실행하지 않았다(로그인 노드). 줄 번호는 2026-10-30 작성 시점(A–E절은 첫 작성, F절은 리뷰 수정 뒤의 줄 번호;
F절 수정으로 A–E절의 뒤쪽 줄 번호는 수십 줄 밀렸다). run_tests.py 결정 G-1 .. G-17(+ G-12b)은 그 파일 docstring과
`GEO_DECISIONS`에, evaluate.py 결정 E3-1 .. E3-8(+ E3-2b)은 `DECISIONS3`에도 있다.

## A. run_tests.py — 기존 모드와 공용 함수 (G-1)
- **T-1 기존 모드 그대로** `run_tests.py:3053-3100` `main`: `gpu`·`cpu`·`merge`·`l4child`(dft·g1·g2)는 분기·출력이 전과 같다.
  새 분기 `geo`만 더했다. 기존 `CODE_FILES`, `TOL`, `DECISIONS`, `CHECKS`, `PART_FILE`, `OUT`은 바꾸지 않았다(기존
  `C_tests.json` 내용 형식이 바뀌지 않게). 단, 이 파일 자체가 바뀌었으므로 LADDER 모드를 다시 돌리면 `_current`가
  "code changed"로 `C_tests.json`을 all_pass false로 다시 쓴다 → RUNBOOK 0절: GEO2에서는 LADDER 모드를 돌리지 않는다.
  (리뷰 수정 T-34 / G-17: 이제 `gpu`·`cpu`·`merge`는 코드가 `C_tests.json` code_sha256과 다르면 쓰기 전에 exit 3으로 거부한다.)
- **T-2 공용 함수의 rung 일반화(기존 rung 값은 원래 상수)**:
  `table_path` `:480-482`(geo → `$GEO2/inputs/geo.parquet`), `_ri` `:485-487`(= `RUNG_ORDER.index`, geo = 3; 쓰는 곳
  `:1138, 1154, 1158, 1429, 1477`), `per_rung` `:656-657`(`ctx.rungs`가 있으면 그것, 없으면 `RUNG_ORDER`),
  `run_spies` `:1590`(같은 규칙), `H_PARQUET`에 geo 키 + `H_GEOM` `:280-282`, `_l5_provenance` geom `:1050`
  (`H_GEOM.get(r, r)` = 기존 rung은 r 그대로), `l6_xgb(ctx, stage=1)` `:1286-1293`(기본 1 = 기존 호출 그대로),
  `_l9_cross(pe, rungs=RUNG_ORDER)` `:2303`. 모두 기존 rung에서 같은 값·같은 호출이다.
- **T-3 GEO2 상수** `:271-282`: `GEO2` = `$GEO2_SCRATCH`(기본 `/gpfs/tmp_cpu2/yeseo1ee/geo2`, resolve 안 함 = train_atoms·
  geo2_common·feat_geo와 같은 문자열), `GEO2_RES` = `L.REPO/analysis/geo2/results`, `GEO_TABLE`, `P_BUILD_GEO`,
  `P_GEO_REPORT`, `P_GEO_SETS`, `SCR_GEO = $LAD/tests_geo`, `ALL_RUNGS`.

## B. run_tests.py — geo 모드 (명령서 8-2)
- **T-4 출력 위치 (G-2)** `write_json_geo` `:2475-2490`: `C_tests_geo*.json`은 `analysis/geo2/results` 아래에만 쓴다
  (`lad_common.assert_path`는 그 경로를 허용하지 않고, LADA D-22/Q-2대로 lad_common을 바꾸지 않았다). part 파일
  `C_tests_geo_{gpu,cpu}.json`, 병합 `C_tests_geo.json`(인터페이스 이름 그대로). 임시 작업 폴더는 `$LAD/tests_geo`:
  생산 코드 `run_unit`이 `$LAD` 아래에만 쓸 수 있기 때문(GA-11). 명령서 13절 쓰기 목록에 `$LAD/tests_geo`는 이름이
  없어서, LADA D-4(`$LAD/smoke_geo`)처럼 `_geo` 이름 규칙 안으로 보았다. 그 밖에 쓰는 것은 학습 코드 자신의 stage-3
  출력(`results/C_smoke_atoms_geo.json`, `C_smoke_xgb_geo.json`, `C_l6_xgb_geo.json`; 이름에 geo → PREREG 스냅숏 제외)뿐.
- **T-5 `--regress-only` (과제 지시)** `:2425-2441`, CLI `:3067-3083`: 회귀 1–3만 도는 모드. 출력
  `C_tests_geo_regress{,_gpu,_cpu}.json`(`training_gate: false`). 학습 gate가 읽는 `C_tests_geo.json`은 쓰지 않는다.
  전체 모드는 회귀 1–3을 다시 돈다(같은 코드로 통과했다는 보장; 다시 쓰는 비용은 XGB 2 unit 정도).
- **T-6 두 part (LADDER C-1 그대로)** `run_geo_gpu` `:2873`, `run_geo_cpu` `:2911`: torch와 xgboost가 한 env에 없어
  gpu(env_d1fu2)·cpu(env_espley)로 나누고 `merge_geo` `:2984`가 검사별로 합친다. 검사 → part 표: docstring `:144-163`.
  L-7은 gpu만, R-3은 cpu만(`GEO_PARTS_OF` `:2430-2432`).
- **T-7 geo 준비 판정 (G-3)** `resolve_geo` `:2506-2535`: 표 + `P_build_geo.json`(all_pass, stop 아님, outputs sha256 =
  표)일 때 ready. 아니면 geo 검사는 모두 waiting(exit 2)이며 fail이 아니다. `P_geo_report.json`/`P_geo_sets.json`은 기록만
  (이 검사들은 짝 집합을 쓰지 않고, 학습 gate `gate_p_geo`가 본다).
- **T-8 geo 경로 인터페이스 확인** `_geo_interface` `:2493-2503`: 이 파일의 geo HF 원본(`analysis/geo2/results/hf_geo/
  labels_d1_hf_g2.parquet`, geom `g2`)이 `inputs.RUNGS["geo"]`(LADA PG-1 = feat_geo `rung_cfg("geo")`)와 다르면 geo rung을
  fail로 둔다(L-5 provenance가 다른 파일을 읽게 되므로). 인터페이스 규칙상 FEAT의 값이 이기므로 어긋나면 이 상수
  `:280-281`을 그 값으로 맞춘다.
- **T-9 inputs.py loader API 부분검사는 geo에서 skipped (G-5)** `_l5_api` `:1027-1031`, L-11 `:2092, 2112`:
  `read_set`/`reaction_frame`/`LadderData`/`pairs_frame`은 `$LAD/inputs/<rung>.parquet`와 `results/P_report.json`을 읽는데
  둘 다 geo rung이 없다(geo 표는 feat_geo `p`의 파일). 학습이 실제로 쓰는 loader(`train_atoms.LadData`, `train_xgb.XData`)를
  검사한다(LADDER C-4와 같은 원칙). `combine()`은 skipped를 판정에서 뺀다.
- **T-10 L-4 geo (G-6)** `_classify` `:948-962`: LADDER C-7의 child process 방식 그대로(rung geo). 판정 = 기존 조건(foreign·
  unknown 0, 표가 열린 목록에 있음, geo 열 경로 ≥ 1) + 명령서 문구대로 dft·g1(‘dft+g1’ 포함) 경로 0, g2 경로 0, G1 폴더
  (`DP.G1_ROOT_D1F/D1/D1R`) 경로 0을 따로 센다. 공통 파일(라벨, fold, FU2 S_splits, V_report/Q_report, S_folds)은
  `inputs.rung_roots`의 shared. child는 rung이 geo이면 `SCR = $LAD/tests_geo` `:3064-3065`(부모의 own-output 판정과 같은 폴더).
- **T-11 L-5 geo (G-7)** `_l5_refs` `:1095-1097`: geo 표의 기준 에너지 규칙 = 키 최저, 단 constrained 키는 자기 기준
  (feat_geo F-7b `cons_own`; `P_build_geo.json`에 `references_g3_cons_own`이 있으면 own 모드). g2(P20 option c)와 같은 검사식.
  "무작위 50반응 H 9값 = geo 입력표(1e-12)"는 `_l5_trainer` 그대로.
- **T-12 seed (G-8)**: 표본 seed = 기존 offset + `_ri(r)`, geo = 3. 기존 rung의 표본은 그대로다.
- **T-13 smoke (G-9)** `smoke_atoms_geo` `:2759`, `smoke_xgb_geo` `:2792`: `train_atoms.cmd_smoke(3)`(AW·GNN·TF × S·SH,
  구성마다 2 epoch, 가장 작은·큰 칸)와 `train_xgb.cmd_smoke(3)`; 판정 = 모든 fit의 gate M-1′ + 구성 이름이 정확히 geo fold 0
  6개 / 2개 + C-25(이 process가 `$LAD/units`에 쓰지 않음). `estimate_geo`(LADA GA-9)는 `C_tests_geo.json`
  `gpu_estimate_geo`로 옮기기만 하고 판정에 넣지 않았다: 80 GPU시간 STOP은 명령서 8-4 학습 단계 조건이고 prereg_geo2.py
  (LADA D-11)가 같은 값으로 판정한다.
- **T-14 회귀 1 (G-10, G-11)** `r1_atoms` `:2562`, `r1_xgb` `:2588`: 새 process에서 `train_atoms.py list --stage 1/2`
  stdout이 `$GEO2/regress/list_stage{1,2}.before.txt`(job 1053812)와 문자열로 같아야 한다. `--stage 3` = geo 30 unit(unit_list
  순서)도 검사한다. XGB는 "before" 파일이 없어서 `train_xgb.py list --stage 1/2/3`을 원래 unit_list 규칙(rung 순서 ×
  S, SH × fold)으로 만든 목록과 비교한다(추가 검사; LADDER unit json이 있는지는 기록만).
- **T-15 회귀 2와 build 지문 (G-12)** `_rel_fp` `:2607`, `_f13_build_check` `:2618-2663`, `r2_atoms` `:2666`, `r2_xgb`
  `:2677`: LADDER `P_build_<rung>.json` 지문은 d1ladder worktree의 절대 경로와 GEO2 이전 inputs.py sha256(07ab5ba7…,
  = `$GEO2/pristine_head`의 inputs.py)를 담고 있어, 이 worktree에서는 `inputs.p_report(rung)`가 모든 build를 stale(exit 2)로
  본다 → L-6·L-10의 inputs API 부분검사가 영원히 waiting이 된다. 명령서 8-1 마지막 줄(D1-G3 F-13: repo 상대 경로로
  비교하고 기록)을 따라, R-2 동안에만 `inputs._build_ok`를 감싸 지문을 repo 상대 경로로 비교하고, 허용하는 차이는
  `analysis/d1_ladder/inputs.py` 하나(기록된 sha256 = pristine 사본일 때만, 즉 LADA의 geo 추가가 유일한 변경)로 했다. 다른
  차이는 R-2 fail(`build_currency.rungs.<rung>.disallowed`). 표 sha256 검사(`_verified`)는 그대로다. 감싸기는
  context manager로 끝나면 원래 함수로 돌린다(inputs.py 파일은 바꾸지 않음).
- **T-16 회귀 2 L-6 XGB (G-13)**: `l6_xgb(ctx, 3)` → `train_xgb.cmd_l6(3)`(LADA XG-G7) → `results/C_l6_xgb_geo.json`.
  LADDER의 `C_l6_xgb.json`은 다시 쓰지 않는다. L-10 cpu는 LADDER처럼 수정된 evaluate.py의 `line_beta`도 검사한다
  (evaluate.py의 `line_beta`/`line_predict`는 한 글자도 바꾸지 않았다: ADD3 PR-13 sha256 그대로).
- **T-17 회귀 3 (G-14)** `r3_xgb` `:2693-2756`: `train_xgb.run_unit(g2, S, XGB, fold 0)` 전체 grid를 smoke 루트
  (`$LAD/tests_geo/r3_xgb`, `$LAD/units` 아님)에서 다시 학습·예측; 라벨 = `results/L_report.json` labels(= F_pairs_cr2.csv
  여야 함), `L_predictions.parquet` sha256 = L_report 기록이어야 함. 판정: 타깃마다 시험 짝 집합 같음, y ≤ 1e-12,
  yhat ≤ 1e-6, base = beta = 0. 고른 칸·val_kcal_mae·지문 키 차이(code_sha256은 구조상 다름)는 기록만.
- **T-18 병합 (G-15, G-16)** `merge_geo` `:2984-3046`: 검사별 가장 나쁜 part 상태, L-9 part 간 sha256, 두 part가 같은
  라벨·표·코드·P_report·P_build_geo(`_consistency_geo`), 기록한 입력이 지금도 그대로(`_current_geo`), 전체 모드는 geo
  ready도 필요. exit 3 = fail(명령서 14 R·M3 STOP), 2 = waiting/pending/error, 0 = 최종 입력에서 모두 통과.
  `code_sha256`에 evaluate.py를 더했다(`GEO_CODE_FILES` `:2439`; R-2 L-10이 그 LINE을 검사하므로).
- **T-19 PREREG 직전 merge 재실행 (RUNBOOK C)**: 학습 gate `gate_c_geo`는 all_pass만 본다. 검사 뒤 코드가 바뀌면
  all_pass가 낡은 채 남을 수 있으므로, prereg write 직전에 `geo merge`만 다시 돌리도록 RUNBOOK에 적었다(merge가
  `_current_geo`로 코드 sha256을 다시 확인).

## C. evaluate.py — `e3` / `e3-selftest` (명령서 9-2)
- **T-20 기존 명령 그대로 (E3-1)** `evaluate.py:3455-3473`: `e1`·`e2`·selftest 분기와 기존 함수는 한 글자도 바꾸지 않았다
  (choices에 e3, e3-selftest 추가 + 분기 하나, docstring에 E3 절). E3는 기존 함수(`fu2_eval`, `label_info`, `add3_check`,
  `val_table`, `pick_rep`, `check_l_report`, `real_inputs`, `load_predictions`, `model_frames`, `check_pair_attrs`,
  `line_frames`, `derived_frames`, `check_line`, `metrics_e1`, `met_index`, `bends_e1`, `goal_e1`, `common_lists`, `_cmp`,
  `ref_set`)를 그대로 부른다 = "E2 계산을 rung geo에 적용". 그림은 만들지 않는다(figG2_*는 11절 Phase S).
- **T-21 쓰기 위치 (과제 지시, E3-1)** `_assert_e3`/`_write_e3` `:2718-2738`: E3 출력은 `analysis/geo2/results`(GEO2 쓰기
  루트)에만 쓴다: `E3_metrics.csv`, `E3_goal.csv`, `E3_bends.csv`, `E3_bend_rates.json`, `E3_compare.csv`,
  `E3_report.json`. `analysis/d1_ladder/results`에는 아무것도 쓰지 않는다(LADA 스냅숏의 `E3_*` 제외 규칙은 필요 없어짐;
  prereg `seen()`의 `e3_files`는 `analysis/d1_ladder/results/E3_*`를 보므로 늘 빈 목록 — 해가 없음).
- **T-22 LADDER rung 읽기 (E3-2)** `ladder_builds` `:2802-2814`, `cmd_e3` `:3347`: T-15와 같은 이유로 `inputs.p_report(rung)`
  대신 `inputs.p_report()`(rung 없이)를 쓰고 Phase P 읽기에는 모두 그 보고를 넘긴다. build는 표 sha256 = P_build outputs =
  P_report table_sha256으로 확인한다(STOP 아니면 진행).
- **T-23 rung·짝 집합 배치 (E3-3)** `E3_LAYOUT` `:2671`, `geo_inputs` `:2961-3032`: dft는 P_023만, g2는 own(P_g2)·P_23·
  P_023, geo는 own(P_geo)·P_23·P_023. dft own은 E2에 이미 있어 다시 계산하지 않았다. P_geo/P_23/P_023은
  `P_geo_sets.json`(sha256 = P_geo_report outputs, 목록별 sha256 = 인터페이스 규칙)에서 읽고, P_g2·P_dft = LADDER 집합
  파일, P_23 = P_g2 ∩ P_geo, P_023 = P_dft ∩ P_23, P_geo = geo 표 usable 양 끝을 다시 셈 — 하나라도 다르면 STOP. geo fold
  목록 = `$LAD/inputs/fu2_splits` 사본(sha256 = P_report) ∩ P_geo, dH = geo 표 H_<t>(h = cr).
- **T-24 geo 예측·unit (E3-4)** `e3_ready` `:2859`, `geo_unit_prereg` `:2886`, `check_l_report_geo` `:2902`,
  `load_predictions_geo` `:2948`: 과제대로 예측은 `results/L_predictions_geo.parquet`(rung geo만, LADDER 열), unit 기록은
  `L_report_geo.json`/`L_xgb_report_geo.json`에서 확인한다. REP·REP8은 E1-3 규칙대로 unit json의 val_kcal_mae에서(예측 파일을
  읽기 전, `no_test_scores` 안) 고르고, L_report_geo·L_val_geo·L_xgb_report_geo와 같아야 한다. geo unit 지문 `prereg_geo2`
  = 현재 PREREG_GEO2.md sha256, PREREG 두 파일 commit·변경 없음, LADDER 스냅숏 변경 없음(명령서 14) — 아니면 STOP.
- **T-25 REP8과 꺾임 부분 (E3-5)** `pick_rep8` `:3035`, `goal_rep8` `:3102`, `add_bend_passed` `:3083`: E3_goal에 LADDER
  13-2(REP 주, ENS4 보조, 모델 설명용) + REP8 행(9-1 규칙: 2 arm × 4 모델 중 5 fold 평균 val MAE 최저, 동점 SH → S → AW·GNN·
  TF·XGB)과 `bend_passed` 열(민감도 ≥ 0.7, FPR ≤ 0.1)을 더했다. 10-1 "목표 도달" = REP8 oi·cpcm의 MAE 부분(eval_geo2
  goal_mae와 같은 수)과 꺾임 부분 둘 다 → `E3_report.json goal_reached`(reached, mae_part, bend_part). REP8 모델의 꺾임은
  그 (arm, model)의 bend rates 그대로다.
- **T-26 E3_compare (E3-6, 설명용 추가)** `compare_e3` `:3147`: 과제의 세 파일 외에 E2_compare와 같은 형식으로 SH vs S(geo),
  g2 vs geo(P_23), dft vs geo(P_023), REP vs LINE(geo)을 냈다. 10-2 "4 모델 각각의 geo 대 g2"를 CI와 함께 보기 위함. 판정에는
  쓰지 않는다.
- **T-27 E2 재현 확인 (E3-7)** `e2_crosscheck_e3` `:3220-3271`: E3가 LADDER g2 own 행을 `E2_metrics.csv`(모든 수치 1e-12
  상대), `E2_bend_rates.json`, `E2_goal.csv`, dft·g2 REP(`E2_report.json`)와 똑같이 내는지 확인한다. 실제 e3에서는 STOP
  (명령서 14 E), selftest에서는 검사 항목. eval_geo2 `E_report.json`의 REP8·goal_mae REP8과의 비교는 기록만
  (`eval_geo2_crosscheck` `:3274`).
- **T-28 e3-selftest (E3-8)** `selftest_e3_checks` `:3294`, `cmd_e3` `:3339`: geo := g2 사본(예측, 검증 기록, P_geo = P_g2,
  fold, dH)으로 같은 파이프라인을 돌려 `analysis/geo2/results/e3_selftest/`에 쓴다. 검사: geo 행 = g2 행(own·P_23·P_023
  지표, goal, REP, REP8, bend rates), g2 vs geo 비교 d = 0, E2 재현, REP8·goal_mae = eval_geo2 selftest(job 1053810).
  geo 입력이 오기 전에 E3 코드를 확인하는 방법이다.

## D. 실행 계획 (RUNBOOK_tests.md)
- **T-29 파티션**: 오케스트레이터 지시대로 GPU `gpu1,gpu2,gpu3,gpu5,gpu6`, CPU `cpu1,cpu2`; 모든 job `--time=48:00:00`,
  `--output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out`. 자원은 LADDER C 검사와 같게(gpu part 4 CPU·32G, cpu part 32 CPU·
  64G, merge 1 CPU·4G, e3 4 CPU·32G = E2와 같음).
- **T-30 지금 할 수 있는 것**: T-R(회귀만, 3 task)과 E-T(e3-selftest, 1 task)는 geo 입력 없이 지금 낼 수 있다.

## E. 작업 중 실수 (보고)
- **T-31** 로그인 노드에서 빌드 보고 확인용 명령 한 줄에 `for` 반복(grep 3회)과 `python3 -c 1`(아무 일도 하지 않는
  명령)이 섞여 실행되었다. 출력·파일 변화는 없었고, 그 뒤로는 반복문·python을 쓰지 않았다.
- **T-37 (리뷰 수정 중)** 로그인 노드에서 읽기 명령 두 줄에 실수로 `python3 --version >/dev/null 2>&1`(버전 출력만,
  버려짐)과 빈 반복 `for f in run_tests.py; do :; done`(아무 일도 하지 않음)이 섞여 실행되었다. 계산·출력·파일 변화는
  없었다. 그 밖에는 grep / sed -n / cat / ls / readlink / sha256sum(작은 파일 3개: S_folds.csv, inputs.py 두 사본,
  run_tests.py pristine 사본, lad_common.py; 리뷰 증거 확인용)만 썼다.

## F. 리뷰 수정 (2026-10-30, 리뷰 6건; 줄 번호 = 수정 뒤)
- **T-32 R-2의 P_report `inputs` 키 (리뷰 2·3, 같은 결함; 반영, G-12b)** 확인: `analysis/d1_ladder/results/P_report.json:51`
  `inputs`의 유일한 키는 `/gpfs/home1/.../eda-asm-prediction-d1ladder/analysis/d1_pair/results/S_folds.csv`이고,
  `inputs.S_FOLDS`(`inputs.py:223`, `DP.RES`는 resolve된 이 worktree 경로)는 `-geo1` 경로다. `inputs.pairs_frame` →
  `_read_s_folds`(`inputs.py:1637-1639`)가 `rep["inputs"][str(S_FOLDS)]`에서 KeyError → `guard`가 fail → R-2 L-10
  `inputs_pairs_frame` fail → R-2 fail(두 part, `--regress-only`와 전체 모드 모두) → `C_tests_geo.json` all_pass 불가.
  내용은 같다(두 worktree의 S_folds.csv sha256 = 944a1d14…, 기록값과 같음 — sha256sum으로 확인). 수정
  `run_tests.py:2629-2721`: `_f13_build_check`가 R-2 동안 `inputs._read_s_folds`도 감싼다(`rel_s_folds` `:2654-2673`).
  native 키가 있으면 원래 함수 그대로(d1ladder worktree에서는 동작 불변). 없으면 `_rel_fp`와 같은 규칙(`_rel_key`
  `:2632`)으로 repo 상대 키가 같은 기록 항목을 찾아, 그 sha256이 모두 현재 파일과 같을 때만 그 값을 이 worktree 키에 둔
  보고 사본을 원래 `_read_s_folds`에 넘긴다(`_verified`의 sha256 재확인은 그대로). 아니면 `build_currency.
  p_report_inputs.disallowed`에 기록하고 RuntimeError(→ 해당 부분검사 fail). `finally`에서 두 원래 함수를 돌려놓는다.
  `_f13_summary` `:2709-2721`은 rung 기록과 `p_report_inputs`를 나눠 보고한다(`build_currency.rungs.<rung>`의 형식은
  그대로). inputs.py와 P_report.json은 바꾸지 않았다. docstring G-12b `:198-201`, `GEO_DECISIONS` G-12 `:2466`,
  RUNBOOK T-R STOP 문구에 기록. L-6의 `LadderData`(`inputs.py:1762-1768`)는 `rep["inputs"]`를 읽지 않고, 학습 코드
  (train_atoms / train_xgb)도 P_report `inputs`를 쓰지 않음을 grep으로 확인했다(다른 키는 /gpfs/tmp_cpu2 경로라 영향 없음).
- **T-33 E3의 P_report `inputs` 키 (리뷰 1·4, 같은 결함; 반영, E3-2b)** 확인: `cmd_e3`가 그대로 부르는 `real_inputs`
  `evaluate.py:715`가 `rep_p["inputs"].get(str(IN.S_FOLDS))` = None으로 STOP(exit 3), 그 뒤 `IN.pairs_frame(rep=rep_p)`
  `:733`도 같은 KeyError. E3-2(T-22)는 build 지문만 피했다. 수정: 새 함수 `f13_inputs` `evaluate.py:2822-2840`이
  P_report의 S_folds 항목을 `_repo_rel`(기존 함수)로 찾아 sha256 = 현재 파일일 때만 `inputs[str(IN.S_FOLDS)]`를 넣은
  보고 사본을 돌려주고(native 키면 그대로), 아니면 STOP. `cmd_e3` `:3374`가 `IN.p_report()` 직후 그 사본으로 바꿔
  `ladder_builds`·`real_inputs`·`geo_inputs`에 넘긴다; 기록은 `E3_report.json`/`E3_selftest.json` `p_report_inputs`
  `:3443`. `real_inputs`와 e1·e2 경로는 한 글자도 바꾸지 않았다(E3-1 유지). `DECISIONS3["E3-2b"]` `:2690`, 모듈
  docstring, RUNBOOK E-T·E3 STOP 문구에 기록. `real_inputs`의 나머지 조회(labels `:722` = `/gpfs/tmp_cpu2` 키, 집합·fold
  파일 = `/gpfs/tmp_cpu2` 키)는 worktree와 무관함을 P_report.json에서 확인했다.
- **T-34 LADDER 모드 쓰기 방지 (리뷰 6; 반영, G-17)** 확인: `main`의 `gpu`·`cpu`·`merge` 분기에 코드 가드가 없었고,
  `run_gpu`/`run_cpu`는 시작하자마자 `results/C_tests_{gpu,cpu}.json`을, `merge`는 `results/C_tests.json`을 쓴다(`abort`·
  `finish`도 `merge` 호출). 수정 `guard_ladder_tests` `run_tests.py:3111-3130`, 호출 `:3164-3168`: `results/C_tests.json`
  이 있고 그 `code_sha256`이 지금 `code_shas()`와 다르면 쓰기 전에 `STOP: ...`과 exit 3(LADA GA-13 / XG-G9와 같은
  방식; 파일이 없으면 예전 동작). 기록된 code_sha256은 pristine 사본과 같으므로(run_tests.py 420c18bf…, inputs.py
  07ab5ba7…, lad_common.py e0d64e1e… 확인) 수정 전 코드에서는 동작이 같고, GEO2 코드에서만 거부한다. 과제의 "기존 모드
  그대로"와 충돌하는 판단이지만, 이 경우 그 모드는 어차피 all_pass false를 LADDER 결과 파일에 다시 쓰는 일밖에 할 수
  없으므로(명령서 8-1 "기존 결과 파일 불변" 위반, 14 PREREG 스냅숏 STOP) 막는 쪽을 택했다. `geo` 모드와 `l4child`는
  영향 없음. docstring G-17 `:212-214`, `GEO_DECISIONS` G-17 `:2474`, RUNBOOK 0절에 기록.
- **T-35 inputs.py 지문에 inputs.py 자신이 들어감 (리뷰 5; 사실 확인, 이 역할 파일 밖이라 기록·보고만)** 확인:
  `inputs._fingerprint` `inputs.py:1063-1066`의 `_code_files()` `:394-396` 첫 항목이 inputs.py 자신이다. `P_build_{dft,
  g1,g2}.json`은 inputs.py sha256 07ab5ba7…(= pristine 사본)를, 지금 파일은 b80a479a…(LADA의 `RUNGS["geo"]` 추가)를
  가진다. 따라서 `_build_ok`(`:1304-1323`)는 어느 worktree에서든(GEO2 병합 뒤의 d1ladder 포함) dft·g1·g2 build를 stale
  (exit 2)로 본다 → `p_report(rung)`, `LadderData`, `reaction_frame`/`pairs_frame`(rep 없이), `inputs.py pairs`,
  `evaluate.py e1`/`e2`(`:1446`, `:2433`), LADDER `prereg.py`의 `IN._build_ok('g2')`가 기존 rung에서 exit 2. 이 geo1
  worktree에서는 절대 경로 키 때문에 원래도 stale이었다. 명령서 8-1이 `RUNGS['geo']`를 inputs.py에 두라고 하므로 피할
  수 없는 명령서 수준의 편차다: 기존 LADDER 명령 중 currency를 검사하는 것은 geo2 코드에서 쓸 수 없게 된다. inputs.py는
  LADA 소유라 고치지 않았고, LADDER 지문을 다시 기록하지도 않았다(LADDER 결과 파일이 바뀜). 이 역할의 코드는 이
  영향을 R-2 동안의 G-12 감싸기(inputs.py 하나만 pristine sha256일 때 허용)와 E3-2(rung 없는 `p_report()` + 표 sha256)로만
  피한다. 사용자에게 보고할 항목: (a) 이 편차를 받아들일지, (b) `RUNGS['geo']`를 inputs.py가 hash하지 않는 별도 모듈로
  옮길지(LADA 결정 필요).
- **T-36 거부한 리뷰 항목**: 없음(6건 모두 사실; 5번은 다른 역할 파일이라 기록·보고로 처리).
