# GEO2b Phase Z·E 판단 기록 (역할 ZE: build_geo.py, geo2_common.py, eval_geo2.py, eval_d0d1.py, tests_ze.py)

모든 분기는 명령서와 맞는 가장 단순한 추천안이다(사용자 0-user, D0D1, A-G2c). 이 역할은 코드와 검사만 했고 계산 job은 내지 않았다. 검사는 돌고 있는 내 job 안에서 `srun --overlap -c1`로만 돌렸다. 단, 확인 중 login node에서 빈 입력의 `python3 -`를 한 번 실행했다(스크립트 없음, 아무 일도 하지 않음; CLAUDE.md 규칙 위반으로 기록).

## Phase Z (설계 step 5, GEO2.md 4)
- **Z-g2c** (geo2_common.py, build_geo.py `load_g2c_select`, `g2c_state`): G2 루트 = G2c. D1 `G2_FINAL` 기본값 = $G2C/d1/g2_final, D0 `G2_FINAL_D0` = $G2C/d0/g2_final. rel1·rel2와 대체 TS는 선택된 G2c 폴더에서 온다. 출처 검사는 GEO1b data.py와 같다: C_select.csv chosen ≠ '' ⇔ `.done`, g2c.json의 rxn_id·src·chosen 일치, ts·rel1·rel2 있음. 폴더를 쓰기 전에 모든 대상을 검사하고, 어긋나면 STOP. C_select.csv가 없으면 STOP(G2c build 미완).
- **Z-d0** (`load_targets`): D0 대상 = labels_all.json status ok(D0_EXCLUDE 5개 제외) − d0_folds.csv overlap 5개 = 5,255. uid D0_<rid>, job_id D0G2_<rid:04d>, 라벨 TS = Coley profile TS(full_dataset_profiles/<rid>/<ts_file>; GEO1b data.py가 읽는 파일, labjobs 사본과 같은 좌표). 쌍 = labels_all formed_pairs_ts.
- **Z-layout**: D1(rxn_id ≥ 1000001)과 D0(0–5973; 검토 R-4)이 한 루트 $GEO2B/geo_final/<rxn_id>/를 쓴다. 파일은 GEO2.md 4-1 그대로(ts.xyz, rel1.xyz, rel2.xyz, result.json, .done). 번호가 겹치면 STOP.
- **Z-folds** (`load_d0_folds`): d0_folds.csv는 P-stop 읽기 규칙(D0_pairs_report.json stop false, outputs sha256 = 파일)과 overlap ⇔ fold −9 규칙으로 읽는다. D0 예측의 pkl tag가 f<d0 fold>가 아니면 STOP. D1 fold 대조는 GEO1b evaluate.py preds_for가 이미 단언하므로 더하지 않았다.
- **Z-stray**: pkl에 대상이 아닌 uid가 있으면 STOP(오래된 pkl, 다른 라벨).
- **Z-gates**: E_rxn 대조(1e-6 Å)와 대체 비율 ≤ 2 %를 D1·D0 따로 본다. D0 대조 파일 = GEO1b E_rxn_d0.csv(GB-eval-d0, 같은 열 e_short_model·e_long_model). 대조 파일에 없는 반응(D1 T1 fold −1)은 `n_built_not_in_file`로 보고만 한다(검토 R-1: fold 0–4에서 만든 반응 집합 = 대조 파일 행 집합이어야 한다).
- **Z-sources**: result.json `sources`에 G2c ts.xyz sha256(`g2_ts`)와 `g2c_chosen`을 더했다. 대체가 그 TS를 쓰기 때문이다. 검사·대체 규칙·폴더 형식 등 D1 동작은 그대로다.
- **Z-report**: Z_report.json은 시작할 때 stop true('running')로 먼저 쓰고 마지막에 최종 내용을 쓴다. Z_status.csv도 원자적으로 쓴다(원래 코드는 open()으로 바로 씀). Z_status.csv 열 = rxn_id, job_id, src, status, source, fallback, reason, g2c. D0 겹침 5개는 status `d0_overlap_excluded` 행으로 남긴다(ok·kept가 아니므로 feat_geo geo_state 대조와 맞음). Z_report에 `by_src`(D1·D0 각각 개수, 대체 비율, E_rxn 대조)를 더했다. 기존 키는 유지하고, `e_rxn_check`는 {D1, D0} 사전이다.
- **Z-env** (geo2_common): 테스트 전용 덮어쓰기 `GEO2_RESULTS`(RES), `GEO2_GEO1_RES`(GEO1b 결과), `GEO2_G2_FINAL_D0`, `GEO2_D0_LABELS`, `GEO2_D0_PROF`를 더했다. 기본값은 명령서 경로 그대로다.
- **Z-geo1res**: GEO1_RES 기본값 = 이 worktree의 analysis/geo1b/results(geo2b.patch). Z·E 전에 GEO1b 최종 결과(E_rxn.csv, E_rxn_d0.csv, G2c 뒤 D0_pairs_report.json)가 이 worktree에 있어야 한다(geo1b merge). 지금 이 worktree에 있는 D0_pairs_report.json은 G2c 전 예비 판이다.

## Phase E (설계 step 13, GEO2.md 9)
- **E-1**: `eval_geo2.py run --rungs geo` 대신 얇은 `eval_d0d1.py`를 새로 썼다. LADDER evaluate.py는 고치지 않고 import해서 E2 함수(fu2_eval, label_info, prep_S, read_lab, model_frames, check_pair_attrs, ref_set, derived_frames, gate_pairs, check_line, annotate_all, metrics_e1, met_index, bends_e1, goal_e1)를 rung geo에 그대로 부른다. eval_geo2는 metrics, reps, select, boot_diff, app_metrics, by_class, goal을 그대로 쓴다. geo2 브랜치의 검토 전 e3는 쓰지 않았고, goal_rep8·add_bend_passed만 검토 후 옮겼다. LADDER 결과 파일은 읽기만 한다.
- **E-2**: REP·REP8·ENS4는 geo unit 기록의 val_kcal_mae(L_report_geo.json + L_xgb_report_geo.json)로만 고른다. T1D0의 inner_val은 D1뿐이다. D0도 같은 선택을 쓴다(D0 검증 없음). unit 이름은 '__'로 나누고 rung = 둘째, arm·model·f<k> = 마지막 셋으로 읽는다(분할 토큰이 끼어도 됨). $LAD/units의 unit json이 있으면 값이 같아야 하고, smoke unit이면 안 된다.
- **E-3**: geo LINE = SH unit의 base 열(학습한 그대로의 β × dH). check_line으로 4 SH 모델이 같은 base·β이고 S 행은 0임을 확인한다. 입력표로 다시 계산하지 않는다. null = 0.
- **E-4**: D1 행 = 'd0'이 아닌 행(S_folds.csv 짝 id), D0 행 = ref_kind 'd0'(D0_pairs.csv 짝 id). 두 기준이 다르거나 모르는 id가 있으면 STOP.
- **E-5**: D0 평가 짝 = 예측된 D0 짝 중 D0_pairs.csv ok true. ok false가 예측에 있으면 세고 뺀다. 짝 fold ≠ d0_folds.csv(두 끝)이면 STOP(누수). y ≠ D0_pairs.csv dd(1e-9)도 STOP.
- **E-6**: D0 CI = component bootstrap(component = D0_pairs.csv core 'D0/<component>', 2,000회, seed 20261019). eval_geo2.boot_diff(e, 0, component)와 같고, 재표집 가중치만 component 수별로 캐시한다(selftest·합성 검사에서 차이 0).
- **E-7**: D0는 subset all과 no_m1_twin(두 끝 모두 T1 fold −1 D1 twin 없음; DECISIONS_PAIR P-y-m1 권고)으로, edge_kind all·sibling·dipole·leaf로 나눠 낸다.
- **E-8**: 판정 없음. g2·dft 행이나 열도 없다. `eval_geo2.py run`(GEO2 비교와 판정)은 exit 2로 막았다. selftest는 그대로 둔다.
- **E-goal**: LADDER 13-2(oi·cpcm para MAE ≤ 0.13, 꺾임 민감도 ≥ 0.7, 위양성률 ≤ 0.1, 점 추정, CI 상한 ≤ 0.13이면 '확실히')를 유일한 절대 목표로 설명용으로 낸다. REP8이 주(GEO2.md 9-1), REP·S·REP·SH(LADDER 13-1), ENS4 보조, 모델별은 설명용이다. E_metrics(eval_geo2)와 E3(LADDER)의 oi·cpcm para MAE가 1e-9 안에서 같은지 대조해 기록한다.
- **E-sets**: E_app·E_bin은 GEO2.md의 P_023 대신 own_geo(geo 예측이 있는 P_main 기준 짝)에서 낸다. 비교할 rung이 없다. E_bin 구간 = GEO1b E_rxn.csv bin(짝의 나쁜 쪽). run은 E_rxn.csv가 없으면 exit 2다(모두 'na'가 조용히 나오는 것을 막음).
- **E-selftest**: geo := LADDER g2 예측·unit 기록(D1), D0 = 실제 ok D0 짝 위 합성 예측(AW·SH = 정답). 출력은 results/selftest_d0d1/. LADDER E2 파일(E2_report, E2_metrics, E2_bend_rates, E2_goal)과 맞춰 본다. run의 unit 읽기(load_units_geo, unit_json_check)도 실제 g2 파일로 확인한다.
- **E-report**: run은 계산을 시작할 때 E_report.json을 'running'으로 먼저 쓰고, 모든 표를 쓴 뒤 마지막에 최종 보고를 쓴다(완료 표지). 입력이 없으면(exit 2) 아무것도 쓰지 않는다.

## 검사 결과 (2026-10-31, 계산 노드 srun --overlap)
- `tests_ze.py z` (합성 D1 60 + D0 62반응, build_geo.py subprocess): 14/14 PASS. env_d1fu2(py 3.11, numpy 1.26)와 D1 run env(py 3.9, numpy 2.0.2) 모두 통과. results/Z_tests_synth.json.
- `tests_ze.py e` (eval_d0d1 D0 부분, 합성): 24/24 PASS (reactot). results/E_tests_synth.json.
- `eval_geo2.py selftest`: all_pass true(E2 40행, 차이 0, 84 s, env_espley). results/selftest/.
- `eval_d0d1.py run` 경로 임시 점검(저장소 밖 임시 사본: g2 unit 기록을 L__geo__로 바꾼 L_report_geo·L_xgb_report_geo, g2 예측 + 합성 D0의 L_predictions_geo.parquet, 결과 루트 임시): exit 0, 8개 표가 selftest 출력과 바이트 단위로 같음. 지금 실제 run은 L_report_geo.json이 없어 exit 2(아무것도 안 씀). `eval_geo2.py run`은 exit 2.
- `eval_d0d1.py selftest`: 22/22 PASS(85 s, env_espley). E3_metrics 3,200행이 E2_metrics g2 own과 모든 열에서 같음(LINE 1e-8 안). bend rates, goal 행, REP, REP8 같음. 음성 검사 6개 모두 STOP. results/selftest_d0d1/E_selftest.json.

## 검토 (2026-10-31, 적대적 검토 후 고침; 계산 job 없음, 검사는 srun --overlap -c1)
- 검토 R-1 (major, build_geo.py E_rxn gate): 대조 파일에 있는 반응만 비교해서, fold 0–4 pkl로 만든 반응이 E_rxn.csv / E_rxn_d0.csv에 없거나 파일에만 있는 반응이 있어도 통과했다(D1은 fm1 때문에 `n_built_not_in_file` > 0이 정상이라 눈에 띄지 않음). 평가 뒤 바뀐 pkl, 다른 G2c 루트에서 나온 E_rxn이면 GEO1b가 채점하지 않은 구조로 Z가 통과한다(GEO2.md 4-3 "평가에 쓴 바로 그 구조" 위반). 이제 dataset마다 {final_f0..4로 만든 반응} = {대조 파일 rxn_id}가 아니면 STOP(fm1 반응이 파일에 있어도 STOP). e_rxn_check에 n_file, n_built_f04_not_in_file, n_file_not_built_f04를 더했다.
- 검토 R-2 (minor, build_geo.py): D0 pkl tag ≠ d0_folds.csv fold 검사가 쓰기 반복 안에 있어 앞 반응 폴더를 쓴 뒤 STOP했다. 모든 쓰기 전에 검사하도록 옮겼다.
- 검토 R-3 (minor, build_geo.py): 지금 만들지 않는 대상(no_g2, no_geo1_prediction, D-1 겹침)의 geo_final/<rxn_id>/.done이 남아 있어도 Z_report가 stop false였다(feat_geo geo_state가 나중에 STOP). 이제 STOP 이유로 넣고 `n_stale_folders`를 적는다. 자동으로 지우지 않는다.
- 검토 R-4 (minor, 문서): D0 rxn_id 범위를 0–5268로 적었으나 labels_all.json은 0–5973이다. docstring, Z_report layout, Z-layout 항목을 고쳤다(동작 변화 없음).
- 검토 R-5 (minor, eval_d0d1.py check_predictions): D1 시험 짝 목록을 예측 파일 자체에서 만들어(LADDER E2는 Phase P fold 목록) 다른 fold에 들어간 D1 시험 짝(누수)을 잡지 못했다. 이제 D1 행 fold = S_folds.csv fold_T1(T1 시험 = fold_T1 == k, inputs.py·FU2 evaluate D6)이 아니면 STOP. 실제 LADDER g2 행에서 통과함을 selftest로 확인했고 음성 검사를 더했다.
- 검토 R-6 (minor, eval_d0d1.py): eval_geo2 함수의 STOP(SystemExit 'STOP: ...')이 exit 1로 끝나 문서의 STOP 코드 3과 달랐다. run·selftest의 evaluate 호출을 `_stop3`로 감싸 exit 3으로 바꿨다.
- 고치지 않은 것: (a) 이 worktree의 D0_pairs_report.json은 예비판(stop false, g2_source d0_g2_orig)이라 P-stop 규칙을 통과하지만, 최종 D0 파일과 sha256이 다르면 STOP하고 GEO1b data.py의 P-g2-gate가 예비 입력의 pkl을 막으므로 더 막지 않았다(Z-geo1res대로 geo1b 최종 결과를 먼저 옮긴다). (b) E_metrics 대 E3 대조(cross)는 기록만 한다(E-goal). (c) D1 fold −1(fm1) 구조는 E_rxn 행이 없어 따로 대조할 수 없다(train_m1.py 단언에 기댄다).
- 검토 후 검사: `tests_ze.py z` 17/17 PASS(env_d1fu2 py 3.11, D1 run env py 3.9; 새 STOP 3개 e_rxn_missing·e_rxn_extra·stale_folder, d0_wrong_fold는 폴더를 쓰기 전 STOP), `tests_ze.py e` 26/26 PASS(reactot; D1 fold 검사 2개 추가), `eval_geo2.py selftest` all_pass true(E2 40행, 차이 0; 출력 바이트 그대로), `eval_d0d1.py selftest` 23/23 PASS(86 s, env_espley; neg_d1_wrong_fold_stops 추가, 표 8개 바이트 그대로).
