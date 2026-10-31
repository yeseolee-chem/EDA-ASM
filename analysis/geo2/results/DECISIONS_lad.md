# GEO2b 결정 기록 — LADDER 학습 쪽 (설계 step 9–12: D0+D1 라벨·분할, rung geo·split T1D0, C_tests_geo, 사전 등록, M3)

GEO2.md를 GEO1B.md 4-3의 바꿔 읽기로 실행하고, 사용자 결정 D0D1(2026-10-31, GEO1b `DECISIONS.md`)이 GEO2.md와 다른 곳에서 이긴다.
모든 갈림길은 추천안(사용자 0-user, D0D1). 이 역할이 쓴 파일: `analysis/d1_ladder/{train_atoms.py, train_xgb.py, tests/run_tests.py}`,
`analysis/geo2/{d0d1_splits.py, prereg_geo2b.py, lad_selftest.py, m3_launch.sh, jobs/m3_gpu.slurm, jobs/m3_cpu.slurm,
RUNBOOK_lad.md, RUNBOOK_tests.md}`, 이 파일. 출발점은 브랜치 geo2(0a912e78)의 검토 전 GEO2 코드이며 줄 단위로 검토한 뒤
옮겼다(R절). 모델 구조·grid·손실·β 규칙·seed·epoch·patience는 한 글자도 바꾸지 않았다.

## R. geo2 코드 검토 결과 (옮기기 전)
- **R-1** geo2의 `$GEO2` 기본값 `/gpfs/tmp_cpu2/yeseo1ee/geo2`는 GEO2b에서 틀림(GEO1B.md 4-3) → `geo2b`로 바꿈(세 파일).
- **R-2** geo2 `gate_p_geo`는 `P_geo_sets.json`, |P_23| ≥ 0.98 |P_g2|, 그리고 라벨 파일이 LADDER `results/P_report.json`에 기록되어 있기를 요구했다. GEO2b 라벨 `F_pairs_d0d1.csv`는 거기에 없으므로 **항상 STOP**이 된다 → GB-6으로 바꿈.
- **R-3** M3 상한 80 GPU시간 → 160 (D0D1 '상한 조정'). geo2 `estimate_geo`는 g1 unit 시간만 비례 환산했다 → g1·dft 두 기준의 큰 값 + 작업자 수별 slice 시간(GB-9).
- **R-4** geo2 run_tests R-1은 `$GEO2/regress/list_stage*.before.txt`, R-2는 `$GEO2/pristine_head/...inputs.py`를 읽는다. 둘 다 `$GEO2` 삭제(GEO1b DECISIONS 0-2)로 없어져 R-1·R-2가 영원히 waiting → GB-T9(LADDER 결과의 unit 이름), GB-T10(`git show dcd83ab3:analysis/d1_ladder/inputs.py`, 기록된 sha256 `07ab5ba7…`와 같음을 확인)으로 바꿈.
- **R-5** geo2 run_tests는 geo HF parquet 경로·geom을 하드코딩하고 `inputs.RUNGS['geo']`와 비교했다 → inputs 역할의 `inputs.RUNGS['geo']`를 유일한 출처로 읽고, 학습 코드의 geo 경로(fu2_res, tokens, x182)와 같은지 확인(GB-T6, `_geo_interface`).
- **R-6** geo2 L-11은 rung geo에서도 `F_pairs_cr2.csv`로 로드한다 → split T1D0의 D0 id가 라벨에 없어 STOP → L-11′ 새 함수(GB-T3).
- **R-7** geo2 L-5 기준 에너지 검사는 `P_build_geo.json`의 `references_g3_cons_own` 키와 `'|fixed:'` 접미사로 고정 키를 골랐다. D0 키는 `'|cfg:'`·`'|s3d:'`를 쓴다(GEO1b L-key) → 표의 `ref<k>_src`(min / own)로 판정(GB-T6).
- **R-8** geo2 L-4는 `inputs.check_paths`만으로 판정 → D0 원천($GEO1B/d0, analysis/geo1b/results)이 unknown이 되어 fail → GB-T5.
- **R-9** L-8 `val_fraction`(0.15–0.35)은 분모가 학습 짝 전체라 D0가 들어가면 흔들린다 → T1D0에서는 D1 학습 짝 기준(GB-T7).
- **R-10** geo2 `prereg_geo2.py`는 `F_pairs_cr2.csv`를 라벨로 고정하고, GEO1(G1 학습) 숫자·Y plan·10절 판정을 담는다 → GEO2b에서는 쓸 수 없어 `prereg_geo2b.py`로 새로 씀(판정 없음, GEO1b 숫자, D0 규칙).
- **R-11** geo2 `g2_launch.py`(전체 DAG 오케스트레이터, 10분마다 자기 재제출)는 이 역할 범위 밖이고 필요 없다 → 쓰지 않음. M3는 의존성 사슬 한 번 제출(CLAUDE.md 선택지 3, `m3_launch.sh`).
- **R-12** 그대로 둔 것(검토 통과): GA-13 / XG-G9 / G-17 쓰기 거부 가드(LADDER 결과 파일 보호), `res_out` `_geo` 출력, collect geo 전용, `cmd_l6(3)`, R-3(g2·S·XGB fold 0 재현), F-13 build 지문 비교, `_check_root` rung 규칙, 스냅숏(GA-5).

## L. 라벨과 분할 (`analysis/geo2/d0d1_splits.py`, 설계 step 9)
- **LD-1** 라벨 파일 = `F_pairs_cr2.csv` 바이트 그대로(헤더 포함) + `D0_pairs.csv`의 **모든** 행을 F_pairs 열로(텍스트 그대로 복사). `edge_kind`, `ok`는 열이 달라 복사하지 않음(D1 행 바이트 동일 유지; pair_id로 D0_pairs.csv와 join).
- **LD-2** 분할에는 ok True D0 짝만 넣음(설계). D0 짝 id 정렬 순서로 FU2 목록 **뒤에** 붙임 → D1 부분 = FU2 T1__f<k> 그대로(순서 포함).
- **LD-3** D0 짝의 fold = d0_folds.csv의 a_rxn fold = b_rxn fold. 다르면 STOP(지금 0건).
- **LD-4** D0 입력은 GEO1b 최종 build만: `D0_pairs_report.json` stop false, g2_source `g2c`, preliminary false, status `ok`, outputs sha256 = 파일; `Q_d0_report.json` all_pass·stop false, sha256 = D0_pairs_report 입력 기록(경로는 repo 상대로 대조, F-13); labels_d0_q sha256 일치.
- **LD-5** 보고서는 **이 worktree**의 `analysis/geo1b/results/`를 읽는다 → GEO1b 최종 commit(G2c 뒤 d0_pairs 재실행)을 geo2b로 가져와야 split build가 돈다(인터페이스, RUNBOOK_lad S-0).
- **LD-6** preliminary 입력은 `--test-dir`(+`--allow-preliminary`)에서만 받음; 그때 json은 preliminary true이고 학습 코드가 거부. 지금 시험은 `$GEO2B/tests_lad/splits_prelim`.
- **LD-7** D0·D1 rxn_id 겹침 = STOP(한 토큰·표 행 per rxn_id). twin: fold k 학습 D0 반응에 fold k에 묶인 D1 twin(forced / iso_resolved) = STOP, conflict(사용자 승인)는 보고만.
- **LD-8** 실행 순서 고정: d0d1 build는 json을 **마지막에** 씀; 읽는 쪽은 json의 sha256과 파일을 대조 → 중간에 죽어도 안전, 같은 입력이면 바이트 동일(시험 확인).

## T. 학습 코드 (train_atoms.py GB-*, train_xgb.py XG-B*)
- **LT-1** split은 rung이 정함: geo → T1D0, 나머지 → T1(기존 동작 그대로). 기본 라벨은 stage가 정함: 3 → F_pairs_d0d1.csv.
- **LT-2** 공통 규칙은 `d0d1_splits.py`를 두 학습 코드가 import(복사 아님; numpy·pandas만). 지문에 `d0d1_code`(그 sha256), `split`, `prereg_geo2b` 추가(geo unit만; T1 unit 지문 불변).
- **LT-3** L-11′: D1 행 = cr2 바이트, cr2 자체는 LADDER 출처(V_report)·L-11 통과, D0 행 = D0_pairs.csv 텍스트, D##### ↔ ref_kind d0.
- **LT-4** XGB 안쪽 fold: GroupKFold(4)를 D1 부분에만, D0 학습 짝은 모든 안쪽 학습 쪽에 넣고 검증 쪽에는 넣지 않음(arm P 선례); fold 0 = 분할 파일 확인. D0 짝 core는 라벨 파일의 'D0/<component>'.
- **LT-5** gate M-1′를 D0 학습 짝 200개에도 모든 학습마다 돌림(원자: 같은 gate 함수를 따로 한 번, 결과 `gate_M1_d0_train`; XGB: gate 짝에 붙여 같은 기준). 학습 RNG와 무관한 자체 seed. T1 unit에는 키가 생기지 않음.
- **LT-6** P gate(GB-6): `P_build_geo.json` all_pass·not stop·표 sha256, `P_geo_report.json` stop false만. 라벨은 S_splits_d0d1.json이 기록.
- **LT-7** s_y·SH β 진단은 `train_xgb.py diag --stage 3`(CPU) 하나로; D1 학습 짝만 대 D1+D0, 최종·안쪽 fold 0, 결과는 `results/D0D1_diag_geo.json`(이름에 geo → LADDER 스냅숏 밖). 학습은 읽지 않음.
- **LT-8** GPU 예상(GB-9) = max(g1 기준, dft 기준) 환산 + n = 1..10 작업자별 가장 긴 slice. 판정은 prereg(160 초과 STOP), smoke 판정과 분리.
- **LT-9** 단위 배분은 LADDER `[i::n]` 그대로(9 작업자면 slice 0이 GNN 2개); LPT 등 새 배분 규칙은 넣지 않음.

## C. C_tests_geo (`tests/run_tests.py geo`, 설계 step 10)
- **LC-1** 새 검사 L-D0(GB-T4)와 diag(GB-T11) 추가; L-11 = L-11′(GB-T3). 나머지 L-1..L-9, R-1..R-3는 geo2 로직(위 R절 수정 포함).
- **LC-2** geo rung ready = 표·P_build 통과 **그리고** D0D1 색인 final. preliminary는 waiting(실패 아님), 다른 색인 STOP은 fail.
- **LC-3** smoke 통과 = 모든 구성 ok + D0 학습 짝 gate가 있고 ok + D0 학습 짝 > 0. XGB는 unit json의 `n_gate_pairs_d0_train` > 0.
- **LC-4** M-1′ "200 random D0 training pairs"는 smoke만이 아니라 실제 학습(LT-5)에도 들어가므로 smoke가 그 경로를 그대로 시험.

## P. 사전 등록 (`prereg_geo2b.py`, 설계 step 11)
- **LP-1** 분할 색인은 학습 코드와 같은 읽기 규칙(load_s_splits, final)으로 확인; 아니면 STOP.
- **LP-2** gate 목록(inputs_sha256): 라벨·geo 표(need), X182, fu2_geo 보고 4개, S_splits_d0d1.json, T1D0 5개, cr2, F_pairs.csv, FU2 S_splits·T1 5개, S_folds, V_report, D0_pairs.csv, d0_folds.csv, d0_twins.csv, 학습 코드 13개(inputs.py, d0d1_splits.py 포함). D0_pairs_report·Q_d0_report는 내용 검사(학습 코드)와 증거 sha256만(시각이 바뀌는 재실행으로 막히지 않게).
- **LP-3** C_tests_geo.json code_sha256 중 run_tests·inputs·train_atoms·train_xgb·lad_common·d0d1_splits는 지금 파일과 같아야 함. **evaluate.py는 기록만**(Z/E 역할이 e3를 따로 마무리; 학습 코드 아님).
- **LP-4** smoke 예상이 없거나 > 160 GPU시간이면 STOP(FU2 상한식 대체 없음).
- **LP-5** 판정 없음(GEO2.md 10-1 제외), 절대 기준은 LADDER 13-2 하나, D1·D0 분리 보고. "이미 본 결과"는 GEO1b E_report·E_rmsd(GEO1 숫자 아님, GEO1B 4-3), Z, P, smoke, diag.
- **LP-6** commit 지점: C 검사 통과 → `prereg_geo2b.py write` → 사람이 md 읽고 PREREG 2개 + 최종 코드 commit·push → `m3_launch.sh m3`(첫 job이 `prereg_geo2b.py check`).

## M. M3 (설계 step 12)
- **LM-1** `m3_launch.sh m3`: prereg check → XGB run(1 × 32 CPU) → XGB check; 원자 array 0..8(9 GPU) ← prereg check; collect --with-xgb ← 원자·XGB check; prereg check 다시(LADDER 결과 불변). 대기열 14칸(9 + 5), 실행 최대 10.
- **LM-2** 래퍼 `jobs/m3_gpu.slurm`(env_d1fu2), `jobs/m3_cpu.slurm`(M3ENV espley / d1fu2 / reactot): analysis/geo2에서 제출, `GEO2_SCRATCH`·`D1LADDER_SCRATCH`·`D1PAIR_SCRATCH` 고정, 기본 로그 `$GEO2B/logs`. 기존 LADDER 래퍼는 바꾸지 않음.
- **LM-3** 죽은 원자 task는 같은 GPU 종류에 `--array=<k>`로 다시(지문에 GPU 이름; LADDER MA-7 그대로).

## E. 작업 중 실수 (보고)
- **LE-1** 2026-10-31 18:5x, 로그인 노드에서 결과 파일을 `sed`로 보는 명령 앞에 `python3 -`(빈 표준 입력, 아무것도 실행하지 않고 끝남)가 한 번 섞여 실행되었다. 계산·파일 변화 없음. 이후 python은 모두 `srun --overlap`(계산 노드)로만 돌렸다(CLAUDE.md 규칙 위반으로 기록).
