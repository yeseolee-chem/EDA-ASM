# DECISIONS_feat — GEO2b 입력 역할(Phase X·S-1·T·G·F·H·P, D1 + D0) 판단 기록

2026-10-31. 근거: docs/specs/GEO2.md 5–7, GEO1B.md 4-3, 사용자 결정 D0D1(2026-10-31, D0D1_design_investigation.json
설계 6–8), A-G2c. 정책: 갈림길마다 권장안을 고르고 여기 적는다(묻지 않음). 코드 안의 결정 원문은 `feat_geo.py`
DECISIONS(GEO-*, F-*), `d0d1.py` DECISIONS(X-*), `inputs.py` 독스트링(PG-*); 실행 때 `results/DECISIONS_feat_runtime.md`에도 남는다.

## 1. 구조
- 결합 루트 하나(설계 F/H-1, X-6 / GEO-12): `$GEO2B/comb` = labels_d0d1.json(D1 labels_d1_g2 9,266행 그대로 + D0 5,255행),
  input_meta_d0d1.csv, x_manifest_d0d1.csv, prof/·jobs/·hf_templates/ symlink. G·F·H·P·T 모두 이것만 읽는다.
- 새 sha 기록 = `results/X_d0d1_report.json`(outputs sha256 + symlink 목록 digest). feat_geo `inputs_check`가 매번 비교
  (g0f0.g2_inputs_check 대체). D1 행 (job_id, rxn_id) = lad_common.targets()도 확인.
- D0 범위(X-0): labels_all.json status ok 5,260 − GEO1 D-1 겹침 5(3221, 3223, 3260, 3285, 3306; d0_folds fold −9) = 5,255.
  job_id `D0G2_<rid:04d>`. labels_all.json은 main checkout 사본, sha256 62c0e784… 고정.
- dataset 구분 = rxn_id 범위(D1 >= 1,000,001, D0 0..5973). 겹침 없음 확인(코드 STOP).
- 로그인 노드 mkdir 금지 → 폴더는 모두 job 코드가 만든다(`C.assert_path` 안: $GEO2B, analysis/geo2/results).

## 2. Phase X (D0 HF 템플릿, `d0d1.py x`)
- 원본: eda.inp = eda_smd_relabel/inputs(SMD(water) relabel), frag*_dist/rel = main label_true/work/inputs. 읽기만.
- X-1 SMD·키워드 = hf_common.smd_ok + prep.check_job 목록 그대로. X-2 좌표 = Coley TS / 기준 파일 = d0_g2 labjobs(<= 1e-6 Å).
  X-3 (1) 태그 = A_idx, 크기·전하 = 라벨, frag1 = dipole.
- X-4(새로 추가): eda.out·frag*.out 정상 종료 + FINAL SINGLE POINT ENERGY = 라벨 값 정확히. 템플릿이 라벨을 만든 그 입력인지 보인다.
- 템플릿 = prep.hf_text 그대로 + check_job diff 검사 이식. x_fail > 0.5 % = STOP(D1-HF X 규칙).
- 실행 중 report는 stop None·running true(하류 exit 2), 안에서 STOP/오류면 stop true + 이유를 쓰고 끝낸다(오해 STOP 방지).
- dry run(전체 5,260 D0): 실패 0, 전하 0:5002 / +1:61 / −2:197, n_atoms 14–94, 45 s.
- X는 GEO1b 최종 전에 돌린다: d0_folds.csv를 이 트리의 D0_pairs_report.json(예비판)과 대조하므로, GEO1b가 d0_folds를
  다시 쓰면 geo1b 결과 병합 전까지 X 재실행은 STOP. 결합 출력은 fold 값을 담지 않아 이후에도 유효하다.

## 3. S-1 (`d0d1.py s1`, X-7 / GEO-17)
- 선택: X ok D0를 (n_atoms, rxn_id) 순 → 중성 가장 작은 2개 + 전하 가장 작은 1개(전하 경로 포함, 비용 최소).
- 원래 B3LYP 입력(orig)을 %pal 1로(pal1_text) D1-HF run.py(dft / orig1), D1HF_SCRATCH=$GEO2B/x_d0. 비교 = D1-HF S1_KEYS 9값 +
  c_ghost(= bsse_shift_kcal), <= 0.01 kcal/mol, status ok.
- GEO-17: h_inputs / h_pilot / h_batch만 S-1 pass를 기다린다(없음·미완 exit 2, 실패 exit 3). T·G·F는 기다리지 않는다.

## 4. 재사용 geo2 코드(0a912e78) 검토 → 고친 점
- scratch `geo2` → `$GEO2B`(g3c.py G3 기본값, geo2.slurm 로그/환경).
- labels / order_g / G2 입력 검사 / _gcheck / atoms·feature 환경이 D1 전용(labels_d1_g2, d1_jobs) → 결합 입력으로.
- order_g의 rel 파일 이름 검사(ref_dipole / ref_dipolarophile.xyz)는 D1만; D0 기준 = labjobs 사본(실데이터 0, 5, 3312 순서 일치 확인).
- d1_hf prep.geominputs(X_manifest·d1_jobs)와 build.allgeom(hf_common.targets)이 D1 전용 → 파일 수정 없이 자식
  _geominputs / _allgeom가 monkeypatch(GEO-15). geo2 T-3은 이 실제 경로를 거치지 않았다 → T-3이 래퍼를 거치게 함.
- pilot "job_id 첫 50개"는 D0G2_*가 T*보다 앞이라 전부 D0 → dataset·n_atoms 사분위 층화 25 + 25(GEO-4).
- rel gate: A-G2c로 geo_final rel이 G2c 선택 run 사본이라 D1-G2 HF와 비교 불가인 행이 생김 → D1 중 C_select chosen == orig
  행만 + rel 파일 바이트 일치 검사(GEO-5). D0·r1/r2 D1은 기준 없음(보고만).
- 상한 2,500(D1만) → 4,000코어시간(D0D1 결정), 예상 = 층화 추정(GEO-4).
- fu2_common.QA_MAX_FAIL = 42(절대 수) → 자식 _qgate가 ceil(0.5 % × n)로 다시 묶음(GEO-13). 파일 수정 없음.
- 실패 비율 gate를 합친 수 → dataset마다(GEO-16). h_batch 누적 검사만 합친 비율(가중 평균 > 1 %면 한쪽은 > 1 %).
- 실행 결정 기록이 공용 results/DECISIONS.md에 씀 → 자기 파일 DECISIONS_feat_runtime.md(F-11).
- Z_report n_folders가 정수/ dataset별 dict 어느 쪽이어도 받게(GEO-1).
- inputs.py make_table / refs_min이 L.targets()·Q_keys(D1)만 → targets_of / _load_keys(PG-3, PG-4), rung geo에서만 다름.
- T 기록 currency: CODE_REL sha256 묶음에 d0d1.py·d1_g2/phase_fh.py 추가(F-10, F-16).

## 5. gate·기준
- G/F/H STOP 비율은 명령서 6-2 그대로, dataset마다. n_imag 0 / >= 2는 실패 아님.
- T-D0(GEO-14): D0 기준값이 없으므로(X182는 G1 구조, 삭제됨) 경로 검사: 경로 오류 0개, 화학적 실패(Freq, B6, ext, SCF) <= 2개/20.
  atoms.py 상태 `h_map`(소유 H 수 문제)은 구조 성질이라 화학적 실패로 분류.
- T-D0 20반응 = X ok D0 n_atoms 사분위마다 5개(rng 20261024), 사분위마다 전하 1개 포함 → 전하 4개.
- H pilot은 64코어(50 작업자) 한 job; h_batch는 W = 10 × 64코어(cpu2 QOS 상한). cpu1이면 48코어·200G.
- 메모리: %maxcore 3500 × 작업자 → 64코어 256G(G2c 64코어 job과 같은 요청), T-3/T-D0H 10코어 64G.

## 6. Phase P
- inputs.RUNGS['geo'] 고정(PG-1) = feat_geo rung_cfg('geo'); cmd_p가 문자열로 비교(GEO-8). 키 targets(결합 라벨),
  keys(D1 Q_keys + D0 Q_keys_d0)는 geo에만. 다른 rung 기본 동작 그대로(unit: 9,266 대상, 18,532 키).
- rung_roots에 $GEO2B, analysis/geo2/results, $GEO1B/d0를 'geo' 열로(PG-2; LADDER check_paths용).
- refs 'min'은 키별(D0 키는 |s3d:/|err:를 가져 D1 키와 겹치지 않음 → 이어 붙이기 = dataset별). constrained 키는 자기 기준
  유지(F-7b): D1 = |fixed, D0 = alt.
- P_23 >= 0.98 P_g2 STOP은 D0D1로 판정에는 안 쓰지만, geo가 반응을 잃지 않았는지 보는 파이프라인 검사로 남김(GEO-9). D0는 fold·
  edge_kind별 usable 짝, 끝이 unusable이라 잃은 짝과 이유.
- p_report는 D0_pairs_report g2_source != g2c(예비판)면 exit 2. 출력 형식 = LADDER gate_p_geo가 읽는 키(all_pass, stop,
  outputs[$GEO2B/inputs/geo.parquet]) 확인함.

## 7. 알려진 결과·부작용
- inputs.py를 고치면 sha256이 바뀐다. 다만 geo2b 트리에서는 원래도 inputs._build_ok(dft/g1/g2)의 지문 키가 d1ladder 작업 트리
  절대 경로라 맞지 않는다(inputs.py pairs / p_report만 해당). feat_geo p_report는 _build_ok 대신 P_build 출력 sha + LADDER
  P_report sha로 확인한다.
- LADDER 역할 C 검사·prereg는 inputs.py sha를 잠그므로 inputs.py는 이 상태로 고정해야 한다.

## 8. 규칙 위반 기록(로그인 노드)
- V-1: 로그인 노드에서 `mkdir -p analysis/geo2/jobs analysis/geo2/results` + `cp`(geo2 브랜치 파일 복사) 실행.
- V-2: srun 명령 앞부분의 `mkdir -p /gpfs/tmp_cpu2/yeseo1ee/geo2b/_selftest`가 로그인 노드에서 실행됨.
- V-3: 로그인 노드에서 `python3 --version` 한 번 실행(버전 출력만, 계산 없음).
- 이후 모든 python·폴더 생성은 srun(job 1055985, --overlap) 안에서만. $GEO2B/_selftest는 비어 있음(내가 만든 것).

## 9. 남은 문제
- Z/E 역할 형식 가정: Z_status.csv(rxn_id, status ok/kept), Z_report.json(stop, n_folders 정수 또는 dataset dict),
  geo_final/<rxn_id>/ 폴더 이름 = rxn_id, D0 rel1/rel2 원소 순서 = Coley 기준(D0-G2 orig에서 확인).
- D0_pairs.csv·Q_keys_d0는 예비판: GEO1b 최종 + geo1b 결과 병합 뒤에야 p_report가 통과.
- C_select.csv(g2c 작업 트리)는 G2c build 뒤에 생김. 없으면 rel gate exit 2.
- T_report가 bad token 행 하나에도 STOP하는 FU2 규칙 그대로(D0 포함 결합 집합에서 더 자주 걸릴 수 있음).
- ext_used(layout) STOP 위험: 결합 집합에서 첫 실행 때 확인.

## 10. 적대적 검토 (2026-10-31, 계산 전)
- 검토 R-1(d0d1.py d0_pairs_state): 이 트리의 D0_pairs_report.json이 예비판(g2_source != g2c)인데 GEO1b 최종
  d0_pairs.py가 D0_pairs.csv를 이미 다시 쓴 경우 sha256 불일치가 exit 3(STOP)이 되어 "병합 전 = exit 2" 규칙과 달랐다.
  예비판 report의 sha256 불일치는 exit 2, 최종(g2c) report의 불일치만 exit 3으로 고쳤다. unit 시험 d0_pairs_state 추가.
- 검토 R-2(RUNBOOK_feat.md): Wave B 작업 수를 16 + 1이 아니라 17 + 1 = 18(MaxSubmit 20에서 다른 대기 작업 2개까지)로
  고쳤다. G 조각 바이트 예상(약 1.6 / 2 GB)과 F-1 STOP 예상(같은 feature 코드가 D1에서 1.14–1.42 %, D1-G2는 사용자가
  받아들임 → p --accept=F-1은 사용자 결정)을 적었다. unit 시험 수 21.
- 검토 R-3(feat_geo.py rel_gate): C_select.csv에 D1 geo 반응의 G2c 선택이 없으면(다른 파일·형식) orig 행이 0개가 되어
  rel 에너지 gate가 비교 없이 통과했다. geo_final D1 폴더는 모두 G2c 폴더이므로 선택(orig/r1/r2)이 없는 D1 반응이 하나라도
  있으면 STOP으로 했다(GEO-5 문구 갱신). unit 시험 rel_gate에 그 경우 추가.
- 확인만(고치지 않음): 실제 D0-G2 orig 4,491 / G2c d0_r1 3,021 / d0_r2 3,034 폴더와 D1 G2c r1 5,547 / r2 4,413 / orig
  8,616 폴더 모두 ts / rel 원소 순서 = 템플릿(라벨) 순서, rel 결합 그래프(formula + WL hash) = Q_keys(_d0) 값(P6b 통과).
  D1-G2 HF 기준 입력의 rel 좌표 = 지금 d1_g2/g2_final rel 파일(17,232개 모두 같음) → rel gate는 같은 입력을 비교한다.
  D0 HF 템플릿(3312 +1, 3323 −2): 'HF D3BJ 6-31G*', FRAG 문자열 2개, %pal nprocs 1, CPCM(water) + %cpcm smd, 전하·다중도·
  (1)/(2) 태그 = 라벨. S-1 후보(3217, 3216, 3393)의 라벨 eda.out은 g_bond_ref·g2_sum 기준 안.

## 11. 연결 검토 (2026-10-31, 조각 사이 인터페이스 검토 후 고침; 계산 job 없음, 검사는 srun --overlap -c1)
- 연결 S-2 (m-1, feat_geo.py `geo_state`): Z_report running true, 또는 stop false인데 complete true가 아님 = exit 2(기다림,
  Z 다시 제출), stop true = exit 3. running이면 표지·Z_status 대조를 하지 않는다(쓰는 중이라 어긋남이 STOP으로 보이지
  않게). GEO-1 문구 갱신. unit 시험 geo_state에 시작 표지·complete 없음(exit 2)과 멈춘 보고(exit 3)를 더했다.
- 연결 S-3 (m-2, feat_geo.py `cmd_p_report`·`cmd_report`): p_report는 시작할 때 P_geo_report.json을 running 표지(stop
  true)로 원자적으로 바꾼다 → 일찍 끝나도 옛 stop false 보고가 남지 않는다. 최종 보고에 running false. report는 running
  = 미완, inputs[<geo 표>] ≠ 지금 sha256 = 미완(stale, p_report 다시)으로 본다. 학습 gate·prereg 쪽은 DECISIONS_lad 연결 S-3.
- 연결 S-5 (m-4): feat_geo.py GEO1B = d0d1.GEO1B(`$GEO_SCRATCH`, 기본 /gpfs/tmp_cpu2/yeseo1ee/geo1b), inputs.py GEO1B_D0 =
  같은 변수 / d0. 이제 geo2_common·d0d1·feat_geo·d0d1_splits·inputs가 모두 같은 변수와 기본값을 읽는다. job 파일 중
  GEO_SCRATCH를 다르게 주는 것은 없다(geo1b gpu.slurm은 같은 기본값). inputs.py sha256이 바뀌었으므로 C 검사·prereg는
  이 판으로 돈다(아직 돌지 않음).
- 연결 S-7 (m-6, RUNBOOK_feat.md): 모든 의존 sbatch 줄에 `--kill-on-invalid-dep=yes`(m3_launch.sh와 같음). wave B를 B1(G 6 +
  g_report 1 + H pilot 1 = 8, N ≤ 12)과 B2(F 사슬 10, N ≤ 10)로 나눴다. 규칙: 매 제출 전 N = `squeue -u $USER -h -r | wc
  -l`, N + 그 제출의 task ≤ 20(G2c 12 task가 아직 있으면 N ≥ 12). wave C 11 task(N ≤ 9), wave D 1 task.
- 연결 S-8 (m-7, 한계로 받아들임 + 보고만 개수): 모든 D0 Q-2 키는 Coley 라벨 기준 구조의 3D 입체 표지 `|s3d:`를 가진다
  (geo1b d0_labels.py, refs_q.stereo3d_sig). rung geo의 refs 'min'은 D0 G2c 기준 구조를 이 키로 묶는다(alt/cfg 행만 자기
  기준). 추천안대로 받아들인다: D0D1 설계와 같고, 탐색에서 D0 G2 기준 조각 21,092개의 formula·WL 불일치는 0개였다.
  한계: G2c(xTB) 기준 구조의 입체가 키와 다르면 다른 입체 이성질체의 HF 기준이 섞일 수 있다(D1의 refs 'min'에는 없는
  D0만의 성질). 보고만: `feat_geo.d0_s3d_check`가 기준 D0 반응의 geo_final rel<k>.xyz에 d0_labels.py와 같은 함수·같은
  인자(fragmenter.read_xyz, smiles, charge)를 다시 적용해 키 행 stereo3d와 다른 개수(그중 묶이는 행 = constrained 아님,
  계산 불가, |err 키 건너뜀)를 P_geo_report.json `d0_reference_s3d`에 적는다. 판정에 쓰지 않고, 실패해도 p_report를 멈추지
  않는다(error 문자열만). 비용 = 약 10,500회 template_match(d0_labels 전체가 129 s). prereg 한계 목록에도 넣었다.
  unit 시험 d0_s3d_count: Coley 기준 구조 10조각 다름 0; 정보로 D0-G2 orig rel 78조각 중 2개 다름(묶이는 행 1).
- 연결 S-10 (F-1, 기록만): D1의 F-1 STOP은 예상된다(같은 feature 코드가 1.14–1.42 %, D1-G2 1.36 %는 사용자 수용,
  analysis/d1_g2/results/F_DECISION.md). 사용자 정책("spec STOP에서는 추천안을 고르고 기록")에 따른 처리:
  D1과 따로 본 D0 모두 F-1 비율이 선례 범위(≤ 1.42 %)이고 실패 종류가 선례와 같을 때만 `p --accept=F-1`(+ report
  --accept=F-1)과 DECISIONS_feat_runtime.md 'F-1 accept' 기록(선례 인용, 빠지는 반응 = 이탈). 범위 밖·다른 종류·다른
  STOP이면 멈추고 보고. RUNBOOK_geo2b.md 4절, RUNBOOK_feat.md F·P 항목.
- 연결 뒤 검사: `tests_feat.py lint` 미정의 이름 0(4개 파일), `unit` 22/22 PASS(reactot; d0_s3d_count 추가, geo_state 확장),
  `wrap` all_pass true(reactot), `gorder` all_pass true(D1 run env).
