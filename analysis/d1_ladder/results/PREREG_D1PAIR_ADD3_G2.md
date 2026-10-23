# PREREG ADD3 보완판 — D1PAIR 사전 등록 추가본 (D1-REF·LADDER 2단계: g2, 2026-10-23)

이 문서는 `PREREG_D1PAIR_ADD3.md`(commit cae9922b, sha256 `daa82585c0ab06653b582524d9fbb35a99dddff9f5c0380e3f8e6b4514cd5667`) 10절의 2단계 보완 규칙에 따라 새 파일로 쓴 보완판이다. ADD3 md는 한 바이트도 바꾸지 않았다(sha256과 git 상태를 다시 확인했다).
`prereg_d1pair_add3.json`(commit cae9922b, 보완 전 sha256 `a8258d7376d301a7b4e4d614957791af87a4d50e7abc4a1cdf755d101fc89a13`)에는 10절이 허용한 것만 더했다: `inputs_sha256`·`inputs_roles`의 새 항목 8개(10절 목록, 같은 순서)와 새 최상위 키 `supplement`(이 md의 경로·sha256·commit). 이미 있는 항목과 키는 그대로다(더한 것을 빼고 다시 쓰면 commit된 json과 바이트 단위로 같음을 쓰기 전과 쓴 뒤에 확인했다).
이 문서와 json 추가분은 `analysis/d1_ladder/prereg.py supplement`가 결과 파일만 읽어 만들었다(코드 commit `83c8cb358b8cadae35e2e0c7013ed488a0921b95`, 생성 코드 sha256 `b14eb6546d0074b17bce2465e6d622f0d503f22f2a1381596a4fbbaabd918d51`, git 상태 commit됨(HEAD와 같음)). 결과 수치는 모두 결과 파일에서 옮겼고, 규칙은 명령서와 ADD3를 인용했다.
범위: **2단계(rung `g2`, 명령서 12절)**. 1단계 규칙은 바꾸지 않는다(7절).

M2 학습(`train_atoms.py run --stage 2`, `train_xgb.py run --stage 2`)은 이 문서와 json 추가분이 commit된 뒤에만 돈다. 두 학습 코드가 매 실행 전에 확인하는 것은 ADD3 md·json(`gate_add3`), `P_report.json`(MA-16), `C_tests.json`의 `g2_verified`(MA-17)다(6-2절). 학습 코드는 이 보완판을 읽지 않으므로, 이 보완판의 commit·sha256은 `prereg.py check --stage 2`와 E2가 확인한다.

**지금까지 본 것** (작성 2026-10-23 21:37:30):
- 2단계 학습은 한 번도 돌지 않았다(`$LAD/units`의 `L__g2__` 파일 0개).
- 1단계 학습: unit json 29/80개. 있는 1단계 결과 파일: `analysis/d1_ladder/results/L_xgb_report.json`. 이 보완판이 새로 정한 것(2·3·4·5절)은 g2 구조·입력에 관한 결정이고 1단계 점수에 따라 고른 것이 아니다.
- g2에서 지금까지 계산한 것은 구조와 입력뿐이다: D1-G2 Phase G(TS), C(구조 비교, 같은 TS 판정), F(feature·토큰), H(HF), LADDER Phase P(g2 입력표, P_g2, P_012), Phase C 검사의 g2 항목(L-1/cpu, L-1/gpu, L-11/cpu, L-11/gpu, L-2/cpu, L-2/gpu, L-4/cpu, L-4/gpu, L-5/cpu, L-5/gpu, L-7/gpu, L-8/cpu, L-8/gpu, L-9/cpu, L-9/gpu). 명령서 8절의 smoke는 rung dft·g1이다. 이 명령은 g2 예측 파일을 열지 않으며 g2 시험 짝 점수는 이 문서에 없다.

## 1. 2단계가 더하는 입력 (g2 열)

**gate에 더한 항목** (json `inputs_sha256`·`inputs_roles`; 열 = 명령서 4-1의 rung 열, `inputs.check_paths`):

| 파일 | 역할 | 열 | sha256 |
|---|---|---|---|
| `$LAD/inputs/g2.parquet` | input table g2 (need of a g2 process) | g2 | `88ef27c168f2f738ffcbdc2636bb432af91e6b6eb951437da99e012c5aeb0d87` |
| `$LAD/inputs/sets/P_g2.csv` | pair set P_g2 | g2 | `3a475899e9a634f5325c444221d8649f60b18759973defd3fd397a4dbff21240` |
| `$LAD/inputs/sets/P_012.csv` | pair set P_012 | dft+g1+g2 | `ed3f7a24368443fb35369b052fce7776496b58687b71e776b5c22308b92615bf` |
| `analysis/d1_g2/results/fu2/T_report.json` | tokens g2: T_report.json | g2 | `9335ea25eee42649edabb4e72b0130abe511fc27bb15e782a68868f3407aa6ec` |
| `analysis/d1_g2/results/fu2/T_manifest.csv` | tokens g2: T_manifest.csv | g2 | `6590f59d8fbc9fc9d2625bb94388aa1b5257f63154fe94e73dd24e29368e00e5` |
| `analysis/d1_g2/results/fu2/Q_report.json` | tokens g2: Q_report.json | g2 | `e10fb7bb5c9377302ecd8b95a04d354bc14807b469a582d86bb3f0d2c464432d` |
| `analysis/d1_g2/results/fu2/S_splits.json` | tokens g2: S_splits.json | g2 | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `$G2/X182_d1_g2.parquet` | global 182 features g2 | g2 | `2cecf407e5f7cfa3ff75ce4e7a6482ad9ee2802c8b3ae449843dfcbbb636e384` |

- g2 process의 `need`(학습 코드 `cmd_run`: 정답 파일 + 학습하는 rung의 입력표): `$LAD/F_pairs_cr2.csv`, `$LAD/inputs/g2.parquet`.
- 보완된 json으로 학습 코드 gate의 내용 조건을 지금 파일에 그대로 적용한 결과: g2 process는 27개를 확인하고(정답·공유 파일·코드·`P_012.csv`·g2 열) dft·g1 열에만 속한 14개는 열지 않는다(MA-15, L-4). 1단계 process(dft·g1)는 34개를 확인하고 g2 열에만 속한 7개는 열지 않는다. `P_012.csv`는 세 rung에 걸친 파일이라 1단계 process도 확인한다(바뀌면 1단계 학습도 exit 3). 둘 다 통과. git 조건(두 파일이 commit되어 HEAD와 같음)은 commit 뒤에 참이 된다.
- 토큰 npz(반응별)의 sha256은 g2 `T_manifest.csv`에 있고 학습 코드가 읽을 때마다 확인한다(1단계와 같음).
- 코드: commit `83c8cb358b8cadae35e2e0c7013ed488a0921b95`. 학습 코드·FU2 코드(두 학습 코드의 지문 `code_sha256`)는 ADD3 json 항목 그대로이고 g2 process의 gate도 그 sha256을 확인한다(위 결과에 포함).

**기록만 하는 g2 관련 파일** (gate 밖; json `supplement.records`):

| 파일 | sha256 | git |
|---|---|---|
| `analysis/d1_g2/results/labels_d1_hf_g2.parquet` | `387ce1290465faaee470224dec5c785277fdd746c9fbbd278a8f45d251521f2d` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/F_report.json` | `4e384eb95850fc55db1733cc99296dd1945b6d49f05a8df20d8d76191bbac08c` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/H_report.json` | `9826c33d90ec892e56e59f0350f04cf915668c436432f9396d21d0f8cf251f40` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/F_DECISION.md` | `1d2095c94c34424356dfa2911e159317ff98df589eda322c2243025068d869ab` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/C_summary.json` | `f3213e71e0505d4827683529fda31d2eac61d332d224ac7e14e561063004f3db` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/C_compare.csv` | `99a309eb5539dad05e20f4540238d97fb1743743d983cdae6806335de7b19a74` | commit됨(HEAD와 같음) |
| `analysis/d1_g2/results/fu2/Q_manifest.csv` | `144c065c68da86862c4757b65647340a7506af65428cc3dcc78f14304a0e958e` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/results/P_build_g2.json` | `5c8a8c58518fa42f44e59181566e830e34a8751a345b59684286c9d2c9415553` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/results/P_report.json` | `2f5ef4c0a067e44fad8ee913dc2cf43bbcd150228286eb78225551ea0e9b978d` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/results/C_tests.json` | `d1e31dd3d70bbc1463293c6385c138286d721f2cb16867d83a64a1a7eb8ee069` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/results/C_tests_gpu.json` | `fb64ed9c329ecd6b3134327029b7513fb7f9c3994cb4bc08866647e0a1043562` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/results/C_tests_cpu.json` | `11b29cfa2b8ae3123a2b9c5e9d82ec8ca7012d2240b2492762bc0877e9ebbedf` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/inputs.py` | `07ab5ba7f605cfbbbd4dd891df1c13ef6a5c2988ae18f25df160d45d4f41acfd` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/tests/run_tests.py` | `420c18bf338caae41e03ab2aad1bfdbb5553b0bbe26c33f1cee90331a06a9044` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/evaluate.py` | `29b997cc954c66f8ea1ce0654b4f828408756a5cf17c7fabd7a764a44e7aff4b` | commit됨(HEAD와 같음) |
| `analysis/d1_ladder/prereg.py` | `b14eb6546d0074b17bce2465e6d622f0d503f22f2a1381596a4fbbaabd918d51` | commit됨(HEAD와 같음) |

**g2 입력표** `$LAD/inputs/g2.parquet` (`P_build_g2.json` sha256 `5c8a8c58518fa42f44e59181566e830e34a8751a345b59684286c9d2c9415553`, 작성 2026-10-23 19:19:02, git 8aade2964a): 대상 9,266반응 중 토큰 8,499, 182 8,501, H 9값 8,616, 사용 가능 8,499. gate P-a(원래 기준으로 `labels_d1_hf_g2.parquet` 재현): 통과, 최대 차이 0.0 (8,616행). 토큰 `fu2-tokens-3-g2` / `fu2-atoms-3-g2`, 원소 목록이 g1과 같음: True. EDA 6채널은 재조립으로 바뀌지 않았다: {'elst': True, 'pauli': True, 'oi': True, 'disp': True, 'cpcm': True, 'cds': True}.
- H 재조립으로 바뀐 값(g2): 반응 8,607개(사용 가능 8,490); d1 4,785개(> 0.05: 2,420), d2 8,201개(> 0.05: 1,324), barrier 8,607개(> 0.05: 3,789).

## 2. g2 H 재조립의 constrained dipole 기준 (P20, 확정)
- ADD3 10절은 명령서 Q-2 그대로(고정 이면각 상태를 키에 붙여 같은 상태끼리만 묶음, `--g2-q-keys`)를 적고, 그 상태가 D1(DFT) 기준 구조에서 정해지므로 g2 H에 DFT 정보가 일부 들어간다는 점(P20)을 이 보완판에서 확정·기록한다고 했다.
- **결정 (2026-10-23, 사용자 규칙 '모든 판단 = 추천안'; `inputs.py` P20, commit 04c9f7e9): 선택지 (c), `inputs.py build g2 --g2-cons-own`.** constrained dipole 키에 속한 g2 행은 기준을 바꾸지 않는다(반응마다 자기 G2 기준 구조의 HF 에너지). 나머지 키는 명령서 4-1대로 키마다 g2 기준 구조 HF 에너지의 최저값을 쓴다(문턱 없음, 동점은 job_id가 작은 것).
- **이유:** Q-2 키의 `|fixed:` 접미사(고정 이면각 상태)는 D1(DFT) 기준 구조에서 계산된다. 이것으로 g2 기준을 묶으면 DFT에서만 얻는 정보가 g2 H에 들어간다(명령서 4-1: g2가 DFT 파일을 하나라도 읽으면 DFT 정보가 새는 것). 배포 때(SMILES → xTB)는 이 상태를 알 수 없으므로 반응마다 자기 G2 기준을 쓰는 것이 배포와 같다. 쓰지 않은 선택지: (a) 명령서 Q-2 키 그대로(DFT 유래 상태 사용), (b) g2만 SMILES 수준 키로 묶기(명령서에 없는 새 키).
- **한계 (이탈로 기록):** constrained 키에는 constrained 행만 들어 있으므로 `|fixed:` 상태 자체는 g2 값에 들어가지 않는다. 그러나 어느 g2 행을 그대로 둘지는 `$LAD/Q_keys.parquet`의 행별 `constrained` 표시로 고르고, 이 표시는 D1 라벨의 `ref_port.constrained`(D1, 곧 DFT 흐름의 기준 구조 결과)다. G2 자신의 기준 구조 결과(G2 job의 `ref_result.json` `port.constrained`)는 commit된 입력이 아니어서 읽지 않았다. 따라서 'constrained인가'라는 DFT 유래 정보 한 가지가 행 선택에 남는다: D1에서는 자유인데 G2에서는 constrained인 행은 자유 키의 최저값으로 바뀌고, 반대 경우는 자기 기준을 유지한다. 두 표시가 다른 g2 행의 수는 세지 않았다(G2 job 폴더를 읽어야 함).
- 기록(`P_build_g2.json`): `g2_q_keys` = `own`, 지문 `g2_q_keys` = `own`, `references.g2_mode` = `own`. constrained 키의 기준 행 3,631개 중 키 최저값이 더 낮았지만 자기 기준을 유지한 조각 3,186개(키 최저값과의 차 중앙값 1.508, 최대 10.49 kcal/mol), constrained 조각 교체 0개. 바뀐 조각 12,986개(반응 8,607개; dipolarophile 8,201, dipole 4,785), 바뀐 양 중앙값 0.020, 최대 77.30 kcal/mol; 분자 키 963개. 기준 구조 그래프 검사(P6b): True.
- 이 결정은 g2 입력표를 만들기 전에 정했다(g2 학습·평가 결과 없음).

## 3. D1-G2 Phase F 멈춤의 수용 (이탈)
- `analysis/d1_g2/results/F_report.json`(sha256 `4e384eb95850fc55db1733cc99296dd1945b6d49f05a8df20d8d76191bbac08c`): G2 ok 8,616반응 중 전역 182 8,501, Q 사용 8,499, 토큰 8,499. gate 실패 117/8,616 = 1.36 % > 1 % → stop (D1-G2 명령서 14절 F; 구성 feature 115, Q 제외 2, T 실패 0).
- 결정 `analysis/d1_g2/results/F_DECISION.md`(sha256 `1d2095c94c34424356dfa2911e159317ff98df589eda322c2243025068d869ab`, commit 881cdade) 원문: “결정: 사용자 지시("모든 판단은 너의 추천대로 진행", 2026-10-22·23)에 따라, 실패한 117반응을 g2 rung에서 제외하고 나머지 8,499반응의 feature·토큰으로 진행한다. 이탈로 기록한다(D1LADDER 2단계 입력의 사용 가능 반응은 7절 규칙대로 토큰 + 182 + H가 모두 있는 반응).”
- 명령서 16절 '2단계: D1-G2 F·H gate 실패 → 그쪽 해결까지 대기'에서 벗어난다. D1-G2의 결정으로 해결된 것으로 보고 진행하며 이탈로 기록한다. 실패 반응은 g2에서 토큰·182가 없어 사용 가능 반응이 아니다(명령서 7절). 그 반응이 들어간 짝은 P_g2에서 빠지고, 4절 적용 범위에서 따로 센다(`no_tokens_or_182`).
- g2 입력표는 `inputs.py build g2 --accept-g2f-stop --g2-cons-own`(commit 11aeef90·04c9f7e9; `jobs/p_stage2.sh`, cdbaa3ec)으로 만들었다. `--accept-g2f-stop`은 멈춤 이유가 F gate 비율뿐이고 `F_DECISION.md`가 commit되어 있을 때만 이 멈춤을 통과시키고 `P_build_g2.json` `g2_gates.F_stop_accepted`에 그 파일의 sha256(`1d2095c94c34424356dfa2911e159317ff98df589eda322c2243025068d869ab`)을 남긴다. 이 명령은 그 값이 지금 파일의 sha256과 같음을 확인했다(`F_DECISION.md`는 build 지문에 들어 있지 않음).
- D1-G2 `H_report.json`: complete True, stop False; `labels_d1_hf_g2.parquet` sha256 = H_report outputs의 값(경로 키 `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g2/analysis/d1_g2/results/labels_d1_hf_g2.parquet`, 저장소 상대 경로로 맞춤, inputs.py P21). D1-G2 파일 10개 commit됨(HEAD와 같음). `origin/d1-g2`가 HEAD에 들어 있음: True.
- `inputs.py build g2`의 D1-G2 확인 기록(`P_build_g2.json` `g2_gates`): `{"F_report.json": {"complete": true, "stop": true, "sha256": "4e384eb95850fc55db1733cc99296dd1945b6d49f05a8df20d8d76191bbac08c"}, "F_stop_accepted": {"stop_reasons": ["F gate failure rate 117/8616 = 0.0136 > 0.01 (spec 14 F)"], "decision": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_g2/results/F_DECISION.md", "decision_sha256": "1d2095c94c34424356dfa2911e159317ff98df589eda322c2243025068d869ab"}, "H_report.json": {"complete": true, "stop": false, "sha256": "9826c33d90ec892e56e59f0350f04cf915668c436432f9396d21d0f8cf251f40"}, "H_report_output_key": ["/gpfs/home1/yeseo1e`.

## 4. 짝 집합과 적용 범위
| 집합 | 짝 | 기준 | anchor | para | P_main | P_main para | 시험 짝 fold 0–4 | 빠진 짝 (T1에 쓰인 것) | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| P_dft | 8,325 | 7,192 | 1,133 | 2,116 | 6,874 | 1,759 | 1,709 / 1,667 / 1,444 / 1,357 / 1,685 | 64 (14) | `c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5` |
| P_g1 | 8,092 | 6,956 | 1,136 | 2,057 | 6,884 | 1,764 | 1,713 / 1,670 / 1,446 / 1,357 / 1,688 | 297 (1) | `ba08a648d0c0068a2027bafdb213d8e3eed671f2894ff23ef7f50dfef1413156` |
| P_01 | 8,079 | 6,946 | 1,133 | 2,052 | 6,874 | 1,759 | 1,709 / 1,667 / 1,444 / 1,357 / 1,685 | 310 (14) | `f81b99d363bae44104a92e7907a436b8beab6bf8c87204ba6dd2c4e8966ea5be` |
| P_g2 | 7,458 | 6,494 | 964 | 1,916 | 6,401 | 1,658 | 1,590 / 1,460 / 1,344 / 1,300 / 1,583 | 931 (652) | `3a475899e9a634f5325c444221d8649f60b18759973defd3fd397a4dbff21240` |
| P_012 | 7,413 | 6,452 | 961 | 1,905 | 6,394 | 1,655 | 1,586 / 1,458 / 1,343 / 1,300 / 1,581 | 976 (662) | `ed3f7a24368443fb35369b052fce7776496b58687b71e776b5c22308b92615bf` |

- 판정(13-2)은 g2 자기 집합 `P_g2`, rung 사이 비교(13-4)는 `P_012` = P_01 ∩ P_g2에서 한다(ADD3 4절·5-4).
- 학습 코드 규칙의 짝 목록 sha256(지문 `pair_list`와 비교할 값): P_g2 `ab84b94f18128176…` (7,386짝).
- **적용 범위 (명령서 12절, 정의 학습 전 고정):** T1 fold 0–4 시험 목록(FU2 split 파일 `test`)의 짝 중 P_main 짝(`S_folds.csv` `in_P_main`) 6,885짝(para 1,764) 가운데 그 집합에 든 비율. g2에서 실패한 반응은 배포 때도 예측할 수 없으므로 P_g2 밖의 짝은 '예측 불가'로 세고, g2 성적(13-2)에는 넣지 않는다(빈 예측을 null 등으로 채우지 않음).

| 집합 | P_main 시험 짝 | 비율 | para | para 비율 | fold 0–4 |
|---|---|---|---|---|---|
| P_dft | 6,874 | 99.8 % | 1,759 | 99.7 % | 1,473/1,476 / 1,408/1,411 / 1,323/1,325 / 1,301/1,301 / 1,369/1,372 |
| P_g1 | 6,884 | 100.0 % | 1,764 | 100.0 % | 1,475/1,476 / 1,411/1,411 / 1,325/1,325 / 1,301/1,301 / 1,372/1,372 |
| P_01 | 6,874 | 99.8 % | 1,759 | 99.7 % | 1,473/1,476 / 1,408/1,411 / 1,323/1,325 / 1,301/1,301 / 1,369/1,372 |
| P_g2 | 6,401 | 93.0 % | 1,658 | 94.0 % | 1,364/1,476 / 1,224/1,411 / 1,229/1,325 / 1,264/1,301 / 1,320/1,372 |
| P_012 | 6,394 | 92.9 % | 1,655 | 93.8 % | 1,362/1,476 / 1,222/1,411 / 1,228/1,325 / 1,264/1,301 / 1,318/1,372 |

- P_g2에 없는 P_main 시험 짝 484개(para 106)의 이유(두 반응 중 먼저 걸리는 것): G2 TS 없음(D1-G2 G 실패) 454, 대상 반응 아님 0, 토큰·182 없음(D1-G2 F gate 실패, 3절) 30, HF 9값 없음(D1-G2 H) 0, 두 반응 모두 사용 가능(불일치) 0.

## 5. 같은 TS에 따른 나눔 (E2, 설명용)
- 명령서 12절: g2는 G2가 라벨과 다른 TS를 찾은 짝도 그대로 학습·평가한다(배포 때 구별할 수 없다). 나눔은 설명용이고 13-2 판정, 13-3 해석, REP 고르기에 쓰지 않는다.
- **나눔 규칙 (학습 전 고정):** 반응별 판정 = D1-G2 Phase C `analysis/d1_g2/results/C_compare.csv`(sha256 `99a309eb5539dad05e20f4540238d97fb1743743d983cdae6806335de7b19a74`)의 `same_ts_g1`: G2 TS 대 G1 TS(라벨 TS를 xTB 표면으로 옮긴 구조; 둘 다 xTB), VALIDATION_SPEC.md 6.4 (TS comparison: heavy-atom graph isomorphism + mirror-aware Kabsch RMSD; same TS) 규칙(무거운 원자 RMSD ≤ 0.1 Å, 형성 결합 거리 차 ≤ 0.02 Å). 짝 (a, b)는 두 반응이 모두 같은 TS면 `same`, 하나만 같으면 `mixed`, 둘 다 다르면 `different`, 한쪽이라도 판정이 비면 `unclassified`.
- **이 열을 쓰는 이유:** 명령서 12절은 'D1-G2 Phase C의 "같은 TS" 판정'을 따르라고 한다. D1-G2 Phase C는 결정 C-13으로 그 판정(쓰는 곳에 'D1LADDER 12 E2 (same TS split)'가 적혀 있음)을 `same_ts_g1`로 정했다. `same_ts_dft`는 xTB·DFT 수준 차이를 넘어 비교하므로 같은 TS도 대부분 '다름'으로 판정한다(같은 TS를 xTB로 옮긴 G1 대 DFT 기준선 99/8,433 (1.17 %), G2 = DFT TS 73/8,616 (0.85 %)).
- **E2 보고:** P_g2 시험 짝(fold 0–4 합) 중 P_main 짝을 이 네 부류로 나눠, 부류마다 REP·ENS4·4 모델·LINE·null의 MAE(9 주 타깃; 전체 위치와 para)를 적는다. 부류 사이 검정이나 기준 판정은 하지 않는다.
- **두 번째 나눔:** `same_ts_dft`(G2 TS 대 라벨(DFT) TS; 2단계 작업 지시 문구에 적힌 열)도 같은 방식으로 네 부류로 나눠 함께 보고한다. 둘 다 설명용이고 어느 것도 판정에 쓰지 않는다.
- D1-G2 Phase C 비율(`C_summary.json` `overall`): G2 = DFT TS 73/8,616 (0.85 %), G2 = G1 4,521/8,059 (56.10 %).

| 반응 판정 열 | same | mixed | different | unclassified |
|---|---|---|---|---|
| `same_ts_g1` | 3,166 (840) | 2,290 (565) | 945 (253) | 0 (0) |
| `same_ts_dft` | 24 (11) | 133 (32) | 6,244 (1,615) | 0 (0) |

(P_g2의 P_main 시험 짝 6,401개, 괄호 = para 1,658개. g2 사용 가능 반응의 판정: `same_ts_g1` 같음 4,489 / 다름 3,464 / 없음 546; `same_ts_dft` 같음 73 / 다름 8,426 / 없음 0. 부류 크기는 구조만으로 정해지는 수이고 점수가 아니다.)

## 6. M2 학습 단위와 gate
- 단위 `L__g2__<arm>__<model>__f<k>` = g2 × S·SH × AW·GNN·TF·XGB × fold 0–4 = **40 unit**(원자 30 + XGB 10; ADD3 json `ladder.units.counts.stage2` = 40), GPU 학습 270회(원자 unit마다 튜닝 칸 수 + 최종 seed 5개).
- 명령: `train_xgb.py run <i> <n> --stage 2` → `train_xgb.py check --stage 2` → `train_atoms.py run <i> <n> --stage 2` → `train_atoms.py collect --stage 2 --with-xgb`(1단계 unit까지 모아 `L_predictions.parquet`).

### 6-1. 학습 규칙 — ADD3 3절 그대로
- arm·모델·grid·β 규칙·SH 학습 규칙·학습 상수·분할(T1 fold 0–4)·검증 짝(`inner_val`, L-9)·gate M-1′·캐시 지문은 ADD3 3절과 같다. unit 지문 `add3`는 ADD3 md의 sha256(바뀌지 않음)이고, 이 보완판의 sha256은 지문에 들어가지 않는다.

### 6-2. 매 실행 전 gate (학습 코드 원문, `train_atoms.py` = `train_xgb.py`; 두 파일의 sha256도 gate가 확인)

```python
rs = sorted({u["rung"] for u in us})
need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}
add3 = gate_add3(need, rs)
gp = gate_p(rs, labels)
if int(stage) == 2:
    gate_c_stage2()
# gate_add3:
ins = rec.get("inputs_sha256") or {}
lack = sorted(set(need) - set(ins))
for path, s in ins.items():
    if path not in need and _foreign_to(path, rungs):
        skipped.append(path)
        continue
    if not Path(path).is_file() or L.sha256(path) != s:
        bad.append(f"input {path} changed or missing")
# _foreign_to:
return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)
# gate_c_stage2 (MA-17):
if not (c.get("all_pass") and c.get("g2_verified")):
```

- 2단계 process: `rs` = [`g2`], `need` = 정답 파일 + `$LAD/inputs/g2.parquet`. 보완 전 json에는 g2 입력표가 없어 `lack`이 생기므로 2단계 학습은 이 보완 commit 전에는 exit 3이다(명령서 16절 A).

## 7. 1단계 규칙은 모두 그대로 적용한다
- ADD3의 1단계 규칙이 g2에 바뀌지 않고 적용된다: 3절(rung 열 분리 L-4, arm S·SH, β 규칙, SH 학습 규칙, 모델 네 개와 grid, 학습 상수, 짝·분할·검증 짝, 캐시 지문), 5절(REP·ENS4, 13-2 기준 oi·cpcm 모두 para MAE ≤ 0.13·민감도 ≥ 0.7·위양성률 ≤ 0.1, 꺾임 모집단, 단계 0·1 설명용), 6절(지표), 11절(멈춤 규칙). 결과를 본 뒤 이 규칙을 바꾸지 않는다.
- g2에 맞춰 읽는 곳(ADD3가 이미 적은 것): 13-2 판정 집합 = P_g2(json `decision_rules.stage2.set`); 13-3 해석 표 전체(a·b·c)를 1단계 dft 판정과 함께 적용; 13-4 rung 사이 비교 = P_012; E2 = 명령서 11절 지표 + P_012 비교(명령서 12절) + 적용 범위(4절) + 같은 TS 나눔(5절).
- 이 보완판이 새로 정한 것은 2절(P20), 3절(F 이탈 수용), 4절의 적용 범위 정의, 5절의 나눔 규칙뿐이다. 모두 g2 학습 전, g2 점수 없이 정했다.
- E2는 시작할 때 이 보완판이 commit되어 있고 sha256이 json `supplement.md_sha256`과 같은지 확인해야 한다(`prereg.py check --stage 2`가 같은 확인과 학습 코드 gate의 내용 조건을 확인한다).

## 8. Phase C 결과 (g2 포함)
`analysis/d1_ladder/results/C_tests.json` sha256 `d1e31dd3d70bbc1463293c6385c138286d721f2cb16867d83a64a1a7eb8ee069` (commit 83c8cb35), tests `d1-ladder-c1`, merge 2026-10-23 21:32:37: all_pass True, 범위 "rungs dft, g1, g2", stage_verified 2, g2_verified True, 정답 `$LAD/F_pairs_cr2.csv`.

| L-1 | L-2 | L-3 | L-4 | L-5 | L-6 | L-7 | L-8 | L-9 | L-10 | L-11 | smoke |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |

- g2 하위 결과: L-1/gpu pass, L-1/cpu pass, L-2/gpu pass, L-2/cpu pass, L-4/gpu pass, L-4/cpu pass, L-5/gpu pass, L-5/cpu pass, L-7/gpu pass, L-8/gpu pass, L-8/cpu pass, L-9/gpu pass, L-9/cpu pass, L-11/gpu pass, L-11/cpu pass.
- 검사 job: gpu job 1041251 (n001.hpc, git 5faac8f6); cpu job 1041385 (n107.hpc, git 5faac8f6).

## 9. 계획과 다르게 한 점·해석 (2단계)
1. **P20:** ADD3 10절은 명령서 Q-2 그대로(`--g2-q-keys`)를 적었고 확정을 이 보완판에 미뤘다. 선택지 (c)로 확정했다(2절). 한계: 그대로 둘 행은 D1 라벨의 `ref_port.constrained`(DFT 흐름) 표시로 고른다(2절 '한계').
2. **D1-G2 Phase F 멈춤 수용:** 명령서 16절 '2단계 대기'에서 벗어난다(3절, D1-G2 `F_DECISION.md`).
3. **같은 TS 판정 열:** 명령서 12절이 따르라고 한 D1-G2 Phase C 판정(C-13)대로 `same_ts_g1`을 나눔 규칙으로 두었다. 2단계 작업 지시 문구는 `same_ts_dft`였으나 이 열은 수준 차이 때문에 같은 TS도 거의 모두 '다름'으로 판정하므로 두 번째 설명용 나눔으로 함께 보고한다(5절).
4. **Phase C 먼저 commit:** 1단계(명령서 8절, ADD3 PR-4)와 같이 g2를 포함한 `C_tests.json`과 Phase C 코드 5개가 commit된 상태에서만 이 보완판을 썼다.

## 10. 멈춤 규칙 (명령서 16절, 2단계)
- **A:** 이 보완판·json 추가분 commit 전 2단계 학습 시도(`gate_add3`가 exit 3: json에 g2 입력표가 없거나 json이 HEAD와 다름).
- **2단계:** D1-G2 F·H gate 실패 → 그쪽 해결까지 대기(F의 1.36 % 멈춤은 3절의 이탈로 수용; 다른 F·H 멈춤은 대기).
- **M·E·원칙:** ADD3 11절 그대로(gate M-1′, 캐시 지문, 시험 짝 목록, 결과를 본 뒤 규칙 변경 금지).

