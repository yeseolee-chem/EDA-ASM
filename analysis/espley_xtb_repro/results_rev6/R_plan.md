# R_plan — rev 6 Phase R 재시도 설정 (Phase R 실행 전 고정, 2026-10-01)

명세 `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md` §5. 이 문서는 Phase R의 어떤 계산보다 먼저 commit 한다.
여기 없는 설정을 쓰면 STOP(R-5)이다. 코드: `r6_recover.py`(사다리), `r6_partition.py`(라벨 partition),
`r6_features.py`(R-3·R-4), 잡 `r6_R_*.sh`.

## 0. 대상과 입력

- 대상: rev 4 G1 실패 400행(`results_rev4/g1_geometry_summary.csv`, status ≠ ok). 모두 라벨 ok 행이다.
  처리 순서: Espley ds3 3,510행에 든 rxn 먼저(slice 앞쪽), 그다음 나머지. 계산이 싸서 한 번에 모두 돈다.
- 추가 대상: rxn 4994(R-4, 확장 feature B1 엔진 gate만 실패).
- rev 4 입력은 읽기만 한다: `espley_xtb_g1/<rid>/`(result.json, ts.xyz, rel*.xyz, work/ts.xyz, work/ts.hess,
  work/ref*/), rev 4·5 feature parquet. 새 결과는 `espley_xtb_g1_rev6/<rid>/`와 `$R6_SCRATCH/xtb_r6/`에만 쓴다.
- 엔진: ORCA 6.1.1 `XTB2 ALPB(water)`(otool_xtb = xtb 6.7.1 edcfbbe), xtb 6.7.1 `--opt tight --alpb water --gfn 2`,
  `%pal nprocs 2`(rev 4와 같음).

## 1. gate (rev 4와 같은 값, 느슨하게 하지 않음)

`g1_geom.GATES` = 첫 허수 ≤ −40 cm⁻¹, 나머지 허수 > −50 cm⁻¹, 형성 결합 mode share ≥ 0.5, 긴 형성 결합 ≥ 1.6 Å,
짧은 형성 결합 < 3.3 Å, 외부 결합 없음(`fragmenter.audit_forming_bonds`), 참조 1·2 heavy 그래프가 DFT 참조와 동형,
OptTS 수렴, .hess가 최종 구조(평행이동 제외 1e-4 Å)·최종 Freq 블록(0.01 cm⁻¹)과 일치. 판정 코드는 `g1_geom.process`의
gate 부분을 그대로 쓴다(`r6_recover.evaluate`). 바뀌는 것은 partition 하나다(§2).

## 2. partition (R-1, rev 6 전체 규칙)

- 조각 A = 라벨 A_idx(`input_meta.csv`), B = 나머지. 모든 gate·feature가 이것을 쓴다.
- 반응물 원자 ↔ TS 원자 대응(map0/map1, 외부 결합 감사와 feature의 반응물 원자 지정에 필요):
  G1 TS에서 `fragmenter.partition`으로 다시 찾은 결과가 A_idx와 같으면 그 대응(= rev 4/5 경로, 원래 성공 행과 같은 값),
  아니면 DFT TS(라벨 ts_file + rel 파일)에서 찾은 대응(원자 순서가 같으므로 그대로 옮겨진다).
- 진단(기록만): G1 TS 재탐색 상태·A_idx 일치 여부, DFT TS 재탐색의 A_idx 일치 여부.
  **DFT TS 재탐색 결과가 라벨 A_idx와 다르면 STOP**(R-5).
- `partition_mismatch`(재탐색 ≠ A_idx)는 더는 gate가 아니다(명세 R-1). 해당 행 수를 보고한다.

## 3. 처리 순서 (행마다)

1. 참조: rev 4 참조 그래프가 바뀐 쪽만 참조 사다리(§4-E)를 돈다. 나머지는 rev 4 참조를 그대로 쓴다.
2. TS 후보 = rev 4 TS(수렴하고 .hess 검사를 통과했을 때). 최종 참조와 라벨 partition으로 모든 gate를 평가한다.
   통과하면 끝(TS 재계산 없음).
3. 실패하면 TS 사다리는 **rev 4 판정 순서로 첫 번째 TS 실패 사유**로 고른다: OptTS 미수렴·.hess 불일치 → §4-A,
   첫 허수 → §4-B, 추가 허수 → §4-C, mode share·외부 결합·형성 결합 거리 → §4-D. 사다리 단계를 차례로 돌리고,
   매 시도 뒤 모든 gate를 다시 평가해 처음 통과한 시도를 쓴다. 사다리를 바꿔 타지 않는다.
4. 사다리를 다 돌아도 실패하면 실패로 남긴다(마지막 시도의 사유를 기록).
5. 시도마다 `R2_attempts.csv`에 한 줄: rxn, 사다리, 단계, 방향, 출발 구조, ORCA header 원문, 수렴, 허수 목록, mode share,
   형성 결합 거리, 외부 결합, 참조 그래프, partition 진단, 통과 여부와 사유, e_ts.

## 4. 사다리 설정

공통 OptTS header(rev 4와 같고 MaxIter만 다름):
`! XTB2 ALPB(water) OptTS Freq` / `%geom Calc_Hess true Recalc_Hess 5 MaxIter 500 end` / `%pal nprocs 2 end`.

| 사다리 | 단계 (`recovered_by`) | 출발 구조 | 설정 |
|---|---|---|---|
| A OptTS 미수렴 (112) | a `optts_restart` | rev 4 마지막 구조(`work/ts.xyz`) | 공통 header (Recalc_Hess 5, MaxIter 500) |
| | b `exact_hess` | rev 4 마지막 구조 | `Calc_Hess true Recalc_Hess 1 MaxIter 500` (매 걸음 정확한 Hessian) |
| | c `constrained_pre` | DFT TS | 1단계 `! XTB2 ALPB(water) Opt` + `%geom Constraints {B i j C} {B k l C} end MaxIter 500 end` (라벨 형성 결합 2개를 DFT TS 거리에 고정) → 2단계 그 구조에서 공통 header |
| B 첫 허수 (160) | a `exact_hess` | DFT TS | `! XTB2 ALPB(water) OptTS Freq TightOpt` + `Calc_Hess true Recalc_Hess 1 MaxIter 500` |
| | b `displace_mode` | rev 4 G1 TS ± 변위 | rev 4 `ts.hess`의 가장 낮은 진동수 모드(0 cm⁻¹ 병진·회전 모드 제외, \|ν\| > 1e-6), 원자 최대 변위 **0.10 Å**로 +, − 두 방향, 각각 공통 header |
| | c `constrained_pre` | DFT TS | A-c와 같음 |
| C 추가 허수 (3) | a `displace_mode` | rev 4 G1 TS ± 변위 | 첫 허수를 뺀 허수 가운데 가장 음수인 모드, 0.10 Å, ± 두 방향, 공통 header |
| D mode share 46 · 외부 결합 23 · 형성 거리 1 | a `constrained_pre` | DFT TS | A-c와 같음 |
| E 참조 그래프 변화 (49 + 2) | a `alt_start` | 라벨과 같은 선택 규칙 안의 다른 Coley 반응물 파일 | 라벨이 `r{k}_alt.xyz`를 썼으면 `r{k}.xyz`, 아니면 `r{k}_alt.xyz`(있을 때). 결합 행렬이 원자 번호 그대로 라벨 참조와 같을 때만(원자 순서 보존) 쓴다. xtb `--opt tight` (rev 4와 같음) |
| | b `constrained_ref` | 라벨 참조 | xtb `--opt tight` + `--input`: `$constrain` `force constant=1.0`, `distance: i, j, auto`(라벨 참조 거리). b1 = rev 4 최적화에서 결합 여부가 바뀐 원자 쌍만; 그래도 바뀌면 b2 = 라벨 참조의 모든 결합 쌍. b1·b2 모두 `constrained_ref` |

- ± 두 방향이 모두 gate를 통과하면 e_ts가 낮은 쪽을 쓴다(둘 다 기록).
- `recovered_by`: TS를 살린 단계 이름, 참조를 살린 단계 이름을 `;`로 잇는다(예 `constrained_pre;alt_start`).
  재계산 없이 라벨 partition만으로 통과한 행은 `label_aidx`. 원래 성공 행은 `g1_rev4`.
  민감도 분석은 `constrained_ref`가 들어간 행을 뺀다.
- OptTS 시간 제한: ORCA 실행 하나당 6 시간(넘으면 그 시도 실패).

## 5. R-3·R-4 feature

- 되살린 행(`espley_xtb_g1_rev6/<rid>/.done`)에만 rev 4 `xtb_slice.process_one`(ESPLEY73 등)과 rev 5
  `ext_features.run_rxn`(코드 `r5-ext-3`, B1–B6 109열)을 그대로 돌린다. 환경: `G1_ROOT=espley_xtb_g1_rev6`,
  `ESPLEY_PARTITION=label`(§2; 기본값은 rev 4/5 동작 그대로), 캐시 `$R6_SCRATCH/ext/`.
  출력: `$R6_SCRATCH/xtb_r6/xtb_features_g1.parquet`, `xtb_features_ext_g1.parquet`(rev 4·5 parquet과 같은 열).
  원래 성공 행은 rev 4·5 parquet 값을 그대로 쓴다(재계산 안 함).
- R-4: rxn 4994만 B1 엔진 gate 허용치 1e-5 Eh(`R6_B1_TOL=4994:1e-5`, 다른 행은 1e-6 그대로),
  `recovered_by = gate_tol_user_decision`. 기하는 rev 4 G1 그대로라 다른 블록 값은 rev 5와 같아야 한다(확인).

## 6. R-5 보고와 STOP

- 사유별 표(원래 실패 / 단계별 되살림 / 남음), 되살린 TS와 원래 성공 TS의 heavy RMSD·형성 결합 차이 분포,
  행 흐름표 R_esp → R_lab → R_feat → C(사유별).
- STOP: 되살린 행에 gate 기록이 하나라도 빠짐 / DFT TS 재탐색 ≠ 라벨 A_idx / 이 문서에 없는 설정 사용.
