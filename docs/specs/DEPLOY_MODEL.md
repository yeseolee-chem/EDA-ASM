# DEPLOY-MODEL — 라벨 없는 g2 입력으로 배포용 TF 모델을 처음부터 만들고 학습 — Claude Code 실행 지시서

**저장소:** `origin/geo2b` HEAD(`0314ad9`)에서 새 브랜치 **`deploy`**를 만든다. 새 worktree
`/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy` (`git worktree add ... -b deploy origin/geo2b`).
**이 파일:** `docs/specs/DEPLOY_MODEL.md`로 commit한다.
**새 코드:** `analysis/deploy/` · **결과:** `analysis/deploy/results/` · **scratch:** `$DEP=/gpfs/tmp_cpu2/yeseo1ee/deploy`
**읽기만 하는 것:** `$G2=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`,
`$D1PAIR_SCRATCH`, D1 run(`/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`), `analysis/` 아래 기존 코드 전부.
**기존 코드는 고치지 않는다.** 필요한 함수는 import해서 부른다. 고쳐야 하는 동작은 `analysis/deploy/`의 새 파일에 둔다.

**순서:** 0 → L(라벨 없는 마무리) → M(메타데이터) → F(182열) → T(토큰) → Q(재현 보고) → C(코드 검사) → R(사전 등록)
→ K(교차검증 학습) → E(평가) → W(최종 모델) → I(추론기·연쇄 검사) → S(요약 후 **멈춤**)
- 16절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출 외에 아무것도 하지 않는다.

---

## 0. 무엇을, 왜

### 0-1. 목표

배포 경로(SMILES → xTB → 8채널)에서 **학습 입력과 배포 입력을 같은 코드로 만든다.** 지금 있는 g2 모델(LADDER
`L__g2__S__TF__f*`)은 입력을 만들 때 DFT 라벨에 기댄 곳이 있어 새 반응에 그대로 쓸 수 없다.

| 라벨 의존 | 위치 | 이 명령서에서 |
|---|---|---|
| TS · 반응물 원자 순서를 라벨 구조에 맞춰 재매김 | `analysis/d1_g2/g2_autode.py` `finalize()` | 재매김 없음 (autodE 순서 그대로) |
| 형성결합 쌍 = 라벨 `formed_pairs_ts` | 같은 곳 | autodE 실행 결과 `ts_result.json`의 `formed_pairs_ts` |
| 조각 구분 = 라벨 `eda.inp`의 (1) 원자 | `lab_A_idx()`, `prep_g.py` | `ts_result.json`의 `A_idx` |
| 특성 메타데이터 = 라벨 레코드 복사 | `prep_g.py` → `labels_d1_g2.json` | G2 결과로 만든 레코드 (DFT 값은 NaN) |
| 조각 재검출이 안 맞으면 DFT TS에서 원자 대응 차용 | `ESPLEY_PARTITION=label` (`r6_partition.label_partition`) | `redetect`만 사용. 실패하면 탈락 |
| S arm도 HF 9값이 있어야 "사용 가능" | `analysis/d1_ladder/train_atoms.py` `rung_base` | 182열 + 토큰만 요구 |

### 0-2. 확정된 설계 (사용자 결정 2026-11-03)

| 항목 | 값 |
|---|---|
| 구조 | g2 = SMILES → autodE 1.4.5 + `XTB2 ALPB(water)`, **반응당 1회** (G2c 3회 아님) |
| 입력 arm | **S** — 원자 토큰 + 전역 182열. HF 없음 |
| 모델 | **TF 단독** (FU2 M-2 TF, 변형 A+G) |
| 학습 데이터 | **D1만** (D0 없음 — LADDER g2와 비교 가능하게) |
| 라벨 | `$LAD/F_pairs_cr2.csv` (LADDER와 같은 파일, 같은 y) |
| 분할 | FU2 T1 5 fold (LADDER와 같은 파일) |
| 배포 모델 | **전체 D1으로 한 번 학습** (설정 = 교차검증 선택의 최빈값, epoch = 최적 epoch 중앙값; GEO1b Y 모델 선례), seed 5개 |

### 0-3. TS 탐색을 다시 하지 않는 이유

`g2_autode.keep_small()`이 반응마다 `$G2/jobs/<job_id>/`에 작업 폴더 최상위의 `.xyz` · `.json` 파일을 남겼다:
`ts.xyz`(autodE 원자 순서), `ts_result.json`(`formed_pairs_ts`, `A_idx`, 허수 진동수, 판정), `ref_dipole.xyz`,
`ref_dipolarophile.xyz`, `.done_*` · `.fail_*` 표지. **LADDER g2가 쓴 TS와 같은 구조**이고 원자 순서만 다르다(L-2 검사로
확인). 그래서 같은 TS에서 마무리만 라벨 없이 다시 한다. 진동수만 autodE 순서 구조에서 새로 계산한다(반응당 수 초).

### 0-4. 기대 결과와 비교 기준

LADDER g2 · S · TF (D1 T1 시험, para 짝 1,658개; 이번 대화에서 원자료로 계산):

| 타깃 | oi | cpcm | pauli | elst | d1 | d2 | disp | cds | barrier |
|---|---|---|---|---|---|---|---|---|---|
| para MAE (kcal/mol) | 1.014 | 0.299 | 1.735 | 0.879 | 0.374 | 0.298 | 0.131 | 0.027 | 0.327 |
| 연속 g AUC (para 89계열) | 0.845 | 0.787 | 0.775 | 0.757 | 0.916 | 0.686 | — | 0.466 | 0.834 |

입력이 대부분 같으므로(같은 TS, 같은 특성 코드) 성능은 비슷해야 한다. 크게 나빠지면 버그로 본다(G-C).

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | `origin/geo2b` = `0314ad9` 이후이고 `analysis/d1_g2`, `analysis/d1_pair`, `analysis/d1_ladder`, `analysis/espley_xtb_repro`, `analysis/geo2/deploy`가 있다 |
| 0-2 | `$G2/jobs/<job_id>/`가 D1 대상 9,266개 모두에 있고, `analysis/d1_g2/results/G_status.csv`의 ok 8,616개는 `ts.xyz`, `ts_result.json`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`를 가진다 |
| 0-3 | `$G2/g2_final/<rxn_id>/`(LADDER g2 입력, L-2 대조용)와 `$G2/X182_d1_g2.parquet`, `$G2/fu2/tokens`가 있다 |
| 0-4 | `$LAD/F_pairs_cr2.csv`의 sha256 = `$LAD/units/L__g2__S__TF__f0.json` 지문의 라벨 sha256 |
| 0-5 | FU2 T1 분할 `$D1PAIR_SCRATCH/fu2/splits/T1__f{0..4}.json`의 sha256 = `analysis/d1_pair/results/fu2/S_splits.json` |
| 0-6 | `$LAD/units/L__g2__S__TF__f{0..4}.{json,parquet}`, `__s{0..4}.pt`가 있다 (C-3 회귀 검사용) |
| 0-7 | D1 manifest `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv`의 sha256 = `g2_common.MANIFEST_SHA` |
| 0-8 | 디스크 여유 30 GB 이상 |

폴더: `sbatch --time=48:00:00 --wrap "mkdir -p $DEP/{logs,g2lf_final,F,fu2_lf,units,model,smoke,prof_empty}"`.
`$DEP/prof_empty`는 **빈 폴더로 둔다** — 특성 코드가 DFT 구조를 읽으려 하면 실패하도록.

---

## 2. 공통 모듈 `analysis/deploy/dep_common.py`

- 경로: `DEP`, `G2`, `G2_JOBS`, `G2_FINAL`, `LAD`, `LF_FINAL = DEP/"g2lf_final"`, `LABELS_LF = DEP/"labels_lf.json"`,
  `META_LF = DEP/"input_meta_lf.csv"`, `X182_LF = DEP/"X182_lf.parquet"`, `TOK_LF = DEP/"fu2_lf"/"tokens"`, `RES`.
  모두 환경 변수로 덮어쓸 수 있게(테스트 전용) 하고 기본값은 위 경로.
- 상수: `RUNG = "lf"`, `ARM = "S"`, `MODEL = "TF"`, `TARGETS` = `fu2_common.TARGETS`(d1 d2 elst pauli oi disp cpcm cds c_ghost barrier),
  `ELEMENTS = (1, 6, 7, 8, 9, 14, 16, 17, 35)`, `TOKEN_VERSION = "fu2-tokens-3-lf"`, `ATOM_VERSION = "fu2-atoms-3-lf"`.
- 도우미: 원자적 쓰기(임시 파일 → `os.replace`), sha256, STOP(exit 3), not ready(exit 2), json 쓰기.
- 쓰기 가드: `$DEP`와 `analysis/deploy/results` 밖에 쓰려 하면 STOP.

---

## 3. Phase L — 라벨 없는 마무리 (`analysis/deploy/finalize_lf.py`)

### 3-1. 함수 `finalize_lf(job_row, src_dir, out_root, cores)`

**배포 때도 이 함수를 그대로 쓴다.** 입력은 G2 작업 폴더(학습: `$G2/jobs/<job_id>/`, 배포: 새 autodE 작업 폴더).

1. 앞 단계 표지: `ts`, `stereo`, `ref` 중 `.fail_*`이 있으면 `.fail_<stage>_<사유>`로 끝낸다 (`g2_autode.finalize`와 같은 사유 문자열).
2. 읽기: `ts.xyz`(autodE 순서), `ts_result.json`, `ref_dipole.xyz` → `rel1.xyz`, `ref_dipolarophile.xyz` → `rel2.xyz`. **재매김 없음.**
3. 형성결합 쌍 `fp = ts_result["formed_pairs_ts"]`, 조각 `A = ts_result["A_idx"]` (0-기반, TS 번호).
4. 진동수: ORCA `! XTB2 ALPB(water) Freq`, `%pal nprocs {cores} end`, `%maxcore 3000`, 좌표 = `ts.xyz` 그대로(최적화 없음),
   전하 = manifest `charge`, 다중도 1. **`g2_autode`의 `FREQ_HEADER`와 D1 run `orca_direct`의 `write_inp`, `run`,
   `terminated`, `imag_and_mode`, `last_fspe`를 import해서** 쓴다.
5. `forming_share = run_ts.forming_share(X, mode, fp, heavy)` — **자기 형성결합 쌍으로**.
6. 실패: ORCA 비정상 종료 → `.fail_freq_not_terminated`; 허수 진동수 없음 → `.fail_hess_no_imaginary_frequency`.
7. 출력 `out_root/<rxn_id>/`: `ts.xyz`, `rel1.xyz`, `rel2.xyz`, `ts.hess`, `work/ts.hess`, `work/ts.inp`, `work/ts.out`, `result.json`,
   `.done` 또는 `.fail_<사유>`. 임시 폴더 → `os.replace`. 이미 `.done`/`.fail_*`이면 건너뜀(멱등).
8. `result.json` 키: `rxn_id`, `job_id`, `status`, `fail_reasons`, `formed_pairs_ts`, `A_idx`, `imag_freqs_cm`, `n_imag`,
   `imag_mode_forming_share`, `d_form` (형성결합 거리), `e_ts_freq_eh`, `remapped: false`, `source_dir`, 입력 파일 sha256,
   `freq_header`, `core_h`. (B6 블록이 `imag_freqs_cm[0]`과 `imag_mode_forming_share`를 대조한다.)

### 3-2. 명령

```
python finalize_lf.py run <i|@I> <n> <cores>   # D1 대상 [i::n], $G2/jobs → $DEP/g2lf_final
python finalize_lf.py report                   # results/L_report.json, results/L_status.csv
```
`jobs/cpu.slurm`(D1 run env, `G2ENV=d1`과 같은 환경). 배열 `--array=0-7%8`, 작업당 반응 병렬 `cpus/4`.

### 3-3. 관문

| # | 검사 | 실패 시 |
|---|---|---|
| L-1 | `.done` 수 = G2 ok 8,616 − (진동수 실패), 진동수 실패 ≤ 0.5 % | STOP |
| L-2 | 모든 `.done` 반응에서 `$G2/g2_final/<rxn_id>/ts.xyz`(라벨 순서)와 새 `ts.xyz`가 **같은 (원소, 좌표) 집합**(순열 관계, 좌표 차 ≤ 1e-6 Å). rel1 · rel2도 같음 | STOP (작업 폴더가 LADDER가 쓴 구조가 아님) |
| L-3 | 순열로 맞춘 뒤 형성결합 쌍 · `A_idx`가 LADDER `$G2/g2_final/<rxn_id>/result.json`의 `formed_pairs_label`과 라벨 `A_idx`에 대응 | 다른 반응 수 보고 (STOP 아님 — Q에서 원인 분석) |
| L-4 | 새 허수 진동수 최저값과 LADDER `$G2/g2_final` `result.json`의 값 차 | 분포 보고 (같은 구조이므로 ≈ 0이어야 함; 1 cm⁻¹ 넘는 반응이 1 %를 넘으면 STOP) |

---

## 4. Phase M — 라벨 없는 메타데이터 (`analysis/deploy/meta_lf.py`)

### 4-1. `labels_lf.json` (반응당 레코드 1개)

| 키 | 값 |
|---|---|
| `rxn_id` | manifest |
| `role1`, `role2` | `"dipole"`, `"dipolarophile"` |
| `charge1`, `charge2` | `0`, `0` (manifest에서 반응물 전하를 읽고, 0이 아니면 STOP — 학습 데이터는 모두 중성) |
| `n_atoms` | `ts.xyz` 원자 수 |
| `n_f1` | `len(A_idx)` |
| `n_f2` | `n_atoms − n_f1` |
| `formed_pairs_ts` | `"i-j k-l"` (TS 0-기반, 공백 구분; `labels_all.json`과 같은 형식) |
| `flag_async`, `flag_foreign_bond` | `ts_result.json` checks |
| `status` | `"ok"` |
| `alt_used` | `ref_result.json`의 `alt_used` |
| **DFT 값 키** — `d1_kcal`, `d2_kcal`, `e_bond_kcal`, `e_ab_eh`, `e_frag1_dist_eh`, `e_frag2_dist_eh`, `e_frag1_rel_eh`, `e_frag2_rel_eh`, `elst_dft`, `pauli_dft`, `oi_dft`, `disp_dft`, `cpcm_dft`, `cds_dft` | **NaN** |

DFT 값 키를 NaN으로 넣는 이유: `xtb_slice.process_one`이 이 키들을 필수로 읽는다(302–311행; 행의 `dft_*` 비특성 열과
`flag_negative_strain`에만 쓰임). C-1 검사로 특성 · 토큰에 영향이 없음을 증명한다.

`.done`이 아닌 반응은 레코드를 만들지 않는다. 레코드가 있는 반응 = `$DEP/g2lf_final`의 `.done` 반응.

### 4-2. `input_meta_lf.csv`

`rxn_id`, `A_idx`(공백 구분, 0-기반). `result.json`의 `A_idx` 그대로.

### 4-3. 산출 · 보고

`$DEP/labels_lf.json`, `$DEP/input_meta_lf.csv`, `results/M_report.json`(레코드 수, sha256, 전하 ≠ 0 수 = 0).

---

## 5. Phase F — 전역 특성 182열 (`analysis/deploy/feat_lf.py`)

### 5-1. 왜 새 실행기가 필요한가

`analysis/d1_pair/d1_features.py`의 `_rootrun`(389행)과 `fu2/atoms.py`의 `_env_root`(162행)는
**`xtb_slice.PARTITION_MODE != "label"`이면 STOP**한다. 또 `_env_root`는 `d1_g1.g_env()`(`ESPLEY_PARTITION=label` 포함)와
환경을 비교하고, `d1_features.assert_root_write_ok`는 `ROOT_WRITE_OK` 밖(= `$DEP`) 쓰기를 막는다. 원본을 고치지 말고,
같은 작업 함수를 부르는 얇은 실행기를 `analysis/deploy/`에 쓴다. **바꾸는 것은 세 가지뿐이다**: (1) partition mode
조건 = `redetect`, (2) `g_env()` 비교에서 예외 = `ROOT_ENV` 6개 + `ESPLEY_PARTITION`, (3) 쓰기 허용 = `$DEP`. 나머지
검사(모듈들이 같은 루트를 읽었는지, Si 항목, rev 4–6 · D1 입력 쓰기 금지 `C.assert_write_ok`)는 그대로 복사한다.

### 5-2. 환경

| 변수 | 값 |
|---|---|
| `G1_ROOT` | `$DEP/g2lf_final` (xtb_slice의 `"g1"` geom 모드가 이 루트의 `<rid>/{rel1,rel2,ts}.xyz`를 읽는다) |
| `ESPLEY_LABELS` | `$DEP/labels_lf.json` |
| `ESPLEY_META` | `$DEP/input_meta_lf.csv` |
| `ESPLEY_PROF` | `$DEP/prof_empty` (빈 폴더) |
| `ESPLEY_PARTITION` | **`redetect`** |
| `ESPLEY_OUT` | `$DEP/F/xtb` |
| `R5_SCRATCH` | `$DEP/F/r5` (ext 캐시) |
| `D1PAIR_SI_PATCH` | `1` (`site_si/sitecustomize.py`가 Si 항목을 넣는다; `g.slurm`이 켬). 시작 때 `(fr.COV, fr.Z, R5.BONDI, R5.GFN2_VALENCE)["Si"] == (1.11, 14, 2.10, 4)` 단언 |
| 그 밖 | `d1_g1.g_env()`의 나머지 값(`ESPLEY_SMILES`, `G1_ROOT_REV4`, `G1_ROOT_R6`, `G2_ROOT`, `R6_SCRATCH`, `R4_SCRATCH`, `D1_SNAPSHOT`) 그대로 |
| `XTB_BIN` / tblite | xtb 6.7.1, tblite 0.7.0 (D1-G2 Phase F와 같은 `analysis/d1_pair/jobs/g.slurm`, `PYENV=venv5`) |

### 5-3. 계산

`d1_features.py`의 `runroot` → `_rootrun` → `run_ids`가 하는 일을 같은 순서 · 같은 함수로: 반응마다
`r6_features._proc_one`(= `xtb_slice.process_one(label, "g1")`, spawn pool) → `aggregate.derive` → `r6_features._ext`
→ 병합(`d1_features.merge_rows`) 후 `aggregate.derive` → rev 4 G1 열 순서로 reindex → `ext_features`(B1–B6, `CODE_VERSION r5-ext-3`)를
`xtb_status = ok` 행에 → `X182_lf.parquet` = `rxn_id` + 182열(`rev6_common.our_columns` 순서 = `X182_d1_g2.parquet`
열 순서; 다르면 STOP). 통과 조건: `xtb_status ok`, `ext_status ok`, 182열 모두 유한.

```
python feat_lf.py slice <i|@I> <n>     # 반응 [i::n] → $DEP/F/slices/
python feat_lf.py merge                # → $DEP/X182_lf.parquet, results/F_report.json
```

### 5-4. 관문

| # | 검사 | 실패 시 |
|---|---|---|
| F-1 | 열 이름 · 순서 = `X182_d1_g2.parquet` | STOP |
| F-2 | `partition:*` 또는 `partition_mismatch_g1` 상태 반응 수 = **재검출 실패** | 비율 보고 → G-A |
| F-3 | 특성 실패율 (조각 외 사유) ≤ 2 % (D1-G2: 1.3 %) | STOP |
| F-4 | 반응당 wall · core 시간 기록 (배포 예산용) | — |

---

## 6. Phase T — 원자 토큰 (`analysis/deploy/tokens_lf.py`)

`fu2/atoms.py`의 per-reaction 계산(원자별 xTB 7회: rel1, rel2, fA, fB, TS 단일점 + TS `--json`, `--vfukui`)과 토큰 생성
(`fu2/tokens.py`)을 import해서, 5-2의 환경으로 부른다. root 검사의 partition mode 조건만 `redetect`로 둔다.

| 항목 | 값 |
|---|---|
| 출력 | `$DEP/fu2_lf/atoms/<rxn_id>.npz`, `$DEP/fu2_lf/tokens/`, `results/fu2_lf/{Q_report, Q_manifest, T_report, T_manifest}` |
| 버전 | `FU2_ATOM_VERSION=fu2-atoms-3-lf`, `FU2_TOKEN_VERSION=fu2-tokens-3-lf` |
| 전역 토큰 | `FU2_X182=$DEP/X182_lf.parquet` |
| 원소 | `T_report.json` elements = `[1, 6, 7, 8, 9, 14, 16, 17, 35]` 아니면 STOP |
| 열 | `bin_cols` = frag_B, dip_a, dip_mid, dip_b, dph_a, dph_b; `num_cols` = bond_dist, q_rel, q_frag, q_TS, dq_dist, dq_int, v_rel, v_frag, v_TS, dv_dist, dv_int, mu_abs, f_plus, f_minus; `d_in = 29`. g2 `T_report.json`과 다르면 STOP |
| 관문 | Q-a (30열 재현 ≤ 1e-6, 실패 ≤ 0.5 %), Q-b (조각 전하 합 0 ± 1e-3), T-a (토큰 수 = 원자 수, NaN 없음, 형성 간선 2개) — FU2 그대로 |

```
python tokens_lf.py atoms <i|@I> <n> ; python tokens_lf.py gate ; python tokens_lf.py tokens
```

---

## 7. Phase Q — 재현 보고 (`analysis/deploy/parity.py`) — 학습 전, 판정 대상 아님 (G-A만)

LADDER g2 입력(라벨 의존 경로)과 lf 입력(라벨 없는 경로)을 **같은 반응**에서 비교한다. L-2의 순열로 원자를 맞춘다.

| 산출 | 내용 |
|---|---|
| `results/Q_parity.csv` | 반응마다: 두 경로 모두 성공 여부, 182열 최대 \|차\|(열 이름 포함), 토큰 최대 \|차\|(순열 후), 재검출 상태, LADDER `map_source` 추정 |
| `results/Q_parity.json` | 요약: 공통 반응 수, 182열이 1e-6 안에서 같은 반응 수, 다른 반응의 열 분포, 재검출 실패 수와 그 반응들의 dipole · 골격 분포 |

**해석 규칙(미리 적음)**: 같은 TS · 같은 코드이므로 대부분의 반응은 182열과 토큰이 1e-6 안에서 같아야 한다. 차이가
나는 반응은 (a) LADDER에서 DFT TS 원자 대응을 빌린 반응, (b) 진동수 재계산의 수치 차(B6 3열), (c) 버그 중 하나다.
(c)가 의심되면(예: B6 외 열이 무작위로 다름) STOP하고 보고한다.

**G-A (사전 등록 관문)**: 재검출 실패 반응 ≤ **2 %** of L의 `.done` 반응. 넘으면 STOP — 배포에서 그만큼의 반응을 잃는다는
뜻이므로 사용자 판단이 필요하다.

---

## 8. 사용 가능 반응 · 짝 (`analysis/deploy/data_lf.py`)

### 8-1. 사용 가능 규칙

반응 사용 가능 ⇔ lf 토큰 있음(T-a 통과) **그리고** `X182_lf`에 행이 있고 182열 모두 유한. **HF는 요구하지 않는다.**
짝 사용 가능 ⇔ a와 b가 모두 사용 가능. 이 규칙으로 걸러낸 짝 집합 = `P_lf`.

### 8-2. `DeployData(FT.Data)`

`fu2/train.py`의 `Data` 하위 클래스. **`train_atoms.LadData`가 S arm에서 만드는 속성과 같은 속성을 같은 방식으로**
만든다(`onehot`, `bin`, `num`, `pos`, `mask`, `etype`, `frag`, `n`, `glob`, `y`, `ra`, `rb`, `P`, `d_in`, `d_glob`,
`static()` 상속, `split(k)`). 차이는 사용 가능 규칙(H 없음)과 토큰 · 182 경로뿐이다.

- 라벨 = `$LAD/F_pairs_cr2.csv`의 `dd_<t>` (10 타깃).
- 분할 = FU2 T1 파일. `split(k)` → `train`(바깥 학습 T_main), `test`(P_sens ∪ A_main), `inner_train`, `inner_val`(첫 내부
  `GroupKFold(4)` fold). 사용 불가 짝은 네 목록에서 모두 뺀다(LADDER 규칙).
- 누출 검사(`LadData.split`과 같음): 시험 짝이 학습에 없음, 내부 fold가 학습을 분할함, 시험 반응이 학습 반응에 없음.
- `split("all")` (W 단계 전용): 사용 가능한 `in_T_main` 짝 전부(fold −1 core 포함), 시험 · 검증 목록 없음.

---

## 9. Phase C — 코드 검사 (`analysis/deploy/tests.py`) — 하나라도 실패하면 STOP

| # | 검사 | 기준 |
|---|---|---|
| C-1 | **DFT 값 비유입**: D1 반응 20개를 (a) 진짜 DFT 값이 든 라벨 레코드 (b) NaN 레코드로 각각 특성 · 토큰 계산 | 182열과 토큰이 비트 단위로 같음 |
| C-2 | **데이터 회귀**: `DeployData`를 LADDER g2 입력(`$G2/fu2/tokens`, `X182_d1_g2.parquet`)에 **LADDER 사용 가능 규칙(H 포함)**으로 만들어 `TA.LadData("g2", "S")`와 비교 | 모든 배열 `np.array_equal`, 짝 목록 · 분할 목록 sha256 같음 |
| C-3 | **학습 회귀**: C-2 데이터로 `train_tf.py`의 fold 0 학습(튜닝 4칸 + seed 5개)을 **LADDER unit과 같은 GPU 종류(`L_report.json` 기록: NVIDIA GeForce RTX 3090)**에서 실행 → `L__g2__S__TF__f0`(칸 3층 × 128, 최적 epoch 70)과 비교 | (i) 선택 칸 같음 (ii) 최적 epoch 차 ≤ 2 (iii) 시험 para MAE가 타깃마다 ±2 % 안. 최대 \|Δyhat\| 기록 |
| C-4 | 반대칭 (gate M-1′): 모든 튜닝 · 최종 fit, 무작위 시험 짝 200개 | \|g(a,b)+g(b,a)\| ≤ 1e-4, \|g(a,a)\| ≤ 1e-4 kcal/mol (float64) |
| C-5 | 표준화 누출: 토큰 수치열 · 전역 토큰 · y 통계를 만든 반응 = 그 fit의 학습 반응만 | 시험 쪽 0개 (assert) |
| C-6 | 입력 금지: σ, 계열 · core 번호, 치환기 이름이 모델 입력 텐서에 없음 | 코드 검사 + 텐서 열 목록 |
| C-7 | `finalize_lf` 합성 검사: 가짜 작업 폴더(성공 3, 각 단계 실패 3)로 표지 · 멱등 · 원자적 쓰기 | 모두 통과 |
| C-8 | smoke: fold 0, 칸 최소 · 최대, 2 epoch, seed 1개 | 끝까지 돌고 C-4 통과 |

C-3가 학습 규칙의 정확한 재현을 보장한다. 비트 수준 일치는 요구하지 않는다(GPU 비결정성). 결과는
`results/C_tests.json`(`all_pass`).

---

## 10. Phase R — 사전 등록 (`results/PREREG_DEPLOY.md` + `prereg_deploy.json`)

K 학습 전에 commit한다. 학습 명령은 이 두 파일이 git에 있고, 변경이 없고, json의 `md_sha256` = md의 sha256일 때만 돈다.

**md에 적을 것**
1. 0-2절 설계와 8절 사용 가능 규칙
2. 학습 규칙(11-2), 최종 모델 규칙(13절 W), 평가 지표(12절)
3. 관문 G-A(7절), G-C(아래)와 그 기준값
4. **지금까지 본 것**: LADDER g2 · S · TF 결과(0-4절 표), GEO2b 결과, Q 재현 보고. lf로 학습한 결과는 0개
5. 입력 파일 sha256: `labels_lf.json`, `input_meta_lf.csv`, `X182_lf.parquet`, `T_manifest.csv`, `F_pairs_cr2.csv`, T1 분할 5개

**G-C (비열등, 사전 등록 관문)** — 기준값은 사용자가 commit 전에 바꿀 수 있다
- 대상: lf · S · TF 교차검증 예측 대 LADDER `L__g2__S__TF` 예측, **두 쪽 모두 예측이 있는 para P_main 짝**
- 9개 주요 타깃(d1 d2 elst pauli oi disp cpcm cds barrier)마다 **상대 차이 (MAE_lf − MAE_g2) / MAE_g2의 core 부트스트랩
  95 % 상한 ≤ +5 %**
- 하나라도 넘으면 STOP — W(최종 모델)를 만들지 않고 보고

---

## 11. Phase K — 교차검증 학습 (`analysis/deploy/train_tf.py`)

### 11-1. 단위

`D__lf__S__TF__f{0..4}` — `$DEP/units/<unit>.{json,parquet}`, `__s{0..4}.pt`. 각 `.pt`에 모델 state dict + 그 fit의
표준화 통계(std dict) + 칸 · `d_in` · `d_glob` · `TARGETS`를 저장.

### 11-2. 학습 규칙 (FU2 M-1 · LADDER S arm 그대로; `fu2/train.py`의 `fit`, `Std`, `gate_m1`, `predict`, `batch`, `epoch_batches`를 import해서 수정 없이 호출)

| 항목 | 값 |
|---|---|
| 모델 | `Mo.build("TF", cell, d_in=29, n_out=10, d_glob=182, "A+G")` |
| grid | 층 ∈ {2, 3} × 폭 ∈ {64, 128} (4칸) |
| 튜닝 | `inner_train`으로 학습, `inner_val`로 선택. 지표 = 주요 9개 타깃 MAE / SD(y_train) 평균. 동률은 grid 순서 |
| 최종 (fold) | 바깥 학습 짝 전부, 튜닝 최적 epoch에서 멈춤(같은 cosine 일정), seed 20261019 + 0..4, 예측 평균 |
| 표준화 | 토큰 수치열 · 전역 토큰 = fit 학습 반응의 평균 · SD. y / SD(y_train) (타깃별, ddof 0, 평균 제거 없음) |
| 손실 | 타깃별 평균 \|ŷ − y\| (표준화) + 0.1 × 평균 \|ŷ_barrier − Σ₉ ŷ\| / s_barrier |
| 최적화 | AdamW lr 1e-3, wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64 짝, patience 30, 튜닝 seed 20261019 |
| 증강 | epoch마다 짝마다 (a, b)/(b, a) 순서 무작위, y 부호 함께 |
| 관문 | 모든 fit에 C-4 |

### 11-3. 명령 · 자원

```
python train_tf.py smoke                 # = C-8
python train_tf.py run <i|@I> <n>        # 단위 [i::n], PREREG 관문 통과해야 실행, 멱등(지문 비교)
python train_tf.py collect               # → results/K_predictions.parquet, K_val.csv, K_report.json
```
GPU: `env_d1fu2`(torch 2.2.1 + cu118). RTX 3090이 있는 분할을 우선. LADDER g2 · S · TF 실측 wall: 884 / 881 / 736 /
752 / 465 s (f0–f4, 합 ≈ 1.0 GPU-h) → K ≈ 1 GPU-h, C-3 ≈ 0.25 GPU-h, W ≈ 0.2 GPU-h. 배열 `--array=0-4%5`.

`K_predictions.parquet` 열 = LADDER `L_predictions.parquet`와 같음: `pair_id, ref_kind, pos, fold, rung("lf"), arm, model,
target, y, yhat, base(0), beta(0), yhat_seed_sd`.

---

## 12. Phase E — 평가 (`analysis/deploy/eval_tf.py`)

| 산출 | 내용 |
|---|---|
| `E_metrics.csv` | lf · S · TF의 타깃 × 위치(all, p, m, o, direct) × 집합(P_lf 자기 집합, 공통 집합) → n, MAE, RMSE, para MAE core 부트스트랩 95 % CI(2,000회, seed 20261019) |
| `E_compare.csv` | 공통 para P_main 짝에서 lf 대 LADDER g2 · S · TF: MAE 차, 상대 차와 CI (`analysis/geo2/eval_geo2.py`의 `boot_diff`와 같은 core 부트스트랩) → **G-C 판정** |
| `E_bends.csv` | para 계열(점 ≥ 12, `hammett_eval.family_series`)마다 타깃별: DFT 꺾임 판정, 예측 판정, AICc 이득 g(예측), ψ (`hammett_eval.calls`, `bent_gain`) |
| `E_bend_rates.json` | 타깃별 민감도 · 위양성률(+CI), **연속 g AUC**(DFT 판정이 정답, 예측 g가 점수; 계열 부트스트랩 2,000회 CI), 숨은 꺾임 계열(장벽 직선 & 채널 ≥ 1 꺾임) 민감도 · 위양성률 · 정밀도. LADDER g2 · S · TF를 같은 계열에서 나란히 |
| `E_ad.json` | **적용 범위 문턱**: 타깃별 시험 짝 `yhat_seed_sd`의 95 백분위 (배포에서 범위 밖 표시에 사용) |
| `E_report.json` | G-C 판정, 위 요약, 입력 sha256 |

meta 위치(σ_m)는 설명용으로 같은 표를 낸다(판정에 쓰지 않음).

---

## 13. Phase W — 최종 배포 모델 (`train_tf.py final`)

G-C 통과 후에만 실행한다.

| 항목 | 규칙 |
|---|---|
| 학습 짝 | `DeployData.split("all")` — 사용 가능한 `in_T_main` 짝 전부(5 fold의 시험 core 포함, fold −1 core 포함) |
| 칸 | 5 fold 선택 칸의 **최빈값**. 동률이면 fold별 튜닝 기록의 그 칸 검증 지표 평균이 가장 낮은 칸, 그것도 같으면 grid 순서 |
| epoch | 5 fold 최적 epoch의 **중앙값** (5개라 정수). `fu2/train.py` `fit`의 `best_epoch`는 1-기반 epoch 번호이고(`for ep in range(1, …)`), fold 최종 fit은 `n_epochs=best_epoch`로 그만큼 학습한다(`run_unit` 399–405행). 그래서 `n_epochs = median(best_epoch)`. GEO1b Y 모델의 `floor(median(best_ep + 1))`는 0-기반 번호에 대한 같은 규칙이다 |
| seed | 20261019 + 0..4 |
| 참고 (예상) | LADDER g2 · S · TF에 이 규칙을 대면: 칸 2층 × 64 (f1, f2), epoch 47 (70, 39, 47, 57, 25의 중앙값). lf 결과가 다르면 그대로 따른다 |
| 표준화 | 학습 짝 전체의 반응으로 통계 (각 `.pt`에 `mu, sd, gmu, gsd, sy`로 저장 — `run_unit`과 같은 형식) |
| 관문 | C-4 (무작위 학습 짝 200개로) |
| 산출 | `$DEP/model/D__lf__S__TF__all__s{0..4}.pt`, `$DEP/model/MODEL_CARD.json`, `results/W_report.json` |

`MODEL_CARD.json`: 칸, epoch, seed, 학습 짝 수 · sha256, 토큰 버전 · 원소 · 열 목록, 182열 이름, `s_y`(타깃별), 표준화
통계 sha256, 코드 sha256(`fu2/train.py`, `models.py`, 이 디렉터리 파일 전부), torch · CUDA · GPU, 교차검증 성능(E 요약),
적용 범위 문턱(`E_ad.json`), 적용 범위 조건(원소 9종, 중성, TS 원자 8–63, D1 dipole 5종, 위치 p · m).

최종 모델에는 시험 집합이 없다. **성능 추정치는 K의 교차검증 값을 인용한다.** 참고로 학습 짝 MAE만 보고한다.

---

## 14. Phase I — 추론기와 연쇄 검사 (`analysis/deploy/predict_tf.py`, `g2_deploy.py`)

### 14-1. `predict_tf.py`

```
python predict_tf.py run <tokens_root> <x182.parquet> <pairs.csv> <out.parquet> [--model final|cv]
```
- `--model final`: `D__lf__S__TF__all__s{0..4}.pt` 5개 평균, `yhat_sd` = 5개 SD
- `--model cv`: fold 단위 25개 (검증용)
- 각 `.pt`의 표준화 통계로 입력을 표준화(새 입력의 통계로 다시 계산하지 않음), 출력 × `s_y`
- 출력 열: `pair_id, a_rxn, b_rxn, target, yhat, yhat_sd, out_of_range`(타깃별 `yhat_sd` > `E_ad` 문턱)
- 입력 검사: 원소 ⊂ 9종, 토큰 열 = `MODEL_CARD`, 182열 이름 · 순서 = `MODEL_CARD`, 모두 유한. 아니면 그 짝 제외 + 사유 기록

### 14-2. `g2_deploy.py` — 새 반응용 TS 탐색 (배포 실행 명세에서 쓸 코드)

```
python g2_deploy.py one <job_id> <cores> --manifest <manifest.csv> --out <root>
python g2_deploy.py pool <list.csv> <n_parallel> <cores> --manifest <manifest.csv> --out <root>
```
- `g2_autode.py`의 `g2_configure`, `stage_ts`, `stage_stereo`, `stage_ref`를 import해서 같은 순서로 부르고, 마지막에
  `finalize_lf`. `stage_stereo`는 `--manifest`로 받은 파일을 `stereo_check.py`에 넘긴다(D1 manifest 고정 경로 쓰지 않음).
- manifest 열 = D1 manifest 형식(`job_id, kind=D1, ts_id, rxn_smiles, mapped_smiles, product_smiles, dipole_smiles,
  dipolarophile_smiles, charge, mult`). **`mapped_smiles` 필수.**
- `g2_autode.cmd_one` · `cmd_pool`은 라벨 목록(`G.targets()`)과 `G.G2_FINAL`을 쓰므로 부르지 않는다. 같은 흐름(노드 로컬
  `$TMPDIR` 작업 폴더 → 세 단계 → `finalize_lf` → `keep_small`)을 `g2_deploy.py`에 다시 쓴다.
- 경로: `g2_common`은 `D1G2_SCRATCH`를 import 때 읽어 `G2`(쓰기 허용 루트 `assert_path`, 템플릿 폴더 `G2/ts_templates`)를
  정한다. **import 전에 `D1G2_SCRATCH=<root>`로 설정**해서 모든 쓰기가 `<root>` 안에 머물게 한다. 시작 때 `G.G2 == <root>`
  단언.
- TS 템플릿: 처음 한 번 `$G2/ts_templates`를 `<root>/ts_templates`로 복사해서 시작(기본값; 배포 실행 명세에서 바꿀 수 있음).
  autodE가 새 템플릿을 저장하므로 원본 `$G2/ts_templates`는 건드리지 않는다.
- 반응당 1회, 인프라 재시도 1회(D1 규칙).

### 14-3. 검사 (`results/I_selftest.json`)

| # | 검사 | 기준 |
|---|---|---|
| I-1 | `predict_tf.py --model cv`로 K의 시험 짝을 예측 → `K_predictions.parquet`의 fold 단위 값과 비교 (fold별 5 seed 평균끼리) | ≤ 1e-4 kcal/mol |
| I-2 | `--model final`로 학습 짝 일부(1,000)를 예측 → W 학습 직후 기록한 예측과 비교 | ≤ 1e-6 (같은 GPU) |
| I-3 | **연쇄 smoke**: D1 반응 10개(dipole 5종 × 2, 기준 반응 포함 짝 5개가 되게)를 **SMILES부터** `g2_deploy.py` → M → F → T → `predict_tf.py`로 끝까지 | 모두 돌아가고 산출 형식이 맞음. autodE가 비결정적이므로 값 일치는 요구하지 않음 — 같은 TS를 다시 찾은 반응만 LADDER 구조와 비교해 보고 |

---

## 15. Phase S — 요약 `results/SUMMARY_DEPLOY_MODEL.md` (한국어) 후 멈춤

포함: L · Q의 재검출 실패율과 원인, G-A · G-C 판정, 교차검증 성능 표(lf 대 LADDER g2 · S · TF, MAE와 연속 g AUC),
최종 모델 설정(칸 · epoch · 학습 짝 수), 적용 범위 문턱, 반응당 시간(L · F · T 단계, 배포 예산용), I 검사 결과,
배포 실행 명세에서 정할 것(기준 반응 재시도 규칙, 템플릿, 열거 · 필터).

---

## 16. STOP 조건

| 단계 | 조건 |
|---|---|
| 0 | 1절 시작 조건 하나라도 실패 |
| L | L-1 · L-2 · L-4 실패 |
| M | 반응물 전하 ≠ 0 |
| F | F-1 · F-3 실패, Si 상수 불일치 |
| T | 원소 · 열 목록이 g2 `T_report`와 다름, Q-a 실패 > 0.5 %, T-a 실패 |
| Q | G-A 실패, 또는 B6 외 열의 무작위 차이(버그 의심) |
| C | 하나라도 실패 |
| R | PREREG가 git에 없거나 변경됨 |
| K | 지문 불일치 캐시, C-4 실패 |
| E | G-C 실패 → W 실행 안 함 |
| I | I-1 · I-2 실패 |

---

## 17. 실행 규칙 (저장소 `CLAUDE.md`)

- `sbatch`만, `--time=48:00:00`. 로그인 노드에 상주 프로세스 없음.
- 실행 10 / 대기 20 상한 (배열 원소마다 1개). 제출 전 `squeue -u $USER -h -r | wc -l`. 긴 흐름은 `--dependency=afterany`
  체인으로 잇는다: L 배열 → L report → M → F 배열 → F merge → T 배열 → T gate · tokens → Q → C(CPU 부분) → C-3(GPU) → (사용자
  PREREG 확인) → K 배열 → collect → E → W → I.
- 분할: CPU 단계는 `sinfo -p cpu1,cpu2`에서 빈 쪽, GPU 단계는 LADDER g2 TF unit의 GPU 종류.
- 모든 단계 멱등 · 원자적 쓰기. 48 h에 잘리면 같은 스크립트를 다시 제출.

| 단계 | 환경 |
|---|---|
| L, M, I-3의 `g2_deploy.py` | D1 run env (python 3.9.5 venv, autodE 1.4.5, ORCA 6.1.1, xtb 6.7.1, OpenMPI 4.1.8f) |
| F, T, Q | `analysis/d1_pair/jobs/g.slurm`, `PYENV=venv5` (xtb 6.7.1, tblite 0.7.0) |
| C-3, K, W, I-1 · I-2 | `env_d1fu2` (torch 2.2.1 + cu118) |
| E | `espley` 또는 `reactot` |

---

## 18. 이 명령서의 범위 밖 (다음 명령서: 배포 실행)

- 새 골격 열거, 적용 범위 필터, 원자 매핑된 반응 SMILES 생성
- 새 반응의 TS 탐색 실행(`g2_deploy.py pool`)과 기준 반응 재시도 규칙
- 계열 단위 분석(연속 g, 2단계 회귀)과 DFT-500 외부 검증
