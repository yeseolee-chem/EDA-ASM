# AUDIT_G0 — rev 3 결과의 DFT 기하(G0) 의존성 감사 (Phase 0)

작성 2026-09-30. 대상: rev 4 worktree `eda-asm-prediction-rev4` (branch `espley-rev4`, 기준 `1ada37aa`)의
`analysis/espley_xtb_repro/`, scratch `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/`, `/gpfs/tmp_cpu2/yeseo1ee/espley_compare/`.
방법: 파일 읽기만 (계산·sbatch 없음). 경로를 따로 적지 않으면 `analysis/espley_xtb_repro/` 상대.
수치는 파일에서 직접 읽은 값만 적었다.
**줄 번호는 기준 커밋 `1ada37aa`(HEAD)의 파일 기준이다.** rev 4 작업으로 `xtb_slice.py`, `aggregate.py` 등이 병렬로 수정되고 있으므로,
해당 파일은 `git show HEAD:analysis/espley_xtb_repro/<file>`로 확인한다. scratch 파일은 2026-09-30 12시 상태 기준이다.

---

## 0. 요약

- **G0의 출처는 한 곳이다.** `xtb_slice.py:76, 170–172`가 `PROF/<rid>/`에서 라벨의 `ts_file`, `rel1_file`, `rel2_file`
  (`_alt` 선택 포함)을 그대로 읽고, 거리·xTB 단일점 5개·D3·SASA를 모두 그 기하에서 계산한다. ESPLEY46 / 54 / 73의
  feature 중 기하와 무관한 것은 `is_charged` 하나뿐이다 (46/46, 54/54, 72/73이 G0 의존, §1-1).
- **스크립트:** repo py 14개 중 G0 의존 11, 무관 2 (`fragmenter.py`, `mmp_utils.py`), 설계상 PROF 읽음 1 (`g1_geom.py`, G1 출발점).
  repo sh 12개 중 rev 3 파이프라인 10개(`s01`–`s08`) 전부 G0 의존. scratch 헬퍼 9개 중 G0 의존 3, PROF 직접 1, 무관 5.
  rev 4 입력인 `espley_rev4/reference/espley_fairness_ablation_original.py`도 G0 의존 (§1-2, §1-4).
- **결과 파일:** `results/`의 git 추적 34개 중 G0 의존 31, 혼합 1 (`espley_vs_ours_ds3_checks.json`), 내용상 기하 무관 2
  (`mmp_pairs_v2.csv`, `verify_s1s5.json`). 34개 모두 삭제 대상 (§1-3, §3-1).
- **문구:** 배포 가능 성능 또는 공정한 동일 조건 비교처럼 읽히는 곳이 SUMMARY.md 32곳, README.md 8곳, CLAUDE.md 2곳,
  그림 문구 10곳(`plot_results.py` 3, `compare_espley.py` 7)이다. 전부 §2에 file:line과 rev 4 문구를 적었다.
- **rev 4 실행 전에 막아야 할 위험 (코드, §1-6):**
  1. `aggregate.py:10`은 출력 이름이 `xtb_features.parquet`로 고정되어 있다. rev 4 G0 집계를 이대로 돌리면 Phase 2 검증 기준 파일을 덮어쓴다.
  2. `compare_espley.py:38`은 출력 경로(`espley_compare/results/`)를 기하와 상관없이 하나만 쓴다. 게다가 `:173–175`는 파일이 있으면 학습을 건너뛴다.
     그래서 rev 4 비교가 rev 3 G0 결과(`ours_per_seed_*.csv`)를 조용히 재사용하게 된다.
  3. `s01`–`s07`, `s03_smoketest`, `s08`은 main checkout으로 `cd`한다 (`s02:10`, `s03_ml_array.sh:16` 등). rev 4 worktree에서 제출해도
     다른 세션의 코드를 실행하고 결과를 rev 3 scratch에 쓴다.
  4. G1에서 틀리게 되는 disp 처리가 세 곳 있다: `evaluate_pairs.py:49`(disp 예측값 = 참값), `analyze_extra.py:101`(disp 건너뜀),
     `plot_results.py:48–50, 78–80, 114–116`(“analytic identity” 문구를 조건 없이 출력).
  5. `analyze_extra.py`와 `evaluate_pairs.py`는 parquet의 모든 행을 쓴다(위생 필터 없음). rev 4에서는 `rows_rev4.csv`를 써야 한다.
  6. `aggregate.py:45–46`의 99% 성공률 gate에 걸리면 G1 집계가 멈춘다. Phase 1은 90% 이상을 허용하므로 rev 4용으로 고쳐야 한다.
  7. `compare_espley.py`의 우리 쪽 학습·채점 행(`:183–189`)은 `have_feat`(`:85`, Espley 재채점 `:144`)와 따로 정해진다.
     G0와 G1이 같은 행에서 학습·채점되려면 공통 마스크 하나가 둘 다 정해야 한다 (§1-6).

---

## 1. G0 의존 목록

### 1-1. G0의 정의와 열 단위 의존 (`xtb_features.parquet`)

| 열 묶음 | 계산 위치 | G0 의존 |
|---|---|---|
| `dist_R_dip_ab/bc/ac`, `dist_R_dph_ab` (4) | `xtb_slice.py:208–212` — DFT 참조 구조 `rel1_file`/`rel2_file` | 예 |
| `dist_TS_*` (7) | `xtb_slice.py:213–216` — DFT TS | 예 |
| `mulliken_*`, `wbo_valence_*` (30) | `xtb_slice.py:220–238` — DFT 기하 위 xtb 단일점 5개 | 예 |
| E5 `xtb_e_barrier_kcal` … `xtb_interaction_kcal`, `xtb_e_*_eh` | `xtb_slice.py:241–248` | 예 |
| B_CH8 `b_strain_1/2`, `b_elst`, `b_pauli`, `b_oi`, `b_cpcm`, `b_cds` | `xtb_slice.py:251–259` | 예 |
| `b_disp` | `xtb_slice.py:257` — DFT TS 위 D3(BJ)/B3LYP | 예. **라벨 `dft_disp_dft`와 해석적으로 같은 양** (SUMMARY:96, MAE 2.4e-6) |
| AUX `b_elst_scc`, `b_disp_d4`, `b_axc`, `b_ct`, `gap_*`, `mu_*`, `dmu_complexation`, `dsasa_*` | `xtb_slice.py:262–277` | 예 |
| `is_charged` | `aggregate.py:27` — `charge2`에서 파생 | 아니오 |
| `dft_*` 타깃, `dft_c_ghost_kcal` | `xtb_slice.py:161–167`, `aggregate.py:26`, `refresh_targets.py:74–80` — `labels_all.json` | 아니오 (라벨은 설계상 DFT 기하 기준, rev 4에서도 불변) |
| 메타 `charge1/2`, `role1/2`, `n_atoms`, `n_f1/2`, `alt_used`, `flag_*`, `xtb_solvation` | `xtb_slice.py:153–159` | 아니오 |
| `xtb_status` | `xtb_slice.py:174–278` | 부분 — 실패 여부가 기하에 따라 달라진다 (G1은 `g1_fail:<이유>` 추가) |

- partition(`A_idx`)도 DFT 기하에서 다시 계산된다 (`xtb_slice.py:176`).
- G0에서 `b_disp`는 disp 채널 라벨 그 자체다. `e_bond`(= Σ6채널), `eint_spe`(= e_bond + c_ghost), 헤드라인 타깃 `barrier`
  (= d1 + d2 + eint_spe)는 모두 disp 라벨을 정확한 가수로 포함하므로, ESPLEY54/73에서는 세 타깃 모두에 라벨의 가수 하나가 feature로 들어간다.
  크기는 SUMMARY:138의 “제거 시 e_bond +0.02, eint_spe +0.02, barrier −0.00”이다 (barrier에서는 무시할 만하지만 누출이 없는 것은 아니다).
  ESPLEY54/73의 채널 블록 이득(SUMMARY:56)도 이 누출을 포함한다.

### 1-2. repo 스크립트 (`analysis/espley_xtb_repro/`)

판정: **G0-직접** = PROF 기하에서 feature를 계산. **G0-전이** = G0 parquet이나 그 산출물(ml_targets, predictions, ml_report)을 읽음.
**무관** = 둘 다 읽지 않음. **PROF-설계상** = PROF를 G1 출발점으로만 읽음.

| 파일 | 판정 | 근거 (file:line) | rev 4 조치 |
|---|---|---|---|
| `xtb_slice.py` | G0-직접 (G0의 정의) | `:2` docstring “on Coley DFT geometries”, `:76` PROF, `:170–172` `ts_file`/`rel1_file`/`rel2_file` 로드, `:176` partition | `--geom {dft,g1,g2}` (Phase 2). `:75` 기본 라벨 경로가 main checkout을 가리키므로 `ESPLEY_LABELS`(r4_env)로 덮어써야 한다. `:300` 비원자적 쓰기 |
| `aggregate.py` | G0-전이 | `:10` `SLICES = ROOT/"slices"`, `OUT = ROOT/"xtb_features.parquet"`, `:16–19` slice 읽기 | 입력 slice 폴더와 출력 경로를 옵션으로 (`slices_{g0,g1}`, `ESPLEY_FEAT`). `:26–27` 파생열 유지. `:45–46` 99% gate는 G1에 맞게 |
| `refresh_targets.py` | G0 파일 경유 (feature는 안 건드림, `dft_*`만 교체) | `:62–64` in/out parquet 인자, `:74–82` 라벨 교체, `:93` 쓰기 | rev 4 불필요 (§3-4) |
| `train_ml_single.py` | G0-전이 | `:33–34` `FEAT_PATH = ROOT/"xtb_features.parquet"`, `:193` 읽기, `:195–199` 행 선택(ok ∩ 위생) | `ESPLEY_FEAT`, `ESPLEY_ML_OUT`, `ESPLEY_ROWS`. `:36` import 시 `mkdir` 부작용(`compare_espley.py:35`, ablation `:15`가 import) |
| `train_ml.py` | G0-전이 | `:44` 같은 parquet, `:45–47` `ml_report.json`/`ml_table_espley.csv`/`predictions.parquet`를 같은 ROOT에 씀 | 삭제 (§3-4) |
| `aggregate_ml.py` | G0-전이 | `:10–14` `ROOT/ml_targets` → `ml_report.json`, `ml_table_espley.csv`, `predictions.parquet` | `ESPLEY_ML_OUT` |
| `plot_results.py` | G0-전이 | `:24–25` ROOT/figures, `:136–137` predictions + ml_report | `ESPLEY_ML_OUT`, 제목에 기하 표기, disp 문구 조건부 (§2-4) |
| `analyze_extra.py` | G0-전이 | `:27` `RES = HERE/"results"`, `:32–33` `results/xtb_features.parquet`, `results/predictions.parquet`, `:97` `results/ml_report.json`, `:104` X = G0 feature | `results/` 읽기 금지 → `ESPLEY_ML_OUT`/`ESPLEY_FEAT`/`ESPLEY_ROWS`, 출력 `results_rev4/`. G1에서는 disp 포함 (`:101`) |
| `evaluate_pairs.py` | G0-전이 | `:16` `RES/xtb_features.parquet`, `:21` ESPLEY73 X, `:49` disp = 참값 | 행 = `rows_rev4`. G1에서는 disp를 예측. `:19`는 `RES/../train_ml_single.py`로 찾으므로 RES가 scratch면 깨진다 → `HERE` 기준으로. 출력이 CWD(`:88–90`) |
| `compare_espley.py` | G0-전이 | `:39` `FEAT = ESPLEY_OUT/xtb_features.parquet`, `:77, :178` 읽기, `:85` `have_feat`, `:117` G0 `xtb_dist_dipole_kcal`, `:155` pre-ML | `ESPLEY_FEAT`. 출력은 기하별로 (`:38` OUT, `:41` RES → results_rev4). `:85` 마스크와 우리 쪽 학습·채점 행(`:183–189`)을 G0/G1 공통 마스크 하나로 (§1-6) |
| `rev3_doc_numbers.py` | G0-전이 | `:13–14` main checkout 경로 하드코딩, `:19` predictions, `:46–47` rev 3/rev 2 parquet, `:61–85` results CSV | 삭제 (§3-4) |
| `fragmenter.py` | 무관 | 라이브러리, 파일 경로 상수 없음 | 그대로 |
| `mmp_utils.py` | 무관 | `:29` SMILES CSV만 읽음 | 그대로 |
| `g1_geom.py` | PROF-설계상 | `:56` PROF, `:197` `pdir = PROF/str(rid)` — spec §2의 G1 출발 기하. `xtb_features.parquet`는 읽지 않음 | 해당 없음 (orchestrator 파일) |
| `s01_xtb_array.sh` | G0-직접 | `:29` `CODE=/home1/.../eda-asm-prediction/...`, `:30–31` `xtb_slice.py`, 출력 `espley_xtb/slices/` | r4_ 래퍼로 교체 후 삭제 |
| `s02_aggregate.sh` | G0-전이 | `:10–11` main checkout으로 cd 후 `aggregate.py` 실행 | 교체 후 삭제 |
| `s03_ml_array.sh` | G0-전이 | `:16–18` `train_ml_single.py` | 교체 후 삭제 |
| `s03_ml.sh` | G0-전이 | `:12–13` `train_ml.py` | 삭제 |
| `s03_smoketest.sh` | G0-전이 | `:21` `espley_xtb/xtb_features.parquet` 하드코딩 | `ESPLEY_FEAT`를 읽는 r4 smoke로 교체 후 삭제 |
| `s04_aggregate_ml.sh`, `s05_plot.sh`, `s06_analyze_extra.sh`, `s07_evaluate_pairs.sh` | G0-전이 | 각 `:10` main checkout cd, `s07:11` `evaluate_pairs.py results …` | 교체 후 삭제 |
| `s08_publish_rev3.sh` | G0-전이 | `:34` G0 parquet·ML 산출물을 `results/`로 복사, `:41, :44` s06/s07, `:50` rev 2 백업 | 삭제 |
| `r4_env.sh`, `r4_g1_smoke.sh` | 무관 / PROF-설계상 (`g1_geom.py smoke`) | — | orchestrator 파일 |

### 1-3. 결과 파일 (`results/`, git 추적 34개)

| 파일 | 만든 곳 | 판정 | 근거 |
|---|---|---|---|
| `xtb_features.parquet` | `s08_publish_rev3.sh:34` (scratch 사본) | G0 | §1-1 |
| `ml_report.json`, `ml_table_espley.csv`, `predictions.parquet` | `aggregate_ml.py` → `s08:34` | G0 | `train_ml_single.py:34` |
| `ml_targets/target_00…11_*.json` (12) | `train_ml_single.py:226–227` → `s08:36` | G0 | 같음 |
| `figures/scatter_{Ridge,KRR_rbf,SVR_rbf,XGB}_ESPLEY73.png`, `figures/mae_bar_espley73.png` | `plot_results.py:83, 124` → `s08:37` | G0 | `plot_results.py:136–137` |
| `figures/espley_vs_ours_ds3.png` | `compare_espley.py:264` | G0 | `compare_espley.py:39` |
| `espley_vs_ours_ds3.csv`, `_all_models.csv`, `_per_seed.csv` | `compare_espley.py:283–285` | G0 (우리 행). Espley 행도 G0 ok 행으로 재채점 (`:144`) | `:85, :144` |
| `espley_vs_ours_ds3_checks.json` | `compare_espley.py:286–288` | 혼합. 기하 무관: `label_agreement`, `split_replicated`, `d1d2_index_vs_role`의 `n_swapped`·`mae_*`·`am1_same_index_r` 0.746·`am1_swapped_index_r` 0.355. G0 의존: `pre_ml_xtb_mae`, `swapped_rows_corr_xtbdipole_vs_t1/t2` | `:113–120, :152–156` |
| `charge_breakdown.csv` | `analyze_extra.py:49` | G0 | predictions |
| `group_split.csv` | `analyze_extra.py:127` | G0 | `:104` G0 X, `:103` G0 튜닝 HP |
| `split_metrics.csv`, `margin_calibration.csv` | `evaluate_pairs.py:88–89` → `s08:45` | G0 | `:16, :21` |
| `mmp_pairs_v2.csv` | `evaluate_pairs.py:90` | 내용은 기하 무관 (SMILES 쌍, `mmp_utils.build_pairs`). 행 집합만 parquet 행을 따른다 | `mmp_utils.py:29, 96–116` |
| `rev3_vs_rev2.csv` | `s08:48–53` | G0 (rev 3, rev 2 모두 같은 G0 feature) | `s08:50` |
| `verify_s1s5.json` | scratch `verify_s1s5.sh:76` | 내용은 라벨만 (`dft_*`, flag). CPCM 시기 라벨이라 낡음 (S1 gap −3.39) | `verify_s1s5.sh:14–75` |
| `SUMMARY.md` | 수기 | G0 (대부분의 수치). S1·S5 라벨 통계, Espley 라벨 일치, Bath 아카이브 사실은 기하 무관 | §2-1 |

### 1-4. scratch 스크립트

| 파일 | 판정 | 근거 | 권고 |
|---|---|---|---|
| `espley_xtb/publish_rev3.sh` | G0-전이 | `s08_publish_rev3.sh`와 첫 주석 한 줄 빼고 동일 (diff 확인) | 삭제 |
| `espley_xtb/rev3_doc_numbers.py` | G0-전이 | repo `rev3_doc_numbers.py`와 동일 (diff 확인) | 삭제 |
| `espley_xtb/verify_s1s5.sh` | parquet 경유, 라벨 열만 | `:13–14` main checkout `results/xtb_features.parquet`, `:76` main checkout `results/verify_s1s5.json`에 씀 | 삭제 (다시 돌리면 다른 checkout에 쓴다) |
| `espley_xtb/preflight.sh` | PROF 직접 (tblite 시절 점검, 산출물 없음) | `:30–31` `PROF/0/r0_*.xyz`, `:12` tblite | 삭제 |
| `espley_xtb/preflight2.sh` | 무관 (tblite API 탐색) | `:10–37` | 삭제 |
| `espley_compare/diag_frag.py` | 무관 | `:7–12` Espley pkl, `labels_all.json`, `input_meta.csv`만 | 유지 (d1/d2 인덱스 진단, Phase 4-2 참고용) |
| `espley_compare/stage1_inspect.py` | 무관 | `:7–20` Espley pkl만 | 유지 |
| `espley_compare/machine_learning/ml_analysis.py`, `test_ranges.py` | 무관 (Espley 원본) | 우리 파일 참조 없음 | 유지 (다운로드 입력) |
| `espley_rev4/reference/espley_fairness_ablation_original.py` | G0-전이 | `:27` `results/xtb_features.parquet`, `:52–56` arm B의 DFT 거리 11개, `:60` arm O의 ESPLEY46, `:15` `train_ml_single` import | Phase 4-3에서 repo로 옮길 때 G0 parquet(arm B)과 G1 parquet(arm B_g1)을 env로 읽게. `results/` 읽기 금지 |

### 1-5. 의존 그래프

```
PROF (Coley DFT TS + 참조, _alt 포함)
  └─ xtb_slice.py ─ s01 ─> espley_xtb/slices/slice_XX.parquet            (G0 feature, 09-14; dft_* = CPCM 시기 라벨)
       └─ aggregate.py ─ s02 ─> espley_xtb/xtb_features.parquet          (09-28 refresh_targets.py로 dft_*만 rev 3 라벨로 교체)
            ├─ train_ml_single.py ─ s03_ml_array ─> ml_targets/ ─ aggregate_ml.py ─ s04 ─> ml_report.json, ml_table_espley.csv, predictions.parquet
            │     ├─ plot_results.py ─ s05 ─> figures/*.png
            │     ├─ analyze_extra.py ─ s06 ─> charge_breakdown.csv, group_split.csv   (feature도 직접 읽음)
            │     ├─ rev3_doc_numbers.py ─> SUMMARY 수치
            │     └─ s08 ─> results/ 사본, rev3_vs_rev2.csv
            ├─ evaluate_pairs.py ─ s07 ─> split_metrics.csv, margin_calibration.csv, mmp_pairs_v2.csv
            ├─ compare_espley.py ─> espley_compare/results/{prep.pkl, *_per_seed*.csv} ─> results/espley_vs_ours_ds3*
            ├─ espley_fairness_ablation_original.py (rev 4 입력)
            ├─ s03_smoketest.sh (스키마 점검)
            └─ verify_s1s5.sh (라벨 열만)
```

### 1-6. rev 4 코드가 지켜야 할 것 (위 목록에서 나온 것)

- **Phase 2 `--geom dft` 검증 기준.**
  - 기준 파일은 `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features.parquet`다 (09-28 19:55, 3,736,126 B, `results/` 사본과 같은 크기).
    feature는 09-14 계산분이고 (`slices/` mtime 09-14 17:03), `dft_*`는 09-28 `refresh_targets.py`로 교체됐다.
  - `slices/`를 `dft_*` 비교 기준으로 쓰지 않는다. 09-14 당시(CPCM 시기) 라벨이 들어 있다.
  - rev 3 라벨 커밋 `780e90aa`(09-28 21:23) 이후 라벨 변경은 `cc56a652`(09-29) 하나뿐이다. 제외 5건의 사유 필드만 바뀌었고,
    커밋 메시지에 “every label value unchanged”라고 적혀 있다.
  - **가정 (git으로 확인 불가):** 기준 parquet의 `dft_*`는 09-28 19:55에 `refresh_targets.py`가 그 시점 디스크의 `labels_all.json`에서 썼다
    (`espley_xtb/logs/refresh.995369.out`, parquet mtime 19:55:04). 라벨이 커밋된 것은 88분 뒤(`780e90aa`)다.
    그 사이 디스크 파일이 바뀌지 않았다고, 즉 19:55의 라벨 = `780e90aa`라고 가정한다.
  - 그래서 **Phase 2 STOP gate는 feature 열(+ 아래 문자열·불리언 메타 열)만 1e-8로 판정한다** (spec Phase 2-1도 “feature 값”만 요구한다).
    `dft_*` 일치는 별도의 report-only 점검으로 보고한다. 여기서 어긋나면 원인은 라벨 쪽이고 `--geom dft` 동작과는 무관하므로 STOP 사유가 아니다.
  - 행은 `rxn_id`로 맞춘다. `xtb_slice.py:285–287`은 정렬한 뒤 연속 구간으로 자른다(18 × 293).
    `dft_c_ghost_kcal`, `is_charged`는 집계 후에만 생기므로 slice 비교에서 뺀다.
    문자열·불리언 열(`role*`, `alt_used`, `xtb_status`, `xtb_solvation`, `flag_*`)은 정확히 같아야 한다.
  - 결정성: xtb는 1 스레드(`xtb_slice.py:109`, `s01:20–21`)로 돌렸으니 rev 4도 `XTB_THREADS=1`로 둔다.
    `dftd3`·`morfeus` 버전이 09-14 이후 바뀌었으면 `b_disp`, `dsasa_*`가 달라질 수 있으므로 버전을 로그에 남긴다.
- **기준 파일을 덮어쓰지 않는다.** `aggregate.py:10`은 이름이 고정되어 있다. G0 출력은 `xtb_features_g0.parquet`, slice는 `slices_g0/`로 한다.
- **Espley 비교의 행 마스크.** HEAD `compare_espley.py`에서 행은 두 갈래로 정해진다.
  - `:85` `have_feat`(우리 feature 파일의 ok 행)가 Espley 쪽 재채점 행(`:144`)과 pre-ML 기준선 행(`:154–155`)을 정한다.
  - 우리 쪽 학습·채점 행은 `have_feat`를 쓰지 않는다. `:183–189`가 arm마다 `ok = ~isnan(X_all).any(1) & ~isnan(y)`를 만들고
    `tr[ok[tr]]`, `te[ok[te]]`로 자른다.
  - 그래서 G0 실행에서 `have_feat`만 G1 ok로 바꾸면, 우리 G0 모델은 여전히 G0 ok 행 전부(G1이 실패한 행 포함)로 학습·채점된다.
    G0와 G1의 “Ours” 두 줄이 같은 test 행을 쓰지 않고, Espley 재채점 행과도 어긋난다.
  - 공통 마스크 하나(G0 사용 가능 ∩ G1 사용 가능 ∩ 라벨 ok)가 `:85`, `:144`, `:154–155`와 `:183–189`를 모두 정해야
    두 기하가 같은 행에서 학습·채점된다. rev 4 작업본(작성 시점)은 `ok`를 기하별 `have_feat`로 만들고, 교집합은 plot 단계의
    채점에서만 적용한다. 그러면 test 행은 같지만 학습 행은 기하마다 다르다. 학습 행까지 맞출지 PREREG에 적는다 (spec 3-1은 “같은 행 집합”).
  - 출력 폴더도 기하별로 나눠야 한다 (`:38` OUT 공유, `:173–175` 있으면 건너뜀).
- **행 집합.** `train_ml_single.py:195–199`의 ok ∩ 위생 필터 대신 `ESPLEY_ROWS`를 쓴다. `analyze_extra.py:32, 89–104`와
  `evaluate_pairs.py:16`은 현재 필터가 전혀 없다 (rev 3에서 5,260행 전부 사용).
- **disp.** G0에서는 `b_disp` = 라벨이다. G1에서는 disp가 진짜 예측 과제가 되므로 `evaluate_pairs.py:22–24, 49`(참값 대입),
  `analyze_extra.py:101`(건너뜀), `plot_results.py:48–50`(항등 문구)을 기하에 따라 나눈다.
- **G1 집계 gate.** `aggregate.py:20`(5,260행)은 G1 실패 행도 행으로 남기면 그대로 쓸 수 있다.
  `:45–46`(ok ≥ 99%)은 `g1_fail:*`을 뺀 행 기준으로 바꿔야 한다.
- **그 밖.** `evaluate_pairs.py:19`는 train_ml_single을 RES 기준 상대 경로로 찾는다 → `HERE` 기준으로.
  `train_ml_single.py:36`은 import할 때 옛 scratch에 `ml_targets/`를 만든다.
  `plot_results.py:131–134`는 FIG_DIR의 PNG를 모두 지운다 → 기하별 폴더를 써야 한다.

### 1-7. rev 3 코드–문서 불일치 (rev 4 재실행 때 반영)

- `group_split.csv`: SUMMARY:162는 “5,234 rxn (위생 필터 후)”라고 적었다. 그러나 `analyze_extra.py:32, 89–104`에는 행 필터가 없어
  5,260행 전부를 썼다. 또 `:118`은 `TransformedTargetRegressor` 없이 KRR을 만든다. 그런데 `:103`에서 가져오는 HP는
  y 표준화 파이프라인에서 튜닝된 값이다(`train_ml_single.py:96–100`). random 열이 헤드라인과 다른 이유는 “분할 기준”(SUMMARY:162)만이 아니다.
  rev 4에서는 `rows_rev4` + `_make_pipe`로 맞춘다.
- `split_metrics.csv`/`margin_calibration.csv`: KRR HP가 α = γ = 1e-3으로 고정되어 있다(`evaluate_pairs.py:31`, seed별 튜닝 아님).
  rev 4에서도 이렇게 할지 PREREG에 적는다.

---

## 2. rev 3 수치를 배포 가능한 성능이나 공정한 동일 조건 비교처럼 서술한 문장

공통 원칙 (spec Phase 6, spec §2 기하 정의 표):
- G0 수치에는 항상 **“G0 = DFT oracle geometry 상한 (라벨과 같은 DFT TS·참조 기하, 배포 불가)”**를 붙인다.
- **G1도 배포 성능이 아니다.** G1은 DFT 구조에서 출발한 xTB 재최적화라 conformer와 입체 선택 정보가 남아 있다.
  G1은 “Espley 대응 비교 arm”으로, 배포 성능은 G2(Phase 5)로만 부른다.
- Espley 비교에서 다른 것은 **기하 + feature + 모델**이다. “다른 것은 feature뿐”과 “같은 핸디캡”은 쓰지 않는다.
- G0의 `b_disp`는 라벨과 해석적으로 같다. G0 disp는 ML 결과로 보고하지 않는다.
  e_bond·eint_spe·barrier에도 disp 라벨이 가수로 들어가 있다는 점을 G0 주석에 함께 적는다 (§1-1, 크기는 작다).

### 2-1. `results/SUMMARY.md`

| # | 위치 | 원문 | 문제 | rev 4 문구 |
|---|---|---|---|---|
| S1 | :3 | “**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**” | Espley는 AM1로 최적화한 기하에서 feature를 뽑았다. rev 3은 라벨 DFT 기하를 썼는데 이를 “재현”이라 부르고 oracle 표기가 없다 | “Espley 2024 프로토콜(분할·모델·지표)을 GFN2-xTB/ALPB(water) feature로 적용. 헤드라인 = G1(xTB 재최적화 기하). rev 3 수치는 G0(DFT oracle geometry) 상한.” |
| S2 | :22–34 | “### Headline — `ESPLEY73` + KRR(RBF) (a priori), 5-seed 80/10/10, n = 5,234”와 표 (barrier **1.45 ± 0.05**, elst **1.52**, OI **1.16**, CPCM **1.01** …) | 헤드라인 MAE에 기하 표기가 없다. 라벨을 계산한 DFT TS·참조 구조가 입력이다 | 헤드라인은 “G1 · ESPLEY73 · KRR(RBF), n = <rows_rev4>”. 이 표는 “G0 상한 (DFT oracle geometry, 배포 불가)” 열로만 두고 G1 − G0 열을 붙인다 (spec 3-4) |
| S3 | :34, :48 | disp 행 “*0.01 (b_disp와 해석적 등식)*”, “disp \| **0.00** …” | 등식이라는 설명은 맞다. 하지만 성능 표 안에 수치로 남아 있다 | G0 disp 칸은 “— (G0: b_disp = 라벨, ML 결과 아님)”. G1 disp는 일반 예측으로 보고 |
| S4 | :36–48 | “### 4개 모델 — Protocol A, `ESPLEY73`, test MAE …” | 기하 표기 없음 | 제목에 “(G0 상한)”. G1 표는 따로 두고 부록으로 (spec 3-4) |
| S5 | :56 | “**Arm ablation (KRR):** barrier 1.68 / 1.50 / 1.45, CPCM 2.93 / 1.18 / 1.01 (46 / 54 / 73).” | 54/73의 이득에 G0의 `b_disp`(라벨)와 DFT 기하 채널 값이 섞여 있다. 채널 블록의 순수한 기여로 읽힌다 | G1 ablation으로 교체. G0 것은 “상한에서의 ablation, b_disp = 라벨 포함”으로 부록 |
| S6 | :57, :160–175 | “**그룹 분할 robustness** … random 열 대비 1.20× 이내”, :162 “… / 5,234 rxn (위생 필터 후)” | G0 feature 위의 robustness다. 또 행 수가 코드와 다르다 (§1-7) | G1 예측, `rows_rev4`로 다시 계산해 “G1 기준”으로. G0는 상한 열. 행 수는 실제 사용 행으로 |
| S7 | :58, :141–158 | “**전하별** … barrier 중성 1.44 vs 하전 1.67 …”, 표 | G0 | G1로 다시 계산 (spec 3-3). G0는 상한 |
| S8 | :59, :323–346 | “**MMP 쌍** … ΔMAE Pauli 2.54 / elst 1.79 / OI 1.39, dominant-channel 일치 0.826”, :325 “… + disp × 지배 채널 지표” | G0다. 지배 채널 판정의 disp 예측값이 참값이다 (`evaluate_pairs.py:49`) | G1로 다시 계산, disp도 예측. G0는 “상한, disp = 라벨” |
| S9 | :78 | “one GFN2-xTB / ALPB(water) single point per structure …” | 어느 구조인지 없다 | “구조 = G1(xTB/ALPB 재최적화 TS·참조) 또는 G0(라벨 DFT TS·참조, 상한)” |
| S10 | :96 | 표 “Gates (모두 통과)”의 “G4 \| **해석적 적합성** \| mean\|b_disp − dft_disp_dft\| = 2.4 × 10⁻⁶ … 사실상 등식” | 라벨과 feature가 같다는 사실을 통과한 품질 gate처럼 적었다 | gate 표에서 뺀다. “G0 누출 표지: b_disp = dft_disp_dft (MAE 2.4e-6)”. G1 gate는 반대로 MAE(b_disp − dft_disp_dft) > 0 (spec Phase 2-4) |
| S11 | :137–139 | “다른 타깃의 feature로 들어가서 도움이 되지만 (제거 시 e_bond +0.02 …)”, “ESPLEY46(b^ch 블록 없음)에서만 disp는 실제 ML 예측이며 MAE 1.25” | G0에서 `b_disp`(= 라벨)는 e_bond·eint_spe와 헤드라인 타깃 barrier의 정확한 가수다. barrier에서의 크기는 같은 줄의 “barrier −0.00”으로 무시할 만하지만, 이 서술로는 헤드라인 행이 누출 없는 것처럼 읽힌다. ESPLEY46 disp도 DFT 기하 입력이다 | “G0(ESPLEY54/73)에서는 e_bond·eint_spe·barrier에 disp 라벨이 정확한 가수로 feature에 들어간다 (제거 시 +0.02 / +0.02 / −0.00). G0 disp는 모든 arm에서 ML 결과로 보고하지 않는다 (spec 3-1). G1 disp는 세 arm 모두 예측.” |
| S12 | :191–196 | 표 머리 “이번 pre-ML GFN2/ALPB \| 이번 KRR test MAE ± sd (%range) [SVR]”, 굵은 1.12 / 0.82 / 0.77 / 1.45 | 우리 쪽 pre-ML과 ML이 모두 G0다. 표에 기하 표기가 없다 | 열 이름 “우리 G1”과 “우리 G0 (상한)”. 굵게 표시하는 쪽은 G1만 |
| S13 | :199 | “1. 출발점이 다르다. AM1 pre-ML 장벽 MAE 23.07 …, GFN2/ALPB는 6.23.” | “DFT 기하 위 xTB”와 “AM1 기하 위 AM1”을 비교해 방법 차이로 읽힌다 | G1 pre-ML로 교체. G0 값은 “DFT 기하 위 xTB 단일점(상한)” |
| S14 | :202 | “4. 같은 SVR끼리 비교해도 우리 쪽이 낮다(…). feature 수를 46개로 맞춘 `ESPLEY46` KRR의 장벽은 1.68.” | 모델·feature 수를 맞추면 공정하다는 뉘앙스다. 기하가 다르다는 말이 없다 | “G1 · ESPLEY46 · KRR 대 Espley SVR(사전 등록 주 비교): …. G0 줄은 상한.” |
| S15 | :205 | “## Espley 2024 동일 조건 비교 (ds3, 같은 test 행)” | “동일 조건”이 아니다 (기하가 다름) | “## Espley 2024 비교 — 같은 반응·타깃·분할, 다른 기하” |
| S16 | :218 | “**다른 것 (= 비교 대상):** feature. … 우리는 xTB `ESPLEY46`(개수 맞춤) / `ESPLEY73`(DFT 기하 위 단일점 → 상한).” | 다른 것이 feature뿐인 것처럼 적었다. ESPLEY46도 DFT 기하인데 73에만 “상한”을 붙였다 | “**다른 것:** (1) 기하 — Espley AM1 재최적화, 우리 G1 xTB 재최적화(DFT 출발) / G0 라벨 DFT 기하(상한). (2) feature. (3) 모델·튜닝.” ESPLEY46·73 모두 G0/G1 표기 |
| S17 | :220–228 | 표 “우리 최고 (xTB-46) \| 우리 최고 (xTB-73) \| Espley ÷ 우리(73) \| seed별 우리가 낮음” | 최고 모델끼리 G0에서 비율과 승패를 매겼다 | 주 비교는 G1 · ESPLEY46 · KRR 대 Espley SVR (spec 4-1). 최고 모델끼리 비교는 부록. G0 열은 “상한” |
| S18 | :231 | “1. **5 타깃 모두, 5 seed 모두 우리가 낮다.** feature 수를 46개로 맞춰도(ESPLEY46) 결론이 같다. 우리 쪽에서 가장 약한 조합(…2.27)도 Espley 최고(SVR 3.09)보다 낮다.” | G0에서의 우월성 주장이다. spec 0-3: Espley 46에 DFT 거리 11개만 더해도 격차의 28–100%가 사라진다 | G1 결과로 다시 쓴다. G0 결론은 “상한에서의 격차”로만 남긴다 |
| S19 | :232 | “2. 차이는 상호작용(÷3.3)과 장벽(÷2.0–2.1)에서 크고, d1/d2에서는 ÷1.15로 작다. 이유는 3번이다.” | 격차의 원인을 인덱스 혼합으로만 돌린다. 상호작용·장벽 격차의 상당 부분은 기하 몫이다 | G1 비율로 교체하고, 원인은 기하 절제(spec 4-3, `espley_geometry_ablation.csv`)와 역할 기준 비교(4-2) 결과로 쓴다 |
| S20 | :237 | “… 그래서 두 쪽이 같은 핸디캡을 지고 비교 자체는 공정하다. 다만 이 두 타깃은 물리적으로 섞인 양이라 양쪽 모두 ~2 kcal/mol에서 막힌다.” | 틀린 서술이다. Espley AM1 변형 feature는 타깃과 같은 인덱스를 따른다 (`checks.json` `am1_same_index_r` 0.746, `am1_swapped_index_r` 0.355). 그래서 인덱스 혼합의 비용이 한쪽에만 크다 (spec 4-2: 우리 1.03 → 2.45, Espley 2.60 → 2.66) | 문장 삭제 (spec 4-2). 역할 기준 d1/d2를 양쪽에서 비교한 결과를 본문에, 인덱스 기준 비교는 부록에 |
| S21 | :238 | “참고(우리만 …): … dipole 변형 1.03 (ESPLEY46 KRR) / 1.10 …, dipolarophile 0.87 / 0.80 … d1/d2 오차의 절반 이상이 인덱스 혼합에서 온다.” | G0 수치이고 “우리만”이다 | G1 수치 + Espley 역할 기준 arm (4-2 a/b)과 나란히. G0는 상한 |
| S22 | :245 | “5. **남는 비대칭.** 우리 feature는 DFT TS 기하 위 xTB 단일점이고 …, Espley는 AM1로 최적화한 기하다. 이 점에서 우리 수치는 상한이다.” | 가장 큰 차이를 “남는 비대칭”이라며 “공정하다”는 결론 뒤에 두었다 | 비교 절 첫머리로 올린다: “G0는 라벨 기하를 쓴 상한이라 Espley와 입력 정보가 같지 않다. 비교는 G1로 한다.” |
| S23 | :252 | “Feature는 **DFT TS 기하**에서의 xTB 단일점이다 — 성능 상한(upper bound). React-OT 등 생성 기하에서의 배포 성능은 별도 실험이 필요(§S2 후속).” | 경고 자체는 맞다. 다만 TS만 언급했고(참조 구조도 DFT, `_alt` 포함) 가정·주의 절에만 있다 | “G0 = DFT TS + DFT 참조(`_alt` 선택 포함) = 라벨 기하. G1 = xTB/ALPB 재최적화(DFT 출발, conformer·입체 선택 정보 남음). 배포 성능 = G2(Phase 5).” 문서 첫 절에도 둔다 |
| S24 | :272 | “`espley_vs_ours_ds3.png` — Espley 2024 동일 조건 비교 (ds3 반응·타깃·test 행 동일, 최고 모델끼리 …)” | “동일 조건”, 최고 모델끼리 | “같은 반응·타깃·test 행, 다른 기하 (우리 G1 / G0 상한); 주 비교 G1 · ESPLEY46 · KRR 대 Espley SVR” |
| S25 | :343–344 | “랜덤 분할이 성능을 낙관적으로 표시함이 정량 확인됨.”, “**LOSO(치환기 홀드아웃)는 예상보다 견고** …” | G0 예측 위의 외삽 결론이다 | G1로 다시 계산해 “G1 기준”으로. G0는 상한 |
| S26 | :372, :384–386 | “배포 시 dominant channel(…)만이 필요하고, … gate로 쓰면 신뢰 예측만 선별할 수 있다.”, “τ = 5.0에서는 42%가 통과하고 99.3%로 안전.”, “**모델 τ가 실 τ의 신뢰할 만한 대리치**” | G0 예측(그리고 참값 disp)으로 배포 gate와 “안전”을 주장했다 | “G0 상한에서의 보정 곡선.” 배포 gate 임계는 G1(가능하면 G2)으로 다시 정한다. “안전”, “배포 시” 표현은 G2 전까지 쓰지 않는다 |
| S27 | :436 | “**S2 — xTB 기하 민감도**: … `xtb --opt` 로 로컬 최적화한 기하에서 재추출 후 5개 seed 재학습, 배포 성능 확인.” | G1(DFT 출발 재최적화)을 배포 성능이라고 적었다 | “S2 → rev 4: G1(xTB 재최적화, DFT 출발) 수행. 배포 성능은 G2(Phase 5, 사용자 승인 후).” |
| S28 | :106–126 | “## (rev 2, superseded by §Rev 3) Final headline model — `ESPLEY73` + KRR(RBF) …”와 표 (barrier **1.45 ± 0.05**, Pauli **2.00** …), :126 “**Rev 1 → Rev 2 델타** …” | rev 2 기록이지만 “Final headline model”이라는 제목과 굵은 MAE가 기하 표기 없이 남아 있다 | SUMMARY 재작성 때 삭제 (git 이력 `b992d776`). 남긴다면 “rev 2 기록, G0 상한, CPCM 시기 라벨” |
| S29 | :157 | “**`is_charged` 편입 효과 (rev 1 → rev 2 기록):** charged 그룹 elst 3.00→2.40 …” | S7 범위 안이지만 전하 표와 따로 남은 rev 1→2 기록이다. G0 feature 위의 효과 크기다 | G1로 다시 확인할 때만 싣는다. 옛 수치는 “rev 1→2 기록, G0 상한” 또는 삭제 |
| S30 | :179 | “… 같은 반응·같은 타깃·같은 test 행으로 맞춘 비교는 다음 절 **§Espley 2024 동일 조건 비교**를 보라.” | S15와 같은 “동일 조건” | “… 다음 절 **§Espley 2024 비교 (같은 반응·타깃·분할, 다른 기하)**를 보라.” |
| S31 | :286 | 재현 명령 블록의 “# Espley 동일 조건 비교 (…)” | S15와 같음. 블록 전체가 rev 3 s0x 명령이다 | 블록을 r4_* 명령으로 교체, 주석 “# Espley 비교 (같은 반응·타깃·분할, 기하 G0/G1)” |
| S32 | :345 | “3. **dipole_class 분할에서 τ₉₉는 미달** — τ = 8.0에서도 agreement 0.977. 골격 외삽은 99%로는 안전하지 않음.” | S26과 같은 종류의 gate 안전 판정을 G0 예측(disp = 참값)으로 내렸다 | “G0 상한에서의 관찰.” 안전 판정은 G1(가능하면 G2)으로 다시 계산한 뒤에만 |

### 2-2. `README.md`

| # | 위치 | 원문 | rev 4 문구 |
|---|---|---|---|
| R1 | :1 | “# espley_xtb_repro (v6, results rev 3) — …” | “(v7, results rev 4 — G1 xTB geometry; G0 = DFT-geometry upper bound)” |
| R2 | :3 | “This reproduces the Espley 2024 protocol … with **AM1 replaced by GFN2-xTB** and **ALPB(water) solvation**. … Current results … [`results/SUMMARY.md`]” | “… with AM1 replaced by GFN2-xTB/ALPB(water) and the AM1-optimised geometries replaced by GFN2-xTB/ALPB-optimised ones started from the Coley DFT structures (G1, headline). The rev 3 numbers used the label DFT geometries themselves (G0, oracle upper bound, not deployable). Results: `results_rev4/SUMMARY.md`.” |
| R3 | :7 | “Each structure gets one single-point calculation …” | “Each structure (G1: xTB-optimised TS and references; G0: the label DFT TS and references) gets one single point …” |
| R4 | :14 | “**Headline arm.** **`ESPLEY73`** …” | “**Headline:** G1 · `ESPLEY73` · KRR(RBF) (pre-registered, `results_rev4/PREREG_REV4.md`).” |
| R5 | :33–35 | “`b_disp` … This is the SAME quantity as the DFT disp channel (analytic identity, MAE(b_disp − dft_disp_dft) ≈ 2e-6 kcal/mol).” | “At G0 (label DFT geometry) `b_disp` equals the disp label (analytic identity), so G0 disp is not reported as an ML result. At G1 it is an ordinary feature.” |
| R6 | :79–80 | “`compare_espley.py` like-for-like vs Espley on their ds3 rows, targets and splits” | “vs Espley on the same ds3 rows, targets and splits; the geometry differs (ours G1 xTB-optimised, G0 = upper bound; Espley AM1-optimised)” |
| R7 | :72, :78 | `refresh_targets.py …`, `s08_publish_rev3.sh copy outputs into results/ …` | 파이프라인 목록에서 삭제. r4_* 목록으로 교체 |
| R8 | :77 | “`s06/s07             charge / group-split / MMP analyses on results/`” | “`r4_downstream.sh   charge / group-split / MMP analyses per geometry (rows_rev4) -> results_rev4/downstream_<geom>/`” |

### 2-3. repo 루트 `CLAUDE.md` (rev 4 worktree)

| # | 위치 | 원문 | rev 4 문구 |
|---|---|---|---|
| C1 | :94 | “Downstream ML: `analysis/espley_xtb_repro/results/SUMMARY.md` (rev 3).” | “Downstream ML: `analysis/espley_xtb_repro/results_rev4/SUMMARY.md` (rev 4: features on GFN2-xTB-optimised geometries (G1) = headline; the rev 3 DFT-geometry numbers are kept only as the G0 upper bound; rev 3 files deleted, in git history).” |
| C2 | :111 | “`espley_xtb_repro/      Espley 2024 protocol, GFN2-xTB features (results rev 3)`” | “`espley_xtb_repro/      Espley 2024 protocol, GFN2-xTB features on xTB geometries (results_rev4; G0 = upper bound)`” |

### 2-4. 그림 문구 (`plot_results.py`, `compare_espley.py`)

| # | 위치 | 원문 | 문제 | rev 4 |
|---|---|---|---|---|
| F1 | `plot_results.py:81` | `fig.suptitle(f"{model} · {FS}  —  5-seed test-fold predictions\n{LABELS_NOTE}", …)` | 기하 표기 없음 | `ESPLEY_GEOM`에서 한 줄 추가: “G1 — GFN2-xTB/ALPB-optimised geometry” 또는 “G0 — DFT oracle geometry (upper bound, not deployable)” |
| F2 | `plot_results.py:120` | `ax.set_title(f"{FS}  ·  Per-target test MAE (Protocol A)  ·  4 models\n{LABELS_NOTE}", …)` | 같음 | 같음 |
| F3 | `plot_results.py:48–50`, 사용 `:78–80`, `:114–116` | `DISP_NOTE = "same quantity as the b_disp feature\n(analytic identity, not a prediction)"` | G0에서만 참이다. G1 그림에 찍히면 틀린 말이 된다 | `ESPLEY_GEOM == "g0"`일 때만 출력. G1에서는 disp를 다른 패널과 같게 |
| F4 | `compare_espley.py:2, :4, :16–17` | “like-for-like comparison”, “Everything that can be held equal is held equal”, “What differs: the features (…; ours: xTB ESPLEY46 / ESPLEY73 on the DFT geometries) and the model family / tuning.” | 기하 차이를 feature 문구 안에 묻었다 | “same reactions, targets and splits; differs: geometry (ours G1 xTB-optimised or G0 label DFT = upper bound; Espley AM1-optimised), features, models/tuning” |
| F5 | `compare_espley.py:14–15`, `:214` | “selection "best model" per target on both sides …”, `best = g.loc[g.groupby(["side", "target"]).mae.idxmin()]` | 주 비교가 최고 모델끼리다 | 주 비교는 사전 등록한 G1 · ESPLEY46 · KRR 대 Espley SVR (spec 4-1). 최고 모델끼리는 부록 표 |
| F6 | `compare_espley.py:194`, `:216–217` | `side=f"Ours (xTB, {arm[6:]} feat.)"`, 범례 “Ours (xTB, 46 feat.)”, “Ours (xTB, 73 feat.)” | 범례에 기하가 없다 | “Ours G1 (xTB geom., 46 feat.)”, “Ours G0 (DFT geom., upper bound, 46 feat.)” 등 |
| F7 | `compare_espley.py:233` | `"Same ds3 reactions · same DFT targets · identical 80/10/10 test rows (seeds 22/23/14/1/2)"` | 동일 조건 비교로 읽힌다 | 끝에 “· different geometry (ours G1 / G0 upper bound vs Espley AM1)” |
| F8 | `compare_espley.py:248, :251` | `f"÷{ratio}"`, “Paired by seed (same test reactions); ÷ = mean MAE ratio” (최고 73 대 Espley) | G0 최고 모델과의 비율 | seed별 짝 비교는 G1 · ESPLEY46 · KRR 대 Espley SVR. G0 줄은 회색 “upper bound” |
| F9 | `compare_espley.py:256–257` | “Differs: features — Espley 46 AM1 features on AM1-optimised geometries; ours xTB single points on the DFT geometries (an upper bound).” | “Differs: features” | “Differs: geometry and features — Espley AM1 on AM1-optimised geometries; ours GFN2-xTB on xTB-optimised geometries (G1); G0 (label DFT geometry) shown as an upper bound only.” |
| F10 | `compare_espley.py:258–260` | “… Both sides use role-based (dipole / dipolarophile) features, so both carry the same handicap on these two targets.” | 틀린 서술 (S20과 같은 근거) | 삭제. 인덱스 기준 d1/d2는 “index-based (appendix); role-based comparison in Table …” |

---

## 3. 삭제 목록 (사용자 지시 “이전 실험 결과 전부 삭제”)

rev 3 결과는 git 이력에 남는다: rev 1 `2da4e86e`, rev 2 `b992d776`, rev 3 `9ebbd721`/`8f7a5c02`/`cc56a652`/`e7cd7c7f`/`1ada37aa`.

### 3-1. repo `analysis/espley_xtb_repro/results/` — git 추적 34개, 전부 삭제

```
results/SUMMARY.md
results/xtb_features.parquet
results/ml_report.json
results/ml_table_espley.csv
results/predictions.parquet
results/ml_targets/target_00_dft_barrier_kcal.json
results/ml_targets/target_01_dft_d1_kcal.json
results/ml_targets/target_02_dft_d2_kcal.json
results/ml_targets/target_03_dft_eint_spe_kcal.json
results/ml_targets/target_04_dft_e_bond_kcal.json
results/ml_targets/target_05_dft_elst_dft.json
results/ml_targets/target_06_dft_pauli_dft.json
results/ml_targets/target_07_dft_oi_dft.json
results/ml_targets/target_08_dft_disp_dft.json
results/ml_targets/target_09_dft_cpcm_dft.json
results/ml_targets/target_10_dft_cds_dft.json
results/ml_targets/target_11_dft_c_ghost_kcal.json
results/figures/scatter_Ridge_ESPLEY73.png
results/figures/scatter_KRR_rbf_ESPLEY73.png
results/figures/scatter_SVR_rbf_ESPLEY73.png
results/figures/scatter_XGB_ESPLEY73.png
results/figures/mae_bar_espley73.png
results/figures/espley_vs_ours_ds3.png
results/espley_vs_ours_ds3.csv
results/espley_vs_ours_ds3_all_models.csv
results/espley_vs_ours_ds3_per_seed.csv
results/espley_vs_ours_ds3_checks.json
results/charge_breakdown.csv
results/group_split.csv
results/split_metrics.csv
results/margin_calibration.csv
results/mmp_pairs_v2.csv
results/rev3_vs_rev2.csv
results/verify_s1s5.json
```

SUMMARY.md에는 기하와 무관해서 rev 4 문서로 옮길 절이 있다. 삭제 전에 텍스트를 `results_rev4/SUMMARY.md`로 옮긴다
(수치는 파일에서 다시 확인):
- §S1: c_ghost 통계. 라벨만으로 다시 계산할 수 있다.
- §S5: 위생 플래그.
- Espley 절의 사실들: ds3의 DFT 수준·AM1 route line(Bath 아카이브), 라벨 일치 0.18/0.24/0.06/0.05, d1/d2 인덱스 대 역할, 분할 재현, Correction D5DD90005K.

### 3-2. scratch `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/`

| 경로 | 내용 | git에 있나 | 삭제 시점 |
|---|---|---|---|
| `ml_targets/` (target json 12 + preds parquet 12, 09-28) | rev 3 ML | json은 있음. preds는 `predictions.parquet`(git)로 합쳐져 있음 | 즉시 가능. rev 4 Phase 3 전에 지우기를 권장 (같은 ROOT 기본값에 섞이지 않게) |
| `ml_targets_v1_backup/` (json 11 + preds 11, 09-14) | rev 1 ML | json·요약만 (`2da4e86e`). preds는 없음 | 즉시 가능 (되돌릴 수 없음) |
| `rev2_backup_20260915/` (`ml_report.json`, `ml_table_espley.csv`, `predictions.parquet`, `xtb_features.parquet`, `figures/` 5, `ml_targets/` 24) | rev 2 | 결과는 `b992d776`에 있음. ml_targets preds는 없음 | 즉시 가능 |
| `ml_report.json`, `ml_table_espley.csv`, `predictions.parquet` | rev 3 | 있음 (results/ 사본) | 즉시 가능 |
| `figures/` (PNG 5) | rev 3 | 있음 | 즉시 가능 |
| `slices/slice_00…17.parquet` (09-14) | G0 feature + CPCM 시기 `dft_*` | 없음 (`--geom dft`로 다시 만들 수 있음) | **Phase 2 검증 통과 후** (불일치가 나면 slice 단위로 원인을 찾을 때 필요) |
| `xtb_features.parquet` (09-28) | Phase 2 검증 기준 | 있음 (results/ 사본) | **Phase 2 `--geom dft` 검증 통과 후에만** |
| `publish_rev3.sh`, `rev3_doc_numbers.py`, `verify_s1s5.sh`, `preflight.sh`, `preflight2.sh` | rev 3 이하 헬퍼 (§1-4) | publish_rev3·rev3_doc_numbers는 repo 사본이 git에 있음 | 즉시 가능 |
| `logs/` (165개, 177 KB) | rev 1–3 job 로그 | 없음 | 사용자 결정. rev 4 로그는 `$R4_SCRATCH/logs`로 가므로 지워도 rev 4와 무관 |

### 3-3. scratch `/gpfs/tmp_cpu2/yeseo1ee/espley_compare/`

- **삭제:** `results/` — `prep.pkl`, `espley_per_seed.csv`, `ours_per_seed_0…6.csv` (9개, G0).
  per-seed 값은 `results/espley_vs_ours_ds3_per_seed.csv`(git)에 합쳐져 있다. `prep.pkl`은 git에 없지만 내용 대부분이 `_checks.json`(git)에 있다.
  **rev 4 비교를 실행하기 전에 반드시 지우거나 출력 경로를 분리한다** (`compare_espley.py:173–175`가 있는 파일을 재사용함).
- **삭제 권고 (선택):** 최상위 rev 3 비교 로그 `stage1.996579.out`, `diag.996581.out`, `prep.9965{80,601,642}.out`,
  `train.996602_{0..6}.out`, `plot.9966{03,43,48}.out`.
- **유지 (Espley 다운로드 입력):** `feature_selection/` (`_f_selection/tt/manual_tt_solvent.pkl`), `machine_learning/` (`tt/solvent/ml_results.pkl`,
  `ml_analysis.py`, `test_ranges.py`), `hyperparameter_tuning/`, `energy_extraction/`, `feature_extraction/`.
  G0 무관 헬퍼 `diag_frag.py`, `stage1_inspect.py`도 유지한다.
- 관련 유지: `/gpfs/tmp_cpu2/yeseo1ee/espley_archive/` (Bath zip central directory, Phase 4-4 HTTP range에 필요),
  `/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/reference/` (rev 4 입력).

### 3-4. rev 3 전용이거나 대체되는 repo 스크립트

| 파일 | 권고 | 이유 |
|---|---|---|
| `s08_publish_rev3.sh` | 삭제 | rev 3 G0 산출물을 `results/`로 복사한다 (`:34–37`). main checkout 경로 하드코딩(`:16`), 지워질 `rev2_backup` 의존(`:50`) |
| `rev3_doc_numbers.py` | 삭제 | rev 3 SUMMARY 수치용. 하드코딩(`:13–14`), 지워질 파일 의존(`:19, :46–47, :61–85`). 라벨만 쓰는 부분(c_ghost 통계 `:44–53`)은 rev 4에 필요하면 `labels_all.json`에서 새로 계산 |
| `train_ml.py` | 삭제 | rev 2부터 `train_ml_single.py` + `aggregate_ml.py`로 대체. 옛 프로토콜이다: seed 23 한 번 튜닝(`:121–126`), `ESPLEY72`(`:73`), 11 타깃(`:76–77`), y 표준화 없음. 같은 ROOT의 `ml_report.json` 등을 덮어쓴다(`:45–47, :219–221`) |
| `s03_ml.sh` | 삭제 | `train_ml.py` 실행기 (`:13`) |
| `refresh_targets.py` | 삭제 권고 (남겨도 해는 없음) | rev 4에서는 `xtb_slice.py:161–167`이 두 기하 모두 현재 라벨로 `dft_*`를 직접 만드므로 필요 없다. 파생열 규칙(`:80`)이 `aggregate.py:26`과 중복된다. feature는 건드리지 않아 G0 의존은 없다 |
| `s03_smoketest.sh` | r4 판으로 교체한 뒤 삭제 | 옛 parquet 경로 하드코딩(`:21`) |
| `s01_xtb_array.sh`, `s02_aggregate.sh`, `s03_ml_array.sh`, `s04_aggregate_ml.sh`, `s05_plot.sh`, `s06_analyze_extra.sh`, `s07_evaluate_pairs.sh` | r4_* 래퍼로 교체한 뒤 Phase 6 커밋에서 삭제 | 모두 main checkout으로 cd하거나 그 경로를 CODE로 쓴다 (`s01:29`, `s02/s04/s05/s06/s07:10`, `s03_ml_array:16`). rev 4 규칙(`source "$SLURM_SUBMIT_DIR/r4_env.sh"`, 하드코딩 금지)과 맞지 않는다 |
| `xtb_slice.py`, `aggregate.py`, `train_ml_single.py`, `aggregate_ml.py`, `plot_results.py`, `analyze_extra.py`, `evaluate_pairs.py`, `compare_espley.py` | 유지·수정 | §1-2, §1-6의 경로·행·disp 조치. 공개 이름 유지 (`FEATURE_SETS`, `GRIDS`, `_make_pipe`, `split_80_10_10`) |
| `fragmenter.py`, `mmp_utils.py` | 유지 (변경 없음) | G0 무관 |

### 3-5. 삭제 순서

1. **지금 해도 되는 것:** §3-2의 `ml_targets/`, `ml_targets_v1_backup/`, `rev2_backup_20260915/`, `ml_report.json`, `ml_table_espley.csv`,
   `predictions.parquet`, `figures/`, 헬퍼 5개. §3-3의 `espley_compare/results/`와 로그.
   `espley_compare/results/`는 rev 4 `compare_espley` 실행 **전**에 반드시 지운다.
2. **Phase 2 `--geom dft` 검증 통과 후:** `espley_xtb/xtb_features.parquet`, `espley_xtb/slices/`. 검증에서 STOP이 나면 지우지 않는다.
3. **Phase 6 문서 정정 때:** repo `results/` 34개(`git rm -r`). 그 전에 §3-1의 라벨 전용 절을 `results_rev4/SUMMARY.md`로 옮긴다.
   §3-4의 스크립트도 같은 커밋에서 정리한다. rev 4 코드는 처음부터 `results/`를 읽지 않으므로 이 시점까지 남아 있어도 영향은 없다.
4. 되돌릴 수 없는 것 (git에 없음): `slices/`(다시 만들 수 있음), `ml_targets_v1_backup/`·`rev2_backup_20260915/ml_targets/`의 preds parquet,
   `espley_compare/results/prep.pkl`, `logs/`.
