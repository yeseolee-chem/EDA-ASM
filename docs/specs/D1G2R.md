# D1-G2R — 9,252 대상 g2 입력 다시 만들기 (기존 TS 재사용) + 기존 g2 삭제 — Claude Code 실행 지시서

**저장소:** `origin/gnn3-clean` HEAD(`dd36bc0`)에서 새 브랜치 **`g2r`**를 만든다. 새 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2r`.
**이 파일:** `docs/specs/D1G2R.md`로 commit한다.
**새 코드:** `analysis/g2r/` · **결과:** `analysis/g2r/results/` · **scratch:** `$G2R=/gpfs/tmp_cpu2/yeseo1ee/g2r`
**기존 g2 (D 단계 전까지 읽기만, D 단계에서 삭제):** `$G2=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `$CH3/lf=/gpfs/tmp_cpu2/yeseo1ee/ch3/lf`
**읽기만:** D1 run(`/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`), `analysis/` 아래 기존 코드 전부. **기존 코드는 고치지 않는다.** import해서 부르고, 바꿀 동작은 `analysis/g2r/`에 둔다.

**순서:** 1 시작 조건 → 2 브랜치·폴더 → C(TS 원본 복사) → X(대상) → R(기준 구조 코드 오류 수리) → L(진동수 마무리) → M(메타데이터) → F(전역 182열) → T(원자 토큰) → U(사용 가능 목록) → S(보고·commit·push) → D(기존 g2 삭제) → **멈춤**
- 14절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다(예외: 5-2절에서 미리 허용한 범위).
- 모든 계산과 삭제는 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출, git, `squeue`·`scancel`만 한다.
- 짝 재구성과 모델 학습은 이 문서에 없다. D가 끝나면 멈춘다.

---

## 0. 무엇을, 왜

### 0-1. 목표

1. 모델을 다시 학습하기 위해, 대상을 9,252개로 바꾸고 아래 결정을 반영한 g2 입력(진동수 마무리, 메타데이터, 전역 182열, 원자 토큰)을 새 폴더 `$G2R`에 만든다.
2. TS는 다시 찾지 않는다. 기존 autodE 결과(TS, 반응물, 판정)를 `$G2R/jobs`로 **복사해 와서** 쓴다.
3. 새 입력이 확인을 모두 통과하면, 혼동을 막기 위해 기존 g2(`$G2` 전체, `$CH3/lf`)를 지운다. 이후 g2는 `$G2R` 하나뿐이다.

### 0-2. 확정 결정 (사용자, 2026-11-13)

| # | 항목 | 결정 |
|---|---|---|
| D-1 | 대상 | D1 비D″ 라벨 9,266 − `flag_short_forming` 9 − `flag_regio_ambiguous` 5 (겹침 0) = **9,252**. `flag_async` 209, cr2 전 기준 초과(> 20 kcal/mol) 8, 위생 근소 음수 4는 **포함** |
| D-2 | TS 탐색 | **기존 autodE 결과 재사용.** 새 TS 탐색 없음(반응당 1회 규칙, DEPLOY 0-2) |
| D-3 | 기준 구조 단계 코드 오류 20 (RDKit `AtomValenceException` 19, `KeyError` 1) | 원인 진단 → 5-2절 허용 범위의 우회로 **기준 구조 단계만** 다시 돈다 |
| D-4 | TS 검사 탈락 16, 조각 분할 탈락 3 | **제외.** 실패한 검사 이름과 값만 기록한다 |
| D-5 | 특성·토큰 검산 허용치 | B1 engine gate `rev5_common.GATE_E_EH` 1e-6 → **1e-4 Eh**, 쌍극자 합 검산 `ext_features.TOL_DIP_AU` 1e-3 → **1e-2 a.u.** 나머지 검산은 그대로 |
| D-6 | TS 없음 487, autodE 오류 108, 입체 그래프 불일치 3, 허수 진동 없음 1 | 제외. 다시 돌리지 않는다 |
| D-7 | 기존 g2 | 새 g2가 확인을 모두 통과하고 commit·push한 **뒤에** `$G2` 전체와 `$CH3/lf`를 지운다. TS 원본은 C 단계에서 9,266개 모두 `$G2R/jobs`로 옮겨 남긴다 |

### 0-3. D-5 근거

| 검산 | 무엇을 확인하나 | 지금 | 새 값 | 예전에 이 검산으로 탈락한 값 |
|---|---|---|---|---|
| B1 engine gate | xtb 6.7.1과 tblite 0.7.0이 같은 구조에서 같은 에너지를 내는지 (같은 전자 상태) | 1e-6 Eh | 1e-4 Eh (0.063 kcal/mol) | 96건, 1.0–9.3 × 10⁻⁶ Eh |
| 쌍극자 합 | Σ q r + Σ μ = xtb 분자 쌍극자 (원자 쌍극자의 부호·좌표계·단위) | 1e-3 a.u. | 1e-2 a.u. | 특성 16건 1.0–1.4 × 10⁻³, 토큰 2건 2.0 × 10⁻³(rxn 1007289)·3.7 × 10⁻³(rxn 1007205) |

- 예전 탈락 차이는 모두 반올림 수준이다. 에너지 차이는 최대 0.006 kcal/mol로, 같은 반응을 autodE로 두 번 계산할 때의 장벽 차이(MAE 0.7 kcal/mol; Stuyver, Jorner & Coley, *Sci. Data* 2023, 10, 66, doi:10.1038/s41597-023-01977-8)보다 100배 이상 작다.
- 새 값은 예전 최대 탈락값의 약 10배(에너지), 약 3배(쌍극자)다.
- 바꾸지 않는 검산: `TOL_Q_JSON`, `TOL_EPS_EV`, `TOL_TRACE`, `TOL_FUKUI`, `TOL_HESS`, `GATE_ORTHO`, `GATE_S_BLOCK`, FU2 Q-a·Q-b·T-a.

### 0-4. 기대 수 (기존 파일로 미리 계산)

| 단계 | 빠짐 | 남음 | 출처 |
|---|---|---|---|
| D1 비D″ 라벨 | — | 9,266 | `labels_d1.json` |
| short_forming 9, regio_ambiguous 5 (겹침 0) | 14 | 9,252 | 라벨 flags |
| 기준 초과 > 20, 위생 (cr2 값, −0.5 kcal/mol 허용) | 0 | 9,252 | `labels_d1_cr2.parquet` (위생을 엄격히 ≥ 0으로 하면 4) |
| g2: TS 없음 487, autodE 오류 108, 입체 불일치 3, 허수 진동 없음 1 (D-6) | 599 | 8,653 | `analysis/d1_g2/results/G_status.csv` |
| g2: TS 검사 16, 조각 분할 3 (D-4) | 19 | 8,634 | 같음 |
| g2: 기준 구조 코드 오류 20 (D-3) | 20 − k | 8,614 + k | k = 복구 성공 수 |
| 진동수 마무리 (L) | 0 기대 | | `$CH3` L 보고: 진동수 실패 0 |
| 182열 (예전 탈락 112, D-5) | 0 기대 | | `analysis/ch3/results/lf_inputs/F_report.json` |
| 토큰 (예전 탈락 2, D-5) | 0 기대 | **8,614 + k** | `.../fu2_lf/Q_manifest.csv` |

### 0-5. 삭제 뒤의 약속

- g2 = `$G2R` 하나. TS 원본 = `$G2R/jobs`(9,266개 전부), 모델 입력 = `$G2R/lf`.
- deploy·d1_g2 코드의 기본 경로는 지워진 `$G2`를 가리킨다. 기본값으로 돌리면 경로가 없어 실패한다(조용히 옛 데이터를 쓰는 일이 없다). 앞으로의 지시서는 `DEP_G2_JOBS=$G2R/jobs`, `DEP_SCRATCH=$G2R/lf`를 넘긴다.
- 지운 뒤에는 LADDER g2 rung, D1HF@G2, GEO2·D1G3가 `$G2`에서 읽던 중간 산출물을 그대로 다시 읽을 수 없다. commit된 결과 파일은 남는다. 필요하면 `$G2R/jobs`와 각 코드로 다시 만든다.

---

## 1. 시작 조건 (하나라도 아니면 STOP) → `results/S0_start.json`

| # | 조건 |
|---|---|
| 1-1 | `origin/gnn3-clean` = `dd36bc0` 이후이고 `analysis/{d1_g2, d1_pair, d1_ladder, deploy, ch3, espley_xtb_repro}`가 있다 |
| 1-2 | `labels_d1.json` sha256 = `d1pair_common.D1_LABELS_SHA`, D1 manifest sha256 = `g2_common.MANIFEST_SHA` |
| 1-3 | `$G2/jobs/<job_id>/`가 D1 대상 9,266개 모두에 있고, 표지(`.done_*`·`.fail_*`)로 정한 상태가 `analysis/d1_g2/results/G_status.csv`의 `marker`와 행마다 같다 |
| 1-4 | `$CH3/lf/{g2lf_final, labels_lf.json, input_meta_lf.csv, X182_lf.parquet, fu2_lf/tokens}`가 있고, sha256이 `analysis/ch3/results/lf_inputs/F_report.json`(`labels_sha256`, `meta_sha256`)과 `.../fu2_lf/T_report.json`(`x182_sha256`, `manifest_sha256`)에 적힌 값과 같다 |
| 1-5 | `analysis/d1_ladder/results/labels_d1_cr2.parquet`가 9,266행이다 |
| 1-6 | 환경: D1 run env, `analysis/deploy/jobs/cpu_d1.slurm`, `analysis/d1_pair/jobs/g.slurm` + `PYENV=venv5`(tblite 0.7.0), xtb 6.7.1, ORCA 6.1.1 |
| 1-7 | 디스크 여유 30 GB 이상. `$G2`와 `$CH3/lf`의 크기(파일 수, 바이트)를 적어 둔다 |
| 1-8 | `$G2R`가 `$G2`·`$CH3` 아래에 있지 않고, `$G2`·`$CH3/lf`가 symlink가 아니다(`realpath`로 확인) |

---

## 2. 브랜치 · 폴더

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction        # 주 저장소 (실제 경로 확인)
git fetch origin gnn3-clean
git worktree add -b g2r /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2r origin/gnn3-clean
sbatch --time=48:00:00 --wrap "mkdir -p $G2R/{logs,x,jobs,lf/prof_empty}"
```
- 이 지시서를 `docs/specs/D1G2R.md`로 commit한다.
- `$G2R/lf/prof_empty`는 **빈 폴더로 둔다**(특성 코드가 DFT 구조를 읽으려 하면 실패하도록, DEPLOY 1절).
- deploy·ch3 job 스크립트의 `#SBATCH --output`은 지워진 폴더를 가리킬 수 있다. 제출할 때 항상 `--output=$G2R/logs/%x_%j.out`(배열은 `%x_%A_%a.out`)으로 덮어쓴다.
- deploy 코드는 환경 변수로 경로만 바꿔 쓴다(CH3 3절과 같은 방식). 모든 deploy 명령에 `--export=ALL,...`로 넘긴다.

| 변수 | 값 |
|---|---|
| `DEP_SCRATCH` | `$G2R/lf` |
| `DEP_RESULTS` | `<worktree>/analysis/g2r/results/lf` |
| `DEP_G2_JOBS` | `$G2R/jobs` |
| 나머지 | deploy 기본값. `DEP_G2_FINAL`은 기본값(`$G2/g2_final`)으로 두고 L 단계 대조에만 쓴다(D 단계 전) |

---

## 3. Phase C — TS 원본 복사 (`analysis/g2r/c_copy.py`)

1. 9,266개 대상 모두, `$G2/jobs/<job_id>/` 아래 파일 전부(하위 폴더 포함, 표지 파일 포함)를 `$G2R/jobs/<job_id>/`로 복사한다. 반응 폴더 단위로 임시 폴더 → `os.replace`, 멱등.
2. 파일마다 sha256을 원본과 대조한다.
3. 출력: `$G2R/x/C_manifest.csv.gz`(job_id, 상대 경로, 바이트, sha256), `results/C_report.json`(반응 수, 파일 수, 총 바이트, 불일치 수).

| # | 검사 | 실패 시 |
|---|---|---|
| C-1 | 반응 폴더 9,266개, 파일 수·바이트 = 원본 | STOP |
| C-2 | sha256 불일치 0 | STOP |

이후 모든 단계는 TS 원본을 `$G2R/jobs`에서만 읽는다.

---

## 4. Phase X — 대상 목록 (`analysis/g2r/x_targets.py`)

1. `d1pair_common.load_d1_labels()` → skeleton ∉ `d1pair_common.DPP` → 9,266행. 레코드의 `flags`에서 `flag_short_forming`, `flag_regio_ambiguous`, `flag_async`를 읽는다.
2. 대상 = `flag_short_forming`과 `flag_regio_ambiguous`가 모두 False인 행, job_id 순.
3. 대조: `analysis/d1_g2/results/C_compare.csv`의 `label_flag_*`와 행마다 같아야 한다.
4. 반응마다 그룹을 정한다(`$G2R/jobs`의 표지 기준):

| 그룹 | 표지 | 처리 |
|---|---|---|
| `reuse` | `.done` | 6절 L부터 |
| `repair` | `.fail_ref_AtomValenceException`, `.fail_ref_KeyError` | 5절 R → 성공하면 6절 L |
| `excluded_check` | `.fail_ts_TS_check_failed`, `.fail_ts_partition` | D-4: 제외, 5-5절에서 기록 |
| `excluded_ts` | 그 밖의 `.fail_*` | D-6: 제외 |

5. cr2 확인(기록만, 대상은 바꾸지 않는다): `labels_d1_cr2.parquet`에서
   - 기준 초과 = `*_rel_new_eh` − (같은 `key1`/`key2`의 최저값), kcal/mol. 최대값을 적는다.
   - 위생 실패 수: 엄격(`d1_new`, `d2_new` ≥ 0, `d2_new` ≤ 50)과 허용(≥ −0.5)을 각각 적는다. −0.5는 D1 run 설정의 `d_min_kcal`이다.
6. 출력: `results/X_targets.csv`(rxn_id, job_id, dipole, skeleton, series, attach, flag_async, ex_orig_gt20, g2_marker, group), `$G2R/x/reuse_ids.csv`·`$G2R/x/repair_ids.csv`(`job_id` 열), `results/X_report.json`.

| # | 검사 | 기대 | 다르면 |
|---|---|---|---|
| X-1 | 비D″ 행 수 | 9,266 | STOP |
| X-2 | short_forming, regio_ambiguous, 겹침, 대상 수 | 9, 5, 0, 9,252 | STOP |
| X-3 | 그룹별 수 | reuse 8,614, repair 20, excluded_check 19, excluded_ts 599 | STOP |
| X-4 | C_compare flags와 불일치 | 0 | STOP |
| X-5 | 대상 중 flag_async, ex_orig_gt20 | 209, 8 | STOP |
| X-6 | cr2 기준 초과 최대, 위생 실패(허용 / 엄격) | ≤ 20 (실측 약 0.05), 0 / 4 | 기준 초과 > 20 또는 허용 실패 > 0이면 STOP |

---

## 5. Phase R — 기준 구조 단계 코드 오류 20개 (`analysis/g2r/ref_repair.py`)

### 5-1. 진단 (계산 없음)

- 20개마다 `$G2R/jobs/<job_id>/ref_result.json`의 `traceback`과 `.fail_ref_*` 표지 문장을 읽는다.
- 원인이 같은 것끼리 묶어 `results/R_diagnosis.md`에 쓴다: 예외가 난 함수와 줄, 입력(어느 분자, 어떤 구조), 왜 났는지.
- 이 20개는 TS 단계와 입체 확인을 통과했다(`stereo = match`). 실패는 기준 구조를 만드는 코드에서 났다.

### 5-2. 고쳐도 되는 범위 (미리 허용)

| 고쳐도 되는 것 | 고치면 안 되는 것 |
|---|---|
| 분자 표현을 읽는 부분만. 예: RDKit이 전하 분리 구조(ylide, nitrone)의 원자가를 거부할 때 manifest SMILES의 형식 전하를 그대로 지정하거나 sanitize 단계 일부를 건너뛰는 우회, dict 키 이름이 어긋날 때 같은 값을 맞는 키로 찾는 우회 | Coley 규칙의 판정(회전 가능 문턱, alt 사용 문턱 RMSD > 0.05 Å·ΔG < −0.239 kcal/mol, 고리 수 변화 시 중단, dipolarophile 배치 확인), xtb 명령과 수준, conformer 수 |

- 우회는 **원래 함수가 그 예외를 낼 때만** 실행한다. 예외가 없으면 원래 함수의 결과를 그대로 돌려준다. 그래서 기존 ok 반응에는 영향이 없다.
- D1 run 코드(`make_reference`, `stereo_utils`, `pilot_common`)와 `analysis/d1_g2/g2_autode.py`는 고치지 않는다. 우회는 `analysis/g2r/ref_repair.py`에서 해당 함수를 감싼다.
- 원인이 이 범위 밖이면(화학적 이유이거나, 판정 규칙을 바꿔야 고쳐지는 경우) **STOP**하고 `R_diagnosis.md`를 보고한다.

### 5-3. 검사 (`analysis/g2r/tests_g2r.py`, sbatch로)

| # | 검사 | 통과 |
|---|---|---|
| R-T1 | 예외가 없는 입력에서 wrapper 호출 | 원래 함수가 1번 불리고 결과가 그대로 나옴 (mock) |
| R-T2 | 20개 중 원인 묶음마다 1개, 우회 경로를 함수 단위로 (xtb 계산 없이) | 예외 없이 끝남 |

### 5-4. 실행

1. 반응마다 `$G2R/jobs/<job_id>/`의 실패 기록을 보존한다: `ref_result.json` → `ref_result.orig.json`, 표지 이름은 `R_status.csv`에 적는다. 그다음 `.fail_ref_*` 표지를 지운다(복사본에서만. 원본 기록은 C_manifest에 있다).
2. `g2_autode.stage_ref(cfg, job, jd=$G2R/jobs/<job_id>)`를 wrapper가 켜진 상태로 부른다. cfg = `analysis/d1_g2/config_g2.yaml`. ts·stereo 단계는 다시 하지 않는다(표지가 `.done`).
3. 화학적 이유(`StageFail`: 고리 수 변화, dipolarophile 배치 등)로 실패하면 그 이유로 실패로 남긴다.
4. job: `analysis/g2r/jobs/cpu_g2r.slurm`(D1 run env), 1 task, 멱등(`.done_ref`가 있으면 건너뜀).
5. 출력: `results/R_status.csv`(job_id, 원래 이유, 새 상태, 새 이유, alt_used, 코어시간), `$G2R/x/repaired_ok_ids.csv`.

### 5-5. TS 검사·분할 탈락 19개 기록 (D-4, 계산 없음)

- 반응마다 `$G2R/jobs/<job_id>/`의 `ts_result.json`(있으면)과 `.fail_ts_*` 표지 문장에서 실패한 검사 이름(`one_imag`, `imag_below_min`, `extra_imag_small`, `no_foreign_bond`, `not_product_like`, `forming_bond_present`, `imag_mode_on_forming_bonds`, partition)과 값(허수 진동수 목록, 형성 결합 거리, 허수 모드의 형성 결합 비율)을 옮긴다.
- 출력: `results/R_check_fails.csv`. 값이 없으면 빈칸으로 둔다. **포함하지 않는다.**

---

## 6. Phase L — 진동수 마무리 (deploy `finalize_lf.py` 그대로)

```
# $G2R/x/lf_ids.csv = reuse_ids + repaired_ok_ids (job_id 열)
finalize_lf.py run @I 8 <cores> --jobs $G2R/x/lf_ids.csv        # 배열 0-7%8, DEP_G2_JOBS=$G2R/jobs
finalize_lf.py report --jobs $G2R/x/reuse_ids.csv               # deploy L 관문은 reuse 반응에만
```
`<cores>`, 스크립트(`jobs/cpu_d1.slurm`)는 DEPLOY 3-2절·CH3 3절과 같다.

| # | 검사 | 실패 시 |
|---|---|---|
| L-1 | reuse: deploy L-1·L-2·L-4 (진동수 실패 ≤ 0.5 %, `$G2/g2_final` 구조와 같음, 허수 진동수 차) | STOP |
| L-R | reuse 반응의 새 `ts.xyz`·`rel1.xyz`·`rel2.xyz`가 `$CH3/lf/g2lf_final`의 같은 파일과 byte 단위로 같고, 최저 허수 진동수 차 ≤ 1e-3 cm⁻¹ (모든 반응) | STOP |
| L-K | 복구 반응: `.done`이면 허수 진동수 ≥ 1개, `imag_mode_forming_share`를 적는다. `results/L_repaired.csv`. deploy의 L-2·L-4는 대조할 `$G2/g2_final` 구조가 없어 적용하지 않는다 | 보고 |

---

## 7. Phase M — 메타데이터 (deploy `meta_lf.py run`, `jobs/cpu_d1.slurm`)

| # | 검사 | 실패 시 |
|---|---|---|
| M-1 | 레코드 수 = L `.done` 수, 전하 ≠ 0 = 0 | STOP |
| M-2 | reuse 반응의 레코드와 `input_meta_lf.csv` 행이 `$CH3/lf`의 같은 rxn_id와 같음 | STOP |

---

## 8. Phase F — 전역 182열 (deploy `feat_lf.py` + D-5 허용치)

### 8-1. 허용치 패치

- `analysis/g2r/site_g2r/sitecustomize.py`:
  1. `analysis/d1_pair/site_si/sitecustomize.py`를 `runpy.run_path`로 그대로 실행한다(Si 항목).
  2. `G2R_TOL_PATCH=1`일 때만 `rev5_common.GATE_E_EH = 1e-4`, `ext_features.TOL_DIP_AU = 1e-2`로 둔다.
  - 인터프리터 시작 때 돌므로 spawn worker에도 들어간다(site_si와 같은 방식).
- job 스크립트 `analysis/g2r/jobs/g_g2r.slurm` = `analysis/d1_pair/jobs/g.slurm` 복사 + `PYTHONPATH` 맨 앞에 `analysis/g2r/site_g2r` + `export G2R_TOL_PATCH=1`. 원본 `g.slurm`은 고치지 않는다.
- 시작할 때 단언한다: Si 네 항목 `(1.11, 14, 2.10, 4)`, 두 허용치 값.

### 8-2. 명령

`feat_lf.py slice @I 2` (배열 `0-1`) → `feat_lf.py merge`. `PYENV=venv5`, `g_g2r.slurm`.

### 8-3. 검사

| # | 검사 | 실패 시 |
|---|---|---|
| F-1 | 열 이름·순서 = `$CH3/lf/X182_lf.parquet` | STOP |
| F-2 | ext 보고서의 `tolerances`에 `GATE_E_EH = 1e-4`, `TOL_DIP_AU = 1e-2`, Si 패치 켜짐 | STOP |
| F-3 | 특성 실패 ≤ 1 % | STOP |
| F-R | 예전에 통과한 반응(대상 중 8,502개)의 182열 = `$CH3/lf/X182_lf.parquet`의 같은 행, 최대 \|차\| ≤ 1e-6 | STOP |
| F-N | 새로 통과한 반응 = 예전 탈락 112 (+ 복구 k). 반응마다 engine gate 값과 쌍극자 검산 값을 `results/F_newly_passed.csv`에 쓴다. 112 중 아직 탈락한 반응은 값과 함께 보고 | 보고 |
| F-D | 모든 반응의 \|E_tblite − E_xtb\| 최대값, 쌍극자 검산 값 분포(중앙값, 99 %, 최대) | 보고 |

---

## 9. Phase T — 원자 토큰 (deploy `tokens_lf.py`, 8-1과 같은 환경)

`tokens_lf.py atoms @I 8` (배열 `0-7%8`) → `tokens_lf.py gate` → `tokens_lf.py tokens`

| # | 검사 | 실패 시 |
|---|---|---|
| T-1 | DEPLOY 6절 그대로: 원소 `[1, 6, 7, 8, 9, 14, 16, 17, 35]`, 열(`d_in = 29`), Q-a, Q-b, T-a | STOP |
| T-R | 예전에 사용 가능했던 반응(대상 중 8,500개)의 토큰 = `$CH3/lf/fu2_lf/tokens`의 같은 반응, 최대 \|차\| ≤ 1e-6 | STOP |
| T-N | 예전 확장 실패 2개(rxn 1007205: 3.7 × 10⁻³, 1007289: 2.0 × 10⁻³ a.u.)가 통과했는지 | 보고 |

---

## 10. Phase U — 사용 가능 목록 (`analysis/g2r/u_usable.py`)

- 사용 가능 = 대상 9,252개 중 L `.done` ∧ 182열 모두 유한 ∧ 토큰 있음.
- 출력:
  - `results/U_usable.csv`: rxn_id, job_id, usable, 빠진 단계, 이유
  - `results/U_funnel.csv`: 0-4절 표와 같은 순서로 단계, 빠짐, 남음
  - `results/U_report.json`: 사용 가능 수, sha256(`labels_lf.json`, `input_meta_lf.csv`, `X182_lf.parquet`, 토큰 manifest)
- U-1: 사용 가능 수가 0-4절 기대(8,614 + k)와 다르면 단계별 원인 표를 쓴다(보고, STOP 아님).

---

## 11. Phase S — 보고 · commit · push

- `results/RUN_REPORT_G2R.md` (한국어).
  - 맨 앞 세 문장: 사용 가능 수와 9,252 대비 비율 / 복구 k/20과 원인 / 허용치 완화로 새로 들어온 반응 수.
  - 절: C → X → R → L → M → F → T → U → D(삭제, 12절 뒤에 채움) → 비용 → 한계.
  - 한계에 꼭 적을 것:
    - TS는 재사용했다. autodE는 seed가 고정되지 않아 다시 찾으면 약 절반이 다른 TS가 된다(D1 검증 V3: 같은 TS 0.50).
    - 복구한 반응은 기준 구조의 conformer 탐색을 이번에 처음 끝까지 돌렸다.
    - TS 검사·분할 탈락 19개와 TS가 없는 595개는 배포 때도 예측할 수 없는 반응이다.
  - 수치는 모두 파일에서 옮긴다.
- commit: `analysis/g2r/`(코드, results), `docs/specs/D1G2R.md`. `$G2R` 출력은 commit하지 않는다.
- `git push origin g2r`. push한 HEAD SHA를 보고서에 적는다.

---

## 12. Phase D — 기존 g2 삭제 (`analysis/g2r/d_delete.py`)

### 12-1. 삭제 조건 (하나라도 아니면 STOP, 삭제하지 않는다)

| # | 조건 |
|---|---|
| D-a | 1, C, X, R, L, M, F, T의 STOP 관문이 모두 통과했다 (각 report의 판정을 다시 읽어 확인) |
| D-b | S의 commit이 `origin/g2r`에 push되어 있다 (`git ls-remote`로 SHA 확인) |
| D-c | `$G2R/jobs`가 C_manifest와 지금도 같다 (파일 수·바이트, 무작위 200파일 sha256) |
| D-d | 실행 중이거나 대기 중인 내 job 중 명령·출력 경로에 `$G2` 또는 `$CH3/lf`가 들어간 것이 없다 (`squeue -u $USER -o "%i %j %o"`, `scontrol show job`) |
| D-e | `/gpfs/tmp_cpu2/yeseo1ee`와 `/gpfs/home1/yeseo1ee/projects` 아래에 `$G2` 또는 `$CH3/lf`를 가리키는 symlink가 없다 (`find -lname`). 있으면 목록을 보고 |

### 12-2. 삭제

1. 삭제 전 기록: `results/D_before.json` — 경로마다 `realpath`, 파일 수, 총 바이트, 최상위 항목 목록. `$G2R/x/D_listing_<이름>.txt.gz` — 전체 파일 목록(상대 경로, 바이트).
2. 지울 경로는 **정확히 두 개**: `realpath($G2)` = `/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `realpath($CH3/lf)` = `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf`. 코드에 이 두 문자열을 상수로 두고, 실제 `realpath`가 다르면 STOP.
   - `$CH3`의 다른 항목(`units`, `model`, `c3r`, `logs` 등)과 `$LAD`, `$D1PAIR_SCRATCH`, D1 run은 지우지 않는다.
3. `sbatch --time=48:00:00 --wrap "rm -rf /gpfs/tmp_cpu2/yeseo1ee/d1_g2 /gpfs/tmp_cpu2/yeseo1ee/ch3/lf"` (두 경로만, 변수 확장 없이 문자열 그대로).
4. 확인: 두 경로가 없고, `$G2R`의 파일 수·바이트가 삭제 전과 같다.
5. `results/D_report.json`(삭제한 경로, 파일 수, 바이트, 시각, job id), `RUN_REPORT_G2R.md`의 D 절과 0-5절 내용(앞으로 g2 = `$G2R`)을 채운다.
6. commit · push. push한 HEAD SHA를 보고서에 적는다.

### 12-3. 왜 이 순서인가

- TS 원본(autodE 결과)은 C 단계에서 9,266개 모두 `$G2R/jobs`로 옮겼다. 지운 뒤에도 같은 TS로 모든 입력을 다시 만들 수 있다.
- 새 입력이 기존 입력과 같다는 확인(L-R, F-R, T-R)은 기존 폴더가 있어야 할 수 있다. 그래서 삭제는 확인과 push가 끝난 뒤에 한다.

**D가 끝나면 멈춘다.** 짝 재구성과 학습은 다음 지시서로 한다.

---

## 13. 자원 · 규칙

- 비용(추정): R 약 3 core-h(반응당 기준 구조 평균 0.14 core-h, `G_status.csv`), L·M·F·T 약 28 core-h(CH3 실측). 합계 약 31 core-h. C·D는 파일 복사·삭제뿐이다.
- MaxJobs 10, MaxSubmit 20(배열 원소 포함). 제출 전에 `squeue -u $USER -h | wc -l`을 확인한다.
- 파티션: `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`로 idle이 많은 쪽.
- 48시간 안에 못 끝나면 같은 명령으로 다시 제출한다(멱등, 반응 폴더 단위 원자적 쓰기).
- 쓰기: `analysis/g2r/`, `$G2R/`, 이 지시서만. 삭제: 12절의 두 경로만, 12-1 조건을 모두 통과한 뒤에만.
- 0-2·0-3절에 없는 방법·문턱·gate를 바꾸지 않는다. 결과를 본 뒤 바꿀 필요가 보이면 STOP.
- 원자적 쓰기(임시 파일 → `os.replace`), 멱등. 수치는 파일에서 옮긴다.

## 14. STOP

| 단계 | 조건 |
|---|---|
| 1 | 시작 조건 하나라도 실패 |
| C | C-1, C-2 실패 |
| X | X-1–X-5 중 하나라도 기대와 다름, X-6의 기준 초과 > 20 또는 허용 위생 실패 > 0 |
| R | 원인이 5-2절 허용 범위 밖, R-T1·R-T2 실패 |
| L | L-1, L-R 실패 |
| M | M-1, M-2 실패 |
| F | F-1, F-2, F-3, F-R 실패 |
| T | T-1, T-R 실패 |
| D | 12-1 조건 하나라도 실패, 지울 경로의 `realpath`가 12-2의 두 문자열과 다름 |
| 언제든 | 기존 코드를 고쳐야 할 것 같음, 이 문서에 없는 방법 변경이 필요해 보임 |

STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.
