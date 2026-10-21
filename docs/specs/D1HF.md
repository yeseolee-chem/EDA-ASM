# D1-HF — D1 라벨 전체의 값싼 EDA (HF/6-31G*) — Claude Code 실행 지시서

**저장소:** `d1-pair` 브랜치 HEAD에서 새 브랜치 **`d1-hf`**를 만들어 commit한다. FU2 Phase E 작업과 파일이 겹치지 않게 하기 위해서다.
**이 파일:** `docs/specs/D1HF.md`로 commit한다.
**새 코드:** `analysis/d1_hf/` · **결과:** `analysis/d1_hf/results/` · **scratch:** `/gpfs/tmp_cpu2/yeseo1ee/d1_hf/` (이하 `$HF`)
**순서:** X(입력 준비) → S(smoke + 재현 검사) → P(파일럿 100반응) → **멈춤: 사용자 승인** → M(본계산) → A(정답과 비교) → R(보고)
**보고:** 단계마다. STOP이면 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.

---

## 0. 무엇을, 왜

- 라벨은 반응마다 ORCA 계산 5개(EDA 1 + 조각 단일점 4)를 B3LYP-D3BJ/def2-TZVP, SMD(water)로 한 것이다.
- 이번에는 **같은 5개 계산을, 같은 구조·같은 조각·같은 전하로, 방법 줄만 HF/6-31G\*로 바꿔** 다시 한다.
- 이 값은 나중에 Δ-학습의 **입력**으로 쓴다. 라벨과 같은 값이 나올 필요는 없다. 중요한 것은 계산이 라벨 계산과 **같은 입력에서 방법만 다르다**는 점이다.
- 6-31G\*를 고른 이유: SMD 매개변수를 정할 때 쓴 수준 중 하나가 HF/6-31G\*이다(Marenich, Cramer, Truhlar, *J. Phys. Chem. B* 2009, 113, 6378, DOI 10.1021/jp810292n).

## 1. 범위

- 대상 = `labels_d1.json`(`/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/labels_d1.json`, sha256 `1eaa4171…`, 9,496행 모두 status ok) 중 **skeleton이 `DppC`·`DppT`(D'')가 아닌 행 전부**.
  - 로더는 `d1pair_common.load_d1_labels()`, 제외 목록은 `d1pair_common.DPP`를 그대로 쓴다.
  - 대상 행 수를 X 보고에 적는다.
- 라벨 파일, `d1_run/`, `d1_pair/d1_jobs/`, D1PAIR 결과는 **읽기만** 한다.

## 2. 방법 (고정)

원래 입력의 `!` 줄과 `%eda` 블록의 `FRAG1`·`FRAG2` 문자열에서 **두 단어만** 바꾼다.

| 원래 | 새 값 |
|---|---|
| `B3LYP` | `HF` |
| `def2-TZVP` | `6-31G*` |

나머지(`D3BJ`, `SMD(water)`, `NoSym`, `EDA`, `TightSCF`, 좌표, 조각 표시, `FRAG*_C`·`FRAG*_M`, `%maxcore 3500`)는 그대로 둔다. 예:

```
! HF D3BJ 6-31G* SMD(water) NoSym EDA TightSCF
%maxcore 3500
%pal nprocs 1 end
%eda
  FRAG1 "HF D3BJ 6-31G* SMD(water) NoSym TightSCF"
  FRAG2 "HF D3BJ 6-31G* SMD(water) NoSym TightSCF"
  FRAG1_C …  FRAG1_M …  FRAG2_C …  FRAG2_M …      (원래 값 그대로)
end
```

- `%pal nprocs`는 1로 둔다(계산 하나에 코어 하나). 많은 계산을 동시에 돌리는 쪽이 효율이 좋다.
- X 단계에서 원래 입력들에 나오는 키워드 종류를 모두 모아 보고한다. 위 두 단어 말고 바꿔야 할 것이 보이면(예: def2-TZVP에 묶인 보조 기저) **바꾸지 말고 STOP**.
- ORCA는 라벨과 같은 6.1.1(`/home1/yeseo1ee/orca_6_1_1_avx2`, `jobs/g_env.sh`)만 쓴다. 출력 머리의 버전과 GIT 해시를 모든 계산에서 기록한다.

## 3. Phase X — 입력 준비 (CPU, 읽기 전용 추출)

`d1_pair/d1_jobs/<T>/`에는 `ts.xyz`, `ref_*.xyz`, `label.json`, `sp/eda.inp`만 있다. 조각 입력 4개는 원래 압축본에서 꺼낸다.

1. `d1_extract_jobs.py`와 같은 출처(Nurion 번들 tar 속 `jobs/<T>/archive.tgz`, UBAI 폴더 58개)에서 `sp/eda.inp`와 `sp/frag1_dist.inp`, `sp/frag1_rel.inp`, `sp/frag2_dist.inp`, `sp/frag2_rel.inp`를 읽기 전용으로 꺼내 `$HF/inputs/<T>/orig/`에 둔다. 원본은 바꾸지 않는다.
2. 검사(하나라도 어긋나면 그 행을 `x_fail`로 기록; 전체의 0.5 %를 넘으면 STOP):
   - 꺼낸 `eda.inp`가 `d1_jobs/<T>/sp/eda.inp`와 바이트가 같음
   - `frag*_dist.inp` 좌표·원소 = `eda.inp`의 그 조각 원자(순서 포함, 1e-6 Å)
   - `frag*_rel.inp` 좌표·원소 = `ref_dipole.xyz` / `ref_dipolarophile.xyz` (어느 조각이 dipole인지는 라벨 규격대로: FRAG1 = dipole; 다르면 STOP)
   - 조각 전하·다중도 = `eda.inp`의 `FRAG*_C`·`FRAG*_M`
   - 5개 입력 모두 `SMD(water)`가 `!` 줄과 FRAG 문자열에 있음(라벨 용매 버그 검사, `claude/methods-eda-smd-fragment-consistency` 3절)
3. 2절 규칙으로 새 입력 5개를 만들어 `$HF/inputs/<T>/hf/`에 둔다. 원래 입력과의 차이가 정확히 2절의 두 단어 + `%pal`뿐인지 diff로 확인한다.
4. 출력: `results/X_manifest.csv`(job_id, rxn_id, dipole, skeleton, 원자 수, 원소 목록, 입력 sha256), `results/X_report.json`(대상 수, x_fail, 키워드 목록).

## 4. 계산과 라벨 조립

**runner** (`analysis/d1_hf/run.py`): 반응 하나 = 계산 5개. 조각 단일점 4개를 먼저 돌리고, 그다음 EDA를 돌린다.
- 멱등: 정상 종료한 `.out`이 있으면 건너뛴다. 작업 폴더 `$HF/runs/<geom>/<T>/`.
- 수렴 실패 시 **한 번만** 다시 돌린다. 허용하는 변경은 `SlowConv`와 `%scf MaxIter 500 end`뿐이다(방법·기저·용매는 바꾸지 않음). 무엇을 바꿨는지 `attempts.tsv`에 적는다.
- 정상 종료 후 `.out`은 gzip, `.gbw`·`.tmp`·`.densities`는 지운다.
- 시간: 계산마다 `TOTAL RUN TIME`을 `timing.tsv`에 적는다.

**조립** (`analysis/d1_hf/build.py`): 라벨 규격(`claude/methods-eda-smd-fragment-consistency` 1–2절)과 같은 식, **Hartree 열**에서 읽는다.

```
d1 = FSPE(frag1_dist) − FSPE(frag1_rel)        d2 = FSPE(frag2_dist) − FSPE(frag2_rel)
eint_spe = FSPE(eda) − FSPE(frag1_dist) − FSPE(frag2_dist)
barrier  = FSPE(eda) − FSPE(frag1_rel)  − FSPE(frag2_rel)
bond = ORCA Bond Energy;   bsse_total = bond − eint_spe
elst = Electrostatic · pauli = Pauli + (EDA 표의 교환/XC 항) · oi = Orbital · disp = Delta Dispersion
cpcm = Delta CPCM Dielectric · cds = Delta SMD CDS correction
```

- HF의 EDA 표는 B3LYP와 항목 이름이 다를 수 있다. S 단계에서 HF 출력의 EDA 표 전체를 보고서에 붙이고, 위 대응을 확정한다. 표에 위 목록으로 설명되지 않는 항목이 있으면 **STOP**.
- 반응별 gate(라벨과 같음):

| gate | 기준 |
|---|---|
| g1_asm | \|barrier − (d1 + d2 + eint_spe)\| < 1e-6 |
| g2_sum | \|Σ6채널 − bond\| < 0.01 kcal/mol |
| g3_seam | \|(d1 + d2 + Σ6채널 − barrier) − bsse_total\| < 1e-3 |
| g_enuc | ENuc(frag\*_dist) = ENuc(eda_frag\*) |
| g_setting | 복합체·FRAG 모두 SMD(water), 방법 HF, 기저 6-31G\* |
| orca | 모든 계산이 같은 ORCA 6.1.1 버전·GIT 해시 |

- 출력: `$HF/labels_d1_hf.json`과 `results/labels_d1_hf.parquet`(job_id, rxn_id, geom, d1, d2, elst, pauli, oi, disp, cpcm, cds, eint_spe, barrier, bond, bsse_total, gate 결과, orca 버전, 계산별 코어시간), `results/hf_status.csv`(ok / 실패 이유).

## 5. Phase S — smoke와 재현 검사 (작게, 먼저)

| 검사 | 반응 | 통과 기준 |
|---|---|---|
| S-1 재현 | 원자 수가 가장 작은 D1 반응 2개를 **원래 B3LYP/def2-TZVP 입력 그대로** 새 runner·build로 계산 | `label.json`과 9개 값(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier) 차이 ≤ 0.01 kcal/mol → runner·조립 경로가 라벨과 같음을 확인 |
| S-2 원소 | F, Si, S, Cl, Br를 각각 포함하는 반응 1개씩 + 가장 큰 반응 1개 (6개), HF 입력 | 모두 정상 종료, gate 통과. 6-31G\*가 없는 원소나 `HF D3BJ`를 받지 않으면 **STOP**(대체 기저·대체 분산을 스스로 고르지 않는다) |
| S-3 표 | S-2 출력 하나의 EDA 표 전체 | 4절 대응 확정 |

## 6. Phase P — 파일럿 100반응

**뽑기 (무작위, 고르게):**
- 후보 = 1절 대상 − S-2 반응.
- 층 = (dipole 5 × skeleton) 조합. 층마다 2개를 `numpy.random.default_rng(20261021)`로 비복원 추출한다.
- 층에 행이 1개뿐이면 1개를 뽑는다. 모자란 수는 같은 생성기로 남은 후보 전체에서 채워 정확히 100개를 만든다.
- 뽑힌 목록을 계산 전에 `results/P_sample.csv`로 commit한다.

**계산:**
- (가) 라벨 구조(DFT TS): 5개 계산 전부.
- (나) G1 구조 비교: 100개 중 G1이 있는 반응(`d1_pair` scratch `g1_final/<rxn_id>`, D1PAIR feature가 읽는 TS 좌표)만, **EDA 1개**를 같은 조각·전하로 계산. 원자 순서와 원소가 DFT TS와 같지 않으면 그 반응은 뺀다.

**파일럿 보고 (`results/P_REPORT.md`):**

| 항목 | 내용 |
|---|---|
| 성공 | ok 수, 실패 이유별 수, 재시도 수 |
| gate | 6개 gate 통과율 |
| 비용 | 계산 종류별 코어시간 중앙값, 반응당 코어시간 ~ a × 원자 수^b 맞춤, **본계산 예상 코어시간과 기간**(아래 자원으로) |
| 정답과의 관계 (설명용) | 채널마다 HF 값 대 라벨 값의 Pearson r, 기울기, 절편 (반응 단위, n ≈ 100) |
| 구조 효과 (설명용) | (나)에서 채널별 HF@G1 − HF@DFT-TS의 MAE와 r |

**판정 (본계산으로 가는 조건):** ok ≥ 95 %, ok 행 gate 100 %, S-1 통과. 만족하더라도 **여기서 멈추고 사용자 승인을 기다린다.** 사용자가 정할 것: 본계산 시작, 그리고 G1 구조로도 EDA를 할지.

## 7. Phase M — 본계산

- 1절 대상 전부(파일럿·smoke에서 끝난 것은 건너뜀), 라벨 구조(DFT TS), 계산 5개.
- G1 구조 EDA는 사용자가 6절 뒤에 승인한 경우에만 같은 runner로 돌린다(`geom = g1`).
- 중간 점검: 끝난 반응 1,000개마다 ok 비율과 gate를 확인한다. ok < 90 %이거나 gate 실패가 나오면 **STOP**.
- 끝나면 `labels_d1_hf.json`·parquet·status를 확정하고 sha256을 적는다.

## 8. Phase A — 정답 짝 차이와 비교 (설명용, 사전 등록 아님)

D1PAIR 짝에서 HF 값이 정답의 치환기 효과를 얼마나 따라가는지 본다.
- 짝: `analysis/d1_pair/results/S_folds.csv` + `F_pairs.csv`의 P_main 기준 짝. HF ΔΔ = HF(a_rxn) − HF(b_rxn), 정답 = `dd_<타깃>`.
- 타깃 9개(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier)마다 정답 ΔΔ = α + β · HF ΔΔ를 `fold_T1`의 학습 쪽(fold ≠ k)에서 맞추고 시험 쪽에 적용한다(5 fold).
- 보고: 시험 MAE(전체·para), null(ΔΔ = 0) MAE, 상관 r. 비교 기준선으로 FU2 GNN-A+G para MAE(`results/fu2/M_predictions.parquet`에서 같은 짝으로 계산)를 함께 적는다.
- para MAE에 구간 표시만 한다: ≤ 0.13(단계 2 목표 수준), ≤ 0.26, ≥ 0.6. 판정이나 결론은 쓰지 않는다.
- 출력: `results/A_pairs.csv`, `results/A_summary.csv`, 그림 `results/figA_parity.png`(타깃별 HF ΔΔ 대 정답 ΔΔ, para 짝).

## 9. Phase R — `results/RUN_REPORT_HF.md` (한국어)

- 맨 앞 세 문장: 계산한 반응 수와 성공률, 총 코어시간, Phase A의 oi·cpcm para MAE.
- 절: X → S → P → M → A → 실패·재시도 → 한계.
- 한계에 꼭 적을 것:
  - HF에는 상관 에너지가 없다(분산은 D3BJ).
  - SMD 매개변수는 HF/6-31G\*를 포함하지만 EDA 채널 분해로 검증된 것은 아니다.
  - 라벨 구조(DFT TS)에서 계산했으므로, 새 반응 예측에 쓰려면 G1 구조 계산이 필요하다.
- 수치는 모두 파일에서 옮긴다.

## 10. 자원과 실행 (HPC)

- python과 ORCA는 모두 `sbatch`(`--time=48:00:00`)로만 돌린다. 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만 한다. MaxJobs 10, MaxSubmit 20(배열 원소 포함).
- 파티션 `cpu1,cpu2`. job 하나 = 노드 하나(`--cpus-per-task=48 --mem=200G`), 그 안에서 ORCA 계산 48개를 동시에 돌린다(작업 풀, 계산마다 1코어). `%maxcore 3500` × 48 = 168 GB.
- 환경: `jobs/env.sh` + `jobs/g_env.sh`의 ORCA·MPI 설정(`ORCA_DIR=/home1/yeseo1ee/orca_6_1_1_avx2`).
- 48시간 안에 못 끝나면 같은 명령으로 다시 제출한다(멱등).
- 예상(검증 전 추정): 라벨 EDA는 36원자 반응에서 1코어 5–6시간이었다. PySCF로 잰 HF/6-31G\*급 SCF 비용은 B3LYP/def2-TZVP의 약 1/17이다(용매 없음, 다른 프로그램이라 ±50 %). 그래서 조각 4개를 포함해 반응당 약 0.3–0.6 코어시간, 본계산 약 3,000–6,000 코어시간으로 본다. **파일럿 실측으로 다시 계산해 보고한다.**

## 11. 규칙

- 읽기만: `labels_d1.json`, `d1_run/` 전체, Nurion 번들, `d1_pair/d1_jobs/`, `d1_pair/` scratch와 결과, `d1-pair` 브랜치 파일.
- 새 파일은 `analysis/d1_hf/`, `$HF/`, 이 명령서에만 쓴다.
- 방법·기저·용매·구조·조각·전하는 2절과 다르게 바꾸지 않는다. 바꿀 필요가 보이면 STOP.
- 원자적 쓰기, 멱등. 수치는 파일에서 옮긴다.

## 12. STOP

| 단계 | 조건 |
|---|---|
| X | 원래 입력에서 2절 두 단어 외에 바꿀 것이 보임; FRAG1이 dipole이 아님; x_fail > 0.5 %; SMD 없는 입력 |
| S | S-1 재현 차이 > 0.01 kcal/mol; 6-31G\* 또는 `HF D3BJ`를 ORCA가 받지 않음; HF EDA 표에 설명 안 되는 항목 |
| P | 판정 조건 미달; 조건을 만족해도 사용자 승인 전 본계산 시작 |
| M | ok < 90 % (1,000개 단위 점검); gate 실패; ORCA 버전·해시 불일치 |
| 언제든 | 이 문서에 없는 방법 변경이 필요해 보임 |

STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.
