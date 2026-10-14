# SPEC_D1_ALL — D1 확정 후 분석 지시서 (통합본, 이 파일 하나로 실행)

- 작성 2026-10-14 · 소유자 Yeseo (yeseolee-chem/EDA-ASM) · 실행: Claude Code (D1을 계산한 서버)
- **이 문서 하나가 지금까지 따로 보낸 지시 파일 전부를 대체한다**: `SPEC_D1_ANALYSIS.md`(모든 수정본), `D1_analysis_bundle.zip`, `SPEC_D1_PHASE_E.md`(취소), `D1_phaseE_bundle.zip`. 그 파일들은 쓰지 않는다. 분석 코드 10개와 기준값 파일 1개는 **부록 A·B에 전문이 들어 있다.**
- 선행 문서(저장소 안): `D1_build/d1_run/SPEC_D1_RUN.md`, `SPEC_D1_FIX1.md`(D1 생성), `docs/specs/REV5_FEATURES_FIGURES.md`(현재 모델), `analysis/espley_xtb_repro/results_rev5/SUMMARY.md`, `CLAUDE.md`(HPC 규칙)

이 문서는 순서대로 실행한다. 각 Phase 끝의 게이트가 통과해야 다음으로 간다. 게이트에 걸리면 멈추고 보고한다(§9).

---

## 0. 할 일, 순서대로

| 순서 | Phase | 내용 | 계산 | 끝나면 |
|---|---|---|---|---|
| 0 | **S** 설치 | 부록 A·B의 파일을 저장소에 쓰고 sha256 대조 | 없음 | `analysis/d1_stats/` |
| 1 | **R** 재현 | 코드를 중간 파일에 돌려 기준값과 일치 확인 | CPU 1, 약 10분 | `results_interim/` + 일치표 |
| 2 | **A** 확정 데이터 | D1 본계산 종료 후 `labels_d1.json`·`d1_status.csv` 동결, QC | 없음 | `results_final/qc.json` |
| 3 | **B** 통계 | B1–B8 (아래) | 수십 분 | `results_final/` + `STATS_REPORT.md` |
| 4 | **C** feature | D1 전 행에 G1 xTB feature(rev5 EXT_SEL 133열) | xtb, tblite | `xtb_features_d1_g1.parquet` |
| 5 | **D** 모델 | M0–M4 사전 등록 후 실행 | KRR | `MODEL_REPORT.md` |
| — | ~~E~~ | **취소** (§E) | — | — |

---

## Phase S — 설치 (게이트 S)

1. 저장소 루트에서 아래 스크립트를 이 문서 경로를 인자로 실행한다. 부록의 `<!-- FILE: 경로 sha256=… -->` 표시 바로 아래 코드 블록을 그 경로에 그대로 쓰고 sha256을 대조한다.

````python
# extract_spec_files.py  —  python extract_spec_files.py SPEC_D1_ALL.md
import sys, re, hashlib, pathlib
FENCE = "`" * 3
text = open(sys.argv[1], encoding="utf-8").read().split("\n")
i, n_ok = 0, 0
while i < len(text):
    m = re.match(r"<!-- FILE: (\S+) sha256=([0-9a-f]{64}) -->", text[i])
    if m:
        path, want = m.group(1), m.group(2)
        j = i + 1
        assert text[j].startswith(FENCE), f"no fence after {path}"
        k = j + 1
        while not text[k].startswith(FENCE):
            k += 1
        body = "\n".join(text[j + 1:k]) + "\n"
        got = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if got != want:
            sys.exit(f"SHA MISMATCH {path}: {got}")
        p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body, encoding="utf-8")
        n_ok += 1; i = k
    i += 1
print("files written:", n_ok)
````

2. **게이트 S:** `files written: 11`. sha256 불일치가 하나라도 있으면 STOP(문서가 복사 중 변형됨).
3. 필요 패키지: pandas, numpy, scipy, scikit-learn, rdkit, matplotlib, statsmodels. 없으면 설치 전에 보고한다.

---

## 1. 배경 (처음 보는 실행자용)

- **D0** = Coley [3+2] 5,260건 라벨(저장소 루트 `labels_all.json`). **D1** = 직접 설계한 확장 세트. dipole 5종(nitrile oxide, azide, azomethine ylide, nitrone, carbonyl ylide) × dipolarophile 골격 12종 × 치환기 80종(직결 20 + p/m/o-아릴 60) × 한쪽씩만 교체(다른 쪽은 H, 고정 골격은 `-`).
- 라벨 = TS 한 점의 8채널: `d1_kcal, d2_kcal, elst_dft, pauli_dft, oi_dft, disp_dft, cpcm_dft, cds_dft` + `bsse_shift_kcal`. 항등식 `barrier ≈ Σ8채널 + bsse_shift`(잔차 ≤ 0.11).
- **분석 단위는 짝 차이(ΔΔ)다.** 치환 TS − 같은 자리의 H 부모 TS. 같은 반응 틀에서 치환기만 바뀌었으므로 틀의 효과가 빠진다(분광에서 blank를 빼는 것과 같다). 아릴 치환기(p/m/o-X-C6H4)는 Ph 대비로도 본다(Hammett 본래 설정).
- 모든 채널 해석은 **실제 TS에서의 분해**로 쓴다(일정 기하 계산은 하지 않는다, §E).
- 중간 파일(9,459행, 2026-10-14 사용자 업로드)로 이미 분석했다. 아래 표의 "중간 파일 결과"가 그 값이다. 이 문서는 그것을 **확정 데이터로 다시 하고**, 거기서 할 수 없었던 것(결측 원인, 문헌 σ, GEDT, feature, 모델)을 더한다.

---

## 2. 규칙

- 저장소 `CLAUDE.md`의 HPC 규칙(계산 노드만, `sbatch`, 멱등, 원자적 쓰기, 로그인 노드 상주 금지)을 따른다. 분석 스크립트도 계산 노드에서 돌린다.
- **사전 등록.** Phase D의 분할·모델·지표·판정 기준은 결과를 보기 전에 `PREREG_D1.md`로 commit한다. 그 뒤로 바꾸지 않는다. 바꾸려면 사용자에게 묻는다.
- QC 임계값(`d1s_prep.py` 상단: `Q_BADREF_KCAL = −5`, `Q_MIN_FORMED = 1.65`, `Q_MAX_FORMED = 3.0`), 저랭크 설정(`d1s_lowrank.py`: `LAM = 1.0`, `ITERS = 60`, `NFOLD = 5`, `SEED = 1`), 변곡 판정(`inflection.py`: `MIN_N = 12`, `DAICC = 4`)은 고치지 않는다.
- train과 test MAE를 항상 함께 보고한다. 숫자는 실제로 쓴 파일에서 옮긴다.
- 짝(부모·자식, 형제 TS)은 **항상 같은 fold**에 둔다. 무작위 행 분할은 쓰지 않는다.
- D0 lockbox(`results_rev5/lockbox_ids.csv`, 726건)는 Phase D의 최종 1회 평가 전까지 열지 않는다.
- 부록의 코드는 그대로 쓴다. 버그를 찾으면 고치기 전에 보고한다.
- 새 QM 계산(일정 기하, 저극성 용매 단일점 등)은 하지 않는다. 예외는 없다(B5의 GEDT는 기존 출력만 읽는다).

---

## Phase R — 재현 (게이트 R)

- 중간 파일 = sha256 `a2df136b3d5964b288ed1500ef1393e5ead4d3f8df280191e8d2ad59d11240a7`(9,459행, 사용자가 올린 파일 이름은 `labels_all.json`이지만 D1이다). 서버에서 이 sha256인 파일을 찾는다(D1 실행의 중간 export일 것). 없으면 사용자에게 요청한다. 확정 파일로 게이트 R을 대신하지 않는다.

```bash
python analysis/d1_stats/run_all.py --labels <중간 파일> --d0 labels_all.json --out analysis/d1_stats/results_interim
python analysis/d1_stats/figs.py analysis/d1_stats/results_interim        # results_interim/fig/ 4장
```

- **게이트 R:** `results_interim/headline.json`이 부록 B의 `reference_interim/headline.json`과 모든 키에서 같다(실수는 소수 셋째 자리). 다르면 STOP하고 차이 표를 보고한다(패키지 버전 차이 가능).

---

## Phase A — 확정 데이터 동결

1. D1 본계산이 끝나면(`RUN_REPORT.md` commit) `D1_build/d1_run/results/labels_d1.json`, `d1_status.csv`, `d1_manifest.csv`의 sha256을 `results_final/INPUTS.md`에 적는다.
2. `run_all.py`를 확정 파일로 실행해 `results_final/qc.json`을 본다(명령은 B2).
3. **제외 규칙(사전 고정, 중간 파일 기준 건수):**

| 표식 | 정의 | 중간 파일 | 뜻 |
|---|---|---|---|
| `q_badref` | d1 또는 d2 < −5 kcal/mol | 10 | 기준 구조 붕괴(9건이 −61 ~ −66, 대부분 cyclooctyne 계열). autodE 장벽도 같은 값이라 autodE 반응물 자체가 틀린 구조 |
| `q_stepwise` | 짧은 형성 결합 < 1.65 Å 또는 긴 것 > 3.0 Å | 218 | 한쪽 결합이 다 생긴 단계적 TS(Pauli 최대 470). `flag_async` 216건과 일치 |

   - 제외는 통계·모델 공통이다. 제외 행 목록은 `qc.json`의 `bad_rows`.
   - **추가 점검:** `inspect_large_ddbarrier.csv`(\|ΔΔE‡\| > 15인 짝)를 하나씩 열어 기준 구조(`alt_used`)가 부모와 자식에서 다른지 확인한다. 중간 파일 최대값은 AY6-C o-NO2(+23.5, ΔΔd1 +24.3). 판단만 적고 제외하지 않는다.
4. **게이트 A:** G1–G3 탈락 0, `closure_max` ≤ 0.11, `orient_counts`에 `a`, `b`, `""` 외의 값(`noring`, `tie`, `err:*`)이 0. 하나라도 어긋나면 STOP.

---

## Phase B — 통계 (확정 데이터)

### B1. 기본 분석 (`run_all.py`)

`run_all.py`가 아래를 모두 만든다. 표 이름 = 출력 파일.

| 분석 | 무엇을 묻나 | 출력 | 중간 파일 결과 |
|---|---|---|---|
| 공분산 예산 | 장벽 변화를 어느 채널이 실어 나르나. share_k = Cov(ΔΔ_k, ΔΔE‡)/Var(ΔΔE‡), 합 = 1 | `budget.csv` | d1 0.58, d2 0.44, elst −1.11, Pauli +2.14, OI −1.11 (7,632짝) |
| 변형 vs 상호작용 | r(ΔΔ변형, ΔΔE‡), r(ΔΔE_int, ΔΔE‡) — **전체와 dipole별 모두 보고** | `headline.json` + B1b | 0.917 / −0.044. dipole별 E_int 상관은 +0.56(azide) ~ −0.43(nitrile oxide)로 부호가 갈림: 전체값만 보고 "무상관"이라 쓰지 말 것 |
| TS 위치 결합 | ΔΔ_k ~ Δζ (+Δ비동시성), ζ = 형성 결합 평균 | `zeta.csv` | elst·Pauli·OI 분산의 60–64 %가 Δζ·Δasyn으로 설명. Δζ +0.1 Å → ΔE‡ −2.9 |
| 채널 공선성 | ΔΔ 상관, PCA | `corr_dd.csv`, `pca.csv` | elst–Pauli −0.97, PC1 52 % (Δζ와 r 0.71) |
| 이식성·저랭크 | 치환기 상수 하나(σ)로 충분한가, σ·ρ 몇 개가 필요한가 (5-fold CV, 칸 단위 가림) | `lowrank.csv`, `joint_lowrank.csv` | dipole 쪽 직결 ΔE‡: 스칼라 0.10 → 덧셈 0.54 → rank1 0.52 → rank2 0.65 → rank3 0.73 |
| 선택성 분해 | regio(a−b), exo−endo 형제에서 어느 채널이 이긴 쪽을 고르나 | `selectivity.csv`, `regio_known.csv` | regio: 변형 0.83, E_int 0.39 / exo 100 %, d2가 100 % 결정 / 5-치환 선호 nitrile oxide 0.99, nitrone 0.96, azide 0.57 |
| 지문 | 치환기별 채널 ΔΔ 중앙값 + 코어 단위 부트스트랩 CI | `fingerprint_direct.csv`, `sign_consistency.csv` | 부호 일치율 중앙값 dipole 0.59 / dipolarophile 0.74 |
| 아릴(Ph 기준) | p/m/o-X-C6H4 − Ph | `aryl_lowrank.csv`, `pairs_aryl.csv` | ΔE‡ sd 1–2.5, rank1 R² 0.15–0.39 |
| 영역 | D1이 D0 8채널 공간 밖에 얼마나 있나 | `domain.json` | 2.6 % |
| 존재율(대리) | 칸별 실제/기대 TS 수 | `presence_proxy.csv` | carbonyl ylide 0.62, azomethine ylide 0.77 |

### B1b. 신뢰구간과 dipole별 분해 (`q5q7.py`, `ci_lowrank.py`)

```bash
python analysis/d1_stats/q5q7.py <labels> <results_dir>        # 수 분
python analysis/d1_stats/ci_lowrank.py 200 <results_dir>        # CPU 1코어 약 25분
```

| 스크립트 | 무엇을 묻나 | 중간 파일 결과 |
|---|---|---|
| `q5q7.py` (Q5) | dipole별 r(ΔΔE_int, ΔΔE‡), 코어 부트스트랩 CI, 강건성 3종, Δζ·Δasyn을 통제한 부분상관 | azide +0.56 [0.38, 0.61], carbonyl ylide +0.49 [0.33, 0.65], nitrone +0.05 [−0.05, 0.17], azomethine ylide −0.32 [−0.53, −0.19], nitrile oxide −0.43 [−0.58, −0.26]. 부분상관: azide 0.62, carbonyl ylide 0.71, nitrile oxide 0.05. azomethine ylide는 단계적 TS를 넣으면 +0.15로 부호가 바뀜(불안정) |
| `q5q7.py` (Q7) | regio 형제를 TS 위치가 비슷한 쌍으로 좁혔을 때 어느 채널이 승자를 맞히나. **진단용, headline에 쓰지 않음(§E)** | 전체 1,794쌍: 변형 0.83, E_int 0.39. \|Δζ\| < 0.03 Å 623쌍: 변형 0.62, E_int 0.65 |
| `ci_lowrank.py` | 저랭크 CV R²의 코어 부트스트랩 CI (같은 칸의 복제는 같은 fold) | dipole 쪽 ΔE‡: rank1 − 스칼라 +0.41 [0.26, 0.54], rank2 − rank1 +0.14 [−0.02, 0.21] (P≤0 = 0.055), rank3 − rank2 +0.06 [−0.17, 0.18]. dipolarophile 쪽은 코어가 적어 판정 불가 |

### B2. 결측 원인 (`run_all.py` + `d1_status.csv`)

```bash
python analysis/d1_stats/run_all.py --labels D1_build/d1_run/results/labels_d1.json --d0 labels_all.json \
   --out analysis/d1_stats/results_final \
   --manifest D1_build/d1_run/d1_manifest.csv --status D1_build/d1_run/results/d1_status.csv --status-col <분류 열 이름>
```

- `d1_status.csv`의 분류 열 이름을 먼저 확인해 `--status-col`로 준다(값: `ok`, `no_ts`, `gate_ts`, `stereo_mismatch`, `ref_fail`, `gate_label`, `engine_abnormal`, `infra_fail`, `pipeline`, `held`).
- 출력: `ok_rate_dipole_X.csv`, `status_by_dipole.csv`, `missing_logit.csv`(ok ~ dipole + skeleton + X + attach + series).
- **꼭 답할 것.** 중간 파일에서 azomethine ylide + 전자 부족 알켄·알카인(A, C 골격의 NO2, CN, CHO, Ac, CO2Me, CF3, Ts, SiMe3, Ph, SMe)과 carbonyl ylide의 R2 계열(A 2/80, C 13/80, D 2/80)이 거의 비어 있었다. 이 칸이 `no_ts`인지, 그중 스캔 에너지가 계속 내려가는(고개가 없는) 경우인지 센다(파일럿 T02894와 같은 판정). **고개 없음이면 "가장 반응성 큰 구석이 라벨로 존재하지 않음"을 뜻하므로 결측이 무작위가 아니다(MNAR).** 이 결론은 논문 Methods에 실패율과 함께 적는다.
- `held` 행은 실패가 아니라 미계산으로 따로 센다.

### B3. 문헌 σ로 Hammett 확인

1. `D1_설계.xlsx`(현행 버전)의 σ 표(Hansch, Leo & Taft, Chem. Rev. 1991, doi:10.1021/cr00002a004 원표 값)를 `analysis/d1_stats/sigma_hansch1991.csv`(열: `X, sigma_m, sigma_p, F, R, source`)로 옮긴다. X 이름은 D1 표기(`NMe2, NH2, OMe, OH, SMe, Me, tBu, Ph, Bn, Ac, F, Cl, Br, CN, CHO, CO2Me, CF3, NO2, SiMe3, Ts`)와 같게 한다. 원표에 없는 값은 빈칸으로 두고 채우지 않는다. `source`에 원표 쪽 번호를 적는다.
   - **워크북이 서버에 없으면 사용자에게 `sigma_hansch1991.csv`(또는 워크북)를 요청한다.** 받을 때까지 B3·B6만 보류하고 나머지는 진행한다(STOP 아님).
2. `--sigma analysis/d1_stats/sigma_hansch1991.csv`를 더해 `run_all.py`를 다시 실행 → `hammett.csv`(문맥별 ρ, R²; σ 하나 vs F/R 둘), `sigma_vs_rank2.csv`(데이터가 배운 2차원 치환기 벡터와 (F, R)의 정준상관).
3. 보고: 채널별 R²(σ) 분포, R²(F,R) − R²(σ), 정준상관 두 개. **가설:** 데이터 rank-2 벡터가 (F, R)과 높은 정준상관을 보이면 "8채널 치환기 상수 = Swain–Lupton 2축의 채널별 확장"으로 쓸 수 있다(Swain & Lupton 1968, doi:10.1021/ja01018a024). 상관이 낮으면 데이터가 F/R 밖의 축(크기·분극·입체)을 쓴다는 뜻이다. 어느 쪽이든 결과로 보고한다.
4. 선행 연구: 계산 장벽에서 σ·ρ를 데이터로 학습하는 것은 Bragato, von Rudorff & von Lilienfeld 2020(SN2, doi:10.1039/d0sc04235h)이 했다. D1에서 새로운 부분은 **채널별** σ 벡터다.

### B4. 강건성 (headline 숫자 전부)

| 변형 | 방법 |
|---|---|
| 제외 규칙 | `q_stepwise` 포함 / 제외 |
| 기준 구조 | `alt_used`가 부모·자식에서 같은 짝만 |
| 형제 | (cell, regio, face) 안에서 장벽 최저 TS 하나만 |
| carbonyl ylide | 포함 / 제외 (B8) |
| 불확실성 | 코어 단위 클러스터 부트스트랩 1,000회로 95 % CI (짝 단위 재표집 금지: 같은 코어의 짝은 독립이 아니다). 저랭크 CV R²는 `ci_lowrank.py` |
| 층별 | 모든 headline을 dipole별로도 계산해 전체값과 부호가 다른 dipole을 표시 |
| 다중 비교 | 치환기×채널 지문의 부호 검정은 Benjamini–Hochberg FDR 0.05 (doi:10.1111/j.2517-6161.1995.tb02031.x) |

`STATS_REPORT.md`에 각 headline 숫자를 "기본 / 변형별 / CI"로 한 줄씩 적는다.

### B5. Houk vs Domingo 판별 (`q2_houk_domingo.py` + GEDT + `inflection.py` 비대칭)

사용자 요청(2026-10-14): 두 주장을 판별한다.

- Chen et al. 2021(Houk, doi:10.1021/acs.joc.1c00239): phenyl azide는 양쪽성(ambiphilic). 전자 풍부·전자 부족 알켄 모두 가속(Sustmann 포물선). 반응성은 변형·전하 이동·Pauli·분극·정전기의 조합. 장벽은 CPCM(CCl4·benzene), EDA는 기체상.
- Domingo, Ríos-Gutiérrez & Pérez 2023(doi:10.1039/d3qo00811h): phenyl azide는 온건한 친전자체. 초친핵성 알켄(vinyl amine)만 크게 가속(평균 −9.9 kcal/mol; nitroethylene −3.4, acrylonitrile −0.7). 전자 부족 알켄과는 regio가 약하다. 핵심 변수는 GEDT. 기체상.
- D1은 물(SMD), B3LYP. D1에는 phenyl azide + 치환 알켄이 없다(한쪽씩만 교체) → HN3 + 비닐-X가 대리.

```bash
python analysis/d1_stats/q2_houk_domingo.py <labels> <results_dir>
```

| 검정 | 방법 | 중간 파일 결과 |
|---|---|---|
| 비대칭 (도너 vs 억셉터) | HN3 + 비닐-X: NMe2·NH2 vs NO2·CN의 ΔΔE‡(각 X의 최저 TS), 전체와 용매화 채널(CPCM+CDS)을 뺀 값 | 골격 A: 전체 도너 −5.3 / 억셉터 −4.7 (대칭, Houk 쪽); 용매화 제외 도너 −4.2 / 억셉터 −0.6 (비대칭, Domingo 쪽). 골격 B: −6.2 / −3.4 → −4.4 / +1.0 |
| regio 크기 | \|ΔE‡(a−b)\| 중앙값 | azide: NMe2 10.2, NH2 8.2 vs CN 2.9, CO2Me 2.4, CF3 0.8 → Domingo의 regio 주장과 일치 |
| 변형 vs 상호작용 | azide, dipolarophile 쪽 치환기 | r(ΔΔ변형, ΔΔE‡) 0.86, r(ΔΔE_int, ΔΔE‡) 0.14 → Chen 2021의 "상호작용은 반응성과 거의 무상관"과 일치 |
| **GEDT (Domingo 핵심)** | 기존 `eda.out`의 복합체 원자 전하(Mulliken·Löwdin)를 조각별로 합해 GEDT를 만든다(**새 QM 없음**, 조각 원자 분할은 라벨과 같은 `fragmenter.partition`). 출력에 전하가 없으면 Phase C의 xTB B4 블록 GEDT를 쓴다. Domingo는 NPA 전하를 쓰므로 방법 차이를 명시. 출력: `gedt_d1.csv`(`job_id, gedt_mulliken, gedt_loewdin, gedt_xtb`) | — |
| GEDT 판정 | (1) ΔΔE‡ ~ \|ΔGEDT\| vs ΔΔE‡ ~ ΔΔ변형: dipole별 부분상관. (2) GEDT 부호(REDF: dipolarophile → dipole / FEDF: 반대)별 기울기 비대칭. (3) 용매화 제외 장벽으로도 같은 분석 | — |
| σ 비대칭 | `inflection.py`의 꺾임점 σ = 0 고정 기울기(`asymmetry_slopes.csv`), 전체 장벽과 `nosolv` 각각 | σ 표 필요 (B3) |

- 해석 가설(사전 기록): "Houk–Domingo 차이의 전자 부족 쪽 가지는 용매화 채널에 실려 있다." 용매화 제외 값은 기체상 장벽이 아니라 물 기하에서의 분해이므로 가설 수준으로만 쓴다.
- 판정 문구는 세 가지 중 하나로 쓴다: "D1(물)은 Houk 쪽", "용매화 제외 시 Domingo 쪽", "GEDT가 변형보다 장벽을 더 잘 설명함/못함". 제약(HN3 대리, 범함수, 용매)을 같은 문단에 적는다.

### B6. EDG→EWG 순서의 채널 변곡점 (`inflection.py`, σ 표 필요)

사용자 질문(2026-10-14): 치환기를 EDG → EWG 순서로 놓으면, Hammett(장벽 하나)에서는 안 보이던 변곡점이 채널에서 보이는가.

```bash
python analysis/d1_stats/inflection.py <labels> <results_dir> analysis/d1_stats/sigma_hansch1991.csv <results_dir>/inflection
```

- 계열(dipole × 골격 × 쪽 × 부착 × 문맥)마다 치환기를 σ 순서로 놓고(직결·para는 σp, meta는 σm, ortho 제외) 선형 · 꺾은선(꺾임점 1개) · 이차를 맞춘다. 변곡 판정: AICc가 4 이상 낮고 LOO 오차도 낮을 때(사전 고정). 치환기 12개 이상인 계열만. 중간 파일 기준 315개 계열.
- 핵심 지표: 채널별 변곡 비율, **장벽은 선형인데 채널이 꺾이는 비율**(`hidden_frac_vs_barrier`), 꺾임점 σ 분포(σ ≈ 0, 즉 전자 흐름이 뒤집히는 지점에 모이는지 → B5의 GEDT 부호와 연결).
- 거짓 양성 기준: σ를 섞은 순열의 판정 비율(`null_rate_permutation`). 무작위 σ 시험에서 약 1 %였다. 관측 비율을 이 값과 나란히 보고한다.
- B8에 따라 치환기 범위가 한쪽만 남은 계열(예: carbonyl ylide R2 계열)은 해석에서 뺀다.

### B7. 메틸 치환 패턴과 cis/trans (`q5_methyl.py`)

```bash
python analysis/d1_stats/q5_methyl.py <labels> <results_dir>
```

- 같은 dipole·같은 R1에서 골격만 바꾼 차이: gem-Me2 알켄(Ap − A), trans/cis-2-butene(AppE/AppZ − A), cis − trans, 2-butyne − acetylene, 2,3-diMe-norbornene − norbornene, 3,3-diMe-cyclooctyne − cyclooctyne, 3,8-diMe(cis/trans) − cyclooctyne. 각 (dipole, R1, 골격)의 최저 장벽 TS를 쓴다. R1 81종이 반복 단위(부트스트랩 CI).
- gem-Me2의 regio(Ap, Dp): 어느 방향이 이기고 변형/상호작용 중 무엇이 고르는지.

| 중간 파일 결과 (ΔE‡ 중앙값) | azide | azomethine ylide | carbonyl ylide | nitrile oxide | nitrone |
|---|---|---|---|---|---|
| gem-Me2 알켄 − 에틸렌 | +0.8 | +3.5 | −0.8 | +0.3 | +1.4 |
| cis − trans 2-butene | −0.5 | −1.1 | +0.2 | −0.2 | −0.4 (CI 0 포함) |
| 2-butyne − acetylene | +2.7 | +4.7 | +2.6 | +3.3 | +3.0 |
| 2,3-diMe-norbornene − norbornene | +0.7 | +2.9 | −1.6 | +1.0 | +1.6 |

- 메틸은 변형을 늘리고(+0.5 ~ +5.8) 상호작용을 일부 되돌린다(−0.1 ~ −3.2, 분산이 큰 몫). azomethine ylide만 Pauli +13 ~ +17.
- gem-Me2 regio(Ap): 5개 dipole 모두 방향 a 선호 87–100 %. 고르는 채널: nitrile oxide·nitrone은 변형(100 %), azide는 상호작용(93 %).
- 3,8-diMe-cyclooctyne(cis/trans)은 짝 9–25개 → 확정 데이터에서도 적으면 "판정 불가"로 보고.

### B8. 결측 처리 방침 (`q4_missing.py`)

```bash
python analysis/d1_stats/q4_missing.py <labels> <results_dir>
```

- carbonyl ylide를 지우지 않는다. 있는 라벨은 게이트를 통과한 유효 값이고, 지워도 다른 dipole의 편향은 그대로다.
- 대신 (1) 모든 headline을 carbonyl ylide 포함/제외로 함께 보고, (2) 치환기 범위가 한쪽만 남은 계열은 B6 변곡 분석과 저랭크 분석에서 빼고, (3) 결측을 MNAR로 Methods에 보고한다(Rubin 1976, doi:10.1093/biomet/63.3.581).
- 중간 파일 결과: carbonyl ylide 존재율 57 %(1,339 / 2,334), 있는 행의 86 %가 음의 전자 장벽(중앙값 −3.9), H 부모가 있는 코어 4/12(짝 560). 빼도 headline 거의 불변(r(변형) 0.917 → 0.921, regio 변형 정답률 0.831 → 0.828).

**게이트 B:** 기본 결과와 강건성 변형의 부호가 모든 headline에서 같다. 부호가 바뀌는 항목은 STOP 없이 목록으로 보고한다(결론 문구를 약하게 바꿔야 할 대상).

---

## Phase C — G1 xTB feature (D1 전 행)

- rev5와 **같은 코드·같은 버전**으로 만든다: `analysis/espley_xtb_repro/g1_geom.py`(G1 = DFT TS·참조 구조에서 출발한 GFN2-xTB/ALPB(water) 기하), feature 추출(ESPLEY73 + B1 + B4 = EXT_SEL 133열). 2026-10-13 결정대로 DFT TS에서 출발하는 이 조건 하나만 쓴다(SMILES→TS 배포 경로 없음).
- 입력 기하: `D1_build/d1_run/results/geometries/<job_id>/`의 `ts.xyz`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`. 조각 분할은 `fragmenter.partition`(Si 추가판).
- xtb 6.7.1, tblite 0.7.0. 경로는 `rev5_common.py`에 D1용 상수를 **추가**한다(기존 상수는 건드리지 않음).
- 출력: `xtb_features_d1_g1.parquet`(`job_id` 키), `xtb_features_ext_d1_g1.parquet`. B4 블록의 GEDT 열은 B5의 대체 GEDT로도 쓴다.
- **게이트 C:** (i) 열 이름·순서가 `rows_rev5`의 EXT_SEL 열과 같음, (ii) feature 결측 행 ≤ 2 %, 실패 사유 표, (iii) D0 대조 2건(C0020, C0105)을 같은 스크립트로 다시 뽑아 기존 `xtb_features_g1.parquet` 값과 최대 차이 ≤ 1e-6. 어긋나면 STOP.

---

## Phase D — 모델 (사전 등록 후)

### D0. 공통

- 타깃: 8채널 + ΔE‡. 모델: rev5와 같은 `KRR_rbf`(GridSearchCV, rev 4 grid, X·y 표준화). 부록으로 Ridge.
- 지표(모두 train/test 같이):
  - 절대값 MAE, NMAE
  - **짝 지표:** 같은 test 문맥 안에서 ΔΔ̂ = ŷ(자식) − ŷ(부모) vs 참 ΔΔ의 MAE, 부호 일치율, 문맥 안 Spearman ρ
  - **선택성 지표:** regio 형제·exo/endo 형제에서 이긴 TS를 맞춘 비율
- 분할(그룹 단위, 짝과 형제는 함께):
  - `LOCO`: D1 코어 하나씩 빼기 (코어 = dipole × 골격)
  - `LODO`: dipole 하나씩 빼기
  - `LOXO`: 치환기 X 하나씩 빼기 (직결 20종; 아릴 p/m/o-X도 같은 X로 묶음)
- 결과는 carbonyl ylide 포함/제외를 함께 적는다(B8).

### 실험

| id | 학습 | 예측 | 묻는 것 |
|---|---|---|---|
| **M0** | D0 dev 4,112 (rev5 그대로, 재학습 없음) | D1 전체 | D0 모델이 치환기 추세(ΔΔ, 선택성)를 이미 아는가. **절대 MAE와 짝 지표를 따로 본다** |
| **M1** | D0 dev + D1 학습 fold | D1 test fold | 합쳐 학습하면 짝 지표가 얼마나 좋아지나 (단일 단계) |
| **M2** | 짝 Δ 모델: 입력 = [x(자식) − x(부모), x(부모)], 타깃 = ΔΔ. 예측 = M1(부모) + Δ̂ | 같은 test 자식 | 차이를 직접 배우는 것이 M1보다 나은가 (Tynes 2021 doi:10.1021/acs.jcim.1c00670, Fralish 2023 doi:10.1186/s13321-023-00769-x) |
| **M3** | 학습 곡선: D1 코어를 무작위 순서로 늘려 가며 M1 (5 seed) | LOCO test | D1을 더 만들 가치가 있나 |
| **M4** | M1 최종 모델 | D0 lockbox 726 (1회) | D1을 넣어도 D0 성능이 나빠지지 않나 |

- **판정 기준(사전 등록에 숫자로 적을 것):** M2가 M1보다 짝 MAE에서 코어 부트스트랩 95 % CI가 0 아래면 "M2 채택". M4에서 EXT_SEL rev5 lockbox MAE 대비 악화가 CI로 0을 넘으면 보고하고 멈춘다.
- **주의:** B1에서 "치환기 상수 하나로는 이식되지 않음"(코어 하나를 빼고 예측한 스칼라 R²: dipole 쪽 elst·Pauli·OI·ΔE‡ 0.08–0.13, dipolarophile 쪽 elst·Pauli·OI 0 이하)이 나왔으므로 M0의 짝 지표가 낮게 나올 가능성이 높다. 그것 자체가 D1이 필요한 이유이므로 그대로 보고한다.

---

## §E — 일정 기하 계산: 취소 (2026-10-14)

- 사용자 결정: TS 위치를 맞춘(일정 기하) 데이터는 만들지 않는다. 합성 화학자에게 유용한 정보는 실제 TS에서의 분해이기 때문이다(2026-09-17 결정 유지). 이전에 보낸 `SPEC_D1_PHASE_E.md`, `phaseE_pairs.csv`는 실행하지 않는다.
- TS 한 점 분석의 한계(Vermeeren et al. 2020, doi:10.1038/s41596-019-0265-0)는 Methods에 한 문장으로만 적는다.
- `q5q7.py`의 Q7은 진단용으로만 남긴다.

---

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `analysis/d1_stats/*.py` | 부록 A의 코드 10개 |
| `analysis/d1_stats/reference_interim/headline.json` | 부록 B |
| `analysis/d1_stats/results_interim/` | Phase R |
| `analysis/d1_stats/results_final/` | Phase A·B 출력 전부 + `INPUTS.md` + `inflection/` + `gedt_d1.csv` |
| `analysis/d1_stats/sigma_hansch1991.csv` | B3 σ 표(원표 쪽 번호 포함) |
| `analysis/d1_stats/STATS_REPORT.md` | headline 표(기본/강건성/CI), B2 결측, B3 Hammett, B5 판별, B6 변곡, B7 메틸, B8 민감도 |
| `analysis/d1_stats/PREREG_D1.md` | Phase D 사전 등록 (결과 전 commit) |
| `analysis/d1_stats/MODEL_REPORT.md` | M0–M4 |
| `xtb_features_d1_g1.parquet` 등 | scratch에 두고 sha256만 commit |

## 8. 보고 순서

1. Phase S·R: 설치 확인, 재현 일치표
2. 확정 데이터 크기, 제외 건수, 게이트 A
3. B1·B1b headline 표: 중간값 → 확정값, 강건성, CI, dipole별
4. B2 결측: 분류표, 로지스틱 계수 상위, azomethine ylide·carbonyl ylide 빈칸의 원인
5. B3 Hammett (σ 표를 못 받았으면 보류 사실)
6. B5 Houk vs Domingo 판정 (세 문구 중 하나 + 제약)
7. B6 변곡점 (관측 비율 vs 순열 기준, hidden 비율, 꺾임점 위치)
8. B7 메틸·cis/trans, B8 carbonyl ylide 포함/제외 민감도
9. C 게이트
10. D 표 (M0–M4, train/test 함께)
11. 한계와 미해결

## 9. STOP 조건

| 시점 | 조건 |
|---|---|
| S | sha256 불일치 |
| R | headline 불일치, 중간 파일을 찾을 수 없고 사용자도 줄 수 없음 |
| A | 게이트 A 실패, 확정 파일의 열 구성이 중간 파일과 다름 |
| B | `--status-col`을 정할 수 없음. (σ 표가 없으면 STOP하지 않고 B3·B6만 보류하고 보고) |
| C | 게이트 C 실패 |
| D | 사전 등록 전 결과를 본 경우, lockbox를 두 번 이상 연 경우 |
| 언제든 | QC 임계값·저랭크 설정·변곡 판정·모델 grid를 바꾸고 싶어짐, 새 QM 계산(일정 기하·저극성 용매 등)을 하고 싶어짐 |

STOP하면: 무엇이 어긋났는지, 해당 파일·줄, 선택지를 적어 보고하고 기다린다.

## 10. 참고 문헌 (DOI 확인 완료)

- Bickelhaupt & Houk, Angew. Chem. Int. Ed. 56, 10070 (2017). doi:10.1002/anie.201701486
- Vermeeren, van der Lubbe, Fonseca Guerra, Bickelhaupt & Hamlin, Nat. Protoc. 15, 649 (2020). doi:10.1038/s41596-019-0265-0
- Ess & Houk, J. Am. Chem. Soc. 130, 10187 (2008). doi:10.1021/ja800009z
- Chen, Ma, He, Svatunek, Liu & Houk, J. Org. Chem. 86, 5792 (2021). doi:10.1021/acs.joc.1c00239
- Domingo, Ríos-Gutiérrez & Pérez, Org. Chem. Front. 10, 5579 (2023). doi:10.1039/d3qo00811h
- Houk, Sims, Duke, Strozier & George, J. Am. Chem. Soc. 95, 7287 (1973). doi:10.1021/ja00803a017
- Hansch, Leo & Taft, Chem. Rev. 91, 165 (1991). doi:10.1021/cr00002a004
- Swain & Lupton, J. Am. Chem. Soc. 90, 4328 (1968). doi:10.1021/ja01018a024
- Bragato, von Rudorff & von Lilienfeld, Chem. Sci. 11, 11859 (2020). doi:10.1039/d0sc04235h
- Free & Wilson, J. Med. Chem. 7, 395 (1964). doi:10.1021/jm00334a001
- Rubin, Biometrika 63, 581 (1976). doi:10.1093/biomet/63.3.581
- Benjamini & Hochberg, J. R. Stat. Soc. B 57, 289 (1995). doi:10.1111/j.2517-6161.1995.tb02031.x
- Tynes et al., J. Chem. Inf. Model. 61, 3846 (2021). doi:10.1021/acs.jcim.1c00670
- Fralish, Chen, Skaluba & Reker, J. Cheminform. 15, 101 (2023). doi:10.1186/s13321-023-00769-x
- van Tilborg, Alenicheva & Grisoni, J. Chem. Inf. Model. 62, 5938 (2022). doi:10.1021/acs.jcim.2c01073

---

## 부록 A — 분석 코드 10개 (Phase S의 추출 스크립트로 저장)

실행 순서: `run_all.py` → (`figs.py`) → `q5q7.py`, `ci_lowrank.py`, `q2_houk_domingo.py`, `q5_methyl.py`, `q4_missing.py` → σ 표가 있으면 `inflection.py`. 앞의 스크립트 출력(`pairs_H.csv`, `sel_regio.csv` 등)을 뒤 스크립트가 읽는다. 각 스크립트 첫머리의 설명(docstring)에 사용법이 있다.

### `analysis/d1_stats/d1s_prep.py` — QC 표식, regio 방향(RDKit), 짝 만들기 — 다른 모든 스크립트가 import

<!-- FILE: analysis/d1_stats/d1s_prep.py sha256=283de371d381341238dfbd2578aef9b9ea82dc09db2f9bcce7be15ad48bd9fd0 -->
```python
"""d1s_prep.py — D1 라벨 로드, QC 표식, regio 방향(RDKit), H 부모 짝 / Ph 기준 아릴 짝 생성.

모든 함수는 순수 함수다(파일을 쓰지 않음). run_all.py가 호출한다.
에너지 단위 kcal/mol, 길이 Å.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CH = ["d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
BUD = CH + ["bsse_shift_kcal"]          # 9 terms; Σ = barrier (closure ≤ 0.11)
CHN = {"d1_kcal": "d1", "d2_kcal": "d2", "elst_dft": "elst", "pauli_dft": "Pauli", "oi_dft": "OI",
       "disp_dft": "disp", "cpcm_dft": "CPCM", "cds_dft": "CDS", "bsse_shift_kcal": "bsse", "barrier_kcal": "ΔE‡"}
VALS = CH + ["bsse_shift_kcal", "barrier_kcal", "eint_spe_kcal", "e_bond_kcal", "zeta", "asyn"]

# QC thresholds (pre-registered; do not change after seeing results)
Q_BADREF_KCAL = -5.0      # d1 or d2 below this -> broken reference structure
Q_MIN_FORMED = 1.65       # shorter forming bond below this -> product-like / stepwise-like
Q_MAX_FORMED = 3.0        # longer forming bond above this -> stepwise-like (= flag_async)


def load_labels(path: str) -> pd.DataFrame:
    d = json.load(open(path))
    rows = []
    for r in d:
        x = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        x["gates_ok"] = bool(all((r.get("gates") or {"missing": False}).values()))
        for f, v in (r.get("flags") or {}).items():
            if f != "audit_flags":
                x[f if f.startswith("flag_") else "flag_" + f] = v
        rows.append(x)
    del d
    df = pd.DataFrame(rows)
    df["face"] = df["face"].fillna("")
    df["stereo_index"] = df["stereo_index"].astype(str)
    df["zeta"] = (df.formed_d1 + df.formed_d2) / 2
    df["asyn"] = (df.formed_d1 - df.formed_d2).abs()
    df["side"] = np.where(df.series == "R1", "dipole", "dipolarophile")
    df["R"] = np.where(df.series == "R1", df.R1, df.R2)
    df["X"] = df.R.str.replace(r"^[pmo]-", "", regex=True)
    df["R2p"] = np.where(df.series == "R2", "H", df.R2)
    df["is_parent"] = (df.R1 == "H") & (df.R2.isin(["H", "-"]))
    mf = np.minimum(df.formed_d1, df.formed_d2)
    Mf = np.maximum(df.formed_d1, df.formed_d2)
    df["q_badref"] = (df.d1_kcal < Q_BADREF_KCAL) | (df.d2_kcal < Q_BADREF_KCAL)
    df["q_stepwise"] = (mf < Q_MIN_FORMED) | (Mf > Q_MAX_FORMED)
    df["q_bad"] = df.q_badref | df.q_stepwise
    df["closure"] = df[BUD].sum(axis=1) - df.barrier_kcal
    df["orient"] = ""
    need = df.regio.isin(["r1/2", "r2/2"])
    df.loc[need, "orient"] = [orientation(s, dp) for s, dp in zip(df.loc[need, "rxn_smiles"], df.loc[need, "dipole"])]
    return df


# ---------------------------------------------------------------- regio orientation
def _new_ring(m, dip):
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        s = set(ring)
        el = [m.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if dip in ("nitrile oxide", "nitrone"):
            for i in ring:
                a = m.GetAtomWithIdx(i)
                if a.GetSymbol() == "O" and any(nb.GetSymbol() == "N" and nb.GetIdx() in s for nb in a.GetNeighbors()):
                    return ring
        elif dip == "azide":
            for i in ring:
                a = m.GetAtomWithIdx(i)
                if a.GetSymbol() == "N" and sum(nb.GetIdx() in s and nb.GetSymbol() == "N" for nb in a.GetNeighbors()) == 2:
                    return ring
        elif dip == "azomethine ylide" and "N" in el:
            return ring
        elif dip == "carbonyl ylide" and "O" in el:
            return ring
    return None


def _partition(m, ring, dip):
    s = set(ring)
    A = m.GetAtomWithIdx
    rnb = lambda i: [nb.GetIdx() for nb in A(i).GetNeighbors() if nb.GetIdx() in s]
    if dip in ("nitrile oxide", "nitrone"):
        O = [i for i in ring if A(i).GetSymbol() == "O" and any(A(j).GetSymbol() == "N" for j in rnb(i))][0]
        N = [j for j in rnb(O) if A(j).GetSymbol() == "N"][0]
        Cb = [j for j in rnb(N) if j != O][0]
        return [O, N, Cb], O
    if dip == "azide":
        mid = [i for i in ring if A(i).GetSymbol() == "N" and sum(A(j).GetSymbol() == "N" for j in rnb(i)) == 2][0]
        t = rnb(mid)
        sub = [i for i in t if A(i).GetTotalNumHs() > 0 or any(nb.GetIdx() not in s for nb in A(i).GetNeighbors())]
        return [mid] + t, (sub[0] if len(sub) == 1 else None)
    c = [i for i in ring if A(i).GetSymbol() == ("N" if dip == "azomethine ylide" else "O")][0]
    t = rnb(c)
    hv = [i for i in t if any(nb.GetIdx() not in s and nb.GetAtomicNum() > 1 for nb in A(i).GetNeighbors())]
    return [c] + t, (hv[0] if len(hv) == 1 else None)


def orientation(rxn_smiles: str, dip: str) -> str:
    """'a' = dipolarophile ring atom nearest the anchor is bonded to terminus a
    (O for nitrile oxide/nitrone; substituted N1 for azide; R1-bearing C for ylides)."""
    try:
        m = Chem.MolFromSmiles(rxn_smiles.split(">>")[1])
        ring = _new_ring(m, dip)
        if ring is None:
            return "noring"
        dset, a = _partition(m, ring, dip)
        if a is None:
            return "sym"
        dpo = [i for i in ring if i not in dset]
        dm = Chem.GetDistanceMatrix(m)
        ds = set(dset)
        cands = []
        for at in m.GetAtoms():
            i = at.GetIdx()
            if i in ds:
                continue
            for bd in at.GetBonds():
                o = bd.GetOtherAtom(at)
                if not bd.IsInRing() and o.GetAtomicNum() > 1 and o.GetIdx() not in ds:
                    cands.append(i)
                    break
        if not cands:
            return "noanchor"
        best = min(cands, key=lambda c: min(dm[c, dpo[0]], dm[c, dpo[1]]))
        d0, d1 = dm[best, dpo[0]], dm[best, dpo[1]]
        if d0 == d1:
            return "tie"
        prox = dpo[0] if d0 < d1 else dpo[1]
        return "a" if m.GetBondBetweenAtoms(prox, a) is not None else "b"
    except Exception as e:  # noqa: BLE001
        return "err:" + type(e).__name__


# ---------------------------------------------------------------- pairs
def pairs_vs_H(df: pd.DataFrame) -> pd.DataFrame:
    """child − H parent (same core, same fixed R2, same face; orientation if parent has two)."""
    par = df[df.is_parent]
    pg = {k: v for k, v in par.groupby(["core", "R2p", "face"])}
    out = []
    for _, r in df[~df.is_parent].iterrows():
        p = pg.get((r.core, r.R2p, r.face))
        if p is None:
            continue
        if len(p) > 1 and r.orient in ("a", "b") and r.orient in set(p.orient):
            p = p[p.orient == r.orient]
        if len(p) != 1:
            continue
        p = p.iloc[0]
        x = {c: r[c] - p[c] for c in VALS}
        x.update(job_id=r.job_id, parent_job=p.job_id, core=r.core, dipole=r.dipole, skeleton=r.skeleton,
                 side=r.side, R=r.R, X=r.X, attach=r.attach, regio=r.regio, orient=r.orient, face=r.face,
                 stereo_index=r.stereo_index, cell_id=r.cell_id,
                 q_bad=bool(r.q_bad or p.q_bad))
        out.append(x)
    P = pd.DataFrame(out)
    P["ctx"] = P.core + "|" + P.orient + "|" + P.face + "|" + P.stereo_index
    return P


def pairs_vs_Ph(df: pd.DataFrame) -> pd.DataFrame:
    """aryl (p-/m-/o-X-C6H4) − Ph in the same core/series/face/orientation/stereo slot (clean rows only)."""
    d = df[~df.q_bad]
    key = ["core", "series", "R2p", "face", "orient", "stereo_index"]
    ph = {k: v for k, v in d[d.R == "Ph"].groupby(key)}
    out = []
    for _, r in d[d.attach.isin(["p", "m", "o"])].iterrows():
        k = tuple(r[c] for c in key)
        p = ph.get(k)
        if p is None or len(p) != 1:
            continue
        p = p.iloc[0]
        x = {c: r[c] - p[c] for c in VALS}
        x.update(job_id=r.job_id, ref_job=p.job_id, core=r.core, dipole=r.dipole, skeleton=r.skeleton,
                 side=r.side, X=r.X, pos=r.attach, ctx="|".join(map(str, k)))
        out.append(x)
    return pd.DataFrame(out)


def sibling_pairs(df: pd.DataFrame, key, col, A, B) -> pd.DataFrame:
    """A − B inside groups that hold exactly one clean A row and one clean B row."""
    d = df[~df.q_bad]
    out = []
    for _, g in d.groupby(key):
        a, b = g[g[col] == A], g[g[col] == B]
        if len(a) == 1 and len(b) == 1:
            a, b = a.iloc[0], b.iloc[0]
            x = {c: a[c] - b[c] for c in VALS}
            x.update(job_a=a.job_id, job_b=b.job_id, core=a.core, dipole=a.dipole, skeleton=a.skeleton,
                     series=a.series, R=a.R, X=a.X, attach=a.attach)
            out.append(x)
    return pd.DataFrame(out)


def budget(D: pd.DataFrame, target: str = "barrier_kcal") -> dict:
    v = D[target].var()
    return {CHN[c]: D[c].cov(D[target]) / v for c in BUD}
```

### `analysis/d1_stats/d1s_lowrank.py` — 저랭크(σ·ρ) 적합과 교차검증

<!-- FILE: analysis/d1_stats/d1s_lowrank.py sha256=b4f0debfa46171782acfc73af955466188e3523ca0bc62f4f0160cbbcc1621f9 -->
```python
"""d1s_lowrank.py — 결측이 있는 (context × substituent) 행렬의 저랭크(σ·ρ) 적합과 교차검증.

ΔΔ는 H에서 0이므로 절편 없는 M ≈ U Vᵀ (rank k)가 Hammett형 σ_X·ρ_ctx 모형이다(k = 1).
"""
import numpy as np

LAM = 1.0      # ridge on factors (fixed; pre-registered)
ITERS = 60
NFOLD = 5
SEED = 1


def als(M, mask, k, lam=LAM, iters=ITERS, seed=0):
    n, m = M.shape
    if k == 0:
        return np.zeros_like(M), None, None
    rng = np.random.default_rng(seed)
    U = rng.normal(scale=0.1, size=(n, k))
    V = rng.normal(scale=0.1, size=(m, k))
    Mz = np.where(mask, M, 0.0)
    I = lam * np.eye(k)
    for _ in range(iters):
        for i in range(n):
            o = mask[i]
            U[i] = np.linalg.solve(V[o].T @ V[o] + I, V[o].T @ Mz[i, o])
        for j in range(m):
            o = mask[:, j]
            V[j] = np.linalg.solve(U[o].T @ U[o] + I, U[o].T @ Mz[o, j])
    return U @ V.T, U, V


def additive(M, mask):
    n, m = M.shape
    a, b = np.zeros(n), np.zeros(m)
    mu = M[mask].mean()
    for _ in range(100):
        a = np.nan_to_num(np.nanmean(np.where(mask, M - mu - b[None, :], np.nan), 1))
        b = np.nan_to_num(np.nanmean(np.where(mask, M - mu - a[:, None], np.nan), 0))
    return mu + a[:, None] + b[None, :]


def subst_only(M, mask):
    b = np.nan_to_num(np.nanmean(np.where(mask, M, np.nan), 0))
    return np.tile(b, (M.shape[0], 1))


def cv(M, mask, fitter, nfold=NFOLD, seed=SEED):
    """mask된 칸을 무작위 5등분해 하나씩 가리고 예측. 반환: (CV R², CV MAE)."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    yall = M[mask]
    sse = sst = ae = 0.0
    cnt = 0
    for f in np.array_split(idx, nfold):
        tr = mask.copy()
        tr[f[:, 0], f[:, 1]] = False
        pred = fitter(M, tr)
        y = M[f[:, 0], f[:, 1]]
        p = pred[f[:, 0], f[:, 1]]
        sse += ((y - p) ** 2).sum()
        sst += ((y - yall.mean()) ** 2).sum()
        ae += np.abs(y - p).sum()
        cnt += len(y)
    return 1 - sse / sst, ae / cnt


def matrix(P, value, index="ctx", columns="R"):
    piv = P.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    M = piv.values
    mask = ~np.isnan(M)
    return np.nan_to_num(M), mask, piv
```

### `analysis/d1_stats/run_all.py` — B1 전부 + B2(--manifest/--status) + B3(--sigma)

<!-- FILE: analysis/d1_stats/run_all.py sha256=b81f3beef65611e39fd845733f1d5cb1bb5f5329796304089873fb6c44a2b713 -->
```python
"""run_all.py — D1 통계 분석 일괄 실행 (QM 없음, 수 분).

python run_all.py --labels LABELS_D1.json --d0 labels_all.json --out OUTDIR
                  [--manifest d1_manifest.csv --status d1_status.csv --status-col CLS]
                  [--sigma sigma_hansch1991.csv]

출력(OUTDIR): qc.json, pairs_H.csv, pairs_aryl.csv, sel_regio.csv, sel_exoendo.csv,
  budget.csv, zeta.csv, corr_dd.csv, transfer.csv, lowrank.csv, joint_lowrank.csv, pca.csv,
  selectivity.csv, regio_known.csv, fingerprint_direct.csv, sign_consistency.csv, presence_proxy.csv,
  aryl_lowrank.csv, domain.json, [missing_logit.csv], [hammett.csv, sigma_vs_rank2.csv], headline.json
기존 파일을 덮어쓴다(분석은 결정론적: seed 고정).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1s_prep import *          # noqa: F401,F403
from d1s_lowrank import als, additive, subst_only, cv, matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--d0", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--manifest"); ap.add_argument("--status"); ap.add_argument("--status-col", default="cls")
ap.add_argument("--sigma")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
H = {}
def w(df, name, index=True):
    df.to_csv(OUT / name, index=index)

# ------------------------------------------------------------- A. load + QC
df = load_labels(a.labels)
qc = dict(n_rows=len(df), n_parent=int(df.is_parent.sum()),
          q_badref=int(df.q_badref.sum()), q_stepwise=int(df.q_stepwise.sum()), q_bad=int(df.q_bad.sum()),
          closure_max=float(df.closure.abs().max()),
          orient_counts=df.orient.value_counts().to_dict(),
          n_gates_failed=int((~df.gates_ok).sum()),
          n_dipole=df.dipole.value_counts().to_dict(), n_skeleton=df.skeleton.value_counts().to_dict(),
          barrier_neg_frac=float((df.barrier_kcal < 0).mean()),
          bad_rows=df.loc[df.q_bad, ["job_id", "ts_id", "d1_kcal", "d2_kcal", "formed_d1", "formed_d2", "barrier_kcal"]].round(3).to_dict("records"))
json.dump(qc, open(OUT / "qc.json", "w"), indent=1, ensure_ascii=False, default=str)
H.update(n_rows=len(df), q_bad=qc["q_bad"])

# ------------------------------------------------------------- B. pairs
P = pairs_vs_H(df); P.to_csv(OUT / "pairs_H.csv", index=False)
Pc = P[~P.q_bad].copy()
H.update(n_pairs=len(P), n_pairs_clean=len(Pc), n_children_no_parent=int((~df.is_parent).sum() - len(P)))
big = Pc[Pc.barrier_kcal.abs() > 15][["job_id", "parent_job", "core", "R", "barrier_kcal", "d1_kcal", "d2_kcal"]]
w(big, "inspect_large_ddbarrier.csv", index=False)

# ------------------------------------------------------------- C. covariance budget
rows = []
def add(name, D):
    b = budget(D); b.update(set=name, n=len(D), sd_dE=D.barrier_kcal.std()); rows.append(b)
add("ALL pairs", Pc)
for s in ["dipole", "dipolarophile"]:
    add(f"side={s}", Pc[Pc.side == s])
    for at in ["direct", "p", "m", "o"]:
        add(f"side={s}, attach={at}", Pc[(Pc.side == s) & (Pc.attach == at)])
for dp in sorted(Pc.dipole.unique()):
    add(f"dipole={dp}", Pc[Pc.dipole == dp])
add("D1 absolute (between all TS)", df[~df.q_bad])
d0 = pd.DataFrame(json.load(open(a.d0)))
d0 = d0[(~d0.flag_foreign_bond.astype(bool)) & (d0.oi_dft < 0)].copy()
d0["bsse_shift_kcal"] = d0.eint_spe_kcal - d0.e_bond_kcal
add("D0 absolute (between all TS)", d0)
B = pd.DataFrame(rows).set_index("set"); w(B, "budget.csv")
H["budget_all_pairs"] = {k: round(float(B.loc["ALL pairs", k]), 3) for k in ["d1", "d2", "elst", "Pauli", "OI", "disp", "CPCM", "CDS", "bsse"]}
Pc["dstrain"] = Pc.d1_kcal + Pc.d2_kcal
Pc["dint"] = Pc[["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]].sum(axis=1)
H["r_strain_barrier"] = round(float(np.corrcoef(Pc.dstrain, Pc.barrier_kcal)[0, 1]), 3)
H["r_eint_barrier"] = round(float(np.corrcoef(Pc.dint, Pc.barrier_kcal)[0, 1]), 3)

# ------------------------------------------------------------- D. TS-position coupling
T = CH + ["bsse_shift_kcal", "barrier_kcal"]
X1, X2 = Pc[["zeta"]].values, Pc[["zeta", "asyn"]].values
zr = []
for c in T:
    y = Pc[c].values
    m1, m2 = LinearRegression().fit(X1, y), LinearRegression().fit(X2, y)
    zr.append(dict(ch=CHN[c], r_dzeta=np.corrcoef(Pc.zeta, y)[0, 1], slope_per_0p1A=m1.coef_[0] * 0.1,
                   R2_dzeta=m1.score(X1, y), R2_dzeta_dasyn=m2.score(X2, y), sd=y.std()))
Z = pd.DataFrame(zr).set_index("ch"); w(Z, "zeta.csv")
H["r_dzeta_barrier"] = round(float(Z.loc["ΔE‡", "r_dzeta"]), 3)
w(Pc[CH].rename(columns=CHN).corr(), "corr_dd.csv")

# ------------------------------------------------------------- E. transferability + low rank
lr = []
for side in ["dipole", "dipolarophile"]:
    for lab, sub in [("direct", Pc[(Pc.side == side) & (Pc.attach == "direct")]), ("all80", Pc[Pc.side == side])]:
        for c in T:
            if c == "bsse_shift_kcal":
                continue
            M, mask, piv = matrix(sub, c)
            r = dict(side=side, set=lab, ch=CHN[c], n_ctx=M.shape[0], n_sub=M.shape[1], fill=mask.mean(), zero_MAE=np.abs(M[mask]).mean())
            r["R2_subst_only"], r["MAE_subst_only"] = cv(M, mask, subst_only)
            r["R2_additive"], r["MAE_additive"] = cv(M, mask, additive)
            for k in (1, 2, 3):
                r[f"R2_rank{k}"], r[f"MAE_rank{k}"] = cv(M, mask, lambda MM, mm, k=k: als(MM, mm, k)[0])
            # leave-one-core-out transfer of a scalar substituent constant
            pred = np.full(len(sub), np.nan)
            for core in sub.core.unique():
                tr = (sub.core != core).values
                mu = sub[tr].groupby("R")[c].mean()
                pred[~tr] = sub.loc[~tr, "R"].map(mu).values
            ok = ~np.isnan(pred); y = sub[c].values
            r["LOCO_R2_scalar"] = 1 - np.mean((y[ok] - pred[ok]) ** 2) / np.var(y[ok])
            lr.append(r)
L = pd.DataFrame(lr); w(L, "lowrank.csv", index=False)
q = L[(L.side == "dipole") & (L.set == "direct") & (L.ch == "ΔE‡")].iloc[0]
H["lowrank_dipole_direct_dE"] = {k: round(float(q[k]), 3) for k in ["R2_subst_only", "R2_additive", "R2_rank1", "R2_rank2", "R2_rank3", "LOCO_R2_scalar"]}
jl = []
for side in ["dipole", "dipolarophile"]:
    for lab, sub in [("direct", Pc[(Pc.side == side) & (Pc.attach == "direct")]), ("all80", Pc[Pc.side == side])]:
        blocks = []
        for c in CH:
            piv = sub.pivot_table(index="ctx", columns="R", values=c, aggfunc="mean")
            piv = piv / np.nanstd(piv.values); piv.index = [f"{CHN[c]}|{i}" for i in piv.index]; blocks.append(piv)
        Bm = pd.concat(blocks); M = Bm.values; mask = ~np.isnan(M); M = np.nan_to_num(M)
        r = dict(side=side, set=lab, rows=M.shape[0], n_sub=M.shape[1])
        r["R2_subst_only"] = cv(M, mask, subst_only)[0]; r["R2_additive"] = cv(M, mask, additive)[0]
        for k in (1, 2, 3, 4, 5):
            r[f"R2_rank{k}"] = cv(M, mask, lambda MM, mm, k=k: als(MM, mm, k)[0])[0]
        jl.append(r)
w(pd.DataFrame(jl), "joint_lowrank.csv", index=False)
Zs = (Pc[CH] - Pc[CH].mean()) / Pc[CH].std()
pca = PCA().fit(Zs.values); sc = pca.transform(Zs.values)
pc = pd.DataFrame(pca.components_.T, index=[CHN[c] for c in CH], columns=[f"PC{i+1}" for i in range(len(CH))])
pc.loc["explained_var_ratio"] = pca.explained_variance_ratio_
pc.loc["r_with_dzeta"] = [np.corrcoef(sc[:, i], Pc.zeta)[0, 1] for i in range(len(CH))]
pc.loc["r_with_ddE"] = [np.corrcoef(sc[:, i], Pc.barrier_kcal)[0, 1] for i in range(len(CH))]
w(pc, "pca.csv")
H["pca_evr_top3"] = [round(float(v), 3) for v in pca.explained_variance_ratio_[:3]]

# ------------------------------------------------------------- F. selectivity
R = sibling_pairs(df[df.orient.isin(["a", "b"])], ["cell_id", "face", "stereo_index"], "orient", "a", "b")
dB = df[df.face.isin(["exo", "endo"])]
E = sibling_pairs(dB, ["cell_id", "orient"], "face", "exo", "endo")
R.to_csv(OUT / "sel_regio.csv", index=False); E.to_csv(OUT / "sel_exoendo.csv", index=False)
def summ(S, name):
    res = dict(set=name, n=len(S), sd=S.barrier_kcal.std()); res.update(budget(S))
    win = np.sign(S.barrier_kcal)
    for c in BUD:
        res["acc_" + CHN[c]] = (np.sign(S[c]) == win).mean()
    res["acc_strain"] = (np.sign(S.d1_kcal + S.d2_kcal) == win).mean()
    ei = S[["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal"]].sum(axis=1)
    res["acc_Eint"] = (np.sign(ei) == win).mean()
    res["acc_earlierTS"] = (np.sign(-S.zeta) == win).mean()
    return res
rows = [summ(R, "regio a−b (all)")] + [summ(R[R.dipole == d], f"regio a−b {d}") for d in sorted(R.dipole.unique())]
rows += [summ(E, "exo−endo (all)")] + [summ(E[E.dipole == d], f"exo−endo {d}") for d in sorted(E.dipole.unique())]
S = pd.DataFrame(rows).set_index("set"); w(S, "selectivity.csv")
R2s = R[R.series == "R2"].copy(); R2s["a_pref"] = R2s.barrier_kcal < 0
R2s["skel_grp"] = np.where(R2s.skeleton.isin(["A", "B", "C"]), "A+B+C", R2s.skeleton)
known = R2s.groupby(["dipole", "skel_grp"]).a_pref.agg(["mean", "size"]); w(known, "regio_known.csv")
H["regio_a_pref_ABC"] = {d: round(float(known.loc[(d, "A+B+C"), "mean"]), 3) for d in known.index.get_level_values(0).unique() if (d, "A+B+C") in known.index}
H["regio_acc_strain"] = round(float(S.loc["regio a−b (all)", "acc_strain"]), 3)
H["regio_acc_Eint"] = round(float(S.loc["regio a−b (all)", "acc_Eint"]), 3)
H["exo_pref_frac"] = round(float((E.barrier_kcal < 0).mean()), 3)
H["exo_acc_d2"] = round(float(S.loc["exo−endo (all)", "acc_d2"]), 3)
H["n_regio_pairs"], H["n_exoendo_pairs"] = len(R), len(E)

# ------------------------------------------------------------- G. fingerprints
rng = np.random.default_rng(0)
fp = []
for side in ["dipole", "dipolarophile"]:
    s = Pc[(Pc.side == side) & (Pc.attach == "direct")]
    for Xs, g in s.groupby("X"):
        r = dict(side=side, X=Xs, n=len(g), n_core=g.core.nunique())
        cores = g.core.unique()
        for c in T:
            r[CHN[c]] = g[c].median()
            byc = {k: g.loc[g.core == k, c].values for k in cores}
            bs = [np.median(np.concatenate([byc[k] for k in rng.choice(cores, len(cores))])) for _ in range(300)]
            r[CHN[c] + "_lo"], r[CHN[c] + "_hi"] = np.percentile(bs, [2.5, 97.5])
        fp.append(r)
FP = pd.DataFrame(fp); w(FP, "fingerprint_direct.csv", index=False)
cons = Pc[Pc.attach == "direct"].groupby(["side", "X"]).apply(lambda g: (np.sign(g.barrier_kcal) == np.sign(g.barrier_kcal.median())).mean())
w(cons.rename("sign_consistency").to_frame(), "sign_consistency.csv")
H["sign_consistency_median"] = {s: round(float(cons.loc[s].median()), 2) for s in ["dipole", "dipolarophile"]}

# ------------------------------------------------------------- H. presence proxy (without status file)
c = df[~df.is_parent].groupby(["dipole", "skeleton", "series", "R"]).size().rename("n").reset_index()
sib = c.groupby(["dipole", "skeleton", "series"]).n.max().rename("n_sib")
allR = sorted(df.loc[~df.is_parent, "R"].unique())
G = pd.DataFrame([(*k, Rr, ns) for k, ns in sib.items() for Rr in allR], columns=["dipole", "skeleton", "series", "R", "n_sib"]).merge(c, how="left").fillna({"n": 0})
G["present_frac"] = G.n / G.n_sib; w(G, "presence_proxy.csv", index=False)
H["presence_proxy_by_dipole"] = {k: round(float(v), 3) for k, v in G.groupby("dipole").apply(lambda g: g.n.sum() / g.n_sib.sum()).items()}

# ------------------------------------------------------------- I. aryl vs Ph
A = pairs_vs_Ph(df); A.to_csv(OUT / "pairs_aryl.csv", index=False)
ar = []
for side in ["dipole", "dipolarophile"]:
    for pos in ["p", "m", "o"]:
        s = A[(A.side == side) & (A.pos == pos)]
        for c in ["barrier_kcal", "elst_dft", "pauli_dft", "oi_dft", "d1_kcal", "d2_kcal"]:
            M, mask, piv = matrix(s, c, columns="X")
            r = dict(side=side, pos=pos, ch=CHN[c], n_ctx=M.shape[0], sd=M[mask].std())
            r["R2_subst_only"] = cv(M, mask, subst_only)[0]
            r["R2_rank1"] = cv(M, mask, lambda MM, mm: als(MM, mm, 1)[0])[0]
            r["R2_rank2"] = cv(M, mask, lambda MM, mm: als(MM, mm, 2)[0])[0]
            ar.append(r)
w(pd.DataFrame(ar), "aryl_lowrank.csv", index=False)
H["n_aryl_pairs"] = len(A)

# ------------------------------------------------------------- J. domain D1 vs D0
mu, sd = d0[CH].mean(), d0[CH].std()
Z0 = ((d0[CH] - mu) / sd).values; Z1 = ((df.loc[~df.q_bad, CH] - mu) / sd).values
nn = NearestNeighbors(n_neighbors=2).fit(Z0)
thr = np.percentile(nn.kneighbors(Z0)[0][:, 1], 95)
d10 = nn.kneighbors(Z1, n_neighbors=1)[0][:, 0]
dom = dict(d0_nn95=float(thr), d1_frac_beyond=float((d10 > thr).mean()))
json.dump(dom, open(OUT / "domain.json", "w"), indent=1)
H["d1_frac_outside_d0"] = round(dom["d1_frac_beyond"], 3)

# ------------------------------------------------------------- K. optional: missingness with status
if a.manifest and a.status:
    import statsmodels.formula.api as smf
    man = pd.read_csv(a.manifest); st = pd.read_csv(a.status)
    m = man.merge(st[["job_id", a.status_col]], on="job_id", how="left")
    m["ok"] = (m[a.status_col] == "ok").astype(int)
    m["R"] = np.where(m.series == "R1", m.R1, m.R2); m["X"] = m.R.astype(str).str.replace(r"^[pmo]-", "", regex=True)
    tab = m.groupby(["dipole", "X"]).ok.mean().unstack(0); w(tab, "ok_rate_dipole_X.csv")
    w(m.groupby(["dipole", a.status_col]).size().unstack(fill_value=0), "status_by_dipole.csv")
    fit = smf.logit("ok ~ C(dipole) + C(skeleton) + C(X) + C(attach) + C(series)", data=m).fit(disp=0, maxiter=200)
    w(pd.DataFrame(dict(coef=fit.params, se=fit.bse, p=fit.pvalues)), "missing_logit.csv")
    H["ok_rate_overall"] = round(float(m.ok.mean()), 3)

# ------------------------------------------------------------- L. optional: Hammett with literature sigma
if a.sigma:
    sg = pd.read_csv(a.sigma).set_index("X")      # columns: sigma_m, sigma_p, F, R  (Hansch-Leo-Taft 1991 table values)
    hm = []
    for side in ["dipole", "dipolarophile"]:
        for pos, scol in [("p", "sigma_p"), ("m", "sigma_m")]:
            s = A[(A.side == side) & (A.pos == pos)].join(sg, on="X")
            s = s.dropna(subset=[scol])
            for c in ["barrier_kcal"] + CH:
                for ctx, g in s.groupby("ctx"):
                    if len(g) < 8:
                        continue
                    lr1 = LinearRegression().fit(g[[scol]], g[c]); lr2 = LinearRegression().fit(g[["F", "R"]], g[c])
                    hm.append(dict(side=side, pos=pos, ch=CHN[c], ctx=ctx, n=len(g), rho=lr1.coef_[0], R2_sigma=lr1.score(g[[scol]], g[c]),
                                   rho_F=lr2.coef_[0], rho_R=lr2.coef_[1], R2_FR=lr2.score(g[["F", "R"]], g[c])))
    HM = pd.DataFrame(hm); w(HM, "hammett.csv", index=False)
    # learned rank-2 substituent vectors (direct, both sides) vs (F, R): canonical correlation
    from sklearn.cross_decomposition import CCA
    cc = []
    for side in ["dipole", "dipolarophile"]:
        sub = Pc[(Pc.side == side) & (Pc.attach == "direct")]
        for c in ["barrier_kcal"] + CH:
            M, mask, piv = matrix(sub, c)
            _, U, V = als(M, mask, 2, iters=150)
            Vd = pd.DataFrame(V, index=piv.columns).join(sg[["F", "R"]], how="inner").dropna()
            if len(Vd) < 8:
                continue
            cca = CCA(n_components=2).fit(Vd[[0, 1]].values, Vd[["F", "R"]].values)
            u, v = cca.transform(Vd[[0, 1]].values, Vd[["F", "R"]].values)
            cc.append(dict(side=side, ch=CHN[c], n_sub=len(Vd), can_r1=np.corrcoef(u[:, 0], v[:, 0])[0, 1], can_r2=np.corrcoef(u[:, 1], v[:, 1])[0, 1]))
    w(pd.DataFrame(cc), "sigma_vs_rank2.csv", index=False)

json.dump(H, open(OUT / "headline.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(H, indent=1, ensure_ascii=False))
```

### `analysis/d1_stats/figs.py` — 그림 4장

<!-- FILE: analysis/d1_stats/figs.py sha256=4621e052b461c80490fb2bcd02c27d679a8b134de1d0f1cad100ffde5ab9bc77 -->
```python
"""figs.py — run_all.py 출력 폴더에서 그림 4장을 만든다. python figs.py OUTDIR"""
import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
D=sys.argv[1] if len(sys.argv)>1 else '.'
FIG=os.path.join(D,'fig'); os.makedirs(FIG,exist_ok=True)
SURF='#fcfcfb'; T1='#0b0b0b'; T2='#52514e'; GRID='#e4e3df'
S=['#2a78d6','#eb6834','#1baf7a','#eda100']
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.size':9,'axes.edgecolor':'#bdbcb6',
 'axes.labelcolor':T2,'xtick.color':T2,'ytick.color':T2,'text.color':T1,'axes.titlesize':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
order=['d1','d2','elst','Pauli','OI','disp','CPCM','CDS','bsse']
# ---------- Fig 1 budget
B=pd.read_csv(os.path.join(D,'budget.csv')).set_index('set')
sets=[('ALL pairs','D1 substituent pairs (ΔΔ, n=7,632)'),('D1 absolute (between all TS)','D1 between reactions (n=9,231)'),('D0 absolute (between all TS)','D0 between reactions (n=5,262)')]
fig,ax=plt.subplots(figsize=(7.2,4.2))
y=np.arange(len(order)); h=0.26
for j,(k,lab) in enumerate(sets):
    v=B.loc[k,order].values.astype(float)
    ax.barh(y+(j-1)*h,v,height=h-0.03,color=S[j],label=lab,zorder=3)
    if j==0:
        for yi,vi in zip(y,v):
            if abs(vi)>0.3: ax.text(vi+(0.05 if vi>0 else -0.05),yi-h,f'{vi:+.2f}',va='center',ha='left' if vi>0 else 'right',fontsize=8,color=T1)
ax.axvline(0,color='#8c8b85',lw=1,zorder=2); ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.grid(axis='x',color=GRID,lw=0.8,zorder=0); ax.set_xlabel('share of Var(ΔE‡)  =  Cov(channel, ΔE‡) / Var(ΔE‡)   (shares sum to 1)')
ax.set_title('Who carries the barrier change? Covariance budget of ΔE‡')
ax.legend(frameon=False,loc='lower right',fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig1_budget.png'),dpi=200); plt.close(fig)
# ---------- Fig 2 low-rank
L=pd.read_csv(os.path.join(D,'lowrank.csv'))
models=[('R2_subst_only','scalar\nper X'),('R2_additive','X + context\n(additive)'),('R2_rank1','rank 1\n(σ·ρ)'),('R2_rank2','rank 2'),('R2_rank3','rank 3')]
chs=['ΔE‡','Pauli','OI','disp']
fig,axs=plt.subplots(1,2,figsize=(8.6,3.6),sharey=True)
for ax,side,title in zip(axs,['dipole','dipolarophile'],['substituent on dipole (74 contexts × 20 X)','substituent on dipolarophile (31 contexts × 20 X)']):
    q=L[(L.side==side)&(L.set=='direct')].set_index('ch'); ends=[]
    for j,c in enumerate(chs):
        v=q.loc[c,[m for m,_ in models]].values.astype(float)
        ax.plot(range(len(models)),v,'-o',color=S[j],lw=2,ms=5,label=c,zorder=3)
        ends.append([v[-1],c])
    ends.sort(key=lambda e:e[0]); ys=[e[0] for e in ends]
    for i in range(1,len(ys)):
        if ys[i]-ys[i-1]<0.045: ys[i]=ys[i-1]+0.045
    for (v,c),yy in zip(ends,ys): ax.text(len(models)-1+0.12,yy,c,color=T1,fontsize=8,va='center')
    ax.set_xticks(range(len(models))); ax.set_xticklabels([l for _,l in models],fontsize=7)
    ax.axhline(0,color='#8c8b85',lw=1); ax.grid(axis='y',color=GRID,lw=0.8,zorder=0); ax.set_title(title,fontsize=9)
    ax.set_xlim(-0.3,len(models)-0.2)
axs[0].set_ylabel('5-fold CV R² on held-out (context, X) cells')
axs[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('How many substituent parameters are needed? (direct substituents, ΔΔ vs H parent)',fontweight='bold',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig2_lowrank.png'),dpi=200); plt.close(fig)
# ---------- Fig 3 selectivity
Sx=pd.read_csv(os.path.join(D,'selectivity.csv')).set_index('set')
items=[('acc_d1','d1'),('acc_d2','d2'),('acc_strain','d1+d2'),('acc_elst','elst'),('acc_Pauli','Pauli'),('acc_OI','OI'),('acc_Eint','E_int (6ch+bsse)'),('acc_earlierTS','earlier TS (larger ζ)')]
fig,axs=plt.subplots(1,2,figsize=(8.6,3.4),sharey=True)
for ax,k,title in zip(axs,['regio a−b (all)','exo−endo (all)'],[f"regioselectivity (n={int(Sx.loc['regio a−b (all)','n'])} pairs)",f"exo/endo selectivity (n={int(Sx.loc['exo−endo (all)','n'])} pairs)"]):
    v=Sx.loc[k,[a for a,_ in items]].values.astype(float)
    ax.bar(range(len(items)),v,color=S[0],width=0.7,zorder=3)
    for i,vi in enumerate(v): ax.text(i,vi+0.02,f'{vi:.2f}',ha='center',fontsize=7.5,color=T1)
    ax.axhline(0.5,color='#8c8b85',lw=1,ls='--',zorder=2); ax.text(-0.45,0.515,'coin flip',fontsize=7,color=T2,ha='left',va='bottom')
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for _,l in items],rotation=35,ha='right',fontsize=7.5)
    ax.set_ylim(0,1.08); ax.grid(axis='y',color=GRID,lw=0.8,zorder=0); ax.set_title(title,fontsize=9)
axs[0].set_ylabel('fraction of pairs where the channel\nfavours the lower-barrier TS')
fig.suptitle('Which channel picks the winning TS? (single point at each TS; dashed line = 0.5)',fontweight='bold',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig3_selectivity.png'),dpi=200); plt.close(fig)
# ---------- Fig 4 fingerprint heatmap
FP=pd.read_csv(os.path.join(D,'fingerprint_direct.csv'))
cols=['d1','d2','elst','Pauli','OI','disp','CPCM','CDS','ΔE‡']
cmap=LinearSegmentedColormap.from_list('div',['#104281','#3987e5','#f0efec','#e34948','#9e2423'])
fig,axs=plt.subplots(1,2,figsize=(9.4,5.6))
for ax,side,title in zip(axs,['dipolarophile','dipole'],['X on dipolarophile','X on dipole']):
    q=FP[FP.side==side].set_index('X')[cols].sort_values('ΔE‡')
    Z=q/q.abs().max()  # per-column scale
    ax.imshow(Z.values,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            val=q.values[i,j]; ax.text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=6.5,color=T1 if abs(Z.values[i,j])<0.6 else 'white')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,fontsize=8); ax.set_yticks(range(q.shape[0])); ax.set_yticklabels(q.index,fontsize=8)
    ax.set_title(title,fontsize=9); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle('Median ΔΔ (kcal/mol) vs H parent, direct substituents (colour scaled per column; blue = lowers, red = raises)',fontweight='bold',fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig4_fingerprint.png'),dpi=200); plt.close(fig)
print('ok')
```

### `analysis/d1_stats/q5q7.py` — B1b: dipole별 E_int 상관·CI·부분상관, Q7 진단

<!-- FILE: analysis/d1_stats/q5q7.py sha256=0f529d2d6d1e4f1e28069657b3bd1b3e9b37450ccbeddba89c9d0d459b9ec651 -->
```python
"""q5q7.py — dipole별 E_int–장벽 상관(CI, 강건성, Δζ 통제)과 TS 위치를 맞춘 regio 형제 분석.
python q5q7.py LABELS RESULTS_DIR   (RESULTS_DIR = run_all.py 출력 폴더)"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LAB, RD = sys.argv[1], sys.argv[2]
from d1s_prep import load_labels
df=load_labels(LAB)
P=pd.read_csv(os.path.join(RD,'pairs_H.csv'))
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
alt=df.set_index('job_id').alt_used
P['alt_same']=P.job_id.map(alt).values==P.parent_job.map(alt).values
P['Eint']=P[INT].sum(axis=1); P['strain']=P.d1_kcal+P.d2_kcal
rng=np.random.default_rng(3)
def r_ci(g,B=1000):
    r0=np.corrcoef(g.Eint,g.barrier_kcal)[0,1]; cores=g.core.unique(); byc={k:x for k,x in g.groupby('core')}; bs=[]
    for _ in range(B):
        D=pd.concat([byc[k] for k in rng.choice(cores,len(cores))]); bs.append(np.corrcoef(D.Eint,D.barrier_kcal)[0,1])
    lo,hi=np.percentile(bs,[2.5,97.5]); return r0,lo,hi,len(g),len(cores)
print('=== Q5: r(ΔΔEint, ΔΔE‡) by dipole, core-bootstrap 95% CI')
variants={'base (clean)':P[~P.q_bad],'incl. stepwise':P,'alt_used same':P[(~P.q_bad)&P.alt_same]}
# min-barrier sibling only (per cell, regio, face) on clean
c=P[~P.q_bad].copy(); c['key']=c.cell_id+'|'+c.regio+'|'+c.face.fillna('')
variants['lowest sibling only']=c.loc[c.groupby('key').barrier_kcal.idxmin()]
rows=[]
for vn,V in variants.items():
    for dp,g in V.groupby('dipole'):
        r0,lo,hi,n,nc=r_ci(g,500 if vn!='base (clean)' else 1000); rows.append(dict(variant=vn,dipole=dp,n=n,n_core=nc,r=r0,lo=lo,hi=hi))
Q5=pd.DataFrame(rows); print(Q5.round(2).pivot_table(index='dipole',columns='variant',values=['r']).round(2).to_string())
print(Q5[Q5.variant=='base (clean)'].round(2).to_string(index=False))
Q5.to_csv(os.path.join(RD,'q5_dipole_eint.csv'),index=False)
# by side within dipole
Pc=P[~P.q_bad]
print(Pc.groupby(['dipole','side']).apply(lambda g: pd.Series({'n':len(g),'r_Eint':np.corrcoef(g.Eint,g.barrier_kcal)[0,1],'r_strain':np.corrcoef(g.strain,g.barrier_kcal)[0,1],'r_Eint_dzeta':np.corrcoef(g.Eint,g.zeta)[0,1]})).round(2).to_string())
# partial: Eint vs barrier controlling Δζ, Δasyn, per dipole
from sklearn.linear_model import LinearRegression
def resid(y,X): return y-LinearRegression().fit(X,y).predict(X)
print('partial r(Eint, ΔE‡ | Δζ, Δasyn) by dipole')
for dp,g in Pc.groupby('dipole'):
    X=g[['zeta','asyn']].values; print(dp, round(np.corrcoef(resid(g.Eint.values,X),resid(g.barrier_kcal.values,X))[0,1],2))
print()
print('=== Q7: regio pairs, controlling TS position')
R=pd.read_csv(os.path.join(RD,'sel_regio.csv')); R['Eint']=R[INT].sum(axis=1); R['strain']=R.d1_kcal+R.d2_kcal
w=np.sign(R.barrier_kcal)
for thr in [None,0.05,0.03,0.02]:
    S=R if thr is None else R[R.zeta.abs()<thr]
    ws=np.sign(S.barrier_kcal)
    print(f'|Δζ|<{thr}', 'n',len(S), 'acc strain',round((np.sign(S.strain)==ws).mean(),2),'acc Eint',round((np.sign(S.Eint)==ws).mean(),2),'acc OI',round((np.sign(S.oi_dft)==ws).mean(),2),'acc elst',round((np.sign(S.elst_dft)==ws).mean(),2),'acc Pauli',round((np.sign(S.pauli_dft)==ws).mean(),2))
X=R[['zeta','asyn']].values
for nm in ['Eint','strain','oi_dft','elst_dft','pauli_dft']:
    print('partial r(Δ',nm,', ΔE‡ | Δζ,Δasyn) =', round(np.corrcoef(resid(R[nm].values,X),resid(R.barrier_kcal.values,X))[0,1],3), ' raw r =', round(np.corrcoef(R[nm],R.barrier_kcal)[0,1],3))
# bootstrap CI of acc in |Δζ|<0.03 subset
S=R[R.zeta.abs()<0.03]; cores=S.core.unique(); byc={k:x for k,x in S.groupby('core')}; bs=[]
for _ in range(1000):
    D=pd.concat([byc[k] for k in rng.choice(cores,len(cores))]); ws=np.sign(D.barrier_kcal)
    bs.append(((np.sign(D.strain)==ws).mean(),(np.sign(D.Eint)==ws).mean()))
bs=np.array(bs); print('|Δζ|<0.03 CI strain',np.percentile(bs[:,0],[2.5,97.5]).round(2),'Eint',np.percentile(bs[:,1],[2.5,97.5]).round(2), 'n cores',len(cores))
print(S.groupby('dipole').apply(lambda g: pd.Series({'n':len(g),'acc_strain':(np.sign(g.strain)==np.sign(g.barrier_kcal)).mean(),'acc_Eint':(np.sign(g.Eint)==np.sign(g.barrier_kcal)).mean()})).round(2).to_string())
```

### `analysis/d1_stats/ci_lowrank.py` — B1b: 저랭크 CV R² 코어 부트스트랩

<!-- FILE: analysis/d1_stats/ci_lowrank.py sha256=dd606a3dde29ae1137634422ee271570d4211b2caae5f9d68a4e8f89d5f8859d -->
```python
"""ci_lowrank.py — python ci_lowrank.py N_BOOT RESULTS_DIR
Core-cluster bootstrap CI for CV R² of subst-only / additive / rank1-3 (cell-grouped folds so duplicated cells never leak)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = sys.argv[2] if len(sys.argv) > 2 else '.'
from d1s_lowrank import als, additive, subst_only
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); P=P[~P.q_bad].copy()
P['ctx']=P.core+'|'+P.orient.fillna('')+'|'+P.face.fillna('')+'|'+P.stereo_index.astype(str)
def cv_grouped(M,mask,cellid,fitter,nfold=5,seed=1):
    rng=np.random.default_rng(seed); ids=np.unique(cellid[mask]); rng.shuffle(ids)
    fold_of={c:i%nfold for i,c in enumerate(ids)}
    F=np.full(M.shape,-1); F[mask]=[fold_of[c] for c in cellid[mask]]
    yall=M[mask]; sse=sst=0.0
    for f in range(nfold):
        te=(F==f); tr=mask&~te
        pred=fitter(M,tr); y=M[te]; p=pred[te]
        sse+=((y-p)**2).sum(); sst+=((y-yall.mean())**2).sum()
    return 1-sse/sst
FIT={'subst':subst_only,'additive':additive,'rank1':lambda M,m:als(M,m,1)[0],'rank2':lambda M,m:als(M,m,2)[0],'rank3':lambda M,m:als(M,m,3)[0]}
def build(sub,col,cores):
    rows=[];cells=[]
    for bi,k in enumerate(cores):
        g=sub[sub.core==k]
        piv=g.pivot_table(index='ctx',columns='R',values=col,aggfunc='mean')
        for ctx,r in piv.iterrows(): rows.append(r.rename(f'{bi}|{ctx}')); cells.append(ctx)
    D=pd.DataFrame(rows); Xs=sorted(sub.R.unique()); D=D.reindex(columns=Xs)
    M=D.values; mask=~np.isnan(M)
    cellid=np.array([[f'{c}|{x}' for x in Xs] for c in cells],dtype=object)
    return np.nan_to_num(M),mask,cellid
jobs=[('dipole','barrier_kcal'),('dipolarophile','barrier_kcal'),('dipole','pauli_dft')]
NB=int(sys.argv[1]) if len(sys.argv)>1 else 200
out=[]
for side,col in jobs:
    sub=P[(P.side==side)&(P.attach=='direct')]
    cores=np.array(sorted(sub.core.unique())); rng=np.random.default_rng(11)
    for b in range(NB+1):
        cs=cores if b==0 else rng.choice(cores,len(cores))
        M,mask,cid=build(sub,col,cs)
        r={'side':side,'ch':col,'boot':b}
        for k,f in FIT.items(): r[k]=cv_grouped(M,mask,cid,f)
        out.append(r)
        if b%20==0: print(side,col,b,{k:round(v,3) for k,v in r.items() if k in FIT},flush=True); pd.DataFrame(out).to_csv(os.path.join(RD,'ci_lowrank_boot.csv'),index=False)
pd.DataFrame(out).to_csv(os.path.join(RD,'ci_lowrank_boot.csv'),index=False)
print('DONE')
```

### `analysis/d1_stats/q2_houk_domingo.py` — B5: 도너/억셉터 비대칭, 용매화 제외, regio 크기

<!-- FILE: analysis/d1_stats/q2_houk_domingo.py sha256=9d7f9f9fb3ae7d59aa9c8a3a27a54cc6e3bda7546d4eaa41a9cefc1e317bf943 -->
```python
import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels
df=load_labels(LAB)
d=df[~df.q_bad].copy()
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); P=P[~P.q_bad]
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
P['solv']=P.cpcm_dft+P.cds_dft; P['strain']=P.d1_kcal+P.d2_kcal; P['Eint']=P[INT].sum(axis=1)
P['nosolv']=P.barrier_kcal-P.solv
# Domingo's categories (his own classification of these alkenes): vinyl amines = supernucleophiles; nitroethylene, acrylonitrile = strong electrophiles
DON=['NMe2','NH2']; ACC=['NO2','CN']; MID=['OMe','OH','CO2Me','CHO','Ac','CF3','Me','Cl','F','Br']
rows=[]
for dp in ['azide','nitrile oxide','nitrone']:
    for sk in ['A','B','C','D']:
        s=P[(P.dipole==dp)&(P.side=='dipolarophile')&(P.attach=='direct')&(P.skeleton==sk)]
        if s.empty: continue
        # kinetically relevant TS per X: lowest barrier over regio/stereo
        best=s.loc[s.groupby('X').barrier_kcal.idxmin()].set_index('X')
        for X in DON+ACC+['OMe','CO2Me','Me']:
            if X in best.index:
                b=best.loc[X]; rows.append(dict(dipole=dp,skel=sk,X=X,ddE=b.barrier_kcal,dd_strain=b.strain,dd_Eint=b.Eint,dd_solv=b.solv,dd_nosolv=b.nosolv,dd_elst=b.elst_dft,dd_pauli=b.pauli_dft,dd_oi=b.oi_dft))
T=pd.DataFrame(rows)
pd.set_option('display.width',250)
print(T.pivot_table(index=['dipole','X'],columns='skel',values='ddE').round(1).to_string())
print('\n-- solvation-stripped (ΔΔE‡ − ΔΔ(CPCM+CDS))')
print(T.pivot_table(index=['dipole','X'],columns='skel',values='dd_nosolv').round(1).to_string())
# asymmetry index per dipole/skeleton: mean(donor lowering) / mean(acceptor lowering)
print('\n-- asymmetry: mean ΔΔE‡ donors (NMe2,NH2) vs acceptors (NO2,CN)')
for (dp,sk),g in T.groupby(['dipole','skel']):
    dn=g[g.X.isin(DON)].ddE.mean(); ac=g[g.X.isin(ACC)].ddE.mean(); dn2=g[g.X.isin(DON)].dd_nosolv.mean(); ac2=g[g.X.isin(ACC)].dd_nosolv.mean()
    print(f'{dp:14s} {sk}  donors {dn:+.1f}  acceptors {ac:+.1f}   | no-solv: donors {dn2:+.1f} acceptors {ac2:+.1f}')
# regioselectivity magnitude for azide R2 by X
R=pd.read_csv(os.path.join(RD,'sel_regio.csv'))
R=R[(R.series=='R2')&(R.attach=='direct')]
print('\n-- |ΔE‡(regio a−b)| (kcal/mol), median over contexts, by dipole and X')
print(R.assign(absd=R.barrier_kcal.abs()).pivot_table(index='X',columns='dipole',values='absd',aggfunc='median').round(1).loc[DON+['OMe','OH']+ACC+['CO2Me','CHO','Ac','CF3']].to_string())
T.to_csv(os.path.join(RD,'q2_azide_table.csv'),index=False)
```

### `analysis/d1_stats/inflection.py` — B5·B6: σ 순서 변곡점, 비대칭 기울기 (σ 표 필요)

<!-- FILE: analysis/d1_stats/inflection.py sha256=9c5cee5df7eab421bad34f8cca7b1d4fbc00439a7fd2d5fbc1a7b1b40bac1028 -->
```python
"""inflection.py — EDG→EWG 순서(σ)에서 채널별 변곡점(기울기 변화) 탐지 + 도너/억셉터 비대칭(Houk vs Domingo).

python inflection.py LABELS RESULTS_DIR SIGMA_CSV OUTDIR
  SIGMA_CSV 열: X, sigma_m, sigma_p (Hansch, Leo & Taft 1991 원표 값; 빈칸 허용)
계열 = (dipole, 골격, 쪽, 부착, 문맥[orient|face|stereo]). direct·para는 σp, meta는 σm, ortho는 제외.
  direct: H 부모 대비 ΔΔ (pairs_H.csv).  para/meta: Ph 대비 ΔΔ (pairs_aryl.csv) — Hammett 본래 설정.
모형 (각 계열 × 타깃): L 선형 / S 연속 꺾은선(꺾임점 1개, 격자 탐색) / Q 이차.
변곡 판정(사전 고정): AICc(S) ≤ AICc(L) − 4 이고 LOO RMSE(S) < LOO RMSE(L). 계열당 치환기 ≥ 12.
거짓 양성률: 계열 안에서 σ를 섞은 순열 200회로 같은 판정을 돌린 비율(경험적 귀무).
비대칭: 꺾임점을 σ = 0에 고정한 꺾은선의 기울기 b_donor(σ<0 쪽), b_acceptor(σ>0 쪽).
"""
import sys, os, numpy as np, pandas as pd
LAB, RD, SIG, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
MIN_N = 12; DAICC = 4.0; NPERM = 200; SEED = 20261014
TGT = ['barrier_kcal', 'd1_kcal', 'd2_kcal', 'elst_dft', 'pauli_dft', 'oi_dft', 'disp_dft', 'cpcm_dft', 'cds_dft']
INT = ['elst_dft', 'pauli_dft', 'oi_dft', 'disp_dft', 'cpcm_dft', 'cds_dft', 'bsse_shift_kcal']
sg = pd.read_csv(SIG).set_index('X')

def design(x, kind, bp=None):
    one = np.ones_like(x)
    if kind == 'L': return np.c_[one, x]
    if kind == 'Q': return np.c_[one, x, x ** 2]
    return np.c_[one, x, np.maximum(x - bp, 0.0)]

def fit(x, y, kind, bp=None):
    X = design(x, kind, bp); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return b, (r ** 2).sum()

def aicc(rss, n, k):
    return n * np.log(rss / n + 1e-12) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

def best_seg(x, y):
    grid = np.unique(np.quantile(x, np.linspace(0.2, 0.8, 13)))
    best = None
    for bp in grid:
        b, rss = fit(x, y, 'S', bp)
        if best is None or rss < best[1]: best = (bp, rss, b)
    return best

def loo(x, y, kind, bp=None):
    e = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        if kind == 'S':
            bpi = best_seg(x[m], y[m])[0]; b, _ = fit(x[m], y[m], 'S', bpi); e.append(y[i] - (design(x[i:i+1], 'S', bpi) @ b)[0])
        else:
            b, _ = fit(x[m], y[m], kind); e.append(y[i] - (design(x[i:i+1], kind) @ b)[0])
    return np.sqrt(np.mean(np.square(e)))

def judge(x, y):
    n = len(x); _, rL = fit(x, y, 'L'); bp, rS, bS = best_seg(x, y); _, rQ = fit(x, y, 'Q')
    aL, aS, aQ = aicc(rL, n, 2), aicc(rS, n, 4), aicc(rQ, n, 3)
    infl = (aS <= aL - DAICC)
    if infl: infl = loo(x, y, 'S') < loo(x, y, 'L')
    return dict(aicc_L=aL, aicc_S=aS, aicc_Q=aQ, bp=bp, slope_left=bS[1], slope_right=bS[1] + bS[2], inflection=bool(infl))

P = pd.read_csv(os.path.join(RD, 'pairs_H.csv')); P = P[~P.q_bad & (P.attach == 'direct')].copy(); P['pos'] = 'direct'
A = pd.read_csv(os.path.join(RD, 'pairs_aryl.csv')); A = A[A.pos.isin(['p', 'm'])].copy()
for D in (P, A):
    D['strain'] = D.d1_kcal + D.d2_kcal; D['Eint'] = D[INT].sum(axis=1); D['nosolv'] = D.barrier_kcal - D.cpcm_dft - D.cds_dft
P['series'] = P.dipole + '|' + P.skeleton + '|' + P.side + '|direct|' + P.ctx.astype(str)
A['series'] = A.dipole + '|' + A.skeleton + '|' + A.side + '|' + A.pos + '|' + A.ctx.astype(str)
D = pd.concat([P, A], ignore_index=True)
D['sigma'] = np.where(D.pos == 'm', D.X.map(sg['sigma_m']), D.X.map(sg['sigma_p']))
D = D.dropna(subset=['sigma'])
targets = TGT + ['strain', 'Eint', 'nosolv']
rng = np.random.default_rng(SEED); rows = []; null = []
for s, g in D.groupby('series'):
    g = g.groupby('X').mean(numeric_only=True)        # one value per substituent (siblings averaged)
    if len(g) < MIN_N: continue
    x = g.sigma.values
    dp, sk, side, pos = s.split('|')[:4]
    for t in targets:
        y = g[t].values
        r = judge(x, y); r.update(series=s, dipole=dp, skeleton=sk, side=side, pos=pos, target=t, n=len(g))
        # fixed breakpoint at sigma = 0: donor vs acceptor slopes
        if (x < 0).sum() >= 3 and (x > 0).sum() >= 3:
            b0, _ = fit(x, y, 'S', 0.0); r['slope_donor'] = b0[1]; r['slope_acceptor'] = b0[1] + b0[2]
        rows.append(r)
        if t in ('barrier_kcal', 'elst_dft', 'pauli_dft', 'oi_dft'):
            for _ in range(NPERM // 10):            # 20 permutations per series/target -> pooled null
                null.append(judge(rng.permutation(x), y)['inflection'])
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'inflection_series.csv'), index=False)
summ = R.groupby('target').agg(n_series=('inflection', 'size'), frac_inflection=('inflection', 'mean'), median_bp=('bp', 'median'))
summ['null_rate_permutation'] = np.mean(null) if null else np.nan
# hidden inflections: channel bends while the barrier does not
piv = R.pivot_table(index='series', columns='target', values='inflection')
hid = {t: float(((piv[t] == 1) & (piv['barrier_kcal'] == 0)).mean()) for t in targets if t != 'barrier_kcal'}
summ['hidden_frac_vs_barrier'] = pd.Series(hid)
summ.to_csv(os.path.join(OUT, 'inflection_summary.csv'))
asym = R.dropna(subset=['slope_donor']).groupby(['dipole', 'skeleton', 'side', 'pos', 'target'])[['slope_donor', 'slope_acceptor']].median()
asym.to_csv(os.path.join(OUT, 'asymmetry_slopes.csv'))
print(summ.round(3).to_string())
```

### `analysis/d1_stats/q5_methyl.py` — B7: 메틸 패턴, cis/trans, gem-Me2 regio

<!-- FILE: analysis/d1_stats/q5_methyl.py sha256=8cf433efbe5ae1faffeeabb2222e7c1ace284a0f9ef190911ebac17a87f71937 -->
```python
import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, CH
df=load_labels(LAB)
d=df[(~df.q_bad)&(df.series=='R1')].copy()      # dipole substituent varies, dipolarophile skeleton fixed (R2 = H or '-')
d['strain']=d.d1_kcal+d.d2_kcal
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
d['Eint']=d[INT].sum(axis=1)
COLS=['barrier_kcal','strain','Eint','d1_kcal','d2_kcal','elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft']
# kinetically relevant TS per (dipole, R1, skeleton[, orient]): lowest barrier among stereo siblings (and faces)
best=d.loc[d.groupby(['dipole','R1','skeleton']).barrier_kcal.idxmin()].set_index(['dipole','R1','skeleton'])
bestO=d[d.orient.isin(['a','b'])]
bestO=bestO.loc[bestO.groupby(['dipole','R1','skeleton','orient']).barrier_kcal.idxmin()].set_index(['dipole','R1','skeleton','orient'])
COMP=[('Ap','A','gem-Me2 alkene (2-methylpropene − ethylene)'),('AppE','A','trans-2-butene − ethylene'),('AppZ','A','cis-2-butene − ethylene'),
      ('AppZ','AppE','cis − trans 2-butene'),('Cp','C','2-butyne − acetylene'),('Bp','B','2,3-diMe-norbornene − norbornene'),
      ('Dp','D','3,3-diMe-cyclooctyne − cyclooctyne'),('DppC','D','cis-3,8-diMe − cyclooctyne'),('DppT','D','trans-3,8-diMe − cyclooctyne'),('DppC','DppT','cis − trans 3,8-diMe-cyclooctyne')]
rng=np.random.default_rng(5); rows=[]
for s1,s0,lab in COMP:
    for dp in sorted(d.dipole.unique()):
        try:
            A=best.xs(s1,level='skeleton').loc[dp]; B=best.xs(s0,level='skeleton').loc[dp]
        except KeyError: continue
        common=A.index.intersection(B.index)
        if len(common)<8: rows.append(dict(comp=lab,dipole=dp,n=len(common))); continue
        D=A.loc[common,COLS]-B.loc[common,COLS]
        r=dict(comp=lab,dipole=dp,n=len(common))
        for c in COLS:
            v=D[c].values; bs=[np.median(rng.choice(v,len(v))) for _ in range(500)]
            r[c]=np.median(v); r[c+'_lo'],r[c+'_hi']=np.percentile(bs,[2.5,97.5])
        r['frac_slower']=(D.barrier_kcal>0).mean()
        rows.append(r)
M=pd.DataFrame(rows); M.to_csv(os.path.join(RD,'q5_methyl.csv'),index=False)
show=['comp','dipole','n','barrier_kcal','barrier_kcal_lo','barrier_kcal_hi','frac_slower','strain','Eint','d1_kcal','d2_kcal','elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft']
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
print(M[show].round(1).to_string(index=False))
# gem-dimethyl regio: in Ap/Dp, which orientation wins (Me2-carbon bonded to terminus a or b) and by how much
print('\n-- gem-dimethyl orientation (Ap, Dp): ΔE‡(a − b) median over R1, frac a preferred')
for sk in ['Ap','Dp']:
    for dp in sorted(d.dipole.unique()):
        try:
            a=bestO.xs((sk,'a'),level=['skeleton','orient']).loc[dp]; b=bestO.xs((sk,'b'),level=['skeleton','orient']).loc[dp]
        except KeyError: continue
        cm=a.index.intersection(b.index)
        if len(cm)<5: continue
        dd=a.loc[cm,'barrier_kcal']-b.loc[cm,'barrier_kcal']; ds=(a.loc[cm,'strain']-b.loc[cm,'strain']); de=(a.loc[cm,'Eint']-b.loc[cm,'Eint'])
        print(f'{sk:3s} {dp:16s} n={len(cm):3d}  ΔE‡(a−b) median {dd.median():+.1f}  a preferred {(dd<0).mean():.2f}  | sign agree strain {(np.sign(ds)==np.sign(dd)).mean():.2f}  Eint {(np.sign(de)==np.sign(dd)).mean():.2f}')
```

### `analysis/d1_stats/q4_missing.py` — B8: carbonyl ylide 결측과 포함/제외 민감도

<!-- FILE: analysis/d1_stats/q4_missing.py sha256=39e7bfe5729249b5dc6a045ba3a93b9c14cf210fb2eb45b818ad76c7db3640a3 -->
```python
import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, budget
df=load_labels(LAB)
G=pd.read_csv(os.path.join(RD,'presence_proxy.csv'))
MAN={'nitrile oxide':2096,'azide':2096,'azomethine ylide':2334,'nitrone':2736,'carbonyl ylide':2334}  # SPEC_D1_RUN §3.3
pres=df.groupby('dipole').size()
clean=df[~df.q_bad].groupby('dipole').size()
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); Pc=P[~P.q_bad]
neg=df[~df.q_bad].groupby('dipole').apply(lambda g:(g.barrier_kcal<0).mean())
par_cores=df[df.is_parent].groupby('dipole').core.nunique(); all_cores=df.groupby('dipole').core.nunique()
T=pd.DataFrame({'manifest':pd.Series(MAN),'present':pres,'present_frac':pres/pd.Series(MAN),'clean':clean,'pairs_clean':Pc.groupby('dipole').size(),'cores':all_cores,'cores_with_H_parent':par_cores,'neg_barrier_frac':neg})
print(T.round(3).to_string())
cy=G[G.dipole=='carbonyl ylide']
print('\ncarbonyl ylide presence by skeleton x series (proxy):')
print(cy.groupby(['skeleton','series']).apply(lambda g: f"{int(g.n.sum())}/{int(g.n_sib.sum())}").unstack().to_string())
print('\ncarbonyl ylide presence by X (direct, proxy):')
x=cy[~cy.R.str.match(r'^[pmo]-')].groupby('R').apply(lambda g:g.n.sum()/g.n_sib.sum()).sort_values()
print(x.round(2).to_string())
print('\nbarrier of carbonyl ylide present rows (clean):', df[(~df.q_bad)&(df.dipole=='carbonyl ylide')].barrier_kcal.describe().round(1).to_dict())
# sensitivity: headline with vs without carbonyl ylide
def head(Pc,R,E):
    INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
    y=Pc.barrier_kcal; st=Pc.d1_kcal+Pc.d2_kcal; it=Pc[INT].sum(axis=1); b=budget(Pc)
    w=np.sign(R.barrier_kcal)
    return dict(n_pairs=len(Pc),r_strain=np.corrcoef(st,y)[0,1],r_Eint=np.corrcoef(it,y)[0,1],sh_Pauli=b['Pauli'],sh_elst=b['elst'],sh_OI=b['OI'],sh_strain=b['d1']+b['d2'],
                regio_acc_strain=(np.sign(R.d1_kcal+R.d2_kcal)==w).mean(),regio_acc_Eint=(np.sign(R[INT].sum(axis=1))==w).mean(),exo_frac=(E.barrier_kcal<0).mean(),n_regio=len(R),n_exo=len(E))
R=pd.read_csv(os.path.join(RD,'sel_regio.csv')); E=pd.read_csv(os.path.join(RD,'sel_exoendo.csv'))
S=pd.DataFrame({'with carbonyl ylide':head(Pc,R,E),'without':head(Pc[Pc.dipole!='carbonyl ylide'],R[R.dipole!='carbonyl ylide'],E[E.dipole!='carbonyl ylide'])})
print('\n',S.round(3).to_string())
S.to_csv(os.path.join(RD,'q4_sensitivity.csv')); T.to_csv(os.path.join(RD,'q4_missing_by_dipole.csv'))
```

---

## 부록 B — 게이트 R 기준값 (중간 파일 9,459행에서 `run_all.py`가 낸 headline)

<!-- FILE: analysis/d1_stats/reference_interim/headline.json sha256=a34746b2a6f2882d02733aac211308fe7899c8db7e86b4270a858ad8046f67fd -->
```json
{
 "n_rows": 9459,
 "q_bad": 228,
 "n_pairs": 7776,
 "n_pairs_clean": 7632,
 "n_children_no_parent": 1626,
 "budget_all_pairs": {
  "d1": 0.577,
  "d2": 0.443,
  "elst": -1.113,
  "Pauli": 2.136,
  "OI": -1.107,
  "disp": -0.098,
  "CPCM": 0.145,
  "CDS": -0.003,
  "bsse": 0.021
 },
 "r_strain_barrier": 0.917,
 "r_eint_barrier": -0.044,
 "r_dzeta_barrier": -0.551,
 "lowrank_dipole_direct_dE": {
  "R2_subst_only": 0.098,
  "R2_additive": 0.539,
  "R2_rank1": 0.518,
  "R2_rank2": 0.653,
  "R2_rank3": 0.73,
  "LOCO_R2_scalar": 0.083
 },
 "pca_evr_top3": [
  0.519,
  0.152,
  0.106
 ],
 "regio_a_pref_ABC": {
  "azide": 0.565,
  "nitrile oxide": 0.989,
  "nitrone": 0.961
 },
 "regio_acc_strain": 0.831,
 "regio_acc_Eint": 0.387,
 "exo_pref_frac": 1.0,
 "exo_acc_d2": 1.0,
 "n_regio_pairs": 1794,
 "n_exoendo_pairs": 839,
 "sign_consistency_median": {
  "dipole": 0.59,
  "dipolarophile": 0.74
 },
 "presence_proxy_by_dipole": {
  "azide": 0.92,
  "azomethine ylide": 0.774,
  "carbonyl ylide": 0.618,
  "nitrile oxide": 0.926,
  "nitrone": 0.894
 },
 "n_aryl_pairs": 6339,
 "d1_frac_outside_d0": 0.026
}
```
