# D1 autodE 파일럿 — 진행 계획과 확인 사항

작성 2026-09-29 · 기준 문서 [`d1_autode_pilot/SPEC_D1_pilot.md`](d1_autode_pilot/SPEC_D1_pilot.md)

**현재 상태 (2026-09-29 14:24):** 파일럿 실행이 끝났습니다. 제출 09:00, 보고서 11:57로 약 3시간 걸렸습니다.
- 결과: A1–A4 통과, A5 실패(진단용 기준). D1 8/10이 라벨까지 완료됐습니다.
- 결과 요약과 남은 결정 사항은 [§11](#11-실행-결과-2026-09-29), 전체 보고서는 [`d1_autode_pilot/results/pilot_report.md`](d1_autode_pilot/results/pilot_report.md)에 있습니다.
- 본실험 전 검증은 끝났습니다(2026-09-30). 결과는 [§12](#12-본실험-전-검증-결과-2026-09-30)와 [`validation/results/VALIDATION_REPORT.md`](validation/results/VALIDATION_REPORT.md)에 있고, §8 권고는 **조건부**(사용자 결정 2건)입니다. D1 본실험은 시작하지 않았습니다.

---

## 1. 목적

D1 전체(backbone TS 2,846행)를 돌리기 전에 한 줄의 sbatch 체인을 검증합니다. 체인은 다음 순서로 진행합니다.

> autodE TS 탐색 → strain 기준 구조(Coley 규칙 포팅) → ORCA single point 5개 → method-① 라벨

라벨은 barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds의 9개 target입니다. 대상은 D1 TS 10행과 D0 대조 4개(D0_control 2개, D0_replay 2개)입니다. 파일럿으로 답할 질문은 SPEC §1의 Q1–Q5입니다.

| # | 질문 | 판정 |
|---|---|---|
| Q1 | autodE 1.4.5 + ORCA 6.1.1이 Coley 수준(87.7 %)으로 D1 TS를 찾는가 | A1 |
| Q2 | D1 입출력이 D0 gate를 그대로 통과하는가 | A2 + S0 parity |
| Q3 | 같은 구조에서 SP→라벨 절반이 D0와 같은 값을 내는가 | A3 (D0_replay) |
| Q4 | 포팅한 stereo 기준 규칙이 Coley와 같은 결정을 내리는가 | A4 |
| Q5 | autodE로 새로 돌린 D0 반응이 D0 라벨과 얼마나 가깝고, D1 전체 비용은 얼마인가 | A5, 비용표 |

---

## 2. 완료된 준비 작업

| 항목 | 결과 | 근거 |
|---|---|---|
| 번들 압축 해제 | `d1_autode_pilot/` (27개 파일) | — |
| 번들이 쓰는 repo 모듈 | 모두 존재하고, 함수 시그니처가 번들 호출과 일치 | `fragmenter`, `stage3_parse`, `build_inputs_graph`, `smd_relabel_build_inputs`, `smd_relabel_audit` |
| `fragmenter.py` 3개 사본 | md5가 모두 같음 (import 순서와 무관) | `label_true/scripts`, `analysis/espley_xtb_repro`, `analysis/b3lyp_full/build_strategy/graph` |
| config host 경로 | 모두 존재 | ORCA, xtb, MPI, conda, D0 profiles/csv, D0 입력 2곳 |
| D0 대조 데이터 | 있음 | rxn 20 (`r1_C3H8N2_alt.xyz` 포함), rxn 105 |
| autodE 소스 | tag v1.4.5 = `e7e71b33` (SPEC 고정값과 일치) | `external/autodE` |
| 환경 설치 | `INSTALL_VERIFY PASS` | job 995653, `logs/install_995653.log` |
| 정적 점검 | 모든 `.py` 컴파일 OK, `pilot_env.sh` 경유 import OK, `bash -n` OK | job 995654, `logs/static_995654.log` |

---

## 3. 폴더 구성

```
D1_build/
  D1_PLAN.md              이 문서
  d1_autode_pilot/        파일럿 번들 (코드, manifest, SPEC, tests, evidence)
    install_env.sh        [추가] venv 설치 sbatch 스크립트
    results/              (실행 후 생성) pilot_report.md, S0_REPORT.md 등
  env/d1ade/              autodE venv                          ┐
  external/autodE/        autodE v1.4.5 소스                    │ git 제외
  external/wheels/        Cython 3.1.8 wheel                   │ (.gitignore)
  scratch -> /gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot           │
  logs/                   설치·점검 로그                         │
  d1_autode_pilot.zip     원본 번들                              ┘
```

---

## 4. 환경

`env/d1ade`는 **reactot 위에 얹은 venv**(`--system-site-packages`)입니다. autodE와 Cython만 venv에 들어 있고, 나머지 패키지는 D0 라벨을 만든 reactot 버전을 그대로 씁니다. reactot 자체는 수정하지 않았습니다.

| 패키지 / 바이너리 | 버전 | 위치 |
|---|---|---|
| Python | 3.10.14 | reactot |
| autodE | **1.4.5** (C++ 확장 3개 빌드) | venv |
| Cython | 3.1.8 (빌드 전용) | venv |
| RDKit / networkx / numpy / pandas / scipy | 2026.03.2 / 3.4.2 / 1.26.4 / 2.3.3 / 1.15.3 | reactot |
| xtb | 6.7.1 (edcfbbe) | `~/xtb-dist` |
| ORCA | 6.1.1 | `~/orca_6_1_1_avx2` |

autodE가 실제로 쓰는 키워드는 설치 검증 로그에서 확인했고, SPEC §5.1과 같습니다.

- `opt_ts`: `OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J` + `%geom Calc_Hess true / Recalc_Hess 30 / Trust -0.1 / MaxIter 100`
- `sp`: `B3LYP RIJCOSX D3BJ def2-TZVP def2/J`
- 용매: SMD(water). autodE는 `CPCM(Water)` + `%cpcm smd true SMDsolvent "water"` 형태로 씁니다.

재설치(멱등):
```bash
cd D1_build
sbatch -p cpu2 -o "$PWD/logs/install_%j.log" --export=ALL,D1_BUILD="$PWD" d1_autode_pilot/install_env.sh
```

---

## 5. 번들·SPEC 대비 변경 사항

| 변경 | 내용 | 이유 |
|---|---|---|
| 위치 | `analysis/d1_autode_pilot/` → `D1_build/d1_autode_pilot/` | D1 작업을 `D1_build`에 모으기로 함. 코드가 상대 경로라 동작은 같음 |
| 환경 | SPEC의 `conda create --clone reactot` 대신 venv | reactot이 7 GB(대부분 pip 패키지)라 clone이 무겁고, offline clone은 네트워크가 필요할 수 있음. venv는 수십 MB이고 패키지 버전이 reactot과 같음 |
| `config.yaml` | host 블록에 `venv:` 한 줄 추가 | method, gate 블록은 수정하지 않음 |
| `pilot_env.sh` | venv 활성화 2줄 추가, 오류 메시지 경로 문구 수정 | — |
| 신규 파일 | `install_env.sh`, `.gitignore`, `scratch` 링크, `logs/` | — |

---

## 6. 실행 계획

### 6.1 제출 전 확인 (로그인 노드, one-shot)
```bash
squeue -u $USER -h | wc -l             # 4 이하여야 함 (16 task 추가, MaxSubmit 20)
sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"
```

### 6.2 제출 (명령 한 줄, 즉시 반환)
```bash
cd D1_build/d1_autode_pilot && bash submit_pilot.sh
# -> partition=<idle>  S0=<jid>  ARRAY=<jid> (0-13%10)  REPORT=<jid>  tasks=16
```

### 6.3 체인

| 순서 | job | 내용 | 자원 | 의존성 |
|---|---|---|---|---|
| 1 | S0 preflight | ① 버전 확인 ② D0 5,260건 configuration tag 검증 (`_alt` 일치율 ≥ 0.90) ③ D0 입력 byte parity (rxn 20, 105, 0, 1, 2) ④ xtb scan 출력 형식 ⑤ HCNO + C₂H₄ 전체 체인 smoke ⑥ smoke 판정 | 8코어 / 32 GB / 48 h | — |
| 2 | array 0–13%10 | 14개 job 각각 `ts → ref → inputs → sp → label` | job당 8코어 / 32 GB / 48 h, 동시 10개 | `afterok:S0` + `--kill-on-invalid-dep=yes` |
| 3 | report | `report.py` 실행 후 `results/`로 복사 | 1코어 / 4 GB | `afterany:array` |

각 단계는 멱등입니다.

- 성공하면 `.done_<stage>`, 실패하면 `.fail_<stage>`(사유 포함)를 남깁니다.
- 실패한 단계는 자동으로 재시도하지 않습니다.
- 48 h wall에 걸려 마커 없이 끊긴 단계는 재제출하면 이어서 돕니다. autodE checkpoint와 이미 끝난 ORCA 출력은 건너뜁니다.

### 6.4 예상 소요 시간 (추정, 실측 아님)

| 구간 | 추정 | 비고 |
|---|---|---|
| S0 | 수 시간 | smoke 반응이 작음(7원자). D0 5,260건 검증은 순수 Python |
| array | 1–2일 | ts 단계가 지배적. J05(48원자)는 48 h wall에 걸릴 수 있음 → §6.6 |
| report | 수 분 | — |

실측 비용(core-hour, 단계별·job별)은 `pilot_report.md`의 Cost 절에 나옵니다. D1 2,846행 외삽값도 같은 곳에 나옵니다.

### 6.5 모니터링 (요청 시 one-shot 명령만; 루프·poller 금지)
```bash
squeue -u $USER
sacct -j <ARRAY> --format=JobID,State,Elapsed,NodeList
cat D1_build/scratch/s0/S0_REPORT.md
ls D1_build/scratch/jobs/J*/.{done,fail}_* 2>/dev/null
```

### 6.6 개별 job 재제출 (wall kill 또는 승인된 재시도 후)
```bash
cd D1_build/d1_autode_pilot; SCR=/gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot
sbatch -p <idle> --array=<idx> -o $SCR/logs/job_%A_%a.log --export=ALL,PILOT_DIR=$PWD run_job.sh
sbatch -p <idle> -o $SCR/logs/report_%j.log --export=ALL,PILOT_DIR=$PWD run_report.sh
```
`idx`는 manifest 행 번호입니다(J00 = 0). 실패한 단계를 재시도하려면 해당 `.fail_<stage>`를 지우고, 그 사유를 보고서에 적습니다.

---

## 7. 판정 기준과 중단 조건 (SPEC §8)

| id | 기준 | 임계값 |
|---|---|---|
| A1 | D1 TS 행 중 ok 라벨 | ≥ 8 / 10 |
| A2 | 생성된 라벨이 모든 gate 통과 | g1 identity ≤ 1e-6 · Bond 기준 잔차 ≤ 0.02 · \|Σ6ch − Bond\| ≤ 0.11 · 8채널 closure ≤ 0.11 · D0 audit 문제 없음 · ORCA 6.1.1 / SMD / water / ε 78.3550 / CDS / 정상 종료 |
| A3 | D0_replay가 `labels_all` 재현 | 9개 target, max \|Δ\| ≤ 0.1 kcal/mol |
| A4 | Coley 포팅 결정 = Coley의 `alt_used` | replay 대조 2개 모두 |
| A5 | D0_control \|Δbarrier\| | ≤ 1.0 kcal/mol (진단용, hard stop 아님) |

**다음 경우에는 멈추고 보고한 뒤 지시를 기다립니다.**

- S0 실패. 특히 autodE가 ORCA 6 출력을 읽지 못하는 경우의 선택지는 (i) G16, (ii) autodE 파서 최소 패치, (iii) TS 단계만 ORCA 5.
- S0 parity 또는 config 검증 실패
- 같은 단계에서 같은 이유로 D1 job 3개 이상 실패
- A3 실패. 이 경우 D1 전체 실행이 막힙니다.
- method, gate, 기준 규칙을 바꿔야 하는 상황. 이것은 사용자가 결정합니다.

---

## 8. 확인 사항 (실행 전 결정 또는 인지 필요)

| # | 항목 | 현재 처리 | 필요한 결정 |
|---|---|---|---|
| C1 | `D1_설계_v8.xlsx`가 `D1_build`에 없음 | 파일럿 실행에는 불필요. `sample_pilot.py`로 manifest를 다시 뽑을 때만 필요하고, manifest는 이미 번들에 있음 | manifest를 다시 뽑을 경우에만 파일 제공 |
| C2 | scratch 위치 | 무거운 출력은 `/gpfs/tmp_cpu2`에 두고 `D1_build/scratch`로 링크 (CLAUDE.md 규칙, 홈 용량 보호) | 전부 `D1_build` 안에 두려면 `config.yaml`의 `scratch:` 변경 (홈 300 GB 사용) |
| C3 | `config.yaml` 주석의 Coley DOI(`…02043-z`)가 SPEC(`…01977-8`)과 다름 | 주석이라 계산에는 영향 없음. frozen 블록이라 수정하지 않음 | 수정 여부 |
| C4 | 결과 커밋 위치 | SPEC §7.5는 `analysis/d1_autode_pilot/` 기준. 여기서는 `D1_build/d1_autode_pilot/results/` | — |
| C5 | G16 사용 가능 여부 | 확인하지 않음. S0 `versions` 단계가 PATH의 g16 유무를 출력함 | §9-1 TS 엔진 결정 시 참고 |

**알려진 위험**

- **ORCA 6.1.1 출력 파싱 (가장 큰 위험).** autodE 1.4.5를 ORCA 6에서 쓴 사례가 이 환경에는 없습니다. S0 smoke가 이를 판정하도록 설계돼 있어서, 실패하면 array는 시작되지 않습니다.
- **autodE의 `is_v5` 판정.** ORCA 6을 "v5 아님"으로 판단하지만, 이 판정은 CPCM 경로에서만 쓰입니다. 파일럿은 SMD라 영향이 없습니다.
- **autodE 버전 고정.** `git clone --branch v1.4.5`는 같은 이름의 브랜치(02df346)를 받습니다. 반드시 tag(`refs/tags/v1.4.5`)로 받아야 합니다.
- **B block.** `D1_설계_v8.xlsx`의 B block은 PPT 계획(80셀 / 160 TS)과 다릅니다. 파일럿은 v8을 그대로 씁니다(SPEC §6).

---

## 9. 파일럿 이후 결정 사항 (SPEC §10)

1. **TS 엔진:** ORCA(기본) 또는 G16(Coley와 정확히 같은 조건, 라이선스가 있을 때). 근거는 A5와 D0_control의 구조 차이입니다.
2. **D1 기준 규칙:** Coley 규칙 유지(기본, D0와 같은 조건) 또는 lowest TS-compatible(D-3). D-3을 택하면 D0도 다시 기준을 잡아야 두 데이터셋을 비교할 수 있습니다.
3. **B block:** 전체 실행 전에 PPT 계획대로 다시 만들지 여부.
4. **`eda_nprocs`:** 지금은 1입니다. EDA 시간이 전체를 좌우할 때만 올립니다. 에너지는 이 값과 무관합니다.

---

## 10. 실행 후 산출물

1. SPEC §9 순서로 요약합니다.
   - A1–A5 표
   - 실패 목록
   - 라벨 표
   - 대조 표
   - configuration/기준 구조 표
   - 비용과 외삽
   - go / no-go 권고
2. 브랜치에 커밋합니다.
   - 대상: 코드, `pilot_manifest.csv`, `results/{pilot_report.md, pilot_summary.csv, labels_d1_pilot.json, S0_REPORT.md}`
   - 메시지: `d1 autodE pilot: <n_ok>/10 D1 labels, controls <PASS/FAIL>`
   - scratch 내용은 커밋하지 않습니다.
3. D1 전체 실행은 시작하지 않습니다(SPEC §0-5).

---

## 11. 실행 결과 (2026-09-29)

제출한 job은 S0 995656, array 995657(0–13), report 995658이며 모두 cpu2에서 돌았습니다. 결과 파일은 `d1_autode_pilot/results/`에 있습니다.
- `pilot_report.md`
- `pilot_summary.csv`
- `labels_d1_pilot.json`
- `S0_REPORT.md`: 약 730 KB이고, 대부분 D0 입체 표지 검증 중 나온 RDKit "More than one matching pattern" 경고입니다.

| id | 결과 | 값 |
|---|---|---|
| S0 | 통과 | 버전 확인 · 입체 표지 검증 0.945 · 입력 parity 50/50 byte 동일 · xtb scan 형식 · HCNO + C₂H₄ smoke 모두 통과. g16은 PATH에 없음(C5) |
| A1 | 통과 | D1 8/10 |
| A2 | 통과 | 생성된 라벨 12개 모두 gate 통과 |
| A3 | 통과 | D0_replay의 9개 target, max \|Δ\| = 0.003 kcal/mol (J13 d1). J12는 약 1e-9 |
| A4 | 통과 | 기준 규칙 결정이 Coley와 일치 (rxn 20: alt 사용, rxn 105: 미사용) |
| A5 | **실패 (진단용)** | D0_control \|Δbarrier\| rxn 20 = 0.37, **rxn 105 = 2.01** |

**실패한 D1 job 2개**

| job | 원인 | 분류 |
|---|---|---|
| J03 (C2, 니트로 carbonyl ylide + 에틸렌) | autodE 경로(xtb, DFT 모두)에 에너지 최댓값이 없음. 이 계산 수준에서는 배리어 없는 반응으로 보임 | TS 탐색 실패 (화학적 원인) |
| J07 (D5, 메틸아자이드 + 다이메틸아미노 사이클로옥타인) | TS는 정상 (주 허수 모드 −360.7 cm⁻¹, 형성 결합 비중 0.96)이나, −14.9 cm⁻¹ 모드가 하나 더 있어 `n_imag == 1` gate에서 탈락 | gate 기준 문제 (파이프라인 버그 아님) |

**A5 진단 (label − `labels_all`, kcal/mol)**

| rxn | Δbarrier | Δd1 | Δd2 | Δelst | Δpauli | Δoi | Δcpcm | Δd_form max | ΔG‡ − Coley G_act |
|---|---|---|---|---|---|---|---|---|---|
| 20 (J10) | −0.37 | −1.21 | −1.02 | +8.63 | −9.32 | +4.31 | −1.33 | 0.059 Å | −4.22 |
| 105 (J11) | −2.01 | −2.69 | +0.50 | +3.42 | +9.70 | −3.87 | −8.67 | 0.117 Å | −2.09 |

ORCA로 찾은 TS의 형성 결합 길이가 D0(G16)와 0.06–0.12 Å 다릅니다. 그 결과 elst, pauli, oi, cpcm 채널이 최대 약 10 kcal/mol 달라집니다. barrier 합계의 차이는 그보다 작습니다.

**비용 (report.py 기준)**
- D1 TS 한 행당 평균 11.5 core-hour (중간값 10.3, 최대 22.8). 단계별로 ts 8.2, ref 0.5, sp 2.8입니다.
- 2,846행으로 외삽하면 약 32,600 core-hour이고, 10 job × 8코어로 돌리면 약 17일 걸립니다.
- 파일럿 전체 벽시계 시간은 약 3시간이었습니다. §6.4의 추정(1–2일)보다 훨씬 짧았습니다.

**남은 결정 사항**
1. **J07 gate:** 현행 유지, 또는 −40 cm⁻¹보다 작은 모드만 허수로 세도록 완화(config의 `min_imag_cm: -40`과 같은 기준).
2. **A5:** ORCA TS와 G16 TS의 차이로 채널 라벨이 체계적으로 달라지는지 확인해야 합니다. SPEC §10-1 TS 엔진 결정과 D0·D1 비교 가능성에 직결되는 문제입니다.
3. **J03 유형:** D1 C 패널에 배리어 없는 carbonyl ylide 계열이 얼마나 있는지 확인이 필요합니다.
4. **SPEC §10의 2–4항:** 기준 규칙, B block, `eda_nprocs`.

---

## 12. 본실험 전 검증 결과 (2026-09-30)

- 기준 문서: [`validation/VALIDATION_SPEC.md`](validation/VALIDATION_SPEC.md)
- 전체 보고서: [`validation/results/VALIDATION_REPORT.md`](validation/results/VALIDATION_REPORT.md)
- 재실행 기록: [`validation/RETRIES.md`](validation/RETRIES.md)
- 사전 등록 커밋: `43b962b4` (`config_val.yaml`은 `85214ce9`). 실행 후 사전 등록 파일은 수정하지 않음.
- 실행 job: worker 996236(10개), 재실행 996555(5개), 보고서 996766. 148 task 중 147개 완료, 1개 실패(V7).

### 판정 (사전 등록 기준)

| id | 결과 | 값 |
|---|---|---|
| A1 | PASS | D1 19/20 ok. 실패 1건은 J03(TS 없음, 허용 실패)이고, `engine_abnormal`과 파이프라인 결함은 0건 |
| A2 | PASS | 생성된 라벨 113개가 모든 gate 통과 |
| A3 | PASS | D0 replay 24/24, 최대 \|Δ\| 0.013 kcal/mol |
| A4 | PASS | 기준 구조 규칙이 Coley와 24/24 일치 |
| A5-eng | **FAIL** | L0 재최적화: Pauli RMS 1.62 (기준 1.01), OI RMS 0.78 (기준 0.58). TS 이동 중앙값은 0.009 Å. L2는 전 채널 PASS |
| A5-run | 보고 전용, **NF > 1** | 반복 실행 SD: Pauli 6.97, elst 4.07, OI 3.99, barrier 2.75 kcal/mol (NF 최대 3.45) |
| A5-e2e | PASS | n = 23. 편향 기준과 초과 분산 기준 모두 통과 |
| A6 | PASS | 같은 입력 ΔE 0 Eh · nprocs 1 vs 4 1.4×10⁻⁹ Eh · 같은 OptTS 두 번 3×10⁻¹⁵ Å |
| A7 | PASS | 음성 대조 8/8 검출 |
| A8 | PASS | D0 ok 5,260건 탈락 0건, 제외 5건은 모두 탈락 |
| A9 | 보고 전용 | D1 행당 10.7 core-h (95% CI 6.1–15.4). 2,846행이면 약 30,600 core-h, 10 job × 8코어로 약 16일 |
| A10 | 판정 불가 | J07은 F1 적용 후 ok. J03 scan은 ORCA가 같은 지점(89/100)에서 두 번 비정상 종료 |

**§8 권고: 조건부.** 해당하는 행은 두 개입니다: "A5-eng FAIL (L0), L2 기준 충족"과 "A5-run NF > 1".

### 데이터가 말하는 것

- **H0 (같은 입력 → 같은 결과):** 성립. 같은 ORCA 입력은 에너지, 라벨, 구조가 완전히 같았음.
- **엔진 차이 (H1):** Coley TS를 ORCA L0로 재최적화했을 때 TS는 거의 움직이지 않음(중앙값 0.009 Å). barrier 변화는 평균 0.04 kcal/mol.
- **반복 실행 잡음 (H2):** 결과를 좌우하는 요인.
  - 같은 반응을 autodE로 다시 돌리면 같은 TS를 찾는 비율이 50%(반복 쌍 기준)입니다.
  - SMILES부터 돌린 D0 대조가 Coley와 같은 TS를 찾은 비율은 48%입니다.
  - 그래서 채널 라벨은 autodE가 어느 TS를 찾았느냐에 따라 SD 2–7 kcal/mol로 흩어집니다. **이는 D0 라벨에도 똑같이 해당합니다.**
- **L2 비용:** L2(NORI, DefGrid3) 재최적화는 L0의 약 11배입니다(12.4 vs 1.1 core-h, 같은 Coley TS에서 출발).

### 사후 관찰 (사전 등록 밖, 판정은 바꾸지 않음)

- **A5-eng FAIL은 반응 하나(rxn 3530) 때문입니다.**
  - rxn 3530: Pauli −7.63, OI +3.60, elst +3.54 kcal/mol. 그러나 barrier 변화는 −0.04이고, TS 이동은 0.048 Å, 형성 결합 이동은 0.027 Å였습니다.
  - 이 반응을 빼면 RMS가 Pauli 0.47, OI 0.26으로 기준보다 훨씬 작습니다.
  - 즉 barrier는 거의 그대로인데, TS가 조금만 움직여도 Pauli와 OI가 서로 반대 방향으로 크게 바뀌는 경우입니다. TS 한 점에서만 분석할 때의 위험(Fernández & Bickelhaupt 2014)과 같은 현상입니다.
- **파이프라인 결함:** `assemble.py`의 `sum_mismatch_promoted` 버그를 찾아 고쳤습니다(`d573111e`). 이 버그가 있으면 승격이 필요한 라벨이 모두 멈추므로, 본실험 전에 꼭 필요한 수정이었습니다.
- **ORCA 하위 프로그램 비정상 종료 4건, 원인 미확인:** autodE가 stderr를 남기지 않아 원인을 알 수 없습니다.
  - V7 scan과 V3_0020_r2는 같은 계산에서 반복됐습니다.
  - 나머지 2건은 재실행에서 성공했습니다.
- **V4_0500:** autodE 내부 오류(원자 0개 conformer)로 실패했습니다.

### 사용자 결정이 필요한 것 (본실험 전)

1. **엔진 효과(A5-eng) 대응:**
   - (a) 본실험 TS 수준을 L2로 변경. 재최적화 비용만 약 11배이고, 전체 autodE 탐색이면 더 늘어납니다.
   - (b) D0 TS를 L0로 재최적화해 균질화. D0 재최적화 비용은 반응당 약 1 core-h(V2b 실측)이고, 여기에 SP 재계산이 더해집니다.
   - (c) 위 사후 관찰을 근거로 대안 판정을 병기하고 L0를 유지.
2. **라벨 정의(A5-run NF > 1):** TS 한 점 라벨을 그대로 쓸지, 아니면 TS conformer 여럿의 최저·평균이나 고정 반응 좌표 지점 라벨로 바꿀지 결정해야 합니다. D0에도 적용되는 문제입니다.
3. **J03:** 89점까지만 계산된 scan을 진단용으로 분석할지 결정해야 합니다. D-1 규칙상 J03은 이미 허용 실패로 처리돼 A1에는 영향이 없습니다.
4. **ORCA 오류 원인 추적:** 본실험에서 ORCA stderr를 남길 방법이 필요한지 결정해야 합니다.
