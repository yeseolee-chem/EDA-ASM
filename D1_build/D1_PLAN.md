# D1 autodE 파일럿 — 진행 계획과 확인 사항

작성 2026-09-29 · 기준 문서 [`d1_autode_pilot/SPEC_D1_pilot.md`](d1_autode_pilot/SPEC_D1_pilot.md)

**현재 상태:** 환경 설치와 정적 점검까지 끝났습니다. 파일럿(`submit_pilot.sh`)은 **아직 제출하지 않았고**, 실행 지시를 기다립니다.

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
