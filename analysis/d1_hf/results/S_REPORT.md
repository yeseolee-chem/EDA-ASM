# D1-HF Phase S — smoke와 재현 검사

S-1은 원래 B3LYP/def2-TZVP 입력(`%pal`만 1)으로 두 반응의 9개 값을 라벨과 6.3e-8 kcal/mol 안에서 재현했다. S-2는 F·Si·S·Cl·Br 반응과 가장 큰 63원자 반응, 모두 6개를 HF/6-31G\*로 계산했고 모두 정상 종료·gate 통과였다. S-3의 HF EDA 표 항목은 B3LYP와 같은 8개이며, 설명되지 않는 항목은 없다.
수치 출처: `S_select.csv`, `S1_compare.csv`, `S2_status.csv`, `S3_eda_table.txt`; 계산 폴더는 `$HF/runs/dft_orig1/`와 `$HF/runs/dft/`이다.

## 선택 (`S_select.csv`)

| 검사 | 반응 | 원자 수 | 이유 |
|---|---|---|---|
| S-1 | T00312, T00317 | 8, 8 | 원자 수가 가장 작은 2개(같으면 job_id 순) |
| S-2 | T00318 | 8 | F를 포함하는 반응 중 가장 작은 것 |
| S-2 | T00328 | 20 | Si를 포함하는 반응 중 가장 작은 것 |
| S-2 | T00327 | 12 | S를 포함하는 반응 중 가장 작은 것 |
| S-2 | T00345 | 8 | Cl을 포함하는 반응 중 가장 작은 것 |
| S-2 | T00335 | 8 | Br을 포함하는 반응 중 가장 작은 것 |
| S-2 | T04493 | 63 | 가장 큰 반응 |

## S-1 재현 (`S1_compare.csv`)

- 9개 값(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier)이 두 반응 모두 `label.json`과 같다. 최대 |차이|는 6.3e-8 kcal/mol(T00312 elst)로, 기준 0.01을 크게 밑돈다. 모든 gate를 통과했다.
- 처음 조립(job 1032345)은 g_enuc 하나만 실패로 판정했다. 원인은 ORCA가 `eda_frag*.inp` 좌표를 소수 7자리로 반올림해 ENuc가 1e-6 Eh 수준으로 달라지는 것이다. 라벨 파이프라인(`d1_run/gates.py`)과 같은 좌표 비교(1e-6 Å)로 고친 뒤 계산을 다시 돌리지 않고 조립만 다시 했다(job 1032380).

## S-2 원소 (`S2_status.csv`)

- 6개 모두 첫 시도에 정상 종료했고 재시도는 없었다. 판정 gate(g1_asm, g2_sum, g3_seam, g_enuc, g_bond_ref, g_scf, g_setting, orca)를 모두 통과했다.
- 명령서 기준(g2 < 0.01, g3 < 1e-3)으로도 6개 모두 통과한다.
  - Σ6채널 − bond: 최대 |5.0e-6| kcal/mol
  - bond 기준 잔차: ≤ 2.5e-8 kcal/mol
  - B3LYP 라벨에서는 같은 양이 최대 0.098이었다(COSX).
- ORCA는 `HF D3BJ`와 `6-31G*`를 모두 받았다. 출력의 방법은 `Hartree-Fock(GTOs)`, 기저는 `6-31G*`이다. SMD 물(ε 78.3550)과 CDS가 7개 출력 모두에 있고, ORCA는 6.1.1 GIT 487d211c이다.
- 코어시간(1코어 wall): 8원자 반응 0.005–0.011 h, 20원자 0.030 h, 63원자 0.50 h. 63원자 반응은 EDA가 대부분이다.

## S-3 EDA 표 (`S3_eda_table.txt`, T00318, HF/6-31G\*)

```
     Energy Term                      Hartree         Kcal/mol
     Bond Energy                   -0.0074293857        -4.66
     Orbital Energy                -0.0223658367       -14.03
     Electrostatic Energy          -0.0314598056       -19.74
     Pauli Energy                   0.1435191914        90.06
     Delta E^0(XC)                 -0.0815925785       -51.20
     Delta Dispersion              -0.0119328864        -7.49
     Delta CPCM Dielectric          0.0013874090         0.87
     Delta SMD CDS correction      -0.0049848807        -3.13
```

대응을 확정했다.

| 채널 | 표 항목 |
|---|---|
| bond | Bond Energy |
| oi | Orbital Energy |
| elst | Electrostatic Energy |
| pauli | Pauli Energy + Delta E^0(XC) (HF에서는 교환 항) |
| disp | Delta Dispersion |
| cpcm | Delta CPCM Dielectric |
| cds | Delta SMD CDS correction |

이 밖의 항목은 없다.

## 명령서와 다르게 한 점 (사용자 지시 "모든 분기는 추천대로"에 따라 택함)

1. **조각 입력 4개의 SMD 표기:** 라벨 입력은 `SMD(water)` 대신 `CPCM(water)` + `%cpcm smd true / smdsolvent "water"`이다. 이는 라벨 규약이며 같은 SMD 모델이다. X 검사에서 이 형식을 SMD로 인정했고, 입력은 바꾸지 않았다. 출력에서 SMD가 7개 계산 모두에 닿았음을 확인했다(g_setting).
2. **gate 판정 기준:** 명령서 표의 g2 0.01 / g3 1e-3은 "라벨과 같음"이라는 설명과 다르다. 판정은 라벨 기준(0.11)으로 하고 명령서 기준은 `*_spec` 열로 보고한다. HF에서는 두 기준이 같은 결과를 낸다.
3. **g_enuc:** ENuc를 직접 같다고 보지 않고 라벨과 같은 좌표 비교를 쓴다. ENuc 차이는 보고만 한다(`enuc_diff_frag*_eh`).
4. **gate 추가:** 라벨에 있던 bond 기준 gate(g_bond_ref, 0.02)와 SCF 수렴 gate(g_scf)를 더했다.
5. **`bsse_total` 부호:** 명령서대로 bond − eint_spe로 두고, 라벨 부호인 `bsse_shift_kcal`(= eint_spe − bond = c_ghost)도 함께 저장한다.
6. **실행 환경:** ORCA 6.1.1을 1코어로 돌리는 전용 환경(`jobs/hf_env.sh`)을 썼다. MPI는 쓰지 않는다. `jobs/g_env.sh`는 D1PAIR G1 전용(시스템 OpenMPI 등)이라 쓰지 않았다.
7. **EDA 재시도:** ORCA는 EDA 조각 입력을 직접 만들면서 `%scf` 블록을 넘기지 않는다. 그래서 재시도 때 EDA 조각 SCF에는 SlowConv만 들어간다. 실제 MaxIter는 `attempts.tsv`에 적는다. S에서는 재시도가 없었다.
