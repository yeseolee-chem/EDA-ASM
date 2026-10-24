# D1-G3 명령서 이탈 기록 (DEVIATIONS)

명령서 `docs/specs/D1G3.md`와 다르게 진행한 것을 모은다. 명령서 11절의 갈림길 결정은 `DECISIONS.md`에 따로 있다.
수치는 아래 적은 파일에서 옮겼다.

## O-23 (사용자 지시 2026-10-24): 비용 예상과 3,000 코어시간 한도 생략

- 사용자 지시: "비용추정도하지말고 3000core-h도 필요없음 그냥 본계산 10개 넣어줘".
- 한 일:
  - 대기 중이던 예상 job 1044263을 실행 전에 취소했다.
  - `opt.py` `ready()`에서 `O_estimate.json` go 조건을 뺐다(commit 12090a43).
  - 본 계산 배열 1044264(10 task)를 11:34에 파일럿 1044262와 함께 돌렸다.
- 그 결과 명령서 7절의 예상 단계(3,000 코어시간 판정)는 한 번도 평가되지 않았다.

## U-1 (사용자 결정 2026-10-25): Part A TS 계산 중단, 완료분 보존

- 2026-10-25 07:11에 job을 취소했다.
  - 본 계산 배열 1044264(실행 중 10 task)
  - assemble 1044265
  - batch 3 launcher 1044307
- 그때까지 끝난 118개 단위(`$G3/O/units/<uid>/unit.json`)는 그대로 둔다.

  | 단위 | 개수 | 결과 |
  |---|---|---|
  | ts.g3h | 19 | 모두 ok |
  | ts.g3r | 37 | 모두 ok |
  | ref.g3h | 12 | |
  | ref.g3r | 12 | |
  | Part B | 38 | ok 34, fail 4 |
  | T-5 | 3 | |

- 06:45 시점의 117개는 `results/O_done_117.json`에 있다.
- 취소 때 계산 중이던 TS(g3h 30, g3r 10)는 결과 없이 끝났다.
- SLURM이 할당한 코어시간(sacct CPUTimeRAW, g3* job)과 끝난 단위의 합계는 이렇다. 차이에는 취소된 계산과 task 끝의 빈 시간이 들어 있다.

  | 구분 | 코어시간 |
  |---|---|
  | SLURM 할당 합계 | 6,309 |
  | 끝난 단위의 기록 합계 | 2,449 |
  | 차이(결과 없이 쓴 시간) | 약 3,860 |

- Part B 남은 2반응(B.1003803, B.1003824)은 사용자 결정 2에 따라 마무리한다. `opt.py work sel --only results/O_sel_partB.txt`, job 1046693.
- assemble은 선택되지 않은 (방법, 반응)에 `.fail_stopped_user` 표지를 쓴다(DECISIONS U-1b).

## U-3a (사용자 결정 3(a)): 12절 O STOP 조건(예상 합계 > 3,000 코어시간)

- 실제 사용량:
  - 끝난 단위 합계는 2,438 코어시간이다(117개, `O_done_117.json`; 118개 기준 2,449).
  - TS 하나당 코어시간 중앙값:

    | 방법 | 중앙값 | 끝난 수 | 범위 | 합계 |
    |---|---|---|---|---|
    | g3h | 79.2 | 19 | 0.7–146.2 | 1,294 |
    | g3r | 14.4 | 37 | 0.7–113.7 | 710 |

  - 계획된 TS는 방법마다 160개다. 중앙값만으로도 g3h 12,666, g3r 2,298 코어시간이 되어, 예상 합계는 3,000을 크게 넘는다.
- **파일럿 추정이 왜 통과했는가: 통과하지 않았다. 추정이 실행되지 않았다.**
  - 위 O-23대로, 파일럿이 돌던 중 사용자 지시로 예상 job을 실행 전에 취소했다.
  - 본 계산은 예상 판정 없이 시작되었다. 그래서 7절의 STOP이 작동할 기회가 없었다.
- **사후 계산 (U-3a, `results/O_estimate_retro.json`, job 1046695):**
  - 같은 파일럿 단위(T-1 1반응 + 방법별 5반응 + 그 기준 구조, Part B 2반응)에 O-13 규칙을 그대로 적용했다. 규칙은 계열마다 max(파일럿 평균 × N, n_atoms³ 비례)이다.
  - 결과는 **6,022 코어시간 > 3,000 → STOP**이다.

    | 계열 | 코어시간 |
    |---|---|
    | ts_g3h | 2,724 |
    | ts_g3r | 2,785 |
    | ref_g3h | 128.5 |
    | ref_g3r | 101.7 |
    | B_optts | 228.6 |
    | B_sp | 10.3 |
    | F_hf | 41.8 |
    | F_xtb_features | 1.1 |
    | T5 사용분 | 0.2 |

  - 예상이 돌았다면 본 계산 전에 멈췄다.
- **참고: 파일럿도 비용을 낮게 보았다.**
  - 파일럿 TS 코어시간(크기 순서 10/30/50/70/90 % 위치 반응)의 평균은 g3h 14.5, g3r 14.8이다. 범위는 각각 0.7–39.6, 0.7–68.9이다.
  - 본 계산 queue는 무게(n³) 큰 순서였다. 그래서 끝난 본 계산 TS는 큰 반응 쪽으로 치우쳐 있고, 위 중앙값 79는 160반응 전체의 중앙값보다 클 수 있다.
  - 어느 쪽으로 보아도 3,000은 넘는다.
- 비용이 큰 원인:
  - HF·r2SCAN-3c 모두 해석적 Hessian이다(U-6). Calc_Hess + Recalc_Hess 5 + 마지막 Freq로 TS 하나에 Hessian을 3–5번 계산한다.
  - HF/6-31G\* SMD Hessian은 8코어에서 CP-SCF 반복 1회에 약 25분, Hessian 1회에 약 4.5시간이 걸렸다(진행 중 출력에서 확인).

## U-3b (사용자 결정 3(b)): 판정 ②에서 형태 유지율 제외

- 형태 유지 규칙은 같은 수준 재현성을 보려는 D1 검증 규칙이다. 수준이 다른 구조 사이에는 거의 성립하지 않는다.
  - 규칙: `rmsd.same_ts`, 무거운 원자 RMSD 0.1 Å · 형성 결합 0.02 Å.
  - 기준선: G1(xTB) TS 대 라벨 DFT TS가 같은 비율이 1.17 %다(`analysis/d1_g2/results/C_summary.json` `frac_g1_eq_dft_baseline` = 0.01174).
- 이 시범에서 관찰한 값:
  - 끝난 TS의 형태 유지(G2 출발 TS와 같음)는 g3h 0/19, g3r 0/37이다.
  - Part B는 same_as_g2_start 1/38, converged_to_label 0/38이다(관찰만; Part B 규칙은 바꾸지 않음).
- 바꾼 것:
  - 9절 ② = **TS 성공률 ≥ 0.9만** 본다.
  - 형태 유지율은 설명용 수치로만 보고한다.
  - 9절의 나머지(①, ③, 부분/불가 규칙)는 그대로다.
- 명령서 11절("판정 규칙은 바꾸지 않는다")과 다르다. 사용자 결정이므로 여기 적는다.

## U-2 (사용자 결정 5): g3r 완료 반응으로 완성되는 짝만 F·E (설명용)

- 대상:
  - `results/S_pairs.csv`의 짝 중 a·b 두 반응 모두 ts.g3r이 ok인 짝이다(`results/U1_selection.json`).
  - g3h는 F·E를 하지 않는다(stopped_user).
- 그 반응에 필요한데 아직 없는 ref.g3r 단위는 `opt.py work sel`로 계산한다.
- 결과는 명령서 9절의 정식 판정이 아니라 설명용이다. 표본이 계획(same 80짝, different 20짝)보다 작다.

## U-4 (사용자 결정 4): Part B TS의 HF EDA 단일점 (설명용, 추가)

- Part B ok TS에서 D1-HF runner EDA 단일점만 계산한다(geom = partB).
- 라벨 TS의 HF 채널(elst, pauli, oi, cpcm)과의 |차이| 중앙값·최대를 ΔE‡와 함께 표로 낸다(`results/B_hf.csv`, `B_hf_summary.json`).

## U-6 (사용자 결정 6): Hessian 종류 확인

다음 9개 출력 모두 ORCA `SCF HESSIAN`(해석적 2차 미분; 출력에 "SCF Hessian ... YES")이다. 수치 Hessian 표시는 없다.

| 종류 | 단위 | 입력 줄 |
|---|---|---|
| g3h | ts.g3h.1000326, ts.g3h.1000485, ts.g3h.1000715 | `! HF D3BJ 6-31G* TightSCF OptTS Freq CPCM(water)` |
| g3r | ts.g3r.1000326, ts.g3r.1000485, ts.g3r.1000715 | `! r2SCAN-3c TightSCF OptTS Freq CPCM(water)` |
| Part B | B.1001022, B.1002963, B.1002979 | `! OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J CPCM(Water)` |

출력 하나에 Hessian 계산이 2–4번 있다.

## U-7 (사용자 결정 7, 신규·사용자 승인): G3L 시범

- 대상: `same` 분류 반응 20개(G2 xTB Hessian이 있는 반응).
- 입력:
  - Part B와 같은 라벨 키워드(D1 autodE L0 OptTS)를 쓴다.
  - 초기 Hessian은 G2 xTB Hessian(`InHess Read`)이고, Recalc_Hess는 없다.
- 상한은 200 코어시간이다.
- 보고 항목: 코어시간 중앙값, TS gate 통과율, 라벨 TS와의 형성 결합 차이·RMSD, converged_to_label(`results/G3L_status.csv`, `G3L_report.json`).

## 그 밖의 기록 (DECISIONS.md 참고)

- O-22: T-1의 기저 확인을 완화했다. r2SCAN-3c는 보조 기저 def2-mTZVPP/J도 찍기 때문이다.
- 비정상 종료 2건: g3h 2개가 HF Hessian 중 `orca_scfresp_mpi` segfault(signal 11, n116)로 끝났다. O-5 규칙대로 2차 시도로 넘어갔고, U-1 취소 때 끝나지 않았다.
