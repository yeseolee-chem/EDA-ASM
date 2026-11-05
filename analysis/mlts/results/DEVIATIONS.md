# DEPLOY-V2 (MLTS) 편차 기록

명령서 `docs/specs/DEPLOY_V2_MLTS.md`와 다르게 한 것만 모은다. 결정 경위는 `results/DECISIONS.md`.

## FT — AIMNet2 학습 목표와 내장 분산 항 (2026-11-05, 사용자 요청 4)

- **코드로 확인한 사실** (`finetune.py` `aim_cfg` · `B3LYP_D3BJ`, aimnet `modules/lr.py` `DFTD3`, `models/aimnet2_dftd3_wb97m.yaml`):
  - 공개 AIMNet2(`aimnet2_wb97m_d3_0`)의 분산 출력은 `aimnet.modules.DFTD3`이다. BJ 감쇠, C6 · C8 항, 3체 항 없음.
  - 원래 매개변수는 ωB97M용(s6 1.0, s8 0.3908, a1 0.5660, a2 3.1280)으로, 라벨(B3LYP-D3BJ)과 **다르다**.
  - 미세조정에서는 이 항을 **B3LYP-D3(BJ) 매개변수(s6 1.0, s8 1.9889, a1 0.3981, a2 4.4211)로 바꿔 끼운다**. 이는 Grimme B3LYP 값이고, Phase H의 D3_int와 같다.
  - 장거리 상한 없이 모든 원자쌍을 더한다(`add_cutoff(cutoff_lr=inf)`).
- **라벨 쪽**: ORCA `D3BJ`는 2체 항만이다. H-0에서 이 값으로 계산한 D3_int가 라벨 disp 채널과 중앙값 5.1e-7 kcal/mol로 일치했다.
- **학습 목표 처리**:
  - 학습 에너지 = 라벨 총에너지(eV) − 원소별 고정 상수(E_fit, float32 반올림 방지)이고, 원소 기준 에너지는 최소제곱으로 맞춘다.
  - 모델 = 신경망(전자 부분 + 장거리 Coulomb) + 고정된 B3LYP-D3(BJ) 항이다. **분산은 학습하지 않고 라벨과 같은 식으로 더하므로, 신경망은 라벨 − D3(B3LYP-BJ) − E0의 나머지만 배운다.**
  - 손실은 에너지 1 : 힘 0.2, 분자 전하 입력은 구조별 total_charge다.
- **편차인 점**: 명령서는 기반 MLIP를 그대로 미세조정한다고만 했다. 분산 항을 라벨 함수의 것으로 바꾼 것은 명령서에 없는 선택이다. 이유는 라벨의 분산 기여(구조만의 함수)를 정확히 맞추고, 신경망이 범함수 차이를 분산 오차와 섞어 배우지 않게 하려는 것이다.

## Z / FT — MACE-OMOL-0 보류 (2026-11-05, 사용자 요청 1)

- `FTpilot_MACE-OMOL-0_0`: 학습률 1e-3, 5,180만 매개변수 전체 미세조정(float32)에서 epoch 0 뒤 검증 손실 NaN(시작 전 검증 힘 MAE 200 meV/Å는 정상) → **설정 실패, 보류**(탈락 아님).
- 두 후보의 PES 품질은 비교하지 않았다.
- 기반 MLIP = AIMNet2로 **조건부 확정**한다(fold 0 FT-2 (a) 통과가 조건).
- 실패 시 순서(사용자 요청 5): G-4 능동학습(fold 0) 뒤 재측정 → 그래도 실패면 MACE-OMOL-0 lr ≤ 1e-4로 fold 0만 재시도 → 둘 다 실패면 STOP.

## FT — fold 1–4 순서 (2026-11-05, 사용자 요청 2)

- fold 1–4 학습은 fold 0 FT-2 (a)(라벨 TS 출발)가 통과한 뒤 시작한다. 이에 따라 앞서 낸 미리 돌리기(FT-spec) 8개를 취소했다.
- 명령서 5절은 2차 미세조정 뒤에만 FT-2 (a) 관문을 두지만, 사용자 결정으로 fold 0 1차에도 관문을 둔다.

## FT — 감사 결과와 누수 조치 (2026-11-05, 사용자 요청 ①②③, results/AUDIT_REPORT.md)

- **① 출처 일치**: 네 출처(S1 D1 G + Phase I, S2 Nurion TS 근방, S3 Nurion 기타, S4 D0) 모두 ORCA 6.1.1 · B3LYP(VWN-5) · def2-SVP + def2/J RIJCOSX · D3BJ(B3LYP) · CPCM(Water) + SMD · 기본 grid로 라벨 수준과 같다. 출처마다 학습 20구조를 라벨 수준 EnGrad로 다시 계산한 결과, 힘 MAE 0.0005–0.0023 kcal/mol/Å · 반응 안 상대 에너지 MAE 0.001–0.007 kcal/mol로 기준 0.2보다 훨씬 작다 → **모두 유지(제외 없음)**.
  - 남는 차이: autodE LooseOpt 단계(Nurion path · fwd_bwd, Phase I d0_irc)는 SCF가 TolE 1e-6로 느슨하다. 실행마다 SCF 시작점에 따른 에너지 잡음이 최대 약 1e-4 Eh(≈ 0.06 kcal/mol)다(검토자: SCF 뒤 gridx3 교환 재계산의 차). 둘 다 기준보다 훨씬 작다.
- **② 정체성 누수(fold 0)**: fold 0 시험 반응 1,843개(a 1,791 · b 53)의 반응물 canonical SMILES 쌍을 학습 · 검증 자료 전체와 대조했다.
  - 입체 표기를 지운 기준: 학습 50(+ 검토자가 찾은 같은 truncated 하위 트리의 1-조각 행 → 최소 85) · 검증 1,321.
  - 입체 표기를 지키는 기준(사용자 표현 "canonical SMILES" = isomeric): 학습 31(+ 판정 불가 19) · 검증 0(+ 판정 불가 11).
  - 학습 쪽 겹침은 모두 **Nurion 잘린 계(truncated system)** 구조다. 줄인 반응물이 시험 반응물 쌍과 같아도 표지로는 드러나지 않고, 108행은 정체성을 확인할 수도 없다. 검증 1,321은 E/Z 형제 core(예: nitrile oxide AppE 시험 대 AppZ 검증)로, 입체를 지키면 다른 반응이다.
  - **조치(추천안)**: 사용자 규칙("0이 아니면 제거 후 재학습")에 따라 **Nurion truncated 종류(nur_truncated, nur_x_truncated) 전체를 모든 FT 자료에서 뺀다**(finetune.py LEAK_DROP_KINDS, CODE_VERSION mlts-ft-6). 이렇게 하면 이 경로의 누수가 모든 fold에서 없어진다(85 + 확인 불가 108행 포함).
  - 누수 자료로 학습 중이던 AIMNet2 시범(1062600)은 취소하고, 새 분할 → 누수 재검사(verify_data.py leak) → 재학습을 afterok로 이었다.
  - 판정 기준은 사용자 표현대로 입체 표기를 지키는 canonical SMILES 쌍이고, 입체를 지운 수치는 함께 보고한다.
  - 분자 단위 겹침(시험 반응의 반응물과 같은 분자; frag · d0_frag 포함 최대 7,150행, 연결만 본 상한)과 생성물 겹침(356)은 반응물 쌍 규칙 밖이라 따로 보고만 한다. core 단위 T1 분할에서는 다른 core끼리 dipolarophile 분자를 공유하므로 피할 수 없다.
- **③ D3 일치**: 라벨 TS 20개(Si 5 · S 5 · F 2 · Cl 2 · Br 2 · CHNO 4)에서 바꿔 끼운 B3LYP-D3(BJ)가 ORCA 출력의 분산 보정과 max |ΔE| 4.1e-8 Eh · |ΔF| 7.5e-9 Eh/bohr로 일치한다(float32 학습 상태, float64, simple-dftd3 모두 1e-6 기준 통과). AIMNet2 내부 CN 상한이 ORCA와 맞추는 요소다(없애면 S 포함 구조가 최대 1.7e-2 Eh 어긋남).
- **추가**: FT-2 최적화기 = ORCA `! ExtOpt OptTS`(mlts_bridge, D1 geom 블록, MLIP 첫 Hessian, 30 cycle마다 새 MLIP Hessian, TightOpt). Sella는 C-1 대조용이다. C-1 (ii)를 미세조정된 AIMNet2(누수 전 checkpoint)로 다시 하면 5/5 통과(최대 0.00077 Å, 허수 1개)지만, 학습 반응 · NormalOpt에서 잰 것이라 FT-2의 근거는 아니다. 출처별 FT-1(누수 전 checkpoint, 참고): 검증 TS 근방 힘 0.467 · 시험 0.527 kcal/mol/Å, D0 평가 전용 0.917(> 0.6).
