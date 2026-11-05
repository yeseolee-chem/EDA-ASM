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
- 기반 MLIP = AIMNet2로 **조건부 확정**한다(fold 0 FT-2 (a) 통과가 조건; 11-05 15:5x부터 관문은 fold 4, 아래 'FT2-p — 1차 관문 fold 0 → fold 4' 절).
- 실패 시 순서(사용자 요청 5): G-4 능동학습(fold 0) 뒤 재측정 → 그래도 실패면 MACE-OMOL-0 lr ≤ 1e-4로 fold 0만 재시도 → 둘 다 실패면 STOP.

## FT2-p — fold 0 FT-2 판정 규칙: 실패 분류 · recalc 5 재측정 · E/Z 형제 나눔 (2026-11-05, 사용자 요청 (1)–(3))

FT-2 결과를 보기 전에 정했다(`pipeline.decide_ft2p` docstring, `ts_geom.py` · `ez_groups.py` docstring).

- **반응별 기준과 실패 분류**: 수렴 · 허수 1개(MLIP Hessian −50 cm⁻¹ 아래, 전과 같음) · 같은 TS · |Δd| ≤ 0.030 Å(FT-2 (a) 90 % 기준값)이면 통과. 실패는 (i) 수렴 안 함, (ii) n_imag ≠ 1, (iii) 다른 안장점(수렴 · 허수 1개인데 같은 TS가 아니거나 |Δd| > 0.030 Å)으로 나눠 출발(label · g2)별로 센다.
- **recalc 5 재측정**: 1차는 `analytic:30`(D1 config_run.yaml `Recalc_Hess 30`과 같은 빈도)이다. 1차에서 실패한 반응은 PES 실패로 보기 전에 같은 출발에서 MLIP Hessian을 5 cycle마다 새로 넣어(`analytic:5`, G2 · G3L식) 다시 잰다(`ts_geom.py run ... --recalc 5 --only-failed <1차 폴더>` → `<출발>_recalc5/`, FT2-p 단계 안). 판정은 반응마다 재측정 기록이 있으면 그것을 쓴다. 즉 recalc 5에서도 실패해야 실패다. 재측정 기록이 없는 실패 반응은 1차 실패 그대로 센다.
  - 명령서 5절은 한 번의 최적화로 FT-2를 잰다. 실패 반응만 Hessian을 더 자주 바꿔 다시 재는 것은 명령서에 없는 단계다. 이유: 수렴 실패 · 허수 개수 차이 중 일부는 PES가 아니라 30 cycle 동안 낡은 Hessian 때문일 수 있다.
- **E/Z 형제 나눔**: fold 0 시험 반응 가운데 학습 쪽 D1 core 반응(fold ≠ 0, fold −1 anchor core와 fold 0 검증 core 포함)과 입체를 지운 반응물 쌍은 같고 입체를 지킨 쌍은 다른 것(예: nitrile oxide AppE 시험 ↔ AppZ 검증 core)을 '형제 있음'으로 둔다(`$MLT/pipeline/lists/f0_test_ez.csv`, `ez_groups.py`). 11-05 목록: 1,874반응 중 형제 있음 77(a 75 · b 2), 없음 1,797(a 1,716 · b 51 · 짝 없음 31). 형제는 모두 `nitrile oxide/AppZ`(fold 2, fold 0 검증 core)다. 입체까지 같은 쌍은 학습 쪽에 0, InChI 확인 1,874/1,874 일치.
  - FT-2 (a)(라벨 출발) · (b)(G2 출발)를 두 무리로 따로 보고한다. **기준**: 라벨 출발 두 무리의 같은 TS 비율 차 > 0.05 **또는** |Δd| 중앙값 차 > 0.002 Å이면 fold 0 FT-2 (a) 관문은 '형제 없음' 무리만으로 판정한다. 그때 판정 줄과 이 파일 아래(auto 줄)에 적는다. 기준 안이면 fold 0 시험 전체로 판정한다(명령서대로).
- **바뀜 (11-05 15:5x, 사용자 "바꿔줘")**: 위의 관문 fold는 0이 아니라 **4**다(규칙은 그대로, 아래 'FT2-p — 1차 관문 fold 0 → fold 4' 절).
- **누수 재검사 (요청 (1))**: `verify_data.py leak`가 두 기준(입체 제거 · 입체 유지)을 a_rxn과 참조 b_rxn으로 나눠 보고한다(`part_leak.json` `by_role`). 재분할(mlts-ft-6) 뒤 1063008: 반응 쌍 겹침 a — 입체 제거 학습 0 · 검증 1,277구조(75반응), 입체 유지 0 · 0; b — 입체 제거 학습 0 · 검증 33구조(2반응), 입체 유지 0 · 0. 검증 겹침은 위 형제 77반응과 같다.

## FT2-r2 — 2차 FT-2 관문에도 같은 규칙: 실패 분류 · recalc 5 재측정 · E/Z 형제 나눔 (2026-11-05, 사용자 요청 (2) · (3))

2차 FT-2 결과를 보기 전에 정했다(`pipeline.decide_ft2r2` docstring). 위 FT2-p 규칙을 FT_<k>_al(fold k 시험, k = 0..4 합침)의 관문에 그대로 쓴다.

- **실패 분류 · recalc 5 재측정**: 각 (FT_<k>_al, 출발)의 1차가 끝나면 반응별 기준에 실패한 반응만 같은 출발에서 `analytic:5`로 다시 잰다(`<출발>_recalc5/`, FT2-r2 단계 안). 관문은 반응마다 재측정 기록이 있으면 그것을 쓴 합친 기록으로 판정한다. 실패 분류 (i)–(iii)는 출발별로 재측정 전 · 뒤를 5 fold 합침과 fold별로 적는다. `results/FT_geom.csv`도 합친 기록이다(열 `fail_class` · `remeasured` · `first_pass_fail_class`). 명령서 5절은 한 번의 최적화로 잰다. 1차 FT2-r1(보고용, G-4가 읽음)은 1차만 잰다.
- **E/Z 형제 나눔**: fold k마다 같은 정의(학습 쪽 = T1 fold ≠ k, fold −1 anchor core와 fold k 검증 core 포함)로 `$MLT/pipeline/lists/f<k>_test_ez.csv`를 만든다(`ez_groups.py run --folds 0,1,2,3,4`; 없으면 FT2-r2가 CPU job으로 만듦). 11-05 목록(1063043): 형제 있음 f0 77/1,874 · f1 0/1,879 · f2 237/1,503(맞는 core nitrile oxide/AppE 77 · nitrone/AppZ 160) · f3 160/1,435(nitrone/AppE 160) · f4 0/2,195, 합계 474/8,886. 학습 쪽 입체까지 같은 쌍은 모든 fold에서 0, InChI 확인 모두 일치.
  - (a) · (b)를 fold마다 자기 목록으로 나눈 두 무리로 5 fold 합쳐 보고한다. **기준(FT2-p와 같음)**: 라벨 출발 두 무리의 같은 TS 비율 차 > 0.05 **또는** |Δd| 중앙값 차 > 0.002 Å이면 관문은 '형제 없음' 무리만으로 판정하고, 판정 줄과 이 파일 아래(auto FT2-r2 줄)에 적는다. 기준 안이면 5 fold 시험 전체로 판정한다(명령서대로).

## FT — fold 1–4 순서 (2026-11-05, 사용자 요청 2)

- fold 1–4 학습은 fold 0 FT-2 (a)(라벨 TS 출발)가 통과한 뒤 시작한다. 이에 따라 앞서 낸 미리 돌리기(FT-spec) 8개를 취소했다.
- 명령서 5절은 2차 미세조정 뒤에만 FT-2 (a) 관문을 두지만, 사용자 결정으로 fold 0 1차에도 관문을 둔다.
- **바뀜 (11-05 13:0x, 사용자 "지금미리 시작하고 FT-A 실패하면 그때가서 수정")**: 위 첫 줄의 순서를 사용자가 바꿨다. fold 1–4 AIMNet2 1차 미세조정은 관문을 기다리지 않고 손 job `ft_spec_AIMNet2_1–4`(run `FTpilot_AIMNet2_<k>`)로 미리 돌린다. FT2-p가 AIMNet2를 확정하면 `r1_run`이 FT_k로 재사용한다. fold 0 관문(FT2-p)은 그대로 두며, 실패하면 HALT가 이 모델들과 상태(fold마다 학습 끝 / 중 / 없음)를 적고 다음 단계는 사용자가 그때 정한다(`decide_ft2p` docstring · HALT 문구). 실패하면 최대 약 48 GPU-h를 버린다.
- **바뀜 (11-05 15:5x, 사용자 "바꿔줘")**: 관문 fold를 0에서 4로 옮겼다(아래 절). HALT는 이제 fold 0 모델(FT-p)을 포함한 다른 네 fold 모델의 상태를 적는다.

## FT2-p — 1차 관문 fold 0 → fold 4 (2026-11-05 15:5x, 사용자 "바꿔줘")

- **바뀐 것**: 사용자 10:4x 지시 (2)는 1차 FT-2 (a) 관문을 fold 0 시험(`FTpilot_<c>_0`, f0_test)으로 정했다. 15:5x에 사용자가 관문 fold를 **4**로 바꿨다(`FTpilot_<c>_4`, f4_test 2,195반응, `pipeline.GATE_FOLD`). FT-2 수치를 하나도 보기 전에 바꿨다.
- **이유**: fold 4 AIMNet2 모델(ft_spec_AIMNet2_4)이 먼저 학습을 끝냈고, 그 FT-2 측정이 이미 손 배열 ft2_spec_f4_label · ft2_spec_f4_g2로 돌고 있다. fold 0 모델(ft_p_AIMNet2)은 아직 학습 중이다. FT2-p는 이제 fold 0 학습(FT-p)을 기다리지 않는다.
- **그대로인 것**: 반응별 기준 · 실패 분류 · recalc 5 재측정 · E/Z 기준(0.05 / 0.002 Å)은 바꾸지 않고 fold만 바꾼다. 기반 MLIP는 fold 4 수치로 고른다. FT_0은 여전히 `FTpilot_<base>_0`(fold 0)이고, G-4는 fold 0 학습(FT-p)까지 기다린다.
- **E/Z**: 관문은 fold 4 자기 목록 `f4_test_ez.csv`로 나눈다. 형제 있음이 0/2,195라 무리 하나가 비므로 나누지 않고 fold 4 전체로 판정한다(빈 무리는 나누지 않는다는 규칙; 판정 줄에 적음).
- **fold 0 수치**: FT2-r1(보고용, 1차만)이 FT_0을 f0_test로 재고, 보고용 단계 FT2-f0가 fold 0 (a) · (b)를 `f0_test_ez.csv`로 나눠 `results/FT2_f0_report.json`과 DECISIONS 한 줄에 적는다(사용자 12:0x (2)). 관문으로 쓰지 않는다. FT2-r1은 fold 4를 다시 재지 않는다.
- 위 'FT2-p' 절과 'FT — fold 1–4 순서' 절의 "fold 0 관문"은 이 결정으로 "fold 4 관문"이 된다.

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
