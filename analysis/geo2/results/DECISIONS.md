GEO2.md를 GEO1B.md 4절의 바꿔 읽기로 실행. G1 미사용

# GEO2b 결정 기록 (docs/specs/GEO2.md, GEO1B.md 4-1..4-3)

- **D0D1 (사용자 결정 2026-10-31, "모든 데이터를 사용하게 바꿔서 학습해줘. 비교 모델은 필요없음. 그냥 D0+D1 모두 4개 모델로 학습해서 결과를 내는 것으로 해줘. geo1에서 사용된 모든 데이터를 가지고 최종까지 사용되도록 해줘")**: GEO2.md와 다른 곳은 이 결정이 우선한다. 에너지 모델 4개(AW, GNN, TF, XGB) × arm S·SH × 5 fold를 rung geo에서 D0+D1 짝으로 한 번만 학습한다(분할 T1D0). GEO2.md 10절의 geo 대 g2 판정은 하지 않는다. 결과는 설명용이고, D1과 D0는 따로 보고하며 합치지 않는다. 설계: analysis/geo1b/results/D0D1_design_investigation.json result.design, GEO1b DECISIONS.md D0D1.
- **A-G2c (사용자 결정 2026-10-30, "2번으로 바꿔줘. 1번은 필요없음.")**: G2 입력은 G2c(autodE 3회 중 e_ts_freq_eh 최저)다. rung geo의 rel1·rel2와 대체 TS는 G2c 폴더($G2C/{d1,d0}/g2_final/<rxn>)에서 가져온다.
- 역할별 판단 기록: Phase Z·E = `DECISIONS_ZE.md`. 다른 역할은 각자의 `DECISIONS_*.md`.
