# PREREG GEO2B — 보정 구조(geo, GEO1b) rung, D0 + D1 에너지 학습 사전 등록 (2026-11-02)

이 문서는 GEO2b의 사전 등록이다(`docs/specs/GEO2.md` 8-3, `docs/specs/GEO1B.md` 4-3, 사용자 결정 D0D1). `analysis/geo2/prereg_geo2b.py`(sha256 `501fc7c0703decba06bf99f56405911b160a8d99dc72eedbc132b845bc82959d`, git modified, geo2b-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `44809c85bdba1461bc73da004a80cc06d21ed0a8`, branch `geo2b`, 작성 2026-11-02 23:35:22). 수치는 파일에서 옮겼고(표는 유효숫자 6자리, json에는 원래 값), 규칙은 명령서·결정 기록을 그대로 인용했다(인용 블록 = 원문).
범위: **rung `geo` (LADDER 3단계), D0 + D1 짝, 분할 `T1D0`**, 2 arm × 4 모델 × 5 fold = 40 unit. 학습(`train_atoms.py run --stage 3`, `train_xgb.py run --stage 3`)은 이 md와 `prereg_geo2b.json`이 commit된 뒤에만 돈다(8절 gate). **이 문서는 한 번만 쓴다.**

**판정 없음.** 사용자 결정 D0D1로 GEO2.md 10-1절의 주 판정(geo 대 g2)과 비교 모델은 하지 않는다. 결과는 설명용이며, D1과 D0를 따로 보고하고 합치지 않는다. 절대 기준은 LADDER 목표(5-3절) 하나다.

**지금까지 본 것** (작성 2026-11-02 23:35:22): geo 학습은 한 번도 돌지 않았다(`$LAD/units/L__geo__*` 0개, geo 예측·평가 파일 0개). geo에서 계산한 것은 구조·입력·D0 짝과 8-2 검사(smoke는 fold 0, 2 epoch, 시험 예측 평가 안 함)와 s_y/β 진단뿐이다. GEO1b 구조 결과(7-1절)는 이미 보았다.
- 이 파일은 commit되지 않은 초안을 대체했다: {'analysis/geo2/results/PREREG_GEO2B.md': 'dddad4cb5772a3f2933eaa0fc413e4faeed91071ed30e38bb912e4605938cf08', 'analysis/geo2/results/prereg_geo2b.json': 'c67f6195faa6bc8afa3e69da65ff416ed9bb8f8e4c68c23b713787c9a153d834'}

## 0. 사용자 결정 D0D1과 설계 (그대로)
> - **D0D1 (사용자 2026-10-31 "모든 데이터를 사용하게 바꿔서 학습해줘. 비교 모델은 필요없음. 그냥 D0+D1 모두 4개 모델로 학습해서 결과를 내는 것으로 해줘. geo1에서 사용된 모든 데이터를 가지고 최종까지 사용되도록 해줘")**: 읽기만 하는 조사 workflow(에이전트 5개, 기록 results/D0D1_design_investigation.json)의 추천 설계를 따른다. 비교 판정(GEO2.md 10절 geo 대 g2)은 하지 않는다.
>   - **D0 fold (기본값 적용, 추가 GPU 없음)**: D0 반응(5,260 − 겹침 5)에 fold 0–4를 준다. 명령서 GEO1B는 D0를 시험 fold로 두지 않는다고 했다. 이 규칙을 바꿔 GEO1b final_f{k}가 D0 fold k를 빼고 학습한 뒤 D1 fold k와 D0 fold k를 예측한다. 그래야 D0 보정 구조도 그 반응을 본 적 없는 모델에서 나온다. 같은 fold를 에너지 단계에도 쓴다.
>   - **D0 라벨**: labels_all.json(D1과 같은 정의·수준; c_ghost = bsse_shift_kcal). d1, d2, barrier는 LADDER Q 규칙(분자 key별 가장 낮은 기준 구조, 0.05 kcal/mol 초과 시 교체; refs_q.py)으로 다시 묶어 D1 cr2와 맞춘다. D0 Phase V(DFT 형태 탐색)는 하지 않는다(이탈 기록). 나머지 7개 타깃은 그대로 쓴다. 전하가 있는 260반응은 남긴다.
>     - **D0 분자 key (검토 반영, 추천안)**: key = '<kind>:<SMILES>' + '|alt'(설계) + '|cfg:<추적 이면각 상태>'(alt dipole 기준 구조에만) + '|s3d:<좌표에서 읽은 입체>'(모든 기준 구조). D1 cr2가 상태를 키에 넣는 곳(constrained = TS 상태 기준 구조의 '|fixed')에만 상태를 넣고, alt가 아닌 기준 구조는 D1 unconstrained처럼 rotamer를 넘어 묶는다. s3d는 D0 SMILES에 없는 입체('@' 0개)를 되살려 설계의 'isomeric SMILES'를 지킨다. 설계 키 그대로의 결과와 이전 키(모든 dipole에 cfg)의 결과는 Q_d0_report.json에 보고만 하고, 그 차이를 GEO2b PREREG에 이탈로 적는다. 한계: alt 파일 없이 원래 기준 구조를 쓴 constrained dipole은 D0에서 구별할 수 없다(보고서 audit.nonalt_dipole_replacements_across_cfg로 상한을 센다). 근거·줄 번호: DECISIONS_LAB L-key-*.
>     - **D0 Q-2 결합 그래프 (검토 반영)**: refs_q.py와 같이 한 key 또는 한 SMILES 안에서 결합 그래프가 다르면 STOP한다(격리하지 않음). 상태를 계산하지 못한 기준 구조는 혼자 쓰는 key로 Coley 값을 두되(2 % 이하), 그 결합 그래프가 SMILES 그래프와 같아야 한다(Q-2m, 아니면 STOP). DECISIONS_LAB L-q2, L-err.
>     - **D0·D1 key 표 (검토 반영)**: key는 데이터셋마다 따로 만든다(설계 'Keys are built within D0'). 모든 D0 key에 '|s3d:'·'|err:'가 있어 D1 key와 겹치지 않으므로, GEO2b P의 'Q_keys.parquet과 Q_keys_d0.parquet 합집합'은 이어 붙이기이고 refs='min'은 데이터셋별이다. Q_keys_d0의 constrained = alt('key가 이면각 상태를 가짐')이고 D1의 회전성 판정이 아니다. P는 key 문자열과 alt·fixed_state·cfg_state·stereo3d 열을 읽는다. DECISIONS_LAB L-namespace, L-constrained.
>   - **D0 짝**: dipole 분자별 묶음 안에서 anchor 트리를 만든다(regio·stereo 형제 → 형제 묶음 anchor → 묶음 anchor; 고립 dipole은 같은 dipolarophile 반응에 붙임). 같은 전하끼리만 잇고, 짝 하나는 한 fold 안에 둔다. 약 3,000–3,500짝으로 예상한다. D1 반응과 반응물이 같은 D0 반응(twin)은 그 D1 반응의 fold를 따른다.
>   - **에너지 학습**: F_pairs_cr2.csv(D1, 그대로)에 D0 짝을 더한다. 분할 T1D0: D1 목록은 FU2 T1과 똑같고, 다른 fold의 D0 짝은 학습에, fold k의 D0 짝은 시험에 넣는다. inner_val은 D1만이다. 4 모델 × 2 arm × 5 fold = 40 unit. 모델 코드, grid, 규칙은 바꾸지 않는다.
>   - **결과**: D1과 D0을 따로 보고하고 합치지 않는다. 모델·arm별, REP, REP8, ENS4, core(D0는 묶음) bootstrap CI, LADDER 목표(oi·cpcm para MAE ≤ 0.13).
>   - **상한 조정**: H 4,000코어시간(D0 HF 추가, 예상 약 3,300), M3 160 GPU시간.
>   - **한계(기록)**: 비교 모델이 없어 D0가 D1 정확도에 도움이 되는지는 잴 수 없다. D1 시험 core와 같은 유형의 D0 반응이 학습에 들어간다(데이터셋 간 전이, 라벨 누수 아님). D0 TS는 Gaussian, D1은 ORCA.
>   - **순서**: ① d0_labels.py, d0_pairs.py, GEO1b 패치(D0 fold, G2c 입력 경로, G2c 기준값)를 G2c build 전에 끝내고 검토한다 → ② G2c build 뒤 GEO1b D → U → R → E → ③ GEO2b(D0 포함 입력, T1D0 학습, 평가).

### 0-1. 설계의 학습·결과 부분 (`analysis/geo1b/results/D0D1_design_investigation.json` result.design, 그대로)
> TRAINING
> - New label file F_pairs_d0d1.csv = F_pairs_cr2.csv rows unchanged, plus D0 rows with ref_kind 'd0' and ids D#####.
> - New split files T1D0__f<k>.json:
>   - the D1 lists equal the FU2 T1__f<k> lists exactly;
>   - D0 pairs of the other folds are added to train and to every inner_train;
>   - D0 pairs of fold k are added to test;
>   - inner_val stays D1 only, so REP and REP8 keep their LADDER 13-1 meaning.
> - Model code, grids, loss, beta rule, seeds, epochs and patience are unchanged. L-11 becomes 'D1 rows identical to F_pairs_cr2.csv; extra rows ref_kind d0'.
> - 2 arms x 4 models x 5 folds = 40 units.
>
> RESULTS (reported separately, never pooled)
> - D1: P_geo para P_main pairs plus all positions; each of the 4 models x 2 arms, REP, REP8 and ENS4; core bootstrap 95% CI; the LADDER stage-2 absolute goal (oi and cpcm para MAE <= 0.13 kcal/mol, plus the bend criterion).
> - D0: the fold-held-out D0 test pairs; the same model and selection tables; component bootstrap CI; broken down by edge kind.

### 0-2. 명령서 8-3절 (그대로; 이 문서의 항목 5의 GEO1 숫자는 GEO1B.md 4-3에 따라 GEO1b 숫자로 바꾼다)
> ### 8-3. 사전 등록 `PREREG_GEO2.md` + `prereg_geo2.json` (학습 전에 commit·push)
>
> **담을 것:**
> 1. 0-2절 rung `geo` 정의, 4절 상식 검사와 대체 규칙, Y 설정
> 2. 학습 규칙은 LADDER 4-2·4-3 그대로: arm S·SH, AW·GNN·TF·XGB, grid, β 규칙, seed 5개, 300 epoch, patience 30, T1 fold
> 3. 짝 집합과 sha256(P_geo, P_23, P_023), 라벨 파일 `F_pairs_cr2.csv` sha256, geo 입력표 sha256
> 4. 9–10절의 지표와 판정 규칙 (REP·REP8 정의 포함)
> 5. **이미 본 결과도 그대로 적는다:** GEO1 시험 결과(짝 비율 0.142 → 0.347 등), 작성자의 에너지 환산 예측(같은 형태 짝 oi −30 % 안팎, 전체 −10 % 안팎; 추정치)
> 6. 알려진 한계: 1단계 학습 데이터의 fold 교차 의존(10-3절)

## 1. rung `geo` 정의

### 1-1. 정의 (GEO2.md 0-2절 그대로)
> ### 0-2. rung `geo`의 정의 (g2와 **TS 좌표만** 다르다)
>
> | | `g2` (기존) | **`geo` (새로)** |
> |---|---|---|
> | TS 좌표 | `$G2/g2_final/<rxn>/ts.xyz` | **GEO1 예측 구조**: 그 반응을 시험 fold로 둔 GEO1 final 모델 seed 3개의 좌표 평균. fold −1 반응은 Y 단계의 추가 모델 |
> | 반응물 기준 구조 | G2 `rel1.xyz`, `rel2.xyz` | **같음** (byte copy) |
> | `ts.hess` (feature B6) | D1-G2 | **새 계산:** geo TS에서 `! XTB2 ALPB(water) Freq` (최적화 없음; LADDER G0와 같은 방식) |
> | 전역 182 · FU2 토큰 | D1-G2 Phase F | **새 계산** (같은 코드, 루트 = geo 폴더) |
> | HF/6-31G\* 9값 (H) | D1-G2 Phase H | **새 계산** (같은 runner, 'g2' geometry slot에 geo 루트) |
> | H의 d1·d2·barrier 기준 | rung 안 분자 키별 최저 HF 기준(`refs="min"`) | 같음 |
>
> - 반응물 기준 구조를 G2와 같게 두는 이유: **TS 보정의 효과만** 분리해서 보기 위해서다.
> - 예측 구조가 상식 검사(5-2절)를 통과하지 못하면 G2 TS를 쓴다. 배포 때도 쓸 수 있는 규칙이고, 표시를 남긴다.

### 1-2. GEO1B.md 0-1·0-2·4-3절 (그대로: G1 미사용, G2 전부로 보정 모델 학습, 바꿔 읽기)
> ### 0-1. 사용자 결정 (2026-10-30)
>
> | # | 결정 |
> |---|---|
> | 1 | **G1은 앞으로 없는 것으로 한다.** 학습, 평가, 판정 어디에서도 G1 좌표와 G1에서 나온 값을 읽지 않는다. 여기에는 `C_compare.same_ts_g1`, `d1_pair/g1_final`, `espley_xtb_g1`이 포함된다. GEO2b 에너지 비교에서도 LADDER `g1` rung을 뺀다 |
> | 2 | 구조 보정 모델은 **G2 입력 전부**로 학습한다. 같은/다른 형태로 걸러내지 않는다. "쉬운 데이터만 골랐다"는 비판을 막기 위해서다 |
> | 3 | 기존 GEO2는 **중단**한다. G1을 학습에 쓴 GEO1 구조를 기반으로 하기 때문이다 |
> | 4 | **G2c**(G2 TS 형태 재탐색)는 시범 없이 **D1·D0 전체**를 처음부터 계산한다 |
> | 5 | 최종 판정은 **에너지**(GEO2b, GEO2.md 10절 규칙 그대로)로 한다. GEO1b 구조 지표는 설명용이다 |
> | 6 | 학습 설정은 **GEO1 실제 실행값 그대로** 쓴다: MAX_EP 300, patience 없음, AdamW 2e-4, warmup 5 % 후 cosine (horizon 300 epoch), R-select 11-epoch 창, R-once launcher, R-lpt. epoch와 LR은 바꾸지 않는다(0-4) |
>
> ### 0-2. GEO1 → GEO1b에서 바뀌는 것
>
> | 항목 | GEO1 | **GEO1b** |
> |---|---|---|
> | 학습 예제 | G1 → 라벨 (전부) + G2 → 라벨 (`same_ts_g1 == True`만) | **G2 → 라벨, G2가 있는 학습 반응 전부** (D1 학습 fold + D0, D0 겹침 5개 제외) |
> | 기록의 `same` / `xyz_g1` | C_compare / G1 폴더에서 읽음 | 항상 `None` (읽지 않음) |
> | 주 지표 (내부 검증·시험) | **같은 형태** para 짝(840)의 짝 오차 ≤ 0.005 Å 비율 | **모든** para P_main 짝(시험 1,667짝)의 짝 오차 ≤ 0.005 Å 비율 |
> | 형태별 분석 | same / mixed / different (G1 기반) | **G2 대 라벨의 대칭 고려 무거운 원자 RMSD 구간**으로 나눔: low ≤ 0.25 < mid ≤ 0.75 < high (Å). 설명용으로만 씀 |
> | ablation | no_d0, q1_only, r0 | 없음 |
> | 판정 | 구조 (pass / partial / fail) | 구조는 판정 없음. **GEO2b 에너지로 판정** |
> | 모델, 손실, 가짜 궤적, 설정 격자 c0–c3, seed 3개, 앙상블 | — | **그대로** |
>
> ### 4-3. GEO2.md를 실행할 때의 바꿔 읽기 (나머지 규칙, 검사, 판정은 GEO2.md 그대로)
>
> | GEO2.md에 적힌 것 | GEO2b에서는 |
> |---|---|
> | `analysis/geo` (GEO1 코드, 결과) | `analysis/geo1b` |
> | `$GEO` = `/gpfs/tmp_cpu2/yeseo1ee/geo` | `$GEO1B` = `/gpfs/tmp_cpu2/yeseo1ee/geo1b` |
> | `$GEO2` = `/gpfs/tmp_cpu2/yeseo1ee/geo2` | `$GEO2B` = `/gpfs/tmp_cpu2/yeseo1ee/geo2b`. `feat_geo.py`, `g3c.py` 복사본의 scratch 상수도 모두 이 경로로 한다 |
> | 1절 0-1 시작 조건 (GEO1 산출물) | `analysis/geo1b/results/{E_report.json, E_rxn.csv, R_select_f0..4.json}`, `$GEO1B/preds/final_f{k}_s{0,1,2}.pkl` 15개 |
> | 3절 Y: `analysis/geo/train_m1.py`, `analysis/geo/jobs/gpu.slurm` | `analysis/geo1b/train_m1.py`, `analysis/geo1b/jobs/gpu.slurm`. **`--export=ALL,GEO_MAX_EP=300,GEO_PATIENCE=100000` 필수** (final과 같은 LR 일정. 없으면 스크립트가 멈춘다) |
> | 4절 Z: `build_geo.py`를 `analysis/geo/jobs/gpu.slurm`로 | `analysis/geo1b/jobs/gpu.slurm` (`GEO_SCRATCH`=geo1b)로 `cd analysis/geo2; sbatch … ../geo1b/jobs/gpu.slurm build_geo.py build`. `GEO2_SCRATCH`를 따로 주지 않으면 4-2의 기본값 `geo2b`를 쓴다 |
> | 9절 `E_class.csv` (same / mixed / different) | `E_bin.csv` (low / mid / high / na). 설명용이다 |
> | 9·11절 rung `g1` 비교와 그림의 g1 | 뺀다 (dft, g2, geo만) |
> | 11절 5번 "이미 본 결과" (GEO1 숫자, 에너지 환산 예측) | GEO1b 결과(3-5 표 1, 2)로 바꾼다. GEO1 숫자와 예측치는 쓰지 않는다 |
> | 0-4 기준값 (g2 REP8 P_23 para: oi 1.038, cpcm 0.315, barrier 0.397) | 그대로 |
> | 10절 판정 (REP8, P_23 para, oi·cpcm CI_hi < 0) | **그대로.** 이것이 이번 실험의 최종 판정이다 |
>
> - GEO2b `DECISIONS.md`의 첫 줄: "GEO2.md를 GEO1B.md 4절의 바꿔 읽기로 실행. G1 미사용".
> - 요약은 GEO2.md 11절 형식으로 `analysis/geo2/results/SUMMARY_GEO2B.md`에 쓴다. commit하고 `git push -u origin geo2b`.
>
> ---

- Y 설정 (fold −1 D1 반응의 보정 구조; `analysis/geo1b/results/Y_plan.json`, GEO1b `train_m1.py plan`, 20개보다 긴 목록은 개수와 앞 5개):
```json
{
 "cfg": "c1",
 "epochs": 115,
 "chosen": [
  "c1",
  "c0",
  "c1",
  "c3",
  "c2"
 ],
 "best_ep": [
  191,
  114,
  114,
  118,
  98
 ],
 "rule": "cfg = most frequent R_select choice (ties: lowest name); epochs = floor(median(best_ep + 1))"
}
```

### 1-3. 상식 검사·대체 규칙·Z gate (GEO2.md 4절; 4-1의 좌표 규칙만 아래 E2-rule로 바뀜) 와 Z 결과 (`analysis/geo2/results/Z_report.json`)
> ## 4. Phase Z — geo 구조 폴더 (`analysis/geo2/build_geo.py`)
>
> ### 4-1. 규칙
> - 반응마다 TS 하나: 그 반응을 시험 fold로 둔 GEO1 final 모델 seed 3개의 좌표 평균(GEO1 평가와 같은 앙상블). fold −1은 Y의 3개 seed 평균.
> - `rel1.xyz`, `rel2.xyz` = G2 폴더에서 byte copy.
> - 폴더 형식은 D1-G3 Phase O 출력(`<m>_final/<rxn>/`)과 같다: `ts.xyz`, `rel1.xyz`, `rel2.xyz`, `result.json`, `.done`. 그래서 6절의 G 단계(`feat.py` `g_one`)가 그대로 읽는다.
>
> ### 4-2. 상식 검사 (실패하면 G2 TS로 대체하고 `fallback: true`)
> | 검사 | 기준 |
> |---|---|
> | 가까운 원자 | 모든 원자 쌍 d ≥ 0.60 × (r_i + r_j) (Cordero 공유 결합 반지름) |
> | 결합 그래프 | 형성 결합을 뺀 원자 쌍에서, G2에서 결합(d < 1.10 Σr)이던 것이 1.40 Σr를 넘지 않고, 결합이 아니던 것(d > 1.40 Σr)이 1.10 Σr 아래로 오지 않음 |
> | 형성 결합 길이 | 1.4–4.0 Å |
>
> ### 4-3. gate (STOP)
> - GEO1 `E_rxn.csv`의 반응별 형성 결합 오차(`e_short_model`, `e_long_model`)를 이 앙상블 좌표로 다시 계산한 값과 비교해 최대 차이 ≤ 1e-6 Å. **GEO1 평가에 쓴 바로 그 구조인지** 확인하는 검사다.
>   - `E_rxn.csv` 열 이름이 다르면(GEO1에서 `evaluate.py`를 고친 경우) 같은 뜻의 열을 찾아 맞추고 기록한다.
> - 대체(fallback) 비율 ≤ 2 %.
> - 같은 반응이 두 fold에서 예측되면 STOP.

- 좌표 규칙 E2-rule (GEO1b `results/DECISIONS.md` 'E2 STOP → 추천안', `results/DECISIONS_ZE.md` E2-rule; 라벨을 쓰지 않음, 에너지 결과를 보기 전에 정함): 반응마다 4-2 상식 검사를 통과하는 첫 후보를 쓴다. (1) 3 seed 평균(mean3), (2) 혼자 통과하는 seed가 2개 이상이면 그 평균(mean_pass, 평균도 통과해야 함), (3) 통과하는 seed 중 번호가 가장 낮은 하나(seed<k>), (4) G2c TS(g2c_fallback, `fallback: true`). 대체 비율 ≤ 2 %는 (4)만 센다. E_rxn 대조(1e-6 Å)는 mean3 반응에만 하고, 쓴 ts.xyz = 그 규칙의 좌표(1e-9 Å)는 모든 반응에서 본다. 반응별 규칙은 Z_status.csv `coord_rule`·`coord_seeds`, 개수는 아래 Z_report `by_coord_rule`.

```json
{
 "n_targets": 14521,
 "n_folders": 13696,
 "by_status": {
  "ok": 13696,
  "no_g2": 825
 },
 "by_source": {
  "f4": 3128,
  "f1": 2791,
  "f0": 2757,
  "f2": 2437,
  "f3": 2343,
  "fm1": 240
 },
 "n_fallback": 70,
 "fallback_reasons": {
  "close_contact": 62,
  "bond_graph_changed": 8
 },
 "by_coord_rule": {
  "mean3": 13448,
  "mean_pass": 24,
  "seed0": 7,
  "seed1": 18,
  "seed2": 129,
  "g2c_fallback": 70
 },
 "coord_rule": {
  "order": [
   "mean3",
   "mean_pass",
   "seed0",
   "seed1",
   "seed2",
   "g2c_fallback"
  ],
  "rule": "E2-rule (DECISIONS_ZE): the first candidate that passes sanity: mean3, mean_pass (>= 2 passing seeds, mean passes), the lowest passing seed, g2c_fallback; fallback_share counts g2c_fallback only"
 },
 "e_rxn_check": {
  "D1": {
   "file": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/E_rxn.csv",
   "n": 8419,
   "max_abs_dev_A": 8.881784197001252e-16,
   "tol": 1e-06,
   "scope": "coord_rule mean3 rows (the file = the 3-seed mean)",
   "n_in_file_not_mean3": 220,
   "n_built_not_in_file": 240,
   "n_file": 8639,
   "n_built_f04_not_in_file": 0,
   "n_file_not_built_f04": 0
  },
  "D0": {
   "file": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/E_rxn_d0.csv",
   "n": 4789,
   "max_abs_dev_A": 8.881784197001252e-16,
   "tol": 1e-06,
   "scope": "coord_rule mean3 rows (the file = the 3-seed mean)",
   "n_in_file_not_mean3": 28,
   "n_built_not_in_file": 0,
   "n_file": 4817,
   "n_built_f04_not_in_file": 0,
   "n_file_not_built_f04": 0
  }
 },
 "by_src": {
  "D1": {
   "n_targets": 9266,
   "n_folders": 8879,
   "by_status": {
    "ok": 8879,
    "no_g2": 387
   },
   "by_source": {
    "f4": 2152,
    "f1": 1819,
    "f0": 1795,
    "f2": 1473,
    "f3": 1400,
    "fm1": 240
   },
   "n_fallback": 55,
   "fallback_share": 0.006194391260277058,
   "fallback_max": 0.02,
   "fallback_reasons": {
    "close_contact": 54,
    "bond_graph_changed": 1
   },
   "by_coord_rule": {
    "mean3": 8659,
    "mean_pass": 21,
    "seed0": 6,
    "seed1": 17,
    "seed2": 121,
    "g2c_fallback": 55
   },
   "n_rescued": 165,
   "mean3_fail_reasons": {
    "close_contact": 198,
    "bond_graph_changed": 21,
    "forming_distance_range": 1
   },
   "written_check": {
    "n_by_rule": {
     "mean3": 8659,
     "seed2": 121,
     "g2c_fallback": 55,
     "seed1": 17,
     "mean_pass": 21,
     "seed0": 6
    },
    "max_abs_dev_A": 5.000000413701855e-11,
    "tol": 1e-09
   },
   "g2c_chosen": {
    "r2": 2986,
    "orig": 2923,
    "r1": 2970
   },
   "e_rxn_check": {
    "file": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/E_rxn.csv",
    "n": 8419,
    "max_abs_dev_A": 8.881784197001252e-16,
    "tol": 1e-06,
    "scope": "coord_rule mean3 rows (the file = the 3-seed mean)",
    "n_in_file_not_mean3": 220,
    "n_built_not_in_file": 240,
    "n_file": 8639,
    "n_built_f04_not_in_file": 0,
    "n_file_not_built_f04": 0
   }
  },
  "D0": {
   "n_targets": 5255,
   "n_folders": 4817,
   "by_status": {
    "ok": 4817,
    "no_g2": 438
   },
   "by_source": {
    "f4": 976,
    "f0": 962,
    "f3": 943,
    "f1": 972,
    "f2": 964
   },
   "n_fallback": 15,
   "fallback_share": 0.0031139713514635667,
   "fallback_max": 0.02,
   "fallback_reasons": {
    "bond_graph_changed": 7,
    "close_contact": 8
   },
   "by_coord_rule": {
    "mean3": 4789,
    "mean_pass": 3,
    "seed0": 1,
    "seed1": 1,
    "seed2": 8,
    "g2c_fallback": 15
   },
   "n_rescued": 13,
   "mean3_fail_reasons": {
    "bond_graph_changed": 14,
    "close_contact": 14
   },
   "written_check": {
    "n_by_rule": {
     "mean3": 4789,
     "g2c_fallback": 15,
     "seed2": 8,
     "seed0": 1,
     "seed1": 1,
     "mean_pass": 3
    },
    "max_abs_dev_A": 5.000000413701855e-11,
    "tol": 1e-09
   },
   "g2c_chosen": {
    "r1": 1562,
    "r2": 1603,
    "orig": 1652
   },
   "e_rxn_check": {
    "file": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/E_rxn_d0.csv",
    "n": 4789,
    "max_abs_dev_A": 8.881784197001252e-16,
    "tol": 1e-06,
    "scope": "coord_rule mean3 rows (the file = the 3-seed mean)",
    "n_in_file_not_mean3": 28,
    "n_built_not_in_file": 0,
    "n_file": 4817,
    "n_built_f04_not_in_file": 0,
    "n_file_not_built_f04": 0
   }
  }
 },
 "n_d0_overlap_excluded": 5,
 "n_stale_folders": 0,
 "layout": {
  "root": "/gpfs/tmp_cpu2/yeseo1ee/geo2b/geo_final",
  "folder": "<rxn_id> (D1 >= 1000001, D0 0-5973)",
  "files": [
   "ts.xyz",
   "rel1.xyz",
   "rel2.xyz",
   "result.json",
   ".done"
  ],
  "z_status_fields": [
   "rxn_id",
   "job_id",
   "src",
   "status",
   "source",
   "fallback",
   "reason",
   "g2c",
   "coord_rule",
   "coord_seeds"
  ]
 },
 "inputs": {
  "d1_labels": "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json",
  "d0_labels": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/labels_all.json",
  "d0_folds": "/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/d0_folds.csv",
  "d0_folds_sha256": "873724245e7b9897d926e9cc1a042091166a3523f009d2102794bb6d83534c4e",
  "g2c_select": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c/analysis/g2c/results/C_select.csv",
  "g2c_select_sha256": "51d8acfcb8ad07f89f254a1637720448750784e4b879427a7cc518ead7e87413",
  "g2_final_d1": "/gpfs/tmp_cpu2/yeseo1ee/g2c/d1/g2_final",
  "g2_final_d0": "/gpfs/tmp_cpu2/yeseo1ee/g2c/d0/g2_final",
  "preds": "/gpfs/tmp_cpu2/yeseo1ee/geo1b/preds",
  "g2c_vs_geo1b": {
   "d_report": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/D_dataset_report.json",
   "d_report_sha256": "3f35428bd1a7536a70ba8f1f31d781c4e30f3c8b00ed036381ac0b53c2098003",
   "c_select_sha256": "51d8acfcb8ad07f89f254a1637720448750784e4b879427a7cc518ead7e87413",
   "equal": true
  }
 },
 "decisions": "results/DECISIONS.md (A-G2c, D0D1), results/DECISIONS_ZE.md",
 "stop": false,
 "stop_reasons": [],
 "running": false,
 "complete": true,
 "wall_s": 78.57763075828552
}
```

## 2. D0 짝과 fold 규칙

- D0 짝: GEO1b `d0_pairs.py`(code geo1b-d0pairs-2, g2_source `g2c`, preliminary False) — 규칙은 그 docstring과 `analysis/geo1b/results/DECISIONS_PAIR.md`(7-2절 sha256). D0 짝 3804개, 에너지 적격(ok) 3565개.
- 라벨·분할 규칙 (`analysis/geo2/d0d1_splits.py` RULES, 그대로):
  - `labels`: F_pairs_cr2.csv bytes (D1 rows byte-identical) + every D0_pairs.csv row in the F_pairs.csv columns (text copy; edge_kind / ok not copied)
  - `splits`: T1D0__f<k> = FU2 T1__f<k> lists exactly + D0 pairs (ok True, sorted pair_id): train / inner_train += D0 fold != k, test += D0 fold k, inner_val += nothing; D0 never in the GroupKFold call
  - `d0_fold`: pair fold = d0_folds.csv fold of a_rxn = of b_rxn (0..4)
  - `d0_inputs`: D0_pairs_report.json stop false, g2_source g2c, preliminary false, status ok, outputs sha256 = files; Q_d0_report.json all_pass, stop false, sha256 = D0_pairs_report inputs record
  - `twins`: no D0 training reaction of fold k with a D1 twin bound to fold k (forced / iso_resolved)
- fold별 D0 짝 (ok, 분할 파일 기준; geo 입력이 없는 짝은 학습 때 두 끝 사용 가능 필터로 빠진다):
| fold | D0 학습 짝 | D0 시험 짝 | D0 짝 전체(fold) | 시험 짝 edge 종류 |
|---|---|---|---|---|
| 0 | 2846 | 719 | 761 | {'dipole': 137, 'leaf': 147, 'sibling': 435} |
| 1 | 2847 | 718 | 762 | {'dipole': 157, 'leaf': 71, 'sibling': 490} |
| 2 | 2853 | 712 | 760 | {'dipole': 144, 'leaf': 70, 'sibling': 498} |
| 3 | 2854 | 711 | 762 | {'dipole': 160, 'leaf': 71, 'sibling': 480} |
| 4 | 2860 | 705 | 759 | {'dipole': 151, 'leaf': 68, 'sibling': 486} |

- twins: {'n_twins': 0, 'by_fold': {'0': {'n_twin_rxn_in_train': 0, 'reported_only': []}, '1': {'n_twin_rxn_in_train': 0, 'reported_only': []}, '2': {'n_twin_rxn_in_train': 0, 'reported_only': []}, '3': {'n_twin_rxn_in_train': 0, 'reported_only': []}, '4': {'n_twin_rxn_in_train': 0, 'reported_only': []}}, 'rule': 'no D0 training reaction of fold k with a D1 twin bound to fold k (forced / iso_resolved)'}
- 분할 T1D0: D1 목록은 FU2 T1__f<k>와 똑같고(같은 id, 같은 순서), D0 짝은 그 뒤에 붙는다. inner_val은 D1만이다(REP·REP8 검증 MAE의 LADDER 13-1 의미 유지). XGB 안쪽 GroupKFold(4)는 D1 부분에서만 돌고, D0 학습 짝은 모든 안쪽 학습 쪽에 들어간다(arm P 선례).

## 3. 학습 규칙 (LADDER 4-2·4-3 그대로; 바뀐 것은 짝·분할뿐)

### 3-1. 입력 두 가지 (D1LADDER.md 4-2절 그대로)
> ### 4-2. 입력 두 가지 (arm)
>
> | arm | 원자 모델 입력 | XGB 입력 | 학습 대상 | 예측 |
> |---|---|---|---|---|
> | **S** (구조만) | 원자 토큰 + 전역 182 (FU2 변형 "A+G") | 182 (rev 9 arm C) | ΔΔ | 모델 출력 |
> | **SH** (구조 + HF) | 원자 토큰 + 전역 (182 + H 9) | 182 + H 9 | 잔차 r = ΔΔ − β·ΔH | β·ΔH + 모델 출력 |
>
> **SH의 기준선 (검량선):**
> - 타깃 t마다 ΔH_t = H_t(a) − H_t(b).
> - β_t = Σ ΔH_t·Δy_t ÷ Σ ΔH_t², **절편 없음**. 절편이 있으면 반대칭이 깨진다.
> - β는 **그 학습에 쓰는 짝으로만** 맞춘다. 튜닝 때는 안쪽 학습 쪽, 최종 모델은 바깥 학습 쪽 전체(T_main: 기준 짝 + anchor 짝)다.
> - c_ghost는 HF 대응 값이 없으므로 β = 0이다.
>
> **SH의 학습 규칙 (FU2 규칙을 이렇게만 바꾼다):**
> - 출력 = 잔차. 표준화는 학습 쪽 잔차 표준편차로 나누기만 한다(평균은 빼지 않는다).
> - 손실 = 잔차 MAE(표준화 단위) + λ·|ŷ_barrier − Σ 9채널 ŷ| / s_barrier, λ = 0.1. 벌점은 **기준선을 더한 최종 예측 ŷ**로 계산하고, s_barrier는 원래 y의 학습 쪽 표준편차다.
> - 튜닝·early stopping 지표 = 최종 예측 ŷ의 검증 MAE를 **원래 y의 학습 쪽 표준편차**로 나눈 값의 9 주 타깃 평균. FU2와 같은 단위다.
> - XGB: 타깃마다 r_t를 학습한다. 안쪽 CV 점수 = |β·ΔH + ĝ − y|의 평균(kcal)이다.

### 3-2. 모델 네 개 (D1LADDER.md 4-3절 그대로)
> ### 4-3. 모델 네 개
>
> | 모델 | 구조·grid·학습 규칙 |
> |---|---|
> | AW | FU2 M-2 그대로, 변형 A+G |
> | GNN | FU2 M-2 그대로, 변형 A+G |
> | TF | FU2 M-2 그대로, 변형 A+G |
> | XGB | rev 9 arm C XGB 그대로 (`train_ml_single.GRIDS["XGB"]`, ½[h(dx) − h(−dx)], 안쪽 `GroupKFold(4)`) |
>
> - 짝·분할: `d1_train.masks(u)` 또는 `fu2.train.Data.split`와 같은 T1 fold 0–4. 학습 = 학습 쪽 T_main, 시험 = 시험 쪽 P_sens ∪ A_main, 주 지표 = P_main.
> - 원자 모델은 seed 5개(20261019 + 0..4) 평균, 최대 300 epoch, patience 30. FU2와 같다.
> - 입력 금지(FU2와 같음): σ, 계열·core 번호, X 이름.

### 3-3. grid와 상수 (코드에서 그대로; 파일 sha256은 8-1절 gate 표)
```python
# analysis/d1_pair/fu2/models.py (AW / GNN / TF, 변형 A+G)
GRID = {"AW": [dict(n_layers=L, width=W) for L in (1, 2) for W in (64, 128)],
        "GNN": [dict(n_layers=L, width=W) for L in (4, 6) for W in (64, 128)],
        "TF": [dict(n_layers=L, width=W) for L in (2, 3) for W in (64, 128)]}
# analysis/espley_xtb_repro/train_ml_single.py (XGB)
GRIDS["XGB"]: (XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist"),
            {"regressor__xgbregressor__n_estimators": [200, 500],
             "regressor__xgbregressor__max_depth": [3, 5, 7],
             "regressor__xgbregressor__learning_rate": [0.03, 0.1]})
# analysis/d1_pair/fu2/fu2_common.py
SEED = C.SEED
LAMBDA = 0.1
LR, WD, DROPOUT, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 0.1, 64, 300, 30
FINAL_SEEDS = [SEED + i for i in range(5)]
ANTISYM_TOL = 1e-4
ANTISYM_N = 200
# analysis/d1_pair/d1pair_common.py
SEED = 20261019
N_INNER = 4
ANTISYM_TOL = 1e-6
ANTISYM_N = 200
```
- 모델 구조·grid·손실·β 규칙·seed·epoch·patience는 LADDER와 같다(D0 가중치 없음). gate M-1′는 모든 학습에서 시험 짝 200개와 D0 학습 짝 200개 모두에 대해 확인한다(GB-5 / XG-B4).

### 3-4. 학습 단위와 실행 (GEO2.md 8-1·8-4절 그대로; 상한은 D0D1 결정으로 200 GPU시간(11-02 M3-cap))
> ### 8-1. LADDER 코드에 rung `geo` 추가 (기본 동작은 그대로)
>
> | 파일 | 추가 |
> |---|---|
> | `train_atoms.py` | `STAGE_RUNGS[3] = ("geo",)`, `RUNG_ORDER`에 `"geo"`, `RUNGS["geo"] = dict(fu2_res=<repo>/analysis/geo2/results/fu2_geo, tokens=$GEO2/fu2_geo/tokens, tag="geo")`, `inputs_path("geo") = $GEO2/inputs/geo.parquet` (다른 rung은 그대로 `$LAD/inputs/<rung>.parquet`) |
> | `train_xgb.py` | `STAGE_RUNGS[3]`, `RUNGS["geo"] = dict(fu2_res=…/fu2_geo, x182=$GEO2/X182_d1_geo.parquet, tag="geo")`, `inputs_path` 같은 규칙 |
> | `inputs.py` | `RUNGS["geo"]` = `feat_geo.rung_cfg("geo")`와 같은 값 (실행 중 등록이 아니라 고정) |
> | 사전 등록 gate | stage 3은 ADD3 대신 **`analysis/geo2/results/PREREG_GEO2.md` + `prereg_geo2.json`**을 확인한다(`gate_add3`와 같은 방식: commit됨, sha256 일치, `inputs_sha256`에 geo 입력표·라벨 파일이 있음) |
> | stage 3 검사 gate | `results/C_tests.json` 대신 `analysis/geo2/results/C_tests_geo.json` all_pass를 확인한다 |
> | collect | `collect --stage 3`은 **geo 행만** `results/L_predictions_geo.parquet`, `L_report_geo.json`, `L_xgb_report_geo.json`에 쓴다. 기존 `L_predictions.parquet`, `L_report.json`, `L_xgb_report.json`은 건드리지 않는다 |
> | unit | `$LAD/units/L__geo__<arm>__<model>__f<k>.*` 새 파일만 만든다. 기존 unit 파일은 열기만 한다 |
>
> - 쓰기 허용 경로(`lad_common.assert_path`)에 `$GEO2`가 필요하면 추가한다. 기본값은 그대로 둔다.
> - 모델 구조·grid·학습 규칙(`fu2/models.py`, `fu2/train.py`, `train_ml_single.GRIDS["XGB"]`)은 **한 글자도 바꾸지 않는다.** rung 목록과 경로만 추가한다.
> - 절대 경로가 다른 worktree에서 돌기 때문에 지문(fingerprint) 비교가 어긋나면, D1-G3 결정 F-13처럼 **repo 상대 경로로 비교**하고 기록한다.
>
> ### 8-4. 학습 실행
> - 단위: **2 arm(S, SH) × 4 모델(AW, GNN, TF, XGB) × 5 fold = 40 unit.** AW·GNN·TF는 GPU(30 unit; unit마다 튜닝 4칸 + seed 5개, LADDER와 같음), XGB는 CPU(10 unit). **4개 모두 학습해야 E로 간다.** 하나라도 빠지면 REP를 고를 수 없으므로 STOP.
> - 예상: LADDER rung 하나가 14(g2)–49(dft) GPU시간이었다. smoke로 다시 예상해서 **80 GPU시간을 넘으면 STOP**.
> - LADDER와 같은 명령, `--stage 3`:

- unit 40개: `L__geo__S__AW__f0`, `L__geo__S__AW__f1` … `L__geo__SH__XGB__f4`.

## 4. 라벨·분할·입력 sha256
- 라벨 `/gpfs/tmp_cpu2/yeseo1ee/geo2b/F_pairs_d0d1.csv` sha256 `9edb8725c402acb571cd8ec7dabee0506391c27e65fc0acacf82fd9a8402322c` (D1 행 = `F_pairs_cr2.csv` 바이트 그대로, D0 행 = `D0_pairs.csv`; L-11′)
- 분할 색인 `analysis/geo2/results/S_splits_d0d1.json` sha256 `b9ee62bafa16c5d017fdee1b545af9bed37d5f7e5a3bddcbc461a6cb32ac2df1`
| 분할 파일 | sha256 | train | test | inner_train | inner_val | D0 train | D0 test |
|---|---|---|---|---|---|---|---|
| T1D0__f0.json | `d0902e13cf6d75b4fdeaf9126a197dbfdd2f86d1a594f8e01029c257c59381f3` | 9118 | 2433 | 7547 | 1571 | 2846 | 719 |
| T1D0__f1.json | `e6fd840d8d89ecbb5053bec950d8c12cc22014c4cb242c0442b477769fcbec05` | 9195 | 2388 | 7581 | 1614 | 2847 | 718 |
| T1D0__f2.json | `070c553da0daed42b0927237ccd125fc0ecc552beeab5ef1d1847cdc40a97e57` | 9378 | 2158 | 7756 | 1622 | 2853 | 712 |
| T1D0__f3.json | `ce986bcda835a925df519a768aad92fbb909c8cde837f8ed22a7cf067b0e8b32` | 9468 | 2068 | 7803 | 1665 | 2854 | 711 |
| T1D0__f4.json | `0e470830c101cb001316ab50c7fee551c92ea527a5dbda92f0e9b62c1e5d95c0` | 9144 | 2393 | 7569 | 1575 | 2860 | 705 |

- D0 입력: D0_pairs.csv `d647bd320d493f20cc4108c1261907fcfc1ed769da51e51ae3417004c8e6ecd7`, d0_folds.csv `873724245e7b9897d926e9cc1a042091166a3523f009d2102794bb6d83534c4e`, d0_twins.csv `8511ce4c98f2679284660da0a89d13c33eb440cae0a0ea2598c4038a2760e5ee`, D0_pairs_report.json `77a5a0ab40ad0e1cc6970045a3fa879f48db487c49a73f07d47b850a38f95bbd`, Q_d0_report.json `d6a8ec0eed15dc7eba402fc5742de2dcfbb94b3ff9c4fed8da300d81bb6bed83`, labels_d0_q.json `05df8b43299ecbf970aa19a6e4cb81a425f6f0d20e25b0a26445b672c1d2db05`
- geo 입력표 `/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/geo.parquet` sha256 `f528070ca0805afb2644ea37d1ef6387c47491c8da1945f0e52e27373116a5b5` (= `analysis/geo2/results/P_build_geo.json` outputs)

## 5. 지표와 선택 규칙

### 5-1. 선택 정의 (GEO2.md 9-1절 그대로; 검증 MAE = D1 inner_val 짝)
> ### 9-1. 선택(selection) 정의 — 모두 검증 MAE로만 고른다
> | 이름 | 정의 |
> |---|---|
> | REP·SH, REP·S | LADDER 13-1: rung × arm × 타깃마다 4 모델 중 5 fold 평균 검증 kcal MAE가 가장 낮은 것. 동점이면 AW → GNN → TF → XGB |
> | **REP8** | rung × 타깃마다 2 arm × 4 모델(8개) 중 검증 MAE가 가장 낮은 것. 동점이면 SH 먼저, 그다음 모델 순서. **주 판정에 쓴다** |
> | ENS4·SH, ENS4·S | 4 모델 시험 예측의 같은 비중 평균 |
>
> - 검증 MAE는 unit json의 `val_kcal_mae`(5 fold 평균)다. 시험 점수로 고르지 않는다.

### 5-2. 대표 고르기 (D1LADDER.md 13-1절 그대로)
> ### 13-1. 대표 (REP)
> - rung × arm × 타깃마다 4 모델 중 **검증 kcal MAE**(바깥 fold 5개 평균)가 가장 낮은 모델을 고른다.
> - 동점이면 AW → GNN → TF → XGB 순서의 앞 것이다. 시험 점수로 고르지 않는다.
> - ENS4 = 4 모델 시험 예측의 같은 비중 평균이다(조정값 없음). 보조로 쓴다.

### 5-3. 유일한 절대 기준: LADDER 목표 (D1LADDER.md 13-2절 그대로; D1 para 짝, REP 기준)
> ### 13-2. 단계 2 판정 (주 판정, rung × arm 6개)
> - REP 기준. **oi와 cpcm 모두** 다음을 만족하면 통과다.
>   - para 짝 MAE ≤ 0.13 kcal/mol
>   - 꺾임 민감도 ≥ 0.7
>   - 위양성률 ≤ 0.1
> - 집합은 그 rung의 자기 집합이고, 점 추정으로 판정한다. CI 상한까지 ≤ 0.13이면 "확실히 통과"로 따로 표시한다.
> - 근거(rev 9 PREREG 5-3, 그대로 인용): 숨은 꺾임 크기 중앙값이 궤도 0.47, 용매 0.45 kcal/mol이다. 모의실험(σp 18점)에서 잡음 SD ≤ 0.35 × 꺾임 크기 → SD ≤ 약 0.16 → MAE ≈ 0.8 × SD ≤ 약 0.13.
> - 참고: oi·cpcm 정답은 Q·V에서 바뀌지 않는다(Q-b).
> - 보조 판정: ENS4에 같은 기준을 적용한다. 개별 모델의 통과는 **설명용**으로만 적는다(가장 좋은 것을 고르면 사후 선택이 된다).
> - 단계 0·1(rev 9 기준)과 9개 타깃 전부의 수치는 설명용으로 함께 적는다.

- GEO2.md 10-1절(geo 대 g2 주 판정)은 하지 않는다(D0D1). D0 결과는 같은 모델·선택 표를 D0 시험 짝(fold 밖 짝)에서 따로 내고 묶음(component) 단위 bootstrap CI를 쓴다(설계 RESULTS, 0-1절). D1과 D0는 합치지 않는다.

## 6. 알려진 한계
> ### 10-3. 꼭 적을 한계
> - **1단계 학습 데이터의 fold 교차 의존:** fold k를 시험할 때, 학습 짝의 구조를 만든 GEO1 모델(다른 fold의 final 모델)은 fold k 반응의 DFT 구조를 학습에 썼다. 시험 짝의 구조와 에너지 라벨은 쓰지 않았으므로 직접 누수는 아니다. 완전히 막으려면 1단계를 중첩 교차검증으로 약 20회 더 학습해야 하며, 이번에는 하지 않았다.
> - TS 형태 불일치(G2가 다른 형태를 찾은 반응)는 이 파이프라인이 고치지 않는다.
> - 반응물 기준 구조는 G2(xTB) 그대로다.
> - 예측 구조는 xTB 정류점이 아니다(n_imag 분포를 보고).

- 위 GEO2.md 10-3의 1단계 fold 교차 의존은 D0에도 적용된다(GEO1b final_f<k>가 다른 fold의 D0 라벨 TS를 학습).
- D1 시험 core와 같은 종류(dipole × 골격)의 D0 반응이 학습에 들어간다(데이터셋 간 전이; 라벨 누수 아님). D1 T1의 '보지 못한 core' 주장은 약해진다. D-1 겹침 5개는 모든 집합에서 빠지고, 반응물이 같은 D0 twin은 그 D1 twin의 fold를 따른다.
- D0 Phase V(DFT 형태 탐색)는 없다(D0 d1·d2·barrier는 Q 규칙만, GEO1b DECISIONS D0D1).
- D0 Q-2 키의 `|s3d:`(3D 입체 표지)는 Coley 라벨 기준 구조에서 나온다(GEO1b d0_labels.py). rung geo의 refs 'min'은 D0 G2c 기준 구조를 이 키로 묶는다(alt / cfg 행만 자기 기준). D0D1 설계와 같고, 탐색에서 D0 G2 기준 조각 21,092개의 formula·WL 불일치는 0개였다. G2c 기준 구조 자체의 입체가 키와 다른 경우는 P 보고에 개수만 적는다(판정 아님; `P_geo_report.json` d0_reference_s3d: 검사 9634, 다름 551, 그중 묶이는 행 83, 계산 불가 0; DECISIONS_feat 연결 S-8).
- D0 라벨 TS는 Gaussian 16, D1은 ORCA RIJCOSX로 최적화했다(A5-eng: pauli·oi RMS 1.6 / 0.8 kcal/mol, n = 24).
- XGB의 제곱 손실과 SH β(절편 없는 최소제곱)는 더 큰 D0 ΔΔ에 끌릴 수 있다(D0 짝 |dd|가 D1의 2.5–6배). 7-3절의 s_y·β 진단은 설명용이며 학습 규칙을 바꾸지 않는다.
- 비교 모델이 없어 D0가 D1 정확도를 돕는지 해치는지는 잴 수 없다. 기존 LADDER g2/dft 수치는 다른 구조에서 나온 것이라 통제된 비교가 아니다.
- F-1 승인(11-02, `results/DECISIONS_feat_runtime.md` 'F-1 accept'): rung geo feature 실패 D1 113 / 8,879 = 1.27 % (> 1 %, 선례 1.14–1.42 % 안; engine_gate1·parse·spe_fail), D0 7 / 4,817. 사용자 정책(spec STOP에서 추천안)으로 `p --accept=F-1`을 썼고, 이 120반응은 feature가 없어 그 짝이 rung geo에서 빠진다(편차).
- M3 GPU 상한(11-02, `results/DECISIONS_lad.md` 'M3-cap'): smoke 예상 162.2 GPU-h가 설계 상한 160을 넘어 200으로 올렸다. 모델·grid·epoch·seed·짝은 그대로다.

### 6-1. 설계가 받아들인 위험 (그대로)
> RISKS ACCEPTED, RECORDED AS LIMITATIONS
> - Cross-fold dependence of the structure models (GEO2.md 10-3) now covers D0 as well.
> - D0 reactions of the same dipole/dipolarophile type as D1 test cores sit in training. This is cross-dataset transfer, not label leakage, but it weakens the T1 'unseen core' claim.

## 7. 이미 본 결과 (설명용)

### 7-1. GEO1b 구조 결과 (`analysis/geo1b/results/E_report.json`, `E_rmsd.json`; 판정 없음)
```json
{
 "decision": "none here: structure is descriptive; GEO2b decides on energy",
 "primary": {
  "g2_raw": {
   "primary_share": 0.14714368147720716,
   "n_primary": 1733,
   "all_pairs_share": 0.08183037710538082,
   "n_pairs": 6709,
   "pair_pe_median_para": 0.021811223276766434,
   "rxn_form_abs_median": 0.07847607090663211,
   "rmsd_heavy_median": 0.8263619775024261
  },
  "model": {
   "primary_share": 0.23889209463358338,
   "n_primary": 1733,
   "all_pairs_share": 0.15650618572067374,
   "n_pairs": 6709,
   "pair_pe_median_para": 0.01227643720910665,
   "rxn_form_abs_median": 0.025176282054103494,
   "rmsd_heavy_median": 0.8205847640099772
  },
  "seed0": {
   "primary_share": 0.18984420080784767,
   "n_primary": 1733,
   "all_pairs_share": 0.1238634669846475,
   "n_pairs": 6709,
   "pair_pe_median_para": 0.013130577959583611,
   "rxn_form_abs_median": 0.025850238651063506,
   "rmsd_heavy_median": 0.8208008459814153
  },
  "seed1": {
   "primary_share": 0.20484708597807272,
   "n_primary": 1733,
   "all_pairs_share": 0.12609926963779997,
   "n_pairs": 6709,
   "pair_pe_median_para": 0.0131421339279294,
   "rxn_form_abs_median": 0.025434996098198104,
   "rmsd_heavy_median": 0.8196632474480202
  },
  "seed2": {
   "primary_share": 0.19676860934795154,
   "n_primary": 1733,
   "all_pairs_share": 0.1275898047399016,
   "n_pairs": 6709,
   "pair_pe_median_para": 0.013724605507006604,
   "rxn_form_abs_median": 0.026975463252999266,
   "rmsd_heavy_median": 0.820487078087864
  }
 },
 "ci": {
  "model_vs_g2_raw": {
   "ci_a": [
    0.18697819329671958,
    0.3009914012320522
   ],
   "ci_b": [
    0.1156053871868405,
    0.18467796300064965
   ],
   "ci_diff": [
    0.05266256912401817,
    0.13425720306183883
   ]
  },
  "seed0_vs_g2_raw": {
   "ci_a": [
    0.14555244526086475,
    0.24028893887682273
   ],
   "ci_b": [
    0.11479283409746986,
    0.18334711990504668
   ],
   "ci_diff": [
    0.009757119632397868,
    0.07803143035798712
   ]
  },
  "seed1_vs_g2_raw": {
   "ci_a": [
    0.1597842075180638,
    0.26129150446860455
   ],
   "ci_b": [
    0.11420092190809276,
    0.18598412502532333
   ],
   "ci_diff": [
    0.019855435540596003,
    0.09688443961402711
   ]
  },
  "seed2_vs_g2_raw": {
   "ci_a": [
    0.15332628997037281,
    0.2540747648146321
   ],
   "ci_b": [
    0.11375029212432812,
    0.18737039968341904
   ],
   "ci_diff": [
    0.014643822907687207,
    0.08846332241267055
   ]
  }
 },
 "n_test": 8639,
 "g2_raw_expected": {
  "n": 1733,
  "share": 0.14714368147720716,
  "median": 0.021811223276766434,
  "n_test": 8639,
  "records_sha256": "f49343e0a441af38ea33917264f5231d23979bb2ee53cb8c60391c4f1184e859"
 },
 "g2_raw_baseline": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo1b/results/G2RAW_baseline.json",
 "g2_raw_matches_expected": true,
 "thr_A": 0.005,
 "n_boot": 2000,
 "g1_used": false,
 "g2_input": "G2c (decision A-G2c)",
 "d0": {
  "note": "descriptive only (D0 has no reference pairs); E_rxn_d0.csv",
  "n_test": 4817,
  "model": {
   "n": 4817,
   "rxn_form_abs_median": 0.04434024573118167,
   "rmsd_heavy_median": 1.1712021415005194
  },
  "g2_raw": {
   "n": 4817,
   "rxn_form_abs_median": 0.09688815094879555,
   "rmsd_heavy_median": 1.1720006052153598
  },
  "by_fold": {
   "0": {
    "model": {
     "n": 962,
     "rxn_form_abs_median": 0.042884772812800565,
     "rmsd_heavy_median": 1.151058799178879
    },
    "g2_raw": {
     "n": 962,
     "rxn_form_abs_median": 0.09501850998264039,
     "rmsd_heavy_median": 1.151627723884109
    }
   },
   "1": {
    "model": {
     "n": 972,
     "rxn_form_abs_median": 0.043770377716521924,
     "rmsd_heavy_median": 1.1746160108414725
    },
    "g2_raw": {
     "n": 972,
     "rxn_form_abs_median": 0.09617578566843599,
     "rmsd_heavy_median": 1.1769518587205816
    }
   },
   "2": {
    "model": {
     "n": 964,
     "rxn_form_abs_median": 0.04754327073297415,
     "rmsd_heavy_median": 1.1794200581108967
    },
    "g2_raw": {
     "n": 964,
     "rxn_form_abs_median": 0.10320531865073845,
     "rmsd_heavy_median": 1.178015838355559
    }
   },
   "3": {
    "model": {
     "n": 943,
     "rxn_form_abs_median": 0.04283402533476166,
     "rmsd_heavy_median": 1.1791587400785315
    },
    "g2_raw": {
     "n": 943,
     "rxn_form_abs_median": 0.09359215153468314,
     "rmsd_heavy_median": 1.1857079689509886
    }
   },
   "4": {
    "model": {
     "n": 976,
     "rxn_form_abs_median": 0.04451709967508144,
     "rmsd_heavy_median": 1.1570696085122498
    },
    "g2_raw": {
     "n": 976,
     "rxn_form_abs_median": 0.09627920774299215,
     "rmsd_heavy_median": 1.1651091397884445
    }
   }
  },
  "n_twin_m1": 0,
  "twin_kind_counts": {
   "": 4817
  },
  "without_twin_m1": {
   "model": {
    "n": 4817,
    "rxn_form_abs_median": 0.04434024573118167,
    "rmsd_heavy_median": 1.1712021415005194
   },
   "g2_raw": {
    "n": 4817,
    "rxn_form_abs_median": 0.09688815094879555,
    "rmsd_heavy_median": 1.1720006052153598
   }
  },
  "twin_m1_note": "twin_m1 = a stereo-free D1 twin of T1 fold -1, in every final_f{k} training set while this D0 reaction is tested (DECISIONS_PAIR P-twin-m1, accepted limitation)"
 }
}
```
```json
{
 "all": {
  "n": 8639,
  "median_map_g2": 0.28085867143908977,
  "median_map_model": 0.2840693003555339,
  "share_le_025": {
   "g2": 0.48003802281368824,
   "model": 0.48045279495857157
  },
  "share_le_050": {
   "g2": 0.620365969581749,
   "model": 0.6210759715252655
  },
  "share_moved_gt_030": 0.0010724499523355577,
  "n_nan": {
   "g2": 223,
   "model": 70
  }
 },
 "by_bin": {
  "high": {
   "n": 2305,
   "median_map_g2": 1.120798138399787,
   "median_map_model": 1.1230302449276435,
   "share_le_025": {
    "g2": 0.0,
    "model": 0.0
   },
   "share_le_050": {
    "g2": 0.0,
    "model": 0.0
   },
   "share_moved_gt_030": 0.0008768084173608067,
   "n_nan": {
    "g2": 0,
    "model": 24
   }
  },
  "low": {
   "n": 4040,
   "median_map_g2": 0.0991705663559384,
   "median_map_model": 0.09608547924416536,
   "share_le_025": {
    "g2": 1.0,
    "model": 0.9987623762376238
   },
   "share_le_050": {
    "g2": 1.0,
    "model": 1.0
   },
   "share_moved_gt_030": 0.0,
   "n_nan": {
    "g2": 0,
    "model": 0
   }
  },
  "mid": {
   "n": 2071,
   "median_map_g2": 0.47089535272942773,
   "median_map_model": 0.46369015710418787,
   "share_le_025": {
    "g2": 0.0,
    "model": 0.024709302325581394
   },
   "share_le_050": {
    "g2": 0.5702559150169001,
    "model": 0.5876937984496124
   },
   "share_moved_gt_030": 0.003391472868217054,
   "n_nan": {
    "g2": 0,
    "model": 7
   }
  },
  "na": {
   "n": 223,
   "median_map_g2": NaN,
   "median_map_model": 0.6067582069348003,
   "share_le_025": {
    "g2": NaN,
    "model": 0.16847826086956522
   },
   "share_le_050": {
    "g2": NaN,
    "model": 0.375
   },
   "share_moved_gt_030": 0.0,
   "n_nan": {
    "g2": 223,
    "model": 39
   }
  }
 },
 "sanity_fail_share": 0.025465910406297025,
 "sanity_reasons": {
  "close_contact": 198,
  "bond_graph_changed": 21,
  "forming_distance_range": 1
 },
 "map_g2_vs_records_max_dev": 0.0,
 "move_threshold_A": 0.3
}
```

### 7-2. 입력표 P 보고 (`analysis/geo2/results/P_geo_report.json`, 20개보다 긴 목록은 개수와 앞 5개)
```json
{
 "spec": "docs/specs/GEO2.md 7 (D1) + D0D1 design step 8 (D0)",
 "written": "2026-11-02 21:55:11",
 "git_head": "a1dfd5c6",
 "rule": "명령서 7(D1): 짝 universe = D1PAIR S_folds.csv, P_<rung> = a·b 모두 usable. P_dft·P_g2는 LADDER 표에서 다시 계산해 LADDER 집합 파일과 같아야 한다. P_23 = P_g2 ∩ P_geo, P_023 = P_dft ∩ P_g2 ∩ P_geo. |P_23| < 0.98 |P_g2|(전체 또는 P_main 기준 짝)이면 STOP: D0D1 결정으로 geo-g2 판정은 없지만, geo가 g2 위에 만든 것이라 반응을 잃지 않았는지 보는 파이프라인 검사로 남긴다. D0: $GEO1B/d0/D0_pairs.csv (D0_pairs_report.json stop false, sha256, g2_source g2c; 예비판이면 exit 2), usable 짝 = 두 끝 usable + ok, fold(d0_folds.csv)·edge_kind별 수, 끝이 usable이 아니라 잃은 짝과 이유. dataset·fold별 usable 반응 수.",
 "universe": {
  "path": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/d1_pair/results/S_folds.csv",
  "n": 8389
 },
 "inputs": {
  "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/d1_pair/results/S_folds.csv": "944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7",
  "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/d1_ladder/results/P_report.json": "2f5ef4c0a067e44fad8ee913dc2cf43bbcd150228286eb78225551ea0e9b978d",
  "/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/D0_pairs.csv": "d647bd320d493f20cc4108c1261907fcfc1ed769da51e51ae3417004c8e6ecd7",
  "/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/d0_folds.csv": "873724245e7b9897d926e9cc1a042091166a3523f009d2102794bb6d83534c4e",
  "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/D0_pairs_report.json": "77a5a0ab40ad0e1cc6970045a3fa879f48db487c49a73f07d47b850a38f95bbd",
  "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/dft.parquet": "a20802a3eccf8d0d9b4c2020585d8e5aa5b708a8842cee9c346c4a4d834e9a82",
  "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/g2.parquet": "88ef27c168f2f738ffcbdc2636bb432af91e6b6eb951437da99e012c5aeb0d87",
  "/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/geo.parquet": "f528070ca0805afb2644ea37d1ef6387c47491c8da1945f0e52e27373116a5b5",
  "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/sets/P_dft.csv": "c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5",
  "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/sets/P_g2.csv": "3a475899e9a634f5325c444221d8649f60b18759973defd3fd397a4dbff21240"
 },
 "ladder_check": {
  "P_dft": {
   "file": "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/sets/P_dft.csv",
   "sha256": "c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5",
   "P_report_sha256": "c7e18646d41c461ec5126730d0495b4cdc302e4a2955d0de5ad35719c8be24c5",
   "equal_ids": true,
   "n_file": 8325,
   "n_recomputed": 8325
  },
  "P_g2": {
   "file": "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/inputs/sets/P_g2.csv",
   "sha256": "3a475899e9a634f5325c444221d8649f60b18759973defd3fd397a4dbff21240",
   "P_report_sha256": "3a475899e9a634f5325c444221d8649f60b18759973defd3fd397a4dbff21240",
   "equal_ids": true,
   "n_file": 7458,
   "n_recomputed": 7458
  }
 },
 "n": {
  "P_geo": 7787,
  "P_23": 7385,
  "P_023": 7369,
  "P_g2": 7458,
  "P_dft": 8325
 },
 "n_P_main_reference": {
  "P_geo": 6628,
  "P_23": 6336,
  "P_023": 6332,
  "P_g2": 6401,
  "P_dft": 6874
 },
 "sha256": {
  "P_geo": "61e251dbcf6ddf7cde5a490fd138e2e8a34484af7e9cfbd03f11f6b2bfae1c83",
  "P_23": "21230a14376d26ffc8b75e85b1f3316300e066d997602a2f23c43cb523cc7c9f",
  "P_023": "957cbedf1a472b17e9649cf2c835d55c93ec752fe35fa86d98180e46b1ea641c",
  "P_g2": "6967eebcc2b8b2487c58ffcc04c0a287f0b0eda085c5ec1ed70dcbea96395ac8",
  "P_dft": "fb431c0424bb0ea6acdbe6bf87a4459cf05d3e53361001bb1c22dd81c49685d8"
 },
 "sha256_rule": "sha256 of '\\n'.join(sorted pair ids), no trailing newline",
 "sets": {
  "P_geo": {
   "n": 7787,
   "by_ref_kind": {
    "anchor": 1051,
    "reference": 6736
   },
   "para": 1985,
   "para_by_ref_kind": {
    "anchor": 263,
    "reference": 1722
   },
   "by_pos": {
    "direct": 1805,
    "m": 2033,
    "o": 1964,
    "p": 1985
   },
   "in_P_main": 6628,
   "P_main_reference": 6628,
   "P_main_reference_para": 1699,
   "folds_T1": {
    "-1": {
     "n": 152,
     "para": 41,
     "by_ref_kind": {
      "anchor": 152
     },
     "P_main_reference": 0,
     "P_main_reference_para": 0
    },
    "0": {
     "n": 1695,
     "para": 430,
     "by_ref_kind": {
      "anchor": 221,
      "reference": 1474
     },
     "P_main_reference": 1449,
     "P_main_reference_para": 372
    },
    "1": {
     "n": 1572,
     "para": 382,
     "by_ref_kind": {
      "anchor": 214,
      "reference": 1358
     },
     "P_main_reference": 1312,
     "P_main_reference_para": 314
    },
    "2": {
     "n": 1369,
     "para": 353,
     "by_ref_kind": {
      "anchor": 114,
      "reference": 1255
     },
     "P_main_reference": 1250,
     "P_main_reference_para": 322
    },
    "3": {
     "n": 1333,
     "para": 343,
     "by_ref_kind": {
      "anchor": 36,
      "reference": 1297
     },
     "P_main_reference": 1282,
     "P_main_reference_para": 328
    },
    "4": {
     "n": 1666,
     "para": 436,
     "by_ref_kind": {
      "anchor": 314,
      "reference": 1352
     },
     "P_main_reference": 1335,
     "P_main_reference_para": 363
    }
   }
  },
  "P_23": {
   "n": 7385,
   "by_ref_kind": {
    "anchor": 960,
    "reference": 6425
   },
   "para": 1886,
   "para_by_ref_kind": {
    "anchor": 235,
    "reference": 1651
   },
   "by_pos": {
    "direct": 1705,
    "m": 1936,
    "o": 1858,
    "p": 1886
   },
   "in_P_main": 6336,
   "P_main_reference": 6336,
   "P_main_reference_para": 1630,
   "folds_T1": {
    "-1": {
     "n": 110,
     "para": 27,
     "by_ref_kind": {
      "anchor": 110
     },
     "P_main_reference": 0,
     "P_main_reference_para": 0
    },
    "0": {
     "n": 1599,
     "para": 420,
     "by_ref_kind": {
      "anchor": 217,
      "reference": 1382
     },
     "P_main_reference": 1360,
     "P_main_reference_para": 364
    },
    "1": {
     "n": 1469,
     "para": 352,
     "by_ref_kind": {
      "anchor": 206,
      "reference": 1263
     },
     "P_main_reference": 1221,
     "P_main_reference_para": 287
    },
    "2": {
     "n": 1294,
     "para": 331,
     "by_ref_kind": {
      "anchor": 114,
      "reference": 1180
     },
     "P_main_reference": 1177,
     "P_main_reference_para": 300
    },
    "3": {
     "n": 1309,
     "para": 337,
     "by_ref_kind": {
      "anchor": 35,
      "reference": 1274
     },
     "P_main_reference": 1264,
     "P_main_reference_para": 322
    },
    "4": {
     "n": 1604,
     "para": 419,
     "by_ref_kind": {
      "anchor": 278,
      "reference": 1326
     },
     "P_main_reference": 1314,
     "P_main_reference_para": 357
    }
   }
  },
  "P_023": {
   "n": 7369,
   "by_ref_kind": {
    "anchor": 957,
    "reference": 6412
   },
   "para": 1878,
   "para_by_ref_kind": {
    "anchor": 235,
    "reference": 1643
   },
   "by_pos": {
    "direct": 1703,
    "m": 1935,
    "o": 1853,
    "p": 1878
   },
   "in_P_main": 6332,
   "P_main_reference": 6332,
   "P_main_reference_para": 1628,
   "folds_T1": {
    "-1": {
     "n": 109,
     "para": 27,
     "by_ref_kind": {
      "anchor": 109
     },
     "P_main_reference": 0,
     "P_main_reference_para": 0
    },
    "0": {
     "n": 1592,
     "para": 418,
     "by_ref_kind": {
      "anchor": 215,
      "reference": 1377
     },
     "P_main_reference": 1359,
     "P_main_reference_para": 364
    },
    "1": {
     "n": 1467,
     "para": 351,
     "by_ref_kind": {
      "anchor": 206,
      "reference": 1261
     },
     "P_main_reference": 1220,
     "P_main_reference_para": 287
    },
    "2": {
     "n": 1292,
     "para": 330,
     "by_ref_kind": {
      "anchor": 114,
      "reference": 1178
     },
     "P_main_reference": 1176,
     "P_main_reference_para": 299
    },
    "3": {
     "n": 1308,
     "para": 336,
     "by_ref_kind": {
      "anchor": 35,
      "reference": 1273
     },
     "P_main_reference": 1264,
     "P_main_reference_para": 322
    },
    "4": {
     "n": 1601,
     "para": 416,
     "by_ref_kind": {
      "anchor": 278,
      "reference": 1323
     },
     "P_main_reference": 1313,
     "P_main_reference_para": 356
    }
   }
  },
  "P_g2": {
   "n": 7458,
   "by_ref_kind": {
    "anchor": 964,
    "reference": 6494
   },
   "para": 1916,
   "para_by_ref_kind": {
    "anchor": 236,
    "reference": 1680
   },
   "by_pos": {
    "direct": 1706,
    "m": 1964,
    "o": 1872,
    "p": 1916
   },
   "in_P_main": 6401,
   "P_main_reference": 6401,
   "P_main_reference_para": 1658,
   "folds_T1": {
    "-1": {
     "n": 111,
     "para": 27,
     "by_ref_kind": {
      "anchor": 111
     },
     "P_main_reference": 0,
     "P_main_reference_para": 0
    },
    "0": {
     "n": 1604,
     "para": 424,
     "by_ref_kind": {
      "anchor": 217,
      "reference": 1387
     },
     "P_main_reference": 1364,
     "P_main_reference_para": 367
    },
    "1": {
     "n": 1473,
     "para": 353,
     "by_ref_kind": {
      "anchor": 206,
      "reference": 1267
     },
     "P_main_reference": 1224,
     "P_main_reference_para": 288
    },
    "2": {
     "n": 1347,
     "para": 351,
     "by_ref_kind": {
      "anchor": 114,
      "reference": 1233
     },
     "P_main_reference": 1229,
     "P_main_reference_para": 320
    },
    "3": {
     "n": 1309,
     "para": 337,
     "by_ref_kind": {
      "anchor": 35,
      "reference": 1274
     },
     "P_main_reference": 1264,
     "P_main_reference_para": 322
    },
    "4": {
     "n": 1614,
     "para": 424,
     "by_ref_kind": {
      "anchor": 281,
      "reference": 1333
     },
     "P_main_reference": 1320,
     "P_main_reference_para": 361
    }
   }
  },
  "P_dft": {
   "n": 8325,
   "by_ref_kind": {
    "anchor": 1133,
    "reference": 7192
   },
   "para": 2116,
   "para_by_ref_kind": {
    "anchor": 274,
    "reference": 1842
   },
   "by_pos": {
    "direct": 1967,
    "m": 2164,
    "o": 2078,
    "p": 2116
   },
   "in_P_main": 6874,
   "P_main_reference": 6874,
   "P_main_reference_para": 1759,
   "folds_T1": {
    "-1": {
     "n": 289,
     "para": 78,
     "by_ref_kind": {
      "anchor": 175,
      "reference": 114
     },
     "P_main_reference": 0,
     "P_main_reference_para": 0
    },
    "0": {
     "n": 1778,
     "para": 454,
     "by_ref_kind": {
      "anchor": 223,
      "reference": 1555
     },
     "P_main_reference": 1473,
     "P_main_reference_para": 379
    },
    "1": {
     "n": 1692,
     "para": 417,
     "by_ref_kind": {
      "anchor": 220,
      "reference": 1472
     },
     "P_main_reference": 1408,
     "P_main_reference_para": 339
    },
    "2": {
     "n": 1453,
     "para": 375,
     "by_ref_kind": {
      "anchor": 120,
      "reference": 1333
     },
     "P_main_reference": 1323,
     "P_main_reference_para": 343
    },
    "3": {
     "n": 1382,
     "para": 345,
     "by_ref_kind": {
      "anchor": 55,
      "reference": 1327
     },
     "P_main_reference": 1301,
     "P_main_reference_para": 331
    },
    "4": {
     "n": 1731,
     "para": 447,
     "by_ref_kind": {
      "anchor": 340,
      "reference": 1391
     },
     "P_main_reference": 1369,
     "P_main_reference_para": 367
    }
   }
  }
 },
 "check": {
  "rule": "D1 |P_23| >= 0.98 |P_g2| (all pairs and P_main reference pairs)",
  "ratio": 0.9902118530437114,
  "ratio_P_main_reference": 0.9898453366661459
 },
 "missing_from_geo": {
  "n": 73,
  "n_P_main_reference": 65,
  "by_reason": {
   "a:ok;b:no_tokens+no_x182": 54,
   "a:no_tokens+no_x182;b:ok": 16,
   "a:no_tokens;b:ok": 3
  },
  "end_reason_classes": {
   "no_tokens+no_x182": 70,
   "no_tokens": 3
  },
  "pairs": {
   "n": 73,
   "head": [
    {
     "pair_id": "A00397",
     "ref_kind": "anchor",
     "pos": "p",
     "in_P_main": false,
     "fold_T1": 4,
     "reason": "a:no_tokens+no_x182;b:ok",
     "a_reason": "no_tokens+no_x182",
     "b_reason": "ok"
    },
    {
     "pair_id": "A00579",
     "ref_kind": "anchor",
     "pos": "m",
     "in_P_main": false,
     "fold_T1": 4,
     "reason": "a:no_tokens+no_x182;b:ok",
     "a_reason": "no_tokens+no_x182",
     "b_reason": "ok"
    },
    {
     "pair_id": "A00871",
     "ref_kind": "anchor",
     "pos": "m",
     "in_P_main": false,
     "fold_T1": 4,
     "reason": "a:no_tokens;b:ok",
     "a_reason": "no_tokens",
     "b_reason": "ok"
    },
    {
     "pair_id": "A01033",
     "ref_kind": "anchor",
     "pos": "direct",
     "in_P_main": false,
     "fold_T1": -1,
     "reason": "a:ok;b:no_tokens+no_x182",
     "a_reason": "ok",
     "b_reason": "no_tokens+no_x182"
    },
    {
     "pair_id": "P00369",
     "ref_kind": "reference",
     "pos": "p",
     "in_P_main": true,
     "fold_T1": 1,
     "reason": "a:no_tokens+no_x182;b:ok",
     "a_reason": "no_tokens+no_x182",
     "b_reason": "ok"
    }
   ]
  }
 },
 "usable": {
  "geo": 13568,
  "g2": 8499,
  "dft": 9130
 },
 "reactions_by_dataset": {
  "D1": {
   "n_targets": 9266,
   "n_base": 8879,
   "n_usable": 8759,
   "by_fold": {
    "-1": {
     "targets": 380,
     "base": 240,
     "usable": 232
    },
    "0": {
     "targets": 1874,
     "base": 1795,
     "usable": 1772
    },
    "1": {
     "targets": 1879,
     "base": 1819,
     "usable": 1787
    },
    "2": {
     "targets": 1503,
     "base": 1473,
     "usable": 1463
    },
    "3": {
     "targets": 1435,
     "base": 1400,
     "usable": 1398
    },
    "4": {
     "targets": 2195,
     "base": 2152,
     "usable": 2107
    }
   },
   "unusable_reasons": {
    "no_geo_folder:no_g2": 387,
    "no_tokens+no_x182": 113,
    "no_tokens": 7
   }
  },
  "D0": {
   "n_targets": 5255,
   "n_base": 4817,
   "n_usable": 4809,
   "by_fold": {
    "0": {
     "targets": 1049,
     "base": 962,
     "usable": 961
    },
    "1": {
     "targets": 1050,
     "base": 972,
     "usable": 969
    },
    "2": {
     "targets": 1049,
     "base": 964,
     "usable": 962
    },
    "3": {
     "targets": 1049,
     "base": 943,
     "usable": 941
    },
    "4": {
     "targets": 1058,
     "base": 976,
     "usable": 976
    }
   },
   "unusable_reasons": {
    "no_geo_folder:no_g2": 438,
    "no_tokens+no_x182": 7,
    "no_tokens": 1
   }
  }
 },
 "d0_pairs": {
  "source": "/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/D0_pairs.csv",
  "report": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo1b/results/D0_pairs_report.json",
  "g2_source": "g2c",
  "all": {
   "n": 3804,
   "by_fold": {
    "0": 761,
    "1": 762,
    "2": 760,
    "3": 762,
    "4": 759
   },
   "by_edge_kind": {
    "dipole": 768,
    "leaf": 441,
    "sibling": 2595
   }
  },
  "ok_labels": {
   "n": 3565,
   "by_fold": {
    "0": 719,
    "1": 718,
    "2": 712,
    "3": 711,
    "4": 705
   },
   "by_edge_kind": {
    "dipole": 749,
    "leaf": 427,
    "sibling": 2389
   }
  },
  "usable_geo": {
   "n": 3219,
   "by_fold": {
    "0": 653,
    "1": 655,
    "2": 642,
    "3": 628,
    "4": 641
   },
   "by_edge_kind": {
    "dipole": 688,
    "leaf": 385,
    "sibling": 2146
   }
  },
  "lost_because_end_unusable": {
   "n": 346,
   "by_reason": {
    "a:no_geo_folder:no_g2;b:ok": 268,
    "a:no_geo_folder:no_g2;b:no_geo_folder:no_g2": 61,
    "a:ok;b:no_geo_folder:no_g2": 6,
    "a:no_tokens+no_x182;b:ok": 6,
    "a:ok;b:no_tokens+no_x182": 4,
    "a:no_geo_folder:no_g2;b:no_tokens+no_x182": 1
   },
   "by_fold": {
    "3": 83,
    "1": 63,
    "4": 64,
    "0": 66,
    "2": 70
   },
   "pairs": {
    "n": 346,
    "head": [
     {
      "pair_id": "D00016",
      "fold": 3,
      "edge_kind": "dipole",
      "a_reason": "no_geo_folder:no_g2",
      "b_reason": "ok"
     },
     {
      "pair_id": "D00021",
      "fold": 3,
      "edge_kind": "leaf",
      "a_reason": "ok",
      "b_reason": "no_geo_folder:no_g2"
     },
     {
      "pair_id": "D00027",
      "fold": 3,
      "edge_kind": "sibling",
      "a_reason": "no_geo_folder:no_g2",
      "b_reason": "ok"
     },
     {
      "pair_id": "D00030",
      "fold": 1,
      "edge_kind": "sibling",
      "a_reason": "no_geo_folder:no_g2",
      "b_reason": "ok"
     },
     {
      "pair_id": "D00043",
      "fold": 4,
      "edge_kind": "leaf",
      "a_reason": "no_geo_folder:no_g2",
      "b_reason": "ok"
     }
    ]
   }
  },
  "sha256_usable": "1f021c89d2cdea66a63b81b56fd719b281f62634169c4dd7c4682325832749fa"
 },
 "d0_reference_s3d": {
  "rule": "D0 rung reference (geo_final rel<k>.xyz) refs_q.stereo3d_sig != the Q_keys_d0 stereo3d (= its key's |s3d:, from the Coley label reference); report only (DECISIONS_feat 연결 S-8)",
  "n_checked": 9634,
  "n_differs": 551,
  "n_differs_grouped": 83,
  "n_s3d_not_computable": 0,
  "n_skipped_err_keys": 0,
  "examples": {
   "n": 50,
   "head": [
    {
     "job_id": "D0G2_0014",
     "frag": 1,
     "constrained": false,
     "key": "dipole:COC(=O)[N-][NH+]=C(C)C#N|s3d:COC(=O)[N-]/[NH+]=C(\\C)C#N",
     "rung_s3d": "COC(=O)[N-]/[NH+]=C(/C)C#N"
    },
    {
     "job_id": "D0G2_0041",
     "frag": 1,
     "constrained": true,
     "key": "dipole:[CH2-][O+]=C(C#N)C(C)=O|alt|cfg:Cc2b81a08b2~O1acb729590:t|s3d:[CH2-]/[O+]=C(/C#N)C(C)=O",
     "rung_s3d": "[CH2-]/[O+]=C(\\C#N)C(C)=O"
    },
    {
     "job_id": "D0G2_0062",
     "frag": 1,
     "constrained": true,
     "key": "dipole:COC(=O)[C-](C)[O+]=C(C)C(C)=O|alt|cfg:C2557a0f8a7~Obf7081fbcc:t;Ce1f2399362~Obf7081fbcc:c|s3d:COC(=O)[C-](C)/[O+]=C(\\C)C(C)=O",
     "rung_s3d": "COC(=O)[C-](C)/[O+]=C(/C)C(C)=O"
    },
    {
     "job_id": "D0G2_0078",
     "frag": 1,
     "constrained": true,
     "key": "dipole:CNC(=O)C(=[NH+][CH-]C#N)C(C)=O|alt|cfg:C2f10bec7a5~N40d6f454b5:c;C6bc9bbb16b~N40d6f454b5:t|s3d:CNC(=O)/C(=[NH+]\\[CH-]C#N)C(C)=O",
     "rung_s3d": "CNC(=O)/C(=[NH+]/[CH-]C#N)C(C)=O"
    },
    {
     "job_id": "D0G2_0096",
     "frag": 1,
     "constrained": true,
     "key": "dipole:CC(C#N)=[NH+][N-]c1ccccc1|alt|cfg:C493ac3becc~N603fc545bd:t;N603fc545bd~N65d09d41b8:c|s3d:C/C(C#N)=[NH+]/[N-]c1ccccc1",
     "rung_s3d": "C/C(C#N)=[NH+]\\[N-]c1ccccc1"
    }
   ]
  }
 },
 "outputs": {
  "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis/geo2/results/P_geo_sets.json": "a92fabb0f1d0962bce115e99000393ea4a51b1a8df102f1b1f661a4b3ffae632",
  "/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/sets/P_geo.csv": "658606e016efec215e9bc8ac96169af9f16c487daa907efc71fe021a1e79d0de",
  "/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/sets/P_23.csv": "dfc9de38db16173320806aabbbee2053232f4824cce7f97d85e7039ec617d7b3",
  "/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/sets/P_023.csv": "3632f350d3ec85181659365ef22be3652e2a2cd98aaa7f41b8c43c753e3fc508",
  "/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/sets/P_geo_d0.csv": "6d1f92ac77554af6523b79a37e0cf4f7705af79f4af94a29dfdcd5ab08a838b6"
 },
 "stop": false,
 "stop_reasons": [],
 "running": false
}
```

### 7-3. s_y·SH β 진단 (D1 학습 짝만 대 D1 + D0; `analysis/d1_ladder/results/D0D1_diag_geo.json`, fold 중앙값)
```json
{
 "S/final_T_main": {
  "n_folds": 5,
  "n_d0_median": 2577.0,
  "s_y_d1_median": {
   "d1": 2.1134574591139876,
   "d2": 1.4543503099449084,
   "elst": 4.146620303631068,
   "pauli": 7.919841444946723,
   "oi": 4.426033985827675,
   "disp": 1.1662039355398803,
   "cpcm": 1.2197363059094188,
   "cds": 0.23601088469788767,
   "c_ghost": 0.23069177468402582,
   "barrier": 2.5316409386597223
  },
  "s_y_d1_d0_median": {
   "d1": 3.5184623075395143,
   "d2": 3.1628116137566487,
   "elst": 8.526008246129715,
   "pauli": 15.38534814347171,
   "oi": 9.182593451380379,
   "disp": 2.2599380842299643,
   "cpcm": 3.852858372043285,
   "cds": 0.41690360028461304,
   "c_ghost": 0.43980815347949737,
   "barrier": 4.23766753777165
  }
 },
 "S/inner_fold0": {
  "n_folds": 5,
  "n_d0_median": 2577.0,
  "s_y_d1_median": {
   "d1": 2.066340980255859,
   "d2": 1.337572765529126,
   "elst": 3.8087953106902637,
   "pauli": 7.167720466018104,
   "oi": 4.091868702278428,
   "disp": 1.128187002056421,
   "cpcm": 1.1339855362681053,
   "cds": 0.23800671450910496,
   "c_ghost": 0.21867341504217802,
   "barrier": 2.4776410408715606
  },
  "s_y_d1_d0_median": {
   "d1": 3.7271498337519935,
   "d2": 3.3962872980068903,
   "elst": 9.119519230626159,
   "pauli": 16.35539031461629,
   "oi": 9.824926261937652,
   "disp": 2.424249906522018,
   "cpcm": 4.196159009011506,
   "cds": 0.4466218208068479,
   "c_ghost": 0.46543442223958603,
   "barrier": 4.491927275651342
  }
 },
 "SH/final_T_main": {
  "n_folds": 5,
  "n_d0_median": 2577.0,
  "s_y_d1_median": {
   "d1": 2.113583334267435,
   "d2": 1.441049513764049,
   "elst": 3.5027458042088604,
   "pauli": 7.1176641977320045,
   "oi": 3.9586869525629353,
   "disp": 0.7139828062143719,
   "cpcm": 1.0181625902932443,
   "cds": 0.1793474872156698,
   "c_ghost": 0.23069177468402582,
   "barrier": 2.5261136589882773
  },
  "s_y_d1_d0_median": {
   "d1": 3.515533036694156,
   "d2": 3.0823202625872246,
   "elst": 6.083349472784965,
   "pauli": 11.002202174392382,
   "oi": 6.39479237140365,
   "disp": 1.1888562110951852,
   "cpcm": 3.135013946904852,
   "cds": 0.27171143211904314,
   "c_ghost": 0.43980815347949737,
   "barrier": 4.2287500840755055
  },
  "beta_d1_median": {
   "d1": 0.000979040514860119,
   "d2": 0.018280033216888144,
   "elst": 0.40443157148615,
   "pauli": 0.2826979840355314,
   "oi": 0.29131763480071565,
   "disp": 0.6085366473123334,
   "cpcm": 0.42774872438013667,
   "cds": 0.5915586263569826,
   "c_ghost": 0.0,
   "barrier": -0.0028228653815810293
  },
  "beta_d1_d0_median": {
   "d1": 0.006580405509387557,
   "d2": 0.05654653382446187,
   "elst": 0.6195403451401167,
   "pauli": 0.6016325147198618,
   "oi": 0.6645852621693052,
   "disp": 0.5964316655274678,
   "cpcm": 0.5795097204009253,
   "cds": 0.6828156281656557,
   "c_ghost": 0.0,
   "barrier": 0.006318492200736258
  }
 },
 "SH/inner_fold0": {
  "n_folds": 5,
  "n_d0_median": 2577.0,
  "s_y_d1_median": {
   "d1": 2.0660109516109397,
   "d2": 1.3203375637612065,
   "elst": 3.290519568316396,
   "pauli": 6.429979420079321,
   "oi": 3.5917707647619763,
   "disp": 0.6808219202249617,
   "cpcm": 0.9432060587783553,
   "cds": 0.18166885998047905,
   "c_ghost": 0.21867341504217802,
   "barrier": 2.476787512198577
  },
  "s_y_d1_d0_median": {
   "d1": 3.724920502995462,
   "d2": 3.302648095069774,
   "elst": 6.4287868832372626,
   "pauli": 11.226959715135061,
   "oi": 6.692075421370288,
   "disp": 1.2542899152010394,
   "cpcm": 3.4081859173626166,
   "cds": 0.286454841703486,
   "c_ghost": 0.46543442223958603,
   "barrier": 4.480101717859282
  },
  "beta_d1_median": {
   "d1": -7.396152583816036e-05,
   "d2": 0.015997528985102174,
   "elst": 0.3896621070299022,
   "pauli": 0.30696551956415263,
   "oi": 0.3274202423865227,
   "disp": 0.6134575593112138,
   "cpcm": 0.43334733938575964,
   "cds": 0.5845525673558267,
   "c_ghost": 0.0,
   "barrier": -0.005151740073936446
  },
  "beta_d1_d0_median": {
   "d1": 0.006624414210465562,
   "d2": 0.05948665662056477,
   "elst": 0.6210356716717829,
   "pauli": 0.6414964040299533,
   "oi": 0.7148553565967938,
   "disp": 0.5968415804060848,
   "cpcm": 0.5833312143422968,
   "cds": 0.6887406045665772,
   "c_ghost": 0.0,
   "barrier": 0.007423353699270306
  }
 }
}
```

### 7-4. 8-2 검사와 smoke 예상
- `analysis/geo2/results/C_tests_geo.json` all_pass True (sha256 `48e8f743207ead0435cd6bcfb9d346df35c6685fa22dc2354a5343a5a42c2d46`, merged 2026-11-02 23:02:11); 검사별 {'L-1': 'pass', 'L-2': 'pass', 'L-3': 'pass', 'L-4': 'pass', 'L-5': 'pass', 'L-7': 'pass', 'L-8': 'pass', 'L-9': 'pass', 'L-11': 'pass', 'L-D0': 'pass', 'smoke': 'pass', 'diag': 'pass', 'R-1': 'pass', 'R-2': 'pass', 'R-3': 'pass'}
- 검사가 돈 코드 = 지금 코드(['geo2/d0d1_splits.py', 'inputs.py', 'lad_common.py', 'tests/run_tests.py', 'train_atoms.py', 'train_xgb.py']), 검사의 geo 입력표·라벨·분할 색인 = 4절의 파일. evaluate.py는 기록만 (검사 때와 같음: True).
- smoke (rung geo, fold 0): 원자 모델 ok True, XGB ok True (gate M-1′, D0 학습 짝 포함).
- GPU 예상: 162.227 GPU시간 ≤ 상한 200.0 (기준 rung별 {'g1': 110.68330689082264, 'dft': 162.2269059246121}, 사용 dft; FU2 상한식 321.31; LADDER rung별 실제 {'dft': 46.618139009608164, 'g1': 31.71513245999813, 'g2': 12.323918750484784}); 작업자 수별 가장 긴 slice 시간 {'1': 162.22690592461205, '2': 82.16718049759578, '3': 61.859788857125295, '4': 46.76115530614096, '5': 32.44538118492242, '6': 31.85181520413323, '7': 30.007973652992064, '8': 29.414407672202874, '9': 29.17719324494635, '10': 17.276418127750944}; XGB 155.011 CPU시간.

### 7-5. LADDER 기준값 (`analysis/geo2/results/selftest/E_report.json`, eval_geo2 selftest; 참고용)
```json
{
 "n_pairs": {
  "P_23": 6401,
  "P_023": 6394,
  "own_dft": 6874,
  "own_g2": 6401,
  "own_geo": 6401
 },
 "n_para": {
  "P_23": 1658,
  "P_023": 1655
 },
 "rep8": {
  "dft|barrier": [
   "SH",
   "AW"
  ],
  "dft|c_ghost": [
   "SH",
   "AW"
  ],
  "dft|cds": [
   "SH",
   "XGB"
  ],
  "dft|cpcm": [
   "SH",
   "GNN"
  ],
  "dft|d1": [
   "SH",
   "AW"
  ],
  "dft|d2": [
   "SH",
   "AW"
  ],
  "dft|disp": [
   "S",
   "XGB"
  ],
  "dft|elst": [
   "SH",
   "AW"
  ],
  "dft|oi": [
   "SH",
   "GNN"
  ],
  "dft|pauli": [
   "SH",
   "GNN"
  ],
  "g1|barrier": [
   "SH",
   "GNN"
  ],
  "g1|c_ghost": [
   "SH",
   "AW"
  ],
  "g1|cds": [
   "SH",
   "XGB"
  ],
  "g1|cpcm": [
   "SH",
   "GNN"
  ],
  "g1|d1": [
   "SH",
   "AW"
  ],
  "g1|d2": [
   "SH",
   "GNN"
  ],
  "g1|disp": [
   "S",
   "XGB"
  ],
  "g1|elst": [
   "SH",
   "AW"
  ],
  "g1|oi": [
   "SH",
   "AW"
  ],
  "g1|pauli": [
   "SH",
   "AW"
  ],
  "g2|barrier": [
   "SH",
   "AW"
  ],
  "g2|c_ghost": [
   "SH",
   "XGB"
  ],
  "g2|cds": [
   "SH",
   "GNN"
  ],
  "g2|cpcm": [
   "SH",
   "XGB"
  ],
  "g2|d1": [
   "S",
   "GNN"
  ],
  "g2|d2": [
   "SH",
   "XGB"
  ],
  "g2|disp": [
   "S",
   "AW"
  ],
  "g2|elst": [
   "SH",
   "GNN"
  ],
  "g2|oi": [
   "SH",
   "TF"
  ],
  "g2|pauli": [
   "SH",
   "GNN"
  ],
  "geo|barrier": [
   "SH",
   "AW"
  ],
  "geo|c_ghost": [
   "SH",
   "XGB"
  ],
  "geo|cds": [
   "SH",
   "GNN"
  ],
  "geo|cpcm": [
   "SH",
   "XGB"
  ],
  "geo|d1": [
   "S",
   "GNN"
  ],
  "geo|d2": [
   "SH",
   "XGB"
  ],
  "geo|disp": [
   "S",
   "AW"
  ],
  "geo|elst": [
   "SH",
   "GNN"
  ],
  "geo|oi": [
   "SH",
   "TF"
  ],
  "geo|pauli": [
   "SH",
   "GNN"
  ]
 },
 "goal_mae": {
  "ENS4·S": {
   "oi": 0.9918543237749985,
   "cpcm": 0.2885086115417319,
   "mae_ok": false
  },
  "ENS4·SH": {
   "oi": 0.9747820149023556,
   "cpcm": 0.289633866150425,
   "mae_ok": false
  },
  "REP8": {
   "oi": 1.0375224111270578,
   "cpcm": 0.31543552373833583,
   "mae_ok": false
  },
  "REP·S": {
   "oi": 1.0920929530892547,
   "cpcm": 0.2911051673646118,
   "mae_ok": false
  },
  "REP·SH": {
   "oi": 1.0375224111270578,
   "cpcm": 0.31543552373833583,
   "mae_ok": false
  }
 }
}
```

## 8. gate와 멈춤 규칙

### 8-1. gate (학습 코드가 매 실행 전에 확인; train_atoms GA-4 / GA-5 / GA-7 / GA-14 / GB-2 / GB-6, train_xgb XG-G3)
- 이 md와 `prereg_geo2b.json`이 git에 있고 HEAD 이후 그대로; sha256(md) = json `md_sha256`; json `rungs` = ["geo"], `split` = "T1D0".
- json `inputs_sha256`에 라벨 파일과 geo 입력표가 있고(need), 아래 표의 파일이 모두 그 sha256 그대로다.
- `analysis/d1_ladder/results`의 LADDER 결과 파일 84개(json `ladder_results_sha256`)가 그대로다(작성 때 git HEAD와 같음을 확인; `collect --stage 3`도 다시 확인).
- P: `P_build_geo.json` all_pass·입력표 sha256, `P_geo_report.json` stop false이고 그 inputs의 입력표 sha256 = 지금 (연결 S-3). 분할 색인: d0d1_splits 읽기 규칙 전부(final). C: `C_tests_geo.json` all_pass(geo 모드)와 그 code_sha256의 학습 코드 5개 = 지금 파일.

| 파일 | 역할 | 열 | sha256 |
|---|---|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/F_pairs_d0d1.csv` | labels: F_pairs_d0d1.csv = F_pairs_cr2.csv + D0 rows (need) | geo | `9edb8725c402acb571cd8ec7dabee0506391c27e65fc0acacf82fd9a8402322c` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/inputs/geo.parquet` | geo input table D1 + D0 (inputs role, GEO2.md 6-1 P; need) | geo | `f528070ca0805afb2644ea37d1ef6387c47491c8da1945f0e52e27373116a5b5` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/X182_d1_geo.parquet` | geo 182 global features (XGB input = token glob source) | geo | `c54b95d611817715b1f65d332bb96d7c3f0140bfe7729c809089319464b434fc` |
| `analysis/geo2/results/fu2_geo/T_report.json` | geo FU2 report T_report.json (token layout / versions / manifest) | geo | `7254fb86e932b73502ce2bdd23262652551f9b17feedaa236955cce95d71e659` |
| `analysis/geo2/results/fu2_geo/T_manifest.csv` | geo FU2 report T_manifest.csv (token layout / versions / manifest) | geo | `52722bf547cfc9777ce87d014cf6dda0a7bc622102a0eaf9561c527f4e4abf5b` |
| `analysis/geo2/results/fu2_geo/Q_report.json` | geo FU2 report Q_report.json (token layout / versions / manifest) | geo | `88fa104dbf47f0acffcb64b1f2090dc696faa7275106f5cd21030cd957d289b2` |
| `analysis/geo2/results/fu2_geo/S_splits.json` | geo FU2 report S_splits.json (token layout / versions / manifest) | geo | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `analysis/geo2/results/S_splits_d0d1.json` | D0D1 split index S_splits_d0d1.json (d0d1_splits.py build) | geo | `b9ee62bafa16c5d017fdee1b545af9bed37d5f7e5a3bddcbc461a6cb32ac2df1` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/splits/T1D0__f0.json` | T1D0 fold 0 split file | geo | `d0902e13cf6d75b4fdeaf9126a197dbfdd2f86d1a594f8e01029c257c59381f3` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/splits/T1D0__f1.json` | T1D0 fold 1 split file | geo | `e6fd840d8d89ecbb5053bec950d8c12cc22014c4cb242c0442b477769fcbec05` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/splits/T1D0__f2.json` | T1D0 fold 2 split file | geo | `070c553da0daed42b0927237ccd125fc0ecc552beeab5ef1d1847cdc40a97e57` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/splits/T1D0__f3.json` | T1D0 fold 3 split file | geo | `ce986bcda835a925df519a768aad92fbb909c8cde837f8ed22a7cf067b0e8b32` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo2b/splits/T1D0__f4.json` | T1D0 fold 4 split file | geo | `0e470830c101cb001316ab50c7fee551c92ea527a5dbda92f0e9b62c1e5d95c0` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | F_pairs_cr2.csv (the D1 rows of the label file, byte-identical; LADDER y) | shared | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `analysis/d1_pair/results/F_pairs.csv` | original F_pairs.csv (L-11 invariant of the D1 rows) | shared | `aeaba8f8794358c126a57e2f6829792970d1e5c7d8433100ac3134a56c73f043` |
| `analysis/d1_pair/results/fu2/S_splits.json` | FU2 S_splits.json (split index of the D1 lists) | shared | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `analysis/d1_pair/results/S_folds.csv` | S_folds.csv (D1 folds, cores; XGB inner GroupKFold) | shared | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `analysis/d1_ladder/results/V_report.json` | label provenance of F_pairs_cr2.csv (V_report outputs) | shared | `d4b2c0dfd340cb81596352dd6ca74452c1cce4a1d8601e05170abe1097a46f1f` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | FU2 T1 fold 0 split file (the D1 part of T1D0) | shared | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | FU2 T1 fold 1 split file (the D1 part of T1D0) | shared | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | FU2 T1 fold 2 split file (the D1 part of T1D0) | shared | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | FU2 T1 fold 3 split file (the D1 part of T1D0) | shared | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | FU2 T1 fold 4 split file (the D1 part of T1D0) | shared | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/D0_pairs.csv` | GEO1b D0_pairs.csv (the D0 rows; final build) | geo | `d647bd320d493f20cc4108c1261907fcfc1ed769da51e51ae3417004c8e6ecd7` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/d0_folds.csv` | GEO1b d0_folds.csv (frozen D0 folds) | geo | `873724245e7b9897d926e9cc1a042091166a3523f009d2102794bb6d83534c4e` |
| `/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/d0_twins.csv` | GEO1b d0_twins.csv (D1 twins of D0 reactions) | geo | `8511ce4c98f2679284660da0a89d13c33eb440cae0a0ea2598c4038a2760e5ee` |
| `analysis/d1_ladder/train_atoms.py` | stage-3 training code | code | `161c4c07a07c4ff48f358bfb44b154bb2c78e1599fba4ee05124650371ceafbc` |
| `analysis/d1_ladder/train_xgb.py` | stage-3 training code | code | `64a5c449248f126d9db9cd411c7e81f26e16476b02bd77d27410fd0d7f1af570` |
| `analysis/d1_ladder/lad_common.py` | stage-3 training code | code | `e0d64e1e70bc0df60bd9f4befb6cdeebca3effcecc0eb57ab6b50f555ff62afe` |
| `analysis/d1_ladder/inputs.py` | stage-3 training code | code | `5bc5625e300e15944a8f89076a2d9e2feb56e960e0e0b09111e4b8ecb98a48c4` |
| `analysis/d1_pair/fu2/models.py` | stage-3 training code | code | `de2065a7d39c1e008b31274ab1b4fc94fa123ce53eda14a3c2f426dbd0d6197a` |
| `analysis/d1_pair/fu2/train.py` | stage-3 training code | code | `2f1ab7a06bfec20901cfd9130a9784961070e6477728ade92fed9a1cce729469` |
| `analysis/d1_pair/fu2/fu2_common.py` | stage-3 training code | code | `66dcf72f5b23a015e1750e4b2ac458dd61a4722050a7be24a03b8269a628dfe2` |
| `analysis/d1_pair/fu2/tokens.py` | stage-3 training code | code | `f37207bbf987764cfbadd064ad4f9ffbdb33e54daa7563825b3966d0c5e95514` |
| `analysis/d1_pair/fu2/atoms.py` | stage-3 training code | code | `fa6ad97f3436f4eecb62085bc78edb2ddf8fcb92a012c73469ae97368cc40ac8` |
| `analysis/d1_pair/d1pair_common.py` | stage-3 training code | code | `65a6e2da81e4dedf20c51423cbfe9c97e8b8f2272b24454a739a09e8552aeedc` |
| `analysis/d1_pair/d1_train.py` | stage-3 training code | code | `414b5cc66762391691b252235c9133f0d193312a8039d35369299a8dfe30027c` |
| `analysis/espley_xtb_repro/train_ml_single.py` | stage-3 training code | code | `05c6b38cdf7803f4a6c6b2d643555db95b5e366ea02714049232add38a55da4c` |
| `analysis/geo2/d0d1_splits.py` | stage-3 training code | code | `5c1d9c5f21e2859e2b31c36661003a0c0667a86b2204004f19f0336f1ba16a20` |

- 학습 코드 git 상태(작성 시): {'analysis/d1_ladder/train_atoms.py': 'committed', 'analysis/d1_ladder/train_xgb.py': 'committed', 'analysis/d1_ladder/lad_common.py': 'committed', 'analysis/d1_ladder/inputs.py': 'committed', 'analysis/d1_pair/fu2/models.py': 'committed', 'analysis/d1_pair/fu2/train.py': 'committed', 'analysis/d1_pair/fu2/fu2_common.py': 'committed', 'analysis/d1_pair/fu2/tokens.py': 'committed', 'analysis/d1_pair/fu2/atoms.py': 'committed', 'analysis/d1_pair/d1pair_common.py': 'committed', 'analysis/d1_pair/d1_train.py': 'committed', 'analysis/espley_xtb_repro/train_ml_single.py': 'committed', 'analysis/geo2/d0d1_splits.py': 'committed'}
- geo unit 지문의 code_sha256 기준값(json `code.fingerprint_keys`; 기록만, 뒤 단계가 읽지 않음): train_atoms: train_atoms.py `161c4c07a07c`, lad_common.py `e0d64e1e70bc`, fu2/models.py `de2065a7d39c`, fu2/train.py `2f1ab7a06bfe`, fu2/fu2_common.py `66dcf72f5b23`, fu2/tokens.py `f37207bbf987`, d1pair_common.py `65a6e2da81e4`; train_xgb: train_xgb.py `64a5c449248f`, lad_common.py `e0d64e1e70bc`, d1_train.py `414b5cc66762`, d1pair_common.py `65a6e2da81e4`, train_ml_single.py `05c6b38cdf78`, fu2/fu2_common.py `66dcf72f5b23`, fu2/tokens.py `f37207bbf987`; d0d1_code: `5c1d9c5f21e2`

### 8-2. GEO2.md 13절 (HPC 규칙, 그대로)
> ## 13. HPC 규칙
> - python, ORCA, xtb, RDKit은 모두 `sbatch`(`--time=48:00:00`)로만 돌린다. 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만 한다.
> - MaxJobs 10, MaxSubmit 20(배열 원소 포함).
> - **읽기만 하는 것:** GEO1(`$GEO`, `analysis/geo/results`; Y의 새 파일 제외), LADDER 결과 파일과 기존 unit, D1-G2, D1-HF, D1-G3, D1 run·job 폴더, `labels_d1.json`, `F_pairs*.csv`
> - **새 파일은 여기에만 쓴다:** `analysis/geo2/`, `$GEO2`, `analysis/geo/train_m1.py`·`$GEO/preds/final_fm1_*`·`analysis/geo/results/Y_plan.json`, LADDER의 `*_geo.*` 새 결과 파일과 `$LAD/units/L__geo__*`, 8-1절의 코드 추가
> - 원자적 쓰기, 멱등. 수치는 파일에서 옮긴다.
> - 결과를 본 뒤 판정·선택 규칙·짝 집합을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

### 8-3. GEO2.md 14절 (결정 기록과 STOP, 그대로; M3 상한은 D0D1로 200 GPU시간(11-02 M3-cap))
> ## 14. 결정 기록과 STOP
>
> - 명령서에 없는 갈림길이 생기면 추천안을 고르고 `results/DECISIONS.md`에 적은 뒤 진행한다.
>
> | 단계 | STOP 조건 |
> |---|---|
> | 0 | 1절 시작 조건 중 하나라도 아님 (특히 0-6 selftest) |
> | Y | 예상 > 30 GPU시간; fold −1 반응이 학습 목록에 섞임 |
> | Z | `E_rxn.csv` 재현 실패; 대체 > 2 %; 한 반응이 두 fold에서 예측됨 |
> | T | T-2–T-4 중 하나라도 실패 |
> | G·F·H·P | 6-2절 기준; H 예상 > 2,500코어시간 |
> | P | \|P_23\| < 0.98 × \|P_g2\| |
> | R·M3 | PREREG commit 전 학습 시도; 8-2 검사 실패; 예상 > 80 GPU시간; 캐시 지문 불일치; 40 unit 중 하나라도 없음; 기존 LADDER 결과 파일이 바뀜(sha256 전후 비교) |
> | E | selftest 실패; 비교하는 rung 사이 시험 짝 목록 불일치 |
> | 언제든 | 라벨·기존 결과 파일을 바꿔야 할 것 같음; 이 문서에 없는 계산 방법 변경이 필요해 보임 |
>
> STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.

- **원칙:** 결과를 본 뒤 선택 규칙·짝 집합·분할을 바꾸지 않는다. 바꾸면 이탈로 기록한다(GEO2.md 13절).
