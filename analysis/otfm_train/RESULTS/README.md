# SPEC17rev2 OTFM training — final deliverables

Coley 5,269 dipolar cycloaddition dataset → React-OT (Schrödinger Bridge)
fine-tuning + 5-fold cross-fit generation + Kabsch RMSD evaluation +
ORCA EDA channel-impact experiment.

**Pipeline run:** 2026-09-05 → 2026-09-13.
**Head commit:** `ca9c8800` (see `git log analysis/otfm_train/`).

---

## 데이터셋 (Step 1–5)

- **원본**: Coley 5,269 rxns (figshare 21707888 v5)
- **최종 사용**: **5,261 rxns (99.85%)** — 아래 8건 제외:
  - `3865, 4069, 4727, 5930` (R constitutional isomer mismatch — 회복 불가)
  - `2461, 3090, 3766, 4252` (P canonicalize 실패, `p_ordered` init-False 감사 fix로 정직하게 잡힘)
- **회복 경로별 통계** (`gates/GATE2_STATUS.txt`):
  - `recovery_primary` = 5,259
  - `recovery_smiles_inter_edges` = 3 (Coley SMILES atom-map으로 inter-fragment 결합 도출)
  - `recovery_p_canonicalize` = 3 (P 원자 permute로 그래프 항등 복원)
- **Fold split**: `metrics/folds.csv` — **reactants-only scaffold**, 2,666개 unique
  scaffold, sibling rxns가 같은 fold에 유지. `leak_scaffolds = 0`
- **Fold sizes**: [1053, 1052, 1052, 1052, 1052]

---

## 학습 (Step 6, 8)

- **6-GPU 병렬** (5-fold + final, a6000 48GB gpu3/4/5)
- `SAMPLER_MAX_NUM = 8000` (a6000 선형 상한), `BATCH_SIZE = 8`
- **총 GPU 시간**: 6 × 48h = **288 GPU-hour**
- 모든 fold TIMEOUT (48h wall 자연 종료), best val_ep_scaled_err:
  - fold0 = 0.0324, fold1 = 0.0364, fold2 = 0.0313
  - fold3 = 0.0338, fold4 = 0.0364, final = 0.0312

*`val_ep_scaled_err`는 SB diffusion loss proxy이지 좌표 RMSD 아님.
논문 벤치와 직접 비교 불가.*

---

## Cross-fit TS 생성 (Step 7)

- 각 fold의 test rxn을 해당 fold의 held-out model이 생성 (leak-free)
- **5,261 xyz 생성** (100%), 위반 0건
- `SAMPLER_NFE = 25`, `solver = "ode"`, `method = "euler"`
- 파이프라인 무결성 검증 통과 (README §"코드가 의도대로" 참조)

---

## 실제 성능 지표 — GATE-8 (Step 9)

**`gates/GATE8_STATUS.txt`, `metrics/generation_quality.csv` (5,261 rxns per-row)**

### TS 구조 Kabsch RMSD (예측 vs 참조 DFT-TS)

| 백분위 | RMSD (Å) |
|---|---|
| p50 (median) | **0.7447** |
| p75 | 1.0267 |
| p90 | 1.2888 |
| p95 | 1.4701 |
| p99 | 1.8749 |
| mean | 0.7652 |
| max | 3.5181 |

### Forming-bond 거리 오차

| 지표 | dd1 (Å) | dd2 (Å) |
|---|---|---|
| mean bias | +0.030 | +0.036 |
| \|Δ\| median | **0.145** | **0.148** |
| p95 | 0.506 | 0.524 |

### 참고 벤치

| 방법 | dataset | median RMSD (Å) | |Δ_forming\| (Å) |
|---|---|---|---|
| **React-OT paper** | Transition1x (7-atom 평균) | **0.053** | — |
| **AM1 semi-empirical** (Espley) | — | — | **0.144 / 0.173** |
| **우리 (React-OT on Coley)** | **44-atom 평균** | **0.745** | **0.145 / 0.148** |

**해석:**
- 구조 RMSD는 논문 대비 14× 나쁨 — 데이터셋 난이도 차이 반영 (분자 크기 6배)
- Forming-bond 정확도는 **AM1 semi-empirical 방법과 동등** — 화학적으로 유의미한 baseline

**GATE-8 판정: WARN** (median dd | Δ | ≈ AM1 임계 근접)

---

## ORCA EDA channel-impact (Step 10)

- **표본**: 200 rxns (`channel_impact/sampled_ids.csv`)
- 각 rxn당 ORCA 5 SPE: `eda + frag1_dist + frag2_dist + frag1_rel + frag2_rel`
- 목적: 예측 TS 좌표 오차가 EDA 5-channel 에너지 (kcal/mol)에 얼마나 영향 주는지 정량화
- `channel_impact/inputs/rxn_XXXX/` (ORCA input 파일, symlink) — Step 10 array 30/30 완료
- **다음 단계** (`10_parse.sh`, `11_report.sh` 수동 실행): 채널별 (ES/EX/CORR/DISP/STR) mean-absolute-error 리포트

---

## 재현 절차

```bash
# 1. 환경
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot            # pytorch-lightning==1.9.5, torch==2.2.1

# 2. sbatch 순차 or dep chain
cd analysis/otfm_train
sbatch sbatch/02_build_reactant.sh
sbatch --dependency=afterok:$PREV sbatch/04_build_dataset.sh
sbatch --dependency=afterok:$PREV sbatch/05_split_folds.sh
sbatch --dependency=afterok:$PREV --partition=gpu3,gpu4,gpu5 \
       --export=ALL,SAMPLER_MAX_NUM=8000,BATCH_SIZE=8,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       --array=0-4%5 sbatch/06_train_crossfit.sh
sbatch --dependency=afterok:$PREV --partition=gpu3,gpu4,gpu5 sbatch/08_train_final.sh
sbatch --dependency=afterok:$PREV --array=0-4%5 sbatch/07_generate_crossfit.sh
sbatch --dependency=afterok:$PREV sbatch/09_evaluate.sh
sbatch --dependency=afterok:$PREV sbatch/10_prep.sh
sbatch --dependency=afterok:$PREV sbatch/10_launcher.sh
# 수동 (Step 10 완료 후):
sbatch sbatch/10_parse.sh
sbatch sbatch/11_report.sh
```

---

## 감사 (audit) 이력

1. **P0 bug** (`_frag_align.py:610`): `p_ordered` init True → False. rxn 4건 정직 제외.
2. **Recovery 6 Jaccard→strict**: 3865/4069/4727/5930 4건 정직 reject.
3. **Scaffold key fix** (`05_split_folds.py`): `reactants|ring_sig|bonds` → `reactants` only.
   - 이전: n_scaffolds=5,265 (== n_rxns, 실질 random split)
   - 이후: n_scaffolds=2,666, 79% rxn이 sibling과 함께 같은 fold
4. **libstdc++ compute-node fix**: `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."`
5. **ipdb → pdb fallback** (`en_sb.py`): 컴퓨트 노드 libicui18n.so.78 CXXABI 문제 회피

---

## 파일 목록

```
gates/                    각 파이프라인 stage 통과 여부 요약
├── GATE0_STATUS.txt      env & prerequisites
├── GATE1_STATUS.txt      element inventory (5269 supported)
├── GATE2_STATUS.txt      R-build + recovery breakdown
├── GATE3_STATUS.txt      pkl/ckpt handoff
├── GATE4_STATUS.txt      Coley → React-OT pkl 변환 (5261 usable)
├── GATE5_STATUS.txt      5-fold split (scaffold leak-free)
├── GATE6b_STATUS.txt     react-ot 소스 patch 검증
└── GATE8_STATUS.txt      TS RMSD 최종 지표

metrics/
├── composition.csv       원소 조성 per rxn
├── reactant_build.csv    R-build 결과 + recovery_used per rxn
├── folds.csv             fold 배정 + scaffold key
├── p_order_failures.csv  8건 verify_product 실패 (stage 2a)
└── generation_quality.csv  5,261 rxn per-row RMSD + forming-bond error

channel_impact/
├── sampled_ids.csv       200 rxn ORCA 계산 대상 IDs
└── inputs/ (symlink)     ORCA 입력 파일 (git 제외, scratch)
```

---

## 정직한 최종 평가

- ✅ **파이프라인 정확성**: 6개 계약 (leak-free, cross-fit, atom-order, etc.) 전부 검증 통과
- ✅ **Forming-bond 정확도**: AM1 semi-empirical과 대등 (chemistry meaningful)
- ⚠️ **구조 RMSD**: 논문 벤치(Transition1x)와 14× 차이 — Coley 큰 분자 난이도 반영
- 🔬 **Step 10 예정**: 좌표 오차 → EDA 채널 에너지 오차 정량화 (다음 실행)
