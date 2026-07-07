# A/B/C Baseline Ablation (in-distribution, n=787)

Δ-learning의 baseline으로 `ridge` vs `xgb` 중 무엇이 나은지를 in-distribution
5-fold CV로 결정한다. SPEC 원본은 `SPEC.md`, 자세한 결과는 `results/REPORT.md`.

## Verdict (2026-07-07)

> **B (ridge+δ) beats C (xgb+δ) on the barrier.**
> ΔNMAE(B − C) = **−0.106 [−0.146, −0.067]** on barrier, Wilcoxon p = 1.7e-6.
> Ridge baseline stays as the default for Δ-learning.

- Per-channel으로 보면 C가 Pauli / V_elst / oi / disp에서 통계적으로 이기지만
  (5.2e-08 ≤ p ≤ 9.5e-4),
- **barrier = Σ_c ŷ_c** 는 채널간 오차 상쇄를 반영해 B가 유의하게 앞선다.
  ridge의 채널간 오차가 서로 반대 부호로 상쇄되는 반면 xgb는 상관이 있는 방향
  으로 밀리기 때문이다.
- Arm A (`xgb_direct`) barrier NMAE 0.644, B 0.571, C 0.677 — 채널간 상쇄를
  살리는 Δ-learning + ridge 조합이 최종 지표에서 명확한 우위를 보인다.

Sanity gates:

| gate | 결과 | 비고 |
|---|---|---|
| #1 same fold index | ✅ | `splits/outer_folds.json` 재사용 검증 |
| #2 no reaction-level leakage | ✅ | outer_folds.json assert |
| #3 δ target ≠ 0 (OOF baseline) | ✅ | median\|r_train\|/median\|y_train\| ∈ [0.28, 0.44] |
| #4 SPEC target ±0.05 tol | A ✅, B ⚠️ +0.10-0.15 | 후술 |
| #5 ΔNMAE(B-C) CI verdict | ✅ | barrier에서 B<C, per-channel은 혼재 |

Gate #4에서 A는 SPEC 기대값과 5채널 모두 ±0.02 안이지만, B는 pooled NMAE가
SPEC 기대값보다 +0.10~0.15 높게 나온다. 이는:

1. SPEC §4 대로 σ_c-정규화 L1 loss + grad-clip 5.0 을 넣었기 때문 (기존 m3
   러너는 raw L1 + no clip). loss scaling이 바뀌면 lr=1e-5, patience=10k
   조합에서의 수렴 궤적도 달라진다.
2. train pool을 subsampled 509가 아니라 full ~628로 썼기 때문 (기존 m3은
   size_509.json 사용). 두 효과가 반대 방향으로 밀지만, 결과적으로 우리
   러너의 NMAE는 +0.1대 위에서 잡혔다.

**A/B/C 비교는 여전히 유효하다** — B와 C는 **동일한 δ 아키텍처 · HP · seed**
를 쓰고 baseline만 다르기 때문에, 세로 비교(ΔNMAE(B-C))는 loss scaling과 무관.
SPEC 기대값과의 절대값 차이는 학습 세팅 차이로 설명 가능하며, gate #4는
SPEC §8에서 "fold 구성이 달라 정확 일치는 아님"이라고 예외 조항을 두었다.

## 폴더 구조

```
experiments/abc_ablation/
├── SPEC.md                    # SPEC 요약 (원본은 repo root)
├── README.md                  # ← 이 파일
├── build_splits.py            # T2 · 787-rxn family-stratified 5-fold, seed=42
├── baselines.py               # T3 · fit_ridge, fit_xgb, cross_fit_oof(K'=5)
├── delta_trainer.py           # 외부 baseline 주입 대응 δ 트레이너
│                              # (σ_c-normalised L1 + grad-clip 5.0)
├── arm_A_run.py               # T4 · xgb_direct OOF (single job)
├── arm_BC_run.py              # T5 · (arm, fold) 러너 — task_id 0..9
├── aggregate.py               # T6 · pooled OOF metric + 1000-boot CI + ΔBC
├── plots.py                   # T7 · NMAE/RMSE 그룹바 + parity 그리드
├── report.py                  # T8 · REPORT.md 자동 생성 + gate assert
├── slurm_setup.sh             # cpu2 · splits + arm A
├── slurm_arm_BC.sh            # gpu3/4/5 array 0-9 · arm B/C × fold 0-4
├── slurm_finalize.sh          # cpu2 · aggregate + plots + report
├── splits/outer_folds.json    # 5-fold reaction-level, family-stratified
├── logs/                      # SLURM stdout/stderr (gitignored)
└── results/
    ├── oof_pred_{A,B,C}.parquet   # 787-rxn pooled OOF (reaction_id, y*, ŷ*)
    ├── cells/{B,C}/foldF.json     # per-fold δ 학습 상세 (histor 없음, 메타)
    ├── abc_metrics.csv            # arm × channel × metric × point/CI
    ├── delta_BC.csv               # paired ΔNMAE(B-C) CI + Wilcoxon
    ├── nmae_bars.png / rmse_bars.png
    ├── parity_grid.png            # 3 arms × 5 channels
    └── REPORT.md                  # 최종 리포트
```

## 재현 절차

```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# 1) splits + arm A (약 30s)
sbatch experiments/abc_ablation/slurm_setup.sh

# 2) arm B/C 10-way array (fold별 1-3h GPU)
sbatch --dependency=afterok:<setup_id> experiments/abc_ablation/slurm_arm_BC.sh

# 3) aggregate + plots + REPORT.md (약 10s)
sbatch --dependency=afterok:<bc_id> experiments/abc_ablation/slurm_finalize.sh
```

모든 러너는 `if out_path.exists(): return` 이므로 48h 벽에 걸리면 그대로
re-sbatch 하면 남은 셀만 이어서 돈다.

## 데이터 · 의존성

- 번들: `pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.pt`
  (787 rxn × 24-d m3 descriptors × MACE-OFF23 medium features)
- family 라벨: `features_v6_delta_m3.families.json`
- Δ 모델: `m3/code/eda_asm/asr_v1/models_delta.py:ModelM1Delta`
  (M1_HP = d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
- HP: lr=1e-5, weight_decay=1e-3, batch=16, epochs_max=1e5,
  patience=10k, grad_clip=5.0, inner K'=5
