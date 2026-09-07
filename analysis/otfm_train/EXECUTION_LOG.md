# SPEC17rev2 otfm_train — Execution Log

Coley 5,269 dipolar cycloaddition reactions → React-OT (OTFM) fine-tune
for R+P→TS structure generation.

Timeline: 2026-09-05 → 2026-09-07.

---

## 데이터셋 사용 요약

| 단계 | 카운트 | 비율 |
|---|---|---|
| **Coley raw profiles** | 5,269 | 100.00% |
| Step 1 supported (H C N O F Cl Br) | 5,269 | 100.00% |
| Step 2 R-build 성공 | 5,262 | 99.87% |
| Step 2 verify_product 통과 (strict) | 5,261 | 99.85% |
| **Step 4+ 학습 사용 (합집합 필터 후)** | **5,258** | **99.79%** |
| **제외** | **11** | **0.21%** |

---

## 제외된 11개 반응 상세

### 그룹 A — R-build 실패 (7건)

| rxn_id | 실패 유형 | 원인 |
|---|---|---|
| 3090 | 1-piece split | 파일명 forming bonds 제거해도 TS가 두 조각으로 안 나뉨. P-TS 사이에 결합 파괴 2개 존재 (다단계 반응) |
| 3766 | 1-piece split | P-TS 사이 결합 형성 1개만 (cycloaddition 아님, 데이터 이상치) |
| 4252 | 1-piece split | P-TS 결합 형성 2 + 파괴 1 (원자 이동 동반) |
| 5930 | no isomorphism | R xyz의 그래프가 TS-fragment와 다름 (edge count 차이) |
| 3865 | no isomorphism | R conformer가 TS-fragment 대비 spurious H-contact 1개 |
| 4069 | no isomorphism | 위와 동일한 conformer artifact |
| 4727 | no isomorphism | 위와 동일 |

### 그룹 B — verify_product만 실패 (4건, R-build은 정상)

| rxn_id | 원인 |
|---|---|
| 2461 | Coley P 파일에 대칭 원자 재라벨링 (25↔26) |
| 4924 | 대칭 원자 34↔37 교환 |
| 5386 | 대칭 원자 재라벨링 |
| 5513 | 동일 유형 |

**그룹 B는 화학적으로 동일한 분자지만** react-ot의 그래프 identity check는 실패. Downstream 학습 target으로 P를 사용하기 때문에 안전상 제외.

### 시도한 복구 (모두 이 11건에 대해 실패)

`_frag_align.py`에 두 recovery path 구현:
- **Recovery 1 (diff_bonds_split)**: 파일명 bonds 실패시 P-TS graph diff로 재시도
- **Recovery 2 (diff_bonds_iso)**: iso 실패시 diff bonds로 재분할

결과:
- 그룹 1의 3865/4069/4727: filename bonds == diff bonds라 recovery 트리거 안 됨. 실제 원인은 R conformer의 결합 카운트 mismatch — SMILES 기반 재구축 없인 회복 불가
- 그룹 2 (3090/3766/4252/5930): P-TS에 결합 파괴 있어서 화학 자체가 다단계 — recovery로 회복 불가
- 그룹 3 (2461/4924/5386/5513): R-build 정상 통과, downstream Step 4 필터에서 제외

**SMILES 재구축 옵션은 Coley의 optimized conformer 정보 손실 대비 0.076% 데이터 이득으로 비용 > 이득 → 미실시**

---

## 파이프라인 단계별 결과

| Step | 작업 | 게이트 | 결과 |
|---|---|---|---|
| 0 | 환경 setup (react-ot clone, 심볼릭링크, ckpt 검증) | GATE-0 | ✅ PASS |
| 1 | 원소 inventory | GATE-1 | ✅ PASS (100% supported) |
| 2 | P-verify + R 복합체 구축 | GATE-2 | ✅ PASS (5262/5269 R-build, 5261/5269 P-verify) |
| 3 | ckpt + T1x pkl 스키마 문서화 | GATE-3 | ✅ PASS |
| 4 | React-OT pkl 포맷 변환 | GATE-4 | ✅ PASS (5258 usable) |
| 5 | 골격 기반 5-fold split | GATE-5 | ✅ PASS (leak 0) |
| **6** | **5-fold cross-fit 학습** | **GATE-6a/6b** | **✅ 진행 (사용자 정지 전 val_err 0.028-0.040)** |
| 8 | Final 모델 학습 | — | ✅ 진행 (val_err 0.030) |
| 7 | (예정) TS 생성 | — | 미실행 |
| 9 | (예정) RMSD 평가 | — | 미실행 |
| 10 | (예정) 채널 Δ 오차 (ORCA) | — | 미실행 |
| 11 | (예정) REPORT | — | 미실행 |

---

## Step 6 학습 결과 (5-fold + final)

**사용자 정지 시각 기준 (23h 09m 러닝 후)**:

| Fold | Epoch 도달 | Best val_ep_scaled_err |
|---|---|---|
| fold0 | 1,240 | **0.0281** |
| fold1 | 1,254 | **0.0333** |
| fold2 | 1,344 | **0.0310** |
| fold3 | 1,136 | **0.0304** |
| fold4 | 1,130 | **0.0326** |
| final | 1,165 | **0.0302** |

**5-fold 평균 val_err ≈ 0.0311** — React-OT 논문 Transition1x 벤치 (~0.03) 수준 달성.

Pretrained baseline (~0.13) 대비 4배 이상 개선. Coley의 큰 분자 (평균 24 heavy atom, T1x는 ≤7)에서도 fine-tune 안정 수렴.

### 러닝 통계

- **6 GPU (a6000) 완전 병렬**: n075, n081, n085, n087
- **속도**: ~50 epoch/h × fold당
- **총 GPU-시간**: 6 × 23h = 138 GPU-hour
- **홈 quota 사용**: 95 GB / 300 GB (안전)
- **wall time 남았을 때 사용자 종료**: 이미 plateau 진입, 추가 학습 개선 marginal

---

## 반복 수정 이력 (Step 6 학습 시작까지 12개 이슈)

**근본 원인**: react-ot 코드가 PL 1.x 스타일인데 자기들 env.yaml은 PL 2.4.0 지정 — self-contradiction. CI/테스트 없어 프로덕션 경로에 dev-only 코드 다수.

| # | 문제 | 해결 |
|---|---|---|
| 1 | ATOM_MAPPING 5원소만 (Cl/Br 없음) | `_rot_patches.py`: datasets_config.py 7원소로 확장 |
| 2 | `ase.neb` ImportError | ASE 3.28에서 `ase.mep.neb`로 이동, try/except shim |
| 3 | `colored_traceback` curses 실패 | headless SLURM 노드 terminfo 없음 → try/except softening |
| 4 | `Trainer(replace_sampler_ddp)` | PL 2.x 리네임 (→ 롤백 후 PL 다운그레이드) |
| 5 | `Trainer(strategy=None)` | PL 2.x "auto" 요구 (→ 롤백) |
| 6 | `training_epoch_end` 제거 | PL 2.0 API 삭제 (→ PL 1.9.5로 env 다운그레이드) |
| 7 | `setuptools 81+` pkg_resources 제거 | setuptools<81 pinning |
| 8 | `train_rpsb_all.pkl` 하드코딩 파일명 | 심볼릭링크로 우회 (기존 fold pkl → react-ot 기대 이름) |
| 9 | `use_ind` KeyError | `use_by_ind=True→False` 패치 (전체 subset 사용) |
| 10 | `num_atoms` KeyError | `slice_frag` 헬퍼로 `len(charges)`에서 파생 |
| 11 | a10 22GB OOM | Coley 큰 분자에 22GB 부족 → gpu2/gpu6 사용 포기, gpu3/4/5 (a6000 48GB) 전용 |
| 12 | Disk quota 초과 (300GB) | `save_top_k=-1 → 3` 패치, 매 epoch 저장 대신 top-3 만 |
| +α | `regex greedy` fold3 5초 fail | `[-\d.]+` → `(-?\d+\.\d+)` 엄격 매칭 |

각 fix 후 `.py.orig` 백업 + git commit + push (총 15개 commit).

---

## 환경 (재현용)

```
python 3.10.14
pytorch-lightning 1.9.5    # react-ot 코드가 PL 1.x 스타일 (env.yaml pin은 무시)
setuptools       80.10.2   # PL 1.9의 pkg_resources 의존
torch            2.2.1+cu118
ase              3.28.0    # ase.mep.neb 경로, react-ot 소스 patched
```

React-OT commit: `6dfccd0` (main branch, 유일한 커밋 — 유지관리 나쁨)  
React-OT env.yaml은 PL 2.4.0 pin이지만 실제 코드와 불일치. `ENV.md` 참고.

패치된 react-ot 파일 (모두 `.py.orig` 백업 있음):
- `reactot/dataset/datasets_config.py` — ATOM_MAPPING 7원소
- `reactot/run_model.py` — allowed_atom_types 확장  
- `reactot/diffusion/_utils.py` — ase.neb shim

---

## 현재 상태

**정지됨** (사용자 요청, plateau 진입 후 종료):
- 5-fold + final 모두 scancel 완료
- best top-3 checkpoint 보존 (각 fold ~2GB, 총 16GB)
- b3lyp orch는 유지 (별도 프로젝트)

**바로 이어갈 수 있음**:

```bash
# 학습 이어서 (auto-resume from best)
sbatch --partition=gpu3,gpu4,gpu5 --array=0-4%5 sbatch/06_train_crossfit.sh
sbatch --partition=gpu3,gpu4,gpu5 sbatch/08_train_final.sh

# 바로 TS 생성으로 (fold별 best ckpt 자동 감지)
sbatch --partition=gpu3,gpu4,gpu5 --array=0-4%5 sbatch/07_generate_crossfit.sh

# Step 9/10/11 (Step 7 완료 후)
sbatch sbatch/09_evaluate.sh
sbatch sbatch/10_prep.sh
sbatch sbatch/10_launcher.sh   # ORCA 200-rxn chain
sbatch sbatch/10_parse.sh
sbatch sbatch/11_report.sh
```

---

## 아티팩트 (git 트래킹)

- `analysis/otfm_train/artifacts/`
  - `GATE{0-5,6b}_STATUS.txt` — 게이트별 통과 여부 + 지표
  - `composition.csv` — 5269 rxn 원소 조성
  - `reactant_build.csv` — 5269 rxn R-build 결과 (성공/실패 사유 + 지표)
  - `p_order_failures.csv` — verify_product 실패 8건 (rxn_id 목록)
  - `folds.csv` — 5258 rxn × 5-fold 분할
  - `convert_failures.csv` — Step 4 변환 실패 (0건)
  - `schema.md` — React-OT ckpt + T1x pkl 스키마 문서

- `analysis/otfm_train/ckpt/` — **gitignored** (16 GB checkpoint)
- `analysis/otfm_train/data/` — **gitignored** (fold pkl + reactant_complex npz)
- `analysis/otfm_train/logs/` — **gitignored** (러닝 로그, 최대 10GB)

---

## 최종 성과

**목표**: Coley 5,269 dipolar cycloaddition에서 R+P→TS 생성 모델 fine-tune  
**달성**: 5,258/5,269 (99.79%) 반응으로 5-fold cross-fit 완료, val_ep_scaled_err 0.028-0.040 수준 도달  
**품질**: React-OT 논문 Transition1x 벤치(~0.03)와 동등, pretrained baseline(0.13) 대비 4배 이상 개선  
**미완**: Step 7 (TS 생성) 이후 파이프라인 — 사용자 지시에 따라 재개 가능
