# SPEC18 `otfm_guess` — xTB path guess + React-OT GUESS-mode fine-tune

**목적**: `x1 = TS_xTB → OT → x0 = TS_DFT`. React-OT의 GUESS 모드로 xTB 경로 탐색
결과를 초기 추정으로 사용해 잔차만 학습. spec17rev2 (`RP` 모드, x1=(R+P)/2)의
후속 실험이며 그와 완전 분리된 폴더.

의존:
- `analysis/otfm_train/data/coley_all.pkl`, `artifacts/folds.csv`,
  `reactant_build.csv`, `generation_quality.csv`
- `analysis/otfm_train/data/reactant_complex/rxn_XXXX.npz` (R, P TS-order)
- `analysis/otfm_train/ckpt/fold{0..4}` (RP-mode best ckpt — 옵션 A/B용)
- `analysis/otfm_train/coley_profiles/full_dataset_profiles/{rid}/` (Coley TS 파일명)

---

## 단계별 요약

| Step | 스크립트 | sbatch | 의도 |
|---|---|---|---|
| 0 | `00_setup.py` | `sbatch/00_setup.sh` | xtb 확인, 이전 산출물 존재 검증 |
| 1 | `01_select_pilot.py` | `sbatch/01_select_pilot.sh` | fold별 40 rxn 층화 표본 (총 200) |
| 2 | `02_xtb_path.py` (`--shard/--nshard`) | `sbatch/02_xtb_path.sh` (array) | xTB `--path` 실행, `ts_xtb.npy` 저장 |
| 2b | `02b_merge_shards.py` | `sbatch/02b_merge.sh` | shard CSV 병합 + GATE-2 판정 |
| 3 | `03_eval_xtb_ts.py` | `sbatch/03_eval_xtb_ts.sh` | xTB TS vs Coley DFT, OTFM-RP와 비교 |
| 4 | `04_build_guess_pkl.py` | `sbatch/04_build_guess_pkl.sh` | `coley_all_guess.pkl` + fold별 pkl 생성 |
| 5 | `05_train_guess.py` | `sbatch/05_train_guess.sh` (array) | GUESS 모드 5-fold 학습 |
| 6 | `06_generate_guess.py` | `sbatch/06_generate_guess.sh` (array) | 각 fold test에 대해 TS 생성 |
| 7 | `07_compare.py` | `sbatch/07_compare.sh` | xTB / RP / GUESS 3자 비교 |
| 8 | `08_channel_impact.py` | `sbatch/08_channel_impact_*.sh` | ORCA EDA로 채널 δ_lin 측정 |

**GATE-3에서 branch**: xTB 단독이 이미 RP보다 나으면 GUESS 필요성 재검토.

---

## 실행 (권장 순서)

```bash
cd analysis/otfm_guess

# 1) 환경 확인
sbatch sbatch/00_setup.sh

# 2) 파일럿 표본
sbatch sbatch/01_select_pilot.sh

# 3) xTB 파일럿 (200 rxns, 10 shard)
XTB_MODE=pilot sbatch --array=0-9%10 sbatch/02_xtb_path.sh
# 완료 후:
XTB_MODE=pilot sbatch sbatch/02b_merge.sh

# 4) 파일럿 품질 평가 → GATE-3 판정
sbatch sbatch/03_eval_xtb_ts.sh

# ─── GATE-3 통과 후에만 진행 ───

# 5) 전체 5,261 xTB 실행 (20 shard)
XTB_MODE=full sbatch --array=0-19%20 sbatch/02_xtb_path.sh
XTB_MODE=full sbatch sbatch/02b_merge.sh

# 6) pkl 생성 + fold 분할
sbatch sbatch/04_build_guess_pkl.sh

# 7) GUESS 모드 학습
sbatch --partition=gpu3,gpu4,gpu5 \
       --export=ALL,SAMPLER_MAX_NUM=8000,BATCH_SIZE=8,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       --array=0-4%5 sbatch/05_train_guess.sh

# 8) 생성 + 비교
sbatch --partition=gpu3,gpu4,gpu5 --array=0-4%5 sbatch/06_generate_guess.sh
sbatch sbatch/07_compare.sh

# 9) 채널 영향
sbatch sbatch/08_channel_impact_sample.sh
sbatch sbatch/08_channel_impact_inputs.sh
# ORCA 실행 (spec17rev2 launcher와 동일한 방식으로 array 제출)
sbatch sbatch/08_channel_impact_parse.sh
```

---

## 실행 규칙 (SPEC §6 정리)

- **번호 순.** 게이트 실패 시 다음 단계 금지.
- **`path.inp`는 반응별로 바꾸지 말 것.** 전부 같은 설정 (SPEC §3).
- **R, P는 반드시 `reactant_complex/*.npz`에서 읽는다.**
  Coley 원본 `r0/r1/p0.xyz` 직접 넣으면 원자 순서가 어긋난다.
- **`order_mismatch`가 하나라도 나오면 STOP** — 파일 쓰기 버그.
- **분할은 spec17rev2 `folds.csv` 그대로.**
- **React-OT 모델 코드 수정 금지.** 바꾸는 것은 `mapping_initial`, `ts_guess`
  두 줄 설정과 pkl 키뿐.
- xtb 실행은 idempotent — `ts_xtb.npy` 있으면 건너뜀.
- 모든 sbatch: `--time=48:00:00`, `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."`.

---

## 산출물 규격 (예상)

```
analysis/otfm_guess/
├── 00_setup.py  01_select_pilot.py  02_xtb_path.py  02b_merge_shards.py
├── 03_eval_xtb_ts.py  04_build_guess_pkl.py
├── 05_train_guess.py  06_generate_guess.py
├── 07_compare.py  08_channel_impact.py
├── sbatch/                     # 각 단계별 sbatch
├── xtb_runs/rxn_NNNN/
│   ├── R.xyz  P.xyz  path.inp  xtb.log
│   ├── xtbpath_ts.xyz          # xtb 원본 출력
│   └── ts_xtb.npy              # ★ 성공 시
├── data/
│   └── coley_all_guess.pkl     # ★ ts_guess_xtbpath 키 포함
├── ckpt/guess_fold{0..4}/data/{train,val,test}_fold{k}.pkl
├── ckpt/guess_fold{0..4}/checkpoint/RPSB-FT-Schedule/leftnet-*/sb-*.ckpt
├── generated/guess/rxn_NNNN.xyz
├── channel_impact/
│   ├── sampled_ids.csv         # spec17rev2 200개 그대로
│   ├── inputs/rxn_NNNN/*.inp
│   └── channel_impact_guess.csv
└── artifacts/
    ├── pilot_sample.csv
    ├── xtb_path_pilot.csv  xtb_path_full.csv
    ├── xtb_ts_quality_pilot.csv
    ├── compare_three.csv       # ★ xTB / RP / GUESS 3자
    └── GATE{0..8}_STATUS.txt
```

---

## 성공 기준 (SPEC §9)

- [ ] GATE-0: xtb 실행 가능, 이전 산출물 존재
- [ ] GATE-1: 파일럿 200 (fold별 40)
- [ ] GATE-2: `xtb --path` 성공률 ≥ 80%, `order_mismatch=0`
- [ ] GATE-3: xTB TS 형성 결합 `|Δ| < 0.145 Å` — **진행 여부 결정**
- [ ] GATE-4: pkl에 `ts_guess_xtbpath` 키, 대체 비율 기록
- [ ] GATE-5: 5 fold 수렴, `# of data` 일치
- [ ] GATE-6: 5,261 생성, 누출 assert 통과
- [ ] GATE-7: GUESS `|Δ|` < 0.145, std < 0.26, GUESS > xTB 단독
- [ ] GATE-8: 채널 δ_lin 측정 → 최종 판정
