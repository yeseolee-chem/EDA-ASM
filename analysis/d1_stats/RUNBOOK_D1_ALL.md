# RUNBOOK — SPEC_D1_ALL 실행 순서 (docs/specs/SPEC_D1_ALL.md)

모든 계산은 계산 노드에서 `sbatch` 로 한다 (CLAUDE.md). 잡은 **저장소 루트에서** 제출한다 (스크립트가 `$SLURM_SUBMIT_DIR` 로 경로를 잡는다).
제출 전마다 `squeue -u $USER -h | wc -l` 로 대기열을 본다 (배열 원소도 한 칸씩, MaxSubmit 20 · MaxJobs 10). 모든 단계는 멱등이다: 48 h 벽에 걸리면 같은 명령을 다시 제출한다.

> **범위 (2026-10-14 사용자):** 지금 목표는 통계 판정뿐이다 — Phase R → (RELAY_DONE) 확정 export → A → B. **Phase C · D 는 보류**: 코드는 두고 제출하지 않는다. 게이트 C 분모 · D0 dev 겹침 처리 · PREREG 승인은 C/D 를 다시 열 때 정한다.
> RELAY_DONE 뒤에는 `bash analysis/d1_stats/jobs/submit_after_relay.sh` 한 번으로 export → A → B 를 afterok 사슬로 낸다 (로그인 노드에서 sbatch 세 번만).
> B2 판정은 d1_archive_scan.py v4 (= 사전 고정 v3 규칙): 극대 없음 + 반응물 쪽 끝 형성 결합 큰 거리 ≥ 3.0 Å 일 때만 barrierless, 미만이면 no_peak_short_scan. autodE 가 고개 없는 경로를 늘 2 × (생성물 결합) 에서 끝내므로 (검토 · 표본 68 경로 ratio 2.000) short_scan = 생성물 결합 < 1.5 Å — 사용자 결정 (2026-10-14): v3 를 주 판정으로 두고 '극대 없음 전체' 를 민감도로 함께 보고.
D1 본계산 relay 가 아직 돌고 있으면 그 잡들이 10칸 중 대부분을 쓴다 — Phase R 외에는 본계산이 끝난 뒤(RELAY_DONE)에 한다.

| 단계 | 명령 | 끝나면 | 멈춤 |
|---|---|---|---|
| S · R | `sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_r_%j.out analysis/d1_stats/jobs/phase_r.slurm` | `results_interim/GATE_R.md` | sha256 불일치, 게이트 R 불일치 |
| 확정 export | `sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/d1_final_export_%j.out analysis/d1_stats/jobs/d1_final_export.slurm` (RELAY_DONE 뒤) | `D1_build/d1_run/results/` | |
| A | `sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_a_%j.out analysis/d1_stats/jobs/phase_a.slurm` | `results_final/INPUTS.md · qc.json · gate_a.json` | 게이트 A |
| B | `sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_b_%j.out analysis/d1_stats/jobs/phase_b.slurm` | `results_final/*` + `STATS_REPORT.md` | (게이트 B 는 목록 보고만) |
| C 묶음 1 (11칸) | `J=$(sbatch --parsable -p cpu1,cpu2 -J d1sC_prep -o analysis/d1_stats/logs/c_prep_%j.out analysis/d1_stats/jobs/phase_c_step.slurm prep)` 다음 `sbatch -p cpu1,cpu2 --dependency=afterok:$J --array=0-8 --cpus-per-task=16 --mem=32G -o analysis/d1_stats/logs/c_g1_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm g1` 그리고 `sbatch -p cpu1,cpu2 -J d1sC_ctl -o analysis/d1_stats/logs/c_ctl_%j.out analysis/d1_stats/jobs/phase_c_step.slurm control` | `d1_features/g1/<rid>/` · `results_final/gate_c_control.json` | control rc 1 = 게이트 C (iii) |
| C 묶음 2 (10칸) | `J=$(sbatch --parsable -p cpu1,cpu2 --array=0-8 --cpus-per-task=2 --mem=8G -o analysis/d1_stats/logs/c_xtb_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm xtb)` 다음 `sbatch -p cpu1,cpu2 --dependency=afterok:$J -J d1sC_agg -o analysis/d1_stats/logs/c_agg_%j.out analysis/d1_stats/jobs/phase_c_step.slurm agg` | `d1_features/xtb/xtb_features_g1.parquet` | |
| C probe (1칸) | `sbatch -p cpu1,cpu2 --cpus-per-task=2 --mem=8G -J d1sC_probe -o analysis/d1_stats/logs/c_probe_%j.out analysis/d1_stats/jobs/phase_c_step.slurm probe` (agg 뒤) | S 행 · Si 행의 ext_status (`d1_features/logs/probe_<rid>.json`) | rc 1 = D0 에 없던 원소에서 블록 실패 — ext 배열을 내기 전에 본다 |
| C 묶음 3 (10칸) | `J=$(sbatch --parsable -p cpu1,cpu2 --array=0-8 --cpus-per-task=16 --mem=32G -o analysis/d1_stats/logs/c_ext_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm ext)` 다음 `sbatch -p cpu1,cpu2 --dependency=afterok:$J -J d1sC_merge -o analysis/d1_stats/logs/c_merge_%j.out analysis/d1_stats/jobs/phase_c_step.slurm merge` | `xtb_features_d1_g1.parquet` · `xtb_features_ext_d1_g1.parquet` · `results_final/GATE_C.md` | 게이트 C |
| B5 다시 | `FORCE=1` 없이 `.done_b5_gedt` 만 지우고 phase_b.slurm 다시 (xTB GEDT 열 채움) | `gedt_d1.csv` | |
| D | PREREG_D1.md 승인 · commit 뒤: `phase_d.slurm check` → `m0` → `split` (배열 0-8) → `report` → `m4` → `report` | `MODEL_REPORT.md` | check · m0 재현 실패, M4 악화 |

- Phase C 의 D1 뿌리: `/gpfs/tmp_cpu2/yeseo1ee/d1_features/` (`prof/ g1/ xtb/ r5/ logs/`). D0 의 rev 5 폴더는 읽기만 한다 (control 단계만 D0 환경으로 돈다; 쓰는 곳은 `results_final/control/`).
- σ 표 `analysis/d1_stats/sigma_hansch1991.csv` (2026-10-14 사용자 제공, sha256 `f042f1335d5dea88217f0857a63baa5aded3e342ec61e3a86ea7ccd2d0f20def`, CRLF 그대로 — 내용 수정 금지, `.gitattributes` 로 줄 끝 변환 막음). phase_a.slurm (`run_all.py --sigma` → B3 hammett · sigma_vs_rank2) 과 phase_b.slurm (B6 inflection, B5 σ 비대칭) 이 자동으로 읽는다. Phase R 은 σ 를 쓰지 않는다. Ts 는 Hansch 1991 Table I 에 없어 빈칸 — σ 분석에서 Ts 가 빠지는 것은 의도된 것 (대체값 없음).
- 결과 중 커밋하는 것: `results_interim/`, `results_final/` (parquet 제외), `STATS_REPORT.md`, `PREREG_D1.md`, `MODEL_REPORT.md`, `results_models/`. feature parquet 은 scratch 에 두고 `results_final/FEATURES_SHA256.txt` 만 커밋한다.
- 게이트 C (ii) 의 분모는 SPEC 문구대로 D1 라벨 전체다. D0 rev 4 의 G1 성공률이 92.4 % 였으므로 G1 실패만으로 2 % 를 넘어 STOP 할 가능성이 높다 — `GATE_C.md` 에 G1 성공 행 기준 비율도 함께 적어 두었다. 그때 어느 분모로 판정할지는 사용자가 정한다.
- 게이트 C (iii): C0020 (D0 rxn 20) 은 D0 에서 G1 이 실패한 반응이라 상태 대조만 된다. 수치 대조는 C0105 한 건이고, rxn 2434 를 참고용으로 같은 방식으로 비교한다 (판정에 넣지 않음).
- ESPLEY73 의 `dsasa_*` 는 H · C · N · O · F · Cl · Br 만 있다 (rev 4 정의 그대로). D1 에 새로 생긴 S · Si 원자의 표면적 변화는 feature 에 들어가지 않는다 — Methods 에 적는다.
- Phase D 는 `PREREG_D1.md` 에 승인 줄을 넣고 commit 한 뒤에만 돈다. D1 행 중 D0 lockbox 와 같은 반응은 M4 학습에서 빠진다 (`d1_models.py` 가 정준 반응 SMILES 로 찾는다).
- 그림: Phase R · A 의 `figs.py` 가 B1 그림 4장 (`fig1–fig4`), Phase B 끝의 `figs_b.py` 가 판정 그림 `fig05–fig19` 를 `results_final/fig/` 에 그린다 (B1b E_int · 저랭크 CI, B2 결과 분류 · 결측 원인 · d_react 분포 · logit, B3 Hammett, B4 강건성 · 지문 FDR, B5 Houk–Domingo · regio 크기 · GEDT · σ 비대칭, B6 변곡, B7 메틸, B8 carbonyl ylide). 입력이 없는 그림 (σ 표가 필요한 B3 · B6 · B5 σ 비대칭) 은 건너뛰고 `fig/figs_b_index.json` 에 적는다. `STATS_REPORT.md` 가 그림을 절마다 넣는다.
- 표본 점검 (2026-10-14, 잡 1009123): `checks/b2_v3_spotcheck*` — v3 판정을 no_ts 34건에 실제 코드로 돌린 결과. 파일럿 T02894 = barrierless 3.037 Å (기대값 그대로), 읽기 실패 0. 고개 없는 경로의 d_react 는 거의 3.00–3.12 Å 에 몰려 있어 (autodE 경로 끝이 약 3.0 Å) 3.0 Å 기준 근처 판정이 민감하다 — B2 결론에 함께 적는다.
