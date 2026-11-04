# DEPLOY-V2 (MLTS) 결정 기록

명령서 `docs/specs/DEPLOY_V2_MLTS.md`(commit e7f87ce6). 사용자 지시(2026-11-04): "모든 분기마다 추천 옵션으로 자동 진행. claude.md 준수. md파일대로 진행".
STOP 조건에서도 추천안을 골라 여기 기록하고 진행한다. 삭제와 돌고 있는 job 취소만 먼저 묻는다.

- **0-base (11-04 11:5x)**: 명령서는 "`origin/deploy` HEAD(`2f9ffa3`)에서" 만들라고 하고, 시작 조건 0-1은 "`2f9ffa3` 이후"를 요구한다. 지금 origin/deploy HEAD는 fc890aba(2f9ffa3 뒤에 I2 결과와 최종 요약 commit 하나가 더 있음)다. 추천안: HEAD fc890aba에서 `git worktree add .../eda-asm-prediction-mlts -b mlts origin/deploy`(2f9ffa3 포함 확인). 명령서를 docs/specs/에 그대로 저장하고 commit했다(e7f87ce6).
- **0-8 디스크 (STOP → 추천안)**: login `mmlsquota` 191/300 GB, 여유 109 GB < 200 GB. 명령서 1절 시작 조건 실패다. 추천안: 멈추지 않고 진행하되 디스크를 작게 쓴다.
  - ORCA는 `.out`만 gzip해서 두고 `.gbw` · 밀도 · tmp는 지운다. 학습 자료는 extxyz로 모은다.
  - G · FT · Z · A-3는 각각 수 GB 이하로 예상된다. 가장 큰 A-4(9,266반응 autodE-MLTS 사본)는 A-3에서 반응당 디스크를 재고, 필요량 + 30 GB 여유가 없으면 A-4 전에 사용자에게 지울 대상을 묻는다.
  - 제출 전마다 login `mmlsquota` 여유가 30 GB 아래이면 새 계산을 내지 않는다.
- **0-folders**: `$MLT/{logs,data,dft,ft,models,ade,final,inputs,units,preds,tmp}`를 job 1061567(sbatch --wrap mkdir)로 만들었다.
- **규칙 기록 (11-04 12:0x, 나)**: GPU 종류를 보려고 login에서 `sinfo`를 한 번 실행했다(읽기 전용; CLAUDE.md는 권하지만 이전 프로젝트들의 login 금지 목록에 있었다). 이후에는 `sinfo`를 쓰지 않고 `-p`에 여러 partition을 넣어 SLURM이 고르게 한다. 확인한 GPU: gpu1 RTX 3090, gpu2 · gpu6 A10, gpu3 · gpu4 · gpu5 A6000 계열.
