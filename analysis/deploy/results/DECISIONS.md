# DEPLOY-MODEL 결정 기록 (docs/specs/DEPLOY_MODEL.md, 이어서 DEPLOY_MODEL_W.md)

- **0-user (2026-11-03)**: 사용자가 DEPLOY_MODEL.md와 DEPLOY_MODEL_W.md를 주며 "이대로 진행. 모든 분기마다 추천 분기로 진행. claude.md 준수"라고 했다. DEPLOY_MODEL을 먼저 끝내고 W를 이어서 한다. 명령서의 STOP 조건에서는 원인을 기록하고 추천안으로 진행하되, 삭제와 돌고 있는 job 취소는 먼저 묻는다.
- **0-setup (11-03 13:3x)**: `git worktree add .../eda-asm-prediction-deploy -b deploy origin/geo2b`(0314ad96)와 `$DEP` 폴더(logs, g2lf_final, F, fu2_lf, units, model, smoke, prof_empty, w)를 job 1059737 안에서 만들었다. 두 명령서를 docs/specs/에 그대로 저장하고 commit했다(4b547f39). `$DEP/prof_empty`는 빈 폴더다.
- **0-start (1절)**: `check_start.py`(job 1059760, 다시 sandbox 1059761 안에서) 0-1..0-8 모두 통과. 0-2: D1 대상 9,266 작업 폴더 모두 있음, ok 8,616은 필요한 파일 4개를 모두 가짐. 0-4: F_pairs_cr2.csv sha256 = L__g2__S__TF__f0 지문. 0-5: T1 분할 5개 sha256 = S_splits.json. 0-7: manifest sha256 = g2_common.MANIFEST_SHA. 0-8: 계산 노드에서는 mmlsquota가 돌지 않아 login에서 읽은 값(183/300 GB, 여유 117 GB ≥ 30)을 인자로 넘겼다.
- **0-sandbox**: 코드 검사용 CPU job `dep_sandbox` 1059761(8코어, 48 h)을 띄우고, 짧은 검사는 그 안에서 `srun --overlap … timeout ≤ 560`으로 돌린다. login에서는 python을 돌리지 않는다.
- **0-PREREG 확인**: 17절 체인의 "(사용자 PREREG 확인)"은 사용자 지시(모든 분기 추천안)에 따라 기본 기준값(G-A 2 %, G-C +5 %)으로 확정하고 commit·push한 뒤 K를 낸다. 사용자가 기준값을 바꾸려면 K 전에 말하면 된다.
- **0-code**: 코드는 세 갈래로 나눠 작성한다(L·M / F·T·Q·C-1 / data·train·eval·predict·tests·prereg). `dep_common.py`는 내가 썼다.
