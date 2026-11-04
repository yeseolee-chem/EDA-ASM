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
- **0-7 env job**: gpu3·4·5만 지정한 1061580이 대기(Priority)라 시작 전에 취소하고 모든 GPU partition(gpu1–6)으로 다시 냈다. env 확인 목적상 GPU 종류는 상관없다.
- **0-5 예비 확인 (11-04 12:12, tests/ext_probe.py, sandbox)**: ORCA 6.1.1 `! ExtOpt EnGrad` + `%method ProgExt "<script>" Ext_Params "..." end`가 동작한다. ORCA는 `<base>_EXT.xyz`와 `<base>_EXT.extinp.tmp`(xyz 파일명 · 전하 · 다중도 · NCores · do-gradient, 줄마다 '# 설명' 주석)를 쓰고 `<script> <base>_EXT.extinp.tmp <Ext_Params>`를 부른 뒤 `<base>_EXT.engrad`(ORCA engrad 형식)를 읽는다. 시험 계산은 ORCA TERMINATED NORMALLY. 다리 · 최적화기 · Hessian 검사(A-1)는 C-1에서 한다.
- **0-7 env 1차 실패 → 추천안 (11-04 12:4x)**: 1061590에서 clone은 됐지만 env_check가 실패했다(pass_0_7 false). 원인: sella가 끌어온 jax 0.6.2가 numpy를 2.2.6으로 올려 matscipy(MACE) · tblite가 깨졌고, aimnet은 torch ≥ 2.5(orca-mlips의 mace · aimnet2 extra는 ≥ 2.8)를 요구해 torch 2.2.1 고정과 충돌했다. 추천안: reactot은 그대로 두고 mlts env 안에서만 torch 2.8.0(cu128) + numpy 1.26.4(reactot과 같음; aimnet이 < 2 요구)로 고정해 aimnet · sella · orca-mlips[mace,aimnet2]를 다시 설치한다(jobs/env_build.slurm 2단계). UMA-S는 HuggingFace 토큰이 없어 설치하지 않는다(명령서 4절 "없으면 뺀다").
- **A-1 도구**: orca-mlips 1.3.0(모듈 `orca_mlips_plugins`)에 서버 방식(`mlip_server.py`), MACE `--weights-file`, 해석적 Hessian, ORCA ext 입출력이 있다. 명령서대로 우리 모델을 올리면 이것을 쓰고, 안 되면 `oet_mlts.py`를 만든다(C-1에서 확인).
- **Phase 0 결과 (check_start.py run --free-gb 102, sandbox)**: 0-1 · 0-2(210/210) · 0-3(8,616) · 0-4 · 0-6 통과. 0-6: 원소 {H, C, N, O, F, Si, S, Cl, Br}(Si 465반응), 9,266반응 모두 전하 0/0 · 짝수 전자 · ORCA 입력 `* xyz 0 1`. 0-5는 C-1로(ExtOpt 예비 확인 통과), 0-7은 env 재설치 중, 0-8은 위 편차(그 사이 clone으로 여유 102 GB).
- **Phase I 결과 (inv_dft.py scan, sandbox)**: 후보 2,947파일 중 라벨 수준 1,034파일, 18,958구조(`$MLT/dft/existing.extxyz`): G3L ts_path 486 + ts_final 20, Part B 1,805 + 39, V 반응물 conformer react_opt 9,735(130분자), D0 검증(d1_validation) 5,931. 파서: 단계 수 = 'GEOMETRY OPTIMIZATION CYCLE' 수(1,034파일 모두), `_trj.xyz`가 있는 409파일 중 407 일치(2개는 비정상 종료), 수렴 TS의 최대 기울기 중앙값 3.9e-5 Eh/bohr. **I-1 실패**(라벨 TS 형성 결합 차 ≤ 0.3 Å인 구조 1,771개, 60반응 < 50,000 · 5,000) → G-2를 한다.
  - 판단(agent 제안, 추천안으로 채택): (1) D0 검증 자료는 D1 반응이 아니므로 표지 ''이지만, 같은 반응이 D1 대상인 2건(S0 → 1000010, V6a_08 → 1002155, 81구조)은 그 반응의 표지를, D1 반응물과 같은 분자의 D0 반응물 최적화(369구조)는 그 분자를 쓰는 D1 반응 전체의 표지를 단다(누수 방지 쪽). (2) 비정상 종료 6파일(1,289단계)은 넣지 않는다. (3) 기울기 > 0.1 Eh/bohr인 58구조는 남기고 FT에서 상한을 정한다.
  - 명령서 2절의 "`_trj.xyz`와 단계 수 일치"는 G3 · V 출력에 `_trj.xyz`가 없어 "파싱된 단계 수 = 최적화 cycle 수"로 바꿨다(`_trj.xyz`가 있는 D0 파일은 둘 다 확인).
  - agent가 `mlt_common.write_extxyz`의 버그(빈 문자열 표지가 ase에서 다음 키와 합쳐짐)를 고쳤다: 표지 3개를 JSON 목록으로 쓰고 `read_extxyz`로 되돌린다.
  - 규칙 기록: agent가 login에서 읽기 전용 for 루프 약 5번, `ls -R | wc -l` 1번, `find -maxdepth` 2번, `mmlsquota` 1번.
