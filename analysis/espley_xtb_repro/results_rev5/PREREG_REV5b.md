# PREREG_REV5b — EXT_SEL 고정 (Phase D 전, 2026-10-01)

명세 `docs/specs/REV5_FEATURES_FIGURES.md` C-3. PREREG_REV5a(commit `358b2d10`)의 규칙으로 dev 행(4,112)에서만 선별했다
(잡 997696). 이 문서와 `prereg_rev5b.json`을 commit 한 뒤에 Phase D를 시작하고, 이후 EXT_SEL은 바꾸지 않는다.

## EXT_SEL

- **블록: B1(궤도 overlap) + B4(반응성 지수·응답).** 채택 순서 B4 → B1.
- **feature 수: 133** = BASE(ESPLEY73) 73 + B1 21 + B4 39. arm 이름 `BASE+B1+B4`(`rev5_common.arm_columns("EXT_SEL", …)`).
- 선별 기준(elst·Pauli·OI의 바깥 fold test NMAE 평균, 5 fold 평균): BASE 0.29509 → +B4 0.27456(상대 −6.95 %, 5/5 fold)
  → +B1 0.26770(상대 −2.50 %, 5/5 fold). 3단계 최선 후보 +B6은 상대 −1.15 %(< 2 %)로 탈락, 선택 종료.
- 경로 전체: `C2_selection_path.csv` / `.json`; 블록별 9 타깃 dev CV: `C2_block_cv.csv`, fold별 `C2_block_cv_folds.csv`;
  feature–타깃 상관(dev): `C2_feature_target_r.csv`.

## 고정되는 것

- Phase D의 arm은 BASE(ESPLEY73), **EXT_SEL = BASE+B1+B4**, EXT_ALL(BASE+B1…B6, 사전 고정)이다.
- 입력 파일: G1 feature sha256 `64ee5da7…`, 확장 feature sha256 `35d1a84f…`, dev `19a8e212…`, lockbox `97c05514…`
  (값 전체는 `prereg_rev5b.json`과 PREREG_REV5a).
- 이후 절차는 PREREG_REV5a §4–§5 그대로다. D 스크립트는 lockbox를 처음 쓰기 전에 선별 캐시에 lockbox id가 없음을 검사한다.
