# D1-G2R R_diagnosis — 기준 구조 단계 코드 오류 20개 (spec 5-1)

계산 없이 `$G2/jobs/<job_id>/ref_result.json`(traceback), `.fail_ref`(이유 문장), `ts_result.json`, D1 run 코드
(`/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/d1_autode_pilot/`), `analysis/d1_g2/g2_autode.py`,
D1 venv의 rdkit 2024.3.2 `rdkit/Chem/AllChem.py`를 읽어 정리했다. 줄 번호는 이 파일들 기준이다.

표지 이름: 작업 폴더에는 단계 표지만 있다(`.done_ts`, `.done_stereo`, `.fail_ref`; 이유는 `.fail_ref` 안의 문장).
`.fail_ref_AtomValenceException` / `.fail_ref_KeyError`라는 이름은 `$G2/g2_final/<rxn_id>/`와 `G_status.csv`의
`marker`에만 있다. 그래서 5-4-1의 "`.fail_ref_*` 표지를 지운다"는 작업 폴더의 `.fail_ref` 파일을 지우는 것으로 했다
(기록된 결정; 원래 문장은 `$G2R/x/R/<job_id>.pre.json`과 C_manifest에 남는다).

## 원인 묶음 A — RDKit `AtomValenceException` (19개)

- 대상: T09757, T09797, T09798, T09822, T09837, T09838, T09877, T09878, T09902, T09917, T09920, T10038, T10039,
  T10117, T10157 (azomethine ylide AY6), T11023, T11058, T11070, T11203 (carbonyl ylide CY11). 모두 stereo = match,
  TS·입체 단계 통과(`.done_ts`, `.done_stereo`).
- 이유 문장(19개 같음): `AtomValenceException: Explicit valence for atom # 0 C, 5, is greater than permitted`.
- 예외가 난 곳(19개 traceback 같음):
  `g2_autode.py:186 stage_ref` → `make_reference.py:244 coley_port`
  `alt_cfg = su.config_tags(su.xyz_block_from_atoms(a_syms, a_xyz), dip_smi, bb, q)[0]` →
  `stereo_utils.py:153 config_tags` → `stereo_utils.py:60 template_match`
  `m = AllChem.AssignBondOrdersFromTemplate(tmpl, geo)` → `AllChem.py:434 SanitizeMol(mol2)`.
- 입력: Coley 단계 4–6에서 만든 **alt 쌍극자 conformer**(xtb 최적화 뒤, 반응물 쌍극자 원자 순서). 원자 0 = 일라이드
  CH2⁻ 탄소(map 1). 네 번의 xtb 호출(orig opt+ohess, alt opt+ohess)은 이미 끝난 뒤였다(`xtb_calls` 4개).
- 왜: 쌍극자 아릴 고리의 ortho 받개(o-Ac, o-NO2, o-CO2Me)와 쌍극자의 음이온 끝(CH2⁻)이 같은 쪽에 있다
  (`ts_dipole_config` = 'c'). RDKit은 결합을 거리로 정한다(`stereo_utils.py:43`
  `rdDetermineBonds.DetermineConnectivity`). 이 conformer에서 CH2⁻ 탄소가 받개(카보닐 C 또는 니트로 N)에 가까워져
  결합 하나가 더 생긴다. `AssignBondOrdersFromTemplate`는 템플릿을 부분 구조로 맞추므로(`AllChem.py:414`,
  `uniquify=False`, 여분 결합 허용) 템플릿의 결합 차수와 형식 전하(CH2⁻의 −1)를 그대로 옮긴 뒤(`AllChem.py:421-433`)
  `SanitizeMol`에서 원자가 검사로 거부한다: 템플릿 결합 3개 + 여분 1개 + 전하 −1 → 유효 원자가 5.
- 실패한 alt 구조는 남아 있지 않다(`work/dipole_alt.xyz`는 노드 로컬 `jd/ref`에 있었고 `keep_small`은 최상위 파일만
  복사). 여분 결합의 상대는 위 구조에서 추론한 것이다. 이번 실행의 wrapper가 여분 결합(원소, map 번호, 거리)을
  기록한다(`results/R_status.csv` `extra_bonds`).
- alt_cfg(244행)는 기록만 하고 Coley 판정(249–263행)에는 쓰지 않는다. 판정 전에 예외가 나서 단계 전체가 실패했다.

## 원인 묶음 B — `KeyError: 'dipole_mapped_smiles'` (1개, T10518)

- T10518 (nitrone NI8-B_o-Ac). 이유 문장 `KeyError: 'dipole_mapped_smiles'`, `wall_s` 0.00058 s, xtb 호출 없음.
- 예외가 난 곳: `g2_autode.py:186` → `make_reference.py:158`
  `dip_smi, bb = ts_res["dipole_mapped_smiles"], tuple(ts_res["dipole_backbone_maps"])`.
- 왜: TS 단계(`run_ts.py:237-245`)에서 같은 A 메커니즘이 먼저 났다. `ts_result.json`에
  `config_error: AtomValenceException: Explicit valence for atom # 1 C, 5` — TS 조각에서 니트론 O⁻(TS 원자 17)가
  아세틸 카보닐 C(TS 원자 1)에서 1.83 Å(RDKit C–O 기준 약 1.85 Å 미만)이다. 두 키는 241행에서 config_tags 두 번이
  모두 성공해야 쓰이므로 빠졌다. 두 값은 다른 키에도 없다.
- T10518은 키 우회 뒤 `make_reference.py:192`의 config_tags(TS 조각, 스캔할 이면각을 고른다)에서 같은 A 예외를 낼
  수 있으므로 두 wrapper가 모두 필요하다.

## 우회 (spec 5-2 허용 범위) — `analysis/g2r/ref_repair.py`

| 묶음 | 감싼 함수 | 원래 함수가 이 예외를 낼 때만 | 판정 |
|---|---|---|---|
| A | `stereo_utils.template_match` (모듈 속성; `config_tags`·`dipolarophile_check`가 호출 때 찾음) | `AssignBondOrdersFromTemplate`를 `AllChem.py:394-433` 그대로 다시 하고 마지막 `SanitizeMol`만 원자가 검사(SANITIZE_PROPERTIES) 없이 한다(`UpdatePropertyCache(strict=False)` 뒤 나머지 sanitize 단계는 수행). 반환값 형식 같음 | sanitize 단계 일부를 건너뛰는 우회 = 허용 예시 |
| B | `make_reference.coley_port` (`stage_ref`가 호출 때 `MR.coley_port`로 찾음) | `KeyError('dipole_mapped_smiles'/'dipole_backbone_maps')`이면 `run_ts.py:238`과 같은 `su.dipole_backbone(job["mapped_smiles"])`(manifest 행)로 두 값을 만들어 `ts_result`의 메모리 사본에 넣고 원래 함수를 한 번 더 부른다. `ts_result.json`은 다시 쓰지 않는다 | 분자 표현(SMILES)을 읽는 부분만. 같은 D1 함수, 구조·판정과 무관 |

- 설정 형식 전하를 manifest SMILES에서 지정하는 우회(spec 예시)는 쓰지 않았다: 템플릿이 이미 같은 SMILES의 전하를
  넣고 있어(`AllChem.py:433`) 바꿔도 결과가 같다. 원자가 초과는 전하가 아니라 거리로 생긴 여분 결합 때문이다.
- 바꾸지 않는 것: Coley 판정(회전 문턱, alt 사용 문턱 RMSD > 0.05 Å · ΔG < −0.239 kcal/mol, 고리 수 변화 시 중단,
  dipolarophile 배치 확인), xtb 명령과 수준(`g2_autode.xtb_refine`, GFN2 ALPB(water)), conformer 수 `N_RR = 1000`.
- 범위 판단(기록된 결정, 추천안): 예외는 분자 표현을 읽는 단계의 실패이므로 5-2 범위 안이다. 다만 여분 결합은 실제
  근접(음이온 끝이 ortho C=O·NO2에 붙거나 거의 붙음)이므로, 우회 뒤에는 바뀌지 않은 Coley 규칙
  (`make_reference.py:238`, `:249` 고리 수 변화 → `StageFail`)이 많은 반응을 `ring count changed in alt` 실패로 끝낼 수
  있다. 그것은 화학적 실패로 그대로 실패로 남긴다(5-4-3). 그래서 복구 수 k는 작을 수 있다.
- 재실행은 새 표본이다: autodE conformer 생성(`conf_gen.py:417 np.random.RandomState()`)에 seed가 없고 rr 폴더는
  지워졌다. 반응마다 **끝난 시도는 한 번만** 한다(기록된 결정): `ref_result.orig.json`과 새 `.done_ref`/`.fail_ref`가
  있으면 다시 돌리지 않는다. 중단된 시도(표지 없음)만 `jd/ref`를 지우고 다시 한다.

## 검사

- R-T1·R-T2(`tests_g2r.py` → `results/R_tests.json`): 예외 없는 입력에서 원래 함수 1회 호출·결과 그대로(mock과 실제
  TS 조각 5개), 묶음 A(T10518 TS 조각, 보조로 T09757 TS 조각에서 CH2⁻ 기(탄소 + H 2개)를 한 덩어리로 카보닐 C
  쪽으로 0.05 Å씩 옮겨 원래 함수가 처음 AtomValenceException을 내는 구조)와
  묶음 B(T10518 `coley_port` KeyError)의 우회 경로를 xtb 없이 함수 단위로.
- 실행 결과: `results/R_status.csv`, `results/R_report.json`.
