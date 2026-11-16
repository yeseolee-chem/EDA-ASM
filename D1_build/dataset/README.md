# D1 데이터셋 — 전체 라벨 (B3LYP-D3(BJ)/def2-TZVP, SMD(water), EDA-NOCV + strain)

D1은 1,3-쌍극자 고리화 첨가 11,596개 대상이다. 쌍극자 5종(azide, azomethine ylide, carbonyl ylide, nitrile oxide, nitrone) × 골격 12개 × 치환기, 위치 direct/o/m/p, series R1/R2로 구성된다. TS는 autodE로 찾았다(Nurion → UBAI). 라벨은 ORCA 6.1.1 B3LYP-D3(BJ)/def2-TZVP SMD(water)로 계산한 EDA-NOCV 6채널, strain(d1, d2), BSSE, 장벽이다. 이 폴더는 scratch에만 있던 최종 export와 그 뒤의 기준 구조 보정판(cr2)을 GitHub에 올린 것이다(2026-11-17).

## 어느 라벨을 쓰나

| 파일 | 행 | 용도 |
|---|---|---|
| **`labels_d1_cr2.json`** | 9,266 | **현재 정답 라벨.** D1LADDER Phase Q·V에서 같은 분자의 반응물 기준 구조를 최저 에너지 하나로 통일했다. GNN3R2, ESPB, CH3, CH3M, ABS3가 쓴 라벨이다. |
| `labels_d1.json` | 9,496 | 원본 export(2026-10-18, job 1023364). cr2의 9,266행에 D'' 골격(DppC 111, DppT 119) 230행이 더 있다. 기준 구조 보정 전 값이다. D1PAIR rev 9가 이 판을 썼다. |

- cr2가 바꾼 것은 `e_frag1_rel_eh` · `e_frag2_rel_eh` · `d1_kcal` · `d2_kcal` · `barrier_kcal`(3,131행)과 잔차 열 두 개뿐이다. TS 에너지와 EDA 채널은 같다. cr2 레코드에는 `cr` 필드가 더 있다: 바뀐 조각(`changed`), 분자 key, 기준 출처(`src1`/`src2`: `Q:label_min` = 다른 반응의 기준 구조, `V:label_min` = Phase V 형태 탐색), 바뀌기 전 값(`orig`).
- `cr`의 `ref*_xyz` · `ref*_inp`는 scratch 경로다. 같은 구조가 이 폴더에 있다: `Q:label_min` → `geometries.tar.gz`의 `geometries/<src*_job>/ref_dipole.xyz`(조각 1) 또는 `ref_dipolarophile.xyz`(조각 2), `V:label_min` → `cr2_V_refs/<V id>_<c id>.xyz`(`cr2_V_refs/index.csv`에 key, 쓰는 행, 원래 경로).
- D1LADDER 정의와 보고: 브랜치 `d1-ladder`의 `docs/specs/D1LADDER.md` §2–3, `analysis/d1_ladder/results/{Q_report.json, V_summary.md, V_report.json, labels_d1_cr2.parquet}`.

## 파일

| 파일 | 크기 | 내용 |
|---|---|---|
| `labels_d1_cr2.json` | 39.4 MB | 반응별 라벨 9,266개, 레코드당 필드 79개(아래) |
| `labels_d1.json` | 45.0 MB | 반응별 라벨 9,496개, 필드 78개(`cr` 없음) |
| `d1_status.csv` | 1.6 MB | 대상 11,596개 전체의 결과 분류(`class`), 실패 단계·사유, 재실행, core-h |
| `d1_manifest.csv` | 6.4 MB | 대상 11,596개 정의: mapped/반응 SMILES, 쌍극자·쌍극자친화체 SMILES, 골격, 치환기, 위치, 입체 |
| `geometries.tar.gz` | 18.3 MB | ok 9,496개 × `ts.xyz`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`, `product.xyz` (37,984개, Å) |
| `cr2_V_refs/` | 작음 | cr2가 쓰는 Phase V 기준 구조 xyz 20개(분자 key 20개, 사용 행 123) + `index.csv` |
| `F_pairs_cr2.csv` | 3.1 MB | cr2 짝 ΔΔ 라벨 8,389짝(reference 7,253 · anchor 1,136; `in_P_main` 6,885) — D1LADDER 이후의 짝 정답 |
| `RUN_REPORT.md` | | 본계산 최종 보고: 분류, 게이트 잔차, 쌍극자·골격별 ok, 코어시간 1,394,444 h |
| `RETRIES.md` | | 재실행 68행 기록 |
| `SHA256SUMS` | | 위 파일 전부의 sha256 (`sha256sum -c SHA256SUMS`) |

`d1_status.csv`의 `class`: ok 9,496 · no_ts 805 · pending 668(계산 안 함) · stereo_mismatch 345 · pipeline 160 · gate_ts 101 · infra_fail 10 · engine_abnormal 8 · gate_label 2 · ref_fail 1.

## 라벨 필드 (단위: `_kcal`·`_dft` = kcal/mol, `_eh` = hartree)

- 식별: `job_id`(T00001…), `rxn_id`(1000001…), `ts_id`, `cell_id`, `core`, `dipole`, `skeleton`, `series`, `attach`(direct/o/m/p), `stage`, `regio`, `face`, `stereo_index`, `R1`, `R2`, `rxn_smiles`
- 라벨 10개: `d1_kcal`, `d2_kcal`(조각 변형), `elst_dft`, `pauli_dft`, `oi_dft`, `disp_dft`, `cpcm_dft`, `cds_dft`(EDA 6채널), `bsse_shift_kcal`, `barrier_kcal`
- 관계: `barrier = d1 + d2 + eint_spe`, `eint_spe = e_bond + bsse_shift`, `e_bond = 6채널 합 − eda_sum_minus_bond`(|잔차| ≤ 0.098 kcal/mol). 그래서 barrier ≈ 10개 라벨의 합이다.
- 원 에너지: `e_ab_eh`, `e_frag{1,2}_dist_eh`, `e_frag{1,2}_dist_ghost_eh`, `e_frag{1,2}_rel_eh`, `eda_raw_eh`(bond, orb, elst, pauli, xc, disp, cpcm, smd_cds)
- 품질: `status`, `gates`(게이트 6개), `flags`, `audit_problems`, 잔차 3종, `ts_imag_cm`, `formed_d1`/`formed_d2`(Å)
- 조각 1 = dipole, 조각 2 = dipolarophile(`role1`, `role2`)

```python
import json, tarfile
labels = json.load(open("labels_d1_cr2.json"))          # list of dict, 9,266
by_job = {r["job_id"]: r for r in labels}
with tarfile.open("geometries.tar.gz") as t:
    ts = t.extractfile("geometries/T00001/ts.xyz").read().decode()
```

## 들어 있지 않은 것

- ORCA 원 출력(.out/.gbw)과 autodE 작업 폴더: scratch `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_scratch/jobs`(UBAI)와 Nurion 결과 묶음(약 90 GiB).
- 라벨 생성 코드(`aggregate_run.py` 등): D1 run tree `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`(원격이 없는 별도 git). cr·cr2 코드는 `d1-ladder` 브랜치 `analysis/d1_ladder/refs_q.py`, `refs_v.py`.
- HF/6-31G* 라벨: `d1-hf` 브랜치 `analysis/d1_hf/results/labels_d1_hf.parquet`. GNN3R2 짝 데이터셋(8,594반응 / 8,344짝): `gnn3r2` 브랜치 `analysis/gnn3r2/results/F_pairs_r.csv`.

## 출처 (scratch 원본)

| 파일 | 원본 경로 |
|---|---|
| `labels_d1.json`, `d1_status.csv`, `geometries/`, `RUN_REPORT.md`, `RETRIES.md` | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/` |
| `d1_manifest.csv` | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv` |
| `labels_d1_cr2.json`, `F_pairs_cr2.csv` | `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/` (Phase V, commit `d510de6c`, 2026-10-23) |
| `cr2_V_refs/*.xyz` | `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/V/mols/<V id>/<c id>/final.xyz` |

파일은 바이트 그대로 복사했다(sha256 = 원본, 하위 코드 `d1pair_common.py` 등이 확인하는 값과 같음). `geometries.tar.gz`는 `results/geometries/`를 이름순, 고정 mtime으로 묶었다. cr2의 `Q:label_min` 기준 619개는 모두 `geometries/<src_job>/` 쪽 xyz와 바이트 단위로 같다(2026-11-17 확인).
