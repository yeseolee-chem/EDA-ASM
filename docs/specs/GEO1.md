# GEO-1 — 낮은 수준 TS 구조(xTB) → 라벨 TS 구조 보정 모델 (D0-G2 계산 포함) — Claude Code 실행 지시서

**저장소:** 기존 clone `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3`의 **git worktree**로 새 폴더를 만들고, `origin/d1-g3l`을 기준으로 새 브랜치 **`geo1`**을 만든다.
- 명령(로그인 노드, git만):
  ```bash
  D1G3=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3
  git -C $D1G3 fetch origin
  git -C $D1G3 worktree add -b geo1 /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1 origin/d1-g3l
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
  ```
- 이 문서의 모든 상대 경로(`analysis/geo/...`)는 이 새 폴더 기준이다.
- **d1g3 clone의 작업 트리는 건드리지 않는다**(브랜치 전환·commit 금지). D1-G3/G3L job이 그 폴더의 코드를 쓰고 있을 수 있고, `analysis/d1_g3` 코드가 그 clone 경로를 절대 경로로 쓰기 때문이다.
- 새 clone은 만들지 않는다(디스크 부족). worktree는 `.git` 객체를 공유하므로 소스 파일만 추가로 차지한다.

**이 파일:** `docs/specs/GEO1.md`로 commit한다.
**새 코드:** `analysis/geo/` · **결과:** `analysis/geo/results/`
**scratch:** `$GEO=/gpfs/tmp_cpu2/yeseo1ee/geo`(모델·데이터), `$D0G2=/gpfs/tmp_cpu2/yeseo1ee/d0_g2`(D0-G2 계산)

**순서:** 0 → I(목록·정합성) → Z(D0-G2 계산) → D(데이터셋) → U(단위 검사) → R(학습) → E(평가) → S(요약 후 **멈춤**)
- Z의 본 계산(CPU, 수 시간)이 도는 동안 M·U의 코드 작성과 `tests.py synth`(가짜 분자로 하는 검사)를 먼저 해도 된다. D는 Z가 끝난 뒤 만든다.
- **폴더 만들기(맨 처음 1회):** 로그인 노드에서는 mkdir도 하지 않는다. 아래 job 하나로 만든다.
  `sbatch --time=48:00:00 --output=/gpfs/home1/yeseo1ee/geo_init_%j.out --wrap "mkdir -p /gpfs/tmp_cpu2/yeseo1ee/geo/logs /gpfs/tmp_cpu2/yeseo1ee/geo/data /gpfs/tmp_cpu2/yeseo1ee/geo/runs /gpfs/tmp_cpu2/yeseo1ee/geo/preds /gpfs/tmp_cpu2/yeseo1ee/d0_g2/logs"`

**작성자 사전 검증(2026-10-26, 로컬 CPU, 가짜 데이터):**
- 이 문서의 코드 블록 9개를 그대로 추출해 pyflakes 검사를 통과했다.
- `inventory.py`, `data.py`: 가짜 파일 트리에서 끝까지 돈다.
- `tests.py synth` 전부 통과. `tests.py data`의 U-8·U-9 통과.
- `train.py`(smoke → tune → select → final → ablation)와 `evaluate.py`: 작은 모델로 끝까지 돈다(부분 fold ablation 포함).
- **`d0g2.py`는 HPC 모듈(autodE, ORCA, D1 코드 사본)이 필요해 실행하지 못했다.** `g2_autode.py`(origin/d1-g3l)와 `pilot_common.py`·`run_ts.py`(origin/d1-build) 코드와 대조만 했다. Z-1·Z-2가 그 검증이다.

**멈춤 규칙:**
- 15절 STOP 조건에서 멈추고 묻는다.
- **E가 끝나면 결과를 보고하고 반드시 멈춘다.** 에너지 단계(HF EDA, dft 모델 적용)는 이 명령서에 없다.
- 버그를 찾으면 고치기 전에 보고한다.
- 이 문서의 코드는 **틀이자 기준**이다.
  - 실제 함수 이름이나 경로가 다르면 실행 전에 확인하고, 차이를 `results/DECISIONS.md`에 적은 뒤 맞춘다.
  - 계산 설정(키워드, 수준, 판정 기준)은 바꾸지 않는다.

---

## 0. 무엇을, 왜 (사용자 결정 2026-10-26)

- **배포 구조:** SMILES → **G2 구조**(autodE, xTB) → **[이 명령서] 라벨 구조 예측** → (다음 명령서) 라벨 에너지
- 이 명령서는 **구조만** 다룬다. 에너지는 다루지 않는다.
- **모델:** Uni-Mol+(Lu 등, Nat. Commun. 15, 7104, 2024, doi:10.1038/s41467-024-51321-w)를 차용한다.
  - 차용: 원자 트랙 + 원자쌍 트랙 Transformer, outer product, 삼각형 갱신, 좌표 출력부, 가짜 궤적 학습, 반복 보정 R.
  - 변경점(7절 코드):
    - 좌표 출력부를 원자 간 방향 벡터로 만든 엄밀한 등변형으로 바꾼다.
    - TS용 특성(형성 결합, 조각, 출처, 전하)을 추가한다.
    - 성질 출력은 뺀다.
  - Uni-Mol 공식 코드(Unicore)는 쓰지 않는다. 이 문서의 PyTorch 구현을 쓴다.
- **데이터:** D1 + D0. RGD1 등 외부 데이터는 쓰지 않는다.
- **D0-G2를 새로 계산한다:** D0 반응에 D1-G2와 같은 프로토콜(autodE를 xTB로)을 적용한다.

## 1. 구조 정의

| 이름 | 출발점 | 만드는 방법 | 수준 | 위치 | 원자 순서 |
|---|---|---|---|---|---|
| D1 라벨 TS | SMILES | autodE(xTB 형태 탐색) → ORCA OptTS | B3LYP-D3BJ/def2-SVP SMD(water) | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs/<job_id>/ts.xyz` | 기준 |
| D1 G1 | D1 라벨 TS | ORCA `XTB2 ALPB(water) OptTS Freq` | GFN2-xTB | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1_final/<rxn_id>/ts.xyz` (`.done`) | 라벨과 같음 |
| D1 G2 | SMILES | `analysis/d1_g2/g2_autode.py` (autodE, 모든 고수준 계산 = XTB2 ALPB(water)) | GFN2-xTB | `/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final/<rxn_id>/ts.xyz` (`.done`) | 라벨 순서로 다시 매핑됨 |
| D0 라벨 TS | SMILES | Coley autodE → Gaussian16 | B3LYP-D3(BJ)/def2-SVP SMD(water) | `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles/<rid>/<ts_file>` | 기준 |
| D0 G1 | D0 라벨 TS | ORCA `XTB2 ALPB(water) OptTS Freq` (rev 4 `g1_geom.py`) | GFN2-xTB | `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1/<rid>/ts.xyz` (`.done`) | 라벨과 같음 |
| **D0 G2 (새로 계산)** | SMILES | D1-G2와 같은 코드 (Phase Z) | GFN2-xTB | `$D0G2/g2_final/<rid>/ts.xyz` (`.done`) | 라벨 순서로 다시 매핑 |

## 2. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | 1절의 모든 경로가 있다. `labels_all.json`(repo root), `full_dataset.csv`, D0 `input_meta.csv`(`/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv`) |
| 0-2 | **D1G3L_C DISK-C가 `espley_xtb_g1/`, `espley_xtb_g2/`를 지우지 않았다.**<br>① `analysis/d1_g3l/results/` 아래 DISK 기록(DISK-C 행동 목록 CSV/JSON)에 이 두 경로의 삭제 행이 없다<br>② `espley_xtb_g1/<rid>/.done` 개수(Phase I에서 셈)가 4,500개 이상이다(지워졌다면 0에 가깝다). `analysis/espley_xtb_repro/results/`에 rev 4 G1 성공 수 기록이 있으면 같이 적는다(참고용)<br>③ DISK-C가 아직 안 돌았으면 실행하지 말고 STOP하여 사용자에게 묻는다 |
| 0-3 | `analysis/d1_g2/results/C_compare.csv`(9,266행), `analysis/d1_pair/results/S_folds.csv`, `analysis/d1_pair/results/F_pairs.csv`가 있다 |
| 0-4 | 계산 환경 두 개:<br>• CPU = D1 run 환경(`source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh; PY=$D1_PY`, python 3.9, autodE 1.4.5, rdkit, networkx, xtb 6.7.1, ORCA 6.1.1)<br>• GPU = `analysis/d1_ladder/jobs/gpu.slurm`과 같은 환경(conda `reactot` 활성 후 `PY=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python`, torch 2.2.1+cu118)<br>GPU job으로 `$PY -c "import torch, pandas, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"` 확인 |
| 0-5 | `/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis/rmsd.py`에 `same_ts(a, b, formed_a, formed_b, rmsd_A=0.10, dform_A=0.02)`가 있다 |
| 0-6 | 쓸 디스크 여유가 20 GB 이상이다 (D0-G2 약 5,260반응 × 작은 출력 + 데이터 pickle + 체크포인트) |

---

## 3. 공통 모듈 `analysis/geo/geo_common.py` (그대로 작성)

```python
"""GEO-1 common paths and helpers (docs/specs/GEO1.md). Python 3.9 compatible (D1 run env) and the GPU env (env_d1fu2)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True                  # never write .pyc into the read-only D1 code copy (as c_compare.py)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
GEO = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo"))
D0G2 = Path(os.environ.get("D0G2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2"))

# ---- D1 (read only)
D1_G2_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")
D1_G2_FINAL = D1_G2_ROOT / "g2_final"
D1_LABELS_G2 = D1_G2_ROOT / "G" / "labels_d1_g2.json"      # 9,266 D1 label rows (prep_g.py inputs)
D1_META_G2 = D1_G2_ROOT / "G" / "input_meta_d1_g2.csv"     # rxn_id, A_idx (frag1 = dipole, 0-based)
D1_PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
D1_JOBS = D1_PAIR / "d1_jobs"
D1_G1_FINAL = D1_PAIR / "g1_final"
D1_MANIFEST = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv")
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
S_FOLDS = REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"

# ---- D0 (read only)
D0_LABELS = REPO / "labels_all.json"
D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
D0_META = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv")
D0_G1_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
D0_EXCLUDE = {3090, 3766, 4252, 3400, 5783}                # labels_all non-ok (= g1_geom.EXCLUDE)

RMSD_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis")
D1_FOLD_COUNTS = {0: 1874, 1: 1879, 2: 1503, 3: 1435, 4: 2195, -1: 380}   # C_compare rows per T1 fold (checked in I)
G2RAW_PRIMARY = dict(n=840, share=0.1595, median=0.0155)                  # C_compare d_g2 vs d_dft, label-length rule
ELEM = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]   # type index = position
TYPE = {e: i for i, e in enumerate(ELEM)}


def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], dtype=float)


def done(folder):
    return (Path(folder) / ".done").is_file()


def read_if_done(folder):
    return read_xyz(Path(folder) / "ts.xyz") if done(folder) else None


def parse_pairs(s):
    """'1-2 4-8' -> [(1, 2), (4, 8)] (0-based atom indices, each pair sorted, list sorted) = g2_autode.label_pairs."""
    return sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(s).split())


def dmat(x):
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def form_d(x, pairs):
    D = dmat(x)
    return [float(D[i, j]) for i, j in pairs]


def load_meta(path):
    m = pd.read_csv(path, dtype={"A_idx": str})
    return {int(r.rxn_id): sorted(int(v) for v in str(r.A_idx).split()) for r in m.itertuples()}


def load_d1_labels():
    return json.loads(D1_LABELS_G2.read_text())


def load_d0_labels():
    L = json.loads(D0_LABELS.read_text())
    ok = sorted((d for d in L if int(d["rxn_id"]) not in D0_EXCLUDE), key=lambda d: int(d["rxn_id"]))
    assert all(d["status"] == "ok" for d in ok), "a non-excluded D0 label is not status ok"
    return ok


def load_d0_smiles():
    df = pd.read_csv(D0_SMILES)
    return dict(zip(df.rxn_id.astype(int), df.rxn_smiles))


def kabsch_align(mob, ref, reflect=True):
    """mob aligned onto ref (same atom order). Returns (aligned, rmsd, reflected). With reflect=True the improper
    solution is taken when its RMSD is lower (mirror-image TS of an achiral system = the same TS)."""
    mob, ref = np.asarray(mob, float), np.asarray(ref, float)
    mc, rc = mob.mean(0), ref.mean(0)
    X, Y = mob - mc, ref - rc
    U, S, Vt = np.linalg.svd(X.T @ Y)
    V = Vt.T
    s = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    best = None
    for dd in (s, -s):                       # dd = s -> best proper rotation; dd = -s -> best improper (mirror)
        R = V @ np.diag([1.0, 1.0, dd]) @ U.T
        improper = bool(np.linalg.det(R) < 0)
        if improper and not reflect:
            continue
        A = X @ R.T + rc
        r = float(np.sqrt(((A - ref) ** 2).sum(1).mean()))
        if best is None or r < best[1] - 1e-12:
            best = (A, r, improper)
    return best


COV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, As=1.19, Se=1.20,
           Br=1.20, I=1.39)                     # Cordero et al. 2008 covalent radii (A)
COV_F = 1.25                                    # bond if d < 1.25 (r_i + r_j)
SPD_MAX = 9


def bond_types(syms, xyz, pairs):
    """0 none / 1 covalent (from the INPUT geometry) / 2 forming pair. Never computed from the label geometry."""
    D = dmat(xyz)
    r = np.array([COV[s] for s in syms])
    B = (D < COV_F * (r[:, None] + r[None, :])).astype(np.int8)
    np.fill_diagonal(B, 0)
    for i, j in pairs:
        B[i, j] = B[j, i] = 2                   # forming bond (overrides a covalent call)
    return B


def spd_matrix(B):
    """shortest-path hop counts over bond types 1 and 2, clipped at SPD_MAX (unreachable -> SPD_MAX)."""
    n = len(B)
    S = np.where(B > 0, 1, 99).astype(np.int32)
    np.fill_diagonal(S, 0)
    for k in range(n):                          # Floyd-Warshall (n <= ~70)
        S = np.minimum(S, S[:, k:k + 1] + S[k:k + 1, :])
    return np.minimum(S, SPD_MAX).astype(np.int8)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def core_fold_map():
    """D1 core 'dipole/skeleton' (e.g. 'nitrile oxide/A') -> T1 outer fold (S_folds.csv fold_T1, one value per core;
    -1 = core never a test core in D1PAIR T1)."""
    S = pd.read_csv(S_FOLDS, usecols=["core", "fold_T1"]).drop_duplicates()
    assert S.groupby("core").fold_T1.nunique().max() == 1, "a core has two T1 folds"
    return dict(zip(S.core, S.fold_T1.astype(int)))


def d1_core_table():
    """C_compare (all 9,266 D1 rows): rxn_id -> (core 'dipole/skeleton', T1 fold, same_ts_g1 as True/False/None)."""
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton", "same_ts_g1"])
    fm = core_fold_map()
    out = {}
    for r in C.itertuples():
        core = f"{r.dipole}/{r.skeleton}"
        s = r.same_ts_g1
        s = True if s in (True, "True") else (False if s in (False, "False") else None)
        out[int(r.rxn_id)] = (core, int(fm.get(core, -9)), s)
    return out
```

## 4. Phase I — 목록·정합성 검사 (`inventory.py`, 계산 없음, D1 run 환경 sbatch)

**하는 일:** 아래 표를 계산해 `results/I_inventory.json`에 적는다. ✗가 하나라도 있으면 STOP한다.

| 검사 | 기준 |
|---|---|
| D1 라벨 수 | 9,266 |
| D1 core → fold 매핑 (`GC.d1_core_table()`) | 모든 반응이 fold를 받음(−9 = 0개). fold별 반응 수 = `GC.D1_FOLD_COUNTS` |
| D1 G2 `.done` 수, G1 `.done` 수 | 8,616 / 8,433 (C_compare `g2_status == ok` / `g1_status == ok` 개수와 같음) |
| D1 원소 순서: G2·G1 `ts.xyz`의 원소 목록 = 라벨 | 다른 반응 수 = 0 (다르면 그 반응 목록과 함께 STOP) |
| D1 형성 결합 인덱스(0-based) 확인 | ① 모든 반응에서 각 형성 결합 짝이 서로 다른 조각(A_idx 안 1개, 밖 1개)이고 라벨 거리 1.4–4.0 Å<br>② 앞 300반응에서 라벨 거리(정렬) = C_compare `d_dft_1`, `d_dft_2`(정렬), 최대 차이 ≤ 1e-4 Å |
| D0 라벨 수 | 5,260 |
| D0 형성 결합 인덱스(0-based) 확인 | 모든 반응에서 각 형성 결합 짝이 서로 다른 조각(A_idx 안 1개, 밖 1개)이고 Coley TS 거리 1.4–4.0 Å. `formed_d1/2`와의 차이는 참고로만 적는다 |
| D0 G1 `.done` 수, 원소 순서 = Coley TS | ≥ 4,500 (0-2); 원소 순서가 다른 반응 수 = 0 |
| D0 조각 정의: `input_meta.csv`의 A_idx = 쌍극자 원자 | 모든 반응에서 `len(A_idx) == (n_f1 if role1 == "dipole" else n_f2)` |
| D0 기준 구조 파일 | `role1`/`role2`가 {"dipole", "dipolarophile"}이고, `rel1_file`/`rel2_file` 원자 수 = `n_f1`/`n_f2` |
| 원소 집합 | 모두 `ELEM` 안. 아니면 STOP하고 원소 목록 보고 |
| D0 SMILES | `full_dataset.csv`에 `rxn_id`, `rxn_smiles`(atom-mapped)가 있고 5,260개 모두 있음 |
| D1 SMILES | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/d1_manifest.csv`에서 `kind == "D1"` 행 수 ≥ 9,266, `rxn_smiles` 열이 있음 (D0 겹침 검사에 씀) |

```python
#!/usr/bin/env python3
"""Phase I: inventory and consistency checks (spec 4). D1 run env, sbatch:  python inventory.py
Writes results/I_inventory.json; exit 3 if any check fails."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

OUT = []


def chk(name, value, expected, ok):
    OUT.append(dict(check=name, value=value, expected=expected, ok=bool(ok)))
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value} (expected {expected})", flush=True)


def pair_ok(xyz, pairs, A):
    """each forming pair joins the two fragments and is 1.4-4.0 A long (0-based index check)."""
    A = set(A)
    return all(((i in A) != (j in A)) and 1.4 <= float(np.linalg.norm(xyz[i] - xyz[j])) <= 4.0 for i, j in pairs)


def main():
    C = pd.read_csv(GC.C_COMPARE)
    Ci = C.set_index("rxn_id")
    labs1 = GC.load_d1_labels()
    chk("D1 labels", len(labs1), 9266, len(labs1) == 9266)
    T1 = GC.d1_core_table()
    fc = pd.Series([T1.get(int(d["rxn_id"]), (None, -9, None))[1] for d in labs1]).value_counts()
    fc = {int(k): int(v) for k, v in fc.items()}
    chk("D1 fold counts", fc, GC.D1_FOLD_COUNTS, fc == GC.D1_FOLD_COUNTS)
    meta1 = GC.load_meta(GC.D1_META_G2)
    n2 = n1 = 0
    bad_order, bad_pair, elems, dev, n_dev = [], [], set(), 0.0, 0
    for i, d in enumerate(labs1):
        rid = int(d["rxn_id"])
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        elems |= set(lab[0])
        pairs = GC.parse_pairs(d["formed_pairs_ts"])
        if rid not in meta1 or not pair_ok(lab[1], pairs, meta1[rid]):
            bad_pair.append(f"D1 {rid}")
        for kind, root in (("g2", GC.D1_G2_FINAL), ("g1", GC.D1_G1_FINAL)):
            x = GC.read_if_done(root / str(rid))
            if x is None:
                continue
            n2, n1 = (n2 + 1, n1) if kind == "g2" else (n2, n1 + 1)
            if list(x[0]) != list(lab[0]):
                bad_order.append(f"D1 {rid} {kind}")
        if n_dev < 300 and rid in Ci.index and pd.notna(Ci.at[rid, "d_dft_1"]):
            ref = np.sort([Ci.at[rid, "d_dft_1"], Ci.at[rid, "d_dft_2"]])
            dev = max(dev, float(np.abs(np.sort(GC.form_d(lab[1], pairs)) - ref).max()))
            n_dev += 1
    n2e, n1e = int((C.g2_status == "ok").sum()), int((C.g1_status == "ok").sum())
    chk("D1 G2 .done", n2, n2e, n2 == n2e)
    chk("D1 G1 .done", n1, n1e, n1 == n1e)
    chk("D1 forming distance vs C_compare (300 rxns)", dev, "<= 1e-4", dev <= 1e-4 and n_dev == 300)
    labs0 = GC.load_d0_labels()
    chk("D0 labels", len(labs0), 5260, len(labs0) == 5260)
    meta0, smi = GC.load_meta(GC.D0_META), GC.load_d0_smiles()
    n0g1, bad_role, bad_meta, dev0 = 0, [], [], 0.0
    for L in labs0:
        rid = int(L["rxn_id"])
        pdir = GC.D0_PROF / str(rid)
        lab = GC.read_xyz(pdir / L["ts_file"])
        elems |= set(lab[0])
        pairs = GC.parse_pairs(L["formed_pairs_ts"])
        if rid not in meta0 or not pair_ok(lab[1], pairs, meta0[rid]):
            bad_pair.append(f"D0 {rid}")
        try:
            dev0 = max(dev0, float(np.abs(np.sort(GC.form_d(lab[1], pairs))
                                          - np.sort([float(L["formed_d1"]), float(L["formed_d2"])])).max()))
        except (KeyError, TypeError, ValueError):
            pass
        g1 = GC.read_if_done(GC.D0_G1_ROOT / str(rid))
        if g1 is not None:
            n0g1 += 1
            if list(g1[0]) != list(lab[0]):
                bad_order.append(f"D0 {rid} g1")
        if {L["role1"], L["role2"]} != {"dipole", "dipolarophile"}:
            bad_role.append(rid)
            continue
        nf = {L["role1"]: int(L["n_f1"]), L["role2"]: int(L["n_f2"])}
        if len(meta0.get(rid, [])) != nf["dipole"]:
            bad_meta.append(rid)
        for k in (1, 2):
            if int((pdir / L[f"rel{k}_file"]).read_text().split()[0]) != int(L[f"n_f{k}"]):
                bad_role.append(rid)
    chk("D0 G1 .done", n0g1, ">= 4500", n0g1 >= 4500)
    chk("D0 A_idx = dipole atoms", bad_meta[:20], [], not bad_meta)
    chk("D0 roles / reference files", sorted(set(bad_role))[:20], [], not bad_role)
    chk("D0 formed_d1/2 vs Coley TS (info)", dev0, "info only", True)
    chk("forming pairs join the fragments, 1.4-4.0 A", bad_pair[:20], [], not bad_pair)
    chk("element order input = label", bad_order[:20], [], not bad_order)
    chk("elements in ELEM", sorted(elems - set(GC.ELEM)), [], elems <= set(GC.ELEM))
    n_smi = sum(int(L["rxn_id"]) in smi for L in labs0)
    chk("D0 SMILES present", n_smi, 5260, n_smi == 5260)
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False, usecols=["kind", "rxn_smiles"])
    n_d1 = int((man.kind == "D1").sum())
    chk("D1 manifest rows kind D1", n_d1, ">= 9266", n_d1 >= 9266)
    GC.write_json(GC.RES / "I_inventory.json", OUT)
    return 0 if all(o["ok"] for o in OUT) else 3


if __name__ == "__main__":
    sys.exit(main())
```

## 5. Phase Z — D0-G2 계산 (`analysis/geo/d0g2.py`)

### 5-1. 원칙
- **계산 내용은 D1-G2와 완전히 같다.**
  - `analysis/d1_g2/g2_autode.py`를 **import해서** 그 함수들을 그대로 부른다.
  - 단계: autodE ts → stereo → ref → final(라벨 순서로 다시 매핑, XTB2 Freq).
  - 키워드, config(`analysis/d1_g2/config_g2.yaml`), 판정 기준을 바꾸지 않는다.
- 바꾸는 것은 **"라벨이 어디 있나"**와 **"작업 목록"**뿐이다(아래 덮어쓰기 목록).
- `g2_autode.cmd_pool`은 쓰지 않는다. 그 함수는 `sys.executable g2_autode.py one ...`으로 자기 파일을 다시 부르기 때문에 D0 덮어쓰기가 사라진다. 자체 pool을 쓴다.
- autodE TS template 폴더는 `$D0G2/ts_templates`(= `G.G2/"ts_templates"`)에 빈 상태에서 새로 쌓인다. D1-G2도 빈 상태에서 시작했다. D1-G2의 template을 복사하지 않는다.
- 실행 종류(`kind`)는 D1 검증 단계에서 SMILES로부터 autodE를 돌릴 때 쓴 `D0_control`이다. `run_ts.run_autode`/`analyse`는 `kind`로 갈라지지 않는다(코드 확인함). stereo 단계는 Z-2 ④에서 확인한다.

### 5-2. 코드 (그대로 작성; 함수 이름이 다르면 맞추고 기록)

```python
#!/usr/bin/env python3
"""Phase Z: D0-G2 = the D1-G2 protocol (analysis/d1_g2/g2_autode.py imported, compute unchanged) on the D0 reactions.
sbatch only, D1 run env (jobs/d0g2.slurm).

  python d0g2.py manifest                        # $D0G2/d0_manifest.csv, $D0G2/G/labels_d0_g2.json, $D0G2/labjobs/
  python d0g2.py pilot_list                      # results/Z_pilot.csv
  python d0g2.py one <job_id> <cores>            # one reaction (g2_autode.cmd_one with the D0 overrides)
  python d0g2.py pool <list.csv> <n_par> <cores> [i n]
  python d0g2.py report                          # results/Z_status.csv, results/Z_report.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo_common as GC  # noqa: E402

os.environ["D1G2_SCRATCH"] = str(GC.D0G2)          # MUST precede the g2 imports (g2_common.G2 is read at import)
sys.path.insert(0, str(GC.REPO / "analysis" / "d1_g2"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import g2_common as G  # noqa: E402
import g2_autode as GA  # noqa: E402

assert G.G2 == GC.D0G2 and str(G.G2) != "/gpfs/tmp_cpu2/yeseo1ee/d1_g2", "D0G2 scratch must not be the D1-G2 scratch"
MANIFEST = GC.D0G2 / "d0_manifest.csv"
LABJOBS = GC.D0G2 / "labjobs"                       # <job_id>/{ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz} = Coley
LABELS = GC.D0G2 / "G" / "labels_d0_g2.json"

# ---- overrides (this process only); every name below is looked up by g2_autode at call time
G.MANIFEST = MANIFEST                               # cmd_one (pc.load_job) and stage_stereo (--manifest)
G.G2_FINAL, G.G2_JOBS, G.LOGS = GC.D0G2 / "g2_final", GC.D0G2 / "jobs", GC.D0G2 / "logs"
G.DP.D1_JOBS = LABJOBS                              # finalize reads the label TS / references from here
G.targets = lambda: json.loads(LABELS.read_text())  # cmd_one: lab = {job_id: row}
_META = {}


def _lab_A_idx(job):                                # finalize -> lab_A_idx(job): label partition (frag1 = dipole)
    if not _META:
        _META.update(GC.load_meta(GC.D0_META))
    return _META[int(float(job["d0_rxn_id"]))]


GA.lab_A_idx = _lab_A_idx
SKIP_STEREO = os.environ.get("D0G2_SKIP_STEREO") == "1"   # decision Z-2 only (5-4); default off
_orig_stereo = GA.stage_stereo


def _stage_stereo(cfg_path, job, jd):
    if SKIP_STEREO:
        (jd / "stereo_skipped.txt").write_text("D0G2_SKIP_STEREO=1 (decision Z-2)\n")
        GA.pc.mark_done(jd, "stereo")
        return True
    return _orig_stereo(cfg_path, job, jd)


GA.stage_stereo = _stage_stereo


def _atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cmd_manifest():
    sys.path.insert(0, str(G.PILOT))
    import sample_pilot as sp                        # dipole_of, strip_maps, unspecified_db (as make_tasks.d0_row)
    hdr = list(pd.read_csv(G.D1_REPO / "D1_build" / "validation" / "val_manifest.csv", nrows=0).columns)
    labs, smi = GC.load_d0_labels(), GC.load_d0_smiles()
    rows, tg = [], []
    for L in labs:
        rid = int(L["rxn_id"])
        jid, s = f"D0G2_{rid:04d}", smi[rid]
        dsmi, psmi = sp.dipole_of(s)
        r = {k: "" for k in hdr}
        r.update(job_id=jid, kind="D0_control", ts_id=f"D0-{rid:04d}", core="D0", panel="D0_G2",
                 charge=int(L["charge1"]) + int(L["charge2"]), mult=1,     # make_tasks.d0_row hard-codes 0: not here
                 product_smiles=sp.strip_maps(s).split(">>")[1], mapped_smiles=s, dipole_smiles=dsmi,
                 dipolarophile_smiles=psmi, n_unspec_dipole_db=sp.unspecified_db(dsmi), rxn_smiles=sp.strip_maps(s),
                 d0_rxn_id=rid, exp="D0G2", ade_name="", meta_rxn_id=rid)
        for k in ("barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft",
                  "cds_dft", "formed_d1", "formed_d2", "alt_used", "rel1_file", "rel2_file"):
            if f"d0_{k}" in r and k in L:                                   # informational columns only
                r[f"d0_{k}"] = L[k]
        rows.append(r)
        jd = LABJOBS / jid
        jd.mkdir(parents=True, exist_ok=True)
        pdir = GC.D0_PROF / str(rid)
        shutil.copy(pdir / L["ts_file"], jd / "ts.xyz")
        rel = {L["role1"]: L["rel1_file"], L["role2"]: L["rel2_file"]}
        shutil.copy(pdir / rel["dipole"], jd / "ref_dipole.xyz")
        shutil.copy(pdir / rel["dipolarophile"], jd / "ref_dipolarophile.xyz")
        tg.append(dict(job_id=jid, rxn_id=rid, formed_pairs_ts=L["formed_pairs_ts"], charge1=int(L["charge1"]),
                       charge2=int(L["charge2"]), n_f1=int(L["n_f1"]), n_f2=int(L["n_f2"]), ts_file="ts.xyz",
                       dipole="D0", skeleton="D0"))
    _atomic_csv(pd.DataFrame(rows, columns=hdr), MANIFEST)
    GC.write_json(LABELS, tg)
    print(f"manifest {len(rows)} rows -> {MANIFEST}", flush=True)
    return 0


def cmd_pilot_list():
    labs = GC.load_d0_labels()
    df = pd.DataFrame(dict(rxn_id=[int(d["rxn_id"]) for d in labs], n_atoms=[int(d["n_atoms"]) for d in labs],
                           charged=[int(d["charge1"]) + int(d["charge2"]) != 0 for d in labs]))
    df["q"] = pd.qcut(df.n_atoms, 4, labels=False, duplicates="drop")
    rng = np.random.default_rng(20261026)
    pick = []
    for q in sorted(df.q.unique()):
        pool = sorted(df[(df.q == q) & ~df.charged].rxn_id)
        pick += sorted(int(v) for v in rng.choice(pool, 5, replace=False))
    pick += sorted(int(v) for v in rng.choice(sorted(df[df.charged].rxn_id), 2, replace=False))
    _atomic_csv(pd.DataFrame(dict(job_id=[f"D0G2_{r:04d}" for r in pick])), GC.RES / "Z_pilot.csv")
    return 0


def cmd_one(T, cores):
    return GA.cmd_one(T, cores)


def cmd_pool(lst, n_par, cores, i=0, m=1):
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(m)]
    lab = {d["job_id"]: d for d in G.targets()}
    t0, stop_h = time.time(), float(os.environ.get("D0G2_STOP_AFTER_H", "44"))

    def one(T):
        fin = G.G2_FINAL / str(lab[T]["rxn_id"])
        if (fin / ".done").is_file() or list(fin.glob(".fail_*")):
            return T, "skip"
        if time.time() - t0 > stop_h * 3600:
            return T, "not_started"
        log = G.LOGS / "rxn" / f"{T}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores)],
                               stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        fs = [p.name for p in fin.glob(".*")] if fin.is_dir() else []
        return T, (fs[0] if fs else f"error rc={r.returncode}")

    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for fu in as_completed([ex.submit(one, T) for T in ids]):
            T, s = fu.result()
            k = ".fail" if s.startswith(".fail") else s
            res[k] = res.get(k, 0) + 1
            print(f"  {T}: {s}", flush=True)
    print(f"[d0g2 pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


def cmd_report():
    """status / core-h / forming distances (label atom order, no isomorphism search). The 'same TS' judgment of
    D0 G2 vs D0 G1 (rmsd.same_ts) is made in data.py, which imports rmsd before anything else (fragmenter path)."""
    rows = []
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        fin = G.G2_FINAL / str(rid)
        r = dict(rxn_id=rid, status="missing", n_atoms=int(L["n_atoms"]))
        if (fin / "result.json").is_file():
            js = json.loads((fin / "result.json").read_text())
            r.update(status=js.get("status"), core_h=js.get("core_h_total"), remapped=js.get("remapped"),
                     fail=";".join(js.get("fail_reasons", [])))
        if GC.done(fin):
            pairs = GC.parse_pairs(L["formed_pairs_ts"])
            g2 = GC.read_xyz(fin / "ts.xyz")
            lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
            dg2 = np.sort(GC.form_d(g2[1], pairs))
            r.update(dform_label=float(np.abs(dg2 - np.sort(GC.form_d(lab[1], pairs))).max()))
            g1 = GC.read_if_done(GC.D0_G1_ROOT / str(rid))
            if g1 is not None and list(g1[0]) == list(g2[0]):
                r.update(dform_g1=float(np.abs(dg2 - np.sort(GC.form_d(g1[1], pairs))).max()))
        rows.append(r)
    D = pd.DataFrame(rows)
    D["core_h"] = pd.to_numeric(D.get("core_h"), errors="coerce")
    _atomic_csv(D, GC.RES / "Z_status.csv")
    ok = D[D.status == "ok"]
    med = lambda c: float(ok[c].median()) if c in ok and ok[c].notna().any() else None   # noqa: E731
    GC.write_json(GC.RES / "Z_report.json", dict(
        n=len(D), by_status={str(k): int(v) for k, v in D.status.value_counts().items()},
        ok_rate=float((D.status == "ok").mean()),
        core_h=dict(sum=float(D.core_h.sum()), mean=float(D.core_h.mean()), median=float(D.core_h.median()),
                    p90=float(D.core_h.quantile(.9))),
        dform_label_median=med("dform_label"), dform_g1_median=med("dform_g1"),
        share_dform_g1_le_002=float((ok.dform_g1 <= 0.02).mean()) if "dform_g1" in ok else None,
        reference_D1=dict(g2_ok_rate=0.930, same_g1_rate=0.561, core_h_mean=0.336)))   # C_compare, comparison only
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = dict(manifest=(cmd_manifest, 0), pilot_list=(cmd_pilot_list, 0), one=(cmd_one, 2), report=(cmd_report, 0))
    if a and a[0] == "pool" and len(a) in (4, 6):
        sys.exit(cmd_pool(*a[1:]))
    if a and a[0] in cmds and len(a) - 1 == cmds[a[0]][1]:
        sys.exit(cmds[a[0]][0](*a[1:]))
    raise SystemExit(__doc__)
```

### 5-3. SLURM

`analysis/geo/jobs/cpu.slurm` (D1 run 환경; Phase I, Z, D와 CPU 작업 전부):

```bash
#!/bin/bash
#SBATCH --job-name=geo_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo/logs/%x_%j.out
# submit from analysis/geo/:  sbatch [-p ... --cpus-per-task=N --mem=M -J name --array=...] jobs/cpu.slurm <script.py> <args...>
# "@I" -> $SLURM_ARRAY_TASK_ID.  D1G2_SCRATCH: $D0G2 unless GEO_D1G2_SCRATCH is exported (only for check Z-1).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
export D1G2_SCRATCH="${GEO_D1G2_SCRATCH:-$D0G2_SCRATCH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 D1G2_JOB_T0=$(date +%s)
echo "[geo_cpu] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) D1G2_SCRATCH=$D1G2_SCRATCH $($PY -V 2>&1) $(date '+%F %T')"
args=(); for a in "$@"; do args+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" "${args[@]}"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
```

배치 크기:
- pilot: `sbatch -p <cpu 파티션> --cpus-per-task=32 --mem=96G jobs/cpu.slurm d0g2.py pool results/Z_pilot.csv 4 8`
- 본 계산: `sbatch -p <cpu 파티션> --array=0-9 --cpus-per-task=32 --mem=96G jobs/cpu.slurm d0g2.py pool /gpfs/tmp_cpu2/yeseo1ee/d0_g2/d0_manifest.csv 4 8 @I 10`
  - manifest의 `job_id` 열을 그대로 작업 목록으로 쓴다.
  - pool은 `.done`/`.fail_*`이 있는 반응을 건너뛴다. 다시 제출해도 안전하다.
- CPU 파티션 이름과 노드당 메모리는 `g2.slurm`으로 D1-G2를 돌릴 때 쓴 값을 따른다(`analysis/d1_g2/results/`의 기록 참고). 확인한 값을 `DECISIONS.md`에 적는다.
- `--time=48:00:00`. pool 안에서 44시간이 지나면 새 반응을 시작하지 않는다.

### 5-4. 단계와 판정

| 단계 | 할 일 | 통과 기준 / 미리 정한 결정 |
|---|---|---|
| Z-0 | `manifest` | 5,260행. `val_manifest.csv` 헤더와 열이 같음(추가 열 없음). `labjobs` 5,260개. 반응마다 `ts.xyz` 원소 순서 = Coley TS. `charge` ≠ 0인 행 수를 보고 |
| Z-1 | **환경 확인.** 원래 `analysis/d1_g2/g2_autode.py one <job_id> 8`을 D1 반응 2개(`C_compare`에서 `g2_status == ok`이고 `n_atoms`가 가장 작은 2개)에 대해 **새 임시 scratch**로 다시 돌린다:<br>`sbatch --export=ALL,GEO_D1G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo/z1_check --cpus-per-task=8 jobs/cpu.slurm ../d1_g2/g2_autode.py one <job_id> 8` | ① status ok<br>② 새 `z1_check/g2_final/<rxn_id>/result.json`의 `ts_result.autode_version`, `ts_result.keywords`가 기존 `/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final/<rxn_id>/result.json`의 같은 항목과 같다<br>③ 같은 TS 여부와 형성 결합 차이는 **보고만** 한다(autodE 형태 탐색에는 무작위성이 있다)<br>①·② 중 하나라도 아니면 환경이 바뀐 것이므로 STOP |
| Z-2 | 3개 D0 반응(가장 작은 중성 2개 + 전하 있는 가장 작은 1개)을 `d0g2.py one <job_id> 8`로 | ① `$D0G2/jobs/<job_id>/inputs.tar.gz` 안 모든 `.inp`의 `!` 줄 집합이 Z-1 반응의 `!` 줄 집합 안에 있다(`XTB2 ALPB(water)` + 실행 종류; B3LYP·def2 같은 DFT 키워드 없음)<br>② `g2_final/<rid>/ts.xyz` 원소 순서 = Coley<br>③ `result.json` status ok 또는 의미 있는 실패 이유<br>④ stereo 단계 확인: `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/stereo_check.py`를 읽어서 `kind`에 따라 검사가 달라지는지 확인하고 `DECISIONS.md`에 적는다<br>**stereo 단계가 D1 전용 manifest 열 때문에 예외(traceback)로 실패하면** `D0G2_SKIP_STEREO=1`로 다시 돌린다. 이것은 미리 승인된 결정이며 `DECISIONS.md`에 기록한다. stereo 판정이 정상적으로 "불일치"를 낸 경우는 그대로 실패로 둔다 |
| Z-3 | pilot 22반응 | 성공률 ≥ 0.7. 전체 예상 = pilot 반응당 **평균** 코어시간 × 5,260 × 1.2 |
| Z-4 | 예상이 **4,000코어시간 이하**면 본 계산 배열 제출, 넘으면 STOP | — |
| Z-5 | `report` | `Z_report.json`: 성공률(D1-G2 0.930), 코어시간, 형성 결합 차이. **D0 G2와 D0 G1이 같은 TS인 비율**(D1은 0.561)은 Phase D(`D_dataset_report.json`의 `by_src.D0.same`)에서 나온다. 성공률이 0.7 미만이거나 파이프라인 오류(`error_*`, `missing`)가 5 %를 넘으면 STOP |

## 6. Phase D — 데이터셋 (`analysis/geo/data.py`, D1 run 환경 sbatch; rdkit·rmsd 필요)

### 6-1. 규칙
- 반응마다 레코드 1개: 라벨 좌표, 입력 좌표(G2, G1; 없으면 None), 원자 유형, 조각 표시, 형성 결합 표시, 전하, 출처, core, fold, 같은 형태 여부, 겹침 여부.
- D1의 core, fold, 같은 형태 여부는 `GC.d1_core_table()`(C_compare + S_folds)에서 가져온다.
- pickle 안의 배열은 numpy 배열이다. D1 run 환경(numpy 1.x)과 GPU 환경의 numpy 버전이 다르면 읽기가 실패할 수 있다. U-0에서 GPU 환경으로 먼저 읽어 본다.
- **정답 정렬:** 라벨을 각 입력에 Kabsch로 겹친다. 거울상이 더 맞으면 거울상을 쓴다(같은 TS).
- **결합 그래프:** 입력 구조에서 계산한다(공유 결합 반지름 × 1.25). 형성 결합은 종류 2로 표시한다. **라벨 구조에서는 계산하지 않는다**(누수 방지).
- **같은 형태 여부:**
  - D1: `C_compare.same_ts_g1`
  - D0: `rmsd.same_ts(G2, G1)`. 먼저 D1 300반응에서 같은 함수가 `C_compare`를 재현하는지(일치 ≥ 0.99) 확인한다. 아니면 STOP.
- **D0 겹침:** 원자 번호(map)를 뗀 정규 반응 SMILES가 D1 반응과 같은 D0 반응은 `overlap=True`로 표시하고 학습에서 뺀다.

### 6-2. 코드

```python
#!/usr/bin/env python3
"""Phase D: GEO-1 records -> $GEO/data/records.pkl + results/D_dataset_report.json (D1 run env)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

bond_types, spd_matrix = GC.bond_types, GC.spd_matrix


def canon_rxn(smi):
    """map-free canonical reaction SMILES (explicit [H:n] atoms of the mapped D0 SMILES removed)."""
    from rdkit import Chem
    out = []
    for side in smi.split(">>"):
        ms = []
        for s in side.split("."):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            m = Chem.RemoveHs(m)
            ms.append(Chem.MolToSmiles(m))
        out.append(".".join(sorted(ms)))
    return ">>".join(out)


def d0_overlap_ids():
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False)
    s1 = man[man.kind == "D1"].rxn_smiles
    d1 = {c for c in (canon_rxn(s) for s in s1) if c is not None}
    smi = GC.load_d0_smiles()
    c0 = {rid: canon_rxn(s) for rid, s in smi.items()}
    stats = dict(n_d1_smiles=int(len(s1)), n_d1_canon=len(d1), n_d0_canon_none=sum(v is None for v in c0.values()))
    return {rid for rid, c in c0.items() if c is not None and c in d1}, stats


def make_rec(src, rid, core, fold, lab, g2, g1, A_idx, pairs, charge, same, overlap, log):
    syms, lx = lab
    n = len(syms)
    if any(s not in GC.TYPE for s in syms):
        log.append(f"{src} {rid}: element outside ELEM -> dropped")
        return None
    rec = dict(uid=f"{src}_{rid}", src=src, src_id=0 if src == "D1" else 1, rxn_id=int(rid), core=core, fold=int(fold),
               n=n, types=np.array([GC.TYPE[s] for s in syms], np.int64), lab=lx.astype(np.float64),
               frag=np.array([1 if i in set(A_idx) else 2 for i in range(n)], np.int64),
               form=np.array([1 if any(i in p for p in pairs) else 0 for i in range(n)], np.int64),
               pairs=np.array(pairs, np.int64), charge=int(charge), same=same, overlap=bool(overlap))
    for kind, inp in (("g2", g2), ("g1", g1)):
        rec[f"xyz_{kind}"] = None
        if inp is None:
            continue
        if list(inp[0]) != list(syms):
            log.append(f"{src} {rid}: {kind} element order differs -> input dropped")
            continue
        x = inp[1].astype(np.float64)
        al, r, refl = GC.kabsch_align(lx, x, reflect=True)
        B = bond_types(syms, x, pairs)
        rec.update({f"xyz_{kind}": x, f"lab_on_{kind}": al, f"rmsd_{kind}": r, f"refl_{kind}": refl,
                    f"bond_{kind}": B, f"spd_{kind}": spd_matrix(B)})
    return rec


def same_ts_check_d1(labs, T1, k=300):
    """rmsd.same_ts(G2, G1) as computed here vs C_compare.same_ts_g1 on k random D1 reactions."""
    rng = np.random.default_rng(20261026)
    cand = [d for d in labs if T1.get(int(d["rxn_id"]), (None, None, None))[2] is not None]
    pick = [cand[i] for i in rng.choice(len(cand), min(k, len(cand)), replace=False)]
    agree = []
    for d in pick:
        rid = int(d["rxn_id"])
        g2, g1 = GC.read_if_done(GC.D1_G2_FINAL / str(rid)), GC.read_if_done(GC.D1_G1_FINAL / str(rid))
        if g2 is None or g1 is None:
            continue
        pairs = GC.parse_pairs(d["formed_pairs_ts"])
        s, _, _ = rm.same_ts(g2, g1, GC.form_d(g2[1], pairs), GC.form_d(g1[1], pairs))
        agree.append(bool(s) == T1[rid][2])
    return float(np.mean(agree)), len(agree)


def main():
    log, recs = [], []
    T1 = GC.d1_core_table()                          # rxn_id -> (core, fold, same_ts_g1)
    labs1 = GC.load_d1_labels()
    agree, n_agree = same_ts_check_d1(labs1, T1)
    if agree < 0.99 or n_agree < 250:
        print(f"STOP: rmsd.same_ts reproduces C_compare only {agree:.3f} (n={n_agree})", file=sys.stderr)
        return 3
    meta1 = GC.load_meta(GC.D1_META_G2)
    for d in labs1:
        rid = int(d["rxn_id"])
        core, fold, same = T1[rid]
        if rid not in meta1:
            log.append(f"D1 {rid}: no A_idx in input_meta_d1_g2.csv -> dropped")
            continue
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        r = make_rec("D1", rid, core, fold, lab, GC.read_if_done(GC.D1_G2_FINAL / str(rid)),
                     GC.read_if_done(GC.D1_G1_FINAL / str(rid)), meta1[rid], GC.parse_pairs(d["formed_pairs_ts"]),
                     int(d["charge1"]) + int(d["charge2"]), same, False, log)
        if r is not None:
            recs.append(r)
    meta0 = GC.load_meta(GC.D0_META)
    ov, ov_stats = d0_overlap_ids()
    if ov_stats["n_d1_canon"] < 9000:
        print(f"STOP: only {ov_stats['n_d1_canon']} D1 reaction SMILES parsed for the overlap check", file=sys.stderr)
        return 3
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        pairs = GC.parse_pairs(L["formed_pairs_ts"])
        lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
        g2, g1 = GC.read_if_done(GC.D0G2 / "g2_final" / str(rid)), GC.read_if_done(GC.D0_G1_ROOT / str(rid))
        same = None
        if g2 is not None and g1 is not None and list(g2[0]) == list(g1[0]):
            same = bool(rm.same_ts(g2, g1, GC.form_d(g2[1], pairs), GC.form_d(g1[1], pairs))[0])
        r = make_rec("D0", rid, "D0", -1, lab, g2, g1, meta0[rid], pairs, int(L["charge1"]) + int(L["charge2"]),
                     same, rid in ov, log)
        if r is not None:
            recs.append(r)
    out = GC.GEO / "data" / "records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    tmp.replace(out)
    df = pd.DataFrame([dict(src=r["src"], fold=r["fold"], g2=r["xyz_g2"] is not None, g1=r["xyz_g1"] is not None,
                            same=str(r["same"]), overlap=r["overlap"], refl_g2=bool(r.get("refl_g2")),
                            refl_g1=bool(r.get("refl_g1"))) for r in recs])
    cnt = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}   # noqa: E731  JSON-safe keys
    n_bad_order = sum("element order differs" in m for m in log)
    GC.write_json(GC.RES / "D_dataset_report.json", dict(
        n=len(recs), same_ts_check=dict(agree=agree, n=n_agree), overlap_check=ov_stats, log=log[:200], n_log=len(log),
        n_element_order_differs=n_bad_order,
        by_src={s: dict(n=int(len(g)), g2=int(g.g2.sum()), g1=int(g.g1.sum()), same=cnt(g.same))
                for s, g in df.groupby("src")},
        d0_overlap=int(df.overlap.sum()), d1_fold_counts=cnt(df[df.src == "D1"].fold),
        reflected=dict(g2=int(df.refl_g2.sum()), g1=int(df.refl_g1.sum())),
        records=str(out), sha256=GC.sha256(out)))
    if n_bad_order > 0.01 * len(recs):
        print(f"STOP: element order differs for {n_bad_order} inputs (> 1 %)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 7. Phase M — 모델 (`analysis/geo/model.py`, GPU 환경 = env_d1fu2)

```python
"""GEO-1 refiner: Uni-Mol+ two-track trunk (Lu et al. 2024) + E(3)-equivariant position head. PyTorch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


class Gaussian(nn.Module):
    """Uni-Mol Gaussian distance encoding with per-element-pair affine scaling."""
    def __init__(self, K, n_edge):
        super().__init__()
        self.K = K
        self.means = nn.Parameter(torch.empty(K).uniform_(0, 3))
        self.stds = nn.Parameter(torch.empty(K).uniform_(0, 3))
        self.mul = nn.Embedding(n_edge, 1)
        self.bias = nn.Embedding(n_edge, 1)
        nn.init.constant_(self.mul.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

    def forward(self, dist, et):
        x = self.mul(et).squeeze(-1) * dist + self.bias(et).squeeze(-1)
        std = self.stds.abs() + 1e-2
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.means) / std) ** 2) / (math.sqrt(2 * math.pi) * std)


class PairBiasAttention(nn.Module):
    def __init__(self, dx, dp, H, drop):
        super().__init__()
        self.H, self.dh = H, dx // H
        self.qkv, self.out = nn.Linear(dx, 3 * dx), nn.Linear(dx, dx)
        self.bias = nn.Sequential(nn.LayerNorm(dp), nn.Linear(dp, H, bias=False))
        self.drop = nn.Dropout(drop)

    def forward(self, x, p, mask):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.H, self.dh).unbind(2)
        lg = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh) + self.bias(p).permute(0, 3, 1, 2)
        lg = lg.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.drop(lg.softmax(-1))
        o = torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, N, -1)
        return self.out(o) * mask[..., None]


class OuterProduct(nn.Module):
    def __init__(self, dx, dp, do):
        super().__init__()
        self.ln, self.a, self.b, self.o = nn.LayerNorm(dx), nn.Linear(dx, do), nn.Linear(dx, do), nn.Linear(do * do, dp)

    def forward(self, x, pm):
        x = self.ln(x)
        op = torch.einsum("bic,bjd->bijcd", self.a(x), self.b(x)).flatten(-2)
        return self.o(op) * pm[..., None]


class TriangularUpdate(nn.Module):
    """outgoing + incoming multiplicative updates merged into one module (Uni-Mol+)."""
    def __init__(self, dp, dt):
        super().__init__()
        self.ln = nn.LayerNorm(dp)
        self.a, self.ag, self.b, self.bg = (nn.Linear(dp, dt) for _ in range(4))
        self.ln_o, self.o, self.g = nn.LayerNorm(dt), nn.Linear(dt, dp), nn.Linear(dp, dp)

    def forward(self, p, pm):
        z, m = self.ln(p), pm[..., None].to(p.dtype)
        a = torch.sigmoid(self.ag(z)) * self.a(z) * m
        b = torch.sigmoid(self.bg(z)) * self.b(z) * m
        t = torch.einsum("bikc,bjkc->bijc", a, b) + torch.einsum("bkic,bkjc->bijc", a, b)
        return torch.sigmoid(self.g(z)) * self.o(self.ln_o(t)) * m


class Block(nn.Module):
    def __init__(self, dx, dp, H, do, dt, drop):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(dx), PairBiasAttention(dx, dp, H, drop)
        self.ln2, self.ffx = nn.LayerNorm(dx), FFN(dx, dx, drop)
        self.opm, self.tri = OuterProduct(dx, dp, do), TriangularUpdate(dp, dt)
        self.lnp, self.ffp = nn.LayerNorm(dp), FFN(dp, dp, drop)

    def forward(self, x, p, mask, pm):
        x = x + self.attn(self.ln1(x), p, mask)
        x = x + self.ffx(self.ln2(x)) * mask[..., None]
        p = p + self.opm(x, pm)
        p = p + self.tri(p, pm)
        p = p + self.ffp(self.lnp(p)) * pm[..., None]
        return x, p


class PosHead(nn.Module):
    """delta_i = sum_h w_h sum_j a_hij v_hj (r_i - r_j)/|r_i - r_j|  -> E(3)-equivariant, translation invariant.
    The last layer starts at zero, so an untrained model returns its input (identity map)."""
    def __init__(self, dx, dp, H):
        super().__init__()
        self.H, self.dh = H, dx // H
        self.q, self.k, self.v = nn.Linear(dx, dx), nn.Linear(dx, dx), nn.Linear(dx, H)
        self.bias = nn.Linear(dp, H)
        self.w = nn.Linear(H, 1, bias=False)
        nn.init.zeros_(self.w.weight)

    def forward(self, x, p, pos, mask):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.H, self.dh)
        k = self.k(x).view(B, N, self.H, self.dh)
        lg = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh) + self.bias(p).permute(0, 3, 1, 2)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        lg = lg.masked_fill(~mask[:, None, None, :] | eye[None, None], float("-inf"))
        a = lg.softmax(-1)
        v = self.v(x).permute(0, 2, 1)                                   # (B, H, N)
        rel = pos[:, :, None, :] - pos[:, None, :, :]                    # r_i - r_j
        u = rel / torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-12)
        vec = torch.einsum("bhij,bijc->bihc", a * v[:, :, None, :], u)  # (B, N, H, 3)
        return self.w(vec.transpose(-1, -2)).squeeze(-1) * mask[..., None]


class GeoRefiner(nn.Module):
    def __init__(self, n_types=14, L=6, dx=256, dp=64, H=8, K=128, do=16, dt=32, drop=0.1, spd_max=9):
        super().__init__()
        self.V = n_types
        self.atom = nn.Embedding(n_types, dx)
        self.frag, self.formf, self.srcf, self.chg = (nn.Embedding(3, dx), nn.Embedding(2, dx),
                                                      nn.Embedding(2, dx), nn.Embedding(7, dx))
        self.gauss = Gaussian(K, n_types * n_types)
        self.p_lin = nn.Linear(K, dp)
        self.spd = nn.Embedding(spd_max + 1, dp)
        self.bond = nn.Embedding(3, dp)
        self.blocks = nn.ModuleList(Block(dx, dp, H, do, dt, drop) for _ in range(L))
        self.ln_f = nn.LayerNorm(dx)
        self.head = PosHead(dx, dp, H)

    def pair_init(self, pos, b):
        dist = torch.sqrt(((pos[:, :, None] - pos[:, None]) ** 2).sum(-1) + 1e-12)
        et = b["types"][:, :, None] * self.V + b["types"][:, None, :]
        return self.p_lin(self.gauss(dist, et)) + self.spd(b["spd"]) + self.bond(b["bond"])

    def forward(self, b, n_pass):
        mask = b["mask"]
        pm = mask[:, :, None] & mask[:, None, :]
        x0 = (self.atom(b["types"]) + self.frag(b["frag"]) + self.formf(b["form"])
              + self.srcf(b["src"])[:, None] + self.chg(b["chg"] + 3)[:, None])
        pos = b["x0"]
        for it in range(n_pass):
            p = self.pair_init(pos, b) * pm[..., None]
            x = x0
            for blk in self.blocks:
                x, p = blk(x, p, mask, pm)
            new = pos + self.head(self.ln_f(x), p, pos, mask)
            pos = new if it == n_pass - 1 else new.detach()      # loss only on the last pass (Uni-Mol+)
        return pos
```

## 8. Phase U — 단위 검사 (`analysis/geo/tests.py`, GPU 환경 job; 모두 통과해야 R로)

- `tests.py synth`: U-1–U-6, U-7s (레코드 없이). `tests.py data`: U-0, U-7–U-9 (D 이후).
- 이 검사 코드(synth 전체, data의 U-8·U-9)는 작성자가 로컬 CPU에서 가짜 데이터로 미리 돌려 통과를 확인했다(U-8은 C_compare 거리로 만든 가짜 좌표에서 840짝, 0.1595, 0.0155 Å 재현). HPC에서는 실제 데이터로 다시 돌린다.

| # | 검사 | 기준 |
|---|---|---|
| U-0 | GPU 환경에서 `records.pkl` 읽기 | 읽힘. 레코드 수 = `D_dataset_report.json`의 `n`. 실패하면(numpy 버전 차이) data.py가 배열을 `.tolist()`로 저장하고 train.py가 읽을 때 `np.asarray`로 되돌리도록 고친 뒤 기록 |
| U-1 | `kabsch_align`: 무작위 회전·이동 → 원래 구조; 거울상 | RMSD < 1e-8이고 `reflected=False`; 거울상 입력은 `reflect=True`에서 RMSD < 1e-8이고 `reflected=True`, `reflect=False`에서 `reflected=False` |
| U-2 | 초기화 직후 모델 = 항등 (`PosHead.w = 0`) | 출력 = 입력, 최대 차이 < 1e-10 |
| U-3 | 등변성: `w`에 무작위 값을 넣은 모델(float64)에서 입력을 회전(거울상 포함)·이동 → 출력도 같이 회전·이동 | 최대 오차 < 1e-8 Å |
| U-4 | 순서 등변성: 원자 순서를 섞으면 출력도 같이 섞임 (float64) | < 1e-8 Å |
| U-5 | padding: 크기가 다른 두 분자를 한 batch로 넣은 결과 = 각각 넣은 결과 (float64) | < 1e-8 Å |
| U-6 | 가짜 궤적 표집 10만 번 | q = 1 비율 0.10 ± 0.01, q = 0 비율 0.80 ± 0.01, 나머지는 [0.4, 0.6] |
| U-7s | 학습 가능성(가짜 분자 32개, 목표 = 중심 기준 3 % 확대, dropout 0, lr 1e-3, 600 step) | 최종 좌표 L1 < 0.3 × 시작값 (로컬 CPU 시험: 0.037 → 0.006 Å) |
| U-7 | 과적합: D1 학습 G1 샘플 32개, dropout 0, lr 1e-3, 2,000 step | 최종 좌표 L1 < 0.5 × 시작값. 시작값·최종값을 보고 |
| U-8 | 지표 회귀: 평가 함수(10-2절)로 **G2를 그대로 쓴 기준선**을 fold 0–4 시험 반응 전체에서 계산 | 같은 형태 para 짝 수 = **840**, 비율 = **0.1595 ± 0.005**, 중앙값 = **0.0155 ± 0.001 Å**(C_compare `d_g2_*`, `d_dft_*`로 미리 계산한 값, `GC.G2RAW_PRIMARY`). 반응 300개에서 레코드 좌표로 잰 형성 결합 거리 = C_compare `d_dft_1/2`, `d_g2_1/2`(정렬 비교) ± 1e-4 Å. 하나라도 벗어나면 지표나 데이터를 확인하고 STOP |
| U-9 | 누수 검사 (fold마다) | 시험·내부 검증 반응이 학습 목록에 0개; 겹침 D0가 학습에 0개; 학습 목록의 G2 샘플은 모두 `same is True`; 시험 예제의 입력 = G2 좌표(중심 이동만); 저장된 결합 그래프 = 입력 좌표로 다시 계산한 것 |

**U-8 기준값 설명:** 0.1595는 결합 짝을 **라벨 길이 순서(짧은 것·긴 것)**로 맞춘 값이다. 이전 보고서의 0.13(중앙값 0.0196 Å)은 `formed_pairs_ts`의 원자 번호 순서로 맞춘 값이어서 다르다. 이 명령서는 라벨 길이 순서를 쓴다.

```python
#!/usr/bin/env python3
"""Phase U: GEO-1 unit tests (GPU env).  python tests.py synth   (U-1..U-6, U-7s; no records needed)
                                          python tests.py data    (U-0, U-7, U-8, U-9; needs records.pkl)
Writes results/U_tests_<mode>.json; exit 3 if any test fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
import train as T  # noqa: E402
from metrics import evaluate_preds  # noqa: E402
from model import GeoRefiner  # noqa: E402

RES = {}


def check(name, ok, **info):
    RES[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


def fake_rec(n, rng, uid, rid):
    syms = list(rng.choice(["C", "N", "O", "H"], n))
    x = rng.normal(0.0, 1.6, (n, 3))
    pairs = np.array([[0, 1], [2, 3]])
    B = GC.bond_types(syms, x, pairs)
    return dict(uid=uid, src="D1", src_id=0, rxn_id=rid, core="fake/A", fold=0, n=n,
                types=np.array([GC.TYPE[s] for s in syms]), lab=x + rng.normal(0, 0.05, x.shape),
                frag=np.array([1] * (n // 2) + [2] * (n - n // 2)), form=np.array([1, 1, 1, 1] + [0] * (n - 4)),
                pairs=pairs, charge=0, same=True, overlap=False, xyz_g2=x, lab_on_g2=x + 0.03, bond_g2=B,
                spd_g2=GC.spd_matrix(B), xyz_g1=None)


def batch_of(exs, double=True):
    """T.collate, then (double=True) x0 / y rebuilt in float64 from the examples (collate stores float32)."""
    b = T.collate(exs)
    if double:
        N = b["x0"].shape[1]
        for key in ("x0", "y"):
            z = torch.zeros((len(exs), N, 3), dtype=torch.float64)
            for i, e in enumerate(exs):
                z[i, :len(e["types"])] = torch.as_tensor(e[key], dtype=torch.float64)
            b[key] = z.to(T.DEV)
    return b


def model_random_head(double=True):
    torch.manual_seed(1)
    m = GeoRefiner(**T.SIZE["S"], drop=0.0).to(T.DEV).eval()
    torch.nn.init.normal_(m.head.w.weight, std=0.5)
    return m.double() if double else m


def synth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    R, t = T.rand_rot(rng), np.array([3.0, -1.0, 2.0])
    _, r1, f1 = GC.kabsch_align(X, X @ R.T + t)
    M = (X @ R.T + t) * np.array([-1.0, 1.0, 1.0])
    _, r2, f2 = GC.kabsch_align(X, M, reflect=True)
    _, r3, f3 = GC.kabsch_align(X, M, reflect=False)
    check("U-1", r1 < 1e-8 and not f1 and r2 < 1e-8 and f2 and not f3, rmsd_rot=r1, rmsd_mirror=r2)
    recs = [fake_rec(n, rng, f"D1_{i}", i) for i, n in enumerate((14, 23))]
    exs = [T.make_example((r, "g2"), False, 0.0, "mix", rng) for r in recs]
    m0 = GeoRefiner(**T.SIZE["S"]).to(T.DEV).eval().double()
    b = batch_of(exs)
    with torch.no_grad():
        out = m0(b, 2)
    check("U-2", float((out - b["x0"]).abs().max()) < 1e-10, max_diff=float((out - b["x0"]).abs().max()))
    m = model_random_head()
    e = exs[0]
    with torch.no_grad():
        o1 = m(batch_of([e]), 2)[0].cpu().numpy()
        err = 0.0
        for refl in (False, True):
            Rm = T.rand_rot(rng) @ (np.diag([-1.0, 1.0, 1.0]) if refl else np.eye(3))
            e2 = dict(e, x0=e["x0"] @ Rm.T + t)
            o2 = m(batch_of([e2]), 2)[0].cpu().numpy()
            err = max(err, float(np.abs(o2 - (o1 @ Rm.T + t)).max()))
    check("U-3", err < 1e-8, max_err=err)
    n = len(e["types"])
    p = rng.permutation(n)
    inv = np.argsort(p)
    e3 = dict(e, types=e["types"][p], frag=e["frag"][p], form=e["form"][p], spd=e["spd"][np.ix_(p, p)],
              bond=e["bond"][np.ix_(p, p)], x0=e["x0"][p], y=e["y"][p], pairs=inv[e["pairs"]])
    with torch.no_grad():
        o3 = m(batch_of([e3]), 2)[0].cpu().numpy()
    check("U-4", float(np.abs(o3 - o1[p]).max()) < 1e-8, max_err=float(np.abs(o3 - o1[p]).max()))
    with torch.no_grad():
        ob = m(batch_of(exs), 2).cpu().numpy()
        oa = [m(batch_of([x]), 2)[0].cpu().numpy() for x in exs]
    err5 = max(float(np.abs(ob[i, :len(x["types"])] - oa[i]).max()) for i, x in enumerate(exs))
    check("U-5", err5 < 1e-8, max_err=err5)
    qs = np.array([T.sample_q(rng) for _ in range(100000)])
    mid = qs[(qs > 0) & (qs < 1)]
    check("U-6", abs((qs == 1).mean() - 0.1) < 0.01 and abs((qs == 0).mean() - 0.8) < 0.01
          and mid.min() >= 0.4 and mid.max() <= 0.6, q1=(qs == 1).mean(), q0=(qs == 0).mean())
    # U-7s: a learnable equivariant target (uniform 3 % expansion about the centroid), 32 random molecules
    items = []
    for i in range(32):
        r = fake_rec(int(rng.integers(12, 30)), rng, f"D1_{100 + i}", 100 + i)
        x = r["xyz_g2"]
        r.update(xyz_g1=x, lab_on_g1=x.mean(0) + 1.03 * (x - x.mean(0)), bond_g1=r["bond_g2"], spd_g1=r["spd_g2"])
        items.append((r, "g1"))
    start, end = overfit(items, 600)
    check("U-7s", end < 0.3 * start, start_L1=start, end_L1=end)


def overfit(items, n_steps, lr=1e-3):
    """train a dropout-free S model on the items (input = q 1, no noise); mean coordinate L1 before / after."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = GeoRefiner(**T.SIZE["S"], drop=0.0).to(T.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    def l1():
        P = T.predict(m, items, 2)
        return float(np.mean([np.abs(P[r["uid"]] - r[f"lab_on_{k}"]).mean() for r, k in items]))
    start, step = l1(), 0
    while step < n_steps:
        for _, b in T.batches(items, True, 0.0, "q1", rng, True):
            m.train()
            loss, _ = T.loss_fn(m(b, 2), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return start, l1()


def data():
    recs = T.load_records()
    rep = json.loads((GC.RES / "D_dataset_report.json").read_text())
    check("U-0", len(recs) == rep["n"], n=len(recs))
    # U-7 overfit
    tr, _, _ = T.split(recs, 0)
    items = [(r, "g1") for r in tr if r["src"] == "D1" and r["xyz_g1"] is not None][:32]
    start, end = overfit(items, 2000)
    check("U-7", end < 0.5 * start, start_L1=start, end_L1=end)
    # U-8 metric regression on the G2-raw baseline
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in (0, 1, 2, 3, 4) and r["xyz_g2"] is not None]
    mt = evaluate_preds({r["uid"]: r["xyz_g2"] for r in test}, test, with_rmsd=False)
    ex = GC.G2RAW_PRIMARY
    C = pd.read_csv(GC.C_COMPARE, usecols=["rxn_id", "d_dft_1", "d_dft_2", "d_g2_1", "d_g2_2"]).set_index("rxn_id")
    dev = 0.0
    for r in test[:300]:
        dl = sorted(GC.form_d(r["lab"], r["pairs"]))
        dg = sorted(GC.form_d(r["xyz_g2"], r["pairs"]))
        c = C.loc[r["rxn_id"]]
        dev = max(dev, float(np.abs(np.array(dl) - np.sort([c.d_dft_1, c.d_dft_2])).max()),
                  float(np.abs(np.array(dg) - np.sort([c.d_g2_1, c.d_g2_2])).max()))
    check("U-8", mt["n_primary"] == ex["n"] and abs(mt["primary_share"] - ex["share"]) <= 0.005
          and abs(mt["pair_pe_median_same_para"] - ex["median"]) <= 0.001 and dev <= 1e-4,
          n=mt["n_primary"], share=mt["primary_share"], median=mt["pair_pe_median_same_para"], max_dist_dev=dev)
    # U-9 leakage
    bad = []
    for k in range(5):
        tr, va, te = T.split(recs, k)
        tr_ids = {r["uid"] for r in tr}
        if tr_ids & {r["uid"] for r in te + va}:
            bad.append(f"fold {k}: test/inner-val reaction in train")
        if any(r["src"] == "D0" and r["overlap"] for r in tr):
            bad.append(f"fold {k}: overlapping D0 in train")
        if any(kind == "g2" and r["same"] is not True for r, kind in T.train_items(tr)):
            bad.append(f"fold {k}: non-same G2 sample in train")
        for r, kind in T.eval_items(te)[:200]:
            e = T.make_example((r, kind), False, 0.0, "mix", np.random.default_rng(0))
            if not np.allclose(e["x0"] + e["center"], r["xyz_g2"], atol=1e-10, rtol=0):
                bad.append(f"fold {k}: test input != G2 for {r['uid']}")
            syms = [GC.ELEM[t] for t in r["types"]]
            if not np.array_equal(GC.bond_types(syms, r["xyz_g2"], r["pairs"]), r["bond_g2"]):
                bad.append(f"fold {k}: bond graph not from the input for {r['uid']}")
    check("U-9", not bad, problems=bad[:20])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"synth": synth, "data": data}.get(mode, lambda: sys.exit(__doc__))()
    GC.write_json(GC.RES / f"U_tests_{mode}.json", RES)
    sys.exit(0 if all(v["ok"] for v in RES.values()) else 3)
```

## 9. Phase R — 학습 (`analysis/geo/train.py`, GPU 환경 = env_d1fu2)

### 9-1. 분할 (누수 방지; 바꾸지 않음)
- 바깥 fold k ∈ {0, …, 4} = D1 T1 fold (`core_fold_map`)
  - **시험:** D1 반응 중 `fold == k`, 입력은 G2만(형태 분류와 상관없이 전부)
  - **내부 검증:** D1 `fold == (k + 1) % 5`, 입력 G2. 설정과 epoch 선택에만 쓴다
  - **학습:** 그 밖의 D1(`fold == −1`, 즉 D1PAIR T1에서 시험 core로 쓰지 않은 6개 core의 380반응 포함) + D0(`overlap == False`)
- **학습 샘플:**
  - (G2 → 라벨): `same == True`인 반응만
  - (G1 → 라벨): G1이 있는 모든 반응
  - **다른 형태의 G2 짝은 학습하지 않는다.** 형태가 다른 좌표를 섞으면 비물리적 중간 구조가 생기기 때문이다.
- **시험 성적은 어떤 선택에도 쓰지 않는다.**

### 9-2. 설정 격자 (fold마다 따로 선택)

| cfg | 크기 (L, dx, dp, H, do, dt) | 가짜 궤적 잡음 σ (Å) | R (반복) → n_pass = R + 1 |
|---|---|---|---|
| c0 | S = (6, 256, 64, 8, 16, 32) | 0.05 | 1 |
| c1 | S | 0.05 | 3 |
| c2 | S | 0.20 | 1 |
| c3 | S | 0.20 | 3 |

**공통 설정**
- AdamW(lr 2e-4, betas (0.9, 0.999), weight decay 1e-4), warmup 5 % 후 cosine, grad clip 5.0, EMA 0.999 (평가는 EMA 가중치)
- batch 16, 최대 100 epoch, 조기 종료 patience 15 (내부 검증 주 지표)
- float32, `num_workers=0`
- 가짜 궤적: q = 1 (10 %), 0 (80 %), U[0.4, 0.6] (10 %). 입력 = q·x_in + (1 − q)·(y + N(0, σ²)). 학습 때 무작위 회전(입력과 정답에 같이)
- 손실 = 좌표 L1 + 4 Å 이내 원자 간 거리 L1 + **10 × 형성 결합 거리 L1**
- 모델 크기는 S 하나로 고정한다. 더 큰 모델은 이번 결과를 본 뒤 사용자가 정한다.

### 9-3. 코드

```python
#!/usr/bin/env python3
"""Phase R: GEO-1 training.
  python train.py smoke <k> | tune <k> <cfg> | select <k> | final <k> <seed> | ablation <k> <no_d0|q1_only|r0>"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from model import GeoRefiner  # noqa: E402
from metrics import evaluate_preds  # noqa: E402     (10-2)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = dict(S=dict(L=6, dx=256, dp=64, H=8, do=16, dt=32))
CFGS = {"c0": ("S", 0.05, 1), "c1": ("S", 0.05, 3), "c2": ("S", 0.20, 1), "c3": ("S", 0.20, 3)}
BS, LR, MAX_EP, PATIENCE, W_FORM, LOCAL_CUT = 16, 2e-4, 100, 15, 10.0, 4.0


def load_records():
    with open(GC.GEO / "data" / "records.pkl", "rb") as f:
        return pickle.load(f)


def split(recs, k):
    inner = (k + 1) % 5
    test = [r for r in recs if r["src"] == "D1" and r["fold"] == k]
    ival = [r for r in recs if r["src"] == "D1" and r["fold"] == inner]
    train = [r for r in recs if (r["src"] == "D1" and r["fold"] not in (k, inner))
             or (r["src"] == "D0" and not r["overlap"])]
    return train, ival, test


def train_items(recs, use_d0=True):
    S = []
    for r in recs:
        if r["src"] == "D0" and not use_d0:
            continue
        if r["xyz_g2"] is not None and r["same"] is True:
            S.append((r, "g2"))
        if r["xyz_g1"] is not None:
            S.append((r, "g1"))
    return S


def eval_items(recs):
    return [(r, "g2") for r in recs if r["xyz_g2"] is not None]


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def sample_q(rng):
    u = rng.random()
    return 1.0 if u < 0.1 else (0.0 if u < 0.9 else float(rng.uniform(0.4, 0.6)))


def make_example(item, train, sigma, q_mode, rng):
    r, kind = item
    x_in, y = r[f"xyz_{kind}"], r[f"lab_on_{kind}"]
    if train:
        q = 1.0 if q_mode == "q1" else sample_q(rng)
        x0 = q * x_in + (1.0 - q) * (y + rng.normal(0.0, sigma, y.shape))
        c, R = x0.mean(0), rand_rot(rng)
        x0, y = (x0 - c) @ R.T, (y - c) @ R.T
    else:
        c = x_in.mean(0)
        x0, y = x_in - c, y - c
    return dict(types=r["types"], frag=r["frag"], form=r["form"], src=r["src_id"], chg=r["charge"],
                spd=r[f"spd_{kind}"], bond=r[f"bond_{kind}"], x0=x0, y=y, pairs=r["pairs"], center=c, uid=r["uid"])


def collate(exs):
    B, N = len(exs), max(len(e["types"]) for e in exs)
    t = {k: torch.zeros((B, N), dtype=torch.long) for k in ("types", "frag", "form")}
    spd, bond = torch.zeros((B, N, N), dtype=torch.long), torch.zeros((B, N, N), dtype=torch.long)
    x0, y = torch.zeros((B, N, 3)), torch.zeros((B, N, 3))
    mask = torch.zeros((B, N), dtype=torch.bool)
    for i, e in enumerate(exs):
        n = len(e["types"])
        for k in t:
            t[k][i, :n] = torch.as_tensor(e[k])
        spd[i, :n, :n], bond[i, :n, :n] = torch.as_tensor(e["spd"]), torch.as_tensor(e["bond"])
        x0[i, :n], y[i, :n] = torch.as_tensor(e["x0"], dtype=torch.float32), torch.as_tensor(e["y"], dtype=torch.float32)
        mask[i, :n] = True
    b = dict(**t, spd=spd, bond=bond, x0=x0, y=y, mask=mask,
             src=torch.tensor([e["src"] for e in exs]), chg=torch.tensor([e["chg"] for e in exs]).clamp(-3, 3),
             pairs=torch.tensor(np.stack([e["pairs"] for e in exs])))
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def batches(items, train, sigma, q_mode, rng, shuffle):
    idx = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for s in range(0, len(idx), BS):
        exs = [make_example(items[i], train, sigma, q_mode, rng) for i in idx[s:s + BS]]
        yield exs, collate(exs)


def pdist(x):
    return torch.sqrt(((x[:, :, None] - x[:, None]) ** 2).sum(-1) + 1e-12)


def loss_fn(pred, b):
    m = b["mask"]
    mf = m[..., None].float()
    l_coord = ((pred - b["y"]).abs() * mf).sum() / (mf.sum() * 3)
    dp, dl = pdist(pred), pdist(b["y"])
    eye = torch.eye(m.shape[1], dtype=torch.bool, device=m.device)
    pm = m[:, :, None] & m[:, None, :] & ~eye[None] & (dl < LOCAL_CUT)
    l_local = ((dp - dl).abs() * pm).sum() / pm.sum()
    bi = torch.arange(m.shape[0], device=m.device)[:, None]
    i, j = b["pairs"][..., 0], b["pairs"][..., 1]
    df_p = torch.sqrt(((pred[bi, i] - pred[bi, j]) ** 2).sum(-1) + 1e-12)
    df_l = torch.sqrt(((b["y"][bi, i] - b["y"][bi, j]) ** 2).sum(-1) + 1e-12)
    l_form = (df_p - df_l).abs().mean()
    return l_coord + l_local + W_FORM * l_form, dict(coord=l_coord.item(), local=l_local.item(), form=l_form.item())


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def model(self, template):
        m = GeoRefiner(**template)
        m.load_state_dict(self.shadow)
        return m.to(DEV).eval()


@torch.no_grad()
def predict(model, items, n_pass):
    model.eval()
    out, rng = {}, np.random.default_rng(0)
    for exs, b in batches(items, False, 0.0, "mix", rng, False):
        pred = model(b, n_pass).cpu().numpy()
        for i, e in enumerate(exs):
            n = len(e["types"])
            out[e["uid"]] = pred[i, :n] + e["center"]          # back to the input frame
    return out


def fit(tr, va, cfg_name, seed, fixed_epochs=None, q_mode="mix", n_pass_override=None, tag=""):
    size, sigma, R = CFGS[cfg_name]
    n_pass = n_pass_override or (R + 1)
    arch = dict(SIZE[size])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GeoRefiner(**arch).to(DEV)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
    n_ep = fixed_epochs or MAX_EP
    total = MAX_EP * math.ceil(len(tr) / BS)    # same LR horizon in tune and final (final just stops at n_ep)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist, best, best_ep, t0 = [], None, -1, time.time()
    for ep in range(n_ep):
        model.train()
        agg = []
        for _, b in batches(tr, True, sigma, q_mode, rng, True):
            loss, parts = loss_fn(model(b, n_pass), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            agg.append(parts)
        h = dict(epoch=ep, **{k: float(np.mean([a[k] for a in agg])) for k in agg[0]}, wall_s=time.time() - t0)
        if va is not None:
            m = evaluate_preds(predict(ema.model(arch), va, n_pass), [r for r, _ in va])
            h.update(val_primary=m["primary_share"], val_rxn_median=m["rxn_form_abs_median"])
            key = (m["primary_share"], -m["rxn_form_abs_median"])
            if best is None or key > best:
                best, best_ep = key, ep
                torch.save(ema.shadow, GC.GEO / "runs" / f"{tag}.best.pt")
            if ep - best_ep >= PATIENCE:
                hist.append(h)
                break
        hist.append(h)
    if va is None:
        torch.save(ema.shadow, GC.GEO / "runs" / f"{tag}.final.pt")
    GC.write_json(GC.GEO / "runs" / f"{tag}.history.json",
                  dict(cfg=cfg_name, seed=seed, n_pass=n_pass, q_mode=q_mode, best_ep=best_ep, best=best, hist=hist,
                       n_train=len(tr), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    return ema, arch, n_pass, best_ep


def main(a):
    (GC.GEO / "runs").mkdir(parents=True, exist_ok=True)
    (GC.GEO / "preds").mkdir(parents=True, exist_ok=True)
    recs = load_records()
    cmd, k = a[0], int(a[1])
    tr_r, va_r, te_r = split(recs, k)
    if cmd == "smoke":                                    # timing only (9-4): 2 epochs of c0 (R=1) and c1 (R=3)
        for c in ("c0", "c1"):
            fit(train_items(tr_r), eval_items(va_r), c, 0, fixed_epochs=2, tag=f"smoke_f{k}_{c}")
        return 0
    if cmd == "tune":
        fit(train_items(tr_r), eval_items(va_r), a[2], 0, tag=f"tune_f{k}_{a[2]}")
        return 0
    if cmd == "select":
        H = {c: json.loads((GC.GEO / "runs" / f"tune_f{k}_{c}.history.json").read_text())
             for c in CFGS if (GC.GEO / "runs" / f"tune_f{k}_{c}.history.json").is_file()}
        best = max(H, key=lambda c: tuple(H[c]["best"]))
        GC.write_json(GC.RES / f"R_select_f{k}.json",
                      dict(fold=k, chosen=best, best_ep=H[best]["best_ep"], candidates={c: H[c]["best"] for c in H}))
        return 0
    sel = json.loads((GC.RES / f"R_select_f{k}.json").read_text())
    cfg, ep = sel["chosen"], sel["best_ep"] + 1
    full = tr_r + va_r                                    # final models: train + inner validation, fixed epochs
    if cmd == "final":
        seed = int(a[2])
        ema, arch, n_pass, _ = fit(train_items(full), None, cfg, seed, fixed_epochs=ep, tag=f"final_f{k}_s{seed}")
        P = predict(ema.model(arch), eval_items(te_r), n_pass)
        with open(GC.GEO / "preds" / f"final_f{k}_s{seed}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
        return 0
    if cmd == "ablation":
        name = a[2]
        kw = dict(no_d0=dict(use_d0=False), q1_only=dict(q_mode="q1"), r0=dict(n_pass_override=1))[name]
        items = train_items(full, use_d0=kw.pop("use_d0", True))
        ema, arch, n_pass, _ = fit(items, None, cfg, 0, fixed_epochs=ep, tag=f"abl_{name}_f{k}", **kw)
        P = predict(ema.model(arch), eval_items(te_r), n_pass)
        with open(GC.GEO / "preds" / f"abl_{name}_f{k}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

### 9-4. 실행 순서와 GPU 예산
1. **smoke:** `train.py smoke 0` → c0(R = 1)과 c1(R = 3)을 각각 2 epoch 돌려 epoch당 시간 t1, t3(검증 포함)를 잰다(`runs/smoke_f0_c*.history.json`의 `wall_s`).
   - **상한 예상** = 100 epoch × (10·t1 + 10·t3 [tune] + 15·t3 [final 상한] + 15·t3 [ablation 상한]).
   - 200 GPU시간 이하면 진행한다.
   - 넘으면 **미리 승인된 축소**를 순서대로 적용하고 다시 계산한다: ① `MAX_EP` 100 → 60, `PATIENCE` 15 → 10; ② ablation을 fold 0–2에서만. 그래도 넘으면 STOP. 적용한 축소는 `DECISIONS.md`에 적는다.
2. `tune k c` (k = 0..4, c = c0..c3): 20 job
3. `select k`: 5회 (CPU에서 돌려도 된다)
4. `final k s` (s = 0, 1, 2): 15 job
5. `ablation k {no_d0, q1_only, r0}`: 15 job
6. 동시에 도는 job은 MaxJobs 10 이내, 제출 대기는 MaxSubmit 20 이내로 나눠 낸다.

제출 예: `sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 --time=48:00:00 -J geo_tune jobs/gpu.slurm train.py tune 0 c0`

`analysis/geo/jobs/gpu.slurm` (`analysis/d1_ladder/jobs/gpu.slurm`과 같은 환경):

```bash
#!/bin/bash
#SBATCH --job-name=geo_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/geo/logs/%x_%j.out
# submit from analysis/geo/:  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 jobs/gpu.slurm <script.py> <args...>
# CPU-only use (select / evaluate):  sbatch -p <cpu partition> jobs/gpu.slurm <script.py> <args...>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
ENV=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$ENV/.d1fu2_env_ok" ] || { echo "STOP: env_d1fu2 not built ($ENV/.d1fu2_env_ok missing)"; exit 2; }
export PY=$ENV/bin/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GEO_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo D0G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d0_g2
gpu=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[geo_gpu] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD 2>/dev/null)" \
     "$($PY --version 2>&1) gpu=${gpu:-none} $(date '+%F %T')"
"$PY" "$@"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
```

## 10. Phase E — 평가

### 10-1. 원칙
- 시험 = 각 fold의 D1 시험 반응. 입력은 **G2만**(q = 1, 잡음 없음).
- 최종 예측 = seed 3개의 좌표 평균. 같은 입력 좌표계라 평균할 수 있다.
- 지표는 거리 기반(회전 무관)과 Kabsch RMSD(거울상 허용)다.

### 10-2. 지표 코드 (`analysis/geo/metrics.py`)

```python
"""GEO-1 metrics. Bond identity across a pair = rank by LABEL length (short, long), fixed before results."""
from __future__ import annotations

import numpy as np
import pandas as pd

import geo_common as GC

_FP = None


def f_pairs():
    global _FP
    if _FP is None:
        F = pd.read_csv(GC.F_PAIRS, usecols=["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main"])
        _FP = F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].copy()
    return _FP


def rxn_errors(pred, lab, pairs):
    dl = np.array([np.linalg.norm(lab[i] - lab[j]) for i, j in pairs])
    dp = np.array([np.linalg.norm(pred[i] - pred[j]) for i, j in pairs])
    o = np.argsort(dl)
    return dp[o] - dl[o]                                  # (e_short, e_long)


def pair_class(sa, sb):
    if sa is True and sb is True:
        return "same"
    if sa is False and sb is False:
        return "different"
    return "mixed"                                        # one True / one False, or a None


def evaluate_preds(P, recs, with_rmsd=True):
    """P: uid -> predicted xyz (any frame); recs: records of the evaluated reactions (D1 only).
    Primary metric (spec 12): share of 'same'-class para P_main reference pairs with pair error <= 0.005 A, where
    pair error = max over (short, long) of |e_a - e_b| and e = predicted - label forming distance."""
    R = {r["rxn_id"]: r for r in recs if r["src"] == "D1" and r["uid"] in P}
    E, rows = {}, []
    for rid, r in R.items():
        e = rxn_errors(P[r["uid"]], r["lab"], r["pairs"])
        E[rid] = e
        row = dict(rxn_id=rid, core=r["core"], e_short=float(e[0]), e_long=float(e[1]),
                   abs_max=float(np.abs(e).max()), same=r["same"])
        if with_rmsd:
            heavy = r["types"] != 0
            row["rmsd_heavy"] = GC.kabsch_align(P[r["uid"]][heavy], r["lab"][heavy], reflect=True)[1]
        rows.append(row)
    F = f_pairs()
    F = F[F.a_rxn.isin(E) & F.b_rxn.isin(E)].copy()
    F["pe"] = [float(np.max(np.abs(E[a] - E[b]))) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["cls"] = [pair_class(R[a]["same"], R[b]["same"]) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["same"] = F.cls == "same"
    F["core"] = [R[a]["core"] for a in F.a_rxn]
    F["dipole"] = [R[a]["core"].split("/")[0] for a in F.a_rxn]
    prim = F[(F.pos == "p") & F.same]
    D = pd.DataFrame(rows)
    if not with_rmsd:
        D["rmsd_heavy"] = np.nan
    return dict(primary_share=float((prim.pe <= 0.005).mean()) if len(prim) else float("nan"), n_primary=int(len(prim)),
                all_pairs_share=float((F.pe <= 0.005).mean()) if len(F) else float("nan"), n_pairs=int(len(F)),
                pair_pe_median_same_para=float(prim.pe.median()) if len(prim) else float("nan"),
                rxn_form_abs_median=float(np.median(np.abs(np.concatenate([D.e_short, D.e_long])))),
                rmsd_heavy_median=float(D.rmsd_heavy.median()), table_pairs=F, table_rxn=D)
```

### 10-3. 평가 실행 (`analysis/geo/evaluate.py`)

| 비교 대상 | 예측 |
|---|---|
| **모델 (seed 3개 앙상블)** | `final_f{k}_s{0,1,2}.pkl` 좌표 평균 |
| 기준선: G2 그대로 | 입력 G2 좌표 |
| 제거 실험 | `abl_no_d0`, `abl_q1_only`, `abl_r0` |
| seed 간 차이 | seed별 지표 |

**출력**
- `results/E_main.csv`: 대상 × 지표
- `results/E_pairs.csv`, `E_rxn.csv`: 짝·반응별 오차
- `results/E_by_class.csv`: same / mixed / different 분류별
- `results/E_by_dipole.csv`
- `results/E_report.json`

**신뢰구간:** 주 지표는 core 단위 bootstrap 2,000회(seed 20261026, 시험 core를 복원 추출)로 95 % CI를 낸다. 모델과 기준선의 **차이**도 같은 bootstrap 표본에서 짝지어 계산한다.

**ablation을 일부 fold에서만 돌린 경우(9-4 축소 ②):** 그 ablation과의 비교는 같은 fold의 모델 결과로만 한다.

```python
#!/usr/bin/env python3
"""Phase E: GEO-1 evaluation on the D1 test folds (G2 input only). python evaluate.py"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds  # noqa: E402

B_BOOT, SEED, THR = 2000, 20261026, 0.005


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def preds_for(name, folds):
    """uid -> xyz for the given test folds; None if a fold file is missing."""
    P = {}
    for k in folds:
        if name == "model":
            S = [GC.GEO / "preds" / f"final_f{k}_s{s}.pkl" for s in (0, 1, 2)]
            if not all(p.is_file() for p in S):
                return None
            parts = [load_pkl(p) for p in S]
            assert all(set(q) == set(parts[0]) for q in parts), f"seed prediction sets differ in fold {k}"
            P.update({u: np.mean([q[u] for q in parts], axis=0) for u in parts[0]})   # same input frame -> averageable
        elif name.startswith("seed"):
            p = GC.GEO / "preds" / f"final_f{k}_s{name[4:]}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
        else:
            p = GC.GEO / "preds" / f"abl_{name}_f{k}.pkl"
            if not p.is_file():
                return None
            P.update(load_pkl(p))
    return P


def core_counts(F):
    """per core: (#primary pairs with pe <= THR, #primary pairs)."""
    prim = F[(F.pos == "p") & F.same]
    g = prim.groupby("core").pe
    return pd.DataFrame(dict(ok=g.apply(lambda s: int((s <= THR).sum())), n=g.size()))


def boot_diff(Fa, Fb, rng):
    """core bootstrap of share(a), share(b), share(a) - share(b) on the same primary pairs."""
    A, B = core_counts(Fa), core_counts(Fb)
    assert A.index.equals(B.index) and (A.n == B.n).all(), "primary pair sets differ between the compared targets"
    cores = A.index.to_numpy()
    out = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(cores), len(cores), replace=True)
        na = A.n.to_numpy()[pick].sum()
        sa, sb = A.ok.to_numpy()[pick].sum() / na, B.ok.to_numpy()[pick].sum() / na
        out.append((sa, sb, sa - sb))
    q = np.percentile(np.array(out), [2.5, 97.5], axis=0)
    return dict(ci_a=list(q[:, 0]), ci_b=list(q[:, 1]), ci_diff=list(q[:, 2]))


def main():
    with open(GC.GEO / "data" / "records.pkl", "rb") as f:
        recs = pickle.load(f)
    folds = [0, 1, 2, 3, 4]
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in folds and r["xyz_g2"] is not None]
    by_fold = {k: [r for r in test if r["fold"] == k] for k in folds}
    targets = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
    for name in ("model", "seed0", "seed1", "seed2", "no_d0", "q1_only", "r0"):
        avail = [k for k in folds if preds_for(name, [k]) is not None]
        if avail:
            targets[name] = preds_for(name, avail)
    if "model" not in targets or len(targets["model"]) != len(test):
        print("STOP: model predictions missing for some test reactions", file=sys.stderr)
        return 3
    res, tabs = {}, {}
    for name, P in targets.items():
        uids = set(P)
        m = evaluate_preds(P, [r for r in test if r["uid"] in uids])
        tabs[name] = m
        res[name] = {k: v for k, v in m.items() if not k.startswith("table_")}
        res[name]["folds"] = sorted({r["fold"] for r in test if r["uid"] in uids})
    rng = np.random.default_rng(SEED)
    ci = {"model_vs_g2_raw": boot_diff(tabs["model"]["table_pairs"], tabs["g2_raw"]["table_pairs"], rng)}
    for name in ("no_d0", "q1_only", "r0"):
        if name in tabs:
            fl = res[name]["folds"]
            Pm = {u: x for u, x in targets["model"].items() if int(u.split("_")[1]) in
                  {r["rxn_id"] for k in fl for r in by_fold[k]}}
            mm = evaluate_preds(Pm, [r for k in fl for r in by_fold[k]], with_rmsd=False)
            ci[f"model_vs_{name}"] = dict(folds=fl, model_share=mm["primary_share"],
                                          other_share=res[name]["primary_share"],
                                          **boot_diff(mm["table_pairs"], tabs[name]["table_pairs"], rng))
    s_m, d_lo = res["model"]["primary_share"], ci["model_vs_g2_raw"]["ci_diff"][0]
    verdict = "pass" if (s_m >= 0.5 and d_lo > 0) else ("partial" if d_lo > 0 else "fail")
    GC.RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).T.to_csv(GC.RES / "E_main.csv")
    tabs["model"]["table_pairs"].merge(tabs["g2_raw"]["table_pairs"][["pair_id", "pe"]], on="pair_id",
                                       suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_pairs.csv", index=False)
    tabs["model"]["table_rxn"].merge(tabs["g2_raw"]["table_rxn"][["rxn_id", "e_short", "e_long", "rmsd_heavy"]],
                                     on="rxn_id", suffixes=("_model", "_g2raw")).to_csv(GC.RES / "E_rxn.csv", index=False)
    rows_c, rows_d = [], []
    for name in ("model", "g2_raw"):
        F = tabs[name]["table_pairs"]
        for (pos, cls), g in F.groupby(["pos", "cls"]):
            rows_c.append(dict(target=name, pos=pos, cls=cls, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
        for dip, g in F[(F.pos == "p") & F.same].groupby("dipole"):
            rows_d.append(dict(target=name, dipole=dip, n=len(g), share=float((g.pe <= THR).mean()),
                               median=float(g.pe.median())))
    pd.DataFrame(rows_c).to_csv(GC.RES / "E_by_class.csv", index=False)
    pd.DataFrame(rows_d).to_csv(GC.RES / "E_by_dipole.csv", index=False)
    GC.write_json(GC.RES / "E_report.json", dict(verdict=verdict, primary=res, ci=ci, n_test=len(test),
                                                 g2_raw_expected=GC.G2RAW_PRIMARY, thr_A=THR, n_boot=B_BOOT))
    print(f"verdict {verdict}: model {s_m:.3f} vs g2_raw {res['g2_raw']['primary_share']:.3f}, "
          f"diff CI {ci['model_vs_g2_raw']['ci_diff']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 11. Phase S — `results/SUMMARY_GEO1.md` (한국어) 후 **멈춤**

- **맨 앞 네 문장:**
  1. 주 지표(모델 대 G2 그대로)와 95 % CI
  2. 판정(12절)
  3. D0가 도움이 됐는지(제거 실험)
  4. D0-G2 계산 결과 요약(성공률, 코어시간)
- **절:** 시작 조건 → I → Z → D → U → R(선택된 설정, 학습 곡선) → E(표) → 판정 → 한계
- **꼭 적을 한계:**
  - 다른 형태 G2는 학습하지 않았다.
  - 정밀도 바닥: 라벨 방법 그대로 재최적화해도 0.0036 Å 차이가 난다(D1-G3L-C).
  - Gaussian(D0)과 ORCA(D1)의 B3LYP 정의 차이를 출처 표시로만 흡수했다.
  - 에너지 단계는 하지 않았다.
- commit·push 후 **다음 단계(에너지)는 사용자 지시를 기다린다.**

## 12. 판정 (결과 보기 전에 고정)

| 판정 | 조건 (시험 fold 합산, 모델 앙상블) |
|---|---|
| **통과** | 같은 형태 para 짝의 짝 단위 형성 결합 오차 ≤ 0.005 Å 비율 ≥ **0.5**, 그리고 (모델 − G2 그대로) 차이의 core bootstrap 95 % CI 하한 > 0 |
| 부분 | 차이의 CI 하한 > 0이지만 비율 < 0.5 |
| 불가 | 차이의 CI 하한 ≤ 0 |

- G2 그대로의 기대값: 840짝, 비율 0.1595, 중앙값 0.0155 Å(U-8에서 재현 확인).

- 보조 판정은 문장으로만 쓴다:
  - D0 효과: `abl_no_d0` 대비 주 지표 차이와 CI
  - 가짜 궤적 효과: `abl_q1_only` 대비
  - 반복 효과: `abl_r0` 대비

## 13. 계산량·상한

| 항목 | 상한 | 근거 |
|---|---|---|
| D0-G2 (CPU) | 4,000코어시간 | D1-G2 반응당 중앙값 0.253, 평균 0.336코어시간(C_compare `g2_core_h_total`, 실패 포함; D1 원자 수 중앙값 35). D0는 원자 수 중앙값 44, 최대 94로 더 크다(labels_all). 5,260 × 0.34–0.55 ≈ 1,800–2,900 예상. pilot로 다시 예상 |
| GPU | 200 GPU시간 | smoke로 다시 예상 |
| 데이터 build (CPU) | 소량 | — |

## 14. HPC 규칙
- python, ORCA, xtb, find/du는 모두 `sbatch`(`--time=48:00:00`)로만 돌린다. 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만 한다.
- MaxJobs 10, MaxSubmit 20(배열 원소 포함).
- **빈 할당을 줄인다:** pool·학습 job은 할 일이 없으면 바로 끝난다. job마다 `sacct` CPUTimeRAW와 기록 시간을 `results/HPC_efficiency.csv`에 적는다.
- **읽기만 하는 것:**
  - D1·D1-G2·D1PAIR·D1-HF·D1LADDER·D1-G3·D1-G3L 결과와 scratch
  - `espley_*`, `eda_asm_raw`, `d1/`
- **새 파일은 여기에만 쓴다:** `analysis/geo/`, `$GEO`, `$D0G2`, 이 명령서
- 원자적 쓰기, 멱등. 수치는 파일에서 옮긴다.

## 15. 결정 기록과 STOP

- 명령서에 없는 갈림길이 생기면 추천안을 고르고 `results/DECISIONS.md`에 적은 뒤 진행한다.
- 판정(12절), 분할(9-1절), 설정 격자(9-2절)는 바꾸지 않는다. 9-4의 미리 승인된 예산 축소만 예외다.

| 단계 | STOP 조건 |
|---|---|
| 0 | 2절 시작 조건 중 하나라도 아님 (특히 0-2 D0 G1 보존) |
| I | 4절 검사 중 ✗ |
| Z | Z-1 실패(환경 변화); Z-2 키워드 차이; pilot 성공률 < 0.7; 예상 > 4,000코어시간; 파이프라인 오류 > 5 % |
| D | `same_ts` 재현 < 0.99; 원소 순서 불일치가 반응의 1 % 초과 |
| U | U-1–U-9 중 하나라도 실패 |
| R | 예상 GPU > 200시간; NaN 손실; 학습 목록에 시험 반응이 섞임 |
| E | 모델과 기준선의 시험 반응 목록이 다름 |
| 언제든 | 라벨·학습 결과 파일을 바꿔야 할 것 같음; 이 문서에 없는 계산 방법 변경이 필요해 보임 |

STOP하면 무엇이 어긋났는지, 해당 파일과 줄, 가능한 선택지를 적어 보고하고 기다린다.
