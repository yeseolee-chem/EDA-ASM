#!/usr/bin/env python3
"""ESPB 1: start conditions 1-1 .. 1-9 (docs/specs/ESPB.md 1) -> results/S0_start.json.

    python analysis/espb/s0_start.py <free_gb> <method>      env_espley python; job: analysis/espb/jobs/s0.slurm

The job fetches origin/gnn3r2 and measures the free disk (compute nodes cannot always run mmlsquota) and passes it.
Also checks that the constants written again in eb_common.py equal d1pair_common / train_ml_single / r6_nn (spec 2-3).
Exit 0 all pass, 3 = STOP.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eb_common as C  # noqa: E402

BASE = "f6d9ee39"
RES_FILES = {"F_pairs_r.csv": "9d75fae8b7ba39ab7d83112342a7ddd299017ad7eb3b05f97a4409deec4f3455",
             "S_folds_r.csv": "0f0895611d42c276bcf28e0346027aac33c7c0c8454b4e64a4366e7cd7ce278a",
             "S_splits_r.json": "8b660f9fb8f3e3d794b166d18c0e93bce50a1c3ae93f0f4de3fe57fbfc260ef5",
             "B_label_report.json": "a3e30bf438d21aa817f317903135e06cc7ef9ff81ae464d4c8c361e2405d5007",
             "eval/metrics.csv": "9d176e94eac49bc12912ffa2c3f1d27f2d61352bce7a159e3727dacabbe30cc1"}
SPLIT_SHA = {0: "b4353087b159dd1e8ef92d570c6907288729c42f4701c4e0df85979b8aada835",
             1: "3de69f5ad84174beb3ee444f2d119fec41ccea4c3cec50f8f1dbc2e08dbd0d0e",
             2: "92d9ddac1c40883a9ee8ceacf70c25378e3ebe3d416f1690040b506d61d6a84d",
             3: "887fb7a50e127f7a14d336a818b05b54f3cdf327067f9b32113bdc424f4ad17d",
             4: "deeb6cb4fcce2afb4005a98eac2cd2690d5e4df04fcf31fc101c1cbf0c6f6452"}
CV_SHA = {0: "5fcbdad85b7b96ce202e2412ecb2ba45069e82c85504c434671ad4a73333db29",
          1: "7c40ec36a26a9887887f86ca0e547ea2aedc53a81953858199cc7d6620d830a3",
          2: "843a44eace18ead6d4556d1866b62899af2f56226389a67d08e1606c4b765543",
          3: "c852ccdd4b288938a94db59d7529f99ce7fcddf0486a72edee7137b4669c482d",
          4: "3f9949c9b391e1c09a01799455c100d9cabc33556d254aba076b97441cb1aabe"}
X182_SHA = "66c40316319870214e4773b5a838af380d90220f1318536d5b6ac7524955754d"
CR2_SHA = "7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755"
CODE_SHA = {"analysis/d1_pair/d1_train.py": "414b5cc66762391691b252235c9133f0d193312a8039d35369299a8dfe30027c",
            "analysis/d1_pair/d1_f0.py": "6cc918984cd63feb1a9ec262c73ce2e9272cada95f1cf6130ea8d8ccb89f152b",
            "analysis/d1_pair/d1pair_common.py": "65a6e2da81e4dedf20c51423cbfe9c97e8b8f2272b24454a739a09e8552aeedc",
            "analysis/espley_xtb_repro/r6_nn.py": "9824c153e835f3c977f027bfac578f6253c30f113e938ce3b7b1c110dedf0846",
            "analysis/espley_xtb_repro/train_ml_single.py": "05c6b38cdf7803f4a6c6b2d643555db95b5e366ea02714049232add38a55da4c",
            "gnn3/gnn3/metrics.py": "86bb77c685255f1da07bc99dee33cfd79902c499f647ade31d041d04644a6d88",
            "gnn3/scripts/evaluate.py": "e8b089b81e9ec5cedc898a3769ef27e365da7fabc5f53ccafb3324c281f1adc1"}
FU2_ENV = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2")                   # gnn3/jobs/run.slurm ENV
NEED_GB = 20


def git(*a, raw=False):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=not raw)
    return r.returncode, (r.stdout if raw else r.stdout.strip())


def sh(cmd, timeout=600):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:                                                    # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def main(free_gb: float, method: str) -> int:
    res = {}
    # 1-1 origin/gnn3r2 at or after f6d9ee3, every GNN3R2 stage passed
    rc_o, o = git("rev-parse", "origin/gnn3r2")
    in_origin = git("merge-base", "--is-ancestor", BASE, "origin/gnn3r2")[0] == 0
    in_head = git("merge-base", "--is-ancestor", BASE, "HEAD")[0] == 0
    report = (C.GR2_RES / "RUN_REPORT_GNN3R2.md").read_text()
    rows = [l for l in report.split("## 2.")[0].splitlines() if l.startswith("| ") and not l.startswith("| 단계")]
    stages = len(rows) == 10 and all("| 통과" in l and "STOP" not in l for l in rows)
    res["1-1"] = dict(origin_gnn3r2=o, base=BASE, base_is_ancestor_of_origin=in_origin, base_is_ancestor_of_HEAD=in_head,
                      head=git("rev-parse", "HEAD")[1], branch=git("rev-parse", "--abbrev-ref", "HEAD")[1],
                      run_report_stage_rows=len(rows), run_report_all_pass=stages,
                      ok=rc_o == 0 and in_origin and in_head and stages)
    # 1-2 .. 1-4 GNN3R2 results, splits, CV predictions
    files = {}
    for name, want in RES_FILES.items():
        files[name] = (C.GR2_RES / name, want)
    for k, want in SPLIT_SHA.items():
        files[f"splits/T1__f{k}.json"] = (C.SPLIT_DIR / f"T1__f{k}.json", want)
    for k, want in CV_SHA.items():
        files[f"cv/gnn3_fold{k}.parquet"] = (C.GNN3_CV / f"gnn3_fold{k}.parquet", want)
    got = {n: dict(sha256=C.sha256(p) if p.is_file() else None, want=w) for n, (p, w) in files.items()}
    for key, names in (("1-2", list(RES_FILES)), ("1-3", [f"splits/T1__f{k}.json" for k in SPLIT_SHA]),
                       ("1-4", [f"cv/gnn3_fold{k}.parquet" for k in CV_SHA])):
        res[key] = dict(files={n: got[n] for n in names}, ok=all(got[n]["sha256"] == got[n]["want"] for n in names))
    # 1-5 X182, the value GNN3R2 trained on
    rec0 = C.read_json(C.GNN3_CV / "gnn3_fold0.json")
    x_rec = rec0["settings"]["inputs_sha256"]["x182"]
    s_x = C.sha256(C.X182)
    res["1-5"] = dict(path=str(C.X182), sha256=s_x, want=X182_SHA, gnn3r2_fold0_x182=x_rec,
                      ok=s_x == X182_SHA == x_rec)
    # 1-6 cr2 labels = the GNN3R2 B input record
    s_l = C.sha256(C.LABELS_CR2)
    b_rec = C.read_json(C.GR2_RES / "B_report.json")["inputs"].get(str(C.LABELS_CR2))
    res["1-6"] = dict(path=str(C.LABELS_CR2), sha256=s_l, want=CR2_SHA, gnn3r2_b_report=b_rec, ok=s_l == CR2_SHA == b_rec)
    # 1-7 the code to copy, at origin/gnn3r2 and in this worktree
    code = {}
    for path, want in CODE_SHA.items():
        rc, blob = git("show", f"origin/gnn3r2:{path}", raw=True)
        code[path] = dict(origin_sha256=hashlib.sha256(blob).hexdigest() if rc == 0 else None,
                          worktree_sha256=C.sha256(C.REPO / path), want=want)
    res["1-7"] = dict(code=code, ok=all(v["origin_sha256"] == v["worktree_sha256"] == v["want"] for v in code.values()))
    # 1-8 environments: env_espley (this process) and env_d1fu2 (gnn3.metrics import)
    from importlib.metadata import PackageNotFoundError, version

    def ver(k):
        try:
            return version(k)
        except PackageNotFoundError:
            return None
    vers = {k: ver(k) for k in ("scikit-learn", "tensorflow", "keras", "keras_tuner", "numpy", "pandas", "scipy",
                                "pyarrow", "fastparquet")}
    rc_f, out_f = sh(f"cd {C.GNN3_PACKAGE} && {FU2_ENV}/bin/python -c \"import sys, numpy, pandas, gnn3.metrics as M; "
                     "print('py', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, "
                     "'metrics', M.__file__)\"")
    res["1-8"] = dict(env_espley=dict(python=sys.executable, py=sys.version.split()[0], versions=vers),
                      env_d1fu2=dict(path=str(FU2_ENV), env_ok_marker=(FU2_ENV / ".d1fu2_env_ok").is_file(), rc=rc_f,
                                     out=out_f.splitlines()[-1:]),
                      ok=vers["scikit-learn"] == "1.3.1" and vers["tensorflow"] == "2.16.1" and vers["keras"] == "3.3.3"
                      and bool(vers["keras_tuner"]) and rc_f == 0 and (FU2_ENV / ".d1fu2_env_ok").is_file())
    # 2-3: the constants written again in eb_common.py = the D1PAIR / Espley values (import, read only)
    C.path_setup()
    sys.path.insert(0, str(C.D1PAIR_DIR))
    import d1pair_common as DP
    import r6_nn as NN
    import train_ml_single as T
    pairs = dict(SEED=(C.SEED, DP.SEED), CV_SEED=(C.CV_SEED, DP.CV_SEED), TUNE_SEED=(C.TUNE_SEED, DP.TUNE_SEED),
                 r6_TUNE_SEED=(C.TUNE_SEED, NN.TUNE_SEED), N_INNER=(C.N_INNER, DP.N_INNER),
                 SCALE_EPS=(C.SCALE_EPS, DP.SCALE_EPS), ANTISYM_TOL=(C.ANTISYM_TOL, DP.ANTISYM_TOL),
                 ANTISYM_N=(C.ANTISYM_N, DP.ANTISYM_N), GAMMAS=(C.GAMMAS, DP.GAMMAS), SVR_C=(C.SVR_C, DP.SVR_C),
                 SVR_EPS=(C.SVR_EPS, DP.SVR_EPS), NN_SIZE=(C.NN_SIZE, DP.NN_SIZE),
                 EB_RIDGE_ALPHA=(C.EB_RIDGE_ALPHA, DP.grid_values("Ridge")["alpha"]),
                 EB_KRR_ALPHA=(C.EB_KRR_ALPHA, DP.grid_values("KRR_rbf")["alpha"]),
                 D1_TCOL=(C.D1_TCOL, DP.D1_TCOL), NN_FINAL_SEEDS=(list(C.NN_FINAL_SEEDS), list(T.SEEDS)))
    grids_ok = {m: int(len(__import__("sklearn.model_selection", fromlist=["ParameterGrid"]).ParameterGrid(g)))
                for m, g in C.EA_GRIDS.items()}
    res["2-3"] = dict(constants={k: dict(eb_common=a, original=b, equal=a == b) for k, (a, b) in pairs.items()},
                      ea_grid_cells=grids_ok, want_cells=C.EA_GRID_CELLS,
                      ok=all(a == b for a, b in pairs.values()) and grids_ok == C.EA_GRID_CELLS)
    # 1-9 $EB absent or empty, free disk
    empty = not C.EB.exists() or (C.EB.is_dir() and not any(C.EB.iterdir()))
    res["1-9"] = dict(eb=str(C.EB), absent_or_empty=empty, free_gb=free_gb, method=method, need_gb=NEED_GB,
                      ok=empty and free_gb >= NEED_GB)
    bad = [k for k, v in res.items() if not v["ok"]]
    out = dict(spec="docs/specs/ESPB.md 1", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: v["ok"] for k, v in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(float(sys.argv[1]), sys.argv[2]))
