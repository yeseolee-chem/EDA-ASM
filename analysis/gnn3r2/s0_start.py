#!/usr/bin/env python3
"""GNN3R2 1: start conditions 1-1 .. 1-5 (docs/specs/GNN3R2.md 1) -> results/S0_start.json.

    python analysis/gnn3r2/s0_start.py <free_gb> <method>      reactot python; job: analysis/gnn3r2/jobs/s0_gr.slurm

1-3 = the GNN3R start conditions 1-2 .. 1-10 (docs/specs/GNN3R.md 1), run by gnn3r_checks (the GNN3R s0_start.py code,
copied; its 1-1 branch base and 1-11 empty-scratch checks are replaced by GNN3R2 1-1 and 1-5).
The job fetches origin/gnn3r and measures the free disk (compute nodes cannot always run mmlsquota) and passes it.
Exit 0 all pass, 3 = STOP.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

G2R_BASE = "df3deb30"
NEED_DIRS = ("analysis/g2r", "analysis/d1_pair", "analysis/d1_ladder", "gnn3")
G2R_FILES = {"X182_lf.parquet": C.X182, "labels_lf.json": C.LABELS_LF, "input_meta_lf.csv": C.INPUT_META_LF,
             "T_manifest.csv": C.TOKEN_REPORTS / "T_manifest.csv"}
G2R_PREFIX = {"X182_lf.parquet": "66c40316", "labels_lf.json": "3f2bf2cc", "input_meta_lf.csv": "a3968398",
              "T_manifest.csv": "f424d12d"}
USABLE_SHA = "8a4b4f478fea9d9e58979ae4253d3d61c9ccfbae7f975accf16c7d3cfbd96db1"
TARGETS_SHA = "8e89a3920704519559794b4dba03bca2a9455345861a704d4e14483253b2b7a5"
CR2_SHA = "7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755"
F_PAIRS_CR2_PREFIX = "f2fe5cab"
N_ROWATTR_ALL, N_ROWATTR_KEPT = 9496, 9035
D1PAIR_FILES = {C.PAIRS_D1: ("227063285d65a05195a0d662df040618ac52bd175718837423ded2343c83ecce", 7253),
                C.ANCHOR_D1: ("2121c684bd11a77d4b59128432d3e6e22d3dab23081bd25bd9864e5172a9bb6c", 1159),
                C.P4_STEREO: ("5fb1155b4cc35ad39b4721b7bc8e1900cc5d7c61c9fd7f6b7089e5633035dac5", None),
                C.P4_DROP: ("91a50b8fabf2b94a1c4d499bff50edc323fee1e0af2fd4450fd22b78ba394ee6", 29),
                C.S_FOLDS: ("944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7", None),
                C.SIGMA: ("2ba63224928301657f09707cb3be1b0e05a6c69ef1e733b490b9a4ac6261d06f", None)}
PAIR_CODE = {"d1_pairs.py": "424c2bc24b4ed5111c35a00688a88d43f47dffb2e61e50c5326256f31be603e1",
             "d1_anchor_pairs.py": "f979e1e336df850a0b55d6ed762718b035bf2b151468263919b53a5aa40bac18",
             "build_rowattr.py": "81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4",
             "d1_phase_p.py": "630e0d9bb7f57ea31b542f82e2533965f25aff931ef4d528b5c1f3636e92cefd"}
REF_SHA = {C.REF_REACTIONS: ("95e69518023a59420f74d59be7687585af83f7eb86030c6f28c4dca1b7ea901b", 8604),
           C.REF_PAIRS: ("77b78095cf8fec29d5f47dcd081419442d4e582b85f66ff797a8ea0c0fb16e0f", 8355)}
ESPLEY_PY = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python")     # d1_pair/jobs/env.sh PYENV=espley
FU2_ENV = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2")                   # gnn3/jobs/run.slurm ENV
NEED_GB = 20
# ---------------------------------------------------------------- GNN3R2 1-1, 1-2, 1-4
GNN3R_BASE = "4c8709c7"
GNN3R_RES = C.REPO / "analysis" / "gnn3r" / "results"
GNN3R_REPORTS = ("S0_start.json", "A_report.json", "B_report.json", "S_splits_r.json", "D_check.json",
                 "cv/K_check.json", "eval/E_check.json", "final/W_check.json")
GNN3R_FILES = {GNN3R_RES / "F_pairs_r.csv": "5597ad11558efb1bc3fc6322f99b5920026166e161a1df383662c6905b95a467",
               GNN3R_RES / "B_row_status.csv": "9207e71631e5e8883c9b6403c09bafa219601143cbc44a6310455b8cad514efe",
               C.GR_GNN3R / "A" / "rowattr_r.pkl": "f0e7fe41001a933d20af756be5db724eb6db33ef40b36109c88ed2657b7e81e3"}
P4_CACHE_GNN3R = C.GR_GNN3R / "B" / "p4_cache.pkl"
N_P4_CACHE_MIN = 8605
RDKIT_VERSION = "2026.03.2"


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def sh(cmd, timeout=600):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:                                                    # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def n_csv_rows(path) -> int:
    with open(path, newline="") as fh:
        return sum(1 for _ in fh) - 1


def gnn3r_checks() -> dict:
    """GNN3R start conditions 1-2 .. 1-10 (GNN3R s0_start.py code; its 1-1 and 1-11 are not part of GNN3R2 1-3)."""
    import pandas as pd
    C.import_d1pair()
    import d1pair_common as DP                                  # read only: D1_TCOL, D1_LABELS(_SHA)
    res = {}
    # 1-2 g2r inputs = g2r.json, token reports, token files
    man = C.read_json(C.G2R_MANIFEST)["header"]["files"]
    got = {k: C.sha256(p) if p.is_file() else None for k, p in G2R_FILES.items()}
    eq = {k: got[k] is not None and got[k] == man[k]["sha256"] and got[k].startswith(G2R_PREFIX[k]) for k in got}
    tr = C.read_json(C.TOKEN_REPORTS / "T_report.json")
    tm = pd.read_csv(C.TOKEN_REPORTS / "T_manifest.csv")
    t_ok = dict(manifest_sha256=tr.get("manifest_sha256") == got["T_manifest.csv"],
                q_report_sha256=tr.get("q_report_sha256") == C.sha256(C.TOKEN_REPORTS / "Q_report.json"))
    npz = sorted(p.name for p in C.TOKENS.glob("*.npz"))
    want_npz = sorted(f"{int(r)}.npz" for r in tm.rxn_id)
    res["1-2"] = dict(sha256=got, g2r_json={k: man[k]["sha256"] for k in got}, equal=eq, t_report=t_ok,
                      n_manifest_rows=int(len(tm)), n_npz=len(npz), npz_equal_manifest=npz == want_npz,
                      ok=all(eq.values()) and all(t_ok.values()) and len(tm) == C.N_USABLE and len(npz) == C.N_USABLE
                      and npz == want_npz)
    # 1-3 usable list and target list
    U = pd.read_csv(C.USABLE)
    X = pd.read_csv(C.TARGETS)
    s_u, s_x = C.sha256(C.USABLE), C.sha256(C.TARGETS)
    n_us = int(U.usable.astype(bool).sum())
    res["1-3"] = dict(usable_sha256=s_u, n_usable_rows=int(len(U)), n_usable=n_us, targets_sha256=s_x,
                      n_targets=int(len(X)), same_job_ids=set(U.job_id) == set(X.job_id),
                      ok=s_u == USABLE_SHA and len(U) == C.N_TARGETS and n_us == C.N_USABLE and s_x == TARGETS_SHA
                      and len(X) == C.N_TARGETS and set(U.job_id) == set(X.job_id))
    # 1-4 cr2 labels, V_report, F_pairs_cr2
    s_cr2 = C.sha256(C.LABELS_CR2)
    cr2 = {d["job_id"]: d for d in json.loads(C.LABELS_CR2.read_text())}
    bad_lab = [j for j in X.job_id if j not in cr2 or not all(
        isinstance(cr2[j].get(c), (int, float)) and math.isfinite(cr2[j][c]) for c in DP.D1_TCOL.values())]
    v = C.read_json(C.V_REPORT)
    s_fp = C.sha256(C.F_PAIRS_CR2)
    rec_fp = (v.get("outputs") or {}).get(str(C.F_PAIRS_CR2))
    res["1-4"] = dict(cr2_sha256=s_cr2, n_cr2=len(cr2), n_targets_bad_labels=len(bad_lab), bad=bad_lab[:20],
                      label_columns=list(DP.D1_TCOL.values()), v_report_all_pass=v.get("all_pass"),
                      f_pairs_cr2_sha256=s_fp, v_report_outputs_sha256=rec_fp,
                      ok=s_cr2 == CR2_SHA and not bad_lab and v.get("all_pass") is True and rec_fp == s_fp
                      and s_fp.startswith(F_PAIRS_CR2_PREFIX))
    # 1-5 row attribute tables
    ra, rk = pd.read_pickle(C.ROWATTR_ALL), pd.read_pickle(C.ROWATTR_KEPT)
    miss = sorted(set(X.job_id) - set(ra.job_id))
    res["1-5"] = dict(n_all=int(len(ra)), n_kept=int(len(rk)), all_job_id_unique=bool(ra.job_id.is_unique),
                      n_targets_missing=len(miss), missing=miss[:20],
                      ok=len(ra) == N_ROWATTR_ALL and len(rk) == N_ROWATTR_KEPT and not miss and ra.job_id.is_unique)
    # 1-6 D1PAIR files
    files = {}
    for p, (want, n) in D1PAIR_FILES.items():
        s = C.sha256(p) if p.is_file() else None
        rows = n_csv_rows(p) if p.is_file() else None
        files[str(p)] = dict(sha256=s, want=want, rows=rows, want_rows=n, ok=s == want and (n is None or rows == n))
    res["1-6"] = dict(files=files, ok=all(f["ok"] for f in files.values()))
    # 1-7 pair code
    code = {k: dict(sha256=C.sha256(C.D1PAIR_CODE / k), want=w) for k, w in PAIR_CODE.items()}
    res["1-7"] = dict(code=code, ok=all(c["sha256"] == c["want"] for c in code.values()))
    # 1-8 labels_d1.json and the DFT job folders of the usable reactions (P-4 reads them)
    s_d1 = C.sha256(DP.D1_LABELS)
    usable_jobs = sorted(U.loc[U.usable.astype(bool), "job_id"])
    no_job = [j for j in usable_jobs if not ((C.D1_JOBS / j / "ts.xyz").is_file()
                                             and (C.D1_JOBS / j / "sp" / "eda.inp").is_file())]
    res["1-8"] = dict(labels=str(DP.D1_LABELS), sha256=s_d1, D1_LABELS_SHA=DP.D1_LABELS_SHA, d1_jobs=str(DP.D1_JOBS),
                      n_usable=len(usable_jobs), n_job_folders_missing=len(no_job), missing=no_job[:20],
                      ok=s_d1 == DP.D1_LABELS_SHA and DP.D1_JOBS == C.D1_JOBS and not no_job)
    # 1-9 reference lists
    refs = {}
    for p, (want, n) in REF_SHA.items():
        s = C.sha256(p) if p.is_file() else None
        refs[str(p)] = dict(sha256=s, want=want, rows=n_csv_rows(p) if p.is_file() else None, want_rows=n,
                            ok=s == want and n_csv_rows(p) == n)
    res["1-9"] = dict(files=refs, ok=all(f["ok"] for f in refs.values()))
    # 1-10 environments and the RTX 3090 partition
    import rdkit
    import sklearn  # noqa: F401  (reactot has it; the version that counts is env_espley's)
    rc_e, out_e = sh(f"{ESPLEY_PY} -c \"import sys, sklearn; print('py', sys.version.split()[0], 'sklearn', "
                     "sklearn.__version__)\"")
    rc_f, out_f = sh(f"cd {C.REPO / 'gnn3'} && {FU2_ENV}/bin/python -c \"import sys, torch, pandas, pyarrow, gnn3; "
                     "print('py', sys.version.split()[0], 'torch', torch.__version__, 'pandas', pandas.__version__)\"")
    rc_s, out_s = sh("sinfo -h -p gpu1 -o '%N %G %t'")
    jobs = {p: (C.REPO / p).is_file() for p in ("analysis/d1_pair/jobs/py.slurm", "analysis/d1_pair/jobs/env.sh",
                                                "gnn3/jobs/run.slurm")}
    env = dict(reactot=dict(python=sys.executable, py=sys.version.split()[0], rdkit=rdkit.__version__,
                            pandas=pd.__version__),
               espley=dict(python=str(ESPLEY_PY), rc=rc_e, out=out_e.splitlines()[-1:], ok=rc_e == 0
                           and "sklearn 1.3.1" in out_e),
               env_d1fu2=dict(path=str(FU2_ENV), env_ok_marker=(FU2_ENV / ".d1fu2_env_ok").is_file(), rc=rc_f,
                              out=out_f.splitlines()[-1:], ok=(FU2_ENV / ".d1fu2_env_ok").is_file() and rc_f == 0),
               gpu1=dict(rc=rc_s, sinfo=out_s.splitlines(), ok=rc_s == 0 and "rtx3090" in out_s),
               job_scripts=jobs)
    res["1-10"] = dict(env, ok=env["espley"]["ok"] and env["env_d1fu2"]["ok"] and env["gpu1"]["ok"]
                       and all(jobs.values()))
    return res


def main(free_gb: float, method: str) -> int:
    import pickle
    import rdkit
    res = {}
    # 1-1 origin/gnn3r at or after 4c8709c, every GNN3R stage passed (report table + stage reports)
    rc_o, o = git("rev-parse", "origin/gnn3r")
    in_origin = git("merge-base", "--is-ancestor", GNN3R_BASE, "origin/gnn3r")[0] == 0
    in_head = git("merge-base", "--is-ancestor", GNN3R_BASE, "HEAD")[0] == 0
    report = (GNN3R_RES / "RUN_REPORT_GNN3R.md").read_text()
    stage_rows = [l for l in report.split("## 2.")[0].splitlines() if l.startswith("| ") and not l.startswith("| 단계")]
    stages_pass = len(stage_rows) == 8 and all("| 통과 |" in l for l in stage_rows)
    reports = {n: bool(C.read_json(GNN3R_RES / n).get("all_pass")) if (GNN3R_RES / n).is_file() else None
               for n in GNN3R_REPORTS}
    res["1-1"] = dict(origin_gnn3r=o, base=GNN3R_BASE, base_is_ancestor_of_origin=in_origin,
                      base_is_ancestor_of_HEAD=in_head, head=git("rev-parse", "HEAD")[1],
                      branch=git("rev-parse", "--abbrev-ref", "HEAD")[1], run_report_stage_rows=len(stage_rows),
                      run_report_all_pass=stages_pass, stage_reports_all_pass=reports,
                      ok=rc_o == 0 and in_origin and in_head and stages_pass and all(v is True for v in reports.values()))
    # 1-2 GNN3R files and the P-4 cache
    files = {str(p): dict(sha256=C.sha256(p) if p.is_file() else None, want=w) for p, w in GNN3R_FILES.items()}
    n_cache = len(pickle.loads(P4_CACHE_GNN3R.read_bytes())) if P4_CACHE_GNN3R.is_file() else 0
    res["1-2"] = dict(files=files, p4_cache=str(P4_CACHE_GNN3R), p4_cache_reactions=n_cache, need=N_P4_CACHE_MIN,
                      ok=all(f["sha256"] == f["want"] for f in files.values()) and n_cache >= N_P4_CACHE_MIN)
    # 1-3 GNN3R start conditions 1-2 .. 1-10 still hold
    sub = gnn3r_checks()
    res["1-3"] = dict(gnn3r_conditions=sub, failed=[k for k, v in sub.items() if not v["ok"]],
                      ok=all(v["ok"] for v in sub.values()))
    # 1-4 RDKit version of PYENV=reactot (= GNN3R A_report.json)
    a_rep = C.read_json(GNN3R_RES / "A_report.json")["versions"]["rdkit"]
    res["1-4"] = dict(rdkit=rdkit.__version__, want=RDKIT_VERSION, gnn3r_a_report=a_rep,
                      ok=rdkit.__version__ == RDKIT_VERSION == a_rep)
    # 1-5 $GR2 absent or empty, free disk
    gr_empty = not C.GR.exists() or (C.GR.is_dir() and not any(C.GR.iterdir()))
    res["1-5"] = dict(gr2=str(C.GR), gr2_absent_or_empty=gr_empty, free_gb=free_gb, method=method, need_gb=NEED_GB,
                      ok=gr_empty and free_gb >= NEED_GB)
    bad = [k for k, v in res.items() if not v["ok"]]
    out = dict(spec="docs/specs/GNN3R2.md 1", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, sub_1_3_failed=res["1-3"]["failed"],
                          **{k: v["ok"] for k, v in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(float(sys.argv[1]), sys.argv[2]))
