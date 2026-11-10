#!/usr/bin/env python3
"""CH3-MODEL 1: start conditions 1-1 .. 1-8 (docs/specs/CH3_MODEL.md 1) -> results/S0_start.json.
Stdlib only (reactot or any python 3.9+). The 1-8 free disk is measured by the job (jobs/s0.slurm: mmlsquota if the
node has it, else 300 GB quota - du of the user trees) and passed as arguments.

  python s0_start.py <free_gb> <method>     exit 0 all pass, 3 = STOP"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch3_common as C  # noqa: E402

NEED_OK = ("ts.xyz", "ts_result.json", "ref_dipole.xyz", "ref_dipolarophile.xyz")
IN_COMMIT = ("analysis/deploy", "analysis/d1_pair/fu2", "analysis/d1_ladder", "analysis/geo2/eval_geo2.py",
             "analysis/deploy/results/K_predictions.parquet")
ENV_D1 = C.D1_RUN / "nurion" / "env.sh"
VENV5 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python")        # analysis/d1_pair/jobs/env.sh PYENV=venv5
ENV_D1FU2 = C.D1PAIR / "fu2" / "env_d1fu2"
ENV_ESPLEY = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python")
NEED_GB = 30


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main(free_gb: float, method: str):
    res = {}
    # 1-1 the deploy commit and its content
    rc_t, kind = git("cat-file", "-t", C.DEPLOY_SHA)
    has = {p: git("cat-file", "-e", f"{C.DEPLOY_SHA}:{p}")[0] == 0 for p in IN_COMMIT}
    anc = git("merge-base", "--is-ancestor", C.DEPLOY_SHA, "HEAD")[0] == 0
    same = {p: git("diff", "--quiet", C.DEPLOY_SHA, "--", p)[0] == 0 for p in ("analysis/deploy", "analysis/d1_pair",
                                                                                 "analysis/d1_ladder", "analysis/geo2")}
    res["1-1"] = dict(commit=C.DEPLOY_SHA, type=kind, head=git("rev-parse", "HEAD")[1], deploy_is_ancestor=anc,
                      in_commit=has, worktree_equals_commit=same,
                      ok=rc_t == 0 and kind == "commit" and anc and all(has.values()) and all(same.values()))
    # 1-2 K_predictions sha256 = K_report predictions_sha256
    kr = C.read_json(C.DEPLOY_RES / "K_report.json")
    s_k = C.sha256(C.DEPLOY_RES / "K_predictions.parquet")
    res["1-2"] = dict(K_predictions_sha256=s_k, K_report_predictions_sha256=kr.get("predictions_sha256"),
                      ok=s_k == kr.get("predictions_sha256"))
    # 1-3 G2 work folders (= DEPLOY 0-2)
    with open(C.REPO / "analysis" / "d1_g2" / "results" / "G_status.csv", newline="") as fh:
        st = list(csv.DictReader(fh))
    jobs = [r["job_id"] for r in st]
    miss_dir = [j for j in jobs if not (C.G2 / "jobs" / j).is_dir()]
    ok_rows = [r for r in st if r["status"] == "ok"]
    miss_files = {}
    for r in ok_rows:
        lack = [f for f in NEED_OK if not (C.G2 / "jobs" / r["job_id"] / f).is_file()]
        if lack:
            miss_files[r["job_id"]] = lack
    res["1-3"] = dict(n_targets=len(jobs), n_jobs_missing=len(miss_dir), missing_jobs=miss_dir[:20], n_ok=len(ok_rows),
                      n_ok_missing_files=len(miss_files), examples=dict(list(miss_files.items())[:10]),
                      ok=len(jobs) == 9266 and not miss_dir and len(ok_rows) == 8616 and not miss_files)
    # 1-4 g2_final (per ok reaction, as DEPLOY 0-3), X182_d1_g2, fu2/tokens
    fin_missing = [r["rxn_id"] for r in ok_rows if not (C.G2 / "g2_final" / r["rxn_id"]).is_dir()]
    x182, tok = (C.G2 / "X182_d1_g2.parquet").is_file(), (C.G2 / "fu2" / "tokens").is_dir()
    res["1-4"] = dict(g2_final=(C.G2 / "g2_final").is_dir(), g2_final_missing_for_ok=len(fin_missing),
                      missing_examples=fin_missing[:10], x182=x182, tokens=tok,
                      ok=(C.G2 / "g2_final").is_dir() and not fin_missing and x182 and tok)
    # 1-5 labels sha256 = deploy K_report inputs_sha256["labels"]
    s_lab = C.sha256(C.LABELS) if C.LABELS.is_file() else None
    want = (kr.get("inputs_sha256") or {}).get("labels")
    res["1-5"] = dict(path=str(C.LABELS), F_pairs_cr2_sha256=s_lab, K_report_key="inputs_sha256.labels",
                      K_report_labels_sha256=want, ok=s_lab is not None and s_lab == want)
    # 1-6 FU2 T1 splits
    rec = C.read_json(C.REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json")["units"]
    sp = {}
    for k in range(5):
        f = C.D1PAIR / "fu2" / "splits" / f"T1__f{k}.json"
        sp[f.name] = dict(now=C.sha256(f) if f.is_file() else None, record=rec.get(f.name, {}).get("sha256"))
    res["1-6"] = dict(splits=sp, ok=all(v["now"] and v["now"] == v["record"] for v in sp.values()))
    # 1-7 environments
    env = dict(d1_run_env_sh=ENV_D1.is_file(), venv5_python=VENV5.exists(),
               g_slurm=(C.REPO / "analysis" / "d1_pair" / "jobs" / "g.slurm").is_file(),
               env_d1fu2_ok=(ENV_D1FU2 / ".d1fu2_env_ok").is_file(), env_d1fu2_python=(ENV_D1FU2 / "bin" / "python").exists(),
               env_espley_python=ENV_ESPLEY.exists())
    res["1-7"] = dict(paths=dict(d1_run_env_sh=str(ENV_D1), venv5=str(VENV5), env_d1fu2=str(ENV_D1FU2),
                                 env_espley=str(ENV_ESPLEY)), present=env, ok=all(env.values()))
    # 1-8 manifest sha256 = g2_common.MANIFEST_SHA, free disk
    man = C.D1_RUN / "d1_manifest.csv"
    man_sha = C.sha256(man) if man.is_file() else None
    want_m = None
    for ln in (C.REPO / "analysis" / "d1_g2" / "g2_common.py").read_text().splitlines():
        if ln.startswith("MANIFEST_SHA"):
            want_m = ln.split("=", 1)[1].split("#")[0].strip().strip('"')
    res["1-8"] = dict(manifest=str(man), manifest_sha256=man_sha, g2_common_MANIFEST_SHA=want_m, free_gb=free_gb,
                      free_gb_method=method, need_gb=NEED_GB,
                      ok=man_sha is not None and man_sha == want_m and free_gb >= NEED_GB)
    bad = [k for k, v in res.items() if not v["ok"]]
    out = dict(spec="docs/specs/CH3_MODEL.md 1", written=C.now(), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: v["ok"] for k, v in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(float(sys.argv[1]), sys.argv[2]))
