#!/usr/bin/env python3
"""D1-G2R 1: start conditions 1-1 .. 1-8 (docs/specs/D1G2R.md 1) -> results/S0_start.json.

    python analysis/g2r/s0_start.py <free_gb> <method>       env_d1fu2 (pandas + pyarrow); job: jobs/s0_g2r.slurm

The job measures the free disk (compute nodes cannot always run mmlsquota) and passes it. Exit 0 all pass, 3 = STOP.

1-3 marker rule (recorded decision): $G2/jobs/<job_id>/ holds only the stage markers .done_<stage> / .fail_<stage>
(text = reason); '.done' and '.fail_<stage>_<reason>' exist only in $G2/g2_final/<rxn_id>/ and G_status.csv `marker`.
The stage state is derived with the finalize rule (finalize_lf.py:151-159 = g2_autode.finalize l.266-274, marker_name
l.116-117): first stage in (ts, stereo, ref) with a .fail_<stage> file -> '.fail_' + sub('[^A-Za-z0-9_]+', '_',
f'{stage}_' + text.split(':')[0]); no .done_<stage> -> '.fail_<stage>_not_run'; all three .done_* -> 'stages_done'.
A G_status row whose marker is a stage failure (^.fail_(ts|stereo|ref)_) must derive to that marker; every other row
('.done', '.fail_hess_*', '.fail_freq_*': failures of the g2_final step after the stages) must derive to 'stages_done'.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

GNN3_CLEAN_SHA = "dd36bc0a"
NEED_DIRS = ("d1_g2", "d1_pair", "d1_ladder", "deploy", "ch3", "espley_xtb_repro")
N_TARGETS = 9266
G_STATUS_SHA = "7685cd0612e6842407183c5e2dfbb93421cd26579f06cfced9be561c45c889ec"   # d1_g2 G_report status_csv_sha256
STAGES = ("ts", "stereo", "ref")
STAGE_FAIL = re.compile(r"^\.fail_(ts|stereo|ref)_")
D1_ENV_SH = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh")          # cpu_d1.slurm:19
VENV5 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python")          # d1_pair/jobs/env.sh PYENV=venv5
XTB_G = Path("/home1/yeseo1ee/xtb-dist/bin/xtb")                             # d1_pair/jobs/g_env.sh:19
NEED_GB = 30
CH3 = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3")


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def sh(cmd, timeout=600):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:                                                    # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def derive(jd: Path) -> str:
    for st in STAGES:
        f = jd / f".fail_{st}"
        if f.is_file():
            text = f.read_text()
            return ".fail_" + re.sub(r"[^A-Za-z0-9_]+", "_", f"{st}_" + (text.split(":")[0] if text else "fail"))
        if not (jd / f".done_{st}").is_file():
            return f".fail_{st}_not_run"
    return "stages_done"


def tree_size(root: Path) -> dict:
    n, b = 0, 0
    for folder, _dirs, names in os.walk(root):
        for name in names:
            p = Path(folder) / name
            n += 1
            b += 0 if p.is_symlink() else p.stat().st_size
    return dict(path=str(root), n_files=n, bytes=b, gb=round(b / 1e9, 3))


def main(free_gb: float, method: str) -> int:
    import pandas as pd
    res = {}
    # 1-1 branch base and analysis folders
    rc_o, o = git("rev-parse", "origin/gnn3-clean")
    base_in_origin = git("merge-base", "--is-ancestor", GNN3_CLEAN_SHA, "origin/gnn3-clean")[0] == 0
    base_in_head = git("merge-base", "--is-ancestor", GNN3_CLEAN_SHA, "HEAD")[0] == 0
    dirs = {d: (C.REPO / "analysis" / d).is_dir() for d in NEED_DIRS}
    res["1-1"] = dict(origin_gnn3_clean=o, base=GNN3_CLEAN_SHA, base_is_ancestor_of_origin=base_in_origin,
                      base_is_ancestor_of_HEAD=base_in_head, head=git("rev-parse", "HEAD")[1], dirs=dirs,
                      ok=rc_o == 0 and base_in_origin and base_in_head and all(dirs.values()))
    # 1-2 labels_d1.json and the D1 manifest
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_g2"))
    import g2_common as G                                       # read only: MANIFEST(_SHA), DP.D1_LABELS(_SHA)
    s_lab, s_man = C.sha256(G.DP.D1_LABELS), C.sha256(G.MANIFEST)
    res["1-2"] = dict(labels=str(G.DP.D1_LABELS), labels_sha256=s_lab, D1_LABELS_SHA=G.DP.D1_LABELS_SHA,
                      manifest=str(G.MANIFEST), manifest_sha256=s_man, MANIFEST_SHA=G.MANIFEST_SHA,
                      ok=s_lab == G.DP.D1_LABELS_SHA and s_man == G.MANIFEST_SHA)
    # 1-3 $G2/jobs folders and their stage state vs G_status.csv marker
    s_st = C.sha256(C.G_STATUS)
    with open(C.G_STATUS, newline="") as fh:
        st = list(csv.DictReader(fh))
    miss, diff, kinds, exceptions = [], [], {}, []
    for r in st:
        jd = C.G2_OLD_JOBS / r["job_id"]
        if not jd.is_dir():
            miss.append(r["job_id"])
            continue
        d, m = derive(jd), r["marker"]
        want = m if STAGE_FAIL.match(m) else "stages_done"
        kinds[f"{m} -> {d}"] = kinds.get(f"{m} -> {d}", 0) + 1
        if d != want:
            diff.append(dict(job_id=r["job_id"], g_status_marker=m, derived=d))
        elif m != ".done" and not STAGE_FAIL.match(m):
            exceptions.append(dict(job_id=r["job_id"], rxn_id=r["rxn_id"], g_status_marker=m, derived=d,
                                   g2_final=sorted(p.name for p in (C.G2_OLD_FINAL / r["rxn_id"]).glob(".*"))))
    res["1-3"] = dict(g_status=str(C.G_STATUS), g_status_sha256=s_st, want_g_status_sha256=G_STATUS_SHA,
                      n_rows=len(st), n_distinct_job_ids=len({r["job_id"] for r in st}), n_jobs_missing=len(miss),
                      missing=miss[:20], n_marker_differs=len(diff), differs=diff[:20], marker_to_derived=kinds,
                      post_stage_failures=exceptions,
                      rule="stage-failure rows: derived == marker; other rows: all three .done_<stage> (module doc)",
                      ok=s_st == G_STATUS_SHA and len(st) == N_TARGETS and len({r["job_id"] for r in st}) == N_TARGETS
                      and not miss and not diff)
    # 1-4 $CH3/lf files and their recorded sha256
    lf = C.CH3_LF_OLD
    fr = C.read_json(C.LF_REPORTS / "F_report.json")
    tr = C.read_json(C.LF_REPORTS / "fu2_lf" / "T_report.json")
    tm = C.LF_REPORTS / "fu2_lf" / "T_manifest.csv"
    have = {p: (lf / p).exists() for p in ("g2lf_final", "labels_lf.json", "input_meta_lf.csv", "X182_lf.parquet",
                                           "fu2_lf/tokens")}
    sh_ = dict(labels_lf=C.sha256(lf / "labels_lf.json") if have["labels_lf.json"] else None,
               input_meta_lf=C.sha256(lf / "input_meta_lf.csv") if have["input_meta_lf.csv"] else None,
               X182_lf=C.sha256(lf / "X182_lf.parquet") if have["X182_lf.parquet"] else None,
               T_manifest=C.sha256(tm))
    T = pd.read_csv(tm)
    tdir = lf / "fu2_lf" / "tokens"
    bad_tok = [int(r) for r, s in zip(T.rxn_id, T.sha256)
               if not (tdir / f"{int(r)}.npz").is_file() or C.sha256(tdir / f"{int(r)}.npz") != s]
    eq = dict(labels=sh_["labels_lf"] == fr.get("labels_sha256"), meta=sh_["input_meta_lf"] == fr.get("meta_sha256"),
              x182=sh_["X182_lf"] == tr.get("x182_sha256"), t_manifest=sh_["T_manifest"] == tr.get("manifest_sha256"))
    res["1-4"] = dict(root=str(lf), present=have, sha256=sh_,
                      recorded=dict(F_report_labels_sha256=fr.get("labels_sha256"),
                                    F_report_meta_sha256=fr.get("meta_sha256"),
                                    T_report_x182_sha256=tr.get("x182_sha256"),
                                    T_report_manifest_sha256=tr.get("manifest_sha256")),
                      equal=eq, t_manifest=str(tm), n_tokens_in_manifest=int(len(T)), n_token_files_bad=len(bad_tok),
                      bad_token_examples=bad_tok[:10],
                      ok=all(have.values()) and all(eq.values()) and not bad_tok)
    # 1-5 cr2 labels
    n_cr2 = int(len(pd.read_parquet(C.CR2_LABELS)))
    res["1-5"] = dict(path=str(C.CR2_LABELS), sha256=C.sha256(C.CR2_LABELS), n_rows=n_cr2, ok=n_cr2 == N_TARGETS)
    # 1-6 environments
    cfg = {}
    for ln in (C.REPO / "analysis" / "d1_g2" / "config_g2.yaml").read_text().splitlines():
        k, _, v = ln.partition(":")
        if k in ("orca_bin", "xtb_bin", "venv"):
            cfg[k] = v.split("#")[0].strip()
    rc_d1, out_d1 = sh(f"source {D1_ENV_SH} >/dev/null 2>&1; echo PY=$D1_PY; echo XTB=${{XTB_BIN:-}}; \"$D1_PY\" -c "
                       "\"import sys, rdkit, autode, yaml, pandas; print('py', sys.version.split()[0], 'rdkit', "
                       "rdkit.__version__, 'autode', autode.__version__)\"")
    rc_v5, out_v5 = sh(f"{VENV5} -c \"import sys, importlib.metadata as md; print('py', sys.version.split()[0], "
                       "'tblite', md.version('tblite'))\"")
    xtbs = {}
    for name, path in (("config_g2_xtb_bin", cfg.get("xtb_bin", "")), ("g_env_XTB_BIN", str(XTB_G))):
        rc_x, out_x = sh(f"{path} --version")
        xtbs[name] = dict(path=path, version_lines=[l for l in out_x.splitlines() if "version" in l.lower()][:2],
                          ok=any("6.7.1" in l for l in out_x.splitlines() if "version" in l.lower()))
    orca = cfg.get("orca_bin", "")
    rc_o6, out_o6 = sh(f"{orca} --version", timeout=120)
    orca_ok = Path(orca).is_file() and os.access(orca, os.X_OK) and ("6.1.1" in out_o6 or "6_1_1" in orca)
    jobs_ = {p: (C.REPO / p).is_file() for p in ("analysis/deploy/jobs/cpu_d1.slurm", "analysis/d1_pair/jobs/g.slurm",
                                                 "analysis/d1_pair/jobs/env.sh", "analysis/d1_pair/jobs/g_env.sh")}
    d1_ok = rc_d1 == 0 and "rdkit" in out_d1 and "autode 1.4.5" in out_d1
    v5_ok = rc_v5 == 0 and "tblite 0.7.0" in out_v5
    res["1-6"] = dict(d1_run_env=dict(env_sh=str(D1_ENV_SH), rc=rc_d1, out=out_d1.splitlines()[-4:], ok=d1_ok),
                      venv5=dict(python=str(VENV5), rc=rc_v5, out=out_v5.splitlines()[-2:], ok=v5_ok), xtb=xtbs,
                      orca=dict(path=orca, rc=rc_o6, out=out_o6.splitlines()[:6], ok=orca_ok,
                                note="version from `orca --version` output or the install path name"),
                      job_scripts=jobs_,
                      ok=d1_ok and v5_ok and all(x["ok"] for x in xtbs.values()) and orca_ok and all(jobs_.values()))
    # 1-7 disk, sizes of the two trees that phase D deletes and of the copy
    g2_size, lf_size = tree_size(C.G2_OLD), tree_size(C.CH3_LF_OLD)
    jobs_size = tree_size(C.G2_OLD_JOBS)
    res["1-7"] = dict(free_gb=free_gb, method=method, need_gb=NEED_GB, G2=g2_size, CH3_lf=lf_size,
                      G2_jobs_copy=jobs_size, free_gb_after_copy=round(free_gb - jobs_size["bytes"] / 1e9, 1),
                      ok=free_gb >= NEED_GB)
    # 1-8 paths
    rp = {str(p): os.path.realpath(p) for p in (C.G2R, C.G2_OLD, C.CH3_LF_OLD, CH3)}
    g2r_r = Path(rp[str(C.G2R)])
    under = [str(p) for p in (C.G2_OLD, CH3) if g2r_r == Path(rp[str(p)]) or Path(rp[str(p)]) in g2r_r.parents]
    links = {str(p): os.path.islink(p) for p in (C.G2_OLD, C.CH3_LF_OLD)}
    res["1-8"] = dict(realpath=rp, g2r_under=under, is_symlink=links,
                      ok=not under and not any(links.values()) and rp[str(C.G2_OLD)] == str(C.G2_OLD)
                      and rp[str(C.CH3_LF_OLD)] == str(C.CH3_LF_OLD))
    bad = [k for k, v in res.items() if not v["ok"]]
    out = dict(spec="docs/specs/D1G2R.md 1", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: v["ok"] for k, v in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(float(sys.argv[1]), sys.argv[2]))
