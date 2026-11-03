#!/usr/bin/env python3
"""DEPLOY-MODEL 1: start conditions 0-1 .. 0-7 (0-8 disk: compute nodes cannot run mmlsquota; the free GB read on the
login node is passed as an argument). Stdlib only (D1 run env or any python).
  python check_start.py <free_gb>      -> results/S0_start.json; exit 0 all pass, 3 = STOP"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dep_common as C  # noqa: E402

NEED_OK = ("ts.xyz", "ts_result.json", "ref_dipole.xyz", "ref_dipolarophile.xyz")
D1PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True).stdout.strip()


def main(free_gb: float):
    res, bad = {}, []
    # 0-1
    head = git("rev-parse", "HEAD")
    anc = subprocess.run(["git", "-C", str(C.REPO), "merge-base", "--is-ancestor", "0314ad9", "HEAD"]).returncode == 0
    dirs = {d: (C.REPO / "analysis" / d).is_dir() for d in ("d1_g2", "d1_pair", "d1_ladder", "espley_xtb_repro",
                                                              "geo2/deploy")}
    res["0-1"] = dict(head=head, geo2b_0314ad9_ancestor=anc, dirs=dirs, ok=anc and all(dirs.values()))
    # 0-2
    with open(C.REPO / "analysis" / "d1_g2" / "results" / "G_status.csv", newline="") as fh:
        st = list(csv.DictReader(fh))
    jobs = [r["job_id"] for r in st]
    miss_dir = [j for j in jobs if not (C.G2_JOBS / j).is_dir()]
    ok_rows = [r for r in st if r["status"] == "ok"]
    miss_files = {}
    for r in ok_rows:
        lack = [f for f in NEED_OK if not (C.G2_JOBS / r["job_id"] / f).is_file()]
        if lack:
            miss_files[r["job_id"]] = lack
    res["0-2"] = dict(n_targets=len(jobs), n_jobs_missing=len(miss_dir), missing_jobs=miss_dir[:20], n_ok=len(ok_rows),
                      n_ok_missing_files=len(miss_files), examples=dict(list(miss_files.items())[:10]),
                      ok=len(jobs) == 9266 and not miss_dir and len(ok_rows) == 8616 and not miss_files)
    # 0-3
    fin_missing = [r["rxn_id"] for r in ok_rows if not (C.G2_FINAL / r["rxn_id"]).is_dir()]
    res["0-3"] = dict(g2_final_missing_for_ok=len(fin_missing), x182=(C.G2 / "X182_d1_g2.parquet").is_file(),
                      tokens=(C.G2 / "fu2" / "tokens").is_dir(),
                      ok=not fin_missing and (C.G2 / "X182_d1_g2.parquet").is_file() and (C.G2 / "fu2" / "tokens").is_dir())
    # 0-4
    lab = C.LAD / "F_pairs_cr2.csv"
    fp = json.loads((C.LAD / "units" / "L__g2__S__TF__f0.json").read_text()).get("fingerprint", {})
    s_lab = C.sha256(lab)
    res["0-4"] = dict(F_pairs_cr2_sha256=s_lab, unit_label_sha256=fp.get("labels"), ok=s_lab == fp.get("labels"))
    # 0-5
    rec = json.loads((C.REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json").read_text())["units"]
    sp = {}
    for k in range(5):
        f = D1PAIR / "fu2" / "splits" / f"T1__f{k}.json"
        sp[f.name] = dict(now=C.sha256(f) if f.is_file() else None, record=rec.get(f.name, {}).get("sha256"))
    res["0-5"] = dict(splits=sp, ok=all(v["now"] and v["now"] == v["record"] for v in sp.values()))
    # 0-6
    lack = [f"L__g2__S__TF__f{k}{ext}" for k in range(5)
            for ext in [".json", ".parquet"] + [f"__s{s}.pt" for s in range(5)]
            if not (C.LAD / "units" / f"L__g2__S__TF__f{k}{ext}").is_file()]
    res["0-6"] = dict(missing=lack, ok=not lack)
    # 0-7
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_g2"))
    man_sha = C.sha256(C.D1_MANIFEST)
    want = None
    for ln in (C.REPO / "analysis" / "d1_g2" / "g2_common.py").read_text().splitlines():
        if ln.startswith("MANIFEST_SHA"):
            want = ln.split("=", 1)[1].split("#")[0].strip().strip('"')
    res["0-7"] = dict(manifest_sha256=man_sha, g2_common_MANIFEST_SHA=want, ok=man_sha == want)
    # 0-8
    res["0-8"] = dict(free_gb_login_mmlsquota=free_gb, need_gb=30, ok=free_gb >= 30)
    bad = [k for k, v in res.items() if not v["ok"]]
    out = dict(all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: v["ok"] for k, v in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(float(sys.argv[1])))
