#!/usr/bin/env python3
"""ABS3 1: start conditions 1-1 .. 1-7 (docs/specs/ABS3_MODELS.md 1) -> results/S0_start.json. Stdlib + pandas (CPU,
env_d1fu2); the job (jobs/s0_abs.slurm) fetches origin/ch3m and passes the env_espley versions json and the free disk
estimate. Exit 0 all pass, 3 = STOP.

  python s0_abs.py <espley_versions.json> <free_gb> <method>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abs_common as C  # noqa: E402

WANT = dict(x182="cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199",
            manifest="d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72",
            labels="f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce",
            labels_json="7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755",
            labels_pq="696d2ec6ee1b75b5f33bad91fcf9368c528644eecdde4c3d226be3200cd447df",
            k3="4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e",
            km="da01baf40b125f75b03cf5f8e79b37468ddd87cca36c0196334952d63ea6ea6e")
XGB_VERSIONS = dict(xgboost="3.2.0", sklearn="1.3.1")
NEED_GB = 10


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main(ver_json, free_gb, method):
    import pandas as pd
    res = {}
    # 1-1
    _, och3m = git("rev-parse", "origin/ch3m")
    anc = git("merge-base", "--is-ancestor", C.CH3M_SHA, "HEAD")[0] == 0
    res["1-1"] = dict(origin_ch3m=och3m, want=C.CH3M_SHA, ch3m_sha_is_ancestor_of_HEAD=anc, ok=och3m == C.CH3M_SHA and anc)
    # 1-2
    tm = C.CH3_RES / "lf_inputs" / "fu2_lf" / "T_manifest.csv"
    s_x, s_m = C.sha256(C.CH3_LF / "X182_lf.parquet"), C.sha256(tm)
    T = pd.read_csv(tm)
    tdir = C.CH3_LF / "fu2_lf" / "tokens"
    bad = [int(r) for r, s in zip(T.rxn_id, T.sha256) if not (tdir / f"{int(r)}.npz").is_file()
           or C.sha256(tdir / f"{int(r)}.npz") != s]
    res["1-2"] = dict(x182_sha256=s_x, want_x182=WANT["x182"], manifest_sha256=s_m, want_manifest=WANT["manifest"],
                      n_tokens=int(len(T)), n_token_files_bad=len(bad), bad_examples=bad[:10],
                      ok=s_x == WANT["x182"] and s_m == WANT["manifest"] and len(T) == 8502 and not bad)
    # 1-3 (= V_report.json outputs)
    V = C.read_json(C.V_REPORT)
    outs = V.get("outputs") or {}
    vrep = {Path(k).name: v for k, v in outs.items()}
    got = dict(labels=C.sha256(C.LABELS), labels_json=C.sha256(C.LABELS_JSON), labels_pq=C.sha256(C.LABELS_PQ))
    names = dict(labels="F_pairs_cr2.csv", labels_json="labels_d1_cr2.json", labels_pq="labels_d1_cr2.parquet")
    res["1-3"] = dict(sha256=got, want={k: WANT[k] for k in got}, V_report_outputs={k: vrep.get(names[k]) for k in got},
                      V_report_all_pass=bool(V.get("all_pass")),
                      ok=all(got[k] == WANT[k] == vrep.get(names[k]) for k in got) and bool(V.get("all_pass")))
    # 1-4
    s_k3, s_km = C.sha256(C.K3_PRED), C.sha256(C.KM_PRED)
    res["1-4"] = dict(k3_sha256=s_k3, want_k3=WANT["k3"], km_sha256=s_km, want_km=WANT["km"],
                      ok=s_k3 == WANT["k3"] and s_km == WANT["km"])
    # 1-5: split files + S_folds = prereg_ch3m.json inputs_sha256 (matched by file name)
    pr = C.read_json(C.CH3M_RES / "prereg_ch3m.json").get("inputs_sha256") or {}
    by_name = {}
    for k, v in pr.items():
        by_name.setdefault(Path(k).name, set()).add(v)
    files = {p.name: p for p in [*C.SPLIT_FILES, C.S_FOLDS]}
    cmp = {n: dict(sha256=C.sha256(p) if p.is_file() else None, prereg_ch3m=sorted(by_name.get(n, set())))
           for n, p in files.items()}
    res["1-5"] = dict(files={n: str(p) for n, p in files.items()}, compare=cmp,
                      ok=all(v["sha256"] is not None and v["prereg_ch3m"] == [v["sha256"]] for v in cmp.values()))
    # 1-6
    v = json.loads(Path(ver_json).read_text())
    d1fu2_ok = (C.D1PAIR / "fu2" / "env_d1fu2" / ".d1fu2_env_ok").is_file()
    eq = {k: (str(v.get(k)) == w) for k, w in XGB_VERSIONS.items()}
    ch3m_s0 = (C.read_json(C.CH3M_RES / "S0_start.json").get("checks") or {}).get("1-6") or {}
    same_as_ch3m = all(str(v.get(k)) == str((ch3m_s0.get("env_espley_versions") or {}).get(k))
                       for k in ("xgboost", "sklearn", "numpy", "pandas", "python"))
    res["1-6"] = dict(env_d1fu2_ok=d1fu2_ok, env_espley_versions=v, want=XGB_VERSIONS, equal=eq,
                      same_as_ch3m_s0=same_as_ch3m, ok=d1fu2_ok and all(eq.values()) and same_as_ch3m)
    # 1-7
    res["1-7"] = dict(free_gb=free_gb, method=method, need_gb=NEED_GB, ok=free_gb >= NEED_GB)
    bad = [k for k, x in res.items() if not x["ok"]]
    out = dict(spec="docs/specs/ABS3_MODELS.md 1", written=C.now(), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: x["ok"] for k, x in res.items()}), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], float(sys.argv[2]), sys.argv[3]))
