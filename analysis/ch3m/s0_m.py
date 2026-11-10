#!/usr/bin/env python3
"""CH3M 1: start conditions 1-1 .. 1-8 (docs/specs/CH3M_MODELS.md 1) -> results/S0_start.json.
env_d1fu2 on CPU (1-5 builds DeployData('g2')); the job (jobs/s0_m.slurm) passes the env_espley versions json and the
free disk estimate. Exit 0 all pass, 3 = STOP.

  python s0_m.py <espley_versions.json> <free_gb> <method>

1-8 (deploy final model files present) is recorded only (no STOP)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch3m_common as C  # noqa: E402

WANT = dict(x182="cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199",
            manifest="d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72",
            k3="4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e",
            dep_k="a7c30339e46b5455972ba4bb93493d3f50f33db4b77d8bc6f3a83a5b53f58616",
            l_pred="1de11e77ea82571b139d6070ad556386e2a74b482af5c357498614bee37b3e4d")
XGB_VERSIONS = dict(xgboost="3.2.0", sklearn="1.3.1", numpy="1.26.0", pandas="2.1.1", python="3.11.16")
NEED_GB = 20
DEPLOY_MODEL = Path("/gpfs/tmp_cpu2/yeseo1ee/deploy/model")


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main(ver_json, free_gb, method):
    import pandas as pd
    res = {}
    # 1-1
    _, och3 = git("rev-parse", "origin/ch3")
    anc = git("merge-base", "--is-ancestor", C.CH3_SHA, "HEAD")[0] == 0
    res["1-1"] = dict(origin_ch3=och3, want=C.CH3_SHA, ch3_sha_is_ancestor_of_HEAD=anc,
                      ok=och3 == C.CH3_SHA and anc)
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
    # 1-3
    s_k3, s_dk = C.sha256(C.CH3_RES / "K3_predictions.parquet"), C.sha256(C.DEPLOY_RES / "K_predictions.parquet")
    res["1-3"] = dict(k3_sha256=s_k3, want_k3=WANT["k3"], deploy_k_sha256=s_dk, want_deploy_k=WANT["dep_k"],
                      ok=s_k3 == WANT["k3"] and s_dk == WANT["dep_k"])
    # 1-4
    s_l = C.sha256(C.LAD_RES / "L_predictions.parquet")
    rec = C.read_json(C.LAD_RES / "L_report.json").get("predictions_sha256")
    res["1-4"] = dict(l_predictions_sha256=s_l, want=WANT["l_pred"], l_report_predictions_sha256=rec,
                      ok=s_l == WANT["l_pred"] == rec)
    # 1-5
    files = {"labels": C.LABELS, "X182_g2": C.G2 / "X182_d1_g2.parquet", "inputs_g2": C.LAD / "inputs" / "g2.parquet",
             **{f"T1__f{k}": C.D1PAIR / "fu2" / "splits" / f"T1__f{k}.json" for k in range(5)}}
    have = {k: Path(p).is_file() for k, p in files.items()}
    have["G2_tokens_dir"] = (C.G2 / "fu2" / "tokens").is_dir()
    g2 = dict(built=False)
    if all(have.values()):
        import data_ch3 as DC3                       # binds DEP_SCRATCH / DEP_RESULTS (the ch3 regeneration paths)
        D = DC3.DL.DeployData("g2")
        g2 = dict(built=True, n_tokens=int(len(D.rids)), n_pairs_P=len(D.pair_list), pair_list_sha=D.pair_list_sha)
    res["1-5"] = dict(files={k: str(v) for k, v in files.items()}, present=have, deploydata_g2=g2,
                      ok=all(have.values()) and g2["built"])
    # 1-6
    v = json.loads(Path(ver_json).read_text())
    lx = (C.read_json(C.LAD_RES / "L_xgb_report.json").get("versions") or {})
    d1fu2_ok = (C.D1PAIR / "fu2" / "env_d1fu2" / ".d1fu2_env_ok").is_file()
    eq = {k: (str(v.get(k)) == w) for k, w in XGB_VERSIONS.items()}
    res["1-6"] = dict(env_d1fu2_ok=d1fu2_ok, env_espley_versions=v, want=XGB_VERSIONS, equal=eq,
                      L_xgb_report_versions=lx, ok=d1fu2_ok and all(eq.values()))
    # 1-7
    res["1-7"] = dict(free_gb=free_gb, method=method, need_gb=NEED_GB, ok=free_gb >= NEED_GB)
    # 1-8 (record only)
    pts = {f"s{j}": (DEPLOY_MODEL / f"D__lf__S__TF__all__s{j}.pt").is_file() for j in range(5)}
    res["1-8"] = dict(deploy_final_model_dir=str(DEPLOY_MODEL), present=pts, all_present=all(pts.values()),
                      ok=True, note="record only (spec 1-8); used in 9")
    bad = [k for k, x in res.items() if not x["ok"]]
    out = dict(spec="docs/specs/CH3M_MODELS.md 1", written=C.now(), all_pass=not bad, failed=bad, checks=res)
    C.write_json(C.RES / "S0_start.json", out)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{k: x["ok"] for k, x in res.items()},
                          deploy_model_present=res["1-8"]["all_present"]), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], float(sys.argv[2]), sys.argv[3]))
