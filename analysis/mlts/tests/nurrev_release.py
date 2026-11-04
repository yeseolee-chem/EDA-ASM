#!/usr/bin/env python3
"""tests/nurrev_release.py — READ-ONLY review probe (DEPLOY-V2 Nurion, lens B: release safety). Sandbox only:
  bash jobs/sbx.sh reactot tests/nurrev_release.py dry      `inv_nurion.py release <tar> --dry` (the CLI, subprocess) for
                                                           every uploaded result tar; every check must be true
  bash jobs/sbx.sh reactot tests/nurrev_release.py verify   `inv_nurion.py verify <tar>` for every done tar
  bash jobs/sbx.sh reactot tests/nurrev_release.py nodel    post-deletion simulation WITHOUT deleting anything:
        MLT_NUR_UPLOAD -> a directory that does not exist (MLT_NUR_MANI stays the real D1_build/nurion), then scan (must
        skip), release --dry (must refuse), verify_upload (not uploaded), pending, status, verify, collect (up to date) and
        collect --dry --force (full recount, nothing written) must all be harmless.
Never passes --force to scan, never runs release without --dry. Records mtimes of every production file before/after.
Writes only $MLT/tmp/nurrev/release_<mode>.json."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402

OUT = C.MLT / "dft" / "nurion"
REV = C.MLT / "tmp" / "nurrev"
UPLOAD = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
MANI = UPLOAD / "D1_build" / "nurion"


def snapshot():
    s = {}
    for p in sorted(UPLOAD.glob("d1_*.tar")):
        st = p.stat()
        s[str(p)] = (st.st_size, st.st_mtime_ns)
    for p in sorted(OUT.glob("d1_*/*")) + sorted(OUT.glob("d1_*/.done")) + [OUT / "deleted_tars.tsv",
                                                                         C.MLT / "dft" / "nurion.extxyz.gz",
                                                                         C.RES / "N_report.json"]:
        s[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None
    hs = sorted((OUT / "hess").glob("*.npz"))
    s["hess_n"] = len(hs)
    s["hess_max_mtime"] = max(p.stat().st_mtime_ns for p in hs) if hs else None
    return s


def run(args, env=None, timeout=520):
    t0 = time.time()
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([sys.executable, "inv_nurion.py"] + args, cwd=str(HERE), env=e, capture_output=True, text=True,
                       timeout=timeout)
    js = None
    o = r.stdout.strip()
    i = o.find("{")
    if i >= 0:
        try:
            js = json.loads(o[i:])
        except Exception:                                          # noqa: BLE001
            js = None
    return dict(args=args, rc=r.returncode, s=round(time.time() - t0, 1), json=js, stdout=o[-1500:] if js is None else
                None, stderr=r.stderr.strip()[-800:])


def stems():
    return sorted(p.name[:-len(".tar")] for p in UPLOAD.glob("d1_*.tar"))


def main():
    mode = sys.argv[1]
    before = snapshot()
    res = dict(mode=mode, time=C.now(), host=os.uname().nodename)
    if mode == "dry":
        with ThreadPoolExecutor(max_workers=7) as ex:
            rr = list(ex.map(lambda s: run(["release", s, "--dry"]), stems()))
        res["runs"] = rr
        res["all_ok"] = all(r["rc"] == 0 and r["json"] and r["json"]["would_delete"]
                            and all(v for v in r["json"]["checks"].values() if isinstance(v, bool)) for r in rr)
        res["summary"] = {r["args"][1]: dict(rc=r["rc"], would_delete=(r["json"] or {}).get("would_delete"),
                                             file=(r["json"] or {}).get("file"),
                                             false=[k for k, v in ((r["json"] or {}).get("checks") or {}).items() if v is False],
                                             n_checks=len(((r["json"] or {}).get("checks") or {})),
                                             extxyz_n=((r["json"] or {}).get("checks") or {}).get("extxyz_n"), s=r["s"])
                          for r in rr}
    elif mode == "verify":
        done = sorted(p.parent.name for p in OUT.glob("d1_*/.done"))
        with ThreadPoolExecutor(max_workers=7) as ex:
            rr = list(ex.map(lambda s: run(["verify", s]), done))
        res["runs"] = rr
        res["all_ok"] = all(r["rc"] == 0 for r in rr)
    elif mode == "nodel":
        env = dict(MLT_NUR_UPLOAD=str(REV / "no_upload_dir_does_not_exist"), MLT_NUR_MANI=str(MANI))
        assert not Path(env["MLT_NUR_UPLOAD"]).exists()
        rr = [run(["scan", "d1_20261006_g00_001", "--workers", "2"], env),
              run(["scan", "d1_20261010_g01_013", "--workers", "2"], env),
              run(["release", "d1_20261006_g00_001", "--dry"], env),
              run(["release", "d1_20261010_g01_002", "--dry", "--shallow"], env),
              run(["verify_upload", "d1_20261006_g00_001"], env),
              run(["pending"], env),
              run(["status"], env),
              run(["verify", "d1_20261006_g00_001"], env)]
        rr.append(run(["collect", "--dry", "--force", "--workers", "8"], env, timeout=520))
        res["runs"] = rr
        # a plain `collect` would be a no-op iff the stored signature equals the current one (checked read-only here,
        # not by running it: a mismatch would rewrite the production files)
        os.environ.update(env)
        import inv_nurion as N
        rep = C.read_json(N.RES_N)
        res["collect_signature_now"] = N._signature()
        res["collect_signature_stored"] = rep.get("signature")
        res["collect_would_skip"] = res["collect_signature_now"] == rep.get("signature") and N.COLLECT_EXT.exists()
        res["n_report_n_structures"] = rep.get("n_structures")
        cj = rr[-1]["json"] or {}
        res["collect_dry_n_structures"] = cj.get("n_structures")
        res["collect_dry_eq_report"] = cj.get("n_structures") == rep.get("n_structures") and \
            cj.get("kinds") == rep.get("kinds") and cj.get("hessians") == rep.get("hessians")
    after = snapshot()
    res["changed"] = {k: [before.get(k), after.get(k)] for k in set(before) | set(after) if before.get(k) != after.get(k)}
    os.makedirs(REV, exist_ok=True)
    (REV / f"release_{mode}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str))
    for r in res["runs"]:
        j = r["json"]
        print(r["args"], "rc", r["rc"], f"{r['s']} s", "| stderr:", r["stderr"][-300:].replace("\n", " / "))
        if mode == "nodel":
            print("   stdout-json:", json.dumps(j, default=str)[:1500] if j is not None else r["stdout"][-1200:])
        elif mode == "verify":
            print("   ", json.dumps(j, default=str)[:900])


if __name__ == "__main__":
    main()
