"""tests/nur_ft_write.py — test job only (compute node), mlts-ft-4 (FT-nurion + FT-cap). Writes only under the test root
$MLT/tmp/nurft4 (MLT_SCRATCH / MLT_RESULTS are pointed there before any import); every input is the production file,
read only (finetune module paths overridden in-process).
  check   ast + pyflakes (or a symtable undefined-name scan) of finetune.py / mlts_bridge.py, import finetune, unit checks
          of cap_set / rr_pick / tier_of on synthetic frames (quotas, shortfall hand-over, round-robin spread, seeds)
  build   a real `finetune.py data FT_0 FT_all_s0` with a pool subsampled to FRAC of its rows and small caps, so the
          written files stay small; then read back train / valid / test (counts, tags, tiers, Nurion rows, energies)
          and run the same command again (must be up to date: nothing rebuilt)."""
import ast
import os
import sys
from collections import Counter
from pathlib import Path

PROD = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
TEST = PROD / "tmp" / "nurft4"
os.environ["MLT_SCRATCH"] = str(TEST)
os.environ["MLT_RESULTS"] = str(TEST / "results")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
FRAC = 0.02
CAPS = ["--cap-train", "4000", "--cap-valid", "400", "--cap-test", "600"]
RUNS = ["FT_0", "FT_all_s0"]


def check():
    files = ["finetune.py", "mlts_bridge.py"]
    for f in files:
        ast.parse((HERE / f).read_text(), f)
    print("ast ok", files, flush=True)
    try:
        from pyflakes import api as PF
        from pyflakes import reporter as PR
        n = sum(PF.checkPath(str(HERE / f), PR.Reporter(sys.stdout, sys.stderr)) for f in files)
        print("pyflakes warnings:", n, flush=True)
    except ImportError:
        import builtins
        import symtable
        for f in files:
            top = symtable.symtable((HERE / f).read_text(), f, "exec")
            bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
            bound |= {c.get_name() for c in top.get_children()}
            bad = set()

            def walk(t):
                for s in t.get_symbols():
                    glob = s.is_global() or t.get_type() == "module"
                    if s.is_referenced() and glob and s.get_name() not in bound and not hasattr(builtins, s.get_name()):
                        bad.add((t.get_name(), s.get_name()))
                for c in t.get_children():
                    walk(c)
            walk(top)
            print("undefined names", f, sorted(bad), flush=True)
    import numpy as np
    import pandas as pd
    import finetune as F
    print("finetune", F.CODE_VERSION, "caps", F.CAP_DEFAULTS, F.CAP_TIER2_FRAC, "TS_NEAR_IF_CLOSE", F.TS_NEAR_IF_CLOSE,
          flush=True)
    ok = []

    def frame(spec):
        """spec: [(tier, n_groups, rows per group list or int)] -> DataFrame tier / grp."""
        rows = []
        for t, groups in spec:
            for g, n in groups.items():
                rows += [dict(tier=t, grp=f"t{t}_{g}")] * n
        return pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)

    g2 = {f"j{i}": (300 if i == 0 else 4) for i in range(50)}            # one long trajectory + 49 short jobs
    g3 = {f"j{i}": 20 for i in range(20)}
    D = frame([(1, {"a": 100}), (2, g2), (3, g3)])
    out, rec = F.cap_set(D, 300, 123, 0)
    c = Counter(out.tier)
    ok.append(("quota 70/30", c[1] == 100 and c[2] == 140 and c[3] == 60, dict(c), rec["quota"]))
    per = Counter(out.grp[out.tier == 2])
    ok.append(("round-robin spread tier 2", max(per.values()) <= 3 and len(per) == 50, per.most_common(3)))
    out2, _ = F.cap_set(D, 300, 123, 0)
    ok.append(("seeded = deterministic", list(out.index) == list(out2.index), None))
    out3, _ = F.cap_set(D, 300, 123, 1)
    ok.append(("other set index -> other draw", list(out.index) != list(out3.index), None))
    Ds = frame([(1, {"a": 100}), (2, {"x": 10}), (3, g3)])
    o, r = F.cap_set(Ds, 300, 1, 0)
    cs = Counter(o.tier)
    ok.append(("short tier 2 gives room to tier 3", cs[2] == 10 and cs[3] == 190 and len(o) == 300, dict(cs), r["quota"]))
    D3 = frame([(1, {"a": 100}), (2, g2), (3, {"y": 5})])
    o, r = F.cap_set(D3, 300, 1, 0)
    c3 = Counter(o.tier)
    ok.append(("short tier 3 gives room to tier 2", c3[3] == 5 and c3[2] == 195, dict(c3), r["quota"]))
    o, r = F.cap_set(D, 5000, 1, 0)
    ok.append(("within the cap: unchanged", len(o) == len(D) and not r["applied"], None))
    o, r = F.cap_set(D, 0, 1, 0)
    ok.append(("cap 0: unchanged", len(o) == len(D) and not r["applied"], None))
    o, r = F.cap_set(D, 50, 1, 0)
    ok.append(("tier 1 over the cap: tier 1 only", len(o) == 100 and set(o.tier) == {1} and r["tier1_over_cap"], None))
    T = pd.DataFrame(dict(input=["I", "NUR", "NUR", "NUR", "NUR", "NUR"],
                          kind=["ts_path", "nur_ts_final", "nur_ts_path", "nur_ts_path", "nur_x_ts_path", "nur_react_opt"],
                          dform=[9.0, 0.5, 0.2, 0.4, 0.1, np.nan]))
    ok.append(("tier_of", list(F.tier_of(T)) == [1, 2, 2, 3, 3, 3], list(F.tier_of(T))))
    ok.append(("FT-1 TS-near incl. nur_", list(F.ts_near_mask(T)) == [False, False, True, False, False, False],
               list(F.ts_near_mask(T))))
    for name, good, *x in ok:
        print(("PASS" if good else "FAIL"), name, *x, flush=True)
    return 0 if all(g for _, g, *_ in ok) else 1


def build():
    import mlt_common as M
    assert M.MLT == TEST and M.RES == TEST / "results", (M.MLT, M.RES)
    import finetune as F
    D, PR = PROD / "dft", HERE / "results"
    F.INPUTS = tuple((n, D / p.name) for n, p in F.INPUTS)
    F.D0_INPUTS = tuple((n, D / p.name) for n, p in F.D0_INPUTS)
    F.AL_INPUT = (F.AL_INPUT[0], D / F.AL_INPUT[1].name)
    F.NUR_INPUT = (F.NUR, D / "nurion.extxyz.gz")
    F.N_REPORT, F.G_REPORT, F.G_REPORT_D0 = PR / "N_report.json", PR / "G_report.json", PR / "G_report_d0.json"
    F.PLAN_D0, F.D0_CLASS, F.D0_HOLDOUT = D / "plan" / "plan_d0.parquet", D / "plan" / "d0_class.csv", \
        D / "plan" / "d0_holdout.csv"
    F.G2_PLAN_D0, F.PLAN_D0_META, F.D0_HOLDOUT_RES = D / "plan" / "g2_plan_d0.json", D / "plan" / "plan_d0_meta.json", \
        PR / "d0_holdout.csv"
    print("test root", F.FT_DATA, F.RES_FT, flush=True)
    base = F.Pool

    class SubPool(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.P = self.P.sample(frac=FRAC, random_state=0).sort_index()
            print(f"[test] pool subsampled to {len(self.P)} rows ({FRAC})", flush=True)
    F.Pool = SubPool
    rc = F.main(["data", *RUNS, *CAPS])
    print("build rc", rc, flush=True)
    bad = []
    for run in RUNS:
        J = M.read_json(F.RES_FT / f"{run}_data.json")
        meta = M.read_json(Path(J["split_dir"]) / "meta.json")
        nur = J.get("nurion") or {}
        print(run, "nurion sha", str(nur.get("extxyz_sha256"))[:12], "N_report sha", str(nur.get("N_report_sha256"))[:12],
              "cap", {k: (v.get("n_before"), v.get("n_after"), v.get("quota")) for k, v in J["cap"].items()
                      if isinstance(v, dict) and "n_before" in v}, "leak", J["leakage_C2"]["pass_"], flush=True)
        if not (nur.get("extxyz_sha256") and nur.get("N_report_sha256")):
            bad.append(f"{run}: sha256 records missing")
        for nm, f in meta["files"].items():
            al = M.read_extxyz(f["path"])
            n_meta = meta["counts"][nm]["n"]
            kinds = Counter(a.info["kind"] for a in al)
            nn = [a for a in al if a.info.get("input") == "NUR"]
            near = sum(1 for a in nn if a.info.get("dform_label_A") is not None and a.info["dform_label_A"] <= 0.3)
            efit = max(abs(a.info["REF_energy"] - a.info["e_eh"] * M.EH2EV) for a in al) if al else 0.0
            tags_ok = all(isinstance(a.info.get(k, ""), str) for a in al for k in ("rxn_ids", "folds", "dipoles"))
            print(f"  {nm}: n {len(al)} (meta {n_meta}) Nurion {len(nn)} (TS-near {near}) by_tier "
                  f"{meta['counts'][nm]['by_tier']} kinds {dict(kinds.most_common(6))} max|REF-e_eh| {efit:.2e} "
                  f"tags str {tags_ok}", flush=True)
            if len(al) != n_meta or efit > 1e-6 or not tags_ok:
                bad.append(f"{run} {nm}: read-back mismatch")
            cap = meta["cap"][nm]
            if cap.get("applied") and len(al) > max(cap["cap"], int(cap["tiers_before"]["1"])):
                bad.append(f"{run} {nm}: over the cap")
    rc2 = F.main(["data", *RUNS, *CAPS])
    J2 = {r: M.read_json(F.RES_FT / f"{r}_data.json")["build"] for r in RUNS}
    print("rerun rc", rc2, {r: v.get("split_built_here") for r, v in J2.items()}, flush=True)
    if any(v.get("split_built_here") for v in J2.values()):
        bad.append("rerun rebuilt a split (not idempotent)")
    print("BUILD", "FAIL " + "; ".join(bad) if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit({"check": check, "build": build}[sys.argv[1]]())
