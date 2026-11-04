#!/usr/bin/env python3
"""tests/a_unit.py — CPU unit checks of mlts_autode.py in the D1 env (python 3.9 + autodE 1.4.5), no GPU, no ORCA run:
  1. mlts_autode / mlts_bridge import under python 3.9 (numpy 2.0) and the hooks install
  2. the ORCA input autodE writes for every keyword set (low_opt, opt, opt_ts, grad, hess, sp, low_sp) after the
     hooks: C-3 wording (no functional / basis / solvent words, ExtOpt on the '!' line), %method block present,
     no %pal (serial), the A-1 OptTS block, modify_internal blocks for an active-bond molecule
  3. the ExtOpt energy line ('FINAL SINGLE POINT ENERGY (From external program) E') through the patched parsers
  4. _seg_text: restart text = same header, new coordinates, MaxIter = cycles left
  5. lists: a3 rule (300 = 60 per fold), d1:<k> sizes, slices disjoint and complete, model_for
Writes only under $MLT/tmp/atest/unit. Run: MLTS_A_BASE=$MLT/tmp/atest bash jobs/sbx.sh d1 tests/a_unit.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLTS_A_BASE", "/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/atest")
import mlt_common as M  # noqa: E402
import mlts_autode as A  # noqa: E402

OUT = M.assert_write_ok(Path(os.environ["MLTS_A_BASE"]) / "unit")
OUT.mkdir(parents=True, exist_ok=True)
res, fails = {}, []


def check(name, cond, info=None):
    res[name] = dict(ok=bool(cond), info=info)
    if not cond:
        fails.append(name)
    print(f"{'PASS' if cond else 'FAIL'} {name} {'' if info is None else json.dumps(info, default=str)[:600]}", flush=True)


# 1 imports / hooks
D = A.d1()
check("py_version", sys.version_info[:2] == (3, 9), sys.version.split()[0])
check("numpy", True, D["np"].__version__)
local = A.local_base() / "mlts_unit_atest"
local.mkdir(parents=True, exist_ok=True)
sock = A.socket_path("unit_atest", "MACE-OMOL-0")
sock.parent.mkdir(parents=True, exist_ok=True)
spec = D["MB"].resolve_spec("MACE-OMOL-0")
A.install(spec, sock, local, "analytic")
cfg = A.base_config(local)
ade = D["pc"].configure_autode(cfg, 4)
check("configure_patched", D["pc"].configure_autode is A.mlts_configure and D["MR"].dft_refine is A.mlts_refine)
check("templates", ade.Config.ts_template_folder_path == str(A.template_dir(spec))         # A2-templates: per model
      and Path(ade.Config.ts_template_folder_path).parent == A.TS_TEMPLATES, ade.Config.ts_template_folder_path)
kw = {k: str(getattr(ade.Config.ORCA.keywords, k)) for k in ("low_opt", "opt", "opt_ts", "grad", "hess", "sp", "low_sp")}
check("keyword_sets", all("ExtOpt" in v for v in kw.values()), kw)

# 2 inputs written by autodE
from autode.calculations import Calculation  # noqa: E402
from autode.wrappers.ORCA import ORCA  # noqa: E402
work = Path(tempfile.mkdtemp(prefix="inp_", dir=str(OUT)))
os.chdir(work)
mol = ade.Molecule(name="ethene", smiles="C=C", solvent_name="water")
texts = {}
for k in kw:
    c = Calculation(name=f"t_{k}", molecule=mol, method=ORCA(), keywords=getattr(ade.Config.ORCA.keywords, k),
                    n_cores=4)
    c.generate_input()
    texts[k] = Path(c.input.filename).read_text()
for k, t in texts.items():
    ok, bad, ext = A.inp_check_text(t)
    check(f"c3_{k}", ok, dict(bad=bad, extopt=ext, first=t.splitlines()[0]))
    check(f"method_block_{k}", "%method" in t and "ProgExt" in t and str(sock) in t)
    check(f"serial_{k}", "%pal" not in t)
    check(f"no_solvent_{k}", "%cpcm" not in t.lower() and "cpcm(" not in t.lower())
check("optts_block", "InHess Read" in texts["opt_ts"] and f'InHessName "{A.INHESS}"' in texts["opt_ts"]
      and "MaxIter 30" in texts["opt_ts"] and "Trust -0.1" in texts["opt_ts"] and "TightOpt" in texts["opt_ts"],
      texts["opt_ts"].split("*xyz")[0])
check("opt_tight", texts["opt"].splitlines()[0].split()[1:4] == ["Opt", "TightOpt", "ExtOpt"], texts["opt"].splitlines()[0])
check("low_opt_maxiter", "MaxIter 10" in texts["low_opt"] and "LooseOpt" in texts["low_opt"])
check("hess_is_engrad", texts["hess"].splitlines()[0].split()[1:3] == ["EnGrad", "ExtOpt"], texts["hess"].splitlines()[0])
# negative control of the C-3 check: the D1 label OptTS header must fail
neg = "! OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J CPCM(Water)\n%cpcm\nsmd true\nend\n*xyz 0 1\nH 0 0 0\n*\n"
ok, bad, ext = A.inp_check_text(neg)
check("c3_negative_control", (not ok) and {"B3LYP", "RIJCOSX", "D3BJ"} <= set(bad) and not ext, bad)
(OUT / "inputs_example.json").write_text(json.dumps(texts, indent=1))

# molecule with an active (forming) bond -> modify_internal blocks
r1 = ade.Molecule(name="r1", smiles="C=C", solvent_name="water")
r1.graph.add_active_edge(0, 1)
c = Calculation(name="t_optts_ts", molecule=r1, method=ORCA(), keywords=ade.Config.ORCA.keywords.opt_ts, n_cores=4)
c.generate_input()
tt = Path(c.input.filename).read_text()
check("optts_modify_internal", "modify_internal" in tt and "{ B 0 1 A }" in tt, tt.split("*xyz")[0][-200:])

# 4 _seg_text
pos = D["np"].asarray(mol.coordinates, float) + 0.01
seg = A._seg_text(texts["opt_ts"], pos, 17)
h0, h1 = texts["opt_ts"].split("*xyz")[0], seg.split("*xyz")[0]
check("seg_text", h1 == h0.replace("MaxIter 30", "MaxIter 17", 1) and seg.count("\n") == texts["opt_ts"].count("\n"),
      seg.split("*xyz")[1][:200])

# 3 energy parsers on the ExtOpt probe output
probe = M.MLT / "tmp" / "extprobe" / "w.out"
if probe.is_file():
    import autode.wrappers.ORCA as AO

    class _O:
        filename = str(probe)
        file_lines = probe.read_text().splitlines()

    class _C:
        output = _O()
    e = ORCA()._energy_from(_C())
    check("energy_extopt_line", abs(float(e) + 76.0) < 1e-9, float(e))
    opt = AO.ORCAOptimiser(output_lines=_O.file_lines + ["FINAL SINGLE POINT ENERGY (From external program)  -76.1"])
    check("last_energy_change", abs(float(opt.last_energy_change) + 0.1) < 1e-9, float(opt.last_energy_change))
else:
    check("energy_extopt_line", False, f"{probe} missing")

# 5 lists
T = A.targets()
rows3 = A.resolve_list("a3")
by = {}
for r in rows3:
    by[r["fold_T1"]] = by.get(r["fold_T1"], 0) + 1
check("a3_list", len(rows3) == 300 and all(v == 60 for v in by.values()), dict(n=len(rows3), by_fold=by,
                                                                               path=str(A.a3_list_path())))
sl = [A.slice_rows(rows3, i, 3) for i in range(3)]
ids = [r["rxn_id"] for s in sl for r in s]
check("slices", sorted(ids) == sorted(r["rxn_id"] for r in rows3) and len(set(ids)) == len(ids),
      [len(s) for s in sl])
check("d1_folds", sum(len(A.resolve_list(f"d1:{k}")) for k in sorted(set(T.fold_T1))) == len(T),
      {int(k): int((T.fold_T1 == k).sum()) for k in sorted(set(T.fold_T1))})
check("model_for", A.model_for(dict(fold_T1=3)) == "FT_3_al" and A.model_for(dict(fold_T1=-1)) == "FT_0_al")
check("roots", A.out_root_for("a3", "Ap") == A.A3_ROOT / "Ap" / "ade" and A.out_root_for("d1:2", "A") == A.ADE_MAIN
      and A.out_root_for("d1:2", "A", main_engine="App") == A.ADE_PRE
      and A.app_roots("d1:1", root="/x/app") == (A.ADE_MAIN, Path("/x/app/ade"))
      and A.app_roots("d1:1") == (A.ADE_PRE, A.ADE_MAIN))
# 6 infra evidence with no server, the timeout publisher, A' without a G2 TS (no GPU needed: fails before any call)
import shutil  # noqa: E402
jd0 = OUT / "infra_jd"
(jd0 / "ts").mkdir(parents=True, exist_ok=True)
dead = A.local_base() / "mlts_unit_atest" / "dead.sock"
ev = A.infra_evidence(jd0, str(dead), 0)
check("infra_server_down", "server_down" in ev, ev)
loc = A.local_base() / "mlts_unit_atest" / "to_local"
(loc / "jobs" / "T99999").mkdir(parents=True, exist_ok=True)
(loc / "jobs" / "T99999" / ".done_ts").write_text("x")
dest = OUT / "timeout" / "jobs" / "T99999"
shutil.rmtree(dest, ignore_errors=True)
A._publish_timeout(loc, dest, "T99999", "> 8 h")
check("timeout_publish", (dest / ".fail_stereo").is_file() and A.stage_state(dest)[1].startswith("stereo_timeout")
      and not loc.exists(), A.stage_state(dest))
noap = OUT / "noap"
shutil.rmtree(noap, ignore_errors=True)
job = A.manifest_rows(["T00348"])["T00348"]
p = A._work_one(job, spec, "Ap", noap, 2, str(dead), "unit_atest", "analytic")
check("ap_no_g2_ts", A.stage_state(p) == ("fail", "ts_no_g2_ts") and (p / "timing.json").is_file()
      and (p / "inputs.tar.gz").is_file() and not A.run_dir("unit_atest", "T00348", arm="Ap").exists(),
      [A.stage_state(p), sorted(x.name for x in p.iterdir())])
shutil.rmtree(A.local_base() / "mlts_unit_atest", ignore_errors=True)
M.write_json(OUT / "a_unit.json", dict(pass_=not fails, fails=fails, checks=res, written=M.now()))
print(json.dumps(dict(pass_=not fails, fails=fails)), flush=True)
sys.exit(0 if not fails else 1)
