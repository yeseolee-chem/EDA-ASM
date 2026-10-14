#!/usr/bin/env python3
"""d1c_control.py — SPEC_D1_ALL gate C (iii): the D0 controls C0020 / C0105 (D0 rxn 20, 105) through the same scripts.

  python d1c_control.py OUT_DIR [--g1-rerun]     (rev 5 venv python, D0 environment: phase_c_step.slurm control)

The D1 rows run the unchanged rev 5 scripts (xtb_slice.process_one, ext_features.py rxn); this job runs exactly
those on the two D0 controls, in the D0 environment (labels_all.json, the D0 G1 geometries $G1_ROOT, input_meta.csv;
nothing of D0 is written: the ext cache goes to a fresh OUT_DIR/control/ext, the G1 rerun to OUT_DIR/control/g1),
and compares with the stored D0 features:
  info     rxn 2434 (INFO_CONTROL): the same comparison, reported only.
  gate     rxn 105: |recomputed − xtb_features_g1.parquet| over the ESPLEY73 columns and |recomputed −
           xtb_features_ext_g1.parquet| over the 109 block columns, max <= D1_CONTROL_TOL; xtb_status / ext_status
           equal. rxn 20 (G1 failed in D0: optts_not_converged): the recomputed xtb_status equals the stored one, and
           the ext parquet has no row (rev 5 computes blocks for G1-ok rows only).
  --g1-rerun  (information, not gated) g1_geom.process from the Coley structures again (ORCA XTB2 OptTS): status vs
           the D0 result.json, max |Δxyz| of the G1 TS vs the D0 G1 TS (an ORCA optimisation need not repeat bit for bit).
Writes OUT_DIR/gate_c_control.json (passed, max_abs_diff, per rxn detail). Exit 0 passed, 1 failed.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
XR = HERE.parent / "espley_xtb_repro"
sys.path.insert(0, str(XR))
import rev5_common as R5  # noqa: E402

for v in ("ESPLEY_LABELS", "ESPLEY_META", "G1_ROOT", "ESPLEY_OUT", "R5_SCRATCH", "ESPLEY_PROF", "ESPLEY_SMILES"):
    if str(R5.D1_FEAT_ROOT) in os.environ.get(v, ""):
        sys.exit(f"STOP: {v}={os.environ[v]} points at the D1 root — the control runs in the D0 environment")
import aggregate as AG  # noqa: E402
import train_ml_single as T  # noqa: E402
import xtb_slice as xs  # noqa: E402

PY = os.environ.get("PY", sys.executable)
INFO_CONTROL = (2434,)          # an extra D0 G1-ok reaction (rev 5 SMOKE_RXNS): compared and reported, not part of the gate


def _f(x):
    return np.nan if x is None else float(x)


def maxdiff(a: dict, b: pd.Series, cols):
    d = {c: 0.0 if np.isnan(_f(a[c])) and np.isnan(_f(b[c])) else abs(_f(a[c]) - _f(b[c])) for c in cols}
    bad = [c for c, x in d.items() if not np.isfinite(x)]
    k = max(d, key=lambda c: d[c] if np.isfinite(d[c]) else np.inf)
    return (float("inf") if bad else d[k]), k, bad


def main():
    out = Path(sys.argv[1]); g1_rerun = "--g1-rerun" in sys.argv
    root = out / "control"
    shutil.rmtree(root / "ext", ignore_errors=True)
    (root / "ext").mkdir(parents=True, exist_ok=True)
    labels = {int(d["rxn_id"]): d for d in json.load(open(xs.LABELS_PATH))}
    meta = xs.load_meta()
    base0 = pd.read_parquet(R5.FEAT_G1).set_index("rxn_id")
    ext0 = pd.read_parquet(R5.FEAT_EXT).set_index("rxn_id")
    e73 = list(T.FEATURE_SETS["ESPLEY73"])
    rep = dict(tol=R5.D1_CONTROL_TOL, features_g1=str(R5.FEAT_G1), features_ext=str(R5.FEAT_EXT),
               features_g1_sha256=R5.sha256(R5.FEAT_G1), features_ext_sha256=R5.sha256(R5.FEAT_EXT), rxns={})
    ok_all, worst = True, 0.0
    plan = [(j, r, True) for j, r in R5.D1_CONTROL.items()] + [(f"info_{r}", r, False) for r in INFO_CONTROL]
    for job, rid, gated in plan:
        rw = 0.0
        r = dict(job_id=job, rxn_id=rid, g1_marker=xs.geom_marker("g1", rid))
        row = AG.derive(pd.DataFrame([xs.process_one(labels[rid], "g1", meta)])).iloc[0].to_dict()   # + is_charged
        r["xtb_status"], r["xtb_status_stored"] = row.get("xtb_status"), base0.loc[rid, "xtb_status"]
        r["xtb_status_equal"] = r["xtb_status"] == r["xtb_status_stored"]
        ok = r["xtb_status_equal"]
        if row.get("xtb_status") == "ok":
            d, k, bad = maxdiff(row, base0.loc[rid], e73)
            num = [c for c in base0.columns if c in row and pd.api.types.is_numeric_dtype(base0[c])
                   and not pd.api.types.is_bool_dtype(base0[c])]
            da, ka, _ = maxdiff(row, base0.loc[rid], num)
            r.update(base_max_abs_diff_espley73=d, base_argmax=k, base_nonfinite=bad,
                     base_max_abs_diff_all_numeric=da, base_argmax_all=ka, n_base_numeric=len(num))
            ok &= d <= R5.D1_CONTROL_TOL; rw = max(rw, d)
        if r["g1_marker"] == ".done":
            p = subprocess.run([PY, str(XR / "ext_features.py"), "rxn", str(rid), "--cache", str(root / "ext")],
                               capture_output=True, text=True, cwd=XR)
            (root / f"ext_{rid}.log").write_text(p.stdout + "\n#### stderr\n" + p.stderr)
            k0 = p.stdout.find('{\n "rxn_id"')                      # cmd_rxn prints the row last (indent=1)
            erow = json.loads(p.stdout[k0:]) if k0 >= 0 else {}
            r["ext_status"], r["ext_status_stored"] = erow.get("ext_status"), ext0.loc[rid, "ext_status"]
            r["ext_status_equal"] = r["ext_status"] == r["ext_status_stored"]
            ok &= r["ext_status_equal"]
            if erow:
                d, k, bad = maxdiff(erow, ext0.loc[rid], R5.EXT_COLS)
                sel = [c for b in R5.ext_sel_blocks() for c in R5.BLOCKS[b]]
                ds, ks, _ = maxdiff(erow, ext0.loc[rid], sel)
                r.update(ext_max_abs_diff_109=d, ext_argmax=k, ext_nonfinite=bad, ext_max_abs_diff_ext_sel=ds, ext_argmax_sel=ks)
                ok &= d <= R5.D1_CONTROL_TOL; rw = max(rw, d)
            else:
                ok = False
        else:
            r["ext_row_stored"] = bool(rid in ext0.index)
            ok &= not r["ext_row_stored"]
        if g1_rerun:
            import g1_geom as G
            groot = root / "g1"
            smiles = G.load_smiles()
            G.process(labels[rid], smiles, G.load_meta(), groot)
            res = json.loads((groot / str(rid) / "result.json").read_text())
            old = json.loads((Path(xs.GEOM_ROOT["g1"]) / str(rid) / "result.json").read_text())
            r["g1_rerun"] = dict(status=res["status"], status_d0=old["status"], equal=res["status"] == old["status"])
            if (groot / str(rid) / "ts.xyz").is_file() and (Path(xs.GEOM_ROOT["g1"]) / str(rid) / "ts.xyz").is_file():
                import fragmenter as fr
                a = fr.read_xyz(groot / str(rid) / "ts.xyz")[1]; b = fr.read_xyz(Path(xs.GEOM_ROOT["g1"]) / str(rid) / "ts.xyz")[1]
                r["g1_rerun"]["ts_max_abs_dxyz_A"] = float(np.abs(a - b).max())
        r.update(passed=bool(ok), gated=gated, max_abs_diff=rw)
        if gated:
            ok_all &= bool(ok); worst = max(worst, rw)
        rep["rxns"][job] = r
    rep["max_abs_diff"] = worst
    rep["passed"] = bool(ok_all)
    R5.write_json(out / "gate_c_control.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
