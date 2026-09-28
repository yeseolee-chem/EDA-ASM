#!/usr/bin/env python3
"""S0 parity gate: the pilot's input path must reproduce the D0 inputs byte for byte.

For each D0 rxn id: take Coley's TS + the references labels_all used (rel1_file/rel2_file),
render the 5 inputs through build_sp_inputs.render(), and compare with
  (a) the repo builder run from scratch (build_inputs_graph.build_one + SMD rewrite), and
  (b) on the cluster, the files that produced labels_all:
        <d0_inputs_standalone>/rxn_XXXX/frag{1,2}_{dist,rel}.inp
        <d0_inputs_eda_smd>/rxn_XXXX/eda.inp
Only the '%pal nprocs' line is normalised. Exit 1 on any difference.
usage: python tests/parity_d0.py --config config.yaml --rxn 20 105 [--csv full_dataset.csv]
"""
import argparse, difflib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pilot_common as pc        # noqa: E402
import build_sp_inputs as bsi    # noqa: E402

NORM = lambda t: re.sub(r"%pal nprocs \d+ end", "%pal nprocs N end", t)   # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--rxn", type=int, nargs="+", required=True)
    ap.add_argument("--csv")
    a = ap.parse_args()
    cfg = pc.load_config(a.config)
    os.environ["EDA_PROF"] = cfg["d0_profiles"]
    fr, _ = pc.import_repo_modules(cfg)
    big, _ = bsi.repo_builders(cfg)
    import pandas as pd
    smi = dict(pd.read_csv(a.csv or cfg["d0_csv"])[["rxn_id", "rxn_smiles"]].values)
    lab = {r["rxn_id"]: r for r in json.load(open(pc.repo_path(cfg, cfg["labels_all"])))}
    bad = 0
    for rid in a.rxn:
        L = lab[rid]; pdir = Path(cfg["d0_profiles"]) / str(rid)
        ts = fr.read_xyz(pdir / L["ts_file"])
        rd, rp = fr.read_xyz(pdir / L["rel1_file"]), fr.read_xyz(pdir / L["rel2_file"])
        P = fr.partition(ts, rd, rp)
        rmols, _, _ = fr.smiles_reactants_and_formed(smi[rid])
        q = fr.formal_charges(rmols, fr.match_smiles_to_reactants(rmols, rd, rp))
        mine = bsi.render(cfg, ts[0], ts[1], P.A, P.B, [rd, rp], q)
        # (a) repo builder from scratch
        (meta, (syms, xyz, A, B, rel, ch)), why = big.build_one(rid, smi[rid], "prefer")
        ref = bsi.render(cfg, syms, xyz, A, B, rel, ch)
        refs = {"repo_builder": ref}
        # (b) files on disk (cluster only)
        d_std = Path(cfg["d0_inputs_standalone"]) / f"rxn_{rid:04d}"
        d_eda = Path(cfg["d0_inputs_eda_smd"]) / f"rxn_{rid:04d}"
        if d_std.is_dir() and d_eda.is_dir():
            refs["d0_files"] = {f: ((d_eda if f == "eda.inp" else d_std) / f).read_text() for f in bsi.SP_FILES}
        else:
            print(f"rxn {rid}: D0 input files not found here ({d_std}) -> compared with the repo builder only")
        for name, R in refs.items():
            for f in bsi.SP_FILES:
                if NORM(mine[f]) != NORM(R[f]):
                    bad += 1
                    print(f"DIFF rxn {rid} {f} vs {name}:")
                    sys.stdout.writelines(list(difflib.unified_diff(NORM(R[f]).splitlines(True),
                                                                    NORM(mine[f]).splitlines(True), n=1))[:20])
                else:
                    print(f"  ok rxn {rid} {f:15s} == {name}")
        errs = bsi.check_method_lines(cfg, mine)
        if errs:
            bad += 1; print(f"g_setting rxn {rid}: {errs}")
    print("PARITY", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
