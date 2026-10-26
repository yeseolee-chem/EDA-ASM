#!/usr/bin/env python3
"""Phase I: inventory and consistency checks (spec 4). D1 run env, sbatch:  python inventory.py
Writes results/I_inventory.json; exit 3 if any check fails."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402

OUT = []


def chk(name, value, expected, ok):
    OUT.append(dict(check=name, value=value, expected=expected, ok=bool(ok)))
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value} (expected {expected})", flush=True)


PAIR_MAX_A = 4.5   # decision I-1: was 4.0; 11 valid asynchronous label TSs have their long forming bond at 4.02-4.36 A


def pair_ok(xyz, pairs, A):
    """each forming pair joins the two fragments and is 1.4-4.5 A long (0-based index check; decision I-1)."""
    A = set(A)
    return all(((i in A) != (j in A)) and 1.4 <= float(np.linalg.norm(xyz[i] - xyz[j])) <= PAIR_MAX_A
               for i, j in pairs)


def main():
    C = pd.read_csv(GC.C_COMPARE)
    Ci = C.set_index("rxn_id")
    labs1 = GC.load_d1_labels()
    chk("D1 labels", len(labs1), 9266, len(labs1) == 9266)
    T1 = GC.d1_core_table()
    fc = pd.Series([T1.get(int(d["rxn_id"]), (None, -9, None))[1] for d in labs1]).value_counts()
    fc = {int(k): int(v) for k, v in fc.items()}
    chk("D1 fold counts", fc, GC.D1_FOLD_COUNTS, fc == GC.D1_FOLD_COUNTS)
    meta1 = GC.load_meta(GC.D1_META_G2)
    n2 = n1 = 0
    bad_order, bad_pair, elems, dev, n_dev = [], [], set(), 0.0, 0
    for i, d in enumerate(labs1):
        rid = int(d["rxn_id"])
        lab = GC.read_xyz(GC.D1_JOBS / d["job_id"] / "ts.xyz")
        elems |= set(lab[0])
        pairs = GC.parse_pairs(d["formed_pairs_ts"])
        if rid not in meta1 or not pair_ok(lab[1], pairs, meta1[rid]):
            bad_pair.append(f"D1 {rid}")
        for kind, root in (("g2", GC.D1_G2_FINAL), ("g1", GC.D1_G1_FINAL)):
            x = GC.read_if_done(root / str(rid))
            if x is None:
                continue
            n2, n1 = (n2 + 1, n1) if kind == "g2" else (n2, n1 + 1)
            if list(x[0]) != list(lab[0]):
                bad_order.append(f"D1 {rid} {kind}")
        if n_dev < 300 and rid in Ci.index and pd.notna(Ci.at[rid, "d_dft_1"]):
            ref = np.sort([Ci.at[rid, "d_dft_1"], Ci.at[rid, "d_dft_2"]])
            dev = max(dev, float(np.abs(np.sort(GC.form_d(lab[1], pairs)) - ref).max()))
            n_dev += 1
    n2e, n1e = int((C.g2_status == "ok").sum()), int((C.g1_status == "ok").sum())
    chk("D1 G2 .done", n2, n2e, n2 == n2e)
    chk("D1 G1 .done", n1, n1e, n1 == n1e)
    chk("D1 forming distance vs C_compare (300 rxns)", dev, "<= 1e-4", dev <= 1e-4 and n_dev == 300)
    labs0 = GC.load_d0_labels()
    chk("D0 labels", len(labs0), 5260, len(labs0) == 5260)
    meta0, smi = GC.load_meta(GC.D0_META), GC.load_d0_smiles()
    n0g1, bad_role, bad_meta, dev0 = 0, [], [], 0.0
    for L in labs0:
        rid = int(L["rxn_id"])
        pdir = GC.D0_PROF / str(rid)
        lab = GC.read_xyz(pdir / L["ts_file"])
        elems |= set(lab[0])
        pairs = GC.parse_pairs(L["formed_pairs_ts"])
        if rid not in meta0 or not pair_ok(lab[1], pairs, meta0[rid]):
            bad_pair.append(f"D0 {rid}")
        try:
            dev0 = max(dev0, float(np.abs(np.sort(GC.form_d(lab[1], pairs))
                                          - np.sort([float(L["formed_d1"]), float(L["formed_d2"])])).max()))
        except (KeyError, TypeError, ValueError):
            pass
        g1 = GC.read_if_done(GC.D0_G1_ROOT / str(rid))
        if g1 is not None:
            n0g1 += 1
            if list(g1[0]) != list(lab[0]):
                bad_order.append(f"D0 {rid} g1")
        if {L["role1"], L["role2"]} != {"dipole", "dipolarophile"}:
            bad_role.append(rid)
            continue
        nf = {L["role1"]: int(L["n_f1"]), L["role2"]: int(L["n_f2"])}
        if len(meta0.get(rid, [])) != nf["dipole"]:
            bad_meta.append(rid)
        for k in (1, 2):
            if int((pdir / L[f"rel{k}_file"]).read_text().split()[0]) != int(L[f"n_f{k}"]):
                bad_role.append(rid)
    chk("D0 G1 .done", n0g1, ">= 4500", n0g1 >= 4500)
    chk("D0 A_idx = dipole atoms", bad_meta[:20], [], not bad_meta)
    chk("D0 roles / reference files", sorted(set(bad_role))[:20], [], not bad_role)
    chk("D0 formed_d1/2 vs Coley TS (info)", dev0, "info only", True)
    chk("forming pairs join the fragments, 1.4-4.5 A (I-1)", bad_pair[:20], [], not bad_pair)
    chk("element order input = label", bad_order[:20], [], not bad_order)
    chk("elements in ELEM", sorted(elems - set(GC.ELEM)), [], elems <= set(GC.ELEM))
    n_smi = sum(int(L["rxn_id"]) in smi for L in labs0)
    chk("D0 SMILES present", n_smi, 5260, n_smi == 5260)
    man = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False, usecols=["kind", "rxn_smiles"])
    n_d1 = int((man.kind == "D1").sum())
    chk("D1 manifest rows kind D1", n_d1, ">= 9266", n_d1 >= 9266)
    GC.write_json(GC.RES / "I_inventory.json", OUT)
    return 0 if all(o["ok"] for o in OUT) else 3


if __name__ == "__main__":
    sys.exit(main())
