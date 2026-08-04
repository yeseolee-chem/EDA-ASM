#!/usr/bin/env python3
"""
End-to-end verification of the 203-rxn bundle before packaging/shipping.

Checks:
  1. Manifest has expected count.
  2. Every rxn has 5 xyz files + meta.json in geometries/.
  3. Every rxn has 5 .gjf files (1 ts + 2 gs + 2 dist_gs).
  4. Labels parquet has correct rowcount + all target columns present.
  5. Grayson pkl files have correct keys + structure.
  6. mol_types are consistent with fragment_assignment (user's is_A_dipole).
  7. Every 5-atom pick landed on the correct fragment (chemistry sanity).
  8. Kabsch reordering: reactant_i.xyz element sequence matches reactant_i_distorted.xyz.
  9. .gjf files are well-formed Gaussian input syntax.
 10. common_atoms.pkl 5-atom indices point to valid TS atoms.

Prints pass/fail per check; returns 0 if all pass.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd


def load_manifest(p: Path) -> list[tuple[int, str]]:
    rxns = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            rxns.append((int(parts[0]), parts[1]))
    return rxns


def xyz_atom_count(p: Path) -> int:
    return int(p.read_text().splitlines()[0])


def xyz_elements(p: Path) -> list[str]:
    lines = p.read_text().splitlines()
    n = int(lines[0])
    return [ln.split()[0] for ln in lines[2:2 + n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent.parent)
    args = ap.parse_args()

    ROOT = args.root
    failures: list[str] = []
    passed: list[str] = []

    def CHECK(name: str, ok: bool, detail: str = ""):
        if ok:
            passed.append(name)
            print(f"  ✓ {name}")
        else:
            failures.append(f"{name} — {detail}")
            print(f"  ✗ {name} — {detail}")

    # 1. Manifest
    print("\n[1] Manifest")
    rxns = load_manifest(ROOT / "MANIFEST.txt")
    N = len(rxns)
    CHECK("manifest loads", N > 0, "manifest empty")
    CHECK("manifest count matches folder name", N == 203, f"got {N}, expected 203")

    # 2. Geometries
    print("\n[2] Geometries")
    geom_ok = True
    missing = []
    for rn, rid in rxns:
        d = ROOT / "geometries" / rid
        for fn in ["reactant_1.xyz", "reactant_2.xyz", "TS.xyz",
                    "reactant_1_distorted.xyz", "reactant_2_distorted.xyz", "meta.json"]:
            if not (d / fn).exists():
                geom_ok = False
                missing.append(f"{rid}/{fn}")
    CHECK("geometry files complete (5 xyz + meta per rxn)", geom_ok,
          f"missing {len(missing)}: first 3={missing[:3]}" if missing else "")

    # 3. Gaussian inputs
    print("\n[3] Gaussian inputs")
    n_ts = len(list((ROOT / "gaussian_inputs" / "ts").glob("*.gjf")))
    n_gs = len(list((ROOT / "gaussian_inputs" / "gs").glob("*.gjf")))
    n_dist = len(list((ROOT / "gaussian_inputs" / "dist_gs").glob("*.gjf")))
    CHECK(f"ts .gjf count: {n_ts}", n_ts == N)
    CHECK(f"gs .gjf count: {n_gs}", n_gs == N * 2)
    CHECK(f"dist_gs .gjf count: {n_dist}", n_dist == N * 2)

    # 4. Labels
    print("\n[4] DFT labels")
    lbl_2ch = pd.read_parquet(ROOT / "labels" / "labels_2ch_paper.parquet")
    lbl_5ch = pd.read_parquet(ROOT / "labels" / "labels_5channel_paper.parquet")
    CHECK("labels_2ch row count", len(lbl_2ch) == N)
    CHECK("labels_5channel row count", len(lbl_5ch) == N)
    required_cols = {"reaction_number", "reaction_id",
                     "distortion_dft", "interaction_dft", "e_barrier_dft"}
    CHECK("2ch labels have required cols", required_cols.issubset(lbl_2ch.columns))
    max_resid = lbl_5ch["bond_vs_sum_residual_kcal"].max()
    CHECK(f"EDA channel sum residual < 0.5 kcal/mol (max={max_resid:.4f})", max_resid < 0.5)

    # 5. Grayson pkls
    print("\n[5] Grayson pkl files")
    ca = pickle.loads((ROOT / "grayson_pkls" / "common_atoms.pkl").read_bytes())
    mp = pickle.loads((ROOT / "grayson_pkls" / "mapping.pkl").read_bytes())
    mt = pickle.loads((ROOT / "grayson_pkls" / "mol_types.pkl").read_bytes())
    CHECK(f"common_atoms.pkl entries: {len(ca)}", len(ca) == N)
    CHECK(f"mapping.pkl entries: {len(mp)}", len(mp) == N)
    CHECK(f"mol_types.pkl entries: {len(mt)}", len(mt) == N)

    all_keys_match = (set(ca.keys()) == set(mp.keys()) == set(mt.keys()))
    CHECK("pkl outer keys consistent", all_keys_match)

    # common_atoms structure
    ca_ok = True
    for k, v in ca.items():
        if not (set(v.keys()) == {"di", "dp", "reacting", "name"}
                and len(v["di"]) == 3 and len(v["dp"]) == 2 and len(v["reacting"]) == 4
                and v["name"].startswith("ts_")):
            ca_ok = False; break
    CHECK("common_atoms entry structure (di=3, dp=2, reacting=4, name=ts_*)", ca_ok)

    # mol_types values ∈ {'di','dp'}
    mt_ok = all(v["reactant_1"] in {"di", "dp"} and v["reactant_2"] in {"di", "dp"}
                and v["reactant_1"] != v["reactant_2"]
                for v in mt.values())
    CHECK("mol_types values are complementary {di, dp}", mt_ok)

    # 6. mol_types consistency with user is_A_dipole
    print("\n[6] mol_types vs user picks consistency")
    picks_path = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/400_rxns_5_atoms_labelling_dataset/5_atom_picks.json")
    picks = json.loads(picks_path.read_text())
    consistent = 0
    inconsistent = []
    for rn, rid in rxns:
        pick = picks[rid]
        meta = json.loads((ROOT / "geometries" / rid / "meta.json").read_text())
        # user_di_in_reactant should match mol_types
        mt_r1 = mt[str(rn)]["reactant_1"]
        mt_r2 = mt[str(rn)]["reactant_2"]
        user_di_in = meta["user_di_in_reactant"]
        expected_r1 = "di" if user_di_in == "frag1" else "dp"
        if mt_r1 == expected_r1 and mt_r2 != expected_r1:
            consistent += 1
        else:
            inconsistent.append(rid)
    CHECK(f"mol_types matches user di location ({consistent}/{N})", consistent == N,
          f"inconsistent: first 3 = {inconsistent[:3]}")

    # 7. 5-atom picks in correct fragment
    print("\n[7] User 5-atom picks land in correct fragment")
    bad_pick = []
    for rn, rid in rxns:
        pick = picks[rid]
        meta = json.loads((ROOT / "geometries" / rid / "meta.json").read_text())
        frag1_set = set(meta["orca_frag1_ts_idx_0based"])
        frag2_set = set(meta["orca_frag2_ts_idx_0based"])
        di_set = set(pick["di_ts"])
        dp_set = set(pick["dp_ts"])
        user_di_in = meta["user_di_in_reactant"]
        if user_di_in == "frag1":
            if not (di_set.issubset(frag1_set) and dp_set.issubset(frag2_set)):
                bad_pick.append(rid)
        else:
            if not (di_set.issubset(frag2_set) and dp_set.issubset(frag1_set)):
                bad_pick.append(rid)
    CHECK("all 5 picks on correct fragment", len(bad_pick) == 0,
          f"bad rxns: {bad_pick[:3]}")

    # 8. Kabsch reordering: relaxed r element seq matches distorted
    print("\n[8] Relaxed reactant reordered to match distorted")
    bad_reorder = []
    for rn, rid in rxns:
        d = ROOT / "geometries" / rid
        for slot in [1, 2]:
            r_e = xyz_elements(d / f"reactant_{slot}.xyz")
            d_e = xyz_elements(d / f"reactant_{slot}_distorted.xyz")
            if r_e != d_e:
                bad_reorder.append(f"{rid} slot{slot}")
    CHECK(f"element sequences match after reorder", len(bad_reorder) == 0,
          f"mismatches: {bad_reorder[:3]}")

    # 9. .gjf syntax spot-check
    print("\n[9] .gjf format")
    sample_gjf = ROOT / "gaussian_inputs" / "ts" / "ts_0.gjf"
    txt = sample_gjf.read_text()
    checks_gjf = [
        ("has %chk", "%chk=" in txt),
        ("has %mem", "%mem=" in txt),
        ("has route", "#P AM1 Freq" in txt),
        ("has 0 1 charge/mult", re.search(r"^0 1\s*$", txt, re.MULTILINE) is not None),
        ("ends with blank line", txt.endswith("\n\n") or txt.endswith("\n \n")),
    ]
    for name, ok in checks_gjf:
        CHECK(f".gjf {name}", ok)

    # 10. common_atoms indices valid TS range
    print("\n[10] common_atoms indices valid")
    bad_idx = []
    for k, v in ca.items():
        rn = int(k)
        rid = next(rid for (rrn, rid) in rxns if rrn == rn)
        meta = json.loads((ROOT / "geometries" / rid / "meta.json").read_text())
        n_ts = meta["n_atoms_ts"]
        for slot in ["di", "dp", "reacting"]:
            for a in v[slot]:
                if not (1 <= a <= n_ts):
                    bad_idx.append(f"{rid}/{slot}={a}/n_ts={n_ts}")
    CHECK("common_atoms 1-based indices ∈ [1, n_ts]", len(bad_idx) == 0,
          f"bad: {bad_idx[:3]}")

    # Summary
    print(f"\n=== SUMMARY: {len(passed)} passed, {len(failures)} failed ===")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("\nBundle is ready for shipment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
