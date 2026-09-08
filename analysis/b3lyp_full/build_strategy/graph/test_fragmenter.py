#!/usr/bin/env python3
"""Regression tests for fragmenter.py on known Coley reactions.

Run:  EDA_PROF=<profiles dir> EDA_CSV=<full_dataset.csv> python test_fragmenter.py
Each test pins one behaviour that distinguishes the graph rule from the primary rule.
"""
import os
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragmenter as fr  # noqa: E402

PROF = Path(os.environ["EDA_PROF"])
CSV = Path(os.environ["EDA_CSV"])
SMILES = dict(zip(*pd.read_csv(CSV)[["rxn_id", "rxn_smiles"]].values.T))


def load(rid):
    d = PROF / str(rid)
    ts_fn = sorted(f for f in os.listdir(d) if f.startswith("TS_") and f != "TS_imag_mode.xyz")[0]
    rs = sorted(f for f in os.listdir(d) if re.match(r"^r\d+_", f) and "_alt" not in f)
    return fr.read_xyz(d / ts_fn), fr.read_xyz(d / rs[0]), fr.read_xyz(d / rs[1])


def run(rid):
    ts, r0, r1 = load(rid)
    P = fr.partition(ts, r0, r1)
    audit = fr.audit_forming_bonds(P, ts, r0, r1, SMILES[rid]) if P.status == "ok" else None
    return ts, r0, r1, P, audit


def test_isomeric_reactants_get_distinct_fragments():
    # 4327: dipole and dipolarophile are both C7H10N2O -> formula matching cannot tell them apart
    ts, r0, r1, P, _ = run(4327)
    G0 = fr.heavy_graph(*r0)[0]; G1 = fr.heavy_graph(*r1)[0]
    assert not nx.is_isomorphic(G0, G1, node_match=fr._nm), "reactants are different molecules"
    assert P.status == "ok" and P.A.isdisjoint(P.B) and len(P.A) == 20 and len(P.B) == 20
    assert set(P.map0.values()) == {i for i in P.A if ts[0][i] != "H"}


def test_broken_fragment_is_rejected():
    # 3865: sydnone ring N-O bond is 4.8 A in the TS -> fragment != reactant
    _, _, _, P, _ = run(3865)
    assert P.status == "no_partition"


def test_late_ts_short_forming_bond_is_fine():
    # 3710: forming bond 1.62 A (< 1.25*sum r_cov) merges the fragments under the primary rule
    ts, r0, r1, P, audit = run(3710)
    assert P.status == "ok" and audit.d_formed[0] < 1.8 and audit.foreign == []


def test_foreign_covalent_contact_is_flagged():
    # 3090: two C-O bonds (1.47, 1.59 A) between atoms the SMILES does not connect
    _, _, _, P, audit = run(3090)
    assert P.status == "ok" and len(audit.foreign) == 2


def test_normal_reaction_forming_bonds_cross_partition():
    ts, _, _, P, audit = run(0)
    assert audit.foreign == [] and len(audit.formed_ts) == 2
    assert all((i in P.A) != (j in P.A) for i, j in audit.formed_ts)


def test_charges_from_smiles():
    ts, r0, r1, P, _ = run(3323)
    rmols, _, _ = fr.smiles_reactants_and_formed(SMILES[3323])
    assign = fr.match_smiles_to_reactants(rmols, r0, r1)
    q = fr.formal_charges(rmols, assign)
    assert q == (0, -2)
    assert fr.n_electrons([ts[0][i] for i in P.B], q[1]) % 2 == 0


def test_symmetry_gives_unique_partition():
    # 2983: 432 equivalent embeddings (three CF3 groups etc.) but a single atom set
    _, _, _, P, _ = run(2983)
    assert P.status == "ok" and P.diag["n_maps"] > 100 and not P.diag["cap_hit"]


def test_dipole_role_from_smiles():
    # 4327: dipolarophile is written first in the SMILES; the azomethine ylide contributes 3 ring atoms
    dip, ring = fr.dipole_smiles_index(SMILES[4327])
    assert dip == 1 and len(ring) == 5
    dip0, _ = fr.dipole_smiles_index(SMILES[0])
    assert dip0 in (0, 1)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
