#!/usr/bin/env python3
"""d1_g1.py — D1PAIR Phase G adapter (§7): connect the D1 inputs to the rev 4 G1 code (g1_geom.process) unchanged.

  python d1_g1.py prep            # §7-2 / §7-3: D1 inputs in the rev 4 formats + gates G-1, G-2 + ENV_D1.md (exit 3 = STOP)
  python d1_g1.py g0              # G-0: 5 D0 rxns through the adapter path vs their rev 4 inputs (exit 3 = STOP)
  python d1_g1.py run <i> <n>     # slice i of n of the G1 targets (run order of results/G_targets.csv)
  python d1_g1.py summary         # first-attempt summary -> results/G_g1_summary.csv / .json
  python d1_g1.py infra           # move rows whose ORCA died in orca_util before any cycle (MPI start-up) for a rerun

What the adapter provides (the code reads them through its environment variables; jobs/g_env.sh sets them):
  ESPLEY_PROF   $D1PAIR_SCRATCH/prof/<rxn_id> -> <D1_JOBS>/<job_id>   (label ts_file / rel1_file / rel2_file are relative)
  ESPLEY_META   input_meta_d1.csv: rxn_id, A_idx = the atoms tagged (1) in sp/eda.inp (frag1 = dipole), 0-based
  ESPLEY_SMILES smiles_d1.csv: rxn_id, rxn_smiles with atom maps = TS index + 1 (the D1 rxn_smiles has none):
                reactants = the DFT TS heavy bond graph (fragmenter.heavy_graph) minus the two formed_pairs_ts bonds
                (and any other inter-fragment contact), each component mapped onto the label reactant with the same
                heavy graph (element + H count; networkx isomorphism), so bond orders / charges come from the label
                SMILES; product = the label product mapped onto reactants + the two formed bonds (element + H count,
                else element only; any isomorphism gives the same formed-bond set).
  ESPLEY_LABELS labels_d1_g.json: the G1 target rows of labels_d1.json (no field needed renaming: every field the rev 4-6
                code reads exists under the same name; checked in prep)
  G1_ROOT etc.  new folders under $D1PAIR_SCRATCH (ENV_D1.md)
Gates: G-1 (a) eda.inp coordinates == ts.xyz (same order / elements, <= 1e-6 A), (b) #(1) = n_f1, #(2) = n_f2,
(c) sorted DFT formed-bond lengths == [formed_d1, formed_d2] (1e-3 A), (d) diagnostic r6_partition.dft_redetect_matches;
G-2 the built reactants are isomorphic to the label reactants and fragmenter.audit_forming_bonds on the DFT TS (label
partition, r6_partition.label_partition) finds exactly formed_pairs_ts; > 87 failures = STOP, else those rows are dropped.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402

G_DIR = C.SCRATCH / "G"
PREP_CSV = C.RES / "G_prep.csv"
PREP_JSON = C.RES / "G_prep_report.json"
SNAP_SRC = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/d1_snapshot_d573111e")
SNAP = C.SCRATCH / "d1_snapshot_d573111e"
# label fields read by g1_geom.process, xtb_slice.process_one, ext_features, r6_partition, r6_recover, aggregate.derive
LABEL_FIELDS = ["rxn_id", "ts_file", "rel1_file", "rel2_file", "formed_pairs_ts", "charge1", "charge2", "role1", "role2",
                "n_atoms", "n_f1", "n_f2", "status", "alt_used", "flag_foreign_bond", "flag_async", "d1_kcal", "d2_kcal",
                "e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh", "e_frag1_rel_eh", "e_frag2_rel_eh", "e_bond_kcal",
                "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
# every environment variable read by the rev 4-6 modules the adapter runs (ENV_D1.md); value = what D1PAIR uses
ENV_MODULES = ["g1_geom.py", "xtb_slice.py", "ext_features.py", "aggregate.py", "r6_partition.py", "r6_recover.py",
               "rev5_common.py", "rev6_common.py", "fragmenter.py", "train_ml_single.py", "r6_features.py", "r6_nn.py"]
ENV_RE = re.compile(r"os\.environ(?:\.get|\.setdefault)?\(\s*[\"']([A-Z0-9_]+)[\"']|os\.environ\[\s*[\"']([A-Z0-9_]+)[\"']\s*\]")


def g_env():
    """the environment of the D1 G / F jobs (jobs/g_env.sh exports exactly these)."""
    s = C.SCRATCH
    g1 = C.G1_ROOT_D1F if os.environ.get("D1PAIR_PHASE", "G") == "F" else C.G1_ROOT_D1    # features read g1_final
    return {"ESPLEY_LABELS": str(C.D1G_LABELS), "ESPLEY_PROF": str(C.PROF_D1), "ESPLEY_SMILES": str(C.D1G_SMILES),
            "ESPLEY_META": str(C.D1G_META), "ESPLEY_PARTITION": "label", "G1_ROOT": str(g1),
            "G1_ROOT_REV4": str(C.G1_ROOT_D1), "G1_ROOT_R6": str(C.G1_ROOT_D1R), "G2_ROOT": str(s / "g2_unused"),
            "ESPLEY_OUT": str(s / "xtb"), "R5_SCRATCH": str(s / "r5"), "R6_SCRATCH": str(s / "r6"),
            "R4_SCRATCH": str(s / "r4"), "D1_SNAPSHOT": str(SNAP)}


# ================================================================ eda.inp / mapped SMILES
def parse_eda_inp(path):
    from d1_phase_p import parse_eda_inp as p
    return p(path)


def mapped_smiles(label, ts, tags):
    """(atom-mapped rxn_smiles, info) for one D1 row; raises ValueError(reason) on failure."""
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    from rdkit import Chem
    import fragmenter as fr
    syms, xyz = ts
    G, owner, adj, D, n_unatt = fr.heavy_graph(syms, xyz)
    if n_unatt:
        raise ValueError("h_unattached")
    if any(tags[h] != tags[o] for h, o in owner.items()):
        raise ValueError("h_owner_other_fragment")
    pairs = [tuple(sorted(map(int, p.split("-")))) for p in str(label["formed_pairs_ts"]).split()]
    removed = [e for e in list(G.edges) if tags[e[0]] != tags[e[1]]]
    G.remove_edges_from(removed)
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: tags[c[0]])
    heavy = {k: sorted(i for i in G.nodes if tags[i] == k) for k in (1, 2)}
    if len(comps) != 2 or comps[0] != heavy[1] or comps[1] != heavy[2]:
        raise ValueError(f"ts_graph_components_{len(comps)}")
    r_str, p_str = label["rxn_smiles"].split(">>")
    rmols = [Chem.MolFromSmiles(s) for s in r_str.split(".")]
    pmol = Chem.MolFromSmiles(p_str)
    if any(m is None for m in rmols) or pmol is None or len(rmols) != 2:
        raise ValueError("label_smiles_unparsable")

    def molg(m):
        g = nx.Graph()
        for a in m.GetAtoms():
            g.add_node(a.GetIdx(), lab=(a.GetSymbol(), a.GetTotalNumHs()))
        g.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
        return g
    nm = lambda a, b: a["lab"] == b["lab"]                      # noqa: E731
    out, used = {}, set()
    for k in (1, 2):
        Gk = G.subgraph(heavy[k])
        hit = None
        for i, m in enumerate(rmols):
            if i in used:
                continue
            gm = GraphMatcher(molg(m), Gk, node_match=nm)
            if gm.is_isomorphic():
                hit = (i, dict(gm.mapping))                     # mol idx -> TS idx
                break
        if hit is None:
            raise ValueError(f"frag{k}_not_isomorphic_to_label_reactant")
        used.add(hit[0])
        m = Chem.Mol(rmols[hit[0]])
        for a in m.GetAtoms():
            a.SetAtomMapNum(int(hit[1][a.GetIdx()]) + 1)
        out[k] = m
    Gp = G.copy()
    Gp.add_edges_from(pairs)
    how = "element+H"
    gm = GraphMatcher(molg(pmol), Gp, node_match=nm)
    if not gm.is_isomorphic():
        how = "element"
        gm = GraphMatcher(molg(pmol), Gp, node_match=lambda a, b: a["lab"][0] == b["lab"][0])
        if not gm.is_isomorphic():
            raise ValueError("product_not_isomorphic_to_reactants_plus_formed_bonds")
    pm = Chem.Mol(pmol)
    for a in pm.GetAtoms():
        a.SetAtomMapNum(int(gm.mapping[a.GetIdx()]) + 1)
    smi = f"{Chem.MolToSmiles(out[1])}.{Chem.MolToSmiles(out[2])}>>{Chem.MolToSmiles(pm)}"
    return smi, dict(product_match=how, n_inter_contacts_removed=len(removed) - len([e for e in removed
                                                                                     if tuple(sorted(e)) in pairs]))


# ================================================================ prep
def cmd_prep():
    # D1 snapshot copy (g1_geom imports run_ts / orca_direct / rmsd from it; no import from the rev 4 scratch)
    if not SNAP.is_dir():
        C.assert_write_ok(SNAP)
        shutil.copytree(SNAP_SRC, str(SNAP) + ".tmp")
        os.replace(str(SNAP) + ".tmp", SNAP)
    snap_same = all(C.sha256(p) == C.sha256(SNAP / p.relative_to(SNAP_SRC)) for p in SNAP_SRC.rglob("*") if p.is_file())
    env_paths = check_env()
    import fragmenter as fr
    import r6_partition as RP
    lab = {d["job_id"]: d for d in C.load_d1_labels()}
    tg = pd.read_csv(C.G_TARGETS)
    if len(tg) != C.N_G_TARGETS:
        print(f"STOP: {len(tg)} G targets", file=sys.stderr); return 3
    for p in (G_DIR, C.PROF_D1):
        C.assert_write_ok(p); p.mkdir(parents=True, exist_ok=True)
    missing_fields = sorted({f for d in lab.values() for f in LABEL_FIELDS if f not in d})
    rows, meta, smis, labs = [], [], [], []
    for t in tg.itertuples():
        d = lab[t.job_id]
        rid = int(d["rxn_id"])
        jd = C.D1_JOBS / t.job_id
        link = C.PROF_D1 / str(rid)
        if not link.is_symlink():
            os.symlink(jd, link)
        r = dict(order=t.order, job_id=t.job_id, rxn_id=rid, role=t.role, core=t.core)
        try:
            ts = fr.read_xyz(jd / d["ts_file"])
            s2, tags, x2 = parse_eda_inp(jd / "sp" / "eda.inp")
            r["g1a_same_elements"] = s2 == list(ts[0])
            r["g1a_max_dev_A"] = float(np.abs(x2 - ts[1]).max()) if len(s2) == len(ts[0]) else np.inf
            r["g1b_n1"], r["g1b_n2"] = int((tags == 1).sum()), int((tags == 2).sum())
            r["g1b_ok"] = r["g1b_n1"] == int(d["n_f1"]) and r["g1b_n2"] == int(d["n_f2"])
            pairs = [tuple(map(int, p.split("-"))) for p in str(d["formed_pairs_ts"]).split()]
            Dm = fr.dist_matrix(ts[1])
            dl = sorted(float(Dm[i, j]) for i, j in pairs)
            r["g1c_d"] = dl
            r["g1c_dev_A"] = float(max(abs(a - b) for a, b in zip(dl, [d["formed_d1"], d["formed_d2"]])))
            a_idx = sorted(int(i) for i in np.where(tags == 1)[0])
            meta.append(dict(rxn_id=rid, A_idx=" ".join(map(str, a_idx))))
            st, match = RP.dft_redetect_matches(d, a_idx, prof=C.PROF_D1)
            r["g1d_dft_partition_status"], r["g1d_dft_A_match"] = st, bool(match)
            # G-2
            try:
                smi, info = mapped_smiles(d, ts, tags)
                r.update(g2_smiles_ok=True, **info)
                rel = [fr.read_xyz(jd / d[k]) for k in ("rel1_file", "rel2_file")]
                P, diag = RP.label_partition(d, ts, rel[0], rel[1], a_idx, prof=C.PROF_D1)
                if P is None:
                    r["g2_audit"] = "label_map_failed"
                else:
                    au = fr.audit_forming_bonds(P, ts, rel[0], rel[1], smi)
                    if au is None:
                        r["g2_audit"] = "map_failed"
                    else:
                        same = {tuple(sorted(p)) for p in au.formed_ts} == {tuple(sorted(p)) for p in pairs}
                        r["g2_audit"] = "ok" if same else "formed_differs"
                        r["g2_foreign_dft"] = len(au.foreign)
                r["map_source"] = (diag or {}).get("map_source")
                smis.append(dict(rxn_id=rid, rxn_smiles=smi))
            except ValueError as e:
                r.update(g2_smiles_ok=False, g2_audit=f"smiles:{e}")
        except Exception as e:                                       # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"
        r["g1_ok"] = bool(r.get("g1a_same_elements") and r.get("g1a_max_dev_A", np.inf) <= C.G1_TOL_COORD_A
                          and r.get("g1b_ok") and r.get("g1c_dev_A", np.inf) <= C.G1_TOL_FORM_A)
        r["g2_ok"] = r.get("g2_audit") == "ok"
        rows.append(r)
        labs.append({k: d[k] for k in d})
    df = pd.DataFrame(rows)
    C.write_atomic(PREP_CSV, lambda t: df.to_csv(t, index=False))
    C.write_atomic(C.D1G_META, lambda t: pd.DataFrame(meta).to_csv(t, index=False))
    C.write_atomic(C.D1G_SMILES, lambda t: pd.DataFrame(smis).to_csv(t, index=False))
    C.write_atomic(C.D1G_LABELS, json.dumps(labs))
    g1_bad = df[~df.g1_ok]
    g2_bad = df[df.g1_ok & ~df.g2_ok]
    rep = dict(n=len(df), g1_fail=int(len(g1_bad)), g1_fail_rows=g1_bad.job_id.tolist()[:50],
               g1a_max_dev_A=float(df.g1a_max_dev_A.max()), g1c_max_dev_A=float(df.g1c_dev_A.max()),
               g1d_dft_partition=df.g1d_dft_partition_status.value_counts().to_dict(),
               g1d_dft_A_match=int(df.g1d_dft_A_match.sum()),
               g2_fail=int(len(g2_bad)), g2_fail_reasons=g2_bad.g2_audit.value_counts().to_dict(),
               g2_fail_rows=g2_bad[["job_id", "rxn_id", "role", "g2_audit"]].to_dict("records")[:100],
               g2_fail_roles=g2_bad.role.value_counts().to_dict(),
               product_match=df.get("product_match", pd.Series(dtype=str)).value_counts().to_dict(),
               map_source=df.get("map_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
               foreign_contacts_dft_ts=int((df.get("g2_foreign_dft", pd.Series(dtype=float)).fillna(0) > 0).sum()),
               label_fields_missing=missing_fields, snapshot_copy_identical=bool(snap_same), env_paths=env_paths,
               si_decision=C.SI_DECISION,
               outputs={str(p): C.sha256(p) for p in (C.D1G_META, C.D1G_SMILES, C.D1G_LABELS, PREP_CSV)})
    rep["stop_g1"] = bool(len(g1_bad))
    rep["stop_g2"] = rep["g2_fail"] > C.G2_MAX_FAIL
    rep["stop"] = bool(rep["stop_g1"] or rep["stop_g2"] or missing_fields or not snap_same)
    C.write_json_safe(PREP_JSON, rep)
    write_env_md()
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


def write_env_md():
    """ENV_D1.md: every os.environ read in the rev 4-6 modules + the D1 snapshot, with the D1PAIR value."""
    want = g_env()
    files = [C.ESP_DIR / m for m in ENV_MODULES] + sorted(SNAP_SRC.rglob("*.py"))
    found = {}
    for f in files:
        for k, l in enumerate(f.read_text().splitlines(), 1):
            for m in ENV_RE.finditer(l):
                v = m.group(1) or m.group(2)
                found.setdefault(v, []).append(f"{f.name}:{k}")
    notes = {"ESPLEY_REPO_DATA": "read-only Espley repository data (rev5_common.ESP_DATA); never read by D1PAIR code",
             "ORCA_BIN": "binary, $HOME/orca_6_1_1_avx2/orca (same as rev 4)", "XTB_BIN": "binary, xtb 6.7.1 (same as rev 4)",
             "XTB_THREADS": "1", "OMP_NUM_THREADS": "1 (set by jobs/env.sh; xtb runs set their own)",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "SLURM_CPUS_PER_TASK": "SLURM", "G1_WORKERS": "unset (cpus / 2)",
             "R6_WORKERS": "unset (cpus / 2)", "R6_ONLY": "unset", "R6_SKIP": "unset", "R6_B1_TOL": "unset (rev 5 B1 gate 1e-6 for every rxn)",
             "ESPLEY_NSLICES": "unused (aggregate.main not run)", "TF_CPP_MIN_LOG_LEVEL": "1 (r6_nn default)",
             "ESPLEY_FEAT": "unused (train_ml_single imported for GRIDS / FEATURE_SETS only)", "ESPLEY_ML_OUT": "unused",
             "ESPLEY_ROWS": "unused", "ESPLEY_GEOM": "unused", "ESPLEY_PROTOCOLS": "unused", "ESPLEY_NJOBS": "unused",
             "ESPLEY_ARMS": "unused", "ESPLEY_EXT": "unused", "ESPLEY_TARGET_SET": "unused"}
    lines = ["# ENV_D1 — environment of the D1PAIR G / F jobs (§7-2)", "",
             "Every `os.environ` read found in " + ", ".join(f"`{m}`" for m in ENV_MODULES) +
             " and the D1 snapshot modules (`run_ts.py`, `orca_direct.py`, `rmsd.py`, ...), with the value the D1 jobs "
             "use (`jobs/g_env.sh` exports the G / F values; the F0 / M jobs do not import the G / F modules' paths).",
             "Write locations are checked against the rev 4-6 scratch / results folders (`d1pair_common.FORBIDDEN_WRITE`) "
             "by `check_env` before any G / F computation.", "", "| variable | read at | D1PAIR value |", "|---|---|---|"]
    for v in sorted(found):
        val = want.get(v, notes.get(v, "**not set by D1PAIR** (module default)"))
        lines.append(f"| `{v}` | {', '.join(found[v][:4])}{' …' if len(found[v]) > 4 else ''} | `{val}` |")
    unknown = sorted(v for v in found if v not in want and v not in notes)
    lines += ["", f"Variables without a D1PAIR value: {unknown or 'none'}.", "",
              "Set by `jobs/g_env.sh` for D1PAIR itself (not read by the rev 4-6 modules):",
              f"- `PYTHONPATH=analysis/d1_pair/site_si`, `D1PAIR_ESP_DIR=analysis/espley_xtb_repro`: {C.SI_DECISION}; "
              "`check_env` exits 3 unless the four Si entries are active.",
              "- `PYTHONDONTWRITEBYTECODE=1`; `D1PAIR_SCRATCH` = the scratch root; the ORCA / OpenMPI / xtb variables of "
              "rev 4 `r4_env.sh` unchanged."]
    C.write_atomic(C.RES / "ENV_D1.md", "\n".join(lines) + "\n")
    return unknown


def check_env():
    """exit 3 unless the G / F environment is exactly g_env() and no module write path is a rev 4-6 location."""
    bad = {k: (os.environ.get(k), v) for k, v in g_env().items() if os.environ.get(k) != v}
    if bad:
        print(f"STOP: environment differs from g_env(): {bad}", file=sys.stderr); sys.exit(3)
    import fragmenter as fr
    import rev5_common as R5x
    si = (os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), R5x.BONDI.get("Si"), R5x.GFN2_VALENCE.get("Si"))
    if si != ("1", 1.11, 14, 2.10, 4):
        print(f"STOP: Si element entries not active (site_si/sitecustomize.py): {si}", file=sys.stderr); sys.exit(3)
    import g1_geom as G
    import xtb_slice as xs
    import rev5_common as R5
    import rev6_common as R6
    import aggregate
    paths = dict(g1_geom_G1_ROOT=G.G1_ROOT, xtb_slice_g1=xs.GEOM_ROOT["g1"], xtb_slice_g2=xs.GEOM_ROOT["g2"],
                 R5_SCRATCH=R5.SCRATCH, R5_XTB_ROOT=R5.XTB_ROOT, R5_G1_ROOT=R5.G1_ROOT, R6_SCRATCH=R6.SCRATCH,
                 R6_G1_ROOT_R6=R6.G1_ROOT_R6, R6_XTB_R6=R6.XTB_R6, aggregate_ROOT=aggregate.ROOT,
                 snapshot=Path(G.D1_SNAPSHOT))
    for k, p in paths.items():
        C.assert_write_ok(p)
    assert Path(G.PROF) == C.PROF_D1 and Path(G.META) == C.D1G_META and Path(G.SMILES_CSV) == C.D1G_SMILES
    assert Path(G.LABELS) == C.D1G_LABELS and Path(xs.META) == C.D1G_META and xs.PARTITION_MODE == "label"
    return {k: str(v) for k, v in paths.items()}


# ================================================================ G-0 (D0 side)
def cmd_g0():
    """5 D0 rxns (seed SEED, rev 6 R_feat) through the adapter path (prof symlinks + labels copy + env + g1_geom.process,
    run for real in $D1PAIR_SCRATCH/G0) -> ts.inp / ref start.xyz / xtb program call vs the rev 4 G1 root."""
    rows = pd.read_csv(C.ESP_DIR / "results_rev6" / "R_rows_final.csv")
    feat = sorted(int(r) for r in rows.loc[rows.R_feat, "rxn_id"])
    rng = np.random.default_rng(C.SEED)
    ids = sorted(int(i) for i in rng.choice(feat, size=C.G_N_CHECK_D0, replace=False))
    base = C.SCRATCH / "G0"
    prof, root = base / "prof", base / "g1"
    for p in (prof, root):
        C.assert_write_ok(p); p.mkdir(parents=True, exist_ok=True)
    d0prof = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
    lab = {int(d["rxn_id"]): d for d in json.loads(C.D0_LABELS.read_text())}
    for r in ids:
        if not (prof / str(r)).is_symlink():
            os.symlink(d0prof / str(r), prof / str(r))
    C.write_atomic(base / "labels_g0.json", json.dumps([lab[r] for r in ids]))
    env = {**os.environ, **g_env(), "ESPLEY_LABELS": str(base / "labels_g0.json"), "ESPLEY_PROF": str(prof),
           "ESPLEY_SMILES": str(C.D0_SMILES), "ESPLEY_META": str(C.D0_META), "G1_ROOT": str(root)}
    r = subprocess.run([sys.executable, str(HERE / "d1_g1.py"), "_g0run", ",".join(map(str, ids))], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    rev4 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
    out = []
    for rid in ids:
        rec = dict(rxn_id=rid)
        for rel in ("work/ts.inp", "work/ref1/start.xyz", "work/ref2/start.xyz"):
            a, b = root / str(rid) / rel, rev4 / str(rid) / rel
            if not a.is_file() and not b.is_file():
                rec[rel] = "same"                                # neither run reached this step (same TS failure)
            else:
                rec[rel] = ("same" if a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()
                            else f"differs(adapter={a.is_file()},rev4={b.is_file()})")
        for k in (1, 2):
            calls = []
            for rt in (root, rev4):
                f = rt / str(rid) / "work" / f"ref{k}" / "opt.out"
                m = re.search(r"program call\s*:\s*(.*)", f.read_text(errors="replace")) if f.is_file() else None
                calls.append(m.group(1).strip() if m else None)
            rec[f"ref{k}_program_call"] = "same" if calls[0] == calls[1] else f"differs:{calls}"
        ra = json.loads((root / str(rid) / "result.json").read_text())
        rb = json.loads((rev4 / str(rid) / "result.json").read_text())
        rec.update(status_adapter=ra.get("status"), status_rev4=rb.get("status"),
                   de_ts_eh=(ra.get("e_ts_eh") or np.nan) - (rb.get("e_ts_eh") or np.nan))
        out.append(rec)
    df = pd.DataFrame(out)
    cols = ["work/ts.inp", "work/ref1/start.xyz", "work/ref2/start.xyz", "ref1_program_call", "ref2_program_call"]
    ok = bool((df[cols] == "same").all().all())
    rep = dict(rxn_ids=ids, rows=out, inputs_identical=ok, stop=not ok)
    C.write_json_safe(C.RES / "G0_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if ok else 3


def _g0run(ids):
    import g1_geom as G
    labels = json.loads(Path(G.LABELS).read_text())
    want = [int(x) for x in ids.split(",")]
    labels = [d for d in labels if int(d["rxn_id"]) in want]
    G.run_many(labels, G.G1_ROOT, workers=len(labels))
    return 0


# ================================================================ run / summary
def targets_ok():
    """G targets in run order minus the G-2 failures (labels from ESPLEY_LABELS)."""
    prep = pd.read_csv(PREP_CSV)
    keep = prep[prep.g1_ok & prep.g2_ok].sort_values("order")
    lab = {int(d["rxn_id"]): d for d in json.loads(C.D1G_LABELS.read_text())}
    return [lab[int(r)] for r in keep.rxn_id]


def cmd_run(i, n):
    paths = check_env()
    import g1_geom as G
    labels = targets_ok()[i::n]
    workers = int(os.environ.get("G1_WORKERS", max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // G.ORCA_NPROCS)))
    print(f"[d1 G1 slice {i}/{n}] {len(labels)} rxns, {workers} workers x ORCA nprocs {G.ORCA_NPROCS}; {paths}", flush=True)
    C.G1_ROOT_D1.mkdir(parents=True, exist_ok=True)
    G.run_many(labels, G.G1_ROOT, workers)
    return 0


def cmd_infra():
    """first-attempt rows whose ORCA died in orca_util before any optimisation cycle (MPI start-up, not chemistry):
    their folders are moved to g1_infra/<rid>.<n> (kept) so that `run` redoes them with the same g1_geom.process."""
    check_env()
    rows = []
    dest = C.SCRATCH / "g1_infra"
    C.assert_write_ok(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for lab in targets_ok():
        rid = int(lab["rxn_id"])
        d = C.G1_ROOT_D1 / str(rid)
        out = d / "work" / "ts.out"
        if not (d / ".fail_orca_not_terminated").is_file() or not out.is_file():
            continue
        t = out.read_text(errors="replace")
        if "error termination in orca_util" in t and "GEOMETRY OPTIMIZATION CYCLE" not in t:
            k = len(list(dest.glob(f"{rid}.*")))
            tgt = dest / f"{rid}.{k}"
            os.replace(d, tgt)
            rows.append(dict(rxn_id=rid, moved_to=str(tgt), host=json.loads((tgt / "result.json").read_text()).get("host"),
                             time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    f = C.RES / "G_infra_retry.csv"
    old = pd.read_csv(f) if f.is_file() else pd.DataFrame()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    C.write_atomic(f, lambda t: df.to_csv(t, index=False))
    print(f"[infra] {len(rows)} rows moved for a rerun of the unchanged first attempt; hosts "
          f"{pd.Series([r['host'] for r in rows]).value_counts().to_dict() if rows else {}}")
    return 0


def cmd_summary():
    check_env()
    import g1_geom as G
    prep = pd.read_csv(PREP_CSV).set_index("rxn_id")
    rows = []
    for lab in targets_ok():
        rid = int(lab["rxn_id"])
        f = C.G1_ROOT_D1 / str(rid) / "result.json"
        r = json.loads(f.read_text()) if f.is_file() else {"status": "missing"}
        dd = r.get("dd_form") or [np.nan, np.nan]
        rows.append(dict(rxn_id=rid, job_id=prep.loc[rid, "job_id"], role=prep.loc[rid, "role"], core=prep.loc[rid, "core"],
                         status=r["status"], fail_reasons=";".join(r.get("fail_reasons", [])),
                         imag1_cm=(r.get("imag_freqs_cm") or [np.nan])[0], forming_share=r.get("imag_mode_forming_share"),
                         rmsd_ts_heavy_A=r.get("rmsd_ts_heavy_A"), dd_form_short_A=dd[0], dd_form_long_A=dd[1],
                         partition_match=r.get("partition_match"), wall_s=r.get("wall_s")))
    df = pd.DataFrame(rows)
    C.write_atomic(C.RES / "G_g1_summary.csv", lambda t: df.to_csv(t, index=False))
    okm = df.status == "ok"
    rep = dict(n=len(df), n_ok=int(okm.sum()), n_missing=int((df.status == "missing").sum()),
               fail_counts=df.loc[~okm, "status"].value_counts().to_dict(),
               ref_fail=int((~okm & (df.role == "reference")).sum()),
               core_h=float(df.wall_s.fillna(0).sum() * G.ORCA_NPROCS / 3600))
    C.write_json_safe(C.RES / "G_g1_summary.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 2 if rep["n_missing"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "prep":
        sys.exit(cmd_prep())
    if cmd == "g0":
        sys.exit(cmd_g0())
    if cmd == "_g0run":
        sys.exit(_g0run(a[1]))
    if cmd == "run":
        sys.exit(cmd_run(int(a[1]), int(a[2])))
    if cmd == "summary":
        sys.exit(cmd_summary())
    if cmd == "infra":
        sys.exit(cmd_infra())
    raise SystemExit(__doc__)
