#!/usr/bin/env python3
"""d0_labels.py — GEO1b D0D1 step 1: the D1LADDER Phase Q rule on the D0 labels (no QM). sbatch only, conda reactot:

  cd analysis/geo1b && sbatch -p <cpu1|cpu2> jobs/d0lab.slurm d0_labels.py run
  python d0_labels.py run [--force]       # exit 0 = all gates pass (or up to date), 3 = STOP, 2 = usage
  python d0_labels.py probe 0 5 2         # smoke test on a few rxn_ids: keys, states, Q-a; prints JSON, writes nothing

Design: results/D0D1_design_investigation.json ('design' steps[1]) and results/DECISIONS.md 'D0D1'; every judgement call
of this file is in results/DECISIONS_LAB.md (L-*). analysis/d1_ladder/refs_q.py (D1LADDER Phase Q) is imported read
only and its functions are called unchanged through a thin D0 adapter (job_id, Coley profile paths, D0 label inputs).

Rule (the D1 cr2 rule 'lowest label-level reference per molecule key, replaced when > 0.05 kcal/mol', keys within D0):
  * population: the 5,260 status-ok rows of labels_all.json (the 5 non-ok rows = geo_common.D0_EXCLUDE dropped); the
    5 D-1 overlap reactions stay (d0_pairs.py keeps them out of pairs); adapter job_id = D0G2_<rxn_id:04d> (= d0_manifest
    job_id; zero padded, so 'ties: smallest job_id' of refs_q.audit = lowest rxn_id)
  * molecule key of each reference (Q-2): key_design = '<kind>:' + refs_q.mirror_min(map-free reactant SMILES of
    full_dataset.csv) + '|alt' when rel{k}_file is a Coley *_alt.xyz (= the design key), then
      + '|cfg:<state>'  *_alt.xyz (TS-state) dipole references only: Coley's tracked backbone dihedrals
                        (stereo_utils.config_tags) of the reference geometry, each named canonically by
                        refs_q.fixed_state ('?' = sp centre) = the D0 stand-in for the D1 '|fixed:' state, which D1 puts
                        on its constrained (TS-state) references only; the other references pool across rotamers as
                        D1's unconstrained ones do (D0 knows no rotatability: all tracked dihedrals of an alt are used)
      + '|s3d:<sig>'    every reference: refs_q.stereo3d_sig (stereo perceived from the reference geometry, mirror
                        merged) = the stereo the D1 isomeric SMILES carry and the D0 SMILES do not ('@' in none of them)
    a key state that cannot be computed -> singleton key '<key_design>|err:<job_id>' (never replaced, never a source;
    cr.unkeyed); its bond graph must still be its SMILES graph (Q-2m). A bond-graph difference within a key or within
    one SMILES is a STOP (Q-2 / Q-2p, the refs_q rule); the graph classes are in the report for the decision.
    key_cfg_all (report only) = the same with '|cfg' on every dipole reference (cfg_all_key_comparison)
  * dipole component = the stereo_utils.dipole_backbone ring rule (the rule behind d0_manifest dipole_smiles), checked
    against the element formula of the role's reference xyz
  * E_min per key = the lowest e_frag{k}_rel_eh (ties: smallest job_id); replaced iff (E - E_min) x KCAL > 0.05
  * re-assembly through label_true/scripts/stage3_parse.parse_one_rxn (refs_q.assemble adapter); gates Q-a .. Q-e
Gates (all must pass, else STOP = exit 3 and no valid output):
  Q-0   labels_all.json: sha256 = LABELS_SHA (62c0e784...), byte-identical to the main checkout copy, 5,265 rows, non-ok = D0_EXCLUDE
  Q-2   bond graph identical within every final key (refs_q.check_graphs on every reference)
  Q-2p  the same per SMILES-level key (key_plain), every reference (+ graph_classes: majority / minority, report)
  Q-2s  references whose key state (alt cfg / s3d) could not be computed <= 2 %
  Q-2m  each of those references has the bond graph of its SMILES (refs_q.bond_graph vs RDKit AddHs(SMILES))
  Q-2l  label inputs label_true/work/inputs/rxn_NNNN/frag{k}_rel.inp: one level per key, charge and coordinates = xyz
  Q-2e  frag{k}_rel.out terminated normally and its single FINAL SINGLE POINT ENERGY == e_frag{k}_rel_eh (exact)
  Q-a .. Q-e  refs_q.gate_qa / refs_q.gates on the JSON round trip of what is written
Outputs (atomic; written only under $GEO1B/d0 and analysis/geo1b/results):
  $GEO1B/d0/labels_d0_q.json   5,260 rows (rxn_id order) = labels_all.json rows + adapter fields job_id, ex1_kcal,
                               ex2_kcal (identical before and after Q) + 'cr' (Q info: tag, keys, changed fragments,
                               sources, original values, unkeyed = fragments with an |err key). Changed rows carry
                               new e_frag{k}_rel_eh, d1_kcal, d2_kcal, barrier_kcal, identity_residual_kcal,
                               channel_closure_residual_kcal (parse_one_rxn values)
  $GEO1B/d0/Q_keys_d0.parquet  one row per (reaction, fragment): keys, states, energies, e_min, dev, source, replaced,
                               ex_kcal (the $LAD/Q_keys.parquet column names where the meaning is the same)
  $GEO1B/d0/Q_audit_d0.csv     one row per key (refs_q.audit_table)
  results/Q_d0_report.json     gates, counts, the design-key and cfg-all-key comparisons, hygiene before / after, sha256
                               of inputs (fingerprint), outputs
Readers must check Q_d0_report.json all_pass and the output sha256 under 'outputs'. A recompute first writes the report
with all_pass false and renames the previous outputs to <name>.stale; a STOP renames what it wrote; a pass deletes them.
Keys are within D0 (design 'Keys are built within D0'): every D0 key carries '|s3d:' or '|err:', which no D1 key
($LAD/Q_keys.parquet) does, so a union of the two key tables is a plain concatenation (refs='min' per dataset).
Q_keys_d0 'constrained' = alt = 'the key carries a dihedral state' (the role of D1 'constrained' in the key, not D1's
xtb rotatability flag); the states are the alt / fixed_state / cfg_state / stereo3d columns.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # never write .pyc into the read-only LADDER / D1PAIR / D1 code copies

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo_common as GC  # noqa: E402

sys.path.insert(0, str(GC.REPO / "analysis" / "d1_ladder"))
import refs_q as Q  # noqa: E402              (D1LADDER Phase Q, read only; it imports lad_common and d1pair_common)

L, DP, KCAL = Q.L, Q.DP, Q.KCAL
CODE_VERSION = "geo1b-d0q-2"                           # 2: review round (key rule, Q-2 STOP, fingerprint)
TAG = "q_d0"                                           # 'cr' tag of every output row

# ---------------------------------------------------------------- inputs (read only)
D0_MAIN = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")       # main checkout (D0 label build)
D0_MAIN_LABELS = D0_MAIN / "labels_all.json"                              # must equal GC.D0_LABELS byte for byte
D0_MAIN_S3 = D0_MAIN / "label_true" / "scripts" / "stage3_parse.py"       # the D0 label assembly code
D0_INPUTS = GC.D0_META.parent / "inputs"                                  # label_true/work/inputs/rxn_NNNN/frag{k}_rel.*
D0_MANIFEST = Path(os.environ.get("D0_MANIFEST", "/gpfs/tmp_cpu2/yeseo1ee/g2c/d0_r1/d0_manifest.csv"))   # cross-check

# ---------------------------------------------------------------- outputs
D0_DIR = GC.GEO / "d0"                                 # $GEO1B/d0
OUT_LABELS = D0_DIR / "labels_d0_q.json"
OUT_KEYS = D0_DIR / "Q_keys_d0.parquet"
OUT_AUDIT = D0_DIR / "Q_audit_d0.csv"
OUT_REJECTED = D0_DIR / "labels_d0_q.REJECTED.json"   # what would have been written, on a Q-a .. Q-e failure
REPORT = GC.RES / "Q_d0_report.json"
OUTPUTS = (OUT_LABELS, OUT_KEYS, OUT_AUDIT)

# ---------------------------------------------------------------- rules
N_ALL, N_OK = 5265, 5260                               # labels_all.json records / status ok (CLAUDE.md)
LABELS_SHA = "62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753"   # D0D1 investigation labels fact 1
STATE_ERR_MAX_SHARE = 0.02                             # Q-2s
EX_MAX = 20.0                                          # D1PAIR build_rowattr ex rule (report only here; d0_pairs.py)
NEED = (set(Q.SPE_KEY.values()) | set(Q.META_KEYS) | set(Q.CHANGED_KEYS) | set(Q.EDA7) | set(Q.SUM9)
        | {"eda_raw_eh", "status"})                    # label fields the adapter and the gates read
ADAPTER_KEYS = ("job_id", "ex1_kcal", "ex2_kcal")      # added to every row before Q (identical before / after)

DECISIONS = {                                          # keys = the bullet ids of results/DECISIONS_LAB.md
    "L-pop": "5,260 status-ok labels_all.json rows; the 5 D-1 overlaps kept (d0_pairs.py keeps them out of pairs)",
    "L-jobid": "adapter job_id D0G2_<rxn_id:04d>; E_min ties = smallest job_id = lowest rxn_id",
    "L-dipole": "dipole = stereo_utils.dipole_backbone ring rule (the manifest rule), formula-checked vs the xyz",
    "L-key": "key_design = <kind>:mirror_min(SMILES)[|alt]; final key + |cfg:<Coley tracked-dihedral state> "
             "(alt dipole references only, the D1 |fixed analogue) + |s3d:<3D stereo signature> (all); the key_design "
             "and key_cfg_all (|cfg on every dipole) results are in design_key_comparison / cfg_all_key_comparison",
    "L-err": "key state not computable -> |err:<job_id> singleton (cr.unkeyed), cap 2 % (Q-2s), its bond graph = "
             "its SMILES graph (Q-2m), else STOP",
    "L-q2": "Q-2 / Q-2p graph checks (refs_q.check_graphs) on every reference: any difference within a key or a "
            "SMILES = STOP (design, refs_q); graph classes reported for the decision; no isolation",
    "L-level/L-prov": "D5 level / coordinate check on the D0 label inputs (Q-2l); frag_rel.out energy == label "
                      "energy, exact (Q-2e)",
    "L-constrained": "Q_keys_d0 'constrained' := alt = the key carries a dihedral state (role of D1 constrained in "
                     "the key); the alt / fixed_state / cfg_state / stereo3d columns carry the states",
    "L-namespace": "keys within D0: every D0 key has |s3d: or |err:, no D1 key has; a D1 + D0 key union is a "
                   "concatenation (refs='min' per dataset)",
    "L-fingerprint": "inputs, code, both labels_all copies, the D0 stage3_parse and a (path, size, mtime_ns) digest "
                     "of the reference xyz and frag{k}_rel.inp/.out decide 'up to date'",
    "L-ex": "ex{k}_kcal = (E_rel - min E_rel of the same SMILES) x KCAL on the Coley references (build_rowattr); "
            "added to every row before Q; d0_pairs.py applies ex > 20",
    "L-d1only": "not run for D0 (D1 inputs only): hf_precheck, g_vs_e, port / ledger checks, D'' report, Q-6 pairs",
    "L-noV": "no D0 Phase V (design): Q only",
}


class Stop(Exception):
    """a spec STOP: recorded in the report, exit 3."""


def _log(t0, msg):
    print(f"[d0_labels {time.time() - t0:8.1f}s] {msg}", flush=True)


def job_id(rid):
    return f"D0G2_{int(rid):04d}"


def assert_path(p):
    """writes only under $GEO1B/d0 and analysis/geo1b/results; anything else is a STOP."""
    q = Path(os.path.abspath(p))
    for root in (D0_DIR, GC.RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    raise Stop(f"write path {q} outside {D0_DIR} / {GC.RES}")


def write_atomic(path, data):
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _git_head():
    try:
        return L.git("rev-parse", "HEAD").strip()
    except Exception:                                                  # noqa: BLE001
        return None


def unmapped(smi):
    """map numbers removed, RDKit canonical SMILES (= sample_pilot.dipole_of, the d0_manifest rule)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"RDKit cannot parse {smi!r}")
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


# ---------------------------------------------------------------- Q-2: molecule keys
def cfg_state(G, syms, xyz, dmapped, bb, charge):
    """(canonical state, raw config_tags string) of a dipole reference: Coley's tracked dihedrals about the two backbone
    bonds (stereo_utils.config_tags on the reference geometry), each named by refs_q.fixed_state (WL-hash bond ends, c/t
    through the highest-hash heavy neighbour, s = equivalent neighbours); a '?' quad (sp centre) keeps '?'.
    ('', '') = nothing tracked (triple / aromatic backbone bond or no heavy neighbours)."""
    su = Q.stereo_utils()
    tags, _, quads = su.config_tags(su.xyz_block_from_atoms(syms, xyz), dmapped, bb, int(charge))
    parts = []
    for t, quad in zip(tags, quads):
        st, _ = Q.fixed_state(G, syms, xyz, quad)
        parts.append(st if t != "?" else st.rsplit(":", 1)[0] + ":?")
    return ";".join(sorted(parts)), tags


def molecule_keys_d0(rows, smi):
    """Q-2 keys of both references of every row -> (K, graphs {(job_id, frag): nx.Graph}, geoms {(job_id, frag):
    (syms, xyz)}, errors [str]). An error = the key's molecule is in doubt (STOP). A key state (alt cfg / s3d) that
    cannot be computed is no error: the reference gets a singleton key (state_error; Q-2s, Q-2m). cfg_state is computed
    for every dipole reference (report: key_cfg_all, cross-rotamer replacements); a cfg failure of a non-alt reference
    only fills cfg_error (its key has no cfg)."""
    import networkx as nx
    fr = Q.fragmenter()
    su = Q.stereo_utils()
    out, graphs, geoms, errors = [], {}, {}, []
    for row in rows:
        j, rid = row["job_id"], int(row["rxn_id"])
        try:
            mapped = smi.get(rid)
            if not isinstance(mapped, str) or ">>" not in mapped:
                raise ValueError(f"no reaction SMILES for rxn {rid} in {GC.D0_SMILES}")
            comps_m = mapped.split(">>")[0].split(".")
            if len(comps_m) != 2:
                raise ValueError(f"{len(comps_m)} reactant components in {mapped!r}")
            kdip, bb, dmapped = su.dipole_backbone(mapped)
            comps = [unmapped(c) for c in comps_m]
            cform = [Q._formula_smiles(c) for c in comps]
            kinds = {k: row[f"role{k}"] for k in (1, 2)}
            if sorted(kinds.values()) != ["dipolarophile", "dipole"]:
                raise ValueError(f"roles {kinds}")
            comp_of = {k: (kdip if kinds[k] == "dipole" else 1 - kdip) for k in (1, 2)}
            geo = {}
            for k in (1, 2):
                p = GC.D0_PROF / str(rid) / str(row[f"rel{k}_file"])
                syms, xyz = fr.read_xyz(p)
                if len(syms) != int(row[f"n_f{k}"]):
                    raise ValueError(f"{p}: {len(syms)} atoms != n_f{k} = {row[f'n_f{k}']}")
                if cform[comp_of[k]] != Counter(syms):
                    raise ValueError(f"ring-rule {kinds[k]} {comps[comp_of[k]]!r} ({Q._fstr(cform[comp_of[k]])}) does "
                                     f"not have the formula of {p.name} ({Q._fstr(Counter(syms))})")
                geo[k] = (p, list(syms), np.asarray(xyz, float))
            recs = []
            for k in (1, 2):
                p, syms, xyz = geo[k]
                kind, rel = kinds[k], str(row[f"rel{k}_file"])
                alt = rel.endswith("_alt.xyz")
                if alt and kind != "dipole":
                    raise ValueError(f"{rel}: an *_alt.xyz reference for the {kind}")
                if kind == "dipole" and alt != bool(int(row["alt_used"])):
                    raise ValueError(f"alt_used = {row['alt_used']} but the dipole reference is {rel}")
                q = int(row[f"charge{k}"])
                G = Q.bond_graph(syms, xyz)
                smiles = comps[comp_of[k]]
                skey = Q.mirror_min(smiles)
                cfg, raw, cfg_err = "", "", ""
                if kind == "dipole":
                    try:
                        cfg, raw = cfg_state(G, syms, xyz, dmapped, bb, q)
                    except Exception as e:                             # noqa: BLE001
                        cfg_err = f"cfg {type(e).__name__}: {e}"
                s3d = Q.stereo3d_sig(syms, xyz, smiles, q)             # 'error:<type>' instead of raising
                s3d_err = f"s3d {s3d}" if s3d.startswith("error") else ""
                err = "; ".join(x for x in ((cfg_err if alt else ""), s3d_err) if x)     # errors of the final key
                err_all = "; ".join(x for x in (cfg_err, s3d_err) if x)                  # errors of key_cfg_all
                kd = f"{kind}:{skey}" + ("|alt" if alt else "")
                key = f"{kd}|err:{j}" if err else kd + (f"|cfg:{cfg}" if alt and cfg else "") + f"|s3d:{s3d}"
                key_all = f"{kd}|err:{j}" if err_all else kd + (f"|cfg:{cfg}" if cfg else "") + f"|s3d:{s3d}"
                recs.append(dict(
                    job_id=j, rxn_id=rid, frag=k, kind=kind, skeleton="D0", dipole="D0", dataset="D0", smiles=smiles,
                    smiles_key=skey, key_plain=f"{kind}:{skey}", alt=alt, constrained=alt,
                    fixed_state=cfg if alt else "", cfg_state=cfg, cfg_error=cfg_err, cfg_tags_raw=raw, stereo3d=s3d,
                    state_error=err, key_design=kd, key=key, key_cfg_all=key_all,
                    formula=Q._fstr(Counter(syms)), n_atoms=len(syms), charge=q, xyz=str(p), rel_file=rel,
                    wl_graph=nx.weisfeiler_lehman_graph_hash(G, node_attr="el"), ring_check="ring_rule+formula",
                    formula_ambiguous=bool(cform[0] == cform[1])))
                graphs[(j, k)] = G
                geoms[(j, k)] = (syms, xyz)
            out += recs
        except Exception as e:                                         # noqa: BLE001
            errors.append(f"{j}: {type(e).__name__}: {e}")
    return pd.DataFrame(out), graphs, geoms, errors


def graph_minorities(K, graphs):
    """report only (the Q-2p STOP decides): within each SMILES-level key (key_plain) the references split into
    bond-graph classes (element composition, WL hash, networkx isomorphism with node label = element, as
    refs_q.check_graphs); the largest class is the majority (ties: the class holding the smallest job_id), every other
    member is a minority. -> ({(job_id, frag) of the minorities}, [groups with more than one class])."""
    import networkx as nx

    def nm(a, b):
        return a["el"] == b["el"]
    iso, groups = [], []
    for kp, g in K.sort_values(["key_plain", "job_id", "frag"]).groupby("key_plain", sort=True):
        cls = []
        for j, k, f, w in zip(g.job_id, g.frag, g.formula, g.wl_graph):
            jk = (j, int(k))
            for c in cls:
                if c["f"] == f and c["w"] == w and nx.is_isomorphic(graphs[c["rep"]], graphs[jk], node_match=nm):
                    c["m"].append(jk)
                    break
            else:
                cls.append(dict(rep=jk, f=f, w=w, m=[jk]))
        if len(cls) > 1:
            keep = max(range(len(cls)), key=lambda i: (len(cls[i]["m"]), -i))
            out = [jk for i, c in enumerate(cls) if i != keep for jk in c["m"]]
            iso += out
            groups.append(dict(key_plain=kp, n=int(len(g)), class_sizes=[len(c["m"]) for c in cls],
                               majority_rep=f"{cls[keep]['rep'][0]}/frag{cls[keep]['rep'][1]}",
                               minority=[f"{j}/frag{k}" for j, k in out][:20]))
    return set(iso), groups


def smiles_graph(smiles):
    """networkx graph of RDKit AddHs(SMILES) (node attribute el = element), the graph refs_q.bond_graph is compared to."""
    import networkx as nx
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    m = Chem.AddHs(m)
    G = nx.Graph()
    G.add_nodes_from((a.GetIdx(), {"el": a.GetSymbol()}) for a in m.GetAtoms())
    G.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
    return G


def check_err_graphs(K, graphs):
    """Q-2m: a reference with an |err key escapes the within-key comparison, and its state usually fails because RDKit
    template_match finds the geometry's connectivity is not the SMILES graph. So its bond graph (refs_q.bond_graph, the
    graph Q-2 compares) must be isomorphic to the SMILES graph with H (node label = element); else the reference may
    be another molecule (for example a ring-closed ylide) and its Coley energy cannot stay in the labels."""
    import networkx as nx

    def nm(a, b):
        return a["el"] == b["el"]
    E = K[K.state_error != ""]
    bad, cache = [], {}
    for j, k, s, err in zip(E.job_id, E.frag, E.smiles, E.state_error):
        if s not in cache:
            cache[s] = smiles_graph(s)
        if not nx.is_isomorphic(cache[s], graphs[(j, int(k))], node_match=nm):
            bad.append(dict(row=f"{j}/frag{int(k)}", smiles=s, state_error=err))
    return dict(ok=not bad, rule="every |err reference: refs_q.bond_graph isomorphic to RDKit AddHs(SMILES), node label "
                "= element", n_checked=int(len(E)), n_fail=len(bad), fail=bad[:30])


def manifest_check(K):
    """report only: d0_manifest.csv job_id and dipole / dipolarophile SMILES (the components d0_pairs.py uses) vs the
    keys built here (both re-canonicalised by refs_q.mirror_min in this RDKit)."""
    if not D0_MANIFEST.is_file():
        return dict(available=False, path=str(D0_MANIFEST))
    M = pd.read_csv(D0_MANIFEST, dtype=str, keep_default_na=False)
    need = {"job_id", "dipole_smiles", "dipolarophile_smiles", "d0_rxn_id"}
    if need - set(M.columns):
        return dict(available=True, path=str(D0_MANIFEST), ok=False, why=f"columns missing: {sorted(need - set(M.columns))}")
    by = {int(float(r)): (j, d, p) for j, d, p, r in zip(M.job_id, M.dipole_smiles, M.dipolarophile_smiles, M.d0_rxn_id)}
    sk = {(int(r), kd): s for r, kd, s in zip(K.rxn_id, K.kind, K.smiles_key)}
    jb = {int(r): j for r, j in zip(K.rxn_id, K.job_id)}
    miss, bad_job, bad_smi = [], [], []
    for rid in sorted(jb):
        if rid not in by:
            miss.append(rid)
            continue
        j, d, p = by[rid]
        if j != jb[rid]:
            bad_job.append(rid)
        try:
            same = Q.mirror_min(d) == sk[(rid, "dipole")] and Q.mirror_min(p) == sk[(rid, "dipolarophile")]
        except Exception:                                              # noqa: BLE001
            same = False
        if not same:
            bad_smi.append(rid)
    return dict(available=True, path=str(D0_MANIFEST), sha256=L.sha256(D0_MANIFEST), n_rows=int(len(M)),
                n_missing=len(miss), missing=miss[:20], n_job_id_differs=len(bad_job), job_id_differs=bad_job[:20],
                n_smiles_differ=len(bad_smi), smiles_differ=bad_smi[:20], ok=not (miss or bad_job or bad_smi),
                note="report only (d0_pairs.py builds its components from the manifest dipole_smiles)")


def _safe(fn, *a):
    """a report-only section never stops the run."""
    try:
        return fn(*a)
    except Exception as e:                                             # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {e}")


def check_level_d0(K, geoms, by_job):
    """Q-2l = refs_q D5 on the D0 label inputs (label_true/work/inputs/rxn_NNNN/frag{k}_rel.inp): one keyword / block /
    charge / multiplicity signature per key (%maxcore, %pal ignored), charge = the key's charge, coordinates = the
    reference xyz (<= refs_q.XYZ_TOL). Q-2e: frag{k}_rel.out terminated normally and its single FINAL SINGLE POINT
    ENERGY equals the label e_frag{k}_rel_eh exactly (the energy that E_min pools comes from that input)."""
    HC, s3 = Q.hf_common(), Q.stage3()
    sig, bad_xyz, missing, e_bad, e_max, n_lv, n_e = {}, [], [], [], 0.0, 0, 0
    for j, k, key, q, rid in zip(K.job_id, K.frag, K.key, K.charge, K.rxn_id):
        k = int(k)
        d = D0_INPUTS / f"rxn_{int(rid):04d}"
        pi, po = d / f"frag{k}_rel.inp", d / f"frag{k}_rel.out"
        miss = [str(x) for x in (pi, po) if not x.is_file()]
        if miss:
            missing += miss
            continue
        x = HC.parse_inp(pi.read_text())
        s = (tuple(x["bang"]), tuple(sorted((n, tuple(b)) for n, b in x["blocks"].items() if n not in ("maxcore", "pal"))),
             x["charge"], x["mult"])
        sig.setdefault(key, {}).setdefault(s, []).append(j)
        n_lv += 1
        syms, xyz = geoms[(j, k)]
        at = x["atoms"]
        if (x["charge"] != int(q) or len(at) != len(syms) or any(a[0] != e for a, e in zip(at, syms))
                or Q._maxabs(np.array([a[2:] for a in at], float), xyz) > Q.XYZ_TOL):
            bad_xyz.append(f"{j}/frag{k}")
        want = float(by_job[j][Q.REF_E[k]])
        try:
            term = bool(s3.orca_terminated_normally(po))
            e = float(s3.read_fspe(po, 1))
        except ValueError as err:
            e_bad.append(f"{j}/frag{k}: {err}")
            continue
        n_e += 1
        e_max = max(e_max, abs(e - want))
        if not term or e != want:
            e_bad.append(f"{j}/frag{k}: terminated {term}, E {e!r} != label {want!r}")
    mixed = {key: {repr(s): v[:5] for s, v in dct.items()} for key, dct in sig.items() if len(dct) > 1}
    levels = sorted({repr(s[:2]) for dct in sig.values() for s in dct})
    lv = dict(ok=not missing and not bad_xyz and not mixed, root=str(D0_INPUTS), n_checked=n_lv,
              n_missing=len(missing), missing=missing[:20], n_inp_xyz_or_charge_mismatch=len(bad_xyz),
              inp_xyz_or_charge_mismatch=bad_xyz[:30], n_keys_mixed_level=len(mixed),
              keys_mixed_level=dict(list(mixed.items())[:10]), n_levels=len(levels), levels=levels)
    pe = dict(ok=not missing and not e_bad, rule="frag{k}_rel.out ORCA TERMINATED NORMALLY and its single FINAL SINGLE "
              "POINT ENERGY == label e_frag{k}_rel_eh (exact float)", n_checked=n_e, n_missing=len(missing),
              n_fail=len(e_bad), fail=e_bad[:30], max_abs_eh=e_max)
    return lv, pe


# ---------------------------------------------------------------- reports
def dev_counts_d0(A):
    """refs_q.dev_counts without the D1 skeleton / dipole classes."""
    R = A.groupby("job_id").dev_kcal.max()
    out = {}
    for t in Q.THRS:
        rm = A.dev_kcal > t
        out[f"{t:g}"] = dict(reactions=int((R > t).sum()), rows_by_kind=Q._vc(A.kind[rm]),
                             rows_alt=int((rm & A.alt.astype(bool)).sum()), molecules=int(A.key[rm].nunique()),
                             molecules_by_kind={k: int(v) for k, v in A[rm].groupby("kind").key.nunique().items()})
    out["dev_of_rows_above_0.05"] = Q._q(A.dev_kcal[A.dev_kcal > Q.THR_REPLACE])
    return out


def design_compare(K, Ad, A, thr=Q.THR_REPLACE, col="key_design", tag="design",
                   rule="design-literal key = '<kind>:<SMILES>' + '|alt' (no |cfg / |s3d state, no |err singletons)"):
    """report only: the audit Ad on key column `col` (default the design-literal key) against the final key A."""
    fk = {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.key)}
    rd = Ad[Ad.dev_kcal > thr].reset_index(drop=True)
    cross = np.array([fk[(s, int(sf))] != fk[(j, int(f))]
                      for j, f, s, sf in zip(rd.job_id, rd.frag, rd.src_job, rd.src_frag)], bool)
    fin = {(j, int(f)) for j, f, d in zip(A.job_id, A.frag, A.dev_kcal) if d > thr}
    dsg = {(j, int(f)) for j, f in zip(rd.job_id, rd.frag)}
    ex = rd[cross].head(15)
    return {"rule": rule, f"n_keys_{tag}": int(K[col].nunique()), "n_keys_final": int(K.key.nunique()),
            f"n_replaced_{tag}": int(len(rd)), "n_replaced_final": len(fin), f"n_{tag}_not_in_final": len(dsg - fin),
            f"n_final_not_in_{tag}": len(fin - dsg),
            f"n_{tag}_replacements_across_final_keys": int(cross.sum()),
            "across_by_kind": Q._vc(rd.kind.to_numpy()[cross]) if cross.any() else {},
            "across_alt": int(rd.alt.to_numpy()[cross].astype(bool).sum()) if cross.any() else 0,
            "across_dev_kcal": Q._q(rd.dev_kcal.to_numpy()[cross]),
            "examples": [dict(row=f"{j}/frag{int(f)}", src=f"{s}/frag{int(sf)}", dev_kcal=float(d),
                              key=fk[(j, int(f))], src_key=fk[(s, int(sf))])
                         for j, f, s, sf, d in zip(ex.job_id, ex.frag, ex.src_job, ex.src_frag, ex.dev_kcal)]}


def cross_cfg(K, A, thr=Q.THR_REPLACE):
    """report only: final replacements of non-alt dipole references whose source has another cfg_state (both
    computable) = the cross-rotamer pooling that D1 allows for its unconstrained references. It is also the upper bound
    of the residual risk that D0 cannot see: a Coley original that was already in a constrained TS state (D1 would key
    it |fixed) pooled with another state of the same SMILES."""
    st = {(j, int(f)): (c, e) for j, f, c, e in zip(K.job_id, K.frag, K.cfg_state, K.cfg_error)}
    r = A[(A.dev_kcal > thr) & (A.kind == "dipole") & ~A.alt.astype(bool)].reset_index(drop=True)
    m = np.array([st[(j, int(f))][1] == "" and st[(s, int(sf))][1] == "" and st[(j, int(f))][0] != st[(s, int(sf))][0]
                  for j, f, s, sf in zip(r.job_id, r.frag, r.src_job, r.src_frag)], bool)
    ex = r[m].head(15)
    return dict(n_nonalt_dipole_replaced=int(len(r)), n_across_cfg=int(m.sum()),
                n_cfg_unknown=int(sum(st[(j, int(f))][1] != "" or st[(s, int(sf))][1] != ""
                                      for j, f, s, sf in zip(r.job_id, r.frag, r.src_job, r.src_frag))),
                across_dev_kcal=Q._q(r.dev_kcal.to_numpy()[m]),
                examples=[dict(row=f"{j}/frag{int(f)}", src=f"{s}/frag{int(sf)}", dev_kcal=float(d),
                               cfg=st[(j, int(f))][0], src_cfg=st[(s, int(sf))][0])
                          for j, f, s, sf, d in zip(ex.job_id, ex.frag, ex.src_job, ex.src_frag, ex.dev_kcal)])


def reassembly_summary(rows, new_rt, ov, K):
    chg = [(o, n) for o, n in zip(rows, new_rt) if n["cr"]["changed"]]
    kinds = {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.kind)}
    alts = {(j, int(f)): bool(a) for j, f, a in zip(K.job_id, K.frag, K.alt)}

    def dq(t):
        return Q._q([n[t] - o[t] for o, n in chg if n[t] != o[t]])
    return dict(n_rows_changed=len(chg), by_changed=dict(Counter("+".join(n["cr"]["changed"]) for _, n in chg)),
                n_fragments_changed=len(ov), fragments_by_kind=dict(Counter(kinds[jk] for jk in ov)),
                n_alt_fragments_changed=int(sum(alts[jk] for jk in ov)),
                n_charged_rows_changed=int(sum(bool(int(o["charge1"]) or int(o["charge2"])) for o, _ in chg)),
                delta_d1=dq(Q.D_KEY[1]), delta_d2=dq(Q.D_KEY[2]), delta_barrier=dq(Q.BARRIER_KEY),
                n_source_molecules=len({o["key"] for o in ov.values()}))


def hygiene_summary(rows, new_rt):
    def mask(rr):
        return np.asarray(DP.hygiene_mask([r[Q.D_KEY[1]] for r in rr], [r[Q.D_KEY[2]] for r in rr]), bool)
    h0, h1 = mask(rows), mask(new_rt)
    rid = np.array([int(r["rxn_id"]) for r in rows])
    ex = np.array([[r["ex1_kcal"], r["ex2_kcal"]] for r in rows], float)
    return dict(rule="R6.hygiene_mask (d1 >= 0, d2 >= 0, d2 <= 50); d0_pairs.py applies it to these Q labels",
                n_fail_coley=int((~h0).sum()), n_fail_q=int((~h1).sum()), fail_q=rid[~h1].tolist(),
                now_pass=rid[~h0 & h1].tolist(), now_fail=rid[h0 & ~h1].tolist(),
                ex_rule=f"ex{{k}}_kcal > {EX_MAX:g} (D1PAIR build_rowattr; applied by d0_pairs.py)",
                n_ex1_gt=int((ex[:, 0] > EX_MAX).sum()), n_ex2_gt=int((ex[:, 1] > EX_MAX).sum()),
                ex_gt=rid[(ex > EX_MAX).any(1)].tolist()[:50])


# ---------------------------------------------------------------- run
def fingerprint():
    files = [GC.D0_LABELS, GC.D0_SMILES, Path(__file__).resolve(), HERE / "geo_common.py", Path(Q.__file__).resolve(),
             Path(L.__file__).resolve(), DP.HERE / "d1pair_common.py", Q.s3_dir() / "stage3_parse.py",
             Q.s3_dir() / "_common.py", Q.g2c().PILOT / "stereo_utils.py", DP.ESP_DIR / "fragmenter.py",
             L.HF_DIR / "hf_common.py"]
    out = {str(p): L.sha256(p) for p in files}
    for p in (D0_MANIFEST, D0_MAIN_LABELS, D0_MAIN_S3):    # Q-0 (main-checkout copy), assembly comparison, manifest
        out[str(p)] = L.sha256(p) if p.is_file() else None
    out["per_reaction_files"] = per_reaction_digest()
    return out


def per_reaction_digest():
    """(path, size, mtime_ns) digest of the per-reaction files the gates and keys read but the fingerprint cannot hash
    cheaply (about 31.5k files, 650 MB): the Coley reference xyz rel{k}_file and label_true/work/inputs/rxn_NNNN/
    frag{k}_rel.inp / .out of every status-ok labels_all.json row (Q-2, Q-2l, Q-2e). A rerun, copy or edit of one of
    them changes the digest, so 'up to date' is never claimed on inputs the last pass did not see."""
    import hashlib
    h, n, miss = hashlib.sha256(), 0, 0
    for d in sorted(json.loads(GC.D0_LABELS.read_text()), key=lambda d: int(d["rxn_id"])):
        rid = int(d["rxn_id"])
        if rid in GC.D0_EXCLUDE or d.get("status") != "ok":
            continue
        paths = [GC.D0_PROF / str(rid) / str(d.get(f"rel{k}_file")) for k in (1, 2)]
        paths += [D0_INPUTS / f"rxn_{rid:04d}" / f"frag{k}_rel.{x}" for k in (1, 2) for x in ("inp", "out")]
        for p in paths:
            n += 1
            try:
                s = os.stat(p)
                h.update(f"{p}\t{s.st_size}\t{s.st_mtime_ns}\n".encode())
            except OSError:
                miss += 1
                h.update(f"{p}\tmissing\n".encode())
    return dict(sha256=h.hexdigest(), n_files=n, n_missing=miss)


def _invalidate():
    """rename the existing outputs to <name>.stale; returns the renamed paths."""
    moved = []
    for p in OUTPUTS:
        if p.is_file():
            q = assert_path(p)
            os.replace(q, q.with_name(q.name + ".stale"))
            moved.append(str(q))
    return moved


def _up_to_date(fp):
    """the last report passed with the same inputs / code and its outputs are unchanged."""
    if not REPORT.is_file():
        return False
    try:
        o = json.loads(REPORT.read_text())
    except ValueError:
        return False
    return bool(o.get("all_pass") and o.get("fingerprint") == fp and o.get("outputs")
                and all(Path(p).is_file() and L.sha256(p) == s for p, s in o["outputs"].items()))


def cmd_run(force=False):
    t0 = time.time()
    rep = dict(phase="GEO1b D0D1 step 1: LADDER Q rule on D0", code_version=CODE_VERSION, refs_q_version=Q.CODE_VERSION,
               started=time.strftime("%F %T"), git_head=_git_head(), decisions=DECISIONS, stop=False, all_pass=False,
               gates={})
    reason = None
    try:
        fp = fingerprint()                                 # a missing input / code file = STOP (report below)
        if not force and _up_to_date(fp):
            print(f"D0 Q up to date ({REPORT}); --force to recompute", flush=True)
            return 0
        rep.update(env=Q.env_info(), fingerprint=fp)
        rep["invalidated_at_start"] = _invalidate()        # a recompute never leaves an earlier pass that looks valid
        if OUT_REJECTED.is_file():
            assert_path(OUT_REJECTED).unlink()
        write_json(REPORT, dict(rep, running=True))
        _run(rep, t0)
        return 0
    except Stop as e:
        reason = str(e)
    except SystemExit as e:                                            # L.stop / DP checks inside an imported helper
        reason = f"exit {e.code} inside an imported helper (message on stderr of the job log)"
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        reason = f"unexpected error {type(e).__name__}: {e} (traceback in the job log)"
    rep.update(stop=True, all_pass=False, stop_reason=reason, finished=time.strftime("%F %T"),
               wall_s=round(time.time() - t0, 1))
    rep["invalidated_on_stop"] = _invalidate()
    write_json(REPORT, rep)
    print(f"STOP: {reason}", file=sys.stderr, flush=True)
    return 3


def _run(rep, t0):
    G = rep["gates"]
    s3 = Q.stage3()
    rep["found"] = dict(
        assembly="label_true/scripts/stage3_parse.py::parse_one_rxn through refs_q.assemble (readers swapped for the "
                 "label energies, parse_one_rxn unchanged)",
        assembly_sha256={"stage3_parse.py": L.sha256(Path(s3.__file__)),
                         "_common.py": L.sha256(Path(sys.modules["_common"].__file__))},
        assembly_same_as_d0_build=bool(D0_MAIN_S3.is_file() and L.sha256(Path(s3.__file__)) == L.sha256(D0_MAIN_S3)),
        d0_inputs=str(D0_INPUTS), profiles=str(GC.D0_PROF), smiles=str(GC.D0_SMILES))

    # Q-0 labels_all.json
    allr = json.loads(GC.D0_LABELS.read_text())
    sha = L.sha256(GC.D0_LABELS)
    sha_main = L.sha256(D0_MAIN_LABELS) if D0_MAIN_LABELS.is_file() else None
    nonok = sorted(int(d["rxn_id"]) for d in allr if d.get("status") != "ok")
    rids = [int(d["rxn_id"]) for d in allr]
    pre = sorted({k for d in allr for k in ADAPTER_KEYS + ("cr",) if k in d})
    G["Q-0"] = dict(path=str(GC.D0_LABELS), sha256=sha, main_checkout=str(D0_MAIN_LABELS), sha256_main=sha_main,
                    sha256_expected=LABELS_SHA, n_records=len(allr), non_ok=nonok,
                    expected_non_ok=sorted(GC.D0_EXCLUDE), rxn_id_unique=len(set(rids)) == len(rids),
                    max_rxn_id=max(rids), adapter_keys_already_present=pre)
    G["Q-0"]["ok"] = bool(sha == LABELS_SHA and sha == sha_main and len(allr) == N_ALL
                          and nonok == sorted(GC.D0_EXCLUDE) and len(set(rids)) == len(rids) and max(rids) < 10000
                          and not pre)
    if not G["Q-0"]["ok"]:
        raise Stop("Q-0: labels_all.json is not the expected file (sha256 / main-checkout copy / 5,265 rows / non-ok ids "
                   "/ adapter keys); see gates Q-0")
    rows = sorted((d for d in allr if int(d["rxn_id"]) not in GC.D0_EXCLUDE), key=lambda d: int(d["rxn_id"]))
    miss = sorted({k for r in rows for k in NEED if k not in r})
    if len(rows) != N_OK or miss:
        raise Stop(f"{len(rows)} rows (expected {N_OK}); label keys missing: {miss}")
    for r in rows:
        r["job_id"] = job_id(r["rxn_id"])
    by_job = {r["job_id"]: r for r in rows}
    roles = Counter((r["role1"], r["role2"]) for r in rows)
    rep["population"] = dict(
        n_rows=len(rows), excluded=sorted(GC.D0_EXCLUDE), roles={f"{a}/{b}": n for (a, b), n in roles.items()},
        n_alt_used=int(sum(int(r["alt_used"]) for r in rows)),
        n_charged=int(sum(bool(int(r["charge1"]) or int(r["charge2"])) for r in rows)),
        n_flag_async=int(sum(bool(r["flag_async"]) for r in rows)),
        note="the 5 D-1 overlap reactions are kept here (reference sources); d0_pairs.py keeps them out of pairs")
    smi = GC.load_d0_smiles()
    _log(t0, f"{len(rows)} D0 rows; molecule keys")

    # Q-2 keys, states, singletons, graph checks (every reference; a graph difference is a STOP as in refs_q)
    K, graphs, geoms, errs = molecule_keys_d0(rows, smi)
    if errs:
        rep["keys"] = dict(n_errors=len(errs), errors=errs[:50])
        raise Stop(f"Q-2: molecule keys could not be built for {len(errs)} reactions, first: {errs[0]}")
    n_ref = int(len(K))
    n_err = int((K.state_error != "").sum())
    G["Q-2s"] = dict(ok=bool(n_err <= STATE_ERR_MAX_SHARE * n_ref), n_state_errors=n_err, n_references=n_ref,
                     max_share=STATE_ERR_MAX_SHARE,
                     errors=K.loc[K.state_error != "", ["job_id", "frag", "kind", "state_error"]].head(30)
                     .to_dict(orient="records"))
    G["Q-2"] = Q.check_graphs(K, graphs, by="key")
    G["Q-2p"] = Q.check_graphs(K, graphs, by="key_plain")
    minor, cls_groups = graph_minorities(K, graphs)            # report: which references a Q-2p STOP is about
    G["Q-2p"]["graph_classes"] = dict(n_smiles_with_several_graphs=len(cls_groups), n_minority_references=len(minor),
                                      groups=cls_groups[:30])
    G["Q-2m"] = check_err_graphs(K, graphs)
    _log(t0, f"keys {K.key.nunique()} (design {K.key_design.nunique()}, cfg-all {K.key_cfg_all.nunique()}), state "
             f"errors {n_err}, graph ok={G['Q-2']['ok']}/{G['Q-2p']['ok']} (minority refs {len(minor)}), "
             f"err graphs ok={G['Q-2m']['ok']}")
    dip = K[K.kind == "dipole"]
    dok = dip[dip.cfg_error == ""]
    rep["keys"] = dict(
        n_references=n_ref, n_keys=int(K.key.nunique()), n_keys_by_kind=Q._vc(K.groupby("key").kind.first()),
        n_keys_multi_row=int((K.groupby("key").size() >= 2).sum()), n_keys_design=int(K.key_design.nunique()),
        n_keys_cfg_all=int(K.key_cfg_all.nunique()), n_smiles_keys=int(K.key_plain.nunique()),
        n_rows_alt=int(K.alt.astype(bool).sum()), n_rows_with_cfg_in_key=int((K.fixed_state != "").sum()),
        n_dipole_rows_with_cfg=int((dip.cfg_state != "").sum()), n_dipole_rows_without_cfg=int((dip.cfg_state == "").sum()),
        n_dipole_cfg_errors=int((dip.cfg_error != "").sum()),
        n_dipole_cfg_errors_alt=int(((dip.cfg_error != "") & dip.alt.astype(bool)).sum()),
        cfg_with_sp_centre=int(dip.cfg_state.str.contains(r":\?", regex=True).sum()),
        n_smiles_with_alt_and_plain=int(dip.groupby("smiles_key").alt.nunique().gt(1).sum()),
        n_smiles_with_several_cfg=int(dok.groupby(["smiles_key", "alt"]).cfg_state.nunique().gt(1).sum()),
        n_smiles_with_several_s3d=int(K[K.state_error == ""].groupby("key_plain").stereo3d.nunique().gt(1).sum()),
        n_reactions_formula_ambiguous=int(K.formula_ambiguous.astype(bool).sum() // 2),
        manifest=_safe(manifest_check, K))
    for g in ("Q-2s", "Q-2", "Q-2p", "Q-2m"):
        if not G[g]["ok"]:
            raise Stop(f"{g} failed (see gates {g})")
    lv, pe = check_level_d0(K, geoms, by_job)
    G["Q-2l"], G["Q-2e"] = lv, pe
    _log(t0, f"level check ok={lv['ok']} ({lv['n_levels']} levels), energy provenance ok={pe['ok']} max {pe['max_abs_eh']}")
    if not lv["ok"]:
        raise Stop(f"Q-2l: label inputs differ in level within a key or from the reference xyz "
                   f"({lv['n_keys_mixed_level']} keys, {lv['n_inp_xyz_or_charge_mismatch']} rows, {lv['n_missing']} missing)")
    if not pe["ok"]:
        raise Stop(f"Q-2e: {pe['n_fail']} frag_rel.out energies differ from the labels or did not terminate normally, "
                   f"first {pe['fail'][:1]}")

    # Q-3 audit, Q-4 override (+ design-key comparison, ex)
    E = {(r["job_id"], k): float(r[Q.REF_E[k]]) for r in rows for k in (1, 2)}
    A = Q.audit(K, E)
    ov = Q.choose_override(A, K, src="Q:label_min_d0")
    for o in ov.values():                                  # refs_q points ref_inp at the D1-HF copies; D0: label input
        o["ref_inp"] = str(D0_INPUTS / f"rxn_{int(o['src_rxn']):04d}" / f"frag{int(o['src_frag'])}_rel.inp")
    Ad = Q.audit(K, E, key="key_design")
    Ac = Q.audit(K, E, key="key_cfg_all")
    rep["audit"] = dev_counts_d0(A)
    rep["audit"]["nonalt_dipole_replacements_across_cfg"] = cross_cfg(K, A)
    rep["design_key_comparison"] = design_compare(K, Ad, A)
    rep["cfg_all_key_comparison"] = design_compare(
        K, Ac, A, col="key_cfg_all", tag="cfg_all",
        rule="key_cfg_all = the final key with '|cfg' on every dipole reference (the geo1b-d0q-1 key; no cross-rotamer "
             "pooling at all)")
    e_arr = np.array([E[(j, int(f))] for j, f in zip(K.job_id, K.frag)], float)
    ex = (e_arr - pd.Series(e_arr).groupby(K.key_plain.to_numpy()).transform("min").to_numpy()) * KCAL
    exm = {(j, int(f)): float(x) for j, f, x in zip(K.job_id, K.frag, ex)}
    for r in rows:
        r["ex1_kcal"], r["ex2_kcal"] = exm[(r["job_id"], 1)], exm[(r["job_id"], 2)]
    _log(t0, f"audit: {len(ov)} fragment references to replace (design key: "
             f"{rep['design_key_comparison']['n_replaced_design']}, cfg-all key: "
             f"{rep['cfg_all_key_comparison']['n_replaced_cfg_all']})")

    # Q-a, re-assembly, Q-b .. Q-e (on the JSON round trip of what is written)
    qa = Q.gate_qa(rows)
    G["Q-a"] = qa
    _log(t0, f"Q-a ok={qa['ok']} max {qa['max_abs']}")
    if not qa["ok"]:
        raise Stop(f"Q-a: {qa['n_fail']} rows above {Q.TOL}, {qa['n_errors']} rows not assembled (before re-assembly)")
    new_rows, recs = Q.reassemble(rows, ov, K, TAG)
    kd = {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.key_design)}
    unk = {(j, int(f)) for j, f, e in zip(K.job_id, K.frag, K.state_error) if e}       # |err keys (Q-2s, Q-2m)
    for n in new_rows:
        n["cr"].update(code=CODE_VERSION, key_design1=kd[(n["job_id"], 1)], key_design2=kd[(n["job_id"], 2)],
                       unkeyed=[f"frag{k}" for k in (1, 2) if (n["job_id"], k) in unk])
    text, new_rt = Q.json_roundtrip(new_rows)
    QG = Q.gates(rows, new_rt, ov, recs, qa)
    G.update(QG)
    _log(t0, "gates " + " ".join(f"{k}={v.get('ok')}" for k, v in G.items()))
    if not Q.gates_ok(QG):
        write_atomic(OUT_REJECTED, text)
        raise Stop("Q-5 gate failure: " + ", ".join(k for k, v in QG.items() if not v.get("ok")))
    bad = [k for k, v in G.items() if not v.get("ok")]
    if bad:
        raise Stop(f"gate failure: {bad}")
    rep["reassembly"] = reassembly_summary(rows, new_rt, ov, K)
    rep["hygiene"] = hygiene_summary(rows, new_rt)

    # outputs
    rep_d = {(j, int(f)): (s, float(d)) for j, f, s, d in zip(Ad.job_id, Ad.frag, Ad.src_job, Ad.dev_kcal)}
    rep_c = {(j, int(f)): (s, float(d)) for j, f, s, d in zip(Ac.job_id, Ac.frag, Ac.src_job, Ac.dev_kcal)}
    keys_out = A.assign(replaced=[(j, int(f)) in ov for j, f in zip(A.job_id, A.frag)],
                        ex_kcal=[exm[(j, int(f))] for j, f in zip(A.job_id, A.frag)],
                        src_design_job=[rep_d[(j, int(f))][0] for j, f in zip(A.job_id, A.frag)],
                        dev_design_kcal=[rep_d[(j, int(f))][1] for j, f in zip(A.job_id, A.frag)],
                        src_cfg_all_job=[rep_c[(j, int(f))][0] for j, f in zip(A.job_id, A.frag)],
                        dev_cfg_all_kcal=[rep_c[(j, int(f))][1] for j, f in zip(A.job_id, A.frag)])
    T = Q.audit_table(K, A).reset_index()
    write_atomic(OUT_LABELS, text)
    write_atomic(OUT_KEYS, lambda t: keys_out.to_parquet(t, index=False))
    write_atomic(OUT_AUDIT, lambda t: T.to_csv(t, index=False))
    rep["outputs"] = {str(p): L.sha256(p) for p in OUTPUTS}
    rep["n_output_rows"] = len(new_rt)
    rep.update(all_pass=True, stop=False, finished=time.strftime("%F %T"), wall_s=round(time.time() - t0, 1))
    write_json(REPORT, rep)
    for p in OUTPUTS:                                      # the pass supersedes the .stale copies
        s = p.with_name(p.name + ".stale")
        if s.is_file():
            assert_path(s).unlink()
    _log(t0, f"D0 Q pass: {rep['reassembly']['n_rows_changed']} reactions changed "
             f"({rep['reassembly']['n_fragments_changed']} references)")


def cmd_probe(rxn_ids):
    """smoke test, writes nothing: keys, states and Q-a of the given rxn_ids (any labels_all.json rows)."""
    allr = {int(d["rxn_id"]): d for d in json.loads(GC.D0_LABELS.read_text())}
    rows = []
    for rid in rxn_ids:
        d = dict(allr[int(rid)])
        d["job_id"] = job_id(rid)
        rows.append(d)
    K, _, _, errs = molecule_keys_d0(rows, GC.load_d0_smiles())
    cols = ["job_id", "frag", "kind", "rel_file", "alt", "smiles_key", "fixed_state", "cfg_state", "cfg_error",
            "cfg_tags_raw", "stereo3d", "state_error", "key_design", "key", "key_cfg_all", "formula", "wl_graph"]
    qa = Q.gate_qa(rows)
    print(json.dumps(dict(errors=errs, keys=K[cols].to_dict(orient="records") if len(K) else [], qa=qa), indent=1,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)
    return 0 if not errs and qa["ok"] else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description="GEO1b D0D1 step 1: LADDER Q rule on the D0 labels (d0_labels.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--force", action="store_true", help="recompute even if Q_d0_report.json is up to date")
    p = sub.add_parser("probe")
    p.add_argument("rxn_ids", nargs="+", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a.force)
    if a.cmd == "probe":
        return cmd_probe(a.rxn_ids)
    return 2


if __name__ == "__main__":
    sys.exit(main())
