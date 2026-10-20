"""FU2 Phase Q — per-atom xTB values for the D1 feature reactions (docs/specs/D1PAIR_FU2.md, Phase Q).

  python fu2/atoms.py run <i> <n>     # slice i of n of the X182_d1 reactions -> $D1PAIR_SCRATCH/fu2/atoms/<rxn_id>.npz
  python fu2/atoms.py gate            # Q-a / Q-b / extension pass rate -> results/fu2/Q_manifest.csv, Q_report.json

Run with the Phase F environment (rev 5 venv, jobs/g_env.sh with D1PAIR_PHASE=F; d1_g1.check_env exits 3 otherwise):
  sbatch -p cpu1,cpu2 --cpus-per-task=48 --mem=96G --array=0-3 -J d1fu2_Q --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/logs/%x_%A_%a.out \
         --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm fu2/atoms.py run @I 4

Per reaction (exactly the xtb_slice.process_one path, values kept per atom):
  g = xtb_slice.ts_fragments(label, "g1", meta)              geometry, label partition, reacting atoms
  5 single points xtb_slice.xtb_sp: gs1 = rel1 (q1), gs2 = rel2 (q2), d1 = fA (q1), d2 = fB (q2), ts = TS (q1 + q2)
  extension on the TS: ext_features.xtb_run(--sp --json) -> parse_json_sp (atomic dipoles; |mu| = norm),
                       ext_features.xtb_run(--vfukui)     -> parse_fukui (f+, f-); self-checks = the parsers' own
TS-order per-atom arrays: q/v_frag from d1 / d2 via posA / posB, q/v_TS from ts, q/v_rel from gs1 / gs2 via the maps.
Atom maps TS -> reactant (spec Q-1: inv0 / inv1 = inverse of P.map0 / P.map1): these maps cover HEAVY atoms only, so an H
is mapped through its owner (nearest heavy atom, fragmenter.heavy_graph rule; in the TS restricted to its own fragment):
  * values: q_rel / v_rel of a TS H = mean over the H atoms owned by the mapped reactant heavy atom (H on one heavy atom
    are not distinguishable by the maps; the mean does not depend on an arbitrary choice);
  * the full bijection rel -> TS (used for the T-2 covalent edges) pairs the owned H atoms of mapped heavy atoms in index
    order (edge set independent of that order while every H bonds only to its owner).
  A reaction whose owned-H counts differ between TS and reactant gets status 'h_map:<...>' and is excluded (reported).
Idempotent: <rid>.npz (ok) or <rid>.fail.json (status) is written atomically; existing files are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu2_common as F  # noqa: E402

C = F.C
ATOM_VERSION = "fu2-atoms-1"
REACT = ("dip_a", "dip_mid", "dip_b", "dph_a", "dph_b")
MULLIKEN15 = ([f"mulliken_R_dip_{k}" for k in (0, 1, 2)] + [f"mulliken_R_dph_{k}" for k in (0, 1)]
              + [f"mulliken_TSdip_{k}" for k in (0, 1, 2)] + [f"mulliken_TSdph_{k}" for k in (0, 1)]
              + [f"mulliken_TS_{k}" for k in range(5)])            # = train_ml_single.MULLIKEN15
VALENCE15 = [c.replace("mulliken_", "wbo_valence_") for c in MULLIKEN15]
Q30 = MULLIKEN15 + VALENCE15
_LABELS = _META = None


def _env():
    import d1_g1 as DG
    return DG.check_env()                                     # exits 3 unless the Phase F env + Si entries are active


def rxn_ids():
    return sorted(int(r) for r in pd.read_parquet(F.X182, columns=["rxn_id"]).rxn_id)


def labels():
    global _LABELS, _META
    if _LABELS is None:
        import xtb_slice as xs
        _LABELS = {int(d["rxn_id"]): d for d in json.loads(Path(os.environ["ESPLEY_LABELS"]).read_text())}
        _META = xs.load_meta()
    return _LABELS, _META


def owners(syms, xyz, allowed=None):
    """H index -> owner heavy index (nearest heavy atom = fragmenter.heavy_graph); allowed restricts the candidates."""
    import fragmenter as fr
    D = fr.dist_matrix(np.asarray(xyz, dtype=float))
    heavy = [i for i, s in enumerate(syms) if s != "H" and (allowed is None or i in allowed)]
    return {i: min(heavy, key=lambda k: D[i, k]) for i, s in enumerate(syms) if s == "H" and (allowed is None or i in allowed)}


def frag_maps(g):
    """per fragment: TS index -> reactant index (full bijection, H by owner + index order) and TS H -> list of the
    reactant H owned by the mapped heavy atom. Raises ValueError('h_map:...') on an owned-H count mismatch."""
    out = {}
    for name, idx, (rsyms, rxyz), hmap in (("A", g.A_idx, g.rel1, g.P.map0), ("B", g.B_idx, g.rel2, g.P.map1)):
        inv = {int(t): int(r) for r, t in hmap.items()}                 # TS heavy -> reactant heavy
        heavy_ts = [i for i in idx if g.ts_syms[i] != "H"]
        if sorted(inv) != sorted(heavy_ts):
            raise ValueError(f"h_map:{name}_heavy_map_incomplete")
        own_ts = owners(g.ts_syms, g.ts_xyz, allowed=set(idx))
        own_r = owners(rsyms, rxyz)
        by_ts, by_r = {}, {}
        for h, o in own_ts.items():
            by_ts.setdefault(o, []).append(h)
        for h, o in own_r.items():
            by_r.setdefault(o, []).append(h)
        full, hgroup = dict(inv), {}
        for t_heavy, r_heavy in inv.items():
            hs_t, hs_r = sorted(by_ts.get(t_heavy, [])), sorted(by_r.get(r_heavy, []))
            if len(hs_t) != len(hs_r):
                raise ValueError(f"h_map:{name}_owned_H_count_{len(hs_t)}_vs_{len(hs_r)}")
            for ht, hr in zip(hs_t, hs_r):
                full[ht] = hr
                hgroup[ht] = hs_r
        if sorted(full) != sorted(idx) or sorted(full.values()) != list(range(len(rsyms))):
            raise ValueError(f"h_map:{name}_not_a_bijection")
        out[name] = (full, hgroup, own_r)
    return out


def compute(rid):
    """one reaction -> dict of arrays (TS order unless named otherwise) + status fields; raises on failure."""
    import ext_features as ef
    import fragmenter as fr
    import xtb_slice as xs
    lab, meta = labels()
    label = lab[rid]
    q1, q2 = int(label.get("charge1", 0) or 0), int(label.get("charge2", 0) or 0)
    g = xs.ts_fragments(label, "g1", meta)
    if isinstance(g, str):
        raise ValueError(f"ts_fragments:{g}")
    maps = frag_maps(g)
    T = {"gs1": xs.xtb_sp(*g.rel1, q1), "gs2": xs.xtb_sp(*g.rel2, q2),
         "d1": xs.xtb_sp(*g.fA, q1), "d2": xs.xtb_sp(*g.fB, q2), "ts": xs.xtb_sp(g.ts_syms, g.ts_xyz, q1 + q2)}
    n = len(g.ts_syms)
    frag = np.array([0 if i in g.P.A else 1 for i in range(n)], dtype=np.int8)
    rel_idx = np.full(n, -1, dtype=np.int64)                 # bijection TS -> own reactant index
    q_rel, v_rel = np.full(n, np.nan), np.full(n, np.nan)
    q_frag, v_frag = np.full(n, np.nan), np.full(n, np.nan)
    for name, sp_r, sp_f, pos in (("A", "gs1", "d1", g.posA), ("B", "gs2", "d2", g.posB)):
        full, hgroup, _ = maps[name]
        for i, r in full.items():
            rel_idx[i] = r
            if i in hgroup:                                   # H: mean over the owned H of the mapped heavy atom
                q_rel[i] = float(np.mean(T[sp_r]["charges"][hgroup[i]]))
                v_rel[i] = float(np.mean(T[sp_r]["valence"][hgroup[i]]))
            else:
                q_rel[i], v_rel[i] = T[sp_r]["charges"][r], T[sp_r]["valence"][r]
            q_frag[i], v_frag[i] = T[sp_f]["charges"][pos[i]], T[sp_f]["valence"][pos[i]]
    react = np.array([g.dip_a, g.dip_mid, g.dip_b, g.dph_a, g.dph_b], dtype=np.int64)
    # covalent bonds of the two reactants (bond_matrix, FACTOR 1.25) in TS numbering
    bonds = []
    for name, (rs, rx) in (("A", g.rel1), ("B", g.rel2)):
        inv_full = {r: t for t, r in maps[name][0].items()}
        adj, _ = fr.bond_matrix(rs, rx)
        own_r = maps[name][2]
        for h, o in own_r.items():                            # the within-owner H order is then irrelevant for edges
            nb = np.flatnonzero(adj[h])
            if len(nb) != 1 or int(nb[0]) != o:
                raise ValueError(f"h_map:{name}_H{h}_bonds_{nb.tolist()}_owner_{o}")
        for u, v in zip(*np.nonzero(np.triu(adj, 1))):
            a, b = inv_full[int(u)], inv_full[int(v)]
            bonds.append((min(a, b), max(a, b)))
    formed = np.array([tuple(sorted(p)) for p in g.pairs], dtype=np.int64)
    # extension: TS --sp --json (atomic dipoles) and --vfukui (f+, f-), with the parsers' self-checks
    ext = dict(ext_ok=False, ext_reason="", mu=np.full((n, 3), np.nan), fk=np.full((n, 3), np.nan),
               q_json_maxdiff=np.nan, dip_check=np.nan)
    for attempt in (1, 2):                                   # an xtb timeout / signal-killed run is retried once
        try:
            out, files = ef.xtb_run(g.ts_syms, g.ts_xyz, q1 + q2, ["--sp", "--json"], keep=("xtbout.json", "charges"))
            js = ef.parse_json_sp(out, files, g.ts_syms, g.ts_xyz, q1 + q2, "ts:json")
            fk = ef.parse_fukui(ef.xtb_run(g.ts_syms, g.ts_xyz, q1 + q2, ["--vfukui"])[0], g.ts_syms, "ts:vfukui")
            qd = float(np.abs(js["q"] - T["ts"]["charges"]).max())  # json-run charges vs the ts single point (same calc)
            if not qd <= ef.TOL_Q_JSON:
                raise ef.ParseError(f"ts: --sp --json charges differ from xtb_sp ts charges by {qd:.1e} e")
            if not (np.all(np.isfinite(js["mu"])) and np.all(np.isfinite(fk))):
                raise ef.BlockFail("nonfinite_ext")
            ext.update(ext_ok=True, ext_reason="", mu=np.asarray(js["mu"], dtype=float), fk=np.asarray(fk, dtype=float),
                       q_json_maxdiff=float(js["q_json_maxdiff"]), dip_check=float(js["dip_check"]))
            break
        except OSError:
            raise
        except Exception as e:                               # noqa: BLE001 — only the extension values are lost
            ext["ext_reason"] = (str(e) if isinstance(e, ef.BlockFail) else f"error:{type(e).__name__}:{e}")[:200]
            if not (ext["ext_reason"].startswith("xtb_timeout") or ":exit-" in ext["ext_reason"]):
                break
    if ext["ext_reason"].startswith("xtb_timeout") or ":exit-" in ext["ext_reason"]:
        raise RetryableError(f"ext_infra:{ext['ext_reason']}")
    return dict(
        rxn_id=rid, n=n, q1=q1, q2=q2, ts_Z=np.array([fr.Z[s] for s in g.ts_syms], dtype=np.int64),
        ts_pos=np.asarray(g.ts_xyz, dtype=float), frag=frag, A_idx=np.array(g.A_idx), B_idx=np.array(g.B_idx),
        react=react, r1_dip=np.array(g.r1_dip), r2_dph=np.array(g.r2_dph), formed=formed,
        bonds=np.array(sorted(set(bonds)), dtype=np.int64).reshape(-1, 2), n_bonds_rel=np.array([len(bonds)]),
        rel1_Z=np.array([fr.Z[s] for s in g.rel1[0]]), rel1_pos=np.asarray(g.rel1[1], dtype=float),
        rel2_Z=np.array([fr.Z[s] for s in g.rel2[0]]), rel2_pos=np.asarray(g.rel2[1], dtype=float), rel_idx=rel_idx,
        **{f"q_{k}": np.asarray(T[k]["charges"], dtype=float) for k in T},
        **{f"v_{k}": np.asarray(T[k]["valence"], dtype=float) for k in T},
        **{f"e_{k}": np.array([T[k]["total"]]) for k in T},
        q_rel=q_rel, v_rel=v_rel, q_frag=q_frag, v_frag=v_frag, q_TS=np.asarray(T["ts"]["charges"], dtype=float),
        v_TS=np.asarray(T["ts"]["valence"], dtype=float), mu_vec=ext["mu"], mu_abs=np.linalg.norm(ext["mu"], axis=1),
        fk=ext["fk"], ext_ok=np.array([ext["ext_ok"]]), ext_reason=np.array([ext["ext_reason"]]),
        q_json_maxdiff=np.array([ext["q_json_maxdiff"]]), dip_check=np.array([ext["dip_check"]]),
        atom_version=np.array([ATOM_VERSION]))


def save_npz(path, arrs):
    F.assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class RetryableError(Exception):
    """infrastructure-type failure (xtb timeout / killed): recorded in <rid>.err.json, never cached, retried on rerun."""


def is_cached(rid, root):
    """True when <rid>.npz or <rid>.fail.json exists AND was written by this ATOM_VERSION (else it is recomputed)."""
    ok, fail = root / f"{rid}.npz", root / f"{rid}.fail.json"
    if ok.is_file():
        with np.load(ok) as z:
            return str(z["atom_version"][0]) == ATOM_VERSION
    if fail.is_file():
        return json.loads(fail.read_text()).get("atom_version") == ATOM_VERSION
    return False


def one(rid, root=None):
    """compute one reaction. Cached: <rid>.npz (ok) or <rid>.fail.json for the expected per-reaction statuses only
    (ValueError 'ts_fragments:...' / 'h_map:...'). Any other exception -> <rid>.err.json (not cached, retried)."""
    root = Path(root) if root else F.ATOMS
    ok, fail, err = root / f"{rid}.npz", root / f"{rid}.fail.json", root / f"{rid}.err.json"
    if is_cached(rid, root):
        return rid, "cached", 0.0
    t0 = time.time()
    try:
        arrs = compute(rid)
    except OSError:
        raise                                                    # missing binary / disk problem: not a result, not cached
    except Exception as e:                                       # noqa: BLE001 — per-rxn status, recorded
        st = f"{type(e).__name__}:{str(e)[:160]}"
        expected = isinstance(e, ValueError) and str(e).startswith(("ts_fragments:", "h_map:"))
        F.write_json(fail if expected else err, dict(rxn_id=rid, status=st, traceback=traceback.format_exc()[-2000:],
                                                     atom_version=ATOM_VERSION, host=os.uname().nodename))
        return rid, st if expected else "err:" + st, time.time() - t0
    save_npz(ok, arrs)
    for p in (fail, err):
        if p.is_file():
            p.unlink()
    return rid, "ok" + ("" if bool(arrs["ext_ok"][0]) else " (ext: " + str(arrs["ext_reason"][0])[:60] + ")"), time.time() - t0


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    labels()


def cmd_run(i, n):
    _env()
    ids = rxn_ids()[int(i)::int(n)]
    todo = [r for r in ids if not is_cached(r, F.ATOMS)]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[Q {i}/{n}] {len(ids)} rxns, {len(todo)} to compute, {workers} workers", flush=True)
    if not todo:
        return 0
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cnt = Counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                rid, st, dt = fu.result()
            except Exception as e:                               # OSError (disk / binary) or a dead worker: stop now
                print(f"STOP: worker error {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                return 3
            cnt[st.split(":")[0].split(" ")[0]] += 1
            if not st.startswith("ok") or k % 200 == 0:
                print(f"  {rid}: {st} ({dt:.0f} s)  [{k}/{len(todo)}] {dict(cnt)}", flush=True)
    print(f"[Q {i}/{n}] done {dict(cnt)}", flush=True)
    return 0


def cmd_smoke(k=24):
    """k reactions spread over the list -> $FU2_SCR/atoms_smoke (never read by Q / T), Q-a / Q-b on them; exit 3 when a
    computed reaction fails Q-a or Q-b, or more than 2 of the k fail to compute (the full Q run waits on this, afterok)."""
    _env()
    ids = rxn_ids()
    pick = [ids[j] for j in np.linspace(0, len(ids) - 1, int(k)).round().astype(int)]
    root = F.FU2_SCR / "atoms_smoke"
    import shutil
    if root.exists():
        shutil.rmtree(root)                                   # always recompute with the current code
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        res = list(ex.map(one, pick, [root] * len(pick)))
    X = pd.read_parquet(F.X182, columns=["rxn_id"] + Q30).set_index("rxn_id")
    n_fail, bad = 0, []
    for rid, st, dt in res:
        p = root / f"{rid}.npz"
        if not p.is_file():
            n_fail += 1
            print(f"  {rid}: FAIL {json.loads((root / f'{rid}.fail.json').read_text())['status']}")
            continue
        a = dict(np.load(p))
        q30, q30t = recompute_q30(a), recompute_q30_ts(a)
        qa = max(max(abs(q30[c] - float(X.at[rid, c])), abs(q30t[c] - float(X.at[rid, c]))) for c in Q30)
        qb = max(abs(float(a[f"q_{s}"].sum()) - q) for s, q in (("gs1", a["q1"]), ("gs2", a["q2"]), ("d1", a["q1"]),
                                                                 ("d2", a["q2"]), ("ts", a["q1"] + a["q2"])))
        hsum = abs(float(a["q_rel"][a["frag"] == 0].sum()) - float(a["q_gs1"].sum()))   # H mean map keeps the A total
        print(f"  {rid}: {st} ({dt:.0f} s) n={int(a['n'])} Q-a {qa:.1e} Q-b {qb:.1e} ext {bool(a['ext_ok'][0])} "
              f"|mu| {np.nanmax(a['mu_abs']):.3f} f+ {np.nanmax(a['fk'][:, 0]):.3f} bonds {len(a['bonds'])} "
              f"sum q_rel(A) - sum q_gs1 {hsum:.1e}")
        if qa > F.QA_TOL or qb > F.QB_TOL:
            bad.append(rid)
    print(f"[Q smoke] {len(pick)} rxns, {n_fail} compute failures, Q-a / Q-b failures {bad}")
    return 3 if (bad or n_fail > 2) else 0


# ---------------------------------------------------------------- Q-2 gates
def recompute_q30(a):
    """the 30 process_one columns from the stored per-atom values (same indices as xtb_slice.process_one)."""
    A, B = list(a["A_idx"]), list(a["B_idx"])
    posA, posB = {t: k for k, t in enumerate(A)}, {t: k for k, t in enumerate(B)}
    ts_dip, ts_dph = list(a["react"][:3]), list(a["react"][3:])
    row = {}
    for k, i in enumerate(a["r1_dip"]):
        row[f"mulliken_R_dip_{k}"], row[f"wbo_valence_R_dip_{k}"] = a["q_gs1"][i], a["v_gs1"][i]
    for k, i in enumerate(a["r2_dph"]):
        row[f"mulliken_R_dph_{k}"], row[f"wbo_valence_R_dph_{k}"] = a["q_gs2"][i], a["v_gs2"][i]
    for k, i in enumerate(ts_dip):
        row[f"mulliken_TSdip_{k}"], row[f"wbo_valence_TSdip_{k}"] = a["q_d1"][posA[i]], a["v_d1"][posA[i]]
    for k, i in enumerate(ts_dph):
        row[f"mulliken_TSdph_{k}"], row[f"wbo_valence_TSdph_{k}"] = a["q_d2"][posB[i]], a["v_d2"][posB[i]]
    for k, i in enumerate(ts_dip + ts_dph):
        row[f"mulliken_TS_{k}"], row[f"wbo_valence_TS_{k}"] = a["q_ts"][i], a["v_ts"][i]
    return {c: float(row[c]) for c in Q30}


def recompute_q30_ts(a):
    """the same 30 columns from the TS-order arrays the tokens use (q/v_rel, q/v_frag, q/v_TS; reacting atoms are heavy,
    so q/v_rel there is the mapped reactant atom's own value)."""
    ts_dip, ts_dph = list(a["react"][:3]), list(a["react"][3:])
    row = {}
    for pre, idx, src in (("R_dip", ts_dip, "rel"), ("R_dph", ts_dph, "rel"), ("TSdip", ts_dip, "frag"),
                          ("TSdph", ts_dph, "frag"), ("TS", ts_dip + ts_dph, "TS")):
        for k, i in enumerate(idx):
            row[f"mulliken_{pre}_{k}"], row[f"wbo_valence_{pre}_{k}"] = a[f"q_{src}"][i], a[f"v_{src}"][i]
    return {c: float(row[c]) for c in Q30}


def cmd_gate():
    ids = rxn_ids()
    X = pd.read_parquet(F.X182, columns=["rxn_id"] + Q30).set_index("rxn_id")
    rows, n_missing = [], 0
    for rid in ids:
        ok, fail, err = F.ATOMS / f"{rid}.npz", F.ATOMS / f"{rid}.fail.json", F.ATOMS / f"{rid}.err.json"
        if (ok.is_file() or fail.is_file()) and not is_cached(rid, F.ATOMS):
            n_missing += 1                                      # written by another ATOM_VERSION: recompute first
            rows.append(dict(rxn_id=rid, status="stale_version"))
            continue
        if ok.is_file():
            a = dict(np.load(ok))
            r = dict(rxn_id=rid, n_atoms=int(a["n"]), status="ok", sha256=F.sha256(ok), ext_ok=bool(a["ext_ok"][0]),
                     ext_reason=str(a["ext_reason"][0]))
            q30, q30t = recompute_q30(a), recompute_q30_ts(a)
            r["qa_maxdiff"] = float(max(abs(q30[c] - float(X.at[rid, c])) for c in Q30))
            r["qa_ts_maxdiff"] = float(max(abs(q30t[c] - float(X.at[rid, c])) for c in Q30))
            r["qa_ok"] = bool(r["qa_maxdiff"] <= F.QA_TOL and r["qa_ts_maxdiff"] <= F.QA_TOL)
            sums = dict(gs1=(a["q_gs1"].sum(), a["q1"]), gs2=(a["q_gs2"].sum(), a["q2"]), d1=(a["q_d1"].sum(), a["q1"]),
                        d2=(a["q_d2"].sum(), a["q2"]), ts=(a["q_ts"].sum(), a["q1"] + a["q2"]))
            r["qb_maxdiff"] = float(max(abs(float(s) - float(q)) for s, q in sums.values()))
            fin = all(np.all(np.isfinite(a[k])) for k in ("q_gs1", "q_gs2", "q_d1", "q_d2", "q_ts", "v_gs1", "v_gs2",
                                                           "v_d1", "v_d2", "v_ts", "q_rel", "v_rel", "q_frag", "v_frag"))
            fin_ext = (not r["ext_ok"]) or (np.all(np.isfinite(a["mu_abs"])) and np.all(np.isfinite(a["fk"])))
            r["qb_ok"] = bool(r["qb_maxdiff"] <= F.QB_TOL and fin and fin_ext)
            r["n_bonds"] = int(len(a["bonds"]))
        elif fail.is_file() or err.is_file():                  # err = retryable, still failing after the reruns
            st = json.loads((fail if fail.is_file() else err).read_text())["status"]
            r = dict(rxn_id=rid, n_atoms=np.nan, status=st if fail.is_file() else "err:" + st, sha256="", ext_ok=False,
                     ext_reason="", qa_maxdiff=np.nan, qa_ok=False, qb_maxdiff=np.nan, qb_ok=False, n_bonds=np.nan)
        else:
            n_missing += 1
            r = dict(rxn_id=rid, n_atoms=np.nan, status="missing", sha256="", ext_ok=False, ext_reason="",
                     qa_maxdiff=np.nan, qa_ok=False, qb_maxdiff=np.nan, qb_ok=False, n_bonds=np.nan)
        rows.append(r)
    M = pd.DataFrame(rows)
    if n_missing:
        print(f"GATE: {n_missing} reactions have neither an npz nor a fail record (Q run incomplete)", file=sys.stderr)
        return 2
    okr = M[M.status == "ok"]
    n_qa_fail = int((~M.qa_ok).sum())                         # incl. failed computations (no values -> not reproduced)
    n_qa_repro = int((~okr.qa_ok).sum())                      # computed but not reproduced
    qb_bad = okr[~okr.qb_ok]
    good = okr[okr.qa_ok & okr.qb_ok]
    ext_rate = float(good.ext_ok.mean()) if len(good) else 0.0  # pass rate over the reproduced reactions
    ext_used = bool(ext_rate >= F.EXT_MIN_PASS)
    # Q-1: with the extension used, only reactions that passed its self-checks are used; below 99 % it is dropped
    M["use"] = M.qa_ok & M.qb_ok & (M.ext_ok | (not ext_used))
    F.write_atomic(F.FU2_RES / "Q_manifest.csv", lambda t: M.to_csv(t, index=False))
    rep = dict(n_target=len(ids), n_ok=int(len(okr)), n_compute_fail=int((M.status != "ok").sum()),
               n_qa_not_reproduced=n_qa_repro, n_excluded_ext_fail=int((M.qa_ok & M.qb_ok & ~M.ext_ok).sum()) if ext_used else 0,
               qa_ts_maxdiff_max=float(okr.qa_ts_maxdiff.max()) if len(okr) else None,
               fail_reasons=Counter(s.split(":")[0] + ":" + s.split(":")[1] if ":" in s else s
                                    for s in M.status[M.status != "ok"]).most_common(),
               qa_tol=F.QA_TOL, qa_maxdiff_max=float(okr.qa_maxdiff.max()) if len(okr) else None,
               n_qa_fail=n_qa_fail, qa_max_fail=F.QA_MAX_FAIL, qb_tol=F.QB_TOL, n_qb_fail=int(len(qb_bad)),
               qb_maxdiff_max=float(okr.qb_maxdiff.max()) if len(okr) else None,
               ext_pass_rate=ext_rate, ext_min_pass=F.EXT_MIN_PASS, ext_used=ext_used,
               ext_fail_reasons=Counter(okr.ext_reason[~okr.ext_ok].str.slice(0, 60)).most_common(10),
               n_use=int(M.use.sum()), excluded=M.loc[~M.use, "rxn_id"].astype(int).tolist(),
               n_atoms=dict(min=float(okr.n_atoms.min()), median=float(okr.n_atoms.median()), max=float(okr.n_atoms.max())),
               manifest_sha256=F.sha256(F.FU2_RES / "Q_manifest.csv"), atom_version=ATOM_VERSION,
               h_map_rule="H mapped through its owner heavy atom; q/v_rel of an H = mean over the owned H of the mapped "
                          "reactant heavy atom; bijection for edges = owned H in index order (spec gap: maps are heavy-only)")
    rep["stop"] = bool(n_qa_fail > F.QA_MAX_FAIL or len(qb_bad) > 0)
    rep["stop_reason"] = ([f"Q-a: {n_qa_fail} > {F.QA_MAX_FAIL} reactions not reproduced"] if n_qa_fail > F.QA_MAX_FAIL else []) \
        + ([f"Q-b: {len(qb_bad)} reactions with fragment charge sum off or non-finite values"] if len(qb_bad) else [])
    F.write_json(F.FU2_RES / "Q_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "excluded"}, indent=1, default=str))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        sys.exit(cmd_run(a[1], a[2]))
    if a[:1] == ["gate"]:
        sys.exit(cmd_gate())
    if a[:1] == ["smoke"]:
        sys.exit(cmd_smoke(*(a[1:2] or [24])))
    print(__doc__); sys.exit(2)
