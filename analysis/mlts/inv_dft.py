#!/usr/bin/env python3
"""inv_dft.py — DEPLOY-V2 (MLTS) Phase I: inventory of the label-level DFT force data that already exists
(docs/specs/DEPLOY_V2_MLTS.md section 2). CPU only, reads only; writes $MLT/dft/existing.extxyz and results/I_*.

  python inv_dft.py scan [--cores N] [--limit-per-source K]   parse every candidate ORCA output (multiprocessing
                                                              over files) -> $MLT/dft/existing.extxyz,
                                                              results/I_inventory.json, results/I_sources.csv
  python inv_dft.py probe <file.out[.gz]> ... [--roundtrip]   parse a few files and print the checks (no outputs;
                                                              --roundtrip writes + deletes one $MLT/tmp file)
  python inv_dft.py verify                                    read the extxyz back (mlt_common.read_extxyz): tags vs
                                                              targets, units, unique ids; exit 3 on a problem
  python inv_dft.py report                                    print results/I_inventory.json (summary part)

Sources (the spec's "where to look", verified 2026-11-04):
  g3l       $G3/O/units/L.<rxn>/optts/ts.out.gz       G3L: G2 TS (label order, g2_final) -> label-level OptTS, 20 rxns
  partb     $G3/O/units/B.<rxn>/{optts,optts_reopt}/ts.out.gz   Part B label-level OptTS, 40 rxns (+ 3 re-opts)
  g3_other  every other .out under $G3/O/units (g3h = HF/6-31G*, g3r = r2SCAN-3c, T5 / sp = def2-TZVP SPs): level
            check only -> excluded (recorded)
  v_conf    $LAD/V/mols/V####/c*/*.out.gz   label-level reactant conformer Opt (opt1, opt2) + Freq / TZVP SPs (no
            gradient -> excluded); molecule -> reactions via spec.json 'key' == labels cr.key1 / cr.key2
  d1val     $D1VAL/jobs/*/ts/*/{transition_states,reactants_and_products}/*_orca.out, jobs/*/ts_reopt/*.out,
            jobs/*/ref/*_orca.out, v2a/*/*.out (EnGrad at D0 TS), v7/*/*.out (relaxed scan): autodE trees of D0 Coley
            (and a few D1-like validation) reactions, with *_trj.xyz
  not scanned (recorded): D1's own autodE trees are gone; $D1RUN and the D1 ledger tars are listed for ORCA outputs
            (expected none); $LAD/dft_final/*/work/ts.out.gz is XTB2 ALPB Freq (sampled inputs recorded); EDA / TZVP
            single points (jobs/*/sp, single_points, v1) have no gradient; thermal/*.out are Freq (no gradient).

Per file: sha256, input echo -> orca_io.label_level + same-grid check (RIJCOSX + def2/J, no grid keyword),
ORCA version, terminated normally, converged, parsed (coords, E, gradient) steps vs 'GEOMETRY OPTIMIZATION CYCLE'
count (the spec's _trj.xyz step check; where *_trj.xyz exists its frames are compared too: frame i = step i,
positions within 2e-6 A and energy within 1e-9 Eh, one extra frame allowed = the final energy evaluation of a
converged run). Excluded: parse error, not label level, not terminated normally, no gradient steps, step-count or
trj mismatch.

Kinds: ts_path (OptTS cycles; the last cycle of a converged OptTS is ts_final instead, so no structure appears
twice), ts_final, react_opt (V conformer Opt cycles), frag (none exist), and for d1val: d0_ts_path / d0_ts_final
(OptTS, ts_reopt), d0_ts_guess (constrained Opt to the TS guess), d0_irc (autodE forwards/backwards LooseOpt from
the TS), d0_react_opt (reactants_and_products r*, ref/), d0_prod_opt (products p*), d0_ts_grad (v2a EnGrad), d0_scan.
Leakage tags (mlt_common.EXT_KEYS): D1 reactions rxn_ids=<rxn_id>, folds=<fold_T1>, dipoles=<dipole>; V molecules =
the union over every D1 target whose cr.key1 / cr.key2 is the molecule key; d1val = '' unless the reaction (all
reactant SMILES + product, refs_q.mirror_min) is a D1 target (then that target's tags) or, for reactant
optimisations, the molecule SMILES is a D1 molecule key (then the union over its D1 reactions, as for V).
I-1: D1 TS-path structures (ts_path + ts_final with label-order atoms) whose forming-bond difference to the label TS
(max over the two formed pairs of |d - d_label|, formed_pairs_ts, label order) is <= 0.3 A; pass iff >= 50,000
structures over >= 5,000 reactions (otherwise G-2 is needed)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlt_common as C  # noqa: E402
import orca_io as O  # noqa: E402

CODE_VERSION = "mlts-inv-1"
OUT_EXT = C.MLT / "dft" / "existing.extxyz"
RES_INV = C.RES / "I_inventory.json"
RES_SRC = C.RES / "I_sources.csv"
I1_THR_A = 0.3
I1_MIN_STRUCT, I1_MIN_RXN = 50000, 5000
FMAX_TS_OK = 5e-4                  # Eh/bohr, max |gradient component| at a converged TS (sanity, recorded)
TRJ_POS_TOL, TRJ_E_TOL = 2e-6, 1e-9
D1_LEDGER = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_ledger/summary")
D1_BUNDLE = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run_bundle.tar.gz")
OPT_TOKENS = {"OPT", "LOOSEOPT", "TIGHTOPT", "VERYTIGHTOPT", "NORMALOPT", "COPT", "ZOPT", "GDIIS-COPT", "GDIIS-ZOPT"}
GRID_RE = re.compile(r"^(DEFGRID\d|GRID\d|GRIDX\d|FINALGRID\d|NOFINALGRID|NOFINALGRIDX|COSX_GRID|SPECIALGRID)$")
TS_KINDS = ("ts_path", "ts_final")
ALL_KINDS = ("ts_path", "ts_final", "react_opt", "frag", "d0_ts_path", "d0_ts_final", "d0_ts_guess", "d0_irc",
             "d0_react_opt", "d0_prod_opt", "d0_ts_grad", "d0_scan")


def n_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _stem(name):
    for suf in (".out.gz", ".out"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


# ---------------------------------------------------------------- candidate discovery
def cands_g3():
    root = C.G3 / "O" / "units"
    out = []
    for u in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        parts = u.name.split(".")
        typ = parts[0]
        if typ in ("L", "B"):
            src, rxn = ("g3l" if typ == "L" else "partb"), int(parts[1])
        else:
            src = "g3_other"
            rxn = int(parts[-1]) if parts[-1].isdigit() else (C.rxn_of(parts[1]) if typ == "T5" else None)
        final_dir = None
        uj = u / "unit.json"
        if uj.exists():
            try:
                final_dir = C.read_json(uj).get("final_dir")
            except Exception:                                    # noqa: BLE001
                final_dir = None
        for f in sorted(u.rglob("*")):
            if f.is_file() and (f.name.endswith(".out") or f.name.endswith(".out.gz")):
                rel = f.relative_to(u)
                out.append(dict(source=src, path=str(f), group=str(rxn) if rxn else u.name, owner=u.name, rxn_id=rxn,
                                stem=_stem(str(rel)), sub=rel.parts[0], unit_final=(rel.parts[0] == final_dir)))
    return out


def cands_v():
    root = C.LAD / "V" / "mols"
    out = []
    for m in sorted(p for p in root.iterdir() if p.is_dir()):
        key = C.read_json(m / "spec.json")["key"]
        for f in sorted(m.glob("c*/*.out.gz")):
            out.append(dict(source="v_conf", path=str(f), group=m.name, owner=m.name, mol_key=key,
                            stem=_stem(f"{f.parent.name}/{f.name}"), sub=f.parent.name))
    return out


def cands_d1val():
    out = []
    jobs = C.D1VAL / "jobs"
    for j in sorted(p for p in jobs.iterdir() if p.is_dir()):
        pats = [("transition_states", "ts/*/transition_states/*_orca.out"),
                ("reactants_and_products", "ts/*/reactants_and_products/*_orca.out"),
                ("ts_reopt", "ts_reopt/*.out"), ("ref", "ref/*_orca.out")]
        for sub, pat in pats:
            for f in sorted(j.glob(pat)):
                out.append(dict(source="d1val", path=str(f), owner=j.name, val_job=j.name, sub=sub,
                                stem=_stem(f"{sub}/{f.name}")))
    for sub in ("v2a", "v7"):
        for f in sorted((C.D1VAL / sub).glob("*/*.out")):
            vj = f"{sub}_{f.parent.name}"
            out.append(dict(source="d1val", path=str(f), owner=vj, val_job=vj, sub=sub, stem=_stem(f.name)))
    return out


def not_scanned():
    """sources looked at but not parsed, with file counts (recorded in I_inventory.json)."""
    rec = {}
    V = C.D1VAL
    rec["d1val_eda_sp"] = dict(n=len(list(V.glob("jobs/*/sp/*.out"))) + len(list(V.glob("v1/*/run/*.out"))),
                               why="EDA / fragment single points at def2-TZVP (sp_stage), no gradient")
    rec["d1val_single_points"] = dict(n=len(list(V.glob("jobs/*/ts/*/single_points/*.out"))),
                                      why="autodE sp keywords (def2-TZVP), no gradient")
    rec["d1val_thermal"] = dict(n=len(list(V.glob("jobs/*/ts/*/thermal/*.out"))),
                                why="label-level Freq (analytic Hessian), no CARTESIAN GRADIENT block")
    rec["d1val_ts_hess"] = dict(n=len(list(V.glob("jobs/*/ts/*/transition_states/*_hess_orca.out"))),
                                why="label-level Freq on TS guesses: parsed, excluded as no_gradient_steps")
    rec["v_repro"] = dict(n=len(list((C.LAD / "V" / "mols").glob("*/repro/*.out.gz"))),
                          why="V reproduction SPs (label def2-TZVP SP, HF/6-31G*), no gradient")
    rec["v_no_conformers"] = dict(n=sum(1 for m in (C.LAD / "V" / "mols").iterdir() if m.is_dir() and not (m / "c0").exists()),
                                  why="V molecules with no new conformer (n_chosen 0): no DFT optimisation was run")
    # LADDER dft_final work: XTB2 ALPB(water) Freq at the label TS (sample of the input headers)
    ids = sorted(p.name for p in C.DFT_FINAL.iterdir())[::max(1, len(list(C.DFT_FINAL.iterdir())) // 20)][:20]
    heads = Counter()
    for r in ids:
        p = C.DFT_FINAL / r / "work" / "ts.inp"
        if p.exists():
            heads[p.read_text().splitlines()[0].strip()] += 1
    rec["lad_dft_final_work"] = dict(n=len(list(C.DFT_FINAL.iterdir())), sample=len(ids), sample_headers=dict(heads),
                                     why="ORCA XTB2 ALPB(water) Freq at the label TS (G-1 normal modes), not DFT")
    return rec


def d1_archives():
    """D1's own autodE trees are gone; list $D1RUN and the ledger / bundle tars for ORCA outputs (expected none)."""
    pat = re.compile(r"(\.out(\.gz)?|\.engrad|_trj\.xyz|\.hess(\.gz)?|\.opt)$")
    hits_run, n_run = [], 0
    for dp, _, fns in os.walk(C.D1RUN):
        for fn in fns:
            n_run += 1
            if pat.search(fn):
                hits_run.append(os.path.join(dp, fn))
    tars = []
    for t in sorted(D1_LEDGER.glob("*.tar")) + ([D1_BUNDLE] if D1_BUNDLE.exists() else []):
        try:
            with tarfile.open(t) as tf:
                names = tf.getnames()
            hit = [n for n in names if pat.search(n)]
            tars.append(dict(tar=str(t), n_members=len(names), n_orca_like=len(hit), examples=hit[:5]))
        except Exception as e:                                   # noqa: BLE001
            tars.append(dict(tar=str(t), error=f"{type(e).__name__}: {e}"))
    return dict(d1run=dict(root=str(C.D1RUN), n_files=n_run, n_orca_like=len(hits_run), examples=hits_run[:10]),
                tars=tars, n_tars=len(tars), n_orca_like_in_tars=sum(t.get("n_orca_like", 0) for t in tars))


# ---------------------------------------------------------------- per-file parsing (worker)
def bang_line(echo):
    return " ".join(" ".join(ln.split("#")[0].split()) for ln in echo.splitlines() if ln.strip().startswith("!"))


def bang_tokens(echo):
    return set(bang_line(echo).upper().replace("!", " ").split())


def same_grid(echo):
    """(ok, reason): D1 autodE inputs use RIJCOSX def2/J and ORCA 6.1.1 default grids (no grid keyword)."""
    toks = bang_tokens(echo)
    miss = {"RIJCOSX", "DEF2/J"} - toks
    if miss:
        return False, f"missing {sorted(miss)}"
    g = sorted(t for t in toks if GRID_RE.match(t))
    if g:
        return False, f"grid keyword {g}"
    if re.search(r"(?is)%method\b.*?\bgrid", echo):
        return False, "%method grid setting"
    return True, "default grid"


def job_type(echo):
    toks = bang_tokens(echo)
    if "OPTTS" in toks:
        return "optts"
    if toks & OPT_TOKENS:
        if re.search(r"(?i)%geom\s+scan", echo):
            return "scan"
        if re.search(r"(?i)%geom\s+constraints", echo):
            return "const_opt"
        return "opt"
    if "ENGRAD" in toks:
        return "engrad"
    if toks & {"FREQ", "NUMFREQ", "ANFREQ"}:
        return "freq"
    return "sp"


def text_flags(path):
    ver, conv, notconv = None, False, False
    with O._open(path) as f:
        for ln in f:
            if ver is None and "Program Version" in ln:
                m = re.search(r"Program Version\s+(\S+)", ln)
                ver = m.group(1) if m else ln.strip()
            elif "THE OPTIMIZATION HAS CONVERGED" in ln:
                conv = True
            elif "did not converge" in ln.lower() and "optimization" in ln.lower():
                notconv = True
    return ver, conv, notconv


def head_version(path, n=300):
    with O._open(path) as f:
        for k, ln in enumerate(f):
            if "Program Version" in ln:
                m = re.search(r"Program Version\s+(\S+)", ln)
                return m.group(1) if m else ln.strip()
            if k > n:
                break
    return None


def read_trj(path):
    lines = Path(path).read_text().splitlines()
    frames, k = [], 0
    while k < len(lines) and lines[k].strip():
        n = int(lines[k].split()[0])
        m = re.search(r"\bE\s+(-?\d+\.\d+)", lines[k + 1])
        pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[k + 2:k + 2 + n]])
        frames.append((float(m.group(1)) if m else None, pos))
        k += 2 + n
    return frames


def trj_check(path, steps, info, converged):
    """frame i == step i (positions, energy). ORCA 6.1.1 appends one more frame after convergence: the geometry of the
    'FINAL ENERGY EVALUATION AT THE STATIONARY POINT', which is one optimizer step beyond the last gradient step (it has
    an energy but no gradient, so it is not a training structure). That frame is allowed iff the run converged and the
    output has exactly one coordinate block more than gradient steps; its offset from the last step is recorded."""
    p = Path(path)
    t = p.with_name(_stem(p.name) + "_trj.xyz")
    if not t.exists():
        return dict(trj=None)
    try:
        fr = read_trj(t)
    except Exception as e:                                       # noqa: BLE001
        return dict(trj=str(t), trj_ok=False, trj_reason=f"trj parse error {type(e).__name__}: {e}")
    n, ns = len(fr), len(steps)
    dpos = de = 0.0
    for (e, pos), s in zip(fr, steps):
        if pos.shape != s["pos_A"].shape:
            return dict(trj=str(t), trj_frames=n, trj_ok=False, trj_reason="atom count differs")
        dpos = max(dpos, float(np.abs(pos - s["pos_A"]).max()))
        if e is not None:
            de = max(de, abs(e - s["energy_eh"]))
    extra_ok, dfin = True, None
    if n == ns + 1 and ns:
        extra_ok = bool(converged and info.get("n_coord_blocks") == ns + 1)
        dfin = float(np.abs(fr[-1][1] - steps[-1]["pos_A"]).max())
    ok = (n in (ns, ns + 1)) and extra_ok and dpos <= TRJ_POS_TOL and de <= TRJ_E_TOL
    why = "" if ok else f"frames {n} vs steps {ns}, max dpos {dpos:.2e} A, max dE {de:.2e} Eh, extra_ok {extra_ok}"
    return dict(trj=str(t), trj_frames=n, trj_ok=ok, trj_reason=why, trj_max_dpos_A=dpos, trj_max_de_eh=de,
                trj_final_offset_A=dfin)


def parse_file(c):
    """one candidate -> (record, steps). steps only when the file is kept."""
    rec = dict(c)
    p = Path(c["path"])
    try:
        rec["bytes"] = p.stat().st_size
        rec["sha256"] = C.sha256(p)
        echo = O.input_echo(p)
        rec["bang"] = bang_line(echo)
        rec["job_type"] = job_type(echo)
        ok1, why1 = O.label_level(echo)
        ok2, why2 = same_grid(echo)
        rec["level_ok"] = bool(ok1 and ok2)
        rec["level_reason"] = why1 if not ok1 else why2
        rec["smd_block"] = bool(re.search(r"(?i)smd\s+true", echo))
        if not rec["level_ok"]:                                  # no full read: version from the header only
            rec.update(orca_version=head_version(p), kept=False, reason=f"not_label_level: {rec['level_reason']}")
            return rec, []
        ver, conv, notconv = text_flags(p)
        rec.update(orca_version=ver, converged=conv, not_converged_msg=notconv)
        steps, info = O.read_out_steps(p)
        rec.update(info)
    except Exception as e:                                       # noqa: BLE001
        rec.update(kept=False, reason=f"parse_error: {type(e).__name__}: {e}"[:300])
        return rec, []
    jt = rec["job_type"]
    n_c, n_s = info["n_opt_cycles"], info["n_steps"]
    if not info["terminated_normally"]:
        reason = "not_terminated_normally"
    elif not steps:
        reason = "no_gradient_steps"
    elif jt in ("optts", "opt", "const_opt", "scan") and n_s != n_c:
        reason = f"step_count_mismatch: {n_s} steps vs {n_c} cycles"
    elif jt == "engrad" and not (n_s == 1 and n_c == 0):
        reason = f"step_count_mismatch: engrad {n_s} steps, {n_c} cycles"
    elif jt not in ("optts", "opt", "const_opt", "scan", "engrad"):
        reason = f"unexpected_job_type_with_gradients: {jt}"
    else:
        reason = ""
    if not reason and any(s["cycle"] != i + 1 for i, s in enumerate(steps)) and jt != "engrad":
        reason = "cycle_numbering_mismatch"
    if steps:
        e = np.array([s["energy_eh"] for s in steps])
        rec.update(e_first_eh=float(e[0]), e_last_eh=float(e[-1]), de_kcal=float((e[-1] - e[0]) * C.EH2KCAL),
                   n_e_rises=int((np.diff(e) > 1e-5).sum()),
                   fmax_last=float(np.abs(steps[-1]["grad_eh_bohr"]).max()),
                   fmax_first=float(np.abs(steps[0]["grad_eh_bohr"]).max()), n_atoms=len(steps[0]["symbols"]))
    tc = trj_check(p, steps, info, rec.get("converged")) if steps else dict(trj=None)
    rec.update(tc)
    if not reason and tc.get("trj") and not tc.get("trj_ok"):
        reason = f"trj_mismatch: {tc.get('trj_reason')}"
    rec["kept"] = not reason
    rec["reason"] = reason
    if not rec["kept"]:
        return rec, []
    return rec, [dict(symbols=s["symbols"], pos_A=s["pos_A"], energy_eh=s["energy_eh"], grad_eh_bohr=s["grad_eh_bohr"],
                      cycle=s["cycle"]) for s in steps]


# ---------------------------------------------------------------- kinds and tags
def kind_of(rec, i, n):
    """kind of step i (0-based) of n of a kept file."""
    src, jt, sub = rec["source"], rec["job_type"], rec.get("sub", "")
    last_conv = (i == n - 1) and rec.get("converged") and jt == "optts"
    name = Path(rec["path"]).name
    if src in ("g3l", "partb", "g3_other"):
        if jt == "optts":
            return "ts_final" if last_conv else "ts_path"
        return "react_opt" if src == "g3_other" and rec["owner"].startswith("ref.") else f"other_{jt}"
    if src == "v_conf":
        return "react_opt" if jt == "opt" else f"other_{jt}"
    if src == "d1val":
        if jt == "optts":
            return "d0_ts_final" if last_conv else "d0_ts_path"
        if sub == "transition_states":
            if jt == "const_opt":
                return "d0_ts_guess"
            if "forwards" in name or "backwards" in name:
                return "d0_irc"
            return f"d0_other_{jt}"
        if sub == "reactants_and_products":
            return "d0_prod_opt" if name.startswith("p") else "d0_react_opt"
        if sub == "ref":
            return "d0_react_opt"
        if sub == "v2a":
            return "d0_ts_grad"
        if sub == "v7":
            return "d0_scan"
    return f"other_{jt}"


class Tagger:
    """leakage tags for D1 reactions, V molecules and d1val reactions / molecules."""

    def __init__(self):
        self.T = C.targets().set_index("rxn_id")
        labels = C.read_json(C.LABELS)
        self.key_rxns = defaultdict(set)          # full key (kind:smiles[|fixed:..]) -> rxn_ids
        self.smi_rxns = defaultdict(set)          # smiles part -> rxn_ids
        self.pair_rxns = defaultdict(set)         # sorted reactant smiles pair -> rxn_ids
        self.rxn_smiles = {}
        for r in labels:
            rid = int(r["rxn_id"])
            cr = r.get("cr") or {}
            sm = []
            for k in ("key1", "key2"):
                key = cr.get(k)
                if key:
                    self.key_rxns[key].add(rid)
                    s = key.split(":", 1)[1].split("|")[0]
                    self.smi_rxns[s].add(rid)
                    sm.append(s)
            if len(sm) == 2:
                self.pair_rxns[tuple(sorted(sm))].add(rid)
            self.rxn_smiles[rid] = r["rxn_smiles"]
        self._mm = None
        self._val = None

    def mm(self, smi):
        if self._mm is None:
            sys.path.insert(0, str(C.ANALYSIS / "d1_ladder"))
            import refs_q                                           # noqa: E402  (mirror-merged canonical SMILES)
            self._mm = refs_q.mirror_min
        try:
            return self._mm(smi)
        except Exception:                                          # noqa: BLE001
            return None

    def tags(self, rxns, rule):
        rxns = sorted(int(x) for x in rxns if int(x) in self.T.index)
        if not rxns:
            return dict(rxn_ids="", folds="", dipoles="", tag_rule=rule if rule != "d1_rxn" else "none")
        t = self.T.loc[rxns]
        return dict(rxn_ids=C.tag_list(rxns), folds=C.tag_list(t.fold_T1.tolist()), dipoles=C.tag_list(t.dipole.tolist()),
                    tag_rule=rule)

    # -- d1val reaction identities
    def val_info(self):
        if self._val is not None:
            return self._val
        info = {}
        by_tsid = {}
        for j in sorted(p for p in (C.D1VAL / "jobs").iterdir() if p.is_dir()):
            meta = None
            for n in ("label.json", "ts_result.json"):
                if (j / n).exists():
                    try:
                        meta = C.read_json(j / n)
                        break
                    except Exception:                              # noqa: BLE001
                        pass
            if meta:
                info[j.name] = dict(kind=meta.get("kind"), ts_id=meta.get("ts_id"), rxn_smiles=meta.get("rxn_smiles"))
                if meta.get("ts_id"):
                    by_tsid.setdefault(meta["ts_id"], info[j.name])
        for d in sorted((C.D1VAL / "v2a").glob("*")):
            info[f"v2a_{d.name}"] = by_tsid.get(f"D0-{d.name}", dict(kind=None, ts_id=f"D0-{d.name}", rxn_smiles=None))
        for d in sorted((C.D1VAL / "v7").glob("*")):
            p = C.D1VAL / "pilot_copy" / "jobs" / d.name / "label.json"
            meta = C.read_json(p) if p.exists() else {}
            info[f"v7_{d.name}"] = dict(kind=meta.get("kind"), ts_id=meta.get("ts_id"), rxn_smiles=meta.get("rxn_smiles"))
        for v in info.values():
            v.update(self.match_rxn(v.get("rxn_smiles")))
        self._val = info
        return info

    def match_rxn(self, rxn_smiles):
        if not rxn_smiles or ">>" not in rxn_smiles:
            return dict(d1_rxn_match=[], reactants_mm=None, comps=None)
        lhs, rhs = rxn_smiles.split(">>")
        comps = lhs.split(".")
        mm = [self.mm(s) for s in comps]
        cand = self.pair_rxns.get(tuple(sorted(mm)), set()) if len(mm) == 2 and None not in mm else set()
        prod = self.mm(rhs)
        hit = sorted(r for r in cand if self.mm(self.rxn_smiles[r].split(">>")[1]) == prod)
        return dict(d1_rxn_match=hit, reactants_mm=mm, comps=comps)

    def val_tags(self, rec, kind, symbols):
        vi = self.val_info().get(rec["val_job"], {})
        if vi.get("d1_rxn_match"):
            return self.tags(vi["d1_rxn_match"], "d1_rxn_match")
        if kind == "d0_react_opt" and vi.get("comps"):
            from rdkit import Chem
            form = Counter(symbols)
            hits = set()
            for s, m in zip(vi["comps"], vi["reactants_mm"]):
                mol = Chem.MolFromSmiles(s)
                if mol is None or m is None:
                    continue
                if Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()) == form:
                    hits |= self.smi_rxns.get(m, set())
            if hits:
                return self.tags(hits, "d1_mol_match")
        return dict(rxn_ids="", folds="", dipoles="", tag_rule="none")


def d_pairs(pos, pairs):
    pos = np.asarray(pos)
    return [float(np.linalg.norm(pos[a] - pos[b])) for a, b in pairs]


# ---------------------------------------------------------------- scan
def gather(limit=None):
    cands = cands_g3() + cands_v() + cands_d1val()
    if limit:
        by = defaultdict(list)
        for c in cands:
            by[c["source"]].append(c)
        cands = [c for v in by.values() for c in v[:limit]]
    return cands


def cmd_scan(args):
    t0 = time.time()
    ncore = args.cores or n_cpus()
    cands = gather(args.limit_per_source)
    print(f"[I] {len(cands)} candidate files, {ncore} processes", flush=True)
    tg = Tagger()
    results = []
    with Pool(ncore) as pool:
        for k, (rec, steps) in enumerate(pool.imap_unordered(parse_file, cands, chunksize=2)):
            results.append((rec, steps))
            if (k + 1) % 200 == 0:
                print(f"[I] parsed {k + 1}/{len(cands)} ({time.time() - t0:.0f} s)", flush=True)
    results.sort(key=lambda x: x[0]["path"])
    t_parse = time.time() - t0
    print(f"[I] parse done {t_parse:.0f} s", flush=True)

    T = tg.T
    label_cache = {}

    def label(rxn):
        if rxn not in label_cache:
            sym, pos = C.read_xyz(C.label_ts(rxn))
            label_cache[rxn] = (sym, np.asarray(pos), C.formed_pairs(T.loc[rxn, "formed_pairs_ts"]))
        return label_cache[rxn]

    atoms, seen = [], set()
    order_chk, gate_chk, ts_final_f = [], [], []
    for rec, steps in results:
        rec["n_structures"] = len(steps)
        rec["kinds"] = ""
        if not steps:
            continue
        src = rec["source"]
        kinds_here = Counter()
        base_tags = None
        rxn = rec.get("rxn_id")
        if src in ("g3l", "partb", "g3_other") and rxn:
            base_tags = tg.tags([rxn], "d1_rxn")
        elif src == "v_conf":
            base_tags = tg.tags(tg.key_rxns.get(rec["mol_key"], set()), "v_key")
        lab = None
        if src in ("g3l", "partb") and rxn in T.index:
            lab = label(rxn)
            same = list(steps[0]["symbols"]) == list(lab[0])
            rec["label_order"] = same
            order_chk.append(dict(path=rec["path"], rxn_id=rxn, same_elements=same))
        n = len(steps)
        sha = rec["sha256"][:16]
        for i, s in enumerate(steps):
            kind = kind_of(rec, i, n)
            tags = base_tags if base_tags is not None else (
                tg.val_tags(rec, kind, s["symbols"]) if src == "d1val" else dict(rxn_ids="", folds="", dipoles="",
                                                                                  tag_rule="none"))
            sid = f"I:{src}:{rxn if rxn else rec['owner']}:{rec['stem']}:{s['cycle']}"
            if sid in seen:
                C.stop(f"duplicate struct_id {sid}")
            seen.add(sid)
            extra = dict(group=rec.get("group") or tg.val_info().get(rec.get("val_job"), {}).get("ts_id") or rec["owner"],
                         cycle=int(s["cycle"]), n_cycles=int(rec.get("n_opt_cycles", 0)), file_sha=sha,
                         fmax=float(np.abs(s["grad_eh_bohr"]).max()), tag_rule=tags.get("tag_rule", "none"))
            if lab is not None and rec.get("label_order"):
                d = d_pairs(s["pos_A"], lab[2])
                dl = d_pairs(lab[1], lab[2])
                extra["dform_label_A"] = float(max(abs(a - b) for a, b in zip(d, dl)))
            a = C.ext_atoms(s["symbols"], s["pos_A"], s["energy_eh"], s["grad_eh_bohr"], struct_id=sid, kind=kind,
                            source=src, rxn_ids=tags["rxn_ids"], folds=tags["folds"], dipoles=tags["dipoles"],
                            level="label", **extra)
            atoms.append(a)
            kinds_here[kind] += 1
            if kind in ("ts_final", "d0_ts_final"):
                ts_final_f.append(dict(path=rec["path"], kind=kind, fmax=extra["fmax"]))
        if lab is not None and rec.get("label_order") and rec.get("unit_final") and rec.get("converged"):
            uj = Path(C.G3 / "O" / "units" / rec["owner"] / "unit.json")
            g = C.read_json(uj).get("gate", {}) if uj.exists() else {}
            if g.get("d_form"):
                d = d_pairs(steps[-1]["pos_A"], lab[2])
                gate_chk.append(dict(owner=rec["owner"], max_dev_A=float(np.abs(np.sort(d) - np.sort(g["d_form"])).max())))
        rec["kinds"] = ";".join(f"{k}:{v}" for k, v in sorted(kinds_here.items()))

    # ---------------------------------------------------------------- summaries
    inv = summarise(results, atoms, tg, order_chk, gate_chk, ts_final_f)
    inv.update(not_scanned=not_scanned(), d1_archives=d1_archives())
    inv["meta"] = dict(code=CODE_VERSION, time=C.now(), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                       n_cpus=ncore, wall_s=round(time.time() - t0, 1), parse_s=round(t_parse, 1),
                       limit_per_source=args.limit_per_source, extxyz=str(OUT_EXT) if not args.dry else None,
                       units="extxyz REF_energy eV, REF_forces eV/A (= -gradient); e_eh, fmax (Eh/bohr) in info")
    if args.dry:
        print(json.dumps(inv["summary"], indent=1, default=str))
        return 0
    C.write_extxyz(OUT_EXT, atoms)
    write_sources(results)
    C.write_json(RES_INV, inv)
    print(json.dumps(inv["summary"], indent=1, default=str))
    print(f"[I] wrote {OUT_EXT} ({len(atoms)} structures), {RES_INV}, {RES_SRC}; {time.time() - t0:.0f} s", flush=True)
    return 0


SRC_COLS = ["source", "sub", "owner", "group", "rxn_id", "path", "bytes", "sha256", "orca_version", "bang", "job_type",
            "level_ok", "level_reason", "smd_block", "terminated_normally", "converged", "n_opt_cycles", "n_coord_blocks",
            "n_energies", "n_gradients", "n_steps", "trj", "trj_frames", "trj_ok", "kept", "reason", "n_structures",
            "kinds", "e_first_eh", "e_last_eh", "de_kcal", "n_e_rises", "fmax_first", "fmax_last", "label_order",
            "trj_final_offset_A"]


def write_sources(results):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SRC_COLS, extrasaction="ignore")
    w.writeheader()
    for rec, _ in results:
        w.writerow({k: rec.get(k) for k in SRC_COLS})
    C.write_text(RES_SRC, buf.getvalue())


def _q(x, qs=(0.0, 0.5, 0.9, 1.0)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {f"q{int(q * 100)}": (float(np.quantile(x, q)) if len(x) else None) for q in qs} | {"n": int(len(x))}


def summarise(results, atoms, tg, order_chk, gate_chk, ts_final_f):
    src = defaultdict(lambda: dict(files_found=0, files_kept=0, files_excluded=0, excluded_reasons=Counter(),
                                   keywords_seen=Counter(), keywords_kept=Counter(), orca_versions=Counter(),
                                   job_types=Counter(), step_mismatch=[], trj_checked=0, trj_ok=0))
    for rec, _ in results:
        s = src[rec["source"]]
        s["files_found"] += 1
        s["keywords_seen"][rec.get("bang") or "?"] += 1
        s["orca_versions"][str(rec.get("orca_version"))] += 1
        s["job_types"][rec.get("job_type") or "?"] += 1
        if rec.get("trj"):
            s["trj_checked"] += 1
            s["trj_ok"] += bool(rec.get("trj_ok"))
        if rec.get("kept"):
            s["files_kept"] += 1
            s["keywords_kept"][rec.get("bang") or "?"] += 1
        else:
            s["files_excluded"] += 1
            r = rec.get("reason", "")
            s["excluded_reasons"][r.split(":")[0] + (":" + r.split(":")[1].strip() if r.startswith("not_label") else "")] += 1
            if r.startswith(("step_count", "trj_mismatch", "cycle_numbering")):
                s["step_mismatch"].append(dict(path=rec["path"], reason=r, n_opt_cycles=rec.get("n_opt_cycles"),
                                               n_steps=rec.get("n_steps"), n_gradients=rec.get("n_gradients"),
                                               n_energies=rec.get("n_energies"), n_coord=rec.get("n_coord_blocks")))
    by_kind = defaultdict(lambda: dict(structures=0, groups=set(), d1_rxns=set(), files=set(), tagged=0))
    by_src_kind = defaultdict(lambda: defaultdict(lambda: dict(structures=0, groups=set(), d1_rxns=set())))
    i1 = defaultdict(int)
    dform = []
    for a in atoms:
        k, s_ = a.info["kind"], a.info["source"]
        g = str(a.info.get("group"))
        rx = [x for x in str(a.info["rxn_ids"]).split(",") if x]
        b = by_kind[k]
        b["structures"] += 1
        b["groups"].add(g)
        b["d1_rxns"].update(rx)
        b["tagged"] += bool(rx)
        b["files"].add(a.info["file_sha"])
        bs = by_src_kind[s_][k]
        bs["structures"] += 1
        bs["groups"].add(g)
        bs["d1_rxns"].update(rx)
        if k in TS_KINDS and "dform_label_A" in a.info:
            dform.append(a.info["dform_label_A"])
            if a.info["dform_label_A"] <= I1_THR_A:
                i1[rx[0]] += 1
    kinds = {k: dict(structures=v["structures"], groups=len(v["groups"]), d1_reactions=len(v["d1_rxns"]),
                     files=len(v["files"]), structures_with_d1_tags=v["tagged"]) for k, v in sorted(by_kind.items())}
    for k in ALL_KINDS:
        kinds.setdefault(k, dict(structures=0, groups=0, d1_reactions=0, files=0, structures_with_d1_tags=0))
    sources = {}
    for name, s in sorted(src.items()):
        sources[name] = dict(
            files_found=s["files_found"], files_kept=s["files_kept"], files_excluded=s["files_excluded"],
            excluded_reasons=dict(s["excluded_reasons"]), job_types=dict(s["job_types"]),
            keywords_seen=dict(s["keywords_seen"].most_common()), keywords_kept=dict(s["keywords_kept"].most_common()),
            orca_versions=dict(s["orca_versions"]), trj_checked=s["trj_checked"], trj_ok=s["trj_ok"],
            step_mismatch=s["step_mismatch"][:40], n_step_mismatch=len(s["step_mismatch"]),
            kinds={k: dict(structures=v["structures"], groups=len(v["groups"]), d1_reactions=len(v["d1_rxns"]))
                   for k, v in sorted(by_src_kind[name].items())})
    n1, r1 = int(sum(i1.values())), len(i1)
    I1 = dict(threshold_A=I1_THR_A, n_structures=n1, n_reactions=r1, min_structures=I1_MIN_STRUCT, min_reactions=I1_MIN_RXN,
              pass_=bool(n1 >= I1_MIN_STRUCT and r1 >= I1_MIN_RXN), need_G2=not (n1 >= I1_MIN_STRUCT and r1 >= I1_MIN_RXN),
              n_d1_ts_structures=len(dform), dform_label_A=_q(dform, (0.0, 0.1, 0.5, 0.9, 1.0)),
              per_reaction=dict(sorted(i1.items())))
    tsf = [x["fmax"] for x in ts_final_f]
    checks = dict(
        atom_order=dict(n_files=len(order_chk), n_label_order=sum(x["same_elements"] for x in order_chk),
                        not_label_order=[x for x in order_chk if not x["same_elements"]][:20],
                        gate_d_form_vs_parsed=dict(n=len(gate_chk), max_dev_A=max((x["max_dev_A"] for x in gate_chk),
                                                                                    default=None),
                                                   worst=sorted(gate_chk, key=lambda x: -x["max_dev_A"])[:5])),
        converged_ts_fmax=dict(threshold_eh_bohr=FMAX_TS_OK, n=len(tsf), n_above=int(sum(f > FMAX_TS_OK for f in tsf)),
                               above=[x for x in ts_final_f if x["fmax"] > FMAX_TS_OK][:20], dist=_q(tsf)),
        trj_final_frame_offset_A=dict(note="ORCA 6.1.1 final energy evaluation geometry vs the last gradient step "
                                           "(max |dx|, files with *_trj.xyz)",
                                      dist=_q([r.get("trj_final_offset_A") for r, _ in results])),
        energies=energy_checks(results))
    val = tg.val_info()
    vsum = dict(n_jobs=len(val), kinds=dict(Counter(str(v.get("kind")) for v in val.values())),
                d1_rxn_matches={j: v["d1_rxn_match"] for j, v in val.items() if v.get("d1_rxn_match")},
                tag_rules=dict(Counter(a.info.get("tag_rule") for a in atoms if a.info["source"] == "d1val")))
    summary = dict(n_files=len(results), n_kept=sum(1 for r, _ in results if r.get("kept")), n_structures=len(atoms),
                   kinds={k: v["structures"] for k, v in kinds.items()}, I1=dict(n_structures=n1, n_reactions=r1,
                                                                                   pass_=I1["pass_"], need_G2=I1["need_G2"]),
                   sources={k: dict(found=v["files_found"], kept=v["files_kept"], excluded=v["files_excluded"],
                                    reasons=v["excluded_reasons"]) for k, v in sources.items()},
                   checks=dict(label_order=f"{checks['atom_order']['n_label_order']}/{checks['atom_order']['n_files']}",
                               gate_d_form_max_dev_A=checks["atom_order"]["gate_d_form_vs_parsed"]["max_dev_A"],
                               ts_final_fmax_above=checks["converged_ts_fmax"]["n_above"],
                               ts_final_fmax_max=checks["converged_ts_fmax"]["dist"].get("q100")))
    return dict(summary=summary, sources=sources, kinds=kinds, I1=I1, checks=checks, d1val=vsum,
                spec="docs/specs/DEPLOY_V2_MLTS.md section 2",
                rules=dict(level="orca_io.label_level (B3LYP D3BJ def2-SVP + SMD water) and RIJCOSX def2/J without a grid "
                                 "keyword (D1 autodE inputs; ORCA 6.1.1 default grid)",
                           steps="parsed (coords, E, CARTESIAN GRADIENT) steps == 'GEOMETRY OPTIMIZATION CYCLE' headers "
                                 "(EnGrad: 1 step, 0 cycles); *_trj.xyz frames compared where present",
                           ts_final="last step of an OptTS whose output says THE OPTIMIZATION HAS CONVERGED (not also "
                                    "counted as ts_path)"))


def energy_checks(results):
    out = {}
    for jt in ("opt", "optts", "const_opt", "scan"):
        rr = [r for r, _ in results if r.get("kept") and r.get("job_type") == jt]
        if not rr:
            continue
        out[jt] = dict(n_files=len(rr), de_kcal=_q([r.get("de_kcal") for r in rr]),
                       frac_e_last_le_first=float(np.mean([r["e_last_eh"] <= r["e_first_eh"] + 1e-8 for r in rr])),
                       n_e_rises=_q([r.get("n_e_rises") for r in rr]), fmax_first=_q([r.get("fmax_first") for r in rr]),
                       fmax_last=_q([r.get("fmax_last") for r in rr]),
                       n_converged=int(sum(bool(r.get("converged")) for r in rr)))
    return out


# ---------------------------------------------------------------- probe / report
def cmd_probe(args):
    for f in args.files:
        c = dict(source="probe", path=f, owner=Path(f).parent.name, stem=_stem(Path(f).name), sub=Path(f).parent.name)
        rec, steps = parse_file(c)
        keep = {k: rec.get(k) for k in SRC_COLS + ["trj_reason", "trj_max_dpos_A", "trj_max_de_eh", "not_converged_msg"]
                if rec.get(k) is not None and k not in ("sha256",)}
        print(json.dumps(keep, indent=1, default=str))
        if steps:
            e = np.array([s["energy_eh"] for s in steps])
            fm = [float(np.abs(s["grad_eh_bohr"]).max()) for s in steps]
            print(f"  E (Eh) first/last {e[0]:.8f} / {e[-1]:.8f}; dE kcal by step:",
                  np.round(np.diff(e) * C.EH2KCAL, 3).tolist()[:40])
            print("  fmax by step (Eh/bohr):", [f"{x:.1e}" for x in fm][:40])
        if args.roundtrip and steps:
            a = [C.ext_atoms(s["symbols"], s["pos_A"], s["energy_eh"], s["grad_eh_bohr"], struct_id=f"probe:{i}",
                             kind="ts_path", source="probe", rxn_ids="" if i % 2 else "1000301,1000387",
                             folds="" if i % 2 else "0,3", dipoles="" if i % 2 else "azide,azomethine ylide",
                             group="1000301", fmax=0.1, cycle=s["cycle"]) for i, s in enumerate(steps[:4])]
            tmp = C.MLT / "tmp" / f"inv_probe_{os.getpid()}.extxyz"
            C.write_extxyz(tmp, a)
            b = C.read_extxyz(tmp)
            tmp.unlink()
            for x, y in zip(a, b):
                print("  roundtrip info:", {k: (x.info.get(k), y.info.get(k)) for k in ("struct_id", "rxn_ids", "folds",
                                                                                        "dipoles", "group", "kind")},
                      "dE", abs(x.info["REF_energy"] - y.info["REF_energy"]),
                      "dF", float(np.abs(x.arrays["REF_forces"] - y.arrays["REF_forces"]).max()),
                      "dpos", float(np.abs(x.positions - y.positions).max()))
    return 0


def cmd_verify(_args):
    """read $MLT/dft/existing.extxyz back (mlt_common.read_extxyz) and check it against I_inventory.json."""
    if not (OUT_EXT.exists() and RES_INV.exists()):
        C.not_ready("run `inv_dft.py scan` first")
    inv = C.read_json(RES_INV)
    al = C.read_extxyz(OUT_EXT)
    T = C.targets().set_index("rxn_id")
    bad = Counter()
    kinds, folds_seen, rule = Counter(), Counter(), Counter()
    fmax_all = []
    for a in al:
        kinds[a.info["kind"]] += 1
        rule[a.info.get("tag_rule")] += 1
        rx = [x for x in a.info["rxn_ids"].split(",") if x]
        fo = [x for x in a.info["folds"].split(",") if x]
        dp = [x for x in a.info["dipoles"].split(",") if x]
        folds_seen.update(fo)
        if any(not x.isdigit() or int(x) not in T.index for x in rx):
            bad["rxn_ids_not_d1"] += 1
        if rx and set(fo) != {str(int(T.loc[int(x), "fold_T1"])) for x in rx}:
            bad["folds_ne_targets"] += 1
        if rx and set(dp) != {T.loc[int(x), "dipole"] for x in rx}:
            bad["dipoles_ne_targets"] += 1
        if bool(rx) != bool(fo) or bool(rx) != bool(dp):
            bad["tags_partial"] += 1
        F = a.arrays["REF_forces"]
        if not (np.isfinite(F).all() and np.isfinite(a.info["REF_energy"])):
            bad["non_finite"] += 1
        if abs(np.abs(F).max() / C.EHBOHR2EVA - float(a.info["fmax"])) > 1e-8:
            bad["fmax_ne_forces"] += 1
        if abs(a.info["REF_energy"] / C.EH2EV - float(a.info["e_eh"])) > 1e-9:
            bad["energy_units"] += 1
        if a.info["kind"] in TS_KINDS and "dform_label_A" not in a.info:
            bad["d1_ts_without_dform"] += 1
        fmax_all.append(float(a.info["fmax"]))
    ids = [a.info["struct_id"] for a in al]
    fm = np.array(fmax_all)
    out = dict(n_read=len(al), n_inventory=inv["summary"]["n_structures"], unique_ids=len(set(ids)) == len(ids),
               kinds_match=dict(kinds) == {k: v for k, v in inv["summary"]["kinds"].items() if v},
               problems=dict(bad), folds_seen=dict(folds_seen), tag_rules=dict(rule),
               fmax_eh_bohr=dict(q50=float(np.median(fm)), q99=float(np.quantile(fm, 0.99)), max=float(fm.max()),
                                 n_gt_0p05=int((fm > 0.05).sum()), n_gt_0p1=int((fm > 0.1).sum())))
    print(json.dumps(out, indent=1))
    return 0 if (out["n_read"] == out["n_inventory"] and out["unique_ids"] and out["kinds_match"] and not bad) else 3


def cmd_report(_args):
    if not RES_INV.exists():
        C.not_ready(f"{RES_INV} missing: run `inv_dft.py scan` first")
    inv = C.read_json(RES_INV)
    print(json.dumps(dict(summary=inv["summary"], I1={k: v for k, v in inv["I1"].items() if k != "per_reaction"},
                          meta=inv.get("meta")), indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--cores", type=int, default=None)
    s.add_argument("--limit-per-source", type=int, default=None, help="test: first K candidate files per source")
    s.add_argument("--dry", action="store_true", help="no writes; print the summary")
    p = sub.add_parser("probe")
    p.add_argument("files", nargs="+")
    p.add_argument("--roundtrip", action="store_true", help="extxyz write/read round trip in memory")
    sub.add_parser("verify")
    sub.add_parser("report")
    args = ap.parse_args()
    return dict(scan=cmd_scan, probe=cmd_probe, verify=cmd_verify, report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
