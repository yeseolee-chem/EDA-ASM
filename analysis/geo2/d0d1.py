#!/usr/bin/env python3
r"""d0d1.py — GEO2b D0 Phase X (D0 HF templates and checks X-1 .. X-4), the combined D1 + D0 input root of the geo
rung (design option F/H-1) and the D0 S-1 check (analysis/geo1b/results/D0D1_design_investigation.json, design steps 6-8
and result.investigations[k=inputs]; user decision D0D1, 2026-10-31). sbatch only (D1-HF job environment:
analysis/d1_hf jobs/hf.slurm = conda reactot + ORCA 6.1.1); nothing in this file runs on the login node.

  d0d1.py x [N] [--force]   D0 Phase X + the combined root; N worker processes (default $SLURM_CPUS_PER_TASK)
  d0d1.py s1 [N]            D0 S-1: the original B3LYP / def2-TZVP inputs of 3 D0 reactions at %pal 1 through the D1-HF
                            runner (run.py run <list> dft orig1 N) and build.py parse_rxn(T, 'dft', 'orig1') vs
                            labels_all.json (<= 0.01 kcal/mol, 9 values of D1-HF S-1 + c_ghost = bsse_shift_kcal)
  internal: _s1parse <list.csv> <out.json>   (child of s1: parse_rxn under D1HF_SCRATCH = $GEO2B/x_d0)

Phase X, per D0 reaction (labels_all.json status ok, not one of the 5 GEO1 D-1 overlaps = d0_folds.csv fold -9; job_id
D0G2_<rid:04d> as d0_manifest / labels_d0_q.json / D0_pairs.csv), the 5 label inputs (read only):
  eda.inp        /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/rxn_NNNN/eda.inp     (the SMD(water) relabel input)
  frag*_dist/rel <main checkout>/label_true/work/inputs/rxn_NNNN/frag{1,2}_{dist,rel}.inp (CPCM(water) + %cpcm smd)
  X-1 SMD + keywords: hf_common.smd_ok of every input; the keyword inventory of prep.check_job (EDA_BANG, FRAG_STR,
      SA_BANG, EDA_BLOCKS, SA_BLOCKS, %maxcore 3500, FRAG1/2 _C / _M) -> a mismatch is a STOP (as D1-HF X)
  X-2 coordinates (<= 1e-6 A, elements in order; prep.same_atoms): eda.inp atoms = the Coley TS file
      (full_dataset_profiles/<rid>/<ts_file>) = d0_g2/labjobs/<T>/ts.xyz; frag{k}_dist = the (k) atoms of eda.inp;
      frag{k}_rel = the Coley reference file of the label (rel{k}_file) = labjobs ref_dipole / ref_dipolarophile.xyz
  X-3 fragments: the (1) tags of eda.inp = input_meta.csv A_idx; n_f1 / n_f2 / charges = the label; frag1 = dipole
      (role1, element multiset of the dipole reference) else STOP
  X-4 provenance (added; DECISIONS_LAB L-prov precedent): eda.out of the SMD relabel and the four label_true frag*.out
      terminated normally with one FINAL SINGLE POINT ENERGY = the label e_ab_eh / e_frag{k}_{dist,rel}_eh exactly
  HF template = prep.hf_text (B3LYP -> HF, def2-TZVP -> 6-31G* in the '!' line and the FRAG strings, eda %pal nprocs 1)
      with the prep.check_job diff check -> $GEO2B/x_d0/inputs/<T>/{orig,hf}/<calc>.inp + check.json (D1-HF layout,
      atomic per reaction; an unchanged check.json with the same sources and code version is kept)
  STOP (= D1-HF X rule): any stop reason, or x_fail > 0.5 % of the D0 reactions.
Combined root $GEO2B/comb (read by feat_geo.py G / F / H / P; recorded in results/X_d0d1_report.json, the 'new sha
record' that feat_geo.inputs_check compares; every symlink dir has a listing digest):
  labels_d0d1.json       labels_d1_g2.json rows verbatim (sha256 = analysis/d1_g2/results/G_inputs_report.json) +
                         the 5,255 D0 rows of labels_all.json verbatim with job_id D0G2_<rid:04d> and kind 'D0'
  input_meta_d0d1.csv    rxn_id, A_idx: input_meta_d1_g2.csv rows verbatim + the D0 rows of label_true input_meta.csv
  x_manifest_d0d1.csv    job_id, rxn_id, dataset, n_atoms, n_f1, n_f2, charge, elements, x_fail, reasons (D1 rows =
                         analysis/d1_hf/results/X_manifest.csv, all x_fail False; D0 rows = Phase X)
  prof/<rxn_id>          -> D1: the d1_g2/prof link target (d1_pair/d1_jobs/T*); D0: full_dataset_profiles/<rid>
  jobs/<job_id>          -> D1: d1_pair/d1_jobs/<T>; D0: d0_g2/labjobs/<T> (ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz:
                         the element-order references of the G gate and of prep.g1_input)
  hf_templates/<job_id>  -> D1: d1_hf/inputs/<T>; D0 (X ok only): $GEO2B/x_d0/inputs/<T>  (<T>/hf/*.inp = D1HF_TEMPLATES)
A combined file or link that exists with other content is a STOP ('x --force' first moves $GEO2B/comb aside to
comb.prev_<time>; explicit only, never automatic).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import g3c as C  # noqa: E402

X_VERSION = "geo2b-d0x-1"
MAIN = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")              # main checkout (read only)
D0_LABELS = MAIN / "labels_all.json"
D0_LABELS_SHA = "62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753"   # = geo1b d0_labels.LABELS_SHA
D0_META = MAIN / "label_true" / "work" / "input_meta.csv"
D0_LABEL_INPUTS = MAIN / "label_true" / "work" / "inputs"                          # rxn_NNNN/frag*.inp / .out
D0_SMD = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")                     # rxn_NNNN/eda.inp / .out
D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
D0_LABJOBS = Path("/gpfs/tmp_cpu2/yeseo1ee/d0_g2/labjobs")                          # D0G2_NNNN/{ts,ref_*}.xyz
GEO1B = Path(os.environ.get("GEO_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo1b"))
D0_FOLDS = GEO1B / "d0" / "d0_folds.csv"
D0_PAIRS_REPORT = C.REPO / "analysis" / "geo1b" / "results" / "D0_pairs_report.json"
D0_N_OVERLAP = 5                                                                   # GEO1 decision D-1
D1_LABELS = C.G2 / "G" / "labels_d1_g2.json"
D1_META = C.G2 / "G" / "input_meta_d1_g2.csv"
D1_PROF = C.G2 / "prof"
D1_G2_INPUTS_REPORT = C.G2_RES / "G_inputs_report.json"
D1_X_MANIFEST = C.HF_DIR / "results" / "X_manifest.csv"
D1_JOBS = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs")
D1_TEMPLATES = C.HF_SCR / "inputs"
D1_RXN_MIN = 1_000_000                                                             # D0 rxn_id 0..5973 (Coley ids), D1 >= 1,000,001

X_DIR = C.G3 / "x_d0"                                     # D1HF_SCRATCH of the D0 templates / S-1 (D1-HF layout)
X_INPUTS = X_DIR / "inputs"
COMB = C.G3 / "comb"
COMB_LABELS = COMB / "labels_d0d1.json"
COMB_META = COMB / "input_meta_d0d1.csv"
COMB_MANIFEST = COMB / "x_manifest_d0d1.csv"
COMB_PROF, COMB_JOBS, COMB_TEMPLATES = COMB / "prof", COMB / "jobs", COMB / "hf_templates"
LINK_DIRS = ("prof", "jobs", "hf_templates")
X_MANIFEST_D0 = C.RES / "X_d0_manifest.csv"
X_REPORT = C.RES / "X_d0d1_report.json"
S1_JSON, S1_CSV = C.RES / "X_S1.json", C.RES / "X_S1_compare.csv"
X_FAIL_MAX = 0.005                                        # = prep.cmd_report (D1-HF X: x_fail > 0.5 % = STOP)
S1_N_NEUTRAL, S1_N_CHARGED = 2, 1
S1_EXTRA = {"c_ghost": "bsse_shift_kcal"}                 # D1-HF S1_KEYS + c_ghost (label key = parse key)
MAN_COLS = ["job_id", "rxn_id", "dataset", "n_atoms", "n_f1", "n_f2", "charge", "elements", "x_fail", "reasons"]
LABEL_E = {"eda": "e_ab_eh", "frag1_dist": "e_frag1_dist_eh", "frag2_dist": "e_frag2_dist_eh",
           "frag1_rel": "e_frag1_rel_eh", "frag2_rel": "e_frag2_rel_eh"}

DECISIONS = {
    "X-0 scope": "D0 = labels_all.json status ok 5,260 − GEO1 D-1 겹침 5(d0_folds.csv fold −9) = 5,255반응. job_id = "
                 "D0G2_<rid:04d>(d0_manifest·labels_d0_q·D0_pairs.csv와 같은 형식). labels_all.json은 main checkout 사본"
                 "(sha256 62c0e784… 고정)을 읽는다.",
    "X-1 SMD": "hf_common.smd_ok(eda: SMD(water)가 '!' 줄과 FRAG 두 문자열에, %cpcm 없음; frag: CPCM(water) + %cpcm smd "
               "true / smdsolvent water)와 prep.check_job의 키워드 목록(EDA_BANG, FRAG_STR, SA_BANG, 블록, %maxcore 3500) "
               "그대로. 어긋나면 STOP(D1-HF X와 같음).",
    "X-2 coordinates": "eda.inp 원자 = Coley TS 파일(full_dataset_profiles/<rid>/<ts_file>) = d0_g2/labjobs/<T>/ts.xyz; "
                       "frag{k}_dist = eda.inp (k) 원자; frag{k}_rel = 라벨 rel{k}_file(Coley 기준 구조) = labjobs "
                       "ref_dipole / ref_dipolarophile.xyz. 원소 순서 같고 좌표 <= 1e-6 Å(prep.same_atoms).",
    "X-3 tags": "eda.inp (1) 태그 = label_true input_meta.csv A_idx; n_f1·n_f2·전하 = 라벨; frag1 = dipole(아니면 STOP).",
    "X-4 provenance": "새 검사(DECISIONS_LAB L-prov 선례): SMD relabel eda.out과 label_true frag*.out이 정상 종료했고 FINAL "
                      "SINGLE POINT ENERGY가 하나이며 라벨 e_ab_eh / e_frag{k}_{dist,rel}_eh와 정확히 같다. 템플릿 입력이 "
                      "라벨을 만든 바로 그 입력인지 보인다. 어긋나면 그 반응만 x_fail.",
    "X-5 HF template": "prep.hf_text(B3LYP -> HF, def2-TZVP -> 6-31G*, eda %pal nprocs 1)를 그대로 쓰고 prep.check_job의 "
                       "줄 단위 diff 검사를 그대로 옮겼다. $GEO2B/x_d0/inputs/<T>/{orig,hf}/*.inp + check.json(D1-HF "
                       "형식). x_fail > 0.5 % = STOP(prep.cmd_report 규칙).",
    "X-6 combined root": "설계 F/H-1: $GEO2B/comb에 labels_d0d1.json(D1 행 = labels_d1_g2 그대로, D0 행 = labels_all 행 "
                         "그대로 + job_id + kind 'D0'), input_meta_d0d1.csv, x_manifest_d0d1.csv, prof·jobs·hf_templates "
                         "symlink 폴더. sha256과 symlink 목록 digest를 results/X_d0d1_report.json에 적고, feat_geo가 "
                         "매번 다시 비교한다(새 sha 기록). 이미 있는 파일·링크가 다르면 STOP('x --force'만 옮김).",
    "X-7 S-1": "D0 S-1: X ok D0 중 (n_atoms, rxn_id) 순으로 중성 2개 + 전하 반응 1개. 원래 B3LYP 입력(orig, %pal 1)을 "
               "D1-HF run.py(dft / orig1)로 다시 돌려 build.parse_rxn 9값(D1-HF S1_KEYS) + c_ghost(bsse_shift_kcal)를 "
               "labels_all.json과 비교, 모두 <= 0.01 kcal/mol이고 status ok이어야 한다(아니면 STOP).",
}


# ================================================================ helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def job_id(rid):
    return f"D0G2_{int(rid):04d}"


def dataset_of(rid):
    return "D1" if int(rid) > D1_RXN_MIN else "D0"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def rxn_dir(root, rid):
    return Path(root) / f"rxn_{int(rid):04d}"


def hf_modules():
    """analysis/d1_hf hf_common and prep (D1-HF job environment: this process keeps the D1-HF defaults; the
    functions used here only parse text and never write)."""
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import hf_common as H
    import prep as PR
    if Path(H.__file__).resolve().parent != C.HF_DIR.resolve() or Path(PR.__file__).resolve().parent != C.HF_DIR.resolve():
        C.stop(f"hf_common / prep not imported from {C.HF_DIR}")
    return H, PR


def listing_digest(d):
    """sha256 of the sorted 'name\\ttarget' lines of a symlink folder (None when the folder is missing)."""
    d = Path(d)
    if not d.is_dir():
        return None, 0
    names = sorted(os.listdir(d))
    h = hashlib.sha256()
    for n in names:
        p = d / n
        h.update(f"{n}\t{os.readlink(p) if p.is_symlink() else '<not a link>'}\n".encode())
    return h.hexdigest(), len(names)


# ================================================================ sources
def load_d0_rows(rec):
    """the D0 rows of the geo rung: labels_all.json status ok (sha256 pinned) minus the d0_folds.csv overlaps (fold -9);
    d0_folds.csv must belong to a non-STOP D0_pairs_report.json (geo_common.load_d0_folds rule) -> {rid: row}."""
    sha = C.sha256(D0_LABELS)
    if sha != D0_LABELS_SHA:
        C.stop(f"{D0_LABELS} sha256 {sha} != {D0_LABELS_SHA} (X-0)")
    rows = json.loads(D0_LABELS.read_text())
    ok = {int(d["rxn_id"]): d for d in rows if d.get("status") == "ok"}
    rep = json.loads(D0_PAIRS_REPORT.read_text()) if D0_PAIRS_REPORT.is_file() else {}
    outs = [v for k, v in (rep.get("outputs") or {}).items() if Path(k).name == D0_FOLDS.name]
    fsha = C.sha256(D0_FOLDS) if D0_FOLDS.is_file() else None
    if rep.get("stop") is not False or len(outs) != 1 or (outs[0] or {}).get("sha256") != fsha:
        C.stop(f"{D0_FOLDS} is not the passing output of {D0_PAIRS_REPORT} (stop {rep.get('stop')!r}, sha256 recorded "
               f"{[o.get('sha256') for o in outs]}, file {fsha})")
    with open(D0_FOLDS, newline="") as f:
        F = list(csv.DictReader(f))
    folds = {int(r["rxn_id"]): r for r in F}
    if set(folds) != set(ok):
        C.stop(f"d0_folds.csv rows ({len(folds)}) != labels_all status-ok rows ({len(ok)})")
    over = sorted(r for r, x in folds.items() if x["overlap"] == "True")
    if len(over) != D0_N_OVERLAP or any(folds[r]["fold"] != "-9" for r in over):
        C.stop(f"d0_folds.csv: {len(over)} overlap rows {over} (need {D0_N_OVERLAP}, fold -9)")
    if any(r >= D1_RXN_MIN for r in ok):
        C.stop("a D0 rxn_id >= 1,000,000 would collide with the D1 rxn_id range")
    keep = {r: d for r, d in ok.items() if r not in set(over)}
    rec["d0_sources"] = dict(labels=str(D0_LABELS), labels_sha256=sha, n_status_ok=len(ok), overlap=over,
                             d0_folds=str(D0_FOLDS), d0_folds_sha256=fsha, n_rows=len(keep),
                             d0_pairs_g2_source=rep.get("g2_source"))
    return keep, {r: folds[r] for r in keep}


def read_meta_csv(path):
    """{rxn_id: A_idx string} of an input_meta csv (columns rxn_id, A_idx at least)."""
    with open(path, newline="") as f:
        return {int(r["rxn_id"]): str(r["A_idx"]).strip() for r in csv.DictReader(f)}


def a_list(s):
    return sorted(int(x) for x in str(s).split())


# ================================================================ Phase X (one D0 reaction)
def hf_diff_bad(H, PR, txt, kind, p):
    """the prep.check_job X-3 diff check of one input (verbatim logic): (new text, bad?)."""
    new, changed = PR.hf_text(txt, kind)
    o, n = txt.splitlines(), new.splitlines()
    allowed = {i for i, ln in enumerate(o) if ln.strip().startswith("!")}
    if kind == "eda":
        allowed |= {i for i, ln in enumerate(o) if ln.strip().split()[:1] in (["FRAG1"], ["FRAG2"])
                    or ln.strip().lower().startswith("%pal")}
    pn = H.parse_inp(new)
    want_bang = [dict(H.SUB).get(t, t) for t in p["bang"]]
    bad = (len(o) != len(n) or not set(changed) <= allowed or pn["bang"] != want_bang
           or pn["atoms"] != p["atoms"] or (pn["charge"], pn["mult"]) != (p["charge"], p["mult"])
           or sum(ln.count("HF") for ln in n) - sum(ln.count("HF") for ln in o) != (3 if kind == "eda" else 1)
           or sum(ln.count("6-31G*") for ln in n) != (3 if kind == "eda" else 1)
           or any(w in new for w in ("B3LYP", "def2-TZVP")))
    if kind == "eda":
        bad |= (pn["pal"] != 1 or any(pn["frag"].get(f, "").split() != [dict(H.SUB).get(t, t) for t in PR.FRAG_STR]
                                      for f in ("FRAG1", "FRAG2"))
                or {k: v for k, v in pn["frag"].items() if k not in ("FRAG1", "FRAG2")}
                != {k: v for k, v in p["frag"].items() if k not in ("FRAG1", "FRAG2")})
    return new, bool(bad)


def source_paths(rid):
    s = {"eda": rxn_dir(D0_SMD, rid) / "eda.inp"}
    s.update({c: rxn_dir(D0_LABEL_INPUTS, rid) / f"{c}.inp" for c in ("frag1_dist", "frag2_dist", "frag1_rel",
                                                                      "frag2_rel")})
    return s


def output_paths(rid):
    s = {"eda": rxn_dir(D0_SMD, rid) / "eda.out"}
    s.update({c: rxn_dir(D0_LABEL_INPUTS, rid) / f"{c}.out" for c in ("frag1_dist", "frag2_dist", "frag1_rel",
                                                                      "frag2_rel")})
    return s


def check_d0(rid, lab, a_idx_s, H, PR):
    """X-1 .. X-4 + HF templates of one D0 reaction -> (record, {relpath: bytes})."""
    T = job_id(rid)
    rec = dict(job_id=T, rxn_id=int(rid), dataset="D0", x_version=X_VERSION, reasons=[], stop=[], keywords={})
    src = source_paths(rid)
    miss = [c for c, p in src.items() if not p.is_file()]
    if miss:
        rec["reasons"].append("missing:" + ",".join(miss))
        return rec, {}
    files = {c: src[c].read_bytes() for c in H.CALCS}
    rec["sources"] = {c: dict(path=str(src[c]), sha256=sha_bytes(files[c])) for c in H.CALCS}
    try:
        P = {c: H.parse_inp(files[c].decode()) for c in H.CALCS}
    except (ValueError, UnicodeDecodeError, IndexError) as e:
        rec["reasons"].append(f"parse:{e}")
        return rec, {}
    e = P["eda"]
    tag = {k: [(a[0], a[2], a[3], a[4]) for a in e["atoms"] if a[1] == k] for k in (1, 2)}
    rec.update(n_atoms=len(e["atoms"]), n_f1=len(tag[1]), n_f2=len(tag[2]),
               elements=";".join(sorted({a[0] for a in e["atoms"]})),
               charge=int(lab.get("charge1", 0) or 0) + int(lab.get("charge2", 0) or 0))
    if any(a[1] not in (1, 2) for a in e["atoms"]):
        rec["reasons"].append("eda_untagged_atom")
    # X-2: coordinates vs the Coley profile files and the d0_g2 labjobs copies
    prof, lj = D0_PROF / str(int(rid)), D0_LABJOBS / T
    try:
        ts_c = H.read_xyz(prof / lab["ts_file"])
        ts_j = H.read_xyz(lj / "ts.xyz")
        refs = {1: H.read_xyz(prof / lab["rel1_file"]), 2: H.read_xyz(prof / lab["rel2_file"])}
        ljr = {1: H.read_xyz(lj / "ref_dipole.xyz"), 2: H.read_xyz(lj / "ref_dipolarophile.xyz")}
    except (OSError, ValueError, IndexError, KeyError) as ex:
        rec["reasons"].append(f"x2_reference_unreadable:{type(ex).__name__}")
        return rec, {}
    eda_atoms = [(a[0], a[2], a[3], a[4]) for a in e["atoms"]]
    chk = {}
    chk["eda_eq_coley_ts"] = PR.same_atoms(eda_atoms, ts_c)
    chk["labjobs_ts_eq_coley_ts"] = PR.same_atoms(ts_j, ts_c)
    for k in (1, 2):
        chk[f"frag{k}_dist_eq_eda_tag{k}"] = PR.same_atoms([(a[0], a[2], a[3], a[4]) for a in P[f"frag{k}_dist"]["atoms"]],
                                                          tag[k])
        chk[f"frag{k}_rel_eq_coley_rel{k}"] = PR.same_atoms([(a[0], a[2], a[3], a[4]) for a in P[f"frag{k}_rel"]["atoms"]],
                                                           refs[k])
        chk[f"labjobs_ref{k}_eq_coley_rel{k}"] = PR.same_atoms(ljr[k], refs[k])
    for k, v in chk.items():
        if not v:
            rec["reasons"].append(f"x2:{k}")
    # X-3: tags = A_idx, sizes, roles, charges
    A = [i for i, a in enumerate(e["atoms"]) if a[1] == 1]
    chk["tags_eq_A_idx"] = A == a_list(a_idx_s)
    chk["sizes_eq_label"] = (len(A) == int(lab["n_f1"]) and len(e["atoms"]) - len(A) == int(lab["n_f2"])
                             and len(e["atoms"]) == int(lab["n_atoms"]))
    for k in ("tags_eq_A_idx", "sizes_eq_label"):
        if not chk[k]:
            rec["reasons"].append(f"x3:{k}")
    if not (lab.get("role1") == "dipole" and lab.get("role2") == "dipolarophile"
            and sorted(a[0] for a in tag[1]) == sorted(a[0] for a in refs[1])
            and sorted(a[0] for a in tag[2]) == sorted(a[0] for a in refs[2])):
        rec["stop"].append("frag1_not_dipole")
    try:
        q = {k: (int(e["frag"][f"FRAG{k}_C"]), int(e["frag"][f"FRAG{k}_M"])) for k in (1, 2)}
    except (KeyError, ValueError):
        q = None
        rec["reasons"].append("eda_frag_charge_mult_missing")
    if q:
        for c in H.CALCS[:4]:
            if (P[c]["charge"], P[c]["mult"]) != q[int(c[4])]:
                rec["reasons"].append(f"{c}_charge_mult_differ_from_eda")
        if e["charge"] != q[1][0] + q[2][0]:
            rec["reasons"].append("eda_total_charge")
        if (q[1][0], q[2][0]) != (int(lab["charge1"]), int(lab["charge2"])):
            rec["reasons"].append("x3:charges_differ_from_label")
    # X-1: SMD + the keyword inventory of prep.check_job
    for c in H.CALCS:
        if not H.smd_ok(P[c], "eda" if c == "eda" else "frag"):
            rec["reasons"].append(f"{c}_no_smd")
            rec["stop"].append("input_without_smd")
    for c in H.CALCS:
        p = P[c]
        kw = dict(bang=p["bang"], blocks=sorted(p["blocks"]), maxcore=p["blocks"].get("maxcore", [""])[0])
        if c == "eda":
            kw.update(FRAG1=p["frag"].get("FRAG1"), FRAG2=p["frag"].get("FRAG2"), pal=p["pal"])
            ok = (p["bang"] == PR.EDA_BANG and sorted(p["blocks"]) == PR.EDA_BLOCKS
                  and all(p["frag"].get(f, "").split() == PR.FRAG_STR for f in ("FRAG1", "FRAG2"))
                  and set(p["frag"]) == {"FRAG1", "FRAG2", "FRAG1_C", "FRAG1_M", "FRAG2_C", "FRAG2_M"})
        else:
            kw.update(cpcm=p["blocks"].get("cpcm"))
            ok = p["bang"] == PR.SA_BANG and sorted(p["blocks"]) == PR.SA_BLOCKS
        ok &= kw["maxcore"].split() == ["%maxcore", "3500"]
        rec["keywords"][c] = kw
        if not ok:
            rec["stop"].append(f"unexpected_keywords:{c}")
    # X-4: provenance of the label energies
    prov = {}
    for c, p in output_paths(rid).items():
        if not p.is_file():
            prov[c] = "missing"
            continue
        t = H.read_text(p)
        f = H.RE_FSPE.findall(t)
        want = lab.get(LABEL_E[c])
        prov[c] = ("ok" if H.terminated(t) and len(f) == 1 and want is not None and float(f[0]) == float(want)
                   else f"terminated {H.terminated(t)}, {len(f)} FSPE, {f[-1] if f else None} vs label {want}")
    rec["x4_provenance"] = prov
    for c, v in prov.items():
        if v != "ok":
            rec["reasons"].append(f"x4:{c}")
    rec["checks"] = chk
    # HF templates (the prep.check_job diff check)
    out = {f"orig/{c}.inp": files[c] for c in H.CALCS}
    for c in H.CALCS:
        kind = "eda" if c == "eda" else "frag"
        new, bad = hf_diff_bad(H, PR, files[c].decode(), kind, P[c])
        if bad:
            rec["reasons"].append(f"hf_diff_check:{c}")
        out[f"hf/{c}.inp"] = new.encode()
        rec[f"sha_orig_{c}"], rec[f"sha_hf_{c}"] = sha_bytes(files[c]), sha_bytes(new.encode())
    return rec, out


def finish(T, rec, out):
    """$GEO2B/x_d0/inputs/<T>/{orig,hf}/*.inp + check.json, written into <T>.tmp/ and renamed (= prep.finish)."""
    final, tmp = X_INPUTS / T, X_INPUTS / f"{T}.tmp"
    C.assert_path(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    for rel, data in out.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(data)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "check.json").write_text(json.dumps(rec, indent=1))
    if final.exists():
        shutil.rmtree(C.assert_path(final))
    os.replace(tmp, final)


def current(T, rid):
    """True when <T>/check.json exists for this code version and the same 5 source sha256 (kept, not rebuilt)."""
    f = X_INPUTS / T / "check.json"
    if not f.is_file():
        return False
    try:
        r = json.loads(f.read_text())
    except ValueError:
        return False
    src = source_paths(rid)
    want = {c: (C.sha256(p) if p.is_file() else None) for c, p in src.items()}
    return r.get("x_version") == X_VERSION and {c: (v or {}).get("sha256") for c, v in (r.get("sources") or {}).items()} \
        == want and all((X_INPUTS / T / d / f"{c}.inp").is_file() for d in ("orig", "hf") for c in src)


_W = {}


def _winit(labs, metas):
    _W["lab"], _W["meta"] = labs, metas
    _W["H"], _W["PR"] = hf_modules()


def _wone(rid):
    T = job_id(rid)
    if current(T, rid):
        return json.loads((X_INPUTS / T / "check.json").read_text()), "kept"
    rec, out = check_d0(rid, _W["lab"][rid], _W["meta"].get(rid, ""), _W["H"], _W["PR"])
    rec.update(written=now(), host=socket.gethostname())
    if out:
        finish(T, rec, out)
    else:                                                  # nothing to write but the record (no template folder)
        C.write_json(X_INPUTS / T / "check.json", rec)
    return rec, "built"


# ================================================================ combined root
def link(dirp, name, target, problems):
    """dirp/name -> target (created if missing; an existing different entry is a problem, never replaced)."""
    p = Path(dirp) / str(name)
    t = str(target)
    if p.is_symlink():
        if os.readlink(p) != t:
            problems.append(f"{p} -> {os.readlink(p)} (want {t})")
        return
    if p.exists():
        problems.append(f"{p} exists and is not a link")
        return
    C.assert_path(p)
    os.symlink(t, p)


def write_same(path, text, problems):
    """write path atomically unless it exists; an existing file with other bytes is a problem (not overwritten)."""
    p = Path(path)
    b = text.encode() if isinstance(text, str) else text
    if p.is_file():
        if p.read_bytes() != b:
            problems.append(f"{p} exists with other content")
        return
    C.write_atomic(p, b)


def build_comb(d0, d0x, rec):
    """the combined root (module docstring); returns (outputs {path: sha256}, digests, problems)."""
    probs = []
    rep = json.loads(D1_G2_INPUTS_REPORT.read_text())
    want = rep.get("outputs") or {}
    for p in (D1_LABELS, D1_META):
        if want.get(str(p)) != C.sha256(p) or rep.get("stop") is not False:
            C.stop(f"{p} sha256 != {D1_G2_INPUTS_REPORT} outputs (or report stop)")
    d1 = json.loads(D1_LABELS.read_text())
    d1ids = [int(d["rxn_id"]) for d in d1]
    d1jobs = {int(d["rxn_id"]): str(d["job_id"]) for d in d1}
    if len(d1) != 9266 or len(set(d1ids)) != len(d1ids) or min(d1ids) <= D1_RXN_MIN:
        C.stop(f"{D1_LABELS}: {len(d1)} rows, unique {len(set(d1ids))}, min rxn_id {min(d1ids)}")
    # labels: D1 rows verbatim + D0 rows verbatim (+ job_id, kind)
    rows = list(d1)
    for rid in sorted(d0):
        r = dict(d0[rid])
        if "job_id" in r or "kind" in r:
            C.stop(f"labels_all row {rid} already has job_id / kind")
        r["job_id"], r["kind"] = job_id(rid), "D0"
        rows.append(r)
    write_same(COMB_LABELS, json.dumps(rows), probs)
    # meta
    m1 = D1_META.read_text().splitlines()
    if m1[0].strip() != "rxn_id,A_idx" or len(m1) - 1 != len(d1):
        C.stop(f"{D1_META}: header {m1[0]!r}, {len(m1) - 1} rows")
    m0 = read_meta_csv(D0_META)
    miss = sorted(set(d0) - set(m0))
    if miss:
        C.stop(f"{D0_META}: no A_idx for D0 rows {miss[:10]}")
    meta_txt = "\n".join(m1) + "\n" + "".join(f"{rid},{m0[rid]}\n" for rid in sorted(d0))
    write_same(COMB_META, meta_txt, probs)
    # manifest: D1 X_manifest rows of the targets (all x_fail False) + the D0 Phase X rows
    with open(D1_X_MANIFEST, newline="") as f:
        X1 = {r["job_id"]: r for r in csv.DictReader(f)}
    bad1 = [T for T in d1jobs.values() if T not in X1 or str(X1[T]["x_fail"]) != "False"]
    if bad1:
        C.stop(f"D1 targets not X-ok in {D1_X_MANIFEST}: {bad1[:10]} ({len(bad1)})")
    out = [MAN_COLS]
    for rid, T in sorted(d1jobs.items(), key=lambda x: x[1]):
        x = X1[T]
        out.append([T, rid, "D1", x["n_atoms"], x["n_f1"], x["n_f2"], 0, x["elements"], "False", ""])
    for rid in sorted(d0):
        x = d0x[rid]
        out.append([job_id(rid), rid, "D0", x.get("n_atoms", ""), x.get("n_f1", ""), x.get("n_f2", ""),
                    x.get("charge", ""), x.get("elements", ""), str(bool(x["reasons"])), ";".join(x["reasons"])])
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out)
    write_same(COMB_MANIFEST, buf.getvalue(), probs)
    # symlink folders
    for d in (COMB_PROF, COMB_JOBS, COMB_TEMPLATES):
        C.assert_path(d).mkdir(parents=True, exist_ok=True)
    for rid, T in d1jobs.items():
        link(COMB_PROF, rid, os.readlink(D1_PROF / str(rid)), probs)
        link(COMB_JOBS, T, D1_JOBS / T, probs)
        link(COMB_TEMPLATES, T, D1_TEMPLATES / T, probs)
    for rid in sorted(d0):
        T = job_id(rid)
        link(COMB_PROF, rid, D0_PROF / str(rid), probs)
        link(COMB_JOBS, T, D0_LABJOBS / T, probs)
        if not d0x[rid]["reasons"]:
            link(COMB_TEMPLATES, T, X_INPUTS / T, probs)
    n_expect = dict(prof=len(d1jobs) + len(d0), jobs=len(d1jobs) + len(d0),
                    hf_templates=len(d1jobs) + sum(1 for r in d0 if not d0x[r]["reasons"]))
    dig = {}
    for nm in LINK_DIRS:
        s, n = listing_digest(COMB / nm)
        dig[nm] = dict(digest=s, n=n, n_expected=n_expect[nm])
        if n != n_expect[nm]:
            probs.append(f"{COMB / nm}: {n} entries != {n_expect[nm]} expected (stale entries?)")
    dangling = [str(COMB / nm / x) for nm in LINK_DIRS for x in sorted(os.listdir(COMB / nm))
                if not (COMB / nm / x).exists()]
    if dangling:
        probs.append(f"{len(dangling)} links point to nothing: {dangling[:5]}")
    outs = {str(p): C.sha256(p) for p in (COMB_LABELS, COMB_META, COMB_MANIFEST) if p.is_file()}
    rec["comb"] = dict(root=str(COMB), n_d1=len(d1jobs), n_d0=len(d0), n_dangling=len(dangling))
    return outs, dig, probs


def comb_state():
    """(report or {}, problems [(rc, why)]): results/X_d0d1_report.json complete, stop false, its outputs sha256 and
    symlink listing digests = the current files (feat_geo.inputs_check)."""
    if not X_REPORT.is_file():
        return {}, [(2, f"{X_REPORT} missing (run d0d1.py x)")]
    r = json.loads(X_REPORT.read_text())
    pr = []
    if r.get("stop") is not False or not r.get("complete"):
        pr.append((3 if r.get("stop") else 2, f"{X_REPORT.name}: stop {r.get('stop')!r}, complete {r.get('complete')!r}"))
        return r, pr
    for p, s in (r.get("outputs") or {}).items():
        if not Path(p).is_file() or C.sha256(p) != s:
            pr.append((3, f"{p} changed since {X_REPORT.name}"))
    for nm, d in (r.get("link_digests") or {}).items():
        if listing_digest(COMB / nm)[0] != (d or {}).get("digest"):
            pr.append((3, f"{COMB / nm} links changed since {X_REPORT.name}"))
    if set((r.get("outputs") or {})) != {str(COMB_LABELS), str(COMB_META), str(COMB_MANIFEST)} or \
            set(r.get("link_digests") or {}) != set(LINK_DIRS):
        pr.append((3, f"{X_REPORT.name}: outputs / link digests incomplete"))
    return r, pr


def load_comb_labels():
    return json.loads(COMB_LABELS.read_text())


def read_comb_meta():
    """{rxn_id: sorted A_idx} of the combined meta csv (= g0f0.read_meta on the combined file)."""
    with open(COMB_META, newline="") as f:
        return {int(r["rxn_id"]): sorted(int(x) for x in str(r["A_idx"]).split()) for r in csv.DictReader(f)}


def read_comb_manifest():
    with open(COMB_MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


D0_PAIRS = GEO1B / "d0" / "D0_pairs.csv"


def d0_pairs_state():
    """(D0_pairs_report.json, problems [(rc, why)]) for the files the P report reads: stop false and the sha256 of
    d0_folds.csv and D0_pairs.csv recorded under 'outputs' (keyed by absolute path; matched by file name, the
    geo_common.load_d0_folds rule). g2_source != 'g2c' (the preliminary build before the G2c selection) = exit 2."""
    if not D0_PAIRS_REPORT.is_file():
        return {}, [(2, f"{D0_PAIRS_REPORT} missing")]
    rep = json.loads(D0_PAIRS_REPORT.read_text())
    pr = []
    if rep.get("stop") is not False:
        pr.append((3, f"{D0_PAIRS_REPORT.name}: stop {rep.get('stop')!r}"))
    for p in (D0_FOLDS, D0_PAIRS):
        outs = [v for k, v in (rep.get("outputs") or {}).items() if Path(k).name == p.name]
        if len(outs) != 1 or not p.is_file() or (outs[0] or {}).get("sha256") != C.sha256(p):
            pr.append((3, f"{p} sha256 differs from {D0_PAIRS_REPORT.name} outputs"))
    if rep.get("g2_source") != "g2c":
        pr.append((2, f"{D0_PAIRS_REPORT.name}: g2_source {rep.get('g2_source')!r} (preliminary; rerun d0_pairs.py build "
                      f"after select_g2c.py build)"))
    return rep, pr


def d0_fold_map():
    """{D0 rxn_id: fold} of d0_folds.csv (overlaps keep fold -9)."""
    with open(D0_FOLDS, newline="") as f:
        return {int(r["rxn_id"]): int(r["fold"]) for r in csv.DictReader(f)}


def read_d0_pairs():
    with open(D0_PAIRS, newline="") as f:
        return list(csv.DictReader(f))


# ================================================================ commands
def cmd_x(n, force=False):
    """while running: stop None + running true (comb_state -> exit 2, not a STOP); a C.stop / error inside writes
    stop true with the reason before the exit."""
    rec = dict(step="x", spec="D0D1 design step 6 (Phase X + combined root)", version=X_VERSION, started=now(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), decisions=DECISIONS,
               complete=False, stop=None, running=True, stop_reasons=["running"])
    C.write_json(X_REPORT, rec)
    try:
        return _cmd_x(n, force, rec)
    except BaseException as e:
        if rec.get("running"):
            why = f"exit {e.code} inside x (STOP line on stderr)" if isinstance(e, SystemExit) else \
                f"unexpected error {type(e).__name__}: {e}"
            rec.update(stop=True, running=False, complete=False, finished=now(), stop_reasons=[why])
            C.write_json(X_REPORT, rec)
        raise


def _cmd_x(n, force, rec):
    t0 = time.time()
    if force and COMB.exists():
        dst = COMB.with_name(f"comb.prev_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(C.assert_path(COMB), C.assert_path(dst))
        rec["comb_moved_to"] = str(dst)
    d0, folds = load_d0_rows(rec)
    m0 = read_meta_csv(D0_META)
    from multiprocessing import Pool
    C.assert_path(X_INPUTS).mkdir(parents=True, exist_ok=True)
    W = max(1, int(n))
    with Pool(W, initializer=_winit, initargs=(d0, m0)) as pool:
        res = pool.map(_wone, sorted(d0), chunksize=16)
    d0x = {int(r["rxn_id"]): r for r, _ in res}
    how = Counter(h for _, h in res)
    rows = [{k: r.get(k) for k in ("job_id", "rxn_id", "n_atoms", "n_f1", "n_f2", "charge", "elements")}
            | dict(reasons=";".join(r["reasons"]), stop=";".join(r["stop"]), x_fail=bool(r["reasons"]),
                   **{f"sha_hf_{c}": r.get(f"sha_hf_{c}") for c in ("eda", "frag1_dist", "frag2_dist", "frag1_rel",
                                                                     "frag2_rel")})
            for r in (d0x[k] for k in sorted(d0x))]
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    C.write_atomic(X_MANIFEST_D0, buf.getvalue())
    nf = sum(1 for r in rows if r["x_fail"])
    stops = sorted({s for r in d0x.values() for s in r["stop"]})
    if nf / len(rows) > X_FAIL_MAX:
        stops.append(f"x_fail {nf}/{len(rows)} > {X_FAIL_MAX:.1%} (D1-HF X rule)")
    kw = {}
    for r in d0x.values():
        for c, k in (r.get("keywords") or {}).items():
            kw.setdefault(c, Counter())[json.dumps(k, sort_keys=True)] += 1
    rec.update(n_d0=len(rows), n_x_fail=nf, x_fail_rate=nf / len(rows), x_built=dict(how),
               x_fail_reasons=Counter(x for r in d0x.values() for x in r["reasons"]).most_common(),
               x_fail_rxn={str(k): r["reasons"] for k, r in sorted(d0x.items()) if r["reasons"]},
               keywords={c: [dict(n=v, **json.loads(k)) for k, v in cnt.most_common()] for c, cnt in kw.items()},
               charged=dict(Counter((int(r.get("charge") or 0)) for r in d0x.values())),
               n_atoms=dict(min=min(int(r["n_atoms"]) for r in d0x.values() if r.get("n_atoms")),
                            max=max(int(r["n_atoms"]) for r in d0x.values() if r.get("n_atoms"))),
               x_manifest=str(X_MANIFEST_D0), x_manifest_sha256=C.sha256(X_MANIFEST_D0))
    if stops:
        rec.update(stop=True, running=False, stop_reasons=stops, complete=False, finished=now(),
                   wall_s=round(time.time() - t0, 1))
        C.write_json(X_REPORT, rec)
        print(json.dumps({k: rec[k] for k in ("n_d0", "n_x_fail", "stop_reasons")}, indent=1), flush=True)
        return 3
    outs, dig, probs = build_comb(d0, d0x, rec)
    inputs = {str(p): (C.sha256(p) if Path(p).is_file() else None)
              for p in (D0_LABELS, D0_META, D0_FOLDS, D0_PAIRS_REPORT, D1_LABELS, D1_META, D1_G2_INPUTS_REPORT,
                        D1_X_MANIFEST, THIS, HERE / "g3c.py", C.HF_DIR / "prep.py", C.HF_DIR / "hf_common.py")}
    rec.update(outputs=outs, link_digests=dig, inputs=inputs, problems=probs, stop=bool(probs), running=False,
               stop_reasons=probs[:50], complete=not probs, finished=now(), wall_s=round(time.time() - t0, 1))
    C.write_json(X_REPORT, rec)
    print(json.dumps({k: rec.get(k) for k in ("n_d0", "n_x_fail", "x_built", "charged", "link_digests", "stop",
                                              "stop_reasons", "wall_s")}, indent=1, default=str), flush=True)
    return 3 if probs else 0


def s1_select(man):
    """D0 S-1 reactions: X-ok D0 rows by (n_atoms, rxn_id): the 2 smallest neutral + the smallest charged one."""
    d0 = sorted((r for r in man if r["dataset"] == "D0" and r["x_fail"] == "False"),
                key=lambda r: (int(r["n_atoms"]), int(r["rxn_id"])))
    neu = [r for r in d0 if int(r["charge"]) == 0][:S1_N_NEUTRAL]
    chg = [r for r in d0 if int(r["charge"]) != 0][:S1_N_CHARGED]
    return neu + chg


def s1_env():
    return dict(os.environ, D1HF_SCRATCH=str(X_DIR), D1HF_RES=str(X_DIR / "results"), PYTHONDONTWRITEBYTECODE="1")


def child(argv, cwd, env):
    print(f"=== [{now()}] {' '.join(map(str, argv))}", flush=True)
    r = subprocess.run([sys.executable] + [str(a) for a in argv], cwd=str(cwd), env=env)
    print(f"=== [{now()}] exit {r.returncode}", flush=True)
    return r.returncode


def cmd_s1(n):
    """as cmd_x: running = stop false / pass false (need_s1 -> exit 2); a C.stop / error inside -> stop true."""
    rep = dict(step="s1", spec="D0D1 design step 6 (D0 S-1)", started=now(), decisions={"X-7 S-1": DECISIONS["X-7 S-1"]},
               slurm_job=os.environ.get("SLURM_JOB_ID"), complete=False, stop=False, running=True, **{"pass": False})
    try:
        rc = _cmd_s1(n, rep)
    except BaseException as e:
        why = f"exit {e.code} inside s1 (STOP line on stderr)" if isinstance(e, SystemExit) else \
            f"unexpected error {type(e).__name__}: {e}"
        rep.update(stop=True, running=False, complete=False, finished=now(), problems=[why], **{"pass": False})
        C.write_json(S1_JSON, rep)
        raise
    if rep.get("running"):
        rep["running"] = False
        C.write_json(S1_JSON, rep)
    return rc


def _cmd_s1(n, rep):
    t0 = time.time()
    st, pr = comb_state()
    if pr:
        rep.update(fail=[w for _, w in pr])
        C.write_json(S1_JSON, rep)
        print(f"STOP / NOT READY: {pr}", file=sys.stderr)
        return max(c for c, _ in pr)
    H, PR = hf_modules()
    sel = s1_select(read_comb_manifest())
    jobs = [r["job_id"] for r in sel]
    if len(jobs) != S1_N_NEUTRAL + S1_N_CHARGED:
        C.stop(f"S-1 selection {jobs}")
    for T in jobs:
        for c in H.CALCS:
            src = X_INPUTS / T / "orig" / f"{c}.inp"
            dst = X_INPUTS / T / "orig1" / f"{c}.inp"
            txt = PR.pal1_text(src.read_text())
            if not (dst.is_file() and dst.read_text() == txt):
                C.write_atomic(dst, txt)
    lst = C.write_atomic(X_DIR / "S1_list.csv", "job_id\n" + "".join(f"{T}\n" for T in jobs))
    env = s1_env()
    rc_run = child([C.HF_DIR / "run.py", "run", lst, "dft", "orig1", max(1, min(len(jobs), int(n)))], C.HF_DIR, env)
    pj = X_DIR / "S1_parse.json"
    rc_p = child([THIS, "_s1parse", lst, pj], C.HF_DIR, env)
    if rc_p != 0:
        rep.update(fail=f"_s1parse exit {rc_p}")
        C.write_json(S1_JSON, rep)
        return 3
    recs = json.loads(pj.read_text())["records"]
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import build as HB
    keys = dict(HB.S1_KEYS, **S1_EXTRA)
    lab = {int(d["rxn_id"]): d for d in json.loads(D0_LABELS.read_text())}
    rows, bad, open_ = [], [], []
    for r in sel:
        T, rid = r["job_id"], int(r["rxn_id"])
        x = recs[T]
        if x.get("status") in ("not_done",):
            open_.append(T)
        for k, lk in keys.items():
            a, b = x.get(k if k != "c_ghost" else "bsse_shift_kcal"), lab[rid][lk]
            try:
                d = float(a) - float(b)
            except (TypeError, ValueError):
                d = float("inf")
            ok = abs(d) <= HB.S1_TOL
            rows.append(dict(job_id=T, rxn_id=rid, charge=int(r["charge"]), n_atoms=int(r["n_atoms"]), value=k,
                             run=a, label=b, diff=d, ok=bool(ok)))
            if not ok:
                bad.append(f"{T} {k}: {a} vs {b}")
        rows.append(dict(job_id=T, rxn_id=rid, charge=int(r["charge"]), n_atoms=int(r["n_atoms"]), value="status",
                         run=x.get("status"), label="ok", diff=None, ok=x.get("status") == "ok"))
        if x.get("status") != "ok":
            bad.append(f"{T}: status {x.get('status')} {x.get('reason')}")
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    C.write_atomic(S1_CSV, buf.getvalue())
    good = not bad and not open_ and rc_run == 0
    rep.update(jobs=jobs, selection=sel, run_exit=rc_run, tol_kcal=HB.S1_TOL, keys=keys,
               max_abs_diff_kcal=max([abs(r["diff"]) for r in rows if isinstance(r["diff"], float)] or [None]),
               problems=bad, open=open_, coreh={T: recs[T].get("coreh_total") for T in jobs}, compare_csv=str(S1_CSV),
               complete=not open_ and rc_run == 0, stop=bool(bad) and not open_, finished=now(),
               wall_s=round(time.time() - t0, 1), **{"pass": bool(good)})
    C.write_json(S1_JSON, rep)
    print(json.dumps({k: rep[k] for k in ("jobs", "max_abs_diff_kcal", "problems", "open", "pass")}, indent=1,
                     default=str), flush=True)
    if open_ or rc_run == 2:
        return 2
    return 0 if good else 3


def _s1parse(lst, out):
    """child of s1 (D1HF_SCRATCH = $GEO2B/x_d0): build.parse_rxn(T, 'dft', 'orig1') of the listed reactions."""
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import build as HB
    if Path(HB.H.HF) != X_DIR:
        C.stop(f"_s1parse: hf_common.HF = {HB.H.HF} (need {X_DIR})")
    with open(lst, newline="") as f:
        ids = [r["job_id"] for r in csv.DictReader(f)]
    C.write_json(out, dict(hf=str(HB.H.HF), records={T: HB.parse_rxn(T, "dft", "orig1") for T in ids}))
    return 0


def main(a):
    if a[:1] == ["x"] and len(a) <= 3:
        rest = [x for x in a[1:] if x != "--force"]
        return cmd_x(int(rest[0]) if rest else cpus(), force="--force" in a[1:])
    if a[:1] == ["s1"] and len(a) <= 2:
        return cmd_s1(int(a[1]) if len(a) == 2 else cpus())
    if a[:1] == ["_s1parse"] and len(a) == 3:
        return _s1parse(a[1], a[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
