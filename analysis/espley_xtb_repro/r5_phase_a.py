#!/usr/bin/env python3
"""r5_phase_a.py — REV5 Phase A helpers (docs/specs/REV5_FEATURES_FIGURES.md A-1 .. A-4), run by r5_phaseA.sh.

  python r5_phase_a.py delete        A-3: record, then delete, the scratch paths listed in r5_phaseA_discard.txt
  python r5_phase_a.py regen-check   A-1: the regenerated results_rev4 tables vs their rev 4 versions (git REF_COMMIT)
  python r5_phase_a.py grep-check    A-4: the spec grep over *.py / *.sh / *.md of this directory

delete -> results_rev5/A_deleted.txt: one TSV row per file (path, size in bytes, sha256), written atomically BEFORE
  anything is deleted; the footer line "# status: deleted ..." is added (atomic rewrite) once every target is cleared.
  Two target kinds: 'delete' (a file or directory, removed whole; its path must match the discard pattern) and
  'delete_arm_json' (a kept directory of rev 4 per-job ML JSONs; only its *.json whose "our_geom" field matches the
  discard pattern are recorded and removed, see arm_jsons). First run (no manifest): every 'delete' target must
  exist and every 'delete_arm_json' directory must hold >= 1 such JSON (exit 2 otherwise). Rerun (manifest exists):
  every file still selected must be in the manifest with the same size and sha256 (exit 3 otherwise), then the rest is
  deleted — so a job clipped mid-deletion resumes; at the end no recorded path may exist (exit 3). Safety: a target
  must be an absolute non-symlink path under SAFE_ROOT, targets must not repeat or nest (exit 3 otherwise).
regen-check -> results_rev5/A_regen_check.json: every regenerated results_rev4 file vs its REF_COMMIT version, both
  ways on the key. Each regenerated row must exist in the REF_COMMIT version with the same values (tolerances below).
  The REF_COMMIT rows split into discarded (a key or MARK_COLS cell matches the discard pattern) and kept: every kept
  row must be regenerated, and no regenerated row may carry the key of a discarded one. The regenerated files must not
  contain the discard pattern. The ablation rows JSONs must carry the rev 4 row sets (rows_sha256, n). Exit 3 on any
  failure.
grep-check -> results_rev5/A_grep_check.txt: every hit, allowed (allow_file, or allow_text with nothing else on the
  line matching) or not; each 'moved' file (moved out of the scanned directory) is verified unedited and disclosed as
  a '# note:' header line with its number of matching lines. Exit 3 if any hit is not allowed or a moved-file check
  fails, grep error -> exit 2.
The paths, the pattern, the allowed exceptions and the moved files come from r5_phaseA_discard.txt (a data file, see
its header).
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rev5_common import REPO, RES4, RES5, sha256, write_atomic, write_json  # noqa: E402

DATA = HERE / "r5_phaseA_discard.txt"
MANIFEST = RES5 / "A_deleted.txt"
SAFE_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee")
REF_COMMIT = "740ba895"                        # rev 4 results, the last commit holding the discarded arm
TOL_EXACT = 1e-9                               # tables recomputed from unchanged model outputs
TOL_ABL = 1e-3                                 # ablation MAE refit from scratch (same rows / seeds / grids)
N_SCORED = 3327                                # Espley comparison rows (labelled AND G1-usable)
STATUS_PREFIX = "# status: deleted"
MARK_COLS = ("our_geom", "geom", "geom_note", "side", "selection")   # columns naming a row's geometry / side


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def job():
    return os.environ.get("SLURM_JOB_ID", "local")


def load_data():
    """{'delete': [...], 'delete_arm_json': [...], 'pattern': str, 'allow_file': [...], 'allow_text': [...],
    'moved': [(old repo path, new repo path), ...]} from DATA."""
    if not DATA.is_file():
        sys.exit(f"GATE: {DATA} missing")
    d = {"delete": [], "delete_arm_json": [], "pattern": [], "allow_file": [], "allow_text": [], "moved": []}
    for n, line in enumerate(DATA.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, val = line.partition("\t")
        if not sep or key not in d or not val.strip():
            sys.exit(f"GATE: {DATA}:{n}: expected '<key><TAB><value>' with key in {sorted(d)}, got {line!r}")
        d[key].append(val.rstrip("\n"))
    if len(d["pattern"]) != 1 or not d["delete"]:
        sys.exit(f"GATE: {DATA}: need exactly one pattern and >= 1 delete line, got {len(d['pattern'])} / {len(d['delete'])}")
    d["pattern"] = d["pattern"][0]
    re.compile(d["pattern"])
    moved = []
    for v in d["moved"]:
        old, sep, new = v.partition("\t")
        if not sep or not old.strip() or not new.strip() or "\t" in new or old.startswith("/") or new.startswith("/"):
            sys.exit(f"GATE: {DATA}: a moved line must be 'moved<TAB><old repo path><TAB><new repo path>' (relative "
                     f"paths), got {v!r}")
        moved.append((old, new))
    d["moved"] = moved
    return d


# ---------------------------------------------------------------------------------------------------- A-3 delete
def check_target(p, pat, match=True):
    """A 'delete' target (match=True: its path must match the discard pattern) or a 'delete_arm_json' directory
    (match=False: the directory is kept and its files are selected by content, arm_jsons)."""
    p = Path(p)
    bad = []
    if not p.is_absolute():
        bad.append("not absolute")
    if p.is_symlink():
        bad.append("is a symlink")
    try:
        p.resolve().relative_to(SAFE_ROOT.resolve())
    except ValueError:
        bad.append(f"not under {SAFE_ROOT}")
    if len(p.parts) < 5:
        bad.append("too shallow")
    if match and not re.search(pat, str(p), re.I):
        bad.append("does not match the discard pattern")
    if bad:
        print(f"GATE: refusing target {p}: {', '.join(bad)}")
        sys.exit(3)
    return p


def arm_jsons(d, pat):
    """The *.json files directly in the kept directory d whose "our_geom" field matches the discard pattern (the
    discarded arms' per-job ML outputs of the rev 4 espley_fairness_ablation.py), sorted; [] if d is not a directory.
    Every *.json in d must be a regular file holding a dict with string "arm" and "our_geom" fields and named
    '<arm>__*.json', else GATE exit 3 (nothing is guessed)."""
    if not d.is_dir():
        return []
    rx = re.compile(pat, re.I)
    out = []
    for f in sorted(d.glob("*.json")):
        why, arm, geo = None, None, None
        if f.is_symlink() or not f.is_file():
            why = "not a regular file"
        else:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                j, why = None, f"unreadable JSON ({type(e).__name__}: {e})"
            if isinstance(j, dict):
                arm, geo = j.get("arm"), j.get("our_geom")
            if why is None and (not isinstance(arm, str) or not isinstance(geo, str) or not f.name.startswith(arm + "__")):
                why = f"expected a dict with string 'arm' / 'our_geom' and the name '<arm>__*.json', got arm={arm!r} our_geom={geo!r}"
        if why:
            print(f"GATE: {f}: {why} (nothing deleted)")
            sys.exit(3)
        if rx.search(geo):
            out.append(f)
    return out


def files_under(p):
    """Every file (not directory) at / under p, sorted; symlinks inside are recorded, never followed."""
    if not p.exists() and not p.is_symlink():
        return []
    if p.is_file() or p.is_symlink():
        return [p]
    out = []
    for root, dirs, files in os.walk(p, followlinks=False):
        for f in files:
            out.append(Path(root) / f)
        out.extend(Path(root) / x for x in dirs if (Path(root) / x).is_symlink())
    return sorted(out)


def record(f):
    if f.is_symlink():
        return dict(path=str(f), size=0, sha256=f"symlink->{os.readlink(f)}")
    return dict(path=str(f), size=f.stat().st_size, sha256=sha256(f))


def read_manifest():
    """(rows by path, status line or None, full text)."""
    text = MANIFEST.read_text()
    rows, status = {}, None
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith(STATUS_PREFIX):
            status = line
        if not line or line.startswith("#") or line.startswith("path\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            sys.exit(f"GATE: {MANIFEST}:{n}: expected path<TAB>size<TAB>sha256, got {line!r}")
        rows[parts[0]] = dict(path=parts[0], size=int(parts[1]), sha256=parts[2])
    return rows, status, text


def write_manifest(targets, recs):
    head = ["# REV5 Phase A-3: scratch files of the discarded geometry arm, recorded before deletion",
            f"# recorded {now()} host {socket.gethostname()} job {job()} by r5_phase_a.py delete; list {DATA.name}"]
    for kind, t in targets:
        sub = [r for r in recs if r["path"] == str(t) or r["path"].startswith(str(t) + "/")]
        what = "" if kind == "path" else " (kept directory; only its *.json whose our_geom matches the discard pattern)"
        head.append(f"# target {t}{what}: {len(sub)} files, {sum(r['size'] for r in sub)} bytes")
    body = ["path\tsize_bytes\tsha256"] + [f"{r['path']}\t{r['size']}\t{r['sha256']}" for r in recs]
    write_atomic(MANIFEST, "\n".join(head + body) + "\n")


def delete():
    D = load_data()
    pat = D["pattern"]
    targets = ([("path", check_target(t, pat)) for t in D["delete"]]
               + [("json", check_target(t, pat, match=False)) for t in D["delete_arm_json"]])
    ps = [str(t) for _, t in targets]
    nested = [(a, b) for a in ps for b in ps if a != b and b.startswith(a + "/")]
    if len(set(ps)) != len(ps) or nested:
        print(f"GATE: delete targets repeat or nest (nothing recorded, nothing deleted): {nested[:3] or ps}")
        sys.exit(3)

    def selected(kind, t):
        return arm_jsons(t, pat) if kind == "json" else files_under(t)

    if not MANIFEST.exists():
        missing = [f"{t} (no such path)" for kind, t in targets if kind == "path" and not t.exists()]
        missing += [f"{t} (not a directory, or no *.json with a matching our_geom in it)" for kind, t in targets
                    if kind == "json" and not arm_jsons(t, pat)]
        if missing:
            print(f"GATE: no manifest yet and targets missing (nothing recorded, nothing deleted): {missing}")
            sys.exit(2)
        recs = [record(f) for kind, t in targets for f in selected(kind, t)]
        write_manifest(targets, recs)
        print(f"recorded {len(recs)} files, {sum(r['size'] for r in recs)} bytes -> {MANIFEST}")
    rows, status, text = read_manifest()
    # every file still selected must be exactly what was recorded
    off = []
    for kind, t in targets:
        for f in selected(kind, t):
            r, m = record(f), rows.get(str(f))
            if m is None or (m["size"], m["sha256"]) != (r["size"], r["sha256"]):
                off.append(dict(path=str(f), manifest=m, now=r))
    if off:
        print(f"GATE: {len(off)} files differ from / are missing in {MANIFEST} (nothing deleted): {off[:5]}")
        sys.exit(3)
    n_del = 0
    for kind, t in targets:
        if kind == "json":
            fs = selected(kind, t)
            for f in fs:
                f.unlink()
            if fs:
                n_del += 1
                print(f"deleted {len(fs)} *.json with a matching our_geom in {t} (directory and other files kept)")
        elif t.exists() or t.is_symlink():
            if t.is_dir() and not t.is_symlink():
                shutil.rmtree(t)
            else:
                t.unlink()
            n_del += 1
            print(f"deleted {t}")
    left = [str(t) for kind, t in targets if (selected(kind, t) if kind == "json" else (t.exists() or t.is_symlink()))]
    left += [p for p in rows if Path(p).exists() or Path(p).is_symlink()]
    if left:
        print(f"GATE: still present after deletion: {left[:10]}")
        sys.exit(3)
    if status is None:                                 # footer once; the recorded lines are kept byte for byte
        write_atomic(MANIFEST, text.rstrip("\n") + "\n"
                     + f"{STATUS_PREFIX} {now()} job {job()}: all {len(targets)} targets cleared, "
                       f"{len(rows)} files ({sum(r['size'] for r in rows.values())} bytes)\n")
    print(f"A-3 done: {n_del} targets cleared in this run, none left; {len(rows)} files recorded in {MANIFEST}")


# ---------------------------------------------------------------------------------------------------- A-1 regen check
def git_show(name):
    rel = (RES4 / name).relative_to(REPO)
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{REF_COMMIT}:{rel}"], capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        sys.exit(f"GATE: git show {REF_COMMIT}:{rel} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def cmp_csv(name, key, *, pat, exact=(), approx=(), tol=TOL_EXACT, soft=()):
    """Regenerated file vs its REF_COMMIT version, both ways on the key.
    Completeness: the REF_COMMIT rows split into discarded (a key or MARK_COLS cell matches the discard pattern pat)
    and kept; every kept row must be regenerated and no regenerated row may carry the key of a discarded one.
    Values: each regenerated row must exist in the REF_COMMIT file (same key) with equal values.
    exact: strings / counts (== after NaN->''), approx: floats within tol, soft: reported, not gated."""
    new = pd.read_csv(RES4 / name)
    old = pd.read_csv(io.StringIO(git_show(name)))
    res = dict(file=name, key=list(key), n_new=len(new), n_ref=len(old), fails=[], soft_diffs=[])
    lacking = [c for c in list(key) + list(exact) + list(approx) + list(soft) if c not in new or c not in old]
    if lacking:
        res["fails"].append(f"columns missing in new or {REF_COMMIT}: {lacking}")
        return res
    if not len(new):
        res["fails"].append("regenerated file is empty")
        return res
    for df, w in ((new, "new"), (old, REF_COMMIT)):
        dup = int(df.duplicated(list(key)).sum())
        if dup:
            res["fails"].append(f"{dup} duplicate keys in {w}")
    if res["fails"]:
        return res
    mark = [c for c in dict.fromkeys(list(key) + list(MARK_COLS)) if c in old]
    disc = old[mark].astype(str).apply(lambda s: s.str.contains(pat, case=False, regex=True)).any(axis=1)
    res.update(mark_cols=mark, n_ref_kept=int((~disc).sum()), n_ref_discarded=int(disc.sum()))
    lost = old.loc[~disc, list(key)].merge(new[list(key)], on=list(key), how="left", indicator=True)
    lost = lost[lost["_merge"] == "left_only"]
    if len(lost):
        res["fails"].append(f"{len(lost)} kept {REF_COMMIT} rows missing from the regenerated file: "
                            f"{lost[list(key)].head(5).to_dict('records')}")
    back = new[list(key)].merge(old.loc[disc, list(key)], on=list(key), how="inner")
    if len(back):
        res["fails"].append(f"{len(back)} regenerated rows carry the key of a discarded {REF_COMMIT} row: "
                            f"{back.head(5).to_dict('records')}")
    m = new.merge(old, on=list(key), how="left", suffixes=("", "__ref"), indicator=True)
    miss = m[m["_merge"] != "both"]
    if len(miss):
        res["fails"].append(f"{len(miss)} regenerated rows absent from {REF_COMMIT}: "
                            f"{miss[list(key)].head(5).to_dict('records')}")
    both = m[m["_merge"] == "both"]
    maxdiff = {}
    for c in approx:
        a, b = both[c].to_numpy(float), both[c + "__ref"].to_numpy(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        maxdiff[c] = float(d.max()) if len(d) else 0.0
        if maxdiff[c] > tol:
            k = int(np.argmax(d))
            res["fails"].append(f"{c}: max |new - ref| {maxdiff[c]:.3g} > {tol} at "
                                f"{both.iloc[k][list(key)].to_dict()}")
    for c, bucket in [(c, "fails") for c in exact] + [(c, "soft_diffs") for c in soft]:
        a = both[c].astype(object).where(both[c].notna(), "").astype(str)
        b = both[c + "__ref"].astype(object).where(both[c + "__ref"].notna(), "").astype(str)
        ne = both[(a != b).to_numpy()]
        if len(ne):
            res[bucket].append(f"{c}: {len(ne)} rows differ, e.g. {ne.iloc[0][list(key)].to_dict()}: "
                               f"{ne.iloc[0][c]!r} vs {ne.iloc[0][c + '__ref']!r}")
    res["max_abs_diff"] = maxdiff
    return res


def cmp_rows_json(name):
    new = json.loads((RES4 / name).read_text())
    old = json.loads(git_show(name))
    res = dict(file=name, fails=[], n_common=new.get("n_common"), rows_sha256=new.get("rows_sha256"))
    for k in ("n_espley", "n_common", "rows_sha256", "n_role_swapped"):
        if new.get(k) != old.get(k):
            res["fails"].append(f"{k}: {new.get(k)!r} vs {REF_COMMIT} {old.get(k)!r}")
    if sorted(new.get("dropped", {})) != sorted(old.get("dropped", {})):
        res["fails"].append("dropped rxn ids differ")
    own_new, own_old = new.get("own", {}), old.get("own", {})
    for k, v in own_new.items():
        if k not in own_old or v.get("n") != own_old[k].get("n"):
            res["fails"].append(f"own[{k}].n {v.get('n')} vs {REF_COMMIT} {own_old.get(k, {}).get('n')}")
    return res


def cmp_checks_json():
    name = "espley_compare_checks.json"
    new = json.loads((RES4 / name).read_text())
    old = json.loads(git_show(name))
    ni, oi = new.get("intersection", {}), old.get("intersection", {})
    res = dict(file=name, fails=[], n_scored=ni.get("n_scored"))
    if ni.get("n_scored") != N_SCORED or oi.get("n_scored") != N_SCORED:
        res["fails"].append(f"n_scored new {ni.get('n_scored')}, {REF_COMMIT} {oi.get('n_scored')}, expected {N_SCORED}")
    dn = sorted(int(r["rxn_id"]) for r in ni.get("dropped", []))
    do = sorted(int(r["rxn_id"]) for r in oi.get("dropped", []))
    if dn != do:
        res["fails"].append(f"dropped rxn ids differ: only new {sorted(set(dn) - set(do))[:10]}, "
                            f"only {REF_COMMIT} {sorted(set(do) - set(dn))[:10]}")
    for k in ("label_agreement", "split_replicated", "d1d2_index_vs_role"):
        if new.get(k) != old.get(k):
            res["fails"].append(f"{k} differs from {REF_COMMIT}")
    return res


def scan_pattern(names, pat):
    hits = []
    rx = re.compile(pat, re.I)
    for n in names:
        for i, line in enumerate((RES4 / n).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{n}:{i}: {line[:160]}")
    return hits


def regen_check():
    D = load_data()
    cmp = partial(cmp_csv, pat=D["pattern"])
    TBL = ["target", "arm", "model"]
    HEADV = ["g1_test_mae", "g1_test_mae_sd", "g1_test_nmae", "g1_test_r2"]
    STATS = ["mae", "se", "sd", "n_seeds", "n_test"]
    checks = [
        cmp("rev4_headline.csv", TBL, exact=["n_rows", "label"], approx=HEADV),
        cmp("rev4_appendix_full.csv", TBL, exact=["n_rows", "label"], approx=HEADV),
        cmp("rev4_appendix_by_arm.csv", TBL, exact=["n_rows"], approx=HEADV),
        cmp("rev4_appendix_by_model.csv", TBL, exact=["n_rows"], approx=HEADV),
        cmp("rev4_pre_ml.csv", ["target"], exact=["feature", "n_rows"],
            approx=["g1_pre_ml_mae", "g1_pre_ml_bias", "g1_pre_ml_pearson_r"]),
        cmp("espley_compare_main.csv", ["target_key"], exact=["target", "espley", "g1"],
            approx=["espley_mae", "espley_se", "g1_mae", "g1_se", "paired_diff_espley_minus_g1", "paired_diff_sd",
                    "mae_ratio_espley_over_g1", "pre_ml_xtb_g1_mae", "n_test_per_seed", "n_test_espley_per_seed"]),
        cmp("espley_compare_role.csv", ["target_key", "side", "protocol", "arm", "model"], approx=STATS),
        cmp("espley_compare_appendix_best.csv", ["side", "protocol", "target"],
            exact=["arm", "model", "n_models_compared"], approx=STATS),
        cmp("espley_compare_index_d1d2_appendix.csv",
            ["target_key", "selection", "side", "protocol", "arm", "model"], approx=STATS),
        cmp("espley_compare_per_seed.csv", ["side", "protocol", "arm", "model", "target", "seed"],
            exact=["n_test", "n_test_espley"], approx=["mae", "se"]),
        cmp_checks_json(),
    ]
    for sfx in ("", "_ownrows"):
        checks += [
            cmp(f"espley_geometry_ablation{sfx}.csv", TBL, exact=["n_rows", "n_tune", "our_geom", "extra"],
                approx=["mae", "se", "sd_seeds", "cv_mae_tune", "n_test_mean"], tol=TOL_ABL, soft=["best_params"]),
            cmp(f"espley_geometry_ablation{sfx}_per_seed.csv", TBL + ["seed"], exact=["n_test", "n_train"],
                approx=["mae", "se"], tol=TOL_ABL),
            cmp_rows_json(f"espley_geometry_ablation{sfx}_rows.json"),
        ]
    regenerated = sorted({c["file"] for c in checks} | {"rev4_headline.md", "espley_compare_checks.json"})
    hits = scan_pattern(regenerated, D["pattern"])
    fails = {c["file"]: c["fails"] for c in checks if c["fails"]}
    if hits:
        fails["discard pattern in regenerated files"] = hits[:20]
    rep = dict(check=f"REV5 A-1: regenerated results_rev4 files vs {REF_COMMIT} (G1 / Espley rows only)",
               verdict="FAIL" if fails else "PASS", ref_commit=REF_COMMIT,
               tolerances=dict(tables=TOL_EXACT, ablation=TOL_ABL), files=checks,
               discard_pattern_hits=hits, failures=fails, time=now(), job=job())
    write_json(RES5 / "A_regen_check.json", rep)
    for c in checks:
        print(f"{'FAIL' if c['fails'] else 'ok  '} {c['file']:52s} new {c.get('n_new', '-')} ref {c.get('n_ref', '-')} "
              f"(kept {c.get('n_ref_kept', '-')}, discarded {c.get('n_ref_discarded', '-')}) "
              f"{c['fails'][:2] if c['fails'] else ''}{' soft: ' + str(c['soft_diffs'][:2]) if c.get('soft_diffs') else ''}")
    print(f"A-1 regen check: {rep['verdict']} -> {RES5 / 'A_regen_check.json'}")
    if fails:
        sys.exit(3)


# ---------------------------------------------------------------------------------------------------- A-4 grep check
def moved_notes(D, rx):
    """(header notes, failures) for the 'moved' entries of DATA: files moved out of the scanned directory. Each must
    have its old path gone from the working tree, its new path a file outside the scanned directory, and bytes equal
    to REF_COMMIT:<old path> (moved, not edited); the note states how many of its lines match the pattern."""
    notes, fails = [], []
    for old, new in D["moved"]:
        po, pn = REPO / old, REPO / new
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{REF_COMMIT}:{old}"], capture_output=True)
        why = []
        if r.returncode != 0:
            why.append(f"git show {REF_COMMIT}:{old} failed: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
        if po.exists() or po.is_symlink():
            why.append(f"{old} still exists")
        if not pn.is_file():
            why.append(f"{new} is not a file")
        else:
            try:
                pn.resolve().relative_to(HERE)
                why.append(f"{new} is inside the scanned directory {HERE}")
            except ValueError:
                pass
            if r.returncode == 0 and pn.read_bytes() != r.stdout:
                why.append(f"{new} differs from {REF_COMMIT}:{old} (edited, not only moved)")
        if why:
            fails += [f"moved {old} -> {new}: {w}" for w in why]
            continue
        n = sum(bool(rx.search(ln)) for ln in pn.read_text(encoding="utf-8", errors="replace").splitlines())
        notes.append(f"# note: {old} was moved, unedited (bytes == {REF_COMMIT}:{old}), to {new}, outside the scanned "
                     f"directory; it holds {n} lines matching the pattern, not scanned here and not allow-listed")
    return notes, fails


def grep_check():
    D = load_data()
    rel = HERE.relative_to(REPO)
    cmd = ["grep", "-rniE", D["pattern"], str(rel), "--include=*.py", "--include=*.sh", "--include=*.md"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode not in (0, 1):
        print(f"grep failed (exit {r.returncode}): {r.stderr.strip()[:300]}")
        sys.exit(2)
    rx = re.compile(D["pattern"], re.I)
    allowed, bad = [], []
    for line in r.stdout.splitlines():
        path, _, rest = line.partition(":")
        lineno, _, text = rest.partition(":")
        if path in D["allow_file"]:
            allowed.append(("allow_file", line))
            continue
        remainder = text
        for t in D["allow_text"]:
            remainder = remainder.replace(t, "")
        if remainder != text and not rx.search(remainder):
            allowed.append(("discard sentence", line))
        else:
            bad.append(line)
    notes, mfails = moved_notes(D, rx)
    verdict = "PASS" if not bad and not mfails else "FAIL"
    out = ([f"# REV5 A-4 grep check ({now()}, job {job()}, git {subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()})",
            f"# command (cwd {REPO}): grep -rniE \"<pattern of {DATA.name}>\" {rel} --include=*.py --include=*.sh --include=*.md",
            f"# allowed: files {D['allow_file']}; lines containing the discard sentence with no other match"]
           + notes
           + [f"verdict\t{verdict}", f"n_hits\t{len(allowed) + len(bad)}", f"n_allowed\t{len(allowed)}",
              f"n_not_allowed\t{len(bad)}", f"n_moved_check_failures\t{len(mfails)}", "", "## not allowed"] + bad
           + ["", "## moved-file check failures"] + mfails
           + ["", "## allowed"] + [f"[{why}] {ln}" for why, ln in allowed])
    write_atomic(RES5 / "A_grep_check.txt", "\n".join(out) + "\n")
    print("\n".join(out[:out.index("## not allowed")]))
    for ln in bad[:40]:
        print("  NOT ALLOWED:", ln[:200])
    for ln in mfails:
        print("  MOVED-FILE CHECK FAILED:", ln[:300])
    print(f"A-4 grep check: {verdict} -> {RES5 / 'A_grep_check.txt'}")
    if bad or mfails:
        sys.exit(3)


if __name__ == "__main__":
    cmds = {"delete": delete, "regen-check": regen_check, "grep-check": grep_check}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: r5_phase_a.py {{{'|'.join(cmds)}}}")
    RES5.mkdir(parents=True, exist_ok=True)
    cmds[sys.argv[1]]()
