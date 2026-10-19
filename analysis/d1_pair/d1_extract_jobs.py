#!/usr/bin/env python3
"""d1_extract_jobs.py — build <D1_JOBS> (user decision 2026-10-19 at the Phase 0 STOP: no unpacked D1 job folder
exists on the server, so the needed files are extracted read-only from the D1 archives).

  python d1_extract_jobs.py run [N_PROC]      # extract every status-ok row of labels_d1.json -> $D1PAIR_SCRATCH/d1_jobs
  python d1_extract_jobs.py check             # MANIFEST.tsv + cross-check vs D1_run/results/geometries

Sources (never modified):
  harvested rows   d1_status.csv `harvested` = bundle name -> NURION_TARS/<bundle>.tar; members
                   jobs/<T>/{ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz, label.json} and jobs/<T>/archive.tgz
                   (nested; member sp/eda.inp)
  other rows       UBAI_JOBS/<T>/{same files} + UBAI_JOBS/<T>/archive.tgz (or an unpacked sp/ folder)
Output per job: D1_JOBS/<T>/{ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz, label.json, sp/eda.inp, source.txt},
written into <T>.tmp/ and renamed (atomic per job); a job whose folder exists with all files is skipped (idempotent).
check: MANIFEST.tsv (job_id, source, sha256 of each file) and EXTRACT_CHECK.json (ts / ref files byte-identical to
results/geometries/<T>/, label.json job_id / rxn_id / file fields = labels_d1.json). Exit 3 if any ok row lacks a
file or its ts / ref files differ from results/geometries (label.json field differences are reported only).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1pair_common as C  # noqa: E402

TOP = ("ts.xyz", "ref_dipole.xyz", "ref_dipolarophile.xyz", "label.json")


def _norm(n):
    return n[2:] if n.startswith("./") else n


def eda_inp_from_tgz(fobj):
    """sp/eda.inp text from a (streamed) archive.tgz, or None."""
    with tarfile.open(fileobj=fobj, mode="r|*") as t:
        for m in t:
            if m.isfile() and _norm(m.name) == "sp/eda.inp":
                return t.extractfile(m).read()
    return None


def finish(jid, files, source):
    """write D1_JOBS/<jid>/ atomically from {relpath: bytes}."""
    final = C.D1_JOBS / jid
    tmp = C.D1_JOBS / f"{jid}.tmp"
    C.assert_write_ok(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "sp").mkdir(parents=True)
    for rel, data in files.items():
        (tmp / rel).write_bytes(data)
    (tmp / "source.txt").write_text(source + "\n")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def complete(jid):
    d = C.D1_JOBS / jid
    return d.is_dir() and all((d / f).is_file() for f in C.JOB_FILES)


def from_tar(args):
    bundle, jobs = args
    tar = C.NURION_TARS / f"{bundle}.tar"
    want = {j for j in jobs if not complete(j)}
    if not want:
        return bundle, len(jobs), 0, []
    got = {j: {} for j in want}
    with tarfile.open(tar, mode="r:") as T:
        for m in T:
            if not m.isfile():
                continue
            parts = _norm(m.name).split("/")
            if len(parts) != 3 or parts[0] != "jobs" or parts[1] not in got:
                continue
            jid, fn = parts[1], parts[2]
            if fn in TOP:
                got[jid][fn] = T.extractfile(m).read()
            elif fn == "archive.tgz":
                inp = eda_inp_from_tgz(T.extractfile(m))
                if inp is not None:
                    got[jid]["sp/eda.inp"] = inp
    bad = []
    for jid, files in got.items():
        if set(files) == set(C.JOB_FILES):
            finish(jid, files, f"nurion:{tar.name}")
        else:
            bad.append((jid, sorted(set(C.JOB_FILES) - set(files))))
    return bundle, len(jobs), len(want) - len(bad), bad


def from_ubai(jid):
    if complete(jid):
        return jid, "skip", []
    d = C.UBAI_JOBS / jid
    files = {f: (d / f).read_bytes() for f in TOP if (d / f).is_file()}
    if (d / "sp" / "eda.inp").is_file():
        files["sp/eda.inp"] = (d / "sp" / "eda.inp").read_bytes()
        src = f"ubai:{d}/sp"
    elif (d / "archive.tgz").is_file():
        with open(d / "archive.tgz", "rb") as f:
            inp = eda_inp_from_tgz(f)
        if inp is not None:
            files["sp/eda.inp"] = inp
        src = f"ubai:{d}/archive.tgz"
    else:
        src = f"ubai:{d}"
    miss = sorted(set(C.JOB_FILES) - set(files))
    if miss:
        return jid, "missing", miss
    finish(jid, files, src)
    return jid, "ok", []


def ok_rows():
    lab = C.load_d1_labels()
    st = pd.read_csv(C.D1_STATUS, dtype=str, keep_default_na=False)
    if C.sha256(C.D1_STATUS) != C.D1_STATUS_SHA:
        print("STOP: d1_status.csv sha256", file=sys.stderr); sys.exit(3)
    jobs = sorted(d["job_id"] for d in lab)
    st = st.set_index("job_id").loc[jobs]
    return lab, st


def cmd_run(nproc):
    lab, st = ok_rows()
    C.D1_JOBS.mkdir(parents=True, exist_ok=True)
    harv = st[st.harvested != ""]
    ubai = st[st.harvested == ""].index.tolist()
    groups = [(b, g.index.tolist()) for b, g in harv.groupby("harvested")]
    print(f"[extract] {len(lab)} ok rows: {len(harv)} in {len(groups)} bundles, {len(ubai)} UBAI folders", flush=True)
    bad = []
    with Pool(nproc) as pool:
        for b, n, done, miss in pool.imap_unordered(from_tar, groups):
            print(f"  {b}: {n} rows, {done} extracted now, {len(miss)} incomplete {miss[:3]}", flush=True)
            bad += miss
    for jid in ubai:
        j, s, miss = from_ubai(jid)
        if miss:
            bad.append((j, miss))
    print(f"[extract] incomplete: {len(bad)} {bad[:10]}", flush=True)
    return 3 if bad else 0


def cmd_check():
    lab, st = ok_rows()
    byj = {d["job_id"]: d for d in lab}
    rows, bad = [], []
    for jid in sorted(byj):
        d = C.D1_JOBS / jid
        r = dict(job_id=jid, rxn_id=int(byj[jid]["rxn_id"]),
                 source=(d / "source.txt").read_text().strip() if (d / "source.txt").is_file() else "")
        for f in C.JOB_FILES:
            r[f] = C.sha256(d / f) if (d / f).is_file() else ""
        miss = [f for f in C.JOB_FILES if not r[f]]
        same = {}
        for f in ("ts.xyz", "ref_dipole.xyz", "ref_dipolarophile.xyz"):
            g = C.D1_GEOM / jid / f
            same[f] = bool(g.is_file() and r[f] and C.sha256(g) == r[f])
        lj = json.loads((d / "label.json").read_text()) if r["label.json"] else {}
        lab_same = lj.get("job_id") == jid and int(lj.get("rxn_id", -1)) == r["rxn_id"]
        rel_same = all(lj.get(k) == byj[jid].get(k) for k in ("ts_file", "rel1_file", "rel2_file", "formed_pairs_ts"))
        r.update(geom_identical=all(same.values()), label_ids_match=lab_same, label_files_match=rel_same,
                 missing=";".join(miss))
        rows.append(r)
        if miss or not all(same.values()):                 # label.json differences are reported, not fatal
            bad.append(jid)
    df = pd.DataFrame(rows)
    C.write_atomic(C.D1_JOBS / "MANIFEST.tsv", lambda t: df.to_csv(t, sep="\t", index=False))
    rep = dict(n_ok_rows=len(df), n_complete=int((df.missing == "").sum()),
               n_geom_identical=int(df.geom_identical.sum()), n_label_ids_match=int(df.label_ids_match.sum()),
               n_label_files_match=int(df.label_files_match.sum()),
               sources=df.source.str.split(":").str[0].value_counts().to_dict(), bad=bad[:50], n_bad=len(bad),
               manifest_sha256=C.sha256(C.D1_JOBS / "MANIFEST.tsv"), d1_jobs=str(C.D1_JOBS))
    C.write_json_safe(C.RES / "EXTRACT_CHECK.json", rep)
    print(json.dumps(rep, indent=1))
    return 3 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        sys.exit(cmd_run(int(a[1]) if len(a) > 1 else int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    if a[:1] == ["check"]:
        sys.exit(cmd_check())
    raise SystemExit(__doc__)
