#!/usr/bin/env python3
"""D1-G2R Phase C (docs/specs/D1G2R.md 3): copy every D1 target's autodE folder $G2/jobs/<job_id>/ to
$G2R/jobs/<job_id>/ (all files, sub-folders and marker files), one reaction folder at a time (temporary folder ->
os.replace, idempotent: an existing destination folder is kept and only verified), then compare every file with its
source (bytes, sha256).

    python analysis/g2r/c_copy.py            any env with python >= 3.8 (stdlib only); 16 workers by SLURM_CPUS_PER_TASK

Outputs: $G2R/x/C_manifest.csv.gz (job_id, relative path, bytes, sha256), results/C_report.json.
Gates: C-1 9,266 reaction folders, file count and bytes equal to the source; C-2 no sha256 mismatch. Exit 0 / 3 (STOP).
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

N_TARGETS = 9266


def targets() -> list[str]:
    with open(C.G_STATUS, newline="") as handle:
        return [r["job_id"] for r in csv.DictReader(handle)]


def files_of(root: Path) -> dict:
    """relative path -> (bytes, sha256 or 'link:<target>') of every file below root (dot files included)."""
    out = {}
    for folder, _dirs, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            rel = str(path.relative_to(root))
            if path.is_symlink():
                out[rel] = (0, "link:" + os.readlink(path))
            else:
                out[rel] = (path.stat().st_size, C.sha256(path))
    return out


def copy_one(job_id: str) -> dict:
    src, dst = C.G2_OLD_JOBS / job_id, C.G2R_JOBS / job_id
    if not src.is_dir():
        return dict(job_id=job_id, error="source folder missing")
    copied = False
    if not dst.exists():
        tmp = C.G2R_JOBS / f".{job_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp, symlinks=True, copy_function=shutil.copy2)
        os.replace(tmp, dst)
        copied = True
    a, b = files_of(src), files_of(dst)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    differ = sorted(r for r in set(a) & set(b) if a[r] != b[r])
    return dict(job_id=job_id, copied=copied, n_files=len(a), bytes=sum(v[0] for v in a.values()),
                n_dst_files=len(b), dst_bytes=sum(v[0] for v in b.values()), missing=missing, extra=extra,
                differ=differ, rows=[(job_id, r, b[r][0], b[r][1]) for r in sorted(b)])


def main() -> int:
    jobs = targets()
    C.G2R_JOBS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as pool:
        results = pool.map(copy_one, jobs, chunksize=20)
    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    rows = [row for r in ok for row in r["rows"]]
    C.write_csv_gz(C.G2R_X / "C_manifest.csv.gz", ["job_id", "relative_path", "bytes", "sha256"], rows)
    n_folders = sum(1 for p in C.G2R_JOBS.iterdir() if p.is_dir() and not p.name.startswith("."))
    record = dict(
        spec="docs/specs/D1G2R.md 3 (Phase C)", source=str(C.G2_OLD_JOBS), destination=str(C.G2R_JOBS),
        n_targets=len(jobs), n_reaction_folders=n_folders, n_copied_now=sum(r["copied"] for r in ok),
        n_files_source=sum(r["n_files"] for r in ok), n_files_destination=sum(r["n_dst_files"] for r in ok),
        bytes_source=sum(r["bytes"] for r in ok), bytes_destination=sum(r["dst_bytes"] for r in ok),
        n_missing=sum(len(r["missing"]) for r in ok), n_extra=sum(len(r["extra"]) for r in ok),
        n_sha256_mismatch=sum(len(r["differ"]) for r in ok), errors=errors[:20],
        n_symlinks=sum(1 for row in rows if str(row[3]).startswith("link:")),
        symlinks=[row[:2] + (row[3],) for row in rows if str(row[3]).startswith("link:")][:20],
        examples=[dict(job_id=r["job_id"], missing=r["missing"][:5], extra=r["extra"][:5], differ=r["differ"][:5])
                  for r in ok if r["missing"] or r["extra"] or r["differ"]][:20],
        manifest=str(C.G2R_X / "C_manifest.csv.gz"), manifest_sha256=C.sha256(C.G2R_X / "C_manifest.csv.gz"),
        seconds=round(time.time() - started), slurm_job=os.environ.get("SLURM_JOB_ID"),
        written=time.strftime("%Y-%m-%d %H:%M:%S"))
    gates = {"C-1": (len(jobs) == N_TARGETS and n_folders == N_TARGETS and not errors
                     and record["n_files_source"] == record["n_files_destination"]
                     and record["bytes_source"] == record["bytes_destination"]
                     and record["n_missing"] == 0 and record["n_extra"] == 0),
             "C-2": record["n_sha256_mismatch"] == 0}
    record.update(gates=gates, all_pass=all(gates.values()))
    C.write_json(C.RES / "C_report.json", record)
    print({k: v for k, v in record.items() if k not in ("examples", "errors")})
    return 0 if record["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
