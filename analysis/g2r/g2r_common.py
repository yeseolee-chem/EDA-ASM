"""D1-G2R (docs/specs/D1G2R.md): paths, constants and small I/O helpers shared by the analysis/g2r scripts.

Standard library only (imported by scripts that run in the D1 run env, venv5 and env_d1fu2).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = Path(os.environ.get("G2R_RESULTS", HERE / "results"))

#: new g2 root (spec header) and the two old roots that phase D deletes (exact strings, spec 12-2)
G2R = Path(os.environ.get("G2R_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/g2r"))
G2R_JOBS, G2R_LF, G2R_X, G2R_LOGS = G2R / "jobs", G2R / "lf", G2R / "x", G2R / "logs"
G2_OLD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")
CH3_LF_OLD = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3/lf")
G2_OLD_JOBS, G2_OLD_FINAL = G2_OLD / "jobs", G2_OLD / "g2_final"

#: committed reports of the old g2 that the checks compare with (read only)
G_STATUS = REPO / "analysis" / "d1_g2" / "results" / "G_status.csv"
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
LF_REPORTS = REPO / "analysis" / "ch3" / "results" / "lf_inputs"
CR2_LABELS = REPO / "analysis" / "d1_ladder" / "results" / "labels_d1_cr2.parquet"


def sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path | str, data: bytes | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "wb") as out:
        out.write(data.encode() if isinstance(data, str) else data)
    os.replace(temporary, path)


def write_json(path: Path | str, obj) -> None:
    write_atomic(path, json.dumps(obj, indent=1, default=str, allow_nan=True) + "\n")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def write_csv_gz(path: Path | str, header: list[str], rows) -> None:
    import csv
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    write_atomic(path, gzip.compress(buffer.getvalue().encode()))
