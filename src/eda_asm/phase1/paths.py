"""Canonical paths used by every Phase 1 stage."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
HALO8_DIR = DATA_DIR / "Halo8"
INDEX_DIR = DATA_DIR / "halo8_index"
OUTPUT_DIR = ROOT / "outputs" / "phase1"
TMP_DIR = OUTPUT_DIR / ".tmp"
REVIEW_DIR = OUTPUT_DIR / "manual_review_queue"
LOG_DIR = ROOT / "logs"

INDEX_PARQUET = INDEX_DIR / "index.parquet"
BOND_CHANGES_PARQUET = INDEX_DIR / "bond_changes_all.parquet"

SELECTED_CSV = OUTPUT_DIR / "selected_reactions.csv"
BOND_CHANGES_JSON = OUTPUT_DIR / "bond_changes.json"
CASE_JSON = OUTPUT_DIR / "case_classification.json"
FRAGMENTS_AUTO_JSON = OUTPUT_DIR / "fragments_auto.json"
FRAGMENTS_FINAL_JSON = OUTPUT_DIR / "fragments_final.json"
MANUAL_REVIEW_LOG = OUTPUT_DIR / "manual_review_log.json"
PHASE1_H5 = OUTPUT_DIR / "phase1_output.h5"
SAMPLING_REPORT_HTML = OUTPUT_DIR / "sampling_report.html"
LOG_FILE = LOG_DIR / "phase1.log"

DB_FILES = [HALO8_DIR / f"Halo_{i}.db" for i in range(1, 11)]


def ensure_dirs() -> None:
    for d in (DATA_DIR, INDEX_DIR, OUTPUT_DIR, TMP_DIR, REVIEW_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
