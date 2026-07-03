"""Patch SCF block of c4/c5 ADF input decks with convergence aids.

Targets: reactions whose status.json calc_status shows c4_fragA_opt or
c5_fragB_opt as 'not_converged' or 'failed' — i.e. the
scf_not_converged and failed_step rows in failures.csv.

Per-step patch applied (only to the step that actually failed):
  ORIGINAL                        PATCHED (idempotent)
  SCF                             SCF                          # PATCHED_BY_SCF_AIDS
    Iterations 200                  Iterations 500
    Converge 1.0e-6                 Converge 1.0e-6
  End                               Mixing 0.05
                                    DIIS
                                      N 25
                                      Cyc 5
                                      OK 1.0e-3
                                    End
                                  End

Idempotency: a `# PATCHED_BY_SCF_AIDS` marker on the SCF line means the
file is already patched and is skipped.

NO ADF calls. Modifies only the targeted c4/c5 .in files.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Optional

PATCH_MARKER = "# PATCHED_BY_SCF_AIDS"

PATCHED_SCF_BLOCK = """  SCF                          {marker}
    Iterations 500
    Converge 1.0e-6
    Mixing 0.05
    DIIS
      N 25
      Cyc 5
      OK 1.0e-3
    End
  End
""".format(marker=PATCH_MARKER)


_SCF_BLOCK_RE = re.compile(
    r"^(?P<indent>\s*)SCF\b(?P<rest>[^\n]*)\n"   # opening SCF
    r"(?P<body>(?:(?!^\s*End\b).*\n)+?)"          # body
    r"^\s*End\b[^\n]*\n",                         # closing End
    re.MULTILINE,
)


def _is_already_patched(text: str) -> bool:
    return PATCH_MARKER in text


def patch_text(text: str) -> tuple[str, bool]:
    """Apply patch to the SCF block of an .in file's full text.

    Returns (new_text, did_change).
    """
    if _is_already_patched(text):
        return text, False
    m = _SCF_BLOCK_RE.search(text)
    if not m:
        # No SCF block present at all — inject one before the EndEngine line.
        # Conservative: if we can't find EndEngine either, refuse to edit.
        if "EndEngine" not in text:
            raise ValueError("no SCF block AND no EndEngine — refuse to patch")
        new = text.replace("EndEngine", PATCHED_SCF_BLOCK + "EndEngine", 1)
        return new, True
    new = text[: m.start()] + PATCHED_SCF_BLOCK + text[m.end():]
    return new, True


def _read_targets_from_status(rxn_dir: Path) -> list[str]:
    """Return the list of step filenames (c4_fragA_opt.in / c5_fragB_opt.in)
    whose calc_status was 'not_converged' or 'failed'."""
    status_p = rxn_dir / "status.json"
    if not status_p.is_file():
        return []
    s = json.loads(status_p.read_text())
    calc = s.get("calc_status", {})
    bad_steps = [k for k, v in calc.items() if v in ("not_converged", "failed")]
    # Only patch c4/c5 — geometry-opt steps are the ones SCF aids target.
    return [f"{k}.in" for k in bad_steps if k in ("c4_fragA_opt", "c5_fragB_opt")]


def _process_one(rxn_dir: Path, dry_run: bool) -> dict:
    record = {"rxn_dir": str(rxn_dir), "targets": [],
              "patched": [], "skipped_already": [], "errors": []}
    targets = _read_targets_from_status(rxn_dir)
    if not targets:
        record["errors"].append("no_c4_c5_targets_in_status")
        return record
    record["targets"] = targets
    for fname in targets:
        path = rxn_dir / fname
        if not path.is_file():
            record["errors"].append(f"missing_input: {fname}")
            continue
        text = path.read_text()
        try:
            new_text, did_change = patch_text(text)
        except ValueError as exc:
            record["errors"].append(f"{fname}: {exc}")
            continue
        if not did_change:
            record["skipped_already"].append(fname)
            continue
        diff = "\n".join(difflib.unified_diff(
            text.splitlines(), new_text.splitlines(),
            fromfile=f"{path} (original)",
            tofile=f"{path} (patched)",
            lineterm="",
        ))
        print(f"\n----- {path} -----\n{diff}")
        if not dry_run:
            backup = path.with_suffix(path.suffix + ".bak_prepatch")
            if not backup.exists():
                backup.write_text(text)
            path.write_text(new_text)
        record["patched"].append(fname)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets-file",
                    default="outputs/asr_v1/retry/scf_aid_targets.txt",
                    help="newline-separated absolute paths to reaction dirs "
                         "(produced by retry_dryrun_report.py)")
    ap.add_argument("--rxn-dir", action="append", default=[],
                    help="add a specific reaction dir to the target list "
                         "(repeatable; overrides --targets-file if given)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show unified diffs but DO NOT modify any file")
    ap.add_argument("--summary-out",
                    default="outputs/asr_v1/retry/scf_patch_summary.json")
    args = ap.parse_args()

    if args.rxn_dir:
        targets = [Path(p) for p in args.rxn_dir]
    else:
        tf = Path(args.targets_file)
        if not tf.is_file():
            print(f"[err] targets file not found: {tf}\n"
                  f"      run scripts/asr_v1/retry_dryrun_report.py first.",
                  file=sys.stderr)
            sys.exit(1)
        targets = [Path(p) for p in tf.read_text().splitlines() if p.strip()]
    print(f"[patch_scf_aids] {len(targets)} reaction(s) — "
          f"{'DRY-RUN' if args.dry_run else 'WRITING CHANGES'}")

    records = []
    n_patched_files = 0
    n_already = 0
    n_err = 0
    for d in targets:
        rec = _process_one(d, dry_run=args.dry_run)
        records.append(rec)
        n_patched_files += len(rec["patched"])
        n_already += len(rec["skipped_already"])
        n_err += len(rec["errors"])

    summary = {
        "dry_run": args.dry_run,
        "n_reactions": len(targets),
        "n_files_patched": n_patched_files,
        "n_files_already_patched": n_already,
        "n_errors": n_err,
        "records": records,
    }
    out_p = Path(args.summary_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))

    tag = "would patch" if args.dry_run else "patched"
    print(f"\n=== Summary ===")
    print(f"  reactions visited     : {len(targets)}")
    print(f"  files {tag:18s}: {n_patched_files}")
    print(f"  files already patched : {n_already}")
    print(f"  errors                : {n_err}")
    print(f"  summary written       : {out_p}")
    if args.dry_run and (n_patched_files or n_err):
        print("\n[hint] re-run without --dry-run to actually write changes.")


if __name__ == "__main__":
    main()
