"""Reset reviewed=False for reactions whose partition was auto-generated but
wound up marked reviewed=True by a stale-cache save from the browser.

Detection: use the `note` field as source of truth.
  - note starts with "auto R" / "auto TS" (bare "auto ...", not "auto-refit")
      → auto-generated, was not user-reviewed → reviewed=False
  - note starts with "replacement " → freshly drawn from raw pool → reviewed=False
  - note starts with "auto-refit (was reviewed)" → user had reviewed before → reviewed=True
  - note is "" or user-supplied text → leave reviewed as-is (respect user's flag)

Writes an atomic backup before touching manual_partitions.json.
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path

MANUAL = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/frag_review/manual_partitions.json")


def main():
    with open(MANUAL) as f:
        m = json.load(f)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = MANUAL.with_name(f"manual_partitions.backup.{ts}.json")
    shutil.copy2(MANUAL, backup)
    print(f"backup → {backup}")

    n_reset_true_to_false = 0
    n_kept_reviewed = 0
    n_leave_alone = 0
    n_no_change = 0
    for rid, v in m.items():
        note = v.get("note", "") or ""
        rev_before = bool(v.get("reviewed"))
        if note.startswith("auto-refit (was reviewed)"):
            new_rev = True
        elif note.startswith("auto ") or note.startswith("replacement "):
            new_rev = False
        else:
            new_rev = rev_before  # respect user
        if new_rev != rev_before:
            v["reviewed"] = new_rev
            if not new_rev:
                n_reset_true_to_false += 1
        elif new_rev:
            n_kept_reviewed += 1
        else:
            n_leave_alone += 1
        n_no_change = len(m) - n_reset_true_to_false - n_kept_reviewed - n_leave_alone

    with open(MANUAL, "w") as f:
        json.dump(m, f, indent=1)

    total_reviewed = sum(1 for v in m.values() if v.get("reviewed"))
    print(f"reviewed=True→False (bogus auto flags cleared): {n_reset_true_to_false}")
    print(f"reviewed=True kept (auto-refit was reviewed):    {n_kept_reviewed}")
    print(f"reviewed status untouched (user-authored notes): {n_leave_alone}")
    print(f"final reviewed=True count: {total_reviewed}")


if __name__ == "__main__":
    main()
