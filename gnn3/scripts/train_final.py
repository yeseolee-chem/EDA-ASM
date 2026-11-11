"""Train the deployable GNN3 on every training pair (5 seeds), with the cell and epoch count chosen by cross-validation.

    python gnn3/scripts/train_final.py --units /gpfs/tmp_cpu2/$USER/gnn3/cv --out /gpfs/tmp_cpu2/$USER/gnn3/final
    python gnn3/scripts/train_final.py --original --dry-run          # only print the setting the published units imply

Rule (``training.choose_final_setting``): the cell chosen by most of the 5 folds (ties: lowest mean tuning validation
value, then grid order); epochs = median of the 5 chosen best epochs. The final networks are written as
gnn3_final__seed<j>.pt with final_report.json.

The published clean-up did NOT run this (out of scope of docs/specs/GNN3_CLEAN.md); it is the documented procedure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn3 import constants as K  # noqa: E402


def unit_records(units: Path, original: bool) -> list[dict]:
    """The 5 cross-validation records: gnn3_fold<k>.json (this package) or D__lf3__S__GNN__f<k>.json (original)."""
    names = [f"D__lf3__S__GNN__f{k}.json" if original else f"gnn3_fold{k}.json" for k in range(K.N_FOLDS)]
    missing = [n for n in names if not (units / n).is_file()]
    if missing:
        raise SystemExit(f"missing cross-validation units in {units}: {missing}")
    return [json.loads((units / n).read_text()) for n in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=Path, default=None, help="directory of the 5 cross-validation units")
    parser.add_argument("--original", action="store_true", help="read the published units (ORIGINAL_UNITS)")
    parser.add_argument("--out", type=Path, default=K.SCRATCH / "final")
    parser.add_argument("--dry-run", action="store_true", help="print the chosen setting and stop")
    args = parser.parse_args()
    from gnn3.training import choose_final_setting
    units = args.units or (K.ORIGINAL_UNITS if args.original else K.SCRATCH / "cv")
    setting = choose_final_setting(unit_records(units, args.original))
    print(json.dumps(setting, indent=1))
    if args.dry_run:
        return 0
    from gnn3.data import PairDataset
    from gnn3.training import train_final_model
    report = train_final_model(PairDataset(), setting, args.out)
    print(f"final model: {report['n_train_pairs']} training pairs, files in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
