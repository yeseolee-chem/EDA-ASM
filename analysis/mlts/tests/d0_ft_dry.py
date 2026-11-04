"""tests/d0_ft_dry.py — sandbox only: `finetune.py data --dry <runs>` on the current $MLT_SCRATCH root; with --with-real-I
the input I is the production $MLT/dft/existing.extxyz (read only, in-process override) so the Phase I D0 class overlay
and the tag check run on real Phase I D0 rows. Writes nothing."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import finetune as F  # noqa: E402

REAL_I = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/existing.extxyz")
args = sys.argv[1:]
if "--with-real-I" in args:
    args.remove("--with-real-I")
    F.INPUTS = (("I", REAL_I),) + tuple(x for x in F.INPUTS if x[0] != "I")
sys.exit(F.main(["data", "--dry"] + args))
