"""tests/d0_check.py — sandbox only: parse gen_dft.py / gen_d0.py / finetune.py with this interpreter, import them
(gen_dft only under python < 3.10 = the D1 env used by run_g2), print the D0 constants. Writes nothing."""
import ast
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
files = ["gen_dft.py"] + (["gen_d0.py", "finetune.py"] if sys.version_info >= (3, 10) else [])
for f in files:
    ast.parse((HERE / f).read_text(), f)
print("ast ok", files, sys.version.split()[0], flush=True)
sys.path.insert(0, str(HERE))
import gen_dft  # noqa: E402

print("gen_dft", gen_dft.PLAN_D0_PQ, gen_dft.HESS_D0, gen_dft.STOP_D0, gen_dft.UNIT_KINDS, flush=True)
if sys.version_info >= (3, 10):
    import gen_d0  # noqa: E402
    import finetune  # noqa: E402
    print("gen_d0", gen_d0.D0_LABELS, gen_d0.D0_G2_FINAL, gen_d0.D1_G0, gen_d0.d1val_ids()[:5], flush=True)
    print("gen_d0 imports the same gen_dft module:", gen_d0.G is gen_dft, flush=True)
    print("finetune inputs", [n for n, _ in finetune.input_files(False)], finetune.TS_NEAR_ALWAYS, flush=True)
    print("classify D1 forms:", [gen_d0.classify(s)[0] for s in (
        "[CH3:1][C:2](=[O:3])[C:4]#[N+:5][O-:6].[CH2:7]=[CH2:8]>>[CH3:1][C:2](=[O:3])[C:4]1=[N:5][O:6][CH2:7][CH2:8]1",
        "[CH2:1]=[CH:2][CH3:3].[CH:4]#[N+:5][C-:6]([CH3:7])[C:8]#[N:9]>>[CH2:1]1[CH:2]([CH3:3])[CH:4]=[N:5][C:6]1([CH3:7])[C:8]#[N:9]")],
          flush=True)
