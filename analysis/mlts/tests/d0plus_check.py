"""tests/d0plus_check.py — sandbox only (D0-plus): parse gen_dft.py / gen_d0.py / finetune.py, run pyflakes on them when
the interpreter has it, import them and print the D0-plus constants. Writes nothing."""
import ast
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
files = ["gen_dft.py", "gen_d0.py", "finetune.py"]
for f in files:
    ast.parse((HERE / f).read_text(), f)
print("ast ok", files, sys.version.split()[0], flush=True)
try:
    from pyflakes import api as PF
    from pyflakes import reporter as PR
    rep = PR.Reporter(sys.stdout, sys.stderr)
    n = sum(PF.checkPath(str(HERE / f), rep) for f in files)
    print("pyflakes warnings:", n, flush=True)
except ImportError:
    import builtins
    import symtable

    def undefined(path):
        """names read in some scope that resolve to the module scope but are never bound there (nor builtins)."""
        src = path.read_text()
        top = symtable.symtable(src, str(path), "exec")
        bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
        bound |= {c.get_name() for c in top.get_children()}
        bad = set()

        def walk(t):
            for s in t.get_symbols():
                n = s.get_name()
                glob = s.is_global() or (t.get_type() == "module")
                if s.is_referenced() and glob and n not in bound and not hasattr(builtins, n):
                    bad.add((t.get_name(), n))
            for c in t.get_children():
                walk(c)
        walk(top)
        return sorted(bad)

    for f in files:
        print("undefined-name scan", f, undefined(HERE / f), flush=True)
sys.path.insert(0, str(HERE))
import gen_dft  # noqa: E402
import gen_d0  # noqa: E402
import finetune  # noqa: E402

print("BUDGET_CORE_H", gen_dft.BUDGET_CORE_H, "gen_d0 sees", gen_d0.BUDGET_CORE_H, "G4 reserve", gen_d0.G4_RESERVE_CORE_H)
print("CODE_D0 (fingerprint)", gen_d0.CODE_D0, "CODE_D0_REV", gen_d0.CODE_D0_REV, "finetune", finetune.CODE_VERSION)
print("holdout files", gen_d0.HOLD_CSV, gen_d0.HOLD_CSV_RES, gen_d0.HOLD_REF, finetune.D0_HOLDOUT)
print("same gen_dft module:", gen_d0.G is gen_dft, "decisions:", [k for k in gen_d0.DECISIONS_D0 if k.startswith("D0-")][-4:],
      "FT-d0hold" in finetune.DECISIONS)
