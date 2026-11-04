"""tests/d1_collect_head.py — sandbox only, TEST ROOTS ONLY: run plain D1 `collect` with the committed (git HEAD)
gen_dft.py and with the working-tree gen_dft.py on the same $MLT_SCRATCH test root and compare the outputs
(gen_G{1,2,3}.extxyz bytes, results/G_report.json without written / host / wall_s). The HEAD source is compiled in
memory with __file__ = the real gen_dft.py (so it imports the same, unchanged mlt_common / orca_io); nothing is written
outside the test root. Prints JSON."""
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

if "/tmp/" not in str(M.MLT):
    raise SystemExit(f"refusing: MLT_SCRATCH {M.MLT} is not a test root")


def snap(G):
    out = {g: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for g, p in G.EXT.items()}
    rep = json.loads(G.G_REPORT.read_text())
    for k in ("written", "host", "wall_s"):
        rep.pop(k, None)
    out["G_report"] = hashlib.sha256(json.dumps(rep, sort_keys=True, default=str).encode()).hexdigest()
    out["complete"] = rep.get("complete")
    out["n_missing"] = rep.get("n_missing")
    return out


src = subprocess.run(["git", "-C", str(HERE), "show", "HEAD:analysis/mlts/gen_dft.py"], capture_output=True, text=True,
                     check=True).stdout
head = types.ModuleType("gen_dft_head")
head.__file__ = str(HERE / "gen_dft.py")
sys.modules["gen_dft_head"] = head
exec(compile(src, str(HERE / "gen_dft.py"), "exec"), head.__dict__)
rc_h = head.cmd_collect()
s_h = snap(head)
import gen_dft as NEW  # noqa: E402

rc_n = NEW.cmd_collect()
s_n = snap(NEW)
print(json.dumps(dict(root=str(M.MLT), head_sha_lines=len(src.splitlines()), rc_head=rc_h, rc_new=rc_n, head=s_h,
                      new=s_n, identical=s_h == s_n,
                      d0_files_in_root=sorted(p.name for p in (M.MLT / "dft" / "plan").glob("*d0*"))), indent=1))
