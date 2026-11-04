"""tests/d0plus_hash.py — sandbox only, TEST ROOTS ONLY (D0-plus idempotency): `save <tag>` writes the sha256 + mtime_ns
of every file in $MLT/dft/plan and of the json / csv files in $MLT_RESULTS to $MLT/hash_<tag>.json; `cmp <tag>` prints
the files whose sha256 or mtime changed since that snapshot (empty = nothing was written)."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlt_common as M  # noqa: E402

if "/tmp/d0plustest" not in str(M.MLT) or "/tmp/d0plustest" not in str(M.RES):
    raise SystemExit("refusing: not a d0plustest root")
files = sorted((M.MLT / "dft" / "plan").glob("*")) + sorted(p for p in M.RES.glob("*") if p.suffix in (".json", ".csv"))
snap = {str(p): dict(sha256=M.sha256(p), mtime_ns=p.stat().st_mtime_ns) for p in files if p.is_file()}
mode, tag = sys.argv[1], sys.argv[2]
f = M.MLT / f"hash_{tag}.json"
if mode == "save":
    M.write_json(f, snap)
    print(json.dumps(dict(saved=str(f), n=len(snap))))
else:
    old = M.read_json(f)
    diff = {k: dict(old=old.get(k), new=snap.get(k)) for k in sorted(set(old) | set(snap)) if old.get(k) != snap.get(k)}
    print(json.dumps(dict(compared=str(f), n=len(snap), n_changed=len(diff), changed=sorted(diff)), indent=1))
