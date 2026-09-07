#!/usr/bin/env python3
"""Flask viewer for the 11 failed rxns from spec17rev2 Step 2.

3-panel R / TS / P layout with 3Dmol.js viewers + editable XYZ textareas.
Saves edits to fixed/<rid>/{r0,r1,ts,p}.xyz — Coley originals untouched.

Runs on a compute node under sbatch (see sbatch/viewer.sh). Browser
access: http://<node>:8765  (SSH tunnel: ssh -L 8765:node:8765 gate1.hpc)
"""
from __future__ import annotations

import os
import re
import socket
from pathlib import Path

from flask import Flask, jsonify, render_template, request

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
PROF = BASE / "coley_profiles/full_dataset_profiles"
FIXED = BASE / "tools/failed_viewer/fixed"
FIXED.mkdir(parents=True, exist_ok=True)

# The 11 excluded rxns from EXECUTION_LOG.md
FAILING = {
    3090: {"group": "A", "reason": "1-piece split + P-fail. TS→P에 결합 파괴 2개 존재 (다단계)"},
    3766: {"group": "A", "reason": "1-piece split + P-fail. 형성 결합 1개만 (cycloaddition 아님)"},
    4252: {"group": "A", "reason": "1-piece split + P-fail. 결합 형성 2 + 파괴 1 (원자 이동)"},
    5930: {"group": "A", "reason": "no isomorphism + P-fail. R xyz vs TS-fragment 그래프 다름"},
    3865: {"group": "A", "reason": "no isomorphism. R conformer의 spurious H-contact 1개"},
    4069: {"group": "A", "reason": "no isomorphism. 위와 동일 conformer artifact"},
    4727: {"group": "A", "reason": "no isomorphism. 위와 동일"},
    2461: {"group": "B", "reason": "verify_product만 실패. P에 대칭 원자 재라벨링 (25↔26)"},
    4924: {"group": "B", "reason": "verify_product만 실패. 대칭 34↔37 교환"},
    5386: {"group": "B", "reason": "verify_product만 실패. 대칭 재라벨링"},
    5513: {"group": "B", "reason": "verify_product만 실패. 동일 유형"},
}

app = Flask(__name__, template_folder="templates")


def _read_xyz_text(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def _find_files(rid: int) -> dict:
    """Locate xyz files for a rxn — Coley original + any saved edit."""
    d = PROF / str(rid)
    ts = next((f for f in d.iterdir()
               if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"), None)
    p0 = next((f for f in d.iterdir() if f.name.startswith("p0_")), None)
    rs = sorted(f for f in d.iterdir()
                if re.match(r"^r\d+_", f.name) and f.suffix == ".xyz"
                and "_alt" not in f.name)
    r0 = rs[0] if len(rs) >= 1 else None
    r1 = rs[1] if len(rs) >= 2 else None

    fixed_dir = FIXED / str(rid)
    return {
        "r0_orig": _read_xyz_text(r0) if r0 else "",
        "r1_orig": _read_xyz_text(r1) if r1 else "",
        "ts_orig": _read_xyz_text(ts) if ts else "",
        "p_orig":  _read_xyz_text(p0) if p0 else "",
        "r0_edit": _read_xyz_text(fixed_dir / "r0.xyz"),
        "r1_edit": _read_xyz_text(fixed_dir / "r1.xyz"),
        "ts_edit": _read_xyz_text(fixed_dir / "ts.xyz"),
        "p_edit":  _read_xyz_text(fixed_dir / "p.xyz"),
        "ts_name": ts.name if ts else None,
    }


@app.route("/")
def index():
    rows = [{"rxn_id": rid, **info} for rid, info in sorted(FAILING.items())]
    return render_template("index.html", rows=rows)


@app.route("/rxn/<int:rid>")
def rxn(rid):
    if rid not in FAILING:
        return jsonify({"error": "rxn not in failing set"}), 404
    return jsonify({
        "rxn_id": rid,
        "info": FAILING[rid],
        **_find_files(rid),
    })


@app.route("/save/<int:rid>/<slot>", methods=["POST"])
def save(rid, slot):
    if rid not in FAILING or slot not in ("r0", "r1", "ts", "p"):
        return jsonify({"error": "bad rxn or slot"}), 400
    xyz = request.get_json(force=True).get("xyz", "")
    if not xyz.strip():
        return jsonify({"error": "empty xyz"}), 400
    # basic sanity: first line integer atom count matches following coord lines
    lines = xyz.strip().split("\n")
    try:
        n = int(lines[0])
    except ValueError:
        return jsonify({"error": "first line not integer"}), 400
    if len(lines) < 2 + n:
        return jsonify({"error": f"expected {n} coord lines, got {len(lines)-2}"}), 400

    d = FIXED / str(rid)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{slot}.xyz"
    tmp = out.with_suffix(".xyz.tmp")
    tmp.write_text(xyz)
    tmp.replace(out)
    return jsonify({"ok": True, "path": str(out), "n_atoms": n})


@app.route("/reset/<int:rid>/<slot>", methods=["POST"])
def reset(rid, slot):
    if rid not in FAILING or slot not in ("r0", "r1", "ts", "p"):
        return jsonify({"error": "bad rxn or slot"}), 400
    out = FIXED / str(rid) / f"{slot}.xyz"
    if out.exists():
        out.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("VIEWER_PORT", "8765"))
    host = socket.gethostname().split(".")[0]
    print(f"\n=== viewer up ===")
    print(f"  browser URL      : http://{host}:{port}")
    print(f"  SSH tunnel (mac) : ssh -N -L {port}:{host}:{port} yeseo1ee@gate1.hpc")
    print(f"  then open        : http://localhost:{port}")
    print(f"===============\n", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
