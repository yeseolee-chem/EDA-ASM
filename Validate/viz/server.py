#!/usr/bin/env python3
"""HTTP server for the ASR validation viewer + manual fragmentation editor.

Endpoints:
  GET  /                          → index.html
  GET  /index.html                → index.html
  GET  /data.json                 → rebuild + serve precomputed reaction data
  POST /api/fragmentation         → save manual fragmentation for one reaction
  POST /api/rerun                 → kick off run_one.sh for one reaction
  GET  /api/rerun_status/<rid>    → status of a running/finished rerun
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shlex
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

ALT_STAGE5A = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
RESULTS_OUT = ROOT / "Validate" / "refrag" / "results"
ORIG_STAGE5A = ROOT / "ADF_500" / "stage5a" / "per_reaction"

# Track ongoing reruns: {rid: {pid, started, log_path}}
RERUNS: dict[str, dict] = {}
RERUNS_LOCK = threading.Lock()

ATOMIC_NUMBER = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
VALID_RID = re.compile(r"^[A-Za-z0-9_]+$")


def is_rid_safe(rid: str) -> bool:
    """Defend against path traversal via the rid query parameter."""
    return bool(VALID_RID.match(rid)) and (ORIG_STAGE5A / rid).is_dir()


def _rebuild_data() -> None:
    """Re-run build_data.py; swallow errors so the page can still load."""
    try:
        env = {**os.environ, "PYTHONPATH": f"{ROOT}/src"}
        subprocess.run(
            [sys.executable, str(HERE / "build_data.py")],
            check=False, env=env, capture_output=True, timeout=60,
        )
    except Exception as exc:
        sys.stderr.write(f"[build_data] {exc}\n")


def _suggest_multiplicities(fragments: list[dict], symbols: list[str]) -> list[int]:
    """Even electrons → singlet (1), odd → doublet (2)."""
    out: list[int] = []
    for f in fragments:
        ne = 0
        for i in f["atom_indices"]:
            sym = symbols[i] if 0 <= i < len(symbols) else "?"
            ne += ATOMIC_NUMBER.get(sym, 0)
        out.append(1 if (ne % 2 == 0) else 2)
    return out


def _save_fragmentation(payload: dict) -> dict:
    """Persist a user-edited fragmentation as a stage5a result.json."""
    rid = payload.get("rid", "")
    if not is_rid_safe(rid):
        return {"ok": False, "error": f"invalid rid: {rid!r}"}
    frags_in = payload.get("fragments")
    if not isinstance(frags_in, list) or len(frags_in) < 2:
        return {"ok": False, "error": "need at least 2 fragments"}

    orig_path = ORIG_STAGE5A / rid / "result.json"
    if not orig_path.exists():
        return {"ok": False, "error": f"no original stage5a for {rid}"}
    orig = json.loads(orig_path.read_text())

    n_atoms = int(orig["n_atoms"])
    all_idx: set[int] = set()
    fragments: list[dict] = []
    for fi, f in enumerate(frags_in):
        atoms = f.get("atom_indices", [])
        if not atoms:
            return {"ok": False, "error": f"fragment {fi} has no atoms"}
        for i in atoms:
            if not isinstance(i, int) or i < 0 or i >= n_atoms:
                return {"ok": False, "error": f"bad atom index {i!r}"}
            if i in all_idx:
                return {"ok": False, "error": f"atom {i} assigned to two fragments"}
            all_idx.add(i)
        try:
            mult = int(f.get("multiplicity", 1))
        except (TypeError, ValueError):
            return {"ok": False, "error": f"bad multiplicity for frag {fi}"}
        if mult < 1 or mult > 5:
            return {"ok": False, "error": f"multiplicity {mult} out of range"}
        role = str(f.get("role") or f"comp_{fi}")
        fragments.append({
            "atom_indices": sorted(atoms),
            "role": role,
            "multiplicity": mult,
            "cap_attachment": None,
        })

    missing = set(range(n_atoms)) - all_idx
    if missing:
        return {"ok": False, "error": f"unassigned atoms: {sorted(missing)}"}

    coupling = payload.get("coupling") or _auto_coupling(fragments)
    spin_signs = _spin_signs(fragments)
    total_spin = sum(s * (f["multiplicity"] - 1) for f, s in zip(fragments, spin_signs))

    new_stage5a = dict(orig)
    new_stage5a["result"] = {
        "pattern": orig["result"]["pattern"] + "_MANUAL",
        "fragments": fragments,
        "spin_signs": spin_signs,
        "total_spin_polarization": int(total_spin),
        "coupling": coupling,
        "n_fragments": len(fragments),
        "cap_h_positions": None,
        "confidence": 1.0,
        "notes": "Manual fragmentation entered via Validate/viz editor.",
        "debug": {"source": "viz/editor"},
    }
    new_stage5a["fragmentation_revision"] = 99

    out_dir = ALT_STAGE5A / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(new_stage5a, indent=2))

    return {
        "ok": True, "rid": rid,
        "n_fragments": len(fragments),
        "coupling": coupling,
        "total_spin": int(total_spin),
        "saved_to": str((out_dir / "result.json").relative_to(ROOT)),
    }


def _auto_coupling(fragments: list[dict]) -> str:
    """Default coupling label from the multiplicity tuple."""
    mults = [f["multiplicity"] for f in fragments]
    open_shells = sum(1 for m in mults if m > 1)
    if open_shells == 0:
        return "closed_shell_singlet"
    if open_shells == 1:
        return f"multiplet_{max(mults)}"
    return "broken_symmetry_singlet"


def _spin_signs(fragments: list[dict]) -> list[int]:
    """Antiferromagnetic alternation for open-shell fragments."""
    out: list[int] = []
    open_i = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            out.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            out.append(1)
    return out


def _trigger_rerun(rid: str) -> dict:
    """Spawn ./Validate/refrag/run_one.sh <rid> in the background."""
    if not is_rid_safe(rid):
        return {"ok": False, "error": f"invalid rid: {rid!r}"}
    stage5a = ALT_STAGE5A / rid / "result.json"
    if not stage5a.exists():
        return {"ok": False, "error": "no saved fragmentation for this rid"}

    with RERUNS_LOCK:
        cur = RERUNS.get(rid)
        if cur and _is_alive(cur.get("pid")):
            return {"ok": False, "error": "already running",
                    "pid": cur["pid"], "started": cur["started"]}

    # Clear any old result so status returns "running" until the new one lands.
    out_path = RESULTS_OUT / f"{rid}.json"
    if out_path.exists():
        out_path.unlink()

    cmd = ["bash", str(ROOT / "Validate/refrag/run_one.sh"), rid]
    log = ROOT / "Validate/refrag/logs" / f"{rid}.editor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=str(ROOT), close_fds=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with RERUNS_LOCK:
        RERUNS[rid] = {"pid": p.pid, "started": started, "log": str(log)}
    return {"ok": True, "rid": rid, "pid": p.pid, "started": started}


def _is_alive(pid: int | None) -> bool:
    """True if process pid is alive (POSIX kill(0))."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _rerun_status(rid: str) -> dict:
    """Report whether a rerun is running, complete, or absent."""
    if not is_rid_safe(rid):
        return {"status": "invalid"}
    with RERUNS_LOCK:
        cur = RERUNS.get(rid)
    out_path = RESULTS_OUT / f"{rid}.json"
    if cur and _is_alive(cur["pid"]):
        log_tail = _tail(cur.get("log"), 4)
        return {"status": "running", "pid": cur["pid"],
                "started": cur["started"], "log_tail": log_tail}
    if out_path.exists():
        return {"status": "done",
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(out_path.stat().st_mtime))}
    return {"status": "idle"}


def _tail(path: str | None, n: int) -> str:
    """Last n lines of a file as a single string; empty on error."""
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            chunk = min(size, 8192)
            fh.seek(size - chunk)
            data = fh.read().decode(errors="replace")
        lines = [l for l in data.split("\n") if l.strip()]
        return "\n".join(lines[-n:])
    except Exception:
        return ""


class Handler(http.server.BaseHTTPRequestHandler):
    """All endpoints in one handler."""

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")

    # ─── GET ────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._serve_file(HERE / "index.html", "text/html; charset=utf-8")
        if path == "/data.json":
            _rebuild_data()
            return self._serve_file(HERE / "data.json", "application/json; charset=utf-8",
                                     no_cache=True)
        if path.startswith("/api/rerun_status/"):
            rid = path.rsplit("/", 1)[-1]
            return self._json(_rerun_status(rid))
        if path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        # Serve static files under /docs/ and /docs_download/.
        for route, link_name in (("/docs/", "docs"),
                                   ("/docs_download/", "docs_download")):
            if path.startswith(route):
                rel = path[len(route):].lstrip("/")
                if ".." in rel.split("/"):
                    return self._404()
                fp = (HERE / link_name / rel).resolve()
                root_dir = (HERE / link_name).resolve()
                if not str(fp).startswith(str(root_dir)) or not fp.exists() or not fp.is_file():
                    return self._404()
                ctype = {
                    ".html": "text/html; charset=utf-8",
                    ".md":   "text/markdown; charset=utf-8",
                    ".json": "application/json",
                    ".csv":  "text/csv",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".zip":  "application/zip",
                    ".png":  "image/png",
                    ".svg":  "image/svg+xml",
                }.get(fp.suffix, "application/octet-stream")
                return self._serve_file(fp, ctype)
        if path in ("/docs", "/docs/"):
            return self._serve_file(HERE / "docs" / "index.html",
                                     "text/html; charset=utf-8")
        self._404()

    # ─── POST ───────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/api/fragmentation":
            data = self._read_json()
            return self._json(_save_fragmentation(data))
        if self.path == "/api/rerun":
            data = self._read_json()
            return self._json(_trigger_rerun(data.get("rid", "")))
        self._404()

    # ─── helpers ───────────────────────────────────────────────────────
    def _serve_file(self, p: Path, ctype: str, no_cache: bool = False):
        if not p.exists():
            return self._404()
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_cache:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            return json.loads(raw.decode() or "{}")
        except Exception:
            return {}

    def _404(self):
        body = b"404 not found"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    """Start the HTTP server."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    with ThreadedServer((args.host, args.port), Handler) as httpd:
        print(f"Serving Validate/viz at http://{args.host}:{args.port}/")
        print(f"  index: http://localhost:{args.port}/index.html")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
