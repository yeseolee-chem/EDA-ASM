"""Tiny HTTP server with cache disabled (so browsers re-fetch HTML/JS every reload).

Usage:  python3 viz/serve.py [PORT]
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8889
    root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(root)
    srv = ThreadingHTTPServer(("0.0.0.0", port), NoCacheHandler)
    print(f"serving {root} at http://0.0.0.0:{port}/viz/  (no-cache)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
