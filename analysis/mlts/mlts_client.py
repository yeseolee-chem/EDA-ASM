#!/usr/bin/env python3
"""mlts_client.py — the ORCA `! ExtOpt` client of the MLTS server (spec 6 A-1). Standard library only.

ORCA 6.1.1 calls  <ProgExt> <base>_EXT.extinp.tmp <Ext_Params>  for every energy / gradient it needs
(DECISIONS.md '0-5 예비 확인'); Ext_Params = '--socket <unix socket> [--log <file>]'.
  1. read <base>_EXT.extinp.tmp (xyz file name, charge, multiplicity, NCores, do-gradient; '#' comments)
  2. read the xyz (A), send one 'evaluate' request to the server (length-prefixed JSON, the orca-mlips wire format)
  3. write <xyz base>.engrad next to the extinp (ORCA engrad format: Eh, Eh/bohr), atomically
  4. append one line to <extinp dir>/mlts_ext.log: wall_ms eval_ms n_atoms do_gradient (call count + overhead)
mlts_bridge.client_exe() installs a copy with the shebang '#!<python of the mlts env> -IS' (isolated, no site
import: start-up is a few tens of ms because nothing outside the standard library is imported).
Units: the server returns eV and eV/A (MLIP convention); converted here with the mlt_common constants (C-4)."""
import json
import os
import socket
import struct
import sys
import time

EH2EV = 27.211386245988          # = mlt_common.EH2EV
BOHR2A = 0.529177210903          # = mlt_common.BOHR2A
_HDR = struct.Struct("!I")


def _recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        b = s.recv(n - len(buf))
        if not b:
            return None
        buf.extend(b)
    return bytes(buf)


def call(path, req, timeout=900.0, tries=30):
    """one request / response over the Unix socket; connection refusals (server busy, backlog full) are retried."""
    data = json.dumps(req).encode()
    last = None
    for k in range(tries):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect(path)
            s.sendall(_HDR.pack(len(data)) + data)
            h = _recv_exact(s, _HDR.size)
            if h is None:
                raise OSError("server closed the connection")
            body = _recv_exact(s, _HDR.unpack(h)[0])
            if body is None:
                raise OSError("server closed mid-message")
            return json.loads(body.decode())
        except (ConnectionRefusedError, BlockingIOError, FileNotFoundError) as e:
            last = e
            time.sleep(min(2.0, 0.05 * (k + 1)))
        finally:
            s.close()
    raise OSError(f"server at {path} not reachable: {last}")


def read_extinp(path):
    ent = []
    with open(path) as f:
        for ln in f:
            t = ln.split("#", 1)[0].strip()
            if t:
                ent.append(t)
    if len(ent) < 5:
        raise ValueError(f"{path}: fewer than 5 entries")
    return ent[0], int(ent[1]), int(ent[2]), int(ent[3]), int(ent[4])


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        sym.append(t[0])
        pos.append([float(t[1]), float(t[2]), float(t[3])])
    if len(sym) != n:
        raise ValueError(f"{path}: {len(sym)} atom lines for {n} atoms")
    return sym, pos


def write_engrad(path, n_atoms, e_eh, grad):
    out = ["#", "# Number of atoms", "#", str(int(n_atoms)), "#", "# Total energy [Eh]", "#", f"{e_eh:.12e}"]
    if grad:
        out += ["#", "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...", "#"] + [f"{g:.12e}" for g in grad]
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def main(argv):
    t0 = time.time()
    if not argv:
        print("usage: mlts_client <base>_EXT.extinp.tmp --socket <path> [--log <file>]", file=sys.stderr)
        return 2
    extinp, sock, log = argv[0], None, None
    k = 1
    while k < len(argv):
        if argv[k] == "--socket" and k + 1 < len(argv):
            sock = argv[k + 1]
            k += 2
        elif argv[k] == "--log" and k + 1 < len(argv):
            log = argv[k + 1]
            k += 2
        else:
            k += 1
    sock = sock or os.environ.get("MLTS_SOCKET")
    if not sock:
        print("[mlts_client] no --socket", file=sys.stderr)
        return 2
    d = os.path.dirname(os.path.abspath(extinp))
    xyz, charge, mult, ncores, dograd = read_extinp(extinp)
    xyz = xyz if os.path.isabs(xyz) else os.path.join(d, xyz)
    sym, pos = read_xyz(xyz)
    r = call(sock, dict(action="evaluate", symbols=sym, coords_ang=pos, charge=charge, multiplicity=mult,
                        need_forces=bool(dograd), need_hessian=False, hessian_mode="Analytical", hessian_step=1e-3))
    if r.get("status") != "ok":
        print(f"[mlts_client] server error: {r.get('message')}", file=sys.stderr)
        return 1
    e_eh = float(r["energy_ev"]) / EH2EV
    grad = []
    if dograd:
        f = r.get("forces_ev_ang")
        if f is None or len(f) != len(sym):
            print("[mlts_client] server returned no / wrong forces", file=sys.stderr)
            return 1
        c = BOHR2A / EH2EV                                    # -F [eV/A] -> gradient [Eh/bohr]
        grad = [-float(x) * c for row in f for x in row]
    base = os.path.basename(xyz)
    base = base[:-4] if base.lower().endswith(".xyz") else base
    write_engrad(os.path.join(d, base + ".engrad"), len(sym), e_eh, grad)
    if ncores > 1:                                            # ORCA parallel NumFreq looks for <base>.hostnames
        hn = os.path.join(d, os.path.basename(extinp).replace("_EXT.extinp.tmp", "") + ".hostnames")
        if not os.path.exists(hn):
            with open(hn, "w") as fh:
                fh.write((socket.gethostname() + "\n") * max(ncores, 1))
    line = f"{(time.time() - t0) * 1e3:.2f} {float(r.get('t_eval_s', 0.0)) * 1e3:.2f} {len(sym)} {int(dograd)}\n"
    try:
        fd = os.open(log or os.path.join(d, "mlts_ext.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.write(fd, line.encode())
        os.close(fd)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:                                  # ORCA sees the non-zero exit and stops
        print(f"[mlts_client] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
