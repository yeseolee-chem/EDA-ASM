"""In-place patches to the cloned react-ot source tree.

Extends ATOM_MAPPING from 5 (H C N O F) to 7 (adds Cl, Br) and the
run_model.py element allowlist to match. Must run BEFORE any react-ot
dataset module is imported — `n_element = len(ATOM_MAPPING)` is evaluated
at import time and baked into `node_nfs`.

Idempotent: a `.py.orig` backup marks that patching already happened; we
still rewrite the file each time so a stale react-ot pull re-patches
cleanly.

GATE-6b (SPEC17rev2 review):
  [ ] datasets_config.py ATOM_MAPPING == 7-element dict
  [ ] n_element == 7
  [ ] node_nfs == [11]*3  (= 3 pos + 7 atom + 1 charge)
  [ ] run_model.py allowed_atom_types includes F, Cl, Br
  [ ] .py.orig backup exists
"""
from __future__ import annotations

import re
from pathlib import Path

ATOM_MAPPING_TEXT = """ATOM_MAPPING = {
    1: 0,    # H
    6: 1,    # C
    7: 2,    # N
    8: 3,    # O
    9: 4,    # F
    17: 5,   # Cl   (SPEC17rev2 §5.3)
    35: 6,   # Br   (SPEC17rev2 §5.3)
}"""

EXPECTED_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 17: 5, 35: 6}
EXPECTED_ALLOWED = {"C", "H", "O", "N", "F", "Cl", "Br"}


def _backup(p: Path) -> None:
    orig = p.with_suffix(p.suffix + ".orig")
    if not orig.exists():
        orig.write_text(p.read_text())


def patch_atom_mapping(rot_root: Path) -> dict:
    """Rewrite reactot/dataset/datasets_config.py ATOM_MAPPING in place."""
    cfg = rot_root / "reactot" / "dataset" / "datasets_config.py"
    if not cfg.exists():
        raise FileNotFoundError(f"datasets_config.py not found: {cfg}")

    text = cfg.read_text()
    _backup(cfg)

    m = re.search(r"^ATOM_MAPPING\s*=\s*\{[^}]*\}", text, re.M)
    if m is None:
        raise RuntimeError("ATOM_MAPPING block not found in datasets_config.py")
    new_text = text[:m.start()] + ATOM_MAPPING_TEXT + text[m.end():]
    cfg.write_text(new_text)

    ns: dict = {}
    exec(ATOM_MAPPING_TEXT, ns)
    got = ns["ATOM_MAPPING"]
    assert got == EXPECTED_MAPPING, f"patch mismatch: {got}"
    n_element = len(got)
    assert n_element == 7, f"n_element={n_element}, expected 7"
    expected_nfs = 3 + n_element + 1
    assert expected_nfs == 11, f"node_nfs {expected_nfs} != 11"
    print(f"[patch] {cfg.name}: ATOM_MAPPING -> {n_element} elements, "
          f"node_nfs = [{expected_nfs}]*3")
    return got


def patch_allowed_atoms(rot_root: Path) -> bool:
    """Extend run_model.py's allowed_atom_types. Returns True if patched."""
    p = rot_root / "reactot" / "run_model.py"
    if not p.exists():
        return False
    text = p.read_text()
    if "allowed_atom_types" not in text:
        return False
    _backup(p)
    new = re.sub(
        r"allowed_atom_types\s*=\s*\{[^}]*\}",
        "allowed_atom_types = {'C', 'H', 'O', 'N', 'F', 'Cl', 'Br'}",
        text,
    )
    p.write_text(new)
    print(f"[patch] {p.name}: allowed_atom_types extended to {sorted(EXPECTED_ALLOWED)}")
    return True


def assert_gate_6b(rot_root: Path) -> None:
    """Verify all patches took effect. Raises on failure."""
    cfg = rot_root / "reactot" / "dataset" / "datasets_config.py"
    txt = cfg.read_text()
    ns: dict = {}
    m = re.search(r"^ATOM_MAPPING\s*=\s*\{[^}]*\}", txt, re.M)
    if m is None:
        raise RuntimeError("GATE-6b FAIL: ATOM_MAPPING missing after patch")
    exec(m.group(0), ns)
    if ns["ATOM_MAPPING"] != EXPECTED_MAPPING:
        raise RuntimeError(
            f"GATE-6b FAIL: ATOM_MAPPING = {ns['ATOM_MAPPING']}, "
            f"expected {EXPECTED_MAPPING}"
        )
    if not cfg.with_suffix(cfg.suffix + ".orig").exists():
        raise RuntimeError(f"GATE-6b FAIL: no .orig backup at {cfg}")

    rm = rot_root / "reactot" / "run_model.py"
    if rm.exists() and "allowed_atom_types" in rm.read_text():
        got = re.search(r"allowed_atom_types\s*=\s*\{([^}]*)\}", rm.read_text())
        if got:
            names = {s.strip().strip("'\"") for s in got.group(1).split(",")}
            missing = EXPECTED_ALLOWED - names
            if missing:
                raise RuntimeError(
                    f"GATE-6b FAIL: run_model.py allowed_atom_types missing {missing}"
                )
    print("[GATE-6b] patch verification PASSED")


def apply_all(rot_root: Path) -> dict:
    """Run every patch + verify. Call once before invoking react-ot training."""
    mapping = patch_atom_mapping(rot_root)
    patch_allowed_atoms(rot_root)
    assert_gate_6b(rot_root)
    return mapping
