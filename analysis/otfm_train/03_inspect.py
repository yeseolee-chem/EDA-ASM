#!/usr/bin/env python3
"""SPEC17rev2 Step 3 — probe Transition1x pkl schema + pretrained ckpt type.

GATE-3 output → artifacts/schema.md documenting:
  1. Transition1x pkl top-level keys + inner dict shapes
  2. checkpoint class (SBModule vs DDPMModule)
  3. atom-count-dependent layer names + shapes (for GATE-6a shape checks)

Read-only inspection. No training triggered.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
ROT = REPO / "external/react-ot"
T1X = ROT / "reactot/data/transition1x"
CKPT = ROT / "reactot-pretrained.ckpt"
OUT = BASE / "artifacts" / "schema.md"

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)

# torch.load on the pretrained checkpoint unpickles reactot.* classes, so
# the react-ot source tree must be importable BEFORE dump_ckpt() runs.
sys.path.insert(0, str(ROT))


def dump_pkl(f) -> list[str]:
    lines = [f"### {f.name}"]
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    lines.append(f"top-level keys: {list(d.keys())}")
    for k in ("reactant", "transition_state", "product"):
        if k in d:
            lines.append(f"  {k}: {list(d[k].keys())}")
            for kk, vv in d[k].items():
                try:
                    lines.append(f"     {kk}: len={len(vv)}  sample={str(vv[0])[:80]}")
                except Exception:
                    lines.append(f"     {kk}: {type(vv)}")
    for k in d:
        if k in ("reactant", "transition_state", "product"):
            continue
        try:
            lines.append(f"  {k}: len={len(d[k])}  sample={str(d[k][0])[:80]}")
        except Exception:
            lines.append(f"  {k}: {type(d[k])}")
    return lines


def dump_ckpt() -> list[str]:
    lines = ["### pretrained checkpoint"]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    lines.append(f"top-level keys: {list(ck.keys())}")
    hp = ck.get("hyper_parameters", {}) or {}
    for k, v in list(hp.items())[:40]:
        lines.append(f"  hp.{k}: {str(v)[:200]}")
    sd = ck.get("state_dict", {}) or {}
    lines.append(f"\nstate_dict param count: {len(sd)}")
    lines.append("atom-count-sensitive layers (embed / one_hot / encoder / decoder):")
    for k, v in sd.items():
        low = k.lower()
        if any(t in low for t in ("embed", "encoder", "decoder", "one_hot")):
            try:
                lines.append(f"  {k}: {tuple(v.shape)}")
            except AttributeError:
                lines.append(f"  {k}: {type(v)}")
    return lines


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "artifacts").mkdir(exist_ok=True)
    body = ["# SPEC17rev2 schema probe", ""]

    if not CKPT.exists():
        print(f"[FATAL] checkpoint missing: {CKPT}", file=sys.stderr)
        return 1

    # Document the shipped checkpoint (5-element atom mapping). Step 6's
    # apply_rot_patches() will widen ATOM_MAPPING to 7 elements, so the
    # atom-count-sensitive layers dumped below are exactly those that
    # partial_load must skip on ckpt reload. GATE-6a asserts that.
    body.append("## Shipped checkpoint (pre-patch)")
    body += dump_ckpt() + [""]
    body.append("## Expected reshape after GATE-6b patch")
    body.append("`ATOM_MAPPING` widens 5 → 7 elements (adds Cl, Br). "
                "`node_nfs` moves from `[9]*3` to `[11]*3`. Embed/output "
                "layers listed above become the partial_load skip set.")
    body.append("")

    pkls = sorted(T1X.glob("*.pkl")) if T1X.exists() else []
    if not pkls:
        msg = (f"[WARN] no Transition1x pkl under {T1X}. "
               "Download from https://zenodo.org/records/13131875 before Step 4/6.")
        print(msg)
        body.append(msg)
    else:
        for f in pkls:
            body += dump_pkl(f) + [""]

    OUT.write_text("\n".join(body) + "\n")
    print(f"wrote {OUT}")
    (BASE / "artifacts" / "GATE3_STATUS.txt").write_text(
        ("PASS" if pkls else "PARTIAL") + "\n"
        f"pkl_files={len(pkls)}\n"
        f"ckpt_ok=1\n"
    )
    print("=== GATE-3 " + ("PASS" if pkls else "PARTIAL (T1x pkl missing)") + " ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
