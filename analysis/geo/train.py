#!/usr/bin/env python3
"""Phase R: GEO-1 training.
  python train.py smoke <k> | tune <k> <cfg> | select <k> | final <k> <seed> | ablation <k> <no_d0|q1_only|r0>"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from model import GeoRefiner  # noqa: E402
from metrics import evaluate_preds  # noqa: E402     (10-2)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = dict(S=dict(L=6, dx=256, dp=64, H=8, do=16, dt=32))
CFGS = {"c0": ("S", 0.05, 1), "c1": ("S", 0.05, 3), "c2": ("S", 0.20, 1), "c3": ("S", 0.20, 3)}
BS, LR, MAX_EP, PATIENCE, W_FORM, LOCAL_CUT = 16, 2e-4, 100, 15, 10.0, 4.0


def load_records():
    return GC.load_records()                              # decision U-0: numpy 2.x / 1.x compatible loader


def split(recs, k):
    inner = (k + 1) % 5
    test = [r for r in recs if r["src"] == "D1" and r["fold"] == k]
    ival = [r for r in recs if r["src"] == "D1" and r["fold"] == inner]
    train = [r for r in recs if (r["src"] == "D1" and r["fold"] not in (k, inner))
             or (r["src"] == "D0" and not r["overlap"])]
    return train, ival, test


def train_items(recs, use_d0=True):
    S = []
    for r in recs:
        if r["src"] == "D0" and not use_d0:
            continue
        if r["xyz_g2"] is not None and r["same"] is True:
            S.append((r, "g2"))
        if r["xyz_g1"] is not None:
            S.append((r, "g1"))
    return S


def eval_items(recs):
    return [(r, "g2") for r in recs if r["xyz_g2"] is not None]


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def sample_q(rng):
    u = rng.random()
    return 1.0 if u < 0.1 else (0.0 if u < 0.9 else float(rng.uniform(0.4, 0.6)))


def make_example(item, train, sigma, q_mode, rng):
    r, kind = item
    x_in, y = r[f"xyz_{kind}"], r[f"lab_on_{kind}"]
    if train:
        q = 1.0 if q_mode == "q1" else sample_q(rng)
        x0 = q * x_in + (1.0 - q) * (y + rng.normal(0.0, sigma, y.shape))
        c, R = x0.mean(0), rand_rot(rng)
        x0, y = (x0 - c) @ R.T, (y - c) @ R.T
    else:
        c = x_in.mean(0)
        x0, y = x_in - c, y - c
    return dict(types=r["types"], frag=r["frag"], form=r["form"], src=r["src_id"], chg=r["charge"],
                spd=r[f"spd_{kind}"], bond=r[f"bond_{kind}"], x0=x0, y=y, pairs=r["pairs"], center=c, uid=r["uid"])


def collate(exs):
    B, N = len(exs), max(len(e["types"]) for e in exs)
    t = {k: torch.zeros((B, N), dtype=torch.long) for k in ("types", "frag", "form")}
    spd, bond = torch.zeros((B, N, N), dtype=torch.long), torch.zeros((B, N, N), dtype=torch.long)
    x0, y = torch.zeros((B, N, 3)), torch.zeros((B, N, 3))
    mask = torch.zeros((B, N), dtype=torch.bool)
    for i, e in enumerate(exs):
        n = len(e["types"])
        for k in t:
            t[k][i, :n] = torch.as_tensor(e[k])
        spd[i, :n, :n], bond[i, :n, :n] = torch.as_tensor(e["spd"]), torch.as_tensor(e["bond"])
        x0[i, :n], y[i, :n] = torch.as_tensor(e["x0"], dtype=torch.float32), torch.as_tensor(e["y"], dtype=torch.float32)
        mask[i, :n] = True
    b = dict(**t, spd=spd, bond=bond, x0=x0, y=y, mask=mask,
             src=torch.tensor([e["src"] for e in exs]), chg=torch.tensor([e["chg"] for e in exs]).clamp(-3, 3),
             pairs=torch.tensor(np.stack([e["pairs"] for e in exs])))
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def batches(items, train, sigma, q_mode, rng, shuffle):
    idx = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for s in range(0, len(idx), BS):
        exs = [make_example(items[i], train, sigma, q_mode, rng) for i in idx[s:s + BS]]
        yield exs, collate(exs)


def pdist(x):
    return torch.sqrt(((x[:, :, None] - x[:, None]) ** 2).sum(-1) + 1e-12)


def loss_fn(pred, b):
    m = b["mask"]
    mf = m[..., None].float()
    l_coord = ((pred - b["y"]).abs() * mf).sum() / (mf.sum() * 3)
    dp, dl = pdist(pred), pdist(b["y"])
    eye = torch.eye(m.shape[1], dtype=torch.bool, device=m.device)
    pm = m[:, :, None] & m[:, None, :] & ~eye[None] & (dl < LOCAL_CUT)
    l_local = ((dp - dl).abs() * pm).sum() / pm.sum()
    bi = torch.arange(m.shape[0], device=m.device)[:, None]
    i, j = b["pairs"][..., 0], b["pairs"][..., 1]
    df_p = torch.sqrt(((pred[bi, i] - pred[bi, j]) ** 2).sum(-1) + 1e-12)
    df_l = torch.sqrt(((b["y"][bi, i] - b["y"][bi, j]) ** 2).sum(-1) + 1e-12)
    l_form = (df_p - df_l).abs().mean()
    return l_coord + l_local + W_FORM * l_form, dict(coord=l_coord.item(), local=l_local.item(), form=l_form.item())


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def model(self, template):
        m = GeoRefiner(**template)
        m.load_state_dict(self.shadow)
        return m.to(DEV).eval()


@torch.no_grad()
def predict(model, items, n_pass):
    model.eval()
    out, rng = {}, np.random.default_rng(0)
    for exs, b in batches(items, False, 0.0, "mix", rng, False):
        pred = model(b, n_pass).cpu().numpy()
        for i, e in enumerate(exs):
            n = len(e["types"])
            out[e["uid"]] = pred[i, :n] + e["center"]          # back to the input frame
    return out


def fit(tr, va, cfg_name, seed, fixed_epochs=None, q_mode="mix", n_pass_override=None, tag=""):
    size, sigma, R = CFGS[cfg_name]
    n_pass = n_pass_override or (R + 1)
    arch = dict(SIZE[size])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GeoRefiner(**arch).to(DEV)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
    n_ep = fixed_epochs or MAX_EP
    total = MAX_EP * math.ceil(len(tr) / BS)    # same LR horizon in tune and final (final just stops at n_ep)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist, best, best_ep, t0 = [], None, -1, time.time()
    for ep in range(n_ep):
        model.train()
        agg = []
        for _, b in batches(tr, True, sigma, q_mode, rng, True):
            loss, parts = loss_fn(model(b, n_pass), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            agg.append(parts)
        h = dict(epoch=ep, **{k: float(np.mean([a[k] for a in agg])) for k in agg[0]}, wall_s=time.time() - t0)
        if va is not None:
            m = evaluate_preds(predict(ema.model(arch), va, n_pass), [r for r, _ in va])
            h.update(val_primary=m["primary_share"], val_rxn_median=m["rxn_form_abs_median"])
            key = (m["primary_share"], -m["rxn_form_abs_median"])
            if best is None or key > best:
                best, best_ep = key, ep
                torch.save(ema.shadow, GC.GEO / "runs" / f"{tag}.best.pt")
            if ep - best_ep >= PATIENCE:
                hist.append(h)
                break
        hist.append(h)
    if va is None:
        torch.save(ema.shadow, GC.GEO / "runs" / f"{tag}.final.pt")
    GC.write_json(GC.GEO / "runs" / f"{tag}.history.json",
                  dict(cfg=cfg_name, seed=seed, n_pass=n_pass, q_mode=q_mode, best_ep=best_ep, best=best, hist=hist,
                       n_train=len(tr), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    return ema, arch, n_pass, best_ep


def main(a):
    (GC.GEO / "runs").mkdir(parents=True, exist_ok=True)
    (GC.GEO / "preds").mkdir(parents=True, exist_ok=True)
    recs = load_records()
    cmd, k = a[0], int(a[1])
    tr_r, va_r, te_r = split(recs, k)
    if cmd == "smoke":                                    # timing only (9-4): 2 epochs of c0 (R=1) and c1 (R=3)
        for c in ("c0", "c1"):
            fit(train_items(tr_r), eval_items(va_r), c, 0, fixed_epochs=2, tag=f"smoke_f{k}_{c}")
        return 0
    if cmd == "tune":
        fit(train_items(tr_r), eval_items(va_r), a[2], 0, tag=f"tune_f{k}_{a[2]}")
        return 0
    if cmd == "select":
        H = {c: json.loads((GC.GEO / "runs" / f"tune_f{k}_{c}.history.json").read_text())
             for c in CFGS if (GC.GEO / "runs" / f"tune_f{k}_{c}.history.json").is_file()}
        best = max(H, key=lambda c: tuple(H[c]["best"]))
        GC.write_json(GC.RES / f"R_select_f{k}.json",
                      dict(fold=k, chosen=best, best_ep=H[best]["best_ep"], candidates={c: H[c]["best"] for c in H}))
        return 0
    sel = json.loads((GC.RES / f"R_select_f{k}.json").read_text())
    cfg, ep = sel["chosen"], sel["best_ep"] + 1
    full = tr_r + va_r                                    # final models: train + inner validation, fixed epochs
    if cmd == "final":
        seed = int(a[2])
        ema, arch, n_pass, _ = fit(train_items(full), None, cfg, seed, fixed_epochs=ep, tag=f"final_f{k}_s{seed}")
        P = predict(ema.model(arch), eval_items(te_r), n_pass)
        with open(GC.GEO / "preds" / f"final_f{k}_s{seed}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
        return 0
    if cmd == "ablation":
        name = a[2]
        kw = dict(no_d0=dict(use_d0=False), q1_only=dict(q_mode="q1"), r0=dict(n_pass_override=1))[name]
        items = train_items(full, use_d0=kw.pop("use_d0", True))
        ema, arch, n_pass, _ = fit(items, None, cfg, 0, fixed_epochs=ep, tag=f"abl_{name}_f{k}", **kw)
        P = predict(ema.model(arch), eval_items(te_r), n_pass)
        with open(GC.GEO / "preds" / f"abl_{name}_f{k}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
