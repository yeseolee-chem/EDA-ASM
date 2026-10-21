"""FU2 Phase M — atom-token pair models AW / GNN / TF x variants A / A+G on the arm C pairs (docs/specs/D1PAIR_FU2.md).

  python fu2/train.py smoke            # Phase C / M-3 smoke: T1 fold 0, every config, 2 epochs, gate M-1'
                                       #   -> results/fu2/C_smoke.json (writes only $FU2_SCR/smoke)
  python fu2/train.py run <i> <n>      # Phase M units [i::n]; refuses (exit 3) unless PREREG ADD2 is committed (Phase A)
  python fu2/train.py list             # unit names

Rules (M-1): pairs = d1_train.masks arm C (exported by fu2/splits.py), tuning validation = the first inner GroupKFold(4)
fold (rev 9 NN rule); per fit: numeric token columns and the global token standardised with the fit's training reactions
only, y / std(y_train) per target (ddof 0, no mean removed); loss = mean_t |yhat - y| (standardised) + 0.1 *
mean |yhat_barrier - sum_9 yhat| / s_barrier (kcal-scale sum); AdamW lr 1e-3 / wd 1e-4, cosine decay over 300 epochs
(the final fit stops at the tuned best epoch on the same schedule), dropout 0.1, batch 64 pairs, early stopping patience
30, random (a, b) / (b, a) order per pair and epoch (y sign flipped with it). Grid cell = lowest validation MAE (mean of
the 9 primary targets, standardised); final = all outer-training pairs, best epoch, seeds 20261019 + 0..4, mean.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fu2_common as F  # noqa: E402
import models as Mo  # noqa: E402

C = F.C
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ADD2_MD = C.RES / "PREREG_D1PAIR_ADD2.md"
ADD2_JSON = C.RES / "prereg_d1pair_add2.json"
NF = {"T1": 5, "LODO": 5, "LOSO": 10}


# ================================================================ data
class Data:
    """every token reaction as padded float64 numpy arrays (row = reaction), the pairs table and the split files."""

    def __init__(self):
        self.trep = F.read_json(F.FU2_RES / "T_report.json")
        if self.trep.get("stop"):
            raise SystemExit("STOP: T_report.json stop")
        # review (F1-F5 check): tokens must come from the current Q report / splits / token code version
        import atoms as _atoms
        import tokens as _tokens
        qrep = F.read_json(F.FU2_RES / "Q_report.json")
        self.q_report_sha, self.t_report_sha = F.sha256(F.FU2_RES / "Q_report.json"), F.sha256(F.FU2_RES / "T_report.json")
        bad = [k for k, ok in (("token_version", self.trep.get("token_version") == _tokens.TOKEN_VERSION),
                               ("q_report_sha256", self.trep.get("q_report_sha256") == self.q_report_sha),
                               ("splits_sha256", self.trep.get("splits_sha256") == F.sha256(F.FU2_RES / "S_splits.json")),
                               ("atom_version", qrep.get("atom_version") == _atoms.ATOM_VERSION)) if not ok]
        if bad:
            raise SystemExit(f"STOP: T_report.json does not match the current inputs: {bad} (rebuild Q gate -> T)")
        man = pd.read_csv(F.FU2_RES / "T_manifest.csv")
        if F.sha256(F.FU2_RES / "T_manifest.csv") != self.trep["manifest_sha256"]:
            raise SystemExit("STOP: T_manifest.csv changed after T_report.json")
        self.elements = [int(z) for z in self.trep["elements"]]
        self.num_cols, self.bin_cols = list(self.trep["num_cols"]), list(self.trep["bin_cols"])
        self.rids = man.rxn_id.astype(int).to_numpy()
        self.row = {int(r): k for k, r in enumerate(self.rids)}
        nr, nmax = len(self.rids), int(man.n_atoms.max())
        self.nmax, self.n = nmax, man.n_atoms.to_numpy(int)
        el = {z: k for k, z in enumerate(self.elements)}
        self.onehot = np.zeros((nr, nmax, len(self.elements)))
        self.bin = np.zeros((nr, nmax, len(self.bin_cols)))
        self.num = np.zeros((nr, nmax, len(self.num_cols)))
        self.pos = np.zeros((nr, nmax, 3))
        self.mask = np.zeros((nr, nmax), dtype=bool)
        self.etype = np.zeros((nr, nmax, nmax), dtype=np.int8)
        self.frag = np.zeros((nr, nmax), dtype=np.int8)
        self.glob = np.zeros((nr, len(self.trep["glob_cols"])))
        for k, (rid, sha) in enumerate(zip(self.rids, man.sha256)):
            p = F.TOKENS / f"{rid}.npz"
            if F.sha256(p) != sha:
                raise SystemExit(f"STOP: {p} sha256 differs from T_manifest")
            t = np.load(p)
            n = len(t["Z"])
            self.onehot[k, np.arange(n), [el[int(z)] for z in t["Z"]]] = 1.0
            self.bin[k, :n], self.num[k, :n], self.pos[k, :n] = t["bin"], t["num"], t["pos"]
            self.mask[k, :n] = True
            self.frag[k, :n] = t["bin"][:, 0]
            for (i, j), e in zip(t["edges"], t["etype"]):
                self.etype[k, i, j] = self.etype[k, j, i] = int(e)
            self.glob[k] = t["glob"]
        P = pd.read_parquet(F.SPLITS / "pairs.parquet")
        srep = F.read_json(F.FU2_RES / "S_splits.json")
        if F.sha256(F.SPLITS / "pairs.parquet") != srep["pairs_sha256"]:
            raise SystemExit("STOP: splits/pairs.parquet changed after S_splits.json")
        self.srep = srep
        self.s_splits_sha = F.sha256(F.FU2_RES / "S_splits.json")
        P["has_atoms"] = P.a_rxn.isin(set(self.row)) & P.b_rxn.isin(set(self.row))
        self.P = P.set_index("pair_id")
        self.y = self.P[[f"dd_{t}" for t in F.TARGETS]].to_numpy(float)
        self.pid_row = {p: k for k, p in enumerate(self.P.index)}
        self.ra = np.array([self.row.get(int(r), -1) for r in self.P.a_rxn])
        self.rb = np.array([self.row.get(int(r), -1) for r in self.P.b_rxn])
        self.d_in = len(self.elements) + len(self.bin_cols) + len(self.num_cols)
        self.d_glob = self.glob.shape[1]

    def static(self, dtype, device):
        """fit-independent tensors (positions, masks, edge types int8, fragment ids, sizes), cached per dtype / device."""
        key = (dtype, str(device))
        if not hasattr(self, "_st"):
            self._st = {}
        if key not in self._st:
            self._st[key] = dict(pos=torch.tensor(self.pos, dtype=dtype, device=device),
                                 mask=torch.tensor(self.mask, device=device),
                                 etype=torch.tensor(self.etype.astype(np.int8), device=device),
                                 frag=torch.tensor(self.frag.astype(np.int8), device=device),
                                 n=torch.tensor(self.n, device=device))
        return self._st[key]

    def split(self, split, k):
        """{train, test, inner_train, inner_val}: pair-table row indices (pairs without atom data dropped) + counts."""
        name = f"{split}__f{k}.json"
        p = F.SPLITS / name
        if F.sha256(p) != self.srep["units"][name]["sha256"]:
            raise SystemExit(f"STOP: {p} changed after S_splits.json")
        rec = F.read_json(p)
        out, dropped = {}, {}
        for key in ("train", "test", "inner_train", "inner_val"):
            rows = np.array([self.pid_row[q] for q in rec[key]])
            keep = self.P.has_atoms.to_numpy()[rows]
            out[key], dropped[key] = rows[keep], int((~keep).sum())
        out["dropped_no_atoms"] = dropped
        out["pairs_sha"] = C.list_sha([(q,) for q in rec["train"] + ["|"] + rec["test"]])
        return out


class Std:
    """per-fit standardisation from the training reactions only (T-3; C7 checks .rxn_rows / .pair_rows)."""

    def __init__(self, D, pair_rows):
        self.pair_rows = np.asarray(pair_rows)
        self.rxn_rows = np.unique(np.r_[D.ra[self.pair_rows], D.rb[self.pair_rows]])
        m = D.mask[self.rxn_rows]
        vals = D.num[self.rxn_rows][m]                                 # [atoms, m]
        self.mu, self.sd = vals.mean(0), vals.std(0)
        self.sd[self.sd < C.SCALE_EPS] = 1.0
        g = D.glob[self.rxn_rows]
        self.gmu, self.gsd = g.mean(0), g.std(0)
        self.gsd[self.gsd < C.SCALE_EPS] = 1.0
        self.sy = D.y[self.pair_rows].std(0)                            # ddof 0 (= d1_train.Scale.sy)

    def tensors(self, D, dtype=torch.float32, device=DEV):
        num = (D.num - self.mu) / self.sd * D.mask[..., None]
        x = np.concatenate([D.onehot, D.bin, num], -1)
        g = (D.glob - self.gmu) / self.gsd
        return dict(D.static(dtype, device), x=torch.tensor(x, dtype=dtype, device=device),
                    glob=torch.tensor(g, dtype=dtype, device=device))


def batch(T, ra, rb):
    """reaction batch for pairs (ra[i], rb[i]) (reaction rows): unique rows, trimmed to the batch's largest reaction."""
    ra, rb = torch.as_tensor(ra, device=T["x"].device), torch.as_tensor(rb, device=T["x"].device)
    U, inv = torch.unique(torch.cat([ra, rb]), return_inverse=True)
    nmax = int(T["n"][U].max())
    R = dict(x=T["x"][U][:, :nmax], pos=T["pos"][U][:, :nmax], mask=T["mask"][U][:, :nmax],
             etype=T["etype"][U][:, :nmax, :nmax].long(), frag=T["frag"][U][:, :nmax].long(), glob=T["glob"][U])
    return R, inv[:len(ra)], inv[len(ra):]


def orient(ra, rb, sw):
    """order augmentation: pairs with sw True become (b, a); returns (ra', rb', sign) and y' = sign * y."""
    return np.where(sw, rb, ra), np.where(sw, ra, rb), np.where(sw, -1.0, 1.0)


def loss_fn(yhat, y, sy):
    l1 = (yhat - y).abs().mean()
    kc = yhat * sy
    pen = ((kc[:, F.I_BAR] - kc[:, F.I_CH9].sum(1)).abs() / sy[F.I_BAR]).mean()
    return l1 + F.LAMBDA * pen, l1, pen


@torch.no_grad()
def predict(model, T, ra, rb, bs=256):
    """standardised predictions [n, 10] for the pairs (ra, rb), eval mode."""
    model.eval()
    out = []
    for s in range(0, len(ra), bs):
        R, ia, ib = batch(T, ra[s:s + bs], rb[s:s + bs])
        out.append(model(R, ia, ib).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(F.TARGETS)))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def epoch_batches(rng, ra_all, rb_all, ytr, batch_size):
    """one epoch of training batches: random pair order + random (a, b) / (b, a) order per pair (M-1), y sign with it.
    ra_all / rb_all: reaction rows of the training pairs, ytr: their standardised y (torch [n, 10])."""
    n = len(ra_all)
    perm, swap = rng.permutation(n), rng.random(n) < 0.5
    for s in range(0, n, batch_size):
        b = perm[s:s + batch_size]
        ra, rb, sign = orient(ra_all[b], rb_all[b], swap[b])
        y = ytr[torch.as_tensor(b, device=ytr.device)] * torch.as_tensor(sign[:, None], dtype=ytr.dtype, device=ytr.device)
        yield ra, rb, y


def fit(D, model_name, cell, variant, tr, val=None, seed=F.SEED, n_epochs=F.MAX_EPOCHS, patience=F.PATIENCE,
        dropout=F.DROPOUT, batch_size=F.BATCH, log=None):
    """one training run -> dict(model, std, sy, history, best_epoch, best_val, best_val_per_target, val_rows)."""
    seed_all(seed)
    rng = np.random.default_rng(seed)
    std = Std(D, tr)
    T = std.tensors(D)
    sy = torch.tensor(std.sy, dtype=torch.float32, device=DEV)
    ytr = torch.tensor(D.y[tr] / std.sy, dtype=torch.float32, device=DEV)
    model = Mo.build(model_name, cell, D.d_in, len(F.TARGETS), D.d_glob, variant, dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=F.LR, weight_decay=F.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=F.MAX_EPOCHS)
    ra_all, rb_all = D.ra[tr], D.rb[tr]
    hist, best = [], dict(val=np.inf, epoch=0, state=None, per_target=None)
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        tl = []
        for ra, rb, y in epoch_batches(rng, ra_all, rb_all, ytr, batch_size):
            R, ia, ib = batch(T, ra, rb)
            loss, l1, pen = loss_fn(model(R, ia, ib), y, sy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append((float(loss), float(l1), float(pen)))
        sched.step()
        rec = dict(epoch=ep, loss=float(np.mean([x[0] for x in tl])), l1=float(np.mean([x[1] for x in tl])),
                   pen=float(np.mean([x[2] for x in tl])), t=time.time() - t0)
        if not all(np.isfinite([x for t3 in tl for x in t3])):
            stop(f"training loss not finite at epoch {ep} ({model_name} {cell} {variant})")       # review F4
        if val is not None:
            yh = predict(model, T, D.ra[val], D.rb[val])
            mae_t = np.abs(yh - D.y[val] / std.sy).mean(0)
            rec["val"] = float(mae_t[F.I_PRIM].mean())
            if not np.isfinite(rec["val"]):
                stop(f"validation MAE not finite at epoch {ep} ({model_name} {cell} {variant})")
            if rec["val"] < best["val"]:
                best = dict(val=rec["val"], epoch=ep, state=copy.deepcopy(model.state_dict()), per_target=mae_t.tolist())
        hist.append(rec)
        if log and (ep % log == 0 or ep == 1):
            print(f"    ep {ep} {rec}", flush=True)
        if val is not None and ep - best["epoch"] >= patience:
            break
    if val is not None and (best["state"] is None or best["epoch"] < 1):
        stop(f"no finite best epoch ({model_name} {cell} {variant})")                            # review F4
    if val is not None:
        model.load_state_dict(best["state"])
    return dict(model=model, std=std, sy=std.sy, T=T, history=hist, best_epoch=best["epoch"] if val is not None else n_epochs,
                best_val=best["val"], best_val_per_target=best["per_target"], seconds=time.time() - t0)


def gate_m1(model, D, std, rows, seed, n=F.ANTISYM_N):
    """gate M-1': |g(a,b) + g(b,a)| and |g(a,a)| (kcal/mol, float64 recomputation) on n random rows of the pair table."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows, size=min(n, len(rows)), replace=False)
    m64 = copy.deepcopy(model).double().eval()
    T64 = std.tensors(D, dtype=torch.float64)
    ra, rb = D.ra[pick], D.rb[pick]
    gab, gba, gaa = (predict(m64, T64, x, y, bs=32) * std.sy for x, y in ((ra, rb), (rb, ra), (ra, ra)))
    anti, selfm = float(np.abs(gab + gba).max()), float(np.abs(gaa).max())
    return dict(n=len(pick), anti_max=anti, self_max=selfm, ok=bool(anti <= F.ANTISYM_TOL and selfm <= F.ANTISYM_TOL))


# ================================================================ provenance / gates
def versions():
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                numpy=np.__version__, pandas=pd.__version__)


ENV_PREFIX = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2")
TORCH_PIN = "2.2.1+cu118"


def require_gpu():
    """exit 3 unless running in env_d1fu2 (pinned torch) on a CUDA device (smoke / tests / Phase M never fall back to CPU)."""
    bad = []
    if not torch.cuda.is_available():
        bad.append("no CUDA device visible (submit with --gres=gpu:a6000:1 on gpu4 / gpu5)")
    if torch.__version__ != TORCH_PIN:
        bad.append(f"torch {torch.__version__} != {TORCH_PIN}")
    if Path(sys.prefix).resolve() != ENV_PREFIX.resolve():
        bad.append(f"python prefix {sys.prefix} is not env_d1fu2")
    if bad:
        print("STOP: " + "; ".join(bad), file=sys.stderr)
        sys.exit(3)


def gate_add2():
    """Phase A gate (= d1_split.gate_add1): ADD2 md / json committed, unchanged, md sha256 = json md_sha256, and the
    Q / T / splits sha256 recorded in the json equal to the files now; exits 3 otherwise."""
    def git(*a):
        return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (ADD2_MD, ADD2_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(ADD2_JSON.read_text())
        if F.sha256(ADD2_MD) != rec.get("md_sha256"):
            bad.append("PREREG_D1PAIR_ADD2.md sha256 differs from prereg_d1pair_add2.json md_sha256")
        need = {str(F.FU2_RES / f) for f in ("Q_report.json", "Q_manifest.csv", "T_report.json", "T_manifest.csv",
                                            "S_splits.json")} | {str(F.SPLITS / "pairs.parquet")}
        if not need <= set(rec.get("inputs_sha256", {})):
            bad.append(f"prereg_d1pair_add2.json inputs_sha256 lacks {sorted(need - set(rec.get('inputs_sha256', {})))}")
        for path, sha in rec.get("inputs_sha256", {}).items():
            if not Path(path).is_file() or F.sha256(path) != sha:
                bad.append(f"input {path} changed or missing")
    if bad:
        print("PREREG ADD2 gate (FU2 Phase A: commit the addendum before any Phase M training):\n  " + "\n  ".join(bad),
              file=sys.stderr)
        sys.exit(3)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def unit_list(split="T1"):
    return [dict(split=split, model=m, variant=v, fold=k) for m in Mo.MODELS for v in Mo.VARIANTS for k in range(NF[split])]


def uname(u):
    return f"{u['split']}__{u['model']}__{u['variant'].replace('+', '')}__f{u['fold']}"


CODE_FILES = ("models.py", "train.py", "fu2_common.py", "tokens.py")          # review F3
# hashed ONCE at import: the code this process actually runs (an edit during a 48 h job cannot relabel later units)
CODE_SHA = {**{f: F.sha256(HERE / f) for f in CODE_FILES}, "d1pair_common.py": F.sha256(Path(C.__file__))}


def fingerprint(D, u, sp):
    return dict(q_report=D.q_report_sha, t_report=D.t_report_sha,
                code_sha256=CODE_SHA,
                s_splits=D.s_splits_sha,                                              # incl. the inner-fold lists (read at Data())
                t_manifest=D.trep["manifest_sha256"], pairs=D.srep["pairs_sha256"], split_pairs=sp["pairs_sha"],
                grid=Mo.GRID[u["model"]], code=F.CODE_VERSION, add2=F.sha256(ADD2_MD) if ADD2_MD.is_file() else None,
                rules=dict(lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                           patience=F.PATIENCE, seeds=F.FINAL_SEEDS), **versions())


def save_torch(path, obj):
    F.assert_path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_unit(D, u, root=F.UNITS, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50):
    """tune (4 cells, first inner fold) -> final (seeds, best epoch) -> predictions + gate M-1' for one unit."""
    root.mkdir(parents=True, exist_ok=True)
    name = uname(u)
    sp = D.split(u["split"], u["fold"])
    fp = fingerprint(D, u, sp)
    js = root / f"{name}.json"
    if js.is_file():
        old = json.loads(js.read_text())
        if old.get("fingerprint") != fp:
            print(f"STOP: cache {js} has a different fingerprint", file=sys.stderr); sys.exit(3)
        return old
    grid = Mo.GRID[u["model"]] if cells is None else [Mo.GRID[u["model"]][c] for c in cells]
    tune = []
    for ci, cell in enumerate(grid):
        r = fit(D, u["model"], cell, u["variant"], sp["inner_train"], sp["inner_val"], seed=F.SEED, n_epochs=n_epochs,
                log=log)
        g = gate_m1(r["model"], D, r["std"], sp["test"], seed=F.SEED + ci)   # inputs only, no labels (M-1)
        tune.append(dict(cell=cell, best_epoch=r["best_epoch"], val=r["best_val"], val_per_target=r["best_val_per_target"],
                         epochs_run=len(r["history"]), seconds=r["seconds"], gate_M1=g))
        print(f"  {name} cell {cell}: val {r['best_val']:.4f} at epoch {r['best_epoch']} ({r['seconds']:.0f} s) gate {g}",
              flush=True)
        if not g["ok"]:
            print(f"STOP: gate M-1' failed ({name}, tuning cell {cell}): {g}", file=sys.stderr); sys.exit(3)
    best = min(range(len(tune)), key=lambda i: (tune[i]["val"], i))
    cell, ep = tune[best]["cell"], tune[best]["best_epoch"]
    if not (isinstance(ep, int) and ep >= 1 and np.isfinite(tune[best]["val"])):
        stop(f"{name}: chosen cell {cell} has best_epoch {ep} / validation {tune[best]['val']}")   # review F4
    preds, finals = [], []
    te = sp["test"]
    for s in seeds:
        r = fit(D, u["model"], cell, u["variant"], sp["train"], None, seed=s, n_epochs=ep, log=log)
        g = gate_m1(r["model"], D, r["std"], te, seed=s)
        if not g["ok"]:
            print(f"STOP: gate M-1' failed ({name}, seed {s}): {g}", file=sys.stderr); sys.exit(3)
        preds.append(predict(r["model"], r["T"], D.ra[te], D.rb[te]) * r["sy"])
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(),
                           history_tail=r["history"][-3:]))
        save_torch(root / f"{name}__s{s - F.SEED}.pt", dict(state=r["model"].state_dict(), cell=cell, unit=u,
                                                              std=dict(mu=r["std"].mu, sd=r["std"].sd, gmu=r["std"].gmu,
                                                                       gsd=r["std"].gsd, sy=r["sy"])))
    Y = np.stack(preds)                                                  # [seeds, n_test, 10] kcal/mol
    rows = []
    for j, t in enumerate(F.TARGETS):
        rows.append(pd.DataFrame(dict(pair_id=D.P.index[te], ref_kind=D.P.ref_kind.to_numpy()[te], fold=u["fold"],
                                      split=u["split"], model=u["model"], variant=u["variant"], target=t,
                                      y=D.y[te, j], yhat=Y[:, :, j].mean(0), yhat_seed_sd=Y[:, :, j].std(0))))
    pq = root / f"{name}.parquet"
    pred = pd.concat(rows, ignore_index=True)
    F.write_atomic(pq, lambda t: pred.to_parquet(t, index=False))
    rec = dict(unit=u, name=name, fingerprint=fp, tune=tune, chosen=dict(cell=cell, best_epoch=ep, index=best),
               final=finals, n_train=len(sp["train"]), n_test=len(te), n_inner_train=len(sp["inner_train"]),
               n_inner_val=len(sp["inner_val"]), dropped_no_atoms=sp["dropped_no_atoms"], parquet_sha256=F.sha256(pq))
    F.write_json(js, rec)
    return rec


# ================================================================ commands
def cmd_smoke():
    """M-3 smoke: T1 fold 0, every config: tuning cell 0 for 2 epochs (validation each epoch), final 1 seed 2 epochs,
    gate M-1' on both; timings -> results/fu2/C_smoke.json. Writes only $FU2_SCR/smoke (never read by Phase M).
    Runs the smallest and the largest grid cell of every model (memory / time check of the whole grid)."""
    require_gpu()
    D = Data()
    out = dict(versions=versions(), device=str(DEV), configs={})
    import shutil
    if F.SMOKE.exists():
        shutil.rmtree(F.SMOKE)
    ok = True
    for m in Mo.MODELS:
        for v in Mo.VARIANTS:
            u = dict(split="T1", model=m, variant=v, fold=0)
            t0 = time.time()
            rec = run_unit(D, u, root=F.SMOKE, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, len(Mo.GRID[m]) - 1], log=1)
            n_params = sum(p.numel() for p in Mo.build(m, Mo.GRID[m][0], D.d_in, 10, D.d_glob, v).parameters())
            g_ok = all(t["gate_M1"]["ok"] for t in rec["tune"]) and all(f["gate_M1"]["ok"] for f in rec["final"])
            ok &= g_ok
            eps = [t["seconds"] / max(1, t["epochs_run"]) for t in rec["tune"]]       # s / epoch, smallest and largest cell
            n_big = sum(p.numel() for p in Mo.build(m, Mo.GRID[m][-1], D.d_in, 10, D.d_glob, v).parameters())
            out["configs"][uname(u)] = dict(n_params=dict(smallest=n_params, largest=n_big), seconds=time.time() - t0,
                                            tune_epoch_seconds=dict(smallest=eps[0], largest=eps[-1]),
                                            gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                                            gate_M1_tune=[t["gate_M1"] for t in rec["tune"]],
                                            gate_M1_final=rec["final"][0]["gate_M1"],
                                            val=[t["val"] for t in rec["tune"]], n_train=rec["n_train"], n_test=rec["n_test"],
                                            n_inner_train=rec["n_inner_train"], n_inner_val=rec["n_inner_val"],
                                            dropped_no_atoms=rec["dropped_no_atoms"], ok=g_ok)
            torch.cuda.reset_peak_memory_stats()
            print(f"[smoke] {uname(u)}: {out['configs'][uname(u)]}", flush=True)
    out["ok"] = bool(ok)
    # upper bound for Phase M (no early stopping): per unit 4 tuning cells (middle cells ~ mean of the two measured) on
    # the inner-training pairs + 5 final seeds on all training pairs (x 4/3 pairs) with the largest cell, 300 epochs each
    ub = 0.0
    for c in out["configs"].values():
        s, l = c["tune_epoch_seconds"]["smallest"], c["tune_epoch_seconds"]["largest"]
        ub += (s + l + (s + l)) * F.MAX_EPOCHS + 5 * l * 4 / 3 * F.MAX_EPOCHS
    out["phase_M_upper_bound_gpu_hours_T1"] = float(ub * 5 / 3600)                  # x 5 outer folds
    out["provenance"] = dict(code_sha256=CODE_SHA, q_report=D.q_report_sha, t_report=D.t_report_sha,
                             s_splits=D.s_splits_sha, code=F.CODE_VERSION)
    F.write_json(F.FU2_RES / "C_smoke.json", out)
    return 0 if ok else 3


def cmd_collect():
    """M-3: every finished T1 unit -> results/fu2/M_predictions.parquet (+ M_report.json); exit 2 while units are missing."""
    names = [uname(u) for u in unit_list("T1")]
    missing = [n for n in names if not (F.UNITS / f"{n}.json").is_file()]
    if missing:
        print(f"GATE: {len(missing)} units missing: {missing[:5]}", file=sys.stderr)
        return 2
    recs = {n: json.loads((F.UNITS / f"{n}.json").read_text()) for n in names}
    for n, r in recs.items():
        if F.sha256(F.UNITS / f"{n}.parquet") != r["parquet_sha256"]:
            print(f"STOP: {n}.parquet changed after its json", file=sys.stderr); return 3
        if not all(t["gate_M1"]["ok"] for t in r["tune"]) or not all(f["gate_M1"]["ok"] for f in r["final"]):
            print(f"STOP: gate M-1' failed in {n}", file=sys.stderr); return 3
    D = Data()                                             # review: every unit must carry the CURRENT fingerprint
    for n, r in recs.items():
        u = r["unit"]
        now = fingerprint(D, u, D.split(u["split"], u["fold"]))
        if {k: v for k, v in r["fingerprint"].items() if k != "gpu"} != {k: v for k, v in now.items() if k != "gpu"}:   # collect may run on CPU
            print(f"STOP: {n} was computed with other code / inputs than the current ones", file=sys.stderr); return 3
    pred = pd.concat([pd.read_parquet(F.UNITS / f"{n}.parquet") for n in names], ignore_index=True)
    F.write_atomic(F.FU2_RES / "M_predictions.parquet", lambda t: pred.to_parquet(t, index=False))
    rep = dict(n_units=len(names),
               units={n: dict(chosen=r["chosen"], n_train=r["n_train"], n_test=r["n_test"],
                              dropped_no_atoms=r["dropped_no_atoms"],
                              tune=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"],
                                         val_per_target=t["val_per_target"]) for t in r["tune"]]) for n, r in recs.items()},
               predictions_sha256=F.sha256(F.FU2_RES / "M_predictions.parquet"))
    F.write_json(F.FU2_RES / "M_report.json", rep)
    return 0


def cmd_run(i, n):
    require_gpu()
    add2 = gate_add2()
    print(f"[M] ADD2 ok (git {add2['git_head']}), {versions()}", flush=True)
    D = Data()
    us = unit_list("T1")[int(i)::int(n)]
    for u in us:
        t0 = time.time()
        rec = run_unit(D, u)
        print(f"[M] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["smoke"]:
        sys.exit(cmd_smoke())
    if a[:1] == ["run"]:
        sys.exit(cmd_run(a[1], a[2]))
    if a[:1] == ["collect"]:
        sys.exit(cmd_collect())
    if a[:1] == ["list"]:
        print("\n".join(uname(u) for u in unit_list("T1"))); sys.exit(0)
    print(__doc__); sys.exit(2)
