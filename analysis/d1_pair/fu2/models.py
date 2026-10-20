"""FU2 M-2 — three atom-token pair models (pure PyTorch, padded reactions + masks; no PyG).

A reaction batch R is a dict of padded tensors (n_r reactions, N = max atoms in the batch):
    x     [n_r, N, D]   atom tokens (element one-hot | binary flags | standardised numeric columns)
    pos   [n_r, N, 3]   TS Cartesian coordinates (A); models use inter-atomic distances only
    mask  [n_r, N]      True for real atoms
    etype [n_r, N, N]   0 none, 1 covalent, 2 formed (symmetric; T-2 graph)
    frag  [n_r, N]      0 = A (dipole), 1 = B (dipolarophile)
    glob  [n_r, G]      standardised global token (X182 row; used only by variant "A+G")
A pair batch indexes two reaction rows per pair: ia, ib [B] -> g(a, b) [B, n_out] in standardised target units.

Antisymmetry (exact by construction):
    AW   g = E(a) - E(b)                      (E = sum of per-atom contributions (+ phi(glob) for A+G))
    GNN  g = 1/2 [h(r_a, r_b) - h(r_b, r_a)]  (r = reaction readout, h = MLP on the concatenation)
    TF   g = 1/2 [h(c_a, c_b) - h(c_b, c_a)]  (c = pooled tokens after self- and shared cross-attention)
Both orderings of h are evaluated from the same pooled vectors in one call, so g(b, a) = -g(a, b) and g(a, a) = 0 hold
to rounding in eval mode. No op mixes batch rows (no BatchNorm), so each pair only sees its own two reactions.
"""
from __future__ import annotations

import math

import torch
from torch import nn

NEG = -1e9                                           # additive attention mask (finite: no NaN on padded query rows)


def rbf(d, n, d_max):
    """Gaussian radial basis on [0, d_max], n centres; d [...] -> [..., n]."""
    mu = torch.linspace(0.0, d_max, n, device=d.device, dtype=d.dtype)
    gamma = 1.0 / (mu[1] - mu[0]) ** 2
    return torch.exp(-gamma * (d.unsqueeze(-1) - mu) ** 2)


def cosine_cutoff(d, rc):
    """0.5 (cos(pi d / rc) + 1) inside rc, exactly 0 outside."""
    return torch.where(d < rc, 0.5 * (torch.cos(math.pi * d / rc) + 1.0), torch.zeros_like(d))


def pdist(pos):
    """[n_r, N, 3] -> [n_r, N, N] Euclidean distances (gradient-safe at d = 0)."""
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-12) * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))


def mlp(d_in, d_hid, d_out, dropout, n_hidden=1):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hid), nn.SiLU(), nn.Dropout(dropout)]
        d = d_hid
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def masked_sum(h, mask):
    return (h * mask.unsqueeze(-1).to(h.dtype)).sum(1)


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class AttnPool(nn.Module):
    """softmax(w . h_i) over the real atoms of one reaction, then sum_i a_i h_i."""

    def __init__(self, width):
        super().__init__()
        self.w = nn.Linear(width, 1)

    def forward(self, h, mask):
        s = self.w(h).squeeze(-1).masked_fill(~mask, NEG)
        a = torch.softmax(s, dim=1) * mask.to(h.dtype)
        return (a.unsqueeze(-1) * h).sum(1)


class PairModel(nn.Module):
    """common pair interface: forward(R, ia, ib) -> g [B, n_out]."""
    antisym = "half-difference"

    def forward(self, R, ia, ib):
        raise NotImplementedError


# ---------------------------------------------------------------- AW: atom-wise additive (SchNet-type cfconv)
class CFConv(nn.Module):
    def __init__(self, width, n_rbf, rc, dropout):
        super().__init__()
        self.rc, self.n_rbf = rc, n_rbf
        self.filt = nn.Sequential(nn.Linear(n_rbf, width), nn.SiLU(), nn.Linear(width, width))
        self.lin_in = nn.Linear(width, width, bias=False)
        self.upd = mlp(width, width, width, dropout)

    def forward(self, h, d, nb):
        # nb [n_r, N, N]: j != i, both real, d_ij < rc; the cosine cutoff is exactly 0 outside rc
        w = self.filt(rbf(d, self.n_rbf, self.rc)) * (cosine_cutoff(d, self.rc) * nb.to(d.dtype)).unsqueeze(-1)
        v = (w * self.lin_in(h).unsqueeze(1)).sum(2)                     # v_i = sum_j W(d_ij) * x_j
        return h + self.upd(v)


class AW(PairModel):
    antisym = "difference"

    def __init__(self, d_in, n_out, width=64, n_layers=1, d_glob=0, rc=5.0, n_rbf=32, dropout=0.1):
        super().__init__()
        self.rc = rc
        self.embed = nn.Sequential(nn.Linear(d_in, width), nn.SiLU())
        self.layers = nn.ModuleList([CFConv(width, n_rbf, rc, dropout) for _ in range(n_layers)])
        self.out = mlp(width, width, n_out, dropout)
        self.phi = mlp(d_glob, width, n_out, dropout) if d_glob else None

    def atom_contrib(self, R):
        """per-atom contributions eps [n_r, N, n_out] (0 on padded atoms)."""
        mask, d = R["mask"], pdist(R["pos"])
        n = mask.shape[1]
        nb = mask.unsqueeze(2) & mask.unsqueeze(1) & (d < self.rc) & ~torch.eye(n, dtype=torch.bool, device=d.device)
        h = self.embed(R["x"])
        for layer in self.layers:
            h = layer(h, d, nb)
        return self.out(h) * mask.unsqueeze(-1).to(h.dtype)

    def energy(self, R):
        E = self.atom_contrib(R).sum(1)
        if self.phi is not None:
            E = E + self.phi(R["glob"])
        return E

    def forward(self, R, ia, ib):
        E = self.energy(R)
        return E[ia] - E[ib]


# ---------------------------------------------------------------- GNN: message passing on the T-2 graph
class MPLayer(nn.Module):
    """m_i = sum_j A_ij psi(W_h h_j + W_e e_ij), h_i <- LN(h_i + upd([h_i, m_i])); e_ij = type embedding + MLP(RBF(d_ij)).
    psi's first linear map on [h_j, e_ij] is split into W_h h_j (per node) + W_e e_ij (per edge) to keep memory O(N^2 W)."""

    def __init__(self, width, n_rbf, d_max, dropout):
        super().__init__()
        self.n_rbf, self.d_max = n_rbf, d_max
        self.etype = nn.Embedding(3, width)                               # 0 none (masked out), 1 covalent, 2 formed
        self.edge = nn.Sequential(nn.Linear(n_rbf, width), nn.SiLU(), nn.Linear(width, width))
        self.w_h, self.w_e = nn.Linear(width, width), nn.Linear(width, width, bias=False)
        self.msg_out = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.upd = mlp(2 * width, width, width, dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, d, etype, adj):
        e = self.etype(etype) + self.edge(rbf(d, self.n_rbf, self.d_max))   # [n_r, N(i), N(j), W]
        pre = self.w_h(h).unsqueeze(1) + self.w_e(e)                       # W_h h_j broadcast over i
        m = (self.msg_out(pre) * adj.unsqueeze(-1).to(h.dtype)).sum(2)
        return self.norm(h + self.upd(torch.cat([h, m], -1)))


class GNN(PairModel):
    def __init__(self, d_in, n_out, width=64, n_layers=4, d_glob=0, n_rbf=32, d_max=4.0, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_in, width), nn.SiLU())
        self.layers = nn.ModuleList([MPLayer(width, n_rbf, d_max, dropout) for _ in range(n_layers)])
        self.pool = AttnPool(width)
        self.gproj = nn.Linear(d_glob, width) if d_glob else None
        d_r = 2 * width + (width if d_glob else 0)
        self.head = mlp(2 * d_r, 2 * width, n_out, dropout, n_hidden=2)

    def readout(self, R):
        mask, d, et = R["mask"], pdist(R["pos"]), R["etype"]
        adj = (et > 0) & mask.unsqueeze(2) & mask.unsqueeze(1)
        h = self.embed(R["x"])
        for layer in self.layers:
            h = layer(h, d, et, adj)
        r = [masked_sum(h, mask), self.pool(h, mask)]
        if self.gproj is not None:
            r.append(self.gproj(R["glob"]))
        return torch.cat(r, -1)

    def forward(self, R, ia, ib):
        r = self.readout(R)
        ra, rb = r[ia], r[ib]
        return 0.5 * (self.head(torch.cat([ra, rb], -1)) - self.head(torch.cat([rb, ra], -1)))


# ---------------------------------------------------------------- TF: Transformer with geometric attention bias
class BiasedMHA(nn.Module):
    """multi-head attention with an additive per-head bias [n, H, Nq, Nk] and a key mask [n, Nk]."""

    def __init__(self, width, n_heads, dropout):
        super().__init__()
        assert width % n_heads == 0
        self.h, self.dk = n_heads, width // n_heads
        self.q, self.k, self.v = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
        self.o = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, xq, xk, key_mask, bias=None, return_attn=False):
        n, nq, w = xq.shape
        nk = xk.shape[1]
        q = self.q(xq).view(n, nq, self.h, self.dk).transpose(1, 2)
        k = self.k(xk).view(n, nk, self.h, self.dk).transpose(1, 2)
        v = self.v(xk).view(n, nk, self.h, self.dk).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if bias is not None:
            s = s + bias
        s = s.masked_fill(~key_mask[:, None, None, :], NEG)
        a = torch.softmax(s, dim=-1)
        out = (self.drop(a) @ v).transpose(1, 2).reshape(n, nq, w)
        return (self.o(out), a) if return_attn else self.o(out)


class TFBlock(nn.Module):
    def __init__(self, width, n_heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.att = BiasedMHA(width, n_heads, dropout)
        self.ff = mlp(width, 2 * width, width, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, bias=None, xkv=None, kv_mask=None):
        xn = self.n1(x)
        kv = xn if xkv is None else self.n1(xkv)                          # cross-attention: keys/values = partner tokens
        x = x + self.drop(self.att(xn, kv, mask if kv_mask is None else kv_mask, bias))
        return x + self.drop(self.ff(self.n2(x)))


class TF(PairModel):
    def __init__(self, d_in, n_out, width=64, n_layers=2, d_glob=0, n_heads=4, n_rbf=32, d_max=10.0, dropout=0.1):
        super().__init__()
        self.h, self.n_rbf, self.d_max = n_heads, n_rbf, d_max
        self.embed = nn.Linear(d_in, width)
        self.gtok = nn.Linear(d_glob, width) if d_glob else None
        self.b_rbf = nn.Linear(n_rbf, n_heads, bias=False)               # per-head distance bias
        self.b_same = nn.Parameter(torch.zeros(n_heads))                  # same fragment
        self.b_formed = nn.Parameter(torch.zeros(n_heads))                # forming-bond pair
        self.b_glob = nn.Parameter(torch.zeros(n_heads))                  # any pair involving the global token
        self.blocks = nn.ModuleList([TFBlock(width, n_heads, dropout) for _ in range(n_layers)])
        self.cross = TFBlock(width, n_heads, dropout)                      # one block, shared for a->b and b->a
        self.norm = nn.LayerNorm(width)
        self.pool = AttnPool(width)
        self.head = mlp(4 * width, 2 * width, n_out, dropout, n_hidden=2)

    def tokens(self, R):
        """token states [n_r, T, W], token mask [n_r, T], self-attention bias [n_r, H, T, T]."""
        mask, d = R["mask"], pdist(R["pos"])
        same = (R["frag"].unsqueeze(2) == R["frag"].unsqueeze(1)).to(d.dtype)
        formed = (R["etype"] == 2).to(d.dtype)
        bias = (self.b_rbf(rbf(d, self.n_rbf, self.d_max)).permute(0, 3, 1, 2)
                + self.b_same[None, :, None, None] * same[:, None] + self.b_formed[None, :, None, None] * formed[:, None])
        x = self.embed(R["x"])
        if self.gtok is not None:                                          # global token appended as token N
            n_r, n = mask.shape
            x = torch.cat([x, self.gtok(R["glob"]).unsqueeze(1)], 1)
            mask = torch.cat([mask, torch.ones(n_r, 1, dtype=torch.bool, device=mask.device)], 1)
            full = self.b_glob[None, :, None, None].expand(n_r, self.h, n + 1, n + 1).clone()
            full[:, :, :n, :n] = bias
            bias = full
        for blk in self.blocks:
            x = blk(x, mask, bias)
        return x, mask

    def pooled(self, x, mask):
        x = self.norm(x)
        return torch.cat([masked_mean(x, mask), self.pool(x, mask)], -1)

    def cross_pair(self, R, ia, ib):
        x, mask = self.tokens(R)
        xa, xb, ma, mb = x[ia], x[ib], mask[ia], mask[ib]
        ya = self.cross(xa, ma, xkv=xb, kv_mask=mb)                       # a's tokens attend to b
        yb = self.cross(xb, mb, xkv=xa, kv_mask=ma)                       # b's tokens attend to a (same weights)
        return self.pooled(ya, ma), self.pooled(yb, mb)

    def forward(self, R, ia, ib):
        ca, cb = self.cross_pair(R, ia, ib)
        return 0.5 * (self.head(torch.cat([ca, cb], -1)) - self.head(torch.cat([cb, ca], -1)))


MODELS = {"AW": AW, "GNN": GNN, "TF": TF}
# grid (4 cells each): layers x width, FU2 M-2
GRID = {"AW": [dict(n_layers=L, width=W) for L in (1, 2) for W in (64, 128)],
        "GNN": [dict(n_layers=L, width=W) for L in (4, 6) for W in (64, 128)],
        "TF": [dict(n_layers=L, width=W) for L in (2, 3) for W in (64, 128)]}
VARIANTS = ("A", "A+G")


def build(model, cell, d_in, n_out, d_glob, variant, dropout=0.1):
    return MODELS[model](d_in, n_out, d_glob=(d_glob if variant == "A+G" else 0), dropout=dropout, **cell)
