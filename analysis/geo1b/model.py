"""GEO-1 refiner: Uni-Mol+ two-track trunk (Lu et al. 2024) + E(3)-equivariant position head. PyTorch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


class Gaussian(nn.Module):
    """Uni-Mol Gaussian distance encoding with per-element-pair affine scaling."""
    def __init__(self, K, n_edge):
        super().__init__()
        self.K = K
        self.means = nn.Parameter(torch.empty(K).uniform_(0, 3))
        self.stds = nn.Parameter(torch.empty(K).uniform_(0, 3))
        self.mul = nn.Embedding(n_edge, 1)
        self.bias = nn.Embedding(n_edge, 1)
        nn.init.constant_(self.mul.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

    def forward(self, dist, et):
        x = self.mul(et).squeeze(-1) * dist + self.bias(et).squeeze(-1)
        std = self.stds.abs() + 1e-2
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.means) / std) ** 2) / (math.sqrt(2 * math.pi) * std)


class PairBiasAttention(nn.Module):
    def __init__(self, dx, dp, H, drop):
        super().__init__()
        self.H, self.dh = H, dx // H
        self.qkv, self.out = nn.Linear(dx, 3 * dx), nn.Linear(dx, dx)
        self.bias = nn.Sequential(nn.LayerNorm(dp), nn.Linear(dp, H, bias=False))
        self.drop = nn.Dropout(drop)

    def forward(self, x, p, mask):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.H, self.dh).unbind(2)
        lg = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh) + self.bias(p).permute(0, 3, 1, 2)
        lg = lg.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.drop(lg.softmax(-1))
        o = torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, N, -1)
        return self.out(o) * mask[..., None]


class OuterProduct(nn.Module):
    def __init__(self, dx, dp, do):
        super().__init__()
        self.ln, self.a, self.b, self.o = nn.LayerNorm(dx), nn.Linear(dx, do), nn.Linear(dx, do), nn.Linear(do * do, dp)

    def forward(self, x, pm):
        x = self.ln(x)
        op = torch.einsum("bic,bjd->bijcd", self.a(x), self.b(x)).flatten(-2)
        return self.o(op) * pm[..., None]


class TriangularUpdate(nn.Module):
    """outgoing + incoming multiplicative updates merged into one module (Uni-Mol+)."""
    def __init__(self, dp, dt):
        super().__init__()
        self.ln = nn.LayerNorm(dp)
        self.a, self.ag, self.b, self.bg = (nn.Linear(dp, dt) for _ in range(4))
        self.ln_o, self.o, self.g = nn.LayerNorm(dt), nn.Linear(dt, dp), nn.Linear(dp, dp)

    def forward(self, p, pm):
        z, m = self.ln(p), pm[..., None].to(p.dtype)
        a = torch.sigmoid(self.ag(z)) * self.a(z) * m
        b = torch.sigmoid(self.bg(z)) * self.b(z) * m
        t = torch.einsum("bikc,bjkc->bijc", a, b) + torch.einsum("bkic,bkjc->bijc", a, b)
        return torch.sigmoid(self.g(z)) * self.o(self.ln_o(t)) * m


class Block(nn.Module):
    def __init__(self, dx, dp, H, do, dt, drop):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(dx), PairBiasAttention(dx, dp, H, drop)
        self.ln2, self.ffx = nn.LayerNorm(dx), FFN(dx, dx, drop)
        self.opm, self.tri = OuterProduct(dx, dp, do), TriangularUpdate(dp, dt)
        self.lnp, self.ffp = nn.LayerNorm(dp), FFN(dp, dp, drop)

    def forward(self, x, p, mask, pm):
        x = x + self.attn(self.ln1(x), p, mask)
        x = x + self.ffx(self.ln2(x)) * mask[..., None]
        p = p + self.opm(x, pm)
        p = p + self.tri(p, pm)
        p = p + self.ffp(self.lnp(p)) * pm[..., None]
        return x, p


class PosHead(nn.Module):
    """delta_i = sum_h w_h sum_j a_hij v_hj (r_i - r_j)/|r_i - r_j|  -> E(3)-equivariant, translation invariant.
    The last layer starts at zero, so an untrained model returns its input (identity map)."""
    def __init__(self, dx, dp, H):
        super().__init__()
        self.H, self.dh = H, dx // H
        self.q, self.k, self.v = nn.Linear(dx, dx), nn.Linear(dx, dx), nn.Linear(dx, H)
        self.bias = nn.Linear(dp, H)
        self.w = nn.Linear(H, 1, bias=False)
        nn.init.zeros_(self.w.weight)

    def forward(self, x, p, pos, mask):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.H, self.dh)
        k = self.k(x).view(B, N, self.H, self.dh)
        lg = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh) + self.bias(p).permute(0, 3, 1, 2)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        lg = lg.masked_fill(~mask[:, None, None, :] | eye[None, None], float("-inf"))
        a = lg.softmax(-1)
        v = self.v(x).permute(0, 2, 1)                                   # (B, H, N)
        rel = pos[:, :, None, :] - pos[:, None, :, :]                    # r_i - r_j
        u = rel / torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-12)
        vec = torch.einsum("bhij,bijc->bihc", a * v[:, :, None, :], u)  # (B, N, H, 3)
        return self.w(vec.transpose(-1, -2)).squeeze(-1) * mask[..., None]


class GeoRefiner(nn.Module):
    def __init__(self, n_types=14, L=6, dx=256, dp=64, H=8, K=128, do=16, dt=32, drop=0.1, spd_max=9):
        super().__init__()
        self.V = n_types
        self.atom = nn.Embedding(n_types, dx)
        self.frag, self.formf, self.srcf, self.chg = (nn.Embedding(3, dx), nn.Embedding(2, dx),
                                                      nn.Embedding(2, dx), nn.Embedding(7, dx))
        self.gauss = Gaussian(K, n_types * n_types)
        self.p_lin = nn.Linear(K, dp)
        self.spd = nn.Embedding(spd_max + 1, dp)
        self.bond = nn.Embedding(3, dp)
        self.blocks = nn.ModuleList(Block(dx, dp, H, do, dt, drop) for _ in range(L))
        self.ln_f = nn.LayerNorm(dx)
        self.head = PosHead(dx, dp, H)

    def pair_init(self, pos, b):
        dist = torch.sqrt(((pos[:, :, None] - pos[:, None]) ** 2).sum(-1) + 1e-12)
        et = b["types"][:, :, None] * self.V + b["types"][:, None, :]
        return self.p_lin(self.gauss(dist, et)) + self.spd(b["spd"]) + self.bond(b["bond"])

    def forward(self, b, n_pass):
        mask = b["mask"]
        pm = mask[:, :, None] & mask[:, None, :]
        x0 = (self.atom(b["types"]) + self.frag(b["frag"]) + self.formf(b["form"])
              + self.srcf(b["src"])[:, None] + self.chg(b["chg"] + 3)[:, None])
        pos = b["x0"]
        for it in range(n_pass):
            p = self.pair_init(pos, b) * pm[..., None]
            x = x0
            for blk in self.blocks:
                x, p = blk(x, p, mask, pm)
            new = pos + self.head(self.ln_f(x), p, pos, mask)
            pos = new if it == n_pass - 1 else new.detach()      # loss only on the last pass (Uni-Mol+)
        return pos
