"""abs_model.py — ABS3 AbsNet: one reaction -> its 4 absolute values (docs/specs/ABS3_MODELS.md 0-3, 3).

The encoder is fu2 models.build(model, cell, d_in=29, n_out=4, d_glob=182, "A+G", dropout) unchanged (enc). Its pair head
(`head`) and the TF cross-attention block (`cross`) are not used: a reaction has no partner. Reaction vector r:
  GNN  enc.readout(R)                       = [atom sum | attention pool | global projection]   d_r = 3 * width
  TF   enc.pooled(*enc.tokens(R))           = [mean | attention pool] of the self-attention tokens  d_r = 2 * width
New head = models.mlp(d_r, 2 * width, 4, dropout, n_hidden=2) (the pair head's hidden layers: 2 x (2 * width)).
d_r is checked with a dummy batch (eval mode, no random numbers drawn). forward(R, ia) -> head(r(R))[ia] [B, 4] in
standardised units. used_parameters() = every parameter except enc.head and enc.cross (the optimiser gets only these).
"""
from __future__ import annotations

import torch
from torch import nn

import models as Mo  # noqa: E402  (analysis/d1_pair/fu2, on sys.path through data_ch3 / data_lf)

UNUSED_PREFIXES = ("enc.head.", "enc.cross.")


def expected_dr(model, width, d_glob):
    if model == "GNN":
        return 2 * width + (width if d_glob else 0)
    if model == "TF":
        return 2 * width
    raise KeyError(model)


class AbsNet(nn.Module):
    def __init__(self, model, cell, d_in, d_glob, variant="A+G", dropout=0.1, n_out=4):
        super().__init__()
        if model not in ("TF", "GNN"):
            raise ValueError(f"AbsNet model {model}")
        self.model_name = model
        self.cell = dict(cell)
        self.enc = Mo.build(model, cell, d_in, n_out, d_glob, variant, dropout)
        width = int(cell["width"])
        self.d_r = self._probe_dr(d_in, d_glob if variant == "A+G" else 0)
        want = expected_dr(model, width, d_glob if variant == "A+G" else 0)
        if self.d_r != want:
            raise RuntimeError(f"AbsNet {model} {cell}: reaction vector width {self.d_r} != {want}")
        self.head = Mo.mlp(self.d_r, 2 * width, n_out, dropout, n_hidden=2)

    @torch.no_grad()
    def _probe_dr(self, d_in, d_glob):
        was = self.enc.training
        self.enc.eval()
        n = 3
        R = dict(x=torch.zeros(1, n, d_in), pos=torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]]]),
                 mask=torch.ones(1, n, dtype=torch.bool), etype=torch.tensor([[[0, 1, 0], [1, 0, 2], [0, 2, 0]]]),
                 frag=torch.tensor([[0, 0, 1]]), glob=torch.zeros(1, d_glob))
        r = self.rvec(R)
        self.enc.train(was)
        return int(r.shape[-1])

    def rvec(self, R):
        if self.model_name == "GNN":
            return self.enc.readout(R)
        x, mask = self.enc.tokens(R)
        return self.enc.pooled(x, mask)

    def forward(self, R, ia):
        return self.head(self.rvec(R))[ia]

    def used_named_parameters(self):
        return [(k, p) for k, p in self.named_parameters() if not k.startswith(UNUSED_PREFIXES)]

    def used_parameters(self):
        return [p for _, p in self.used_named_parameters()]

    def unused_parameter_names(self):
        return [k for k, _ in self.named_parameters() if k.startswith(UNUSED_PREFIXES)]


def build(model, cell, d_in, d_glob, variant="A+G", dropout=0.1):
    return AbsNet(model, cell, d_in, d_glob, variant, dropout)
