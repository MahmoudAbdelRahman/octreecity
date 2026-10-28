"""Optional link-attention model over any tokens and links (needs torch:
pip install octreecity[torch]).

Each token attends only to its keys (the key table from octreecity.learn):
the key's features plus a learned function of its relative offset (dx, dy,
dz, size) and, optionally, of the link kind. Sizes, inputs and outputs are
settings; nothing about cities is built in.

    from octreecity.attention import LinkModel, fit, predict
    model = LinkModel.from_tokens(tok, n_targets=4, dim=64, layers=6)
    history = fit(model, train_tokens, steps=3000)
    pred = predict(model, test_tok)                # (T, 4) NumPy array

This is the model of the paper's experiments (train_parents.py,
train_oct3d.py) in general form: with n_kinds=0, pos_dims=2 and offset_dim
3 it is the same layer.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint


class LinkAttention(nn.Module):
    """One pre-norm attention layer over fixed key tables, then a
    feed-forward block."""

    def __init__(self, dim, heads=4, offset_dim=4, n_kinds=0, ff_mult=4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be a multiple of heads")
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.rel = nn.Sequential(nn.Linear(offset_dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
        self.kind = nn.Embedding(n_kinds + 1, 2 * dim) if n_kinds else None    # +1: the token itself
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))

    def forward(self, x, idx, valid, offset, kind=None):
        T, D = x.shape
        hd, K = D // self.h, idx.shape[1]
        z = self.n1(x)
        q = self.q(z).view(T, 1, self.h, hd)
        kv = self.kv(z)[idx] + self.rel(offset)
        if self.kind is not None and kind is not None:
            kv = kv + self.kind(kind.long() + 1)
        k, v = kv.view(T, K, 2, self.h, hd).unbind(2)
        att = ((q * k).sum(-1) / math.sqrt(hd)).masked_fill(~valid[..., None], -1e9).softmax(1)
        x = x + self.o((att[..., None] * v).sum(1).reshape(T, D))
        return x + self.ff(self.n2(x))


class LinkModel(nn.Module):
    """features -> linear embedding + sin/cos position code -> layers of
    LinkAttention -> linear head (n_targets per token).

    pos_dims: how many position axes to encode (2 = x, y only); pos_scale:
    metres mapped to one period unit; checkpoint_above: recompute layers in
    the backward pass when a batch has more tokens than this (saves memory).
    """

    def __init__(self, n_features, n_targets, dim=64, heads=4, layers=6, offset_dim=4, n_kinds=0,
                 pos_dims=3, pos_freqs=8, pos_scale=256.0, checkpoint_above=None):
        super().__init__()
        self.pos_dims, self.pos_scale, self.checkpoint_above = pos_dims, pos_scale, checkpoint_above
        self.register_buffer("fr", torch.arange(1, pos_freqs + 1).float() * math.pi)
        self.inp = nn.Linear(n_features, dim)
        self.pos = nn.Linear(2 * pos_dims * pos_freqs, dim) if pos_dims else None
        self.layers = nn.ModuleList(LinkAttention(dim, heads, offset_dim, n_kinds) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_targets))

    @classmethod
    def from_tokens(cls, tok, n_targets=None, **settings):
        """Model sized for a Tokens set (features, offsets, link kinds)."""
        n_targets = n_targets if n_targets is not None else (tok.targets.shape[1] if tok.targets is not None else 1)
        settings.setdefault("n_kinds", len(tok.kinds))
        return cls(tok.features.shape[1], n_targets, offset_dim=tok.offset.shape[2], **settings)

    def forward(self, features, position, idx, valid, offset, kind=None):
        x = self.inp(features)
        if self.pos is not None:
            p = (position[:, :self.pos_dims] / self.pos_scale)[..., None] * self.fr
            x = x + self.pos(torch.cat([p.sin(), p.cos()], -1).flatten(1))
        for layer in self.layers:
            if self.training and self.checkpoint_above and x.shape[0] > self.checkpoint_above:
                x = torch.utils.checkpoint.checkpoint(layer, x, idx, valid, offset, kind, use_reentrant=False)
            else:
                x = layer(x, idx, valid, offset, kind)
        return self.out(x)

    def run(self, t):
        """Forward pass on a dict from Tokens.to_torch()."""
        return self(t["features"], t["position"], t["idx"], t["valid"], t["offset"], t.get("kind"))


def masked_mse(pred, target):
    """Mean squared error over the finite targets only."""
    ok = torch.isfinite(target)
    return (pred - torch.nan_to_num(target)).pow(2)[ok].mean()


def target_norm(token_sets):
    """Mean and std of every target over the leaves of some token sets."""
    y = np.concatenate([t.targets[t.is_leaf] for t in token_sets])
    return np.nanmean(y, 0).astype(np.float32), np.nanstd(y, 0).astype(np.float32)


def fit(model, token_sets, steps=1000, sets_per_step=2, lr=1e-3, weight_decay=1e-4, device=None,
        seed=0, norm=None, loss=masked_mse, log_every=0):
    """Train on a list of Tokens (e.g. one per tile). Targets are scaled by
    norm = (mean, std) (default: from these token sets). Returns the loss
    per step; the model ends in eval mode, with model.norm set."""
    from .learn import Tokens

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    mu, sd = target_norm(token_sets) if norm is None else norm
    model.norm = (mu, sd)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    hist = []
    for step in range(steps):
        pick = rng.choice(len(token_sets), min(sets_per_step, len(token_sets)), replace=False)
        b = Tokens.concat([token_sets[i] for i in pick])
        t = b.to_torch(device)
        y = (t["targets"] - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device)
        lv = loss(model.run(t), y)
        opt.zero_grad()
        lv.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(lv))
        if log_every and step % log_every == 0:
            print(step, round(hist[-1], 4), flush=True)
    model.eval()
    return hist


@torch.no_grad()
def predict(model, tok, device=None):
    """Predictions for one Tokens set, in target units (NumPy, (T, n))."""
    device = device or next(model.parameters()).device
    model.eval()
    out = model.run(tok.to_torch(device)).cpu().numpy()
    if getattr(model, "norm", None) is not None:
        out = out * model.norm[1] + model.norm[0]
    return out
