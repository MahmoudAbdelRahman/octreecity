"""Export for learning: cells as tokens, links as fixed-width key tables.

    tok = tokens(tree, features=["height", "size_m"], targets=["svf"],
                 select=tree.in_box((0, 0, 0), (128, 128, 1)),     # one layer
                 parents=tree.parents(levels_above=2),
                 links=neighbours(tree) | hierarchy(tree, p), k=16)
    batch = Tokens.concat([tok_a, tok_b])          # several tiles at once
    t = batch.to_torch("cuda")                     # dict of tensors

Per token (T rows): features (T, F), position (T, 3) centre in metres,
size (T,) metres, level, is_leaf, source index (leaf or parent cell),
targets (T, Y) with NaN where unknown (always for parent cells).

Per token, its keys (the tokens it reads, from the links where it is the
source): idx (T, K) token ids, valid (T, K), kind (T, K) link type, and
offset (T, K, 4) = (dx, dy, dz, key size) / scale in metres. With
self_first, key 0 is the token itself. Everything is a NumPy array; torch
is needed only for to_torch.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tokens:
    features: np.ndarray
    position: np.ndarray
    size: np.ndarray
    level: np.ndarray
    is_leaf: np.ndarray
    source: np.ndarray
    idx: np.ndarray
    valid: np.ndarray
    kind: np.ndarray
    offset: np.ndarray
    targets: np.ndarray = None
    feature_names: tuple = ()
    target_names: tuple = ()
    kinds: tuple = ()
    dropped_links: int = 0
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.size)

    def __repr__(self):
        return (f"Tokens({len(self):,} tokens ({int(self.is_leaf.sum()):,} leaves), "
                f"{self.features.shape[1]} features, {self.idx.shape[1]} keys each, "
                f"{int(self.valid.sum()):,} links; kinds {list(self.kinds)})")

    _ARRAYS = ("features", "position", "size", "level", "is_leaf", "source", "idx", "valid",
               "kind", "offset", "targets")

    @staticmethod
    def concat(parts):
        """One batch from several token sets (key ids shifted; key tables
        padded to the widest)."""
        parts = list(parts)
        p0 = parts[0]
        if any(p.kinds != p0.kinds or p.feature_names != p0.feature_names or p.target_names != p0.target_names
               or (p.targets is None) != (p0.targets is None) for p in parts):
            raise ValueError("token sets differ in link kinds, features or targets")
        K = max(p.idx.shape[1] for p in parts)
        out, base = {k: [] for k in Tokens._ARRAYS}, 0
        for p in parts:
            pad = K - p.idx.shape[1]
            for k in Tokens._ARRAYS:
                v = getattr(p, k)
                if v is None:
                    continue
                if k == "idx":
                    v = np.pad(v + base, ((0, 0), (0, pad)), constant_values=0)
                    v[:, K - pad:] = (base + np.arange(len(p)))[:, None]
                elif k in ("valid", "kind"):
                    v = np.pad(v, ((0, 0), (0, pad)))
                elif k == "offset":
                    v = np.pad(v, ((0, 0), (0, pad), (0, 0)))
                out[k].append(v)
            base += len(p)
        arr = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
        p0 = parts[0]
        return Tokens(**arr, feature_names=p0.feature_names, target_names=p0.target_names,
                      kinds=p0.kinds, dropped_links=sum(p.dropped_links for p in parts))

    def to_torch(self, device="cpu"):
        import torch

        out = {}
        for k in self._ARRAYS:
            v = getattr(self, k)
            if v is not None:
                out[k] = torch.as_tensor(np.ascontiguousarray(v)).to(device)
        return out

    def save(self, path):
        arr = {k: getattr(self, k) for k in self._ARRAYS if getattr(self, k) is not None}
        np.savez_compressed(path, **arr, feature_names=np.array(self.feature_names),
                            target_names=np.array(self.target_names), kinds=np.array(self.kinds),
                            dropped_links=self.dropped_links)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        kw = {k: d[k] for k in cls._ARRAYS if k in d.files}
        return cls(**kw, feature_names=tuple(d["feature_names"].tolist()),
                   target_names=tuple(d["target_names"].tolist()), kinds=tuple(d["kinds"].tolist()),
                   dropped_links=int(d["dropped_links"]))


def _column(tree, name, parents=None):
    """A leaf column by name, or a built-in: size_m, level, x_m, y_m, z_m,
    label (class id), share_<class id>."""
    c = tree.centers()
    builtin = {"size_m": lambda: tree.sizes_m(), "level": lambda: tree.level.astype(np.float64),
               "x_m": lambda: c[:, 0], "y_m": lambda: c[:, 1], "z_m": lambda: c[:, 2],
               "label": lambda: tree.label.astype(np.float64)}
    if name in builtin:
        return builtin[name]()
    if name in getattr(tree, "id_columns", ()):
        raise ValueError(f"{name!r} holds object ids: they are categories, not numbers, and must not "
                         "be used as a numeric feature or averaged. Use an object-table column per "
                         "cell instead (tree.select().cells / CellView.object), or an embedding.")
    if name in tree.cells:
        return np.asarray(tree.cells[name], np.float64)
    if name.startswith("share_") and "share" in tree.cells:
        return np.asarray(tree.cells["share"][:, int(name[6:])], np.float64)
    raise KeyError(f"no column {name!r} (cells: {sorted(tree.cells)})")


def tokens(tree, features, links, targets=(), select=None, parents=None, parent_features=None,
           k=None, self_first=True, scale=16.0, order=None, near_dims=3):
    """Tokens and key tables for a model.

    features / targets: leaf column names (tree.cells or a built-in, see
        _column) or (name, array) pairs with one row per leaf.
    links: a Links object in tree token ids (leaves, then n_leaves + parent
        cell index). Each token reads the tokens it links to.
    select: bool mask of leaves to keep (default all).
    parents: tree.parents(...) to add parent cells as tokens; only those
        above a selected leaf are kept. Their features are
        parent_features {name: array per parent cell} if given, else the
        volume-weighted mean of each leaf feature over the cell.
    k: keys per token (default: the most any token has); extra links are
        dropped (counted in dropped_links), the nearest kept.
    order: link kind names in the order keys are listed (default the order
        in links); within a kind, nearest first.
    near_dims: axes used for "nearest" (2 = distance in the x-y plane, for
        one horizontal layer of cells whose cubes reach different heights).
    """
    n = tree.n_leaves
    sel = np.ones(n, bool) if select is None else np.asarray(select, bool)
    leaves = np.nonzero(sel)[0]

    def cols(spec):
        names, arrs = [], []
        for s in spec:
            name, a = (s, _column(tree, s)) if isinstance(s, str) else (s[0], np.asarray(s[1], np.float64))
            a = a.reshape(n, -1)
            names += [name] if a.shape[1] == 1 else [f"{name}_{j}" for j in range(a.shape[1])]
            arrs.append(a)
        return names, (np.concatenate(arrs, 1) if arrs else np.zeros((n, 0)))

    fnames, F = cols(features)
    tnames, Y = cols(targets)
    pos = tree.centers()[leaves]
    size = tree.sizes_m()[leaves]
    level = tree.level[leaves].astype(np.int64)
    feats, src, is_leaf = F[leaves], leaves, np.ones(leaves.size, bool)
    tmap = np.full(n + (len(parents) if parents is not None else 0), -1, np.int64)
    tmap[leaves] = np.arange(leaves.size)
    if parents is not None:
        anc = parents.ancestors()[leaves]
        keep_nodes = np.unique(anc[anc >= 0])
        tmap[n + keep_nodes] = leaves.size + np.arange(keep_nodes.size)
        if parent_features is None:
            PF = parents.reduce(F)[keep_nodes]
        else:
            PF = np.stack([np.asarray(parent_features[f], np.float64)[keep_nodes] for f in fnames], 1)
        feats = np.concatenate([feats, PF])
        pos = np.concatenate([pos, parents.centers()[keep_nodes]])
        size = np.concatenate([size, parents.sizes()[keep_nodes] * tree.voxel_size])
        level = np.concatenate([level, parents.level[keep_nodes].astype(np.int64)])
        src = np.concatenate([src, keep_nodes])
        is_leaf = np.concatenate([is_leaf, np.zeros(keep_nodes.size, bool)])
    T = len(size)
    L = links.remap(tmap)
    # sort keys: per source, by kind order, then distance
    kind_rank = np.arange(len(L.kinds))
    if order is not None:
        kind_rank = np.array([list(order).index(k) if k in order else len(order) for k in L.kinds])
    d = np.linalg.norm((pos[L.dst] - pos[L.src])[:, :near_dims], axis=1)
    o = np.lexsort((d, kind_rank[L.kind], L.src))
    s_, d_, kd = L.src[o], L.dst[o], L.kind[o]
    rank = np.arange(s_.size) - np.searchsorted(s_, s_)
    first = 1 if self_first else 0
    deg = np.bincount(s_, minlength=T)
    K = (int(deg.max()) if s_.size else 0) + first if k is None else int(k)
    room = K - first
    keep = rank < room
    idx = np.repeat(np.arange(T)[:, None], max(K, 1), 1)
    valid = np.zeros((T, max(K, 1)), bool)
    kind = np.zeros((T, max(K, 1)), np.int16)
    if self_first:
        valid[:, 0] = True
        kind[:, 0] = -1                                     # -1 = the token itself
    idx[s_[keep], first + rank[keep]] = d_[keep]
    valid[s_[keep], first + rank[keep]] = True
    kind[s_[keep], first + rank[keep]] = kd[keep]
    off = np.concatenate([pos[idx] - pos[:, None], size[idx][..., None]], -1) / scale
    tg = None
    if tnames:
        tg = np.full((T, Y.shape[1]), np.nan, np.float32)
        tg[:leaves.size] = Y[leaves]
    return Tokens(feats.astype(np.float32), pos.astype(np.float32), size.astype(np.float32), level,
                  is_leaf, src, idx, valid, kind, off.astype(np.float32), tg, tuple(fnames),
                  tuple(tnames), tuple(L.kinds), int((~keep).sum()))


def pixel_map(tree, select, pixel=1, shape=None):
    """(X, Y) array: for each pixel of `pixel` voxels, the token index
    (position among the selected leaves) of the selected leaf over it, -1
    where none. Use it to score cell predictions on a fine grid, or to
    spread them back to pixels: values[pmap]."""
    sel = np.nonzero(np.asarray(select, bool))[0]
    X = tree.shape[0] // pixel if shape is None else shape[0]
    Y = tree.shape[1] // pixel if shape is None else shape[1]
    out = np.full((X, Y), -1, np.int64)
    x, y, _ = (v[sel] for v in tree.corners())
    w = np.maximum(tree.sizes()[sel] // pixel, 1)
    tok = np.arange(sel.size)
    for k in np.unique(w)[::-1]:                            # small cells drawn last
        m = w == k
        r = np.arange(k)
        xs, ys = x[m][:, None] // pixel + r, y[m][:, None] // pixel + r
        ok = (xs < X)[:, :, None] & (ys < Y)[:, None, :]
        b = np.broadcast_to(tok[m][:, None, None], ok.shape)
        out[np.broadcast_to(xs[:, :, None], ok.shape)[ok], np.broadcast_to(ys[:, None, :], ok.shape)[ok]] = b[ok]
    return out


def reach_m(tok, hops, starts=None):
    """How far information travels in `hops` layers: per start token (default
    all leaves), the largest horizontal distance (m) to a token it can read
    after `hops` steps along its keys, plus half that token's diagonal (a
    parent cell summarises its whole square). Returns the mean."""
    import scipy.sparse as sp

    T, K = tok.idx.shape
    A = sp.csr_matrix((tok.valid.ravel().astype(np.float32), (np.repeat(np.arange(T), K), tok.idx.ravel())),
                      shape=(T, T))
    A = ((A + sp.eye(T)) > 0).astype(np.float32)
    starts = np.nonzero(tok.is_leaf)[0] if starts is None else np.asarray(starts)
    R = sp.csr_matrix((np.ones(len(starts)), (np.arange(len(starts)), starts)), shape=(len(starts), T))
    for _ in range(hops):
        R = ((R @ A) > 0).astype(np.float32)
    R = R.tocoo()
    xy = tok.position[:, :2]
    d = np.hypot(*(xy[R.col] - xy[starts[R.row]]).T) + tok.size[R.col] / np.sqrt(2)
    far = np.zeros(len(starts))
    np.maximum.at(far, R.row, d)
    return float(far.mean())


def batches(token_sets, size, rng=None):
    """Yield Tokens.concat of `size` token sets at a time (shuffled with rng)."""
    order = np.arange(len(token_sets)) if rng is None else rng.permutation(len(token_sets))
    for i in range(0, len(order), size):
        yield Tokens.concat([token_sets[j] for j in order[i:i + size]])


__all__ = ["Tokens", "tokens", "pixel_map", "batches", "reach_m"]
