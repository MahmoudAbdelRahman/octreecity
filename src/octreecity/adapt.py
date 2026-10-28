"""Solution-adaptive refinement: split where a field jumps, merge where it is flat.

Same loop as CFD adaptive mesh refinement (OpenFOAM dynamicRefineFvMesh,
Basilisk adapt_wavelet):

    values = sample(tree, field)              # or a solve on the tree
    for _ in range(n):
        tree, values = adapt(tree, values, refine_tol=0.2)
        values = sample(tree, field)          # solve again on the new mesh

Error indicator per leaf: the largest jump |v_i - v_j| to a face neighbour
(both in the fluid: open space, class 0 = empty by default). Leaves above refine_tol split, and so do `buffer`
layers of neighbours. Groups of 8 sibling leaves merge when every one of
them is below coarsen_tol (default refine_tol / 2, a gap that stops cells
flipping back and forth). Geometry is a hard limit: only leaves with a
fluid label are split, and only siblings with the same label merge, so the
labelled geometry never changes. Cell sizes stay within [min_level,
max_level] and the result is 2:1 balanced.
"""

import numpy as np

from . import morton
from .classes import EMPTY
from .links import FACES, face_pairs  # noqa: F401  (face_pairs kept here for older code)
from .octree import LinearOctree

def sample(tree, field):
    """field(x, y, z) in metres -> value per leaf, at leaf centres."""
    x, y, z = tree.corners()
    h = tree.voxel_size
    c = 0.5 * tree.sizes()
    ox, oy, oz = tree.origin
    return np.asarray(field(ox + h * (x + c), oy + h * (y + c), oz + h * (z + c)), np.float64)


def jump_indicator(tree, values, fluid=(EMPTY,)):
    """Largest |value jump| from each leaf to its fluid face neighbours."""
    ind = np.zeros(tree.n_leaves)
    isf = np.isin(tree.label, fluid)
    a, b = face_pairs(tree, isf)
    keep = isf[b]
    a, b = a[keep], b[keep]
    j = np.abs(values[a] - values[b])
    np.maximum.at(ind, a, j)
    np.maximum.at(ind, b, j)
    return ind


def transfer(old, old_values, new):
    """Map per-leaf values from one octree to another over the same box.

    A new leaf inside an old leaf takes its value (prolongation); a new leaf
    covering several old leaves takes their volume-weighted mean
    (restriction), so the volume integral is kept.
    """
    span = np.left_shift(np.uint64(1), 3 * new.level.astype(np.uint64))
    lo = np.searchsorted(old.code, new.code, side="right") - 1
    hi = np.searchsorted(old.code, new.code + span, side="left")
    out = old_values[lo].astype(np.float64)
    many = hi - lo > 1
    if many.any():
        vol = old.sizes().astype(np.float64) ** 3
        w = np.concatenate([[0.0], np.cumsum(old_values * vol)])
        v = np.concatenate([[0.0], np.cumsum(vol)])
        out[many] = (w[hi[many]] - w[lo[many]]) / (v[hi[many]] - v[lo[many]])
    return out


def adapt(tree, values, refine_tol, coarsen_tol=None, fluid=(EMPTY,), buffer=1,
          min_level=0, max_level=None, max_cells=None, balance=26):
    """One split-and-merge pass. Returns (new_tree, values on new_tree).

    min_level / max_level: cell size limits as levels (cell = 2**level
    voxels of tree.voxel_size metres); default the tree's own limits.
    max_cells: optional budget; if splitting would exceed it, only the
    leaves with the largest indicator are split.
    """
    coarsen_tol = refine_tol / 2 if coarsen_tol is None else coarsen_tol
    if coarsen_tol >= refine_tol:
        raise ValueError("coarsen_tol must be smaller than refine_tol")
    max_level = tree.max_level if max_level is None else min(max_level, tree.max_level)
    values = np.asarray(values, np.float64)
    ind = jump_indicator(tree, values, fluid)
    isf = np.isin(tree.label, fluid)

    split = isf & (ind > refine_tol) & (tree.level > min_level)
    for _ in range(buffer):
        split |= isf & (tree.level > min_level) & _touches(tree, split)
    if max_cells is not None and split.any():
        room = max(0, (max_cells - tree.n_leaves) // 7)
        if split.sum() > room:
            order = np.argsort(-np.where(split, ind, -np.inf))
            split[:] = False
            split[order[:room]] = True

    merge = _mergeable_groups(tree, isf & (ind < coarsen_tol) & ~split, max_level)

    new = _apply(tree, split, merge)
    if balance:
        new.balance(balance)
    return new, transfer(tree, values, new)


def _touches(tree, mask):
    """Leaves face-adjacent to any leaf in mask (exact)."""
    a, b = face_pairs(tree)
    out = np.zeros(tree.n_leaves, bool)
    out[b[mask[a]]] = True
    out[a[mask[b]]] = True
    return out & ~mask


def _mergeable_groups(tree, ok, max_level):
    """Index of the first leaf of every complete group of 8 siblings that
    are all `ok`, share one label, and whose parent is within max_level."""
    n = tree.n_leaves
    if n < 8:
        return np.zeros(0, np.int64)
    lv = tree.level.astype(np.int64)
    i = np.arange(n - 7)
    l0 = lv[i]
    step = np.left_shift(np.uint64(1), 3 * tree.level[i].astype(np.uint64))
    first = (tree.code[i] % (step * np.uint64(8)) == 0)
    full = (lv[i + 7] == l0) & (tree.code[i + 7] == tree.code[i] + np.uint64(7) * step)
    cand = first & full & (l0 + 1 <= max_level)
    win = np.lib.stride_tricks.sliding_window_view
    same_lv = (win(lv, 8) == l0[:, None]).all(1)
    same_lab = (win(tree.label, 8) == tree.label[i][:, None]).all(1)
    all_ok = win(ok, 8).all(1)
    return np.nonzero(cand & same_lv & same_lab & all_ok)[0]


def _apply(tree, split, merge_first):
    keep = ~split
    gone = np.zeros(tree.n_leaves, bool)
    for k in range(8):
        gone[merge_first + k] = True
    keep &= ~gone
    x, y, z = tree.corners()
    codes, levels, labels = [tree.code[keep]], [tree.level[keep]], [tree.label[keep]]
    if split.any():
        cl = tree.level[split].astype(np.int64) - 1
        cs = np.left_shift(1, cl)
        off = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])
        cx = (x[split][:, None] + off[:, 0] * cs[:, None]).ravel()
        cy = (y[split][:, None] + off[:, 1] * cs[:, None]).ravel()
        cz = (z[split][:, None] + off[:, 2] * cs[:, None]).ravel()
        codes.append(morton.encode(cx, cy, cz))
        levels.append(np.repeat(cl, 8).astype(np.uint8))
        labels.append(np.repeat(tree.label[split], 8))
    if merge_first.size:
        codes.append(tree.code[merge_first])
        levels.append(tree.level[merge_first] + 1)
        labels.append(tree.label[merge_first])
    return LinearOctree(tree.shape, tree.max_level, np.concatenate(codes),
                        np.concatenate(levels), np.concatenate(labels),
                        tree.voxel_size, tree.origin)
