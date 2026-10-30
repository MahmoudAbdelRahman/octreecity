"""Top-down build: start from big boxes, ask the source, split by a rule.

    from octreecity.build import build
    from octreecity.rules import mixed, varies
    from octreecity.sources import OccupancySource

    tree = build(OccupancySource(city), mixed() | varies("height", 3.0),
                 max_level=5, min_level=0)

The box is tiled by root cubes of 2**max_level voxels (topped up with
empty space beyond the source's far edges). For each level, from the roots
down, the source summarises the current boxes in one call, the rule picks
the ones to split, and the others become leaves. A box at min_level never
splits. Nothing is voxelised by the builder itself: the cost follows the
number of boxes visited, and each source answers in its own way.

Each leaf gets, in its cell table (tree.cells):
    label           the class with the most volume in the box (classes
                    merged to `label_depth` of the class tree first, if set)
    share           (n, C) volume share of every class   [shares=True]
    <attr>          mean of the attribute over the box     [stats has "mean"]
    <attr>_std, _min, _max, _count                         [if in stats]
    <id layer>      for a source with id layers (source.categorical): the
                    one object id all voxels of the leaf share, as exact
                    int64, or -2 when they hold several (-1 = no object).
                    Never averaged. Listed in tree.id_columns.

The tree also takes the source's crs and object table (tree.objects).
"""

import numpy as np

from . import morton
from .octree import LinearOctree, _check_shape, _children
from .rules import Boxes


def build(source, rule, max_level, min_level=0, shares=False, stats=("mean",), label_depth=None):
    S = 1 << int(max_level)
    shape = tuple(-(-int(n) // S) * S for n in source.shape)
    _check_shape(shape, max_level)
    r = [np.arange(n // S) for n in shape]
    ix, iy, iz = (g.ravel() for g in np.meshgrid(*r, indexing="ij"))
    leaves = []                                    # (x, y, z, level, summary) per level
    for lv in range(int(max_level), int(min_level) - 1, -1):
        if not ix.size:
            break
        s = 1 << lv
        x, y, z = ix * s, iy * s, iz * s
        summ = source.summarize(x, y, z, s)
        if lv > min_level:
            split = np.broadcast_to(np.asarray(rule(Boxes(x, y, z, lv, summ, source)), bool), ix.shape)
        else:
            split = np.zeros(ix.size, bool)
        keep = ~split
        leaves.append((x[keep], y[keep], z[keep], lv, summ.take(keep)))
        ix, iy, iz = _children(ix[split], iy[split], iz[split])
    return _tree(source, shape, max_level, leaves, shares, stats, label_depth)


def _tree(source, shape, max_level, leaves, shares, stats, label_depth=None):
    from .sources import Summary

    x = np.concatenate([p[0] for p in leaves])
    y = np.concatenate([p[1] for p in leaves])
    z = np.concatenate([p[2] for p in leaves])
    level = np.concatenate([np.full(len(p[0]), p[3], np.uint8) for p in leaves])
    summ = Summary.concat([p[4] for p in leaves])
    cells = {}
    if shares:
        cells["share"] = summ.share().astype(np.float32)
    cat = getattr(source, "categorical", None) or {}
    for a in source.attributes:
        if a in cat:
            cells[a] = _ids_of(summ, a, cat[a])
            continue
        for st in stats:
            key = a if st == "mean" else f"{a}_{st}"
            cells[key] = getattr(summ, st)(a)
    counts = summ.counts
    if label_depth is not None and source.classes is not None:
        up = source.classes.at_depth(np.arange(counts.shape[1]), label_depth)
        merged = np.zeros_like(counts)
        np.add.at(merged.T, up, counts.T)
        counts = merged
    tree = LinearOctree(shape, int(max_level), morton.encode(x, y, z), level, counts.argmax(1),
                        source.voxel_size, source.origin, classes=source.classes, cells=cells,
                        crs=getattr(source, "crs", None), id_columns=tuple(cat))
    tree.objects = getattr(source, "objects", None)
    return tree


def _ids_of(summ, name, values):
    """Exact int64 id per box from the min / max of its id codes: the id
    when min == max, -2 (mixed) when not, -1 when no voxel has a code."""
    from .objects import MIXED, NONE

    lo, hi = summ.min(name), summ.max(name)
    out = np.full(lo.shape, NONE, np.int64)
    ok = np.isfinite(lo)
    same = ok & (lo == hi)
    out[same] = np.asarray(values, np.int64)[lo[same].astype(np.int64)]
    out[ok & ~same] = MIXED
    return out


def uniform(source, level, shares=False, stats=("mean",)):
    """A uniform grid of cells of 2**level voxels over the same source:
    the baseline every adaptive tree is compared against."""
    from .rules import never

    return build(source, never(), max_level=level, min_level=level, shares=shares, stats=stats)
