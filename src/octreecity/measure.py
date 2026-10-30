"""Measurements: what adaptive cells cost against a uniform grid.

    from octreecity.measure import growth, table
    rows = growth(OccupancySource(city), mixed(), max_level=6, min_levels=[4, 3, 2, 1, 0])
    print(table(rows))

For each smallest cell size, the same source is built two ways: the
adaptive tree (split by the rule down to that size) and the uniform grid of
that size. Per row: smallest cell (m), cells (leaves) and parent cells of
the tree, uniform cells, their ratio, build time (s) and peak Python memory
(MB, tracemalloc) of both, tree memory (bytes). `region` = (lo, hi) in
voxels counts only the cells that overlap that box, e.g. one horizontal
layer ((0, 0, 0), (nx, ny, 1)); the uniform grid is counted the same way.

The uniform grid is only built (timed) when it has at most
`uniform_build_limit` cells; above that its count is computed, not built.
"""

import time
import tracemalloc

import numpy as np

from .build import build, uniform


def timed(func, *args, **kw):
    """(result, seconds, peak MB) of one call."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = func(*args, **kw)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return out, dt, peak


def growth(source, rule, max_level, min_levels, region=None, uniform_build_limit=2_000_000):
    rows = []
    S = 1 << int(max_level)
    shape = np.array([-(-int(n) // S) * S for n in source.shape])
    for m in min_levels:
        tree, t_tree, mem_tree = timed(build, source, rule, max_level, min_level=m)
        p = tree.parents()
        s = 1 << int(m)
        if region is None:
            sel = np.ones(tree.n_leaves, bool)
            n_uni = int(np.prod(shape // s))
        else:
            lo, hi = (np.asarray(v, np.int64) for v in region)
            sel = tree.in_box(lo, hi)
            n_uni = int(np.prod(-(-np.minimum(hi, shape) // s) - lo // s))
        row = dict(min_cell_m=s * source.voxel_size, cells=int(sel.sum()), parent_cells=len(p),
                   uniform_cells=n_uni, ratio=n_uni / max(int(sel.sum()), 1),
                   tree_s=t_tree, tree_peak_mb=mem_tree, tree_bytes=int(tree.nbytes),
                   levels=tree.level_histogram().tolist())
        if int(np.prod(shape // s)) <= uniform_build_limit:
            _, t_uni, mem_uni = timed(uniform, source, m)
            row.update(uniform_s=t_uni, uniform_peak_mb=mem_uni)
        else:
            row.update(uniform_s=np.nan, uniform_peak_mb=np.nan)
        rows.append(row)
    return rows


def table(rows):
    """Plain-text table of growth() rows."""
    head = ["min cell m", "cells", "parents", "uniform", "uniform/cells", "tree s", "tree MB",
            "uniform s", "uniform MB"]
    keys = ["min_cell_m", "cells", "parent_cells", "uniform_cells", "ratio", "tree_s", "tree_peak_mb",
            "uniform_s", "uniform_peak_mb"]
    fmt = lambda v: (f"{v:,}" if isinstance(v, int) else "-" if v != v else f"{v:,.3g}")
    body = [[fmt(r[k]) for k in keys] for r in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    line = lambda cells: "  ".join(c.rjust(n) for c, n in zip(cells, w))
    return "\n".join([line(head)] + [line(b) for b in body])
