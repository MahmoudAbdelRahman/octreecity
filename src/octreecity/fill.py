"""Fill cells with values from any data, after the tree is built.

Every function returns one value per leaf (it does not change the tree), so
the result can become a static column or one snapshot of a dynamic value:

    tree.cells["svf"] = from_raster(tree, svf_map)                  # 2D field
    tree.cells["temp"] = from_grid(tree, temperature)               # 3D field
    tree.add_snapshot("08:00", wind=from_raster(tree, wind_8))      # over time
    tree.cells["rain"] = from_function(tree, lambda x, y, z: ...)   # any f
    s = from_source(tree, other_source)                             # any Source
    tree.cells["pm25"] = from_points(tree, lon, lat, 2.0, pm25, crs="EPSG:4326")  # points

Values are summaries of the data inside each cell (mean by default; std,
min, max, count on request; NaN data is skipped). Parent cells get theirs
with tree.parents().reduce(column). carry() moves columns onto a changed
tree over the same box (for snapshots taken on different trees).
"""

import warnings

import numpy as np
from scipy import ndimage

from .adapt import transfer, transfer_ids


def from_source(tree, source, levels=None):
    """Ask a Source to summarize every leaf: returns a Summary with one row
    per leaf (counts per class, attribute stats). The source must cover the
    same box (voxel units)."""
    from .sources import Summary

    x, y, z = tree.corners()
    parts, order = [], []
    for lv in np.unique(tree.level):
        m = np.nonzero(tree.level == lv)[0]
        parts.append(source.summarize(x[m], y[m], z[m], 1 << int(lv)))
        order.append(m)
    s = Summary.concat(parts)
    inv = np.empty(tree.n_leaves, np.int64)
    inv[np.concatenate(order)] = np.arange(tree.n_leaves)
    return s.take(inv)


def from_grid(tree, grid, how="mean"):
    """Summary of a 3D field (nx, ny, nz) at the tree's voxel size over each
    leaf (cells beyond the grid see NaN)."""
    from .sources import GridSource

    g = np.asarray(grid, np.float64)
    s = from_source(tree, GridSource(np.zeros(g.shape, np.uint8), attrs={"v": g}))
    return getattr(s, how)("v")


def from_raster(tree, raster, how="mean", pixel=1):
    """Summary of a 2D field over each leaf's footprint (all heights).

    raster: (X, Y) values, one per pixel of `pixel` voxels, indexed [x, y]
    from the tree's corner (pixel 0 starts at voxel 0). A leaf smaller than a
    pixel takes that pixel's value. how: mean, std, min, max, count, sum.
    """
    r = np.asarray(raster, np.float64)
    X, Y = r.shape
    x, y, _ = tree.corners()
    s = tree.sizes()
    p = int(pixel)
    x0, y0 = x // p, y // p
    x1, y1 = np.maximum(-(-(x + s) // p), x0 + 1), np.maximum(-(-(y + s) // p), y0 + 1)
    cx0, cy0 = np.clip(x0, 0, X), np.clip(y0, 0, Y)
    cx1, cy1 = np.clip(x1, 0, X), np.clip(y1, 0, Y)
    ok = np.isfinite(r)
    if how in ("min", "max"):
        out = np.full(tree.n_leaves, np.nan)
        fill = np.inf if how == "min" else -np.inf
        f = ndimage.minimum_filter if how == "min" else ndimage.maximum_filter
        rr = np.where(ok, r, fill)
        wx, wy = x1 - x0, y1 - y0
        for kx, ky in np.unique(np.stack([wx, wy], 1), axis=0):     # one filter per footprint shape
            m = (wx == kx) & (wy == ky) & (cx1 > cx0) & (cy1 > cy0)
            if not m.any():
                continue
            # window [i, i + k) for output i
            v = f(rr, size=(int(kx), int(ky)), origin=(-(int(kx) // 2), -(int(ky) // 2)),
                  mode="constant", cval=fill)
            out[m] = v[cx0[m], cy0[m]]
        out[~np.isfinite(out)] = np.nan
        return out
    sat = lambda a: np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = lambda S: S[cx1, cy1] - S[cx0, cy1] - S[cx1, cy0] + S[cx0, cy0]
    n = box(sat(ok.astype(np.float64)))
    if how == "count":
        return n
    s1 = box(sat(np.where(ok, r, 0.0)))
    if how == "sum":
        return s1
    if how == "std":                      # centred first, so the sums stay small
        r = r - np.nanmean(r) if ok.any() else r
        s1 = box(sat(np.where(ok, r, 0.0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        if how == "mean":
            return mean
        if how == "std":
            s2 = box(sat(np.where(ok, r * r, 0.0)))
            return np.where(n > 0, np.sqrt(np.maximum(s2 / n - mean * mean, 0)), np.nan)
    raise ValueError(f"unknown how={how!r}")


def from_function(tree, func, samples=1):
    """func(x, y, z) in metres -> value. samples=1 takes the value at each
    leaf's centre; samples=k averages a k x k x k lattice inside the leaf."""
    c = tree.centers()
    if samples <= 1:
        return np.asarray(func(c[:, 0], c[:, 1], c[:, 2]), np.float64)
    half = 0.5 * tree.sizes_m()[:, None]
    u = (np.arange(samples) + 0.5) / samples * 2 - 1           # -1..1 lattice
    acc = np.zeros(tree.n_leaves)
    for a in u:
        for b in u:
            for d in u:
                acc += func(c[:, 0] + a * half[:, 0], c[:, 1] + b * half[:, 0], c[:, 2] + d * half[:, 0])
    return acc / samples ** 3


def from_points(tree, x, y, z, values=None, how=None, crs=None):
    """One value per leaf from point data (sensors, POIs, survey points):
    each point goes to the leaf that contains it (tree.cell_at).

    x, y, z: metres in the tree's frame (as centers()), or x, y in `crs`
    (crs="EPSG:4326": lon, lat) with z in metres; scalars and arrays mix.
    values: one value per point; a dict {name: values} (returns a dict); or
    None to count the points in each leaf.
    how: mean, sum, count, min, max, std, or mode (the most common value,
    for categories and ids; a tie goes to the smallest value). Default:
    mode for text and ids, else mean. Ids are values named in
    tree.id_columns or integers too big for float64; they are kept exact
    and only mode or count may be used on them.

    A leaf with no point gets NaN (count and sum: 0; mode: -1 for
    integers, "" for text). NaN values are skipped. Points outside the box
    are dropped, with a warning that says how many.
    """
    cell = np.atleast_1d(tree.cell_at(x, y, z, crs=crs))
    out = cell < 0
    if out.any():
        warnings.warn(f"{int(out.sum()):,} of {cell.size:,} points are outside the tree box "
                      "and were dropped", stacklevel=2)
    if values is None:
        if how not in (None, "count"):
            raise ValueError(f"how={how!r} needs values; without values the points are counted")
        return np.bincount(cell[~out], minlength=tree.n_leaves)
    if isinstance(values, dict):
        return {k: _from_points(tree, cell, v, how, k in tree.id_columns) for k, v in values.items()}
    return _from_points(tree, cell, values, how, False)


def _from_points(tree, cell, v, how, is_id):
    v = np.asarray(v)
    if v.shape != cell.shape:
        raise ValueError(f"values have shape {v.shape}, expected one per point {cell.shape}")
    numeric = v.dtype.kind in "fiub"
    is_id = is_id or (v.dtype.kind in "iu" and v.size > 0
                      and np.abs(v.astype(np.float64)).max() >= 2 ** 53)
    how = how or ("mode" if is_id or not numeric else "mean")
    if how not in ("mean", "sum", "count", "min", "max", "std", "mode"):
        raise ValueError(f"unknown how={how!r}")
    if how not in ("mode", "count") and (is_id or not numeric):
        raise ValueError("these values are ids or text: they must not pass through float; "
                         "use how='mode' (or 'count')")
    keep = cell >= 0
    if v.dtype.kind == "f":
        keep &= np.isfinite(v)
    c, v, n = cell[keep], v[keep], tree.n_leaves
    cnt = np.bincount(c, minlength=n)
    if how == "count":
        return cnt
    if how == "mode":
        return _mode(c, v, n)
    w = v.astype(np.float64)
    if how == "sum":
        return np.bincount(c, weights=w, minlength=n)
    if how in ("min", "max"):
        order = np.lexsort((w, c))
        cs, ws = c[order], w[order]
        new = np.r_[True, cs[1:] != cs[:-1]] if cs.size else np.zeros(0, bool)
        pick = new if how == "min" else np.r_[new[1:], True][:cs.size]
        res = np.full(n, np.nan)
        res[cs[pick]] = ws[pick]
        return res
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(c, weights=w, minlength=n) / cnt
        if how == "mean":
            return mean
        d = w - mean[c]                          # std: centred per leaf first
        return np.sqrt(np.bincount(c, weights=d * d, minlength=n) / cnt)


def _mode(c, v, n):
    """Most common value per leaf, exact for any dtype (ties: smallest)."""
    u, inv = np.unique(v, return_inverse=True)
    inv = inv.ravel()
    if v.dtype.kind in "iub":
        res, u = np.full(n, -1, np.int64), u.astype(np.int64)
    elif v.dtype.kind == "f":
        res = np.full(n, np.nan)
    elif v.dtype.kind in "US":
        res = np.full(n, "", u.dtype)
    else:
        res = np.full(n, None, object)
    if not c.size:
        return res
    (pc, pi), k = np.unique(np.stack([c, inv.astype(np.int64)]), axis=1, return_counts=True)
    order = np.lexsort((pi, -k, pc))             # per leaf: most points first, then smallest value
    pc, pi = pc[order], pi[order]
    first = np.r_[True, pc[1:] != pc[:-1]]
    res[pc[first]] = u[pi[first]]
    return res


def carry(old, new, columns=None):
    """Copy numeric columns from `old` onto `new` (same box): a new leaf
    inside an old one takes its value, a new leaf covering several takes
    their volume-weighted mean. Id columns (old.id_columns) take the id
    the covered leaves share, or -2: they are never averaged. Changes `new`
    in place and returns it."""
    names = old.cells.keys() if columns is None else columns
    for k in names:
        v = np.asarray(old.cells[k])
        if k in old.id_columns:
            new.cells[k] = transfer_ids(old, v, new)
            new.id_columns = tuple(dict.fromkeys(tuple(new.id_columns) + (k,)))
        elif v.ndim == 1 and v.dtype.kind in "fiub":
            new.cells[k] = transfer(old, v.astype(np.float64), new)
        elif v.ndim == 2 and v.dtype.kind == "f":
            new.cells[k] = np.stack([transfer(old, v[:, j], new) for j in range(v.shape[1])], 1)
    return new
