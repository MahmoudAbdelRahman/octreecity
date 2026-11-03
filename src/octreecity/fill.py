"""Fill cells with values from any data, after the tree is built.

Every function returns one value per leaf (it does not change the tree), so
the result can become a static column or one snapshot of a dynamic value:

    tree.cells["svf"] = from_raster(tree, svf_map)                  # 2D field
    tree.cells["temp"] = from_grid(tree, temperature)               # 3D field
    tree.add_snapshot("08:00", wind=from_raster(tree, wind_8))      # over time
    tree.cells["rain"] = from_function(tree, lambda x, y, z: ...)   # any f
    s = from_source(tree, other_source)                             # any Source
    from_points(tree, df, x="lon", y="lat", crs="EPSG:4326")         # point table

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


def from_points(tree, data=None, x=None, y=None, z=None, crs=None, columns=None, how=None,
                write=True, height=1.5):
    """Put point data (sensors, POIs, survey points) on the leaves that
    contain them: one value per leaf.

    data: a table with one row per point (pandas, polars, a pyarrow Table,
    a DuckDB result, a dict of arrays), or a GeoDataFrame (x, y and crs
    then come from its point geometry); or an iterable of tables (chunks,
    e.g. pd.read_csv(path, chunksize=10**6)) for data bigger than memory,
    with the same result. data may also be one array of values (one per
    point), or None to count the points in each leaf.
    x, y, z: column names in data, or arrays (one value per point); z may
    be one number. Metres in the tree's frame (as centers()), or x, y in
    `crs` (crs="EPSG:4326": lon, lat). z=None: `height` metres above the
    ground under each point (street level, see tree.cell_at).
    columns: the columns to put on the cells; default every column except
    x, y, z and the geometry.
    how: one summary for all columns, or a dict {column: how} (the others
    keep their default):
        mean, sum, count, min, max, std   numbers (default: mean)
        mode    the most common value (default for text; a tie goes to the
                smallest). Needs all points at once: not with chunks.
        same    the value all points in the leaf share, else -2 (default
                for ids; exact int64, as Parents.reduce(how="same")).
    Ids are integer columns named in tree.id_columns, "id" or "..._id", and
    integers too big for float64. They are never averaged.

    A leaf with no point gets NaN (count and sum: 0; ids and integer mode:
    -1; text: ""). NaN and missing values are skipped. Points outside the
    box are dropped, with a warning that says how many.

    Returns, for a table: a dict {column: one value per leaf}, also
    written into tree.cells (id columns are added to tree.id_columns;
    write=False leaves the tree alone). For an array or None: one array.
    """
    single = data is None or not (_is_table(data) or _is_chunks(data))
    chunks = [data] if single or _is_table(data) else data
    acc, n_in, n_all, count = None, 0, 0, np.zeros(tree.n_leaves, np.int64)
    for part in chunks:
        if acc is not None and any(a.how == "mode" for a in acc.values()):
            raise ValueError(f"how='mode' (column {_first_mode(acc)!r}) needs all points at once, "
                             "not chunks: read the data in one go, or give that column another how")
        cols, X, Y, Z, c_crs, skip = _read(part, x, y, z, crs, single)
        cell = np.atleast_1d(tree.cell_at(X, Y, Z, crs=c_crs, height=height))
        if acc is None:
            names = ([] if data is None else ["value"]) if single else (
                [k for k in cols if k not in skip] if columns is None else list(columns))
            if not single and how is not None and not isinstance(how, (str, dict)):
                raise ValueError("how must be a name or a dict {column: how}")
            per = how if isinstance(how, dict) else {k: how for k in names}
            unknown = set(per) - set(names)
            if unknown:
                raise ValueError(f"how names columns that are not used: {sorted(unknown)}")
            acc = {k: _Acc(tree, k, _col(cols, k), per.get(k)) for k in names}
        inside = cell >= 0
        n_in, n_all = n_in + int(inside.sum()), n_all + cell.size
        count += np.bincount(cell[inside], minlength=tree.n_leaves)
        for k, a in acc.items():
            a.add(cell, _col(cols, k))
    if n_all > n_in:
        warnings.warn(f"{n_all - n_in:,} of {n_all:,} points are outside the tree box "
                      "and were dropped", stacklevel=2)
    if data is None:
        if how not in (None, "count"):
            raise ValueError(f"how={how!r} needs values; without values the points are counted")
        return count
    out = {k: a.result() for k, a in (acc or {}).items()}
    if single:
        return out["value"]
    if write:
        _write(tree, out, {k for k, a in acc.items() if a.how == "same"})
    return out


_HOWS = ("mean", "sum", "count", "min", "max", "std", "mode", "same")


class _Acc:
    """One column's summary per leaf, built up chunk by chunk."""

    def __init__(self, tree, name, v, how):
        self.n, self.name = tree.n_leaves, name
        kind = v.dtype.kind
        self.text = kind not in "fiub"
        self.is_id = kind in "iu" and (name in tree.id_columns or name == "id" or name.endswith("_id")
                                       or _too_big(v))
        self.how = how or ("mode" if self.text else "same" if self.is_id else "mean")
        if self.how not in _HOWS:
            raise ValueError(f"unknown how={self.how!r} for column {name!r}")
        if self.text and self.how not in ("mode", "count"):
            raise ValueError(f"column {name!r} is text: use how='mode' (or 'count')")
        if self.is_id and self.how not in ("same", "mode", "count"):
            raise ValueError(f"column {name!r} holds ids: they must not be averaged; "
                             "use how='same' (or 'mode', 'count')")
        if self.how == "same" and kind not in "iub":
            raise ValueError(f"how='same' is for integer ids; column {name!r} is not integer")
        self.cnt = np.zeros(self.n, np.int64)
        if self.how in ("mean", "sum", "std"):
            self.s, self.mean, self.m2 = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        elif self.how in ("min", "max"):
            self.v = np.full(self.n, np.inf if self.how == "min" else -np.inf)
        elif self.how == "same":
            self.lo = np.full(self.n, np.iinfo(np.int64).max)
            self.hi = np.full(self.n, np.iinfo(np.int64).min)
        self.keep = None

    def add(self, cell, v):
        if v.shape != cell.shape:
            raise ValueError(f"column {self.name!r} has shape {v.shape}, expected one per point {cell.shape}")
        ok = (cell >= 0) & _valid(v)
        c, v = cell[ok], v[ok]
        n_c = np.bincount(c, minlength=self.n)
        if self.how in ("mean", "sum", "std", "min", "max") and v.dtype.kind in "iu" and _too_big(v):
            raise ValueError(f"column {self.name!r} has integers too big for float64 (ids?): "
                             "use how='same' or 'mode'")
        if self.how == "mode":
            self.keep = (c, v.astype(str) if v.dtype.kind == "O" else v)
        elif self.how in ("mean", "sum", "std"):
            w = v.astype(np.float64)
            s_c = np.bincount(c, weights=w, minlength=self.n)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_c = np.where(n_c > 0, s_c / np.maximum(n_c, 1), 0.0)
                d = w - mean_c[c]
                m2_c = np.bincount(c, weights=d * d, minlength=self.n)
                tot = self.cnt + n_c
                delta = mean_c - self.mean
                frac = np.where(tot > 0, n_c / np.maximum(tot, 1), 0.0)
                self.m2 += m2_c + delta * delta * self.cnt * frac   # Chan et al.: merge two groups
                self.mean += delta * frac
            self.s += s_c
        elif self.how in ("min", "max"):
            (np.minimum if self.how == "min" else np.maximum).at(self.v, c, v.astype(np.float64))
        elif self.how == "same":
            v = v.astype(np.int64)
            np.minimum.at(self.lo, c, v)
            np.maximum.at(self.hi, c, v)
        self.cnt += n_c

    def result(self):
        has = self.cnt > 0
        if self.how == "count":
            return self.cnt
        if self.how == "sum":
            return self.s
        if self.how == "mean":
            return np.where(has, self.mean, np.nan)
        if self.how == "std":
            return np.where(has, np.sqrt(np.maximum(self.m2, 0) / np.maximum(self.cnt, 1)), np.nan)
        if self.how in ("min", "max"):
            return np.where(has, self.v, np.nan)
        if self.how == "same":
            return np.where(~has, -1, np.where(self.lo == self.hi, self.lo, -2)).astype(np.int64)
        c, v = self.keep if self.keep is not None else (np.zeros(0, np.int64), np.zeros(0))
        return _mode(c, v, self.n)


def _mode(c, v, n):
    """Most common value per leaf, exact for any dtype (ties: smallest)."""
    u, inv = np.unique(v, return_inverse=True)
    inv = inv.ravel()
    if v.dtype.kind in "iub":
        res, u = np.full(n, -1, np.int64), u.astype(np.int64)
    elif v.dtype.kind == "f":
        res = np.full(n, np.nan)
    else:
        res = np.full(n, "", u.dtype if u.dtype.kind in "US" else str)
    if not c.size:
        return res
    (pc, pi), k = np.unique(np.stack([c, inv.astype(np.int64)]), axis=1, return_counts=True)
    order = np.lexsort((pi, -k, pc))             # per leaf: most points first, then smallest value
    pc, pi = pc[order], pi[order]
    first = np.r_[True, pc[1:] != pc[:-1]]
    res[pc[first]] = u[pi[first]]
    return res


def _first_mode(acc):
    return next(k for k, a in acc.items() if a.how == "mode")


def _too_big(v):
    return v.size > 0 and np.abs(v.astype(np.float64)).max() >= 2 ** 53


def _valid(v):
    """False for NaN and missing values (None, pandas NA)."""
    if v.dtype.kind == "f":
        return np.isfinite(v)
    if v.dtype.kind == "O":
        return np.array([not _missing(e) for e in v], bool)
    return np.ones(v.shape, bool)


def _missing(e):
    return e is None or type(e).__name__ in ("NAType", "NaTType") or (isinstance(e, float) and e != e)


def _is_table(t):
    return (isinstance(t, dict) or hasattr(t, "fetchnumpy") or hasattr(t, "column_names")
            or (hasattr(t, "columns") and hasattr(t, "__getitem__")))


def _is_chunks(d):
    if isinstance(d, (list, tuple)):
        return len(d) > 0 and _is_table(d[0])
    return hasattr(d, "__iter__") and not hasattr(d, "__array__") and not isinstance(d, (str, bytes))


def _array(a):
    """One column as a numpy array (masked values become NaN or None)."""
    if np.ma.isMaskedArray(a):
        return (a.astype(np.float64).filled(np.nan) if a.dtype.kind in "fiu"
                else a.astype(object).filled(None))
    a = np.asarray(a)
    if a.dtype.kind == "O" and a.size:                 # numbers with gaps (pandas Int64 + NA)
        miss = np.array([_missing(e) for e in a], bool)
        vals = a[~miss]
        if vals.size and all(isinstance(e, (int, float, np.integer, np.floating))
                             and not isinstance(e, (bool, np.bool_)) for e in vals):
            if not miss.any() and all(isinstance(e, (int, np.integer)) for e in vals):
                return vals.astype(np.int64)
            return np.where(miss, np.nan, a).astype(np.float64)
    return a


def _col(cols, k):
    if k not in cols:
        raise ValueError(f"no column {k!r}; the table has {list(cols)}")
    v = cols[k]
    return v() if callable(v) else v


def _read(t, x, y, z, crs, single):
    """Columns (lazy) and coordinates of one table or of the array form."""
    if single:
        if any(isinstance(v, str) for v in (x, y, z)):
            raise ValueError("x, y, z are column names only when data is a table")
        if x is None or y is None:
            raise ValueError("give x= and y= (arrays, one value per point)")
        return {"value": None if t is None else _array(t)}, x, y, z, crs, set()
    geo = None
    if hasattr(t, "geometry") and hasattr(t, "crs") and x is None and y is None:
        geo = t.geometry
        crs = crs if crs is not None else t.crs
    if hasattr(t, "fetchnumpy"):                                  # DuckDB
        t = t.fetchnumpy()
    names = list(t.keys()) if isinstance(t, dict) else list(getattr(t, "column_names", None) or t.columns)
    cols = {str(k): (lambda k=k: _array(t[k])) for k in names}
    if geo is not None:
        X, Y = np.asarray(geo.x, np.float64), np.asarray(geo.y, np.float64)
        skip = {geo.name}
    elif x is None or y is None:
        raise ValueError("give x= and y=: the names of the position columns (e.g. x='lon', y='lat')")
    else:
        X, Y = (_col(cols, v) if isinstance(v, str) else v for v in (x, y))
        skip = {v for v in (x, y) if isinstance(v, str)}
    Z = _col(cols, z) if isinstance(z, str) else z
    if isinstance(z, str):
        skip.add(z)
    return cols, X, Y, Z, crs, skip


def _write(tree, out, ids):
    for k, v in out.items():
        old = tree.cells.get(k)
        if k in tree.id_columns and old is not None and not np.array_equal(old, v):
            raise ValueError(f"tree.cells[{k!r}] already holds other ids: rename the column "
                             f"(e.g. df.rename(columns={{{k!r}: 'sensor_{k}'}})) or use write=False")
        tree.cells[k] = v
    tree.id_columns = tuple(dict.fromkeys(tuple(tree.id_columns) + tuple(k for k in out if k in ids)))


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
