"""Data sources: anything that can answer "what is inside this box?".

The builder (octreecity.build) only ever asks a source one question, for
many boxes at once: summarize(x, y, z, size) -> Summary. A Summary says,
per box, how much of each class it holds (voxel counts, class 0 = empty)
and, per attribute, count / sum / sum of squares / min / max of its values.
All of these add up when boxes are merged, so a source can answer from any
internal structure it likes.

Ready-made sources:
    GridSource     a dense voxel grid of class ids (+ attribute grids):
                   any data already voxelised (scans, simulations, meshes)
    OccupancySource  an Occupancy: runs of classes along z per column, so
                   height maps, LoD1 blocks, canopies over air, bridges and
                   overhangs; answered from the runs, no voxel grid
    SparseSource   occupied voxels only, e.g. from a point cloud
                   (from_points) or surface samples of a mesh; answered by
                   binary search on Morton codes, no voxel grid

Your own source: subclass Source and implement summarize. Boxes are aligned
cubes given by their min corner and edge, in voxels of `voxel_size` metres;
parts of a box outside `shape` count as empty.

Readers for 2D city data (CityGML footprints, GeoJSON, OSM) live in
octreecity.features and are re-exported here for older code.
"""

from dataclasses import dataclass, field

import numpy as np

from . import morton
from .classes import EMPTY
from .features import (Features, Surfaces, read_citygml, read_citygml_surfaces,  # noqa: F401
                       read_geojson, read_osm_overpass, reproject)

CHUNK = 1 << 22          # voxels (or column-run pairs) handled at once


@dataclass
class Summary:
    """Content of n boxes. counts: (n, C) voxels per class; stats: per
    attribute an (n, 5) array of count, sum, sum of squares, min, max over
    the voxels where the attribute is defined (NaN min/max where none)."""
    volume: np.ndarray
    counts: np.ndarray
    stats: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, n, n_classes, attributes=()):
        st = {a: np.tile([0.0, 0.0, 0.0, np.nan, np.nan], (n, 1)) for a in attributes}
        return cls(np.zeros(n), np.zeros((n, n_classes)), st)

    def __len__(self):
        return len(self.volume)

    def take(self, idx):
        return Summary(self.volume[idx], self.counts[idx], {k: v[idx] for k, v in self.stats.items()})

    @staticmethod
    def concat(parts):
        parts = list(parts)
        return Summary(np.concatenate([p.volume for p in parts]), np.concatenate([p.counts for p in parts]),
                       {k: np.concatenate([p.stats[k] for p in parts]) for k in parts[0].stats})

    # what rules and the builder read
    def share(self):
        return self.counts / self.volume[:, None]

    def dominant(self):
        return self.counts.argmax(1)

    def count(self, name):
        return self.stats[name][:, 0]

    def mean(self, name):
        s = self.stats[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s[:, 0] > 0, s[:, 1] / s[:, 0], np.nan)

    def std(self, name):
        s = self.stats[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s[:, 1] / s[:, 0]
            return np.where(s[:, 0] > 0, np.sqrt(np.maximum(s[:, 2] / s[:, 0] - m * m, 0)), np.nan)

    def min(self, name):
        return self.stats[name][:, 3]

    def max(self, name):
        return self.stats[name][:, 4]


def _stats(box, w, v, n):
    """(n, 5) stats of values v with weights w (voxel counts) per box id."""
    ok = np.isfinite(v) & (w > 0)
    b, w, v = box[ok], w[ok], v[ok]
    out = np.empty((n, 5))
    out[:, 0] = np.bincount(b, w, n)
    out[:, 1] = np.bincount(b, w * v, n)
    out[:, 2] = np.bincount(b, w * v * v, n)
    out[:, 3] = np.inf
    out[:, 4] = -np.inf
    np.minimum.at(out[:, 3], b, v)
    np.maximum.at(out[:, 4], b, v)
    out[out[:, 0] == 0, 3:] = np.nan
    return out


def _check_labels(labels, n_classes):
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(f"class ids must be in 0 .. {n_classes - 1} (the classes given); "
                         f"found {int(labels.min())} .. {int(labels.max())}")


def _merge8(kids, n, size):
    """Summaries of n boxes from the summaries of their 8 children each (in
    order: box 0's 8 children first)."""
    c = kids.counts.reshape(n, 8, -1).sum(1)
    st = {}
    for k, v in kids.stats.items():
        v = v.reshape(n, 8, 5)
        out = np.empty((n, 5))
        out[:, :3] = v[:, :, :3].sum(1)
        with np.errstate(invalid="ignore"):
            out[:, 3] = np.fmin.reduce(v[:, :, 3], 1)
            out[:, 4] = np.fmax.reduce(v[:, :, 4], 1)
        st[k] = out
    return Summary(np.full(n, float(size) ** 3), c, st)


class Source:
    """Base class: set shape, voxel_size, origin, classes (optional),
    attributes (names) and implement summarize."""
    shape = (0, 0, 0)
    voxel_size = 1.0
    origin = (0.0, 0.0, 0.0)
    classes = None
    attributes = ()
    n_classes = 1

    def summarize(self, x, y, z, size):
        raise NotImplementedError

    def _fill_empty(self, s, size):
        """Make every box hold size**3 voxels: what is not counted is empty."""
        s.volume = np.full(len(s.counts), float(size) ** 3)
        s.counts[:, EMPTY] += s.volume - s.counts.sum(1)
        return s


# --------------------------------------------------------------------- dense

class GridSource(Source):
    """Dense voxel grid: labels (nx, ny, nz) class ids, attrs {name: grid}
    with NaN where an attribute is undefined (e.g. in empty voxels)."""

    def __init__(self, labels, attrs=None, voxel_size=1.0, origin=(0.0, 0.0, 0.0), classes=None):
        self.labels = np.asarray(labels)
        self.attrs = {k: np.asarray(v, np.float64) for k, v in (attrs or {}).items()}
        self.shape = self.labels.shape
        self.voxel_size, self.origin, self.classes = float(voxel_size), tuple(origin), classes
        self.attributes = tuple(self.attrs)
        self.n_classes = len(classes) if classes is not None else int(self.labels.max()) + 1
        _check_labels(self.labels, self.n_classes)

    def summarize(self, x, y, z, size):
        x, y, z = (np.asarray(v, np.int64) for v in (x, y, z))
        s = int(size)
        if s ** 3 > CHUNK and len(x):                 # very big boxes: add up their 8 children
            h = s // 2
            off = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)]) * h
            kids = self.summarize((x[:, None] + off[:, 0]).ravel(), (y[:, None] + off[:, 1]).ravel(),
                                  (z[:, None] + off[:, 2]).ravel(), h)
            return _merge8(kids, len(x), s)
        per = max(1, CHUNK // s ** 3)
        parts = []
        for i in range(0, len(x), per):
            parts.append(self._chunk(x[i:i + per], y[i:i + per], z[i:i + per], s))
        if not parts:
            return Summary.empty(0, self.n_classes, self.attributes)
        return self._fill_empty(Summary.concat(parts), s)

    def _chunk(self, x, y, z, s):
        m, C = len(x), self.n_classes
        r = np.arange(s)
        idx, ok = [], np.ones((m, s, s, s), bool)
        for a, (v, n) in enumerate(zip((x, y, z), self.shape)):
            g = v[:, None] + r
            shp = [m, 1, 1, 1]
            shp[a + 1] = s
            ok &= (g < n).reshape(shp)
            idx.append(np.minimum(g, n - 1).reshape(shp))
        lab = self.labels[idx[0], idx[1], idx[2]].astype(np.int64)
        box = np.broadcast_to(np.arange(m)[:, None, None, None], ok.shape)
        counts = np.bincount((box * C + lab)[ok], minlength=m * C).reshape(m, C).astype(np.float64)
        stats = {k: _stats(box[ok], np.ones(int(ok.sum())), g[idx[0], idx[1], idx[2]][ok], m)
                 for k, g in self.attrs.items()}
        return Summary(np.zeros(m), counts, stats)


# ------------------------------------------------------------------- columns

class OccupancySource(Source):
    """An Occupancy (runs of classes along z per column). attrs: {name:
    (nx, ny) raster}, a value per column that applies to its non-empty
    voxels (e.g. a building's height or year per footprint pixel)."""

    def __init__(self, city, attrs=None, classes=None):
        self.city = city
        self.attrs = {k: np.asarray(v, np.float64) for k, v in (attrs or {}).items()}
        self.shape, self.voxel_size, self.origin = city.shape, city.voxel_size, city.origin
        self.classes = classes if classes is not None else getattr(city, "classes", None)
        self.attributes = tuple(self.attrs)
        self.n_classes = len(self.classes) if self.classes is not None else int(city.label.max()) + 1
        _check_labels(np.asarray(city.label), self.n_classes)
        self._bottom = city.bottoms().astype(np.int64)

    def summarize(self, x, y, z, size):
        x, y, z = (np.asarray(v, np.int64) for v in (x, y, z))
        s = int(size)
        nx, ny, nz = self.shape
        per = max(1, CHUNK // (s * s * 4))
        parts = []
        for i in range(0, len(x), per):
            parts.append(self._chunk(x[i:i + per], y[i:i + per], z[i:i + per], s))
        if not parts:
            return Summary.empty(0, self.n_classes, self.attributes)
        return self._fill_empty(Summary.concat(parts), s)

    def _chunk(self, x, y, z, s):
        c = self.city
        nx, ny, nz = self.shape
        m, C = len(x), self.n_classes
        r = np.arange(s)
        gx = (x[:, None, None] + r[None, :, None]).repeat(s, 2)
        gy = (y[:, None, None] + r[None, None, :]).repeat(s, 1)
        box = np.broadcast_to(np.arange(m)[:, None, None], gx.shape).ravel()
        gx, gy = gx.ravel(), gy.ravel()
        inside = (gx < nx) & (gy < ny)
        box, col = box[inside], gx[inside] * ny + gy[inside]
        nrun = (c.offsets[col + 1] - c.offsets[col]).astype(np.int64)
        pb = np.repeat(np.arange(col.size), nrun)                  # pair of each run
        run = np.repeat(c.offsets[col] - np.cumsum(nrun) + nrun, nrun) + np.arange(nrun.sum())
        z0 = z[box[pb]]
        ov = np.clip(np.minimum(c.top[run].astype(np.int64), z0 + s) - np.maximum(self._bottom[run], z0), 0, None)
        lab = c.label[run].astype(np.int64)
        counts = np.bincount(box[pb] * C + lab, ov, m * C).reshape(m, C)
        stats = {}
        if self.attrs:
            wpair = np.bincount(pb, np.where(lab != EMPTY, ov, 0), col.size)
            cx, cy = np.divmod(col, ny)
            for k, g in self.attrs.items():
                stats[k] = _stats(box, wpair, g[cx, cy], m)
        return Summary(np.zeros(m), counts, stats)


# -------------------------------------------------------------------- sparse

class SparseSource(Source):
    """Occupied voxels only: integer voxel coordinates (x, y, z), one class
    each, optional attrs {name: value per voxel}. Everything else is empty.
    Boxes are answered by binary search on Morton codes."""

    def __init__(self, x, y, z, labels, attrs=None, shape=None, voxel_size=1.0,
                 origin=(0.0, 0.0, 0.0), classes=None):
        x, y, z = (np.asarray(v, np.int64) for v in (x, y, z))
        code = morton.encode(x, y, z)
        order = np.argsort(code, kind="stable")
        self.code = code[order]
        if (np.diff(self.code) == 0).any():
            raise ValueError("duplicate voxels: give one class per voxel (see from_points)")
        self.labels = np.asarray(labels, np.int64)[order]
        self.shape = tuple(int(v) for v in (shape or (x.max() + 1, y.max() + 1, z.max() + 1)))
        self.voxel_size, self.origin, self.classes = float(voxel_size), tuple(origin), classes
        self.n_classes = len(classes) if classes is not None else int(self.labels.max()) + 1
        _check_labels(self.labels, self.n_classes)
        self.by_class = [self.code[self.labels == k] for k in range(self.n_classes)]
        self.attrs = {}
        for k, v in (attrs or {}).items():
            v = np.asarray(v, np.float64)[order]
            ok = np.isfinite(v)
            cum = np.zeros((v.size + 1, 3))
            cum[1:] = np.cumsum(np.stack([ok, np.where(ok, v, 0), np.where(ok, v * v, 0)], 1), 0)
            self.attrs[k] = (v, cum)
        self.attributes = tuple(self.attrs)

    @classmethod
    def from_points(cls, xyz, labels, voxel_size, origin=None, attrs=None, shape=None, classes=None):
        """Voxelise a point cloud: each occupied voxel takes the most common
        class of its points and the mean of each attribute. Note that points
        sample surfaces: the inside of a solid stays empty."""
        xyz = np.asarray(xyz, np.float64)
        origin = xyz.min(0) if origin is None else np.asarray(origin, np.float64)
        v = np.floor((xyz - origin) / voxel_size).astype(np.int64)
        keep = (v >= 0).all(1) & ((v < shape).all(1) if shape is not None else True)
        v, labels = v[keep], np.asarray(labels, np.int64)[keep]
        code = morton.encode(v[:, 0], v[:, 1], v[:, 2])
        o = np.lexsort((labels, code))
        code, labels, v = code[o], labels[o], v[o]
        # most common class per voxel
        new_pair = np.r_[True, (code[1:] != code[:-1]) | (labels[1:] != labels[:-1])]
        pstart = np.nonzero(new_pair)[0]
        pcount = np.diff(np.r_[pstart, code.size])
        pcode, plab = code[pstart], labels[pstart]
        o2 = np.lexsort((-pcount, pcode))
        first = np.r_[True, pcode[o2][1:] != pcode[o2][:-1]]
        win = o2[first]
        ucode, inv = np.unique(code, return_inverse=True)
        out_attrs = {}
        for k, a in (attrs or {}).items():
            a = np.asarray(a, np.float64)[keep][o]
            ok = np.isfinite(a)
            with np.errstate(invalid="ignore", divide="ignore"):
                out_attrs[k] = np.bincount(inv, np.where(ok, a, 0)) / np.bincount(inv, ok)
        vx, vy, vz = morton.decode(pcode[win])
        return cls(vx, vy, vz, plab[win], out_attrs, shape, voxel_size, tuple(origin), classes)

    @classmethod
    def from_surfaces(cls, polygons, labels, voxel_size, origin=None, attrs=None, shape=None,
                      classes=None, spacing=None):
        """Voxelise 3D polygons (CityGML surfaces, mesh faces): each polygon
        is sampled on a grid of `spacing` (default half a voxel) in its own
        plane, then as from_points. labels / attrs: one value per polygon.
        Like a point cloud, this marks the surfaces; insides stay empty."""
        sp = voxel_size / 2 if spacing is None else float(spacing)
        pts, owner = [], []
        for k, ring in enumerate(polygons):
            q = _sample_polygon(np.asarray(ring, np.float64), sp)
            pts.append(q)
            owner.append(np.full(len(q), k))
        pts, owner = np.concatenate(pts), np.concatenate(owner)
        a = {n: np.asarray(v, np.float64)[owner] for n, v in (attrs or {}).items()}
        return cls.from_points(pts, np.asarray(labels)[owner], voxel_size, origin, a, shape, classes)

    def summarize(self, x, y, z, size):
        x, y, z = (np.asarray(v, np.int64) for v in (x, y, z))
        s = int(size)
        lo = morton.encode(x, y, z)
        hi = lo + np.uint64(s ** 3)
        counts = np.stack([np.searchsorted(c, hi) - np.searchsorted(c, lo) for c in self.by_class], 1)
        out = Summary(np.zeros(len(x)), counts.astype(np.float64), {})
        a, b = np.searchsorted(self.code, lo), np.searchsorted(self.code, hi)
        for k, (v, cum) in self.attrs.items():
            st = np.empty((len(x), 5))
            st[:, :3] = cum[b] - cum[a]
            vp = np.r_[v, np.nan]
            idx = np.stack([a, b], 1).ravel()        # even entries reduce over [a, b)
            if len(idx):
                st[:, 3] = np.fmin.reduceat(vp, idx)[::2]
                st[:, 4] = np.fmax.reduceat(vp, idx)[::2]
            st[st[:, 0] == 0, 3:] = np.nan
            out.stats[k] = st
        return self._fill_empty(out, s)


def _sample_polygon(ring, sp):
    """Points every `sp` metres inside a planar 3D polygon (its corners if
    it is too small to hold any)."""
    import shapely

    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return ring
    c = ring.mean(0)
    n = np.cross(ring - c, np.roll(ring, -1, 0) - c).sum(0)        # Newell normal
    if not np.any(n):
        return ring
    w = int(np.argmax(np.abs(n)))
    u, v = [a for a in range(3) if a != w]
    poly = shapely.Polygon(ring[:, [u, v]])
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    x0, y0, x1, y1 = poly.bounds
    gu, gv = np.meshgrid(np.arange(x0 + sp / 2, x1, sp), np.arange(y0 + sp / 2, y1, sp), indexing="ij")
    gu, gv = gu.ravel(), gv.ravel()
    inside = shapely.contains_xy(poly, gu, gv)
    gu, gv = gu[inside], gv[inside]
    out = np.empty((gu.size, 3))
    out[:, u], out[:, v] = gu, gv
    out[:, w] = c[w] - (n[u] * (gu - c[u]) + n[v] * (gv - c[v])) / n[w]
    return out if len(out) else ring                  # a polygon smaller than the spacing: its corners
