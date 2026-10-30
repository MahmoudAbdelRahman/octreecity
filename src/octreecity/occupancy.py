"""Occupancy: the semantic occupancy of a place, which voxels are filled and
by what class (here "occupancy" never means people), and which object
each voxel belongs to.

Storage: each (x, y) column of voxels as runs along z.

A city is mostly vertical walls, flat roofs and layers. Run-length coding
along z stores a wall once per column instead of once per voxel, so the
size follows the footprint outline, not the wall area. Columns may hold any
number of runs: ground, tree trunk air, canopy, air; bridges; overhangs;
blocks stacked on blocks.

Layout (CSR):
    offsets[c] .. offsets[c+1]   runs of column c = x * ny + y
    top[k]                       exclusive top of run k, in voxels (uint16)
    label[k]                     class of run k (uint8)
    ids[layer][k]                object id of run k in each id layer
                                 (int64; -1 = no object)
Runs start at z = 0 and the last run of each column ends at nz. Two
adjacent runs differ in their label or in at least one id.

Id layers: one per kind of object that can share space with another kind,
named by you, e.g. "id" for buildings, streets and trees and "zone_id" for
land-use zones. All of them point into one object table
(octreecity.objects.ObjectTable), attached as `objects`. Air may hold ids
too (a named courtyard, a room).
"""

from dataclasses import dataclass, field

import numpy as np

from . import _io
from .classes import EMPTY, Classes
from .octree import LinearOctree  # noqa: F401  (kept for older code)

NONE = -1


@dataclass
class Occupancy:
    shape: tuple
    offsets: np.ndarray
    top: np.ndarray
    label: np.ndarray
    voxel_size: float = 1.0
    origin: tuple = (0.0, 0.0, 0.0)
    crs: str = None
    classes: Classes = None
    ids: dict = field(default_factory=dict)
    objects: object = None

    def __post_init__(self):
        self.ids = {k: np.asarray(v, np.int64) for k, v in (self.ids or {}).items()}
        for k, v in self.ids.items():
            if v.shape != (len(self.top),):
                raise ValueError(f"id layer {k!r} has {v.size} values, expected one per run "
                                 f"({len(self.top)})")

    # ------------------------------------------------------------------ build

    @classmethod
    def from_dense(cls, grid, voxel_size=1.0, origin=(0.0, 0.0, 0.0), ids=None):
        """Runs from a dense label grid (nx, ny, nz), and optional dense id
        grids {layer: (nx, ny, nz) int64}. Runs split where the label or
        any id changes."""
        grid = np.asarray(grid, np.uint8)
        ids = {k: np.asarray(v, np.int64) for k, v in (ids or {}).items()}
        nx, ny, nz = grid.shape
        end = np.ones(grid.shape, bool)
        diff = grid[..., 1:] != grid[..., :-1]
        for g in ids.values():
            if g.shape != grid.shape:
                raise ValueError("id grids must have the shape of the label grid")
            diff |= g[..., 1:] != g[..., :-1]
        end[..., :-1] = diff
        flat = end.reshape(nx * ny, nz)
        col, z = np.nonzero(flat)
        offsets = np.zeros(nx * ny + 1, np.int64)
        np.cumsum(flat.sum(1), out=offsets[1:])
        return cls(grid.shape, offsets, (z + 1).astype(np.uint16),
                   grid.reshape(nx * ny, nz)[col, z], voxel_size, tuple(origin),
                   ids={k: g.reshape(nx * ny, nz)[col, z] for k, g in ids.items()})

    @classmethod
    def from_heightfield(cls, ground, surface, surface_label, nz, voxel_size=1.0,
                         origin=(0.0, 0.0, 0.0), air_label=EMPTY, ground_label=1):
        """Same rule as LinearOctree.from_heightfield (voxel centre z + 0.5),
        built directly as runs, without a dense grid."""
        ground = np.asarray(ground, np.float64)
        surface = np.maximum(np.asarray(surface, np.float64), ground)
        lab = np.broadcast_to(np.asarray(surface_label, np.uint8), ground.shape).ravel()
        ng = np.clip(np.ceil(ground - 0.5), 0, nz).astype(np.int64).ravel()
        no = np.clip(np.ceil(surface - 0.5), 0, nz).astype(np.int64).ravel()
        no = np.maximum(no, ng)
        has = np.stack([ng > 0, no > ng, no < nz], 1)
        tops = np.stack([ng, no, np.full_like(ng, nz)], 1)
        labs = np.stack([np.full_like(lab, ground_label), lab, np.full_like(lab, air_label)], 1)
        counts = has.sum(1)
        offsets = np.zeros(ground.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        return cls((*ground.shape, int(nz)), offsets, tops[has].astype(np.uint16),
                   labs[has].astype(np.uint8), voxel_size, tuple(origin))

    @classmethod
    def from_layers(cls, shape, layers, default=EMPTY, voxel_size=1.0,
                    origin=(0.0, 0.0, 0.0), strip=16):
        """Paint layers in order: each is (bottom, top, label) with bottom/top
        rasters in voxel units (voxel filled if bottom <= z + 0.5 < top) and
        label a scalar or raster. Later layers overwrite earlier ones. Works
        in x strips, so memory stays small for any city size."""
        nx, ny, nz = shape
        zc = np.arange(nz) + 0.5
        parts = []
        for x0 in range(0, nx, strip):
            x1 = min(nx, x0 + strip)
            g = np.full((x1 - x0, ny, nz), default, np.uint8)
            for bot, top, lab in layers:
                b = np.broadcast_to(np.asarray(bot, np.float64), (nx, ny))[x0:x1, :, None]
                t = np.broadcast_to(np.asarray(top, np.float64), (nx, ny))[x0:x1, :, None]
                lb = np.broadcast_to(np.asarray(lab, np.uint8), (nx, ny))[x0:x1, :, None]
                m = (b <= zc) & (zc < t)
                g = np.where(m, lb, g)
            parts.append(cls.from_dense(g))
        return cls.concat_x(parts, voxel_size, origin)

    @classmethod
    def concat_x(cls, parts, voxel_size=1.0, origin=(0.0, 0.0, 0.0)):
        """Join strips along x (same ny and nz). A layer missing in a strip
        is -1 there."""
        ny, nz = parts[0].shape[1:]
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p.offsets[1:] + base)
            base += p.top.size
        nx = sum(p.shape[0] for p in parts)
        layers = list(dict.fromkeys(k for p in parts for k in p.ids))
        ids = {k: np.concatenate([p.ids.get(k, np.full(p.n_runs, NONE, np.int64)) for p in parts])
               for k in layers}
        return cls((nx, ny, nz), np.concatenate(offs), np.concatenate([p.top for p in parts]),
                   np.concatenate([p.label for p in parts]), voxel_size, tuple(origin), ids=ids)

    @classmethod
    def concat_y(cls, parts, voxel_size=None, origin=None):
        """Join blocks along y (same nx and nz), e.g. tiles of one row."""
        nx, nz = parts[0].shape[0], parts[0].shape[2]
        if any(p.shape[0] != nx or p.shape[2] != nz for p in parts):
            raise ValueError("parts must have the same nx and nz (see pad_z)")
        ny = sum(p.shape[1] for p in parts)
        layers = list(dict.fromkeys(k for p in parts for k in p.ids))
        # new column of every (part, column), then runs in new column order
        newcol, lo, cnt, base = [], [], [], 0
        run_base = 0
        for p in parts:
            pny = p.shape[1]
            x, y = np.divmod(np.arange(nx * pny), pny)
            newcol.append(x * ny + base + y)
            lo.append(p.offsets[:-1] + run_base)
            cnt.append(np.diff(p.offsets))
            base += pny
            run_base += p.n_runs
        newcol, lo, cnt = np.concatenate(newcol), np.concatenate(lo), np.concatenate(cnt)
        o = np.argsort(newcol, kind="stable")
        lo, cnt = lo[o], cnt[o]
        offsets = np.zeros(nx * ny + 1, np.int64)
        np.cumsum(cnt, out=offsets[1:])
        run = np.repeat(lo - offsets[:-1], cnt) + np.arange(int(cnt.sum()))
        top = np.concatenate([p.top for p in parts])[run]
        label = np.concatenate([p.label for p in parts])[run]
        ids = {k: np.concatenate([p.ids.get(k, np.full(p.n_runs, NONE, np.int64)) for p in parts])[run]
               for k in layers}
        p0 = parts[0]
        return cls((nx, ny, nz), offsets, top, label, p0.voxel_size if voxel_size is None else voxel_size,
                   p0.origin if origin is None else tuple(origin), p0.crs, p0.classes, ids, p0.objects)

    def pad_z(self, nz, label=EMPTY):
        """The same city in a taller box: every column is topped up with
        `label` (empty) voxels with no object, up to nz."""
        nz = int(nz)
        if nz < self.shape[2]:
            raise ValueError("pad_z only makes the box taller")
        if nz == self.shape[2]:
            return self
        last = self.offsets[1:] - 1
        same = self.label[last] == label
        for v in self.ids.values():
            same &= v[last] == NONE
        top = self.top.astype(np.int64).copy()
        top[last[same]] = nz
        add = np.nonzero(~same)[0]                      # columns that need a new run
        pos = last[add] + 1
        top = np.insert(top, pos, nz)
        lab = np.insert(self.label, pos, label)
        ids = {k: np.insert(v, pos, NONE) for k, v in self.ids.items()}
        counts = np.diff(self.offsets)
        counts[add] += 1
        offsets = np.zeros(counts.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        return Occupancy((self.shape[0], self.shape[1], nz), offsets, top.astype(np.uint16), lab,
                         self.voxel_size, self.origin, self.crs, self.classes, ids, self.objects)

    def __repr__(self):
        nx, ny, nz = self.shape
        c = self.voxel_size
        name = (lambda v: self.classes.name(v)) if self.classes is not None else str
        labs = [name(int(v)) for v in np.unique(self.label) if v != EMPTY]
        lay = "; ".join(f"{k}: {np.unique(v[v >= 0]).size:,} objects" for k, v in self.ids.items())
        return (f"Occupancy {nx * c:g} x {ny * c:g} x {nz * c:g} m at {c:g} m cells "
                f"({nx} x {ny} columns, {self.n_runs:,} runs, {self.nbytes / 1e6:.1f} MB in memory)\n"
                f"  contains: {', '.join(labs)}\n"
                + (f"  id layers: {lay}\n" if self.ids else "")
                + f"  origin: ({self.origin[0]:.1f}, {self.origin[1]:.1f}, {self.origin[2]:.1f}) m"
                + (f" in {self.crs}" if self.crs else ""))

    def crop(self, x, y, width, depth):
        """Sub-area in metres from the city's corner (x, y, width, depth).
        Runs and ids are copied as they are."""
        c = self.voxel_size
        i0, j0 = int(round(x / c)), int(round(y / c))
        i1 = min(self.shape[0], i0 + int(round(width / c)))
        j1 = min(self.shape[1], j0 + int(round(depth / c)))
        return self.crop_columns(i0, j0, i1, j1)

    def crop_columns(self, i0, j0, i1, j1):
        """Columns i0 <= i < i1, j0 <= j < j1 (voxel indices), runs and ids
        copied as they are."""
        ny = self.shape[1]
        i0, j0 = max(0, int(i0)), max(0, int(j0))
        i1, j1 = min(self.shape[0], int(i1)), min(ny, int(j1))
        cols = (np.arange(i0, i1)[:, None] * ny + np.arange(j0, j1)[None, :]).ravel()
        lo, hi = self.offsets[cols], self.offsets[cols + 1]
        n = hi - lo
        offsets = np.zeros(cols.size + 1, np.int64)
        np.cumsum(n, out=offsets[1:])
        run = np.repeat(lo - offsets[:-1], n) + np.arange(int(n.sum()))
        h = self.voxel_size
        sub = Occupancy((i1 - i0, j1 - j0, self.shape[2]), offsets, self.top[run], self.label[run], h,
                        (self.origin[0] + i0 * h, self.origin[1] + j0 * h, self.origin[2]),
                        self.crs, self.classes, {k: v[run] for k, v in self.ids.items()}, self.objects)
        return sub

    # ---------------------------------------------------------------- queries

    @property
    def n_runs(self):
        return self.top.size

    @property
    def layers(self):
        """Names of the id layers."""
        return tuple(self.ids)

    @property
    def nbytes(self):
        return (self.offsets.nbytes + self.top.nbytes + self.label.nbytes
                + sum(v.nbytes for v in self.ids.values()))

    def bottoms(self):
        b = np.empty_like(self.top)
        b[1:] = self.top[:-1]
        b[self.offsets[:-1][np.diff(self.offsets) > 0]] = 0
        return b

    def run_columns(self):
        return np.repeat(np.arange(self.offsets.size - 1), np.diff(self.offsets))

    def _run_at(self, x, y, z):
        nz = self.shape[2]
        col = np.asarray(x, np.int64) * self.shape[1] + np.asarray(y, np.int64)
        key = self.run_columns() * (nz + 1) + self.top.astype(np.int64)
        return np.searchsorted(key, col * (nz + 1) + np.asarray(z, np.int64), side="right")

    def label_at(self, x, y, z):
        return self.label[self._run_at(x, y, z)]

    def id_at(self, x, y, z, layer="id"):
        """Object id of voxels (x, y, z) (integer voxel indices) in a layer."""
        return self.ids[layer][self._run_at(x, y, z)]

    def top_of(self, labels):
        """Highest top of any run with a label in labels, per column (0 if
        none), in voxels counted from the bottom of the box (z = 0), so the
        ground layer is included. For a height above the ground use
        height_of."""
        m = np.isin(self.label, labels)
        out = np.zeros(self.offsets.size - 1, np.int64)
        np.maximum.at(out, self.run_columns()[m], self.top[m].astype(np.int64))
        return out.reshape(self.shape[:2])

    def height_of(self, labels, base_labels):
        """Height in metres of the highest run with a label in labels above
        the top of the runs with a label in base_labels (e.g. buildings
        above the ground), per column; 0 where there is none."""
        top = self.top_of(labels)
        base = self.top_of(base_labels)
        return np.where(top > 0, np.maximum(top - base, 0), 0) * float(self.voxel_size)

    def to_dense(self, x0=0, x1=None, layer=None):
        """Dense (x1 - x0, ny, nz) grid of labels, or of ids of `layer`."""
        x1 = self.shape[0] if x1 is None else x1
        ny, nz = self.shape[1:]
        k0, k1 = self.offsets[x0 * ny], self.offsets[x1 * ny]
        lengths = (self.top[k0:k1].astype(np.int64) - self.bottoms()[k0:k1])
        v = self.label if layer is None else self.ids[layer]
        return np.repeat(v[k0:k1], lengths).reshape(x1 - x0, ny, nz)

    def voxel_counts(self, layer="id"):
        """(ids, number of voxels) of every id in a layer (-1 included)."""
        n = self.top.astype(np.int64) - self.bottoms()
        u, inv = np.unique(self.ids[layer], return_inverse=True)
        return u, np.bincount(inv, n, minlength=u.size).astype(np.int64)

    def voxels(self, id=None, layer="id", label=None, members=True):
        """Voxel indices (n, 3) of the voxels of object(s) `id` in `layer`
        and/or with class `label`. With members=True, voxels of overlap rows
        that list the object count too (needs `objects`)."""
        m = np.ones(self.n_runs, bool)
        if id is not None:
            want = np.atleast_1d(np.asarray(id, np.int64))
            if members and self.objects is not None:
                want = np.union1d(want, self.objects.overlaps_of(want))
            m &= np.isin(self.ids[layer], want)
        if label is not None:
            m &= np.isin(self.label, np.atleast_1d(label))
        run = np.nonzero(m)[0]
        bot = self.bottoms()[run].astype(np.int64)
        n = self.top[run].astype(np.int64) - bot
        z = np.repeat(bot - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(int(n.sum()))
        x, y = np.divmod(np.repeat(self.run_columns()[run], n), self.shape[1])
        return np.stack([x, y, z], 1)

    def to_octree(self, max_level, split_ids=True, rule=None, **kw):
        """Octree of the same voxels: the smallest tree whose leaves each hold
        one class and, with split_ids, one object per id layer (tree.cells
        holds each layer as exact int64 ids). Built top-down from the runs
        (octreecity.build with OccupancySource), without a voxel grid.

        The box is topped up with empty space to a multiple of the largest
        cell (2**max_level voxels) in x, y and z, so any city size works: the
        extra cells lie beyond the city's far (east and north) edges and the
        top, and merge into large empty cells.
        """
        from .build import build
        from .rules import changes, mixed
        from .sources import OccupancySource

        if rule is None:
            rule = mixed()
            if split_ids:
                for k in self.ids:
                    rule = rule | changes(k)
        return build(OccupancySource(self), rule, max_level, **kw)

    # --------------------------------------------------------------------- io

    def meta(self):
        return {"format": "octreecity.occupancy", "version": _io.FORMAT_VERSION, "crs": self.crs,
                "voxel_size": self.voxel_size, "origin": list(self.origin), "shape": list(self.shape),
                "units": {"top": "voxels", "voxel_size": "m", "origin": "m"},
                "id_layers": list(self.ids), "reserved_ids": {"-1": "no object", "-2": "mixed"}}

    def save(self, path, objects=True, meta=None):
        """Compressed .npz. Keeps the keys of earlier versions, plus the id
        layers (id__<layer>), __meta__ and, if objects is True and a table is
        attached, the object table (__objects__)."""
        extra = {}
        for k, v in self.ids.items():
            extra.update(_io.pack_ids(f"id__{k}", v))
        if self.classes is not None:
            extra["class_names"], extra["class_parent"] = self.classes.to_arrays()
        if objects:
            extra.update(_io.objects_entry(self.objects))
        np.savez_compressed(path, shape=np.array(self.shape), counts=np.diff(self.offsets).astype(np.uint16),
                            top=self.top, label=self.label, voxel_size=self.voxel_size,
                            origin=np.array(self.origin), crs=str(self.crs or ""),
                            **_io.meta_entry({**self.meta(), **(meta or {})}), **extra)

    @classmethod
    def load(cls, path):
        d = _io.open_npz(path)
        offsets = np.zeros(d["counts"].size + 1, np.int64)
        np.cumsum(d["counts"], out=offsets[1:])
        crs = str(d["crs"]) if "crs" in d.files and str(d["crs"]) else None
        classes = (Classes.from_arrays(d["class_names"], d["class_parent"])
                   if "class_names" in d.files else None)
        ids = {k: _io.unpack_ids(d, f"id__{k}") for k in _io.id_keys(d.files, "id__")}
        meta = _io.read_meta(d)
        order = meta.get("id_layers")
        if order:
            ids = {k: ids[k] for k in order if k in ids} | ids
        return cls(tuple(int(v) for v in d["shape"]), offsets, d["top"], d["label"],
                   float(d["voxel_size"]), tuple(float(v) for v in d["origin"]), crs, classes,
                   ids, _io.read_objects(d))
