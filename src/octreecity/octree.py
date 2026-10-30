"""Linear (pointerless) octree: cells of any content, sized by that content.

A box of nx * ny * nz voxels of size h metres is tiled by cells: cubes of
side 2**level voxels, aligned to multiples of 2**level. Only the leaves are
stored, sorted by the Morton code of their min corner, so:

    - structure costs 9 bytes per leaf in memory (code 8, level 1) and
      1 byte on disk (codes are implied by the order),
    - point location is a binary search (np.searchsorted),
    - any aligned cube is a contiguous slice of the leaf arrays,
    - the parent cells follow from the codes (parents()), not stored.

Content per leaf (the cell table):
    label    class id (uint8 or uint16), meaning set by `classes`
             (0 = empty; see octreecity.classes, octreecity.urban)
    cells    dict of static columns, each an array with one row per leaf
             (any extra axes allowed: class shares, vectors, ...)
    series   dict of dynamic columns (n_leaves, n_times), one value per
             snapshot in `times`; the tree stays the same across snapshots

    id_columns names of the cell columns that hold object ids (int64,
             -1 = no object, -2 = mixed). They are categories: never
             averaged (see Parents.reduce(how="same")); the object table
             they point into is `objects` (octreecity.objects)
    crs      coordinate reference system of origin (text, e.g. "EPSG:32648")

Every change to the leaves (sorting, split, balance) moves all columns
along; a split copies a leaf's row to its 8 children.

Queries (octreecity.query): tree.select(id=7).neighbours(steps=1),
tree.select(where=...).within(10.0), .parents(1), .siblings(), ...

The box does not need to be a cube: nx, ny, nz must be multiples of the
root size 2**max_level, and the domain is tiled by several root cubes.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import _io, morton
from .classes import EMPTY, Classes

MIXED = 255


@dataclass
class LinearOctree:
    shape: tuple
    max_level: int
    code: np.ndarray
    level: np.ndarray
    label: np.ndarray
    voxel_size: float = 1.0
    origin: tuple = (0.0, 0.0, 0.0)
    classes: Classes = None
    cells: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)
    times: np.ndarray = None
    crs: str = None
    id_columns: tuple = ()
    objects: object = None

    def __post_init__(self):
        self.code = np.asarray(self.code, np.uint64)
        n = self.code.size
        lab = np.asarray(self.label)
        lab = np.broadcast_to(lab, (n,)) if lab.ndim == 0 else lab
        self.label = lab.astype(np.uint16 if lab.size and lab.max() > 254 else
                                (lab.dtype if lab.dtype in (np.uint8, np.uint16) else np.uint8))
        self.cells = {k: np.asarray(v) for k, v in self.cells.items()}
        self.series = {k: np.asarray(v) for k, v in self.series.items()}
        self.times = None if self.times is None else np.asarray(self.times)
        for k, v in list(self.cells.items()) + list(self.series.items()):
            if len(v) != n:
                raise ValueError(f"column {k!r} has {len(v)} rows, expected {n}")
        self.id_columns = tuple(self.id_columns or ())
        for k in self.id_columns:
            if k in self.cells:
                self.cells[k] = self.cells[k].astype(np.int64)
        self._reorder(np.argsort(self.code, kind="stable"))

    def _reorder(self, order):
        self.code = np.ascontiguousarray(self.code[order], dtype=np.uint64)
        self.level = np.ascontiguousarray(np.asarray(self.level)[order], dtype=np.uint8)
        self.label = np.ascontiguousarray(self.label[order])
        self.cells = {k: v[order] for k, v in self.cells.items()}
        self.series = {k: v[order] for k, v in self.series.items()}

    # ------------------------------------------------------------------ build

    @classmethod
    def from_dense(cls, grid, max_level, voxel_size=1.0, origin=(0.0, 0.0, 0.0)):
        """Bottom-up merge of a dense uint8 label grid.

        Eight children merge into their parent when they are all leaves with
        the same label. Labels must be < 255 (255 is the internal 'mixed').
        """
        grid = np.asarray(grid, dtype=np.uint8)
        _check_shape(grid.shape, max_level)
        if (grid == MIXED).any():
            raise ValueError("label 255 is reserved")
        codes, levels, labels = [], [], []
        uni = grid
        for lv in range(max_level + 1):
            if lv < max_level:
                a, b, c = (n // 2 for n in uni.shape)
                blocks = uni.reshape(a, 2, b, 2, c, 2)
                first = blocks[:, 0, :, 0, :, 0]
                same = (blocks == first[:, None, :, None, :, None]).all(axis=(1, 3, 5))
                parent = np.where(same & (first != MIXED), first, MIXED).astype(np.uint8)
                emit = (uni != MIXED) & (np.repeat(np.repeat(np.repeat(
                    parent, 2, 0), 2, 1), 2, 2) == MIXED)
            else:
                parent = None
                emit = uni != MIXED
            ix, iy, iz = np.nonzero(emit)
            s = 1 << lv
            codes.append(morton.encode(ix * s, iy * s, iz * s))
            levels.append(np.full(ix.size, lv, np.uint8))
            labels.append(uni[ix, iy, iz])
            uni = parent
        return cls(grid.shape, max_level, np.concatenate(codes),
                   np.concatenate(levels), np.concatenate(labels),
                   voxel_size, tuple(origin))

    @classmethod
    def from_heightfield(cls, ground, surface, surface_label, nz, max_level,
                         voxel_size=1.0, origin=(0.0, 0.0, 0.0),
                         air_label=EMPTY, ground_label=1):
        """Top-down build from 2.5D columns, without a dense grid.

        Per (x, y) column, heights in voxel units: ground for z < ground,
        surface_label (building, tree, ...) for ground <= z < surface, air
        above. A voxel takes the label at its centre z + 0.5, which matches
        rasterize_columns(). Uses min/max pyramids of the height maps, so the
        cost scales with the number of leaves, not the number of voxels.
        """
        ground = np.asarray(ground, dtype=np.float64)
        surface = np.maximum(np.asarray(surface, dtype=np.float64), ground)
        cls_map = np.broadcast_to(np.asarray(surface_label, np.uint8), ground.shape)
        shape = (*ground.shape, int(nz))
        _check_shape(shape, max_level)

        pyr = {k: [v] for k, v in
               (("gmin", ground), ("gmax", ground), ("smin", surface),
                ("smax", surface), ("cmin", cls_map), ("cmax", cls_map))}
        for _ in range(max_level):
            for k, stack in pyr.items():
                a = stack[-1]
                r = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
                stack.append(r.min(axis=(1, 3)) if k.endswith("min") else r.max(axis=(1, 3)))

        S = 1 << max_level
        ix, iy, iz = (g.ravel() for g in np.meshgrid(
            np.arange(shape[0] // S), np.arange(shape[1] // S),
            np.arange(shape[2] // S), indexing="ij"))
        codes, levels, labels = [], [], []
        for lv in range(max_level, -1, -1):
            s = 1 << lv
            P = {k: v[lv][ix, iy] for k, v in pyr.items()}
            lo = iz * s + 0.5
            hi = (iz + 1) * s - 0.5
            is_air = P["smax"] <= lo
            is_ground = P["gmin"] > hi
            is_obj = (P["gmax"] <= lo) & (P["smin"] > hi) & (P["cmin"] == P["cmax"])
            lab = np.full(ix.size, MIXED, np.uint8)
            lab[is_obj] = P["cmin"][is_obj]
            lab[is_ground] = ground_label
            lab[is_air] = air_label
            done = lab != MIXED
            codes.append(morton.encode(ix[done] * s, iy[done] * s, iz[done] * s))
            levels.append(np.full(done.sum(), lv, np.uint8))
            labels.append(lab[done])
            if lv == 0:
                assert done.all()
                break
            m = ~done
            ix, iy, iz = (_children(ix[m], iy[m], iz[m]))
        return cls(shape, max_level, np.concatenate(codes),
                   np.concatenate(levels), np.concatenate(labels),
                   voxel_size, tuple(origin))

    # ---------------------------------------------------------------- queries

    @property
    def n_leaves(self):
        return self.code.size

    @property
    def nbytes(self):
        return self.code.nbytes + self.level.nbytes + self.label.nbytes

    def corners(self):
        """Min corner (voxel units) of every leaf, as (x, y, z) int64 arrays."""
        return morton.decode(self.code)

    def sizes(self):
        return np.left_shift(1, self.level.astype(np.int64))

    def locate(self, x, y, z):
        """Index of the leaf containing integer voxel (x, y, z)."""
        key = morton.encode(x, y, z)
        return np.searchsorted(self.code, key, side="right") - 1

    def label_at(self, x, y, z):
        return self.label[self.locate(x, y, z)]

    def to_dense(self, column=None):
        """Dense voxel grid of the labels, or of a 1D cell column (e.g. an
        id column: to_dense("id")). Small boxes only."""
        v = self.label if column is None else np.asarray(self.cells[column])
        grid = np.empty(self.shape, v.dtype)
        x, y, z = self.corners()
        for lv in np.unique(self.level):
            s = 1 << int(lv)
            m = self.level == lv
            view = grid.reshape(self.shape[0] // s, s, self.shape[1] // s, s,
                                self.shape[2] // s, s)
            view[x[m] // s, :, y[m] // s, :, z[m] // s, :] = v[m][:, None, None, None]
        return grid

    def level_histogram(self):
        return np.bincount(self.level, minlength=self.max_level + 1)

    def centers(self):
        """Centre of every leaf in metres, (n, 3)."""
        x, y, z = self.corners()
        c = 0.5 * self.sizes()
        h = self.voxel_size
        return np.stack([o + h * (v + c) for o, v in zip(self.origin, (x, y, z))], 1)

    def in_box(self, lo, hi):
        """Leaves that overlap the box [lo, hi) given in voxels, e.g. one
        horizontal layer: tree.in_box((0, 0, 0), (nx, ny, 1))."""
        s = self.sizes()
        out = np.ones(self.n_leaves, bool)
        for v, a, b in zip(self.corners(), lo, hi):
            out &= (v < b) & (v + s > a)
        return out

    def sizes_m(self):
        return self.sizes() * float(self.voxel_size)

    # ---------------------------------------------------------------- queries

    def select(self, id=None, **kw):
        """Leaves by object id, cell column, object-table column or any
        function: a Group (see octreecity.query.Query.select)."""
        from .query import Query

        return Query(self).select(id=id, **kw)

    def group(self, idx, **values):
        """Leaves given by index, as a Group."""
        from .query import Group, Query

        return Group(Query(self), idx, **values)

    def touching(self, id, **kw):
        """Objects whose cells touch object `id` (octreecity.query.Query.touching)."""
        from .query import Query

        return Query(self).touching(id, **kw)

    def near(self, id, metres, **kw):
        """Objects with a cell within `metres` of object `id`."""
        from .query import Query

        return Query(self).near(id, metres, **kw)

    # ------------------------------------------------------------- cell table

    def add_snapshot(self, time, **values):
        """Append one snapshot of dynamic values (one row per leaf each),
        e.g. tree.add_snapshot("08:00", temperature=t8, wind=w8)."""
        self.times = np.array([time]) if self.times is None else np.append(self.times, time)
        nt = len(self.times)
        for k, v in values.items():
            v = np.asarray(v, np.float64 if np.asarray(v).dtype.kind != "f" else None)
            if v.shape != (self.n_leaves,):
                raise ValueError(f"{k!r}: expected {self.n_leaves} values")
            old = self.series.get(k, np.full((self.n_leaves, nt - 1), np.nan, v.dtype))
            self.series[k] = np.concatenate([old, v[:, None]], 1)
        for k in self.series:
            if self.series[k].shape[1] < nt:
                pad = np.full((self.n_leaves, nt - self.series[k].shape[1]), np.nan)
                self.series[k] = np.concatenate([self.series[k], pad], 1)
        return self

    def class_names(self):
        """Name of each leaf's class (needs `classes`)."""
        if self.classes is None:
            raise ValueError("this tree has no classes; set tree.classes")
        return np.array(self.classes.names)[self.label]

    def parents(self, levels_above=0):
        """The internal cells above the leaves (see Parents).

        levels_above > 0 adds that many levels above the root cubes, grouping
        2 x 2 x 2 roots (or fewer at the box edge) per step, up to one cell
        for the whole box when enough levels are added.
        """
        return Parents.of(self, levels_above)

    def check(self):
        """Leaves are aligned, inside the box, and tile it with no gap or overlap."""
        x, y, z = self.corners()
        s = self.sizes()
        assert ((x % s == 0) & (y % s == 0) & (z % s == 0)).all(), "unaligned leaf"
        assert ((x + s <= self.shape[0]) & (y + s <= self.shape[1])
                & (z + s <= self.shape[2])).all(), "leaf outside box"
        assert (s ** 3).sum() == np.prod(self.shape), "volume mismatch"
        span = np.left_shift(np.uint64(1), (3 * self.level).astype(np.uint64))
        assert (self.code[:-1] + span[:-1] <= self.code[1:]).all(), "overlap"
        return True

    # ------------------------------------------------------------- adaptivity

    def split(self, mask):
        """Replace the leaves in mask by their 8 children (same label)."""
        mask = np.asarray(mask, bool) & (self.level > 0)
        if not mask.any():
            return self
        x, y, z = self.corners()
        lv = self.level[mask].astype(np.int64) - 1
        cx, cy, cz = _children(x[mask], y[mask], z[mask], scale=np.left_shift(1, lv))
        keep = ~mask
        take = np.concatenate([np.nonzero(keep)[0], np.repeat(np.nonzero(mask)[0], 8)])
        self.code = np.concatenate([self.code[keep], morton.encode(cx, cy, cz)])
        self.level = np.concatenate([self.level[keep], np.repeat(lv, 8).astype(np.uint8)])
        self.label = self.label[take]
        self.cells = {k: v[take] for k, v in self.cells.items()}
        self.series = {k: v[take] for k, v in self.series.items()}
        self._reorder(np.argsort(self.code, kind="stable"))
        return self

    def refine_where(self, predicate, min_level=0, max_iter=64):
        """Split leaves while predicate(octree) -> bool mask is true.

        predicate sees the current octree and returns, per leaf, whether
        that leaf is too big. Leaves at min_level are never split.
        """
        for _ in range(max_iter):
            m = np.asarray(predicate(self), bool) & (self.level > min_level)
            if not m.any():
                return self
            self.split(m)
        raise RuntimeError("refine_where did not converge")

    def balance(self, connectivity=26):
        """Enforce 2:1 balance: neighbouring leaves differ by at most one level.

        connectivity 6 checks faces, 18 adds edges, 26 adds corners. Needed
        by most finite-volume and finite-element solvers on octree meshes.
        """
        dirs = _directions(connectivity)
        # first pass probes from every small leaf; afterwards a violation can
        # only involve a leaf created by the last split, so probe from those
        cand = np.nonzero(self.level.astype(int) <= self.max_level - 2)[0]
        for _ in range(64):
            x, y, z = (v[cand] for v in self.corners())
            s = np.left_shift(1, self.level[cand].astype(np.int64))
            lv = self.level[cand].astype(int)
            too_big = np.zeros(self.n_leaves, bool)
            prober = np.zeros(cand.size, bool)
            for dx, dy, dz in dirs:
                px = x + s if dx > 0 else x - 1 if dx < 0 else x
                py = y + s if dy > 0 else y - 1 if dy < 0 else y
                pz = z + s if dz > 0 else z - 1 if dz < 0 else z
                inside = ((px >= 0) & (py >= 0) & (pz >= 0) & (px < self.shape[0])
                          & (py < self.shape[1]) & (pz < self.shape[2]))
                nb = self.locate(px[inside], py[inside], pz[inside])
                bad = self.level[nb].astype(int) > lv[inside] + 1
                too_big[nb[bad]] = True
                prober[np.nonzero(inside)[0][bad]] = True
            if not too_big.any():
                return self
            # leaves that found a violation probe again (the neighbour may
            # still be too big after one split), plus every new leaf
            keep = cand[prober]
            keep_key = self.code[keep] * np.uint64(32) + self.level[keep]
            old = self.code[~too_big] * np.uint64(32) + self.level[~too_big]
            self.split(too_big)
            key = self.code * np.uint64(32) + self.level
            cand = np.nonzero(~np.isin(key, old, assume_unique=True) | np.isin(key, keep_key))[0]
        raise RuntimeError("balance did not converge")

    def limit_size_near(self, labels, distance, max_level):
        """Split leaves within `distance` voxels of any voxel whose label is
        in `labels` until they are at most 2**max_level voxels wide.
        Typical use: keep air cells small close to buildings and the ground.

        Works on a coarse grid with cells of the target size c: mark coarse
        cells touched by a source leaf, dilate by ceil(distance / c), then
        test each leaf against the dilated mask with an integral image.
        """
        c = 1 << max_level
        cshape = tuple(n // c for n in self.shape)
        x, y, z = self.corners()
        s = self.sizes()
        src = np.isin(self.label, np.atleast_1d(labels))
        mask = np.zeros(cshape, bool)
        big = src & (s >= c)
        for lv in np.unique(self.level[big]):
            m = big & (self.level == lv)
            k = (1 << int(lv)) // c
            view = mask.reshape(cshape[0] // k, k, cshape[1] // k, k, cshape[2] // k, k)
            view[x[m] // (k * c), :, y[m] // (k * c), :, z[m] // (k * c), :] = True
        small = src & (s < c)
        mask[x[small] // c, y[small] // c, z[small] // c] = True
        r = -(-int(distance) // c)
        if r > 0:
            mask = ndimage.maximum_filter(mask, size=2 * r + 1, mode="constant")
        sat = np.zeros(tuple(n + 1 for n in cshape), np.int64)
        sat[1:, 1:, 1:] = mask.cumsum(0).cumsum(1).cumsum(2)

        def pred(t):
            x, y, z = t.corners()
            k = t.sizes() // c
            big = k >= 2
            x0, y0, z0 = x // c, y // c, z // c
            x1, y1, z1 = x0 + k, y0 + k, z0 + k
            tot = (sat[x1, y1, z1] - sat[x0, y1, z1] - sat[x1, y0, z1] - sat[x1, y1, z0]
                   + sat[x0, y0, z1] + sat[x0, y1, z0] + sat[x1, y0, z0] - sat[x0, y0, z0])
            return big & (tot > 0)

        return self.refine_where(pred, min_level=max_level)

    # --------------------------------------------------------------------- io

    def root_codes(self):
        """Morton codes of the root cubes, sorted."""
        S = 1 << self.max_level
        r = [np.arange(n // S) * S for n in self.shape]
        x, y, z = (g.ravel() for g in np.meshgrid(*r, indexing="ij"))
        return np.sort(morton.encode(x, y, z))

    def to_compact(self):
        """(level, label) per leaf, in Morton order. Codes are implied:
        leaves tile each root cube in order, so code = root base + running
        sum of 8**level. 2 bytes per leaf before compression."""
        return self.level.copy(), self.label.copy()

    @classmethod
    def from_compact(cls, shape, max_level, level, label, voxel_size=1.0,
                     origin=(0.0, 0.0, 0.0)):
        level = np.asarray(level, np.uint8)
        span = np.left_shift(np.uint64(1), 3 * level.astype(np.uint64))
        off = np.concatenate([[np.uint64(0)], np.cumsum(span)[:-1]]).astype(np.uint64)
        root = np.uint64(1) << np.uint64(3 * max_level)
        t = cls(tuple(shape), max_level, np.zeros(level.size, np.uint64), level,
                np.asarray(label), voxel_size, tuple(origin))
        t.code = t.root_codes()[(off // root).astype(np.int64)] + off % root
        return t

    def meta(self):
        return {"format": "octreecity.octree", "version": _io.FORMAT_VERSION, "crs": self.crs,
                "voxel_size": self.voxel_size, "origin": list(self.origin), "shape": list(self.shape),
                "max_level": self.max_level, "id_columns": list(self.id_columns),
                "units": {"voxel_size": "m", "origin": "m"},
                "reserved_ids": {"-1": "no object", "-2": "mixed"}}

    def save(self, path, objects=True, meta=None):
        """Compressed .npz: structure as (level, label) per leaf, plus the
        cell table, the snapshots, the class names, __meta__ (crs, id
        columns, units ...) and, if objects is True and a table is attached,
        the object table. Text columns are stored as fixed-width text;
        columns of other Python objects cannot be saved (ValueError)."""
        level, label = self.to_compact()
        extra = {}
        for k, v in self.cells.items():
            if k in self.id_columns:
                extra.update(_io.pack_ids(f"cell__{k}", v))
            else:
                extra[f"cell__{k}"] = _savable(k, v)
        extra.update({f"series__{k}": _savable(k, v) for k, v in self.series.items()})
        if self.times is not None:
            extra["times"] = _savable("times", self.times)
        if self.classes is not None:
            extra["class_names"], extra["class_parent"] = self.classes.to_arrays()
        if objects:
            extra.update(_io.objects_entry(self.objects))
        np.savez_compressed(path, shape=np.array(self.shape), max_level=self.max_level,
                            level=level, label=label, voxel_size=self.voxel_size,
                            origin=np.array(self.origin),
                            **_io.meta_entry({**self.meta(), **(meta or {})}), **extra)

    @classmethod
    def load(cls, path):
        d = _io.open_npz(path)
        t = cls.from_compact(tuple(int(v) for v in d["shape"]), int(d["max_level"]),
                             d["level"], d["label"], float(d["voxel_size"]),
                             tuple(float(v) for v in d["origin"]))
        meta = _io.read_meta(d)
        ids = meta.get("id_columns", [])
        # from_compact builds leaves already in Morton order: columns line up
        t.cells = {}
        for k in d.files:
            if not k.startswith("cell__"):
                continue
            name = k[6:]
            if name.endswith("__codes"):
                continue
            if name.endswith("__values"):
                name = name[:-8]
                t.cells[name] = _io.unpack_ids(d, f"cell__{name}")
            else:
                t.cells[name] = d[k]
        t.series = {k[8:]: d[k] for k in d.files if k.startswith("series__")}
        t.times = d["times"] if "times" in d.files else None
        if "class_names" in d.files:
            t.classes = Classes.from_arrays(d["class_names"], d["class_parent"])
        t.crs = meta.get("crs")
        t.id_columns = tuple(k for k in ids if k in t.cells)
        t.objects = _io.read_objects(d)
        return t


# ------------------------------------------------------------------- parents

@dataclass
class Parents:
    """Internal cells of a LinearOctree: every cube that holds smaller leaves.

    code, level     Morton code (min corner) and level of each internal cell,
                    sorted by level, then code
    leaf_parent     per leaf, index of its direct parent (-1 for a leaf that
                    is a root cube)
    node_parent     per internal cell, index of its parent (-1 at the top)

    Leaves under an internal cell are a contiguous run of the tree's leaves
    (leaf_range), so summaries over parents are cheap (reduce).
    """
    tree: "LinearOctree"
    code: np.ndarray
    level: np.ndarray
    leaf_parent: np.ndarray
    node_parent: np.ndarray

    @classmethod
    def of(cls, tree, levels_above=0):
        top = tree.max_level + int(levels_above)
        if top > morton.MAX_BITS:
            raise ValueError("too many levels above the roots")
        codes, levels = [], []
        for lv in range(int(tree.level.min()) + 1 if tree.n_leaves else 1, top + 1):
            below = tree.level < lv
            if not below.any():
                continue
            c = np.unique(_mask(tree.code[below], lv))
            codes.append(c)
            levels.append(np.full(c.size, lv, np.uint8))
        code = np.concatenate(codes) if codes else np.zeros(0, np.uint64)
        level = np.concatenate(levels) if levels else np.zeros(0, np.uint8)
        p = cls(tree, code, level, np.full(tree.n_leaves, -1), np.full(code.size, -1))
        p.leaf_parent = p.find(tree.code, tree.level.astype(np.int64) + 1)
        p.node_parent = p.find(code, level.astype(np.int64) + 1)
        return p

    def find(self, codes, levels):
        """Index of the internal cell at `levels` containing each code (-1 if none)."""
        codes = np.asarray(codes, np.uint64)
        levels = np.broadcast_to(np.asarray(levels, np.int64), codes.shape)
        out = np.full(codes.size, -1, np.int64)
        for lv in np.unique(levels):
            m = levels == lv
            sel = np.nonzero(self.level == lv)[0]
            if not sel.size:
                continue
            want = _mask(codes[m], int(lv))
            k = np.searchsorted(self.code[sel], want)
            k = np.minimum(k, sel.size - 1)
            hit = self.code[sel][k] == want
            out[np.nonzero(m)[0][hit]] = sel[k[hit]]
        return out

    def __len__(self):
        return self.code.size

    def corners(self):
        return morton.decode(self.code)

    def sizes(self):
        return np.left_shift(1, self.level.astype(np.int64))

    def centers(self):
        t = self.tree
        x, y, z = self.corners()
        c = 0.5 * self.sizes()
        return np.stack([o + t.voxel_size * (v + c) for o, v in zip(t.origin, (x, y, z))], 1)

    def leaf_range(self):
        """[lo, hi) leaf indices under each internal cell."""
        span = np.left_shift(np.uint64(1), 3 * self.level.astype(np.uint64))
        lo = np.searchsorted(self.tree.code, self.code)
        hi = np.searchsorted(self.tree.code, self.code + span)
        return lo, hi

    def ancestors(self):
        """(n_leaves, depth) table of ancestor indices, nearest first, -1 padded."""
        chain = [self.leaf_parent]
        while (chain[-1] >= 0).any():
            nxt = np.where(chain[-1] >= 0, self.node_parent[np.maximum(chain[-1], 0)], -1)
            chain.append(nxt)
        return np.stack(chain[:-1], 1) if len(chain) > 1 else np.zeros((self.tree.n_leaves, 0), int)

    def reduce(self, values, how=None, weight="volume"):
        """Summary of a leaf column over each internal cell.

        values: an array with one row per leaf, or the name of a cell column.
        how: mean (weighted by leaf volume, or by `weight` per leaf; NaN
        values are skipped), sum, min, max, or same: the value all leaves
        under the cell share, else -2 (for categories such as object ids,
        exact for int64). values may have extra axes (e.g. class shares
        (n, C)). Default: same for an id column, else mean. Id columns are
        never averaged: mean / sum on one raises ValueError.
        """
        is_id = isinstance(values, str) and values in self.tree.id_columns
        if isinstance(values, str):
            values = self.tree.cells[values]
        how = how or ("same" if is_id else "mean")
        if how == "same":
            return self._same(np.asarray(values))
        if how not in ("mean", "sum", "min", "max"):
            raise ValueError(f"unknown how={how!r}")
        raw = np.asarray(values)
        if is_id or (how in ("mean", "sum") and raw.dtype.kind in "iu" and raw.size
                     and np.abs(raw.astype(np.float64)).max() >= 2 ** 53):
            raise ValueError("these values are object ids (or too big for float64): they must not "
                             "be averaged or summed; use how='same'")
        v = np.asarray(values, np.float64)
        lo, hi = self.leaf_range()
        if how in ("min", "max"):
            f = np.fmin if how == "min" else np.fmax
            out = np.full((len(lo),) + v.shape[1:], np.nan)
            vp = np.concatenate([v, np.full((1,) + v.shape[1:], np.nan)])
            for lv in np.unique(self.level):        # same-level ranges never overlap
                m = np.nonzero(self.level == lv)[0]
                idx = np.stack([lo[m], hi[m]], 1).ravel()
                out[m] = f.reduceat(vp, idx, axis=0)[::2]
            return out
        w = (self.tree.sizes().astype(np.float64) ** 3 if isinstance(weight, str)
             else np.asarray(weight, np.float64))
        w = w.reshape((-1,) + (1,) * (v.ndim - 1))
        ok = np.isfinite(v)
        cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(np.where(ok, v * (w if how == "mean" else 1), 0), 0)])
        tot = cs[hi] - cs[lo]
        if how == "sum":
            return tot
        cw = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(np.where(ok, np.broadcast_to(w, v.shape), 0), 0)])
        den = cw[hi] - cw[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, tot / den, np.nan)

    def _same(self, v, mixed=-2):
        lo, hi = self.leaf_range()
        if not len(lo):
            return np.zeros(0, v.dtype)
        vp = np.r_[v, v[-1:]]                       # hi may equal n_leaves
        idx = np.stack([lo, hi], 1).ravel()          # even entries reduce over [lo, hi)
        mn = np.minimum.reduceat(vp, idx)[::2]
        mx = np.maximum.reduceat(vp, idx)[::2]
        return np.where(mn == mx, mn, np.asarray(mixed, v.dtype))


def _savable(name, v):
    """An array np.load can read back without pickle: text objects become
    fixed-width text; other Python objects are refused."""
    v = np.asarray(v)
    if v.dtype != object:
        return v
    flat = v.ravel().tolist()
    if all(isinstance(x, str) for x in flat):
        return np.array(flat, dtype=str).reshape(v.shape)
    raise ValueError(f"column {name!r} holds Python objects, which cannot be saved; "
                     "use numbers or text (or keep the values in the object table)")


def _mask(code, level):
    """Morton code of the aligned cube of `level` that contains each code."""
    keep = ~np.uint64((1 << (3 * int(level))) - 1) if 3 * int(level) < 64 else np.uint64(0)
    return np.asarray(code, np.uint64) & keep


# ------------------------------------------------------------------- helpers

def rasterize_columns(ground, surface, surface_label, nz,
                      air_label=EMPTY, ground_label=1):
    """Dense label grid from 2.5D columns (same rule as from_heightfield)."""
    ground = np.asarray(ground, np.float64)[..., None]
    surface = np.maximum(np.asarray(surface, np.float64)[..., None], ground)
    lab = np.broadcast_to(np.asarray(surface_label, np.uint8), ground.shape[:2])[..., None]
    zc = np.arange(nz) + 0.5
    out = np.full((*ground.shape[:2], nz), air_label, np.uint8)
    out = np.where(zc < surface, lab, out)
    out = np.where(zc < ground, np.uint8(ground_label), out)
    return out.astype(np.uint8)


def pad_to_multiple(n, max_level):
    s = 1 << max_level
    return -(-int(n) // s) * s


def _check_shape(shape, max_level):
    s = 1 << max_level
    if any(n % s for n in shape):
        raise ValueError(f"shape {shape} must be a multiple of 2**max_level = {s}")
    if max(shape) > morton.MAX_COORD:
        raise ValueError("box too large for 21-bit Morton codes")


def _children(x, y, z, scale=None):
    """8 children of each cell. Without scale, (x, y, z) are cell indices at
    the parent level and children are indices one level down. With scale,
    (x, y, z) are voxel corners and scale is the child size."""
    off = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])
    if scale is None:
        bx, by, bz = 2 * x, 2 * y, 2 * z
        scale = 1
    else:
        bx, by, bz = x, y, z
    sc = np.asarray(scale)[..., None] if np.ndim(scale) else scale
    cx = (bx[:, None] + off[:, 0] * sc).ravel()
    cy = (by[:, None] + off[:, 1] * sc).ravel()
    cz = (bz[:, None] + off[:, 2] * sc).ravel()
    return cx, cy, cz


def _directions(connectivity):
    d = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
         if (i, j, k) != (0, 0, 0)]
    nz = {6: 1, 18: 2, 26: 3}[connectivity]
    return [v for v in d if sum(map(abs, v)) <= nz]
