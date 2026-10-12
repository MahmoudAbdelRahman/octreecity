"""Linear (pointerless) octree for labelled city voxels.

A city is a box of nx * ny * nz voxels of size h metres. Each voxel has a
uint8 label (air, ground, building, tree, ... the meaning is up to the
caller). The octree stores only leaves: cubes of side 2**level voxels,
aligned to multiples of 2**level, that hold one label. Leaves are kept
sorted by the Morton code of their min corner, so:

    - in memory: 10 bytes per leaf (code 8, level 1, label 1),
    - on disk: 2 bytes per leaf (level, label); codes are implied by order,
    - point location is a binary search (np.searchsorted),
    - any aligned cube is a contiguous slice of the leaf arrays.

The box does not need to be a cube: nx, ny, nz must be multiples of the
root size 2**max_level, and the domain is tiled by several root cubes.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import morton

MIXED = 255

AIR = 0
GROUND = 1
BUILDING = 2
TREE = 3


@dataclass
class LinearOctree:
    shape: tuple
    max_level: int
    code: np.ndarray
    level: np.ndarray
    label: np.ndarray
    voxel_size: float = 1.0
    origin: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        order = np.argsort(self.code, kind="stable")
        self.code = np.ascontiguousarray(self.code[order], dtype=np.uint64)
        self.level = np.ascontiguousarray(self.level[order], dtype=np.uint8)
        self.label = np.ascontiguousarray(self.label[order], dtype=np.uint8)

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
                         air_label=AIR, ground_label=GROUND):
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

    def to_dense(self):
        grid = np.empty(self.shape, np.uint8)
        x, y, z = self.corners()
        for lv in np.unique(self.level):
            s = 1 << int(lv)
            m = self.level == lv
            view = grid.reshape(self.shape[0] // s, s, self.shape[1] // s, s,
                                self.shape[2] // s, s)
            view[x[m] // s, :, y[m] // s, :, z[m] // s, :] = \
                self.label[m][:, None, None, None]
        return grid

    def level_histogram(self):
        return np.bincount(self.level, minlength=self.max_level + 1)

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
        self.code = np.concatenate([self.code[keep], morton.encode(cx, cy, cz)])
        self.level = np.concatenate([self.level[keep], np.repeat(lv, 8).astype(np.uint8)])
        self.label = np.concatenate([self.label[keep], np.repeat(self.label[mask], 8)])
        self.__post_init__()
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
        for _ in range(64):
            x, y, z = self.corners()
            s = self.sizes()
            too_big = np.zeros(self.n_leaves, bool)
            for dx, dy, dz in dirs:
                px = np.where(dx > 0, x + s, np.where(dx < 0, x - 1, x))
                py = np.where(dy > 0, y + s, np.where(dy < 0, y - 1, y))
                pz = np.where(dz > 0, z + s, np.where(dz < 0, z - 1, z))
                inside = ((px >= 0) & (py >= 0) & (pz >= 0) & (px < self.shape[0])
                          & (py < self.shape[1]) & (pz < self.shape[2]))
                nb = self.locate(px[inside], py[inside], pz[inside])
                bad = self.level[nb].astype(int) > self.level[inside].astype(int) + 1
                too_big[nb[bad]] = True
            if not too_big.any():
                return self
            self.split(too_big)
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
                np.asarray(label, np.uint8), voxel_size, tuple(origin))
        t.code = t.root_codes()[(off // root).astype(np.int64)] + off % root
        return t

    def save(self, path):
        level, label = self.to_compact()
        np.savez_compressed(path, shape=np.array(self.shape), max_level=self.max_level,
                            level=level, label=label, voxel_size=self.voxel_size,
                            origin=np.array(self.origin))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls.from_compact(tuple(int(v) for v in d["shape"]), int(d["max_level"]),
                                d["level"], d["label"], float(d["voxel_size"]),
                                tuple(float(v) for v in d["origin"]))


# ------------------------------------------------------------------- helpers

def rasterize_columns(ground, surface, surface_label, nz,
                      air_label=AIR, ground_label=GROUND):
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
