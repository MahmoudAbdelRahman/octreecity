"""Column store: each (x, y) column of voxels as runs of equal labels along z.

A city is mostly vertical walls, flat roofs and layers. Run-length coding
along z stores a wall once per column instead of once per voxel, so the
size follows the footprint outline, not the wall area. Columns may hold any
number of runs: ground, tree trunk air, canopy, air; bridges; overhangs.

Layout (CSR):
    offsets[c] .. offsets[c+1]   runs of column c = x * ny + y
    top[k]                       exclusive top of run k, in voxels (uint16)
    label[k]                     label of run k (uint8)
Runs start at z = 0 and the last run of each column ends at nz. Adjacent
runs always have different labels.
"""

from dataclasses import dataclass

import numpy as np

from . import morton
from .octree import AIR, GROUND, LinearOctree


@dataclass
class ColumnCity:
    shape: tuple
    offsets: np.ndarray
    top: np.ndarray
    label: np.ndarray
    voxel_size: float = 1.0
    origin: tuple = (0.0, 0.0, 0.0)
    crs: str = None

    # ------------------------------------------------------------------ build

    @classmethod
    def from_dense(cls, grid, voxel_size=1.0, origin=(0.0, 0.0, 0.0)):
        grid = np.asarray(grid, np.uint8)
        nx, ny, nz = grid.shape
        end = np.ones(grid.shape, bool)
        end[..., :-1] = grid[..., 1:] != grid[..., :-1]
        flat = end.reshape(nx * ny, nz)
        col, z = np.nonzero(flat)
        offsets = np.zeros(nx * ny + 1, np.int64)
        np.cumsum(flat.sum(1), out=offsets[1:])
        return cls(grid.shape, offsets, (z + 1).astype(np.uint16),
                   grid.reshape(nx * ny, nz)[col, z], voxel_size, tuple(origin))

    @classmethod
    def from_heightfield(cls, ground, surface, surface_label, nz, voxel_size=1.0,
                         origin=(0.0, 0.0, 0.0), air_label=AIR, ground_label=GROUND):
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
    def from_layers(cls, shape, layers, default=AIR, voxel_size=1.0,
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
        ny, nz = parts[0].shape[1:]
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p.offsets[1:] + base)
            base += p.top.size
        nx = sum(p.shape[0] for p in parts)
        return cls((nx, ny, nz), np.concatenate(offs), np.concatenate([p.top for p in parts]),
                   np.concatenate([p.label for p in parts]), voxel_size, tuple(origin))

    def __repr__(self):
        nx, ny, nz = self.shape
        c = self.voxel_size
        from .builder import LABELS
        labs = [LABELS.get(int(v), str(v)) for v in np.unique(self.label) if v != 0]
        return (f"ColumnCity {nx * c:g} x {ny * c:g} x {nz * c:g} m at {c:g} m cells "
                f"({nx} x {ny} columns, {self.n_runs:,} runs, {self.nbytes / 1e6:.1f} MB in memory)\n"
                f"  contains: {', '.join(labs)}\n"
                f"  origin: ({self.origin[0]:.1f}, {self.origin[1]:.1f}, {self.origin[2]:.1f}) m"
                + (f" in {self.crs}" if self.crs else ""))

    def crop(self, x, y, width, depth):
        """Sub-area in metres from the city's corner (x, y, width, depth)."""
        c = self.voxel_size
        i0, j0 = int(round(x / c)), int(round(y / c))
        i1, j1 = min(self.shape[0], i0 + int(round(width / c))), min(self.shape[1], j0 + int(round(depth / c)))
        sub = ColumnCity.from_dense(self.to_dense(i0, i1)[:, j0:j1], c,
                                    (self.origin[0] + i0 * c, self.origin[1] + j0 * c, self.origin[2]))
        sub.crs = self.crs
        return sub

    # ---------------------------------------------------------------- queries

    @property
    def n_runs(self):
        return self.top.size

    @property
    def nbytes(self):
        return self.offsets.nbytes + self.top.nbytes + self.label.nbytes

    def bottoms(self):
        b = np.empty_like(self.top)
        b[1:] = self.top[:-1]
        b[self.offsets[:-1][np.diff(self.offsets) > 0]] = 0
        return b

    def run_columns(self):
        return np.repeat(np.arange(self.offsets.size - 1), np.diff(self.offsets))

    def label_at(self, x, y, z):
        nz = self.shape[2]
        col = np.asarray(x, np.int64) * self.shape[1] + np.asarray(y, np.int64)
        key = self.run_columns() * (nz + 1) + self.top.astype(np.int64)
        k = np.searchsorted(key, col * (nz + 1) + np.asarray(z, np.int64), side="right")
        return self.label[k]

    def top_of(self, labels):
        """Highest top of any run with a label in labels, per column (0 if none)."""
        m = np.isin(self.label, labels)
        out = np.zeros(self.offsets.size - 1, np.int64)
        np.maximum.at(out, self.run_columns()[m], self.top[m].astype(np.int64))
        return out.reshape(self.shape[:2])

    def to_dense(self, x0=0, x1=None):
        x1 = self.shape[0] if x1 is None else x1
        ny, nz = self.shape[1:]
        k0, k1 = self.offsets[x0 * ny], self.offsets[x1 * ny]
        lengths = (self.top[k0:k1].astype(np.int64) - self.bottoms()[k0:k1])
        return np.repeat(self.label[k0:k1], lengths).reshape(x1 - x0, ny, nz)

    def to_octree(self, max_level, strip=None):
        """Octree of the same voxels, built strip by strip (no full dense grid).

        The top is filled up with air to a multiple of the largest cell
        (2**max_level voxels); x and y must already be multiples of it.
        """
        S = 1 << max_level
        if self.shape[0] % S or self.shape[1] % S:
            raise ValueError(f"x and y sizes {self.shape[:2]} must be multiples of 2**max_level = {S}; "
                             "choose a smaller max_level or a city size that divides evenly")
        nz = -(-self.shape[2] // S) * S
        strip = S if strip is None else strip
        if strip % S:
            raise ValueError("strip must be a multiple of 2**max_level")
        codes, levels, labels = [], [], []
        for x0 in range(0, self.shape[0], strip):
            x1 = min(self.shape[0], x0 + strip)
            d = self.to_dense(x0, x1)
            if nz != d.shape[2]:
                d = np.concatenate([d, np.full(d.shape[:2] + (nz - d.shape[2],), AIR, np.uint8)], 2)
            t = LinearOctree.from_dense(d, max_level)
            x, y, z = t.corners()
            codes.append(morton.encode(x + x0, y, z))
            levels.append(t.level)
            labels.append(t.label)
        return LinearOctree((*self.shape[:2], nz), max_level, np.concatenate(codes),
                            np.concatenate(levels), np.concatenate(labels),
                            self.voxel_size, self.origin)

    # --------------------------------------------------------------------- io

    def save(self, path):
        np.savez_compressed(path, shape=np.array(self.shape), counts=np.diff(self.offsets).astype(np.uint16),
                            top=self.top, label=self.label, voxel_size=self.voxel_size,
                            origin=np.array(self.origin), crs=str(self.crs or ""))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        offsets = np.zeros(d["counts"].size + 1, np.int64)
        np.cumsum(d["counts"], out=offsets[1:])
        crs = str(d["crs"]) if "crs" in d.files and str(d["crs"]) else None
        return cls(tuple(int(v) for v in d["shape"]), offsets, d["top"], d["label"],
                   float(d["voxel_size"]), tuple(float(v) for v in d["origin"]), crs)
