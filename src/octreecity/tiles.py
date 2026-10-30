"""Tiled city store: one zip file, one Occupancy per square tile.

Any tile (or a rectangle of tiles) can be read without touching the rest,
so a whole city never has to fit in memory. Layout inside the zip:

    meta.json                  tile size, cell, origin, CRS, tile list
    t/<i>_<j>/counts.npy       runs per column (uint16)
    t/<i>_<j>/top.npy          run tops (uint16, voxels)
    t/<i>_<j>/label.npy        run labels (uint8)

Tile (i, j) covers x0 + i * tile .. x0 + (i + 1) * tile (same for y). All
tiles share the z origin and cell size; each has its own height (nz).
"""

import io
import json
import zipfile

import numpy as np

from .occupancy import Occupancy
from .classes import EMPTY as AIR  # empty space


class TiledCityWriter:
    def __init__(self, path, x0, y0, z0, tile, cell, crs=None, extra=None):
        self.path = path
        self.zf = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6)
        self.meta = dict(format="octreecity.tiles", version=1, x0=float(x0), y0=float(y0),
                         z0=float(z0), tile=float(tile), cell=float(cell), crs=crs,
                         tiles={}, extra=extra or {})

    def add(self, i, j, city):
        name = f"{i}_{j}"
        for key, arr in (("counts", np.diff(city.offsets).astype(np.uint16)),
                         ("top", city.top), ("label", city.label)):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            self.zf.writestr(f"t/{name}/{key}.npy", buf.getvalue())
        self.meta["tiles"][name] = dict(nz=int(city.shape[2]), runs=int(city.n_runs))

    def close(self):
        self.zf.writestr("meta.json", json.dumps(self.meta))
        self.zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TiledCity:
    def __init__(self, path):
        self.path = path
        self.zf = zipfile.ZipFile(path, "r")
        self.meta = json.loads(self.zf.read("meta.json"))
        m = self.meta
        self.tile_size, self.cell = m["tile"], m["cell"]
        self.n = int(round(m["tile"] / m["cell"]))

    @property
    def tiles(self):
        return [tuple(map(int, k.split("_"))) for k in self.meta["tiles"]]

    def _arr(self, name, key):
        return np.lib.format.read_array(io.BytesIO(self.zf.read(f"t/{name}/{key}.npy")))

    def tile(self, i, j):
        name = f"{i}_{j}"
        m = self.meta
        if name not in m["tiles"]:
            return None
        counts = self._arr(name, "counts")
        offsets = np.zeros(counts.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        return Occupancy((self.n, self.n, m["tiles"][name]["nz"]), offsets,
                          self._arr(name, "top"), self._arr(name, "label"), self.cell,
                          (m["x0"] + i * self.tile_size, m["y0"] + j * self.tile_size, m["z0"]))

    def index_of(self, x, y):
        m = self.meta
        return int((x - m["x0"]) // self.tile_size), int((y - m["y0"]) // self.tile_size)

    def region(self, i0, j0, i1, j1, nz=None):
        """Tiles i0 <= i < i1, j0 <= j < j1 stitched into one Occupancy.
        Missing tiles become ground-free air; shorter tiles are topped up
        with air to a common height."""
        tiles = {(i, j): self.tile(i, j) for i in range(i0, i1) for j in range(j0, j1)}
        nz = nz or max([t.shape[2] for t in tiles.values() if t is not None] + [1])
        n, ny = self.n, (j1 - j0) * self.n
        counts, tops, labs = [], [], []
        for i in range(i0, i1):
            row = [tiles[(i, j)] for j in range(j0, j1)]
            for x in range(n):
                for j, t in enumerate(row):
                    c, tp, lb = _column_slice(t, x, n, nz)
                    counts.append(c)
                    tops.append(tp)
                    labs.append(lb)
        counts = np.concatenate(counts)
        offsets = np.zeros(counts.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        m = self.meta
        return Occupancy(((i1 - i0) * n, ny, nz), offsets, np.concatenate(tops),
                          np.concatenate(labs), self.cell,
                          (m["x0"] + i0 * self.tile_size, m["y0"] + j0 * self.tile_size, m["z0"]))


def _column_slice(t, x, n, nz):
    """Runs of the n columns at row x of tile t, raised to height nz."""
    if t is None:
        return np.ones(n, np.int64), np.full(n, nz, np.uint16), np.full(n, AIR, np.uint8)
    k0, k1 = t.offsets[x * n], t.offsets[(x + 1) * n]
    counts = np.diff(t.offsets[x * n:(x + 1) * n + 1])
    top = t.top[k0:k1].copy()
    lab = t.label[k0:k1]
    if nz == t.shape[2]:
        return counts, top, lab
    last = t.offsets[x * n + 1:(x + 1) * n + 1] - 1 - k0
    air_last = lab[last] == AIR
    top[last[air_last]] = nz
    # columns ending in a solid run get an extra air run on top
    need = np.nonzero(~air_last)[0]
    if need.size == 0:
        return counts, top, lab
    ins = last[need] + 1
    top = np.insert(top, ins, nz)
    lab = np.insert(lab, ins, AIR)
    counts = counts.copy()
    counts[need] += 1
    return counts, top, lab


def _build_one(args):
    from .builder import CityBuilder
    from .features import Features
    import shapely

    i, j, x0, y0, tile, cell, air_above, bld, layers = args
    b = CityBuilder(x0, y0, tile, tile, cell)
    for wkbs, label, line_width in layers:
        if wkbs:
            b.add_landcover(Features(shapely.from_wkb(wkbs)), label, line_width=line_width)
    tops = [0.0]
    if bld[0]:
        wkbs, h, base = bld
        b.add_buildings(Features(shapely.from_wkb(wkbs), height=h, base=base))
        tops.append(max(h))
    nz = int(np.ceil((max(tops) + cell + air_above) / cell))
    return i, j, b.build(nz=nz)


def build_tiled(path, buildings, cell=1.0, tile=256.0, bbox=None, landcover=(), workers=8,
                air_above=20.0, crs=None, progress=None):
    """Build a whole city into a tiled store, in parallel.

    buildings: Features with heights (flat terrain). landcover: list of
    (Features, label) or (Features, label, line_width). bbox: (x0, y0, x1,
    y1), default the buildings' extent snapped to the tile grid. Tiles with
    no building are skipped. Returns the number of tiles written.
    """
    from concurrent.futures import ProcessPoolExecutor
    import shapely

    if bbox is None:
        bx0, by0, bx1, by1 = buildings.bounds()
        bx0, by0 = np.floor(bx0 / tile) * tile, np.floor(by0 / tile) * tile
    else:
        bx0, by0, bx1, by1 = bbox
    ni, nj = int(np.ceil((bx1 - bx0) / tile)), int(np.ceil((by1 - by0) / tile))
    btree = shapely.STRtree(buildings.geoms)
    lc = [(f, lab, (rest[0] if rest else None), shapely.STRtree(f.geoms)) for f, lab, *rest in landcover]
    wkb_b = shapely.to_wkb(buildings.geoms)
    h = np.where(np.isfinite(buildings.height), buildings.height, 10.0)
    jobs = []
    for i in range(ni):
        for j in range(nj):
            x0, y0 = bx0 + i * tile, by0 + j * tile
            box = shapely.box(x0, y0, x0 + tile, y0 + tile)
            idx = btree.query(box)
            if idx.size == 0:
                continue
            layers = []
            for f, lab, lw, tr in lc:
                k = tr.query(box)
                layers.append((list(shapely.to_wkb(f.geoms[k])), lab, None if lw is None else np.asarray(lw)[k] if np.ndim(lw) else lw))
            jobs.append((i, j, x0, y0, tile, cell, air_above,
                         (list(wkb_b[idx]), h[idx].tolist(), buildings.base[idx].tolist()), layers))
    with TiledCityWriter(path, bx0, by0, -cell, tile, cell, crs=crs) as w:
        with ProcessPoolExecutor(workers) as ex:
            for n, (i, j, city) in enumerate(ex.map(_build_one, jobs, chunksize=4)):
                w.add(i, j, city)
                if progress:
                    progress(n + 1, len(jobs))
    return len(jobs)
