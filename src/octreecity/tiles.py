"""Tiled city store: one zip file, one Occupancy per square tile.

Any tile (or a rectangle of tiles) can be read without touching the rest,
so a whole city never has to fit in memory. Layout inside the zip:

    meta.json                  tile size, cell, origin, CRS, tile list
    t/<i>_<j>/counts.npy       runs per column (uint16)
    t/<i>_<j>/top.npy          run tops (uint16, voxels)
    t/<i>_<j>/label.npy        run labels (uint8)
    t/<i>_<j>/id__<layer>.npy  object id per run, per id layer (int64)

For a city with an object table, appends and edits, use the folder store
(octreecity.store.Store); this one-zip format is kept for older code.

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
        arrays = [("counts", np.diff(city.offsets).astype(np.uint16)), ("top", city.top),
                  ("label", city.label)] + [(f"id__{k}", v) for k, v in city.ids.items()]
        for key, arr in arrays:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            self.zf.writestr(f"t/{name}/{key}.npy", buf.getvalue())
        self.meta["tiles"][name] = dict(nz=int(city.shape[2]), runs=int(city.n_runs), layers=list(city.ids))

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
        ids = {k: self._arr(name, f"id__{k}") for k in m["tiles"][name].get("layers", [])}
        return Occupancy((self.n, self.n, m["tiles"][name]["nz"]), offsets,
                          self._arr(name, "top"), self._arr(name, "label"), self.cell,
                          (m["x0"] + i * self.tile_size, m["y0"] + j * self.tile_size, m["z0"]),
                          crs=m.get("crs"), ids=ids)

    def index_of(self, x, y):
        m = self.meta
        return int((x - m["x0"]) // self.tile_size), int((y - m["y0"]) // self.tile_size)

    def region(self, i0, j0, i1, j1, nz=None):
        """Tiles i0 <= i < i1, j0 <= j < j1 stitched into one Occupancy.
        Missing tiles become ground-free air; shorter tiles are topped up
        with air to a common height. Id layers come along (-1 where a tile
        has none)."""
        tiles = {(i, j): self.tile(i, j) for i in range(i0, i1) for j in range(j0, j1)}
        nz = nz or max([t.shape[2] for t in tiles.values() if t is not None] + [1])
        n, m = self.n, self.meta
        rows = []
        for i in range(i0, i1):
            row = []
            for j in range(j0, j1):
                t = tiles[(i, j)]
                if t is None:
                    t = Occupancy((n, n, nz), np.arange(n * n + 1, dtype=np.int64),
                                  np.full(n * n, nz, np.uint16), np.full(n * n, AIR, np.uint8), self.cell)
                row.append(t.pad_z(nz))
            rows.append(Occupancy.concat_y(row))
        out = Occupancy.concat_x(rows, self.cell, (m["x0"] + i0 * self.tile_size,
                                                   m["y0"] + j0 * self.tile_size, m["z0"]))
        out.crs = m.get("crs")
        return out


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
