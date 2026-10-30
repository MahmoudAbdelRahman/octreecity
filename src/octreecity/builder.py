"""Turn Features and rasters into an Occupancy, one rectangle at a time,
keeping which object each voxel belongs to.

    b = CityBuilder(x0, y0, width=1000, depth=1000, cell=1.0)
    b.add_terrain(dem)                     # raster (m) or callable(x, y)
    b.add_landcover(parks, GRASS)          # top ground voxel takes the label
    b.add_trees(canopies, top=12, base=3)  # canopy with air below
    b.add_buildings(buildings, ids=ids)    # Features with height (+ min_height)
    b.add_zones(landuse, zone_ids)         # a second id layer ("zone_id")
    city = b.build()                       # Occupancy (+ city.ids, city.objects)

All lengths in metres, x east and y north, in the CRS of the Features.
Rasters are indexed [i, j] with i along x and j along y, at cell centres
x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell.

One rule for geometry, label and id together: a column belongs to a shape
when its cell centre is inside the shape (holes respected; lines are
buffered by line_width / 2 first). A centre that lies exactly on a shared
wall goes to one side only, so two buildings that touch never claim the
same column. Along z, a voxel is filled when bottom <= z_centre < top.

Ids: every add_* call takes ids (one per shape) and an id layer name
(default "id"). Within one call, two shapes that claim the same voxel
overlap, and `overlap` decides:
    "keep"   (default) the voxel gets an overlap row in the object table,
             with the claimants as members (one row per set of claimants;
             nothing is lost). A shape over its own ancestor (a building
             part over its building) simply replaces it.
    "first" / "last"  the first / last shape given wins
    "by"     the highest `priority` wins (an array, or the name of an
             object-table column, e.g. "width_m")
Between calls, the later call wins; build() warns when that takes voxels
from objects of an earlier call in the same layer.
"""

import warnings

import numpy as np
import shapely

from .objects import NONE, ObjectTable
from .occupancy import Occupancy
from .urban import CLASSES, AIR, BUILDING, GRASS, GROUND, LABELS, PAVEMENT, ROAD, TREE, WATER  # noqa: F401

# the test point is moved off the cell centre by these fractions of a cell
# (an irrational ratio), so a centre on an edge goes to exactly one side
_SHIFT = (1.234567e-7, 2.345679e-7 * np.pi / 3)
_LINES = ("LineString", "MultiLineString", "LinearRing")
_POINTS = ("Point", "MultiPoint")


def footprint_pairs(geoms, x0, y0, nx, ny, cell, line_width=None, chunk=1 << 20):
    """(shape index, column index) for every column whose cell centre is
    inside a shape. Column index = i * ny + j. Lines and points are
    buffered by line_width / 2 (lines with flat ends); without a width they
    claim nothing."""
    geoms = list(geoms)
    widths = (None if line_width is None
              else np.broadcast_to(np.asarray(line_width, np.float64), (len(geoms),)))
    F, C = [], []
    dx, dy = _SHIFT[0] * cell, _SHIFT[1] * cell
    for k, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        if g.geom_type in _LINES or g.geom_type in _POINTS:
            if widths is None or not np.isfinite(widths[k]) or widths[k] <= 0:
                continue
            g = shapely.buffer(g, widths[k] / 2, cap_style="flat" if g.geom_type in _LINES else "round")
        elif g.geom_type == "GeometryCollection":
            g = shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])
            if g.is_empty:
                continue
        bx0, by0, bx1, by1 = g.bounds
        i0 = max(0, int(np.floor((bx0 - x0) / cell - 0.5)))
        i1 = min(nx, int(np.ceil((bx1 - x0) / cell + 0.5)))
        j0 = max(0, int(np.floor((by0 - y0) / cell - 0.5)))
        j1 = min(ny, int(np.ceil((by1 - y0) / cell + 0.5)))
        if i1 <= i0 or j1 <= j0:
            continue
        shapely.prepare(g)
        ys = y0 + (np.arange(j0, j1) + 0.5) * cell + dy
        step = max(1, chunk // (j1 - j0))
        for a in range(i0, i1, step):
            b = min(i1, a + step)
            xs = x0 + (np.arange(a, b) + 0.5) * cell + dx
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            inside = shapely.contains_xy(g, X.ravel(), Y.ravel())
            ii, jj = np.divmod(np.nonzero(inside)[0], j1 - j0)
            F.append(np.full(ii.size, k, np.int64))
            C.append((ii + a) * ny + (jj + j0))
    if not F:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(F), np.concatenate(C)


def rasterize(geoms, values, x0, y0, nx, ny, cell, fill=0.0, dtype=np.float32, line_width=None):
    """Burn shapes into an [nx, ny] raster, later shapes on top.

    A cell takes a shape's value when its centre is inside the shape (holes
    respected; a centre on a shared edge goes to one side only). Lines are
    buffered by line_width / 2 first (e.g. road widths). NaN values are
    skipped.
    """
    values = np.broadcast_to(np.asarray(values, np.float64), (len(geoms),))
    f, col = footprint_pairs(geoms, x0, y0, nx, ny, cell, line_width)
    ok = np.isfinite(values[f])
    f, col = f[ok], col[ok]
    out = np.full(nx * ny, fill, np.float64)
    o = np.argsort(f, kind="stable")                    # later shapes written last
    out[col[o]] = values[f[o]]
    return out.reshape(nx, ny).astype(dtype)


class CityBuilder:
    """See the module docstring. objects: the ObjectTable the ids point into
    (overlap rows are added to it); a new one by default."""

    def __init__(self, x0, y0, width, depth, cell=1.0, objects=None, crs=None):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.nx, self.ny = int(round(width / cell)), int(round(depth / cell))
        self.terrain = np.zeros((self.nx, self.ny))
        self.objects = ObjectTable() if objects is None else objects
        self.crs = crs
        self.ops = []
        self.layers = []
        self.stats = {}

    def centres(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def add_terrain(self, dem):
        """dem: [nx, ny] raster of ground elevation (m) or callable(x, y)."""
        self.terrain = np.asarray(dem(*self.centres()) if callable(dem) else dem, np.float64)
        return self

    # ------------------------------------------------------------ shape ops

    def _shapes(self, kind, feats, label, ids, layer, overlap, priority, line_width, **z):
        if overlap not in ("keep", "first", "last", "by"):
            raise ValueError("overlap must be 'keep', 'first', 'last' or 'by'")
        geoms = list(feats.geoms) if hasattr(feats, "geoms") else list(feats)
        n = len(geoms)
        if ids is not None:
            ids = np.broadcast_to(np.asarray(ids, np.int64), (n,)).copy()
            if layer not in self.layers:
                self.layers.append(layer)
        f, col = footprint_pairs(geoms, self.x0, self.y0, self.nx, self.ny, self.cell, line_width)
        if overlap == "by":
            if priority is None:
                raise ValueError("overlap='by' needs priority= (an array, or an object-table column)")
            if isinstance(priority, str):
                if ids is None:
                    raise ValueError("priority by column needs ids")
                priority = self.objects.column(priority)[self.objects.rows(ids)]
            key = np.argsort(np.argsort(np.asarray(priority, np.float64), kind="stable"), kind="stable")
        elif overlap == "first":
            key = np.arange(n)[::-1].copy()
        else:
            key = np.arange(n)
        o = np.lexsort((key[f], col))                   # by column, the winner last
        lab = None if label is None else np.broadcast_to(np.asarray(label, np.uint8), (n,))
        self.ops.append(dict(kind=kind, f=f[o], col=col[o], ids=ids, label=lab, layer=layer,
                             overlap=overlap, n=n, index=len(self.ops), **z))
        return self

    def add_shapes(self, feats, label, bottom, top, ids=None, layer="id", overlap="keep",
                   priority=None, line_width=None):
        """Any extruded shapes: bottom / top in metres above the terrain (one
        value or one per shape), e.g. a bridge deck over a road."""
        n = len(feats)
        return self._shapes("solid", feats, label, ids, layer, overlap, priority, line_width,
                            bottom=np.broadcast_to(np.asarray(bottom, np.float64), (n,)),
                            top=np.broadcast_to(np.asarray(top, np.float64), (n,)), ground="terrain")

    def add_landcover(self, feats, label, line_width=None, ids=None, layer="id", overlap="keep",
                      priority=None):
        """Top ground voxel takes `label` (and the id) inside the shapes
        (water, grass, road ...)."""
        return self._shapes("cover", feats, label, ids, layer, overlap, priority, line_width)

    def add_buildings(self, feats, label=BUILDING, default_height=10.0, ids=None, layer="id",
                      overlap="keep", priority=None):
        """Buildings stand on the terrain: base = feature base if known, else
        the lowest terrain under the footprint, never above the terrain (no
        gap below). A building fills base + min_height .. base + height:
        min_height (Features.min_height, 0 when unknown) lifts parts that
        start above the ground, such as blocks stacked on other blocks.
        Each shape is painted on its own, so stacked and overlapping shapes
        keep their own voxels."""
        h = np.where(np.isfinite(feats.height), feats.height, default_height)
        mh = getattr(feats, "min_height", None)
        mh = np.zeros(len(feats)) if mh is None else np.where(np.isfinite(mh), mh, 0.0)
        return self._shapes("solid", feats, label, ids, layer, overlap, priority, None,
                            bottom=mh, top=h, base=np.asarray(feats.base, np.float64),
                            ground="building")

    def add_trees(self, feats, top=None, base=None, label=TREE, ids=None, layer="id",
                  overlap="keep", priority=None, line_width=None):
        """Canopies: feats with height = canopy top above ground (or `top`),
        `base` = crown base above ground (default 0.3 * top). Air stays
        below. Points need line_width (the crown diameter)."""
        h = feats.height if top is None else np.full(len(feats), float(top))
        h = np.where(np.isfinite(h), h, 10.0)
        b = 0.3 * h if base is None else np.broadcast_to(np.asarray(base, np.float64), h.shape)
        return self._shapes("solid", feats, label, ids, layer, overlap, priority, line_width,
                            bottom=b, top=h, ground="terrain")

    def add_zones(self, feats, ids, layer="zone_id", on="solid", overlap="keep", priority=None):
        """Another kind of object that shares space with the first one (land-
        use zones, districts ...): ids only, no label, in their own layer,
        on the voxels chosen by `on`: "solid" (every non-air voxel of the
        column), "ground" (the top ground voxel) or "all". Painted after
        everything else."""
        if on not in ("solid", "ground", "all"):
            raise ValueError("on must be 'solid', 'ground' or 'all'")
        return self._shapes("zone", feats, None, ids, layer, overlap, priority, None, on=on)

    def add_solid(self, base_abs, top_abs, label, ids=None, layer="id"):
        """Any raster layer of absolute bottom/top elevations (nan = none);
        ids: one id for all or a raster of ids (-1 = none)."""
        if ids is not None and layer not in self.layers:
            self.layers.append(layer)
        self.ops.append(dict(kind="raster", bottom=np.asarray(base_abs, np.float64),
                             top=np.asarray(top_abs, np.float64), label=label, layer=layer,
                             ids=None if ids is None else np.broadcast_to(
                                 np.asarray(ids, np.int64), (self.nx, self.ny)),
                             index=len(self.ops)))
        return self

    # --------------------------------------------------------------- build

    def _z_ranges(self, op, z0):
        """Per pair (or per column for rasters): bottom and top in voxel
        units from z0."""
        c, ter = self.cell, self.terrain.ravel()
        if op["kind"] == "raster":
            return (op["bottom"] - z0) / c, (op["top"] - z0) / c
        if op["kind"] in ("cover", "zone"):
            g = (ter[op["col"]] - z0) / c
            return g - 1.0, g
        f, col = op["f"], op["col"]
        t = ter[col]
        if op["ground"] == "building":
            low = np.full(op["n"], np.inf)
            np.minimum.at(low, f, t)
            base = np.where(np.isfinite(op["base"][f]), op["base"][f], low[f])
            t = np.minimum(base, t)
        return (t + op["bottom"][f] - z0) / c, (t + op["top"][f] - z0) / c

    def build(self, air_above=20.0, ground_thickness=None, nz=None, strip=16, prune=True):
        """Occupancy with city.ids (one int64 per run per id layer) and
        city.objects. z = 0 is one ground_thickness (default one cell) below
        the lowest terrain point; the box reaches air_above over the top."""
        c = self.cell
        gt = c if ground_thickness is None else float(ground_thickness)
        z0 = float(self.terrain.min() - gt)
        for op in self.ops:
            op["zb"], op["zt"] = self._z_ranges(op, z0)
        tops = [np.nanmax(op["zt"]) for op in self.ops if op["kind"] in ("solid", "raster")
                and np.isfinite(op["zt"]).any()]
        zmax = max([(self.terrain.max() - z0) / c] + tops)
        nz = int(np.ceil(zmax + air_above / c)) if nz is None else int(nz)
        if nz > 65535:
            raise ValueError("too tall: more than 65535 voxels along z")
        if "id" in self.layers:                          # the main layer first
            self.layers.remove("id")
            self.layers.insert(0, "id")
        rank = {"cover": 0, "solid": 1, "raster": 1, "zone": 2}
        ops = sorted(self.ops, key=lambda o: (rank[o["kind"]], o["index"]))
        self.stats = {k: {"overlap_rows": 0, "overlap_voxels": 0, "max_members": 0,
                          "overwritten_voxels": 0, "voxels": 0} for k in self.layers}
        self._rows, self._new_rows = {}, set()
        parts = [self._strip(ops, x0, min(self.nx, x0 + strip), nz, z0)
                 for x0 in range(0, self.nx, strip)]
        city = Occupancy.concat_x(parts, c, (self.x0, self.y0, z0))
        city.ids = {k: city.ids.get(k, np.full(city.n_runs, NONE, np.int64)) for k in self.layers}
        city.crs, city.classes = self.crs, CLASSES
        if self.layers or len(self.objects):
            city.objects = self.objects
        self._finish(city, prune)
        return city

    def _finish(self, city, prune):
        present = set()
        for v in city.ids.values():
            present.update(np.unique(v).tolist())
        orphans = [i for i in self._new_rows if i not in present]
        if prune and orphans:
            self.objects.drop(orphans)
        for k, st in self.stats.items():
            u, n = city.voxel_counts(k)
            st["voxels"] = int(n[u >= 0].sum())
            st["objects"] = int((u >= 0).sum())
            rows = [i for i in self._new_rows if i in present and i in self.objects]
            mine = [i for i in rows if self.objects._src[self.objects._row[i]].startswith(f"overlap:{k}:")]
            st["overlap_rows"] = len(mine)
            ov = np.isin(u, mine)
            st["overlap_voxels"] = int(n[ov].sum())
            st["max_members"] = max([len(self.objects.members(i)) for i in mine], default=0)
            share = st["overlap_voxels"] / max(st["voxels"], 1)
            if st["max_members"] >= 6 or share > 0.5:
                warnings.warn(f"id layer {k!r}: shapes overlap a lot ({st['overlap_rows']} overlap rows, "
                              f"up to {st['max_members']} objects in one voxel, {share:.0%} of its voxels). "
                              "Are these buffers? Put them in another layer, or use overlap='first', "
                              "'last' or 'by'.", stacklevel=3)
            if st["overwritten_voxels"]:
                warnings.warn(f"id layer {k!r}: later calls painted over {st['overwritten_voxels']:,} voxels "
                              "of objects from earlier calls (later calls win; see builder.stats).",
                              stacklevel=3)

    def _strip(self, ops, x0, x1, nz, z0):
        ny, c = self.ny, self.cell
        sx = x1 - x0
        zc = np.arange(nz) + 0.5
        g = (self.terrain[x0:x1] - z0) / c
        lab = np.where(zc < g[..., None], np.uint8(GROUND), np.uint8(AIR))
        grids = {k: np.full((sx, ny, nz), NONE, np.int64) for k in self.layers}
        painted = {k: np.full((sx, ny, nz), -1, np.int32) for k in self.layers}   # op index per voxel
        flat_lab = lab.reshape(-1)
        c0, c1 = x0 * ny, x1 * ny
        for op in ops:
            if op["kind"] == "raster":
                self._paint_raster(op, x0, x1, lab, grids, painted, zc)
                continue
            a, b = np.searchsorted(op["col"], c0), np.searchsorted(op["col"], c1)
            if a == b:
                continue
            f, col = op["f"][a:b], op["col"][a:b] - c0
            if op["kind"] == "zone":
                if op["on"] == "solid":
                    m = lab.reshape(sx * ny, nz)[col] != AIR
                elif op["on"] == "all":
                    m = np.ones((col.size, nz), bool)
                else:
                    m = (op["zb"][a:b, None] <= zc) & (zc < op["zt"][a:b, None])
                p, z = np.nonzero(m)
            else:
                k0 = np.clip(np.ceil(op["zb"][a:b] - 0.5), 0, nz).astype(np.int64)
                k1 = np.clip(np.ceil(op["zt"][a:b] - 0.5), 0, nz).astype(np.int64)
                n = np.maximum(k1 - k0, 0)
                p = np.repeat(np.arange(col.size), n)
                z = np.repeat(k0 - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(int(n.sum()))
            if not p.size:
                continue
            vox = col[p] * nz + z                        # pairs are sorted by column, winner last
            o = np.lexsort((np.arange(vox.size), vox))   # stable: keeps winner-last order per voxel
            vox, fp = vox[o], f[p][o]
            last = np.r_[vox[1:] != vox[:-1], True]
            win_vox, win_f = vox[last], fp[last]
            if op["label"] is not None:
                flat_lab[win_vox] = op["label"][win_f]
            if op["ids"] is None:
                continue
            new = op["ids"][win_f]
            if op["overlap"] == "keep":
                first = np.r_[True, vox[1:] != vox[:-1]]
                multi = ~(first & last)                  # voxels with 2+ claims
                if multi.any():
                    new = self._resolve(op, vox, fp, first, last, new, win_vox)
            grid = grids[op["layer"]].reshape(-1)
            pnt = painted[op["layer"]].reshape(-1)
            old, old_op = grid[win_vox], pnt[win_vox]
            clash = (old >= 0) & (old != new) & (old_op >= 0) & (old_op != op["index"])
            if clash.any():
                self._count_overwrite(op["layer"], old[clash], new[clash])
            grid[win_vox] = new
            pnt[win_vox] = op["index"]
        return Occupancy.from_dense(lab, voxel_size=c, ids=grids)

    def _paint_raster(self, op, x0, x1, lab, grids, painted, zc):
        b = op["zb"][x0:x1, :, None] if np.ndim(op["zb"]) else op["zb"]
        t = op["zt"][x0:x1, :, None] if np.ndim(op["zt"]) else op["zt"]
        m = (np.nan_to_num(b, nan=np.inf) <= zc) & (zc < np.nan_to_num(t, nan=-np.inf))
        lb = np.broadcast_to(np.asarray(op["label"], np.uint8), (self.nx, self.ny))[x0:x1, :, None]
        lab[...] = np.where(m, lb, lab)
        if op["ids"] is not None:
            k = op["layer"]
            new = np.broadcast_to(op["ids"][x0:x1, :, None], m.shape)
            clash = m & (grids[k] >= 0) & (grids[k] != new) & (painted[k] >= 0)
            if clash.any():
                self._count_overwrite(k, grids[k][clash], new[clash])
            grids[k][m] = new[m]
            painted[k][m] = op["index"]

    def _count_overwrite(self, layer, old, new):
        """Voxels a later call took from another object (not its own part)."""
        pairs, cnt = np.unique(np.stack([old, new], 1), axis=0, return_counts=True)
        objs = self.objects
        n = 0
        for (a, b), k in zip(pairs.tolist(), cnt.tolist()):
            if b >= 0 and not objs.is_ancestor(a, b) and not (a in objs and set(objs.members(a)) >= {b}):
                n += k
        self.stats[layer]["overwritten_voxels"] += n

    def _resolve(self, op, vox, fp, first, last, new, win_vox):
        """keep mode: one id per voxel claimed by several shapes of one call.
        The claimants' set (minus ancestors of other claimants) gives one
        overlap row; a single remaining claimant keeps its own id."""
        ids = op["ids"][fp]
        start = np.nonzero(first)[0]
        cnt = np.diff(np.r_[start, vox.size])
        multi = np.nonzero(cnt > 1)[0]
        out = new.copy()
        objs = self.objects
        for w in multi:                               # w-th distinct voxel (same order as win_vox)
            s = start[w]
            key = tuple(sorted(set(ids[s:s + cnt[w]].tolist())))
            if len(key) == 1:
                out[w] = key[0]
                continue
            if key not in self._rows:
                keep = [i for i in key if not any(objs.is_ancestor(i, j) for j in key if j != i)]
                if len(keep) == 1:
                    self._rows[key] = keep[0]
                else:
                    before = len(objs)
                    rid = objs.add_overlap(keep, layer=op["layer"])
                    if len(objs) > before:
                        self._new_rows.add(rid)
                    self._rows[key] = rid
            out[w] = self._rows[key]
        return out
