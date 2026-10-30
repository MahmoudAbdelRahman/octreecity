"""Turn Features and rasters into an Occupancy, one rectangle at a time.

    b = CityBuilder(x0, y0, width=1000, depth=1000, cell=1.0)
    b.add_terrain(dem)                     # raster (m) or callable(x, y)
    b.add_landcover(parks, GRASS)          # top ground voxel takes the label
    b.add_trees(canopies, top=12, base=3)  # canopy with air below
    b.add_buildings(buildings)             # Features with height (+ base)
    city = b.build()                       # Occupancy

All lengths in metres, x east and y north, in the CRS of the Features.
Rasters are indexed [i, j] with i along x and j along y, at cell centres
x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell.
"""

import numpy as np
import shapely
from PIL import Image, ImageDraw

from .occupancy import Occupancy
from .urban import CLASSES, AIR, BUILDING, GRASS, GROUND, LABELS, PAVEMENT, ROAD, TREE, WATER  # noqa: F401


def rasterize(geoms, values, x0, y0, nx, ny, cell, fill=0.0, dtype=np.float32, line_width=None):
    """Burn shapes into an [nx, ny] raster, later shapes on top.

    Polygons fill the cells whose centres fall inside (holes respected).
    Lines are buffered by line_width / 2 first (e.g. road widths).
    Uses PIL's scanline fill: fast enough for a whole city tile by tile.
    """
    img = Image.new("F", (nx, ny), float(fill))
    draw = ImageDraw.Draw(img)
    values = np.broadcast_to(np.asarray(values, np.float64), (len(geoms),))
    widths = None if line_width is None else np.broadcast_to(np.asarray(line_width, np.float64), (len(geoms),))

    def px(coords):
        c = np.asarray(coords)[:, :2]
        return [((x - x0) / cell - 0.5, (y - y0) / cell - 0.5) for x, y in c]

    for k, (g, v) in enumerate(zip(geoms, values)):
        if g is None or g.is_empty or not np.isfinite(v):
            continue
        if g.geom_type in ("LineString", "MultiLineString") and widths is not None:
            g = g.buffer(widths[k] / 2, cap_style="flat")
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon":
                continue
            draw.polygon(px(p.exterior.coords), fill=float(v))
            for hole in p.interiors:
                draw.polygon(px(hole.coords), fill=float(fill))
    # PIL images are (width=x, height=y); we index [x, y]
    return np.asarray(img, dtype=dtype).T.copy()


class CityBuilder:
    def __init__(self, x0, y0, width, depth, cell=1.0):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.nx, self.ny = int(round(width / cell)), int(round(depth / cell))
        self.terrain = np.zeros((self.nx, self.ny))
        self.cover = np.full((self.nx, self.ny), GROUND, np.uint8)
        self.solids = []          # (base_abs, top_abs, label) rasters, painted in order

    def centres(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def _burn(self, geoms, values, fill=np.nan, line_width=None):
        return rasterize(geoms, values, self.x0, self.y0, self.nx, self.ny, self.cell,
                         fill=fill, dtype=np.float64, line_width=line_width)

    def add_terrain(self, dem):
        """dem: [nx, ny] raster of ground elevation (m) or callable(x, y)."""
        self.terrain = np.asarray(dem(*self.centres()) if callable(dem) else dem, np.float64)
        return self

    def add_landcover(self, feats, label, line_width=None):
        """Top ground voxel takes `label` inside the shapes (water, grass, road...)."""
        m = ~np.isnan(self._burn(feats.geoms, 1.0, line_width=line_width))
        self.cover[m] = label
        return self

    def add_buildings(self, feats, label=BUILDING, default_height=10.0):
        """Buildings stand on the terrain: base = feature base if known, else
        the lowest terrain cell under the footprint (no floating buildings)."""
        h = np.where(np.isfinite(feats.height), feats.height, default_height)
        order = np.argsort(h)
        geoms = feats.geoms[order]
        top_rel = self._burn(geoms, h[order])
        idx = self._burn(geoms, np.arange(len(geoms), dtype=np.float64))
        base = np.full_like(top_rel, np.nan)
        known = feats.base[order]
        has = ~np.isnan(idx)
        k = idx[has].astype(np.int64)
        # lowest terrain under each footprint
        low = np.full(len(geoms), np.inf)
        np.minimum.at(low, k, self.terrain[has])
        b = np.where(np.isfinite(known[k]), known[k], low[k])
        base[has] = np.minimum(b, self.terrain[has])
        top = base + top_rel
        self.solids.append((base, top, label))
        return self

    def add_trees(self, feats, top=None, base=None, label=TREE):
        """Canopies: feats with height = canopy top above ground (or `top`),
        `base` = crown base above ground (default 0.3 * top). Air stays below."""
        h = feats.height if top is None else np.full(len(feats), float(top))
        h = np.where(np.isfinite(h), h, 10.0)
        t = self._burn(feats.geoms, h)
        b = 0.3 * t if base is None else np.where(np.isnan(t), np.nan, float(base))
        self.solids.append((self.terrain + b, self.terrain + t, label))
        return self

    def add_solid(self, base_abs, top_abs, label):
        """Any raster layer of absolute bottom/top elevations (nan = none)."""
        self.solids.append((np.asarray(base_abs, np.float64), np.asarray(top_abs, np.float64), label))
        return self

    def build(self, air_above=20.0, ground_thickness=None, nz=None):
        """Occupancy. z = 0 is one ground_thickness (default one cell) below
        the lowest terrain point; the box reaches air_above over the top."""
        c = self.cell
        gt = c if ground_thickness is None else float(ground_thickness)
        z0 = self.terrain.min() - gt
        tops = [np.nanmax(t) for _, t, _ in self.solids if np.isfinite(t).any()]
        zmax = max([self.terrain.max()] + tops)
        nz = int(np.ceil((zmax - z0 + air_above) / c)) if nz is None else int(nz)
        g = (self.terrain - z0) / c
        layers = [(0.0, g - 1.0, GROUND), (g - 1.0, g, self.cover)]
        for b, t, lab in self.solids:
            bb = np.where(np.isnan(b), np.inf, (b - z0) / c)
            tt = np.where(np.isnan(t), -np.inf, (t - z0) / c)
            layers.append((bb, tt, lab))
        city = Occupancy.from_layers((self.nx, self.ny, nz), layers, default=AIR,
                                      voxel_size=c, origin=(self.x0, self.y0, z0))
        city.classes = CLASSES
        return city
