"""Readers for 2D city data (footprints, lines, points), all returning Features in a projected CRS (metres).

Features is the common currency: 2D shapes with an optional height, base
elevation and label. Buildings, tree canopies, land cover, roads and water
all travel as Features; the builder decides how each becomes voxels.
"""

import gzip
import json
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import wkb as _wkb
from shapely.geometry import shape as _shape

GML = "http://www.opengis.net/gml"


@dataclass
class Features:
    geoms: np.ndarray                 # shapely geometries (object array)
    height: np.ndarray = None         # height above base (m), nan = unknown
    base: np.ndarray = None           # absolute base elevation (m), nan = unknown
    label: np.ndarray = None          # uint8 per feature, or None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.geoms = np.asarray(self.geoms, dtype=object)
        n = len(self.geoms)
        nan = lambda: np.full(n, np.nan)
        self.height = nan() if self.height is None else np.asarray(self.height, np.float64)
        self.base = nan() if self.base is None else np.asarray(self.base, np.float64)

    def __len__(self):
        return len(self.geoms)

    def bounds(self):
        return shapely.total_bounds(self.geoms)

    def subset(self, mask):
        mask = np.asarray(mask)
        return Features(self.geoms[mask], self.height[mask], self.base[mask],
                        None if self.label is None else np.asarray(self.label)[mask],
                        {k: np.asarray(v)[mask] for k, v in self.attrs.items()})

    def clip(self, x0, y0, x1, y1):
        """Features whose bounding box touches the rectangle (geometry unchanged)."""
        b = shapely.bounds(self.geoms)
        return self.subset((b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1))

    @classmethod
    def from_wkb(cls, pairs):
        """[(wkb_bytes, height), ...], e.g. the urban_wind building pickles."""
        geoms = shapely.from_wkb([p[0] for p in pairs])
        return cls(geoms, height=[p[1] for p in pairs])


def read_citygml(path, min_area=1.0):
    """Buildings from CityGML 2.0/3.0 (LoD0-LoD2), streamed with iterparse.

    Footprint: union of the building's polygons projected to 2D (roof,
    ground, solid or lod0 surfaces). Base: the 'base_height_m' attribute if
    present, else the lowest z. Height: bldg:measuredHeight if present and
    plausible, else highest z - base. Coordinates stay in the file's CRS.
    """
    from lxml import etree

    opener = gzip.open if str(path).endswith(".gz") else open
    geoms, heights, bases, ids = [], [], [], []
    with opener(path, "rb") as f:
        for _, el in etree.iterparse(f, events=("end",), tag="{*}Building", huge_tree=True):
            polys, zs = [], []
            for pl in el.iter(f"{{{GML}}}posList"):
                v = np.array(pl.text.split(), float)
                dim = int(pl.get("srsDimension", 3))
                if v.size < 3 * dim:
                    continue
                v = v.reshape(-1, dim)
                zs.append(v[:, 2] if dim == 3 else np.zeros(len(v)))
                p = shapely.Polygon(v[:, :2])
                if p.is_valid and p.area > 1e-6:
                    polys.append(p)
            if not polys:
                el.clear()
                continue
            fp = shapely.union_all(polys)
            if fp.area < min_area:
                el.clear()
                continue
            z = np.concatenate(zs)
            base, top = float(z.min()), float(z.max())
            for a in el.iter("{*}doubleAttribute"):
                if a.get("name") == "base_height_m":
                    base = float(a[0].text)
            h = top - base
            mh = el.find("{*}measuredHeight")
            if mh is not None:
                try:
                    v = float(mh.text)
                    if 0.5 < v < 1000:
                        h = v
                except (TypeError, ValueError):
                    pass
            geoms.append(fp)
            heights.append(h)
            bases.append(base)
            ids.append(el.get(f"{{{GML}}}id", ""))
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    return Features(geoms, heights, bases, attrs={"id": np.array(ids)})


def read_geojson(path, height_key=None, levels_key=None, level_height=3.0, crs_to=None):
    """Features from GeoJSON (e.g. an OSM export). Optionally reproject from
    the file's lon/lat to a projected EPSG code with crs_to."""
    data = json.load(open(path))
    feats = data["features"] if "features" in data else data
    geoms = [_shape(f["geometry"]) for f in feats]
    props = [f.get("properties", {}) or {} for f in feats]
    h = np.full(len(geoms), np.nan)
    for k, p in enumerate(props):
        h[k] = _height_from_tags(p, height_key, levels_key, level_height)
    geoms = np.array(geoms, dtype=object)
    if crs_to is not None:
        geoms = reproject(geoms, 4326, crs_to)
    return Features(geoms, h, attrs={"props": np.array(props, dtype=object)})


def reproject(geoms, crs_from, crs_to):
    from pyproj import Transformer

    tr = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return shapely.transform(np.asarray(geoms, dtype=object),
                             lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def _height_from_tags(p, height_key=None, levels_key=None, level_height=3.0):
    for k in [height_key, "height", "building:height"]:
        if k and k in p:
            try:
                return float(str(p[k]).split()[0].replace("m", ""))
            except ValueError:
                pass
    for k in [levels_key, "building:levels"]:
        if k and k in p:
            try:
                return float(p[k]) * level_height
            except ValueError:
                pass
    return np.nan


def read_osm_overpass(json_data, crs_to):
    """Ways/relations from an Overpass API JSON answer ('out geom;') as
    Features with their OSM tags. Closed ways become polygons, open ways
    lines (roads, rivers); multipolygon relations use their outer rings."""
    geoms, tags = [], []
    for el in json_data.get("elements", []):
        t = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            xy = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(xy) >= 4 and xy[0] == xy[-1]:
                g = shapely.Polygon(xy)
            elif len(xy) >= 2:
                g = shapely.LineString(xy)
            else:
                continue
        elif el["type"] == "relation" and "members" in el:
            rings = [[(p["lon"], p["lat"]) for p in m["geometry"]]
                     for m in el["members"] if m.get("role") == "outer" and "geometry" in m]
            polys = [shapely.Polygon(r) for r in rings if len(r) >= 4 and r[0] == r[-1]]
            if not polys:
                continue
            g = shapely.MultiPolygon(polys)
        elif el["type"] == "node" and "lat" in el:
            g = shapely.Point(el["lon"], el["lat"])
        else:
            continue
        geoms.append(shapely.make_valid(g))
        tags.append(t)
    geoms = reproject(np.array(geoms, dtype=object), 4326, crs_to)
    h = np.array([_height_from_tags(t) for t in tags])
    return Features(geoms, h, attrs={"tags": np.array(tags, dtype=object)})
