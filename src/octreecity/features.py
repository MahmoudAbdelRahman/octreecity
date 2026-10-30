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
from shapely import wkb as _wkb  # noqa: F401
from shapely.geometry import shape as _shape

GML = "http://www.opengis.net/gml"


@dataclass
class Features:
    geoms: np.ndarray                 # shapely geometries (object array)
    height: np.ndarray = None         # top above base (m), nan = unknown
    base: np.ndarray = None           # absolute base elevation (m), nan = unknown
    label: np.ndarray = None          # uint8 per feature, or None
    attrs: dict = field(default_factory=dict)
    min_height: np.ndarray = None     # bottom above base (m), nan/0 = on the ground

    def __post_init__(self):
        geoms = np.empty(len(self.geoms), dtype=object)
        geoms[:] = list(self.geoms)
        self.geoms = geoms
        n = len(self.geoms)
        nan = lambda: np.full(n, np.nan)
        self.height = nan() if self.height is None else np.asarray(self.height, np.float64)
        self.base = nan() if self.base is None else np.asarray(self.base, np.float64)
        self.min_height = nan() if self.min_height is None else np.asarray(self.min_height, np.float64)

    def __len__(self):
        return len(self.geoms)

    def bounds(self):
        return shapely.total_bounds(self.geoms)

    def subset(self, mask):
        mask = np.asarray(mask)
        return Features(self.geoms[mask], self.height[mask], self.base[mask],
                        None if self.label is None else np.asarray(self.label)[mask],
                        {k: _take(v, mask) for k, v in self.attrs.items()}, self.min_height[mask])

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


def _num(v):
    try:
        return float(str(v).split()[0].replace("m", "").replace(",", "."))
    except (TypeError, ValueError, IndexError):
        return np.nan


def _min_height_from_tags(p, level_height=3.0):
    """Bottom of a building part above the ground (m): min_height, else
    building:min_level x level_height, else nan (on the ground)."""
    v = _num(p.get("min_height", p.get("building:min_height")))
    if np.isfinite(v):
        return v
    lv = _num(p.get("building:min_level"))
    return lv * level_height if np.isfinite(lv) else np.nan


def _is_line(t):
    """Closed ways that are still lines (a roundabout, a fence, a river)."""
    return (("highway" in t or "barrier" in t or "railway" in t
             or t.get("natural") == "tree_row" or t.get("waterway") not in (None, "riverbank"))
            and t.get("area") != "yes")


def _ring(member):
    xy = [(p["lon"], p["lat"]) for p in member.get("geometry") or []]
    return xy if len(xy) >= 2 else None


def _rings_to_polygons(rings):
    """Closed polygons from ring pieces (ways that join end to end)."""
    lines = [shapely.LineString(r) for r in rings]
    if not lines:
        return shapely.Polygon()
    polys = shapely.get_parts(shapely.polygonize(lines))
    if not len(polys):
        return shapely.Polygon()
    return shapely.union_all([shapely.make_valid(p) for p in polys])


def read_osm_overpass(json_data, crs_to, level_height=3.0):
    """Ways, relations and nodes from an Overpass API JSON answer ('out
    geom;') as Features, keeping who each one is.

    attrs:
        source_id   "way/123", "relation/7", "node/9" (the OSM type and id;
                    None if the answer has no ids)
        tags        the OSM tags (dict per feature)
        members     for relations: [(source id, role), ...] (else None)
    Closed ways become polygons, except highways, barriers, waterways and
    tree rows (lines unless area=yes); open ways are lines. Multipolygon
    relations are their outer rings minus their inner rings (holes kept).
    type=building relations (which only group an outline and its parts)
    get an empty geometry. height: top above the ground from height /
    building:height, else building:levels x level_height. min_height: the
    bottom of building parts that start above the ground (min_height, else
    building:min_level x level_height).
    """
    geoms, tags, sids, members = [], [], [], []
    for el in json_data.get("elements", []):
        t = el.get("tags", {}) or {}
        typ = el.get("type")
        mem = None
        if typ == "way" and "geometry" in el:
            xy = [(p["lon"], p["lat"]) for p in el["geometry"] if p]
            if len(xy) >= 4 and xy[0] == xy[-1] and not _is_line(t):
                g = shapely.Polygon(xy)
            elif len(xy) >= 2:
                g = shapely.LineString(xy)
            else:
                continue
        elif typ == "relation" and "members" in el:
            mem = [(f"{m.get('type')}/{m.get('ref')}", m.get("role", "")) for m in el["members"]]
            if t.get("type") == "building":
                g = shapely.Polygon()
            else:
                outer = [r for m in el["members"] if m.get("role") in ("outer", "") and (r := _ring(m))]
                inner = [r for m in el["members"] if m.get("role") == "inner" and (r := _ring(m))]
                g = _rings_to_polygons(outer)
                if g.is_empty:
                    continue
                if inner:
                    g = g.difference(_rings_to_polygons(inner))
        elif typ == "node" and "lat" in el:
            g = shapely.Point(el["lon"], el["lat"])
        else:
            continue
        geoms.append(shapely.make_valid(g) if not g.is_empty else g)
        tags.append(t)
        sids.append(f"{typ}/{el['id']}" if "id" in el else None)
        members.append(mem)
    geoms = reproject(_objects(geoms), 4326, crs_to)
    h = np.array([_height_from_tags(t, level_height=level_height) for t in tags])
    mh = np.array([_min_height_from_tags(t, level_height) for t in tags])
    return Features(geoms, h, attrs={"tags": _objects(tags), "source_id": _objects(sids),
                                     "members": _objects(members)}, min_height=mh)


def _objects(values):
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _take(v, mask):
    return (v if isinstance(v, np.ndarray) else _objects(list(v)))[mask]


@dataclass
class Surfaces:
    """3D polygons of city objects, with the kind of surface each one is.

    polygons    list of (k, 3) arrays (exterior rings, metres, file CRS)
    kind        name of the surface element each polygon sits in, e.g.
                "WallSurface", "RoofSurface", "GroundSurface" (LoD2+), or the
                geometry property for objects without them, e.g. "lod1Solid"
    obj         index of the object (building) each polygon belongs to
    objects     per object: {"id": gml id, attribute name: value, ...}
    """
    polygons: list
    kind: np.ndarray
    obj: np.ndarray
    objects: list

    def __len__(self):
        return len(self.polygons)


def read_citygml_surfaces(path, tag="Building"):
    """3D surfaces of every `tag` object in a CityGML file (any LoD), with
    their semantic kind (wall, roof, ground ... as named in the file) and
    the object's attributes (gen:*Attribute values, measuredHeight).
    Turn them into voxels with octreecity.sources.SparseSource.from_surfaces."""
    from lxml import etree

    opener = gzip.open if str(path).endswith(".gz") else open
    polys, kinds, objs, objects = [], [], [], []
    with opener(path, "rb") as f:
        for _, el in etree.iterparse(f, events=("end",), tag=f"{{*}}{tag}", huge_tree=True):
            attrs = {"id": el.get(f"{{{GML}}}id", "")}
            for a in el.iter("{*}stringAttribute", "{*}intAttribute", "{*}doubleAttribute"):
                v = a.find("{*}value")
                if v is not None and a.get("name"):
                    t = etree.QName(a).localname
                    attrs[a.get("name")] = (float(v.text) if t != "stringAttribute" else v.text)
            mh = el.find("{*}measuredHeight")
            if mh is not None:
                try:
                    attrs["measuredHeight"] = float(mh.text)
                except (TypeError, ValueError):
                    pass
            n = 0
            for pl in el.iter(f"{{{GML}}}posList"):
                dim = int(pl.get("srsDimension", 3))
                v = np.array(pl.text.split(), float)
                if dim != 3 or v.size < 9:
                    continue
                kind, p = None, pl.getparent()
                while p is not None and p is not el:
                    name = etree.QName(p).localname
                    if (name.endswith("Surface") and not name.startswith("lod")
                            and name not in ("MultiSurface", "CompositeSurface", "OrientableSurface")):
                        kind = name
                        break
                    if kind is None and name.startswith("lod"):
                        kind = name
                    p = p.getparent()
                polys.append(v.reshape(-1, 3))
                kinds.append(kind or "surface")
                objs.append(len(objects))
                n += 1
            if n:
                objects.append(attrs)
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    return Surfaces(polys, np.array(kinds), np.array(objs, np.int64), objects)
