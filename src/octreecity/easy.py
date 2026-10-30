"""One-call cities from OpenStreetMap, for any place on Earth.

    from octreecity.easy import city_from_osm
    city = city_from_osm(36.370, 127.3625, size_m=600)   # KAIST, Daejeon

Downloads buildings, land cover, trees, water and roads from the Overpass
API, projects them to the local UTM zone and builds an Occupancy. Terrain
is flat (no elevation model is downloaded). Building heights come from the
OSM tags height or building:levels; others get `default_height`.
"""

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import shapely

from . import urban
from .builder import CityBuilder

from .urban import GRASS, PAVEMENT, ROAD, WATER
from .features import Features, read_osm_overpass

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
ROAD_WIDTH = {"motorway": 20, "trunk": 16, "primary": 14, "secondary": 11, "tertiary": 9,
              "unclassified": 6, "residential": 6, "service": 4, "living_street": 5}
PATH_WIDTH = {"footway": 2, "path": 2, "cycleway": 2.5, "pedestrian": 4, "steps": 2}
GRASS_TAGS = {"landuse": ("grass", "meadow", "recreation_ground", "village_green"),
              "leisure": ("park", "pitch", "garden", "golf_course"),
              "natural": ("grassland", "scrub", "heath")}


def utm_epsg(lat, lon):
    """EPSG code of the UTM zone containing (lat, lon)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def bbox_latlon(lat, lon, size_m):
    """(south, west, north, east) of a square of size_m centred on lat, lon."""
    dlat = size_m / 2 / 111_320
    dlon = size_m / 2 / (111_320 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def fetch_osm(lat, lon, size_m, cache=None, timeout=90):
    """Overpass JSON for buildings, land cover, trees, water and roads.
    With cache (a file path), the answer is saved and reused."""
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
    s, w, n, e = bbox_latlon(lat, lon, size_m)
    b = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    q = (f"[out:json][timeout:{timeout}];("
         f'way["building"]{b};relation["building"]{b};'
         f'way["landuse"]{b};way["natural"]{b};way["leisure"]{b};way["highway"]{b};'
         f'way["waterway"]{b};relation["natural"]{b};relation["landuse"]{b};node["natural"="tree"]{b};'
         ");out geom;")
    data = urllib.parse.urlencode({"data": q}).encode()
    errors = []
    for attempt in range(2):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "octreecity"})
                with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                    out = json.loads(r.read())
                break
            except Exception as exc:          # busy server: try the next mirror
                errors.append(f"{url}: {exc}")
        else:
            time.sleep(5 * (attempt + 1))
            continue
        break
    else:
        raise RuntimeError("all Overpass servers failed:\n" + "\n".join(errors))
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def _subset(osm, test):
    return osm.subset(np.array([bool(test(t)) for t in osm.attrs["tags"]], bool))


def city_from_osm(lat, lon, size_m=500.0, cell=1.0, cache=None, osm_json=None, buildings=None,
                  default_height=10.0, level_height=3.0, tree_height=12.0, air_above=20.0,
                  city_cache=None):
    """Build an Occupancy of a size_m square around (lat, lon) from OSM.

    cache: file to keep the downloaded OSM data (.json). city_cache: file to
    keep the finished city (.npz); if it exists it is loaded instead of
    building again (delete it to rebuild, e.g. after changing settings).
    osm_json: use this Overpass answer instead of downloading. buildings:
    Features (same UTM CRS) to use instead of OSM buildings, e.g. from
    read_citygml + reproject. The city's origin is in UTM metres; city.crs
    holds the EPSG code.
    """
    if city_cache is not None and Path(city_cache).exists():
        from .occupancy import Occupancy

        return Occupancy.load(city_cache)
    epsg = utm_epsg(lat, lon)
    data = osm_json if osm_json is not None else fetch_osm(lat, lon, size_m, cache)
    osm = read_osm_overpass(data, crs_to=epsg)
    from pyproj import Transformer

    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0, y0 = cx - size_m / 2, cy - size_m / 2
    b = CityBuilder(x0, y0, size_m, size_m, cell)

    tags = osm.attrs["tags"]
    under = np.array([_underground(t) for t in tags], bool)
    is_bld = np.array(["building" in t for t in tags], bool) & ~under
    land = osm.subset(~is_bld & ~under)
    b.add_landcover(_subset(land, lambda t: any(t.get(k) in v for k, v in GRASS_TAGS.items())), GRASS)
    b.add_landcover(_subset(land, lambda t: t.get("natural") == "water" or t.get("waterway") == "riverbank"
                            or t.get("landuse") in ("reservoir", "basin")), WATER)
    b.add_landcover(_subset(land, lambda t: t.get("waterway") in ("stream", "river", "canal", "ditch")),
                    WATER, line_width=4)
    for widths, lab in ((ROAD_WIDTH, ROAD), (PATH_WIDTH, PAVEMENT)):
        r = _subset(land, lambda t: t.get("highway") in widths)
        if len(r):
            b.add_landcover(r, lab, line_width=np.array([widths[t["highway"]] for t in r.attrs["tags"]], float))
    woods = _subset(land, lambda t: t.get("natural") == "wood" or t.get("landuse") == "forest")
    if len(woods):
        b.add_trees(woods, top=tree_height, base=0.3 * tree_height)
    single = _subset(land, lambda t: t.get("natural") == "tree")
    if len(single):
        b.add_trees(Features(shapely.buffer(single.geoms, 3.0), height=np.full(len(single), 0.75 * tree_height)),
                    base=2.5)

    if buildings is None:
        bl = osm.subset(is_bld)
        polys = np.array([g.geom_type in ("Polygon", "MultiPolygon") for g in bl.geoms], bool)
        bl = bl.subset(polys)
        h = bl.height.copy()
        lv = np.array([_levels(t, level_height) for t in bl.attrs["tags"]])
        h = np.where(np.isfinite(h), h, np.where(np.isfinite(lv), lv, default_height))
        buildings = Features(bl.geoms, height=h)
    if len(buildings):
        b.add_buildings(buildings)
    city = b.build(air_above=air_above)
    city.crs = f"EPSG:{epsg}"
    if city_cache is not None:
        Path(city_cache).parent.mkdir(parents=True, exist_ok=True)
        city.save(city_cache)
    return city


def _underground(t):
    """Mapped features that are below ground: tunnels, culverts, basements."""
    try:
        if float(t.get("layer", 0)) < 0:
            return True
    except ValueError:
        pass
    return (t.get("location") in ("underground", "tunnel") or t.get("tunnel") not in (None, "no")
            or t.get("covered") == "yes" and "waterway" in t)


def _levels(tags, level_height):
    try:
        return float(tags["building:levels"]) * level_height
    except (KeyError, ValueError):
        return np.nan
