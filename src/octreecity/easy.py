"""One-call cities from OpenStreetMap, for any place on Earth.

    from octreecity.easy import city_from_osm
    city = city_from_osm(36.370, 127.3625, size_m=600)   # KAIST, Daejeon

Downloads buildings, land cover, trees, water and roads from the Overpass
API, projects them to the local UTM zone and builds an Occupancy. Terrain
is flat (no elevation model is downloaded). Building heights come from the
OSM tags height or building:levels; others get `default_height`.

Every object keeps who it is: city.ids["id"] holds, per voxel, the id of
the building, building part, street, path, waterway or tree there, and
city.ids["zone_id"] the land-use zone (landuse, leisure, natural areas).
Both point into city.objects, whose rows keep the OSM type and id
("osm:way/123"), the tags (JSON text, see ObjectTable.tag) and the parent
(a building part's building, a zone inside another zone).

Building parts (Simple 3D Buildings): parts with min_height or
building:min_level start above the ground, so blocks stacked on blocks
keep their own voxels. A building with parts is painted only where its
parts do not cover it (not at all when they cover 90% or more). Bridges
(bridge=* or layer > 0) are lifted to layer x bridge_height above the
ground; tunnels and underground features are left out.
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

from .urban import BUILDING, GRASS, PAVEMENT, ROAD, WATER
from .features import Features, read_osm_overpass
from .objects import NONE, object_id

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
         f'way["building"]{b};relation["building"]{b};way["building:part"]{b};relation["building:part"]{b};'
         f'relation["type"="building"]{b};'
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
                  city_cache=None, overlap="keep", bridge_height=5.0, buildings_namespace=None):
    """Build an Occupancy of a size_m square around (lat, lon) from OSM,
    with object ids (city.ids, city.objects; see the module docstring).

    cache: file to keep the downloaded OSM data (.json). city_cache: file to
    keep the finished city (.npz, with its object table); if it exists it
    is loaded instead of building again (delete it to rebuild, e.g. after
    changing settings). osm_json: use this Overpass answer instead of
    downloading. buildings: Features (same UTM CRS) to use instead of OSM
    buildings, e.g. from read_citygml + reproject; their ids come from
    attrs "source_id" or "id" in buildings_namespace (a dataset name, e.g.
    "plateau-tokyo-2023"). overlap: what a voxel claimed by two objects of
    one kind gets ("keep" = an overlap row; "first", "last"). The city's
    origin is in UTM metres; city.crs holds the EPSG code.
    """
    if city_cache is not None and Path(city_cache).exists():
        from .occupancy import Occupancy

        return Occupancy.load(city_cache)
    epsg = utm_epsg(lat, lon)
    data = osm_json if osm_json is not None else fetch_osm(lat, lon, size_m, cache)
    osm = read_osm_overpass(data, crs_to=epsg, level_height=level_height)
    from pyproj import Transformer

    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    b = osm_builder(osm, cx - size_m / 2, cy - size_m / 2, size_m, size_m, cell, buildings=buildings,
                    default_height=default_height, tree_height=tree_height, overlap=overlap,
                    bridge_height=bridge_height, buildings_namespace=buildings_namespace,
                    crs=f"EPSG:{epsg}")
    city = b.build(air_above=air_above)
    if city_cache is not None:
        Path(city_cache).parent.mkdir(parents=True, exist_ok=True)
        city.save(city_cache)
    return city


def osm_builder(osm, x0, y0, width, depth, cell=1.0, buildings=None, default_height=10.0,
                tree_height=12.0, overlap="keep", bridge_height=5.0, buildings_namespace=None,
                crs=None, objects=None):
    """A CityBuilder (not built yet) for a rectangle, from read_osm_overpass
    Features in a projected CRS. city_from_osm uses it; call it yourself
    for a tile of a bigger area (give it the OSM data of the tile plus a
    margin of at least the widest road or crown, e.g. 15 m)."""
    b = CityBuilder(x0, y0, width, depth, cell, objects=objects, crs=crs)
    objs = b.objects
    for k, u in (("height_m", "m"), ("min_height_m", "m"), ("width_m", "m")):
        objs.describe(k, unit=u)
    tags = osm.attrs["tags"]
    sid = np.array([s if s is not None else f"anon/{i}" for i, s in enumerate(osm.attrs["source_id"])],
                   dtype=object)
    empty = np.array([g.is_empty for g in osm.geoms], bool)
    poly = np.array([g.geom_type in ("Polygon", "MultiPolygon") for g in osm.geoms], bool) & ~empty
    under = np.array([_underground(t) for t in tags], bool)

    # ------------------------------------------------------------- zones
    zone = poly & ~under & np.array([("landuse" in t or "leisure" in t or
                                      ("natural" in t and t.get("natural") not in ("tree", "tree_row")))
                                     and "building" not in t and "building:part" not in t for t in tags], bool)
    zi = np.nonzero(zone)[0]
    if zi.size:
        zid = _add_osm(objs, sid, tags, zi, "zone")
        _nest(objs, zid, osm.geoms[zi])
        b.add_zones(osm.geoms[zi], zid, layer="zone_id", on="solid", overlap=overlap)

    # ------------------------------------------------------- land cover
    land = ~under & ~empty
    for test, lab in ((lambda t: any(t.get(k) in v for k, v in GRASS_TAGS.items()), GRASS),
                      (lambda t: t.get("natural") == "water" or t.get("waterway") == "riverbank"
                       or t.get("landuse") in ("reservoir", "basin"), WATER)):
        m = land & poly & np.array([bool(test(t)) for t in tags], bool)
        if m.any():
            b.add_landcover(osm.geoms[m], lab)
    # ------------------------------------- waterways, paths, roads (one kind)
    ww = land & ~poly & np.array([t.get("waterway") in ("stream", "river", "canal", "ditch") for t in tags])
    hw = land & np.array([t.get("highway") in ROAD_WIDTH or t.get("highway") in PATH_WIDTH for t in tags])
    hw &= np.array([g.geom_type in ("LineString", "MultiLineString", "Polygon", "MultiPolygon")
                    for g in osm.geoms])
    if ww.any() or hw.any():
        wi, hi = np.nonzero(ww)[0], np.nonzero(hw)[0]
        is_path = np.array([tags[i]["highway"] in PATH_WIDTH for i in hi], bool)
        hi = np.r_[hi[is_path], hi[~is_path]]                 # later wins the label: roads on top
        li = np.r_[wi, hi].astype(np.int64)
        cls = ["waterway"] * wi.size + ["path"] * int(is_path.sum()) + ["road"] * int((~is_path).sum())
        lab = np.array([WATER] * wi.size + [PAVEMENT] * int(is_path.sum()) + [ROAD] * int((~is_path).sum()),
                       np.uint8)
        width = np.array([4.0] * wi.size + [PATH_WIDTH.get(tags[i]["highway"], ROAD_WIDTH.get(tags[i]["highway"]))
                                            for i in hi], float)
        lid = _add_osm(objs, sid, tags, li, cls, width_m=width)
        lift = np.array([_bridge_layer(tags[i]) for i in li])
        flat = lift == 0
        if flat.any():
            b.add_landcover(osm.geoms[li[flat]], lab[flat], line_width=width[flat], ids=lid[flat],
                            overlap=overlap)
        if (~flat).any():
            z = lift[~flat] * bridge_height
            b.add_shapes(osm.geoms[li[~flat]], lab[~flat], z, z + b.cell, ids=lid[~flat],
                         line_width=width[~flat], overlap=overlap)

    # ------------------------------------------------------------- trees
    woods = land & poly & np.array([t.get("natural") == "wood" or t.get("landuse") == "forest" for t in tags])
    if woods.any():
        b.add_trees(osm.geoms[woods], top=tree_height, base=0.3 * tree_height)
    single = land & np.array([t.get("natural") == "tree" for t in tags]) & np.array(
        [g.geom_type == "Point" for g in osm.geoms])
    if single.any():
        si = np.nonzero(single)[0]
        b.add_trees(osm.geoms[si], top=0.75 * tree_height, base=2.5, line_width=6.0,
                    ids=_add_osm(objs, sid, tags, si, "tree"), overlap="first")

    # --------------------------------------------------------- buildings
    if buildings is None:
        _osm_buildings(b, osm, sid, poly & ~under, default_height, overlap)
    elif len(buildings):
        ids = None
        key = "source_id" if "source_id" in buildings.attrs else "id" if "id" in buildings.attrs else None
        if key is not None:
            ns = buildings_namespace or "buildings"
            ids = objs.add(ns, [str(v) for v in buildings.attrs[key]], "building",
                           height_m=buildings.height)
        b.add_buildings(buildings, default_height=default_height, ids=ids, overlap=overlap)
    return b


def _add_osm(objs, sid, tags, idx, cls, **cols):
    tg = [json.dumps(tags[i], ensure_ascii=False, sort_keys=True) for i in idx]
    return objs.add("osm", [sid[i] for i in idx], cls, tags=tg, **cols)


def _bridge_layer(t):
    """0 for a road on the ground, else how many layers up it is."""
    try:
        layer = float(t.get("layer", 0))
    except ValueError:
        layer = 0.0
    if t.get("bridge") not in (None, "no"):
        return max(layer, 1.0)
    return max(layer, 0.0)


def _nest(objs, ids, geoms):
    """A zone inside another zone (95% of its area or more) gets the
    smallest such zone as its parent."""
    geoms = list(geoms)
    if len(geoms) < 2:
        return
    area = shapely.area(geoms)
    tree = shapely.STRtree(geoms)
    for k, g in enumerate(geoms):
        best, best_area = NONE, np.inf
        for j in tree.query(g):
            if j == k or area[j] <= area[k]:
                continue
            if shapely.area(shapely.intersection(g, geoms[j])) >= 0.95 * area[k] and area[j] < best_area:
                best, best_area = int(ids[j]), area[j]
        if best != NONE:
            objs.set_parent([ids[k]], [best])


def _osm_buildings(b, osm, sid, ok, default_height, overlap):
    """Buildings and building parts (Simple 3D Buildings) with parents."""
    objs = b.objects
    tags = osm.attrs["tags"]
    is_part = ok & np.array([t.get("building:part", "no") != "no" for t in tags], bool)
    is_bld = ok & ~is_part & np.array([t.get("building", "no") != "no" for t in tags], bool)
    bi, pi = np.nonzero(is_bld)[0], np.nonzero(is_part)[0]
    if not bi.size and not pi.size:
        return
    h, mh = osm.height, osm.min_height
    bid = _add_osm(objs, sid, tags, bi, "building", height_m=h[bi], min_height_m=mh[bi])
    pid = _add_osm(objs, sid, tags, pi, "building_part", height_m=h[pi], min_height_m=mh[pi])
    # parents: type=building relations first (outline role, else the relation)
    rel_parent = {}
    for t, mem, s in zip(tags, osm.attrs["members"], sid):
        if mem and t.get("type") == "building":
            outline = [m for m, role in mem if role == "outline"]
            for m, role in mem:
                if role == "part":
                    rel_parent[m] = outline[0] if outline else s
    where = {s: i for i, s in enumerate(sid)}
    bgeo = list(osm.geoms[bi])
    btree = shapely.STRtree(bgeo) if bgeo else None
    parent = np.full(pi.size, NONE, np.int64)
    for k, i in enumerate(pi):
        if sid[i] in rel_parent:
            p = rel_parent[sid[i]]
            if p in where:              # the outline (or the relation) is in the data: give it a row
                cls = "building" if where[p] not in set(pi.tolist()) else "building_part"
                parent[k] = _add_osm(objs, sid, tags, [where[p]], cls)[0]
            else:
                parent[k] = object_id(f"osm:{p}")
        elif btree is not None:
            g = osm.geoms[i]
            share = 0.5
            for j in btree.query(g):
                a = shapely.area(shapely.intersection(g, bgeo[j])) / max(g.area, 1e-9)
                if a >= share:
                    parent[k], share = bid[j], a
    if pi.size:
        has = parent != NONE
        objs.set_parent(pid[has], parent[has])
    # heights: a part with no height takes its building's; levels = 0 means no volume
    btop = dict(zip(bid.tolist(), h[bi].tolist()))
    ph = np.array([h[i] if np.isfinite(h[i]) else btop.get(int(p), np.nan)
                   for i, p in zip(pi, parent)], np.float64)
    ph = np.where(np.isfinite(ph), ph, default_height)
    pmh = np.where(np.isfinite(mh[pi]), mh[pi], 0.0)
    pvol = ph > pmh
    # buildings with parts: painted only where the parts leave them uncovered
    geoms, top, bot, ids = [], [], [], []
    for k, i in enumerate(bi):
        g = osm.geoms[i]
        kids = pi[parent == bid[k]]
        if kids.size:
            cover = shapely.union_all(list(osm.geoms[kids]))
            if shapely.area(shapely.intersection(g, cover)) >= 0.9 * g.area:
                continue
            g = g.difference(cover)
        geoms.append(g)
        top.append(h[i] if np.isfinite(h[i]) else default_height)
        bot.append(mh[i] if np.isfinite(mh[i]) else 0.0)
        ids.append(bid[k])
    geoms += list(osm.geoms[pi[pvol]])
    top += ph[pvol].tolist()
    bot += pmh[pvol].tolist()
    ids += pid[pvol].tolist()
    if geoms:
        f = Features(geoms, height=top, min_height=bot)
        b.add_buildings(f, BUILDING, default_height=default_height, ids=np.array(ids, np.int64),
                        overlap=overlap)


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
