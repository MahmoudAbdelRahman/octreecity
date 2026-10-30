"""CityBuilder and the OSM reader: one centre rule for geometry, label and
id; overlaps; stacked parts; holes; bridges."""

import warnings

import numpy as np
import pytest
import shapely

from octreecity.builder import CityBuilder, footprint_pairs, rasterize
from octreecity.features import Features, read_osm_overpass
from octreecity.urban import AIR, BUILDING, GROUND, ROAD


def test_touching_shapes_claim_each_column_once():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 64, (60, 2)) + 360_000.0
    from shapely.ops import voronoi_diagram

    cells = shapely.get_parts(voronoi_diagram(shapely.MultiPoint(pts)))
    box = shapely.box(360_000, 360_000, 360_064, 360_064)
    cells = [c.intersection(box) for c in cells]
    f, col = footprint_pairs(cells, 360_000, 360_000, 64, 64, 1.0)
    assert np.bincount(col, minlength=64 * 64).max() == 1 and np.unique(col).size == 64 * 64
    # rectangles whose edges go exactly through cell centres
    rects = [shapely.box(0.5, 0.5, 10.5, 20.5), shapely.box(10.5, 0.5, 30.5, 20.5)]
    f, col = footprint_pairs(rects, 0, 0, 40, 40, 1.0)
    assert np.bincount(col).max() == 1 and col.size == 30 * 20


def test_rasterize_uses_the_centre_rule():
    a, b = shapely.box(10, 10, 20, 20), shapely.box(20, 10, 30, 20)
    r = rasterize([a, b], [1.0, 2.0], 0, 0, 40, 40, 1.0)
    assert (r == 1).sum() == 100 and (r == 2).sum() == 100      # no cell taken twice, no extra ring


def build_two_buildings_and_parts():
    b = CityBuilder(0, 0, 64, 64)
    o = b.objects
    bid = o.add("test", ["a", "b"], "building")
    b.add_buildings(Features([shapely.box(10, 10, 20, 20), shapely.box(20, 10, 30, 20)], height=[10, 12]),
                    ids=bid)
    pid = o.add("test", ["a1", "a2"], "building_part", parent=bid[0])
    # a part over its own building, and a block floating 12-20 m above it
    b.add_buildings(Features([shapely.box(10, 10, 14, 14), shapely.box(10, 10, 15, 15)],
                             height=[10, 20], min_height=[0, 12]), ids=pid)
    return b, bid, pid


def test_stacked_parts_and_shared_walls():
    b, bid, pid = build_two_buildings_and_parts()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # a part over its building is not a clash
        c = b.build(air_above=5)
    g, lab = c.to_dense(layer="id"), c.to_dense()
    def k(z):                                                   # voxel index of height z
        return int(np.ceil(z - c.origin[2] - 0.5))

    col = g[12, 12]
    assert (col[k(0.5):k(9.5) + 1] == pid[0]).all()            # the part replaced its building
    assert (col[k(10.5):k(11.5) + 1] == -1).all() and (lab[12, 12, k(10.5):k(11.5) + 1] == AIR).all()
    assert (col[k(12.5):k(19.5) + 1] == pid[1]).all()
    assert (g[16, 16, k(5)] == bid[0]) and (g[25, 16, k(5)] == bid[1])
    assert ((g == bid[0]).any(-1).sum() + (g == bid[1]).any(-1).sum()
            + (g == pid[0]).any(-1).sum()) == 200               # 2 x 100 columns, nothing claimed twice
    assert len(b.objects.select(cls="overlap")) == 0


def test_overlap_rows_and_winner_rules():
    lines = [shapely.LineString([(0, 30), (64, 30)]), shapely.LineString([(30, 0), (30, 64)]),
             shapely.LineString([(0, 0), (64, 64)])]
    res = {}
    for mode in ("keep", "first", "last", "by"):
        b = CityBuilder(0, 0, 64, 64)
        r = b.objects.add("test", ["r1", "r2", "r3"], "road", width_m=[6.0, 4.0, 2.0])
        b.add_landcover(lines, ROAD, line_width=[6, 4, 2], ids=r, overlap=mode,
                        priority="width_m" if mode == "by" else None)
        c = b.build(air_above=2)
        top = c.to_dense(layer="id")[:, :, 0]
        res[mode] = (top, r, b.objects)
    top, r, o = res["keep"]
    rows = o.select(cls="overlap")
    assert len(rows) >= 2 and all(set(o.members(i)) <= set(r.tolist()) for i in rows)
    assert set(np.unique(top[top >= 0])) <= set(r.tolist()) | set(rows.tolist())
    assert all(int((top == i).sum()) > 0 for i in rows)         # no orphan rows
    centre = top[30, 30]
    assert set(o.members(centre)) == set(r.tolist())            # 3 roads at the centre
    for mode, want in (("first", 0), ("last", 2), ("by", 0)):
        t, rr, oo = res[mode]
        assert len(oo.select(cls="overlap")) == 0 and t[30, 30] == rr[want]


def test_zones_layer_and_buffer_warning():
    b = CityBuilder(0, 0, 64, 64)
    z = b.objects.add("test", ["z1", "z2"], "zone")
    bid = b.objects.add("test", ["b"], "building")[0]
    b.add_buildings(Features([shapely.box(5, 5, 15, 15)], height=[8]), ids=[bid])
    b.add_zones([shapely.box(0, 0, 32, 64), shapely.box(32, 0, 64, 64)], z)
    c = b.build(air_above=2)
    assert c.layers == ("id", "zone_id")
    zz, lab = c.to_dense(layer="zone_id"), c.to_dense()
    assert (zz[lab == AIR] == -1).all() and (zz[10, 10][lab[10, 10] != AIR] == z[0]).all()
    assert (c.to_dense(layer="id")[10, 10][lab[10, 10] == BUILDING] == bid).all()
    # big buffers in one id layer overlap everywhere: warn
    b = CityBuilder(0, 0, 64, 64)
    pts = [shapely.Point(x, y) for x in range(8, 64, 8) for y in range(8, 64, 8)]
    ids = b.objects.add("test", [str(k) for k in range(len(pts))], "buffer")
    b.add_landcover(pts, GROUND, line_width=60.0, ids=ids)
    with pytest.warns(UserWarning, match="overlap a lot"):
        b.build(air_above=2)


def test_later_call_overwriting_warns():
    b = CityBuilder(0, 0, 32, 32)
    t = b.objects.add("test", ["tree"], "tree")
    h = b.objects.add("test", ["house"], "building")
    b.add_trees([shapely.Point(10, 10)], top=8, base=0, line_width=8, ids=t)
    b.add_buildings(Features([shapely.box(10, 0, 20, 20)], height=[10]), ids=h)
    with pytest.warns(UserWarning, match="painted over"):
        b.build(air_above=2)
    assert b.stats["id"]["overwritten_voxels"] > 0


def test_terrain_and_base():
    b = CityBuilder(0, 0, 32, 32)
    b.add_terrain(lambda x, y: 0.2 * x)
    i = b.objects.add("test", ["a"], "building")
    b.add_buildings(Features([shapely.box(10, 10, 20, 20)], height=[6], min_height=[2]), ids=i)
    c = b.build(air_above=2)
    g, lab = c.to_dense(layer="id"), c.to_dense()
    col = lab[15, 15]
    zb = np.argmax(g[15, 15] == i[0])
    assert col[zb - 1] == AIR                                    # lifted 2 m above its base
    assert (lab[11, 15] == BUILDING).sum() == 4                  # 2 .. 6 m: 4 voxels


# --------------------------------------------------------------------- OSM

def sq(lat, lon, s):
    return [{"lat": lat + a, "lon": lon + b} for a, b in ((0, 0), (0, s), (s, s), (s, 0), (0, 0))]


def test_osm_reader_keeps_ids_holes_and_part_heights():
    lat, lon, d = 1.28, 103.80, 0.001
    data = {"elements": [
        {"type": "relation", "id": 7, "tags": {"type": "multipolygon", "building": "yes", "building:levels": "2"},
         "members": [{"type": "way", "ref": 1, "role": "outer", "geometry": sq(lat, lon, d)},
                     {"type": "way", "ref": 2, "role": "inner", "geometry": sq(lat + d / 4, lon + d / 4, d / 2)}]},
        {"type": "way", "id": 3, "tags": {"building:part": "yes", "building:levels": "12", "building:min_level": "6"},
         "geometry": sq(lat, lon, d / 5)},
        {"type": "way", "id": 4, "tags": {"building:part": "yes", "height": "30 m", "min_height": "18"},
         "geometry": sq(lat + d / 2, lon, d / 5)},
        {"type": "way", "id": 5, "tags": {"highway": "residential", "junction": "roundabout"},
         "geometry": sq(lat, lon + 2 * d, d / 5)},
    ]}
    f = read_osm_overpass(data, 32648)
    assert f.attrs["source_id"].tolist() == ["relation/7", "way/3", "way/4", "way/5"]
    a = f.geoms[0].area
    assert 0.7 < a / (shapely.area(shapely.Polygon(shapely.get_coordinates(f.geoms[0].exterior)))) < 0.8
    assert f.height[1] == 36 and f.min_height[1] == 18 and f.height[2] == 30 and f.min_height[2] == 18
    assert f.geoms[3].geom_type == "LineString"                  # a closed road is still a road


def interlace_like():
    lat, lon, d = 1.28, 103.80, 0.0009
    return lat, lon, {"elements": [
        {"type": "way", "id": 1, "tags": {"building": "residential", "building:levels": "2"},
         "geometry": sq(lat - d / 2, lon - d / 2, d)},
        {"type": "way", "id": 2, "tags": {"building:part": "yes", "building:levels": "6"},
         "geometry": sq(lat - d / 2, lon - d / 2, d / 4)},
        {"type": "way", "id": 3, "tags": {"building:part": "yes", "building:levels": "12", "building:min_level": "6"},
         "geometry": sq(lat - d / 2, lon - d / 2, d / 4)},
        {"type": "way", "id": 4, "tags": {"building:part": "yes", "building:levels": "18", "building:min_level": "14"},
         "geometry": sq(lat - d / 2 + d / 8, lon - d / 2 + d / 8, d / 4)},
        {"type": "way", "id": 5, "tags": {"highway": "primary", "bridge": "yes", "layer": "1"},
         "geometry": [{"lat": lat + 0.7 * d, "lon": lon - d}, {"lat": lat + 0.7 * d, "lon": lon + d}]},
        {"type": "way", "id": 6, "tags": {"highway": "residential"},
         "geometry": [{"lat": lat - d, "lon": lon + d / 3}, {"lat": lat + d, "lon": lon + d / 3}]},
        {"type": "way", "id": 7, "tags": {"landuse": "residential"}, "geometry": sq(lat - d, lon - d, 2 * d)},
        {"type": "way", "id": 8, "tags": {"leisure": "park"}, "geometry": sq(lat + d / 4, lon + d / 2, d / 4)},
    ]}


def test_city_from_osm_stacked_parts_bridges_zones():
    from octreecity.easy import city_from_osm
    from octreecity.objects import object_id

    lat, lon, data = interlace_like()
    city = city_from_osm(lat, lon, size_m=240, osm_json=data)
    o = city.objects
    ids = {k: object_id(f"osm:way/{k}") for k in range(1, 9)}
    assert all(ids[k] in o for k in ids)
    assert o.parent(ids[2]) == ids[1] and o.parent(ids[3]) == ids[1]
    assert o.parent(ids[8]) == ids[7]                               # park inside residential land use
    def z(k):
        return city.voxels(id=ids[k])[:, 2] * city.voxel_size + city.origin[2] + 0.5

    for k, lo, hi in ((2, 0, 18), (3, 18, 36), (4, 42, 54)):
        zz = z(k)
        assert zz.size and zz.min() > lo and zz.max() < hi and abs(zz.max() - hi) < 1 and abs(zz.min() - lo) < 1
    # the outline is painted only where no part covers it (2 levels = 6 m)
    zo = z(1)
    assert zo.size and zo.max() < 6
    # the bridge is lifted to 5 m; the road under it stays on the ground
    zb, zr = z(5), z(6)
    assert zb.min() > 4 and zr.max() < 0.5
    assert city.layers == ("id", "zone_id")
    tree = city.to_octree(5)
    nx, ny, nz = city.shape
    assert (tree.to_dense("id")[:nx, :ny, :nz] == city.to_dense(layer="id")).all()


def test_tiles_with_terrain_give_the_same_buildings():
    def dem(x, y):
        return 0.5 * x

    feats = Features([shapely.box(2, 2, 14, 10)], height=[5.0])

    def area(x0, w):
        b = CityBuilder(x0, 0, w, 16)
        b.add_terrain(dem)
        i = b.objects.add("test", ["a"], "building")
        b.add_buildings(feats, ids=i)
        return b.build(nz=24, z0=-1.0)

    whole, right = area(0, 16), area(8, 8)
    assert (whole.crop_columns(8, 0, 16, 16).to_dense(layer="id") == right.to_dense(layer="id")).all()
    assert (whole.crop_columns(8, 0, 16, 16).to_dense() == right.to_dense()).all()


def test_city_from_osm_with_no_data():
    from octreecity.easy import city_from_osm

    city = city_from_osm(36.37, 127.3625, 50, osm_json={"elements": []})
    assert city.shape[:2] == (50, 50)
