import gzip

import numpy as np
import shapely

from octreecity.builder import GRASS, ROAD, CityBuilder, rasterize
from octreecity.octree import AIR, BUILDING, GROUND, TREE
from octreecity.sources import Features, read_citygml


def test_rasterize_square_and_hole():
    sq = shapely.Polygon([(10, 10), (30, 10), (30, 30), (10, 30)], holes=[[(15, 15), (25, 15), (25, 25), (15, 25)]])
    r = rasterize([sq], [5.0], 0, 0, 40, 40, 1.0)
    assert abs((r == 5).sum() - (400 - 100)) <= 40
    assert r[20, 20] == 0 and r[12, 12] == 5 and r[35, 12] == 0


def test_builder_layers():
    b = CityBuilder(0, 0, 64, 64, cell=1.0)
    b.add_terrain(lambda x, y: 0.1 * x)
    b.add_landcover(Features([shapely.LineString([(0, 5), (64, 5)])]), ROAD, line_width=4)
    b.add_landcover(Features([shapely.box(0, 40, 64, 64)]), GRASS)
    b.add_trees(Features([shapely.box(5, 45, 20, 60)]), top=10, base=3)
    b.add_buildings(Features([shapely.box(30, 20, 40, 30)], height=[15.0]))
    c = b.build(air_above=5)
    g = c.to_dense()
    x, y = 35, 25
    z0 = int(np.ceil((0.1 * 30 - c.origin[2]) - 0.5))
    col = g[x, y]
    assert BUILDING in col and col[-1] == AIR
    # building sits on the lowest terrain under it (x = 30 side), no gap below
    zb = np.argmax(col == BUILDING)
    assert col[zb - 1] != AIR
    tree_col = g[10, 50]
    zt = np.nonzero(tree_col == TREE)[0]
    assert zt.size and tree_col[zt[0] - 1] == AIR
    assert ROAD in g[20, 5] and GRASS in g[50, 50]


def test_read_citygml(tmp_path):
    xml = """<?xml version="1.0"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:bldg="http://www.opengis.net/citygml/building/2.0" xmlns:gml="http://www.opengis.net/gml">
<core:cityObjectMember><bldg:Building gml:id="b1"><bldg:measuredHeight>12.5</bldg:measuredHeight>
<gml:posList srsDimension="3">0 0 50 10 0 50 10 8 50 0 8 50 0 0 50</gml:posList>
<gml:posList srsDimension="3">0 0 62.5 10 0 62.5 10 8 62.5 0 8 62.5 0 0 62.5</gml:posList>
</bldg:Building></core:cityObjectMember></core:CityModel>"""
    p = tmp_path / "x.gml.gz"
    with gzip.open(p, "wt") as f:
        f.write(xml)
    feats = read_citygml(p)
    assert len(feats) == 1 and feats.height[0] == 12.5 and feats.base[0] == 50
    assert abs(feats.geoms[0].area - 80) < 1e-6


def test_city_from_osm_offline():
    from octreecity.easy import city_from_osm, utm_epsg

    lat, lon = 36.37, 127.36
    d = 0.0009  # ~100 m
    sq = lambda la, lo, s: [{"lat": la + a, "lon": lo + b} for a, b in ((0, 0), (0, s), (s, s), (s, 0), (0, 0))]
    data = {"elements": [
        {"type": "way", "tags": {"building": "yes", "building:levels": "5"}, "geometry": sq(lat - d / 4, lon - d / 4, d / 4)},
        {"type": "way", "tags": {"leisure": "park"}, "geometry": sq(lat + d / 8, lon + d / 8, d / 3)},
        {"type": "way", "tags": {"highway": "residential"},
         "geometry": [{"lat": lat - d / 2, "lon": lon - d}, {"lat": lat - d / 2, "lon": lon + d}]},
        {"type": "node", "tags": {"natural": "tree"}, "lat": lat + d / 2, "lon": lon - d / 2},
    ]}
    city = city_from_osm(lat, lon, size_m=200, cell=1.0, osm_json=data)
    assert utm_epsg(lat, lon) == 32652 and city.crs == "EPSG:32652"
    labels = set(np.unique(city.label).tolist())
    assert {BUILDING, GRASS, ROAD, TREE} <= labels
    assert city.top_of([BUILDING]).max() * city.voxel_size + city.origin[2] >= 14.0
