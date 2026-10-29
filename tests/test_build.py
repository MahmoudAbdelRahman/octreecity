import numpy as np
import pytest

from octreecity.build import build, uniform
from octreecity.classes import Classes
from octreecity.columns import ColumnCity
from octreecity.octree import LinearOctree
from octreecity.rules import changes, larger_than, mixed, varies, where
from octreecity.sources import ColumnSource, GridSource, SparseSource


def random_columns(seed=0, n=40, nz=24):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, (n, n)).astype(float)
    s = g + np.where(rng.random((n, n)) < 0.3, rng.integers(1, 15, (n, n)), 0)
    lab = rng.integers(2, 4, (n, n))
    city = ColumnCity.from_heightfield(g, s, lab, nz)
    h = np.where(s > g, s - g, np.nan)
    return city, h


def three_sources(city, h):
    d = city.to_dense()
    hg = np.where(d != 0, np.repeat(h[..., None], d.shape[2], 2), np.nan)
    X, Y, Z = np.nonzero(d)
    return (ColumnSource(city, attrs={"h": h}),
            GridSource(d, attrs={"h": hg}),
            SparseSource(X, Y, Z, d[X, Y, Z], attrs={"h": hg[X, Y, Z]}, shape=d.shape))


@pytest.mark.parametrize("seed", [0, 1])
def test_sources_agree_and_match_bottom_up(seed):
    city, h = random_columns(seed)
    d = city.to_dense()
    S = 8
    pad = [(0, -(-n // S) * S - n) for n in d.shape]
    ref = LinearOctree.from_dense(np.pad(d, pad), 3)
    for src in three_sources(city, h):
        t = build(src, mixed(), 3, shares=True)
        assert t.check()
        assert np.array_equal(t.code, ref.code) and np.array_equal(t.label, ref.label)
        assert np.allclose(t.cells["share"].sum(1), 1)
    a, b, c = three_sources(city, h)
    for lv in (3, 2, 1, 0):
        s = 1 << lv
        x, y, z = (v.ravel() for v in np.meshgrid(*[np.arange(0, n, s) for n in (48, 48, 24)], indexing="ij"))
        sa, sb, sc = (src.summarize(x, y, z, s) for src in (a, b, c))
        assert np.allclose(sa.counts[:, :4], sb.counts) and np.allclose(sc.counts, sb.counts)
        for st in (sa, sc):
            assert np.allclose(st.stats["h"], sb.stats["h"], equal_nan=True)


def test_rules_semantic_depth_and_attributes():
    c = Classes.from_tree({"building": {"wall": {}, "roof": {}}})
    lab = np.full((8, 8, 8), c.id("wall"), np.uint8)
    lab[:, :, 5:] = c.id("roof")
    year = np.full(lab.shape, 1990.0)
    year[:4, :4] = 2010.0
    src = GridSource(lab, attrs={"year": year}, classes=c)
    coarse = build(src, mixed(depth=0), 3, label_depth=0)    # wall + roof = one building
    assert coarse.n_leaves == 1 and coarse.label[0] == c.id("building")
    fine = build(src, mixed(), 3)                             # wall and roof differ
    assert fine.n_leaves == 4 + 4 * (4 * 8 + 4)              # wall/roof boundary at z = 5 down to 1 voxel
    assert set(np.unique(fine.label)) == {c.id("wall"), c.id("roof")}
    assert build(src, mixed(depth=0) | changes("year"), 3).n_leaves == 8
    assert build(src, varies("year", 5), 3, stats=("std",)).cells["year_std"].max() == 0
    assert build(src, varies("year", 50), 3).n_leaves == 1
    assert build(src, larger_than(2.0), 3).sizes_m().max() == 2.0
    custom = build(src, where(lambda b: b.z == 0) & mixed(depth=0), 3)
    assert custom.n_leaves == 1                               # the root has z == 0 but is not mixed
    assert build(src, where(lambda b: b.z == 0) | mixed(), 3).level.min() == 0
    assert (build(src, lambda b: np.zeros(len(b), bool), 3).level == 3).all()
    assert build(src, ~mixed(), 3).n_leaves == 1                  # the root is mixed, so it is kept


def test_uniform_and_padding():
    city, h = random_columns(2, n=20, nz=10)
    u = uniform(ColumnSource(city), 1)
    assert (u.level == 1).all() and u.shape == (20, 20, 10) and u.check()
    t = build(ColumnSource(city), mixed(), 3, min_level=1)
    assert t.level.min() == 1 and t.shape == (24, 24, 16) and t.check()


def test_points_majority_class():
    xyz = np.array([[0.2, 0.2, 0.2], [0.7, 0.1, 0.3], [0.5, 0.5, 0.9], [2.5, 0.5, 0.5]])
    src = SparseSource.from_points(xyz, [1, 2, 2, 3], voxel_size=1.0, origin=(0, 0, 0), shape=(4, 4, 4),
                                   attrs={"i": [1.0, 3.0, 5.0, 7.0]})
    assert src.code.size == 2 and list(src.labels) == [2, 3]
    s = src.summarize([0], [0], [0], 4)
    assert s.counts[0, 2] == 1 and s.counts[0, 3] == 1 and s.counts[0, 0] == 62
    assert s.mean("i")[0] == pytest.approx((3.0 + 7.0) / 2)


def test_citygml_surfaces_and_from_surfaces(tmp_path):
    from octreecity.features import read_citygml_surfaces

    xml = """<?xml version="1.0"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
 xmlns:gml="http://www.opengis.net/gml" xmlns:gen="http://www.opengis.net/citygml/generics/2.0">
<core:cityObjectMember><bldg:Building gml:id="b1"><gen:doubleAttribute name="year"><gen:value>1999</gen:value></gen:doubleAttribute>
<bldg:measuredHeight>4</bldg:measuredHeight>
<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>
<gml:posList srsDimension="3">0 0 4 4 0 4 4 4 4 0 4 4 0 0 4</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>
<bldg:boundedBy><bldg:WallSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>
<gml:posList srsDimension="3">0 0 0 4 0 0 4 0 4 0 0 4 0 0 0</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:WallSurface></bldg:boundedBy>
</bldg:Building></core:cityObjectMember></core:CityModel>"""
    p = tmp_path / "b.gml"
    p.write_text(xml)
    s = read_citygml_surfaces(p)
    assert list(s.kind) == ["RoofSurface", "WallSurface"] and s.objects[0]["year"] == 1999.0
    assert s.objects[0]["measuredHeight"] == 4.0 and s.objects[0]["id"] == "b1"
    c = Classes.from_tree({"building": {"RoofSurface": {}, "WallSurface": {}}})
    src = SparseSource.from_surfaces(s.polygons, c.ids(s.kind), 1.0, origin=(0, 0, 0), shape=(8, 8, 8), classes=c)
    counts = np.bincount(src.labels, minlength=len(c))
    assert counts[c.id("RoofSurface")] == 16                 # the 4 x 4 roof at z = 4 fills one voxel layer
    assert counts[c.id("WallSurface")] == 16                 # the wall at y = 0, 4 m x 4 m
    t = build(src, mixed(depth=1), 3, label_depth=1)
    assert set(np.unique(t.label)) == {0, c.id("RoofSurface"), c.id("WallSurface")}
