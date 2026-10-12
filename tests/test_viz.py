import numpy as np

from octreecity.columns import ColumnCity
from octreecity.octree import BUILDING
from octreecity.viz import downsample, show, show_cells, show_values


def city():
    g = np.ones((64, 64))
    s = g.copy()
    s[10:20, 10:30] += 20
    s[40:50, 30:40] += 8
    return ColumnCity.from_heightfield(g, s, BUILDING, 32)


def test_show_and_values_and_cells():
    c = city()
    f = show(c)
    assert any(t.type == "mesh3d" and t.name == "buildings" for t in f.data)
    grid = np.random.default_rng(0).random((32, 32))
    f2 = show_values(c, grid, "test", every=2)
    assert f2.data[0].type == "surface"
    f3 = show_cells(c.to_octree(3))
    assert any("cells" in (t.name or "") for t in f3.data)


def test_downsample_keeps_buildings():
    d = downsample(city(), 2)
    assert d.shape == (32, 32, 16) and (d.label == BUILDING).any()
