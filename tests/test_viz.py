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


def test_show_octree_styles():
    from octreecity.viz import show_octree

    t = city().to_octree(3)
    t.limit_size_near([BUILDING], 3, 0)
    t.balance(26)
    shells = show_octree(t, style="shells")
    assert any("cells up to" in (tr.name or "") for tr in shells.data)
    cut = show_octree(t, style="cutaway")
    names = [tr.name for tr in cut.data if tr.showlegend]
    assert len(names) == len(set(names)) and any("m cells" in n for n in names)
