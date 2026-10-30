import numpy as np

from octreecity.occupancy import Occupancy
from octreecity.urban import BUILDING
from octreecity.viz import downsample, show, show_cells, show_values


def city():
    g = np.ones((64, 64))
    s = g.copy()
    s[10:20, 10:30] += 20
    s[40:50, 30:40] += 8
    return Occupancy.from_heightfield(g, s, BUILDING, 32)


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


def test_wireframes():
    import pytest

    from octreecity.viz import show_octree

    c = city()
    f = show(c, wireframe=True, step=1)
    edges = [t for t in f.data if t.type == "scatter3d"]
    assert edges and None in edges[0].x            # segments separated by gaps
    assert all(t.type == "scatter3d" for t in show(c, wireframe="only", step=1).data)
    with pytest.raises(ValueError, match="crop"):
        show(c, wireframe=True, step=1, max_edges=10)
    w = show_octree(c.to_octree(3), style="wire")
    assert any("m cells" in (t.name or "") for t in w.data if t.type == "scatter3d")


def test_wire_hides_smallest_when_too_many():
    from octreecity.viz import show_octree

    t = city().to_octree(3)
    n_air = int((t.label == 0).sum())
    f = show_octree(t, style="wire", max_cells=n_air // 3)
    names = [tr.name or "" for tr in f.data]
    assert any("hidden" in n for n in names)


def test_setup_notebook_colab(monkeypatch):
    import sys
    import types

    import plotly.io as pio

    from octreecity.viz import setup_notebook

    old = pio.renderers.default
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    assert setup_notebook() == "colab"
    pio.renderers.default = old


def selection_tree():
    import shapely

    from octreecity.builder import CityBuilder
    from octreecity.features import Features

    b = CityBuilder(0, 0, 64, 64)
    ids = b.objects.add("test", ["a", "b"], "building")
    b.add_buildings(Features([shapely.box(10, 10, 20, 30), shapely.box(40, 30, 50, 40)], height=[20, 8]), ids=ids)
    return b.build(air_above=6).to_octree(4), ids


def wire_names(fig):
    return [t.name for t in fig.data if t.type == "scatter3d"]


def test_any_group_draws():
    t, ids = selection_tree()
    g = t.select(id=ids[0])
    for sel in (g, g.neighbours(steps=2), g.within(5.0), g.siblings(), t.select(where=lambda c: c["z"] > 10)):
        f = sel.show()
        names = wire_names(f)
        assert names and all("m cells" in n for n in names)
        assert any(tr.type == "mesh3d" and "context" in tr.name and tr.opacity < 1 for tr in f.data)
        drawn = sum(len([x for x in tr.x if x is None]) for tr in f.data if tr.type == "scatter3d")
        assert drawn <= 12 * len(sel)                       # at most 12 edges per cube (shared edges merged)
    # the cubes sit in the same frame as show_octree(style="wire")
    from octreecity.viz import show_octree

    whole = show_octree(t, style="wire")
    xs = [x for tr in whole.data if tr.type == "scatter3d" for x in tr.x if x is not None]
    xg = [x for tr in g.show().data if tr.type == "scatter3d" for x in tr.x if x is not None]
    assert min(xs) <= min(xg) and max(xg) <= max(xs)
    assert min(xg) == 10.0 and max(xg) == 20.0            # the building's own footprint, in metres from the corner


def test_several_groups_one_colour_each():
    from octreecity.viz import show_octree

    t, ids = selection_tree()
    g = t.select(id=ids[0])
    f = show_octree(t, style="wire", cells={"building": g, "neighbours": g.neighbours(steps=1),
                                            "other": t.select(id=ids[1])})
    w = [tr for tr in f.data if tr.type == "scatter3d"]
    assert [tr.legendgroup for tr in w] == ["building", "neighbours", "other"]
    assert len({tr.line.color for tr in w}) == 3
    assert w[0].name == f"building ({len(g):,} cells)"
    # masks and plain indices work too; nothing selected still draws
    assert wire_names(show_octree(t, cells=t.label == BUILDING))
    assert show_octree(t, cells=np.zeros(0, int)).layout.title.text == "No cells selected"


def test_big_selection_is_thinned_with_a_note():
    t, ids = selection_tree()
    g = t.select(where=lambda c: c["label"] == 0)
    f = g.show(max_cells=len(g) // 4)
    names = wire_names(f)
    assert any("hidden" in n for n in names)
    from octreecity.viz import show_octree

    f2 = show_octree(t, cells={"air": g}, max_cells=len(g) // 4)
    assert "drawn)" in wire_names(f2)[0] and any("hidden" in n for n in wire_names(f2))
