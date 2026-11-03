"""Colour any cell by a value: color= on show_octree, show_selection /
Group.show, show_cells and render.plot_slice."""

import warnings

import numpy as np
import pytest

from octreecity import fill
from octreecity.occupancy import Occupancy
from octreecity.urban import AIR, BUILDING
from octreecity.viz import STYLE, show_cells, show_octree, show_selection

plotly = pytest.importorskip("plotly")


def tree():
    g = np.ones((32, 32))
    s = g.copy()
    s[4:12, 4:14] += 12
    s[20:26, 16:22] += 6
    t = Occupancy.from_heightfield(g, s, BUILDING, 16).to_octree(3)
    t.limit_size_near([BUILDING], 3, 0)
    t.balance(26)
    t.cells["height"] = fill.from_function(t, lambda x, y, z: z)
    return t


def scatter(fig):
    return [tr for tr in fig.data if tr.type == "scatter3d"]


def legend(fig):
    return [tr.name for tr in fig.data if tr.showlegend is not False]


def drawn_values(fig):
    """Colour values on all wire traces (one per point, gaps included)."""
    return np.concatenate([np.asarray(tr.line.color, float) for tr in scatter(fig)
                           if tr.line.coloraxis == "coloraxis"])


# ---------------------------------------------------------------- numbers

def test_numeric_wire_uses_one_scale_over_the_whole_tree():
    t = tree()
    f = show_octree(t, style="wire", color="height")
    h = t.cells["height"]
    assert f.layout.coloraxis.cmin == pytest.approx(h.min())
    assert f.layout.coloraxis.cmax == pytest.approx(h.max())
    assert f.layout.coloraxis.colorbar.title.text == "height"
    v = drawn_values(f)
    air = t.label == AIR
    assert set(np.round(v, 3)) <= set(np.round(h[air].astype(np.float32), 3))   # each edge has its cube's value
    assert all(n.endswith("m cells") for n in legend(f) if n != "buildings")


def test_thinning_keeps_colours_and_says_how_many_are_drawn():
    t = tree()
    n_air = int((t.label == AIR).sum())
    full = show_octree(t, style="wire", color="height")
    thin = show_octree(t, style="wire", color="height", max_cells=n_air // 5)
    assert thin.layout.coloraxis.cmin == full.layout.coloraxis.cmin
    assert thin.layout.coloraxis.cmax == full.layout.coloraxis.cmax
    names = legend(thin)
    assert any(f"of {n_air:,} air cells drawn" in n for n in names)
    assert len(names) == len(set(names))
    cubes = sum(np.sum([x is None for x in tr.x]) for tr in scatter(thin)) / 12
    assert cubes <= n_air // 5
    sizes = {n.split(" m cells")[0] for n in names if " m cells" in n}
    assert len(sizes) == len(np.unique(t.level[t.label == AIR]))     # every size still drawn


def test_cmin_cmax_and_colorscale():
    t = tree()
    f = show_octree(t, style="cutaway", color="height", cmin=0, cmax=5, colorscale="Inferno")
    assert (f.layout.coloraxis.cmin, f.layout.coloraxis.cmax) == (0, 5)
    assert f.layout.coloraxis.colorscale[0][1] == plotly.colors.get_colorscale("Inferno")[0][1]


def test_builtin_columns_and_arrays():
    t = tree()
    for c in ("size_m", "z", "x"):
        assert show_octree(t, style="cutaway", color=c).layout.coloraxis.colorbar.title.text == c
    a = np.arange(t.n_leaves, dtype=float)
    f = show_octree(t, style="cutaway", color=a)
    assert f.layout.coloraxis.cmax == t.n_leaves - 1
    with pytest.raises(ValueError, match="one per leaf"):
        show_octree(t, color=a[:-1])
    with pytest.raises(KeyError, match="no cell column"):
        show_octree(t, color="nope")
    t.cells["vec"] = np.zeros((t.n_leaves, 3))
    with pytest.raises(ValueError, match="one value per cell"):
        show_octree(t, color="vec")


def test_cutaway_faces_and_shells_carry_values():
    t = tree()
    cut = show_octree(t, style="cutaway", color="height")
    faces = [tr for tr in cut.data if tr.type == "mesh3d" and tr.intensity is not None]
    assert faces and all(tr.coloraxis == "coloraxis" and tr.intensitymode == "cell" for tr in faces)
    assert len(faces[0].intensity) == len(faces[0].i)            # one value per triangle
    assert "%{text}" in faces[0].hovertemplate                   # hover shows the value
    shells = show_octree(t, style="shells", color="height")
    sh = [tr for tr in shells.data if tr.type == "mesh3d" and tr.intensity is not None]
    assert sh and all("cells up to" in tr.name for tr in sh)
    assert all(len(tr.intensity) == len(tr.i) for tr in sh)
    # every shell face sits on an air cell no bigger than the shell
    for tr in sh:
        assert np.nanmin(tr.intensity) >= t.cells["height"].min()


def test_show_cells_by_value():
    t = tree()
    f = show_cells(t, color="height")
    assert f.layout.coloraxis.colorbar.title.text == "height"
    assert any(tr.type == "mesh3d" and tr.intensity is not None for tr in f.data)


# ------------------------------------------------------------- categories

def test_class_and_label_are_categories_in_their_own_colours():
    t = tree()
    for c in ("label", "class"):
        f = show_octree(t, style="cutaway", color=c, cut=(8, 8))   # through the tall building
        assert not f.layout.coloraxis.colorbar.title.text           # no colour bar
        keys = {tr.name.split(" (")[0]: tr.marker.color for tr in scatter(f) if tr.mode == "markers"}
        assert keys.get("building") == STYLE[BUILDING][1] or keys.get("buildings") == STYLE[BUILDING][1]
        faces = [tr for tr in f.data if tr.type == "mesh3d" and tr.facecolor is not None]
        assert faces and STYLE[BUILDING][1] in set(faces[0].facecolor)


def test_ids_are_exact_categories_never_a_gradient():
    t = tree()
    big = 2 ** 53 + 1                                          # not exact as a float64
    ids = np.full(t.n_leaves, -1, np.int64)
    air = np.nonzero(t.label == AIR)[0]
    ids[air[: air.size // 2]] = big
    ids[air[air.size // 2:]] = big + 2
    ids[air[:3]] = -2                                          # mixed
    t.cells["zone_id"] = ids
    f = show_octree(t, style="wire", color="zone_id")
    assert not f.layout.coloraxis.colorbar.title.text
    names = legend(f)
    assert any(n.startswith(f"{big} (") for n in names)
    assert any(n.startswith(f"{big + 2} (") for n in names)
    assert any(n.startswith("mixed (") for n in names)
    colors = [tr.line.color for tr in scatter(f) if tr.name.split(" (")[0] in (str(big), str(big + 2))]
    assert len(set(colors)) == 2 and all(isinstance(c, str) for c in colors)


def test_few_integers_are_categories_many_are_numbers():
    t = tree()
    t.cells["zone"] = (t.centers()[:, 0] // 10).astype(np.int64)        # 4 values
    t.cells["count"] = np.arange(t.n_leaves, dtype=np.int64)             # many values
    assert not show_octree(t, style="cutaway", color="zone").layout.coloraxis.colorbar.title.text
    assert show_octree(t, style="cutaway", color="count").layout.coloraxis.colorbar.title.text == "count"
    # text and bools are categories; .astype(float) asks for a scale
    t.cells["kind"] = np.where(t.centers()[:, 2] > 8, "high", "low")
    f = show_octree(t, style="wire", color="kind")
    assert {n.split(" (")[0] for n in legend(f)} >= {"high", "low"}
    assert show_octree(t, style="cutaway", color=t.cells["zone"].astype(float)).layout.coloraxis.cmax == 3


def test_many_categories_repeat_colours_with_one_note():
    t = tree()
    t.cells["obj_id"] = np.arange(t.n_leaves, dtype=np.int64)
    f = show_octree(t, style="wire", color="obj_id", max_cells=400)
    names = legend(f)
    assert any("different obj_id values (colours repeat)" in n for n in names)
    assert len(names) < 10


# ---------------------------------------------------------------- missing

def test_missing_values_grey_or_hidden():
    t = tree()
    h = t.cells["height"].copy()
    air = np.nonzero(t.label == AIR)[0]
    h[air[::3]] = np.nan
    n_missing = int(np.isnan(h[air]).sum())
    t.cells["h"] = h
    grey = show_octree(t, style="wire", color="h")
    assert any(n == f"no value ({n_missing:,} cells)" for n in legend(grey))
    grey_cubes = [tr for tr in scatter(grey) if tr.name == "no value" and tr.x]
    assert grey_cubes and "128,133,141" in grey_cubes[0].line.color
    hide = show_octree(t, style="wire", color="h", nan="hide")
    assert any(n == f"no value: {n_missing:,} cells hidden" for n in legend(hide))
    assert not [tr for tr in scatter(hide) if tr.name == "no value" and tr.x]
    assert np.isfinite(drawn_values(hide)).all()
    with pytest.raises(ValueError, match="nan="):
        show_octree(t, color="h", nan="white")


def test_solids_without_a_value_keep_their_colour_on_the_cut():
    t = tree()
    h = np.where(t.label == AIR, t.cells["height"], np.nan)
    f = show_octree(t, style="cutaway", color=h, cut=(8, 8))
    solid = [tr for tr in f.data if tr.type == "mesh3d" and tr.color == STYLE[BUILDING][1]
             and tr.name == "buildings" and tr.lighting.diffuse == 0]     # flat: on the cut, not the 3D mesh
    assert solid                                               # buildings on the cut faces stay buildings
    assert not any(n.startswith("no value") for n in legend(f))


# --------------------------------------------------------------- selections

def test_group_show_by_value_and_by_group_values():
    t = tree()
    g = t.select(where=lambda c: c["label"] == BUILDING)
    f = g.neighbours(steps=2).show(color="steps")             # steps 1, 2: categories
    assert {n.split(" (")[0] for n in legend(f) if "context" not in n} == {"1", "2"}
    f = g.show(color="z")
    zs = t.centers()[g.idx, 2]
    assert f.layout.coloraxis.cmin == pytest.approx(zs.min())   # the selection's range
    assert f.layout.coloraxis.cmax == pytest.approx(zs.max())
    assert "coloured by z" in f.layout.title.text
    f = show_selection(t, g, color=np.arange(len(g), dtype=float))   # one value per selected cell
    assert f.layout.coloraxis.cmax == len(g) - 1
    with pytest.raises(ValueError, match="Join the groups"):
        show_selection(t, {"a": g, "b": g.neighbours()}, color="z")


def test_style_with_cells_warns():
    t = tree()
    g = t.select(where=lambda c: c["label"] == BUILDING)
    with pytest.warns(UserWarning, match="not used with cells="):
        show_octree(t, style="cutaway", cells=g)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        show_octree(t, cells=g)
        show_octree(t, style="wire", cells=g)
    with pytest.raises(ValueError, match="style="):
        show_octree(t, style="cubes")


# ----------------------------------------------------------- plot_slice

def test_plot_slice_by_value():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from octreecity.render import plot_slice

    t = tree()
    h = t.cells["height"].copy()
    h[(t.label == AIR) & (np.arange(t.n_leaves) % 4 == 0)] = np.nan
    t.cells["h"] = h
    fig, ax = plt.subplots()
    pc = plot_slice(t, ax, color="h", colorscale="Viridis")
    assert len(fig.axes) == 2                                   # the colour bar
    assert fig.axes[1].get_ylabel() == "h"
    fc = [to_hex(c) for c in pc.get_facecolors()]
    assert "#80858d" in fc                                      # missing air in grey
    n_all = len(fc)
    pc2 = plot_slice(t, ax, color="h", nan="hide", legend=False)
    assert len(pc2.get_facecolors()) < n_all
    fig2, ax2 = plt.subplots()
    plot_slice(t, ax2, color="class")
    labels = [x.get_text() for x in ax2.get_legend().get_texts()]
    assert any(s.startswith("building (") for s in labels)
    assert len(fig2.axes) == 1                                  # no colour bar for categories
    plt.close("all")


def test_figure_size_limits():
    t = tree()
    with pytest.raises(ValueError, match="crop a smaller block"):
        show_octree(t, style="shells", color="height", max_faces=10)
    f = show_octree(t, style="wire", color="height")             # 15,000 cubes at most by default
    n_air = int((t.label == AIR).sum())
    assert n_air > 15_000 or not any("drawn (evenly spread)" in n for n in legend(f))


def test_hidden_cells_are_not_counted_as_drawn():
    t = tree()
    h = t.cells["height"].copy()
    air = np.nonzero(t.label == AIR)[0]
    h[air[::2]] = np.nan
    t.cells["h"] = h
    pool = int(np.isfinite(h[air]).sum())
    f = show_octree(t, style="wire", color="h", nan="hide", max_cells=pool // 2)
    note = [n for n in legend(f) if "air cells drawn" in n][0]
    drawn = int(note.split(" of ")[0].replace(",", ""))
    assert f"of {pool:,} air cells" in note
    cubes = sum(np.sum([x is None for x in tr.x]) for tr in scatter(f)) / 12
    assert cubes <= drawn                                      # shared edges are drawn once
