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
    cubes = sum(np.sum([x is None for x in tr.x]) for tr in scatter(thin) if len(tr.x) > 1) / 12
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
        show_octree(t, style="dots")


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
    cubes = sum(np.sum([x is None for x in tr.x]) for tr in scatter(f) if len(tr.x) > 1) / 12
    assert cubes == drawn                                      # each cube draws its own 12 edges


# ------------------------------------------------------------ filled cubes

def _face_leaf(t, tr):
    """Leaf of each quad of a cube mesh: the voxel just inside the quad's centre."""
    v = np.stack([tr.x, tr.y, tr.z], 1).astype(float)
    n = len(tr.i) // 2
    quad = np.stack([tr.i[:n], tr.j[:n], tr.k[:n], tr.k[n:]], 1)
    corners = v[quad]                                     # (n, 4, 3)
    centre = corners.mean(1)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    c = t.voxel_size
    inward = []
    for sgn in (1, -1):                                    # the side that is inside a selected leaf
        p = centre + sgn * normal * 0.1 * c
        p[:, 2] -= t.origin[2]
        inward.append(t.locate(*(np.floor(p / c).astype(np.int64).T)))
    return np.array(inward), corners


def test_cubes_are_the_cells_with_their_own_value():
    t = tree()
    g = t.select(where=lambda c: c["label"] == BUILDING).neighbours(steps=1)
    f = g.show(color="height", style="cubes")
    mesh = [tr for tr in f.data if tr.type == "mesh3d" and tr.intensity is not None]
    assert len(mesh) == 1 and mesh[0].coloraxis == "coloraxis"
    tr = mesh[0]
    n = len(tr.i) // 2
    assert len(tr.intensity) == 2 * n
    leaves, corners = _face_leaf(t, tr)
    sel = np.zeros(t.n_leaves, bool)
    sel[g.idx] = True
    owner = np.where(sel[np.maximum(leaves[0], 0)] & (leaves[0] >= 0), leaves[0], leaves[1])
    assert sel[owner].all()                                  # every face belongs to a selected cell
    h = t.cells["height"].astype(np.float32)
    np.testing.assert_allclose(tr.intensity[:n], h[owner], rtol=1e-6)   # each face shows its own cell's value
    side = np.ptp(corners, axis=1).max(1)                    # faces are the cells' size, minus the seams
    np.testing.assert_allclose(side, t.sizes_m()[owner] - 0.12 * t.voxel_size, atol=1e-3)


def test_cubes_leave_out_faces_between_drawn_cells():
    from octreecity.viz import _cube_faces

    t = tree()
    one = np.array([int(np.nonzero(t.level == 0)[0][0])])
    drawn = np.zeros(t.n_leaves, bool)
    drawn[one] = True
    q, _ = _cube_faces(t, one, drawn)
    assert len(q) == 6
    x, y, z = (v[one[0]] for v in t.corners())
    nb = int(t.locate(x + 1, y, z))
    if t.level[nb] == 0:                                     # two equal neighbours: 10 faces, not 12
        pair = np.array(sorted([one[0], nb]))
        drawn[pair] = True
        q, _ = _cube_faces(t, pair, drawn)
        assert len(q) == 10


def test_cubes_by_category_size_and_group():
    t = tree()
    t.cells["zone"] = (t.centers()[:, 0] // 10).astype(np.int64)
    g = t.select(where=lambda c: c["label"] == BUILDING).neighbours(steps=1)
    f = show_octree(t, cells=g, style="cubes", color="zone")
    mesh = [tr for tr in f.data if tr.type == "mesh3d" and tr.facecolor is not None]
    keys = {tr.marker.color for tr in scatter(f) if tr.mode == "markers"}
    assert mesh and set(mesh[0].facecolor) <= keys            # face colours are legend colours
    f = g.show(style="cubes")                                  # no color=: by cell size, like wire
    names = [tr.name for tr in f.data if tr.type == "mesh3d" and "context" not in tr.name]
    assert names and all(n.split(" (")[0].endswith("m cells") for n in names)
    f = show_selection(t, {"a": g, "b": t.select(where=lambda c: c["z"] > 14)}, style="cubes", opacity=0.5)
    meshes = [tr for tr in f.data if tr.type == "mesh3d" and "context" not in tr.name]
    assert [tr.legendgroup for tr in meshes] == ["a", "b"] and all(tr.opacity == 0.5 for tr in meshes)
    with pytest.raises(ValueError, match="cells="):
        show_octree(t, style="cubes")
    with pytest.raises(ValueError, match="'wire' or 'cubes'"):
        show_selection(t, g, style="shells")


def test_cubes_thin_evenly_and_use_the_whole_budget():
    t = tree()
    g = t.select(where=lambda c: c["label"] == AIR)
    f = g.show(style="cubes", color="height", max_cells=1000)
    note = [n for n in legend(f) if "cells drawn (evenly spread)" in n][0]
    assert note.startswith("1,000 of ")


# ------------------------------------------------- trees split by a value

def test_trees_split_by_a_value_show_their_cells():
    """Cells split where a value changes (build + varies, adapt) are drawn
    as those cells, each with its own value."""
    from octreecity.adapt import adapt, sample
    from octreecity.build import build
    from octreecity.rules import mixed, varies
    from octreecity.sources import GridSource

    lab = np.zeros((32, 32, 16), np.uint8)
    lab[:, :, 0] = 1                                         # ground
    lab[6:12, 6:12, 1:10] = 2                                # a building
    x = np.arange(32)[:, None, None] + np.zeros((32, 32, 16))
    sun = np.where(x < 20, 2.0, 8.0)                          # a sharp change at x = 20 m
    t = build(GridSource(lab, attrs={"sun": sun}), mixed() | varies("sun", 1.0), max_level=4)
    s = t.cells["sun"]
    assert set(np.unique(np.round(s, 6))) <= {2.0, 8.0}       # no cell straddles the change
    for style in ("wire", "cutaway", "shells"):
        f = show_octree(t, style=style, color="sun")
        assert (f.layout.coloraxis.cmin, f.layout.coloraxis.cmax) == (2.0, 8.0)
    f = show_octree(t, style="cutaway", color="sun", cut=(24, 16))
    faces = [tr for tr in f.data if tr.type == "mesh3d" and tr.intensity is not None][0]
    assert set(np.unique(faces.intensity)) == {2.0, 8.0}
    t.select(where=lambda c: c["sun"] > 5).show(color="sun", style="cubes")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from octreecity.render import plot_slice
    fig, ax = plt.subplots()
    plot_slice(t, ax, color="sun")
    plt.close(fig)

    # solution-adaptive refinement: colour by the values that drove it
    a = Occupancy.from_dense(lab, 1.0).to_octree(4)
    field = lambda x, y, z: np.tanh((x - 20.0) / 1.5)        # noqa: E731
    vals = sample(a, field)
    for _ in range(2):
        a, vals = adapt(a, vals, refine_tol=0.3)
        vals = sample(a, field)
    f = show_octree(a, style="wire", color=vals)
    v = drawn_values(f)
    assert np.isclose(v.min(), vals[a.label == AIR].min(), atol=1e-6)
    with pytest.raises(ValueError, match="one per leaf"):
        show_octree(a, color=vals[:-3])                       # values of another tree


# ------------------------------------------- regressions from the stress test

def _segments(fig):
    """(start, end, value or colour) of every wire edge."""
    out = []
    for tr in scatter(fig):
        if len(tr.x) < 2:
            continue
        cols = tr.line.color if tr.line.coloraxis else [tr.line.color] * len(tr.x)
        for k in range(0, len(tr.x), 3):
            out.append((np.array([tr.x[k], tr.y[k], tr.z[k]], float),
                        np.array([tr.x[k + 1], tr.y[k + 1], tr.z[k + 1]], float), cols[k]))
    return out


def test_each_wire_cube_draws_its_own_edges_in_its_own_value():
    from octreecity.octree import LinearOctree
    from octreecity.viz import _SEAM

    t = LinearOctree.from_dense(np.zeros((8, 8, 8), np.uint8), 0)
    v = np.zeros(t.n_leaves)
    hot = int(t.locate(3, 4, 5))
    v[hot] = 100.0
    segs = _segments(show_octree(t, style="wire", color=v))
    assert len(segs) == 12 * t.n_leaves                       # no edge shared, none lost
    mine = [s for s in segs if float(s[2]) == 100.0]
    assert len(mine) == 12
    pts = np.array([p for s in mine for p in s[:2]])
    assert pts.min() >= 3 + _SEAM - 1e-6 and (pts <= np.array([4, 5, 6]) - _SEAM + 1e-6).all()


def test_wire_cubes_sit_on_their_cells_for_any_voxel_size():
    from octreecity.octree import LinearOctree
    from octreecity.viz import _SEAM

    for h in (0.25, 0.05):
        t = LinearOctree.from_dense(np.zeros((8, 8, 8), np.uint8), 1, voxel_size=h)
        plain = _segments(show_selection(t, np.arange(t.n_leaves), context=()))
        xs = np.unique(np.concatenate([[a[0], b[0]] for a, b, _ in plain]))
        np.testing.assert_allclose(xs, np.arange(0, 8 * h + 1e-9, 2 * h), atol=1e-6)   # cell corners, exact
        valued = _segments(show_octree(t, style="wire", color="z"))
        xs = np.unique(np.round(np.concatenate([[a[0], b[0]] for a, b, _ in valued]), 6))
        want = np.unique(np.round(np.concatenate([np.arange(0, 8 * h, 2 * h) + _SEAM * h,
                                                  np.arange(2 * h, 8 * h + 1e-9, 2 * h) - _SEAM * h]), 6))
        np.testing.assert_allclose(xs, want, atol=1e-6)


def test_large_values_keep_their_colours():
    t = tree()
    v = 1.7e9 + t.centers()[:, 0]                              # e.g. epoch seconds
    cut = [tr for tr in show_octree(t, style="cutaway", color=v).data
           if tr.type == "mesh3d" and tr.intensity is not None][0]
    assert len(np.unique(cut.intensity)) > 10
    assert len(set(cut.text)) > 10                             # hover tells them apart too
    assert len(np.unique(drawn_values(show_octree(t, style="wire", color=v)))) > 10


def test_categories_in_one_view_never_share_a_colour():
    t = tree()
    zone = (t.centers()[:, 0] // 2).astype(np.int64)           # 16 zones, more than the palette
    t.cells["zone_id"] = zone
    a, b = np.nonzero(zone == 0)[0], np.nonzero(zone == 12)[0]     # 12 ranks apart
    g = t.select(where=lambda c: np.isin(c["zone_id"], [0, 12]))
    colours = {tr.line.color for tr in scatter(g.show(color="zone_id", context=())) if len(tr.x) > 1}
    assert len(colours) == 2 and a.size and b.size


def test_cutaway_corner_counts_each_cell_once_and_stays_in_the_corner():
    t = tree()
    t.cells["zone"] = (t.centers()[:, 1] // 10).astype(np.int64)
    f = show_octree(t, style="cutaway", color="zone", cut=(13.0, 13.0))   # off the cell grid
    keys = {tr.name.split(" (")[0]: int(tr.name.split("(")[1].split(" ")[0].replace(",", ""))
            for tr in scatter(f) if tr.mode == "markers"}
    X, Y, Z = t.corners()
    S = t.sizes()
    on = np.unique(np.concatenate([np.nonzero((X <= 13) & (13 < X + S) & (Y < 13))[0],
                                   np.nonzero((Y <= 13) & (13 < Y + S) & (X < 13))[0]]))
    assert sum(keys.values()) == on.size                       # each cut cell counted once
    faces = [tr for tr in f.data if tr.type == "mesh3d" and tr.facecolor is not None][0]
    assert max(faces.x) <= 13.0 + 1e-6 and max(faces.y) <= 13.0 + 1e-6


def test_cuts_outside_the_block_say_so():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from octreecity.render import plot_slice

    t = tree()
    for call in (lambda: show_octree(t, style="cutaway", cut=(500.0, 8.0)),
                 lambda: show_octree(t, style="cutaway", color="height", cut=(-1.0, 8.0)),
                 lambda: show_cells(t, y=40.0, color="height"),
                 lambda: show_cells(t, y=40.0)):
        with pytest.raises(ValueError, match="outside the block"):
            call()
    with pytest.raises(ValueError, match="outside the tree"):
        plot_slice(t, plt.subplots()[1], index=99, color="height")
    plt.close("all")


def test_plot_slice_takes_plotly_scales_and_value_colours():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from octreecity.render import plot_slice

    t = tree()
    for scale in ("Portland", "Electric", [[0, "red"], [1, "rgb(0, 0, 255)"]]):
        plot_slice(t, plt.subplots()[1], color="height", colorscale=scale)
    t.cells["wet"] = t.centers()[:, 2] < 4
    pc = plot_slice(t, plt.subplots()[1], color="wet", colors={True: "#0000ff", False: "#ff0000"})
    assert {to_hex(c) for c in pc.get_facecolors()} >= {"#0000ff", "#ff0000"}
    fig, ax = plt.subplots()
    plot_slice(t, ax, color=np.full(t.n_leaves, np.nan))
    assert len(fig.axes) == 1                                  # no value anywhere: no colour bar
    plt.close("all")


def test_missing_values_of_every_kind():
    pd = pytest.importorskip("pandas")
    t = tree()
    n = t.n_leaves
    nums = [None if i % 3 == 0 else float(i) for i in range(n)]       # numbers with None
    f = show_octree(t, style="cutaway", color=np.array(nums, dtype=object))
    assert f.layout.coloraxis.colorbar.title.text == "value"
    text = pd.Series(["" if i % 2 else "shop" for i in range(n)], dtype="string")
    names = legend(show_octree(t, style="wire", color=text))
    assert not any(nm.startswith(" (") or nm.startswith("<NA>") for nm in names)
    ids = pd.Series([2 ** 53 + 1 if i % 2 else None for i in range(n)], dtype="Int64", name="building_id")
    names = legend(show_octree(t, style="wire", color=ids))
    assert any(nm.startswith(f"{2 ** 53 + 1} (") for nm in names)
    big = np.full(n, 2 ** 64 - 1, np.uint64)
    big[::2] = 2 ** 63 + 5
    t.cells["obj_id"] = big
    names = legend(show_octree(t, style="wire", color="obj_id"))
    assert any(nm.startswith(f"{2 ** 64 - 1} (") for nm in names) and any(nm.startswith(f"{2 ** 63 + 5} (") for nm in names)
    empty = show_octree(t, style="cutaway", color=np.full(n, np.nan))
    assert empty.layout.coloraxis.showscale is False


def test_colour_range_given_is_kept_or_refused():
    t = tree()
    with pytest.raises(ValueError, match="must be above"):
        show_octree(t, style="cutaway", color="height", cmin=10, cmax=4)
    with pytest.raises(ValueError, match="give cmax too"):
        show_octree(t, style="cutaway", color="height", cmin=1e6)
    f = show_octree(t, style="cutaway", color="height", cmax=5)
    assert f.layout.coloraxis.cmax == 5


def test_selection_inputs_are_checked():
    t = tree()
    g = t.select(where=lambda c: c["label"] == BUILDING)
    other = tree()
    with pytest.raises(ValueError, match="another tree"):
        show_selection(other, g)
    leaves = np.nonzero(t.label == AIR)[0][[30, 2, 17]]
    f = show_selection(t, leaves, color=np.array([3.0, 1.0, 2.0]), context=())
    got = {}
    for a, b, v in _segments(f):
        leaf = int(t.locate(*np.floor((a + b) / 2 / t.voxel_size - [0, 0, t.origin[2] / t.voxel_size]).astype(int)))
        got.setdefault(leaf, set()).add(float(v))
    assert got == {int(leaves[0]): {3.0}, int(leaves[1]): {1.0}, int(leaves[2]): {2.0}}


def test_warnings_when_a_view_cannot_show_the_values():
    from octreecity.octree import LinearOctree

    t = tree()
    on_solids = np.where(t.label == AIR, np.nan, 1.0)
    with pytest.warns(UserWarning, match="no air cell has"):
        show_octree(t, style="wire", color=on_solids)
    one = LinearOctree.from_dense(np.zeros((8, 8, 8), np.uint8), 1)
    with pytest.warns(UserWarning, match="at least two sizes"):
        show_octree(one, style="shells", color="z")


def test_many_classes_are_not_drawn_as_buildings():
    from octreecity.octree import LinearOctree

    lab = np.zeros((8, 8, 8), np.uint16)
    lab[2:4, 2:4, :4] = BUILDING + 256                         # class 258: not a building
    t = LinearOctree.from_dense(lab.astype(np.uint8), 2)
    t.label = np.where(t.label == BUILDING, BUILDING + 256, t.label.astype(np.uint16)).astype(np.uint16)
    for f in (show_octree(t, style="cutaway", color="z"), show_octree(t, style="wire", color="z"),
              show_selection(t, t.label == AIR)):
        assert not any(tr.type == "mesh3d" and tr.name.startswith("buildings") for tr in f.data)
