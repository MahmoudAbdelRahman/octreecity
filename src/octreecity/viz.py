"""Interactive 3D views in a notebook (plotly): rotate, zoom, hover.

    from octreecity.viz import show, show_values, show_cells
    show(city)                                  # the city in 3D
    show_values(city, svf_map, "Sky view")      # street-level results on the 3D city
    show_cells(tree, y=300)                     # adaptive cells on a vertical cut
    tree.select(id=7).show()                    # any selection of cells, as wire cubes
    show_octree(tree, cells={"building": g, "around": g.neighbours(steps=1)})

Big cities are simplified for display only (cells merged by `step`), so a
view stays quick; the data itself is untouched.
"""

import numpy as np

from .occupancy import Occupancy
from .urban import AIR, BUILDING, GRASS, GROUND, PAVEMENT, ROAD, TREE, WATER

STYLE = {
    BUILDING: ("buildings", "#d9dce3"),
    TREE: ("trees", "#4d9a54"),
    GROUND: ("ground", "#9a8a6c"),
    GRASS: ("grass", "#8cbf5c"),
    WATER: ("water", "#4a8fd8"),
    PAVEMENT: ("paths", "#c9bfae"),
    ROAD: ("roads", "#6b6f76"),
}
BG = "#11151b"
DISPLAY_COLUMNS = 150_000      # bigger cities are simplified for display only; lower for lighter figures


def downsample(city, step):
    """Coarser Occupancy for display: each step^3 block takes its most
    common label, except that any solid wins over air when it fills at
    least a quarter of the block (keeps thin buildings visible)."""
    if step <= 1:
        return city
    nx, ny, nz = city.shape
    X, Y, Z = nx // step, ny // step, nz // step
    labels = [int(v) for v in np.unique(city.label)]
    out = np.zeros((X, Y, Z), np.uint8)
    for x0 in range(0, X * step, 32 * step):
        x1 = min(X * step, x0 + 32 * step)
        g = city.to_dense(x0, x1)[:, :Y * step, :Z * step]
        b = g.reshape((x1 - x0) // step, step, Y, step, Z, step)
        counts = np.stack([(b == lab).sum(axis=(1, 3, 5)) for lab in labels])
        best = np.array(labels, np.uint8)[counts.argmax(0)]
        solid = counts.copy()
        solid[labels.index(AIR)] = -1 if AIR in labels else 0
        sbest = np.array(labels, np.uint8)[solid.argmax(0)]
        enough = solid.max(0) * 4 >= step ** 3
        out[x0 // step:x1 // step] = np.where(enough, sbest, best)
    return Occupancy.from_dense(out, city.voxel_size * step, city.origin)


def _auto_step(city, max_columns=None):
    max_columns = DISPLAY_COLUMNS if max_columns is None else max_columns
    n = city.shape[0] * city.shape[1]
    return max(1, int(np.ceil(np.sqrt(n / max_columns))))


def _meshes(city, labels):
    import plotly.graph_objects as go

    from .exports import surface_quads

    q, lab = surface_quads(city, labels=labels)
    ox, oy, oz = city.origin
    traces = []
    for L in labels:
        sel = q[lab == L]
        if len(sel) == 0:
            continue
        v = sel.reshape(-1, 3) - np.array([ox, oy, 0])
        # share vertices between faces and use 32-bit numbers: 3-4x smaller figures
        v, inv = np.unique(np.round(v, 3), axis=0, return_inverse=True)
        quad = inv.reshape(-1, 4).astype(np.int32)
        i = np.concatenate([quad[:, 0], quad[:, 0]])
        j = np.concatenate([quad[:, 1], quad[:, 2]])
        k = np.concatenate([quad[:, 2], quad[:, 3]])
        v = v.astype(np.float32)
        name, color = STYLE.get(L, (f"label {L}", "#cc4444"))
        traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=i, j=j, k=k, color=color,
                                name=name, showlegend=True, flatshading=True, hoverinfo="name",
                                lighting=dict(ambient=0.32, diffuse=1.0, specular=0.12, roughness=0.7,
                                              fresnel=0.05, facenormalsepsilon=0),
                                lightposition=dict(x=-60000, y=-100000, z=90000)))
    return traces


def _layout(fig, city, title, height):
    nx, ny, nz = city.shape
    c = city.voxel_size
    w, d = nx * c, ny * c
    zmax = nz * c
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6e9ee", size=15), x=0.02),
        height=height, autosize=True, paper_bgcolor=BG, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(color="#e6e9ee"), bgcolor="rgba(0,0,0,0)", x=0.01, y=0.95),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="manual", aspectratio=dict(x=1, y=d / w, z=max(0.08, zmax / w)),
            bgcolor=BG, camera=dict(eye=dict(x=-0.85, y=-1.0, z=0.62), up=dict(x=0, y=0, z=1)),
        ),
    )
    return fig


def _edge_trace(segments, name, color="rgba(20,24,30,0.85)", width=1.5, legend=True, values=None):
    """Line segments (N, 2, 3) as one Scatter3d trace. Plain lists with None
    between segments: compact typed arrays would join all segments into one
    long line. values: one number per segment, drawn on the figure's colour
    scale (coloraxis) instead of `color`."""
    import plotly.graph_objects as go

    seg = np.round(np.asarray(segments, np.float64), 3)
    if len(seg):                                   # drop repeated edges (shared by neighbouring cubes or faces)
        a, b = seg[:, 0], seg[:, 1]
        lo = np.where((a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & ((a[:, 1] < b[:, 1]) | ((a[:, 1] == b[:, 1]) & (a[:, 2] <= b[:, 2])))), 0, 1)
        canon = np.where(lo[:, None, None] == 0, seg, seg[:, ::-1])
        _, keep = np.unique(canon.reshape(len(seg), 6), axis=0, return_index=True)
        keep = np.sort(keep)
        seg = canon[keep]
        values = None if values is None else np.asarray(values)[keep]
    xs, ys, zs = [], [], []
    for (a, b) in seg.tolist():
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    line = dict(color=color, width=width)
    if values is not None:                         # same value at both ends and at the gap: one colour per edge
        line = dict(color=np.repeat(np.asarray(values), 3), coloraxis="coloraxis", width=width)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", connectgaps=False, name=name,
                        line=line, hoverinfo="skip", showlegend=legend)


def _quad_edges(q):
    """Unique edges of quads (N, 4, 3) as (M, 2, 3) segments."""
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    key = np.sort(np.round(e, 2).reshape(len(e), 2, 3).view([("", np.float64)] * 3).reshape(len(e), 2), axis=1)
    _, first = np.unique(key.view(np.dtype((np.void, key.dtype.itemsize * 2))), return_index=True)
    return e[np.sort(first)]


def show(city, labels=None, step=None, title=None, height=700, wireframe=False, max_edges=60_000):
    """Interactive 3D view of an Occupancy (drag to rotate, scroll to zoom).

    labels: which labels to draw (default: every solid label).
    step: display simplification (merge step x step cells); chosen
    automatically for big cities.
    wireframe: True draws the faces with their edges on top, "only" draws
    just the edges (see-through), to explain how the model is built. For
    wireframes, crop a small area first (city.crop): at most max_edges
    edges are drawn.
    """
    import plotly.graph_objects as go

    from .exports import surface_quads

    step = _auto_step(city) if step is None else step
    view = downsample(city, step)
    labels = [int(v) for v in np.unique(view.label) if v != AIR] if labels is None else labels
    traces = [] if wireframe == "only" else _meshes(view, labels)
    if wireframe:
        q, lab = surface_quads(view, labels=labels)
        q = q - np.array([view.origin[0], view.origin[1], 0])
        edges = _quad_edges(q)
        if len(edges) > max_edges:
            raise ValueError(f"{len(edges):,} edges: crop a smaller area (city.crop) or raise max_edges")
        color = "rgba(20,24,30,0.85)" if wireframe != "only" else "rgba(240,166,90,0.9)"
        traces.append(_edge_trace(edges, "edges", color=color, width=1.5 if wireframe != "only" else 2))
    fig = go.Figure(traces)
    km = city.shape[0] * city.voxel_size
    note = "" if step == 1 else f", shown at {view.voxel_size:g} m"
    return _layout(fig, view, title or f"{km:g} x {city.shape[1] * city.voxel_size:g} m city{note}", height)


def street_map(city, pts_index, values, every):
    """Put point values (from street_points(city, every)) back on a 2D grid."""
    ii, jj = pts_index
    g = np.full((city.shape[0] // every + (city.shape[0] % every > 0),
                 city.shape[1] // every + (city.shape[1] % every > 0)), np.nan)
    g[ii // every, jj // every] = values
    return g


def show_values(city, grid, name, every=None, colorscale="Inferno", cmin=None, cmax=None,
                step=None, height=700, height_above=0.6):
    """3D city with a 2D result (sky view, sun hours, wind...) painted just
    above the ground. grid: [nx/every, ny/every] values, nan where none."""
    import plotly.graph_objects as go

    from .analysis import ground_level

    every = every or max(1, round(city.shape[0] / grid.shape[0]))
    step = _auto_step(city) if step is None else step
    view = downsample(city, step)
    traces = _meshes(view, [BUILDING, TREE])
    c = city.voxel_size
    g = ground_level(city)[::every, ::every][:grid.shape[0], :grid.shape[1]] - city.origin[2]
    xs = (np.arange(grid.shape[0]) + 0.5) * every * c
    ys = (np.arange(grid.shape[1]) + 0.5) * every * c
    z = np.where(np.isnan(grid), np.nan, g + height_above + city.origin[2])
    traces.insert(0, go.Surface(x=xs, y=ys, z=z.T, surfacecolor=grid.T, colorscale=colorscale,
                                cmin=cmin, cmax=cmax, name=name, showscale=True,
                                colorbar=dict(title=dict(text=name, font=dict(color="#e6e9ee")),
                                              tickfont=dict(color="#e6e9ee"), len=0.6, x=0.97),
                                lighting=dict(ambient=0.9, diffuse=0.2), hovertemplate=f"{name}: %{{surfacecolor:.2f}}<extra></extra>"))
    # ground for context where there are no values
    fig = go.Figure(traces)
    fig = _layout(fig, view, f"{name} at street level", height)
    fig.update_layout(scene_camera=dict(eye=dict(x=-0.35, y=-0.55, z=1.05)))   # look down into the streets
    return fig


def show_cells(tree, y=None, title=None, height=650, max_cells=80_000, color=None, colorscale=None,
               cmin=None, cmax=None, nan="grey"):
    """Adaptive octree cells cut by a vertical plane, as filled squares:
    air cells coloured by size, solids in their label colour, thin edges.
    y: metres from the city edge; default: the row crossing most buildings.
    color: colour the squares by a value instead (see show_octree)."""
    import plotly.graph_objects as go

    if color is not None:
        return _show_cells_by_value(tree, y, title, height, max_cells, color, colorscale, cmin, cmax, nan)

    c = tree.voxel_size
    X, Y, Z = tree.corners()
    S = tree.sizes()
    yv = _cut_row(tree, y)
    hit = (Y <= yv) & (yv < Y + S)
    x0, z0, s, lab = X[hit] * c, Z[hit] * c + tree.origin[2], S[hit] * c, tree.label[hit]
    if hit.sum() > max_cells:
        keep = np.argsort(-s)[:max_cells]
        x0, z0, s, lab = x0[keep], z0[keep], s[keep], lab[keep]
    yy = (yv + 0.5) * c
    palette = ["#f6c48f", "#f3e1a0", "#cfe6b4", "#a9dccf", "#a8c8e8", "#c3c4ee", "#dccdf0"]
    sizes = list(np.unique(s[lab == AIR]))
    traces = []

    gap = 0.08 * c          # thin dark gap between squares shows the cell edges

    def quads(xa, za, sa, color, name, legend=True):
        n = len(xa)
        xa, za, sa = xa + gap, za + gap, sa - 2 * gap
        vx = np.stack([xa, xa + sa, xa + sa, xa], 1).ravel().astype(np.float32)
        vz = np.stack([za, za, za + sa, za + sa], 1).ravel().astype(np.float32)
        base = (4 * np.arange(n)).astype(np.int32)
        return go.Mesh3d(x=vx, y=np.full(4 * n, yy, np.float32), z=vz,
                         i=np.concatenate([base, base]), j=np.concatenate([base + 1, base + 2]),
                         k=np.concatenate([base + 2, base + 3]), color=color, name=name,
                         showlegend=legend, hoverinfo="name", flatshading=True,
                         lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0))

    for k, sz in enumerate(sizes):
        m = (lab == AIR) & (s == sz)
        traces.append(quads(x0[m], z0[m], s[m], palette[k % len(palette)], f"air, {sz:g} m cells"))
    for L in np.unique(lab[lab != AIR]):
        m = lab == L
        name, color = STYLE.get(int(L), (f"label {L}", "#cc4444"))
        traces.append(quads(x0[m], z0[m], s[m], color, name))
    fig = go.Figure(traces)
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), c, tree.origin),
                  title or f"Adaptive cells on a vertical cut at y = {yy:.0f} m", height)
    fig.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-0.85, z=0.05)))
    return fig


def _cut_row(tree, y):
    """Voxel row of the vertical cut: y metres, or the row crossing most buildings."""
    if y is not None:
        depth = tree.shape[1] * tree.voxel_size
        if not 0 <= y < depth:
            raise ValueError(f"y={y:g} is outside the block: give metres from the tree's corner, 0 <= y < {depth:g}"
                             + _not_absolute(tree))
        return int(y / tree.voxel_size)
    X, Y, Z = tree.corners()
    S = tree.sizes()
    b = tree.label == BUILDING
    if not b.any():
        return tree.shape[1] // 2                    # no buildings: the middle row
    score = np.bincount(np.repeat(Y[b], S[b]) + np.concatenate([np.arange(s) for s in S[b]]),
                        minlength=tree.shape[1])
    return int(np.argmax(score))


def _not_absolute(tree):
    ox, oy = tree.origin[0], tree.origin[1]
    return f" (not absolute coordinates: the origin {ox:g}, {oy:g} is not included)" if (ox or oy) else ""


def _cut_corner(tree, cut):
    """Voxel column (cx, cy) of the cutaway corner: cut=(x, y) metres from the tree's corner."""
    nx, ny, _ = tree.shape
    if cut is None:
        return nx // 2, ny // 2
    c = tree.voxel_size
    x, y = (float(v) for v in cut)
    if not (0 < x < nx * c and 0 < y < ny * c):
        raise ValueError(f"cut=({x:g}, {y:g}) is outside the block: give metres from the tree's corner, "
                         f"0 < x < {nx * c:g} and 0 < y < {ny * c:g}" + _not_absolute(tree))
    return min(max(int(round(x / c)), 1), nx - 1), min(max(int(round(y / c)), 1), ny - 1)


def _solid_view(lab, labels, c, z0):
    """An Occupancy of just these labels (any other label empty), for the
    building meshes around a view; safe for label ids above 254."""
    keep = np.where(np.isin(lab, labels), lab, AIR).astype(np.uint8)
    return Occupancy.from_dense(keep, c, (0.0, 0.0, z0))


def _compact(v, lo, hi):
    """Float32 when it keeps every value to 1e-4 of the colour range (half
    the size), else float64 (e.g. epoch seconds)."""
    v = np.asarray(v, np.float64)
    v32 = v.astype(np.float32)
    ok = np.isfinite(v)
    if not ok.any() or np.abs(v32[ok].astype(np.float64) - v[ok]).max() <= (hi - lo) * 1e-4:
        return v32
    return v


def _values_only_on_solids(tree, col, where):
    """Warn when no cell of the view has a value but other cells do."""
    if not col.missing[where].all() or col.missing.all():
        return
    import warnings

    warnings.warn(f"no air cell has a {col.name!r} value; the values are on other cells (solids): "
                  "use style='cutaway', or cells= to pick them, e.g. "
                  "show_octree(tree, cells=tree.label != 0, color=...)", UserWarning, stacklevel=3)


def _show_cells_by_value(tree, y, title, height, max_cells, color, colorscale, cmin, cmax, nan):
    import plotly.graph_objects as go

    from ._colors import check_nan

    nan = check_nan(nan)
    c = tree.voxel_size
    col = _coloring(tree, color, None, colorscale, cmin, cmax)
    X, Y, Z = tree.corners()
    S = tree.sizes()
    yv = _cut_row(tree, y)
    yy = (yv + 0.5) * c
    cut = np.nonzero((Y <= yv) & (yv < Y + S))[0]
    shown = cut if nan == "grey" else cut[~((tree.label[cut] == AIR) & col.missing[cut])]
    ix = shown
    if ix.size > max_cells:                          # biggest cells first, as without color=
        ix = np.sort(ix[np.argsort(-S[ix], kind="stable")[:max_cells]])
    q = _plane_quads(tree, ix, "y", yy, gap=0.08 * c)
    traces = _cut_by_value(tree, q, ix, col, nan, full=cut)
    if ix.size < shown.size:
        traces.append(_hidden_note(f"{ix.size:,} of {shown.size:,} cells drawn (the biggest)"))
    fig = go.Figure(traces)
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), c, tree.origin),
                  title or f"{col.name} on a vertical cut at y = {yy:.0f} m", height)
    fig.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-0.85, z=0.05)))
    return _color_scale(fig, col)


# ------------------------------------------------------- colour by value

FG = "#e6e9ee"
AIR_COLOR = "#a8c8e8"
_FLAT = dict(ambient=1.0, diffuse=0.0, specular=0.0)


def _coloring(tree, color, idx=None, colorscale=None, cmin=None, cmax=None, group=None):
    from ._colors import resolve

    style = dict(STYLE)
    style[AIR] = ("air", AIR_COLOR)
    return resolve(tree, color, idx, colorscale or "Viridis", cmin, cmax, label_style=style, group=group)


def _color_scale(fig, col):
    """The colour bar of a numeric colouring, shared by every trace (coloraxis)."""
    if col is not None and col.numeric:
        fig.update_layout(coloraxis=dict(
            colorscale=col.colorscale, cmin=col.cmin, cmax=col.cmax,
            showscale=not col.empty,                   # no value anywhere: no colour bar to read
            colorbar=dict(title=dict(text=col.name, font=dict(color=FG)), tickfont=dict(color=FG),
                          len=0.6, x=0.97)))
    return fig


def _count(n, drawn=None):
    text = f"{n:,} cell" + ("" if n == 1 else "s")
    return text + (f", {drawn:,} drawn" if drawn is not None and drawn < n else "")


def _key(name, color):
    """A legend entry only: a coloured square."""
    import plotly.graph_objects as go

    return go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=name, hoverinfo="skip",
                        marker=dict(color=color, size=8, symbol="square"))


def _category_keys(col, full, drawn=None):
    """Legend entries for the categories among leaves `full`, with counts."""
    from ._colors import LEGEND_MAX

    counts = col.counts(full)
    dc = counts if drawn is None else col.counts(drawn)
    present = np.nonzero(counts)[0]
    if present.size > LEGEND_MAX:
        return [_hidden_note(f"{present.size:,} different {col.name} values (colours repeat)")]
    return [_key(f"{col.names[k]} ({_count(int(counts[k]), int(dc[k]))})", col.colors[k]) for k in present]


def _missing_key(col, full, drawn, nan):
    from ._colors import MISSING

    n = int(col.missing[full].sum())
    if not n:
        return []
    if nan == "hide":
        return [_hidden_note(f"no value: {n:,} cells hidden")]
    return [_key(f"no value ({_count(n, int(col.missing[drawn].sum()))})", MISSING)]


def _wire_by_value(tree, full, drawn, col, nan):
    """Wire cubes of the leaves `drawn` (all of `full`, or thinned from it),
    coloured by col. Numbers: one trace per cell size on the colour scale;
    categories: one trace per category, with counts in the legend."""
    from ._colors import LEGEND_MAX, MISSING

    col.focus(full)
    seam = _SEAM * tree.voxel_size                    # each cube draws its own edges, in its own colour
    traces = []
    val = drawn[~col.missing[drawn]]
    if col.numeric:
        levels = sorted(np.unique(tree.level[val]))
        for k, L in enumerate(levels):
            m = val[tree.level[val] == L]
            n_all = int(((tree.level[full] == L) & ~col.missing[full]).sum())
            name = f"{(1 << int(L)) * tree.voxel_size:g} m cells" + (f" ({m.size:,} of {n_all:,} drawn)" if m.size < n_all else "")
            traces.append(_cube_paths(tree, m, seam, name, width=2.0 if k == 0 else 1.4,
                                      values=_compact(col.values[m], col.cmin, col.cmax)))
    else:
        counts, dc = col.counts(full), col.counts(drawn)
        present = np.nonzero(counts)[0]
        many = present.size > LEGEND_MAX
        bins = {}
        for k in present:                              # many categories: one trace per colour
            bins.setdefault(col.colors[k] if many else k, []).append(k)
        codes = col.codes[val]
        for key, ks in bins.items():
            m = val[np.isin(codes, ks)]
            name = str(key) if many else f"{col.names[key]} ({_count(int(counts[key]), int(dc[key]))})"
            traces.append(_cube_paths(tree, m, seam, name, color=col.colors[ks[0]], width=2.0, legend=not many))
        if many:
            traces.append(_hidden_note(f"{present.size:,} different {col.name} values (colours repeat)"))
    miss = drawn[col.missing[drawn]]
    if nan == "grey" and miss.size:
        traces.append(_cube_paths(tree, miss, seam, "no value", color=_rgba(MISSING, 0.55), width=1.0, legend=False))
    return traces + _missing_key(col, full, drawn, nan)


# one path along all 12 edges of a cube (3 edges twice): 16 corners, against 36 points for 12 segments
_CUBE_PATH = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]


def _cube_paths(tree, idx, inset, name, color=None, width=1.5, legend=True, values=None):
    """Wire cubes of leaves idx, each its own line (inset so that cubes
    never share an edge): every cube shows its 12 edges in its own colour.
    values: one number per cube, on the figure's colour scale."""
    import plotly.graph_objects as go

    c = tree.voxel_size
    X, Y, Z = (v[idx] for v in tree.corners())
    corner = np.stack([X, Y, Z], 1) * c + np.array([0, 0, tree.origin[2]]) + inset
    size = (tree.sizes()[idx] * c - 2 * inset)[:, None, None]
    pts = np.round(corner[:, None, :] + _CUBE[_CUBE_PATH][None] * size, 3)       # (n, 16, 3)
    n, k = len(idx), len(_CUBE_PATH) + 1
    xyz = [np.full((n, k), None, dtype=object) for _ in range(3)]               # None ends each cube
    for a in range(3):
        xyz[a][:, :-1] = pts[:, :, a]
    xs, ys, zs = (v.ravel().tolist() for v in xyz)
    line = dict(color=color, width=width)
    if values is not None:
        line = dict(color=np.repeat(np.asarray(values), k), coloraxis="coloraxis", width=width)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", connectgaps=False, name=name, line=line,
                        hoverinfo="skip", showlegend=legend)


def _quad_mesh(q, name, legend=True, share=False, quantum=1e-3, **kw):
    """Quads (n, 4, 3) in metres as one flat Mesh3d (two triangles each: the
    first n triangles are (0, 1, 2) of each quad, the next n (0, 2, 3)).
    share: quads share corners that coincide to `quantum` metres (a smaller
    figure); text per corner (hover) is then taken from the first quad."""
    import plotly.graph_objects as go

    n = len(q)
    v = np.asarray(q, np.float64).reshape(-1, 3)
    quad = np.arange(4 * n, dtype=np.int32).reshape(n, 4)
    if share and n:
        key = np.round((v - v.min(0)) / quantum).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        v = v[first]
        quad = inv.reshape(n, 4).astype(np.int32)
        if kw.get("text") is not None:
            kw["text"] = np.asarray(kw["text"])[first]
    v = v.astype(np.float32)
    kw.setdefault("lighting", _FLAT)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=np.concatenate([quad[:, 0], quad[:, 0]]),
                     j=np.concatenate([quad[:, 1], quad[:, 2]]), k=np.concatenate([quad[:, 2], quad[:, 3]]),
                     name=name, showlegend=legend, flatshading=True, **kw)


def _faces_by_value(q, cell, col, nan, name, opacity=1.0, lighting=_FLAT, legend=False, hover=True,
                    share=None, quantum=1e-3):
    """Quads q of leaves `cell` coloured by col, one colour per face; faces
    without a value grey or left out. hover: show the value under the mouse.
    share: share coinciding corners (default: only without hover; with
    hover only where no two cells meet at a corner, e.g. inset cubes)."""
    share = (not hover) if share is None else share
    from ._colors import MISSING

    miss = col.missing[cell]
    ok = ~miss
    out = []
    common = dict(opacity=opacity, lighting=lighting, legendgroup=name)
    if ok.any():
        if col.numeric:
            v = col.values[cell[ok]]
            paint = dict(intensity=_compact(np.concatenate([v, v]), col.cmin, col.cmax), intensitymode="cell",
                         coloraxis="coloraxis")
            text = _numbers(v) if hover else None
        else:
            fc = col.hex(cell[ok])
            paint = dict(facecolor=np.concatenate([fc, fc]))
            text = np.array(col.names, dtype=object)[col.codes[cell[ok]]] if hover else None
        if hover:
            paint.update(text=np.repeat(text, 4), hovertemplate=f"{col.name}: %{{text}}<extra></extra>")
        else:
            paint.update(hoverinfo="name")
        out.append(_quad_mesh(q[ok], name, legend=legend, share=share, quantum=quantum, **paint, **common))
    if nan == "grey" and miss.any():
        out.append(_quad_mesh(q[miss], name, legend=legend and not ok.any(), share=True, quantum=quantum,
                              color=MISSING,
                              hovertemplate=f"{col.name}: no value<extra></extra>", **common))
    return out


def _numbers(v):
    """Numbers as short text: the fewest significant digits (4 to 15) that
    keep different values apart (35.5, 3.333, 1700000000.25)."""
    u = np.unique(v[np.isfinite(v)])
    for d in (4, 6, 8, 10, 12, 15):
        if np.unique(np.char.mod(f"%.{d}g", u)).size == u.size:
            break
    return np.char.mod(f"%.{d}g", v)


def _plane_quads(tree, ix, axis, at, gap=0.0, upto=None):
    """Squares (n, 4, 3) where leaves ix cross the vertical plane x = at or
    y = at (metres, show_octree frame), each inset by `gap` (number or one
    per leaf) so the cell edges show; upto: cut off along the plane there
    (the corner of a cutaway)."""
    c = tree.voxel_size
    X, Y, Z = tree.corners()
    S = tree.sizes()
    u = (Y if axis == "x" else X)[ix] * c
    z = Z[ix] * c + tree.origin[2]
    s = S[ix] * c
    gap = np.asarray(gap, np.float64)
    u0, z, s = u + gap, z + gap, s - 2 * gap
    u1 = u0 + s if upto is None else np.minimum(u0 + s, upto)
    a = np.stack([u0, u1, u1, u0], 1)
    b = np.stack([z, z, z + s, z + s], 1)
    f = np.full_like(a, at)
    return np.stack([f, a, b], -1) if axis == "x" else np.stack([a, f, b], -1)


def _cut_by_value(tree, q, ix, col, nan, full=None):
    """Cut faces coloured by value. Solids without a value keep their own
    colour (buildings stay buildings); air without a value is grey or hidden.
    ix: the leaf of each square (a leaf on both walls of a cutaway has two);
    full: every leaf on the cut (before max_cells), for the legend counts."""
    lab = tree.label
    shown = np.unique(ix)
    full = shown if full is None else np.unique(full)
    col.focus(full)
    by_label = (lab[ix] != AIR) & col.missing[ix]
    v = ~by_label
    traces = _faces_by_value(q[v], ix[v], col, nan, "cut faces")
    names = _class_names(tree)
    for L in np.unique(lab[ix[by_label]]):
        m = by_label & (lab[ix] == L)
        name, color = STYLE.get(int(L), (f"label {L}", "#cc4444"))
        name = names.get(int(L), name)
        traces.append(_quad_mesh(q[m], name, color=color, hoverinfo="name", legendgroup=name))

    def valued(leaves):                               # the leaves drawn by value (not solids without one)
        return leaves[~((lab[leaves] != AIR) & col.missing[leaves])]

    f_val, s_val = valued(full), valued(shown)
    drawn = s_val if nan == "grey" else s_val[~col.missing[s_val]]
    keys = [] if col.numeric else _category_keys(col, f_val, drawn)
    return traces + keys + _missing_key(col, f_val[lab[f_val] == AIR], drawn[lab[drawn] == AIR], nan)


def _class_names(tree):
    """{label: name} from the tree's own classes, when they are not the urban ones."""
    from .urban import CLASSES

    if tree.classes is None or list(tree.classes.names) == list(CLASSES.names):
        return {}
    return dict(enumerate(tree.classes.names))


_CUBE_LIGHT = dict(ambient=0.7, diffuse=0.3, specular=0.0, roughness=1.0, fresnel=0.0)
_SEAM = 0.06          # cubes drawn by value are inset by this many voxels: each shows its own edges


def _cube_faces(tree, idx, drawn):
    """Faces of the cubes of leaves idx at their real size, in the
    show_octree frame, slightly inset so cell boundaries show as thin seams.
    A face is left out when a drawn leaf at least as big covers it (it could
    not be seen). drawn: bool per leaf. Returns quads (n, 4, 3), leaf per quad."""
    c = tree.voxel_size
    X, Y, Z = (v[idx] for v in tree.corners())
    s = tree.sizes()[idx]
    P = np.stack([X, Y, Z], 1)
    lv = tree.level[idx]
    gap = _SEAM * c
    quads, cells = [], []
    for axis in range(3):
        u, w = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            nb = P.copy()
            nb[:, axis] += s if sign > 0 else -1
            j = np.atleast_1d(tree.locate(nb[:, 0], nb[:, 1], nb[:, 2])) if len(nb) else np.zeros(0, np.int64)
            jj = np.maximum(j, 0)
            hidden = (j >= 0) & drawn[jj] & (tree.level[jj] >= lv)
            f = np.nonzero(~hidden)[0]
            if not f.size:
                continue
            lo = P[f].astype(np.float64) * c
            sz = s[f] * c
            lo[:, axis] += sz - gap if sign > 0 else gap
            corners = []
            for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = lo.copy()
                q[:, u] += gap + du * (sz - 2 * gap)
                q[:, w] += gap + dw * (sz - 2 * gap)
                corners.append(q)
            quads.append(np.stack(corners, 1))
            cells.append(idx[f])
    if not quads:
        return np.zeros((0, 4, 3)), np.zeros(0, np.int64)
    q = np.concatenate(quads) + np.array([0.0, 0.0, tree.origin[2]])
    return q, np.concatenate(cells)


def _cubes(tree, ix, name, color, opacity, legend=True):
    """Filled cubes of leaves ix in one colour (sizes, groups)."""
    drawn = np.zeros(tree.n_leaves, bool)
    drawn[ix] = True
    q, _ = _cube_faces(tree, np.asarray(ix, np.int64), drawn)
    return _quad_mesh(q, name, legend=legend, share=True, quantum=0.01 * tree.voxel_size, color=color,
                      opacity=opacity,
                      lighting=_CUBE_LIGHT, hoverinfo="name", legendgroup=name)


def _cubes_by_value(tree, full, drawn_ix, col, nan, opacity):
    """Filled cubes of leaves drawn_ix coloured by col, legend as for wire."""
    col.focus(full)
    drawn = np.zeros(tree.n_leaves, bool)
    drawn[drawn_ix] = True
    q, cell = _cube_faces(tree, np.asarray(drawn_ix, np.int64), drawn)
    shown = drawn_ix if nan == "grey" else drawn_ix[~col.missing[drawn_ix]]
    val = int((~col.missing[full]).sum())
    n_val = int((~col.missing[shown]).sum())
    name = f"cells by {col.name} ({_count(val, n_val)})"
    traces = _faces_by_value(q, cell, col, nan, name, opacity=opacity, lighting=_CUBE_LIGHT,
                             legend=col.numeric, share=True, quantum=0.01 * tree.voxel_size)
    keys = [] if col.numeric else _category_keys(col, full, shown)
    return traces + keys + _missing_key(col, full, shown, nan)

SIZE_COLORS = ["#f2a65a", "#e9cf6e", "#a6cf8a", "#5fb8a8", "#4f8fc4", "#6b6fc9", "#9b6bc9"]


def _level_grid(tree):
    """Dense grid of leaf levels (cell size class) and labels."""
    lev = np.empty(tree.shape, np.uint8)
    X, Y, Z = tree.corners()
    for L in np.unique(tree.level):
        s = 1 << int(L)
        m = tree.level == L
        view = lev.reshape(tree.shape[0] // s, s, tree.shape[1] // s, s, tree.shape[2] // s, s)
        view[X[m] // s, :, Y[m] // s, :, Z[m] // s, :] = L
    return lev, tree.to_dense()


GROUP_COLORS = ["#ff6b6b", "#4dabf7", "#ffd43b", "#69db7c", "#da77f2", "#ffa94d", "#3bc9db", "#f783ac"]
_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _rgba(color, alpha):
    rgb = color.lstrip("#")
    r, g, b = (int(rgb[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _cube_segments(tree, idx, inset=0.0):
    """The 12 edges of each leaf in idx, as (12 n, 2, 3) segments in the
    frame of show_octree: x, y from the tree's corner, z absolute (metres);
    inset: metres in from each face (neighbours then keep separate edges)."""
    c = tree.voxel_size
    X, Y, Z = (v[idx] for v in tree.corners())
    corner = np.stack([X, Y, Z], 1) * c + np.array([0, 0, tree.origin[2]]) + inset
    size = (tree.sizes()[idx] * c - 2 * inset)[:, None, None]
    v = corner[:, None, :] + _CUBE[None] * size
    return np.stack([v[:, [a, b]] for a, b in _PAIRS], 1).reshape(-1, 2, 3)


def _thin(tree, idx, max_cells):
    """Leave out the smallest cell sizes until at most max_cells remain.
    Returns (kept indices, [(size_m, how many hidden), ...])."""
    idx = np.asarray(idx, np.int64)
    hidden = []
    for L in sorted(np.unique(tree.level[idx])):
        if idx.size <= max_cells:
            break
        m = tree.level[idx] == L
        hidden.append(((1 << int(L)) * tree.voxel_size, int(m.sum())))
        idx = idx[~m]
    return idx, hidden


def _thin_evenly(tree, idx, max_cells):
    """At most max_cells leaves, every size kept: the biggest sizes are
    drawn in full, the most numerous (smallest) are thinned to evenly spaced
    cells in Morton order, which keeps them spread over the whole selection.
    Returns (kept indices, [(size_m, drawn, of how many), ...])."""
    idx = np.asarray(idx, np.int64)
    if idx.size <= max_cells:
        return idx, []
    keep, notes, budget = [], [], int(max_cells)
    levels = sorted(np.unique(tree.level[idx]), reverse=True)
    for k, L in enumerate(levels):
        m = idx[tree.level[idx] == L]
        share = budget // (len(levels) - k)                 # what is left, shared by the sizes left
        if m.size <= share:
            keep.append(m)
            budget -= m.size
            continue
        pick = np.unique(np.linspace(0, m.size - 1, max(share, 1)).round().astype(np.int64))
        t = m[pick]                                         # evenly spread, the whole share used
        keep.append(t)
        budget -= t.size
        notes.append(((1 << int(L)) * tree.voxel_size, int(t.size), int(m.size)))
    return np.sort(np.concatenate(keep)), notes


def _hidden_note(name):
    import plotly.graph_objects as go

    return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", name=name,
                        line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip")


def _leaf_index(tree, cells):
    """A Group, a bool mask over the leaves or leaf indices -> sorted indices."""
    if hasattr(cells, "idx"):
        if getattr(cells, "tree", tree) is not tree:
            raise ValueError("this group belongs to another tree (selected before adapt, a rebuild or a "
                             "load?): select it again on this tree")
        return np.asarray(cells.idx, np.int64)
    a = np.asarray(cells)
    if a.dtype == bool:
        if a.shape != (tree.n_leaves,):
            raise ValueError(f"a mask needs one value per leaf ({tree.n_leaves})")
        return np.nonzero(a)[0]
    return np.unique(a.astype(np.int64))


def _in_leaf_order(tree, cells, color):
    """Values given for a list of leaf indices in the caller's order, put in
    leaf order (the order of the selection's cells)."""
    if color is None or isinstance(color, str) or isinstance(cells, dict) or hasattr(cells, "idx"):
        return color
    leaves = np.asarray(cells)
    values = np.asarray(color)
    if leaves.dtype == bool or values.ndim != 1 or values.size != leaves.size or values.size == tree.n_leaves:
        return color
    if np.unique(leaves).size != leaves.size:
        raise ValueError("the leaf indices repeat: give one value per leaf, or each leaf once")
    return values[np.argsort(leaves, kind="stable")]


def _context(tree, labels, opacity):
    """The tree's solids with these labels, drawn lightly around a selection."""
    if not labels:
        return []
    cols = _solid_view(tree.to_dense(), list(labels), tree.voxel_size, tree.origin[2])
    present = [int(v) for v in labels if (cols.label == v).any()]
    traces = _meshes(cols, present) if present else []
    for t in traces:
        t.opacity = opacity
        t.name = f"{t.name} (context)"
    return traces


def show_selection(tree, cells, title=None, height=700, max_cells=None, context=(BUILDING, TREE),
                   context_opacity=0.25, color=None, colorscale=None, cmin=None, cmax=None, nan="grey",
                   style="wire", opacity=1.0):
    """Any selection of leaves drawn as cubes at their real size, in the
    frame of show_octree, with the tree's buildings and trees drawn lightly
    around for context.

    style="wire": the cubes' edges (see-through); style="cubes": filled
        cubes, each cell one cube with thin seams between cells (opacity
        below 1 for see-through cubes). Faces between two drawn cells are
        left out, so a layer of cells (e.g. along the walls) reads as a skin.

    cells: one selection (a Group from tree.select, neighbours, within,
        siblings ..., a bool mask over the leaves, or leaf indices): cubes
        coloured by cell size, as in show_octree(style="wire");
        or {name: selection}: one colour per name, with a legend.
    color: colour one selection by a value (see show_octree): a tree.cells
        column, a built-in column, a value of the group ("steps",
        "distance"), or an array with one value per leaf or per selected
        cell (in leaf order, as g.idx). The colour scale spans the
        selection's values unless cmin / cmax are given.
    max_cells: at most this many cubes (default 15,000 wire or 25,000
        filled cubes: a figure of up to about 10 MB), so the view stays
        responsive. Past it every size is still drawn: the most numerous
        sizes are thinned evenly (every k-th cell in Morton order, spread
        over the whole selection), and the legend says how many are drawn.
    context: labels drawn around the selection (() for none).
    """
    import plotly.graph_objects as go

    from ._colors import check_nan

    nan = check_nan(nan)
    if style not in ("wire", "cubes"):
        raise ValueError(f"style={style!r}: a selection is drawn as 'wire' or 'cubes'")
    cubes = style == "cubes"
    max_cells = (25_000 if cubes else 15_000) if max_cells is None else max_cells
    named = isinstance(cells, dict)
    if named and color is not None:
        raise ValueError("color= colours one selection by a value; with {name: group} each group has its "
                         "own colour. Join the groups first (g1 | g2) to colour them by a value.")
    color = _in_leaf_order(tree, cells, color)
    groups = cells if named else {"cells": cells}
    idx = {k: _leaf_index(tree, v) for k, v in groups.items()}
    every = np.unique(np.concatenate([v for v in idx.values()] or [np.zeros(0, np.int64)]))
    keep, thinned = _thin_evenly(tree, every, max_cells)
    traces = []
    col = None
    if color is not None:
        group = cells if hasattr(cells, "values") and hasattr(cells, "idx") else None
        col = _coloring(tree, color, every, colorscale, cmin, cmax, group=group)
        pool = every if nan == "grey" else every[~col.missing[every]]     # the cells that can be drawn
        keep, _ = _thin_evenly(tree, pool, max_cells)
        traces = (_cubes_by_value(tree, every, keep, col, nan, opacity) if cubes
                  else _wire_by_value(tree, every, keep, col, nan))
        if keep.size < pool.size:
            traces.append(_hidden_note(f"{keep.size:,} of {pool.size:,} cells drawn (evenly spread)"))
        thinned = []
    elif named:
        for k, (name, ix) in enumerate(idx.items()):
            shown = np.intersect1d(ix, keep)
            gc = GROUP_COLORS[k % len(GROUP_COLORS)]
            label = f"{name} ({ix.size:,} cells" + (f", {shown.size:,} drawn)" if shown.size < ix.size else ")")
            if cubes:
                traces.append(_cubes(tree, shown, label, gc, opacity))
            else:
                traces.append(_edge_trace(_cube_segments(tree, shown), label, color=_rgba(gc, 0.9), width=2.0))
            traces[-1].legendgroup = name
        for size, drawn, n in thinned:
            traces.append(_hidden_note(f"{size:g} m cells: {drawn:,} of {n:,} drawn (evenly spread)"))
    else:
        # one legend entry per size; a thinned size says so in its own entry (no second "1 m cells")
        notes = {f"{size:g}": (drawn, n) for size, drawn, n in thinned}
        levels = sorted(np.unique(tree.level[keep]))
        for k, L in enumerate(levels):
            m = keep[tree.level[keep] == L]
            size = f"{(1 << int(L)) * tree.voxel_size:g}"
            name = f"{size} m cells" + (" ({:,} of {:,} drawn, evenly spread)".format(*notes[size]) if size in notes else "")
            sc = SIZE_COLORS[int(L) % len(SIZE_COLORS)]
            if cubes:
                traces.append(_cubes(tree, m, name, sc, opacity))
                continue
            alpha = 0.95 - 0.6 * k / max(1, len(levels) - 1)       # small cells stand out
            traces.append(_edge_trace(_cube_segments(tree, m), name, color=_rgba(sc, alpha),
                                      width=2.0 if k == 0 else 1.4))
    traces = _context(tree, context, context_opacity) + traces
    fig = go.Figure(traces)
    default = f"{every.size:,} selected cells" if every.size else "No cells selected"
    if col is not None and every.size:
        default += f", coloured by {col.name}"
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), tree.voxel_size, tree.origin), title or default, height)
    fig.update_layout(scene_camera=dict(eye=dict(x=-1.45, y=-1.65, z=0.75)))
    return _color_scale(fig, col)


def show_octree(tree, style=None, title=None, height=700, cut=None, max_cells=None, cells=None,
                color=None, colorscale=None, cmin=None, cmax=None, nan="grey", max_faces=200_000,
                opacity=1.0):
    """3D view of an adaptive octree.

    style="shells" (the default): nested see-through envelopes; shell k
        holds every air cell up to 2**k cells wide, so you see where the
        mesh is fine in 3D. Click legend entries to peel shells away.
    style="cutaway": the octree as a block with one corner removed; the two
        cut faces show the cells (coloured by size), buildings stand in the
        rest. cut=(x, y) in metres sets the corner (default: block centre).
    style="wire": every air cell drawn as a wire cube coloured by size,
        around solid buildings. If there are more than max_cells (40,000) air cells,
        the smallest sizes are hidden (and listed in the legend) so the view
        stays responsive; crop a smaller block to see every cell.

    cells: draw only a selection (see show_selection): a Group, a mask or
        leaf indices (coloured by size), or {name: group} (one colour per
        name), as wire cubes (style "wire", the default) or filled cubes
        (style="cubes", opacity= for see-through cubes); other styles give
        a warning. max_cells thins the selection evenly instead of hiding
        the smallest sizes.

    color: colour the cells by a value instead of by size, in every style:
        a tree.cells column name, a built-in column ("size_m", "z",
        "level", "label", "class", ...) or an array with one value per leaf.
        wire: the air cubes; cutaway: every cell on the cut faces (solids
        without a value keep their own colour); shells: each shell's outer
        faces, by the cell inside them. Buildings stand around as usual.
        - Numbers: a colour scale with a colour bar (colorscale="Viridis",
          or any plotly scale; cmin / cmax, default: the range over the
          whole tree, so thinning or the cut never changes the colours).
        - Categories (text, bools, label / class, object ids in
          tree.id_columns or named "id" / "..._id", integers with at most
          12 different values): one distinct colour each, with a legend;
          ids stay exact integers. Use .astype(str) or .astype(float) on an
          array to choose the other way.
        - nan="grey" draws cells without a value (NaN, None, id -1) in grey;
          nan="hide" leaves them out. The legend counts them.
        With color=, a wire view past max_cells (15,000: about 7 MB) is
        thinned evenly (every size kept, evenly spaced cells) and the legend
        says how many are drawn. Shells by value draw one face per cell, so
        they are several times bigger than plain shells (a 160 m block:
        about 7 MB); past max_faces (200,000) they stop with a message: crop
        a smaller block.
    """
    import plotly.graph_objects as go

    from .exports import surface_quads

    if cells is not None:
        if style not in (None, "wire", "cubes"):
            import warnings

            warnings.warn(f"style={style!r} is not used with cells=: a selection is drawn as wire cubes "
                          "(or style='cubes'; see show_selection). Leave out cells= for a cutaway or "
                          "shells of the whole tree.", UserWarning, stacklevel=2)
        return show_selection(tree, cells, title=title, height=height, max_cells=max_cells, color=color,
                              colorscale=colorscale, cmin=cmin, cmax=cmax, nan=nan,
                              style="cubes" if style == "cubes" else "wire", opacity=opacity)
    style = "shells" if style is None else style
    if style == "cubes":
        raise ValueError("style='cubes' draws a selection of cells: pass cells= (for example the cells along "
                         "the walls, tree.select(...).neighbours()); for the whole tree use 'cutaway'")
    if style not in ("shells", "cutaway", "wire"):
        raise ValueError(f"style={style!r}: use 'shells', 'cutaway' or 'wire' (or 'cubes' with cells=)")
    if color is not None:
        return _octree_by_value(tree, style, title, height, cut, 15_000 if max_cells is None else max_cells,
                                color, colorscale, cmin, cmax, nan, max_faces)
    max_cells = 40_000 if max_cells is None else max_cells
    c = tree.voxel_size
    lev, lab = _level_grid(tree)
    solid = lab != AIR
    levels = [int(v) for v in np.unique(lev[~solid])]
    traces = []
    if style == "wire":
        # too many cubes to draw: hide the smallest sizes first and say so in the legend
        air, hidden = _thin(tree, np.nonzero(tree.label == AIR)[0], max_cells)
        levels_air = sorted(np.unique(tree.level[air]))
        for k, L in enumerate(levels_air):
            m = air[tree.level[air] == L]
            alpha = 0.95 - 0.6 * k / max(1, len(levels_air) - 1)       # big cells fade, small cells stand out
            traces.append(_edge_trace(_cube_segments(tree, m), f"{(1 << int(L)) * c:g} m cells",
                                      color=_rgba(SIZE_COLORS[int(L) % len(SIZE_COLORS)], alpha),
                                      width=2.0 if k == 0 else 1.4))
        for size, n in hidden:                       # legend note for sizes left out
            traces.append(_hidden_note(f"{size:g} m cells hidden ({n:,}, too many to draw)"))
        cols = _solid_view(lab, [BUILDING, TREE], c, tree.origin[2])
        traces = _meshes(cols, [BUILDING, TREE]) + traces
        default_title = "Octree cells as wire cubes: small near surfaces, large in open air"
    elif style == "shells":
        shell_colors = ["#ff8a3d", "#ffd23f", "#5fd3a8", "#4f9cf0", "#a77bf0"]
        shell_opac = [0.65, 0.32, 0.18, 0.1, 0.07]
        _one_size_warning(levels, c)
        for k, L in enumerate(levels[:-1]):
            region = solid | (lev <= L)
            cols = Occupancy.from_dense(region.astype(np.uint8), c, (0.0, 0.0, tree.origin[2]))
            q, _ = surface_quads(cols, labels=[1])
            if len(q) == 0:
                continue
            v, inv = np.unique(np.round(q.reshape(-1, 3), 3), axis=0, return_inverse=True)
            quad = inv.reshape(-1, 4).astype(np.int32)
            traces.append(go.Mesh3d(x=v[:, 0].astype(np.float32), y=v[:, 1].astype(np.float32),
                                    z=v[:, 2].astype(np.float32),
                                    i=np.concatenate([quad[:, 0], quad[:, 0]]),
                                    j=np.concatenate([quad[:, 1], quad[:, 2]]),
                                    k=np.concatenate([quad[:, 2], quad[:, 3]]),
                                    color=shell_colors[k % 5], opacity=shell_opac[k % 5],
                                    name=f"cells up to {(1 << L) * c:g} m", flatshading=True,
                                    showlegend=True, hoverinfo="name",
                                    lighting=dict(ambient=0.6, diffuse=0.6, specular=0.05)))
        cols = _solid_view(lab, [BUILDING, TREE], c, tree.origin[2])
        traces = _meshes(cols, [BUILDING, TREE]) + traces
        default_title = "Where the cells are fine: nested shells by cell size"
    else:
        cx, cy = _cut_corner(tree, cut)
        X, Y, Z = tree.corners()
        S = tree.sizes()
        gap = 0.08 * c

        def face(mask, plane, at):
            """Squares of the cells crossing the plane x=at or y=at, kept part only."""
            u = (Y if plane == "x" else X)[mask]
            z = Z[mask]
            s = S[mask]
            L = tree.label[mask]
            lv = tree.level[mask]
            out = []
            for key in np.unique(np.where(L == AIR, lv.astype(int), 100 + L.astype(int))):
                m = np.where(L == AIR, lv.astype(int), 100 + L.astype(int)) == key
                g = gap if key < 100 else 0.0          # gaps show air cell edges; solids stay solid
                ua, za, sa = u[m] * c + g, z[m] * c + g + tree.origin[2], s[m] * c - 2 * g
                n = int(m.sum())
                a = np.stack([ua, ua + sa, ua + sa, ua], 1).ravel().astype(np.float32)
                b = np.stack([za, za, za + sa, za + sa], 1).ravel().astype(np.float32)
                fixed = np.full(4 * n, at * c - 0.15 * c, np.float32)   # just in front of the cut walls
                xs, ys = (fixed, a) if plane == "x" else (a, fixed)
                base = (4 * np.arange(n)).astype(np.int32)
                if key >= 100:
                    name, color = STYLE.get(key - 100, (f"label {key - 100}", "#cc4444"))
                else:
                    name, color = f"{(1 << key) * c:g} m cells", SIZE_COLORS[key % len(SIZE_COLORS)]
                out.append(go.Mesh3d(x=xs, y=ys, z=b, i=np.concatenate([base, base]),
                                     j=np.concatenate([base + 1, base + 2]), k=np.concatenate([base + 2, base + 3]),
                                     color=color, name=name, legendgroup=name, hoverinfo="name",
                                     flatshading=True, lighting=dict(ambient=1.0, diffuse=0.0)))
            return out

        # the removed corner is x < cx, y < cy; its two walls are the cut faces:
        # plane x = cx for y < cy, and plane y = cy for x < cx
        m1 = (X <= cx) & (cx < X + S) & (Y < cy)
        m2 = (Y <= cy) & (cy < Y + S) & (X < cx)
        f1 = face(m1, "x", cx)
        f2 = face(m2, "y", cy)
        # buildings outside the removed corner
        keep = lab.copy()
        keep[:cx, :cy] = AIR
        cols = _solid_view(keep, [BUILDING, TREE], c, tree.origin[2])
        traces = _meshes(cols, [BUILDING, TREE]) + f1 + f2
        seen = set()
        for t in traces:
            t.legendgroup = t.name
            t.showlegend = t.name not in seen
            seen.add(t.name)
        default_title = "Octree cut-away: cells on two cut faces, coloured by size"
    fig = go.Figure(traces)
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), c, tree.origin), title or default_title, height)
    if style == "cutaway":
        fig.update_layout(scene_camera=dict(eye=dict(x=-1.1, y=-1.1, z=0.7)))
    if style == "wire":
        fig.update_layout(scene_camera=dict(eye=dict(x=-1.45, y=-1.65, z=0.75)))
    return fig


def _octree_by_value(tree, style, title, height, cut, max_cells, color, colorscale, cmin, cmax, nan, max_faces):
    import plotly.graph_objects as go

    from ._colors import check_nan

    nan = check_nan(nan)
    c = tree.voxel_size
    col = _coloring(tree, color, None, colorscale, cmin, cmax)
    lev, lab = _level_grid(tree)
    keep_lab = lab
    if style == "wire":
        air = np.nonzero(tree.label == AIR)[0]
        _values_only_on_solids(tree, col, air)
        pool = air if nan == "grey" else air[~col.missing[air]]          # the cells that can be drawn
        drawn, _ = _thin_evenly(tree, pool, max_cells)
        traces = _wire_by_value(tree, air, drawn, col, nan)
        if drawn.size < pool.size:
            traces.append(_hidden_note(f"{drawn.size:,} of {pool.size:,} air cells drawn (evenly spread)"))
        default_title = f"{col.name} on the air cells (wire cubes)"
        camera = dict(eye=dict(x=-1.45, y=-1.65, z=0.75))
    elif style == "shells":
        shell_opac = [0.65, 0.32, 0.18, 0.1, 0.07]
        levels = [int(v) for v in np.unique(lev[lab == AIR])]
        _one_size_warning(levels, c)
        _values_only_on_solids(tree, col, np.nonzero(tree.label == AIR)[0])
        traces = []
        shells = [_shell_quads(tree, lev, lab, L) for L in levels[:-1]]
        n_faces = sum(len(q) for q, _ in shells)
        if n_faces > max_faces:
            raise ValueError(f"{n_faces:,} shell faces: crop a smaller block (city.crop) or raise max_faces, "
                             "or use style='cutaway' or 'wire'")
        seen = [cell for _, cell in shells]
        seen = np.unique(np.concatenate(seen)) if seen else np.zeros(0, np.int64)
        col.focus(seen)
        for k, (L, (q, cell)) in enumerate(zip(levels[:-1], shells)):
            if len(q) == 0:
                continue
            traces += _faces_by_value(q, cell, col, nan, f"cells up to {(1 << L) * c:g} m",
                                      opacity=shell_opac[k % 5], legend=True, hover=False,
                                      lighting=dict(ambient=0.6, diffuse=0.6, specular=0.05))
        drawn = seen if nan == "grey" else seen[~col.missing[seen]]
        traces += ([] if col.numeric else _category_keys(col, seen)) + _missing_key(col, seen, drawn, nan)
        default_title = f"{col.name} on the shells of cells by size"
        camera = None
    else:
        cx, cy = _cut_corner(tree, cut)
        X, Y, Z = tree.corners()
        S = tree.sizes()
        # the removed corner is x < cx, y < cy; its walls are the cut faces, each ending at the corner
        i1 = np.nonzero((X <= cx) & (cx < X + S) & (Y < cy))[0]
        i2 = np.nonzero((Y <= cy) & (cy < Y + S) & (X < cx))[0]
        gap = lambda ix: np.where(tree.label[ix] == AIR, 0.08 * c, 0.0)      # noqa: E731
        q = np.concatenate([_plane_quads(tree, i1, "x", cx * c - 0.15 * c, gap(i1), upto=cy * c),
                            _plane_quads(tree, i2, "y", cy * c - 0.15 * c, gap(i2), upto=cx * c)])
        traces = _cut_by_value(tree, q, np.concatenate([i1, i2]), col, nan)
        keep_lab = lab.copy()
        keep_lab[:cx, :cy] = AIR
        default_title = f"{col.name} on two cut faces"
        camera = dict(eye=dict(x=-1.1, y=-1.1, z=0.7))
    cols = _solid_view(keep_lab, [BUILDING, TREE], c, tree.origin[2])
    traces = _meshes(cols, [BUILDING, TREE]) + traces
    seen_names = set()
    for t in traces:                                   # one legend entry per name
        if t.showlegend is not False and t.name in seen_names:
            t.showlegend = False
        seen_names.add(t.name)
    fig = go.Figure(traces)
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), c, tree.origin), title or default_title, height)
    if camera:
        fig.update_layout(scene_camera=camera)
    return _color_scale(fig, col)


def _one_size_warning(levels, c):
    if len(levels) < 2:
        import warnings

        size = f"{(1 << levels[0]) * c:g} m" if levels else "none"
        warnings.warn(f"shells need air cells of at least two sizes; here they are all {size}: "
                      "use style='cutaway' or 'wire'", UserWarning, stacklevel=3)


def _shell_quads(tree, lev, lab, L):
    """Outer faces of the shell 'solids and air cells up to level L': the
    faces of those air leaves that touch a bigger air leaf or the side of
    the box (not the floor). A bigger neighbour covers the whole face, so
    each face is one square of its leaf. Returns quads (n, 4, 3) in the
    show_octree frame and the leaf of each."""
    c = tree.voxel_size
    X, Y, Z = tree.corners()
    S = tree.sizes()
    ix = np.nonzero((tree.label == AIR) & (tree.level <= L))[0]
    P = np.stack([X[ix], Y[ix], Z[ix]], 1)
    s = S[ix]
    quads, cells = [], []
    for axis in range(3):
        u, w = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            nb = P.copy()
            nb[:, axis] += s if sign > 0 else -1
            inside = (nb[:, axis] >= 0) & (nb[:, axis] < tree.shape[axis])
            face = ~inside if not (axis == 2 and sign < 0) else np.zeros(len(P), bool)
            n = nb[inside]
            face[inside] = (lab[n[:, 0], n[:, 1], n[:, 2]] == AIR) & (lev[n[:, 0], n[:, 1], n[:, 2]] > L)
            f = np.nonzero(face)[0]
            if not f.size:
                continue
            lo = P[f].astype(np.float64)
            if sign > 0:
                lo[:, axis] += s[f]
            corners = []
            for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = lo.copy()
                q[:, u] += du * s[f]
                q[:, w] += dw * s[f]
                corners.append(q)
            quads.append(np.stack(corners, 1))
            cells.append(ix[f])
    if not quads:
        return np.zeros((0, 4, 3)), np.zeros(0, np.int64)
    q = np.concatenate(quads) * c + np.array([0.0, 0.0, tree.origin[2]])
    return q, np.concatenate(cells)


def setup_notebook(fallback=None):
    """Make 3D views show up in any notebook front end. Call once at the top.

    In Google Colab it uses Colab's own renderer. Elsewhere every view is
    sent in two forms: plotly's own format (JupyterLab with the plotly
    plugin and VS Code draw it directly, offline) and a web-page
    fallback that loads plotly.js from the internet for any other front end.
    fallback=False sends only the first form (smaller saved notebooks).
    The environment variable OCTREECITY_LIGHT_NOTEBOOK=1 also turns it off.
    """
    import os

    import plotly.io as pio

    import sys

    if "google.colab" in sys.modules:              # Colab draws plotly with its own renderer
        pio.renderers.default = "colab"
        return pio.renderers.default
    if fallback is None:
        fallback = os.environ.get("OCTREECITY_LIGHT_NOTEBOOK") != "1"
    pio.renderers.default = "plotly_mimetype+notebook_connected" if fallback else "plotly_mimetype"
    return pio.renderers.default
