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


def _edge_trace(segments, name, color="rgba(20,24,30,0.85)", width=1.5, legend=True):
    """Line segments (N, 2, 3) as one Scatter3d trace. Plain lists with None
    between segments: compact typed arrays would join all segments into one
    long line."""
    import plotly.graph_objects as go

    seg = np.round(np.asarray(segments, np.float64), 1)
    if len(seg):                                   # drop repeated edges (shared by neighbouring cubes or faces)
        a, b = seg[:, 0], seg[:, 1]
        lo = np.where((a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & ((a[:, 1] < b[:, 1]) | ((a[:, 1] == b[:, 1]) & (a[:, 2] <= b[:, 2])))), 0, 1)
        canon = np.where(lo[:, None, None] == 0, seg, seg[:, ::-1])
        _, keep = np.unique(canon.reshape(len(seg), 6), axis=0, return_index=True)
        seg = canon[np.sort(keep)]
    xs, ys, zs = [], [], []
    for (a, b) in seg.tolist():
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", connectgaps=False, name=name,
                        line=dict(color=color, width=width), hoverinfo="skip", showlegend=legend)


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


def show_cells(tree, y=None, title=None, height=650, max_cells=80_000):
    """Adaptive octree cells cut by a vertical plane, as filled squares:
    air cells coloured by size, solids in their label colour, thin edges.
    y: metres from the city edge; default: the row crossing most buildings."""
    import plotly.graph_objects as go

    c = tree.voxel_size
    X, Y, Z = tree.corners()
    S = tree.sizes()
    if y is None:
        b = tree.label == BUILDING
        score = np.bincount(np.repeat(Y[b], S[b]) + np.concatenate([np.arange(s) for s in S[b]]) if b.any() else [0],
                            minlength=tree.shape[1])
        yv = int(np.argmax(score))
    else:
        yv = int(y / c)
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


def _cube_segments(tree, idx):
    """The 12 edges of each leaf in idx, as (12 n, 2, 3) segments in the
    frame of show_octree: x, y from the tree's corner, z absolute (metres)."""
    c = tree.voxel_size
    X, Y, Z = (v[idx] for v in tree.corners())
    corner = np.stack([X, Y, Z], 1) * c + np.array([0, 0, tree.origin[2]])
    size = (tree.sizes()[idx] * c)[:, None, None]
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
    drawn in full, the most numerous (smallest) are thinned to every k-th
    cell in Morton order, which keeps them spread over the whole selection.
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
        step = -(-m.size // max(share, 1))
        t = m[::step]
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
        return np.asarray(cells.idx, np.int64)
    a = np.asarray(cells)
    if a.dtype == bool:
        if a.shape != (tree.n_leaves,):
            raise ValueError(f"a mask needs one value per leaf ({tree.n_leaves})")
        return np.nonzero(a)[0]
    return np.unique(a.astype(np.int64))


def _context(tree, labels, opacity):
    """The tree's solids with these labels, drawn lightly around a selection."""
    if not labels:
        return []
    lab = tree.to_dense()
    cols = Occupancy.from_dense(lab, tree.voxel_size, (0.0, 0.0, tree.origin[2]))
    present = [int(v) for v in labels if (cols.label == v).any()]
    traces = _meshes(cols, present) if present else []
    for t in traces:
        t.opacity = opacity
        t.name = f"{t.name} (context)"
    return traces


def show_selection(tree, cells, title=None, height=700, max_cells=15_000, context=(BUILDING, TREE),
                   context_opacity=0.25):
    """Any selection of leaves drawn as wire cubes at their real size, in the
    frame of show_octree, with the tree's buildings and trees drawn lightly
    around for context.

    cells: one selection (a Group from tree.select, neighbours, within,
        siblings ..., a bool mask over the leaves, or leaf indices): cubes
        coloured by cell size, as in show_octree(style="wire");
        or {name: selection}: one colour per name, with a legend.
    max_cells: at most this many cubes (15,000: a figure of about 9 MB), so
        the view stays responsive. Past it every size is still drawn: the most numerous sizes are thinned
        evenly (every k-th cell in Morton order, spread over the whole
        selection), and the legend says how many are drawn.
    context: labels drawn around the selection (() for none).
    """
    import plotly.graph_objects as go

    named = isinstance(cells, dict)
    groups = cells if named else {"cells": cells}
    idx = {k: _leaf_index(tree, v) for k, v in groups.items()}
    every = np.unique(np.concatenate([v for v in idx.values()] or [np.zeros(0, np.int64)]))
    keep, thinned = _thin_evenly(tree, every, max_cells)
    traces = []
    if named:
        for k, (name, ix) in enumerate(idx.items()):
            shown = np.intersect1d(ix, keep)
            color = GROUP_COLORS[k % len(GROUP_COLORS)]
            label = f"{name} ({ix.size:,} cells" + (f", {shown.size:,} drawn)" if shown.size < ix.size else ")")
            traces.append(_edge_trace(_cube_segments(tree, shown), label, color=_rgba(color, 0.9), width=2.0))
            traces[-1].legendgroup = name
    else:
        levels = sorted(np.unique(tree.level[keep]))
        for k, L in enumerate(levels):
            m = keep[tree.level[keep] == L]
            alpha = 0.95 - 0.6 * k / max(1, len(levels) - 1)       # small cells stand out
            traces.append(_edge_trace(_cube_segments(tree, m), f"{(1 << int(L)) * tree.voxel_size:g} m cells",
                                      color=_rgba(SIZE_COLORS[int(L) % len(SIZE_COLORS)], alpha),
                                      width=2.0 if k == 0 else 1.4))
    for size, drawn, n in thinned:
        traces.append(_hidden_note(f"{size:g} m cells: {drawn:,} of {n:,} drawn (evenly spread)"))
    traces = _context(tree, context, context_opacity) + traces
    fig = go.Figure(traces)
    fig = _layout(fig, Occupancy(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), tree.voxel_size, tree.origin),
                  title or (f"{every.size:,} selected cells" if every.size else "No cells selected"), height)
    fig.update_layout(scene_camera=dict(eye=dict(x=-1.45, y=-1.65, z=0.75)))
    return fig


def show_octree(tree, style="shells", title=None, height=700, cut=None, max_cells=None, cells=None):
    """3D view of an adaptive octree.

    cells: draw only a selection, as wire cubes (see show_selection): a
        Group, a mask or leaf indices (coloured by size), or {name: group}
        (one colour per name). The style is then "wire", and max_cells
        thins the selection evenly instead of hiding the smallest sizes.

    style="shells": nested see-through envelopes; shell k holds every air
        cell up to 2**k cells wide, so you see where the mesh is fine in 3D.
        Click legend entries to peel shells away.
    style="cutaway": the octree as a block with one corner removed; the two
        cut faces show the cells (coloured by size), buildings stand in the
        rest. cut=(x, y) in metres sets the corner (default: block centre).
    style="wire": every air cell drawn as a wire cube coloured by size,
        around solid buildings. If there are more than max_cells (40,000) air cells,
        the smallest sizes are hidden (and listed in the legend) so the view
        stays responsive; crop a smaller block to see every cell.
    """
    import plotly.graph_objects as go

    from .exports import surface_quads

    if cells is not None:
        return show_selection(tree, cells, title=title, height=height,
                              max_cells=15_000 if max_cells is None else max_cells)
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
        cols = Occupancy.from_dense(lab, c, (0.0, 0.0, tree.origin[2]))
        traces = _meshes(cols, [BUILDING, TREE]) + traces
        default_title = "Octree cells as wire cubes: small near surfaces, large in open air"
    elif style == "shells":
        shell_colors = ["#ff8a3d", "#ffd23f", "#5fd3a8", "#4f9cf0", "#a77bf0"]
        shell_opac = [0.65, 0.32, 0.18, 0.1, 0.07]
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
        cols = Occupancy.from_dense(lab, c, (0.0, 0.0, tree.origin[2]))
        traces = _meshes(cols, [BUILDING, TREE]) + traces
        default_title = "Where the cells are fine: nested shells by cell size"
    else:
        nx, ny, nz = tree.shape
        cx, cy = (nx // 2, ny // 2) if cut is None else (int(cut[0] / c), int(cut[1] / c))
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
        cols = Occupancy.from_dense(keep, c, (0.0, 0.0, tree.origin[2]))
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
