"""Interactive 3D views in a notebook (plotly): rotate, zoom, hover.

    from octreecity.viz import show, show_values, show_cells
    show(city)                                  # the city in 3D
    show_values(city, svf_map, "Sky view")      # street-level results on the 3D city
    show_cells(tree, y=300)                     # adaptive cells on a vertical cut

Big cities are simplified for display only (cells merged by `step`), so a
view stays quick; the data itself is untouched.
"""

import numpy as np

from .builder import GRASS, PAVEMENT, ROAD, WATER
from .columns import ColumnCity
from .octree import AIR, BUILDING, GROUND, TREE

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


def downsample(city, step):
    """Coarser ColumnCity for display: each step^3 block takes its most
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
    return ColumnCity.from_dense(out, city.voxel_size * step, city.origin)


def _auto_step(city, max_columns=150_000):
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


def show(city, labels=None, step=None, title=None, height=700):
    """Interactive 3D view of a ColumnCity (drag to rotate, scroll to zoom).

    labels: which labels to draw (default: every solid label).
    step: display simplification (merge step x step cells); chosen
    automatically for big cities.
    """
    import plotly.graph_objects as go

    step = _auto_step(city) if step is None else step
    view = downsample(city, step)
    labels = [int(v) for v in np.unique(view.label) if v != AIR] if labels is None else labels
    fig = go.Figure(_meshes(view, labels))
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
    fig = _layout(fig, ColumnCity(tree.shape, np.zeros(1, np.int64), np.zeros(0, np.uint16),
                                  np.zeros(0, np.uint8), c, tree.origin),
                  title or f"Adaptive cells on a vertical cut at y = {yy:.0f} m", height)
    fig.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-0.85, z=0.05)))
    return fig
