"""Quick images: shaded top views and sections of an Occupancy, slices of an octree."""

import numpy as np

from .urban import AIR, BUILDING, GRASS, GROUND, PAVEMENT, ROAD, TREE, WATER

COLORS = {GROUND: "#8a7a62", BUILDING: "#c9ccd3", TREE: "#3f8a4a", WATER: "#3d7fc4",
          GRASS: "#86b35a", PAVEMENT: "#b7ad9c", ROAD: "#5d6066", AIR: "#0f1318"}
NAMES = {GROUND: "ground", BUILDING: "building", TREE: "tree", WATER: "water",
         GRASS: "grass", PAVEMENT: "path", ROAD: "road"}


def _rgb(hex_):
    hex_ = hex_.lstrip("#")
    return np.array([int(hex_[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def surface(city):
    """Top of the highest non-air run (voxels) and its label, per column."""
    nonair = city.label != AIR
    cols = city.run_columns()
    top = np.zeros(city.offsets.size - 1, np.int64)
    lab = np.zeros(city.offsets.size - 1, np.uint8)
    idx = np.nonzero(nonair)[0]
    # runs are ordered bottom-up inside each column: the last non-air run wins
    top[cols[idx]] = city.top[idx]
    lab[cols[idx]] = city.label[idx]
    return top.reshape(city.shape[:2]), lab.reshape(city.shape[:2])


def render_top(city, ax, sun=(315, 40), legend=True):
    top, lab = surface(city)
    h = top * city.voxel_size
    gy, gx = np.gradient(h.T, city.voxel_size)
    az, alt = np.radians(sun[0]), np.radians(sun[1])
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(0.35 + 0.75 * shade, 0.2, 1.15)
    base = np.zeros(lab.T.shape + (3,))
    for k, c in COLORS.items():
        base[lab.T == k] = _rgb(c)
    # buildings brighten with height so tall ones read at a glance
    hb = np.clip((h.T - np.median(h)) / 80.0, 0, 1)[..., None]
    base = np.where((lab.T == BUILDING)[..., None], base * (0.8 + 0.35 * hb), base)
    img = np.clip(base * shade[..., None], 0, 1)
    nx, ny = city.shape[:2]
    c = city.voxel_size
    ax.imshow(img, origin="lower", extent=(0, nx * c, 0, ny * c), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if legend:
        present = [k for k in NAMES if (lab == k).any()]
        for n, k in enumerate(present):
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (0.01 + n * 0.12, -0.035), 0.018, 0.018, transform=ax.transAxes,
                color=COLORS[k], clip_on=False))
            ax.text(0.033 + n * 0.12, -0.033, NAMES[k], transform=ax.transAxes, color="#c8ced8",
                    fontsize=9, va="bottom")
        ax.plot([0.98 - 200 / (nx * c), 0.98], [0.02, 0.02], transform=ax.transAxes, color="white", lw=2)
        ax.text(0.98, 0.03, "200 m", transform=ax.transAxes, color="white", fontsize=9, ha="right")
    return img


def render_section(city, ax, y, x0=0, x1=None):
    """Vertical slice at column row y (voxel index), labels coloured."""
    x1 = city.shape[0] if x1 is None else x1
    g = city.to_dense(x0, x1)[:, y, :]
    img = np.zeros(g.T.shape + (3,))
    for k, c in COLORS.items():
        img[g.T == k] = _rgb(c)
    c = city.voxel_size
    ax.imshow(img, origin="lower", extent=(x0 * c, x1 * c, city.origin[2], city.origin[2] + city.shape[2] * c),
              interpolation="nearest", aspect="auto")
    return img


def plot_slice(tree, ax, axis="y", index=None, colors=None, edge=True, color=None, colorscale=None,
               cmin=None, cmax=None, nan="grey", legend=True):
    """Draw the leaves cut by the plane axis = index (voxel units).

    colors: {label: colour} for the cells (default: light air, darker
    buildings); with color=, {category: colour} for its categories.
    color: colour the cells by a value instead of by label, as
    viz.show_octree(color=...): a tree.cells column name, a built-in column
    ("size_m", "z", "class", ...) or an array with one value per leaf.
    Numbers get a colour bar (colorscale: a matplotlib or plotly name,
    default "viridis"; cmin / cmax, default: the range over the whole tree);
    categories get a legend (legend=False for none). Solids without a value
    keep their label colour; air without a value is grey (nan="grey") or
    left out (nan="hide").
    """
    from matplotlib.collections import PolyCollection

    k = "xyz".index(axis)
    if index is None:
        index = tree.shape[k] // 2
    c = tree.corners()
    s = tree.sizes()
    hit = (c[k] <= index) & (index < c[k] + s)
    ix = np.nonzero(hit)[0]
    base = dict(COLORS)
    base.update({0: "#f4f6fb", 2: "#5b6b8c"})            # light air, darker buildings on a light plot
    col = None
    if color is None:
        colors = base if colors is None else colors
        fc = np.array([colors.get(int(l), "#d33") for l in tree.label[ix]], dtype=object)
    else:
        from ._colors import MISSING, check_nan, resolve

        nan = check_nan(nan)
        style = {L: (NAMES.get(L, "air" if L == AIR else f"label {L}"), base.get(L)) for L in base}
        col = resolve(tree, color, None, colorscale or "viridis", cmin, cmax, colors=colors, label_style=style)
        miss = col.missing[ix]
        solid_miss = miss & (tree.label[ix] != AIR)
        if col.numeric:
            cmap, norm = _cmap(col.colorscale), _norm(col)
            rgb = np.round(cmap(norm(np.nan_to_num(col.values[ix])))[:, :3] * 255).astype(int)
            fc = np.array(["#%02x%02x%02x" % tuple(v) for v in rgb.tolist()], dtype=object)
        else:
            fc = col.hex(ix).astype(object)
        fc[miss] = MISSING
        fc[solid_miss] = [base.get(int(l), "#d33") for l in tree.label[ix[solid_miss]]]
        if nan == "hide":
            show = ~miss | solid_miss
            ix, fc = ix[show], fc[show]
    u, v = [c[i][ix] for i in range(3) if i != k]
    w = s[ix]
    quads = np.stack([np.stack([u, v], 1), np.stack([u + w, v], 1),
                      np.stack([u + w, v + w], 1), np.stack([u, v + w], 1)], 1)
    pc = PolyCollection(quads * tree.voxel_size, facecolors=list(fc),
                        edgecolors="#33333355" if edge else "none", linewidths=0.3)
    ax.add_collection(pc)
    dims = [tree.shape[i] * tree.voxel_size for i in range(3) if i != k]
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_aspect("equal")
    names = [n for n in "xyz" if n != axis]
    ax.set_xlabel(f"{names[0]} (m)")
    ax.set_ylabel(f"{names[1]} (m)")
    if col is not None and legend:
        _slice_key(ax, col, ix, nan, int(col.missing[ix].sum() - ((tree.label[ix] != AIR) & col.missing[ix]).sum()))
    return pc


def _cmap(name):
    """A matplotlib colormap from a matplotlib or plotly name ("Viridis", "RdBu_r")."""
    import matplotlib

    if not isinstance(name, str):
        return name
    for n in (name, name.lower()):
        try:
            return matplotlib.colormaps[n]
        except KeyError:
            continue
    raise ValueError(f"unknown colour scale {name!r}; use a matplotlib name such as 'viridis'")


def _norm(col):
    from matplotlib.colors import Normalize

    return Normalize(col.cmin, col.cmax)


def _slice_key(ax, col, ix, nan, n_missing_air):
    """Colour bar (numbers) or legend (categories) for plot_slice."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch

    from ._colors import LEGEND_MAX, MISSING

    if col.numeric:
        ax.figure.colorbar(ScalarMappable(_norm(col), _cmap(col.colorscale)), ax=ax, label=col.name, shrink=0.8)
        handles = []
    else:
        counts = col.counts(ix)
        present = np.nonzero(counts)[0]
        if present.size > LEGEND_MAX:
            handles = [Patch(color="none", label=f"{present.size:,} {col.name} values (colours repeat)")]
        else:
            handles = [Patch(color=col.colors[j], label=f"{col.names[j]} ({counts[j]:,})") for j in present]
    if n_missing_air and nan == "grey":
        handles.append(Patch(color=MISSING, label=f"no value ({n_missing_air:,})"))
    if handles:
        ax.legend(handles=handles, title=None if col.numeric else col.name, fontsize=8, loc="upper right")
