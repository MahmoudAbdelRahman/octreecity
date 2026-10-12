"""Quick images of a ColumnCity: shaded top view and vertical sections."""

import numpy as np

from .builder import GRASS, PAVEMENT, ROAD, WATER
from .octree import AIR, BUILDING, GROUND, TREE

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
