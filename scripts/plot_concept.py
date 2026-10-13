"""Concept figure: fixed-size voxels vs octree cells on the same vertical slice."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"
AIR_C, BLD_C, GND_C = "#1b2230", "#c9ccd3", "#8a7a62"
SIZE_C = {1: "#f2a65a", 2: "#e9cf6e", 4: "#a6cf8a", 8: "#5fb8a8", 16: "#4f8fc4", 32: "#6b6fc9"}

city = TiledCity(ROOT / "data/osaka_1m.ocz").region(41, 47, 42, 48)        # 256 m tile, downtown
tree = city.to_octree(5)
W, H = 256, 128                                                            # metres shown
dense = city.to_dense()
y = int(np.argmax((dense[:W, :, :H] == 2).sum(axis=(0, 2))))                # slice crossing most buildings
grid = dense[:W, y, :H]

fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# fixed voxels: every cell 1 m, drawn as a fine grid
ax = axes[0]
col = np.zeros(grid.T.shape + (3,))
col[:] = matplotlib.colors.to_rgb(AIR_C)
col[grid.T == 2] = matplotlib.colors.to_rgb(BLD_C)
col[grid.T == 1] = matplotlib.colors.to_rgb(GND_C)
ax.imshow(col, origin="lower", extent=(0, W, 0, H), interpolation="nearest")
lines = [[(x, 0), (x, H)] for x in range(0, W + 1)] + [[(0, z), (W, z)] for z in range(0, H + 1)]
ax.add_collection(LineCollection(lines, colors=(1, 1, 1, 0.13), linewidths=0.3))
ax.set_title(f"Fixed-size voxels: every cell is 1 m, in the air too.  {W * H:,} cells on this slice",
             color=FG, fontsize=11.5, loc="left")

# octree cells on the same slice
ax = axes[1]
X, Y, Z = tree.corners()
S = tree.sizes()
hit = (Y <= y) & (y < Y + S) & (X < W) & (Z < H)
x0, z0, s, lab = X[hit], Z[hit], S[hit], tree.label[hit]
quads = np.stack([np.stack([x0, z0], 1), np.stack([x0 + s, z0], 1), np.stack([x0 + s, z0 + s], 1),
                  np.stack([x0, z0 + s], 1)], 1).astype(float)
fc = [BLD_C if l == 2 else GND_C if l == 1 else SIZE_C.get(int(v), "#888") for l, v in zip(lab, s)]
alpha = np.where(lab == 0, 0.55, 1.0)
from matplotlib.colors import to_rgba
fcs = [to_rgba(c, a) for c, a in zip(fc, alpha)]
ax.add_collection(PolyCollection(quads, facecolors=fcs, edgecolors=(1, 1, 1, 0.35), linewidths=0.4))
ax.set_title(f"Octree: cells grow from 1 m next to surfaces to 32 m in open air.  {hit.sum():,} cells on this slice",
             color=FG, fontsize=11.5, loc="left")
for k, (sz, c) in enumerate(SIZE_C.items()):
    ax.add_patch(plt.Rectangle((0.01 + k * 0.075, -0.075), 0.018, 0.035, transform=ax.transAxes,
                               color=c, alpha=0.8, clip_on=False))
    ax.text(0.032 + k * 0.075, -0.07, f"{sz} m", transform=ax.transAxes, color=MUTED, fontsize=9)
ax.text(0.50, -0.07, "air cells coloured by size; buildings grey, ground brown", transform=ax.transAxes,
        color=MUTED, fontsize=9)
fig.text(0.01, 0.005, "Same 256 m vertical slice through downtown Osaka (PLATEAU, 1 m). Same detail at every "
         "wall and roof; the difference is the empty air.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = ROOT / "outputs/report/fixed_vs_octree.png"
fig.savefig(out, dpi=120, facecolor=BG)
n_vox = int(np.prod(tree.shape))
print(out, "| whole tile: voxels", n_vox, "octree cells", tree.n_leaves, "ratio", round(n_vox / tree.n_leaves, 1))
