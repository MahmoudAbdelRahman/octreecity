"""Concept figure: fixed-size voxels vs octree cells on the same vertical slice (light theme)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb

from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#ffffff", "#1d2430", "#5b6472"
AIR_C, BLD_C, GND_C = "#eef1f6", "#8f96a3", "#a08a68"
SIZE_C = {1: "#f4a259", 2: "#f2d27a", 4: "#a8d08d", 8: "#7cc6b8", 16: "#8fb3e0", 32: "#b9b6ee"}

city = TiledCity(ROOT / "data/osaka_1m.ocz").region(41, 47, 42, 48)        # 256 m tile, downtown
tree = city.to_octree(5)
dense = city.to_dense()
W, H = 128, 96                                                             # window shown (m)
y = int(np.argmax((dense[:W, :, :H] == 2).sum(axis=(0, 2))))                # slice crossing most buildings
grid = dense[:W, y, :H]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_yticks([0, 32, 64, 96])
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    for sp in ax.spines.values():
        sp.set_color("#c9ced8")

# fixed voxels: every cell 1 m
ax = axes[0]
col = np.zeros(grid.T.shape + (3,))
col[:] = to_rgb(AIR_C)
col[grid.T == 2] = to_rgb(BLD_C)
col[grid.T == 1] = to_rgb(GND_C)
ax.imshow(col, origin="lower", extent=(0, W, 0, H), interpolation="nearest")
lines = [[(x, 0), (x, H)] for x in range(W + 1)] + [[(0, z), (W, z)] for z in range(H + 1)]
ax.add_collection(LineCollection(lines, colors=(0.15, 0.18, 0.24, 0.28), linewidths=0.35))
n_fixed = W * H
ax.set_title(f"Fixed-size voxels: every cell is 1 m, also in the air\n{n_fixed:,} cells in this window",
             color=FG, fontsize=11.5, loc="left")

# octree cells, clipped to the same window
ax = axes[1]
X, Y, Z = tree.corners()
S = tree.sizes()
hit = (Y <= y) & (y < Y + S) & (X < W) & (Z < H)
x0, z0, s, lab = X[hit], Z[hit], S[hit], tree.label[hit]
quads = np.stack([np.stack([x0, z0], 1), np.stack([x0 + s, z0], 1), np.stack([x0 + s, z0 + s], 1),
                  np.stack([x0, z0 + s], 1)], 1).astype(float)
fc = [BLD_C if l == 2 else GND_C if l == 1 else SIZE_C.get(int(v), "#bbb") for l, v in zip(lab, s)]
ax.add_collection(PolyCollection(quads, facecolors=fc, edgecolors=(0.15, 0.18, 0.24, 0.55), linewidths=0.45))
ax.set_title(f"Octree: 1 m cells next to surfaces, up to 32 m in open air\n{int(hit.sum()):,} cells in this window",
             color=FG, fontsize=11.5, loc="left")

handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#555", lw=0.4) for c in SIZE_C.values()]
handles += [plt.Rectangle((0, 0), 1, 1, color=BLD_C), plt.Rectangle((0, 0), 1, 1, color=GND_C)]
labels = [f"{k} m air cell" for k in SIZE_C] + ["building", "ground"]
fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, fontsize=9, labelcolor=FG,
           bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.075, f"Same vertical slice through downtown Osaka (PLATEAU, 1 m): {W} m wide, {H} m high. "
         "The detail at walls and roofs is identical; the difference is the empty air.",
         color=MUTED, fontsize=9, ha="center")
fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.17, wspace=0.08)
out = ROOT / "outputs/report/fixed_vs_octree.png"
fig.savefig(out, dpi=130, facecolor=BG)
n_vox = int(np.prod(tree.shape))
print(out, f"window: {n_fixed} vs {int(hit.sum())} cells | tile: {n_vox} vs {tree.n_leaves} ({n_vox / tree.n_leaves:.0f}x)")
