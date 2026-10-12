"""adapt() on a non-CFD field: 3D sun hours on 21 Dec in the air over Osaka.

Columns -> octree (min 1 m, max 32 m) -> sample sun hours at air-cell
centres -> adapt (split where neighbours differ > 1 h, merge where < 0.5 h)
-> repeat. Shows the octree following shadow edges in 3D.

Usage: .venv/bin/python scripts/demo_adapt_sun.py
"""

import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from octreecity.adapt import adapt, sample
from octreecity.analysis import sun_hours
from octreecity.octree import AIR, BUILDING, LinearOctree, pad_to_multiple
from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"
LAT, LON, TZ, DAY = 34.69, 135.50, 9, date(2026, 12, 21)
TOL = 1.0

tc = TiledCity(ROOT / "data/osaka_1m.ocz")
cols = tc.region(41, 47, 43, 49)                      # 512 x 512 m, downtown
L = 5
nz = pad_to_multiple(cols.shape[2], L)
from octreecity.columns import ColumnCity
if nz != cols.shape[2]:
    d = cols.to_dense()
    d = np.concatenate([d, np.zeros(d.shape[:2] + (nz - d.shape[2],), np.uint8)], 2)
    cols = ColumnCity.from_dense(d, cols.voxel_size, cols.origin)
tree = cols.to_octree(L)
print("geometry octree", tree.n_leaves, "leaves", tree.shape)


def field(x, y, z):
    pts = np.stack([x, y, z], 1)
    return sun_hours(cols, pts, LAT, LON, DAY, TZ, step_min=30)


def air_values(t):
    v = np.zeros(t.n_leaves)
    air = t.label == AIR
    sub = LinearOctree(t.shape, t.max_level, t.code[air], t.level[air], t.label[air], t.voxel_size, t.origin)
    v[air] = sample(sub, field)
    return v


snap = []
v = air_values(tree)
snap.append((tree, v))
for r in range(4):
    t0 = time.perf_counter()
    tree, _ = adapt(tree, v, refine_tol=TOL, min_level=0)
    v = air_values(tree)
    snap.append((tree, v))
    print(f"round {r + 1}: {tree.n_leaves:,} leaves ({(tree.label == AIR).sum():,} air), {time.perf_counter() - t0:.0f}s",
          flush=True)

y_cut = 300
fig, axes = plt.subplots(2, 1, figsize=(14, 8.4), facecolor=BG)
cmap = plt.get_cmap("inferno")
for ax, (t, vals), name in ((axes[0], snap[0], "start: octree from geometry only"),
                            (axes[1], snap[-1], f"after 4 adapt rounds on sun hours (tol {TOL:g} h)")):
    ax.set_facecolor(BG)
    x, y, z = t.corners()
    s = t.sizes()
    hit = (y <= y_cut) & (y_cut < y + s)
    u, w, ss = x[hit], z[hit], s[hit]
    quads = np.stack([np.stack([u, w], 1), np.stack([u + ss, w], 1), np.stack([u + ss, w + ss], 1),
                      np.stack([u, w + ss], 1)], 1).astype(float)
    lab = t.label[hit]
    fc = np.where((lab == AIR)[:, None], cmap(np.clip(vals[hit] / 9.5, 0, 1)),
                  np.array([[0.62, 0.64, 0.68, 1.0]]))
    fc[(lab != AIR) & (lab != BUILDING)] = [0.45, 0.4, 0.33, 1]
    ax.add_collection(PolyCollection(quads, facecolors=fc, edgecolors=(1, 1, 1, 0.25), linewidths=0.25))
    ax.set_xlim(0, t.shape[0])
    ax.set_ylim(0, min(t.shape[2], 200))
    ax.set_aspect("equal")
    n_air = int((t.label == AIR).sum())
    ax.set_title(f"{name}: {t.n_leaves:,} cells in 3D ({n_air:,} air)", color=FG, fontsize=10.5, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("height (m)", color=MUTED, fontsize=9)
axes[1].set_xlabel(f"x (m), vertical section at y = {y_cut} m; colour = hours of direct sun on 21 Dec, grey = buildings",
                   color=MUTED, fontsize=9)
fig.tight_layout()
out = ROOT / "outputs/report/adapt_sun_section.png"
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
