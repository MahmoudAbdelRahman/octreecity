"""Overview of a tiled city: whole-city height map + a 1 m zoom from region().

Usage: .venv/bin/python scripts/plot_city.py data/osaka_1m.ocz [zoom_x zoom_y zoom_tiles]
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from octreecity.render import render_top, surface
from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"

path = Path(sys.argv[1])
city = TiledCity(path)
DS = 8
n = city.n // DS
tiles = city.tiles
I = max(i for i, _ in tiles) + 1
J = max(j for _, j in tiles) + 1
H = np.full((I * n, J * n), np.nan)
t0 = time.perf_counter()
for i, j in tiles:
    top, lab = surface(city.tile(i, j))
    h = (top - 1) * city.cell
    H[i * n:(i + 1) * n, j * n:(j + 1) * n] = h.reshape(n, DS, n, DS).max(axis=(1, 3))
t_read = time.perf_counter() - t0

fig = plt.figure(figsize=(15, 8.2), facecolor=BG)
ax = fig.add_axes([0.03, 0.12, 0.48, 0.81])
cmap = plt.get_cmap("magma").copy()
cmap.set_bad(BG)
im = ax.imshow(H.T, origin="lower", cmap=cmap, vmin=0, vmax=80, interpolation="nearest",
               extent=(0, I * city.tile_size / 1000, 0, J * city.tile_size / 1000))
ax.set_title(f"All of Osaka: {len(tiles):,} tiles of 256 m read one by one in {t_read:.0f} s "
             f"(max height per 8 m)", color=FG, fontsize=10.5, loc="left")
ax.tick_params(colors=MUTED, labelsize=8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("km east", color=MUTED, fontsize=9, labelpad=2)
cb = fig.add_axes([0.06, 0.03, 0.4, 0.014])
fig.colorbar(im, cax=cb, orientation="horizontal")
cb.tick_params(colors=MUTED, labelsize=8)
cb.set_xlabel("building height (m), capped at 80", color=MUTED, fontsize=8.5)

if len(sys.argv) > 4:
    zi, zj, zn = map(int, sys.argv[2:5])
else:
    # zoom on the densest block of tall buildings
    k = DS
    score = np.nan_to_num(H)
    best, zi, zj, zn = -1, 0, 0, 8
    for i in range(0, I - zn):
        for j in range(0, J - zn):
            s = score[i * n:(i + zn) * n, j * n:(j + zn) * n].mean()
            if s > best:
                best, zi, zj = s, i, j
t1 = time.perf_counter()
reg = city.region(zi, zj, zi + zn, zj + zn)
t_reg = time.perf_counter() - t1
ax.add_patch(matplotlib.patches.Rectangle((zi * city.tile_size / 1000, zj * city.tile_size / 1000),
                                          zn * city.tile_size / 1000, zn * city.tile_size / 1000,
                                          fill=False, edgecolor="#5aa9f2", lw=1.5))
ax2 = fig.add_axes([0.53, 0.07, 0.46, 0.86])
render_top(reg, ax2, legend=False)
km = zn * city.tile_size / 1000
ax2.set_title(f"Blue box at 1 m: {zn} x {zn} tiles ({km:g} x {km:g} km) stitched by region() in {t_reg:.1f} s",
              color=FG, fontsize=10.5, loc="left")
out = ROOT / "outputs/report" / f"{path.stem}_overview.png"
fig.savefig(out, dpi=110, facecolor=BG)
print(out, f"read {t_read:.0f}s, region {t_reg:.1f}s, zoom tiles {zi},{zj}")
