"""Surface vs exact rays on KAIST: air under tree canopies lets the sun through.

Usage: .venv/bin/python scripts/demo_voids.py
"""

import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from octreecity.analysis import street_points, sun_hours, surface_height
from octreecity.occupancy import Occupancy
from octreecity.urban import BUILDING, TREE

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"
full = Occupancy.load(ROOT / "data/kaist_1400m_1m.npz")
x0, x1, y0, y1 = 100, 600, 450, 950          # the big wood west of the campus centre
d = full.to_dense(x0, x1)[:, y0:y1]
city = Occupancy.from_dense(d, full.voxel_size,
                             (full.origin[0] + x0, full.origin[1] + y0, full.origin[2]))
EVERY = 2
pts, (ii, jj) = street_points(city, every=EVERY)
res, times = {}, {}
for mode in ("surface", "exact"):
    t = time.perf_counter()
    res[mode] = sun_hours(city, pts, 36.37, 127.36, date(2026, 6, 21), 9, step_min=30, mode=mode)
    times[mode] = time.perf_counter() - t
print({k: round(v, 1) for k, v in times.items()}, len(pts), "points")
treecol = np.isin(d, [TREE]).any(-1)
under = treecol[ii, jj]
print("mean sun hours under canopy: surface", res["surface"][under].mean().round(2),
      "exact", res["exact"][under].mean().round(2))

shape = (d.shape[0] // EVERY, d.shape[1] // EVERY)
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG,
                         gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
for ax, mode in zip(axes[:2], ("surface", "exact")):
    g = np.full(shape, np.nan)
    g[ii // EVERY, jj // EVERY] = res[mode]
    ax.set_facecolor(BG)
    bld = np.where(np.isin(d, [BUILDING]).any(-1), 1.0, np.nan)
    ax.imshow(bld.T, origin="lower", cmap="gray", vmin=0, vmax=1.6, extent=(0, 500, 0, 500))
    im = ax.imshow(g.T, origin="lower", cmap="inferno", vmin=0, vmax=14, extent=(0, 500, 0, 500),
                   interpolation="nearest")
    ax.contour(treecol.T.astype(float), levels=[0.5], colors=["#7bd389"], linewidths=0.8,
               extent=(0, 500, 0, 500))
    label = "canopy treated as solid to the ground" if mode == "surface" else "rays pass through air under canopies"
    ax.set_title(f"{mode} rays: {label}\n({times[mode]:.0f} s)", color=FG, fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
cb = fig.colorbar(im, ax=axes[:2], orientation="horizontal", fraction=0.04, pad=0.03)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.set_label("direct sun on 21 June (hours); green line = tree canopy edge", color=MUTED, fontsize=9)
ax = axes[2]
ax.set_facecolor(BG)
row = 250
sec = d[:, row, :]
from octreecity.render import COLORS, _rgb
img = np.zeros(sec.T.shape + (3,))
for k, c in COLORS.items():
    img[sec.T == k] = _rgb(c)
z0 = city.origin[2]
ax.imshow(img, origin="lower", extent=(0, 500, z0, z0 + d.shape[2]), aspect=3, interpolation="nearest")
ax.set_title(f"Section at y = {row} m: canopy (green) over air, terrain below", color=FG, fontsize=10, loc="left")
ax.tick_params(colors=MUTED, labelsize=8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_ylabel("elevation (m)", color=MUTED, fontsize=9)
out = ROOT / "outputs/report/kaist_voids_sun.png"
fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight")
print(out)
