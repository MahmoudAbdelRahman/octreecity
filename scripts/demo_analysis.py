"""Sky view factor and sun hours at street level for a block of a tiled city.

Usage: .venv/bin/python scripts/demo_analysis.py data/osaka_1m.ocz i0 j0 n_tiles lat lon tz
"""

import sys
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from octreecity.analysis import sky_view_factor, street_points, sun_hours, surface_height
from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"

path = Path(sys.argv[1])
i0, j0, nt = map(int, sys.argv[2:5])
lat, lon, tz = map(float, sys.argv[5:8])
tc = TiledCity(path)
M = 1   # one tile of margin so buildings just outside the block still cast shadows
city = tc.region(i0 - M, j0 - M, i0 + nt + M, j0 + nt + M)
EVERY = 2
pts, (ii, jj) = street_points(city, every=EVERY)
lo, hi = M * tc.n, (M + nt) * tc.n
keep = (ii >= lo) & (ii < hi) & (jj >= lo) & (jj < hi)
pts, ii, jj = pts[keep], ii[keep] - lo, jj[keep] - lo
t0 = time.perf_counter()
svf = sky_view_factor(city, pts, n_az=36)
t_svf = time.perf_counter() - t0
t0 = time.perf_counter()
hours = sun_hours(city, pts, lat, lon, date(2026, 12, 21), tz, step_min=20)
t_sun = time.perf_counter() - t0
print(f"{len(pts):,} street points; SVF {t_svf:.1f}s, sun hours (Dec 21, every 20 min) {t_sun:.1f}s")

shape = (nt * tc.n // EVERY, nt * tc.n // EVERY)


def grid(vals):
    g = np.full(shape, np.nan)
    g[ii // EVERY, jj // EVERY] = vals
    return g


surf = (surface_height(city) - city.origin[2])[lo:hi, lo:hi]
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), facecolor=BG)
km = nt * tc.n * city.voxel_size
for ax, data, cmap, vmax, title, unit in (
        (axes[0], grid(svf), "cividis", 1.0, f"Sky view factor at 1.5 m ({t_svf:.0f} s)", "SVF"),
        (axes[1], grid(hours), "inferno", np.ceil(hours.max()), f"Direct sun on 21 Dec ({t_sun:.0f} s)", "hours")):
    ax.set_facecolor(BG)
    bld = np.where(surf > 3, 1.0, np.nan)
    ax.imshow(bld.T, origin="lower", cmap="gray", vmin=0, vmax=1.6, extent=(0, km, 0, km))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad((0, 0, 0, 0))
    im = ax.imshow(data.T, origin="lower", cmap=cm, vmin=0, vmax=vmax, extent=(0, km, 0, km),
                   interpolation="nearest")
    ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label(unit, color=MUTED, fontsize=9)
fig.text(0.02, 0.015, f"{path.stem}: {km / 1000:g} x {km / 1000:g} km, {len(pts):,} street points every {EVERY} m, "
         "buildings in grey, one tile of margin loaded around the block", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
out = ROOT / "outputs/report" / f"{path.stem}_svf_sun.png"
fig.savefig(out, dpi=110, facecolor=BG)
print(out)
