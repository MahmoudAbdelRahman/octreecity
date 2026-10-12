"""Bar chart: storage of the same 20 Osaka tiles in each format."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "outputs/report/formats.png"
BG, FG, MUTED, GRID, HI, LO = "#0f1318", "#e6e9ee", "#8b95a3", "#232a33", "#f2a65a", "#56606d"

panels = [
    ("On disk, zipped (kB)", ["dense grid", "octree", "columns", "height map*"], [662, 575, 228, 88]),
    ("In memory (MB)", ["dense grid", "octree", "columns"], [113.2, 55.3, 20.0]),
    ("Build time per tile (ms)", ["octree", "columns"], [17.4, 1.2]),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=BG,
                         gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.55))
for ax, (title, names, vals) in zip(axes, panels):
    ax.set_facecolor(BG)
    cols = [HI if n == "columns" else LO for n in names]
    y = range(len(names))[::-1]
    ax.barh(list(y), vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:g}", va="center", color=FG, fontsize=10)
    ax.set_yticks(list(y), names, color=FG, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=FG, fontsize=11.5, loc="left", pad=10)
fig.text(0.01, 0.01, "Same voxels: 20 Osaka tiles, 256 x 256 m, 1 m cells.  * height map stores buildings "
         "only, no labels or layers: a lower bound.", color=MUTED, fontsize=8.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16)
fig.savefig(OUT, dpi=130, facecolor=BG)
print(OUT)
