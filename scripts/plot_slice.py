"""Vertical slice through an Osaka tile: plain octree vs meshing octree."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from octreecity import BUILDING, GROUND, LinearOctree, pad_to_multiple
from octreecity.export import plot_slice

SRC = Path.home() / "Projects/urban_wind/data/cities/osaka/height_pictures.npz"
OUT = Path(__file__).resolve().parent.parent / "outputs"

tile = int(sys.argv[1]) if len(sys.argv) > 1 else 868
h = np.load(SRC)["pictures"][tile].astype(np.float64)
ground = np.ones_like(h)
surface = ground + np.where(h > 0, h, 0)
nz = pad_to_multiple(1 + h.max() + 16, 5)
plain = LinearOctree.from_heightfield(ground, surface, BUILDING, nz, 5)
mesh = LinearOctree.from_heightfield(ground, surface, BUILDING, nz, 5)
mesh.limit_size_near([BUILDING, GROUND], distance=8, max_level=1).balance(26)
y = int(np.unravel_index(h.argmax(), h.shape)[1])

fig, axes = plt.subplots(2, 1, figsize=(11, 2 * 1.1 * 11 * nz / 256 + 1.2))
for ax, t, name in [(axes[0], plain, "plain octree (merge equal labels)"),
                    (axes[1], mesh, "meshing octree (air <= 2 m within 8 m of solid, 2:1 balanced)")]:
    plot_slice(t, ax, axis="y", index=y)
    ax.set_title(f"tile {tile}, y = {y} m: {name}, {t.n_leaves:,} leaves in 3D", fontsize=9)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / f"slice_tile{tile}.png", dpi=130)
print(OUT / f"slice_tile{tile}.png")
