"""Write viewer/data_tile<N>.js: compact (level, label) streams for the 3D viewer.

Usage: .venv/bin/python scripts/export_viewer.py [tile]
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

from octreecity import BUILDING, GROUND, LinearOctree, pad_to_multiple, rasterize_columns

SRC = Path.home() / "Projects/urban_wind/data/cities/osaka/height_pictures.npz"
OUT = Path(__file__).resolve().parent.parent / "viewer"
LEVEL = 5


def npz_kb(**a):
    b = io.BytesIO()
    np.savez_compressed(b, **a)
    return round(b.getbuffer().nbytes / 1e3, 1)


def pack(t):
    lv, lb = t.to_compact()
    return {
        "level": base64.b64encode(lv.tobytes()).decode(),
        "label": base64.b64encode(lb.tobytes()).decode(),
        "n": int(t.n_leaves),
        "hist": [int(v) for v in t.level_histogram()],
        "hist_air": [int(v) for v in np.bincount(t.level[t.label == 0], minlength=LEVEL + 1)],
        "npz_kB": npz_kb(l=lv, b=lb),
    }


tile = int(sys.argv[1]) if len(sys.argv) > 1 else 868
h = np.load(SRC)["pictures"][tile].astype(np.float64)
ground = np.ones_like(h)
surface = ground + np.where(h > 0, h, 0)
nz = pad_to_multiple(1 + h.max() + 16, LEVEL)
plain = LinearOctree.from_heightfield(ground, surface, BUILDING, nz, LEVEL)
mesh = LinearOctree.from_heightfield(ground, surface, BUILDING, nz, LEVEL)
mesh.limit_size_near([BUILDING, GROUND], distance=8, max_level=1).balance(26)
grid = rasterize_columns(ground, surface, BUILDING, nz)
data = {
    "tile": tile, "shape": list(plain.shape), "max_level": LEVEL, "max_h": round(float(h.max()), 1),
    "n_buildings_cov": round(float((h > 0).mean()), 3),
    "dense_MB": round(grid.nbytes / 1e6, 2), "dense_npz_kB": npz_kb(g=grid),
    "heightmap_npz_kB": npz_kb(h=h.astype(np.float16)),
    "plain": pack(plain), "mesh": pack(mesh),
}
OUT.mkdir(exist_ok=True)
(OUT / f"data_tile{tile}.js").write_text("window.OCTREE_DATA = " + json.dumps(data) + ";\n")
print({k: v for k, v in data.items() if k not in ("plain", "mesh")})
print("plain", plain.n_leaves, (plain.label == 2).sum(), "mesh", mesh.n_leaves, (mesh.label == 2).sum())
