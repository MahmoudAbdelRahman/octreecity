"""Columns vs octree vs dense on Osaka tiles (same voxels, 1 m).

Usage: .venv/bin/python scripts/bench_formats.py [n_tiles]
"""

import io
import sys
import time
from pathlib import Path

import numpy as np

from octreecity import LinearOctree, pad_to_multiple, rasterize_columns

from octreecity.urban import BUILDING
from octreecity.occupancy import Occupancy

SRC = Path.home() / "Projects/urban_wind/data/cities/osaka/height_pictures.npz"


def npz_kb(**a):
    b = io.BytesIO()
    np.savez_compressed(b, **a)
    return b.getbuffer().nbytes / 1e3


def timed(f):
    t = time.perf_counter()
    r = f()
    return r, 1e3 * (time.perf_counter() - t)


n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
pics = np.load(SRC)["pictures"]
maxh = pics.reshape(len(pics), -1).max(1).astype(float)
built = np.nonzero(maxh > 3)[0]
pick = built[np.argsort(maxh[built])][np.linspace(0, len(built) - 1, n).astype(int)]
tot = {}
print(f"{'tile':>5} {'h':>6} {'dense.z':>8} {'octree.z':>9} {'cols.z':>7} {'hmap.z':>7} {'runs':>7} {'leaves':>8} {'t_col':>6} {'t_oct':>6} {'col->oct':>8}")
for i in pick:
    h = pics[i].astype(np.float64)
    g = np.ones_like(h)
    s = g + np.where(h > 0, h, 0)
    nz = pad_to_multiple(1 + h.max() + 16, 5)
    col, tc = timed(lambda: Occupancy.from_heightfield(g, s, BUILDING, nz))
    oct_, to = timed(lambda: LinearOctree.from_heightfield(g, s, BUILDING, nz, 5))
    _, tco = timed(lambda: col.to_octree(5))
    grid = rasterize_columns(g, s, BUILDING, nz)
    r = dict(dense=npz_kb(g=grid), octree=npz_kb(l=oct_.level, b=oct_.label),
             cols=npz_kb(c=np.diff(col.offsets).astype(np.uint16), t=col.top, b=col.label),
             hmap=npz_kb(h=pics[i]), runs=col.n_runs, leaves=oct_.n_leaves, tc=tc, to=to, tco=tco,
             cols_mem=col.nbytes / 1e3, oct_mem=oct_.nbytes / 1e3, dense_mem=grid.nbytes / 1e3)
    for k, v in r.items():
        tot[k] = tot.get(k, 0) + v
    print(f"{i:5d} {h.max():6.1f} {r['dense']:8.1f} {r['octree']:9.1f} {r['cols']:7.1f} {r['hmap']:7.1f} "
          f"{r['runs']:7d} {r['leaves']:8d} {tc:6.1f} {to:6.1f} {tco:8.1f}")
print(f"\ntotal kB  dense.z {tot['dense']:.0f}  octree.z {tot['octree']:.0f}  columns.z {tot['cols']:.0f}  heightmap.z {tot['hmap']:.0f}")
print(f"in memory MB  dense {tot['dense_mem'] / 1e3:.1f}  octree {tot['oct_mem'] / 1e3:.1f}  columns {tot['cols_mem'] / 1e3:.1f}")
print(f"build ms/tile  columns {tot['tc'] / n:.1f}  octree {tot['to'] / n:.1f}  columns->octree {tot['tco'] / n:.1f}")
