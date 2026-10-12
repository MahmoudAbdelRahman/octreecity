"""Whole-city build: every Osaka building (PLATEAU, 556k) at 1 m, tiled.

Reads urban_wind/data/cities/osaka/buildings.pkl.gz (read only). Flat
ground (PLATEAU heights are above ground; no DEM here). Writes
data/osaka_1m.ocz (zip of 256 m tiles).

Usage: .venv/bin/python scripts/build_osaka.py [cell_m] [workers]
"""

import gzip
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely

from octreecity.builder import CityBuilder
from octreecity.sources import Features
from octreecity.tiles import TiledCityWriter

SRC = Path.home() / "Projects/urban_wind/data/cities/osaka/buildings.pkl.gz"
ROOT = Path(__file__).resolve().parent.parent
TILE = 256.0
AIR_ABOVE = 20.0


def build_tile(args):
    i, j, x0, y0, cell, wkbs, heights = args
    b = CityBuilder(x0, y0, TILE, TILE, cell)
    if wkbs:
        b.add_buildings(Features(shapely.from_wkb(wkbs), height=heights))
    nz = int(np.ceil((max(heights, default=0) + cell + AIR_ABOVE) / cell))
    city = b.build(nz=nz)
    return i, j, city


def main():
    cell = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.perf_counter()
    blds = pickle.load(gzip.open(SRC))
    geoms = shapely.from_wkb([b[0] for b in blds])
    heights = np.array([b[1] for b in blds])
    tree = shapely.STRtree(geoms)
    bx0, by0, bx1, by1 = shapely.total_bounds(geoms)
    X0, Y0 = np.floor(bx0 / TILE) * TILE, np.floor(by0 / TILE) * TILE
    ni, nj = int(np.ceil((bx1 - X0) / TILE)), int(np.ceil((by1 - Y0) / TILE))
    jobs = []
    for i in range(ni):
        for j in range(nj):
            x0, y0 = X0 + i * TILE, Y0 + j * TILE
            idx = tree.query(shapely.box(x0, y0, x0 + TILE, y0 + TILE))
            if idx.size == 0:
                continue
            jobs.append((i, j, x0, y0, cell, [blds[k][0] for k in idx], heights[idx].tolist()))
    t_prep = time.perf_counter() - t0
    print(f"{len(blds):,} buildings, grid {ni} x {nj}, {len(jobs):,} tiles with buildings, prep {t_prep:.0f}s",
          flush=True)

    out = ROOT / "data" / f"osaka_{cell:g}m.ocz"
    t1 = time.perf_counter()
    runs = cols = 0
    with TiledCityWriter(out, X0, Y0, -cell, TILE, cell, crs="EPSG:6674",
                         extra=dict(source="PLATEAU Osaka 2020 LOD1 (via urban_wind)")) as w:
        with ProcessPoolExecutor(workers) as ex:
            for n, (i, j, city) in enumerate(ex.map(build_tile, jobs, chunksize=4)):
                w.add(i, j, city)
                runs += city.n_runs
                cols += city.shape[0] * city.shape[1]
                if n % 500 == 0:
                    print(f"  {n:,}/{len(jobs):,} tiles, {time.perf_counter() - t1:.0f}s", flush=True)
    dt = time.perf_counter() - t1
    area = len(jobs) * TILE * TILE / 1e6
    print(f"built {len(jobs):,} tiles ({area:.0f} km2) in {dt:.0f}s with {workers} workers "
          f"({1e3 * dt / len(jobs):.0f} ms/tile wall)")
    print(f"{cols:,} columns, {runs:,} runs, file {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
