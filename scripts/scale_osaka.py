"""Scale test on all of Osaka (data/osaka_1m.ocz, 3,475 tiles of 256 m at 1 m).

tiles   every tile: octree (cells 1-64 m), and unless --octree-only also parent cells, face and
        parent links and token arrays. Prints totals, time and peak memory.
region  one tree over a block of tiles (default 16 x 16 tiles = 4.1 km x 4.1 km, 16.8 M columns),
        cells 1-256 m, built top-down from the columns.

--octree-only uses only ColumnCity.to_octree, so the same command also runs on older versions of
the library (for a speed comparison).
Usage: python scripts/scale_osaka.py tiles [--octree-only] [--limit N]
       python scripts/scale_osaka.py region [--block 16]
"""
import argparse, resource, sys, time
from pathlib import Path

import numpy as np

from octreecity.tiles import TiledCity

ROOT = Path(__file__).resolve().parent.parent
OCZ = ROOT / "data/osaka_1m.ocz"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def tiles(a):
    tc = TiledCity(OCZ)
    keys = sorted(tc.meta["tiles"])[:a.limit or None]
    leaves = tokens_n = links_n = 0
    t_read = t_tree = t_rest = 0.0
    for k, name in enumerate(keys):
        i, j = map(int, name.split("_"))
        t0 = time.perf_counter()
        city = tc.tile(i, j)
        t1 = time.perf_counter()
        tree = city.to_octree(6)
        t2 = time.perf_counter()
        leaves += tree.n_leaves
        if not a.octree_only:
            from octreecity.learn import tokens
            from octreecity.links import hierarchy, neighbours

            p = tree.parents()
            L = neighbours(tree) | hierarchy(tree, p)
            tok = tokens(tree, ["size_m", "label"], L, parents=p, k=32)
            tokens_n += len(tok)
            links_n += int(tok.valid.sum())
        t3 = time.perf_counter()
        t_read, t_tree, t_rest = t_read + t1 - t0, t_tree + t2 - t1, t_rest + t3 - t2
        if k % 500 == 0:
            print(f"{k}/{len(keys)} tiles, {time.perf_counter() - t0:.2f} s for the last one", flush=True)
    print(f"{len(keys)} tiles: {leaves:,} cells" + ("" if a.octree_only else f", {tokens_n:,} tokens, {links_n:,} key slots used"))
    print(f"time: read {t_read:.0f} s, octrees {t_tree:.0f} s" + ("" if a.octree_only else f", parents+links+tokens {t_rest:.0f} s"))
    print(f"peak memory {rss_mb():.0f} MB")


def region(a):
    from octreecity.build import build
    from octreecity.rules import mixed
    from octreecity.sources import ColumnSource

    tc = TiledCity(OCZ)
    ij = np.array([list(map(int, k.split("_"))) for k in tc.meta["tiles"]])
    i0, j0 = (ij.min(0) + ij.max(0)) // 2 - a.block // 2          # a block in the middle of the city
    t0 = time.perf_counter()
    city = tc.region(i0, j0, i0 + a.block, j0 + a.block)
    t1 = time.perf_counter()
    print(f"region {city.shape} columns, {city.n_runs:,} runs, read in {t1 - t0:.0f} s, memory {rss_mb():.0f} MB", flush=True)
    tree = build(ColumnSource(city), mixed(), 8)
    t2 = time.perf_counter()
    uniform1 = np.prod([-(-n // 256) * 256 for n in city.shape])
    print(f"tree: {tree.n_leaves:,} cells (uniform 1 m grid: {uniform1:,}), levels {tree.level_histogram().tolist()}")
    print(f"built in {t2 - t1:.0f} s, peak memory {rss_mb():.0f} MB, tree {tree.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tiles", "region"])
    ap.add_argument("--octree-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--block", type=int, default=16)
    a = ap.parse_args()
    {"tiles": tiles, "region": region}[a.what](a)
    sys.exit(0)
