"""Check: the library rebuilds the 3D token sets of the paper (train_oct3d.py, train_vox3d.py).

oct3d: 2 m voxels from LoD1 occupancy (0-64 m), 64 m root cubes, split while mixed, plus the
"ground cap" rule (a ground-touching cube splits while it is larger than the 2D leaf under it),
written here as a user rule. Tokens: all leaves, all internal cells and two levels above the roots
(128 m, whole tile). Links: same-size cell across each face, ancestors (leaves), parent and children
(internal cells). Compared per tile: the cells (position, size, leaf or not) and the key set of every
token. The paper places its two top levels as 64 m tall blocks, the library as cubes, so their
centres differ in z; they are matched by x, y and size.

Known difference (a bug in train_oct3d.py line 99, found by this check): for a 64 m root cube the
paper links to S128[p[0] // 2, p[1] // 2] with p already halved, so every root (and every leaf, via
its ancestors) reads the south-west 128 m block, wherever it is. The library links each cube to the
128 m block that contains it. The check reports differences with and without those links.

vox3d: uniform 2 m voxels, z 0-16 m, 6 face neighbours each.

Needs the paper data and scripts (torch for importing them); reads only.
Usage: python scripts/check_paper_3d.py [tile ...]
"""
import math, pickle, sys, time

import numpy as np

from octreecity.build import build, uniform
from octreecity.columns import ColumnCity
from octreecity.learn import tokens
from octreecity.links import across, hierarchy, neighbours
from octreecity.rules import mixed, where
from octreecity.sources import ColumnSource
from octreecity.urban import BUILDING

PC = "/Users/mahmoud/Projects/octreecity-paper-code"
sys.path.insert(0, f"{PC}/experiments/scripts")
import train_oct3d as o3  # noqa: E402
import train_vox3d as v3  # noqa: E402


def occupancy_city(pic):
    h = pic.astype(np.float64)[::-1, :].T
    b = (h.reshape(128, 2, 128, 2) > 0).mean((1, 3)) >= 0.5
    h2 = np.where(b, h.reshape(128, 2, 128, 2).mean((1, 3)), 0.0)
    nl = np.where(b, np.clip(np.ceil(h2 / 2), 1, 32), 0)           # building voxels per 2 m column
    return ColumnCity.from_heightfield(np.zeros_like(nl), nl, BUILDING, 32, voxel_size=2.0)


def ground_cap(cs):
    """Rule: split a ground-touching box while it is larger than the smallest 2D leaf under it."""
    pyr = {1: cs}
    for s in (2, 4, 8, 16, 32):
        a = pyr[s // 2]
        pyr[s] = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).min((1, 3))

    def rule(b):
        g = pyr[b.size][b.x // b.size, b.y // b.size]
        return (b.z == 0) & (g < b.size)
    return rule


def key(xyz, s):
    z = round(float(xyz[2]), 3) if s <= 64 else None
    return (round(float(xyz[0]), 3), round(float(xyz[1]), 3), z, round(float(s), 3))


def compare(t, g):
    """Same tokens (matched by position and size) and same key set per token."""
    if len(t) != g["T"]:
        return f"tokens {len(t)} vs {g['T']}"
    ids = {key(g["xyz"][i], g["size"][i]): i for i in range(g["T"])}
    m = np.array([ids[key(t.position[i], t.size[i])] for i in range(len(t))])
    bad = other = 0
    for i in range(len(t)):
        diff = set(m[t.idx[i][t.valid[i]]].tolist()) ^ set(g["idx"][m[i]][g["valid"][m[i]]].tolist())
        bad += bool(diff)
        other += any(g["size"][j] != 128 for j in diff)
    return bad, other


def main():
    geo = pickle.load(open(f"{PC}/data/geo/geo_oct.pkl", "rb"))
    pics = np.load("/Users/mahmoud/Projects/urban_wind/outputs/height_pictures.npz")["pictures"]
    want = [int(a) for a in sys.argv[1:]] or [d["tile"] for d in geo]
    tl = tp = 0.0
    bad_tiles, n128 = [], 0
    for d in (d for d in geo if d["tile"] in want):
        t0 = time.time()
        city = occupancy_city(pics[d["tile"]])
        cs = (d["size"][d["pix2cell"]].reshape(128, 128) / 2).astype(int)   # 2D leaf size per pixel (voxels)
        tree = build(ColumnSource(city), mixed() | where(ground_cap(cs)), 5, shares=True)
        tree.cells["occ"] = tree.cells["share"][:, BUILDING]
        p = tree.parents(levels_above=2)
        links = across(tree, p) | hierarchy(tree, p)
        tok = tokens(tree, ["occ"], links, parents=p, k=16)
        tl += time.time() - t0
        t0 = time.time()
        g = o3.build(d, pics[d["tile"]])
        tp += time.time() - t0
        r = compare(tok, g)
        # vox3d: uniform 2 m voxels, z 0-16 m, 6 face neighbours
        low = ColumnCity.from_dense(city.to_dense()[:, :, :8], 2.0)
        u = uniform(ColumnSource(low), 0)
        vt = tokens(u, ["label"], neighbours(u, 6), k=7)
        gv = v3.build_vox(d, pics[d["tile"]])
        rv = compare(vt, gv)
        print(f"tile {d['tile']}: oct3d {len(tok)} tokens, key sets differ: {r[0]} "
              f"({r[1]} beyond the 128 m block links); vox3d {len(vt)} tokens, differ: {rv[0]}", flush=True)
        n128 += r[0] > 0
        if isinstance(r, str) or r[1] or rv[0]:
            bad_tiles.append(d["tile"])
    print(f"{len(want)} tiles; with a different 128 m block link: {n128}; "
          f"with any other difference: {len(bad_tiles)} {bad_tiles[:10]}")
    print(f"time per tile: library {tl / len(want):.2f} s, paper build {tp / len(want):.2f} s")


if __name__ == "__main__":
    main()
