"""Check: the library rebuilds the token sets of the paper's masked-infill test (train_infill.py).

A 24 m x 24 m patch of a tile is hidden. Every street-level cell that touches the patch is split
into 2 m cells (in the library: refine_where on the tree, a user rule), and each cell reads its 8
nearest face neighbours (neighbours in the layer, nearest in the plane first, k = 9 with itself). Compared on
several random patches per tile: the cells (position, size) and every token's key set. Where a cell
has more than 8 neighbours and several sit at the same distance at the cut, the two codes may keep
different ones; such a token counts as a match when the kept distances agree.

Needs the paper data and scripts (torch for importing them); reads only.
Usage: python scripts/check_paper_infill.py [n_tiles] [patches_per_tile]
"""
import math, pickle, sys

import numpy as np

from octreecity.build import build
from octreecity.occupancy import Occupancy
from octreecity.fill import from_raster
from octreecity.learn import pixel_map, tokens
from octreecity.links import neighbours
from octreecity.rules import mixed
from octreecity.sources import OccupancySource
from octreecity.urban import BUILDING

PC = "/Users/mahmoud/Projects/octreecity-paper-code"
sys.path.insert(0, f"{PC}/experiments/scripts")
import train_infill as ti  # noqa: E402

W = ti.W


def library_tokens(pic, wx, wy):
    h = pic.astype(np.float64)[::-1, :].T
    b = (h.reshape(128, 2, 128, 2) > 0).mean((1, 3)) >= 0.5
    h2 = np.where(b, h.reshape(128, 2, 128, 2).mean((1, 3)), 0.0)
    city = Occupancy.from_heightfield(np.zeros_like(h2), h2 / 2, BUILDING, int(math.ceil((h2.max() + 16) / 2)),
                                       voxel_size=2.0)
    tree = build(OccupancySource(city), mixed(), 3).balance(6)
    layer = tree.in_box((0, 0, 0), (128, 128, 1))
    pm = pixel_map(tree, layer)
    hit = np.unique(pm[wx:wx + W, wy:wy + W])
    split_px = np.isin(pm, hit).astype(float)                    # pixels of the cells touching the patch
    tree.refine_where(lambda t: (t.corners()[2] == 0) & (from_raster(t, split_px, "max") > 0))
    layer = tree.in_box((0, 0, 0), (128, 128, 1))
    return tokens(tree, ["size_m"], neighbours(tree, 6, mask=layer), select=layer, k=9, near_dims=2)


def main():
    n_tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    geo = pickle.load(open(f"{PC}/data/geo/geo_oct.pkl", "rb"))[:n_tiles]
    vox = {d["tile"]: d for d in pickle.load(open(f"{PC}/data/geo/geo_vox.pkl", "rb")) if d["tile"] < 400}
    pics = np.load("/Users/mahmoud/Projects/urban_wind/outputs/height_pictures.npz")["pictures"]
    rng = np.random.default_rng(0)
    cases = bad = ties = 0
    for d in geo:
        v = vox[d["tile"]]
        for _ in range(per):
            wx, wy = ti.pick_window(v, rng)
            s = ti.tokens(d, v, wx, wy)
            t = library_tokens(pics[d["tile"]], wx, wy)
            cases += 1
            if len(t) != len(s["feat"]):
                bad += 1
                print(f"tile {d['tile']} patch {wx},{wy}: {len(t)} vs {len(s['feat'])} tokens")
                continue
            xy_s = np.round(s["xy"], 3)
            ids = {(x, y, round(float(z), 3)): i for i, ((x, y), z) in enumerate(zip(map(tuple, xy_s), s["feat"][:, 2] * 16))}
            m = np.array([ids[(round(float(p[0]), 3), round(float(p[1]), 3), round(float(z), 3))]
                          for p, z in zip(t.position, t.size)])
            for i in range(len(t)):
                new = set(m[t.idx[i][t.valid[i]]].tolist())
                old = set(s["idx"][m[i]][s["valid"][m[i]]].tolist())
                if new != old:
                    dist = lambda ks: sorted(round(float(np.hypot(*(s["xy"][k] - s["xy"][m[i]]))), 3) for k in ks)
                    if dist(new) == dist(old):
                        ties += 1
                    else:
                        bad += 1
                        print(f"tile {d['tile']} patch {wx},{wy}: token {i} differs")
                        break
    print(f"{cases} patches on {len(geo)} tiles: patches with a real difference {bad}; "
          f"tokens that kept a different neighbour among equally near ones: {ties}")


if __name__ == "__main__":
    main()
