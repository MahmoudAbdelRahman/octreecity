"""Check: the library rebuilds the paper cells (build_octree_ds / geo_oct.pkl) and the
train_parents.py variant B graph exactly (cells, order, flags, heights, pixel map,
parent squares, key set of every token).

Needs the paper data (paper-code repo, urban_wind height pictures); reads only.
Usage: .venv/bin/python scripts/check_paper_cells.py [tile ...]
"""
import ast, math, pickle, sys, time
import numpy as np
from octreecity.columns import ColumnCity
from octreecity.build import build
from octreecity.rules import mixed
from octreecity.sources import ColumnSource
from octreecity.links import neighbours, hierarchy, across
from octreecity.fill import from_raster
from octreecity.learn import tokens, pixel_map
from octreecity.urban import BUILDING

PC = "/Users/mahmoud/Projects/octreecity-paper-code"
src = open(f"{PC}/experiments/scripts/train_parents.py").read()
ns = {"np": np, "KMAX": 16}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("h2_of", "graph"):
        exec(compile(ast.Module([node], []), "tp", "exec"), ns)
pics = np.load("/Users/mahmoud/Projects/urban_wind/outputs/height_pictures.npz")["pictures"]
geo = pickle.load(open(f"{PC}/data/geo/geo_oct.pkl", "rb"))
tiles = [int(a) for a in sys.argv[1:]] or [0, 7, 50, 123, 200, 333]
t_old = t_new = 0
for d in (g for g in geo if g["tile"] in tiles):
    t = d["tile"]
    h2, b = ns["h2_of"](pics[t])
    nz = int(math.ceil((h2.max() + 16) / 2))
    city = ColumnCity.from_heightfield(np.zeros_like(h2), h2 / 2, BUILDING, nz, voxel_size=2.0)
    hcol = np.where(b, h2, 0.0)
    t0 = time.time()
    tree = build(ColumnSource(city), mixed(), 3).balance(6)
    sel = tree.in_box((0, 0, 0), (128, 128, 1))
    tree.cells["building"] = (tree.label == BUILDING).astype(float)
    # height of a cell = tallest building pixel under it (geo_oct 'height')
    tree.cells["h"] = from_raster(tree, hcol, "max")
    tree.cells["s16"] = tree.sizes_m() / 16
    p = tree.parents(levels_above=4)
    n = tree.n_leaves
    L = (neighbours(tree, 6, mask=sel)
         | hierarchy(tree, p)
         | across(tree, p, leaves=False).where(lambda s, d: d >= n))
    tok = tokens(tree, ["building", "h", "s16"], L, select=sel, parents=p, k=16)
    t_new += time.time() - t0
    t0 = time.time(); g = ns["graph"](d, "B", pics[t]); t_old += time.time() - t0
    nl = int(sel.sum())
    # leaves: same cells, order, flags, heights, pixel map
    assert nl == len(d["cx"]) == g["n"], (nl, len(d["cx"]))
    assert np.allclose(tok.position[:nl, 0], d["cx"]) and np.allclose(tok.position[:nl, 1], d["cy"])
    assert np.allclose(tok.size[:nl], d["size"]) and (tok.features[:nl, 0] == d["building"]).all()
    assert np.allclose(tok.features[:nl, 1], d["height"])
    assert (pixel_map(tree, sel).ravel() == d["pix2cell"]).all()
    # parents: same squares
    assert len(tok) == g["T"], (len(tok), g["T"])
    key_old = {(round(x, 3), round(y, 3), round(s, 3)) for (x, y), s in zip(g["xy"][nl:], g["size"][nl:])}
    key_new = {(round(x, 3), round(y, 3), round(s, 3)) for (x, y), s in zip(tok.position[nl:, :2], tok.size[nl:])}
    assert key_old == key_new
    # same key sets per token (match parents by position and size)
    pid = lambda xy, s: (round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(s), 3))
    old_ids = {pid(g["xy"][i], g["size"][i]): i for i in range(g["T"])}
    m = np.array([old_ids[pid(tok.position[i, :2], tok.size[i])] for i in range(len(tok))])
    bad = 0
    for i in range(len(tok)):
        new = set(m[tok.idx[i][tok.valid[i]]].tolist())
        old = set(g["idx"][m[i]][g["valid"][m[i]]].tolist())
        bad += new != old
    print(f"tile {t}: {nl} leaves, {len(tok)} tokens, key sets differ for {bad} tokens, dropped {tok.dropped_links}")
print(f"time per tile: new {t_new / len(tiles):.3f} s (tree+links+tokens), old graph only {t_old / len(tiles):.3f} s")
