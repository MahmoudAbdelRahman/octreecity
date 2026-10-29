"""Real 3D data with semantic structure: LoD2 CityGML buildings (walls, roofs, ground surfaces).

The surfaces become occupied 0.5 m voxels (SparseSource.from_surfaces) with a class tree
building -> wall / roof / ground surface, and two attributes per building (an object number and
its measured height). The same data is then split by different rules:

    mixed(depth=0)                 building vs empty: walls and roofs are one class
    mixed(depth=1)                 wall vs roof vs ground surface also split
    mixed(depth=0) | changes(obj)  two buildings never share a cell

and the cells are counted against a uniform 0.5 m grid. Saves a vertical section.

Needs a CityGML file with LoD2 surfaces (default: a V-World KAIST tile).
Usage: python scripts/demo_lod2.py [file.gml]
"""
import sys
import time
from pathlib import Path

import numpy as np

from octreecity.build import build
from octreecity.classes import Classes
from octreecity.features import read_citygml_surfaces
from octreecity.learn import tokens
from octreecity.links import hierarchy, neighbours
from octreecity.rules import changes, mixed
from octreecity.sources import SparseSource

ROOT = Path(__file__).resolve().parent.parent
GML = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Projects/kaist_citygml/tile_b1_96_32.gml"
CELL = 0.5

surf = read_citygml_surfaces(GML)
kinds = sorted(set(surf.kind))
classes = Classes.from_tree({"building": {k: {} for k in kinds}})
labels = classes.ids(surf.kind)
height = np.array([o.get("measuredHeight", np.nan) for o in surf.objects])
print(f"{len(surf.objects)} buildings, {len(surf):,} polygons, kinds {dict(zip(*np.unique(surf.kind, return_counts=True)))}")
print(classes)

allp = np.concatenate(surf.polygons)
origin = allp.min(0) - CELL
t0 = time.perf_counter()
src = SparseSource.from_surfaces(surf.polygons, labels, CELL, origin=origin, classes=classes,
                                 attrs={"obj": surf.obj.astype(float), "height": height[surf.obj]})
print(f"{src.code.size:,} occupied {CELL} m voxels in a box of {src.shape} voxels "
      f"({time.perf_counter() - t0:.1f} s)")

L = 6                                              # largest cell 2**6 voxels = 32 m
rows = [("building vs empty, mixed(depth=0)", mixed(depth=0)),
        ("wall / roof / ground, mixed(depth=1)", mixed(depth=1)),
        ("one building per cell, mixed(depth=0) | changes('obj')", mixed(depth=0) | changes("obj"))]
uniform = int(np.prod(src.shape))                  # uniform 0.5 m cells over the data's own box
for name, rule in rows:
    t0 = time.perf_counter()
    tree = build(src, rule, L, label_depth=0 if "depth=0" in name else None)
    dt = time.perf_counter() - t0
    tree_in = int(tree.in_box((0, 0, 0), src.shape).sum())
    print(f"{name:58s} {tree_in:>9,} cells  ({uniform / tree_in:,.0f}x fewer than {uniform:,} "
          f"uniform {CELL} m cells), {dt:.2f} s")

tree = build(src, mixed(depth=1), L, shares=True)
p = tree.parents()
tok = tokens(tree, ["size_m", "height"], neighbours(tree) | hierarchy(tree, p), parents=p, k=32)
print(tok)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    x, y, z = tree.corners()
    s = tree.sizes()
    yc = int(np.median(y[tree.label > 0]))
    cut = (y <= yc) & (y + s > yc)
    named = {"WallSurface": "#c9ccd3", "RoofSurface": "#d9734e", "GroundSurface": "#8a7a62"}
    col = {0: "#0f1318", **{classes.id(k): named.get(k, "#4f8fc4") for k in kinds}}
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f1318")
    ax.set_facecolor("#0f1318")
    rects = [Rectangle((x[i] * CELL, z[i] * CELL), s[i] * CELL, s[i] * CELL) for i in np.nonzero(cut)[0]]
    ax.add_collection(PatchCollection(rects, facecolor=[col.get(int(v), "#888") for v in tree.label[cut]],
                                      edgecolor="#2a3038", linewidth=0.2))
    solid = cut & (tree.label > 0)
    ax.set_xlim(x[solid].min() * CELL - 4, (x[solid] + s[solid]).max() * CELL + 4)
    ax.set_ylim(z[solid].min() * CELL - 2, (z[solid] + s[solid]).max() * CELL + 4)
    for k in kinds:
        ax.plot([], [], "s", color=named.get(k, "#4f8fc4"), label=k)
    ax.legend(loc="upper right", frameon=False, labelcolor="#e6e9ee", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(colors="#8b95a3")
    ax.set_title(f"LoD2 walls, roofs and ground surfaces as cells, vertical section at y = {yc * CELL:.0f} m",
                 color="#e6e9ee", loc="left")
    out = ROOT / "outputs/report/lod2_cells.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#0f1318", bbox_inches="tight")
    print(out)
except ImportError:
    pass
