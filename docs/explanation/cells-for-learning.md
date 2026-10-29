# Cells for learning: the general pipeline

OctreeCity cuts space into cubes (cells) whose size follows what they hold:
big where nothing changes, small where something does. The core knows no
domain: it does not know what a building or a tree is. You say what your
classes are, what your attributes are, and when a cell should split.

The pipeline has five steps, each a small module:

```
data  ──>  source  ──>  build (rule)  ──>  fill  ──>  links  ──>  tokens (arrays)
          "what is      split or keep      values     who reads   rows + key tables
           inside?"                        per cell   whom        for a model
```

## 1. Classes: what things are

As in CityGML, cell content has a **semantic structure** (what something is,
and what it is part of) and **thematic attributes** (its properties). The
semantic structure is a tree of class names. Class 0 is always "empty".

```python
from octreecity.classes import Classes

c = Classes.from_tree({"building": {"wall": {}, "roof": {"window": {}}},
                       "vegetation": {"tree": {}}})
```

The city names used by the readers and viewers (air, ground, building, tree,
water, grass, pavement, road) are only a preset: `octreecity.urban.CLASSES`.

## 2. Sources: "what is inside this box?"

A source answers one question for many boxes at once: how much of each class
is inside, and the count, sum, min and max of each attribute. Three are ready:

| Source | Holds | Good for |
|---|---|---|
| `GridSource` | a full voxel grid of class ids (+ attribute grids) | any data already voxelised: scans, simulations, meshes |
| `ColumnSource` | runs of classes along each vertical column | height maps, LoD1 blocks, tree canopies over air, bridges, overhangs |
| `SparseSource` | occupied voxels only | point clouds (`from_points`), 3D polygons such as CityGML LoD2-4 surfaces or mesh faces (`from_surfaces`) |

For anything else (a mesh tested box by box, a database), write a class with
one method, `summarize(x, y, z, size)`.

CityGML surfaces keep their meaning: `read_citygml_surfaces` returns every
polygon with its kind (WallSurface, RoofSurface, GroundSurface ...) and the
building's attributes, so walls and roofs become classes of your tree
(`scripts/demo_lod2.py`). Points and surfaces mark only the surfaces: the
inside of a closed solid stays empty.

## 3. Build: split by your rule

The build starts from big boxes, asks the source, and splits the boxes your
rule picks, down to the smallest size you allow.

```python
from octreecity.build import build
from octreecity.rules import mixed, varies, changes, larger_than

tree = build(source, mixed(), max_level=6)                       # occupancy
tree = build(source, mixed(depth=1), max_level=6)                # wall vs roof matters
tree = build(source, mixed(depth=0) | changes("year"), max_level=6)
tree = build(source, lambda b: b.filled() > 0.1, max_level=6)    # your own rule
```

A rule sees, for each box, its position and size, the share of every class
(at any depth of the class tree) and the attribute summaries. `|` means
"either", `&` means "both", `~` means "not". `uniform(source, level)` builds a
uniform grid over the same data, the baseline to compare with.

## 4. Fill: values per cell

Each cell has a table: its class (`tree.label`), static columns
(`tree.cells`) and values over time (`tree.series`, one column per snapshot of
the same tree).

```python
from octreecity.fill import from_raster, from_grid, from_function

tree.cells["svf"] = from_raster(tree, svf_map)               # mean over each footprint
tree.add_snapshot("08:00", wind=from_raster(tree, wind_8))   # a value over time
```

Parent cells (the bigger cubes above the leaves) are not stored but follow
from the cell codes: `p = tree.parents()`, and `p.reduce(column)` summarises
any column over them.

## 5. Links and tokens

A link says "cell A is related to cell B", with a kind and optional values.
Neighbours, parents and ancestors are built in; any other rule is yours.

```python
from octreecity.links import neighbours, hierarchy, across, within
from octreecity.learn import tokens

layer = tree.in_box((0, 0, 0), (nx, ny, 1))                  # street-level cells
p = tree.parents(levels_above=2)
links = neighbours(tree, mask=layer) | hierarchy(tree, p)
tok = tokens(tree, ["height", "size_m"], links, targets=["svf"],
             select=layer, parents=p, k=16)
```

For one horizontal layer of cells, pass `near_dims=2` so "nearest" means
nearest in the plane (the cubes of a layer reach different heights).

`tok` holds plain arrays: one row per cell (features, position, size, targets)
and, per cell, the cells it reads with their relative offsets. That is what a
model needs. `tok.to_torch()` turns them into PyTorch tensors.

`scripts/check_paper_cells.py` shows that these few lines rebuild the cells and
links of the paper experiments exactly.

## Measuring and the optional model

`octreecity.measure.growth` counts cells, build time and memory against a
uniform grid as the smallest cell shrinks. `octreecity.attention` (install
with `pip install octreecity[torch]`) is a general link-attention model over
any tokens and links; its sizes, inputs and outputs are settings.
