# 3D views

The examples on this page start from a city, for example:

```python
from octreecity.easy import city_from_osm
city = city_from_osm(36.3700, 127.3625, size_m=1000)   # KAIST, Daejeon; any lat, lon works
```

Interactive 3D in Jupyter or VS Code (plotly): drag to rotate, scroll to zoom,
click a legend entry to hide it.

```python
from octreecity.viz import show, show_values, show_cells, show_octree, show_selection, street_map
from octreecity.analysis import street_points, sky_view_factor

show(city)                                       # the city in 3D
show(city, labels=[2, 3])                        # only buildings and trees

pts, where = street_points(city, every=4)
svf = sky_view_factor(city, pts)
show_values(city, street_map(city, where, svf, 4), "Sky view factor",
            colorscale="Cividis", cmin=0, cmax=1)  # results painted on the street

show(city.crop(0, 0, 160, 160), wireframe=True)    # faces with their edges, to explain the model
show(city.crop(0, 0, 160, 160), wireframe="only")  # edges only, see-through

tree = city.crop(0, 0, 256, 256).to_octree(4)
show_octree(tree, style="cutaway")              # block with a corner removed, cells by size
show_octree(tree, style="shells")               # nested envelopes of 1 m, 2 m, 4 m ... cells
show_cells(tree)                                # a single vertical cut

from octreecity.viz import downsample
small = downsample(city.crop(0, 0, 128, 128), 2).to_octree(3)
show_octree(small, style="wire")                # every air cell as a wire cube, by size
```

## Any selection of cells

Any group of cells (from `tree.select`, `neighbours`, `within`, `siblings`,
...) draws in the same wire style, or as filled cubes (`style="cubes"`), at
the cells' real size, with the buildings drawn lightly around it:

```python
g = tree.select(id=building_id)
g.show()                                        # coloured by cell size
g.neighbours(steps=2).show()
show_octree(tree, style="wire", cells={"building": g,                       # one colour per group,
                                       "neighbours": g.neighbours(steps=1),  # with a legend
                                       "within 10 m": g.within(10.0)})
```

The views use the frame of `show_octree`, so they line up with it. A group
can also be a mask over the leaves (`tree.label == BUILDING`) or a list of
leaf numbers. Past `max_cells` cubes (15,000 wire, 25,000 filled) the most
numerous sizes are thinned evenly (spread over the whole selection) and the
legend says how many are drawn, so big selections stay quick and keep their
shape; select a smaller area to see every cell. `context=()` leaves out the
buildings around.

## Data on cells: colour any cell by a value

Any value stored per cell in `tree.cells` can be seen in every view with
`color=`. The cells themselves take the colour, at their real size: filled
cubes or wire cubes for any selection (such as the cells along the
building walls), the cell squares on cut faces and sections, the faces of
the shells. This works on any tree, also one whose cells were split where a
value changes (`build(..., varies("sun", 1.0))`, `adapt`).

```python
import numpy as np
from datetime import date
from octreecity import fill
from octreecity.analysis import sun_hours
from octreecity.render import plot_slice
from octreecity.viz import show_cells, show_octree

block = city.crop(0, 0, 160, 160)                       # views of a 160 m block stay under ~10 MB
tree = block.to_octree(4)

tree.cells["height"] = fill.from_function(tree, lambda x, y, z: z)   # any value per cell
air = tree.label == 0
hours = np.full(tree.n_leaves, np.nan)                   # no value inside solids
hours[air] = sun_hours(block, tree.centers()[air], 36.37, 127.3625,
                       date(2026, 12, 21), 9, mode="exact")
tree.cells["sun_hours"] = hours                          # winter sun per air cell

show_octree(tree, style="wire", color="sun_hours")       # in the air
show_octree(tree, style="cutaway", color="sun_hours")    # on two cut faces
show_octree(tree, style="shells", color="sun_hours")     # on the shells by cell size
show_cells(tree, color="sun_hours")                      # one vertical cut, 3D

walls = tree.select(where=lambda c: c["label"] == 2).neighbours(steps=1)
walls.show(color="sun_hours", style="cubes", nan="hide") # the cells along the walls, filled
walls.show(color="sun_hours", nan="hide")                # the same cells as wire cubes

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 3))
plot_slice(tree, ax, axis="y", color="sun_hours")        # a section as an image
```

`color=` takes a `tree.cells` column name, a built-in column (`"size_m"`,
`"z"`, `"level"`, `"class"`, `"label"` ...), a value of a group (`"steps"`,
`"distance"`), or an array with one value per cell.

- **Numbers** get a colour scale and a colour bar: `colorscale="Viridis"`
  (any plotly scale; in `plot_slice` also any matplotlib name), `cmin=`,
  `cmax=`. By default the scale spans the values of the whole tree (of the
  selection for `g.show`), so a cut, a crop of the view or thinning never
  changes the colours.
- **Categories** get one distinct colour each (up to 20 in a view; past
  that colours repeat and the legend says so) and a legend, never a
  gradient: text, `True`/`False`, `class` and `label`, object ids (columns in
  `tree.id_columns`, `"id"` or `"..._id"`) and integers with at most 12
  different values. Ids stay exact integers (-1 is "no value", -2 "mixed").
  To choose the other way, pass `tree.cells["zone"].astype(str)` or
  `.astype(float)`.
- **Cells without a value** (NaN, None, id -1) are grey (`nan="grey"`) or
  left out (`nan="hide"`); the legend counts them. On cut faces and
  sections, buildings and ground without a value keep their own colour.

What each view colours: `g.show(style="cubes")` (or
`show_octree(tree, cells=g, style="cubes")`) fills each selected cell as a
cube, with thin seams between cells and `opacity=` for see-through cubes;
faces between two drawn cells are left out, so a layer of cells reads as a
skin. `wire` draws the edges of the air cells (use `cells=` or `g.show` for
others); by value, each cube sits a hair inside its cell so that it shows
all 12 of its own edges in its own colour. `cutaway`, `show_cells` and
`plot_slice` colour every cell on the cut (`cut=(x, y)` and `y=` are metres
from the tree's corner, not absolute coordinates); `shells` the outer faces
of each shell, by the cell inside. Past
`max_cells` (15,000 wire cubes, about 7 MB, or 25,000 filled cubes, up to
about 10 MB) views are thinned evenly, every size kept, and the legend says
how many cells are drawn. Shells by value draw one face per cell, several
times bigger than plain shells: a block of about 160 m is fine (about 7 MB);
bigger ones stop with a message to crop.

Integer columns from `fill.from_points(how="mode")` hold -1 where a cell
has no point. -1 counts as "no value" only for id columns, so mark it
yourself: `color=np.where(tree.cells["use"] == -1, None, tree.cells["use"])`
(the integers stay exact).

Big areas are simplified for display only (`step`, chosen automatically to
keep views quick); the data is not changed. Save a view as a web page with
`fig.write_html("city.html")`.

In a notebook, set the renderer once:

```python
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"
```

Wireframes draw every edge, so use them on small areas (a block of
100 to 200 m); they stop with a clear message if an area is too big.
