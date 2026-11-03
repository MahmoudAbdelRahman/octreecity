# Your own point data on cells

Sensors, shops, survey points, tree inventories: any data with a location
can go onto the cells that contain it. Each point goes to one leaf; each
leaf gets one value (the mean of its points, their count, the most common
category ...).

## Points in lon, lat

```python
from octreecity.easy import city_from_osm
from octreecity.fill import from_points

city = city_from_osm(36.3725, 127.3604, size_m=500)
tree = city.to_octree(5)

# your data: one row per sensor
lon, lat, pm25 = ...                       # arrays of the same length
z = 2.0 + tree.origin[2]                   # height in the tree's frame (see below)

tree.cells["pm25"] = from_points(tree, lon, lat, z, pm25, crs="EPSG:4326")
```

`crs="EPSG:4326"` means x, y are lon, lat (in that order). They are
converted to the tree's own system (`tree.crs`, metres). Without `crs`,
x, y are metres in `tree.crs`, the same frame as `tree.centers()`.

A leaf with no point gets NaN. Points outside the tree's box are dropped,
and a warning says how many.

## Which summary

| `how=` | gives | empty leaf |
|---|---|---|
| `mean` (default for numbers) | mean of the points' values | NaN |
| `sum`, `count` | total, number of points | 0 |
| `min`, `max`, `std` | as named | NaN |
| `mode` (default for text and ids) | most common value; a tie goes to the smallest | -1 (text: `""`) |

NaN values are skipped. Without values, `from_points(tree, x, y, z)`
counts the points in each leaf.

Several columns at once, with a dict:

```python
out = from_points(tree, lon, lat, z, {"pm25": pm25, "no2": no2}, crs="EPSG:4326")
tree.cells.update(out)
```

## Ids and categories

Ids (sensor ids, object ids) must stay exact and must never be averaged.
Use `how="mode"`: the leaf gets the id of most of its points, exact as
int64, and -1 where there is none. Integers too big for float64 and
names listed in `tree.id_columns` are treated as ids: `mode` is their
default, and `mean`, `sum`, `min`, `max`, `std` refuse them.

```python
tree.cells["sensor"] = from_points(tree, lon, lat, z, sensor_id, how="mode", crs="EPSG:4326")
tree.id_columns += ("sensor",)            # parents use how="same" for it, carry() keeps it exact
```

Text categories work the same way (`how="mode"` gives the most common
word per leaf).

## Height (z)

z is metres in the tree's frame: `tree.origin[2]` is the bottom of the box,
not the ground. For points without a height on flat ground, add the top of
the ground layer:

```python
from octreecity import GROUND

ground = tree.origin[2] + city.top_of([GROUND]).max() * tree.voxel_size
z = ground + 1.5                           # 1.5 m above the street
```

Check where the points landed: `tree.label[tree.cell_at(x, y, z, crs=...)]`
should be air, not ground or building.

## Just the cell of each point

```python
cell = tree.cell_at(lon, lat, z, crs="EPSG:4326")    # leaf index per point, -1 outside
tree.centers()[cell[cell >= 0]]                        # where those cells are
```

`tree.locate(i, j, k)` does the same for integer voxel indices (counted
from the box's corner), also with -1 outside.
