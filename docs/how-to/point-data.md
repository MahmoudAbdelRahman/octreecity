# Your own point data on cells

Sensors, shops, survey points, tree inventories: any data with a location
can go onto the cells that contain them. Each point goes to one leaf, and
each leaf gets one value per column: the mean of its points, the most
common category, the id they share, and so on.

## A CSV with lon, lat

```python
import pandas as pd
from octreecity.easy import city_from_osm
from octreecity.fill import from_points

city = city_from_osm(36.3725, 127.3604, size_m=500)
tree = city.to_octree(5)

df = pd.read_csv("sensors.csv")            # columns: lon, lat, pm25, no2, kind, sensor_id
from_points(tree, df, x="lon", y="lat", crs="EPSG:4326")

tree.cells["pm25"]                         # one value per leaf, NaN where there is no sensor
tree.cells["kind"]                         # the most common kind per leaf
tree.cells["sensor_id"]                    # the sensor's id, -2 if a leaf holds several
```

`crs="EPSG:4326"` means x, y are lon, lat. They are converted to the
tree's own system (`tree.crs`, metres). Without `crs`, x, y are metres in
`tree.crs`, the same frame as `tree.centers()`.

Every column except the positions goes onto the cells, written straight
into `tree.cells`. The call also returns them as a dict.
`columns=["pm25", "no2"]` picks some columns, and `write=False` leaves the
tree alone. Points outside the tree's box are dropped, and a warning says
how many.

Any table works: pandas, polars, a pyarrow Table, a DuckDB result, or a
plain dict of arrays. pandas is not needed by OctreeCity itself.

## A GeoDataFrame

```python
import geopandas as gpd

gdf = gpd.read_file("trees.gpkg")          # point geometry, any crs
from_points(tree, gdf)                     # x, y and crs come from the geometry
```

## Which summary

Each column gets a default from its type. `how=` changes it for all
columns, or for some with a dict: `how={"pm25": "max"}` (the other
columns keep their default).

| `how=` | gives | empty leaf |
|---|---|---|
| `mean` (default for numbers) | mean of the points' values | NaN |
| `sum`, `count` | total, number of points | 0 |
| `min`, `max`, `std` | as named | NaN |
| `mode` (default for text) | most common value; a tie goes to the smallest | `""` (integers: -1) |
| `same` (default for ids) | the id all points in the leaf share, else -2 | -1 |

NaN and missing values are skipped. `from_points(tree, x=xs, y=ys)`
with no data counts the points in each leaf.

## Ids

Ids (sensor ids, object ids) stay exact and are never averaged. Integer
columns named `id` or `..._id`, columns listed in `tree.id_columns`, and
integers too big for float64 are ids. They get `same`, the library's rule
for ids everywhere: a leaf keeps the id all its points share, -2 if they
differ, -1 if it has none. They are added to `tree.id_columns`, so
`tree.parents().reduce` and `carry()` keep them exact too.

A city tree already has an `id` column (the buildings' ids). A point
column with the same name and other values is refused. Rename it first:
`df.rename(columns={"id": "sensor_id"})`.

## Height

Most point files have no height. With no `z`, each point sits 1.5 m above
the ground under it (street level). The ground is the `ground` class and
its parts (road, pavement, grass, water) stacked from the bottom of the
box. A point inside a building's footprint lands on its ground floor.
`height=10` puts the points 10 m above the ground instead.

If your data has a height, give it: `z="height"` (a column) or `z=12.0`
(one number for all). z is in metres in the tree's frame:
`tree.origin[2]` is the bottom of the box, not the ground.

## Data bigger than memory

Give the chunks instead of the whole table. The result is the same:

```python
chunks = pd.read_csv("big.csv", chunksize=1_000_000)
from_points(tree, chunks, x="lon", y="lat", crs="EPSG:4326", columns=["pm25", "sensor_id"])
```

Any iterable of tables works: pyarrow record batches, polars
`iter_slices()`, a list of DataFrames. `mean`, `sum`, `count`, `min`,
`max`, `std` and `same` combine across chunks. `mode` does not (it needs
all points at once), so a text column with chunks is refused. Leave it out
with `columns=`, or read that column in one go.

## Plain arrays

```python
tree.cells["pm25"] = from_points(tree, pm25, x=lon, y=lat, crs="EPSG:4326")
```

With one array of values (or none, to count the points), the result is
one array and nothing is written.

## Just the cell of each point

```python
cell = tree.cell_at(lon, lat, crs="EPSG:4326")      # leaf per point, -1 outside; z=None: street level
tree.centers()[cell[cell >= 0]]                       # where those cells are
tree.label[cell[cell >= 0]]                           # what they hold (air, building ...)
```

`tree.locate(i, j, k)` does the same for integer voxel indices (counted
from the box's corner), also with -1 outside.
