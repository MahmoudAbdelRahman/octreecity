# Where tiles are on Earth

Inside a tile, cells are in local metres (the tile's crs and origin). On
top of that, every tile records **where it is on Earth**. Then you can find
tiles by place, match tiles from different sources, put several cities in
one dataset, and split training data by area.

## What a tile records

```python
from octreecity.store import Store

store = Store.create("city.octreecity", voxel_size=1.0, max_level=5,
                     tile_m=256, origin=(x0, y0), crs="EPSG:32648")
# ... write tiles ...
store.place(0, 0)
# {"lonlat": [103.8051, 1.2838],                     centre, WGS84 degrees
#  "z_m": [0.0, 64.0],                               bottom and top, metres
#  "places": [{"scheme": "s2", "level": 13, "id": "31da1bcc"}]}
```

- `lonlat` is exact and needs no scheme.
- `places` gives the tile's cell under each **cell scheme**. By default that
  is [S2](https://s2geometry.io) level 13: cells of about 1 km. S2 cells are
  not metric squares, so they name the place; they do not replace the
  local metres inside the tile.
- Ids are text: S2 ids do not fit a signed 64-bit number, and other codes
  are not numbers at all.

A store without a crs has no places. An Occupancy or octree with a crs
gives its place too: `city.place()`, `tree.place()`. It is also saved
inside the `.npz` files.

## Find tiles

```python
from octreecity.places import S2, near, find

t = store.place_table()       # one row per tile: source, i, j, lon, lat, z_bottom_m, z_top_m, s2_13
near(t, 103.804, 1.2826, 2000)       # tiles within 2 km, nearest first (distance_m column)
find(t, "s2_13", "31da1bcc")         # tiles in one cell, by the stored id
find(t, S2(10), "31da1bd")           # tiles in a bigger cell (about 8 km)
```

`t` is a pyarrow table: `t.to_pandas()` for pandas, or save it as parquet.

## Several cities, several sources

```python
from octreecity.places import concat

t = concat(singapore.place_table("singapore"), tokyo.place_table("tokyo"))
```

Tiles from two sources over the same area get the same S2 id, so you can
match them by that id.

## Train and test by area

Neighbouring tiles look alike, so a random split leaks. Split by blocks
instead:

```python
from octreecity.places import split

train, test = split(t, S2(10), test=0.2, seed=0)   # no 8 km cell in both
```

## Your own scheme

A scheme is any object with `name`, `level` and
`key(lon, lat, z_bottom_m, z_top_m)` that returns the id (text) of the cell
holding the tile. 3D codes use the heights; 2D ones ignore them. Japan's 3D
Spatial ID (ZFXY), the BeiDou grid code, or a code of your own all fit:

```python
class MyCode:
    name = "mycode"

    def __init__(self, level):
        self.level = level

    def key(self, lon, lat, z_bottom_m=None, z_top_m=None):
        return ...                       # your id, as text

store = Store.create(path, ..., crs="EPSG:32648", places=[S2(13), MyCode(5)])
store = Store.open(path, places=[MyCode(5)])   # needed again to write tiles
```

Nothing is registered: you pass scheme objects where they are needed.
Files keep only the name, level and id, so anyone can read and match the
ids without your code. Add a scheme to an existing store, or fill in a
store written before 0.5, with `store.add_places(MyCode(5))`.
