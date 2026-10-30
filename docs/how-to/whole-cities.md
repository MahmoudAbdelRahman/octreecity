# Whole cities: tiled files

A city too large for memory is stored as square tiles in one zip file
(`.ocz`). Any tile can be read alone.

`buildings`, `parks` and `roads` below are `Features`, for example from
`read_citygml` or `read_geojson` (see [your own data](own-data.md)).

```python
from octreecity.tiles import build_tiled, TiledCity

build_tiled("osaka.ocz", buildings, cell=1.0, tile=256, workers=8,
            landcover=[(parks, 5), (roads, 7, 6.0)])

city = TiledCity("osaka.ocz")
one = city.tile(40, 46)                    # an Occupancy
block = city.region(39, 45, 45, 51)        # 6 x 6 tiles stitched, missing tiles = air
i, j = city.index_of(x, y)                 # which tile holds the point (x, y), in metres
```

All of Osaka (556,724 buildings, 228 km² at 1 m) builds in 26 s on 8 cores
into 37 MB. For analyses, load one tile of margin around the area you need
and crop the result, so buildings just outside still count.

## A folder store with objects

For a city with object ids, edits and appends, use the folder store. It
keeps one object table for all tiles, and several processes can write tiles
at the same time.

```python
from octreecity import Store

store = Store.create("city.octreecity", voxel_size=1.0, max_level=5,
                     tile_m=240, origin=(x0, y0), crs="EPSG:32648")
store.tile_m                       # 256: rounded up to a multiple of the biggest cell
store.write_tile(0, 0, occupancy, tree, objects)
store.objects()                    # the whole object table
store.read_tile(0, 0)              # an Occupancy with its ids
store.read_tree(0, 0)              # its octree
store.region(0, 0, 4, 4)           # 4 x 4 tiles as one Occupancy
store.edit_objects(rows)           # change rows (the new values win)
store.compact()                    # merge the table parts, delete old files
store.snapshot("city.zip")         # a read-only copy in one file
Store.open("city.zip")
```

The tile size is free, but it rounds **up** to a multiple of the biggest
cell (2**max_level voxels: 32 m at 1 m voxels and max_level 5), so 240 m
becomes 256 m and the area you asked for is never cut. Then the tiles' octrees
joined are the same octree as one build of the whole area.

To build many tiles in parallel, give `store.build` a function that builds
one tile:

```python
def one_tile(store, i, j):
    x0, y0, x1, y1 = store.tile_box(i, j)
    b = osm_builder(osm, x0, y0, x1 - x0, y1 - y0)   # data of the tile plus a margin of about 15 m
    city = b.build(nz=200)                            # the same height for every tile
    return city, city.to_octree(store.max_level), city.objects

store.build(one_tile, [(i, j) for i in range(8) for j in range(8)], workers=8)
```

Each tile needs the source data of the tile plus a margin at least as wide as
the widest road or tree crown, so shapes that cross the tile edge are drawn
the same way in both tiles. The same object seen by two tiles is one row in
the table (its id is computed from its source, not counted).
