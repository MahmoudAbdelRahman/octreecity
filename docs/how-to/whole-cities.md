# Whole cities: tiled files

A city too large for memory is stored as square tiles in one zip file
(`.ocz`). Any tile can be read alone.

```python
from octreecity.tiles import build_tiled, TiledCity

build_tiled("osaka.ocz", buildings, cell=1.0, tile=256, workers=8,
            landcover=[(parks, 5), (roads, 7, 6.0)])

city = TiledCity("osaka.ocz")
one = city.tile(40, 46)                    # a ColumnCity
block = city.region(39, 45, 45, 51)        # 6 x 6 tiles stitched, missing tiles = air
i, j = city.index_of(x, y)                 # which tile holds a point
```

All of Osaka (556,724 buildings, 228 km² at 1 m) builds in 26 s on 8 cores
into 37 MB. For analyses, load one tile of margin around the area you need
and crop the result, so buildings just outside still count.
