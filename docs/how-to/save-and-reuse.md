# Save and reuse results

Building a city, an octree or an analysis takes seconds to minutes. Save the
result once and load it next time in well under a second.

## Cities from OpenStreetMap: one option

```python
from octreecity.easy import city_from_osm

city = city_from_osm(36.3700, 127.3625, size_m=1000,
                     cache="data/kaist_osm.json",       # the downloaded map data
                     city_cache="data/kaist_city.npz")  # the finished city
```

The first run downloads and builds, then saves both files. Later runs load the
city straight from `kaist_city.npz`. Delete that file to rebuild, for example
after changing `cell` or `size_m`.

## Any city, octree or result

```python
from octreecity.occupancy import Occupancy
from octreecity.octree import LinearOctree
import numpy as np

city.save("kaist_city.npz")               # a city (any source)
city = Occupancy.load("kaist_city.npz")

cells = city.to_octree(4)
cells.save("kaist_cells.npz")             # an octree (2 bytes per cell)
cells = LinearOctree.load("kaist_cells.npz")

np.savez_compressed("kaist_svf.npz", svf=svf, pts=pts)   # analysis results
r = np.load("kaist_svf.npz"); svf, pts = r["svf"], r["pts"]
```

## In Google Colab

Colab deletes files when the runtime restarts. Save to Google Drive instead:

```python
from google.colab import drive
drive.mount("/content/drive")
folder = "/content/drive/MyDrive/octreecity"

city = city_from_osm(36.3700, 127.3625, size_m=1000,
                     cache=f"{folder}/kaist_osm.json", city_cache=f"{folder}/kaist_city.npz")
```
