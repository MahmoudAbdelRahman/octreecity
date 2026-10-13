# Quickstart

From nothing to a 3D city, sky view and sun maps in about 10 lines.

## 1. Install

```bash
git clone https://github.com/MahmoudAbdelRahman/octreecity
cd octreecity
uv sync
```

## 2. Build a city from OpenStreetMap (any place)

```python
from octreecity.easy import city_from_osm

city = city_from_osm(37.4979, 127.0276, size_m=800, cell=1.0, cache="gangnam.json")
city.save("gangnam.npz")
print(city.shape, city.crs)      # (800, 800, 171) EPSG:32652
```

`cache` keeps the downloaded OSM data, so the second run works offline.
Buildings take their height from the OSM tags `height` or `building:levels`
(3 m per level), otherwise 10 m. Terrain is flat.

## 3. Look at it in 3D

```python
from octreecity.viz import show
show(city)          # interactive in Jupyter; or show(city).write_html("city.html")
```

## 4. Sky view factor and sun hours at street level

```python
from datetime import date
from octreecity.analysis import street_points, sky_view_factor, sun_hours

pts, (i, j) = street_points(city, every=2)          # 1.5 m above ground, every 2 m
svf = sky_view_factor(city, pts)                     # 0 = enclosed, 1 = open sky
sun = sun_hours(city, pts, lat=37.4979, lon=127.0276, day=date(2026, 12, 21), tz_hours=9)
```

## 5. Adaptive cells, meshes, tensors

```python
from octreecity.exports import write_obj, tensors_2d

tree = city.to_octree(5)                 # cells from 1 m to 32 m
write_obj(city, "gangnam.obj")           # for Blender / Rhino
x, names = tensors_2d(city, factor=2)    # [channels, 400, 400] for a U-Net
```

## Next

- [Use your own data](../how-to/own-data.md): CityGML, GeoJSON, footprints, terrain, land cover, trees
- [3D views](../how-to/3d-views.md)
- [Sky view, shadows and sun hours](../how-to/sky-and-sun.md)
- [Whole cities](../how-to/whole-cities.md): tiled files, parallel builds
- [Export](../how-to/export.md): OBJ, ML arrays, OpenFOAM
- [How OctreeCity works](../explanation/how-it-works.md)
- [Command line](../reference/cli.md) and [file formats](../reference/file-formats.md)
