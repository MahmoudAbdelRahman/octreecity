# Inputs

Everything becomes `Features` (shapes with optional height, base elevation and
label) in a projected CRS, in metres. `CityBuilder` turns them into voxels.

## Readers

| Source | Function |
|---|---|
| CityGML 2.0/3.0, LoD0-2, `.gml` or `.gml.gz` (PLATEAU, V-World, 3D BAG...) | `sources.read_citygml(path)` |
| GeoJSON (e.g. an OSM or QGIS export) | `sources.read_geojson(path, height_key="height", crs_to=EPSG)` |
| Overpass API JSON (`out geom;`) | `sources.read_osm_overpass(json, crs_to=EPSG)`: keeps the OSM type and id (`attrs["source_id"]`), holes, `min_height` of building parts |
| A list of (WKB, height) pairs | `Features.from_wkb(pairs)` |
| Anything shapely can make | `Features(geoms, height=..., base=...)` |

`read_citygml` takes the footprint from the union of all polygons of a
building, the base from the attribute `base_height_m` or the lowest point, and
the height from `measuredHeight` or highest minus lowest point. It streams the
file, so large files are fine. `features.clip(x0, y0, x1, y1)` keeps the
features touching a rectangle. `sources.reproject(geoms, from_epsg, to_epsg)`
changes CRS.

## CityBuilder

Replace the names in the example with your own data: `buildings`, `parks`,
`rivers`, `roads` and `woods` are `Features` (from the readers above), `dem`
is a terrain raster or a function of (x, y), and `x0, y0` is the south-west
corner of the area in the data's coordinate system.

```python
from octreecity.builder import CityBuilder, GRASS, WATER, ROAD

b = CityBuilder(x0, y0, width=1000, depth=1000, cell=1.0)
b.add_terrain(dem)                          # [nx, ny] raster (m) or callable(x, y)
b.add_landcover(parks, GRASS)               # top ground voxel takes the label
b.add_landcover(rivers, WATER, line_width=6)
b.add_landcover(roads, ROAD, line_width=widths)   # lines are buffered
b.add_trees(woods, top=14, base=4)          # canopy from 4 to 14 m, air below
b.add_buildings(buildings)                  # stand on the lowest terrain under them
b.add_solid(base_abs, top_abs, label)       # any other layer: bottom and top rasters (m) and a label
city = b.build(air_above=20)
```

Layers are painted in order; later ones win. Rasters are indexed `[i, j]` with
`i` along x (east) and `j` along y (north), at cell centres.

A column belongs to a shape when its cell centre is inside the shape (holes
respected). A centre that lies exactly on a wall shared by two buildings goes
to one side only, so touching buildings never claim the same column. Each
shape is painted on its own, so blocks stacked on blocks keep their own
voxels: `Features(min_height=...)` lifts a building part above the ground.

Every `add_*` call also takes `ids` (one object id per shape) and a `layer`
name, so each voxel knows which object it belongs to. See
[objects and queries](objects-and-queries.md).

## Labels

| value | name | value | name |
|---|---|---|---|
| 0 | air | 4 | water |
| 1 | ground | 5 | grass |
| 2 | building | 6 | pavement / path |
| 3 | tree | 7 | road |

Any value up to 254 can be used for your own classes.

## Heights and sizes

All lengths are metres. `cell` is the voxel size. The city's z = 0 sits one
cell below the lowest terrain point; `city.origin` gives the real position of
the corner `(x0, y0, z0)`.
