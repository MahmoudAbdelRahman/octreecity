# Use 3D models of buildings

Footprints with a height (OpenStreetMap, LoD1 CityGML) give vertical
extrusions: every wall is straight up. Real buildings lean, slope, step out
and bridge over streets. A 3D model keeps that form: a triangle mesh (OBJ,
PLY, STL) or a LoD2 CityGML solid becomes solid voxels, with its own id and
class, next to footprint buildings and terrain in one `Occupancy`. Views, crops,
sky view, sun hours and octrees then work on it as on any other city.

## Read a model

```python
from octreecity import read_mesh

m = read_mesh("towers.obj")        # OBJ, PLY or STL
m.names                            # one name per part, e.g. ['tower 1', 'tower 2', 'tower 3']
m.n_parts, len(m), m.bounds()      # parts, triangles, (min xyz, max xyz)
```

A part is one object: one building, one label, one id. In OBJ files each `o`
line starts a part (or each `g` line, if the file has no `o` lines; choose
with `read_mesh(path, by="o" | "g" | None)`). Pick the lines that split the
file into buildings, not into materials: each part should be one closed
solid. PLY and STL files are one part each. `Mesh.concat([...])` joins
several meshes; `m.select([0, 2])` keeps some parts.

Other formats (glTF, FBX, IFC, ...): load them with any package and pass the
arrays. For example, with [trimesh](https://trimesh.org):

```python
import trimesh
from octreecity import Mesh

t = trimesh.load("tower.glb", force="mesh")
m = Mesh(t.vertices, t.faces)      # part= (one number per triangle) for several objects
```

## Put it in the city

`add_meshes` reads coordinates in metres in the builder's CRS, with z up.
Models exported from a city model or georeferenced software usually already
are. A model drawn around its own origin needs moving:

```python
m = m.place(at=(x, y), z=0, rotate=30, scale=1.0, up="y")
```

`up="y"` first turns a y-up model (Blender, glTF, many OBJ exports) to z up;
then the model is scaled, rotated (degrees, counter-clockwise seen from above,
about its own origin), and its origin moved to `(x, y, z)`. Nothing is
guessed: check `m.bounds()` after placing.

## Add it next to other data

```python
from octreecity.builder import CityBuilder

b = CityBuilder(x0, y0, width=600, depth=600, cell=1.0)
b.add_terrain(dem)
b.add_buildings(footprints, ids=footprint_ids)       # extrusions
ids = b.objects.add("model", m.names, "building")    # one object per part
b.add_meshes(m, ids=ids, base="terrain")             # solids, sitting on the terrain
city = b.build()
```

`base="mesh"` (the default) keeps z as it is, in the terrain's height system.
`base="terrain"` moves each part up or down so its lowest point sits on the
lowest terrain under it, the rule `add_buildings` uses. Use it for models
whose z starts at 0 at their foot.

`label` (default `BUILDING`), `ids`, `layer`, `overlap` and `priority` work as
in `add_buildings`, one value per part. When a model and a footprint cover the
same place, the later call wins, and `build()` warns if that takes voxels from
objects of an earlier call. To replace an OSM building with a model of it,
leave the footprint out.

A runnable example without a file: a 10 x 10 m tower, 60 m tall, leaning
12 m to the east, next to a plain block.

```python
import numpy as np
import shapely
from octreecity import Mesh
from octreecity.builder import CityBuilder
from octreecity.features import Features
from octreecity.viz import show

v = np.array([[10, 10, 0], [20, 10, 0], [20, 20, 0], [10, 20, 0],
              [22, 10, 60], [32, 10, 60], [32, 20, 60], [22, 20, 60]], float)
f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
tower = Mesh(v, f, names=["tower"])

b = CityBuilder(0, 0, 50, 30, cell=1.0)
b.add_buildings(Features([shapely.box(38, 5, 46, 25)], height=[20]), ids=[1])
b.add_meshes(tower, ids=[2])
city = b.build()
show(city)
```

## What counts as inside

The same rule as for every other shape: a voxel is filled when its centre is
inside the solid. For each column, a vertical line through the column centre
crosses a closed part an even number of times; the voxels between crossing 1
and 2, 3 and 4, ... are inside. So:

- a leaning wall moves column by column;
- a sloped roof steps with the slope;
- an overhang, a sky bridge or an arch gives two runs in one column, with
  air between;
- a hollow part (a cavity modelled as an inner surface) stays empty inside.

Parts thinner than a voxel (a spire, a railing) can fall between centres and
vanish, as thin footprints do. Use a smaller `cell` where that matters.

## Models that are not closed

Vertical lines never cross walls, so holes in walls do no harm. Only holes in
the top or bottom faces matter:

- **No floor** (common): every column has one crossing too few. Each column is
  closed at the part's lowest point. This is counted in
  `b.stats["id"]["open_columns"]` with no warning.
- **Holes in a roof or floor in some columns only**: those columns are closed
  using the direction of the faces (a face seen from above is a roof, so the
  solid lies below it), and `add_meshes` warns, naming the first such part.
  If its faces point inwards, OctreeCity notices and reads them the other way.
- **Not a solid at all** (a terrain mesh, a single wall, a scan):
  `add_meshes(m, fill=False)` marks only the voxels the faces pass through.

## CityGML LoD2 solids

`read_citygml_surfaces` reads the 3D polygons of every building. `to_mesh()`
splits each polygon into triangles in its own plane, with one part per
building (named by its gml id), so LoD2 buildings are filled as solids:

```python
from octreecity.features import read_citygml_surfaces

s = read_citygml_surfaces("lod2.gml")
m = s.to_mesh()                              # or s.to_mesh(keep=s.kind != "lod1Solid")
ids = b.objects.add("citygml", m.names, "building")
b.add_meshes(m, ids=ids)
```

Many LoD2 files have no ground surfaces; that is the "no floor" case above.
If a file holds two geometries of the same building (LoD1 and LoD2), keep one
with `keep=`, or each column sees both and the fill is wrong.
`SparseSource.from_surfaces` still marks only the surfaces, for when that is
what you want.
