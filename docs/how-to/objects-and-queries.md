# Objects and queries

Every voxel and every octree cell can say which object it belongs to: a
building, a building part, a street, a zone, a tree, a room. The cell stores
only a number, the **object id**. Everything else about the object (its
class, where it came from, its parent, its tags) is in one **object table**.

## A city with ids

`city_from_osm` gives ids to everything it reads from OpenStreetMap:

```python
from octreecity.easy import city_from_osm

city = city_from_osm(1.2826, 103.8040, size_m=500)
city.ids["id"]          # one id per run of voxels: buildings, parts, streets, trees
city.ids["zone_id"]     # land-use zones (a second layer: they share space with buildings)
city.objects            # the object table
tree = city.to_octree(5)
tree.cells["id"]        # one id per cell, exact int64 (-1 = no object, -2 = mixed)
```

`city.objects` has one row per object:

| column | meaning |
|---|---|
| `id` | `blake2b("osm:way/123")`, a 63-bit number: the same object gets the same id in every tile, on every machine, on every rebuild |
| `class` | `building`, `building_part`, `road`, `path`, `waterway`, `tree`, `zone`, `overlap` (your own words for your own data) |
| `source` | `osm:way/123`: the way back to the source data |
| `parent` | the object this one is part of (a part's building, a park's land-use zone), -1 = none |
| `members` | only on overlap rows: the objects that overlap there (two streets at a junction) |
| `tags` | the OSM tags as JSON text: `city.objects.tag("name")` |
| `height_m`, `min_height_m`, `width_m` | numbers read from the tags |

```python
o = city.objects
o.get(i)                               # one row as a dict
o.parent(i), o.children(i), o.siblings(i), o.ancestors(i), o.descendants(i)
o.select(cls="building", column="height_m", between=(30, 200))     # ids
o.select(tags={"name": "Seng Poh Road"})
o.select(where=lambda t: t["height_m"] > 30)                        # any function
o.to_pandas()                          # if pandas is installed
```

## Selecting cells

```python
g = tree.select(id=i)                          # the object and its parts (descendants=False: only it)
g = tree.select(cls="road")                    # by an object-table column
g = tree.select(tags={"highway": "primary"})
g = tree.select(column="z", between=(10, 20))  # by a cell column: label, class, level, size_m, x, y, z, any tree.cells
g = tree.select(where=lambda c: (c["z"] > 10) & (c.object("height_m") > 30))   # any function
```

`g` is a **group** of cells. You can combine groups with `|` (or), `&` (and)
and `-` (but not).

| what | how |
|---|---|
| the objects in the cells | `g.objects()` (`members=True` turns overlap rows into their members) |
| their rows | `g.table()` |
| the cells' columns | `g.cells()` (a dict of arrays), `g.to_pandas()` |
| the voxels | `g.voxels()`: (n, 3) voxel indices; or `city.voxels(id=i)` on the Occupancy |
| neighbours by steps | `g.neighbours(steps=2, connectivity=6)`: `.steps` says 1 or 2 |
| neighbours by distance | `g.within(10.0)`: every cell within 10 m (`.distance` in metres) |
| the nearest cells | `g.nearest(50)` |
| the outer cells | `g.boundary()` |
| parent cells | `g.parents(1)`, `g.ancestors()`; `.ids()` gives the id they share, or -2 |
| sibling cells | `g.siblings()`: cells with the same parent cell |
| children of parent cells | `g.parents(1).children()`, `g.parents(1).leaves()` |

A **step** is a move between two cells that touch: `connectivity=6` through
faces, 18 also through edges, 26 also through corners. On an adaptive tree a
step into a 32 m cell covers 32 m, so steps are not metres: use `within` for
metres. `within` measures the gap between cells (0 = touching).

To see a group in 3D: `g.show()` (wire cubes coloured by cell size, the
buildings around), or several groups in one view with one colour each:
`show_octree(tree, style="wire", cells={"building": g, "neighbours":
g.neighbours(steps=1)})`. See [3D views](3d-views.md#any-selection-of-cells).

## Objects near objects

```python
tree.touching(i)                          # objects whose cells touch object i (a block resting on another touches it)
tree.touching(i, other_layer="zone_id")   # the zones it touches
tree.near(i, 25.0, return_distance=True)  # objects within 25 m, nearest first
```

## Overlaps

A voxel holds one id per layer. When two objects claim it:

- **Part of** (a building part inside its building): the part wins; the table
  keeps the relation in `parent`.
- **Touching** (two buildings that share a wall): not an overlap. A voxel
  belongs to the shape that holds its centre, and a centre exactly on the wall
  goes to one side only.
- **Different kinds** (buildings and land-use zones): separate layers, for
  example `id` and `zone_id`. All layers point into the same table.
- **The same kind** (two streets at a junction): by default the voxel gets an
  **overlap row** with both streets as members, so nothing is lost. Choose
  `overlap="first"`, `"last"` or `"by"` (with `priority=`, an array or a
  table column such as `"width_m"`) to keep one winner instead.

Shapes that overlap by design (100 m buffers around every building) should
not be an id layer: the builder warns when one voxel holds 6 or more objects
or when more than half of a layer is overlap.

## Your own data

```python
from octreecity import ObjectTable
from octreecity.builder import CityBuilder

b = CityBuilder(x0, y0, 1000, 1000)
ids = b.objects.add("plateau-tokyo-2023", gml_ids, "building", height_m=heights)
parts = b.objects.add("plateau-tokyo-2023", part_ids, "building_part", parent_source=owner_gml_ids)
b.add_buildings(buildings, ids=ids)                 # one id per shape
b.add_buildings(building_parts, ids=parts)          # Features(min_height=...) lifts parts off the ground
b.add_zones(districts, district_ids, layer="district_id")
city = b.build()
```

The first argument of `add` is the **namespace**. `"osm"` is the only fixed
one, because OpenStreetMap is one database. Anything else needs a dataset
name, because a CityGML `gml:id` or a point-cloud instance number is unique
only inside one file.

Point clouds: `SparseSource.from_points(xyz, labels, voxel_size,
ids={"id": instance_per_point})` gives each voxel the (class, id) pair held by
the most points. Name your instance field yourself; the LAS point source id
is the flight line, not an object.

## Ids are categories

Ids are never averaged, anywhere:

- the builder gives a cell the one id all its voxels share, or -2;
- `tree.parents().reduce("id")` gives the shared id or -2 (`how="same"`), and
  refuses `mean` or `sum`;
- `learn.tokens` refuses an id column as a numeric feature;
- `fill.carry` moves id columns the same way.

Ids are 63-bit numbers. Python, numpy, parquet and duckdb keep them exact.
Float64 (and JavaScript, and a pandas column that gets a missing value) keeps
only 53 bits, so do not put ids through floats.

## Saving

`city.save("city.npz")` and `tree.save("tree.npz")` keep the ids and, inside
the same file, the object table. For many tiles, use the
[folder store](whole-cities.md#a-folder-store-with-objects).
