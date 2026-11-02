# Changelog

## 0.4.1

### New

- See any selection of cells in 3D: `g.show()` on any Group (from
  `tree.select`, `neighbours`, `within`, `siblings` ...) draws its leaves as
  wire cubes at their real size, coloured by cell size as in
  `show_octree(style="wire")`, with the buildings drawn lightly around.
  `show_octree(tree, style="wire", cells={"building": g, "neighbours":
  g.neighbours(steps=1)})` draws several groups in one view, one colour per
  group, with a legend. A selection can also be a mask over the leaves or
  leaf numbers. Past `max_cells` cubes (15,000) the most numerous sizes are
  thinned evenly and the legend says how many are drawn. Same frame as `show_octree`, so the views line up.
  `octreecity.viz.show_selection` is the function behind both.

## 0.4.0

Object ids: every voxel and every cell can say which object it belongs to,
and one object table keeps everything else about the objects.

### New

- **Object table** (`octreecity.objects.ObjectTable`): one row per object
  with `id`, `class`, `source` (e.g. `osm:way/123`), `parent`, `members` (only
  on overlap rows) and any user columns, stored and never interpreted. The id
  is `blake2b("<namespace>:<source id>")` cut to 63 bits, so the same object
  gets the same id in every tile and process without coordination. `"osm"`
  is the one fixed namespace; other data needs a dataset name. Merging tables
  concatenates and removes duplicates; a hash collision, or a stored id that
  is not the hash of its source, raises `IdCollision`; different values raise
  unless `on_conflict="keep"` or `"replace"`; missing values are filled from
  the other table, so the order of merging does not matter. Tag access:
  `tag("name")`, `select(tags={...})`. Saved as parquet with units and
  descriptions in the metadata.
- **Ids per voxel, in 3D**: `Occupancy.ids` holds one int64 id per run for
  each id layer you name (`"id"`, `"zone_id"`, ...). Runs split wherever the
  label or any id changes, so stacked blocks, bridges over roads and canopies
  over streets keep their own objects. Air may hold ids. Ids are kept by
  `crop` (runs copied as they are), `concat_x`, the new `concat_y` and
  `pad_z`, `save` / `load`, tiles and the store. `id_at`, `voxels(id=...)`,
  `voxel_counts`, `to_dense(layer=...)`.
- **Ids per cell**: `Occupancy.to_octree` splits cells that hold two objects
  (`changes(layer)` for every id layer) and gives `tree.cells["id"]` as exact
  int64 (-1 no object, -2 mixed). Sources hand the builder small codes
  (`Source.categorical`), so big ids never pass through floats. `GridSource`
  and `SparseSource` take `ids=` too; `SparseSource.from_points(ids=...)`
  gives each voxel the (class, id) pair held by the most points.
- **Queries** (`octreecity.query`, also `tree.select`, `tree.touching`,
  `tree.near`): select cells by object id (with parts and overlap rows), by
  an object-table column or tag, by a cell column or by any function; then
  `neighbours(steps, connectivity)`, `within(metres)` (exact gap between
  cells, from a distance transform on a crop around the group), `nearest`,
  `boundary`, `parents`, `ancestors`, `siblings`, `children`, `objects`,
  `table`, `voxels`; `|`, `&`, `-` keep the per-cell values. Object level:
  `touching`, `near`, and `parent`, `children`, `siblings`, `ancestors`,
  `descendants` on the table.
- **Folder store** (`octreecity.store.Store`): `manifest.json` (written last,
  by an atomic rename under a lock file, after re-reading it), the object
  table in parquet parts, and per tile an Occupancy and an octree `.npz`
  under unique file names, so several processes can write tiles at once and
  a crash leaves the old state readable. `write_tile`, `read_tile`,
  `read_tree`, `region`, `objects`, `add_objects`, `edit_objects` (edits
  win, also after a tile is rebuilt), `delete_tile`, `build` (tiles in
  parallel), `compact`, `snapshot` (a read-only `.zip`). Each tile keeps its
  own object rows: writing a tile again replaces them. The tile size is free but rounds up to a multiple of the biggest
  cell (240 m becomes 256 m at 1 m voxels and max_level 5), so the tiles'
  octrees joined equal one build of the whole area.
- `CityBuilder`: every `add_*` takes `ids`, `layer` and `overlap` (`"keep"`:
  an overlap row with the claimants as members, the default; `"first"`,
  `"last"`, `"by"` with a priority array or table column). A shape over its
  own ancestor (a part over its building) replaces it. `add_zones` (a second
  id layer), `add_shapes` (any extrusion above the terrain), `build(z0=...)`
  so tiles share one z origin; with a terrain function, a building that
  crosses a tile edge gets the same base in every tile. `builder.stats`
  counts overlap rows and voxels; warnings for heavy overlap (buffers) and
  for later calls that take voxels of earlier objects.
- `city_from_osm` keeps the OSM type and id of every object, gives ids to
  buildings, building parts, roads, paths, waterways and trees (layer `id`)
  and land-use zones (layer `zone_id`, nested zones get a parent), stores the
  tags, and returns them in `city.objects`. `easy.osm_builder` builds one
  tile.
- `LinearOctree.crs`, `id_columns`, `objects`; `to_dense(column)`.
  `Parents.reduce(how="same")` (the value all leaves share, else -2);
  `adapt.transfer_ids`. Every `.npz` holds `__meta__` (format, crs, units)
  and, when attached, the object table.
- `Occupancy.height_of(labels, base_labels)`: heights above the ground.

### Fixed

- `read_osm_overpass` dropped the OSM type and id: now in
  `attrs["source_id"]` (and relation members in `attrs["members"]`).
- Multipolygon holes were filled (outer rings only): inner rings are now
  holes. Closed roads (roundabouts) stay lines.
- `rasterize` (PIL) gave a shared wall to both buildings and added about 6%
  to footprints: it now uses the cell-centre rule (a centre on a shared edge
  goes to one side only). One rule for geometry, label and id.
- `city_from_osm` ignored `building:part`, `min_height` and
  `building:min_level`, and `add_buildings` kept one solid per column and
  pushed floating parts down to the ground: on The Interlace (Singapore)
  98.5% of the stacked blocks' volume was lost. Parts are now read, get their
  building as parent, start at `min_height` (`Features.min_height`), and each
  shape is painted on its own. A building with parts is painted only where
  its parts do not cover it.
- Bridges (`bridge=*` or `layer > 0`) were drawn on the ground; they are now
  lifted to `layer x bridge_height`.
- Ids (and any attribute) on air were given weight 0 and lost: id layers
  count on every voxel.
- `Parents.reduce`, `learn.tokens` and `fill.carry` averaged every column,
  ids included: ids are now refused as numbers (`reduce` mean/sum,
  `tokens`) or moved as categories (`reduce(how="same")`, `carry`).
- Building heights read with `top_of` counted the ground layer (1 m too
  high): `top_of` now says it counts from the bottom of the box, and
  `height_of` gives heights above the ground.
- Object-dtype cell columns were saved but could not be loaded: text is now
  saved as text, other Python objects are refused with a clear error.
- `LinearOctree` had no CRS: it now has `crs` (saved, and taken from the
  source when built).

### Changed

- **pyarrow is now a required dependency** (for the object table).
- Two adjacent runs of an Occupancy may now have the same label when their
  ids differ.
- `CityBuilder` paints every shape on its own with the centre rule; the
  `solids` and `cover` attributes are gone. Results differ slightly from 0.3
  at footprint edges (no extra PIL outline ring).
- `Features` has a new last field `min_height`.

## 0.3.0

### Changed (breaking)

- Renamed to match the words of the City Tokens paper, where a cell carries
  semantic occupancy (which voxels are filled, and by what) and thematic
  attributes: `ColumnCity` is now `Occupancy`, `ColumnSource` is now
  `OccupancySource`, and the module `octreecity.columns` is now
  `octreecity.occupancy`. The old names are removed. Saved `.npz` files are
  unchanged and load as before.

## 0.2.0

A general core for adaptive cells as the basic unit of 3D space for AI. The
city tools of 0.1 are unchanged and now sit on top of it.

### New

- **Classes** (`octreecity.classes`): a tree of class names you define, as in
  CityGML (building → wall, roof → window). Class 0 is "empty"; the core knows
  no other class. The city names (air, ground, building, tree, water, grass,
  pavement, road) are a preset in `octreecity.urban`.
- **Sources** (`octreecity.sources`): anything that answers "what is inside
  this box?". `GridSource` (voxel grid), `ColumnSource` (runs per column:
  height maps, canopies over air, bridges), `SparseSource` (occupied voxels;
  `from_points` for point clouds, `from_surfaces` for 3D polygons such as
  CityGML surfaces or mesh faces), or your own class with one method.
- **Build** (`octreecity.build`): top-down, split by a rule you choose;
  `uniform()` for the uniform grid over the same data.
- **Rules** (`octreecity.rules`): `mixed` (classes, at any depth of the class
  tree), `varies` and `changes` (attributes), `larger_than`, `where` (any
  function), combined with `|`, `&`, `~`.
- **Cell table**: `tree.cells` (static columns), `tree.series` with
  `add_snapshot` (values over time on the same tree), `tree.classes`; all are
  saved with the tree. `tree.parents()` gives the parent cells, derived from
  the cell codes, with `reduce` to summarise any column over them.
- **Fill** (`octreecity.fill`): per-cell values from rasters, 3D grids,
  functions or any source; `carry` moves columns onto another tree.
- **Links** (`octreecity.links`): "cell A relates to cell B" with a kind and
  values. Built in: `neighbours` (faces, edges, corners), `hierarchy`
  (parents, ancestors, children), `across` (the same-size cell across each
  face), `within` (distance). Any other rule is yours.
- **Export for learning** (`octreecity.learn`): `tokens` gives per-cell rows
  (features, position, size, targets) and fixed-width key tables with
  relative offsets; `pixel_map`, `Tokens.concat`, `to_torch`, `save`,
  `reach_m`.
- **Measurements** (`octreecity.measure`): cells, time and memory against a
  uniform grid as the smallest cell shrinks.
- **Attention model** (`octreecity.attention`, `pip install
  octreecity[torch]`): a general link-attention model over any tokens.
- `features.read_citygml_surfaces`: 3D CityGML surfaces with their semantic
  kind (wall, roof, ground ...) and the object's attributes.
- `LinearOctree.in_box`, `centers`, `sizes_m`; the CLI `--region` now also
  crops a single city (`.npz`).

### Changed

- `ColumnCity.to_octree` is built top-down from the runs (same cells, faster);
  its `strip` argument is gone.
- The 2D readers moved to `octreecity.features`; `octreecity.sources` still
  re-exports them, so `from octreecity.sources import read_citygml` works.
- The label constants moved to `octreecity.urban`. `from octreecity import
  BUILDING` and `from octreecity.builder import GRASS` still work;
  `from octreecity.octree import BUILDING` does not.
- `face_pairs` moved to `octreecity.links` (still importable from
  `octreecity.adapt`). `adapt` keeps the tree's classes, and `transfer` skips
  NaN values.
- Labels can be `uint16` (more than 254 classes).

### Checked against the paper experiments

`scripts/check_paper_cells.py`, `check_paper_3d.py` and `check_paper_infill.py`
rebuild the token sets of the experiments in octreecity-paper-code exactly
(all 334 Daejeon tiles), and `train_parents_lib.py` reproduces the
parent-links result with the library's model.

### Checked at city scale

All 3,475 Osaka tiles (891 million cells): the same cells as 0.1.2, built in
155 s instead of 210 s. Cells, parents, links and tokens for every tile:
about 1 billion tokens, under 2 hours, 4 GB peak memory. One tree over
4 km x 4 km at 1 m (16 x 16 tiles): 87 million cells instead of 4.3 billion
uniform ones, built in 28 s.

## 0.1.2

Octrees for any city size (air padding), `city_cache` for `city_from_osm`,
wire view limit 40k, save-and-reuse docs.
