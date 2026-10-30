# Changelog

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
