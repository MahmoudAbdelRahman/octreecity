# Modules and main functions

Core (no domain words):

| Module | What it contains |
|---|---|
| `octreecity.classes` | `Classes`: the tree of class names (semantic structure), `EMPTY` |
| `octreecity.sources` | `GridSource`, `OccupancySource`, `SparseSource` (`from_points`, `from_surfaces`), `Source`, `Summary` |
| `octreecity.build` | `build(source, rule, max_level, min_level)`, `uniform(source, level)` |
| `octreecity.rules` | `mixed`, `varies`, `changes`, `larger_than`, `where`, `never`; combine with `\|` `&` `~` |
| `octreecity.octree` | `LinearOctree`: cell table (`label`, `cells`, `series`, `add_snapshot`), `id_columns`, `crs`, `objects`, `parents` (`reduce`, `how="same"` for ids), `in_box`, `balance`, `refine_where`, `select`, `touching`, `near` |
| `octreecity.objects` | `ObjectTable` (`add`, `add_overlap`, `merge`, `select`, `tag`, `parent`, `children`, `siblings`, `ancestors`, `descendants`, `save` / `load`), `object_id`, `IdCollision` |
| `octreecity.query` | `Query`, `Group` (`neighbours`, `within`, `nearest`, `boundary`, `parents`, `siblings`, `objects`, `voxels`), `Nodes`, `neighbour_pairs`, `cube_ids` |
| `octreecity.store` | `Store` (`create`, `open`, `write_tile`, `read_tile`, `read_tree`, `region`, `objects`, `edit_objects`, `build`, `compact`, `snapshot`), `round_tile` |
| `octreecity.fill` | `from_raster`, `from_grid`, `from_function`, `from_source`, `carry` |
| `octreecity.links` | `Links`, `neighbours`, `hierarchy`, `across`, `within`, `face_pairs` |
| `octreecity.learn` | `tokens`, `Tokens` (`concat`, `to_torch`, `save`), `pixel_map`, `batches`, `reach_m` |
| `octreecity.measure` | `growth`, `table`, `timed` |
| `octreecity.attention` | `LinkModel`, `LinkAttention`, `fit`, `predict` (needs `octreecity[torch]`) |

City tools (use the `octreecity.urban` preset of class names):

| Module | What it contains |
|---|---|
| `octreecity.urban` | `CLASSES`, `AIR`, `GROUND`, `BUILDING`, `TREE`, `WATER`, `GRASS`, `PAVEMENT`, `ROAD`, colours |
| `octreecity.easy` | `city_from_osm(lat, lon, size_m)`: one call, any place, data from OpenStreetMap, with object ids; `osm_builder` for one tile |
| `octreecity.features` | `read_citygml` (footprints), `read_citygml_surfaces` (3D surfaces with their kind), `read_geojson`, `read_osm_overpass`, `Features`, `Surfaces`, `reproject` |
| `octreecity.builder` | `CityBuilder`: terrain, land cover, trees, buildings, zones, any shapes, each with ids; `rasterize`, `footprint_pairs` |
| `octreecity.occupancy` | `Occupancy`: the storage format, ids per run (`ids`, `id_at`, `voxels`, `voxel_counts`), `crop`, `concat_x`, `concat_y`, `pad_z`, `height_of`, `to_octree`, `to_dense`, `save` / `load` |
| `octreecity.tiles` | `build_tiled`, `TiledCity` (`tile`, `region`), `TiledCityWriter` |
| `octreecity.city` | `build_city(heights, pixel_size, min_cell, max_cell)` |
| `octreecity.adapt` | `adapt`, `sample`, `transfer`, `transfer_ids`, `jump_indicator` (split and merge by a field, CFD style) |
| `octreecity.analysis` | `street_points`, `sky_view_factor`, `shadow`, `sun_position`, `sun_hours` |
| `octreecity.viz` | `show`, `show_values`, `show_octree`, `show_cells`, `setup_notebook` |
| `octreecity.render` | `render_top`, `render_section`, `plot_slice` (images) |
| `octreecity.exports` | `write_obj`, `surface_quads`, `tensors_2d`, `tensors_3d`, `write_vtk` |
| `octreecity.foam` | `build_polymesh`, `check_polymesh`, `write_polymesh` (OpenFOAM) |
| `octreecity.cli` | the `octreecity` command |

Every function has a docstring: `help(octreecity.occupancy.Occupancy)` in Python.
