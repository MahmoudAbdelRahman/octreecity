# Modules and main functions

| Module | What it contains |
|---|---|
| `octreecity.easy` | `city_from_osm(lat, lon, size_m)`: one call, any place, data from OpenStreetMap |
| `octreecity.sources` | `read_citygml`, `read_geojson`, `read_osm_overpass`, `Features`, `reproject` |
| `octreecity.builder` | `CityBuilder`: terrain, land cover, trees, buildings; `rasterize`; label constants |
| `octreecity.columns` | `ColumnCity`: the storage format, `crop`, `to_octree`, `to_dense`, `save` / `load` |
| `octreecity.tiles` | `build_tiled`, `TiledCity` (`tile`, `region`), `TiledCityWriter` |
| `octreecity.octree` | `LinearOctree`: point location, `balance`, `limit_size_near`, `refine_where` |
| `octreecity.city` | `build_city(heights, pixel_size, min_cell, max_cell)` |
| `octreecity.adapt` | `adapt`, `sample`, `transfer`, `face_pairs`, `jump_indicator` |
| `octreecity.analysis` | `street_points`, `sky_view_factor`, `shadow`, `sun_position`, `sun_hours` |
| `octreecity.viz` | `show`, `show_values`, `show_octree`, `show_cells`, `setup_notebook` |
| `octreecity.render` | `render_top`, `render_section`, `plot_slice` (images) |
| `octreecity.exports` | `write_obj`, `surface_quads`, `tensors_2d`, `tensors_3d`, `write_vtk` |
| `octreecity.foam` | `build_polymesh`, `check_polymesh`, `write_polymesh` (OpenFOAM) |
| `octreecity.cli` | the `octreecity` command |

Every function has a docstring: `help(octreecity.columns.ColumnCity)` in Python.
