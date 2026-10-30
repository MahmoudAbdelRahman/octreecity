# File formats

## Occupancy (`.npz`)

| array | type | meaning |
|---|---|---|
| `shape` | int | nx, ny, nz (voxels) |
| `counts` | uint16 | runs per column, column c = x * ny + y |
| `top` | uint16 | exclusive top of each run (voxels), bottom-up per column |
| `label` | uint8 | label of each run |
| `voxel_size` | float | metres |
| `origin` | float[3] | metres, corner (x0, y0, z0) |
| `crs` | str | e.g. EPSG:32652 (may be empty) |
| `class_names`, `class_parent` | str, int | optional class tree (see `octreecity.classes`) |
| `id__<layer>__values`, `id__<layer>__codes` | int64, uint8/16/32 | object id of each run in an id layer: `values[codes]` (0.4) |
| `__meta__` | JSON text | format, version, crs, voxel size, origin, units, id layers (0.4) |
| `__objects__` | bytes | the object table as parquet, when one is attached (0.4) |

Runs start at z = 0, the last run of a column ends at nz, adjacent runs
differ in their label or in at least one id. Files from 0.3 load as before;
0.3 reads 0.4 files too (it ignores the new keys).

## Tiled city (`.ocz`)

A zip: `meta.json` (format, version, x0, y0, z0, tile, cell, crs, tile list
with nz, run counts and id layers) and `t/<i>_<j>/{counts,top,label}.npy`
(plus `id__<layer>.npy` from 0.4) per tile.

## Octree (`.npz`, `LinearOctree.save`)

`level` and `label` per leaf (2 bytes) plus `shape`, `max_level`,
`voxel_size`, `origin`. Leaves are in Morton order and tile each root cube, so
their positions are rebuilt from the levels. The cell table follows in the
same order: `cell__<name>` per static column, `series__<name>` (leaves x
snapshots) per dynamic value with `times`, and the optional class tree
(`class_names`, `class_parent`). From 0.4: id columns as
`cell__<name>__values` / `cell__<name>__codes` (exact int64), `__meta__` (crs,
id columns, units) and `__objects__` (the object table). Text columns are
saved as text; columns of other Python objects are refused when saving.

## Object table (`.parquet`, `ObjectTable.save`)

One row per object, sorted by id: `id` (int64), `class`, `source` (text),
`parent` (int64, -1 = none), `members` (list of int64, only on overlap rows),
then the user columns. Units and descriptions are in the schema metadata
(`octreecity:columns`). pandas, duckdb, QGIS and any parquet reader open it.
Reserved values in id columns: -1 = no object, -2 = mixed.

## Folder store (`city.octreecity/`, `octreecity.store.Store`)

```
city.octreecity/
  manifest.json                        written last; only what it lists is valid
  objects/part-<unique>.parquet        object table parts (merged on reading)
  tiles/<i>_<j>/occupancy-<unique>.npz an Occupancy (above)
  tiles/<i>_<j>/octree-<unique>.npz    an octree (above)
```

`manifest.json`: format, version, generation, crs, voxel_size, max_level,
tile_m (rounded up to a multiple of the biggest cell), origin, id layers, the
object parts (with `mode` "add" or "edit"), the tiles, and column units and
descriptions. A `.zip` of the live files and the manifest is a read-only
snapshot (`Store.snapshot`).

## Tokens (`.npz`, `octreecity.learn.Tokens.save`)

Per token: `features`, `position`, `size`, `level`, `is_leaf`, `source`,
`targets`; per token and key: `idx`, `valid`, `kind`, `offset`; plus
`feature_names`, `target_names`, `kinds`.
