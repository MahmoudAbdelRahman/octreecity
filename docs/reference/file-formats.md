# File formats

## ColumnCity (`.npz`)

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

Runs start at z = 0, the last run of a column ends at nz, adjacent runs differ.

## Tiled city (`.ocz`)

A zip: `meta.json` (format, version, x0, y0, z0, tile, cell, crs, tile list
with nz and run counts) and `t/<i>_<j>/{counts,top,label}.npy` per tile.

## Octree (`.npz`, `LinearOctree.save`)

`level` and `label` per leaf (2 bytes) plus `shape`, `max_level`,
`voxel_size`, `origin`. Leaves are in Morton order and tile each root cube, so
their positions are rebuilt from the levels. The cell table follows in the
same order: `cell__<name>` per static column, `series__<name>` (leaves x
snapshots) per dynamic value with `times`, and the optional class tree
(`class_names`, `class_parent`).

## Tokens (`.npz`, `octreecity.learn.Tokens.save`)

Per token: `features`, `position`, `size`, `level`, `is_leaf`, `source`,
`targets`; per token and key: `idx`, `valid`, `kind`, `offset`; plus
`feature_names`, `target_names`, `kinds`.
