# OctreeCity

A sparse, adaptive octree representation of labelled city voxels: a lighter,
application-independent alternative to dense VoxCity-style grids. One
structure for storage, queries, CFD / ML meshes and visualization.

## Status (12 Oct 2026): proof of concept

`src/octreecity/octree.py`: linear (pointerless) octree.
- Leaves are aligned cubes of side 2**level voxels with one uint8 label
  (air 0, ground 1, building 2, tree 3 by default; any labels < 255 work).
- Sorted by Morton code: point location is a binary search, and any aligned
  cube is a contiguous slice.
- On disk, each leaf is stored as just (level, label), 2 bytes. Codes are
  rebuilt from the order.
- Non-cubic boxes: nx, ny, nz must be multiples of the root size 2**max_level.
- Builders: `from_dense(grid)` (bottom-up merge) and `from_heightfield(ground,
  surface, class)` (top-down with min/max pyramids, no dense grid; the
  result is identical, which the tests check).
- Adaptivity: `split`, `refine_where(predicate)`, `limit_size_near(labels,
  distance, max_level)` (small cells near surfaces), `balance(6|18|26)`
  (2:1 rule for solvers).
- `export.py`: legacy VTK hexahedra for ParaView, matplotlib slice plots.

## First benchmark (20 Osaka tiles, 256 x 256 m, 1 m voxels)

`scripts/bench_osaka.py` reads `urban_wind/data/cities/osaka/height_pictures.npz`
(read only). It uses 1 voxel of ground, buildings on top, and the tallest
building + 16 m of air, rounded up to 32 m.

| | total over 20 tiles |
|---|---|
| dense uint8 in memory | 113 MB |
| octree in memory (10 B/leaf) | 55 MB (20x fewer elements) |
| dense, npz compressed | 0.66 MB |
| octree, npz compressed (level + label) | 0.58 MB |
| height map, npz (2.5D lower bound) | 0.09 MB |
| build from height map | 19 ms / tile |
| locate 1M random points | 23 ms (dense indexing: 5 ms) |
| meshing octree (air <= 2 m within 8 m of solid, 2:1) | ~430k cells / tile, 1.5 s |

What this says (honest reading):
- The gain grows with empty space: 4x smaller than compressed dense for
  the 328 m tower tile, about equal for low-rise tiles.
- About 80% of leaves are 1 m cubes. Building walls at 1 m resolution,
  with footprints and roof heights not aligned to powers of two, force the
  finest level along every wall, over its full height. An octree pays
  O(wall area); a 2.5D column format pays O(footprint perimeter).
- Ground alignment matters: a 1-voxel ground slab (air starting at z = 1)
  makes ~1.5-2x more leaves than no ground voxels or a 2-voxel slab
  (tile 2311: 259k vs 133k vs 160k leaves).

## Open design questions

1. Pure octree vs hybrid: quadtree in x, y with exact z spans per column
   (much smaller for 2.5D cities, but less uniform for meshing), or
   anisotropic leaves (tall cells) inside the octree.
2. Should building interiors be one "solid, don't care" label that is
   allowed to merge freely (most applications never look inside)?
3. Alignment: snap footprints and heights to the grid, or allow
   sub-voxel geometry in boundary leaves (cut cells)?
4. Baselines to compare against: VoxCity itself, OpenVDB, sparse voxel
   DAGs.

## Run

```
uv sync
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_osaka.py 20
.venv/bin/python scripts/plot_slice.py 868
```
