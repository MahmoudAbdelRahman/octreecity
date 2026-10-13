# Octrees and adaptation

## From columns to octree

```python
tree = city.to_octree(max_level=5)    # cells of 1, 2, 4, ..., 32 voxels
```

`max_level` sets the largest cell (2^max_level voxels). The box sides must be
multiples of that size; pad with air if needed. Or start from height maps and
let OctreeCity pick the sizes in metres:

```python
from octreecity import build_city
tree = build_city(heights, pixel_size=1.0, min_cell=1, max_cell=32)
```

`min_cell` is the finest cell (the raster is resampled to it), `max_cell` the
largest; their ratio must be a power of two.

## Refining for a purpose

```python
tree.limit_size_near([2, 1], distance=8, max_level=1)   # <= 2 m within 8 m of buildings/ground
tree.balance(26)                                         # neighbours differ by at most one size
tree.refine_where(lambda t: my_mask(t))                  # any rule of your own
```

## adapt(): follow any field

Split where a field changes between neighbouring cells, merge where it is flat,
the way adaptive CFD codes do. The field can be anything: wind, sun hours,
noise, temperature, an ML prediction.

```python
from octreecity.adapt import adapt, sample

values = sample(tree, field)                 # field(x, y, z) in metres -> value
for _ in range(4):
    tree, values = adapt(tree, values, refine_tol=1.0, min_level=0, max_cells=5_000_000)
    values = sample(tree, field)             # or: solve again on the new cells
```

- Error indicator: largest jump to a face neighbour.
- Split above `refine_tol` (plus `buffer` layers of neighbours); merge groups
  of 8 when all are below `coarsen_tol` (default `refine_tol / 2`).
- Geometry never changes: only `fluid` labels (default air) are split, and
  only cells with the same label merge.
- Cell sizes stay between `min_level` and `max_level`; the result is 2:1
  balanced; values are carried over (children inherit, merges average by volume).

Sharp fields (like shadow edges) keep refining towards the minimum size: use
`max_cells` or a looser tolerance to bound the cell count.
