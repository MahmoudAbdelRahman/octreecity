# How OctreeCity works

A city is mostly vertical walls, flat roofs and layers. OctreeCity keeps the
same voxels in two forms, each suited to a different job.

## Columns, for storage

Each vertical column of cells keeps a short list of **runs**: a label and the
height where it stops, from the bottom up.

```
ground   up to  1 m
building up to 67 m
air      up to 90 m
```

A column can hold any number of runs, so a tree canopy with air beneath it, a
bridge or an overhang is stored exactly. The size of the model then follows
the outlines of buildings rather than their volume: a wall is stored once per
column, not once per cell. This is what makes 1 m detail practical for a whole
city (all of Osaka, 228 km², fits in 37 MB).

Columns are fast to build from footprints and height rasters, fast to query
(the label at any point is a binary search), and convert to and from dense
voxel grids exactly. See [file formats](../reference/file-formats.md) for the layout.

## Octrees, for computation

Some applications want cells of different sizes: fine where detail matters,
coarse in open air. An **octree** divides space into cubes of 1, 2, 4, 8 ...
cells, splitting only where needed. OctreeCity builds one from the columns on
demand:

- following geometry, so cells stay small near buildings and the ground, or
- following any field you provide (wind, sun hours, noise, a model output)
  with `adapt()`, the split-and-merge idea used by adaptive CFD codes.

Neighbouring cells differ by at most one size step (the 2:1 rule), which most
solvers and many graph methods expect. See [adaptive cells](adaptive-cells.md).

## Labels

Every cell carries a label: air, ground, building, tree, water, grass, path,
road, or your own classes (any value up to 254). Analyses use labels to decide
what blocks light, what counts as ground, and which cells are fluid.

## Units and coordinates

All lengths are metres in a projected coordinate system (for example UTM or a
national grid). `city.origin` is the position of the model's corner and
`city.voxel_size` the cell size.
