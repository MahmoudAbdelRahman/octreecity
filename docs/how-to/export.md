# Exports

The examples on this page start from a city, for example:

```python
from octreecity.easy import city_from_osm
city = city_from_osm(36.3700, 127.3625, size_m=1000)   # KAIST, Daejeon; any lat, lon works
```

## 3D meshes (OBJ)

```python
from octreecity.exports import write_obj
write_obj(city, "city.obj", labels=[2])     # buildings only; omit labels for all
```

Faces between labels, merged into strips (walls along z, roofs along y), one
group and material per label, outward normals. Opens in Blender, Rhino,
MeshLab, three.js. 512 x 512 m of downtown Osaka: 28,000 faces, 4 MB.

## Tensors for machine learning

```python
from octreecity.exports import tensors_2d, tensors_3d
x, names = tensors_2d(city, factor=2)       # [C, nx/2, ny/2]: height, ground, per-label cover and top
v = tensors_3d(city, labels=[2, 3], factor=4)   # [2, nx/4, ny/4, nz/4] label fractions
```

These are uniform grids, for models that need them (U-Nets, 3D CNNs). To give
a model the adaptive cells themselves, one row per cell and the cells each one
reads, use `octreecity.learn.tokens`; see
[cells for learning](../explanation/cells-for-learning.md).


## OpenFOAM mesh (CFD plugin)

```python
from octreecity.foam import build_polymesh, check_polymesh, write_polymesh

tree = city.crop(372, 372, 256, 256).to_octree(4)   # a 256 m block; sides must divide by the largest cell (16 m)
tree.limit_size_near([1, 2], distance=6, max_level=0)
tree.balance(26)
mesh = build_polymesh(tree)          # fluid = air cells
print(check_polymesh(mesh))          # every cell closed, volumes positive
write_polymesh(mesh, "case/")        # case/constant/polyMesh, binary
```

Walls: one patch per solid label (ground, building, tree...). Box sides:
`xmin xmax ymin ymax zmin zmax`. Where a big cell meets four small ones its
face is split and hanging points are listed, as in snappyHexMesh meshes.
Status: passes the package's own checks; not yet verified with OpenFOAM
`checkMesh`.

## Viewing octrees

`exports.write_vtk(tree, "tree.vtk")` writes hexahedra for ParaView;
`render.plot_slice(tree, ax, axis="y")` draws a section.
