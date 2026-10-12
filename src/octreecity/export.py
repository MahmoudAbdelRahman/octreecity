"""Export leaves for viewing: legacy VTK hexahedra (ParaView) and slice plots."""

import numpy as np


def write_vtk(tree, path, labels=None):
    """Write leaves as VTK_HEXAHEDRON cells with 'label' and 'level' data.

    labels: optional list of labels to keep (e.g. only buildings, or only
    air to inspect the adaptive mesh). Points are not shared between cells.
    """
    keep = np.ones(tree.n_leaves, bool) if labels is None else np.isin(tree.label, labels)
    x, y, z = (a[keep].astype(np.float64) for a in tree.corners())
    s = tree.sizes()[keep].astype(np.float64)
    h = tree.voxel_size
    ox, oy, oz = tree.origin
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float64)
    pts = np.empty((x.size, 8, 3), np.float64)
    pts[..., 0] = ox + h * (x[:, None] + corner[:, 0] * s[:, None])
    pts[..., 1] = oy + h * (y[:, None] + corner[:, 1] * s[:, None])
    pts[..., 2] = oz + h * (z[:, None] + corner[:, 2] * s[:, None])
    n = x.size
    cells = np.hstack([np.full((n, 1), 8), np.arange(8 * n).reshape(n, 8)]).astype(">i4")
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\noctreecity leaves\nBINARY\n"
                b"DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {8 * n} float\n".encode())
        f.write(pts.reshape(-1, 3).astype(">f4").tobytes())
        f.write(f"\nCELLS {n} {9 * n}\n".encode())
        f.write(cells.tobytes())
        f.write(f"\nCELL_TYPES {n}\n".encode())
        f.write(np.full(n, 12, ">i4").tobytes())
        f.write(f"\nCELL_DATA {n}\nSCALARS label int 1\nLOOKUP_TABLE default\n".encode())
        f.write(tree.label[keep].astype(">i4").tobytes())
        f.write(b"\nSCALARS level int 1\nLOOKUP_TABLE default\n")
        f.write(tree.level[keep].astype(">i4").tobytes())
        f.write(b"\n")


def plot_slice(tree, ax, axis="y", index=None, colors=None, edge=True):
    """Draw the leaves cut by the plane axis = index (voxel units)."""
    from matplotlib.collections import PolyCollection

    colors = colors or {0: "#f4f6fb", 1: "#9c8b6e", 2: "#5b6b8c", 3: "#4f8f4a"}
    k = "xyz".index(axis)
    if index is None:
        index = tree.shape[k] // 2
    c = tree.corners()
    s = tree.sizes()
    hit = (c[k] <= index) & (index < c[k] + s)
    u, v = [c[i][hit] for i in range(3) if i != k]
    w = s[hit]
    quads = np.stack([np.stack([u, v], 1), np.stack([u + w, v], 1),
                      np.stack([u + w, v + w], 1), np.stack([u, v + w], 1)], 1)
    fc = [colors.get(int(l), "#d33") for l in tree.label[hit]]
    pc = PolyCollection(quads * tree.voxel_size, facecolors=fc,
                        edgecolors="#33333355" if edge else "none", linewidths=0.3)
    ax.add_collection(pc)
    dims = [tree.shape[i] * tree.voxel_size for i in range(3) if i != k]
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_aspect("equal")
    names = [n for n in "xyz" if n != axis]
    ax.set_xlabel(f"{names[0]} (m)")
    ax.set_ylabel(f"{names[1]} (m)")
    return pc
