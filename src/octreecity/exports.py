"""Exports: surface meshes (OBJ), octree cells (VTK) and tensors for machine learning.

Meshes: the boundary between labels, as quads merged into strips (walls
along z, roofs and ground along y), one OBJ group and material per label.
Opens in Blender, Rhino, MeshLab, three.js, Unity.

Tensors: 2D channels (heights, label fractions) or 3D label fractions at
any coarser resolution, as float32 arrays ready for a U-Net or a 3D CNN.
"""

from pathlib import Path

import numpy as np

from .octree import AIR

DEFAULT_COLORS = {1: (0.54, 0.48, 0.38), 2: (0.79, 0.80, 0.83), 3: (0.25, 0.54, 0.29), 4: (0.24, 0.50, 0.77),
                  5: (0.53, 0.70, 0.35), 6: (0.72, 0.68, 0.61), 7: (0.36, 0.38, 0.40)}


def _strips(mask, axis):
    """Runs of True along `axis` of a 3D mask: (start index tuple, length)."""
    m = np.moveaxis(mask, axis, -1)
    pad = np.zeros(m.shape[:-1] + (1,), bool)
    a = np.concatenate([pad, m, pad], -1).astype(np.int8)
    d = np.diff(a, axis=-1)
    s = np.nonzero(d == 1)
    e = np.nonzero(d == -1)
    length = e[-1] - s[-1]
    return s, length


def surface_quads(city, labels=None, x0=0, x1=None):
    """Quads (4 corners, in metres) of every face between a solid voxel of a
    label in `labels` and a different label, with the solid's label."""
    x1 = city.shape[0] if x1 is None else x1
    g = city.to_dense(x0, x1)
    labels = [int(v) for v in np.unique(g) if v != AIR] if labels is None else list(labels)
    c = city.voxel_size
    ox, oy, oz = city.origin
    ox += x0 * c
    quads, labs = [], []
    nx, ny, nz = g.shape
    for lab in labels:
        s = g == lab
        # pad with "outside" so boundary faces of the box are not emitted for air,
        # but solids touching the box sides are closed
        p = np.pad(s, 1, constant_values=False)
        gp = np.pad(g, 1, constant_values=AIR)
        for axis in range(3):
            for sign in (1, -1):
                nb = np.roll(gp, -sign, axis=axis)
                face = p & (nb != lab)
                face = face[1:-1, 1:-1, 1:-1]
                if axis == 2 and sign == -1:
                    face[:, :, 0] = False          # no faces under the box floor
                run_axis = 1 if axis == 2 else 2   # strips: along y for horizontal faces, along z for walls
                (i, j, k), n = _strips(face, run_axis)
                idx = [None, None, None]
                other = [a for a in range(3) if a != run_axis]
                idx[other[0]], idx[other[1]], idx[run_axis] = i, j, k
                X, Y, Z = (np.asarray(v, float) for v in idx)
                lo = [X.copy(), Y.copy(), Z.copy()]
                lo[axis] = lo[axis] + (1 if sign == 1 else 0)
                # the face spans one cell along the third axis and n along run_axis
                third = [a for a in range(3) if a not in (axis, run_axis)][0]
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    q = [lo[0].copy(), lo[1].copy(), lo[2].copy()]
                    q[third] = q[third] + du
                    q[run_axis] = q[run_axis] + dv * n
                    corners.append(np.stack(q, -1))
                qd = np.stack(corners, 1)
                # outward winding: normal must point along sign * axis
                nrm = np.cross(qd[0:1, 1] - qd[0:1, 0], qd[0:1, 3] - qd[0:1, 0]) if len(qd) else None
                if len(qd) and nrm[0, axis] * sign < 0:
                    qd = qd[:, ::-1]
                quads.append(qd * c + np.array([ox, oy, oz]))
                labs.append(np.full(len(qd), lab, np.uint8))
    if not quads:
        return np.zeros((0, 4, 3)), np.zeros(0, np.uint8)
    return np.concatenate(quads), np.concatenate(labs)


def write_obj(city, path, labels=None, names=None, colors=None, strip=256):
    """Write the label boundaries as an OBJ (+ .mtl). Works in x strips."""
    from .builder import LABELS

    names = names or LABELS
    colors = colors or DEFAULT_COLORS
    path = Path(path)
    mtl = path.with_suffix(".mtl")
    present = set()
    nv = 0
    with open(path, "w") as f:
        f.write(f"# octreecity surface mesh\nmtllib {mtl.name}\n")
        for xs in range(0, city.shape[0], strip):
            q, lab = surface_quads(city, labels, xs, min(city.shape[0], xs + strip))
            for L in np.unique(lab):
                sel = q[lab == L]
                present.add(int(L))
                f.write(f"g {names.get(int(L), f'label{L}')}_{xs}\nusemtl {names.get(int(L), f'label{L}')}\n")
                v = sel.reshape(-1, 3)
                np.savetxt(f, v, fmt="v %.2f %.2f %.2f")
                idx = nv + 1 + np.arange(len(v)).reshape(-1, 4)
                np.savetxt(f, idx, fmt="f %d %d %d %d")
                nv += len(v)
    with open(mtl, "w") as f:
        for L in sorted(present):
            r, g, b = colors.get(L, (0.8, 0.2, 0.2))
            f.write(f"newmtl {names.get(L, f'label{L}')}\nKd {r:.3f} {g:.3f} {b:.3f}\n\n")
    return nv // 4


def tensors_2d(city, labels=None, factor=1):
    """Channels [C, nx/factor, ny/factor] float32 and their names:
    surface height above the lowest ground (m), ground elevation (m), then
    for each label the fraction of each 2D cell it covers (seen from above)
    and its top height above ground (m)."""
    from .analysis import ground_level, surface_height

    g = ground_level(city)
    s = surface_height(city)
    labels = [int(v) for v in np.unique(city.label) if v != AIR] if labels is None else labels
    chans, names = [s - g, g - g.min()], ["height", "ground"]
    for lab in labels:
        top = city.origin[2] + city.top_of([lab]) * city.voxel_size
        has = city.top_of([lab]) > 0
        chans += [has.astype(np.float32), np.where(has, top - g, 0)]
        names += [f"frac_{lab}", f"top_{lab}"]
    out = np.stack(chans).astype(np.float32)
    if factor > 1:
        C, nx, ny = out.shape
        out = out[:, :nx // factor * factor, :ny // factor * factor]
        out = out.reshape(C, nx // factor, factor, ny // factor, factor).mean(axis=(2, 4))
    return out, names


def tensors_3d(city, labels, factor=2):
    """Label fractions [len(labels), nx/f, ny/f, nz/f] float32 (each coarse
    voxel: share of its fine voxels with that label)."""
    nx, ny, nz = city.shape
    f = factor
    out = np.zeros((len(labels), nx // f, ny // f, nz // f), np.float32)
    for xs in range(0, nx // f * f, 32 * f):
        xe = min(nx // f * f, xs + 32 * f)
        g = city.to_dense(xs, xe)[:, :ny // f * f, :nz // f * f]
        b = g.reshape((xe - xs) // f, f, ny // f, f, nz // f, f)
        for k, lab in enumerate(labels):
            out[k, xs // f:xe // f] = (b == lab).mean(axis=(1, 3, 5))
    return out


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
