"""CFD plugin: write fluid cells of a 2:1 balanced octree as an OpenFOAM polyMesh.

Fluid = leaves with a label in `fluid` (air). Other leaves are solids whose
faces become wall patches (one patch per label); the box sides become six
patches (xmin, xmax, ymin, ymax, zmin, zmax). Where a big cell meets four
small ones its face is split into four, and every face lists the hanging
points on its edges, so cells are closed polyhedra as OpenFOAM expects
(same idea as snappyHexMesh output).

Only the mesh is written (constant/polyMesh); case setup is up to the user.
"""

from pathlib import Path

import numpy as np

from .octree import AIR

# per direction: axis, side, and the face's 4 corner offsets (unit cube),
# ordered counter-clockwise seen from outside (normal = +side * axis)
_FACES = {
    (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, +1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (1, +1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (2, +1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}
_SIDE_NAMES = {(0, -1): "xmin", (0, 1): "xmax", (1, -1): "ymin", (1, 1): "ymax", (2, -1): "zmin", (2, 1): "zmax"}


def build_polymesh(tree, fluid=(AIR,), label_names=None):
    """Return dict(points, faces, owner, neighbour, patches) in OpenFOAM order.

    points: (P, 3) float metres. faces: list of int arrays. owner/neighbour:
    int arrays (neighbour only for internal faces). patches: list of
    (name, start, n).
    """
    from .builder import LABELS

    names = label_names or LABELS
    isf = np.isin(tree.label, fluid)
    fid = -np.ones(tree.n_leaves, np.int64)
    fid[isf] = np.arange(isf.sum())
    X, Y, Z = tree.corners()
    S = tree.sizes()
    shape = np.array(tree.shape)
    cells = np.nonzero(isf)[0]

    F_own, F_nb, F_dir, F_x, F_y, F_z, F_s, F_patch = [], [], [], [], [], [], [], []

    def emit(own, nb, d, x, y, z, s, patch):
        F_own.append(own); F_nb.append(nb); F_dir.append(np.full(len(own), d))
        F_x.append(x); F_y.append(y); F_z.append(z); F_s.append(s); F_patch.append(patch)

    dirs = list(_FACES)
    for d, (axis, side) in enumerate(dirs):
        c = cells
        x, y, z, s = X[c], Y[c], Z[c], S[c]
        p = [x.copy(), y.copy(), z.copy()]
        p[axis] = p[axis] + np.where(side > 0, s, -1)
        outside = (p[axis] < 0) | (p[axis] >= shape[axis])
        # box boundary: whole face
        o = outside
        emit(fid[c[o]], np.full(o.sum(), -1), d, x[o], y[o], z[o], s[o], np.full(o.sum(), -(d + 1)))
        ins = ~outside
        c, x, y, z, s = c[ins], x[ins], y[ins], z[ins], s[ins]
        q = [v[ins] for v in p]
        nb = tree.locate(*q)
        bigger_or_same = tree.level[nb] >= tree.level[c]
        # case A: neighbour same size or bigger -> whole face
        a = bigger_or_same
        nbf = isf[nb]
        same = tree.level[nb] == tree.level[c]
        internal = a & nbf & (~same | (c < nb))
        emit(fid[c[internal]], fid[nb[internal]], d, x[internal], y[internal], z[internal], s[internal],
             np.zeros(internal.sum(), np.int64))
        wall = a & ~nbf
        emit(fid[c[wall]], np.full(wall.sum(), -1), d, x[wall], y[wall], z[wall], s[wall],
             tree.label[nb[wall]].astype(np.int64) + 1000)
        # case B: smaller neighbours -> four quarter faces; fluid quarters are
        # emitted by the small cells (case A on their side), so only walls here
        b = ~bigger_or_same
        if b.any():
            cb, xb, yb, zb, sb = c[b], x[b], y[b], z[b], s[b]
            h = sb // 2
            others = [ax for ax in range(3) if ax != axis]
            for du in (0, 1):
                for dv in (0, 1):
                    qq = [xb.copy(), yb.copy(), zb.copy()]
                    qq[axis] = qq[axis] + np.where(side > 0, sb, -1)
                    qq[others[0]] = qq[others[0]] + du * h
                    qq[others[1]] = qq[others[1]] + dv * h
                    nq = tree.locate(*qq)
                    w = ~isf[nq]
                    # quarter face of the big cell: corner of the quarter, size h
                    fx, fy, fz = xb.copy(), yb.copy(), zb.copy()
                    corner = [fx, fy, fz]
                    corner[others[0]] = corner[others[0]] + du * h
                    corner[others[1]] = corner[others[1]] + dv * h
                    if side > 0:
                        corner[axis] = corner[axis] + sb - h      # quarter cube adjacent to the face
                    emit(fid[cb[w]], np.full(w.sum(), -1), d, corner[0][w], corner[1][w], corner[2][w], h[w],
                         tree.label[nq[w]].astype(np.int64) + 1000)

    own = np.concatenate(F_own); nbr = np.concatenate(F_nb); dd = np.concatenate(F_dir)
    fx = np.concatenate(F_x); fy = np.concatenate(F_y); fz = np.concatenate(F_z)
    fs = np.concatenate(F_s); patch = np.concatenate(F_patch)

    # orientation for internal faces: owner = lower cell index, normal owner -> neighbour
    swap = (nbr >= 0) & (nbr < own)
    own2 = np.where(swap, nbr, own)
    nbr2 = np.where(swap, own, nbr)
    flip = swap.copy()

    # corners of every face (integer voxel coordinates)
    off = np.array([_FACES[dirs[k]] for k in range(6)])          # (6, 4, 3)
    base = np.stack([fx, fy, fz], 1)
    corners = base[:, None, :] + off[dd] * fs[:, None, None]       # (F, 4, 3)
    corners[flip] = corners[flip][:, ::-1]

    # hanging points: edge midpoints that are corners of other faces
    ny1, nz1 = 2 * shape[1] + 1, 2 * shape[2] + 1
    key = lambda p: ((2 * p[..., 0]) * ny1 + 2 * p[..., 1]) * nz1 + 2 * p[..., 2]
    ck = key(corners)
    allk = np.unique(ck)
    mids = (corners + np.roll(corners, -1, axis=1)) / 2.0          # midpoint of edge k -> k+1
    mk = key(mids)
    integer_mid = (fs[:, None] >= 2) & np.ones((1, 4), bool)
    has_mid = integer_mid & np.isin(mk, allk)

    pts_keys = np.unique(np.concatenate([allk, mk[has_mid]]))
    face_lists = []
    cidx = np.searchsorted(pts_keys, ck)
    midx = np.searchsorted(pts_keys, mk)
    for f in range(len(ck)):
        lst = []
        for k in range(4):
            lst.append(cidx[f, k])
            if has_mid[f, k]:
                lst.append(midx[f, k])
        face_lists.append(np.array(lst, np.int64))

    # decode point keys to coordinates (metres)
    kz = pts_keys % nz1
    ky = (pts_keys // nz1) % ny1
    kx = pts_keys // (nz1 * ny1)
    h = tree.voxel_size
    ox, oy, oz = tree.origin
    points = np.stack([ox + h * kx / 2.0, oy + h * ky / 2.0, oz + h * kz / 2.0], 1)

    # order: internal faces sorted by (owner, neighbour), then patches
    internal = nbr2 >= 0
    order_int = np.nonzero(internal)[0]
    order_int = order_int[np.lexsort((nbr2[order_int], own2[order_int]))]
    groups = []
    for d in range(6):
        groups.append((_SIDE_NAMES[dirs[d]], np.nonzero(~internal & (patch == -(d + 1)))[0]))
    for lab in np.unique(patch[patch >= 1000]):
        groups.append((names.get(int(lab - 1000), f"label{lab - 1000}"), np.nonzero(patch == lab)[0]))
    order = [order_int]
    patches, start = [], len(order_int)
    for name, idx in groups:
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(own2[idx], kind="stable")]
        order.append(idx)
        patches.append((name, start, len(idx)))
        start += len(idx)
    order = np.concatenate(order)
    return dict(points=points, faces=[face_lists[k] for k in order], owner=own2[order],
                neighbour=nbr2[order][: len(order_int)], patches=patches, n_cells=int(isf.sum()))


def check_polymesh(m):
    """Closedness and volume checks (no OpenFOAM needed). Returns a dict."""
    P = m["points"]
    n = m["n_cells"]
    area = np.zeros((n, 3))
    vol = np.zeros(n)
    nint = len(m["neighbour"])
    for f, (fl, o) in enumerate(zip(m["faces"], m["owner"])):
        p = P[fl]
        c = p.mean(0)
        a = 0.5 * np.cross(p - c, np.roll(p, -1, 0) - c).sum(0)
        area[o] += a
        vol[o] += a @ c / 3
        if f < nint:
            nb = m["neighbour"][f]
            area[nb] -= a
            vol[nb] -= a @ c / 3
    scale = np.abs(vol) ** (2 / 3) + 1e-12
    return dict(cells=n, faces=len(m["faces"]), points=len(P),
                max_open=float((np.linalg.norm(area, axis=1) / scale).max()),
                min_volume=float(vol.min()), total_volume=float(vol.sum()))


def write_polymesh(m, case_dir):
    """Write constant/polyMesh/{points,faces,owner,neighbour,boundary} (ASCII)."""
    d = Path(case_dir) / "constant" / "polyMesh"
    d.mkdir(parents=True, exist_ok=True)

    def header(cls, obj, note=""):
        return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                f"    class       {cls};\n    location    \"constant/polyMesh\";\n"
                f"    object      {obj};\n{note}}}\n\n")

    with open(d / "points", "w") as f:
        f.write(header("vectorField", "points"))
        f.write(f"{len(m['points'])}\n(\n")
        np.savetxt(f, m["points"], fmt="(%.6g %.6g %.6g)")
        f.write(")\n")
    with open(d / "faces", "w") as f:
        f.write(header("faceList", "faces"))
        f.write(f"{len(m['faces'])}\n(\n")
        f.write("\n".join(f"{len(fl)}(" + " ".join(map(str, fl)) + ")" for fl in m["faces"]))
        f.write("\n)\n")
    note = (f"    note        \"nPoints:{len(m['points'])} nCells:{m['n_cells']} "
            f"nFaces:{len(m['faces'])} nInternalFaces:{len(m['neighbour'])}\";\n")
    for name, arr in (("owner", m["owner"]), ("neighbour", m["neighbour"])):
        with open(d / name, "w") as f:
            f.write(header("labelList", name, note))
            f.write(f"{len(arr)}\n(\n")
            f.write("\n".join(map(str, arr.tolist())))
            f.write("\n)\n")
    with open(d / "boundary", "w") as f:
        f.write(header("polyBoundaryMesh", "boundary"))
        f.write(f"{len(m['patches'])}\n(\n")
        for name, start, n in m["patches"]:
            kind = "patch" if name in _SIDE_NAMES.values() else "wall"
            f.write(f"    {name}\n    {{\n        type            {kind};\n"
                    f"        nFaces          {n};\n        startFace       {start};\n    }}\n")
        f.write(")\n")
    return d
