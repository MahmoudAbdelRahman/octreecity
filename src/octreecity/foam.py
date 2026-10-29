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

from .classes import EMPTY

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


def build_polymesh(tree, fluid=(EMPTY,), label_names=None):
    """Return dict(points, faces, owner, neighbour, patches) in OpenFOAM order.

    points: (P, 3) float metres. faces: list of int arrays. owner/neighbour:
    int arrays (neighbour only for internal faces). patches: list of
    (name, start, n).
    """
    from .urban import LABELS

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
    cidx = np.searchsorted(pts_keys, ck)
    midx = np.searchsorted(pts_keys, mk)
    # interleave corner k and (optional) midpoint of edge k -> up to 8 points per face
    pad = np.full((len(ck), 8), -1, np.int64)
    pad[:, 0::2] = cidx
    pad[:, 1::2] = np.where(has_mid, midx, -1)

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
    return dict(points=points, faces=pad[order], owner=own2[order],
                neighbour=nbr2[order][: len(order_int)], patches=patches, n_cells=int(isf.sum()))


def _face_geometry(m):
    """Area vectors and centres of all faces (padded face arrays)."""
    F = m["faces"]
    P = m["points"]
    valid = F >= 0
    n = valid.sum(1)
    p = P[np.where(valid, F, 0)]
    c = (p * valid[..., None]).sum(1) / n[:, None]
    # next valid point of each slot (wrap around), padded slots contribute 0
    order = np.argsort(~valid, axis=1, kind="stable")
    q = np.take_along_axis(p, order[..., None], 1)
    k = np.arange(F.shape[1])[None]
    nxt = np.take_along_axis(q, np.where(k + 1 < n[:, None], k + 1, 0)[..., None], 1)
    tri = 0.5 * np.cross(q - c[:, None], nxt - c[:, None])
    tri *= (k < n[:, None])[..., None]
    return tri.sum(1), c


def check_polymesh(m):
    """Closedness and volume checks (no OpenFOAM needed). Returns a dict."""
    n = m["n_cells"]
    a, c = _face_geometry(m)
    own, nb = m["owner"], m["neighbour"]
    nint = len(nb)
    av = np.einsum("ij,ij->i", a, c) / 3
    acc = lambda w: (np.bincount(own, w, minlength=n)
                     - np.bincount(nb, w[:nint], minlength=n))
    area = np.stack([acc(a[:, k]) for k in range(3)], 1)
    vol = acc(av)
    scale = np.abs(vol) ** (2 / 3) + 1e-12
    return dict(cells=n, faces=len(m["faces"]), points=len(m["points"]),
                max_open=float((np.linalg.norm(area, axis=1) / scale).max()),
                min_volume=float(vol.min()), total_volume=float(vol.sum()))


def write_polymesh(m, case_dir, binary=True):
    """Write constant/polyMesh/{points,faces,owner,neighbour,boundary}.

    binary=True writes OpenFOAM binary lists (32-bit labels, 64-bit scalars,
    faces as faceCompactList): fast and about half the size of ASCII.
    """
    d = Path(case_dir) / "constant" / "polyMesh"
    d.mkdir(parents=True, exist_ok=True)
    fmt = "binary" if binary else "ascii"
    F = m["faces"]
    valid = F >= 0
    flat = F[valid].astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(valid.sum(1))]).astype(np.int32)

    def header(cls, obj, note="", arch=True):
        a = '    arch        "LSB;label=32;scalar=64";\n' if binary and arch else ""
        return ("FoamFile\n{\n    version     2.0;\n"
                f"    format      {fmt};\n{a}    class       {cls};\n"
                f"    location    \"constant/polyMesh\";\n    object      {obj};\n{note}}}\n\n")

    def write_list(f, arr, ascii_fmt):
        f.write(f"{len(arr)}\n(".encode())
        if binary:
            f.write(np.ascontiguousarray(arr).tobytes())
            f.write(b")\n")
        else:
            f.write(b"\n")
            np.savetxt(f, arr, fmt=ascii_fmt)
            f.write(b")\n")

    note = (f"    note        \"nPoints:{len(m['points'])} nCells:{m['n_cells']} "
            f"nFaces:{len(F)} nInternalFaces:{len(m['neighbour'])}\";\n")
    with open(d / "points", "wb") as f:
        f.write(header("vectorField", "points").encode())
        write_list(f, m["points"].astype("<f8"), "(%.6g %.6g %.6g)")
    with open(d / "faces", "wb") as f:
        if binary:
            f.write(header("faceCompactList", "faces").encode())
            write_list(f, offsets.astype("<i4"), "%d")
            f.write(b"\n")
            write_list(f, flat.astype("<i4"), "%d")
        else:
            f.write(header("faceList", "faces").encode())
            f.write(f"{len(F)}\n(\n".encode())
            f.write("\n".join(f"{v.size}(" + " ".join(map(str, v)) + ")"
                              for v in np.split(flat, offsets[1:-1])).encode())
            f.write(b"\n)\n")
    for name, arr in (("owner", m["owner"]), ("neighbour", m["neighbour"])):
        with open(d / name, "wb") as f:
            f.write(header("labelList", name, note).encode())
            write_list(f, np.asarray(arr).astype("<i4"), "%d")
    with open(d / "boundary", "w") as f:
        f.write(header("polyBoundaryMesh", "boundary", arch=False).replace(f"format      {fmt}", "format      ascii"))
        f.write(f"{len(m['patches'])}\n(\n")
        for name, start, n in m["patches"]:
            kind = "patch" if name in _SIDE_NAMES.values() else "wall"
            f.write(f"    {name}\n    {{\n        type            {kind};\n"
                    f"        nFaces          {n};\n        startFace       {start};\n    }}\n")
        f.write(")\n")
    return d
