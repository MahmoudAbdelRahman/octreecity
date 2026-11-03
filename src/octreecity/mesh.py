"""3D models (triangle meshes) as solid voxels.

    m = read_mesh("tower.obj")                      # OBJ, PLY or STL
    m = m.place(at=(x, y), up="y")                  # a local model into the city
    b.add_meshes(m, ids=ids)                        # octreecity.builder.CityBuilder

A Mesh is triangles grouped into parts (objects): one part per building,
one label and one id per part.

Solid fill: for every column of the grid, a vertical ray through the column
centre crosses a closed part an even number of times; the voxels between
crossing 1 and 2, 3 and 4 ... are inside. The pairs are z runs, so leaning
walls, sloped roofs, overhangs (two runs in one column) and hollow shapes
come out as they are. Same rule as extruded shapes: a voxel is filled when
its centre is inside.

Open meshes: holes in walls do not matter (vertical rays never cross a
wall). A column with an odd number of crossings has a hole in a top or
bottom face. It is closed with the direction of the faces (a face seen
from above is a roof: the solid lies below it): a missing floor is closed
at the part's lowest point, a missing roof at its highest point.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# the same off-centre test point as CityBuilder (an irrational ratio), so a
# ray never meets an edge or a vertex exactly
_SHIFT = (1.234567e-7, 2.345679e-7 * np.pi / 3)


@dataclass
class Mesh:
    """Triangles in metres (x east, y north, z up).

    vertices  (V, 3) floats
    faces     (F, 3) vertex indices
    part      (F,) which part (object) each triangle belongs to (default 0)
    names     one name per part (default "0", "1", ...)
    """
    vertices: np.ndarray
    faces: np.ndarray
    part: np.ndarray = None
    names: list = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, np.int64).reshape(-1, 3)
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("faces point to vertices that do not exist")
        n = len(self.faces)
        self.part = np.zeros(n, np.int64) if self.part is None else np.asarray(self.part, np.int64)
        if self.part.shape != (n,):
            raise ValueError(f"part has {self.part.size} values, expected one per face ({n})")
        k = int(self.part.max()) + 1 if n else 0
        if self.names is None:
            self.names = [str(i) for i in range(k)]
        self.names = list(self.names)
        if len(self.names) < k:
            raise ValueError(f"{len(self.names)} names for {k} parts")

    @property
    def n_parts(self):
        return len(self.names)

    def __len__(self):
        return len(self.faces)

    def bounds(self):
        """(min xyz, max xyz)."""
        return self.vertices.min(0), self.vertices.max(0)

    def place(self, at=(0.0, 0.0), z=0.0, rotate=0.0, scale=1.0, up="z"):
        """A copy moved into the city: up="y" first turns a y-up model
        (Blender, glTF, many OBJ exports) to z up; then scale, rotate
        (degrees, counter-clockwise seen from above, about the model's
        (0, 0)) and move the model's (0, 0, 0) to (at[0], at[1], z)."""
        v = self.vertices.copy()
        if up == "y":
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
        elif up != "z":
            raise ValueError("up must be 'z' or 'y'")
        v *= float(scale)
        a = np.deg2rad(rotate)
        c, s = np.cos(a), np.sin(a)
        v[:, :2] = v[:, :2] @ np.array([[c, s], [-s, c]])
        v += (float(at[0]), float(at[1]), float(z))
        return Mesh(v, self.faces, self.part, self.names)

    def select(self, parts):
        """A mesh with only the given parts (numbered from 0 again)."""
        parts = np.atleast_1d(np.asarray(parts, np.int64))
        keep = np.isin(self.part, parts)
        f = self.faces[keep]
        used, inv = np.unique(f, return_inverse=True)
        new = np.full(self.n_parts, -1, np.int64)
        new[parts] = np.arange(parts.size)
        return Mesh(self.vertices[used], inv.reshape(-1, 3), new[self.part[keep]],
                    [self.names[i] for i in parts])

    @staticmethod
    def concat(meshes):
        """One mesh from several; the parts keep their order (the parts of
        the first mesh first)."""
        meshes = list(meshes)
        v, f, p, names, nv, npart = [], [], [], [], 0, 0
        for m in meshes:
            v.append(m.vertices)
            f.append(m.faces + nv)
            p.append(m.part + npart)
            names += m.names
            nv += len(m.vertices)
            npart += m.n_parts
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        return Mesh(np.concatenate(v), np.concatenate(f), np.concatenate(p), names)


# ------------------------------------------------------------------ readers

def read_mesh(path, by="auto"):
    """Read an OBJ, PLY or STL file (by suffix) into a Mesh. Polygons with
    more than three corners are split into triangles.

    by (OBJ only): which lines start a new part: "o" (objects), "g"
    (groups), "auto" ("o" if the file has any, else "g") or None (one
    part). Each part should be one closed solid, so choose the lines that
    split the file into buildings, not into materials.
    PLY and STL files are one part.

    Other formats (glTF, FBX, IFC ...): load them with another package and
    pass the arrays, e.g. t = trimesh.load(path, force="mesh");
    Mesh(t.vertices, t.faces)."""
    path = Path(path)
    s = path.suffix.lower()
    if s == ".obj":
        return _read_obj(path, by)
    if s == ".ply":
        return _read_ply(path)
    if s == ".stl":
        return _read_stl(path)
    raise ValueError(f"unknown mesh format {s!r} (OBJ, PLY or STL); for others load the "
                     "vertices and faces yourself and use Mesh(vertices, faces)")


def _read_obj(path, by):
    if by not in ("auto", "o", "g", None):
        raise ValueError("by must be 'auto', 'o', 'g' or None")
    lines = Path(path).read_text(errors="replace").splitlines()
    if by == "auto":
        by = "o" if any(ln.startswith("o ") for ln in lines) else "g"
    verts, polys, owner, names, index = [], [], [], [], {}
    cur = None
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "v":
            verts.append(t[1:4])
        elif t[0] == "f":
            n = len(verts)
            idx = [int(w.split("/")[0]) for w in t[1:]]
            polys.append([i - 1 if i > 0 else n + i for i in idx])
            if cur is None:
                cur = index.setdefault("", len(index))
            owner.append(cur)
        elif by is not None and t[0] == by:
            name = " ".join(t[1:])
            cur = index.setdefault(name, len(index))
    v = np.array(verts, np.float64).reshape(-1, 3)
    faces, part = _triangulate_faces(v, polys, owner)
    names = list(index)                         # in order of first appearance
    used = np.unique(part)                      # drop names with no faces
    new = np.full(len(names), -1, np.int64)
    new[used] = np.arange(used.size)
    names = [names[i] or path.stem for i in used]
    return Mesh(v, faces, new[part] if part.size else part, names)


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _read_ply(path):
    data = Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"{path} is not a PLY file")
    body = data[data.index(b"\n", end) + 1:]
    fmt, elements = None, []
    for ln in data[:end].decode("ascii", "replace").splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            elements.append((t[1], int(t[2]), []))
        elif t[0] == "property":
            if t[1] == "list":
                elements[-1][2].append((t[4], _PLY_TYPES[t[2]], _PLY_TYPES[t[3]]))
            else:
                elements[-1][2].append((t[2], _PLY_TYPES[t[1]], None))
    if fmt == "ascii":
        values = _ply_ascii(body.decode("ascii", "replace").split(), elements)
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        values = _ply_binary(body, elements, "<" if fmt == "binary_little_endian" else ">")
    else:
        raise ValueError(f"unknown PLY format {fmt!r}")
    vx = values.get("vertex", {})
    v = np.stack([np.asarray(vx[k], np.float64) for k in ("x", "y", "z")], 1)
    fc = values.get("face", {})
    lists = fc.get("vertex_indices", fc.get("vertex_index", []))
    faces, part = _triangulate_faces(v, lists, np.zeros(len(lists), np.int64))
    return Mesh(v, faces, part, [path.stem])


def _ply_ascii(tok, elements):
    out, pos = {}, 0
    for name, count, props in elements:
        if all(p[2] is None for p in props):         # fixed rows: one block
            k = len(props)
            a = np.array(tok[pos:pos + count * k], np.float64).reshape(count, k)
            out[name] = {p[0]: a[:, j] for j, p in enumerate(props)}
            pos += count * k
            continue
        cols = {p[0]: [] for p in props}
        for _ in range(count):
            for pname, _t, it in props:
                if it is None:
                    cols[pname].append(float(tok[pos]))
                    pos += 1
                else:
                    n = int(float(tok[pos]))
                    cols[pname].append([int(float(w)) for w in tok[pos + 1:pos + 1 + n]])
                    pos += 1 + n
        out[name] = cols
    return out


def _ply_binary(body, elements, e):
    out, pos = {}, 0
    for name, count, props in elements:
        if all(p[2] is None for p in props):
            dt = np.dtype([(p[0], e + p[1]) for p in props])
            a = np.frombuffer(body, dt, count, pos)
            out[name] = {p[0]: a[p[0]] for p in props}
            pos += dt.itemsize * count
            continue
        # fast path: every list has the length of the first one
        if count:
            n0, fields, off = None, [], pos
            for pname, t, it in props:
                if it is None:
                    fields.append((pname, e + t))
                    off += np.dtype(t).itemsize
                else:
                    n0 = int(np.frombuffer(body, e + t, 1, off)[0]) if n0 is None else n0
                    fields += [(pname + "#n", e + t), (pname, e + it, (n0,))]
                    off += np.dtype(t).itemsize + n0 * np.dtype(it).itemsize
            dt = np.dtype(fields)
            if pos + dt.itemsize * count <= len(body):
                a = np.frombuffer(body, dt, count, pos)
                if all((a[f[0]] == n0).all() for f in fields if f[0].endswith("#n")):
                    out[name] = {p[0]: a[p[0]] for p in props}
                    pos += dt.itemsize * count
                    continue
        cols = {p[0]: [] for p in props}
        for _ in range(count):
            for pname, t, it in props:
                if it is None:
                    cols[pname].append(np.frombuffer(body, e + t, 1, pos)[0])
                    pos += np.dtype(t).itemsize
                else:
                    n = int(np.frombuffer(body, e + t, 1, pos)[0])
                    pos += np.dtype(t).itemsize
                    cols[pname].append(np.frombuffer(body, e + it, n, pos).tolist())
                    pos += n * np.dtype(it).itemsize
        out[name] = cols
    return out


def _read_stl(path):
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data, "<u4", 1, 80)[0]) if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84 + 50 * n:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tri = np.frombuffer(data, dt, n, 84)["v"].astype(np.float64)
    else:
        txt = data.decode("ascii", "replace").split()
        at = [i for i, w in enumerate(txt) if w == "vertex"]
        tri = np.array([txt[i + 1:i + 4] for i in at], np.float64).reshape(-1, 3, 3)
    v, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return Mesh(v, inv.reshape(-1, 3), None, [path.stem])


# ---------------------------------------------------------- triangulation

def _triangulate_faces(v, polys, owner):
    """Triangles (and their owner) from polygons given as vertex index
    lists. Convex polygons are fanned; others are split in their plane."""
    tris, own = [], []
    for k, p in enumerate(polys):
        p = list(p)
        if len(p) == 3:
            tris.append(p)
            own.append(owner[k])
            continue
        if len(p) < 3:
            continue
        for t in _split_polygon(v[p]):
            tris.append([p[i] for i in t])
            own.append(owner[k])
    return np.array(tris, np.int64).reshape(-1, 3), np.array(own, np.int64)


def _split_polygon(ring):
    """Triangles (as corner numbers) of one planar 3D polygon: a fan if it
    is convex, else a constrained Delaunay triangulation in its plane."""
    n = len(ring)
    fan = [(0, i, i + 1) for i in range(1, n - 1)]
    nrm = np.cross(ring - ring.mean(0), np.roll(ring, -1, 0) - ring.mean(0)).sum(0)   # Newell
    if not np.any(nrm):
        return fan
    w = int(np.argmax(np.abs(nrm)))
    uv = ring[:, [a for a in range(3) if a != w]]
    d = np.roll(uv, -1, 0) - uv
    turn = d[:, 0] * np.roll(d, -1, 0)[:, 1] - d[:, 1] * np.roll(d, -1, 0)[:, 0]
    if (turn >= 0).all() or (turn <= 0).all():
        return fan
    import shapely

    try:
        tri = shapely.constrained_delaunay_triangles(shapely.Polygon(uv))
    except Exception:                                   # noqa: BLE001  (invalid ring: fall back)
        return fan
    where = {tuple(q): i for i, q in enumerate(uv.tolist())}
    ccw = (uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1]).sum() > 0
    out = []
    for t in shapely.get_parts(tri):
        c = shapely.get_coordinates(t)[:3]
        idx = [where.get(tuple(q)) for q in c.tolist()]
        if None in idx:
            return fan
        a, b, d = c
        turn = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
        out.append(tuple(idx) if (turn > 0) == ccw else tuple(idx[::-1]))   # keep the ring's direction
    return out or fan


# --------------------------------------------------------------- the fill

def _grid(mesh, x0, y0, cell):
    """Vertices in column units: column (i, j) has its (shifted) centre at
    exactly (i, j)."""
    v = mesh.vertices
    u = (v[:, 0] - x0) / cell - 0.5 - _SHIFT[0]
    w = (v[:, 1] - y0) / cell - 0.5 - _SHIFT[1]
    return u, w


def _edge(u, w, p, q, i, j):
    """Signed area of (p, q, point), computed the same way for p->q and
    q->p (so two triangles sharing an edge never both claim, or both miss,
    a point on it), and its sign with ties broken by moving the point by
    (e, e^2) (simulation of simplicity)."""
    swap = (u[p] > u[q]) | ((u[p] == u[q]) & (w[p] > w[q]))
    a, b = np.where(swap, q, p), np.where(swap, p, q)
    e = (u[b] - u[a]) * (j - w[a]) - (w[b] - w[a]) * (i - u[a])
    e = np.where(swap, -e, e)
    du, dw = u[q] - u[p], w[q] - w[p]
    tie = np.where(dw != 0, -np.sign(dw), np.sign(du))
    return e, np.where(e != 0, np.sign(e), tie)


def crossings(mesh, x0, y0, nx, ny, cell, chunk=1 << 22):
    """Where vertical rays through the column centres cross the triangles:
    (part, column, z, faces_up) for every crossing. Column = i * ny + j, as
    in CityBuilder. faces_up: the triangle is seen from above
    (counter-clockwise), i.e. a roof of a solid with outward faces."""
    u, w = _grid(mesh, x0, y0, cell)
    z = mesh.vertices[:, 2]
    a, b, c = mesh.faces.T
    area = (u[b] - u[a]) * (w[c] - w[a]) - (w[b] - w[a]) * (u[c] - u[a])
    tu, tw = np.stack([u[a], u[b], u[c]], 1), np.stack([w[a], w[b], w[c]], 1)
    i0 = np.maximum(np.ceil(tu.min(1)), 0)
    i1 = np.minimum(np.floor(tu.max(1)), nx - 1)
    j0 = np.maximum(np.ceil(tw.min(1)), 0)
    j1 = np.minimum(np.floor(tw.max(1)), ny - 1)
    ni = np.where(area != 0, np.maximum(i1 - i0 + 1, 0), 0).astype(np.int64)
    nj = np.maximum(j1 - j0 + 1, 0).astype(np.int64)
    cnt = ni * nj
    tri = np.nonzero(cnt)[0]
    out = [], [], [], []
    cum = np.cumsum(cnt[tri])
    s = 0
    while s < tri.size:
        e = max(s + 1, int(np.searchsorted(cum, (cum[s - 1] if s else 0) + chunk, "right")))
        t = tri[s:e]
        n = cnt[t]
        tt = np.repeat(t, n)
        k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        ii = i0[tt] + k // nj[tt]
        jj = j0[tt] + k % nj[tt]
        A, B, C = a[tt], b[tt], c[tt]
        sg = np.sign(area[tt])
        eab, sab = _edge(u, w, A, B, ii, jj)
        ebc, sbc = _edge(u, w, B, C, ii, jj)
        eca, sca = _edge(u, w, C, A, ii, jj)
        hit = (sab == sg) & (sbc == sg) & (sca == sg)
        ar = area[tt][hit]
        zz = (ebc[hit] * z[A[hit]] + eca[hit] * z[B[hit]] + eab[hit] * z[C[hit]]) / ar
        out[0].append(mesh.part[tt[hit]])
        out[1].append(ii[hit].astype(np.int64) * ny + jj[hit].astype(np.int64))
        out[2].append(zz)
        out[3].append(ar > 0)
        s = e
    if not out[0]:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool)
    return tuple(np.concatenate(o) for o in out)


def column_runs(mesh, x0, y0, nx, ny, cell):
    """Solid runs of every part in every column of the grid whose centre the
    part covers: (part, column, bottom, top) in metres, plus the number of
    open columns per part (odd crossings, closed with the face directions;
    see the module docstring) and the number of columns per part."""
    p, col, z, up = crossings(mesh, x0, y0, nx, ny, cell)
    up = up ^ (_signed_volume(mesh) < 0)[p]          # faces turned inwards: read them the other way
    o = np.lexsort((z, col, p))
    p, col, z, up = p[o], col[o], z[o], up[o]
    new = np.r_[True, (p[1:] != p[:-1]) | (col[1:] != col[:-1])] if p.size else np.zeros(0, bool)
    start = np.nonzero(new)[0]
    n = np.diff(np.r_[start, p.size])
    odd = start[n % 2 == 1]
    npart = mesh.n_parts
    n_open = np.bincount(p[odd], minlength=npart)
    n_cols = np.bincount(p[start], minlength=npart)
    if odd.size:
        lo = np.full(npart, np.inf)
        hi = np.full(npart, -np.inf)
        np.minimum.at(lo, mesh.part, mesh.vertices[mesh.faces, 2].min(1))
        np.maximum.at(hi, mesh.part, mesh.vertices[mesh.faces, 2].max(1))
        last = odd + n[n % 2 == 1] - 1
        # first crossing is a roof (no floor below it) -> close at the bottom;
        # else the last one is a floor (no roof above) -> close at the top
        at_top = ~up[odd] & ~up[last]
        extra = np.where(at_top, hi[p[odd]], lo[p[odd]])
        p, col, z = np.r_[p, p[odd]], np.r_[col, col[odd]], np.r_[z, extra]
        o = np.lexsort((z, col, p))
        p, col, z = p[o], col[o], z[o]
    zb, zt = z[0::2], z[1::2]
    keep = zt > zb
    return p[0::2][keep], col[0::2][keep], zb[keep], zt[keep], n_open, n_cols


def _signed_volume(mesh):
    """Volume of each part from its faces' directions (> 0 when they face
    outwards), seen from the part's lowest point below its centre, so a
    missing floor adds nothing."""
    npart = mesh.n_parts
    t = mesh.vertices[mesh.faces]
    lo = np.full(npart, np.inf)
    np.minimum.at(lo, mesh.part, t[:, :, 2].min(1))
    n = np.maximum(np.bincount(mesh.part, None, npart), 1)
    cx = np.bincount(mesh.part, t[:, :, 0].mean(1), npart) / n
    cy = np.bincount(mesh.part, t[:, :, 1].mean(1), npart) / n
    o = np.stack([cx, cy, np.where(np.isfinite(lo), lo, 0)], 1)[mesh.part]
    a, b, c = (t[:, k] - o for k in range(3))
    return np.bincount(mesh.part, np.einsum("ij,ij->i", a, np.cross(b, c)), npart) / 6


def surface_points(mesh, x0, y0, nx, ny, cell, spacing=None):
    """Points on the triangles every `spacing` metres (default half a cell)
    that fall in the grid: (part, column, z)."""
    sp = cell / 2 if spacing is None else float(spacing)
    t = mesh.vertices[mesh.faces]                          # (F, 3, 3)
    edge = np.linalg.norm(t - np.roll(t, 1, 1), axis=2).max(1)
    steps = np.maximum(np.ceil(edge / sp), 1).astype(np.int64)
    P, C, Z = [], [], []
    for k in np.unique(steps):
        r, s = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = r + s <= k
        r, s = r[m] / k, s[m] / k
        every = np.nonzero(steps == k)[0]
        step = max(1, (1 << 21) // r.size)
        for a in range(0, every.size, step):
            f = every[a:a + step]
            _sample(t[f], mesh.part[f], r, s, x0, y0, nx, ny, cell, P, C, Z)
    if not P:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(P), np.concatenate(C), np.concatenate(Z)


def _sample(tf, owner, r, s, x0, y0, nx, ny, cell, P, C, Z):
    q = (tf[:, None, 0] + r[None, :, None] * (tf[:, None, 1] - tf[:, None, 0])
         + s[None, :, None] * (tf[:, None, 2] - tf[:, None, 0])).reshape(-1, 3)
    ow = np.repeat(owner, r.size)
    i = np.floor((q[:, 0] - x0) / cell).astype(np.int64)
    j = np.floor((q[:, 1] - y0) / cell).astype(np.int64)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    P.append(ow[ok])
    C.append(i[ok] * ny + j[ok])
    Z.append(q[ok, 2])
