"""Queries: pick cells, then walk to their neighbours, parents, siblings.

    g = tree.select(id=7)                       # a Group of leaves (object 7 and its parts)
    g = tree.select(cls="road")                 # by an object-table column
    g = tree.select(column="z", between=(10, 20))    # by a cell column
    g = tree.select(where=lambda c: c["size_m"] > 4)  # by any function
    ring = g.neighbours(steps=2)                # cells 1 or 2 steps away
    near = g.within(10.0)                       # cells within 10 m (ring.steps, near.distance)
    near.objects()                              # object ids found there
    g.parents(1).ids()                          # parent cells' shared id (-2 = mixed)
    g.siblings()                                # leaves with the same parent cell
    tree.touching(7)                            # objects whose cells touch object 7
    tree.near(7, 25.0)                          # objects within 25 m of object 7

Words used
    leaf / cell   a leaf of the octree (one row of tree.cells)
    Group         a set of leaves (sorted leaf indices), with optional values
                  per leaf (steps, distance)
    Nodes         a set of octree cubes (Morton code, level): parent cells,
                  ancestors, children; each may be a leaf or an internal cell
    steps         moves between leaves that touch: connectivity 6 = faces,
                  18 = + edges, 26 = + corners. One step into a 32 m leaf
                  covers 32 m, so on an adaptive tree steps are not metres;
                  use within() for metres.
    distance      the gap between two cubes in metres (0 = touching)

Nothing global is precomputed: neighbours come from the Morton codes
(binary search), and within() works on a voxel crop around the group with a
distance transform, so the cost follows the size of the group and the
radius, not the size of the city.
"""

import inspect

import numpy as np
from scipy import ndimage

from . import morton
from .links import directions
from .objects import MIXED, NONE

# ------------------------------------------------------------------- helpers


def _expand_ranges(lo, hi):
    """Concatenate arange(lo[i], hi[i])."""
    n = (hi - lo).astype(np.int64)
    start = np.repeat(lo - np.r_[0, np.cumsum(n)[:-1]], n)
    return start + np.arange(int(n.sum()))


def _touching_children(d):
    """Morton offsets (0..7) of the children of a cube that touch the side
    facing direction -d (the side next to the source cell)."""
    out = []
    for k in range(8):
        o = (k & 1, (k >> 1) & 1, (k >> 2) & 1)
        if all(dv == 0 or (dv > 0 and ov == 0) or (dv < 0 and ov == 1) for dv, ov in zip(d, o)):
            out.append(k)
    return np.array(out, np.uint64)


def neighbour_pairs(tree, idx, connectivity=6, chunk=200_000):
    """All (i, j): leaf j touches leaf i (i in idx) by a face (6), an edge
    (18) or a corner (26). Exact for any mix of sizes: for each direction,
    the same-size cube next to i is inside one leaf (that leaf) or split;
    then only its children on the side facing i are visited, level by level."""
    idx = np.asarray(idx, np.int64)
    A, B = [], []
    for c0 in range(0, idx.size, chunk):
        a, b = _pairs_chunk(tree, idx[c0:c0 + chunk], connectivity)
        A.append(a)
        B.append(b)
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(A), np.concatenate(B)


def _pairs_chunk(tree, idx, connectivity):
    x, y, z = morton.decode(tree.code[idx])
    L = tree.level[idx].astype(np.int64)
    s = np.left_shift(1, L)
    A, B = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for d in directions(connectivity):
        n = [v + dv * s for v, dv in zip((x, y, z), d)]
        ok = np.ones(idx.size, bool)
        for v, lim in zip(n, tree.shape):
            ok &= (v >= 0) & (v < lim)
        src = idx[ok]
        code = morton.encode(*(v[ok] for v in n))
        lvl = L[ok]
        kids = _touching_children(d)
        while src.size:
            j = np.searchsorted(tree.code, code, side="right") - 1
            done = tree.level[j].astype(np.int64) >= lvl
            A.append(src[done])
            B.append(j[done])
            src, code, lvl = src[~done], code[~done], lvl[~done] - 1
            if not src.size:
                break
            step = np.left_shift(np.uint64(1), 3 * lvl.astype(np.uint64))
            code = (code[:, None] + kids[None, :] * step[:, None]).ravel()
            src = np.repeat(src, kids.size)
            lvl = np.repeat(lvl, kids.size)
    return np.concatenate(A), np.concatenate(B)


def _cube_of(code, levels):
    """Morton code of the aligned cube at `levels` that holds each code."""
    shift = 3 * np.asarray(levels, np.uint64)
    return (np.asarray(code, np.uint64) >> shift) << shift


def cube_ids(tree, codes, levels, layer="id"):
    """Id shared by all leaves inside each cube (code, level), else -2
    (mixed). Works for parent cells and leaves alike; exact (int64)."""
    v = np.asarray(tree.cells[layer])
    lo, hi = _leaf_range(tree, codes, levels)
    vp = np.r_[v, v[-1:]]
    idx = np.stack([lo, hi], 1).ravel()
    mn = np.minimum.reduceat(vp, idx)[::2]
    mx = np.maximum.reduceat(vp, idx)[::2]
    return np.where(mn == mx, mn, MIXED)


def _leaf_range(tree, codes, levels):
    """[lo, hi) of the leaves inside each cube; a cube inside one bigger
    leaf gives that leaf."""
    codes = np.asarray(codes, np.uint64)
    span = np.left_shift(np.uint64(1), 3 * np.asarray(levels, np.uint64))
    lo = np.searchsorted(tree.code, codes)
    hi = np.searchsorted(tree.code, codes + span)
    inner = lo == hi
    lo = np.where(inner, lo - 1, lo)
    hi = np.where(inner, lo + 1, hi)
    return lo, hi


def _key(code, level):
    """One comparable key per (code, level) pair (no overflow)."""
    a = np.ascontiguousarray(np.stack([np.asarray(code, np.uint64),
                                       np.asarray(level, np.uint64)], 1))
    return a.view(np.dtype((np.void, 16))).ravel()


def _call(func, *args):
    """Call func with as many of args as it takes (1 or 2)."""
    try:
        n = len([p for p in inspect.signature(func).parameters.values()
                 if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        n = len(args)
    return func(*args[:max(1, min(n, len(args)))])


# --------------------------------------------------------------------- Query

class Query:
    """A tree (with id columns in tree.cells) and its object table."""

    def __init__(self, tree, objects=None):
        self.tree = tree
        self.objects = objects if objects is not None else getattr(tree, "objects", None)
        self._centres = None

    @property
    def centres(self):
        if self._centres is None:
            self._centres = self.tree.centers()
        return self._centres

    def view(self):
        """What a where= function sees: per-leaf columns (see CellView)."""
        return CellView(self)

    def all(self):
        return Group(self, np.arange(self.tree.n_leaves))

    def _need_objects(self, what):
        if self.objects is None:
            raise ValueError(f"{what} needs the object table: set tree.objects (or pass objects=)")

    def select(self, id=None, layer="id", descendants=True, overlaps=True, cls=None, column=None,
               equals=None, between=None, isin=None, object_column=None, value=None,
               object_between=None, object_isin=None, tags=None, object_where=None, where=None):
        """Leaves that pass ALL the given tests.

        id                object id(s) in `layer`. descendants=True (default)
                          also takes their parts (a building and its
                          building parts); overlaps=True (default) also
                          takes overlap rows that list them (junctions).
        cls, object_column + value / object_between / object_isin, tags,
        object_where      tests on the object table (see ObjectTable.select);
                          the leaves of the matching objects are taken
        column + equals / between=(lo, hi) / isin   a cell column: "label",
                          "class", "level", "size_m", "x", "y", "z" (leaf
                          centre, metres) or any tree.cells column
        where             any function of the cells (and the object table)
                          -> bool per leaf: lambda c: c["z"] > 10, or
                          lambda c, objects: ...
        """
        t = self.tree
        m = np.ones(t.n_leaves, bool)
        want = None
        if id is not None:
            want = np.atleast_1d(np.asarray(id, np.int64))
        if any(v is not None for v in (cls, object_column, object_where, tags)):
            self._need_objects("selecting by object-table columns")
            o = self.objects.select(cls=cls, column=object_column, value=value, between=object_between,
                                    isin=object_isin, where=object_where, tags=tags)
            want = o if want is None else np.intersect1d(want, o)
        if want is not None:
            if layer not in t.cells:
                raise KeyError(f"no id layer {layer!r}; this tree has {list(t.id_columns)}")
            if self.objects is not None and want.size:
                if descendants:
                    want = np.union1d(want, self.objects.descendants(want))
                if overlaps:
                    want = np.union1d(want, self.objects.overlaps_of(want))
            m &= np.isin(t.cells[layer], want)
        if column is not None:
            c = self.view()[column]
            if equals is not None:
                m &= c == equals
            if between is not None:
                m &= (c >= between[0]) & (c <= between[1])
            if isin is not None:
                m &= np.isin(c, list(isin))
        if where is not None:
            m &= np.asarray(_call(where, self.view(), self.objects), bool)
        return Group(self, np.nonzero(m)[0])

    # ------------------------------------------------------------ objects
    def _object_ids(self, idx, layer):
        """Real objects in these leaves: overlap rows -> members; no -1 / -2."""
        v = np.unique(self.tree.cells[layer][idx])
        v = v[v >= 0]
        return self.objects.expand(v) if self.objects is not None else v

    def touching(self, id, connectivity=6, layer="id", other_layer=None):
        """Objects whose cells touch the cells of object `id` (6 = through a
        face, so a block resting on another one touches it). other_layer:
        look for them in another id layer (e.g. which zones a building
        touches)."""
        g = self.select(id=id, layer=layer, descendants=False)
        ring = g.neighbours(1, connectivity, outside=other_layer is None)
        found = self._object_ids(ring.idx, other_layer or layer)
        return found[found != int(id)] if other_layer is None else found

    def near(self, id, metres, layer="id", other_layer=None, return_distance=False):
        """Objects with a cell within `metres` of object `id` (gap between
        cells), nearest first; with return_distance, also the distances."""
        g = self.select(id=id, layer=layer, descendants=False)
        w = g.within(metres, outside=False)
        lay = other_layer or layer
        v = self.tree.cells[lay][w.idx]
        keep = v >= 0
        v, d = v[keep], w.distance[keep]
        if self.objects is not None and len(v):           # overlap rows -> their members
            ids, dist = [], []
            for i, dd in zip(v.tolist(), d.tolist()):
                for mbr in self.objects.expand([i]).tolist():
                    ids.append(mbr)
                    dist.append(dd)
            v, d = np.asarray(ids, np.int64), np.asarray(dist)
        if other_layer is None:
            keep = v != int(id)
            v, d = v[keep], d[keep]
        o = np.lexsort((v, d))
        v, d = v[o], d[o]
        first = np.unique(v, return_index=True)[1]
        first.sort()
        return (v[first], d[first]) if return_distance else v[first]


class CellView:
    """Per-leaf columns by name: label, class, level, size_m, x, y, z
    (leaf centre, metres), any tree.cells column; object(column, layer)
    gives an object-table column per leaf (where a leaf has no object:
    -1 for integer columns such as parent, NaN for numbers, None else)."""

    def __init__(self, q, idx=None):
        self.q, self.idx = q, idx

    def keys(self):
        return ["label", "class", "level", "size_m", "x", "y", "z"] + list(self.q.tree.cells)

    def _sub(self, a):
        return a if self.idx is None else a[self.idx]

    def __getitem__(self, k):
        t = self.q.tree
        if k == "label":
            return self._sub(t.label)
        if k == "class":
            return self._sub(t.class_names())
        if k == "level":
            return self._sub(t.level)
        if k == "size_m":
            return self._sub(t.sizes_m())
        if k in ("x", "y", "z"):
            return self._sub(self.q.centres[:, "xyz".index(k)])
        if k in t.cells:
            return self._sub(t.cells[k])
        raise KeyError(f"no cell column {k!r} (columns: {self.keys()})")

    def object(self, column, layer="id"):
        self.q._need_objects("object columns per cell")
        objs = self.q.objects
        r = objs.rows(self[layer])
        col = objs.column(column)
        hit = r >= 0
        if col.dtype.kind in "iu":                       # ids and integers stay exact: -1 where none
            out = np.full(r.size, -1, np.int64)
        elif col.dtype.kind == "f":
            out = np.full(r.size, np.nan)
        else:
            out = np.empty(r.size, dtype=object)          # text, bool, lists: None where none
        out[hit] = col[r[hit]]
        return out


# --------------------------------------------------------------------- Group

class Group:
    """A set of leaves. Values (steps, distance) follow the leaves."""

    def __init__(self, q, idx, **values):
        idx = np.asarray(idx, np.int64)
        idx, first = np.unique(idx, return_index=True)
        self.q, self.idx = q, idx
        self.values = {k: np.asarray(v)[first] for k, v in values.items()}

    @property
    def tree(self):
        return self.q.tree

    def __len__(self):
        return self.idx.size

    def __repr__(self):
        extra = "".join(f", {k}" for k in self.values)
        return f"Group({len(self):,} leaves, {self.volume_m3():,.0f} m3{extra})"

    def __getattr__(self, k):
        if k in ("q", "idx", "values"):
            raise AttributeError(k)
        if k in self.values:
            return self.values[k]
        raise AttributeError(k)

    # set operations: values are kept (the smaller one where both have it)
    def _combine(self, other, idx):
        vals = {}
        for k in set(self.values) | set(other.values):
            v = np.full(idx.size, np.inf)
            for g in (self, other):
                if k in g.values:
                    pos = np.searchsorted(g.idx, idx)
                    pos = np.minimum(pos, max(len(g) - 1, 0))
                    hit = (g.idx[pos] == idx) if len(g) else np.zeros(idx.size, bool)
                    v[hit] = np.minimum(v[hit], g.values[k][pos[hit]])
            vals[k] = v
        return Group(self.q, idx, **vals)

    def __or__(self, o):
        return self._combine(o, np.union1d(self.idx, o.idx))

    def __and__(self, o):
        return self._combine(o, np.intersect1d(self.idx, o.idx))

    def __sub__(self, o):
        idx = np.setdiff1d(self.idx, o.idx)
        keep = np.isin(self.idx, idx)
        return Group(self.q, idx, **{k: v[keep] for k, v in self.values.items()})

    def mask(self):
        m = np.zeros(self.q.tree.n_leaves, bool)
        m[self.idx] = True
        return m

    def filter(self, func):
        """Keep the leaves where func(cells of this group) is True; func gets
        a CellView of this group (c["z"], c["size_m"], ...)."""
        keep = np.asarray(_call(func, CellView(self.q, self.idx), self.q.objects), bool)
        return Group(self.q, self.idx[keep], **{k: v[keep] for k, v in self.values.items()})

    # ----------------------------------------------------------- contents
    def ids(self, layer="id"):
        """The id of each leaf in a layer (-1 none, -2 mixed)."""
        t = self.q.tree
        if layer not in t.cells:
            raise KeyError(f"no id layer {layer!r}; this tree has {list(t.id_columns)}")
        return t.cells[layer][self.idx]

    def objects(self, layer="id", members=False):
        """Distinct object ids in these leaves (no -1 / -2). Overlap rows
        stay as they are; members=True replaces them by their members
        (e.g. the crossing streets at a junction)."""
        if members:
            return self.q._object_ids(self.idx, layer)
        v = np.unique(self.ids(layer))
        return v[v >= 0]

    def table(self, layer="id", members=False):
        """The object-table rows of the objects in these leaves (ObjectTable)."""
        self.q._need_objects("table()")
        return self.q.objects.subset(self.objects(layer, members))

    def cells(self):
        """This group's cell columns as a dict of arrays (leaf index first)."""
        v = CellView(self.q, self.idx)
        out = {"leaf": self.idx}
        for k in v.keys():  # noqa: SIM118 (CellView is not a dict)
            a = v[k]
            if a.ndim == 1:
                out[k] = a
        out.update(self.values)
        return out

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.cells())

    def centres(self):
        return self.q.centres[self.idx]

    def show(self, **kw):
        """Draw these cells in 3D as wire cubes at their real size, coloured
        by cell size, with the buildings around (octreecity.viz.show_selection;
        needs octreecity[viz]). Several groups in one view:
        show_octree(tree, cells={"a": g1, "b": g2})."""
        from .viz import show_selection

        return show_selection(self.q.tree, self, **kw)

    def sizes_m(self):
        return self.q.tree.sizes_m()[self.idx]

    def volume_m3(self):
        return float((self.sizes_m() ** 3).sum()) if len(self) else 0.0

    def voxels(self):
        """Voxel indices (n, 3) of every voxel inside these leaves."""
        t = self.q.tree
        x, y, z = (v[self.idx] for v in t.corners())
        lv = t.level[self.idx]
        out = []
        for L in np.unique(lv):
            m = lv == L
            s = 1 << int(L)
            r = np.arange(s)
            off = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
            base = np.stack([x[m], y[m], z[m]], 1)
            out.append((base[:, None, :] + off[None]).reshape(-1, 3))
        return np.concatenate(out) if out else np.zeros((0, 3), np.int64)

    # ------------------------------------------------- spatial neighbours
    def neighbours(self, steps=1, connectivity=6, outside=True):
        """Leaves reachable in 1 .. steps moves between touching leaves
        (6 = faces, 18 = + edges, 26 = + corners). outside=True leaves out
        the group itself. The result has .steps (1, 2, ...)."""
        t = self.q.tree
        step = np.full(t.n_leaves, -1, np.int64)
        step[self.idx] = 0
        front = self.idx
        for h in range(1, int(steps) + 1):
            if not front.size:
                break
            _, b = neighbour_pairs(t, front, connectivity)
            b = np.unique(b)
            b = b[step[b] < 0]
            step[b] = h
            front = b
        out = np.nonzero(step > 0 if outside else step >= 0)[0]
        return Group(self.q, out, steps=step[out])

    neighbors = neighbours

    def boundary(self, connectivity=26):
        """Leaves of the group that touch a leaf outside it."""
        a, b = neighbour_pairs(self.q.tree, self.idx, connectivity)
        m = self.mask()
        return Group(self.q, np.unique(a[~m[b]]))

    def within(self, metres, outside=True, measure="box"):
        """Leaves within `metres` of the group. measure="box" (default): the
        gap between the cells (0 = touching), exact; "centre": centre to
        centre. The result has .distance (to the nearest leaf of the group).

        The box measure works on a voxel crop around the group plus the
        radius (a distance transform), so memory follows that crop: about
        16 bytes per voxel."""
        q = self.q
        if not len(self):
            return Group(q, [], distance=[])
        if measure == "centre":
            from scipy.spatial import cKDTree

            c = q.centres
            lo, hi = c[self.idx].min(0) - metres, c[self.idx].max(0) + metres
            near = np.nonzero(((c >= lo) & (c <= hi)).all(1))[0]
            d, _ = cKDTree(c[self.idx]).query(c[near], k=1, distance_upper_bound=metres + 1e-9)
            keep = np.isfinite(d)
            idx, dist = near[keep], d[keep]
        elif measure == "box":
            idx, dist = self._within_box(metres)
        else:
            raise ValueError("measure must be 'box' or 'centre'")
        if outside:
            keep = ~np.isin(idx, self.idx)
            idx, dist = idx[keep], dist[keep]
        return Group(q, idx, distance=dist)

    def _within_box(self, metres):
        t = self.q.tree
        h = float(t.voxel_size)
        S = 1 << t.max_level
        x, y, z = t.corners()
        s = t.sizes()
        corner = np.stack([x[self.idx], y[self.idx], z[self.idx]], 1)
        pad = int(np.ceil(metres / h)) + 1
        shape = np.asarray(t.shape)
        a = np.maximum(corner.min(0) - pad, 0) // S * S
        b = np.minimum(-(-np.minimum((corner + s[self.idx, None]).max(0) + pad, shape) // S) * S, shape)
        leaves = np.nonzero(t.in_box(a, b))[0]
        dims = tuple(int(v) for v in b - a)
        sub = np.empty(dims, np.int32)
        lx, ly, lz = x[leaves] - a[0], y[leaves] - a[1], z[leaves] - a[2]
        lv = t.level[leaves]
        for L in np.unique(lv):                      # leaves are aligned: fill per level
            k = 1 << int(L)
            m = lv == L
            view = sub.reshape(dims[0] // k, k, dims[1] // k, k, dims[2] // k, k)
            view[lx[m] // k, :, ly[m] // k, :, lz[m] // k, :] = leaves[m][:, None, None, None]
        lut = np.zeros(t.n_leaves, bool)
        lut[self.idx] = True
        gm = lut[sub]
        grown = ndimage.binary_dilation(gm, np.ones((3, 3, 3), bool))
        d = ndimage.distance_transform_edt(~grown) * h
        del grown
        d[gm] = 0.0
        best = np.asarray(ndimage.minimum(d, sub, leaves), np.float64)
        keep = best <= metres + 1e-9
        return leaves[keep], best[keep]

    def nearest(self, k, outside=True):
        """The k leaves nearest to the group (gap between cells)."""
        r = float(self.sizes_m().min()) if len(self) else 1.0
        limit = float(np.max(self.q.tree.shape)) * self.q.tree.voxel_size * 2
        while True:
            g = self.within(r, outside=outside)
            if len(g) >= k or r > limit:
                o = np.lexsort((g.idx, g.distance))[:k]
                return Group(self.q, g.idx[o], distance=g.distance[o])
            r *= 2

    # --------------------------------------------------- octree hierarchy
    def nodes(self):
        """These leaves as Nodes (to walk the hierarchy)."""
        t = self.q.tree
        return Nodes(self.q, t.code[self.idx], t.level[self.idx])

    def parents(self, levels=1):
        """The cells `levels` above each leaf (distinct), as Nodes. Leaves
        that are already root cubes (max_level) have no parent."""
        return self.nodes().parents(levels)

    def ancestors(self):
        """All cells above the group's leaves, up to the root cubes."""
        out, n = [], self.nodes()
        while True:
            n = n.parents(1)
            if not len(n):
                break
            out.append(n)
        return Nodes.concat(self.q, out)

    def siblings(self, leaves_only=True, include_group=False):
        """Leaves that share a direct parent cell with a leaf of the group.
        leaves_only=False also returns the leaves inside sibling cells that
        are split further (all leaves under the parent cells)."""
        t = self.q.tree
        lv = t.level[self.idx].astype(np.int64)
        ok = lv < t.max_level
        par = Nodes(self.q, _cube_of(t.code[self.idx][ok], lv[ok] + 1), lv[ok] + 1)
        g = par.leaves()
        if leaves_only:                     # a direct child's parent cube is one of `par`
            jl = t.level[g.idx].astype(np.int64) + 1
            own = _key(_cube_of(t.code[g.idx], jl), jl)
            g = Group(self.q, g.idx[np.isin(own, _key(par.code, par.level))])
        return g if include_group else g - self

    def children(self):
        """Leaves have no children: an empty Nodes (see Nodes.children)."""
        return Nodes(self.q, [], [])


# --------------------------------------------------------------------- Nodes

class Nodes:
    """Octree cubes (code, level), distinct. A cube is a leaf if a leaf has
    exactly that code and level, an internal (parent) cell if smaller
    leaves fill it, and does not exist if it lies inside a bigger leaf."""

    def __init__(self, q, code, level):
        code = np.asarray(code, np.uint64)
        level = np.asarray(level, np.uint8)
        if code.size:
            key = np.unique(np.stack([level.astype(np.uint64), code], 1), axis=0)
        else:
            key = np.zeros((0, 2), np.uint64)
        self.q, self.level, self.code = q, key[:, 0].astype(np.uint8), key[:, 1].astype(np.uint64)

    @staticmethod
    def concat(q, parts):
        if not parts:
            return Nodes(q, [], [])
        return Nodes(q, np.concatenate([p.code for p in parts]), np.concatenate([p.level for p in parts]))

    def __len__(self):
        return self.code.size

    def __repr__(self):
        lv = np.bincount(self.level).tolist() if len(self) else []
        return f"Nodes({len(self):,} cells; per level {lv})"

    def is_leaf(self):
        t = self.q.tree
        k = np.clip(np.searchsorted(t.code, self.code), 0, t.n_leaves - 1)
        return (t.code[k] == self.code) & (t.level[k] == self.level)

    def is_internal(self):
        """True for cubes that hold smaller leaves."""
        t = self.q.tree
        span = np.left_shift(np.uint64(1), 3 * self.level.astype(np.uint64))
        lo = np.searchsorted(t.code, self.code)
        hi = np.searchsorted(t.code, self.code + span)
        first = t.level[np.minimum(lo, t.n_leaves - 1)]
        return (hi > lo) & (first < self.level)          # else one leaf covers the whole cube

    def ids(self, layer="id"):
        """Object id shared by all leaves inside each cube; -2 if mixed."""
        return cube_ids(self.q.tree, self.code, self.level, layer)

    def corners(self):
        return morton.decode(self.code)

    def sizes_m(self):
        return np.left_shift(1, self.level.astype(np.int64)) * self.q.tree.voxel_size

    def centres(self):
        t = self.q.tree
        x, y, z = self.corners()
        half = 0.5 * np.left_shift(1, self.level.astype(np.int64))
        return np.stack([o + t.voxel_size * (v + half) for o, v in zip(t.origin, (x, y, z))], 1)

    def parents(self, levels=1):
        lv = self.level.astype(np.int64) + int(levels)
        ok = lv <= self.q.tree.max_level
        return Nodes(self.q, _cube_of(self.code[ok], lv[ok]), lv[ok])

    def children(self):
        """The 8 child cubes of each internal cell (leaves and cubes inside a
        bigger leaf have none)."""
        inner = self.is_internal() & (self.level > 0)
        c, lv = self.code[inner], self.level[inner].astype(np.int64) - 1
        step = np.left_shift(np.uint64(1), 3 * lv.astype(np.uint64))
        kids = (c[:, None] + step[:, None] * np.arange(8, dtype=np.uint64)[None]).ravel()
        return Nodes(self.q, kids, np.repeat(lv, 8))

    def leaves(self):
        """All leaves inside these cubes, as a Group (a cube inside a bigger
        leaf gives that leaf)."""
        lo, hi = _leaf_range(self.q.tree, self.code, self.level)
        return Group(self.q, _expand_ranges(lo, hi))

    descendants = leaves

    def siblings(self):
        """The other children of each cube's parent."""
        kids = self.parents(1).children()
        keep = ~np.isin(_key(kids.code, kids.level), _key(self.code, self.level))
        return Nodes(self.q, kids.code[keep], kids.level[keep])


__all__ = ["Query", "Group", "Nodes", "CellView", "neighbour_pairs", "cube_ids", "NONE", "MIXED"]
