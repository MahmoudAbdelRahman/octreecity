"""Links: "cell A is related to cell B", with an optional type and values.

A Links object is a list of directed links src -> dst between tokens, each
with a kind (a name such as "face" or "parent") and optional values per
link. Tokens are the leaves (0 .. n_leaves - 1) and, when parent cells are
used, the internal cells after them (n_leaves + index in tree.parents()).

Built in:
    neighbours(tree)          leaves that touch (faces; or edges, corners)
    hierarchy(tree, parents)  leaf -> parent / all ancestors, parent -> children
    across(tree, parents)     the cell of the same size across each face (an
                              internal cell where the space there is split)
    within(tree, radius)      leaves whose centres are within a distance

Your own: Links(src, dst, kind_name, values=...), from any rule you like,
and Links.where(...) to keep only some (e.g. same object on both ends).
Combine with |. The library knows nothing about what a link means.
"""

from dataclasses import dataclass, field

import numpy as np

FACES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@dataclass
class Links:
    src: np.ndarray
    dst: np.ndarray
    kind: object = "link"                # one name for all, or (codes, names)
    values: dict = field(default_factory=dict)

    def __post_init__(self):
        self.src = np.asarray(self.src, np.int64)
        self.dst = np.asarray(self.dst, np.int64)
        if isinstance(self.kind, str):
            self.kinds = (self.kind,)
            self.kind = np.zeros(self.src.size, np.int16)
        else:
            codes, names = self.kind
            self.kinds = tuple(names)
            self.kind = np.asarray(codes, np.int16)
        self.values = {k: np.asarray(v) for k, v in self.values.items()}

    def __len__(self):
        return self.src.size

    def __repr__(self):
        counts = np.bincount(self.kind, minlength=len(self.kinds))
        parts = ", ".join(f"{n}: {c:,}" for n, c in zip(self.kinds, counts))
        return f"Links({len(self):,} links; {parts})"

    def kind_names(self):
        return np.array(self.kinds)[self.kind]

    def __or__(self, other):
        names = list(self.kinds) + [k for k in other.kinds if k not in self.kinds]
        remap = np.array([names.index(k) for k in other.kinds], np.int16)
        keys = set(self.values) | set(other.values)
        vals = {}
        for k in keys:
            a = self.values.get(k, np.full(len(self), np.nan))
            b = other.values.get(k, np.full(len(other), np.nan))
            vals[k] = np.concatenate([a, b])
        return Links(np.concatenate([self.src, other.src]), np.concatenate([self.dst, other.dst]),
                     (np.concatenate([self.kind, remap[other.kind]]), names), vals)

    def take(self, mask):
        idx = np.asarray(mask)
        return Links(self.src[idx], self.dst[idx], (self.kind[idx], self.kinds),
                     {k: v[idx] for k, v in self.values.items()})

    def where(self, keep):
        """Keep links where keep(src, dst) (or a bool array) is True."""
        m = keep(self.src, self.dst) if callable(keep) else keep
        return self.take(np.asarray(m, bool))

    def reverse(self, kind=None):
        """Same links pointing the other way, optionally all renamed `kind`."""
        if kind is None:
            return Links(self.dst, self.src, (self.kind, self.kinds), dict(self.values))
        return Links(self.dst, self.src, kind, dict(self.values))

    def unique(self):
        """Drop repeated (src, dst, kind) triples, keeping the first."""
        key = np.stack([self.src, self.dst, self.kind.astype(np.int64)], 1)
        _, first = np.unique(key, axis=0, return_index=True)
        return self.take(np.sort(first))

    def remap(self, index, n=None):
        """Renumber tokens: index[old] = new (-1 drops links that touch it)."""
        index = np.asarray(index, np.int64)
        s, d = index[self.src], index[self.dst]
        keep = (s >= 0) & (d >= 0)
        out = self.take(keep)
        out.src, out.dst = s[keep], d[keep]
        return out


# --------------------------------------------------------------- neighbours

def face_pairs(tree, mask=None):
    """All face-adjacent leaf pairs (a, b), a in mask (default all leaves).

    Each leaf probes the voxel just across each face at its own corner. The
    leaf found is the same size or larger (then it covers the whole face) or
    smaller (then the pair is also found from the smaller side, provided
    that side is in mask), so with mask = all leaves every pair appears.
    """
    return _probe(tree, FACES, mask)


def _probe(tree, dirs, mask=None):
    x, y, z = tree.corners()
    s = tree.sizes()
    idx = np.arange(tree.n_leaves) if mask is None else np.nonzero(mask)[0]
    A, B = [], []
    for dx, dy, dz in dirs:
        px = x[idx] + (s[idx] if dx > 0 else -1 if dx < 0 else 0)
        py = y[idx] + (s[idx] if dy > 0 else -1 if dy < 0 else 0)
        pz = z[idx] + (s[idx] if dz > 0 else -1 if dz < 0 else 0)
        ok = ((px >= 0) & (py >= 0) & (pz >= 0) & (px < tree.shape[0])
              & (py < tree.shape[1]) & (pz < tree.shape[2]))
        A.append(idx[ok])
        B.append(tree.locate(px[ok], py[ok], pz[ok]))
    return np.concatenate(A), np.concatenate(B)


def directions(connectivity):
    d = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    nz = {6: 1, 18: 2, 26: 3}[connectivity]
    return [v for v in d if sum(map(abs, v)) <= nz]


def neighbours(tree, connectivity=6, mask=None, kind="face"):
    """Links between leaves that touch: by a face (6), also an edge (18), or
    also a corner (26). Both directions, each pair once per direction. With
    mask, only links between leaves in mask (e.g. one horizontal layer)."""
    a, b = _probe(tree, directions(connectivity))
    pairs = np.unique(np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)]), axis=0)
    if mask is not None:
        m = np.asarray(mask, bool)
        pairs = pairs[m[pairs[:, 0]] & m[pairs[:, 1]]]
    return Links(pairs[:, 0], pairs[:, 1], kind)


def within(tree, radius_m, mask=None, kind="near", max_links=None):
    """Links between leaves (in mask) whose centres are at most radius_m
    apart, with the distance as value 'distance'. max_links keeps the
    nearest ones per leaf."""
    from scipy.spatial import cKDTree

    idx = np.arange(tree.n_leaves) if mask is None else np.nonzero(mask)[0]
    c = tree.centers()[idx]
    pairs = cKDTree(c).query_pairs(radius_m, output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    d = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)
    if max_links is not None and len(pairs):
        o = np.lexsort((d, pairs[:, 0]))
        pairs, d = pairs[o], d[o]
        rank = np.arange(len(pairs)) - np.searchsorted(pairs[:, 0], pairs[:, 0])
        pairs, d = pairs[rank < max_links], d[rank < max_links]
    return Links(idx[pairs[:, 0]], idx[pairs[:, 1]], kind, {"distance": d})


# ---------------------------------------------------------------- hierarchy

def hierarchy(tree, parents, ancestors=True, children=True, up_between_parents=True):
    """Links along the tree. Parent cells are tokens n_leaves + j.

    leaf -> its parent ("parent"), or -> all its ancestors ("ancestor") when
    ancestors=True; parent cell -> its parent ("parent") when
    up_between_parents; parent cell -> each child, leaf or parent cell
    ("child") when children.
    """
    n = tree.n_leaves
    out = []
    if ancestors:
        anc = parents.ancestors()
        r, c = np.nonzero(anc >= 0)
        out.append(Links(r, n + anc[r, c], "ancestor", {"steps": c + 1}))
    else:
        ok = parents.leaf_parent >= 0
        out.append(Links(np.nonzero(ok)[0], n + parents.leaf_parent[ok], "parent"))
    ok = parents.node_parent >= 0
    nodes = np.nonzero(ok)[0]
    if up_between_parents:
        out.append(Links(n + nodes, n + parents.node_parent[ok], "parent"))
    if children:
        lk = np.nonzero(parents.leaf_parent >= 0)[0]
        out.append(Links(n + parents.leaf_parent[lk], lk, "child"))
        out.append(Links(n + parents.node_parent[ok], n + nodes, "child"))
    res = out[0]
    for o in out[1:]:
        res = res | o
    return res


def across(tree, parents, connectivity=6, leaves=True, nodes=True, kind="across"):
    """For each cell (leaves and/or parent cells), a link to the cell of the
    same size across each face (edge, corner): the parent cell there if
    that space is split further, else the leaf that covers it (same size or
    larger). This keeps one link per direction whatever the sizes nearby."""
    n = tree.n_leaves
    src, codes, levels = [], [], []
    if leaves:
        src.append(np.arange(n)), codes.append(tree.code), levels.append(tree.level)
    if nodes:
        src.append(n + np.arange(len(parents))), codes.append(parents.code), levels.append(parents.level)
    src, code, level = np.concatenate(src), np.concatenate(codes), np.concatenate(levels).astype(np.int64)
    from . import morton

    x, y, z = morton.decode(code)
    s = np.left_shift(1, level)
    A, B = [], []
    for dx, dy, dz in directions(connectivity):
        px, py, pz = x + dx * s, y + dy * s, z + dz * s
        ok = ((px >= 0) & (py >= 0) & (pz >= 0) & (px < tree.shape[0])
              & (py < tree.shape[1]) & (pz < tree.shape[2]))
        hit = np.full(src.size, -1)
        node = parents.find(morton.encode(px[ok], py[ok], pz[ok]), level[ok])
        leaf = tree.locate(px[ok], py[ok], pz[ok])
        same_node = node >= 0
        h = np.where(same_node, n + node, np.where(tree.level[leaf] >= level[ok], leaf, -1))
        hit[ok] = h
        good = hit >= 0
        A.append(src[good]), B.append(hit[good])
    return Links(np.concatenate(A), np.concatenate(B), kind)
