"""Queries checked against brute force on small cities."""

import warnings

import numpy as np
import pytest
import shapely

from octreecity.builder import CityBuilder
from octreecity.features import Features
from octreecity.links import directions as offsets
from octreecity.urban import BUILDING, ROAD


@pytest.fixture(scope="module")
def city():
    b = CityBuilder(0, 0, 64, 64)
    o = b.objects
    rng = np.random.default_rng(1)
    boxes, h = [], []
    for k in range(10):
        x, y = rng.integers(2, 50, 2)
        w, d = rng.integers(3, 12, 2)
        boxes.append(shapely.box(x, y, x + w, y + d))
        h.append(float(rng.integers(3, 25)))
    bid = o.add("test", [f"b{k}" for k in range(10)], "building", height_m=h,
                tags=[f'{{"name": "B{k}"}}' for k in range(10)])
    b.add_buildings(Features(boxes, height=h), ids=bid, overlap="first")
    pid = o.add("test", ["p0"], "building_part", parent=bid[0], height_m=[30.0])
    x0, y0, x1, y1 = boxes[0].bounds
    b.add_buildings(Features([shapely.box(x0, y0, x0 + 2, y0 + 2)], height=[30], min_height=[h[0]]), ids=pid)
    rid = o.add("test", ["r0", "r1"], "road")
    b.add_landcover([shapely.LineString([(0, 30), (64, 30)]), shapely.LineString([(30, 0), (30, 64)])],
                    ROAD, line_width=4, ids=rid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # the part overlaps another building: fine here
        c = b.build(air_above=4)
    tree = c.to_octree(4)
    return c, tree, bid, pid, rid


def leaf_grid(tree, c):
    nx, ny, nz = c.shape
    out = np.empty(tree.shape, np.int64)
    x, y, z = tree.corners()
    for lv in np.unique(tree.level):
        s = 1 << int(lv)
        m = np.nonzero(tree.level == lv)[0]
        v = out.reshape(tree.shape[0] // s, s, tree.shape[1] // s, s, tree.shape[2] // s, s)
        v[x[m] // s, :, y[m] // s, :, z[m] // s, :] = m[:, None, None, None]
    return out


def adjacency(tree, grid, connectivity):
    """Leaf pairs that touch, from the dense leaf grid."""
    pairs = set()
    for d in offsets(connectivity):
        a = grid[max(0, -d[0]):grid.shape[0] - max(0, d[0]), max(0, -d[1]):grid.shape[1] - max(0, d[1]),
                 max(0, -d[2]):grid.shape[2] - max(0, d[2])]
        b = grid[max(0, d[0]):grid.shape[0] - max(0, -d[0]) or None, max(0, d[1]):grid.shape[1] - max(0, -d[1]) or None,
                 max(0, d[2]):grid.shape[2] - max(0, -d[2]) or None]
        m = a != b
        pairs.update(zip(a[m].tolist(), b[m].tolist()))
    nb = {}
    for i, j in pairs:
        nb.setdefault(i, set()).add(j)
    return nb


def boxes_of(tree):
    x, y, z = tree.corners()
    s = tree.sizes()
    lo = np.stack([x, y, z], 1).astype(np.float64)
    return lo, lo + s[:, None]


def test_select_by_id_column_object_and_function(city):
    c, tree, bid, pid, rid = city
    ids = c.to_dense(layer="id")
    nx, ny, nz = c.shape
    tid = tree.to_dense("id")[:nx, :ny, :nz]
    assert (tid == ids).all()
    g = tree.select(id=bid[3])
    assert (tree.cells["id"][g.idx] == bid[3]).all()
    assert len(g.voxels()) == (ids == bid[3]).sum()
    both = tree.select(id=bid[0])                                  # the building and its part
    assert set(np.unique(tree.cells["id"][both.idx])) == {bid[0], pid[0]}
    assert set(np.unique(tree.cells["id"][tree.select(id=bid[0], descendants=False).idx])) == {bid[0]}
    road = tree.select(id=rid[0])
    rows = tree.objects.overlaps_of(rid[0])
    assert len(rows) == 1 and set(tree.cells["id"][road.idx]) == {rid[0], rows[0]}
    assert set(tree.cells["id"][tree.select(id=rid[0], overlaps=False).idx]) == {rid[0]}
    tall = tree.select(cls="building", object_column="height_m", object_between=(15, 100))
    want = [i for i, hh in zip(bid, tree.objects.column("height_m")[tree.objects.rows(bid)]) if hh >= 15]
    assert set(tree.objects.expand(tree.cells["id"][tall.idx])) <= set(want) | {pid[0]}
    named = tree.select(tags={"name": "B3"})
    assert (named.idx == g.idx).all()
    z = tree.centers()[:, 2]
    f1 = tree.select(where=lambda cv: cv["z"] > 10)
    f2 = tree.select(where=lambda cv, objs: (cv["z"] > 10) & (cv["label"] == BUILDING))
    assert (f1.idx == np.nonzero(z > 10)[0]).all()
    assert (f2.idx == np.nonzero((z > 10) & (tree.label == BUILDING))[0]).all()
    col = tree.select(column="size_m", equals=1.0)
    assert (tree.sizes_m()[col.idx] == 1).all() and len(col) == (tree.sizes_m() == 1).sum()
    h = tree.select(where=lambda cv: cv.object("height_m") > 20)
    hh = tree.objects.column("height_m")[tree.objects.rows(tree.cells["id"][h.idx])]
    assert (hh > 20).all()


@pytest.mark.parametrize("conn", [6, 18, 26])
def test_neighbours_match_brute_force(city, conn):
    c, tree, bid, pid, rid = city
    nb = adjacency(tree, leaf_grid(tree, c), conn)
    g = tree.select(id=bid[2])
    ring = g.neighbours(steps=3, connectivity=conn)
    seen = {int(i): 0 for i in g.idx}
    front = set(seen)
    for h in (1, 2, 3):
        nxt = set()
        for i in front:
            nxt |= nb.get(i, set())
        nxt -= set(seen)
        for i in nxt:
            seen[i] = h
        front = nxt
    want = {i: h for i, h in seen.items() if h > 0}
    assert dict(zip(ring.idx.tolist(), ring.steps.tolist())) == want
    inner = g.neighbours(1, conn, outside=False)
    assert set(g.idx.tolist()) <= set(inner.idx.tolist())
    b = g.boundary(connectivity=conn) if conn == 26 else None
    if b is not None:
        want_b = {i for i in g.idx.tolist() if nb.get(i, set()) - set(g.idx.tolist())}
        assert set(b.idx.tolist()) == want_b


@pytest.mark.parametrize("metres", [0.0, 2.5, 7.0])
def test_within_box_and_centre_match_brute_force(city, metres):
    c, tree, bid, pid, rid = city
    lo, hi = boxes_of(tree)
    g = tree.select(id=bid[5])
    w = g.within(metres)
    gap = np.full(tree.n_leaves, np.inf)
    for i in g.idx:
        d = np.maximum(np.maximum(lo[i] - hi, lo - hi[i]), 0)
        gap = np.minimum(gap, np.sqrt((d * d).sum(1)))
    want = np.setdiff1d(np.nonzero(gap <= metres + 1e-9)[0], g.idx)
    assert (w.idx == want).all() and np.allclose(w.distance, gap[want])
    cen = tree.centers()
    wc = g.within(metres + 3, measure="centre")
    dc = np.min(np.linalg.norm(cen[:, None] - cen[g.idx][None], axis=2), 1)
    assert (wc.idx == np.setdiff1d(np.nonzero(dc <= metres + 3)[0], g.idx)).all()
    near = g.nearest(25)
    assert len(near) == 25 and near.distance.max() <= np.sort(gap[np.setdiff1d(np.arange(tree.n_leaves), g.idx)])[24] + 1e-9


def test_parents_siblings_children(city):
    c, tree, bid, pid, rid = city
    g = tree.select(id=bid[1])
    small = g.idx[tree.level[g.idx] < tree.max_level]
    par = g.parents(1)
    x, y, z = tree.corners()
    s = tree.sizes()
    want = {(int(xx // (2 * ss) * 2 * ss), int(yy // (2 * ss) * 2 * ss), int(zz // (2 * ss) * 2 * ss), int(lv) + 1)
            for xx, yy, zz, ss, lv in zip(x[small], y[small], z[small], s[small], tree.level[small])}
    px, py, pz = par.corners()
    assert set(zip(px.tolist(), py.tolist(), pz.tolist(), par.level.astype(int).tolist())) == want
    ids = par.ids()
    for k in range(len(par)):
        lo = np.array([px[k], py[k], pz[k]])
        n = 1 << int(par.level[k])
        m = (x >= lo[0]) & (x < lo[0] + n) & (y >= lo[1]) & (y < lo[1] + n) & (z >= lo[2]) & (z < lo[2] + n)
        u = np.unique(tree.cells["id"][m])
        assert ids[k] == (u[0] if u.size == 1 else -2)
    sib = g.siblings()
    pk = {(int(a), int(b), int(cc), int(lv)) for a, b, cc, lv in zip(px, py, pz, par.level)}
    for j in sib.idx:
        lv = int(tree.level[j]) + 1
        n = 1 << lv
        assert (int(x[j] // n * n), int(y[j] // n * n), int(z[j] // n * n), lv) in pk
    assert not set(sib.idx.tolist()) & set(g.idx.tolist())
    kids = par.children()
    assert len(kids) == 8 * int(par.is_internal().sum())
    big = np.nonzero(tree.level == tree.max_level)[0][0]
    inner = tree.group([big]).nodes()
    from octreecity.query import Nodes

    sub = Nodes(inner.q, inner.code, inner.level - 1)               # a cube inside a bigger leaf
    assert len(sub.children()) == 0 and (sub.leaves().idx == [big]).all()
    assert len(g.ancestors()) >= len(par)
    assert "np.int64" not in repr(par)


def test_object_level_queries(city):
    c, tree, bid, pid, rid = city
    o = tree.objects
    assert o.parent(pid[0]) == bid[0] and o.children(bid[0]).tolist() == [pid[0]]
    ids = c.to_dense(layer="id")
    k = offsets(6)
    for b in bid[:5]:
        m = ids == b
        if not m.any():
            continue
        grown = np.zeros_like(m)
        for d in k:
            grown |= np.roll(m, tuple(d), (0, 1, 2))
        touch = set(np.unique(ids[grown & ~m]).tolist()) - {-1, int(b)}
        want = set(o.expand(list(touch)).tolist()) - {int(b)}
        assert set(tree.touching(b).tolist()) == want
    near, dist = tree.near(bid[0], 6.0, return_distance=True)
    assert (np.diff(dist) >= 0).all() and pid[0] in near.tolist()


def test_set_operations_keep_values(city):
    c, tree, bid, pid, rid = city
    g = tree.select(id=bid[2])
    a, b = g.neighbours(2), g.within(3.0)
    u = a | b
    assert hasattr(u, "steps") and hasattr(u, "distance") and len(u) == len(np.union1d(a.idx, b.idx))
    i = a & b
    assert np.isfinite(i.steps).all() and np.isfinite(i.distance).all()
    d = a - b
    assert len(d) == len(np.setdiff1d(a.idx, b.idx)) and hasattr(d, "steps")


def test_short_user_code(city):
    c, tree, bid, pid, rid = city
    out = tree.select(id=bid[4]).neighbours(steps=1).objects()
    assert out.dtype == np.int64
    assert isinstance(tree.select(id=bid[4]).table(), type(tree.objects))
