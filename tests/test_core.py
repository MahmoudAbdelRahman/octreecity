import numpy as np
import pytest

from octreecity.classes import Classes
from octreecity.octree import LinearOctree


def random_tree(seed=0, n=16, max_level=3):
    rng = np.random.default_rng(seed)
    g = (rng.random((n, n, n)) < 0.1).astype(np.uint8) * 2
    t = LinearOctree.from_dense(g, max_level)
    t.cells["v"] = rng.random(t.n_leaves)
    return t


def test_classes_tree_and_depth():
    c = Classes.from_tree({"building": {"wall": {}, "roof": {"window": {}}}, "vegetation": {"tree": {}}})
    assert c.names == ["empty", "building", "wall", "roof", "window", "vegetation", "tree"]
    assert c.depth.tolist() == [0, 0, 1, 1, 2, 0, 1]
    assert c.at_depth(c.ids(["wall", "window", "tree"]), 0).tolist() == [1, 1, 5]
    assert c.at_depth([c.id("window")], 1).tolist() == [c.id("roof")]
    assert c.descendants("roof").tolist() == [3, 4]
    with pytest.raises(ValueError):
        c.add("wall")
    d = Classes.from_arrays(*c.to_arrays())
    assert d.names == c.names and (d.parent == c.parent).all()


def test_columns_follow_split_and_balance():
    t = random_tree()
    old = LinearOctree(t.shape, t.max_level, t.code.copy(), t.level.copy(), t.label.copy(), cells=dict(t.cells))
    t.split(t.level == 3).balance(26)
    assert t.check() and t.cells["v"].shape == (t.n_leaves,)
    # every new leaf carries the row of the old leaf it came from
    assert np.array_equal(t.cells["v"], old.cells["v"][old.locate(*t.corners())])
    assert np.array_equal(t.label, old.label[old.locate(*t.corners())])


def test_snapshots_and_save_load(tmp_path):
    t = random_tree(1)
    t.classes = Classes.from_tree({"a": {}, "b": {"c": {}}})
    t.add_snapshot("t0", w=np.arange(t.n_leaves))
    t.add_snapshot("t1", w=2 * np.arange(t.n_leaves), u=np.ones(t.n_leaves))
    assert t.series["w"].shape == (t.n_leaves, 2) and np.isnan(t.series["u"][:, 0]).all()
    t.split(t.level == 3)
    assert t.series["w"].shape[0] == t.n_leaves
    t.save(tmp_path / "t.npz")
    u = LinearOctree.load(tmp_path / "t.npz")
    assert (u.code == t.code).all() and (u.label == t.label).all()
    assert np.allclose(u.cells["v"], t.cells["v"])
    assert np.allclose(u.series["w"], t.series["w"]) and list(u.times) == ["t0", "t1"]
    assert u.classes.names == t.classes.names


def test_parents_structure_and_reduce():
    t = random_tree(2)
    t.split(t.level == 3).balance(26)
    p = t.parents(levels_above=1)
    lo, hi = p.leaf_range()
    vol = t.sizes() ** 3
    assert np.allclose([vol[a:b].sum() for a, b in zip(lo, hi)], p.sizes() ** 3)
    ok = p.leaf_parent >= 0
    assert (p.level[p.leaf_parent[ok]] == t.level[ok] + 1).all()
    x, y, z = t.corners()
    px, py, pz = p.corners()
    lp = p.leaf_parent[ok]
    for a, pa in ((x, px), (y, py), (z, pz)):
        assert ((a[ok] >= pa[lp]) & (a[ok] < pa[lp] + p.sizes()[lp])).all()
    # one top cell for the whole 16^3 box (2 x 2 x 2 roots of 8)
    assert (p.node_parent < 0).sum() == 1 and p.sizes()[p.node_parent < 0][0] == 16
    v = t.cells["v"]
    assert np.allclose(p.reduce(v, "max"), [v[a:b].max() for a, b in zip(lo, hi)])
    assert np.allclose(p.reduce(v), [(v[a:b] * vol[a:b]).sum() / vol[a:b].sum() for a, b in zip(lo, hi)])
    anc = p.ancestors()
    assert (anc[:, 0] == p.leaf_parent).all()
    assert ((anc >= 0).sum(1) == np.where(ok, 4 - t.level.astype(int), 1)).all()
