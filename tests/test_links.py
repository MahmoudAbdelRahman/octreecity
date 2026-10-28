import numpy as np
import pytest

from octreecity.links import Links, across, hierarchy, neighbours, within
from octreecity.octree import LinearOctree


@pytest.fixture
def tree():
    rng = np.random.default_rng(0)
    g = (rng.random((16, 16, 16)) < 0.08).astype(np.uint8)
    return LinearOctree.from_dense(g, 3)


def brute_pairs(tree, dirs):
    ids = np.empty(tree.shape, np.int64)
    x, y, z = tree.corners()
    for i, (a, b, c, s) in enumerate(zip(x, y, z, tree.sizes())):
        ids[a:a + s, b:b + s, c:c + s] = i
    out = set()
    n = np.array(tree.shape)
    for d in dirs:
        sl0 = tuple(slice(max(0, -k), m - max(0, k)) for k, m in zip(d, n))
        sl1 = tuple(slice(max(0, k), m - max(0, -k)) for k, m in zip(d, n))
        a, b = ids[sl0].ravel(), ids[sl1].ravel()
        m = a != b
        out |= set(zip(a[m].tolist(), b[m].tolist()))
    return out


@pytest.mark.parametrize("conn", [6, 18, 26])
def test_neighbours_complete(tree, conn):
    from octreecity.links import directions

    L = neighbours(tree, conn)
    got = set(zip(L.src.tolist(), L.dst.tolist()))
    assert got == brute_pairs(tree, directions(conn)) and len(got) == len(L)


def test_neighbours_in_one_layer(tree):
    m = tree.in_box((0, 0, 0), (16, 16, 1))
    L = neighbours(tree, mask=m)
    assert m[L.src].all() and m[L.dst].all() and len(L) > 0


def test_hierarchy_and_across(tree):
    tree.balance(6)
    p = tree.parents(levels_above=1)
    n = tree.n_leaves
    H = hierarchy(tree, p)
    anc = H.take(H.kind_names() == "ancestor")
    assert len(anc) == (p.ancestors() >= 0).sum()
    child = H.take(H.kind_names() == "child")
    assert len(child) == (p.leaf_parent >= 0).sum() + (p.node_parent >= 0).sum()
    assert (child.src >= n).all()
    A = across(tree, p)
    # every target is the same size or (for leaves) larger, and touches the source across a face
    lv = np.concatenate([tree.level, p.level]).astype(int)
    assert (lv[A.dst] >= lv[A.src]).all()
    same = A.dst >= n
    assert (lv[A.dst][same] == lv[A.src][same]).all()
    # interior leaves have one link per face
    counts = np.bincount(A.src[A.src < n], minlength=n)
    x, y, z = tree.corners()
    s = tree.sizes()
    inner = (x > 0) & (y > 0) & (z > 0) & (x + s < 16) & (y + s < 16) & (z + s < 16)
    assert (counts[inner] == 6).all()


def test_within_and_link_ops(tree):
    W = within(tree, 3.0, max_links=4)
    assert (W.values["distance"] <= 3.0).all() and np.bincount(W.src).max() <= 4
    F = neighbours(tree)
    both = F | W
    assert set(both.kinds) == {"face", "near"} and len(both) == len(F) + len(W)
    assert np.isnan(both.values["distance"][:len(F)]).all()
    assert len((F | F).unique()) == len(F)
    R = F.reverse("back")
    assert R.kinds == ("back",) and (R.src == F.dst).all()
    same_label = F.where(lambda a, b: tree.label[a] == tree.label[b])
    assert (tree.label[same_label.src] == tree.label[same_label.dst]).all()
    keep = np.full(tree.n_leaves, -1)
    keep[:10] = np.arange(10)
    sub = F.remap(keep)
    assert (sub.src < 10).all() and (sub.dst < 10).all()
    custom = Links([0, 1], [1, 0], "mine", {"w": [0.5, 0.5]})
    assert repr(custom).startswith("Links(2 links")
