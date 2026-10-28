import numpy as np
import pytest

from octreecity.build import build
from octreecity.columns import ColumnCity
from octreecity.learn import Tokens, batches, pixel_map, tokens
from octreecity.links import hierarchy, neighbours
from octreecity.rules import mixed
from octreecity.sources import ColumnSource


@pytest.fixture
def tree():
    rng = np.random.default_rng(0)
    h = np.where(rng.random((32, 32)) < 0.3, rng.integers(1, 12, (32, 32)), 0).astype(float)
    city = ColumnCity.from_heightfield(np.zeros_like(h), h, 2, 16)
    t = build(ColumnSource(city, attrs={"h": np.where(h > 0, h, np.nan)}), mixed(), 3).balance(6)
    t.cells["svf"] = np.random.default_rng(1).random(t.n_leaves)
    return t


def test_tokens_leaves_and_parents(tree):
    sel = tree.in_box((0, 0, 0), (32, 32, 1))
    p = tree.parents(levels_above=2)
    L = neighbours(tree, mask=sel) | hierarchy(tree, p)
    tok = tokens(tree, ["h", "size_m"], L, targets=["svf"], select=sel, parents=p)
    nl = int(sel.sum())
    assert tok.is_leaf[:nl].all() and not tok.is_leaf[nl:].any()
    assert (tok.idx[:, 0] == np.arange(len(tok))).all() and tok.valid[:, 0].all() and (tok.kind[:, 0] == -1).all()
    assert np.isnan(tok.targets[nl:]).all() and np.allclose(tok.targets[:nl, 0], tree.cells["svf"][sel])
    # every kept parent is an ancestor of a selected leaf; every leaf reads all its ancestors
    anc = p.ancestors()[sel]
    assert len(tok) - nl == np.unique(anc[anc >= 0]).size
    deg_anc = (tok.kind[:nl] == tok.kinds.index("ancestor")).sum(1)
    assert (deg_anc == (anc >= 0).sum(1)).all()
    # offsets are relative positions of the keys
    i, j = np.nonzero(tok.valid)
    assert np.allclose(tok.offset[i, j, :3] * 16, tok.position[tok.idx[i, j]] - tok.position[i], atol=1e-4)
    # parent features: volume-weighted mean of the leaf features (size_m)
    assert (tok.features[nl:, 1] > 0).all()


def test_key_limit_concat_save(tree, tmp_path):
    L = neighbours(tree, 26)
    tok = tokens(tree, ["size_m"], L, k=5)
    assert tok.idx.shape[1] == 5 and tok.dropped_links > 0
    near = np.linalg.norm(tok.offset[:, 1:, :3], axis=2)
    assert (np.diff(np.where(tok.valid[:, 1:], near, np.inf), axis=1) >= -1e-6).all()   # nearest first
    a = tokens(tree, ["size_m"], neighbours(tree), k=4)
    b = Tokens.concat([a, tok])
    assert len(b) == len(a) + len(tok) and b.idx.shape[1] == 5
    assert (b.idx[len(a):, 0] == len(a) + np.arange(len(tok))).all()
    assert len(list(batches([a, a, a], 2))) == 2
    tok.save(tmp_path / "t.npz")
    u = Tokens.load(tmp_path / "t.npz")
    assert (u.idx == tok.idx).all() and u.kinds == tok.kinds and u.dropped_links == tok.dropped_links


def test_pixel_map(tree):
    sel = tree.in_box((0, 0, 0), (32, 32, 1))
    pm = pixel_map(tree, sel)
    assert (pm >= 0).all()
    x, y, _ = (v[sel] for v in tree.corners())
    assert (pm[x, y] == np.arange(sel.sum())).all()
    assert pixel_map(tree, sel, pixel=2).shape == (16, 16)


def test_reach_grows_with_hops(tree):
    from octreecity.learn import reach_m

    tok = tokens(tree, ["size_m"], neighbours(tree, mask=tree.in_box((0, 0, 0), (32, 32, 1))),
                 select=tree.in_box((0, 0, 0), (32, 32, 1)))
    assert 0 < reach_m(tok, 1) < reach_m(tok, 3)
