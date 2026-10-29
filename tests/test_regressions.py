"""Bugs found by an independent review of the redesign, one test each."""
import numpy as np
import pytest

from octreecity.build import build
from octreecity.classes import Classes
from octreecity.fill import carry, from_raster
from octreecity.learn import Tokens, pixel_map, tokens
from octreecity.links import neighbours
from octreecity.octree import LinearOctree
from octreecity.rules import changes, mixed, varies
from octreecity.sources import GridSource


def test_parent_mean_skips_nan_leaves():
    t = LinearOctree.from_dense(np.zeros((4, 4, 4), np.uint8), 2)
    t.split(np.ones(1, bool))
    v = np.arange(8.0)
    v[0] = np.nan
    assert t.parents().reduce(v)[0] == pytest.approx(np.arange(1, 8).mean())


def test_carry_skips_nan():
    lab = np.zeros((8, 8, 8), np.uint8)
    lab[0, 0, 0] = 1
    lab[4:, :4, :4] = 1
    h = np.where(lab > 0, 7.0, np.nan)
    h[0, 0, 0] = 5.0
    t = build(GridSource(lab, attrs={"h": h}), mixed() | varies("h", 1), 3)
    other = LinearOctree.from_dense(np.zeros((8, 8, 8), np.uint8), 2)
    other.split(np.ones(1, bool))
    carry(t, other)
    cell = other.locate(4, 0, 0)
    assert other.cells["h"][cell] == pytest.approx(7.0)
    assert other.cells["h"][other.locate(0, 0, 0)] == pytest.approx(5.0)


def test_raster_max_with_uneven_pixels():
    t = LinearOctree.from_dense(np.zeros((16, 16, 8), np.uint8), 3)
    r = np.zeros((6, 6))
    r[0, 5] = 9.0
    leaf = t.locate(0, 8, 0)
    assert from_raster(t, r, "max", pixel=3)[leaf] == 9.0


def test_pixel_map_ignores_padding():
    lab = np.zeros((5, 5, 4), np.uint8)
    lab[4, 4, 0] = 1
    t = build(GridSource(lab), mixed(), 2)
    layer = t.in_box((0, 0, 0), (8, 8, 1))
    pm = pixel_map(t, layer, shape=(5, 5))
    x, y, _ = (v[layer] for v in t.corners())
    assert (x[pm[4, 4]], y[pm[4, 4]]) == (4, 4)


def test_concat_rejects_mixed_targets():
    t = LinearOctree.from_dense(np.zeros((4, 4, 4), np.uint8), 1)
    t.cells["y"] = np.ones(t.n_leaves)
    a = tokens(t, ["size_m"], neighbours(t), targets=["y"])
    b = tokens(t, ["size_m"], neighbours(t))
    with pytest.raises(ValueError):
        Tokens.concat([a, b])


def test_sources_reject_unknown_classes():
    c = Classes.from_tree({"a": {}})
    lab = np.zeros((4, 4, 4), np.uint8)
    lab[0, 0, 0] = 3
    with pytest.raises(ValueError):
        GridSource(lab, classes=c)


def test_ignore_by_name_and_big_boxes():
    c = Classes.from_tree({"a": {}, "b": {}})
    lab = np.zeros((8, 8, 8), np.uint8)
    lab[:4] = c.id("a")
    lab[4:, :, :1] = c.id("b")
    src = GridSource(lab, classes=c)
    assert build(src, mixed(ignore=["b"]), 3).n_leaves < build(src, mixed(), 3).n_leaves
    import octreecity.sources as S
    old = S.CHUNK
    try:
        S.CHUNK = 8                                   # force the "add up 8 children" path
        a = src.summarize([0], [0], [0], 8)
    finally:
        S.CHUNK = old
    b = src.summarize([0], [0], [0], 8)
    assert np.allclose(a.counts, b.counts) and a.volume[0] == 512
