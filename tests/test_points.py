"""Point data on cells: tree.cell_at and fill.from_points."""

import warnings

import numpy as np
import pytest

from octreecity.fill import from_points
from octreecity.octree import LinearOctree

ORIGIN = (500_000.0, 4_000_000.0, 10.0)        # UTM 52N metres, z from 10 m


@pytest.fixture
def tree():
    """Mixed cell sizes over a 64 x 64 x 16 box of 2 m voxels, in UTM 52N."""
    rng = np.random.default_rng(0)
    t = LinearOctree.from_dense((rng.random((64, 64, 16)) < 0.03).astype(np.uint8), 3,
                                voxel_size=2.0, origin=ORIGIN)
    t.crs = "EPSG:32652"
    assert len(np.unique(t.level)) > 1
    return t


def random_points(t, n, seed=1):
    rng = np.random.default_rng(seed)
    hi = np.array(t.shape) * t.voxel_size
    return [o + rng.random(n) * h for o, h in zip(t.origin, hi)]


def brute_cell(t, x, y, z):
    """Leaf whose box contains each point, by checking every leaf."""
    lo = np.stack(t.corners(), 1) * t.voxel_size + np.array(t.origin)
    hi = lo + t.sizes_m()[:, None]
    p = np.stack([x, y, z], 1)[:, None, :]
    hit = ((p >= lo) & (p < hi)).all(2)
    return np.where(hit.any(1), hit.argmax(1), -1)


def test_cell_at_matches_leaf_boxes(tree):
    x, y, z = random_points(tree, 3000)
    assert np.array_equal(tree.cell_at(x, y, z), brute_cell(tree, x, y, z))
    c = tree.centers()
    assert np.array_equal(tree.cell_at(c[:, 0], c[:, 1], c[:, 2]), np.arange(tree.n_leaves))


def test_cell_at_outside_is_minus_one(tree):
    ox, oy, oz = tree.origin
    X, Y, Z = (n * tree.voxel_size for n in tree.shape)
    x = np.array([ox - 0.01, ox + X, ox + 1, ox + 1, ox + 1, np.nan, ox])
    y = np.array([oy + 1, oy + 1, oy - 5, oy + 1, oy + 1, oy + 1, oy])
    z = np.array([oz + 1, oz + 1, oz + 1, oz + Z + 3, oz - 0.5, oz + 1, oz])
    got = tree.cell_at(x, y, z)
    assert (got[:-1] == -1).all() and got[-1] == tree.locate(0, 0, 0)
    assert tree.cell_at(ox - 1, oy, oz) == -1                          # scalar in, scalar out


def test_locate_outside_box_is_minus_one(tree):
    nx, ny, nz = tree.shape
    assert (tree.locate([-1, nx, 0, 0, 0], [0, 0, ny, 0, -3], [0, 0, 0, nz, 0]) == -1).all()
    assert tree.locate(0, 0, 0) >= 0


def test_lon_lat_round_trip(tree):
    from pyproj import Transformer

    c = tree.centers()
    lon, lat = Transformer.from_crs(tree.crs, "EPSG:4326", always_xy=True).transform(c[:, 0], c[:, 1])
    assert np.array_equal(tree.cell_at(lon, lat, c[:, 2], crs="EPSG:4326"), np.arange(tree.n_leaves))
    # same crs as the tree: no conversion
    assert np.array_equal(tree.cell_at(c[:, 0], c[:, 1], c[:, 2], crs="EPSG:32652"),
                          np.arange(tree.n_leaves))


def test_crs_needs_tree_crs(tree):
    tree.crs = None
    with pytest.raises(ValueError, match="no crs"):
        tree.cell_at(127.0, 36.1, 0.0, crs="EPSG:4326")


def test_mean_sum_count_min_max_std(tree):
    x, y, z = random_points(tree, 4000)
    v = np.random.default_rng(2).normal(20, 5, x.size)
    v[::50] = np.nan                                                   # skipped
    cell = brute_cell(tree, x, y, z)
    for how, f in [("mean", np.mean), ("sum", np.sum), ("min", np.min), ("max", np.max),
                   ("std", np.std), ("count", len)]:
        got = from_points(tree, x, y, z, v, how=how)
        for i in np.unique(cell)[:200]:
            vi = v[(cell == i) & np.isfinite(v)]
            if vi.size:
                assert got[i] == pytest.approx(f(vi)), how
    empty = np.setdiff1d(np.arange(tree.n_leaves), cell)
    assert empty.size
    assert np.isnan(from_points(tree, x, y, z, v)[empty]).all()
    assert (from_points(tree, x, y, z, v, how="count")[empty] == 0).all()
    assert (from_points(tree, x, y, z, v, how="sum")[empty] == 0).all()
    n = from_points(tree, x, y, z)                                     # no values: count points
    assert n.dtype.kind == "i" and n.sum() == x.size and (n == np.bincount(cell, minlength=tree.n_leaves)).all()


def test_points_outside_are_dropped_with_warning(tree):
    x, y, z = random_points(tree, 100)
    x = np.r_[x, tree.origin[0] - 10, tree.origin[0] - 20]
    y, z = np.r_[y, y[:2]], np.r_[z, z[:2]]
    with pytest.warns(UserWarning, match="2 of 102 points are outside"):
        n = from_points(tree, x, y, z)
    assert n.sum() == 100
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                 # all inside: no warning
        from_points(tree, x[:100], y[:100], z[:100])


def test_mode_and_ids_stay_exact(tree):
    x, y, z = random_points(tree, 3000)
    cell = brute_cell(tree, x, y, z)
    big = np.array([2 ** 62 + 1, 2 ** 62 + 2, 2 ** 62 + 3], np.int64)  # differ only below float64
    ids = big[np.random.default_rng(3).integers(0, 3, x.size)]
    got = from_points(tree, x, y, z, ids)                              # default for ids: mode
    assert got.dtype == np.int64
    for i in np.unique(cell)[:200]:
        u, k = np.unique(ids[cell == i], return_counts=True)
        assert got[i] == u[k == k.max()].min()                         # tie: smallest
    assert (got[np.setdiff1d(np.arange(tree.n_leaves), cell)] == -1).all()
    with pytest.raises(ValueError, match="ids or text"):
        from_points(tree, x, y, z, ids, how="mean")


def test_mode_one_cell_by_hand(tree):
    c = tree.centers()[5]
    p = [np.full(5, a) for a in c]
    got = from_points(tree, *p, np.array([3, 7, 7, 3, 7]), how="mode")
    assert got[5] == 7 and (np.delete(got, 5) == -1).all()
    lab = from_points(tree, *p, np.array(["shop", "cafe", "cafe", "shop", "bank"]))
    assert lab[5] == "cafe" and lab[0] == ""                           # tie: smallest, empty: ""


def test_dict_of_columns_and_id_columns(tree):
    x, y, z = random_points(tree, 500)
    tree.id_columns = ("sensor",)
    sensor = np.arange(x.size, dtype=np.int64)                         # small ints, but named as ids
    out = from_points(tree, x, y, z, {"pm25": np.ones(x.size), "sensor": sensor})
    assert set(out) == {"pm25", "sensor"}
    assert out["sensor"].dtype == np.int64 and np.isin(out["sensor"], np.r_[sensor, -1]).all()
    hit = out["sensor"] >= 0
    assert (out["pm25"][hit] == 1).all() and np.isnan(out["pm25"][~hit]).all()
    with pytest.raises(ValueError, match="ids or text"):
        from_points(tree, x, y, z, {"sensor": sensor}, how="mean")


def test_bad_input(tree):
    with pytest.raises(ValueError, match="one per point"):
        from_points(tree, [ORIGIN[0]] * 3, [ORIGIN[1]] * 3, ORIGIN[2], [1.0, 2.0])
    with pytest.raises(ValueError, match="unknown how"):
        from_points(tree, ORIGIN[0], ORIGIN[1], ORIGIN[2], [1.0], how="median")
    with pytest.raises(ValueError, match="needs values"):
        from_points(tree, ORIGIN[0], ORIGIN[1], ORIGIN[2], how="mean")
