"""Object ids per voxel and per cell: exact, in 3D, never averaged."""

import numpy as np
import pytest

from octreecity.build import build
from octreecity.objects import MIXED, ObjectTable
from octreecity.occupancy import Occupancy
from octreecity.octree import LinearOctree
from octreecity.rules import changes, mixed
from octreecity.sources import GridSource, OccupancySource, SparseSource

BIG = 2 ** 62          # ids this big do not survive float64


def stacked_city(seed, n=32, nz=32):
    """Labels and two id layers with stacked blocks, air gaps and big ids."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((n, n, nz), np.uint8)
    lab[:, :, 0] = 1
    ids = np.full(lab.shape, -1, np.int64)
    zone = np.full(lab.shape, -1, np.int64)
    for k in range(12):
        x, y = rng.integers(0, n - 6, 2)
        w, d = rng.integers(2, 7, 2)
        z0 = rng.integers(1, nz - 8)
        h = rng.integers(1, 8)
        lab[x:x + w, y:y + d, z0:z0 + h] = 2
        ids[x:x + w, y:y + d, z0:z0 + h] = BIG + k        # stacked / overlapping blocks
    ids[3:6, 3:6, 20:25] = BIG + 99                       # an object made of air (a named void)
    zone[: n // 2, :, :] = BIG + 1000
    zone[n // 2:, : n // 2, 1:] = BIG + 1001
    return lab, {"id": ids, "zone_id": zone}


@pytest.mark.parametrize("seed", range(3))
def test_round_trip_dense_runs_tree_id(seed):
    lab, ids = stacked_city(seed)
    occ = Occupancy.from_dense(lab, ids=ids)
    for k, g in ids.items():
        assert (occ.to_dense(layer=k) == g).all()
    assert (occ.to_dense() == lab).all()
    p = np.random.default_rng(seed).integers(0, 32, (3, 3000))
    assert (occ.id_at(*p) == ids["id"][tuple(p)]).all()
    tree = occ.to_octree(4)
    tree.check()
    assert set(tree.id_columns) == {"id", "zone_id"}
    for k, g in ids.items():
        assert tree.cells[k].dtype == np.int64
        assert (tree.to_dense(k) == g).all()               # every voxel, exact
    assert (tree.cells["id"] != MIXED).all()
    # without the id rule leaves may hold two objects: -2, never an average
    t2 = occ.to_octree(4, split_ids=False)
    v = t2.cells["id"]
    assert ((v == MIXED) | np.isin(v, np.unique(ids["id"]))).all() and (v == MIXED).any()


def test_runs_split_where_an_id_changes():
    lab = np.zeros((1, 1, 10), np.uint8)
    lab[0, 0, :8] = 2
    ids = np.full(lab.shape, -1, np.int64)
    ids[0, 0, :4], ids[0, 0, 4:8] = 5, 6                  # one block on another: same label
    occ = Occupancy.from_dense(lab, ids={"id": ids})
    assert occ.top.tolist() == [4, 8, 10] and occ.label.tolist() == [2, 2, 0]


def test_crop_concat_pad_keep_ids():
    lab, ids = stacked_city(4)
    occ = Occupancy.from_dense(lab, ids=ids)
    sub = occ.crop_columns(5, 7, 21, 30)
    assert (sub.to_dense(layer="id") == ids["id"][5:21, 7:30]).all()
    assert sub.origin[:2] == (5.0, 7.0)
    a, b = occ.crop_columns(0, 0, 32, 10), occ.crop_columns(0, 10, 32, 32)
    y = Occupancy.concat_y([a, b])
    assert (y.to_dense(layer="zone_id") == ids["zone_id"]).all() and (y.to_dense() == lab).all()
    x = Occupancy.concat_x([occ.crop_columns(0, 0, 11, 32), occ.crop_columns(11, 0, 32, 32)])
    assert (x.to_dense(layer="id") == ids["id"]).all()
    tall = occ.pad_z(40)
    assert (tall.to_dense(layer="id")[..., :32] == ids["id"]).all()
    assert (tall.to_dense()[..., 32:] == 0).all() and (tall.to_dense(layer="id")[..., 32:] == -1).all()
    # crop in metres goes through the same path
    assert (occ.crop(5, 7, 16, 23).to_dense(layer="id") == sub.to_dense(layer="id")).all()


def test_save_load_ids_objects_meta(tmp_path):
    lab, ids = stacked_city(5)
    occ = Occupancy.from_dense(lab, ids=ids)
    occ.crs = "EPSG:32648"
    occ.objects = ObjectTable()
    occ.objects.add("test-set", ["a", "b"], "thing", height_m=[1.0, 2.0])
    occ.save(tmp_path / "c.npz")
    d = Occupancy.load(tmp_path / "c.npz")
    assert d.layers == ("id", "zone_id") and d.crs == "EPSG:32648"
    assert all((d.ids[k] == occ.ids[k]).all() for k in ids)
    assert len(d.objects) == 2
    raw = np.load(tmp_path / "c.npz", allow_pickle=False)
    assert "__meta__" in raw.files and raw["id__id__codes"].dtype == np.uint8
    tree = occ.to_octree(4)
    tree.save(tmp_path / "t.npz")
    t = LinearOctree.load(tmp_path / "t.npz")
    assert t.crs == "EPSG:32648" and t.id_columns == tree.id_columns
    assert (t.cells["id"] == tree.cells["id"]).all() and len(t.objects) == 2


def test_object_columns_save_or_say_why(tmp_path):
    lab, _ = stacked_city(6)
    tree = Occupancy.from_dense(lab).to_octree(4)
    tree.cells["name"] = np.array(["a"] * tree.n_leaves, dtype=object)
    tree.save(tmp_path / "t.npz")
    assert (LinearOctree.load(tmp_path / "t.npz").cells["name"] == "a").all()
    tree.cells["name"] = np.array([{"x": 1}] * tree.n_leaves, dtype=object)
    with pytest.raises(ValueError, match="cannot be saved"):
        tree.save(tmp_path / "t2.npz")


def test_air_carries_ids():
    lab = np.zeros((8, 8, 8), np.uint8)
    lab[..., 0] = 1
    ids = np.full(lab.shape, -1, np.int64)
    ids[2:6, 2:6, 1:5] = BIG + 3                              # a named courtyard of air
    tree = Occupancy.from_dense(lab, ids={"id": ids}).to_octree(3)
    assert (tree.to_dense("id") == ids).all()


def test_grid_and_sparse_sources_with_ids():
    lab, ids = stacked_city(7)
    t = build(GridSource(lab, ids=ids), mixed() | changes("id") | changes("zone_id"), 4)
    assert (t.to_dense("id") == ids["id"]).all() and (t.to_dense("zone_id") == ids["zone_id"]).all()
    # point cloud: each voxel takes the (class, id) pair with the most points
    xyz = np.array([[0.5, 0.5, 0.5]] * 3 + [[0.6, 0.4, 0.5]] * 2 + [[3.5, 0.5, 0.5]])
    s = SparseSource.from_points(xyz, [2, 2, 2, 3, 3, 2], 1.0, origin=(0, 0, 0), shape=(4, 4, 4),
                                 ids={"id": [BIG + 1, BIG + 1, BIG + 1, BIG + 2, BIG + 2, BIG + 5]})
    t = build(s, mixed() | changes("id"), 2)
    assert t.to_dense("id")[0, 0, 0] == BIG + 1 and t.to_dense()[0, 0, 0] == 2
    assert t.to_dense("id")[3, 0, 0] == BIG + 5 and t.to_dense("id")[1, 1, 1] == -1


def test_ids_are_never_averaged():
    lab, ids = stacked_city(8)
    tree = Occupancy.from_dense(lab, ids=ids).to_octree(4)
    p = tree.parents()
    same = p.reduce("id")
    lo, hi = p.leaf_range()
    v = tree.cells["id"]
    for k in range(0, len(p), 37):
        u = np.unique(v[lo[k]:hi[k]])
        assert same[k] == (u[0] if u.size == 1 else MIXED)
    with pytest.raises(ValueError, match="ids"):
        p.reduce("id", how="mean")
    with pytest.raises(ValueError, match="ids"):
        p.reduce(v, how="mean")                            # big ints are refused as numbers too
    from octreecity.learn import _column

    with pytest.raises(ValueError, match="object ids"):
        _column(tree, "id")


def test_carry_moves_ids_without_averaging():
    from octreecity.fill import carry

    lab, ids = stacked_city(9)
    occ = Occupancy.from_dense(lab, ids=ids)
    fine = occ.to_octree(4)
    coarse = build(OccupancySource(occ, layers=()), mixed(), 4)
    carry(fine, coarse, ["id"])
    assert coarse.cells["id"].dtype == np.int64 and "id" in coarse.id_columns
    ok = np.isin(coarse.cells["id"], np.r_[np.unique(ids["id"]), MIXED])
    assert ok.all() and (coarse.cells["id"] == MIXED).any()
    fine2 = occ.to_octree(4)
    fine2.cells.pop("id")
    carry(fine, fine2, ["id"])                              # same leaves: ids copied exactly
    assert (fine2.cells["id"] == fine.cells["id"]).all()


def test_height_of_counts_from_the_ground():
    lab = np.zeros((2, 1, 20), np.uint8)
    lab[:, :, :1] = 1
    lab[0, 0, 1:13] = 2                                    # a 12 m building on 1 m of ground
    occ = Occupancy.from_dense(lab)
    assert occ.top_of([2])[0, 0] == 13                     # counted from the bottom of the box
    assert occ.height_of([2], [1])[0, 0] == 12.0 and occ.height_of([2], [1])[1, 0] == 0.0


def test_voxels_of_an_object():
    lab, ids = stacked_city(10)
    occ = Occupancy.from_dense(lab, ids=ids)
    want = BIG + 3
    v = occ.voxels(id=want)
    ref = np.argwhere(ids["id"] == want)
    assert sorted(map(tuple, v.tolist())) == sorted(map(tuple, ref.tolist()))
