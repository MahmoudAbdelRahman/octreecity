import numpy as np
import pytest

from octreecity import LinearOctree, rasterize_columns

from octreecity.urban import CLASSES, BUILDING, GROUND, TREE
from octreecity.occupancy import Occupancy

from test_octree import random_city


@pytest.mark.parametrize("seed", range(3))
def test_heightfield_matches_dense(seed):
    g, s, c, nz = random_city(seed)
    grid = rasterize_columns(g, s, c, nz)
    a = Occupancy.from_heightfield(g, s, c, nz)
    b = Occupancy.from_dense(grid)
    assert (a.to_dense() == grid).all()
    assert (a.offsets == b.offsets).all() and (a.top == b.top).all() and (a.label == b.label).all()


def test_to_octree_matches_direct():
    g, s, c, nz = random_city(1)
    col = Occupancy.from_heightfield(g, s, c, nz)
    t1 = col.to_octree(5)
    t2 = LinearOctree.from_heightfield(g, s, c, nz, 5)
    t1.check()
    assert (t1.code == t2.code).all() and (t1.label == t2.label).all()


def test_label_at_and_top_of():
    g, s, c, nz = random_city(2)
    grid = rasterize_columns(g, s, c, nz)
    col = Occupancy.from_dense(grid)
    p = np.random.default_rng(0).integers(0, 64, (3, 4000))
    assert (col.label_at(*p) == grid[tuple(p)]).all()
    top = col.top_of([BUILDING, TREE])
    ref = np.where(np.isin(grid, [BUILDING, TREE]), np.arange(nz) + 1, 0).max(-1)
    assert (top == ref).all()


def test_layers_tree_canopy_over_air():
    shape = (32, 32, 32)
    canopy = (np.full((32, 32), 4.0), np.full((32, 32), 10.0), TREE)
    ground = (0, np.full((32, 32), 1.0), GROUND)
    col = Occupancy.from_layers(shape, [ground, canopy], strip=8)
    assert col.n_runs == 32 * 32 * 4
    assert (col.to_dense()[0, 0, [0, 2, 5, 20]] == [GROUND, 0, TREE, 0]).all()


def test_save_load(tmp_path):
    g, s, c, nz = random_city(3)
    col = Occupancy.from_heightfield(g, s, c, nz, voxel_size=0.5)
    col.save(tmp_path / "c.npz")
    d = Occupancy.load(tmp_path / "c.npz")
    assert (d.to_dense() == col.to_dense()).all() and d.voxel_size == 0.5


def test_to_octree_pads_height():
    g, s, c, nz = random_city(4, n=64, nz=45)
    col = Occupancy.from_heightfield(g, s, c, 45)
    t = col.to_octree(5)
    t.check()
    assert t.shape == (64, 64, 64)
    d = t.to_dense()
    assert (d[..., :45] == col.to_dense()).all() and (d[..., 45:] == 0).all()


def test_crop_and_repr():
    g, s, c, nz = random_city(5)
    col = Occupancy.from_heightfield(g, s, c, nz, voxel_size=2.0, origin=(100.0, 50.0, 0.0))
    sub = col.crop(10, 20, 40, 30)
    assert sub.shape[:2] == (20, 15) and sub.origin[:2] == (110.0, 70.0)
    assert (sub.to_dense() == col.to_dense()[5:25, 10:25]).all()
    assert "Occupancy" in repr(col) and "contains: 1, 2, 3" in repr(col)
    col.classes = CLASSES
    assert "building" in repr(col) and sub.classes is None


def test_to_octree_pads_any_size():
    g, s, c, nz = random_city(6, n=64, nz=40)
    col = Occupancy.from_heightfield(g[:50, :45], s[:50, :45], c[:50, :45], 40)
    t = col.to_octree(4)
    t.check()
    assert t.shape == (64, 48, 48)
    d = t.to_dense()
    assert (d[:50, :45, :40] == col.to_dense()).all()
    assert (d[50:] == 0).all() and (d[:, 45:] == 0).all() and (d[..., 40:] == 0).all()
