import numpy as np
import pytest

from octreecity import BUILDING, GROUND, TREE, LinearOctree, rasterize_columns
from octreecity.columns import ColumnCity

from test_octree import random_city


@pytest.mark.parametrize("seed", range(3))
def test_heightfield_matches_dense(seed):
    g, s, c, nz = random_city(seed)
    grid = rasterize_columns(g, s, c, nz)
    a = ColumnCity.from_heightfield(g, s, c, nz)
    b = ColumnCity.from_dense(grid)
    assert (a.to_dense() == grid).all()
    assert (a.offsets == b.offsets).all() and (a.top == b.top).all() and (a.label == b.label).all()


def test_to_octree_matches_direct():
    g, s, c, nz = random_city(1)
    col = ColumnCity.from_heightfield(g, s, c, nz)
    t1 = col.to_octree(5)
    t2 = LinearOctree.from_heightfield(g, s, c, nz, 5)
    t1.check()
    assert (t1.code == t2.code).all() and (t1.label == t2.label).all()


def test_label_at_and_top_of():
    g, s, c, nz = random_city(2)
    grid = rasterize_columns(g, s, c, nz)
    col = ColumnCity.from_dense(grid)
    p = np.random.default_rng(0).integers(0, 64, (3, 4000))
    assert (col.label_at(*p) == grid[tuple(p)]).all()
    top = col.top_of([BUILDING, TREE])
    ref = np.where(np.isin(grid, [BUILDING, TREE]), np.arange(nz) + 1, 0).max(-1)
    assert (top == ref).all()


def test_layers_tree_canopy_over_air():
    shape = (32, 32, 32)
    canopy = (np.full((32, 32), 4.0), np.full((32, 32), 10.0), TREE)
    ground = (0, np.full((32, 32), 1.0), GROUND)
    col = ColumnCity.from_layers(shape, [ground, canopy], strip=8)
    assert col.n_runs == 32 * 32 * 4
    assert (col.to_dense()[0, 0, [0, 2, 5, 20]] == [GROUND, 0, TREE, 0]).all()


def test_save_load(tmp_path):
    g, s, c, nz = random_city(3)
    col = ColumnCity.from_heightfield(g, s, c, nz, voxel_size=0.5)
    col.save(tmp_path / "c.npz")
    d = ColumnCity.load(tmp_path / "c.npz")
    assert (d.to_dense() == col.to_dense()).all() and d.voxel_size == 0.5


def test_to_octree_pads_height():
    g, s, c, nz = random_city(4, n=64, nz=45)
    col = ColumnCity.from_heightfield(g, s, c, 45)
    t = col.to_octree(5)
    t.check()
    assert t.shape == (64, 64, 64)
    d = t.to_dense()
    assert (d[..., :45] == col.to_dense()).all() and (d[..., 45:] == 0).all()
