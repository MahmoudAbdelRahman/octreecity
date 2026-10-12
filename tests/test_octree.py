import numpy as np
import pytest

from octreecity import morton
from octreecity.octree import (AIR, BUILDING, GROUND, LinearOctree,
                               rasterize_columns)


def random_city(seed=0, n=64, nz=64):
    rng = np.random.default_rng(seed)
    ground = np.full((n, n), 2.0)
    surface = ground.copy()
    cls = np.full((n, n), BUILDING, np.uint8)
    for _ in range(12):
        x0, y0 = rng.integers(0, n - 8, 2)
        w, d = rng.integers(3, 16, 2)
        surface[x0:x0 + w, y0:y0 + d] = 2 + rng.uniform(3, nz - 6)
        cls[x0:x0 + w, y0:y0 + d] = rng.choice([BUILDING, 3])
    return ground, surface, cls, nz


def test_morton_roundtrip():
    rng = np.random.default_rng(1)
    x, y, z = rng.integers(0, morton.MAX_COORD, (3, 1000))
    assert all((a == b).all() for a, b in zip(morton.decode(morton.encode(x, y, z)), (x, y, z)))


@pytest.mark.parametrize("seed", range(4))
def test_dense_roundtrip(seed):
    g, s, c, nz = random_city(seed)
    grid = rasterize_columns(g, s, c, nz)
    t = LinearOctree.from_dense(grid, max_level=5)
    t.check()
    assert (t.to_dense() == grid).all()


@pytest.mark.parametrize("seed", range(4))
def test_heightfield_matches_dense(seed):
    g, s, c, nz = random_city(seed)
    a = LinearOctree.from_dense(rasterize_columns(g, s, c, nz), max_level=5)
    b = LinearOctree.from_heightfield(g, s, c, nz, max_level=5)
    b.check()
    assert (a.code == b.code).all() and (a.level == b.level).all() and (a.label == b.label).all()


def test_non_cubic_box():
    g, s, c, _ = random_city(2, n=64, nz=96)
    grid = rasterize_columns(g, s, c, 96)
    t = LinearOctree.from_dense(grid, max_level=5)
    t.check()
    assert (t.to_dense() == grid).all()


def test_locate():
    g, s, c, nz = random_city(3)
    grid = rasterize_columns(g, s, c, nz)
    t = LinearOctree.from_heightfield(g, s, c, nz, max_level=5)
    rng = np.random.default_rng(0)
    p = rng.integers(0, 64, (3, 5000))
    assert (t.label_at(*p) == grid[tuple(p)]).all()


@pytest.mark.parametrize("conn", [6, 26])
def test_balance_keeps_labels_and_is_balanced(conn):
    g, s, c, nz = random_city(4)
    grid = rasterize_columns(g, s, c, nz)
    t = LinearOctree.from_dense(grid, max_level=5).balance(conn)
    t.check()
    assert (t.to_dense() == grid).all()
    t2 = LinearOctree(t.shape, t.max_level, t.code.copy(), t.level.copy(), t.label.copy())
    n = t2.n_leaves
    assert t2.balance(conn).n_leaves == n


def test_limit_size_near():
    g, s, c, nz = random_city(5)
    grid = rasterize_columns(g, s, c, nz)
    t = LinearOctree.from_dense(grid, max_level=5)
    t.limit_size_near([BUILDING, GROUND], distance=6, max_level=1)
    t.check()
    assert (t.to_dense() == grid).all()
    solid = np.isin(grid, [BUILDING, GROUND])
    from scipy import ndimage
    near = ndimage.maximum_filter(solid, size=2 * 6 + 1, mode="constant")
    x, y, z = t.corners()
    sz = t.sizes()
    for i in np.nonzero(sz > 2)[0]:
        assert not near[x[i]:x[i] + sz[i], y[i]:y[i] + sz[i], z[i]:z[i] + sz[i]].any()


def test_save_load(tmp_path):
    g, s, c, nz = random_city(6)
    t = LinearOctree.from_heightfield(g, s, c, nz, max_level=5, voxel_size=2.0)
    t.save(tmp_path / "t.npz")
    u = LinearOctree.load(tmp_path / "t.npz")
    assert u.shape == t.shape and u.voxel_size == 2.0
    assert (u.code == t.code).all() and (u.label == t.label).all()
    assert AIR in u.label


def test_compact_roundtrip_non_cubic():
    g, s, c, _ = random_city(7, n=64, nz=96)
    t = LinearOctree.from_heightfield(g, s, c, 96, max_level=5).balance(6)
    u = LinearOctree.from_compact(t.shape, t.max_level, *t.to_compact())
    assert (u.code == t.code).all()
