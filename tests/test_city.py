import numpy as np
import pytest

from octreecity import build_city

from octreecity.urban import BUILDING


def heights(n=100, seed=0):
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n))
    for _ in range(10):
        x, y = rng.integers(0, n - 20, 2)
        h[x:x + rng.integers(4, 20), y:y + rng.integers(4, 20)] = rng.uniform(3, 60)
    return h


@pytest.mark.parametrize("min_cell,max_cell", [(1, 32), (2, 16), (0.5, 8), (4, 4)])
def test_cell_size_limits(min_cell, max_cell):
    t = build_city(heights(), pixel_size=1.0, min_cell=min_cell, max_cell=max_cell)
    t.check()
    sizes = t.sizes() * t.voxel_size
    assert sizes.min() >= min_cell and sizes.max() <= max_cell
    assert all(n * t.voxel_size % max_cell == 0 for n in t.shape)
    assert (t.label == BUILDING).any()


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        build_city(heights(), 1.0, min_cell=1, max_cell=24)


def test_building_volume_roughly_kept():
    h = heights()
    for c in (1, 2, 4):
        t = build_city(h, 1.0, min_cell=c, max_cell=32)
        vol = ((t.sizes() * t.voxel_size) ** 3)[t.label == BUILDING].sum()
        assert abs(vol - h.sum()) / h.sum() < 0.15


def test_terrain():
    h = heights()
    g = np.linspace(0, 10, 100)[:, None] * np.ones((1, 100))
    t = build_city(h, 1.0, 1, 16, ground=g)
    t.check()
    assert t.origin[2] == -1.0
