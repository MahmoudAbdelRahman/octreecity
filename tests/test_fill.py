import numpy as np
import pytest

from octreecity.fill import carry, from_function, from_grid, from_raster, from_source
from octreecity.octree import LinearOctree
from octreecity.sources import GridSource


@pytest.fixture
def tree():
    rng = np.random.default_rng(0)
    return LinearOctree.from_dense((rng.random((32, 32, 16)) < 0.05).astype(np.uint8), 3)


@pytest.mark.parametrize("how,f", [("mean", np.nanmean), ("std", np.nanstd), ("min", np.nanmin), ("max", np.nanmax)])
def test_from_raster_matches_brute_force(tree, how, f):
    rng = np.random.default_rng(1)
    r = rng.random((32, 32))
    r[rng.random(r.shape) < 0.1] = np.nan
    x, y, _ = tree.corners()
    ref = [f(r[a:a + k, b:b + k]) if np.isfinite(r[a:a + k, b:b + k]).any() else np.nan
           for a, b, k in zip(x, y, tree.sizes())]
    assert np.allclose(from_raster(tree, r, how), ref, equal_nan=True, atol=1e-6)


def test_from_raster_coarse_pixels(tree):
    r = np.random.default_rng(2).random((16, 16))
    x, y, _ = tree.corners()
    ref = [r[a // 2:max((a + k + 1) // 2, a // 2 + 1), b // 2:max((b + k + 1) // 2, b // 2 + 1)].mean()
           for a, b, k in zip(x, y, tree.sizes())]
    assert np.allclose(from_raster(tree, r, pixel=2), ref)


def test_grid_function_source_and_carry(tree):
    G = np.random.default_rng(3).random(tree.shape)
    x, y, z = tree.corners()
    ref = [G[a:a + k, b:b + k, c:c + k].mean() for a, b, c, k in zip(x, y, z, tree.sizes())]
    assert np.allclose(from_grid(tree, G), ref)
    s = from_source(tree, GridSource(np.zeros(tree.shape, np.uint8), attrs={"g": G}))
    assert np.allclose(s.mean("g"), ref) and np.allclose(s.volume, tree.sizes() ** 3)
    f = lambda X, Y, Z: X + 2 * Y + 3 * Z
    assert np.allclose(from_function(tree, f), f(*tree.centers().T))
    assert np.allclose(from_function(tree, f, samples=3), f(*tree.centers().T))
    tree.cells["g"] = from_grid(tree, G)
    other = LinearOctree.from_dense(np.zeros(tree.shape, np.uint8), 2)
    carry(tree, other)
    vol = lambda t: t.sizes() ** 3
    assert np.isclose((tree.cells["g"] * vol(tree)).sum(), (other.cells["g"] * vol(other)).sum())
    tree.add_snapshot(0, g=tree.cells["g"])
    tree.add_snapshot(1, g=2 * tree.cells["g"])
    assert np.allclose(tree.parents().reduce(tree.series["g"])[:, 1], 2 * tree.parents().reduce(tree.cells["g"]))
